/**
 * @file conv2d_tiled_bn_act_fp32.hlsl
 * @author dhkim92.dev@gmail.com
 * @brief Tiled NCHW Conv2D with pre-applied BatchNorm + Activation (fused BN->Act->Conv).
 *        This implements the same pattern as OpenCL's winogradBNActTransformNCHW:
 *        BN and activation are applied to the INPUT before convolution.
 *
 * Push constants: Conv2dTiledBnActParams
 * Buffers:
 *  - inputBuf : StructuredBuffer<float> (NCHW)
 *  - filters  : StructuredBuffer<float> (oc, ic, fh, fw)
 *  - outputBuf: RWStructuredBuffer<float> (NCHW)
 *  - scale    : StructuredBuffer<float> (per-input-channel scale, length = inChannels)
 *  - bias     : StructuredBuffer<float> (per-input-channel bias, length = inChannels)
 *  - mask     : StructuredBuffer<float> (N, H, W)
 *
 * Notes:
 *  - BN + Act + mask are applied to INPUT, not output
 *  - This matches the NormActConv pattern: BN -> Act -> Conv
 *  - Bank padding applied to shared memory to avoid conflicts
 */

#include "common.h"
#include "functions.h"

struct Conv2dTiledBnActParams
{
    uint batchSize;
    uint inChannels;
    uint outChannels;
    uint nnYLen;
    uint nnXLen;
    uint filterH;
    uint filterW;
    uint activation;
};

[[vk::push_constant]]
Conv2dTiledBnActParams params;

// Buffers: set 0
[[vk::binding(0, 0)]]
StructuredBuffer<float> inputBuf; // NCHW flattened
[[vk::binding(1, 0)]]
StructuredBuffer<float> filters;  // (oc, ic, fh, fw) flattened
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> outputBuf; // NCHW flattened
[[vk::binding(3, 0)]]
StructuredBuffer<float> scale;    // length = inChannels (BN scale for input)
[[vk::binding(4, 0)]]
StructuredBuffer<float> bias;     // length = inChannels (BN bias for input)
[[vk::binding(5, 0)]]
StructuredBuffer<float> mask;     // (N, H, W) flattened

// Shared memory with padding to avoid bank conflicts
groupshared float inputCache[CONV_BNACT_TILE_K][CONV_BNACT_TILE_N + 1];   // K x (N+1)
groupshared float filterCache[CONV_BNACT_TILE_M][CONV_BNACT_TILE_K + 1]; // M x (K+1)

[numthreads(CONV_BNACT_DISPATCH_X, CONV_BNACT_DISPATCH_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 GId : SV_GroupID)
{
    uint groupX = GId.x;               // tile index in X direction
    uint outY = GId.y;                 // output Y coordinate
    uint groupZ = GId.z;               // encodes batch and oc-block

    uint ocGroupsPerBatch = (params.outChannels + CONV_BNACT_TILE_M - 1u) / CONV_BNACT_TILE_M;
    uint batch = groupZ / ocGroupsPerBatch;
    uint ocGroup = groupZ % ocGroupsPerBatch;

    uint localX = GTid.x; // 0..TILE_N-1 (output X offset in tile)
    uint localY = GTid.y; // 0..TILE_M-1 (output channel offset in tile)

    uint outX = groupX * CONV_BNACT_TILE_N + localX;
    uint oc = ocGroup * CONV_BNACT_TILE_M + localY;

    uint tid = localX + localY * CONV_BNACT_TILE_N;
    uint numThreads = CONV_BNACT_TILE_N * CONV_BNACT_TILE_M;

    uint fhfw = params.filterH * params.filterW;
    uint K = params.inChannels * fhfw;

    uint filterHalfW = params.filterW / 2u;
    uint filterHalfH = params.filterH / 2u;

    uint nnXYLen = params.nnXLen * params.nnYLen;

    float acc = 0.0f;

    uint numKBlocks = (K + CONV_BNACT_TILE_K - 1u) / CONV_BNACT_TILE_K;

    uint inputEltsPerTile = CONV_BNACT_TILE_K * CONV_BNACT_TILE_N;
    uint filterEltsPerTile = CONV_BNACT_TILE_M * CONV_BNACT_TILE_K;

    for (uint kb = 0u; kb < numKBlocks; ++kb) {
        uint kBase = kb * CONV_BNACT_TILE_K;

        // Cooperative load input tile with BN + Activation + Mask applied
        for (uint i = tid; i < inputEltsPerTile; i += numThreads) {
            uint kOff = i / CONV_BNACT_TILE_N;
            uint xOff = i % CONV_BNACT_TILE_N;
            uint k = kBase + kOff;
            uint x = groupX * CONV_BNACT_TILE_N + xOff;

            float val = 0.0f;
            if (k < K && x < params.nnXLen && outY < params.nnYLen && batch < params.batchSize) {
                uint ic = k / fhfw;
                uint rem = k % fhfw;
                uint fh = rem / params.filterW;
                uint fw = rem % params.filterW;

                int inX = int(x) + int(fw) - int(filterHalfW);
                int inY = int(outY) + int(fh) - int(filterHalfH);

                if (inX >= 0 && inX < int(params.nnXLen) && inY >= 0 && inY < int(params.nnYLen)) {
                    uint inIdx = ((batch * params.inChannels + ic) * params.nnYLen + uint(inY)) * params.nnXLen + uint(inX);
                    uint maskIdx = batch * nnXYLen + uint(inY) * params.nnXLen + uint(inX);

                    // Load input and apply BN: (input * scale + bias)
                    float rawInput = inputBuf[inIdx];
                    float s = scale[ic];
                    float b = bias[ic];
                    float bnOut = rawInput * s + b;

                    // Apply activation
                    float actOut;
                    if (params.activation == 0u)
                        actOut = IDENTITY(bnOut);
                    else if (params.activation == 1u)
                        actOut = RELU(bnOut);
                    else if (params.activation == 2u)
                        actOut = MISH(bnOut);
                    else if (params.activation == 3u)
                        actOut = MISH_SCALE8(bnOut);
                    else
                        actOut = RELU(bnOut); // Default to ReLU

                    // Apply mask
                    float m = mask[maskIdx];
                    val = actOut * m;
                }
            }
            inputCache[kOff][xOff] = val;
        }

        // Cooperative load filter tile
        uint ocBase = ocGroup * CONV_BNACT_TILE_M;
        for (uint j = tid; j < filterEltsPerTile; j += numThreads) {
            uint ocOff = j / CONV_BNACT_TILE_K;
            uint kOff = j % CONV_BNACT_TILE_K;
            uint ocLoad = ocBase + ocOff;
            uint k = kBase + kOff;

            float val = 0.0f;
            if (ocLoad < params.outChannels && k < K) {
                uint ic = k / fhfw;
                uint rem = k % fhfw;
                uint fh = rem / params.filterW;
                uint fw = rem % params.filterW;

                uint filtIdx = (((ocLoad * params.inChannels + ic) * params.filterH + fh) * params.filterW) + fw;
                val = filters[filtIdx];
            }
            filterCache[ocOff][kOff] = val;
        }

        GroupMemoryBarrierWithGroupSync();

        // Compute partial dot product
        [unroll]
        for (uint kk = 0u; kk < CONV_BNACT_TILE_K; ++kk) {
            float inVal = inputCache[kk][localX];
            float w = filterCache[localY][kk];
            acc += inVal * w;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write result directly (BN+Act+Mask already applied to input)
    if (outX < params.nnXLen && outY < params.nnYLen && oc < params.outChannels && batch < params.batchSize) {
        uint outIdx = ((batch * params.outChannels + oc) * params.nnYLen + outY) * params.nnXLen + outX;
        outputBuf[outIdx] = acc;
    }
}
