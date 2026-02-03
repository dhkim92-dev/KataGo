/**
 * @file conv2d_tiled_bn_act_5x5_fp32.hlsl
 * @author dhkim92.dev@gmail.com
 * @brief Optimized 5x5 Conv2D with fused BN->Act->Conv for KataGo.
 *        Fixed 5x5 kernel enables aggressive loop unrolling and constant folding.
 *
 * Push constants: Conv2dTiledBnActParams
 * Buffers:
 *  - inputBuf : StructuredBuffer<float> (NCHW)
 *  - filters  : StructuredBuffer<float> (oc, ic, 5, 5)
 *  - outputBuf: RWStructuredBuffer<float> (NCHW)
 *  - scale    : StructuredBuffer<float> (per-input-channel scale)
 *  - bias     : StructuredBuffer<float> (per-input-channel bias)
 *  - mask     : StructuredBuffer<float> (N, H, W)
 *
 * Optimizations:
 *  - 5x5 kernel constants enable compile-time optimizations
 *  - Full loop unrolling for 25 filter positions
 *  - Integer division/modulo replaced with multiply-subtract
 *  - Mask cache with unrolled loading
 */

#include "common.h"
#include "functions.h"

// 5x5 kernel constants - enables compiler optimizations
#define FILTER_SIZE 5
#define FILTER_HALF 2
#define FHFW 25

struct Conv2dTiledBnActParams
{
    uint batchSize;
    uint inChannels;
    uint outChannels;
    uint nnYLen;
    uint nnXLen;
    uint filterH;    // Always 5
    uint filterW;    // Always 5
    uint activation;
};

[[vk::push_constant]]
Conv2dTiledBnActParams params;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputBuf;
[[vk::binding(1, 0)]] StructuredBuffer<float> filters;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> outputBuf;
[[vk::binding(3, 0)]] StructuredBuffer<float> scale;
[[vk::binding(4, 0)]] StructuredBuffer<float> bias;
[[vk::binding(5, 0)]] StructuredBuffer<float> mask;

// Shared memory
groupshared float inputCache[CONV_BNACT_TILE_Y][CONV_BNACT_TILE_K][CONV_BNACT_TILE_X + 1];
groupshared float filterCache[CONV_BNACT_TILE_OC][CONV_BNACT_TILE_K + 1];
groupshared float scaleCache[CONV_BNACT_TILE_K];
groupshared float biasCache[CONV_BNACT_TILE_K];
groupshared float maskCache[FHFW][CONV_BNACT_TILE_Y][CONV_BNACT_TILE_X + 1];

// Inline function to apply activation
float applyActivation(float x, uint activation)
{
    if (activation == 0u) return IDENTITY(x);
    if (activation == 1u) return RELU(x);
    if (activation == 2u) return MISH(x);
    if (activation == 3u) return MISH_SCALE8(x);
    return RELU(x);
}

[numthreads(CONV_BNACT_DISPATCH_X, CONV_BNACT_DISPATCH_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 GId : SV_GroupID)
{
    uint groupX = GId.x;
    uint groupY = GId.y;
    uint groupZ = GId.z;

    uint ocGroupsPerBatch = (params.outChannels + CONV_BNACT_TILE_OC - 1u) / CONV_BNACT_TILE_OC;
    uint batch = groupZ / ocGroupsPerBatch;
    uint ocGroup = groupZ % ocGroupsPerBatch;

    uint localX = GTid.x;
    uint localOC = GTid.y;

    uint outXBase = groupX * CONV_BNACT_TILE_X;
    uint outYBase = groupY * CONV_BNACT_TILE_Y;
    uint oc = ocGroup * CONV_BNACT_TILE_OC + localOC;

    uint tid = localX + localOC * CONV_BNACT_TILE_X;
    uint numThreads = CONV_BNACT_TILE_X * CONV_BNACT_TILE_OC;

    uint K = params.inChannels * FHFW;
    uint nnXYLen = params.nnXLen * params.nnYLen;

    // Accumulators
    float acc[CONV_BNACT_TILE_Y];
    [unroll] for (uint yy = 0u; yy < CONV_BNACT_TILE_Y; ++yy) {
        acc[yy] = 0.0f;
    }

    uint numKBlocks = (K + CONV_BNACT_TILE_K - 1u) / CONV_BNACT_TILE_K;
    uint inputEltsPerTile = CONV_BNACT_TILE_Y * CONV_BNACT_TILE_K * CONV_BNACT_TILE_X;
    uint filterEltsPerTile = CONV_BNACT_TILE_OC * CONV_BNACT_TILE_K;

    // ============================================
    // Load mask cache - UNROLLED for 25 positions (5x5)
    // Each position (fh, fw) loaded in parallel by threads
    // ============================================
    uint maskEltsPerPos = CONV_BNACT_TILE_Y * CONV_BNACT_TILE_X;

    // Unrolled mask loading for all 25 filter positions
    #define LOAD_MASK_POS(FH, FW) \
    { \
        uint pos = (FH) * FILTER_SIZE + (FW); \
        for (uint i = tid; i < maskEltsPerPos; i += numThreads) { \
            uint yOff = i / CONV_BNACT_TILE_X; \
            uint xOff = i % CONV_BNACT_TILE_X; \
            uint x = outXBase + xOff; \
            uint outY = outYBase + yOff; \
            float m = 0.0f; \
            if (x < params.nnXLen && outY < params.nnYLen && batch < params.batchSize) { \
                int inX = int(x) + (FW) - FILTER_HALF; \
                int inY = int(outY) + (FH) - FILTER_HALF; \
                if (inX >= 0 && inX < int(params.nnXLen) && inY >= 0 && inY < int(params.nnYLen)) { \
                    uint maskIdx = batch * nnXYLen + uint(inY) * params.nnXLen + uint(inX); \
                    m = mask[maskIdx]; \
                } \
            } \
            maskCache[pos][yOff][xOff] = m; \
        } \
    }

    // Row 0
    LOAD_MASK_POS(0, 0)
    LOAD_MASK_POS(0, 1)
    LOAD_MASK_POS(0, 2)
    LOAD_MASK_POS(0, 3)
    LOAD_MASK_POS(0, 4)
    // Row 1
    LOAD_MASK_POS(1, 0)
    LOAD_MASK_POS(1, 1)
    LOAD_MASK_POS(1, 2)
    LOAD_MASK_POS(1, 3)
    LOAD_MASK_POS(1, 4)
    // Row 2
    LOAD_MASK_POS(2, 0)
    LOAD_MASK_POS(2, 1)
    LOAD_MASK_POS(2, 2)
    LOAD_MASK_POS(2, 3)
    LOAD_MASK_POS(2, 4)
    // Row 3
    LOAD_MASK_POS(3, 0)
    LOAD_MASK_POS(3, 1)
    LOAD_MASK_POS(3, 2)
    LOAD_MASK_POS(3, 3)
    LOAD_MASK_POS(3, 4)
    // Row 4
    LOAD_MASK_POS(4, 0)
    LOAD_MASK_POS(4, 1)
    LOAD_MASK_POS(4, 2)
    LOAD_MASK_POS(4, 3)
    LOAD_MASK_POS(4, 4)

    #undef LOAD_MASK_POS

    GroupMemoryBarrierWithGroupSync();

    // ============================================
    // Main K-block loop
    // ============================================
    [loop]
    for (uint kb = 0u; kb < numKBlocks; ++kb) {
        uint kBase = kb * CONV_BNACT_TILE_K;

        // Load scale/bias for this K block
        for (uint i = tid; i < CONV_BNACT_TILE_K; i += numThreads) {
            uint k = kBase + i;
            if (k < K) {
                uint ic = k / FHFW;
                scaleCache[i] = scale[ic];
                biasCache[i] = bias[ic];
            } else {
                scaleCache[i] = 0.0f;
                biasCache[i] = 0.0f;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Load input tile with BN + Activation + Mask
        for (uint i = tid; i < inputEltsPerTile; i += numThreads) {
            uint yOff = i / (CONV_BNACT_TILE_K * CONV_BNACT_TILE_X);
            uint rem1 = i % (CONV_BNACT_TILE_K * CONV_BNACT_TILE_X);
            uint kOff = rem1 / CONV_BNACT_TILE_X;
            uint xOff = rem1 % CONV_BNACT_TILE_X;

            uint k = kBase + kOff;
            uint x = outXBase + xOff;
            uint outY = outYBase + yOff;

            float val = 0.0f;
            if (k < K && x < params.nnXLen && outY < params.nnYLen && batch < params.batchSize) {
                // Decode k -> (ic, fh, fw) for 5x5: k = ic * 25 + fh * 5 + fw
                uint ic = k / FHFW;
                uint rem = k - ic * FHFW;  // Faster than k % 25
                uint fh = rem / FILTER_SIZE;
                uint fw = rem - fh * FILTER_SIZE;  // Faster than rem % 5

                int inX = int(x) + int(fw) - FILTER_HALF;
                int inY = int(outY) + int(fh) - FILTER_HALF;

                if (inX >= 0 && inX < int(params.nnXLen) && inY >= 0 && inY < int(params.nnYLen)) {
                    uint inIdx = ((batch * params.inChannels + ic) * params.nnYLen + uint(inY)) * params.nnXLen + uint(inX);

                    float rawInput = inputBuf[inIdx];
                    float s = scaleCache[kOff];
                    float b = biasCache[kOff];
                    float bnOut = rawInput * s + b;

                    float actOut = applyActivation(bnOut, params.activation);

                    // Use cached mask (rem = fh * 5 + fw = filter position)
                    float m = maskCache[rem][yOff][xOff];
                    val = actOut * m;
                }
            }
            inputCache[yOff][kOff][xOff] = val;
        }

        // Load filter tile
        uint ocBase = ocGroup * CONV_BNACT_TILE_OC;
        for (uint j = tid; j < filterEltsPerTile; j += numThreads) {
            uint ocOff = j / CONV_BNACT_TILE_K;
            uint kOff = j % CONV_BNACT_TILE_K;
            uint ocLoad = ocBase + ocOff;
            uint k = kBase + kOff;

            float val = 0.0f;
            if (ocLoad < params.outChannels && k < K) {
                uint ic = k / FHFW;
                uint rem = k - ic * FHFW;

                // Filter layout: (oc, ic, 5, 5)
                uint filtIdx = ((ocLoad * params.inChannels + ic) * FHFW) + rem;
                val = filters[filtIdx];
            }
            filterCache[ocOff][kOff] = val;
        }

        GroupMemoryBarrierWithGroupSync();

        // Compute dot product - fully unrolled
        [unroll]
        for (uint kk = 0u; kk < CONV_BNACT_TILE_K; ++kk) {
            float w = filterCache[localOC][kk];
            [unroll]
            for (uint yy = 0u; yy < CONV_BNACT_TILE_Y; ++yy) {
                acc[yy] += inputCache[yy][kk][localX] * w;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write results
    uint outX = outXBase + localX;
    if (outX < params.nnXLen && oc < params.outChannels && batch < params.batchSize) {
        [unroll]
        for (uint yy = 0u; yy < CONV_BNACT_TILE_Y; ++yy) {
            uint outY = outYBase + yy;
            if (outY < params.nnYLen) {
                uint outIdx = ((batch * params.outChannels + oc) * params.nnYLen + outY) * params.nnXLen + outX;
                outputBuf[outIdx] = acc[yy];
            }
        }
    }
}
