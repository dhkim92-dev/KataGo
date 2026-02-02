/**
 * @file conv2d_fp32.hlsl
 * @author dhkim92.dev@gmail.com
 * @brief Optimized batched NCHW Conv2D with Shared Memory
 * 
 * Uses shared memory tiling for both input and filter to reduce global memory traffic.
 * Bank conflict avoidance through padding.
 * 
 * Push constants: batchSize, inChannels, outChannels, nnYLen, nnXLen, filterH, filterW
 */

#include "common.h"
#include "functions.h"

struct Conv2DParams 
{
    uint batchSize;
    uint inChannels;
    uint outChannels;
    uint nnYLen;
    uint nnXLen;
    uint filterH;
    uint filterW;
};

[[vk::push_constant]]
Conv2DParams params;

// Buffers: set 0
// input: layout NCHW flattened
// filters: layout (oc, ic, fh, fw) flattened
// output: layout NCHW flattened
[[vk::binding(0, 0)]]
StructuredBuffer<float> inputBuf;
[[vk::binding(1, 0)]]
StructuredBuffer<float> filters;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> outputBuf;

// Tile configuration
// CONV_2D_TILE_N (8): number of output X positions per workgroup
// CONV_2D_TILE_M (8): number of output channels per workgroup  
// CONV_2D_TILE_K (16): block size along K dimension (ic * fh * fw)

// Shared memory with padding to avoid bank conflicts
// inputCache[k][x]: same input is used by all oc threads at same x position
// filterCache[oc][k]: different oc threads use different filter rows
groupshared float inputCache[CONV_2D_TILE_K][CONV_2D_TILE_N + 1];   // [16][9] - padding for bank conflict
groupshared float filterCache[CONV_2D_TILE_M][CONV_2D_TILE_K + 1]; // [8][17] - padding for bank conflict

[numthreads(CONV_2D_DISPATCH_X, CONV_2D_DISPATCH_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 GId : SV_GroupID)
{
    // Group mapping:
    // - GId.x: tile index in X direction (numGroupsX = ceil(nnXLen / TILE_N))
    // - GId.y: output Y coordinate (numGroupsY = nnYLen)
    // - GId.z: encodes batch and oc-block: groupZ = batch * ocGroupsPerBatch + ocGroup

    uint groupX = GId.x;
    uint outY = GId.y;
    uint groupZ = GId.z;

    uint ocGroupsPerBatch = (params.outChannels + CONV_2D_TILE_M - 1u) / CONV_2D_TILE_M;
    uint batch = groupZ / ocGroupsPerBatch;
    uint ocGroup = groupZ % ocGroupsPerBatch;

    uint localX = GTid.x; // 0..TILE_N-1 (output X offset in tile)
    uint localY = GTid.y; // 0..TILE_M-1 (output channel offset in tile)

    uint outX = groupX * CONV_2D_TILE_N + localX;
    uint oc = ocGroup * CONV_2D_TILE_M + localY;

    // Linear thread ID for cooperative loading
    uint tid = localX + localY * CONV_2D_TILE_N; // 0..63
    uint numThreads = CONV_2D_TILE_N * CONV_2D_TILE_M; // 64

    // Total K dimension = inChannels * filterH * filterW
    uint fhfw = params.filterH * params.filterW;
    uint K = params.inChannels * fhfw;

    // Precompute filter radius for padding
    uint filterHalfW = params.filterW / 2u;
    uint filterHalfH = params.filterH / 2u;

    // Accumulator
    float acc = 0.0f;

    // Number of K blocks
    uint numKBlocks = (K + CONV_2D_TILE_K - 1u) / CONV_2D_TILE_K;

    // Elements to load per tile
    uint inputEltsPerTile = CONV_2D_TILE_K * CONV_2D_TILE_N;   // 16 * 8 = 128
    uint filterEltsPerTile = CONV_2D_TILE_M * CONV_2D_TILE_K;  // 8 * 16 = 128

    [loop]
    for (uint kb = 0u; kb < numKBlocks; ++kb) {
        uint kBase = kb * CONV_2D_TILE_K;

        // ============================================
        // Cooperative load of input tile into shared memory
        // inputCache[kOff][xOff] = input at (batch, ic, inY, inX)
        // where k = kBase + kOff determines (ic, fh, fw)
        // ============================================
        for (uint i = tid; i < inputEltsPerTile; i += numThreads) {
            uint kOff = i / CONV_2D_TILE_N;
            uint xOff = i % CONV_2D_TILE_N;
            uint k = kBase + kOff;
            uint x = groupX * CONV_2D_TILE_N + xOff;

            float val = 0.0f;
            if (k < K && x < params.nnXLen && outY < params.nnYLen && batch < params.batchSize) {
                // Decode k -> ic, fh, fw
                uint ic = k / fhfw;
                uint rem = k % fhfw;
                uint fh = rem / params.filterW;
                uint fw = rem % params.filterW;

                // Compute input coordinates (filter centered)
                int inX = int(x) + int(fw) - int(filterHalfW);
                int inY = int(outY) + int(fh) - int(filterHalfH);

                // Boundary check (zero padding)
                if (inX >= 0 && inX < int(params.nnXLen) && inY >= 0 && inY < int(params.nnYLen)) {
                    uint inIdx = ((batch * params.inChannels + ic) * params.nnYLen + uint(inY)) * params.nnXLen + uint(inX);
                    val = inputBuf[inIdx];
                }
            }
            inputCache[kOff][xOff] = val;
        }

        // ============================================
        // Cooperative load of filter tile into shared memory
        // filterCache[ocOff][kOff] = filter weight at (oc, ic, fh, fw)
        // ============================================
        uint ocBase = ocGroup * CONV_2D_TILE_M;
        for (uint j = tid; j < filterEltsPerTile; j += numThreads) {
            uint ocOff = j / CONV_2D_TILE_K;
            uint kOff = j % CONV_2D_TILE_K;
            uint ocLoad = ocBase + ocOff;
            uint k = kBase + kOff;

            float val = 0.0f;
            if (ocLoad < params.outChannels && k < K) {
                // Decode k -> ic, fh, fw
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

        // ============================================
        // Compute partial dot product from shared memory
        // All threads read from shared memory - no global memory access
        // inputCache[kk][localX] is same for all oc (localY) at same localX
        // filterCache[localY][kk] is specific to each oc
        // ============================================
        [unroll]
        for (uint kk = 0u; kk < CONV_2D_TILE_K; ++kk) {
            float inVal = inputCache[kk][localX];
            float w = filterCache[localY][kk];
            acc += inVal * w;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write result to global memory
    if (outX < params.nnXLen && outY < params.nnYLen && oc < params.outChannels && batch < params.batchSize) {
        uint outIdx = ((batch * params.outChannels + oc) * params.nnYLen + outY) * params.nnXLen + outX;
        outputBuf[outIdx] = acc;
    }
}

