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

// Tile configuration (from common.h)
// CONV_2D_TILE_X: number of output X positions per workgroup
// CONV_2D_TILE_Y: number of output Y positions per workgroup (tiled in Y direction)
// CONV_2D_TILE_OC: number of output channels per workgroup  
// CONV_2D_TILE_K: block size along K dimension (ic * fh * fw)

// Shared memory with padding to avoid bank conflicts
// inputCache[y][k][x]: input data for Y tile positions
// filterCache[oc][k]: filter weights
groupshared float inputCache[CONV_2D_TILE_Y][CONV_2D_TILE_K][CONV_2D_TILE_X + 1];  // +1 for bank conflict avoidance
groupshared float filterCache[CONV_2D_TILE_OC][CONV_2D_TILE_K + 1];               // +1 for bank conflict avoidance

[numthreads(CONV_2D_DISPATCH_X, CONV_2D_DISPATCH_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 GId : SV_GroupID)
{
    // Group mapping:
    // - GId.x: tile index in X direction (numGroupsX = ceil(nnXLen / TILE_X))
    // - GId.y: tile index in Y direction (numGroupsY = ceil(nnYLen / TILE_Y))
    // - GId.z: encodes batch and oc-block: groupZ = batch * ocGroupsPerBatch + ocGroup

    uint groupX = GId.x;
    uint groupY = GId.y;
    uint groupZ = GId.z;

    uint ocGroupsPerBatch = (params.outChannels + CONV_2D_TILE_OC - 1u) / CONV_2D_TILE_OC;
    uint batch = groupZ / ocGroupsPerBatch;
    uint ocGroup = groupZ % ocGroupsPerBatch;

    uint localX = GTid.x; // 0..TILE_X-1 (thread X index, maps to output X offset)
    uint localOC = GTid.y; // 0..TILE_OC-1 (thread Y index, maps to output channel offset)

    uint outXBase = groupX * CONV_2D_TILE_X;
    uint outYBase = groupY * CONV_2D_TILE_Y;
    uint oc = ocGroup * CONV_2D_TILE_OC + localOC;

    // Linear thread ID for cooperative loading
    uint tid = localX + localOC * CONV_2D_TILE_X; // 0..(TILE_X * TILE_OC - 1)
    uint numThreads = CONV_2D_TILE_X * CONV_2D_TILE_OC; // 16 * 8 = 128

    // Total K dimension = inChannels * filterH * filterW
    uint fhfw = params.filterH * params.filterW;
    uint K = params.inChannels * fhfw;

    // Precompute filter radius for padding
    uint filterHalfW = params.filterW / 2u;
    uint filterHalfH = params.filterH / 2u;

    // Accumulators for TILE_Y output Y positions
    float acc[CONV_2D_TILE_Y];
    [unroll]
    for (uint yy = 0; yy < CONV_2D_TILE_Y; ++yy) {
        acc[yy] = 0.0f;
    }

    // Number of K blocks
    uint numKBlocks = (K + CONV_2D_TILE_K - 1u) / CONV_2D_TILE_K;

    // Elements to load per tile
    uint inputEltsPerTile = CONV_2D_TILE_Y * CONV_2D_TILE_K * CONV_2D_TILE_X;  // Y * K * X
    uint filterEltsPerTile = CONV_2D_TILE_OC * CONV_2D_TILE_K;                  // OC * K

    [loop]
    for (uint kb = 0u; kb < numKBlocks; ++kb) {
        uint kBase = kb * CONV_2D_TILE_K;

        // ============================================
        // Cooperative load of input tile into shared memory
        // inputCache[yOff][kOff][xOff] = input at (batch, ic, inY, inX)
        // where k = kBase + kOff determines (ic, fh, fw)
        // ============================================
        for (uint i = tid; i < inputEltsPerTile; i += numThreads) {
            uint yOff = i / (CONV_2D_TILE_K * CONV_2D_TILE_X);
            uint rem1 = i % (CONV_2D_TILE_K * CONV_2D_TILE_X);
            uint kOff = rem1 / CONV_2D_TILE_X;
            uint xOff = rem1 % CONV_2D_TILE_X;
            
            uint k = kBase + kOff;
            uint x = outXBase + xOff;
            uint outY = outYBase + yOff;

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
            inputCache[yOff][kOff][xOff] = val;
        }

        // ============================================
        // Cooperative load of filter tile into shared memory
        // filterCache[ocOff][kOff] = filter weight at (oc, ic, fh, fw)
        // ============================================
        uint ocBase = ocGroup * CONV_2D_TILE_OC;
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
        // Each thread computes TILE_Y output positions
        // inputCache[yy][kk][localX] varies by Y position
        // filterCache[localOC][kk] is specific to each output channel
        // ============================================
        [unroll]
        for (uint kk = 0u; kk < CONV_2D_TILE_K; ++kk) {
            float w = filterCache[localOC][kk];
            [unroll]
            for (uint yy = 0u; yy < CONV_2D_TILE_Y; ++yy) {
                float inVal = inputCache[yy][kk][localX];
                acc[yy] += inVal * w;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write results to global memory for all TILE_Y positions
    uint outX = outXBase + localX;
    if (outX < params.nnXLen && oc < params.outChannels && batch < params.batchSize) {
        [unroll]
        for (uint yy = 0u; yy < CONV_2D_TILE_Y; ++yy) {
            uint outY = outYBase + yy;
            if (outY < params.nnYLen) {
                uint outIdx = ((batch * params.outChannels + oc) * params.nnYLen + outY) * params.nnXLen + outX;
                outputBuf[outIdx] = acc[yy];
            }
        }
    }
}

