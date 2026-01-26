// conv2d_fp32.hlsl
// Basic batched NCHW Conv2D implemented in HLSL for Vulkan (compiled to SPIR-V).
// Uses a tiled/blocked matrix-multiplication style (blocking over K dimension).
// Push constants must follow Conv2dPushConstantParams in vulkanbackend.h:
//   uint batchSize, inChannels, outChannels, nnYLen, nnXLen, filterH, filterW

#include "common.h"
#include "functions.h"

cbuffer Covn2DParams : register(b0)
{
    uint batchSize;
    uint inChannels;
    uint outChannels;
    uint nnYLen;
    uint nnXLen;
    uint filterH;
    uint filterW;
};

// Buffers: bindings chosen generically. Host should bind accordingly.
// filters: layout (oc, ic, fh, fw) flattened
// input: layout NCHW flattened
// output: layout NCHW flattened

StructuredBuffer<float> filters : register(t0);
StructuredBuffer<float> inputBuf : register(t1);
RWStructuredBuffer<float> outputBuf : register(u0);

// Tile configuration (tunable)
// TILE_N: number of output pixels computed per workgroup in X-direction
// TILE_M: number of output channels computed per workgroup
// TILE_K: block size along K (ic * filterH * filterW)
// Workgroup layout: threads (localX, localY)
// localX in [0,TILE_N) -> pixel offset inside tile
// localY in [0,TILE_M) -> output-channel offset inside oc-block
[numthreads(CONV_2D_DISPATCH_X, CONV_2D_DISPATCH_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 GId : SV_GroupID)
{
    // Group mapping assumptions made by host dispatcher:
    // - Groups X cover output X in tiles of TILE_N (numGroupsX = ceil(nnXLen / TILE_N))
    // - Groups Y correspond to output Y coordinate (numGroupsY = nnYLen)
    // - Groups Z encode batch and oc-block: ocGroupsPerBatch = ceil(outChannels / TILE_M)
    //   numGroupsZ = batchSize * ocGroupsPerBatch

    uint groupX = GId.x; // tile index in X
    uint outY = GId.y;   // exact output Y coordinate
    uint groupZ = GId.z; // encodes batch and oc-block

    uint ocGroupsPerBatch = (outChannels + CONV_2D_TILE_M - 1u) / CONV_2D_TILE_M;
    uint batch = groupZ / ocGroupsPerBatch;
    uint ocGroup = groupZ % ocGroupsPerBatch;

    uint localX = GTid.x; // 0..TILE_N-1
    uint localY = GTid.y; // 0..TILE_M-1

    uint outX = groupX * CONV_2D_TILE_N + localX;
    if(outX >= nnXLen || outY >= nnYLen || batch >= batchSize) {
        // out of bounds; threads can early exit
        return;
    }

    uint oc = ocGroup * CONV_2D_TILE_M + localY; // output channel for this thread
    if(oc >= outChannels) {
        // this thread corresponds to out-of-range output channel
        return;
    }

    // We'll compute output value for (batch, oc, outY, outX)
    // Accumulate over K = inChannels * filterH * filterW
    uint K = inChannels * filterH * filterW;

    float acc = 0.0f;

    // Blocked over K
    uint numKBlocks = (K + CONV_2D_TILE_K - 1u) / CONV_2D_TILE_K;
    for(uint kb = 0u; kb < numKBlocks; ++kb) {
        uint kBase = kb * CONV_2D_TILE_K;
        uint kMax = min(CONV_2D_TILE_K, K - kBase);

        // For this small-block, iterate directly (no shared memory for simplicity)
        for(uint kOff = 0u; kOff < kMax; ++kOff) {
            uint k = kBase + kOff; // linear index in K
            // decode k -> ic, fh, fw
            uint fhfw = filterH * filterW;
            uint ic = k / fhfw;
            uint rem = k % fhfw;
            uint fh = rem / filterW;
            uint fw = rem % filterW;

            // compute input coords corresponding to filter (centered)
            int inX = int(outX) + int(fw) - int(filterW/2u);
            int inY = int(outY) + int(fh) - int(filterH/2u);

            float inVal = 0.0f;
            if(inX >= 0 && inX < int(nnXLen) && inY >= 0 && inY < int(nnYLen)) {
                // input index for NCHW: ((n*inChannels + ic) * nnY + inY) * nnX + inX
                uint inIndex = ((batch * inChannels + ic) * nnYLen + uint(inY)) * nnXLen + uint(inX);
                inVal = inputBuf[inIndex];
            }

            // filter index: (((oc * inChannels + ic) * filterH + fh) * filterW + fw)
            uint filtIndex = (((oc * inChannels + ic) * filterH + fh) * filterW) + fw;
            float w = filters[filtIndex];

            acc += w * inVal;
        }
    }

    // write output
    uint outIndex = ((batch * outChannels + oc) * nnYLen + outY) * nnXLen + outX;
    outputBuf[outIndex] = acc;
}
