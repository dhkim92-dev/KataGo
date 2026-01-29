// conv2d_fp32.hlsl
// Basic batched NCHW Conv2D implemented in HLSL for Vulkan (compiled to SPIR-V).
// Uses a tiled/blocked matrix-multiplication style (blocking over K dimension).
// Push constants must follow Conv2dPushConstantParams in vulkanbackend.h:
//   uint batchSize, inChannels, outChannels, nnYLen, nnXLen, filterH, filterW

#include "common.h"
#include "functions.h"

struct Covn2DParams 
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
Covn2DParams params;

// Buffers: bindings chosen generically. Host should bind accordingly.
// filters: layout (oc, ic, fh, fw) flattened
// input: layout NCHW flattened
// output: layout NCHW flattened

[[vk::binding(0, 0)]]
StructuredBuffer<float> inputBuf;
[[vk::binding(1, 0)]]
StructuredBuffer<float> filters;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> outputBuf;

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

    uint ocGroupsPerBatch = (params.outChannels + CONV_2D_TILE_M - 1u) / CONV_2D_TILE_M;
    uint batch = groupZ / ocGroupsPerBatch;
    uint ocGroup = groupZ % ocGroupsPerBatch;

    uint localX = GTid.x; // 0..TILE_N-1
    uint localY = GTid.y; // 0..TILE_M-1

    uint outX = groupX * CONV_2D_TILE_N + localX;
    if(outX >= params.nnXLen || outY >= params.nnYLen || batch >= params.batchSize) {
        // out of bounds; threads can early exit
        return;
    }

    uint oc = ocGroup * CONV_2D_TILE_M + localY; // output channel for this thread
    if(oc >= params.outChannels) {
        // this thread corresponds to out-of-range output channel
        return;
    }

    // We'll compute output value for (batch, oc, outY, outX)
    // Accumulate over K = inChannels * filterH * filterW
    uint K = params.inChannels * params.filterH * params.filterW;

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
            uint fhfw = params.filterH * params.filterW;
            uint ic = k / fhfw;
            uint rem = k % fhfw;
            uint fh = rem / params.filterW;
            uint fw = rem % params.filterW;

            // compute input coords corresponding to filter (centered)
            int inX = int(outX) + int(fw) - int(params.filterW/2u);
            int inY = int(outY) + int(fh) - int(params.filterH/2u);

            float inVal = 0.0f;
            if(inX >= 0 && inX < int(params.nnXLen) && inY >= 0 && inY < int(params.nnYLen)) {
                // input index for NCHW: ((n*inChannels + ic) * nnY + inY) * nnX + inX
                uint inIndex = ((batch * params.inChannels + ic) * params.nnYLen + uint(inY)) * params.nnXLen + uint(inX);
                inVal = inputBuf[inIndex];
            }

            // filter index: (((oc * inChannels + ic) * filterH + fh) * filterW + fw)
            uint filtIndex = (((oc * params.inChannels + ic) * params.filterH + fh) * params.filterW) + fw;
            float w = filters[filtIndex];

            acc += w * inVal;
        }
    }

    // write output
    uint outIndex = ((batch * params.outChannels + oc) * params.nnYLen + outY) * params.nnXLen + outX;
    outputBuf[outIndex] = acc;
}
