/**
* @author dhkim92.dev@gmail.com
* @brief Value Head Pooling Channels NCHW FP32 compute shader
* Computes mean pooling over spatial dimensions for value head.
*
* Output layout per (batch, channel):
*   output[n * cSize * 3 + c]            = mean
*   output[n * cSize * 3 + c + cSize]    = mean * (sqrt(maskSum) - 14) * 0.1
*   output[n * cSize * 3 + c + cSize*2]  = mean * ((sqrt(maskSum) - 14)^2 * 0.01 - 0.1)
*
* Binding 0: input    - input tensor [N, C, H*W]
* Binding 1: output   - output tensor [N, C*3]
* Binding 2: maskSum  - sum of mask per batch [N]
*
* Dispatch: (1, gpoolChannels, batchSize)
* Each workgroup processes one (batch, channel) and reduces over spatial dimension.
*
* Compile-time defines required:
*   XYSTRIDE - power of 2 reduction stride (e.g., 64)
*/

#include "common.h"
#include "functions.h"

struct ValueHeadPoolingChannelsParams {
    uint batchSize;
    uint gpoolChannels;
    uint nnXYLen;
};

[[vk::push_constant]]
ValueHeadPoolingChannelsParams params;

StructuredBuffer<float>   g_input   : register(t0);
RWStructuredBuffer<float> g_output  : register(u0);
StructuredBuffer<float>   g_maskSum : register(t1);

groupshared float s_partialSums[POOLING_XYSTRIDE];

[numthreads(POOLING_DISPATCH_X, 1, 1)]
void main(
    uint3 GTid : SV_GroupThreadID,
    uint3 Gid  : SV_GroupID
)
{
    uint xyBase = GTid.x;
    uint c = Gid.y;
    uint n = Gid.z;

    uint cSize = params.gpoolChannels;
    uint xySize = params.nnXYLen;

    // Initialize partial sum
    float sum = 0.0f;

    if (n < params.batchSize && c < cSize) {
        // Sum over spatial dimension with stride
        for (uint xy = xyBase; xy < xySize; xy += POOLING_XYSTRIDE) {
            uint idx = (n * cSize + c) * xySize + xy;
            float v = g_input[idx];
            sum += v;
        }
    }

    // Write to shared memory for reduction
    uint localIdx = GTid.x;
    s_partialSums[localIdx] = sum;

    // Parallel reduction
    [unroll]
    for (uint span = POOLING_XYSTRIDE / 2; span > 0; span /= 2) {
        GroupMemoryBarrierWithGroupSync();

        if (xyBase < span) {
            s_partialSums[localIdx] += s_partialSums[localIdx + span];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Write final results
    if (n < params.batchSize && c < cSize && xyBase == 0) {
        float finalSum = s_partialSums[0];

        float div = g_maskSum[n];
        float sqrtdiv = sqrt(div);
        float finalMean = finalSum / div;

        // Precompute (sqrtdiv - 14.0)
        float sqrtdivM14 = sqrtdiv - 14.0f;

        uint outBase = n * cSize * 3 + c;
        g_output[outBase] = finalMean;
        g_output[outBase + cSize] = finalMean * sqrtdivM14 * 0.1f;
        g_output[outBase + cSize * 2] = finalMean * (sqrtdivM14 * sqrtdivM14 * 0.01f - 0.1f);
    }
}
