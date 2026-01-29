/**
* @author dhkim92.dev@gmail.com
* @brief Global Pooling Channels NCHW FP32 compute shader (with mask)
* Computes mean and max pooling over spatial dimensions with mask support.
*
* Output layout per (batch, channel):
*   output[n * cSize * 3 + c]            = mean
*   output[n * cSize * 3 + c + cSize]    = mean * (sqrt(maskSum) - 14) * 0.1
*   output[n * cSize * 3 + c + cSize*2]  = max
*
* Binding 0: input    - input tensor [N, C, H*W]
* Binding 1: output   - output tensor [N, C*3]
* Binding 2: mask     - spatial mask [N, H*W]
* Binding 3: maskSum  - sum of mask per batch [N]
*
* Dispatch: (1, gpoolChannels, batchSize)
* Each workgroup processes one (batch, channel) and reduces over spatial dimension.
*
* Compile-time defines required:
*   XYSTRIDE - power of 2 reduction stride (e.g., 64)
*/

#include "common.h"
#include "functions.h"

struct GlobalPoolingChannelsParams {
    uint batchSize;
    uint gpoolChannels;
    uint nnXYLen;
};

[[vk::push_constant]]
GlobalPoolingChannelsParams params;

[[vk::binding(0, 0)]]
StructuredBuffer<float>   g_input;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> g_output;
[[vk::binding(2, 0)]]
StructuredBuffer<float>   g_mask;
[[vk::binding(3, 0)]]
StructuredBuffer<float>   g_maskSum;

groupshared float s_partialSums[POOLING_XYSTRIDE];
groupshared float s_partialMaxes[POOLING_XYSTRIDE];

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

    // Initialize partial results
    float sum = 0.0f;
    float maxVal = -1.0f;

    if (n < params.batchSize && c < cSize) {
        // Sum and max over spatial dimension with stride
        for (uint xy = xyBase; xy < xySize; xy += POOLING_XYSTRIDE) {
            uint idx = (n * cSize + c) * xySize + xy;
            float v = g_input[idx];
            sum += v;

            // Apply mask for max computation
            // Init to -1.0 and + mask - 1.0 makes padded space into -1.0
            // which is lower than any activation function output
            uint maskIdx = n * xySize + xy;
            float maskVal = g_mask[maskIdx];
            maxVal = max(maxVal, v + (maskVal - 1.0f));
        }
    }

    // Write to shared memory for reduction
    uint localIdx = GTid.x;
    s_partialSums[localIdx] = sum;
    s_partialMaxes[localIdx] = maxVal;

    // Parallel reduction
    [unroll]
    for (uint span = POOLING_XYSTRIDE / 2; span > 0; span /= 2) {
        GroupMemoryBarrierWithGroupSync();

        if (xyBase < span) {
            s_partialSums[localIdx] += s_partialSums[localIdx + span];
            s_partialMaxes[localIdx] = max(s_partialMaxes[localIdx], s_partialMaxes[localIdx + span]);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Write final results
    if (n < params.batchSize && c < cSize && xyBase == 0) {
        float finalSum = s_partialSums[0];
        float finalMax = s_partialMaxes[0];

        float div = g_maskSum[n];
        float sqrtdiv = sqrt(div);
        float finalMean = finalSum / div;

        uint outBase = n * cSize * 3 + c;
        g_output[outBase] = finalMean;
        g_output[outBase + cSize] = finalMean * (sqrtdiv - 14.0f) * 0.1f;
        g_output[outBase + cSize * 2] = finalMax;
    }
}
