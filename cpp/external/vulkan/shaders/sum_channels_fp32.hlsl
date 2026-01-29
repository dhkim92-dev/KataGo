/**
* @author dhkim92.dev@gmail.com
* @brief Sum Channels NCHW FP32 compute shader
* Sums all spatial elements for each (batch, channel) pair
* Input: N, C, HW (NCHW format)
* Output: N, C (sum over spatial dimensions)
*/

#include "common.h"
#include "functions.h"

// Workgroup configuration
// SUM_CHANNELS_XYSTRIDE: number of threads for parallel reduction over spatial dimension
// Must match local_size_x
struct SumChannelsParams {
    uint batchSize;
    uint numChannels;
    uint nnXYLen;
};

[[vk::push_constant]]
SumChannelsParams params;

[[vk::binding(0, 0)]]
StructuredBuffer<float> g_input;   // N, C, HW
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> g_output; // N, C

// Shared memory for parallel reduction
groupshared float partialSums[SUM_CHANNELS_XYSTRIDE];

[numthreads(SUM_CHANNELS_DISPATCH_X, 1, 1)]
void main(
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint3 groupId : SV_GroupID
)
{
    uint xyBase = groupThreadId.x;  // Local thread index for reduction
    uint c = groupId.y;             // Channel index
    uint n = groupId.z;             // Batch index

    uint nSize = params.batchSize;
    uint cSize = params.numChannels;
    uint xySize = params.nnXYLen;

    float sum = 0.0f;
    
    if (n < nSize && c < cSize) {
        // Sum up the elements that this thread is responsible for
        // Each thread handles elements spaced XYSTRIDE apart
        for (uint xy = xyBase; xy < xySize; xy += SUM_CHANNELS_XYSTRIDE) {
            uint idx = (n * cSize + c) * xySize + xy;
            sum += g_input[idx];
        }
    }

    // Write to shared memory for performing the reduction
    partialSums[xyBase] = sum;

    // Parallel folding downward
    [unroll]
    for (uint span = SUM_CHANNELS_XYSTRIDE / 2; span > 0; span /= 2) {
        GroupMemoryBarrierWithGroupSync();

        if (xyBase < span) {
            partialSums[xyBase] += partialSums[xyBase + span];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Write final result
    if (n < nSize && c < cSize && xyBase == 0) {
        float finalSum = partialSums[0];
        uint outIdx = n * cSize + c;
        g_output[outIdx] = finalSum;
    }
}
