/**
* @author dhkim92.dev@gmail.com
* @brief Add Channel Bias NC FP32 compute shader (Identity activation)
* Adds bias to each channel: accum[n,c] += bias[c]
* Matches OpenCL addCBiasesNCAct kernel behavior
* Uses 1D dispatch for better GPU utilization with small batch sizes
*
* Binding 0: accum (RW) - input/output tensor [N, C]
* Binding 1: bias       - bias vector [C]
*/
#include "common.h"
#include "functions.h"

struct AddChannelBiasNCParams {
    uint nSize;
    uint cSize;
};

[[vk::push_constant]]
AddChannelBiasNCParams params;

[[vk::binding(0, 0)]]
RWStructuredBuffer<float> g_accum;
[[vk::binding(1, 0)]]
StructuredBuffer<float>   g_bias;

[numthreads(ADD_CHANNEL_BIAS_NC_DISPATCH_X, ADD_CHANNEL_BIAS_NC_DISPATCH_Y, ADD_CHANNEL_BIAS_NC_DISPATCH_Z)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint totalSize = params.nSize * params.cSize;
    uint idx = DTid.x;
    if (idx < totalSize) {
        uint c = idx % params.cSize;  // channel index
        g_accum[idx] = IDENTITY(g_accum[idx] + g_bias[c]);
    }
}
