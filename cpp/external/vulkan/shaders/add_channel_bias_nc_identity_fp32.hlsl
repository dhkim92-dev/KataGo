/**
* @author dhkim92.dev@gmail.com
* @brief Add Channel Bias NC FP32 compute shader (Identity activation)
* Adds bias to each channel: accum[n,c] += bias[c]
* Matches OpenCL addCBiasesNCAct kernel behavior
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

[numthreads(ADD_CHANNELS_DISPATCH_X, ADD_CHANNELS_DISPATCH_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint n = DTid.x;   // batch
    uint c = DTid.y;   // channel
    if (n < params.nSize && c < params.cSize) {
        uint idx = n * params.cSize + c;
        g_accum[idx] = IDENTITY(g_accum[idx] + g_bias[c]);
    }
}
