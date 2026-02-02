/**
* @author dhkim92.dev@gmail.com
* @brief Add Channel Bias NCHW FP32 compute shader (Identity activation)
* Adds bias to each channel: accum[nc, xy] += bias[nc]
* Matches OpenCL addChannelBiasesNCHW kernel behavior
*
* Binding 0: accum (RW) - input/output tensor [NC, HW]
* Binding 1: bias       - bias vector [NC]
*/
#include "common.h"
#include "functions.h"

struct AddChannelBiasNCHWParams {
    uint ncSize;
    uint xySize;
};

[[vk::push_constant]]
AddChannelBiasNCHWParams params;

[[vk::binding(0, 0)]]
RWStructuredBuffer<float> g_accum;
[[vk::binding(1, 0)]]
StructuredBuffer<float>   g_bias;

[numthreads(ADD_CHANNELS_DISPATCH_X, ADD_CHANNELS_DISPATCH_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint xy = DTid.x;
    uint nc = DTid.y;
    if (nc < params.ncSize && xy < params.xySize) {
        uint idx = nc * params.xySize + xy;
        g_accum[idx] = g_accum[idx] + g_bias[nc];
    }
}
