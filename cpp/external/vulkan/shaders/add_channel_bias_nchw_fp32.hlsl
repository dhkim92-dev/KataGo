/**
* @author dhkim92.dev@gmail.com
* @brief Add Channel Bias NCHW FP32 compute shader (Identity activation)
* Adds bias to each channel: accum[nc,xy] += bias[nc % C]
*
* Binding 0: accum (RW) - input/output tensor [N*C, H*W]
* Binding 1: bias       - bias vector [C]
*/
#include "common.h"
#include "functions.h"

struct AddChannelBiasNCHWParams {
    uint nnXYLen;
    uint ncSize;
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
    if (nc >= params.ncSize || xy >= params.nnXYLen) return;
    uint idx = nc * params.nnXYLen + xy;
    float val = g_accum[idx] + g_bias[nc];
    g_accum[idx] = val;
}
