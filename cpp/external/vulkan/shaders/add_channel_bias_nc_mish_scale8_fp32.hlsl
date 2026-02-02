/**
* @author dhkim92.dev@gmail.com
* @brief Add Channel Bias NC FP32 compute shader (Mish Scale8 activation)
* Adds bias to each channel with Mish Scale8:
*   accum[n,c] = (a < 2.5) ? a * tanh(log(1 + exp(a*8))) : a
* where a = accum[n,c] + bias[c]
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
        g_accum[idx] = MISH_SCALE8(g_accum[idx] + g_bias[c]);
    }
}
