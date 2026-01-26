/**
* @author dhkim92.dev@gmail.com
* @brief Add Point Wise FP32 compute shader
* Accumulates value into accum: accum[i] += value[i]
*/
#include "common.h"
#include "functions.h"

struct AddPointWiseParams {
    uint totalSize;
};

[[vk::push_constant]]
AddPointWiseParams params;
RWStructuredBuffer<float> g_accum : register(u0);
StructuredBuffer<float> g_value  : register(t0);

[numthreads(ADD_POINTWISE_DISPATCH_X,ADD_POINTWISE_DISPATCH_Y,ADD_POINTWISE_DISPATCH_Z)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= params.totalSize)
        return;
    g_accum[idx] = g_accum[idx] + g_value[idx];
}
