/**
* @author dhkim92.dev@gmail.com
* @brief BatchNorm (Mask) + Mish Scale8 Activation FP32 compute shader
* Format: NCHW
* Thread mapping optimized for memory coalescing:
*   x -> spatial (contiguous in NCHW memory layout)
*   y -> channel
*   z -> batch
*/
#include "common.h"
#include "functions.h"

struct BatchNormMaskFp32Params {
    uint batchSize;
    uint numChannels;
    uint nnXYLen;
};

[[vk::push_constant]]
BatchNormMaskFp32Params params;
[[vk::binding(0, 0)]]
StructuredBuffer<float> g_input;
[[vk::binding(1, 0)]]
StructuredBuffer<float> g_mask;
[[vk::binding(2, 0)]]
StructuredBuffer<float> g_scale;
[[vk::binding(3, 0)]]
StructuredBuffer<float> g_bias;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> g_output;

[numthreads(BN_DISPATCH_X, BN_DISPATCH_Y, BN_DISPATCH_Z)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint spatialIdx = DTid.x;   // x -> spatial (contiguous memory access)
    uint channelIdx = DTid.y;   // y -> channel
    uint batchIdx = DTid.z;     // z -> batch
    
    if (batchIdx >= params.batchSize || channelIdx >= params.numChannels || spatialIdx >= params.nnXYLen) {
        return; 
    }
    uint idx = (batchIdx * params.numChannels + channelIdx) * params.nnXYLen + spatialIdx;
    uint maskIdx = batchIdx * params.nnXYLen + spatialIdx;
    float x = g_input[idx]; 
    float m = g_mask[maskIdx]; 
    float s = g_scale[channelIdx]; 
    float b = g_bias[channelIdx]; 
    float a = m * (x * s + b);
    // MishScale8 activation: a < 2.5f ? a * tanh(log1p(exp(a*8.0f))) : a
    g_output[idx] = MISH_SCALE8(a);
}
