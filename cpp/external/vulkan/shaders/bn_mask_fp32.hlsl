/**
* @author dhkim92.dev@gmail.com
* @brief BatchNorm (Mask) + Identity Activation FP32 compute shader
* Format: NCHW
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
StructuredBuffer<float> g_input   : register(t0);
StructuredBuffer<float> g_mask    : register(t1);
StructuredBuffer<float> g_scale   : register(t2);
StructuredBuffer<float> g_bias    : register(t3);
RWStructuredBuffer<float> g_output : register(u0);

[numthreads(BN_DISPATCH_X, BN_DISPATCH_Y, BN_DISPATCH_Z)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint channelIdx = DTid.x;
    uint spatialIdx = DTid.y;
    uint batchIdx = DTid.z;
    
    if (batchIdx >= params.batchSize || channelIdx >= params.numChannels || spatialIdx >= params.nnXYLen) {
        return; 
    }
    uint idx = (batchIdx * params.numChannels + channelIdx) * params.nnXYLen + spatialIdx;
    uint maskIdx = batchIdx * params.nnXYLen + spatialIdx;
    float x = g_input[idx]; 
    float m = g_mask[maskIdx]; 
    float s = g_scale[channelIdx]; 
    float b = g_bias[channelIdx]; 
    float outv = m * (x * s + b);
    // Identity activation
    g_output[idx] = IDENTITY(outv);
}