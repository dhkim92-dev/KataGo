/**
* @author dhkim92.dev@gmail.com
*/
// BatchNorm (Mask) FP32 compute shader - HLSL version
// Globals: input, output, mask
// Push constants: BatchNormMaskFp32Params { uint batchSize; uint numChannels; uint nnXYLen; }

struct BatchNormMaskFp32Params {
    uint batchSize;
    uint numChannels;
    uint nnXYLen;
};

cbuffer PushConsts : register(b0)
{
    BatchNormMaskFp32Params params;
};

StructuredBuffer<float> g_input   : register(t0);
StructuredBuffer<float> g_mask    : register(t1);
StructuredBuffer<float> g_scale   : register(t2);
StructuredBuffer<float> g_bias    : register(t3);
RWStructuredBuffer<float> g_output : register(u0);

// Workgroup size
[numthreads(16,16,1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint gx = DTid.x; // channel index
    uint gy = DTid.y; // spatial index
    uint gz = DTid.z; // batch index

    uint B = params.batchSize;
    uint C = params.numChannels;
    uint XY = params.nnXYLen;

    if (gz >= B || gx >= C || gy >= XY)
        return;

    // index = ((b*C)+c)*XY + xy
    uint idx = ((gz * C) + gx) * XY + gy;

    float x = g_input[idx];
    float m = g_mask[gz * XY + gy];
    float s = g_scale[gx];
    float b = g_bias[gx];

    float outv = s * x * m + b;

    g_output[idx] = outv;
}
