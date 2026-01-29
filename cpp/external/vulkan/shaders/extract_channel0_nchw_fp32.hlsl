/**
 * @author dhkim92.dev@gmail.com
 * @brief Extract Channel 0 NCHW FP32 compute shader
 * Extracts the first channel (channel 0) from NCHW formatted input tensor
 * Input: N, C, H*W (NCHW format)
 * Output: N, H*W (first channel only)
 * 
 * This kernel extracts the spatial data from channel 0 for each batch,
 * effectively reducing the tensor from (N, C, H, W) to (N, H, W).
 */

#include "common.h"
#include "functions.h"

struct ExtractChannel0NCHWParams {
    uint batchSize;        // N: number of batches
    uint numInputChannels; // C: number of input channels
    uint nnXYLen;          // H*W: spatial size
};

[[vk::push_constant]]
ExtractChannel0NCHWParams params;

// Descriptor Set bindings
// binding 0: input buffer (N, C, H*W) - read only
// binding 1: output buffer (N, H*W) - read/write
[[vk::binding(0, 0)]]
StructuredBuffer<float> g_input;   // N, C, H*W
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> g_output; // N, H*W

[numthreads(EXTRACT_CHANNEL0_XYSTRIDE, EXTRACT_CHANNEL0_NSTRIDE, 1)]
void main(
    uint3 DTid : SV_DispatchThreadID
)
{
    uint xyIdx = DTid.x;  // Spatial index (H*W)
    uint nIdx = DTid.y;   // Batch index
    uint nSize = params.batchSize;
    uint cSize = params.numInputChannels;
    uint xySize = params.nnXYLen;

    // Bounds check
    if (xyIdx < xySize && nIdx < nSize) {
        // Input index: channel 0 for batch nIdx at spatial position xyIdx
        // Input layout: [N][C][H*W]
        // For channel 0: inputIdx = nIdx * cSize * xySize + 0 * xySize + xyIdx
        //              = nIdx * cSize * xySize + xyIdx
        uint inputIdx = nIdx * cSize * xySize + xyIdx;
        
        // Output index: batch nIdx at spatial position xyIdx
        // Output layout: [N][H*W]
        uint outputIdx = nIdx * xySize + xyIdx;
        
        // Extract channel 0 value and write to output
        float value = g_input[inputIdx];
        g_output[outputIdx] = value;
    }
}
