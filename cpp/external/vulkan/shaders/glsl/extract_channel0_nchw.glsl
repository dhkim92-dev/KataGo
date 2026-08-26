
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

layout(push_constant) uniform ExtractChannel0NCHWParams {
    uint nSize;        // N: number of batches
    uint cSize; // C: number of input channels
    uint xySize;          // H*W: spatial size
};

// Descriptor Set bindings
// binding 0: input buffer (N, C, H*W) - read only
// binding 1: output buffer (N, H*W) - read/write
layout(set = 0, binding = 0) readonly buffer InputBlock {
    realstore d_input[];   // N, C, H*W
};
layout(set = 0, binding = 1) buffer OutputBlock {
    realstore d_output[]; // N, H*W
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main()
{
    const int xyIdx = int(gl_GlobalInvocationID.x);
    const int nIdx = int(gl_GlobalInvocationID.y);
    if ( xyIdx < xySize && nIdx < nSize ) {
        real result = LOAD(d_input, nIdx * cSize * xySize + xyIdx );
        STORE(d_output, nIdx * xySize + xyIdx, result);
    }
}
