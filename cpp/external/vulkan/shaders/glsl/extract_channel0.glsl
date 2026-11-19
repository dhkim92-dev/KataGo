/**
 * @author dhkim92-dev
 * Vulkan API: 1.1
 * SPIR-V: 1.3
 * Extract the first spatial channel.
 */
#include "common.glsl"

layout(constant_id = 3) const int USE_NHWC = 0;

layout(push_constant) uniform ExtractChannel0Params {
    int nSize;        // N: number of batches
    int cSize; // C: number of input channels
    int nhwcSpatialSize; // NHWC staging spatial size
    int nchwSpatialStride; // External NCHW spatial stride
    int logicalSpatialSize; // Unpadded spatial size
    int channelsPadded; // NHWC row stride in elements
};

// Descriptor Set bindings
// binding 0: input buffer (N, C, H*W) - read only
// binding 1: output buffer (N, H*W) - read/write
layout(set = 0, binding = 0) readonly buffer InputBlock {
    realstore d_input[];   // N, C, H*W or N, H*W, C
};
layout(set = 0, binding = 1) buffer OutputBlock {
    realstore d_output[]; // N, H*W
};

void extractNCHW(const int nIdx, const int xyIdx) {
    real result = xyIdx < logicalSpatialSize
        ? LOAD(d_input, nIdx * cSize * nchwSpatialStride + xyIdx)
        : ZERO;
    STORE(d_output, nIdx * nchwSpatialStride + xyIdx, result);
}

void extractNHWC(const int nIdx, const int xyIdx) {
    const int channelStride = int(channelsPadded);
    real result = xyIdx < logicalSpatialSize
        ? LOAD(d_input, (nIdx * nhwcSpatialSize + xyIdx) * channelStride)
        : ZERO;
    STORE(d_output, nIdx * nchwSpatialStride + xyIdx, result);
}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
    const int xyIdx = int(gl_GlobalInvocationID.x);
    const int nIdx = int(gl_GlobalInvocationID.y);
    if (xyIdx < nchwSpatialStride && nIdx < nSize) {
        if (USE_NHWC == 1)
            extractNHWC(nIdx, xyIdx);
        else
            extractNCHW(nIdx, xyIdx);
    }
}
