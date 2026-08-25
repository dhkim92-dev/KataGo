/**
* @author dhkim92.dev@gmail.com
* @brief Add Channel Bias NCHW FP32 compute shader (Identity activation)
* Adds bias to each channel: accum[nc, xy] += bias[nc]
* Matches OpenCL addChannelBiasesNCHW kernel behavior
*
* Binding 0: accum (RW) - input/output tensor [NC, HW]
* Binding 1: bias       - bias vector [NC]
* Defines
* - XY_ELTS_PER_THREAD
* - NC_ELTS_PER_THREAD
*/

#include "common.h"
#include "functions.h"

layout(constant_id = 3) const int XY_ELTS_PER_THREAD = 1;
layout(constant_id = 4) const int NC_ELTS_PER_THREAD = 1;

layout(push_constant) uniform AddChannelBiasNCHWParams {
    int ncSize;
    int xySize;
};

layout(set = 0, binding = 0) buffer Accum {
    realstore accum[];
};

layout(set = 0, binding = 1) readonly buffer Bias {
    float biases[];
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  const int xyOffsetInTile = int(gl_LocalInvocationID.x);
  const int xyTileStart = int(gl_WorkGroupID.x) * int(gl_WorkGroupSize.x) * XY_ELTS_PER_THREAD;
  const int ncBase = int(gl_GlobalInvocationID.y)  * NC_ELTS_PER_THREAD;

  for(int r = 0; r < NC_ELTS_PER_THREAD; r++) {
    const int nc = ncBase + r;
    if(nc >= ncSize)
      return;
    real bias = floatToReal(biases[nc]);
    int baseIdx = nc * xySize;
    #pragma unroll
    for(int d = 0; d < XY_ELTS_PER_THREAD; d++) {
      int xy = xyTileStart + d * int(gl_WorkGroupSize.x) + xyOffsetInTile;
      if(xy < xySize) {
        int idx = baseIdx + xy;
        real result = LOAD(accum,idx) + bias;
        STORE(accum, idx, result);
      }
    }
  }
}
