/**
* @author dhkim92.dev@gmail.com
* @brief BatchNorm (Mask) + Activation compute shader
* Format: NCHW
* Thread mapping optimized for memory coalescing:
*   x -> spatial (contiguous in NCHW memory layout)
*   y -> channel
*   z -> batch
*/

#include "common.h"
#include "functions.h"

layout(push_constant) uniform BatchNormMaskFp32Params {
    int nSize;
    int cSize;
    int xySize;
};

layout(set = 0, binding = 0) readonly buffer g_input_block {
    realstore d_input[];
};

layout(set = 0, binding = 1) buffer g_output_block {
    realstore d_output[];
};

layout(set = 0, binding = 2) readonly buffer g_scale_block {
    realstore scale[];
};

layout(set = 0, binding = 3) readonly buffer g_bias_block {
    realstore bias[];
};

layout(set = 0, binding = 4) readonly buffer g_mask_block {
    realstore mask[];
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  const int xy = int(gl_GlobalInvocationID.x);
  const int c = int(gl_GlobalInvocationID.y);
  if(c < cSize && xy < xySize) {
    for(int n = 0; n < nSize; n++) {
      int idx = (n * cSize + c) * xySize + xy;
    #if ACTIVATION == 0
      real result = (LOAD(d_input,idx) * LOAD(scale,c) + LOAD(bias,c)) * LOAD(mask,n * xySize + xy);
    #elif ACTIVATION == 1
      real result = fmax(LOAD(d_input,idx) * LOAD(scale,c) + LOAD(bias,c), ZERO) * LOAD(mask,n * xySize + xy);
    #elif ACTIVATION == 2
      float a = LOAD(d_input,idx) * LOAD(scale,c) + LOAD(bias,c);
      real result = floatToReal(a * tanh(a < LOG1PEXPTHRESHOLD ? log1p(exp(a)) : a)) * LOAD(mask,n * xySize + xy);
    #elif ACTIVATION == 12
      float a = LOAD(d_input,idx) * LOAD(scale,c) + LOAD(bias,c);
      real result = floatToReal(a < (LOG1PEXPTHRESHOLD*0.125f) ? a * tanh(log1p(exp(a*8.0f))) : a) * LOAD(mask,n * xySize + xy);
    #elif ACTIVATION == 3
      float a = LOAD(d_input,idx) * LOAD(scale,c) + LOAD(bias,c);
      real result = floatToReal(a / (1.0f + exp(-a))) * LOAD(mask,n * xySize + xy);
    #endif
      STORE(d_output,idx,result);
    }
  }
}

