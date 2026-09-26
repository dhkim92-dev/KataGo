/**
* @author dhkim92.dev@gmail.com
* @brief BatchNorm (Mask) + Activation compute shader
* Format: NCHW or NHWC, selected by USE_NHWC specialization
* Thread mapping optimized for memory coalescing:
*   x -> spatial
*   y -> channel
*   z -> batch
*/

#include "common.glsl"
#include "functions.glsl"

layout(push_constant) uniform BatchNormMaskFp32Params {
    int nSize;
    int cSize;
    int xySize;
    int maskSpatialStride;
    int channelsPadded;
};

layout(constant_id = 3) const int USE_NHWC = 0;

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
void bnMaskElement(int idx, int n, int c, int xy) {
  const real maskValue = xy < maskSpatialStride
    ? LOAD(mask,n * maskSpatialStride + xy)
    : ZERO;
    #if ACTIVATION == 0
  real result = (LOAD(d_input,idx) * LOAD(scale,c) + LOAD(bias,c)) * maskValue;
    #elif ACTIVATION == 1
  real result = fmax(LOAD(d_input,idx) * LOAD(scale,c) + LOAD(bias,c), ZERO) * maskValue;
    #elif ACTIVATION == 2
  float a = LOAD(d_input,idx) * LOAD(scale,c) + LOAD(bias,c);
  real result = floatToReal(a * tanh(a < LOG1PEXPTHRESHOLD ? log1p(exp(a)) : a)) * maskValue;
    #elif ACTIVATION == 12
  float a = LOAD(d_input,idx) * LOAD(scale,c) + LOAD(bias,c);
  real result = floatToReal(a < (LOG1PEXPTHRESHOLD*0.125f) ? a * tanh(log1p(exp(a*8.0f))) : a) * maskValue;
    #elif ACTIVATION == 3
  float a = LOAD(d_input,idx) * LOAD(scale,c) + LOAD(bias,c);
  real result = floatToReal(a / (1.0f + exp(-a))) * maskValue;
    #endif
  STORE(d_output,idx,result);
}

void bnMaskNCHW(int xy, int c) {
  for(int n = 0; n < nSize; n++) {
    const int idx = (n * cSize + c) * xySize + xy;
    bnMaskElement(idx, n, c, xy);
  }
}

void bnMaskNHWC(int xy, int c) {
  const int channelStride = int(channelsPadded);
  for(int n = 0; n < nSize; n++) {
    const int idx = (n * xySize + xy) * channelStride + c;
    if(c < cSize)
      bnMaskElement(idx, n, c, xy);
    else
      STORE(d_output, idx, ZERO);
  }
}

void main() {
  const int xy = int(gl_GlobalInvocationID.x);
  const int c = int(gl_GlobalInvocationID.y);
  if(xy < xySize) {
    if(USE_NHWC == 1 && c < channelsPadded) {
      bnMaskNHWC(xy, c);
    } else if(USE_NHWC == 0 && c < cSize) {
      bnMaskNCHW(xy, c);
    }
  }
}
