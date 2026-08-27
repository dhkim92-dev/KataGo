// @author dhkim92.dev@gmail.com
// Add Channel Bias NC - GLSL compute shader

#include "common.h"
#include "functions.h"

layout(push_constant) uniform AddChannelBiasesNCParams {
    uint nSize;
    uint cSize;
};

layout(std430, binding = 0) buffer Accum {
    float accum[];
};

layout(std430, binding = 1) readonly buffer Bias {
    float biases[];
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  const int c =  int(gl_GlobalInvocationID.x);  
  const int n = int(gl_GlobalInvocationID.y);

  if(n < nSize && c < cSize) {
#if ACTIVATION == 0
    accum[n * cSize + c] += biases[c];
#elif ACTIVATION == 1
    accum[n * cSize + c] = fmax(accum[n * cSize + c] + biases[c], 0.0f);
#elif ACTIVATION == 2
    float a = accum[n * cSize + c] + biases[c];
    accum[n * cSize + c] = floatToReal(a * tanh(a < LOG1PEXPTHRESHOLD ? log1p(exp(a)) : a));
#elif ACTIVATION == 12
    float a = accum[n * cSize + c] + biases[c];
    accum[n * cSize + c] = floatToReal(a < (LOG1PEXPTHRESHOLD*0.125f) ? a * tanh(log1p(exp(a*8.0f))) : a);
#elif ACTIVATION == 3
    float a = accum[n * cSize + c] + biases[c];
    accum[n * cSize + c] = a / (1.0f + exp(-a));
#endif
  }
}
