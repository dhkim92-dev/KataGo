// @NOTE 
// main_proj and d_ouptut are same buffer vulkan buffer handle
// it can make problem because spir-v handle main_proj, and d_output as different object 
#extension GL_EXT_spirv_intrinsics : require
#define SPV_DECORATION_ALIASED 20

#include "common.glsl"

layout(constant_id = 3) const int ELTS_PER_THREAD = 1;

layout(set = 0, binding = 0) 
spirv_decorate(SPV_DECORATION_ALIASED)
buffer MainProj {
  realstore main_proj[];
};

layout(set = 0, binding = 1) readonly buffer GateProj {
  realstore gate_proj[];
};

// comment in below line if exists alias problem. just use main_proj[] only
layout(set = 0, binding = 2) 
spirv_decorate(SPV_DECORATION_ALIASED)
writeonly buffer OutputBuffer {
  realstore d_output[];
};

layout(push_constant) uniform TransformerSwiGLUParams {
  int size;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  const int tileStart = int(gl_WorkGroupID.x) * int(gl_WorkGroupSize.x) * ELTS_PER_THREAD;
  const int lid = int(gl_LocalInvocationID.x);

  for ( int d = 0 ; d < ELTS_PER_THREAD ; d++ ) {
    int s = tileStart + d * int(gl_WorkGroupSize.x) + lid;
    if ( s < size ) {
      float a = LOAD(main_proj, s);
      float b = LOAD(gate_proj, s);
      float silu_a = a / (1.0f + exp(-a));
      STORE(d_output, s, floatToReal(silu_a * b));
      //STORE(main_proj, s, floatToReal(silu_a * b)); comment out this line, and comment in above line if alias problem occur.
    }
  }
}
