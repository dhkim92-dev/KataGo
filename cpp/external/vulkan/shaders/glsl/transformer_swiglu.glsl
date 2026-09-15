// The packed FFN path binds the main and gate descriptors to the same input buffer.
#extension GL_EXT_spirv_intrinsics : require
#define SPV_DECORATION_ALIASED 20

#include "common.glsl"

layout(constant_id = 3) const int ELTS_PER_THREAD = 1;

layout(set = 0, binding = 0) 
spirv_decorate(SPV_DECORATION_ALIASED)
buffer MainProj {
  realstore main_proj[];
};

layout(set = 0, binding = 1)
spirv_decorate(SPV_DECORATION_ALIASED)
readonly buffer GateProj {
  realstore gate_proj[];
};

layout(set = 0, binding = 2) 
spirv_decorate(SPV_DECORATION_ALIASED)
writeonly buffer OutputBuffer {
  realstore d_output[];
};

layout(push_constant) uniform TransformerSwiGLUParams {
  int size;
  int packedInputBatchStride;
  int outputBatchStride;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  const int tileStart = int(gl_WorkGroupID.x) * int(gl_WorkGroupSize.x) * ELTS_PER_THREAD;
  const int lid = int(gl_LocalInvocationID.x);
  const int batchIndex = packedInputBatchStride > 0 ? int(gl_WorkGroupID.y) : 0;
  const int inputBase = batchIndex * packedInputBatchStride;
  const int outputBase = batchIndex * outputBatchStride;

  for ( int d = 0 ; d < ELTS_PER_THREAD ; d++ ) {
    int s = tileStart + d * int(gl_WorkGroupSize.x) + lid;
    if ( s < size ) {
      int mainIndex = inputBase + s;
      int gateIndex = packedInputBatchStride > 0 ? inputBase + outputBatchStride + s : s;
      int outputIndex = outputBase + s;
      float a = LOAD(main_proj, mainIndex);
      float b = LOAD(gate_proj, gateIndex);
      float silu_a = a / (1.0f + exp(-a));
      STORE(d_output, outputIndex, floatToReal(silu_a * b));
      //STORE(main_proj, s, floatToReal(silu_a * b)); comment out this line, and comment in above line if alias problem occur.
    }
  }
}
