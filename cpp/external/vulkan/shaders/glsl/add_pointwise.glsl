
/**
* @brief Add Point Wise FP32 compute shader
* Accumulates value into accum: accum[i] += value[i]
*/
#include "common.glsl"

layout(constant_id=3) const int ELTS_PER_THREAD = 1;

layout(push_constant) uniform AddPointWiseParams {
    uint size;
};

layout(set = 0, binding = 0) buffer accum_block {
    realstore accum[];
};

layout(set = 0, binding = 1) readonly buffer value_block {
    realstore value[];
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  const int offsetInTile = LocalId0();
  const int tileStart = GroupId0() * (LocalSize0() * ELTS_PER_THREAD);

  for(int d = 0; d < ELTS_PER_THREAD; d++) {
    int s = tileStart + d * LocalSize0() + offsetInTile;
    if(s < size) {
      real result = LOAD(accum,s) + LOAD(value,s);
      STORE(accum,s,result);
    }
  }
}
