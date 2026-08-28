#include "common.glsl"
#include "functions.glsl"

layout(constant_id=3) const int XY_ELTS_PER_THREAD=1;
layout(constant_id=4) const int NC_ELTS_PER_THREAD=1;

layout(push_constant) uniform AddChannelBiasNCHWParams {
    int ncSize;
    int xySize;
};

layout(set = 0, binding = 0) buffer Accum {
    realstore accum[];
};

layout(set = 0, binding = 1) readonly buffer Biases {
    float biases[];
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int xyOffsetInTile = LocalId0();
  const int xyTileStart = GroupId0() * (LocalSize0() * XY_ELTS_PER_THREAD);
  const int ncBase = GlobalId1() * NC_ELTS_PER_THREAD;

  for(int r = 0; r < NC_ELTS_PER_THREAD; r++) {
    const int nc = ncBase + r;
    if(nc >= ncSize)
      return;
    real bias = floatToReal(biases[nc]);
    int baseIdx = nc * xySize;
    for(int d = 0; d < XY_ELTS_PER_THREAD; d++) {
      int xy = xyTileStart + d * LocalSize0() + xyOffsetInTile;
      if(xy < xySize) {
        int idx = baseIdx + xy;
        real result = LOAD(accum,idx) + bias;
        STORE(accum, idx, result);
      }
    }
  }
}
