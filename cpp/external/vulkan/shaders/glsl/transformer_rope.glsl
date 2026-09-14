#ifndef TRANSFORMER_ROPE_GLSL
#define TRANSFORMER_ROPE_GLSL

void applyTransformerRoPE(inout float x0, inout float x1, float cosVal, float sinVal) {
  float out0 = x0 * cosVal - x1 * sinVal;
  float out1 = x0 * sinVal + x1 * cosVal;
  x0 = out0;
  x1 = out1;
}

#endif
