
#if PRECISION_STORAGE == 16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : enable
  #define realstore float16_t
  #define LOAD(__buf,__x) vload_half((__x),(__buf))
  #define STORE(__buf,__x,__y) vstore_half((__y),(__x),(__buf))
#else
  #define realstore float
  #define LOAD(__buf,__x) ((__buf)[(__x)])
  #define STORE(__buf,__x,__y) ((__buf)[(__x)] = (__y))
#endif

layout(constant_id=3) const int ATTN_HEAD_DIM = 1;
layout(constant_id=4) const int ATTN_V_HEAD_DIM = 1;

layout(set = 0, binding = 0) readonly buffer Query {
    realstore Q[];
};

layout(set = 0, binding = 1) readonly buffer Key {
    realstore K[];
};


layout(set = 0, binding = 2) readonly buffer Value {
    realstore V[];
};

layout(set = 0, binding = 3) writeonly buffer Output {
    realstore d_output[];
};

layout(set = 0, binding = 4) readonly buffer Mask {
    realstore mask[];
};

layout(push_constant) uniform ScaleDotProductAttentionParams {
    int seqLen;
    int numHeads;
    int numKVHeads;
    float scale; // 1/sqrt(headDim)
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  const int qPos = int(gl_GlobalInvocationID.x);
  const int bh = int(gl_GlobalInvocationID.y);  // batch * numHeads + head
  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int kvh = h / (numHeads / numKVHeads);
  const int kvBase = n * numKVHeads + kvh;

  if(qPos >= seqLen)
    return;

  float qMask = LOAD(mask, n * seqLen + qPos);
  if(qMask == 0.0f) {
    for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
      STORE(d_output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, 0.0f);
    }
    return;
  }

  // Load query vector into private registers
  float q[ATTN_HEAD_DIM];
  for(int d = 0; d < ATTN_HEAD_DIM; d++) {
    q[d] = LOAD(Q, (bh * ATTN_HEAD_DIM + d) * seqLen + qPos);
  }

  // Online softmax: iterate over all key positions
  float runningMax = -1e30f;
  float runningSum = 0.0f;
  float acc[ATTN_V_HEAD_DIM];
  for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
    acc[d] = 0.0f;
  }

  for(int kPos = 0; kPos < seqLen; kPos++) {
    float kMask = LOAD(mask, n * seqLen + kPos);
    if(kMask == 0.0f)
      continue;

    // Dot product Q . K
    float _dot = 0.0f;
    for(int d = 0; d < ATTN_HEAD_DIM; d++) {
      float kVal = LOAD(K, (kvBase * ATTN_HEAD_DIM + d) * seqLen + kPos);
      _dot += q[d] * kVal;
    }
    _dot *= scale;

    // Online softmax update
    float newMax = max(runningMax, _dot);
    float expOldMax = exp(runningMax - newMax);
    float expCur = exp(_dot - newMax);

    for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
      acc[d] *= expOldMax;
    }
    runningSum = runningSum * expOldMax + expCur;
    runningMax = newMax;

    for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
      float vVal = LOAD(V, (kvBase * ATTN_V_HEAD_DIM + d) * seqLen + kPos);
      acc[d] += expCur * vVal;
    }
  }

  // Normalize and write output
  float invSum = (runningSum > 0.0f) ? (1.0f / runningSum) : 0.0f;
  for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
    float result = acc[d] * invSum;
    STORE(d_output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, result);
  }
}
