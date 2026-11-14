
#include "common.glsl"
#include "transformer_rope.glsl"

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

layout(set = 0, binding = 5) readonly buffer RopeCosine {
    float ropeCosTable[];
};

layout(set = 0, binding = 6) readonly buffer RopeSine {
    float ropeSinTable[];
};

layout(push_constant) uniform ScaleDotProductAttentionParams {
    int seqLen;
    int numHeads;
    int numKVHeads;
    float scale; // 1/sqrt(headDim)
    int qOffset;
    int kOffset;
    int vOffset;
    int qBatchStride;
    int kBatchStride;
    int vBatchStride;
    int useRope;
    int learnableRope;
    int ropeNumPairs;
    int ropeReserved;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  const int qPos = int(gl_GlobalInvocationID.x);
  const int bh = int(gl_GlobalInvocationID.y);  // batch * numHeads + head
  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int kvh = h / (numHeads / numKVHeads);
  const int qBatchBase = n * qBatchStride;
  const int kBatchBase = n * kBatchStride;
  const int vBatchBase = n * vBatchStride;

  if(qPos >= seqLen)
    return;

  float qMask = LOAD(mask, n * seqLen + qPos);
  if(qMask == 0.0f) {
    for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
      STORE(d_output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, floatToReal(0.0f));
    }
    return;
  }

  // Load query vector into private registers
  float q[ATTN_HEAD_DIM];
  for(int d = 0; d < ATTN_HEAD_DIM; d++) {
    q[d] = LOAD(Q, qBatchBase + qOffset + (h * ATTN_HEAD_DIM + d) * seqLen + qPos);
  }
  if(useRope != 0) {
    int ropeTableHead = h * numKVHeads / numHeads;
    for(int pairIdx = 0; pairIdx < ropeNumPairs; pairIdx++) {
      int d0 = pairIdx * 2;
      int d1 = d0 + 1;
      int tableIdx = learnableRope != 0
        ? (ropeTableHead * ropeNumPairs + pairIdx) * seqLen + qPos
        : pairIdx * seqLen + qPos;
      float q0 = q[d0];
      float q1 = q[d1];
      applyTransformerRoPE(q0, q1, ropeCosTable[tableIdx], ropeSinTable[tableIdx]);
      q[d0] = q0;
      q[d1] = q1;
    }
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
    int ropeAwarePairCount = (ATTN_HEAD_DIM + 1) / 2;
    for(int pairIdx = 0; pairIdx < ropeAwarePairCount; pairIdx++) {
      int d0 = pairIdx * 2;
      int d1 = d0 + 1;
      float k0 = LOAD(K, kBatchBase + kOffset + (kvh * ATTN_HEAD_DIM + d0) * seqLen + kPos);
      float k1 = d1 < ATTN_HEAD_DIM
        ? LOAD(K, kBatchBase + kOffset + (kvh * ATTN_HEAD_DIM + d1) * seqLen + kPos)
        : 0.0f;
      if(useRope != 0 && pairIdx < ropeNumPairs) {
        int tableIdx = learnableRope != 0
          ? (kvh * ropeNumPairs + pairIdx) * seqLen + kPos
          : pairIdx * seqLen + kPos;
        applyTransformerRoPE(k0, k1, ropeCosTable[tableIdx], ropeSinTable[tableIdx]);
      }
      _dot += q[d0] * k0;
      if(d1 < ATTN_HEAD_DIM)
        _dot += q[d1] * k1;
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
      float vVal = LOAD(V, vBatchBase + vOffset + (kvh * ATTN_V_HEAD_DIM + d) * seqLen + kPos);
      acc[d] += expCur * vVal;
    }
  }

  // Normalize and write output
  float invSum = (runningSum > 0.0f) ? (1.0f / runningSum) : 0.0f;
  for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
    float result = acc[d] * invSum;
    STORE(d_output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, floatToReal(result));
  }
}
