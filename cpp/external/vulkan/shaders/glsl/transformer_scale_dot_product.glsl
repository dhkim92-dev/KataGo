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

layout(constant_id=3) const int ATTN_BLOCK_Q = 32;
layout(constant_id=4) const int ATTN_BLOCK_KV = 32;
layout(constant_id=5) const int Q_PER_THREAD = 1;
layout(constant_id=6) const int ATTN_HEAD_DIM = 1;
layout(constant_id=7) const int ATTN_V_HEAD_DIM = 1;

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

shared float kTile[ATTN_BLOCK_KV * ATTN_HEAD_DIM];
shared float vTile[ATTN_BLOCK_KV * ATTN_V_HEAD_DIM];
shared float kMaskTile[ATTN_BLOCK_KV];

layout(local_size_x_id=0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  const int localIdx = int(gl_LocalInvocationID.x);
  const int qBlockStart = int(gl_WorkGroupID.x) * (ATTN_BLOCK_Q * Q_PER_THREAD);
  const int bh = int(gl_GlobalInvocationID.y);

  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int kvh = h / (numHeads / numKVHeads);
  const int kvBase = n * numKVHeads + kvh;

  float q[Q_PER_THREAD * ATTN_HEAD_DIM];
  float qMask[Q_PER_THREAD];
  float runningMax[Q_PER_THREAD];
  float runningSum[Q_PER_THREAD];
  float acc[Q_PER_THREAD * ATTN_V_HEAD_DIM];

  for(int qi = 0; qi < Q_PER_THREAD; qi++) {
    int qPos = qBlockStart + qi * ATTN_BLOCK_Q + localIdx;
    qMask[qi] = 0.0f;
    if(qPos < seqLen) {
      qMask[qi] = LOAD(mask, n * seqLen + qPos);
      if(qMask[qi] != 0.0f) {
        for(int d = 0; d < ATTN_HEAD_DIM; d++) {
          q[qi * ATTN_HEAD_DIM + d] = LOAD(Q, (bh * ATTN_HEAD_DIM + d) * seqLen + qPos);
        }
      }
    }
    runningMax[qi] = -1e30f;
    runningSum[qi] = 0.0f;
    for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
      acc[qi * ATTN_V_HEAD_DIM + d] = 0.0f;
    }
  }

  // Iterate over key/value positions in tiles
  for(int kvStart = 0; kvStart < seqLen; kvStart += ATTN_BLOCK_KV) {
    // Cooperatively load K tile into local memory
    for(int t = localIdx; t < ATTN_BLOCK_KV * ATTN_HEAD_DIM; t += ATTN_BLOCK_Q) {
      int tileKPos = t / ATTN_HEAD_DIM;
      int tileD = t % ATTN_HEAD_DIM;
      int globalKPos = kvStart + tileKPos;
      if(globalKPos < seqLen) {
        kTile[tileKPos * ATTN_HEAD_DIM + tileD] = LOAD(K, (kvBase * ATTN_HEAD_DIM + tileD) * seqLen + globalKPos);
      } else {
        kTile[tileKPos * ATTN_HEAD_DIM + tileD] = 0.0f;
      }
    }

    // Cooperatively load V tile
    for(int t = localIdx; t < ATTN_BLOCK_KV * ATTN_V_HEAD_DIM; t += ATTN_BLOCK_Q) {
      int tileKPos = t / ATTN_V_HEAD_DIM;
      int tileD = t % ATTN_V_HEAD_DIM;
      int globalKPos = kvStart + tileKPos;
      if(globalKPos < seqLen) {
        vTile[tileKPos * ATTN_V_HEAD_DIM + tileD] = LOAD(V, (kvBase * ATTN_V_HEAD_DIM + tileD) * seqLen + globalKPos);
      } else {
        vTile[tileKPos * ATTN_V_HEAD_DIM + tileD] = 0.0f;
      }
    }

    // Cooperatively load mask for this KV tile
    for(int t = localIdx; t < ATTN_BLOCK_KV; t += ATTN_BLOCK_Q) {
      int globalKPos = kvStart + t;
      if(globalKPos < seqLen) {
        kMaskTile[t] = LOAD(mask, n * seqLen + globalKPos);
      } else {
        kMaskTile[t] = 0.0f;
      }
    }

    barrier();

    int kvEnd = min(ATTN_BLOCK_KV, seqLen - kvStart);

    // Each thread processes Q_PER_THREAD query positions against the shared KV tile
    for(int qi = 0; qi < Q_PER_THREAD; qi++) {
      int qPos = qBlockStart + qi * ATTN_BLOCK_Q + localIdx;
      if(qPos < seqLen && qMask[qi] != 0.0f) {
        for(int tileK = 0; tileK < kvEnd; tileK++) {
          if(kMaskTile[tileK] == 0.0f)
            continue;

          // Dot product Q . K
          float _dot = 0.0f;
          for(int d = 0; d < ATTN_HEAD_DIM; d++) {
            _dot += q[qi * ATTN_HEAD_DIM + d] * kTile[tileK * ATTN_HEAD_DIM + d];
          }
          _dot *= scale;

          // Online softmax update
          float newMax = max(runningMax[qi], _dot);
          float expOldMax = exp(runningMax[qi] - newMax);
          float expCur = exp(_dot - newMax);

          for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
            acc[qi * ATTN_V_HEAD_DIM + d] *= expOldMax;
          }
          runningSum[qi] = runningSum[qi] * expOldMax + expCur;
          runningMax[qi] = newMax;

          for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
            acc[qi * ATTN_V_HEAD_DIM + d] += expCur * vTile[tileK * ATTN_V_HEAD_DIM + d];
          }
        }
      }
    }

    barrier();
  }

  // Write output for all Q_PER_THREAD positions
  for(int qi = 0; qi < Q_PER_THREAD; qi++) {
    int qPos = qBlockStart + qi * ATTN_BLOCK_Q + localIdx;
    if(qPos < seqLen) {
      if(qMask[qi] == 0.0f) {
        for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
          STORE(d_output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, 0.0f);
        }
      } else {
        float invSum = (runningSum[qi] > 0.0f) ? (1.0f / runningSum[qi]) : 0.0f;
        for(int d = 0; d < ATTN_V_HEAD_DIM; d++) {
          float result = acc[qi * ATTN_V_HEAD_DIM + d] * invSum;
          STORE(d_output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, result);
        }
      }
    }
  }
}
