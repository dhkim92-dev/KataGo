#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

#include "common.glsl"
#include "transformer_rope.glsl"

#ifndef ATTENTION_COOPERATIVE_ACC_TYPE
#define ATTENTION_COOPERATIVE_ACC_TYPE 32
#endif

#if ATTENTION_COOPERATIVE_ACC_TYPE == 16
#define coop_acc_dtype float16_t
#elif ATTENTION_COOPERATIVE_ACC_TYPE == 32
#define coop_acc_dtype float
#else
#error "ATTENTION_COOPERATIVE_ACC_TYPE must be 16 or 32"
#endif

layout(constant_id = 3) const int COOP_M_SIZE = 16;
layout(constant_id = 4) const int COOP_N_SIZE = 16;
layout(constant_id = 5) const int COOP_K_SIZE = 16;
layout(constant_id = 6) const int ATTN_HEAD_DIM = 64;
layout(constant_id = 7) const int ATTN_V_HEAD_DIM = 64;
layout(constant_id = 8) const int COOP_Q_TILES_PER_WORKGROUP = 1;
layout(constant_id = 9) const int COOP_PV_N_SIZE = 16;

const int HEAD_DIM_PAD = ((ATTN_HEAD_DIM + COOP_K_SIZE - 1) / COOP_K_SIZE) * COOP_K_SIZE;
const int KV_PAD = ((COOP_N_SIZE + COOP_K_SIZE - 1) / COOP_K_SIZE) * COOP_K_SIZE;
const int V_HEAD_DIM_PAD = ((ATTN_V_HEAD_DIM + COOP_PV_N_SIZE - 1) / COOP_PV_N_SIZE) * COOP_PV_N_SIZE;
const int Q_BLOCK = COOP_M_SIZE * COOP_Q_TILES_PER_WORKGROUP;
const int Q_FRAGMENTS = HEAD_DIM_PAD / COOP_K_SIZE;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

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
  float scale;
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

struct QTileStorage {
  uvec4 alignment;
  float16_t values[HEAD_DIM_PAD * Q_BLOCK];
};
shared QTileStorage qTileStorage;
#define qTile qTileStorage.values

struct KTileStorage {
  uvec4 alignment;
  float16_t values[HEAD_DIM_PAD * COOP_N_SIZE];
};
shared KTileStorage kTileStorage;
#define kTile kTileStorage.values

struct VTileStorage {
  uvec4 alignment;
  float16_t values[KV_PAD * V_HEAD_DIM_PAD];
};
shared VTileStorage vTileStorage;
#define vTile vTileStorage.values

#if ATTENTION_COOPERATIVE_ACC_TYPE == 16
struct ScoreTileStorage {
  uvec4 alignment;
  float16_t values[Q_BLOCK * COOP_N_SIZE];
};
shared ScoreTileStorage scoreTileStorage;
#define scoreTile scoreTileStorage.values

#else
struct ScoreTileStorage {
  uvec4 alignment;
  float values[Q_BLOCK * COOP_N_SIZE];
};
shared ScoreTileStorage scoreTileStorage;
#define scoreTile scoreTileStorage.values
#endif

struct KeyMaskStorage {
  uvec4 alignment;
  float values[COOP_N_SIZE];
};
shared KeyMaskStorage keyMaskStorage;
#define keyMaskTile keyMaskStorage.values

struct ProbabilityTileStorage {
  uvec4 alignment;
  float16_t values[Q_BLOCK * COOP_N_SIZE];
};
shared ProbabilityTileStorage probabilityTileStorage;
#define probabilityTile probabilityTileStorage.values

struct PvTileStorage {
  uvec4 alignment;
  float values[Q_BLOCK * V_HEAD_DIM_PAD];
};
shared PvTileStorage pvTileStorage;
#define pvTile pvTileStorage.values

void loadQueryTile(int qBlockStart, int batchBase, int head) {
  const int localIdx = int(gl_LocalInvocationID.x);
  const int localSize = int(gl_WorkGroupSize.x);
  for(int i = localIdx; i < HEAD_DIM_PAD * Q_BLOCK; i += localSize) {
    const int d = i / Q_BLOCK;
    const int q = i % Q_BLOCK;
    const int qPos = qBlockStart + q;
    if(d < ATTN_HEAD_DIM && qPos < seqLen) {
      qTile[i] = float16_t(LOAD(Q, batchBase + qOffset + (head * ATTN_HEAD_DIM + d) * seqLen + qPos));
    }
    else {
      qTile[i] = float16_t(0.0);
    }
  }
}

void loadKeyTile(int kvStart, int batchIndex, int batchBase, int kvHead) {
  const int localIdx = int(gl_LocalInvocationID.x);
  const int localSize = int(gl_WorkGroupSize.x);
  for(int i = localIdx; i < HEAD_DIM_PAD * COOP_N_SIZE; i += localSize) {
    const int d = i / COOP_N_SIZE;
    const int k = i % COOP_N_SIZE;
    const int globalKPos = kvStart + k;
    if(d < ATTN_HEAD_DIM && globalKPos < seqLen) {
      kTile[i] = float16_t(LOAD(K, batchBase + kOffset + (kvHead * ATTN_HEAD_DIM + d) * seqLen + globalKPos));
    }
    else {
      kTile[i] = float16_t(0.0);
    }
  }
  for(int i = localIdx; i < COOP_N_SIZE; i += localSize) {
    const int globalKPos = kvStart + i;
    keyMaskTile[i] = globalKPos < seqLen ? LOAD(mask, batchIndex * seqLen + globalKPos) : 0.0;
  }
}

void applyQueryRope(int qBlockStart, int head) {
  const int subgroupIdx = int(gl_SubgroupID);
  const int localIdx = int(gl_SubgroupInvocationID);
  if(localIdx >= COOP_M_SIZE)
    return;
  const int qInBlock = subgroupIdx * COOP_M_SIZE + localIdx;
  const int qPos = qBlockStart + qInBlock;
  if(qPos >= seqLen)
    return;
  const int ropeTableHead = head * numKVHeads / numHeads;
  const int pairCount = min(ropeNumPairs, ATTN_HEAD_DIM / 2);
  for(int pairIdx = 0; pairIdx < pairCount; pairIdx++) {
    const int d0 = pairIdx * 2;
    const int d1 = d0 + 1;
    const int tableIdx = learnableRope != 0
      ? (ropeTableHead * ropeNumPairs + pairIdx) * seqLen + qPos
      : pairIdx * seqLen + qPos;
    float q0 = float(qTile[d0 * Q_BLOCK + qInBlock]);
    float q1 = float(qTile[d1 * Q_BLOCK + qInBlock]);
    applyTransformerRoPE(q0, q1, ropeCosTable[tableIdx], ropeSinTable[tableIdx]);
    qTile[d0 * Q_BLOCK + qInBlock] = float16_t(q0);
    qTile[d1 * Q_BLOCK + qInBlock] = float16_t(q1);
  }
}

void applyKeyRope(int kvStart, int kvHead) {
  const int localIdx = int(gl_LocalInvocationID.x);
  const int localSize = int(gl_WorkGroupSize.x);
  const int ropeTableHead = kvHead;
  const int pairCount = min(ropeNumPairs, ATTN_HEAD_DIM / 2);
  for(int i = localIdx; i < COOP_N_SIZE * pairCount; i += localSize) {
    const int pairIdx = i / COOP_N_SIZE;
    const int tileKPos = i % COOP_N_SIZE;
    const int globalKPos = kvStart + tileKPos;
    if(globalKPos < seqLen) {
      const int d0 = pairIdx * 2;
      const int d1 = d0 + 1;
      const int tableIdx = learnableRope != 0
        ? (ropeTableHead * ropeNumPairs + pairIdx) * seqLen + globalKPos
        : pairIdx * seqLen + globalKPos;
      float k0 = float(kTile[d0 * COOP_N_SIZE + tileKPos]);
      float k1 = float(kTile[d1 * COOP_N_SIZE + tileKPos]);
      applyTransformerRoPE(k0, k1, ropeCosTable[tableIdx], ropeSinTable[tableIdx]);
      kTile[d0 * COOP_N_SIZE + tileKPos] = float16_t(k0);
      kTile[d1 * COOP_N_SIZE + tileKPos] = float16_t(k1);
    }
  }
}

void loadValueTile(int kvStart, int batchBase, int kvHead) {
  const int localIdx = int(gl_LocalInvocationID.x);
  const int localSize = int(gl_WorkGroupSize.x);
  for(int i = localIdx; i < KV_PAD * V_HEAD_DIM_PAD; i += localSize) {
    const int k = i / V_HEAD_DIM_PAD;
    const int d = i % V_HEAD_DIM_PAD;
    const int globalKPos = kvStart + k;
    if(k < COOP_N_SIZE && globalKPos < seqLen && d < ATTN_V_HEAD_DIM) {
      vTile[i] = float16_t(LOAD(V, batchBase + vOffset +
        (kvHead * ATTN_V_HEAD_DIM + d) * seqLen + globalKPos));
    }
    else {
      vTile[i] = float16_t(0.0);
    }
  }
}

void storeScoreTile(
  coopmat<coop_acc_dtype, gl_ScopeSubgroup, COOP_M_SIZE, COOP_N_SIZE, gl_MatrixUseAccumulator> scoreFrag,
  int qBase
) {
  coopMatStore(scoreFrag, scoreTile, qBase, Q_BLOCK, gl_CooperativeMatrixLayoutColumnMajor);
}

float getScore(int qBase, int q, int k) {
#if ATTENTION_COOPERATIVE_ACC_TYPE == 16
  return float(scoreTile[k * Q_BLOCK + qBase + q]);
#else
  return scoreTile[k * Q_BLOCK + qBase + q];
#endif
}

void main() {
  const int localIdx = int(gl_LocalInvocationID.x);
  const int subgroupIdx = int(gl_SubgroupID);
  const int subgroupLocalIdx = int(gl_SubgroupInvocationID);
  const int qBase = subgroupIdx * COOP_M_SIZE;
  const int qBlockStart = int(gl_WorkGroupID.x) * Q_BLOCK;
  const int bh = int(gl_WorkGroupID.y);
  const int batch = bh / numHeads;
  const int head = bh % numHeads;
  const int kvHead = head / (numHeads / numKVHeads);
  const int qBatchBase = batch * qBatchStride;
  const int kBatchBase = batch * kBatchStride;
  const int vBatchBase = batch * vBatchStride;
  const int qPos = qBlockStart + qBase + subgroupLocalIdx;

  float qMask = 0.0;
  float runningMax = -1e30;
  float runningSum = 0.0;
  float acc[ATTN_V_HEAD_DIM];
  for(int d = 0; d < ATTN_V_HEAD_DIM; d++)
    acc[d] = 0.0;

  loadQueryTile(qBlockStart, qBatchBase, head);
  barrier();

  if(subgroupLocalIdx < COOP_M_SIZE && qPos < seqLen)
    qMask = LOAD(mask, batch * seqLen + qPos);
  const bool qValid = subgroupLocalIdx < COOP_M_SIZE && qPos < seqLen && qMask != 0.0;

  if(useRope != 0)
    applyQueryRope(qBlockStart, head);
  subgroupBarrier();

  coopmat<float16_t, gl_ScopeSubgroup, COOP_M_SIZE, COOP_K_SIZE, gl_MatrixUseA> qFrags[Q_FRAGMENTS];
  coopmat<float16_t, gl_ScopeSubgroup, COOP_K_SIZE, COOP_N_SIZE, gl_MatrixUseB> kFrag;
  coopmat<coop_acc_dtype, gl_ScopeSubgroup, COOP_M_SIZE, COOP_N_SIZE, gl_MatrixUseAccumulator> scoreFrag;

  for(int kFragIdx = 0; kFragIdx < Q_FRAGMENTS; kFragIdx++)
    coopMatLoad(qFrags[kFragIdx], qTile, kFragIdx * COOP_K_SIZE * Q_BLOCK + qBase, Q_BLOCK, gl_CooperativeMatrixLayoutColumnMajor);

  for(int kvStart = 0; kvStart < seqLen; kvStart += COOP_N_SIZE) {
    loadKeyTile(kvStart, batch, kBatchBase, kvHead);
    loadValueTile(kvStart, vBatchBase, kvHead);
    barrier();

    if(useRope != 0)
      applyKeyRope(kvStart, kvHead);
    barrier();

    scoreFrag = coopmat<coop_acc_dtype, gl_ScopeSubgroup, COOP_M_SIZE, COOP_N_SIZE, gl_MatrixUseAccumulator>(coop_acc_dtype(0.0));
    for(int kOffset = 0; kOffset < HEAD_DIM_PAD; kOffset += COOP_K_SIZE) {
      const int kFragIdx = kOffset / COOP_K_SIZE;
      coopMatLoad(kFrag, kTile, kOffset * COOP_N_SIZE, COOP_N_SIZE, gl_CooperativeMatrixLayoutRowMajor);
      scoreFrag = coopMatMulAdd(qFrags[kFragIdx], kFrag, scoreFrag);
    }
    storeScoreTile(scoreFrag, qBase);
    subgroupBarrier();

    // Cooperative matrix operands are FP16 on the supported devices. Keep the
    // online-softmax state and final accumulation in FP32, but round only the
    // current probability tile to FP16 for the cooperative PV multiplication.
    // This is the same precision boundary used by the CUDA tensor-core flash
    // attention path, while avoiding the scalar k*d inner loop.
    float oldWeight = 1.0;
    if(subgroupLocalIdx < COOP_M_SIZE) {
      float tileMax = -1e30;
      if(qValid) {
        for(int k = 0; k < COOP_N_SIZE; k++) {
          if(keyMaskTile[k] != 0.0)
            tileMax = max(tileMax, getScore(qBase, subgroupLocalIdx, k) * scale);
        }
      }
      const float newMax = max(runningMax, tileMax);
      oldWeight = exp(runningMax - newMax);
      if(qValid) {
        runningSum *= oldWeight;
        runningMax = newMax;
      }
      for(int k = 0; k < COOP_N_SIZE; k++) {
        float weight = 0.0;
        if(qValid && keyMaskTile[k] != 0.0)
          weight = exp(getScore(qBase, subgroupLocalIdx, k) * scale - newMax);
        if(qValid)
          runningSum += weight;
        probabilityTile[k * Q_BLOCK + qBase + subgroupLocalIdx] = float16_t(weight);
      }
    }

    subgroupBarrier();

    coopmat<float16_t, gl_ScopeSubgroup, COOP_M_SIZE, COOP_N_SIZE, gl_MatrixUseA> probabilityFrag;
    coopMatLoad(
      probabilityFrag, probabilityTile, qBase, Q_BLOCK,
      gl_CooperativeMatrixLayoutColumnMajor
    );

    for(int valueStart = 0; valueStart < V_HEAD_DIM_PAD; valueStart += COOP_PV_N_SIZE) {
      coopmat<float16_t, gl_ScopeSubgroup, COOP_N_SIZE, COOP_PV_N_SIZE, gl_MatrixUseB> valueFrag;
      coopmat<float, gl_ScopeSubgroup, COOP_M_SIZE, COOP_PV_N_SIZE, gl_MatrixUseAccumulator> pvFrag;
      pvFrag = coopmat<float, gl_ScopeSubgroup, COOP_M_SIZE, COOP_PV_N_SIZE, gl_MatrixUseAccumulator>(0.0);
      coopMatLoad(
        valueFrag, vTile, valueStart, V_HEAD_DIM_PAD,
        gl_CooperativeMatrixLayoutRowMajor
      );
      pvFrag = coopMatMulAdd(probabilityFrag, valueFrag, pvFrag);
      coopMatStore(
        pvFrag, pvTile, valueStart * Q_BLOCK + qBase, Q_BLOCK,
        gl_CooperativeMatrixLayoutColumnMajor
      );
      subgroupBarrier();

      if(subgroupLocalIdx < COOP_M_SIZE && qValid) {
        const int valueCount = min(COOP_PV_N_SIZE, ATTN_V_HEAD_DIM - valueStart);
        for(int d = 0; d < valueCount; d++)
          acc[valueStart + d] = acc[valueStart + d] * oldWeight +
            pvTile[(valueStart + d) * Q_BLOCK + qBase + subgroupLocalIdx];
      }
      subgroupBarrier();
    }
  }

  if(subgroupLocalIdx < COOP_M_SIZE && qPos < seqLen) {
    if(qMask == 0.0) {
      for(int d = 0; d < ATTN_V_HEAD_DIM; d++)
        STORE(d_output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, floatToReal(0.0));
    }
    else {
      const float invSum = runningSum > 0.0 ? 1.0 / runningSum : 0.0;
      for(int d = 0; d < ATTN_V_HEAD_DIM; d++)
        STORE(d_output, (bh * ATTN_V_HEAD_DIM + d) * seqLen + qPos, floatToReal(acc[d] * invSum));
    }
  }
}
