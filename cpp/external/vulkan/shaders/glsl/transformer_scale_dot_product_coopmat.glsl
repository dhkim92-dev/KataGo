#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

#include "common.glsl"

// This shader is the cooperative-matrix attention path. It is intended for
// FP16 storage; PRECISION selects the arithmetic precision used by the scalar
// online-softmax update while the cooperative-matrix inputs remain FP16.
#ifndef ACC_TYPE
#define ACC_TYPE 32
#endif

#if ACC_TYPE == 16
#define acc_dtype float16_t
#elif ACC_TYPE == 32
#define acc_dtype float
#else
#error "ACC_TYPE must be 16 or 32"
#endif

// Local-size specialization IDs are reserved for the workgroup layout.
layout(constant_id = 3) const int MSize = 16;
layout(constant_id = 4) const int NSize = 16;
layout(constant_id = 5) const int KSize = 16;
layout(constant_id = 6) const int MWG = 32;
layout(constant_id = 7) const int NWG = 32;
layout(constant_id = 8) const int KWG = 32;
layout(constant_id = 9) const int MWAVE = 32;
layout(constant_id = 10) const int NWAVE = 32;
layout(constant_id = 11) const int ATTN_HEAD_DIM = 64;
layout(constant_id = 12) const int ATTN_V_HEAD_DIM = 16;

#define MWI (MWG / MWAVE)
#define NWI (NWG / NWAVE)
#define KPAD (((ATTN_HEAD_DIM + KWG - 1) / KWG) * KWG)

layout(set = 0, binding = 0) readonly buffer Query {
    float16_t Q[];
};

layout(set = 0, binding = 1) readonly buffer Key {
    float16_t K[];
};

layout(set = 0, binding = 2) readonly buffer Value {
    float16_t V[];
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
    float scale;
    int qOffset;
    int kOffset;
    int vOffset;
    int qBatchStride;
    int kBatchStride;
    int vBatchStride;
};

// Q is [query, dimension] column-major, K is [dimension, key] row-major,
// and V is [key, value] column-major. The shared tiles use those same matrix
// layouts so cooperative loads do not need a second transpose.
struct QTileStorage {
  uvec4 alignment;
  float16_t values[KPAD * MWG];
};
struct KTileStorage {
  uvec4 alignment;
  float16_t values[KPAD * NWG];
};
struct ProbabilityTileStorage {
  uvec4 alignment;
  float16_t values[MWG * NWG];
};
struct AccumulatorTileStorage {
  uvec4 alignment;
  acc_dtype values[MWG * NWG];
};
struct OutputAccumulatorStorage {
  uvec4 alignment;
  float values[MWG * ATTN_V_HEAD_DIM];
};

shared QTileStorage qTileStorage;
shared KTileStorage kTileStorage;
shared ProbabilityTileStorage probabilityTileStorage;
shared AccumulatorTileStorage scoreTileStorage;
shared OutputAccumulatorStorage outputAccumulatorStorage;
shared float qMaskTile[MWG];
shared float kMaskTile[NWG];
shared float runningMaxTile[MWG];
shared float runningSumTile[MWG];
shared float tileMaxTile[MWG];
shared float tileSumTile[MWG];

#define qTile qTileStorage.values
#define kTile kTileStorage.values
#define vTile kTileStorage.values
#define probabilityTile probabilityTileStorage.values
#define scoreTile scoreTileStorage.values
#define tileOutput scoreTileStorage.values
#define outputAccumulator outputAccumulatorStorage.values

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int tid =
    int(gl_LocalInvocationID.x) +
    int(gl_WorkGroupSize.x) * (
      int(gl_LocalInvocationID.y) +
      int(gl_WorkGroupSize.y) * int(gl_LocalInvocationID.z)
    );
  const int numThreads = int(gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z);
  const int subgroupId = int(gl_SubgroupID);
  const int subgroupLane = int(gl_SubgroupInvocationID);
  const int subgroupSize = int(gl_SubgroupSize);
  const int numSubgroups = int(gl_NumSubgroups);

  const int qBlockStart = int(gl_WorkGroupID.x) * MWG;
  const int bh = int(gl_WorkGroupID.y);
  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int kvh = h / (numHeads / numKVHeads);
  const int qBatchBase = n * qBatchStride;
  const int kBatchBase = n * kBatchStride;
  const int vBatchBase = n * vBatchStride;

  // Load the complete Q tile once. K and V are reloaded for each sequence
  // tile, while all out-of-range elements are explicitly zero-filled.
  for(int i = tid; i < KPAD * MWG; i += numThreads) {
    const int d = i / MWG;
    const int q = i - d * MWG;
    const int globalQ = qBlockStart + q;
    if(d < ATTN_HEAD_DIM && globalQ < seqLen) {
      qTile[i] = float16_t(LOAD(Q, qBatchBase + qOffset +
        (h * ATTN_HEAD_DIM + d) * seqLen + globalQ));
    } else {
      qTile[i] = float16_t(0.0);
    }
  }
  for(int q = tid; q < MWG; q += numThreads) {
    const int globalQ = qBlockStart + q;
    qMaskTile[q] = globalQ < seqLen ? float(LOAD(mask, n * seqLen + globalQ)) : 0.0;
  }

  for(int i = tid; i < MWG * ATTN_V_HEAD_DIM; i += numThreads) {
    outputAccumulator[i] = 0.0;
  }
  for(int q = tid; q < MWG; q += numThreads) {
    runningMaxTile[q] = -1e30;
    runningSumTile[q] = 0.0;
  }
  barrier();

  // Each workgroup computes one MWG x NWG QK tile at a time. The cooperative
  // matrix fragment ownership follows the same subgroup tiling as HGEMM:
  // subgroups are arranged over the MWAVE x NWAVE wave tile and reused over
  // the full MWG x NWG workgroup tile.
  const int subgroupCountM = MWAVE / MSize;
  const int subgroupM = subgroupId % subgroupCountM;
  const int subgroupN = subgroupId / subgroupCountM;

  for(int kvStart = 0; kvStart < seqLen; kvStart += NWG) {
    for(int i = tid; i < KPAD * NWG; i += numThreads) {
      const int d = i / NWG;
      const int k = i - d * NWG;
      const int globalK = kvStart + k;
      if(d < ATTN_HEAD_DIM && globalK < seqLen) {
        kTile[i] = float16_t(LOAD(K, kBatchBase + kOffset +
          (kvh * ATTN_HEAD_DIM + d) * seqLen + globalK));
      } else {
        kTile[i] = float16_t(0.0);
      }
    }
    for(int k = tid; k < NWG; k += numThreads) {
      const int globalK = kvStart + k;
      kMaskTile[k] = globalK < seqLen ? float(LOAD(mask, n * seqLen + globalK)) : 0.0;
    }
    barrier();

    coopmat<float16_t, gl_ScopeSubgroup, MSize, KSize, gl_MatrixUseA> qFrag[MWI];
    coopmat<float16_t, gl_ScopeSubgroup, KSize, NSize, gl_MatrixUseB> kFrag;
    coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> scoreFrag[NWI][MWI];

    for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
      for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
        scoreFrag[bWaveId][aWaveId] =
          coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(acc_dtype(0.0));
      }
    }

    for(int kBase = 0; kBase < KPAD; kBase += KWG) {
      for(int kOffset = 0; kOffset < KWG; kOffset += KSize) {
        for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
          const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
          coopMatLoad(
            qFrag[aWaveId], qTile,
            (kBase + kOffset) * MWG + aLocalOffset,
            MWG,
            gl_CooperativeMatrixLayoutColumnMajor
          );
        }

        for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
          const int bLocalOffset = bWaveId * NWAVE + subgroupN * NSize;
          coopMatLoad(
            kFrag, kTile,
            (kBase + kOffset) * NWG + bLocalOffset,
            NWG,
            gl_CooperativeMatrixLayoutRowMajor
          );
          for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
            scoreFrag[bWaveId][aWaveId] =
              coopMatMulAdd(qFrag[aWaveId], kFrag, scoreFrag[bWaveId][aWaveId]);
          }
        }
      }
    }

    for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
      const int bLocalOffset = bWaveId * NWAVE + subgroupN * NSize;
      for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
        const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
        coopMatStore(
          scoreFrag[bWaveId][aWaveId], scoreTile,
          bLocalOffset * MWG + aLocalOffset,
          MWG,
          gl_CooperativeMatrixLayoutColumnMajor
        );
      }
    }
    barrier();

    // One subgroup owns a sequence of query rows. Its lanes walk the key
    // dimension, so subgroup max/add replace a workgroup shared reduction for
    // every row. The probability tile is kept unnormalized; this lets the PV
    // cooperative matrix produce the tile accumulator used by online softmax.
    for(int qLocal = subgroupId; qLocal < MWG; qLocal += numSubgroups) {
      const bool queryValid = qMaskTile[qLocal] != 0.0;
      float localMax = -1e30;
      for(int k = subgroupLane; k < NWG; k += subgroupSize) {
        const int globalK = kvStart + k;
        float score = -1e30;
        if(queryValid && globalK < seqLen && kMaskTile[k] != 0.0) {
          score = float(scoreTile[k * MWG + qLocal]) * scale;
        }
        localMax = max(localMax, score);
      }
      const float tileMax = subgroupMax(localMax);
      const bool tileValid = tileMax > -1e20;
      float localSum = 0.0;
      for(int k = subgroupLane; k < NWG; k += subgroupSize) {
        const int globalK = kvStart + k;
        float probability = 0.0;
        if(tileValid && queryValid && globalK < seqLen && kMaskTile[k] != 0.0) {
          probability = exp(float(scoreTile[k * MWG + qLocal]) * scale - tileMax);
        }
        probabilityTile[k * MWG + qLocal] = float16_t(probability);
        localSum += probability;
      }
      const float tileSum = subgroupAdd(localSum);
      if(subgroupLane == 0) {
        tileMaxTile[qLocal] = tileMax;
        tileSumTile[qLocal] = tileSum;
      }
    }
    barrier();

    // Apply the probability tile to V. V is processed in NWG-wide value
    // tiles, so vHeadDim does not need to be a multiple of the cooperative N
    // tile; the final tile is zero padded in shared memory.
    for(int vStart = 0; vStart < ATTN_V_HEAD_DIM; vStart += NWG) {
      for(int i = tid; i < NWG * NWG; i += numThreads) {
        const int vLocal = i / NWG;
        const int k = i - vLocal * NWG;
        const int globalK = kvStart + k;
        const int v = vStart + vLocal;
        if(v < ATTN_V_HEAD_DIM && globalK < seqLen) {
          vTile[i] = float16_t(LOAD(V, vBatchBase + vOffset +
            (kvh * ATTN_V_HEAD_DIM + v) * seqLen + globalK));
        } else {
          vTile[i] = float16_t(0.0);
        }
      }
      barrier();

      coopmat<float16_t, gl_ScopeSubgroup, MSize, KSize, gl_MatrixUseA> probabilityFrag[MWI];
      coopmat<float16_t, gl_ScopeSubgroup, KSize, NSize, gl_MatrixUseB> valueFrag;
      coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> outputFrag[NWI][MWI];

      for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
        for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
          outputFrag[bWaveId][aWaveId] =
            coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(acc_dtype(0.0));
        }
      }

      for(int pKOffset = 0; pKOffset < NWG; pKOffset += KSize) {
        for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
          const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
          coopMatLoad(
            probabilityFrag[aWaveId], probabilityTile,
            pKOffset * MWG + aLocalOffset,
            MWG,
            gl_CooperativeMatrixLayoutColumnMajor
          );
        }

        for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
          const int bLocalOffset = bWaveId * NWAVE + subgroupN * NSize;
          coopMatLoad(
            valueFrag, vTile,
            pKOffset + bLocalOffset * NWG,
            NWG,
            gl_CooperativeMatrixLayoutColumnMajor
          );
          for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
            outputFrag[bWaveId][aWaveId] =
              coopMatMulAdd(probabilityFrag[aWaveId], valueFrag, outputFrag[bWaveId][aWaveId]);
          }
        }
      }

      for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
        const int bLocalOffset = bWaveId * NWAVE + subgroupN * NSize;
        for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
          const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
          coopMatStore(
            outputFrag[bWaveId][aWaveId], tileOutput,
            bLocalOffset * MWG + aLocalOffset,
            MWG,
            gl_CooperativeMatrixLayoutColumnMajor
          );
        }
      }
      barrier();

      // Only lane zero updates the online state. The cooperative matrix has
      // already performed the expensive QK/PV work; the update is scalar and
      // is shared by the subgroup's lanes on the next tile.
      for(int qLocal = subgroupId; qLocal < MWG; qLocal += numSubgroups) {
        if(subgroupLane == 0) {
          const float oldMax = runningMaxTile[qLocal];
          const float tileMax = tileMaxTile[qLocal];
          const float tileSum = tileSumTile[qLocal];
          const float newMax = max(oldMax, tileMax);
          const float oldWeight = exp(oldMax - newMax);
          const float tileWeight = tileSum > 0.0 ? exp(tileMax - newMax) : 0.0;
          const int vEnd = min(ATTN_V_HEAD_DIM, vStart + NWG);
          for(int v = vStart; v < vEnd; v++) {
            const float tileValue = float(tileOutput[(v - vStart) * MWG + qLocal]);
            outputAccumulator[v * MWG + qLocal] =
              outputAccumulator[v * MWG + qLocal] * oldWeight + tileValue * tileWeight;
          }
        }
      }
      barrier();
    }

    for(int qLocal = subgroupId; qLocal < MWG; qLocal += numSubgroups) {
      if(subgroupLane == 0) {
        const float oldMax = runningMaxTile[qLocal];
        const float tileMax = tileMaxTile[qLocal];
        const float tileSum = tileSumTile[qLocal];
        const float newMax = max(oldMax, tileMax);
        const float oldWeight = exp(oldMax - newMax);
        const float tileWeight = tileSum > 0.0 ? exp(tileMax - newMax) : 0.0;
        runningMaxTile[qLocal] = newMax;
        runningSumTile[qLocal] = runningSumTile[qLocal] * oldWeight + tileSum * tileWeight;
      }
    }
    barrier();
  }

  // Each lane writes a disjoint subset of the value dimension for its query
  // row. This avoids serializing the final output stores through lane zero.
  for(int qLocal = subgroupId; qLocal < MWG; qLocal += numSubgroups) {
    const int globalQ = qBlockStart + qLocal;
    if(globalQ < seqLen) {
      const float invSum = runningSumTile[qLocal] > 0.0
        ? 1.0 / runningSumTile[qLocal] : 0.0;
      for(int v = subgroupLane; v < ATTN_V_HEAD_DIM; v += subgroupSize) {
        const float result = outputAccumulator[v * MWG + qLocal] * invSum;
        STORE(d_output, (bh * ATTN_V_HEAD_DIM + v) * seqLen + globalQ, floatToReal(result));
      }
    }
  }
}
