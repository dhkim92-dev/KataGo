#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

#ifndef ACC_TYPE
#define ACC_TYPE 16
#endif
#ifndef SA
#define SA 0
#endif
#ifndef SB
#define SB 0
#endif
#ifndef VWK
#define VWK 1
#endif
#ifndef VWN
#define VWN 1
#endif

#if ACC_TYPE == 16
#define acc_dtype float16_t
#elif ACC_TYPE == 32
#define acc_dtype float
#else
#error "ACC_TYPE must be 16 or 32"
#endif

#if VWK == 1
#define realstoreK float16_t
#elif VWK == 2
#define realstoreK f16vec2
#elif VWK == 4
#define realstoreK f16vec4
#else
#error "VWK must be 1, 2, or 4"
#endif

#if VWN == 1
#define realstoreN float16_t
#elif VWN == 2
#define realstoreN f16vec2
#elif VWN == 4
#define realstoreN f16vec4
#else
#error "VWN must be 1, 2, or 4"
#endif

layout(constant_id = 3) const int MSize = 16;
layout(constant_id = 4) const int NSize = 16;
layout(constant_id = 5) const int KSize = 16;
layout(constant_id = 6) const int MWG = 32;
layout(constant_id = 7) const int NWG = 32;
layout(constant_id = 8) const int KWG = 32;
layout(constant_id = 9) const int MWAVE = 32;
layout(constant_id = 10) const int NWAVE = 32;

#define MWI (MWG / MWAVE)
#define NWI (NWG / NWAVE)

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly buffer MatA {
#if SA == 1
  realstoreK aData[];
#else
  float16_t aData[];
#endif
};
layout(set = 0, binding = 1) readonly buffer MatB {
#if SB == 1
  realstoreN bData[];
#else
  float16_t bData[];
#endif
};
layout(set = 0, binding = 2) writeonly buffer MatC {
#if ACC_TYPE == 32
  realstoreN cData[];
#else
  float16_t cData[];
#endif
};

layout(push_constant) uniform HGemmCooperativeMatrixNHWCParams {
  int mSize;
  int nSize;
  int kSize;
  int aRowStride;
  int cRowStride;
};

#if SA == 1
struct ATileStorage {
  uvec4 alignment;
  realstoreK values[(MWG * KWG) / VWK];
};
shared ATileStorage aTileStorage;
#define aTile aTileStorage.values
#endif

#if SB == 1
struct BTileStorage {
  uvec4 alignment;
  realstoreN values[(KWG * NWG) / VWN];
};
shared BTileStorage bTileStorage;
#define bTile bTileStorage.values
#endif

#if ACC_TYPE == 32
struct CTileStorage {
  uvec4 alignment;
  float values[MWG * NWG];
};
shared CTileStorage cTileStorage;
#define cTile cTileStorage.values
#endif

void loadSharedTiles(int kwg, int baseA, int baseB, int groupMBase, int groupNBase) {
  const int tid = int(gl_LocalInvocationIndex);
  const int numThreads = int(gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z);
#if SA == 1
  const int aVectorCount = (MWG * KWG) / VWK;
  const int aVectorColumns = KWG / VWK;
  for(int i = tid; i < aVectorCount; i += numThreads) {
    const int m = i / aVectorColumns;
    const int k = i - m * aVectorColumns;
    aTile[i] = aData[(baseA + (groupMBase + m) * aRowStride + kwg + k * VWK) / VWK];
  }
#endif
#if SB == 1
  const int bVectorCount = (KWG * NWG) / VWN;
  const int bVectorColumns = NWG / VWN;
  for(int i = tid; i < bVectorCount; i += numThreads) {
    const int k = i / bVectorColumns;
    const int n = i - k * bVectorColumns;
    bTile[i] = bData[(baseB + (kwg + k) * nSize + groupNBase + n * VWN) / VWN];
  }
#endif
#if SA == 1 || SB == 1
  barrier();
#endif
}

void main() {
  const int groupM = int(gl_WorkGroupID.x);
  const int groupN = int(gl_WorkGroupID.y);
  const int batch = int(gl_WorkGroupID.z);
  const int subgroupCountM = MWAVE / MSize;
  const int subgroupLinear = int(gl_SubgroupID);
  const int subgroupM = subgroupLinear % subgroupCountM;
  const int subgroupN = subgroupLinear / subgroupCountM;
  const int groupMBase = groupM * MWG;
  const int groupNBase = groupN * NWG;
  const int baseA = batch * mSize * aRowStride;
  // The NHWC Conv filter is one shared [K,N] matrix for every batch item.
  // Only A and C carry the dispatch-z batch stride.
  const int baseB = 0;
  const int baseC = batch * mSize * cRowStride;

  coopmat<float16_t, gl_ScopeSubgroup, MSize, KSize, gl_MatrixUseA> aFrag[MWI];
  coopmat<float16_t, gl_ScopeSubgroup, KSize, NSize, gl_MatrixUseB> bFrag;
  coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> cFrag[NWI][MWI];

  for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
    for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
      cFrag[bWaveId][aWaveId] =
        coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(acc_dtype(0.0));
    }
  }

  for(int kwg = 0; kwg < kSize; kwg += KWG) {
    loadSharedTiles(kwg, baseA, baseB, groupMBase, groupNBase);
    for(int kOffset = 0; kOffset < KWG; kOffset += KSize) {
      for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
        const int aOffset = aWaveId * MWAVE + subgroupM * MSize;
#if SA == 1
        coopMatLoad(
          aFrag[aWaveId], aTile,
          (aOffset * KWG + kOffset) / VWK,
          KWG / VWK,
          gl_CooperativeMatrixLayoutRowMajor
        );
#else
        coopMatLoad(
          aFrag[aWaveId], aData,
          baseA + (groupMBase + aOffset) * aRowStride + kwg + kOffset,
          aRowStride,
          gl_CooperativeMatrixLayoutRowMajor
        );
#endif
      }

      for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
        const int bOffset = bWaveId * NWAVE + subgroupN * NSize;
#if SB == 1
        coopMatLoad(
          bFrag, bTile,
          (kOffset * NWG + bOffset) / VWN,
          NWG / VWN,
          gl_CooperativeMatrixLayoutRowMajor
        );
#else
        coopMatLoad(
          bFrag, bData,
          (kwg + kOffset) * nSize + groupNBase + bOffset,
          nSize,
          gl_CooperativeMatrixLayoutRowMajor
        );
#endif
        for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
          cFrag[bWaveId][aWaveId] = coopMatMulAdd(aFrag[aWaveId], bFrag, cFrag[bWaveId][aWaveId]);
        }
      }
    }
#if SA == 1 || SB == 1
    barrier();
#endif
  }

  for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
    const int bOffset = bWaveId * NWAVE + subgroupN * NSize;
    for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
      const int aOffset = aWaveId * MWAVE + subgroupM * MSize;
#if ACC_TYPE == 16
      coopMatStore(
        cFrag[bWaveId][aWaveId], cData,
         baseC + (groupMBase + aOffset) * cRowStride + groupNBase + bOffset, cRowStride,
        gl_CooperativeMatrixLayoutRowMajor
      );
#else
      coopMatStore(
        cFrag[bWaveId][aWaveId], cTile,
        aOffset * NWG + bOffset, NWG,
        gl_CooperativeMatrixLayoutRowMajor
      );
#endif
    }
  }

#if ACC_TYPE == 32
  barrier();
  const int tid = int(gl_LocalInvocationIndex);
  const int numThreads = int(gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z);
  const int tileVectorCount = (MWG * NWG) / VWN;
  const int tileVectorColumns = NWG / VWN;
  for(int tileVector = tid; tileVector < tileVectorCount; tileVector += numThreads) {
    const int m = tileVector / tileVectorColumns;
    const int n = (tileVector - m * tileVectorColumns) * VWN;
    const int cIndex = (baseC + (groupMBase + m) * cRowStride + groupNBase + n) / VWN;
#if VWN == 1
    cData[cIndex] = float16_t(cTile[m * NWG + n]);
#else
    for(int lane = 0; lane < VWN; lane++)
      cData[cIndex][lane] = float16_t(cTile[m * NWG + n + lane]);
#endif
  }
#endif
}
