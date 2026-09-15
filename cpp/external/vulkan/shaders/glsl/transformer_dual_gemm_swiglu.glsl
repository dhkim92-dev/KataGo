#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

#ifndef ACC_TYPE
#define ACC_TYPE 16
#endif
#if ACC_TYPE == 16
#define acc_dtype float16_t
#elif ACC_TYPE == 32
#define acc_dtype float
#else
#error "ACC_TYPE must be 16 or 32"
#endif

// Only scalar external addressing is supported initially. The matrix fragments
// still use the device-selected cooperative dimensions below.
#ifndef SB
#define SB 0
#endif

#define GroupId0() (int(gl_WorkGroupID.x))
#define GroupId1() (int(gl_WorkGroupID.y))
#define GroupId2() (int(gl_WorkGroupID.z))
#define LocalId0() (int(gl_LocalInvocationID.x))
#define LocalId1() (int(gl_LocalInvocationID.y))
#define LocalId2() (int(gl_LocalInvocationID.z))
#define LocalSize0() (int(gl_WorkGroupSize.x))
#define LocalSize1() (int(gl_WorkGroupSize.y))
#define LocalSize2() (int(gl_WorkGroupSize.z))

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

// A is [K, HW], interpreted as [HW, K] column-major. Each K row of the packed
// filter is [main F, gate F, padding], with packedOCSize as its physical stride.
// C is [F, HW], interpreted as [HW, F] column-major.
layout(set = 0, binding = 0) readonly buffer Input {
  float16_t d_input[];
};

layout(set = 0, binding = 1) readonly buffer PackedFilter {
  float16_t d_filter[];
};

layout(set = 0, binding = 2) writeonly buffer Output {
  float16_t d_output[];
};

layout(push_constant) uniform TransformerDualGemmSwiGLUParams {
  int cSize;
  int hwSize;
  int packedOCSize;
  int ffnSize;
};

#if SB == 1
// Keep the cooperative pointee 16-byte aligned. Both packed B regions are
// staged so each input fragment can be reused by the two matrix products.
struct BTileStorage {
  uvec4 alignment;
  float16_t values[KWG * NWG];
};
shared BTileStorage mainBTileStorage;
shared BTileStorage gateBTileStorage;
#endif

// Cooperative stores use scalar element offsets and strides. The explicit
// scalar epilogue reads the two stored tiles without assuming fragment layout.
struct CTileStorage {
  uvec4 alignment;
  acc_dtype values[MWG * NWG];
};
shared CTileStorage mainCTileStorage;
shared CTileStorage gateCTileStorage;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#if SB == 1
void loadBTiles(int kBase, int groupNBase) {
  const int tid = LocalId0() + LocalSize0() * (LocalId1() + LocalSize1() * LocalId2());
  const int numThreads = LocalSize0() * LocalSize1() * LocalSize2();
  const int tileElementCount = KWG * NWG;
  for(int tileElement = tid; tileElement < tileElementCount; tileElement += numThreads) {
    const int k = tileElement / NWG;
    const int n = tileElement - k * NWG;
    const int filterRow = (kBase + k) * packedOCSize;
    mainBTileStorage.values[tileElement] = d_filter[filterRow + groupNBase + n];
    gateBTileStorage.values[tileElement] = d_filter[filterRow + ffnSize + groupNBase + n];
  }
  barrier();
}
#endif

void main() {
  const int groupM = GroupId0();
  const int groupN = GroupId1();
  const int batch = GroupId2();

  const int subgroupCountM = MWAVE / MSize;
  const int subgroupLinear = int(gl_SubgroupID);
  const int subgroupM = subgroupLinear % subgroupCountM;
  const int subgroupN = subgroupLinear / subgroupCountM;

  coopmat<float16_t, gl_ScopeSubgroup, MSize, KSize, gl_MatrixUseA> matA[MWI];
  coopmat<float16_t, gl_ScopeSubgroup, KSize, NSize, gl_MatrixUseB> matB;
  coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> mainAcc[NWI][MWI];
  coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> gateAcc[NWI][MWI];

  for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
    for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
      mainAcc[bWaveId][aWaveId] = coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(acc_dtype(0.0));
      gateAcc[bWaveId][aWaveId] = coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(acc_dtype(0.0));
    }
  }

  const int groupMBase = groupM * MWG;
  const int groupNBase = groupN * NWG;
  const int batchInputBase = batch * cSize * hwSize;
  const int batchOutputBase = batch * ffnSize * hwSize;

  // As in NCHW HGEMM, K and the selected K tile dimensions are aligned by the
  // host eligibility check. Load each A fragment once and feed both GEMMs.
  for(int kBase = 0; kBase < cSize; kBase += KWG) {
#if SB == 1
    loadBTiles(kBase, groupNBase);
#endif
    for(int kOffset = 0; kOffset < KWG; kOffset += KSize) {
      for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
        const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
        const bool aFragmentInBounds = groupMBase + aLocalOffset < hwSize;
        const int aGlobalOffset =
          batchInputBase + (kBase + kOffset) * hwSize + groupMBase + aLocalOffset;
        if(aFragmentInBounds) {
          coopMatLoad(
            matA[aWaveId], d_input,
            aGlobalOffset,
            hwSize,
            gl_CooperativeMatrixLayoutColumnMajor
          );
        }
      }

      for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
        const int bLocalOffset = bWaveId * NWAVE + subgroupN * NSize;
#if SB == 1
        coopMatLoad(
          matB, mainBTileStorage.values,
          kOffset * NWG + bLocalOffset,
          NWG,
          gl_CooperativeMatrixLayoutRowMajor
        );
#else
        coopMatLoad(
          matB, d_filter,
          (kBase + kOffset) * packedOCSize + groupNBase + bLocalOffset,
          packedOCSize,
          gl_CooperativeMatrixLayoutRowMajor
        );
#endif
        for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
          const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
          if(groupMBase + aLocalOffset < hwSize)
            mainAcc[bWaveId][aWaveId] = coopMatMulAdd(matA[aWaveId], matB, mainAcc[bWaveId][aWaveId]);
        }

#if SB == 1
        coopMatLoad(
          matB, gateBTileStorage.values,
          kOffset * NWG + bLocalOffset,
          NWG,
          gl_CooperativeMatrixLayoutRowMajor
        );
#else
        coopMatLoad(
          matB, d_filter,
          (kBase + kOffset) * packedOCSize + ffnSize + groupNBase + bLocalOffset,
          packedOCSize,
          gl_CooperativeMatrixLayoutRowMajor
        );
#endif
        for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
          const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
          if(groupMBase + aLocalOffset < hwSize)
            gateAcc[bWaveId][aWaveId] = coopMatMulAdd(matA[aWaveId], matB, gateAcc[bWaveId][aWaveId]);
        }
      }
    }
#if SB == 1
    // The next K tile overwrites both shared B tiles. Wait until every
    // subgroup has finished cooperatively loading the current tile.
    barrier();
#endif
  }

  for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
    const int bLocalOffset = bWaveId * NWAVE + subgroupN * NSize;
    for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
      const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
      if(groupMBase + aLocalOffset < hwSize) {
        coopMatStore(
          mainAcc[bWaveId][aWaveId], mainCTileStorage.values,
          bLocalOffset * MWG + aLocalOffset,
          MWG,
          gl_CooperativeMatrixLayoutColumnMajor
        );
        coopMatStore(
          gateAcc[bWaveId][aWaveId], gateCTileStorage.values,
          bLocalOffset * MWG + aLocalOffset,
          MWG,
          gl_CooperativeMatrixLayoutColumnMajor
        );
      }
    }
  }

  barrier();

  const int tid = LocalId0() + LocalSize0() * (LocalId1() + LocalSize1() * LocalId2());
  const int numThreads = LocalSize0() * LocalSize1() * LocalSize2();
  const int tileElementCount = MWG * NWG;
  for(int tileElement = tid; tileElement < tileElementCount; tileElement += numThreads) {
    const int m = tileElement % MWG;
    const int n = tileElement / MWG;
    const int hw = groupMBase + m;
    const int oc = groupNBase + n;
    if(hw < hwSize && oc < ffnSize) {
      const float mainValue = float(mainCTileStorage.values[tileElement]);
      const float gateValue = float(gateCTileStorage.values[tileElement]);
      const float siluMain = mainValue / (1.0f + exp(-mainValue));
      d_output[batchOutputBase + oc * hwSize + hw] = float16_t(siluMain * gateValue);
    }
  }
}
