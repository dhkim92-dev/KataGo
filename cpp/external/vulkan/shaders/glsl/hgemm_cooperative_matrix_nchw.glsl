#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

#ifndef VWM
#define VWM 4
#endif
#ifndef VWN
#define VWN 4
#endif

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
#if VWM == 1
#define realstoreM float16_t
#elif VWM == 2
#define realstoreM f16vec2
#elif VWM == 4
#define realstoreM f16vec4
#else
#error "VWM must be 1, 2, or 4"
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

#define GroupId0() (int(gl_WorkGroupID.x))
#define GroupId1() (int(gl_WorkGroupID.y))
#define GroupId2() (int(gl_WorkGroupID.z))
#define LocalId0() (int(gl_LocalInvocationID.x))
#define LocalId1() (int(gl_LocalInvocationID.y))
#define LocalId2() (int(gl_LocalInvocationID.z))
#define LocalSize0() (int(gl_WorkGroupSize.x))
#define LocalSize1() (int(gl_WorkGroupSize.y))
#define LocalSize2() (int(gl_WorkGroupSize.z))

// Cooperative-matrix dimensions are supplied by the selected device property.
// MSize/NSize/KSize correspond to the M/N/K dimensions of one cooperative
// matrix operation (the OpenCL WMMA fragment was 16x16x16 on NVIDIA).
layout(constant_id = 3) const int MSize = 16;
layout(constant_id = 4) const int NSize = 16;
layout(constant_id = 5) const int KSize = 16;
// Workgroup and wave tiling. The local-size specialization IDs 0, 1, and 2
// are reserved for layout(local_size_*_id=...).
layout(constant_id = 6) const int MWG = 32;
layout(constant_id = 7) const int NWG = 32;
layout(constant_id = 8) const int KWG = 32;
layout(constant_id = 9) const int MWAVE = 32;
layout(constant_id = 10) const int NWAVE = 32;
#ifndef SB
// SB is deliberately a preprocessor option. A shared buffer declaration and
// its barrier/copy path must be selected by GLSLC, not by specialization info.
#define SB 0
#endif

#define MWI (MWG / MWAVE)
#define NWI (NWG / NWAVE)
#define KDIM KSize

// A is [C, HW] in memory. Interpreted as a matrix, it is [HW, C] column-major.
// B is [C, OC] in memory, i.e. [C, OC] row-major.
// C is [OC, HW] in memory, i.e. [HW, OC] column-major.
layout(set = 0, binding = 0) readonly buffer Input {
  float16_t d_input[];
};

layout(set = 0, binding = 1) readonly buffer Filter {
#if SB == 1
  realstoreN d_filter[];
#else
  float16_t d_filter[];
#endif
};

layout(set = 0, binding = 2) writeonly buffer Output {
  float16_t d_output[];
};

layout(push_constant) uniform HGemmCooperativeMatrixNCHWParams {
  int cSize;
  int hwSize;
  int ocSize;
};

#if SB == 1
// The leading uvec4 gives the cooperative-matrix pointee a 16-byte-aligned
// address, matching the explicit alignment on the OpenCL local tile.
struct BTileStorage {
  uvec4 alignment;
  realstoreN values[(KWG * NWG) / VWN];
};
shared BTileStorage bTileStorage;
#define bTile bTileStorage.values
#endif
// Keep the scalar accumulator tile 16-byte aligned for cooperative stores.
struct CTileStorage {
  uvec4 alignment;
  acc_dtype values[MWG * NWG];
};
shared CTileStorage cTileStorage;
#define cTile cTileStorage.values

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void loadBTile(int kBase) {
#if SB == 1
  const int tid = LocalId0() + LocalSize0() * (LocalId1() + LocalSize1() * LocalId2());
  const int numThreads = LocalSize0() * LocalSize1() * LocalSize2();
  const int groupNBase = GroupId1() * NWG;

  const int tileVectorCount = (KWG * NWG) / VWN;
  for(int tileVector = tid; tileVector < tileVectorCount; tileVector += numThreads) {
    const int tileIndex = tileVector * VWN;
    const int k = tileIndex / NWG;
    const int n = tileIndex - k * NWG;
    bTile[tileVector] = d_filter[((kBase + k) * ocSize + groupNBase + n) / VWN];
  }
  barrier();
#else
  // No shared-memory copy is needed in this branch. The B cooperative matrix
  // is loaded directly from d_filter below.
#endif
}

void main() {
  const int groupM = GroupId0();
  const int groupN = GroupId1();
  const int batch = GroupId2();

  // Fragment ownership is derived from the subgroup itself. Vulkan does not
  // define a mapping between LocalInvocationId and SubgroupLocalInvocationId.
  const int subgroupCountM = MWAVE / MSize;
  const int subgroupLinear = int(gl_SubgroupID);
  const int subgroupM = subgroupLinear % subgroupCountM;
  const int subgroupN = subgroupLinear / subgroupCountM;

  // Every workgroup computes one MWG x NWG tile. Only a MWAVE x NWAVE
  // collection of fragments is resident at once; aWaveId/bWaveId reuse it
  // over the remainder of the workgroup tile.
  // A, B, and C use FP16 storage; the accumulator type is selected by ACC_TYPE.
  coopmat<float16_t, gl_ScopeSubgroup, MSize, KDIM, gl_MatrixUseA> matA[MWI];
  coopmat<float16_t, gl_ScopeSubgroup, KDIM, NSize, gl_MatrixUseB> matB;
  coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> acc[NWI][MWI];

  for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
    for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
      acc[bWaveId][aWaveId] = coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(acc_dtype(0.0));
    }
  }

  const int groupMBase = groupM * MWG;
  const int groupNBase = groupN * NWG;
  const int batchInputBase = batch * cSize * hwSize;
  const int batchOutputBase = batch * ocSize * hwSize;

  // The OpenCL source requires cSize to be divisible by KWG and KWG to be
  // divisible by the cooperative-matrix K dimension. The same contract is used here.
  for(int kBase = 0; kBase < cSize; kBase += KWG) {
    loadBTile(kBase);

    for(int kOffset = 0; kOffset < KWG; kOffset += KDIM) {
      // Match OpenCL's register preload: each A fragment is loaded once and
      // reused for every B fragment in this K step.
      for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
        const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
        const bool aFragmentInBounds = groupMBase + aLocalOffset < hwSize;
        const int aGlobalOffset =
          batchInputBase + (kBase + kOffset) * hwSize + groupMBase + aLocalOffset;

        // OpenCL skips fragments that begin outside the padded spatial range.
        // Since hwSize is aligned to MSize, an in-range fragment cannot straddle
        // the end of the input row.
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
          matB, bTile,
          (kOffset * NWG + bLocalOffset) / VWN,
          NWG / VWN,
          gl_CooperativeMatrixLayoutRowMajor
        );
#else
        const int bGlobalOffset =
          (kBase + kOffset) * ocSize + groupNBase + bLocalOffset;
        coopMatLoad(
          matB, d_filter,
          bGlobalOffset,
          ocSize,
          gl_CooperativeMatrixLayoutRowMajor
        );
#endif

        for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
          const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
          if(groupMBase + aLocalOffset < hwSize) {
            acc[bWaveId][aWaveId] = coopMatMulAdd(matA[aWaveId], matB, acc[bWaveId][aWaveId]);
          }
        }
      }
    }

#if SB == 1
    barrier();
#endif
  }

  for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
    const int bLocalOffset = bWaveId * NWAVE + subgroupN * NSize;
    for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
      const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
      if(groupMBase + aLocalOffset < hwSize)
        coopMatStore(
          acc[bWaveId][aWaveId], cTile,
          bLocalOffset * MWG + aLocalOffset,
          MWG,
          gl_CooperativeMatrixLayoutColumnMajor
        );
    }
  }

  barrier();

  // Match OpenCL's LocalToGlobalC{Complete,Edge}: use an unchecked vector
  // copy for complete tiles and only test bounds for the final edge tile.
  const int tid = LocalId0() + LocalSize0() * (LocalId1() + LocalSize1() * LocalId2());
  const int numThreads = LocalSize0() * LocalSize1() * LocalSize2();
  const int tileVectorCount = (MWG * NWG) / VWM;
  if(groupMBase + MWG <= hwSize) {
    for(int tileVector = tid; tileVector < tileVectorCount; tileVector += numThreads) {
      const int tileBase = tileVector * VWM;
      for(int lane = 0; lane < VWM; lane++) {
        const int tileIndex = tileBase + lane;
        const int m = tileIndex % MWG;
        const int n = tileIndex / MWG;
        d_output[batchOutputBase + (groupNBase + n) * hwSize + groupMBase + m] =
          float16_t(cTile[tileIndex]);
      }
    }
  }
  else {
    const int tileSize = MWG * NWG;
    for(int tileIndex = tid; tileIndex < tileSize; tileIndex += numThreads) {
      const int m = tileIndex % MWG;
      const int n = tileIndex / MWG;
      const int hw = groupMBase + m;
      if(hw < hwSize)
        d_output[batchOutputBase + (groupNBase + n) * hwSize + hw] = float16_t(cTile[tileIndex]);
    }
  }
}
