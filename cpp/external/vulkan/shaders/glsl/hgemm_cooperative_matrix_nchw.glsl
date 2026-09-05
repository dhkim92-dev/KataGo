#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

#define COMPONENT_TYPE_FLOAT16 0
#define COMPONENT_TYPE_FLOAT32 1

// Match the OpenCL NCHW kernel's fixed vectorization configuration. The
// storage buffers and the SB1 shared tile use four-half vectors. Cooperative
// matrix load/store offsets and strides are therefore expressed in f16vec4
// elements, just as OpenCL's vload4/vstore4 paths use vector indices.
#ifndef VWM
#define VWM 4
#endif
#ifndef VWN
#define VWN 4
#endif
#if VWM != 4 || VWN != 4
#error "hgemm_cooperative_matrix_nchw requires VWM=4 and VWN=4"
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
// Cooperative-matrix accumulator and result component types.  The valid
// floating-point combinations are FP16/FP16 and FP32/FP32; A and B remain
// FP16 for this hgemm kernel.
layout(constant_id = 11) const int CType = COMPONENT_TYPE_FLOAT16;
layout(constant_id = 12) const int ResultType = COMPONENT_TYPE_FLOAT16;



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
  f16vec4 d_input[];
};

layout(set = 0, binding = 1) readonly buffer Filter {
  f16vec4 d_filter[];
};

layout(set = 0, binding = 2) writeonly buffer Output {
  f16vec4 d_output[];
};

layout(push_constant) uniform HGemmCooperativeMatrixNCHWParams {
  int cSize;
  int hwSize;
  int ocSize;
};

#if SB == 1
shared f16vec4 bTile[(KWG * NWG) / VWN];
#endif

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

  // Vulkan subgroup size is device-dependent. The host must specialize
  // local_size_x to (MWAVE / MSize) * gl_SubgroupSize for this mapping.
  const int subgroupM = LocalId0() / int(gl_SubgroupSize);
  const int subgroupN = LocalId1();

  // Every workgroup computes one MWG x NWG tile. Only a MWAVE x NWAVE
  // collection of fragments is resident at once; aWaveId/bWaveId reuse it
  // over the remainder of the workgroup tile.
  // A and B are always FP16.  Keep both accumulator variants in the module;
  // the specialization constants below select the matching operation and
  // store path when the pipeline is created.
  coopmat<float16_t, gl_ScopeSubgroup, MSize, KDIM, gl_MatrixUseA> matA[MWI];
  coopmat<float16_t, gl_ScopeSubgroup, KDIM, NSize, gl_MatrixUseB> matB;
  coopmat<float16_t, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> accFP16[NWI][MWI];
  coopmat<float, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> accFP32[NWI][MWI];

  const bool useFP16Accumulator =
    CType == COMPONENT_TYPE_FLOAT16 && ResultType == COMPONENT_TYPE_FLOAT16;
  const bool useFP32Accumulator =
    CType == COMPONENT_TYPE_FLOAT32 && ResultType == COMPONENT_TYPE_FLOAT32;

  if(!useFP16Accumulator && !useFP32Accumulator)
    return;

  for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
    for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
      accFP16[bWaveId][aWaveId] = coopmat<float16_t, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(0.0hf);
      accFP32[bWaveId][aWaveId] = coopmat<float, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(0.0f);
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
            aGlobalOffset / VWM,
            hwSize / VWM,
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
          bGlobalOffset / VWN,
          ocSize / VWN,
          gl_CooperativeMatrixLayoutRowMajor
        );
#endif

        for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
          const int aLocalOffset = aWaveId * MWAVE + subgroupM * MSize;
          if(groupMBase + aLocalOffset < hwSize) {
            if(useFP16Accumulator)
              accFP16[bWaveId][aWaveId] = coopMatMulAdd(matA[aWaveId], matB, accFP16[bWaveId][aWaveId]);
            else
              accFP32[bWaveId][aWaveId] = coopMatMulAdd(matA[aWaveId], matB, accFP32[bWaveId][aWaveId]);
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
      const int cGlobalOffset =
        batchOutputBase + (groupNBase + bLocalOffset) * hwSize + groupMBase + aLocalOffset;

      if(groupMBase + aLocalOffset < hwSize) {
        if(useFP16Accumulator) {
          coopMatStore(
            accFP16[bWaveId][aWaveId], d_output,
            cGlobalOffset / VWM,
            hwSize / VWM,
            gl_CooperativeMatrixLayoutColumnMajor
          );
        } else {
          coopMatStore(
            accFP32[bWaveId][aWaveId], d_output,
            cGlobalOffset / VWM,
            hwSize / VWM,
            gl_CooperativeMatrixLayoutColumnMajor
          );
        }
      }
    }
  }
}
