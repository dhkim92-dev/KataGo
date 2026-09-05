#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

// Vulkan cooperative-matrix port of the OpenCL WMMA kernel. Matrices use the same layout as
// the OpenCL kernel: A[k*M+m] and C[n*M+m] are column-major, while B[k*N+n] is
// row-major.  The four-lane vectorization is fixed by the port.
#define VWM 4
#define VWN 4

#ifndef SA
#define SA 0
#endif
#ifndef SB
#define SB 0
#endif

// IDs 0, 1 and 2 are reserved for local_size_*_id.  Keep the remaining
// fields in this order: the host fills them through specialization info.
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
  f16vec4 agm[];
};

layout(set = 0, binding = 1) readonly buffer MatB {
  f16vec4 bgm[];
};

layout(set = 0, binding = 2) writeonly buffer MatC {
  f16vec4 cgm[];
};

layout(push_constant) uniform HGemmCooperativeMatrixParams {
  int kSizeM;
  int kSizeN;
  int kSizeK;
};

#if SA == 1
shared f16vec4 alm[(MWG * KWG) / VWM];
#endif
#if SB == 1
shared f16vec4 blm[(NWG * KWG) / VWN];
#endif

void loadSharedTiles(int kwg, int baseA, int baseB) {
  const int tid = int(gl_LocalInvocationIndex);
  const int numThreads = int(gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z);

#if SA == 1
  const int aVectorCount = (MWG * KWG) / VWM;
  for(int i = tid; i < aVectorCount; i += numThreads) {
    const int m = i % (MWG / VWM);
    const int k = i / (MWG / VWM);
    alm[i] = agm[(baseA + (kwg + k) * kSizeM + m * VWM) / VWM];
  }
#endif
#if SB == 1
  const int bVectorCount = (NWG * KWG) / VWN;
  for(int i = tid; i < bVectorCount; i += numThreads) {
    const int n = i % (NWG / VWN);
    const int k = i / (NWG / VWN);
    blm[i] = bgm[(baseB + (kwg + k) * kSizeN + n * VWN) / VWN];
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
  const int subgroupM = int(gl_LocalInvocationID.x) / int(gl_SubgroupSize);
  const int subgroupN = int(gl_LocalInvocationID.y);
  const int groupMBase = groupM * MWG;
  const int groupNBase = groupN * NWG;
  const int baseA = batch * kSizeM * kSizeK;
  const int baseB = batch * kSizeN * kSizeK;
  const int baseC = batch * kSizeM * kSizeN;

  coopmat<float16_t, gl_ScopeSubgroup, MSize, KSize, gl_MatrixUseA> aFrag;
  coopmat<float16_t, gl_ScopeSubgroup, KSize, NSize, gl_MatrixUseB> bFrag;
  coopmat<float16_t, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> cFrag[NWI][MWI];

  for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
    for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
      cFrag[bWaveId][aWaveId] =
        coopmat<float16_t, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(0.0hf);
    }
  }

  for(int kwg = 0; kwg < kSizeK; kwg += KWG) {
    loadSharedTiles(kwg, baseA, baseB);

    for(int kOffset = 0; kOffset < KWG; kOffset += KSize) {
      for(int aWaveId = 0; aWaveId < MWI; aWaveId++) {
        const int aOffset = aWaveId * MWAVE + subgroupM * MSize;
#if SA == 1
        const int aSharedIndex = (kOffset * MWG + aOffset) / VWM;
        coopMatLoad(
          aFrag, alm, aSharedIndex, MWG / VWM,
          gl_CooperativeMatrixLayoutColumnMajor
        );
#else
        coopMatLoad(
          aFrag, agm,
          (baseA + (kwg + kOffset) * kSizeM + groupMBase + aOffset) / VWM,
          kSizeM / VWM,
          gl_CooperativeMatrixLayoutColumnMajor
        );
#endif

        for(int bWaveId = 0; bWaveId < NWI; bWaveId++) {
          const int bOffset = bWaveId * NWAVE + subgroupN * NSize;
#if SB == 1
          const int bSharedIndex = (kOffset * NWG + bOffset) / VWN;
          coopMatLoad(
            bFrag, blm, bSharedIndex, NWG / VWN,
            gl_CooperativeMatrixLayoutRowMajor
          );
#else
          const int bIndex = groupNBase + bOffset;
          coopMatLoad(
            bFrag, bgm,
            (baseB + (kwg + kOffset) * kSizeN + bIndex) / VWN,
            kSizeN / VWN,
            gl_CooperativeMatrixLayoutRowMajor
          );
#endif
          cFrag[bWaveId][aWaveId] =
            coopMatMulAdd(aFrag, bFrag, cFrag[bWaveId][aWaveId]);
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
      const int cIndex = baseC + (groupNBase + bOffset) * kSizeM + groupMBase + aOffset;
      coopMatStore(
        cFrag[bWaveId][aWaveId], cgm, cIndex / VWM, kSizeM / VWM,
        gl_CooperativeMatrixLayoutColumnMajor
      );
    }
  }
}
