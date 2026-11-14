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
#else
#define acc_dtype float
#endif
#ifndef SB
#define SB 1
#endif

layout(constant_id = 3) const int MSize = 16;
layout(constant_id = 4) const int NSize = 8;
layout(constant_id = 5) const int KSize = 16;
layout(constant_id = 6) const int MWG = 128;
layout(constant_id = 7) const int NWG = 64;
layout(constant_id = 8) const int KWG = 32;
layout(constant_id = 9) const int MWAVE = 32;
layout(constant_id = 10) const int NWAVE = 16;
layout(constant_id = 11) const int STAGES = 1;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define MWI (MWG / MWAVE)
#define NWI (NWG / NWAVE)
#define THREADS int(gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z)
#define A_COUNT (MWG * KWG / 4)
#define B_COUNT (NWG * KWG / 4)
#define A_PREFETCH ((A_COUNT + THREADS - 1) / THREADS)
#define B_PREFETCH ((B_COUNT + THREADS - 1) / THREADS)

// Activations remain NHWC. Fused-only weights are [main F,K][gate F,K],
// transposed once during model initialization; each vector follows K.
layout(set = 0, binding = 0) readonly buffer Input { f16vec4 d_input[]; };
layout(set = 0, binding = 1) readonly buffer PackedFilter { f16vec4 d_filter[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float16_t d_output[]; };
layout(push_constant) uniform TransformerDualGemmSwiGLUParams {
  int cSize;
  int hwSize;
  int packedOCSize;
  int ffnSize;
  int inputChannelStride;
  int outputChannelStride;
};

struct ATileStorage { uvec4 alignment; f16vec4 values[STAGES * A_COUNT]; };
shared ATileStorage aTile;
#if SB == 1
struct BTileStorage { uvec4 alignment; f16vec4 values[STAGES * B_COUNT]; };
shared BTileStorage mainBTile;
shared BTileStorage gateBTile;
#endif

f16vec4 readA(int index, int kBase, int mBase, int batchBase) {
  int row = index / (KWG / 4);
  int k = index % (KWG / 4);
  return mBase + row < hwSize
    ? d_input[(batchBase + (mBase + row) * inputChannelStride + kBase) / 4 + k]
    : f16vec4(0.0);
}
f16vec4 readB(int index, int kBase, int nBase, int gate) {
  int n = index / (KWG / 4);
  int k = index % (KWG / 4);
  return d_filter[((gate * ffnSize + nBase + n) * cSize + kBase) / 4 + k];
}
void stageTile(int kBase, int stage, int mBase, int nBase, int batchBase) {
  int tid = int(gl_LocalInvocationIndex);
  for(int i = tid; i < A_COUNT; i += THREADS)
    aTile.values[stage * A_COUNT + i] = readA(i, kBase, mBase, batchBase);
#if SB == 1
  for(int i = tid; i < B_COUNT; i += THREADS) {
    mainBTile.values[stage * B_COUNT + i] = readB(i, kBase, nBase, 0);
    gateBTile.values[stage * B_COUNT + i] = readB(i, kBase, nBase, 1);
  }
#endif
}

void main() {
  int mBase = int(gl_WorkGroupID.x) * MWG;
  int nBase = int(gl_WorkGroupID.y) * NWG;
  int inputBase = int(gl_WorkGroupID.z) * hwSize * inputChannelStride;
  int outputBase = int(gl_WorkGroupID.z) * hwSize * outputChannelStride;
  int subgroupM = int(gl_SubgroupID) % (MWAVE / MSize);
  int subgroupN = int(gl_SubgroupID) / (MWAVE / MSize);
  int tid = int(gl_LocalInvocationIndex);

  coopmat<float16_t, gl_ScopeSubgroup, MSize, KSize, gl_MatrixUseA> matA[MWI];
  coopmat<float16_t, gl_ScopeSubgroup, KSize, NSize, gl_MatrixUseB> matB;
  coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> mainAcc[NWI][MWI];
  coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> gateAcc[NWI][MWI];
  for(int n = 0; n < NWI; n++) for(int m = 0; m < MWI; m++) {
    mainAcc[n][m] = coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(acc_dtype(0));
    gateAcc[n][m] = coopmat<acc_dtype, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(acc_dtype(0));
  }

  f16vec4 nextA[A_PREFETCH];
#if SB == 1
  f16vec4 nextMain[B_PREFETCH];
  f16vec4 nextGate[B_PREFETCH];
#endif
  if(STAGES == 2) {
    stageTile(0, 0, mBase, nBase, inputBase);
    barrier();
  }
  for(int kBase = 0; kBase < cSize; kBase += KWG) {
    int stage = STAGES == 2 ? (kBase / KWG) & 1 : 0;
    bool hasNext = kBase + KWG < cSize;
    if(STAGES == 1) {
      stageTile(kBase, 0, mBase, nBase, inputBase);
      barrier();
    }
    else if(hasNext) {
      // Loads are independent of the current MMA. Registers retain the next
      // tile until its alternate shared buffer is published below.
      for(int j = 0; j < A_PREFETCH; j++) {
        int i = tid + j * THREADS;
        if(i < A_COUNT) nextA[j] = readA(i, kBase + KWG, mBase, inputBase);
      }
#if SB == 1
      for(int j = 0; j < B_PREFETCH; j++) {
        int i = tid + j * THREADS;
        if(i < B_COUNT) {
          nextMain[j] = readB(i, kBase + KWG, nBase, 0);
          nextGate[j] = readB(i, kBase + KWG, nBase, 1);
        }
      }
#endif
    }
    for(int kOffset = 0; kOffset < KWG; kOffset += KSize) {
      for(int m = 0; m < MWI; m++)
        coopMatLoad(matA[m], aTile.values,
          stage * A_COUNT + ((m * MWAVE + subgroupM * MSize) * KWG + kOffset) / 4,
          KWG / 4, gl_CooperativeMatrixLayoutRowMajor);
      for(int n = 0; n < NWI; n++) {
        int column = n * NWAVE + subgroupN * NSize;
#if SB == 1
        coopMatLoad(matB, mainBTile.values, stage * B_COUNT + (column * KWG + kOffset) / 4,
          KWG / 4, gl_CooperativeMatrixLayoutColumnMajor);
#else
        coopMatLoad(matB, d_filter, ((nBase + column) * cSize + kBase + kOffset) / 4,
          cSize / 4, gl_CooperativeMatrixLayoutColumnMajor);
#endif
        for(int m = 0; m < MWI; m++) mainAcc[n][m] = coopMatMulAdd(matA[m], matB, mainAcc[n][m]);
#if SB == 1
        coopMatLoad(matB, gateBTile.values, stage * B_COUNT + (column * KWG + kOffset) / 4,
          KWG / 4, gl_CooperativeMatrixLayoutColumnMajor);
#else
        coopMatLoad(matB, d_filter, ((ffnSize + nBase + column) * cSize + kBase + kOffset) / 4,
          cSize / 4, gl_CooperativeMatrixLayoutColumnMajor);
#endif
        for(int m = 0; m < MWI; m++) gateAcc[n][m] = coopMatMulAdd(matA[m], matB, gateAcc[n][m]);
      }
    }
    if(STAGES == 1) barrier();
    else if(hasNext) {
      int nextStage = stage ^ 1;
      for(int j = 0; j < A_PREFETCH; j++) {
        int i = tid + j * THREADS;
        if(i < A_COUNT) aTile.values[nextStage * A_COUNT + i] = nextA[j];
      }
#if SB == 1
      for(int j = 0; j < B_PREFETCH; j++) {
        int i = tid + j * THREADS;
        if(i < B_COUNT) {
          mainBTile.values[nextStage * B_COUNT + i] = nextMain[j];
          gateBTile.values[nextStage * B_COUNT + i] = nextGate[j];
        }
      }
#endif
      barrier();
    }
  }

  for(int n = 0; n < NWI; n++) for(int m = 0; m < MWI; m++) {
    // Preserve the fallback GEMM's FP16 rounding before FP32 SwiGLU.
    for(int i = 0; i < mainAcc[n][m].length(); i++) {
      float a = float(float16_t(mainAcc[n][m][i]));
      float b = float(float16_t(gateAcc[n][m][i]));
      mainAcc[n][m][i] = acc_dtype(float16_t((a / (1.0 + exp(-a))) * b));
    }
    coopmat<float16_t, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator> result =
      coopmat<float16_t, gl_ScopeSubgroup, MSize, NSize, gl_MatrixUseAccumulator>(mainAcc[n][m]);
    coopMatStore(result, d_output,
      outputBase + (mBase + m * MWAVE + subgroupM * MSize) * outputChannelStride + nBase + n * NWAVE + subgroupN * NSize,
      outputChannelStride, gl_CooperativeMatrixLayoutRowMajor);
  }
}
