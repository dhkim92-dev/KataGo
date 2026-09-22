#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

#ifndef ACC_BITS
#define ACC_BITS 16
#endif
#ifndef VWK
#define VWK 1
#endif
#ifndef VWN
#define VWN 1
#endif
#ifndef FUSE_BN_ACT
#define FUSE_BN_ACT 0
#endif

#if ACC_BITS == 16
#define acc_type float16_t
#elif ACC_BITS == 32
#define acc_type float
#else
#error "ACC_BITS must be 16 or 32"
#endif

#if VWK == 1
#define input_vec_type float16_t
#elif VWK == 2
#define input_vec_type f16vec2
#elif VWK == 4
#define input_vec_type f16vec4
#else
#error "VWK must be 1, 2, or 4"
#endif

#if VWN == 1
#define output_vec_type float16_t
#define output_float_vec_type float
#elif VWN == 2
#define output_vec_type f16vec2
#define output_float_vec_type vec2
#elif VWN == 4
#define output_vec_type f16vec4
#define output_float_vec_type vec4
#else
#error "VWN must be 1, 2, or 4"
#endif

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int CM = 16;
layout(constant_id = 4) const int CN = 16;
layout(constant_id = 5) const int CK = 16;
layout(constant_id = 6) const int MWG = 32;
layout(constant_id = 7) const int NWG = 32;
layout(constant_id = 8) const int KWG = 32;
layout(constant_id = 9) const int MDIMC = 2;
layout(constant_id = 10) const int NDIMC = 2;
layout(constant_id = 11) const int MDIMA = 8;
layout(constant_id = 12) const int KDIMA = 4;
layout(constant_id = 13) const int KDIMB = 4;
layout(constant_id = 14) const int NDIMB = 8;
layout(constant_id = 15) const int SA = 32;
layout(constant_id = 16) const int SB = 32;
layout(constant_id = 17) const int DOUBLE_BUFFER = 1;
layout(constant_id = 18) const int FILTER_SIZE = 3;
layout(constant_id = 19) const int ACTIVATION = 0;

layout(set = 0, binding = 0) readonly buffer InputBuffer {
  input_vec_type input_data[];
};
layout(set = 0, binding = 1) readonly buffer WeightBuffer {
  output_vec_type weight_data[];
};
layout(set = 0, binding = 2) writeonly buffer OutputBuffer {
#if !FUSE_BN_ACT && ACC_BITS == 16
  float16_t output_data[];
#else
  output_vec_type output_data[];
#endif
};
#if FUSE_BN_ACT
layout(set = 0, binding = 3) readonly buffer ScaleBuffer {
  output_vec_type scale_data[];
};
layout(set = 0, binding = 4) readonly buffer BiasBuffer {
  output_vec_type bias_data[];
};
#endif

layout(push_constant) uniform ImplicitConvPushParams {
  int input_batch_stride;
  int output_batch_stride;
  int width;
  int height;
  int logical_spatial_size;
  int spatial_size;
  int input_channels;
  int input_channel_stride;
  int output_channels;
  int output_channel_stride;
  int logical_k_size;
  int padded_k_size;
};

// DOUBLE_BUFFER selects one reusable tile or two alternating tiles.
shared input_vec_type As[DOUBLE_BUFFER != 0 ? 2 : 1][(MWG * SA) / VWK];
shared output_vec_type Bs[DOUBLE_BUFFER != 0 ? 2 : 1][(KWG * SB) / VWN];
#if FUSE_BN_ACT || ACC_BITS == 32
shared acc_type Cs[MWG * NWG];
#endif

input_vec_type zero_input_vector() {
#if VWK == 1
  return float16_t(0.0);
#elif VWK == 2
  return f16vec2(0.0);
#else
  return f16vec4(0.0);
#endif
}

output_vec_type zero_output_vector() {
#if VWN == 1
  return float16_t(0.0);
#elif VWN == 2
  return f16vec2(0.0);
#else
  return f16vec4(0.0);
#endif
}

input_vec_type load_input_vector(
  int batch,
  int row,
  int kernel_y,
  int kernel_x,
  int channel
) {
  if(row >= logical_spatial_size)
    return zero_input_vector();

  const int output_y = row / width;
  const int output_x = row - output_y * width;
  const int input_y = output_y + kernel_y - FILTER_SIZE / 2;
  const int input_x = output_x + kernel_x - FILTER_SIZE / 2;
  if(input_y < 0 || input_y >= height || input_x < 0 || input_x >= width)
    return zero_input_vector();

  const int scalar_index = batch * input_batch_stride +
    (input_y * width + input_x) * input_channel_stride + channel;
  return input_data[scalar_index / VWK];
}

void stage_tiles(int tile, int kwg, int batch, int group_m, int group_n) {
  const int tid = int(gl_LocalInvocationIndex);
  const int a_vector_columns = KWG / VWK;
  const int a_k_lane = tid % KDIMA;
  const int a_m_lane = tid / KDIMA;
  for(int k_vector = a_k_lane; k_vector < a_vector_columns; k_vector += KDIMA) {
    const int global_k = kwg + k_vector * VWK;
    int kernel_y = 0;
    int kernel_x = 0;
    int channel = 0;
    if(global_k < logical_k_size) {
      const int kernel_index = global_k / input_channels;
      channel = global_k - kernel_index * input_channels;
      kernel_y = kernel_index / FILTER_SIZE;
      kernel_x = kernel_index - kernel_y * FILTER_SIZE;
    }
    for(int m = a_m_lane; m < MWG; m += MDIMA) {
      As[tile][(m * SA) / VWK + k_vector] = global_k < logical_k_size
        ? load_input_vector(batch, group_m + m, kernel_y, kernel_x, channel)
        : zero_input_vector();
    }
  }

  const int b_vector_columns = NWG / VWN;
  const int b_n_lane = tid % NDIMB;
  const int b_k_lane = tid / NDIMB;
  for(int k = b_k_lane; k < KWG; k += KDIMB) {
    for(int n_vector = b_n_lane; n_vector < b_vector_columns; n_vector += NDIMB) {
      const int scalar_index = (kwg + k) * output_channel_stride + group_n + n_vector * VWN;
      Bs[tile][(k * SB) / VWN + n_vector] = weight_data[scalar_index / VWN];
    }
  }
}

output_float_vec_type activate(output_float_vec_type value) {
  if(ACTIVATION == 0)
    return value;
  if(ACTIVATION == 1)
    return max(value, output_float_vec_type(0.0));
  if(ACTIVATION == 2) {
    const output_float_vec_type one = output_float_vec_type(1.0);
    return value / (one + exp(-value));
  }
  if(ACTIVATION == 3) {
    const float k = 0.7978845608028654;
    const output_float_vec_type one = output_float_vec_type(1.0);
    return 0.5 * value * (one + tanh(k * (value + 0.044715 * value * value * value)));
  }
  return value;
}

#if FUSE_BN_ACT || ACC_BITS == 32
void store_output_tile(int batch, int group_m, int group_n) {
  const int tid = int(gl_LocalInvocationIndex);
  const int thread_count = int(gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z);
  const int vector_columns = NWG / VWN;
  const int vector_count = MWG * vector_columns;
  for(int linear = tid; linear < vector_count; linear += thread_count) {
    const int m = linear / vector_columns;
    const int n = (linear - m * vector_columns) * VWN;
    const int row = group_m + m;
    const int channel = group_n + n;
    output_vec_type result = zero_output_vector();
    if(row < logical_spatial_size && channel < output_channels) {
#if VWN == 1
      output_float_vec_type value = float(Cs[m * NWG + n]);
#elif VWN == 2
      output_float_vec_type value = vec2(
        float(Cs[m * NWG + n]),
        float(Cs[m * NWG + n + 1])
      );
#else
      output_float_vec_type value = vec4(
        float(Cs[m * NWG + n]),
        float(Cs[m * NWG + n + 1]),
        float(Cs[m * NWG + n + 2]),
        float(Cs[m * NWG + n + 3])
      );
#endif
#if FUSE_BN_ACT
      const output_float_vec_type scale = output_float_vec_type(scale_data[channel / VWN]);
      const output_float_vec_type bias = output_float_vec_type(bias_data[channel / VWN]);
      value = activate(value * scale + bias);
#endif
      result = output_vec_type(value);
    }
    const int scalar_index = batch * output_batch_stride + row * output_channel_stride + channel;
    output_data[scalar_index / VWN] = result;
  }
}
#endif

void main() {
  const int group_m = int(gl_WorkGroupID.x) * MWG;
  const int group_n = int(gl_WorkGroupID.y) * NWG;
  const int batch = int(gl_WorkGroupID.z);
  const int subgroup = int(gl_SubgroupID);
  const int subgroup_m = subgroup % MDIMC;
  const int subgroup_n = subgroup / MDIMC;

  coopmat<float16_t, gl_ScopeSubgroup, CM, CK, gl_MatrixUseA> a_frag;
  coopmat<float16_t, gl_ScopeSubgroup, CK, CN, gl_MatrixUseB> b_frag;
  coopmat<acc_type, gl_ScopeSubgroup, CM, CN, gl_MatrixUseAccumulator> c_frag =
    coopmat<acc_type, gl_ScopeSubgroup, CM, CN, gl_MatrixUseAccumulator>(acc_type(0.0));

  stage_tiles(0, 0, batch, group_m, group_n);
  barrier();

  int tile = 0;
  for(int kwg = 0; kwg < padded_k_size; kwg += KWG) {
    for(int k_offset = 0; k_offset < KWG; k_offset += CK) {
      coopMatLoad(
        a_frag, As[tile],
        ((subgroup_m * CM) * SA + k_offset) / VWK,
        SA / VWK,
        gl_CooperativeMatrixLayoutRowMajor
      );
      coopMatLoad(
        b_frag, Bs[tile],
        (k_offset * SB + subgroup_n * CN) / VWN,
        SB / VWN,
        gl_CooperativeMatrixLayoutRowMajor
      );
      c_frag = coopMatMulAdd(a_frag, b_frag, c_frag);
    }

    const int next_kwg = kwg + KWG;
    if(next_kwg < padded_k_size) {
      if(DOUBLE_BUFFER == 0)
        barrier();
      const int next_tile = DOUBLE_BUFFER != 0 ? 1 - tile : tile;
      stage_tiles(next_tile, next_kwg, batch, group_m, group_n);
      barrier();
      tile = next_tile;
    }
  }

#if !FUSE_BN_ACT && ACC_BITS == 16
  // The dispatch covers the physically padded M/N matrix, and both padded
  // input rows and padded weight columns are zero. Store the half accumulator
  // directly instead of round-tripping it through shared memory.
  coopMatStore(
    c_frag, output_data,
    batch * output_batch_stride +
      (group_m + subgroup_m * CM) * output_channel_stride +
      group_n + subgroup_n * CN,
    output_channel_stride,
    gl_CooperativeMatrixLayoutRowMajor
  );
#else
  coopMatStore(
    c_frag, Cs,
    (subgroup_m * CM) * NWG + subgroup_n * CN,
    NWG,
    gl_CooperativeMatrixLayoutRowMajor
  );
  barrier();
  store_output_tile(batch, group_m, group_n);
#endif
}
