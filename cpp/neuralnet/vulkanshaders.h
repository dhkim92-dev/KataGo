#ifdef USE_VULKAN_BACKEND
#pragma once

#include <vector>
#include <cstddef>

/**
 * External symbols generated from SPIR-V shader binaries.
 * These are produced by cmake/GenerateShaderSource.cmake during build.
 * Symbol naming convention: _binary_<shader_name>_start, _binary_<shader_name>_end, _binary_<shader_name>_size
 */
extern "C" {
  // conv2d_fp32.hlsl
  extern const unsigned char _binary_conv2d_fp32_start[];
  extern const unsigned char* _binary_conv2d_fp32_end;
  extern const size_t _binary_conv2d_fp32_size;

  // conv2d_tiled_bn_act_fp32.hlsl
  extern const unsigned char _binary_conv2d_tiled_bn_act_3x3_fp32_start[];
  extern const unsigned char* _binary_conv2d_tiled_bn_act_3x3_fp32_end;
  extern const size_t _binary_conv2d_tiled_bn_act_3x3_fp32_size;

  // conv2d_tiled_bn_act_5x5_fp32.hlsl
  extern const unsigned char _binary_conv2d_tiled_bn_act_5x5_fp32_start[];
  extern const unsigned char* _binary_conv2d_tiled_bn_act_5x5_fp32_end;
  extern const size_t _binary_conv2d_tiled_bn_act_5x5_fp32_size;

  // add_pointwise_fp32.hlsl
  extern const unsigned char _binary_add_pointwise_fp32_start[];
  extern const unsigned char* _binary_add_pointwise_fp32_end;
  extern const size_t _binary_add_pointwise_fp32_size;

  // matmul_fp32.hlsl
  extern const unsigned char _binary_matmul_fp32_start[];
  extern const unsigned char* _binary_matmul_fp32_end;
  extern const size_t _binary_matmul_fp32_size;

  // strided_batched_matmul_fp32.hlsl
  extern const unsigned char _binary_strided_batched_matmul_fp32_start[];
  extern const unsigned char* _binary_strided_batched_matmul_fp32_end;
  extern const size_t _binary_strided_batched_matmul_fp32_size;

  // bn_mask_fp32.hlsl
  extern const unsigned char _binary_bn_mask_fp32_start[];
  extern const unsigned char* _binary_bn_mask_fp32_end;
  extern const size_t _binary_bn_mask_fp32_size;

  // bn_mask_relu_fp32.hlsl
  extern const unsigned char _binary_bn_mask_relu_fp32_start[];
  extern const unsigned char* _binary_bn_mask_relu_fp32_end;
  extern const size_t _binary_bn_mask_relu_fp32_size;

  // bn_mask_mish_fp32.hlsl
  extern const unsigned char _binary_bn_mask_mish_fp32_start[];
  extern const unsigned char* _binary_bn_mask_mish_fp32_end;
  extern const size_t _binary_bn_mask_mish_fp32_size;

  // bn_mask_mish_scale8_fp32.hlsl
  extern const unsigned char _binary_bn_mask_mish_scale8_fp32_start[];
  extern const unsigned char* _binary_bn_mask_mish_scale8_fp32_end;
  extern const size_t _binary_bn_mask_mish_scale8_fp32_size;

  // global_pooling_channels_fp32.hlsl
  extern const unsigned char _binary_global_pooling_channels_fp32_start[];
  extern const unsigned char* _binary_global_pooling_channels_fp32_end;
  extern const size_t _binary_global_pooling_channels_fp32_size;

  // value_head_pool_channels_fp32.hlsl
  extern const unsigned char _binary_value_head_pool_channels_fp32_start[];
  extern const unsigned char* _binary_value_head_pool_channels_fp32_end;
  extern const size_t _binary_value_head_pool_channels_fp32_size;

  // sum_channels_fp32.hlsl
  extern const unsigned char _binary_sum_channels_fp32_start[];
  extern const unsigned char* _binary_sum_channels_fp32_end;
  extern const size_t _binary_sum_channels_fp32_size;

  // add_channel_bias_nchw_fp32.hlsl
  extern const unsigned char _binary_add_channel_bias_nchw_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nchw_fp32_end;
  extern const size_t _binary_add_channel_bias_nchw_fp32_size;

  // add_channel_bias_nc_identity_fp32.hlsl
  extern const unsigned char _binary_add_channel_bias_nc_identity_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_identity_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_identity_fp32_size;

  // add_channel_bias_nc_relu_fp32.hlsl
  extern const unsigned char _binary_add_channel_bias_nc_relu_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_relu_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_relu_fp32_size;

  // add_channel_bias_nc_mish_fp32.hlsl
  extern const unsigned char _binary_add_channel_bias_nc_mish_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_mish_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_mish_fp32_size;

  // add_channel_bias_nc_mish_scale8_fp32.hlsl
  extern const unsigned char _binary_add_channel_bias_nc_mish_scale8_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_mish_scale8_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_mish_scale8_fp32_size;

  // extract channel0_nchw_fp32.hlsl
  extern const unsigned char _binary_extract_channel0_nchw_fp32_start[];
  extern const unsigned char* _binary_extract_channel0_nchw_fp32_end;
  extern const size_t _binary_extract_channel0_nchw_fp32_size;
}

/**
 * @brief SPIR-V shader data accessors.
 * Provides convenient access to embedded shader bytecode.
 * NCHW is the default data layout unless otherwise specified.
 */
namespace VkSPIRVShaders {

  // conv2d_fp32 - Direct convolution layer
  extern const unsigned char* spirv_conv2d_fp32;
  extern size_t spirv_conv2d_fp32_size;

  extern const unsigned char* spirv_conv2d_tiled_bn_act_3x3_fp32;
  extern size_t spirv_conv2d_tiled_bn_act_3x3_fp32_size;

  extern const unsigned char* spirv_conv2d_tiled_bn_act_5x5_fp32;
  extern size_t spirv_conv2d_tiled_bn_act_5x5_fp32_size;

  // Convolution layers 3x3 with identity activation function
  // extern const unsigned char* spirv_conv2d_3x3_bn_fp32;
  // extern size_t spirv_conv2d_3x3_bn_fp32_size;

  // Convolution layers 3x3 with ReLU activation function
  // extern const unsigned char* spirv_conv2d_3x3_bn_relu_fp32;
  // extern size_t spirv_conv2d_3x3_bn_relu_fp32_size;

  // Convolution layers 3x3 with Mish activation function
  // extern const unsigned char* spirv_conv2d_3x3_bn_mish_fp32;
  // extern size_t spirv_conv2d_3x3_bn_mish_fp32_size;

  // Convolution layers 5x5 with identity activation function
  // extern const unsigned char* spirv_conv2d_5x5_bn_fp32;
  // extern size_t spirv_conv2d_5x5_bn_fp32_size;

  // Convolution layers 5x5 with ReLU activation function
  // extern const unsigned char* spirv_conv2d_5x5_bn_relu_fp32;
  // extern size_t spirv_conv2d_5x5_bn_relu_fp32_size;

  // Convolution layers 5x5 with Mish activation function
  // extern const unsigned char* spirv_conv2d_5x5_bn_mish_fp32;
  // extern size_t spirv_conv2d_5x5_bn_mish_fp32_size;

  // add_pointwise_fp32 - Element-wise add for skip connections
  extern const unsigned char* spirv_add_pointwise_fp32;
  extern size_t spirv_add_pointwise_fp32_size;

  // matmul_fp32 - Matrix multiplication
  extern const unsigned char* spirv_matmul_fp32;
  extern size_t spirv_matmul_fp32_size;

  // strided batched matmul fp32 - for 1x1 conv
  extern const unsigned char* spirv_strided_batched_matmul_fp32;;
  extern size_t spirv_strided_batched_matmul_fp32_size;

  // Future support, NCHW matmul Fp32
  // extern const unsigned char* spirv_matmul_tiled_chw_4x4x32_fp32;
  // extern size_t spirv_matmul_tiled_chw_4x4x32_fp32_size;

  // bn_mask_fp32 - Batch normalization with mask (identity activation)
  extern const unsigned char* spirv_bn_mask_fp32;
  extern size_t spirv_bn_mask_fp32_size;

  // bn_mask_relu_fp32 - Batch normalization with mask + ReLU
  extern const unsigned char* spirv_bn_mask_relu_fp32;
  extern size_t spirv_bn_mask_relu_fp32_size;

  // bn_mask_mish_fp32 - Batch normalization with mask + Mish
  extern const unsigned char* spirv_bn_mask_mish_fp32;
  extern size_t spirv_bn_mask_mish_fp32_size;

  // bn_mask_mish_scale8_fp32 - Batch normalization with mask + Mish + Scale8
  extern const unsigned char* spirv_bn_mask_mish_scale8_fp32;
  extern size_t spirv_bn_mask_mish_scale8_fp32_size;

  // sum_channels_fp32 - Sum over channels
  extern const unsigned char* spirv_sum_channels_fp32;
  extern size_t spirv_sum_channels_fp32_size;

  // global_pooling_channels_fp32 - Global pooling with mask
  extern const unsigned char* spirv_global_pooling_channels_fp32;
  extern size_t spirv_global_pooling_channels_fp32_size;

  // value_head_pool_channels_fp32 - Value head pooling
  extern const unsigned char* spirv_value_head_pool_channels_fp32;
  extern size_t spirv_value_head_pool_channels_fp32_size;

  // add_channel_bias_nchw_fp32 - Add channel bias (identity)
  extern const unsigned char* spirv_add_channel_bias_nchw_fp32;
  extern size_t spirv_add_channel_bias_nchw_fp32_size;

  // add_channel_bias_nc_identity_fp32 - Add channel bias + Identity
  extern const unsigned char* spirv_add_channel_bias_nc_identity_fp32;
  extern size_t spirv_add_channel_bias_nc_identity_fp32_size;

  // add_channel_bias_nc_relu_fp32 - Add channel bias + ReLU
  extern const unsigned char* spirv_add_channel_bias_nc_relu_fp32;
  extern size_t spirv_add_channel_bias_nc_relu_fp32_size;

  // add_channel_bias_nc_mish_fp32 - Add channel bias + Mish
  extern const unsigned char* spirv_add_channel_bias_nc_mish_fp32;
  extern size_t spirv_add_channel_bias_nc_mish_fp32_size;

  // add_channel_bias_nc_mish_scale8_fp32 - Add channel bias + Mish + Scale8
  extern const unsigned char* spirv_add_channel_bias_nc_mish_scale8_fp32;
  extern size_t spirv_add_channel_bias_nc_mish_scale8_fp32_size;

  // Extract channel 0 from NCHW fp32
  extern const unsigned char* spirv_extract_channel0_nchw_fp32;
  extern size_t spirv_extract_channel0_nchw_fp32_size;
}

#endif
