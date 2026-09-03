/**
 * @file vulkanshaders.h
 * @author Dohoon Kim(https://github.com/dhkim92-dev)
 * @details External symbols generated from SPIR-V shader binaries. 
 * These are produced by cmake/GenerateShaderSource.cmake during build. 
 * Symbol naming convention: _binary_<shader_name>_start, _binary_<shader_name>_end, _binary_<shader_name>_size
 */
#ifdef USE_VULKAN_BACKEND
#pragma once

#include <vector>
#include <cstddef>
#include <string>
#include "../neuralnet/activations.h"
#include "../neuralnet/vulkanhelpers.h"

extern "C" {
  // conv2d_fp32.glsl
  extern const unsigned char _binary_conv2d_fp32_start[];
  extern const unsigned char* _binary_conv2d_fp32_end;
  extern const size_t _binary_conv2d_fp32_size;

  // conv2d_p32s16.glsl
  extern const unsigned char _binary_conv2d_p32s16_start[];
  extern const unsigned char* _binary_conv2d_p32s16_end;
  extern const size_t _binary_conv2d_p32s16_size;

  // conv2d_p16s16.glsl
  extern const unsigned char _binary_conv2d_p16s16_start[];
  extern const unsigned char* _binary_conv2d_p16s16_end;
  extern const size_t _binary_conv2d_p16s16_size;

  // winograd_input_transform.glsl
  extern const unsigned char _binary_winograd_input_transform_fp32_start[];
  extern const unsigned char* _binary_winograd_input_transform_fp32_end;
  extern const size_t _binary_winograd_input_transform_fp32_size;

  extern const unsigned char _binary_winograd_input_transform_p32s16_start[];
  extern const unsigned char* _binary_winograd_input_transform_p32s16_end;
  extern const size_t _binary_winograd_input_transform_p32s16_size;

  extern const unsigned char _binary_winograd_input_transform_p16s16_start[];
  extern const unsigned char* _binary_winograd_input_transform_p16s16_end;
  extern const size_t _binary_winograd_input_transform_p16s16_size;

  extern const unsigned char _binary_winograd_input_transform_bnact_fp32_start[];
  extern const unsigned char* _binary_winograd_input_transform_bnact_fp32_end;
  extern const size_t _binary_winograd_input_transform_bnact_fp32_size;

  extern const unsigned char _binary_winograd_input_transform_bnact_p32s16_start[];
  extern const unsigned char* _binary_winograd_input_transform_bnact_p32s16_end;
  extern const size_t _binary_winograd_input_transform_bnact_p32s16_size;

  extern const unsigned char _binary_winograd_input_transform_bnact_p16s16_start[];
  extern const unsigned char* _binary_winograd_input_transform_bnact_p16s16_end;
  extern const size_t _binary_winograd_input_transform_bnact_p16s16_size;

  // winograd_output_transform.glsl
  extern const unsigned char _binary_winograd_output_transform_fp32_start[];
  extern const unsigned char* _binary_winograd_output_transform_fpo32__end;
  extern const size_t _binary_winograd_output_transform_fp32_size;

  extern const unsigned char _binary_winograd_output_transform_p32s16_start[];
  extern const unsigned char* _binary_winograd_output_transform_p32s16_end;
  extern const size_t _binary_winograd_output_transform_p32s16_size;

  extern const unsigned char _binary_winograd_output_transform_p16s16_start[];
  extern const unsigned char* _binary_winograd_output_transform_p16s16_end;
  extern const size_t _binary_winograd_output_transform_p16s16_size;

  // add_pointwise_fp32.glsl
  extern const unsigned char _binary_add_pointwise_fp32_start[];
  extern const unsigned char* _binary_add_pointwise_fp32_end;
  extern const size_t _binary_add_pointwise_fp32_size;

  // add_pointwise_p32s16.glsl
  extern const unsigned char _binary_add_pointwise_p32s16_start[];
  extern const unsigned char* _binary_add_pointwise_p32s16_end;
  extern const size_t _binary_add_pointwise_p32s16_size;

  // add_pointwise_p16s16.glsl
  extern const unsigned char _binary_add_pointwise_p16s16_start[];
  extern const unsigned char* _binary_add_pointwise_p16s16_end;
  extern const size_t _binary_add_pointwise_p16s16_size;

  // xgemm_batched_fp32.glsl
  extern const unsigned char _binary_xgemm_batched_fp32_start[];
  extern const unsigned char* _binary_xgemm_batched_fp32_end;
  extern const size_t _binary_xgemm_batched_fp32_size;

  // xgemm_batched_p32s16.glsl
  extern const unsigned char _binary_xgemm_batched_p32s16_start[];
  extern const unsigned char* _binary_xgemm_batched_p32s16_end;
  extern const size_t _binary_xgemm_batched_p32s16_size;

  // xgemm_batched_p16s16.glsl
  extern const unsigned char _binary_xgemm_batched_p16s16_start[];
  extern const unsigned char* _binary_xgemm_batched_p16s16_end;
  extern const size_t _binary_xgemm_batched_p16s16_size;

  // xgemm_direct_batched_tt_fp32_fp32.glsl
  extern const unsigned char _binary_xgemm_direct_batched_tt_fp32_start[];
  extern const unsigned char* _binary_xgemm_direct_batched_tt_fp32_end;
  extern const size_t _binary_xgemm_direct_batched_tt_fp32_size;

  // xgemm_direct_batched_tt_p32s16.glsl
  extern const unsigned char _binary_xgemm_direct_batched_tt_p32s16_start[];
  extern const unsigned char* _binary_xgemm_direct_batched_tt_p32s16_end;
  extern const size_t _binary_xgemm_direct_batched_tt_p32s16_size;

  // xgemm_direct_batched_tt_p16s16.glsl
  extern const unsigned char _binary_xgemm_direct_batched_tt_p16s16_start[];
  extern const unsigned char* _binary_xgemm_direct_batched_tt_p16s16_end;
  extern const size_t _binary_xgemm_direct_batched_tt_p16s16_size;

  // xgemm_strided_batched_nn_fp32.glsl
  extern const unsigned char _binary_xgemm_strided_batched_nn_fp32_start[];
  extern const unsigned char* _binary_xgemm_strided_batched_nn_fp32_end;
  extern const size_t _binary_xgemm_strided_batched_nn_fp32_size;

  // xgemm_strided_batched_nn_p32s16.glsl
  extern const unsigned char _binary_xgemm_strided_batched_nn_p32s16_start[];
  extern const unsigned char* _binary_xgemm_strided_batched_nn_p32s16_end;
  extern const size_t _binary_xgemm_strided_batched_nn_p32s16_size;

  // xgemm_strided_batched_nn_p16s16.glsl
  extern const unsigned char _binary_xgemm_strided_batched_nn_p16s16_start[];
  extern const unsigned char* _binary_xgemm_strided_batched_nn_p16s16_end;
  extern const size_t _binary_xgemm_strided_batched_nn_p16s16_size;

  // bn_mask_identity_fp32.glsl
  extern const unsigned char _binary_bn_mask_identity_fp32_start[];
  extern const unsigned char* _binary_bn_mask_identity_fp32_end;
  extern const size_t _binary_bn_mask_identity_fp32_size;

  extern const unsigned char _binary_bn_mask_identity_p32s16_start[];
  extern const unsigned char* _binary_bn_mask_identity_p32s16_end;
  extern const size_t _binary_bn_mask_identity_p32s16_size;

  extern const unsigned char _binary_bn_mask_identity_p16s16_start[];
  extern const unsigned char* _binary_bn_mask_identity_p16s16_end;
  extern const size_t _binary_bn_mask_identity_p16s16_size;

  // bn_mask_relu_fp32.glsl
  extern const unsigned char _binary_bn_mask_relu_fp32_start[];
  extern const unsigned char* _binary_bn_mask_relu_fp32_end;
  extern const size_t _binary_bn_mask_relu_fp32_size;

  extern const unsigned char _binary_bn_mask_relu_p32s16_start[];
  extern const unsigned char* _binary_bn_mask_relu_p32s16_end;
  extern const size_t _binary_bn_mask_relu_p32s16_size;

  extern const unsigned char _binary_bn_mask_relu_p16s16_start[];
  extern const unsigned char* _binary_bn_mask_relu_p16s16_end;
  extern const size_t _binary_bn_mask_relu_p16s16_size;

  // bn_mask_mish_fp32.glsl
  extern const unsigned char _binary_bn_mask_mish_fp32_start[];
  extern const unsigned char* _binary_bn_mask_mish_fp32_end;
  extern const size_t _binary_bn_mask_mish_fp32_size;

  extern const unsigned char _binary_bn_mask_mish_p32s16_start[];
  extern const unsigned char* _binary_bn_mask_mish_p32s16_end;
  extern const size_t _binary_bn_mask_mish_p32s16_size;

  extern const unsigned char _binary_bn_mask_mish_p16s16_start[];
  extern const unsigned char* _binary_bn_mask_mish_p16s16_end;
  extern const size_t _binary_bn_mask_mish_p16s16_size;

  // bn_mask_mish_scale8_fp32.glsl
  extern const unsigned char _binary_bn_mask_mish_scale8_fp32_start[];
  extern const unsigned char* _binary_bn_mask_mish_scale8_fp32_end;
  extern const size_t _binary_bn_mask_mish_scale8_fp32_size;

  // bn_mask_mish_scale8_fp32.glsl
  extern const unsigned char _binary_bn_mask_mish_scale8_p32s16_start[];
  extern const unsigned char* _binary_bn_mask_mish_scale8_p32s16_end;
  extern const size_t _binary_bn_mask_mish_scale8_p32s16_size;

  // bn_mask_mish_scale8_fp32.glsl
  extern const unsigned char _binary_bn_mask_mish_scale8_p16s16_start[];
  extern const unsigned char* _binary_bn_mask_mish_scale8_p16s16_end;
  extern const size_t _binary_bn_mask_mish_scale8_p16s16_size;

  // bn_mask_silu_fp32.glsl
  extern const unsigned char _binary_bn_mask_silu_fp32_start[];
  extern const unsigned char* _binary_bn_mask_silu_fp32_end;
  extern const size_t _binary_bn_mask_silu_fp32_size;

  extern const unsigned char _binary_bn_mask_silu_p32s16_start[];
  extern const unsigned char* _binary_bn_mask_silu_p32s16_end;
  extern const size_t _binary_bn_mask_silu_p32s16_size;

  extern const unsigned char _binary_bn_mask_silu_p16s16_start[];
  extern const unsigned char* _binary_bn_mask_silu_p16s16_end;
  extern const size_t _binary_bn_mask_silu_p16s16_size;

  // global_pooling_channels_fp32.glsl
  extern const unsigned char _binary_global_pooling_channels_fp32_start[];
  extern const unsigned char* _binary_global_pooling_channels_fp32_end;
  extern const size_t _binary_global_pooling_channels_fp32_size;

  // global_pooling_channels_p32s16.glsl
  extern const unsigned char _binary_global_pooling_channels_p32s16_start[];
  extern const unsigned char* _binary_global_pooling_channels_p32s16_end;
  extern const size_t _binary_global_pooling_channels_p32s16_size;

  // value_head_pool_channels_fp32.glsl
  extern const unsigned char _binary_value_head_pool_channels_fp32_start[];
  extern const unsigned char* _binary_value_head_pool_channels_fp32_end;
  extern const size_t _binary_value_head_pool_channels_fp32_size;

  // value_head_pool_channels_p32s16.glsl
  extern const unsigned char _binary_value_head_pool_channels_p32s16_start[];
  extern const unsigned char* _binary_value_head_pool_channels_p32s16_end;
  extern const size_t _binary_value_head_pool_channels_p32s16_size;

  // sum_channels_fp32.glsl
  extern const unsigned char _binary_sum_channels_fp32_start[];
  extern const unsigned char* _binary_sum_channels_fp32_end;
  extern const size_t _binary_sum_channels_fp32_size;

  // sum_channels_p32s16.glsl
  extern const unsigned char _binary_sum_channels_p32s16_start[];
  extern const unsigned char* _binary_sum_channels_p32s16_end;
  extern const size_t _binary_sum_channels_p32s16_size;

  // add_channel_bias_nchw_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nchw_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nchw_fp32_end;
  extern const size_t _binary_add_channel_bias_nchw_fp32_size;

  extern const unsigned char _binary_add_channel_bias_nchw_p32s16_start[];
  extern const unsigned char* _binary_add_channel_bias_nchw_p32s16_end;
  extern const size_t _binary_add_channel_bias_nchw_p32s16_size;

  extern const unsigned char _binary_add_channel_bias_nchw_p16s16_start[];
  extern const unsigned char* _binary_add_channel_bias_nchw_p16s16_end;
  extern const size_t _binary_add_channel_bias_nchw_p16s16_size;

  // add_channel_bias_nc_identity_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nc_identity_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_identity_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_identity_fp32_size;

  // add_channel_bias_nc_relu_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nc_relu_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_relu_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_relu_fp32_size;

  // add_channel_bias_nc_mish_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nc_mish_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_mish_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_mish_fp32_size;

  // add_channel_bias_nc_mish_scale8_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nc_mish_scale8_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_mish_scale8_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_mish_scale8_fp32_size;

  // add_channel_bias_nc_silu_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nc_silu_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_silu_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_silu_fp32_size;

  // extract channel0_nchw_fp32
  extern const unsigned char _binary_extract_channel0_nchw_fp32_start[];
  extern const unsigned char* _binary_extract_channel0_nchw_fp32_end;
  extern const size_t _binary_extract_channel0_nchw_fp32_size;

  // extract_channel0_nchw_p32s16
  extern const unsigned char _binary_extract_channel0_nchw_p32s16_start[];
  extern const unsigned char* _binary_extract_channel0_nchw_p32s16_end;
  extern const size_t _binary_extract_channel0_nchw_p32s16_size;

  // extract_channel0_nchw_p16s16
  extern const unsigned char _binary_extract_channel0_nchw_p16s16_start[];
  extern const unsigned char* _binary_extract_channel0_nchw_p16s16_end;
  extern const size_t _binary_extract_channel0_nchw_p16s16_size;

  // transformer_rms_norm_fp32
  extern const unsigned char _binary_transformer_rms_norm_fp32_start[];
  extern const unsigned char* _binary_transformer_rms_norm_fp32_end;
  extern const size_t _binary_transformer_rms_norm_fp32_size;

  extern const unsigned char _binary_transformer_rms_norm_p32s16_start[];
  extern const unsigned char* _binary_transformer_rms_norm_p32s16_end;
  extern const size_t _binary_transformer_rms_norm_p32s16_size;

  extern const unsigned char _binary_transformer_rms_norm_p16s16_start[];
  extern const unsigned char* _binary_transformer_rms_norm_p16s16_end;
  extern const size_t _binary_transformer_rms_norm_p16s16_size;

  // transformer_rope_fp32
  extern const unsigned char _binary_transformer_apply_rope_fp32_start[];
  extern const unsigned char* _binary_transformer_apply_rope_fp32_end;
  extern const size_t _binary_transformer_apply_rope_fp32_size;

  extern const unsigned char _binary_transformer_apply_rope_p32s16_start[];
  extern const unsigned char* _binary_transformer_apply_rope_p32s16_end;
  extern const size_t _binary_transformer_apply_rope_p32s16_size;

  extern const unsigned char _binary_transformer_apply_rope_p16s16_start[];
  extern const unsigned char* _binary_transformer_apply_rope_p16s16_end;
  extern const size_t _binary_transformer_apply_rope_p16s16_size;

  // transformer_scale_dot_product_naive
  extern const unsigned char _binary_transformer_scale_dot_product_naive_fp32_start[];
  extern const unsigned char* _binary_transformer_scale_dot_product_naive_fp32_end;
  extern const size_t _binary_transformer_scale_dot_product_naive_fp32_size;

  extern const unsigned char _binary_transformer_scale_dot_product_naive_p32s16_start[];
  extern const unsigned char* _binary_transformer_scale_dot_product_naive_p32s16_end;
  extern const size_t _binary_transformer_scale_dot_product_naive_p32s16_size;

  extern const unsigned char _binary_transformer_scale_dot_product_naive_p16s16_start[];
  extern const unsigned char* _binary_transformer_scale_dot_product_naive_p16s16_end;
  extern const size_t _binary_transformer_scale_dot_product_naive_p16s16_size;

  // transformer_scale_dot_product
  extern const unsigned char _binary_transformer_scale_dot_product_fp32_start[];
  extern const unsigned char* _binary_transformer_scale_dot_product_fp32_end;
  extern const size_t _binary_transformer_scale_dot_product_fp32_size;

  extern const unsigned char _binary_transformer_scale_dot_product_p32s16_start[];
  extern const unsigned char* _binary_transformer_scale_dot_product_p32s16_end;
  extern const size_t _binary_transformer_scale_dot_product_p32s16_size;

  extern const unsigned char _binary_transformer_scale_dot_product_p16s16_start[];
  extern const unsigned char* _binary_transformer_scale_dot_product_p16s16_end;
  extern const size_t _binary_transformer_scale_dot_product_p16s16_size;

  // transformer_swiglu_fp32
  extern const unsigned char _binary_transformer_swiglu_fp32_start[];
  extern const unsigned char* _binary_transformer_swiglu_fp32_end;
  extern const size_t _binary_transformer_swiglu_fp32_size;

  extern const unsigned char _binary_transformer_swiglu_p32s16_start[];
  extern const unsigned char* _binary_transformer_swiglu_p32s16_end;
  extern const size_t _binary_transformer_swiglu_p32s16_size;

  extern const unsigned char _binary_transformer_swiglu_p16s16_start[];
  extern const unsigned char* _binary_transformer_swiglu_p16s16_end;
  extern const size_t _binary_transformer_swiglu_p16s16_size;

  // transformer_spatial_rms_norm_apply_fp32
  extern const unsigned char _binary_transformer_spatial_rms_norm_apply_fp32_start[];
  extern const unsigned char* _binary_transformer_spatial_rms_norm_apply_fp32_end;
  extern const size_t _binary_transformer_spatial_rms_norm_apply_fp32_size;

  extern const unsigned char _binary_transformer_spatial_rms_norm_apply_p32s16_start[];
  extern const unsigned char* _binary_transformer_spatial_rms_norm_apply_p32s16_end;
  extern const size_t _binary_transformer_spatial_rms_norm_apply_p32s16_size;

  extern const unsigned char _binary_transformer_spatial_rms_norm_apply_p16s16_start[];
  extern const unsigned char* _binary_transformer_spatial_rms_norm_apply_p16s16_end;
  extern const size_t _binary_transformer_spatial_rms_norm_apply_p16s16_size;

  // transformer_spatial_rms_norm_reduce_fp32
  extern const unsigned char _binary_transformer_spatial_rms_norm_reduce_fp32_start[];
  extern const unsigned char* _binary_transformer_spatial_rms_norm_reduce_fp32_end;
  extern const size_t _binary_transformer_spatial_rms_norm_reduce_fp32_size;

  extern const unsigned char _binary_transformer_spatial_rms_norm_reduce_p32s16_start[];
  extern const unsigned char* _binary_transformer_spatial_rms_norm_reduce_p32s16_end;
  extern const size_t _binary_transformer_spatial_rms_norm_reduce_p32s16_size;

  extern const unsigned char _binary_transformer_spatial_rms_norm_reduce_p16s16_start[];
  extern const unsigned char* _binary_transformer_spatial_rms_norm_reduce_p16s16_end;
  extern const size_t _binary_transformer_spatial_rms_norm_reduce_p16s16_size;

  // transformer_spatial_rms_norm_sum_sq_fp32
  extern const unsigned char _binary_transformer_spatial_rms_norm_sum_sq_fp32_start[];
  extern const unsigned char* _binary_transformer_spatial_rms_norm_sum_sq_fp32_end;
  extern const size_t _binary_transformer_spatial_rms_norm_sum_sq_fp32_size;

  extern const unsigned char _binary_transformer_spatial_rms_norm_sum_sq_p32s16_start[];
  extern const unsigned char* _binary_transformer_spatial_rms_norm_sum_sq_p32s16_end;
  extern const size_t _binary_transformer_spatial_rms_norm_sum_sq_p32s16_size;

  extern const unsigned char _binary_transformer_spatial_rms_norm_sum_sq_p16s16_start[];
  extern const unsigned char* _binary_transformer_spatial_rms_norm_sum_sq_p16s16_end;
  extern const size_t _binary_transformer_spatial_rms_norm_sum_sq_p16s16_size;

}

/**
 * @brief SPIR-V shader data accessors.
 * Provides convenient access to embedded shader bytecode.
 * NCHW is the default data layout unless otherwise specified.
 */
namespace vk_shader {

  // conv2d_fp32 - Direct convolution layer
  extern const unsigned char* spirv_conv2d_fp32;
  extern size_t spirv_conv2d_fp32_size;

  extern const unsigned char* spirv_conv2d_p32s16;
  extern size_t spirv_conv2d_p32s16_size;

  extern const unsigned char* spirv_conv2d_p16s16;
  extern size_t spirv_conv2d_p16s16_size;

  extern const unsigned char* spirv_winograd_input_transform_fp32;
  extern size_t spirv_winograd_input_transform_fp32_size;

  extern const unsigned char* spirv_winograd_input_transform_p32s16;
  extern size_t spirv_winograd_input_transform_p32s16_size;

  extern const unsigned char* spirv_winograd_input_transform_p16s16;
  extern size_t spirv_winograd_input_transform_p16s16_size;

  extern const unsigned char* spirv_winograd_input_transform_bnact_fp32;
  extern size_t spirv_winograd_input_transform_bnact_fp32_size;

  extern const unsigned char* spirv_winograd_input_transform_bnact_p32s16;
  extern size_t spirv_winograd_input_transform_bnact_p32s16_size;

  extern const unsigned char* spirv_winograd_input_transform_bnact_p16s16;
  extern size_t spirv_winograd_input_transform_bnact_p16s16_size;

  extern const unsigned char* spirv_winograd_output_transform_fp32;
  extern size_t spirv_winograd_output_transform_fp32_size;

  extern const unsigned char* spirv_winograd_output_transform_p32s16;
  extern size_t spirv_winograd_output_transform_p32s16_size;

  extern const unsigned char* spirv_winograd_output_transform_p16s16;
  extern size_t spirv_winograd_output_transform_p16s16_size;

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

  extern const unsigned char* spirv_add_pointwise_p32s16;
  extern size_t spirv_add_pointwise_p32s16_size;

  extern const unsigned char* spirv_add_pointwise_p16s16;
  extern size_t spirv_add_pointwise_p16s16_size;

  extern const unsigned char* spirv_xgemm_batched_fp32;
  extern size_t spirv_xgemm_batched_fp32_size;

  extern const unsigned char* spirv_xgemm_batched_p32s16;
  extern size_t spirv_xgemm_batched_p32s16_size;

  extern const unsigned char* spirv_xgemm_batched_p16s16;
  extern size_t spirv_xgemm_batched_p16s16_size;

  // xgemm_direct_batched_tt_fp32_fp32 - batched xgemm direct fp32
  extern const unsigned char* spirv_xgemm_direct_batched_tt_fp32;
  extern size_t spirv_xgemm_direct_batched_tt_fp32_size;

  extern const unsigned char* spirv_xgemm_direct_batched_tt_p32s16;
  extern size_t spirv_xgemm_direct_batched_tt_p32s16_size;

  extern const unsigned char* spirv_xgemm_direct_batched_tt_p16s16;
  extern size_t spirv_xgemm_direct_batched_tt_p16s16_size;

  // strided batched matmul fp32 - for 1x1 conv
  extern const unsigned char* spirv_xgemm_strided_batched_nn_fp32;;
  extern size_t spirv_xgemm_strided_batched_nn_fp32_size;

  extern const unsigned char* spirv_xgemm_strided_batched_nn_p32s16;
  extern size_t spirv_xgemm_strided_batched_nn_p32s16_size;

  extern const unsigned char* spirv_xgemm_strided_batched_nn_p16s16;
  extern size_t spirv_xgemm_strided_batched_nn_p16s16_size;

  // Future support, NCHW matmul Fp32
  // extern const unsigned char* spirv_matmul_tiled_chw_4x4x32_fp32;
  // extern size_t spirv_matmul_tiled_chw_4x4x32_fp32_size;

  // bn_mask_fp32 - Batch normalization with mask (identity activation)
  extern const unsigned char* spirv_bn_mask_identity_fp32;
  extern size_t spirv_bn_mask_identity_fp32_size;

  extern const unsigned char* spirv_bn_mask_identity_p32s16;
  extern size_t spirv_bn_mask_identity_p32s16_size;

  extern const unsigned char* spirv_bn_mask_identity_p16s16;
  extern size_t spirv_bn_mask_identity_p16s16_size;

  // bn_mask_relu_fp32 - Batch normalization with mask + ReLU
  extern const unsigned char* spirv_bn_mask_relu_fp32;
  extern size_t spirv_bn_mask_relu_fp32_size;

  extern const unsigned char* spirv_bn_mask_relu_p32s16;
  extern size_t spirv_bn_mask_relu_p32s16_size;

  extern const unsigned char* spirv_bn_mask_relu_p16s16;
  extern size_t spirv_bn_mask_relu_p16s16_size;

  // bn_mask_mish_fp32 - Batch normalization with mask + Mish
  extern const unsigned char* spirv_bn_mask_mish_fp32;
  extern size_t spirv_bn_mask_mish_fp32_size;

  extern const unsigned char* spirv_bn_mask_mish_p32s16;
  extern size_t spirv_bn_mask_mish_p32s16_size;

  extern const unsigned char* spirv_bn_mask_mish_p16s16;
  extern size_t spirv_bn_mask_mish_p16s16_size;

  // bn_mask_mish_scale8_fp32 - Batch normalization with mask + Mish + Scale8
  extern const unsigned char* spirv_bn_mask_mish_scale8_fp32;
  extern size_t spirv_bn_mask_mish_scale8_fp32_size;

  extern const unsigned char* spirv_bn_mask_mish_scale8_p32s16;
  extern size_t spirv_bn_mask_mish_scale8_p32s16_size;

  extern const unsigned char* spirv_bn_mask_mish_scale8_p16s16;
  extern size_t spirv_bn_mask_mish_scale8_p16s16_size;

  // bn_mask_silu_fp32 - Batch normalization with mask + silu
  extern const unsigned char* spirv_bn_mask_silu_fp32;
  extern size_t spirv_bn_mask_silu_fp32_size;

  extern const unsigned char* spirv_bn_mask_silu_p32s16;
  extern size_t spirv_bn_mask_silu_p32s16_size;

  extern const unsigned char* spirv_bn_mask_silu_p16s16;
  extern size_t spirv_bn_mask_silu_p16s16_size;

  // sum_channels_fp32 - Sum over channels
  extern const unsigned char* spirv_sum_channels_fp32;
  extern size_t spirv_sum_channels_fp32_size;

  extern const unsigned char* spirv_sum_channels_p32s16;
  extern size_t spirv_sum_channels_p32s16_size;

  // global_pooling_channels_fp32 - Global pooling with mask
  extern const unsigned char* spirv_global_pooling_channels_fp32;
  extern size_t spirv_global_pooling_channels_fp32_size;

  extern const unsigned char* spirv_global_pooling_channels_p32s16;
  extern size_t spirv_global_pooling_channels_p32s16_size;

  // value_head_pool_channels_fp32 - Value head pooling
  extern const unsigned char* spirv_value_head_pool_channels_fp32;
  extern size_t spirv_value_head_pool_channels_fp32_size;

  extern const unsigned char* spirv_value_head_pool_channels_p32s16;
  extern size_t spirv_value_head_pool_channels_p32s16_size;

  // add_channel_bias_nchw_fp32 - Add channel bias (identity)
  extern const unsigned char* spirv_add_channel_bias_nchw_fp32;
  extern size_t spirv_add_channel_bias_nchw_fp32_size;

  extern const unsigned char* spirv_add_channel_bias_nchw_p32s16;
  extern size_t spirv_add_channel_bias_nchw_p32s16_size;

  extern const unsigned char* spirv_add_channel_bias_nchw_p16s16;
  extern size_t spirv_add_channel_bias_nchw_p16s16_size;

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

  // add_channel_bias_nc_relu_fp32 - Add channel bias + ReLU
  extern const unsigned char* spirv_add_channel_bias_nc_silu_fp32;
  extern size_t spirv_add_channel_bias_nc_silu_fp32_size;

  // Extract channel 0 from NCHW fp32
  extern const unsigned char* spirv_extract_channel0_nchw_fp32;
  extern size_t spirv_extract_channel0_nchw_fp32_size;

  extern const unsigned char* spirv_extract_channel0_nchw_p32s16;
  extern size_t spirv_extract_channel0_nchw_p32s16_size;

  extern const unsigned char* spirv_extract_channel0_nchw_p16s16;
  extern size_t spirv_extract_channel0_nchw_p16s16_size;

  // Transformer RMS Norm f32
  extern const unsigned char* spirv_transformer_rms_norm_fp32;
  extern size_t spirv_transformer_rms_norm_fp32_size;
  extern const unsigned char* spirv_transformer_rms_norm_p32s16;
  extern size_t spirv_transformer_rms_norm_p32s16_size;
  extern const unsigned char* spirv_transformer_rms_norm_p16s16;
  extern size_t spirv_transformer_rms_norm_p16s16_size;

  // Transformer apply RoPE fp32
  extern const unsigned char* spirv_transformer_apply_rope_fp32;
  extern size_t spirv_transformer_apply_rope_fp32_size;;
  extern const unsigned char* spirv_transformer_apply_rope_p32s16;
  extern size_t spirv_transformer_apply_rope_p32s16_size;
  extern const unsigned char* spirv_transformer_apply_rope_p16s16;
  extern size_t spirv_transformer_apply_rope_p16s16_size;

  // Transformer scale dot product naive
  extern const unsigned char* spirv_transformer_scale_dot_product_naive_fp32;
  extern size_t spirv_transformer_scale_dot_product_naive_fp32_size;
  extern const unsigned char* spirv_transformer_scale_dot_product_naive_p32s16;
  extern size_t spirv_transformer_scale_dot_product_naive_p32s16_size;
  extern const unsigned char* spirv_transformer_scale_dot_product_naive_p16s16;
  extern size_t spirv_transformer_scale_dot_product_naive_p16s16_size;

  // Transformer scale dot product
  extern const unsigned char* spirv_transformer_scale_dot_product_fp32;
  extern size_t spirv_transformer_scale_dot_product_fp32_size;;
  extern const unsigned char* spirv_transformer_scale_dot_product_p32s16;
  extern size_t spirv_transformer_scale_dot_product_p32s16_size;
  extern const unsigned char* spirv_transformer_scale_dot_product_p16s16;
  extern size_t spirv_transformer_scale_dot_product_p16s16_size;

  // Transformer SwiGLU fp32
  extern const unsigned char* spirv_transformer_swiglu_fp32;
  extern size_t spirv_transformer_swiglu_fp32_size;
  extern const unsigned char* spirv_transformer_swiglu_p32s16;
  extern size_t spirv_transformer_swiglu_p32s16_size;
  extern const unsigned char* spirv_transformer_swiglu_p16s16;
  extern size_t spirv_transformer_swiglu_p16s16_size;

  // Transformer spatial RMS norm apply fp32
  extern const unsigned char* spirv_transformer_spatial_rms_norm_apply_fp32;
  extern size_t spirv_transformer_spatial_rms_norm_apply_fp32_size;
  extern const unsigned char* spirv_transformer_spatial_rms_norm_apply_p32s16;
  extern size_t spirv_transformer_spatial_rms_norm_apply_p32s16_size;
  extern const unsigned char* spirv_transformer_spatial_rms_norm_apply_p16s16;
  extern size_t spirv_transformer_spatial_rms_norm_apply_p16s16_size;

  // Transformer spatial RMS norm reduce fp32
  extern const unsigned char* spirv_transformer_spatial_rms_norm_reduce_fp32;
  extern size_t spirv_transformer_spatial_rms_norm_reduce_fp32_size;
  extern const unsigned char* spirv_transformer_spatial_rms_norm_reduce_p32s16;
  extern size_t spirv_transformer_spatial_rms_norm_reduce_p32s16_size;
  extern const unsigned char* spirv_transformer_spatial_rms_norm_reduce_p16s16;
  extern size_t spirv_transformer_spatial_rms_norm_reduce_p16s16_size;

  // Transformer spatial RMS norm sum square fp32
  extern const unsigned char* spirv_transformer_spatial_rms_norm_sum_sq_fp32;
  extern size_t spirv_transformer_spatial_rms_norm_sum_sq_fp32_size;
  extern const unsigned char* spirv_transformer_spatial_rms_norm_sum_sq_p32s16;
  extern size_t spirv_transformer_spatial_rms_norm_sum_sq_p32s16_size;
  extern const unsigned char* spirv_transformer_spatial_rms_norm_sum_sq_p16s16;
  extern size_t spirv_transformer_spatial_rms_norm_sum_sq_p16s16_size;

  struct LocalDim {
    int x;
    int y;
    int z;

    bool operator==(const LocalDim& other) const {
        return x == other.x &&
                y == other.y &&
                z == other.z;
    }

    bool operator<(const LocalDim& other) const {
        if (x != other.x)
            return x < other.x;
        if (y != other.y)
            return y < other.y;
        return z < other.z;
    }
  };

struct LocalDimHash {
  std::size_t operator()(const LocalDim& dim) const noexcept {
    std::size_t seed = 0;

    auto hash_combine = [&seed](int value) {
        seed ^= std::hash<int>{}(value)
              + 0x9e3779b9
              + (seed << 6)
              + (seed >> 2);
    };

    hash_combine(dim.x);
    hash_combine(dim.y);
    hash_combine(dim.z);

    return seed;
  }
};



  namespace spec {
    struct Conv2DSpec {
      uint32_t localSizeX = 16;
      uint32_t localSizeY = 8;
      uint32_t localSizeZ = 1;
    };

    struct AddPointWiseSpec {
      uint32_t localSizeX = 256;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int ELTS_PER_THREAD = 1;
    };

    struct BatchNormMaskSpec {
      uint32_t localSizeX = 32;
      uint32_t localSizeY = 8;
      uint32_t localSizeZ = 1;
    };

    struct GlobalPoolingChannelsSpec {
      uint32_t localSizeX = 128;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int XYSTRIDE;
      int CHANNELSTRIDE;
      int LOCALSIZE_TOTAL;
    };

    struct ValueHeadPoolingChannelsSpec {
      uint32_t localSizeX = 128;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int XYSTRIDE;
      int CHANNELSTRIDE;
      int LOCALSIZE_TOTAL;
    };

    /**
     * @brief SumChannels Kernel Specialization Info
     * @param XYSTRIDE - power of two parallelism stride for reduction, should be gl_WorkGroupSize.x;
     * @param CHANNELSTRIDE - stride for channels, should be get gl_WorkGroupSize.y
     * @param LOCALSIZE_TOTAL = should be get gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z
     */
    struct SumChannelsSpec {
      uint32_t localSizeX = 32;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int XYSTRIDE;
      int CHANNELSTRIDE;
      int LOCALSIZE_TOTAL;
    };

    struct AddChannelBiasNCHWSpec {
      uint32_t localSizeX = 32;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int XY_ELTS_PER_THREAD = 1;
      int NC_ELTS_PER_THREAD = 1;
    };

    struct AddChannelBiasNCSpec {
      uint32_t localSizeX = 64;
      uint32_t localSizeY = 4;
      uint32_t localSizeZ = 1;
    };

    struct ExtractChannel0NCHWSpec {
      uint32_t localSizeX = 64;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
    };

    struct WinogradInputTransformBnActSpec {
      uint32_t localSizeX = 1;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int inTileYSize = 4;
      int inTileXSize = 4;
      int outTileYSize = 2;
      int outTileXSize = 2;
      int inTileYOffset = -1;
      int inTileXOffset = -1;
      int convY = 3;
      int convX = 3;
      int activation = 0; // 0: Identity, 1: ReLU, 2: Mish, 12: Mish + Scale8
    };

    struct WinogradInputTransformSpec {
      uint32_t localSizeX = 1;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int inTileYSize = 4;
      int inTileXSize = 4;
      int outTileYSize = 2;
      int outTileXSize = 2;
      int inTileYOffset = -1;
      int inTileXOffset = -1;
      int convY = 3;
      int convX = 3;
    };

    struct WinogradOutputTransformSpec {
      uint32_t localSizeX = 1;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int inTileYSize = 4;
      int inTileXSize = 4;
      int outTileYSize = 2;
      int outTileXSize = 2;
      int convY = 3;
      int convX = 3;
    };

    struct XGEMMBatchedSpec {
      uint32_t localSizeX = 16;
      uint32_t localSizeY = 16;
      uint32_t localSizeZ = 1;
      uint32_t MWG=64;
      uint32_t NWG=64;
      uint32_t KWG=32;
      uint32_t MDIMC=16;
      uint32_t NDIMC=16;
      uint32_t MDIMA=16;
      uint32_t NDIMB=16;
    };

    struct XgemmDirectSpec {
      uint32_t localSizeX = 16;
      uint32_t localSizeY = 16;
      uint32_t localSizeZ = 1;
      int WGD = 32;
      int MDIMCD = 8;
      int NDIMCD = 8;
      int MDIMAD = 16;
      int NDIMBD = 16;
      int KWID = 2;
      int PADA = 1;
      int PADB = 1;
    };

    struct TransformerRMSNormSpec {
      uint32_t localSizeX;
      uint32_t localSizeY;
      uint32_t localSizeZ;
      int WG_C_SIZE = 64;
      int WG_XY_SIZE = 1;
      int C_PER_THREAD = 4;
    };

    struct TransformerApplyRoPESpec {
      uint32_t localSizeX = 64;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
    };

    struct ScaleDotProductSpec {
      uint32_t localSizeX = 32;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int ATTN_BLOCK_Q = 32;
      int ATTN_BLOCK_KV = 32;
      int Q_PER_THREAD = 1;
      int ATTN_HEAD_DIM = 1;
      int ATTN_V_HEAD_DIM = 1;
    };

    struct ScaleDotProductNaiveSpec {
      uint32_t localSizeX = 1;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int ATTN_HEAD_DIM = 1;
      int ATTN_V_HEAD_DIM = 1;
    };

    struct TransformerSwiGLUSpec {
      uint32_t localSizeX = 32;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int ELTS_PER_THREAD = 1;
    };

    struct TransformerSpatialRMSNormApplySpec {
      uint32_t localSizeX = 32;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int APPLY_ELTS_PER_THREAD = 1;
    };

    struct TransformerSpatialRMSNormReduceSpec {
      uint32_t localSizeX = 32;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int TILE_SIZE = 32;
    };

    struct TransformerSpatialRMSNormSumSqSpec {
      uint32_t localSizeX = 32;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int TILE_SIZE = 32;
    };
  };

  namespace push {
    struct Conv2DPushConstantParams {
      int nSize;
      int xSize;
      int ySize;
      int ocSize;
      int icSize;
      int filterXRadius;
      int filterYRadius;
      int xyStride;
    };

    struct WinogradInputTransformParams {
      int batchSize;
      int nnXLen;
      int nnYLen;
      int numTilesX;
      int numTilesY;
      int inChannels;
      int inChannelsPadded;
      int ntxtySizePadded;
      int xyStride;
    };

    struct WinogradOutputTransformParams {
      int batchSize;
      int ySize;
      int xSize;
      int numTilesY;
      int numTilesX;
      int outChannels;
      int outChannelsPadded;
      int ntxtySizePadded;
      int xyStride;
    };

    struct XGEMMBatchedParams{
      uint32_t M;  
      uint32_t N;  
      uint32_t K;
      uint32_t aOne;
      uint32_t aTwo;
      uint32_t bOne;
      uint32_t bTwo;
      uint32_t cOne;
      uint32_t cTwo;
    };

    struct XgemmDirectBatchedTTParams {
      uint32_t M;  
      uint32_t N;
      uint32_t K;  
      uint32_t aLead;
      uint32_t bLead;
      uint32_t cLead;
      uint32_t aTranspose; // Input A Transpose
      uint32_t bTranspose; // Input B Transpose
      uint32_t cTranspose; // Output C Transpose
    };

    struct XgemmStridedBatchedFp32Params {
      uint32_t kSizeM;  
      uint32_t kSizeN;
      uint32_t kSizeK;  
      uint32_t aLead;
      uint32_t aStride;
      uint32_t bLead;
      uint32_t bStride;
      uint32_t cLead;
      uint32_t cStride;
      uint32_t cTranspose;
    };

    struct BatchedXgemmDirectFp32Params {
      uint32_t kSizeM;  
      uint32_t kSizeN;
      uint32_t kSizeK;  
      uint32_t aLead;
      uint32_t bLead;
      uint32_t cLead;
      uint32_t cTranspose; // Output C Transpose
    };

    /**
     * @brief Batch Normalization Mask Fp32 Push Constant Parameters 
     * @param batchSize
     * @param numChannels
     * @param nnYLen
     */
    struct BatchNormMaskParams {
      uint32_t batchSize;
      uint32_t numChannels;
      uint32_t nnXYLen;
    };

    /**
     * @brief Sum Channels Fp32 Push Constant Parameters
     * @param nSize - number of batch
     * @param cSize - number of channels
     * @param xySize - H*W spatial size
     */
    struct SumChannelsParams {
      uint32_t nSize;
      uint32_t cSize;
      uint32_t xySize;
    };

    /**
     * @brief MatBias Push Constant Parameters
     */
    struct MatBiasPushParams {
      uint32_t batchSize;
      uint32_t numChannels;
    };

    /**
      * @brief Global Pooling Channels Push Constant Parameters
      */
    struct GlobalPoolingChannelsParams {
      int nSize;
      int cSize;
      int xySize;
    };

    /**
      * @brief Value Head Pooling Channels Push Constant Parameters
      */
    struct ValueHeadPoolingChannelsParams {
      int nSize;
      int cSize;
      int xySize;
    };

    /**
    *  @brief Add Point Wise Push Constant Parameters
    **/
    struct AddPointWiseParams {
      uint32_t size;
    };

    /**
     * @brief Add Channel Bias NCHW Push Constant Parameters
     */
    struct AddChannelBiasNCHWParams {
      uint32_t ncSize;
      uint32_t xySize;
    };

    /**
     * @brief Add Channel Bias NC Push Constant Parameters
     */
    struct AddChannelBiasNCParams {
      uint32_t nSize;
      uint32_t cSize;
    };

    /**
     * @brief Push parameters for ExtractChannel0NCHW shader
     * @param nSize: number of batches
     * @param cSize: number of input channels
     * @param xySize: H*W spatial size
     */
    struct ExtractChannel0NCHWParams {
      int nSize;  
      int cSize;
      int xySize;
    };

    struct NCHWPushConstantParams {
      uint32_t N; // Batch size
      uint32_t C; // Channels
      uint32_t H; // Height
      uint32_t W; // Width
    };

    struct TransformerRMSNormPushParams {
      int nSize;
      int cSize;
      int xySize;
      float epsilon;
    };

    struct TransformerApplyRoPEPushParams {
      int nSize;
      int numBufHeads;
      int numKVHeads;
      int headDim;
      int xySize;
      int numPairs;
      int learnableRope;
    };

    /**
     * Scale dot product shader params, transformer_scale_dot_product & transformer_scale_dot_product_naive both share this struct.
     */
    struct ScaleDotProductPushParam {
      int seqLen;
      int numHeads;
      int numKVHeads;
      float scale;
    };

    struct TransformerSwiGLUPushParams {
      int size;
    };

    struct TransformerSpatialRMSNormApplyPushParams {
      int nSize;
      int cSize;
      int xySize;
      float eps;
    };

    struct TransformerSpatialRMSNormReducePushParams {
      int nSize;
      int numPartials;
      int tilesPerGroup;
    };

    struct TransformerSpatialRMSNormSumSqPushParams {
      int nSize;
      int cSize;
      int xySize;
      int tilesPerGroup;
    };
  };

  namespace tune {

    struct AddPointWiseTuneParams {
      int LOCAL_SIZE=64;
      int ELTS_PER_THREAD=4;

      bool isValid() const;
    };

    struct AddChannelBiasesNCHWTuneParams {
      int XY_ELTS_PER_THREAD=4;
      int NC_ELTS_PER_THREAD=4;

      bool isValid() const;
    };

    struct GPoolTuneParams {
      int XYSTRIDE=32;
      int CHANNELSTRIDE=1;
      int BATCHSTRIDE=4;

      bool isValid() const;
    };

    struct ConvTuneParams {
      uint32_t inTileYSize;
      uint32_t inTileXSize;
      uint32_t outTileYSize;
      uint32_t outTileXSize;
      uint32_t inputTransformLocalXSize;
      uint32_t inputTransformLocalYSize;
      uint32_t outputTransformLocalXSize;
      uint32_t outputTransformLocalYSize;
      uint32_t outputTransformLocalZSize;

      bool isValid(uint32_t expectedOutTileSize) const;
    };

    struct XgemmTuneParams {
      uint32_t MDIMC=8;
      uint32_t NDIMC=8;
      uint32_t MWG=32;
      uint32_t NWG=32;
      uint32_t KWG=32;
      uint32_t MDIMA=8;
      uint32_t NDIMB=8;

      bool isValid() const;
    };

    struct XgemmDirectTuneParams {
      uint32_t WGD = 32;
      uint32_t MDIMCD = 8;
      uint32_t NDIMCD = 16;
      uint32_t MDIMAD = 8;
      uint32_t NDIMBD = 8;
      uint32_t KWID = 2;
      uint32_t PADA = 1;
      uint32_t PADB = 1;

      bool isValid() const;
    };

    struct TransformerTuneParams {
      int ATTN_BLOCK_Q=128  ;
      int ATTN_BLOCK_KV=32;
      int Q_PER_THREAD=1;
      int USE_TILED_ATTN=1;

      bool isValid() const;
    };

    struct TransformerRMSNormTuneParms {
      int WG_C_SIZE = 32;
      int WG_XY_SIZE = 8;
      int C_PER_THREAD = 1;

      bool isValid() const;
    };

    struct TransformerSpatialRmsNormTuneParams {
      int TILE_SIZE = 128;
      int APPLY_ELTS_PER_THREAD = 16;

      bool isValid() const;
    };

    struct VulkanParams {
      bool canUseFP16Storage = false;
      bool canUseFP16Compute = false;
      bool canUseCooperativMatrix = false;
      bool canUseSubgroup = false;
      bool shouldUseFP16Storage = false;
      bool shouldUseFP16Compute = false;
      bool shouldUseCooperativeMatrix = false;
      bool shouldUseSubgroup = false;
    };

    struct VulkanTuneParams {
      VulkanParams vulkan;
      AddChannelBiasesNCHWTuneParams addChannelBiases;
      AddPointWiseTuneParams pointwise;
      GPoolTuneParams gPool;
      ConvTuneParams conv3x3;
      ConvTuneParams conv5x5;
      XgemmTuneParams xgemm;
      XgemmDirectTuneParams xgemmDirect;
      TransformerTuneParams transformer;
      TransformerRMSNormTuneParms rmsNorm;
      TransformerSpatialRmsNormTuneParams spatialRMSNorm;

      VulkanTuneParams(const VulkanTuneParams& other) = default;
      VulkanTuneParams& operator=(const VulkanTuneParams& other) = default;

      bool isValid() const;
      bool operator==(const VulkanTuneParams& other) const;
      bool operator!=(const VulkanTuneParams& other) const { return !(*this == other); }

      static void save(const std::string& filename, const VulkanTuneParams& config);
      static VulkanTuneParams load(const std::string& filename);

      VulkanTuneParams() {
        conv3x3 = ConvTuneParams();
        conv3x3.inTileYSize = 6;
        conv3x3.inTileXSize = 6;
        conv3x3.outTileYSize = 4;
        conv3x3.outTileXSize = 4;
        conv3x3.inputTransformLocalXSize = 4;
        conv3x3.inputTransformLocalYSize = 2;
        conv3x3.outputTransformLocalXSize = 8;
        conv3x3.outputTransformLocalYSize = 2;
        conv3x3.outputTransformLocalZSize = 2;

        conv5x5 = ConvTuneParams();
        conv5x5.inTileYSize = 6;
        conv5x5.inTileXSize = 6;
        conv5x5.outTileYSize = 2;
        conv5x5.outTileXSize = 2;
        conv5x5.inputTransformLocalXSize = 4;
        conv5x5.inputTransformLocalYSize = 2;
        conv5x5.outputTransformLocalXSize = 8;
        conv5x5.outputTransformLocalYSize = 2;
        conv5x5.outputTransformLocalZSize = 2;
        // xgemm = XgemmTuneParams{
        //   .MDIMC = 16,
        //   .NDIMC = 16,
        //   .MWG = 64,
        //   .NWG = 64,
        //   .KWG = 16,
        //   .MDIMA = 16,
        //   .NDIMB = 16
        // };

        // xgemmDirect = XgemmDirectTuneParams{
        //   .WGD = 32,
        //   .MDIMCD = 8,
        //   .NDIMCD = 8,
        //   .MDIMAD = 8,
        //   .NDIMBD = 8,
        //   .KWID = 2,
        //   .PADA = 1,
        //   .PADB = 1
        // };

        xgemm = XgemmTuneParams();

        xgemmDirect = XgemmDirectTuneParams();
      }
    };

  }

  struct ComputePipelines {
    VkDevice device;
    VkPipelineCache cache;
    Logger* logger;
    bool printPipelineCreation = false;

    // In this code, assume that NCHW is default format if no postfix is given.

    // Conv2D pipelines
    Pipeline conv2dFp32; 
    Pipeline winogradInputTransform3x3;
    Pipeline winogradInputTransform5x5;

    Pipeline winogradInputTransform3x3_bnact_identity;
    Pipeline winogradInputTransform3x3_bnact_relu;
    Pipeline winogradInputTransform3x3_bnact_mish;
    Pipeline winogradInputTransform3x3_bnact_mish_scale8;
    Pipeline winogradInputTransform5x5_bnact_identity;
    Pipeline winogradInputTransform5x5_bnact_relu;
    Pipeline winogradInputTransform5x5_bnact_mish;
    Pipeline winogradInputTransform5x5_bnact_mish_scale8;

    Pipeline winogradOutputTransform3x3;
    Pipeline winogradOutputTransform5x5;

    Pipeline addPointWise;  // operation for skipping connections

    // Pipeline for matrix multiplication
    // Pipeline batchedXgemmDirect;
    Pipeline xgemmDirectBatchedTT;
    Pipeline xgemmBatchedFp32;

    // note that conv1x1 can be implemented as matmul operation
    // Pipeline stridedBatchedMatmulFp32;
    Pipeline xgemmStridedBatchedFp32;

    // Batch Normalization pipelines
    // note that prediction phase does not need batch normalization operation separately
    // as the parameters can be folded into convolution scale and bias.
    Pipeline batchNormMaskIdentity;
    Pipeline batchNormMaskRelu;
    Pipeline batchNormMaskMish;
    Pipeline batchNormMaskMishScale8;
    Pipeline batchNormMaskSilu;

    // Pooling pipelines
    Pipeline globalPoolingChannelsFp32;
    
    std::map<LocalDim, Pipeline> valueHeadPoolingChannels;
    
    // Element wise operations
    std::map<LocalDim, Pipeline> sumChannels;

    Pipeline addChannelBiasNCHW;
    Pipeline addChannelBiasNCIdentity;
    Pipeline addChannelBiasNCRelu;
    Pipeline addChannelBiasNCMish;
    Pipeline addChannelBiasNCMishScale8;
    Pipeline addChannelBiasNCSilu;
    Pipeline extractChannel0NCHWFp32;

    // Transformer

    Pipeline transformerRmsNorm;
    Pipeline transformerApplyRoPE;
    Pipeline transformerScaleDotProduct;
    Pipeline transformerScaleDotProductNaive;
    Pipeline transformerSwiGLU;
    Pipeline transformerSpatialRMSNormApply;
    Pipeline transformerSpatialRMSNormReduce;
    Pipeline transformerSpatialRMSNormSumSq;

    ComputePipelines(VkDevice device_, Logger* logger_);
    ComputePipelines() = delete;
    ComputePipelines(const ComputePipelines&) = delete;
    ComputePipelines& operator=(const ComputePipelines&) = delete;

    ~ComputePipelines();

    VkResult createPipelines(const tune::VulkanTuneParams& tuneParams, int qHeadDim, int vHeadDim, bool print);
    VkResult createWinogradInputTransform(Pipeline& pipeline, const tune::ConvTuneParams& tuneParams, int convSize, const tune::VulkanParams& vulkanParams);
    VkResult createWinogradInputTransformBnAct(Pipeline& pipeline, const tune::ConvTuneParams& tuneParams, int convSize, int activation, const tune::VulkanParams& vulkanParams);
    VkResult createWinogradOutputTransform(Pipeline& pipeline, const tune::ConvTuneParams& tuneParams, int convSize, const tune::VulkanParams& vulkanParams);
    VkResult createAddPointWise(Pipeline& pipeline, const tune::AddPointWiseTuneParams& tuneParams, const tune::VulkanParams& vulkanParams);
    VkResult createXgemmDirectBatchedTT(Pipeline& pipeline, const tune::XgemmDirectTuneParams& tuneParams, const tune::VulkanParams& vulkanParams);
    VkResult createXgemmBatched(Pipeline& pipeline, const tune::XgemmTuneParams& tuneParams, const tune::VulkanParams& vulkanParams);
    VkResult createXgemmStridedBatched(Pipeline& pipeline, const tune::XgemmDirectTuneParams& tuneParams, const tune::VulkanParams& vulkanParams);
    VkResult createBatchNormMaskIdentity(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createBatchNormMaskRelu(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createBatchNormMaskMish(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createBatchNormMaskMishScale8(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createBatchNormMaskSilu(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createGlobalPoolingChannelsFp32(Pipeline& pipeline, const tune::GPoolTuneParams& tuneParams, const tune::VulkanParams& vulkanParams);
    VkResult createValueHeadPoolingChannels(Pipeline& pipeline, const tune::GPoolTuneParams& tuneParams, uint32_t localSizeY, uint32_t localSizeZ, const tune::VulkanParams& vulkanParams);
    VkResult createSumChannels(Pipeline& pipeline, const tune::GPoolTuneParams& tuneParams, uint32_t localSizeZ, const tune::VulkanParams& vulkanParams);
    VkResult createAddChannelBiasNCHW(Pipeline& pipeline, const tune::AddChannelBiasesNCHWTuneParams& tuneParams, const tune::VulkanParams& vulkanParams);
    VkResult createAddChannelBiasNCIdentity(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createAddChannelBiasNCRelu(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createAddChannelBiasNCMish(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createAddChannelBiasNCMishScale8(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createAddChannelBiasNCSilu(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createExtractChannel0NCHWFp32(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createTransformerRMSNorm(Pipeline& pipeline, const tune::TransformerRMSNormTuneParms& tuneParams, const tune::VulkanParams& vulkanParams);
    VkResult createTransformerApplyRoPE(Pipeline& pipeline, const tune::VulkanParams& vulkanParams);
    VkResult createTransformerScaleDotProduct(Pipeline& pipeline, const tune::TransformerTuneParams& tuneParams, int qHeadDim, int vHeadDim, const tune::VulkanParams& vulkanParams);
    VkResult createTransformerScaleDotProductNaive(Pipeline& pipeline, int qHeadDim, int vHeadDim, const tune::VulkanParams& vulkanParams);
    VkResult createTransformerSwiGLU(Pipeline& pipeline, const tune::AddPointWiseTuneParams& tuneParams, const tune::VulkanParams& vulkanParams);
    VkResult createTransformerSpatialRMSNormApply(Pipeline& pipeline, const tune::TransformerSpatialRmsNormTuneParams& tuneParams, const tune::VulkanParams& vulkanParams);
    VkResult createTransformerSpatialRMSNormReduce(Pipeline& pipeline, const tune::TransformerSpatialRmsNormTuneParams& tuneParams, const tune::VulkanParams& vulkanParams);
    VkResult createTransformerSpatialRMSNormSumSq(Pipeline& pipeline, const tune::TransformerSpatialRmsNormTuneParams& tuneParams, const tune::VulkanParams& vulkanParams);
    void destroyPipeline(Pipeline& pipeline);
    VkResult createPipeline(
      std::string pipelineName,
      const unsigned char* spirvBytes,
      size_t spirvSize,
      size_t bindingSize,
      uint32_t pushConstantSize,
      Pipeline &outPipeline,
      VkSpecializationInfo* specializationInfo = nullptr,
      uint32_t localSizeX = 1,
      uint32_t localSizeY = 1,
      uint32_t localSizeZ = 1
    );

  private :
    void destroyPipelines();
  };
}

#endif
