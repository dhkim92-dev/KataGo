#include "../neuralnet/vulkanshaders.h"

namespace VkSPIRVShaders {

  // conv2d_fp32
  const unsigned char* spirv_conv2d_fp32 = _binary_conv2d_fp32_start;
  size_t spirv_conv2d_fp32_size = _binary_conv2d_fp32_size;

  // conv2d_tiled_bn_act_3x3_fp32
  const unsigned char* spirv_conv2d_tiled_bn_act_3x3_fp32 = _binary_conv2d_tiled_bn_act_3x3_fp32_start;
  size_t spirv_conv2d_tiled_bn_act_3x3_fp32_size = _binary_conv2d_tiled_bn_act_3x3_fp32_size;

  // conv2d_tiled_bn_act_5x5_fp32
  const unsigned char* spirv_conv2d_tiled_bn_act_5x5_fp32 = _binary_conv2d_tiled_bn_act_5x5_fp32_start;
  size_t spirv_conv2d_tiled_bn_act_5x5_fp32_size = _binary_conv2d_tiled_bn_act_5x5_fp32_size;

  // add_pointwise_fp32
  const unsigned char* spirv_add_pointwise_fp32 = _binary_add_pointwise_fp32_start;
  size_t spirv_add_pointwise_fp32_size = _binary_add_pointwise_fp32_size;

  // matmul_fp32
  const unsigned char* spirv_matmul_fp32 = _binary_matmul_fp32_start;
  size_t spirv_matmul_fp32_size = _binary_matmul_fp32_size;

  // batched_xgemm_direct_fp32
  const unsigned char* spirv_batched_xgemm_direct = _binary_batched_xgemm_direct_start;
  size_t spirv_batched_xgemm_direct_size = _binary_batched_xgemm_direct_size;

  // strided_batched_matmul_fp32
  const unsigned char* spirv_strided_batched_matmul_fp32 = _binary_strided_batched_matmul_fp32_start;
  size_t spirv_strided_batched_matmul_fp32_size = _binary_strided_batched_matmul_fp32_size;

  // bn_mask_fp32
  const unsigned char* spirv_bn_mask_fp32 = _binary_bn_mask_fp32_start;
  size_t spirv_bn_mask_fp32_size = _binary_bn_mask_fp32_size;

  // bn_mask_relu_fp32
  const unsigned char* spirv_bn_mask_relu_fp32 = _binary_bn_mask_relu_fp32_start;
  size_t spirv_bn_mask_relu_fp32_size = _binary_bn_mask_relu_fp32_size;

  // bn_mask_mish_fp32
  const unsigned char* spirv_bn_mask_mish_fp32 = _binary_bn_mask_mish_fp32_start;
  size_t spirv_bn_mask_mish_fp32_size = _binary_bn_mask_mish_fp32_size;

  // bn_mask_mish_scale8_fp32
  const unsigned char* spirv_bn_mask_mish_scale8_fp32 = _binary_bn_mask_mish_scale8_fp32_start;
  size_t spirv_bn_mask_mish_scale8_fp32_size = _binary_bn_mask_mish_scale8_fp32_size;

  // sum_channels_fp32
  const unsigned char* spirv_sum_channels_fp32 = _binary_sum_channels_fp32_start;
  size_t spirv_sum_channels_fp32_size = _binary_sum_channels_fp32_size;

  // global_pooling_channels_fp32
  const unsigned char* spirv_global_pooling_channels_fp32 = _binary_global_pooling_channels_fp32_start;
  size_t spirv_global_pooling_channels_fp32_size = _binary_global_pooling_channels_fp32_size;

  // value_head_pool_channels_fp32
  const unsigned char* spirv_value_head_pool_channels_fp32 = _binary_value_head_pool_channels_fp32_start;
  size_t spirv_value_head_pool_channels_fp32_size = _binary_value_head_pool_channels_fp32_size;

  // add_channel_bias_nchw_fp32
  const unsigned char* spirv_add_channel_bias_nchw_fp32 = _binary_add_channel_bias_nchw_fp32_start;
  size_t spirv_add_channel_bias_nchw_fp32_size = _binary_add_channel_bias_nchw_fp32_size;

  // add_channel_bias_nc_fp32
  const unsigned char* spirv_add_channel_bias_nc_identity_fp32 = _binary_add_channel_bias_nc_identity_fp32_start;
  size_t spirv_add_channel_bias_nc_identity_fp32_size = _binary_add_channel_bias_nc_identity_fp32_size;

  // add_channel_bias_nc_relu_fp32
  const unsigned char* spirv_add_channel_bias_nc_relu_fp32 = _binary_add_channel_bias_nc_relu_fp32_start;
  size_t spirv_add_channel_bias_nc_relu_fp32_size = _binary_add_channel_bias_nc_relu_fp32_size;

  // add_channel_bias_nc_mish_fp32
  const unsigned char* spirv_add_channel_bias_nc_mish_fp32 = _binary_add_channel_bias_nc_mish_fp32_start;
  size_t spirv_add_channel_bias_nc_mish_fp32_size = _binary_add_channel_bias_nc_mish_fp32_size;

  // add_channel_bias_nc_mish_scale8_fp32
  const unsigned char* spirv_add_channel_bias_nc_mish_scale8_fp32 = _binary_add_channel_bias_nc_mish_scale8_fp32_start;
  size_t spirv_add_channel_bias_nc_mish_scale8_fp32_size = _binary_add_channel_bias_nc_mish_scale8_fp32_size;

  // extract_channel0_nchw_fp32
  const unsigned char* spirv_extract_channel0_nchw_fp32 = _binary_extract_channel0_nchw_fp32_start;
  size_t spirv_extract_channel0_nchw_fp32_size = _binary_extract_channel0_nchw_fp32_size;
}
