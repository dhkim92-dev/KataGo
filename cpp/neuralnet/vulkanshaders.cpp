#include "../neuralnet/vulkanshaders.h"

namespace VkSPIRVShaders {

  const unsigned char* spirv_conv2d_fp32 = _binary_conv2d_fp32_start;
  size_t spirv_conv2d_fp32_size = _binary_conv2d_fp32_end - _binary_conv2d_fp32_start;

  const unsigned char* spirv_conv2d_3x3_bn_fp32 = _binary_conv2d_3x3_bn_fp32_start;
  size_t spirv_conv2d_3x3_bn_fp32_size = _binary_conv2d_3x3_bn_fp32_end - _binary_conv2d_3x3_bn_fp32_start;

  const unsigned char* spirv_conv2d_3x3_bn_relu_fp32 = _binary_conv2d_3x3_bn_relu_fp32_start;
  size_t spirv_conv2d_3x3_bn_relu_fp32_size = _binary_conv2d_3x3_bn_relu_fp32_end - _binary_conv2d_3x3_bn_relu_fp32_start;

  const unsigned char* spirv_conv2d_3x3_bn_mish_fp32 = _binary_conv2d_3x3_bn_mish_fp32_start;
  size_t spirv_conv2d_3x3_bn_mish_fp32_size = _binary_conv2d_3x3_bn_mish_fp32_end - _binary_conv2d_3x3_bn_mish_fp32_start;

  const unsigned char* spirv_conv2d_5x5_bn_fp32 = _binary_conv2d_5x5_bn_fp32_start;
  size_t spirv_conv2d_5x5_bn_fp32_size = _binary_conv2d_5x5_bn_fp32_end - _binary_conv2d_5x5_bn_fp32_start;

  const unsigned char* spirv_conv2d_5x5_bn_relu_fp32 = _binary_conv2d_5x5_bn_relu_fp32_start;
  size_t spirv_conv2d_5x5_bn_relu_fp32_size = _binary_conv2d_5x5_bn_relu_fp32_end - _binary_conv2d_5x5_bn_relu_fp32_start;

  const unsigned char* spirv_conv2d_5x5_bn_mish_fp32 = _binary_conv2d_5x5_bn_mish_fp32_start;
  size_t spirv_conv2d_5x5_bn_mish_fp32_size = _binary_conv2d_5x5_bn_mish_fp32_end - _binary_conv2d_5x5_bn_mish_fp32_start;

  const unsigned char* spirv_add_pointwise_fp32 = _binary_add_pointwise_fp32_start;
  size_t spirv_add_pointwise_fp32_size = _binary_add_pointwise_fp32_end - _binary_add_pointwise_fp32_start;

  const unsigned char* spirv_matmul_fp32 = _binary_matmul_fp32_start;
  size_t spirv_matmul_fp32_size = _binary_matmul_fp32_end - _binary_matmul_fp32_start;

  // const unsigned char* spirv_matmul_tiled_chw_4x4x32_fp32 = _binary_matmul_tiled_chw_4x4x32_fp32_start;
  // size_t spirv_matmul_tiled_chw_4x4x32_fp32_size = _binary_matmul_tiled_chw_4x4x32_fp32_end - _binary_matmul_tiled_chw_4x4x32_fp32_start;

  const unsigned char* spirv_bn_mask_fp32 = _binary_bn_mask_fp32_start;
  size_t spirv_bn_mask_fp32_size = _binary_bn_mask_fp32_end - _binary_bn_mask_fp32_start;

  const unsigned char* spirv_bn_mask_relu_fp32 = _binary_bn_mask_relu_fp32_start;
  size_t spirv_bn_mask_relu_fp32_size = _binary_bn_mask_relu_fp32_end - _binary_bn_mask_relu_fp32_start;

  const unsigned char* spirv_bn_mask_mish_fp32 = _binary_bn_mask_mish_fp32_start;
  size_t spirv_bn_mask_mish_fp32_size = _binary_bn_mask_mish_fp32_end - _binary_bn_mask_mish_fp32_start;

  const unsigned char* spirv_sum_channels_fp32 = _binary_sum_channels_fp32_start;
  size_t spirv_sum_channels_fp32_size = _binary_sum_channels_fp32_end - _binary_sum_channels_fp32_start;

  const unsigned char* spirv_global_pooling_channels_fp32 = _binary_global_pooling_channels_fp32_start;
  size_t spirv_global_pooling_channels_fp32_size = _binary_global_pooling_channels_fp32_end - _binary_global_pooling_channels_fp32_start;

  const unsigned char* spirv_value_head_pool_channels_fp32 = _binary_value_head_pool_channels_fp32_start;
  size_t spirv_value_head_pool_channels_fp32_size = _binary_value_head_pool_channels_fp32_end - _binary_value_head_pool_channels_fp32_start;

  const unsigned char* spirv_add_channel_bias_nchw_fp32 = _binary_add_channel_bias_nchw_fp32_start;
  size_t spirv_add_channel_bias_nchw_fp32_size = _binary_add_channel_bias_nchw_fp32_end - _binary_add_channel_bias_nchw_fp32_start;

  const unsigned char* spirv_add_channel_bias_nc_relu_fp32 = _binary_add_channel_bias_nc_relu_fp32_start;
  size_t spirv_add_channel_bias_nc_relu_fp32_size = _binary_add_channel_bias_nc_relu_fp32_end - _binary_add_channel_bias_nc_relu_fp32_start;

  const unsigned char* spirv_add_channel_bias_nc_mish_fp32 = _binary_add_channel_bias_nc_mish_fp32_start;
  size_t spirv_add_channel_bias_nc_mish_fp32_size = _binary_add_channel_bias_nc_mish_fp32_end - _binary_add_channel_bias_nc_mish_fp32_start;

  const unsigned char* spirv_extract_channel0_nchw_fp32 = _binary_extract_channel0_nchw_fp32_start;
  size_t spirv_extract_channel0_nchw_fp32_size = _binary_extract_channel0_nchw_fp32_end - _binary_extract_channel0_nchw_fp32_start;
}
