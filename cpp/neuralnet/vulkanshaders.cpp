/**
 * @file vulkanshaders.cpp
 * @author Dohoon Kim(https://github.com/dhkim92-dev)
 * @details connect external symbols data to cpp variables
 */
#ifdef USE_VULKAN_BACKEND
#include "../neuralnet/vulkanshaders.h"

#include <iostream>

using namespace vk_shader;
using namespace vk_shader::push;
using namespace vk_shader::spec;
using namespace vk_shader::tune;

namespace vk_shader {

  namespace {
    template <typename Spec>
    struct SpecializationData {
      std::vector<int32_t> data;
      std::vector<VkSpecializationMapEntry> mapEntries;
      VkSpecializationInfo info;

      explicit SpecializationData(Spec& spec)
        : data(vk_helper::createSpecData(&spec, sizeof(Spec))),
          mapEntries(vk_helper::createSpecMapEntries(data.size())),
          info(vk_helper::createSpecializationInfo(data, mapEntries))
      {}
    };

  }


  // conv2d_fp32
  const unsigned char* spirv_conv2d_fp32 = _binary_conv2d_fp32_start;
  size_t spirv_conv2d_fp32_size = _binary_conv2d_fp32_size;

  // conv2d_p32s16
  const unsigned char* spirv_conv2d_p32s16 = _binary_conv2d_p32s16_start;
  size_t spirv_conv2d_p32s16_size = _binary_conv2d_p32s16_size;

  // conv2d_p16s16
  const unsigned char* spirv_conv2d_p16s16 = _binary_conv2d_p16s16_start;
  size_t spirv_conv2d_p16s16_size = _binary_conv2d_p16s16_size;

  // winograd_input_transform 
  const unsigned char* spirv_winograd_input_transform_fp32 = _binary_winograd_input_transform_fp32_start;
  size_t spirv_winograd_input_transform_fp32_size = _binary_winograd_input_transform_fp32_size;
  const unsigned char* spirv_winograd_input_transform_p32s16 = _binary_winograd_input_transform_p32s16_start;
  size_t spirv_winograd_input_transform_p32s16_size = _binary_winograd_input_transform_p32s16_size;
  const unsigned char* spirv_winograd_input_transform_p16s16 = _binary_winograd_input_transform_p16s16_start;
  size_t spirv_winograd_input_transform_p16s16_size = _binary_winograd_input_transform_p16s16_size;

  // winograd_input_transform_bnact
  const unsigned char* spirv_winograd_input_transform_bnact_fp32 = _binary_winograd_input_transform_bnact_fp32_start;
  size_t spirv_winograd_input_transform_bnact_fp32_size = _binary_winograd_input_transform_bnact_fp32_size;
  const unsigned char* spirv_winograd_input_transform_bnact_p32s16 = _binary_winograd_input_transform_bnact_p32s16_start;
  size_t spirv_winograd_input_transform_bnact_p32s16_size = _binary_winograd_input_transform_bnact_p32s16_size;
  const unsigned char* spirv_winograd_input_transform_bnact_p16s16 = _binary_winograd_input_transform_bnact_p16s16_start;
  size_t spirv_winograd_input_transform_bnact_p16s16_size = _binary_winograd_input_transform_bnact_p16s16_size;

  // winograd_output_transform
  const unsigned char* spirv_winograd_output_transform_fp32 = _binary_winograd_output_transform_fp32_start;
  size_t spirv_winograd_output_transform_fp32_size = _binary_winograd_output_transform_fp32_size;
  const unsigned char* spirv_winograd_output_transform_p32s16 = _binary_winograd_output_transform_p32s16_start;
  size_t spirv_winograd_output_transform_p32s16_size = _binary_winograd_output_transform_p32s16_size;
  const unsigned char* spirv_winograd_output_transform_p16s16 = _binary_winograd_output_transform_p16s16_start;
  size_t spirv_winograd_output_transform_p16s16_size = _binary_winograd_output_transform_p16s16_size;

  // add_pointwise_fp32
  const unsigned char* spirv_add_pointwise_fp32 = _binary_add_pointwise_fp32_start;
  size_t spirv_add_pointwise_fp32_size = _binary_add_pointwise_fp32_size;
  const unsigned char* spirv_add_pointwise_p32s16 = _binary_add_pointwise_p32s16_start;
  size_t spirv_add_pointwise_p32s16_size = _binary_add_pointwise_p32s16_size;
  const unsigned char* spirv_add_pointwise_p16s16 = _binary_add_pointwise_p16s16_start;
  size_t spirv_add_pointwise_p16s16_size = _binary_add_pointwise_p16s16_size;

  // xgemm_batched_fp32
  const unsigned char* spirv_xgemm_batched_fp32 = _binary_xgemm_batched_fp32_start;
  size_t spirv_xgemm_batched_fp32_size = _binary_xgemm_batched_fp32_size;

  // xgemm_batched_p32s16
  const unsigned char* spirv_xgemm_batched_p32s16 = _binary_xgemm_batched_p32s16_start;
  size_t spirv_xgemm_batched_p32s16_size = _binary_xgemm_batched_p32s16_size;

  // xgemm_batched_p16s16
  const unsigned char* spirv_xgemm_batched_p16s16 = _binary_xgemm_batched_p16s16_start;
  size_t spirv_xgemm_batched_p16s16_size = _binary_xgemm_batched_p16s16_size;

  // xgemm_direct_batched_tt_fp32_fp32
  const unsigned char* spirv_xgemm_direct_batched_tt_fp32 = _binary_xgemm_direct_batched_tt_fp32_start;
  size_t spirv_xgemm_direct_batched_tt_fp32_size = _binary_xgemm_direct_batched_tt_fp32_size;

  // xgemm_direct_batched_tt_p32s16
  const unsigned char* spirv_xgemm_direct_batched_tt_p32s16 = _binary_xgemm_direct_batched_tt_p32s16_start;
  size_t spirv_xgemm_direct_batched_tt_p32s16_size = _binary_xgemm_direct_batched_tt_p32s16_size;

  // xgemm_direct_batched_tt_p16s16
  const unsigned char* spirv_xgemm_direct_batched_tt_p16s16 = _binary_xgemm_direct_batched_tt_p16s16_start;
  size_t spirv_xgemm_direct_batched_tt_p16s16_size = _binary_xgemm_direct_batched_tt_p16s16_size;

  // xgemm_strided_batched_nn_fp32
  const unsigned char* spirv_xgemm_strided_batched_nn_fp32 = _binary_xgemm_strided_batched_nn_fp32_start;
  size_t spirv_xgemm_strided_batched_nn_fp32_size = _binary_xgemm_strided_batched_nn_fp32_size;

  // xgemm_strided_batched_nn_p32s16
  const unsigned char* spirv_xgemm_strided_batched_nn_p32s16 = _binary_xgemm_strided_batched_nn_p32s16_start;
  size_t spirv_xgemm_strided_batched_nn_p32s16_size = _binary_xgemm_strided_batched_nn_p32s16_size;

  // xgemm_strided_batched_nn_p16s16
  const unsigned char* spirv_xgemm_strided_batched_nn_p16s16 = _binary_xgemm_strided_batched_nn_p16s16_start;
  size_t spirv_xgemm_strided_batched_nn_p16s16_size = _binary_xgemm_strided_batched_nn_p16s16_size;

  // bn_mask_identity_fp32
  const unsigned char* spirv_bn_mask_identity_fp32 = _binary_bn_mask_identity_fp32_start;
  size_t spirv_bn_mask_identity_fp32_size = _binary_bn_mask_identity_fp32_size;

  // bn_mask_identity_p32s16
  const unsigned char* spirv_bn_mask_identity_p32s16 = _binary_bn_mask_identity_p32s16_start;
  size_t spirv_bn_mask_identity_p32s16_size = _binary_bn_mask_identity_p32s16_size;

  // bn_mask_identity_p16s16
  const unsigned char* spirv_bn_mask_identity_p16s16 = _binary_bn_mask_identity_p16s16_start;
  size_t spirv_bn_mask_identity_p16s16_size = _binary_bn_mask_identity_p16s16_size;

  // bn_mask_relu_fp32
  const unsigned char* spirv_bn_mask_relu_fp32 = _binary_bn_mask_relu_fp32_start;
  size_t spirv_bn_mask_relu_fp32_size = _binary_bn_mask_relu_fp32_size;

  // bn_mask_relu_p32s16
  const unsigned char* spirv_bn_mask_relu_p32s16 = _binary_bn_mask_relu_p32s16_start;
  size_t spirv_bn_mask_relu_p32s16_size = _binary_bn_mask_relu_p32s16_size;

  // bn_mask_relu_p16s16
  const unsigned char* spirv_bn_mask_relu_p16s16 = _binary_bn_mask_relu_p16s16_start;
  size_t spirv_bn_mask_relu_p16s16_size = _binary_bn_mask_relu_p16s16_size;

  // bn_mask_mish_fp32
  const unsigned char* spirv_bn_mask_mish_fp32 = _binary_bn_mask_mish_fp32_start;
  size_t spirv_bn_mask_mish_fp32_size = _binary_bn_mask_mish_fp32_size;

  // bn_mask_mish_p32s16
  const unsigned char* spirv_bn_mask_mish_p32s16 = _binary_bn_mask_mish_p32s16_start;
  size_t spirv_bn_mask_mish_p32s16_size = _binary_bn_mask_mish_p32s16_size;

  // bn_mask_mish_p16s16
  const unsigned char* spirv_bn_mask_mish_p16s16 = _binary_bn_mask_mish_p16s16_start;
  size_t spirv_bn_mask_mish_p16s16_size = _binary_bn_mask_mish_p16s16_size;

  // bn_mask_mish_scale8_fp32
  const unsigned char* spirv_bn_mask_mish_scale8_fp32 = _binary_bn_mask_mish_scale8_fp32_start;
  size_t spirv_bn_mask_mish_scale8_fp32_size = _binary_bn_mask_mish_scale8_fp32_size;

  // bn_mask_mish_scale8_p32s16
  const unsigned char* spirv_bn_mask_mish_scale8_p32s16 = _binary_bn_mask_mish_scale8_p32s16_start;
  size_t spirv_bn_mask_mish_scale8_p32s16_size = _binary_bn_mask_mish_scale8_p32s16_size;

  // bn_mask_mish_scale8_p16s16
  const unsigned char* spirv_bn_mask_mish_scale8_p16s16 = _binary_bn_mask_mish_scale8_p16s16_start;
  size_t spirv_bn_mask_mish_scale8_p16s16_size = _binary_bn_mask_mish_scale8_p16s16_size;

  // bn_mask_silu_fp32
  const unsigned char* spirv_bn_mask_silu_fp32 = _binary_bn_mask_silu_fp32_start;
  size_t spirv_bn_mask_silu_fp32_size = _binary_bn_mask_silu_fp32_size;

  // bn_mask_silu_p32s16
  const unsigned char* spirv_bn_mask_silu_p32s16 = _binary_bn_mask_silu_p32s16_start;
  size_t spirv_bn_mask_silu_p32s16_size = _binary_bn_mask_silu_p32s16_size;

  // bn_mask_silu_p16s16
  const unsigned char* spirv_bn_mask_silu_p16s16 = _binary_bn_mask_silu_p16s16_start;
  size_t spirv_bn_mask_silu_p16s16_size = _binary_bn_mask_silu_p16s16_size;

  // sum_channels_fp32
  const unsigned char* spirv_sum_channels_fp32 = _binary_sum_channels_fp32_start;
  size_t spirv_sum_channels_fp32_size = _binary_sum_channels_fp32_size;

  // sum_channels_p32s16
  const unsigned char* spirv_sum_channels_p32s16 = _binary_sum_channels_p32s16_start;
  size_t spirv_sum_channels_p32s16_size = _binary_sum_channels_p32s16_size;

  // global_pooling_channels_fp32
  const unsigned char* spirv_global_pooling_channels_fp32 = _binary_global_pooling_channels_fp32_start;
  size_t spirv_global_pooling_channels_fp32_size = _binary_global_pooling_channels_fp32_size;

  // global_pooling_channels_p32s16
  const unsigned char* spirv_global_pooling_channels_p32s16 = _binary_global_pooling_channels_p32s16_start;
  size_t spirv_global_pooling_channels_p32s16_size = _binary_global_pooling_channels_p32s16_size;

  // value_head_pool_channels_fp32
  const unsigned char* spirv_value_head_pool_channels_fp32 = _binary_value_head_pool_channels_fp32_start;
  size_t spirv_value_head_pool_channels_fp32_size = _binary_value_head_pool_channels_fp32_size;

  // value_head_pool_channels_p32s16
  const unsigned char* spirv_value_head_pool_channels_p32s16 = _binary_value_head_pool_channels_p32s16_start;
  size_t spirv_value_head_pool_channels_p32s16_size = _binary_value_head_pool_channels_p32s16_size;

  // add_channel_bias_nchw_fp32
  const unsigned char* spirv_add_channel_bias_nchw_fp32 = _binary_add_channel_bias_nchw_fp32_start;
  size_t spirv_add_channel_bias_nchw_fp32_size = _binary_add_channel_bias_nchw_fp32_size;
  const unsigned char* spirv_add_channel_bias_nchw_p32s16 = _binary_add_channel_bias_nchw_p32s16_start;
  size_t spirv_add_channel_bias_nchw_p32s16_size = _binary_add_channel_bias_nchw_p32s16_size;
  const unsigned char* spirv_add_channel_bias_nchw_p16s16 = _binary_add_channel_bias_nchw_p16s16_start;
  size_t spirv_add_channel_bias_nchw_p16s16_size = _binary_add_channel_bias_nchw_p16s16_size;

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


  // add_channel_bias_nc_silu_fp32
  const unsigned char* spirv_add_channel_bias_nc_silu_fp32 = _binary_add_channel_bias_nc_silu_fp32_start;
  size_t spirv_add_channel_bias_nc_silu_fp32_size = _binary_add_channel_bias_nc_silu_fp32_size;


  // extract_channel0_nchw_fp32
  const unsigned char* spirv_extract_channel0_nchw_fp32 = _binary_extract_channel0_nchw_fp32_start;
  size_t spirv_extract_channel0_nchw_fp32_size = _binary_extract_channel0_nchw_fp32_size;

  // extract_channel0_nchw_p32s16
  const unsigned char* spirv_extract_channel0_nchw_p32s16 = _binary_extract_channel0_nchw_p32s16_start;
  size_t spirv_extract_channel0_nchw_p32s16_size = _binary_extract_channel0_nchw_p32s16_size;

  // extract_channel0_nchw_p16s16
  const unsigned char* spirv_extract_channel0_nchw_p16s16 = _binary_extract_channel0_nchw_p16s16_start;
  size_t spirv_extract_channel0_nchw_p16s16_size = _binary_extract_channel0_nchw_p16s16_size;

  // transformer_rms_norm_fp32
  const unsigned char* spirv_transformer_rms_norm_fp32 = _binary_transformer_rms_norm_fp32_start;
  size_t spirv_transformer_rms_norm_fp32_size = _binary_transformer_rms_norm_fp32_size;
  const unsigned char* spirv_transformer_rms_norm_p32s16 = _binary_transformer_rms_norm_p32s16_start;
  size_t spirv_transformer_rms_norm_p32s16_size = _binary_transformer_rms_norm_p32s16_size;
  const unsigned char* spirv_transformer_rms_norm_p16s16 = _binary_transformer_rms_norm_p16s16_start;
  size_t spirv_transformer_rms_norm_p16s16_size = _binary_transformer_rms_norm_p16s16_size;

  // transformer_rope_fp32
  const unsigned char* spirv_transformer_apply_rope_fp32 = _binary_transformer_apply_rope_fp32_start;
  size_t spirv_transformer_apply_rope_fp32_size = _binary_transformer_apply_rope_fp32_size;
  const unsigned char* spirv_transformer_apply_rope_p32s16 = _binary_transformer_apply_rope_p32s16_start;
  size_t spirv_transformer_apply_rope_p32s16_size = _binary_transformer_apply_rope_p32s16_size;
  const unsigned char* spirv_transformer_apply_rope_p16s16 = _binary_transformer_apply_rope_p16s16_start;
  size_t spirv_transformer_apply_rope_p16s16_size = _binary_transformer_apply_rope_p16s16_size;

  // transformer_scale_dot_product_fp32

  const unsigned char* spirv_transformer_scale_dot_product_fp32 = _binary_transformer_scale_dot_product_fp32_start;
  size_t spirv_transformer_scale_dot_product_fp32_size = _binary_transformer_scale_dot_product_fp32_size;
  const unsigned char* spirv_transformer_scale_dot_product_p32s16 = _binary_transformer_scale_dot_product_p32s16_start;
  size_t spirv_transformer_scale_dot_product_p32s16_size = _binary_transformer_scale_dot_product_p32s16_size;
  const unsigned char* spirv_transformer_scale_dot_product_p16s16 = _binary_transformer_scale_dot_product_p16s16_start;
  size_t spirv_transformer_scale_dot_product_p16s16_size = _binary_transformer_scale_dot_product_p16s16_size;

  // transformer_scale_dot_product_naive_fp32
  const unsigned char* spirv_transformer_scale_dot_product_naive_fp32 = _binary_transformer_scale_dot_product_naive_fp32_start;
  size_t spirv_transformer_scale_dot_product_naive_fp32_size = _binary_transformer_scale_dot_product_naive_fp32_size;
  const unsigned char* spirv_transformer_scale_dot_product_naive_p32s16 = _binary_transformer_scale_dot_product_naive_p32s16_start;
  size_t spirv_transformer_scale_dot_product_naive_p32s16_size = _binary_transformer_scale_dot_product_naive_p32s16_size;
  const unsigned char* spirv_transformer_scale_dot_product_naive_p16s16 = _binary_transformer_scale_dot_product_naive_p16s16_start;
  size_t spirv_transformer_scale_dot_product_naive_p16s16_size = _binary_transformer_scale_dot_product_naive_p16s16_size;

  // transformer_swiglu_fp32
  const unsigned char* spirv_transformer_swiglu_fp32 = _binary_transformer_swiglu_fp32_start;
  size_t spirv_transformer_swiglu_fp32_size = _binary_transformer_swiglu_fp32_size;
  const unsigned char* spirv_transformer_swiglu_p32s16 = _binary_transformer_swiglu_p32s16_start;
  size_t spirv_transformer_swiglu_p32s16_size = _binary_transformer_swiglu_p32s16_size;
  const unsigned char* spirv_transformer_swiglu_p16s16 = _binary_transformer_swiglu_p16s16_start;
  size_t spirv_transformer_swiglu_p16s16_size = _binary_transformer_swiglu_p16s16_size;

  // transformer_spatial_rms_norm_apply_fp32
  const unsigned char* spirv_transformer_spatial_rms_norm_apply_fp32 = _binary_transformer_spatial_rms_norm_apply_fp32_start;
  size_t spirv_transformer_spatial_rms_norm_apply_fp32_size = _binary_transformer_spatial_rms_norm_apply_fp32_size;
  const unsigned char* spirv_transformer_spatial_rms_norm_apply_p32s16 = _binary_transformer_spatial_rms_norm_apply_p32s16_start;
  size_t spirv_transformer_spatial_rms_norm_apply_p32s16_size = _binary_transformer_spatial_rms_norm_apply_p32s16_size;
  const unsigned char* spirv_transformer_spatial_rms_norm_apply_p16s16 = _binary_transformer_spatial_rms_norm_apply_p16s16_start;
  size_t spirv_transformer_spatial_rms_norm_apply_p16s16_size = _binary_transformer_spatial_rms_norm_apply_p16s16_size;

  // transformer_spatial_rms_norm_reduce_fp32
  const unsigned char* spirv_transformer_spatial_rms_norm_reduce_fp32 = _binary_transformer_spatial_rms_norm_reduce_fp32_start;
  size_t spirv_transformer_spatial_rms_norm_reduce_fp32_size = _binary_transformer_spatial_rms_norm_reduce_fp32_size;
  const unsigned char* spirv_transformer_spatial_rms_norm_reduce_p32s16 = _binary_transformer_spatial_rms_norm_reduce_p32s16_start;
  size_t spirv_transformer_spatial_rms_norm_reduce_p32s16_size = _binary_transformer_spatial_rms_norm_reduce_p32s16_size;
  const unsigned char* spirv_transformer_spatial_rms_norm_reduce_p16s16 = _binary_transformer_spatial_rms_norm_reduce_p16s16_start;
  size_t spirv_transformer_spatial_rms_norm_reduce_p16s16_size = _binary_transformer_spatial_rms_norm_reduce_p16s16_size;

  // transformer_spatial_rms_norm_sum_sq_fp32
  const unsigned char* spirv_transformer_spatial_rms_norm_sum_sq_fp32 = _binary_transformer_spatial_rms_norm_sum_sq_fp32_start;
  size_t spirv_transformer_spatial_rms_norm_sum_sq_fp32_size = _binary_transformer_spatial_rms_norm_sum_sq_fp32_size;
  const unsigned char* spirv_transformer_spatial_rms_norm_sum_sq_p32s16 = _binary_transformer_spatial_rms_norm_sum_sq_p32s16_start;
  size_t spirv_transformer_spatial_rms_norm_sum_sq_p32s16_size = _binary_transformer_spatial_rms_norm_sum_sq_p32s16_size;
  const unsigned char* spirv_transformer_spatial_rms_norm_sum_sq_p16s16 = _binary_transformer_spatial_rms_norm_sum_sq_p16s16_start;
  size_t spirv_transformer_spatial_rms_norm_sum_sq_p16s16_size = _binary_transformer_spatial_rms_norm_sum_sq_p16s16_size;

  ComputePipelines::ComputePipelines(VkDevice device_, Logger* logger_): device(device_), logger(logger_) {
    VkResult res = VK_ERROR_UNKNOWN;
    cache = vk_helper::createPipelineCache(device, &res);
    if(res != VK_SUCCESS)
      throw StringError("Failed to create Vulkan pipeline cache: " + vk_helper::vkErrorToString(res));
  }

  ComputePipelines::~ComputePipelines() {
    vkDeviceWaitIdle(device);
    destroyPipelines();
    if( cache != VK_NULL_HANDLE ) {
      vkDestroyPipelineCache(device, cache, nullptr);
      cache = VK_NULL_HANDLE;
    }
  }

  VkResult ComputePipelines::createPipelines(const VulkanTuneParams& tuneParams, int qHeadDim, int vHeadDim) {
    VkResult result;
    // Tile base conv no longer used.
    // createConv2dFp32();
    if((result = createWinogradInputTransform(winogradInputTransform3x3, tuneParams.conv3x3, 3, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createWinogradInputTransform(winogradInputTransform5x5, tuneParams.conv5x5, 5, tuneParams.vulkan)) != VK_SUCCESS) return result;
    struct PipelineActivation { Pipeline* pipeline; int activation; };
    const PipelineActivation winogradBnActPipelines[] = {
      {&winogradInputTransform3x3_bnact_identity, ACTIVATION_IDENTITY},
      {&winogradInputTransform3x3_bnact_relu, ACTIVATION_RELU},
      {&winogradInputTransform3x3_bnact_mish, ACTIVATION_MISH},
      {&winogradInputTransform3x3_bnact_mish_scale8, ACTIVATION_MISH_SCALE8},
      {&winogradInputTransform5x5_bnact_identity, ACTIVATION_IDENTITY},
      {&winogradInputTransform5x5_bnact_relu, ACTIVATION_RELU},
      {&winogradInputTransform5x5_bnact_mish, ACTIVATION_MISH},
      {&winogradInputTransform5x5_bnact_mish_scale8, ACTIVATION_MISH_SCALE8},
    };
    for(int i = 0; i < 8; i++) {
      const int convSize = i < 4 ? 3 : 5;
      const ConvTuneParams& convParams = convSize == 3 ? tuneParams.conv3x3 : tuneParams.conv5x5;
      if((result = createWinogradInputTransformBnAct(*winogradBnActPipelines[i].pipeline, convParams, convSize, winogradBnActPipelines[i].activation, tuneParams.vulkan)) != VK_SUCCESS) return result;
    }
    if((result = createWinogradOutputTransform(winogradOutputTransform3x3, tuneParams.conv3x3, 3, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createWinogradOutputTransform(winogradOutputTransform5x5, tuneParams.conv5x5, 5, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createAddPointWise(addPointWise, tuneParams.pointwise, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createXgemmDirectBatchedTT(xgemmDirectBatchedTT, tuneParams.xgemmDirect, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createXgemmBatched(xgemmBatchedFp32, tuneParams.xgemm, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createXgemmStridedBatched(xgemmStridedBatchedFp32, tuneParams.xgemmDirect, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createBatchNormMaskIdentity(batchNormMaskIdentity, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createBatchNormMaskRelu(batchNormMaskRelu, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createBatchNormMaskMish(batchNormMaskMish, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createBatchNormMaskMishScale8(batchNormMaskMishScale8, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createBatchNormMaskSilu(batchNormMaskSilu, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createGlobalPoolingChannelsFp32(globalPoolingChannelsFp32, tuneParams.gPool, tuneParams.vulkan)) != VK_SUCCESS) return result;
    for(uint32_t localSizeY = 1; localSizeY <= static_cast<uint32_t>(tuneParams.gPool.CHANNELSTRIDE); localSizeY *= 2) {
      for(uint32_t localSizeZ = 1; localSizeZ <= static_cast<uint32_t>(tuneParams.gPool.BATCHSTRIDE); localSizeZ *= 2) {
        LocalDim dim = {tuneParams.gPool.XYSTRIDE, static_cast<int>(localSizeY), static_cast<int>(localSizeZ)};
        Pipeline pipeline;
        if((result = createValueHeadPoolingChannels(pipeline, tuneParams.gPool, localSizeY, localSizeZ, tuneParams.vulkan)) != VK_SUCCESS) return result;
        valueHeadPoolingChannels.emplace(dim, pipeline);
      }
    }
    for(uint32_t localSizeZ = 1; localSizeZ <= static_cast<uint32_t>(tuneParams.gPool.BATCHSTRIDE); localSizeZ *= 2) {
      LocalDim dim = {tuneParams.gPool.XYSTRIDE, 1, static_cast<int>(localSizeZ)};
      Pipeline pipeline;
      if((result = createSumChannels(pipeline, tuneParams.gPool, localSizeZ, tuneParams.vulkan)) != VK_SUCCESS) return result;
      sumChannels.emplace(dim, pipeline);
    }
    if((result = createAddChannelBiasNCHW(addChannelBiasNCHW, tuneParams.addChannelBiases, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createAddChannelBiasNCIdentity(addChannelBiasNCIdentity, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createAddChannelBiasNCRelu(addChannelBiasNCRelu, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createAddChannelBiasNCMish(addChannelBiasNCMish, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createAddChannelBiasNCMishScale8(addChannelBiasNCMishScale8, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createAddChannelBiasNCSilu(addChannelBiasNCSilu, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createExtractChannel0NCHWFp32(extractChannel0NCHWFp32, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createTransformerRMSNorm(transformerRmsNorm, tuneParams.rmsNorm, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createTransformerApplyRoPE(transformerApplyRoPE, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createTransformerSwiGLU(transformerSwiGLU, tuneParams.pointwise, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createTransformerSpatialRMSNormApply(transformerSpatialRMSNormApply, tuneParams.spatialRMSNorm, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createTransformerSpatialRMSNormReduce(transformerSpatialRMSNormReduce, tuneParams.spatialRMSNorm, tuneParams.vulkan)) != VK_SUCCESS) return result;
    if((result = createTransformerSpatialRMSNormSumSq(transformerSpatialRMSNormSumSq, tuneParams.spatialRMSNorm, tuneParams.vulkan)) != VK_SUCCESS) return result;

    if ( qHeadDim != -1 && vHeadDim != -1 ) {
      if((result = createTransformerScaleDotProduct(transformerScaleDotProduct, tuneParams.transformer, qHeadDim, vHeadDim, tuneParams.vulkan)) != VK_SUCCESS) return result;
      if((result = createTransformerScaleDotProductNaive(transformerScaleDotProductNaive, qHeadDim, vHeadDim, tuneParams.vulkan)) != VK_SUCCESS) return result;
    }
    return VK_SUCCESS;
  }

  void ComputePipelines::destroyPipelines() {
    destroyPipeline(conv2dFp32);
    destroyPipeline(addPointWise);
    destroyPipeline(winogradInputTransform3x3);
    destroyPipeline(winogradInputTransform5x5);
    destroyPipeline(winogradInputTransform3x3_bnact_identity);
    destroyPipeline(winogradInputTransform3x3_bnact_relu);
    destroyPipeline(winogradInputTransform3x3_bnact_mish);
    destroyPipeline(winogradInputTransform3x3_bnact_mish_scale8);
    destroyPipeline(winogradInputTransform5x5_bnact_identity);
    destroyPipeline(winogradInputTransform5x5_bnact_relu);
    destroyPipeline(winogradInputTransform5x5_bnact_mish);
    destroyPipeline(winogradInputTransform5x5_bnact_mish_scale8);
    destroyPipeline(winogradOutputTransform3x3);
    destroyPipeline(winogradOutputTransform5x5);
    destroyPipeline(xgemmDirectBatchedTT);
    destroyPipeline(xgemmStridedBatchedFp32);
    destroyPipeline(xgemmBatchedFp32);
    destroyPipeline(batchNormMaskIdentity);
    destroyPipeline(batchNormMaskRelu);
    destroyPipeline(batchNormMaskMish);
    destroyPipeline(batchNormMaskMishScale8);
    destroyPipeline(batchNormMaskSilu);
    destroyPipeline(globalPoolingChannelsFp32);

    for ( auto it : valueHeadPoolingChannels ) {
      destroyPipeline(it.second);
    }
    valueHeadPoolingChannels.clear();
    for ( auto it : sumChannels ) {
      destroyPipeline(it.second);
    }
    sumChannels.clear();
    destroyPipeline(addChannelBiasNCHW);
    destroyPipeline(addChannelBiasNCIdentity);
    destroyPipeline(addChannelBiasNCRelu);
    destroyPipeline(addChannelBiasNCMish);
    destroyPipeline(addChannelBiasNCMishScale8);
    destroyPipeline(addChannelBiasNCSilu);
    destroyPipeline(extractChannel0NCHWFp32);

    destroyPipeline(transformerRmsNorm);
    destroyPipeline(transformerApplyRoPE);
    destroyPipeline(transformerScaleDotProduct);
    destroyPipeline(transformerScaleDotProductNaive);
    destroyPipeline(transformerSwiGLU);
    destroyPipeline(transformerSpatialRMSNormApply);
    destroyPipeline(transformerSpatialRMSNormReduce);
    destroyPipeline(transformerSpatialRMSNormSumSq);
  }

  /**
   * @brief cleanup pipeline resources
   */
  void ComputePipelines::destroyPipeline(Pipeline& pipeline) {
    if( pipeline.pipeline != VK_NULL_HANDLE ) {
      vkDestroyPipeline(device, pipeline.pipeline, nullptr);
      pipeline.pipeline = VK_NULL_HANDLE;
    }

    if ( pipeline.layout != VK_NULL_HANDLE ) {
      vkDestroyPipelineLayout(device, pipeline.layout, nullptr);
      pipeline.layout = VK_NULL_HANDLE;
    }

    if ( pipeline.descriptorSetLayout != VK_NULL_HANDLE ) {
      vkDestroyDescriptorSetLayout(device, pipeline.descriptorSetLayout, nullptr);
      pipeline.descriptorSetLayout = VK_NULL_HANDLE;
    }
    pipeline.name.clear();
  }

  VkResult ComputePipelines::createPipeline(
    std::string pipelineName,
    const unsigned char* spirvBytes,
    size_t spirvSize,
    size_t bindingSize,
    uint32_t pushConstantSize,
    Pipeline &outPipeline,
    VkSpecializationInfo* specializationInfo,
    uint32_t localSizeX,
    uint32_t localSizeY,
    uint32_t localSizeZ
  ) {
    VkResult res = VK_ERROR_UNKNOWN;
    if(spirvSize % 4 != 0)
      return VK_ERROR_INITIALIZATION_FAILED;

    size_t numWords = spirvSize / 4;
    std::vector<uint32_t> spirvWords(numWords);
    // Safe memcpy to properly align bytes into 32-bit words
    memcpy(spirvWords.data(), spirvBytes, spirvSize);

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const auto fail = [&](VkResult result) {
      if(pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device,pipeline,nullptr);
      if(layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device,layout,nullptr);
      if(descriptorSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device,descriptorSetLayout,nullptr);
      if(shaderModule != VK_NULL_HANDLE)
        vkDestroyShaderModule(device,shaderModule,nullptr);
      return result;
    };
    VkShaderModuleCreateInfo shaderModuleCI = {};
    shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCI.codeSize = spirvSize;
    shaderModuleCI.pCode = spirvWords.data();
    // std::cout << "Creating Compute Pipeline: " << pipelineName <<  " code size : " << shaderModuleCI.codeSize << std::endl;
    res = vkCreateShaderModule(device, &shaderModuleCI, nullptr, &shaderModule);
    if(res != VK_SUCCESS)
      return fail(res);
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for ( size_t i = 0 ; i < bindingSize ; i++ ) {
      bindings.push_back(
        vk_helper::descriptorSetLayoutBinding(
          static_cast<uint32_t>(i),
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        )
      );
    }

    descriptorSetLayout = vk_helper::createDescriptorSetLayout(device,bindings,&res);
    if(res != VK_SUCCESS)
      return fail(res);

    std::vector<VkPushConstantRange> pushConstants;
    VkPushConstantRange pushConstant = {};

    if ( pushConstantSize > 0 ) {
      pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      pushConstant.offset = 0;
      pushConstant.size = static_cast<uint32_t>(pushConstantSize);
      pushConstants.push_back(pushConstant);
    }
    layout = vk_helper::createPipelineLayout(device,{ descriptorSetLayout }, pushConstants, &res);
    if(res != VK_SUCCESS)
      return fail(res);

    pipeline = vk_helper::createComputePipeline(device, layout, cache, shaderModule, &res, specializationInfo);

    if(res != VK_SUCCESS)
      return fail(res);
    outPipeline.descriptorSetLayout = descriptorSetLayout;
    outPipeline.layout = layout;
    outPipeline.pipeline = pipeline;
    outPipeline.name = pipelineName;
    outPipeline.localSizeX = localSizeX;
    outPipeline.localSizeY = localSizeY;
    outPipeline.localSizeZ = localSizeZ;
    const std::string message = "Built Vulkan compute pipeline with shader: " + pipelineName;
    if(logger != nullptr)
      logger->write(message);
    if(logger == nullptr || (!logger->isLoggingToStdout() && !logger->isLoggingToStderr()))
      std::cerr << message << std::endl;
    vkDestroyShaderModule(device, shaderModule, nullptr);
    return VK_SUCCESS;
  }

  /**
   * @brief Create a Conv2d Fp32 object
  */
  // void ComputePipelines::createConv2dFp32() {
    // auto spec = Conv2DSpec();
    // SpecializationData specData(spec);
    // createPipeline("Conv2dFp32", vk_shader::spirv_conv2d_fp32, vk_shader::spirv_conv2d_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2dFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  // }

  VkResult ComputePipelines::createWinogradInputTransform(Pipeline& pipeline, const ConvTuneParams& tuneParams, int convSize, const VulkanParams& vulkanParams) {
    WinogradInputTransformSpec spec;
    spec.localSizeX = tuneParams.inputTransformLocalXSize;
    spec.localSizeY = tuneParams.inputTransformLocalYSize;
    spec.localSizeZ = 1;
    spec.inTileYSize = static_cast<int>(tuneParams.inTileYSize);
    spec.inTileXSize = static_cast<int>(tuneParams.inTileXSize);
    spec.outTileYSize = static_cast<int>(tuneParams.outTileYSize);
    spec.outTileXSize = static_cast<int>(tuneParams.outTileXSize);
    spec.inTileYOffset = -convSize / 2;
    spec.inTileXOffset = -convSize / 2;
    spec.convY = convSize;
    spec.convX = convSize;
    std::vector<int32_t> specData = vk_helper::createSpecData(&spec, sizeof(spec));
    std::vector<VkSpecializationMapEntry> specEntry = vk_helper::createSpecMapEntries(specData.size());
    VkSpecializationInfo specializationInfo = vk_helper::createSpecializationInfo(specData, specEntry);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("winograd_input_transform_p16s16", spirv_winograd_input_transform_p16s16, spirv_winograd_input_transform_p16s16_size, 2, sizeof(WinogradInputTransformParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("winograd_input_transform_p32s16", spirv_winograd_input_transform_p32s16, spirv_winograd_input_transform_p32s16_size, 2, sizeof(WinogradInputTransformParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("winograd_input_transform_fp32", spirv_winograd_input_transform_fp32, spirv_winograd_input_transform_fp32_size, 2, sizeof(WinogradInputTransformParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createWinogradInputTransformBnAct(Pipeline& pipeline, const ConvTuneParams& tuneParams, int convSize, int activation, const VulkanParams& vulkanParams) {
    auto spec = WinogradInputTransformBnActSpec();
    spec.localSizeX = tuneParams.inputTransformLocalXSize;
    spec.localSizeY = tuneParams.inputTransformLocalYSize;
    spec.localSizeZ = 1;
    spec.inTileYSize = static_cast<int>(tuneParams.inTileYSize);
    spec.inTileXSize = static_cast<int>(tuneParams.inTileXSize);
    spec.outTileYSize = static_cast<int>(tuneParams.outTileYSize);
    spec.outTileXSize = static_cast<int>(tuneParams.outTileXSize);
    spec.inTileYOffset = -convSize / 2;
    spec.inTileXOffset = -convSize / 2;
    spec.convY = convSize;
    spec.convX = convSize;
    spec.activation = activation;
    std::vector<int32_t> specData = vk_helper::createSpecData(&spec, sizeof(spec));
    std::vector<VkSpecializationMapEntry> specEntry = vk_helper::createSpecMapEntries(specData.size());
    VkSpecializationInfo specializationInfo = vk_helper::createSpecializationInfo(specData, specEntry);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("winograd_input_transform_bnact_p16s16", spirv_winograd_input_transform_bnact_p16s16, spirv_winograd_input_transform_bnact_p16s16_size, 5, sizeof(WinogradInputTransformParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("winograd_input_transform_bnact_p32s16", spirv_winograd_input_transform_bnact_p32s16, spirv_winograd_input_transform_bnact_p32s16_size, 5, sizeof(WinogradInputTransformParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("winograd_input_transform_bnact_fp32", spirv_winograd_input_transform_bnact_fp32, spirv_winograd_input_transform_bnact_fp32_size, 5, sizeof(WinogradInputTransformParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createWinogradOutputTransform(Pipeline& pipeline, const ConvTuneParams& tuneParams, int convSize, const VulkanParams& vulkanParams) {
    WinogradOutputTransformSpec spec;
    spec.localSizeX = tuneParams.outputTransformLocalXSize;
    spec.localSizeY = tuneParams.outputTransformLocalYSize;
    spec.localSizeZ = tuneParams.outputTransformLocalZSize;
    spec.outTileXSize = tuneParams.outTileXSize;
    spec.outTileYSize = tuneParams.outTileYSize;
    spec.inTileXSize = tuneParams.inTileXSize;
    spec.inTileYSize = tuneParams.inTileYSize;
    spec.convX = convSize;
    spec.convY = convSize;
    std::vector<int32_t> specData = vk_helper::createSpecData(&spec, sizeof(spec));
    std::vector<VkSpecializationMapEntry> specEntry = vk_helper::createSpecMapEntries(specData.size());
    VkSpecializationInfo specializationInfo = vk_helper::createSpecializationInfo(specData, specEntry);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("winograd_output_transform_p16s16", spirv_winograd_output_transform_p16s16, spirv_winograd_output_transform_p16s16_size, 2, sizeof(WinogradOutputTransformParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("winograd_output_transform_p32s16", spirv_winograd_output_transform_p32s16, spirv_winograd_output_transform_p32s16_size, 2, sizeof(WinogradOutputTransformParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("winograd_output_transform_fp32", spirv_winograd_output_transform_fp32, spirv_winograd_output_transform_fp32_size, 2, sizeof(WinogradOutputTransformParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  /**
   * @brief Create a Conv2d3x3 Bn + Identity Activation fused Fp32 objects.
   */
  // void ComputePipelines::createConv2d3x3BnFp32() {
  //   createPipeline("Conv2d3x3BnFp32", vk_shader::spirv_conv2d_3x3_bn_fp32, vk_shader::spirv_conv2d_3x3_bn_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d3x3BnFp32);
  // }

  /**
   * @brief Create a Conv2d3x3 Bn + ReLU Activation fused Fp32 objects.
   */
  // void ComputePipelines::createConv2d3x3BnReluFp32() {
  //   createPipeline("Conv2d3x3BnReluFp32", vk_shader::spirv_conv2d_3x3_bn_relu_fp32, vk_shader::spirv_conv2d_3x3_bn_relu_fp32_size,   3, sizeof(Conv2DPushConstantParams), conv2d3x3BnReluFp32);
  // }
  /**
   * @brief Create a Conv2d3x3 Bn Mish Fp32 object
   */
  // void ComputePipelines::createConv2d3x3BnMishFp32() {
  //   createPipeline("Conv2d3x3BnMishFp32", vk_shader::spirv_conv2d_3x3_bn_mish_fp32,vk_shader::spirv_conv2d_3x3_bn_mish_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d3x3BnMishFp32);
  // }

  /**
   * @brief Create a Conv2d5x5 Bn + Identity Activation fused Fp32 objects.
   */
  // void ComputePipelines::createConv2d5x5BnFp32() {
  //   createPipeline("Conv2d5x5BnFp32", vk_shader::spirv_conv2d_5x5_bn_fp32, vk_shader::spirv_conv2d_5x5_bn_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnFp32);
  // }

  /**
   * @brief Create a Conv2d5x5 Bn + ReLU Activation fused Fp32 objects.
   */
  // void ComputePipelines::createConv2d5x5BnReluFp32() {
  //   createPipeline("Conv2d5x5BnReluFp32",vk_shader::spirv_conv2d_5x5_bn_relu_fp32, vk_shader::spirv_conv2d_5x5_bn_relu_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnReluFp32);
  // }

  /**
   * @brief Create a Conv2d5x5 Bn Mish Fp32 object
   */
  // void ComputePipelines::createConv2d5x5BnMishFp32() {
  //   createPipeline("Conv2d5x5BnMishFp32",vk_shader::spirv_conv2d_5x5_bn_mish_fp32, vk_shader::spirv_conv2d_5x5_bn_mish_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnMishFp32);
  // }

  /**
   * @brief Create a Add Point Wise object
   */
  VkResult ComputePipelines::createAddPointWise(Pipeline& pipeline, const AddPointWiseTuneParams& tuneParams, const VulkanParams& vulkanParams) {
    auto spec = AddPointWiseSpec();
    spec.ELTS_PER_THREAD = tuneParams.ELTS_PER_THREAD;
    spec.localSizeX = tuneParams.LOCAL_SIZE;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("add_pointwise_p16s16", spirv_add_pointwise_p16s16, spirv_add_pointwise_p16s16_size, 2, sizeof(AddPointWiseParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("add_pointwise_p32s16", spirv_add_pointwise_p32s16, spirv_add_pointwise_p32s16_size, 2, sizeof(AddPointWiseParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("add_pointwise_fp32", spirv_add_pointwise_fp32, spirv_add_pointwise_fp32_size, 2, sizeof(AddPointWiseParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createXgemmDirectBatchedTT(Pipeline& pipeline, const XgemmDirectTuneParams& tuneParams, const VulkanParams& vulkanParams [[maybe_unused]]) {
    auto spec = XgemmDirectSpec();
    spec.localSizeX = tuneParams.MDIMCD;
    spec.localSizeY = tuneParams.NDIMCD;
    spec.localSizeZ = 1;
    spec.WGD = static_cast<int>(tuneParams.WGD);
    spec.MDIMCD = static_cast<int>(tuneParams.MDIMCD);
    spec.NDIMCD = static_cast<int>(tuneParams.NDIMCD);
    spec.MDIMAD = static_cast<int>(tuneParams.MDIMAD);
    spec.NDIMBD = static_cast<int>(tuneParams.NDIMBD);
    spec.KWID = static_cast<int>(tuneParams.KWID);
    spec.PADA = static_cast<int>(tuneParams.PADA);
    spec.PADB = static_cast<int>(tuneParams.PADB);
    std::vector<VkSpecializationMapEntry> mapEntries = vk_helper::createSpecMapEntries(sizeof(spec) / sizeof(int32_t));
    std::vector<int32_t> specData = vk_helper::createSpecData(&spec, sizeof(spec));
    VkSpecializationInfo specializationInfo = vk_helper::createSpecializationInfo(specData, mapEntries);
    return createPipeline("xgemm_direct_batched_tt_fp32", vk_shader::spirv_xgemm_direct_batched_tt_fp32, vk_shader::spirv_xgemm_direct_batched_tt_fp32_size, 3, sizeof(XgemmDirectBatchedTTParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createXgemmBatched(Pipeline& pipeline, const XgemmTuneParams& tuneParams, const VulkanParams& vulkanParams) {

    auto spec = XGEMMBatchedSpec();
    spec.localSizeX = tuneParams.MDIMC;
    spec.localSizeY = tuneParams.NDIMC;
    spec.localSizeZ = 1;
    spec.MWG = tuneParams.MWG;
    spec.NWG = tuneParams.NWG;
    spec.KWG = tuneParams.KWG;
    spec.MDIMC = tuneParams.MDIMC;
    spec.NDIMC = tuneParams.NDIMC;
    spec.MDIMA = tuneParams.MDIMA;
    spec.NDIMB = tuneParams.NDIMB;
    std::vector<VkSpecializationMapEntry> mapEntries = vk_helper::createSpecMapEntries(sizeof(spec) / sizeof(int32_t));
    std::vector<int32_t> specData = vk_helper::createSpecData(&spec, sizeof(spec));
    VkSpecializationInfo specializationInfo = vk_helper::createSpecializationInfo(specData, mapEntries);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("xgemm_batched_p16s16", spirv_xgemm_batched_p16s16, spirv_xgemm_batched_p16s16_size, 3, sizeof(XGEMMBatchedParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("xgemm_batched_p32s16", spirv_xgemm_batched_p32s16, spirv_xgemm_batched_p32s16_size, 3, sizeof(XGEMMBatchedParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("xgemm_batched_fp32", spirv_xgemm_batched_fp32, spirv_xgemm_batched_fp32_size, 3, sizeof(XGEMMBatchedParams), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createXgemmStridedBatched(Pipeline& pipeline, const XgemmDirectTuneParams& tuneParams, const VulkanParams& vulkanParams) {
    auto spec = XgemmDirectSpec();
    spec.localSizeX = tuneParams.MDIMCD;
    spec.localSizeY = tuneParams.NDIMCD;
    spec.localSizeZ = 1;
    spec.WGD = static_cast<int>(tuneParams.WGD);
    spec.MDIMCD = static_cast<int>(tuneParams.MDIMCD);
    spec.NDIMCD = static_cast<int>(tuneParams.NDIMCD);
    spec.MDIMAD = static_cast<int>(tuneParams.MDIMAD);
    spec.NDIMBD = static_cast<int>(tuneParams.NDIMBD);
    spec.KWID = static_cast<int>(tuneParams.KWID);
    spec.PADA = static_cast<int>(tuneParams.PADA);
    spec.PADB = static_cast<int>(tuneParams.PADB);
    auto specData = vk_helper::createSpecData(&spec, sizeof(spec));
    auto mapEntries = vk_helper::createSpecMapEntries(sizeof(spec) / sizeof(int32_t));
    VkSpecializationInfo specializationInfo = vk_helper::createSpecializationInfo(specData, mapEntries);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("xgemm_strided_batched_nn_p16s16", spirv_xgemm_strided_batched_nn_p16s16, spirv_xgemm_strided_batched_nn_p16s16_size, 3, sizeof(XgemmStridedBatchedFp32Params), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("xgemm_strided_batched_nn_p32s16", spirv_xgemm_strided_batched_nn_p32s16, spirv_xgemm_strided_batched_nn_p32s16_size, 3, sizeof(XgemmStridedBatchedFp32Params), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("xgemm_strided_batched_nn_fp32", spirv_xgemm_strided_batched_nn_fp32, spirv_xgemm_strided_batched_nn_fp32_size, 3, sizeof(XgemmStridedBatchedFp32Params), pipeline, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createBatchNormMaskIdentity(Pipeline& pipeline, const VulkanParams& vulkanParams) {
    auto spec = BatchNormMaskSpec();
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("bn_mask_identity_p16s16", spirv_bn_mask_identity_p16s16, spirv_bn_mask_identity_p16s16_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("bn_mask_identity_p32s16", spirv_bn_mask_identity_p32s16, spirv_bn_mask_identity_p32s16_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("bn_mask_identity_fp32", spirv_bn_mask_identity_fp32, spirv_bn_mask_identity_fp32_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createBatchNormMaskRelu(Pipeline& pipeline, const VulkanParams& vulkanParams) {
    auto spec = BatchNormMaskSpec();
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("bn_mask_relu_p16s16", spirv_bn_mask_relu_p16s16, spirv_bn_mask_relu_p16s16_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("bn_mask_relu_p32s16", spirv_bn_mask_relu_p32s16, spirv_bn_mask_relu_p32s16_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("bn_mask_relu_fp32", spirv_bn_mask_relu_fp32, spirv_bn_mask_relu_fp32_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createBatchNormMaskMish(Pipeline& pipeline, const VulkanParams& vulkanParams) {
    auto spec = BatchNormMaskSpec();
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("bn_mask_mish_p16s16", spirv_bn_mask_mish_p16s16, spirv_bn_mask_mish_p16s16_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("bn_mask_mish_p32s16", spirv_bn_mask_mish_p32s16, spirv_bn_mask_mish_p32s16_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("bn_mask_mish_fp32", spirv_bn_mask_mish_fp32, spirv_bn_mask_mish_fp32_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createBatchNormMaskMishScale8(Pipeline& pipeline, const VulkanParams& vulkanParams) {
    auto spec = BatchNormMaskSpec();
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("bn_mask_mish_scale8_p16s16", spirv_bn_mask_mish_scale8_p16s16, spirv_bn_mask_mish_scale8_p16s16_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("bn_mask_mish_scale8_p32s16", spirv_bn_mask_mish_scale8_p32s16, spirv_bn_mask_mish_scale8_p32s16_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("bn_mask_mish_scale8_fp32", spirv_bn_mask_mish_scale8_fp32, spirv_bn_mask_mish_scale8_fp32_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createBatchNormMaskSilu(Pipeline& pipeline, const VulkanParams& vulkanParams) {
    auto spec = BatchNormMaskSpec();
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("bn_mask_silu_p16s16", spirv_bn_mask_silu_p16s16, spirv_bn_mask_silu_p16s16_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("bn_mask_silu_p32s16", spirv_bn_mask_silu_p32s16, spirv_bn_mask_silu_p32s16_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("bn_mask_silu_fp32", spirv_bn_mask_silu_fp32, spirv_bn_mask_silu_fp32_size, 5, sizeof(BatchNormMaskParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createGlobalPoolingChannelsFp32(Pipeline& pipeline, const GPoolTuneParams& tuneParams, const VulkanParams& vulkanParams) {
    // TODO: create multiple local sizes pipelines to optimize performance.
    auto spec = GlobalPoolingChannelsSpec();
    spec.localSizeX = tuneParams.XYSTRIDE;
    spec.localSizeY = tuneParams.CHANNELSTRIDE;
    spec.localSizeZ = tuneParams.BATCHSTRIDE;
    spec.XYSTRIDE = tuneParams.XYSTRIDE;
    spec.CHANNELSTRIDE = tuneParams.CHANNELSTRIDE;
    spec.LOCALSIZE_TOTAL = spec.XYSTRIDE * spec.CHANNELSTRIDE * tuneParams.BATCHSTRIDE;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage)
      return createPipeline("global_pooling_channels_p32s16", spirv_global_pooling_channels_p32s16, spirv_global_pooling_channels_p32s16_size, 4, sizeof(GlobalPoolingChannelsParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    return createPipeline("global_pooling_channels_fp32", spirv_global_pooling_channels_fp32, spirv_global_pooling_channels_fp32_size, 4, sizeof(GlobalPoolingChannelsParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createValueHeadPoolingChannels(Pipeline& pipeline, const GPoolTuneParams& tuneParams, uint32_t localSizeY, uint32_t localSizeZ, const VulkanParams& vulkanParams) {
    auto spec = ValueHeadPoolingChannelsSpec();
    spec.localSizeX = tuneParams.XYSTRIDE;
    spec.localSizeY = localSizeY;
    spec.localSizeZ = localSizeZ;
    spec.XYSTRIDE = tuneParams.XYSTRIDE;
    spec.CHANNELSTRIDE = tuneParams.CHANNELSTRIDE;
    spec.LOCALSIZE_TOTAL = tuneParams.XYSTRIDE * tuneParams.CHANNELSTRIDE * tuneParams.BATCHSTRIDE;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage)
      return createPipeline("value_head_pool_channels_p32s16", spirv_value_head_pool_channels_p32s16, spirv_value_head_pool_channels_p32s16_size, 3, sizeof(ValueHeadPoolingChannelsParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    return createPipeline("value_head_pool_channels_fp32", spirv_value_head_pool_channels_fp32, spirv_value_head_pool_channels_fp32_size, 3, sizeof(ValueHeadPoolingChannelsParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createSumChannels(Pipeline& pipeline, const GPoolTuneParams& tuneParams, uint32_t localSizeZ, const VulkanParams& vulkanParams) {
    auto spec = SumChannelsSpec();
    spec.CHANNELSTRIDE = 1;
    spec.XYSTRIDE = tuneParams.XYSTRIDE;
    spec.LOCALSIZE_TOTAL = tuneParams.BATCHSTRIDE * tuneParams.CHANNELSTRIDE * tuneParams.XYSTRIDE;
    spec.localSizeX = spec.XYSTRIDE;
    spec.localSizeY = 1;
    spec.localSizeZ = localSizeZ;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage)
      return createPipeline("sum_channels_p32s16", spirv_sum_channels_p32s16, spirv_sum_channels_p32s16_size, 2, sizeof(SumChannelsParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    return createPipeline("sum_channels_fp32", spirv_sum_channels_fp32, spirv_sum_channels_fp32_size, 2, sizeof(SumChannelsParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createAddChannelBiasNCHW(Pipeline& pipeline, const AddChannelBiasesNCHWTuneParams& tuneParams, const VulkanParams& vulkanParams) {
    auto spec = AddChannelBiasNCHWSpec();
    spec.XY_ELTS_PER_THREAD = tuneParams.XY_ELTS_PER_THREAD;
    spec.NC_ELTS_PER_THREAD = tuneParams.NC_ELTS_PER_THREAD;
    spec.localSizeX=32;
    spec.localSizeY=1;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("add_channel_bias_nchw_p16s16", spirv_add_channel_bias_nchw_p16s16, spirv_add_channel_bias_nchw_p16s16_size, 2, sizeof(AddChannelBiasNCHWParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("add_channel_bias_nchw_p32s16", spirv_add_channel_bias_nchw_p32s16, spirv_add_channel_bias_nchw_p32s16_size, 2, sizeof(AddChannelBiasNCHWParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("add_channel_bias_nchw_fp32", spirv_add_channel_bias_nchw_fp32, spirv_add_channel_bias_nchw_fp32_size, 2, sizeof(AddChannelBiasNCHWParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createAddChannelBiasNCIdentity(Pipeline& pipeline, const VulkanParams& vulkanParams [[maybe_unused]]) {
    auto spec = AddChannelBiasNCSpec();
    SpecializationData specData(spec);
    return createPipeline("add_channel_bias_nc_identity_fp32", vk_shader::spirv_add_channel_bias_nc_identity_fp32, vk_shader::spirv_add_channel_bias_nc_identity_fp32_size, 2, sizeof(AddChannelBiasNCParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createAddChannelBiasNCRelu(Pipeline& pipeline, const VulkanParams& vulkanParams [[maybe_unused]]) {
    auto spec = AddChannelBiasNCSpec();
    SpecializationData specData(spec);
    return createPipeline("add_channel_bias_nc_relu_fp32", vk_shader::spirv_add_channel_bias_nc_relu_fp32, vk_shader::spirv_add_channel_bias_nc_relu_fp32_size, 2, sizeof(AddChannelBiasNCParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createAddChannelBiasNCMish(Pipeline& pipeline, const VulkanParams& vulkanParams [[maybe_unused]]) {
    auto spec = AddChannelBiasNCSpec();
    SpecializationData specData(spec);
    return createPipeline("add_channel_bias_nc_mish_fp32", vk_shader::spirv_add_channel_bias_nc_mish_fp32, vk_shader::spirv_add_channel_bias_nc_mish_fp32_size, 2, sizeof(AddChannelBiasNCParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createAddChannelBiasNCMishScale8(Pipeline& pipeline, const VulkanParams& vulkanParams [[maybe_unused]]) {
    auto spec = AddChannelBiasNCSpec();
    SpecializationData specData(spec);
    return createPipeline("add_channel_bias_nc_mish_scale8_fp32", vk_shader::spirv_add_channel_bias_nc_mish_scale8_fp32, vk_shader::spirv_add_channel_bias_nc_mish_scale8_fp32_size, 2, sizeof(AddChannelBiasNCParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createAddChannelBiasNCSilu(Pipeline& pipeline, const VulkanParams& vulkanParams [[maybe_unused]]) {
    auto spec = AddChannelBiasNCSpec();
    SpecializationData specData(spec);
    return createPipeline("add_channel_bias_nc_silu_fp32", vk_shader::spirv_add_channel_bias_nc_silu_fp32, vk_shader::spirv_add_channel_bias_nc_silu_fp32_size, 2, sizeof(AddChannelBiasNCParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createExtractChannel0NCHWFp32(Pipeline& pipeline, const VulkanParams& vulkanParams) {
    auto spec = ExtractChannel0NCHWSpec();
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("extract_channel0_nchw_p16s16", spirv_extract_channel0_nchw_p16s16, spirv_extract_channel0_nchw_p16s16_size, 2, sizeof(ExtractChannel0NCHWParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("extract_channel0_nchw_p32s16", spirv_extract_channel0_nchw_p32s16, spirv_extract_channel0_nchw_p32s16_size, 2, sizeof(ExtractChannel0NCHWParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("extract_channel0_nchw_fp32", spirv_extract_channel0_nchw_fp32, spirv_extract_channel0_nchw_fp32_size, 2, sizeof(ExtractChannel0NCHWParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createTransformerRMSNorm(Pipeline& pipeline, const TransformerRMSNormTuneParms& tuneParams, const VulkanParams& vulkanParams) {
    auto spec = TransformerRMSNormSpec();
    spec.localSizeX = tuneParams.WG_C_SIZE * tuneParams.WG_XY_SIZE;
    spec.localSizeY = 1;
    spec.localSizeZ = 1;
    spec.C_PER_THREAD = tuneParams.C_PER_THREAD;
    spec.WG_C_SIZE = tuneParams.WG_C_SIZE;
    spec.WG_XY_SIZE = tuneParams.WG_XY_SIZE;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("transformer_rms_norm_p16s16", spirv_transformer_rms_norm_p16s16, spirv_transformer_rms_norm_p16s16_size, 5, sizeof(TransformerRMSNormPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("transformer_rms_norm_p32s16", spirv_transformer_rms_norm_p32s16, spirv_transformer_rms_norm_p32s16_size, 5, sizeof(TransformerRMSNormPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("transformer_rms_norm_fp32", spirv_transformer_rms_norm_fp32, spirv_transformer_rms_norm_fp32_size, 5, sizeof(TransformerRMSNormPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createTransformerApplyRoPE(Pipeline& pipeline, const VulkanParams& vulkanParams) {
    auto spec = TransformerApplyRoPESpec();
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("transformer_apply_rope_p16s16", spirv_transformer_apply_rope_p16s16, spirv_transformer_apply_rope_p16s16_size, 3, sizeof(TransformerApplyRoPEPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("transformer_apply_rope_p32s16", spirv_transformer_apply_rope_p32s16, spirv_transformer_apply_rope_p32s16_size, 3, sizeof(TransformerApplyRoPEPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("transformer_apply_rope_fp32", spirv_transformer_apply_rope_fp32, spirv_transformer_apply_rope_fp32_size, 3, sizeof(TransformerApplyRoPEPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createTransformerScaleDotProduct(Pipeline& pipeline, const TransformerTuneParams& tuneParams, int qHeadDim, int vHeadDim, const VulkanParams& vulkanParams) {
    auto spec = ScaleDotProductSpec();
    spec.ATTN_BLOCK_KV = tuneParams.ATTN_BLOCK_KV;
    spec.ATTN_BLOCK_Q = tuneParams.ATTN_BLOCK_Q;
    spec.ATTN_HEAD_DIM = qHeadDim;
    spec.ATTN_V_HEAD_DIM = vHeadDim;
    spec.Q_PER_THREAD = tuneParams.Q_PER_THREAD;
    spec.localSizeX = spec.ATTN_BLOCK_Q;
    spec.localSizeY = 1;
    spec.localSizeZ = 1;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("transformer_scale_dot_product_p16s16", spirv_transformer_scale_dot_product_p16s16, spirv_transformer_scale_dot_product_p16s16_size, 5, sizeof(ScaleDotProductPushParam), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("transformer_scale_dot_product_p32s16", spirv_transformer_scale_dot_product_p32s16, spirv_transformer_scale_dot_product_p32s16_size, 5, sizeof(ScaleDotProductPushParam), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("transformer_scale_dot_product_fp32", spirv_transformer_scale_dot_product_fp32, spirv_transformer_scale_dot_product_fp32_size, 5, sizeof(ScaleDotProductPushParam), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createTransformerScaleDotProductNaive(Pipeline& pipeline, int qHeadDim, int vHeadDim, const VulkanParams& vulkanParams) {
    auto spec = ScaleDotProductNaiveSpec();
    spec.ATTN_HEAD_DIM = qHeadDim;
    spec.ATTN_V_HEAD_DIM = vHeadDim;
    spec.localSizeX = 32;
    spec.localSizeY = 1;
    spec.localSizeZ = 1;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("transformer_scale_dot_product_naive_p16s16", spirv_transformer_scale_dot_product_naive_p16s16, spirv_transformer_scale_dot_product_naive_p16s16_size, 5, sizeof(ScaleDotProductPushParam), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("transformer_scale_dot_product_naive_p32s16", spirv_transformer_scale_dot_product_naive_p32s16, spirv_transformer_scale_dot_product_naive_p32s16_size, 5, sizeof(ScaleDotProductPushParam), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("transformer_scale_dot_product_naive_fp32", spirv_transformer_scale_dot_product_naive_fp32, spirv_transformer_scale_dot_product_naive_fp32_size, 5, sizeof(ScaleDotProductPushParam), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createTransformerSwiGLU(Pipeline& pipeline, const AddPointWiseTuneParams& tuneParams, const VulkanParams& vulkanParams) {
    auto spec = TransformerSwiGLUSpec();
    spec.ELTS_PER_THREAD = tuneParams.ELTS_PER_THREAD;
    spec.localSizeX = tuneParams.LOCAL_SIZE;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("transformer_swiglu_p16s16", spirv_transformer_swiglu_p16s16, spirv_transformer_swiglu_p16s16_size, 3, sizeof(TransformerSwiGLUPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("transformer_swiglu_p32s16", spirv_transformer_swiglu_p32s16, spirv_transformer_swiglu_p32s16_size, 3, sizeof(TransformerSwiGLUPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("transformer_swiglu_fp32", spirv_transformer_swiglu_fp32, spirv_transformer_swiglu_fp32_size, 3, sizeof(TransformerSwiGLUPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createTransformerSpatialRMSNormApply(Pipeline& pipeline, const TransformerSpatialRmsNormTuneParams& tuneParams, const VulkanParams& vulkanParams) {
    auto spec = TransformerSpatialRMSNormApplySpec();
    spec.APPLY_ELTS_PER_THREAD = tuneParams.APPLY_ELTS_PER_THREAD;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("transformer_spatial_rms_norm_apply_p16s16", spirv_transformer_spatial_rms_norm_apply_p16s16, spirv_transformer_spatial_rms_norm_apply_p16s16_size, 7, sizeof(TransformerSpatialRMSNormApplyPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("transformer_spatial_rms_norm_apply_p32s16", spirv_transformer_spatial_rms_norm_apply_p32s16, spirv_transformer_spatial_rms_norm_apply_p32s16_size, 7, sizeof(TransformerSpatialRMSNormApplyPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("transformer_spatial_rms_norm_apply_fp32", spirv_transformer_spatial_rms_norm_apply_fp32, spirv_transformer_spatial_rms_norm_apply_fp32_size, 7, sizeof(TransformerSpatialRMSNormApplyPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createTransformerSpatialRMSNormReduce(Pipeline& pipeline, const TransformerSpatialRmsNormTuneParams& tuneParams, const VulkanParams& vulkanParams) {
    auto spec = TransformerSpatialRMSNormReduceSpec();
    spec.TILE_SIZE = tuneParams.TILE_SIZE;
    spec.localSizeX = spec.TILE_SIZE;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("transformer_spatial_rms_norm_reduce_p16s16", spirv_transformer_spatial_rms_norm_reduce_p16s16, spirv_transformer_spatial_rms_norm_reduce_p16s16_size, 2, sizeof(TransformerSpatialRMSNormReducePushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("transformer_spatial_rms_norm_reduce_p32s16", spirv_transformer_spatial_rms_norm_reduce_p32s16, spirv_transformer_spatial_rms_norm_reduce_p32s16_size, 2, sizeof(TransformerSpatialRMSNormReducePushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("transformer_spatial_rms_norm_reduce_fp32", spirv_transformer_spatial_rms_norm_reduce_fp32, spirv_transformer_spatial_rms_norm_reduce_fp32_size, 2, sizeof(TransformerSpatialRMSNormReducePushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  VkResult ComputePipelines::createTransformerSpatialRMSNormSumSq(Pipeline& pipeline, const TransformerSpatialRmsNormTuneParams& tuneParams, const VulkanParams& vulkanParams) {
    auto spec = TransformerSpatialRMSNormSumSqSpec();
    spec.TILE_SIZE = tuneParams.TILE_SIZE;
    spec.localSizeX = spec.TILE_SIZE;
    SpecializationData specData(spec);
    if(vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Storage) {
      if(vulkanParams.shouldUseFP16Compute)
        return createPipeline("transformer_spatial_rms_norm_sum_sq_p16s16", spirv_transformer_spatial_rms_norm_sum_sq_p16s16, spirv_transformer_spatial_rms_norm_sum_sq_p16s16_size, 3, sizeof(TransformerSpatialRMSNormSumSqPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      return createPipeline("transformer_spatial_rms_norm_sum_sq_p32s16", spirv_transformer_spatial_rms_norm_sum_sq_p32s16, spirv_transformer_spatial_rms_norm_sum_sq_p32s16_size, 3, sizeof(TransformerSpatialRMSNormSumSqPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    }
    return createPipeline("transformer_spatial_rms_norm_sum_sq_fp32", spirv_transformer_spatial_rms_norm_sum_sq_fp32, spirv_transformer_spatial_rms_norm_sum_sq_fp32_size, 3, sizeof(TransformerSpatialRMSNormSumSqPushParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }
}
#endif
