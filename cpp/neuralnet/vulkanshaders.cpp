/**
 * @file vulkanshaders.cpp
 * @author Dohoon Kim(https://github.com/dhkim92-dev)
 * @details connect external symbols data to cpp variables
 */
#include "../neuralnet/vulkanshaders.h"

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

  // conv2d_tiled_bn_act_3x3_fp32
  const unsigned char* spirv_conv2d_tiled_bn_act_3x3_fp32 = _binary_conv2d_tiled_bn_act_3x3_fp32_start;
  size_t spirv_conv2d_tiled_bn_act_3x3_fp32_size = _binary_conv2d_tiled_bn_act_3x3_fp32_size;

  // conv2d_tiled_bn_act_5x5_fp32
  const unsigned char* spirv_conv2d_tiled_bn_act_5x5_fp32 = _binary_conv2d_tiled_bn_act_5x5_fp32_start;
  size_t spirv_conv2d_tiled_bn_act_5x5_fp32_size = _binary_conv2d_tiled_bn_act_5x5_fp32_size;

  // winograd_input_transform 
  const unsigned char* spirv_winograd_input_transform = _binary_winograd_input_transform_start;
  size_t spirv_winograd_input_transform_size = _binary_winograd_input_transform_size;

  // winograd_input_transform_bnact
  const unsigned char* spirv_winograd_input_transform_bnact = _binary_winograd_input_transform_bnact_start;
  size_t spirv_winograd_input_transform_bnact_size = _binary_winograd_input_transform_bnact_size;

  // winograd_output_transform
  const unsigned char* spirv_winograd_output_transform = _binary_winograd_output_transform_start;
  size_t spirv_winograd_output_transform_size = _binary_winograd_output_transform_size;

  // add_pointwise_fp32
  const unsigned char* spirv_add_pointwise_fp32 = _binary_add_pointwise_fp32_start;
  size_t spirv_add_pointwise_fp32_size = _binary_add_pointwise_fp32_size;

  // matmul_fp32
  const unsigned char* spirv_matmul_fp32 = _binary_matmul_fp32_start;
  size_t spirv_matmul_fp32_size = _binary_matmul_fp32_size;

  // xgemm_batched_fp32
  const unsigned char* spirv_xgemm_batched_fp32 = _binary_xgemm_batched_fp32_start;
  size_t spirv_xgemm_batched_fp32_size = _binary_xgemm_batched_fp32_size;

  // batched_xgemm_direct_fp32
  const unsigned char* spirv_batched_xgemm_direct = _binary_batched_xgemm_direct_start;
  size_t spirv_batched_xgemm_direct_size = _binary_batched_xgemm_direct_size;

  // xgemm_strided_batched_nn_fp32
  const unsigned char* spirv_xgemm_strided_batched_nn_fp32 = _binary_xgemm_strided_batched_nn_fp32_start;
  size_t spirv_xgemm_strided_batched_nn_fp32_size = _binary_xgemm_strided_batched_nn_fp32_size;

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

  ComputePipelines::ComputePipelines(
    VkDevice device_,
    const vk_shader::tune::VulkanTuneParams& params
  ): device(device_), tuneParams(params) {
    VkResult res = VK_ERROR_UNKNOWN;
    cache = vk_helper::createPipelineCache(device, &res);
    if(res != VK_SUCCESS)
      throw StringError("Failed to create Vulkan pipeline cache: " + vk_helper::vkErrorToString(res));
    try {
      createPipelines();
    }
    catch(...) {
      destroyPipelines();
      vkDestroyPipelineCache(device,cache,nullptr);
      cache = VK_NULL_HANDLE;
      throw;
    }
  }

  ComputePipelines::~ComputePipelines() {
    vkDeviceWaitIdle(device);
    destroyPipelines();
    if( cache != VK_NULL_HANDLE ) {
      vkDestroyPipelineCache(device, cache, nullptr);
      cache = VK_NULL_HANDLE;
    }
  }

  void ComputePipelines::createPipelines() {
    createConv2dFp32();
    createConv2dTiledBnAct3x3Fp32();
    createConv2dTiledBnAct5x5Fp32();
    createWinogradInputTransform();
    createWinogradInputTransformBnAct();
    createWinogradOutputTransform();
    // createConv2d3x3BnFp32();
    // createConv2d3x3BnReluFp32();
    // createConv2d5x5BnFp32();
    // createConv2d5x5BnReluFp32();
    // createConv2d5x5BnMishFp32();
    createAddPointWiseFp32();
    createMatmulFp32();
    createBatchedXgemmDirect();
    createXGEMMBatchedFp32();
    createXGEMMStridedBatchedFp32();
    // createMatmulTiled4x4x32Fp32();
    createBatchNormMaskFp32();
    createBatchNormMaskReluFp32();
    createBatchNormMaskMishFp32();
    createBatchNormMaskMishScale8Fp32();
    createGlobalPoolingChannelsFp32();
    createValueHeadPoolingChannelsFp32();
    createSumChannelsFp32();
    createAddChannelBiasNCHWFp32();
    createAddChannelBiasNCIdentityFp32();
    createAddChannelBiasNCReluFp32();
    createAddChannelBiasNCMishFp32();
    createAddChannelBiasNCMishScale8Fp32();
    createExtractChannel0NCHWFp32();
  }

  void ComputePipelines::destroyPipelines() {
    destroyPipeline(conv2dFp32);
    destroyPipeline(conv2dTiledBnAct3x3Fp32);
    destroyPipeline(conv2dTiledBnAct5x5Fp32);
    // destroyPipeline(conv2d3x3BnFp32);
    // destroyPipeline(conv2d3x3BnReluFp32);
    // destroyPipeline(conv2d5x5BnFp32);
    // destroyPipeline(conv2d5x5BnReluFp32);
    // destroyPipeline(conv2d5x5BnMishFp32);
    destroyPipeline(addPointWiseFp32);
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
    destroyPipeline(matmulFp32);
    destroyPipeline(batchedXgemmDirect);
    destroyPipeline(xgemmStridedBatchedFp32);
    destroyPipeline(xgemmBatchedFp32);
    // destroyPipeline(matmulTiledChw4x4x32Fp32);
    destroyPipeline(batchNormMaskFp32);
    destroyPipeline(batchNormMaskReluFp32);
    destroyPipeline(batchNormMaskMishFp32);
    destroyPipeline(batchNormMaskMishScale8Fp32);
    destroyPipeline(globalPoolingChannelsFp32);
    destroyPipeline(valueHeadPoolingChannelsFp32);
    for ( int bs =1 ; bs <= 4 ; bs++ ) {
      destroyPipeline(sumChannels[bs]);
    }
    sumChannels.clear();
    destroyPipeline(addChannelBiasNCHWFp32);
    destroyPipeline(addChannelBiasNCIdentityFp32);
    destroyPipeline(addChannelBiasNCReluFp32);
    destroyPipeline(addChannelBiasNCMishFp32);
    destroyPipeline(addChannelBiasNCMishScale8Fp32);
    destroyPipeline(extractChannel0NCHWFp32);
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
  }

  void ComputePipelines::createPipeline(
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

    if(spirvSize % 4 != 0) {
      throw StringError(pipelineName + " SPIR-V size is not a multiple of 4 bytes");
    }

    size_t numWords = spirvSize / 4;
    std::vector<uint32_t> spirvWords(numWords);
    // Safe memcpy to properly align bytes into 32-bit words
    memcpy(spirvWords.data(), spirvBytes, spirvSize);

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const auto fail = [&](const std::string& operation, VkResult result) {
      if(pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device,pipeline,nullptr);
      if(layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device,layout,nullptr);
      if(descriptorSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device,descriptorSetLayout,nullptr);
      if(shaderModule != VK_NULL_HANDLE)
        vkDestroyShaderModule(device,shaderModule,nullptr);
      throw StringError(pipelineName + " " + operation + " failed: " + vk_helper::vkErrorToString(result));
    };
    VkShaderModuleCreateInfo shaderModuleCI = {};
    shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCI.codeSize = spirvSize;
    shaderModuleCI.pCode = spirvWords.data();
    // std::cout << "Creating Compute Pipeline: " << pipelineName <<  " code size : " << shaderModuleCI.codeSize << std::endl;
    res = vkCreateShaderModule(device, &shaderModuleCI, nullptr, &shaderModule);
    if(res != VK_SUCCESS)
      fail("shader module creation",res);
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
      fail("descriptor set layout creation",res);

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
      fail("pipeline layout creation",res);

    pipeline = vk_helper::createComputePipeline(device, layout, cache, shaderModule, &res, specializationInfo);

    if(res != VK_SUCCESS)
      fail("compute pipeline creation",res);
    outPipeline.descriptorSetLayout = descriptorSetLayout;
    outPipeline.layout = layout;
    outPipeline.pipeline = pipeline;
    outPipeline.localSizeX = localSizeX;
    outPipeline.localSizeY = localSizeY;
    outPipeline.localSizeZ = localSizeZ;
    // std::cout << "Created Compute Pipeline: " << pipelineName << " result : " << res << std::endl;
    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  /**
   * @brief Create a Conv2d Fp32 object
  */
  void ComputePipelines::createConv2dFp32() {
    auto spec = Conv2DSpec();
    SpecializationData specData(spec);
    createPipeline("Conv2dFp32", vk_shader::spirv_conv2d_fp32, vk_shader::spirv_conv2d_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2dFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createConv2dTiledBnAct3x3Fp32() {
    auto spec = Conv2DTiledBnAct3x3Spec();
    SpecializationData specData(spec);
    createPipeline("Conv2dTiledBnAct3x3Fp32", vk_shader::spirv_conv2d_tiled_bn_act_3x3_fp32, vk_shader::spirv_conv2d_tiled_bn_act_3x3_fp32_size, 6, sizeof(Conv2DTiledBnActParams), conv2dTiledBnAct3x3Fp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createConv2dTiledBnAct5x5Fp32() {
    auto spec = Conv2DTiledBnAct5x5Spec();
    SpecializationData specData(spec);
    createPipeline("Conv2dTiledBnAct5x5Fp32", vk_shader::spirv_conv2d_tiled_bn_act_5x5_fp32, vk_shader::spirv_conv2d_tiled_bn_act_5x5_fp32_size, 6, sizeof(Conv2DTiledBnActParams), conv2dTiledBnAct5x5Fp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createWinogradInputTransform() {
    WinogradInputTransformSpec spec;
    spec.localSizeX = (tuneParams.conv3x3.inputTransformLocalXSize);
    spec.localSizeY = (tuneParams.conv3x3.inputTransformLocalYSize);
    spec.localSizeZ = 1;
    spec.inTileYSize = static_cast<int>(tuneParams.conv3x3.inTileYSize);
    spec.inTileXSize = static_cast<int>(tuneParams.conv3x3.inTileXSize);
    spec.outTileYSize = static_cast<int>(tuneParams.conv3x3.outTileYSize);
    spec.outTileXSize = static_cast<int>(tuneParams.conv3x3.outTileXSize);
    spec.inTileYOffset= -1;
    spec.inTileXOffset= -1;
    spec.convY = 3;
    spec.convX = 3;
    std::vector<int32_t> specData_3322 = vk_helper::createSpecData(&spec, sizeof(WinogradInputTransformSpec));
    std::vector<VkSpecializationMapEntry> specEntry = vk_helper::createSpecMapEntries(specData_3322.size());
    VkSpecializationInfo si3322 = vk_helper::createSpecializationInfo(specData_3322, specEntry);

    createPipeline("WinogradInputTransform for 3x3 kernels", vk_shader::spirv_winograd_input_transform, vk_shader::spirv_winograd_input_transform_size, 2, sizeof(WinogradInputTransformParams), winogradInputTransform3x3, &si3322, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    spec.convX = 5;
    spec.convY = 5;
    spec.outTileXSize = tuneParams.conv5x5.outTileXSize;
    spec.outTileYSize = tuneParams.conv5x5.outTileYSize;
    spec.inTileYSize = tuneParams.conv5x5.inTileYSize;
    spec.inTileXSize = tuneParams.conv5x5.inTileXSize;
    spec.inTileYOffset= -2;
    spec.inTileXOffset= -2;
    spec.localSizeX = tuneParams.conv5x5.inputTransformLocalXSize;
    spec.localSizeY = tuneParams.conv5x5.inputTransformLocalYSize;
    std::vector<int32_t> specData_5544 = vk_helper::createSpecData(&spec, sizeof(WinogradInputTransformSpec));
    VkSpecializationInfo si5544 = vk_helper::createSpecializationInfo(specData_5544, specEntry);
    createPipeline("WinogradInputTransform for 5x5 kernels", vk_shader::spirv_winograd_input_transform, vk_shader::spirv_winograd_input_transform_size, 2, sizeof(WinogradInputTransformParams), winogradInputTransform5x5, &si5544, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createWinogradInputTransformBnAct() {
    auto spec = WinogradInputTransformBnActSpec();
    spec.localSizeX = tuneParams.conv3x3.inputTransformLocalXSize;
    spec.localSizeY = tuneParams.conv3x3.inputTransformLocalYSize;
    spec.localSizeZ = 1;
    spec.inTileYSize = static_cast<int>(tuneParams.conv3x3.inTileYSize);
    spec.inTileXSize = static_cast<int>(tuneParams.conv3x3.inTileXSize);
    spec.outTileYSize = static_cast<int>(tuneParams.conv3x3.outTileYSize);
    spec.outTileXSize = static_cast<int>(tuneParams.conv3x3.outTileXSize);
    spec.inTileYOffset= -1;
    spec.inTileXOffset= -1;
    spec.convY = 3;
    spec.convX = 3;
    spec.activation = ACTIVATION_IDENTITY;
    std::vector<int32_t> specData_3322_identity = vk_helper::createSpecData(&spec, sizeof(WinogradInputTransformBnActSpec));
    std::vector<VkSpecializationMapEntry> specEntry = vk_helper::createSpecMapEntries(specData_3322_identity.size());
    VkSpecializationInfo si3322_identity = vk_helper::createSpecializationInfo(specData_3322_identity, specEntry);
    createPipeline("winogradInputTransformBnAct3x3 identity", vk_shader::spirv_winograd_input_transform_bnact, vk_shader::spirv_winograd_input_transform_bnact_size, 5, sizeof(WinogradInputTransformParams), winogradInputTransform3x3_bnact_identity, &si3322_identity, spec.localSizeX, spec.localSizeY, spec.localSizeZ);

    spec.activation = ACTIVATION_RELU;
    std::vector<int32_t> specData_3322_relu = vk_helper::createSpecData(&spec, sizeof(WinogradInputTransformBnActSpec));
    VkSpecializationInfo si3322_relu = vk_helper::createSpecializationInfo(specData_3322_relu, specEntry);
    createPipeline("winogradInputTransformBnAct3x3 relu", vk_shader::spirv_winograd_input_transform_bnact, vk_shader::spirv_winograd_input_transform_bnact_size, 5, sizeof(WinogradInputTransformParams), winogradInputTransform3x3_bnact_relu, &si3322_relu, spec.localSizeX, spec.localSizeY, spec.localSizeZ);

    spec.activation = ACTIVATION_MISH;
    std::vector<int32_t> specData_3322_mish = vk_helper::createSpecData(&spec, sizeof(WinogradInputTransformBnActSpec));
    VkSpecializationInfo si3322_mish = vk_helper::createSpecializationInfo(specData_3322_mish, specEntry);
    createPipeline("winogradInputTransformBnAct3x3 mish", vk_shader::spirv_winograd_input_transform_bnact, vk_shader::spirv_winograd_input_transform_bnact_size, 5, sizeof(WinogradInputTransformParams), winogradInputTransform3x3_bnact_mish, &si3322_mish, spec.localSizeX, spec.localSizeY, spec.localSizeZ);

    spec.activation = ACTIVATION_MISH_SCALE8;
    std::vector<int32_t> specData_3322_mish_scale8 = vk_helper::createSpecData(&spec, sizeof(WinogradInputTransformBnActSpec));
    VkSpecializationInfo si3322_mish_scale8 = vk_helper::createSpecializationInfo(specData_3322_mish_scale8, specEntry);
    createPipeline("winogradInputTransformBnAct3x3_2x2_mish_scale8", vk_shader::spirv_winograd_input_transform_bnact, vk_shader::spirv_winograd_input_transform_bnact_size, 5, sizeof(WinogradInputTransformParams), winogradInputTransform3x3_bnact_mish_scale8, &si3322_mish_scale8, spec.localSizeX, spec.localSizeY, spec.localSizeZ);

    // 5x5
    spec.convX = 5;
    spec.convY = 5;
    spec.outTileXSize = tuneParams.conv5x5.outTileXSize;
    spec.outTileYSize = tuneParams.conv5x5.outTileYSize;
    spec.inTileXSize = tuneParams.conv5x5.inTileXSize;
    spec.inTileYSize = tuneParams.conv5x5.inTileYSize;
    spec.inTileXOffset= -2;
    spec.inTileYOffset= -2;
    spec.localSizeX = tuneParams.conv5x5.inputTransformLocalXSize;
    spec.localSizeY = tuneParams.conv5x5.inputTransformLocalYSize;
    spec.activation = ACTIVATION_IDENTITY;
    std::vector<int32_t> specData_55_identity = vk_helper::createSpecData(&spec, sizeof(WinogradInputTransformBnActSpec));
    VkSpecializationInfo si55_identity = vk_helper::createSpecializationInfo(specData_55_identity, specEntry);
    createPipeline("winogradInputTransformBnAct5x5 bn act identity", vk_shader::spirv_winograd_input_transform_bnact, vk_shader::spirv_winograd_input_transform_bnact_size, 5, sizeof(WinogradInputTransformParams), winogradInputTransform5x5_bnact_identity, &si55_identity, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    spec.activation = ACTIVATION_RELU;
    std::vector<int32_t> specData_55_relu = vk_helper::createSpecData(&spec, sizeof(WinogradInputTransformBnActSpec));
    VkSpecializationInfo si55_relu = vk_helper::createSpecializationInfo(specData_55_relu, specEntry);
    createPipeline("winogradInputTransformBnAct5x5 bn act relu", vk_shader::spirv_winograd_input_transform_bnact, vk_shader::spirv_winograd_input_transform_bnact_size, 5, sizeof(WinogradInputTransformParams), winogradInputTransform5x5_bnact_relu, &si55_relu, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    spec.activation = ACTIVATION_MISH;
    std::vector<int32_t> specData_55_mish = vk_helper::createSpecData(&spec, sizeof(WinogradInputTransformBnActSpec));
    VkSpecializationInfo si55_mish = vk_helper::createSpecializationInfo(specData_55_mish, specEntry);
    createPipeline("winogradInputTransformBnAct5x5 bn act mish", vk_shader::spirv_winograd_input_transform_bnact, vk_shader::spirv_winograd_input_transform_bnact_size, 5, sizeof(WinogradInputTransformParams), winogradInputTransform5x5_bnact_mish, &si55_mish, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
    spec.activation = ACTIVATION_MISH_SCALE8;
    std::vector<int32_t> specData_55_mish_scale8 = vk_helper::createSpecData(&spec, sizeof(WinogradInputTransformBnActSpec));
    VkSpecializationInfo si55_mish_scale8 = vk_helper::createSpecializationInfo(specData_55_mish_scale8, specEntry);
    createPipeline("winogradInputTransformBnAct5x5 bn act mish scale8", vk_shader::spirv_winograd_input_transform_bnact, vk_shader::spirv_winograd_input_transform_bnact_size, 5, sizeof(WinogradInputTransformParams), winogradInputTransform5x5_bnact_mish_scale8, &si55_mish_scale8, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createWinogradOutputTransform() {
    WinogradOutputTransformSpec spec;
    // Ensure specialization constants for the 3x3 output transform
    // match the dispatch-local sizes used at runtime (8,2,2).
    spec.localSizeX = tuneParams.conv3x3.outputTransformLocalXSize;
    spec.localSizeY = tuneParams.conv3x3.outputTransformLocalYSize;
    spec.localSizeZ = tuneParams.conv3x3.outputTransformLocalZSize;
    spec.outTileXSize= tuneParams.conv3x3.outTileXSize;
    spec.outTileYSize= tuneParams.conv3x3.outTileYSize;
    spec.inTileXSize = tuneParams.conv3x3.inTileXSize;
    spec.inTileYSize = tuneParams.conv3x3.inTileYSize;
    spec.convX = 3;
    spec.convY = 3;
    std::vector<int32_t> specData_33 = vk_helper::createSpecData(&spec, sizeof(WinogradOutputTransformSpec));
    std::vector<VkSpecializationMapEntry> specEntry = vk_helper::createSpecMapEntries(specData_33.size());
    VkSpecializationInfo si33 = vk_helper::createSpecializationInfo(specData_33, specEntry);
    createPipeline("WinogradOutputTransform", vk_shader::spirv_winograd_output_transform, vk_shader::spirv_winograd_output_transform_size, 2, sizeof(WinogradOutputTransformParams), winogradOutputTransform3x3, &si33, spec.localSizeX, spec.localSizeY, spec.localSizeZ);

    spec.localSizeX = tuneParams.conv5x5.outputTransformLocalXSize;
    spec.localSizeY = tuneParams.conv5x5.outputTransformLocalYSize;
    spec.localSizeZ =  tuneParams.conv5x5.outputTransformLocalZSize;
    spec.outTileXSize= tuneParams.conv5x5.outTileXSize;
    spec.outTileYSize= tuneParams.conv5x5.outTileYSize;
    spec.inTileXSize = tuneParams.conv5x5.inTileXSize;
    spec.inTileYSize = tuneParams.conv5x5.inTileYSize;
    spec.convX = 5;
    spec.convY = 5;
    std::vector<int32_t> specData_55 = vk_helper::createSpecData(&spec, sizeof(WinogradOutputTransformSpec));
    VkSpecializationInfo si55 = vk_helper::createSpecializationInfo(specData_55, specEntry);
    createPipeline("WinogradOutputTransform", vk_shader::spirv_winograd_output_transform, vk_shader::spirv_winograd_output_transform_size, 2, sizeof(WinogradOutputTransformParams), winogradOutputTransform5x5, &si55, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
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
   * @brief Create a Add Point Wise Fp32 object
   */
  void ComputePipelines::createAddPointWiseFp32() {
    auto spec = AddPointWiseSpec();
    SpecializationData specData(spec);
    createPipeline("AddPointWiseFp32", vk_shader::spirv_add_pointwise_fp32, vk_shader::spirv_add_pointwise_fp32_size, 2, sizeof(AddPointWiseParams), addPointWiseFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createMatmulFp32() {
    auto spec = MatmulSpec();
    SpecializationData specData(spec);
    createPipeline("MatmulFp32", vk_shader::spirv_matmul_fp32, vk_shader::spirv_matmul_fp32_size, 3, sizeof(MatmulFp32Params), matmulFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createBatchedXgemmDirect() {
    auto spec = XgemmDirectSpec();
    spec.localSizeX = tuneParams.xgemmDirect.MDIMCD;
    spec.localSizeY = tuneParams.xgemmDirect.NDIMCD;
    spec.localSizeZ = 1;
    spec.WGD = static_cast<int>(tuneParams.xgemmDirect.WGD);
    spec.MDIMCD = static_cast<int>(tuneParams.xgemmDirect.MDIMCD);
    spec.NDIMCD = static_cast<int>(tuneParams.xgemmDirect.NDIMCD);
    spec.MDIMAD = static_cast<int>(tuneParams.xgemmDirect.MDIMAD);
    spec.NDIMBD = static_cast<int>(tuneParams.xgemmDirect.NDIMBD);
    spec.KWID = static_cast<int>(tuneParams.xgemmDirect.KWID);
    spec.PADA = static_cast<int>(tuneParams.xgemmDirect.PADA);
    spec.PADB = static_cast<int>(tuneParams.xgemmDirect.PADB);
    std::vector<VkSpecializationMapEntry> mapEntries = vk_helper::createSpecMapEntries(sizeof(spec) / sizeof(int32_t));
    std::vector<int32_t> specData = vk_helper::createSpecData(&spec, sizeof(spec));
    VkSpecializationInfo specializationInfo = vk_helper::createSpecializationInfo(specData, mapEntries);
    createPipeline("BatchedXGEMMDirect", vk_shader::spirv_batched_xgemm_direct, vk_shader::spirv_batched_xgemm_direct_size, 3, sizeof(BatchedXGEMMDirectParams), batchedXgemmDirect, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createXGEMMBatchedFp32() {

    auto spec = XGEMMBatchedSpec();
    spec.localSizeX = tuneParams.xgemm.MDIMC;
    spec.localSizeY = tuneParams.xgemm.NDIMC;
    spec.localSizeZ = 1;
    spec.MWG = tuneParams.xgemm.MWG;
    spec.NWG = tuneParams.xgemm.NWG;
    spec.KWG = tuneParams.xgemm.KWG;
    spec.MDIMC = tuneParams.xgemm.MDIMC;
    spec.NDIMC = tuneParams.xgemm.NDIMC;
    spec.MDIMA = tuneParams.xgemm.MDIMA;
    spec.NDIMB = tuneParams.xgemm.NDIMB;
    std::vector<VkSpecializationMapEntry> mapEntries = vk_helper::createSpecMapEntries(sizeof(spec) / sizeof(int32_t));
    std::vector<int32_t> specData = vk_helper::createSpecData(&spec, sizeof(spec));
    VkSpecializationInfo specializationInfo = vk_helper::createSpecializationInfo(specData, mapEntries);
    createPipeline("XGEMMBatchedFp32", vk_shader::spirv_xgemm_batched_fp32, vk_shader::spirv_xgemm_batched_fp32_size, 3, sizeof(XGEMMBatchedParams), xgemmBatchedFp32, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createXGEMMStridedBatchedFp32() {
    auto spec = XgemmDirectSpec();
    spec.localSizeX = tuneParams.xgemmDirect.MDIMCD;
    spec.localSizeY = tuneParams.xgemmDirect.NDIMCD;
    spec.localSizeZ = 1;
    spec.WGD = static_cast<int>(tuneParams.xgemmDirect.WGD);
    spec.MDIMCD = static_cast<int>(tuneParams.xgemmDirect.MDIMCD);
    spec.NDIMCD = static_cast<int>(tuneParams.xgemmDirect.NDIMCD);
    spec.MDIMAD = static_cast<int>(tuneParams.xgemmDirect.MDIMAD);
    spec.NDIMBD = static_cast<int>(tuneParams.xgemmDirect.NDIMBD);
    spec.KWID = static_cast<int>(tuneParams.xgemmDirect.KWID);
    spec.PADA = static_cast<int>(tuneParams.xgemmDirect.PADA);
    spec.PADB = static_cast<int>(tuneParams.xgemmDirect.PADB);
    auto specData = vk_helper::createSpecData(&spec, sizeof(spec));
    auto mapEntries = vk_helper::createSpecMapEntries(sizeof(spec) / sizeof(int32_t));
    VkSpecializationInfo specializationInfo = vk_helper::createSpecializationInfo(specData, mapEntries);
    createPipeline("XGEMMStridedBatchedFp32", vk_shader::spirv_xgemm_strided_batched_nn_fp32, vk_shader::spirv_xgemm_strided_batched_nn_fp32_size, 3, sizeof(XgemmStridedBatchedFp32Params), xgemmStridedBatchedFp32, &specializationInfo, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createBatchNormMaskFp32() {
    auto spec = BatchNormMaskSpec();
    SpecializationData specData(spec);
    createPipeline("BatchNormMaskFp32", vk_shader::spirv_bn_mask_fp32, vk_shader::spirv_bn_mask_fp32_size, 5, sizeof(BatchNormMaskParams), batchNormMaskFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createBatchNormMaskReluFp32() {
    auto spec = BatchNormMaskSpec();
    SpecializationData specData(spec);
    createPipeline("BatchNormMaskReluFp32", vk_shader::spirv_bn_mask_relu_fp32, vk_shader::spirv_bn_mask_relu_fp32_size, 5, sizeof(BatchNormMaskParams), batchNormMaskReluFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createBatchNormMaskMishFp32() {
    auto spec = BatchNormMaskSpec();
    SpecializationData specData(spec);
    createPipeline("BatchNormMaskMishFp32", vk_shader::spirv_bn_mask_mish_fp32, vk_shader::spirv_bn_mask_mish_fp32_size, 5, sizeof(BatchNormMaskParams), batchNormMaskMishFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createBatchNormMaskMishScale8Fp32() {
    auto spec = BatchNormMaskSpec();
    SpecializationData specData(spec);
    createPipeline("BatchNormMaskMishScale8Fp32", vk_shader::spirv_bn_mask_mish_scale8_fp32, vk_shader::spirv_bn_mask_mish_scale8_fp32_size, 5, sizeof(BatchNormMaskParams), batchNormMaskMishScale8Fp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createGlobalPoolingChannelsFp32() {
    auto spec = GlobalPoolingChannelsSpec();
    SpecializationData specData(spec);
    createPipeline("GlobalPoolingChannelsFp32", vk_shader::spirv_global_pooling_channels_fp32, vk_shader::spirv_global_pooling_channels_fp32_size, 4, sizeof(GlobalPoolingChannelsParams), globalPoolingChannelsFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createValueHeadPoolingChannelsFp32() {
    auto spec = ValueHeadPoolingChannelsSpec();
    SpecializationData specData(spec);
    createPipeline("ValueHeadPoolingChannelsFp32", vk_shader::spirv_value_head_pool_channels_fp32, vk_shader::spirv_value_head_pool_channels_fp32_size, 3, sizeof(ValueHeadPoolingChannelsParams), valueHeadPoolingChannelsFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createSumChannelsFp32() {
    auto spec = SumChannelsSpec();
    spec.CHANNELSTRIDE = tuneParams.gPool.CHANNELSTRIDE;
    spec.XYSTRIDE = tuneParams.gPool.XYSTRIDE;
    spec.localSizeX = spec.XYSTRIDE;
    spec.localSizeY = spec.CHANNELSTRIDE;
    sumChannels.clear();
    for ( int bs = 1 ; bs <= 4 ; ++bs ) {
      spec.localSizeZ = bs;
      spec.LOCALSIZE_TOTAL = spec.localSizeX * spec.localSizeY * spec.localSizeZ;
      SpecializationData specData(spec);
      Pipeline pipeline{};
      createPipeline("SumChannelsFp32", vk_shader::spirv_sum_channels_fp32, vk_shader::spirv_sum_channels_fp32_size, 2, sizeof(SumChannelsParams), pipeline, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
      sumChannels.push_back(pipeline);
    }
  }

  void ComputePipelines::createAddChannelBiasNCHWFp32() {
    auto spec = AddChannelBiasNCHWSpec();
    SpecializationData specData(spec);
    createPipeline("AddChannelBiasNCHWFp32", vk_shader::spirv_add_channel_bias_nchw_fp32, vk_shader::spirv_add_channel_bias_nchw_fp32_size, 2, sizeof(AddChannelBiasNCHWParams), addChannelBiasNCHWFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createAddChannelBiasNCIdentityFp32() {
    auto spec = AddChannelBiasNCSpec();
    SpecializationData specData(spec);
    createPipeline("AddChannelBiasNCIdentityFp32", vk_shader::spirv_add_channel_bias_nc_identity_fp32, vk_shader::spirv_add_channel_bias_nc_identity_fp32_size, 2, sizeof(AddChannelBiasNCParams), addChannelBiasNCIdentityFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createAddChannelBiasNCReluFp32() {
    auto spec = AddChannelBiasNCSpec();
    SpecializationData specData(spec);
    createPipeline("AddChannelBiasNCReluFp32", vk_shader::spirv_add_channel_bias_nc_relu_fp32, vk_shader::spirv_add_channel_bias_nc_relu_fp32_size, 2, sizeof(AddChannelBiasNCParams), addChannelBiasNCReluFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createAddChannelBiasNCMishFp32() {
    auto spec = AddChannelBiasNCSpec();
    SpecializationData specData(spec);
    createPipeline("AddChannelBiasNCMishFp32", vk_shader::spirv_add_channel_bias_nc_mish_fp32, vk_shader::spirv_add_channel_bias_nc_mish_fp32_size, 2, sizeof(AddChannelBiasNCParams), addChannelBiasNCMishFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createAddChannelBiasNCMishScale8Fp32() {
    auto spec = AddChannelBiasNCSpec();
    SpecializationData specData(spec);
    createPipeline("AddChannelBiasNCMishScale8Fp32", vk_shader::spirv_add_channel_bias_nc_mish_scale8_fp32, vk_shader::spirv_add_channel_bias_nc_mish_scale8_fp32_size, 2, sizeof(AddChannelBiasNCParams), addChannelBiasNCMishScale8Fp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }

  void ComputePipelines::createExtractChannel0NCHWFp32() {
    auto spec = ExtractChannel0NCHWSpec();
    SpecializationData specData(spec);
    createPipeline("ExtractChannel0NCHWFp32", vk_shader::spirv_extract_channel0_nchw_fp32, vk_shader::spirv_extract_channel0_nchw_fp32_size, 2, sizeof(ExtractChannel0NCHWParams), extractChannel0NCHWFp32, &specData.info, spec.localSizeX, spec.localSizeY, spec.localSizeZ);
  }


}
