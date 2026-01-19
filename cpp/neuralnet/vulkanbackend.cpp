/**
 * @file vulkanbackend.cpp
 * @author Dohoon Kim(https://github.com/dhkim92-dev, dhkim92-dev@gmail.com, https://www.dohoon-kim.kr)
 * @brief Vulkan backend for Neural Net evaluation
 */
#ifdef USE_VULKAN_BACKEND

#include <unordered_map>
#include <memory>
#include "../core/global.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../external/vulkan/vulkanshaders.h"
#include "../neuralnet/nninterface.h"

struct ComputePipelines;
struct Buffers;
struct ScratchBuffers;
struct ComputeHandleInternal;
struct VulkanTuneParams;

struct ComputeContext {
  const std::vector<uint32_t>* gIdx;
  const int nnXLen;
  const int nnYLen;
  const enabled_t usingFP16Mode;
  const enabled_t usingNHWCMode;

  VulkanContext* vulkanContext; 
  std::unordered_map<uint32_t, ComputePipelines&> pipelinesPerDev;

  
  ComputeContext(
    int nnXLen,
    int nnYLen,
    enabled_t useFP16Mode,
    enabled_t useNHWCModel,
    const std::vector<uint32_t>& gpuIdxsToUse,
    Logger* logger)
  : nnXLen(nnXLen),
    nnYLen(nnYLen),
    usingFP16Mode(useFP16Mode),
    usingNHWCMode(useNHWCModel),
    gIdx(&gpuIdxsToUse) {
      VkInstance instance = VkHelpers::createVulkanInstance();
      std::vector<VulkanDeviceInfo> allDeviceInfos = VkHelpers::enumerateVulkanDevices(instance, logger);
      std::vector<VulkanDevice *> vulkanDevices = {};

      for ( size_t i = 0 ; i < gpuIdxsToUse.size() ; i++ ) {
        uint32_t gpuIdx = gpuIdxsToUse[i];
        if ( gpuIdx >= allDeviceInfos.size() ) {
          throw StringError("Requested GPU index " + std::to_string(gpuIdx) + " but only " + std::to_string(allDeviceInfos.size()) + " Vulkan devices available");
        }

        VulkanDeviceInfo& deviceInfo = allDeviceInfos[gpuIdx];

        std::vector<const char*> requiredExtensions = {
        };

        // Check for FP16 support if requested
        if ( usingFP16Mode == enabled_t::True && !isDeviceSupportFp16(deviceInfo) ) {
          throw StringError("Requested FP16 mode but device " + deviceInfo.deviceName + " does not support it");
        } 

        if ( usingFP16Mode != enabled_t::False && isDeviceSupportFp16(deviceInfo) ) {
          requiredExtensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        }

        // TODO: Not like OpenCL, Vulkan can access Tensor cores via extensions. I will support it later.
        //       VK_KHR_cooperative_matrix extension is required for NHWC mode.
        // Check for NHWC support if requested
        if ( usingNHWCMode == enabled_t::True && !isDeviceSupportNHWC(deviceInfo) ) {
          throw StringError("Requested NHWC mode but device " + deviceInfo.deviceName + " does not support it");
        }

        if ( usingNHWCMode != enabled_t::False && isDeviceSupportNHWC(deviceInfo) ) {
          requiredExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        }

        VulkanDevice* vulkanDevice = VkHelpers::createVulkanDevice(
          deviceInfo,
          requiredExtensions,
          logger
        );
        vulkanDevices.push_back(vulkanDevice);
      }

      vulkanContext = new VulkanContext(
        instance,
        vulkanDevices,
        logger
      );
  }

private:

  bool isDeviceSupportFp16(const VulkanDeviceInfo& deviceInfo) {
    //Check for fp16 feature
    return deviceInfo.shaderFloat16Int8Features.shaderFloat16 == VK_TRUE;
  }

  bool isDeviceSupportNHWC(const VulkanDeviceInfo& deviceInfo) {
    return deviceInfo.properties.apiVersion >= VK_API_VERSION_1_1 &&
           deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE;
  }
};

struct Buffers {
  VulkanBuffer input;
  VulkanBuffer inputGlobal;  
  VulkanBuffer inputMeta;
  size_t inputElts;
  size_t inputGlobalElts;
  size_t inputMetaElts;

  VulkanBuffer outputBuffer; 
  VulkanBuffer mask;
  VulkanBuffer maskSum;
  VulkanBuffer trunk;
  VulkanBuffer policyPass;
  VulkanBuffer policy;
  size_t policyPassElts;
  size_t policyElts;

  VulkanBuffer value;
  size_t valueElts;
  VulkanBuffer scoreValue;
  size_t scoreValueElts;
  VulkanBuffer ownership;
  size_t ownershipElts;

  VulkanBuffer convWorkspace;
  VulkanBuffer convWorkspace2;
};

struct Model;

struct ComputeHandle {
  std::unique_ptr<ComputeHandleInternal> handle;
  std::unique_ptr<Model> model;
  std::unique_ptr<ScratchBuffers> scratch;
  std::unique_ptr<Buffers> buffers;
  const int nnXLen;
  const int nnYLen;
  const int policySize;
  const bool inputUseNHWC;

  ComputeHandle(
    ComputeContext* context,
    const LoadedModel** loadedModel,
    int maxBatchSize,
    int gpuIdx,
    bool inputUseNHWC_
  ): 
    nnXLen(context->nnXLen),
    nnYLen(context->nnYLen),
    policySize(NNPos::getPolicySize(context->nnXLen,context->nnYLen)),
    inputUseNHWC(inputUseNHWC_)
  {
    bool useNHWC = context->usingNHWCMode == enabled_t::True ? true : false;
    ComputeHandleInternal* handlePtr = new ComputeHandleInternal(
      context,
      gpuIdx,
      inputUseNHWC_,
      useNHWC
    );
    this->handle = std::unique_ptr<ComputeHandleInternal>(handlePtr);
  }
};

struct ComputeHandleInternal {
  ComputeContext* context;
  VkDevice vulkanDevice;
  VkQueue vulkanQueue;

  bool usingFP16Storage;
  bool usingFP16Compute;
  bool usingFP16TensorCores;
  bool usingFP16TensorCoreForConv1x1;

  ComputeHandleInternal(
    ComputeContext* ctx,
    int gpuIdx,
    bool inputsUseNHWC,
    bool useNHWC
  ) {
    this->context = ctx;
    const VulkanDevice* vulkanDevice = ctx->vulkanContext->findGpuExn(gpuIdx);
    this->vulkanDevice = vulkanDevice->device;
    this->vulkanQueue = vulkanDevice->queue;

  }

};

/**
 * @brief Will be used to tune various parameters for different devices
 *        Not implemented yet. support in future. 
 */
struct VulkanTuneParams {

};

/**
 * @brief Vulkan Compute  Pipeline structure
 * 
 */
struct Pipeline {
  VkPipelineLayout layout;
  VkPipeline pipeline;
  VkDescriptorSetLayout descriptorSetLayout;
  // VkDescriptorSet descriptorSet;
};

/**
 * @brief Push constant parameters for Conv2D operation. it makes easier to pass small parameters to shader.
 */
struct Conv2DPushConstantParams {
  int N; // Batch size
  int inC; // Input channels
  int inH; // Input height
  int inW; // Input width
  int outC; // Output channels
  int outH; // Output height
  int outW; // Output width
  int filterH; // Filter height
  int filterW; // Filter width
};

struct NCHWPushConstantParams {
  int N; // Batch size
  int C; // Channels
  int H; // Height
  int W; // Width
};

struct MatmulPushConstantParams {
  int M; // Rows of A and C, in case A is number of channels. 
  int N; // Columns of B and C
  int K; // Columns of A and Rows of B
};

/**
 * @brief Compute pipelines for various operations
 */
struct ComputePipelines {
  // const VulkanTuneParams tuneParams;
  // const bool usingFP16Storage;
  // const bool usingFP16Compute;
  // const bool usingFP16TensorCores;
  VkDevice device;
  VkPipelineCache cache;

  // In this code, assume that NCHW is default format if no postfix is given.

  // Conv2D pipelines
  Pipeline conv2dFp32; 
  Pipeline conv2d3x3BnFp32;
  Pipeline conv2d3x3BnReluFp32;
  Pipeline conv2d3x3BnMishFp32;
  Pipeline conv2d5x5BnFp32;
  Pipeline conv2d5x5BnReluFp32;
  Pipeline conv2d5x5BnMishFp32;
  Pipeline addPointWiseFp32;  // operation for skipping connections

  // Pipeline for matrix multiplication
  // note that conv1x1 can be implemented as matmul operation
  Pipeline matmulFp32; 
  Pipeline matmulTiledChw4x4x32Fp32;

  // Batch Normalization pipelines
  // note that prediction phase does not need batch normalization operation separately
  // as the parameters can be folded into convolution scale and bias.
  Pipeline batchNormMaskFp32;
  Pipeline batchNormMaskReluFp32;
  Pipeline batchNormMaskMishFp32;

  // pooling pipelines
  Pipeline globalPoolingChannelsFp32;
  Pipeline valueHeadPoolingChannelsFp32;
  
  // Element wise operations
  Pipeline sumChannelsFp32;
  Pipeline addChannelBiasNCHWFp32;
  Pipeline addChannelBiasNCReluFp32;
  Pipeline addChannelBiasNCMishFp32;
  Pipeline extractChannel0NCHWFp32;

  ComputePipelines(
    VkDevice device_
  ): device(device_) {
    VkResult res = VK_ERROR_UNKNOWN;
    cache = VkHelpers::createPipelineCache(device, &res);
    CHECK_VK(res);
    createPipelines();
  }

  ComputePipelines() = delete;
  ComputePipelines(const ComputePipelines&) = delete;
  ComputePipelines& operator=(const ComputePipelines&) = delete;

  ~ComputePipelines() {
    vkDeviceWaitIdle(device);
    destroyPipelines();
    if( cache != VK_NULL_HANDLE ) {
      vkDestroyPipelineCache(device, cache, nullptr);
      cache = VK_NULL_HANDLE; 
    }
  }

private :
  void createPipelines() {
    createConv2dFp32();
    createConv2d3x3BnFp32();
    createConv2d3x3BnReluFp32();
    createConv2d5x5BnFp32();
    createConv2d5x5BnReluFp32();
    createConv2d5x5BnMishFp32();
    createAddPointWiseFp32();
    createMatmulFp32();
    createMatmulTiled4x4x32Fp32();
    createBatchNormMaskFp32();
    createBatchNormMaskReluFp32();
    createBatchNormMaskMishFp32();
    createGlobalPoolingChannelsFp32();
    createValueHeadPoolingChannelsFp32();
    createSumChannelsFp32();
    createAddChannelBiasNCHWFp32();
    createAddChannelBiasNCReluFp32();
    createAddChannelBiasNCMishFp32();
    createExtractChannel0NCHWFp32();
  }

  void destroyPipelines() {
    destroyPipeline(conv2dFp32);
    destroyPipeline(conv2d3x3BnFp32);
    destroyPipeline(conv2d3x3BnReluFp32);
    destroyPipeline(conv2d5x5BnFp32);
    destroyPipeline(conv2d5x5BnReluFp32);
    destroyPipeline(conv2d5x5BnMishFp32);
    destroyPipeline(addPointWiseFp32);
    destroyPipeline(matmulFp32);
    destroyPipeline(matmulTiledChw4x4x32Fp32);
    destroyPipeline(batchNormMaskFp32);
    destroyPipeline(batchNormMaskReluFp32);
    destroyPipeline(batchNormMaskMishFp32);
    destroyPipeline(globalPoolingChannelsFp32);
    destroyPipeline(valueHeadPoolingChannelsFp32);
    destroyPipeline(sumChannelsFp32);
    destroyPipeline(addChannelBiasNCHWFp32);
    destroyPipeline(addChannelBiasNCReluFp32);
    destroyPipeline(addChannelBiasNCMishFp32);
    destroyPipeline(extractChannel0NCHWFp32);
  }

  /**
   * @brief cleanup pipeline resources
   */
  void destroyPipeline(Pipeline& pipeline) {
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

  void createPipeline(
    std::string pipelineName,
    const unsigned char* spirvBytes,
    size_t spirvSize,
    size_t bindingSize,
    uint32_t pushConstantSize,
    Pipeline &outPipeline
  ) {
    VkResult res = VK_ERROR_UNKNOWN;
    std::vector<unsigned char> spirvVec(spirvBytes, spirvBytes + spirvSize);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK_MSG(pipelineName + "ShaderModule",res);
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for ( size_t i = 0 ; i < bindingSize ; i++ ) {
      bindings.push_back(
        VkHelpers::descriptorSetLayoutBinding(
          static_cast<uint32_t>(i),
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        )
      );
    }

    outPipeline.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(device,bindings,&res);
    CHECK_VK_MSG(pipelineName + "DescriptorSetLayout",res);

    std::vector<VkPushConstantRange> pushConstants;
    VkPushConstantRange pushConstant = {};

    if ( pushConstantSize > 0 ) {
      pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      pushConstant.offset = 0;
      pushConstant.size = static_cast<uint32_t>(pushConstantSize);
      pushConstants.push_back(pushConstant);
    }
    outPipeline.layout = VkHelpers::createPipelineLayout(device,{ outPipeline.descriptorSetLayout }, pushConstants, &res);
    CHECK_VK_MSG(pipelineName + "PipelineLayout",res);
    outPipeline.pipeline = VkHelpers::createComputePipeline(device, outPipeline.layout, cache, shaderModule, &res);
    CHECK_VK_MSG(pipelineName + "ComputePipeline",res);
    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  /**
   * @brief Create a Conv2d Fp32 object
  */
  void createConv2dFp32() {
    createPipeline("Conv2dFp32",  VkSPIRVShaders::spirv_conv2d_fp32, VkSPIRVShaders::spirv_conv2d_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2dFp32);
  }

  /**
   * @brief Create a Conv2d3x3 Bn + Identity Activation fused Fp32 objects.
   */
  void createConv2d3x3BnFp32() {
    createPipeline("Conv2d3x3BnFp32", VkSPIRVShaders::spirv_conv2d_3x3_bn_fp32, VkSPIRVShaders::spirv_conv2d_3x3_bn_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d3x3BnFp32);
  }

  /**
   * @brief Create a Conv2d3x3 Bn + ReLU Activation fused Fp32 objects.
   */
  void createConv2d3x3BnReluFp32() {
    createPipeline("Conv2d3x3BnReluFp32", VkSPIRVShaders::spirv_conv2d_3x3_bn_relu_fp32, VkSPIRVShaders::spirv_conv2d_3x3_bn_relu_fp32_size,   3, sizeof(Conv2DPushConstantParams), conv2d3x3BnReluFp32);
  }
  /**
   * @brief Create a Conv2d3x3 Bn Mish Fp32 object
   */
  void createConv2d3x3BnMishFp32() {
    createPipeline("Conv2d3x3BnMishFp32", VkSPIRVShaders::spirv_conv2d_3x3_bn_mish_fp32,VkSPIRVShaders::spirv_conv2d_3x3_bn_mish_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d3x3BnMishFp32);
  }

  /**
   * @brief Create a Conv2d5x5 Bn + Identity Activation fused Fp32 objects.
   */
  void createConv2d5x5BnFp32() {
    createPipeline("Conv2d5x5BnFp32", VkSPIRVShaders::spirv_conv2d_5x5_bn_fp32, VkSPIRVShaders::spirv_conv2d_5x5_bn_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnFp32);
  }

  /**
   * @brief Create a Conv2d5x5 Bn + ReLU Activation fused Fp32 objects.
   */
  void createConv2d5x5BnReluFp32() {
    createPipeline("Conv2d5x5BnReluFp32",VkSPIRVShaders::spirv_conv2d_5x5_bn_relu_fp32, VkSPIRVShaders::spirv_conv2d_5x5_bn_relu_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnReluFp32);
  }

  /**
   * @brief Create a Conv2d5x5 Bn Mish Fp32 object
   */
  void createConv2d5x5BnMishFp32() {
    createPipeline("Conv2d5x5BnMishFp32",VkSPIRVShaders::spirv_conv2d_5x5_bn_mish_fp32, VkSPIRVShaders::spirv_conv2d_5x5_bn_mish_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnMishFp32);
  }

  /**
   * @brief Create a Add Point Wise Fp32 object
   */
  void createAddPointWiseFp32() {
    createPipeline("AddPointWiseFp32",VkSPIRVShaders::spirv_add_pointwise_fp32, VkSPIRVShaders::spirv_add_pointwise_fp32_size, 3, sizeof(NCHWPushConstantParams), addPointWiseFp32);
  }

  /**
   * @brief Create a Matmul Fp32 object
   */
  void createMatmulFp32() {
    createPipeline("MatmulFp32", VkSPIRVShaders::spirv_matmul_fp32, VkSPIRVShaders::spirv_matmul_fp32_size, 3, sizeof(MatmulPushConstantParams), matmulFp32);
  }

  /**
   * @brief Create a Matmul Tiled 4x4x32 Fp32 object
   */
  void createMatmulTiled4x4x32Fp32() {
    createPipeline("MatmulTiled4x4x32Fp32", VkSPIRVShaders::spirv_matmul_tiled_chw_4x4x32_fp32, VkSPIRVShaders::spirv_matmul_tiled_chw_4x4x32_fp32_size, 3, sizeof(MatmulPushConstantParams), matmulTiledChw4x4x32Fp32);
  }

  /**
   * @brief Create a BatchNorm Mask Fp32 object
   */
  void createBatchNormMaskFp32() {
    createPipeline("BatchNormMaskFp32", VkSPIRVShaders::spirv_bn_mask_fp32, VkSPIRVShaders::spirv_bn_mask_fp32_size, 3, sizeof(NCHWPushConstantParams), batchNormMaskFp32);
  }

  /**
   * @brief Create a BatchNorm Mask + ReLU Fp32 object
   */
  void createBatchNormMaskReluFp32() {
    createPipeline("BatchNormMaskReluFp32", VkSPIRVShaders::spirv_bn_mask_relu_fp32, VkSPIRVShaders::spirv_bn_mask_relu_fp32_size, 3, sizeof(NCHWPushConstantParams), batchNormMaskReluFp32);
  }

  /**
   * @brief Create a BatchNorm Mask + Mish Fp32 object
   */
  void createBatchNormMaskMishFp32() {
    createPipeline("BatchNormMaskMishFp32", VkSPIRVShaders::spirv_bn_mask_mish_fp32, VkSPIRVShaders::spirv_bn_mask_mish_fp32_size, 3, sizeof(NCHWPushConstantParams), batchNormMaskMishFp32);
  }

  /**
   * @brief Create a Global Average Pool Fp32 object
   */
  void createGlobalPoolingChannelsFp32() {
    createPipeline("GlobalPoolingChannelsFp32", VkSPIRVShaders::spirv_global_pooling_channels_fp32, VkSPIRVShaders::spirv_global_pooling_channels_fp32_size, 2, sizeof(NCHWPushConstantParams), globalPoolingChannelsFp32);
  }

  /**
   * @brief Create a Value Head Pool Channels Fp32 object
   */
  void createValueHeadPoolingChannelsFp32() {
    createPipeline("ValueHeadPoolingChannelsFp32", VkSPIRVShaders::spirv_value_head_pool_channels_fp32, VkSPIRVShaders::spirv_value_head_pool_channels_fp32_size, 2, sizeof(NCHWPushConstantParams), valueHeadPoolingChannelsFp32);
  }

  /**
   * @brief Create a Sum Channels Fp32 object
   */
  void createSumChannelsFp32() {
    createPipeline("SumChannelsFp32", VkSPIRVShaders::spirv_sum_channels_fp32, VkSPIRVShaders::spirv_sum_channels_fp32_size, 2, sizeof(NCHWPushConstantParams), sumChannelsFp32);
  }

  /**
   * @brief Create a Add Channel Bias NCHW Fp32 object
   */
  void createAddChannelBiasNCHWFp32() {
    createPipeline("AddChannelBiasNCHWFp32", VkSPIRVShaders::spirv_add_channel_bias_nchw_fp32, VkSPIRVShaders::spirv_add_channel_bias_nchw_fp32_size, 2, sizeof(NCHWPushConstantParams), addChannelBiasNCHWFp32);
  }

  /**
   * @brief Create a Add Channel Bias NC + ReLU Fp32 object
   */
  void createAddChannelBiasNCReluFp32() {
    createPipeline("AddChannelBiasNCReluFp32", VkSPIRVShaders::spirv_add_channel_bias_nc_relu_fp32, VkSPIRVShaders::spirv_add_channel_bias_nc_relu_fp32_size, 2, sizeof(NCHWPushConstantParams), addChannelBiasNCReluFp32);
  }

  /**
   * @brief Create a Add Channel Bias NC + Mish Fp32 object
   */
  void createAddChannelBiasNCMishFp32() {
    createPipeline("AddChannelBiasNCMishFp32", VkSPIRVShaders::spirv_add_channel_bias_nc_mish_fp32, VkSPIRVShaders::spirv_add_channel_bias_nc_mish_fp32_size, 2, sizeof(NCHWPushConstantParams), addChannelBiasNCMishFp32);
  }
  
  /**
   * @brief Create a Extract Channel 0 NCHW Fp32 object
   */
  void createExtractChannel0NCHWFp32() {
    createPipeline("ExtractChannel0NCHWFp32", VkSPIRVShaders::spirv_extract_channel0_nchw_fp32, VkSPIRVShaders::spirv_extract_channel0_nchw_fp32_size, 2, sizeof(NCHWPushConstantParams), extractChannel0NCHWFp32);
  }
};




ComputePipelines* createComputePipelines(
  ComputeContext* context,
  int gpuIdx,
  bool useFP16TensorCores,
  bool useFP16TensorCoresFor1x1,
  bool useNHWC,
  Logger* logger
) {
  VkDevice device = context->vulkanContext->findGpuExn(gpuIdx)->device;
  ComputePipelines* pipelines = new ComputePipelines(device);
  return pipelines;
}

void NeuralNet::globalInitialize() {
  static_assert(sizeof(int) >= 4, "");
}

void NeuralNet::globalCleanup() {
}

#endif