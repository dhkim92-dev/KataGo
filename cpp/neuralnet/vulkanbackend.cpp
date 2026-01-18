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
  Pipeline globalAvgPoolFp32;
  Pipeline valueHeadPoolChannelsFp32;

  Pipeline sumChannelsFp32;


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
    createConv2dFp32(device);
    createConv2d3x3BnFp32(device);
    createConv2d3x3BnReluFp32(device);
    createConv2d5x5BnFp32(device);
    createConv2d5x5BnReluFp32(device);
    createConv2d5x5BnMishFp32(device);
    createAddPointWiseFp32(device);
    createMatmulFp32(device);
    createMatmulTiled4x4x32Fp32(device);
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

  /**
   * @brief Create a Conv2d Fp32 object
  */
  void createConv2dFp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirvBytes = VkSPIRVShaders::spirv_conv2d_fp32;
    const size_t spirvSize = VkSPIRVShaders::spirv_conv2d_fp32_size;
    std::vector<unsigned char> spirvVec(spirvBytes, spirvBytes + spirvSize);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK_MSG("createConv2dFp32ShaderModule",res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // filter buffers
    };

    conv2dFp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );
    CHECK_VK_MSG("createConv2dFp32DescriptorSetLayout",res);

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(Conv2DPushConstantParams); // e.g., could be used for kernel size, stride, padding, etc.
    conv2dFp32.layout = VkHelpers::createPipelineLayout(
      device,
      { conv2dFp32.descriptorSetLayout },
      { pushConstant },
      &res
    );
    CHECK_VK_MSG("createConv2dFp32PipelineLayout",res);

    conv2dFp32.pipeline = VkHelpers::createComputePipeline(
      device,
      conv2dFp32.layout,
      cache,
      shaderModule,
      &res
    );
    CHECK_VK(res);
    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  /**
   * @brief Create a Conv2d3x3 Bn + Identity Activation fused Fp32 objects.
   * @param device 
   */
  void createConv2d3x3BnFp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirvBytes = VkSPIRVShaders::spirv_conv2d_3x3_bn_fp32;
    const size_t spirvSize = VkSPIRVShaders::spirv_conv2d_3x3_bn_fp32_size;
    std::vector<unsigned char> spirvVec(spirvBytes, spirvBytes + spirvSize);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK_MSG("createConv2d3x3Fp32ShaderModule",res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // filter buffers
    };

    conv2d3x3BnFp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );
    CHECK_VK_MSG("createConv2d3x3Fp32DescriptorSetLayout", res);

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(Conv2DPushConstantParams); // e.g., could be used for kernel size, stride, padding, etc.
    conv2d3x3BnFp32.layout = VkHelpers::createPipelineLayout(
      device,
      { conv2d3x3BnFp32.descriptorSetLayout },
      { pushConstant },
      &res
    );
    CHECK_VK_MSG("createConv2d3x3Fp32PipelineLayout", res);

    conv2d3x3BnFp32.pipeline = VkHelpers::createComputePipeline(
      device,
      conv2d3x3BnFp32.layout,
      cache,
      shaderModule,
      &res
    );
    CHECK_VK(res);
    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  /**
   * @brief Create a Conv2d3x3 Bn + ReLU Activation fused Fp32 objects.
   */
  void createConv2d3x3BnReluFp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirvBytes = VkSPIRVShaders::spirv_conv2d_3x3_bn_relu_fp32;
    const size_t spirvSize = sizeof(VkSPIRVShaders::spirv_conv2d_3x3_bn_relu_fp32);
    std::vector<unsigned char> spirvVec(spirvBytes, spirvBytes + spirvSize);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK_MSG("createConv2d3x3Fp32ShaderModule",res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // filter buffers
    };

    conv2d3x3BnReluFp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );
    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(Conv2DPushConstantParams);

    // 3x3 Conv Pipeline 
    conv2d3x3BnReluFp32.layout = VkHelpers::createPipelineLayout(
      device,
      { conv2d3x3BnReluFp32.descriptorSetLayout },
      { pushConstant },
      &res
    );
    CHECK_VK_MSG("createConv2d3x3Fp32PipelineLayout",res);

    conv2d3x3BnReluFp32.pipeline = VkHelpers::createComputePipeline(
      device,
      conv2d3x3BnReluFp32.layout,
      cache,
      shaderModule,
      &res
    );

    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  /**
   * @brief Create a Conv2d3x3 Bn Mish Fp32 object
   */
  void createConv2d3x3BnMishFp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirvBytes = VkSPIRVShaders::spirv_conv2d_3x3_bn_mish_fp32;
    const size_t spirvSize = VkSPIRVShaders::spirv_conv2d_3x3_bn_mish_fp32_size;
    std::vector<unsigned char> spirvVec(spirvBytes, spirvBytes + spirvSize);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK_MSG("createConv2d3x3MishFp32ShaderModule",res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // filter buffers
    };

    conv2d3x3BnMishFp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );
    CHECK_VK_MSG("createConv2d3x3MishFp32DescriptorSetLayout", res);

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(Conv2DPushConstantParams); // e.g., could be used for kernel size, stride, padding, etc.
    conv2d3x3BnMishFp32.layout = VkHelpers::createPipelineLayout(
      device,
      { conv2d3x3BnMishFp32.descriptorSetLayout },
      { pushConstant },
      &res
    );
    CHECK_VK_MSG("createConv2d3x3MishFp32PipelineLayout", res);

    conv2d3x3BnMishFp32.pipeline = VkHelpers::createComputePipeline(
      device,
      conv2d3x3BnMishFp32.layout,
      cache,
      shaderModule,
      &res
    );
    CHECK_VK(res);
    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  /**
   * @brief Create a Conv2d5x5 Bn + Identity Activation fused Fp32 objects.
   */
  void createConv2d5x5BnFp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirvBytes = VkSPIRVShaders::spirv_conv2d_5x5_bn_fp32;
    const size_t spirvSize = VkSPIRVShaders::spirv_conv2d_5x5_bn_fp32_size;
    std::vector<unsigned char> spirvVec(spirvBytes, spirvBytes + spirvSize);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK_MSG("createConv2d5x5Fp32ShaderModule",res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // filter buffers
    };

    conv2d5x5BnFp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );
    CHECK_VK_MSG("createConv2d5x5Fp32DescriptorSetLayout", res);

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(Conv2DPushConstantParams); // e.g., could be used for kernel size, stride, padding, etc.
    conv2d5x5BnFp32.layout = VkHelpers::createPipelineLayout(
      device,
      { conv2d5x5BnFp32.descriptorSetLayout },
      { pushConstant },
      &res
    );
    CHECK_VK_MSG("createConv2d5x5Fp32PipelineLayout", res);

    conv2d5x5BnFp32.pipeline = VkHelpers::createComputePipeline(
      device,
      conv2d5x5BnFp32.layout,
      cache,
      shaderModule,
      &res
    );
    CHECK_VK(res);
    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  /**
   * @brief Create a Conv2d5x5 Bn + ReLU Activation fused Fp32 objects.
   */
  void createConv2d5x5BnReluFp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirvBytes = VkSPIRVShaders::spirv_conv2d_5x5_bn_relu_fp32;
    const size_t spirvSize = sizeof(VkSPIRVShaders::spirv_conv2d_5x5_bn_relu_fp32);
    std::vector<unsigned char> spirvVec(spirvBytes, spirvBytes + spirvSize);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK_MSG("createConv2d5x5BnReluFp32ShaderModule",res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // filter buffers
    };

    conv2d5x5BnReluFp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );
    CHECK_VK_MSG("createConv2d5x5BnReluFp32DescriptorSetLayout", res);

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(Conv2DPushConstantParams);

    // 5x5 Conv Pipeline 
    conv2d5x5BnReluFp32.layout = VkHelpers::createPipelineLayout(
      device,
      { conv2d5x5BnReluFp32.descriptorSetLayout },
      { pushConstant },
      &res
    );
    CHECK_VK_MSG("createConv2d5x5BnReluFp32PipelineLayout",res);
    conv2d5x5BnReluFp32.pipeline = VkHelpers::createComputePipeline(
      device,
      conv2d5x5BnReluFp32.layout,
      cache,
      shaderModule,
      &res
    );

    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  void createConv2d5x5BnMishFp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirvBytes = VkSPIRVShaders::spirv_conv2d_5x5_bn_mish_fp32;
    const size_t spirvSize = VkSPIRVShaders::spirv_conv2d_5x5_bn_mish_fp32_size;
    std::vector<unsigned char> spirvVec(spirvBytes, spirvBytes + spirvSize);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK_MSG("createConv2d5x5MishFp32ShaderModule",res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // filter buffers
    };

    conv2d5x5BnMishFp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );
    CHECK_VK_MSG("createConv2d5x5MishFp32DescriptorSetLayout", res);

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(Conv2DPushConstantParams); // e.g., could be used for kernel size, stride, padding, etc.
    conv2d5x5BnMishFp32.layout = VkHelpers::createPipelineLayout(
      device,
      { conv2d5x5BnMishFp32.descriptorSetLayout },
      { pushConstant },
      &res
    );
    CHECK_VK_MSG("createConv2d5x5MishFp32PipelineLayout", res);

    conv2d5x5BnMishFp32.pipeline = VkHelpers::createComputePipeline(
      device,
      conv2d5x5BnMishFp32.layout,
      cache,
      shaderModule,
      &res
    );
    CHECK_VK_MSG("createConv2d5x5MishFp32Pipeline", res);
    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  /**
   * @brief Create a Add Point Wise Fp32 object
   */
  void createAddPointWiseFp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirvBytes = VkSPIRVShaders::spirv_add_pointwise_fp32;
    const size_t spirvSize = VkSPIRVShaders::spirv_add_pointwise_fp32_size;
    std::vector<unsigned char> spirvVec(spirvBytes, spirvBytes + spirvSize);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK_MSG("createAddPointWiseFp32ShaderModule",res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer A
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer B
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
    };

    addPointWiseFp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );
    CHECK_VK_MSG("createAddPointWiseFp32DescriptorSetLayout", res);

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(NCHWPushConstantParams); // e.g., could be used for N, C, H, W
    addPointWiseFp32.layout = VkHelpers::createPipelineLayout(
      device,
      { addPointWiseFp32.descriptorSetLayout },
      { pushConstant },
      &res
    );
    CHECK_VK_MSG("createAddPointWiseFp32PipelineLayout", res);

    addPointWiseFp32.pipeline = VkHelpers::createComputePipeline(
      device,
      addPointWiseFp32.layout,
      cache,
      shaderModule,
      &res
    );
    CHECK_VK(res);
    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  /**
   * @brief create a Matmul Fp32 object
   */
  void createMatmulFp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirvBytes = VkSPIRVShaders::spirv_matmul_fp32;
    const size_t spirvSize = VkSPIRVShaders::spirv_matmul_fp32_size;
    std::vector<unsigned char> spirvVec(spirvBytes, spirvBytes + spirvSize);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK_MSG("createMatmulFp32ShaderModule",res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer A
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer B
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
    };

    matmulFp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );
    CHECK_VK_MSG("createMatmulFp32DescriptorSetLayout", res);

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(MatmulPushConstantParams); // e.g., could be used for M, N, K
    matmulFp32.layout = VkHelpers::createPipelineLayout(
      device,
      { matmulFp32.descriptorSetLayout },
      { pushConstant },
      &res
    );
    CHECK_VK_MSG("createMatmulFp32PipelineLayout", res);

    matmulFp32.pipeline = VkHelpers::createComputePipeline(
      device,
      matmulFp32.layout,
      cache,
      shaderModule,
      &res
    );
    CHECK_VK_MSG("createMatmulFp32Pipeline", res);
    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  /**
   * @brief create a Matmul Tiled 4x4x32 Fp32 object
   */
  void createMatmulTiled4x4x32Fp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirvBytes = VkSPIRVShaders::spirv_matmul_tiled_chw_4x4x32_fp32;
    const size_t spirvSize = VkSPIRVShaders::spirv_matmul_tiled_chw_4x4x32_fp32_size;
    std::vector<unsigned char> spirvVec(spirvBytes, spirvBytes + spirvSize);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK_MSG("createMatmulTiled4x4x32Fp32ShaderModule",res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer A
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer B
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
    };

    matmulTiledChw4x4x32Fp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );
    CHECK_VK_MSG("createMatmulTiled4x4x32Fp32DescriptorSetLayout", res);

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(MatmulPushConstantParams); // e.g., could be used for M, N, K
    matmulTiledChw4x4x32Fp32.layout = VkHelpers::createPipelineLayout(
      device,
      { matmulTiledChw4x4x32Fp32.descriptorSetLayout },
      { pushConstant },
      &res
    );
    CHECK_VK_MSG("createMatmulTiled4x4x32Fp32PipelineLayout", res);

    matmulTiledChw4x4x32Fp32.pipeline = VkHelpers::createComputePipeline(
      device,
      matmulTiledChw4x4x32Fp32.layout,
      cache,
      shaderModule,
      &res
    );
    CHECK_VK_MSG("createMatmulTiled4x4x32Fp32Pipeline", res);
    vkDestroyShaderModule(device, shaderModule, nullptr);
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