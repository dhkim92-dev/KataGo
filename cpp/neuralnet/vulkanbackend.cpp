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


// TODO: Tune params to be filled later
struct VulkanTuneParams {

};

struct Pipeline {
  VkPipelineLayout layout;
  VkPipeline pipeline;
  VkDescriptorSetLayout descriptorSetLayout;
  VkDescriptorSet descriptorSet;
};

struct Conv2DPushConstantParams {
  int N; // Batch size
  int inC; // Input channels
  int inH; // Input height
  int inW; // Input width
  int dilation;
  int outC; // Output channels
  int outH; // Output height
  int outW; // Output width
};

/**
 * @brief Compute pipelines for various operations
 */
struct ComputePipelines {
  const VulkanTuneParams tuneParams;
  const bool usingFP16Storage;
  const bool usingFP16Compute;
  const bool usingFP16TensorCores;
  const bool usingFP16TensorCoresForConv1x1;
  VkDevice device;
  VkPipelineCache cache;

  // In this code, assume that NCHW is default format unless no prefix is given.
  Pipeline conv2d3x3Fp32;
  Pipeline conv2d5x5Fp32;
  Pipeline conv2d1x1F32;   
  Pipeline matmulTiled4x4Fp32;
  Pipeline matmulTiled8x8Fp32;
  Pipeline matBiasFp32;
  Pipeline batchNormFp32;  
  Pipeline gPoolFp32;
  Pipeline sumMaskFp32;
  // Activation functions
  Pipeline reluFp32;
  Pipeline mishFp32;  
  Pipeline identityFp32;
  // Elemental wise operations
  Pipeline addPointWiseFp32;
  Pipeline multiplyPointWisepFp32;
  Pipeline axpbyPointWiseFp32;
  Pipeline scaleFp32;
  
  //

  ComputePipelines(VkDevice device): device(device) {
    VkResult res = VK_ERROR_UNKNOWN;
    cache = VkHelpers::createPipelineCache(device, &res);
    CHECK_VK(res);
    createConv2D3x3Fp32(device);
    createConv2D5x5Fp32(device);
    createConv2D1x1Fp32(device);
    createMatmulTiled4x4Fp32(device);
    createMatmulTiled8x8Fp32(device);
  }

  ComputePipelines() = delete;
  ComputePipelines(const ComputePipelines&) = delete;
  ComputePipelines& operator=(const ComputePipelines&) = delete;

  ~ComputePipelines() {

    // Destroy pipelines and layouts descriptor set layouts
    vkDestroyPipelineCache(device, cache, nullptr);
  }

private :
  Pipeline createConv2D3x3Fp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirv_bytes = VkSPIRVShaders::spirv_conv2d_3x3_fp32;
    const size_t spirv_size = sizeof(VkSPIRVShaders::spirv_conv2d_3x3_fp32);
    std::vector<unsigned char> spirvVec(spirv_bytes, spirv_bytes + spirv_size);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK(res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // filter buffers
    };

    conv2d3x3Fp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(Conv2DPushConstantParams); // e.g., could be used for kernel size, stride, padding, etc.
    conv2d3x3Fp32.layout = VkHelpers::createPipelineLayout(
      device,
      { conv2d3x3Fp32.descriptorSetLayout },
      { pushConstant },
      &res
    );

    CHECK_VK(res);

    conv2d3x3Fp32.pipeline = VkHelpers::createComputePipeline(
      device,
      conv2d3x3Fp32.layout,
      cache,
      shaderModule,
      &res
    );

    CHECK_VK(res);
    vkDestroyShaderModule(device, shaderModule, nullptr);
  }

  Pipeline createConv2D3x3Fp32(VkDevice& device) {
    VkResult res = VK_ERROR_UNKNOWN;
    const unsigned char* spirv_bytes = VkSPIRVShaders::spirv_conv2d_3x3_fp32;
    const size_t spirv_size = sizeof(VkSPIRVShaders::spirv_conv2d_3x3_fp32);
    std::vector<unsigned char> spirvVec(spirv_bytes, spirv_bytes + spirv_size);
    VkShaderModule shaderModule = VkHelpers::createShaderModuleFromSPIRVBytes(
      device,
      spirvVec,
      &res
    );
    CHECK_VK(res);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      VkHelpers::descriptorSetLayoutBinding( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Input buffer
      VkHelpers::descriptorSetLayoutBinding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // Ouptut buffer
      VkHelpers::descriptorSetLayoutBinding( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ), // filter buffers
    };

    conv2d3x3Fp32.descriptorSetLayout = VkHelpers::createDescriptorSetLayout(
      device,
      bindings,
      &res
    );

    VkPushConstantRange pushConstant = {};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(Conv2DPushConstantParams); // e.g., could be used for kernel size, stride, padding, etc.
    conv2d3x3Fp32.layout = VkHelpers::createPipelineLayout(
      device,
      { conv2d3x3Fp32.descriptorSetLayout },
      { pushConstant },
      &res
    );

    CHECK_VK(res);

    conv2d3x3Fp32.pipeline = VkHelpers::createComputePipeline(
      device,
      conv2d3x3Fp32.layout,
      cache,
      shaderModule,
      &res
    );

    CHECK_VK(res);
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