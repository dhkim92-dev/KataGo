/**
 * @file vulkanbackend.cpp
 * @author Dohoon Kim(https://github.com/dhkim92-dev, dhkim92-dev@gmail.com, https://www.dohoon-kim.kr)
 * @brief Vulkan backend for Neural Net evaluation
 */
#ifdef USE_VULKAN_BACKEND

#include <unordered_map>
#include <memory>
#include "../core/global.h"
#include "../core/simpleallocator.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/vulkanbackend.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkanshaders.h"

struct ComputeContext {
  const std::vector<uint32_t>* gIdx;
  const int nnXLen;
  const int nnYLen;
  const enabled_t usingFP16Mode;
  const enabled_t usingNHWCMode;
  VulkanContext* vulkanContext; 
  std::unordered_map<uint32_t, KatagoVulkan::ComputePipelines *> pipelinesPerDev;
  
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
        //       VK_KHR_cooperative_matrix extension is required for tensor cores usage.

        // Check for NHWC support if requested
        if ( usingNHWCMode == enabled_t::True && !isDeviceSupportNHWC(deviceInfo) ) {
          throw StringError("Requested NHWC mode but device " + deviceInfo.deviceName + " does not support it");
        }

        if ( usingNHWCMode != enabled_t::False && isDeviceSupportNHWC(deviceInfo) ) {
          requiredExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        }

        VulkanDevice* vulkanDevice = VkHelpers::createVulkanDevice(
          instance,
          deviceInfo,
          requiredExtensions,
          logger
        );
        vulkanDevices.push_back(vulkanDevice);
        KatagoVulkan::ComputePipelines* pipelines = new KatagoVulkan::ComputePipelines(vulkanDevice->device);
        this->pipelinesPerDev.emplace(gpuIdx, pipelines);
      }

      vulkanContext = new VulkanContext(
        instance,
        vulkanDevices,
        logger
      );
  }

  ~ComputeContext() {
    for ( auto& kv : pipelinesPerDev ) {
      KatagoVulkan::ComputePipelines* pipelines = kv.second;
      delete pipelines;
    }

    for ( VulkanDevice *device : vulkanContext->devicesToUse ) {
      delete device;
    }
    vulkanContext->devicesToUse.clear();

    delete vulkanContext;
  }

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;

  bool isDeviceSupportFp16(const VulkanDeviceInfo& deviceInfo) {
    //Check for fp16 feature
    return deviceInfo.shaderFloat16Int8Features.shaderFloat16 == VK_TRUE;
  }

  bool isDeviceSupportNHWC(const VulkanDeviceInfo& deviceInfo) {
    return deviceInfo.properties.apiVersion >= VK_API_VERSION_1_1 &&
           deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE;
  }
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger *logger,
  int nnXLen,
  int nnYLen,
  const std::string& openCLTunerFile,
  const std::string& homeDataDirOverride,
  bool openCLTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  // VkDeviceSize requiredMemorySize = getRequiredMemorySize(loadedModel);
  return new ComputeContext(
    nnXLen,
    nnYLen,
    useFP16Mode,
    useNHWCMode,
    // requiredMemorySize,
    std::vector<uint32_t>(gpuIdxs.begin(), gpuIdxs.end()),
    
    logger
  );
}

VkDeviceSize getRequiredMemorySize(
  const LoadedModel* loadedModel
) {
  // For simplicity, return a fixed size for now.
  // In future, we can calculate based on model parameters.
  return static_cast<VkDeviceSize>(512) * 1024 * 1024; // 512 MB
}

void NeuralNet::freeComputeContext(ComputeContext* context) {
  delete context;
}

static ComputeContext* createComputeContextForTesting(
  const std::vector<int>& gpuIdxs,
  Logger *logger,
  int nnXLen,
  int nnYLen,
  bool useFp16,
  bool useNHWC
) {
  enabled_t useFP16Mode = useFp16 ? enabled_t::True : enabled_t::False;
  enabled_t useNHWCMode = useNHWC ? enabled_t::True : enabled_t::False;

  return new ComputeContext(
    nnXLen,
    nnYLen,
    useFP16Mode,
    useNHWCMode,
    std::vector<uint32_t>(gpuIdxs.begin(), gpuIdxs.end()),
    logger
  );
}

/* ########################### Buffers ######################### */
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


struct ScratchBuffers {
  const size_t batchXYFloatBytes;
  const size_t batchFloatBytes;
  const size_t batchXYBytes;
  const size_t batchBytes;

  const ComputeHandleInternal *handle;
  SimpleAllocator<VulkanBuffer *> *allocator;

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;

  ScratchBuffers(
    ComputeHandleInternal* handle_,
    int maxBatchSize,
    int nnXLen,
    int nnYLen
  ): 
    batchXYFloatBytes(sizeof(float) * maxBatchSize * nnXLen * nnYLen),
    batchFloatBytes(sizeof(float) * maxBatchSize),
    batchXYBytes(sizeof(uint8_t) * maxBatchSize * nnXLen * nnYLen),
    batchBytes(sizeof(uint8_t) * maxBatchSize),
    handle(handle_)
  {
    std::function<VulkanBuffer*(size_t)> allocFunc = [this](size_t size) {
      return VkHelpers::createDeviceBuffer(
        handle->vulkanDevice,
        size,
        false,
        nullptr
      );
    };
    std::function<void(VulkanBuffer *)> freeFunc = [this](VulkanBuffer *buffer) {
      VkHelpers::releaseVulkanBuffer(
        handle->vulkanDevice,
        buffer
      );
    };

    allocator = new SimpleAllocator<VulkanBuffer *>(
      allocFunc,
      freeFunc
    );
  }

  size_t getBufSizeXY(int channels) const {
    return static_cast<size_t>(channels) * batchXYBytes;
  }

  size_t getBufSizeXYFloat(int channels) const {
    return static_cast<size_t>(channels) * batchXYFloatBytes;
  }

  size_t getBufSizeFloat(int channels) const {
    return static_cast<size_t>(channels) * batchFloatBytes;
  }

  size_t getBufSize(int channels) const {
    return static_cast<size_t>(channels) * batchBytes;
  }
};

/**
 * @brief Copy of OpenCL ConvWorkspaceEltsNeeded struct
 */
struct ConvWorkspaceEltsNeeded {
  size_t size1;
  size_t size2;
  ConvWorkspaceEltsNeeded()
    :size1(0),size2(0)
  {}
  ConvWorkspaceEltsNeeded(size_t s1, size_t s2)
    :size1(s1),size2(s2)
  {}
  static ConvWorkspaceEltsNeeded getMax(ConvWorkspaceEltsNeeded a, ConvWorkspaceEltsNeeded b) {
    return ConvWorkspaceEltsNeeded(std::max(a.size1,b.size1),std::max(a.size2,b.size2));
  }
};


/**
 * @brief Matrix Multiplication Layer
 */
struct MatmulLayer {
  const std::string name;
  const ComputeHandleInternal *handle;
  const int inChannels;
  const int outChannels;

  VulkanBuffer* matBuf = nullptr;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

  MatmulLayer(
    ComputeHandleInternal *handle_,
    const MatMulLayerDesc* desc
  ): 
    name(desc->name),
    handle(handle_),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels)
  {

    if ( inChannels > 0 && outChannels > 0 ) {
      assert(desc->weights.size() == static_cast<size_t>(inChannels) * static_cast<size_t>(outChannels));
      std::vector<float> weights(desc->weights.size());

      // Transpose weights from (inC x outC) to (outC x inC)
      // It is because Input Matrix I( M x K ) x Weights W( K x N ) = Output Matrix O( M x N )
      // If keep the originial layout, then memory access efficiency would be low.
      for ( int oc = 0 ; oc < outChannels ; oc++ ) {
        for ( int ic = 0 ; ic < inChannels ; ic++ ) {
          weights[oc * inChannels + ic] = desc->weights[ic * outChannels + oc];
        }
      }
      VkResult res;
      matBuf = VkHelpers::createDeviceBufferWithData(
        handle->vulkanDevice,
        sizeof(float) * weights.size(),
        weights.data(),
        true,
        &res
      );
      CHECK_VK_MSG("Create MatmulLayer: " + name + " buffer", res);
    }
  }

  ~MatmulLayer() {
    if ( matBuf != nullptr ) {
      VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, matBuf);
      delete matBuf;
      matBuf = nullptr;
    }
  }

  /**
   * @brief create command buffer and record for matmul layer
   * @param batchSize 
   * @param input 
   * @param output 
   */
  void record(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    if ( commandBuffer != VK_NULL_HANDLE ) {
      return; // Already recorded
    }

    commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
    VkResult res = VkHelpers::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for MatmulLayer: " + name, res);
    doMatmulFp32(
      batchSize,
      input,
      output
    );
    res = VkHelpers::endCommandBuffer(commandBuffer);
    CHECK_VK_MSG("End command buffer for MatmulLayer: " + name, res);
  }

  /**
   * @brief 
   * @param batchSize 
   * @param input 
   * @param output 
   */
  VkCommandBuffer apply(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    assert(commandBuffer != VK_NULL_HANDLE);
    return commandBuffer;
  }

private:
  void doMatmulFp32(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    auto *pipelines = handle->context->pipelinesPerDev.at(gpuId);
    VkResult res;
    descriptorSet = VkHelpers::allocateDescriptorSet(
      handle->vulkanDevice,
      pipelines->matmulFp32.descriptorSetLayout,
      &res
    );
    CHECK_VK_MSG("Allocate descriptor set for MatmulLayer: " + name, res);

    // update descriptor set
    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
      VkHelpers::writeDescriptorSetBuffer(
        descriptorSet,
        0,
        input
      ),
      VkHelpers::writeDescriptorSetBuffer(
        descriptorSet,
        1,
        matBuf
      ),
      VkHelpers::writeDescriptorSetBuffer(
        descriptorSet,
        2,
        output
      )
    };
    VkHelpers::updateDescriptorSets(
      handle->vulkanDevice,
      writeDescriptorSets
    );

    auto pushConstants = KatagoVulkan::MatmulFp32Params();
    pushConstants.M = static_cast<uint32_t>(inChannels);
    pushConstants.K = static_cast<uint32_t>(outChannels);
    pushConstants.N = static_cast<uint32_t>(outChannels);
    pushConstants.numBatchElts = static_cast<uint32_t>(batchSize);
    pushConstants.cTranspose = 1; // Weights always transposed, so Output should be transposed.

    vkCmdPushConstants(
      commandBuffer,
      pipelines->matmulFp32.layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(KatagoVulkan::MatmulFp32Params),
      &pushConstants
    );
    vkCmdBindPipeline(
      commandBuffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipelines->matmulFp32.pipeline
    );
    vkCmdBindDescriptorSets(
      commandBuffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipelines->matmulFp32.layout,
      0, 
      1,
      &descriptorSet,
      0,
      nullptr
    );

    uint32_t groupCountX = (outChannels + 31) / 32;
    uint32_t groupCountY = batchSize;
    vkCmdDispatch(
      commandBuffer,
      groupCountX,
      groupCountY,
      1
    );
  }
};


/**
 * @brief Convolution Layer in Vulkan Backend
 * Currently not support winograd and dilation
 * Simple tiled convolution only except 1x1 conv.
 * 1x1 conv implemented with matmul approach. Maybe replaced by cooperative matrix extension later.
 * TODO: Support bn and activation fused conv layers
 */
struct ConvLayer {
  const ComputeHandleInternal* handle;
  const std::string name;
  const int convYSize;
  const int convXSize;
  const int inChannels;
  const int outChannels;
  const int dilationY;
  const int dilationX;
  const int nnXLen;
  const int nnYLen;

  VulkanBuffer* filterBuf = nullptr;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

  static constexpr int nKernelDims = 3;

  ConvLayer(
    ComputeHandleInternal *handle_,
    const ConvLayerDesc* desc,
    int nnXLen_,
    int nnYLen_
  ): 
    handle(handle_),
    name(desc->name),
    convYSize(desc->convYSize),
    convXSize(desc->convXSize),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels),
    dilationY(desc->dilationY),
    dilationX(desc->dilationX),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_)
  {
    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);

    if ( dilationX != 1 || dilationY != 1 ) {
      throw StringError("Vulkan ConvLayer: " + name + " dilation not supported yet");
    }

    VkResult res;
    filterBuf = VkHelpers::createDeviceBufferWithData(
      handle->vulkanDevice,
      sizeof(float) * desc->weights.size(),
      desc->weights.data(),
      true,
      &res
    );
    CHECK_VK_MSG("Create ConvLayer: " + name + " filter buffer", res);
  }

  ~ConvLayer() {
    if ( filterBuf != nullptr ) {
      VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, filterBuf);
      delete filterBuf;
      filterBuf = nullptr;
    }
  }

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  bool canApplyWithBNActivation() {
    return (convXSize == 3 || convXSize == 5) && (convYSize == 3 || convYSize == 5);
  }

  ConvWorkspaceEltsNeeded getConvWorkspaceEltsNeeded(int batchSize) {
    // TODO: Implement workspace size calculation if needed
    return ConvWorkspaceEltsNeeded(0, 0);
  }

  /**
   * @brief create command buffer and record for conv layer, only tiled conv now.
   * @param batchSize 
   * @param input 
   * @param output 
   */
  void record(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    if (  commandBuffer != VK_NULL_HANDLE ) {
      return; // Already recorded
    }

    commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
    VkResult res = VkHelpers::beginCommandBuffer(commandBuffer);
    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    KatagoVulkan::ComputePipelines* pipelines = this->handle->context->pipelinesPerDev.at(gpuId);
    CHECK_VK_MSG("Begin command buffer for ConvLayer: " + name, res);
    
    descriptorSet = VkHelpers::allocateDescriptorSet(
      handle->vulkanDevice,
      pipelines->conv2dFp32.descriptorSetLayout,
      &res
    );

    vkCmdBindPipeline(
      commandBuffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipelines->conv2dFp32.pipeline
    );

    // update descriptor set
    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, input),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, filterBuf),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, output)
    };
    VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
    auto pushConstants = KatagoVulkan::Conv2DPushConstantParams();
    pushConstants.batchSize = static_cast<uint32_t>(batchSize);
    pushConstants.inChannels = static_cast<uint32_t>(inChannels);
    pushConstants.outChannels = static_cast<uint32_t>(outChannels);
    pushConstants.filterH = static_cast<uint32_t>(convYSize);
    pushConstants.filterW = static_cast<uint32_t>(convXSize);
    pushConstants.nnXLen = static_cast<uint32_t>(nnXLen);
    pushConstants.nnYLen = static_cast<uint32_t>(nnYLen);
    vkCmdPushConstants(
      commandBuffer,
      pipelines->conv2dFp32.layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(KatagoVulkan::Conv2DPushConstantParams),
      &pushConstants
    );
    vkCmdBindDescriptorSets(
      commandBuffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipelines->conv2dFp32.layout,
      0, 
      1,
      &descriptorSet,
      0,
      nullptr
    );

    // Compute dispatch dimensions matching HLSL numthreads(TILE_N,TILE_M,1)
    const uint32_t TILE_N = 8u; // local X (numthreads x)
    const uint32_t TILE_M = 8u; // local Y (numthreads y)
    uint32_t dispatchX = (pushConstants.nnXLen + TILE_N - 1u) / TILE_N;
    uint32_t dispatchY = pushConstants.nnYLen;
    uint32_t ocGroupsPerBatch = (pushConstants.outChannels + TILE_M - 1u) / TILE_M;
    uint32_t dispatchZ = pushConstants.batchSize * ocGroupsPerBatch;

    vkCmdDispatch(
      commandBuffer,
      dispatchX,
      dispatchY,
      dispatchZ
    );

    res = VkHelpers::endCommandBuffer(commandBuffer);
  }

  /**
   * @brief return command buffer for conv layer
   * @param batchSize 
   * @param input 
   * @param output 
   * @return VkCommandBuffer 
   */
  VkCommandBuffer apply(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    assert(commandBuffer != VK_NULL_HANDLE);
    return commandBuffer;
  }


};

/**
 * @brief Batch Normalization Layer
 */
struct BatchNormLayer {

  ComputeHandleInternal *handle;

  const std::string name;
  const int numChannels;
  const float epsilon;
  const int activation;

  const int nnXLen;
  const int nnYLen;
  const int nnXYLen;

  VulkanBuffer* mergedScaleBuf;
  VulkanBuffer* mergedBiasBuf;

  static constexpr int nKernelDims = 2;
  size_t globalSizes[nKernelDims];

  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

  ~BatchNormLayer() {
    if ( mergedScaleBuf != nullptr ) {
      VkHelpers::releaseVulkanBuffer(mergedScaleBuf->device, mergedScaleBuf);
      delete mergedScaleBuf;
      mergedScaleBuf = nullptr;
    }
    if ( mergedBiasBuf != nullptr ) {
      VkHelpers::releaseVulkanBuffer(mergedBiasBuf->device, mergedBiasBuf);
      delete mergedBiasBuf;
      mergedBiasBuf = nullptr;
    }
  }

  BatchNormLayer(
    ComputeHandleInternal *handle_,
    const BatchNormLayerDesc* desc,
    const ActivationLayerDesc* actDesc,
    int nnXLen_,
    int nnYLen_
  ): 
    handle(handle_),
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    activation(actDesc->activation),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    nnXYLen(nnXLen_ * nnYLen_)
  {
    assert(desc->scale.size() == static_cast<size_t>(numChannels));
    assert(desc->bias.size() == static_cast<size_t>(numChannels));
    assert(desc->mean.size() == static_cast<size_t>(numChannels));
    assert(desc->variance.size() == static_cast<size_t>(numChannels));
    assert(desc->mergedScale.size() == static_cast<size_t>(numChannels));
    assert(desc->mergedBias.size() == static_cast<size_t>(numChannels));

    // Precompute merged scale and bias
    std::vector<float> mergedScale = desc->mergedScale;
    std::vector<float> mergedBias = desc->mergedBias;

    VkResult res;

    mergedBiasBuf = VkHelpers::createDeviceBufferWithData(
      handle->vulkanDevice,
      sizeof(float) * mergedBias.size(),
      mergedBias.data(),
      true,
      &res
    );
    CHECK_VK_MSG("Create BatchNormLayer: " + name + " merged bias buffer", res);

    mergedScaleBuf = VkHelpers::createDeviceBufferWithData(
      handle->vulkanDevice,
      sizeof(float) * mergedScale.size(),
      mergedScale.data(),
      true,
      &res
    );
    CHECK_VK_MSG("Create BatchNormLayer: " + name + " merged scale buffer", res);

    globalSizes[0] = VkHelpers::powerOf2ify(static_cast<size_t>(nnXYLen));
    globalSizes[1] = VkHelpers::powerOf2ify(static_cast<size_t>(numChannels));
  }

  void record(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* mask
  ) {
    if ( commandBuffer != VK_NULL_HANDLE ) {
      return; // Already recorded
    }
    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    KatagoVulkan::ComputePipelines* pipelines = this->handle->context->pipelinesPerDev.at(gpuId);
    KatagoVulkan::Pipeline targetPipeline;

    switch ( activation ) {
      case ACTIVATION_IDENTITY:
        targetPipeline = pipelines->batchNormMaskFp32;
        break;
      case ACTIVATION_RELU:
        targetPipeline = pipelines->batchNormMaskReluFp32;
        break;
      case ACTIVATION_MISH:
        targetPipeline = pipelines->batchNormMaskMishFp32;
        break;
      default:
        Global::fatalError("Unsupported activation in BatchNormLayer: " + name);
    }

    commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
    VkResult res = VkHelpers::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for BatchNormLayer: " + name, res);
    res = VkHelpers::endCommandBuffer(commandBuffer);

    descriptorSet = VkHelpers::allocateDescriptorSet(
      handle->vulkanDevice,
      targetPipeline.descriptorSetLayout,
      &res
    );
    CHECK_VK_MSG("Allocate descriptor set for BatchNormLayer: " + name, res);

    // update descriptor set
    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, input),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, output),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, mergedScaleBuf),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 3, mergedBiasBuf),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 4, mask)
    };
    VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

    KatagoVulkan::BatchNormMaskFp32Params pushConstants = {};
    pushConstants.batchSize = static_cast<uint32_t>(batchSize);
    pushConstants.numChannels = static_cast<uint32_t>(numChannels);
    pushConstants.nnXYLen = static_cast<uint32_t>(nnXYLen);
    vkCmdPushConstants(commandBuffer, targetPipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KatagoVulkan::BatchNormMaskFp32Params), &pushConstants);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.layout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(
      commandBuffer,
      static_cast<uint32_t>(globalSizes[0]),
      static_cast<uint32_t>(globalSizes[1]),
      1
    );

    CHECK_VK_MSG("End command buffer for BatchNormLayer: " + name, res);
  }

  VkCommandBuffer apply(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* mask
  ) {
    assert(commandBuffer != VK_NULL_HANDLE);
    return commandBuffer;
  }
};

/**
 * @brief Matrix Bias Layer
 * Add bias per channel after matmul layer
 * Also apply activation function if needed
 * Currently support identity, relu, mish activations
 * @param handle_ : ComputeHandleInternal pointer
 * @param desc : MatBiasLayerDesc pointer
 * @param activation_ : activation type
 */
struct MatBiasLayer {
  ComputeHandleInternal *handle;
  const std::string name;
  const int numChannels;
  const int activation;

  VulkanBuffer *biasBuf;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

  static constexpr int nKernelDims = 2;

  ~MatBiasLayer() {
    if ( biasBuf != nullptr ) {
      VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, biasBuf);
      delete biasBuf;
      biasBuf = nullptr;
    }
  }

  MatBiasLayer(
    ComputeHandleInternal *handle_,
    const MatBiasLayerDesc* desc,
    int activation_
  ): 
    handle(handle_),
    name(desc->name),
    numChannels(desc->numChannels),
    activation(activation_)
  {
    if ( numChannels > 0 ) {
      assert(desc->weights.size() == static_cast<size_t>(numChannels));
      std::vector<float> weights = desc->weights;
      VkResult res;
      biasBuf = VkHelpers::createDeviceBufferWithData(
        handle->vulkanDevice,
        sizeof(float) * weights.size(),
        weights.data(),
        true,
        &res
      );
      CHECK_VK_MSG("Create MatBiasLayer: " + name + " bias buffer", res);
    } else {
      biasBuf = nullptr;
    }
  }

  void record(int batchSize, VulkanBuffer* input) {
    if ( commandBuffer != VK_NULL_HANDLE ) {
      return; // Already recorded
    }
    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    KatagoVulkan::ComputePipelines* pipelines = this->handle->context->pipelinesPerDev.at(gpuId);
    KatagoVulkan::Pipeline targetPipeline;

    switch ( activation ) {
      case ACTIVATION_IDENTITY:
        targetPipeline = pipelines->addChannelBiasNCHWFp32;
        break;
      case ACTIVATION_RELU:
        targetPipeline = pipelines->addChannelBiasNCHWReluFp32;
        break;
      case ACTIVATION_MISH:
        targetPipeline = pipelines->addChannelBiasNCHWMishFp32;
        break;
      default:
        Global::fatalError("Unsupported activation in MatBiasLayer: " + name);
    }
    VkResult res;
    commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
    descriptorSet = VkHelpers::allocateDescriptorSet(
      handle->vulkanDevice,
      targetPipeline.descriptorSetLayout,
      &res
    );
    CHECK_VK_MSG("Allocate descriptor set for MatBiasLayer: " + name, res);
    VkHelpers::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for MatBiasLayer: " + name, res);
    auto pushConstants = KatagoVulkan::MatBiasFp32Params();
    pushConstants.numChannels = static_cast<uint32_t>(numChannels);
    pushConstants.batchSize = static_cast<uint32_t>(batchSize);

    // update descriptor set
    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, input),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, biasBuf),
    };

    VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.pipeline);

    vkCmdPushConstants(
      commandBuffer,
      targetPipeline.layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(KatagoVulkan::MatBiasFp32Params),
      &pushConstants
    );

    vkCmdBindDescriptorSets(
      commandBuffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      targetPipeline.layout,
      0, 
      1,
      &descriptorSet,
      0,
      nullptr
    );

    uint32_t globalSizes[2] = {
      static_cast<uint32_t>(VkHelpers::powerOf2ify(static_cast<size_t>(numChannels))),
      static_cast<uint32_t>(VkHelpers::powerOf2ify(static_cast<size_t>(batchSize)))
    };
    vkCmdDispatch(commandBuffer, globalSizes[0], globalSizes[1], 1);
    res = VkHelpers::endCommandBuffer(commandBuffer);
    CHECK_VK_MSG("End command buffer for MatBiasLayer: " + name, res);
  }

  VkCommandBuffer apply(int batchSize, VulkanBuffer* input) {
    assert(commandBuffer != VK_NULL_HANDLE);
    return commandBuffer;
  }

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;
};

/**
 * @brief Convolution Layer with BatchNorm and Activation fused
 * Currently no fused gpu kernel.
 * @param handle_ : ComputeHandleInternal pointer
 * @param convDesc : ConvLayerDesc pointer
 * @param bnDesc : BatchNormLayerDesc pointer
 * @param actDesc : ActivationLayerDesc pointer
 * @param nnXLen_ : neural net x length
 * @param nnYLen_ : neural net y length
 * TODO: Implement fused conv + bn + act kernel later for better performance.
 */
struct NormActConv {
  ComputeHandleInternal* handle;
  ConvLayer conv;
  BatchNormLayer bn;
  const int inChannels;
  const int outChannels;

  std::vector<VkCommandBuffer> commandBuffers;

  NormActConv(
    ComputeHandleInternal *handle_,
    const ConvLayerDesc* convDesc,
    const BatchNormLayerDesc* bnDesc,
    const ActivationLayerDesc* actDesc,
    int nnXLen_,
    int nnYLen_
  ): 
    handle(handle_),
    conv(handle_, convDesc, nnXLen_, nnYLen_),
    bn(handle_, bnDesc, actDesc, nnXLen_, nnYLen_),
    inChannels(convDesc->inChannels),
    outChannels(convDesc->outChannels)
  {
    assert( bn.numChannels == conv.inChannels );
  }

  /**
   * @brief record command buffers for conv and bn layers, no winograd algorithm, so convworkspace not required now.
   * @param batchSize
   * @param input
   * @param inputScratchOrInput
   * @param output
   * @param mask
   */
  void record(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* inputScratchOrInput, //It's okay if this is the same as input, if it's okay to mutate input.
    VulkanBuffer* output,
    VulkanBuffer* mask
    // VulkanBuffer* convWorkspace,
    // VulkanBuffer* convWorkspace2
  ) {
    // bn.record(batchSize, input, inputScratchOrInput, mask);
    // conv.record(batchSize, inputScratchOrInput, output, convWorkspace, convWorkspace2);
    // commandBuffers.push_back( conv.apply(batchSize, input, output) );
    // commandBuffers.push_back( bn.apply(batchSize, output, output, mask) );
    bn.record(batchSize, input, inputScratchOrInput, mask);
    conv.record(batchSize, inputScratchOrInput, output);
  }

  /**
   * @brief return command buffers for conv and bn layers
   * @param batchSize
   * @param input
   * @param inputScratchOrInput
   * @param output
   */
  std::vector<VkCommandBuffer> apply(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* inputScratchOrInput,
    VulkanBuffer* output,
    VulkanBuffer* mask
    // VulkanBuffer* convWorkspace,
    // VulkanBuffer* convWorkspace2
  ) {
    std::vector<VkCommandBuffer> cmds;
    cmds.push_back( conv.apply(batchSize, input, output) );
    cmds.push_back( bn.apply(batchSize, output, output, mask) );
    return cmds;
  }

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;
};

VkCommandBuffer performAddChannelBiases(
  ComputeHandleInternal *handle,
  VulkanBuffer* input,
  VulkanBuffer* bias,
  int ncSize,
  int nnXYLen
) {
  static constexpr int nKernelDims = 2;
  size_t globalSizes[nKernelDims] = {
    VkHelpers::powerOf2ify(static_cast<size_t>(nnXYLen)),
    VkHelpers::powerOf2ify(static_cast<size_t>(ncSize))
  };
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  KatagoVulkan::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  KatagoVulkan::Pipeline targetPipeline = pipelines->addChannelBiasNCHWFp32;

  VkCommandBuffer commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
  VkResult res = VkHelpers::beginCommandBuffer(commandBuffer);
  CHECK_VK_MSG("Begin command buffer for AddChannelBiases", res);
  VkDescriptorSet descriptorSet = VkHelpers::allocateDescriptorSet(
    handle->vulkanDevice,
    targetPipeline.descriptorSetLayout,
    &res
  );
  CHECK_VK_MSG("Allocate descriptor set for AddChannelBiases", res);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.pipeline);
  vkCmdBindDescriptorSets(
    commandBuffer,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    targetPipeline.layout,
    0, 
    1,
    &descriptorSet,
    0,
    nullptr
  );
  // update descriptor set
  std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, input),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, bias)
  };
  VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
  KatagoVulkan::MatBiasFp32Params pushConstants = {};
  pushConstants.numChannels = static_cast<uint32_t>(ncSize);
  pushConstants.batchSize = static_cast<uint32_t>(nnXYLen);
  vkCmdPushConstants(
    commandBuffer,
    targetPipeline.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(KatagoVulkan::MatBiasFp32Params),
    &pushConstants
  );
  vkCmdDispatch(
    commandBuffer,
    static_cast<uint32_t>(globalSizes[0]),
    static_cast<uint32_t>(globalSizes[1]),
    1
  );
  return commandBuffer;
}

VkCommandBuffer performAddPointWise(
  ComputeHandleInternal *handle,
  VulkanBuffer* input,
  VulkanBuffer* value,
  int totalSize
) {
  static constexpr int nKernelDims = 1;
  size_t globalSizes[nKernelDims] = {
    VkHelpers::powerOf2ify(static_cast<size_t>(totalSize))
  };
  VkCommandBuffer commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
  VkResult res = VkHelpers::beginCommandBuffer(commandBuffer);
  CHECK_VK_MSG("Begin command buffer for AddPointWise", res);
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  KatagoVulkan::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  VkDescriptorSet descriptorSet = VkHelpers::allocateDescriptorSet(
    handle->vulkanDevice,
    pipelines->addPointWiseFp32.descriptorSetLayout,
    &res
  );
  CHECK_VK_MSG("Allocate descriptor set for AddPointWise", res);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines->addPointWiseFp32.pipeline);
  vkCmdBindDescriptorSets(
    commandBuffer,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    pipelines->addPointWiseFp32.layout,
    0, 
    1,
    &descriptorSet,
    0,
    nullptr
  );
  // update descriptor set
  std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, input),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, value)
  };
  VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
  KatagoVulkan::AddPointWiseParams pushConstants = {};
  pushConstants.totalSize = static_cast<uint32_t>(totalSize);
  vkCmdPushConstants(
    commandBuffer,
    pipelines->addPointWiseFp32.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(KatagoVulkan::AddPointWiseParams),
    &pushConstants
  );
  uint32_t groupCountX = (static_cast<uint32_t>(totalSize) + 255) / 256;
  vkCmdDispatch(commandBuffer, groupCountX, 1, 1);
  return commandBuffer;
}

VkCommandBuffer performGpoolMask(
  ComputeHandleInternal *handle,
  VulkanBuffer* gpoolConvOut,
  VulkanBuffer* gpoolConcat,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  int batchSize,
  int gpoolChannels,
  int nnXYLen,
  VkResult* result
) {
  static constexpr int nKernelDims = 3;
  static constexpr int localSizes[nKernelDims] = {1, 1, 1};
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  KatagoVulkan::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  VkCommandBuffer commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
  VkResult res = VkHelpers::beginCommandBuffer(commandBuffer);
  CHECK_VK_MSG("Begin command buffer for GlobalPoolingMask", res);
  VkDescriptorSet descriptorSet = VkHelpers::allocateDescriptorSet(
    handle->vulkanDevice,
    pipelines->globalPoolingChannelsFp32.descriptorSetLayout,
    &res
  );
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines->globalPoolingChannelsFp32.pipeline);
  vkCmdBindDescriptorSets(
    commandBuffer,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    pipelines->globalPoolingChannelsFp32.layout,
    0, 
    1,
    &descriptorSet,
    0,
    nullptr
  );
  // update descriptor set
  std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, gpoolConvOut),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, gpoolConcat),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, mask),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 3, maskSum)
  };

  VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

  KatagoVulkan::GlobalPoolingChannelsParams pushConstants = {};
  pushConstants.batchSize = static_cast<uint32_t>(batchSize);
  pushConstants.gpoolChannels = static_cast<uint32_t>(gpoolChannels);
  pushConstants.nnXYLen = static_cast<uint32_t>(nnXYLen);
  vkCmdPushConstants(
    commandBuffer,
    pipelines->globalPoolingChannelsFp32.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(KatagoVulkan::GlobalPoolingChannelsParams),
    &pushConstants
  );

  // Assume local sizes = 1,1,1 (as in OpenCL defaults used elsewhere).
  // Then groupCountX = 1 (XYSTRIDE), groupCountY = ceil(gpoolChannels / 1) = gpoolChannels,
  // and groupCountZ = ceil(batchSize / 1) = batchSize.
  uint32_t groupCountX = 1u;
  uint32_t groupCountY = static_cast<uint32_t>(gpoolChannels);
  uint32_t groupCountZ = static_cast<uint32_t>(batchSize);
  vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);

  CHECK_VK_MSG("Allocate descriptor set for GlobalPoolingMask", res);
  return commandBuffer;
}

VkCommandBuffer performValueHeadPool(
  ComputeHandleInternal *handle,
  VulkanBuffer* gpoolConvOut,
  VulkanBuffer* gpoolConcat,
  VulkanBuffer* maskSum,
  int batchSize,
  int valueHeadChannels,
  int nnXYLen
) {
  static constexpr int nKernelDims = 3;
  static constexpr int localSizes[nKernelDims] = {1, 1, 1};
  VkCommandBuffer commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
  VkResult res = VkHelpers::beginCommandBuffer(commandBuffer);
  CHECK_VK_MSG("Begin command buffer for ValueHeadPool", res);
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  KatagoVulkan::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  VkDescriptorSet descriptorSet = VkHelpers::allocateDescriptorSet(
    handle->vulkanDevice,
    pipelines->valueHeadPoolingChannelsFp32.descriptorSetLayout,
    &res
  );
  CHECK_VK_MSG("Allocate descriptor set for ValueHeadPool", res);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines->valueHeadPoolingChannelsFp32.pipeline);
  vkCmdBindDescriptorSets(
    commandBuffer,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    pipelines->valueHeadPoolingChannelsFp32.layout,
    0, 
    1,
    &descriptorSet,
    0,
    nullptr
  );

  // update descriptor set
  std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, gpoolConvOut),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, gpoolConcat),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, maskSum)
  };
  VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
  KatagoVulkan::GlobalPoolingChannelsParams pushConstants = {};
  pushConstants.batchSize = static_cast<uint32_t>(batchSize);
  pushConstants.gpoolChannels = static_cast<uint32_t>(valueHeadChannels);
  pushConstants.nnXYLen = static_cast<uint32_t>(nnXYLen);
  vkCmdPushConstants(
    commandBuffer,
    pipelines->valueHeadPoolingChannelsFp32.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(KatagoVulkan::GlobalPoolingChannelsParams),
    &pushConstants
  );
  // Assume local sizes = 1,1,1. Compute group counts accordingly.
  uint32_t groupCountX = 1u;
  uint32_t groupCountY = static_cast<uint32_t>(valueHeadChannels);
  uint32_t groupCountZ = static_cast<uint32_t>(batchSize);
  vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);

  return commandBuffer;
}

/**
 * @brief Basic Residual Block, Consist of two conv layers with BN and Activation and one skip connection
 */
struct ResidualBlock {
  ComputeHandleInternal *handle;
  const std::string name;
  const int nnXLen;
  const int nnYLen;
  NormActConv *normActConv;
  NormActConv *normActConv2;
  std::vector<VkCommandBuffer> commandBuffers;

  ResidualBlock(
    ComputeHandleInternal *handle_,
    const ResidualBlockDesc* desc,
    int nnXLen_,
    int nnYLen_
  ): 
    handle(handle_),
    name(desc->name),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_)
  {
    normActConv = new NormActConv(
      handle,
      &desc->regularConv,
      &desc->preBN,
      &desc->preActivation,
      nnXLen,
      nnYLen
    );
    normActConv2 = new NormActConv(
      handle,
      &desc->finalConv,
      &desc->midBN,
      &desc->midActivation,
      nnXLen,
      nnYLen
    );
  }

  ~ResidualBlock() {
    delete normActConv;
    delete normActConv2;
  }

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  void record(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask
    // VulkanBuffer* convWorkspace,
    // VulkanBuffer* convWorkspace2
  ) {

    if ( !commandBuffers.empty() ) {
      Global::fatalError("ResidualBlock: " + name + " record called multiple times");
    }

    SizedBuf<VulkanBuffer*> mid(
      scratch->allocator,
      scratch->getBufSizeXY(normActConv->outChannels)
    );
    normActConv->record(batchSize, trunk, trunkScratch, mid.buf , mask);
    normActConv2->record(batchSize, mid.buf, mid.buf, trunkScratch, mask);
    VkCommandBuffer pointWiseCB = performAddPointWise(handle, trunk, trunkScratch, static_cast<int>(batchSize * normActConv2->outChannels * nnXLen * nnYLen));
    commandBuffers.insert(commandBuffers.end(), normActConv->commandBuffers.begin(), normActConv->commandBuffers.end());
    commandBuffers.insert(commandBuffers.end(), normActConv2->commandBuffers.begin(), normActConv2->commandBuffers.end());
    commandBuffers.push_back( pointWiseCB );
  }

  std::vector<VkCommandBuffer> apply(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask
  ) {
    if ( commandBuffers.empty() ) {
      Global::fatalError("ResidualBlock: " + name + " apply called before record");
    }
    return commandBuffers;
  }
};

struct GlobalPoolingResidualBlock {
  ComputeHandleInternal *handle;
  BatchNormLayer* preBN;
  ConvLayer *regularConv;
  ConvLayer *gpoolConv;
  BatchNormLayer* gpoolBN;
  MatmulLayer* gpoolToBiasMul;
  NormActConv* normActConv2;

  const int nnXLen;
  const int nnYLen;
  const int nnXYLen;
  const int regularChannels;
  const int gpoolChannels;

  std::vector<VkCommandBuffer> commandBuffers;

  GlobalPoolingResidualBlock(
    ComputeHandleInternal *handle_,
    const GlobalPoolingResidualBlockDesc* desc,
    int nnXLen_,
    int nnYLen_
  ): 
    handle(handle_),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    nnXYLen(nnXLen_ * nnYLen_),
    regularChannels(desc->regularConv.outChannels),
    gpoolChannels(desc->gpoolConv.outChannels)
  {
    preBN = new BatchNormLayer(
      handle,
      &desc->preBN,
      &desc->preActivation,
      nnXLen,
      nnYLen
    );
    regularConv = new ConvLayer(
      handle,
      &desc->regularConv,
      nnXLen,
      nnYLen
    );
    gpoolConv = new ConvLayer(
      handle,
      &desc->gpoolConv,
      nnXLen,
      nnYLen
    );
    gpoolBN = new BatchNormLayer(
      handle,
      &desc->gpoolBN,
      &desc->gpoolActivation,
      nnXLen,
      nnYLen
    );
    gpoolToBiasMul = new MatmulLayer(
      handle,
      &desc->gpoolToBiasMul
    );
    normActConv2 = new NormActConv(
      handle,
      &desc->finalConv,
      &desc->midBN,
      &desc->midActivation,
      nnXLen,
      nnYLen
    );
  }

  ~GlobalPoolingResidualBlock() {
    delete preBN;
    delete regularConv;
    delete gpoolConv;
    delete gpoolBN;
    delete gpoolToBiasMul;
    delete normActConv2;
  };

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  void record(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum
    // VulkanBuffer* convWorkspace,
    // VulkanBuffer* convWorkspace2,
  ) {
    SizedBuf<VulkanBuffer*> regularOut(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(gpoolChannels * 3));
    SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(regularChannels));

    if ( !commandBuffers.empty() ) {
      Global::fatalError("GlobalPoolingResidualBlock record called multiple times");
    }

    preBN->record(batchSize, trunk, trunkScratch, mask);
    regularConv->record(batchSize, trunkScratch, regularOut.buf);
    gpoolConv->record(batchSize, trunkScratch, gpoolOut.buf);
    gpoolBN->record(batchSize, gpoolOut.buf, gpoolOut.buf, mask);
    VkResult res;;
    VkCommandBuffer gpoolCB = performGpoolMask(handle, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, batchSize, gpoolChannels, nnXYLen, &res);
    CHECK_VK_MSG("Record GlobalPoolingResidualBlock gpool mask", res);
    gpoolToBiasMul->record(batchSize, gpoolConcat.buf, gpoolBias.buf);
    VkCommandBuffer addChannelCB = performAddChannelBiases(handle, regularOut.buf, gpoolBias.buf, regularChannels, nnXYLen);
    normActConv2->record(batchSize, regularOut.buf, regularOut.buf, trunkScratch, mask);
    VkCommandBuffer addPointWiseCB = performAddPointWise(handle, trunk, trunkScratch, batchSize * normActConv2->outChannels * nnXLen * nnYLen);

    commandBuffers.push_back( preBN->commandBuffer );
    commandBuffers.push_back( regularConv->commandBuffer );
    commandBuffers.push_back( gpoolConv->commandBuffer );
    commandBuffers.push_back( gpoolBN->commandBuffer );
    commandBuffers.push_back( gpoolCB );
    commandBuffers.push_back( gpoolToBiasMul->commandBuffer );
    commandBuffers.push_back( addChannelCB );
    commandBuffers.insert(commandBuffers.end(), normActConv2->commandBuffers.begin(), normActConv2->commandBuffers.end());
    commandBuffers.push_back( addPointWiseCB );
  }

  std::vector<VkCommandBuffer> apply(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum
  ) {
    if ( commandBuffers.empty() ) {
      Global::fatalError("GlobalPoolingResidualBlock apply called before record");
    }
    return commandBuffers;
  }
};

BlockStack::BlockStack(
  ComputeHandleInternal *handle_,
  const std::vector<std::pair<int, unique_ptr_void>> &descBlocks,
  int numBlocks_,
  int trunkNumChannels_,
  int nnXLen_,
  int nnYLen_
): 
  handle(handle_),
  numBlocks(numBlocks_),
  trunkNumChannels(trunkNumChannels_),
  nnXLen(nnXLen_),
  nnYLen(nnYLen_)
{
  assert(descBlocks.empty() == false);

  for ( int i = 0 ; i < numBlocks ; ++i ) {
    int blockType = descBlocks[i].first;

    if ( blockType == ORDINARY_BLOCK_KIND ) {
      const ResidualBlockDesc* resDesc = static_cast<const ResidualBlockDesc*>(descBlocks[i].second.get());
      unique_ptr_void blockPtr = make_unique_void( 
        new ResidualBlock(
          handle,
          resDesc,
          nnXLen,
          nnYLen
        )
      );
      blocks.push_back(std::make_pair(blockType, std::move(blockPtr)));
    } else if ( blockType == GLOBAL_POOLING_BLOCK_KIND ) {
      const GlobalPoolingResidualBlockDesc* gpoolDesc = static_cast<const GlobalPoolingResidualBlockDesc*>(descBlocks[i].second.get());
      unique_ptr_void blockPtr = make_unique_void( 
        new GlobalPoolingResidualBlock(
          handle,
          gpoolDesc,
          nnXLen,
          nnYLen
        )
      );
      blocks.push_back(std::make_pair(blockType, std::move(blockPtr)));
    } else if ( blockType == NESTED_BOTTLENECK_BLOCK_KIND ) {
      const NestedBottleneckResidualBlockDesc* nestedDesc = static_cast<const NestedBottleneckResidualBlockDesc*>(descBlocks[i].second.get());
      unique_ptr_void blockPtr = make_unique_void( 
        new NestedResidualBlock(
          handle,
          nestedDesc,
          nnXLen,
          nnYLen
        )
      );
      blocks.push_back(std::make_pair(blockType, std::move(blockPtr)));
    } else {
      Global::fatalError("Unsupported block type in BlockStack");
    }
  }
}

struct NestedResidualBlock {
  ComputeHandleInternal *handle;
  const std::string name;
  NormActConv *normActConv;
  BlockStack *blocks;
  NormActConv *normActConv2;
  const int nnXLen;
  const int nnYLen;
  std::vector<VkCommandBuffer> commandBuffers;

  NestedResidualBlock(
    ComputeHandleInternal *handle_,
    const NestedBottleneckResidualBlockDesc* desc,
    int nnXLen_,
    int nnYLen_
  ): 
    handle(handle_),
    name(desc->name),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_)
  {
    normActConv = new NormActConv(
      handle,
      &desc->preConv,
      &desc->preBN,
      &desc->preActivation,
      nnXLen,
      nnYLen
    );

    blocks = new BlockStack(
      handle,
      desc->blocks,
      desc->numBlocks,
      desc->preConv.outChannels,
      nnXLen,
      nnYLen
    );
    normActConv2 = new NormActConv(
      handle,
      &desc->postConv,
      &desc->postBN,
      &desc->postActivation,
      nnXLen,
      nnYLen
    );
  }

  ~NestedResidualBlock() {
    delete normActConv2;
    delete blocks;
    delete normActConv;
  }

  NestedResidualBlock() = delete;
  NestedResidualBlock(const NestedResidualBlock&) = delete;
  NestedResidualBlock& operator=(const NestedResidualBlock&) = delete;

  void record(
    int batchSize, 
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum
  ) {
    SizedBuf<VulkanBuffer*> mid1(scratch->allocator, scratch->getBufSizeXY(normActConv->outChannels));
    SizedBuf<VulkanBuffer*> mid2(scratch->allocator, scratch->getBufSizeXY(normActConv->outChannels));

    if ( !commandBuffers.empty() ) {
      Global::fatalError("NestedResidualBlock: " + name + " record called multiple times");
    }
    normActConv->record(batchSize, trunk, trunkScratch, mid1.buf , mask);
    blocks->record(batchSize, scratch, mid1.buf, mid2.buf, mask, maskSum);
    normActConv2->record(batchSize, mid2.buf, mid2.buf, trunkScratch, mask);
    VkCommandBuffer pointWiseCB = performAddPointWise(handle, trunk, trunkScratch, static_cast<int>(batchSize * normActConv2->outChannels * nnXLen * nnYLen));
    commandBuffers.push_back( normActConv->commandBuffers[0] );
    commandBuffers.insert(commandBuffers.end(), blocks->commandBuffers.begin(), blocks->commandBuffers.end());
    commandBuffers.insert(commandBuffers.end(), normActConv2->commandBuffers.begin(), normActConv2->commandBuffers.end());
    commandBuffers.push_back( pointWiseCB );
  }

  void apply(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum
  ) {
    // TODO: changethis method to launch command buffers
    //       only for debug
    if ( commandBuffers.empty() ) {
      Global::fatalError("NestedResidualBlock: " + name + " apply called before record");
    }
  }
};

struct SGFMetadataEncoder {
  const std::string name;
  MatmulLayer *matmul1;
  MatBiasLayer *matBias1;
  MatmulLayer *matmul2;
  MatBiasLayer *matBias2;
  MatmulLayer *matmul3;

  std::vector<VkCommandBuffer> commandBuffers;

  SGFMetadataEncoder(
    ComputeHandleInternal *handle,
    const SGFMetadataEncoderDesc* desc
  ): 
    name(desc->name)
  {
    matmul1 = new MatmulLayer(
      handle,
      &desc->mul1
    );
    matBias1 = new MatBiasLayer(
      handle,
      &desc->bias1,
      desc->act1.activation
    );
    matmul2 = new MatmulLayer(
      handle,
      &desc->mul2
    );
    matBias2 = new MatBiasLayer(
      handle,
      &desc->bias2,
      desc->act2.activation
    );
    matmul3 = new MatmulLayer(
      handle,
      &desc->mul3
    );
  }

  ~SGFMetadataEncoder() {
    delete matmul1;
    delete matBias1;
    delete matmul2;
    delete matBias2;
    delete matmul3;
  }

  SGFMetadataEncoder() = delete;
  SGFMetadataEncoder(const SGFMetadataEncoder&) = delete;
  SGFMetadataEncoder& operator=(const SGFMetadataEncoder&) = delete;

  /**
   * @brief record SGFMetadataEncoder
   */
  void record(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    SizedBuf<VulkanBuffer*> internalBuf1(scratch->allocator, scratch->getBufSizeFloat(std::max(matmul1->outChannels, matmul2->outChannels)));
    SizedBuf<VulkanBuffer*> internalBuf2(scratch->allocator, scratch->getBufSizeFloat(std::max(matmul1->outChannels, matmul2->outChannels)));

    matmul1->record(batchSize, input, internalBuf1.buf);
    matBias1->record(batchSize, internalBuf1.buf);
    matmul2->record(batchSize, internalBuf1.buf, internalBuf2.buf);
    matBias2->record(batchSize, internalBuf2.buf);
    matmul3->record(batchSize, internalBuf2.buf, output);

    commandBuffers.push_back( matmul1->commandBuffer );
    commandBuffers.push_back( matBias1->commandBuffer );
    commandBuffers.push_back( matmul2->commandBuffer );
    commandBuffers.push_back( matBias2->commandBuffer );
    commandBuffers.push_back( matmul3->commandBuffer );
  }

  /**
   * @brief execute SGFMetadataEncoder, only for debug now
   */
  void apply(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
  }
};

struct Trunk {
  ComputeHandleInternal *handle;
  const std::string name;
  const int modelVersion;
  const int trunkNumChannels;
  const int midNumChannels;
  const int regularNumChannels;
  const int gpoolNumChannels;

  const int nnXLen;
  const int nnYLen;

  std::unique_ptr<ConvLayer> initialConv;
  std::unique_ptr<MatmulLayer> initialMatmul;
  std::unique_ptr<SGFMetadataEncoder> sgfMetadataEncoder;
  BlockStack blockStack;
  std::unique_ptr<BatchNormLayer> trunkTipBN;
  std::vector<VkCommandBuffer> commandBuffers;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(
    ComputeHandleInternal *handle_,
    const TrunkDesc* desc,
    int maxBatchSize_,
    int nnXLen_,
    int nnYLen_
  ): 
    handle(handle_),
    name(desc->name),
    modelVersion(desc->modelVersion),
    trunkNumChannels(desc->trunkNumChannels),
    midNumChannels(desc->midNumChannels),
    regularNumChannels(desc->regularNumChannels),
    gpoolNumChannels(desc->gpoolNumChannels),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    blockStack( handle, desc->blocks, desc->numBlocks, trunkNumChannels, nnXLen_, nnYLen_) {
  }

  ~Trunk() {

  }

  void record(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* input,
    VulkanBuffer* inputGlobal,
    VulkanBuffer* inputMeta,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum
  ) { 
    if ( !commandBuffers.empty() ) {
      return;
    }

    SizedBuf<VulkanBuffer*> trunkScratch( scratch->allocator, scratch->getBufSizeXY(trunkNumChannels) );

    initialConv->record(batchSize, input, trunk);
    initialMatmul->record(batchSize, inputGlobal, trunkScratch.buf);
    VkCommandBuffer addChannelBiasCB = performAddChannelBiases(handle, trunk, trunkScratch.buf, trunkNumChannels, nnXLen * nnYLen);
    VkCommandBuffer addChannelBiasCB2 = VK_NULL_HANDLE;
    if ( sgfMetadataEncoder != nullptr ) {
      SizedBuf<VulkanBuffer*> sgfEncodedMeta(scratch->allocator, scratch->getBufSizeFloat(sgfMetadataEncoder->matmul3->outChannels));
      sgfMetadataEncoder->record(batchSize, scratch, inputMeta, sgfEncodedMeta.buf);
      addChannelBiasCB2 = performAddChannelBiases(handle, trunk, sgfEncodedMeta.buf, trunkNumChannels, nnXLen * nnYLen);
    }
    blockStack.record(batchSize, scratch, trunk, trunkScratch.buf, mask, maskSum);
    trunkTipBN->record(batchSize, trunk, trunk, mask);

    commandBuffers.push_back( initialConv->commandBuffer );
    commandBuffers.push_back( initialMatmul->commandBuffer );
    commandBuffers.push_back( addChannelBiasCB );
    if ( sgfMetadataEncoder != nullptr ) {
      commandBuffers.insert(commandBuffers.end(), sgfMetadataEncoder->commandBuffers.begin(), sgfMetadataEncoder->commandBuffers.end());
      commandBuffers.push_back( addChannelBiasCB2 );
    }
    commandBuffers.insert(commandBuffers.end(), blockStack.commandBuffers.begin(), blockStack.commandBuffers.end());
    commandBuffers.push_back( trunkTipBN->commandBuffer );
  }

  /**
   * @brief Execute inference on the trunk. Only for debug now
   */
  void apply(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* input,
    VulkanBuffer* inputGlobal,
    VulkanBuffer* inputMeta,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum
  ) {
  }
};

struct PolicyHead {
  ComputeHandleInternal *handle;
  const std::string name;
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int p1Channels;
  const int g1Channels;
  const int p2Channels;

  std::unique_ptr<ConvLayer> p1Conv;
  std::unique_ptr<ConvLayer> g1Conv;
  std::unique_ptr<BatchNormLayer> g1BN;
  std::unique_ptr<MatmulLayer> gpoolToBiasMul;
  std::unique_ptr<BatchNormLayer> p1BN;
  std::unique_ptr<ConvLayer> p2Conv;
  std::unique_ptr<MatmulLayer> gpoolToPassMul;
  std::unique_ptr<MatBiasLayer> gpoolToPassBias;
  std::unique_ptr<MatmulLayer> gpoolToPassMul2;

  std::vector<VkCommandBuffer> commandBuffers;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(
    ComputeHandleInternal *handle,
    const PolicyHeadDesc* desc,
    int nnXLen_,
    int nnYLen_
  ): 
    handle(handle),
    name(desc->name),
    modelVersion(desc->modelVersion),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    p1Channels(desc->p1Conv.outChannels),
    g1Channels(desc->g1Conv.outChannels),
    p2Channels(desc->p2Conv.outChannels)
  {
    p1Conv = std::make_unique<ConvLayer>(handle, &desc->p1Conv, nnXLen, nnYLen);
    g1Conv = std::make_unique<ConvLayer>(handle, &desc->g1Conv, nnXLen, nnYLen);
    g1BN = std::make_unique<BatchNormLayer>(handle, &desc->g1BN, &desc->g1Activation, nnXLen, nnYLen);
    gpoolToBiasMul = std::make_unique<MatmulLayer>(handle, &desc->gpoolToBiasMul);
    p1BN = std::make_unique<BatchNormLayer>(handle, &desc->p1BN, &desc->p1Activation, nnXLen, nnYLen);
    p2Conv = std::make_unique<ConvLayer>(handle, &desc->p2Conv, nnXLen, nnYLen);
    gpoolToPassMul = std::make_unique<MatmulLayer>(handle, &desc->gpoolToPassMul);
    gpoolToPassBias = std::make_unique<MatBiasLayer>(handle, &desc->gpoolToPassBias, desc->passActivation.activation);
    gpoolToPassMul2 = std::make_unique<MatmulLayer>(handle, &desc->gpoolToPassMul2);
  }

  ~PolicyHead() {

  }

  void record(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* policyPass,
    VulkanBuffer* policy
  ) {
    if (  !commandBuffers.empty() ) {
      return;
    }

    SizedBuf<VulkanBuffer*> p1ConvOut(scratch->allocator, scratch->getBufSizeXY(p1Channels));
    SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(g1Channels * 3));
    SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(p1Channels));
    SizedBuf<VulkanBuffer*> p1Pass(scratch->allocator, scratch->getBufSizeFloat(p1Channels));

    p1Conv->record(batchSize, trunk, p1ConvOut.buf);
    g1Conv->record(batchSize, trunk, gpoolOut.buf);
    g1BN->record(batchSize, gpoolOut.buf, gpoolOut.buf, mask);
    VkResult res;;
    VkCommandBuffer gpoolCB = performGpoolMask(handle,gpoolOut.buf, gpoolConcat.buf, mask, maskSum, batchSize, g1Channels, nnXLen * nnYLen, &res);
    CHECK_VK_MSG("Record PolicyHead gpool mask", res);
    gpoolToBiasMul->record(batchSize, gpoolConcat.buf, gpoolBias.buf);
    VkCommandBuffer adChannelBiasCB = performAddChannelBiases(handle, p1ConvOut.buf, gpoolBias.buf, p1Channels * batchSize, nnXLen * nnYLen);
    p1BN->record(batchSize, p1ConvOut.buf, p1ConvOut.buf, mask);
    p2Conv->record(batchSize, p1ConvOut.buf, p1ConvOut.buf);

    if ( modelVersion >= 15 ) {
      gpoolToPassMul->apply(batchSize, gpoolConcat.buf, p1Pass.buf);
      gpoolToPassBias->apply(batchSize, p1Pass.buf);
      gpoolToPassMul2->apply(batchSize, p1Pass.buf, policyPass);
    } else {
      gpoolToPassMul->record(batchSize, gpoolConcat.buf, policyPass);
    }

    commandBuffers.push_back( p1Conv->commandBuffer );
    commandBuffers.push_back( g1Conv->commandBuffer );
    commandBuffers.push_back( g1BN->commandBuffer );
    commandBuffers.push_back( gpoolCB );
    commandBuffers.push_back( gpoolToBiasMul->commandBuffer );
    commandBuffers.push_back( adChannelBiasCB );
    commandBuffers.push_back( p1BN->commandBuffer );
    commandBuffers.push_back( p2Conv->commandBuffer );
    if ( modelVersion >= 15 ) {
      commandBuffers.push_back( gpoolToPassMul->commandBuffer );
      commandBuffers.push_back( gpoolToPassBias->commandBuffer );
      commandBuffers.push_back( gpoolToPassMul2->commandBuffer );
    } else {
      commandBuffers.push_back( gpoolToPassMul->commandBuffer );
    }
  }

  /*
  * @brief Execute inference on the policy head. Only for debug now
  */
  void apply() {

  }
};

struct ValueHead {
  ComputeHandleInternal *handle;
  const std::string name;
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int v1Channels;
  const int v2Channels;
  const int valueChannels;
  const int scoreValueChannels;
  const int ownershipChannels;

  std::unique_ptr<ConvLayer> v1Conv;
  std::unique_ptr<BatchNormLayer> v1BN;
  std::unique_ptr<MatmulLayer> v2Mul;
  std::unique_ptr<MatBiasLayer> v2Bias;
  std::unique_ptr<MatmulLayer> v3Mul;
  std::unique_ptr<MatBiasLayer> v3Bias;
  std::unique_ptr<MatmulLayer> sv3Mul;
  std::unique_ptr<MatBiasLayer> sv3Bias;
  std::unique_ptr<ConvLayer> vOwnershipConv;

  std::vector<VkCommandBuffer> commandBuffers;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(
    ComputeHandleInternal *handle_,
    const ValueHeadDesc* desc,
    int nnXLen_,
    int nnYLen_
  ): 
    handle(handle_),
    name(desc->name),
    modelVersion(desc->modelVersion),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    v1Channels(desc->v1Conv.outChannels),
    v2Channels(desc->v2Mul.outChannels),
    valueChannels(desc->v3Mul.outChannels),
    scoreValueChannels(desc->sv3Mul.outChannels),
    ownershipChannels(desc->vOwnershipConv.outChannels)
  {
    v1Conv = std::make_unique<ConvLayer>(handle, &desc->v1Conv, nnXLen, nnYLen);
    v1BN = std::make_unique<BatchNormLayer>(handle, &desc->v1BN, &desc->v1Activation, nnXLen, nnYLen);
    v2Mul = std::make_unique<MatmulLayer>(handle, &desc->v2Mul);
    v2Bias = std::make_unique<MatBiasLayer>(handle, &desc->v2Bias, desc->v2Activation.activation);
    v3Mul = std::make_unique<MatmulLayer>(handle, &desc->v3Mul);
    v3Bias = std::make_unique<MatBiasLayer>(handle, &desc->v3Bias, ACTIVATION_IDENTITY);
    sv3Mul = std::make_unique<MatmulLayer>(handle, &desc->sv3Mul);
    sv3Bias = std::make_unique<MatBiasLayer>(handle, &desc->sv3Bias, ACTIVATION_IDENTITY);
    vOwnershipConv = std::make_unique<ConvLayer>(handle, &desc->vOwnershipConv, nnXLen, nnYLen);
  }

  ~ValueHead() {

  }

  void record(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* value,
    VulkanBuffer* scoreValue,
    VulkanBuffer* ownership
  ) {
    SizedBuf<VulkanBuffer*> v1Out(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<VulkanBuffer*> v1Mean(scratch->allocator, scratch->getBufSizeFloat(v1Channels*3));
    SizedBuf<VulkanBuffer*> v2Out(scratch->allocator, scratch->getBufSizeFloat(valueChannels));

    v1Conv->record(batchSize, trunk, v1Out.buf);
    v1BN->record(batchSize, v1Out.buf, v1Out.buf, mask);
    VkResult res;;
    VkCommandBuffer gpoolCB = performValueHeadPool(handle, v1Out.buf, v1Mean.buf, maskSum, batchSize, v1Channels, nnXLen * nnYLen);

    v2Mul->record(batchSize, v1Mean.buf, v2Out.buf);
    v2Bias->record(batchSize, v2Out.buf);
    v3Mul->record(batchSize, v2Out.buf, value);
    v3Bias->record(batchSize, scoreValue);
    sv3Mul->record(batchSize, v2Out.buf, scoreValue);
    sv3Bias->record(batchSize, scoreValue);
    vOwnershipConv->record(batchSize, v1Out.buf, ownership);

    commandBuffers.push_back( v1Conv->commandBuffer );
    commandBuffers.push_back( v1BN->commandBuffer );
    commandBuffers.push_back( gpoolCB );
    commandBuffers.push_back( v2Mul->commandBuffer );
    commandBuffers.push_back( v2Bias->commandBuffer );
    commandBuffers.push_back( v3Mul->commandBuffer );
    commandBuffers.push_back( v3Bias->commandBuffer );
    commandBuffers.push_back( sv3Mul->commandBuffer );
    commandBuffers.push_back( sv3Bias->commandBuffer );
    commandBuffers.push_back( vOwnershipConv->commandBuffer );
  }

  void apply(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* value,
    VulkanBuffer* scoreValue,
    VulkanBuffer* ownership
  ) {

  }
};

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const std::string& fileName, const std::string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc,expectedSha256);
    modelDesc.applyScale8ToReduceActivations();
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

struct Model {
  std::string modelName;
  int modelVersion;
  int maxBatchSize;
  int nnXLen;
  int nnYLen;
  int numInputChannels;
  int numInputGlobalChannels;
  int numInputMetaChannels;
  int numPolicyChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;

  std::unique_ptr<Trunk> trunk;
  std::unique_ptr<PolicyHead> policyHead;
  std::unique_ptr<ValueHead> valueHead;
  std::vector<VkCommandBuffer> commandBuffers;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(
    ComputeHandleInternal *handle,
    const ModelDesc& desc,
    int maxBatchSize_,
    int nnXLen_,
    int nnYLen_
  ) {
    modelName = desc.name;
    modelVersion = desc.modelVersion;
    maxBatchSize = maxBatchSize_;
    nnXLen = nnXLen_;
    nnYLen = nnYLen_;

    if ( nnXLen > NNPos::MAX_BOARD_LEN ) {
      throw StringError(
        Global::strprintf("Neural net X length %d exceeds maximum supported %d", nnXLen, NNPos::MAX_BOARD_LEN)
      );
    }

    if ( nnYLen > NNPos::MAX_BOARD_LEN ) {
      throw StringError(
        Global::strprintf("Neural net Y length %d exceeds maximum supported %d", nnYLen, NNPos::MAX_BOARD_LEN)
      );
    }

    numInputChannels = desc.numInputChannels;
    numInputGlobalChannels = desc.numInputGlobalChannels;
    numInputMetaChannels = desc.numInputMetaChannels;
    numPolicyChannels = desc.numPolicyChannels;
    numValueChannels = desc.numValueChannels;
    numScoreValueChannels = desc.numScoreValueChannels;
    numOwnershipChannels = desc.numOwnershipChannels;

    // TODO: Check required workspaces sizes

    // TODO: Check partial models constructor parameters
    trunk = std::make_unique<Trunk>(handle, desc.trunk, nnXLen, nnYLen);
    policyHead = std::make_unique<PolicyHead>(handle, desc.policyHead, nnXLen, nnYLen);
    valueHead = std::make_unique<ValueHead>(handle, desc.valueHead, nnXLen, nnYLen);
  }

  ~Model() {

  }

  ConvWorkspaceEltsNeeded getWorkspaceElts(ComputeHandleInternal *handle) const {
    ConvWorkspaceEltsNeeded maxElts;
    // TODO: Implement workspace calculation
    return maxElts;
  }

  void record() {
  //   std::vector<VkCommandBuffer> trunkCommandBuffers = trunk->record();
  //   std::vector<VkCommandBuffer> policyHeadCommandBuffers = policyHead->record();
  //   std::vector<VkCommandBuffer> valueHeadCommandBuffers = valueHead->record();
  //   commandBuffers.emplace_back(trunkCommandBuffers.begin(), trunkCommandBuffers.end());
  //   commandBuffers.emplace_back(policyHeadCommandBuffers.begin(), policyHeadCommandBuffers.end());
  //   commandBuffers.emplace_back(valueHeadCommandBuffers.begin(), valueHeadCommandBuffers.end());
  }

  /**
   * @brief Execute inference on the model.
   */
  void apply( ComputeHandleInternal* handle ) const {
    VkSubmitInfo si = {};;
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    si.pCommandBuffers = commandBuffers.data();
    vkQueueSubmit(handle->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(handle->queue);
  }
};

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
    const LoadedModel* loadedModel,
    int maxBatchSize,
    int gpuIdx,
    bool inputUseNHWC_
  ): 
    handle( std::make_unique<ComputeHandleInternal>(
      context,
      gpuIdx,
      inputUseNHWC_,
      context->usingNHWCMode == enabled_t::True ? true : false
    )),
    nnXLen(context->nnXLen),
    nnYLen(context->nnYLen),
    policySize(NNPos::getPolicySize(context->nnXLen,context->nnYLen)),
    inputUseNHWC(inputUseNHWC_)
  {
  }

  ~ComputeHandle() {}

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger *logger,
  int maxBatchSize,
  bool requiredExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  auto deviceStr = [&]() {
    if(gpuIdxForThisThread < 0) 
      return std::string("");
    return " Device " + Global::intToString(gpuIdxForThisThread);
  };

  if ( logger != nullptr ) {
    // logger->write("Vulkan backend trhead " + Global::intToString(serverThreadIdx) + " Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));
    // logger->write("Vulkan backend thread " + Global::intToString(serverThreadIdx) + " using FP16 mode: " + context->usingFP16Mode.toString() + "," + " NHWC mode: " + context->usingNHWCMode.toString() + deviceStr() );
  }
}

ComputeHandleInternal::ComputeHandleInternal(ComputeContext* ctx, int gpuIdx, bool inputsUseNHWC, bool useNHWC) {
  this->context = ctx;
  this->vulkanDevice = ctx->vulkanContext->findGpuExn(gpuIdx);
  this->queue = this->vulkanDevice->queue;
  this->device = this->vulkanDevice->device;
};



void NeuralNet::globalInitialize() {
  static_assert(sizeof(int) >= 4, "");
}

void NeuralNet::globalCleanup() {
}

namespace KatagoVulkan {

// ########################### Compute Pipelines #########################
  ComputePipelines::ComputePipelines(
    VkDevice device_
  ): device(device_) {
    VkResult res = VK_ERROR_UNKNOWN;
    cache = VkHelpers::createPipelineCache(device, &res);
    CHECK_VK(res);
    createPipelines();
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
    createConv2d3x3BnFp32();
    createConv2d3x3BnReluFp32();
    createConv2d5x5BnFp32();
    createConv2d5x5BnReluFp32();
    createConv2d5x5BnMishFp32();
    createAddPointWiseFp32();
    createMatmulFp32();
    // createMatmulTiled4x4x32Fp32();
    createBatchNormMaskFp32();
    createBatchNormMaskReluFp32();
    createBatchNormMaskMishFp32();
    createGlobalPoolingChannelsFp32();
    createValueHeadPoolingChannelsFp32();
    createSumChannelsFp32();
    createAddChannelBiasNCHWFp32();
    createAddChannelBiasNCHWReluFp32();
    createAddChannelBiasNCHWMishFp32();
    createExtractChannel0NCHWFp32();
  }

  void ComputePipelines::destroyPipelines() {
    destroyPipeline(conv2dFp32);
    destroyPipeline(conv2d3x3BnFp32);
    destroyPipeline(conv2d3x3BnReluFp32);
    destroyPipeline(conv2d5x5BnFp32);
    destroyPipeline(conv2d5x5BnReluFp32);
    destroyPipeline(conv2d5x5BnMishFp32);
    destroyPipeline(addPointWiseFp32);
    destroyPipeline(matmulFp32);
    // destroyPipeline(matmulTiledChw4x4x32Fp32);
    destroyPipeline(batchNormMaskFp32);
    destroyPipeline(batchNormMaskReluFp32);
    destroyPipeline(batchNormMaskMishFp32);
    destroyPipeline(globalPoolingChannelsFp32);
    destroyPipeline(valueHeadPoolingChannelsFp32);
    destroyPipeline(sumChannelsFp32);
    destroyPipeline(addChannelBiasNCHWFp32);
    destroyPipeline(addChannelBiasNCHWReluFp32);
    destroyPipeline(addChannelBiasNCHWMishFp32);
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
  void ComputePipelines::createConv2dFp32() {
    createPipeline("Conv2dFp32",  VkSPIRVShaders::spirv_conv2d_fp32, VkSPIRVShaders::spirv_conv2d_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2dFp32);
  }

  /**
   * @brief Create a Conv2d3x3 Bn + Identity Activation fused Fp32 objects.
   */
  void ComputePipelines::createConv2d3x3BnFp32() {
    createPipeline("Conv2d3x3BnFp32", VkSPIRVShaders::spirv_conv2d_3x3_bn_fp32, VkSPIRVShaders::spirv_conv2d_3x3_bn_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d3x3BnFp32);
  }

  /**
   * @brief Create a Conv2d3x3 Bn + ReLU Activation fused Fp32 objects.
   */
  void ComputePipelines::createConv2d3x3BnReluFp32() {
    createPipeline("Conv2d3x3BnReluFp32", VkSPIRVShaders::spirv_conv2d_3x3_bn_relu_fp32, VkSPIRVShaders::spirv_conv2d_3x3_bn_relu_fp32_size,   3, sizeof(Conv2DPushConstantParams), conv2d3x3BnReluFp32);
  }
  /**
   * @brief Create a Conv2d3x3 Bn Mish Fp32 object
   */
  void ComputePipelines::createConv2d3x3BnMishFp32() {
    createPipeline("Conv2d3x3BnMishFp32", VkSPIRVShaders::spirv_conv2d_3x3_bn_mish_fp32,VkSPIRVShaders::spirv_conv2d_3x3_bn_mish_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d3x3BnMishFp32);
  }

  /**
   * @brief Create a Conv2d5x5 Bn + Identity Activation fused Fp32 objects.
   */
  void ComputePipelines::createConv2d5x5BnFp32() {
    createPipeline("Conv2d5x5BnFp32", VkSPIRVShaders::spirv_conv2d_5x5_bn_fp32, VkSPIRVShaders::spirv_conv2d_5x5_bn_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnFp32);
  }

  /**
   * @brief Create a Conv2d5x5 Bn + ReLU Activation fused Fp32 objects.
   */
  void ComputePipelines::createConv2d5x5BnReluFp32() {
    createPipeline("Conv2d5x5BnReluFp32",VkSPIRVShaders::spirv_conv2d_5x5_bn_relu_fp32, VkSPIRVShaders::spirv_conv2d_5x5_bn_relu_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnReluFp32);
  }

  /**
   * @brief Create a Conv2d5x5 Bn Mish Fp32 object
   */
  void ComputePipelines::createConv2d5x5BnMishFp32() {
    createPipeline("Conv2d5x5BnMishFp32",VkSPIRVShaders::spirv_conv2d_5x5_bn_mish_fp32, VkSPIRVShaders::spirv_conv2d_5x5_bn_mish_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnMishFp32);
  }

  /**
   * @brief Create a Add Point Wise Fp32 object
   */
  void ComputePipelines::createAddPointWiseFp32() {
    createPipeline("AddPointWiseFp32",VkSPIRVShaders::spirv_add_pointwise_fp32, VkSPIRVShaders::spirv_add_pointwise_fp32_size, 3, sizeof(KatagoVulkan::MatmulFp32Params), addPointWiseFp32);
  }

  /**
   * @brief Create a Matmul Fp32 object
   */
  void ComputePipelines::createMatmulFp32() {
    createPipeline("MatmulFp32", VkSPIRVShaders::spirv_matmul_fp32, VkSPIRVShaders::spirv_matmul_fp32_size, 3, sizeof(KatagoVulkan::MatmulFp32Params), matmulFp32);
  }

  /**
   * @brief Create a Matmul Tiled 4x4x32 Fp32 object
  //  */
  // void ComputePipelines::createMatmulTiled4x4x32Fp32() {
  //   createPipeline("MatmulTiled4x4x32Fp32", VkSPIRVShaders::spirv_matmul_tiled_chw_4x4x32_fp32, VkSPIRVShaders::spirv_matmul_tiled_chw_4x4x32_fp32_size, 3, sizeof(KatagoVulkan::MatmulTiledChw4x4x32Fp32Params), matmulTiledChw4x4x32Fp32);
  // }

  /**
   * @brief Create a BatchNorm Mask Fp32 object
   */
  void ComputePipelines::createBatchNormMaskFp32() {
    createPipeline("BatchNormMaskFp32", VkSPIRVShaders::spirv_bn_mask_fp32, VkSPIRVShaders::spirv_bn_mask_fp32_size, 3, sizeof(KatagoVulkan::NCHWPushConstantParams), batchNormMaskFp32);
  }

  /**
   * @brief Create a BatchNorm Mask + ReLU Fp32 object
   */
  void ComputePipelines::createBatchNormMaskReluFp32() {
    createPipeline("BatchNormMaskReluFp32", VkSPIRVShaders::spirv_bn_mask_relu_fp32, VkSPIRVShaders::spirv_bn_mask_relu_fp32_size, 3, sizeof(KatagoVulkan::NCHWPushConstantParams), batchNormMaskReluFp32);
  }

  /**
   * @brief Create a BatchNorm Mask + Mish Fp32 object
   */
  void ComputePipelines::createBatchNormMaskMishFp32() {
    createPipeline("BatchNormMaskMishFp32", VkSPIRVShaders::spirv_bn_mask_mish_fp32, VkSPIRVShaders::spirv_bn_mask_mish_fp32_size, 3, sizeof(KatagoVulkan::NCHWPushConstantParams), batchNormMaskMishFp32);
  }

  /**
   * @brief Create a Global Average Pool Fp32 object
   */
  void ComputePipelines::createGlobalPoolingChannelsFp32() {
    createPipeline("GlobalPoolingChannelsFp32", VkSPIRVShaders::spirv_global_pooling_channels_fp32, VkSPIRVShaders::spirv_global_pooling_channels_fp32_size, 2, sizeof(KatagoVulkan::NCHWPushConstantParams), globalPoolingChannelsFp32);
  }

  /**
   * @brief Create a Value Head Pool Channels Fp32 object
   */
  void ComputePipelines::createValueHeadPoolingChannelsFp32() {
    createPipeline("ValueHeadPoolingChannelsFp32", VkSPIRVShaders::spirv_value_head_pool_channels_fp32, VkSPIRVShaders::spirv_value_head_pool_channels_fp32_size, 2, sizeof(KatagoVulkan::NCHWPushConstantParams), valueHeadPoolingChannelsFp32);
  }

  /**
   * @brief Create a Sum Channels Fp32 object
   */
  void ComputePipelines::createSumChannelsFp32() {
    createPipeline("SumChannelsFp32", VkSPIRVShaders::spirv_sum_channels_fp32, VkSPIRVShaders::spirv_sum_channels_fp32_size, 2, sizeof(KatagoVulkan::NCHWPushConstantParams), sumChannelsFp32);
  }

  /**
   * @brief Create a Add Channel Bias NCHW Fp32 object
   */
  void ComputePipelines::createAddChannelBiasNCHWFp32() {
    createPipeline("AddChannelBiasNCHWFp32", VkSPIRVShaders::spirv_add_channel_bias_nchw_fp32, VkSPIRVShaders::spirv_add_channel_bias_nchw_fp32_size, 2, sizeof(KatagoVulkan::MatBiasFp32Params), addChannelBiasNCHWFp32);
  }

  /**
   * @brief Create a Add Channel Bias NC + ReLU Fp32 object
   */
  void ComputePipelines::createAddChannelBiasNCHWReluFp32() {
    createPipeline("AddChannelBiasNCHWReluFp32", VkSPIRVShaders::spirv_add_channel_bias_nc_relu_fp32, VkSPIRVShaders::spirv_add_channel_bias_nc_relu_fp32_size, 2, sizeof(KatagoVulkan::MatBiasFp32Params), addChannelBiasNCHWReluFp32);
  }

  /**
   * @brief Create a Add Channel Bias NC + Mish Fp32 object
   */
  void ComputePipelines::createAddChannelBiasNCHWMishFp32() {
    createPipeline("AddChannelBiasNCHWMishFp32", VkSPIRVShaders::spirv_add_channel_bias_nc_mish_fp32, VkSPIRVShaders::spirv_add_channel_bias_nc_mish_fp32_size, 2, sizeof(KatagoVulkan::MatBiasFp32Params), addChannelBiasNCHWMishFp32);
  }
  
  /**
   * @brief Create a Extract Channel 0 NCHW Fp32 object
   */
  void ComputePipelines::createExtractChannel0NCHWFp32() {
    createPipeline("ExtractChannel0NCHWFp32", VkSPIRVShaders::spirv_extract_channel0_nchw_fp32, VkSPIRVShaders::spirv_extract_channel0_nchw_fp32_size, 2, sizeof(KatagoVulkan::NCHWPushConstantParams), extractChannel0NCHWFp32);
  }

  // ########################### End of Compute Pipelines #########################
}


#endif