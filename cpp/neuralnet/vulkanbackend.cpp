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
 */
struct ConvLayer {
  const ComputeHandleInternal* handle;
  const std::string name;
  const int convYSize;
  const int convXSize;
  const int convYRadius;
  const int convXRadius;
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
    convYRadius(desc->convYSize / 2),
    convXRadius(desc->convXSize / 2),
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

  bool canApplyWithBNActivation() {
    return (convXSize == 3 || convXSize == 5) && (convYSize == 3 || convYSize == 5);
  }

  void do5x5ConvBnReluFp32(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    if ( commandBuffer != VK_NULL_HANDLE ) {
      return; // Already recorded
    }

    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    KatagoVulkan::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
    VkResult res;
    descriptorSet = VkHelpers::allocateDescriptorSet(
      handle->vulkanDevice,
      pipelines->conv2d5x5BnFp32.descriptorSetLayout,
      &res
    );
    CHECK_VK_MSG("Allocate descriptor set for ConvLayer: " + name, res);

    // update descriptor set
    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {};
  }

  void do3x3ConvBnReluFp32(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {

  }

  void do1x1ConvFp32(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
  }

  /**
   * @brief create command buffer and record for conv layer
   * @param batchSize 
   * @param input 
   * @param output 
   */
  void record(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
    VkResult res = VkHelpers::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for ConvLayer: " + name, res);
    if ( convXSize == 5 && convYSize == 5 ) {
      do5x5ConvFp32(
        batchSize,
        input,
        output
      );
    } else if ( convXSize == 3 && convYSize == 3 ) {
      do3x3ConvFp32(
        batchSize,
        input,
        output
      );
    } else if ( convXSize == 1 && convYSize == 1 ) {
      do1x1ConvFp32(
        batchSize,
        input,
        output
      );
      // 1x1 conv implemented with matmul approach
    } else {
      // TODO: Support batch tiled conv for other sizes
      throw StringError("Vulkan ConvLayer: " + name + " unsupported conv size");
    }
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
};

struct BatchNormLayer {

};

struct MatBiasLayer {

};

struct NormActConv {
};

/**
 * @brief Basic Residual Block, Consist of two conv layers with BN and Activation and one skip connection
 */
struct ResidualBlock {
};

struct GlobalPoolingResidualBlock {

};

struct NestedResidualBlock {
};

// constexpr int ORDINARY_BLOCK_KIND = 0;
// constexpr int GLOBAL_POOLING_BLOCK_KIND = 1;
// constexpr int NESTED_BLOCK_KIND = 2;

struct BlockStack {
};

struct Trunk {
};

struct PolicyHead {
};

struct ValueHead {
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

struct ComputeHandleInternal {
  const ComputeContext* context;
  const VulkanDevice* vulkanDevice;
  VkDevice device;
  VkQueue queue;
  // bool usingFP16Storage;
  // bool usingFP16Compute;
  // bool usingFP16TensorCores;
  // bool usingFP16TensorCoreForConv1x1;

  ComputeHandleInternal(
    ComputeContext* ctx,
    int gpuIdx,
    bool inputsUseNHWC,
    bool useNHWC
  ) {
    this->context = ctx;
    this->vulkanDevice = ctx->vulkanContext->findGpuExn(gpuIdx);
    this->queue = this->vulkanDevice->queue;
    this->device = this->vulkanDevice->device;
  }
};

struct ScratchBuffers {
  const size_t batchXYFloatBytes;
  const size_t batchFloatBytes;
  const size_t batchXYBytes;
  const size_t batchBytes;

  const ComputeHandleInternal *handle;
  SimpleAllocator<VulkanBuffer*>* allocator;

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
    createAddChannelBiasNCReluFp32();
    createAddChannelBiasNCMishFp32();
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
    destroyPipeline(addChannelBiasNCReluFp32);
    destroyPipeline(addChannelBiasNCMishFp32);
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
    createPipeline("AddChannelBiasNCHWFp32", VkSPIRVShaders::spirv_add_channel_bias_nchw_fp32, VkSPIRVShaders::spirv_add_channel_bias_nchw_fp32_size, 2, sizeof(KatagoVulkan::NCHWPushConstantParams), addChannelBiasNCHWFp32);
  }

  /**
   * @brief Create a Add Channel Bias NC + ReLU Fp32 object
   */
  void ComputePipelines::createAddChannelBiasNCReluFp32() {
    createPipeline("AddChannelBiasNCReluFp32", VkSPIRVShaders::spirv_add_channel_bias_nc_relu_fp32, VkSPIRVShaders::spirv_add_channel_bias_nc_relu_fp32_size, 2, sizeof(KatagoVulkan::NCHWPushConstantParams), addChannelBiasNCReluFp32);
  }

  /**
   * @brief Create a Add Channel Bias NC + Mish Fp32 object
   */
  void ComputePipelines::createAddChannelBiasNCMishFp32() {
    createPipeline("AddChannelBiasNCMishFp32", VkSPIRVShaders::spirv_add_channel_bias_nc_mish_fp32, VkSPIRVShaders::spirv_add_channel_bias_nc_mish_fp32_size, 2, sizeof(KatagoVulkan::NCHWPushConstantParams), addChannelBiasNCMishFp32);
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