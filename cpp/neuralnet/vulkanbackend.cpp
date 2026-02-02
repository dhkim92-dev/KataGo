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
#include "../core/test.h"
#include "../core/using.h"
#include "../external/vulkan/shaders/common.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/vulkanbackend.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkanshaders.h"

static void printHostBuffer(
  std::string prefix,
  const float* hostBuffer,
  size_t numElts
) {
  // print prefix first
  std::cout << prefix << " = " << std::endl;
  // print vector as python format, that can copy it to python code
  std::cout << "[";
  for ( size_t i = 0 ; i < numElts ; i++ ) {
    std::cout << hostBuffer[i];
    if ( i != numElts - 1 ) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

static void printDeviceBuffer(
  std::string prefix,
  const VulkanDevice* device,
  VulkanBuffer* buffer,
  size_t numElts
) {
  VkResult res;
  std::vector<float> hostBuffer(numElts);
  VkHelpers::copyDeviceBufferToHost(
    device,
    buffer,
    numElts * sizeof(float),
    hostBuffer.data(),
    true,
    &res
  );
  CHECK_VK_MSG("printDeviceBuffer copyDeviceToHostBuffer", res);

  // print prefix first
  std::cout << prefix << " = " << std::endl;
  // print vector as python format, that can copy it to python code
  std::cout << "[";
  for ( size_t i = 0 ; i < numElts ; i++ ) {
    std::cout << hostBuffer[i];
    if ( i != numElts - 1 ) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

struct ComputeContext {
  std::vector<uint32_t> gIdx;
  const int nnXLen;
  const int nnYLen;
  const enabled_t usingFP16Mode;
  const enabled_t usingNHWCMode;
  VulkanContext* vulkanContext; 
  std::unordered_map<uint32_t, KatagoVulkan::ComputePipelines *> pipelinesPerDev;
  Logger* logger;

  ComputeContext(
    int nnXLen,
    int nnYLen,
    enabled_t useFP16Mode,
    enabled_t useNHWCModel,
    const std::vector<uint32_t>& gpuIdxsToUse,
    Logger* logger_)
  : nnXLen(nnXLen),
    nnYLen(nnYLen),
    usingFP16Mode(useFP16Mode),
    usingNHWCMode(useNHWCModel),
    gIdx(gpuIdxsToUse),
    logger(logger_)
     {
      VkInstance instance = VkHelpers::createVulkanInstance();
      std::vector<VulkanDeviceInfo> allDeviceInfos = VkHelpers::enumerateVulkanDevices(instance, logger);
      std::vector<VulkanDevice *> vulkanDevices = {};

      if ( gpuIdxsToUse.size() == 1 && gpuIdxsToUse[0] == UINT32_MAX ) {
        if ( logger ) {
          logger->write("No GPU index specified, using default GPU 0");
        }
        gIdx[0] = 0; // use default GPU
      }

      std::sort(gIdx.begin(), gIdx.end());
      gIdx.erase(std::unique(gIdx.begin(), gIdx.end()), gIdx.end());

      for ( size_t i = 0 ; i < gIdx.size() ; i++ ) {
        uint32_t gpuIdx = gIdx[i];
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
        if ( logger ) {
          logger->write("Created Vulkan Compute Pipelines for device: " + deviceInfo.deviceName);
        }
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


VkDeviceSize getRequiredMemorySize(
  const LoadedModel* loadedModel
) {
  // For simplicity, return a fixed size for now.
  // In future, we can calculate based on model parameters.
  return static_cast<VkDeviceSize>(512) * 1024 * 1024; // 512 MB
}

\
/**
 * @brief Print float buffer for debugging
 * @param prefix Prefix string to identify the buffer
 * @param buffer Pointer to the float buffer
 * @param numElts Number of elements in the buffer
 * @param batchSize Batch size
 * @param nChannels Number of channels
 * @param nRows Number of rows
 * @param nCols Number of columns
 */
static void printFloatBuffer(
  std::string prefix,
  const float* buffer,
  size_t numElts,
  int batchSize,
  int nChannels,
  int nRows,
  int nCols
) {
  std::printf("[%s] buffer size: %zu\n", prefix.c_str(), numElts);
  nChannels =nChannels > 3 ? 3 : nChannels; // limit channels to print
  batchSize = batchSize > 1 ? 1 : batchSize; // limit batch size to print
  for ( int b = 0 ; b < batchSize ; ++b ) {
    for ( int c = 0 ; c < nChannels ; ++c ) {
      std::printf("[%s] Batch %d Channel %d:\n", prefix.c_str(), b, c);
      std::printf("[ \n");
      for ( int r = 0 ; r < nRows ; ++r ) {
        std::printf("  [    ");
        for ( int col = 0 ; col < nCols ; ++col ) {
          size_t idx = static_cast<size_t>(b) * static_cast<size_t>(nChannels) * static_cast<size_t>(nRows) * static_cast<size_t>(nCols)
                       + static_cast<size_t>(c) * static_cast<size_t>(nRows) * static_cast<size_t>(nCols)
                       + static_cast<size_t>(r) * static_cast<size_t>(nCols)
                       + static_cast<size_t>(col);
          if ( idx < numElts ) {
            std::printf("%f ", buffer[idx]);
          } else {
            std::printf("X ");
          }
        }
        std::printf("]\n");
      }
      std::printf("]\n"); 
    }
  }
}

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
      // printFloatBuffer("MatmulLayer " + name + " weights: ", weights.data(), weights.size(), 1, outChannels, 1, inChannels);
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
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] try to destroy");
    if ( matBuf != nullptr ) {
      VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, matBuf);
      matBuf = nullptr;
    }
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] destroyed");
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
    VkHelpers::barrierCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for MatmulLayer: " + name, res);
    doMatmulFp32(batchSize, input, output);
    res = VkHelpers::endCommandBuffer(commandBuffer);
    CHECK_VK_MSG("End command buffer for MatmulLayer: " + name, res);
  }

  /**
   * @brief Launch the recorded command buffer, only for debug now.
   * @param batchSize 
   * @param input 
   * @param output 
   */
  void apply(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    assert(commandBuffer != VK_NULL_HANDLE);
    // if ( name == "v3/w" ) {
      // printDeviceBuffer("MatmulLayer " + name + " Input : ", handle->vulkanDevice, input, static_cast<size_t>(batchSize) * static_cast<size_t>(inChannels));
    // }
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, {commandBuffer});
    size_t outputSize = sizeof(float) * static_cast<size_t>(batchSize) * static_cast<size_t>(outChannels);
    std::vector<float> outputCopy(outputSize / sizeof(float));
    VkResult res = VK_ERROR_UNKNOWN;
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, output, outputSize, outputCopy.data(), true, &res);
    CHECK_VK_MSG("Copy output buffer to host for MatmulLayer: " + name, res);
    printFloatBuffer(name + " Output : ", outputCopy.data(), outputCopy.size(), batchSize, outChannels, 1, 1);
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
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, input),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, matBuf),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, output)
    };
    VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

    auto pushConstants = KatagoVulkan::MatmulFp32Params();
    pushConstants.M = static_cast<uint32_t>(batchSize);
    pushConstants.K = static_cast<uint32_t>(inChannels);
    pushConstants.N = static_cast<uint32_t>(outChannels);
    pushConstants.numBatchElts = 1;
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

    // Matmul uses numthreads(MATMUL_DISPATCH_X=16, MATMUL_DISPATCH_Y=16) with TILE_M=16, TILE_N=16
    // HLSL: GId.x = groupM (M axis), GId.y = groupN (N axis)
    uint32_t groupCountX = (static_cast<uint32_t>(batchSize) + MATMUL_TILE_M - 1) / MATMUL_TILE_M;  // M Direction
    uint32_t groupCountY = (static_cast<uint32_t>(outChannels) + MATMUL_TILE_N - 1) / MATMUL_TILE_N; // N Direction
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
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
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] try to destroy");
    if ( filterBuf != nullptr ) {
      VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, filterBuf);
      filterBuf = nullptr;
    }
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] destroyed");
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
    CHECK_VK_MSG("Begin command buffer for ConvLayer: " + name, res);
    VkHelpers::barrierCommandBuffer(commandBuffer);
    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    KatagoVulkan::ComputePipelines* pipelines = this->handle->context->pipelinesPerDev.at(gpuId);
    
    descriptorSet = VkHelpers::allocateDescriptorSet(
      handle->vulkanDevice,
      pipelines->conv2dFp32.descriptorSetLayout,
      &res
    );
    CHECK_VK_MSG("Allocate descriptor set for ConvLayer: " + name, res);
    // update descriptor set
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, input),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, filterBuf),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, output)
    };
    VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines->conv2dFp32.pipeline);
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
    
    // Compute dispatch dimensions matching HLSL numthreads(TILE_N,TILE_M,1)
    const uint32_t TILE_N = 8u; // local X (numthreads x)
    const uint32_t TILE_M = 8u; // local Y (numthreads y)
    uint32_t dispatchX = (pushConstants.nnXLen + TILE_N - 1u) / TILE_N;
    uint32_t dispatchY = pushConstants.nnYLen;
    uint32_t ocGroupsPerBatch = (pushConstants.outChannels + TILE_M - 1u) / TILE_M;
    uint32_t dispatchZ = pushConstants.batchSize * ocGroupsPerBatch;

    vkCmdDispatch(commandBuffer, dispatchX, dispatchY, dispatchZ);
    res = VkHelpers::endCommandBuffer(commandBuffer);
  }

  /**
   * @brief Launch the recorded command buffer, only for debug now.
   * @param batchSize 
   * @param input 
   * @param output 
   * @return VkCommandBuffer 
   */
  void apply(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    if ( commandBuffer == VK_NULL_HANDLE ) {
      throw StringError("ConvLayer: " + name + " command buffer not recorded yet");
    }

    VkHelpers::submitCommandBuffers(handle->vulkanDevice, {commandBuffer});
    size_t outputSize = sizeof(float) * static_cast<size_t>(batchSize) * static_cast<size_t>(outChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen);
    std::vector<float> outputCopy(outputSize / sizeof(float));
    VkResult res = VK_ERROR_UNKNOWN;
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, output, outputSize, outputCopy.data(), true, &res);
    CHECK_VK_MSG("Copy output buffer to host for ConvLayer: " + name, res);
    printFloatBuffer(name + " Output: ", outputCopy.data(), outputCopy.size(), batchSize, outChannels, nnYLen, nnXLen);
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
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] try to destroy");
    if ( mergedScaleBuf != nullptr ) {
      VkHelpers::releaseVulkanBuffer(mergedScaleBuf->device, mergedScaleBuf);
      // delete mergedScaleBuf;
      mergedScaleBuf = nullptr;
    }
    if ( mergedBiasBuf != nullptr ) {
      VkHelpers::releaseVulkanBuffer(mergedBiasBuf->device, mergedBiasBuf);
      // delete mergedBiasBuf;
      mergedBiasBuf = nullptr;
    }
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] destroyed");
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

    // globalSizes are dispatch group counts, computed by dividing problem size by local workgroup size
    // BatchNorm uses numthreads(BN_DISPATCH_X=16, BN_DISPATCH_Y=16, BN_DISPATCH_Z=1)
    // Thread mapping: x->channel, y->spatial, z->batch
    globalSizes[0] = (static_cast<size_t>(numChannels) + BN_DISPATCH_X - 1) / BN_DISPATCH_X;
    globalSizes[1] = (static_cast<size_t>(nnXYLen) + BN_DISPATCH_Y - 1) / BN_DISPATCH_Y;
  }

  void record(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* mask,
    VulkanBuffer* output
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
      case ACTIVATION_MISH_SCALE8:
        targetPipeline = pipelines->batchNormMaskMishScale8Fp32;
        break;
      default:
        Global::fatalError("Unsupported activation in BatchNormLayer: " + name);
    }

    commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
    VkResult res = VkHelpers::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for BatchNormLayer: " + name, res);
    VkHelpers::barrierCommandBuffer(commandBuffer);

    descriptorSet = VkHelpers::allocateDescriptorSet(
      handle->vulkanDevice,
      targetPipeline.descriptorSetLayout,
      &res
    );
    CHECK_VK_MSG("Allocate descriptor set for BatchNormLayer: " + name, res);

    // update descriptor set
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, input),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, mask),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, mergedScaleBuf),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 3, mergedBiasBuf),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 4, output)
    };
    VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

    KatagoVulkan::BatchNormMaskParams pushConstants = {};
    pushConstants.batchSize = static_cast<uint32_t>(batchSize);
    pushConstants.numChannels = static_cast<uint32_t>(numChannels);
    pushConstants.nnXYLen = static_cast<uint32_t>(nnXYLen);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.layout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, targetPipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KatagoVulkan::BatchNormMaskParams), &pushConstants);
    // Dispatch: x=channels, y=spatial, z=batch (shader uses BN_DISPATCH_X=16, BN_DISPATCH_Y=16, BN_DISPATCH_Z=1)
    uint32_t dispatchX = (static_cast<uint32_t>(numChannels) + BN_DISPATCH_X - 1) / BN_DISPATCH_X;
    uint32_t dispatchY = (static_cast<uint32_t>(nnXYLen) + BN_DISPATCH_Y - 1) / BN_DISPATCH_Y;
    uint32_t dispatchZ = static_cast<uint32_t>(batchSize);
    vkCmdDispatch(commandBuffer, dispatchX, dispatchY, dispatchZ);
    res = VkHelpers::endCommandBuffer(commandBuffer);

    CHECK_VK_MSG("End command buffer for BatchNormLayer: " + name, res);
  }

  /**
   * @brief Launch the recorded command buffer, only for debug now.
   * @param batchSize 
   * @param input 
   * @param mask
   * @param output 
   */
  void apply(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* mask,
    VulkanBuffer* output
  ) {
    if ( commandBuffer == VK_NULL_HANDLE ) {
      throw StringError("BatchNormLayer: " + name + " command buffer not recorded yet");
    }

    // copy input and printing
    std::vector<float> inputCopy(static_cast<size_t>(batchSize) * static_cast<size_t>(numChannels) * static_cast<size_t>(nnXYLen));
    VkResult res = VK_ERROR_UNKNOWN;
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, input, sizeof(float) * static_cast<size_t>(batchSize) * static_cast<size_t>(numChannels) * static_cast<size_t>(nnXYLen), inputCopy.data(), true, &res);
    CHECK_VK_MSG("Copy input buffer to host for BatchNormLayer: " + name, res);
    printFloatBuffer(name + " Input: ", inputCopy.data(), inputCopy.size(), batchSize, numChannels, nnYLen, nnXLen);
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, {commandBuffer});
    size_t outputSize = sizeof(float) * static_cast<size_t>(batchSize) * static_cast<size_t>(numChannels) * static_cast<size_t>(nnXYLen);
    std::vector<float> outputCopy(outputSize / sizeof(float));
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, output, outputSize, outputCopy.data(), true, &res);
    CHECK_VK_MSG("Copy output buffer to host for BatchNormLayer: " + name, res);
    printFloatBuffer(name + " Output: ", outputCopy.data(), outputCopy.size(), batchSize, numChannels, nnYLen, nnXLen);
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
  float bias;

  ~MatBiasLayer() {
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] try to destroy");
    if ( biasBuf != nullptr ) {
      VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, biasBuf);
      // delete biasBuf;
      biasBuf = nullptr;
    }
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] destroyed");
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

      // printHostBuffer("MatBiasLayer " + name + " numChannels: " + std::to_string(numChannels) + " weights: ", desc->weights.data(), desc->weights.size());

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
        targetPipeline = pipelines->addChannelBiasNCIdentityFp32;
        break;
      case ACTIVATION_RELU:
        targetPipeline = pipelines->addChannelBiasNCReluFp32;
        break;
      case ACTIVATION_MISH:
        targetPipeline = pipelines->addChannelBiasNCMishFp32;
        break;
      case ACTIVATION_MISH_SCALE8: 
        targetPipeline = pipelines->addChannelBiasNCMishScale8Fp32;
        break;
      default:
        Global::fatalError("Unsupported activation in MatBiasLayer: " + name);
    }

    VkResult res;
    commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
    descriptorSet = VkHelpers::allocateDescriptorSet(handle->vulkanDevice, targetPipeline.descriptorSetLayout, &res);
    CHECK_VK_MSG("Allocate descriptor set for MatBiasLayer: " + name, res);
    VkHelpers::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for MatBiasLayer: " + name, res);
    VkHelpers::barrierCommandBuffer(commandBuffer);
    auto pushConstants = KatagoVulkan::AddChannelBiasNCParams();
    pushConstants.nSize = batchSize;  // No spatial dimension for NC tensor
    pushConstants.cSize = numChannels;
    // update descriptor set
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
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
      sizeof(KatagoVulkan::AddChannelBiasNCParams),
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

    static constexpr int nKernelDims = 2;
    uint32_t dispatchX = (static_cast<uint32_t>(batchSize) + ADD_CHANNELS_DISPATCH_X - 1) / ADD_CHANNELS_DISPATCH_X;
    uint32_t dispatchY = (static_cast<uint32_t>(numChannels) + ADD_CHANNELS_DISPATCH_Y - 1) / ADD_CHANNELS_DISPATCH_Y;
    vkCmdDispatch(commandBuffer, dispatchX, dispatchY, 1);
    res = VkHelpers::endCommandBuffer(commandBuffer);
    CHECK_VK_MSG("End command buffer for MatBiasLayer: " + name, res);
  }

  /**
   * @brief Launch the recorded command buffer, only for debug now.
   * @param batchSize
   * @param input
   */
  void apply(int batchSize, VulkanBuffer* input) {
    if ( commandBuffer == VK_NULL_HANDLE ) {
      throw StringError("MatBiasLayer: " + name + " command buffer not recorded yet");
    }

    // std::cout << "MatBiasLayer " << name << " Activation : " << activation << std::endl;
    // printDeviceBuffer("MatBiasLayer " + name + " Input: ", handle->vulkanDevice, input, static_cast<size_t>(batchSize) * static_cast<size_t>(numChannels));
    // printDeviceBuffer("MatBiasLayer " + name + " Bias: ", handle->vulkanDevice, biasBuf, static_cast<size_t>(numChannels));

    VkHelpers::submitCommandBuffers(handle->vulkanDevice, {commandBuffer});
    // size_t outputSize = sizeof(float) * static_cast<size_t>(batchSize) * static_cast<size_t>(numChannels);
    // printDeviceBuffer("MatBiasLayer " + name + " Output: ", handle->vulkanDevice, input, static_cast<size_t>(batchSize) * static_cast<size_t>(numChannels));
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
    bn.record(batchSize, input, mask, inputScratchOrInput);
    conv.record(batchSize, inputScratchOrInput, output);
    commandBuffers.push_back( bn.commandBuffer );
    commandBuffers.push_back( conv.commandBuffer );
  }

  /**
   * @brief Launch the recorded command buffers, only for debug now.
   * @param batchSize
   * @param input
   * @param inputScratchOrInput
   * @param output
   */
  void apply(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* inputScratchOrInput,
    VulkanBuffer* output,
    VulkanBuffer* mask
    // VulkanBuffer* convWorkspace,
    // VulkanBuffer* convWorkspace2
  ) {
    if ( commandBuffers.size() != 2 ) {
      throw StringError("NormActConv command buffers not recorded yet");
    }
    bn.apply(batchSize, input, mask, inputScratchOrInput);
    conv.apply(batchSize, inputScratchOrInput, output);
  }

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;
};

VkCommandBuffer performExtractChannel0NCHW(
  ComputeHandleInternal *handle,
  VulkanBuffer* input,
  VulkanBuffer* output,
  int batchSIze,
  int numInputChannels,
  int nnXYLen
) {
  static constexpr int nKernelDims = 2;
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  KatagoVulkan::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  KatagoVulkan::Pipeline targetPipeline = pipelines->extractChannel0NCHWFp32;
  VkCommandBuffer commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
  VkResult res = VkHelpers::beginCommandBuffer(commandBuffer);
  VkHelpers::barrierCommandBuffer(commandBuffer);
  CHECK_VK_MSG("Begin command buffer for ExtractChannel0NCHW", res);
  VkDescriptorSet descriptorSet = VkHelpers::allocateDescriptorSet(handle->vulkanDevice, targetPipeline.descriptorSetLayout, &res);
  // update descriptor set
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, input),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, output)
  };
  VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

  CHECK_VK_MSG("Allocate descriptor set for ExtractChannel0NCHW", res);
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
  KatagoVulkan::ExtractChannel0NCHWParams pushConstants = {};
  pushConstants.numInputChannels = static_cast<uint32_t>(numInputChannels);
  pushConstants.batchSize = static_cast<uint32_t>(batchSIze);
  pushConstants.nnXYLen = static_cast<uint32_t>(nnXYLen);
  vkCmdPushConstants(
    commandBuffer,
    targetPipeline.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(KatagoVulkan::ExtractChannel0NCHWParams),
    &pushConstants
  );
  // ExtractChannel0 uses numthreads(EXTRACT_CHANNEL0_DISPATCH_X=64, EXTRACT_CHANNEL0_DISPATCH_Y=1)
  // Thread mapping: x->spatial, y->batch
  uint32_t dispatchX = (static_cast<uint32_t>(nnXYLen) + EXTRACT_CHANNEL0_DISPATCH_X - 1) / EXTRACT_CHANNEL0_DISPATCH_X;
  uint32_t dispatchY = static_cast<uint32_t>(batchSIze);
  vkCmdDispatch(commandBuffer, dispatchX, dispatchY, 1);
  VkHelpers::endCommandBuffer(commandBuffer);
  return commandBuffer;
}

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
  VkHelpers::barrierCommandBuffer(commandBuffer);
  CHECK_VK_MSG("Begin command buffer for AddChannelBiases", res);
  VkDescriptorSet descriptorSet = VkHelpers::allocateDescriptorSet(
    handle->vulkanDevice,
    targetPipeline.descriptorSetLayout,
    &res
  );
  // update descriptor set
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, input),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, bias)
  };
  VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

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
  KatagoVulkan::AddChannelBiasNCHWParams pushConstants = {};
  pushConstants.ncSize = static_cast<uint32_t>(ncSize);
  pushConstants.xySize = static_cast<uint32_t>(nnXYLen);
  vkCmdPushConstants(
    commandBuffer,
    targetPipeline.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(KatagoVulkan::AddChannelBiasNCHWParams),
    &pushConstants
  );
  // Dispatch count = globalSizes / localSizes
  // AddChannelBias shader uses numthreads(ADD_CHANNELS_DISPATCH_X=16, ADD_CHANNELS_DISPATCH_Y=16, 1)
  uint32_t dispatchX = (static_cast<uint32_t>(globalSizes[0]) + ADD_CHANNELS_DISPATCH_X - 1) / ADD_CHANNELS_DISPATCH_X;
  uint32_t dispatchY = (static_cast<uint32_t>(globalSizes[1]) + ADD_CHANNELS_DISPATCH_Y - 1) / ADD_CHANNELS_DISPATCH_Y;
  // vkCmdDispatch(commandBuffer, dispatchX, dispatchY, 1);
  vkCmdDispatch(commandBuffer, globalSizes[0], globalSizes[1], 1);
  VkHelpers::endCommandBuffer(commandBuffer);
  return commandBuffer;
}

VkCommandBuffer performAddPointWise(
  ComputeHandleInternal *handle,
  VulkanBuffer* acc,
  VulkanBuffer* value,
  int totalSize
) {
  static constexpr int nKernelDims = 1;
  size_t globalSizes[nKernelDims] = {
    VkHelpers::powerOf2ify(static_cast<size_t>(totalSize))
  };
  VkCommandBuffer commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
  VkResult res = VkHelpers::beginCommandBuffer(commandBuffer);
  VkHelpers::barrierCommandBuffer(commandBuffer);
  CHECK_VK_MSG("Begin command buffer for AddPointWise", res);
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  KatagoVulkan::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  VkDescriptorSet descriptorSet = VkHelpers::allocateDescriptorSet(
    handle->vulkanDevice,
    pipelines->addPointWiseFp32.descriptorSetLayout,
    &res
  );
  // update descriptor set
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, acc),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, value)
  };
  VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

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
  // AddPointWise shader uses numthreads(ADD_POINTWISE_DISPATCH_X=256, 1, 1)
  uint32_t groupCountX = (static_cast<uint32_t>(totalSize) + ADD_POINTWISE_DISPATCH_X - 1) / ADD_POINTWISE_DISPATCH_X;
  vkCmdDispatch(commandBuffer, groupCountX, 1, 1);
  VkHelpers::endCommandBuffer(commandBuffer);
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
  VkHelpers::barrierCommandBuffer(commandBuffer);
  CHECK_VK_MSG("Begin command buffer for GlobalPoolingMask", res);
  VkDescriptorSet descriptorSet = VkHelpers::allocateDescriptorSet(
    handle->vulkanDevice,
    pipelines->globalPoolingChannelsFp32.descriptorSetLayout,
    &res
  );
  CHECK_VK_MSG("Allocate descriptor set for GlobalPoolingMask", res);
  // update descriptor set
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, gpoolConvOut),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, gpoolConcat),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, mask),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 3, maskSum)
  };
  VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

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

  // GlobalPooling shader uses numthreads(POOLING_DISPATCH_X=64, 1, 1)
  // Shader uses SV_GroupID.y for channel, SV_GroupID.z for batch
  // OpenCL equivalent: localSizes={XYSTRIDE,CHANNELSTRIDE,BATCHSTRIDE}, globalSizes={XYSTRIDE, roundUp(channels), roundUp(batch)}
  // Since Vulkan shader has fixed numthreads with Y,Z=1, dispatch Y,Z directly equals channel and batch counts
  uint32_t groupCountX = 1u;
  uint32_t groupCountY = static_cast<uint32_t>(gpoolChannels);
  uint32_t groupCountZ = static_cast<uint32_t>(batchSize);
  vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
  res = VkHelpers::endCommandBuffer(commandBuffer);
  *result = res;
  CHECK_VK_MSG("End command buffer for GlobalPoolingMask", res);
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
  VkHelpers::barrierCommandBuffer(commandBuffer);
  CHECK_VK_MSG("Begin command buffer for ValueHeadPool", res);
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  KatagoVulkan::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  VkDescriptorSet descriptorSet = VkHelpers::allocateDescriptorSet(
    handle->vulkanDevice,
    pipelines->valueHeadPoolingChannelsFp32.descriptorSetLayout,
    &res
  );
  // update descriptor set
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, gpoolConvOut),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, gpoolConcat),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, maskSum)
  };
  VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

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
  KatagoVulkan::ValueHeadPoolingChannelsParams pushConstants = {};
  pushConstants.batchSize = static_cast<uint32_t>(batchSize);
  pushConstants.gpoolChannels = static_cast<uint32_t>(valueHeadChannels);
  pushConstants.nnXYLen = static_cast<uint32_t>(nnXYLen);
  vkCmdPushConstants(
    commandBuffer,
    pipelines->valueHeadPoolingChannelsFp32.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(KatagoVulkan::ValueHeadPoolingChannelsParams),
    &pushConstants
  );
  // ValueHeadPool shader uses numthreads(POOLING_DISPATCH_X=64, 1, 1)
  // Shader uses SV_GroupID.y for channel, SV_GroupID.z for batch
  // Same pattern as GlobalPooling - dispatch Y,Z directly equals channel and batch counts
  uint32_t groupCountX = 1u;
  uint32_t groupCountY = static_cast<uint32_t>(valueHeadChannels);
  uint32_t groupCountZ = static_cast<uint32_t>(batchSize);
  vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
  VkHelpers::endCommandBuffer(commandBuffer);

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
  VkCommandBuffer pointWiseAddCommandBuffer = VK_NULL_HANDLE;

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
    pointWiseAddCommandBuffer = pointWiseCB;
  }

  /**
   * @brief Launch the recorded command buffers, only for debug.
   */
  std::vector<VkCommandBuffer> apply(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask
  ) {
    assert( !commandBuffers.empty() );
    SizedBuf<VulkanBuffer*> mid(scratch->allocator, scratch->getBufSizeXY(normActConv->outChannels));
    normActConv->apply(batchSize, trunk, trunkScratch, mid.buf, mask);
    normActConv2->apply(batchSize, mid.buf, mid.buf, trunkScratch, mask);

    VkResult res = VK_ERROR_UNKNOWN;
    std::vector<float> trunkData(batchSize * normActConv2->outChannels * nnXLen * nnYLen);
    std::vector<float> trunkScratchData(batchSize * normActConv2->outChannels * nnXLen * nnYLen);
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, trunk, sizeof(float) * trunkData.size(), trunkData.data(), true, &res);
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, trunkScratch, sizeof(float) * trunkScratchData.size(), trunkScratchData.data(), true, &res);
    CHECK_VK_MSG("Copy trunk scratch buffer to host for ResidualBlock: " + name, res);
    printFloatBuffer(name + " Pointwise trunk : ", trunkData.data(), trunkData.size(), batchSize, normActConv2->outChannels, nnYLen, nnXLen);
    printFloatBuffer(name + " Pointwise trunkScratch: ", trunkScratchData.data(), trunkScratchData.size(), batchSize, normActConv2->outChannels, nnYLen, nnXLen);

    VkHelpers::submitCommandBuffers(handle->vulkanDevice, { pointWiseAddCommandBuffer });
    vkQueueWaitIdle(handle->vulkanDevice->queue);
    // print pointwise result.
    size_t totalSize = static_cast<size_t>(batchSize) * static_cast<size_t>(normActConv2->outChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen);
    std::vector<float> valueData(totalSize);
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, trunkScratch, sizeof(float) * valueData.size(), valueData.data(), true, &res);
    CHECK_VK_MSG("Copy trunk buffer to host for ResidualBlock: " + name, res);
    printFloatBuffer(name + " Pointwise output: ", valueData.data(), valueData.size(), batchSize, normActConv2->outChannels, nnYLen, nnXLen);
    return commandBuffers;
  }
};

struct GlobalPoolingResidualBlock {
  ComputeHandleInternal *handle;
  const std::string name;
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
    name(desc->name),
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
    if ( !commandBuffers.empty() ) {
      return;
    }

    SizedBuf<VulkanBuffer*> regularOut(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(gpoolChannels * 3));
    SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(regularChannels));

    preBN->record(batchSize, trunk, mask, trunkScratch);
    regularConv->record(batchSize, trunkScratch, regularOut.buf);
    gpoolConv->record(batchSize, trunkScratch, gpoolOut.buf);
    gpoolBN->record(batchSize, gpoolOut. buf,mask, gpoolOut.buf);
    VkResult res;;
    VkCommandBuffer gpoolCB = performGpoolMask(handle, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, batchSize, gpoolChannels, nnXYLen, &res);
    CHECK_VK_MSG("Record GlobalPoolingResidualBlock gpool mask", res);
    gpoolToBiasMul->record(batchSize, gpoolConcat.buf, gpoolBias.buf);
    VkCommandBuffer addChannelCB = performAddChannelBiases(handle, regularOut.buf, gpoolBias.buf, batchSize * regularChannels, nnXYLen);
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
      throw StringError("GlobalPoolingResidualBlock command buffers not recorded yet");
    }
    SizedBuf<VulkanBuffer*> regularOut(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(gpoolChannels * 3));
    SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(regularChannels));

    preBN->apply(batchSize, trunk, mask, trunkScratch);
    regularConv->apply(batchSize, trunkScratch, regularOut.buf);
    gpoolConv->apply(batchSize, trunkScratch, gpoolOut.buf);
    gpoolBN->apply(batchSize, gpoolOut.buf, mask, gpoolOut.buf);

    VkHelpers::submitCommandBuffers(handle->vulkanDevice, { commandBuffers[4] }); // gpoolCB
    std::vector<float> gpoolConcatData(batchSize * gpoolChannels * 3);
    VkResult res = VK_ERROR_UNKNOWN;
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, gpoolConcat.buf, sizeof(float) * gpoolConcatData.size(), gpoolConcatData.data(), true, &res);
    CHECK_VK_MSG("Copy gpoolConcat buffer to host for GlobalPoolingResidualBlock", res);
    printFloatBuffer(name + " GpoolConcat output: ", gpoolConcatData.data(), gpoolConcatData.size() * sizeof(float), batchSize, gpoolChannels * 3, 1, 1);

    gpoolToBiasMul->apply(batchSize, gpoolConcat.buf, gpoolBias.buf);
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, { commandBuffers[6] }); // addChannelCB
    CHECK_VK_MSG("Submit addChannelCB for GlobalPoolingResidualBlock", res);
    std::vector<float> gpoolBiasData(batchSize * regularChannels);
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, gpoolBias.buf, sizeof(float) * gpoolBiasData.size(), gpoolBiasData.data(), true, &res);
    CHECK_VK_MSG("Copy gpoolBias buffer to host for GlobalPoolingResidualBlock]", res);
    printFloatBuffer(name + " GpoolBias output: ", gpoolBiasData.data(), gpoolBiasData.size() * sizeof(float), batchSize, regularChannels, 1, 1);
    normActConv2->apply(batchSize, regularOut.buf, regularOut.buf, trunkScratch, mask);
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, { commandBuffers[commandBuffers.size() - 1] }); // addPointWiseCB
    std::vector<float> trunkData(batchSize * normActConv2->outChannels * nnXLen * nnYLen);
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, trunkScratch, sizeof(float) * trunkData.size(), trunkData.data(), true, &res);
    CHECK_VK_MSG("Copy trunk buffer to host for GlobalPoolingResidualBlock", res);
    printFloatBuffer(name + " AddPointwise output: ", trunkData.data(), trunkData.size() * sizeof(float), batchSize, normActConv2->outChannels, nnYLen, nnXLen);
    return commandBuffers;
  }
};


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
    if ( !commandBuffers.empty() ) {
      return;
    }
    SizedBuf<VulkanBuffer*> mid1(scratch->allocator, scratch->getBufSizeXY(normActConv->outChannels));
    SizedBuf<VulkanBuffer*> mid2(scratch->allocator, scratch->getBufSizeXY(normActConv->outChannels));
    normActConv->record(batchSize, trunk, trunkScratch, mid1.buf , mask);
    blocks->record(batchSize, scratch, mid1.buf, mid2.buf, mask, maskSum);
    normActConv2->record(batchSize, mid2.buf, mid2.buf, trunkScratch, mask);
    VkCommandBuffer pointWiseCB = performAddPointWise(handle, trunk, trunkScratch, static_cast<int>(batchSize * normActConv2->outChannels * nnXLen * nnYLen));
    commandBuffers.push_back( normActConv->commandBuffers[0] );
    commandBuffers.insert(commandBuffers.end(), blocks->commandBuffers.begin(), blocks->commandBuffers.end());
    commandBuffers.insert(commandBuffers.end(), normActConv2->commandBuffers.begin(), normActConv2->commandBuffers.end());
    commandBuffers.push_back( pointWiseCB );
  }

  /**
   * @brief execute NestedResidualBlock, only for debug now
   */
  void apply(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum
  ) {
    assert( !commandBuffers.empty() );
    VkHelpers::submitCommandBuffers(
      handle->vulkanDevice,
      commandBuffers
    );
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

BlockStack::~BlockStack() {
  // unique_ptr will clean up automatically
}

void BlockStack::record(
  int batchSize,
  ScratchBuffers *scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* trunkScratch,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum
) {
  auto logger = handle->context->logger;
  for ( int i = 0 ; i < numBlocks ; ++i ) {
    int blockType = blocks[i].first;
    // logger->write("Recording BlockStack - ResidualBlock index: " + Global::intToString(i));
    if ( blockType == ORDINARY_BLOCK_KIND ) {
      ResidualBlock* blockPtr = static_cast<ResidualBlock*>(blocks[i].second.get());
      if(blockPtr == nullptr) {
        Global::fatalError("BlockStack::record: ResidualBlock pointer is null at index " + Global::intToString(i));
      }
      blockPtr->record(batchSize, scratch, trunk, trunkScratch, mask);
      commandBuffers.insert(commandBuffers.end(), blockPtr->commandBuffers.begin(), blockPtr->commandBuffers.end());
    } else if ( blockType == GLOBAL_POOLING_BLOCK_KIND ) {
      GlobalPoolingResidualBlock* blockPtr = static_cast<GlobalPoolingResidualBlock*>(blocks[i].second.get());
      if(blockPtr == nullptr) {
        Global::fatalError("BlockStack::record: GlobalPoolingResidualBlock pointer is null at index " + Global::intToString(i));
      }
      blockPtr->record(batchSize, scratch, trunk, trunkScratch, mask, maskSum);
      commandBuffers.insert(commandBuffers.end(), blockPtr->commandBuffers.begin(), blockPtr->commandBuffers.end());
    } else if ( blockType == NESTED_BOTTLENECK_BLOCK_KIND ) {
      NestedResidualBlock* blockPtr = static_cast<NestedResidualBlock*>(blocks[i].second.get());
      if(blockPtr == nullptr) {
        Global::fatalError("BlockStack::record: NestedResidualBlock pointer is null at index " + Global::intToString(i));
      }
      blockPtr->record(batchSize, scratch, trunk, trunkScratch, mask, maskSum);
      commandBuffers.insert(commandBuffers.end(), blockPtr->commandBuffers.begin(), blockPtr->commandBuffers.end());
    } else {
      ASSERT_UNREACHABLE;
    }
  }
}

void BlockStack::apply(
  int batchSize,
  ScratchBuffers *scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* trunkScratch,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum
) {
  if ( commandBuffers.empty() ) {
    Global::fatalError("BlockStack apply called before record");
  }

  for (int i = 0 ; i< numBlocks ; ++i ) {
    int blockType = blocks[i].first;
    if ( blockType == ORDINARY_BLOCK_KIND ) {
      ResidualBlock* blockPtr = static_cast<ResidualBlock*>(blocks[i].second.get());
      if(blockPtr == nullptr) {
        Global::fatalError("BlockStack::apply: ResidualBlock pointer is null at index " + Global::intToString(i));
      }
      blockPtr->apply(batchSize, scratch, trunk, trunkScratch, mask);
    } else if ( blockType == GLOBAL_POOLING_BLOCK_KIND ) {
      GlobalPoolingResidualBlock* blockPtr = static_cast<GlobalPoolingResidualBlock*>(blocks[i].second.get());
      if(blockPtr == nullptr) {
        Global::fatalError("BlockStack::apply: GlobalPoolingResidualBlock pointer is null at index " + Global::intToString(i));
      }
      blockPtr->apply(batchSize, scratch, trunk, trunkScratch, mask, maskSum);
    } else if ( blockType == NESTED_BOTTLENECK_BLOCK_KIND ) {
      NestedResidualBlock* blockPtr = static_cast<NestedResidualBlock*>(blocks[i].second.get());
      if(blockPtr == nullptr) {
        Global::fatalError("BlockStack::apply: NestedResidualBlock pointer is null at index " + Global::intToString(i));
      }
      blockPtr->apply(batchSize, scratch, trunk, trunkScratch, mask, maskSum);
    } else {
      ASSERT_UNREACHABLE;
    }
  }
}


struct SGFMetadataEncoder {
  ComputeHandleInternal *handle;
  const std::string name;
  MatmulLayer *matmul1;
  MatBiasLayer *matBias1;
  MatmulLayer *matmul2;
  MatBiasLayer *matBias2;
  MatmulLayer *matmul3;

  std::vector<VkCommandBuffer> commandBuffers;

  SGFMetadataEncoder(
    ComputeHandleInternal *handle_,
    const SGFMetadataEncoderDesc* desc
  ): 
    handle(handle_),
    name(desc->name)
  {
    matmul1 = new MatmulLayer(handle, &desc->mul1);
    matBias1 = new MatBiasLayer(handle, &desc->bias1, desc->act1.activation);
    matmul2 = new MatmulLayer(handle, &desc->mul2);
    matBias2 = new MatBiasLayer(handle, &desc->bias2, desc->act2.activation);
    matmul3 = new MatmulLayer(handle, &desc->mul3);
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
    assert( !commandBuffers.empty() );
    VkHelpers::submitCommandBuffers(
      handle->vulkanDevice,
      commandBuffers
    );
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
    blockStack( handle, desc->blocks, desc->numBlocks, trunkNumChannels, nnXLen_, nnYLen_) 
  {
    checkBufferSize(maxBatchSize_,nnXLen_,nnYLen_,trunkNumChannels);
    checkBufferSize(maxBatchSize_,nnXLen_,nnYLen_,midNumChannels);
    checkBufferSize(maxBatchSize_,nnXLen_,nnYLen_,regularNumChannels);
    checkBufferSize(maxBatchSize_,nnXLen_,nnYLen_,gpoolNumChannels);

    initialConv = std::make_unique<ConvLayer>(handle, &desc->initialConv, nnXLen, nnYLen);
    initialMatmul = std::make_unique<MatmulLayer>(handle, &desc->initialMatMul);
    if ( desc->metaEncoderVersion >0) {
      sgfMetadataEncoder = std::make_unique<SGFMetadataEncoder>(handle, &desc->sgfMetadataEncoder);
      testAssert(sgfMetadataEncoder->matmul3->outChannels == initialMatmul->outChannels);
    }  
    trunkTipBN = std::make_unique<BatchNormLayer>(handle, &desc->trunkTipBN, &desc->trunkTipActivation, nnXLen, nnYLen);
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
    trunkTipBN->record(batchSize, trunk, mask, trunk);

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
    assert( !commandBuffers.empty() );
    SizedBuf<VulkanBuffer*> trunkScratch( scratch->allocator, scratch->getBufSizeXY(trunkNumChannels) );
    VkResult res = VK_ERROR_UNKNOWN;

    initialConv->apply(batchSize, input, trunk);
    initialMatmul->apply(batchSize, inputGlobal, trunkScratch.buf);

    {
      VkHelpers::submitCommandBuffers(handle->vulkanDevice, { commandBuffers[2] }); // addChannelBiasCB
      std::vector<float> retVec(batchSize * trunkNumChannels * nnXLen * nnYLen);
      VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, trunk, sizeof(float) * retVec.size(), retVec.data(), true, &res);
      CHECK_VK_MSG("Copy trunk buffer to host after initial addChannelBiases", res);
      std::cout << "Trunk data size: " << retVec.size() << std::endl;
      printFloatBuffer(name + " After initial addChannelBiases Output:", retVec.data(), retVec.size(), batchSize, trunkNumChannels, nnYLen, nnXLen);
    }

    if ( sgfMetadataEncoder != nullptr ) {
      SizedBuf<VulkanBuffer*> sgfEncodedMeta(scratch->allocator, scratch->getBufSizeFloat(sgfMetadataEncoder->matmul3->outChannels));
      sgfMetadataEncoder->apply(batchSize, scratch, inputMeta, sgfEncodedMeta.buf);
      VkHelpers::submitCommandBuffers(handle->vulkanDevice, { commandBuffers[4 + sgfMetadataEncoder->commandBuffers.size()] }); // addChannelBiasCB2
      std::vector<float> retVec(batchSize * trunkNumChannels * nnXLen * nnYLen);
      VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, trunk, sizeof(float) * retVec.size(), retVec.data(), true, &res);
      CHECK_VK_MSG("Copy trunk buffer to host after SGF metadata addChannelBiases", res);
      printFloatBuffer(name + " After SGF metadata addChannelBiases: ", retVec.data(), retVec.size(), batchSize, trunkNumChannels, nnYLen, nnXLen);
    }

    blockStack.apply(batchSize, scratch, trunk, trunkScratch.buf, mask, maskSum);

    // print current trunk 
    {
      std::vector<float> retVec(batchSize * trunkNumChannels * nnXLen * nnYLen);
      VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, trunk, sizeof(float) * retVec.size(), retVec.data(), true, &res);
      CHECK_VK_MSG("Copy trunk buffer to host before trunk tip BN", res);
      printFloatBuffer(name + " Before trunk tip BN: ", retVec.data(), retVec.size(), batchSize, trunkNumChannels, nnYLen, nnXLen);
    }
    trunkTipBN->apply(batchSize, trunk, mask, trunk);

    // VkHelpers::submitCommandBuffers(
      // handle->vulkanDevice,
      // commandBuffers
    // );
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
  VkCommandBuffer gpoolCB, addChannelBiasCB;

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

    SizedBuf<VulkanBuffer*> p1Out(scratch->allocator, scratch->getBufSizeXY(p1Channels));
    SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(g1Channels * 3));
    SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(p1Channels));
    SizedBuf<VulkanBuffer*> p1Pass(scratch->allocator, scratch->getBufSizeFloat(p1Channels));

    p1Conv->record(batchSize, trunk, p1Out.buf);
    g1Conv->record(batchSize, trunk, gpoolOut.buf);
    g1BN->record(batchSize, gpoolOut.buf, mask, gpoolOut.buf);
    VkResult res;;
    gpoolCB = performGpoolMask(handle,gpoolOut.buf, gpoolConcat.buf, mask, maskSum, batchSize, g1Channels, nnXLen * nnYLen, &res);
    CHECK_VK_MSG("Record PolicyHead gpool mask", res);
    gpoolToBiasMul->record(batchSize, gpoolConcat.buf, gpoolBias.buf);
    addChannelBiasCB = performAddChannelBiases(handle, p1Out.buf, gpoolBias.buf, p1Channels * batchSize, nnXLen * nnYLen);
    p1BN->record(batchSize, p1Out.buf, mask, p1Out.buf);
    p2Conv->record(batchSize, p1Out.buf, policy);

    if ( modelVersion >= 15 ) {
      gpoolToPassMul->record(batchSize, gpoolConcat.buf, p1Pass.buf);
      gpoolToPassBias->record(batchSize, p1Pass.buf);
      gpoolToPassMul2->record(batchSize, p1Pass.buf, policyPass);
    } else {
      gpoolToPassMul->record(batchSize, gpoolConcat.buf, policyPass);
    }

    commandBuffers.push_back( p1Conv->commandBuffer );
    commandBuffers.push_back( g1Conv->commandBuffer );
    commandBuffers.push_back( g1BN->commandBuffer );
    commandBuffers.push_back( gpoolCB );
    commandBuffers.push_back( gpoolToBiasMul->commandBuffer );
    commandBuffers.push_back( addChannelBiasCB );
    commandBuffers.push_back( p1BN->commandBuffer );
    commandBuffers.push_back( p2Conv->commandBuffer );
    commandBuffers.push_back( gpoolToPassMul->commandBuffer );
    if ( modelVersion >= 15 ) {
      commandBuffers.push_back( gpoolToPassBias->commandBuffer );
      commandBuffers.push_back( gpoolToPassMul2->commandBuffer );
    }  
  }

  /*
  * @brief Execute inference on the policy head. Only for debug now
  */
  void apply(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* policyPass,
    VulkanBuffer* policy
  ) {
    assert( !commandBuffers.empty() );
    SizedBuf<VulkanBuffer*> p1Out(scratch->allocator, scratch->getBufSizeXY(p1Channels));
    SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(g1Channels * 3));
    SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(p1Channels));
    SizedBuf<VulkanBuffer*> p1Pass(scratch->allocator, scratch->getBufSizeFloat(p1Channels));

    p1Conv->apply(batchSize, trunk, p1Out.buf);
    g1Conv->apply(batchSize, trunk, gpoolOut.buf);
    g1BN->apply(batchSize, gpoolOut.buf, mask, gpoolOut.buf);
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, { gpoolCB }); // gpoolCB
    gpoolToBiasMul->apply(batchSize, gpoolConcat.buf, gpoolBias.buf);
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, { addChannelBiasCB }); // addChannelBiasCB
    p1BN->apply(batchSize, p1Out.buf, mask, p1Out.buf);
    p2Conv->apply(batchSize, p1Out.buf, policy);
    if ( modelVersion >= 15 ) {
      gpoolToPassMul->apply(batchSize, gpoolConcat.buf, p1Pass.buf);
      gpoolToPassBias->apply(batchSize, p1Pass.buf);
      gpoolToPassMul2->apply(batchSize, p1Pass.buf, policyPass);
    } else {
      gpoolToPassMul->apply(batchSize, gpoolConcat.buf, policyPass);
    }
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
  VkCommandBuffer gpoolCB;

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
    if (  !commandBuffers.empty() ) {
      return;
    }

    SizedBuf<VulkanBuffer*> v1Out(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<VulkanBuffer*> v1Mean(scratch->allocator, scratch->getBufSizeFloat(v1Channels*3));
    SizedBuf<VulkanBuffer*> v2Out(scratch->allocator, scratch->getBufSizeFloat(v2Channels));

    v1Conv->record(batchSize, trunk, v1Out.buf);
    v1BN->record(batchSize, v1Out.buf, mask, v1Out.buf);
    VkResult res;;
    gpoolCB = performValueHeadPool(handle, v1Out.buf, v1Mean.buf, maskSum, batchSize, v1Channels, nnXLen * nnYLen);

    v2Mul->record(batchSize, v1Mean.buf, v2Out.buf);
    v2Bias->record(batchSize, v2Out.buf);
    v3Mul->record(batchSize, v2Out.buf, value);
    v3Bias->record(batchSize, value);

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

  /**
   * @brief Execute inference on the value head. Only for debug now
   */
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
    assert( !commandBuffers.empty() );
    SizedBuf<VulkanBuffer*> v1Out(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<VulkanBuffer*> v1Mean(scratch->allocator, scratch->getBufSizeFloat(v1Channels*3));
    SizedBuf<VulkanBuffer*> v2Out(scratch->allocator, scratch->getBufSizeFloat(v2Channels));

    v1Conv->apply(batchSize, trunk, v1Out.buf);
    v1BN->apply(batchSize, v1Out.buf, mask, v1Out.buf);
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, { gpoolCB }); // gpoolCB
    // {
    //   std::vector<float> retVec(batchSize * v1Channels * 3);
    //   VkResult res;
    //   VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, v1Mean.buf, sizeof(float) * retVec.size(), retVec.data(), true, &res);
    //   CHECK_VK_MSG("Copy v1Mean buffer to host after value head gpool", res);
    //   printFloatBuffer(name + " Value Head v1Mean after gpool: ", retVec.data(), retVec.size(), batchSize, v1Channels, 1, 3);
    // }
    v2Mul->apply(batchSize, v1Mean.buf, v2Out.buf);
    v2Bias->apply(batchSize, v2Out.buf);
    v3Mul->apply(batchSize, v2Out.buf, value);
    v3Bias->apply(batchSize, value);
    sv3Mul->apply(batchSize, v2Out.buf, scoreValue);
    sv3Bias->apply(batchSize, scoreValue);
    vOwnershipConv->apply(batchSize, v1Out.buf, ownership);
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

/**
 * @brief Record command buffer to compute mask sums
 * @param handle Compute handle
 * @param batchSize Batch size
 * @param nnXLen Neural net X length
 * @param nnYLen Neural net Y length
 * @param mask Input mask buffer
 * @param maskSum Output mask sum buffer
 */
VkCommandBuffer computeMaskSums(
  ComputeHandleInternal *handle,
  int batchSize,
  int nnXLen,
  int nnYLen,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum
) {
  static constexpr int nKernelDims = 3;
  // SumChannels shader uses numthreads(SUM_CHANNELS_DISPATCH_X=64, 1, 1)
  // Dispatch: X=1 (single workgroup for reduction), Y=numChannels, Z=batchSize
  // Each workgroup processes one (batch, channel) pair
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

  int numChannels = 1;
  int nnXYLen = nnXLen * nnYLen;
  // Determine GPU and select appropriate compute pipeline
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  KatagoVulkan::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  KatagoVulkan::Pipeline targetPipeline = pipelines->sumChannelsFp32;

  VkResult res = VK_SUCCESS;
  VkDescriptorSet descriptorSet = VkHelpers::allocateDescriptorSet(handle->vulkanDevice, targetPipeline.descriptorSetLayout, &res);
  CHECK_VK_MSG("Allocate compute mask sum descriptor set", res);
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, mask),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, maskSum)
  };
  VkHelpers::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

  commandBuffer = VkHelpers::allocateCommandBuffer(handle->vulkanDevice);
  res = VkHelpers::beginCommandBuffer(commandBuffer);
  CHECK_VK_MSG("Begin compute mask sum command buffer", res);
  VkHelpers::barrierCommandBuffer(commandBuffer);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.layout, 0, 1, &descriptorSet, 0, nullptr);
  KatagoVulkan::SumChannelsParams pushConstants;
  pushConstants.batchSize = batchSize;
  pushConstants.numChannels = numChannels;
  pushConstants.nnXYLen = nnXYLen;
  vkCmdPushConstants(commandBuffer, targetPipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KatagoVulkan::SumChannelsParams), &pushConstants);
  // Dispatch: X=1 (reduction), Y=numChannels(=1 for mask sum), Z=batchSize
  vkCmdDispatch(commandBuffer, 1u, static_cast<uint32_t>(numChannels), static_cast<uint32_t>(batchSize));
  VkHelpers::endCommandBuffer(commandBuffer);
  return commandBuffer;
}

/**
 * @brief Model structure containing trunk and heads
 */
struct Model {
  std::string modelName;
  ComputeHandleInternal *handle;
  const int modelVersion;
  const int maxBatchSize;
  const int numInputChannels;
  const int numInputGlobalChannels;
  const int numInputMetaChannels;
  const int numPolicyChannels;
  const int numValueChannels;
  const int numScoreValueChannels;
  const int numOwnershipChannels;
  const int nnXLen;
  const int nnYLen;

  std::unique_ptr<Trunk> trunk;
  std::unique_ptr<PolicyHead> policyHead;
  std::unique_ptr<ValueHead> valueHead;
  std::vector<VkCommandBuffer> commandBuffers;

  VkFence fence = VK_NULL_HANDLE;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(
    ComputeHandleInternal *handle_,
    const ModelDesc& desc,
    int maxBatchSize_,
    int nnXLen_,
    int nnYLen_
  ): 
    handle(handle_),
    modelName(desc.name),
    modelVersion(desc.modelVersion),
    maxBatchSize(maxBatchSize_),
    numInputChannels(desc.numInputChannels),
    numInputGlobalChannels(desc.numInputGlobalChannels),
    numInputMetaChannels(desc.numInputMetaChannels),
    numPolicyChannels(desc.numPolicyChannels),
    numValueChannels(desc.numValueChannels),
    numScoreValueChannels(desc.numScoreValueChannels),
    numOwnershipChannels(desc.numOwnershipChannels),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_)
  {

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

    int numFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);

    if ( numFeatures != numInputChannels ) {
      throw StringError(
        Global::strprintf("Model version %d expects %d input channels but model provides %d", modelVersion, numFeatures, numInputChannels)
      );
    }

    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
    if ( numGlobalFeatures != numInputGlobalChannels ) {
      throw StringError(
        Global::strprintf("Model version %d expects %d global input channels but model provides %d", modelVersion, numGlobalFeatures, numInputGlobalChannels)
      );
    }

    if ( numInputMetaChannels > 0 && numInputMetaChannels != SGFMetadata::METADATA_INPUT_NUM_CHANNELS ) {
      throw StringError(
        Global::strprintf("Model version %d expects %d metadata input channels but model provides %d", modelVersion, SGFMetadata::METADATA_INPUT_NUM_CHANNELS, numInputMetaChannels)
      );
    }

    // TODO: Check required workspaces sizes
    // TODO: Check partial models constructor parameters
    trunk = std::make_unique<Trunk>(handle, &desc.trunk, maxBatchSize, nnXLen, nnYLen);
    policyHead = std::make_unique<PolicyHead>(handle, &desc.policyHead, nnXLen, nnYLen);
    valueHead = std::make_unique<ValueHead>(handle, &desc.valueHead, nnXLen, nnYLen);

    VkResult res = VK_SUCCESS;
    fence = VkHelpers::createFence(handle->vulkanDevice, &res);
    CHECK_VK_MSG("Create model fence", res);
  }

  ~Model() {
    if( fence != VK_NULL_HANDLE ) {
      VkHelpers::destroyFence(handle->vulkanDevice, fence);
      fence = VK_NULL_HANDLE; 
    }
  }

  void record(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* input,
    VulkanBuffer* inputGlobal,
    VulkanBuffer* inputMeta,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* trunkBuf,
    VulkanBuffer* policyPass,
    VulkanBuffer* policy,
    VulkanBuffer* value,
    VulkanBuffer* scoreValue,
    VulkanBuffer* ownership
  ) {
    if ( !commandBuffers.empty() ) {
      return;
    }

    VkCommandBuffer extractChannel0CB = performExtractChannel0NCHW(handle, input, mask, batchSize, numInputChannels, nnXLen * nnYLen);
    VkCommandBuffer computeMaskSumCB = computeMaskSums(handle, batchSize, nnXLen, nnYLen, mask, maskSum);
    trunk->record(batchSize, scratch, input, inputGlobal, inputMeta, trunkBuf,  mask, maskSum);
    policyHead->record(batchSize, scratch, trunkBuf, mask, maskSum, policyPass, policy);
    valueHead->record(batchSize, scratch, trunkBuf, mask, maskSum, value, scoreValue, ownership); 
    commandBuffers.push_back( extractChannel0CB );
    commandBuffers.push_back( computeMaskSumCB );
    commandBuffers.insert(commandBuffers.end(), trunk->commandBuffers.begin(), trunk->commandBuffers.end());
    commandBuffers.insert(commandBuffers.end(), policyHead->commandBuffers.begin(), policyHead->commandBuffers.end());
    commandBuffers.insert(commandBuffers.end(), valueHead->commandBuffers.begin(), valueHead ->commandBuffers.end()); 
  }

  /**
   * @brief execute the model for debug purposes
   */
  void debug(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* input,
    VulkanBuffer* inputGlobal,
    VulkanBuffer* inputMeta,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* trunkBuf,
    VulkanBuffer* policyPass,
    VulkanBuffer* policy,
    VulkanBuffer* value,
    VulkanBuffer* scoreValue,
    VulkanBuffer* ownership
  ) {
    assert( !commandBuffers.empty() );
    VkResult res = VK_ERROR_UNKNOWN;
    const VulkanDevice* device = handle->vulkanDevice;
    // For debug purposes, just run extract channel 0 compute shader
    {
      VkHelpers::submitCommandBuffers(handle->vulkanDevice, { commandBuffers[0] });
      vkQueueWaitIdle(handle->vulkanDevice->queue);
      std::vector<float> retVec(batchSize * nnXLen * nnYLen);
      // VkHelpers::copyDeviceBufferToHost(device, mask, sizeof(float) * batchSize * nnXLen * nnYLen, retVec.data(), true, &res);  
      // CHECK_VK_MSG("Copy extract channel 0 result to host", res);
      // printFloatBuffer("Model::debug Extract Channel 0 Result", retVec.data(), batchSize * nnXLen * nnYLen, batchSize, 1, nnYLen, nnXLen);
    }

    // compute mask sums
    {
      VkHelpers::submitCommandBuffers(handle->vulkanDevice, { commandBuffers[1] });
      vkQueueWaitIdle(handle->vulkanDevice->queue);
      std::vector<float> retVec(batchSize);
      VkHelpers::copyDeviceBufferToHost(device, maskSum, sizeof(float) * batchSize, retVec.data(), true, &res);  
      // CHECK_VK_MSG("Copy mask sum result to host", res);
      printFloatBuffer("Model::debug Mask Sum Result", retVec.data(), batchSize, batchSize, 1, 1, 1);
    }

    // trunk
    {
      trunk->apply(batchSize, scratch, input, inputGlobal, inputMeta, trunkBuf, mask, maskSum);
      vkQueueWaitIdle(handle->vulkanDevice->queue);

      std::vector<float> retVec(batchSize * trunk->trunkNumChannels * nnXLen * nnYLen);
      VkHelpers::copyDeviceBufferToHost(device, trunkBuf, sizeof(float) * batchSize * trunk->trunkNumChannels * nnXLen * nnYLen, retVec.data(), true, &res);  
      // CHECK_VK_MSG("Copy trunk result to host", res);
      printFloatBuffer("Model::debug Trunk Output", retVec.data(), batchSize, trunk->trunkNumChannels, nnXLen, nnYLen, trunk->trunkNumChannels);
    }

    // Policy
    {
      policyHead->apply(batchSize, scratch, trunkBuf, mask, maskSum, policyPass, policy);
      vkQueueWaitIdle(handle->vulkanDevice->queue);
      std::vector<float> retVec(batchSize * policyHead->p2Channels * nnXLen * nnYLen);
      VkHelpers::copyDeviceBufferToHost(device, policy, sizeof(float) * batchSize * policyHead->p2Channels * nnXLen * nnYLen, retVec.data(), true, &res);  
      // CHECK_VK_MSG("Copy policy result to host", res);
      // printFloatBuffer("Model::debug Policy Result", retVec.data(), batchSize, policyHead->p2Channels * nnXLen * nnYLen, batchSize, policyHead->p2Channels, nnYLen, nnXLen); 
      printFloatBuffer("Model::debug Policy Output", retVec.data(), batchSize, policyHead->p2Channels, nnXLen, nnYLen, policyHead->p2Channels);
    }

    // Value
    {
      valueHead->apply(batchSize, scratch, trunkBuf, mask, maskSum, value, scoreValue, ownership);
      vkQueueWaitIdle(handle->vulkanDevice->queue);
      {
        std::vector<float> retVec(batchSize * valueHead->valueChannels);
        VkHelpers::copyDeviceBufferToHost(device, value, sizeof(float) * batchSize * valueHead->valueChannels, retVec.data(), true, &res);  
        // CHECK_VK_MSG("Copy value result to host", res);
        printFloatBuffer("Model::debug Value Output", retVec.data(), batchSize, valueHead->valueChannels, 1, 1, valueHead->valueChannels);
      }
      {
        std::vector<float> retVec(batchSize * valueHead->scoreValueChannels);
        VkHelpers::copyDeviceBufferToHost(device, scoreValue, sizeof(float) * batchSize * valueHead->scoreValueChannels, retVec.data(), true, &res);  
        // CHECK_VK_MSG("Copy score value result to host", res);
        printFloatBuffer("Model::debug Score Value Output", retVec.data(), batchSize, valueHead->scoreValueChannels, 1, 1, valueHead->scoreValueChannels);
      }
      {
        std::vector<float> retVec(batchSize * valueHead->ownershipChannels * nnXLen * nnYLen);
        VkHelpers::copyDeviceBufferToHost(device, ownership, sizeof(float) * batchSize * valueHead->ownershipChannels * nnXLen * nnYLen, retVec.data(), true, &res);  
        // CHECK_VK_MSG("Copy ownership result to host", res);
        printFloatBuffer("Model::debug Ownership Output", retVec.data(), batchSize, valueHead->ownershipChannels, nnXLen, nnYLen, valueHead->ownershipChannels);
      }
    }

    // exit(EXIT_SUCCESS);
  }

  /**
   * run all command buffers to perform inference.
   */
  void apply() {
    assert( !commandBuffers.empty() );
    VkHelpers::resetFence(handle->vulkanDevice, fence);
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, commandBuffers, fence);
    vkWaitForFences(handle->device, 1, &fence, VK_TRUE, UINT32_MAX);
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
  logger->write("Create Vulkan Compute Context with GPUs: ");
  for(size_t i = 0; i<gpuIdxs.size(); i++) {
    logger->write("  GPU Index " + Global::intToString(gpuIdxs[i]));
  }

  return new ComputeContext(
    nnXLen,
    nnYLen,
    useFP16Mode,
    useNHWCMode,
    std::vector<uint32_t>(gpuIdxs.begin(), gpuIdxs.end()),
    logger
  );
}

void NeuralNet::freeComputeContext(ComputeContext* context) {
  delete context;
}

static ComputeContext* createComputeContextForTesting(
  const std::vector<int>& gpuIdxs,
  Logger *logger,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC
) {
  // std::cout << "[createComputeContextForTesting] create Compute Context with GPUs: ";
  return new ComputeContext(
    nnXLen,
    nnYLen,
    useFP16 ? enabled_t::True : enabled_t::False,
    useNHWC ? enabled_t::True : enabled_t::False,
    std::vector<uint32_t>(gpuIdxs.begin(), gpuIdxs.end()),
    logger
  );
}

/* ########################### Buffers ######################### */
/**
 * @brief All buffers used in Vulkan Backend
 */
struct Buffers {
  VulkanBuffer* input;
  VulkanBuffer* inputGlobal;  
  VulkanBuffer* inputMeta;
  size_t inputElts;
  size_t inputGlobalElts;
  size_t inputMetaElts;

  VulkanBuffer* outputBuffer; 
  VulkanBuffer* mask;
  VulkanBuffer* maskSum;
  VulkanBuffer* trunk;
  VulkanBuffer* policyPass;
  VulkanBuffer* policy;
  size_t policyPassElts;
  size_t policyElts;

  VulkanBuffer* value;
  size_t valueElts;
  VulkanBuffer* scoreValue;
  size_t scoreValueElts;
  VulkanBuffer* ownership;
  size_t ownershipElts;

  VulkanBuffer* convWorkspace;
  VulkanBuffer* convWorkspace2;

  /**
   * @brief Buffers Constructor
   * @param handle Compute handle internal
   * @param m Model
   */
  Buffers(
    ComputeHandleInternal* handle,
    const Model& m
  ) {
    size_t batchXYElts = (size_t)m.maxBatchSize * m.nnXLen * m.nnYLen;
    size_t batchElts = (size_t)m.maxBatchSize;

    bool useFP16 = handle->usingFP16Storage;

    inputElts = m.numInputChannels * batchXYElts;
    inputGlobalElts = m.numInputGlobalChannels * batchElts;
    inputMetaElts = m.numInputMetaChannels * batchElts;

    // TODO:  Modify this when fp16 input is supported.
    const size_t dtypeSize = sizeof(float);
    // input = createReadWriteBuffer(handle, inputElts, useFP16);
    VkResult res = VK_SUCCESS;
    input = VkHelpers::createDeviceBuffer(handle->vulkanDevice, dtypeSize * inputElts, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create input buffer", res);
    inputGlobal = VkHelpers::createDeviceBuffer(handle->vulkanDevice, dtypeSize * inputGlobalElts, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create input global buffer", res);
    if(m.numInputMetaChannels > 0) {
      inputMeta = VkHelpers::createDeviceBuffer(handle->vulkanDevice, dtypeSize * inputMetaElts, false, &res);
    }
    else {
      inputMeta = NULL;
    }

    mask = VkHelpers::createDeviceBuffer(handle->vulkanDevice, batchXYElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create mask buffer", res);
    maskSum = VkHelpers::createDeviceBuffer(handle->vulkanDevice, batchElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create mask sum buffer", res);
    trunk = VkHelpers::createDeviceBuffer(handle->vulkanDevice, m.trunk->trunkNumChannels * batchXYElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create trunk buffer", res);

    if(m.modelVersion >= 16)
      testAssert(m.policyHead->p2Channels == 4);
    else if(m.modelVersion >= 12)
      testAssert(m.policyHead->p2Channels == 2);
    else
      testAssert(m.policyHead->p2Channels == 1);

    policyPassElts = m.policyHead->p2Channels * batchElts;
    policyPass = VkHelpers::createDeviceBuffer(handle->vulkanDevice, policyPassElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create policy pass buffer", res);
    policyElts = m.policyHead->p2Channels * batchXYElts;
    policy = VkHelpers::createDeviceBuffer(handle->vulkanDevice, policyElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create policy buffer", res);

    valueElts = m.valueHead->valueChannels * batchElts;
    value = VkHelpers::createDeviceBuffer(handle->vulkanDevice, valueElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create value buffer", res);

    scoreValueElts = m.valueHead->scoreValueChannels * batchElts;
    scoreValue = VkHelpers::createDeviceBuffer(handle->vulkanDevice, scoreValueElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create score value buffer", res);

    ownershipElts = m.valueHead->ownershipChannels * batchXYElts;
    ownership = VkHelpers::createDeviceBuffer(handle->vulkanDevice, ownershipElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create ownership buffer", res);

    // TODO: Implement workspace allocation when winograd or other conv algorithms are added.
    // ConvWorkspaceEltsNeeded convWorkspaceElts = m.requiredConvWorkspaceElts(handle);
    // convWorkspace = createReadWriteBuffer(handle, convWorkspaceElts.size1, useFP16);
    // convWorkspace2 = createReadWriteBuffer(handle, convWorkspaceElts.size2, useFP16);
  }

  ~Buffers() {
    VkHelpers::releaseVulkanBuffer(input->device, input);
    VkHelpers::releaseVulkanBuffer(inputGlobal->device, inputGlobal);
    if(inputMeta != nullptr)
      VkHelpers::releaseVulkanBuffer(inputMeta->device, inputMeta);
    VkHelpers::releaseVulkanBuffer(mask->device, mask);
    VkHelpers::releaseVulkanBuffer(maskSum->device, maskSum);
    VkHelpers::releaseVulkanBuffer(trunk->device, trunk);
    VkHelpers::releaseVulkanBuffer(policyPass->device, policyPass);
    VkHelpers::releaseVulkanBuffer(policy->device, policy);
    VkHelpers::releaseVulkanBuffer(value->device, value);
    VkHelpers::releaseVulkanBuffer(scoreValue->device, scoreValue);
    VkHelpers::releaseVulkanBuffer(ownership->device, ownership);
    if(convWorkspace != nullptr)
      VkHelpers::releaseVulkanBuffer(convWorkspace->device, convWorkspace);
    if(convWorkspace2 != nullptr)
      VkHelpers::releaseVulkanBuffer(convWorkspace2->device, convWorkspace2);
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
  const bool inputUsingNHWC;

  ComputeHandle(
    ComputeContext* context,
    const LoadedModel* loadedModel,
    int maxBatchSize,
    int gpuIdx,
    bool inputUsingNHWC_
  ): 
    handle( std::make_unique<ComputeHandleInternal>(
      context,
      gpuIdx,
      inputUsingNHWC_,
      context->usingNHWCMode == enabled_t::True ? true : false
    )),
    nnXLen(context->nnXLen),
    nnYLen(context->nnYLen),
    policySize(NNPos::getPolicySize(context->nnXLen,context->nnYLen)),
    inputUsingNHWC(inputUsingNHWC_),
    model( std::make_unique<Model>(
      handle.get(),
      loadedModel->modelDesc,
      maxBatchSize,
      context->nnXLen,
      context->nnYLen
    )),
    buffers( std::make_unique<Buffers>(
      handle.get(),
      *model
    ))
  {
    scratch = std::make_unique<ScratchBuffers>(
      handle.get(),
      model->maxBatchSize,
      model->nnXLen,
      model->nnYLen
    );
    // // Buffers* ptr = new Buffers(handle.get(), *model);
    // // buffers = std::make_unique<Buffers>(handle.get(), *model);
    // buffers.reset(ptr);
  }

  ~ComputeHandle() {}
  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

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

void NeuralNet::printDevices() {
  VkInstance inst = VkHelpers::createVulkanInstance();
  auto infos = VkHelpers::enumerateVulkanDevices(inst, nullptr);

  for ( const auto& info : infos ) {
    std::cout << "Found Vulkan Device " << Global::intToString(static_cast<int>(info.deviceId)) 
              << ": " << info.deviceName << std::endl;
  }
}

LoadedModel* NeuralNet::loadModelFile(const std::string& file, const std::string& expectedSha256) {
  // std::cout << "[NeuralNet::loadModelFile] Loading model file: " << file << std::endl;
  LoadedModel* loadedModel = new LoadedModel(file, expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

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
  // TODO: Check requiredExactNNLen required or not
  (void)requiredExactNNLen;
  ComputeHandle* handle = new ComputeHandle(
    context,
    loadedModel,
    maxBatchSize,
    gpuIdxForThisThread,
    inputsUseNHWC
  );
  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* handle) {
  delete handle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle *handle) {
  return false;
}

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;
  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;

  size_t userInputBufferElts;
  size_t userInputGlobalBufferElts;
  size_t userInputMetaBufferElts;
  size_t policyPassResultBufferElts;
  size_t policyResultBufferElts;
  size_t valueResultBufferElts;
  size_t scoreValueResultBufferElts;
  size_t ownershipResultBufferElts;

  float* userInputBuffer; //Host pointer
  // half_t* userInputBufferHalf; //Host pointer
  float* userInputGlobalBuffer; //Host pointer
  float* userInputMetaBuffer; //Host pointer

  float* policyPassResults; //Host pointer
  float* policyResults; //Host pointer
  // half_t* policyResultsHalf; //Host pointer
  float* valueResults; //Host pointer
  float* scoreValueResults; //Host pointer
  float* ownershipResults; //Host pointer
  // half_t* ownershipResultsHalf; //Host pointer

  InputBuffers(
    const LoadedModel* loadedModel,
    int maxBatchSize_,
    int nnXLen,
    int nnYLen
  ) {
    // Bytes size will not be computed because of fp16.
    const ModelDesc& m = loadedModel->modelDesc;
    maxBatchSize = maxBatchSize_;
    singleInputElts = static_cast<size_t>(m.numInputChannels) * nnXLen * nnYLen;
    singleInputGlobalElts = static_cast<size_t>(m.numInputGlobalChannels);
    singleInputMetaElts = static_cast<size_t>(m.numInputMetaChannels);
    singlePolicyPassResultElts = static_cast<size_t>(m.numPolicyChannels);
    singlePolicyResultElts = static_cast<size_t>(m.numPolicyChannels * nnXLen * nnYLen);
    singleValueResultElts = static_cast<size_t>(m.numValueChannels);
    singleScoreValueResultElts = static_cast<size_t>(m.numScoreValueChannels);
    singleOwnershipResultElts = static_cast<size_t>(m.numOwnershipChannels * nnXLen * nnYLen);

    assert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);

    if ( m.numInputMetaChannels > 0 ) {
      assert(m.numInputMetaChannels == SGFMetadata::METADATA_INPUT_NUM_CHANNELS);
    }

    userInputBufferElts = static_cast<size_t>( m.numInputChannels ) * maxBatchSize * nnXLen * nnYLen;
    userInputGlobalBufferElts = static_cast<size_t>( m.numInputGlobalChannels ) * maxBatchSize;
    userInputMetaBufferElts = static_cast<size_t>( m.numInputMetaChannels ) * maxBatchSize;
    policyPassResultBufferElts = static_cast<size_t>( maxBatchSize ) * m.numPolicyChannels;
    policyResultBufferElts = static_cast<size_t>( maxBatchSize ) * m.numPolicyChannels * nnXLen * nnYLen;
    valueResultBufferElts = static_cast<size_t>( maxBatchSize ) * m.numValueChannels;
    scoreValueResultBufferElts = static_cast<size_t>( maxBatchSize ) * m.numScoreValueChannels;
    ownershipResultBufferElts = static_cast<size_t>( maxBatchSize ) * m.numOwnershipChannels * nnXLen * nnYLen;

    userInputBuffer = new float[userInputBufferElts];
    userInputGlobalBuffer = new float[userInputGlobalBufferElts];
    if ( m.numInputMetaChannels > 0 ) {
      userInputMetaBuffer = new float[userInputMetaBufferElts];
    } else {
      userInputMetaBuffer = nullptr;
    }

    policyPassResults = new float[ static_cast<size_t>( m.numPolicyChannels * maxBatchSize ) ];
    policyResults = new float[ static_cast<size_t>( m.numPolicyChannels * nnXLen * nnYLen * maxBatchSize ) ];
    valueResults = new float[ static_cast<size_t>( m.numValueChannels * maxBatchSize ) ];
    scoreValueResults = new float[ static_cast<size_t>( m.numScoreValueChannels * maxBatchSize ) ];
    ownershipResults = new float[ static_cast<size_t>( m.numOwnershipChannels * nnXLen * nnYLen * maxBatchSize ) ];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    if ( userInputMetaBuffer != nullptr ) {
      delete[] userInputMetaBuffer;
    }
    delete[] policyPassResults;
    delete[] policyResults;
    delete[] valueResults;
    delete[] scoreValueResults;
    delete[] ownershipResults;
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(
    loadedModel,
    maxBatchSize,
    nnXLen,
    nnYLen
  );
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

void NeuralNet::getOutput(
  ComputeHandle *computeHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  std::vector<NNOutput*>& outputs
) {
  assert( numBatchEltsFilled <= inputBuffers->maxBatchSize );
  assert( numBatchEltsFilled > 0 );
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = computeHandle->nnXLen;
  const int nnYLen = computeHandle->nnYLen;
  const int modelVersion = computeHandle->model->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = static_cast<int>(inputBuffers->singleInputMetaElts);
  assert(numSpatialFeatures == computeHandle->model->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts );
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);
  const int numPolicyChannels = computeHandle->model->numPolicyChannels;

  for (int nIdx = 0 ; nIdx < batchSize ; ++nIdx) {
    float* rowSpatialInput = inputBuffers->userInputBuffer + ( inputBuffers->singleInputElts * nIdx );
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + ( inputBuffers->singleInputGlobalElts * nIdx );
    float* rowMetaInput = inputBuffers->userInputMetaBuffer + ( inputBuffers->singleInputMetaElts * nIdx );

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    const bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;
    std::copy(rowGlobal, rowGlobal+numGlobalFeatures, rowGlobalInput);

    if ( numMetaFeatures > 0 ) {
      testAssert(rowMeta != NULL);
      testAssert(hasRowMeta);
      std::copy(rowMeta, rowMeta+numMetaFeatures, rowMetaInput);
    } else {
      testAssert(!hasRowMeta);
    }
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, computeHandle->inputUsingNHWC, inputBufs[nIdx]->symmetry);
  }
  Buffers* buffers = computeHandle->buffers.get();

  assert(inputBuffers->userInputBufferElts == buffers->inputElts);
  assert(inputBuffers->userInputGlobalBufferElts == buffers->inputGlobalElts);
  assert(inputBuffers->userInputMetaBufferElts == buffers->inputMetaElts);
  assert(inputBuffers->policyResultBufferElts == buffers->policyElts);
  assert(inputBuffers->valueResultBufferElts == buffers->valueElts);
  assert(inputBuffers->singlePolicyPassResultElts == numPolicyChannels);
  assert(inputBuffers->singlePolicyResultElts == numPolicyChannels * nnXLen * nnYLen);
  assert(inputBuffers->singlePolicyResultElts + inputBuffers->singlePolicyPassResultElts == computeHandle->policySize * numPolicyChannels);
  assert(inputBuffers->scoreValueResultBufferElts == buffers->scoreValueElts);
  assert(inputBuffers->ownershipResultBufferElts == buffers->ownershipElts);
  assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);

  ComputeHandleInternal* handle = computeHandle->handle.get();
  bool useFP16Storage = false; // TODO: enable fp16 storage.

  VkResult res = VK_ERROR_UNKNOWN;

  if ( useFP16Storage ) {
    // TODO: implement fp16 storage path
  } else {
    VkHelpers::copyHostToDeviceBuffer(
      handle->vulkanDevice,
      inputBuffers->userInputBuffer, // Host pointer
      buffers->input,
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleInputElts),
      false,
      &res
    );
    CHECK_VK_MSG("Copy input buffer to device", res);

    VkHelpers::copyHostToDeviceBuffer(
      handle->vulkanDevice,
      inputBuffers->userInputGlobalBuffer, // Host pointer
      buffers->inputGlobal,
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleInputGlobalElts),
      false,
      &res
    );
    CHECK_VK_MSG("Copy input global buffer to device", res);

    if ( numMetaFeatures > 0 ) {
      VkHelpers::copyHostToDeviceBuffer(
        handle->vulkanDevice,
        inputBuffers->userInputMetaBuffer, // Host pointer
        buffers->inputMeta,
        static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleInputMetaElts),
        true,
        &res
      );
      CHECK_VK_MSG("Copy input meta buffer to device", res);
    }

    computeHandle->model->record(
      batchSize,
      computeHandle->scratch.get(),
      buffers->input,
      buffers->inputGlobal,
      buffers->inputMeta,
      buffers->mask,
      buffers->maskSum,
      buffers->trunk,
      buffers->policyPass,
      buffers->policy,
      buffers->value,
      buffers->scoreValue,
      buffers->ownership
    );
    // computeHandle->model->debug(
    //   batchSize,
    //   computeHandle->scratch.get(),
    //   buffers->input,
    //   buffers->inputGlobal,
    //   buffers->inputMeta,
    //   buffers->mask,
    //   buffers->maskSum,
    //   buffers->trunk,
    //   buffers->policyPass,
    //   buffers->policy,
    //   buffers->value,
    //   buffers->scoreValue,
    //   buffers->ownership
    // );
    computeHandle->model->apply();

    // Read back PolicyPass result
    VkHelpers::copyDeviceBufferToHost(
      handle->vulkanDevice,
      buffers->policyPass,
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singlePolicyPassResultElts),
      inputBuffers->policyPassResults,
      true,
      &res
    );
    CHECK_VK_MSG("Copy policy pass results buffer to host", res);

    #ifdef VULKAN_API_DEBUG
    printHostBuffer(
      """[NeuralNet::getOutput] policy pass results",
      inputBuffers->policyPassResults,
      batchSize * inputBuffers->singlePolicyPassResultElts
    );
    #endif
    // std::cout << "policy pass result[0]: " << inputBuffers->policyPassResults[0] << std::endl;

    // Read back Policy result
    if ( useFP16Storage ) {
      // TODO: implement fp16 storage path
    } else {
      VkHelpers::copyDeviceBufferToHost(
        handle->vulkanDevice,
        buffers->policy,
        static_cast<VkDeviceSize>(sizeof(float) * batchSize * (inputBuffers->singlePolicyResultElts)),
        inputBuffers->policyResults,
        true,
        &res
      );
      CHECK_VK_MSG("Copy policy results buffer to host", res);
      #ifdef VULKAN_API_DEBUG
      printHostBuffer(
        "[NeuralNet::getOutput] policy results",
        inputBuffers->policyResults,
        batchSize * inputBuffers->singlePolicyResultElts
      );
      #endif
    }

    // Read back Value result
    VkHelpers::copyDeviceBufferToHost(
      handle->vulkanDevice,
      buffers->value,
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleValueResultElts),
      inputBuffers->valueResults,
      true,
      &res
    );
    CHECK_VK_MSG("Copy value results buffer to host", res);

    #ifdef VULKAN_API_DEBUG
    printHostBuffer(
      "[NeuralNet::getOutput] value results",
      inputBuffers->valueResults,
      batchSize * inputBuffers->singleValueResultElts
    );
    #endif

    // Read back ScoreValue result
    VkHelpers::copyDeviceBufferToHost(
      handle->vulkanDevice,
      buffers->scoreValue,
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleScoreValueResultElts),
      inputBuffers->scoreValueResults,
      true,
      &res
    );
    CHECK_VK_MSG("Copy score value results buffer to host", res);

    #ifdef VULKAN_API_DEBUG
    printHostBuffer(
      "[NeuralNet::getOutput] score value results",
      inputBuffers->scoreValueResults,
      batchSize * inputBuffers->singleScoreValueResultElts
    );
    #endif 

    // Read back Ownership result
    if ( useFP16Storage ) {
      // TODO: implement fp16 storage path
    } else {
      VkHelpers::copyDeviceBufferToHost(
        handle->vulkanDevice,
        buffers->ownership,
        static_cast<VkDeviceSize>(sizeof(float) * batchSize * (inputBuffers->singleOwnershipResultElts)),
        inputBuffers->ownershipResults,
        true,
        &res
      );
      CHECK_VK_MSG("Copy ownership results buffer to host", res);
      #ifdef VULKAN_API_DEBUG
      printHostBuffer(
        "[NeuralNet::getOutput] ownership results",
        inputBuffers->ownershipResults,
        batchSize * inputBuffers->singleOwnershipResultElts
      );
      #endif
    }
    #ifdef VULKAN_API_DEBUG
    exit(EXIT_FAILURE);
    #endif

    assert(outputs.size() == static_cast<size_t>(batchSize));

    float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

    for ( int row = 0 ; row < batchSize ; ++row ) {
      NNOutput* output = outputs[row];
      assert(output->nnXLen == nnXLen);
      assert(output->nnYLen == nnYLen);
      float policyOptimism = static_cast<float>(inputBufs[row]->policyOptimism);
      
      const float* policyPassSrcBuf = inputBuffers->policyPassResults + row * numPolicyChannels;
      const float* policySrcBuf = inputBuffers->policyResults + row * (numPolicyChannels * nnXLen * nnYLen);
      float* policyProbs = output->policyProbs;

      if ( numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16) ) {
        // Vulkan NCHW
        for ( int i = 0 ; i < nnXLen * nnYLen ; ++i ) {
          float p = policySrcBuf[i];
          float pOpt = policySrcBuf[nnXLen * nnYLen + i];
          policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      } else {
        assert(numPolicyChannels == 1);
        SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0];
      }

      int numValueChannels = computeHandle->model->numValueChannels;
      assert(numValueChannels == 3);
      output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
      output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
      output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

      if ( output->whiteOwnerMap != NULL ) {
        const float* ownershipSrcBuf = inputBuffers->ownershipResults + row * (nnXLen * nnYLen);
        assert(computeHandle->model->numOwnershipChannels == 1);
        SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      }
      if(modelVersion >= 9) {
        int numScoreValueChannels = computeHandle->model->numScoreValueChannels;
        assert(numScoreValueChannels == 6);
        output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
        output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
        output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
        output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
        output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
      }
      else if(modelVersion >= 8) {
        int numScoreValueChannels = computeHandle->model->numScoreValueChannels;
        assert(numScoreValueChannels == 4);
        output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
        output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
        output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
        output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(modelVersion >= 4) {
        int numScoreValueChannels = computeHandle->model->numScoreValueChannels;
        assert(numScoreValueChannels == 2);
        output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
        output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
        output->whiteLead = output->whiteScoreMean;
        output->varTimeLeft = 0;
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(modelVersion >= 3) {
        int numScoreValueChannels = computeHandle->model->numScoreValueChannels;
        assert(numScoreValueChannels == 1);
        output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
        //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
        output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
        output->whiteLead = output->whiteScoreMean;
        output->varTimeLeft = 0;
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else {
        ASSERT_UNREACHABLE;
      }
    }
  }
}

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  std::vector<float>& outputBuffer
) {
  Logger* logger = nullptr;
  VkResult res = VK_ERROR_UNKNOWN;
  int gpuId = 0;

  if ( useNHWC ) {
    // TODO: NHWC not supported yet
    return false;
  }

  if ( useFP16 ) {
    // TODO: FP16 not supported yet
    return false;
  }

  // print test configs
  // std::cout << "[testEvaluateConv] batchSize: " << batchSize
  //           << " nnXLen: " << nnXLen
  //           << " nnYLen: " << nnYLen
  //           << " inChannels: " << desc->inChannels
  //           << " outChannels: " << desc->outChannels
  //           << " convYSize: " << desc->convYSize
  //           << " convXSize: " << desc->convXSize 
  //           << " useFP16: " << (useFP16 ? "true" : "false")
  //           << " useNHWC: " << (useNHWC ? "true" : "false")
  //           << std::endl;
  // // print default input state;
  // std::cout << "[testEvaluateConv] inputBuffer size: " << inputBuffer.size() << std::endl;
  // printFloatBuffer("testEvaluateConv Input", inputBuffer.data(), inputBuffer.size(), batchSize, desc->inChannels, nnYLen, nnXLen);
  // std::cout << "[testEvaluateConv] filter size: " << desc->inChannels * desc->outChannels * desc->convYSize * desc->convXSize << std::endl; 
  // printFloatBuffer("testEvaluateConv Filter", desc->weights.data(), desc->inChannels * desc->outChannels * desc->convYSize * desc->convXSize, batchSize, desc->outChannels, desc->convYSize, desc->convXSize);

  ComputeContext* ctx = createComputeContextForTesting({gpuId}, logger, nnXLen, nnYLen, false, false);
  // std::cout << "[testEvaluateConv] Created compute context" << std::endl;
  ComputeHandleInternal* handle = new ComputeHandleInternal(ctx,static_cast<int>(gpuId), useNHWC, useNHWC);
  const VulkanDevice* device = handle->vulkanDevice;
  ConvLayer *layer = new ConvLayer(handle, desc, nnXLen, nnYLen);
  size_t numInputFloats = static_cast<size_t>(batchSize) * static_cast<size_t>(desc->inChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen);
  size_t numOutputFloats = static_cast<size_t>(batchSize) * static_cast<size_t>(desc->outChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen);

  if ( numInputFloats != inputBuffer.size() ) {
    // std::cerr << "testEvaluateConv input size mismatch, expected " << numInputFloats << " got " << inputBuffer.size() << std::endl;
    delete layer;
    delete handle;
    delete ctx;
    return false;
  }
  outputBuffer.resize(numOutputFloats);
  // std::cout << "  expected output size " <<  numOutputFloats << std::endl;
  std::vector<float> inputTmp = inputBuffer;
  VulkanBuffer* dInput = VkHelpers::createDeviceBufferWithData(
    device,
    byteSizeofVectorContents(inputTmp),
    inputTmp.data(),
    true,
    &res
  );
  CHECK_VK_MSG("[TestConv] Failed to create device input buffer with data", res);
  VulkanBuffer* dOutput = VkHelpers::createDeviceBuffer(
    device,
    byteSizeofVectorContents(outputBuffer),
    false,
    &res
  );
  CHECK_VK_MSG("[TestConv] Failed to create device buffer", res);

  layer->record(batchSize,dInput,dOutput);
  // layer->apply(batchSize, dInput, dOutput);
  VkHelpers::submitCommandBuffers(handle->vulkanDevice, {layer->commandBuffer}, nullptr);
  VkHelpers::copyDeviceBufferToHost(device, dOutput, static_cast<VkDeviceSize>(sizeof(float) * numOutputFloats), outputBuffer.data(), true, &res);
  CHECK_VK_MSG("[TestConv] Failed to copy device output buffer to host", res);
  vkQueueWaitIdle(device->queue);
  vkDeviceWaitIdle(device->device);
  VkHelpers::releaseVulkanBuffer(device, dInput);
  VkHelpers::releaseVulkanBuffer(device, dOutput);
  dInput = nullptr;
  dOutput = nullptr;
  delete layer;
  delete handle;
  freeComputeContext(ctx);
  return true;
}

bool NeuralNet::testEvaluateBatchNorm(
    const BatchNormLayerDesc* desc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC,
    const std::vector<float>& inputBuffer,
    const std::vector<float>& maskBuffer,
    std::vector<float>& outputBuffer
  ) {
    if ( useNHWC ) { 
      // TODO: NHWC not supported yet
      return false;
    }

    if( useFP16 ) {
      // TODO: FP16 not supported yet
      return false;
    }

    // std::cout << "[testEvaluateBatchNorm] batchSize: " << batchSize
    //           << " nnXLen: " << nnXLen
    //           << " nnYLen: " << nnYLen
    //           << " numChannels: " << desc->numChannels
    //           << " useFP16: " << (useFP16 ? "true" : "false")
    //           << " useNHWC: " << (useNHWC ? "true" : "false")
    //           << std::endl;

    // printFloatBuffer(
    //   "[testEvaluateBatchNorm] Input",
    //   inputBuffer.data(),
    //   inputBuffer.size(),
    //   batchSize,
    //   desc->numChannels,
    //   nnYLen,
    //   nnXLen
    // );

    // printFloatBuffer(
    //   "[testEvaluateBatchNorm] Mask",
    //   maskBuffer.data(),
    //   maskBuffer.size(),
    //   batchSize,
    //   1,
    //   nnYLen,
    //   nnXLen
    // );

    Logger* logger = nullptr;
    auto ctx = createComputeContextForTesting({0}, nullptr, nnXLen, nnYLen, useFP16, useNHWC);
    useNHWC = false; // TODO: enable NHWC testing later.
    auto handle = new ComputeHandleInternal(ctx,0, useNHWC, useNHWC);
    // BatchNormLayer *layer = new BatchNormLayer(handle, desc, nnXLen
    ActivationLayerDesc actDesc;
    actDesc.activation = ACTIVATION_IDENTITY;
    BatchNormLayer *layer = new BatchNormLayer(handle, desc, &actDesc, nnXLen, nnYLen);
    size_t numInputFloats = static_cast<size_t>(batchSize) * static_cast<size_t>(desc->numChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen);
    size_t numOutputFloats = static_cast<size_t>(batchSize) * static_cast<size_t>(desc->numChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen);
    size_t numMaskFloats = static_cast<size_t>(batchSize) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen);
    VkResult res = VK_ERROR_UNKNOWN;
    if ( numInputFloats != inputBuffer.size() ) {
      throw StringError("[testEvaluateBatchNorm] unexpected input size");
      delete layer;
      delete handle;
      freeComputeContext(ctx);
      return false;
    }
    outputBuffer.resize(numOutputFloats);
    std::vector<float> inputTmp = inputBuffer;
    std::vector<float> maskTmp = maskBuffer;
    VulkanBuffer* dInput = VkHelpers::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(inputTmp),
      inputTmp.data(),
      true,
      &res
    );
    CHECK_VK_MSG("[testEvaluateBatchNorm] Failed to create device input buffer with data", res);
    VulkanBuffer* dMask = VkHelpers::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(maskTmp),
      maskTmp.data(),
      true,
      &res
    );
    CHECK_VK_MSG("[testEvaluateBatchNorm] Failed to create device mask buffer", res);
    VulkanBuffer* dOutput = VkHelpers::createDeviceBuffer(
      handle->vulkanDevice,
      byteSizeofVectorContents(outputBuffer),
      false,
      &res
    );
    CHECK_VK_MSG("[testEvaluateBatchNorm] Failed to create device output buffer", res);

    layer->record(batchSize, dInput, dMask, dOutput);
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, {layer->commandBuffer}, nullptr);
    // layer->apply(batchSize, dInput, dMask, dOutput);
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, dOutput, static_cast<VkDeviceSize>(sizeof(float) * outputBuffer.size()), outputBuffer.data(), true, &res);
    CHECK_VK_MSG("[testEvaluateBatchNorm] Failed to copy device output buffer to host", res);
    // delete dOutput;
    // delete dMask;
    // delete dInput;
    VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, dInput);
    VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, dMask);
    VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, dOutput);
    dInput = nullptr;
    dMask = nullptr;
    dOutput = nullptr;
    delete layer;
    delete handle;
    freeComputeContext(ctx);
    return true;
  }

  bool NeuralNet::testEvaluateResidualBlock(
    const ResidualBlockDesc* desc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC,
    const std::vector<float>& inputBuffer,
    const std::vector<float>& maskBuffer,
    std::vector<float>& outputBuffer
  ) {
    Logger *logger = nullptr;
    VkResult res = VK_ERROR_UNKNOWN;
    int gpuId = 0;
    // std::cout << "[testEvaluateResidualBlock] Starting testEvaluateResidualBlock test case... " << std::endl;
    // return false;

    if ( useNHWC ) {
      // TODO: NHWC not supported yet
      return false;
    }

    if ( useFP16 ) {
      // TODO: FP16 not supported yet
      return false;
    }

    // std::cout << "[testEvaluateResidualBlock]\n batchSize: " << batchSize
    //           << "\n nnXLen: " << nnXLen
    //           << "\n nnYLen: " << nnYLen
    //           << "\n preBN numChannels: " << desc->preBN.numChannels
    //           << "\n useFP16: " << (useFP16 ? "true" : "false")
    //           << "\n useNHWC: " << (useNHWC ? "true" : "false")
    //           << std::endl;

    // printFloatBuffer("[testEvaluateResidualBlock] Input", inputBuffer.data(), inputBuffer.size(), batchSize, desc->preBN.numChannels, nnYLen, nnXLen);
    // printFloatBuffer("[testEvaluateResidualBlock] Mask", maskBuffer.data(), maskBuffer.size(), batchSize, 1, nnYLen, nnXLen);
    // printFloatBuffer("[testEvaluateResidualBlock] regularConv Weights", 
    //   desc->regularConv.weights.data(), 
    //   desc->regularConv.inChannels * desc->regularConv.outChannels * desc->regularConv.convYSize * desc->regularConv.convXSize,
    //   batchSize,
    //   desc->regularConv.outChannels, 
    //   desc->regularConv.convYSize, 
    //   desc->regularConv.convXSize
    // );
    // printFloatBuffer()


    useNHWC = false; // TODO: enable NHWC testing later.
    ComputeContext* ctx = createComputeContextForTesting({gpuId}, logger, nnXLen, nnYLen, false, false);
    ComputeHandleInternal* handle = new ComputeHandleInternal(ctx,static_cast<int>(gpuId), useNHWC, useNHWC);
    ResidualBlock *layer = new ResidualBlock(handle, desc, nnXLen, nnYLen);

    size_t numTrunkFloats =  static_cast<size_t>(batchSize * nnXLen * nnYLen * desc->preBN.numChannels);
    size_t numMaskFloats = static_cast<size_t>(batchSize * nnXLen * nnYLen);
    if ( numTrunkFloats != inputBuffer.size() ) {
      delete layer;
      delete handle;
      freeComputeContext(ctx);
      throw StringError("[testEvaluateResidualBlock] unexpected input size");
    }

    if ( numMaskFloats != maskBuffer.size() ) {
      delete layer;
      delete handle;
      freeComputeContext(ctx);
      throw StringError("[testEvaluateResidualBlock] unexpected mask size");
    }

    outputBuffer.resize(numTrunkFloats);
    ScratchBuffers* scratch = new ScratchBuffers(handle, batchSize, nnXLen, nnYLen);
    std::vector<float> inputTmp = inputBuffer;
    std::vector<float> maskTmp = maskBuffer;
    VulkanBuffer* dTrunk = VkHelpers::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(inputTmp),
      inputTmp.data(),
      false,
      &res
    );
    CHECK_VK_MSG("[testEvaluateResidualBlock] Failed to create device trunk buffer with data", res);
    VulkanBuffer* dMask = VkHelpers::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(maskTmp),
      maskTmp.data(),
      true,
      &res
    );
    CHECK_VK_MSG("[testEvaluateResidualBlock] Failed to create device mask buffer with data", res);
    VulkanBuffer* dTrunkScratch = VkHelpers::createDeviceBuffer(
      handle->vulkanDevice,
      numTrunkFloats * sizeof(float),
      false,
      &res
    );
    CHECK_VK_MSG("[testEvaluateResidualBlock] Failed to create device trunk scratch buffer", res);

    layer->record(batchSize, scratch, dTrunk, dTrunkScratch, dMask);
    // layer->apply(batchSize, scratch, dTrunk, dTrunkScratch, dMask);
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, layer->commandBuffers, nullptr);
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, dTrunk, static_cast<VkDeviceSize>(sizeof(float) * numTrunkFloats), outputBuffer.data(), true, &res);
    CHECK_VK_MSG("[testEvaluateResidualBlock] Failed to copy device trunk buffer to host", res);
    vkQueueWaitIdle(handle->vulkanDevice->queue);
    vkDeviceWaitIdle(handle->vulkanDevice->device);
    VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, dTrunk);
    VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, dMask);
    VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, dTrunkScratch);
    dTrunk = nullptr;
    dMask = nullptr;
    dTrunkScratch = nullptr;
    delete scratch;
    delete layer;
    delete handle;
    freeComputeContext(ctx);

    return true;
  }

  bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
    const GlobalPoolingResidualBlockDesc* desc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC,
    const std::vector<float>& inputBuffer,
    const std::vector<float>& maskBuffer,
    std::vector<float>& outputBuffer
  ) {
    Logger *logger = nullptr;
    VkResult res = VK_ERROR_UNKNOWN;
    int gpuId = 0;

    if ( useNHWC ) {
      // TODO: NHWC not supported yet
      return false;
    }
    if ( useFP16 ) {
      // TODO: FP16 not supported yet
      return false;
    }

    // std::cout << "[testEvaluateGlobalPoolingResidualBlock] batchSize: " << batchSize
    //           << " nnXLen: " << nnXLen
    //           << " nnYLen: " << nnYLen
    //           << " preBN numChannels: " << desc->preBN.numChannels
    //           << " useFP16: " << (useFP16 ? "true" : "false")
    //           << " useNHWC: " << (useNHWC ? "true" : "false")
    //           << std::endl;

    ComputeContext* ctx = createComputeContextForTesting({gpuId}, logger, nnXLen, nnYLen, useFP16, useNHWC);
    ComputeHandleInternal* handle = new ComputeHandleInternal(ctx,static_cast<int>(gpuId), useNHWC, useNHWC);
    GlobalPoolingResidualBlock *layer = new GlobalPoolingResidualBlock(handle, desc, nnXLen, nnYLen);

    size_t numTrunkFloats = static_cast<size_t>(batchSize * nnXLen * nnYLen * desc->preBN.numChannels);
    size_t numMaskFloats = static_cast<size_t>(batchSize * nnXLen * nnYLen);
    size_t numMaskSumFloats = static_cast<size_t>(batchSize);

    if ( numTrunkFloats != inputBuffer.size() ) {
      delete layer;
      delete handle;
      freeComputeContext(ctx);
      throw StringError("[testEvaluateGlobalPoolingResidualBlock] unexpected input size");
    }
    if ( numMaskFloats != maskBuffer.size() ) {
      delete layer;
      delete handle;
      freeComputeContext(ctx);
      throw StringError("[testEvaluateGlobalPoolingResidualBlock] unexpected mask size");
    }
    // printFloatBuffer("[testEvaluateGlobalPoolingResidualBlock] Input", inputBuffer.data(), inputBuffer.size(), batchSize, desc->preBN.numChannels, nnYLen, nnXLen);
    // printFloatBuffer("[testEvaluateGlobalPoolingResidualBlock] Mask", maskBuffer.data(), maskBuffer.size(), batchSize, 1, nnYLen, nnXLen);

    outputBuffer.resize(numTrunkFloats);
    ScratchBuffers* scratch = new ScratchBuffers(handle, batchSize, nnXLen, nnYLen);
    std::vector<float> inputTmp = inputBuffer;
    std::vector<float> maskTmp = maskBuffer;
    VulkanBuffer* dTrunk = VkHelpers::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(inputTmp),
      inputTmp.data(),
      false,
      &res
    );
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to create device trunk buffer with data", res);
    VulkanBuffer* dMask = VkHelpers::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(maskTmp),
      maskTmp.data(),
      true,
      &res
    );
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to create device mask buffer with data", res);
    VulkanBuffer* dTrunkScratch = VkHelpers::createDeviceBuffer(
      handle->vulkanDevice,
      numTrunkFloats * sizeof(float),
      false,
      &res
    );
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to create device trunk scratch buffer", res);
    VulkanBuffer* dMaskSum = VkHelpers::createDeviceBuffer(
      handle->vulkanDevice,
      numMaskSumFloats * sizeof(float),
      false,
      &res
    );
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to create device mask sum buffer", res);  
    std::vector<float> maskSumTmp(numMaskSumFloats, 0.0f);
    VkCommandBuffer maskSumsCB = computeMaskSums(handle, batchSize, nnXLen, nnYLen, dMask, dMaskSum);
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, {maskSumsCB}, nullptr);
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, dMaskSum, static_cast<VkDeviceSize>(sizeof(float) * numMaskSumFloats), maskSumTmp.data(), true, &res);
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to copy device mask sum buffer to host", res);
    // printFloatBuffer("[testEvaluateGlobalPoolingResidualBlock] Mask Sums", maskSumTmp.data(), maskSumTmp.size(), batchSize, 1, 1, 1);
    layer->record(batchSize, scratch, dTrunk, dTrunkScratch, dMask, dMaskSum);
    // layer->apply(batchSize, scratch, dTrunk, dTrunkScratch, dMask, dMaskSum);
    std::vector<VkCommandBuffer> allCommands;
    allCommands.push_back(maskSumsCB);
    allCommands.insert(allCommands.end(), layer->commandBuffers.begin(), layer->commandBuffers.end());
    VkHelpers::submitCommandBuffers(handle->vulkanDevice, allCommands, nullptr);
    VkHelpers::copyDeviceBufferToHost(handle->vulkanDevice, dTrunk, static_cast<VkDeviceSize>(sizeof(float) * numTrunkFloats), outputBuffer.data(), true, &res);
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to copy device trunk buffer to host", res);
    vkQueueWaitIdle(handle->vulkanDevice->queue);
    vkDeviceWaitIdle(handle->vulkanDevice->device);

    printFloatBuffer("[testEvaluateGlobalPoolingResidualBlock] Output", outputBuffer.data(), outputBuffer.size(), batchSize, desc->preBN.numChannels, nnYLen, nnXLen);

    VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, dTrunk);
    VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, dMask);
    VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, dTrunkScratch);
    VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, dMaskSum);
    dTrunk = nullptr;
    dMask = nullptr;
    dTrunkScratch = nullptr;
    dMaskSum = nullptr;
    delete scratch;
    delete layer;
    delete handle;
    freeComputeContext(ctx);
    return true;
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
    // createConv2d3x3BnFp32();
    // createConv2d3x3BnReluFp32();
    // createConv2d5x5BnFp32();
    // createConv2d5x5BnReluFp32();
    // createConv2d5x5BnMishFp32();
    createAddPointWiseFp32();
    createMatmulFp32();
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
    // destroyPipeline(conv2d3x3BnFp32);
    // destroyPipeline(conv2d3x3BnReluFp32);
    // destroyPipeline(conv2d5x5BnFp32);
    // destroyPipeline(conv2d5x5BnReluFp32);
    // destroyPipeline(conv2d5x5BnMishFp32);
    destroyPipeline(addPointWiseFp32);
    destroyPipeline(matmulFp32);
    // destroyPipeline(matmulTiledChw4x4x32Fp32);
    destroyPipeline(batchNormMaskFp32);
    destroyPipeline(batchNormMaskReluFp32);
    destroyPipeline(batchNormMaskMishFp32);
    destroyPipeline(batchNormMaskMishScale8Fp32);
    destroyPipeline(globalPoolingChannelsFp32);
    destroyPipeline(valueHeadPoolingChannelsFp32);
    destroyPipeline(sumChannelsFp32);
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
    Pipeline &outPipeline
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
    VkShaderModuleCreateInfo shaderModuleCI = {};
    shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCI.codeSize = spirvSize;
    shaderModuleCI.pCode = spirvWords.data();
    // std::cout << "Creating Compute Pipeline: " << pipelineName <<  " code size : " << shaderModuleCI.codeSize << std::endl;
    res = vkCreateShaderModule(device, &shaderModuleCI, nullptr, &shaderModule);
    CHECK_VK_MSG(pipelineName + "ShaderModule", res);
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
    // std::cout << "Created Compute Pipeline: " << pipelineName << " result : " << res << std::endl;
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
  // void ComputePipelines::createConv2d3x3BnFp32() {
  //   createPipeline("Conv2d3x3BnFp32", VkSPIRVShaders::spirv_conv2d_3x3_bn_fp32, VkSPIRVShaders::spirv_conv2d_3x3_bn_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d3x3BnFp32);
  // }

  /**
   * @brief Create a Conv2d3x3 Bn + ReLU Activation fused Fp32 objects.
   */
  // void ComputePipelines::createConv2d3x3BnReluFp32() {
  //   createPipeline("Conv2d3x3BnReluFp32", VkSPIRVShaders::spirv_conv2d_3x3_bn_relu_fp32, VkSPIRVShaders::spirv_conv2d_3x3_bn_relu_fp32_size,   3, sizeof(Conv2DPushConstantParams), conv2d3x3BnReluFp32);
  // }
  /**
   * @brief Create a Conv2d3x3 Bn Mish Fp32 object
   */
  // void ComputePipelines::createConv2d3x3BnMishFp32() {
  //   createPipeline("Conv2d3x3BnMishFp32", VkSPIRVShaders::spirv_conv2d_3x3_bn_mish_fp32,VkSPIRVShaders::spirv_conv2d_3x3_bn_mish_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d3x3BnMishFp32);
  // }

  /**
   * @brief Create a Conv2d5x5 Bn + Identity Activation fused Fp32 objects.
   */
  // void ComputePipelines::createConv2d5x5BnFp32() {
  //   createPipeline("Conv2d5x5BnFp32", VkSPIRVShaders::spirv_conv2d_5x5_bn_fp32, VkSPIRVShaders::spirv_conv2d_5x5_bn_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnFp32);
  // }

  /**
   * @brief Create a Conv2d5x5 Bn + ReLU Activation fused Fp32 objects.
   */
  // void ComputePipelines::createConv2d5x5BnReluFp32() {
  //   createPipeline("Conv2d5x5BnReluFp32",VkSPIRVShaders::spirv_conv2d_5x5_bn_relu_fp32, VkSPIRVShaders::spirv_conv2d_5x5_bn_relu_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnReluFp32);
  // }

  /**
   * @brief Create a Conv2d5x5 Bn Mish Fp32 object
   */
  // void ComputePipelines::createConv2d5x5BnMishFp32() {
  //   createPipeline("Conv2d5x5BnMishFp32",VkSPIRVShaders::spirv_conv2d_5x5_bn_mish_fp32, VkSPIRVShaders::spirv_conv2d_5x5_bn_mish_fp32_size, 3, sizeof(Conv2DPushConstantParams), conv2d5x5BnMishFp32);
  // }

  /**
   * @brief Create a Add Point Wise Fp32 object
   */
  void ComputePipelines::createAddPointWiseFp32() {
    createPipeline("AddPointWiseFp32",VkSPIRVShaders::spirv_add_pointwise_fp32, VkSPIRVShaders::spirv_add_pointwise_fp32_size, 2, sizeof(KatagoVulkan::AddPointWiseParams), addPointWiseFp32);
  }

  void ComputePipelines::createMatmulFp32() {
    createPipeline("MatmulFp32", VkSPIRVShaders::spirv_matmul_fp32, VkSPIRVShaders::spirv_matmul_fp32_size, 3, sizeof(KatagoVulkan::MatmulFp32Params), matmulFp32);
  }

  void ComputePipelines::createBatchNormMaskFp32() {
    createPipeline("BatchNormMaskFp32", VkSPIRVShaders::spirv_bn_mask_fp32, VkSPIRVShaders::spirv_bn_mask_fp32_size, 5, sizeof(KatagoVulkan::BatchNormMaskParams), batchNormMaskFp32);
  }

  void ComputePipelines::createBatchNormMaskReluFp32() {
    createPipeline("BatchNormMaskReluFp32", VkSPIRVShaders::spirv_bn_mask_relu_fp32, VkSPIRVShaders::spirv_bn_mask_relu_fp32_size, 5, sizeof(KatagoVulkan::BatchNormMaskParams), batchNormMaskReluFp32);
  }

  void ComputePipelines::createBatchNormMaskMishFp32() {
    createPipeline("BatchNormMaskMishFp32", VkSPIRVShaders::spirv_bn_mask_mish_fp32, VkSPIRVShaders::spirv_bn_mask_mish_fp32_size, 5, sizeof(KatagoVulkan::BatchNormMaskParams), batchNormMaskMishFp32);
  }

  void ComputePipelines::createBatchNormMaskMishScale8Fp32() {
    createPipeline("BatchNormMaskMishScale8Fp32", VkSPIRVShaders::spirv_bn_mask_mish_scale8_fp32, VkSPIRVShaders::spirv_bn_mask_mish_scale8_fp32_size, 5, sizeof(KatagoVulkan::BatchNormMaskParams), batchNormMaskMishScale8Fp32);
  }

  void ComputePipelines::createGlobalPoolingChannelsFp32() {
    createPipeline("GlobalPoolingChannelsFp32", VkSPIRVShaders::spirv_global_pooling_channels_fp32, VkSPIRVShaders::spirv_global_pooling_channels_fp32_size, 4, sizeof(KatagoVulkan::GlobalPoolingChannelsParams), globalPoolingChannelsFp32);
  }

  void ComputePipelines::createValueHeadPoolingChannelsFp32() {
    createPipeline("ValueHeadPoolingChannelsFp32", VkSPIRVShaders::spirv_value_head_pool_channels_fp32, VkSPIRVShaders::spirv_value_head_pool_channels_fp32_size, 3, sizeof(KatagoVulkan::ValueHeadPoolingChannelsParams), valueHeadPoolingChannelsFp32);
  }

  void ComputePipelines::createSumChannelsFp32() {
    createPipeline("SumChannelsFp32", VkSPIRVShaders::spirv_sum_channels_fp32, VkSPIRVShaders::spirv_sum_channels_fp32_size, 2, sizeof(KatagoVulkan::SumChannelsParams), sumChannelsFp32);
  }

  void ComputePipelines::createAddChannelBiasNCHWFp32() {
    createPipeline("AddChannelBiasNCHWFp32", VkSPIRVShaders::spirv_add_channel_bias_nchw_fp32, VkSPIRVShaders::spirv_add_channel_bias_nchw_fp32_size, 2, sizeof(KatagoVulkan::AddChannelBiasNCHWParams), addChannelBiasNCHWFp32);
  }

  void ComputePipelines::createAddChannelBiasNCIdentityFp32() {
    createPipeline("AddChannelBiasNCIdentityFp32", VkSPIRVShaders::spirv_add_channel_bias_nc_identity_fp32, VkSPIRVShaders::spirv_add_channel_bias_nc_identity_fp32_size, 2, sizeof(KatagoVulkan::AddChannelBiasNCParams), addChannelBiasNCIdentityFp32);
  }

  void ComputePipelines::createAddChannelBiasNCReluFp32() {
    createPipeline("AddChannelBiasNCReluFp32", VkSPIRVShaders::spirv_add_channel_bias_nc_relu_fp32, VkSPIRVShaders::spirv_add_channel_bias_nc_relu_fp32_size, 2, sizeof(KatagoVulkan::AddChannelBiasNCParams), addChannelBiasNCReluFp32);
  }

  void ComputePipelines::createAddChannelBiasNCMishFp32() {
    createPipeline("AddChannelBiasNCMishFp32", VkSPIRVShaders::spirv_add_channel_bias_nc_mish_fp32, VkSPIRVShaders::spirv_add_channel_bias_nc_mish_fp32_size, 2, sizeof(KatagoVulkan::AddChannelBiasNCParams), addChannelBiasNCMishFp32);
  }

  void ComputePipelines::createAddChannelBiasNCMishScale8Fp32() {
    createPipeline("AddChannelBiasNCMishScale8Fp32", VkSPIRVShaders::spirv_add_channel_bias_nc_mish_scale8_fp32, VkSPIRVShaders::spirv_add_channel_bias_nc_mish_scale8_fp32_size, 2, sizeof(KatagoVulkan::AddChannelBiasNCParams), addChannelBiasNCMishScale8Fp32);
  }
  
  void ComputePipelines::createExtractChannel0NCHWFp32() {
    createPipeline("ExtractChannel0NCHWFp32", VkSPIRVShaders::spirv_extract_channel0_nchw_fp32, VkSPIRVShaders::spirv_extract_channel0_nchw_fp32_size, 2, sizeof(KatagoVulkan::ExtractChannel0NCHWParams), extractChannel0NCHWFp32);
  }

  // ########################### End of Compute Pipelines #########################
}


#endif