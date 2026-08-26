/**
 * @file vulkanbackend.cpp
 * @author Dohoon Kim(https://github.com/dhkim92-dev)
 * @brief Vulkan backend for Neural Net evaluation
 */
#ifdef USE_VULKAN_BACKEND

#include <unordered_map>
#include <memory>
#include <chrono>
#include "../core/global.h"
#include "../core/simpleallocator.h"
#include "../core/test.h"
#include "../core/using.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/vulkancompute.h"
#include "../neuralnet/vulkanbackend.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkanshaders.h"
#include "../neuralnet/vulkantuner.h"

int globalBatchCount = 4;
using namespace vk_shader;
using namespace vk_shader::tune;
using namespace vk_shader::push;

std::vector<float> makeInputDataFromFile(const std::string& filePath)
{
    std::ifstream inFile(filePath);
    if (!inFile) {
        throw StringError("Failed to open input data file: " + filePath);
    }

    std::string line;
    std::getline(inFile, line);

    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);

    if (line.size() < 2 || line.front() != '[' || line.back() != ']') {
        throw StringError("Invalid format. Must start with '[' and end with ']'");
    }

    std::string dataStr = line.substr(1, line.size() - 2);
    std::istringstream ss(dataStr);

    std::vector<float> data;
    std::string token;

    while (std::getline(ss, token, ',')) {
        try {
            data.push_back(std::stof(token));
        }
        catch (const std::exception& e) {
            throw StringError("Failed to parse float from token: " + token);
        }
    }

    return data;
}

#ifdef VULKAN_API_DEBUG
#define VK_BENCHMARK(msg, code) \
  {  \
    auto start = std::chrono::high_resolution_clock::now(); \
    code \
    auto end = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); \
    std::cout << "[Vulkan Benchmark] " << msg << " took " << duration << " microseconds." << std::endl; \
  }
#else
#define VK_BENCHMARK(msg, code) code
#endif

static void printHostBuffer(
  std::string prefix,
  const float* hostBuffer,
  size_t numElts,
  bool summarized = true
) {
#ifdef VULKAN_API_DEBUG
  // print prefix first
  std::cout << prefix << " = " << std::endl;
  // print vector as python format, that can copy it to python code
  std::cout << "[";
  int limits = summarized ? 10 : numElts;
  limits  = limits > numElts ? numElts : limits;
  for ( size_t i = 0 ; i < limits ; i++ ) {
    std::cout << hostBuffer[i];
    if ( i != limits - 1 ) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
#endif
}

static void printDeviceBuffer(
  std::string prefix,
  const VulkanDevice* device,
  VulkanBuffer* buffer,
  size_t numElts,
  bool summarized = true
) {
#ifdef VULKAN_API_DEBUG

if ( globalBatchCount > 1 ) {
  VkResult res;
  std::vector<float> hostBuffer(numElts);
  vk_helper::copyDeviceBufferToHost(
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

  // int limits = summarized ? 10 : numElts;
  // limits  = limits > numElts ? numElts : limits;
  // limits = summary?
  size_t limits = summarized ? std::min(hostBuffer.size(), size_t(10)) : hostBuffer.size();

  for ( size_t i = 0 ; i < limits ; i++ ) {
    std::cout << hostBuffer[i];
    if ( i != limits - 1 ) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}
#endif
}

static void printDeviceBufferIndices(
  std::string prefix,
  const VulkanDevice* device,
  VulkanBuffer* buffer,
  size_t numElts,
  const std::vector<size_t>& indices
) {
#ifdef VULKAN_API_DEBUG
  VkResult res;
  std::vector<float> hostBuffer(numElts);
  vk_helper::copyDeviceBufferToHost(
    device,
    buffer,
    numElts * sizeof(float),
    hostBuffer.data(),
    true,
    &res
  );
  CHECK_VK_MSG("printDeviceBufferIndices copyDeviceToHostBuffer", res);

  std::cout << prefix << " = " << std::endl;
  std::cout << "[";
  for ( size_t i = 0 ; i < indices.size() ; ++i ) {
    size_t idx = indices[i];
    if ( idx < hostBuffer.size() ) std::cout << hostBuffer[idx];
    else std::cout << "<OOB>";
    if ( i + 1 != indices.size() ) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
#endif
}

struct ComputeContext {
  std::vector<uint32_t> gIdx;
  const int nnXLen;
  const int nnYLen;
  const enabled_t usingFP16Mode;
  const enabled_t usingNHWCMode;
  VulkanContext* vulkanContext;
  std::unordered_map<uint32_t, vk_shader::ComputePipelines *> pipelinesPerDev;
  Logger* logger;

  ComputeContext(
    int nnXLen,
    int nnYLen,
    enabled_t useFP16Mode_,
    enabled_t useNHWCMode_,
    const std::vector<uint32_t>& gpuIdxsToUse,
    Logger* logger_,
    const std::string& tunerFile,
    const std::string& homeDataDirOverride,
    const VulkanTuner::ModelInfoForTuning* modelInfo,
    bool fullTuning,
    bool forceRetune)
  : nnXLen(nnXLen),
    nnYLen(nnYLen),
    usingFP16Mode(useFP16Mode_),
    usingNHWCMode(useNHWCMode_),
    gIdx(gpuIdxsToUse),
    logger(logger_)
     {
      VkInstance instance = vk_helper::createVulkanInstance();
      std::vector<VulkanDeviceInfo> allDeviceInfos = vk_helper::enumerateVulkanDevices(instance, logger);
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

        VulkanDevice* vulkanDevice = vk_helper::createVulkanDevice(
          instance,
          deviceInfo,
          requiredExtensions,
          logger
        );
        vk_shader::ComputePipelines* pipelines = nullptr;
        try {
          VulkanTuneParams tuneParams;
          if(modelInfo != nullptr) {
            const auto recreateDevice = [&]() {
              return vk_helper::createVulkanDevice(instance,deviceInfo,requiredExtensions,logger);
            };
            tuneParams = VulkanTuner::loadOrAutoTune(
              tunerFile,homeDataDirOverride,vulkanDevice,recreateDevice,
              nnXLen,nnYLen,*modelInfo,logger,fullTuning,forceRetune
            );
          }
          pipelines = new vk_shader::ComputePipelines(vulkanDevice->device,tuneParams);
        }
        catch(...) {
          delete vulkanDevice;
          throw;
        }
        vulkanDevices.push_back(vulkanDevice);
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
      vk_shader::ComputePipelines* pipelines = kv.second;
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


VkDeviceSize getRequiredMemorySize(const LoadedModel* loadedModel) {
  // For simplicity, return a fixed size for now.
  // In future, we can calculate based on model parameters.
  return static_cast<VkDeviceSize>(512) * 1024 * 1024; // 512 MB
}

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
#ifdef VULKAN_API_DEBUG
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
  nChannels =nChannels > 2 ? 2 : nChannels; // limit channels to print
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
#else
static void printFloatBuffer(
  std::string prefix,
  const float* buffer,
  size_t numElts,
  int batchSize,
  int nChannels,
  int nRows,
  int nCols
) {
  // Do nothing
}
#endif

/**
 * @brief Matrix Multiplication Layer
 */
struct MatmulLayer {
  ComputeHandleInternal *handle;
  const std::string name;
  const int inChannels;
  const int outChannels;

  VulkanBuffer* matBuf = nullptr;
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
      matBuf = vk_helper::createDeviceBufferWithData(
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
    #ifdef VULKAN_API_DEBUG
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] try to destroy");
    #endif
    if ( matBuf != nullptr ) {
      vk_helper::releaseVulkanBuffer(handle->vulkanDevice, matBuf);
      matBuf = nullptr;
    }
    #ifdef VULKAN_API_DEBUG
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] destroyed");
    #endif
  }

  void debug(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    VkCommandBuffer commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
    // printDeviceBuffer(name + " input", handle->vulkanDevice, input, batchSize * inChannels);
    VkResult res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for MatmulLayer: " + name, res);
    // doMatmulFp32(commandBuffer, batchSize, input, output);
    doBatchedXGEMMDirectFP32_MK_NK_MN(commandBuffer, batchSize, input, output);
    res = vk_helper::endCommandBuffer(commandBuffer);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {commandBuffer});
    CHECK_VK_MSG("End command buffer for MatmulLayer: " + name, res);
    std::printf("Debug MatmulLayer: %s, batchSize: %d, inChannels: %d, outChannels: %d\n", name.c_str(), batchSize, inChannels, outChannels);
    printDeviceBuffer(name + " Output : ", handle->vulkanDevice, output, batchSize * outChannels, true);
  }

  /**
   * @brief create command buffer and record for matmul layer
   * @param batchSize
   * @param input
   * @param output
   */
  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    assert(cb != VK_NULL_HANDLE);
    // doMatmulFp32(cb, batchSize, input, output);
    doBatchedXGEMMDirectFP32_MK_NK_MN(cb, batchSize, input, output);
    vk_helper::barrierCommandBufferForBuffer(cb, output);
    vk_helper::barrierCommandBuffer(cb);
  }

private:
  void doMatmulFp32(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    auto *pipelines = handle->context->pipelinesPerDev.at(gpuId);
    VkResult res;
    if ( descriptorSet == VK_NULL_HANDLE ) {
      descriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        pipelines->matmulFp32.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for MatmulLayer: " + name, res);
    }

    // update descriptor set
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, matBuf),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, output)
    };
    vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

    auto pushConstants = MatmulFp32Params();
    pushConstants.M = static_cast<uint32_t>(batchSize);
    pushConstants.K = static_cast<uint32_t>(inChannels);
    pushConstants.N = static_cast<uint32_t>(outChannels);
    pushConstants.numBatchElts = 1;
    pushConstants.cTranspose = 1; // Weights always transposed, so Output should be transposed.

    vkCmdPushConstants(
      cb,
      pipelines->matmulFp32.layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(MatmulFp32Params),
      &pushConstants
    );
    vkCmdBindPipeline(
      cb,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipelines->matmulFp32.pipeline
    );
    vkCmdBindDescriptorSets(
      cb,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipelines->matmulFp32.layout,
      0,
      1,
      &descriptorSet,
      0,
      nullptr
    );

    // Each invocation computes a fixed 4x4 register tile, so one workgroup covers
    // four times its specialized local size in each output dimension.
    // HLSL: GId.x = groupM (M axis), GId.y = groupN (N axis)
    const uint32_t matmulTileX = pipelines->matmulFp32.localSizeX * 4u;
    const uint32_t matmulTileY = pipelines->matmulFp32.localSizeY * 4u;
    uint32_t wgCountX = (static_cast<uint32_t>(batchSize) + matmulTileX - 1u) / matmulTileX;
    uint32_t wgCountY = (static_cast<uint32_t>(outChannels) + matmulTileY - 1u) / matmulTileY;
    uint32_t wgCountZ = 1u;
    SHADER_PROFILE_START("MATMUL_FP32", cb);
    vkCmdDispatch(cb, wgCountX, wgCountY, wgCountZ);
    SHADER_PROFILE_END("MATMUL_FP32", cb);
  }

  void doBatchedXGEMMDirectFP32_MK_NK_MN(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    auto pipelines = handle->context->pipelinesPerDev.at(gpuId);
    VkResult res;
    SHADER_PROFILE_START("BATCHED_XGEMM_DIRECT_FP32", cb);
    Pipeline pipeline = pipelines->batchedXgemmDirect;
    if( descriptorSet == VK_NULL_HANDLE ) {
      descriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        pipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for BatchedXGEMMDirectFP32_MK_NK_MN MatmulLayer: " + name, res);
    }
    vkcompute::batchedXGemmDirect_MK_NK_MN(
      handle->vulkanDevice,
      handle->tuneParams,
      &pipeline,
      cb,
      descriptorSet,
      batchSize, outChannels, inChannels,
      input, matBuf, output,
      1, &res
    );
    SHADER_PROFILE_END("BATCHED_XGEMM_DIRECT_FP32", cb);
    // vk_helper::barrierCommandBufferForBuffer(cb, output);
    vk_helper::barrierCommandBuffer(cb);
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

  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

  ~BatchNormLayer() {
    #ifdef VULKAN_API_DEBUG
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] try to destroy");
    #endif
    if ( mergedScaleBuf != nullptr ) {
      vk_helper::releaseVulkanBuffer(mergedScaleBuf->device, mergedScaleBuf);
      // delete mergedScaleBuf;
      mergedScaleBuf = nullptr;
    }
    if ( mergedBiasBuf != nullptr ) {
      vk_helper::releaseVulkanBuffer(mergedBiasBuf->device, mergedBiasBuf);
      // delete mergedBiasBuf;
      mergedBiasBuf = nullptr;
    }
    #ifdef VULKAN_API_DEBUG
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] destroyed");
    #endif
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

    mergedBiasBuf = vk_helper::createDeviceBufferWithData(
      handle->vulkanDevice,
      sizeof(float) * mergedBias.size(),
      mergedBias.data(),
      true,
      &res
    );
    CHECK_VK_MSG("Create BatchNormLayer: " + name + " merged bias buffer", res);

    mergedScaleBuf = vk_helper::createDeviceBufferWithData(
      handle->vulkanDevice,
      sizeof(float) * mergedScale.size(),
      mergedScale.data(),
      true,
      &res
    );
    CHECK_VK_MSG("Create BatchNormLayer: " + name + " merged scale buffer", res);

  }

  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* mask,
    VulkanBuffer* output
  ) {
    assert(cb != VK_NULL_HANDLE);
    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    vk_shader::ComputePipelines* pipelines = this->handle->context->pipelinesPerDev.at(gpuId);
    Pipeline targetPipeline;
    VkResult res = VK_ERROR_UNKNOWN;

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

    if (descriptorSet == VK_NULL_HANDLE ) {
      descriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        targetPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for BatchNormLayer: " + name, res);
    }
      // update descriptor set
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, mask),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, mergedScaleBuf),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 3, mergedBiasBuf),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 4, output)
    };
    vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

    BatchNormMaskParams pushConstants = {};
    pushConstants.batchSize = static_cast<uint32_t>(batchSize);
    pushConstants.numChannels = static_cast<uint32_t>(numChannels);
    pushConstants.nnXYLen = static_cast<uint32_t>(nnXYLen);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.layout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cb, targetPipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BatchNormMaskParams), &pushConstants);
    // Dispatch: x=spatial, y=channels, z=batch (optimized for memory coalescing in NCHW layout)
    const uint32_t localSizeX = targetPipeline.localSizeX;
    const uint32_t localSizeY = targetPipeline.localSizeY;
    uint32_t wgCountX = (static_cast<uint32_t>(nnXYLen) + localSizeX - 1u) / localSizeX;
    uint32_t wgCountY = (static_cast<uint32_t>(numChannels) + localSizeY - 1u) / localSizeY;
    uint32_t wgCountZ = static_cast<uint32_t>(batchSize);
    SHADER_PROFILE_START("BATCHNORM_MASK_FP32", cb);
    vkCmdDispatch(cb, wgCountX, wgCountY, wgCountZ);
    SHADER_PROFILE_END("BATCHNORM_MASK_FP32", cb);
    vk_helper::barrierCommandBufferForBuffer(cb, output);
    // res = vk_helper::endCommandBuffer(cb);

    // CHECK_VK_MSG("End command buffer for BatchNormLayer: " + name, res);
  }

  /**
   * @brief Launch the recorded command buffer, only for debug now.
   * @param batchSize
   * @param input
   * @param mask
   * @param output
   */
  void debug(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* mask,
    VulkanBuffer* output
  ) {
    VkCommandBuffer commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
    VkResult res = vk_helper::beginCommandBuffer(commandBuffer);
    std::vector<float> inputCopy(static_cast<size_t>(batchSize) * static_cast<size_t>(numChannels) * static_cast<size_t>(nnXYLen));
    forward(commandBuffer, batchSize, input, mask, output);
    res = vk_helper::endCommandBuffer(commandBuffer);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {commandBuffer});
    printDeviceBuffer(name + " Output : ", handle->vulkanDevice, output, static_cast<size_t>(batchSize) * static_cast<size_t>(numChannels) * static_cast<size_t>(nnXYLen));
  }
};


/**
 * @brief Convolution Layer in Vulkan Backend
 * Currently not support winograd and dilation
 * Simple tiled convolution only except 1x1 conv.
 * 1x1 conv implemented with matmul approach. Maybe replaced by cooperative matrix extension later.
 */
struct ConvLayer {
  ComputeHandleInternal* handle;
  const std::string name;
  const int convYSize;
  const int convXSize;
  const int inChannels;
  const int outChannels;
  const int dilationY;
  const int dilationX;
  const int nnXLen;
  const int nnYLen;

  int inTilesY;
  int inTilesX;
  int outTilesY;
  int outTilesX;
  int numTilesX;
  int numTilesY;
  int inTilesXYSize;
  int outTilesXYSize;
  int numTilesTotal;

  VulkanBuffer* filterBuf = nullptr;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

  VkDescriptorSet winogradInputTransformDS = VK_NULL_HANDLE;
  VkDescriptorSet winogradOutputTransformDS = VK_NULL_HANDLE;
  VkDescriptorSet xgemmBatchedDS = VK_NULL_HANDLE;

  VulkanBuffer* bnScaleBuf = nullptr; // For batchnorm scale
  VulkanBuffer* bnBiasBuf = nullptr;  // For batchnorm bias

  static constexpr int nKernelDims = 3;
  uint32_t act;

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
    numTilesX = 0;
    numTilesY = 0;
    numTilesTotal = 0;
    inTilesXYSize = 0;
    outTilesXYSize = 0;

    // if ( convYSize == 3 || convYSize == 5 ) {
      // outTilesY = convYSize == 3 ? handle->tuneParams.conv3x3.outTileYSize : handle->tuneParams.conv5x5.outTileYSize;
      // outTilesX = convYSize == 3 ? handle->tuneParams.conv3x3.outTileXSize : handle->tuneParams.conv5x5.outTileXSize;
      // numTilesY = (nnYLen + outTilesY - 1) / outTilesY;
      // numTilesX = (nnXLen + outTilesX - 1) / outTilesX;
      // inTilesX = outTilesX + convXSize - 1;
      // inTilesY = outTilesY + convYSize - 1;
      // inTilesXYSize = inTilesY * inTilesX;
      // outTilesXYSize = outTilesY * outTilesX;
      // numTilesTotal = numTilesX * numTilesY;
    // }

    // For 1x1 conv, transpose weights from [OC][IC] to [IC][OC] for KM_KN_NM matmul convention
    // This matches OpenCL's transWeights layout for 1x1 convolutions
    int inChannelsPadded = vk_helper::roundUpToMultipleInt(inChannels, handle->getXgemmKPaddingMult());
    int outChannelsPadded = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
    if (convXSize == 1 && convYSize == 1) {
      std::vector<float> transWeights(inChannels * outChannels);
      for (int oc = 0; oc < outChannels; oc++) {
        for (int ic = 0; ic < inChannels; ic++) {
          transWeights[ic * outChannels + oc] = desc->weights[oc * inChannels + ic];
        }
      }
      filterBuf = vk_helper::createDeviceBufferWithData(
        handle->vulkanDevice,
        sizeof(float) * transWeights.size(),
        transWeights.data(),
        true,
        &res
      );
    } else if(convXSize == 3 && convYSize == 3) {
      outTilesY = handle->tuneParams.conv3x3.outTileYSize;
      outTilesX = handle->tuneParams.conv3x3.outTileXSize;
      inTilesY = handle->tuneParams.conv3x3.inTileYSize;
      inTilesX = handle->tuneParams.conv3x3.inTileXSize;
      inTilesXYSize = inTilesY * inTilesX;
      outTilesXYSize = outTilesY * outTilesX;
      numTilesX = (nnXLen + outTilesX - 1) / outTilesX;
      numTilesY = (nnYLen + outTilesY - 1) / outTilesY;
      numTilesTotal = numTilesX * numTilesY;
      std::vector<float> winograd3x3Weights = vkcompute::convWeightsToWinogradDomain(
        desc->weights, inChannels, inChannelsPadded,
        outChannels, outChannelsPadded,
        convYSize, convXSize, 
        inTilesY, inTilesX
      );

      filterBuf = vk_helper::createDeviceBufferWithData(
        handle->vulkanDevice,
        sizeof(float) * winograd3x3Weights.size(),
        winograd3x3Weights.data(),
        true,
        &res
      );
    } else if (convXSize == 5 && convYSize == 5) {
      outTilesY = handle->tuneParams.conv5x5.outTileYSize;
      outTilesX = handle->tuneParams.conv5x5.outTileXSize;
      inTilesY = handle->tuneParams.conv5x5.inTileYSize;
      inTilesX = handle->tuneParams.conv5x5.inTileXSize;
      inTilesXYSize = inTilesY * inTilesX;
      outTilesXYSize = outTilesY * outTilesX;
      numTilesX = (nnXLen + outTilesX - 1) / outTilesX;
      numTilesY = (nnYLen + outTilesY - 1) / outTilesY;
      numTilesTotal = numTilesX * numTilesY;
      std::vector<float> winograd5x5Weights = vkcompute::convWeightsToWinogradDomain(
        desc->weights, inChannels, inChannelsPadded,
        outChannels, outChannelsPadded,
        convYSize, convXSize, 
        inTilesY, inTilesX
      );
      filterBuf = vk_helper::createDeviceBufferWithData(
        handle->vulkanDevice,
        sizeof(float) * winograd5x5Weights.size(),
        winograd5x5Weights.data(),
        true,
        &res
      );
    } else {
      // For larger convolutions, use weights as-is
      filterBuf = vk_helper::createDeviceBufferWithData(
        handle->vulkanDevice,
        sizeof(float) * desc->weights.size(),
        desc->weights.data(),
        true,
        &res
      );
    }
    CHECK_VK_MSG("Create ConvLayer: " + name + " filter buffer", res);
  }

  ~ConvLayer() {
    #ifdef VULKAN_API_DEBUG
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] try to destroy");
    #endif
    if ( filterBuf != nullptr ) {
      vk_helper::releaseVulkanBuffer(handle->vulkanDevice, filterBuf);
      filterBuf = nullptr;
    }
    #ifdef VULKAN_API_DEBUG
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] destroyed");
    #endif
  }

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  bool isBNActFusedPossible() {
    return (convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5);
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) {
    int numTilesTotalPadded = vk_helper::roundUpToMultipleInt(maxBatchSize * numTilesY * numTilesX, handle->getXGemmMPaddingMult());
    int outChannelsPadded = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
    int inChannelsPadded = vk_helper::roundUpToMultipleInt(inChannels, handle->getXgemmKPaddingMult());

    return ConvWorkspaceEltsNeeded {
      static_cast<size_t>(numTilesTotalPadded) * static_cast<size_t>(inChannelsPadded) * static_cast<size_t>(inTilesXYSize),
      static_cast<size_t>(numTilesTotalPadded) * static_cast<size_t>(outChannelsPadded) * static_cast<size_t>(inTilesXYSize)
    };
  }

  void doConv2DTiledBnActFp32(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* bnScale,
    VulkanBuffer* bnBias,
    VulkanBuffer* mask,
    uint32_t activation
  ) {
    VkResult res = VK_ERROR_UNKNOWN;
    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    vk_shader::ComputePipelines* pipelines = this->handle->context->pipelinesPerDev.at(gpuId);
    std::string shaderName = (convXSize == 3 && convYSize == 3) ? "Conv2DTiledBnAct3x3Fp32" :
                             (convXSize == 5 && convYSize == 5) ? "Conv2DTiledBnAct5x5Fp32" :
                             "Unsupported";

    auto targetPipeline = (convXSize == 3 && convYSize == 3) ?
                          pipelines->conv2dTiledBnAct3x3Fp32 :
                          (convXSize == 5 && convYSize == 5) ?
                          pipelines->conv2dTiledBnAct5x5Fp32 : 
                          throw StringError("Conv2DTiledBnActFp32: Unsupported conv size " + std::to_string(convXSize) + "x" + std::to_string(convYSize) + " in layer " + name);

    if ( descriptorSet == VK_NULL_HANDLE ) {
      descriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        targetPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for ConvLayer: " + name, res);
    }

    // update descriptor set
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, filterBuf),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, output),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 3, bnScale),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 4, bnBias),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 5, mask)
    };
    vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.pipeline);
    vkCmdBindDescriptorSets(
      cb,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      targetPipeline.layout,
      0,
      1,
      &descriptorSet,
      0,
      nullptr
    );
    auto pushConstants = Conv2DTiledBnActParams();
    pushConstants.batchSize = static_cast<uint32_t>(batchSize);
    pushConstants.inChannels = static_cast<uint32_t>(inChannels);
    pushConstants.outChannels = static_cast<uint32_t>(outChannels);
    pushConstants.nnYLen = static_cast<uint32_t>(nnYLen);
    pushConstants.nnXLen = static_cast<uint32_t>(nnXLen);
    pushConstants.filterH = static_cast<uint32_t>(convYSize);
    pushConstants.filterW = static_cast<uint32_t>(convXSize);
    pushConstants.activation = activation;

    vkCmdPushConstants(cb, targetPipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Conv2DTiledBnActParams), &pushConstants);

    const uint32_t outputTileY = 8u;
    uint32_t wgCountX = (pushConstants.nnXLen + targetPipeline.localSizeX - 1u) / targetPipeline.localSizeX;
    uint32_t wgCountY = (pushConstants.nnYLen + outputTileY - 1u) / outputTileY;
    uint32_t ocGroupsPerBatch = (pushConstants.outChannels + targetPipeline.localSizeY - 1u) / targetPipeline.localSizeY;
    uint32_t wgCountZ = pushConstants.batchSize * ocGroupsPerBatch;

    SHADER_PROFILE_START(shaderName, cb);
    vkCmdDispatch(cb, wgCountX, wgCountY, wgCountZ);
    SHADER_PROFILE_END(shaderName, cb);
    vk_helper::barrierCommandBufferForBuffer(cb, output);
    vk_helper::barrierCommandBuffer(cb);
  }

  void doConv2DTiledFp32(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    // Implement convolution logic here if needed
    VkResult res;
    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    vk_shader::ComputePipelines* pipelines = this->handle->context->pipelinesPerDev.at(gpuId);

    if ( descriptorSet == VK_NULL_HANDLE ) {
      descriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        pipelines->conv2dFp32.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for ConvLayer: " + name, res);
    }
    // update descriptor set
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, filterBuf),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, output)
    };
    vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines->conv2dFp32.pipeline);
    vkCmdBindDescriptorSets(
      cb,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipelines->conv2dFp32.layout,
      0,
      1,
      &descriptorSet,
      0,
      nullptr
    );
    auto pushConstants = Conv2DPushConstantParams();
    pushConstants.batchSize = static_cast<uint32_t>(batchSize);
    pushConstants.inChannels = static_cast<uint32_t>(inChannels);
    pushConstants.outChannels = static_cast<uint32_t>(outChannels);
    pushConstants.filterH = static_cast<uint32_t>(convYSize);
    pushConstants.filterW = static_cast<uint32_t>(convXSize);
    pushConstants.nnXLen = static_cast<uint32_t>(nnXLen);
    pushConstants.nnYLen = static_cast<uint32_t>(nnYLen);
    vkCmdPushConstants(
      cb,
      pipelines->conv2dFp32.layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(Conv2DPushConstantParams),
      &pushConstants
    );

    uint32_t wgCountX = (pushConstants.nnXLen + pipelines->conv2dFp32.localSizeX - 1u) / pipelines->conv2dFp32.localSizeX;
    uint32_t wgCountY = (pushConstants.nnYLen + pipelines->conv2dFp32.localSizeY - 1u) / pipelines->conv2dFp32.localSizeY;
    uint32_t ocGroupsPerBatch = (pushConstants.outChannels + pipelines->conv2dFp32.localSizeY - 1u) / pipelines->conv2dFp32.localSizeY;
    uint32_t wgCountZ = pushConstants.batchSize * ocGroupsPerBatch;
    SHADER_PROFILE_START("CONV2D_TILED_FP32", cb);
    vkCmdDispatch(cb, wgCountX, wgCountY, wgCountZ);
    SHADER_PROFILE_END("CONV2D_TILED_FP32", cb);
    vk_helper::barrierCommandBufferForBuffer(cb, output);
    vk_helper::barrierCommandBuffer(cb);
  }

  void doConv1x1AsMatmulFp32(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    VkResult res;
    Pipeline targetPipeline = handle->context->pipelinesPerDev.at(handle->vulkanDevice->info.deviceId)->xgemmStridedBatchedFp32;

    if ( descriptorSet == VK_NULL_HANDLE ) {
      descriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        targetPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for ConvLayer: " + name, res);
    }

    int filterStride = 0;
    int inputStride = nnXLen * nnYLen * inChannels;
    int outputStride = nnXLen * nnYLen * outChannels;

    vkcompute::xgemmStridedBatchedNN(
      handle->vulkanDevice,
      handle->tuneParams,
      &targetPipeline,
      cb,
      descriptorSet,
      nnXLen * nnYLen, outChannels, inChannels,
      inputStride, filterStride, outputStride,
      input, filterBuf, output,
      static_cast<uint32_t>(batchSize), &res
    );
    vk_helper::barrierCommandBufferForBuffer(cb, output);
    vk_helper::barrierCommandBuffer(cb);
  }

  void doWinogradConvolutionBnActMask(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* convWorkspace1,
    VulkanBuffer* convWorkspace2,
    VulkanBuffer* output,
    VulkanBuffer* bnScale,
    VulkanBuffer* bnBias,
    VulkanBuffer* mask,
    int activation
  ) {
    // Implement convolution logic here if needed
    VkResult res = VK_ERROR_UNKNOWN;
    auto *pipelines = this->handle->context->pipelinesPerDev.at(handle->vulkanDevice->info.deviceId);
    Pipeline winogradInputTransformBnActMaskPipeline;
    Pipeline xgemmBatchedPipeline = pipelines->xgemmBatchedFp32;
    Pipeline winogradOutputTransformPipeline = (convXSize == 3 && convYSize == 3) ? pipelines->winogradOutputTransform3x3 
                                         : (convXSize == 5 && convYSize == 5) ? pipelines->winogradOutputTransform5x5
                                         : throw StringError("Winograd convolution only supported for 3x3 and 5x5 kernels in layer " + name);

    switch(activation) {
      case ACTIVATION_IDENTITY: 
        winogradInputTransformBnActMaskPipeline = (convXSize == 3 && convYSize == 3) ? pipelines->winogradInputTransform3x3_bnact_identity
                        : (convXSize == 5 && convYSize == 5) ? pipelines->winogradInputTransform5x5_bnact_identity
                        : throw StringError("Unsupported conv size for fused Winograd convolution in layer " + name);
        break;
      case ACTIVATION_RELU:
        winogradInputTransformBnActMaskPipeline = (convXSize == 3 && convYSize == 3) ? pipelines->winogradInputTransform3x3_bnact_relu
                        : (convXSize == 5 && convYSize == 5) ? pipelines->winogradInputTransform5x5_bnact_relu
                        : throw StringError("Unsupported conv size for fused Winograd convolution in layer " + name);
        break;
      case ACTIVATION_MISH:
        winogradInputTransformBnActMaskPipeline = (convXSize == 3 && convYSize == 3) ? pipelines->winogradInputTransform3x3_bnact_mish
                        : (convXSize == 5 && convYSize == 5) ? pipelines->winogradInputTransform5x5_bnact_mish
                        : throw StringError("Unsupported conv size for fused Winograd convolution in layer " + name);
        break;
      case ACTIVATION_MISH_SCALE8:
        winogradInputTransformBnActMaskPipeline = (convXSize == 3 && convYSize == 3) ? pipelines->winogradInputTransform3x3_bnact_mish_scale8
                        : (convXSize == 5 && convYSize == 5) ? pipelines->winogradInputTransform5x5_bnact_mish_scale8
                        : throw StringError("Unsupported conv size for fused Winograd convolution in layer " + name);
        break;
      default:
        throw StringError("Unsupported activation for fused Winograd convolution in layer " + name);
    }

    if ( winogradInputTransformDS == VK_NULL_HANDLE ) {
      winogradInputTransformDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        winogradInputTransformBnActMaskPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for ConvLayer: " + name, res);
    }

    if ( winogradOutputTransformDS == VK_NULL_HANDLE ) {
      winogradOutputTransformDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        winogradOutputTransformPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for ConvLayer: " + name, res);
    }

    if ( xgemmBatchedDS == VK_NULL_HANDLE ) {
      xgemmBatchedDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        xgemmBatchedPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for ConvLayer: " + name, res);
    }

    // first winograd input transform with fused bn+act and require barrier 
    {
      SHADER_PROFILE_START("WINOGRAD_INPUT_TRANSFORM_BN_ACT_MASK", cb);
      vkcompute::convInputToWinogradDomainBnActMask(
        handle->vulkanDevice,
        handle->tuneParams,
        &winogradInputTransformBnActMaskPipeline,
        cb,
        winogradInputTransformDS,
        input,
        convWorkspace1,
        bnScale,
        bnBias,
        mask,
        nnYLen, nnXLen,
        batchSize, numTilesY, numTilesX, handle->getXGemmMPaddingMult(),
        inChannels, handle->getXgemmKPaddingMult(),
        convYSize,
        &res
      );
      SHADER_PROFILE_END("WINOGRAD_INPUT_TRANSFORM_BN_ACT_MASK", cb);
    }

    vk_helper::barrierCommandBuffer(cb);
    vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace1);
    // Then xgemm and winograd output transform same as before
    {
      SHADER_PROFILE_START("WINOGRAD_GEMM", cb);
      // {
        // uint32_t dbg_numTilesTotal = vk_helper::roundUpToMultipleInt(numTilesTotal, handle->getXGemmMPaddingMult());
        // uint32_t dbg_outCh = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
        // uint32_t dbg_inCh = vk_helper::roundUpToMultipleInt(inChannels, handle->getXgemmKPaddingMult());
        // std::printf("[xGEMM] Before dispatch (BNAct path) numTilesTotal=%u outChPadded=%u inChPadded=%u inTilesXYSize=%d descriptorSet=%p\n",
          // dbg_numTilesTotal, dbg_outCh, dbg_inCh, this->inTilesXYSize, (void*)xgemmBatchedDS);
      // }
      vkcompute::xgemmBatched(
        handle->vulkanDevice,
        handle->tuneParams,
        &xgemmBatchedPipeline,
        cb,
        xgemmBatchedDS,
        vk_helper::roundUpToMultipleInt(batchSize * numTilesTotal, handle->getXGemmMPaddingMult()), 
        vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult()), 
        vk_helper::roundUpToMultipleInt(inChannels, handle->getXgemmKPaddingMult()),
        convWorkspace1,
        filterBuf,
        convWorkspace2,
        this->inTilesXYSize,
        &res
      );
      SHADER_PROFILE_END("WINOGRAD_GEMM", cb);
    }

    // Barrier before output transform
    vk_helper::barrierCommandBuffer(cb);
    vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace2);
    // Output transform is the same as before, since the fused bn+act is only on input transform side and doesn't change the data layout for xgemm
    {
      SHADER_PROFILE_START("WINOGRAD_OUTPUT_TRANSFORM", cb);
      vkcompute::winogradOutputToSpatialDomain(
        handle->vulkanDevice,
        &winogradOutputTransformPipeline,
        cb,
        winogradOutputTransformDS,
        convWorkspace2,
        output,
        nnYLen, nnXLen,
        batchSize, numTilesY, numTilesX, handle->getXGemmMPaddingMult(),
        outChannels, handle->getXGemmNPaddingMult(),
        &res
      );
      SHADER_PROFILE_END("WINOGRAD_OUTPUT_TRANSFORM", cb);
    }
    vk_helper::barrierCommandBufferForBuffer(cb, output);
  }

  void doWinogradConvolution(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* convWorkspace1,
    VulkanBuffer* convWorkspace2,
    VulkanBuffer* output
  ) {
    VkResult res = VK_ERROR_UNKNOWN;
    const VulkanDevice* device = handle->vulkanDevice;
    const vk_shader::ComputePipelines *pipelines = handle->context->pipelinesPerDev.at(device->info.deviceId);

    Pipeline inputTransformPipeline = (convXSize == 3 && convYSize == 3) ? pipelines->winogradInputTransform3x3
                                         : (convXSize == 5 && convYSize == 5) ? pipelines->winogradInputTransform5x5
                                         : throw StringError("Winograd convolution only supported for 3x3 and 5x5 kernels in layer " + name);

    Pipeline outputTransformPipeline = (convXSize == 3 && convYSize == 3) ? pipelines->winogradOutputTransform3x3
                                         : (convXSize == 5 && convYSize == 5) ? pipelines->winogradOutputTransform5x5
                                         : throw StringError("Winograd convolution only supported for 3x3 and 5x5 kernels in layer " + name);

    Pipeline xgemmPipeline = pipelines->xgemmBatchedFp32;

    if (  winogradInputTransformDS == VK_NULL_HANDLE ) {
      winogradInputTransformDS = vk_helper::allocateDescriptorSet(
        device,
        inputTransformPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for Winograd input transform in ConvLayer: " + name, res);
    }

    if ( winogradOutputTransformDS == VK_NULL_HANDLE ) {
      winogradOutputTransformDS = vk_helper::allocateDescriptorSet(
        device,
        outputTransformPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for Winograd output transform in ConvLayer: " + name, res);
    }


    if(  xgemmBatchedDS == VK_NULL_HANDLE ) {
      xgemmBatchedDS = vk_helper::allocateDescriptorSet(
        device,
        xgemmPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for Winograd GEMM in ConvLayer: " + name, res);
    }
    // first winograd input transform and require barrier 
    {
      SHADER_PROFILE_START("WINOGRAD_INPUT_TRANSFORM", cb);
      vkcompute::convInputsToWinogradDomain(
        device,
        handle->tuneParams,
        &inputTransformPipeline,
        cb,
        winogradInputTransformDS,
        input,
        convWorkspace1,
        nnYLen, nnXLen,
        batchSize, numTilesY, numTilesX, handle->getXGemmMPaddingMult(),
        inChannels, handle->getXgemmKPaddingMult(),
        convYSize,
        &res
      );
      SHADER_PROFILE_END("WINOGRAD_INPUT_TRANSFORM", cb);
    }
    vk_helper::barrierCommandBuffer(cb);
    vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace1);

    uint32_t numTilesTotal = vk_helper::roundUpToMultipleInt(batchSize * numTilesY * numTilesX, handle->getXGemmMPaddingMult());
    uint32_t inChannelsPadded = vk_helper::roundUpToMultipleInt(inChannels, handle->getXgemmKPaddingMult());
    uint32_t outChannelsPadded = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
    {
      SHADER_PROFILE_START("WINOGRAD_GEMM", cb);
      // // std::printf("[xGEMM] Before dispatch numTilesTotal=%u outChPadded=%u inChPadded=%u inTilesXYSize=%d descriptorSet=%p\n",
      //   numTilesTotal, outChannelsPadded, inChannelsPadded, this->inTilesXYSize, (void*)xgemmBatchedDS);
      vkcompute::xgemmBatched(
        device,
        handle->tuneParams,
        &xgemmPipeline,
        cb,
        xgemmBatchedDS,
        numTilesTotal, outChannelsPadded, inChannelsPadded,
        convWorkspace1,
        filterBuf,
        convWorkspace2,
        this->inTilesXYSize,
        &res
      );
      SHADER_PROFILE_END("WINOGRAD_GEMM", cb);
    }
    vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace2);
    vk_helper::barrierCommandBuffer(cb);

    {
      SHADER_PROFILE_START("WINOGRAD_OUTPUT_TRANSFORM", cb);
      vkcompute::winogradOutputToSpatialDomain(
        device,
        &outputTransformPipeline,
        cb,
        winogradOutputTransformDS,
        convWorkspace2,
        output,
        nnYLen, nnXLen,
        batchSize, numTilesY, numTilesX, handle->getXGemmMPaddingMult(),
        outChannels, handle->getXGemmNPaddingMult(),
        &res
      );
      SHADER_PROFILE_END("WINOGRAD_OUTPUT_TRANSFORM", cb);
    }
    vk_helper::barrierCommandBufferForBuffer(cb, output);
  }

  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* convWorkspace1,
    VulkanBuffer* convWorkspace2
  ) {
    assert(cb != VK_NULL_HANDLE);
    if ( convXSize == 1 && convYSize == 1 ) {
      doConv1x1AsMatmulFp32(cb, batchSize,  input, output);
    } else if ( (convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5) ) {
      assert(convWorkspace1 != nullptr); 
      assert(convWorkspace2 != nullptr);
      doWinogradConvolution(cb, batchSize, input, convWorkspace1, convWorkspace2, output);
    } else {
      doConv2DTiledFp32(cb, batchSize,  input, output);
    }
  }

  void forwardBnActConv(
    VkCommandBuffer& cb,
    BatchNormLayer* bnLayer,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* convWorkspace1,
    VulkanBuffer* convWorkspace2,
    VulkanBuffer* mask
  ) {
    assert(cb != VK_NULL_HANDLE);
    assert(bnLayer != nullptr);
    assert(mask != nullptr);
    doWinogradConvolutionBnActMask(cb, batchSize, input, convWorkspace1, convWorkspace2, output, bnLayer->mergedScaleBuf, bnLayer->mergedBiasBuf, mask, bnLayer->activation);
    // doConv2DTiledBnActFp32(cb, batchSize, input, output, bnLayer->mergedScaleBuf, bnLayer->mergedBiasBuf, mask, bnLayer->activation);
  }

  /**
   * @brief Launch the recorded command buffer, only for debug now.
   * @param batchSize
   * @param input
   * @param output
   * @return VkCommandBuffer
   */
  void debug(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output,
    BatchNormLayer* bnLayer,
    VulkanBuffer* mask,
    VulkanBuffer* convWorkspace1,
    VulkanBuffer* convWorkspace2
  ) {

    std::printf("[%s] metadata batchSize=%d inCh=%d outCh=%d nnXLen=%d nnYLen=%d convX=%d convY=%d numTilesX=%d numTilesY=%d\n",
      name.c_str(), batchSize, inChannels, outChannels, nnXLen, nnYLen, convXSize, convYSize, numTilesX, numTilesY);
    VkCommandBuffer commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
    int icPadded = vk_helper::roundUpToMultipleInt(inChannels, handle->getXgemmKPaddingMult());
    int ocPadded = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
    int ntxtyPadded = vk_helper::roundUpToMultipleInt(batchSize * numTilesY * numTilesX, handle->getXGemmMPaddingMult());


    VkResult res = vk_helper::beginCommandBuffer(commandBuffer);
    if ( bnLayer != nullptr && mask != nullptr ) {
      forwardBnActConv(commandBuffer, bnLayer, batchSize, input, output, convWorkspace1, convWorkspace2, mask);
    } else {
      forward(commandBuffer, batchSize, input, output, convWorkspace1, convWorkspace2);
    }
    res = vk_helper::endCommandBuffer(commandBuffer);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {commandBuffer});


    // if ( bnLayer != nullptr ) {
    //   // norm act conv 
    //   if ( (convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5) ) {
    //       // printDeviceBuffer(name + " Raw Input : ", handle->vulkanDevice, input, static_cast<size_t>(batchSize) * static_cast<size_t>(inChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen), true);
    //       bool summarized = true;
    //       std::printf("%s BNACT Winograd Activation : %d\n", this->name.c_str(), this->act);
    //       printDeviceBuffer(name + " BNACT Winograd Input Transform Output : ", handle->vulkanDevice, convWorkspace1, static_cast<size_t>(ntxtyPadded) * static_cast<size_t>(icPadded) * static_cast<size_t>(inTilesXYSize), summarized);
    //       printDeviceBuffer(name + " BNACT Winograd GEMM Output : ", handle->vulkanDevice, convWorkspace2, static_cast<size_t>(ntxtyPadded) * static_cast<size_t>(ocPadded) * static_cast<size_t>(inTilesXYSize),  summarized);
    //       printDeviceBuffer(name + " BNACT Winograd Output Transform Output : ", handle->vulkanDevice, output, static_cast<size_t>(batchSize) * static_cast<size_t>(outChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen), summarized);
    //   } else {
    //     bool summarized = true;
    //     // printDeviceBuffer(name + " Filter : ", handle->vulkanDevice, filterBuf, static_cast<size_t>(outChannels) * static_cast<size_t>(inChannels) * static_cast<size_t>(convYSize) * static_cast<size_t>(convXSize));
    //     printDeviceBuffer(name + " BNACT Output : ", handle->vulkanDevice, output, static_cast<size_t>(batchSize) * static_cast<size_t>(outChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen), summarized);
    //   }
    // } else {

    //   if ( (convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5) ) {
    //     icPadded = vk_helper::roundUpToMultipleInt(inChannels, handle->getXgemmKPaddingMult());
    //     ocPadded = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
    //     ntxtyPadded = vk_helper::roundUpToMultipleInt(batchSize * numTilesY * numTilesX, handle->getXGemmMPaddingMult());
    //     bool summarized = true;
    //     size_t szWinoFilters = inTilesXYSize * icPadded* ocPadded;
    //     // printDeviceBuffer(name + " Raw Input : ", handle->vulkanDevice, input, static_cast<size_t>(batchSize) * static_cast<size_t>(inChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen), printAll);
    //     // printDeviceBuffer(name + " Winograd Filter : ", handle->vulkanDevice, filterBuf,szWinoFilters, printAll);
    //     printDeviceBuffer(name + " Winograd Input Transform Output : ", handle->vulkanDevice, convWorkspace1, static_cast<size_t>(ntxtyPadded) * static_cast<size_t>(icPadded) * static_cast<size_t>(inTilesXYSize), summarized);
    //     printDeviceBuffer(name + " Winograd GEMM Output : ", handle->vulkanDevice, convWorkspace2, vk_helper::roundUpToMultipleInt(numTilesTotal, handle->getXGemmMPaddingMult()) * vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult()) * static_cast<size_t>(inTilesXYSize), summarized);
    //     printDeviceBuffer(name + " Winograd Output Transform Output : ", handle->vulkanDevice, output, static_cast<size_t>(batchSize) * static_cast<size_t>(outChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen), summarized);
    //   } else  {
    //     bool summarized = true;
    //     // printDeviceBuffer(name + " Raw Input : ", handle->vulkanDevice, input, static_cast<size_t>(batchSize) * static_cast<size_t>(inChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen), printAll);
    //     // printDeviceBuffer(name + " Filter : ", handle->vulkanDevice, filterBuf, static_cast<size_t>(outChannels) * static_cast<size_t>(inChannels) * static_cast<size_t>(convYSize) * static_cast<size_t>(convXSize), printAll);
    //     printDeviceBuffer(name + " 1x1 Conv Output : ", handle->vulkanDevice, output, static_cast<size_t>(batchSize) * static_cast<size_t>(outChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen), summarized);
    //     // exit(EXIT_FAILURE);
    //   }
    // }
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
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  static constexpr int nKernelDims = 2;
  float bias;

  ~MatBiasLayer() {
    #ifdef VULKAN_API_DEBUG
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] try to destroy");
    #endif
    if ( biasBuf != nullptr ) {
      vk_helper::releaseVulkanBuffer(handle->vulkanDevice, biasBuf);
      // delete biasBuf;
      biasBuf = nullptr;
    }
    #ifdef VULKAN_API_DEBUG
    if(handle && handle->context && handle->context->logger) handle->context->logger->write("[" + name + "] destroyed");
    #endif
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
      biasBuf = vk_helper::createDeviceBufferWithData(
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

  void forward(VkCommandBuffer& cb, int batchSize, VulkanBuffer* input) {
    assert(cb != VK_NULL_HANDLE);
    VkResult res = VK_ERROR_UNKNOWN;
    uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    vk_shader::ComputePipelines* pipelines = this->handle->context->pipelinesPerDev.at(gpuId);
    Pipeline targetPipeline;

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

    if ( descriptorSet == VK_NULL_HANDLE ) {
      descriptorSet = vk_helper::allocateDescriptorSet(handle->vulkanDevice, targetPipeline.descriptorSetLayout, &res);
      CHECK_VK_MSG("Allocate descriptor set for MatBiasLayer: " + name, res);
    }
    // CHECK_VK_MSG("Begin command buffer for MatBiasLayer: " + name, res);
    auto pushConstants = AddChannelBiasNCParams();
    pushConstants.nSize = batchSize;  // No spatial dimension for NC tensor
    pushConstants.cSize = numChannels;
    // update descriptor set
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, biasBuf),
    };

    vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.pipeline);
    vkCmdPushConstants(
      cb,
      targetPipeline.layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(AddChannelBiasNCParams),
      &pushConstants
    );
    vkCmdBindDescriptorSets(
      cb,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      targetPipeline.layout,
      0,
      1,
      &descriptorSet,
      0,
      nullptr
    );

    // 1D dispatch: total elements = batchSize * numChannels
    uint32_t totalSize = static_cast<uint32_t>(batchSize) * static_cast<uint32_t>(numChannels);
    uint32_t wgCountX = (totalSize + targetPipeline.localSizeX - 1u) / targetPipeline.localSizeX;
    uint32_t wgCountY = 1u;
    uint32_t wgCountZ = 1u;
    SHADER_PROFILE_START("ADD_CHANNEL_BIAS_NC_FP32", cb);
    vkCmdDispatch(cb, wgCountX, wgCountY, wgCountZ);
    SHADER_PROFILE_END("ADD_CHANNEL_BIAS_NC_FP32", cb);
    vk_helper::barrierCommandBuffer(cb);
      vk_helper::barrierCommandBufferForBuffer(cb, input);
    // res = vk_helper::endCommandBuffer(commandBuffer);
    // CHECK_VK_MSG("End command buffer for MatBiasLayer: " + name, res);
  }

  /**
   * @brief Launch the recorded command buffer, only for debug now.
   * @param batchSize
   * @param input
   */
  void debug(int batchSize, VulkanBuffer* input) {
    VkCommandBuffer commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
    VkResult res = vk_helper::beginCommandBuffer(commandBuffer);
    forward(commandBuffer, batchSize, input);
    res = vk_helper::endCommandBuffer(commandBuffer);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {commandBuffer});
    printDeviceBuffer(name + " Output : ", handle->vulkanDevice, input, static_cast<size_t>(batchSize) * static_cast<size_t>(numChannels));
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
 */
struct NormActConv {
  ComputeHandleInternal* handle;
  ConvLayer conv;
  BatchNormLayer bn;
  const int inChannels;
  const int outChannels;

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

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, int maxBatchSize) {
    return conv.requiredConvWorkspaceElts(handle, maxBatchSize);
  }

  /**
   * @brief record command buffers for conv and bn layers, no winograd algorithm, so convworkspace not required now.
   * @param batchSize
   * @param input
   * @param inputScratchOrInput
   * @param output
   * @param mask
   */
  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* inputScratchOrInput, //It's okay if this is the same as input, if it's okay to mutate input.
    VulkanBuffer* output,
    VulkanBuffer* mask,
    VulkanBuffer* convWorkspace = nullptr,
    VulkanBuffer* convWorkspace2 = nullptr
  ) {
    // if ( conv.isBNActFusedPossible() ) {
      // conv.forwardBnActConv(cb, &bn, batchSize, input, output, convWorkspace, convWorkspace2, mask);
      // vk_helper::barrierCommandBufferForBuffer(cb, output);
    // } else {
      bn.forward(cb, batchSize, input, mask, inputScratchOrInput);
      vk_helper::barrierCommandBufferForBuffer(cb, inputScratchOrInput);
      conv.forward(cb, batchSize, inputScratchOrInput, output, convWorkspace, convWorkspace2);
      vk_helper::barrierCommandBufferForBuffer(cb, output);
    // }
  }


  /**
   * @brief Launch the recorded command buffers, only for debug now.
   * @param batchSize
   * @param input
   * @param inputScratchOrInput
   * @param output
   */
  void debug(
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* inputScratchOrInput,
    VulkanBuffer* output,
    VulkanBuffer* mask,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    if ( conv.isBNActFusedPossible() ) {
      conv.debug(batchSize, input, output, &bn, mask, convWorkspace, convWorkspace2);
    } else {
      bn.debug(batchSize, input, mask, inputScratchOrInput);
      conv.debug(batchSize, inputScratchOrInput, output, nullptr, nullptr, convWorkspace, convWorkspace2);
    }
  }

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;
};

void performExtractChannel0NCHW(
  ComputeHandleInternal *handle,
  VkCommandBuffer& commandBuffer,
  VkDescriptorSet& descriptorSet,
  VulkanBuffer* input,
  VulkanBuffer* output,
  int batchSize,
  int numInputChannels,
  int nnXYLen,
  bool begin = true
) {
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  vk_shader::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  Pipeline targetPipeline = pipelines->extractChannel0NCHWFp32;
  if ( commandBuffer == VK_NULL_HANDLE ) {
    commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
  }

  VkResult res = VK_ERROR_UNKNOWN;
  if ( begin ) {
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for ExtractChannel0NCHW", res);
  }

  if ( descriptorSet == VK_NULL_HANDLE ) {
    descriptorSet = vk_helper::allocateDescriptorSet(handle->vulkanDevice, targetPipeline.descriptorSetLayout, &res);
    CHECK_VK_MSG("Allocate descriptor set for ExtractChannel0NCHW", res);
  }
  // update descriptor set
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, output)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

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
  ExtractChannel0NCHWParams pushConstants = {};
  pushConstants.cSize = static_cast<uint32_t>(numInputChannels);
  pushConstants.nSize = static_cast<uint32_t>(batchSize);
  pushConstants.xySize = static_cast<uint32_t>(nnXYLen);
  vkCmdPushConstants(
    commandBuffer,
    targetPipeline.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(ExtractChannel0NCHWParams),
    &pushConstants
  );
  // Thread mapping: x->spatial, y->batch
  uint32_t globalSizeX = static_cast<uint32_t>(vk_helper::powerOf2ify(nnXYLen));
  uint32_t globalSizeY = static_cast<uint32_t>(vk_helper::powerOf2ify(batchSize));
  uint32_t wgCountX = (globalSizeX + targetPipeline.localSizeX - 1u) / targetPipeline.localSizeX;
  uint32_t wgCountY = (globalSizeY + targetPipeline.localSizeY - 1u) / targetPipeline.localSizeY;
  uint32_t wgCountZ = 1u;
  SHADER_PROFILE_START("EXTRACT_CHANNEL0_NCHW_FP32", commandBuffer);
  vkCmdDispatch(commandBuffer, wgCountX, wgCountY, wgCountZ);
  SHADER_PROFILE_END("EXTRACT_CHANNEL0_NCHW_FP32", commandBuffer);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, output);

  if ( begin ) {
    vk_helper::endCommandBuffer(commandBuffer);
  }
  // return commandBuffer;
}

void performAddChannelBiases(
  ComputeHandleInternal *handle,
  VkCommandBuffer& commandBuffer,
  VkDescriptorSet& descriptorSet,
  VulkanBuffer* input,
  VulkanBuffer* bias,
  int ncSize,
  int nnXYLen,
  bool begin = true
) {
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  vk_shader::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  Pipeline targetPipeline = pipelines->addChannelBiasNCHWFp32;

  if( commandBuffer == VK_NULL_HANDLE ) {
    commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
  }
  // VkCommandBuffer commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
  VkResult res = VK_ERROR_UNKNOWN;
  if ( begin ) {
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for AddChannelBiases", res);
  }
  if ( descriptorSet == VK_NULL_HANDLE ) {
    descriptorSet = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice,
      targetPipeline.descriptorSetLayout,
      &res
    );
  }
  // update descriptor set
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, bias)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

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
  AddChannelBiasNCHWParams pushConstants = {};
  pushConstants.ncSize = static_cast<uint32_t>(ncSize);
  pushConstants.xySize = static_cast<uint32_t>(nnXYLen);
  vkCmdPushConstants(
    commandBuffer,
    targetPipeline.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(AddChannelBiasNCHWParams),
    &pushConstants
  );
  uint32_t wgCountX = (static_cast<uint32_t>(nnXYLen) + targetPipeline.localSizeX - 1u) / targetPipeline.localSizeX;
  uint32_t wgCountY = (static_cast<uint32_t>(ncSize) + targetPipeline.localSizeY - 1u) / targetPipeline.localSizeY;
  uint32_t wgCountZ = 1u;
  SHADER_PROFILE_START("ADD_CHANNEL_BIAS_NCHW_FP32", commandBuffer);
  vkCmdDispatch(commandBuffer, wgCountX, wgCountY, wgCountZ);
  SHADER_PROFILE_END("ADD_CHANNEL_BIAS_NCHW_FP32", commandBuffer);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, input);
  if ( begin ) {
    vk_helper::endCommandBuffer(commandBuffer);
  }
  // return commandBuffer;
}

void performAddPointWise(
  ComputeHandleInternal *handle,
  VkCommandBuffer& commandBuffer,
  VkDescriptorSet& descriptorSet,
  VulkanBuffer* acc,
  VulkanBuffer* value,
  int totalSize,
  bool begin = true
) {
  if( commandBuffer == VK_NULL_HANDLE ) {
    commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
  }
  // VkCommandBuffer commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
  VkResult res = VK_ERROR_UNKNOWN;
  if ( begin ) {
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for AddPointWise", res);
  }
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  vk_shader::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);

  if ( descriptorSet == VK_NULL_HANDLE ) {
    descriptorSet = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice,
      pipelines->addPointWiseFp32.descriptorSetLayout,
      &res
    );
  }
  // update descriptor set
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, acc),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, value)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

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
  AddPointWiseParams pushConstants = {};
  pushConstants.totalSize = static_cast<uint32_t>(totalSize);
  vkCmdPushConstants(
    commandBuffer,
    pipelines->addPointWiseFp32.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(AddPointWiseParams),
    &pushConstants
  );
  const Pipeline& targetPipeline = pipelines->addPointWiseFp32;
  uint32_t wgCountX = (static_cast<uint32_t>(totalSize) + targetPipeline.localSizeX - 1u) / targetPipeline.localSizeX;
  uint32_t wgCountY = 1u;
  uint32_t wgCountZ = 1u;
  SHADER_PROFILE_START("ADD_POINTWISE_FP32", commandBuffer);
  vkCmdDispatch(commandBuffer, wgCountX, wgCountY, wgCountZ);
  SHADER_PROFILE_END("ADD_POINTWISE_FP32", commandBuffer);
  vk_helper::barrierCommandBuffer(commandBuffer);
    vk_helper::barrierCommandBufferForBuffer(commandBuffer, acc);
  if ( begin ) {
    vk_helper::endCommandBuffer(commandBuffer);
  }
}

void performGpoolMask(
  ComputeHandleInternal *handle,
  VkCommandBuffer& commandBuffer,
  VkDescriptorSet& descriptorSet,
  VulkanBuffer* gpoolConvOut,
  VulkanBuffer* gpoolConcat,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  int batchSize,
  int gpoolChannels,
  int nnXYLen,
  VkResult* result,
  bool begin = true
) {
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  vk_shader::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  if ( commandBuffer == VK_NULL_HANDLE ) {
    commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
  }
  VkResult res = VK_ERROR_UNKNOWN;
  if ( begin ) {
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for GlobalPoolingMask", res);
  }
  if ( descriptorSet == VK_NULL_HANDLE ) {
    descriptorSet = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice,
      pipelines->globalPoolingChannelsFp32.descriptorSetLayout,
      &res
    );
    CHECK_VK_MSG("Allocate descriptor set for GlobalPoolingMask", res);
  }
  // update descriptor set
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, gpoolConvOut),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, gpoolConcat),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, mask),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 3, maskSum)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

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
  GlobalPoolingChannelsParams pushConstants = {};
  pushConstants.batchSize = static_cast<uint32_t>(batchSize);
  pushConstants.gpoolChannels = static_cast<uint32_t>(gpoolChannels);
  pushConstants.nnXYLen = static_cast<uint32_t>(nnXYLen);
  vkCmdPushConstants(
    commandBuffer,
    pipelines->globalPoolingChannelsFp32.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(GlobalPoolingChannelsParams),
    &pushConstants
  );

  // Shader uses SV_GroupID.y for channel, SV_GroupID.z for batch
  // Each workgroup of 128 threads processes one (batch, channel) pair with parallel reduction
  // Since Vulkan shader has fixed numthreads with Y,Z=1, dispatch Y,Z directly equals channel and batch counts
  uint32_t wgCountX = 1u;
  uint32_t wgCountY = static_cast<uint32_t>(gpoolChannels);
  uint32_t wgCountZ = static_cast<uint32_t>(batchSize);
  SHADER_PROFILE_START("GLOBAL_POOLING_CHANNELS_FP32", commandBuffer);
  vkCmdDispatch(commandBuffer, wgCountX, wgCountY, wgCountZ);
  SHADER_PROFILE_END("GLOBAL_POOLING_CHANNELS_FP32", commandBuffer);
  vk_helper::barrierCommandBuffer(commandBuffer);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, maskSum);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, mask);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, gpoolConcat); 
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, gpoolConvOut);
  if ( begin ) {
    res = vk_helper::endCommandBuffer(commandBuffer);
  }
  *result = res;
}

void performValueHeadPool(
  ComputeHandleInternal *handle,
  VkCommandBuffer& commandBuffer,
  VkDescriptorSet& descriptorSet,
  VulkanBuffer* gpoolConvOut,
  VulkanBuffer* gpoolConcat,
  VulkanBuffer* maskSum,
  int batchSize,
  int valueHeadChannels,
  int nnXYLen,
  bool begin = true
) {
  if ( commandBuffer == VK_NULL_HANDLE ) {
    commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
  }
  VkResult res = VK_ERROR_UNKNOWN;
  if ( begin ) {
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for ValueHeadPool", res);
  }
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  vk_shader::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);

  if ( descriptorSet == VK_NULL_HANDLE ) {
    descriptorSet = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice,
      pipelines->valueHeadPoolingChannelsFp32.descriptorSetLayout,
      &res
    );
  }
  // update descriptor set
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, gpoolConvOut),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, gpoolConcat),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, maskSum)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

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
  ValueHeadPoolingChannelsParams pushConstants = {};
  pushConstants.batchSize = static_cast<uint32_t>(batchSize);
  pushConstants.gpoolChannels = static_cast<uint32_t>(valueHeadChannels);
  pushConstants.nnXYLen = static_cast<uint32_t>(nnXYLen);
  vkCmdPushConstants(
    commandBuffer,
    pipelines->valueHeadPoolingChannelsFp32.layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(ValueHeadPoolingChannelsParams),
    &pushConstants
  );
  // Shader uses SV_GroupID.y for channel, SV_GroupID.z for batch
  // Same pattern as GlobalPooling - each workgroup processes one (batch, channel) pair
  uint32_t wgCountX = 1u;
  uint32_t wgCountY = static_cast<uint32_t>(valueHeadChannels);
  uint32_t wgCountZ = static_cast<uint32_t>(batchSize);
  SHADER_PROFILE_START("VALUE_HEAD_POOLING_CHANNELS_FP32", commandBuffer);
  vkCmdDispatch(commandBuffer, wgCountX, wgCountY, wgCountZ);
  SHADER_PROFILE_END("VALUE_HEAD_POOLING_CHANNELS_FP32", commandBuffer);
  vk_helper::barrierCommandBuffer(commandBuffer);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, maskSum);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, gpoolConcat);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, gpoolConvOut);
  if ( begin ) {
    vk_helper::endCommandBuffer(commandBuffer);
  }
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
  // std::vector<VkCommandBuffer> commandBuffers;
  // VkCommandBuffer addPointWiseCB = VK_NULL_HANDLE;
  VkDescriptorSet addPointWiseDS = VK_NULL_HANDLE;

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

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    return ConvWorkspaceEltsNeeded::getMax(
      normActConv2->requiredConvWorkspaceElts(handle, maxBatchSize),
      normActConv->requiredConvWorkspaceElts(handle, maxBatchSize)
    );
  }

  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {

    SizedBuf<VulkanBuffer*> mid(
      scratch->allocator,
      scratch->getBufSizeXY(normActConv->outChannels)
    );
    normActConv->forward(cb, batchSize, trunk, trunkScratch, mid.buf , mask, convWorkspace, convWorkspace2);
    normActConv2->forward(cb, batchSize, mid.buf, mid.buf, trunkScratch, mask, convWorkspace, convWorkspace2);
    performAddPointWise(handle, cb, addPointWiseDS, trunk, trunkScratch, static_cast<int>(batchSize * normActConv2->outChannels * nnXLen * nnYLen), false);
  }

  void debug(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    SizedBuf<VulkanBuffer*> mid(
      scratch->allocator,
      scratch->getBufSizeXY(normActConv->outChannels)
    );
    normActConv->debug(batchSize, trunk, trunkScratch, mid.buf , mask, convWorkspace, convWorkspace2);
    normActConv2->debug(batchSize, mid.buf, mid.buf, trunkScratch, mask, convWorkspace, convWorkspace2);
    VkCommandBuffer addPointWiseCB = VK_NULL_HANDLE;
    performAddPointWise(handle, addPointWiseCB, addPointWiseDS, trunk, trunkScratch, static_cast<int>(batchSize * normActConv2->outChannels * nnXLen * nnYLen));
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {addPointWiseCB});
    printDeviceBuffer(name + " RB Output : ", handle->vulkanDevice, trunk, static_cast<size_t>(batchSize) * static_cast<size_t>(normActConv2->outChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen));
  }
};

struct GlobalPoolingResidualBlock {
  ComputeHandleInternal* handle;
  const std::string name;
  BatchNormLayer* preBN;
  ConvLayer* regularConv;
  ConvLayer* gpoolConv;
  BatchNormLayer* gpoolBN;
  MatmulLayer* gpoolToBiasMul;
  NormActConv* normActConv2;

  const int nnXLen;
  const int nnYLen;
  const int nnXYLen;
  const int regularChannels;
  const int gpoolChannels;

  // VkCommandBuffer gpoolCB = VK_NULL_HANDLE;
  VkDescriptorSet gpoolDS = VK_NULL_HANDLE;
  // VkCommandBuffer addChannelCB = VK_NULL_HANDLE;
  VkDescriptorSet addChannelDS = VK_NULL_HANDLE;
  // VkCommandBuffer addPointWiseCB = VK_NULL_HANDLE;
  VkDescriptorSet addPointWiseDS = VK_NULL_HANDLE;

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
    preBN = new BatchNormLayer(handle, &desc->preBN, &desc->preActivation, nnXLen, nnYLen);
    regularConv = new ConvLayer(handle, &desc->regularConv, nnXLen, nnYLen);
    gpoolConv = new ConvLayer(handle, &desc->gpoolConv, nnXLen, nnYLen);
    gpoolBN = new BatchNormLayer(handle, &desc->gpoolBN, &desc->gpoolActivation, nnXLen, nnYLen);
    gpoolToBiasMul = new MatmulLayer(handle, &desc->gpoolToBiasMul);
    normActConv2 = new NormActConv(handle, &desc->finalConv, &desc->midBN, &desc->midActivation, nnXLen, nnYLen);
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

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, regularConv->requiredConvWorkspaceElts(handle, maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, gpoolConv->requiredConvWorkspaceElts(handle, maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, normActConv2->requiredConvWorkspaceElts(handle, maxBatchSize));
    return maxElts;
  }

  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    SizedBuf<VulkanBuffer*> regularOut(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(gpoolChannels * 3));
    SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(regularChannels));

    preBN->forward(cb, batchSize, trunk, mask, trunkScratch);
    regularConv->forward(cb, batchSize, trunkScratch, regularOut.buf, convWorkspace, convWorkspace2);
    gpoolConv->forward(cb, batchSize, trunkScratch, gpoolOut.buf, convWorkspace, convWorkspace2);
    gpoolBN->forward(cb, batchSize, gpoolOut. buf,mask, gpoolOut.buf);
    VkResult res;;
    performGpoolMask(handle, cb, gpoolDS, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, batchSize, gpoolChannels, nnXYLen, &res, false);
    gpoolToBiasMul->forward(cb, batchSize, gpoolConcat.buf, gpoolBias.buf);
    performAddChannelBiases(handle, cb, addChannelDS, regularOut.buf, gpoolBias.buf, batchSize * regularChannels, nnXYLen, false);
    normActConv2->forward(cb, batchSize, regularOut.buf, regularOut.buf, trunkScratch, mask, convWorkspace, convWorkspace2);
    performAddPointWise(handle, cb, addPointWiseDS, trunk, trunkScratch, batchSize * normActConv2->outChannels * nnXLen * nnYLen, false);
  }

  void debug(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    SizedBuf<VulkanBuffer*> regularOut(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(gpoolChannels * 3));
    SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(regularChannels));

    preBN->debug(batchSize, trunk, mask, trunkScratch);
    regularConv->debug(batchSize, trunkScratch, regularOut.buf, nullptr, nullptr, convWorkspace, convWorkspace2);
    gpoolConv->debug(batchSize, trunkScratch, gpoolOut.buf, nullptr, nullptr, convWorkspace, convWorkspace2);
    gpoolBN->debug(batchSize, gpoolOut. buf,mask, gpoolOut.buf);
    VkResult res;;
    VkCommandBuffer gpoolCB = VK_NULL_HANDLE;
    VkCommandBuffer addChannelCB = VK_NULL_HANDLE;
    VkCommandBuffer addPointWiseCB = VK_NULL_HANDLE;
    performGpoolMask(handle, gpoolCB, gpoolDS, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, batchSize, gpoolChannels, nnXYLen, &res);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {gpoolCB});
    CHECK_VK_MSG("Record GlobalPoolingResidualBlock gpool mask", res);
    gpoolToBiasMul->debug(batchSize, gpoolConcat.buf, gpoolBias.buf);
    performAddChannelBiases(handle, addChannelCB, addChannelDS, regularOut.buf, gpoolBias.buf, batchSize * regularChannels, nnXYLen);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {addChannelCB});
    normActConv2->debug(batchSize, regularOut.buf, regularOut.buf, trunkScratch, mask, convWorkspace, convWorkspace2);
    performAddPointWise(handle, addPointWiseCB, addPointWiseDS, trunk, trunkScratch, batchSize * normActConv2->outChannels * nnXLen * nnYLen);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {addPointWiseCB});
    printDeviceBuffer(name + " GPRB Output : " , handle->vulkanDevice, trunk, static_cast<size_t>(batchSize) * static_cast<size_t>(normActConv2->outChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen));
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
  // std::vector<VkCommandBuffer> commandBuffers;
  // VkCommandBuffer addPointWiseCB = VK_NULL_HANDLE;
  VkDescriptorSet addPointWiseDS = VK_NULL_HANDLE;

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
    normActConv = new NormActConv(handle, &desc->preConv, &desc->preBN, &desc->preActivation, nnXLen, nnYLen);
    blocks = new BlockStack(handle, desc->blocks, desc->numBlocks, desc->preConv.outChannels, nnXLen, nnYLen);
    normActConv2 = new NormActConv(handle, &desc->postConv, &desc->postBN, &desc->postActivation, nnXLen, nnYLen);
  }

  ~NestedResidualBlock() {
    delete normActConv2;
    delete blocks;
    delete normActConv;
  }

  NestedResidualBlock() = delete;
  NestedResidualBlock(const NestedResidualBlock&) = delete;
  NestedResidualBlock& operator=(const NestedResidualBlock&) = delete;

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    return ConvWorkspaceEltsNeeded::getMax(
      normActConv->requiredConvWorkspaceElts(handle, maxBatchSize),
      ConvWorkspaceEltsNeeded::getMax(
        blocks->requiredConvWorkspaceElts(handle, maxBatchSize),
        normActConv2->requiredConvWorkspaceElts(handle, maxBatchSize)
      )
    );
  }

  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    SizedBuf<VulkanBuffer*> mid(scratch->allocator, scratch->getBufSizeXY(normActConv->outChannels));
    SizedBuf<VulkanBuffer*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv->outChannels));
    normActConv->forward(cb, batchSize, trunk, trunkScratch, mid.buf , mask, convWorkspace, convWorkspace2);
    blocks->forward(cb, batchSize, scratch, mid.buf, midScratch.buf, mask, maskSum, convWorkspace, convWorkspace2);
    normActConv2->forward(cb, batchSize, mid.buf, mid.buf, trunkScratch, mask, convWorkspace, convWorkspace2);
    performAddPointWise(handle, cb, addPointWiseDS, trunk, trunkScratch, static_cast<int>(batchSize * normActConv2->outChannels * nnXLen * nnYLen), false);
  }

  void debug(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    SizedBuf<VulkanBuffer*> mid(scratch->allocator, scratch->getBufSizeXY(normActConv->outChannels));
    SizedBuf<VulkanBuffer*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv->outChannels));
    VkCommandBuffer addPointWiseCB = VK_NULL_HANDLE;
    normActConv->debug(batchSize, trunk, trunkScratch, mid.buf , mask, convWorkspace, convWorkspace2);
    blocks->debug(batchSize, scratch, mid.buf, midScratch.buf, mask, maskSum, convWorkspace, convWorkspace2);
    normActConv2->debug(batchSize, mid.buf, mid.buf, trunkScratch, mask, convWorkspace, convWorkspace2);
    performAddPointWise(handle, addPointWiseCB, addPointWiseDS, trunk, trunkScratch, static_cast<int>(batchSize * normActConv2->outChannels * nnXLen * nnYLen));
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {addPointWiseCB});
    printDeviceBuffer(name + " NestedRB Output : " , handle->vulkanDevice, trunk, static_cast<size_t>(batchSize) * static_cast<size_t>(normActConv2->outChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen));
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

ConvWorkspaceEltsNeeded BlockStack::requiredConvWorkspaceElts(ComputeHandleInternal *handle, size_t maxBatchSize) const {
  ConvWorkspaceEltsNeeded maxElts;
  for(int i = 0; i<blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,block->requiredConvWorkspaceElts(handle,maxBatchSize));
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,block->requiredConvWorkspaceElts(handle,maxBatchSize));
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedResidualBlock* block = (NestedResidualBlock*)blocks[i].second.get();
      maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,block->requiredConvWorkspaceElts(handle,maxBatchSize));
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  return maxElts;
}

void BlockStack::debug(
  int batchSize,
  ScratchBuffers *scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* trunkScratch,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  VulkanBuffer* convWorkspace,
  VulkanBuffer* convWorkspace2
) 
{
    // logger->write("Recording BlockStack - ResidualBlock index: " + Global::intToString(i));
  for ( int i = 0 ; i< numBlocks ; ++i ) {
    int blockType = blocks[i].first;
    if ( blockType == ORDINARY_BLOCK_KIND ) {
      ResidualBlock* blockPtr = static_cast<ResidualBlock*>(blocks[i].second.get());
      if(blockPtr == nullptr) {
        Global::fatalError("BlockStack::debug: ResidualBlock pointer is null at index " + Global::intToString(i));
      }
      blockPtr->debug(batchSize, scratch, trunk, trunkScratch, mask, convWorkspace, convWorkspace2);
    } else if ( blockType == GLOBAL_POOLING_BLOCK_KIND ) {
        GlobalPoolingResidualBlock* blockPtr = static_cast<GlobalPoolingResidualBlock*>(blocks[i].second.get());
        if(blockPtr == nullptr) {
          Global::fatalError("BlockStack::debug: GlobalPoolingResidualBlock pointer is null at index " + Global::intToString(i));
        }
        blockPtr->debug(batchSize, scratch, trunk, trunkScratch, mask, maskSum, convWorkspace, convWorkspace2);
    } else if ( blockType == NESTED_BOTTLENECK_BLOCK_KIND ) {
      NestedResidualBlock* blockPtr = static_cast<NestedResidualBlock*>(blocks[i].second.get());
      if(blockPtr == nullptr) {
        Global::fatalError("BlockStack::debug: NestedResidualBlock pointer is null at index " + Global::intToString(i));
      }
      blockPtr->debug(batchSize, scratch, trunk, trunkScratch, mask, maskSum, convWorkspace, convWorkspace2);
    } else {
      ASSERT_UNREACHABLE;
    }
  }
}

void BlockStack::forward(
  VkCommandBuffer& cb,
  int batchSize,
  ScratchBuffers *scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* trunkScratch,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  VulkanBuffer* convWorkspace,
  VulkanBuffer* convWorkspace2
) {
  assert(cb != VK_NULL_HANDLE);

  for (int i = 0 ; i< numBlocks ; ++i ) {
    int blockType = blocks[i].first;
    if ( blockType == ORDINARY_BLOCK_KIND ) {
      ResidualBlock* blockPtr = static_cast<ResidualBlock*>(blocks[i].second.get());
      if(blockPtr == nullptr) {
        Global::fatalError("BlockStack::forward: ResidualBlock pointer is null at index " + Global::intToString(i));
      }
      blockPtr->forward(cb, batchSize, scratch, trunk, trunkScratch, mask, convWorkspace, convWorkspace2);
    } else if ( blockType == GLOBAL_POOLING_BLOCK_KIND ) {
      GlobalPoolingResidualBlock* blockPtr = static_cast<GlobalPoolingResidualBlock*>(blocks[i].second.get());
      if(blockPtr == nullptr) {
        Global::fatalError("BlockStack::forward: GlobalPoolingResidualBlock pointer is null at index " + Global::intToString(i));
      }
      blockPtr->forward(cb, batchSize, scratch, trunk, trunkScratch, mask, maskSum, convWorkspace, convWorkspace2);
    } else if ( blockType == NESTED_BOTTLENECK_BLOCK_KIND ) {
      NestedResidualBlock* blockPtr = static_cast<NestedResidualBlock*>(blocks[i].second.get());
      if(blockPtr == nullptr) {
        Global::fatalError("BlockStack::forward: NestedResidualBlock pointer is null at index " + Global::intToString(i));
      }
      blockPtr->forward(cb, batchSize, scratch, trunk, trunkScratch, mask, maskSum, convWorkspace, convWorkspace2);
    } else {
      ASSERT_UNREACHABLE;
    }
  }
}

struct SGFMetadataEncoder {
  ComputeHandleInternal *handle;
  const std::string name;
  MatmulLayer* matmul1;
  MatBiasLayer* matBias1;
  MatmulLayer* matmul2;
  MatBiasLayer* matBias2;
  MatmulLayer* matmul3;

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

  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    assert(cb != VK_NULL_HANDLE);
    SizedBuf<VulkanBuffer*> internalBuf1(scratch->allocator, scratch->getBufSizeFloat(std::max(matmul1->outChannels, matmul2->outChannels)));
    SizedBuf<VulkanBuffer*> internalBuf2(scratch->allocator, scratch->getBufSizeFloat(std::max(matmul1->outChannels, matmul2->outChannels)));
    matmul1->forward(cb, batchSize, input, internalBuf1.buf);
    matBias1->forward(cb, batchSize, internalBuf1.buf);
    matmul2->forward(cb, batchSize, internalBuf1.buf, internalBuf2.buf);
    matBias2->forward(cb, batchSize, internalBuf2.buf);
    matmul3->forward(cb, batchSize, internalBuf2.buf, output);
  }

  /**
   * @brief record SGFMetadataEncoder
   */
  void debug(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    SizedBuf<VulkanBuffer*> internalBuf1(scratch->allocator, scratch->getBufSizeFloat(std::max(matmul1->outChannels, matmul2->outChannels)));
    SizedBuf<VulkanBuffer*> internalBuf2(scratch->allocator, scratch->getBufSizeFloat(std::max(matmul1->outChannels, matmul2->outChannels)));

    matmul1->debug(batchSize, input, internalBuf1.buf);
    matBias1->debug(batchSize, internalBuf1.buf);
    matmul2->debug(batchSize, internalBuf1.buf, internalBuf2.buf);
    matBias2->debug(batchSize, internalBuf2.buf);
    matmul3->debug(batchSize, internalBuf2.buf, output);
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
  VkDescriptorSet addChannelBiasDS = VK_NULL_HANDLE;
  VkDescriptorSet addChannelBiasDS2 = VK_NULL_HANDLE;

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

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(
    ComputeHandleInternal* handle,
    int maxBatchSize
  ) const {
    return ConvWorkspaceEltsNeeded::getMax(
      initialConv->requiredConvWorkspaceElts(handle, maxBatchSize),
      blockStack.requiredConvWorkspaceElts(handle, maxBatchSize)
    );
  }

  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* input,
    VulkanBuffer* inputGlobal,
    VulkanBuffer* inputMeta,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    assert(cb != VK_NULL_HANDLE);
    SizedBuf<VulkanBuffer*> trunkScratch( scratch->allocator, scratch->getBufSizeXY(trunkNumChannels) );

    initialConv->forward(cb, batchSize, input, trunk, convWorkspace, convWorkspace2);
    initialMatmul->forward(cb, batchSize, inputGlobal, trunkScratch.buf);
    performAddChannelBiases(handle, cb, addChannelBiasDS, trunk, trunkScratch.buf, batchSize * trunkNumChannels, nnXLen * nnYLen, false);
    if ( sgfMetadataEncoder != nullptr ) {
      SizedBuf<VulkanBuffer*> sgfEncodedMeta(scratch->allocator, scratch->getBufSizeFloat(sgfMetadataEncoder->matmul3->outChannels));
      sgfMetadataEncoder->forward(cb, batchSize, scratch, inputMeta, sgfEncodedMeta.buf);
      performAddChannelBiases(handle, cb, addChannelBiasDS2, trunk, sgfEncodedMeta.buf, batchSize * trunkNumChannels, nnXLen * nnYLen, false);
    }
    blockStack.forward(cb, batchSize, scratch, trunk, trunkScratch.buf, mask, maskSum, convWorkspace, convWorkspace2);
    trunkTipBN->forward(cb, batchSize, trunk, mask, trunk);
  }

  void debug(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* input,
    VulkanBuffer* inputGlobal,
    VulkanBuffer* inputMeta,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    SizedBuf<VulkanBuffer*> trunkScratch( scratch->allocator, scratch->getBufSizeXY(trunkNumChannels) );

    VkCommandBuffer cb = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
    vk_helper::beginCommandBuffer(cb);
    vkCmdFillBuffer(cb, convWorkspace->buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cb, convWorkspace2->buffer, 0, VK_WHOLE_SIZE, 0);
    vk_helper::endCommandBuffer(cb);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {cb});

    initialConv->debug(batchSize, input, trunk, nullptr, nullptr, convWorkspace, convWorkspace2);
    initialMatmul->debug(batchSize, inputGlobal, trunkScratch.buf);
    VkCommandBuffer addChannelBiasCB = VK_NULL_HANDLE;
    performAddChannelBiases(handle, addChannelBiasCB, addChannelBiasDS, trunk, trunkScratch.buf, batchSize * trunkNumChannels, nnXLen * nnYLen);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {addChannelBiasCB});
    {
      printDeviceBuffer(name  + "/add_channel_bias0 Output : ", handle->vulkanDevice, trunk, static_cast<size_t>(batchSize) * static_cast<size_t>(trunkNumChannels) * static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen));
    }
    if ( sgfMetadataEncoder != nullptr ) {
      VkCommandBuffer addChannelBiasCB2 = VK_NULL_HANDLE;
      SizedBuf<VulkanBuffer*> sgfEncodedMeta(scratch->allocator, scratch->getBufSizeFloat(sgfMetadataEncoder->matmul3->outChannels));
      sgfMetadataEncoder->debug(batchSize, scratch, inputMeta, sgfEncodedMeta.buf);
      performAddChannelBiases(handle, addChannelBiasCB2, addChannelBiasDS2, trunk, sgfEncodedMeta.buf, batchSize * trunkNumChannels, nnXLen * nnYLen);
      vk_helper::submitCommandBuffers(handle->vulkanDevice, {addChannelBiasCB2});
    }
    blockStack.debug(batchSize, scratch, trunk, trunkScratch.buf, mask, maskSum, convWorkspace, convWorkspace2);
    trunkTipBN->debug(batchSize, trunk, mask, trunk);
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
  VkDescriptorSet gpoolDS = VK_NULL_HANDLE;
  VkDescriptorSet addChannelBiasDS = VK_NULL_HANDLE;

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

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,p1Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,g1Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,p2Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    return maxElts;
  }

  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* policyPass,
    VulkanBuffer* policy,
    VulkanBuffer *convWorkspace,
    VulkanBuffer *convWorkspace2
  ) {
    assert(cb != VK_NULL_HANDLE);
    SizedBuf<VulkanBuffer*> p1Out(scratch->allocator, scratch->getBufSizeXY(p1Channels));
    SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(g1Channels * 3));
    SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(p1Channels));
    SizedBuf<VulkanBuffer*> p1Pass(scratch->allocator, scratch->getBufSizeFloat(p1Channels));

    p1Conv->forward(cb, batchSize, trunk, p1Out.buf, convWorkspace, convWorkspace2);
    g1Conv->forward(cb, batchSize, trunk, gpoolOut.buf, convWorkspace, convWorkspace2);
    g1BN->forward(cb, batchSize, gpoolOut.buf, mask, gpoolOut.buf);
    VkResult res;;
    performGpoolMask(handle, cb, gpoolDS, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, batchSize, g1Channels, nnXLen * nnYLen, &res, false);
    gpoolToBiasMul->forward(cb, batchSize, gpoolConcat.buf, gpoolBias.buf);
    performAddChannelBiases(handle, cb, addChannelBiasDS, p1Out.buf, gpoolBias.buf, p1Channels * batchSize, nnXLen * nnYLen, false);
    p1BN->forward(cb, batchSize, p1Out.buf, mask, p1Out.buf);
    p2Conv->forward(cb, batchSize, p1Out.buf, policy, convWorkspace, convWorkspace2);

    if ( modelVersion >= 15 ) {
      gpoolToPassMul->forward(cb, batchSize, gpoolConcat.buf, p1Pass.buf);
      gpoolToPassBias->forward(cb, batchSize, p1Pass.buf);
      gpoolToPassMul2->forward(cb, batchSize, p1Pass.buf, policyPass);
    } else {
      gpoolToPassMul->forward(cb, batchSize, gpoolConcat.buf, policyPass);
    }
  }

  void debug(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* policyPass,
    VulkanBuffer* policy,
    VulkanBuffer *convWorkspace,
    VulkanBuffer *convWorkspace2
  ) {
    SizedBuf<VulkanBuffer*> p1Out(scratch->allocator, scratch->getBufSizeXY(p1Channels));
    SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(g1Channels * 3));
    SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(p1Channels));
    SizedBuf<VulkanBuffer*> p1Pass(scratch->allocator, scratch->getBufSizeFloat(p1Channels));

    p1Conv->debug(batchSize, trunk, p1Out.buf, nullptr, nullptr, convWorkspace, convWorkspace2);
    g1Conv->debug(batchSize, trunk, gpoolOut.buf, nullptr, nullptr, convWorkspace, convWorkspace2);
    g1BN->debug(batchSize, gpoolOut.buf, mask, gpoolOut.buf);
    VkResult res;;
    VkCommandBuffer gpoolCB = VK_NULL_HANDLE;
    performGpoolMask(handle, gpoolCB, gpoolDS, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, batchSize, g1Channels, nnXLen * nnYLen, &res);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {gpoolCB});
    CHECK_VK_MSG("Record PolicyHead gpool mask", res);
    gpoolToBiasMul->debug(batchSize, gpoolConcat.buf, gpoolBias.buf);
    VkCommandBuffer addChannelBiasCB = VK_NULL_HANDLE;
    performAddChannelBiases(handle, addChannelBiasCB, addChannelBiasDS, p1Out.buf, gpoolBias.buf, p1Channels * batchSize, nnXLen * nnYLen);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {addChannelBiasCB});
    p1BN->debug(batchSize, p1Out.buf, mask, p1Out.buf);
    p2Conv->debug(batchSize, p1Out.buf, policy, nullptr, nullptr, convWorkspace, convWorkspace2);

    if ( modelVersion >= 15 ) {
      gpoolToPassMul->debug(batchSize, gpoolConcat.buf, p1Pass.buf);
      gpoolToPassBias->debug(batchSize, p1Pass.buf);
      gpoolToPassMul2->debug(batchSize, p1Pass.buf, policyPass);
    } else {
      gpoolToPassMul->debug(batchSize, gpoolConcat.buf, policyPass);
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
  VkDescriptorSet gpoolDS = VK_NULL_HANDLE;

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

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,v1Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,vOwnershipConv->requiredConvWorkspaceElts(handle,maxBatchSize));
    return maxElts;
  }

  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* value,
    VulkanBuffer* scoreValue,
    VulkanBuffer* ownership,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    assert(cb != VK_NULL_HANDLE);
    SizedBuf<VulkanBuffer*> v1Out(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<VulkanBuffer*> v1Mean(scratch->allocator, scratch->getBufSizeFloat(v1Channels*3));
    SizedBuf<VulkanBuffer*> v2Out(scratch->allocator, scratch->getBufSizeFloat(v2Channels));

    v1Conv->forward(cb, batchSize, trunk, v1Out.buf, convWorkspace, convWorkspace2);
    v1BN->forward(cb, batchSize, v1Out.buf, mask, v1Out.buf);
    VkResult res;;
    performValueHeadPool(handle, cb, gpoolDS, v1Out.buf, v1Mean.buf, maskSum, batchSize, v1Channels, nnXLen * nnYLen, false);

    v2Mul->forward(cb, batchSize, v1Mean.buf, v2Out.buf);
    v2Bias->forward(cb, batchSize, v2Out.buf);
    v3Mul->forward(cb, batchSize, v2Out.buf, value);
    v3Bias->forward(cb, batchSize, value);

    sv3Mul->forward(cb, batchSize, v2Out.buf, scoreValue);
    sv3Bias->forward(cb, batchSize, scoreValue);
    vOwnershipConv->forward(cb, batchSize, v1Out.buf, ownership, convWorkspace, convWorkspace2);
  }

  void debug(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* value,
    VulkanBuffer* scoreValue,
    VulkanBuffer* ownership,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    SizedBuf<VulkanBuffer*> v1Out(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<VulkanBuffer*> v1Mean(scratch->allocator, scratch->getBufSizeFloat(v1Channels*3));
    SizedBuf<VulkanBuffer*> v2Out(scratch->allocator, scratch->getBufSizeFloat(v2Channels));

    v1Conv->debug(batchSize, trunk, v1Out.buf, nullptr, nullptr, convWorkspace, convWorkspace2);
    v1BN->debug(batchSize, v1Out.buf, mask, v1Out.buf);
    VkResult res;
    VkCommandBuffer gpoolCB =  VK_NULL_HANDLE;
    performValueHeadPool(handle, gpoolCB, gpoolDS, v1Out.buf, v1Mean.buf, maskSum, batchSize, v1Channels, nnXLen * nnYLen);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {gpoolCB});

    v2Mul->debug(batchSize, v1Mean.buf, v2Out.buf);
    v2Bias->debug(batchSize, v2Out.buf);
    v3Mul->debug(batchSize, v2Out.buf, value);
    v3Bias->debug(batchSize, value);

    sv3Mul->debug(batchSize, v2Out.buf, scoreValue);
    sv3Bias->debug(batchSize, scoreValue);
    vOwnershipConv->debug(batchSize, v1Out.buf, ownership, nullptr, nullptr, convWorkspace, convWorkspace2);
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
void computeMaskSums(
  ComputeHandleInternal *handle,
  VkCommandBuffer& commandBuffer,
  VkDescriptorSet& descriptorSet,
  int batchSize,
  int nnXLen,
  int nnYLen,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  bool begin = true
) {
  // SumChannels uses one workgroup for each (batch, channel) reduction.
  // Dispatch: X=1 (single workgroup for reduction), Y=numChannels, Z=batchSize
  // Each workgroup of 128 threads processes one (batch, channel) pair with parallel reduction
  // VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

  int numChannels = 1;
  int nnXYLen = nnXLen * nnYLen;
  // Determine GPU and select appropriate compute pipeline
  uint32_t gpuId = handle->vulkanDevice->info.deviceId;
  vk_shader::ComputePipelines* pipelines = handle->context->pipelinesPerDev.at(gpuId);
  Pipeline targetPipeline = pipelines->sumChannelsFp32;

  VkResult res = VK_SUCCESS;
  if ( descriptorSet == VK_NULL_HANDLE ) {
    descriptorSet = vk_helper::allocateDescriptorSet(handle->vulkanDevice, targetPipeline.descriptorSetLayout, &res);
    CHECK_VK_MSG("Allocate compute mask sum descriptor set", res);
  }
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, mask),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, maskSum)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

  if ( commandBuffer == VK_NULL_HANDLE ) {
    commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
  }

  if ( begin ) {
    res = vk_helper::beginCommandBuffer(commandBuffer);
  }
  CHECK_VK_MSG("Begin compute mask sum command buffer", res);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, targetPipeline.layout, 0, 1, &descriptorSet, 0, nullptr);
  SumChannelsParams pushConstants;
  pushConstants.batchSize = batchSize;
  pushConstants.numChannels = numChannels;
  pushConstants.nnXYLen = nnXYLen;
  vkCmdPushConstants(commandBuffer, targetPipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SumChannelsParams), &pushConstants);
  // Dispatch: X=1 (reduction), Y=numChannels(=1 for mask sum), Z=batchSize
  uint32_t wgCountX = 1u;
  uint32_t wgCountY = static_cast<uint32_t>(numChannels);
  uint32_t wgCountZ = static_cast<uint32_t>(batchSize);
  vkCmdDispatch(commandBuffer, wgCountX, wgCountY, wgCountZ);
  vk_helper::barrierCommandBuffer(commandBuffer);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, maskSum);

  if ( begin ) {
    vk_helper::endCommandBuffer(commandBuffer);
  }
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
  VkDescriptorSet extractChannel0DS = VK_NULL_HANDLE;
  VkDescriptorSet computeMaskSumDS = VK_NULL_HANDLE;

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
    fence = vk_helper::createFence(handle->vulkanDevice, &res);
    CHECK_VK_MSG("Create model fence", res);
  }

  ~Model() {
    if( fence != VK_NULL_HANDLE ) {
      vk_helper::destroyFence(handle->vulkanDevice, fence);
      fence = VK_NULL_HANDLE;
    }
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(
    ComputeHandleInternal *handle
  ) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, trunk->requiredConvWorkspaceElts(handle, maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, policyHead->requiredConvWorkspaceElts(handle, maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, valueHead->requiredConvWorkspaceElts(handle, maxBatchSize));
    return maxElts;
  }

  void forward(
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
    VulkanBuffer* ownership,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    VkCommandBuffer forwardCB = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
    VkResult res = vk_helper::beginCommandBuffer(forwardCB);
    CHECK_VK_MSG("Begin model forward command buffer", res);
    performExtractChannel0NCHW(handle, forwardCB, extractChannel0DS, input, mask, batchSize, numInputChannels,handle->paddedNNXYLen, false);
    computeMaskSums(handle, forwardCB, computeMaskSumDS, batchSize, nnXLen, nnYLen, mask, maskSum, false);
    trunk->forward(forwardCB, batchSize, scratch, input, inputGlobal, inputMeta, trunkBuf,  mask, maskSum, convWorkspace, convWorkspace2);
    policyHead->forward(forwardCB, batchSize, scratch, trunkBuf, mask, maskSum, policyPass, policy, convWorkspace, convWorkspace2);
    valueHead->forward(forwardCB, batchSize, scratch, trunkBuf, mask, maskSum, value, scoreValue, ownership, convWorkspace, convWorkspace2);
    vk_helper::endCommandBuffer(forwardCB);
    vkResetFences(handle->vulkanDevice->device, 1, &fence);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, { forwardCB }, fence);
    vkWaitForFences(handle->vulkanDevice->device, 1, &fence, VK_TRUE, UINT64_MAX);
  }

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
    VulkanBuffer* ownership,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    VkCommandBuffer extractChannel0CB = VK_NULL_HANDLE;
    VkCommandBuffer computeMaskSumCB = VK_NULL_HANDLE;

    if ( batchSize > 1 ) {
      std::printf("numInputChannels : %d, batchSize: %d, nnXLen: %d, nnYLen: %d\n", numInputChannels, batchSize, nnXLen, nnYLen);
      printDeviceBuffer("First Input: ", handle->vulkanDevice, input, batchSize * numInputChannels * nnXLen * nnYLen, false);
    }

    performExtractChannel0NCHW(handle, extractChannel0CB, extractChannel0DS, input, mask, batchSize, numInputChannels, nnXLen * nnYLen);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {extractChannel0CB});
    printDeviceBuffer("Model::debug Extract Channel 0 Result", handle->vulkanDevice, mask, batchSize * nnXLen * nnYLen);
    computeMaskSums(handle, computeMaskSumCB, computeMaskSumDS, batchSize, nnXLen, nnYLen, mask, maskSum);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {computeMaskSumCB});
    printDeviceBuffer("Model::debug Mask Sum Result", handle->vulkanDevice, maskSum, batchSize);
    trunk->debug(batchSize, scratch, input, inputGlobal, inputMeta, trunkBuf,  mask, maskSum, convWorkspace, convWorkspace2);
    printDeviceBuffer("Model::debug Trunk Output", handle->vulkanDevice, trunkBuf, batchSize * trunk->trunkNumChannels * nnXLen * nnYLen);
    policyHead->debug(batchSize, scratch, trunkBuf, mask, maskSum, policyPass, policy, convWorkspace, convWorkspace2);
    printDeviceBuffer("Model::debug Policy Output", handle->vulkanDevice, policy, batchSize * policyHead->p2Channels * nnXLen * nnYLen);
    valueHead->debug(batchSize, scratch, trunkBuf, mask, maskSum, value, scoreValue, ownership, convWorkspace, convWorkspace2);
    printDeviceBuffer("Model::debug Value Output", handle->vulkanDevice, value, batchSize * valueHead->valueChannels);
  }
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger *logger,
  int nnXLen,
  int nnYLen,
  const std::string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  if(gpuIdxs.empty())
    throw StringError("NeuralNet::createComputeContext - specified no GPUs to use");
  if(loadedModel == nullptr)
    throw StringError("NeuralNet::createComputeContext - loaded model was null");
  if(loadedModel->modelDesc.modelVersion > 16)
    throw StringError("Vulkan backend currently supports model versions through 16");

  std::string tunerFile;
  if(cfg.contains("vulkanTunerFile"))
    tunerFile = cfg.getString("vulkanTunerFile");
  bool fullTuning = cfg.contains("vulkanTunerFull") && cfg.getBool("vulkanTunerFull");
  bool forceRetune = cfg.contains("vulkanForceRetune") && cfg.getBool("vulkanForceRetune");
  VulkanTuner::ModelInfoForTuning modelInfo = VulkanTuner::ModelInfoForTuning::ofDesc(loadedModel->modelDesc);

  if(logger != nullptr) {
    logger->write("Create Vulkan Compute Context with GPUs: ");
    for(size_t i = 0; i<gpuIdxs.size(); i++) {
      logger->write("  GPU Index " + Global::intToString(gpuIdxs[i]));
    }
  }

  return new ComputeContext(
    nnXLen,
    nnYLen,
    useFP16Mode,
    enabled_t::False,
    std::vector<uint32_t>(gpuIdxs.begin(), gpuIdxs.end()),
    logger,
    tunerFile,
    homeDataDirOverride,
    &modelInfo,
    fullTuning,
    forceRetune
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
    logger,
    "",
    "",
    nullptr,
    false,
    false
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
    input = vk_helper::createDeviceBuffer(handle->vulkanDevice, dtypeSize * inputElts, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create input buffer", res);
    inputGlobal = vk_helper::createDeviceBuffer(handle->vulkanDevice, dtypeSize * inputGlobalElts, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create input global buffer", res);
    if(m.numInputMetaChannels > 0) {
      inputMeta = vk_helper::createDeviceBuffer(handle->vulkanDevice, dtypeSize * inputMetaElts, false, &res);
    }
    else {
      inputMeta = NULL;
    }

    mask = vk_helper::createDeviceBuffer(handle->vulkanDevice, batchXYElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create mask buffer", res);
    maskSum = vk_helper::createDeviceBuffer(handle->vulkanDevice, batchElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create mask sum buffer", res);
    trunk = vk_helper::createDeviceBuffer(handle->vulkanDevice, m.trunk->trunkNumChannels * batchXYElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create trunk buffer", res);

    if(m.modelVersion >= 16)
      testAssert(m.policyHead->p2Channels == 4);
    else if(m.modelVersion >= 12)
      testAssert(m.policyHead->p2Channels == 2);
    else
      testAssert(m.policyHead->p2Channels == 1);

    policyPassElts = m.policyHead->p2Channels * batchElts;
    policyPass = vk_helper::createDeviceBuffer(handle->vulkanDevice, policyPassElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create policy pass buffer", res);
    policyElts = m.policyHead->p2Channels * batchXYElts;
    policy = vk_helper::createDeviceBuffer(handle->vulkanDevice, policyElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create policy buffer", res);

    valueElts = m.valueHead->valueChannels * batchElts;
    value = vk_helper::createDeviceBuffer(handle->vulkanDevice, valueElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create value buffer", res);

    scoreValueElts = m.valueHead->scoreValueChannels * batchElts;
    scoreValue = vk_helper::createDeviceBuffer(handle->vulkanDevice, scoreValueElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create score value buffer", res);

    ownershipElts = m.valueHead->ownershipChannels * batchXYElts;
    ownership = vk_helper::createDeviceBuffer(handle->vulkanDevice, ownershipElts * dtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create ownership buffer", res);

    // TODO: Implement workspace allocation when winograd or other conv algorithms are added.
    ConvWorkspaceEltsNeeded convWorkspaceElts = m.requiredConvWorkspaceElts(handle);

    std::printf("Conv workspace elts needed: size1=%zu, size2=%zu\n", convWorkspaceElts.size1, convWorkspaceElts.size2);

    convWorkspace = vk_helper::createDeviceBuffer(handle->vulkanDevice, convWorkspaceElts.size1 * sizeof(float), false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create conv workspace buffer", res);
    convWorkspace2 = vk_helper::createDeviceBuffer(handle->vulkanDevice, convWorkspaceElts.size2 * sizeof(float), false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create conv workspace2 buffer", res);
  }

  ~Buffers() {
    vk_helper::releaseVulkanBuffer(input->device, input);
    vk_helper::releaseVulkanBuffer(inputGlobal->device, inputGlobal);
    if(inputMeta != nullptr)
    vk_helper::releaseVulkanBuffer(inputMeta->device, inputMeta);
    vk_helper::releaseVulkanBuffer(mask->device, mask);
    vk_helper::releaseVulkanBuffer(maskSum->device, maskSum);
    vk_helper::releaseVulkanBuffer(trunk->device, trunk);
    vk_helper::releaseVulkanBuffer(policyPass->device, policyPass);
    vk_helper::releaseVulkanBuffer(policy->device, policy);
    vk_helper::releaseVulkanBuffer(value->device, value);
    vk_helper::releaseVulkanBuffer(scoreValue->device, scoreValue);
    vk_helper::releaseVulkanBuffer(ownership->device, ownership);
    if(convWorkspace != nullptr)
      vk_helper::releaseVulkanBuffer(convWorkspace->device, convWorkspace);
    if(convWorkspace2 != nullptr)
      vk_helper::releaseVulkanBuffer(convWorkspace2->device, convWorkspace2);
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

ComputeHandleInternal::ComputeHandleInternal(ComputeContext* ctx, int gpuIdx, bool inputsUseNHWC, bool useNHWC):
  context(ctx),
  vulkanDevice(ctx->vulkanContext->findGpuExn(gpuIdx)),
  tuneParams(ctx->pipelinesPerDev.at(vulkanDevice->info.deviceId)->tuneParams)
{
  this->queue = this->vulkanDevice->queue;
  this->device = this->vulkanDevice->device;
  this->nnXLen = ctx->nnXLen;
  this->nnYLen = ctx->nnYLen;

  if ( usingFP16TensorCoresFor1x1 ) {
    throw StringError("Define paddedNNXYLen required for WMMA 1x1 conv");
  } else {
    this->paddedNNXYLen = nnXLen * nnYLen;
  }

  #ifdef SHADER_PROFILE 
  VkQueryPoolCreateInfo qpCI = {};
  qpCI.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpCI.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpCI.queryCount = 4096;
  VkResult res = vkCreateQueryPool(this->device, &qpCI, nullptr, &queryPool);
  CHECK_VK_MSG("Create query pool", res);
  commandDispatchInfos.reserve(4096);
  #endif
};

void NeuralNet::globalInitialize() {
  static_assert(sizeof(int) >= 4, "");
}

void NeuralNet::globalCleanup() {
}

void NeuralNet::printDevices() {
  VkInstance inst = vk_helper::createVulkanInstance();
  auto infos = vk_helper::enumerateVulkanDevices(inst, nullptr);

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

bool NeuralNet::setIsWarmup(const ComputeHandle* handle, bool isWarmup) {
  (void)handle;
  (void)isWarmup;
  return false;
}

NeuralNet::BatchPolicy NeuralNet::getBatchPolicy(ConfigParser& cfg) {
  (void)cfg;
  return NeuralNet::BatchPolicy::Dynamic;
}

int NeuralNet::getNumEffectiveDevices(ConfigParser& cfg, const std::vector<int>& gpuIdxByServerThread) {
  (void)cfg;
  std::set<int> distinctDevices(gpuIdxByServerThread.begin(), gpuIdxByServerThread.end());
  return std::max(1, (int)distinctDevices.size());
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
  // globalBatchCount = batchSize;

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
    vk_helper::copyHostToDeviceBuffer(
      handle->vulkanDevice,
      inputBuffers->userInputBuffer, // Host pointer
      buffers->input,
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleInputElts),
      false,
      &res
    );
    CHECK_VK_MSG("Copy input buffer to device", res);

    vk_helper::copyHostToDeviceBuffer(
      handle->vulkanDevice,
      inputBuffers->userInputGlobalBuffer, // Host pointer
      buffers->inputGlobal,
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleInputGlobalElts),
      false,
      &res
    );
    CHECK_VK_MSG("Copy input global buffer to device", res);

    if ( numMetaFeatures > 0 ) {
      vk_helper::copyHostToDeviceBuffer(
        handle->vulkanDevice,
        inputBuffers->userInputMetaBuffer, // Host pointer
        buffers->inputMeta,
        static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleInputMetaElts),
        true,
        &res
      );
      CHECK_VK_MSG("Copy input meta buffer to device", res);
    }


    #ifdef VULKAN_API_DEBUG
    VK_BENCHMARK("model->debug",
      computeHandle->model->debug(
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
        buffers->ownership,
        buffers->convWorkspace,
        buffers->convWorkspace2
      );
    );
    #else
    VK_BENCHMARK("model->forward", computeHandle->model->forward(
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
      buffers->ownership,
      buffers->convWorkspace,
      buffers->convWorkspace2
    ););
    #endif

    // Read back PolicyPass result
    vk_helper::copyDeviceBufferToHost(
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
      vk_helper::copyDeviceBufferToHost(
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
    vk_helper::copyDeviceBufferToHost(
      handle->vulkanDevice,
      buffers->value,
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleValueResultElts),
      inputBuffers->valueResults,
      true,
      &res
    );
    CHECK_VK_MSG("Copy value results buffer to host", res);
    vkQueueWaitIdle(handle->queue);

    #ifdef VULKAN_API_DEBUG
    printHostBuffer(
      "[NeuralNet::getOutput] value results",
      inputBuffers->valueResults,
      batchSize * inputBuffers->singleValueResultElts
    );
    #endif

    // Read back ScoreValue result
    vk_helper::copyDeviceBufferToHost(
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
      vk_helper::copyDeviceBufferToHost(
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

    vkResetCommandPool(handle->device, handle->vulkanDevice->commandPool, 0);

  #ifdef SHADER_PROFILE
    handle->dumpShaderProfile();
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
      // #define VULKAN_API_DEBUG
  #ifdef VULKAN_API_DEBUG
  setvbuf(stdout, NULL, _IONBF, 0);
      // Debug: print final output for this row
      std::cout << "=== NNOutput row " << row << " ===" << std::endl;
      std::cout << "whiteWinProb: " << output->whiteWinProb << std::endl;
      std::cout << "whiteLossProb: " << output->whiteLossProb << std::endl;
      std::cout << "whiteNoResultProb: " << output->whiteNoResultProb << std::endl;
      std::cout << "whiteScoreMean: " << output->whiteScoreMean << std::endl;
      std::cout << "whiteScoreMeanSq: " << output->whiteScoreMeanSq << std::endl;
      std::cout << "whiteLead: " << output->whiteLead << std::endl;
      std::cout << "varTimeLeft: " << output->varTimeLeft << std::endl;
      std::cout << "shorttermWinlossError: " << output->shorttermWinlossError << std::endl;
      std::cout << "shorttermScoreError: " << output->shorttermScoreError << std::endl;

      // Print policy probs (top 10 moves)
      std::cout << "policyProbs (top 10): ";
      std::vector<std::pair<float, int>> policyPairs;
      for (int i = 0; i <= nnXLen * nnYLen; i++) {
        policyPairs.push_back({output->policyProbs[i], i});
      }
      std::sort(policyPairs.begin(), policyPairs.end(), [](auto& a, auto& b) { return a.first > b.first; });
      for (int i = 0; i < std::min(10, (int)policyPairs.size()); i++) {
        int idx = policyPairs[i].second;
        if (idx == nnXLen * nnYLen) {
          std::cout << "pass=" << policyPairs[i].first << " ";
        } else {
          int x = idx % nnXLen;
          int y = idx / nnXLen;
          std::cout << "(" << x << "," << y << ")=" << policyPairs[i].first << " ";
        }
      }
      std::cout << std::endl;

      // Print ownership map if available
      if (output->whiteOwnerMap != NULL) {
        std::cout << "whiteOwnerMap (first 19 values): ";
        for (int i = 0; i < std::min(19, nnXLen * nnYLen); i++) {
          std::cout << output->whiteOwnerMap[i] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "========================" << std::endl;
  #endif
    }
  }

  #ifdef VULKAN_API_DEBUG
  // exit(0);
  #endif
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
  std::cout << "[testEvaluateConv] batchSize: " << batchSize
            << " nnXLen: " << nnXLen
            << " nnYLen: " << nnYLen
            << " inChannels: " << desc->inChannels
            << " outChannels: " << desc->outChannels
            << " convYSize: " << desc->convYSize
            << " convXSize: " << desc->convXSize
            << " useFP16: " << (useFP16 ? "true" : "false")
            << " useNHWC: " << (useNHWC ? "true" : "false")
            << std::endl;
  // print default input state;
  std::cout << "[testEvaluateConv] inputBuffer size: " << inputBuffer.size() << std::endl;
  printFloatBuffer("testEvaluateConv Input", inputBuffer.data(), inputBuffer.size(), batchSize, desc->inChannels, nnYLen, nnXLen);
  std::cout << "[testEvaluateConv] filter size: " << desc->inChannels * desc->outChannels * desc->convYSize * desc->convXSize << std::endl;
  printFloatBuffer("testEvaluateConv Filter", desc->weights.data(), desc->inChannels * desc->outChannels * desc->convYSize * desc->convXSize, batchSize, desc->outChannels, desc->convYSize, desc->convXSize);

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
  VulkanBuffer* dInput = vk_helper::createDeviceBufferWithData(
    device,
    byteSizeofVectorContents(inputTmp),
    inputTmp.data(),
    true,
    &res
  );
  CHECK_VK_MSG("[TestConv] Failed to create device input buffer with data", res);

  ConvWorkspaceEltsNeeded needed = layer->requiredConvWorkspaceElts(handle, batchSize);

  size_t maxSize = std::max(needed.size1, needed.size2);

  VulkanBuffer* convWorkspace = vk_helper::createDeviceBuffer(
    device,
    maxSize * sizeof(float),
    false,
    &res
  );

  VulkanBuffer* convWorkspace2 = vk_helper::createDeviceBuffer(
    device,
    maxSize * sizeof(float),
    false,
    &res
  );

  VulkanBuffer* dOutput = vk_helper::createDeviceBuffer(
    device,
    byteSizeofVectorContents(outputBuffer),
    false,
    &res
  );
  CHECK_VK_MSG("[TestConv] Failed to create device buffer", res);

  layer->debug(batchSize,dInput,dOutput, nullptr, nullptr, convWorkspace, convWorkspace2);
  // layer->apply(batchSize, dInput, dOutput);
  // vk_helper::submitCommandBuffers(handle->vulkanDevice, {layer->commandBuffer}, nullptr);
  vk_helper::copyDeviceBufferToHost(device, dOutput, static_cast<VkDeviceSize>(sizeof(float) * numOutputFloats), outputBuffer.data(), true, &res);
  CHECK_VK_MSG("[TestConv] Failed to copy device output buffer to host", res);
  vkQueueWaitIdle(device->queue);
  vkDeviceWaitIdle(device->device);
  vk_helper::releaseVulkanBuffer(device, dInput);
  vk_helper::releaseVulkanBuffer(device, dOutput);
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
    VulkanBuffer* dInput = vk_helper::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(inputTmp),
      inputTmp.data(),
      true,
      &res
    );
    CHECK_VK_MSG("[testEvaluateBatchNorm] Failed to create device input buffer with data", res);
    VulkanBuffer* dMask = vk_helper::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(maskTmp),
      maskTmp.data(),
      true,
      &res
    );
    CHECK_VK_MSG("[testEvaluateBatchNorm] Failed to create device mask buffer", res);
    VulkanBuffer* dOutput = vk_helper::createDeviceBuffer(
      handle->vulkanDevice,
      byteSizeofVectorContents(outputBuffer),
      false,
      &res
    );
    CHECK_VK_MSG("[testEvaluateBatchNorm] Failed to create device output buffer", res);

    layer->debug(batchSize, dInput, dMask, dOutput);
    // vk_helper::submitCommandBuffers(handle->vulkanDevice, {layer->commandBuffer}, nullptr);
    // layer->apply(batchSize, dInput, dMask, dOutput);
    vk_helper::copyDeviceBufferToHost(handle->vulkanDevice, dOutput, static_cast<VkDeviceSize>(sizeof(float) * outputBuffer.size()), outputBuffer.data(), true, &res);
    CHECK_VK_MSG("[testEvaluateBatchNorm] Failed to copy device output buffer to host", res);
    // delete dOutput;
    // delete dMask;
    // delete dInput;
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, dInput);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, dMask);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, dOutput);
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
    VulkanBuffer* dTrunk = vk_helper::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(inputTmp),
      inputTmp.data(),
      false,
      &res
    );
    CHECK_VK_MSG("[testEvaluateResidualBlock] Failed to create device trunk buffer with data", res);
    VulkanBuffer* dMask = vk_helper::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(maskTmp),
      maskTmp.data(),
      true,
      &res
    );
    CHECK_VK_MSG("[testEvaluateResidualBlock] Failed to create device mask buffer with data", res);
    VulkanBuffer* dTrunkScratch = vk_helper::createDeviceBuffer(
      handle->vulkanDevice,
      numTrunkFloats * sizeof(float),
      false,
      &res
    );
    CHECK_VK_MSG("[testEvaluateResidualBlock] Failed to create device trunk scratch buffer", res);

    VulkanBuffer *convWorkspace = vk_helper::createDeviceBuffer(
      handle->vulkanDevice,
      layer->requiredConvWorkspaceElts(handle, batchSize).size1 * sizeof(float),
      false,
      &res
    );

    VulkanBuffer *convWorkspace2 = vk_helper::createDeviceBuffer(
      handle->vulkanDevice,
      layer->requiredConvWorkspaceElts(handle, batchSize).size2 * sizeof(float),
      false,
      &res
    );

    layer->debug(batchSize, scratch, dTrunk, dTrunkScratch, dMask, convWorkspace, convWorkspace2);
    // layer->apply(batchSize, scratch, dTrunk, dTrunkScratch, dMask, convWorkspace, convWorkspace2);
    // vk_helper::submitCommandBuffers(handle->vulkanDevice, layer->commandBuffers, nullptr);
    vk_helper::copyDeviceBufferToHost(handle->vulkanDevice, dTrunk, static_cast<VkDeviceSize>(sizeof(float) * numTrunkFloats), outputBuffer.data(), true, &res);
    CHECK_VK_MSG("[testEvaluateResidualBlock] Failed to copy device trunk buffer to host", res);
    vkQueueWaitIdle(handle->vulkanDevice->queue);
    vkDeviceWaitIdle(handle->vulkanDevice->device);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, dTrunk);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, dMask);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, dTrunkScratch);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, convWorkspace);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, convWorkspace2);
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
    VulkanBuffer* dTrunk = vk_helper::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(inputTmp),
      inputTmp.data(),
      false,
      &res
    );
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to create device trunk buffer with data", res);
    VulkanBuffer* dMask = vk_helper::createDeviceBufferWithData(
      handle->vulkanDevice,
      byteSizeofVectorContents(maskTmp),
      maskTmp.data(),
      true,
      &res
    );
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to create device mask buffer with data", res);
    VulkanBuffer* dTrunkScratch = vk_helper::createDeviceBuffer(
      handle->vulkanDevice,
      numTrunkFloats * sizeof(float),
      false,
      &res
    );
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to create device trunk scratch buffer", res);
    VulkanBuffer* dMaskSum = vk_helper::createDeviceBuffer(
      handle->vulkanDevice,
      numMaskSumFloats * sizeof(float),
      false,
      &res
    );
    VulkanBuffer* convWorkspace = vk_helper::createDeviceBuffer(
      handle->vulkanDevice,
      layer->requiredConvWorkspaceElts(handle, batchSize).size1 * sizeof(float),
      false,
      &res
    );
    VulkanBuffer* convWorkspace2 = vk_helper::createDeviceBuffer(
      handle->vulkanDevice,
      layer->requiredConvWorkspaceElts(handle, batchSize).size2 * sizeof(float),
      false,      
      &res
    );
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to create device mask sum buffer", res);
    std::vector<float> maskSumTmp(numMaskSumFloats, 0.0f);
    VkCommandBuffer maskSumsCB = VK_NULL_HANDLE;
    VkDescriptorSet maskSumsDS = VK_NULL_HANDLE;
    computeMaskSums(handle, maskSumsCB, maskSumsDS, batchSize, nnXLen, nnYLen, dMask, dMaskSum);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {maskSumsCB}, nullptr);
    vk_helper::copyDeviceBufferToHost(handle->vulkanDevice, dMaskSum, static_cast<VkDeviceSize>(sizeof(float) * numMaskSumFloats), maskSumTmp.data(), true, &res);
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to copy device mask sum buffer to host", res);
    // printFloatBuffer("[testEvaluateGlobalPoolingResidualBlock] Mask Sums", maskSumTmp.data(), maskSumTmp.size(), batchSize, 1, 1, 1);
    layer->debug(batchSize, scratch, dTrunk, dTrunkScratch, dMask, dMaskSum, convWorkspace, convWorkspace2);
    // layer->apply(batchSize, scratch, dTrunk, dTrunkScratch, dMask, dMaskSum, convWorkspace, convWorkspace2);
    vk_helper::copyDeviceBufferToHost(handle->vulkanDevice, dTrunk, static_cast<VkDeviceSize>(sizeof(float) * numTrunkFloats), outputBuffer.data(), true, &res);
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to copy device trunk buffer to host", res);
    vkQueueWaitIdle(handle->vulkanDevice->queue);
    vkDeviceWaitIdle(handle->vulkanDevice->device);

    printFloatBuffer("[testEvaluateGlobalPoolingResidualBlock] Output", outputBuffer.data(), outputBuffer.size(), batchSize, desc->preBN.numChannels, nnYLen, nnXLen);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, dTrunk);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, dMask);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, dTrunkScratch);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, dMaskSum);
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


  // ########################### End of Compute Pipelines #########################


#endif
