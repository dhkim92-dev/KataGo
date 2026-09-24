/**
 * @file vulkanbackend.cpp
 * @author dhkim92-dev
 * @brief Vulkan backend for Neural Net evaluation
 */
#ifdef USE_VULKAN_BACKEND

#include <unordered_map>
#include <memory>
#include <numeric>
#include <chrono>
#include <iostream>
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

struct ComputeContext;
using namespace vk_shader::push;

static int checkedTotalElts(int64_t a, int64_t b, int64_t c, const char* whatKernel) {
  int64_t total = a * b * c;
  if(total >= (int64_t)2147483647)
    throw StringError(
      std::string(whatKernel) + ": total element count " + Global::int64ToString(total) +
      " exceeds the 32-bit index limit used by this kernel");
  return (int)total;
}

static int checkedTensorElts(
  const ComputeHandleInternal* handle,
  int64_t batchSize,
  int channels,
  int64_t spatialSize,
  const char* whatKernel
) {
  const int storageChannels = handle->getNHWCChannelsPadded(channels);
  return checkedTotalElts(batchSize, storageChannels, spatialSize, whatKernel);
}

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

static void printHostBuffer(
  std::string prefix,
  const float* hostBuffer,
  size_t numElts,
  bool summarized = true
) {
#ifdef VK_DUMP_BUFFER
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
#ifdef VK_DUMP_BUFFER

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
#ifdef VK_DUMP_BUFFER
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
  std::unordered_map<uint32_t, vk_shader::tune::VulkanTuneParams> tuneParamsPerDev;
  std::pair<int, int> transformerHeadDims = {0, 0};
  Logger* logger;

  static void findTransformerHeadDims(
    const std::vector<std::pair<int, unique_ptr_void>>& blocks,
    std::pair<int, int>& headDims,
    bool& foundHeadDims
  ) {
    for ( const auto& block : blocks ) {
      if ( block.first == TRANSFORMER_ATTENTION_BLOCK_KIND ) {
        const TransformerAttentionDesc* attentionDesc =
          static_cast<const TransformerAttentionDesc*>(block.second.get());
        const std::pair<int, int> currentHeadDims = {
          attentionDesc->qHeadDim,
          attentionDesc->vHeadDim
        };

        if ( !foundHeadDims ) {
          headDims = currentHeadDims;
          foundHeadDims = true;
        } else if ( headDims != currentHeadDims ) {
          throw StringError(
            "Vulkan transformer attention blocks use different qHeadDim/vHeadDim combinations: ("
            + std::to_string(headDims.first) + ", " + std::to_string(headDims.second)
            + ") and (" + std::to_string(currentHeadDims.first) + ", "
            + std::to_string(currentHeadDims.second) + ")"
          );
        }
      } else if ( block.first == NESTED_BOTTLENECK_BLOCK_KIND ) {
        const NestedBottleneckResidualBlockDesc* nestedDesc =
          static_cast<const NestedBottleneckResidualBlockDesc*>(block.second.get());
        findTransformerHeadDims(nestedDesc->blocks, headDims, foundHeadDims);
      }
    }
  }

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
    const ModelDesc* modelDesc)
  : nnXLen(nnXLen),
    nnYLen(nnYLen),
    usingFP16Mode(useFP16Mode_),
    usingNHWCMode(useNHWCMode_),
    gIdx(gpuIdxsToUse),
    logger(logger_)
     {
      if ( modelDesc != nullptr ) {
        bool foundHeadDims = false;
        findTransformerHeadDims(modelDesc->trunk.blocks, transformerHeadDims, foundHeadDims);
      }

      VkInstance instance = vk_helper::createVulkanInstance();
      std::vector<VulkanDeviceInfo> allDeviceInfos = vk_helper::enumerateVulkanDevices(instance, logger);
      std::vector<VulkanDevice *> vulkanDevices = {};

      if ( gpuIdxsToUse.size() == 1 && gpuIdxsToUse[0] == UINT32_MAX ) {
        if ( logger ) {
          logger->write("No GPU index specified, using default GPU 0");
        }

        // TODO: select device to use using config
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

        const VulkanParams hardwareParams = VulkanTuner::getHardwareParams(deviceInfo);
        const bool supportsFP16Storage = hardwareParams.canUseFP16Storage;
        const bool supportsFP16Compute = hardwareParams.canUseFP16Compute;

        if ( usingFP16Mode == enabled_t::True && (!supportsFP16Storage || !supportsFP16Compute) ) {
          throw StringError("Requested FP16 mode but device " + deviceInfo.deviceName + " does not support FP16 storage and compute");
        }

        // The tuner probes hardware FP16 capability independently of whether
        // this run will use FP16. Actual use is controlled by shouldUseFP16*.
        if ( supportsFP16Compute ) {
          requiredExtensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        }
        if (
          supportsFP16Storage &&
          VK_VERSION_MAJOR(deviceInfo.properties.apiVersion) == 1 &&
          VK_VERSION_MINOR(deviceInfo.properties.apiVersion) < 1
        ) {
          requiredExtensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
        }

        if (deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE) {
          requiredExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        }

        VulkanDevice* vulkanDevice = vk_helper::createVulkanDevice(
          instance,
          deviceInfo,
          requiredExtensions,
          logger
        );
        vk_shader::ComputePipelines* pipelines = nullptr;
        VulkanTuneParams tuneParams;
        try {
          if(modelInfo != nullptr) {
            tuneParams = VulkanTuner::loadOrAutoTune(
              tunerFile,homeDataDirOverride,deviceInfo.deviceName,
              nnXLen,nnYLen,*modelInfo,vulkanDevice,logger,nullptr
            );
          }

          // Pipeline creation must use the same effective precision as the
          // buffers created by ComputeHandleInternal.  The tuning file stores
          // the Auto-mode decision, but useFP16=true/false overrides that
          // decision for this process.
          const bool useFP16Storage =
            usingFP16Mode == enabled_t::True
              ? supportsFP16Storage && supportsFP16Compute
              : usingFP16Mode == enabled_t::Auto
                ? tuneParams.vulkan.canUseFP16Storage &&
                  tuneParams.vulkan.canUseFP16Compute &&
                  tuneParams.vulkan.shouldUseFP16Storage
                : false;
          const bool useFP16Compute =
            usingFP16Mode == enabled_t::True
              ? supportsFP16Compute
              : usingFP16Mode == enabled_t::Auto
                ? tuneParams.vulkan.canUseFP16Compute &&
                  tuneParams.vulkan.shouldUseFP16Compute
                : false;
          tuneParams.vulkan.shouldUseFP16Storage = useFP16Storage;
          tuneParams.vulkan.shouldUseFP16Compute = useFP16Compute;
          if(!useFP16Storage || !useFP16Compute)
            tuneParams.vulkan.shouldUseTransformerDualGemmSwiGLU = false;

          // Keep model tensors in NCHW for this ConvLayer-only NHWC experiment.
          const bool isTransformerModel = transformerHeadDims.first > 0 && transformerHeadDims.second > 0;
          const bool useNHWC = usingNHWCMode == enabled_t::True ||
            (usingNHWCMode == enabled_t::Auto && isTransformerModel);
          const bool useCooperativeMatrixMode =
            tuneParams.vulkan.shouldUseCooperativeMatrix &&
            tuneParams.vulkan.canUseCooperativeMatrix &&
            tuneParams.vulkan.canUseFP16Storage &&
            tuneParams.vulkan.canUseFP16Compute &&
            tuneParams.vulkan.shouldUseFP16Storage &&
            tuneParams.vulkan.shouldUseFP16Compute;
          tuneParams.vulkan.shouldUseCooperativeMatrix = useCooperativeMatrixMode;
          if(!useCooperativeMatrixMode) {
            tuneParams.transformer.USE_COOPERATIVE_ATTN = 0;
            tuneParams.vulkan.shouldUseTransformerDualGemmSwiGLU = false;
          }
          pipelines = new vk_shader::ComputePipelines(vulkanDevice->device, vulkanDevice->info, logger);
          VkResult result = pipelines->createPipelines(
            tuneParams, transformerHeadDims.first, transformerHeadDims.second, useNHWC, true
          );
          if(result != VK_SUCCESS)
            throw StringError("Failed to create Vulkan compute pipelines: " + vk_helper::vkErrorToString(result));
        }
        catch(...) {
          delete pipelines;
          delete vulkanDevice;
          throw;
        }
        vulkanDevices.push_back(vulkanDevice);
        if ( logger ) {
          logger->write("Created Vulkan Compute Pipelines for device: " + deviceInfo.deviceName);
        }
        this->pipelinesPerDev.emplace(gpuIdx, pipelines);
        this->tuneParamsPerDev.emplace(gpuIdx, tuneParams);
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

  bool isDeviceSupportNHWC(const VulkanDeviceInfo& deviceInfo) {
    // return deviceInfo.properties.apiVersion >= VK_API_VERSION_1_1 &&
          //  deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE;
      return false;
  }
};


/**
 * @deprecated
 */
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
#ifdef VK_DUMP_BUFFER
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

    // TODO useFP16 support

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
    if ( matBuf != nullptr ) {
      vk_helper::releaseVulkanBuffer(handle->vulkanDevice, matBuf);
      matBuf = nullptr;
    }
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
    doBatchedXGEMMDirectFP32_MK_NK_MN(cb, batchSize, input, output);
    vk_helper::barrierCommandBufferForBuffer(cb, output);
    // vk_helper::barrierCommandBuffer(cb);
  }

private:
  void doBatchedXGEMMDirectFP32_MK_NK_MN(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    // uint32_t gpuId = handle->vulkanDevice->info.deviceId;
    // auto pipelines = handle->context->pipelinesPerDev.at(gpuId);
    VkResult res;
    Pipeline pipeline = handle->pipelines->xgemmDirectBatchedTT;
    if( descriptorSet == VK_NULL_HANDLE ) {
      descriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        pipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for BatchedXGEMMDirectFP32_MK_NK_MN MatmulLayer: " + name, res);
    }
    vkcompute::batchedXGemmDirect_MK_NK_MN(
      handle,
      handle->vulkanDevice,
      handle->tuneParams,
      &pipeline,
      cb,
      descriptorSet,
      batchSize, outChannels, inChannels,
      input, matBuf, output,
      1, &res
    );
    vk_helper::barrierCommandBufferForBuffer(cb, output);
    // vk_helper::barrierCommandBuffer(cb);
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
  const int paddedNNXYLen;

  VulkanBuffer* mergedScaleBuf;
  VulkanBuffer* mergedBiasBuf;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet nhwcDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet nchwToNhwcDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet nhwcToNchwDescriptorSet = VK_NULL_HANDLE;
  Pipeline pipeline;
  const bool useNHWC;

  ~BatchNormLayer() {
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
  }

  BatchNormLayer(
    ComputeHandleInternal *handle_,
    const BatchNormLayerDesc* desc,
    const ActivationLayerDesc* actDesc,
    bool useFP16
  ):
    handle(handle_),
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    activation(actDesc->activation),
    paddedNNXYLen(handle_->paddedNNXYLen),
    useNHWC(handle_->pipelines->useNHWC)
  {
    assert(desc->mean.size() == static_cast<size_t>(numChannels));
    assert(desc->variance.size() == static_cast<size_t>(numChannels));
    assert(desc->scale.size() == static_cast<size_t>(numChannels));
    assert(desc->bias.size() == static_cast<size_t>(numChannels));
    assert(desc->mergedScale.size() == static_cast<size_t>(numChannels));
    assert(desc->mergedBias.size() == static_cast<size_t>(numChannels));

    // Precompute merged scale and bias
    std::vector<float> mergedScale = desc->mergedScale;
    std::vector<float> mergedBias = desc->mergedBias;

    VkResult res;

    mergedBiasBuf = vk_helper::createReadOnlyBuffer(
      handle->vulkanDevice,
      mergedBias,
      useFP16,
      &res
    );
    CHECK_VK_MSG("Create BatchNormLayer: " + name + " merged bias buffer", res);

    mergedScaleBuf = vk_helper::createReadOnlyBuffer(
      handle->vulkanDevice,
      mergedScale,
      useFP16,
      &res
    );
    CHECK_VK_MSG("Create BatchNormLayer: " + name + " merged scale buffer", res);
    const vk_shader::ComputePipelines* pipelines = this->handle->pipelines;

    switch ( activation ) {
      case ACTIVATION_IDENTITY:
        pipeline = pipelines->batchNormMaskIdentity;
        break;
      case ACTIVATION_RELU:
        pipeline = pipelines->batchNormMaskRelu;
        break;
      case ACTIVATION_MISH:
        pipeline = pipelines->batchNormMaskMish;
        break;
      case ACTIVATION_MISH_SCALE8:
        pipeline = pipelines->batchNormMaskMishScale8;
        break;
      case ACTIVATION_SILU:
        pipeline = pipelines->batchNormMaskSilu;
        break;
      default:
        Global::fatalError("Unsupported activation in BatchNormLayer: " + name);
    }
    descriptorSet = vk_helper::allocateDescriptorSet(handle->vulkanDevice, pipeline.descriptorSetLayout, &res);
    CHECK_VK_MSG("Allocate descriptor set for BatchNormLayer: " + name, res);
  }

  void forward(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* mask,
    VulkanBuffer* output,
    VulkanBuffer* nhwcInput = nullptr,
    VulkanBuffer* nhwcOutput = nullptr,
    int nhwcSpatialSize = 0,
    int logicalSpatialSize = 0
  ) {
    assert(cb != VK_NULL_HANDLE);
    if(useNHWC) {
      forwardNHWC(
        cb, batchSize, input, mask, output, nhwcInput, nhwcOutput,
        paddedNNXYLen, logicalSpatialSize
      );
      return;
    }
    
    vkcompute::doBatchNormMask(
      handle, &pipeline, cb, descriptorSet, input, output,
      mergedScaleBuf, mergedBiasBuf, mask, batchSize, numChannels,
      paddedNNXYLen, paddedNNXYLen,
      useNHWC ? handle->getNHWCChannelsPadded(numChannels) : numChannels,
      "BATCHNORM_MASK_FP32"
    );
    // res = vk_helper::endCommandBuffer(cb);
    // CHECK_VK_MSG("End command buffer for BatchNormLayer: " + name, res);
  }

  void forwardNHWC(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* mask,
    VulkanBuffer* output,
    VulkanBuffer* nhwcInput,
    VulkanBuffer* nhwcOutput,
    int nhwcSpatialSize,
    int logicalSpatialSize
  ) {
    assert(cb != VK_NULL_HANDLE);
    assert(input != nullptr && mask != nullptr && output != nullptr);
    assert(nhwcSpatialSize >= logicalSpatialSize);
    (void)nhwcInput;
    (void)nhwcOutput;

    VkResult res = VK_ERROR_UNKNOWN;
    if(nhwcDescriptorSet == VK_NULL_HANDLE) {
      nhwcDescriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, pipeline.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("Allocate NHWC BatchNormLayer descriptor set: " + name, res);
    }

    vkcompute::doBatchNormMask(
      handle, &pipeline, cb, nhwcDescriptorSet, input, output,
      mergedScaleBuf, mergedBiasBuf, mask, batchSize, numChannels,
      paddedNNXYLen, paddedNNXYLen,
      handle->getNHWCChannelsPadded(numChannels), "BATCHNORM_MASK_NHWC"
    );
  }

  /**
   * @brief Launch the recorded command buffer, only for debug now.
   * @param batchSize
   * @param input
   * @param mask
   * @param output
   */

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(
    ComputeHandleInternal* handle_,
    size_t maxBatchSize
  ) const {
    if(!useNHWC)
      return ConvWorkspaceEltsNeeded();
    const size_t channelsPadded = handle->getNHWCChannelsPadded(numChannels);
    const size_t logicalSpatialSize = static_cast<size_t>(handle_->nnXLen) * static_cast<size_t>(handle_->nnYLen);
    const size_t conversionElts = maxBatchSize * logicalSpatialSize * channelsPadded;
    return ConvWorkspaceEltsNeeded(conversionElts, conversionElts);
  }

};


/**
 * @brief Convolution Layer in Vulkan Backend
 * Currently not support winograd and dilation
 * Simple tiled convolution only except 1x1 conv.
 * 1x1 conv implemented with matmul approach. Maybe replaced by cooperative matrix extension later.
 */
struct ConvLayer {
  // ConvLayer may use an NHWC execution path while the surrounding model stays NCHW.
  enum class NhwcConvPath {
    None,
    DirectMatmul1x1,
    Winograd,
  };

  ComputeHandleInternal* handle;
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
  const int paddedNNXYLen;

  int numTilesX;
  int numTilesY;
  int inTileXYSize;
  int outTileXYSize;

  bool usingHgemmCooperativeMatrix;
  bool usingHgemmCooperativeMatrixNCHW;
  bool usingNHWC;
  NhwcConvPath nhwcConvPath;
  int nhwcSpatialSize;
  int nhwcKSize;
  int nhwcNSize;

  VulkanBuffer* filterBuf = nullptr;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

  VkDescriptorSet winogradInputTransformDS = VK_NULL_HANDLE;
  VkDescriptorSet winogradOutputTransformDS = VK_NULL_HANDLE;
  VkDescriptorSet xgemmBatchedDS = VK_NULL_HANDLE;
  VkDescriptorSet hgemmCooperativeMatrixDS = VK_NULL_HANDLE;
  VkDescriptorSet hgemmCooperativeMatrixNHWCDS = VK_NULL_HANDLE;
  VkDescriptorSet nchwToNhwcDS = VK_NULL_HANDLE;
  VkDescriptorSet nhwcToNchwDS = VK_NULL_HANDLE;

  VulkanBuffer* bnScaleBuf = nullptr; // For batchnorm scale
  VulkanBuffer* bnBiasBuf = nullptr;  // For batchnorm bias

  static constexpr int nKernelDims = 3;
  uint32_t act;

  const vk_shader::tune::HGemmCooperativeMatrixNHWCTuneParams& getNhwcConvTuneParams() const {
    return handle->tuneParams.hgemmCooperativeMatrixNHWC;
  }

  ConvLayer(
    ComputeHandleInternal *handle_,
    const ConvLayerDesc* desc,
    int nnXLen_,
    int nnYLen_,
    bool useFP16
  ):
    handle(handle_),
    name(desc->name),
    convYSize(desc->convYSize),
    convXSize(desc->convXSize),
    convYRadius(desc->convYSize/2),
    convXRadius(desc->convXSize/2),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels),
    dilationY(desc->dilationY),
    dilationX(desc->dilationX),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    paddedNNXYLen(handle_->paddedNNXYLen)
  {
    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);

    if ( dilationX != 1 || dilationY != 1 ) {
      throw StringError("Vulkan ConvLayer: " + name + " dilation not supported yet");
    }

    usingHgemmCooperativeMatrix = false;
    usingHgemmCooperativeMatrixNCHW = false;
    usingNHWC = false;
    nhwcConvPath = NhwcConvPath::None;
    nhwcSpatialSize = 0;
    nhwcKSize = 0;
    nhwcNSize = 0;
    VkResult res;
    numTilesX = 0;
    numTilesY = 0;
    inTileXYSize = 0;
    outTileXYSize = 0;

    if ((convXSize == 1 && convYSize == 1) ||
        (convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5)) {
      const auto& hgemmParams = getNhwcConvTuneParams();
      usingHgemmCooperativeMatrix =
        handle_->usingFP16Storage &&
        handle_->tuneParams.vulkan.canUseCooperativeMatrix &&
        handle_->tuneParams.vulkan.shouldUseCooperativeMatrix &&
        handle_->tuneParams.vulkan.canUseFP16Storage &&
        handle_->tuneParams.vulkan.canUseFP16Compute &&
        handle_->tuneParams.vulkan.shouldUseFP16Storage &&
        handle_->tuneParams.vulkan.shouldUseFP16Compute &&
        hgemmParams.isValid();
      usingNHWC = usingHgemmCooperativeMatrix;
    }

    if(convXSize == 1 && convYSize == 1) {
      const auto& hgemmParams = handle_->tuneParams.hgemmCooperativeMatrixNCHW;
      usingHgemmCooperativeMatrixNCHW =
        handle_->usingFP16Storage &&
        handle_->tuneParams.vulkan.canUseCooperativeMatrix &&
        handle_->tuneParams.vulkan.shouldUseFP16Storage &&
        handle_->tuneParams.vulkan.shouldUseFP16Compute &&
        handle_->tuneParams.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
        hgemmParams.isValid() &&
        inChannels % hgemmParams.KWG == 0 &&
        outChannels % hgemmParams.NWG == 0;
    }

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
    int inChannelsPadded = vk_helper::roundUpToMultipleInt(inChannels, handle->getXGemmKPaddingMult());
    int outChannelsPadded = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());

    if(usingNHWC) {
      // The NHWC tensor batch stride is the model-wide padded spatial stride.
      // Use the model-wide padded spatial stride for the NHWC tensor layout.
      nhwcSpatialSize = handle_->paddedNNXYLen;
      if(convXSize == 1 && convYSize == 1) {
        nhwcKSize = handle_->getNHWCChannelsPadded(inChannels, usingNHWC);
        // The matrix N extent must cover the full model-wide NHWC row stride.
        // A smaller NWG-rounded extent would leave its tail unwritten, even
        // though a following 1x1 GEMM can consume that tail as K padding.
        nhwcNSize = handle_->getNHWCChannelsPadded(outChannels, usingNHWC);
        // 1x1 convolution is exactly a matrix multiplication.  The
        // model-wide NHWC channel stride is padded to lcm(NWG, KWG), so it
        // always contains the cooperative GEMM K padding.
        nhwcConvPath = NhwcConvPath::DirectMatmul1x1;
      } else if((convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5)) {
        const auto& convParams = convXSize == 3
          ? handle_->tuneParams.conv3x3 : handle_->tuneParams.conv5x5;
        nhwcKSize = handle_->getNHWCChannelsPadded(inChannels, usingNHWC);
        nhwcNSize = handle_->getNHWCChannelsPadded(outChannels, usingNHWC);
        numTilesX = (nnXLen + convParams.outTileXSize - 1) / convParams.outTileXSize;
        numTilesY = (nnYLen + convParams.outTileYSize - 1) / convParams.outTileYSize;
        inTileXYSize = convParams.inTileXSize * convParams.inTileYSize;
        outTileXYSize = convParams.outTileXSize * convParams.outTileYSize;
        nhwcConvPath = NhwcConvPath::Winograd;
      }
    }

    if (convXSize == 1 && convYSize == 1) {
      if(usingNHWC) {
        std::vector<float> nhwcGemmWeights = vkcompute::convWeightsToNHWC1x1Gemm(
          desc->weights, inChannels, outChannels, nhwcKSize, nhwcNSize
        );
        filterBuf = vk_helper::createReadOnlyBuffer(handle->vulkanDevice, nhwcGemmWeights, true, &res);
      } else {
        std::vector<float> transWeights(inChannels * outChannels);
        for (int oc = 0; oc < outChannels; oc++) {
          for (int ic = 0; ic < inChannels; ic++) {
            transWeights[ic * outChannels + oc] = desc->weights[oc * inChannels + ic];
          }
        }
        filterBuf = vk_helper::createReadOnlyBuffer(
          handle->vulkanDevice,
          transWeights,
          usingHgemmCooperativeMatrixNCHW || useFP16,
          &res
        );
      }
    } else if(nhwcConvPath == NhwcConvPath::Winograd) {
      const int inChannelsPaddedForNhwc = handle_->getNHWCChannelsPadded(inChannels, usingNHWC);
      const int outChannelsPaddedForNhwc = handle_->getNHWCChannelsPadded(outChannels, usingNHWC);
      std::vector<float> winogradWeights = vkcompute::convWeightsToWinogradDomain(
        desc->weights, inChannels, inChannelsPaddedForNhwc,
        outChannels, outChannelsPaddedForNhwc,
        convYSize, convXSize,
        convYSize == 3 ? handle_->tuneParams.conv3x3.inTileYSize : handle_->tuneParams.conv5x5.inTileYSize,
        convXSize == 3 ? handle_->tuneParams.conv3x3.inTileXSize : handle_->tuneParams.conv5x5.inTileXSize
      );
      filterBuf = vk_helper::createReadOnlyBuffer(
        handle->vulkanDevice, winogradWeights, true, &res
      );
    } else if( (convXSize == 3 && convYSize == 3) || (convXSize==5 &&convYSize == 5)) {
      // outTilesY = handle->tuneParams.conv3x3.outTileYSize;
      // outTilesX = handle->tuneParams.conv3x3.outTileXSize;
      // inTilesY = handle->tuneParams.conv3x3.inTileYSize;
      // inTilesX = handle->tuneParams.conv3x3.inTileXSize;
      int inTileXSize = convXSize == 3 ? handle->tuneParams.conv3x3.inTileXSize : handle->tuneParams.conv5x5.inTileXSize;
      int inTileYSize = convYSize == 3 ? handle->tuneParams.conv3x3.inTileYSize : handle->tuneParams.conv5x5.inTileYSize;
      int outTileXSize = convXSize == 3 ? handle->tuneParams.conv3x3.outTileXSize : handle->tuneParams.conv5x5.outTileXSize;
      int outTileYSize = convYSize == 3 ? handle->tuneParams.conv3x3.outTileYSize : handle->tuneParams.conv5x5.outTileYSize;

      int outChannelsPadded = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
      int inChannelsPadded = vk_helper::roundUpToMultipleInt(inChannels, handle->getXGemmKPaddingMult());

      numTilesX = (nnXLen + outTileXSize - 1) / outTileXSize;
      numTilesY = (nnYLen + outTileYSize - 1) / outTileYSize;
      inTileXYSize = inTileXSize * inTileYSize;
      outTileXYSize = outTileXSize * outTileYSize;

      static constexpr int maxTileXSize = 6;
      static constexpr int maxTileYSize = 6;

      testAssert((convXSize == 3 && convYSize == 3) ? (inTileXSize == 4 && outTileXSize == 2) || (inTileXSize == 6 && outTileXSize == 4) : true);
      testAssert((convXSize == 5 && convYSize == 5) ? (inTileYSize == 6 && outTileYSize == 2) : true);

      std::vector<float> winogradWeights = vkcompute::convWeightsToWinogradDomain(
        desc->weights, inChannels, inChannelsPadded,
        outChannels, outChannelsPadded,
        convYSize, convXSize,
        inTileYSize, inTileXSize
      );

      filterBuf = vk_helper::createReadOnlyBuffer(
        handle->vulkanDevice,
        winogradWeights,
        useFP16 || usingHgemmCooperativeMatrix,
        &res
      );
    } else {
      // For larger convolutions, use weights as-is
      filterBuf = vk_helper::createReadOnlyBuffer(
        handle->vulkanDevice,
        desc->weights,
        useFP16,
        &res
      );
    }
    CHECK_VK_MSG("Create ConvLayer: " + name + " filter buffer", res);
  }

  ~ConvLayer() {
    if ( filterBuf != nullptr ) {
      vk_helper::releaseVulkanBuffer(handle->vulkanDevice, filterBuf);
      filterBuf = nullptr;
    }
  }

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  bool isBNActFusedPossible() {
    // KataGo's NormActConv is pre-activation (BN/activation before Conv),
    // whereas the NHWC path uses an output epilogue. Keep the existing
    // Winograd input fusion, but do not substitute an output epilogue for the
    // NHWC pre-activation operation.
    return (usingNHWC && nhwcConvPath == NhwcConvPath::Winograd) ||
      (!usingNHWC &&
       ((convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5)));
  }

  bool needsNCHWBoundaryConversion() const {
    return usingNHWC && !handle->pipelines->useNHWC;
  }

  void convertNCHWInputToNHWC(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* nhwcInput
  ) {
    VkResult res = VK_ERROR_UNKNOWN;
    if(nchwToNhwcDS == VK_NULL_HANDLE) {
      nchwToNhwcDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, handle->pipelines->nchwToNhwc.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("Allocate ConvLayer NCHW-to-NHWC descriptor set: " + name, res);
    }
    vkcompute::convertNCHWToNHWC(
      handle, handle->vulkanDevice, &handle->pipelines->nchwToNhwc,
      cb, nchwToNhwcDS, input, nhwcInput, batchSize, inChannels,
      paddedNNXYLen, paddedNNXYLen, nnXLen * nnYLen, &res
    );
    CHECK_VK_MSG("Convert ConvLayer input from NCHW to NHWC: " + name, res);
  }

  void convertNHWCOutputToNCHW(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* nhwcOutput,
    VulkanBuffer* output
  ) {
    VkResult res = VK_ERROR_UNKNOWN;
    if(nhwcToNchwDS == VK_NULL_HANDLE) {
      nhwcToNchwDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, handle->pipelines->nhwcToNchw.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("Allocate ConvLayer NHWC-to-NCHW descriptor set: " + name, res);
    }
    vkcompute::convertNHWCToNCHW(
      handle, handle->vulkanDevice, &handle->pipelines->nhwcToNchw,
      cb, nhwcToNchwDS, nhwcOutput, output, batchSize, outChannels,
      paddedNNXYLen, paddedNNXYLen, nnXLen * nnYLen, &res
    );
    CHECK_VK_MSG("Recover ConvLayer output to NCHW: " + name, res);
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) {
    int numTilesTotalPadded = vk_helper::roundUpToMultipleInt(maxBatchSize * numTilesY * numTilesX, handle->getXGemmMPaddingMult());
    int outChannelsPadded = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
    int inChannelsPadded = vk_helper::roundUpToMultipleInt(inChannels, handle->getXGemmKPaddingMult());

    ConvWorkspaceEltsNeeded needed {
      static_cast<size_t>(numTilesTotalPadded) * static_cast<size_t>(inChannelsPadded) * static_cast<size_t>(inTileXYSize),
      static_cast<size_t>(numTilesTotalPadded) * static_cast<size_t>(outChannelsPadded) * static_cast<size_t>(inTileXYSize)
    };
    if(usingNHWC) {
      const int inChannelsPaddedForNhwc = handle->getNHWCChannelsPadded(inChannels, true);
      const int outChannelsPaddedForNhwc = handle->getNHWCChannelsPadded(outChannels, true);
      const size_t inputElts = maxBatchSize * static_cast<size_t>(nhwcSpatialSize) *
        static_cast<size_t>(inChannelsPaddedForNhwc);
      const size_t outputElts = maxBatchSize * static_cast<size_t>(nhwcSpatialSize) *
        static_cast<size_t>(outChannelsPaddedForNhwc);
      if(nhwcConvPath == NhwcConvPath::DirectMatmul1x1) {
        needed.size1 = std::max(needed.size1, needsNCHWBoundaryConversion() ? inputElts : outputElts);
        if(needsNCHWBoundaryConversion())
          needed.size2 = std::max(needed.size2, outputElts);
      } else if(nhwcConvPath == NhwcConvPath::Winograd) {
      const int mPadded = vk_helper::roundUpToMultipleInt(
        maxBatchSize * numTilesY * numTilesX,
        handle->tuneParams.hgemmCooperativeMatrixNHWC.MWG
      );
      const int kPadded = inChannelsPaddedForNhwc;
      const int nPadded = outChannelsPaddedForNhwc;
      const size_t coefficientCount = static_cast<size_t>(inTileXYSize);
      const size_t transformedInputElts = coefficientCount * static_cast<size_t>(mPadded) * static_cast<size_t>(kPadded);
      const size_t transformedOutputElts = coefficientCount * static_cast<size_t>(mPadded) * static_cast<size_t>(nPadded);
      if(needsNCHWBoundaryConversion()) {
        needed.size1 = std::max(needed.size1, std::max(inputElts, transformedOutputElts));
        needed.size2 = std::max(needed.size2, std::max(transformedInputElts, outputElts));
      } else {
        needed.size1 = std::max(needed.size1, transformedInputElts);
        needed.size2 = std::max(needed.size2, transformedOutputElts);
      }
      }
    }
    return needed;
  }

  bool nhwcConvNeedsTemporaryOutput(VulkanBuffer* input, VulkanBuffer* output) const {
    return input == output &&
      nhwcConvPath == NhwcConvPath::DirectMatmul1x1;
  }

  void copyNhwcConvOutput(
    VkCommandBuffer& cb,
    VulkanBuffer* source,
    VulkanBuffer* destination,
    int batchSize
  ) {
    const int outputChannelsPadded = handle->getNHWCChannelsPadded(outChannels);
    const VkDeviceSize copySize = static_cast<VkDeviceSize>(batchSize) *
      static_cast<VkDeviceSize>(nhwcSpatialSize) *
      static_cast<VkDeviceSize>(outputChannelsPadded) * sizeof(half_t);
    vk_helper::barrierCommandBufferForBuffer(
      cb, source,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
    );
    vk_helper::barrierCommandBufferForBuffer(
      cb, destination,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT
    );
    vk_helper::recordBufferCopy(cb, source, destination, 0, 0, copySize);
    vk_helper::barrierCommandBufferForBuffer(
      cb, source,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    );
    vk_helper::barrierCommandBufferForBuffer(
      cb, destination,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    );
  }

  void doNHWCDirectConv1x1(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    assert(nhwcConvPath == NhwcConvPath::DirectMatmul1x1);
    VkResult res = VK_ERROR_UNKNOWN;
    const Pipeline& pipeline = handle->pipelines->hgemmCooperativeMatrixNHWC;
    if(hgemmCooperativeMatrixNHWCDS == VK_NULL_HANDLE) {
      hgemmCooperativeMatrixNHWCDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, pipeline.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("Allocate direct NHWC 1x1 descriptor set for ConvLayer: " + name, res);
    }
    vkcompute::doHgemmCooperativeMatrixNHWC(
      handle, handle->vulkanDevice, handle->tuneParams, &pipeline, cb, hgemmCooperativeMatrixNHWCDS,
      input, filterBuf, output, batchSize, nhwcSpatialSize, nhwcNSize, nhwcKSize,
      handle->getNHWCChannelsPadded(inChannels, usingNHWC), handle->getNHWCChannelsPadded(outChannels, usingNHWC),
      handle->tuneParams.hgemmCooperativeMatrixNHWC, &res
    );
    CHECK_VK_MSG("Execute direct NHWC 1x1 ConvLayer: " + name, res);
  }

  void doConv1x1AsMatmulFp32(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output
  ) {
    VkResult res;
    Pipeline targetPipeline = usingHgemmCooperativeMatrixNCHW
      ? handle->pipelines->hgemmCooperativeMatrixNCHW
      : handle->pipelines->xgemmStridedBatchedFp32;

    if ( descriptorSet == VK_NULL_HANDLE ) {
      descriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        targetPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for ConvLayer: " + name, res);
    }

    if(usingHgemmCooperativeMatrixNCHW) {
      vkcompute::doHgemmCooperativeMatrixNCHW(
        handle,
        handle->vulkanDevice,
        handle->tuneParams,
        &targetPipeline,
        cb,
        descriptorSet,
        input,
        filterBuf,
        output,
        batchSize,
        paddedNNXYLen,
        outChannels,
        inChannels,
        &res
      );
      CHECK_VK_MSG("Execute hgemmCooperativeMatrixNCHW for ConvLayer: " + name, res);
      vk_helper::barrierCommandBufferForBuffer(cb, output);
      return;
    }

    int filterStride = 0;
    int inputStride = paddedNNXYLen * inChannels;
    int outputStride = paddedNNXYLen * outChannels;

    vkcompute::xgemmStridedBatchedNN(
      handle,
      handle->vulkanDevice,
      handle->tuneParams,
      &targetPipeline,
      cb,
      descriptorSet,
      paddedNNXYLen, outChannels, inChannels,
      inputStride, filterStride, outputStride,
      input, filterBuf, output,
      static_cast<uint32_t>(batchSize), &res
    );
    vk_helper::barrierCommandBufferForBuffer(cb, output);
    // vk_helper::barrierCommandBuffer(cb);
  }

  void doNHWCWinogradConvolution(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* convWorkspace1,
    VulkanBuffer* convWorkspace2,
    VulkanBuffer* output,
    VulkanBuffer* bnScale,
    VulkanBuffer* bnBias,
    VulkanBuffer* mask,
    int activation,
    bool nchwBoundaryLayout = false
  ) {
    assert(nhwcConvPath == NhwcConvPath::Winograd);
    assert(convWorkspace1 != nullptr && convWorkspace2 != nullptr);
    const bool useBnAct = bnScale != nullptr;
    assert(useBnAct == (bnBias != nullptr && mask != nullptr));

    const auto* pipelines = handle->pipelines;
    const Pipeline& outputPipeline = convXSize == 3
      ? pipelines->winogradOutputTransform3x3
      : pipelines->winogradOutputTransform5x5;
    Pipeline inputPipeline;
    if(!useBnAct) {
      inputPipeline = convXSize == 3
        ? pipelines->winogradInputTransform3x3
        : pipelines->winogradInputTransform5x5;
    } else {
      switch(activation) {
        case ACTIVATION_IDENTITY:
          inputPipeline = convXSize == 3
            ? pipelines->winogradInputTransform3x3_bnact_identity
            : pipelines->winogradInputTransform5x5_bnact_identity;
          break;
        case ACTIVATION_RELU:
          inputPipeline = convXSize == 3
            ? pipelines->winogradInputTransform3x3_bnact_relu
            : pipelines->winogradInputTransform5x5_bnact_relu;
          break;
        case ACTIVATION_MISH:
          inputPipeline = convXSize == 3
            ? pipelines->winogradInputTransform3x3_bnact_mish
            : pipelines->winogradInputTransform5x5_bnact_mish;
          break;
        case ACTIVATION_MISH_SCALE8:
          inputPipeline = convXSize == 3
            ? pipelines->winogradInputTransform3x3_bnact_mish_scale8
            : pipelines->winogradInputTransform5x5_bnact_mish_scale8;
          break;
        case ACTIVATION_SILU:
          inputPipeline = convXSize == 3
            ? pipelines->winogradInputTransform3x3_bnact_silu
            : pipelines->winogradInputTransform5x5_bnact_silu;
          break;
        default:
          throw StringError("Unsupported activation for fused NHWC Winograd convolution in layer " + name);
      }
    }

    VkResult res = VK_ERROR_UNKNOWN;
    if(winogradInputTransformDS == VK_NULL_HANDLE) {
      winogradInputTransformDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, inputPipeline.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("Allocate NHWC Winograd input descriptor set for ConvLayer: " + name, res);
    }
    if(winogradOutputTransformDS == VK_NULL_HANDLE) {
      winogradOutputTransformDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, outputPipeline.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("Allocate NHWC Winograd output descriptor set for ConvLayer: " + name, res);
    }
    const Pipeline& gemmPipeline = pipelines->hgemmCooperativeMatrixNHWC;
    if(hgemmCooperativeMatrixNHWCDS == VK_NULL_HANDLE) {
      hgemmCooperativeMatrixNHWCDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, gemmPipeline.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("Allocate NHWC Winograd GEMM descriptor set for ConvLayer: " + name, res);
    }

    const auto& gemmParams = handle->tuneParams.hgemmCooperativeMatrixNHWC;
    const uint32_t mPadding = static_cast<uint32_t>(gemmParams.MWG);
    const uint32_t kPadding = static_cast<uint32_t>(gemmParams.KWG);
    VulkanBuffer* transformedInput = nchwBoundaryLayout ? convWorkspace2 : convWorkspace1;
    VulkanBuffer* transformedOutput = nchwBoundaryLayout ? convWorkspace1 : convWorkspace2;
    VulkanBuffer* spatialOutput = nchwBoundaryLayout ? convWorkspace2 : output;
    if(useBnAct) {
      vkcompute::convInputToWinogradDomainBnActMask(
        handle, handle->vulkanDevice, handle->tuneParams, &inputPipeline, cb,
        winogradInputTransformDS, input, transformedInput, bnScale, bnBias, mask,
        nnYLen, nnXLen, paddedNNXYLen,
        batchSize, numTilesY, numTilesX, mPadding,
        inChannels, kPadding, convYSize, &res, true
      );
    } else {
      vkcompute::convInputsToWinogradDomain(
        handle, handle->vulkanDevice, handle->tuneParams, &inputPipeline, cb,
        winogradInputTransformDS, input, transformedInput,
        nnYLen, nnXLen, paddedNNXYLen,
        batchSize, numTilesY, numTilesX, mPadding,
        inChannels, kPadding, convYSize, &res, true
      );
    }
    CHECK_VK_MSG("Execute NHWC Winograd input transform for ConvLayer: " + name, res);
    if(input == output)
      vk_helper::barrierCommandBufferForBuffer(cb, input);

    const int M = vk_helper::roundUpToMultipleInt(
      batchSize * numTilesY * numTilesX, gemmParams.MWG
    );
    const int K = handle->getNHWCChannelsPadded(inChannels, usingNHWC);
    const int N = handle->getNHWCChannelsPadded(outChannels, usingNHWC);
    const int aBatchStride = M * K;
    const int bBatchStride = K * N;
    const int cBatchStride = M * N;
    vkcompute::doHgemmCooperativeMatrixNHWC(
      handle, handle->vulkanDevice, handle->tuneParams, &gemmPipeline, cb,
      hgemmCooperativeMatrixNHWCDS, transformedInput, filterBuf, transformedOutput,
      inTileXYSize, M, N, K, K, N, gemmParams, &res,
      aBatchStride, bBatchStride, cBatchStride
    );
    CHECK_VK_MSG("Execute NHWC Winograd GEMM for ConvLayer: " + name, res);

    vkcompute::winogradOutputToSpatialDomain(
      handle, handle->vulkanDevice, &outputPipeline, cb, winogradOutputTransformDS,
      transformedOutput, spatialOutput, nnYLen, nnXLen, paddedNNXYLen,
      batchSize, numTilesY, numTilesX, mPadding,
      outChannels, static_cast<uint32_t>(gemmParams.NWG), &res, true
    );
    CHECK_VK_MSG("Execute NHWC Winograd output transform for ConvLayer: " + name, res);
    if(nchwBoundaryLayout)
      convertNHWCOutputToNCHW(cb, batchSize, spatialOutput, output);
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
    int activation,
    bool nchwBoundaryLayout = false
  ) {
    if(nhwcConvPath == NhwcConvPath::Winograd) {
      doNHWCWinogradConvolution(
        cb, batchSize, input, convWorkspace1, convWorkspace2, output,
        bnScale, bnBias, mask, activation, nchwBoundaryLayout
      );
      return;
    }
    // Implement convolution logic here if needed
    VkResult res = VK_ERROR_UNKNOWN;
    const auto *pipelines = this->handle->pipelines;
    Pipeline winogradInputTransformBnActMaskPipeline;
    Pipeline xgemmBatchedPipeline = pipelines->xgemmBatchedFp32;
    VkDescriptorSet& gemmDescriptorSet = xgemmBatchedDS;
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
      case ACTIVATION_SILU:
        winogradInputTransformBnActMaskPipeline = (convXSize == 3 && convYSize == 3) ? pipelines->winogradInputTransform3x3_bnact_silu
                        : (convXSize == 5 && convYSize == 5) ? pipelines->winogradInputTransform5x5_bnact_silu
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

    if ( gemmDescriptorSet == VK_NULL_HANDLE ) {
      gemmDescriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        xgemmBatchedPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for ConvLayer: " + name, res);
    }

    // first winograd input transform with fused bn+act and require barrier 
    {
      vkcompute::convInputToWinogradDomainBnActMask(
        handle,
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
        nnYLen, nnXLen, paddedNNXYLen,
        batchSize, numTilesY, numTilesX, handle->getXGemmMPaddingMult(),
        inChannels, handle->getXGemmKPaddingMult(),
        convYSize,
        &res
      );
    }

    // vk_helper::barrierCommandBuffer(cb);
    vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace1);
    // Then xgemm and winograd output transform same as before
    {
      // {
        // uint32_t dbg_numTilesTotal = vk_helper::roundUpToMultipleInt(numTilesTotal, handle->getXGemmMPaddingMult());
        // uint32_t dbg_outCh = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
        // uint32_t dbg_inCh = vk_helper::roundUpToMultipleInt(inChannels, handle->getXgemmKPaddingMult());
        // std::printf("[xGEMM] Before dispatch (BNAct path) numTilesTotal=%u outChPadded=%u inChPadded=%u inTilesXYSize=%d descriptorSet=%p\n",
          // dbg_numTilesTotal, dbg_outCh, dbg_inCh, this->inTilesXYSize, (void*)xgemmBatchedDS);
      // }
      const int M = vk_helper::roundUpToMultipleInt(batchSize * numTilesX * numTilesY, handle->getXGemmMPaddingMult());
      const int N = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
      const int K = vk_helper::roundUpToMultipleInt(inChannels, handle->getXGemmKPaddingMult());
      vkcompute::xgemmBatched(
        handle, handle->vulkanDevice, handle->tuneParams, &xgemmBatchedPipeline, cb, gemmDescriptorSet,
        M, N, K, convWorkspace1, filterBuf, convWorkspace2, inTileXYSize, &res
      );
    }

    // Barrier before output transform
    // vk_helper::barrierCommandBuffer(cb);
    vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace2);
    // Output transform is the same as before, since the fused bn+act is only on input transform side and doesn't change the data layout for xgemm
    {
      vkcompute::winogradOutputToSpatialDomain(
        handle,
        handle->vulkanDevice,
        &winogradOutputTransformPipeline,
        cb,
        winogradOutputTransformDS,
        convWorkspace2,
        output,
        nnYLen, nnXLen, paddedNNXYLen,
        batchSize, numTilesY, numTilesX, handle->getXGemmMPaddingMult(),
        outChannels, handle->getXGemmNPaddingMult(),
        &res
      );
    }
    vk_helper::barrierCommandBufferForBuffer(cb, output);
  }

  void doWinogradConvolution(
    VkCommandBuffer& cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* convWorkspace1,
    VulkanBuffer* convWorkspace2,
    VulkanBuffer* output,
    bool nchwBoundaryLayout = false
  ) {
    if(nhwcConvPath == NhwcConvPath::Winograd) {
      doNHWCWinogradConvolution(
        cb, batchSize, input, convWorkspace1, convWorkspace2, output,
        nullptr, nullptr, nullptr, -1, nchwBoundaryLayout
      );
      return;
    }
    VkResult res = VK_ERROR_UNKNOWN;
    const VulkanDevice* device = handle->vulkanDevice;
    const vk_shader::ComputePipelines *pipelines = handle->pipelines;

    Pipeline inputTransformPipeline = (convXSize == 3 && convYSize == 3) ? pipelines->winogradInputTransform3x3
                                         : (convXSize == 5 && convYSize == 5) ? pipelines->winogradInputTransform5x5
                                         : throw StringError("Winograd convolution only supported for 3x3 and 5x5 kernels in layer " + name);

    Pipeline outputTransformPipeline = (convXSize == 3 && convYSize == 3) ? pipelines->winogradOutputTransform3x3
                                         : (convXSize == 5 && convYSize == 5) ? pipelines->winogradOutputTransform5x5
                                         : throw StringError("Winograd convolution only supported for 3x3 and 5x5 kernels in layer " + name);

    Pipeline xgemmPipeline = pipelines->xgemmBatchedFp32;
    VkDescriptorSet& gemmDescriptorSet = xgemmBatchedDS;

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


    if(  gemmDescriptorSet == VK_NULL_HANDLE ) {
      gemmDescriptorSet = vk_helper::allocateDescriptorSet(
        device,
        xgemmPipeline.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("Allocate descriptor set for Winograd GEMM in ConvLayer: " + name, res);
    }
    // first winograd input transform and require barrier 
    {
      vkcompute::convInputsToWinogradDomain(
        handle,
        device,
        handle->tuneParams,
        &inputTransformPipeline,
        cb,
        winogradInputTransformDS,
        input,
        convWorkspace1,
        nnYLen, nnXLen, paddedNNXYLen,
        batchSize, numTilesY, numTilesX, handle->getXGemmMPaddingMult(),
        inChannels, handle->getXGemmKPaddingMult(),
        convYSize,
        &res
      );
    }
    // vk_helper::barrierCommandBuffer(cb);
    vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace1);

    uint32_t numTilesTotal = vk_helper::roundUpToMultipleInt(batchSize * numTilesY * numTilesX, handle->getXGemmMPaddingMult());
    uint32_t inChannelsPadded = vk_helper::roundUpToMultipleInt(inChannels, handle->getXGemmKPaddingMult());
    uint32_t outChannelsPadded = vk_helper::roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
    {
      // // std::printf("[xGEMM] Before dispatch numTilesTotal=%u outChPadded=%u inChPadded=%u inTilesXYSize=%d descriptorSet=%p\n",
      //   numTilesTotal, outChannelsPadded, inChannelsPadded, this->inTilesXYSize, (void*)xgemmBatchedDS);
      vkcompute::xgemmBatched(
        handle, device, handle->tuneParams, &xgemmPipeline, cb, gemmDescriptorSet,
        numTilesTotal, outChannelsPadded, inChannelsPadded,
        convWorkspace1, filterBuf, convWorkspace2, inTileXYSize, &res
      );
    }
    vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace2);
    // vk_helper::barrierCommandBuffer(cb);

    {
      vkcompute::winogradOutputToSpatialDomain(
        handle,
        device,
        &outputTransformPipeline,
        cb,
        winogradOutputTransformDS,
        convWorkspace2,
        output,
        nnYLen, nnXLen, paddedNNXYLen,
        batchSize, numTilesY, numTilesX, handle->getXGemmMPaddingMult(),
        outChannels, handle->getXGemmNPaddingMult(),
        &res
      );
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
    if(handle->pipelines->useNHWC && !usingNHWC)
      throw StringError("Vulkan ConvLayer " + name + " cannot fall back from the NHWC path to NCHW");
    if(usingNHWC) {
      const bool nchwBoundaryLayout = needsNCHWBoundaryConversion();
      VulkanBuffer* nhwcInput = input;
      if(nchwBoundaryLayout) {
        if(convWorkspace1 == nullptr || convWorkspace2 == nullptr)
          throw StringError("Vulkan ConvLayer " + name + " requires workspaces for NCHW/NHWC conversion");
        convertNCHWInputToNHWC(cb, batchSize, input, convWorkspace1);
        nhwcInput = convWorkspace1;
      }
      if(nhwcConvPath == NhwcConvPath::DirectMatmul1x1) {
        VulkanBuffer* convOutput = output;
        if(nchwBoundaryLayout || nhwcConvNeedsTemporaryOutput(input, output))
          convOutput = nchwBoundaryLayout ? convWorkspace2 : convWorkspace1;
        if(convOutput == nullptr)
          throw StringError("Vulkan ConvLayer " + name + " requires an output workspace for in-place NHWC convolution");
        doNHWCDirectConv1x1(cb, batchSize, nhwcInput, convOutput);
        if(nchwBoundaryLayout)
          convertNHWCOutputToNCHW(cb, batchSize, convOutput, output);
        else if(nhwcConvNeedsTemporaryOutput(input, output))
          copyNhwcConvOutput(cb, convOutput, output, batchSize);
      } else if(nhwcConvPath == NhwcConvPath::Winograd) {
        doWinogradConvolution(cb, batchSize, nhwcInput, convWorkspace1, convWorkspace2, output, nchwBoundaryLayout);
      } else {
        throw StringError("Vulkan ConvLayer " + name + " has no supported NHWC convolution path");
      }
      return;
    }
    if ( convXSize == 1 && convYSize == 1 ) {
      doConv1x1AsMatmulFp32(cb, batchSize,  input, output);
    } else if ( (convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5) ) {
      assert(convWorkspace1 != nullptr); 
      assert(convWorkspace2 != nullptr);
      doWinogradConvolution(cb, batchSize, input, convWorkspace1, convWorkspace2, output);
    } else {
      // katago only support 3x3 or 5x5. no required this block.
      // doConv2DTiledFp32(cb, batchSize,  input, output);
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
    if(handle->pipelines->useNHWC && !usingNHWC)
      throw StringError("Vulkan ConvLayer " + name + " cannot fall back from the NHWC path to NCHW");
    if(usingNHWC) {
      const bool nchwBoundaryLayout = needsNCHWBoundaryConversion();
      VulkanBuffer* nhwcInput = input;
      if(nchwBoundaryLayout) {
        if(convWorkspace1 == nullptr || convWorkspace2 == nullptr)
          throw StringError("Vulkan ConvLayer " + name + " requires workspaces for NCHW/NHWC conversion");
        convertNCHWInputToNHWC(cb, batchSize, input, convWorkspace1);
        nhwcInput = convWorkspace1;
      }
      if(nhwcConvPath == NhwcConvPath::Winograd) {
        doWinogradConvolutionBnActMask(
          cb, batchSize, nhwcInput, convWorkspace1, convWorkspace2, output,
          bnLayer->mergedScaleBuf, bnLayer->mergedBiasBuf, mask, bnLayer->activation,
          nchwBoundaryLayout
        );
        return;
      }
      throw StringError("Vulkan ConvLayer " + name + " has no fused NHWC Winograd path");
    }
    doWinogradConvolutionBnActMask(cb, batchSize, input, convWorkspace1, convWorkspace2, output, bnLayer->mergedScaleBuf, bnLayer->mergedBiasBuf, mask, bnLayer->activation);
  }

  /**
   * @brief Launch the recorded command buffer, only for debug now.
   * @param batchSize
   * @param input
   * @param output
   * @return VkCommandBuffer
   */
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
    if ( biasBuf != nullptr ) {
      vk_helper::releaseVulkanBuffer(handle->vulkanDevice, biasBuf);
      // delete biasBuf;
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
      // TODO: add FP16 buffer allocation
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
    const vk_shader::ComputePipelines* pipelines = this->handle->pipelines;
    Pipeline targetPipeline;

    switch ( activation ) {
      case ACTIVATION_IDENTITY:
        targetPipeline = pipelines->addChannelBiasNCIdentity;
        break;
      case ACTIVATION_RELU:
        targetPipeline = pipelines->addChannelBiasNCRelu;
        break;
      case ACTIVATION_MISH:
        targetPipeline = pipelines->addChannelBiasNCMish;
        break;
      case ACTIVATION_MISH_SCALE8:
        targetPipeline = pipelines->addChannelBiasNCMishScale8;
        break;
      case ACTIVATION_SILU:
        targetPipeline = pipelines->addChannelBiasNCSilu;
        break;
      default:
        Global::fatalError("Unsupported activation in MatBiasLayer: " + name);
    }

    if ( descriptorSet == VK_NULL_HANDLE ) {
      descriptorSet = vk_helper::allocateDescriptorSet(handle->vulkanDevice, targetPipeline.descriptorSetLayout, &res);
      CHECK_VK_MSG("Allocate descriptor set for MatBiasLayer: " + name, res);
    }
    vkcompute::doMatBiasNC(
      handle, &targetPipeline, cb, descriptorSet, input, biasBuf,
      batchSize, numChannels
    );
    // CHECK_VK_MSG("End command buffer for MatBiasLayer: " + name, res);
  }

  /**
   * @brief Launch the recorded command buffer, only for debug now.
   * @param batchSize
   * @param input
   */

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
    int nnYLen_,
    bool useFP16
  ):
    handle(handle_),
    conv(handle_, convDesc, nnXLen_, nnYLen_, useFP16),
    bn(handle_, bnDesc, actDesc, useFP16),
    inChannels(convDesc->inChannels),
    outChannels(convDesc->outChannels)
  {
    assert( bn.numChannels == conv.inChannels );
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, int maxBatchSize) {
    return ConvWorkspaceEltsNeeded::getMax(
      conv.requiredConvWorkspaceElts(handle, maxBatchSize),
      bn.requiredConvWorkspaceElts(handle, maxBatchSize)
    );
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
    if (conv.isBNActFusedPossible()) {
      conv.forwardBnActConv(cb, &bn, batchSize, input, output, convWorkspace, convWorkspace2, mask);
    } else {
      bn.forward(
        cb, batchSize, input, mask, inputScratchOrInput,
        convWorkspace, convWorkspace2,
        conv.nnXLen * conv.nnYLen,
        conv.nnXLen * conv.nnYLen
      );
      vk_helper::barrierCommandBufferForBuffer(cb, inputScratchOrInput);
      conv.forward(cb, batchSize, inputScratchOrInput, output, convWorkspace, convWorkspace2);
    }
    vk_helper::barrierCommandBufferForBuffer(cb, output);
  }


  /**
   * @brief Launch the recorded command buffers, only for debug now.
   * @param batchSize
   * @param input
   * @param inputScratchOrInput
   * @param output
   */

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;
};

struct TransformerMatMulLayer {
  // TODO: Require to implement class definition
  ComputeHandleInternal* handle;
  const std::string name;
  const int inChannels;
  const int outChannels;
  const int paddedNNXYLen;
  bool usingHgemmCooperativeMatrixNHWC;
  VkDescriptorSet descriptorSet;
  VkDescriptorSet nchwToNhwcDescriptorSet;
  VkDescriptorSet nhwcToNchwDescriptorSet;
  VulkanBuffer* filter;

  TransformerMatMulLayer(
    ComputeHandleInternal *handle,
    const MatMulLayerDesc* desc
  ):
    handle(handle),
    name(desc->name),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels),
    paddedNNXYLen(handle->paddedNNXYLen),
    usingHgemmCooperativeMatrixNHWC(false),
    descriptorSet(VK_NULL_HANDLE),
    nchwToNhwcDescriptorSet(VK_NULL_HANDLE),
    nhwcToNchwDescriptorSet(VK_NULL_HANDLE),
    filter(nullptr)
  {
    testAssert(desc->weights.size() == static_cast<size_t>(inChannels * outChannels));
    std::vector<float> weights = desc->weights;
    const auto& hgemmParams = handle->tuneParams.hgemmCooperativeMatrixNHWC;
    usingHgemmCooperativeMatrixNHWC =
      handle->usingFP16Storage &&
      handle->tuneParams.vulkan.canUseCooperativeMatrix &&
      handle->tuneParams.vulkan.shouldUseCooperativeMatrix &&
      handle->tuneParams.vulkan.shouldUseFP16Storage &&
      handle->tuneParams.vulkan.shouldUseFP16Compute &&
      hgemmParams.isValid() &&
      handle->paddedNNXYLen % hgemmParams.MWG == 0 &&
      inChannels % 4 == 0 &&
      outChannels % 4 == 0 &&
      inChannels % hgemmParams.KWG == 0 &&
      outChannels % hgemmParams.NWG == 0;
    if(handle->pipelines->useNHWC && !usingHgemmCooperativeMatrixNHWC)
      throw StringError("Vulkan TransformerMatMulLayer " + name + " cannot fall back from the NHWC path to NCHW");
    bool useFP16 = handle->usingFP16Storage || usingHgemmCooperativeMatrixNHWC;
    VkResult res = VK_ERROR_UNKNOWN;
    filter = vk_helper::createReadOnlyBuffer(handle->vulkanDevice, weights, useFP16, &res);
    CHECK_VK_MSG("[TransformerMatMulLayer::TransformerMatmulLayer()] create filter vulkan buffer", res);

    const Pipeline& descriptorPipeline = usingHgemmCooperativeMatrixNHWC
      ? handle->pipelines->hgemmCooperativeMatrixNHWC
      : handle->pipelines->xgemmStridedBatchedFp32;
    descriptorSet = vk_helper::allocateDescriptorSet(handle->vulkanDevice, descriptorPipeline.descriptorSetLayout, &res);
    CHECK_VK_MSG("[TransformerMatMulLayer::TransformerMatMulLayer()] allocate descriptorSet", res);
    if(usingHgemmCooperativeMatrixNHWC) {
      nchwToNhwcDescriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, handle->pipelines->nchwToNhwc.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("[TransformerMatMulLayer::TransformerMatMulLayer()] allocate nchwToNhwcDescriptorSet", res);
      nhwcToNchwDescriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, handle->pipelines->nhwcToNchw.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("[TransformerMatMulLayer::TransformerMatMulLayer()] allocate nhwcToNchwDescriptorSet", res);
    }
  }

  ~TransformerMatMulLayer() {
    if ( filter ) {
      vk_helper::releaseVulkanBuffer(handle->vulkanDevice, filter);
    }
  }

  void forward(
    VkCommandBuffer cb,
    ScratchBuffers* scratch,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* mask,
    VulkanBuffer* convWorkspace
  ) {
    VkResult res;
    if (!usingHgemmCooperativeMatrixNHWC) {
      int filterStride = 0;
      int inputStride = paddedNNXYLen * inChannels;
      int outputStride = paddedNNXYLen * outChannels;
      Pipeline pipeline = handle->pipelines->xgemmStridedBatchedFp32;

    vkcompute::xgemmStridedBatchedNN(
      handle,
      handle->vulkanDevice,
        handle->tuneParams,
        &pipeline,
        cb,
        descriptorSet,
        paddedNNXYLen, outChannels, inChannels,
        inputStride, filterStride, outputStride,
        input, filter, output,
        static_cast<uint32_t>(batchSize), &res
      );
    } else {
      Pipeline pipeline = handle->pipelines->hgemmCooperativeMatrixNHWC;
    vkcompute::doHgemmCooperativeMatrixNHWC(
      handle,
      handle->vulkanDevice,
        handle->tuneParams,
        &pipeline,
        cb,
        descriptorSet,
        input,
        filter,
        output,
        batchSize,
        paddedNNXYLen,
        outChannels,
        inChannels,
        handle->getNHWCChannelsPadded(inChannels),
        handle->getNHWCChannelsPadded(outChannels),
        handle->tuneParams.hgemmCooperativeMatrixNHWC,
        &res
      );
    }
    CHECK_VK_MSG("Execute matmul for TransformerMatMulLayer: " + name, res);
  }


  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    // No separate pad buffer needed - input is pre-padded to paddedNNXYLen
    (void)handle;
    (void)maxBatchSize;
    return ConvWorkspaceEltsNeeded();
  }

  TransformerMatMulLayer() = delete;
  TransformerMatMulLayer(const TransformerMatMulLayer&) = delete;
  TransformerMatMulLayer& operator=(const TransformerMatMulLayer&) = delete;
};


struct TransformerApplyRoPELayer {

  ComputeHandleInternal *handle;
  const bool useNHWC;
  const int channels;
  const int channelsPadded;
  Pipeline pipeline;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet nchwToNhwcDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet nhwcToNchwDescriptorSet = VK_NULL_HANDLE;

  TransformerApplyRoPELayer(
    ComputeHandleInternal *handle,
    int channels
  ):
    handle(handle),
    useNHWC(handle->pipelines->useNHWC),
    channels(channels),
    channelsPadded(handle->getNHWCChannelsPadded(channels)),
    pipeline(useNHWC ? handle->pipelines->transformerApplyRoPENHWC : handle->pipelines->transformerApplyRoPE)
  {
    VkResult res = VK_ERROR_UNKNOWN;
    descriptorSet = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice,
      pipeline.descriptorSetLayout,
      &res
    );
    CHECK_VK_MSG("[TransformerApplyRoPE] allocate ropeDescriptorSet", res);
    if(useNHWC) {
      nchwToNhwcDescriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        handle->pipelines->nchwToNhwc.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("[TransformerApplyRoPE] allocate NCHW to NHWC descriptorSet", res);
      nhwcToNchwDescriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        handle->pipelines->nhwcToNchw.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("[TransformerApplyRoPE] allocate NHWC to NCHW descriptorSet", res);
    }
  }

  ~TransformerApplyRoPELayer() = default;

  void forward(
    VkCommandBuffer cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* cosTable,
    VulkanBuffer* sinTable,
    const int numHeads,
    const int numKVHeads,
    const int headDim,
    const int seqLen,
    const int numPairs,
    const int learnableInt,
    const int regionOffset,
    const int batchStride,
    VulkanBuffer* nhwcScratch = nullptr
  ) {
    assert(cb != VK_NULL_HANDLE);
    VkResult res = VK_ERROR_UNKNOWN;
    vkcompute::transformerApplyRoPE(
      handle,
      handle->vulkanDevice,
      &pipeline,
      cb,
      descriptorSet,
      useNHWC ? &handle->pipelines->nchwToNhwc : nullptr,
      nchwToNhwcDescriptorSet,
      useNHWC ? &handle->pipelines->nhwcToNchw : nullptr,
      nhwcToNchwDescriptorSet,
      input,
      nhwcScratch,
      cosTable,
      sinTable,
      batchSize,
      numHeads,
      numKVHeads,
      headDim,
      seqLen,
      numPairs,
      learnableInt,
      regionOffset,
      batchStride,
      channels,
      channelsPadded,
      handle->paddedNNXYLen,
      handle->nnXLen * handle->nnYLen,
      useNHWC,
      &res
    );
    CHECK_VK_MSG("Execute TransformerApplyRoPE", res);
  }

};


struct TransformerAttentionLayer {
  ComputeHandleInternal* handle;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet scalarDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet nchwToNhwcDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet nhwcToNchwDescriptorSet = VK_NULL_HANDLE;
  Pipeline pipeline;
  Pipeline scalarPipeline;
  ScaleDotProductPushParam params;
  const int numHeads;
  const int numKVHeads;
  const int qHeadDim;
  const int vHeadDim;
  const bool useTiled;
  const bool useCooperative;

  explicit TransformerAttentionLayer(
    ComputeHandleInternal* handle,
    const int numHeads,
    const int numKVHeads
  ):
    handle(handle),
    numHeads(numHeads),
    numKVHeads(numKVHeads),
    qHeadDim(handle->qHeadDim),
    vHeadDim(handle->vHeadDim),
    useTiled(handle->tuneParams.transformer.USE_TILED_ATTN != 0),
    useCooperative(handle->tuneParams.transformer.USE_COOPERATIVE_ATTN != 0)
  {

    if(handle->pipelines->useNHWC && !useCooperative)
      throw StringError("Vulkan TransformerAttentionLayer cannot fall back from the NHWC path to a NCHW attention kernel");

    const vk_shader::ComputePipelines* pipelines = this->handle->pipelines;
    params.seqLen = handle->paddedNNXYLen;
    params.numHeads = numHeads;
    params.numKVHeads = numKVHeads;
    params.scale = 1.0f / sqrtf(static_cast<float>(handle->qHeadDim));
    if(useCooperative) {
      pipeline = pipelines->transformerScaleDotProductCooperativeNHWC;
      scalarPipeline = pipelines->transformerScaleDotProduct;
    } else if(useTiled) {
      pipeline = pipelines->transformerScaleDotProduct;
    } else {
      pipeline = pipelines->transformerScaleDotProductNaive;
    }
    VkResult res = VK_SUCCESS;
    descriptorSet = vk_helper::allocateDescriptorSet(handle->vulkanDevice, pipeline.descriptorSetLayout, &res);
    CHECK_VK_MSG("[TransformerAttentionLayer::TransformerAttentionLayer] create descriptor set", res);
    if(useCooperative) {
      scalarDescriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, scalarPipeline.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("[TransformerAttentionLayer::TransformerAttentionLayer] create scalar fallback descriptor set", res);
      nchwToNhwcDescriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, pipelines->nchwToNhwc.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("[TransformerAttentionLayer::TransformerAttentionLayer] create NCHW-to-NHWC descriptor set", res);
      nhwcToNchwDescriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, pipelines->nhwcToNchw.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("[TransformerAttentionLayer::TransformerAttentionLayer] create NHWC-to-NCHW descriptor set", res);
    }
  }

  ~TransformerAttentionLayer()=default;

  void forward(
    VkCommandBuffer cb,
    int batchSize,
    VulkanBuffer* packedQKV,
    VulkanBuffer* output,
    VulkanBuffer* nhwcQKV,
    VulkanBuffer* nhwcOutput,
    VulkanBuffer* mask,
    VulkanBuffer* ropeCosTable,
    VulkanBuffer* ropeSinTable,
    const int qOffset,
    const int kOffset,
    const int vOffset,
    const int batchStride,
    const bool useRope,
    const bool learnableRope,
    const int ropeNumPairs
  ) {
    if(useCooperative) {
      VkResult res = VK_SUCCESS;
      const int seqLen = handle->paddedNNXYLen;
      const int qkvChannels = batchStride / seqLen;
      vkcompute::transformerScaleDotProductCooperative(
        handle,
        handle->vulkanDevice,
        &pipeline,
        cb,
        descriptorSet,
        &handle->pipelines->nchwToNhwc,
        nchwToNhwcDescriptorSet,
        &handle->pipelines->nhwcToNchw,
        nhwcToNchwDescriptorSet,
        packedQKV,
        output,
        nhwcQKV,
        nhwcOutput,
        mask,
        ropeCosTable,
        ropeSinTable,
        batchSize,
        numHeads,
        numKVHeads,
        qHeadDim,
        vHeadDim,
        seqLen,
        handle->nnXLen * handle->nnYLen,
        seqLen,
        qkvChannels,
        numHeads * vHeadDim,
        handle->getNHWCChannelsPadded(qkvChannels),
        handle->getNHWCChannelsPadded(numHeads * vHeadDim),
        numHeads * qHeadDim,
        numKVHeads * qHeadDim,
        params.scale,
        useRope,
        learnableRope,
        ropeNumPairs,
        handle->tuneParams.transformer,
        &res
      );
      CHECK_VK_MSG("Execute cooperative Transformer attention in NHWC", res);
      return;
    }

    // The cooperative shader applies RoPE while loading Q/K. Its FP16
    // rounding is accepted only when the tuner has measured it within the
    // attention error tolerance.
    const bool useCooperativeDispatch = useCooperative;
    const Pipeline& activePipeline = useCooperativeDispatch ? pipeline : (useCooperative ? scalarPipeline : pipeline);
    const VkDescriptorSet activeDescriptorSet = useCooperativeDispatch ? descriptorSet : (useCooperative ? scalarDescriptorSet : descriptorSet);
    auto writeDescriptors = {
      vk_helper::writeDescriptorSetBuffer(activeDescriptorSet, 0, packedQKV),
      vk_helper::writeDescriptorSetBuffer(activeDescriptorSet, 1, packedQKV),
      vk_helper::writeDescriptorSetBuffer(activeDescriptorSet, 2, packedQKV),
      vk_helper::writeDescriptorSetBuffer(activeDescriptorSet, 3, output),
      vk_helper::writeDescriptorSetBuffer(activeDescriptorSet, 4, mask),
      vk_helper::writeDescriptorSetBuffer(activeDescriptorSet, 5, ropeCosTable),
      vk_helper::writeDescriptorSetBuffer(activeDescriptorSet, 6, ropeSinTable),
    };
    vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptors);

    params.qOffset = qOffset;
    params.kOffset = kOffset;
    params.vOffset = vOffset;
    params.qBatchStride = batchStride;
    params.kBatchStride = batchStride;
    params.vBatchStride = batchStride;
    params.useRope = useRope ? 1 : 0;
    params.learnableRope = learnableRope ? 1 : 0;
    params.ropeNumPairs = ropeNumPairs;
    params.ropeReserved = 0;

    if (useTiled) {
      auto tuneParams = handle->tuneParams.transformer;
      uint32_t qPerThread = tuneParams.Q_PER_THREAD;
      uint32_t totalQPerWG = useCooperativeDispatch
        ? static_cast<uint32_t>(tuneParams.ATTN_BLOCK_Q) * qPerThread
        : activePipeline.localSizeX * qPerThread;
      uint32_t numQGroups = (params.seqLen + totalQPerWG - 1) / totalQPerWG;
      uint32_t wgCount[3] = {
        numQGroups,
        (static_cast<uint32_t>(batchSize) * params.numHeads + activePipeline.localSizeY - 1) / activePipeline.localSizeY,
        (1 + activePipeline.localSizeZ - 1) / activePipeline.localSizeZ
      };
      vkcompute::dispatchPipeline(
        handle, handle->vulkanDevice, &activePipeline, cb, activeDescriptorSet,
        &params, sizeof(params), wgCount[0], wgCount[1], wgCount[2],
        "SCALE_DOT_PRODUCT_ATTENTION"
      );
    } else {
      uint32_t gs[2] = {
        static_cast<uint32_t>(vk_helper::powerOf2ify(params.seqLen)),
        static_cast<uint32_t>(batchSize) * params.numHeads
      };
      uint32_t wgCount[3] = {
        (gs[0] + activePipeline.localSizeX - 1) / activePipeline.localSizeX,
        (gs[1] + activePipeline.localSizeY - 1) / activePipeline.localSizeY,
        (1 + activePipeline.localSizeZ - 1) / activePipeline.localSizeZ
      };
      vkcompute::dispatchPipeline(
        handle, handle->vulkanDevice, &activePipeline, cb, activeDescriptorSet,
        &params, sizeof(params), wgCount[0], wgCount[1], wgCount[2],
        "SCALE_DOT_PRODUCT_ATTENTION_NAIVE"
      );
    }
  }

};

struct TransformerRMSNormLayer {
  ComputeHandleInternal* handle;
  const std::string name;
  const int numChannels;
  const float epsilon;
  const int paddedNNXYLen;
  const int channelsPadded;
  const bool useNHWC;
  const Pipeline* pipeline;
  VulkanBuffer* weightBuf;
  VulkanBuffer* zeroBetaBuf;
  VkDescriptorSet descriptorSet;
  VkDescriptorSet nchwToNhwcDescriptorSet;
  VkDescriptorSet nhwcToNchwDescriptorSet;


  TransformerRMSNormLayer(
    ComputeHandleInternal *handle,
    const TransformerRMSNormDesc* desc
  ) : 
    handle(handle),
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    paddedNNXYLen(handle->paddedNNXYLen),
    channelsPadded(handle->getNHWCChannelsPadded(desc->numChannels)),
    useNHWC(handle->pipelines->useNHWC),
    pipeline(useNHWC ? &handle->pipelines->transformerRmsNormNHWC : &handle->pipelines->transformerRmsNorm),
    descriptorSet(VK_NULL_HANDLE),
    nchwToNhwcDescriptorSet(VK_NULL_HANDLE),
    nhwcToNchwDescriptorSet(VK_NULL_HANDLE)
  {
    testAssert(desc->weight.size() == numChannels);
    vector<float> weight = desc->weight;
    bool useFP16 = false;
    VkResult res;
    weightBuf = vk_helper::createReadOnlyBuffer(handle->vulkanDevice, weight, useFP16, &res);
    CHECK_VK_MSG("[TransformerRMSNormLayer::TransformerRMSNormLayer()] create weight buf", res);
    vector<float> zeroBeta(numChannels, 0.0f);
    zeroBetaBuf = vk_helper::createReadOnlyBuffer(handle->vulkanDevice, zeroBeta, useFP16, &res);
    CHECK_VK_MSG("[TransformerRMSNormLayer::TransformerRMSNormLayer()] create zeroBeta buf", res);
    descriptorSet = vk_helper::allocateDescriptorSet(handle->vulkanDevice, pipeline->descriptorSetLayout, &res);
    CHECK_VK_MSG("[TransformerRMSNormLayer::TransformerRMSNormLayer()] allocate descriptor set.", res);
    if(useNHWC) {
      nchwToNhwcDescriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, handle->pipelines->nchwToNhwc.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("[TransformerRMSNormLayer::TransformerRMSNormLayer()] allocate NCHW-to-NHWC descriptor set", res);
      nhwcToNchwDescriptorSet = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, handle->pipelines->nhwcToNchw.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("[TransformerRMSNormLayer::TransformerRMSNormLayer()] allocate NHWC-to-NCHW descriptor set", res);
    }
  }

  ~TransformerRMSNormLayer() {
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, weightBuf);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, zeroBetaBuf);
  }

  void forward(
    VkCommandBuffer cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* mask,
    VulkanBuffer* nhwcInput = nullptr,
    VulkanBuffer* nhwcOutput = nullptr
  ) {
    VkResult res = VK_SUCCESS;
    vkcompute::transformerRMSNorm(
      handle,
      handle->vulkanDevice,
      pipeline,
      cb,
      descriptorSet,
      useNHWC ? &handle->pipelines->nchwToNhwc : nullptr,
      nchwToNhwcDescriptorSet,
      useNHWC ? &handle->pipelines->nhwcToNchw : nullptr,
      nhwcToNchwDescriptorSet,
      input,
      output,
      nhwcInput,
      nhwcOutput,
      weightBuf,
      zeroBetaBuf,
      mask,
      batchSize,
      numChannels,
      paddedNNXYLen,
      paddedNNXYLen,
      handle->nnXLen * handle->nnYLen,
      channelsPadded,
      epsilon,
      handle->tuneParams.rmsNorm,
      useNHWC,
      &res
    );
    CHECK_VK_MSG("Execute TransformerRMSNorm", res);

  }


  TransformerRMSNormLayer() = delete;
  TransformerRMSNormLayer(const TransformerRMSNormLayer&) = delete;
  TransformerRMSNormLayer& operator=(const TransformerRMSNormLayer&) = delete;
};

struct RMSNormLayer {
  ComputeHandleInternal* handle;
  const string name;
  const int numChannels;
  const float epsilon;
  const bool spatial;
  const int paddedNNXYLen;
  const int activation;
  VulkanBuffer* gammaBuf;
  VulkanBuffer* betaBuf;
  VulkanBuffer* actOnesBuf;
  VulkanBuffer* actZerosBuf;
  vkcompute::SpatialRMSNormSizing sizing;
  VkDescriptorSet rmsNormDS;
  VkDescriptorSet rmsNormSumSqDS;
  VkDescriptorSet rmsNormReduceDS;
  VkDescriptorSet rmsNormApplyDS;
  VkDescriptorSet scaleBiasMaskDS;

  RMSNormLayer(
    ComputeHandleInternal* handle_,
    const RMSNormLayerDesc* desc,
    int activation_
  ) :
    handle(handle_),
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    spatial(desc->spatial),
    paddedNNXYLen(handle->paddedNNXYLen),
    activation(activation_),
    actOnesBuf(nullptr),
    actZerosBuf(nullptr)
  {
    testAssert(desc->gamma.size() == numChannels);
    testAssert(desc->beta.size() == numChannels);
    vector<float> gamma = desc->gamma;
    vector<float> beta = desc->beta;
    bool useFP16 = false;
    VkResult res;
    gammaBuf = vk_helper::createReadOnlyBuffer(handle->vulkanDevice, gamma, useFP16, &res);
    CHECK_VK_MSG("[RMSNormLayer::RMSNormLayer()] allocate gammaBuf",res);
    betaBuf = vk_helper::createReadOnlyBuffer(handle->vulkanDevice, beta, useFP16, &res);
    CHECK_VK_MSG("[RMSNormLayer::RMSNormLayer()] allocate betaBuf",res);

    if(spatial) {
      int tileSize = handle->tuneParams.spatialRMSNorm.TILE_SIZE;
      int chwSize = numChannels * paddedNNXYLen;
      sizing = vkcompute::computeSpatialRMSNormSizing(tileSize, chwSize);
      rmsNormSumSqDS = vk_helper::allocateDescriptorSet(handle->vulkanDevice, handle->pipelines->transformerSpatialRMSNormSumSq.descriptorSetLayout, &res);
      CHECK_VK_MSG("[RMSNormLayer::RMSNormLayer()] allocate rmsNormSumSqDS",res);
      rmsNormReduceDS = vk_helper::allocateDescriptorSet(handle->vulkanDevice, handle->pipelines->transformerSpatialRMSNormReduce.descriptorSetLayout, &res);
      CHECK_VK_MSG("[RMSNormLayer::RMSNormLayer()] allocate rmsNormReduce",res);
      rmsNormApplyDS = vk_helper::allocateDescriptorSet(handle->vulkanDevice, handle->pipelines->transformerSpatialRMSNormApply.descriptorSetLayout, &res);
      CHECK_VK_MSG("[RMSNormLayer::RMSNormLayer()] allocate rmsNormApplyDS",res);
    } else {
      rmsNormDS = vk_helper::allocateDescriptorSet(handle->vulkanDevice, handle->pipelines->transformerRmsNorm.descriptorSetLayout, &res);
      CHECK_VK_MSG("[RMSNormLayer::RMSNormLayer()] allocate rmsNormDS",res);
    }

    if( activation != ACTIVATION_IDENTITY) {
      if ( activation != ACTIVATION_SILU ) {
        throw StringError("RMSNormLayer: Unupported activation: " + Global::intToString(activation));
      }
      vector<float> ones(numChannels, 1.0f);
      vector<float> zeros(numChannels, 0.0f);
      bool useFP16Act = handle->usingFP16Storage;
      VkResult res;
      actOnesBuf = vk_helper::createReadOnlyBuffer(handle->vulkanDevice, ones, useFP16Act, &res);
      actZerosBuf = vk_helper::createReadOnlyBuffer(handle->vulkanDevice, zeros, useFP16Act, &res);
      const Pipeline& scaleBiasMaskPipeline = handle->pipelines->batchNormMaskSilu;
      scaleBiasMaskDS = vk_helper::allocateDescriptorSet(handle->vulkanDevice, scaleBiasMaskPipeline.descriptorSetLayout, &res);
      CHECK_VK_MSG("[RMSNormLayer::RMSNormLayer()] allocate scaleBiasMaskDS",res);
    }
  }

  ~RMSNormLayer() {
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, gammaBuf);
    vk_helper::releaseVulkanBuffer(handle->vulkanDevice, betaBuf);
    if(actOnesBuf) vk_helper::releaseVulkanBuffer(handle->vulkanDevice, actOnesBuf);
    if(actZerosBuf) vk_helper::releaseVulkanBuffer(handle->vulkanDevice, actZerosBuf);
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    if(!spatial)
      return ConvWorkspaceEltsNeeded();
    size_t floatToEltScale = handle->usingFP16Storage ? 2 : 1;
    size_t partialSumsFloats = maxBatchSize * (size_t)sizing.numCHWWorkgroups;
    size_t finalSumFloats = maxBatchSize;
    return ConvWorkspaceEltsNeeded(partialSumsFloats * floatToEltScale, finalSumFloats * floatToEltScale);
  }

  void forward(
    VkCommandBuffer cb,
    int batchSize,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  ) {
    if(!spatial) {
      Pipeline pipeline = handle->pipelines->useNHWC
        ? handle->pipelines->transformerRmsNormNHWC
        : handle->pipelines->transformerRmsNorm;
      auto writeDescriptorSets = {
        vk_helper::writeDescriptorSetBuffer(rmsNormDS, 0, input),
        vk_helper::writeDescriptorSetBuffer(rmsNormDS, 1, output),
        vk_helper::writeDescriptorSetBuffer(rmsNormDS, 2, gammaBuf),
        vk_helper::writeDescriptorSetBuffer(rmsNormDS, 3, betaBuf),
        vk_helper::writeDescriptorSetBuffer(rmsNormDS, 4, mask),
      };
      vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
      auto params = TransformerRMSNormPushParams();
      params.nSize = batchSize;
      params.cSize = numChannels;
      params.xySize = paddedNNXYLen;
      params.epsilon = epsilon;
      params.channelsPadded = handle->pipelines->useNHWC
        ? handle->getNHWCChannelsPadded(numChannels)
        : numChannels;
      int wgCSize = handle->tuneParams.rmsNorm.WG_C_SIZE;
      int wgXYSize = handle->tuneParams.rmsNorm.WG_XY_SIZE;
      int numXYGroups = (paddedNNXYLen + wgXYSize - 1) / wgXYSize;

      uint32_t wgCountX = numXYGroups;
      uint32_t wgCountY = batchSize;
      uint32_t wgCountZ = 1;
      vkcompute::dispatchPipeline(
        handle, handle->vulkanDevice, &pipeline, cb, rmsNormDS,
        &params, sizeof(params), wgCountX, wgCountY, wgCountZ, "TRANSFORMER_RMS_NORM"
      );
      vk_helper::barrierCommandBufferForBuffer(cb, output);
    }
    else {
      int tileSize = handle->tuneParams.spatialRMSNorm.TILE_SIZE;

      // Pass 1: SumSq
      {
        Pipeline pipeline = handle->pipelines->transformerSpatialRMSNormSumSq;
        auto writeDescriptorSets = {
          vk_helper::writeDescriptorSetBuffer(rmsNormSumSqDS, 0, input),
          vk_helper::writeDescriptorSetBuffer(rmsNormSumSqDS, 1, mask),
          vk_helper::writeDescriptorSetBuffer(rmsNormSumSqDS, 2, convWorkspace)
        };
        VkResult res = vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
        CHECK_VK_MSG("RMSNorm::forward() update descriptor sets for spatial rms sum sq", res);
        auto params = TransformerSpatialRMSNormSumSqPushParams();
        params.nSize = batchSize;
        params.cSize = numChannels;
        params.xySize = paddedNNXYLen;
        params.tilesPerGroup = sizing.tilesPerGroupPass1;
        params.channelsPadded = handle->pipelines->useNHWC
          ? handle->getNHWCChannelsPadded(numChannels)
          : numChannels;
        uint32_t globalSizeX = sizing.numCHWWorkgroups * tileSize;
        uint32_t globalSizeY = batchSize;
        uint32_t wgCountX = (globalSizeX + pipeline.localSizeX - 1) / pipeline.localSizeX;
        uint32_t wgCountY = (globalSizeY + pipeline.localSizeY - 1) / pipeline.localSizeY;
        uint32_t wgCountZ = 1;
        vkcompute::dispatchPipeline(
          handle, handle->vulkanDevice, &pipeline, cb, rmsNormSumSqDS,
          &params, sizeof(params), wgCountX, wgCountY, wgCountZ, "TRANSFORMER_SPATIAL_RMS_NORM_SUM_SQ"
        );
        vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace);
      }

      // Pass 2: Reduce partial sums to final sum
      {
        Pipeline pipeline = handle->pipelines->transformerSpatialRMSNormReduce;
        auto writeDescriptorSets = {
          vk_helper::writeDescriptorSetBuffer(rmsNormReduceDS, 0, convWorkspace),
          vk_helper::writeDescriptorSetBuffer(rmsNormReduceDS, 1, convWorkspace2),
        };
        VkResult res = vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
        CHECK_VK_MSG("RMSNorm::forward() update descriptor sets for spatial rms reduce", res);
        auto params = TransformerSpatialRMSNormReducePushParams();
        params.nSize = batchSize;
        params.numPartials = sizing.numCHWWorkgroups;
        params.tilesPerGroup = sizing.tilesPerGroupPass2;
        uint32_t globalSizeX = tileSize;
        uint32_t globalSizeY = batchSize;
        uint32_t wgCountX = (globalSizeX + pipeline.localSizeX - 1) / pipeline.localSizeX;
        uint32_t wgCountY = globalSizeY;
        uint32_t wgCountZ = 1;
        vkcompute::dispatchPipeline(
          handle, handle->vulkanDevice, &pipeline, cb, rmsNormReduceDS,
          &params, sizeof(params), wgCountX, wgCountY, wgCountZ, "TRANSFORMER_SPATIAL_RMS_NORM_REDUCE"
        );
        vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace2);
      }

      // Apply normalization
      {
        Pipeline pipeline = handle->pipelines->transformerSpatialRMSNormApply;
        auto writeDescriptorSets = {
          vk_helper::writeDescriptorSetBuffer(rmsNormApplyDS, 0, input),
          vk_helper::writeDescriptorSetBuffer(rmsNormApplyDS, 1, output),
          vk_helper::writeDescriptorSetBuffer(rmsNormApplyDS, 2, gammaBuf),
          vk_helper::writeDescriptorSetBuffer(rmsNormApplyDS, 3, betaBuf),
          vk_helper::writeDescriptorSetBuffer(rmsNormApplyDS, 4, mask),
          vk_helper::writeDescriptorSetBuffer(rmsNormApplyDS, 5, maskSum),
          vk_helper::writeDescriptorSetBuffer(rmsNormApplyDS, 6, convWorkspace2)
        };
        VkResult res = vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
        CHECK_VK_MSG("RMSNorm::forward() update descriptor sets for spatial rms apply", res);
        auto params = TransformerSpatialRMSNormApplyPushParams();
        params.nSize = batchSize;
        params.cSize = numChannels;
        params.xySize = paddedNNXYLen;
        params.eps = epsilon;
        params.channelsPadded = handle->pipelines->useNHWC
          ? handle->getNHWCChannelsPadded(numChannels)
          : numChannels;
        uint32_t totalElem = params.cSize * params.xySize;
        uint32_t eltsPerThread = handle->tuneParams.spatialRMSNorm.APPLY_ELTS_PER_THREAD;
        uint32_t numThreads = (totalElem + eltsPerThread - 1) / eltsPerThread;

        uint32_t wgCountX = (numThreads + pipeline.localSizeX - 1) / pipeline.localSizeX;
        uint32_t wgCountY = batchSize;
        uint32_t wgCountZ = 1;
        vkcompute::dispatchPipeline(
          handle, handle->vulkanDevice, &pipeline, cb, rmsNormApplyDS,
          &params, sizeof(params), wgCountX, wgCountY, wgCountZ, "TRANSFORMER_SPATIAL_RMS_NORM_APPLY"
        );
        vk_helper::barrierCommandBufferForBuffer(cb, output);
      }
    }

    // Apply activation in-place on output if needed
    if(activation == ACTIVATION_SILU) {
      Pipeline pipeline = handle->pipelines->batchNormMaskSilu;
      auto writeDescriptorSets = {
        vk_helper::writeDescriptorSetBuffer(scaleBiasMaskDS, 0, output),
        vk_helper::writeDescriptorSetBuffer(scaleBiasMaskDS, 1, output),
        vk_helper::writeDescriptorSetBuffer(scaleBiasMaskDS, 2, actOnesBuf),
        vk_helper::writeDescriptorSetBuffer(scaleBiasMaskDS, 3, actZerosBuf),
        vk_helper::writeDescriptorSetBuffer(scaleBiasMaskDS, 4, mask)
      };
      vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
      auto params = BatchNormMaskParams();
      params.batchSize = batchSize;
      params.numChannels = numChannels;
      params.nnXYLen = paddedNNXYLen;
      params.maskSpatialStride = paddedNNXYLen;
      uint32_t globalSizeX = static_cast<uint32_t>(vk_helper::powerOf2ify((size_t)paddedNNXYLen));
      uint32_t globalSizeY = static_cast<uint32_t>(vk_helper::powerOf2ify((size_t)numChannels));
      uint32_t wgCountX = (globalSizeX + pipeline.localSizeX - 1) / pipeline.localSizeX;
      uint32_t wgCountY = (globalSizeY + pipeline.localSizeY - 1) / pipeline.localSizeY;
      uint32_t wgCountZ = 1;
      vkcompute::dispatchPipeline(
        handle, handle->vulkanDevice, &pipeline, cb, scaleBiasMaskDS,
        &params, sizeof(params), wgCountX, wgCountY, wgCountZ, "BATCHNORM_MASK_ACT_SILU"
      );
      vk_helper::barrierCommandBufferForBuffer(cb, output);
    }
  }

};

static bool canUsePackedQKVHgemm(
  const ComputeHandleInternal* handle,
  const int inChannels
) {
  const auto& hgemmParams = handle->tuneParams.hgemmCooperativeMatrixNCHW;
  return
    handle->usingFP16Storage &&
    handle->tuneParams.vulkan.canUseCooperativeMatrix &&
    handle->tuneParams.vulkan.shouldUseFP16Storage &&
    handle->tuneParams.vulkan.shouldUseFP16Compute &&
    handle->tuneParams.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
    hgemmParams.isValid() &&
    handle->paddedNNXYLen % hgemmParams.getRequiredSpatialAlignment() == 0 &&
    inChannels % hgemmParams.KWG == 0;
}

static int getPackedQKVOutChannels(
  const ComputeHandleInternal* handle,
  const int inChannels,
  const int logicalOutChannels
) {
  if(!canUsePackedQKVHgemm(handle, inChannels))
    return logicalOutChannels;
  return vk_helper::roundUpToMultipleInt(
    logicalOutChannels,
    handle->tuneParams.hgemmCooperativeMatrixNCHW.NWG
  );
}

static MatMulLayerDesc makePackedQKVDesc(
  const TransformerAttentionDesc* desc,
  const int physicalOutChannels
) {
  const MatMulLayerDesc& qDesc = desc->qProj;
  const MatMulLayerDesc& kDesc = desc->kProj;
  const MatMulLayerDesc& vDesc = desc->vProj;
  testAssert(qDesc.inChannels == kDesc.inChannels);
  testAssert(qDesc.inChannels == vDesc.inChannels);
  testAssert(qDesc.weights.size() == static_cast<size_t>(qDesc.inChannels * qDesc.outChannels));
  testAssert(kDesc.weights.size() == static_cast<size_t>(kDesc.inChannels * kDesc.outChannels));
  testAssert(vDesc.weights.size() == static_cast<size_t>(vDesc.inChannels * vDesc.outChannels));
  testAssert(physicalOutChannels >= qDesc.outChannels + kDesc.outChannels + vDesc.outChannels);

  MatMulLayerDesc result;
  result.name = qDesc.name + "_packed_qkv";
  result.inChannels = qDesc.inChannels;
  result.outChannels = physicalOutChannels;
  result.weights.assign(
    static_cast<size_t>(result.inChannels) * static_cast<size_t>(result.outChannels),
    0.0f
  );

  for(int ic = 0; ic < result.inChannels; ic++) {
    const size_t qSrcBase = static_cast<size_t>(ic) * qDesc.outChannels;
    const size_t kSrcBase = static_cast<size_t>(ic) * kDesc.outChannels;
    const size_t vSrcBase = static_cast<size_t>(ic) * vDesc.outChannels;
    const size_t dstBase = static_cast<size_t>(ic) * result.outChannels;
    for(int oc = 0; oc < qDesc.outChannels; oc++)
      result.weights[dstBase + oc] = qDesc.weights[qSrcBase + oc];
    for(int oc = 0; oc < kDesc.outChannels; oc++)
      result.weights[dstBase + qDesc.outChannels + oc] = kDesc.weights[kSrcBase + oc];
    for(int oc = 0; oc < vDesc.outChannels; oc++)
      result.weights[dstBase + qDesc.outChannels + kDesc.outChannels + oc] = vDesc.weights[vSrcBase + oc];
  }
  return result;
}

static MatMulLayerDesc makePackedFFNDesc(
  const TransformerFFNDesc* desc,
  const int physicalOutChannels
) {
  if(!desc->useSwiGLU)
    throw StringError("Non-SwiGLU transformer FFN is not yet supported in Vulkan backend");

  const MatMulLayerDesc& linearDesc = desc->linear1;
  const MatMulLayerDesc& gateDesc = desc->linearGate;
  testAssert(linearDesc.inChannels == gateDesc.inChannels);
  testAssert(linearDesc.outChannels == gateDesc.outChannels);
  testAssert(linearDesc.inChannels == desc->numChannels);
  testAssert(linearDesc.outChannels == desc->ffnChannels);
  testAssert(linearDesc.weights.size() == static_cast<size_t>(linearDesc.inChannels * linearDesc.outChannels));
  testAssert(gateDesc.weights.size() == static_cast<size_t>(gateDesc.inChannels * gateDesc.outChannels));
  testAssert(physicalOutChannels >= linearDesc.outChannels + gateDesc.outChannels);

  MatMulLayerDesc result;
  result.name = linearDesc.name + "_packed_linear_gate";
  result.inChannels = linearDesc.inChannels;
  result.outChannels = physicalOutChannels;
  result.weights.assign(
    static_cast<size_t>(result.inChannels) * static_cast<size_t>(result.outChannels),
    0.0f
  );

  for(int ic = 0; ic < result.inChannels; ic++) {
    const size_t linearSrcBase = static_cast<size_t>(ic) * linearDesc.outChannels;
    const size_t gateSrcBase = static_cast<size_t>(ic) * gateDesc.outChannels;
    const size_t dstBase = static_cast<size_t>(ic) * result.outChannels;
    for(int oc = 0; oc < linearDesc.outChannels; oc++)
      result.weights[dstBase + oc] = linearDesc.weights[linearSrcBase + oc];
    for(int oc = 0; oc < gateDesc.outChannels; oc++)
      result.weights[dstBase + linearDesc.outChannels + oc] = gateDesc.weights[gateSrcBase + oc];
  }
  return result;
}

static int getPackedFFNOutChannels(
  const ComputeHandleInternal* handle,
  const TransformerFFNDesc* desc
) {
  return getPackedQKVOutChannels(
    handle,
    desc->linear1.inChannels,
    desc->linear1.outChannels + desc->linearGate.outChannels
  );
}

struct TransformerAttentionBlock {
  ComputeHandleInternal *handle;
  const std::string name;
  const int numHeads;
  const int numKVHeads;
  const int qHeadDim;
  const int vHeadDim;
  const bool useRope;
  const bool learnableRope;
  const int nnXLen;
  const int nnYLen;
  const int paddedNNXYLen;
  const int inChannels;  // = numHeads * qHeadDim (or whatever c_main is)
  const int qkvPhysicalOutChannels;

  MatMulLayerDesc qkvProjDesc;
  TransformerRMSNormLayer* preLN;
  TransformerMatMulLayer* qkvProj;
  TransformerMatMulLayer* outProj;
  TransformerApplyRoPELayer* qRoPE;
  TransformerApplyRoPELayer* kRoPE;
  TransformerAttentionLayer* attention;

  // RoPE data
  VulkanBuffer* ropeCosTable;
  VulkanBuffer* ropeSinTable;
  VkDescriptorSet pointwiseDS = VK_NULL_HANDLE;

  int ropeNumPairs;

  TransformerAttentionBlock(
    ComputeHandleInternal* handle,
    const TransformerAttentionDesc* desc,
    int nnX,
    int nnY
  ) : 
    handle(handle),
    numHeads(desc->numHeads),
    numKVHeads(desc->numKVHeads),
    qHeadDim(desc->qHeadDim),
    vHeadDim(desc->vHeadDim),
    useRope(desc->useRope),
    learnableRope(desc->learnableRope),
    nnXLen(nnX),
    nnYLen(nnY),
    paddedNNXYLen(handle->paddedNNXYLen),
    inChannels(desc->qProj.inChannels),
    qkvPhysicalOutChannels(
      getPackedQKVOutChannels(
        handle,
        desc->qProj.inChannels,
        numHeads * qHeadDim + numKVHeads * qHeadDim + numKVHeads * vHeadDim
      )
    ),
    qkvProjDesc(makePackedQKVDesc(desc, qkvPhysicalOutChannels)),
    preLN(new TransformerRMSNormLayer(handle, &desc->preLN)),
    qkvProj(new TransformerMatMulLayer(handle, &qkvProjDesc)),
    outProj(new TransformerMatMulLayer(handle, &desc->outProj)),
    qRoPE(new TransformerApplyRoPELayer(handle, qkvPhysicalOutChannels)),
    kRoPE(new TransformerApplyRoPELayer(handle, qkvPhysicalOutChannels)),
    ropeCosTable(nullptr),
    ropeSinTable(nullptr),
    ropeNumPairs(0),
    attention(new TransformerAttentionLayer(handle, numHeads, numKVHeads))
  {
    qkvProjDesc.releaseWeights();
    ropeNumPairs = qHeadDim / 2;
    vector<float> cosTableData;
    vector<float> sinTableData;
    if ( useRope ) {
      desc->computeRopeCosSin(nnXLen, nnYLen, paddedNNXYLen, cosTableData, sinTableData);
    }
    else {
      // Attention always has the RoPE descriptor bindings. These tables are
      // never read when useRope is false, but still need valid descriptors.
      cosTableData = {1.0f};
      sinTableData = {0.0f};
    }
    bool useFP16 = false;
    VkResult res;
    ropeCosTable = vk_helper::createReadOnlyBuffer(handle->vulkanDevice, cosTableData, useFP16, &res);
    CHECK_VK_MSG("[TransformerAttentionBlock::TransformerAttentionBlock()] allocate ropeCosTable", res);
    ropeSinTable = vk_helper::createReadOnlyBuffer(handle->vulkanDevice, sinTableData, useFP16, &res);
    CHECK_VK_MSG("[TransformerAttentionBlock::TransformerAttentionBlock()] allocate ropeSinTable", res);

    pointwiseDS = vk_helper::allocateDescriptorSet(handle->vulkanDevice, handle->pipelines->addPointWise.descriptorSetLayout, &res);
    CHECK_VK_MSG("[TransformerAttentionBlock::TransformerAttentionBlock() allocate pointwise descriptorset]", res);
  }

  ~TransformerAttentionBlock() {
    if ( ropeCosTable != nullptr ) {
      vk_helper::releaseVulkanBuffer(handle->vulkanDevice, ropeCosTable);
    } 
    if ( ropeSinTable != nullptr ) {
      vk_helper::releaseVulkanBuffer(handle->vulkanDevice, ropeSinTable);
    }

    delete preLN;
    delete outProj;
    delete qkvProj;
    delete qRoPE;
    delete kRoPE;
    delete attention;
  }

  void forward(
    VkCommandBuffer cb,
    ScratchBuffers* scratch,
    int batchSize,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* convWorkspace
  ) {
    const int seqLen = paddedNNXYLen;
    const int qTotalDim = numHeads * qHeadDim;
    const int kTotalDim = numKVHeads * qHeadDim;
    const int qOffset = 0;
    const int kOffset = qTotalDim * seqLen;
    const int vOffset = (qTotalDim + kTotalDim) * seqLen;
    const int packedBatchStride = qkvPhysicalOutChannels * seqLen;

    // Step 1: RMSNorm
    // preLN: trunk -> trunkScratch (normalized)
    preLN->forward(cb, batchSize, trunk, trunkScratch, mask);

    // Step 2: Combined Q/K/V projection into [Q | K | V | padding]
    SizedBuf<VulkanBuffer*> packedQKV(
      scratch->allocator,
      scratch->getBufSizeXY(qkvPhysicalOutChannels)
    );
    qkvProj->forward(cb, scratch, batchSize, trunkScratch, packedQKV.buf, mask, convWorkspace);

    if(useRope) {
      const bool useNHWC = handle->pipelines->useNHWC;
      const int channelsPadded = handle->getNHWCChannelsPadded(qkvPhysicalOutChannels);
      const int ropeBatchStride = useNHWC ? seqLen * channelsPadded : packedBatchStride;
      const int qRegionOffset = useNHWC ? 0 : qOffset;
      const int kRegionOffset = useNHWC ? qTotalDim : kOffset;
      qRoPE->forward(
        cb, batchSize, packedQKV.buf, ropeCosTable, ropeSinTable,
        numHeads, numKVHeads, qHeadDim, seqLen, ropeNumPairs,
        learnableRope ? 1 : 0, qRegionOffset, ropeBatchStride
      );
      kRoPE->forward(
        cb, batchSize, packedQKV.buf, ropeCosTable, ropeSinTable,
        numKVHeads, numKVHeads, qHeadDim, seqLen, ropeNumPairs,
        learnableRope ? 1 : 0, kRegionOffset, ropeBatchStride
      );
    }

    // Step 3: Scaled dot product attention. RoPE has already been applied to
    // the packed Q and K regions, so the attention shader must not rotate them again.
    SizedBuf<VulkanBuffer*> attnOut(scratch->allocator, scratch->getBufSizeXY(numHeads * vHeadDim));
    attention->forward(
      cb,
      batchSize,
      packedQKV.buf,
      attnOut.buf,
      nullptr,
      nullptr,
      mask,
      ropeCosTable,
      ropeSinTable,
      qOffset,
      kOffset,
      vOffset,
      packedBatchStride,
      false,
      false,
      0
    );
    vk_helper::barrierCommandBufferForBuffer(cb, attnOut.buf);
    // Step 4: Output projection: attnOut (N, numHeads*vHeadDim, H, W) -> trunkScratch (N, C, H, W)
    outProj->forward(cb, scratch, batchSize, attnOut.buf, trunkScratch, mask, convWorkspace);
    // Step 5: Add residual: trunk += trunkScratch
    vkcompute::performAddPointWise(handle, cb, pointwiseDS, trunk, trunkScratch, checkedTensorElts(handle, batchSize, inChannels, paddedNNXYLen, "Vulkan addPointwise"), false);
  }


  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, qkvProj->requiredConvWorkspaceElts(handle, maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, outProj->requiredConvWorkspaceElts(handle, maxBatchSize));
    return maxElts;
  }

  TransformerAttentionBlock() = delete;
  TransformerAttentionBlock(const TransformerAttentionBlock&) = delete;
  TransformerAttentionBlock& operator=(const TransformerAttentionBlock&) = delete;
};

struct TransformerFFNBlock {
  ComputeHandleInternal *handle;
  const std::string name;
  const int numChannels;
  const int ffnChannels;
  const int paddedNNXYLen;
  const int packedOutChannels;

  MatMulLayerDesc packedLinearDesc;
  TransformerRMSNormLayer* preLN;
  TransformerMatMulLayer* linear1AndGate;
  TransformerMatMulLayer* linear2;

  VkDescriptorSet pointwiseDS;
  VkDescriptorSet swigluDS;
  bool usingTransformerDualGemmSwiGLU;
  VkDescriptorSet transformerDualGemmSwiGLUDS;
  VkDescriptorSet nchwToNhwcDS;
  VkDescriptorSet nhwcToNchwDS;

  TransformerFFNBlock(
    ComputeHandleInternal *handle,
    const TransformerFFNDesc* desc
  ) :
    handle(handle),
    name(desc->name),
    numChannels(desc->numChannels),
    ffnChannels(desc->ffnChannels),
    paddedNNXYLen(handle->paddedNNXYLen),
    packedOutChannels(getPackedFFNOutChannels(handle, desc)),
    packedLinearDesc(makePackedFFNDesc(desc, packedOutChannels)),
    preLN(new TransformerRMSNormLayer(handle, &desc->preLN)),
    linear1AndGate(new TransformerMatMulLayer(handle, &packedLinearDesc)),
    linear2(new TransformerMatMulLayer(handle, &desc->linear2)),
    pointwiseDS(VK_NULL_HANDLE),
    swigluDS(VK_NULL_HANDLE),
    usingTransformerDualGemmSwiGLU(false),
    transformerDualGemmSwiGLUDS(VK_NULL_HANDLE),
    nchwToNhwcDS(VK_NULL_HANDLE),
    nhwcToNchwDS(VK_NULL_HANDLE)
  {
    packedLinearDesc.releaseWeights();
    VkResult res;
    pointwiseDS = vk_helper::allocateDescriptorSet(handle->vulkanDevice, handle->pipelines->addPointWise.descriptorSetLayout, &res);
    CHECK_VK_MSG("[TransformerFFNBlock::TransformerFFNBlock()] allocate pointwiseDS", res);
    swigluDS = vk_helper::allocateDescriptorSet(handle->vulkanDevice, handle->pipelines->transformerSwiGLU.descriptorSetLayout, &res);
    CHECK_VK_MSG("[TransformerFFNBlock::TransformerFFNBlock()] allocate swigluDS", res);

    const auto& dualParams = handle->tuneParams.transformerDualGemmSwiGLU;
    usingTransformerDualGemmSwiGLU =
      handle->usingFP16Storage &&
      handle->pipelines->useNHWC &&
      handle->tuneParams.vulkan.canUseCooperativeMatrix &&
      handle->tuneParams.vulkan.shouldUseFP16Storage &&
      handle->tuneParams.vulkan.shouldUseFP16Compute &&
      handle->tuneParams.vulkan.shouldUseTransformerDualGemmSwiGLU &&
      dualParams.isValid() &&
      numChannels % dualParams.KWG == 0 &&
      ffnChannels % dualParams.NWG == 0 &&
    paddedNNXYLen % dualParams.getRequiredSpatialAlignment() == 0 &&
      handle->pipelines->transformerDualGemmSwiGLU.pipeline != VK_NULL_HANDLE;
    if(usingTransformerDualGemmSwiGLU) {
      nchwToNhwcDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, handle->pipelines->nchwToNhwc.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("[TransformerFFNBlock::TransformerFFNBlock()] allocate nchwToNhwcDS", res);
      nhwcToNchwDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, handle->pipelines->nhwcToNchw.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("[TransformerFFNBlock::TransformerFFNBlock()] allocate nhwcToNchwDS", res);
      transformerDualGemmSwiGLUDS = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice,
        handle->pipelines->transformerDualGemmSwiGLU.descriptorSetLayout,
        &res
      );
      CHECK_VK_MSG("[TransformerFFNBlock::TransformerFFNBlock()] allocate transformerDualGemmSwiGLUDS", res);
    }
  }

   ~TransformerFFNBlock() {
    delete preLN;
    delete linear1AndGate;
    delete linear2;
  }

  void forward(
    VkCommandBuffer cb,
    ScratchBuffers* scratch,
    int batchSize,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* convWorkspace
  ) {
     // Step 1: RMSNorm
    preLN->forward(cb, batchSize, trunk, trunkScratch, mask);

    // Step 2: write compact SwiGLU output for linear2.
    SizedBuf<VulkanBuffer*> ffnOutBuf(scratch->allocator, scratch->getBufSizeXY(ffnChannels));
    if(usingTransformerDualGemmSwiGLU) {
      VkResult res;
      vkcompute::doTransformerDualGemmSwiGLU(
        handle,
        handle->vulkanDevice, handle->tuneParams, &handle->pipelines->transformerDualGemmSwiGLU,
        cb, transformerDualGemmSwiGLUDS,
        trunkScratch, nullptr, linear1AndGate->filter, nullptr, ffnOutBuf.buf,
        &handle->pipelines->nchwToNhwc, nchwToNhwcDS,
        &handle->pipelines->nhwcToNchw, nhwcToNchwDS,
        batchSize, paddedNNXYLen, handle->nnXLen * handle->nnYLen,
        ffnChannels, numChannels, packedOutChannels, &res
      );
      CHECK_VK_MSG("Execute fused transformer dual-GEMM SwiGLU", res);
    }
    else {
      SizedBuf<VulkanBuffer*> ffnBuf(scratch->allocator, scratch->getBufSizeXY(packedOutChannels));
      linear1AndGate->forward(cb, scratch, batchSize, trunkScratch, ffnBuf.buf, mask, convWorkspace);
      const int totalSize = checkedTotalElts(batchSize, ffnChannels, paddedNNXYLen, "Vulkan SwiGLU");
      const int packedInputBatchStride = checkedTotalElts(1, packedOutChannels, paddedNNXYLen, "Vulkan packed FFN");
      const int outputBatchStride = checkedTotalElts(1, ffnChannels, paddedNNXYLen, "Vulkan SwiGLU");
      vkcompute::doSwiGLU(
        handle, handle->vulkanDevice, cb, swigluDS, handle->pipelines->transformerSwiGLU,
        handle->tuneParams, ffnBuf.buf, ffnBuf.buf, ffnOutBuf.buf, totalSize,
        packedInputBatchStride, outputBatchStride
      );
    }
    // Step 3: linear2 projection: ffnOutBuf (N, ffnC, H, W) -> trunkScratch (N, C, H, W)
    linear2->forward(cb, scratch, batchSize, ffnOutBuf.buf, trunkScratch, mask, convWorkspace);

    // Step 4: Add residual
    vkcompute::performAddPointWise(handle, cb, pointwiseDS, trunk, trunkScratch, checkedTensorElts(handle, batchSize, numChannels, paddedNNXYLen, "Vulkan addPointWise"), false);
  }


  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, linear1AndGate->requiredConvWorkspaceElts(handle, maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, linear2->requiredConvWorkspaceElts(handle, maxBatchSize));
    return maxElts;
  }

  TransformerFFNBlock() = delete;
  TransformerFFNBlock(const TransformerFFNBlock&) = delete;
  TransformerFFNBlock& operator=(const TransformerFFNBlock&) = delete;
};

/**
 * @brief Basic Residual Block, Consist of two conv layers with BN and Activation and one skip connection
 */
struct ResidualBlock {
  ComputeHandleInternal *handle;
  const std::string name;
  NormActConv *normActConv;
  NormActConv *normActConv2;
  const int nnXLen; // TODO: remove this after paddedNNXY apply
  const int nnYLen; // TODO: remove this after paddedNNXY apply
  const int paddedNNXYLen;
  VkDescriptorSet addPointWiseDS = VK_NULL_HANDLE;

  ResidualBlock(
    ComputeHandleInternal *handle_,
    const ResidualBlockDesc* desc,
    int nnXLen,
    int nnYLen,
    bool useFP16
  ):
    handle(handle_),
    name(desc->name),
    nnXLen(nnXLen),
    nnYLen(nnYLen),
    paddedNNXYLen(handle_->paddedNNXYLen)
  {
    // TODO: FP16 support
    normActConv = new NormActConv(
      handle,
      &desc->regularConv,
      &desc->preBN,
      &desc->preActivation,
      nnXLen,
      nnYLen,
      useFP16
    );
    normActConv2 = new NormActConv(
      handle,
      &desc->finalConv,
      &desc->midBN,
      &desc->midActivation,
      nnXLen,
      nnYLen,
      useFP16
    );
    VkResult res = VK_SUCCESS;
    addPointWiseDS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice, handle->pipelines->addPointWise.descriptorSetLayout, &res
    );
    CHECK_VK_MSG("Allocate ResidualBlock pointwise descriptor set", res);
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
    vkcompute::performAddPointWise(handle, cb, addPointWiseDS, trunk, trunkScratch, checkedTensorElts(handle, batchSize, normActConv2->outChannels, paddedNNXYLen, "Vulkan addPointWise"), false);
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
  const int paddedNNXYLen;
  const int regularChannels;
  const int gpoolChannels;

  // VkCommandBuffer gpoolCB = VK_NULL_HANDLE;
  VkDescriptorSet gpoolDS = VK_NULL_HANDLE;
  VkDescriptorSet gpoolNchwToNhwcDS = VK_NULL_HANDLE;
  // VkCommandBuffer addChannelCB = VK_NULL_HANDLE;
  VkDescriptorSet addChannelDS = VK_NULL_HANDLE;
  VkDescriptorSet addChannelNchwToNhwcDS = VK_NULL_HANDLE;
  VkDescriptorSet addChannelNhwcToNchwDS = VK_NULL_HANDLE;
  // VkCommandBuffer addPointWiseCB = VK_NULL_HANDLE;
  VkDescriptorSet addPointWiseDS = VK_NULL_HANDLE;

  GlobalPoolingResidualBlock(
    ComputeHandleInternal *handle_,
    const GlobalPoolingResidualBlockDesc* desc,
    int nnXLen_,
    int nnYLen_,
    bool useFP16
  ):
    handle(handle_),
    name(desc->name),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    nnXYLen(nnXLen_ * nnYLen_),
    regularChannels(desc->regularConv.outChannels),
    gpoolChannels(desc->gpoolConv.outChannels),
    paddedNNXYLen(handle->paddedNNXYLen)
  {
    preBN = new BatchNormLayer(handle, &desc->preBN, &desc->preActivation, useFP16);
    regularConv = new ConvLayer(handle, &desc->regularConv, nnXLen, nnYLen, useFP16);
    gpoolConv = new ConvLayer(handle, &desc->gpoolConv, nnXLen, nnYLen, useFP16);
    gpoolBN = new BatchNormLayer(handle, &desc->gpoolBN, &desc->gpoolActivation, useFP16);
    gpoolToBiasMul = new MatmulLayer(handle, &desc->gpoolToBiasMul);
    normActConv2 = new NormActConv(handle, &desc->finalConv, &desc->midBN, &desc->midActivation, nnXLen, nnYLen, useFP16);
    VkResult res = VK_SUCCESS;
    gpoolDS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice, handle->pipelines->globalPoolingChannelsFp32.descriptorSetLayout, &res
    );
    CHECK_VK_MSG("Allocate GlobalPoolingResidualBlock pooling descriptor set", res);
    addChannelDS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice, handle->pipelines->addChannelBias.descriptorSetLayout, &res
    );
    CHECK_VK_MSG("Allocate GlobalPoolingResidualBlock channel-bias descriptor set", res);
    addPointWiseDS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice, handle->pipelines->addPointWise.descriptorSetLayout, &res
    );
    CHECK_VK_MSG("Allocate GlobalPoolingResidualBlock pointwise descriptor set", res);
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
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, preBN->requiredConvWorkspaceElts(handle, maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, gpoolBN->requiredConvWorkspaceElts(handle, maxBatchSize));
    if(handle->pipelines->useNHWC) {
      const size_t channelsPadded = handle->getNHWCChannelsPadded(gpoolChannels);
      const size_t conversionElts = maxBatchSize * static_cast<size_t>(nnXYLen) * channelsPadded;
      maxElts.size1 = std::max(maxElts.size1, conversionElts);
    }
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

    preBN->forward(
      cb, batchSize, trunk, mask, trunkScratch,
      convWorkspace, convWorkspace2, nnXLen * nnYLen, nnXLen * nnYLen
    );
    regularConv->forward(cb, batchSize, trunkScratch, regularOut.buf, convWorkspace, convWorkspace2);
    gpoolConv->forward(cb, batchSize, trunkScratch, gpoolOut.buf, convWorkspace, convWorkspace2);
    gpoolBN->forward(
      cb, batchSize, gpoolOut.buf, mask, gpoolOut.buf,
      convWorkspace, convWorkspace2, nnXLen * nnYLen, nnXLen * nnYLen
    );
    VkResult res;;
    vkcompute::performGpoolMask(handle, cb, gpoolDS, gpoolNchwToNhwcDS, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, batchSize, gpoolChannels, paddedNNXYLen, convWorkspace, &res, false);
    gpoolToBiasMul->forward(cb, batchSize, gpoolConcat.buf, gpoolBias.buf);
    vkcompute::performAddChannelBiases(
      handle, cb, addChannelDS, addChannelNchwToNhwcDS, addChannelNhwcToNchwDS,
      regularOut.buf, gpoolBias.buf, batchSize * regularChannels, regularChannels,
      paddedNNXYLen, convWorkspace, false
    );
    normActConv2->forward(cb, batchSize, regularOut.buf, regularOut.buf, trunkScratch, mask, convWorkspace, convWorkspace2);
    vkcompute::performAddPointWise(handle, cb, addPointWiseDS, trunk, trunkScratch, checkedTensorElts(handle, batchSize, normActConv2->outChannels, paddedNNXYLen, "Vulkan addPointWise"), false);
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
  const int paddedNNXYLen;
  // std::vector<VkCommandBuffer> commandBuffers;
  // VkCommandBuffer addPointWiseCB = VK_NULL_HANDLE;
  VkDescriptorSet addPointWiseDS = VK_NULL_HANDLE;

  NestedResidualBlock(
    ComputeHandleInternal *handle_,
    const NestedBottleneckResidualBlockDesc* desc,
    int nnXLen_,
    int nnYLen_,
    bool useFP16
  ):
    handle(handle_),
    name(desc->name),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    paddedNNXYLen(handle->paddedNNXYLen)
  {
    normActConv = new NormActConv(handle, &desc->preConv, &desc->preBN, &desc->preActivation, nnXLen, nnYLen, useFP16);
    blocks = new BlockStack(handle, desc->blocks, desc->numBlocks, desc->preConv.outChannels, nnXLen, nnYLen, useFP16);
    normActConv2 = new NormActConv(handle, &desc->postConv, &desc->postBN, &desc->postActivation, nnXLen, nnYLen, useFP16);
    VkResult res = VK_SUCCESS;
    addPointWiseDS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice, handle->pipelines->addPointWise.descriptorSetLayout, &res
    );
    CHECK_VK_MSG("Allocate NestedResidualBlock pointwise descriptor set", res);
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
    vkcompute::performAddPointWise(handle, cb, addPointWiseDS, trunk, trunkScratch, checkedTensorElts(handle, batchSize, normActConv2->outChannels, paddedNNXYLen, "Vulkan addPointWise"), false);
  }

};

BlockStack::BlockStack(
  ComputeHandleInternal *handle_,
  const std::vector<std::pair<int, unique_ptr_void>> &descBlocks,
  int numBlocks_,
  int trunkNumChannels_,
  int nnXLen_,
  int nnYLen_,
  bool useFP16
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
          nnYLen,
          useFP16
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
          nnYLen,
          useFP16
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
          nnYLen,
          useFP16
        )
      );
      blocks.push_back(std::make_pair(blockType, std::move(blockPtr)));
    } else if ( blockType == TRANSFORMER_ATTENTION_BLOCK_KIND ) {
      const TransformerAttentionDesc* attentionDesc = static_cast<const TransformerAttentionDesc*>(descBlocks[i].second.get());
      unique_ptr_void blockPtr = make_unique_void(
        new TransformerAttentionBlock(handle, attentionDesc, nnXLen, nnYLen)
      );
      blocks.emplace_back(TRANSFORMER_ATTENTION_BLOCK_KIND, std::move(blockPtr));
    } else if ( blockType == TRANSFORMER_FFN_BLOCK_KIND ) {
      const TransformerFFNDesc* ffnDesc = static_cast<const TransformerFFNDesc*>(descBlocks[i].second.get());
      unique_ptr_void blockPtr = make_unique_void(
        new TransformerFFNBlock(handle, ffnDesc)
      );
      blocks.emplace_back(TRANSFORMER_FFN_BLOCK_KIND, std::move(blockPtr));
    }else {
      ASSERT_UNREACHABLE;
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
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionBlock* block = static_cast<TransformerAttentionBlock*>(blocks[i].second.get());
      maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,block->requiredConvWorkspaceElts(handle,maxBatchSize));
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNBlock* block = static_cast<TransformerFFNBlock*>(blocks[i].second.get());
      maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,block->requiredConvWorkspaceElts(handle,maxBatchSize));
    }

    else {
      ASSERT_UNREACHABLE;
    }
  }
  return maxElts;
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
    } else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionBlock* block = static_cast<TransformerAttentionBlock*>(blocks[i].second.get());
      if(block == nullptr) {
        Global::fatalError("BlockStack::forward: TransformerAttentionBlock pointer is null at index " + Global::intToString(i));
      }
      block->forward(cb, scratch, batchSize, trunk, trunkScratch, mask, maskSum, convWorkspace);
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNBlock* block = static_cast<TransformerFFNBlock*>(blocks[i].second.get());
      if(block == nullptr) {
        Global::fatalError("BlockStack::forward: TransformerFFNBlock pointer is null at index " + Global::intToString(i));
      }
      block->forward(cb, scratch, batchSize, trunk, trunkScratch, mask, maskSum, convWorkspace);
    }
    else {
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
    const SGFMetadataEncoderDesc* desc,
    bool useFP16
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
};

struct Trunk {
  ComputeHandleInternal *handle;
  const std::string name;
  const int modelVersion;
  const int trunkNumChannels;
  const int midNumChannels;
  const int regularNumChannels;
  const int gpoolNumChannels;
  const int trunkNormKind;
  const int nnXLen;
  const int nnYLen;
  const int paddedNNXYLen;

  std::unique_ptr<ConvLayer> initialConv;
  std::unique_ptr<MatmulLayer> initialMatmul;
  std::unique_ptr<SGFMetadataEncoder> sgfMetadataEncoder;
  BlockStack blockStack;
  std::unique_ptr<BatchNormLayer> trunkTipBN;
  std::unique_ptr<RMSNormLayer> trunkTipRMSNorm;
  VkDescriptorSet addChannelBiasDS = VK_NULL_HANDLE;
  VkDescriptorSet addChannelBiasDS2 = VK_NULL_HANDLE;
  VkDescriptorSet addChannelBiasNchwToNhwcDS = VK_NULL_HANDLE;
  VkDescriptorSet addChannelBiasNhwcToNchwDS = VK_NULL_HANDLE;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(
    ComputeHandleInternal *handle_,
    const TrunkDesc* desc,
    int maxBatchSize_,
    int nnXLen_,
    int nnYLen_,
    bool useFP16
  ):
    handle(handle_),
    name(desc->name),
    modelVersion(desc->modelVersion),
    trunkNumChannels(desc->trunkNumChannels),
    midNumChannels(desc->midNumChannels),
    regularNumChannels(desc->regularNumChannels),
    gpoolNumChannels(desc->gpoolNumChannels),
    trunkNormKind(desc->trunkNormKind),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    paddedNNXYLen(handle->paddedNNXYLen),
    blockStack(handle, desc->blocks, desc->numBlocks, trunkNumChannels, nnXLen_, nnYLen_, useFP16)
  {
    checkBufferSize(maxBatchSize_,nnXLen_,nnYLen_,trunkNumChannels);
    checkBufferSize(maxBatchSize_,nnXLen_,nnYLen_,midNumChannels);
    checkBufferSize(maxBatchSize_,nnXLen_,nnYLen_,regularNumChannels);
    checkBufferSize(maxBatchSize_,nnXLen_,nnYLen_,gpoolNumChannels);

    initialConv = std::make_unique<ConvLayer>(handle, &desc->initialConv, nnXLen, nnYLen, useFP16);
    initialMatmul = std::make_unique<MatmulLayer>(handle, &desc->initialMatMul);
    if ( desc->metaEncoderVersion >0) {
      sgfMetadataEncoder = std::make_unique<SGFMetadataEncoder>(handle, &desc->sgfMetadataEncoder, useFP16);
      testAssert(sgfMetadataEncoder->matmul3->outChannels == initialMatmul->outChannels);
    }

    if ( desc->trunkNormKind == TRUNK_NORM_KIND_STANDARD  ){
      trunkTipBN = std::make_unique<BatchNormLayer>(handle, &desc->trunkTipBN, &desc->trunkTipActivation, useFP16);
    } else {
      trunkTipRMSNorm = std::make_unique<RMSNormLayer>(
        handle,
        &desc->trunkTipRMSNorm,
        desc->trunkTipActivation.activation
      );
    }

    VkResult res = VK_SUCCESS;
    addChannelBiasDS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice, handle->pipelines->addChannelBias.descriptorSetLayout, &res
    );
    CHECK_VK_MSG("Allocate Trunk channel-bias descriptor set", res);
    if(sgfMetadataEncoder != nullptr) {
      addChannelBiasDS2 = vk_helper::allocateDescriptorSet(
        handle->vulkanDevice, handle->pipelines->addChannelBias.descriptorSetLayout, &res
      );
      CHECK_VK_MSG("Allocate Trunk metadata channel-bias descriptor set", res);
    }
  }

  ~Trunk() {

  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(
    ComputeHandleInternal* handle,
    int maxBatchSize
  ) const {
    ConvWorkspaceEltsNeeded maxElts = ConvWorkspaceEltsNeeded::getMax(
      initialConv->requiredConvWorkspaceElts(handle,maxBatchSize),
      blockStack.requiredConvWorkspaceElts(handle,maxBatchSize)
    );

    if(trunkTipRMSNorm)
      maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, trunkTipRMSNorm->requiredConvWorkspaceElts(handle,maxBatchSize));
    if(trunkTipBN)
      maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, trunkTipBN->requiredConvWorkspaceElts(handle,maxBatchSize));
    return maxElts;
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
    vkcompute::performAddChannelBiases(
      handle, cb, addChannelBiasDS, addChannelBiasNchwToNhwcDS, addChannelBiasNhwcToNchwDS,
      trunk, trunkScratch.buf, batchSize * trunkNumChannels, trunkNumChannels,
      paddedNNXYLen, convWorkspace, false
    );
    if ( sgfMetadataEncoder != nullptr ) {
      SizedBuf<VulkanBuffer*> sgfEncodedMeta(scratch->allocator, scratch->getBufSizeFloat(sgfMetadataEncoder->matmul3->outChannels));
      sgfMetadataEncoder->forward(cb, batchSize, scratch, inputMeta, sgfEncodedMeta.buf);
      vkcompute::performAddChannelBiases(
        handle, cb, addChannelBiasDS2, addChannelBiasNchwToNhwcDS, addChannelBiasNhwcToNchwDS,
        trunk, sgfEncodedMeta.buf, batchSize * trunkNumChannels, trunkNumChannels,
        handle->paddedNNXYLen, convWorkspace, false
      );
    } else {
      testAssert(inputMeta == NULL);
    }

    blockStack.forward(cb, batchSize, scratch, trunk, trunkScratch.buf, mask, maskSum, convWorkspace, convWorkspace2);

    if (trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
      trunkTipBN->forward(
        cb, batchSize, trunk, mask, trunk,
        convWorkspace, convWorkspace2, nnXLen * nnYLen, nnXLen * nnYLen
      );
    } else {
      trunkTipRMSNorm->forward(cb, batchSize, trunk, trunk, mask, maskSum, convWorkspace, convWorkspace2);
    }
  }

};

struct PolicyHead {
  ComputeHandleInternal *handle;
  const std::string name;
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int paddedNNXYLen;
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
  VkDescriptorSet gpoolNchwToNhwcDS = VK_NULL_HANDLE;
  VkDescriptorSet addChannelBiasDS = VK_NULL_HANDLE;
  VkDescriptorSet addChannelBiasNchwToNhwcDS = VK_NULL_HANDLE;
  VkDescriptorSet addChannelBiasNhwcToNchwDS = VK_NULL_HANDLE;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(
    ComputeHandleInternal *handle,
    const PolicyHeadDesc* desc,
    int nnXLen_,
    int nnYLen_,
    bool useFP16
  ):
    handle(handle),
    name(desc->name),
    modelVersion(desc->modelVersion),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    p1Channels(desc->p1Conv.outChannels),
    g1Channels(desc->g1Conv.outChannels),
    p2Channels(desc->p2Conv.outChannels),
    paddedNNXYLen(handle->paddedNNXYLen)
  {
    p1Conv = std::make_unique<ConvLayer>(handle, &desc->p1Conv, nnXLen, nnYLen, useFP16);
    g1Conv = std::make_unique<ConvLayer>(handle, &desc->g1Conv, nnXLen, nnYLen, useFP16);
    g1BN = std::make_unique<BatchNormLayer>(handle, &desc->g1BN, &desc->g1Activation, useFP16);
    gpoolToBiasMul = std::make_unique<MatmulLayer>(handle, &desc->gpoolToBiasMul);
    p1BN = std::make_unique<BatchNormLayer>(handle, &desc->p1BN, &desc->p1Activation, useFP16);
    p2Conv = std::make_unique<ConvLayer>(handle, &desc->p2Conv, nnXLen, nnYLen, useFP16);
    gpoolToPassMul = std::make_unique<MatmulLayer>(handle, &desc->gpoolToPassMul);
    gpoolToPassBias = std::make_unique<MatBiasLayer>(handle, &desc->gpoolToPassBias, desc->passActivation.activation);
    gpoolToPassMul2 = std::make_unique<MatmulLayer>(handle, &desc->gpoolToPassMul2);
    VkResult res = VK_SUCCESS;
    gpoolDS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice, handle->pipelines->globalPoolingChannelsFp32.descriptorSetLayout, &res
    );
    CHECK_VK_MSG("Allocate PolicyHead pooling descriptor set", res);
    addChannelBiasDS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice, handle->pipelines->addChannelBias.descriptorSetLayout, &res
    );
    CHECK_VK_MSG("Allocate PolicyHead channel-bias descriptor set", res);
  }

  ~PolicyHead() {

  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,p1Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,g1Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,p2Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,g1BN->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,p1BN->requiredConvWorkspaceElts(handle,maxBatchSize));
    if(handle->pipelines->useNHWC) {
      const size_t channelsPadded = handle->getNHWCChannelsPadded(g1Channels);
      const size_t conversionElts = maxBatchSize * static_cast<size_t>(nnXLen * nnYLen) * channelsPadded;
      maxElts.size1 = std::max(maxElts.size1, conversionElts);
    }
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
    g1BN->forward(
      cb, batchSize, gpoolOut.buf, mask, gpoolOut.buf,
      convWorkspace, convWorkspace2, nnXLen * nnYLen, nnXLen * nnYLen
    );
    VkResult res;;
    vkcompute::performGpoolMask(handle, cb, gpoolDS, gpoolNchwToNhwcDS, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, batchSize, g1Channels, paddedNNXYLen, convWorkspace, &res, false);
    gpoolToBiasMul->forward(cb, batchSize, gpoolConcat.buf, gpoolBias.buf);
    vkcompute::performAddChannelBiases(
      handle, cb, addChannelBiasDS, addChannelBiasNchwToNhwcDS, addChannelBiasNhwcToNchwDS,
      p1Out.buf, gpoolBias.buf, p1Channels * batchSize, p1Channels,
      paddedNNXYLen, convWorkspace, false
    );
    p1BN->forward(
      cb, batchSize, p1Out.buf, mask, p1Out.buf,
      convWorkspace, convWorkspace2, nnXLen * nnYLen, nnXLen * nnYLen
    );
    p2Conv->forward(cb, batchSize, p1Out.buf, policy, convWorkspace, convWorkspace2);

    if ( modelVersion >= 15 ) {
      gpoolToPassMul->forward(cb, batchSize, gpoolConcat.buf, p1Pass.buf);
      gpoolToPassBias->forward(cb, batchSize, p1Pass.buf);
      gpoolToPassMul2->forward(cb, batchSize, p1Pass.buf, policyPass);
    } else {
      gpoolToPassMul->forward(cb, batchSize, gpoolConcat.buf, policyPass);
    }
  }

};

struct ValueHead {
  ComputeHandleInternal *handle;
  const std::string name;
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int paddedNNXYLen;
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
  VkDescriptorSet gpoolNchwToNhwcDS = VK_NULL_HANDLE;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(
    ComputeHandleInternal *handle_,
    const ValueHeadDesc* desc,
    int nnXLen_,
    int nnYLen_,
    bool useFP16
  ):
    handle(handle_),
    name(desc->name),
    modelVersion(desc->modelVersion),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    paddedNNXYLen(handle->paddedNNXYLen),
    v1Channels(desc->v1Conv.outChannels),
    v2Channels(desc->v2Mul.outChannels),
    valueChannels(desc->v3Mul.outChannels),
    scoreValueChannels(desc->sv3Mul.outChannels),
    ownershipChannels(desc->vOwnershipConv.outChannels)
  {
    v1Conv = std::make_unique<ConvLayer>(handle, &desc->v1Conv, nnXLen, nnYLen, useFP16);
    v1BN = std::make_unique<BatchNormLayer>(handle, &desc->v1BN, &desc->v1Activation, useFP16);
    v2Mul = std::make_unique<MatmulLayer>(handle, &desc->v2Mul);
    v2Bias = std::make_unique<MatBiasLayer>(handle, &desc->v2Bias, desc->v2Activation.activation);
    v3Mul = std::make_unique<MatmulLayer>(handle, &desc->v3Mul);
    v3Bias = std::make_unique<MatBiasLayer>(handle, &desc->v3Bias, ACTIVATION_IDENTITY);
    sv3Mul = std::make_unique<MatmulLayer>(handle, &desc->sv3Mul);
    sv3Bias = std::make_unique<MatBiasLayer>(handle, &desc->sv3Bias, ACTIVATION_IDENTITY);
    vOwnershipConv = std::make_unique<ConvLayer>(handle, &desc->vOwnershipConv, nnXLen, nnYLen, useFP16);
    assert(!handle->pipelines->valueHeadPoolingChannels.empty());
    VkResult res = VK_SUCCESS;
    gpoolDS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice,
      handle->pipelines->valueHeadPoolingChannels.begin()->second.descriptorSetLayout,
      &res
    );
    CHECK_VK_MSG("Allocate ValueHead pooling descriptor set", res);
  }

  ~ValueHead() {

  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,v1Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,vOwnershipConv->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,v1BN->requiredConvWorkspaceElts(handle,maxBatchSize));
    if(handle->pipelines->useNHWC) {
      const size_t logicalSpatialSize = static_cast<size_t>(handle->nnXLen) * static_cast<size_t>(handle->nnYLen);
      const size_t channelsPadded = static_cast<size_t>(handle->getNHWCChannelsPadded(v1Channels));
      const size_t conversionElts = maxBatchSize * logicalSpatialSize * channelsPadded;
      maxElts.size1 = std::max(maxElts.size1, conversionElts);
    }
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
    v1BN->forward(
      cb, batchSize, v1Out.buf, mask, v1Out.buf,
      convWorkspace, convWorkspace2, nnXLen * nnYLen, nnXLen * nnYLen
    );
    VkResult res;;
    vkcompute::performValueHeadPool(handle, cb, gpoolDS, gpoolNchwToNhwcDS, v1Out.buf, v1Mean.buf, maskSum, convWorkspace, batchSize, v1Channels, paddedNNXYLen, false);

    v2Mul->forward(cb, batchSize, v1Mean.buf, v2Out.buf);
    v2Bias->forward(cb, batchSize, v2Out.buf);
    v3Mul->forward(cb, batchSize, v2Out.buf, value);
    v3Bias->forward(cb, batchSize, value);

    sv3Mul->forward(cb, batchSize, v2Out.buf, scoreValue);
    sv3Bias->forward(cb, batchSize, scoreValue);
    vOwnershipConv->forward(cb, batchSize, v1Out.buf, ownership, convWorkspace, convWorkspace2);
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
  const int paddedNNXYLen;

  std::unique_ptr<Trunk> trunk;
  std::unique_ptr<PolicyHead> policyHead;
  std::unique_ptr<ValueHead> valueHead;
  std::vector<VkCommandBuffer> commandBuffers;
  VkDescriptorSet extractChannel0DS = VK_NULL_HANDLE;
  VkDescriptorSet extractChannel0NCHWToNHWCDS = VK_NULL_HANDLE;
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
    nnYLen(nnYLen_),
    paddedNNXYLen(handle->paddedNNXYLen)
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

    bool useFP16 = handle->usingFP16Storage;
    trunk = std::make_unique<Trunk>(handle, &desc.trunk, maxBatchSize, nnXLen, nnYLen, useFP16);
    policyHead = std::make_unique<PolicyHead>(handle, &desc.policyHead, nnXLen, nnYLen, useFP16);
    valueHead = std::make_unique<ValueHead>(handle, &desc.valueHead, nnXLen, nnYLen, useFP16);

    VkResult res = VK_SUCCESS;
    extractChannel0DS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice, handle->pipelines->extractChannel0Fp32.descriptorSetLayout, &res
    );
    CHECK_VK_MSG("Allocate model ExtractChannel0 descriptor set", res);
    const LocalDim maskSumDim = {
      handle->tuneParams.gPool.XYSTRIDE,
      1,
      std::min(
        handle->tuneParams.gPool.BATCHSTRIDE,
        static_cast<int>(vk_helper::powerOf2ify(maxBatchSize))
      )
    };
    const Pipeline& maskSumPipeline = handle->pipelines->sumChannels.at(maskSumDim);
    computeMaskSumDS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice, maskSumPipeline.descriptorSetLayout, &res
    );
    CHECK_VK_MSG("Allocate model mask sum descriptor set", res);
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
    if(handle->pipelines->useNHWC) {
      const size_t logicalSpatialSize = static_cast<size_t>(handle->nnXLen) * static_cast<size_t>(handle->nnYLen);
      const size_t inputStagingElts = static_cast<size_t>(maxBatchSize) * logicalSpatialSize *
        static_cast<size_t>(handle->getNHWCChannelsPadded(numInputChannels));
      maxElts = ConvWorkspaceEltsNeeded::getMax(
        maxElts,
        ConvWorkspaceEltsNeeded(inputStagingElts, inputStagingElts)
      );
    }
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, trunk->requiredConvWorkspaceElts(handle, maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, policyHead->requiredConvWorkspaceElts(handle, maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts, valueHead->requiredConvWorkspaceElts(handle, maxBatchSize));
    return maxElts;
  }

  void forward(
    VkCommandBuffer forwardCB,
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
    const bool useNHWC = handle->pipelines->useNHWC;
    vkcompute::extractChannel0(
      handle,
      handle->vulkanDevice,
      &handle->pipelines->extractChannel0Fp32,
      forwardCB,
      extractChannel0DS,
      input,
      mask,
      convWorkspace,
      batchSize,
      numInputChannels,
      handle->paddedNNXYLen,
      handle->paddedNNXYLen,
      nnXLen * nnYLen,
      useNHWC ? handle->getNHWCChannelsPadded(numInputChannels) : numInputChannels,
      useNHWC,
      false
    );
    vkcompute::computeMaskSums(handle, forwardCB, computeMaskSumDS, batchSize, mask, maskSum, false);
    trunk->forward(forwardCB, batchSize, scratch, input, inputGlobal, inputMeta, trunkBuf,  mask, maskSum, convWorkspace, convWorkspace2);
    policyHead->forward(forwardCB, batchSize, scratch, trunkBuf, mask, maskSum, policyPass, policy, convWorkspace, convWorkspace2);
    valueHead->forward(forwardCB, batchSize, scratch, trunkBuf, mask, maskSum, value, scoreValue, ownership, convWorkspace, convWorkspace2);
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
  if(loadedModel->modelDesc.modelVersion > NNModelVersion::latestModelVersionImplemented)
    throw StringError("Vulkan backend does not support this model version");
  // if(useFP16Mode == enabled_t::True && logger != nullptr)
    // logger->write("Vulkan backend supports FP32 execution only; ignoring useFP16=true");

  std::string tunerFile;
  if(cfg.contains("vulkanTunerFile"))
    tunerFile = cfg.getString("vulkanTunerFile");
  VulkanTuner::ModelInfoForTuning modelInfo = VulkanTuner::ModelInfoForTuning::ofDesc(loadedModel->modelDesc);

  if(logger != nullptr) {
    logger->write("Create Vulkan Compute Context with GPUs: ");
    for(size_t i = 0; i<gpuIdxs.size(); i++) {
      logger->write("  GPU Index " + Global::intToString(gpuIdxs[i]));
    }
  }

  enabled_t useFP16 = useFP16Mode;
  enabled_t useNHWC = enabled_t::False;

  return new ComputeContext(
    nnXLen,
    nnYLen,
    useFP16,
    useNHWC,
    std::vector<uint32_t>(gpuIdxs.begin(), gpuIdxs.end()),
    logger,
    tunerFile,
    homeDataDirOverride,
    &modelInfo,
    &loadedModel->modelDesc
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
    nullptr
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

  VulkanBuffer* uploadBuffer;
  VkDeviceSize inputUploadOffset;
  VkDeviceSize inputGlobalUploadOffset;
  VkDeviceSize inputMetaUploadOffset;

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

  VulkanBuffer* readbackBuffer;
  VkDeviceSize policyPassReadbackOffset;
  VkDeviceSize policyReadbackOffset;
  VkDeviceSize valueReadbackOffset;
  VkDeviceSize scoreValueReadbackOffset;
  VkDeviceSize ownershipReadbackOffset;

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
    size_t batchXYElts = (size_t)m.maxBatchSize * m.paddedNNXYLen;
    size_t batchElts = (size_t)m.maxBatchSize;

    bool useFP16 = handle->usingFP16Storage;

    const int inputChannels = handle->pipelines->useNHWC
      ? handle->getNHWCChannelsPadded(m.numInputChannels)
      : m.numInputChannels;
    const int policyChannels = handle->pipelines->useNHWC
      ? handle->getNHWCChannelsPadded(m.policyHead->p2Channels)
      : m.policyHead->p2Channels;
    const int ownershipChannels = handle->pipelines->useNHWC
      ? handle->getNHWCChannelsPadded(m.valueHead->ownershipChannels)
      : m.valueHead->ownershipChannels;

    inputElts = inputChannels * batchXYElts;
    inputGlobalElts = m.numInputGlobalChannels * batchElts;
    inputMetaElts = m.numInputMetaChannels * batchElts;

    const size_t spatialDtypeSize = useFP16 ? sizeof(half_t) : sizeof(float);
    VkResult res = VK_SUCCESS;

    const auto reserveTransferRange = [](VkDeviceSize& totalSize, VkDeviceSize size) {
      totalSize = (totalSize + 3) & ~VkDeviceSize(3);
      const VkDeviceSize offset = totalSize;
      totalSize += size;
      return offset;
    };
    VkDeviceSize uploadBytes = 0;
    inputUploadOffset = reserveTransferRange(uploadBytes, spatialDtypeSize * inputElts);
    inputGlobalUploadOffset = reserveTransferRange(uploadBytes, sizeof(float) * inputGlobalElts);
    inputMetaUploadOffset = reserveTransferRange(uploadBytes, sizeof(float) * inputMetaElts);
    uploadBuffer = vk_helper::createStagingBuffer(handle->vulkanDevice, static_cast<size_t>(uploadBytes), &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create persistent upload buffer", res);

    input = vk_helper::createDeviceBuffer(handle->vulkanDevice, spatialDtypeSize * inputElts, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create input buffer", res);
    inputGlobal = vk_helper::createDeviceBuffer(handle->vulkanDevice, sizeof(float) * inputGlobalElts, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create input global buffer", res);
    if(m.numInputMetaChannels > 0) {
      inputMeta = vk_helper::createDeviceBuffer(handle->vulkanDevice, sizeof(float) * inputMetaElts, false, &res);
    }
    else {
      inputMeta = NULL;
    }

    mask = vk_helper::createDeviceBuffer(handle->vulkanDevice, batchXYElts * spatialDtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create mask buffer", res);
    maskSum = vk_helper::createDeviceBuffer(handle->vulkanDevice, batchElts * sizeof(float), false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create mask sum buffer", res);
    const int trunkChannels = handle->pipelines->useNHWC
      ? handle->getNHWCChannelsPadded(m.trunk->trunkNumChannels)
      : m.trunk->trunkNumChannels;
    trunk = vk_helper::createDeviceBuffer(handle->vulkanDevice, trunkChannels * batchXYElts * spatialDtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create trunk buffer", res);

    if(m.modelVersion >= 17)
      testAssert(m.policyHead->p2Channels == 2 || m.policyHead->p2Channels == 4);
    else if(m.modelVersion >= 16)
      testAssert(m.policyHead->p2Channels == 4);
    else if(m.modelVersion >= 12)
      testAssert(m.policyHead->p2Channels == 2);
    else
      testAssert(m.policyHead->p2Channels == 1);

    policyPassElts = m.policyHead->p2Channels * batchElts;
    policyPass = vk_helper::createDeviceBuffer(handle->vulkanDevice, policyPassElts * sizeof(float), false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create policy pass buffer", res);
    policyElts = policyChannels * batchXYElts;
    policy = vk_helper::createDeviceBuffer(handle->vulkanDevice, policyElts * spatialDtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create policy buffer", res);

    valueElts = m.valueHead->valueChannels * batchElts;
    value = vk_helper::createDeviceBuffer(handle->vulkanDevice, valueElts * sizeof(float), false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create value buffer", res);

    scoreValueElts = m.valueHead->scoreValueChannels * batchElts;
    scoreValue = vk_helper::createDeviceBuffer(handle->vulkanDevice, scoreValueElts * sizeof(float), false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create score value buffer", res);

    ownershipElts = ownershipChannels * batchXYElts;
    ownership = vk_helper::createDeviceBuffer(handle->vulkanDevice, ownershipElts * spatialDtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create ownership buffer", res);

    VkDeviceSize readbackBytes = 0;
    policyPassReadbackOffset = reserveTransferRange(readbackBytes, policyPassElts * sizeof(float));
    policyReadbackOffset = reserveTransferRange(readbackBytes, policyElts * spatialDtypeSize);
    valueReadbackOffset = reserveTransferRange(readbackBytes, valueElts * sizeof(float));
    scoreValueReadbackOffset = reserveTransferRange(readbackBytes, scoreValueElts * sizeof(float));
    ownershipReadbackOffset = reserveTransferRange(readbackBytes, ownershipElts * spatialDtypeSize);
    readbackBuffer = vk_helper::createReadbackBuffer(handle->vulkanDevice, static_cast<size_t>(readbackBytes), &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create persistent readback buffer", res);

    // TODO: Implement workspace allocation when winograd or other conv algorithms are added.
    ConvWorkspaceEltsNeeded convWorkspaceElts = m.requiredConvWorkspaceElts(handle);

    std::printf("Conv workspace elts needed: size1=%zu, size2=%zu\n", convWorkspaceElts.size1, convWorkspaceElts.size2);

    convWorkspace = vk_helper::createDeviceBuffer(handle->vulkanDevice, convWorkspaceElts.size1 * spatialDtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create conv workspace buffer", res);
    convWorkspace2 = vk_helper::createDeviceBuffer(handle->vulkanDevice, convWorkspaceElts.size2 * spatialDtypeSize, false, &res);
    CHECK_VK_MSG("[Buffers::Buffers] Create conv workspace2 buffer", res);
  }

  ~Buffers() {
    vk_helper::releaseVulkanBuffer(uploadBuffer->device, uploadBuffer);
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
    vk_helper::releaseVulkanBuffer(readbackBuffer->device, readbackBuffer);
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
  const bool inputsUseNHWC;
  const bool inputUsingNHWC;

  static bool shouldUseNHWC(const ComputeContext* context, int gpuIdx) {
    const uint32_t normalizedGpuIdx = gpuIdx < 0 ? 0u : static_cast<uint32_t>(gpuIdx);
    return context->pipelinesPerDev.at(normalizedGpuIdx)->useNHWC;
  }

  ComputeHandle(
    ComputeContext* context,
    const LoadedModel* loadedModel,
    int maxBatchSize,
    int gpuIdx,
    bool inputsUseNHWC_
  ):
    handle( std::make_unique<ComputeHandleInternal>(
      context,
      gpuIdx,
      shouldUseNHWC(context, gpuIdx),
      shouldUseNHWC(context, gpuIdx)
    )),
    nnXLen(context->nnXLen),
    nnYLen(context->nnYLen),
    policySize(NNPos::getPolicySize(context->nnXLen,context->nnYLen)),
    inputsUseNHWC(inputsUseNHWC_),
    inputUsingNHWC(
      shouldUseNHWC(context, gpuIdx)
    ),
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
    if(inputsUseNHWC_ && !inputUsingNHWC)
      throw StringError("Vulkan NHWC input was requested, but the selected device has no supported NHWC cooperative-matrix path");
    scratch = std::make_unique<ScratchBuffers>(
      handle.get(),
      model->maxBatchSize
    );

    const VkDeviceSize allocatedGpuMemory = vk_helper::getTotalAllocatedGpuMemory(handle->vulkanDevice);
    const std::string memoryMessage =
      "Vulkan GPU memory allocated after ComputeHandle initialization: " +
      Global::uint64ToString(static_cast<uint64_t>(allocatedGpuMemory)) + " bytes (" +
      Global::doubleToString(static_cast<double>(allocatedGpuMemory) / (1024.0 * 1024.0)) + " MiB)";
    if(context->logger != nullptr)
      context->logger->write(memoryMessage);
    if(context->logger == nullptr || (!context->logger->isLoggingToStdout() && !context->logger->isLoggingToStderr()))
      std::cerr << memoryMessage << std::endl;

    // // Buffers* ptr = new Buffers(handle.get(), *model);
    // // buffers = std::make_unique<Buffers>(handle.get(), *model);
    // buffers.reset(ptr);
  }

  ~ComputeHandle() {}
  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

ComputeHandleInternal::ComputeHandleInternal(
  ComputeContext* ctx,
  int gpuIdx,
  bool inputsUseNHWC,
  bool useNHWC
):
  context(ctx),
  vulkanDevice(ctx->vulkanContext->findGpuExn(gpuIdx)),
  pipelines(ctx->pipelinesPerDev.at(vulkanDevice->info.deviceId)),
  tuneParams(ctx->tuneParamsPerDev.at(vulkanDevice->info.deviceId)),
  qHeadDim(ctx->transformerHeadDims.first),
  vHeadDim(ctx->transformerHeadDims.second)
{
  this->queue = this->vulkanDevice->queue;
  this->device = this->vulkanDevice->device;
  this->nnXLen = ctx->nnXLen;
  this->nnYLen = ctx->nnYLen;

  const VulkanParams& vulkanParams = tuneParams.vulkan;
  if(ctx->usingFP16Mode == enabled_t::True) {
    usingFP16Storage = vulkanParams.canUseFP16Storage && vulkanParams.canUseFP16Compute;
    usingFP16Compute = vulkanParams.canUseFP16Compute;
  }
  else if(ctx->usingFP16Mode == enabled_t::Auto) {
    usingFP16Storage =
      vulkanParams.canUseFP16Storage &&
      vulkanParams.canUseFP16Compute &&
      vulkanParams.shouldUseFP16Storage;
    usingFP16Compute = vulkanParams.canUseFP16Compute && vulkanParams.shouldUseFP16Compute;
  }

  if(usingFP16Storage || usingFP16Compute) {
    const std::string message =
      "Vulkan FP16 enabled (storage: " + std::string(usingFP16Storage ? "Yes" : "No") +
      ", compute: " + std::string(usingFP16Compute ? "Yes" : "No") + ")";
    if(ctx->logger != nullptr)
      ctx->logger->write(message);
    if(ctx->logger == nullptr || (!ctx->logger->isLoggingToStdout() && !ctx->logger->isLoggingToStderr()))
      std::cerr << message << std::endl;
  }

  const bool usingFP16TransformerDualGemmSwiGLU =
    tuneParams.vulkan.canUseFP16Storage &&
    tuneParams.vulkan.canUseFP16Compute &&
    tuneParams.vulkan.canUseCooperativeMatrix &&
    tuneParams.vulkan.shouldUseFP16Storage &&
    tuneParams.vulkan.shouldUseFP16Compute &&
    tuneParams.vulkan.shouldUseTransformerDualGemmSwiGLU &&
    tuneParams.transformerDualGemmSwiGLU.isValid() &&
    usingFP16Storage;
  const bool usingFP16NHWCCooperativeMatrix =
    usingFP16Storage &&
    tuneParams.vulkan.canUseCooperativeMatrix &&
    tuneParams.vulkan.shouldUseCooperativeMatrix &&
    tuneParams.vulkan.shouldUseFP16Compute &&
    tuneParams.hgemmCooperativeMatrixNHWC.isValid();
  const bool usingFP16NCHWCooperativeMatrix =
    usingFP16Storage &&
    tuneParams.vulkan.canUseCooperativeMatrix &&
    tuneParams.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
    tuneParams.vulkan.shouldUseFP16Compute &&
    tuneParams.hgemmCooperativeMatrixNCHW.isValid();
  if(usingFP16NHWCCooperativeMatrix || usingFP16NCHWCooperativeMatrix || usingFP16TransformerDualGemmSwiGLU) {
    int spatialAlignment = 1;
    if(usingFP16NHWCCooperativeMatrix) {
      const auto addNhwcSpatialAlignment = [&](const auto& params) {
        if(params.MWG > 0)
          spatialAlignment = std::lcm(spatialAlignment, params.MWG);
      };
      addNhwcSpatialAlignment(tuneParams.hgemmCooperativeMatrixNHWC);
    }
    if(usingFP16NCHWCooperativeMatrix)
      spatialAlignment = std::lcm(
        spatialAlignment, tuneParams.hgemmCooperativeMatrixNCHW.getRequiredSpatialAlignment()
      );
    if(usingFP16TransformerDualGemmSwiGLU)
      spatialAlignment = std::lcm(spatialAlignment, tuneParams.transformerDualGemmSwiGLU.getRequiredSpatialAlignment());
    this->paddedNNXYLen = vk_helper::roundUpToMultipleInt(nnXLen * nnYLen, spatialAlignment);
  } else {
    this->paddedNNXYLen = nnXLen * nnYLen;
  }

  #ifdef VK_BENCHMARK
  VkQueryPoolCreateInfo qpCI = {};
  qpCI.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpCI.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpCI.queryCount = 4096;
  VkResult res = vkCreateQueryPool(this->device, &qpCI, nullptr, &queryPool);
  CHECK_VK_MSG("Create query pool", res);
  benchmarkDispatchInfos.reserve(4096);
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
  if(inputsUseNHWC && !ComputeHandle::shouldUseNHWC(context, gpuIdxForThisThread))
    throw StringError("Vulkan NHWC input requires an active cooperative-matrix NHWC path");
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
  return handle->handle->usingFP16Storage;
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
  half_t* userInputBufferHalf; //Host pointer
  float* userInputGlobalBuffer; //Host pointer
  float* userInputMetaBuffer; //Host pointer

  float* policyPassResults; //Host pointer
  float* policyResults; //Host pointer
  half_t* policyResultsHalf; //Host pointer
  float* valueResults; //Host pointer
  float* scoreValueResults; //Host pointer
  float* ownershipResults; //Host pointer
  half_t* ownershipResultsHalf; //Host pointer
  int halfSpatialCapacity;
  int spatialCapacity;
  int spatialSize;
  int inputStorageChannels;
  int policyStorageChannels;
  int ownershipStorageChannels;

  void ensureStorageCapacity(
    int inputChannelsPadded,
    int policyChannelsPadded,
    int ownershipChannelsPadded,
    int spatialCapacityNeeded,
    int halfSpatialCapacityNeeded
  ) {
    bool inputStorageChanged = false;
    if(inputChannelsPadded > inputStorageChannels || spatialCapacityNeeded > spatialCapacity) {
      delete[] userInputBuffer;
      inputStorageChannels = std::max(inputStorageChannels, inputChannelsPadded);
      spatialCapacity = std::max(spatialCapacity, spatialCapacityNeeded);
      inputStorageChanged = true;
      userInputBuffer = new float[
        static_cast<size_t>(inputStorageChannels) * maxBatchSize * spatialCapacity
      ];
      userInputBufferElts = static_cast<size_t>(inputStorageChannels) * maxBatchSize * spatialCapacity;
    }
    if(
      halfSpatialCapacityNeeded > halfSpatialCapacity ||
      inputStorageChanged ||
      policyChannelsPadded > policyStorageChannels ||
      ownershipChannelsPadded > ownershipStorageChannels
    ) {
      delete[] userInputBufferHalf;
      delete[] policyResultsHalf;
      delete[] ownershipResultsHalf;
      halfSpatialCapacity = std::max(halfSpatialCapacity, halfSpatialCapacityNeeded);
      policyStorageChannels = std::max(policyStorageChannels, policyChannelsPadded);
      ownershipStorageChannels = std::max(ownershipStorageChannels, ownershipChannelsPadded);
      userInputBufferHalf = new half_t[
        static_cast<size_t>(inputStorageChannels) * maxBatchSize * halfSpatialCapacity
      ];
      policyResultsHalf = new half_t[
        static_cast<size_t>(maxBatchSize) * policyStorageChannels * halfSpatialCapacity
      ];
      ownershipResultsHalf = new half_t[
        static_cast<size_t>(maxBatchSize) * halfSpatialCapacity * ownershipStorageChannels
      ];
    }
  }

  void ensureHalfSpatialCapacity(
    int inputChannels, int policyChannels, int ownershipChannels, int paddedNNXYLen,
    bool useNHWC
  ) {
    const int inputChannelsPadded = useNHWC ? inputStorageChannels : inputChannels;
    const int policyChannelsPadded = useNHWC ? policyStorageChannels : policyChannels;
    const int ownershipChannelsPadded = useNHWC ? ownershipStorageChannels : ownershipChannels;
    ensureStorageCapacity(
      inputChannelsPadded, policyChannelsPadded, ownershipChannelsPadded,
      useNHWC ? paddedNNXYLen : spatialCapacity, paddedNNXYLen
    );
  }

  InputBuffers(
    const LoadedModel* loadedModel,
    int maxBatchSize_,
    int nnXLen,
    int nnYLen
  ) {
    // Bytes size will not be computed because of fp16.
    const ModelDesc& m = loadedModel->modelDesc;
    maxBatchSize = maxBatchSize_;
    spatialSize = nnXLen * nnYLen;
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

    inputStorageChannels = vk_helper::roundUpToMultipleInt(m.numInputChannels, 4);
    policyStorageChannels = vk_helper::roundUpToMultipleInt(m.numPolicyChannels, 4);
    ownershipStorageChannels = vk_helper::roundUpToMultipleInt(m.numOwnershipChannels, 4);
    spatialCapacity = spatialSize;
    userInputBufferElts = static_cast<size_t>( inputStorageChannels ) * maxBatchSize * spatialSize;
    userInputGlobalBufferElts = static_cast<size_t>( m.numInputGlobalChannels ) * maxBatchSize;
    userInputMetaBufferElts = static_cast<size_t>( m.numInputMetaChannels ) * maxBatchSize;
    policyPassResultBufferElts = static_cast<size_t>( maxBatchSize ) * m.numPolicyChannels;
    policyResultBufferElts = static_cast<size_t>( maxBatchSize ) * m.numPolicyChannels * nnXLen * nnYLen;
    valueResultBufferElts = static_cast<size_t>( maxBatchSize ) * m.numValueChannels;
    scoreValueResultBufferElts = static_cast<size_t>( maxBatchSize ) * m.numScoreValueChannels;
    ownershipResultBufferElts = static_cast<size_t>( maxBatchSize ) * m.numOwnershipChannels * nnXLen * nnYLen;

    // userInputBuffer = new float[userInputBufferElts];
    userInputBuffer = new float[(size_t)inputStorageChannels * maxBatchSize * spatialSize];
    halfSpatialCapacity = vk_helper::roundUpToMultipleInt(nnXLen * nnYLen, 16);
    userInputBufferHalf = new half_t[(size_t)inputStorageChannels * maxBatchSize * halfSpatialCapacity];
    userInputGlobalBuffer = new float[(size_t)m.numInputGlobalChannels * maxBatchSize];
    if(m.numInputMetaChannels > 0)
      userInputMetaBuffer = new float[(size_t)m.numInputMetaChannels * maxBatchSize];
    else
      userInputMetaBuffer = nullptr;

    policyPassResults = new float[(size_t)maxBatchSize * m.numPolicyChannels];
    policyResults = new float[(size_t)maxBatchSize * m.numPolicyChannels * nnXLen * nnYLen];
    policyResultsHalf = new half_t[(size_t)maxBatchSize * policyStorageChannels * halfSpatialCapacity];
    valueResults = new float[(size_t)maxBatchSize * m.numValueChannels];

    scoreValueResults = new float[(size_t)maxBatchSize * m.numScoreValueChannels];
    ownershipResults = new float[(size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels];
    ownershipResultsHalf = new half_t[(size_t)maxBatchSize * halfSpatialCapacity * ownershipStorageChannels];
    // userInputGlobalBuffer = new float[userInputGlobalBufferElts];
    // if ( m.numInputMetaChannels > 0 ) {
    //   userInputMetaBuffer = new float[userInputMetaBufferElts];
    // } else {
    //   userInputMetaBuffer = nullptr;
    // }

    // policyPassResults = new float[ static_cast<size_t>( m.numPolicyChannels * maxBatchSize ) ];
    // policyResults = new float[ static_cast<size_t>( m.numPolicyChannels * nnXLen * nnYLen * maxBatchSize ) ];
    // valueResults = new float[ static_cast<size_t>( m.numValueChannels * maxBatchSize ) ];
    // scoreValueResults = new float[ static_cast<size_t>( m.numScoreValueChannels * maxBatchSize ) ];
    // ownershipResults = new float[ static_cast<size_t>( m.numOwnershipChannels * nnXLen * nnYLen * maxBatchSize ) ];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputBufferHalf;
    delete[] userInputGlobalBuffer;
    if ( userInputMetaBuffer != nullptr ) {
      delete[] userInputMetaBuffer;
    }
    delete[] policyPassResults;
    delete[] policyResults;
    delete[] policyResultsHalf;
    delete[] valueResults;
    delete[] scoreValueResults;
    delete[] ownershipResults;
    delete[] ownershipResultsHalf;
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

static void copyNCHWInputToNHWCWithSymmetry(
  const float* src,
  float* dst,
  int hSize,
  int wSize,
  int cSize,
  int symmetry
) {
  const int spatialSize = hSize * wSize;
  bool transpose = (symmetry & 0x4) != 0 && hSize == wSize;
  bool flipX = (symmetry & 0x2) != 0;
  bool flipY = (symmetry & 0x1) != 0;
  if(transpose)
    std::swap(flipX, flipY);

  const int hStride = wSize;
  const int wStride = 1;
  int hBaseNew = 0;
  int hStrideNew = hStride;
  int wBaseNew = 0;
  int wStrideNew = wStride;

  if(flipY) {
    hBaseNew = (hSize - 1) * hStrideNew;
    hStrideNew = -hStrideNew;
  }
  if(flipX) {
    wBaseNew = (wSize - 1) * wStrideNew;
    wStrideNew = -wStrideNew;
  }
  if(transpose)
    std::swap(hStrideNew, wStrideNew);

  for(int c = 0; c < cSize; c++) {
    const int sourceChannelBase = c * spatialSize;
    for(int h = 0; h < hSize; h++) {
      const int sourceRowBase = sourceChannelBase + h * hStride;
      const int destinationRowBase = hBaseNew + h * hStrideNew;
      for(int w = 0; w < wSize; w++) {
        const int sourceIndex = sourceRowBase + w * wStride;
        const int destinationXY = destinationRowBase + wBaseNew + w * wStrideNew;
        dst[destinationXY * cSize + c] = src[sourceIndex];
      }
    }
  }
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
  const int nnXYLen = nnXLen * nnYLen;
  const int paddedNNXYLen = computeHandle->handle->paddedNNXYLen;
  const int modelVersion = computeHandle->model->modelVersion;
  // globalBatchCount = batchSize;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = static_cast<int>(inputBuffers->singleInputMetaElts);
  assert(numSpatialFeatures == computeHandle->model->numInputChannels);
    assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);
  const int numPolicyChannels = computeHandle->model->numPolicyChannels;
  ComputeHandleInternal* handle = computeHandle->handle.get();

  const int inputChannelStride = computeHandle->inputUsingNHWC
    ? handle->getNHWCChannelsPadded(numSpatialFeatures)
    : numSpatialFeatures;
  const int policyChannelStride = computeHandle->inputUsingNHWC
    ? handle->getNHWCChannelsPadded(numPolicyChannels)
    : numPolicyChannels;
  const int ownershipChannelStride = computeHandle->inputUsingNHWC
    ? handle->getNHWCChannelsPadded(computeHandle->model->numOwnershipChannels)
    : computeHandle->model->numOwnershipChannels;
  const int inputSpatialStride = computeHandle->inputUsingNHWC
    ? paddedNNXYLen
    : nnXYLen;
  inputBuffers->ensureStorageCapacity(
    inputChannelStride, policyChannelStride, ownershipChannelStride,
    inputSpatialStride, paddedNNXYLen
  );
  const size_t inputRowElts = static_cast<size_t>(inputChannelStride) * inputSpatialStride;
  const size_t tightInputRowElts = static_cast<size_t>(numSpatialFeatures) * nnXLen * nnYLen;
  std::vector<float> tightSpatialInput;
  if(computeHandle->inputUsingNHWC)
    tightSpatialInput.resize(tightInputRowElts);

  for (int nIdx = 0 ; nIdx < batchSize ; ++nIdx) {
    float* rowSpatialInput = inputBuffers->userInputBuffer + inputRowElts * nIdx;
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
    if(computeHandle->inputUsingNHWC)
      std::fill(rowSpatialInput, rowSpatialInput + inputRowElts, 0.0f);
    if(computeHandle->inputUsingNHWC) {
      if(computeHandle->inputsUseNHWC) {
        SymmetryHelpers::copyInputsWithSymmetry(
          rowSpatial, tightSpatialInput.data(), 1, nnYLen, nnXLen, numSpatialFeatures,
          true, inputBufs[nIdx]->symmetry
        );
      } else {
        copyNCHWInputToNHWCWithSymmetry(
          rowSpatial, tightSpatialInput.data(), nnYLen, nnXLen, numSpatialFeatures,
          inputBufs[nIdx]->symmetry
        );
      }
      for(int xy = 0; xy < nnXYLen; xy++) {
        std::copy(
          tightSpatialInput.data() + static_cast<size_t>(xy) * numSpatialFeatures,
          tightSpatialInput.data() + static_cast<size_t>(xy + 1) * numSpatialFeatures,
          rowSpatialInput + static_cast<size_t>(xy) * inputChannelStride
        );
      }
    } else {
      SymmetryHelpers::copyInputsWithSymmetry(
        rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures,
        false, inputBufs[nIdx]->symmetry
      );
    }
  }
  Buffers* buffers = computeHandle->buffers.get();

  // assert(inputBuffers->userInputBufferElts == buffers->inputElts);
  assert(inputBuffers->userInputGlobalBufferElts == buffers->inputGlobalElts);
  assert(inputBuffers->userInputMetaBufferElts == buffers->inputMetaElts);
  // assert(inputBuffers->policyResultBufferElts == buffers->policyElts);
  assert(inputBuffers->valueResultBufferElts == buffers->valueElts);
  assert(inputBuffers->singlePolicyPassResultElts == numPolicyChannels);
  assert(inputBuffers->singlePolicyResultElts == numPolicyChannels * nnXLen * nnYLen);
  assert(inputBuffers->singlePolicyResultElts + inputBuffers->singlePolicyPassResultElts == computeHandle->policySize * numPolicyChannels);
  assert(inputBuffers->scoreValueResultBufferElts == buffers->scoreValueElts);
  assert(inputBuffers->ownershipResultBufferElts == buffers->ownershipElts);
  assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);

  bool useFP16Storage = handle->usingFP16Storage;
  if(useFP16Storage) {
    inputBuffers->ensureHalfSpatialCapacity(
      numSpatialFeatures, numPolicyChannels, computeHandle->model->numOwnershipChannels, paddedNNXYLen,
      computeHandle->inputUsingNHWC
    );
  }

  VkResult res = VK_ERROR_UNKNOWN;
  const void* spatialInputData = inputBuffers->userInputBuffer;
  VkDeviceSize spatialInputBytes = static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputRowElts);

  if ( useFP16Storage ) {
    size_t paddedInputElts = static_cast<size_t>(inputChannelStride) * paddedNNXYLen * batchSize;
    size_t totalChannels = static_cast<size_t>(numSpatialFeatures * batchSize);
    // Convert float to half with spatial and channel padding.
    if(computeHandle->inputUsingNHWC) {
      for(int n = 0; n < batchSize; n++) {
        for(int xy = 0; xy < paddedNNXYLen; xy++) {
          for(int c = 0; c < inputChannelStride; c++) {
            const size_t dst = (static_cast<size_t>(n) * paddedNNXYLen + xy) * inputChannelStride + c;
            const bool valid = xy < nnXYLen && c < numSpatialFeatures;
            inputBuffers->userInputBufferHalf[dst] = valid
              ? half_float::half_cast<half_t>(inputBuffers->userInputBuffer[(static_cast<size_t>(n) * inputSpatialStride + xy) * inputChannelStride + c])
              : half_float::half_cast<half_t>(0.0f);
          }
        }
      }
    } else if(paddedNNXYLen == nnXYLen) {
      for ( size_t i = 0 ; i < totalChannels * nnXYLen ; ++i ) {
        inputBuffers->userInputBufferHalf[i] = half_float::half_cast<half_t>(inputBuffers->userInputBuffer[i]);
      }
    } else {
      for (size_t c = 0 ; c < totalChannels ; ++c ) {
        for ( int xy = 0 ; xy < nnXYLen ; xy++) {
          inputBuffers->userInputBufferHalf[c * paddedNNXYLen + xy] = half_float::half_cast<half_t>(inputBuffers->userInputBuffer[c * nnXYLen + xy]);
        }
        for(int xy = nnXYLen; xy < paddedNNXYLen; xy++) {
          inputBuffers->userInputBufferHalf[c * paddedNNXYLen + xy] = half_float::half_cast<half_t>(0.0f);
        }
      }
    }

    spatialInputData = inputBuffers->userInputBufferHalf;
    spatialInputBytes = static_cast<VkDeviceSize>(paddedInputElts * sizeof(half_t));
  } else if(computeHandle->inputUsingNHWC) {
    throw StringError("Vulkan NHWC input requires FP16 storage");
  } else {
    if ( paddedNNXYLen != nnXYLen ) {
      ASSERT_UNREACHABLE;
    }
  }

  {
    vk_helper::copyHostToStagingBuffer(
      handle->vulkanDevice,
      spatialInputData,
      buffers->uploadBuffer,
      buffers->inputUploadOffset,
      spatialInputBytes,
      &res
    );
    CHECK_VK_MSG("Copy spatial input to persistent upload buffer", res);

    const VkDeviceSize inputGlobalBytes =
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleInputGlobalElts);
    vk_helper::copyHostToStagingBuffer(
      handle->vulkanDevice,
      inputBuffers->userInputGlobalBuffer,
      buffers->uploadBuffer,
      buffers->inputGlobalUploadOffset,
      inputGlobalBytes,
      &res
    );
    CHECK_VK_MSG("Copy global input to persistent upload buffer", res);

    VkDeviceSize inputMetaBytes = 0;
    if ( numMetaFeatures > 0 ) {
      inputMetaBytes = static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleInputMetaElts);
      vk_helper::copyHostToStagingBuffer(
        handle->vulkanDevice,
        inputBuffers->userInputMetaBuffer,
        buffers->uploadBuffer,
        buffers->inputMetaUploadOffset,
        inputMetaBytes,
        &res
      );
      CHECK_VK_MSG("Copy metadata input to persistent upload buffer", res);
    }


    const VkDeviceSize policyPassBytes =
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singlePolicyPassResultElts);
    const int policyStorageChannels = computeHandle->inputUsingNHWC
      ? handle->getNHWCChannelsPadded(numPolicyChannels)
      : numPolicyChannels;
    const size_t paddedPolicyElts = static_cast<size_t>(policyStorageChannels) * paddedNNXYLen * batchSize;
    const VkDeviceSize policyBytes = static_cast<VkDeviceSize>(
      paddedPolicyElts * (useFP16Storage ? sizeof(half_t) : sizeof(float))
    );
    const VkDeviceSize valueBytes =
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleValueResultElts);
    const VkDeviceSize scoreValueBytes =
      static_cast<VkDeviceSize>(sizeof(float) * batchSize * inputBuffers->singleScoreValueResultElts);
    const int ownershipChannels = computeHandle->model->numOwnershipChannels;
    const int ownershipStorageChannels = computeHandle->inputUsingNHWC
      ? handle->getNHWCChannelsPadded(ownershipChannels)
      : ownershipChannels;
    const size_t paddedOwnershipElts =
      static_cast<size_t>(ownershipStorageChannels) * paddedNNXYLen * batchSize;
    const VkDeviceSize ownershipBytes = static_cast<VkDeviceSize>(
      paddedOwnershipElts * (useFP16Storage ? sizeof(half_t) : sizeof(float))
    );

    const auto submitAndWait = [&](VkCommandBuffer commandBuffer) {
      res = vk_helper::endCommandBuffer(commandBuffer);
      CHECK_VK_MSG("End model evaluation command buffer", res);
      res = vkResetFences(handle->vulkanDevice->device, 1, &computeHandle->model->fence);
      CHECK_VK_MSG("Reset model evaluation fence", res);
      res = vk_helper::submitCommandBuffers(handle->vulkanDevice, {commandBuffer}, computeHandle->model->fence);
      CHECK_VK_MSG("Submit model evaluation command buffer", res);
      res = vkWaitForFences(
        handle->vulkanDevice->device, 1, &computeHandle->model->fence, VK_TRUE, UINT64_MAX
      );
      CHECK_VK_MSG("Wait for model evaluation fence", res);
    };

    VkCommandBuffer evaluationCB = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
    res = vk_helper::beginCommandBuffer(evaluationCB);
    CHECK_VK_MSG("Begin model evaluation command buffer", res);
  #ifdef VK_BENCHMARK
    if(handle->benchmarkEnabled)
      vkCmdResetQueryPool(evaluationCB, handle->queryPool, 0, 4096);
  #endif
    vk_helper::recordBufferCopy(
      evaluationCB, buffers->uploadBuffer, buffers->input,
      buffers->inputUploadOffset, 0, spatialInputBytes
    );
    vk_helper::recordBufferCopy(
      evaluationCB, buffers->uploadBuffer, buffers->inputGlobal,
      buffers->inputGlobalUploadOffset, 0, inputGlobalBytes
    );
    if(numMetaFeatures > 0) {
      vk_helper::recordBufferCopy(
        evaluationCB, buffers->uploadBuffer, buffers->inputMeta,
        buffers->inputMetaUploadOffset, 0, inputMetaBytes
      );
    }
    for(VulkanBuffer* inputBuffer: {buffers->input, buffers->inputGlobal, buffers->inputMeta}) {
      if(inputBuffer != nullptr) {
        vk_helper::barrierCommandBufferForBuffer(
          evaluationCB, inputBuffer,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT
        );
      }
    }

    computeHandle->model->forward(
      evaluationCB,
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

    for(VulkanBuffer* outputBuffer: {
      buffers->policyPass, buffers->policy, buffers->value, buffers->scoreValue, buffers->ownership
    }) {
      vk_helper::barrierCommandBufferForBuffer(
        evaluationCB, outputBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
      );
    }
    vk_helper::recordBufferCopy(
      evaluationCB, buffers->policyPass, buffers->readbackBuffer,
      0, buffers->policyPassReadbackOffset, policyPassBytes
    );
    vk_helper::recordBufferCopy(
      evaluationCB, buffers->policy, buffers->readbackBuffer,
      0, buffers->policyReadbackOffset, policyBytes
    );
    vk_helper::recordBufferCopy(
      evaluationCB, buffers->value, buffers->readbackBuffer,
      0, buffers->valueReadbackOffset, valueBytes
    );
    vk_helper::recordBufferCopy(
      evaluationCB, buffers->scoreValue, buffers->readbackBuffer,
      0, buffers->scoreValueReadbackOffset, scoreValueBytes
    );
    vk_helper::recordBufferCopy(
      evaluationCB, buffers->ownership, buffers->readbackBuffer,
      0, buffers->ownershipReadbackOffset, ownershipBytes
    );
    submitAndWait(evaluationCB);

    vk_helper::copyReadbackBufferToHost(
      handle->vulkanDevice, buffers->readbackBuffer, buffers->policyPassReadbackOffset,
      policyPassBytes, inputBuffers->policyPassResults, &res
    );
    CHECK_VK_MSG("Copy policy pass results from persistent readback buffer", res);
    vk_helper::copyReadbackBufferToHost(
      handle->vulkanDevice, buffers->readbackBuffer, buffers->policyReadbackOffset,
      policyBytes, useFP16Storage ? static_cast<void*>(inputBuffers->policyResultsHalf) : inputBuffers->policyResults, &res
    );
    CHECK_VK_MSG("Copy policy results from persistent readback buffer", res);
    vk_helper::copyReadbackBufferToHost(
      handle->vulkanDevice, buffers->readbackBuffer, buffers->valueReadbackOffset,
      valueBytes, inputBuffers->valueResults, &res
    );
    CHECK_VK_MSG("Copy value results from persistent readback buffer", res);
    vk_helper::copyReadbackBufferToHost(
      handle->vulkanDevice, buffers->readbackBuffer, buffers->scoreValueReadbackOffset,
      scoreValueBytes, inputBuffers->scoreValueResults, &res
    );
    CHECK_VK_MSG("Copy score value results from persistent readback buffer", res);
    vk_helper::copyReadbackBufferToHost(
      handle->vulkanDevice, buffers->readbackBuffer, buffers->ownershipReadbackOffset,
      ownershipBytes, useFP16Storage ? static_cast<void*>(inputBuffers->ownershipResultsHalf) : inputBuffers->ownershipResults, &res
    );
    CHECK_VK_MSG("Copy ownership results from persistent readback buffer", res);

    bool hasNonfiniteValueHeadOutput = false;
    for(size_t i = 0; i < static_cast<size_t>(batchSize) * inputBuffers->singleValueResultElts; i++) {
      if(!std::isfinite(inputBuffers->valueResults[i])) {
        hasNonfiniteValueHeadOutput = true;
        break;
      }
    }
    if(!hasNonfiniteValueHeadOutput) {
      for(size_t i = 0; i < static_cast<size_t>(batchSize) * inputBuffers->singleScoreValueResultElts; i++) {
        if(!std::isfinite(inputBuffers->scoreValueResults[i])) {
          hasNonfiniteValueHeadOutput = true;
          break;
        }
      }
    }
    if(hasNonfiniteValueHeadOutput) {
      std::vector<float> maskSums(static_cast<size_t>(batchSize));
      VkResult maskSumReadbackResult = VK_SUCCESS;
      vk_helper::copyDeviceBufferToHost(
        handle->vulkanDevice, buffers->maskSum,
        static_cast<VkDeviceSize>(sizeof(float) * maskSums.size()),
        maskSums.data(), true, &maskSumReadbackResult
      );
      if(maskSumReadbackResult == VK_SUCCESS) {
        std::cerr << "Vulkan value-head nonfinite: batchSize=" << batchSize
                  << " useNHWC=" << (handle->pipelines->useNHWC ? 1 : 0)
                  << " generic1x1.accType=" << handle->tuneParams.hgemmCooperativeMatrixNHWC.accType
                  << " maskSums=";
        for(float maskSum: maskSums)
          std::cerr << " " << maskSum;
        std::cerr << std::endl;
      }
      else {
        std::cerr << "Vulkan value-head nonfinite: failed to read mask sums: "
                  << vk_helper::vkErrorToString(maskSumReadbackResult) << std::endl;
      }

      if(handle->pipelines->useNHWC) {
        const int trunkChannels = handle->getNHWCChannelsPadded(computeHandle->model->trunk->trunkNumChannels);
        const size_t trunkElts = static_cast<size_t>(batchSize) * paddedNNXYLen * trunkChannels;
        std::vector<half_t> trunkValues(trunkElts);
        VkResult trunkReadbackResult = VK_SUCCESS;
        vk_helper::copyDeviceBufferToHost(
          handle->vulkanDevice, buffers->trunk,
          static_cast<VkDeviceSize>(sizeof(half_t) * trunkValues.size()),
          trunkValues.data(), true, &trunkReadbackResult
        );
        if(trunkReadbackResult == VK_SUCCESS) {
          for(size_t i = 0; i < trunkValues.size(); i++) {
            if(!std::isfinite(static_cast<float>(trunkValues[i]))) {
              const size_t batchStride = static_cast<size_t>(paddedNNXYLen) * trunkChannels;
              const size_t withinBatch = i % batchStride;
              std::cerr << "Vulkan trunk nonfinite: batch=" << i / batchStride
                        << " xy=" << withinBatch / trunkChannels
                        << " channel=" << withinBatch % trunkChannels << std::endl;
              break;
            }
            if(i + 1 == trunkValues.size())
              std::cerr << "Vulkan trunk finite before value head" << std::endl;
          }
        }
        else {
          std::cerr << "Vulkan value-head nonfinite: failed to read trunk: "
                    << vk_helper::vkErrorToString(trunkReadbackResult) << std::endl;
        }
      }
    }

    #ifdef VK_DUMP_BUFFER
    printHostBuffer(
      """[NeuralNet::getOutput] policy pass results",
      inputBuffers->policyPassResults,
      batchSize * inputBuffers->singlePolicyPassResultElts
    );
    #endif
    // std::cout << "policy pass result[0]: " << inputBuffers->policyPassResults[0] << std::endl;


    // Read back Policy result
    if ( useFP16Storage ) {
      if (computeHandle->inputUsingNHWC) {
        const int storageChannels = handle->getNHWCChannelsPadded(numPolicyChannels);
        for (int n = 0; n < batchSize; ++n) {
          for (int xy = 0; xy < nnXYLen; ++xy) {
            for (int c = 0; c < numPolicyChannels; ++c) {
              const size_t src = (static_cast<size_t>(n) * paddedNNXYLen + xy) * storageChannels + c;
              const size_t dst = (static_cast<size_t>(n) * numPolicyChannels + c) * nnXYLen + xy;
              inputBuffers->policyResults[dst] = inputBuffers->policyResultsHalf[src];
            }
          }
        }
      } else if ( paddedNNXYLen == nnXYLen ) {
        size_t totalChannels = static_cast<size_t>(numPolicyChannels) * batchSize;
        for ( size_t i = 0 ; i < totalChannels * nnXYLen ; ++i )
          inputBuffers->policyResults[i] = inputBuffers->policyResultsHalf[i];
      } else {
        size_t totalChannels = static_cast<size_t>(numPolicyChannels) * batchSize;
        for ( size_t c = 0 ; c < totalChannels ; ++c )
          for ( int xy = 0 ; xy < nnXYLen ; ++xy )
            inputBuffers->policyResults[c * nnXYLen + xy] = inputBuffers->policyResultsHalf[c * paddedNNXYLen + xy];
      }
    } else {
      if ( paddedNNXYLen != nnXYLen ) {
        ASSERT_UNREACHABLE;
      }
      #ifdef VK_DUMP_BUFFER
      printHostBuffer(
        "[NeuralNet::getOutput] policy results",
        inputBuffers->policyResults,
        batchSize * inputBuffers->singlePolicyResultElts
      );
      #endif
    }

    for(int n = 0; n < batchSize; n++) {
      for(int c = 0; c < numPolicyChannels; c++) {
        for(int xy = 0; xy < nnXYLen; xy++) {
          const size_t index =
            (static_cast<size_t>(n) * numPolicyChannels + c) * nnXYLen + xy;
          if(!std::isfinite(inputBuffers->policyResults[index])) {
            std::cerr << "Vulkan policy readback nonfinite: batch=" << n
                      << " channel=" << c
                      << " xy=" << xy
                      << " value=" << inputBuffers->policyResults[index]
                      << " nnXYLen=" << nnXYLen
                      << " paddedNNXYLen=" << paddedNNXYLen
                      << " storageChannels=" << policyStorageChannels
                      << " useNHWC=" << (handle->pipelines->useNHWC ? 1 : 0)
                      << std::endl;
            n = batchSize;
            c = numPolicyChannels;
            break;
          }
        }
      }
    }
    for(int n = 0; n < batchSize; n++) {
      for(int c = 0; c < numPolicyChannels; c++) {
        if(!std::isfinite(inputBuffers->policyPassResults[
             static_cast<size_t>(n) * numPolicyChannels + c])) {
          std::cerr << "Vulkan policy pass readback nonfinite: batch=" << n
                    << " channel=" << c
                    << " value=" << inputBuffers->policyPassResults[
                      static_cast<size_t>(n) * numPolicyChannels + c]
                    << std::endl;
          n = batchSize;
          break;
        }
      }
    }

    #ifdef VK_DUMP_BUFFER
    printHostBuffer(
      "[NeuralNet::getOutput] value results",
      inputBuffers->valueResults,
      batchSize * inputBuffers->singleValueResultElts
    );
    #endif

    #ifdef VK_DUMP_BUFFER
    printHostBuffer(
      "[NeuralNet::getOutput] score value results",
      inputBuffers->scoreValueResults,
      batchSize * inputBuffers->singleScoreValueResultElts
    );
    #endif

    // Read back Ownership result
    if ( useFP16Storage ) {
      if (computeHandle->inputUsingNHWC) {
        const int storageChannels = handle->getNHWCChannelsPadded(ownershipChannels);
        for (int n = 0; n < batchSize; ++n) {
          for (int xy = 0; xy < nnXYLen; ++xy) {
            for (int c = 0; c < ownershipChannels; ++c) {
              const size_t src = (static_cast<size_t>(n) * paddedNNXYLen + xy) * storageChannels + c;
              const size_t dst = (static_cast<size_t>(n) * ownershipChannels + c) * nnXYLen + xy;
              inputBuffers->ownershipResults[dst] = inputBuffers->ownershipResultsHalf[src];
            }
          }
        }
      } else if ( paddedNNXYLen == nnXYLen ) {
        size_t totalChannels = static_cast<size_t>(ownershipChannels) * batchSize;
        for ( size_t i = 0 ; i < totalChannels * nnXYLen ; ++i )
          inputBuffers->ownershipResults[i] = inputBuffers->ownershipResultsHalf[i];
      } else {
        size_t totalChannels = static_cast<size_t>(ownershipChannels) * batchSize;
        for ( size_t c = 0 ; c < totalChannels ; ++c )
          for ( int xy = 0 ; xy < nnXYLen ; ++xy )
            inputBuffers->ownershipResults[c * nnXYLen + xy] = inputBuffers->ownershipResultsHalf[c * paddedNNXYLen + xy];
      }
    } else {
      if ( paddedNNXYLen != nnXYLen ) {
        ASSERT_UNREACHABLE;
      }
      #ifdef VK_DUMP_BUFFER
      printHostBuffer(
        "[NeuralNet::getOutput] ownership results",
        inputBuffers->ownershipResults,
        batchSize * inputBuffers->singleOwnershipResultElts
      );
      #endif
    }

    vkResetCommandPool(handle->device, handle->vulkanDevice->commandPool, 0);

  #ifdef VK_BENCHMARK
    handle->dumpVulkanBenchmark(computeHandle->model->modelName);
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
  #ifdef VK_DUMP_BUFFER
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

  // The NHWC ConvLayer experiment requires the FP16 cooperative-matrix path.
  // Keep the existing NCHW test support unchanged for all other combinations.
  if(useNHWC != useFP16)
    return false;

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

  ComputeContext* ctx = createComputeContextForTesting({gpuId}, logger, nnXLen, nnYLen, useFP16, false);
  // std::cout << "[testEvaluateConv] Created compute context" << std::endl;
  ComputeHandleInternal* handle = new ComputeHandleInternal(ctx,static_cast<int>(gpuId), false, false);
  const VulkanDevice* device = handle->vulkanDevice;
  ConvLayer *layer = new ConvLayer(handle, desc, nnXLen, nnYLen, useFP16);
  if(useNHWC && !layer->usingNHWC) {
    delete layer;
    delete handle;
    freeComputeContext(ctx);
    return false;
  }
  const size_t logicalSpatialSize = static_cast<size_t>(nnXLen) * static_cast<size_t>(nnYLen);
  const size_t numInputFloats = static_cast<size_t>(batchSize) * static_cast<size_t>(desc->inChannels) * logicalSpatialSize;
  const size_t numOutputFloats = static_cast<size_t>(batchSize) * static_cast<size_t>(desc->outChannels) * logicalSpatialSize;

  if ( numInputFloats != inputBuffer.size() ) {
    // std::cerr << "testEvaluateConv input size mismatch, expected " << numInputFloats << " got " << inputBuffer.size() << std::endl;
    delete layer;
    delete handle;
    delete ctx;
    return false;
  }
  outputBuffer.resize(numOutputFloats);
  // std::cout << "  expected output size " <<  numOutputFloats << std::endl;
  const size_t paddedSpatialSize = static_cast<size_t>(handle->paddedNNXYLen);
  std::vector<half_t> inputHalf;
  const void* inputData = inputBuffer.data();
  size_t inputBytes = byteSizeofVectorContents(inputBuffer);
  if(useNHWC) {
    inputHalf.assign(
      static_cast<size_t>(batchSize) * desc->inChannels * paddedSpatialSize,
      half_float::half_cast<half_t>(0.0f)
    );
    for(int n = 0; n < batchSize; ++n) {
      for(int c = 0; c < desc->inChannels; ++c) {
        for(size_t xy = 0; xy < logicalSpatialSize; ++xy) {
          inputHalf[(static_cast<size_t>(n) * desc->inChannels + c) * paddedSpatialSize + xy] =
            half_float::half_cast<half_t>(inputBuffer[(static_cast<size_t>(n) * logicalSpatialSize + xy) * desc->inChannels + c]);
        }
      }
    }
    inputData = inputHalf.data();
    inputBytes = byteSizeofVectorContents(inputHalf);
  }
  std::vector<float> inputTmp = inputBuffer;
  VulkanBuffer* dInput = vk_helper::createDeviceBufferWithData(
    device,
    inputBytes,
    useNHWC ? inputData : static_cast<const void*>(inputTmp.data()),
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

  const size_t outputStorageElts = useNHWC
    ? static_cast<size_t>(batchSize) * desc->outChannels * paddedSpatialSize
    : numOutputFloats;
  VulkanBuffer* dOutput = vk_helper::createDeviceBuffer(
    device,
    outputStorageElts * (useFP16 ? sizeof(half_t) : sizeof(float)),
    false,
    &res
  );
  CHECK_VK_MSG("[TestConv] Failed to create device buffer", res);

  VkCommandBuffer commandBuffer = vk_helper::allocateCommandBuffer(device);
  res = vk_helper::beginCommandBuffer(commandBuffer);
  CHECK_VK_MSG("[TestConv] Failed to begin command buffer", res);
  layer->forward(commandBuffer, batchSize, dInput, dOutput, convWorkspace, convWorkspace2);
  res = vk_helper::endCommandBuffer(commandBuffer);
  CHECK_VK_MSG("[TestConv] Failed to end command buffer", res);
  vk_helper::submitCommandBuffers(device, {commandBuffer}, nullptr);
  if(useNHWC) {
    std::vector<half_t> outputHalf(outputStorageElts);
    vk_helper::copyDeviceBufferToHost(
      device, dOutput, static_cast<VkDeviceSize>(sizeof(half_t) * outputStorageElts), outputHalf.data(), true, &res
    );
    CHECK_VK_MSG("[TestConv] Failed to copy device output buffer to host", res);
    for(int n = 0; n < batchSize; ++n) {
      for(int c = 0; c < desc->outChannels; ++c) {
        for(size_t xy = 0; xy < logicalSpatialSize; ++xy) {
          outputBuffer[(static_cast<size_t>(n) * logicalSpatialSize + xy) * desc->outChannels + c] =
            static_cast<float>(outputHalf[(static_cast<size_t>(n) * desc->outChannels + c) * paddedSpatialSize + xy]);
        }
      }
    }
  } else {
    vk_helper::copyDeviceBufferToHost(
      device, dOutput, static_cast<VkDeviceSize>(sizeof(float) * numOutputFloats), outputBuffer.data(), true, &res
    );
  }
  CHECK_VK_MSG("[TestConv] Failed to copy device output buffer to host", res);
  vkQueueWaitIdle(device->queue);
  vkDeviceWaitIdle(device->device);
  vk_helper::releaseVulkanBuffer(device, dInput);
  vk_helper::releaseVulkanBuffer(device, dOutput);
  vk_helper::releaseVulkanBuffer(device, convWorkspace);
  vk_helper::releaseVulkanBuffer(device, convWorkspace2);
  dInput = nullptr;
  dOutput = nullptr;
  convWorkspace = nullptr;
  convWorkspace2 = nullptr;
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

    Logger* logger = nullptr;
    auto ctx = createComputeContextForTesting({0}, nullptr, nnXLen, nnYLen, useFP16, useNHWC);
    useNHWC = false; // TODO: enable NHWC testing later.
    auto handle = new ComputeHandleInternal(ctx,0, useNHWC, useNHWC);
    // BatchNormLayer *layer = new BatchNormLayer(handle, desc, nnXLen
    ActivationLayerDesc actDesc;
    actDesc.activation = ACTIVATION_IDENTITY;
    BatchNormLayer *layer = new BatchNormLayer(handle, desc, &actDesc, false);
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

    VkCommandBuffer commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("[testEvaluateBatchNorm] Failed to begin command buffer", res);
    layer->forward(commandBuffer, batchSize, dInput, dMask, dOutput);
    res = vk_helper::endCommandBuffer(commandBuffer);
    CHECK_VK_MSG("[testEvaluateBatchNorm] Failed to end command buffer", res);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {commandBuffer}, nullptr);
    vk_helper::copyDeviceBufferToHost(handle->vulkanDevice, dOutput, static_cast<VkDeviceSize>(sizeof(float) * outputBuffer.size()), outputBuffer.data(), true, &res);
    CHECK_VK_MSG("[testEvaluateBatchNorm] Failed to copy device output buffer to host", res);
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

    useNHWC = false; // TODO: enable NHWC testing later.
    ComputeContext* ctx = createComputeContextForTesting({gpuId}, logger, nnXLen, nnYLen, false, false);
    ComputeHandleInternal* handle = new ComputeHandleInternal(ctx,static_cast<int>(gpuId), useNHWC, useNHWC);
    ResidualBlock *layer = new ResidualBlock(handle, desc, nnXLen, nnYLen, false);

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
    ScratchBuffers* scratch = new ScratchBuffers(handle, batchSize);
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

    VkCommandBuffer commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("[testEvaluateResidualBlock] Failed to begin command buffer", res);
    layer->forward(commandBuffer, batchSize, scratch, dTrunk, dTrunkScratch, dMask, convWorkspace, convWorkspace2);
    res = vk_helper::endCommandBuffer(commandBuffer);
    CHECK_VK_MSG("[testEvaluateResidualBlock] Failed to end command buffer", res);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {commandBuffer}, nullptr);
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
    GlobalPoolingResidualBlock *layer = new GlobalPoolingResidualBlock(handle, desc, nnXLen, nnYLen, false);

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
    ScratchBuffers* scratch = new ScratchBuffers(handle, batchSize);
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
    VkCommandBuffer maskSumsCB = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
    const LocalDim maskSumDim = {
      handle->tuneParams.gPool.XYSTRIDE,
      1,
      std::min(
        handle->tuneParams.gPool.BATCHSTRIDE,
        static_cast<int>(vk_helper::powerOf2ify(batchSize))
      )
    };
    const Pipeline& maskSumPipeline = handle->pipelines->sumChannels.at(maskSumDim);
    VkDescriptorSet maskSumsDS = vk_helper::allocateDescriptorSet(
      handle->vulkanDevice, maskSumPipeline.descriptorSetLayout, &res
    );
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to allocate mask sum descriptor set", res);
    vkcompute::computeMaskSums(handle, maskSumsCB, maskSumsDS, batchSize, dMask, dMaskSum);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {maskSumsCB}, nullptr);
    vk_helper::copyDeviceBufferToHost(handle->vulkanDevice, dMaskSum, static_cast<VkDeviceSize>(sizeof(float) * numMaskSumFloats), maskSumTmp.data(), true, &res);
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to copy device mask sum buffer to host", res);
    // printFloatBuffer("[testEvaluateGlobalPoolingResidualBlock] Mask Sums", maskSumTmp.data(), maskSumTmp.size(), batchSize, 1, 1, 1);
    VkCommandBuffer commandBuffer = vk_helper::allocateCommandBuffer(handle->vulkanDevice);
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to begin command buffer", res);
    layer->forward(commandBuffer, batchSize, scratch, dTrunk, dTrunkScratch, dMask, dMaskSum, convWorkspace, convWorkspace2);
    res = vk_helper::endCommandBuffer(commandBuffer);
    CHECK_VK_MSG("[testEvaluateGlobalPoolingResidualBlock] Failed to end command buffer", res);
    vk_helper::submitCommandBuffers(handle->vulkanDevice, {commandBuffer}, nullptr);
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
