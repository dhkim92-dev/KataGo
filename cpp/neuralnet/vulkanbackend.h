/**
 * @author Dohoon Kim(https://github.com/dhkim92-dev)
 * @brief Vulkan backend for Neural Net evaluation
 */
#ifdef USE_VULKAN_BACKEND
#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <iomanip>
#include "../external/half-2.2.0/include/half.hpp"
#include "../core/simpleallocator.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkancompute.h"
#include "../neuralnet/vulkanshaders.h"

using half_t = half_float::half;

// #define SHADER_PROFILE
#ifdef SHADER_PROFILE
#define SHADER_PROFILE_START(shaderName, cb) { \
  uint64_t beginQueryIdx = handle->queryIdx; \
  uint64_t endQueryIdx = handle->queryIdx+1; \
  handle->queryIdx+=2; \
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, handle->queryPool, beginQueryIdx);
#define SHADER_PROFILE_END(shaderName, cb)  \
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, handle->queryPool, endQueryIdx); \
  handle->commandDispatchInfos.push_back( { shaderName, beginQueryIdx, endQueryIdx } );
}
#else 
#define SHADER_PROFILE_START(shaderName, cb)
#define SHADER_PROFILE_END(shaderName, cb) 
#endif
#define FLOAT16_SIZE_IN_BYTES 2

static void checkBufferSize(int batchSize, int nnXLen, int nnYLen, int channels) {
  if((int64_t)batchSize * nnXLen * nnYLen * channels >= (int64_t)1 << 31)
    throw StringError("Batch size too large, resulting GPU buffers might exceed 2^31 entries which is not currently supported");
}

template<typename T>
static size_t byteSizeofVectorContents(const typename std::vector<T>& vec) {
  return sizeof(T) * vec.size();
}

struct CommandDispatchInfo {
  std::string shaderName;
  uint64_t startQueryIdx;
  uint64_t endQueryIdx;
};

struct ShaderExecutionInfo {
  uint64_t callCount;
  uint64_t totalExecutionTimeNs;
};

struct ComputeHandleInternal {
  const ComputeContext* context;
  const VulkanDevice* vulkanDevice;
  const vk_shader::ComputePipelines* pipelines;
  const vk_shader::tune::VulkanTuneParams& tuneParams;
  VkDevice device;
  VkQueue queue;

  int nnXLen;
  int nnYLen;
  int paddedNNXYLen; // nnXLen * nnYLen rounded up for spatial alignment
  int qHeadDim=-1, vHeadDim=-1;

  bool usingFP16Storage = false;
  bool usingFP16Compute = false;
  bool usingFP16TensorCores = false;
  bool usingFP16TensorCoresFor1x1 = false;

#ifdef SHADER_PROFILE
  uint64_t runCount = 0;
  uint64_t queryIdx = 0;
  VkQueryPool queryPool = VK_NULL_HANDLE;
  std::vector<CommandDispatchInfo> commandDispatchInfos;
  std::unordered_map<std::string, ShaderExecutionInfo> shaderProfileInfos;

  void dumpShaderProfile() {
    runCount++;

    // Get timestampPeriod for tick -> nanosecond conversion
    double timestampPeriod = 1.0;
    if (vulkanDevice != nullptr) {
      timestampPeriod = static_cast<double>(vulkanDevice->info.properties.limits.timestampPeriod);
    }

    // Query results for each dispatch and accumulate into shaderProfileInfos
    for (const auto& dispatchInfo : commandDispatchInfos) {
      const std::string& name = dispatchInfo.shaderName;
      uint64_t startIdx = dispatchInfo.startQueryIdx;
      uint64_t endIdx = dispatchInfo.endQueryIdx;

      if (endIdx <= startIdx) continue;

      // Read start and end timestamps (2 queries per dispatch)
      uint64_t timestamps[2] = {0, 0};
      VkResult res = vkGetQueryPoolResults(
        device,
        queryPool,
        static_cast<uint32_t>(startIdx),
        2,  // start and end timestamps
        sizeof(timestamps),
        timestamps,
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
      );

      uint64_t executionNs = 0;
      if (res == VK_SUCCESS && timestamps[1] >= timestamps[0]) {
        uint64_t ticks = timestamps[1] - timestamps[0];
        double ns = static_cast<double>(ticks) * timestampPeriod;
        executionNs = static_cast<uint64_t>(ns + 0.5);
      }

      // Accumulate into shaderProfileInfos
      auto iter = shaderProfileInfos.find(name);
      if (iter == shaderProfileInfos.end()) {
        ShaderExecutionInfo newInfo;
        newInfo.callCount = 1;
        newInfo.totalExecutionTimeNs = executionNs;
        shaderProfileInfos[name] = newInfo;
      } else {
        iter->second.callCount += 1;
        iter->second.totalExecutionTimeNs += executionNs;
      }
    }

    // Print profile results every 100 runs
    if (runCount % 100 == 0) {
      std::cout << "## Vulkan Shader Profile Results (Run " << runCount << "):\n";
      std::cout << std::fixed << std::setprecision(3);

      for (auto& pair : shaderProfileInfos) {
        const std::string& name = pair.first;
        const ShaderExecutionInfo& info = pair.second;

        double totalTimeSec = static_cast<double>(info.totalExecutionTimeNs) / 1e9;  // ns -> sec
        double avgTimeMs = (info.callCount > 0)
          ? static_cast<double>(info.totalExecutionTimeNs) / static_cast<double>(info.callCount) / 1e6  // ns -> ms
          : 0.0;

        std::cout << name << " / " << info.callCount
                  << " / " << totalTimeSec << " s / " << avgTimeMs << " ms\n";

        // Reset counters for next interval
        pair.second.callCount = 0;
        pair.second.totalExecutionTimeNs = 0;
      }

      std::cout << std::defaultfloat;
      std::cout << "## End of Vulkan Shader Profile Results\n";
    }

    // Clear dispatch infos and reset query pool for next batch
    commandDispatchInfos.clear();
    queryIdx = 0;
    vkResetQueryPool(device, queryPool, 0, 4096);
  }

  
#endif

  ComputeHandleInternal(ComputeContext* ctx, int gpuIdx, bool inputsUseNHWC, bool useNHWC);

  int getXGemmMPaddingMult() const {
    return tuneParams.xgemm.MWG;
  }

  int getXGemmNPaddingMult() const {
    return tuneParams.xgemm.NWG;
  }

  int getXGemmKPaddingMult() const {
    return tuneParams.xgemm.KWG;
  }
};


struct ScratchBuffers {
  const size_t batchXYFloatBytes;
  const size_t batchFloatBytes;
  const size_t batchXYBytes;
  const size_t batchBytes;
  ComputeHandleInternal *handle;
  SimpleAllocator<VulkanBuffer *> *allocator;

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;

  ~ScratchBuffers() {
    delete allocator;
  }

  ScratchBuffers(
    ComputeHandleInternal* handle_,
    int maxBatchSize
  ): 
    batchXYFloatBytes((size_t)maxBatchSize * handle_->paddedNNXYLen * sizeof(float)),
    batchFloatBytes((size_t)maxBatchSize * sizeof(float)),
    batchXYBytes((size_t)maxBatchSize * handle_->paddedNNXYLen * (handle_->usingFP16Storage ? sizeof(half_t) : sizeof(float))),
    batchBytes((size_t)maxBatchSize * (handle_->usingFP16Storage ? sizeof(half_t) : sizeof(float))),
    handle(handle_)
  {
    // std::cout << "Allocating ScratchBuffers for max batch size " << maxBatchSize << ", nnXLen " << nnXLen << ", nnYLen " << nnYLen << std::endl;
    std::function<VulkanBuffer*(size_t)> allocFunc = [this](size_t size) {
      VkResult res = VK_SUCCESS;
      return vk_helper::createDeviceBuffer(handle->vulkanDevice, size, false, &res);
      CHECK_VK_MSG("Allocate Scratch Buffer of size " + std::to_string(size), res);
      // std::cout<<"Allocated Scratch Buffer of size " << size << std::endl;
    };
    std::function<void(VulkanBuffer *)> freeFunc = [this](VulkanBuffer *buffer) {
      vk_helper::releaseVulkanBuffer(handle->vulkanDevice, buffer);
      // std::cout << "Released Scratch Buffer" << std::endl;
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

struct RMSNormLayer;

/**
 * @brief NN Block Stack structure
 */
struct BlockStack {
  ComputeHandleInternal *handle;
  std::vector<std::pair<int, unique_ptr_void>> blocks;
  const int numBlocks;
  const int trunkNumChannels;
  const int nnXLen;
  const int nnYLen;

  /**
   * @brief Constructor for BlockStack
   * @param handle_: Compute handle
   * @param descBlocks: vector of block descriptions
   * @param numBlocks_: number of blocks
   * @param trunkNumChannels_: number of channels in trunk
   * @param nnXLen_: neural net X length
   * @param nnYLen_: neural net Y length
   */
  BlockStack(
    ComputeHandleInternal *handle,
    const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
    int numBlocks_,
    int trunkNumChannels_,
    int nnXLen_,
    int nnYLen_,
    bool useFP16
  );
  BlockStack() = delete;
  BlockStack(const BlockStack&) = delete;
  BlockStack& operator=(const BlockStack&) = delete;
  ~BlockStack();

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal *handle, size_t maxBatchSize) const;

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
  );

  void debug(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* convWorkspace,
    VulkanBuffer* convWorkspace2
  );
};

#endif
