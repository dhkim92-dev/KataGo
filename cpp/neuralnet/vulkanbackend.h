/**
 * @author Dohoon Kim(https://github.com/dhkim92-dev, dhkim92.dev@gmail.com, https://www.dohoon-kim.kr)
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
#include "../core/simpleallocator.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkancompute.h"
#include "../neuralnet/vulkanshaders.h"

#define ACTIVATION_IDENTITY 0
#define ACTIVATION_RELU 1
#define ACTIVATION_MISH 2
#define ACTIVATION_MISH_SCALE8 12

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
  VulkanTuneParams tuneParams;
  VkDevice device;
  VkQueue queue;
  bool usingFP16Storage = false;
  bool usingFp16Compute = false;
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

  int getXgemmKPaddingMult() const {
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
    int maxBatchSize,
    int nnXLen,
    int nnYLen
  ): 
    handle(handle_),
    batchXYFloatBytes(sizeof(float) * maxBatchSize * nnXLen * nnYLen),
    batchFloatBytes(sizeof(float) * maxBatchSize),
    batchXYBytes( (handle_->usingFP16Storage ? FLOAT16_SIZE_IN_BYTES : sizeof(float)) * maxBatchSize * nnXLen * nnYLen),
    batchBytes((handle_->usingFP16Storage ? FLOAT16_SIZE_IN_BYTES : sizeof(float))  * maxBatchSize)
  {
    // std::cout << "Allocating ScratchBuffers for max batch size " << maxBatchSize << ", nnXLen " << nnXLen << ", nnYLen " << nnYLen << std::endl;
    std::function<VulkanBuffer*(size_t)> allocFunc = [this](size_t size) {
      VkResult res = VK_SUCCESS;
      return VkHelpers::createDeviceBuffer(handle->vulkanDevice, size, false, &res);
      CHECK_VK_MSG("Allocate Scratch Buffer of size " + std::to_string(size), res);
      // std::cout<<"Allocated Scratch Buffer of size " << size << std::endl;
    };
    std::function<void(VulkanBuffer *)> freeFunc = [this](VulkanBuffer *buffer) {
      VkHelpers::releaseVulkanBuffer(handle->vulkanDevice, buffer);
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
    int nnYLen_
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

/**
 * @brief Vulkan Compute  Pipeline structure
 * 
 */

namespace KatagoVulkan {

  /**
   * @brief Will be used to tune various parameters for different devices
   *        Not implemented yet. support in future. Maybe profiled tuning will be adopted.
   *        Becuase vulkan can not decide optimal parameters at runtime like OpenCL.
   */

  /**
   * @brief Push constant parameters for Conv2D operation. it makes easier to pass small parameters to shader.
   */
  struct Conv2DPushConstantParams {
    uint32_t batchSize; // Batch size
    uint32_t inChannels; // Input channels
    uint32_t outChannels; // Output channels
    uint32_t nnYLen;
    uint32_t nnXLen;
    uint32_t filterH; // Filter height
    uint32_t filterW; // Filter width
  };

  struct Conv2DTiledBnActParams {
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t nnYLen;
    uint32_t nnXLen;
    uint32_t filterH;
    uint32_t filterW;
    uint32_t activation; // 0: Identity, 1: ReLU, 2: Mish, 3: Mish + Scale8
  };

  struct WinogradInputTransformParams {
    uint32_t batchSize;
    uint32_t nnYLen;
    uint32_t nnXLen;
    uint32_t numTilesY;
    uint32_t numTilesX;
    uint32_t inChannels;
    uint32_t inChannelsPadded;
    uint32_t ntxtySizePadded;
  };

  struct WinogradInputTransformBnActSpec {
    uint32_t localSizeX = 1;
    uint32_t localSizeY = 1;
    uint32_t localSizeZ = 1;
    int inTileYSize = 4;
    int inTileXSize = 4;
    int outTileYSize = 2;
    int outTileXSize = 2;
    int inTileYOffset = -1;
    int inTileXOffset = -1;
    int convY = 3;
    int convX = 3;
    int activation = 0; // 0: Identity, 1: ReLU, 2: Mish, 12: Mish + Scale8
  };

  struct WinogradInputTransformSpec {
    uint32_t localSizeX = 1;
    uint32_t localSizeY = 1;
    uint32_t localSizeZ = 1;
    int inTileYSize = 4;
    int inTileXSize = 4;
    int outTileYSize = 2;
    int outTileXSize = 2;
    int inTileYOffset = -1;
    int inTileXOffset = -1;
    int convY = 3;
    int convX = 3;
  };

  struct WinogradOutputTransformParams {
    int batchSize;
    int ySize;
    int xSize;
    int numTilesY;
    int numTilesX;
    int outChannels;
    int outChannelsPadded;
    int ntxtySizePadded;
  };

  struct WinogradOutputTransformSpec {
    uint32_t localSizeX = 1;
    uint32_t localSizeY = 1;
    uint32_t localSizeZ = 1;
    int inTileYSize = 4;
    int inTileXSize = 4;
    int outTileYSize = 2;
    int outTileXSize = 2;
    int convY = 3;
    int convX = 3;
  };

  /**
   * @brief Matmul pipeline Push Constant Parameters
   * @param M: rows of A and C, each batch
   * @param K: cols of A and rows of B, inChannels
   * @param N: cols of B and C, outChannels
   * @param numBatchElts: number of batches
   * @param cTranspose: whether output C is transposed or not
   */
  struct MatmulFp32Params {
    uint32_t M;  
    uint32_t K;  
    uint32_t N;
    uint32_t numBatchElts;
    uint32_t cTranspose; // Output Transpose
  };

  struct XGEMMBatchedParams{
    uint32_t M;  
    uint32_t N;  
    uint32_t K;
    uint32_t aOne;
    uint32_t aTwo;
    uint32_t bOne;
    uint32_t bTwo;
    uint32_t cOne;
    uint32_t cTwo;
  };

  struct XGEMMBatchedSpec {
    uint32_t localSizeX = 16;
    uint32_t localSizeY = 16;
    uint32_t localSizeZ = 1;
    uint32_t MWG=64;
    uint32_t NWG=64;
    uint32_t KWG=32;
    uint32_t MDIMC=16;
    uint32_t NDIMC=16;
   };


  struct BatchedXGEMMDirectParams {
    uint32_t M;  
    uint32_t N;
    uint32_t K;  
    uint32_t aLead;
    uint32_t bLead;
    uint32_t cLead;
    uint32_t aTranspose; // Input A Transpose
    uint32_t bTranspose; // Input B Transpose
    uint32_t cTranspose; // Output C Transpose
  };

  struct XgemmDirectSpec {
    uint32_t localSizeX = 16;
    uint32_t localSizeY = 16;
    uint32_t localSizeZ = 1;
    int WGD = 32;
    int MDIMCD = 8;
    int NDIMCD = 8;
    int MDIMAD = 16;
    int NDIMBD = 16;
    int KWID = 2;
    int PADA = 1;
    int PADB = 1;
  };

  struct XgemmStridedBatchedFp32Params {
    uint32_t kSizeM;  
    uint32_t kSizeN;
    uint32_t kSizeK;  
    uint32_t aLead;
    uint32_t aStride;
    uint32_t bLead;
    uint32_t bStride;
    uint32_t cLead;
    uint32_t cStride;
    uint32_t cTranspose;
  };

  struct BatchedXgemmDirectFp32Params {
    uint32_t kSizeM;  
    uint32_t kSizeN;
    uint32_t kSizeK;  
    uint32_t aLead;
    uint32_t bLead;
    uint32_t cLead;
    uint32_t cTranspose; // Output C Transpose
  };

  /**
   * @brief Batch Normalization Mask Fp32 Push Constant Parameters 
   * @param batchSize
   * @param numChannels
   * @param nnYLen
   */
  struct BatchNormMaskParams {
    uint32_t batchSize;
    uint32_t numChannels;
    uint32_t nnXYLen;
  };

  /**
   * @brief Sum Channels Fp32 Push Constant Parameters
   */
  struct SumChannelsParams {
    uint32_t batchSize;
    uint32_t numChannels;
    uint32_t nnXYLen;
  };

  /**
   * @brief MatBias Push Constant Parameters
   */
  struct MatBiasFp32Params {
    uint32_t batchSize;
    uint32_t numChannels;
  };

   /**
    * @brief Global Pooling Channels Push Constant Parameters
    */
  struct GlobalPoolingChannelsParams {
    uint32_t batchSize;
    uint32_t gpoolChannels;
    uint32_t nnXYLen;
  };

   /**
    * @brief Value Head Pooling Channels Push Constant Parameters
    */
   struct ValueHeadPoolingChannelsParams {
    uint32_t batchSize;
    uint32_t gpoolChannels;
    uint32_t nnXYLen;
   };

   /**
    *  @brief Add Point Wise Push Constant Parameters
    **/
  struct AddPointWiseParams {
    uint32_t totalSize;
  };

  /**
   * @brief Add Channel Bias NCHW Push Constant Parameters
   */
  struct AddChannelBiasNCHWParams {
    uint32_t ncSize;
    uint32_t xySize;
  };

  /**
   * @brief Add Channel Bias NC Push Constant Parameters
   */
  struct AddChannelBiasNCParams {
    uint32_t nSize;
    uint32_t cSize;
  };

  struct ExtractChannel0NCHWParams {
    uint32_t batchSize;
    uint32_t numInputChannels;
    uint32_t nnXYLen;
  };

  struct NCHWPushConstantParams {
    uint32_t N; // Batch size
    uint32_t C; // Channels
    uint32_t H; // Height
    uint32_t W; // Width
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
    const VulkanTuneParams tuneParams;
    VkPipelineCache cache;

    // In this code, assume that NCHW is default format if no postfix is given.

    // Conv2D pipelines
    Pipeline conv2dFp32; 
    Pipeline conv2dTiledBnAct3x3Fp32; // Conv2d + Tiled + BatchNorm + Activation fused pipeline
    Pipeline conv2dTiledBnAct5x5Fp32; // Conv2d + Tiled + BatchNorm + Activation fused pipeline

    Pipeline winogradInputTransform3x3;
    Pipeline winogradInputTransform5x5;

    Pipeline winogradInputTransform3x3_bnact_identity;
    Pipeline winogradInputTransform3x3_bnact_relu;
    Pipeline winogradInputTransform3x3_bnact_mish;
    Pipeline winogradInputTransform3x3_bnact_mish_scale8;
    Pipeline winogradInputTransform5x5_bnact_identity;
    Pipeline winogradInputTransform5x5_bnact_relu;
    Pipeline winogradInputTransform5x5_bnact_mish;
    Pipeline winogradInputTransform5x5_bnact_mish_scale8;

    Pipeline winogradOutputTransform3x3;
    Pipeline winogradOutputTransform5x5;

    Pipeline addPointWiseFp32;  // operation for skipping connections

    // Pipeline for matrix multiplication
    Pipeline matmulFp32; 
    Pipeline batchedXgemmDirect;
    Pipeline xgemmBatchedFp32;

    // note that conv1x1 can be implemented as matmul operation
    // Pipeline stridedBatchedMatmulFp32;
    Pipeline xgemmStridedBatchedFp32;

    // Batch Normalization pipelines
    // note that prediction phase does not need batch normalization operation separately
    // as the parameters can be folded into convolution scale and bias.
    Pipeline batchNormMaskFp32;
    Pipeline batchNormMaskReluFp32;
    Pipeline batchNormMaskMishFp32;
    Pipeline batchNormMaskMishScale8Fp32;

    // Pooling pipelines
    Pipeline globalPoolingChannelsFp32;
    Pipeline valueHeadPoolingChannelsFp32;
    
    // Element wise operations
    Pipeline sumChannelsFp32;

    Pipeline addChannelBiasNCHWFp32;
    Pipeline addChannelBiasNCIdentityFp32;
    Pipeline addChannelBiasNCReluFp32;
    Pipeline addChannelBiasNCMishFp32;
    Pipeline addChannelBiasNCMishScale8Fp32;
    Pipeline extractChannel0NCHWFp32;

    ComputePipelines(VkDevice device_, const VulkanTuneParams& tuneParams_);
    ComputePipelines() = delete;
    ComputePipelines(const ComputePipelines&) = delete;
    ComputePipelines& operator=(const ComputePipelines&) = delete;

    ~ComputePipelines();

  private :
    void createPipelines();
    void destroyPipelines();
    void destroyPipeline(Pipeline& pipeline);
    void createPipeline(
      std::string pipelineName,
      const unsigned char* spirvBytes,
      size_t spirvSize,
      size_t bindingSize,
      uint32_t pushConstantSize,
      Pipeline &outPipeline,
      VkSpecializationInfo* specializationInfo = nullptr
    );
    /**
     * @brief Create a Conv2d Fp32 object
    */
    void createConv2dFp32();

    /**
     * @brief Create Conv2d Tiled Bn + Activation 3x3 Fp32 object
     */
    void createConv2dTiledBnAct3x3Fp32();

    /**
     * @brief Create Conv2d Tiled Bn + Activation 5x5 Fp32 object
     */
    void createConv2dTiledBnAct5x5Fp32();

    void createWinogradInputTransform();

    void createWinogradInputTransformBnAct();

    void createWinogradOutputTransform();

    /**
     * @brief Create a Conv2d3x3 Bn + Identity Activation fused Fp32 objects.
     */
    // void createConv2d3x3BnFp32();

    /**
     * @brief Create a Conv2d3x3 Bn + ReLU Activation fused Fp32 objects.
     */
    // void createConv2d3x3BnReluFp32();
    /**
     * @brief Create a Conv2d3x3 Bn Mish Fp32 object
     */
    // void createConv2d3x3BnMishFp32();

    /**
     * @brief Create a Conv2d5x5 Bn + Identity Activation fused Fp32 objects.
     */
    // void createConv2d5x5BnFp32();

    /**
     * @brief Create a Conv2d5x5 Bn + ReLU Activation fused Fp32 objects.
     */
    // void createConv2d5x5BnReluFp32();

    /**
     * @brief Create a Conv2d5x5 Bn + ReLU Activation fused Fp32 objects.
     */
    // void createConv2d5x5BnReluFp32();

    /**
     * @brief Create a Conv2d5x5 Bn Mish Fp32 object
     */
    // void createConv2d5x5BnMishFp32();

    /**
     * @brief Create a Add Point Wise Fp32 object
     */
    void createAddPointWiseFp32();

    /**
     * @brief Create a Matmul Fp32 object
     */
    void createMatmulFp32();

    /**
     * @brief Create a Batched XGEMM Direct Fp32 object
     */
    void createBatchedXgemmDirect();

    /**
     * @brief Create a XGEMM Batched Fp32 object
     */
    void createXGEMMBatchedFp32();

     /**
     * @brief Create a Strided Batched Matmul Fp32 object
     */
    void createXGEMMStridedBatchedFp32();

    /**
     * @brief Create a BatchNorm Mask Fp32 object
     */
    void createBatchNormMaskFp32();

    /**
     * @brief Create a BatchNorm Mask + ReLU Fp32 object
     */
    void createBatchNormMaskReluFp32();

    /**
     * @brief Create a BatchNorm Mask + Mish Fp32 object
     */
    void createBatchNormMaskMishFp32();

    /**
     * @brief Create a BatchNorm Mask + Mish + Scale8 Fp32 object
     */
    void createBatchNormMaskMishScale8Fp32();

    /**
     * @brief Create a Global Average Pool Fp32 object
     */
    void createGlobalPoolingChannelsFp32();

    /**
     * @brief Create a Value Head Pool Channels Fp32 object
     */
    void createValueHeadPoolingChannelsFp32();

    /**
     * @brief Create a Sum Channels Fp32 object
     */
    void createSumChannelsFp32();

    /**
     * @brief Create a Add Channel Bias NCHW Fp32 object
     */
    void createAddChannelBiasNCHWFp32();

    /**
     * @brief Create a Add Channel Bias NC Fp32 object
     */
    void createAddChannelBiasNCIdentityFp32();

    /**
     * @brief Create a Add Channel Bias NC + ReLU Fp32 object
     */
    void createAddChannelBiasNCReluFp32();

    /**
     * @brief Create a Add Channel Bias NC + Mish Fp32 object
     */
    void createAddChannelBiasNCMishFp32();

    /**
     * @brief Create a Add Channel Bias NC + Mish + Scale8 Fp32 object
     */
    void createAddChannelBiasNCMishScale8Fp32();
    
    /**
     * @brief Create a Extract Channel 0 NCHW Fp32 object
     */
    void createExtractChannel0NCHWFp32();
  };

}


#endif