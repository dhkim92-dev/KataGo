/**
 * @file vulkanbackend.h
 * @author Dohoon Kim(https://github.com/dhkim92-dev, dhkim92-dev@gmail.com, https://www.dohoon-kim.kr)
 * @brief Vulkan backend for Neural Net evaluation
 */
#ifdef USE_VULKAN_BACKEND
#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkanshaders.h"


#define FLOAT16_SIZE_IN_BYTES 2

static void checkBufferSize(int batchSize, int nnXLen, int nnYLen, int channels) {
  if((int64_t)batchSize * nnXLen * nnYLen * channels >= (int64_t)1 << 31)
    throw StringError("Batch size too large, resulting GPU buffers might exceed 2^31 entries which is not currently supported");
}

template<typename T>
static size_t byteSizeofVectorContents(const typename std::vector<T>& vec) {
  return sizeof(T) * vec.size();
}

struct ComputeHandleInternal {
  const ComputeContext* context;
  const VulkanDevice* vulkanDevice;
  VkDevice device;
  VkQueue queue;
  bool usingFP16Storage = false;
  bool usingFp16Compute = false;
  bool usingFP16TensorCores = false;
  bool usingFP16TensorCoresFor1x1 = false;

  ComputeHandleInternal(ComputeContext* ctx, int gpuIdx, bool inputsUseNHWC, bool useNHWC);
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
 * @param handle: Compute handle
 * @param blocks: vector of blocks
 * @param commandBuffers: command buffers for all blocks
 * @param numBlocks: number of blocks
 * @param trunkNumChannels: number of channels in trunk
 * @param nnXLen: neural net X length
 * @param nnYLen: neural net Y length
 */
struct BlockStack {
  ComputeHandleInternal *handle;
  std::vector<std::pair<int, unique_ptr_void>> blocks;
  const int numBlocks;
  const int trunkNumChannels;
  const int nnXLen;
  const int nnYLen;

  std::vector<VkCommandBuffer> commandBuffers;

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

  void record(
    // VkCommandBuffer commandBuffer,
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum
  );

  void apply(
    int batchSize,
    ScratchBuffers *scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum
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
  struct VulkanTuneParams {

  };

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

  struct StridedBatchedMatmulFp32Params {
    uint32_t M;  
    uint32_t N;
    uint32_t K;  
    uint32_t inputStride;
    uint32_t filterStride;
    uint32_t outputStride;
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

  struct Pipeline {
    VkPipelineLayout layout;
    VkPipeline pipeline;
    VkDescriptorSetLayout descriptorSetLayout;
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
    // Pipeline conv2d3x3BnFp32;
    // Pipeline conv2d3x3BnReluFp32;
    // Pipeline conv2d3x3BnMishFp32;
    // Pipeline conv2d5x5BnFp32;
    // Pipeline conv2d5x5BnReluFp32;
    // Pipeline conv2d5x5BnMishFp32;
    Pipeline addPointWiseFp32;  // operation for skipping connections

    // Pipeline for matrix multiplication
    Pipeline matmulFp32; 

    // note that conv1x1 can be implemented as matmul operation
    Pipeline stridedBatchedMatmulFp32;

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

    ComputePipelines(VkDevice device_);
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
      Pipeline &outPipeline
    );
    /**
     * @brief Create a Conv2d Fp32 object
    */
    void createConv2dFp32();

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
     * @brief Create a Strided Batched Matmul Fp32 object
     */
    void createStridedBatchedMatmulFp32();

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