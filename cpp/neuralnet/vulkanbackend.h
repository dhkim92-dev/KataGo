/**
 * @file vulkanbackend.cpp
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

/**
 * @brief Vulkan Compute  Pipeline structure
 * 
 */

namespace KatagoVulkan {

  /**
   * @brief Will be used to tune various parameters for different devices
   *        Not implemented yet. support in future. 
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

  /**
   * @brief Batch Normalization Mask Fp32 Push Constant Parameters 
   * @param batchSize
   * @param numChannels
   * @param nnYLen
   */
  struct BatchNormMaskFp32Params {
    uint32_t batchSize;
    uint32_t numChannels;
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
    // Pipeline matmulTiledChw4x4x32Fp32;

    // Batch Normalization pipelines
    // note that prediction phase does not need batch normalization operation separately
    // as the parameters can be folded into convolution scale and bias.
    Pipeline batchNormMaskFp32;
    Pipeline batchNormMaskReluFp32;
    Pipeline batchNormMaskMishFp32;

    // Pooling pipelines
    Pipeline globalPoolingChannelsFp32;
    Pipeline valueHeadPoolingChannelsFp32;
    
    // Element wise operations
    Pipeline sumChannelsFp32;
    Pipeline addChannelBiasNCHWFp32;
    Pipeline addChannelBiasNCReluFp32;
    Pipeline addChannelBiasNCMishFp32;
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
    void createConv2d3x3BnFp32();

    /**
     * @brief Create a Conv2d3x3 Bn + ReLU Activation fused Fp32 objects.
     */
    void createConv2d3x3BnReluFp32();
    /**
     * @brief Create a Conv2d3x3 Bn Mish Fp32 object
     */
    void createConv2d3x3BnMishFp32();

    /**
     * @brief Create a Conv2d5x5 Bn + Identity Activation fused Fp32 objects.
     */
    void createConv2d5x5BnFp32();

    /**
     * @brief Create a Conv2d5x5 Bn + ReLU Activation fused Fp32 objects.
     */
    void createConv2d5x5BnReluFp32();

    /**
     * @brief Create a Conv2d5x5 Bn + ReLU Activation fused Fp32 objects.
     */
    void createConv2d5x5BnReluFp32();

    /**
     * @brief Create a Conv2d5x5 Bn Mish Fp32 object
     */
    void createConv2d5x5BnMishFp32();

    /**
     * @brief Create a Add Point Wise Fp32 object
     */
    void createAddPointWiseFp32();

    /**
     * @brief Create a Matmul Fp32 object
     */
    void createMatmulFp32();

    /**
     * @brief Create a Matmul Tiled 4x4x32 Fp32 object
     */
    void createMatmulTiled4x4x32Fp32();

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
     * @brief Create a Add Channel Bias NC + ReLU Fp32 object
     */
    void createAddChannelBiasNCReluFp32();

    /**
     * @brief Create a Add Channel Bias NC + Mish Fp32 object
     */
    void createAddChannelBiasNCMishFp32();
    
    /**
     * @brief Create a Extract Channel 0 NCHW Fp32 object
     */
    void createExtractChannel0NCHWFp32();
  };

}


#endif