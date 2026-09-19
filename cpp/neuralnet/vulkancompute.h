/**
 * @file vulkancompute.h
 * @author Dohoon Kim(https://github.com/dhkim92-dev)
 * @brief define tune params for each shaders and host/device side operations
 */
#ifdef USE_VULKAN_BACKEND
#ifndef __VULKAN_COMPUTE_H__
#define __VULKAN_COMPUTE_H__

#include <cstdint>
#include "../neuralnet/vulkanshaders.h"
#include "../neuralnet/vulkanhelpers.h"

struct ComputeHandleInternal;

namespace vkcompute {

  void convertNCHWToNHWC(
    const VulkanDevice* device,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* input,
    VulkanBuffer* output,
    int batchSize,
    int channels,
    int spatialSize,
    int spatialStride,
    int logicalSpatialSize,
    VkResult* result
  );

  void convertNHWCToNCHW(
    const VulkanDevice* device,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* input,
    VulkanBuffer* output,
    int batchSize,
    int channels,
    int spatialSize,
    int spatialStride,
    int logicalSpatialSize,
    VkResult* result
  );

  void transformerApplyRoPE(
    const VulkanDevice* device,
    const Pipeline* ropePipeline,
    VkCommandBuffer cb,
    VkDescriptorSet ropeDescriptorSet,
    const Pipeline* nchwToNhwcPipeline,
    VkDescriptorSet nchwToNhwcDescriptorSet,
    const Pipeline* nhwcToNchwPipeline,
    VkDescriptorSet nhwcToNchwDescriptorSet,
    VulkanBuffer* input,
    VulkanBuffer* nhwcScratch,
    VulkanBuffer* cosTable,
    VulkanBuffer* sinTable,
    int batchSize,
    int numHeads,
    int numKVHeads,
    int headDim,
    int seqLen,
    int numPairs,
    int learnableRope,
    int regionOffset,
    int batchStride,
    int channels,
    int channelsPadded,
    int spatialStride,
    int logicalSpatialSize,
    bool useNHWC,
    VkResult* result
  );

  void transformerRMSNorm(
    const VulkanDevice* device,
    const Pipeline* rmsNormPipeline,
    VkCommandBuffer cb,
    VkDescriptorSet rmsNormDescriptorSet,
    const Pipeline* nchwToNhwcPipeline,
    VkDescriptorSet nchwToNhwcDescriptorSet,
    const Pipeline* nhwcToNchwPipeline,
    VkDescriptorSet nhwcToNchwDescriptorSet,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* nhwcInput,
    VulkanBuffer* nhwcOutput,
    VulkanBuffer* weight,
    VulkanBuffer* beta,
    VulkanBuffer* mask,
    int batchSize,
    int channels,
    int spatialSize,
    int spatialStride,
    int logicalSpatialSize,
    int channelsPadded,
    float epsilon,
    const vk_shader::tune::TransformerRMSNormTuneParms& tuneParams,
    bool useNHWC,
    VkResult* result
  );

  void extractChannel0(
    const VulkanDevice* device,
    const Pipeline* extractPipeline,
    VkCommandBuffer& commandBuffer,
    VkDescriptorSet& extractDescriptorSet,
    const Pipeline* nchwToNhwcPipeline,
    VkDescriptorSet& nchwToNhwcDescriptorSet,
    const VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* nhwcScratch,
    int batchSize,
    int numInputChannels,
    int nhwcSpatialSize,
    int nchwSpatialStride,
    int logicalSpatialSize,
    bool useNHWC,
    bool begin = true
  );

  void im2colNHWC(
    const VulkanDevice* device,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* input,
    VulkanBuffer* output,
    const VulkanBuffer* scale,
    const VulkanBuffer* bias,
    const VulkanBuffer* mask,
    int batchSize,
    int xSize,
    int ySize,
    int logicalSpatialSize,
    int spatialSize,
    int maskSpatialStride,
    int channels,
    int channelsPadded,
    int kSize,
    int logicalKSize,
    int convYSize,
    int convXSize,
    int activation,
    VkResult* result
  );

  void convertNHWCMatrixToNCHW(
    const VulkanDevice* device,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* input,
    VulkanBuffer* output,
    int batchSize,
    int channels,
    int logicalSpatialSize,
    int spatialSize,
    int outputSpatialStride,
    int matrixChannels,
    VkResult* result
  );

  void doHgemmCooperativeMatrixNHWC(
    const VulkanDevice* device,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* A,
    const VulkanBuffer* B,
    VulkanBuffer* C,
    int batchSize,
    int M,
    int N,
    int K,
    const vk_shader::tune::HGemmCooperativeMatrixTuneParams& params,
    VkResult* result
  );

  void doHgemmCooperativeMatrixNHWC(
    const VulkanDevice* device,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* A,
    const VulkanBuffer* B,
    VulkanBuffer* C,
    int batchSize,
    int M,
    int N,
    int K,
    const vk_shader::tune::HGemmCooperativeMatrixNCHWTuneParams& params,
    VkResult* result
  );

  void winogradFilterTransform3x3_2x2(float& a0, float& a1, float& a2, float& a3);

  void winogradFilterTransform3x3_4x4(float& a0, float& a1, float& a2, float& a3, float& a4, float& a5);

  void winogradFilterTransform5x5_2x2(float &a0, float &a1, float &a2, float &a3, float &a4, float& a5);

  std::vector<float> convWeightsToWinogradDomain(
    const std::vector<float>& weights,
    uint32_t inChannels,
    uint32_t inChannelsPadded,
    uint32_t outChannels,
    uint32_t outChannelsPadded,
    uint32_t convY,
    uint32_t convX,
    uint32_t inTileYSize,
    uint32_t inTileXSize
  );

  std::vector<float> convWeightsToNHWCIm2Col(
    const std::vector<float>& weights,
    uint32_t inChannels,
    uint32_t outChannels,
    uint32_t convY,
    uint32_t convX,
    uint32_t kSize,
    uint32_t nSize
  );

  void convInputsToWinogradDomain(
    const VulkanDevice* device,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* inputBuffer,
    VulkanBuffer* convWorkspace,
    uint32_t nnYLen,
    uint32_t nnXLen,
    uint32_t xyStride,
    uint32_t batchSize, uint32_t numTilesY, uint32_t numTilesX, uint32_t batchNumTilesPadMultiple,
    uint32_t inChannels, uint32_t inChannelsPaddedMultiple,
    uint32_t convSize,
    VkResult *result
  ); 

  void convInputToWinogradDomainBnActMask(
    const VulkanDevice* device,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* inputBuffer,
    VulkanBuffer* convWorkspace,
    const VulkanBuffer* bnScale,
    const VulkanBuffer* bnBias,
    const VulkanBuffer* mask,
    uint32_t nnYLen,
    uint32_t nnXLen,
    uint32_t xyStride,
    uint32_t batchSize, uint32_t numTilesY, uint32_t numTilesX, uint32_t batchNumTilesPadMultiple,
    uint32_t inChannels, uint32_t inChannelsPaddedMultiple,
    uint32_t convSize,
    VkResult *result
  );

  void winogradOutputToSpatialDomain(
    const VulkanDevice* device,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* convWorkspace2,
    VulkanBuffer* output,
    uint32_t nnYLen, uint32_t nnXLen, uint32_t xyStride,
    uint32_t batchSize, uint32_t numTilesY, uint32_t numTilesX, uint32_t batchNumTilesPadMultiple,
    uint32_t outChannels, uint32_t outChannelsPadMultiple,
    VkResult *result
  ); 

  void xgemmBatched(
    const VulkanDevice* device,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    uint32_t M, uint32_t N, uint32_t K,
    const VulkanBuffer* A,
    const VulkanBuffer* B,
    VulkanBuffer* C,
    uint32_t numBatchElts,
    VkResult *result
  );

  void xgemmStridedBatchedNN(
    const VulkanDevice* device,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    const VkCommandBuffer cb,
    const VkDescriptorSet descriptorSet,
    const uint32_t kSizeM, const uint32_t kSizeN, const uint32_t kSizeK,
    const uint32_t aStride, const uint32_t bStride, const uint32_t cStride,
    const VulkanBuffer* A, const VulkanBuffer* B, VulkanBuffer* C,
    uint32_t numBatchElts,
    VkResult *result
  );

  void batchedXGemmDirect_MK_NK_MN(
    const VulkanDevice* device,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    const VkCommandBuffer cb,
    const VkDescriptorSet descriptorSet,
    const int M, const int N, const int K,
    const VulkanBuffer* A,
    const VulkanBuffer* B,
    VulkanBuffer* C,
    int numBatchElts,
    VkResult* result
  );

  void doHgemmCooperativeMatrix(
    const VulkanDevice* device,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    const VkCommandBuffer cb,
    const VkDescriptorSet descriptorSet,
    const VulkanBuffer* A, const VulkanBuffer* B, VulkanBuffer* C,
    const int batchSize,
    const int M, const int N, const int K,
    VkResult *result
  );

  void doHgemmCooperativeMatrixNCHW(
    const VulkanDevice* device,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    const VkCommandBuffer cb,
    const VkDescriptorSet descriptorSet,
    const VulkanBuffer* A, const VulkanBuffer* B, VulkanBuffer* C,
    const int batchSize,
    // Matrix dimensions are M=HW, N=OC, K=C for the NCHW layout.
    const int M, const int N, const int K,
    VkResult *result
  );

  void doTransformerDualGemmSwiGLU(
    const VulkanDevice* device,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    const VkCommandBuffer cb,
    const VkDescriptorSet descriptorSet,
    const VulkanBuffer* input,
    const VulkanBuffer* packedFilter,
    VulkanBuffer* output,
    int batchSize,
    int hwSize,
    int ffnSize,
    int cSize,
    int packedOCSize,
    VkResult* result
  );

  struct SpatialRMSNormSizing {
    int numCHWWorkgroups;   // workgroups per batch element for pass 1
    int tilesPerGroupPass1; // tiles per group for pass 1
    int tilesPerGroupPass2; // tiles per group for pass 2 (reduces numCHWWorkgroups values -> 1)
  };
  
  SpatialRMSNormSizing computeSpatialRMSNormSizing(int tileSize, int chwSize);

  void doSwiGLU(
    const VulkanDevice* device,
    const VkCommandBuffer& cb,
    const VkDescriptorSet& descriptorSet,
    const Pipeline& pipeline,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    VulkanBuffer* mainProj,
    VulkanBuffer* gateProj,
    VulkanBuffer* output,
    int totalSize,
    int packedInputBatchStride = 0,
    int outputBatchStride = 0
  );
}

#endif
#endif // __VULKAN_COMPUTE_H__
