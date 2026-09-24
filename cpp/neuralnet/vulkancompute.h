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

  void dispatchPipeline(
    ComputeHandleInternal* benchmarkHandle,
    const VulkanDevice* device,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const void* pushConstants,
    uint32_t pushConstantSize,
    uint32_t workgroupCountX,
    uint32_t workgroupCountY,
    uint32_t workgroupCountZ,
    const char* benchmarkName
  );

  void doBatchNormMask(
    ComputeHandleInternal* handle,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* mergedScale,
    VulkanBuffer* mergedBias,
    VulkanBuffer* mask,
    int batchSize,
    int numChannels,
    int spatialSize,
    int maskSpatialStride,
    int channelsPadded,
    const char* benchmarkName
  );

  void doMatBiasNC(
    ComputeHandleInternal* handle,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    VulkanBuffer* input,
    VulkanBuffer* bias,
    int batchSize,
    int numChannels
  );

  void convertNCHWToNHWC(
    ComputeHandleInternal* handle,
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
    ComputeHandleInternal* handle,
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
    ComputeHandleInternal* handle,
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
    ComputeHandleInternal* handle,
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

  void transformerScaleDotProductCooperative(
    ComputeHandleInternal* handle,
    const VulkanDevice* device,
    const Pipeline* attentionPipeline,
    VkCommandBuffer cb,
    VkDescriptorSet attentionDescriptorSet,
    const Pipeline* nchwToNhwcPipeline,
    VkDescriptorSet nchwToNhwcDescriptorSet,
    const Pipeline* nhwcToNchwPipeline,
    VkDescriptorSet nhwcToNchwDescriptorSet,
    const VulkanBuffer* packedQKV,
    VulkanBuffer* output,
    VulkanBuffer* nhwcQKV,
    VulkanBuffer* nhwcOutput,
    VulkanBuffer* mask,
    VulkanBuffer* ropeCosTable,
    VulkanBuffer* ropeSinTable,
    int batchSize,
    int numHeads,
    int numKVHeads,
    int qHeadDim,
    int vHeadDim,
    int seqLen,
    int logicalSpatialSize,
    int spatialStride,
    int qkvChannels,
    int outputChannels,
    int qkvChannelsPadded,
    int outputChannelsPadded,
    int qTotalDim,
    int kTotalDim,
    float scale,
    bool useRope,
    bool learnableRope,
    int ropeNumPairs,
    const vk_shader::tune::TransformerTuneParams& tuneParams,
    VkResult* result
  );

  void extractChannel0(
    ComputeHandleInternal* handle,
    const VulkanDevice* device,
    const Pipeline* extractPipeline,
    VkCommandBuffer& commandBuffer,
    VkDescriptorSet& extractDescriptorSet,
    const VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* nhwcScratch,
    int batchSize,
    int numInputChannels,
    int nhwcSpatialSize,
    int nchwSpatialStride,
    int logicalSpatialSize,
    int channelsPadded,
    bool useNHWC,
    bool begin = true
  );

  void convertNHWCMatrixToNCHW(
    ComputeHandleInternal* handle,
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
    ComputeHandleInternal* handle,
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
    int aRowStride,
    int cRowStride,
    const vk_shader::tune::HGemmCooperativeMatrixNHWCTuneParams& params,
    VkResult* result,
    int aBatchStride = 0,
    int bBatchStride = 0,
    int cBatchStride = 0
  );

  void doHgemmCooperativeMatrixNHWC(
    ComputeHandleInternal* handle,
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
    int aRowStride,
    int cRowStride,
    const vk_shader::tune::HGemmCooperativeMatrixNCHWTuneParams& params,
    VkResult* result,
    int aBatchStride = 0,
    int bBatchStride = 0,
    int cBatchStride = 0
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

  std::vector<float> convWeightsToNHWC1x1Gemm(
    const std::vector<float>& weights,
    uint32_t inChannels,
    uint32_t outChannels,
    uint32_t kSize,
    uint32_t nSize
  );

  void convInputsToWinogradDomain(
    ComputeHandleInternal* handle,
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
    VkResult *result,
    bool useNHWC = false
  ); 

  void convInputToWinogradDomainBnActMask(
    ComputeHandleInternal* handle,
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
    VkResult *result,
    bool useNHWC = false
  );

  void winogradOutputToSpatialDomain(
    ComputeHandleInternal* handle,
    const VulkanDevice* device,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* convWorkspace2,
    VulkanBuffer* output,
    uint32_t nnYLen, uint32_t nnXLen, uint32_t xyStride,
    uint32_t batchSize, uint32_t numTilesY, uint32_t numTilesX, uint32_t batchNumTilesPadMultiple,
    uint32_t outChannels, uint32_t outChannelsPadMultiple,
    VkResult *result,
    bool useNHWC = false
  ); 

  void xgemmBatched(
    ComputeHandleInternal* handle,
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
    ComputeHandleInternal* handle,
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
    ComputeHandleInternal* handle,
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
    ComputeHandleInternal* handle,
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
    ComputeHandleInternal* handle,
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
    ComputeHandleInternal* handle,
    const VulkanDevice* device,
    const vk_shader::tune::VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    const VkCommandBuffer cb,
    const VkDescriptorSet descriptorSet,
    const VulkanBuffer* input,
    VulkanBuffer* nhwcInput,
    const VulkanBuffer* packedFilter,
    VulkanBuffer* nhwcOutput,
    VulkanBuffer* output,
    const Pipeline* nchwToNhwcPipeline,
    const VkDescriptorSet nchwToNhwcDescriptorSet,
    const Pipeline* nhwcToNchwPipeline,
    const VkDescriptorSet nhwcToNchwDescriptorSet,
    int batchSize,
    int hwSize,
    int logicalSpatialSize,
    int ffnSize,
    int cSize,
    int packedOCSize,
    VkResult* result
  );

  void performAddChannelBiases(
    ComputeHandleInternal* handle,
    VkCommandBuffer& commandBuffer,
    VkDescriptorSet& descriptorSet,
    VkDescriptorSet& nchwToNhwcDescriptorSet,
    VkDescriptorSet& nhwcToNchwDescriptorSet,
    VulkanBuffer* input,
    VulkanBuffer* bias,
    int ncSize,
    int cSize,
    int nchwSpatialStride,
    VulkanBuffer* nhwcScratch,
    bool begin = true
  );

  void performAddPointWise(
    ComputeHandleInternal* handle,
    VkCommandBuffer& commandBuffer,
    VkDescriptorSet& descriptorSet,
    VulkanBuffer* acc,
    VulkanBuffer* value,
    int totalSize,
    bool begin = true
  );

  void performGpoolMask(
    ComputeHandleInternal* handle,
    VkCommandBuffer& commandBuffer,
    VkDescriptorSet& descriptorSet,
    VkDescriptorSet& nchwToNhwcDescriptorSet,
    VulkanBuffer* gpoolConvOut,
    VulkanBuffer* gpoolConcat,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    int batchSize,
    int gpoolChannels,
    int nnXYLen,
    VulkanBuffer* nhwcScratch,
    VkResult* result,
    bool begin = true
  );

  void performValueHeadPool(
    ComputeHandleInternal* handle,
    VkCommandBuffer& commandBuffer,
    VkDescriptorSet& descriptorSet,
    VkDescriptorSet& nchwToNhwcDescriptorSet,
    VulkanBuffer* gpoolConvOut,
    VulkanBuffer* gpoolConcat,
    VulkanBuffer* maskSum,
    VulkanBuffer* nhwcScratch,
    int batchSize,
    int gPoolChannels,
    int nnXYLen,
    bool begin = true
  );

  void computeMaskSums(
    ComputeHandleInternal* handle,
    VkCommandBuffer& commandBuffer,
    VkDescriptorSet& descriptorSet,
    int batchSize,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    bool begin = true
  );

  struct SpatialRMSNormSizing {
    int numCHWWorkgroups;   // workgroups per batch element for pass 1
    int tilesPerGroupPass1; // tiles per group for pass 1
    int tilesPerGroupPass2; // tiles per group for pass 2 (reduces numCHWWorkgroups values -> 1)
  };
  
  SpatialRMSNormSizing computeSpatialRMSNormSizing(int tileSize, int chwSize);

  void doSwiGLU(
    ComputeHandleInternal* handle,
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
