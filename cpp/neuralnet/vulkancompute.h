/**
 * @file vulkancompute.h
 * @author dhkim92-dev
 * @brief define tune params for each shaders and host/device side operations
 */
#ifdef USE_VULKAN_BACKEND
#ifndef __VULKAN_COMPUTE_H__
#define __VULKAN_COMPUTE_H__

#include <cstdint>
#include <algorithm>
#include <functional>
#include <unordered_set>
#include "../neuralnet/vulkanhelpers.h"

struct ModelDesc;

struct AttnDims {
  int qHeadDim;
  int vHeadDim;

  bool operator==(const AttnDims& other) const {
    return qHeadDim == other.qHeadDim && vHeadDim == other.vHeadDim;
  }
};

struct AttnDimsHash {
  size_t operator()(const AttnDims& dims) const {
    size_t h1 = std::hash<int>{}(dims.qHeadDim);
    size_t h2 = std::hash<int>{}(dims.vHeadDim);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

struct GPoolTuneParams {
  uint32_t XYSTRIDE;
  uint32_t CHANNELSTRIDE;
  uint32_t BATCHSTRIDE;
};

struct AddPointwiseTuneParams {
  uint32_t ELTS_PER_THREAD=16;
  uint32_t LOCAL_SIZE=256;
};

struct AddChannelBiasesNCHWTuneParams {
  int XY_ELTS_PER_THREAD = 1;
  int NC_ELTS_PER_THREAD = 1;
};

struct ConvTuneParams {
  uint32_t inTileYSize;
  uint32_t inTileXSize;
  uint32_t outTileYSize;
  uint32_t outTileXSize;
  uint32_t inputTransformLocalXSize;
  uint32_t inputTransformLocalYSize;
  uint32_t outputTransformLocalXSize;
  uint32_t outputTransformLocalYSize;
  uint32_t outputTransformLocalZSize;
};

struct XgemmTuneParams {
  uint32_t MDIMC;
  uint32_t NDIMC;
  uint32_t MWG;
  uint32_t NWG;
  uint32_t KWG;
  uint32_t MDIMA;
  uint32_t NDIMB;
};

struct XgemmDirectTuneParams {
  uint32_t WGD;
  uint32_t MDIMCD;
  uint32_t NDIMCD;
  uint32_t MDIMAD;
  uint32_t NDIMBD;
  uint32_t KWID;
  uint32_t PADA;
  uint32_t PADB;
};

struct RMSNormTuneParams {
  uint32_t WG_C_SIZE;
  uint32_t WG_XY_SIZE;
  uint32_t C_PER_THREAD;
};

struct SpatialRMSNormSumSqTuneParams {
  uint32_t TILE_SIZE;
};

struct SpatialRMSNormReduceTuneParams {
  uint32_t TILE_SIZE;
};

struct SpatialRMSNormApplyTuneParams {
  uint32_t APPLY_ELTS_PER_THREAD;
};

struct TransformerTuneParams {
  uint32_t USE_TILED_ATTN;
};

struct ScaleDotProductAttentionTuneParams {
  uint32_t ATTN_BLOCK_Q;
  uint32_t ATTN_BLOCK_KV;
  uint32_t Q_PER_THREAD;
};

struct ScaleDotProductAttentionNaiveTuneParams {

};

struct VulkanTuneParams {
  AddPointwiseTuneParams pointwise;
  GPoolTuneParams gPool;
  ConvTuneParams conv3x3;
  ConvTuneParams conv5x5;
  XgemmTuneParams xgemm;
  XgemmDirectTuneParams xgemmDirect;
  RMSNormTuneParams rmsNorm;
  TransformerTuneParams transformer;
  SpatialRMSNormSumSqTuneParams spatialRMSNormSumSq;
  SpatialRMSNormReduceTuneParams spatialRMSNormReduce;
  SpatialRMSNormApplyTuneParams spatialRMSNormApply;
  ScaleDotProductAttentionTuneParams scaleDotProductAttention;
  ScaleDotProductAttentionNaiveTuneParams scaleDotProductAttentionNaive;
  AddChannelBiasesNCHWTuneParams addChannelBiasesNCHW;

  VulkanTuneParams(const VulkanTuneParams& other) = default;
  VulkanTuneParams& operator=(const VulkanTuneParams& other) = default;

  bool isValid() const;
  bool operator==(const VulkanTuneParams& other) const;
  bool operator!=(const VulkanTuneParams& other) const { return !(*this == other); }

  static void save(const std::string& filename, const VulkanTuneParams& config);
  static VulkanTuneParams load(const std::string& filename);

  VulkanTuneParams();
};


namespace vkcompute {

  std::unordered_set<AttnDims, AttnDimsHash> getAttentionDims(const ModelDesc& modelDesc);

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

  void convInputsToWinogradDomain(
    const VulkanDevice* device,
    const VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* inputBuffer,
    VulkanBuffer* convWorkspace,
    uint32_t nnYLen,
    uint32_t nnXLen,
    uint32_t batchSize, uint32_t numTilesY, uint32_t numTilesX, uint32_t batchNumTilesPadMultiple,
    uint32_t inChannels, uint32_t inChannelsPaddedMultiple,
    uint32_t convSize,
    VkResult *result
  ); 

  void convInputToWinogradDomainBnActMask(
    const VulkanDevice* device,
    const VulkanTuneParams& tuneParams,
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
    uint32_t batchSize, uint32_t numTilesY, uint32_t numTilesX, uint32_t batchNumTilesPadMultiple,
    uint32_t inChannels, uint32_t inChannelsPaddedMultiple,
    uint32_t convSize,
    VkResult *result
  );

  void winogradOutputToSpatialDomain(
    const VulkanDevice* device,
    const VulkanTuneParams& tuneParams,
    const Pipeline* pipeline,
    VkCommandBuffer cb,
    VkDescriptorSet descriptorSet,
    const VulkanBuffer* convWorkspace2,
    VulkanBuffer* output,
    uint32_t nnYLen, uint32_t nnXLen,
    uint32_t batchSize, uint32_t numTilesY, uint32_t numTilesX, uint32_t batchNumTilesPadMultiple,
    uint32_t outChannels, uint32_t outChannelsPadMultiple,
    uint32_t convSize,
    VkResult *result
  ); 

  void xgemmBatched(
    const VulkanDevice* device,
    const VulkanTuneParams& tuneParams,
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
    const VulkanTuneParams& tuneParams,
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
    const VulkanTuneParams& tuneParams,
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
}

#endif
#endif // __VULKAN_COMPUTE_H__
