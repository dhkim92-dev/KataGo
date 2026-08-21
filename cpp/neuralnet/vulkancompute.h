/**
 * @file vulkancompute.h
 * @author Dohoon Kim(https://github.com/dhkim92-dev)
 * @brief define tune params for each shaders and host/device side operations
 */
#ifdef USE_VULKAN_BACKEND
#ifndef __VULKAN_COMPUTE_H__
#define __VULKAN_COMPUTE_H__

#include <cstdint>
#include "vulkanhelpers.h"

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

struct VulkanTuneParams {
  ConvTuneParams conv3x3;
  ConvTuneParams conv5x5;
  XgemmTuneParams xgemm;
  XgemmDirectTuneParams xgemmDirect;

  VulkanTuneParams(const VulkanTuneParams& other) = default;
  VulkanTuneParams& operator=(const VulkanTuneParams& other) = default;

  bool isValid() const;
  bool operator==(const VulkanTuneParams& other) const;
  bool operator!=(const VulkanTuneParams& other) const { return !(*this == other); }

  static void save(const std::string& filename, const VulkanTuneParams& config);
  static VulkanTuneParams load(const std::string& filename);

  VulkanTuneParams() {
    // conv3x3 = ConvTuneParams{
    //   .inTileYSize = 6,
    //   .inTileXSize = 6,
    //   .outTileYSize = 4,
    //   .outTileXSize = 4,
    //   .inputTransformLocalXSize = 4,
    //   .inputTransformLocalYSize = 2,
    //   .outputTransformLocalXSize = 8,
    //   .outputTransformLocalYSize = 2,
    //   .outputTransformLocalZSize = 2
    // };
    // conv5x5 = ConvTuneParams{
    //   .inTileYSize = 6,
    //   .inTileXSize = 6,
    //   .outTileYSize = 2,
    //   .outTileXSize = 2,
    //   .inputTransformLocalXSize = 4,
    //   .inputTransformLocalYSize = 2,
    //   .outputTransformLocalXSize = 8,
    //   .outputTransformLocalYSize = 2,
    //   .outputTransformLocalZSize = 2
    // };
    // xgemm = XgemmTuneParams{
    //   .MDIMC = 16,
    //   .NDIMC = 16,
    //   .MWG = 64,
    //   .NWG = 64,
    //   .KWG = 16,
    //   .MDIMA = 16,
    //   .NDIMB = 16
    // };

    // xgemmDirect = XgemmDirectTuneParams{
    //   .WGD = 32,
    //   .MDIMCD = 8,
    //   .NDIMCD = 8,
    //   .MDIMAD = 8,
    //   .NDIMBD = 8,
    //   .KWID = 2,
    //   .PADA = 1,
    //   .PADB = 1
    // };

    conv3x3 = ConvTuneParams();
    conv3x3.inTileYSize = 6;
    conv3x3.inTileXSize = 6;
    conv3x3.outTileYSize = 4;
    conv3x3.outTileXSize = 4;
    conv3x3.inputTransformLocalXSize = 128;
    conv3x3.inputTransformLocalYSize = 2;
    conv3x3.outputTransformLocalXSize = 8;
    conv3x3.outputTransformLocalYSize = 4;
    conv3x3.outputTransformLocalZSize = 8;

    conv5x5 = ConvTuneParams();
    conv5x5.inTileYSize = 6;
    conv5x5.inTileXSize = 6;
    conv5x5.outTileYSize = 2;
    conv5x5.outTileXSize = 2;
    conv5x5.inputTransformLocalXSize = 128;
    conv5x5.inputTransformLocalYSize = 2;
    conv5x5.outputTransformLocalXSize = 8;
    conv5x5.outputTransformLocalYSize = 2;
    conv5x5.outputTransformLocalZSize = 2;

    xgemm = XgemmTuneParams();
    xgemm.MDIMC = 16;
    xgemm.NDIMC = 16;
    xgemm.MWG = 64;
    xgemm.NWG = 64;
    xgemm.KWG = 16;
    xgemm.MDIMA = 16;
    xgemm.NDIMB = 16;

    xgemmDirect = XgemmDirectTuneParams();
    xgemmDirect.WGD = 32;
    xgemmDirect.MDIMCD = 8;
    xgemmDirect.NDIMCD = 8;
    xgemmDirect.MDIMAD = 8;
    xgemmDirect.NDIMBD = 8;
    xgemmDirect.KWID = 2;
    xgemmDirect.PADA = 1;
    xgemmDirect.PADB = 1;
  }
};


namespace vkcompute {

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
