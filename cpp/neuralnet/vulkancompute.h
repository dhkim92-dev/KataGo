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
}

#endif
#endif // __VULKAN_COMPUTE_H__
