/**
 * @file vulkancompute.cpp
 * @author Dohoon Kim(https://github.com/dhkim92-dev)
 * @brief set of kernel calls or host side operations that used frequently on vulkanbackend.cpp
 */
#ifdef USE_VULKAN_BACKEND
#include <cassert>
#include <vector>
#include <iostream>
#include "vulkanbackend.h"
#include "vulkancompute.h"

namespace vkcompute {

void winogradFilterTransform3x3_2x2(
  float& a0, float& a1, float& a2, float& a3
) {
  float z0 = a0, z1 = a1, z2 = a2, z3 = a3;
  a0 = z0;
  a1 = (z0 + z1 + z2) * 0.5;
  a2 = (z0 - z1 + z2) * 0.5;
  a3 = z2;
}

void winogradFilterTransform3x3_4x4(float& a0, float& a1, float& a2, float& a3, float& a4, float& a5) {
  float z0 = a0; float z1 = a1; float z2 = a2;
  a0 = 0.25f * z0;
  a1 = (float)( (1.0 / 6.0) * (-z0 - z1 - z2) );
  a2 = (float)( (1.0 / 6.0) * (-z0 + z1 - z2) );
  a3 = (float)( (1.0 / 24.0) * (z0 + 2.0*z1 + 4.0*z2) );
  a4 = (float)( (1.0 / 24.0) * (z0 - 2.0*z1 + 4.0*z2) );
  a5 = 1.0f * z2;
}

void winogradFilterTransform5x5_2x2(float &a0, float &a1, float &a2, float &a3, float &a4, float& a5) {
  float z0 = a0; float z1 = a1; float z2 = a2; float z3 = a3; float z4 = a4;
  a0 = 0.25f * z0;
  a1 = (float)( (1.0 / 6.0) * (-z0 - z1 - z2 - z3 - z4) );
  a2 = (float)( (1.0 / 6.0) * (-z0 + z1 - z2 + z3 - z4) );
  a3 = (float)( (1.0 / 24.0) * (z0 + 2.0*z1 + 4.0*z2 + 8.0*z3 + 16.0*z4) );
  a4 = (float)( (1.0 / 24.0) * (z0 - 2.0*z1 + 4.0*z2 - 8.0*z3 + 16.0*z4) );
  a5 = 1.0f * z4;
}

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
) {
  // weights order [ OC, IC, H ,W ]
  // transWeights order should be [ InTileY ][ InTileX ] [ InChannels ] [OutChannels ]
  constexpr int maxTileSize = 6;
  std::vector<float> transWeights(
    inTileYSize * inTileXSize * inChannelsPadded * outChannelsPadded,
    0.0f
  );

  for ( int oc = 0 ; oc < outChannelsPadded ; ++oc ) {
    for ( int ic = 0 ; ic < inChannelsPadded ; ic++ ) {
      float tmp[maxTileSize][maxTileSize];

      for ( int subY = 0 ; subY < convY ; ++subY ) {
        for ( int subX = 0 ; subX < convX ; ++subX ) {
          float w = 0.0f;
          if ( oc < outChannels && ic < inChannels ) {
            w = weights[((oc * inChannels + ic) * convY  + subY) * convX + subX];
          }
          tmp[subY][subX] = w;
        }
      }
      // std::printf("convy : %d convx : %d inTileYSize : %d inTileXSize : %d\n", convY, convX, inTileYSize, inTileXSize);
      if ( (convY == 3 && convX == 3) && (inTileYSize == 4 && inTileXSize == 4) ) {
        for( int subY = 0 ; subY < convY ; ++subY ) {
          winogradFilterTransform3x3_2x2(tmp[subY][0], tmp[subY][1], tmp[subY][2], tmp[subY][3]);
        }
        for ( int subX = 0 ; subX < 4 ; ++subX ) {
          winogradFilterTransform3x3_2x2(tmp[0][subX], tmp[1][subX], tmp[2][subX], tmp[3][subX]);
        }
      } else if ( (convY == 3 && convX == 3) && (inTileYSize == 6 && inTileXSize == 6) ) {
        for( int subY = 0 ; subY < convY ; ++subY ) {
          winogradFilterTransform3x3_4x4(tmp[subY][0], tmp[subY][1], tmp[subY][2], tmp[subY][3], tmp[subY][4], tmp[subY][5]);
        }
        for ( int subX = 0 ; subX < 6 ; ++subX ) {
          winogradFilterTransform3x3_4x4(tmp[0][subX], tmp[1][subX], tmp[2][subX], tmp[3][subX], tmp[4][subX], tmp[5][subX]);
        }
      } else if ( (convY == 5 && convX == 5) && (inTileYSize == 6 && inTileXSize == 6) ) {
        for ( int subY = 0 ; subY < convY ; ++subY ) {
          winogradFilterTransform5x5_2x2(tmp[subY][0], tmp[subY][1], tmp[subY][2], tmp[subY][3], tmp[subY][4], tmp[subY][5]);
        }
        // Legacy implementation kept for easy rollback.
        // for ( int subX = 0 ; subX < 4 ; ++subX ) {
        //   winogradFilterTransform5x5_2x2(tmp[0][subX], tmp[1][subX], tmp[2][subX], tmp[3][subX], tmp[4][subX], tmp[5][subX]);
        // }

        // Corrected implementation: transform all six columns of the 6x6 Winograd tile.
        for ( int subX = 0 ; subX < inTileXSize ; ++subX ) {
          winogradFilterTransform5x5_2x2(tmp[0][subX], tmp[1][subX], tmp[2][subX], tmp[3][subX], tmp[4][subX], tmp[5][subX]);
        }
      } else {
        throw StringError("Unsupported convolution kernel size for Winograd transformation: " + std::to_string(convY) + "x" + std::to_string(convX));
      }

      for ( int subY = 0 ; subY < inTileYSize ; ++subY ) {
        for ( int subX = 0 ; subX < inTileXSize ; ++subX ) {
          transWeights[((subY*inTileXSize + subX) * inChannelsPadded + ic) * outChannelsPadded + oc] = tmp[subY][subX];
        }
      }
    }
  }
  return transWeights;
}

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
) {
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, inputBuffer),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, convWorkspace)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for Winograd Input Transform", *result);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr);

  int outTileYSize = 2;
  int outTileXSize = 2;

  if ( convSize == 3 ) {
    outTileYSize = tuneParams.conv3x3.outTileYSize;
    outTileXSize = tuneParams.conv3x3.outTileXSize;
  } else {
    outTileYSize = tuneParams.conv5x5.outTileYSize;
    outTileXSize = tuneParams.conv5x5.outTileXSize;
  }

  const int batchNumTilesPadded = vk_helper::roundUpToMultipleInt(batchSize * numTilesY * numTilesX, batchNumTilesPadMultiple);
  const int inChannelsPadded = vk_helper::roundUpToMultipleInt(inChannels, inChannelsPaddedMultiple);

  auto params = vk_shader::push::WinogradInputTransformParams();
  params.batchSize = batchSize;
  params.nnYLen = nnYLen;
  params.nnXLen = nnXLen;
  params.numTilesY = (nnYLen + outTileYSize - 1) / outTileYSize;
  params.numTilesX = (nnXLen + outTileXSize - 1) / outTileXSize;
  params.inChannels = inChannels;
  params.inChannelsPadded = inChannelsPadded;
  params.ntxtySizePadded = batchNumTilesPadded;
  params.xyStride = xyStride;
  // };
  // std::printf("convInputsToWinogradDomain: batchSize = %d nnYLen = %d nnXLen = %d numTilesY = %d numTilesX = %d inChannels = %d inChannelsPadded = %d batchNumTilesPadded = %d\n",
    // params.batchSize, params.nnYLen, params.nnXLen, params.numTilesY, params.numTilesX, params.inChannels, params.inChannelsPadded, params.ntxtySizePadded
  // );

  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
  const uint32_t localSizeX = pipeline->localSizeX;
  const uint32_t localSizeY = pipeline->localSizeY;
  uint32_t wgCountX = static_cast<uint32_t>(vk_helper::roundUpToMultiple(batchNumTilesPadded, localSizeX)) / localSizeX;
  uint32_t wgCountY = static_cast<uint32_t>(vk_helper::roundUpToMultiple(inChannelsPadded, localSizeY)) / localSizeY;
  uint32_t wgCountZ = 1u;

  wgCountX = (wgCountX == 0) ? 1 : wgCountX;
  wgCountY = (wgCountY == 0) ? 1 : wgCountY;

  // std::printf(
  //   "convInputsToWinogradDomain: localSizes = %d,%d globalSizes = %d,%d groupCounts = %d,%d\n",
  //   localSizeX, localSizeY,
  //   wgCountX, wgCountY
  // );
  vkCmdDispatch(cb, wgCountX, wgCountY, wgCountZ);
  vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace);
}


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
  uint32_t batchSize, uint32_t numTilesY, uint32_t numTilesX, uint32_t batchNumTilesPadMultiple,
  uint32_t inChannels, uint32_t inChannelsPaddedMultiple,
  uint32_t convSize,
  VkResult *result
) {
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, inputBuffer),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, convWorkspace),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, bnScale),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 3, bnBias),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 4, mask)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for Winograd Input Transform", *result);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr);

  // int outTileSize = convSize == 3 ? 2 : 4;

  int outTileYSize = 2;
  int outTileXSize = 2;

  if ( convSize == 3 ) {
    outTileYSize = tuneParams.conv3x3.outTileYSize;
    outTileXSize = tuneParams.conv3x3.outTileXSize;
  } else {
    outTileYSize = tuneParams.conv5x5.outTileYSize;
    outTileXSize = tuneParams.conv5x5.outTileXSize;
  }

  const int batchNumTilesPadded = vk_helper::roundUpToMultipleInt(batchSize * numTilesY * numTilesX, batchNumTilesPadMultiple);
  const int inChannelsPadded = vk_helper::roundUpToMultipleInt(inChannels, inChannelsPaddedMultiple);

  auto params = vk_shader::push::WinogradInputTransformParams();
  params.batchSize = batchSize;
  params.nnYLen = nnYLen;
  params.nnXLen = nnXLen;
  params.numTilesY = (nnYLen + outTileYSize - 1) / outTileYSize;
  params.numTilesX = (nnXLen + outTileXSize - 1) / outTileXSize;
  params.inChannels = inChannels;
  params.inChannelsPadded = static_cast<uint32_t>(inChannelsPadded);
  params.ntxtySizePadded = static_cast<uint32_t>(batchNumTilesPadded);

  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
  const uint32_t localSizeX = pipeline->localSizeX;
  const uint32_t localSizeY = pipeline->localSizeY;
  uint32_t wgCountX = static_cast<uint32_t>(vk_helper::roundUpToMultiple(batchNumTilesPadded, localSizeX)) / localSizeX;
  uint32_t wgCountY = static_cast<uint32_t>(vk_helper::roundUpToMultiple(inChannelsPadded, localSizeY)) / localSizeY;
  uint32_t wgCountZ = 1u;

  wgCountX = (wgCountX == 0) ? 1 : wgCountX;
  wgCountY = (wgCountY == 0) ? 1 : wgCountY;

  vkCmdDispatch(cb, wgCountX, wgCountY, wgCountZ);
  vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace);
}

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
) {
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, convWorkspace2),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, output)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for Winograd Output Transform", *result);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr);
  auto params = vk_shader::push::WinogradOutputTransformParams();
  params.batchSize = static_cast<int>(batchSize);
  params.ySize = static_cast<int>(nnYLen);
  params.xSize = static_cast<int>(nnXLen);
  params.numTilesY = static_cast<int>(numTilesY);
  params.numTilesX = static_cast<int>(numTilesX);
  params.outChannels = static_cast<int>(outChannels);
  params.outChannelsPadded = vk_helper::roundUpToMultipleInt(outChannels, outChannelsPadMultiple);
  params.ntxtySizePadded = vk_helper::roundUpToMultipleInt(batchSize * numTilesY * numTilesX, batchNumTilesPadMultiple);
  params.xyStride = xyStride;

  // std::printf(
  //   "winogradOutputToSpatialDomain: batchSize=%d ySize=%d xSize=%d numTilesY=%d numTilesX=%d outChannels=%d outChannelsPadded=%d ntxtySizePadded=%d\n",
  //   params.batchSize, params.ySize, params.xSize, params.numTilesY, params.numTilesX, params.outChannels, params.outChannelsPadded, params.ntxtySizePadded
  // );

  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
  const uint32_t localSizeX = pipeline->localSizeX;
  const uint32_t localSizeY = pipeline->localSizeY;
  const uint32_t localSizeZ = pipeline->localSizeZ;
  uint32_t wgCountX = static_cast<uint32_t>(vk_helper::roundUpToMultiple(static_cast<uint32_t>(vk_helper::powerOf2ify(numTilesX)), localSizeX)) / localSizeX;
  uint32_t wgCountY = static_cast<uint32_t>(vk_helper::roundUpToMultiple(static_cast<uint32_t>(vk_helper::powerOf2ify(numTilesY)), localSizeY)) / localSizeY;
  uint32_t wgCountZ = static_cast<uint32_t>(vk_helper::roundUpToMultiple(static_cast<uint32_t>(batchSize * outChannels), localSizeZ)) / localSizeZ;
  wgCountX = (wgCountX == 0) ? 1 : wgCountX;
  wgCountY = (wgCountY == 0) ? 1 : wgCountY;
  wgCountZ = (wgCountZ == 0) ? 1 : wgCountZ;
  // Debug: Log push-constant params and dispatch sizes to help diagnose missing outputs
  // std::printf("[VK WINOGRAD OUT] batchSize=%d ySize=%d xSize=%d numTilesY=%d numTilesX=%d outChannels=%d outChannelsPadded=%d ntxtySizePadded=%d\\n",
  //   params.batchSize, params.ySize, params.xSize, params.numTilesY, params.numTilesX, params.outChannels, params.outChannelsPadded, params.ntxtySizePadded);
  // std::printf("[VK WINOGRAD OUT] local=%u,%u,%u global=%u,%u,%u groups=%u,%u,%u\\n",
  //   (unsigned)localSizeX, (unsigned)localSizeY, (unsigned)localSizeZ, (unsigned)wgCountX, (unsigned)wgCountY, (unsigned)wgCountZ);
  vkCmdDispatch(cb, wgCountX, wgCountY, wgCountZ);
  vk_helper::barrierCommandBufferForBuffer(cb, output);
}

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
) {
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, A),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, B),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, C)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for Batched GEMM", *result);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr);

  auto params = vk_shader::push::XGEMMBatchedParams();
  params.M = M;
  params.N = N;
  params.K = K;
  params.aOne = M;
  params.aTwo = K;
  params.bOne = N;
  params.bTwo = K;
  params.cOne = M;
  params.cTwo = N;
  // std::printf(
  //   "Launching BatchedXGemm_KM_KN_NM with M=%d, N=%d, K=%d, numBatchElts=%d\n",
  //   M, N, K, numBatchElts
  // );

  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
  
  const uint32_t MWG = tuneParams.xgemm.MWG;
  const uint32_t NWG = tuneParams.xgemm.NWG;
  const uint32_t localSizeX = pipeline->localSizeX;
  const uint32_t localSizeY = pipeline->localSizeY;
  const uint32_t localSizeZ = pipeline->localSizeZ;
  uint32_t wgCountX = (M * localSizeX / MWG) / localSizeX;
  uint32_t wgCountY = (N * localSizeY / NWG) / localSizeY;
  uint32_t wgCountZ = numBatchElts / localSizeZ;
  wgCountX = (wgCountX == 0) ? 1 : wgCountX;
  wgCountY = (wgCountY == 0) ? 1 : wgCountY;
  wgCountZ = (wgCountZ == 0) ? 1 : wgCountZ;
  
  // std::printf(
  //   "BatchedXGemm_KM_KN_NM: localSizes = %u,%u,%u globalSizes = %u,%u,%u groupCounts = %u,%u,%u\n",
  //   (unsigned)localSizeX, (unsigned)localSizeY, (unsigned)localSizeZ,
  //   (unsigned)wgCountX, (unsigned)wgCountY, (unsigned)wgCountZ
  // );
  vkCmdDispatch(cb, wgCountX, wgCountY, wgCountZ);
  vk_helper::barrierCommandBufferForBuffer(cb, C);
}


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
) {
  assert( device != nullptr );
  assert( cb != VK_NULL_HANDLE );
  assert( pipeline != nullptr );
  assert( descriptorSet != VK_NULL_HANDLE );

  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, A),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, B),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, C)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for Strided Batched GEMM", *result);

  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr);
  auto params = vk_shader::push::XgemmStridedBatchedFp32Params();
  params.kSizeM = kSizeM;
  params.kSizeN = kSizeN;
  params.kSizeK = kSizeK;
  params.aLead = kSizeM;
  params.aStride = aStride;
  params.bLead = kSizeN;
  params.bStride = bStride;
  params.cLead = kSizeM;
  params.cStride = cStride;
  params.cTranspose = 0;
  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

  size_t mCeiled = vk_helper::roundUpToMultiple(kSizeM, tuneParams.xgemmDirect.WGD);
  size_t nCeiled = vk_helper::roundUpToMultiple(kSizeN, tuneParams.xgemmDirect.WGD);
  uint32_t global[3] = {
    static_cast<uint32_t>(mCeiled * pipeline->localSizeX / tuneParams.xgemmDirect.WGD),
    static_cast<uint32_t>(nCeiled * pipeline->localSizeY / tuneParams.xgemmDirect.WGD),
    static_cast<uint32_t>(numBatchElts)
  };
  uint32_t wgCountX = global[0] / pipeline->localSizeX;
  uint32_t wgCountY = global[1] / pipeline->localSizeY;
  uint32_t wgCountZ = global[2] / pipeline->localSizeZ;
  wgCountX = (wgCountX == 0) ? 1 : wgCountX;
  wgCountY = (wgCountY == 0) ? 1 : wgCountY;
  wgCountZ = (wgCountZ == 0) ? 1 : wgCountZ;
  vkCmdDispatch(cb, wgCountX, wgCountY, wgCountZ);
  vk_helper::barrierCommandBufferForBuffer(cb, C);
  // Debug: print push-constant params and dispatch sizes for strided batched gemm
  #ifdef VULKAN_API_DEBUG
  uint32_t localDbg[3] = { pipeline->localSizeX, pipeline->localSizeY, pipeline->localSizeZ };
  uint32_t globalDbg[3] = { (uint32_t)mCeiled, (uint32_t)nCeiled, numBatchElts };
  uint32_t groupX = (globalDbg[0] + localDbg[0] - 1) / localDbg[0];
  uint32_t groupY = (globalDbg[1] + localDbg[1] - 1) / localDbg[1];
  uint32_t groupZ = (globalDbg[2] + localDbg[2] - 1) / localDbg[2];

  std::printf("[VK XGEMM STRIDED] kSizeM=%u kSizeN=%u kSizeK=%u aStride=%u bStride=%u cStride=%u numBatch=%u local=%u,%u,%u global=%u,%u,%u groups=%u,%u,%u\n",
    (unsigned)kSizeM, (unsigned)kSizeN, (unsigned)kSizeK, (unsigned)aStride, (unsigned)bStride, (unsigned)cStride, (unsigned)numBatchElts,
    (unsigned)localDbg[0], (unsigned)localDbg[1], (unsigned)localDbg[2],
    (unsigned)globalDbg[0], (unsigned)globalDbg[1], (unsigned)globalDbg[2],
    (unsigned)groupX, (unsigned)groupY, (unsigned)groupZ
  );
  #endif
}

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
) {
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, A),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, B),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, C)
    };
    *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
    CHECK_VK_MSG("Update Descriptor Sets for BatchedXGemmDirect_MK_NK_MN", *result);
  
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr);
    
    auto params = vk_shader::push::BatchedXgemmDirectFp32Params();
    params.kSizeM = static_cast<uint32_t>(M);
    params.kSizeN = static_cast<uint32_t>(N);
    params.kSizeK = static_cast<uint32_t>(K);
    params.aLead = static_cast<uint32_t>(K);
    params.bLead = static_cast<uint32_t>(K);
    params.cLead = static_cast<uint32_t>(N);
    params.cTranspose = 1;
    vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    const uint32_t WGD = tuneParams.xgemmDirect.WGD;
    size_t mCeiled = vk_helper::roundUpToMultiple(M, WGD);
    size_t nCeiled = vk_helper::roundUpToMultiple(N, WGD);
    uint32_t global[3] = {
      static_cast<uint32_t>(mCeiled * pipeline->localSizeX / WGD),
      static_cast<uint32_t>(nCeiled * pipeline->localSizeY / WGD),
      static_cast<uint32_t>(numBatchElts)
    };
    uint32_t wgCountX = global[0] / pipeline->localSizeX;
    uint32_t wgCountY = global[1] / pipeline->localSizeY;
    uint32_t wgCountZ = global[2] / pipeline->localSizeZ;
    wgCountX = (wgCountX == 0) ? 1 : wgCountX;
    wgCountY = (wgCountY == 0) ? 1 : wgCountY;
    wgCountZ = (wgCountZ == 0) ? 1 : wgCountZ;
    vkCmdDispatch(cb, wgCountX, wgCountY, wgCountZ);
    vk_helper::barrierCommandBufferForBuffer(cb, C);
}

};

#endif
