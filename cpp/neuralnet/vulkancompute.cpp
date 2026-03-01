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
        for ( int subX = 0 ; subX < 4 ; ++subX ) {
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
) {
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, inputBuffer),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, convWorkspace)
  };
  *result = VkHelpers::updateDescriptorSets(device, writeDescriptorSets);
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

  const int batchNumTilesPadded = VkHelpers::roundUpToMultipleInt(batchSize * numTilesY * numTilesX, batchNumTilesPadMultiple);
  const int inChannelsPadded = VkHelpers::roundUpToMultipleInt(inChannels, inChannelsPaddedMultiple);

  auto params = KatagoVulkan::WinogradInputTransformParams{
    .batchSize = batchSize,
    .nnYLen = nnYLen,
    .nnXLen = nnXLen,
    .numTilesY = (nnYLen + outTileYSize - 1) / outTileYSize,
    .numTilesX = (nnXLen + outTileXSize - 1) / outTileXSize,
    .inChannels = inChannels,
    .inChannelsPadded = static_cast<uint32_t>(inChannelsPadded),
    .ntxtySizePadded = static_cast<uint32_t>(batchNumTilesPadded)
  };
  // std::printf("convInputsToWinogradDomain: batchSize = %d nnYLen = %d nnXLen = %d numTilesY = %d numTilesX = %d inChannels = %d inChannelsPadded = %d batchNumTilesPadded = %d\n",
    // params.batchSize, params.nnYLen, params.nnXLen, params.numTilesY, params.numTilesX, params.inChannels, params.inChannelsPadded, params.ntxtySizePadded
  // );

  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
  constexpr uint32_t dim = 3;
  uint32_t local[dim];

  if ( convSize == 3 ) {
    local[0] = tuneParams.conv3x3.inputTransformLocalXSize;
    local[1] = tuneParams.conv3x3.inputTransformLocalYSize;
    local[2] = 1;
  } else {
    local[0] = tuneParams.conv5x5.inputTransformLocalXSize;
    local[1] = tuneParams.conv5x5.inputTransformLocalYSize;
    local[2] = 1;
  }

  uint32_t global[dim] = {
    static_cast<uint32_t>(VkHelpers::roundUpToMultiple(batchNumTilesPadded, local[0])),
    static_cast<uint32_t>(VkHelpers::roundUpToMultiple(inChannelsPadded, local[1])),
  };

  uint32_t groupCountX = global[0] / local[0];
  uint32_t groupCountY = global[1] / local[1];  

  groupCountX = (groupCountX == 0) ? 1 : groupCountX;
  groupCountY = (groupCountY == 0) ? 1 : groupCountY;

  // std::printf(
  //   "convInputsToWinogradDomain: localSizes = %d,%d globalSizes = %d,%d groupCounts = %d,%d\n",
  //   local[0], local[1],
  //   global[0], global[1],
  //   groupCountX, groupCountY
  // );
  vkCmdDispatch(cb, groupCountX, groupCountY, 1);
  VkHelpers::barrierCommandBufferForBuffer(cb, convWorkspace);
}


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
) {
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, inputBuffer),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, convWorkspace),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, bnScale),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 3, bnBias),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 4, mask)
  };
  *result = VkHelpers::updateDescriptorSets(device, writeDescriptorSets);
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

  const int batchNumTilesPadded = VkHelpers::roundUpToMultipleInt(batchSize * numTilesY * numTilesX, batchNumTilesPadMultiple);
  const int inChannelsPadded = VkHelpers::roundUpToMultipleInt(inChannels, inChannelsPaddedMultiple);

  auto params = KatagoVulkan::WinogradInputTransformParams{
    .batchSize = batchSize,
    .nnYLen = nnYLen,
    .nnXLen = nnXLen,
    .numTilesY = (nnYLen + outTileYSize - 1) / outTileYSize,
    .numTilesX = (nnXLen + outTileXSize - 1) / outTileXSize,
    .inChannels = inChannels,
    .inChannelsPadded = static_cast<uint32_t>(inChannelsPadded),
    .ntxtySizePadded = static_cast<uint32_t>(batchNumTilesPadded)
  };

  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
  constexpr uint32_t dim = 3;
  uint32_t local[dim];
  if ( convSize == 3 ) {
    local[0] = tuneParams.conv3x3.inputTransformLocalXSize;
    local[1] = tuneParams.conv3x3.inputTransformLocalYSize;
    local[2] = 1;
  } else {
    local[0] = tuneParams.conv5x5.inputTransformLocalXSize;
    local[1] = tuneParams.conv5x5.inputTransformLocalYSize;
    local[2] = 1;
  }
  uint32_t global[dim] = {
    static_cast<uint32_t>(VkHelpers::roundUpToMultiple(batchNumTilesPadded, local[0])),
    static_cast<uint32_t>(VkHelpers::roundUpToMultiple(inChannelsPadded, local[1])),
  };

  uint32_t groupCountX = global[0] / local[0];
  uint32_t groupCountY = global[1] / local[1];  

  groupCountX = (groupCountX == 0) ? 1 : groupCountX;
  groupCountY = (groupCountY == 0) ? 1 : groupCountY;

  vkCmdDispatch(cb, groupCountX, groupCountY, 1);
  VkHelpers::barrierCommandBufferForBuffer(cb, convWorkspace);
}

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
) {
  const uint32_t nKernelDims = 3;
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, convWorkspace2),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, output)
  };
  *result = VkHelpers::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for Winograd Output Transform", *result);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr);
  auto params = KatagoVulkan::WinogradOutputTransformParams{
    .batchSize = static_cast<int>(batchSize),
    .ySize = static_cast<int>(nnYLen),
    .xSize = static_cast<int>(nnXLen),
    .numTilesY = static_cast<int>(numTilesY),
    .numTilesX = static_cast<int>(numTilesX),
    .outChannels = static_cast<int>(outChannels),
    .outChannelsPadded = VkHelpers::roundUpToMultipleInt(outChannels, outChannelsPadMultiple),
    .ntxtySizePadded = VkHelpers::roundUpToMultipleInt(batchSize * numTilesY * numTilesX, batchNumTilesPadMultiple)
  };

  // std::printf(
  //   "winogradOutputToSpatialDomain: batchSize=%d ySize=%d xSize=%d numTilesY=%d numTilesX=%d outChannels=%d outChannelsPadded=%d ntxtySizePadded=%d\n",
  //   params.batchSize, params.ySize, params.xSize, params.numTilesY, params.numTilesX, params.outChannels, params.outChannelsPadded, params.ntxtySizePadded
  // );

  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
  uint32_t local[nKernelDims] = {8, 2, 2};

  if ( convSize == 3 ) {
    local[0] = tuneParams.conv3x3.outputTransformLocalXSize;
    local[1] = tuneParams.conv3x3.outputTransformLocalYSize;
    local[2] = tuneParams.conv3x3.outputTransformLocalZSize;
  } else {
    local[0] = tuneParams.conv5x5.outputTransformLocalXSize;
    local[1] = tuneParams.conv5x5.outputTransformLocalYSize;
    local[2] = tuneParams.conv5x5.outputTransformLocalZSize;
  }
  uint32_t global[nKernelDims] = {
    static_cast<uint32_t>(VkHelpers::roundUpToMultiple(static_cast<uint32_t>(VkHelpers::powerOf2ify(numTilesX)), local[0])),
    static_cast<uint32_t>(VkHelpers::roundUpToMultiple(static_cast<uint32_t>(VkHelpers::powerOf2ify(numTilesY)), local[1])),
    static_cast<uint32_t>(VkHelpers::roundUpToMultiple(static_cast<uint32_t>(batchSize * outChannels), local[2]))
  };
  uint32_t groupCountX = global[0] / local[0];
  uint32_t groupCountY = global[1] / local[1];
  uint32_t groupCountZ = global[2] / local[2];
  groupCountX = (groupCountX == 0) ? 1 : groupCountX;
  groupCountY = (groupCountY == 0) ? 1 : groupCountY;
  groupCountZ = (groupCountZ == 0) ? 1 : groupCountZ;
  // Debug: Log push-constant params and dispatch sizes to help diagnose missing outputs
  // std::printf("[VK WINOGRAD OUT] batchSize=%d ySize=%d xSize=%d numTilesY=%d numTilesX=%d outChannels=%d outChannelsPadded=%d ntxtySizePadded=%d\\n",
  //   params.batchSize, params.ySize, params.xSize, params.numTilesY, params.numTilesX, params.outChannels, params.outChannelsPadded, params.ntxtySizePadded);
  // std::printf("[VK WINOGRAD OUT] local=%u,%u,%u global=%u,%u,%u groups=%u,%u,%u\\n",
  //   (unsigned)local[0], (unsigned)local[1], (unsigned)local[2], (unsigned)global[0], (unsigned)global[1], (unsigned)global[2], (unsigned)groupCountX, (unsigned)groupCountY, (unsigned)groupCountZ);
  vkCmdDispatch(cb, groupCountX, groupCountY, groupCountZ);
  VkHelpers::barrierCommandBufferForBuffer(cb, output);
}

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
) {
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, A),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, B),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, C)
  };
  *result = VkHelpers::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for Batched GEMM", *result);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr);

  auto params = KatagoVulkan::XGEMMBatchedParams{
    .M = M,
    .N = N,
    .K = K,
    .aOne = M,
    .aTwo = K,
    .bOne = N,
    .bTwo = K,
    .cOne = M,
    .cTwo = N,
  };
  // std::printf(
  //   "Launching BatchedXGemm_KM_KN_NM with M=%d, N=%d, K=%d, numBatchElts=%d\n",
  //   M, N, K, numBatchElts
  // );

  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
  
  constexpr uint32_t dim = 3;
  const uint32_t MDIMC = tuneParams.xgemm.MDIMC;
  const uint32_t NDIMC = tuneParams.xgemm.NDIMC;
  const uint32_t MWG = tuneParams.xgemm.MWG;
  const uint32_t NWG = tuneParams.xgemm.NWG;
  uint32_t global[dim] = { M * MDIMC / MWG, N * NDIMC / NWG, numBatchElts };
  uint32_t local[dim] = {MDIMC, NDIMC, 1};
  uint32_t groupCountX = global[0] / local[0];
  uint32_t groupCountY = global[1] / local[1];
  uint32_t groupCountZ = global[2] / local[2];
  groupCountX = (groupCountX == 0) ? 1 : groupCountX;
  groupCountY = (groupCountY == 0) ? 1 : groupCountY;
  groupCountZ = (groupCountZ == 0) ? 1 : groupCountZ;
  
  // std::printf(
  //   "BatchedXGemm_KM_KN_NM: localSizes = %u,%u,%u globalSizes = %u,%u,%u groupCounts = %u,%u,%u\n",
  //   (unsigned)local[0], (unsigned)local[1], (unsigned)local[2],
  //   (unsigned)global[0], (unsigned)global[1], (unsigned)global[2],
  //   (unsigned)groupCountX, (unsigned)groupCountY, (unsigned)groupCountZ
  // );
  vkCmdDispatch(cb, groupCountX, groupCountY, groupCountZ);
  VkHelpers::barrierCommandBufferForBuffer(cb, C);
}


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
) {
  assert( device != nullptr );
  assert( cb != VK_NULL_HANDLE );
  assert( pipeline != nullptr );
  assert( descriptorSet != VK_NULL_HANDLE );

  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, A),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, B),
    VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, C)
  };
  *result = VkHelpers::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for Strided Batched GEMM", *result);

  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr);
  auto params = KatagoVulkan::XgemmStridedBatchedFp32Params{
    .kSizeM = kSizeM,
    .kSizeN = kSizeN,
    .kSizeK = kSizeK,
    .aLead = kSizeM,
    .aStride = aStride,
    .bLead = kSizeN,
    .bStride = bStride,
    .cLead = kSizeM,
    .cStride = cStride,
    .cTranspose = 0
  };
  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

  size_t mCeiled = VkHelpers::roundUpToMultiple(kSizeM, tuneParams.xgemmDirect.WGD);
  size_t nCeiled = VkHelpers::roundUpToMultiple(kSizeN, tuneParams.xgemmDirect.WGD);
  uint32_t global[3] = {
    static_cast<uint32_t>(mCeiled * tuneParams.xgemmDirect.MDIMCD / tuneParams.xgemmDirect.WGD),
    static_cast<uint32_t>(nCeiled * tuneParams.xgemmDirect.NDIMCD / tuneParams.xgemmDirect.WGD),
    static_cast<uint32_t>(numBatchElts)
  };
  uint32_t local[3] = { tuneParams.xgemmDirect.MDIMCD, tuneParams.xgemmDirect.NDIMCD, 1 };
  uint32_t groupCountX = global[0] / local[0];
  uint32_t groupCountY = global[1] / local[1];
  uint32_t groupCountZ = global[2] / local[2];
  groupCountX = (groupCountX == 0) ? 1 : groupCountX;
  groupCountY = (groupCountY == 0) ? 1 : groupCountY;
  groupCountZ = (groupCountZ == 0) ? 1 : groupCountZ;
  vkCmdDispatch(cb, groupCountX, groupCountY, groupCountZ);
  VkHelpers::barrierCommandBufferForBuffer(cb, C);
  // Debug: print push-constant params and dispatch sizes for strided batched gemm
  #ifdef VULKAN_API_DEBUG
  uint32_t localDbg[3] = { (uint32_t)tuneParams.xgemmDirect.MDIMCD, (uint32_t)tuneParams.xgemmDirect.NDIMCD, 1 };
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
) {
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 0, A),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 1, B),
      VkHelpers::writeDescriptorSetBuffer(descriptorSet, 2, C)
    };
    *result = VkHelpers::updateDescriptorSets(device, writeDescriptorSets);
    CHECK_VK_MSG("Update Descriptor Sets for BatchedXGemmDirect_MK_NK_MN", *result);
  
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr);
    
    auto params = KatagoVulkan::BatchedXgemmDirectFp32Params{
      .kSizeM = static_cast<uint32_t>(M),
      .kSizeN = static_cast<uint32_t>(N),
      .kSizeK = static_cast<uint32_t>(K),
      .aLead = static_cast<uint32_t>(K),
      .bLead = static_cast<uint32_t>(K),
      .cLead = static_cast<uint32_t>(N),
      .cTranspose = 1
    };
    vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    const uint32_t WGD = tuneParams.xgemmDirect.WGD;
    const uint32_t MDIMCD = tuneParams.xgemmDirect.MDIMCD;
    const uint32_t NDIMCD = tuneParams.xgemmDirect.NDIMCD;

    size_t mCeiled = VkHelpers::roundUpToMultiple(M, MDIMCD);
    size_t nCeiled = VkHelpers::roundUpToMultiple(N, NDIMCD);
    uint32_t global[3] = {
      static_cast<uint32_t>(mCeiled * MDIMCD / WGD),
      static_cast<uint32_t>(nCeiled * NDIMCD / WGD),
      static_cast<uint32_t>(numBatchElts)
    };
    uint32_t local[3] = { MDIMCD, NDIMCD, 1 };
    uint32_t groupCountX = global[0] / local[0];
    uint32_t groupCountY = global[1] / local[1];
    uint32_t groupCountZ = global[2] / local[2];
    groupCountX = (groupCountX == 0) ? 1 : groupCountX;
    groupCountY = (groupCountY == 0) ? 1 : groupCountY;
    groupCountZ = (groupCountZ == 0) ? 1 : groupCountZ;
    vkCmdDispatch(cb, groupCountX, groupCountY, groupCountZ);
    VkHelpers::barrierCommandBufferForBuffer(cb, C);
}

};

#endif