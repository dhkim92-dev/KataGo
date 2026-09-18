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

namespace {

void convertNCHWNHWC(
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
  bool nhwcToNchw,
  VkResult* result
) {
  assert(device != nullptr);
  assert(pipeline != nullptr);
  assert(cb != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  assert(input != nullptr && output != nullptr);
  assert(result != nullptr);

  if(batchSize <= 0 || channels <= 0 || spatialSize <= 0 || spatialStride < logicalSpatialSize ||
     logicalSpatialSize <= 0 || logicalSpatialSize > spatialSize) {
    *result = VK_ERROR_INITIALIZATION_FAILED;
    return;
  }

  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, output)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update descriptors for NCHW/NHWC conversion", *result);

  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(
    cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr
  );

  const int channelsPadded = vk_helper::roundUpToMultipleInt(channels, 4);
  const vk_shader::push::NCHWNHWCParams pushParams = {
    batchSize, channels, spatialSize, spatialStride, channelsPadded, logicalSpatialSize
  };
  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushParams), &pushParams);

  const int dispatchSpatialSize = nhwcToNchw ? std::max(spatialSize, spatialStride) : spatialSize;
  const size_t vectorCount = static_cast<size_t>(batchSize) * static_cast<size_t>(dispatchSpatialSize) * static_cast<size_t>(channelsPadded / 4);
  const uint32_t workgroupCount = static_cast<uint32_t>(
    (vectorCount + pipeline->localSizeX - 1) / pipeline->localSizeX
  );
  vkCmdDispatch(cb, workgroupCount == 0 ? 1 : workgroupCount, 1, 1);
  vk_helper::barrierCommandBufferForBuffer(cb, output);
}

}

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
) {
  convertNCHWNHWC(
    device, pipeline, cb, descriptorSet, input, output,
    batchSize, channels, spatialSize, spatialStride, logicalSpatialSize, false, result
  );
}

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
) {
  convertNCHWNHWC(
    device, pipeline, cb, descriptorSet, input, output,
    batchSize, channels, spatialSize, spatialStride, logicalSpatialSize, true, result
  );
}

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
  bool begin
) {
  assert(device != nullptr);
  assert(extractPipeline != nullptr);
  assert(input != nullptr && output != nullptr);
  if(useNHWC) {
    assert(nchwToNhwcPipeline != nullptr);
    assert(nhwcScratch != nullptr);
  }

  if(commandBuffer == VK_NULL_HANDLE)
    commandBuffer = vk_helper::allocateCommandBuffer(device);

  VkResult result = VK_ERROR_UNKNOWN;
  if(begin) {
    result = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for ExtractChannel0", result);
  }

  if(useNHWC && nchwToNhwcDescriptorSet == VK_NULL_HANDLE) {
    nchwToNhwcDescriptorSet = vk_helper::allocateDescriptorSet(
      device, nchwToNhwcPipeline->descriptorSetLayout, &result
    );
    CHECK_VK_MSG("Allocate NCHW to NHWC descriptor set for ExtractChannel0", result);
  }
  const VulkanBuffer* extractInput = input;
  if(useNHWC) {
    convertNCHWToNHWC(
      device,
      nchwToNhwcPipeline,
      commandBuffer,
      nchwToNhwcDescriptorSet,
      input,
      nhwcScratch,
      batchSize,
      numInputChannels,
      nhwcSpatialSize,
      nchwSpatialStride,
      logicalSpatialSize,
      &result
    );
    CHECK_VK_MSG("Execute NCHW to NHWC conversion for ExtractChannel0", result);
    extractInput = nhwcScratch;
  }

  if(extractDescriptorSet == VK_NULL_HANDLE) {
    extractDescriptorSet = vk_helper::allocateDescriptorSet(
      device, extractPipeline->descriptorSetLayout, &result
    );
    CHECK_VK_MSG("Allocate descriptor set for ExtractChannel0", result);
  }
  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(extractDescriptorSet, 0, extractInput),
    vk_helper::writeDescriptorSetBuffer(extractDescriptorSet, 1, output)
  };
  result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update descriptors for ExtractChannel0", result);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, extractPipeline->pipeline);
  vkCmdBindDescriptorSets(
    commandBuffer,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    extractPipeline->layout,
    0,
    1,
    &extractDescriptorSet,
    0,
    nullptr
  );
  const vk_shader::push::ExtractChannel0Params pushParams = {
    batchSize,
    numInputChannels,
    nhwcSpatialSize,
    nchwSpatialStride,
    logicalSpatialSize
  };
  vkCmdPushConstants(
    commandBuffer,
    extractPipeline->layout,
    VK_SHADER_STAGE_COMPUTE_BIT,
    0,
    sizeof(pushParams),
    &pushParams
  );
  const uint32_t globalSizeX = static_cast<uint32_t>(vk_helper::powerOf2ify(nchwSpatialStride));
  const uint32_t globalSizeY = static_cast<uint32_t>(vk_helper::powerOf2ify(batchSize));
  const uint32_t wgCountX = (globalSizeX + extractPipeline->localSizeX - 1u) / extractPipeline->localSizeX;
  const uint32_t wgCountY = (globalSizeY + extractPipeline->localSizeY - 1u) / extractPipeline->localSizeY;
  SHADER_PROFILE_START("EXTRACT_CHANNEL0", commandBuffer);
  vkCmdDispatch(commandBuffer, wgCountX, wgCountY, 1u);
  SHADER_PROFILE_END("EXTRACT_CHANNEL0", commandBuffer);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, output);

  if(begin) {
    result = vk_helper::endCommandBuffer(commandBuffer);
    CHECK_VK_MSG("End command buffer for ExtractChannel0", result);
  }
}

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
) {
  assert(device != nullptr);
  assert(pipeline != nullptr);
  assert(cb != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  assert(input != nullptr && output != nullptr && scale != nullptr && bias != nullptr && mask != nullptr);
  assert(result != nullptr);
  if(batchSize <= 0 || xSize <= 0 || ySize <= 0 || logicalSpatialSize <= 0 ||
     logicalSpatialSize > spatialSize || maskSpatialStride < logicalSpatialSize ||
     channels <= 0 || channelsPadded < channels ||
     kSize <= 0 || logicalKSize <= 0 || logicalKSize > kSize ||
     convYSize <= 0 || convXSize <= 0) {
    *result = VK_ERROR_INITIALIZATION_FAILED;
    return;
  }

  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, output),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, scale),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 3, bias),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 4, mask)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for NHWC im2col", *result);

  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(
    cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr
  );
  const vk_shader::push::Im2ColNHWCParams pushParams = {
    batchSize, xSize, ySize, logicalSpatialSize, spatialSize, maskSpatialStride,
    channels, channelsPadded, kSize, logicalKSize,
    convYSize, convXSize, activation
  };
  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushParams), &pushParams);
  const size_t total = static_cast<size_t>(batchSize) * static_cast<size_t>(spatialSize) * static_cast<size_t>(kSize);
  const uint32_t workgroupCount = static_cast<uint32_t>((total + pipeline->localSizeX - 1) / pipeline->localSizeX);
  vkCmdDispatch(cb, workgroupCount == 0 ? 1 : workgroupCount, 1, 1);
  vk_helper::barrierCommandBufferForBuffer(cb, output);
}

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
) {
  assert(device != nullptr);
  assert(pipeline != nullptr);
  assert(cb != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  assert(input != nullptr && output != nullptr);
  assert(result != nullptr);
  if(batchSize <= 0 || channels <= 0 || logicalSpatialSize <= 0 || spatialSize < logicalSpatialSize ||
     outputSpatialStride < logicalSpatialSize || matrixChannels < channels) {
    *result = VK_ERROR_INITIALIZATION_FAILED;
    return;
  }
  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, output)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for NHWC matrix to NCHW", *result);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(
    cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr
  );
  const vk_shader::push::NHWCMatrixToNCHWParams pushParams = {
    batchSize, channels, logicalSpatialSize, spatialSize, outputSpatialStride, matrixChannels
  };
  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushParams), &pushParams);
  const size_t total = static_cast<size_t>(batchSize) * static_cast<size_t>(logicalSpatialSize) * static_cast<size_t>(channels);
  const uint32_t workgroupCount = static_cast<uint32_t>((total + pipeline->localSizeX - 1) / pipeline->localSizeX);
  vkCmdDispatch(cb, workgroupCount == 0 ? 1 : workgroupCount, 1, 1);
  vk_helper::barrierCommandBufferForBuffer(cb, output);
}

namespace {

template<typename TuneParams>
void doHgemmCooperativeMatrixNHWCImpl(
  const VulkanDevice* device,
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
  const TuneParams& params,
  VkResult* result
) {
  assert(device != nullptr && pipeline != nullptr && cb != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE && A != nullptr && B != nullptr && C != nullptr && result != nullptr);
  if(batchSize <= 0 || M <= 0 || N <= 0 || K <= 0 || !params.isValid() ||
     M % params.MWG != 0 || N % params.NWG != 0 || K % params.KWG != 0) {
    *result = VK_ERROR_INITIALIZATION_FAILED;
    return;
  }
  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, A),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, B),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, C)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for NHWC cooperative matrix GEMM", *result);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(
    cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr
  );
  const vk_shader::push::HGemmCooperativeMatrixParams pushParams = {M, N, K};
  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushParams), &pushParams);
  vkCmdDispatch(
    cb,
    static_cast<uint32_t>(M / params.MWG),
    static_cast<uint32_t>(N / params.NWG),
    static_cast<uint32_t>(batchSize)
  );
  vk_helper::barrierCommandBufferForBuffer(cb, C);
}

}

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
) {
  (void)tuneParams;
  doHgemmCooperativeMatrixNHWCImpl(
    device, pipeline, cb, descriptorSet, A, B, C, batchSize, M, N, K, params, result
  );
}

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
) {
  (void)tuneParams;
  doHgemmCooperativeMatrixNHWCImpl(
    device, pipeline, cb, descriptorSet, A, B, C, batchSize, M, N, K, params, result
  );
}

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

std::vector<float> convWeightsToNHWCIm2Col(
  const std::vector<float>& weights,
  uint32_t inChannels,
  uint32_t outChannels,
  uint32_t convY,
  uint32_t convX,
  uint32_t kSize,
  uint32_t nSize
) {
  std::vector<float> im2colWeights(static_cast<size_t>(kSize) * nSize, 0.0f);
  for(uint32_t y = 0; y < convY; ++y) {
    for(uint32_t x = 0; x < convX; ++x) {
      for(uint32_t ic = 0; ic < inChannels; ++ic) {
        const uint32_t k = (y * convX + x) * inChannels + ic;
        for(uint32_t oc = 0; oc < outChannels; ++oc) {
          const uint32_t n = oc;
          im2colWeights[static_cast<size_t>(k) * nSize + n] =
            weights[(static_cast<size_t>(oc) * inChannels + ic) * convY * convX + y * convX + x];
        }
      }
    }
  }
  return im2colWeights;
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
  uint32_t xyStride,
  uint32_t batchSize, uint32_t numTilesY, uint32_t numTilesX, uint32_t batchNumTilesPadMultiple,
  uint32_t inChannels, uint32_t inChannelsPaddedMultiple,
  uint32_t convSize,
  VkResult *result
) {
  // throw StringError("winogradTransformBnAct is inactivated");
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
  params.xyStride = xyStride;

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
  
  const vk_shader::tune::XgemmTuneParams& xgemmParams =
    tuneParams.vulkan.shouldUseFP16Compute ? tuneParams.xgemm16 : tuneParams.xgemm;
  const uint32_t MWG = xgemmParams.MWG;
  const uint32_t NWG = xgemmParams.NWG;
  // OpenCL launches global sizes {M*MDIMC/MWG, N*NDIMC/NWG, batch}
  // with local sizes {MDIMC, NDIMC, 1}; Vulkan takes the resulting
  // workgroup counts directly.
  uint32_t wgCountX = M / MWG;
  uint32_t wgCountY = N / NWG;
  uint32_t wgCountZ = numBatchElts;
  wgCountX = (wgCountX == 0) ? 1 : wgCountX;
  wgCountY = (wgCountY == 0) ? 1 : wgCountY;
  wgCountZ = (wgCountZ == 0) ? 1 : wgCountZ;
  
  // std::printf(
  //   "BatchedXGemm_KM_KN_NM: groupCounts = %u,%u,%u\n",
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
) {
  assert(device != nullptr);
  assert(pipeline != nullptr);
  assert(cb != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  assert(A != nullptr && B != nullptr && C != nullptr);
  assert(result != nullptr);

  const auto& params = tuneParams.hgemmCooperativeMatrix;
  if(batchSize <= 0 || M <= 0 || N <= 0 || K <= 0 ||
     !params.isValid() || M % params.MWG != 0 || N % params.NWG != 0 || K % params.KWG != 0) {
    *result = VK_ERROR_INITIALIZATION_FAILED;
    return;
  }

  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, A),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, B),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, C)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for hgemmCooperativeMatrix", *result);

  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(
    cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr
  );

  const vk_shader::push::HGemmCooperativeMatrixParams pushParams = {M, N, K};
  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushParams), &pushParams);
  vkCmdDispatch(
    cb,
    static_cast<uint32_t>(M / params.MWG),
    static_cast<uint32_t>(N / params.NWG),
    static_cast<uint32_t>(batchSize)
  );
  vk_helper::barrierCommandBufferForBuffer(cb, C);
}

void doHgemmCooperativeMatrixNCHW(
  const VulkanDevice* device,
  const vk_shader::tune::VulkanTuneParams& tuneParams,
  const Pipeline* pipeline,
  const VkCommandBuffer cb,
  const VkDescriptorSet descriptorSet,
  const VulkanBuffer* A, const VulkanBuffer* B, VulkanBuffer* C,
  const int batchSize,
  const int M, const int N, const int K,
  VkResult *result
) {
  assert(device != nullptr);
  assert(pipeline != nullptr);
  assert(cb != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  assert(A != nullptr && B != nullptr && C != nullptr);
  assert(result != nullptr);
  const auto& hgemmParams = tuneParams.hgemmCooperativeMatrixNCHW;
  if(batchSize <= 0 || M <= 0 || N <= 0 || K <= 0 ||
     !hgemmParams.isValid() || M % hgemmParams.getRequiredSpatialAlignment() != 0 ||
     N % hgemmParams.NWG != 0 || K % hgemmParams.KWG != 0) {
    *result = VK_ERROR_INITIALIZATION_FAILED;
    return;
  }

  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, A),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, B),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, C)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for hgemmCooperativeMatrixNCHW", *result);

  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(
    cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr
  );

  const vk_shader::push::HGemmCooperativeMatrixNCHWParams pushParams = {K, M, N};
  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushParams), &pushParams);

  const uint32_t wgCountX = static_cast<uint32_t>(
    (M + hgemmParams.MWG - 1) / hgemmParams.MWG
  );
  const uint32_t wgCountY = static_cast<uint32_t>(
    (N + hgemmParams.NWG - 1) / hgemmParams.NWG
  );
  vkCmdDispatch(cb, wgCountX, wgCountY, static_cast<uint32_t>(batchSize));
  vk_helper::barrierCommandBufferForBuffer(cb, C);
}

void doTransformerDualGemmSwiGLU(
  const VulkanDevice* device,
  const vk_shader::tune::VulkanTuneParams& tuneParams,
  const Pipeline* pipeline,
  const VkCommandBuffer cb,
  const VkDescriptorSet descriptorSet,
  const VulkanBuffer* input,
  const VulkanBuffer* packedFilter,
  VulkanBuffer* output,
  const int batchSize,
  const int hwSize,
  const int ffnSize,
  const int cSize,
  const int packedOCSize,
  VkResult* result
) {
  assert(device != nullptr);
  assert(pipeline != nullptr);
  assert(cb != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  assert(input != nullptr && packedFilter != nullptr && output != nullptr);
  assert(result != nullptr);

  const auto& params = tuneParams.transformerDualGemmSwiGLU;
  if(batchSize <= 0 || hwSize <= 0 || ffnSize <= 0 || cSize <= 0 || packedOCSize < 2 * ffnSize ||
     !params.isValid() || hwSize % params.getRequiredSpatialAlignment() != 0 ||
     ffnSize % params.NWG != 0 || cSize % params.KWG != 0) {
    *result = VK_ERROR_INITIALIZATION_FAILED;
    return;
  }

  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, packedFilter),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, output)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for transformerDualGemmSwiGLU", *result);

  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(
    cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr
  );
  const vk_shader::push::TransformerDualGemmSwiGLUPushParams pushParams = {
    cSize, hwSize, packedOCSize, ffnSize
  };
  vkCmdPushConstants(cb, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushParams), &pushParams);
  SHADER_PROFILE_START("TRANSFORMER_DUAL_GEMM_SWIGLU", cb);
  vkCmdDispatch(
    cb,
    static_cast<uint32_t>((hwSize + params.MWG - 1) / params.MWG),
    static_cast<uint32_t>(ffnSize / params.NWG),
    static_cast<uint32_t>(batchSize)
  );
  SHADER_PROFILE_END("TRANSFORMER_DUAL_GEMM_SWIGLU", cb);
  vk_helper::barrierCommandBufferForBuffer(cb, output);
}

SpatialRMSNormSizing computeSpatialRMSNormSizing(int tileSize, int chwSize) {
  SpatialRMSNormSizing sizing;
  // Choose numCHWWorkgroups for pass 1 such that:
  // 1. Each workgroup handles a reasonable chunk (tileSize * tilesPerGroup elements)
  // 2. numCHWWorkgroups <= tileSize so pass 2 can reduce them in a single workgroup
  // Start from the natural number of workgroups, then cap.
  int naturalWorkgroups = (chwSize + tileSize - 1) / tileSize;
  sizing.numCHWWorkgroups = std::min(naturalWorkgroups, tileSize);
  // Compute tilesPerGroup for pass 1: each workgroup covers ceil(chwSize / (numCHWWorkgroups * tileSize)) tiles
  sizing.tilesPerGroupPass1 = (chwSize + (sizing.numCHWWorkgroups * tileSize) - 1) / (sizing.numCHWWorkgroups * tileSize);
  // Pass 2: reduce numCHWWorkgroups values to 1, in a single workgroup
  sizing.tilesPerGroupPass2 = (sizing.numCHWWorkgroups + tileSize - 1) / tileSize;
  return sizing;
}

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
    int packedInputBatchStride,
    int outputBatchStride
  ) {
    auto writeDescriptorSets = {
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, mainProj),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, gateProj),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, output),
    };

    vk_helper::updateDescriptorSets(device, writeDescriptorSets);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    int eltsPerThread = tuneParams.pointwise.ELTS_PER_THREAD;
    const bool usePackedBatches = packedInputBatchStride > 0;
    const int dispatchSize = usePackedBatches ? outputBatchStride : totalSize;
    const uint32_t batchCount = usePackedBatches
      ? static_cast<uint32_t>(totalSize / outputBatchStride)
      : 1;
    auto params = vk_shader::push::TransformerSwiGLUPushParams();
    params.size = dispatchSize;
    params.packedInputBatchStride = usePackedBatches ? packedInputBatchStride : 0;
    params.outputBatchStride = usePackedBatches ? outputBatchStride : 0;
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cb, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    size_t numThreads = ((size_t)dispatchSize + eltsPerThread - 1) / eltsPerThread;
    size_t globalSize = vk_helper::roundUpToMultiple(numThreads, pipeline.localSizeX);
    uint32_t wgCountX = (globalSize + pipeline.localSizeX - 1) / pipeline.localSizeX;
    vkCmdDispatch(cb, wgCountX, batchCount, 1);
    vk_helper::barrierCommandBufferForBuffer(cb, output);
  }


};

#endif
