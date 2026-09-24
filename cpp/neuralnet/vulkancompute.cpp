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

  const int channelsPadded = handle->getNHWCChannelsPadded(channels, true);
  const vk_shader::push::NCHWNHWCParams pushParams = {
    batchSize, channels, spatialSize, spatialStride, channelsPadded, logicalSpatialSize
  };

  const int dispatchSpatialSize = nhwcToNchw ? std::max(spatialSize, spatialStride) : spatialSize;
  const size_t vectorCount = static_cast<size_t>(batchSize) * static_cast<size_t>(dispatchSpatialSize) * static_cast<size_t>(channelsPadded / 4);
  const uint32_t workgroupCount = static_cast<uint32_t>(
    (vectorCount + pipeline->localSizeX - 1) / pipeline->localSizeX
  );
  dispatchPipeline(
    handle, device, pipeline, cb, descriptorSet, &pushParams, sizeof(pushParams),
    workgroupCount == 0 ? 1 : workgroupCount, 1, 1,
    nhwcToNchw ? "NHWC_TO_NCHW" : "NCHW_TO_NHWC"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, output);
}

}

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
) {
  assert(device != nullptr);
  assert(pipeline != nullptr);
  assert(cb != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
#ifndef VK_BENCHMARK
  (void)benchmarkHandle;
  (void)benchmarkName;
#endif
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(
    cb,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    pipeline->layout,
    0,
    1,
    &descriptorSet,
    0,
    nullptr
  );
  if(pushConstants != nullptr && pushConstantSize > 0) {
    vkCmdPushConstants(
      cb,
      pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      pushConstantSize,
      pushConstants
    );
  }
  VK_BENCHMARK_START(benchmarkHandle, benchmarkName, cb);
  vkCmdDispatch(cb, workgroupCountX, workgroupCountY, workgroupCountZ);
  VK_BENCHMARK_END(benchmarkHandle, benchmarkName, cb);
}

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
) {
  assert(handle != nullptr);
  assert(pipeline != nullptr);
  assert(cb != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, output),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, mergedScale),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 3, mergedBias),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 4, mask)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
  const vk_shader::push::BatchNormMaskParams pushParams = {
    static_cast<uint32_t>(batchSize),
    static_cast<uint32_t>(numChannels),
    static_cast<uint32_t>(spatialSize),
    static_cast<uint32_t>(maskSpatialStride),
    static_cast<uint32_t>(channelsPadded)
  };
  const uint32_t globalSizeX = static_cast<uint32_t>(vk_helper::powerOf2ify(spatialSize));
  const uint32_t globalSizeY = static_cast<uint32_t>(vk_helper::powerOf2ify(numChannels));
  dispatchPipeline(
    handle, handle->vulkanDevice, pipeline, cb, descriptorSet,
    &pushParams, sizeof(pushParams),
    (globalSizeX + pipeline->localSizeX - 1u) / pipeline->localSizeX,
    (globalSizeY + pipeline->localSizeY - 1u) / pipeline->localSizeY,
    1u, benchmarkName
  );
  vk_helper::barrierCommandBufferForBuffer(cb, output);
}

void doMatBiasNC(
  ComputeHandleInternal* handle,
  const Pipeline* pipeline,
  VkCommandBuffer cb,
  VkDescriptorSet descriptorSet,
  VulkanBuffer* input,
  VulkanBuffer* bias,
  int batchSize,
  int numChannels
) {
  assert(handle != nullptr);
  assert(pipeline != nullptr);
  assert(cb != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, bias)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
  const vk_shader::push::AddChannelBiasNCParams pushConstants = {
    static_cast<uint32_t>(batchSize), static_cast<uint32_t>(numChannels)
  };
  const uint32_t globalSizeX = static_cast<uint32_t>(vk_helper::powerOf2ify(numChannels));
  const uint32_t globalSizeY = static_cast<uint32_t>(vk_helper::powerOf2ify(batchSize));
  dispatchPipeline(
    handle, handle->vulkanDevice, pipeline, cb, descriptorSet,
    &pushConstants, sizeof(pushConstants),
    (globalSizeX + pipeline->localSizeX - 1u) / pipeline->localSizeX,
    (globalSizeY + pipeline->localSizeY - 1u) / pipeline->localSizeY,
    1u, "ADD_CHANNEL_BIAS_NC"
  );
  vk_helper::barrierCommandBuffer(cb);
  vk_helper::barrierCommandBufferForBuffer(cb, input);
}

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
) {
  convertNCHWNHWC(
    handle, device, pipeline, cb, descriptorSet, input, output,
    batchSize, channels, spatialSize, spatialStride, logicalSpatialSize, false, result
  );
}

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
) {
  convertNCHWNHWC(
    handle, device, pipeline, cb, descriptorSet, input, output,
    batchSize, channels, spatialSize, spatialStride, logicalSpatialSize, true, result
  );
}

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
) {
  assert(device != nullptr);
  assert(ropePipeline != nullptr);
  assert(cb != VK_NULL_HANDLE);
  assert(ropeDescriptorSet != VK_NULL_HANDLE);
  assert(input != nullptr && cosTable != nullptr && sinTable != nullptr);
  assert(result != nullptr);

  if(batchSize <= 0 || numHeads <= 0 || numKVHeads <= 0 || headDim <= 0 ||
     seqLen <= 0 || numPairs < 0 || numPairs > headDim / 2 || channels <= 0 ||
     channelsPadded < channels || channelsPadded % 4 != 0 || spatialStride < seqLen ||
     logicalSpatialSize <= 0 || logicalSpatialSize > seqLen) {
    *result = VK_ERROR_INITIALIZATION_FAILED;
    return;
  }

  VulkanBuffer* shaderInput = input;
  (void)nchwToNhwcPipeline;
  (void)nchwToNhwcDescriptorSet;
  (void)nhwcToNchwPipeline;
  (void)nhwcToNchwDescriptorSet;
  (void)nhwcScratch;

  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(ropeDescriptorSet, 0, shaderInput),
    vk_helper::writeDescriptorSetBuffer(ropeDescriptorSet, 1, cosTable),
    vk_helper::writeDescriptorSetBuffer(ropeDescriptorSet, 2, sinTable)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update TransformerApplyRoPE descriptors", *result);
  if(*result != VK_SUCCESS)
    return;

  vk_shader::push::TransformerApplyRoPEPushParams params = {};
  params.nSize = batchSize;
  params.numBufHeads = numHeads;
  params.numKVHeads = numKVHeads;
  params.headDim = headDim;
  params.xySize = seqLen;
  params.numPairs = numPairs;
  params.learnableRope = learnableRope;
  params.regionOffset = regionOffset;
  params.batchStride = useNHWC ? seqLen * channelsPadded : batchStride;
  params.channelsPadded = channelsPadded;

  const uint32_t globalSize[3] = {
    static_cast<uint32_t>(vk_helper::powerOf2ify(seqLen)),
    static_cast<uint32_t>(vk_helper::powerOf2ify(numPairs)),
    static_cast<uint32_t>(vk_helper::powerOf2ify(batchSize * numHeads))
  };
  const uint32_t workgroupCount[3] = {
    (globalSize[0] + ropePipeline->localSizeX - 1u) / ropePipeline->localSizeX,
    (globalSize[1] + ropePipeline->localSizeY - 1u) / ropePipeline->localSizeY,
    (globalSize[2] + ropePipeline->localSizeZ - 1u) / ropePipeline->localSizeZ
  };
  dispatchPipeline(
    handle, device, ropePipeline, cb, ropeDescriptorSet, &params, sizeof(params),
    workgroupCount[0], workgroupCount[1], workgroupCount[2],
    useNHWC ? "TRANSFORMER_APPLY_ROPE_NHWC" : "TRANSFORMER_APPLY_ROPE"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, shaderInput);

}

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
) {
  assert(device != nullptr);
  assert(rmsNormPipeline != nullptr);
  assert(cb != VK_NULL_HANDLE);
  assert(rmsNormDescriptorSet != VK_NULL_HANDLE);
  assert(input != nullptr && output != nullptr);
  assert(weight != nullptr && beta != nullptr && mask != nullptr);
  assert(result != nullptr);

  (void)nchwToNhwcPipeline;
  (void)nchwToNhwcDescriptorSet;
  (void)nhwcToNchwPipeline;
  (void)nhwcToNchwDescriptorSet;
  (void)nhwcInput;
  (void)nhwcOutput;

  VulkanBuffer* shaderInput = input;
  VulkanBuffer* shaderOutput = output;
  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(rmsNormDescriptorSet, 0, shaderInput),
    vk_helper::writeDescriptorSetBuffer(rmsNormDescriptorSet, 1, shaderOutput),
    vk_helper::writeDescriptorSetBuffer(rmsNormDescriptorSet, 2, weight),
    vk_helper::writeDescriptorSetBuffer(rmsNormDescriptorSet, 3, beta),
    vk_helper::writeDescriptorSetBuffer(rmsNormDescriptorSet, 4, mask)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update TransformerRMSNorm descriptors", *result);
  if(*result != VK_SUCCESS)
    return;

  const vk_shader::push::TransformerRMSNormPushParams params = {
    batchSize, channels, spatialSize, epsilon, channelsPadded
  };
  const uint32_t numXYGroups = static_cast<uint32_t>(
    (spatialSize + tuneParams.WG_XY_SIZE - 1) / tuneParams.WG_XY_SIZE
  );
  const uint32_t globalSizes[3] = {
    rmsNormPipeline->localSizeX * numXYGroups,
    static_cast<uint32_t>(batchSize),
    1u
  };
  const uint32_t workgroupCounts[3] = {
    (globalSizes[0] + rmsNormPipeline->localSizeX - 1) / rmsNormPipeline->localSizeX,
    (globalSizes[1] + rmsNormPipeline->localSizeY - 1) / rmsNormPipeline->localSizeY,
    (globalSizes[2] + rmsNormPipeline->localSizeZ - 1) / rmsNormPipeline->localSizeZ
  };
  dispatchPipeline(
    handle, device, rmsNormPipeline, cb, rmsNormDescriptorSet, &params, sizeof(params),
    workgroupCounts[0], workgroupCounts[1], workgroupCounts[2],
    useNHWC ? "TRANSFORMER_RMS_NORM_NHWC" : "TRANSFORMER_RMS_NORM"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, shaderOutput);

}

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
) {
  assert(device != nullptr && attentionPipeline != nullptr && cb != VK_NULL_HANDLE);
  assert(attentionDescriptorSet != VK_NULL_HANDLE);
  assert(packedQKV != nullptr && output != nullptr);
  assert(mask != nullptr && ropeCosTable != nullptr && ropeSinTable != nullptr && result != nullptr);

  if(batchSize <= 0 || numHeads <= 0 || numKVHeads <= 0 || qHeadDim <= 0 || vHeadDim <= 0 ||
     seqLen <= 0 || logicalSpatialSize <= 0 || logicalSpatialSize > seqLen || spatialStride < seqLen ||
     qkvChannels <= 0 || outputChannels <= 0 || qTotalDim <= 0 || kTotalDim <= 0 ||
     qTotalDim + kTotalDim + numKVHeads * vHeadDim > qkvChannels ||
     numHeads * vHeadDim > outputChannels || tuneParams.ATTN_BLOCK_Q <= 0) {
    *result = VK_ERROR_INITIALIZATION_FAILED;
    return;
  }

  if(qkvChannelsPadded < qkvChannels || outputChannelsPadded < outputChannels) {
    *result = VK_ERROR_INITIALIZATION_FAILED;
    return;
  }
  const int qkvBatchStride = seqLen * qkvChannelsPadded;
  const int outputBatchStride = seqLen * outputChannelsPadded;

  (void)nchwToNhwcPipeline;
  (void)nchwToNhwcDescriptorSet;
  (void)nhwcToNchwPipeline;
  (void)nhwcToNchwDescriptorSet;

  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(attentionDescriptorSet, 0, packedQKV),
    vk_helper::writeDescriptorSetBuffer(attentionDescriptorSet, 1, packedQKV),
    vk_helper::writeDescriptorSetBuffer(attentionDescriptorSet, 2, packedQKV),
    vk_helper::writeDescriptorSetBuffer(attentionDescriptorSet, 3, output),
    vk_helper::writeDescriptorSetBuffer(attentionDescriptorSet, 4, mask),
    vk_helper::writeDescriptorSetBuffer(attentionDescriptorSet, 5, ropeCosTable),
    vk_helper::writeDescriptorSetBuffer(attentionDescriptorSet, 6, ropeSinTable)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update cooperative attention NHWC descriptors", *result);
  if(*result != VK_SUCCESS)
    return;

  vk_shader::push::ScaleDotProductCooperativePushParam params = {};
  params.seqLen = seqLen;
  params.numHeads = numHeads;
  params.numKVHeads = numKVHeads;
  params.scale = scale;
  params.qOffset = 0;
  params.kOffset = qTotalDim;
  params.vOffset = qTotalDim + kTotalDim;
  params.qBatchStride = qkvBatchStride;
  params.kBatchStride = qkvBatchStride;
  params.vBatchStride = qkvBatchStride;
  params.qRowStride = qkvChannelsPadded;
  params.kRowStride = qkvChannelsPadded;
  params.vRowStride = qkvChannelsPadded;
  params.outputBatchStride = outputBatchStride;
  params.outputRowStride = outputChannelsPadded;
  params.useRope = useRope ? 1 : 0;
  params.learnableRope = learnableRope ? 1 : 0;
  params.ropeNumPairs = ropeNumPairs;
  params.ropeReserved = 0;

  const uint32_t qGroups = static_cast<uint32_t>(
    (seqLen + tuneParams.ATTN_BLOCK_Q - 1) / tuneParams.ATTN_BLOCK_Q
  );
  const uint32_t bhGroups = static_cast<uint32_t>(
    (batchSize * numHeads + attentionPipeline->localSizeY - 1) / attentionPipeline->localSizeY
  );
  const uint32_t zGroups = (1u + attentionPipeline->localSizeZ - 1u) / attentionPipeline->localSizeZ;
  dispatchPipeline(
    handle, device, attentionPipeline, cb, attentionDescriptorSet, &params, sizeof(params),
    qGroups, bhGroups, zGroups, "SCALE_DOT_PRODUCT_ATTENTION_NHWC"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, output);
}

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
  bool begin
) {
  assert(device != nullptr);
  assert(extractPipeline != nullptr);
  assert(commandBuffer != VK_NULL_HANDLE);
  assert(extractDescriptorSet != VK_NULL_HANDLE);
  assert(input != nullptr && output != nullptr);
  (void)nhwcScratch;

  VkResult result = VK_ERROR_UNKNOWN;
  if(begin) {
    result = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for ExtractChannel0", result);
  }

  const VulkanBuffer* extractInput = input;

  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(extractDescriptorSet, 0, extractInput),
    vk_helper::writeDescriptorSetBuffer(extractDescriptorSet, 1, output)
  };
  result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update descriptors for ExtractChannel0", result);

  const vk_shader::push::ExtractChannel0Params pushParams = {
    batchSize,
    numInputChannels,
    nhwcSpatialSize,
    nchwSpatialStride,
    logicalSpatialSize,
    channelsPadded
  };
  const uint32_t globalSizeX = static_cast<uint32_t>(vk_helper::powerOf2ify(nchwSpatialStride));
  const uint32_t globalSizeY = static_cast<uint32_t>(vk_helper::powerOf2ify(batchSize));
  const uint32_t wgCountX = (globalSizeX + extractPipeline->localSizeX - 1u) / extractPipeline->localSizeX;
  const uint32_t wgCountY = (globalSizeY + extractPipeline->localSizeY - 1u) / extractPipeline->localSizeY;
  dispatchPipeline(
    handle, device, extractPipeline, commandBuffer, extractDescriptorSet,
    &pushParams, sizeof(pushParams), wgCountX, wgCountY, 1u, "EXTRACT_CHANNEL0"
  );
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, output);

  if(begin) {
    result = vk_helper::endCommandBuffer(commandBuffer);
    CHECK_VK_MSG("End command buffer for ExtractChannel0", result);
  }
}

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
  const vk_shader::push::NHWCMatrixToNCHWParams pushParams = {
    batchSize, channels, logicalSpatialSize, spatialSize, outputSpatialStride, matrixChannels
  };
  const size_t total = static_cast<size_t>(batchSize) * static_cast<size_t>(logicalSpatialSize) * static_cast<size_t>(channels);
  const uint32_t workgroupCount = static_cast<uint32_t>((total + pipeline->localSizeX - 1) / pipeline->localSizeX);
  dispatchPipeline(
    handle, device, pipeline, cb, descriptorSet, &pushParams, sizeof(pushParams),
    workgroupCount == 0 ? 1 : workgroupCount, 1, 1, "NHWC_TO_NCHW_MATRIX"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, output);
}

namespace {

template<typename TuneParams>
void doHgemmCooperativeMatrixNHWCImpl(
  ComputeHandleInternal* handle,
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
  int aRowStride,
  int cRowStride,
  const TuneParams& params,
  VkResult* result,
  int aBatchStride,
  int bBatchStride,
  int cBatchStride
) {
  assert(device != nullptr && pipeline != nullptr && cb != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE && A != nullptr && B != nullptr && C != nullptr && result != nullptr);
  if(batchSize <= 0 || M <= 0 || N <= 0 || K <= 0 ||
     aRowStride < K || cRowStride < N || !params.isValid() ||
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
  if(aBatchStride == 0)
    aBatchStride = M * aRowStride;
  if(bBatchStride == 0)
    bBatchStride = 0;
  if(cBatchStride == 0)
    cBatchStride = M * cRowStride;
  const vk_shader::push::HGemmCooperativeMatrixNHWCParams pushParams = {
    M, N, K, aRowStride, cRowStride, aBatchStride, bBatchStride, cBatchStride
  };
  dispatchPipeline(
    handle, device, pipeline, cb, descriptorSet, &pushParams, sizeof(pushParams),
    static_cast<uint32_t>(M / params.MWG),
    static_cast<uint32_t>(N / params.NWG),
    static_cast<uint32_t>(batchSize),
    "HGEMM_COOPERATIVE_NHWC"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, C);
}

}

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
  int aBatchStride,
  int bBatchStride,
  int cBatchStride
) {
  (void)tuneParams;
  doHgemmCooperativeMatrixNHWCImpl(
    handle, device, pipeline, cb, descriptorSet, A, B, C, batchSize, M, N, K,
    aRowStride, cRowStride, params, result, aBatchStride, bBatchStride, cBatchStride
  );
}

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
  int aBatchStride,
  int bBatchStride,
  int cBatchStride
) {
  (void)tuneParams;
  doHgemmCooperativeMatrixNHWCImpl(
    handle, device, pipeline, cb, descriptorSet, A, B, C, batchSize, M, N, K,
    aRowStride, cRowStride, params, result, aBatchStride, bBatchStride, cBatchStride
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

std::vector<float> convWeightsToNHWC1x1Gemm(
  const std::vector<float>& weights,
  uint32_t inChannels,
  uint32_t outChannels,
  uint32_t kSize,
  uint32_t nSize
) {
  testAssert(kSize >= inChannels);
  std::vector<float> gemmWeights(static_cast<size_t>(kSize) * nSize, 0.0f);
  for(uint32_t ic = 0; ic < inChannels; ++ic) {
    for(uint32_t oc = 0; oc < outChannels; ++oc) {
      gemmWeights[static_cast<size_t>(ic) * nSize + oc] =
        weights[static_cast<size_t>(oc) * inChannels + ic];
    }
  }
  return gemmWeights;
}

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
  bool useNHWC
) {
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, inputBuffer),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, convWorkspace)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for Winograd Input Transform", *result);
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
  const int inChannelsPadded = useNHWC
    ? handle->getNHWCChannelsPadded(inChannels, true)
    : vk_helper::roundUpToMultipleInt(inChannels, inChannelsPaddedMultiple);

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

  const uint32_t localSizeX = pipeline->localSizeX;
  const uint32_t localSizeY = pipeline->localSizeY;
  uint32_t wgCountX = static_cast<uint32_t>(vk_helper::roundUpToMultiple(
    useNHWC ? inChannelsPadded : batchNumTilesPadded, localSizeX
  )) / localSizeX;
  uint32_t wgCountY = static_cast<uint32_t>(vk_helper::roundUpToMultiple(
    useNHWC ? batchNumTilesPadded : inChannelsPadded, localSizeY
  )) / localSizeY;
  uint32_t wgCountZ = 1u;

  wgCountX = (wgCountX == 0) ? 1 : wgCountX;
  wgCountY = (wgCountY == 0) ? 1 : wgCountY;

  // std::printf(
  //   "convInputsToWinogradDomain: localSizes = %d,%d globalSizes = %d,%d groupCounts = %d,%d\n",
  //   localSizeX, localSizeY,
  //   wgCountX, wgCountY
  // );
  dispatchPipeline(
    handle, device, pipeline, cb, descriptorSet, &params, sizeof(params),
    wgCountX, wgCountY, wgCountZ, "WINOGRAD_INPUT_TRANSFORM"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace);
}


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
  bool useNHWC
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
  const int inChannelsPadded = useNHWC
    ? handle->getNHWCChannelsPadded(inChannels, true)
    : vk_helper::roundUpToMultipleInt(inChannels, inChannelsPaddedMultiple);

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

  const uint32_t localSizeX = pipeline->localSizeX;
  const uint32_t localSizeY = pipeline->localSizeY;
  uint32_t wgCountX = static_cast<uint32_t>(vk_helper::roundUpToMultiple(
    useNHWC ? inChannelsPadded : batchNumTilesPadded, localSizeX
  )) / localSizeX;
  uint32_t wgCountY = static_cast<uint32_t>(vk_helper::roundUpToMultiple(
    useNHWC ? batchNumTilesPadded : inChannelsPadded, localSizeY
  )) / localSizeY;
  uint32_t wgCountZ = 1u;

  wgCountX = (wgCountX == 0) ? 1 : wgCountX;
  wgCountY = (wgCountY == 0) ? 1 : wgCountY;

  dispatchPipeline(
    handle, device, pipeline, cb, descriptorSet, &params, sizeof(params),
    wgCountX, wgCountY, wgCountZ, "WINOGRAD_INPUT_TRANSFORM_BN_ACT_MASK"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, convWorkspace);
}

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
  bool useNHWC
) {
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, convWorkspace2),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, output)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for Winograd Output Transform", *result);
  auto params = vk_shader::push::WinogradOutputTransformParams();
  params.batchSize = static_cast<int>(batchSize);
  params.ySize = static_cast<int>(nnYLen);
  params.xSize = static_cast<int>(nnXLen);
  params.numTilesY = static_cast<int>(numTilesY);
  params.numTilesX = static_cast<int>(numTilesX);
  params.outChannels = static_cast<int>(outChannels);
  params.outChannelsPadded = useNHWC
    ? handle->getNHWCChannelsPadded(outChannels, true)
    : vk_helper::roundUpToMultipleInt(outChannels, outChannelsPadMultiple);
  params.ntxtySizePadded = vk_helper::roundUpToMultipleInt(batchSize * numTilesY * numTilesX, batchNumTilesPadMultiple);
  params.xyStride = xyStride;

  // std::printf(
  //   "winogradOutputToSpatialDomain: batchSize=%d ySize=%d xSize=%d numTilesY=%d numTilesX=%d outChannels=%d outChannelsPadded=%d ntxtySizePadded=%d\n",
  //   params.batchSize, params.ySize, params.xSize, params.numTilesY, params.numTilesX, params.outChannels, params.outChannelsPadded, params.ntxtySizePadded
  // );

  const uint32_t localSizeX = pipeline->localSizeX;
  const uint32_t localSizeY = pipeline->localSizeY;
  const uint32_t localSizeZ = pipeline->localSizeZ;
  uint32_t wgCountX;
  uint32_t wgCountY;
  uint32_t wgCountZ;
  if(useNHWC) {
    wgCountX = static_cast<uint32_t>(vk_helper::roundUpToMultiple(params.outChannelsPadded, localSizeX)) / localSizeX;
    wgCountY = static_cast<uint32_t>(vk_helper::roundUpToMultiple(params.ntxtySizePadded, localSizeY)) / localSizeY;
    wgCountZ = 1u;
  } else {
    wgCountX = static_cast<uint32_t>(vk_helper::roundUpToMultiple(static_cast<uint32_t>(vk_helper::powerOf2ify(numTilesX)), localSizeX)) / localSizeX;
    wgCountY = static_cast<uint32_t>(vk_helper::roundUpToMultiple(static_cast<uint32_t>(vk_helper::powerOf2ify(numTilesY)), localSizeY)) / localSizeY;
    wgCountZ = static_cast<uint32_t>(vk_helper::roundUpToMultiple(static_cast<uint32_t>(batchSize * outChannels), localSizeZ)) / localSizeZ;
  }
  wgCountX = (wgCountX == 0) ? 1 : wgCountX;
  wgCountY = (wgCountY == 0) ? 1 : wgCountY;
  wgCountZ = (wgCountZ == 0) ? 1 : wgCountZ;
  // Debug: Log push-constant params and dispatch sizes to help diagnose missing outputs
  // std::printf("[VK WINOGRAD OUT] batchSize=%d ySize=%d xSize=%d numTilesY=%d numTilesX=%d outChannels=%d outChannelsPadded=%d ntxtySizePadded=%d\\n",
  //   params.batchSize, params.ySize, params.xSize, params.numTilesY, params.numTilesX, params.outChannels, params.outChannelsPadded, params.ntxtySizePadded);
  // std::printf("[VK WINOGRAD OUT] local=%u,%u,%u global=%u,%u,%u groups=%u,%u,%u\\n",
  //   (unsigned)localSizeX, (unsigned)localSizeY, (unsigned)localSizeZ, (unsigned)wgCountX, (unsigned)wgCountY, (unsigned)wgCountZ);
  dispatchPipeline(
    handle, device, pipeline, cb, descriptorSet, &params, sizeof(params),
    wgCountX, wgCountY, wgCountZ, "WINOGRAD_OUTPUT_TRANSFORM"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, output);
}

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
) {
  std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, A),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, B),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, C)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for Batched GEMM", *result);
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
  dispatchPipeline(
    handle, device, pipeline, cb, descriptorSet, &params, sizeof(params),
    wgCountX, wgCountY, wgCountZ, "XGEMM_BATCHED"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, C);
}


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
  dispatchPipeline(
    handle, device, pipeline, cb, descriptorSet, &params, sizeof(params),
    wgCountX, wgCountY, wgCountZ, "XGEMM_STRIDED_BATCHED_NN"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, C);
}

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
) {
    std::vector<WriteDescriptorSet> writeDescriptorSets = {
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, A),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, B),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, C)
    };
    *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
    CHECK_VK_MSG("Update Descriptor Sets for BatchedXGemmDirect_MK_NK_MN", *result);
  
    auto params = vk_shader::push::BatchedXgemmDirectFp32Params();
    params.kSizeM = static_cast<uint32_t>(M);
    params.kSizeN = static_cast<uint32_t>(N);
    params.kSizeK = static_cast<uint32_t>(K);
    params.aLead = static_cast<uint32_t>(K);
    params.bLead = static_cast<uint32_t>(K);
    params.cLead = static_cast<uint32_t>(N);
    params.cTranspose = 1;

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
    dispatchPipeline(
      handle, device, pipeline, cb, descriptorSet, &params, sizeof(params),
      wgCountX, wgCountY, wgCountZ, "BATCHED_XGEMM_DIRECT_FP32"
    );
    vk_helper::barrierCommandBufferForBuffer(cb, C);
}

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

  const vk_shader::push::HGemmCooperativeMatrixParams pushParams = {M, N, K};
  dispatchPipeline(
    handle, device, pipeline, cb, descriptorSet, &pushParams, sizeof(pushParams),
    static_cast<uint32_t>(M / params.MWG),
    static_cast<uint32_t>(N / params.NWG),
    static_cast<uint32_t>(batchSize),
    "HGEMM_COOPERATIVE"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, C);
}

void doHgemmCooperativeMatrixNCHW(
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

  const vk_shader::push::HGemmCooperativeMatrixNCHWParams pushParams = {K, M, N};

  const uint32_t wgCountX = static_cast<uint32_t>(
    (M + hgemmParams.MWG - 1) / hgemmParams.MWG
  );
  const uint32_t wgCountY = static_cast<uint32_t>(
    (N + hgemmParams.NWG - 1) / hgemmParams.NWG
  );
  dispatchPipeline(
    handle, device, pipeline, cb, descriptorSet, &pushParams, sizeof(pushParams),
    wgCountX, wgCountY, static_cast<uint32_t>(batchSize), "HGEMM_COOPERATIVE_NCHW"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, C);
}

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
  const int batchSize,
  const int hwSize,
  const int logicalSpatialSize,
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
  if(batchSize <= 0 || hwSize <= 0 || logicalSpatialSize <= 0 || logicalSpatialSize > hwSize ||
     ffnSize <= 0 || cSize <= 0 || packedOCSize < 2 * ffnSize ||
     !params.isValid() || hwSize % params.getRequiredSpatialAlignment() != 0 ||
     ffnSize % params.NWG != 0 || cSize % params.KWG != 0) {
    *result = VK_ERROR_INITIALIZATION_FAILED;
    return;
  }

  const int nhwcChannelAlignment = tuneParams.hgemmCooperativeMatrixNHWC.NWG;
  const int inputChannelStride = vk_helper::roundUpToMultipleInt(cSize, nhwcChannelAlignment);
  const int outputChannelStride = vk_helper::roundUpToMultipleInt(ffnSize, nhwcChannelAlignment);
  (void)nchwToNhwcPipeline;
  (void)nchwToNhwcDescriptorSet;
  (void)nhwcToNchwPipeline;
  (void)nhwcToNchwDescriptorSet;
  (void)nhwcInput;
  (void)nhwcOutput;

  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, packedFilter),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, output)
  };
  *result = vk_helper::updateDescriptorSets(device, writeDescriptorSets);
  CHECK_VK_MSG("Update Descriptor Sets for transformerDualGemmSwiGLU", *result);

  const vk_shader::push::TransformerDualGemmSwiGLUPushParams pushParams = {
    cSize, hwSize, packedOCSize, ffnSize, inputChannelStride, outputChannelStride
  };
  dispatchPipeline(
    handle, device, pipeline, cb, descriptorSet, &pushParams, sizeof(pushParams),
    static_cast<uint32_t>((hwSize + params.MWG - 1) / params.MWG),
    static_cast<uint32_t>(ffnSize / params.NWG),
    static_cast<uint32_t>(batchSize),
    "TRANSFORMER_DUAL_GEMM_SWIGLU"
  );
  vk_helper::barrierCommandBufferForBuffer(cb, output);
}

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
  bool begin
) {
  (void)nchwToNhwcDescriptorSet;
  (void)nhwcToNchwDescriptorSet;
  (void)nhwcScratch;
  const vk_shader::ComputePipelines* pipelines = handle->pipelines;
  const bool useNHWC = pipelines->useNHWC;
  const Pipeline targetPipeline = pipelines->addChannelBias;
  const int batchSize = ncSize / cSize;
  const int logicalSpatialSize = handle->nnXLen * handle->nnYLen;
  const int xySize = useNHWC ? handle->paddedNNXYLen : nchwSpatialStride;

  assert(cSize > 0 && ncSize % cSize == 0);
  assert(nchwSpatialStride >= logicalSpatialSize);
  assert(commandBuffer != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);

  VkResult res = VK_ERROR_UNKNOWN;
  if(begin) {
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for AddChannelBiases", res);
  }
  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, input),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, bias)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

  const int xyEltsPerThread = handle->tuneParams.addChannelBiases.XY_ELTS_PER_THREAD;
  const int ncEltsPerThread = handle->tuneParams.addChannelBiases.NC_ELTS_PER_THREAD;
  const int xyThreads = (xySize + xyEltsPerThread - 1) / xyEltsPerThread;
  const int ncThreads = (ncSize + ncEltsPerThread - 1) / ncEltsPerThread;
  vk_shader::push::AddChannelBiasNCHWParams pushConstants = {};
  pushConstants.ncSize = static_cast<uint32_t>(ncSize);
  pushConstants.xySize = static_cast<uint32_t>(xySize);
  pushConstants.cSize = static_cast<uint32_t>(cSize);
  pushConstants.channelsPadded = static_cast<uint32_t>(
    useNHWC ? handle->getNHWCChannelsPadded(cSize) : cSize
  );

  const uint32_t globalSizeX = vk_helper::roundUpToMultiple(xyThreads, 32);
  const uint32_t wgCountX = (globalSizeX + targetPipeline.localSizeX - 1) / targetPipeline.localSizeX;
  const uint32_t wgCountY = static_cast<uint32_t>(ncThreads);
  dispatchPipeline(
    handle, handle->vulkanDevice, &targetPipeline, commandBuffer, descriptorSet,
    &pushConstants, sizeof(pushConstants), wgCountX, wgCountY, 1u, "ADD_CHANNEL_BIAS"
  );
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, input);
  if(begin)
    vk_helper::endCommandBuffer(commandBuffer);
}

void performAddPointWise(
  ComputeHandleInternal* handle,
  VkCommandBuffer& commandBuffer,
  VkDescriptorSet& descriptorSet,
  VulkanBuffer* acc,
  VulkanBuffer* value,
  int totalSize,
  bool begin
) {
  assert(commandBuffer != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  VkResult res = VK_ERROR_UNKNOWN;
  if(begin) {
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for AddPointWise", res);
  }
  const Pipeline& targetPipeline = handle->pipelines->addPointWise;
  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, acc),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, value)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

  vk_shader::push::AddPointWiseParams pushConstants = {};
  pushConstants.size = static_cast<uint32_t>(totalSize);
  const uint32_t eltsPerThread = static_cast<uint32_t>(handle->tuneParams.pointwise.ELTS_PER_THREAD);
  const uint32_t wgCountX =
    (static_cast<uint32_t>(totalSize) + targetPipeline.localSizeX * eltsPerThread - 1u) /
    (targetPipeline.localSizeX * eltsPerThread);
  dispatchPipeline(
    handle, handle->vulkanDevice, &targetPipeline, commandBuffer, descriptorSet,
    &pushConstants, sizeof(pushConstants), wgCountX, 1u, 1u, "ADD_POINTWISE_FP32"
  );
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, acc);
  if(begin)
    vk_helper::endCommandBuffer(commandBuffer);
}

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
  bool begin
) {
  (void)nchwToNhwcDescriptorSet;
  (void)nhwcScratch;
  const vk_shader::ComputePipelines* pipelines = handle->pipelines;
  const bool useNHWC = pipelines->useNHWC;
  const Pipeline pipeline = pipelines->globalPoolingChannelsFp32;
  const int spatialSize = useNHWC ? handle->paddedNNXYLen : nnXYLen;
  assert(commandBuffer != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  VkResult res = VK_ERROR_UNKNOWN;
  if(begin) {
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for GlobalPoolingMask", res);
  }
  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, gpoolConvOut),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, gpoolConcat),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, mask),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 3, maskSum)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

  vk_shader::push::GlobalPoolingChannelsParams pushConstants = {};
  pushConstants.nSize = batchSize;
  pushConstants.cSize = gpoolChannels;
  pushConstants.xySize = spatialSize;
  pushConstants.maskSpatialStride = useNHWC ? handle->paddedNNXYLen : nnXYLen;
  pushConstants.channelsPadded = useNHWC
    ? handle->getNHWCChannelsPadded(gpoolChannels) : gpoolChannels;
  const uint32_t globalSizeY = static_cast<uint32_t>(
    vk_helper::roundUpToMultiple(gpoolChannels, pipeline.localSizeY)
  );
  const uint32_t globalSizeZ = static_cast<uint32_t>(
    vk_helper::roundUpToMultiple(batchSize, pipeline.localSizeZ)
  );
  dispatchPipeline(
    handle, handle->vulkanDevice, &pipeline, commandBuffer, descriptorSet,
    &pushConstants, sizeof(pushConstants), 1u, globalSizeY / pipeline.localSizeY,
    globalSizeZ / pipeline.localSizeZ, "GLOBAL_POOLING_CHANNELS_FP32"
  );
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, maskSum);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, mask);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, gpoolConcat);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, gpoolConvOut);
  if(begin)
    res = vk_helper::endCommandBuffer(commandBuffer);
  *result = res;
}

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
  bool begin
) {
  (void)nchwToNhwcDescriptorSet;
  (void)nhwcScratch;
  assert(commandBuffer != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  VkResult res = VK_ERROR_UNKNOWN;
  if(begin) {
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin command buffer for ValueHeadPool", res);
  }
  const auto pipelines = handle->pipelines;
  const bool useNHWC = pipelines->useNHWC;
  const int spatialSize = useNHWC ? handle->paddedNNXYLen : nnXYLen;
  const vk_shader::LocalDim dim = {
    handle->tuneParams.gPool.XYSTRIDE,
    std::min(handle->tuneParams.gPool.CHANNELSTRIDE, static_cast<int>(vk_helper::powerOf2ify(gPoolChannels))),
    std::min(handle->tuneParams.gPool.BATCHSTRIDE, static_cast<int>(vk_helper::powerOf2ify(batchSize)))
  };
  const Pipeline pipeline = pipelines->valueHeadPoolingChannels.at(dim);
  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, gpoolConvOut),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, gpoolConcat),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, maskSum)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);

  const uint32_t localSizeY = std::min(
    handle->tuneParams.gPool.CHANNELSTRIDE,
    static_cast<int>(vk_helper::powerOf2ify(gPoolChannels))
  );
  const uint32_t localSizeZ = std::min(
    handle->tuneParams.gPool.BATCHSTRIDE,
    static_cast<int>(vk_helper::powerOf2ify(batchSize))
  );
  vk_shader::push::ValueHeadPoolingChannelsParams pushConstants = {};
  pushConstants.nSize = batchSize;
  pushConstants.cSize = gPoolChannels;
  pushConstants.xySize = spatialSize;
  pushConstants.channelsPadded = useNHWC
    ? handle->getNHWCChannelsPadded(gPoolChannels) : gPoolChannels;
  const uint32_t globalSizeY = vk_helper::roundUpToMultiple(gPoolChannels, localSizeY);
  const uint32_t globalSizeZ = vk_helper::roundUpToMultiple(batchSize, localSizeZ);
  dispatchPipeline(
    handle, handle->vulkanDevice, &pipeline, commandBuffer, descriptorSet,
    &pushConstants, sizeof(pushConstants),
    (handle->tuneParams.gPool.XYSTRIDE + pipeline.localSizeX - 1) / pipeline.localSizeX,
    (globalSizeY + pipeline.localSizeY - 1) / pipeline.localSizeY,
    (globalSizeZ + pipeline.localSizeZ - 1) / pipeline.localSizeZ,
    "VALUE_HEAD_POOLING_CHANNELS_FP32"
  );
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, maskSum);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, gpoolConcat);
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, gpoolConvOut);
  if(begin)
    vk_helper::endCommandBuffer(commandBuffer);
}

void computeMaskSums(
  ComputeHandleInternal* handle,
  VkCommandBuffer& commandBuffer,
  VkDescriptorSet& descriptorSet,
  int batchSize,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  bool begin
) {
  const int numChannels = 1;
  const int paddedNNXYLen = handle->paddedNNXYLen;
  const vk_shader::ComputePipelines* pipelines = handle->pipelines;
  const vk_shader::LocalDim dim = {
    handle->tuneParams.gPool.XYSTRIDE,
    1,
    std::min(handle->tuneParams.gPool.BATCHSTRIDE, static_cast<int>(vk_helper::powerOf2ify(batchSize)))
  };
  const Pipeline& targetPipeline = pipelines->sumChannels.at(dim);
  const uint32_t globalSizeZ = static_cast<uint32_t>(
    vk_helper::roundUpToMultiple(static_cast<size_t>(batchSize), targetPipeline.localSizeZ)
  );
  const uint32_t wgCountX = (handle->tuneParams.gPool.XYSTRIDE + targetPipeline.localSizeX - 1) / targetPipeline.localSizeX;
  const uint32_t wgCountY = (1u + targetPipeline.localSizeY - 1) / targetPipeline.localSizeY;
  const uint32_t wgCountZ = (globalSizeZ + targetPipeline.localSizeZ - 1) / targetPipeline.localSizeZ;

  assert(commandBuffer != VK_NULL_HANDLE);
  assert(descriptorSet != VK_NULL_HANDLE);
  VkResult res = VK_SUCCESS;
  const std::vector<WriteDescriptorSet> writeDescriptorSets = {
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, mask),
    vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, maskSum)
  };
  vk_helper::updateDescriptorSets(handle->vulkanDevice, writeDescriptorSets);
  if(begin) {
    res = vk_helper::beginCommandBuffer(commandBuffer);
    CHECK_VK_MSG("Begin compute mask sum command buffer", res);
  }
  vk_shader::push::SumChannelsParams pushConstants = {};
  pushConstants.nSize = batchSize;
  pushConstants.cSize = numChannels;
  pushConstants.xySize = paddedNNXYLen;
  dispatchPipeline(
    handle, handle->vulkanDevice, &targetPipeline, commandBuffer, descriptorSet,
    &pushConstants, sizeof(pushConstants), wgCountX, wgCountY, wgCountZ,
    "COMPUTE_MASK_SUMS"
  );
  vk_helper::barrierCommandBufferForBuffer(commandBuffer, maskSum);
  if(begin)
    vk_helper::endCommandBuffer(commandBuffer);
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
    int packedInputBatchStride,
    int outputBatchStride
  ) {
    auto writeDescriptorSets = {
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 0, mainProj),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 1, gateProj),
      vk_helper::writeDescriptorSetBuffer(descriptorSet, 2, output),
    };

    vk_helper::updateDescriptorSets(device, writeDescriptorSets);
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

    size_t numThreads = ((size_t)dispatchSize + eltsPerThread - 1) / eltsPerThread;
    size_t globalSize = vk_helper::roundUpToMultiple(numThreads, pipeline.localSizeX);
    uint32_t wgCountX = (globalSize + pipeline.localSizeX - 1) / pipeline.localSizeX;
    dispatchPipeline(
      handle, device, &pipeline, cb, descriptorSet, &params, sizeof(params),
      wgCountX, batchCount, 1, "TRANSFORMER_SWIGLU"
    );
    vk_helper::barrierCommandBufferForBuffer(cb, output);
  }


};

#endif
