#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../dataio/homedata.h"
#include "../neuralnet/vulkanbackend.h"

using namespace std;
using namespace vk_shader;
using namespace vk_shader::tune;

namespace {
  const char* VERSION_LINE = "VERSION=1";

  bool isMultipleOf(uint64_t x, uint64_t y) {
    return y != 0 && x % y == 0;
  }

  void writeParam(ofstream& out, const char* name, uint32_t value) {
    out << name << "=" << value << "\n";
  }

  uint32_t getParam(const map<string, string>& values, const string& name, const string& filename) {
    auto iter = values.find(name);
    if(iter == values.end())
      throw IOError("VulkanTuneParams::load: missing parameter " + name + " in " + filename);
    int value;
    if(!Global::tryStringToInt(iter->second, value) || value < 0)
      throw IOError("VulkanTuneParams::load: invalid integer for " + name + " in " + filename);
    return static_cast<uint32_t>(value);
  }

  enum class TuneGroup { Conv3x3, Conv5x5, Xgemm, XgemmDirect };
  enum class CandidateStatus { Success, Reject, RecreateDevice };

  struct CandidateResult {
    CandidateStatus status;
    double milliseconds;
    string error;
  };

  uint32_t roundUp(uint32_t x, uint32_t multiple) {
    return (x + multiple - 1) / multiple * multiple;
  }

  bool isDeviceFailure(VkResult result) {
    return result == VK_ERROR_DEVICE_LOST || result == VK_TIMEOUT;
  }

  CandidateResult evaluateCandidate(
    VulkanDevice* device,
    const VulkanTuneParams& params,
    TuneGroup group,
    int nnXLen,
    int nnYLen,
    int trunkChannels) {
    if(!params.isValid())
      return {CandidateStatus::Reject, 0.0, "invalid parameter relationships"};

    vk_shader::ComputePipelines* pipelines = nullptr;
    vector<VulkanBuffer*> buffers;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkResult result = VK_SUCCESS;

    const auto cleanup = [&]() {
      if(fence != VK_NULL_HANDLE)
        vkDestroyFence(device->device, fence, nullptr);
      for(VulkanBuffer* buffer: buffers)
        vk_helper::releaseVulkanBuffer(device, buffer);
      delete pipelines;
      vkResetDescriptorPool(device->device, device->descriptorPool, 0);
      vkResetCommandPool(device->device, device->commandPool, 0);
    };
    const auto cleanupAfterDeviceFailure = [&]() {
      for(VulkanBuffer* buffer: buffers)
        delete buffer;
      buffers.clear();
      ::operator delete(pipelines);
      pipelines = nullptr;
      fence = VK_NULL_HANDLE;
      commandBuffer = VK_NULL_HANDLE;
    };
    const auto reject = [&](const string& error) {
      cleanup();
      return CandidateResult{CandidateStatus::Reject, 0.0, error};
    };

    try {
      pipelines = new vk_shader::ComputePipelines(device->device, params);
    } catch(const StringError& e) {
      return {CandidateStatus::Reject, 0.0, e.what()};
    }

    const auto makeBuffer = [&](size_t numFloats) -> VulkanBuffer* {
      VulkanBuffer* buffer = vk_helper::createDeviceBuffer(device, numFloats * sizeof(float), false, &result);
      if(result == VK_SUCCESS && buffer != nullptr)
        buffers.push_back(buffer);
      return buffer;
    };

    commandBuffer = vk_helper::allocateCommandBuffer(device, &result);
    if(result != VK_SUCCESS || commandBuffer == VK_NULL_HANDLE)
      return reject("command buffer allocation failed: " + vk_helper::vkErrorToString(result));
    result = vk_helper::beginCommandBuffer(commandBuffer);
    if(result != VK_SUCCESS)
      return reject("command buffer begin failed: " + vk_helper::vkErrorToString(result));

    vector<pair<VulkanBuffer*, size_t>> verifiedOutputs;
    const uint32_t channels = static_cast<uint32_t>(std::max(32, trunkChannels));
    if(group == TuneGroup::Xgemm) {
      const uint32_t m =
        roundUp(static_cast<uint32_t>(VulkanTuner::DEFAULT_BATCH_SIZE * nnXLen * nnYLen), params.xgemm.MWG);
      const uint32_t n = roundUp(channels, params.xgemm.NWG);
      const uint32_t k = roundUp(channels, params.xgemm.KWG);
      const uint32_t batches = 1;
      VulkanBuffer* a = makeBuffer(static_cast<size_t>(m) * k * batches);
      VulkanBuffer* b = makeBuffer(static_cast<size_t>(n) * k * batches);
      VulkanBuffer* c = makeBuffer(static_cast<size_t>(m) * n * batches);
      if(result != VK_SUCCESS || a == nullptr || b == nullptr || c == nullptr)
        return reject("XGEMM buffer allocation failed: " + vk_helper::vkErrorToString(result));
      vkCmdFillBuffer(commandBuffer, a->buffer, 0, VK_WHOLE_SIZE, 0);
      vkCmdFillBuffer(commandBuffer, b->buffer, 0, VK_WHOLE_SIZE, 0);
      vkCmdFillBuffer(commandBuffer, c->buffer, 0, VK_WHOLE_SIZE, 0x7fc00000U);
      vk_helper::barrierCommandBuffer(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
      VkDescriptorSet descriptorSet =
        vk_helper::allocateDescriptorSet(device, pipelines->xgemmBatchedFp32.descriptorSetLayout, &result);
      if(result != VK_SUCCESS)
        return reject("XGEMM descriptor allocation failed: " + vk_helper::vkErrorToString(result));
      vkcompute::xgemmBatched(
        device, params, &pipelines->xgemmBatchedFp32, commandBuffer, descriptorSet, m, n, k, a, b, c, batches, &result);
      verifiedOutputs.push_back({c, static_cast<size_t>(m) * n * batches});
    } else if(group == TuneGroup::XgemmDirect) {
      const uint32_t m = static_cast<uint32_t>(VulkanTuner::DEFAULT_BATCH_SIZE * nnXLen * nnYLen);
      const uint32_t n = channels;
      const uint32_t k = channels;
      VulkanBuffer* a = makeBuffer(static_cast<size_t>(m) * k);
      VulkanBuffer* b = makeBuffer(static_cast<size_t>(n) * k);
      VulkanBuffer* c = makeBuffer(static_cast<size_t>(m) * n);
      if(result != VK_SUCCESS || a == nullptr || b == nullptr || c == nullptr)
        return reject("direct XGEMM buffer allocation failed: " + vk_helper::vkErrorToString(result));
      vkCmdFillBuffer(commandBuffer, a->buffer, 0, VK_WHOLE_SIZE, 0);
      vkCmdFillBuffer(commandBuffer, b->buffer, 0, VK_WHOLE_SIZE, 0);
      vkCmdFillBuffer(commandBuffer, c->buffer, 0, VK_WHOLE_SIZE, 0x7fc00000U);
      vk_helper::barrierCommandBuffer(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
      VkDescriptorSet descriptorSet =
        vk_helper::allocateDescriptorSet(device, pipelines->batchedXgemmDirect.descriptorSetLayout, &result);
      if(result != VK_SUCCESS)
        return reject("direct XGEMM descriptor allocation failed: " + vk_helper::vkErrorToString(result));
      vkcompute::batchedXGemmDirect_MK_NK_MN(
        device,
        params,
        &pipelines->batchedXgemmDirect,
        commandBuffer,
        descriptorSet,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        a,
        b,
        c,
        1,
        &result);
      verifiedOutputs.push_back({c, static_cast<size_t>(m) * n});
    } else {
      const bool is3x3 = group == TuneGroup::Conv3x3;
      const ConvTuneParams& conv = is3x3 ? params.conv3x3 : params.conv5x5;
      const uint32_t convSize = is3x3 ? 3 : 5;
      const uint32_t numTilesY = (static_cast<uint32_t>(nnYLen) + conv.outTileYSize - 1) / conv.outTileYSize;
      const uint32_t numTilesX = (static_cast<uint32_t>(nnXLen) + conv.outTileXSize - 1) / conv.outTileXSize;
      const uint32_t paddedTiles = roundUp(VulkanTuner::DEFAULT_BATCH_SIZE * numTilesY * numTilesX, params.xgemm.MWG);
      const size_t inputWorkspaceFloats =
        static_cast<size_t>(conv.inTileYSize) * conv.inTileXSize * paddedTiles * roundUp(channels, params.xgemm.KWG);
      const size_t outputWorkspaceFloats =
        static_cast<size_t>(conv.inTileYSize) * conv.inTileXSize * paddedTiles * roundUp(channels, params.xgemm.NWG);
      VulkanBuffer* input =
        makeBuffer(static_cast<size_t>(VulkanTuner::DEFAULT_BATCH_SIZE) * nnXLen * nnYLen * channels);
      VulkanBuffer* transformed = makeBuffer(inputWorkspaceFloats);
      VulkanBuffer* outputTransformInput = makeBuffer(outputWorkspaceFloats);
      VulkanBuffer* output =
        makeBuffer(static_cast<size_t>(VulkanTuner::DEFAULT_BATCH_SIZE) * nnXLen * nnYLen * channels);
      if(
        result != VK_SUCCESS || input == nullptr || transformed == nullptr || outputTransformInput == nullptr ||
        output == nullptr)
        return reject("Winograd buffer allocation failed: " + vk_helper::vkErrorToString(result));
      vkCmdFillBuffer(commandBuffer, input->buffer, 0, VK_WHOLE_SIZE, 0);
      vkCmdFillBuffer(commandBuffer, transformed->buffer, 0, VK_WHOLE_SIZE, 0x7fc00000U);
      vkCmdFillBuffer(commandBuffer, outputTransformInput->buffer, 0, VK_WHOLE_SIZE, 0);
      vkCmdFillBuffer(commandBuffer, output->buffer, 0, VK_WHOLE_SIZE, 0x7fc00000U);
      vk_helper::barrierCommandBuffer(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
      const Pipeline& inputPipeline =
        is3x3 ? pipelines->winogradInputTransform3x3 : pipelines->winogradInputTransform5x5;
      const Pipeline& outputPipeline =
        is3x3 ? pipelines->winogradOutputTransform3x3 : pipelines->winogradOutputTransform5x5;
      VkDescriptorSet inputSet = vk_helper::allocateDescriptorSet(device, inputPipeline.descriptorSetLayout, &result);
      if(result != VK_SUCCESS)
        return reject("Winograd input descriptor allocation failed: " + vk_helper::vkErrorToString(result));
      VkDescriptorSet outputSet = vk_helper::allocateDescriptorSet(device, outputPipeline.descriptorSetLayout, &result);
      if(result != VK_SUCCESS)
        return reject("Winograd output descriptor allocation failed: " + vk_helper::vkErrorToString(result));
      vkcompute::convInputsToWinogradDomain(
        device,
        params,
        &inputPipeline,
        commandBuffer,
        inputSet,
        input,
        transformed,
        nnYLen,
        nnXLen,
        VulkanTuner::DEFAULT_BATCH_SIZE,
        numTilesY,
        numTilesX,
        params.xgemm.MWG,
        channels,
        params.xgemm.KWG,
        convSize,
        &result);
      if(result == VK_SUCCESS) {
        vkcompute::winogradOutputToSpatialDomain(
          device,
          params,
          &outputPipeline,
          commandBuffer,
          outputSet,
          outputTransformInput,
          output,
          nnYLen,
          nnXLen,
          VulkanTuner::DEFAULT_BATCH_SIZE,
          numTilesY,
          numTilesX,
          params.xgemm.MWG,
          channels,
          params.xgemm.NWG,
          convSize,
          &result);
      }
      verifiedOutputs.push_back({transformed, inputWorkspaceFloats});
      verifiedOutputs.push_back(
        {output, static_cast<size_t>(VulkanTuner::DEFAULT_BATCH_SIZE) * nnXLen * nnYLen * channels});
    }
    if(result != VK_SUCCESS)
      return reject("candidate command recording failed: " + vk_helper::vkErrorToString(result));

    size_t verifiedFloats = 0;
    for(const auto& output: verifiedOutputs)
      verifiedFloats += output.second;
    VulkanBuffer* readback = vk_helper::createReadbackBuffer(device, verifiedFloats * sizeof(float), &result);
    if(result != VK_SUCCESS || readback == nullptr)
      return reject("readback buffer allocation failed: " + vk_helper::vkErrorToString(result));
    buffers.push_back(readback);
    VkDeviceSize readbackOffset = 0;
    for(const auto& output: verifiedOutputs) {
      vk_helper::barrierCommandBufferForBuffer(
        commandBuffer,
        output.first,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT);
      VkBufferCopy copyRegion = {};
      copyRegion.dstOffset = readbackOffset;
      copyRegion.size = output.second * sizeof(float);
      vkCmdCopyBuffer(commandBuffer, output.first->buffer, readback->buffer, 1, &copyRegion);
      readbackOffset += copyRegion.size;
    }
    result = vk_helper::endCommandBuffer(commandBuffer);
    if(result != VK_SUCCESS)
      return reject("command buffer end failed: " + vk_helper::vkErrorToString(result));

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = vkCreateFence(device->device, &fenceInfo, nullptr, &fence);
    if(result != VK_SUCCESS)
      return reject("fence creation failed: " + vk_helper::vkErrorToString(result));

    double totalMilliseconds = 0.0;
    constexpr int warmupRuns = 1;
    constexpr int timedRuns = 3;
    for(int run = 0; run < warmupRuns + timedRuns; run++) {
      result = vkResetFences(device->device, 1, &fence);
      if(result == VK_SUCCESS) {
        auto start = chrono::steady_clock::now();
        result = vk_helper::submitCommandBuffers(device, {commandBuffer}, fence);
        if(result == VK_SUCCESS)
          result = vkWaitForFences(device->device, 1, &fence, VK_TRUE, 5000000000ULL);
        auto end = chrono::steady_clock::now();
        if(run >= warmupRuns)
          totalMilliseconds += chrono::duration<double, milli>(end - start).count();
      }
      if(result != VK_SUCCESS) {
        if(isDeviceFailure(result)) {
          cleanupAfterDeviceFailure();
          return {
            CandidateStatus::RecreateDevice, 0.0, "candidate execution failed: " + vk_helper::vkErrorToString(result)};
        }
        return reject("candidate execution failed: " + vk_helper::vkErrorToString(result));
      }
    }

    void* mapped = nullptr;
    result = vmaMapMemory(device->allocator, readback->allocation, &mapped);
    if(result != VK_SUCCESS)
      return reject("readback map failed: " + vk_helper::vkErrorToString(result));
    vmaInvalidateAllocation(device->allocator, readback->allocation, 0, VK_WHOLE_SIZE);
    const float* values = static_cast<const float*>(mapped);
    bool correct = true;
    for(size_t i = 0; i < verifiedFloats; i++) {
      if(!std::isfinite(values[i]) || std::abs(values[i]) > 1e-6f) {
        correct = false;
        break;
      }
    }
    vmaUnmapMemory(device->allocator, readback->allocation);
    if(!correct)
      return reject("candidate produced an incorrect numerical result");

    double milliseconds = totalMilliseconds / timedRuns;
    cleanup();
    return {CandidateStatus::Success, milliseconds, ""};
  }

  vector<VulkanTuneParams> candidatesForGroup(const VulkanTuneParams& base, TuneGroup group, bool full) {
    vector<VulkanTuneParams> candidates;
    candidates.push_back(base);
    if(group == TuneGroup::Conv3x3 || group == TuneGroup::Conv5x5) {
      const uint32_t inputSizes[][2] = {{64, 4}, {128, 2}, {256, 1}, {32, 8}};
      const uint32_t outputSizes[][3] = {{8, 4, 8}, {16, 4, 4}, {32, 2, 4}, {8, 8, 4}};
      size_t count = full ? 4 : 3;
      for(size_t i = 0; i < count; i++) {
        VulkanTuneParams candidate = base;
        ConvTuneParams& conv = group == TuneGroup::Conv3x3 ? candidate.conv3x3 : candidate.conv5x5;
        conv.inputTransformLocalXSize = inputSizes[i][0];
        conv.inputTransformLocalYSize = inputSizes[i][1];
        conv.outputTransformLocalXSize = outputSizes[i][0];
        conv.outputTransformLocalYSize = outputSizes[i][1];
        conv.outputTransformLocalZSize = outputSizes[i][2];
        candidates.push_back(candidate);
      }
    } else if(group == TuneGroup::Xgemm) {
      const uint32_t values[][7] = {
        {8, 8, 32, 32, 8, 8, 8},
        {8, 8, 64, 64, 8, 8, 8},
        {16, 8, 64, 32, 16, 16, 8},
        {8, 16, 32, 64, 16, 8, 16},
        {16, 16, 64, 64, 32, 16, 16}};
      size_t count = full ? 5 : 3;
      for(size_t i = 0; i < count; i++) {
        VulkanTuneParams candidate = base;
        candidate.xgemm = {
          values[i][0], values[i][1], values[i][2], values[i][3], values[i][4], values[i][5], values[i][6]};
        candidates.push_back(candidate);
      }
    } else {
      const uint32_t values[][8] = {
        {16, 4, 4, 4, 4, 1, 1, 1},
        {32, 8, 8, 8, 8, 1, 1, 1},
        {32, 8, 8, 8, 8, 2, 1, 1},
        {64, 16, 16, 16, 16, 2, 1, 1},
        {64, 8, 16, 8, 16, 2, 1, 1}};
      size_t count = full ? 5 : 3;
      for(size_t i = 0; i < count; i++) {
        VulkanTuneParams candidate = base;
        candidate.xgemmDirect = {
          values[i][0],
          values[i][1],
          values[i][2],
          values[i][3],
          values[i][4],
          values[i][5],
          values[i][6],
          values[i][7]};
        candidates.push_back(candidate);
      }
    }
    vector<VulkanTuneParams> valid;
    for(const VulkanTuneParams& candidate: candidates) {
      if(candidate.isValid() && std::find(valid.begin(), valid.end(), candidate) == valid.end())
        valid.push_back(candidate);
    }
    return valid;
  }
}  // namespace

bool VulkanTuneParams::isValid() const {
  const auto validConv = [](const ConvTuneParams& p, uint32_t outTileSize) {
    if(p.inTileXSize != 6 || p.inTileYSize != 6 || p.outTileXSize != outTileSize || p.outTileYSize != outTileSize)
      return false;
    if(p.inputTransformLocalXSize == 0 || p.inputTransformLocalYSize == 0)
      return false;
    if(p.outputTransformLocalXSize == 0 || p.outputTransformLocalYSize == 0 || p.outputTransformLocalZSize == 0)
      return false;
    if(static_cast<uint64_t>(p.inputTransformLocalXSize) * p.inputTransformLocalYSize > 1024)
      return false;
    if(
      static_cast<uint64_t>(p.outputTransformLocalXSize) * p.outputTransformLocalYSize * p.outputTransformLocalZSize >
      1024)
      return false;
    return true;
  };
  if(!validConv(conv3x3, 4) || !validConv(conv5x5, 2))
    return false;

  if(
    xgemm.MDIMC == 0 || xgemm.NDIMC == 0 || xgemm.MWG == 0 || xgemm.NWG == 0 || xgemm.KWG == 0 || xgemm.MDIMA == 0 ||
    xgemm.NDIMB == 0)
    return false;
  const uint64_t xgemmWorkgroupSize = static_cast<uint64_t>(xgemm.MDIMC) * xgemm.NDIMC;
  if(xgemmWorkgroupSize > 1024)
    return false;
  if(
    !isMultipleOf(xgemm.MWG, static_cast<uint64_t>(xgemm.MDIMC) * 4) ||
    !isMultipleOf(xgemm.NWG, static_cast<uint64_t>(xgemm.NDIMC) * 4))
    return false;
  if(
    !isMultipleOf(xgemm.MWG, static_cast<uint64_t>(xgemm.MDIMA) * 4) ||
    !isMultipleOf(xgemm.NWG, static_cast<uint64_t>(xgemm.NDIMB) * 4))
    return false;
  if(!isMultipleOf(xgemm.KWG, 4))
    return false;
  if(!isMultipleOf(xgemm.KWG, xgemmWorkgroupSize / xgemm.MDIMA))
    return false;
  if(!isMultipleOf(xgemm.KWG, xgemmWorkgroupSize / xgemm.NDIMB))
    return false;

  if(
    xgemmDirect.WGD == 0 || xgemmDirect.MDIMCD == 0 || xgemmDirect.NDIMCD == 0 || xgemmDirect.MDIMAD == 0 ||
    xgemmDirect.NDIMBD == 0 || xgemmDirect.KWID == 0)
    return false;
  const uint64_t directWorkgroupSize = static_cast<uint64_t>(xgemmDirect.MDIMCD) * xgemmDirect.NDIMCD;
  if(directWorkgroupSize > 1024)
    return false;
  if(xgemmDirect.PADA > 1 || xgemmDirect.PADB > 1)
    return false;
  if(!isMultipleOf(xgemmDirect.WGD, xgemmDirect.KWID))
    return false;
  if(
    !isMultipleOf(xgemmDirect.WGD, static_cast<uint64_t>(xgemmDirect.MDIMCD) * 4) ||
    !isMultipleOf(xgemmDirect.WGD, static_cast<uint64_t>(xgemmDirect.NDIMCD) * 4))
    return false;
  if(
    !isMultipleOf(xgemmDirect.WGD, static_cast<uint64_t>(xgemmDirect.MDIMAD) * 4) ||
    !isMultipleOf(xgemmDirect.WGD, static_cast<uint64_t>(xgemmDirect.NDIMBD) * 4))
    return false;
  if(!isMultipleOf(xgemmDirect.WGD, directWorkgroupSize / xgemmDirect.MDIMAD))
    return false;
  if(!isMultipleOf(xgemmDirect.WGD, directWorkgroupSize / xgemmDirect.NDIMBD))
    return false;
  return true;
}

bool VulkanTuneParams::operator==(const VulkanTuneParams& other) const {
  return conv3x3.inTileYSize == other.conv3x3.inTileYSize && conv3x3.inTileXSize == other.conv3x3.inTileXSize &&
         conv3x3.outTileYSize == other.conv3x3.outTileYSize && conv3x3.outTileXSize == other.conv3x3.outTileXSize &&
         conv3x3.inputTransformLocalXSize == other.conv3x3.inputTransformLocalXSize &&
         conv3x3.inputTransformLocalYSize == other.conv3x3.inputTransformLocalYSize &&
         conv3x3.outputTransformLocalXSize == other.conv3x3.outputTransformLocalXSize &&
         conv3x3.outputTransformLocalYSize == other.conv3x3.outputTransformLocalYSize &&
         conv3x3.outputTransformLocalZSize == other.conv3x3.outputTransformLocalZSize &&
         conv5x5.inTileYSize == other.conv5x5.inTileYSize && conv5x5.inTileXSize == other.conv5x5.inTileXSize &&
         conv5x5.outTileYSize == other.conv5x5.outTileYSize && conv5x5.outTileXSize == other.conv5x5.outTileXSize &&
         conv5x5.inputTransformLocalXSize == other.conv5x5.inputTransformLocalXSize &&
         conv5x5.inputTransformLocalYSize == other.conv5x5.inputTransformLocalYSize &&
         conv5x5.outputTransformLocalXSize == other.conv5x5.outputTransformLocalXSize &&
         conv5x5.outputTransformLocalYSize == other.conv5x5.outputTransformLocalYSize &&
         conv5x5.outputTransformLocalZSize == other.conv5x5.outputTransformLocalZSize &&
         xgemm.MDIMC == other.xgemm.MDIMC && xgemm.NDIMC == other.xgemm.NDIMC && xgemm.MWG == other.xgemm.MWG &&
         xgemm.NWG == other.xgemm.NWG && xgemm.KWG == other.xgemm.KWG && xgemm.MDIMA == other.xgemm.MDIMA &&
         xgemm.NDIMB == other.xgemm.NDIMB && xgemmDirect.WGD == other.xgemmDirect.WGD &&
         xgemmDirect.MDIMCD == other.xgemmDirect.MDIMCD && xgemmDirect.NDIMCD == other.xgemmDirect.NDIMCD &&
         xgemmDirect.MDIMAD == other.xgemmDirect.MDIMAD && xgemmDirect.NDIMBD == other.xgemmDirect.NDIMBD &&
         xgemmDirect.KWID == other.xgemmDirect.KWID && xgemmDirect.PADA == other.xgemmDirect.PADA &&
         xgemmDirect.PADB == other.xgemmDirect.PADB;
}

void VulkanTuneParams::save(const string& filename, const VulkanTuneParams& config) {
  if(!config.isValid())
    throw StringError("VulkanTuneParams::save: refusing to save invalid parameters to " + filename);
  ofstream out;
  FileUtils::open(out, filename);
  out << VERSION_LINE << "\n";
#define WRITE_CONV(prefix, p) \
  writeParam(out, prefix ".inTileYSize", p.inTileYSize); \
  writeParam(out, prefix ".inTileXSize", p.inTileXSize); \
  writeParam(out, prefix ".outTileYSize", p.outTileYSize); \
  writeParam(out, prefix ".outTileXSize", p.outTileXSize); \
  writeParam(out, prefix ".inputTransformLocalXSize", p.inputTransformLocalXSize); \
  writeParam(out, prefix ".inputTransformLocalYSize", p.inputTransformLocalYSize); \
  writeParam(out, prefix ".outputTransformLocalXSize", p.outputTransformLocalXSize); \
  writeParam(out, prefix ".outputTransformLocalYSize", p.outputTransformLocalYSize); \
  writeParam(out, prefix ".outputTransformLocalZSize", p.outputTransformLocalZSize)
  WRITE_CONV("conv3x3", config.conv3x3);
  WRITE_CONV("conv5x5", config.conv5x5);
#undef WRITE_CONV
  writeParam(out, "xgemm.MDIMC", config.xgemm.MDIMC);
  writeParam(out, "xgemm.NDIMC", config.xgemm.NDIMC);
  writeParam(out, "xgemm.MWG", config.xgemm.MWG);
  writeParam(out, "xgemm.NWG", config.xgemm.NWG);
  writeParam(out, "xgemm.KWG", config.xgemm.KWG);
  writeParam(out, "xgemm.MDIMA", config.xgemm.MDIMA);
  writeParam(out, "xgemm.NDIMB", config.xgemm.NDIMB);
  writeParam(out, "xgemmDirect.WGD", config.xgemmDirect.WGD);
  writeParam(out, "xgemmDirect.MDIMCD", config.xgemmDirect.MDIMCD);
  writeParam(out, "xgemmDirect.NDIMCD", config.xgemmDirect.NDIMCD);
  writeParam(out, "xgemmDirect.MDIMAD", config.xgemmDirect.MDIMAD);
  writeParam(out, "xgemmDirect.NDIMBD", config.xgemmDirect.NDIMBD);
  writeParam(out, "xgemmDirect.KWID", config.xgemmDirect.KWID);
  writeParam(out, "xgemmDirect.PADA", config.xgemmDirect.PADA);
  writeParam(out, "xgemmDirect.PADB", config.xgemmDirect.PADB);
  out.close();
}

VulkanTuneParams VulkanTuneParams::load(const string& filename) {
  vector<string> lines = FileUtils::readFileLines(filename, '\n');
  map<string, string> values;
  bool foundVersion = false;
  for(const string& rawLine: lines) {
    string line = Global::trim(Global::stripComments(rawLine));
    if(line.empty())
      continue;
    if(!foundVersion) {
      if(line != VERSION_LINE)
        throw IOError("VulkanTuneParams::load: expected first line to be " + string(VERSION_LINE) + " in " + filename);
      foundVersion = true;
      continue;
    }
    size_t eq = line.find('=');
    if(eq == string::npos || eq == 0 || eq + 1 >= line.size())
      throw IOError("VulkanTuneParams::load: malformed parameter line in " + filename);
    string key = Global::trim(line.substr(0, eq));
    string value = Global::trim(line.substr(eq + 1));
    if(values.find(key) != values.end())
      throw IOError("VulkanTuneParams::load: duplicate parameter " + key + " in " + filename);
    values[key] = value;
  }
  if(!foundVersion)
    throw IOError("VulkanTuneParams::load: no parameters in " + filename);
  if(values.size() != 33)
    throw IOError("VulkanTuneParams::load: unexpected number of parameters in " + filename);

  VulkanTuneParams config;
#define READ_CONV(prefix, p) \
  p.inTileYSize = getParam(values, prefix ".inTileYSize", filename); \
  p.inTileXSize = getParam(values, prefix ".inTileXSize", filename); \
  p.outTileYSize = getParam(values, prefix ".outTileYSize", filename); \
  p.outTileXSize = getParam(values, prefix ".outTileXSize", filename); \
  p.inputTransformLocalXSize = getParam(values, prefix ".inputTransformLocalXSize", filename); \
  p.inputTransformLocalYSize = getParam(values, prefix ".inputTransformLocalYSize", filename); \
  p.outputTransformLocalXSize = getParam(values, prefix ".outputTransformLocalXSize", filename); \
  p.outputTransformLocalYSize = getParam(values, prefix ".outputTransformLocalYSize", filename); \
  p.outputTransformLocalZSize = getParam(values, prefix ".outputTransformLocalZSize", filename)
  READ_CONV("conv3x3", config.conv3x3);
  READ_CONV("conv5x5", config.conv5x5);
#undef READ_CONV
  config.xgemm.MDIMC = getParam(values, "xgemm.MDIMC", filename);
  config.xgemm.NDIMC = getParam(values, "xgemm.NDIMC", filename);
  config.xgemm.MWG = getParam(values, "xgemm.MWG", filename);
  config.xgemm.NWG = getParam(values, "xgemm.NWG", filename);
  config.xgemm.KWG = getParam(values, "xgemm.KWG", filename);
  config.xgemm.MDIMA = getParam(values, "xgemm.MDIMA", filename);
  config.xgemm.NDIMB = getParam(values, "xgemm.NDIMB", filename);
  config.xgemmDirect.WGD = getParam(values, "xgemmDirect.WGD", filename);
  config.xgemmDirect.MDIMCD = getParam(values, "xgemmDirect.MDIMCD", filename);
  config.xgemmDirect.NDIMCD = getParam(values, "xgemmDirect.NDIMCD", filename);
  config.xgemmDirect.MDIMAD = getParam(values, "xgemmDirect.MDIMAD", filename);
  config.xgemmDirect.NDIMBD = getParam(values, "xgemmDirect.NDIMBD", filename);
  config.xgemmDirect.KWID = getParam(values, "xgemmDirect.KWID", filename);
  config.xgemmDirect.PADA = getParam(values, "xgemmDirect.PADA", filename);
  config.xgemmDirect.PADB = getParam(values, "xgemmDirect.PADB", filename);
  if(!config.isValid())
    throw IOError("VulkanTuneParams::load: parameters are invalid in " + filename);
  return config;
}

VulkanTuner::ModelInfoForTuning VulkanTuner::ModelInfoForTuning::ofDesc(const ModelDesc& desc) {
  return {desc.trunk.trunkNumChannels, desc.modelVersion};
}

string VulkanTuner::defaultDirectory(bool makeDir, const string& homeDataDirOverride) {
  string dir = HomeData::getHomeDataDir(true, homeDataDirOverride) + "/vulkantuning";
  if(makeDir)
    MakeDir::make(dir);
  return dir;
}

string
VulkanTuner::defaultFileName(const string& gpuName, int nnXLen, int nnYLen, int trunkNumChannels, int modelVersion) {
  string gpuNameForFile;
  for(char c: gpuName) {
    if(contains("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", c))
      gpuNameForFile += c;
  }
  return Global::strprintf(
    "tune%d_gpu%s_x%d_y%d_c%d_mv%d.txt",
    TUNER_VERSION,
    gpuNameForFile.c_str(),
    nnXLen,
    nnYLen,
    trunkNumChannels,
    modelVersion);
}

string
VulkanTuner::defaultFileName(const string& gpuName, int nnXLen, int nnYLen, const ModelInfoForTuning& modelInfo) {
  return defaultFileName(gpuName, nnXLen, nnYLen, modelInfo.trunkNumChannels, modelInfo.modelVersion);
}

VulkanTuneParams VulkanTuner::tune(
  VulkanDevice*& device,
  const function<VulkanDevice*()>& recreateDevice,
  const VulkanTuneParams& initialParams,
  int nnXLen,
  int nnYLen,
  const ModelInfoForTuning& modelInfo,
  Logger* logger,
  bool full) {
  VulkanTuneParams current = initialParams;
  if(!current.isValid())
    current = VulkanTuneParams();

  const TuneGroup groups[] = {TuneGroup::Conv3x3, TuneGroup::Conv5x5, TuneGroup::Xgemm, TuneGroup::XgemmDirect};
  const char* groupNames[] = {"conv3x3", "conv5x5", "xgemm", "xgemmDirect"};
  for(size_t groupIdx = 0; groupIdx < 4; groupIdx++) {
    vector<VulkanTuneParams> candidates = candidatesForGroup(current, groups[groupIdx], full);
    double bestMilliseconds = std::numeric_limits<double>::infinity();
    VulkanTuneParams best = current;
    bool found = false;
    for(size_t candidateIdx = 0; candidateIdx < candidates.size(); candidateIdx++) {
      CandidateResult result = evaluateCandidate(
        device, candidates[candidateIdx], groups[groupIdx], nnXLen, nnYLen, modelInfo.trunkNumChannels);
      if(result.status == CandidateStatus::RecreateDevice) {
        if(logger != nullptr)
          logger->write(
            "Vulkan tuner rejected " + string(groupNames[groupIdx]) +
            " candidate after device execution failure: " + result.error);
        device->skipWaitOnDestruction = true;
        delete device;
        device = nullptr;
        device = recreateDevice();
        continue;
      }
      if(result.status == CandidateStatus::Reject) {
        if(logger != nullptr)
          logger->write("Vulkan tuner rejected " + string(groupNames[groupIdx]) + " candidate: " + result.error);
        continue;
      }
      if(logger != nullptr) {
        logger->write(
          "Vulkan tuner " + string(groupNames[groupIdx]) + " candidate " +
          Global::intToString(static_cast<int>(candidateIdx + 1)) + "/" +
          Global::intToString(static_cast<int>(candidates.size())) + ": " +
          Global::doubleToString(result.milliseconds) + " ms");
      }
      if(result.milliseconds < bestMilliseconds) {
        bestMilliseconds = result.milliseconds;
        best = candidates[candidateIdx];
        found = true;
      }
    }
    if(!found)
      throw StringError("Vulkan tuner found no runnable candidates for " + string(groupNames[groupIdx]));
    current = best;
  }
  return current;
}

VulkanTuneParams VulkanTuner::loadOrAutoTune(
  const string& tunerFile,
  const string& homeDataDirOverride,
  VulkanDevice*& device,
  const function<VulkanDevice*()>& recreateDevice,
  int nnXLen,
  int nnYLen,
  const ModelInfoForTuning& modelInfo,
  Logger* logger,
  bool full,
  bool forceRetune) {
  string filename = tunerFile;
  if(filename.empty()) {
    filename = defaultDirectory(true, homeDataDirOverride) + "/" +
               defaultFileName(device->info.deviceName, nnXLen, nnYLen, modelInfo);
  }

  VulkanTuneParams initialParams;
  if(!forceRetune) {
    try {
      VulkanTuneParams loaded = VulkanTuneParams::load(filename);
      if(logger != nullptr)
        logger->write("Loaded Vulkan tuning parameters from: " + filename);
      return loaded;
    } catch(const StringError&) {
    }
  } else {
    try {
      initialParams = VulkanTuneParams::load(filename);
    } catch(const StringError&) {
    }
  }

  if(logger != nullptr) {
    logger->write("No usable Vulkan tuning parameters found at: " + filename);
    logger->write("Performing Vulkan autotuning");
  }
  VulkanTuneParams tuned = tune(device, recreateDevice, initialParams, nnXLen, nnYLen, modelInfo, logger, full);
  VulkanTuneParams::save(filename, tuned);
  if(logger != nullptr)
    logger->write("Saved Vulkan tuning parameters to: " + filename);
  return tuned;
}

#endif
