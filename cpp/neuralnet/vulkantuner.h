#ifdef USE_VULKAN_BACKEND
#ifndef NEURALNET_VULKANTUNER_H_
#define NEURALNET_VULKANTUNER_H_

#include "../core/logger.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkanshaders.h"

using namespace vk_shader;
using namespace vk_shader::tune;

namespace VulkanTuner {
  constexpr int TUNER_VERSION = 4;
  constexpr int DEFAULT_BATCH_SIZE = 4;

  struct ModelInfoForTuning {
    int maxConvChannels1x1 = 0;
    int maxConvChannels3x3 = 0;
    int trunkNumChannels = 0;
    int midNumChannels = 0;
    int regularNumChannels = 0;
    int gpoolNumChannels = 0;
    int modelVersion = 0;
    int transformerHeadDim = 0;
    int transformerVHeadDim = 0;
    int transformerNumHeads = 0;
    int transformerNumKVHeads = 0;
    int transformerFFNChannels = 0;

    static ModelInfoForTuning ofDesc(const ModelDesc& desc);
  };

  void tune(
    const VulkanDevice* device,
    int batchSize,
    int nnXLen,
    int nnYLen,
    const ModelInfoForTuning& modelInfo,
    bool full,
    Logger* logger,
    VulkanTuneParams& tunedConfig);

  std::string defaultDirectory(bool makeDir, const std::string& homeDataDirOverride);
  std::string
  defaultFileName(const std::string& gpuName, int nnXLen, int nnYLen, int trunkNumChannels, int modelVersion);
  std::string defaultFileName(const std::string& gpuName, int nnXLen, int nnYLen, const ModelInfoForTuning& modelInfo);

  VulkanTuneParams loadOrCreate(
    const std::string& tunerFile,
    const std::string& homeDataDirOverride,
    const std::string& gpuName,
    int nnXLen,
    int nnYLen,
    const ModelInfoForTuning& modelInfo,
    const VulkanDeviceInfo& deviceInfo,
    Logger* logger);

  VulkanTuneParams loadOrAutoTune(
    const std::string& tunerFile,
    const std::string& homeDataDirOverride,
    const std::string& gpuName,
    int nnXLen,
    int nnYLen,
    const ModelInfoForTuning& modelInfo,
    const VulkanDevice* device,
    Logger* logger);
}  // namespace VulkanTuner

#endif
#endif
