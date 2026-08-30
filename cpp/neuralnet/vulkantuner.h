#ifdef USE_VULKAN_BACKEND
#ifndef NEURALNET_VULKANTUNER_H_
#define NEURALNET_VULKANTUNER_H_

#include "../core/logger.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/vulkanshaders.h"

using namespace vk_shader;
using namespace vk_shader::tune;

namespace VulkanTuner {
  constexpr int TUNER_VERSION = 1;
  constexpr int DEFAULT_BATCH_SIZE = 4;

  struct ModelInfoForTuning {
    int trunkNumChannels;
    int modelVersion;

    static ModelInfoForTuning ofDesc(const ModelDesc& desc);
  };

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
    Logger* logger);
}  // namespace VulkanTuner

#endif
#endif
