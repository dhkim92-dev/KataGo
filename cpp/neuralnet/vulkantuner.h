/**
 * Vulkan tuner for pipelines.
 * @author dhkim92-dev
 */
#ifdef USE_VULKAN_BACKEND
#ifndef NEURALNET_VULKANTUNER_H_
#define NEURALNET_VULKANTUNER_H_

#include "../core/logger.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/vulkancompute.h"

#include <functional>

struct LoadedModel;

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

  VulkanTuneParams tune(
    VulkanDevice*& device,
    const std::function<VulkanDevice*()>& recreateDevice,
    const VulkanTuneParams& initialParams,
    int nnXLen,
    int nnYLen,
    const ModelInfoForTuning& modelInfo,
    Logger* logger,
    bool full);

  VulkanTuneParams loadOrAutoTune(
    const std::string& tunerFile,
    const std::string& homeDataDirOverride,
    VulkanDevice*& device,
    const std::function<VulkanDevice*()>& recreateDevice,
    int nnXLen,
    int nnYLen,
    const LoadedModel* loadedModel,
    Logger* logger,
    bool full,
    bool forceRetune);
}  // namespace VulkanTuner

#endif
#endif
