#ifdef USE_VULKAN_BACKEND
#ifndef NEURALNET_VULKANTUNER_H_
#define NEURALNET_VULKANTUNER_H_

#include <cstddef>

#include "../core/logger.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkanshaders.h"

using namespace vk_shader;
using namespace vk_shader::tune;

namespace VulkanTuner {
  constexpr int TUNER_VERSION = 32;
  constexpr int DEFAULT_BATCH_SIZE = 4;

  // Minimum candidate/baseline throughput ratios used to enable optional Vulkan paths.
  // FP16 Winograd reduces transform, workspace, and pointwise bandwidth, so
  // its GEMM may be modestly slower in isolation while the network is faster.
  constexpr double FP16_COMPUTE_MIN_THROUGHPUT_RATIO = 0.85;
  constexpr double FP16_STORAGE_MIN_THROUGHPUT_RATIO = 1.20;
  constexpr double COOPERATIVE_MATRIX_MIN_THROUGHPUT_RATIO = 1.10;
  constexpr double COOPERATIVE_MATRIX_ROPE_ERROR_TOLERANCE = 0.001;
  constexpr double COOPERATIVE_MATRIX_1X1_MIN_THROUGHPUT_RATIO = 1.20;
  constexpr double TRANSFORMER_DUAL_GEMM_MIN_THROUGHPUT_RATIO = 1.10;
  constexpr double TRANSFORMER_DUAL_GEMM_ERROR_TOLERANCE = 0.002;
  constexpr double COOPERATIVE_MATRIX_SHAPE_SCORE_RATIO = 0.90;
  constexpr std::size_t COOPERATIVE_MATRIX_MIN_SHAPES_PER_ACCUMULATOR = 3;
  constexpr std::size_t COOPERATIVE_MATRIX_NON_FULL_PROPERTY_LIMIT = 2;

  double computeErrorProp(const std::vector<float>& reference, const std::vector<float>& values);
  std::vector<float> computeTransformerDualGemmSwiGLUReference(
    const std::vector<float>& input,
    const std::vector<float>& packedFilter,
    int batchSize,
    int hwSize,
    int cSize,
    int packedOCSize,
    int ffnSize
  );
  std::vector<float> packTransformerDualGemmSwiGLUFilter(
    const std::vector<float>& mainWeights,
    const std::vector<float>& gateWeights,
    int cSize,
    int ffnSize,
    int packedOCSize
  );
  double computeTuningScore(double callsPerSecond, double errorProp, double errorToleranceScale);
  double computeCooperativeMatrixTuningScore(double callsPerSecond, double errorProp, double errorTolerance);
  bool isFastEnough(double callsPerSecond, double baselineCallsPerSecond, double requiredRatio);
  VulkanParams getHardwareParams(const VulkanDeviceInfo& deviceInfo);

  struct ModelInfoForTuning {
    int numInputChannels = 0;
    int maxConvChannels1x1 = 0;
    int maxConvChannels3x3 = 0;
    bool hasConv5x5 = false;
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
    bool transformerUseRope = false;
    bool transformerLearnableRope = false;
    float transformerRopeTheta = 0.0f;
    std::vector<float> transformerRopeFreqs;

    static ModelInfoForTuning ofDesc(const ModelDesc& desc);
  };

  int getTransformerFFNInputChannelsForTuning(const ModelInfoForTuning& modelInfo);

  struct HgemmCooperativeMatrixNCHWTuner {
    static bool selectCooperativeMatrixProperties(
      const VulkanDevice* device,
      HGemmCooperativeMatrixNCHWTuneParams& params
    );
  };

  struct HgemmCooperativeMatrixTuner {
    static bool selectCooperativeMatrixProperties(
      const VulkanDevice* device,
      HGemmCooperativeMatrixTuneParams& params
    );
  };

  void tune(
    const VulkanDevice* device,
    int batchSize,
    int nnXLen,
    int nnYLen,
    const ModelInfoForTuning& modelInfo,
    bool full,
    Logger* logger,
    VulkanTuneParams& tunedConfig,
    bool printOnlyOnImprovement = true);

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
    Logger* logger,
    bool* didAutoTune = nullptr);
}  // namespace VulkanTuner

#endif
#endif
