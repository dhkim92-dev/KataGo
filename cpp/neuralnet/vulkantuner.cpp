#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"
#include "../neuralnet/vulkancompute.h"

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/rand.h"
#include "../core/test.h"
#include "../dataio/homedata.h"

using namespace std;
using namespace vk_shader;
using namespace vk_shader::tune;

int VulkanTuner::getPaddedNNXYLen(const VulkanTuneParams& config, bool usingFP16Storage, int logicalXYSize) {
  const auto& mode = config.vulkan;
  if(!usingFP16Storage || !mode.canUseCooperativeMatrix || !mode.shouldUseFP16Compute)
    return logicalXYSize;

  int alignment = 1;
  if(mode.shouldUseCooperativeMatrix && config.hgemmCooperativeMatrixNHWC.isValid())
    alignment = config.hgemmCooperativeMatrixNHWC.MWG;
  if(mode.shouldUseHgemmCooperativeMatrixNCHW && config.hgemmCooperativeMatrixNCHW.isValid())
    alignment = std::lcm(alignment, config.hgemmCooperativeMatrixNCHW.getRequiredSpatialAlignment());
  if(mode.canUseFP16Storage && mode.canUseFP16Compute && mode.shouldUseFP16Storage &&
     mode.shouldUseTransformerDualGemmSwiGLU && config.transformerDualGemmSwiGLU.isValid())
    alignment = std::lcm(alignment, config.transformerDualGemmSwiGLU.getRequiredSpatialAlignment());
  return vk_helper::roundUpToMultipleInt(logicalXYSize, alignment);
}

int VulkanTuner::getPackedProjectionOutChannels(
  const VulkanTuneParams& config, bool usingFP16Storage, int paddedNNXYLen, int inChannels, int logicalOutChannels
) {
  const auto& mode = config.vulkan;
  const auto& nchw = config.hgemmCooperativeMatrixNCHW;
  if(usingFP16Storage && mode.canUseCooperativeMatrix && mode.shouldUseFP16Storage && mode.shouldUseFP16Compute &&
     mode.shouldUseHgemmCooperativeMatrixNCHW && nchw.isValid() &&
     paddedNNXYLen % nchw.getRequiredSpatialAlignment() == 0 && inChannels % nchw.KWG == 0)
    logicalOutChannels = vk_helper::roundUpToMultipleInt(logicalOutChannels, nchw.NWG);

  const auto& nhwc = config.hgemmCooperativeMatrixNHWC;
  return nhwc.isValid() ? vk_helper::roundUpToMultipleInt(logicalOutChannels, nhwc.NWG) : logicalOutChannels;
}

bool VulkanTuner::isTransformerGemmUpgrade(
  const vector<double>& referenceSeconds,
  const vector<double>& candidateSeconds
) {
  if(referenceSeconds.size() != 6 || candidateSeconds.size() != 6)
    return false;
  int wins = 0;
  for(size_t i = 0; i < referenceSeconds.size(); i++) {
    if(!isfinite(referenceSeconds[i]) || referenceSeconds[i] <= 0.0 ||
       !isfinite(candidateSeconds[i]) || candidateSeconds[i] <= 0.0)
      return false;
    wins += candidateSeconds[i] < referenceSeconds[i];
  }
  vector<double> reference = referenceSeconds, candidate = candidateSeconds;
  sort(reference.begin(), reference.end());
  sort(candidate.begin(), candidate.end());
  return wins >= 5 && reference[2] + reference[3] >= 1.02 * (candidate[2] + candidate[3]);
}

double VulkanTuner::computeErrorProp(const vector<float>& reference, const vector<float>& values) {
  double squaredError = 0.0;
  double squaredMagnitude = 0.0;
  if(reference.size() != values.size())
    squaredError = numeric_limits<double>::infinity();
  else {
    for(size_t i = 0; i < values.size(); i++) {
      if(!isfinite(reference[i]) || !isfinite(values[i])) {
        squaredError = numeric_limits<double>::infinity();
        break;
      }
      const double diff = static_cast<double>(reference[i]) - values[i];
      squaredError += diff * diff;
      squaredMagnitude += static_cast<double>(reference[i]) * reference[i];
    }
  }
  const double errorProp = sqrt(squaredError / (squaredMagnitude + 1e-30));
  return isfinite(errorProp) ? errorProp : 1.0;
}

vector<float> VulkanTuner::packTransformerDualGemmSwiGLUFilterForNHWC(
  const vector<float>& mainWeights,
  const vector<float>& gateWeights,
  int cSize,
  int ffnSize
) {
  if(cSize <= 0 || ffnSize <= 0 || mainWeights.size() != static_cast<size_t>(cSize) * ffnSize ||
     gateWeights.size() != mainWeights.size())
    throw StringError("Invalid dimensions for NHWC fused FFN weights");
  vector<float> packed(2 * mainWeights.size());
  for(int n = 0; n < ffnSize; n++)
    for(int k = 0; k < cSize; k++) {
      packed[static_cast<size_t>(n) * cSize + k] = mainWeights[static_cast<size_t>(k) * ffnSize + n];
      packed[(static_cast<size_t>(ffnSize) + n) * cSize + k] = gateWeights[static_cast<size_t>(k) * ffnSize + n];
    }
  return packed;
}

vector<float> VulkanTuner::packTransformerDualGemmSwiGLUFilter(
  const vector<float>& mainWeights,
  const vector<float>& gateWeights,
  int cSize,
  int ffnSize,
  int packedOCSize
) {
  if(cSize <= 0 || ffnSize <= 0 || packedOCSize < ffnSize || packedOCSize - ffnSize < ffnSize)
    throw StringError("Invalid dimensions for Transformer dual-GEMM SwiGLU filter packing");
  const size_t logicalWeightCount = static_cast<size_t>(cSize) * ffnSize;
  if(mainWeights.size() < logicalWeightCount || gateWeights.size() < logicalWeightCount)
    throw StringError("Transformer dual-GEMM SwiGLU weights are too small to pack");

  vector<float> packed(static_cast<size_t>(cSize) * packedOCSize, 0.0f);
  for(int ic = 0; ic < cSize; ic++) {
    const size_t srcBase = static_cast<size_t>(ic) * ffnSize;
    const size_t dstBase = static_cast<size_t>(ic) * packedOCSize;
    copy_n(mainWeights.begin() + srcBase, ffnSize, packed.begin() + dstBase);
    copy_n(gateWeights.begin() + srcBase, ffnSize, packed.begin() + dstBase + ffnSize);
  }
  return packed;
}

vector<float> VulkanTuner::computeTransformerDualGemmSwiGLUReference(
  const vector<float>& input,
  const vector<float>& packedFilter,
  int batchSize,
  int hwSize,
  int cSize,
  int packedOCSize,
  int ffnSize
) {
  if(batchSize <= 0 || hwSize <= 0 || cSize <= 0 || ffnSize <= 0 ||
     packedOCSize < ffnSize || packedOCSize - ffnSize < ffnSize)
    throw StringError("Invalid dimensions for Transformer dual-GEMM SwiGLU reference");
  const size_t inputSize = static_cast<size_t>(batchSize) * cSize * hwSize;
  const size_t filterSize = static_cast<size_t>(cSize) * packedOCSize;
  if(input.size() < inputSize || packedFilter.size() < filterSize)
    throw StringError("Input or packed filter is too small for Transformer dual-GEMM SwiGLU reference");

  vector<float> output(static_cast<size_t>(batchSize) * ffnSize * hwSize);
  for(int batch = 0; batch < batchSize; batch++) {
    const size_t inputBatchBase = static_cast<size_t>(batch) * cSize * hwSize;
    const size_t outputBatchBase = static_cast<size_t>(batch) * ffnSize * hwSize;
    for(int oc = 0; oc < ffnSize; oc++) {
      for(int hw = 0; hw < hwSize; hw++) {
        float mainValue = 0.0f;
        float gateValue = 0.0f;
        for(int ic = 0; ic < cSize; ic++) {
          const float inputValue = input[inputBatchBase + static_cast<size_t>(ic) * hwSize + hw];
          const size_t filterBase = static_cast<size_t>(ic) * packedOCSize;
          mainValue += inputValue * packedFilter[filterBase + oc];
          gateValue += inputValue * packedFilter[filterBase + ffnSize + oc];
        }
        const float siluMain = mainValue / (1.0f + expf(-mainValue));
        output[outputBatchBase + static_cast<size_t>(oc) * hwSize + hw] =
          half_float::half_cast<float>(half_float::half_cast<half_t>(siluMain * gateValue));
      }
    }
  }
  return output;
}

double VulkanTuner::computeTuningScore(
  double callsPerSecond,
  double errorProp,
  double errorToleranceScale
) {
  if(!isfinite(callsPerSecond) || callsPerSecond <= 0.0 || !isfinite(errorProp) ||
     errorProp > std::min(0.5, 5.0 * errorToleranceScale))
    return 0.0;
  double penalty = 1.0 - sqrt(errorProp / (errorProp + errorToleranceScale));
  if(errorProp > 1e-5)
    penalty *= 0.90;
  return callsPerSecond * penalty;
}

double VulkanTuner::computeCooperativeMatrixTuningScore(
  double callsPerSecond,
  double errorProp,
  double errorTolerance
) {
  if(!isfinite(callsPerSecond) || callsPerSecond <= 0.0 || !isfinite(errorProp) ||
     errorProp > errorTolerance)
    return 0.0;
  return callsPerSecond;
}

bool VulkanTuner::isFastEnough(
  double callsPerSecond,
  double baselineCallsPerSecond,
  double requiredRatio
) {
  if(!isfinite(callsPerSecond) || !isfinite(baselineCallsPerSecond) ||
     callsPerSecond <= 0.0 || baselineCallsPerSecond <= 0.0)
    return false;
  const double threshold = baselineCallsPerSecond * requiredRatio;
  const double boundaryTolerance = std::max(1.0, fabs(threshold)) * 1e-12;
  return callsPerSecond + boundaryTolerance >= threshold;
}

namespace {
  bool isPositivePowerOfTwo(int value) {
    return value > 0 && (value & (value - 1)) == 0;
  }

#if defined(VK_KHR_cooperative_matrix)
  bool supportsCooperativeMatrixType(const VkCooperativeMatrixPropertiesKHR& property, int accType) {
    const VkComponentTypeKHR expectedType = accType == 16
      ? VK_COMPONENT_TYPE_FLOAT16_KHR
      : VK_COMPONENT_TYPE_FLOAT32_KHR;
    return property.scope == VK_SCOPE_SUBGROUP_KHR &&
           property.saturatingAccumulation == VK_FALSE &&
           property.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
           property.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
           property.CType == expectedType &&
           property.ResultType == expectedType;
  }

  bool supportsCooperativeMatrixShape(const VkCooperativeMatrixPropertiesKHR& property) {
    return isPositivePowerOfTwo(static_cast<int>(property.MSize)) &&
           isPositivePowerOfTwo(static_cast<int>(property.NSize)) &&
           isPositivePowerOfTwo(static_cast<int>(property.KSize));
  }

  int largestCooperativeMatrixVectorWidth(int dimension) {
    for(int width: {4, 2, 1}) {
      if(dimension % width == 0)
        return width;
    }
    return 0;
  }

  bool getSupportedCooperativeMatrixProperties(
    VkPhysicalDevice physicalDevice,
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getProperties,
    VkBool32 cooperativeMatrix,
    vector<VkCooperativeMatrixPropertiesKHR>& properties
  ) {
    if(cooperativeMatrix != VK_TRUE || getProperties == nullptr)
      return false;

    while(true) {
      uint32_t propertyCount = 0;
      VkResult result = getProperties(physicalDevice, &propertyCount, nullptr);
      if((result != VK_SUCCESS && result != VK_INCOMPLETE) || propertyCount == 0)
        return false;

      properties.assign(propertyCount, {});
      for(VkCooperativeMatrixPropertiesKHR& property: properties) {
        property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        property.pNext = nullptr;
      }
      result = getProperties(physicalDevice, &propertyCount, properties.data());
      if(result == VK_INCOMPLETE)
        continue;
      if(result != VK_SUCCESS)
        return false;
      properties.resize(propertyCount);
      break;
    }

    properties.erase(
      remove_if(properties.begin(), properties.end(), [](const VkCooperativeMatrixPropertiesKHR& property) {
        return !supportsCooperativeMatrixShape(property) ||
               (!supportsCooperativeMatrixType(property, 16) &&
                !supportsCooperativeMatrixType(property, 32));
      }),
      properties.end()
    );
    vector<VkCooperativeMatrixPropertiesKHR> uniqueProperties;
    for(const VkCooperativeMatrixPropertiesKHR& property: properties) {
      const bool duplicate = any_of(
        uniqueProperties.begin(), uniqueProperties.end(), [&](const VkCooperativeMatrixPropertiesKHR& previous) {
          return previous.MSize == property.MSize && previous.NSize == property.NSize &&
                 previous.KSize == property.KSize && previous.AType == property.AType &&
                 previous.BType == property.BType && previous.CType == property.CType &&
                 previous.ResultType == property.ResultType && previous.scope == property.scope;
        }
      );
      if(!duplicate)
        uniqueProperties.push_back(property);
    }
    properties = std::move(uniqueProperties);
    return !properties.empty();
  }

#endif
  struct CooperativeMatrixTuneShape {
    int accType;
    int MSize;
    int NSize;
    int KSize;
    uint32_t subgroupSize;
  };

  vector<CooperativeMatrixTuneShape> getCooperativeMatrixTuneShapes(const VulkanDevice* device) {
#if defined(VK_KHR_cooperative_matrix)
    vector<CooperativeMatrixTuneShape> shapes;
    if(device == nullptr)
      return shapes;

    vector<VkCooperativeMatrixPropertiesKHR> properties;
    if(!getSupportedCooperativeMatrixProperties(
         device->info.physicalDevice,
         device->info.cooperativeMatrixPropertiesFn,
         device->info.cooperativeMatrixFeatures.cooperativeMatrix,
         properties
       ))
      return shapes;

    for(const VkCooperativeMatrixPropertiesKHR& property: properties) {
      for(int accType: {16, 32}) {
        if(!supportsCooperativeMatrixType(property, accType))
          continue;
        const CooperativeMatrixTuneShape shape = {
          accType,
          static_cast<int>(property.MSize),
          static_cast<int>(property.NSize),
          static_cast<int>(property.KSize),
          device->info.subgroupProperties.subgroupSize
        };
        const bool alreadyAdded = any_of(
          shapes.begin(), shapes.end(), [&shape](const CooperativeMatrixTuneShape& previous) {
            return previous.accType == shape.accType &&
                   previous.MSize == shape.MSize &&
                   previous.NSize == shape.NSize &&
                   previous.KSize == shape.KSize;
          }
        );
        if(!alreadyAdded)
          shapes.push_back(shape);
      }
    }
    return shapes;

#else
    (void)device;
    return {};
#endif
  }

  bool supportsCooperativeMatrix(const VulkanDeviceInfo& deviceInfo) {
#if defined(VK_KHR_cooperative_matrix)
    vector<VkCooperativeMatrixPropertiesKHR> properties;
    return getSupportedCooperativeMatrixProperties(
      deviceInfo.physicalDevice,
      deviceInfo.cooperativeMatrixPropertiesFn,
      deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix,
      properties
    );

#else
    (void)deviceInfo;
    return false;
#endif
  }

  bool isSupportedCooperativeMatrixShape(
    const VulkanDeviceInfo& deviceInfo,
    int accType,
    int mSize,
    int nSize,
    int kSize,
    uint32_t subgroupSize
  ) {
#if defined(VK_KHR_cooperative_matrix) && defined(VK_EXT_subgroup_size_control)
    if(deviceInfo.subgroupSizeControlFeatures.computeFullSubgroups != VK_TRUE ||
       (deviceInfo.subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) == 0 ||
       (deviceInfo.subgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0)
      return false;
    vector<VkCooperativeMatrixPropertiesKHR> properties;
    if(!getSupportedCooperativeMatrixProperties(
         deviceInfo.physicalDevice,
         deviceInfo.cooperativeMatrixPropertiesFn,
         deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix,
         properties
       ))
      return false;
    return subgroupSize == deviceInfo.subgroupProperties.subgroupSize && any_of(
      properties.begin(), properties.end(), [&](const VkCooperativeMatrixPropertiesKHR& property) {
        return supportsCooperativeMatrixType(property, accType) &&
               static_cast<int>(property.MSize) == mSize &&
               static_cast<int>(property.NSize) == nSize &&
               static_cast<int>(property.KSize) == kSize;
      }
    );

#else
    (void)deviceInfo;
    (void)accType;
    (void)mSize;
    (void)nSize;
    (void)kSize;
    (void)subgroupSize;
    return false;
#endif
  }

  bool isWithinCooperativeMatrixDeviceLimits(
    const VulkanDeviceInfo& deviceInfo,
    const HGemmCooperativeMatrixTuneParams& params
  ) {
    if(!params.isValid())
      return false;
    const uint64_t localSizeX = static_cast<uint64_t>(params.MWAVE / params.MWARP) * params.subgroupSize;
    const uint64_t localSizeY = static_cast<uint64_t>(params.NWAVE / params.NWARP);
    const uint64_t sharedHalfElements =
      (params.SA == 1 ? static_cast<uint64_t>(params.MWG) * params.KWG : 0) +
      (params.SB == 1 ? static_cast<uint64_t>(params.NWG) * params.KWG : 0);
    const uint64_t accumulatorTileBytes = params.accType == 32
      ? static_cast<uint64_t>(params.MWG) * params.NWG * sizeof(float)
      : 0;
    const uint64_t sharedAlignmentBytes = 4 * sizeof(uint32_t) *
      ((params.SA == 1 ? 1 : 0) + (params.SB == 1 ? 1 : 0) + (params.accType == 32 ? 1 : 0));
    const VkPhysicalDeviceLimits& limits = deviceInfo.properties.limits;
    return localSizeX <= limits.maxComputeWorkGroupSize[0] &&
           localSizeY <= limits.maxComputeWorkGroupSize[1] &&
           1 <= limits.maxComputeWorkGroupSize[2] &&
           localSizeX * localSizeY <= limits.maxComputeWorkGroupInvocations &&
           sharedHalfElements * sizeof(uint16_t) + accumulatorTileBytes + sharedAlignmentBytes <=
             limits.maxComputeSharedMemorySize;
  }

  bool isWithinCooperativeMatrixDeviceLimits(
    const VulkanDeviceInfo& deviceInfo,
    const HGemmCooperativeMatrixNHWCTuneParams& params
  ) {
    if(!params.isValid())
      return false;
    const uint64_t localSizeX = static_cast<uint64_t>(params.MWAVE / params.MWARP) * params.subgroupSize;
    const uint64_t localSizeY = static_cast<uint64_t>(params.NWAVE / params.NWARP);
    const uint64_t sharedHalfElements =
      (params.SA == 1 ? static_cast<uint64_t>(params.MWG) * params.KWG : 0) +
      (params.SB == 1 ? static_cast<uint64_t>(params.KWG) * params.NWG : 0);
    const uint64_t accumulatorTileBytes = params.accType == 32
      ? static_cast<uint64_t>(params.MWG) * params.NWG * sizeof(float) : 0;
    const uint64_t sharedAlignmentBytes = 4 * sizeof(uint32_t) *
      ((params.SA == 1 ? 1 : 0) + (params.SB == 1 ? 1 : 0) + (params.accType == 32 ? 1 : 0));
    const VkPhysicalDeviceLimits& limits = deviceInfo.properties.limits;
    return localSizeX <= limits.maxComputeWorkGroupSize[0] &&
           localSizeY <= limits.maxComputeWorkGroupSize[1] &&
           1 <= limits.maxComputeWorkGroupSize[2] &&
           localSizeX * localSizeY <= limits.maxComputeWorkGroupInvocations &&
           sharedHalfElements * sizeof(uint16_t) + accumulatorTileBytes + sharedAlignmentBytes <=
             limits.maxComputeSharedMemorySize;
  }

  bool isWithinCooperativeMatrixDeviceLimits(
    const VulkanDeviceInfo& deviceInfo,
    const HGemmCooperativeMatrixNCHWTuneParams& params
  ) {
    if(!params.isValid())
      return false;
    const uint64_t localSizeX = static_cast<uint64_t>(params.MWAVE / params.MWARP) * params.subgroupSize;
    const uint64_t localSizeY = static_cast<uint64_t>(params.NWAVE / params.NWARP);
    const uint64_t accumulatorTileBytes = static_cast<uint64_t>(params.MWG) * params.NWG *
      (params.accType == 32 ? sizeof(float) : sizeof(uint16_t));
    const uint64_t sharedFilterBytes = params.SB == 1
      ? static_cast<uint64_t>(params.NWG) * params.KWG * sizeof(uint16_t)
      : 0;
    const uint64_t sharedAlignmentBytes = 4 * sizeof(uint32_t) * (params.SB == 1 ? 2 : 1);
    const VkPhysicalDeviceLimits& limits = deviceInfo.properties.limits;
    return localSizeX <= limits.maxComputeWorkGroupSize[0] &&
           localSizeY <= limits.maxComputeWorkGroupSize[1] &&
           1 <= limits.maxComputeWorkGroupSize[2] &&
           localSizeX * localSizeY <= limits.maxComputeWorkGroupInvocations &&
           accumulatorTileBytes + sharedFilterBytes + sharedAlignmentBytes <= limits.maxComputeSharedMemorySize;
  }

  bool isWithinCooperativeMatrixDeviceLimits(
    const VulkanDeviceInfo& deviceInfo,
    const TransformerDualGemmSwiGLUTuneParams& params
  ) {
    if(!params.isValid())
      return false;
    const uint64_t localSizeX = static_cast<uint64_t>(params.MWAVE / params.MWARP) * params.subgroupSize;
    const uint64_t localSizeY = static_cast<uint64_t>(params.NWAVE / params.NWARP);
    const VkPhysicalDeviceLimits& limits = deviceInfo.properties.limits;
    if(localSizeX > limits.maxComputeWorkGroupSize[0] ||
       localSizeY > limits.maxComputeWorkGroupSize[1] ||
       limits.maxComputeWorkGroupSize[2] < 1 ||
       localSizeX * localSizeY > limits.maxComputeWorkGroupInvocations)
      return false;

    const uint64_t sharedMemoryLimit = limits.maxComputeSharedMemorySize;
    const uint64_t tileElements = static_cast<uint64_t>(params.MWG) * params.KWG +
      (params.SB == 1 ? 2ull * params.KWG * params.NWG : 0);
    const uint64_t sharedBytes = tileElements * sizeof(uint16_t) * params.STAGES;
    const uint64_t alignmentBytes = 16ull * (1 + 2 * params.SB);
    return alignmentBytes <= sharedMemoryLimit && sharedBytes <= sharedMemoryLimit - alignmentBytes;
  }
}

VulkanParams VulkanTuner::getHardwareParams(const VulkanDeviceInfo& deviceInfo) {
  VulkanParams params;
  params.canUseFP16Storage =
    deviceInfo.storage16BitFeatures.storageBuffer16BitAccess == VK_TRUE ||
    deviceInfo.storage16BitFeatures.uniformAndStorageBuffer16BitAccess == VK_TRUE;
  params.canUseFP16Compute = deviceInfo.shaderFloat16Int8Features.shaderFloat16 == VK_TRUE;
#if defined(VK_EXT_subgroup_size_control)
  params.canUseSubgroup =
    (deviceInfo.subgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
    (deviceInfo.subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0 &&
    deviceInfo.subgroupSizeControlFeatures.computeFullSubgroups == VK_TRUE;
#else
  params.canUseSubgroup = false;
#endif
  params.canUseCooperativeMatrix =
    params.canUseSubgroup &&
    deviceInfo.vulkanMemoryModelFeatures.vulkanMemoryModel == VK_TRUE &&
    supportsCooperativeMatrix(deviceInfo);
  return params;
}

bool vk_shader::tune::isValidCooperativeMatrixConfig(
  const VulkanDeviceInfo& deviceInfo,
  const HGemmCooperativeMatrixTuneParams& params
) {
  return isWithinCooperativeMatrixDeviceLimits(deviceInfo, params) &&
         isSupportedCooperativeMatrixShape(
           deviceInfo, params.accType, params.MWARP, params.NWARP, params.KDIM, params.subgroupSize
         );
}

bool vk_shader::tune::isValidCooperativeMatrixConfig(
  const VulkanDeviceInfo& deviceInfo,
  const HGemmCooperativeMatrixNHWCTuneParams& params
) {
  return isWithinCooperativeMatrixDeviceLimits(deviceInfo, params) &&
         isSupportedCooperativeMatrixShape(
           deviceInfo, params.accType, params.MWARP, params.NWARP, params.KDIM, params.subgroupSize
         );
}

bool vk_shader::tune::isValidCooperativeMatrixConfig(
  const VulkanDeviceInfo& deviceInfo,
  const HGemmCooperativeMatrixNCHWTuneParams& params
) {
  return isWithinCooperativeMatrixDeviceLimits(deviceInfo, params) &&
         isSupportedCooperativeMatrixShape(
           deviceInfo, params.accType, params.MWARP, params.NWARP, params.KDIM, params.subgroupSize
         );
}

bool vk_shader::tune::isValidCooperativeMatrixConfig(
  const VulkanDeviceInfo& deviceInfo,
  const TransformerDualGemmSwiGLUTuneParams& params
) {
  return isWithinCooperativeMatrixDeviceLimits(deviceInfo, params) &&
         isSupportedCooperativeMatrixShape(
           deviceInfo, params.accType, params.MWARP, params.NWARP, params.KDIM, params.subgroupSize
         );
}

bool vk_shader::tune::canUseRegisterAttentionNHWC(
  const VulkanDeviceInfo& deviceInfo,
  const TransformerCooperativeMatrixTuneParams& params,
  int qHeadDim,
  int vHeadDim
) {
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
  const auto& limits = deviceInfo.cooperativeMatrix2Properties;
  if(params.COOP_USE_NV2 != 1 || !deviceInfo.canUseRegisterAttentionNHWC || qHeadDim != 32 || vHeadDim != 32 ||
     params.COOP_M_SIZE != 16 || params.COOP_N_SIZE != 16 || params.COOP_K_SIZE != 16 ||
     params.COOP_SUBGROUP_SIZE != 32 || params.COOP_ACC_TYPE != 32 || params.COOP_PV_N_SIZE != 16 ||
     params.ATTN_BLOCK_Q < 32 || params.ATTN_BLOCK_Q > 128 ||
     params.ATTN_BLOCK_KV < 32 || params.ATTN_BLOCK_KV > 128 ||
     limits.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize < 128 ||
     limits.cooperativeMatrixFlexibleDimensionsMaxDimension < uint32_t(std::max(params.ATTN_BLOCK_Q, params.ATTN_BLOCK_KV)))
    return false;
  for(const auto& p: deviceInfo.flexibleMatrixProperties)
    if(p.scope == VK_SCOPE_WORKGROUP_KHR && p.workgroupInvocations == 128 &&
       p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
       p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
       !p.saturatingAccumulation && p.MGranularity > 0 && p.NGranularity > 0 && p.KGranularity > 0 &&
       params.ATTN_BLOCK_Q % p.MGranularity == 0 && params.ATTN_BLOCK_KV % p.NGranularity == 0 &&
       32 % p.KGranularity == 0 && 32 % p.NGranularity == 0 && params.ATTN_BLOCK_KV % p.KGranularity == 0)
      return true;
  return false;
#else
  (void)deviceInfo;
  (void)params;
  (void)qHeadDim;
  (void)vHeadDim;
  return false;
#endif
}

bool vk_shader::tune::isValidCooperativeMatrixConfig(
  const VulkanDeviceInfo& deviceInfo,
  const TransformerCooperativeMatrixTuneParams& params,
  int qHeadDim,
  int vHeadDim,
  bool useNHWC
) {
  if(!params.isValid() || params.COOP_ACC_TYPE != 32 ||
     qHeadDim <= 0 || vHeadDim <= 0 ||
     params.COOP_M_SIZE > static_cast<int>(params.COOP_SUBGROUP_SIZE))
    return false;
  if(params.COOP_USE_NV2 && (!useNHWC || !canUseRegisterAttentionNHWC(deviceInfo, params, qHeadDim, vHeadDim)))
    return false;
  if(!isSupportedCooperativeMatrixShape(
       deviceInfo,
       params.COOP_ACC_TYPE,
       params.COOP_M_SIZE,
       params.COOP_N_SIZE,
       params.COOP_K_SIZE,
       params.COOP_SUBGROUP_SIZE
     ))
    return false;
  if(!isSupportedCooperativeMatrixShape(
       deviceInfo,
       params.COOP_ACC_TYPE,
       params.COOP_M_SIZE,
       params.COOP_PV_N_SIZE,
       params.COOP_N_SIZE,
       params.COOP_SUBGROUP_SIZE
     ))
    return false;

  const int headDimPad = ((qHeadDim + params.COOP_K_SIZE - 1) / params.COOP_K_SIZE) * params.COOP_K_SIZE;
  const int kvPad = ((params.COOP_N_SIZE + params.COOP_K_SIZE - 1) / params.COOP_K_SIZE) * params.COOP_K_SIZE;
  const int vHeadDimPad = ((vHeadDim + params.COOP_PV_N_SIZE - 1) / params.COOP_PV_N_SIZE) * params.COOP_PV_N_SIZE;
  const int qBlock = params.COOP_M_SIZE * static_cast<int>(params.COOP_Q_TILES_PER_WORKGROUP);
  const int kvStages = params.ATTN_BLOCK_KV / params.COOP_N_SIZE;
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
  if(useNHWC && canUseRegisterAttentionNHWC(deviceInfo, params, qHeadDim, vHeadDim)) {
    const uint64_t shared = 4 * 16ull + 2ull * qBlock * 32 + 4ull * params.ATTN_BLOCK_KV * 32 +
      4ull * qBlock * 32 + 4ull * (qBlock + params.ATTN_BLOCK_KV) +
      deviceInfo.cooperativeMatrix2Properties.cooperativeMatrixWorkgroupScopeReservedSharedMemory;
    const auto& limits = deviceInfo.properties.limits;
    return params.getLocalSizeX() <= limits.maxComputeWorkGroupSize[0] &&
      params.getLocalSizeX() <= limits.maxComputeWorkGroupInvocations &&
      shared <= limits.maxComputeSharedMemorySize;
  }
#endif
  const uint64_t localSize = params.getLocalSizeX();
  const uint64_t sharedBytes =
    6 * 16ull +
    2ull * (
      static_cast<uint64_t>(headDimPad) * qBlock +
      static_cast<uint64_t>(headDimPad) * params.ATTN_BLOCK_KV +
      static_cast<uint64_t>(kvPad) * kvStages * vHeadDimPad +
      static_cast<uint64_t>(qBlock) * params.COOP_N_SIZE
    ) +
    static_cast<uint64_t>(params.COOP_ACC_TYPE == 32 ? 4 : 2) *
      qBlock * params.COOP_N_SIZE +
    4ull * qBlock * vHeadDimPad +
    4ull * params.ATTN_BLOCK_KV;
  const VkPhysicalDeviceLimits& limits = deviceInfo.properties.limits;
  return localSize <= limits.maxComputeWorkGroupSize[0] &&
         localSize <= limits.maxComputeWorkGroupInvocations &&
         sharedBytes <= limits.maxComputeSharedMemorySize;
}

namespace {
  const string VERSION_LINE = Global::strprintf("VERSION=%d", VulkanTuner::TUNER_VERSION);

  template<typename Function>
  class ScopeGuard {
   public:
    explicit ScopeGuard(Function&& function): function(std::forward<Function>(function)) {}
    ~ScopeGuard() noexcept {
      if(active)
        function();
    }
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    void dismiss() { active = false; }

   private:
    Function function;
    bool active = true;
  };

  template<typename Function>
  ScopeGuard<Function> makeScopeGuard(Function&& function) {
    return ScopeGuard<Function>(std::forward<Function>(function));
  }

  bool isMultipleOf(uint64_t x, uint64_t y) {
    return y != 0 && x % y == 0;
  }

  void writeParam(
    ofstream& out,
    const char* name,
    uint32_t value
  ) {
    out << name << "=" << value << "\n";
  }

  void writeParam(
    ofstream& out,
    const string& name,
    uint32_t value
  ) {
    out << name << "=" << value << "\n";
  }

  uint32_t getParam(
    const map<string, string>& values,
    const string& name,
    const string& filename
  ) {
    auto iter = values.find(name);
    if(iter == values.end())
      throw IOError("VulkanTuneParams::load: missing parameter " + name + " in " + filename);
    int value;
    if(!Global::tryStringToInt(iter->second, value) || value < 0)
      throw IOError("VulkanTuneParams::load: invalid integer for " + name + " in " + filename);
    return static_cast<uint32_t>(value);
  }

  bool getBoolParam(
    const map<string, string>& values,
    const string& name,
    const string& filename
  ) {
    uint32_t value = getParam(values, name, filename);
    if(value > 1)
      throw IOError("VulkanTuneParams::load: invalid boolean for " + name + " in " + filename);
    return value == 1;
  }

}  // namespace

bool AddPointWiseTuneParams::isValid() const {
  return LOCAL_SIZE >= 32 && LOCAL_SIZE <= 512 &&
         (LOCAL_SIZE & (LOCAL_SIZE - 1)) == 0 &&
         ELTS_PER_THREAD > 0 && ELTS_PER_THREAD <= 32 &&
         (ELTS_PER_THREAD & (ELTS_PER_THREAD - 1)) == 0;
}

bool AddChannelBiasesNCHWTuneParams::isValid() const {
  return XY_ELTS_PER_THREAD > 0 && XY_ELTS_PER_THREAD <= 4 &&
         (XY_ELTS_PER_THREAD & (XY_ELTS_PER_THREAD - 1)) == 0 &&
         NC_ELTS_PER_THREAD > 0 && NC_ELTS_PER_THREAD <= 8 &&
         (NC_ELTS_PER_THREAD & (NC_ELTS_PER_THREAD - 1)) == 0;
}

bool GPoolTuneParams::isValid() const {
  if(XYSTRIDE <= 0 || CHANNELSTRIDE <= 0 || BATCHSTRIDE <= 0)
    return false;
  if((XYSTRIDE & (XYSTRIDE - 1)) != 0)
    return false;
  return static_cast<uint64_t>(XYSTRIDE) * CHANNELSTRIDE * BATCHSTRIDE <= 1024;
}

bool ConvTuneParams::isValid(uint32_t convSize) const {
  const bool supportedTileSize = convSize == 3
    ? ((inTileXSize == 4 && inTileYSize == 4 && outTileXSize == 2 && outTileYSize == 2) ||
       (inTileXSize == 6 && inTileYSize == 6 && outTileXSize == 4 && outTileYSize == 4))
    : (convSize == 5 && inTileXSize == 6 && inTileYSize == 6 && outTileXSize == 2 && outTileYSize == 2);
  if(!supportedTileSize)
    return false;
  if(inputTransformLocalXSize == 0 || inputTransformLocalYSize == 0 ||
     outputTransformLocalXSize == 0 || outputTransformLocalYSize == 0 || outputTransformLocalZSize == 0 ||
     inputTransformNHWCLocalXSize == 0 || inputTransformNHWCLocalYSize == 0 ||
     outputTransformNHWCLocalXSize == 0 || outputTransformNHWCLocalYSize == 0)
    return false;
  return static_cast<uint64_t>(inputTransformLocalXSize) * inputTransformLocalYSize <= 1024 &&
         static_cast<uint64_t>(outputTransformLocalXSize) * outputTransformLocalYSize * outputTransformLocalZSize <= 1024 &&
         static_cast<uint64_t>(inputTransformNHWCLocalXSize) * inputTransformNHWCLocalYSize <= 1024 &&
         static_cast<uint64_t>(outputTransformNHWCLocalXSize) * outputTransformNHWCLocalYSize <= 1024;
}

bool XgemmTuneParams::isValid() const {
  if(MDIMC == 0 || NDIMC == 0 || MWG == 0 || NWG == 0 || KWG == 0 || KWI == 0 || MDIMA == 0 || NDIMB == 0 ||
     (VWM != 1 && VWM != 2 && VWM != 4) || (VWN != 1 && VWN != 2 && VWN != 4))
    return false;
  const uint64_t workgroupSize = static_cast<uint64_t>(MDIMC) * NDIMC;
  if(workgroupSize == 0 || workgroupSize > 1024)
    return false;
  return isMultipleOf(MWG, static_cast<uint64_t>(MDIMC) * VWM) &&
         isMultipleOf(NWG, static_cast<uint64_t>(NDIMC) * VWN) &&
         isMultipleOf(MWG, static_cast<uint64_t>(MDIMA) * VWM) &&
         isMultipleOf(NWG, static_cast<uint64_t>(NDIMB) * VWN) &&
         isMultipleOf(KWG, VWM) &&
         isMultipleOf(KWG, KWI) &&
         isMultipleOf(KWG, workgroupSize / MDIMA) &&
         isMultipleOf(KWG, workgroupSize / NDIMB);
}

bool XgemmTuneParams::isSimple() const {
  return MDIMC == MDIMA && NDIMC == NDIMB && VWM == VWN && MWG == NWG;
}

bool XgemmDirectTuneParams::isValid() const {
  if(WGD == 0 || MDIMCD == 0 || NDIMCD == 0 || MDIMAD == 0 || NDIMBD == 0 || KWID == 0 ||
     (VWMD != 1 && VWMD != 2 && VWMD != 4) || (VWND != 1 && VWND != 2 && VWND != 4))
    return false;
  if(PADA != 1 || PADB != 1)
    return false;
  const uint64_t workgroupSize = static_cast<uint64_t>(MDIMCD) * NDIMCD;
  if(workgroupSize > 1024)
    return false;
  if(!isMultipleOf(WGD, KWID) ||
     !isMultipleOf(WGD, static_cast<uint64_t>(MDIMCD) * VWMD) ||
     !isMultipleOf(WGD, static_cast<uint64_t>(NDIMCD) * VWND) ||
     !isMultipleOf(WGD, static_cast<uint64_t>(MDIMAD) * VWMD) ||
     !isMultipleOf(WGD, static_cast<uint64_t>(NDIMBD) * VWND))
    return false;
  return isMultipleOf(WGD, workgroupSize / MDIMAD) &&
         isMultipleOf(WGD, workgroupSize / NDIMBD);
}

bool HGemmCooperativeMatrixTuneParams::isValid() const {
  if(MWARP <= 0 || NWARP <= 0 || KDIM <= 0 || subgroupSize == 0 ||
     MWG <= 0 || NWG <= 0 || KWG <= 0 || MWAVE <= 0 || NWAVE <= 0 ||
     (accType != 16 && accType != 32) ||
     SA < 0 || SA > 1 || SB < 0 || SB > 1 ||
     (VWM != 1 && VWM != 2 && VWM != 4) || (VWN != 1 && VWN != 2 && VWN != 4))
    return false;
  if(!isPositivePowerOfTwo(MWARP) || !isPositivePowerOfTwo(NWARP) || !isPositivePowerOfTwo(KDIM))
    return false;
  if(!isMultipleOf(MWG, MWAVE) || !isMultipleOf(NWG, NWAVE) ||
     !isMultipleOf(KWG, KDIM) || !isMultipleOf(MWAVE, MWARP) ||
     !isMultipleOf(NWAVE, NWARP))
    return false;
  const int subgroupCountM = MWAVE / MWARP;
  const int subgroupCountN = NWAVE / NWARP;
  const int reuseM = MWG / MWAVE;
  const int reuseN = NWG / NWAVE;
  const int reuseK = KWG / KDIM;
  if(!isPositivePowerOfTwo(subgroupCountM) || !isPositivePowerOfTwo(subgroupCountN) ||
     !isPositivePowerOfTwo(reuseM) || !isPositivePowerOfTwo(reuseN) ||
     !isPositivePowerOfTwo(reuseK))
    return false;
  const uint64_t localSizeX = static_cast<uint64_t>(subgroupCountM) * subgroupCountN * subgroupSize;
  if(localSizeX == 0 || localSizeX > 1024)
    return false;
  return isMultipleOf(MWG, VWM) && isMultipleOf(NWG, VWN) &&
         isMultipleOf(MWARP, VWM) && isMultipleOf(NWARP, VWN);
}

bool HGemmCooperativeMatrixTuneParams::isSimple() const {
  if(MWAVE != MWARP && MWAVE == MWG)
    return false;
  if(NWAVE != NWARP && NWAVE == NWG)
    return false;
  return SA == SB && VWM == VWN && MWG == NWG;
}

bool HGemmCooperativeMatrixNCHWTuneParams::isValid() const {
  if(MWARP <= 0 || NWARP <= 0 || KDIM <= 0 || subgroupSize == 0 ||
     MWG <= 0 || NWG <= 0 || KWG <= 0 ||
     MWAVE <= 0 || NWAVE <= 0 || SB < 0 || SB > 1 ||
     (accType != 16 && accType != 32) ||
     (VWM != 1 && VWM != 2 && VWM != 4) || (VWN != 1 && VWN != 2 && VWN != 4))
    return false;
  if(!isPositivePowerOfTwo(MWARP) || !isPositivePowerOfTwo(NWARP) || !isPositivePowerOfTwo(KDIM))
    return false;
  if(!isMultipleOf(MWG, MWAVE) || !isMultipleOf(NWG, NWAVE) ||
     !isMultipleOf(KWG, KDIM) || !isMultipleOf(MWAVE, MWARP) ||
     !isMultipleOf(NWAVE, NWARP))
    return false;
  const int subgroupCountM = MWAVE / MWARP;
  const int subgroupCountN = NWAVE / NWARP;
  const int reuseM = MWG / MWAVE;
  const int reuseN = NWG / NWAVE;
  const int reuseK = KWG / KDIM;
  if(!isPositivePowerOfTwo(subgroupCountM) || !isPositivePowerOfTwo(subgroupCountN) ||
     !isPositivePowerOfTwo(reuseM) || !isPositivePowerOfTwo(reuseN) ||
     !isPositivePowerOfTwo(reuseK))
    return false;
  const uint64_t localSizeX = static_cast<uint64_t>(subgroupCountM) * subgroupCountN * subgroupSize;
  if(localSizeX == 0 || localSizeX > 1024)
    return false;
  return isMultipleOf(MWG, VWM) && isMultipleOf(NWG, VWN) &&
         isMultipleOf(MWARP, VWM) && isMultipleOf(NWARP, VWN);
}

int HGemmCooperativeMatrixNCHWTuneParams::getRequiredSpatialAlignment() const {
  return std::lcm(std::lcm(16, MWARP), VWM);
}

bool HGemmCooperativeMatrixNHWCTuneParams::isValid() const {
  if(MWARP <= 0 || NWARP <= 0 || KDIM <= 0 || subgroupSize == 0 ||
     MWG <= 0 || NWG <= 0 || KWG <= 0 || MWAVE <= 0 || NWAVE <= 0 ||
     (accType != 16 && accType != 32) || SA < 0 || SA > 1 || SB < 0 || SB > 1 ||
     DB < 0 || DB > 1 ||
     (VWK != 1 && VWK != 2 && VWK != 4) || (VWN != 1 && VWN != 2 && VWN != 4))
    return false;
  if((SA == 0 && VWK != 1) || (SB == 0 && VWN != 1 && accType == 16))
    return false;
  if(!isPositivePowerOfTwo(MWARP) || !isPositivePowerOfTwo(NWARP) || !isPositivePowerOfTwo(KDIM))
    return false;
  if(!isMultipleOf(MWG, MWAVE) || !isMultipleOf(NWG, NWAVE) ||
     !isMultipleOf(KWG, KDIM) || !isMultipleOf(MWAVE, MWARP) ||
     !isMultipleOf(NWAVE, NWARP))
    return false;
  const int subgroupCountM = MWAVE / MWARP;
  const int subgroupCountN = NWAVE / NWARP;
  const int reuseM = MWG / MWAVE;
  const int reuseN = NWG / NWAVE;
  const int reuseK = KWG / KDIM;
  if(!isPositivePowerOfTwo(subgroupCountM) || !isPositivePowerOfTwo(subgroupCountN) ||
     !isPositivePowerOfTwo(reuseM) || !isPositivePowerOfTwo(reuseN) ||
     !isPositivePowerOfTwo(reuseK))
    return false;
  const uint64_t localSizeX = static_cast<uint64_t>(subgroupCountM) * subgroupCountN * subgroupSize;
  if(localSizeX == 0 || localSizeX > 1024)
    return false;
  return (!SA || isMultipleOf(KWG, VWK)) && (!SB || isMultipleOf(NWG, VWN));
}

bool HGemmCooperativeMatrixNHWCTuneParams::isSimple() const {
  return MWAVE == MWG && NWAVE == NWG && SA == SB && DB == 0 && VWK == VWN;
}

int TransformerDualGemmSwiGLUTuneParams::getRequiredSpatialAlignment() const {
  return std::lcm(HGemmCooperativeMatrixNCHWTuneParams::getRequiredSpatialAlignment(), MWG);
}

bool HGemmCooperativeMatrixNCHWTuneParams::isSimple() const {
  if(MWAVE != MWARP && MWAVE == MWG)
    return false;
  if(NWAVE != NWARP && NWAVE == NWG)
    return false;
  return MWG == NWG;
}

bool TransformerTuneParams::isValid() const {
  if(USE_TILED_ATTN != 0 && USE_TILED_ATTN != 1)
    return false;
  if(ATTN_BLOCK_Q <= 0 || ATTN_BLOCK_KV <= 0)
    return false;
  if((ATTN_BLOCK_Q & (ATTN_BLOCK_Q - 1)) != 0 ||
     (ATTN_BLOCK_KV & (ATTN_BLOCK_KV - 1)) != 0)
    return false;
  if(ATTN_BLOCK_Q > 256 || ATTN_BLOCK_KV > 128)
    return false;
  if(Q_PER_THREAD < 1 || Q_PER_THREAD > 8 || (Q_PER_THREAD & (Q_PER_THREAD - 1)) != 0)
    return false;
  return true;
}

bool TransformerTuneParams::isSupported(const VkPhysicalDeviceLimits& limits, int qHeadDim, int vHeadDim) const {
  if(!isValid() || qHeadDim <= 0 || vHeadDim <= 0)
    return false;
  const uint32_t localSize = USE_TILED_ATTN ? static_cast<uint32_t>(ATTN_BLOCK_Q) : NAIVE_LOCAL_SIZE;
  const uint64_t sharedBytes = USE_TILED_ATTN
    ? sizeof(float) * static_cast<uint64_t>(ATTN_BLOCK_KV) * (static_cast<uint64_t>(qHeadDim) + vHeadDim + 1)
    : 0;
  return localSize <= limits.maxComputeWorkGroupSize[0] &&
    limits.maxComputeWorkGroupSize[1] >= 1 && limits.maxComputeWorkGroupSize[2] >= 1 &&
    localSize <= limits.maxComputeWorkGroupInvocations && sharedBytes <= limits.maxComputeSharedMemorySize;
}

uint32_t TransformerTuneParams::getDispatchGroupCountX(uint32_t seqLen) const {
  const size_t queriesPerGroup = USE_TILED_ATTN ? static_cast<size_t>(ATTN_BLOCK_Q) * Q_PER_THREAD : NAIVE_LOCAL_SIZE;
  const size_t queryCount = USE_TILED_ATTN ? seqLen : vk_helper::powerOf2ify(seqLen);
  return static_cast<uint32_t>((queryCount + queriesPerGroup - 1) / queriesPerGroup);
}

uint32_t TransformerCooperativeMatrixTuneParams::getLocalSizeX() const {
  return COOP_USE_NV2 == 1 ? 128u : COOP_SUBGROUP_SIZE * COOP_Q_TILES_PER_WORKGROUP;
}

bool TransformerCooperativeMatrixTuneParams::isValid() const {
  if(COOP_USE_NV2 != 0 && COOP_USE_NV2 != 1)
    return false;
  if(COOP_ACC_TYPE != 32)
    return false;
  if(COOP_M_SIZE <= 0 || COOP_N_SIZE <= 0 || COOP_K_SIZE <= 0 || COOP_SUBGROUP_SIZE == 0)
    return false;
  if(COOP_PV_N_SIZE <= 0 || (COOP_PV_N_SIZE & (COOP_PV_N_SIZE - 1)) != 0)
    return false;
  if(COOP_Q_TILES_PER_WORKGROUP == 0 ||
     (COOP_Q_TILES_PER_WORKGROUP & (COOP_Q_TILES_PER_WORKGROUP - 1)) != 0)
    return false;
  if(ATTN_BLOCK_Q <= 0 || ATTN_BLOCK_KV <= 0)
    return false;
  if((ATTN_BLOCK_Q & (ATTN_BLOCK_Q - 1)) != 0 ||
     (ATTN_BLOCK_KV & (ATTN_BLOCK_KV - 1)) != 0)
    return false;
  if(ATTN_BLOCK_Q > 256 || ATTN_BLOCK_KV > 128)
    return false;
  if(ATTN_BLOCK_Q != COOP_M_SIZE * static_cast<int>(COOP_Q_TILES_PER_WORKGROUP) ||
     ATTN_BLOCK_KV < COOP_N_SIZE || ATTN_BLOCK_KV % COOP_N_SIZE != 0 ||
     ATTN_BLOCK_KV / COOP_N_SIZE > 8)
    return false;
  return true;
}

bool vk_shader::tune::canUseTransformerRMSNormSubgroup(
  const VulkanDeviceInfo& deviceInfo,
  const TransformerRMSNormTuneParms& params
) {
#if defined(VK_EXT_subgroup_size_control)
  const auto& properties = deviceInfo.subgroupProperties;
  const auto required = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
  const auto& limits = deviceInfo.properties.limits;
  const uint64_t invocations = static_cast<uint64_t>(params.WG_C_SIZE) * params.WG_XY_SIZE;
  return params.isValid() && params.WG_C_SIZE == static_cast<int>(properties.subgroupSize) &&
    static_cast<uint32_t>(params.WG_XY_SIZE) <= deviceInfo.subgroupSizeProperties.maxComputeWorkgroupSubgroups &&
    invocations <= limits.maxComputeWorkGroupSize[0] && invocations <= limits.maxComputeWorkGroupInvocations &&
    invocations * sizeof(float) <= limits.maxComputeSharedMemorySize &&
    (properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
    (properties.supportedOperations & required) == required &&
    deviceInfo.subgroupSizeControlFeatures.computeFullSubgroups == VK_TRUE;
#else
  (void)deviceInfo;
  (void)params;
  return false;
#endif
}

bool TransformerRMSNormTuneParms::isValid() const {
  if(MODE < 0 || MODE > 2)
    return false;
  if(WG_C_SIZE <= 0 || WG_C_SIZE > 1024)
    return false;
  if((WG_C_SIZE & (WG_C_SIZE - 1)) != 0)
    return false;
  if(WG_XY_SIZE <= 0 || WG_XY_SIZE > 32)
    return false;
  if((WG_XY_SIZE & (WG_XY_SIZE - 1)) != 0)
    return false;
  if(static_cast<uint64_t>(WG_C_SIZE) * WG_XY_SIZE > 1024)
    return false;
  if(C_PER_THREAD <= 0 || C_PER_THREAD > 32)
    return false;
  if((C_PER_THREAD & (C_PER_THREAD - 1)) != 0)
    return false;
  return true;
}

bool TransformerSpatialRmsNormTuneParams::isValid() const {
  return TILE_SIZE > 0 && TILE_SIZE <= 1024 &&
         (TILE_SIZE & (TILE_SIZE - 1)) == 0 &&
         APPLY_ELTS_PER_THREAD > 0 && APPLY_ELTS_PER_THREAD <= 32 &&
         (APPLY_ELTS_PER_THREAD & (APPLY_ELTS_PER_THREAD - 1)) == 0;
}

bool VulkanTuningProfile::isValid() const {
  return addChannelBiases.isValid() && pointwise.isValid() && gPool.isValid() &&
         conv3x3.isValid(3) && conv5x5.isValid(5) && hgemmCooperativeMatrix.isValid() &&
         hgemmCooperativeMatrixNHWC.isValid() && transformerHgemmNHWC.isValid() &&
         hgemmCooperativeMatrixNCHW.isValid() &&
         transformerDualGemmSwiGLU.isValid() &&
         xgemm.isValid() && xgemm16.isValid() && xgemmDirect.isValid() && transformerXgemm.isValid() &&
         transformer.isValid() && transformerCooperativeMatrixNCHW.isValid() &&
         transformerCooperativeMatrixNHWC.isValid() && rmsNorm.isValid() && transformerPreNorm.isValid() && spatialRMSNorm.isValid();
}

bool VulkanTuningProfile::operator==(const VulkanTuningProfile& other) const {
  return addChannelBiases.XY_ELTS_PER_THREAD == other.addChannelBiases.XY_ELTS_PER_THREAD &&
         addChannelBiases.NC_ELTS_PER_THREAD == other.addChannelBiases.NC_ELTS_PER_THREAD &&
         pointwise.LOCAL_SIZE == other.pointwise.LOCAL_SIZE &&
         pointwise.ELTS_PER_THREAD == other.pointwise.ELTS_PER_THREAD &&
         gPool.XYSTRIDE == other.gPool.XYSTRIDE &&
         gPool.CHANNELSTRIDE == other.gPool.CHANNELSTRIDE &&
         gPool.BATCHSTRIDE == other.gPool.BATCHSTRIDE &&
         conv3x3.inTileYSize == other.conv3x3.inTileYSize && conv3x3.inTileXSize == other.conv3x3.inTileXSize &&
         conv3x3.outTileYSize == other.conv3x3.outTileYSize && conv3x3.outTileXSize == other.conv3x3.outTileXSize &&
         conv3x3.inputTransformLocalXSize == other.conv3x3.inputTransformLocalXSize &&
         conv3x3.inputTransformLocalYSize == other.conv3x3.inputTransformLocalYSize &&
         conv3x3.outputTransformLocalXSize == other.conv3x3.outputTransformLocalXSize &&
         conv3x3.outputTransformLocalYSize == other.conv3x3.outputTransformLocalYSize &&
         conv3x3.outputTransformLocalZSize == other.conv3x3.outputTransformLocalZSize &&
         conv3x3.inputTransformNHWCLocalXSize == other.conv3x3.inputTransformNHWCLocalXSize &&
         conv3x3.inputTransformNHWCLocalYSize == other.conv3x3.inputTransformNHWCLocalYSize &&
         conv3x3.outputTransformNHWCLocalXSize == other.conv3x3.outputTransformNHWCLocalXSize &&
         conv3x3.outputTransformNHWCLocalYSize == other.conv3x3.outputTransformNHWCLocalYSize &&
         conv5x5.inTileYSize == other.conv5x5.inTileYSize && conv5x5.inTileXSize == other.conv5x5.inTileXSize &&
         conv5x5.outTileYSize == other.conv5x5.outTileYSize && conv5x5.outTileXSize == other.conv5x5.outTileXSize &&
         conv5x5.inputTransformLocalXSize == other.conv5x5.inputTransformLocalXSize &&
         conv5x5.inputTransformLocalYSize == other.conv5x5.inputTransformLocalYSize &&
         conv5x5.outputTransformLocalXSize == other.conv5x5.outputTransformLocalXSize &&
         conv5x5.outputTransformLocalYSize == other.conv5x5.outputTransformLocalYSize &&
         conv5x5.outputTransformLocalZSize == other.conv5x5.outputTransformLocalZSize &&
         conv5x5.inputTransformNHWCLocalXSize == other.conv5x5.inputTransformNHWCLocalXSize &&
         conv5x5.inputTransformNHWCLocalYSize == other.conv5x5.inputTransformNHWCLocalYSize &&
         conv5x5.outputTransformNHWCLocalXSize == other.conv5x5.outputTransformNHWCLocalXSize &&
         conv5x5.outputTransformNHWCLocalYSize == other.conv5x5.outputTransformNHWCLocalYSize &&
         hgemmCooperativeMatrix.MWARP == other.hgemmCooperativeMatrix.MWARP &&
         hgemmCooperativeMatrix.NWARP == other.hgemmCooperativeMatrix.NWARP &&
         hgemmCooperativeMatrix.KDIM == other.hgemmCooperativeMatrix.KDIM &&
         hgemmCooperativeMatrix.subgroupSize == other.hgemmCooperativeMatrix.subgroupSize &&
         hgemmCooperativeMatrix.MWG == other.hgemmCooperativeMatrix.MWG &&
         hgemmCooperativeMatrix.NWG == other.hgemmCooperativeMatrix.NWG &&
         hgemmCooperativeMatrix.KWG == other.hgemmCooperativeMatrix.KWG &&
         hgemmCooperativeMatrix.MWAVE == other.hgemmCooperativeMatrix.MWAVE &&
         hgemmCooperativeMatrix.NWAVE == other.hgemmCooperativeMatrix.NWAVE &&
         hgemmCooperativeMatrix.accType == other.hgemmCooperativeMatrix.accType &&
         hgemmCooperativeMatrix.SA == other.hgemmCooperativeMatrix.SA &&
         hgemmCooperativeMatrix.SB == other.hgemmCooperativeMatrix.SB &&
         hgemmCooperativeMatrix.VWM == other.hgemmCooperativeMatrix.VWM &&
         hgemmCooperativeMatrix.VWN == other.hgemmCooperativeMatrix.VWN &&
         hgemmCooperativeMatrixNHWC.MWARP == other.hgemmCooperativeMatrixNHWC.MWARP &&
         hgemmCooperativeMatrixNHWC.NWARP == other.hgemmCooperativeMatrixNHWC.NWARP &&
         hgemmCooperativeMatrixNHWC.KDIM == other.hgemmCooperativeMatrixNHWC.KDIM &&
         hgemmCooperativeMatrixNHWC.subgroupSize == other.hgemmCooperativeMatrixNHWC.subgroupSize &&
         hgemmCooperativeMatrixNHWC.MWG == other.hgemmCooperativeMatrixNHWC.MWG &&
         hgemmCooperativeMatrixNHWC.NWG == other.hgemmCooperativeMatrixNHWC.NWG &&
         hgemmCooperativeMatrixNHWC.KWG == other.hgemmCooperativeMatrixNHWC.KWG &&
         hgemmCooperativeMatrixNHWC.MWAVE == other.hgemmCooperativeMatrixNHWC.MWAVE &&
         hgemmCooperativeMatrixNHWC.NWAVE == other.hgemmCooperativeMatrixNHWC.NWAVE &&
         hgemmCooperativeMatrixNHWC.accType == other.hgemmCooperativeMatrixNHWC.accType &&
         hgemmCooperativeMatrixNHWC.SA == other.hgemmCooperativeMatrixNHWC.SA &&
         hgemmCooperativeMatrixNHWC.SB == other.hgemmCooperativeMatrixNHWC.SB &&
         hgemmCooperativeMatrixNHWC.DB == other.hgemmCooperativeMatrixNHWC.DB &&
         hgemmCooperativeMatrixNHWC.VWK == other.hgemmCooperativeMatrixNHWC.VWK &&
         hgemmCooperativeMatrixNHWC.VWN == other.hgemmCooperativeMatrixNHWC.VWN &&
         transformerHgemmNHWC.MWARP == other.transformerHgemmNHWC.MWARP &&
         transformerHgemmNHWC.NWARP == other.transformerHgemmNHWC.NWARP &&
         transformerHgemmNHWC.KDIM == other.transformerHgemmNHWC.KDIM &&
         transformerHgemmNHWC.subgroupSize == other.transformerHgemmNHWC.subgroupSize &&
         transformerHgemmNHWC.MWG == other.transformerHgemmNHWC.MWG &&
         transformerHgemmNHWC.NWG == other.transformerHgemmNHWC.NWG &&
         transformerHgemmNHWC.KWG == other.transformerHgemmNHWC.KWG &&
         transformerHgemmNHWC.MWAVE == other.transformerHgemmNHWC.MWAVE &&
         transformerHgemmNHWC.NWAVE == other.transformerHgemmNHWC.NWAVE &&
         transformerHgemmNHWC.accType == other.transformerHgemmNHWC.accType &&
         transformerHgemmNHWC.SA == other.transformerHgemmNHWC.SA &&
         transformerHgemmNHWC.SB == other.transformerHgemmNHWC.SB &&
         transformerHgemmNHWC.DB == other.transformerHgemmNHWC.DB &&
         transformerHgemmNHWC.VWK == other.transformerHgemmNHWC.VWK &&
         transformerHgemmNHWC.VWN == other.transformerHgemmNHWC.VWN &&
          hgemmCooperativeMatrixNCHW.MWARP == other.hgemmCooperativeMatrixNCHW.MWARP &&
         hgemmCooperativeMatrixNCHW.NWARP == other.hgemmCooperativeMatrixNCHW.NWARP &&
         hgemmCooperativeMatrixNCHW.KDIM == other.hgemmCooperativeMatrixNCHW.KDIM &&
         hgemmCooperativeMatrixNCHW.subgroupSize == other.hgemmCooperativeMatrixNCHW.subgroupSize &&
         hgemmCooperativeMatrixNCHW.MWG == other.hgemmCooperativeMatrixNCHW.MWG &&
         hgemmCooperativeMatrixNCHW.NWG == other.hgemmCooperativeMatrixNCHW.NWG &&
         hgemmCooperativeMatrixNCHW.KWG == other.hgemmCooperativeMatrixNCHW.KWG &&
         hgemmCooperativeMatrixNCHW.MWAVE == other.hgemmCooperativeMatrixNCHW.MWAVE &&
         hgemmCooperativeMatrixNCHW.NWAVE == other.hgemmCooperativeMatrixNCHW.NWAVE &&
         hgemmCooperativeMatrixNCHW.accType == other.hgemmCooperativeMatrixNCHW.accType &&
         hgemmCooperativeMatrixNCHW.SB == other.hgemmCooperativeMatrixNCHW.SB &&
         hgemmCooperativeMatrixNCHW.VWM == other.hgemmCooperativeMatrixNCHW.VWM &&
         hgemmCooperativeMatrixNCHW.VWN == other.hgemmCooperativeMatrixNCHW.VWN &&
         transformerDualGemmSwiGLU.MWARP == other.transformerDualGemmSwiGLU.MWARP &&
         transformerDualGemmSwiGLU.NWARP == other.transformerDualGemmSwiGLU.NWARP &&
         transformerDualGemmSwiGLU.KDIM == other.transformerDualGemmSwiGLU.KDIM &&
         transformerDualGemmSwiGLU.subgroupSize == other.transformerDualGemmSwiGLU.subgroupSize &&
         transformerDualGemmSwiGLU.MWG == other.transformerDualGemmSwiGLU.MWG &&
         transformerDualGemmSwiGLU.NWG == other.transformerDualGemmSwiGLU.NWG &&
         transformerDualGemmSwiGLU.KWG == other.transformerDualGemmSwiGLU.KWG &&
         transformerDualGemmSwiGLU.MWAVE == other.transformerDualGemmSwiGLU.MWAVE &&
         transformerDualGemmSwiGLU.NWAVE == other.transformerDualGemmSwiGLU.NWAVE &&
         transformerDualGemmSwiGLU.accType == other.transformerDualGemmSwiGLU.accType &&
         transformerDualGemmSwiGLU.SB == other.transformerDualGemmSwiGLU.SB &&
         transformerDualGemmSwiGLU.STAGES == other.transformerDualGemmSwiGLU.STAGES &&
         transformerDualGemmSwiGLU.VWM == other.transformerDualGemmSwiGLU.VWM &&
         transformerDualGemmSwiGLU.VWN == other.transformerDualGemmSwiGLU.VWN &&
         xgemm.MDIMC == other.xgemm.MDIMC && xgemm.NDIMC == other.xgemm.NDIMC && xgemm.MWG == other.xgemm.MWG &&
         xgemm.NWG == other.xgemm.NWG && xgemm.KWG == other.xgemm.KWG && xgemm.KWI == other.xgemm.KWI && xgemm.MDIMA == other.xgemm.MDIMA &&
         xgemm.NDIMB == other.xgemm.NDIMB && xgemm.VWM == other.xgemm.VWM && xgemm.VWN == other.xgemm.VWN &&
         xgemm16.MDIMC == other.xgemm16.MDIMC && xgemm16.NDIMC == other.xgemm16.NDIMC &&
         xgemm16.MWG == other.xgemm16.MWG && xgemm16.NWG == other.xgemm16.NWG &&
         xgemm16.KWG == other.xgemm16.KWG && xgemm16.KWI == other.xgemm16.KWI && xgemm16.MDIMA == other.xgemm16.MDIMA &&
         xgemm16.NDIMB == other.xgemm16.NDIMB && xgemm16.VWM == other.xgemm16.VWM && xgemm16.VWN == other.xgemm16.VWN &&
         xgemmDirect.WGD == other.xgemmDirect.WGD &&
         xgemmDirect.MDIMCD == other.xgemmDirect.MDIMCD && xgemmDirect.NDIMCD == other.xgemmDirect.NDIMCD &&
         xgemmDirect.MDIMAD == other.xgemmDirect.MDIMAD && xgemmDirect.NDIMBD == other.xgemmDirect.NDIMBD &&
         xgemmDirect.KWID == other.xgemmDirect.KWID && xgemmDirect.PADA == other.xgemmDirect.PADA &&
         xgemmDirect.PADB == other.xgemmDirect.PADB && xgemmDirect.VWMD == other.xgemmDirect.VWMD &&
         xgemmDirect.VWND == other.xgemmDirect.VWND &&
         transformerXgemm.WGD == other.transformerXgemm.WGD &&
         transformerXgemm.MDIMCD == other.transformerXgemm.MDIMCD &&
         transformerXgemm.NDIMCD == other.transformerXgemm.NDIMCD &&
         transformerXgemm.MDIMAD == other.transformerXgemm.MDIMAD &&
         transformerXgemm.NDIMBD == other.transformerXgemm.NDIMBD &&
         transformerXgemm.KWID == other.transformerXgemm.KWID &&
         transformerXgemm.PADA == other.transformerXgemm.PADA &&
         transformerXgemm.PADB == other.transformerXgemm.PADB &&
         transformerXgemm.VWMD == other.transformerXgemm.VWMD &&
         transformerXgemm.VWND == other.transformerXgemm.VWND &&
         transformer.ATTN_BLOCK_Q == other.transformer.ATTN_BLOCK_Q &&
         transformer.ATTN_BLOCK_KV == other.transformer.ATTN_BLOCK_KV &&
         transformer.Q_PER_THREAD == other.transformer.Q_PER_THREAD &&
         transformer.USE_TILED_ATTN == other.transformer.USE_TILED_ATTN &&
         transformerCooperativeMatrixNCHW.ATTN_BLOCK_Q == other.transformerCooperativeMatrixNCHW.ATTN_BLOCK_Q &&
         transformerCooperativeMatrixNCHW.ATTN_BLOCK_KV == other.transformerCooperativeMatrixNCHW.ATTN_BLOCK_KV &&
         transformerCooperativeMatrixNCHW.COOP_ACC_TYPE == other.transformerCooperativeMatrixNCHW.COOP_ACC_TYPE &&
         transformerCooperativeMatrixNCHW.COOP_M_SIZE == other.transformerCooperativeMatrixNCHW.COOP_M_SIZE &&
         transformerCooperativeMatrixNCHW.COOP_N_SIZE == other.transformerCooperativeMatrixNCHW.COOP_N_SIZE &&
         transformerCooperativeMatrixNCHW.COOP_K_SIZE == other.transformerCooperativeMatrixNCHW.COOP_K_SIZE &&
         transformerCooperativeMatrixNCHW.COOP_SUBGROUP_SIZE == other.transformerCooperativeMatrixNCHW.COOP_SUBGROUP_SIZE &&
         transformerCooperativeMatrixNCHW.COOP_Q_TILES_PER_WORKGROUP == other.transformerCooperativeMatrixNCHW.COOP_Q_TILES_PER_WORKGROUP &&
         transformerCooperativeMatrixNCHW.COOP_PV_N_SIZE == other.transformerCooperativeMatrixNCHW.COOP_PV_N_SIZE &&
         transformerCooperativeMatrixNCHW.COOP_USE_NV2 == other.transformerCooperativeMatrixNCHW.COOP_USE_NV2 &&
         transformerCooperativeMatrixNHWC.ATTN_BLOCK_Q == other.transformerCooperativeMatrixNHWC.ATTN_BLOCK_Q &&
         transformerCooperativeMatrixNHWC.ATTN_BLOCK_KV == other.transformerCooperativeMatrixNHWC.ATTN_BLOCK_KV &&
         transformerCooperativeMatrixNHWC.COOP_ACC_TYPE == other.transformerCooperativeMatrixNHWC.COOP_ACC_TYPE &&
         transformerCooperativeMatrixNHWC.COOP_M_SIZE == other.transformerCooperativeMatrixNHWC.COOP_M_SIZE &&
         transformerCooperativeMatrixNHWC.COOP_N_SIZE == other.transformerCooperativeMatrixNHWC.COOP_N_SIZE &&
         transformerCooperativeMatrixNHWC.COOP_K_SIZE == other.transformerCooperativeMatrixNHWC.COOP_K_SIZE &&
         transformerCooperativeMatrixNHWC.COOP_SUBGROUP_SIZE == other.transformerCooperativeMatrixNHWC.COOP_SUBGROUP_SIZE &&
         transformerCooperativeMatrixNHWC.COOP_Q_TILES_PER_WORKGROUP == other.transformerCooperativeMatrixNHWC.COOP_Q_TILES_PER_WORKGROUP &&
         transformerCooperativeMatrixNHWC.COOP_PV_N_SIZE == other.transformerCooperativeMatrixNHWC.COOP_PV_N_SIZE &&
         transformerCooperativeMatrixNHWC.COOP_USE_NV2 == other.transformerCooperativeMatrixNHWC.COOP_USE_NV2 &&
         transformerPreNorm.WG_C_SIZE == other.transformerPreNorm.WG_C_SIZE &&
         transformerPreNorm.WG_XY_SIZE == other.transformerPreNorm.WG_XY_SIZE &&
         transformerPreNorm.C_PER_THREAD == other.transformerPreNorm.C_PER_THREAD &&
         transformerPreNorm.MODE == other.transformerPreNorm.MODE &&
         rmsNorm.WG_C_SIZE == other.rmsNorm.WG_C_SIZE &&
         rmsNorm.WG_XY_SIZE == other.rmsNorm.WG_XY_SIZE &&
         rmsNorm.C_PER_THREAD == other.rmsNorm.C_PER_THREAD &&
         spatialRMSNorm.TILE_SIZE == other.spatialRMSNorm.TILE_SIZE &&
         spatialRMSNorm.APPLY_ELTS_PER_THREAD == other.spatialRMSNorm.APPLY_ELTS_PER_THREAD;
}

bool VulkanTuneParams::isValid() const {
  if(transformerCooperativeMatrixNCHW.COOP_USE_NV2 != 0)
    return false;
  if(vulkan.shouldUseFP16Storage &&
     (!vulkan.canUseFP16Storage || !vulkan.canUseFP16Compute))
    return false;
  if(vulkan.shouldUseFP16Compute && !vulkan.canUseFP16Compute)
    return false;
  if(vulkan.shouldUseCooperativeMatrix && !vulkan.canUseCooperativeMatrix)
    return false;
  if(vulkan.shouldUseNHWC &&
     (!vulkan.shouldUseCooperativeMatrix ||
      !vulkan.shouldUseFP16Storage || !vulkan.shouldUseFP16Compute))
    return false;
  if(vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
     (!vulkan.canUseCooperativeMatrix ||
      !vulkan.shouldUseFP16Storage || !vulkan.shouldUseFP16Compute))
    return false;
  if(vulkan.shouldUseTransformerCooperativeMatrixNCHW &&
     (!vulkan.canUseCooperativeMatrix || !vulkan.canUseFP16Compute || !vulkan.shouldUseFP16Storage))
    return false;
  if(vulkan.shouldUseTransformerCooperativeMatrixNHWC &&
     (!vulkan.canUseCooperativeMatrix || !vulkan.canUseFP16Compute ||
      !vulkan.shouldUseFP16Storage || !vulkan.shouldUseFP16Compute))
    return false;
  if(vulkan.shouldUseTransformerDualGemmSwiGLU &&
     (!vulkan.canUseCooperativeMatrix ||
      !vulkan.shouldUseFP16Storage || !vulkan.shouldUseFP16Compute ||
      !transformerDualGemmSwiGLU.isValid()))
    return false;
  return VulkanTuningProfile::isValid();
}

bool VulkanTuneParams::operator==(const VulkanTuneParams& other) const {
  return vulkan.canUseFP16Storage == other.vulkan.canUseFP16Storage &&
         vulkan.canUseFP16Compute == other.vulkan.canUseFP16Compute &&
         vulkan.canUseCooperativeMatrix == other.vulkan.canUseCooperativeMatrix &&
         vulkan.canUseSubgroup == other.vulkan.canUseSubgroup &&
         vulkan.shouldUseFP16Storage == other.vulkan.shouldUseFP16Storage &&
         vulkan.shouldUseFP16Compute == other.vulkan.shouldUseFP16Compute &&
         vulkan.shouldUseCooperativeMatrix == other.vulkan.shouldUseCooperativeMatrix &&
         vulkan.shouldUseNHWC == other.vulkan.shouldUseNHWC &&
         vulkan.shouldUseHgemmCooperativeMatrixNCHW == other.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
         vulkan.shouldUseTransformerDualGemmSwiGLU == other.vulkan.shouldUseTransformerDualGemmSwiGLU &&
         vulkan.shouldUseTransformerCooperativeMatrixNCHW == other.vulkan.shouldUseTransformerCooperativeMatrixNCHW &&
         vulkan.shouldUseTransformerCooperativeMatrixNHWC == other.vulkan.shouldUseTransformerCooperativeMatrixNHWC &&
         vulkan.shouldUseSubgroup == other.vulkan.shouldUseSubgroup &&
         static_cast<const VulkanTuningProfile&>(*this) == static_cast<const VulkanTuningProfile&>(other);
}

namespace {
  void writeTuningProfile(
    ofstream& out,
    const string& prefix,
    const VulkanTuningProfile& profile
  ) {
#define WRITE(name, value) writeParam(out, prefix.empty() ? name : prefix + "." name, value)
    WRITE("xgemmDirect.WGD", profile.xgemmDirect.WGD);
    WRITE("xgemmDirect.MDIMCD", profile.xgemmDirect.MDIMCD);
    WRITE("xgemmDirect.NDIMCD", profile.xgemmDirect.NDIMCD);
    WRITE("xgemmDirect.MDIMAD", profile.xgemmDirect.MDIMAD);
    WRITE("xgemmDirect.NDIMBD", profile.xgemmDirect.NDIMBD);
    WRITE("xgemmDirect.KWID", profile.xgemmDirect.KWID);
    WRITE("xgemmDirect.PADA", profile.xgemmDirect.PADA);
    WRITE("xgemmDirect.PADB", profile.xgemmDirect.PADB);
    WRITE("xgemmDirect.VWMD", profile.xgemmDirect.VWMD);
    WRITE("xgemmDirect.VWND", profile.xgemmDirect.VWND);
    WRITE("transformerXgemm.WGD", profile.transformerXgemm.WGD);
    WRITE("transformerXgemm.MDIMCD", profile.transformerXgemm.MDIMCD);
    WRITE("transformerXgemm.NDIMCD", profile.transformerXgemm.NDIMCD);
    WRITE("transformerXgemm.MDIMAD", profile.transformerXgemm.MDIMAD);
    WRITE("transformerXgemm.NDIMBD", profile.transformerXgemm.NDIMBD);
    WRITE("transformerXgemm.KWID", profile.transformerXgemm.KWID);
    WRITE("transformerXgemm.PADA", profile.transformerXgemm.PADA);
    WRITE("transformerXgemm.PADB", profile.transformerXgemm.PADB);
    WRITE("transformerXgemm.VWMD", profile.transformerXgemm.VWMD);
    WRITE("transformerXgemm.VWND", profile.transformerXgemm.VWND);
#define WRITE_XGEMM(name, params) \
    WRITE(name ".MWG", params.MWG); \
    WRITE(name ".NWG", params.NWG); \
    WRITE(name ".KWG", params.KWG); \
    WRITE(name ".KWI", params.KWI); \
    WRITE(name ".MDIMC", params.MDIMC); \
    WRITE(name ".NDIMC", params.NDIMC); \
    WRITE(name ".MDIMA", params.MDIMA); \
    WRITE(name ".NDIMB", params.NDIMB); \
    WRITE(name ".VWM", params.VWM); \
    WRITE(name ".VWN", params.VWN)
    WRITE_XGEMM("xgemm", profile.xgemm);
    WRITE_XGEMM("xgemm16", profile.xgemm16);
#undef WRITE_XGEMM
#define WRITE_HGEMM(name, params) \
    WRITE(name ".MWG", params.MWG); \
    WRITE(name ".NWG", params.NWG); \
    WRITE(name ".KWG", params.KWG); \
    WRITE(name ".MWAVE", params.MWAVE); \
    WRITE(name ".NWAVE", params.NWAVE); \
    WRITE(name ".MWARP", params.MWARP); \
    WRITE(name ".NWARP", params.NWARP); \
    WRITE(name ".VWM", params.VWM); \
    WRITE(name ".VWN", params.VWN); \
    WRITE(name ".KDIM", params.KDIM); \
    WRITE(name ".subgroupSize", params.subgroupSize); \
    WRITE(name ".accType", params.accType)
#define WRITE_HGEMM_NHWC(name, params) \
    WRITE(name ".MWG", params.MWG); \
    WRITE(name ".NWG", params.NWG); \
    WRITE(name ".KWG", params.KWG); \
    WRITE(name ".MWAVE", params.MWAVE); \
    WRITE(name ".NWAVE", params.NWAVE); \
    WRITE(name ".MWARP", params.MWARP); \
    WRITE(name ".NWARP", params.NWARP); \
    WRITE(name ".KDIM", params.KDIM); \
    WRITE(name ".subgroupSize", params.subgroupSize); \
    WRITE(name ".accType", params.accType)
    WRITE_HGEMM("hgemmCooperativeMatrix", profile.hgemmCooperativeMatrix);
    WRITE("hgemmCooperativeMatrix.SA", profile.hgemmCooperativeMatrix.SA);
    WRITE("hgemmCooperativeMatrix.SB", profile.hgemmCooperativeMatrix.SB);
    WRITE_HGEMM_NHWC("hgemmCooperativeMatrixNHWC", profile.hgemmCooperativeMatrixNHWC);
    WRITE("hgemmCooperativeMatrixNHWC.SA", profile.hgemmCooperativeMatrixNHWC.SA);
    WRITE("hgemmCooperativeMatrixNHWC.SB", profile.hgemmCooperativeMatrixNHWC.SB);
    WRITE("hgemmCooperativeMatrixNHWC.DB", profile.hgemmCooperativeMatrixNHWC.DB);
    WRITE("hgemmCooperativeMatrixNHWC.VWK", profile.hgemmCooperativeMatrixNHWC.VWK);
    WRITE("hgemmCooperativeMatrixNHWC.VWN", profile.hgemmCooperativeMatrixNHWC.VWN);
    WRITE_HGEMM_NHWC("transformerHgemmNHWC", profile.transformerHgemmNHWC);
    WRITE("transformerHgemmNHWC.SA", profile.transformerHgemmNHWC.SA);
    WRITE("transformerHgemmNHWC.SB", profile.transformerHgemmNHWC.SB);
    WRITE("transformerHgemmNHWC.DB", profile.transformerHgemmNHWC.DB);
    WRITE("transformerHgemmNHWC.VWK", profile.transformerHgemmNHWC.VWK);
    WRITE("transformerHgemmNHWC.VWN", profile.transformerHgemmNHWC.VWN);
    WRITE_HGEMM("hgemmCooperativeMatrixNCHW", profile.hgemmCooperativeMatrixNCHW);
    WRITE("hgemmCooperativeMatrixNCHW.SB", profile.hgemmCooperativeMatrixNCHW.SB);
#undef WRITE_HGEMM
#undef WRITE_HGEMM_NHWC
#define WRITE_CONV(name, params) \
    WRITE(name ".inTileXSize", params.inTileXSize); \
    WRITE(name ".inTileYSize", params.inTileYSize); \
    WRITE(name ".outTileXSize", params.outTileXSize); \
    WRITE(name ".outTileYSize", params.outTileYSize); \
    WRITE(name ".inputTransformLocalXSize", params.inputTransformLocalXSize); \
    WRITE(name ".inputTransformLocalYSize", params.inputTransformLocalYSize); \
    WRITE(name ".outputTransformLocalXSize", params.outputTransformLocalXSize); \
    WRITE(name ".outputTransformLocalYSize", params.outputTransformLocalYSize); \
    WRITE(name ".outputTransformLocalZSize", params.outputTransformLocalZSize); \
    WRITE(name ".inputTransformNHWCLocalXSize", params.inputTransformNHWCLocalXSize); \
    WRITE(name ".inputTransformNHWCLocalYSize", params.inputTransformNHWCLocalYSize); \
    WRITE(name ".outputTransformNHWCLocalXSize", params.outputTransformNHWCLocalXSize); \
    WRITE(name ".outputTransformNHWCLocalYSize", params.outputTransformNHWCLocalYSize)
    WRITE_CONV("conv3x3", profile.conv3x3);
    WRITE_CONV("conv5x5", profile.conv5x5);
#undef WRITE_CONV
    WRITE("gPool.XYSTRIDE", profile.gPool.XYSTRIDE);
    WRITE("gPool.CHANNELSTRIDE", profile.gPool.CHANNELSTRIDE);
    WRITE("gPool.BATCHSTRIDE", profile.gPool.BATCHSTRIDE);
    WRITE("transformer.ATTN_BLOCK_Q", profile.transformer.ATTN_BLOCK_Q);
    WRITE("transformer.ATTN_BLOCK_KV", profile.transformer.ATTN_BLOCK_KV);
    WRITE("transformer.Q_PER_THREAD", profile.transformer.Q_PER_THREAD);
    WRITE("transformer.USE_TILED_ATTN", profile.transformer.USE_TILED_ATTN);
#define WRITE_TRANSFORMER_COOPERATIVE(name, params) \
    WRITE(name ".ATTN_BLOCK_Q", params.ATTN_BLOCK_Q); \
    WRITE(name ".ATTN_BLOCK_KV", params.ATTN_BLOCK_KV); \
    WRITE(name ".COOP_ACC_TYPE", params.COOP_ACC_TYPE); \
    WRITE(name ".COOP_M_SIZE", params.COOP_M_SIZE); \
    WRITE(name ".COOP_N_SIZE", params.COOP_N_SIZE); \
    WRITE(name ".COOP_K_SIZE", params.COOP_K_SIZE); \
    WRITE(name ".COOP_SUBGROUP_SIZE", params.COOP_SUBGROUP_SIZE); \
    WRITE(name ".COOP_Q_TILES_PER_WORKGROUP", params.COOP_Q_TILES_PER_WORKGROUP); \
    WRITE(name ".COOP_PV_N_SIZE", params.COOP_PV_N_SIZE); \
    WRITE(name ".COOP_USE_NV2", params.COOP_USE_NV2)
    WRITE_TRANSFORMER_COOPERATIVE(
      "transformerCooperativeMatrixNCHW",
      profile.transformerCooperativeMatrixNCHW
    );
    WRITE_TRANSFORMER_COOPERATIVE(
      "transformerCooperativeMatrixNHWC",
      profile.transformerCooperativeMatrixNHWC
    );
#undef WRITE_TRANSFORMER_COOPERATIVE
    WRITE("transformerPreNorm.WG_C_SIZE", profile.transformerPreNorm.WG_C_SIZE);
    WRITE("transformerPreNorm.WG_XY_SIZE", profile.transformerPreNorm.WG_XY_SIZE);
    WRITE("transformerPreNorm.C_PER_THREAD", profile.transformerPreNorm.C_PER_THREAD);
    WRITE("transformerPreNorm.MODE", profile.transformerPreNorm.MODE);
    WRITE("rmsNorm.WG_C_SIZE", profile.rmsNorm.WG_C_SIZE);
    WRITE("rmsNorm.WG_XY_SIZE", profile.rmsNorm.WG_XY_SIZE);
    WRITE("rmsNorm.C_PER_THREAD", profile.rmsNorm.C_PER_THREAD);
    WRITE("pointwise.ELTS_PER_THREAD", profile.pointwise.ELTS_PER_THREAD);
    WRITE("pointwise.LOCAL_SIZE", profile.pointwise.LOCAL_SIZE);
    WRITE("addChannelBiases.XY_ELTS_PER_THREAD", profile.addChannelBiases.XY_ELTS_PER_THREAD);
    WRITE("addChannelBiases.NC_ELTS_PER_THREAD", profile.addChannelBiases.NC_ELTS_PER_THREAD);
    WRITE("spatialRMSNorm.TILE_SIZE", profile.spatialRMSNorm.TILE_SIZE);
    WRITE("spatialRMSNorm.APPLY_ELTS_PER_THREAD", profile.spatialRMSNorm.APPLY_ELTS_PER_THREAD);
    writeParam(out, "transformerDualGemmSwiGLU.MWG", profile.transformerDualGemmSwiGLU.MWG);
    writeParam(out, "transformerDualGemmSwiGLU.NWG", profile.transformerDualGemmSwiGLU.NWG);
    writeParam(out, "transformerDualGemmSwiGLU.KWG", profile.transformerDualGemmSwiGLU.KWG);
    writeParam(out, "transformerDualGemmSwiGLU.MWAVE", profile.transformerDualGemmSwiGLU.MWAVE);
    writeParam(out, "transformerDualGemmSwiGLU.NWAVE", profile.transformerDualGemmSwiGLU.NWAVE);
    writeParam(out, "transformerDualGemmSwiGLU.MWARP", profile.transformerDualGemmSwiGLU.MWARP);
    writeParam(out, "transformerDualGemmSwiGLU.NWARP", profile.transformerDualGemmSwiGLU.NWARP);
    writeParam(out, "transformerDualGemmSwiGLU.VWM", profile.transformerDualGemmSwiGLU.VWM);
    writeParam(out, "transformerDualGemmSwiGLU.VWN", profile.transformerDualGemmSwiGLU.VWN);
    writeParam(out, "transformerDualGemmSwiGLU.KDIM", profile.transformerDualGemmSwiGLU.KDIM);
    writeParam(out, "transformerDualGemmSwiGLU.subgroupSize", profile.transformerDualGemmSwiGLU.subgroupSize);
    writeParam(out, "transformerDualGemmSwiGLU.accType", profile.transformerDualGemmSwiGLU.accType);
    writeParam(out, "transformerDualGemmSwiGLU.SB", profile.transformerDualGemmSwiGLU.SB);
    writeParam(out, "transformerDualGemmSwiGLU.STAGES", profile.transformerDualGemmSwiGLU.STAGES);
#undef WRITE
  }

  bool isValidTuningRecordName(const string& name) {
    return !name.empty() && all_of(name.begin(), name.end(), [](char c) {
      return Global::isAlpha(c) || Global::isDigit(c) || c == '_';
    });
  }
}

void VulkanTuneParams::save(const string& filename, const VulkanTuneParams& config) {
  if(!config.isValid())
    throw StringError("VulkanTuneParams::save: refusing to save invalid parameters to " + filename);
  for(const auto& entry: config.tuningRecords) {
    if(!isValidTuningRecordName(entry.first))
      throw StringError("VulkanTuneParams::save: invalid tuning record " + entry.first + " in " + filename);
  }
  Rand rand;
  const string temporaryFilename = filename + "." + Global::uint64ToHexString(rand.nextUInt64()) + ".tmp";
  ofstream out;
  auto temporaryFileGuard = makeScopeGuard([&]() {
    out.close();
    FileUtils::tryRemoveFile(temporaryFilename);
  });
  FileUtils::open(out, temporaryFilename);
  out << VERSION_LINE << "\n";
  writeParam(out, "vulkan.canUseFP16Storage", config.vulkan.canUseFP16Storage);
  writeParam(out, "vulkan.canUseFP16Compute", config.vulkan.canUseFP16Compute);
  writeParam(out, "vulkan.canUseCooperativeMatrix", config.vulkan.canUseCooperativeMatrix);
  writeParam(out, "vulkan.canUseSubgroup", config.vulkan.canUseSubgroup);
  writeParam(out, "vulkan.shouldUseFP16Storage", config.vulkan.shouldUseFP16Storage);
  writeParam(out, "vulkan.shouldUseFP16Compute", config.vulkan.shouldUseFP16Compute);
  writeParam(out, "vulkan.shouldUseCooperativeMatrix", config.vulkan.shouldUseCooperativeMatrix);
  writeParam(out, "vulkan.shouldUseNHWC", config.vulkan.shouldUseNHWC);
  writeParam(
    out,
    "vulkan.shouldUseHgemmCooperativeMatrixNCHW",
    config.vulkan.shouldUseHgemmCooperativeMatrixNCHW
  );
  writeParam(out, "vulkan.shouldUseSubgroup", config.vulkan.shouldUseSubgroup);
  writeParam(
    out,
    "vulkan.shouldUseTransformerDualGemmSwiGLU",
    config.vulkan.shouldUseTransformerDualGemmSwiGLU
  );
  writeParam(
    out,
    "vulkan.shouldUseTransformerCooperativeMatrixNCHW",
    config.vulkan.shouldUseTransformerCooperativeMatrixNCHW
  );
  writeParam(
    out,
    "vulkan.shouldUseTransformerCooperativeMatrixNHWC",
    config.vulkan.shouldUseTransformerCooperativeMatrixNHWC
  );
  writeTuningProfile(out, "", static_cast<const VulkanTuningProfile&>(config));
  for(const auto& entry: config.tuningRecords) {
    const string prefix = "tuning." + entry.first;
    writeParam(out, prefix + ".completed", entry.second.completed);
  }
  out.flush();
  out.close();
  if(!out)
    throw IOError("VulkanTuneParams::save: could not write parameters to " + filename);
  FileUtils::rename(temporaryFilename, filename);
  temporaryFileGuard.dismiss();
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
      if(line != VERSION_LINE && line != "VERSION=37" && line != "VERSION=36")
        throw IOError(
          "VulkanTuneParams::load: expected first line to be " + string(VERSION_LINE) + " in " + filename
        );
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
  map<string, string> metadata;
  map<string, VulkanTuneParams::TuningRecord> tuningRecords;
  for(auto iter = values.begin(); iter != values.end();) {
    if(iter->first.compare(0, 7, "tuning.") != 0) {
      ++iter;
      continue;
    }
    const string& key = iter->first;
    const size_t fieldSeparator = key.rfind('.');
    if(fieldSeparator < 7)
      throw IOError("VulkanTuneParams::load: invalid tuning record field " + key + " in " + filename);
    const string name = key.substr(7, fieldSeparator - 7);
    const string field = key.substr(fieldSeparator + 1);
    if(!isValidTuningRecordName(name) || (field != "completed" && field != "conditions"))
      throw IOError("VulkanTuneParams::load: invalid tuning record field " + key + " in " + filename);
    // Ignore obsolete condition hashes without discarding completion flags.
    if(field == "completed") {
      tuningRecords.emplace(name, VulkanTuneParams::TuningRecord());
      metadata.insert(*iter);
    }
    iter = values.erase(iter);
  }
  for(auto& entry: tuningRecords) {
    const string prefix = "tuning." + entry.first;
    entry.second.completed = getBoolParam(metadata, prefix + ".completed", filename);
  }
  const bool nv2Parameters = values.count("transformerCooperativeMatrixNCHW.COOP_USE_NV2") != 0 &&
    values.count("transformerCooperativeMatrixNHWC.COOP_USE_NV2") != 0;
  const bool stageParameter = values.count("transformerDualGemmSwiGLU.STAGES") != 0;
  const bool preNormParameters = values.count("transformerPreNorm.MODE") != 0;
  const bool transformerXgemmParameters = values.count("transformerXgemm.WGD") != 0;
  if(values.size() != (nv2Parameters ? 175 : 173) + (stageParameter ? 1 : 0) + (preNormParameters ? 4 : 0) +
     (transformerXgemmParameters ? 10 : 0))
    throw IOError("VulkanTuneParams::load: unexpected number of parameters in " + filename);

  const auto readProfile = [&](const string& prefix, VulkanTuningProfile& profile) {
    const auto read = [&](const string& name) {
      return getParam(values, prefix.empty() ? name : prefix + "." + name, filename);
    };
    profile.xgemmDirect.WGD = read("xgemmDirect.WGD");
    profile.xgemmDirect.MDIMCD = read("xgemmDirect.MDIMCD");
    profile.xgemmDirect.NDIMCD = read("xgemmDirect.NDIMCD");
    profile.xgemmDirect.MDIMAD = read("xgemmDirect.MDIMAD");
    profile.xgemmDirect.NDIMBD = read("xgemmDirect.NDIMBD");
    profile.xgemmDirect.KWID = read("xgemmDirect.KWID");
    profile.xgemmDirect.PADA = read("xgemmDirect.PADA");
    profile.xgemmDirect.PADB = read("xgemmDirect.PADB");
    profile.xgemmDirect.VWMD = read("xgemmDirect.VWMD");
    profile.xgemmDirect.VWND = read("xgemmDirect.VWND");
    profile.transformerXgemm = profile.xgemmDirect;
    if(transformerXgemmParameters) {
      profile.transformerXgemm.WGD = read("transformerXgemm.WGD");
      profile.transformerXgemm.MDIMCD = read("transformerXgemm.MDIMCD");
      profile.transformerXgemm.NDIMCD = read("transformerXgemm.NDIMCD");
      profile.transformerXgemm.MDIMAD = read("transformerXgemm.MDIMAD");
      profile.transformerXgemm.NDIMBD = read("transformerXgemm.NDIMBD");
      profile.transformerXgemm.KWID = read("transformerXgemm.KWID");
      profile.transformerXgemm.PADA = read("transformerXgemm.PADA");
      profile.transformerXgemm.PADB = read("transformerXgemm.PADB");
      profile.transformerXgemm.VWMD = read("transformerXgemm.VWMD");
      profile.transformerXgemm.VWND = read("transformerXgemm.VWND");
    }
#define READ_XGEMM(name, params) \
    params.MWG = read(name ".MWG"); \
    params.NWG = read(name ".NWG"); \
    params.KWG = read(name ".KWG"); \
    params.KWI = read(name ".KWI"); \
    params.MDIMC = read(name ".MDIMC"); \
    params.NDIMC = read(name ".NDIMC"); \
    params.MDIMA = read(name ".MDIMA"); \
    params.NDIMB = read(name ".NDIMB"); \
    params.VWM = read(name ".VWM"); \
    params.VWN = read(name ".VWN")
    READ_XGEMM("xgemm", profile.xgemm);
    READ_XGEMM("xgemm16", profile.xgemm16);
#undef READ_XGEMM
#define READ_HGEMM(name, params) \
    params.MWG = read(name ".MWG"); \
    params.NWG = read(name ".NWG"); \
    params.KWG = read(name ".KWG"); \
    params.MWAVE = read(name ".MWAVE"); \
    params.NWAVE = read(name ".NWAVE"); \
    params.MWARP = read(name ".MWARP"); \
    params.NWARP = read(name ".NWARP"); \
    params.VWM = read(name ".VWM"); \
    params.VWN = read(name ".VWN"); \
    params.KDIM = read(name ".KDIM"); \
    params.subgroupSize = read(name ".subgroupSize"); \
    params.accType = read(name ".accType")
#define READ_HGEMM_NHWC(name, params) \
    params.MWG = read(name ".MWG"); \
    params.NWG = read(name ".NWG"); \
    params.KWG = read(name ".KWG"); \
    params.MWAVE = read(name ".MWAVE"); \
    params.NWAVE = read(name ".NWAVE"); \
    params.MWARP = read(name ".MWARP"); \
    params.NWARP = read(name ".NWARP"); \
    params.KDIM = read(name ".KDIM"); \
    params.subgroupSize = read(name ".subgroupSize"); \
    params.accType = read(name ".accType")
    READ_HGEMM("hgemmCooperativeMatrix", profile.hgemmCooperativeMatrix);
    profile.hgemmCooperativeMatrix.SA = read("hgemmCooperativeMatrix.SA");
    profile.hgemmCooperativeMatrix.SB = read("hgemmCooperativeMatrix.SB");
    READ_HGEMM_NHWC("hgemmCooperativeMatrixNHWC", profile.hgemmCooperativeMatrixNHWC);
    profile.hgemmCooperativeMatrixNHWC.SA = read("hgemmCooperativeMatrixNHWC.SA");
    profile.hgemmCooperativeMatrixNHWC.SB = read("hgemmCooperativeMatrixNHWC.SB");
    profile.hgemmCooperativeMatrixNHWC.DB = read("hgemmCooperativeMatrixNHWC.DB");
    profile.hgemmCooperativeMatrixNHWC.VWK = read("hgemmCooperativeMatrixNHWC.VWK");
    profile.hgemmCooperativeMatrixNHWC.VWN = read("hgemmCooperativeMatrixNHWC.VWN");
    READ_HGEMM_NHWC("transformerHgemmNHWC", profile.transformerHgemmNHWC);
    profile.transformerHgemmNHWC.SA = read("transformerHgemmNHWC.SA");
    profile.transformerHgemmNHWC.SB = read("transformerHgemmNHWC.SB");
    profile.transformerHgemmNHWC.DB = read("transformerHgemmNHWC.DB");
    profile.transformerHgemmNHWC.VWK = read("transformerHgemmNHWC.VWK");
    profile.transformerHgemmNHWC.VWN = read("transformerHgemmNHWC.VWN");
    READ_HGEMM("hgemmCooperativeMatrixNCHW", profile.hgemmCooperativeMatrixNCHW);
    profile.hgemmCooperativeMatrixNCHW.SB = read("hgemmCooperativeMatrixNCHW.SB");
#undef READ_HGEMM
#undef READ_HGEMM_NHWC
#define READ_CONV(name, params) \
    params.inTileXSize = read(name ".inTileXSize"); \
    params.inTileYSize = read(name ".inTileYSize"); \
    params.outTileXSize = read(name ".outTileXSize"); \
    params.outTileYSize = read(name ".outTileYSize"); \
    params.inputTransformLocalXSize = read(name ".inputTransformLocalXSize"); \
    params.inputTransformLocalYSize = read(name ".inputTransformLocalYSize"); \
    params.outputTransformLocalXSize = read(name ".outputTransformLocalXSize"); \
    params.outputTransformLocalYSize = read(name ".outputTransformLocalYSize"); \
    params.outputTransformLocalZSize = read(name ".outputTransformLocalZSize"); \
    params.inputTransformNHWCLocalXSize = read(name ".inputTransformNHWCLocalXSize"); \
    params.inputTransformNHWCLocalYSize = read(name ".inputTransformNHWCLocalYSize"); \
    params.outputTransformNHWCLocalXSize = read(name ".outputTransformNHWCLocalXSize"); \
    params.outputTransformNHWCLocalYSize = read(name ".outputTransformNHWCLocalYSize")
    READ_CONV("conv3x3", profile.conv3x3);
    READ_CONV("conv5x5", profile.conv5x5);
#undef READ_CONV
    profile.gPool.XYSTRIDE = read("gPool.XYSTRIDE");
    profile.gPool.CHANNELSTRIDE = read("gPool.CHANNELSTRIDE");
    profile.gPool.BATCHSTRIDE = read("gPool.BATCHSTRIDE");
    profile.transformer.ATTN_BLOCK_Q = read("transformer.ATTN_BLOCK_Q");
    profile.transformer.ATTN_BLOCK_KV = read("transformer.ATTN_BLOCK_KV");
    profile.transformer.Q_PER_THREAD = read("transformer.Q_PER_THREAD");
    profile.transformer.USE_TILED_ATTN = read("transformer.USE_TILED_ATTN");
#define READ_TRANSFORMER_COOPERATIVE(name, params) \
    params.ATTN_BLOCK_Q = read(name ".ATTN_BLOCK_Q"); \
    params.ATTN_BLOCK_KV = read(name ".ATTN_BLOCK_KV"); \
    params.COOP_ACC_TYPE = read(name ".COOP_ACC_TYPE"); \
    params.COOP_M_SIZE = read(name ".COOP_M_SIZE"); \
    params.COOP_N_SIZE = read(name ".COOP_N_SIZE"); \
    params.COOP_K_SIZE = read(name ".COOP_K_SIZE"); \
    params.COOP_SUBGROUP_SIZE = read(name ".COOP_SUBGROUP_SIZE"); \
    params.COOP_Q_TILES_PER_WORKGROUP = read(name ".COOP_Q_TILES_PER_WORKGROUP"); \
    params.COOP_PV_N_SIZE = read(name ".COOP_PV_N_SIZE"); \
    params.COOP_USE_NV2 = nv2Parameters ? read(name ".COOP_USE_NV2") : 0
    READ_TRANSFORMER_COOPERATIVE(
      "transformerCooperativeMatrixNCHW",
      profile.transformerCooperativeMatrixNCHW
    );
    READ_TRANSFORMER_COOPERATIVE(
      "transformerCooperativeMatrixNHWC",
      profile.transformerCooperativeMatrixNHWC
    );
#undef READ_TRANSFORMER_COOPERATIVE
    profile.rmsNorm.WG_C_SIZE = read("rmsNorm.WG_C_SIZE");
    profile.rmsNorm.WG_XY_SIZE = read("rmsNorm.WG_XY_SIZE");
    profile.rmsNorm.C_PER_THREAD = read("rmsNorm.C_PER_THREAD");
    profile.pointwise.ELTS_PER_THREAD = read("pointwise.ELTS_PER_THREAD");
    profile.pointwise.LOCAL_SIZE = read("pointwise.LOCAL_SIZE");
    profile.addChannelBiases.XY_ELTS_PER_THREAD = read("addChannelBiases.XY_ELTS_PER_THREAD");
    profile.addChannelBiases.NC_ELTS_PER_THREAD = read("addChannelBiases.NC_ELTS_PER_THREAD");
    profile.transformerPreNorm = profile.rmsNorm;
    if(preNormParameters) {
      profile.transformerPreNorm.WG_C_SIZE = read("transformerPreNorm.WG_C_SIZE");
      profile.transformerPreNorm.WG_XY_SIZE = read("transformerPreNorm.WG_XY_SIZE");
      profile.transformerPreNorm.C_PER_THREAD = read("transformerPreNorm.C_PER_THREAD");
      profile.transformerPreNorm.MODE = read("transformerPreNorm.MODE");
    }
    profile.spatialRMSNorm.TILE_SIZE = read("spatialRMSNorm.TILE_SIZE");
    profile.spatialRMSNorm.APPLY_ELTS_PER_THREAD = read("spatialRMSNorm.APPLY_ELTS_PER_THREAD");
    profile.transformerDualGemmSwiGLU.MWG = read("transformerDualGemmSwiGLU.MWG");
    profile.transformerDualGemmSwiGLU.NWG = read("transformerDualGemmSwiGLU.NWG");
    profile.transformerDualGemmSwiGLU.KWG = read("transformerDualGemmSwiGLU.KWG");
    profile.transformerDualGemmSwiGLU.MWAVE = read("transformerDualGemmSwiGLU.MWAVE");
    profile.transformerDualGemmSwiGLU.NWAVE = read("transformerDualGemmSwiGLU.NWAVE");
    profile.transformerDualGemmSwiGLU.MWARP = read("transformerDualGemmSwiGLU.MWARP");
    profile.transformerDualGemmSwiGLU.NWARP = read("transformerDualGemmSwiGLU.NWARP");
    profile.transformerDualGemmSwiGLU.VWM = read("transformerDualGemmSwiGLU.VWM");
    profile.transformerDualGemmSwiGLU.VWN = read("transformerDualGemmSwiGLU.VWN");
    profile.transformerDualGemmSwiGLU.KDIM = read("transformerDualGemmSwiGLU.KDIM");
    profile.transformerDualGemmSwiGLU.subgroupSize = read("transformerDualGemmSwiGLU.subgroupSize");
    profile.transformerDualGemmSwiGLU.accType = read("transformerDualGemmSwiGLU.accType");
    profile.transformerDualGemmSwiGLU.SB = read("transformerDualGemmSwiGLU.SB");
    profile.transformerDualGemmSwiGLU.STAGES = stageParameter ? read("transformerDualGemmSwiGLU.STAGES") : 1;
  };

  VulkanTuneParams config;
  config.tuningRecords = std::move(tuningRecords);
  config.vulkan.canUseFP16Storage = getBoolParam(values, "vulkan.canUseFP16Storage", filename);
  config.vulkan.canUseFP16Compute = getBoolParam(values, "vulkan.canUseFP16Compute", filename);
  config.vulkan.canUseCooperativeMatrix = getBoolParam(values, "vulkan.canUseCooperativeMatrix", filename);
  config.vulkan.canUseSubgroup = getBoolParam(values, "vulkan.canUseSubgroup", filename);
  config.vulkan.shouldUseFP16Storage = getBoolParam(values, "vulkan.shouldUseFP16Storage", filename);
  config.vulkan.shouldUseFP16Compute = getBoolParam(values, "vulkan.shouldUseFP16Compute", filename);
  config.vulkan.shouldUseCooperativeMatrix = getBoolParam(
    values,
    "vulkan.shouldUseCooperativeMatrix",
    filename
  );
  config.vulkan.shouldUseNHWC = getBoolParam(values, "vulkan.shouldUseNHWC", filename);
  config.vulkan.shouldUseHgemmCooperativeMatrixNCHW = getBoolParam(
    values,
    "vulkan.shouldUseHgemmCooperativeMatrixNCHW",
    filename
  );
  config.vulkan.shouldUseSubgroup = getBoolParam(values, "vulkan.shouldUseSubgroup", filename);
  config.vulkan.shouldUseTransformerDualGemmSwiGLU = getBoolParam(
    values,
    "vulkan.shouldUseTransformerDualGemmSwiGLU",
    filename
  );
  config.vulkan.shouldUseTransformerCooperativeMatrixNCHW = getBoolParam(
    values,
    "vulkan.shouldUseTransformerCooperativeMatrixNCHW",
    filename
  );
  config.vulkan.shouldUseTransformerCooperativeMatrixNHWC = getBoolParam(
    values,
    "vulkan.shouldUseTransformerCooperativeMatrixNHWC",
    filename
  );
  readProfile("", static_cast<VulkanTuningProfile&>(config));
  const bool useCooperativeMatrix = config.vulkan.shouldUseCooperativeMatrix &&
    config.vulkan.canUseCooperativeMatrix && config.vulkan.canUseFP16Storage &&
    config.vulkan.canUseFP16Compute && config.vulkan.shouldUseFP16Storage &&
    config.vulkan.shouldUseFP16Compute;
  config.vulkan.shouldUseNHWC = config.vulkan.shouldUseNHWC && useCooperativeMatrix;
  config.vulkan.shouldUseTransformerDualGemmSwiGLU =
    config.vulkan.shouldUseTransformerDualGemmSwiGLU && useCooperativeMatrix;
  if(!config.isValid())
    throw IOError("VulkanTuneParams::load: parameters are invalid in " + filename);
  return config;
}

namespace {
  void addChannelCase(vector<int>& cases, int channels) {
    if(channels > 0 && find(cases.begin(), cases.end(), channels) == cases.end())
      cases.push_back(channels);
  }

  void collectBlockTuningInfo(
    const vector<pair<int, unique_ptr_void>>& blocks,
    VulkanTuner::ModelInfoForTuning& modelInfo
  ) {
    int previousNormChannels = 0;
    for(const auto& block: blocks) {
      int normChannels = 0;
      if(block.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
        const TransformerAttentionDesc* attn = static_cast<const TransformerAttentionDesc*>(block.second.get());
        normChannels = attn->qProj.inChannels;
        addChannelCase(modelInfo.residualChannels, attn->outProj.outChannels);
        addChannelCase(modelInfo.rmsNormChannels, attn->preLN.numChannels);
        modelInfo.transformerHeadDim = attn->qHeadDim;
        modelInfo.transformerVHeadDim = attn->vHeadDim;
        modelInfo.transformerNumHeads = attn->numHeads;
        modelInfo.transformerNumKVHeads = attn->numKVHeads;
        modelInfo.transformerAttentionInChannels = attn->qProj.inChannels;
        const VulkanTuner::ModelInfoForTuning::AttentionShape shape{
          attn->numHeads, attn->numKVHeads, attn->qProj.inChannels
        };
        if(find(modelInfo.attentionShapes.begin(), modelInfo.attentionShapes.end(), shape) == modelInfo.attentionShapes.end())
          modelInfo.attentionShapes.push_back(shape);
        modelInfo.transformerUseRope = attn->useRope;
        modelInfo.transformerLearnableRope = attn->learnableRope;
        modelInfo.transformerRopeTheta = attn->ropeTheta;
        modelInfo.transformerRopeFreqs = attn->ropeFreqs;
        modelInfo.transformerProjections.push_back({attn->qProj.inChannels,
          attn->qProj.outChannels + attn->kProj.outChannels + attn->vProj.outChannels, true});
        modelInfo.transformerProjections.push_back(
          {attn->outProj.inChannels,
          attn->outProj.outChannels,
          false}
        );
      }
      else if(block.first == TRANSFORMER_FFN_BLOCK_KIND) {
        const TransformerFFNDesc* ffn = static_cast<const TransformerFFNDesc*>(block.second.get());
        normChannels = ffn->linear1.inChannels;
        addChannelCase(modelInfo.residualChannels, ffn->numChannels);
        addChannelCase(modelInfo.rmsNormChannels, ffn->preLN.numChannels);
        if(ffn->useSwiGLU)
          addChannelCase(modelInfo.swiGLUChannels, ffn->ffnChannels);
        modelInfo.transformerFFNChannels = ffn->ffnChannels;
        modelInfo.transformerProjections.push_back({ffn->linear1.inChannels,
          ffn->linear1.outChannels + (ffn->useSwiGLU ? ffn->linearGate.outChannels : 0), true, ffn->useSwiGLU});
        modelInfo.transformerProjections.push_back(
          {ffn->linear2.inChannels,
          ffn->linear2.outChannels,
          false}
        );
      }
      else if(block.first == NESTED_BOTTLENECK_BLOCK_KIND) {
        const NestedBottleneckResidualBlockDesc* nested =
          static_cast<const NestedBottleneckResidualBlockDesc*>(block.second.get());
        addChannelCase(modelInfo.residualChannels, nested->postConv.outChannels);
        collectBlockTuningInfo(nested->blocks, modelInfo);
      }
      else if(block.first == GLOBAL_POOLING_BLOCK_KIND) {
        const auto* gpool = static_cast<const GlobalPoolingResidualBlockDesc*>(block.second.get());
        addChannelCase(modelInfo.residualChannels, gpool->finalConv.outChannels);
        addChannelCase(modelInfo.globalPoolChannels, gpool->gpoolConv.outChannels);
        addChannelCase(modelInfo.channelBiasChannels, gpool->gpoolToBiasMul.outChannels);
        modelInfo.gemmLayers.push_back({gpool->gpoolToBiasMul.inChannels,
          gpool->gpoolToBiasMul.outChannels, 1.0, false, VulkanTuner::GemmTuneCase::Kind::Dense});
      }
      else if(block.first == ORDINARY_BLOCK_KIND) {
        const auto* residual = static_cast<const ResidualBlockDesc*>(block.second.get());
        addChannelCase(modelInfo.residualChannels, residual->finalConv.outChannels);
      }
      if(normChannels > 0) {
        if(previousNormChannels == normChannels)
          modelInfo.transformerFusedNormCount++;
        else modelInfo.transformerPlainNormCount++;
        modelInfo.transformerUniformNormChannels &= normChannels == modelInfo.midNumChannels;
      }
      previousNormChannels = normChannels;
    }
  }
}

VulkanTuner::ModelInfoForTuning VulkanTuner::ModelInfoForTuning::ofDesc(const ModelDesc& desc) {
  VulkanTuner::ModelInfoForTuning modelInfo;
  modelInfo.numInputChannels = desc.numInputChannels;
  modelInfo.maxConvChannels1x1 = desc.maxConvChannels(1,1);
  modelInfo.maxConvChannels3x3 = desc.maxConvChannels(3,3);
  modelInfo.hasConv5x5 = desc.maxConvChannels(5,5) > 0;
  modelInfo.trunkNumChannels = desc.trunk.trunkNumChannels;
  modelInfo.midNumChannels = desc.trunk.midNumChannels;
  modelInfo.regularNumChannels = desc.trunk.regularNumChannels;
  modelInfo.gpoolNumChannels = desc.trunk.gpoolNumChannels;
  modelInfo.modelVersion = desc.modelVersion;
  modelInfo.hasRMSNorm = desc.trunk.trunkNormKind == TRUNK_NORM_KIND_RMSNORM &&
    !desc.trunk.trunkTipRMSNorm.spatial;
  modelInfo.hasSpatialRMSNorm = desc.trunk.trunkNormKind == TRUNK_NORM_KIND_RMSNORM &&
    desc.trunk.trunkTipRMSNorm.spatial;
  desc.iterConvLayers([&](const ConvLayerDesc& conv) {
    if(conv.convXSize != conv.convYSize ||
       (conv.convXSize != 1 && conv.convXSize != 3 && conv.convXSize != 5))
      return;
    const auto kind = conv.convXSize == 1 ? GemmTuneCase::Kind::Spatial :
      conv.convXSize == 3 ? GemmTuneCase::Kind::Winograd3x3 : GemmTuneCase::Kind::Winograd5x5;
    modelInfo.gemmLayers.push_back({conv.inChannels, conv.outChannels, 1.0, false, kind});
  });
  const auto addDense = [&](const MatMulLayerDesc& mul) {
    if(mul.inChannels > 0 && mul.outChannels > 0)
      modelInfo.gemmLayers.push_back({mul.inChannels, mul.outChannels, 1.0, false, GemmTuneCase::Kind::Dense});
  };
  addDense(desc.trunk.initialMatMul);
  if(desc.trunk.metaEncoderVersion > 0) {
    addDense(desc.trunk.sgfMetadataEncoder.mul1);
    addDense(desc.trunk.sgfMetadataEncoder.mul2);
    addDense(desc.trunk.sgfMetadataEncoder.mul3);
  }
  addDense(desc.policyHead.gpoolToBiasMul);
  addDense(desc.policyHead.gpoolToPassMul);
  addDense(desc.policyHead.gpoolToPassMul2);
  addDense(desc.valueHead.v2Mul);
  addDense(desc.valueHead.v3Mul);
  addDense(desc.valueHead.sv3Mul);
  addChannelCase(modelInfo.globalPoolChannels, desc.policyHead.g1Conv.outChannels);
  addChannelCase(modelInfo.channelBiasChannels, desc.trunk.initialMatMul.outChannels);
  addChannelCase(modelInfo.channelBiasChannels, desc.policyHead.p1Conv.outChannels);
  if(modelInfo.hasRMSNorm)
    addChannelCase(modelInfo.rmsNormChannels, desc.trunk.trunkTipRMSNorm.numChannels);
  collectBlockTuningInfo(desc.trunk.blocks, modelInfo);
  return modelInfo;
}

int VulkanTuner::getTransformerFFNInputChannelsForTuning(const ModelInfoForTuning& modelInfo) {
  return modelInfo.midNumChannels;
}

VulkanTuner::NonGemmTuningPlan VulkanTuner::getNonGemmTuningPlan(
  const ModelInfoForTuning& modelInfo,
  const VulkanTuneParams& config
) {
  const bool nhwc = config.vulkan.shouldUseNHWC;
  const bool attention = modelInfo.transformerHeadDim > 0 && modelInfo.transformerVHeadDim > 0 &&
    modelInfo.transformerNumHeads > 0 && modelInfo.transformerNumKVHeads > 0;
  return {
    modelInfo.maxConvChannels3x3 > 0 || modelInfo.hasConv5x5,
    attention && !nhwc,
    attention && nhwc,
    modelInfo.hasRMSNorm || (!nhwc && !modelInfo.transformerProjections.empty()),
    modelInfo.hasSpatialRMSNorm
  };
}

vector<VulkanTuner::GemmTuneCase> VulkanTuner::getGemmTuningCases(
  const string& tunerName,
  const ModelInfoForTuning& modelInfo,
  const VulkanTuneParams& config
) {
  vector<GemmTuneCase> cases;
  const auto addCase = [&](const GemmTuneCase& c) {
    if(c.inChannels <= 0 || c.outChannels <= 0)
      return;
    auto existing = find_if(cases.begin(), cases.end(), [&](const GemmTuneCase& other) {
      return c.inChannels == other.inChannels && c.outChannels == other.outChannels && c.kind == other.kind;
    });
    if(existing == cases.end())
      cases.push_back(c);
    else existing->weight += c.weight;
  };
  for(const auto& c: modelInfo.gemmLayers) {
    const bool dense = c.kind == GemmTuneCase::Kind::Dense;
    const bool spatial = c.kind == GemmTuneCase::Kind::Spatial;
    const bool nchwCM = config.vulkan.shouldUseCooperativeMatrix &&
      config.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
      config.hgemmCooperativeMatrixNCHW.isValid() &&
      c.inChannels % config.hgemmCooperativeMatrixNCHW.KWG == 0 &&
      c.outChannels % config.hgemmCooperativeMatrixNCHW.NWG == 0;
    if((tunerName == "xgemmDirect" && spatial && !config.vulkan.shouldUseNHWC && !nchwCM) ||
       ((tunerName == "xgemm" || tunerName == "xgemm16") && !dense && !spatial &&
        !config.vulkan.shouldUseCooperativeMatrix) ||
       (tunerName == "hgemmCooperativeMatrix" && !dense && !spatial) ||
       (tunerName == "hgemmCooperativeMatrixNHWC" && !dense) ||
       (tunerName == "hgemmCooperativeMatrixNCHW" && spatial))
      addCase(c);
  }
  if(tunerName == "transformerXgemm" && !config.vulkan.shouldUseNHWC)
    for(const auto& p: modelInfo.transformerProjections)
      addCase({p.inChannels, p.outChannels, 1.0});
  return cases;
}

string VulkanTuner::defaultDirectory(bool makeDir, const string& homeDataDirOverride) {
  string dir = HomeData::getHomeDataDir(true, homeDataDirOverride) + "/vulkantuning";
  if(makeDir)
    MakeDir::make(dir);
  return dir;
}

string
VulkanTuner::defaultFileName(
  const string& gpuName,
  int nnXLen,
  int nnYLen,
  int trunkNumChannels,
  int modelVersion
) {
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
VulkanTuner::defaultFileName(
  const string& gpuName,
  int nnXLen,
  int nnYLen,
  const ModelInfoForTuning& modelInfo
) {
  return defaultFileName(gpuName, nnXLen, nnYLen, modelInfo.trunkNumChannels, modelInfo.modelVersion);
}

namespace {
  bool selectHgemmCooperativeMatrixProperties(
    const VulkanDevice* device,
    HGemmCooperativeMatrixNCHWTuneParams& params,
    int accType
  ) {
#if defined(VK_KHR_cooperative_matrix)
    if(device == nullptr)
      return false;

    vector<VkCooperativeMatrixPropertiesKHR> properties;
    if(!getSupportedCooperativeMatrixProperties(
         device->info.physicalDevice,
         device->info.cooperativeMatrixPropertiesFn,
         device->info.cooperativeMatrixFeatures.cooperativeMatrix,
         properties
       ))
      return false;

    for(bool requireRequestedShape: {true, false}) {
      for(const VkCooperativeMatrixPropertiesKHR& property: properties) {
        if(!supportsCooperativeMatrixType(property, accType) ||
           (requireRequestedShape &&
            (static_cast<int>(property.MSize) != params.MWARP ||
             static_cast<int>(property.NSize) != params.NWARP ||
             static_cast<int>(property.KSize) != params.KDIM)))
          continue;
        params.accType = accType;
        params.MWARP = static_cast<int>(property.MSize);
        params.NWARP = static_cast<int>(property.NSize);
        params.KDIM = static_cast<int>(property.KSize);
        params.subgroupSize = device->info.subgroupProperties.subgroupSize;
        params.MWG = params.MWARP;
        params.NWG = params.NWARP;
        params.KWG = params.KDIM;
        params.MWAVE = params.MWARP;
        params.NWAVE = params.NWARP;
        params.SB = 0;
        params.VWM = largestCooperativeMatrixVectorWidth(params.MWARP);
        params.VWN = largestCooperativeMatrixVectorWidth(params.NWARP);
        if(params.isValid())
          return true;
      }
    }
    return false;

#else
    (void)device;
    (void)params;
    (void)accType;
    return false;
#endif
  }

  bool selectHgemmCooperativeMatrixProperties(
    const VulkanDevice* device,
    HGemmCooperativeMatrixNCHWTuneParams& params
  ) {
    const int preferredAccType = params.accType == 32 ? 32 : 16;
    if(selectHgemmCooperativeMatrixProperties(device, params, preferredAccType))
      return true;
    return selectHgemmCooperativeMatrixProperties(device, params, preferredAccType == 16 ? 32 : 16);
  }

  bool selectHgemmCooperativeMatrixProperties(
    const VulkanDevice* device,
    HGemmCooperativeMatrixTuneParams& params,
    int accType
  ) {
#if defined(VK_KHR_cooperative_matrix)
    if(device == nullptr)
      return false;

    vector<VkCooperativeMatrixPropertiesKHR> properties;
    if(!getSupportedCooperativeMatrixProperties(
         device->info.physicalDevice,
         device->info.cooperativeMatrixPropertiesFn,
         device->info.cooperativeMatrixFeatures.cooperativeMatrix,
         properties
       ))
      return false;

    for(bool requireRequestedShape: {true, false}) {
      for(const VkCooperativeMatrixPropertiesKHR& property: properties) {
        if(!supportsCooperativeMatrixType(property, accType) ||
           (requireRequestedShape &&
            (static_cast<int>(property.MSize) != params.MWARP ||
             static_cast<int>(property.NSize) != params.NWARP ||
             static_cast<int>(property.KSize) != params.KDIM)))
          continue;
        params.accType = accType;
        params.MWARP = static_cast<int>(property.MSize);
        params.NWARP = static_cast<int>(property.NSize);
        params.KDIM = static_cast<int>(property.KSize);
        params.subgroupSize = device->info.subgroupProperties.subgroupSize;
        params.MWG = params.MWARP;
        params.NWG = params.NWARP;
        params.KWG = params.KDIM;
        params.MWAVE = params.MWARP;
        params.NWAVE = params.NWARP;
        params.SA = 0;
        params.SB = 0;
        params.VWM = largestCooperativeMatrixVectorWidth(params.MWARP);
        params.VWN = largestCooperativeMatrixVectorWidth(params.NWARP);
        if(params.isValid())
          return true;
      }
    }
    return false;

#else
    (void)device;
    (void)params;
    (void)accType;
    return false;
#endif
  }

  bool selectHgemmCooperativeMatrixProperties(
    const VulkanDevice* device,
    HGemmCooperativeMatrixTuneParams& params
  ) {
    const int preferredAccType = params.accType == 32 ? 32 : 16;
    if(selectHgemmCooperativeMatrixProperties(device, params, preferredAccType))
      return true;
    return selectHgemmCooperativeMatrixProperties(device, params, preferredAccType == 16 ? 32 : 16);
  }

  class VulkanTimestampTimer;

  struct TuningContext {
    const VulkanDevice* device;
    int batchSize;
    int nnXLen;
    int nnYLen;
    const VulkanTuner::ModelInfoForTuning& modelInfo;
    bool full;
    vector<CooperativeMatrixTuneShape> cooperativeMatrixTuneShapes;
    Logger* logger;
    VulkanTimestampTimer* timer;
    bool printOnlyOnImprovement;
    const HGemmCooperativeMatrixNHWCTuneParams* transformerPhysicalLayout = nullptr;
    int transformerPhysicalSpatialSize = 0;
    bool transformerFusedFFN = false;
    int attentionValidationMask = 0;
    bool recheckAttentionFinalists = false;
    bool preserveGemmMeasurementOrder = false;
    vector<float>* transformerXgemmCpuReference = nullptr;
    const VulkanTuneParams* selectedGemmConfig = nullptr;
    map<string, VulkanTuneParams::TuningRecord>* tuningRecords = nullptr;
    bool onlyMissing = false;
    mutable string lastPrintedTuningParams = string();
    // Temporary scalar Attention single-case diagnostics.
    bool attentionProbe = false;
    bool tegraAttentionTightBuffers = false;
    bool tegraAttentionFreshCommands = false;
    bool attentionSkipTimestamps = false;
    bool attentionSkipSnapshots = false;
  };

  bool usesGenericNHWC(const VulkanTuneParams& config) {
    return config.vulkan.shouldUseNHWC &&
           config.vulkan.canUseCooperativeMatrix &&
           config.vulkan.canUseFP16Storage &&
           config.vulkan.canUseFP16Compute &&
           config.vulkan.shouldUseFP16Storage &&
           config.vulkan.shouldUseFP16Compute &&
           config.vulkan.shouldUseCooperativeMatrix;
  }

  bool comparesScalarAndCooperativeAttention(const TuningContext& context, const VulkanTuneParams& config) {
    const auto& mode = config.vulkan;
    return !context.onlyMissing && mode.canUseCooperativeMatrix &&
      mode.canUseFP16Storage && mode.canUseFP16Compute && mode.shouldUseFP16Storage;
  }

  bool usesNonCooperativeNCHW(const VulkanTuneParams& config) {
    return !config.vulkan.shouldUseNHWC && !config.vulkan.shouldUseCooperativeMatrix &&
           !config.vulkan.shouldUseHgemmCooperativeMatrixNCHW;
  }

  vector<int> getModelChannelCases(const string& tunerName, const VulkanTuner::ModelInfoForTuning& model) {
    if(tunerName == "gPool")
      return model.globalPoolChannels;
    if(tunerName == "pointwise")
      return model.residualChannels;
    if(tunerName == "addChannelBiases")
      return model.channelBiasChannels;
    if(tunerName == "transformerRMSNorm")
      return model.rmsNormChannels;

    vector<int> cases;
    if(tunerName.find("conv") == 0) {
      using Kind = VulkanTuner::GemmTuneCase::Kind;
      const auto kind = tunerName.find("5x5") != string::npos
        ? Kind::Winograd5x5 : Kind::Winograd3x3;
      const bool input = tunerName.find("InputTransform") != string::npos;
      for(const auto& conv: model.gemmLayers)
        if(conv.kind == kind)
          addChannelCase(cases, input ? conv.inChannels : conv.outChannels);
    }
    return cases;
  }

  template<typename Case>
  Case workloadCaseForSlot(const vector<Case>& cases, size_t slot, Case fallback) {
    // Slot zero has no score. Reuse the first case there so it is also timed
    // with a positive weight in slot one.
    return cases.empty() ? fallback : cases[(slot == 0 ? 0 : slot - 1) % cases.size()];
  }

  const HGemmCooperativeMatrixNHWCTuneParams& nhwcTuningParams(
    const VulkanTuneParams& config
  ) {
    return config.hgemmCooperativeMatrixNHWC;
  }

  bool isSupportedCooperativeMatrixShape(
    const TuningContext& context,
    int accType,
    int mSize,
    int nSize,
    int kSize,
    uint32_t subgroupSize
  ) {
    return any_of(
      context.cooperativeMatrixTuneShapes.begin(),
      context.cooperativeMatrixTuneShapes.end(),
      [=](const CooperativeMatrixTuneShape& shape) {
        return shape.accType == accType &&
               shape.MSize == mSize &&
               shape.NSize == nSize &&
               shape.KSize == kSize &&
               shape.subgroupSize == subgroupSize;
      }
    );
  }

  bool isValidCooperativeMatrixTuneParams(
    const TuningContext& context,
    const HGemmCooperativeMatrixTuneParams& params
  ) {
    return context.device != nullptr &&
           isSupportedCooperativeMatrixShape(
             context, params.accType, params.MWARP, params.NWARP, params.KDIM, params.subgroupSize
           ) &&
           isWithinCooperativeMatrixDeviceLimits(context.device->info, params);
  }

  bool isValidCooperativeMatrixTuneParams(
    const TuningContext& context,
    const HGemmCooperativeMatrixNHWCTuneParams& params
  ) {
    return context.device != nullptr &&
           isSupportedCooperativeMatrixShape(
             context, params.accType, params.MWARP, params.NWARP, params.KDIM, params.subgroupSize
           ) &&
           isWithinCooperativeMatrixDeviceLimits(context.device->info, params);
  }

  bool isValidCooperativeMatrixTuneParams(
    const TuningContext& context,
    const HGemmCooperativeMatrixNCHWTuneParams& params
  ) {
    return context.device != nullptr &&
           isSupportedCooperativeMatrixShape(
             context, params.accType, params.MWARP, params.NWARP, params.KDIM, params.subgroupSize
           ) &&
           isWithinCooperativeMatrixDeviceLimits(context.device->info, params);
  }

  bool isValidCooperativeMatrixTuneParams(
    const TuningContext& context,
    const TransformerDualGemmSwiGLUTuneParams& params
  ) {
    return context.device != nullptr &&
           isSupportedCooperativeMatrixShape(
             context, params.accType, params.MWARP, params.NWARP, params.KDIM, params.subgroupSize
           ) &&
           isWithinCooperativeMatrixDeviceLimits(context.device->info, params);
  }

  template<typename Tuner>
  bool isValidTuningConfig(const TuningContext&, const VulkanTuneParams& config) {
    return Tuner::isValid(config);
  }

  using VulkanTuner::GemmTuneCase;

  vector<GemmTuneCase> getGemmTuneCases(
    const TuningContext& context,
    bool includeTransformerCases,
    bool use3x3,
    bool includeInputConvCase = true
  ) {
    int maxConvChannels = use3x3 ? context.modelInfo.maxConvChannels3x3 : context.modelInfo.maxConvChannels1x1;
    maxConvChannels = std::max(context.modelInfo.trunkNumChannels, maxConvChannels);
    maxConvChannels = std::max(context.modelInfo.midNumChannels, maxConvChannels);
    maxConvChannels = std::max(context.modelInfo.regularNumChannels, maxConvChannels);
    maxConvChannels = std::max(context.modelInfo.gpoolNumChannels, maxConvChannels);

    vector<GemmTuneCase> cases;
    const auto addCase = [&](
      int inChannels,
      int outChannels,
      double weight,
      bool padOutputForNCHW = false
    ) {
      if(inChannels <= 0 || outChannels <= 0 || weight <= 0.0)
        return;
      const auto existing = find_if(cases.begin(), cases.end(), [&](const GemmTuneCase& tuneCase) {
        return tuneCase.inChannels == inChannels && tuneCase.outChannels == outChannels &&
               tuneCase.padOutputForNCHW == padOutputForNCHW;
      });
      if(existing == cases.end())
        cases.push_back({inChannels, outChannels, weight, padOutputForNCHW});
      else
        existing->weight += weight;
    };
    if(use3x3 && includeInputConvCase)
      addCase(context.modelInfo.numInputChannels, context.modelInfo.trunkNumChannels, 1.0);
    addCase(context.modelInfo.trunkNumChannels, context.modelInfo.midNumChannels, 1.0);
    addCase(context.modelInfo.midNumChannels, context.modelInfo.trunkNumChannels, 1.0);
    addCase(context.modelInfo.trunkNumChannels, context.modelInfo.trunkNumChannels, 1.0);
    addCase(context.modelInfo.trunkNumChannels, context.modelInfo.regularNumChannels, 0.2);
    addCase(context.modelInfo.regularNumChannels, context.modelInfo.trunkNumChannels, 0.2);
    addCase(context.modelInfo.trunkNumChannels, context.modelInfo.gpoolNumChannels, 0.2);
    addCase(maxConvChannels, maxConvChannels, 1.0);
    if(includeTransformerCases &&
       context.modelInfo.transformerHeadDim > 0 && context.modelInfo.transformerVHeadDim > 0 &&
       context.modelInfo.transformerNumHeads > 0 && context.modelInfo.transformerNumKVHeads > 0) {
      const int transformerQKC = context.modelInfo.transformerNumHeads * context.modelInfo.transformerHeadDim;
      const int transformerKC = context.modelInfo.transformerNumKVHeads * context.modelInfo.transformerHeadDim;
      const int transformerVC = context.modelInfo.transformerNumKVHeads * context.modelInfo.transformerVHeadDim;
      const int transformerOutC = context.modelInfo.transformerNumHeads * context.modelInfo.transformerVHeadDim;
      const int transformerFFNC = context.modelInfo.transformerFFNChannels;
      addCase(context.modelInfo.midNumChannels, transformerQKC, 1.0);
      addCase(context.modelInfo.midNumChannels, transformerKC, 1.0);
      addCase(context.modelInfo.midNumChannels, transformerVC, 1.0);
      addCase(transformerOutC, context.modelInfo.midNumChannels, 1.0);
      addCase(context.modelInfo.midNumChannels, 2 * transformerFFNC, 1.0, true);
      addCase(transformerFFNC, context.modelInfo.midNumChannels, 1.0);
    }
    return cases;
  }

  vector<GemmTuneCase> getTunerGemmTuneCases(
    const string& tunerName,
    const TuningContext& context
  ) {
    if(context.transformerPhysicalLayout != nullptr && tunerName == "hgemmCooperativeMatrixNHWC") {
      vector<GemmTuneCase> cases;
      for(const auto& projection: context.modelInfo.transformerProjections) {
        if(context.transformerFusedFFN && projection.fusedFFNInput)
          continue;
        const int outChannels = projection.outChannels;
        const auto existing = find_if(cases.begin(), cases.end(), [&](const GemmTuneCase& c) {
          return c.inChannels == projection.inChannels && c.outChannels == outChannels;
        });
        if(existing == cases.end())
          cases.push_back({projection.inChannels, outChannels, 1.0});
        else existing->weight += 1.0;
      }
      return cases;
    }
    if(tunerName == "transformerXgemm")
      return VulkanTuner::getGemmTuningCases(tunerName, context.modelInfo,
        context.selectedGemmConfig != nullptr ? *context.selectedGemmConfig : VulkanTuneParams());
    if(context.selectedGemmConfig != nullptr && !context.modelInfo.gemmLayers.empty())
      return VulkanTuner::getGemmTuningCases(tunerName, context.modelInfo, *context.selectedGemmConfig);
    const bool direct = tunerName == "xgemmDirect" || tunerName == "hgemmCooperativeMatrixNCHW";
    const bool includeInputConvCase = tunerName != "hgemmCooperativeMatrix";
    return getGemmTuneCases(context, tunerName == "hgemmCooperativeMatrixNCHW", !direct, includeInputConvCase);
  }

  size_t getWorkloadCaseCount(const string& tunerName, const TuningContext& context);

  vector<double> getWorkloadWeights(const string& tunerName, const TuningContext& context) {
    if(tunerName == "xgemm" || tunerName == "xgemm16") {
      vector<GemmTuneCase> cases = getTunerGemmTuneCases(tunerName, context);
      vector<double> weights;
      weights.reserve(cases.size());
      for(const GemmTuneCase& tuneCase: cases)
        weights.push_back(tuneCase.weight);
      return weights;
    }
    if(tunerName == "xgemmDirect" || tunerName == "transformerXgemm" || tunerName == "hgemmCooperativeMatrixNCHW") {
      vector<GemmTuneCase> cases = getTunerGemmTuneCases(tunerName, context);
      vector<double> weights;
      weights.reserve(cases.size());
      for(const GemmTuneCase& tuneCase: cases)
        weights.push_back(tuneCase.weight);
      return weights;
    }
    if(tunerName == "hgemmCooperativeMatrix" || tunerName == "hgemmCooperativeMatrixNHWC") {
      vector<GemmTuneCase> cases = getTunerGemmTuneCases(tunerName, context);
      vector<double> weights;
      weights.reserve(cases.size());
      for(const GemmTuneCase& tuneCase: cases)
        weights.push_back(tuneCase.weight);
      return weights;
    }

    const size_t workloadCaseCount = getWorkloadCaseCount(tunerName, context);
    vector<double> weights(workloadCaseCount, 1.0);
    if(!weights.empty())
      weights[0] = 0.0;
    return weights;
  }

  struct TuningMeasurementPlan {
    string kernelName;
    size_t totalRuns;
    size_t warmupRuns;
    double errorTolerance;
    double hardCutoff;
    vector<int> batchSizes;
    vector<GemmTuneCase> gemmCases;
    vector<double> workloadWeights;

    size_t timedRuns() const {
      return totalRuns > warmupRuns ? totalRuns - warmupRuns : 0;
    }

    double weightForRun(size_t run) const {
      return workloadWeights.empty() ? 1.0 : workloadWeights[run % workloadWeights.size()];
    }
  };

  bool isTransformerAttentionTuner(const string& name);

  size_t getWorkloadCaseCount(const string& tunerName, const TuningContext& context) {
    if(tunerName == "xgemmDirect" || tunerName == "transformerXgemm" || tunerName == "hgemmCooperativeMatrixNCHW")
      return context.modelInfo.transformerHeadDim > 0 && context.modelInfo.transformerVHeadDim > 0 &&
             context.modelInfo.transformerNumHeads > 0 && context.modelInfo.transformerNumKVHeads > 0 ? 10 : 6;
    if(tunerName == "xgemm" || tunerName == "xgemm16" || tunerName == "hgemmCooperativeMatrix" ||
       tunerName == "hgemmCooperativeMatrixNHWC")
      return 6;
    if(isTransformerAttentionTuner(tunerName))
      return 6;
    return 10;
  }

  vector<int> getTuningBatchSizes(const TuningContext& context) {
    return {std::max(1, context.batchSize)};
  }

  TuningMeasurementPlan makeMeasurementPlan(const string& tunerName, const TuningContext& context) {
    const bool isGemm = tunerName == "xgemmDirect" || tunerName == "transformerXgemm" ||
       tunerName == "xgemm" || tunerName == "xgemm16" ||
       tunerName == "hgemmCooperativeMatrix" || tunerName == "hgemmCooperativeMatrixNHWC" ||
                        tunerName == "hgemmCooperativeMatrixNCHW";
    const vector<int> batchSizes = getTuningBatchSizes(context);
    const vector<GemmTuneCase> gemmCases = isGemm ?
      getTunerGemmTuneCases(tunerName, context) : vector<GemmTuneCase>();
    const vector<double> workloadWeights = getWorkloadWeights(tunerName, context);
    const size_t workloadCaseCount = workloadWeights.size();
    if(isGemm) {
      const size_t warmupRuns = workloadCaseCount * batchSizes.size();
      const size_t totalRuns = 6 * workloadCaseCount * batchSizes.size();
      const double tolerance = (tunerName == "xgemmDirect" || tunerName == "transformerXgemm" ||
                                tunerName == "hgemmCooperativeMatrix") ? 0.01 :
                               (tunerName == "xgemm" || tunerName == "xgemm16") ? 0.005 : 0.002;
      return {
        tunerName, totalRuns, warmupRuns, tolerance, tolerance * 5.0,
        batchSizes, gemmCases, workloadWeights
      };
    }
    if(tunerName == "pointwise" || tunerName == "transformerRMSNorm" || tunerName == "spatialRMSNorm")
      return {tunerName, 20, 0, 0.05, 0.25, batchSizes, {}, workloadWeights};
    if(isTransformerAttentionTuner(tunerName))
      return {
        tunerName, 12, 0, VulkanTuner::COOPERATIVE_MATRIX_ROPE_ERROR_TOLERANCE,
        VulkanTuner::COOPERATIVE_MATRIX_ROPE_ERROR_TOLERANCE * 5.0,
        batchSizes, {}, workloadWeights
      };
    return {tunerName, 20, 0, 0.005, 0.025, batchSizes, {}, workloadWeights};
  }

  TuningMeasurementPlan makeCooperativeMatrixScreeningMeasurementPlan(
    const string& tunerName,
    const TuningContext& context
  ) {
    TuningMeasurementPlan plan = makeMeasurementPlan(tunerName, context);
    assert(tunerName == "hgemmCooperativeMatrix" || tunerName == "hgemmCooperativeMatrixNHWC" ||
           tunerName == "hgemmCooperativeMatrixNCHW");
    assert(!plan.gemmCases.empty());

    // NCHW selection is eligibility-aware, so it must see the complete model
    // case list to account for both cooperative and fallback workloads.
    if(tunerName == "hgemmCooperativeMatrixNCHW")
      return plan;

    // Screen hardware matrix shapes using one common convolution case and the
    // largest available case. The selected shape is always remeasured with the
    // complete plan before it can be saved.
    vector<GemmTuneCase> cases;
    const auto addScreeningCase = [&](const GemmTuneCase& candidate) {
      const bool duplicate = any_of(cases.begin(), cases.end(), [&](const GemmTuneCase& existing) {
        return existing.inChannels == candidate.inChannels && existing.outChannels == candidate.outChannels;
      });
      if(!duplicate)
        cases.push_back(candidate);
    };
    addScreeningCase(plan.gemmCases.front());
    addScreeningCase(plan.gemmCases[std::min<size_t>(1, plan.gemmCases.size() - 1)]);
    addScreeningCase(plan.gemmCases.back());
    plan.gemmCases = cases;
    plan.workloadWeights.clear();
    for(const GemmTuneCase& tuneCase: cases)
      plan.workloadWeights.push_back(tuneCase.weight);
    plan.warmupRuns = cases.size();
    plan.totalRuns = 3 * cases.size();
    return plan;
  }

  bool usesCpuReference(const string& tunerName) {
    return tunerName != "conv3x3InputTransform" && tunerName != "conv3x3OutputTransform" &&
           tunerName != "conv5x5InputTransform" && tunerName != "conv5x5OutputTransform";
  }

  void validateReadback(
    const vector<float>& reference,
    const vector<float>& values,
    const TuningMeasurementPlan& plan,
    double& errorProp
  ) {
    errorProp = VulkanTuner::computeErrorProp(reference, values);
    if(!isfinite(errorProp) || errorProp > plan.hardCutoff)
      errorProp = 1.0;
  }

  template<typename Setter>
  void addCandidates(
    vector<VulkanTuneParams>& configs,
    const vector<int>& values,
    const Setter& setter
  ) {
    vector<VulkanTuneParams> expanded;
    expanded.reserve(configs.size() * values.size());
    for(int value: values) {
      for(const VulkanTuneParams& config: configs) {
        VulkanTuneParams candidate = config;
        setter(candidate, value);
        expanded.push_back(candidate);
      }
    }
    configs = expanded;
  }

  bool checkedMultiply(
    int value,
    int multiplier,
    int& result
  ) {
    if(value <= 0 || multiplier <= 0 || value > numeric_limits<int>::max() / multiplier)
      return false;
    result = value * multiplier;
    return true;
  }

  void dedupCandidates(vector<VulkanTuneParams>& configs) {
    vector<VulkanTuneParams> unique;
    for(const VulkanTuneParams& config: configs) {
      bool found = false;
      for(const VulkanTuneParams& previous: unique) {
        if(config == previous) {
          found = true;
          break;
        }
      }
      if(!found)
        unique.push_back(config);
    }
    configs = unique;
  }

  template<typename Tuner>
  struct KeepsCurrentConfigFirst {
    static constexpr bool value = false;
  };

  template<typename Tuner>
  struct StopsOnReferenceImplFail {
    static bool value(const VulkanTuneParams&) { return false; }
  };

  bool isTransformerAttentionTuner(const string& name) {
    return name == "transformerScaleDotProduct" ||
           name == "transformerScaleDotProductCooperativeMatrixNCHW" ||
           name == "transformerScaleDotProductCooperativeMatrixNHWC";
  }

  int transformerAttentionBlockQ(const VulkanTuneParams& config, const string& tunerName) {
    if(tunerName == "transformerScaleDotProductCooperativeMatrixNCHW")
      return config.transformerCooperativeMatrixNCHW.ATTN_BLOCK_Q;
    if(tunerName == "transformerScaleDotProductCooperativeMatrixNHWC")
      return config.transformerCooperativeMatrixNHWC.ATTN_BLOCK_Q;
    return config.transformer.ATTN_BLOCK_Q;
  }

  string describeTuningParams(const string& tunerName, const VulkanTuneParams& config) {
    ostringstream out;
    bool first = true;
    const auto add = [&](const char* name, auto value) {
      if(!first)
        out << " ";
      out << name << "=" << value;
      first = false;
    };

    add("fp16Storage", config.vulkan.shouldUseFP16Storage);
    add("fp16Compute", config.vulkan.shouldUseFP16Compute);
    if(tunerName == "xgemmDirect" || tunerName == "transformerXgemm") {
      const auto& params = tunerName == "transformerXgemm" ? config.transformerXgemm : config.xgemmDirect;
      add("WGD", params.WGD);
      add("MDIMCD", params.MDIMCD);
      add("NDIMCD", params.NDIMCD);
      add("MDIMAD", params.MDIMAD);
      add("NDIMBD", params.NDIMBD);
      add("KWID", params.KWID);
      add("PADA", params.PADA);
      add("PADB", params.PADB);
      add("VWMD", params.VWMD);
      add("VWND", params.VWND);
    }
    else if(tunerName == "xgemm" || tunerName == "xgemm16") {
      const XgemmTuneParams& params = tunerName == "xgemm" ? config.xgemm : config.xgemm16;
      add("MDIMC", params.MDIMC);
      add("NDIMC", params.NDIMC);
      add("MWG", params.MWG);
      add("NWG", params.NWG);
      add("KWG", params.KWG);
      add("KWI", params.KWI);
      add("MDIMA", params.MDIMA);
      add("NDIMB", params.NDIMB);
      add("VWM", params.VWM);
      add("VWN", params.VWN);
    }
    else if(tunerName == "hgemmCooperativeMatrix") {
      add("MWARP", config.hgemmCooperativeMatrix.MWARP);
      add("NWARP", config.hgemmCooperativeMatrix.NWARP);
      add("KDIM", config.hgemmCooperativeMatrix.KDIM);
      add("subgroupSize", config.hgemmCooperativeMatrix.subgroupSize);
      add("MWG", config.hgemmCooperativeMatrix.MWG);
      add("NWG", config.hgemmCooperativeMatrix.NWG);
      add("KWG", config.hgemmCooperativeMatrix.KWG);
      add("MWAVE", config.hgemmCooperativeMatrix.MWAVE);
      add("NWAVE", config.hgemmCooperativeMatrix.NWAVE);
      add("accType", config.hgemmCooperativeMatrix.accType);
      add("SA", config.hgemmCooperativeMatrix.SA);
      add("SB", config.hgemmCooperativeMatrix.SB);
      add("VWM", config.hgemmCooperativeMatrix.VWM);
      add("VWN", config.hgemmCooperativeMatrix.VWN);
      add("sM", config.hgemmCooperativeMatrix.MWAVE / config.hgemmCooperativeMatrix.MWARP);
      add("sN", config.hgemmCooperativeMatrix.NWAVE / config.hgemmCooperativeMatrix.NWARP);
      add("rM", config.hgemmCooperativeMatrix.MWG / config.hgemmCooperativeMatrix.MWAVE);
      add("rN", config.hgemmCooperativeMatrix.NWG / config.hgemmCooperativeMatrix.NWAVE);
      add("rK", config.hgemmCooperativeMatrix.KWG / config.hgemmCooperativeMatrix.KDIM);
      add("localSubgroups",
          (config.hgemmCooperativeMatrix.MWAVE / config.hgemmCooperativeMatrix.MWARP) *
          (config.hgemmCooperativeMatrix.NWAVE / config.hgemmCooperativeMatrix.NWARP));
      add("sharedBytes",
          2 * (
          (config.hgemmCooperativeMatrix.SA == 1
            ? config.hgemmCooperativeMatrix.MWG * config.hgemmCooperativeMatrix.KWG : 0) +
          (config.hgemmCooperativeMatrix.SB == 1
            ? config.hgemmCooperativeMatrix.NWG * config.hgemmCooperativeMatrix.KWG : 0)) +
          (config.hgemmCooperativeMatrix.accType == 32
            ? 4 * config.hgemmCooperativeMatrix.MWG * config.hgemmCooperativeMatrix.NWG : 0) +
          16 * (config.hgemmCooperativeMatrix.SA + config.hgemmCooperativeMatrix.SB +
            (config.hgemmCooperativeMatrix.accType == 32 ? 1 : 0)));
    }
    else if(tunerName == "hgemmCooperativeMatrixNHWC") {
      const auto& params = nhwcTuningParams(config);
      add("MWARP", params.MWARP);
      add("NWARP", params.NWARP);
      add("KDIM", params.KDIM);
      add("subgroupSize", params.subgroupSize);
      add("MWG", params.MWG);
      add("NWG", params.NWG);
      add("KWG", params.KWG);
      add("MWAVE", params.MWAVE);
      add("NWAVE", params.NWAVE);
      add("accType", params.accType);
      add("SA", params.SA);
      add("SB", params.SB);
      add("DB", params.DB);
      add("VWK", params.VWK);
      add("VWN", params.VWN);
      add("sM", params.MWAVE / params.MWARP);
      add("sN", params.NWAVE / params.NWARP);
      add("rM", params.MWG / params.MWAVE);
      add("rN", params.NWG / params.NWAVE);
      add("rK", params.KWG / params.KDIM);
      add("localSubgroups",
          (params.MWAVE / params.MWARP) * (params.NWAVE / params.NWARP));
      add("sharedBytes",
          2 * ((params.SA == 1 ? params.MWG * params.KWG : 0) +
          (params.SB == 1 ? params.KWG * params.NWG : 0)) +
          (params.accType == 32 ? 4 * params.MWG * params.NWG : 0) +
          16 * (params.SA + params.SB + (params.accType == 32 ? 1 : 0)));
    }
    else if(tunerName == "hgemmCooperativeMatrixNCHW") {
      add("MWARP", config.hgemmCooperativeMatrixNCHW.MWARP);
      add("NWARP", config.hgemmCooperativeMatrixNCHW.NWARP);
      add("KDIM", config.hgemmCooperativeMatrixNCHW.KDIM);
      add("subgroupSize", config.hgemmCooperativeMatrixNCHW.subgroupSize);
      add("MWG", config.hgemmCooperativeMatrixNCHW.MWG);
      add("NWG", config.hgemmCooperativeMatrixNCHW.NWG);
      add("KWG", config.hgemmCooperativeMatrixNCHW.KWG);
      add("MWAVE", config.hgemmCooperativeMatrixNCHW.MWAVE);
      add("NWAVE", config.hgemmCooperativeMatrixNCHW.NWAVE);
      add("accType", config.hgemmCooperativeMatrixNCHW.accType);
      add("SB", config.hgemmCooperativeMatrixNCHW.SB);
      add("VWM", config.hgemmCooperativeMatrixNCHW.VWM);
      add("VWN", config.hgemmCooperativeMatrixNCHW.VWN);
      add("sM", config.hgemmCooperativeMatrixNCHW.MWAVE / config.hgemmCooperativeMatrixNCHW.MWARP);
      add("sN", config.hgemmCooperativeMatrixNCHW.NWAVE / config.hgemmCooperativeMatrixNCHW.NWARP);
      add("rM", config.hgemmCooperativeMatrixNCHW.MWG / config.hgemmCooperativeMatrixNCHW.MWAVE);
      add("rN", config.hgemmCooperativeMatrixNCHW.NWG / config.hgemmCooperativeMatrixNCHW.NWAVE);
      add("rK", config.hgemmCooperativeMatrixNCHW.KWG / config.hgemmCooperativeMatrixNCHW.KDIM);
      add("localSubgroups",
          (config.hgemmCooperativeMatrixNCHW.MWAVE / config.hgemmCooperativeMatrixNCHW.MWARP) *
          (config.hgemmCooperativeMatrixNCHW.NWAVE / config.hgemmCooperativeMatrixNCHW.NWARP));
      add("sharedBytes",
          (config.hgemmCooperativeMatrixNCHW.accType == 32 ? 4 : 2) *
            config.hgemmCooperativeMatrixNCHW.MWG * config.hgemmCooperativeMatrixNCHW.NWG +
          (config.hgemmCooperativeMatrixNCHW.SB == 1
            ? 2 * config.hgemmCooperativeMatrixNCHW.NWG * config.hgemmCooperativeMatrixNCHW.KWG : 0) +
          16 * (1 + config.hgemmCooperativeMatrixNCHW.SB));
    }
    else if(
      tunerName == "conv3x3InputTransform" || tunerName == "conv3x3OutputTransform" ||
      tunerName == "conv5x5InputTransform" || tunerName == "conv5x5OutputTransform"
    ) {
      const ConvTuneParams& conv = tunerName.find("5x5") != string::npos ? config.conv5x5 : config.conv3x3;
      add("inTileYSize", conv.inTileYSize);
      add("inTileXSize", conv.inTileXSize);
      add("outTileYSize", conv.outTileYSize);
      add("outTileXSize", conv.outTileXSize);
      if(tunerName.find("Input") != string::npos) {
        if(usesGenericNHWC(config)) {
          add("inputTransformNHWCLocalXSize", conv.inputTransformNHWCLocalXSize);
          add("inputTransformNHWCLocalYSize", conv.inputTransformNHWCLocalYSize);
        }
        else {
          add("inputTransformLocalXSize", conv.inputTransformLocalXSize);
          add("inputTransformLocalYSize", conv.inputTransformLocalYSize);
        }
      }
      else {
        if(usesGenericNHWC(config)) {
          add("outputTransformNHWCLocalXSize", conv.outputTransformNHWCLocalXSize);
          add("outputTransformNHWCLocalYSize", conv.outputTransformNHWCLocalYSize);
        }
        else {
          add("outputTransformLocalXSize", conv.outputTransformLocalXSize);
          add("outputTransformLocalYSize", conv.outputTransformLocalYSize);
          add("outputTransformLocalZSize", conv.outputTransformLocalZSize);
        }
      }
    }
    else if(tunerName == "gPool") {
      add("XYSTRIDE", config.gPool.XYSTRIDE);
      add("CHANNELSTRIDE", config.gPool.CHANNELSTRIDE);
      add("BATCHSTRIDE", config.gPool.BATCHSTRIDE);
    }
    else if(tunerName == "pointwise") {
      add("LOCAL_SIZE", config.pointwise.LOCAL_SIZE);
      add("ELTS_PER_THREAD", config.pointwise.ELTS_PER_THREAD);
    }
    else if(tunerName == "addChannelBiases") {
      add("XY_ELTS_PER_THREAD", config.addChannelBiases.XY_ELTS_PER_THREAD);
      add("NC_ELTS_PER_THREAD", config.addChannelBiases.NC_ELTS_PER_THREAD);
    }
    else if(tunerName == "transformerScaleDotProduct") {
      add("ATTN_BLOCK_Q", config.transformer.ATTN_BLOCK_Q);
      add("ATTN_BLOCK_KV", config.transformer.ATTN_BLOCK_KV);
      add("Q_PER_THREAD", config.transformer.Q_PER_THREAD);
      add("USE_TILED_ATTN", config.transformer.USE_TILED_ATTN);
    }
    else if(
      tunerName == "transformerScaleDotProductCooperativeMatrixNCHW" ||
            tunerName == "transformerScaleDotProductCooperativeMatrixNHWC"
    ) {
      const auto& params = tunerName == "transformerScaleDotProductCooperativeMatrixNCHW"
        ? config.transformerCooperativeMatrixNCHW : config.transformerCooperativeMatrixNHWC;
      add("ATTN_BLOCK_Q", params.ATTN_BLOCK_Q);
      add("ATTN_BLOCK_KV", params.ATTN_BLOCK_KV);
      add("COOP_ACC_TYPE", params.COOP_ACC_TYPE);
      add("COOP_M_SIZE", params.COOP_M_SIZE);
      add("COOP_N_SIZE", params.COOP_N_SIZE);
      add("COOP_K_SIZE", params.COOP_K_SIZE);
      add("COOP_SUBGROUP_SIZE", params.COOP_SUBGROUP_SIZE);
      add("COOP_Q_TILES_PER_WORKGROUP", params.COOP_Q_TILES_PER_WORKGROUP);
      add("COOP_PV_N_SIZE", params.COOP_PV_N_SIZE);
      add("COOP_USE_NV2", params.COOP_USE_NV2);
    }
    else if(tunerName == "transformerDualGemmSwiGLU") {
      const auto& p = config.transformerDualGemmSwiGLU;
      add("MWARP", p.MWARP);
      add("NWARP", p.NWARP);
      add("KDIM", p.KDIM);
      add("subgroupSize", p.subgroupSize);
      add("accType", p.accType);
      add("MWG", p.MWG);
      add("NWG", p.NWG);
      add("KWG", p.KWG);
      add("MWAVE", p.MWAVE);
      add("NWAVE", p.NWAVE);
      add("SB", p.SB);
      add("STAGES", p.STAGES);
    }
    else if(tunerName == "transformerRMSNorm") {
      add("WG_C_SIZE", config.rmsNorm.WG_C_SIZE);
      add("WG_XY_SIZE", config.rmsNorm.WG_XY_SIZE);
      add("C_PER_THREAD", config.rmsNorm.C_PER_THREAD);
    }
    else if(tunerName == "transformerPreNorm") {
      add("MODE", config.transformerPreNorm.MODE);
      add("WG_C_SIZE", config.transformerPreNorm.WG_C_SIZE);
      add("WG_XY_SIZE", config.transformerPreNorm.WG_XY_SIZE);
      add("C_PER_THREAD", config.transformerPreNorm.C_PER_THREAD);
    }
    else if(tunerName == "spatialRMSNorm") {
      add("TILE_SIZE", config.spatialRMSNorm.TILE_SIZE);
      add("APPLY_ELTS_PER_THREAD", config.spatialRMSNorm.APPLY_ELTS_PER_THREAD);
    }
    return out.str();
  }

  void writeTuningLog(const TuningContext& context, const string& message) {
    if(context.logger != nullptr)
      context.logger->write(message);
    if(context.logger == nullptr || (!context.logger->isLoggingToStdout() && !context.logger->isLoggingToStderr()))
      cerr << message << endl;
  }

  const vector<string>& tuningNames() {
    static const vector<string> names = {
      "xgemmDirect", "transformerXgemm", "xgemm", "xgemm16", "hgemmCooperativeMatrix",
      "hgemmCooperativeMatrixNHWC", "hgemmCooperativeMatrixNCHW",
      "conv3x3InputTransform", "conv3x3OutputTransform",
      "conv5x5InputTransform", "conv5x5OutputTransform",
      "gPool", "pointwise", "addChannelBiases", "transformerScaleDotProduct",
      "transformerScaleDotProductCooperativeMatrixNCHW",
      "transformerScaleDotProductCooperativeMatrixNHWC", "transformerRMSNorm",
      "spatialRMSNorm", "transformerHgemmNHWC", "transformerDualGemmSwiGLU",
      "transformerPreNorm"
    };
    return names;
  }

  void initializeTuningRecords(map<string, VulkanTuneParams::TuningRecord>& records) {
    for(const string& name: tuningNames())
      records.emplace(name, VulkanTuneParams::TuningRecord());
  }

  bool tuningComplete(const string& name, const TuningContext& context) {
    if(context.tuningRecords == nullptr)
      return false;
    const auto found = context.tuningRecords->find(name);
    return found != context.tuningRecords->end() && found->second.completed;
  }

  void requireTuningDevice(const TuningContext& context, const string& work, const string& detail = "") {
    const char* stage = context.device == nullptr ? nullptr : context.device->deviceLossStage.load();
    if(stage != nullptr)
      throw std::runtime_error(
        "Vulkan tuning stopped on " + context.device->info.deviceName + ": VK_ERROR_DEVICE_LOST at " +
        stage + " while " + work + (detail.empty() ? "" : ": " + detail)
      );
  }

  void recordTuning(const string& name, const TuningContext& context) {
    requireTuningDevice(context, name);
    if(context.tuningRecords != nullptr)
      (*context.tuningRecords)[name].completed = true;
  }

  void logTuningStart(const TuningContext& context, const string& tunerName) {
    requireTuningDevice(context, tunerName);
    context.lastPrintedTuningParams.clear();
    writeTuningLog(context, "Tuning " + tunerName);
  }

  bool logTuningResult(
    const TuningContext& context,
    size_t candidateIndex,
    size_t totalCandidates,
    const VulkanTuneParams& candidate,
    const string& tunerName,
    double callsPerSecond
  ) {
    const string params = describeTuningParams(tunerName, candidate);
    if(params == context.lastPrintedTuningParams)
      return false;

    context.lastPrintedTuningParams = params;
    ostringstream out;
    out << "(" << candidateIndex << "/" << totalCandidates << ") " << params << " calls/sec=" << callsPerSecond;
    writeTuningLog(context, out.str());
    return true;
  }

  void logTuningFailure(
    const TuningContext& context,
    size_t candidateIndex,
    size_t totalCandidates,
    const string& error
  ) {
    writeTuningLog(
      context,
      "(" + to_string(candidateIndex) + "/" + to_string(totalCandidates) + ") failed: " + error
    );
  }

  void logTuningProgress(
    const TuningContext& context,
    size_t candidateIndex,
    size_t totalCandidates,
    bool isReference = false
  ) {
    writeTuningLog(
      context,
      "(" + to_string(candidateIndex) + "/" + to_string(totalCandidates) + ")" +
        (isReference ? " (reference)" : "") + " ..."
    );
  }

  void logTuningProgressIfNeeded(
    const TuningContext& context,
    size_t candidateIndex,
    size_t totalCandidates,
    bool loggedResult = false
  ) {
    if(!loggedResult && candidateIndex % 10 == 0)
      logTuningProgress(context, candidateIndex, totalCandidates);
  }

  // Keep each tuning submission short enough that a large model cannot turn a
  // complete warmup/measurement plan into one long-running GPU submission.
  constexpr size_t MAX_TUNER_RUNS_PER_SUBMISSION = 1;

  class VulkanTimestampTimer {
   public:
    explicit VulkanTimestampTimer(const VulkanDevice* device)
    : device(device), timestampPeriod(device->info.properties.limits.timestampPeriod) {
      VkResult result;
      commandPool = vk_helper::createCommandPool(device, &result);
      device->recordDeviceLoss(result, "tuning command pool creation");
      if(result != VK_SUCCESS)
        throw StringError("could not create tuning command pool: " + vk_helper::vkErrorToString(result));
    }

    ~VulkanTimestampTimer() {
      // Cached fixtures must release their command buffers before the pool.
      dualGemmResources.reset();
      vkDestroyCommandPool(device->device, commandPool, nullptr);
    }

    bool isUsable() const {
      return device->info.properties.limits.timestampComputeAndGraphics == VK_TRUE && timestampPeriod > 0.0f;
    }

    bool measure(
      const vector<const Pipeline*>& pipelines,
      const VulkanTuneParams& config,
      const TuningContext& context,
      const TuningMeasurementPlan& plan,
      double& callsPerSecond,
      vector<float>& readback,
      double& errorProp,
      string& error,
      vector<float>* cpuReference = nullptr
    ) {
      errorProp = numeric_limits<double>::quiet_NaN();
      if(!context.attentionSkipTimestamps && !isUsable()) {
        error = "compute timestamps are not supported";
        return false;
      }
      if(pipelines.empty()) {
        error = "no pipeline was created";
        return false;
      }

      // Keep multi-case GEMM measurements case-local. The shared multi-case
      // buffers use maximum dimensions, while these dispatches use each
      // case's own dimensions and strides.
      if(plan.gemmCases.size() > 1) {
        readback.clear();
        double weightedSeconds = 0.0;
        double totalWeight = 0.0;
        for(const GemmTuneCase& gemmCase: plan.gemmCases) {
          TuningMeasurementPlan casePlan = plan;
          casePlan.gemmCases = {gemmCase};
          casePlan.totalRuns = 3;
          casePlan.warmupRuns = 0;
          if(context.transformerPhysicalLayout != nullptr || plan.kernelName == "transformerXgemm") {
            casePlan.warmupRuns = plan.warmupRuns / plan.gemmCases.size();
            casePlan.totalRuns = casePlan.warmupRuns + plan.timedRuns() / plan.gemmCases.size();
          }
          casePlan.workloadWeights = {1.0};
          vector<float> output;
          double rate = 0.0;
          if(!measure(pipelines, config, context, casePlan, rate, output, errorProp, error, cpuReference))
            return false;
          readback.insert(readback.end(), output.begin(), output.end());
          const auto& nchw = config.hgemmCooperativeMatrixNCHW;
          const bool eligible = plan.kernelName != "hgemmCooperativeMatrixNCHW" ||
            (gemmCase.inChannels % nchw.KWG == 0 &&
             (gemmCase.padOutputForNCHW || gemmCase.outChannels % nchw.NWG == 0));
          if(eligible) {
            weightedSeconds += gemmCase.weight / rate;
            totalWeight += gemmCase.weight;
          }
        }
        if(totalWeight <= 0.0) {
          error = "no production layer is eligible for the selected cooperative tiles";
          return false;
        }
        callsPerSecond = totalWeight / weightedSeconds;
        return true;
      }
      const bool denseGemm = !plan.gemmCases.empty() &&
        plan.gemmCases.front().kind == GemmTuneCase::Kind::Dense;
      if(plan.kernelName == "xgemmDirect" && pipelines.size() > 1) {
        vector<const Pipeline*> selected;
        for(const Pipeline* pipeline: pipelines)
          if((pipeline->name.find("xgemm_direct_batched_tt") == 0) == denseGemm)
            selected.push_back(pipeline);
        return measure(
          selected,
          config,
          context,
          plan,
          callsPerSecond,
          readback,
          errorProp,
          error,
          cpuReference
        );
      }

      uint32_t descriptorCount = 0;
      for(const Pipeline* pipeline: pipelines)
        descriptorCount += pipeline->bindingCount;
      if(descriptorCount == 0) {
        error = "pipeline has no descriptor bindings";
        return false;
      }

      ReusableResources resources(device, commandPool);
      activeResources = &resources;
      const auto activeResourcesGuard = makeScopeGuard([&]() { activeResources = nullptr; });

      VkResult result = VK_SUCCESS;
      const auto deviceLossGuard = makeScopeGuard([&]() { device->recordDeviceLoss(result, "tuning measurement"); });
      const size_t batchSize = static_cast<size_t>(std::max(1, context.batchSize));
      // The probe can keep the full buffers while executing fewer rows.
      const size_t validatedBatchSize = context.attentionProbe
        ? static_cast<size_t>(plan.batchSizes.front()) : batchSize;
      const size_t logicalXYSize = static_cast<size_t>(std::max(1, context.nnXLen * context.nnYLen));
      const bool useNHWC = usesGenericNHWC(config);
      const bool modelSized = usesNonCooperativeNCHW(config);
      const vector<int> channelCases = modelSized ? getModelChannelCases(plan.kernelName, context.modelInfo) : vector<int>();
      const vector<int> swiGLUCases = modelSized && plan.kernelName == "pointwise"
        ? context.modelInfo.swiGLUChannels : vector<int>();
      const int trunkChannels = std::max(1, context.modelInfo.trunkNumChannels);
      const bool isGPool = plan.kernelName == "gPool";
      const bool isPointwise = plan.kernelName == "pointwise";
      const bool winogradInput = plan.kernelName.find("InputTransform") != string::npos;
      const bool winogradOutput = plan.kernelName.find("OutputTransform") != string::npos;
      const bool winogradTransform = winogradInput || winogradOutput;
      const bool spatialRMSNorm = plan.kernelName == "spatialRMSNorm";
      const int gpoolChannels = std::max(1, context.modelInfo.gpoolNumChannels);
      const int swiGLUChannels = std::max(trunkChannels, context.modelInfo.transformerFFNChannels);
      const auto maxCaseChannels = [](const vector<int>& cases, int fallback) {
        return cases.empty() ? fallback : *max_element(cases.begin(), cases.end());
      };
      const auto channelsForRun = [&](size_t repeat, int fallback) {
        return channelCases.empty() ? fallback :
          workloadCaseForSlot(channelCases, repeat % plan.workloadWeights.size(), fallback);
      };
      const auto swiGLUChannelsForRun = [&](size_t repeat) {
        return workloadCaseForSlot(swiGLUCases, repeat % plan.workloadWeights.size(), swiGLUChannels);
      };
      using AttentionShape = VulkanTuner::ModelInfoForTuning::AttentionShape;
      const AttentionShape defaultAttentionShape{
        std::max(1, context.modelInfo.transformerNumHeads), std::max(1, context.modelInfo.transformerNumKVHeads),
        std::max(1, context.modelInfo.transformerAttentionInChannels)
      };
      // Keep the same inputs on both sides of a scalar/cooperative comparison.
      const bool packedScalarAttention = plan.kernelName == "transformerScaleDotProduct" &&
        !comparesScalarAndCooperativeAttention(context, config);
      const bool tightScalarAttention = packedScalarAttention && context.tegraAttentionTightBuffers;
      const auto& attentionCases = context.modelInfo.attentionShapes;
      const size_t attentionCaseCount = packedScalarAttention ? std::max(size_t(1), attentionCases.size()) : 1;
      const auto attentionShapeAt = [&](size_t index) -> const AttentionShape& {
        return packedScalarAttention && !attentionCases.empty() ? attentionCases[index] : defaultAttentionShape;
      };
      const auto attentionCaseForRun = [&](size_t repeat) {
        const size_t slot = repeat % plan.workloadWeights.size();
        return (slot == 0 ? 0 : slot - 1) % attentionCaseCount;
      };
      const auto attentionHeadsForRun = [&](size_t repeat) {
        const AttentionShape& shape = attentionShapeAt(attentionCaseForRun(repeat));
        return make_pair(shape.heads, shape.kvHeads);
      };
      int maxAttentionHeads = defaultAttentionShape.heads;
      int maxAttentionKVHeads = defaultAttentionShape.kvHeads;
      for(size_t index = 0; index < attentionCaseCount; index++) {
        maxAttentionHeads = std::max(maxAttentionHeads, attentionShapeAt(index).heads);
        maxAttentionKVHeads = std::max(maxAttentionKVHeads, attentionShapeAt(index).kvHeads);
      }
      const bool nhwcCooperative = plan.kernelName == "hgemmCooperativeMatrixNHWC";
      const bool nhwcWinograd = useNHWC && winogradTransform;
      const auto& activeNhwcParams = nhwcTuningParams(config);
      const auto channelAlignment = [](const HGemmCooperativeMatrixNHWCTuneParams& params) {
        if(params.NWG <= 0)
          return 1;
        return params.KWG > 0 ? std::lcm(params.NWG, params.KWG) : params.NWG;
      };
      int nhwcSpatialAlignment = 1;
      const bool cooperativePadding = config.vulkan.shouldUseCooperativeMatrix &&
        config.vulkan.shouldUseFP16Storage && config.vulkan.shouldUseFP16Compute;
      if(useNHWC || cooperativePadding) {
        const auto addNhwcSpatialAlignment = [&](const auto& params) {
          if(params.MWG > 0)
            nhwcSpatialAlignment = std::lcm(nhwcSpatialAlignment, params.MWG);
        };
        addNhwcSpatialAlignment(context.transformerPhysicalLayout != nullptr
          ? *context.transformerPhysicalLayout : config.hgemmCooperativeMatrixNHWC);
      }
      if(config.vulkan.shouldUseHgemmCooperativeMatrixNCHW || plan.kernelName == "hgemmCooperativeMatrixNCHW")
        nhwcSpatialAlignment = std::lcm(nhwcSpatialAlignment,
          config.hgemmCooperativeMatrixNCHW.getRequiredSpatialAlignment());
      if(config.vulkan.shouldUseTransformerDualGemmSwiGLU)
        nhwcSpatialAlignment = std::lcm(nhwcSpatialAlignment,
          config.transformerDualGemmSwiGLU.getRequiredSpatialAlignment());
      int nhwcChannelAlignment = 1;
      if(useNHWC) {
        nhwcChannelAlignment = std::lcm(
          nhwcChannelAlignment,
          channelAlignment(context.transformerPhysicalLayout != nullptr
            ? *context.transformerPhysicalLayout : config.hgemmCooperativeMatrixNHWC)
        );
      }
      const size_t nhwcSpatialSize = vk_helper::roundUpToMultiple(
        logicalXYSize, static_cast<size_t>(nhwcSpatialAlignment)
      );
      const bool useNCHWCooperativeMatrix =
        config.vulkan.canUseCooperativeMatrix &&
        config.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
        config.hgemmCooperativeMatrixNCHW.isValid();
      const bool usePaddedNCHWXY =
        plan.kernelName == "hgemmCooperativeMatrixNCHW" || useNCHWCooperativeMatrix || cooperativePadding;
      const bool attentionUseFP16Storage = config.vulkan.canUseFP16Storage &&
        config.vulkan.canUseFP16Compute && config.vulkan.shouldUseFP16Storage;
      const size_t xySize = packedScalarAttention
        ? VulkanTuner::getPaddedNNXYLen(config, attentionUseFP16Storage, static_cast<int>(logicalXYSize))
        : (useNHWC || usePaddedNCHWXY ? nhwcSpatialSize : logicalXYSize);

      struct PackedAttentionLayout {
        int qOffset;
        int kOffset;
        int vOffset;
        int batchStride;
      };
      vector<PackedAttentionLayout> attentionLayouts;
      size_t attentionPackedElements = 0;
      if(packedScalarAttention) {
        const int qHeadDim = context.modelInfo.transformerHeadDim;
        const int vHeadDim = context.modelInfo.transformerVHeadDim;
        const size_t elementBytes = attentionUseFP16Storage ? sizeof(half_t) : sizeof(float);
        for(size_t index = 0; index < attentionCaseCount; index++) {
          const auto& shape = attentionShapeAt(index);
          const int qChannels = shape.heads * qHeadDim;
          const int kChannels = shape.kvHeads * qHeadDim;
          const int logicalChannels = qChannels + kChannels + shape.kvHeads * vHeadDim;
          const int physicalChannels = VulkanTuner::getPackedProjectionOutChannels(
            config, attentionUseFP16Storage, static_cast<int>(xySize), shape.inChannels, logicalChannels
          );
          const size_t batchStride = physicalChannels * xySize;
          const size_t nextOffset = attentionPackedElements + batchSize * batchStride;
          if(nextOffset > static_cast<size_t>(numeric_limits<int>::max()) ||
             vk_helper::roundUpToMultiple(nextOffset, size_t(2)) > device->info.properties.limits.maxStorageBufferRange / elementBytes) {
            error = "packed Scalar Attention input exceeds the device storage buffer range";
            return false;
          }
          attentionLayouts.push_back({
            static_cast<int>(attentionPackedElements),
            static_cast<int>(attentionPackedElements + qChannels * xySize),
            static_cast<int>(attentionPackedElements + (qChannels + kChannels) * xySize),
            static_cast<int>(batchStride)
          });
          attentionPackedElements = nextOffset;
        }
        attentionPackedElements = vk_helper::roundUpToMultiple(attentionPackedElements, size_t(2));
      }
      const bool addChannelBiasesUsingFP16Storage =
        config.vulkan.canUseFP16Storage &&
        config.vulkan.canUseFP16Compute &&
        config.vulkan.shouldUseFP16Storage;
      const bool addChannelBiasesUsingFP16TensorCoresFor1x1 =
        config.vulkan.canUseFP16Storage &&
        config.vulkan.canUseFP16Compute &&
        config.vulkan.canUseCooperativeMatrix &&
        config.vulkan.shouldUseFP16Storage &&
        config.vulkan.shouldUseFP16Compute &&
        config.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
        addChannelBiasesUsingFP16Storage;
      const size_t addChannelBiasesXYSize = addChannelBiasesUsingFP16TensorCoresFor1x1
        ? vk_helper::roundUpToMultiple(
            xySize, static_cast<size_t>(std::max(16, config.hgemmCooperativeMatrixNCHW.MWARP))
          )
        : xySize;
      const bool addChannelBiasesUseNHWC = useNHWC;
      const size_t addChannelBiasesSpatialSize = addChannelBiasesUseNHWC
        ? nhwcSpatialSize : addChannelBiasesXYSize;
      const size_t addChannelBiasesChannels = static_cast<size_t>(
        plan.kernelName == "addChannelBiases" ? maxCaseChannels(channelCases, trunkChannels) : trunkChannels
      );
      const size_t addChannelBiasesChannelsPadded = addChannelBiasesUseNHWC
        ? vk_helper::roundUpToMultiple(addChannelBiasesChannels, static_cast<size_t>(nhwcChannelAlignment))
        : addChannelBiasesChannels;
      const XgemmTuneParams& xgemmParams =
        config.vulkan.shouldUseFP16Compute ? config.xgemm16 : config.xgemm;
      const bool cooperativeForPadding = (nhwcCooperative || nhwcWinograd)
        && activeNhwcParams.isValid();
      const bool genericWinograd = !useNHWC && config.vulkan.shouldUseCooperativeMatrix && winogradTransform;
      const int activeMPaddingMult = genericWinograd ? config.hgemmCooperativeMatrix.MWG : cooperativeForPadding
        ? activeNhwcParams.MWG : xgemmParams.MWG;
      const int activeNPaddingMult = genericWinograd ? config.hgemmCooperativeMatrix.NWG : cooperativeForPadding
        ? activeNhwcParams.NWG : xgemmParams.NWG;
      const int activeKPaddingMult = genericWinograd ? config.hgemmCooperativeMatrix.KWG : cooperativeForPadding
        ? activeNhwcParams.KWG : xgemmParams.KWG;
      const size_t maxChannels = static_cast<size_t>(std::max({
        1, context.modelInfo.numInputChannels, context.modelInfo.trunkNumChannels, context.modelInfo.midNumChannels,
        context.modelInfo.regularNumChannels, context.modelInfo.maxConvChannels1x1,
        context.modelInfo.maxConvChannels3x3, context.modelInfo.gpoolNumChannels,
        context.modelInfo.transformerFFNChannels,
        maxAttentionHeads * context.modelInfo.transformerHeadDim,
        maxAttentionKVHeads * context.modelInfo.transformerVHeadDim,
        maxCaseChannels(channelCases, 1), maxCaseChannels(swiGLUCases, 1)
      }));
      const size_t maxChannelsPadded = vk_helper::roundUpToMultiple(
        maxChannels, static_cast<size_t>(nhwcChannelAlignment)
      );
      const size_t maxConvChannels = static_cast<size_t>(maxCaseChannels(channelCases, std::max({
        1, context.modelInfo.numInputChannels, context.modelInfo.trunkNumChannels, context.modelInfo.midNumChannels,
        context.modelInfo.regularNumChannels, context.modelInfo.maxConvChannels3x3,
        context.modelInfo.gpoolNumChannels
      })));
      const bool isGemm = !plan.gemmCases.empty();
      const bool winogradGemm = isGemm &&
        (plan.gemmCases.front().kind == GemmTuneCase::Kind::Winograd3x3 ||
         plan.gemmCases.front().kind == GemmTuneCase::Kind::Winograd5x5);
      const bool directGemm = plan.kernelName == "xgemmDirect" || plan.kernelName == "transformerXgemm" ||
        plan.kernelName == "hgemmCooperativeMatrixNCHW";
      const bool cooperative =
        plan.kernelName == "hgemmCooperativeMatrix" ||
        plan.kernelName == "hgemmCooperativeMatrixNHWC" ||
        plan.kernelName == "hgemmCooperativeMatrixNCHW";
      const bool openCLWinogradInputs = plan.kernelName == "hgemmCooperativeMatrix";
      const ConvTuneParams& transformConvParams =
        plan.kernelName.find("5x5") != string::npos ||
        (winogradGemm && plan.gemmCases.front().kind == GemmTuneCase::Kind::Winograd5x5)
          ? config.conv5x5 : config.conv3x3;
      const int tilesX = (context.nnXLen + transformConvParams.outTileXSize - 1) / transformConvParams.outTileXSize;
      const int tilesY = (context.nnYLen + transformConvParams.outTileYSize - 1) / transformConvParams.outTileYSize;
      const size_t transformTiles = batchSize * static_cast<size_t>(tilesX) * static_cast<size_t>(tilesY);
      const int logicalM = denseGemm ? static_cast<int>(batchSize) : winogradGemm
        ? static_cast<int>(batchSize) * tilesX * tilesY : nhwcCooperative
        ? (context.transformerPhysicalLayout != nullptr ? context.transformerPhysicalSpatialSize : static_cast<int>(logicalXYSize))
        : directGemm ? static_cast<int>(xySize) : static_cast<int>(batchSize) * tilesX * tilesY;
      const int logicalN = isGemm ? std::max(1, std::accumulate(
        plan.gemmCases.begin(), plan.gemmCases.end(), 0,
        [](int maximum, const GemmTuneCase& gemmCase) { return std::max(maximum, gemmCase.outChannels); }
      )) : static_cast<int>(maxChannels);
      const int logicalK = isGemm ? std::max(1, std::accumulate(
        plan.gemmCases.begin(), plan.gemmCases.end(), 0,
        [](int maximum, const GemmTuneCase& gemmCase) {
          return std::max(maximum, gemmCase.inChannels);
        }
      )) : static_cast<int>(maxChannels);
      const double openCLWinogradFilterScale = 1.0 / sqrt(9.0 * std::max(logicalK, logicalN));
      const int gemmBatch = denseGemm ? 1 : winogradGemm
        ? transformConvParams.inTileXSize * transformConvParams.inTileYSize : nhwcCooperative || directGemm
        ? static_cast<int>(batchSize)
        : config.conv3x3.inTileXSize * config.conv3x3.inTileYSize;
      struct GemmDimensions {
        int logicalM;
        int logicalN;
        int logicalK;
        int gemmM;
        int gemmN;
        int gemmK;
      };
      const auto getGemmDimensions = [&](int inChannels, int outChannels) {
        const int runLogicalK = inChannels;
        GemmDimensions dimensions = {
          logicalM, std::max(1, outChannels), std::max(1, runLogicalK), logicalM,
          std::max(1, outChannels), std::max(1, runLogicalK)
        };
        if(cooperative && directGemm) {
          dimensions.gemmM = vk_helper::roundUpToMultipleInt(
            dimensions.logicalM, config.hgemmCooperativeMatrixNCHW.getRequiredSpatialAlignment()
          );
          dimensions.gemmN = vk_helper::roundUpToMultipleInt(
            dimensions.logicalN, config.hgemmCooperativeMatrixNCHW.NWG
          );
          dimensions.gemmK = vk_helper::roundUpToMultipleInt(
            dimensions.logicalK, config.hgemmCooperativeMatrixNCHW.KWG
          );
        }
        else if(!directGemm) {
          dimensions.gemmM = vk_helper::roundUpToMultipleInt(
            dimensions.logicalM, cooperative ? (nhwcCooperative ? activeNhwcParams.MWG : config.hgemmCooperativeMatrix.MWG) : xgemmParams.MWG
          );
          dimensions.gemmN = vk_helper::roundUpToMultipleInt(
            dimensions.logicalN, cooperative ? (nhwcCooperative ? activeNhwcParams.NWG : config.hgemmCooperativeMatrix.NWG) : xgemmParams.NWG
          );
          dimensions.gemmK = vk_helper::roundUpToMultipleInt(
            dimensions.logicalK, cooperative ? (nhwcCooperative ? activeNhwcParams.KWG : config.hgemmCooperativeMatrix.KWG) : xgemmParams.KWG
          );
          if(nhwcCooperative && context.transformerPhysicalLayout == nullptr) {
            const int alignment = std::lcm(activeNhwcParams.NWG, activeNhwcParams.KWG);
            dimensions.gemmN = vk_helper::roundUpToMultipleInt(dimensions.logicalN, alignment);
            dimensions.gemmK = vk_helper::roundUpToMultipleInt(dimensions.logicalK, alignment);
          }
        }
        return dimensions;
      };
      const GemmDimensions maxGemm = getGemmDimensions(
        logicalK, logicalN
      );
      const int gemmM = maxGemm.gemmM;
      const int gemmN = maxGemm.gemmN;
      const int gemmK = maxGemm.gemmK;
      const size_t maxTiles = batchSize * ((context.nnXLen + 1) / 2) * ((context.nnYLen + 1) / 2);
      const size_t paddedTiles = vk_helper::roundUpToMultiple(
        maxTiles,
        static_cast<size_t>(activeMPaddingMult)
      );
      const size_t paddedTransformTiles = vk_helper::roundUpToMultiple(
        transformTiles, static_cast<size_t>(activeMPaddingMult)
      );
      const size_t paddedChannels = vk_helper::roundUpToMultiple(
        maxChannels, static_cast<size_t>(std::max(activeKPaddingMult, activeNPaddingMult))
      );
      const int gemmFilterBatches = directGemm || (nhwcCooperative && !winogradGemm) ? 1 : gemmBatch;
      const size_t gemmOutputElements = static_cast<size_t>(gemmBatch) * gemmM * gemmN;
      // Direct GEMM can access a vector SSBO entry even in its scalar edge path.
      const size_t gemmBindingElements[] = {
        vk_helper::roundUpToMultiple(static_cast<size_t>(gemmBatch) * gemmM * gemmK, size_t(4)),
        vk_helper::roundUpToMultiple(static_cast<size_t>(gemmFilterBatches) * gemmN * gemmK, size_t(4)),
        vk_helper::roundUpToMultiple(gemmOutputElements, size_t(4))
      };
      const bool transformerAttentionNHWC =
        isTransformerAttentionTuner(plan.kernelName) &&
        !pipelines.empty() &&
        pipelines[0]->name.find("_nhwc") != string::npos;
      const bool transformerAttentionCooperative =
        isTransformerAttentionTuner(plan.kernelName) &&
        !pipelines.empty() &&
        pipelines[0]->name.find("_cooperative") != string::npos;
      const size_t attentionOutputChannels = static_cast<size_t>(
        maxAttentionHeads * std::max(1, context.modelInfo.transformerVHeadDim)
      );
      const size_t attentionOutputElements = static_cast<size_t>(batchSize) * xySize *
        (transformerAttentionNHWC
          ? vk_helper::roundUpToMultiple(attentionOutputChannels, size_t(4))
          : attentionOutputChannels);
      const auto attentionOutputElementsForRun = [&](size_t repeat) {
        return attentionCases.empty() ? (attentionOutputElements / batchSize) * validatedBatchSize
          : validatedBatchSize * xySize *
          attentionHeadsForRun(repeat).first * std::max(1, context.modelInfo.transformerVHeadDim);
      };
      const size_t attentionInputChannels = static_cast<size_t>(
        (std::max(1, context.modelInfo.transformerNumHeads) +
         std::max(1, context.modelInfo.transformerNumKVHeads)) *
          std::max(1, context.modelInfo.transformerHeadDim) +
        std::max(1, context.modelInfo.transformerNumKVHeads) *
          std::max(1, context.modelInfo.transformerVHeadDim)
      );
      const size_t attentionInputRowStride = vk_helper::roundUpToMultiple(attentionInputChannels, size_t(4));
      const size_t winogradElements = paddedTransformTiles *
        std::max(vk_helper::roundUpToMultiple(maxConvChannels, static_cast<size_t>(activeKPaddingMult)),
                 vk_helper::roundUpToMultiple(maxConvChannels, static_cast<size_t>(activeNPaddingMult))) *
        transformConvParams.inTileXSize * transformConvParams.inTileYSize;
      const size_t transformElements = std::max({
        static_cast<size_t>(batchSize) * maxChannelsPadded * xySize,
        paddedTiles * paddedChannels * 36,
        winogradElements,
        batchSize * static_cast<size_t>(maxCaseChannels(channelCases, gpoolChannels)) * 3,
        attentionOutputElements,
        transformerAttentionNHWC ? static_cast<size_t>(batchSize) * xySize * attentionInputRowStride : size_t(0)
      });
      const size_t winogradChannelAlignment = winogradInput ? activeKPaddingMult : activeNPaddingMult;
      const auto winogradPaddedChannels = [&](size_t channels) {
        return vk_helper::roundUpToMultiple(channels, winogradChannelAlignment);
      };
      const auto winogradTransformedElements = [&](size_t channels) {
        return paddedTransformTiles * winogradPaddedChannels(channels) *
          transformConvParams.inTileXSize * transformConvParams.inTileYSize;
      };
      const auto winogradSpatialElements = [&](size_t channels) {
        const size_t spatialStride = !useNHWC && (winogradInput || modelSized) ? logicalXYSize : xySize;
        return batchSize * spatialStride * (useNHWC ? winogradPaddedChannels(channels) : channels);
      };
      const size_t maxGpoolChannels = static_cast<size_t>(maxCaseChannels(channelCases, gpoolChannels));
      const size_t gpoolInputStride = useNHWC
        ? vk_helper::roundUpToMultiple(maxGpoolChannels, static_cast<size_t>(nhwcChannelAlignment))
        : maxGpoolChannels;
      const size_t gpoolBindingElements[] = {
        batchSize * xySize * gpoolInputStride, // input
        batchSize * maxGpoolChannels * 3,     // output
        batchSize * xySize,                   // mask
        batchSize                            // mask sums
      };
      const size_t addChannelBiasesElements = batchSize *
        addChannelBiasesChannelsPadded * addChannelBiasesSpatialSize;
      const size_t scratchElements = plan.kernelName == "addChannelBiases" ? addChannelBiasesElements : transformElements;
      const size_t scratchBytes = vk_helper::roundUpToMultiple(
        std::max<size_t>(scratchElements, 4),
        size_t(4)
      ) * sizeof(float);
      // Distinct production layers have distinct filters. Keep the activation
      // buffers shared, as in the runtime workspace, but rotate filter addresses.
      const uint32_t filterCopies = (context.transformerPhysicalLayout != nullptr && nhwcCooperative) ||
        plan.kernelName == "transformerXgemm"
        ? static_cast<uint32_t>(std::max(1.0, plan.gemmCases.front().weight)) : 1;
      vector<VulkanBuffer*>& tuningBuffers = resources.tuningBuffers;
      VulkanBuffer*& pointwiseAccumulatorInitialBuffer = resources.pointwiseAccumulatorInitialBuffer;
      VkDeviceSize pointwiseAccumulatorBytes = 0;
      vector<VulkanBuffer*>& pointwiseValidationBuffers = resources.pointwiseValidationBuffers;
      vector<VulkanBuffer*>& winogradValidationBuffers = resources.winogradValidationBuffers;
      vector<VulkanBuffer*>& gpoolValidationBuffers = resources.gpoolValidationBuffers;
      vector<VulkanBuffer*>& attentionValidationBuffers = resources.attentionValidationBuffers;
      vector<VulkanBuffer*>& spatialValidationBuffers = resources.spatialValidationBuffers;
      vector<VulkanBuffer*>& gemmValidationBuffers = resources.gemmValidationBuffers;
      VkDescriptorPool& descriptorPool = resources.descriptorPool;
      VkQueryPool& queryPool = resources.queryPool;
      VkFence& fence = resources.fence;
      VkCommandBuffer& commandBuffer = resources.commandBuffer;
      bool commandBufferSubmitted = false;
      const auto cleanup = [&]() noexcept {
        if(commandBuffer != VK_NULL_HANDLE && !commandBufferSubmitted)
          vkResetCommandBuffer(commandBuffer, 0);
      };
      const auto cleanupGuard = makeScopeGuard(cleanup);

      const auto outputBinding = [](const Pipeline* pipeline) -> uint32_t {
        const string& name = pipeline->name;
        if(name.find("gemm") != string::npos || name.find("transformer_swiglu") == 0 ||
           name.find("transformer_spatial_rms_norm_sum_sq") == 0)
          return 2;
        if(name.find("transformer_scale_dot_product") == 0)
          return 3;
        if(name.find("add_pointwise") == 0 || name.find("add_channel_bias_") == 0)
          return 0;
        return 1;
      };
      const auto halfBinding = [&](const Pipeline* pipeline, uint32_t binding) {
        const string& name = pipeline->name;
        if(name.find("xgemm_direct_batched_tt") == 0)
          return false;
        if(name.find("hgemm_cooperative_matrix") == 0)
          return true;
        if(name.find("fp32") != string::npos)
          return false;
        if(name.find("global_pooling_channels") == 0)
          return binding == 0 || binding == 2;
        if(name.find("value_head_pool_channels") == 0 || name.find("sum_channels") == 0)
          return binding == 0;
        if(name.find("add_channel_bias_") == 0)
          return binding == 0;
        if(name.find("transformer_rms_norm") == 0 || name.find("transformer_spatial_rms_norm_apply") == 0)
          return binding == 0 || binding == 1 || binding == 4;
        if(name.find("transformer_spatial_rms_norm_sum_sq") == 0)
          return binding < 2;
        if(name.find("transformer_spatial_rms_norm_reduce") == 0)
          return false;
        if(name.find("transformer_scale_dot_product") == 0 && binding >= 5)
          return false;
        return config.vulkan.shouldUseFP16Storage;
      };
      const size_t gemmOutputBytes = isGemm ? gemmOutputElements *
        (halfBinding(pipelines[0], outputBinding(pipelines[0])) ? sizeof(half_t) : sizeof(float)) : 0;
      const size_t filterBytes = filterCopies > 1 ? gemmBindingElements[1] *
        (halfBinding(pipelines[0], 1) ? sizeof(half_t) : sizeof(float)) : 0;
      const size_t filterStride = vk_helper::roundUpToMultiple(
        filterBytes, static_cast<size_t>(std::max(VkDeviceSize(1), device->info.properties.limits.minStorageBufferOffsetAlignment))
      );
      const auto winogradValidationElements = [&](size_t repeat) {
        const size_t channels = channelsForRun(repeat, static_cast<int>(maxConvChannels));
        return winogradInput ? winogradTransformedElements(channels) : winogradSpatialElements(channels);
      };
      const auto bindingSourceIndex = [&](const Pipeline* pipeline, size_t index, size_t binding) {
        if((transformerAttentionNHWC || packedScalarAttention) && binding < 3)
          return index - binding;
        if(spatialRMSNorm) {
          // SumSq owns the input, mask and partials. Reduce owns the final sum.
          if(pipeline == pipelines[1] && binding == 0)
            return size_t(2);
          if(pipeline == pipelines[2]) {
            if(binding == 0 || binding == 4)
              return size_t(binding == 0 ? 0 : 1);
            if(binding == 6)
              return size_t(4);
          }
        }
        return index;
      };
      const auto pointwiseElements = [&](size_t channels) {
        return batchSize * xySize * vk_helper::roundUpToMultiple(channels, static_cast<size_t>(nhwcChannelAlignment));
      };
      const auto pointwiseRunElements = [&](const Pipeline* pipeline, size_t repeat) {
        const int channels = pipeline->name.find("transformer_swiglu") == 0
          ? swiGLUChannelsForRun(repeat) : channelsForRun(repeat, trunkChannels);
        return pointwiseElements(channels);
      };
      const auto pointwiseOutputBytes = [&](const Pipeline* pipeline, size_t repeat) {
        return pointwiseRunElements(pipeline, repeat) *
          (halfBinding(pipeline, outputBinding(pipeline)) ? sizeof(half_t) : sizeof(float));
      };
      vector<float> gemmInput, gemmFilter, gemmScale, gemmBias;
      bool gemmOutputIsHalf = false;
      const bool needsHostFloatBuffers = !isGemm && cpuReference != nullptr;
      vector<vector<float>> hostFloatBuffers;
      if(needsHostFloatBuffers)
        hostFloatBuffers.reserve(descriptorCount);
      struct PendingUpload {
        VulkanBuffer* stagingBuffer;
        VulkanBuffer* destinationBuffer;
        VkDeviceSize size;
      };
      vector<PendingUpload> pendingUploads;
      size_t stagingBufferIndex = 0;
      const auto queueUpload = [&](
        const void* data,
        VkDeviceSize size,
        VulkanBuffer* destination,
        const string& description
      ) {
        if(stagingBufferIndex == resources.uploadStagingBuffers.size())
          resources.uploadStagingBuffers.push_back(nullptr);
        VulkanBuffer*& stagingBuffer = resources.uploadStagingBuffers[stagingBufferIndex++];
        if(!ensureStagingBuffer(stagingBuffer, size, result, error, description))
          return false;
        memcpy(stagingBuffer->allocationInfo.pMappedData, data, static_cast<size_t>(size));
        result = vmaFlushAllocation(
          device->allocator,
          stagingBuffer->allocation,
          0,
          stagingBuffer->requestedSize
        );
        if(result != VK_SUCCESS) {
          error = "could not prepare " + description + ": " + vk_helper::vkErrorToString(result);
          return false;
        }
        pendingUploads.push_back({stagingBuffer, destination, size});
        return true;
      };
      const auto fillPaddedNCHWInput = [&](
        vector<float>& data,
        size_t channels,
        Rand& rand
      ) {
        const size_t channelStride = vk_helper::roundUpToMultiple(
          channels, static_cast<size_t>(nhwcChannelAlignment)
        );
        for(size_t n = 0; n < batchSize; n++)
          for(size_t c = 0; c < channels; c++)
            for(size_t xy = 0; xy < logicalXYSize; xy++)
              data[useNHWC
                ? (n * xySize + xy) * channelStride + c
                : (n * channels + c) * xySize + xy] = static_cast<float>(rand.nextDouble());
      };
      const auto fillTransformerRopeTable = [&](vector<float>& data, bool sine) {
        const int headDim = std::max(1, context.modelInfo.transformerHeadDim);
        const int numPairs = headDim / 2;
        const int numKVHeads = maxAttentionKVHeads;
        const int numPairsPerDim = numPairs / 2;
        const int dimHalf = std::max(1, numPairs);
        const bool learnable = context.modelInfo.transformerLearnableRope;
        for(int tableHead = 0; tableHead < (learnable ? numKVHeads : 1); tableHead++) {
          for(int pairIdx = 0; pairIdx < numPairs; pairIdx++) {
            float freqX = 0.0f;
            float freqY = 0.0f;
            if(learnable) {
              const size_t freqIndex = static_cast<size_t>(tableHead * numPairs + pairIdx) * 2;
              if(freqIndex + 1 < context.modelInfo.transformerRopeFreqs.size()) {
                freqX = context.modelInfo.transformerRopeFreqs[freqIndex];
                freqY = context.modelInfo.transformerRopeFreqs[freqIndex + 1];
              }
            }
            else {
              const int frequencyPair = pairIdx < numPairsPerDim ? pairIdx : pairIdx - numPairsPerDim;
              const float theta = context.modelInfo.transformerRopeTheta;
              const float frequency = theta > 0.0f
                ? 1.0f / powf(theta, static_cast<float>(2 * frequencyPair) / static_cast<float>(dimHalf))
                : 0.0f;
              if(pairIdx < numPairsPerDim)
                freqY = frequency;
              else
                freqX = frequency;
            }
            const int tableIndex = tableHead * numPairs + pairIdx;
            for(int y = 0; y < context.nnYLen; y++) {
              for(int x = 0; x < context.nnXLen; x++) {
                const int xy = y * context.nnXLen + x;
                const float angle = static_cast<float>(x) * freqX + static_cast<float>(y) * freqY;
                const size_t index = static_cast<size_t>(tableIndex) * xySize + xy;
                if(index < data.size())
                  data[index] = sine ? sinf(angle) : cosf(angle);
              }
            }
          }
        }
      };
      const auto fillPackedScalarAttention = [&](vector<float>& data, bool halfStorage, Rand& rand) {
        const int qHeadDim = context.modelInfo.transformerHeadDim;
        const int vHeadDim = context.modelInfo.transformerVHeadDim;
        const int ropePairs = qHeadDim / 2;
        vector<float> cosTable;
        vector<float> sinTable;
        if(context.modelInfo.transformerUseRope) {
          const size_t ropeHeads = context.modelInfo.transformerLearnableRope ? maxAttentionKVHeads : 1;
          cosTable.resize(ropeHeads * ropePairs * xySize);
          sinTable.resize(cosTable.size());
          fillTransformerRopeTable(cosTable, false);
          fillTransformerRopeTable(sinTable, true);
        }
        const auto toStorage = [halfStorage](float value) {
          return halfStorage ? half_float::half_cast<float>(half_float::half_cast<half_t>(value)) : value;
        };
        for(size_t index = 0; index < attentionLayouts.size(); index++) {
          const auto& shape = attentionShapeAt(index);
          const auto& layout = attentionLayouts[index];
          const int channels = (shape.heads + shape.kvHeads) * qHeadDim + shape.kvHeads * vHeadDim;
          for(size_t n = 0; n < batchSize; n++) {
            const size_t batchBase = n * layout.batchStride;
            for(int c = 0; c < channels; c++)
              for(size_t xy = 0; xy < logicalXYSize; xy++)
                data[batchBase + layout.qOffset + c * xySize + xy] = toStorage(static_cast<float>(rand.nextDouble()));

            if(cosTable.empty())
              continue;
            const auto rotate = [&](int offset, int heads) {
              for(int head = 0; head < heads; head++) {
                const int tableHead = context.modelInfo.transformerLearnableRope ? head * shape.kvHeads / heads : 0;
                for(int pair = 0; pair < ropePairs; pair++) {
                  const size_t channelBase = batchBase + offset + (head * qHeadDim + 2 * pair) * xySize;
                  const size_t tableBase = (tableHead * ropePairs + pair) * xySize;
                  for(size_t xy = 0; xy < logicalXYSize; xy++) {
                    const float a = data[channelBase + xy];
                    const float b = data[channelBase + xySize + xy];
                    const float c = cosTable[tableBase + xy];
                    const float s = sinTable[tableBase + xy];
                    data[channelBase + xy] = toStorage(a * c - b * s);
                    data[channelBase + xySize + xy] = toStorage(a * s + b * c);
                  }
                }
              }
            };
            rotate(layout.qOffset, shape.heads);
            rotate(layout.kOffset, shape.kvHeads);
          }
        }
      };
      size_t tuningBufferIndex = 0;
      for(const Pipeline* pipeline: pipelines) {
        for(uint32_t binding = 0; binding < pipeline->bindingCount; binding++) {
          const bool isOutput = binding == outputBinding(pipeline);
          const bool useHalf = halfBinding(pipeline, binding);
          if(isGemm && isOutput)
            gemmOutputIsHalf = useHalf;
          if(bindingSourceIndex(pipeline, tuningBufferIndex, binding) != tuningBufferIndex) {
            tuningBuffers.push_back(nullptr);
            tuningBufferIndex++;
            if(needsHostFloatBuffers)
              hostFloatBuffers.emplace_back();
            continue;
          }
          const bool packedInput = packedScalarAttention && binding == 0;
          const bool upload = !(isGPool || isGemm) || !isOutput;
          size_t elements = scratchBytes / sizeof(float);
          if(isGPool)
            elements = gpoolBindingElements[binding];
          else if(isGemm)
            elements = gemmBindingElements[binding];
          else if(packedInput)
            elements = attentionPackedElements;
          else if(tightScalarAttention)
            // Packed Q/K already have RoPE. The shader still binds both table descriptors.
            elements = isOutput ? attentionOutputElements : binding == 4 ? batchSize * xySize : 1;
          else if(isPointwise) {
            const int channels = pipeline->name.find("transformer_swiglu") == 0
              ? maxCaseChannels(swiGLUCases, swiGLUChannels) : maxCaseChannels(channelCases, trunkChannels);
            elements = pointwiseElements(channels);
          }
          else if(winogradTransform)
            elements = (binding == 0) == winogradInput ? winogradSpatialElements(maxConvChannels)
              : winogradTransformedElements(maxConvChannels);
          vector<float> data(upload ? elements : 0, 0.0f);
          Rand rand(openCLWinogradInputs && binding < 2
            ? string(binding == 0 ? "tuneHGemmWmma3x3Input" : "tuneHGemmWmma3x3Filter")
            : "VulkanTunerInput:" + to_string(binding));
          if(!isOutput) {
            if(packedInput)
              fillPackedScalarAttention(data, useHalf, rand);
            else if(isGemm && binding < 2) {
              const int width = binding == 0 ? gemmM : gemmN;
              const int logicalWidth = binding == 0 ? logicalM : logicalN;
              const int batches = binding == 1 ? gemmFilterBatches : gemmBatch;
              for(int n = 0; n < batches; n++)
                for(int k = 0; k < logicalK; k++)
                  for(int x = 0; x < logicalWidth; x++)
                    data[denseGemm
                      ? x * gemmK + k
                      : nhwcCooperative && binding == 0
                      ? (static_cast<size_t>(n) * width + x) * gemmK + k
                      : (static_cast<size_t>(n) * gemmK + k) * width + x] =
                      openCLWinogradInputs
                        ? static_cast<float>(rand.nextDouble(binding == 0 ? 1.0 : openCLWinogradFilterScale))
                        : static_cast<float>(rand.nextDouble() - 0.5) / sqrtf(static_cast<float>(logicalK));
              if(cpuReference != nullptr) {
                vector<float>& referenceInput = binding == 0 ? gemmInput : gemmFilter;
                referenceInput = data;
                if(useHalf) {
                  for(float& value: referenceInput)
                    value = half_float::half_cast<float>(half_float::half_cast<half_t>(value));
                }
              }
            }
            else {
              const string& name = pipeline->name;
              const bool addPointwiseInput =
                plan.kernelName == "pointwise" && name.find("add_pointwise") == 0 && binding == 1;
              const bool swigluInput =
                plan.kernelName == "pointwise" && name.find("transformer_swiglu") == 0 && binding < 2;
              const bool addChannelBiasesInput =
                plan.kernelName == "addChannelBiases" && name.find("add_channel_bias_") == 0 && binding == 1;
              const bool transformerAttentionInput =
                isTransformerAttentionTuner(plan.kernelName) &&
                name.find("transformer_scale_dot_product") == 0 && binding < 3;
              const bool transformerAttentionRopeTable =
                isTransformerAttentionTuner(plan.kernelName) &&
                name.find("transformer_scale_dot_product") == 0 && (binding == 5 || binding == 6);
              const bool transformerRMSNormInput =
                plan.kernelName == "transformerRMSNorm" && name.find("transformer_rms_norm") == 0 && binding == 0;
              const bool transformerRMSNormGamma =
                plan.kernelName == "transformerRMSNorm" && name.find("transformer_rms_norm") == 0 && binding == 2;
              const bool spatialRMSNormInput =
                plan.kernelName == "spatialRMSNorm" && binding == 0 &&
                (name.find("transformer_spatial_rms_norm_sum_sq") == 0 ||
                 name.find("transformer_spatial_rms_norm_apply") == 0);
              const bool spatialRMSNormGamma =
                plan.kernelName == "spatialRMSNorm" &&
                name.find("transformer_spatial_rms_norm_apply") == 0 && binding == 2;
              const bool spatialRMSNormBeta =
                plan.kernelName == "spatialRMSNorm" &&
                name.find("transformer_spatial_rms_norm_apply") == 0 && binding == 3;
              if(winogradInput) {
                const size_t inputChannelsPadded = winogradPaddedChannels(maxConvChannels);
                for(size_t n = 0; n < batchSize; n++)
                  for(size_t c = 0; c < maxConvChannels; c++)
                    for(size_t xy = 0; xy < logicalXYSize; xy++)
                      data[useNHWC
                        ? (n * xySize + xy) * inputChannelsPadded + c
                        : (n * maxConvChannels + c) * logicalXYSize + xy] =
                        static_cast<float>(rand.nextDouble());
              }
              else if(winogradOutput) {
                const size_t tileElements = static_cast<size_t>(transformConvParams.inTileXSize) * transformConvParams.inTileYSize;
                const size_t outputChannelsPadded = winogradPaddedChannels(maxConvChannels);
                for(size_t tileElement = 0; tileElement < tileElements; tileElement++)
                  for(size_t channel = 0; channel < maxConvChannels; channel++)
                    for(size_t tile = 0; tile < transformTiles; tile++)
                      data[useNHWC
                        ? (tileElement * paddedTransformTiles + tile) * outputChannelsPadded + channel
                        : (tileElement * outputChannelsPadded + channel) * paddedTransformTiles + tile] =
                        static_cast<float>(rand.nextDouble());
              }
              else if(isGPool) {
                if(binding == 0) {
                  for(size_t n = 0; n < batchSize; n++)
                    for(size_t c = 0; c < maxGpoolChannels; c++)
                      for(size_t xy = 0; xy < logicalXYSize; xy++)
                        data[useNHWC
                          ? (n * xySize + xy) * gpoolInputStride + c
                          : (n * gpoolInputStride + c) * xySize + xy] =
                          static_cast<float>(rand.nextDouble());
                }
                else if(binding == 2) {
                  for(size_t n = 0; n < batchSize; n++)
                    std::fill_n(data.begin() + n * xySize, logicalXYSize, 1.0f);
                }
                else if(binding == 3)
                  std::fill(data.begin(), data.end(), static_cast<float>(logicalXYSize));
              }
              else if(addPointwiseInput || swigluInput) {
                const size_t channels = addPointwiseInput
                  ? maxCaseChannels(channelCases, trunkChannels) : maxCaseChannels(swiGLUCases, swiGLUChannels);
                fillPaddedNCHWInput(data, channels, rand);
              }
              else if(
                plan.kernelName == "addChannelBiases" &&
                      name.find("add_channel_bias_") == 0 && binding == 0
              ) {
                if(addChannelBiasesUseNHWC) {
                  for(size_t n = 0; n < batchSize; n++)
                    for(size_t xy = 0; xy < logicalXYSize; xy++)
                      for(size_t c = 0; c < addChannelBiasesChannels; c++)
                        data[(n * addChannelBiasesSpatialSize + xy) * addChannelBiasesChannelsPadded + c] =
                          static_cast<float>(rand.nextDouble());
                }
                else {
                  fillPaddedNCHWInput(data, addChannelBiasesChannels, rand);
                }
              }
              else if(addChannelBiasesInput) {
                const size_t validBiases = batchSize * addChannelBiasesChannels;
                for(size_t i = 0; i < validBiases; i++)
                  data[i] = static_cast<float>(rand.nextDouble());
              }
              else if(transformerAttentionRopeTable) {
                if(!packedScalarAttention && context.modelInfo.transformerUseRope)
                  fillTransformerRopeTable(data, binding == 6);
              }
              else if(transformerAttentionInput) {
                const size_t heads = static_cast<size_t>(maxAttentionHeads);
                const size_t kvHeads = static_cast<size_t>(maxAttentionKVHeads);
                const size_t headDim = static_cast<size_t>(std::max(1, context.modelInfo.transformerHeadDim));
                const size_t vHeadDim = static_cast<size_t>(std::max(1, context.modelInfo.transformerVHeadDim));
                const size_t channels = binding == 0 ? heads : kvHeads;
                const size_t dimension = binding == 2 ? vHeadDim : headDim;
                if(transformerAttentionNHWC) {
                  for(size_t n = 0; n < batchSize; n++)
                    for(size_t xy = 0; xy < xySize; xy++)
                      for(size_t c = 0; c < attentionInputChannels; c++)
                        data[(n * xySize + xy) * attentionInputRowStride + c] = static_cast<float>(rand.nextDouble());
                }
                else
                  fillPaddedNCHWInput(data, channels * dimension, rand);
              }
              else if(transformerRMSNormInput) {
                const size_t channels = maxCaseChannels(channelCases, trunkChannels);
                fillPaddedNCHWInput(data, channels, rand);
              }
              else if(transformerRMSNormGamma) {
                const size_t channels = maxCaseChannels(channelCases, trunkChannels);
                for(size_t i = 0; i < channels; i++)
                  data[i] = static_cast<float>(rand.nextDouble());
              }
              else if(spatialRMSNormInput) {
                const size_t channels = static_cast<size_t>(std::max(1, context.modelInfo.trunkNumChannels));
                fillPaddedNCHWInput(data, channels, rand);
              }
              else if(spatialRMSNormGamma || spatialRMSNormBeta) {
                const size_t channels = static_cast<size_t>(std::max(1, context.modelInfo.trunkNumChannels));
                for(size_t i = 0; i < channels; i++)
                  data[i] = static_cast<float>(rand.nextDouble());
              }
              else {
                for(float& value: data)
                  value = static_cast<float>(rand.nextDouble());
                // Masks and their sums describe a fully valid board.
                const bool mask =
                  (name.find("transformer_scale_dot_product") == 0 && binding == 4) ||
                  ((name.find("transformer_rms_norm") == 0 || name.find("transformer_spatial_rms_norm_apply") == 0) && binding == 4) ||
                  (name.find("transformer_spatial_rms_norm_sum_sq") == 0 && binding == 1);
                const bool maskSum =
                  (name.find("value_head_pool_channels") == 0 && binding == 2) ||
                  (name.find("transformer_spatial_rms_norm_apply") == 0 && binding == 5);
                if(mask) {
                  std::fill(data.begin(), data.end(), 0.0f);
                  const int maskBatchSize = std::max(1, context.batchSize);
                  for(int n = 0; n < maskBatchSize; n++)
                    std::fill(data.begin() + static_cast<size_t>(n) * xySize,
                              data.begin() + static_cast<size_t>(n) * xySize + logicalXYSize, 1.0f);
                  if(isTransformerAttentionTuner(plan.kernelName) && context.attentionValidationMask != 0) {
                    for(int n = 0; n < maskBatchSize; n++)
                      for(size_t xy = 0; xy < logicalXYSize; xy++) {
                        const bool valid = context.attentionValidationMask == 1 ? false :
                          context.attentionValidationMask == 2 ? xy % 5 != 0 : xy < logicalXYSize / 2;
                        data[static_cast<size_t>(n) * xySize + xy] = valid ? 1.0f : 0.0f;
                      }
                  }
                }
                else if(maskSum) {
                  std::fill(data.begin(), data.end(), 0.0f);
                  std::fill(data.begin(), data.begin() + batchSize, static_cast<float>(logicalXYSize));
                }
                else if(name.find("transformer_rms_norm") == 0 && binding == 3)
                  std::fill(data.begin(), data.end(), 0.0f);
              }
            }
          }
          vector<half_t> halfData;
          const void* initialData = data.data();
          size_t initialBytes = data.size() * sizeof(float);
          if(useHalf) {
            halfData.resize(data.size());
            for(size_t j = 0; j < data.size(); j++)
              halfData[j] = half_float::half_cast<half_t>(data[j]);
            initialData = halfData.data();
            initialBytes = halfData.size() * sizeof(half_t);
          }
          if(tuningBufferIndex == tuningBuffers.size())
            tuningBuffers.push_back(nullptr);
          size_t bufferBytes = scratchBytes;
          if(isGPool || isGemm)
            bufferBytes = elements * (useHalf ? sizeof(half_t) : sizeof(float));
          else if(packedInput || tightScalarAttention || isPointwise || winogradTransform)
            bufferBytes = initialBytes;
          if(binding == 1 && filterCopies > 1)
            bufferBytes += (filterCopies - 1) * filterStride;
          if(!ensureBuffer(tuningBuffers[tuningBufferIndex], bufferBytes, result, error, "tuning buffer"))
            return false;
          VulkanBuffer* buffer = tuningBuffers[tuningBufferIndex++];
          if(upload && !queueUpload(initialData, initialBytes, buffer, "tuning buffer"))
            return false;

          // In-place add pipelines write binding 0. Keep an immutable copy of
          // their initialized input so every measured invocation has the same
          // input, as in the OpenCL tuner.
          if((pipeline->name.find("add_pointwise") == 0 ||
              pipeline->name.find("add_channel_bias_") == 0) && binding == 0) {
            if(!ensureBuffer(pointwiseAccumulatorInitialBuffer, bufferBytes, result, error, "pointwise reset buffer"))
              return false;
            if(!queueUpload(initialData, initialBytes, pointwiseAccumulatorInitialBuffer, "pointwise reset buffer"))
              return false;
            pointwiseAccumulatorBytes = initialBytes;
          }
          if(needsHostFloatBuffers)
            hostFloatBuffers.push_back(std::move(data));
        }
      }
      if(isGemm && cpuReference != nullptr) {
        // Match the actual storage conversion so the validation error measures
        // the GEMM itself instead of charging every candidate for FP16 I/O rounding.
        for(const GemmTuneCase& gemmCase: plan.gemmCases) {
          const GemmDimensions dimensions = getGemmDimensions(gemmCase.inChannels, gemmCase.outChannels);
          for(int n = 0; n < gemmBatch; n++) {
            const auto appendValue = [&](int x, int y) {
              double sum = 0.0;
              for(int k = 0; k < dimensions.logicalK; k++)
                sum += static_cast<double>(gemmInput[denseGemm ? x * dimensions.gemmK + k : nhwcCooperative
                  ? (static_cast<size_t>(n) * dimensions.gemmM + x) * dimensions.gemmK + k
                  : (static_cast<size_t>(n) * dimensions.gemmK + k) * dimensions.gemmM + x
                ]) * gemmFilter[
                  denseGemm ? y * dimensions.gemmK + k :
                  (static_cast<size_t>((directGemm || (nhwcCooperative && !winogradGemm)) ? 0 : n) * dimensions.gemmK + k) * dimensions.gemmN + y
                ];
              float resultValue = static_cast<float>(sum);
              if(gemmOutputIsHalf)
                resultValue = half_float::half_cast<float>(half_float::half_cast<half_t>(resultValue));
              cpuReference->push_back(resultValue);
            };
            if(nhwcCooperative || denseGemm) {
              for(int x = 0; x < dimensions.logicalM; x++)
                for(int y = 0; y < dimensions.logicalN; y++)
                  appendValue(x, y);
            }
            else {
              for(int y = 0; y < dimensions.logicalN; y++)
                for(int x = 0; x < dimensions.logicalM; x++)
                  appendValue(x, y);
            }
          }
        }
      }

      const auto appendCpuReference = [&](const Pipeline* pipeline, size_t firstBuffer, size_t repeat) -> bool {
        const string& name = pipeline->name;
        const auto buffer = [&](size_t binding) -> const vector<float>& {
          return hostFloatBuffers[bindingSourceIndex(pipeline, firstBuffer + binding, binding)];
        };
        const int cpuBatchSize = static_cast<int>(validatedBatchSize);
        const int cpuXYSize = static_cast<int>(xySize);
        const int cpuChannels = channelsForRun(repeat, trunkChannels);
        const int cpuChannelsPadded = vk_helper::roundUpToMultipleInt(cpuChannels, nhwcChannelAlignment);

        if(name.find("transformer_spatial_rms_norm_sum_sq") == 0 ||
           name.find("transformer_spatial_rms_norm_reduce") == 0)
          return true;

        if(name.find("global_pooling_channels") == 0) {
          const vector<float>& input = buffer(0);
          const vector<float>& mask = buffer(2);
          const vector<float>& maskSum = buffer(3);
          const int channels = channelsForRun(repeat, gpoolChannels);
          for(int n = 0; n < cpuBatchSize; n++) {
            const float divisor = maskSum[n];
            vector<float> means(channels);
            vector<float> maxima(channels);
            for(int c = 0; c < channels; c++) {
              float sum = 0.0f;
              float maximum = -1.0f;
              for(int xy = 0; xy < cpuXYSize; xy++) {
                const int channelsPadded = vk_helper::roundUpToMultipleInt(channels, nhwcChannelAlignment);
                const float value = input[useNHWC
                  ? (n * cpuXYSize + xy) * channelsPadded + c
                  : (n * channels + c) * cpuXYSize + xy];
                sum += value;
                maximum = std::max(maximum, value + mask[n * cpuXYSize + xy] - 1.0f);
              }
              means[c] = sum / divisor;
              maxima[c] = maximum;
            }
            for(int c = 0; c < channels; c++) cpuReference->push_back(means[c]);
              for(int c = 0; c < channels; c++) cpuReference->push_back(means[c] * (sqrtf(divisor) - 14.0f) * 0.1f);
                for(int c = 0; c < channels; c++) cpuReference->push_back(maxima[c]);
              }
              return true;
        }
        if(name.find("value_head_pool_channels") == 0) {
          const vector<float>& input = buffer(0);
          const vector<float>& maskSum = buffer(2);
          const int channels = std::max(1, context.modelInfo.gpoolNumChannels);
          for(int n = 0; n < cpuBatchSize; n++) {
            const float divisor = maskSum[n];
            vector<float> means(channels);
            for(int c = 0; c < channels; c++) {
              float sum = 0.0f;
              for(int xy = 0; xy < cpuXYSize; xy++)
                sum += input[useNHWC
                  ? (n * cpuXYSize + xy) * vk_helper::roundUpToMultipleInt(channels, nhwcChannelAlignment) + c
                  : (n * channels + c) * cpuXYSize + xy];
              means[c] = sum / divisor;
            }
            const float scale = (sqrtf(divisor) - 14.0f) * 0.1f;
            for(int c = 0; c < channels; c++) cpuReference->push_back(means[c]);
              for(int c = 0; c < channels; c++) cpuReference->push_back(means[c] * scale);
                for(int c = 0; c < channels; c++) cpuReference->push_back(means[c] * (scale * scale - 0.1f));
              }
              return true;
        }
        if(name.find("sum_channels") == 0) {
          const vector<float>& input = buffer(0);
          for(int n = 0; n < cpuBatchSize; n++) {
            float sum = 0.0f;
            for(int xy = 0; xy < cpuXYSize; xy++)
              sum += input[n * cpuXYSize + xy];
            cpuReference->push_back(sum);
          }
          return true;
        }
        if(name.find("add_pointwise") == 0) {
          const vector<float>& accum = buffer(0);
          const vector<float>& value = buffer(1);
          const size_t count = static_cast<size_t>(cpuBatchSize) * cpuChannelsPadded * cpuXYSize;
          for(size_t i = 0; i < count; i++)
            cpuReference->push_back(accum[i] + value[i]);
          return true;
        }
        if(name.find("transformer_swiglu") == 0) {
          const vector<float>& input = buffer(0);
          const vector<float>& gate = buffer(1);
          const size_t count = static_cast<size_t>(cpuBatchSize) *
            vk_helper::roundUpToMultiple(
              static_cast<size_t>(swiGLUChannelsForRun(repeat)),
              static_cast<size_t>(nhwcChannelAlignment)
            ) * cpuXYSize;
          for(size_t i = 0; i < count; i++)
            cpuReference->push_back(input[i] / (1.0f + expf(-input[i])) * gate[i]);
          return true;
        }
        if(name.find("add_channel_bias_") == 0) {
          const vector<float>& accum = buffer(0);
          const vector<float>& bias = buffer(1);
          if(addChannelBiasesUseNHWC) {
            for(int n = 0; n < cpuBatchSize; n++)
              for(size_t xy = 0; xy < addChannelBiasesSpatialSize; xy++)
                for(size_t c = 0; c < addChannelBiasesChannelsPadded; c++) {
                  const size_t index =
                    (static_cast<size_t>(n) * addChannelBiasesSpatialSize + xy) * addChannelBiasesChannelsPadded + c;
                  const float result = xy < logicalXYSize && c < addChannelBiasesChannels
                    ? accum[index] + bias[static_cast<size_t>(n) * addChannelBiasesChannels + c]
                    : 0.0f;
                  cpuReference->push_back(result);
                }
          }
          else {
            const size_t count = static_cast<size_t>(cpuBatchSize) * cpuChannels * addChannelBiasesXYSize;
            for(size_t i = 0; i < count; i++)
              cpuReference->push_back(accum[i] + bias[i / addChannelBiasesXYSize]);
          }
          return true;
        }
        if(name.find("transformer_scale_dot_product") == 0) {
          const vector<float>& query = buffer(0);
          const vector<float>& key = buffer(1);
          const vector<float>& value = buffer(2);
          const PackedAttentionLayout* layout = packedScalarAttention ? &attentionLayouts[attentionCaseForRun(repeat)] : nullptr;
          const vector<float>& mask = buffer(4);
          const auto [heads, kvHeads] = attentionHeadsForRun(repeat);
          const int headDim = std::max(1, context.modelInfo.transformerHeadDim);
          const int vHeadDim = std::max(1, context.modelInfo.transformerVHeadDim);
          const float scale = 1.0f / sqrtf(static_cast<float>(headDim));
          const bool useRope = !packedScalarAttention && context.modelInfo.transformerUseRope;
          const bool learnableRope = context.modelInfo.transformerLearnableRope;
          const int ropeNumPairs = headDim / 2;
          const vector<float>& ropeCosTable = buffer(5);
          const vector<float>& ropeSinTable = buffer(6);
          const auto ropeTableIndex = [&](
            int tableHead,
            int pairIdx,
            int position
          ) {
            return static_cast<size_t>(learnableRope ? tableHead * ropeNumPairs + pairIdx : pairIdx) *
              xySize + position;
          };
          const int qChannels = heads * headDim;
          const int kChannels = kvHeads * headDim;
          const int outputChannelsPadded = vk_helper::roundUpToMultipleInt(heads * vHeadDim, 4);
          vector<float> nhwcOutput;
          if(transformerAttentionNHWC)
            nhwcOutput.resize(static_cast<size_t>(cpuBatchSize) * xySize * outputChannelsPadded, 0.0f);
          vector<float> rotatedKey(headDim);
          for(int bh = 0; bh < cpuBatchSize * heads; bh++) {
            const int n = bh / heads;
            const int head = bh % heads;
            const int kvBase = n * kvHeads + (bh % heads) / (heads / kvHeads);
            const int kvHead = (bh % heads) / (heads / kvHeads);
            const size_t packedBase = layout != nullptr ? static_cast<size_t>(n) * layout->batchStride : 0;
            const size_t qBase = layout != nullptr ? packedBase + layout->qOffset + head * headDim * xySize
              : static_cast<size_t>(bh) * headDim * cpuXYSize;
            const size_t kBase = layout != nullptr ? packedBase + layout->kOffset + kvHead * headDim * xySize
              : static_cast<size_t>(kvBase) * headDim * cpuXYSize;
            const size_t vBase = layout != nullptr ? packedBase + layout->vOffset + kvHead * vHeadDim * xySize
              : static_cast<size_t>(kvBase) * vHeadDim * cpuXYSize;
            const int qRopeTableHead = (bh % heads) * kvHeads / heads;
            const int kRopeTableHead = kvBase % kvHeads;
            vector<float> output(vHeadDim * cpuXYSize, 0.0f);
            for(int qPos = 0; qPos < cpuXYSize; qPos++) {
              if(mask[n * cpuXYSize + qPos] == 0.0f) {
                continue;
              }
              vector<float> rotatedQuery(headDim);
              for(int d = 0; d < headDim; d++)
                rotatedQuery[d] = transformerAttentionNHWC
                  ? query[(static_cast<size_t>(n) * xySize + qPos) * attentionInputRowStride + head * headDim + d]
                  : query[qBase + d * cpuXYSize + qPos];
              if(useRope) {
                for(int pairIdx = 0; pairIdx < ropeNumPairs; pairIdx++) {
                  const size_t tableIndex = ropeTableIndex(qRopeTableHead, pairIdx, qPos);
                  const float q0 = rotatedQuery[2 * pairIdx];
                  const float q1 = rotatedQuery[2 * pairIdx + 1];
                  const float cosVal = ropeCosTable[tableIndex];
                  const float sinVal = ropeSinTable[tableIndex];
                  rotatedQuery[2 * pairIdx] = q0 * cosVal - q1 * sinVal;
                  rotatedQuery[2 * pairIdx + 1] = q0 * sinVal + q1 * cosVal;
                }
              }
              float runningMax = -1e30f;
              float runningSum = 0.0f;
              vector<float> accum(vHeadDim, 0.0f);
              for(int kPos = 0; kPos < cpuXYSize; kPos++) {
                if(mask[n * cpuXYSize + kPos] == 0.0f)
                  continue;
                for(int d = 0; d < headDim; d++)
                  rotatedKey[d] = transformerAttentionNHWC
                    ? key[(static_cast<size_t>(n) * xySize + kPos) * attentionInputRowStride + qChannels + kvHead * headDim + d]
                    : key[kBase + d * cpuXYSize + kPos];
                if(useRope) {
                  for(int pairIdx = 0; pairIdx < ropeNumPairs; pairIdx++) {
                    const size_t tableIndex = ropeTableIndex(kRopeTableHead, pairIdx, kPos);
                    const float k0 = rotatedKey[2 * pairIdx];
                    const float k1 = rotatedKey[2 * pairIdx + 1];
                    const float cosVal = ropeCosTable[tableIndex];
                    const float sinVal = ropeSinTable[tableIndex];
                    rotatedKey[2 * pairIdx] = k0 * cosVal - k1 * sinVal;
                    rotatedKey[2 * pairIdx + 1] = k0 * sinVal + k1 * cosVal;
                  }
                }
                float dot = 0.0f;
                for(int d = 0; d < headDim; d++)
                  dot += rotatedQuery[d] * rotatedKey[d];
                dot *= scale;
                const float nextMax = std::max(runningMax, dot);
                const float oldWeight = expf(runningMax - nextMax);
                const float weight = expf(dot - nextMax);
                for(int d = 0; d < vHeadDim; d++)
                  accum[d] = accum[d] * oldWeight + weight * (
                    transformerAttentionNHWC
                      ? value[(static_cast<size_t>(n) * xySize + kPos) * attentionInputRowStride + qChannels + kChannels + kvHead * vHeadDim + d]
                      : value[vBase + d * cpuXYSize + kPos]
                  );
                runningSum = runningSum * oldWeight + weight;
                runningMax = nextMax;
              }
              for(int d = 0; d < vHeadDim; d++)
                if(transformerAttentionNHWC)
                  nhwcOutput[(static_cast<size_t>(n) * xySize + qPos) * outputChannelsPadded + head * vHeadDim + d] =
                    runningSum > 0.0f ? accum[d] / runningSum : 0.0f;
                else
                  output[d * cpuXYSize + qPos] = runningSum > 0.0f ? accum[d] / runningSum : 0.0f;
            }
            if(!transformerAttentionNHWC)
              cpuReference->insert(cpuReference->end(), output.begin(), output.end());
          }
          if(transformerAttentionNHWC)
            cpuReference->insert(cpuReference->end(), nhwcOutput.begin(), nhwcOutput.end());
          return true;
        }
        if(name.find("transformer_rms_norm") == 0) {
          const vector<float>& input = buffer(0);
          const vector<float>& gamma = buffer(2);
          const vector<float>& beta = buffer(3);
          const vector<float>& mask = buffer(4);
          vector<float> nhwcOutput;
          if(useNHWC)
            nhwcOutput.resize(static_cast<size_t>(cpuBatchSize) * cpuXYSize * cpuChannelsPadded, 0.0f);
          for(int n = 0; n < cpuBatchSize; n++) {
            vector<float> rms(cpuXYSize);
            for(int xy = 0; xy < cpuXYSize; xy++) {
              const float maskValue = mask[n * cpuXYSize + xy];
              float sumSq = 0.0f;
              for(int c = 0; c < cpuChannels; c++) {
                const float value = input[useNHWC
                  ? (n * cpuXYSize + xy) * cpuChannelsPadded + c
                  : (n * cpuChannels + c) * cpuXYSize + xy] * maskValue;
                sumSq += value * value;
              }
              rms[xy] = 1.0f / sqrtf(sumSq / cpuChannels + 1e-6f);
            }
            for(int c = 0; c < cpuChannels; c++)
              for(int xy = 0; xy < cpuXYSize; xy++)
                if(useNHWC)
                  nhwcOutput[(n * cpuXYSize + xy) * cpuChannelsPadded + c] =
                    (input[(n * cpuXYSize + xy) * cpuChannelsPadded + c] * rms[xy] * gamma[c] + beta[c]) *
                    mask[n * cpuXYSize + xy];
                else
                  cpuReference->push_back((input[(n * cpuChannels + c) * cpuXYSize + xy] * rms[xy] * gamma[c] + beta[c]) *
                    mask[n * cpuXYSize + xy]);
          }
          if(useNHWC)
            cpuReference->insert(cpuReference->end(), nhwcOutput.begin(), nhwcOutput.end());
          return true;
        }
        if(name.find("transformer_spatial_rms_norm_apply") == 0) {
          const vector<float>& input = buffer(0);
          const vector<float>& gamma = buffer(2);
          const vector<float>& beta = buffer(3);
          const vector<float>& mask = buffer(4);
          const vector<float>& maskSum = buffer(5);
          vector<float> nhwcOutput;
          if(useNHWC)
            nhwcOutput.resize(static_cast<size_t>(cpuBatchSize) * cpuXYSize * cpuChannelsPadded, 0.0f);
          for(int n = 0; n < cpuBatchSize; n++) {
            float sumSq = 0.0f;
            for(int c = 0; c < cpuChannels; c++)
              for(int xy = 0; xy < cpuXYSize; xy++) {
                const float value = input[useNHWC
                  ? (n * cpuXYSize + xy) * cpuChannelsPadded + c
                  : (n * cpuChannels + c) * cpuXYSize + xy] * mask[n * cpuXYSize + xy];
                sumSq += value * value;
              }
            const float rms = 1.0f / sqrtf(sumSq / (maskSum[n] * cpuChannels) + 1e-6f);
            for(int c = 0; c < cpuChannels; c++)
              for(int xy = 0; xy < cpuXYSize; xy++)
                if(useNHWC)
                  nhwcOutput[(n * cpuXYSize + xy) * cpuChannelsPadded + c] =
                    (input[(n * cpuXYSize + xy) * cpuChannelsPadded + c] * rms * gamma[c] + beta[c]) *
                    mask[n * cpuXYSize + xy];
                else
                  cpuReference->push_back((input[(n * cpuChannels + c) * cpuXYSize + xy] * rms * gamma[c] + beta[c]) *
                    mask[n * cpuXYSize + xy]);
          }
          if(useNHWC)
            cpuReference->insert(cpuReference->end(), nhwcOutput.begin(), nhwcOutput.end());
          return true;
        }
        error = "no CPU reference implementation for " + name;
        return false;
      };
      if(!isGemm && cpuReference != nullptr) {
        const int numValidationRepeats =
          context.attentionSkipSnapshots ? 1 :
          plan.kernelName == "gPool" ? 10 :
          isTransformerAttentionTuner(plan.kernelName) ? 6 :
          plan.kernelName == "pointwise" || plan.kernelName == "addChannelBiases" ||
          plan.kernelName == "transformerRMSNorm" ||
          plan.kernelName == "spatialRMSNorm" ? 10 : 1;
        map<tuple<int, int, size_t>, pair<size_t, size_t>> referenceByShape;
        const size_t firstValidationRepeat = context.attentionSkipSnapshots ? plan.totalRuns - 1 : plan.warmupRuns;
        for(size_t repeat = firstValidationRepeat; repeat < firstValidationRepeat + numValidationRepeats; repeat++) {
          const auto dimensions = isTransformerAttentionTuner(plan.kernelName) ? attentionHeadsForRun(repeat)
            : make_pair(channelsForRun(repeat, trunkChannels), swiGLUChannelsForRun(repeat));
          const auto shape = make_tuple(dimensions.first, dimensions.second, packedScalarAttention ? attentionCaseForRun(repeat) : 0);
          const size_t start = cpuReference->size();
          const auto existing = referenceByShape.find(shape);
          if(existing != referenceByShape.end()) {
            const auto& range = existing->second;
            cpuReference->resize(start + range.second);
            std::copy_n(cpuReference->begin() + range.first, range.second, cpuReference->begin() + start);
            continue;
          }
          size_t firstBuffer = 0;
          for(const Pipeline* pipeline: pipelines) {
            if(!appendCpuReference(pipeline, firstBuffer, repeat))
              return false;
            firstBuffer += pipeline->bindingCount;
          }
          referenceByShape.emplace(shape, make_pair(start, cpuReference->size() - start));
        }
      }

      if(isPointwise) {
        pointwiseValidationBuffers.resize(10 * pipelines.size(), nullptr);
        for(size_t repeat = 0; repeat < 10; repeat++) {
          for(size_t i = 0; i < pipelines.size(); i++) {
            const size_t bytes = pointwiseOutputBytes(pipelines[i], plan.warmupRuns + repeat);
            if(!ensureBuffer(pointwiseValidationBuffers[repeat * pipelines.size() + i], bytes, result, error,
               "pointwise validation buffer"))
              return false;
          }
        }
      }
      else if(plan.kernelName == "addChannelBiases" || plan.kernelName == "transformerRMSNorm") {
        if(!ensureBuffers(pointwiseValidationBuffers, 10, scratchBytes, result, error, "pointwise validation buffer"))
          return false;
      }
      else if(spatialRMSNorm) {
        if(!ensureBuffers(spatialValidationBuffers, 10, scratchBytes, result, error, "spatial RMSNorm validation buffer"))
          return false;
      }
      else if(winogradTransform) {
        const Pipeline* pipeline = pipelines[0];
        const size_t elementBytes = halfBinding(pipeline, outputBinding(pipeline)) ? sizeof(half_t) : sizeof(float);
        winogradValidationBuffers.resize(10, nullptr);
        for(size_t i = 0; i < winogradValidationBuffers.size(); i++) {
          const size_t bytes = winogradValidationElements(plan.warmupRuns + i) * elementBytes;
          if(!ensureBuffer(winogradValidationBuffers[i], bytes, result, error, "Winograd validation buffer"))
            return false;
        }
      }
      else if(isGPool) {
        if(!ensureBuffers(gpoolValidationBuffers, 10, gpoolBindingElements[1] * sizeof(float), result, error,
           "global pooling validation buffer"))
          return false;
      }
      else if(isTransformerAttentionTuner(plan.kernelName)) {
        const size_t validationBytes = tightScalarAttention
          ? attentionOutputElements * (halfBinding(pipelines[0], outputBinding(pipelines[0])) ? sizeof(half_t) : sizeof(float))
          : scratchBytes;
        if(!ensureBuffers(attentionValidationBuffers, 6, validationBytes, result, error, "attention validation buffer"))
          return false;
      }
      if(context.attentionProbe && context.logger != nullptr) {
        const auto bufferBytes = [](const vector<VulkanBuffer*>& buffers) {
          VkDeviceSize bytes = 0;
          for(const VulkanBuffer* buffer: buffers)
            if(buffer != nullptr)
              bytes += buffer->requestedSize;
          return bytes;
        };
        context.logger->write(
          "Attention probe buffers: device=" +
          to_string(bufferBytes(tuningBuffers) + bufferBytes(attentionValidationBuffers)) +
          " B, upload=" + to_string(bufferBytes(resources.uploadStagingBuffers)) + " B"
        );
      }
      else if(isGemm) {
        if(!ensureBuffers(gemmValidationBuffers, plan.gemmCases.size(), gemmOutputBytes, result, error,
           "GEMM validation buffer"))
          return false;
      }

      if(!ensureDescriptorPool(
        descriptorCount * filterCopies, static_cast<uint32_t>(pipelines.size()) * filterCopies, result, error
      ))
        return false;

      vector<VkDescriptorSet> descriptorSets;
      descriptorSets.reserve(pipelines.size() * filterCopies);
      size_t bufferIndex = 0;
      for(size_t setIndex = 0; setIndex < pipelines.size() * filterCopies; setIndex++) {
        const size_t pipelineIndex = setIndex % pipelines.size();
        if(pipelineIndex == 0)
          bufferIndex = 0;
        const Pipeline* pipeline = pipelines[pipelineIndex];
        VkDescriptorSetAllocateInfo descriptorSetInfo = {};
        descriptorSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetInfo.descriptorPool = descriptorPool;
        descriptorSetInfo.descriptorSetCount = 1;
        descriptorSetInfo.pSetLayouts = &pipeline->descriptorSetLayout;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        result = vkAllocateDescriptorSets(device->device, &descriptorSetInfo, &descriptorSet);
        if(result != VK_SUCCESS) {
          error = "could not allocate tuning descriptor set: " + vk_helper::vkErrorToString(result);
          cleanup();
          return false;
        }
        vector<WriteDescriptorSet> writes;
        writes.reserve(pipeline->bindingCount);
        for(uint32_t binding = 0; binding < pipeline->bindingCount; binding++) {
          VulkanBuffer* buffer = tuningBuffers[bindingSourceIndex(pipeline, bufferIndex, binding)];
          auto write = vk_helper::writeDescriptorSetBuffer(descriptorSet, binding, buffer);
          if(binding == 1 && filterCopies > 1) {
            write.bufferInfo.offset = (setIndex / pipelines.size()) * filterStride;
            write.bufferInfo.range = filterBytes;
          }
          writes.push_back(write);
          bufferIndex++;
        }
        result = vk_helper::updateDescriptorSets(device, writes);
        if(result != VK_SUCCESS) {
          error = "could not update tuning descriptor set: " + vk_helper::vkErrorToString(result);
          cleanup();
          return false;
        }
        descriptorSets.push_back(descriptorSet);
      }

      const size_t firstTimedPipeline = plan.kernelName == "spatialRMSNorm" ? 2 : 0;
      const size_t timedPipelineCount = plan.kernelName == "spatialRMSNorm" ? 1 : pipelines.size();
      if(!context.attentionSkipTimestamps && !ensureQueryPool(
        static_cast<uint32_t>(2 * plan.timedRuns() * timedPipelineCount), result, error
      ))
        return false;
      if(!ensureCommandResources(result, error))
        return false;

      uint32_t dispatchX = 0;
      uint32_t dispatchY = 0;
      uint32_t dispatchZ = 0;
      const auto recordPipeline = [&](
        VkCommandBuffer targetCommandBuffer,
        const Pipeline* pipeline,
        VkDescriptorSet descriptorSet,
        int runBatchSize,
        int runInChannels,
        int runOutChannels,
        int runChannels,
        int runGemmM,
        int runGemmN,
        int runGemmK,
        size_t repeat
      ) {
        const int batchSize = std::max(1, runBatchSize);
        const int logicalPipelineXYSize = std::max(1, context.nnXLen * context.nnYLen);
        const bool usesPaddedPipelineXY =
          plan.kernelName == "gPool" || plan.kernelName == "pointwise" ||
          isTransformerAttentionTuner(plan.kernelName) || plan.kernelName == "transformerRMSNorm" ||
          plan.kernelName == "spatialRMSNorm" || nhwcWinograd;
        const int pipelineXYSize = usesPaddedPipelineXY ? static_cast<int>(xySize) : logicalPipelineXYSize;
        const int channels = std::max(1, runChannels);
        const auto dispatch = [&](
          uint32_t x,
          uint32_t y = 1,
          uint32_t z = 1
        ) {
          dispatchX = std::max(1u, x);
          dispatchY = std::max(1u, y);
          dispatchZ = std::max(1u, z);
          const auto& limits = device->info.properties.limits;
          if(dispatchX > limits.maxComputeWorkGroupCount[0] ||
             dispatchY > limits.maxComputeWorkGroupCount[1] ||
             dispatchZ > limits.maxComputeWorkGroupCount[2])
            throw StringError("Vulkan tuning dispatch exceeds the device workgroup count limit: " + pipeline->name);
          vkCmdDispatch(targetCommandBuffer, dispatchX, dispatchY, dispatchZ);
        };
        const auto push = [&](const auto& params) {
          vkCmdPushConstants(
            targetCommandBuffer,
            pipeline->layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(params),
            &params
          );
        };
        vkCmdBindPipeline(targetCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
        vkCmdBindDescriptorSets(
          targetCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr
        );

        if(pipeline->name.find("hgemm_cooperative_matrix_nchw") == 0) {
          vk_shader::push::HGemmCooperativeMatrixNCHWParams params = {runGemmK, runGemmM, runGemmN};
          push(params);
          dispatch(
            (runGemmM + config.hgemmCooperativeMatrixNCHW.MWG - 1) / config.hgemmCooperativeMatrixNCHW.MWG,
            (runGemmN + config.hgemmCooperativeMatrixNCHW.NWG - 1) / config.hgemmCooperativeMatrixNCHW.NWG,
            gemmBatch
          );
        }
        else if(pipeline->name.find("hgemm_cooperative_matrix_nhwc_") == 0) {
          vk_shader::push::HGemmCooperativeMatrixNHWCParams params = {
            runGemmM, runGemmN, runGemmK, runGemmK, runGemmN,
            runGemmM * runGemmK, winogradGemm ? runGemmK * runGemmN : 0, runGemmM * runGemmN
          };
          push(params);
          dispatch(runGemmM / activeNhwcParams.MWG, runGemmN / activeNhwcParams.NWG, gemmBatch);
        }
        else if(pipeline->name.find("hgemm_cooperative_matrix_") == 0) {
          vk_shader::push::HGemmCooperativeMatrixParams params = {runGemmM, runGemmN, runGemmK};
          push(params);
          dispatch(
            runGemmM / config.hgemmCooperativeMatrix.MWG,
            runGemmN / config.hgemmCooperativeMatrix.NWG,
            gemmBatch
          );
        }
        else if(pipeline->name.find("xgemm_batched") == 0) {
          vk_shader::push::XGEMMBatchedParams params = {
            static_cast<uint32_t>(runGemmM), static_cast<uint32_t>(runGemmN), static_cast<uint32_t>(runGemmK),
            static_cast<uint32_t>(runGemmM), static_cast<uint32_t>(runGemmK),
            static_cast<uint32_t>(runGemmN), static_cast<uint32_t>(runGemmK),
            static_cast<uint32_t>(runGemmM), static_cast<uint32_t>(runGemmN)
          };
          push(params);
          dispatch(runGemmM / xgemmParams.MWG, runGemmN / xgemmParams.NWG, gemmBatch);
        }
        else if(pipeline->name.find("xgemm_direct_batched_tt") == 0) {
          vk_shader::push::BatchedXgemmDirectFp32Params params{};
          params.kSizeM = runGemmM;
          params.kSizeN = runGemmN;
          params.kSizeK = runGemmK;
          params.aLead = runGemmK;
          params.bLead = runGemmK;
          params.cLead = runGemmN;
          params.cTranspose = 1;
          push(params);
          dispatch((runGemmM + config.xgemmDirect.WGD - 1) / config.xgemmDirect.WGD,
                   (runGemmN + config.xgemmDirect.WGD - 1) / config.xgemmDirect.WGD, 1);
        }
        else if(pipeline->name.find("xgemm_strided_batched") == 0 ||
                pipeline->name.find("transformer_xgemm") == 0) {
          const auto& directParams = plan.kernelName == "transformerXgemm"
            ? config.transformerXgemm : config.xgemmDirect;
          vk_shader::push::XgemmStridedBatchedFp32Params params = {
            static_cast<uint32_t>(runGemmM), static_cast<uint32_t>(runGemmN), static_cast<uint32_t>(runGemmK),
            static_cast<uint32_t>(runGemmM), static_cast<uint32_t>(runGemmM * runGemmK),
            static_cast<uint32_t>(runGemmN), 0,
            static_cast<uint32_t>(runGemmM), static_cast<uint32_t>(runGemmM * runGemmN), 0
          };
          push(params);
          dispatch(
            (runGemmM + directParams.WGD - 1) / directParams.WGD,
            (runGemmN + directParams.WGD - 1) / directParams.WGD, gemmBatch
          );
        }
        else if(pipeline->name.find("winograd_input_transform") == 0) {
          const int paddedTiles = static_cast<int>(paddedTransformTiles);
          const int paddedChannels = static_cast<int>(winogradPaddedChannels(channels));
          vk_shader::push::WinogradInputTransformParams params = {
            batchSize,context.nnXLen,context.nnYLen,tilesX,tilesY,channels,paddedChannels,paddedTiles,pipelineXYSize
          };
          push(params);
          if(useNHWC)
            dispatch(
              static_cast<uint32_t>((paddedChannels + pipeline->localSizeX - 1) / pipeline->localSizeX),
              static_cast<uint32_t>((params.ntxtySizePadded + pipeline->localSizeY - 1) / pipeline->localSizeY)
            );
          else
            dispatch(
              static_cast<uint32_t>((params.ntxtySizePadded + pipeline->localSizeX - 1) / pipeline->localSizeX),
              static_cast<uint32_t>((paddedChannels + pipeline->localSizeY - 1) / pipeline->localSizeY)
            );
        }
        else if(pipeline->name.find("winograd_output_transform") == 0) {
          const int paddedTiles = static_cast<int>(paddedTransformTiles);
          const int paddedChannels = static_cast<int>(winogradPaddedChannels(channels));
          vk_shader::push::WinogradOutputTransformParams params = {
            batchSize,context.nnYLen,context.nnXLen,tilesY,tilesX,channels,paddedChannels,paddedTiles,pipelineXYSize
          };
          push(params);
          if(useNHWC)
            dispatch(
              static_cast<uint32_t>((paddedChannels + pipeline->localSizeX - 1) / pipeline->localSizeX),
              static_cast<uint32_t>((batchSize * tilesX * tilesY + pipeline->localSizeY - 1) / pipeline->localSizeY)
            );
          else
            dispatch(
              static_cast<uint32_t>((vk_helper::powerOf2ify(tilesX) + pipeline->localSizeX - 1) / pipeline->localSizeX),
              static_cast<uint32_t>((vk_helper::powerOf2ify(tilesY) + pipeline->localSizeY - 1) / pipeline->localSizeY),
              static_cast<uint32_t>((batchSize * channels + pipeline->localSizeZ - 1) / pipeline->localSizeZ)
            );
        }
        else if(pipeline->name.find("global_pooling_channels") == 0) {
          const int gpoolChannelsPadded = vk_helper::roundUpToMultipleInt(
            channels,
            nhwcChannelAlignment
          );
          vk_shader::push::GlobalPoolingChannelsParams params = {
            batchSize, channels, pipelineXYSize, pipelineXYSize,
            useNHWC ? gpoolChannelsPadded : channels
          };
          push(params);
          dispatch(
            1,
            static_cast<uint32_t>((channels + pipeline->localSizeY - 1) / pipeline->localSizeY),
            static_cast<uint32_t>((batchSize + pipeline->localSizeZ - 1) / pipeline->localSizeZ)
          );
        }
        else if(pipeline->name.find("value_head_pool_channels") == 0) {
          const int gpoolChannels = std::max(1, context.modelInfo.gpoolNumChannels);
          const int gpoolChannelsPadded = vk_helper::roundUpToMultipleInt(
            gpoolChannels,
            nhwcChannelAlignment
          );
          vk_shader::push::ValueHeadPoolingChannelsParams params = {
            batchSize, gpoolChannels, pipelineXYSize,
            useNHWC ? gpoolChannelsPadded : gpoolChannels
          };
          push(params);
          dispatch(
            1,
            static_cast<uint32_t>((gpoolChannels + pipeline->localSizeY - 1) / pipeline->localSizeY),
            static_cast<uint32_t>((batchSize + pipeline->localSizeZ - 1) / pipeline->localSizeZ)
          );
        }
        else if(pipeline->name.find("sum_channels") == 0) {
          vk_shader::push::SumChannelsParams params = {
            static_cast<uint32_t>(batchSize),
            1u,
            static_cast<uint32_t>(pipelineXYSize)
          };
          push(params);
          dispatch(
            1,
            1,
            static_cast<uint32_t>((batchSize + pipeline->localSizeZ - 1) / pipeline->localSizeZ)
          );
        }
        else if(pipeline->name.find("add_pointwise") == 0) {
          const size_t storageChannels = vk_helper::roundUpToMultiple(
            static_cast<size_t>(channels), static_cast<size_t>(nhwcChannelAlignment)
          );
          vk_shader::push::AddPointWiseParams params = {
            static_cast<uint32_t>(batchSize * storageChannels * pipelineXYSize)
          };
          push(params);
          dispatch((params.size + config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX - 1) /
                   (config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX));
        }
        else if(pipeline->name.find("add_channel_bias_") == 0) {
          const int channelsPadded = vk_helper::roundUpToMultipleInt(channels, nhwcChannelAlignment);
          vk_shader::push::AddChannelBiasNCHWParams params = {
            static_cast<uint32_t>(batchSize * (useNHWC ? channelsPadded : channels)),
            static_cast<uint32_t>(addChannelBiasesSpatialSize),
            static_cast<uint32_t>(channels), static_cast<uint32_t>(useNHWC ? channelsPadded : channels),
            static_cast<uint32_t>(logicalXYSize)
          };
          push(params);
          dispatch(
            (params.xySize + config.addChannelBiases.XY_ELTS_PER_THREAD * pipeline->localSizeX - 1) /
              (config.addChannelBiases.XY_ELTS_PER_THREAD * pipeline->localSizeX),
            (params.ncSize + config.addChannelBiases.NC_ELTS_PER_THREAD - 1) / config.addChannelBiases.NC_ELTS_PER_THREAD
          );
        }
        else if(pipeline->name.find("transformer_scale_dot_product") == 0) {
          const auto [heads, kvHeads] = attentionHeadsForRun(repeat);
          const int qBatchStride = heads * std::max(1, context.modelInfo.transformerHeadDim) * pipelineXYSize;
          const int kBatchStride = kvHeads * std::max(
            1,
            context.modelInfo.transformerHeadDim
          ) * pipelineXYSize;
          const int vBatchStride = kvHeads * std::max(
            1,
            context.modelInfo.transformerVHeadDim
          ) * pipelineXYSize;
          if(transformerAttentionCooperative) {
            const int qChannels = heads * std::max(1, context.modelInfo.transformerHeadDim);
            const int kChannels = kvHeads * std::max(1, context.modelInfo.transformerHeadDim);
            const int vChannels = kvHeads * std::max(1, context.modelInfo.transformerVHeadDim);
            const int qRowStride = transformerAttentionNHWC ? static_cast<int>(attentionInputRowStride) : qChannels;
            const int kRowStride = transformerAttentionNHWC ? static_cast<int>(attentionInputRowStride) : kChannels;
            const int vRowStride = transformerAttentionNHWC ? static_cast<int>(attentionInputRowStride) : vChannels;
            const int outputChannels = heads * std::max(1, context.modelInfo.transformerVHeadDim);
            const int outputRowStride = transformerAttentionNHWC
              ? vk_helper::roundUpToMultipleInt(outputChannels, 4) : outputChannels;
            vk_shader::push::ScaleDotProductCooperativePushParam params = {
              pipelineXYSize,
              heads,
              kvHeads,
              1.0f / sqrtf((float)std::max(1, context.modelInfo.transformerHeadDim)),
              0,
              transformerAttentionNHWC ? qChannels : 0,
              transformerAttentionNHWC ? qChannels + kChannels : 0,
              transformerAttentionNHWC ? qRowStride * pipelineXYSize : qBatchStride,
              transformerAttentionNHWC ? kRowStride * pipelineXYSize : kBatchStride,
              transformerAttentionNHWC ? vRowStride * pipelineXYSize : vBatchStride,
              qRowStride,
              kRowStride,
              vRowStride,
              outputRowStride * pipelineXYSize,
              outputRowStride,
              context.modelInfo.transformerUseRope ? 1 : 0,
              context.modelInfo.transformerLearnableRope ? 1 : 0,
              std::max(1, context.modelInfo.transformerHeadDim) / 2,
              0
            };
            push(params);
            dispatch(
              (pipelineXYSize + transformerAttentionBlockQ(config, plan.kernelName) - 1) /
                transformerAttentionBlockQ(config, plan.kernelName),
              static_cast<uint32_t>((batchSize * heads + pipeline->localSizeY - 1) / pipeline->localSizeY),
              static_cast<uint32_t>((1 + pipeline->localSizeZ - 1) / pipeline->localSizeZ)
            );
          }
          else {
            const PackedAttentionLayout* layout = packedScalarAttention ? &attentionLayouts[attentionCaseForRun(repeat)] : nullptr;
            const bool useRope = !packedScalarAttention && context.modelInfo.transformerUseRope;
            vk_shader::push::ScaleDotProductPushParam params = {
              pipelineXYSize,
              heads,
              kvHeads,
              1.0f / sqrtf((float)std::max(1, context.modelInfo.transformerHeadDim)),
              layout != nullptr ? layout->qOffset : 0,
              layout != nullptr ? layout->kOffset : 0,
              layout != nullptr ? layout->vOffset : 0,
              layout != nullptr ? layout->batchStride : qBatchStride,
              layout != nullptr ? layout->batchStride : kBatchStride,
              layout != nullptr ? layout->batchStride : vBatchStride,
              useRope ? 1 : 0,
              useRope && context.modelInfo.transformerLearnableRope ? 1 : 0,
              useRope ? context.modelInfo.transformerHeadDim / 2 : 0,
              0
            };
            push(params);
            dispatch(config.transformer.getDispatchGroupCountX(pipelineXYSize), static_cast<uint32_t>(batchSize * heads));
          }
        }
        else if(pipeline->name.find("transformer_rms_norm") == 0) {
          const int channelsPadded = vk_helper::roundUpToMultipleInt(channels, nhwcChannelAlignment);
          vk_shader::push::TransformerRMSNormPushParams params = {
            batchSize, channels, pipelineXYSize, 1e-6f,
            useNHWC ? channelsPadded : channels
          };
          push(params);
          dispatch(
            static_cast<uint32_t>((pipelineXYSize + config.rmsNorm.WG_XY_SIZE - 1) / config.rmsNorm.WG_XY_SIZE),
            static_cast<uint32_t>(batchSize)
          );
        }
        else if(pipeline->name.find("transformer_swiglu") == 0) {
          vk_shader::push::TransformerSwiGLUPushParams params = {batchSize * channels * pipelineXYSize};
          push(params);
          dispatch((params.size + config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX - 1) /
                   (config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX));
        }
        else if(pipeline->name.find("transformer_spatial_rms_norm_sum_sq") == 0) {
          const vkcompute::SpatialRMSNormSizing sizing = vkcompute::computeSpatialRMSNormSizing(
            config.spatialRMSNorm.TILE_SIZE,
            channels * pipelineXYSize
          );
          const int channelsPadded = vk_helper::roundUpToMultipleInt(channels, nhwcChannelAlignment);
          vk_shader::push::TransformerSpatialRMSNormSumSqPushParams params = {
            batchSize, channels, pipelineXYSize, sizing.tilesPerGroupPass1,
            useNHWC ? channelsPadded : channels
          };
          push(params);
          dispatch(static_cast<uint32_t>(sizing.numCHWWorkgroups), static_cast<uint32_t>(batchSize));
        }
        else if(pipeline->name.find("transformer_spatial_rms_norm_reduce") == 0) {
          const vkcompute::SpatialRMSNormSizing sizing = vkcompute::computeSpatialRMSNormSizing(
            config.spatialRMSNorm.TILE_SIZE,
            channels * pipelineXYSize
          );
          vk_shader::push::TransformerSpatialRMSNormReducePushParams params = {batchSize,sizing.numCHWWorkgroups,sizing.tilesPerGroupPass2};
          push(params);
          dispatch(1, static_cast<uint32_t>(batchSize));
        }
        else if(pipeline->name.find("transformer_spatial_rms_norm_apply") == 0) {
          const int channelsPadded = vk_helper::roundUpToMultipleInt(channels, nhwcChannelAlignment);
          vk_shader::push::TransformerSpatialRMSNormApplyPushParams params = {
            batchSize, channels, pipelineXYSize, 1e-6f,
            useNHWC ? channelsPadded : channels
          };
          push(params);
          dispatch(
            static_cast<uint32_t>((channels * pipelineXYSize + config.spatialRMSNorm.APPLY_ELTS_PER_THREAD * pipeline->localSizeX - 1) /
                                  (config.spatialRMSNorm.APPLY_ELTS_PER_THREAD * pipeline->localSizeX)),
            static_cast<uint32_t>(batchSize)
          );
        }
        else {
          vector<uint32_t> params((pipeline->pushConstantSize + 3) / 4, 1);
          if(!params.empty())
            vkCmdPushConstants(
              targetCommandBuffer,
              pipeline->layout,
              VK_SHADER_STAGE_COMPUTE_BIT,
              0,
              pipeline->pushConstantSize,
              params.data()
            );
          dispatch(1, 1, 1);
        }
      };

      const auto recordDispatches = [&](
        VkCommandBuffer targetCommandBuffer,
        size_t repeat,
        size_t firstPipeline,
        size_t pipelineCount
      ) {
        const int runBatchSize = plan.batchSizes.empty() ? std::max(1, context.batchSize) :
          plan.batchSizes[repeat % plan.batchSizes.size()];
        const size_t workload = isGemm
          ? repeat % plan.gemmCases.size()
          : repeat % plan.workloadWeights.size();
        const GemmDimensions runGemm = isGemm
          ? getGemmDimensions(plan.gemmCases[workload].inChannels, plan.gemmCases[workload].outChannels)
          : maxGemm;
        const int maxConvChannels = std::max({
          context.modelInfo.trunkNumChannels, context.modelInfo.midNumChannels,
          context.modelInfo.regularNumChannels, context.modelInfo.gpoolNumChannels,
          context.modelInfo.maxConvChannels3x3
        });
        const int runChannels = channelsForRun(repeat,
          plan.kernelName.find("conv") != string::npos
          ? std::max(1, workload == 2 || workload == 5 || workload == 8 ? context.modelInfo.midNumChannels :
                         workload == 3 || workload == 6 || workload == 9 ? maxConvChannels :
                         context.modelInfo.trunkNumChannels)
          : plan.kernelName == "gPool" ? gpoolChannels : trunkChannels);
        const int runInChannels = isGemm ? plan.gemmCases[workload].inChannels : runChannels;
        const int runOutChannels = isGemm ? plan.gemmCases[workload].outChannels : runChannels;
        const size_t lastPipeline = std::min(pipelines.size(), firstPipeline + pipelineCount);
        size_t firstBuffer = 0;
        for(size_t i = 0; i < firstPipeline; i++)
          firstBuffer += pipelines[i]->bindingCount;
        for(size_t i = firstPipeline; i < lastPipeline; i++) {
          const Pipeline* pipeline = pipelines[i];
          recordPipeline(
            targetCommandBuffer, pipeline, descriptorSets[i + (repeat % filterCopies) * pipelines.size()], runBatchSize,
            runInChannels, runOutChannels, pipeline->name.find("transformer_swiglu") == 0
              ? swiGLUChannelsForRun(repeat) : runChannels,
            runGemm.gemmM, runGemm.gemmN, runGemm.gemmK, repeat
          );
          if(plan.kernelName == "hgemmCooperativeMatrixNCHW") {
            vk_helper::barrierCommandBufferForBuffer(
              targetCommandBuffer, tuningBuffers[firstBuffer + outputBinding(pipeline)]
            );
          }
          firstBuffer += pipeline->bindingCount;
          if(i + 1 < lastPipeline)
            vk_helper::barrierCommandBuffer(targetCommandBuffer);
        }
      };

      const bool resetsInPlaceAccumulator =
        pointwiseAccumulatorInitialBuffer != nullptr &&
        (pipelines[0]->name.find("add_pointwise") == 0 ||
         pipelines[0]->name.find("add_channel_bias_") == 0);
      const auto resetInPlaceAccumulator = [&](size_t repeat) {
        if(!resetsInPlaceAccumulator)
          return;
        vk_helper::barrierCommandBufferForBuffer(
          commandBuffer, tuningBuffers[0],
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT
        );
        VkBufferCopy copyRegion = {};
        copyRegion.size = isPointwise ? pointwiseOutputBytes(pipelines[0], repeat) : pointwiseAccumulatorBytes;
        vkCmdCopyBuffer(
          commandBuffer, pointwiseAccumulatorInitialBuffer->buffer, tuningBuffers[0]->buffer, 1, &copyRegion
        );
        vk_helper::barrierCommandBufferForBuffer(
          commandBuffer, tuningBuffers[0],
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
        );
      };

      result = vk_helper::beginCommandBuffer(commandBuffer);
      if(result != VK_SUCCESS) {
        error = "could not begin tuning command buffer: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }
      size_t probeSubmission = 0;
      vector<pair<VkCommandBuffer, VkFence>> completedProbeSubmissions;
      const auto releaseProbeSubmissions = makeScopeGuard([&]() {
        for(const auto& [completedCommandBuffer, completedFence]: completedProbeSubmissions) {
          vkFreeCommandBuffers(device->device, commandPool, 1, &completedCommandBuffer);
          vkDestroyFence(device->device, completedFence, nullptr);
        }
      });
      const auto submitCommandBuffer = [&](bool beginNext) -> bool {
        vk_helper::barrierCommandBuffer(
          commandBuffer,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT
        );
        result = vk_helper::endCommandBuffer(commandBuffer);
        if(result != VK_SUCCESS) {
          error = "could not end tuning command buffer: " + vk_helper::vkErrorToString(result);
          return false;
        }
        result = vkResetFences(device->device, 1, &fence);
        if(result != VK_SUCCESS) {
          error = "could not reset tuning fence: " + vk_helper::vkErrorToString(result);
          return false;
        }
        const bool logProbe = context.attentionProbe && context.logger != nullptr;
        if(logProbe)
          context.logger->write(
            "Attention probe submit " + to_string(++probeSubmission) + "/" + to_string(plan.timedRuns()) +
            " groups=" + to_string(dispatchX) + "x" + to_string(dispatchY) + "x" + to_string(dispatchZ)
          );
        const auto probeStart = logProbe ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto logProbeResult = [&](const char* stage) {
          if(logProbe) {
            const double elapsedMs = std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - probeStart
            ).count();
            context.logger->write(
              "Attention probe " + to_string(probeSubmission) + " " + stage +
              "=" + vk_helper::vkErrorToString(result) + " elapsedMs=" + to_string(elapsedMs)
            );
          }
        };
        result = vk_helper::submitCommandBuffers(device, {commandBuffer}, fence);
        logProbeResult("vkQueueSubmit");
        if(result != VK_SUCCESS) {
          error = "could not submit tuning command buffer: " + vk_helper::vkErrorToString(result) +
            ", last dispatch groups=" + to_string(dispatchX) + "x" + to_string(dispatchY) + "x" + to_string(dispatchZ);
          return false;
        }
        commandBufferSubmitted = true;
        result = vk_helper::waitForFence(device, fence, "tuning vkWaitForFences");
        logProbeResult("vkWaitForFences");
        if(result != VK_SUCCESS) {
          error = "could not wait for tuning fence: " + vk_helper::vkErrorToString(result) +
            ", last dispatch groups=" + to_string(dispatchX) + "x" + to_string(dispatchY) + "x" + to_string(dispatchZ);
          return false;
        }
        commandBufferSubmitted = false;
        if(!beginNext)
          return true;

        if(context.tegraAttentionFreshCommands) {
          // Hold the completed handles so the driver cannot immediately recycle them.
          completedProbeSubmissions.emplace_back(commandBuffer, fence);
          commandBuffer = VK_NULL_HANDLE;
          fence = VK_NULL_HANDLE;
          if(!ensureCommandResources(result, error))
            return false;
        }
        else {
          result = vkResetCommandBuffer(commandBuffer, 0);
          if(result != VK_SUCCESS) {
            error = "could not reset tuning command buffer: " + vk_helper::vkErrorToString(result);
            return false;
          }
        }
        result = vk_helper::beginCommandBuffer(commandBuffer);
        if(result != VK_SUCCESS) {
          error = "could not begin tuning command buffer: " + vk_helper::vkErrorToString(result);
          return false;
        }
        vk_helper::barrierCommandBuffer(commandBuffer);
        return true;
      };
      const auto zeroBuffer = [&](VulkanBuffer* buffer) {
        vkCmdFillBuffer(commandBuffer, buffer->buffer, 0, VK_WHOLE_SIZE, 0);
        vk_helper::barrierCommandBufferForBuffer(
          commandBuffer, buffer,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT
        );
      };
      // These tuners initialize every consumed element by upload, dispatch or snapshot copy.
      if(!isGPool && !isGemm && !isPointwise && !winogradTransform)
        for(VulkanBuffer* buffer: tuningBuffers)
          if(buffer != nullptr)
            zeroBuffer(buffer);
      if(!isPointwise) {
        if(pointwiseAccumulatorInitialBuffer != nullptr)
          zeroBuffer(pointwiseAccumulatorInitialBuffer);
        for(VulkanBuffer* buffer: pointwiseValidationBuffers)
          zeroBuffer(buffer);
      }
      for(VulkanBuffer* buffer: attentionValidationBuffers)
        zeroBuffer(buffer);
      for(VulkanBuffer* buffer: spatialValidationBuffers)
        zeroBuffer(buffer);
      for(const PendingUpload& upload: pendingUploads) {
        VkBufferCopy copyRegion = {};
        copyRegion.size = upload.size;
        vkCmdCopyBuffer(
          commandBuffer,
          upload.stagingBuffer->buffer,
          upload.destinationBuffer->buffer,
          1,
          &copyRegion
        );
        vk_helper::barrierCommandBufferForBuffer(
          commandBuffer, upload.destinationBuffer,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT
        );
      }
      if(filterCopies > 1) {
        VulkanBuffer* filter = tuningBuffers[1];
        for(uint32_t copy = 1; copy < filterCopies; copy++) {
          VkBufferCopy region = {};
          region.dstOffset = copy * filterStride;
          region.size = filterBytes;
          vkCmdCopyBuffer(commandBuffer, filter->buffer, filter->buffer, 1, &region);
        }
        vk_helper::barrierCommandBufferForBuffer(commandBuffer, filter,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
      }
      if(isTransformerAttentionTuner(plan.kernelName)) {
        VulkanBuffer* outputBuffer = tuningBuffers[outputBinding(pipelines[0])];
        vk_helper::barrierCommandBufferForBuffer(
          commandBuffer, outputBuffer,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_WRITE_BIT
        );
        vkCmdFillBuffer(commandBuffer, outputBuffer->buffer, 0, VK_WHOLE_SIZE, 0);
        vk_helper::barrierCommandBufferForBuffer(
          commandBuffer, outputBuffer,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT
        );
      }
      // Production projections execute in a forward command buffer without a host
      // fence between kernels. Keep other tuners' submission policy unchanged.
      const size_t runsPerSubmission = context.transformerPhysicalLayout != nullptr ||
        plan.kernelName == "transformerXgemm" ? 32 : MAX_TUNER_RUNS_PER_SUBMISSION;
      // Zero-weight runs are timestamped too, matching OpenCL event handling.
      for(size_t repeat = 0; repeat < plan.warmupRuns; repeat++) {
        resetInPlaceAccumulator(repeat);
        if(spatialRMSNorm) {
          recordDispatches(commandBuffer, repeat, 0, 2);
          vk_helper::barrierCommandBuffer(commandBuffer);
        }
        recordDispatches(commandBuffer, repeat, firstTimedPipeline, timedPipelineCount);
        if((repeat + 1) < plan.warmupRuns &&
           (repeat + 1) % runsPerSubmission == 0 &&
           !submitCommandBuffer(true))
          return false;
      }
      const size_t timedRuns = plan.timedRuns();
      const auto wallTimerStart = context.attentionSkipTimestamps
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      if(!context.attentionSkipTimestamps)
        vkCmdResetQueryPool(
          commandBuffer, queryPool, 0, static_cast<uint32_t>(2 * timedRuns * timedPipelineCount)
        );
      const size_t spatialApplyFirstBuffer = spatialRMSNorm ?
        pipelines[0]->bindingCount + pipelines[1]->bindingCount : 0;
      for(size_t timedRepeat = 0; timedRepeat < timedRuns; timedRepeat++) {
        const uint32_t queryStart = static_cast<uint32_t>(2 * timedRepeat * timedPipelineCount);
        resetInPlaceAccumulator(plan.warmupRuns + timedRepeat);
        vk_helper::barrierCommandBuffer(commandBuffer);
        if(spatialRMSNorm) {
          recordDispatches(commandBuffer, plan.warmupRuns + timedRepeat, 0, 2);
          vk_helper::barrierCommandBuffer(commandBuffer);
        }
        for(size_t timedPipeline = 0; timedPipeline < timedPipelineCount; timedPipeline++) {
          const uint32_t pipelineQueryStart = queryStart + static_cast<uint32_t>(2 * timedPipeline);
          if(!context.attentionSkipTimestamps)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, pipelineQueryStart);
          recordDispatches(
            commandBuffer, plan.warmupRuns + timedRepeat, firstTimedPipeline + timedPipeline, 1
          );
          if(!context.attentionSkipTimestamps)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, pipelineQueryStart + 1);
          if(timedPipeline + 1 < timedPipelineCount)
            vk_helper::barrierCommandBuffer(commandBuffer);
        }
        if(isGemm && timedRepeat < plan.gemmCases.size() && timedRepeat < gemmValidationBuffers.size()) {
          const Pipeline* gemmPipeline = pipelines[0];
          const uint32_t binding = outputBinding(gemmPipeline);
          VulkanBuffer* outputBuffer = tuningBuffers[binding];
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          VkBufferCopy copyRegion = {};
          copyRegion.size = gemmOutputBytes;
          vkCmdCopyBuffer(
            commandBuffer, outputBuffer->buffer, gemmValidationBuffers[timedRepeat]->buffer, 1, &copyRegion
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, gemmValidationBuffers[timedRepeat],
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT
          );
        }
        else if(spatialRMSNorm && timedRepeat < spatialValidationBuffers.size()) {
          const Pipeline* applyPipeline = pipelines[2];
          const uint32_t binding = outputBinding(applyPipeline);
          VulkanBuffer* outputBuffer = tuningBuffers[spatialApplyFirstBuffer + binding];
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          VkBufferCopy copyRegion = {};
          copyRegion.size = scratchBytes;
          vkCmdCopyBuffer(
            commandBuffer, outputBuffer->buffer, spatialValidationBuffers[timedRepeat]->buffer, 1, &copyRegion
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, spatialValidationBuffers[timedRepeat],
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT
          );
        }
        else if(plan.kernelName == "gPool" && timedRepeat < gpoolValidationBuffers.size()) {
          const Pipeline* gpoolPipeline = pipelines[0];
          const uint32_t binding = outputBinding(gpoolPipeline);
          VulkanBuffer* outputBuffer = tuningBuffers[binding];
          const size_t count = batchSize * channelsForRun(plan.warmupRuns + timedRepeat, gpoolChannels) * 3;
          const VkDeviceSize outputBytes = count *
            (halfBinding(gpoolPipeline, binding) ? sizeof(half_t) : sizeof(float));
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          VkBufferCopy copyRegion = {};
          copyRegion.size = outputBytes;
          vkCmdCopyBuffer(
            commandBuffer, outputBuffer->buffer, gpoolValidationBuffers[timedRepeat]->buffer, 1, &copyRegion
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, gpoolValidationBuffers[timedRepeat],
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT
          );
        }
        else if(winogradTransform && timedRepeat < winogradValidationBuffers.size()) {
          const Pipeline* pipeline = pipelines[0];
          const uint32_t binding = outputBinding(pipeline);
          VulkanBuffer* outputBuffer = tuningBuffers[binding];
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          VkBufferCopy copyRegion = {};
          copyRegion.size = winogradValidationElements(plan.warmupRuns + timedRepeat) *
            (halfBinding(pipeline, binding) ? sizeof(half_t) : sizeof(float));
          vkCmdCopyBuffer(
            commandBuffer, outputBuffer->buffer, winogradValidationBuffers[timedRepeat]->buffer, 1, &copyRegion
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, winogradValidationBuffers[timedRepeat],
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT
          );
        }
        else if(isPointwise && timedRepeat < 10) {
          size_t firstPipelineBuffer = 0;
          for(size_t pipelineIndex = 0; pipelineIndex < timedPipelineCount; pipelineIndex++) {
            const Pipeline* pipeline = pipelines[firstTimedPipeline + pipelineIndex];
            const uint32_t binding = outputBinding(pipeline);
            const size_t validationIndex = timedRepeat * timedPipelineCount + pipelineIndex;
            VulkanBuffer* outputBuffer = tuningBuffers[firstPipelineBuffer + binding];
            vk_helper::barrierCommandBufferForBuffer(
              commandBuffer, outputBuffer,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
            );
            VkBufferCopy copyRegion = {};
            copyRegion.size = pointwiseOutputBytes(pipeline, plan.warmupRuns + timedRepeat);
            vkCmdCopyBuffer(
              commandBuffer, outputBuffer->buffer, pointwiseValidationBuffers[validationIndex]->buffer, 1, &copyRegion
            );
            vk_helper::barrierCommandBufferForBuffer(
              commandBuffer, pointwiseValidationBuffers[validationIndex],
              VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
            );
            vk_helper::barrierCommandBufferForBuffer(
              commandBuffer, outputBuffer,
              VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
            );
            firstPipelineBuffer += pipeline->bindingCount;
          }
        }
        else if(
          (plan.kernelName == "addChannelBiases" || plan.kernelName == "transformerRMSNorm") &&
                timedRepeat < pointwiseValidationBuffers.size()
        ) {
          const uint32_t binding = outputBinding(pipelines[0]);
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, tuningBuffers[binding],
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          VkBufferCopy copyRegion = {};
          copyRegion.size = scratchBytes;
          vkCmdCopyBuffer(
            commandBuffer, tuningBuffers[binding]->buffer, pointwiseValidationBuffers[timedRepeat]->buffer, 1, &copyRegion
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, pointwiseValidationBuffers[timedRepeat],
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, tuningBuffers[binding],
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
          );
        }
        else if(
          !context.attentionSkipSnapshots &&
          isTransformerAttentionTuner(plan.kernelName) && timedRepeat < attentionValidationBuffers.size()
        ) {
          const Pipeline* attentionPipeline = pipelines[0];
          const uint32_t binding = outputBinding(attentionPipeline);
          VulkanBuffer* outputBuffer = tuningBuffers[binding];
          const VkDeviceSize outputBytes = attentionOutputElements *
            (halfBinding(attentionPipeline, binding) ? sizeof(half_t) : sizeof(float));
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          VkBufferCopy copyRegion = {};
          copyRegion.size = outputBytes;
          vkCmdCopyBuffer(
            commandBuffer, outputBuffer->buffer, attentionValidationBuffers[timedRepeat]->buffer, 1, &copyRegion
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, attentionValidationBuffers[timedRepeat],
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT
          );
        }
        if((timedRepeat + 1) < timedRuns &&
           (timedRepeat + 1) % runsPerSubmission == 0 &&
           !submitCommandBuffer(true))
          return false;
      }
      if(!submitCommandBuffer(false))
        return false;
      if(timedRuns == 0) {
        error = "tuning measurement plan has no timed runs";
        cleanup();
        return false;
      }
      if(context.attentionProbe && context.logger != nullptr)
        context.logger->write("Attention probe dispatches completed");
      if(context.attentionSkipTimestamps) {
        callsPerSecond = timedRuns /
          std::chrono::duration<double>(std::chrono::steady_clock::now() - wallTimerStart).count();
      }
      else {
        vector<uint64_t> timestamps(2 * timedRuns * timedPipelineCount, 0);
        result = vkGetQueryPoolResults(
          device->device, queryPool, 0, static_cast<uint32_t>(timestamps.size()),
          timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT
        );
        if(result != VK_SUCCESS) {
          error = "could not read tuning timestamps: " + vk_helper::vkErrorToString(result);
          cleanup();
          return false;
        }
        double weightCounted = 0.0;
        double weightedTimeTaken = 0.0;
        for(size_t timedRepeat = 0; timedRepeat < timedRuns; timedRepeat++) {
          const size_t planRun = plan.warmupRuns + timedRepeat;
          double weight = plan.weightForRun(planRun);
          for(size_t timedPipeline = 0; timedPipeline < timedPipelineCount; timedPipeline++) {
            const size_t queryIndex = (timedRepeat * timedPipelineCount + timedPipeline) * 2;
            const uint64_t start = timestamps[queryIndex];
            const uint64_t end = timestamps[queryIndex + 1];
            if(end <= start) {
              error = "could not read a valid tuning timestamp";
              cleanup();
              return false;
            }
            const double elapsedSeconds = (end - start) * timestampPeriod * 1e-9;
            weightCounted += weight;
            weightedTimeTaken += elapsedSeconds * weight;
          }
        }
        if(weightCounted <= 0.0 || weightedTimeTaken <= 0.0) {
          error = "tuning measurement has no weighted runs";
          cleanup();
          return false;
        }
        callsPerSecond = weightCounted / weightedTimeTaken;
      }

      // Compare only produced values. Input buffers and unused allocation
      // tails must not dilute the relative error; each output has its own type.
      readback.clear();
      if(isGemm && !gemmValidationBuffers.empty()) {
        const Pipeline* gemmPipeline = pipelines[0];
        const uint32_t binding = outputBinding(gemmPipeline);
        const bool useFP16 = halfBinding(gemmPipeline, binding);
        for(size_t caseIndex = 0; caseIndex < plan.gemmCases.size(); caseIndex++) {
          const GemmDimensions dimensions = getGemmDimensions(
            plan.gemmCases[caseIndex].inChannels, plan.gemmCases[caseIndex].outChannels
          );
          const int outputRows = dimensions.gemmM;
          const int outputChannels = dimensions.gemmN;
          const size_t count = static_cast<size_t>(gemmBatch) * outputRows * outputChannels;
          vector<float> output(count);
          if(useFP16) {
            vector<half_t> halves(count);
            vk_helper::copyDeviceBufferToHost(
              device, gemmValidationBuffers[caseIndex], count * sizeof(half_t), halves.data(), true, &result, commandPool
            );
            for(size_t j = 0; j < count; j++)
              output[j] = half_float::half_cast<float>(halves[j]);
          }
          else {
            vk_helper::copyDeviceBufferToHost(
              device, gemmValidationBuffers[caseIndex], count * sizeof(float), output.data(), true, &result, commandPool
            );
          }
          if(result != VK_SUCCESS) {
            error = "could not read GEMM validation output: " + vk_helper::vkErrorToString(result);
            return false;
          }
          if(nhwcCooperative || denseGemm) {
            for(int n = 0; n < gemmBatch; n++)
              for(int x = 0; x < dimensions.logicalM; x++)
                for(int y = 0; y < dimensions.logicalN; y++)
                  readback.push_back(output[(static_cast<size_t>(n) * outputRows + x) * outputChannels + y]);
          }
          else {
            for(int n = 0; n < gemmBatch; n++)
              for(int y = 0; y < dimensions.logicalN; y++)
                for(int x = 0; x < dimensions.logicalM; x++)
                  readback.push_back(output[(static_cast<size_t>(n) * dimensions.gemmN + y) * dimensions.gemmM + x]);
          }
        }
        cleanup();
        return true;
      }
      if(plan.kernelName == "gPool" && !gpoolValidationBuffers.empty()) {
        const Pipeline* gpoolPipeline = pipelines[0];
        const uint32_t binding = outputBinding(gpoolPipeline);
        const bool useFP16 = halfBinding(gpoolPipeline, binding);
        for(size_t repeat = 0; repeat < gpoolValidationBuffers.size(); repeat++) {
          VulkanBuffer* validationBuffer = gpoolValidationBuffers[repeat];
          const size_t count = batchSize * channelsForRun(plan.warmupRuns + repeat, gpoolChannels) * 3;
          vector<float> output(count);
          if(useFP16) {
            vector<half_t> halves(count);
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(half_t), halves.data(), true, &result, commandPool
            );
            for(size_t j = 0; j < count; j++)
              output[j] = half_float::half_cast<float>(halves[j]);
          }
          else {
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(float), output.data(), true, &result, commandPool
            );
          }
          if(result != VK_SUCCESS) {
            error = "could not read global pooling validation output: " + vk_helper::vkErrorToString(result);
            return false;
          }
          readback.insert(readback.end(), output.begin(), output.end());
        }
        cleanup();
        return true;
      }
      if(winogradTransform && !winogradValidationBuffers.empty()) {
        const Pipeline* pipeline = pipelines[0];
        const bool useFP16 = halfBinding(pipeline, outputBinding(pipeline));
        for(size_t repeat = 0; repeat < winogradValidationBuffers.size(); repeat++) {
          VulkanBuffer* validationBuffer = winogradValidationBuffers[repeat];
          const size_t count = winogradValidationElements(plan.warmupRuns + repeat);
          vector<float> output(count);
          if(useFP16) {
            vector<half_t> halves(count);
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(half_t), halves.data(), true, &result, commandPool
            );
            for(size_t j = 0; j < count; j++)
              output[j] = half_float::half_cast<float>(halves[j]);
          }
          else {
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(float), output.data(), true, &result, commandPool
            );
          }
          if(result != VK_SUCCESS) {
            error = "could not read Winograd validation output: " + vk_helper::vkErrorToString(result);
            return false;
          }
          readback.insert(readback.end(), output.begin(), output.end());
        }
        cleanup();
        return true;
      }
      if(plan.kernelName == "spatialRMSNorm" && !spatialValidationBuffers.empty()) {
        const Pipeline* applyPipeline = pipelines[2];
        const uint32_t binding = outputBinding(applyPipeline);
        const size_t count = batchSize * vk_helper::roundUpToMultiple(
          static_cast<size_t>(std::max(1, context.modelInfo.trunkNumChannels)),
          static_cast<size_t>(nhwcChannelAlignment)
        ) * xySize;
        for(VulkanBuffer* validationBuffer: spatialValidationBuffers) {
          vector<float> output(count);
          if(halfBinding(applyPipeline, binding)) {
            vector<half_t> halves(count);
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(half_t), halves.data(), true, &result, commandPool
            );
            for(size_t j = 0; j < count; j++)
              output[j] = half_float::half_cast<float>(halves[j]);
          }
          else {
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(float), output.data(), true, &result, commandPool
            );
          }
          if(result != VK_SUCCESS) {
            error = "could not read Spatial RMSNorm validation output: " + vk_helper::vkErrorToString(result);
            return false;
          }
          readback.insert(readback.end(), output.begin(), output.end());
        }
        cleanup();
        return true;
      }
      if(isPointwise && !pointwiseValidationBuffers.empty()) {
        for(size_t repeat = 0; repeat < 10; repeat++) {
          for(size_t pipelineIndex = 0; pipelineIndex < pipelines.size(); pipelineIndex++) {
            const Pipeline* pipeline = pipelines[pipelineIndex];
            const uint32_t binding = outputBinding(pipeline);
            const size_t count = pointwiseRunElements(pipeline, plan.warmupRuns + repeat);
            const size_t validationIndex = repeat * pipelines.size() + pipelineIndex;
            vector<float> output(count);
            if(halfBinding(pipeline, binding)) {
              vector<half_t> halves(count);
              vk_helper::copyDeviceBufferToHost(
                device, pointwiseValidationBuffers[validationIndex], count * sizeof(half_t), halves.data(), true, &result, commandPool
              );
              for(size_t j = 0; j < count; j++)
                output[j] = half_float::half_cast<float>(halves[j]);
            }
            else {
              vk_helper::copyDeviceBufferToHost(
                device, pointwiseValidationBuffers[validationIndex], count * sizeof(float), output.data(), true, &result, commandPool
              );
            }
            if(result != VK_SUCCESS) {
              error = "could not read pointwise validation output: " + vk_helper::vkErrorToString(result);
              return false;
            }
            readback.insert(readback.end(), output.begin(), output.end());
          }
        }
        cleanup();
        return true;
      }
      if((plan.kernelName == "addChannelBiases" || plan.kernelName == "transformerRMSNorm") &&
         !pointwiseValidationBuffers.empty()) {
        const uint32_t binding = outputBinding(pipelines[0]);
        const size_t spatialSize = plan.kernelName == "addChannelBiases" ? addChannelBiasesSpatialSize : xySize;
        for(size_t repeat = 0; repeat < pointwiseValidationBuffers.size(); repeat++) {
          VulkanBuffer* validationBuffer = pointwiseValidationBuffers[repeat];
          const size_t channels = channelsForRun(plan.warmupRuns + repeat, trunkChannels);
          const size_t count = batchSize * vk_helper::roundUpToMultiple(
            channels, static_cast<size_t>(nhwcChannelAlignment)
          ) * spatialSize;
          vector<float> output(count);
          if(halfBinding(pipelines[0], binding)) {
            vector<half_t> halves(count);
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(half_t), halves.data(), true, &result, commandPool
            );
            for(size_t j = 0; j < count; j++)
              output[j] = half_float::half_cast<float>(halves[j]);
          }
          else {
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(float), output.data(), true, &result, commandPool
            );
          }
          if(result != VK_SUCCESS) {
            error = "could not read pointwise validation output: " + vk_helper::vkErrorToString(result);
            return false;
          }
          readback.insert(readback.end(), output.begin(), output.end());
        }
        cleanup();
        return true;
      }
      if(isTransformerAttentionTuner(plan.kernelName) && !attentionValidationBuffers.empty()) {
        const Pipeline* attentionPipeline = pipelines[0];
        const uint32_t binding = outputBinding(attentionPipeline);
        const bool useFP16 = halfBinding(attentionPipeline, binding);
        const VkDeviceSize outputBytes = attentionOutputElements * (useFP16 ? sizeof(half_t) : sizeof(float));
        if(context.attentionSkipSnapshots)
          return readAttentionValidationBuffers(
            {tuningBuffers[binding]}, outputBytes, {attentionOutputElementsForRun(plan.totalRuns - 1)},
            useFP16, readback, result, error
          );
        vector<size_t> outputCounts;
        for(size_t repeat = 0; repeat < attentionValidationBuffers.size(); repeat++)
          outputCounts.push_back(attentionOutputElementsForRun(plan.warmupRuns + repeat));
        if(!readAttentionValidationBuffers(
          attentionValidationBuffers, outputBytes, outputCounts, useFP16, readback, result, error
        ))
          return false;
        cleanup();
        return true;
      }
      size_t firstBuffer = 0;
      for(const Pipeline* pipeline: pipelines) {
        const string& name = pipeline->name;
        size_t count = batchSize * std::max(1, context.modelInfo.trunkNumChannels) * xySize;
        if(isGemm)
          count = static_cast<size_t>(gemmBatch) * gemmM * gemmN;
        else if(name.find("winograd_input_transform") == 0) {
          const ConvTuneParams& conv = name.find("5x5") != string::npos ? config.conv5x5 : config.conv3x3;
          const size_t tiles = batchSize * ((context.nnXLen + conv.outTileXSize - 1) / conv.outTileXSize) *
            ((context.nnYLen + conv.outTileYSize - 1) / conv.outTileYSize);
          count = vk_helper::roundUpToMultiple(tiles, static_cast<size_t>(activeMPaddingMult)) *
            vk_helper::roundUpToMultiple(maxConvChannels,
                                         static_cast<size_t>(activeKPaddingMult)) * conv.inTileXSize * conv.inTileYSize;
        }
        else if(name.find("winograd_output_transform") == 0)
          count = batchSize * maxConvChannels * xySize;
        else if(name.find("global_pooling_channels") == 0 || name.find("value_head_pool_channels") == 0)
          count = batchSize * std::max(1, context.modelInfo.gpoolNumChannels) * 3;
        else if(name.find("sum_channels") == 0)
          count = batchSize;
        else if(name.find("transformer_scale_dot_product") == 0)
          count = batchSize * std::max(1, context.modelInfo.transformerNumHeads) *
            std::max(1, context.modelInfo.transformerVHeadDim) * xySize;
        else if(name.find("transformer_swiglu") == 0)
          count = batchSize * std::max(
            context.modelInfo.trunkNumChannels,
            context.modelInfo.transformerFFNChannels
          ) * xySize;
        else if(
          name.find("transformer_spatial_rms_norm_sum_sq") == 0 ||
                name.find("transformer_spatial_rms_norm_reduce") == 0
        ) {
          // Intermediate reduction sizes vary by tile. Validate the connected
          // pipeline's final normalized output instead.
          firstBuffer += pipeline->bindingCount;
          continue;
        }
        const uint32_t binding = outputBinding(pipeline);
        vector<float> output(count);
        if(halfBinding(pipeline, binding)) {
          vector<half_t> halves(count);
          vk_helper::copyDeviceBufferToHost(
            device, tuningBuffers[firstBuffer + binding], count * sizeof(half_t), halves.data(), true, &result, commandPool
          );
          for(size_t j = 0; j < count; j++)
            output[j] = half_float::half_cast<float>(halves[j]);
        }
        else {
          vk_helper::copyDeviceBufferToHost(
            device, tuningBuffers[firstBuffer + binding], count * sizeof(float), output.data(), true, &result, commandPool
          );
        }
        if(result != VK_SUCCESS) {
          error = "could not read tuning output: " + vk_helper::vkErrorToString(result);
          return false;
        }
        if(isGemm) {
          if(nhwcCooperative || denseGemm) {
            for(int n = 0; n < gemmBatch; n++)
              for(int x = 0; x < logicalM; x++)
                for(int y = 0; y < logicalN; y++)
                  readback.push_back(output[(static_cast<size_t>(n) * gemmM + x) * gemmN + y]);
          }
          else {
            for(int n = 0; n < gemmBatch; n++)
              for(int y = 0; y < logicalN; y++)
                for(int x = 0; x < logicalM; x++)
                  readback.push_back(output[(static_cast<size_t>(n) * gemmN + y) * gemmM + x]);
          }
        }
        else
          readback.insert(readback.end(), output.begin(), output.end());
        firstBuffer += pipeline->bindingCount;
      }
      cleanup();
      return true;
    }

    bool measureTransformerDualGemmSwiGLU(
      const VulkanTuneParams& candidateConfig,
      const VulkanTuneParams& legacyConfig,
      const TuningContext& context,
      const Pipeline* fusedPipeline,
      const Pipeline* nchwToNhwcPipeline,
      const Pipeline* nhwcToNchwPipeline,
      const Pipeline* legacyGemmPipeline,
      const Pipeline* swigluPipeline,
      bool legacyNCHW,
      int hwSize,
      int packedOCSize,
      double& fusedCallsPerSecond,
      double& legacyCallsPerSecond,
      double& errorProp,
      vector<float>& readback,
      string& error,
      size_t timedRuns = 16,
      bool reverseOrder = false
    ) {
      if(!isUsable()) {
        error = "compute timestamps are not supported";
        return false;
      }
      const int batchSize = std::max(1, context.batchSize);
      const int cSize = VulkanTuner::getTransformerFFNInputChannelsForTuning(context.modelInfo);
      const int ffnSize = context.modelInfo.transformerFFNChannels;
      const bool legacyNHWC = legacyConfig.vulkan.shouldUseNHWC;
      if(fusedPipeline == nullptr || nchwToNhwcPipeline == nullptr || nhwcToNchwPipeline == nullptr ||
         legacyGemmPipeline == nullptr || swigluPipeline == nullptr ||
         batchSize <= 0 || hwSize <= 0 || cSize <= 0 || ffnSize <= 0 || packedOCSize < 2 * ffnSize) {
        error = "invalid Transformer dual-GEMM measurement dimensions or pipelines";
        return false;
      }

      VkResult result = VK_SUCCESS;
      const auto deviceLossGuard = makeScopeGuard([&]() { device->recordDeviceLoss(result, "fused FFN measurement"); });
      const size_t inputElements = static_cast<size_t>(batchSize) * cSize * hwSize;
      const size_t filterElements = static_cast<size_t>(cSize) * packedOCSize;
      const size_t fusedFilterElements = 2ull * cSize * ffnSize;
      const size_t outputElements = static_cast<size_t>(batchSize) * ffnSize * hwSize;
      const int channelAlignment = legacyNHWC
        ? std::lcm(
          4,
          std::lcm(legacyConfig.hgemmCooperativeMatrixNHWC.NWG, legacyConfig.hgemmCooperativeMatrixNHWC.KWG)
        ) : 4;
      const int inputChannelStride = vk_helper::roundUpToMultipleInt(cSize, channelAlignment);
      const int outputChannelStride = vk_helper::roundUpToMultipleInt(ffnSize, channelAlignment);
      const int packedOutputStride = std::max(
        packedOCSize,
        vk_helper::roundUpToMultipleInt(2 * ffnSize, channelAlignment)
      );
      const size_t legacyGemmElements = static_cast<size_t>(batchSize) *
        (legacyNHWC ? packedOutputStride : packedOCSize) * hwSize;
      const size_t candidateInputNHWCElements =
        static_cast<size_t>(batchSize) * hwSize * inputChannelStride;
      const size_t candidateOutputNHWCElements =
        static_cast<size_t>(batchSize) * hwSize * outputChannelStride;
      const bool legacyStorageHalf =
        legacyNCHW || (legacyConfig.vulkan.canUseFP16Storage && legacyConfig.vulkan.canUseFP16Compute &&
                       legacyConfig.vulkan.shouldUseFP16Storage);
      const size_t legacyElementBytes = legacyStorageHalf ? sizeof(half_t) : sizeof(float);
      const VkDeviceSize candidateInputBytes = inputElements * sizeof(half_t);
      const VkDeviceSize candidateFilterBytes = fusedFilterElements * sizeof(half_t);
      const VkDeviceSize candidateInputNHWCBytes = candidateInputNHWCElements * sizeof(half_t);
      const VkDeviceSize candidateOutputNHWCBytes = candidateOutputNHWCElements * sizeof(half_t);
      const VkDeviceSize legacyInputBytes = (legacyNHWC ? candidateInputNHWCElements : inputElements) * legacyElementBytes;
      const VkDeviceSize legacyFilterBytes = filterElements * legacyElementBytes;
      const VkDeviceSize legacyGemmBytes = legacyGemmElements * legacyElementBytes;
      const VkDeviceSize legacyOutputBytes = (legacyNHWC ? candidateOutputNHWCElements : outputElements) * legacyElementBytes;
      const VkDeviceSize requiredBytes[] = {
        candidateInputBytes, candidateFilterBytes, candidateInputNHWCBytes,
        candidateOutputNHWCBytes, legacyInputBytes, legacyFilterBytes,
        legacyGemmBytes, legacyOutputBytes
      };
      const vector<int> fixtureKey = {
        batchSize, context.nnXLen, context.nnYLen, hwSize, cSize, ffnSize, packedOCSize,
        channelAlignment, legacyNHWC, legacyStorageHalf
      };
      if(dualGemmFixtureKey != fixtureKey) {
        dualGemmResources = std::make_unique<ReusableResources>(device, commandPool);
        dualGemmFixtureKey.clear();
        dualGemmReference.clear();
      }
      ReusableResources& resources = *dualGemmResources;
      activeResources = &resources;
      bool measurementComplete = false;
      const auto fixtureGuard = makeScopeGuard([&]() {
        if(!measurementComplete)
          dualGemmFixtureKey.clear();
      });
      const auto activeResourcesGuard = makeScopeGuard([&]() { activeResources = nullptr; });
      vector<VulkanBuffer*>& buffers = resources.tuningBuffers;
      if(!ensureBuffers(buffers, 8, 4, result, error, "Transformer dual-GEMM buffer"))
        return false;
      for(size_t i = 0; i < 8; i++)
        if(!ensureBuffer(buffers[i], requiredBytes[i], result, error, "Transformer dual-GEMM buffer"))
          return false;

      const size_t logicalHW = static_cast<size_t>(context.nnXLen) * context.nnYLen;
      if(dualGemmFixtureKey.empty()) {
        vector<float> inputSource(inputElements, 0.0f);
        vector<float> mainWeights(static_cast<size_t>(cSize) * ffnSize);
        vector<float> gateWeights(static_cast<size_t>(cSize) * ffnSize);
        Rand rand(
          "VulkanTransformerDualGemmSwiGLU:" + to_string(context.nnXLen) + ":" +
                  to_string(context.nnYLen) + ":" + to_string(cSize) + ":" + to_string(ffnSize)
        );
        const float weightScale = 1.0f / sqrtf(static_cast<float>(cSize));
        for(int batch = 0; batch < batchSize; batch++) {
          for(int ic = 0; ic < cSize; ic++) {
            for(int hw = 0; hw < hwSize; hw++) {
              if(static_cast<size_t>(hw) < logicalHW)
                inputSource[(static_cast<size_t>(batch) * cSize + ic) * hwSize + hw] =
                  static_cast<float>(2.0 * rand.nextDouble() - 1.0);
            }
          }
        }
        for(size_t i = 0; i < mainWeights.size(); i++) {
          mainWeights[i] = static_cast<float>(2.0 * rand.nextDouble() - 1.0) * weightScale;
          gateWeights[i] = static_cast<float>(2.0 * rand.nextDouble() - 1.0) * weightScale;
        }
        const vector<float> packedFilter = VulkanTuner::packTransformerDualGemmSwiGLUFilter(
          mainWeights, gateWeights, cSize, ffnSize, packedOCSize
        );
        vector<half_t> candidateInputHalf(inputElements);
        vector<half_t> candidateInputNHWCHalf(candidateInputNHWCElements, half_t(0));
        const auto fusedWeights = VulkanTuner::packTransformerDualGemmSwiGLUFilterForNHWC(
          mainWeights,
          gateWeights,
          cSize,
          ffnSize
        );
        vector<half_t> candidateFilterHalf(fusedFilterElements);
        vector<half_t> referenceFilterHalf(filterElements);
        for(size_t i = 0; i < inputElements; i++)
          candidateInputHalf[i] = half_float::half_cast<half_t>(inputSource[i]);
        for(int batch = 0; batch < batchSize; batch++)
          for(int hw = 0; hw < hwSize; hw++)
            for(int channel = 0; channel < cSize; channel++)
              candidateInputNHWCHalf[(static_cast<size_t>(batch) * hwSize + hw) * inputChannelStride + channel] =
                candidateInputHalf[(static_cast<size_t>(batch) * cSize + channel) * hwSize + hw];
        for(size_t i = 0; i < fusedFilterElements; i++)
          candidateFilterHalf[i] = half_float::half_cast<half_t>(fusedWeights[i]);
        for(size_t i = 0; i < filterElements; i++)
          referenceFilterHalf[i] = half_float::half_cast<half_t>(packedFilter[i]);
        dualGemmReference = VulkanTuner::computeTransformerDualGemmSwiGLUReference(
          vector<float>(candidateInputHalf.begin(), candidateInputHalf.end()),
          vector<float>(referenceFilterHalf.begin(), referenceFilterHalf.end()),
          batchSize, hwSize, cSize, packedOCSize, ffnSize
        );
        const auto upload = [&](
          const void* data,
          VkDeviceSize bytes,
          VulkanBuffer* destination,
          const string& name
        ) {
          vk_helper::copyHostToDeviceBuffer(device, data, destination, bytes, false, &result, commandPool);
          if(result != VK_SUCCESS) {
            error = "could not upload " + name + ": " + vk_helper::vkErrorToString(result);
            return false;
          }
          return true;
        };
        if(!upload(candidateInputHalf.data(), candidateInputBytes, buffers[0], "dual-GEMM input") ||
           !upload(
             candidateInputNHWCHalf.data(),
             candidateInputNHWCBytes,
             buffers[2],
             "dual-GEMM NHWC input"
           ) ||
           !upload(candidateFilterHalf.data(), candidateFilterBytes, buffers[1], "dual-GEMM packed filter"))
          return false;

        if(legacyStorageHalf) {
          vector<half_t> legacyInputHalf(inputElements);
          vector<half_t> legacyFilterHalf(filterElements);
          for(size_t i = 0; i < inputElements; i++)
            legacyInputHalf[i] = half_float::half_cast<half_t>(inputSource[i]);
          for(size_t i = 0; i < filterElements; i++)
            legacyFilterHalf[i] = half_float::half_cast<half_t>(packedFilter[i]);
          if(!upload(legacyNHWC ? candidateInputNHWCHalf.data() : legacyInputHalf.data(), legacyInputBytes, buffers[4], "legacy packed-GEMM input") ||
             !upload(legacyFilterHalf.data(), legacyFilterBytes, buffers[5], "legacy packed-GEMM filter"))
            return false;
        }
        else if(!upload(inputSource.data(), legacyInputBytes, buffers[4], "legacy packed-GEMM input") ||
                !upload(packedFilter.data(), legacyFilterBytes, buffers[5], "legacy packed-GEMM filter"))
          return false;
        dualGemmFixtureKey = fixtureKey;
      }
      const vector<float>& reference = dualGemmReference;

      if(!ensureDescriptorPool(13, 5, result, error) || !ensureQueryPool(static_cast<uint32_t>(4 * timedRuns), result, error) ||
         !ensureCommandResources(result, error))
        return false;
      const auto allocateDescriptorSet = [&](const Pipeline* pipeline, VkDescriptorSet& descriptorSet) {
        VkDescriptorSetAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = resources.descriptorPool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &pipeline->descriptorSetLayout;
        result = vkAllocateDescriptorSets(device->device, &info, &descriptorSet);
        if(result != VK_SUCCESS)
          error = "could not allocate Transformer dual-GEMM descriptor set: " + vk_helper::vkErrorToString(result);
        return result == VK_SUCCESS;
      };
      VkDescriptorSet fusedDescriptorSet = VK_NULL_HANDLE;
      VkDescriptorSet nchwToNhwcDescriptorSet = VK_NULL_HANDLE;
      VkDescriptorSet nhwcToNchwDescriptorSet = VK_NULL_HANDLE;
      VkDescriptorSet legacyGemmDescriptorSet = VK_NULL_HANDLE;
      VkDescriptorSet swigluDescriptorSet = VK_NULL_HANDLE;
      if(!allocateDescriptorSet(fusedPipeline, fusedDescriptorSet) ||
         !allocateDescriptorSet(nchwToNhwcPipeline, nchwToNhwcDescriptorSet) ||
         !allocateDescriptorSet(nhwcToNchwPipeline, nhwcToNchwDescriptorSet) ||
         !allocateDescriptorSet(legacyGemmPipeline, legacyGemmDescriptorSet) ||
         !allocateDescriptorSet(swigluPipeline, swigluDescriptorSet))
        return false;

      VkCommandBuffer commandBuffer = resources.commandBuffer;
      VkFence fence = resources.fence;
      const auto cleanup = [&]() noexcept {
        if(commandBuffer != VK_NULL_HANDLE)
          vkResetCommandBuffer(commandBuffer, 0);
      };
      const auto cleanupGuard = makeScopeGuard(cleanup);
      result = vk_helper::beginCommandBuffer(commandBuffer);
      if(result != VK_SUCCESS) {
        error = "could not begin Transformer dual-GEMM measurement: " + vk_helper::vkErrorToString(result);
        return false;
      }
      const auto recordFused = [&]() {
        vkcompute::doTransformerDualGemmSwiGLU(
          nullptr, device, candidateConfig, fusedPipeline, commandBuffer, fusedDescriptorSet,
          buffers[2], buffers[1], buffers[3],
          batchSize, hwSize, static_cast<int>(logicalHW),
          ffnSize, cSize, packedOCSize, inputChannelStride, outputChannelStride, &result
        );
      };
      const auto recordLegacy = [&]() {
        if(legacyNHWC) {
          const auto& p = legacyConfig.transformerHgemmNHWC;
          const auto& gemmParams = p.isValid() && hwSize % p.MWG == 0 &&
            cSize % p.KWG == 0 && packedOCSize % p.NWG == 0
            ? p : legacyConfig.hgemmCooperativeMatrixNHWC;
          if(packedOutputStride != packedOCSize)
            vkcompute::clearBufferForCompute(commandBuffer, buffers[6]);
          vkcompute::doHgemmCooperativeMatrixNHWC(
            nullptr, device, legacyConfig, legacyGemmPipeline, commandBuffer, legacyGemmDescriptorSet,
            buffers[4], buffers[5], buffers[6], batchSize, hwSize, packedOCSize, cSize,
            inputChannelStride, packedOutputStride, gemmParams, &result
          );
        }
        else if(legacyNCHW) {
          vkcompute::doHgemmCooperativeMatrixNCHW(
            nullptr, device, legacyConfig, legacyGemmPipeline, commandBuffer, legacyGemmDescriptorSet,
            buffers[4], buffers[5], buffers[6], batchSize, hwSize, packedOCSize, cSize, &result
          );
        }
        else {
          vkcompute::xgemmStridedBatchedNN(
            nullptr, device, legacyConfig, legacyGemmPipeline, commandBuffer, legacyGemmDescriptorSet,
            hwSize, packedOCSize, cSize,
            cSize * hwSize, 0, packedOCSize * hwSize,
            buffers[4], buffers[5], buffers[6], batchSize, &result
          );
        }
        if(result != VK_SUCCESS)
          return;
        vkcompute::doSwiGLU(
          nullptr, device, commandBuffer, swigluDescriptorSet, *swigluPipeline, legacyConfig,
          buffers[6], buffers[6], buffers[7], batchSize * ffnSize * hwSize,
          (legacyNHWC ? packedOutputStride : packedOCSize) * hwSize, ffnSize * hwSize,
          (legacyNHWC ? outputChannelStride : ffnSize) * hwSize,
          legacyNHWC, ffnSize, packedOutputStride, outputChannelStride
        );
      };
      for(int warmup = 0; warmup < 8; warmup++) {
        recordFused();
        vk_helper::barrierCommandBuffer(commandBuffer);
        recordLegacy();
        vk_helper::barrierCommandBuffer(commandBuffer);
        if(result != VK_SUCCESS) {
          error = "could not warm up FFN comparison: " + vk_helper::vkErrorToString(result);
          return false;
        }
      }
      vkCmdResetQueryPool(commandBuffer, resources.queryPool, 0, static_cast<uint32_t>(4 * timedRuns));
      for(size_t repeat = 0; repeat < timedRuns; repeat++) {
        const uint32_t queryStart = static_cast<uint32_t>(4 * repeat);
        const auto timed = [&](bool fused) {
          uint32_t query = queryStart + (fused ? 0 : 2);
          vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            resources.queryPool,
            query
          );
          if(fused)
            recordFused();
          else recordLegacy();
          vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            resources.queryPool,
            query + 1
          );
          vk_helper::barrierCommandBuffer(commandBuffer);
        };
        timed(!reverseOrder);
        timed(reverseOrder);
        if(result != VK_SUCCESS) {
          error = "could not dispatch timed FFN comparison: " + vk_helper::vkErrorToString(result);
          return false;
        }
      }
      result = vk_helper::endCommandBuffer(commandBuffer);
      if(result != VK_SUCCESS) {
        error = "could not end Transformer dual-GEMM measurement: " + vk_helper::vkErrorToString(result);
        return false;
      }
      result = vkResetFences(device->device, 1, &fence);
      if(result != VK_SUCCESS) {
        error = "could not reset Transformer dual-GEMM measurement fence: " + vk_helper::vkErrorToString(result);
        return false;
      }
      result = vk_helper::submitCommandBuffers(device, {commandBuffer}, fence);
      if(result != VK_SUCCESS) {
        error = "could not submit Transformer dual-GEMM measurement: " + vk_helper::vkErrorToString(result);
        return false;
      }
      result = vk_helper::waitForFence(device, fence, "fused FFN vkWaitForFences");
      if(result != VK_SUCCESS) {
        error = "could not wait for Transformer dual-GEMM measurement: " + vk_helper::vkErrorToString(result);
        return false;
      }

      vector<uint64_t> timestamps(4 * timedRuns, 0);
      result = vkGetQueryPoolResults(
        device->device, resources.queryPool, 0, static_cast<uint32_t>(timestamps.size()),
        timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT
      );
      if(result != VK_SUCCESS) {
        error = "could not read Transformer dual-GEMM timestamps: " + vk_helper::vkErrorToString(result);
        return false;
      }
      double fusedSeconds = 0.0;
      double legacySeconds = 0.0;
      for(size_t repeat = 0; repeat < timedRuns; repeat++) {
        const uint64_t fusedStart = timestamps[4 * repeat];
        const uint64_t fusedEnd = timestamps[4 * repeat + 1];
        const uint64_t legacyStart = timestamps[4 * repeat + 2];
        const uint64_t legacyEnd = timestamps[4 * repeat + 3];
        if(fusedEnd <= fusedStart || legacyEnd <= legacyStart) {
          error = "could not read valid Transformer dual-GEMM timestamps";
          return false;
        }
        fusedSeconds += (fusedEnd - fusedStart) * timestampPeriod * 1e-9;
        legacySeconds += (legacyEnd - legacyStart) * timestampPeriod * 1e-9;
      }
      if(fusedSeconds <= 0.0 || legacySeconds <= 0.0) {
        error = "Transformer dual-GEMM timing returned a non-positive duration";
        return false;
      }
      fusedCallsPerSecond = timedRuns / fusedSeconds;
      legacyCallsPerSecond = timedRuns / legacySeconds;

      vector<half_t> outputHalf(candidateOutputNHWCElements);
      vk_helper::copyDeviceBufferToHost(
        device, buffers[3], candidateOutputNHWCBytes, outputHalf.data(), true, &result, commandPool
      );
      if(result != VK_SUCCESS) {
        error = "could not read Transformer dual-GEMM output: " + vk_helper::vkErrorToString(result);
        return false;
      }
      readback.resize(outputElements);
      for(int batch = 0; batch < batchSize; batch++)
        for(int channel = 0; channel < ffnSize; channel++)
          for(int hw = 0; hw < hwSize; hw++)
            readback[(static_cast<size_t>(batch) * ffnSize + channel) * hwSize + hw] =
              half_float::half_cast<float>(
                outputHalf[(static_cast<size_t>(batch) * hwSize + hw) * outputChannelStride + channel]
              );
      errorProp = VulkanTuner::computeErrorProp(reference, readback);
      if(legacyNHWC) {
        vk_helper::copyDeviceBufferToHost(
          device, buffers[7], legacyOutputBytes, outputHalf.data(), true, &result, commandPool
        );
        if(result != VK_SUCCESS) {
          error = "could not read NHWC packed-GEMM plus SwiGLU output";
          return false;
        }
        vector<float> legacyReadback(outputElements);
        for(int b = 0; b < batchSize; b++)
          for(int c = 0; c < ffnSize; c++)
            for(int hw = 0; hw < hwSize; hw++)
          legacyReadback[(static_cast<size_t>(b) * ffnSize + c) * hwSize + hw] =
            static_cast<float>(outputHalf[(static_cast<size_t>(b) * hwSize + hw) * outputChannelStride + c]);
        errorProp = std::max(errorProp, VulkanTuner::computeErrorProp(reference, legacyReadback));
      }
      measurementComplete = true;
      return true;
    }


    bool measureTransformerResidualRMSNorm(
      const VulkanTuneParams& config,
      const TuningContext& context,
      const Pipeline& fused,
      const Pipeline& norm,
      const Pipeline& add,
      int batchSize,
      int hwSize,
      int maskPattern,
      string& error,
      double* weightedSeconds = nullptr,
      int timedRuns = 5,
      bool reverseOrder = false
    ) {
      const int warmup = weightedSeconds != nullptr ? 8 : 0;
      const int repeats = warmup + timedRuns;
      const uint32_t queryCount = 4 * repeats;
      const int channels = VulkanTuner::getTransformerFFNInputChannelsForTuning(context.modelInfo);
      const size_t elements = static_cast<size_t>(batchSize) * hwSize * channels;
      if(!isUsable() || channels <= 0 || elements > UINT32_MAX) {
        error = "invalid residual RMSNorm measurement dimensions or timestamps";
        return false;
      }
      VkResult result = VK_SUCCESS;
      ReusableResources resources(device, commandPool);
      const auto deviceLossGuard = makeScopeGuard([&]() { device->recordDeviceLoss(result, "residual RMSNorm measurement"); });
      activeResources = &resources;
      const auto resourceGuard = makeScopeGuard([&]() { activeResources = nullptr; });
      auto& buffers = resources.tuningBuffers;
      const VkDeviceSize bytes = elements * sizeof(half_t);
      if(!ensureBuffers(buffers, 9, std::max<VkDeviceSize>(bytes, channels * sizeof(float)), result, error, "residual RMSNorm buffer") ||
         !ensureDescriptorPool(12, 3, result, error) || !ensureQueryPool(queryCount, result, error) ||
         !ensureCommandResources(result, error))
        return false;
      vector<half_t> trunk(elements), delta(elements), mask(static_cast<size_t>(batchSize) * hwSize);
      vector<float> gamma(channels), beta(channels, 0.0f), cpuTrunk(elements), cpuOutput(elements);
      Rand rand("TransformerResidualRMSNorm");
      for(size_t i = 0; i < elements; i++) {
        trunk[i] = half_t(2.0 * rand.nextDouble() - 1.0);
        delta[i] = half_t(0.2 * rand.nextDouble() - 0.1);
        cpuTrunk[i] = static_cast<float>(half_t(float(trunk[i]) + float(delta[i])));
      }
      for(int c = 0; c < channels; c++) gamma[c] = 0.5f + static_cast<float>(rand.nextDouble());
        for(int b = 0; b < batchSize; b++)
          for(int xy = 0; xy < hwSize; xy++) {
            const float m = xy >= context.nnXLen * context.nnYLen || maskPattern == 1 ? 0.0f :
              maskPattern == 2 && xy % 5 == 0 ? 0.0f : maskPattern == 3 ? 0.5f : 1.0f;
            mask[b * hwSize + xy] = half_t(m);
            const size_t offset = (static_cast<size_t>(b) * hwSize + xy) * channels;
            float sum = 0.0f;
            for(int c = 0; c < channels; c++) {
              const float val = cpuTrunk[offset + c] * m;
              sum += val * val;
            }
            const float rms = 1.0f / sqrtf(sum / channels + 1e-6f);
            for(int c = 0; c < channels; c++)
              cpuOutput[offset + c] = float(half_t(cpuTrunk[offset + c] * rms * gamma[c] * m));
          }
      const auto upload = [&](
        int index,
        const void* source,
        VkDeviceSize size
      ) {
        vk_helper::copyHostToDeviceBuffer(device, source, buffers[index], size, false, &result, commandPool);
        return result == VK_SUCCESS;
      };
      if(!upload(0, trunk.data(), bytes) || !upload(1, delta.data(), bytes) ||
         !upload(
           6,
           gamma.data(),
           channels * sizeof(float)
         ) || !upload(7, beta.data(), channels * sizeof(float)) ||
         !upload(8, mask.data(), mask.size() * sizeof(half_t))) {
        error = "could not upload residual RMSNorm inputs";
        return false;
      }
      VkDescriptorSet sets[3] = {};
      const Pipeline* pipelines[] = {&fused, &norm, &add};
      for(int i = 0; i < 3; i++) {
        VkDescriptorSetAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = resources.descriptorPool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &pipelines[i]->descriptorSetLayout;
        if(vkAllocateDescriptorSets(device->device, &info, &sets[i]) != VK_SUCCESS) {
          error = "could not allocate residual RMSNorm descriptors";
          return false;
        }
      }
      const auto cb = resources.commandBuffer;
      const auto cbGuard = makeScopeGuard([&]() { vkResetCommandBuffer(cb, 0); });
      result = vk_helper::beginCommandBuffer(cb);
      if(result != VK_SUCCESS) { error = "could not begin residual RMSNorm commands";
      return false; }
      vkCmdResetQueryPool(cb, resources.queryPool, 0, queryCount);
      double seconds[2] = {};
      for(int repeat = 0; repeat < repeats; repeat++)
        for(int order = 0; order < 2; order++) {
          const int path = reverseOrder ? 1 - order : order;
          // Restore BOTH mutable inputs before every sample, outside timestamps.
          const int input = path == 0 ? 2 : 4;
          const int output = path == 0 ? 3 : 5;
          vk_helper::barrierCommandBuffer(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
          VkBufferCopy copy = {0, 0, bytes};
          vkCmdCopyBuffer(cb, buffers[0]->buffer, buffers[input]->buffer, 1, &copy);
          vkCmdCopyBuffer(cb, buffers[1]->buffer, buffers[output]->buffer, 1, &copy);
          vk_helper::barrierCommandBuffer(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
          const uint32_t query = 4 * repeat + 2 * path;
          if(weightedSeconds == nullptr || path == 0)
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, resources.queryPool, query);
          if(path == 1) {
            const auto writes = vector<WriteDescriptorSet>{
              vk_helper::writeDescriptorSetBuffer(sets[2], 0, buffers[input]),
              vk_helper::writeDescriptorSetBuffer(sets[2], 1, buffers[output])
            };
            result = vk_helper::updateDescriptorSets(device, writes);
            const vk_shader::push::AddPointWiseParams params = {static_cast<uint32_t>(elements)};
            vkcompute::dispatchPipeline(nullptr, device, &add, cb, sets[2], &params, sizeof(params),
              static_cast<uint32_t>((elements + add.localSizeX * config.pointwise.ELTS_PER_THREAD - 1) /
                                    (add.localSizeX * config.pointwise.ELTS_PER_THREAD)), 1, 1, "ADD_POINTWISE_FP32");
            vk_helper::barrierCommandBufferForBuffer(cb, buffers[input]);
          }
          if(weightedSeconds != nullptr && path == 1)
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, resources.queryPool, query);
          vkcompute::transformerRMSNorm(nullptr, device, pipelines[path], cb, sets[path],
            buffers[input], buffers[output], buffers[6], buffers[7], buffers[8],
            batchSize, channels, hwSize, hwSize, context.nnXLen * context.nnYLen, channels,
            1e-6f, config.rmsNorm, &result, path == 0);
          if(result != VK_SUCCESS) { error = "could not dispatch residual RMSNorm";
          return false; }
          vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, resources.queryPool, query + 1);
        }
      result = vk_helper::endCommandBuffer(cb);
      if(result == VK_SUCCESS)
        result = vkResetFences(device->device, 1, &resources.fence);
      if(result == VK_SUCCESS)
        result = vk_helper::submitCommandBuffers(device, {cb}, resources.fence);
      if(result == VK_SUCCESS)
        result = vk_helper::waitForFence(device, resources.fence, "residual RMSNorm vkWaitForFences");
      vector<uint64_t> timestamps(queryCount);
      if(result == VK_SUCCESS)
        result = vkGetQueryPoolResults(device->device, resources.queryPool, 0, queryCount,
        timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
      if(result != VK_SUCCESS) { error = "could not execute/read residual RMSNorm measurement";
      return false; }
      for(int repeat = warmup; repeat < repeats; repeat++)
        for(int path = 0; path < 2; path++) {
          const int query = 4 * repeat + 2 * path;
          if(timestamps[query + 1] <= timestamps[query]) { error = "invalid residual RMSNorm timestamps";
          return false; }
          seconds[path] += (timestamps[query + 1] - timestamps[query]) * timestampPeriod * 1e-9;
        }
      vector<half_t> values(elements);
      vector<vector<float>> outputs(4, vector<float>(elements));
      for(int i = 0; i < 4; i++) {
        vk_helper::copyDeviceBufferToHost(device, buffers[i + 2], bytes, values.data(), true, &result, commandPool);
        if(result != VK_SUCCESS) { error = "could not read residual RMSNorm outputs";
        return false; }
        for(size_t j = 0; j < elements; j++) outputs[i][j] = float(values[j]);
      }
      const double cpuError = VulkanTuner::computeErrorProp(cpuOutput, outputs[1]);
      if(outputs[0] != cpuTrunk || outputs[2] != cpuTrunk || outputs[1] != outputs[3] ||
         !isfinite(cpuError) || cpuError > 0.002) {
        error = "residual RMSNorm trunk, baseline output, or CPU reference mismatch";
        return false;
      }
      if(weightedSeconds != nullptr)
        *weightedSeconds = (context.modelInfo.transformerFusedNormCount * seconds[0] +
          context.modelInfo.transformerPlainNormCount * seconds[1]) / timedRuns;
      if(weightedSeconds == nullptr && context.logger != nullptr)
        context.logger->write(
        "Residual RMSNorm B=" + to_string(batchSize) + " S=" + to_string(hwSize) + " C=" + to_string(channels) +
        " mask=" + to_string(maskPattern) + " fused_us=" + to_string(seconds[0] * 1e6 / timedRuns) +
        " add+norm_us=" + to_string(seconds[1] * 1e6 / timedRuns) + " CPUError=" + to_string(cpuError) +
        " trunk/output match=true; inputs restored outside timing"
      );
      return true;
    }

   private:
    struct ReusableResources {
      ReusableResources(const VulkanDevice* device, VkCommandPool commandPool_)
      : device(device), commandPool(commandPool_) {}

      ~ReusableResources() {
        if(commandBuffer != VK_NULL_HANDLE)
          vkFreeCommandBuffers(device->device, commandPool, 1, &commandBuffer);
        if(fence != VK_NULL_HANDLE)
          vkDestroyFence(device->device, fence, nullptr);
        if(queryPool != VK_NULL_HANDLE)
          vkDestroyQueryPool(device->device, queryPool, nullptr);
        if(descriptorPool != VK_NULL_HANDLE)
          vkDestroyDescriptorPool(device->device, descriptorPool, nullptr);
        for(VulkanBuffer* buffer: tuningBuffers)
          vk_helper::releaseVulkanBuffer(device, buffer);
        for(VulkanBuffer* buffer: uploadStagingBuffers)
          vk_helper::releaseVulkanBuffer(device, buffer);
        vk_helper::releaseVulkanBuffer(device, pointwiseAccumulatorInitialBuffer);
        vk_helper::releaseVulkanBuffer(device, attentionReadbackBuffer);
        for(VulkanBuffer* buffer: pointwiseValidationBuffers)
          vk_helper::releaseVulkanBuffer(device, buffer);
        for(VulkanBuffer* buffer: winogradValidationBuffers)
          vk_helper::releaseVulkanBuffer(device, buffer);
        for(VulkanBuffer* buffer: gpoolValidationBuffers)
          vk_helper::releaseVulkanBuffer(device, buffer);
        for(VulkanBuffer* buffer: attentionValidationBuffers)
          vk_helper::releaseVulkanBuffer(device, buffer);
        for(VulkanBuffer* buffer: gemmValidationBuffers)
          vk_helper::releaseVulkanBuffer(device, buffer);
        for(VulkanBuffer* buffer: spatialValidationBuffers)
          vk_helper::releaseVulkanBuffer(device, buffer);
      }

      const VulkanDevice* device;
      VkCommandPool commandPool;
      vector<VulkanBuffer*> tuningBuffers;
      vector<VulkanBuffer*> uploadStagingBuffers;
      VulkanBuffer* pointwiseAccumulatorInitialBuffer = nullptr;
      VulkanBuffer* attentionReadbackBuffer = nullptr;
      vector<VulkanBuffer*> pointwiseValidationBuffers;
      vector<VulkanBuffer*> winogradValidationBuffers;
      vector<VulkanBuffer*> gpoolValidationBuffers;
      vector<VulkanBuffer*> attentionValidationBuffers;
      vector<VulkanBuffer*> gemmValidationBuffers;
      vector<VulkanBuffer*> spatialValidationBuffers;
      VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
      uint32_t descriptorCount = 0;
      uint32_t maxSets = 0;
      VkQueryPool queryPool = VK_NULL_HANDLE;
      uint32_t queryCount = 0;
      VkFence fence = VK_NULL_HANDLE;
      VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    };

    ReusableResources* activeResources = nullptr;
    // Used only by the timer scoped to one fused-FFN tuning pass.
    unique_ptr<ReusableResources> dualGemmResources;
    vector<int> dualGemmFixtureKey;
    vector<float> dualGemmReference;

    bool ensureBuffer(
      VulkanBuffer*& buffer,
      VkDeviceSize requiredBytes,
      VkResult& result,
      string& error,
      const string& description
    ) {
      if(buffer != nullptr && buffer->requestedSize >= requiredBytes)
        return true;
      if(buffer != nullptr) {
        vk_helper::releaseVulkanBuffer(device, buffer);
        buffer = nullptr;
      }
      buffer = vk_helper::createDeviceBuffer(device, requiredBytes, false, &result);
      if(result != VK_SUCCESS || buffer == nullptr) {
        error = "could not allocate reusable " + description + ": " + vk_helper::vkErrorToString(result);
        return false;
      }
      return true;
    }

    bool ensureBuffers(
      vector<VulkanBuffer*>& buffers,
      size_t count,
      VkDeviceSize requiredBytes,
      VkResult& result,
      string& error,
      const string& description
    ) {
      buffers.resize(std::max(buffers.size(), count), nullptr);
      for(size_t i = 0; i < count; i++) {
        if(!ensureBuffer(buffers[i], requiredBytes, result, error, description))
          return false;
      }
      return true;
    }

    bool ensureStagingBuffer(
      VulkanBuffer*& buffer,
      VkDeviceSize requiredBytes,
      VkResult& result,
      string& error,
      const string& description
    ) {
      if(buffer != nullptr && buffer->requestedSize >= requiredBytes)
        return true;
      if(buffer != nullptr) {
        vk_helper::releaseVulkanBuffer(device, buffer);
        buffer = nullptr;
      }
      buffer = vk_helper::createStagingBuffer(device, static_cast<size_t>(requiredBytes), &result);
      if(result != VK_SUCCESS || buffer == nullptr) {
        error = "could not allocate reusable " + description + " staging buffer: " + vk_helper::vkErrorToString(result);
        return false;
      }
      return true;
    }

    bool ensureReadbackBuffer(
      VulkanBuffer*& buffer,
      VkDeviceSize requiredBytes,
      VkResult& result,
      string& error,
      const string& description
    ) {
      if(buffer != nullptr && buffer->requestedSize >= requiredBytes)
        return true;
      if(buffer != nullptr) {
        vk_helper::releaseVulkanBuffer(device, buffer);
        buffer = nullptr;
      }
      buffer = vk_helper::createReadbackBuffer(device, requiredBytes, &result);
      if(result != VK_SUCCESS || buffer == nullptr) {
        error = "could not allocate reusable " + description + ": " + vk_helper::vkErrorToString(result);
        return false;
      }
      return true;
    }

    bool readAttentionValidationBuffers(
      const vector<VulkanBuffer*>& validationBuffers,
      VkDeviceSize copyBytes,
      const vector<size_t>& outputCounts,
      bool useFP16,
      vector<float>& readback,
      VkResult& result,
      string& error
    ) {
      const VkDeviceSize stride = vk_helper::roundUpToMultiple(copyBytes, VkDeviceSize(4));
      const VkDeviceSize totalBytes = stride * validationBuffers.size();
      if(!ensureReadbackBuffer(activeResources->attentionReadbackBuffer, totalBytes, result, error, "attention readback buffer"))
        return false;

      result = vkResetCommandBuffer(activeResources->commandBuffer, 0);
      if(result != VK_SUCCESS) {
        error = "could not reset attention readback command buffer: " + vk_helper::vkErrorToString(result);
        return false;
      }
      result = vk_helper::beginCommandBuffer(activeResources->commandBuffer);
      if(result != VK_SUCCESS) {
        error = "could not begin attention readback command buffer: " + vk_helper::vkErrorToString(result);
        return false;
      }
      vkCmdFillBuffer(
        activeResources->commandBuffer,
        activeResources->attentionReadbackBuffer->buffer,
        0,
        VK_WHOLE_SIZE,
        0
      );
      vk_helper::barrierCommandBufferForBuffer(
        activeResources->commandBuffer, activeResources->attentionReadbackBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT
      );
      for(size_t i = 0; i < validationBuffers.size(); i++) {
        VkBufferCopy copyRegion = {};
        copyRegion.size = copyBytes;
        copyRegion.dstOffset = stride * i;
        vkCmdCopyBuffer(
          activeResources->commandBuffer,
          validationBuffers[i]->buffer,
          activeResources->attentionReadbackBuffer->buffer,
          1,
          &copyRegion
        );
      }
      vk_helper::barrierCommandBufferForBuffer(
        activeResources->commandBuffer, activeResources->attentionReadbackBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT
      );
      result = vk_helper::endCommandBuffer(activeResources->commandBuffer);
      if(result != VK_SUCCESS) {
        error = "could not end attention readback command buffer: " + vk_helper::vkErrorToString(result);
        return false;
      }
      result = vkResetFences(device->device, 1, &activeResources->fence);
      if(result != VK_SUCCESS) {
        error = "could not reset attention readback fence: " + vk_helper::vkErrorToString(result);
        return false;
      }
      result = vk_helper::submitCommandBuffers(
        device,
        {activeResources->commandBuffer},
        activeResources->fence
      );
      if(result != VK_SUCCESS) {
        error = "could not submit attention readback command buffer: " + vk_helper::vkErrorToString(result);
        return false;
      }
      result = vk_helper::waitForFence(device, activeResources->fence, "attention readback vkWaitForFences");
      if(result != VK_SUCCESS) {
        error = "could not wait for attention readback: " + vk_helper::vkErrorToString(result);
        return false;
      }

      void* mappedData = nullptr;
      result = vmaMapMemory(
        device->allocator,
        activeResources->attentionReadbackBuffer->allocation,
        &mappedData
      );
      if(result != VK_SUCCESS) {
        error = "could not map attention readback buffer: " + vk_helper::vkErrorToString(result);
        return false;
      }
      result = vmaInvalidateAllocation(device->allocator, activeResources->attentionReadbackBuffer->allocation, 0, totalBytes);
      if(result != VK_SUCCESS) {
        vmaUnmapMemory(device->allocator, activeResources->attentionReadbackBuffer->allocation);
        error = "could not invalidate attention readback buffer: " + vk_helper::vkErrorToString(result);
        return false;
      }
      readback.clear();
      readback.reserve(std::accumulate(outputCounts.begin(), outputCounts.end(), size_t(0)));
      const unsigned char* raw = static_cast<const unsigned char*>(mappedData);
      for(size_t i = 0; i < validationBuffers.size(); i++) {
        const void* source = raw + stride * i;
        if(useFP16) {
          const half_t* halves = static_cast<const half_t*>(source);
          for(size_t j = 0; j < outputCounts[i]; j++)
            readback.push_back(half_float::half_cast<float>(halves[j]));
        }
        else {
          const float* floats = static_cast<const float*>(source);
          readback.insert(readback.end(), floats, floats + outputCounts[i]);
        }
      }
      vmaUnmapMemory(device->allocator, activeResources->attentionReadbackBuffer->allocation);
      return true;
    }

    bool ensureDescriptorPool(
      uint32_t descriptorCount,
      uint32_t maxSets,
      VkResult& result,
      string& error
    ) {
      if(activeResources->descriptorPool == VK_NULL_HANDLE || activeResources->descriptorCount < descriptorCount || activeResources->maxSets < maxSets) {
        if(activeResources->descriptorPool != VK_NULL_HANDLE)
          vkDestroyDescriptorPool(device->device, activeResources->descriptorPool, nullptr);
        VkDescriptorPoolSize poolSize = {};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = descriptorCount;
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = maxSets;
        result = vkCreateDescriptorPool(device->device, &poolInfo, nullptr, &activeResources->descriptorPool);
        if(result != VK_SUCCESS) {
          error = "could not create tuning descriptor pool: " + vk_helper::vkErrorToString(result);
          activeResources->descriptorPool = VK_NULL_HANDLE;
          return false;
        }
        activeResources->descriptorCount = descriptorCount;
        activeResources->maxSets = maxSets;
        return true;
      }
      result = vkResetDescriptorPool(device->device, activeResources->descriptorPool, 0);
      if(result != VK_SUCCESS) {
        error = "could not reset tuning descriptor pool: " + vk_helper::vkErrorToString(result);
        return false;
      }
      return true;
    }

    bool ensureQueryPool(
      uint32_t queryCount,
      VkResult& result,
      string& error
    ) {
      if(activeResources->queryPool != VK_NULL_HANDLE && activeResources->queryCount >= queryCount)
        return true;
      if(activeResources->queryPool != VK_NULL_HANDLE)
        vkDestroyQueryPool(device->device, activeResources->queryPool, nullptr);
      VkQueryPoolCreateInfo queryPoolInfo = {};
      queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
      queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
      queryPoolInfo.queryCount = queryCount;
      result = vkCreateQueryPool(device->device, &queryPoolInfo, nullptr, &activeResources->queryPool);
      if(result != VK_SUCCESS) {
        error = "could not create tuning query pool: " + vk_helper::vkErrorToString(result);
        activeResources->queryPool = VK_NULL_HANDLE;
        activeResources->queryCount = 0;
        return false;
      }
      activeResources->queryCount = queryCount;
      return true;
    }

    bool ensureCommandResources(VkResult& result, string& error) {
      if(activeResources->fence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        result = vkCreateFence(device->device, &fenceInfo, nullptr, &activeResources->fence);
        if(result != VK_SUCCESS) {
          error = "could not create tuning fence: " + vk_helper::vkErrorToString(result);
          return false;
        }
      }
      if(activeResources->commandBuffer == VK_NULL_HANDLE) {
        activeResources->commandBuffer = vk_helper::allocateCommandBuffer(device, &result, commandPool);
        if(result != VK_SUCCESS) {
          error = "could not allocate tuning command buffer: " + vk_helper::vkErrorToString(result);
          activeResources->commandBuffer = VK_NULL_HANDLE;
          return false;
        }
      }
      result = vkResetCommandBuffer(activeResources->commandBuffer, 0);
      if(result != VK_SUCCESS) {
        error = "could not reset tuning command buffer: " + vk_helper::vkErrorToString(result);
        return false;
      }
      return true;
    }

    const VulkanDevice* device;
    float timestampPeriod;
    VkCommandPool commandPool = VK_NULL_HANDLE;
  };

  class VulkanDummyThread {
   public:
    VulkanDummyThread(const VulkanDevice* device, Logger* logger)
    : device(device), logger(logger) {}

    ~VulkanDummyThread() {
      stopAndJoin();
    }

    void start() {
      worker = thread([this]() { run(); });
      unique_lock<mutex> lock(stateMutex);
      stateCondition.wait(lock, [this]() { return initializedOrDead; });
    }

    void stopAndJoin() {
      shouldStop.store(true);
      if(worker.joinable())
        worker.join();
    }

   private:
    void reportFailure(const string& message) const {
      const string fullMessage = "WARNING: Dummy thread to load the GPU while tuning failed\n" + message;
      if(logger != nullptr)
        logger->write(fullMessage);
      if(logger == nullptr || (!logger->isLoggingToStdout() && !logger->isLoggingToStderr()))
        cerr << fullMessage << endl;
    }

    void signalInitializedOrDead() {
      {
        lock_guard<mutex> lock(stateMutex);
        initializedOrDead = true;
      }
      stateCondition.notify_one();
    }

    void run() {
      if(logger != nullptr)
        logger->write("Dummy tuning thread starting with OpenCL-equivalent mixed workload");

      if(device->dummyQueue == VK_NULL_HANDLE) {
        reportFailure("no dedicated Vulkan queue is available");
        signalInitializedOrDead();
        return;
      }

      VkCommandPool commandPool = VK_NULL_HANDLE;
      VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
      VkFence fence = VK_NULL_HANDLE;
      VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
      VulkanBuffer* matrixA = nullptr;
      VulkanBuffer* matrixB = nullptr;
      VulkanBuffer* matrixC = nullptr;
      VulkanBuffer* matrixD = nullptr;
      VulkanBuffer* buffer = nullptr;
      VulkanBuffer* buffer2 = nullptr;
      VulkanBuffer* readbackBuffer = nullptr;
      unique_ptr<vk_shader::ComputePipelines> pipelines;

      const auto cleanup = [&]() {
        if(commandBuffer != VK_NULL_HANDLE)
          vkFreeCommandBuffers(device->device, commandPool, 1, &commandBuffer);
        if(fence != VK_NULL_HANDLE)
          vkDestroyFence(device->device, fence, nullptr);
        if(descriptorPool != VK_NULL_HANDLE)
          vkDestroyDescriptorPool(device->device, descriptorPool, nullptr);
        if(commandPool != VK_NULL_HANDLE)
          vkDestroyCommandPool(device->device, commandPool, nullptr);
        vk_helper::releaseVulkanBuffer(device, matrixA);
        vk_helper::releaseVulkanBuffer(device, matrixB);
        vk_helper::releaseVulkanBuffer(device, matrixC);
        vk_helper::releaseVulkanBuffer(device, matrixD);
        vk_helper::releaseVulkanBuffer(device, buffer);
        vk_helper::releaseVulkanBuffer(device, buffer2);
        vk_helper::releaseVulkanBuffer(device, readbackBuffer);
      };
      ScopeGuard cleanupGuard(std::move(cleanup));

      try {
        constexpr int batchSize = 1;
        constexpr int mSize = 97;
        constexpr int nSize = 151;
        constexpr int kSize = 151;
        constexpr size_t inputElements = static_cast<size_t>(mSize) * kSize;
        constexpr size_t otherElements = static_cast<size_t>(kSize) * nSize;
        constexpr size_t outputElements = static_cast<size_t>(mSize) * nSize;
        static_assert(inputElements == outputElements, "dummy XGEMM buffer swap requires N == K");

        VkResult result = VK_SUCCESS;
        const auto deviceLossGuard = makeScopeGuard([&]() { device->recordDeviceLoss(result, "dummy tuning work"); });
        VkCommandPoolCreateInfo commandPoolInfo = {};
        commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolInfo.queueFamilyIndex = device->queueFamilyIndex;
        commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        result = vkCreateCommandPool(device->device, &commandPoolInfo, nullptr, &commandPool);
        if(result != VK_SUCCESS)
          throw StringError("could not create dummy command pool: " + vk_helper::vkErrorToString(result));

        VkCommandBufferAllocateInfo commandBufferInfo = {};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferInfo.commandPool = commandPool;
        commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferInfo.commandBufferCount = 1;
        result = vkAllocateCommandBuffers(device->device, &commandBufferInfo, &commandBuffer);
        if(result != VK_SUCCESS)
          throw StringError("could not allocate dummy command buffer: " + vk_helper::vkErrorToString(result));

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        result = vkCreateFence(device->device, &fenceInfo, nullptr, &fence);
        if(result != VK_SUCCESS)
          throw StringError("could not create dummy fence: " + vk_helper::vkErrorToString(result));

        const VkDescriptorPoolSize descriptorPoolSize = {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5
        };
        descriptorPool = vk_helper::createDescriptorPool(
          device, {descriptorPoolSize}, 2, &result
        );
        if(result != VK_SUCCESS)
          throw StringError("could not create dummy descriptor pool: " + vk_helper::vkErrorToString(result));

        XgemmDirectTuneParams xgemmParams;
        xgemmParams.WGD = 8;
        xgemmParams.MDIMCD = 8;
        xgemmParams.NDIMCD = 8;
        xgemmParams.MDIMAD = 8;
        xgemmParams.NDIMBD = 8;
        xgemmParams.KWID = 1;
        xgemmParams.PADA = 1;
        xgemmParams.PADB = 1;
        xgemmParams.VWMD = 1;
        xgemmParams.VWND = 1;
        AddPointWiseTuneParams pointwiseParams;
        VulkanParams vulkanParams;
        pipelines = make_unique<vk_shader::ComputePipelines>(
          device->device, device->info, nullptr, false
        );
        result = pipelines->createXgemmStridedBatched(
          pipelines->xgemmStridedBatchedFp32, xgemmParams, vulkanParams
        );
        if(result != VK_SUCCESS)
          throw StringError("could not create dummy XGEMM pipeline: " + vk_helper::vkErrorToString(result));
        result = pipelines->createAddPointWise(
          pipelines->addPointWise, pointwiseParams, vulkanParams
        );
        if(result != VK_SUCCESS)
          throw StringError(
            "could not create dummy addPointWise pipeline: " + vk_helper::vkErrorToString(result)
          );

        Rand dataRand("dummyThreadData");
        const auto makeRandomVector = [&](size_t size, double scale) {
          vector<float> values(size);
          for(float& value: values)
            value = static_cast<float>(dataRand.nextDouble(-1.0, 1.0) * scale);
          return values;
        };
        const auto makeZeroVector = [&](size_t size) {
          return vector<float>(size, 0.0f);
        };
        matrixA = vk_helper::createReadOnlyBuffer(
          device, makeRandomVector(otherElements, 1.2 / kSize), false, &result, commandPool
        );
        if(result != VK_SUCCESS || matrixA == nullptr)
          throw StringError("could not create dummy matrix A: " + vk_helper::vkErrorToString(result));
        matrixB = vk_helper::createReadOnlyBuffer(
          device, makeRandomVector(otherElements, 1.2 / kSize), false, &result, commandPool
        );
        if(result != VK_SUCCESS || matrixB == nullptr)
          throw StringError("could not create dummy matrix B: " + vk_helper::vkErrorToString(result));
        matrixC = vk_helper::createReadOnlyBuffer(
          device, makeRandomVector(outputElements, 1.0), false, &result, commandPool
        );
        if(result != VK_SUCCESS || matrixC == nullptr)
          throw StringError("could not create dummy matrix C: " + vk_helper::vkErrorToString(result));
        matrixD = vk_helper::createReadOnlyBuffer(
          device, makeRandomVector(outputElements, 1.0), false, &result, commandPool
        );
        if(result != VK_SUCCESS || matrixD == nullptr)
          throw StringError("could not create dummy matrix D: " + vk_helper::vkErrorToString(result));
        buffer = vk_helper::createReadWriteBuffer(device, makeZeroVector(inputElements), false, &result, commandPool);
        if(result != VK_SUCCESS || buffer == nullptr)
          throw StringError("could not create dummy buffer: " + vk_helper::vkErrorToString(result));
        buffer2 = vk_helper::createReadWriteBuffer(device, makeZeroVector(outputElements), false, &result, commandPool);
        if(result != VK_SUCCESS || buffer2 == nullptr)
          throw StringError("could not create second dummy buffer: " + vk_helper::vkErrorToString(result));
        readbackBuffer = vk_helper::createReadbackBuffer(device, outputElements * sizeof(float), &result);
        if(result != VK_SUCCESS || readbackBuffer == nullptr)
          throw StringError("could not create dummy readback buffer: " + vk_helper::vkErrorToString(result));

        VkDescriptorSet xgemmDescriptorSet = VK_NULL_HANDLE;
        VkDescriptorSet pointwiseDescriptorSet = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo descriptorSetInfo = {};
        descriptorSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetInfo.descriptorPool = descriptorPool;
        descriptorSetInfo.descriptorSetCount = 1;
        descriptorSetInfo.pSetLayouts = &pipelines->xgemmStridedBatchedFp32.descriptorSetLayout;
        result = vkAllocateDescriptorSets(device->device, &descriptorSetInfo, &xgemmDescriptorSet);
        if(result != VK_SUCCESS)
          throw StringError(
            "could not allocate dummy XGEMM descriptor set: " + vk_helper::vkErrorToString(result)
          );
        descriptorSetInfo.pSetLayouts = &pipelines->addPointWise.descriptorSetLayout;
        result = vkAllocateDescriptorSets(device->device, &descriptorSetInfo, &pointwiseDescriptorSet);
        if(result != VK_SUCCESS)
          throw StringError(
            "could not allocate dummy pointwise descriptor set: " + vk_helper::vkErrorToString(result)
          );

        const auto updateXgemmDescriptors = [&](
          VulkanBuffer* input,
          VulkanBuffer* other,
          VulkanBuffer* output
        ) {
          return vk_helper::updateDescriptorSets(device, {
            vk_helper::writeDescriptorSetBuffer(xgemmDescriptorSet, 0, input),
            vk_helper::writeDescriptorSetBuffer(xgemmDescriptorSet, 1, other),
            vk_helper::writeDescriptorSetBuffer(xgemmDescriptorSet, 2, output)
          });
        };
        const auto updatePointwiseDescriptors = [&](VulkanBuffer* input, VulkanBuffer* other) {
          return vk_helper::updateDescriptorSets(device, {
            vk_helper::writeDescriptorSetBuffer(pointwiseDescriptorSet, 0, input),
            vk_helper::writeDescriptorSetBuffer(pointwiseDescriptorSet, 1, other)
          });
        };
        const auto addReadWriteBarrier = [&](
          VulkanBuffer* target,
          VkPipelineStageFlags dstStage,
          VkAccessFlags dstAccess
        ) {
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, target,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            dstStage, dstAccess
          );
        };

        signalInitializedOrDead();
        Rand rand("dummyThreadLoop");
        vector<float> output(outputElements, 0.0f);
        double total = 0.0;
        bool first = true;
        while(!shouldStop.load() && device->deviceLossStage.load() == nullptr) {
          int which = rand.nextInt(0, 6);
          if(first) {
            which = 4;
            first = false;
          }

          result = vkResetCommandBuffer(commandBuffer, 0);
          if(result != VK_SUCCESS) {
            reportFailure("could not reset dummy command buffer: " + vk_helper::vkErrorToString(result));
            break;
          }
          result = vk_helper::beginCommandBuffer(commandBuffer);
          if(result != VK_SUCCESS) {
            reportFailure("could not begin dummy command buffer: " + vk_helper::vkErrorToString(result));
            break;
          }

          if(which <= 3) {
            VulkanBuffer* other = (which == 0 || which == 1) ? matrixA : matrixB;
            result = updateXgemmDescriptors(buffer, other, buffer2);
            if(result != VK_SUCCESS) {
              reportFailure("could not update dummy XGEMM descriptors: " + vk_helper::vkErrorToString(result));
              break;
            }
            addReadWriteBarrier(buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            addReadWriteBarrier(other, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            addReadWriteBarrier(buffer2, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
            vkCmdBindPipeline(
              commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
              pipelines->xgemmStridedBatchedFp32.pipeline
            );
            vkCmdBindDescriptorSets(
              commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
              pipelines->xgemmStridedBatchedFp32.layout, 0, 1, &xgemmDescriptorSet, 0, nullptr
            );
            const vk_shader::push::XgemmStridedBatchedFp32Params params = {
              static_cast<uint32_t>(mSize), static_cast<uint32_t>(nSize), static_cast<uint32_t>(kSize),
              static_cast<uint32_t>(mSize), 0,
              static_cast<uint32_t>(nSize), 0,
              static_cast<uint32_t>(mSize), 0, 0
            };
            vkCmdPushConstants(
              commandBuffer, pipelines->xgemmStridedBatchedFp32.layout,
              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params
            );
            vkCmdDispatch(
              commandBuffer,
              (mSize + xgemmParams.WGD - 1) / xgemmParams.WGD,
              (nSize + xgemmParams.WGD - 1) / xgemmParams.WGD,
              batchSize
            );
          }
          else if(which == 4 || which == 5) {
            VulkanBuffer* other = which == 4 ? matrixC : matrixD;
            result = updatePointwiseDescriptors(buffer, other);
            if(result != VK_SUCCESS) {
              reportFailure("could not update dummy pointwise descriptors: " + vk_helper::vkErrorToString(result));
              break;
            }
            addReadWriteBarrier(
              buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
            );
            addReadWriteBarrier(other, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            vkCmdBindPipeline(
              commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines->addPointWise.pipeline
            );
            vkCmdBindDescriptorSets(
              commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
              pipelines->addPointWise.layout, 0, 1, &pointwiseDescriptorSet, 0, nullptr
            );
            const vk_shader::push::AddPointWiseParams params = {static_cast<uint32_t>(outputElements)};
            vkCmdPushConstants(
              commandBuffer, pipelines->addPointWise.layout,
              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params
            );
            vkCmdDispatch(
              commandBuffer,
              (params.size + pointwiseParams.ELTS_PER_THREAD * pipelines->addPointWise.localSizeX - 1) /
                (pointwiseParams.ELTS_PER_THREAD * pipelines->addPointWise.localSizeX),
              1, 1
            );
          }
          else {
            addReadWriteBarrier(buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            vk_helper::recordBufferCopy(
              commandBuffer, buffer, readbackBuffer, 0, 0, outputElements * sizeof(float)
            );
            vk_helper::barrierCommandBufferForBuffer(
              commandBuffer, readbackBuffer,
              VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
              VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT
            );
          }

          result = vk_helper::endCommandBuffer(commandBuffer);
          if(result != VK_SUCCESS) {
            reportFailure("could not end dummy command buffer: " + vk_helper::vkErrorToString(result));
            break;
          }
          result = vkResetFences(device->device, 1, &fence);
          if(result != VK_SUCCESS) {
            reportFailure("could not reset dummy fence: " + vk_helper::vkErrorToString(result));
            break;
          }
          result = vk_helper::submitCommandBuffers(device, {commandBuffer}, fence, device->dummyQueue);
          if(result != VK_SUCCESS) {
            if(result != VK_ERROR_DEVICE_LOST)
              reportFailure("could not submit dummy command buffer: " + vk_helper::vkErrorToString(result));
            break;
          }
          result = vk_helper::waitForFence(device, fence, "dummy vkWaitForFences");
          if(result != VK_SUCCESS) {
            if(result != VK_ERROR_DEVICE_LOST)
              reportFailure("could not wait for dummy command buffer: " + vk_helper::vkErrorToString(result));
            break;
          }

          if(which <= 3)
            swap(buffer, buffer2);
          else if(which > 5) {
            vk_helper::copyReadbackBufferToHost(
              device, readbackBuffer, 0, outputElements * sizeof(float), output.data(), &result
            );
            if(result != VK_SUCCESS) {
              if(result != VK_ERROR_DEVICE_LOST)
                reportFailure("could not read back dummy buffer: " + vk_helper::vkErrorToString(result));
              break;
            }
            float subTotal = 0.0f;
            for(float value: output)
              subTotal += value;
            total += static_cast<double>(subTotal);
          }
        }
        if(logger != nullptr)
          logger->write("Tuning dummy thread numeric total: " + Global::doubleToString(total));
      }
      catch(const exception& e) {
        if(device->deviceLossStage.load() == nullptr)
          reportFailure(e.what());
        signalInitializedOrDead();
      }
      catch(...) {
        reportFailure("unknown error");
        signalInitializedOrDead();
      }
    }

    const VulkanDevice* device;
    Logger* logger;
    thread worker;
    atomic<bool> shouldStop{false};
    bool initializedOrDead = false;
    mutex stateMutex;
    condition_variable stateCondition;
  };

  struct TuningConfigMeasurement {
    VulkanTuneParams config;
    double callsPerSecond;
    double score;
  };

  struct TuningConfigMeasurements {
    vector<TuningConfigMeasurement> values;
    bool referenceFailed = false;
  };

  template<typename Tuner>
  TuningConfigMeasurements measureConfigs(
    const TuningContext& context,
    vector<VulkanTuneParams> configs,
    const TuningMeasurementPlan& plan,
    double* bestScoreAcrossCalls = nullptr,
    bool logSuccessfulResults = true,
    vk_shader::ComputePipelines* sharedPipelines = nullptr
  ) {
    TuningConfigMeasurements measurements;
    dedupCandidates(configs);
    const size_t firstShuffledIndex = KeepsCurrentConfigFirst<Tuner>::value ? 2 : 1;
    if(configs.size() > 2 && !context.preserveGemmMeasurementOrder) {
      Rand rand("VulkanTuner:" + Tuner::name());
      for(size_t i = configs.size() - 1; i > firstShuffledIndex; i--) {
        const size_t j = firstShuffledIndex + static_cast<size_t>(rand.nextUInt64(i - firstShuffledIndex + 1));
        swap(configs[i], configs[j]);
      }
    }
    const bool checkReference = Tuner::name() == "transformerScaleDotProduct";
    const size_t candidateCount = count_if(
      configs.begin(), configs.end(), [&context](const VulkanTuneParams& config) {
        return isValidTuningConfig<Tuner>(context, config);
      }
    ) +
      (!checkReference && !configs.empty() && !isValidTuningConfig<Tuner>(context, configs.front()) ? 1 : 0);
    if(context.timer == nullptr || (!context.attentionSkipTimestamps && !context.timer->isUsable())) {
      if(context.logger != nullptr)
        context.logger->write("Skipping Vulkan tuner " + Tuner::name() + ": compute timestamps are unavailable");
      return measurements;
    }
    VulkanTimestampTimer& timer = *context.timer;

    unique_ptr<vk_shader::ComputePipelines> localPipelines;
    if(sharedPipelines == nullptr) {
      localPipelines = std::make_unique<vk_shader::ComputePipelines>(
        context.device->device,
        context.device->info,
        nullptr,
        false
      );
      sharedPipelines = localPipelines.get();
    }
    vk_shader::ComputePipelines& pipelines = *sharedPipelines;
    vector<Pipeline*> previousTargets;
    const auto pipelineGuard = makeScopeGuard([&]() {
      for(Pipeline* pipeline: previousTargets)
        pipelines.destroyPipeline(*pipeline);
    });
    double localBestScore = 0.0;
    double& bestScore = bestScoreAcrossCalls != nullptr ? *bestScoreAcrossCalls : localBestScore;
    vector<float> referenceReadback = Tuner::name() == "transformerXgemm" &&
      context.transformerXgemmCpuReference != nullptr ? *context.transformerXgemmCpuReference : vector<float>();
    size_t candidateIndex = 0;
    for(size_t configIndex = 0; configIndex < configs.size(); configIndex++) {
      requireTuningDevice(context, Tuner::name());
      const VulkanTuneParams& candidate = configs[configIndex];
      const bool isReferenceCandidate = configIndex == 0;
      if((checkReference || !isReferenceCandidate) && !isValidTuningConfig<Tuner>(context, candidate))
        continue;
      const size_t currentCandidateIndex = ++candidateIndex;
      try {
        // The previous candidate has finished before the next pipeline is built.
        // Reuse the pipeline cache and avoid a device-idle wait for every candidate.
        for(Pipeline* pipeline: previousTargets)
          pipelines.destroyPipeline(*pipeline);
        previousTargets.clear();
        vector<const Pipeline*> targets;
        VkResult result = Tuner::create(context, candidate, pipelines, targets);
        context.device->recordDeviceLoss(result, "tuning pipeline creation");
        if(result == VK_ERROR_DEVICE_LOST) {
          for(const Pipeline* pipeline: targets)
            pipelines.destroyPipeline(*const_cast<Pipeline*>(pipeline));
          requireTuningDevice(context, Tuner::name() + " " + describeTuningParams(Tuner::name(), candidate));
        }
        if(result != VK_SUCCESS) {
          if(isReferenceCandidate) {
            logTuningFailure(
              context,
              currentCandidateIndex,
              candidateCount,
              "pipeline creation failed: " + vk_helper::vkErrorToString(result)
            );
          }
          logTuningProgressIfNeeded(context, currentCandidateIndex, candidateCount);
          for(const Pipeline* pipeline: targets)
            pipelines.destroyPipeline(*const_cast<Pipeline*>(pipeline));
          if(StopsOnReferenceImplFail<Tuner>::value(candidate) && isReferenceCandidate)
          {
            measurements.referenceFailed = true;
            return measurements;
          }
          continue;
        }
        for(const Pipeline* pipeline: targets)
          previousTargets.push_back(const_cast<Pipeline*>(pipeline));
        double callsPerSecond = 0.0;
        vector<float> readback;
        vector<float> cpuReference;
        double errorProp = numeric_limits<double>::quiet_NaN();
        string error;
        const bool measured = timer.measure(
          targets, candidate, context, plan, callsPerSecond, readback, errorProp, error,
          referenceReadback.empty() && usesCpuReference(Tuner::name()) ? &cpuReference : nullptr
        );
        if(context.device->deviceLossStage.load() != nullptr)
          requireTuningDevice(context, Tuner::name() + " " + describeTuningParams(Tuner::name(), candidate), error);
        if(!measured) {
          if(isReferenceCandidate) {
            logTuningFailure(
              context,
              currentCandidateIndex,
              candidateCount,
              error.empty() ? "measurement failed" : error
            );
          }
          logTuningProgressIfNeeded(context, currentCandidateIndex, candidateCount);
          if(StopsOnReferenceImplFail<Tuner>::value(candidate) && isReferenceCandidate)
          {
            measurements.referenceFailed = true;
            return measurements;
          }
          continue;
        }
        if(!isfinite(callsPerSecond) || callsPerSecond <= 0.0) {
          if(isReferenceCandidate) {
            logTuningFailure(
              context,
              currentCandidateIndex,
              candidateCount,
              "measurement returned invalid calls/sec"
            );
          }
          logTuningProgressIfNeeded(context, currentCandidateIndex, candidateCount);
          if(StopsOnReferenceImplFail<Tuner>::value(candidate) && isReferenceCandidate)
          {
            measurements.referenceFailed = true;
            return measurements;
          }
          continue;
        }
        if(referenceReadback.empty() && !cpuReference.empty()) {
          referenceReadback = std::move(cpuReference);
          if(Tuner::name() == "transformerXgemm" && context.transformerXgemmCpuReference != nullptr)
            *context.transformerXgemmCpuReference = referenceReadback;
        }
        if(referenceReadback.empty()) {
          referenceReadback = readback;
          validateReadback(referenceReadback, readback, plan, errorProp);
        }
        else
          validateReadback(referenceReadback, readback, plan, errorProp);
        const bool usesStrictErrorTolerance =
          Tuner::name() == "transformerXgemm" ||
          Tuner::name() == "hgemmCooperativeMatrix" || Tuner::name() == "hgemmCooperativeMatrixNHWC" ||
          Tuner::name() == "hgemmCooperativeMatrixNCHW" ||
          Tuner::name() == "transformerScaleDotProductCooperativeMatrixNCHW" ||
          Tuner::name() == "transformerScaleDotProductCooperativeMatrixNHWC";
        const double score = usesStrictErrorTolerance
          ? VulkanTuner::computeCooperativeMatrixTuningScore(callsPerSecond, errorProp, plan.errorTolerance)
          : VulkanTuner::computeTuningScore(
            callsPerSecond,
            errorProp,
            plan.errorTolerance
          );
        measurements.values.push_back({candidate, callsPerSecond, score});
        const bool isBest = score > bestScore;
        const bool loggedResult = logSuccessfulResults && isBest &&
          logTuningResult(context, currentCandidateIndex, candidateCount, candidate, Tuner::name(), callsPerSecond);
        if(logSuccessfulResults && !isBest && context.printOnlyOnImprovement && isReferenceCandidate)
          logTuningProgress(context, currentCandidateIndex, candidateCount, true);
        if(isBest)
          bestScore = score;
        logTuningProgressIfNeeded(context, currentCandidateIndex, candidateCount, loggedResult);
      }
      catch(const StringError& e) {
        if(context.device->deviceLossStage.load() != nullptr)
          requireTuningDevice(context, Tuner::name() + " " + describeTuningParams(Tuner::name(), candidate), e.what());
        // A failed pipeline specialization is an invalid candidate, not a fatal tuning failure.
        if(isReferenceCandidate)
          logTuningFailure(
            context,
            currentCandidateIndex,
            candidateCount,
            e.what()
          );
        logTuningProgressIfNeeded(context, currentCandidateIndex, candidateCount);
        if(StopsOnReferenceImplFail<Tuner>::value(candidate) && isReferenceCandidate)
        {
          measurements.referenceFailed = true;
          return measurements;
        }
      }
    }
    return measurements;
  }

  template<typename Tuner>
  struct UsesCooperativeMatrixShapeScreening {
    static constexpr bool value = false;
  };

  template<typename Tuner>
  double testAllConfigs(const TuningContext& context, VulkanTuneParams& currentConfig) {
    VulkanTuneParams defaults;
    if constexpr(UsesCooperativeMatrixShapeScreening<Tuner>::value) {
      vector<VulkanTuneParams> screeningConfigs = Tuner::screeningCandidates(currentConfig, context);
      if(screeningConfigs.empty())
        return 0.0;
      const TuningConfigMeasurements screeningMeasurements = measureConfigs<Tuner>(
        context, std::move(screeningConfigs), makeCooperativeMatrixScreeningMeasurementPlan(Tuner::name(), context),
        nullptr, !context.printOnlyOnImprovement
      );
      if(screeningMeasurements.values.empty())
        return 0.0;

      vector<TuningConfigMeasurement> bestByShape;
      for(const TuningConfigMeasurement& measurement: screeningMeasurements.values) {
        if(measurement.score <= 0.0)
          continue;
        const CooperativeMatrixTuneShape candidateShape = Tuner::shape(measurement.config);
        auto previous = find_if(
          bestByShape.begin(), bestByShape.end(), [&](const TuningConfigMeasurement& existing) {
            const CooperativeMatrixTuneShape existingShape = Tuner::shape(existing.config);
            return existingShape.accType == candidateShape.accType &&
                   existingShape.MSize == candidateShape.MSize &&
                   existingShape.NSize == candidateShape.NSize &&
                   existingShape.KSize == candidateShape.KSize &&
                   existingShape.subgroupSize == candidateShape.subgroupSize;
          }
        );
        if(previous == bestByShape.end())
          bestByShape.push_back(measurement);
        else if(measurement.score > previous->score)
          *previous = measurement;
      }

      vector<TuningConfigMeasurement> selectedShapes;
      for(int accType: {16, 32}) {
        vector<TuningConfigMeasurement> accumulatorShapes;
        copy_if(
          bestByShape.begin(), bestByShape.end(), back_inserter(accumulatorShapes),
          [accType](const TuningConfigMeasurement& measurement) {
            return Tuner::accumulatorType(measurement.config) == accType;
          }
        );
        sort(
          accumulatorShapes.begin(), accumulatorShapes.end(),
          [](const TuningConfigMeasurement& a, const TuningConfigMeasurement& b) { return a.score > b.score; }
        );
        if(accumulatorShapes.empty())
          continue;
        const double threshold = accumulatorShapes.front().score *
          VulkanTuner::COOPERATIVE_MATRIX_SHAPE_SCORE_RATIO;
        for(size_t i = 0; i < accumulatorShapes.size(); i++) {
          if(context.full || i < VulkanTuner::COOPERATIVE_MATRIX_MIN_SHAPES_PER_ACCUMULATOR ||
             accumulatorShapes[i].score >= threshold)
            selectedShapes.push_back(accumulatorShapes[i]);
        }
      }
      if(selectedShapes.empty())
        return 0.0;

      TuningContext detailContext = context;
      detailContext.cooperativeMatrixTuneShapes.clear();
      for(const TuningConfigMeasurement& selected: selectedShapes) {
        const CooperativeMatrixTuneShape shape = Tuner::shape(selected.config);
        const bool alreadyAdded = any_of(
          detailContext.cooperativeMatrixTuneShapes.begin(), detailContext.cooperativeMatrixTuneShapes.end(),
          [&shape](const CooperativeMatrixTuneShape& previous) {
            return previous.accType == shape.accType && previous.MSize == shape.MSize &&
                   previous.NSize == shape.NSize && previous.KSize == shape.KSize &&
                   previous.subgroupSize == shape.subgroupSize;
          }
        );
        if(!alreadyAdded)
          detailContext.cooperativeMatrixTuneShapes.push_back(shape);
      }
      if(context.logger != nullptr) {
        context.logger->write(
          "Vulkan tuner " + Tuner::name() + " property screening: mode=" +
          (context.full ? "full" : "quick") + ", property_limit=" +
          (context.full ? "all" : to_string(VulkanTuner::COOPERATIVE_MATRIX_NON_FULL_PROPERTY_LIMIT)) +
          ", supported_properties=" + to_string(context.cooperativeMatrixTuneShapes.size()) +
          ", passing_properties=" + to_string(bestByShape.size()) +
          ", detail_properties=" + to_string(detailContext.cooperativeMatrixTuneShapes.size())
        );
      }

      const TuningMeasurementPlan detailPlan = makeMeasurementPlan(Tuner::name(), detailContext);
      vector<TuningConfigMeasurement> tunedShapes;
      double detailBestScoreForLogging = 0.0;
      for(const TuningConfigMeasurement& selected: selectedShapes) {
        const size_t beamWidth = context.full ? 8 : 3;
        vector<TuningConfigMeasurement> beam = {selected};
        const auto runStage = [&](const auto& makeCandidates) {
          vector<VulkanTuneParams> configs;
          for(const TuningConfigMeasurement& seed: beam) {
            configs.push_back(seed.config);
            vector<VulkanTuneParams> expanded = makeCandidates(seed.config);
            configs.insert(configs.end(), expanded.begin(), expanded.end());
          }
          const TuningConfigMeasurements measurements = measureConfigs<Tuner>(
            detailContext, std::move(configs), detailPlan, &detailBestScoreForLogging
          );
          if(measurements.values.empty())
            return false;
          vector<TuningConfigMeasurement> ranked = measurements.values;
          sort(
            ranked.begin(), ranked.end(),
            [](const TuningConfigMeasurement& a, const TuningConfigMeasurement& b) { return a.score > b.score; }
          );
          ranked.erase(
            remove_if(ranked.begin(), ranked.end(), [](const TuningConfigMeasurement& measurement) {
              return measurement.score <= 0.0;
            }),
            ranked.end()
          );
          if(ranked.empty())
            return false;
          if(ranked.size() > beamWidth)
            ranked.resize(beamWidth);
          beam = std::move(ranked);
          return true;
        };
        if(!runStage([&](const VulkanTuneParams& seed) {
             return Tuner::subgroupCandidates(seed, context.full, detailContext);
           }))
          continue;
        if(!runStage([&](const VulkanTuneParams& seed) {
             return Tuner::reuseCandidates(seed, context.full, detailContext);
           }))
          continue;
        if(!runStage([&](const VulkanTuneParams& seed) {
             return Tuner::sharedMemoryCandidates(seed, context.full, detailContext);
           }))
          continue;
        if(!runStage([&](const VulkanTuneParams& seed) {
             return Tuner::vectorCandidates(seed, context.full, detailContext);
           }))
          continue;
        tunedShapes.push_back(beam.front());
      }
      if(tunedShapes.empty())
        return 0.0;
      const TuningConfigMeasurement& best = *max_element(
        tunedShapes.begin(), tunedShapes.end(),
        [](const TuningConfigMeasurement& a, const TuningConfigMeasurement& b) { return a.score < b.score; }
      );
      currentConfig = best.config;
      if(context.logger != nullptr) {
        context.logger->write(
          "Vulkan tuner " + Tuner::name() + " selected Calls/sec " +
          Global::doubleToString(best.callsPerSecond) + " " + describeTuningParams(Tuner::name(), best.config)
        );
      }
      return best.callsPerSecond;
    }
    vector<VulkanTuneParams> configs = Tuner::candidates(currentConfig, context.full, context);
    configs.insert(configs.begin(), Tuner::reference(currentConfig, defaults));

    const TuningConfigMeasurements measurements = measureConfigs<Tuner>(
      context, std::move(configs), makeMeasurementPlan(Tuner::name(), context)
    );
    if(measurements.referenceFailed || measurements.values.empty())
      return 0.0;

    if(Tuner::name() == "transformerScaleDotProductCooperativeMatrixNHWC" && context.recheckAttentionFinalists) {
      vector<TuningConfigMeasurement> ranked = measurements.values;
      sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
      vector<VulkanTuneParams> finalists = {currentConfig};
      for(size_t i = 0; i < std::min(size_t(3), ranked.size()); i++)
        if(ranked[i].score > 0.0)
          finalists.push_back(ranked[i].config);
      dedupCandidates(finalists);
      TuningMeasurementPlan finalPlan = makeMeasurementPlan(Tuner::name(), context);
      finalPlan.warmupRuns = 20;
      finalPlan.totalRuns = 116;
      finalPlan.workloadWeights = {1.0};
      TuningContext finalContext = context;
      finalContext.preserveGemmMeasurementOrder = true;
      map<string, vector<double>> elapsed;
      for(int round = 0; round < 6; round++) {
        vector<VulkanTuneParams> order = finalists;
        if(round % 2 != 0)
          reverse(order.begin(), order.end());
        const TuningConfigMeasurements repeated = measureConfigs<Tuner>(
          finalContext, order, finalPlan, nullptr, false
        );
        for(const auto& measurement: repeated.values)
          if(measurement.score > 0.0)
            elapsed[describeTuningParams(
              Tuner::name(),
              measurement.config
            )].push_back(1.0 / measurement.callsPerSecond);
      }
      const string incumbent = describeTuningParams(Tuner::name(), currentConfig);
      if(elapsed[incumbent].size() != 6)
        throw StringError("NHWC attention incumbent failed repeated validation; tuning not saved");
      const auto median = [](vector<double> values) {
        sort(values.begin(), values.end());
        return (values[2] + values[3]) * 0.5;
      };
      double bestSeconds = median(elapsed[incumbent]);
      for(const auto& candidate: finalists) {
        const string description = describeTuningParams(Tuner::name(), candidate);
        vector<double>& seconds = elapsed[description];
        if(seconds.size() != 6)
          continue;
        const double medianSeconds = median(seconds);
        if(VulkanTuner::isTransformerGemmUpgrade(elapsed[incumbent], seconds) && medianSeconds < bestSeconds) {
          bestSeconds = medianSeconds;
          currentConfig = candidate;
          logTuningResult(
            context,
            static_cast<size_t>(&candidate - finalists.data()) + 1,
            finalists.size(),
            candidate,
            Tuner::name(),
            1.0 / medianSeconds
          );
        }
      }
      if(!std::isfinite(bestSeconds))
        throw StringError("NHWC attention tuning failed: no finalist passed repeated measurements");
      if(context.logger != nullptr)
        context.logger->write("NHWC attention selected after 6 balanced rounds of warmup20/timed96 (5/6 wins, 2% median guard): " +
                              describeTuningParams(Tuner::name(), currentConfig));
      return 1.0 / bestSeconds;
    }
    const TuningConfigMeasurement& best = *max_element(
      measurements.values.begin(), measurements.values.end(),
      [](const TuningConfigMeasurement& a, const TuningConfigMeasurement& b) { return a.score < b.score; }
    );
    const TuningConfigMeasurement* selected = &best;
    if(selected->score <= 0.0)
      return 0.0;
    currentConfig = selected->config;
    if(context.logger != nullptr)
      context.logger->write(
        "Vulkan tuner " + Tuner::name() + " selected a measured candidate"
      );
    return selected->callsPerSecond;
  }

  template<typename Tuner>
  double runTuner(const TuningContext& context, VulkanTuneParams& currentConfig) {
    if(context.onlyMissing && tuningComplete(Tuner::name(), context))
      return 0.0;
    logTuningStart(context, Tuner::name());
    if(Tuner::name().find("gemm") != string::npos || Tuner::name() == "transformerXgemm") {
      const auto cases = getTunerGemmTuneCases(Tuner::name(), context);
      if(cases.empty()) {
        if(context.logger != nullptr)
          context.logger->write("Skipping Vulkan tuner " + Tuner::name() + ": no production workloads");
        return 0.0;
      }
      if(!context.printOnlyOnImprovement && context.logger != nullptr)
        for(const auto& c: cases)
          context.logger->write("  workload: K=" + to_string(c.inChannels) +
            ", N=" + to_string(c.outChannels) + ", " +
            (c.kind == GemmTuneCase::Kind::Dense ? "Dense M=" + to_string(context.batchSize) : "spatial/Winograd") +
            ", dispatches/forward=" + Global::doubleToString(c.weight));
    }
    const double rate = testAllConfigs<Tuner>(context, currentConfig);
    if(isfinite(rate) && rate > 0.0)
      recordTuning(Tuner::name(), context);
    return rate;
  }

  vector<int> powersOfTwoUpTo(int maximum) {
    vector<int> values;
    for(int value = 1; value <= maximum; value *= 2)
      values.push_back(value);
    return values;
  }

  struct XgemmDirectTuner {
    static string name() { return "xgemmDirect"; }

    static bool isValid(const VulkanTuneParams& config) { return config.xgemmDirect.isValid(); }
    static XgemmDirectTuneParams openCLReferenceParams() {
      XgemmDirectTuneParams result;
      result.WGD = 8;
      result.MDIMCD = 1;
      result.NDIMCD = 1;
      result.MDIMAD = 1;
      result.NDIMBD = 1;
      result.KWID = 1;
      result.PADA = 1;
      result.PADB = 1;
      result.VWMD = 1;
      result.VWND = 1;
      return result;
    }

    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams&) {
      VulkanTuneParams result = current;
      result.xgemmDirect = openCLReferenceParams();
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool full,
      const TuningContext&
    ) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(
        configs,
        full ? vector<int>{8,16,32,64} : vector<int>{8,16,32},
        [](VulkanTuneParams& p, int v) { p.xgemmDirect.WGD = v; }
      );
      addCandidates(
        configs,
        vector<int>{8,16,32},
        [](VulkanTuneParams& p, int v) { p.xgemmDirect.MDIMCD = v; }
      );
      addCandidates(
        configs,
        vector<int>{8,16,32},
        [](VulkanTuneParams& p, int v) { p.xgemmDirect.NDIMCD = v; }
      );
      addCandidates(
        configs,
        vector<int>{8,16,32},
        [](VulkanTuneParams& p, int v) { p.xgemmDirect.MDIMAD = v; }
      );
      addCandidates(
        configs,
        vector<int>{8,16,32},
        [](VulkanTuneParams& p, int v) { p.xgemmDirect.NDIMBD = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{2,8,16} : vector<int>{2,8},
        [](VulkanTuneParams& p, int v) { p.xgemmDirect.KWID = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{1,2,4} : vector<int>{2,4},
        [](VulkanTuneParams& p, int v) { p.xgemmDirect.VWMD = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{1,2,4} : vector<int>{2,4},
        [](VulkanTuneParams& p, int v) { p.xgemmDirect.VWND = v; }
      );
      addCandidates(configs, vector<int>{1}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.PADA = v; });
      addCandidates(configs, vector<int>{1}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.PADB = v; });
      configs.erase(
        remove_if(
          configs.begin(),
          configs.end(),
          [](const VulkanTuneParams& config) { return !isValid(config); }
        ),
        configs.end()
      );

      VulkanTuneParams slightlyTunedConfig = current;
      slightlyTunedConfig.xgemmDirect = openCLReferenceParams();
      slightlyTunedConfig.xgemmDirect.MDIMCD = 8;
      slightlyTunedConfig.xgemmDirect.NDIMCD = 8;
      slightlyTunedConfig.xgemmDirect.MDIMAD = 8;
      slightlyTunedConfig.xgemmDirect.NDIMBD = 8;
      VulkanTuneParams slightlyTunedConfig2 = slightlyTunedConfig;
      slightlyTunedConfig2.xgemmDirect.WGD = 16;
      configs.insert(configs.begin(), current);
      configs.insert(configs.begin(), slightlyTunedConfig2);
      configs.insert(configs.begin(), slightlyTunedConfig);
      return configs;
    }

    static VkResult create(
      const TuningContext& context,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      const auto cases = getTunerGemmTuneCases(name(), context);
      VkResult result = VK_SUCCESS;
      if(any_of(cases.begin(), cases.end(), [](const GemmTuneCase& c) { return c.kind == GemmTuneCase::Kind::Dense; })) {
        result = pipelines.createXgemmDirectBatchedTT(
          pipelines.xgemmDirectBatchedTT,
          config.xgemmDirect,
          config.vulkan
        );
        if(result != VK_SUCCESS)
          return result;
        targets.push_back(&pipelines.xgemmDirectBatchedTT);
      }
      if(any_of(cases.begin(), cases.end(), [](const GemmTuneCase& c) { return c.kind != GemmTuneCase::Kind::Dense; })) {
        result = pipelines.createXgemmStridedBatched(
          pipelines.xgemmStridedBatchedFp32,
          config.xgemmDirect,
          config.vulkan
        );
        if(result != VK_SUCCESS)
          return result;
        targets.push_back(&pipelines.xgemmStridedBatchedFp32);
      }
      return result;
    }
  };

  struct TransformerXgemmTuner {
    static string name() { return "transformerXgemm"; }
    static bool isValid(const VulkanTuneParams& config) { return config.transformerXgemm.isValid(); }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current, bool full, const TuningContext& context
    ) {
      VulkanTuneParams seed = current;
      seed.xgemmDirect = current.transformerXgemm;
      vector<VulkanTuneParams> configs = XgemmDirectTuner::candidates(seed, full, context);
      for(auto& candidate: configs) {
        candidate.transformerXgemm = candidate.xgemmDirect;
        candidate.xgemmDirect = current.xgemmDirect;
      }
      configs.insert(configs.begin(), current);
      return configs;
    }

    static VkResult create(
      const TuningContext& context, const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets
    ) {
      const auto& p = config.transformerXgemm;
      const auto& limits = context.device->info.properties.limits;
      const uint64_t sharedBytes = static_cast<uint64_t>(p.WGD) * (2ULL * p.WGD + p.PADA + p.PADB) *
        (config.vulkan.shouldUseFP16Storage && config.vulkan.shouldUseFP16Compute ? 2 : 4);
      if(!p.isValid() || p.MDIMCD > limits.maxComputeWorkGroupSize[0] ||
         p.NDIMCD > limits.maxComputeWorkGroupSize[1] ||
         static_cast<uint64_t>(p.MDIMCD) * p.NDIMCD > limits.maxComputeWorkGroupInvocations ||
         sharedBytes > limits.maxComputeSharedMemorySize)
        return VK_ERROR_INITIALIZATION_FAILED;
      VkResult result = pipelines.createTransformerXgemm(pipelines.transformerXgemm, p, config.vulkan);
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.transformerXgemm);
      return result;
    }
  };

  void validateTransformerXgemm(
    const TuningContext& context, const VulkanTuneParams& config, vector<float>& incumbentReference
  ) {
    const auto actualCases = getTunerGemmTuneCases("transformerXgemm", context);
    for(bool edge: {false, true}) {
      for(int precision = 0; precision < (edge ? 3 : 1); precision++) {
        if(edge && precision != 0 && (!config.vulkan.canUseFP16Storage || !config.vulkan.canUseFP16Compute))
          continue;
        for(int batch: {1, 4}) {
          TuningContext validation = context;
          validation.batchSize = batch;
          if(edge) {
            validation.nnXLen = 3;
            validation.nnYLen = 5;
          }
          VulkanTuneParams testConfig = config;
          testConfig.xgemmDirect = config.transformerXgemm;
          if(edge) {
            testConfig.vulkan.shouldUseFP16Storage = precision != 0;
            testConfig.vulkan.shouldUseFP16Compute = precision == 2;
            testConfig.vulkan.shouldUseCooperativeMatrix = false;
            testConfig.vulkan.shouldUseHgemmCooperativeMatrixNCHW = false;
            testConfig.vulkan.shouldUseTransformerDualGemmSwiGLU = false;
          }
          auto plan = makeMeasurementPlan("transformerXgemm", validation);
          plan.gemmCases = edge ? vector<GemmTuneCase>{{17, 19, 1.0}} : actualCases;
          plan.workloadWeights.assign(plan.gemmCases.size(), 1.0);
          plan.warmupRuns = 0;
          plan.totalRuns = plan.gemmCases.size();
          vk_shader::ComputePipelines pipelines(context.device->device, context.device->info, nullptr, false);
          vector<const Pipeline*> targets;
          if(TransformerXgemmTuner::create(validation, testConfig, pipelines, targets) != VK_SUCCESS ||
             pipelines.createXgemmStridedBatched(pipelines.xgemmStridedBatchedFp32,
               testConfig.xgemmDirect, testConfig.vulkan) != VK_SUCCESS)
            throw StringError("Transformer XGEMM validation pipeline creation failed");
          vector<float> output, legacy, cpu;
          double rate = 0.0, errorProp = 0.0;
          string error;
          if(!context.timer->measure(targets, testConfig, validation, plan, rate, output, errorProp, error, &cpu) ||
             !context.timer->measure({&pipelines.xgemmStridedBatchedFp32}, testConfig, validation,
               plan, rate, legacy, errorProp, error))
            throw StringError("Transformer XGEMM validation measurement failed: " + error);
          const double cpuError = VulkanTuner::computeErrorProp(cpu, output);
          const double legacyError = VulkanTuner::computeErrorProp(legacy, output);
          double maxAbs = 0.0;
          if(output.size() == legacy.size())
            for(size_t i = 0; i < output.size(); i++)
              maxAbs = std::max(maxAbs, fabs(static_cast<double>(output[i]) - legacy[i]));
          if(!isfinite(cpuError) || cpuError > plan.errorTolerance ||
             !isfinite(legacyError) || legacyError > 1e-6 || output.size() != legacy.size())
            throw StringError("Transformer XGEMM parity failed: CPU ErrorProp=" + Global::doubleToString(cpuError) +
              ", legacy ErrorProp=" + Global::doubleToString(legacyError) +
              ", legacy CPU ErrorProp=" + Global::doubleToString(VulkanTuner::computeErrorProp(cpu, legacy)) +
              ", maxAbs=" + Global::doubleToString(maxAbs) +
              ", " + string(edge ? "M15/N19/K17" : "actual projections") + " B=" + to_string(batch) +
              " " + describeTuningParams("transformerXgemm", testConfig));
          if(!edge && batch == context.batchSize)
            incumbentReference = std::move(cpu);
          if(context.logger != nullptr)
            context.logger->write("  validation: " + string(edge ? "M15/N19/K17" : "actual projections") +
              " B=" + to_string(batch) + " fp16Storage=" + to_string(testConfig.vulkan.shouldUseFP16Storage) +
              " fp16Compute=" + to_string(testConfig.vulkan.shouldUseFP16Compute) +
              " CPU ErrorProp=" + Global::doubleToString(cpuError) +
              " legacy ErrorProp=" + Global::doubleToString(legacyError) +
              " maxAbs=" + Global::doubleToString(maxAbs));
        }
      }
    }
  }

  void tuneTransformerXgemm(const TuningContext& context, VulkanTuneParams& config) {
    if(usesGenericNHWC(config) || context.modelInfo.transformerProjections.empty() ||
       (context.onlyMissing && tuningComplete("transformerXgemm", context)))
      return;
    logTuningStart(context, "transformerXgemm");
    vector<float> cpuReference;
    validateTransformerXgemm(context, config, cpuReference);
    TuningContext projection = context;
    projection.transformerXgemmCpuReference = &cpuReference;
    auto plan = makeMeasurementPlan("transformerXgemm", projection);
    plan.warmupRuns = 8 * plan.gemmCases.size();
    plan.totalRuns = 40 * plan.gemmCases.size();
    vk_shader::ComputePipelines pipelines(context.device->device, context.device->info, nullptr, false);
    const VulkanTuneParams incumbent = config;
    const auto baseline = measureConfigs<TransformerXgemmTuner>(
      projection, {incumbent}, plan, nullptr, true, &pipelines);
    if(baseline.values.size() != 1 || baseline.values.front().score <= 0.0)
      throw StringError("Transformer XGEMM incumbent validation failed; tuning not saved");
    auto measured = measureConfigs<TransformerXgemmTuner>(projection,
      TransformerXgemmTuner::candidates(incumbent, context.full, projection), plan, nullptr, true, &pipelines);
    sort(measured.values.begin(), measured.values.end(), [](const auto& a, const auto& b) {
      return a.score > b.score;
    });
    vector<VulkanTuneParams> finalists{incumbent};
    for(const auto& value: measured.values) {
      if(value.score > 0.0 && value.config != incumbent)
        finalists.push_back(value.config);
      if(finalists.size() == 4)
        break;
    }
    plan.warmupRuns = 20 * plan.gemmCases.size();
    plan.totalRuns = 116 * plan.gemmCases.size();
    projection.preserveGemmMeasurementOrder = true;
    map<string, vector<double>> elapsed;
    for(int round = 0; round < 6; round++) {
      auto order = finalists;
      if(round % 2 != 0)
        reverse(order.begin(), order.end());
      const auto repeated = measureConfigs<TransformerXgemmTuner>(
        projection, order, plan, nullptr, false, &pipelines);
      for(const auto& value: repeated.values)
        if(value.score > 0.0)
          elapsed[describeTuningParams("transformerXgemm", value.config)].push_back(1.0 / value.callsPerSecond);
    }
    const auto median = [](vector<double> seconds) {
      sort(seconds.begin(), seconds.end());
      return (seconds[2] + seconds[3]) * 0.5;
    };
    const auto& reference = elapsed[describeTuningParams("transformerXgemm", incumbent)];
    if(reference.size() != 6)
      throw StringError("Transformer XGEMM incumbent failed finalist validation; tuning not saved");
    double bestSeconds = median(reference);
    VulkanTuneParams best = incumbent;
    for(const auto& candidate: finalists) {
      const auto& seconds = elapsed[describeTuningParams("transformerXgemm", candidate)];
      if(seconds.size() != 6 || !VulkanTuner::isTransformerGemmUpgrade(reference, seconds))
        continue;
      const double candidateSeconds = median(seconds);
      if(candidateSeconds < bestSeconds) {
        bestSeconds = candidateSeconds;
        best = candidate;
        logTuningResult(context, static_cast<size_t>(&candidate - finalists.data()) + 1,
          finalists.size(), best, "transformerXgemm", 1.0 / candidateSeconds);
      }
    }
    config.transformerXgemm = best.transformerXgemm;
    recordTuning("transformerXgemm", context);
    if(context.logger != nullptr)
      context.logger->write("Transformer XGEMM selected after6 balanced rounds, warmup20/timed96, "
        "minimum ratio1.02 and5/6 wins: " + describeTuningParams("transformerXgemm", best));
  }

  struct HgemmCooperativeMatrixTunerImpl {
    // The generic cooperative-matrix HGEMM replaces the xgemmBatched step in
    // both the 3x3 and 5x5 Winograd convolution paths.
    static string name() { return "hgemmCooperativeMatrix"; }

    static bool isValid(const VulkanTuneParams& config) {
      return config.vulkan.canUseCooperativeMatrix &&
             config.vulkan.canUseFP16Storage &&
             config.vulkan.canUseFP16Compute &&
             config.hgemmCooperativeMatrix.isValid();
    }

    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      // The reference uses one property-defined fragment in each dimension.
      // MWARP/NWARP/KDIM and subgroupSize remain hardware-selected values.
      result.hgemmCooperativeMatrix.MWG = result.hgemmCooperativeMatrix.MWARP;
      result.hgemmCooperativeMatrix.NWG = result.hgemmCooperativeMatrix.NWARP;
      result.hgemmCooperativeMatrix.KWG = result.hgemmCooperativeMatrix.KDIM;
      result.hgemmCooperativeMatrix.MWAVE = result.hgemmCooperativeMatrix.MWARP;
      result.hgemmCooperativeMatrix.NWAVE = result.hgemmCooperativeMatrix.NWARP;
      result.hgemmCooperativeMatrix.SA = 0;
      result.hgemmCooperativeMatrix.SB = 0;
      result.hgemmCooperativeMatrix.VWM = defaults.hgemmCooperativeMatrix.VWM;
      result.hgemmCooperativeMatrix.VWN = defaults.hgemmCooperativeMatrix.VWN;
      return result;
    }

    static vector<VulkanTuneParams> screeningCandidates(
      const VulkanTuneParams& current,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      size_t validPropertyCount = 0;
      for(const CooperativeMatrixTuneShape& shape: context.cooperativeMatrixTuneShapes) {
        const size_t previousConfigCount = configs.size();
        VulkanTuneParams config = current;
        HGemmCooperativeMatrixTuneParams& params = config.hgemmCooperativeMatrix;
        params.accType = shape.accType;
        params.MWARP = shape.MSize;
        params.NWARP = shape.NSize;
        params.KDIM = shape.KSize;
        params.subgroupSize = shape.subgroupSize;
        params.MWG = shape.MSize;
        params.NWG = shape.NSize;
        params.KWG = shape.KSize;
        params.MWAVE = shape.MSize;
        params.NWAVE = shape.NSize;
        params.SA = 0;
        params.SB = 0;
        // Direct B loads do not use a vector width.
        params.VWN = 1;
        for(int vwm: {1, 2, 4}) {
          params.VWM = vwm;
          if(isValidCooperativeMatrixTuneParams(context, params))
            configs.push_back(config);
        }
        if(configs.size() > previousConfigCount) {
          validPropertyCount += 1;
          if(!context.full && validPropertyCount >= VulkanTuner::COOPERATIVE_MATRIX_NON_FULL_PROPERTY_LIMIT)
            break;
        }
      }
      return configs;
    }

    static int accumulatorType(const VulkanTuneParams& config) {
      return config.hgemmCooperativeMatrix.accType;
    }

    static CooperativeMatrixTuneShape shape(const VulkanTuneParams& config) {
      const HGemmCooperativeMatrixTuneParams& params = config.hgemmCooperativeMatrix;
      return {params.accType, params.MWARP, params.NWARP, params.KDIM, params.subgroupSize};
    }

    static vector<VulkanTuneParams> subgroupCandidates(
      const VulkanTuneParams& seed,
      bool full,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      const vector<int> ratios = full ? vector<int>{1, 2, 4} : vector<int>{1, 2};
      for(int subgroupM: ratios) {
        for(int subgroupN: ratios) {
          VulkanTuneParams config = seed;
          HGemmCooperativeMatrixTuneParams& params = config.hgemmCooperativeMatrix;
          if(!checkedMultiply(params.MWARP, subgroupM, params.MWAVE) ||
             !checkedMultiply(params.NWARP, subgroupN, params.NWAVE))
            continue;
          params.MWG = params.MWAVE;
          params.NWG = params.NWAVE;
          if(isValidCooperativeMatrixTuneParams(context, params))
            configs.push_back(config);
        }
      }
      return configs;
    }

    static vector<VulkanTuneParams> reuseCandidates(
      const VulkanTuneParams& seed,
      bool full,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      const vector<int> ratios = full ? vector<int>{1, 2, 4} : vector<int>{1, 2};
      for(int reuseM: ratios) {
        for(int reuseN: ratios) {
          for(int reuseK: ratios) {
            VulkanTuneParams config = seed;
            HGemmCooperativeMatrixTuneParams& params = config.hgemmCooperativeMatrix;
            if(!checkedMultiply(params.MWAVE, reuseM, params.MWG) ||
               !checkedMultiply(params.NWAVE, reuseN, params.NWG) ||
               !checkedMultiply(params.KDIM, reuseK, params.KWG))
              continue;
            if(isValidCooperativeMatrixTuneParams(context, params))
              configs.push_back(config);
          }
        }
      }
      return configs;
    }

    static vector<VulkanTuneParams> sharedMemoryCandidates(
      const VulkanTuneParams& seed,
      bool,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      for(int sa: {0, 1}) {
        for(int sb: {0, 1}) {
          VulkanTuneParams config = seed;
          config.hgemmCooperativeMatrix.SA = sa;
          config.hgemmCooperativeMatrix.SB = sb;
          if(sb == 0)
            config.hgemmCooperativeMatrix.VWN = 1;
          if(isValidCooperativeMatrixTuneParams(context, config.hgemmCooperativeMatrix))
            configs.push_back(config);
        }
      }
      return configs;
    }

    static vector<VulkanTuneParams> vectorCandidates(
      const VulkanTuneParams& seed,
      bool,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      const vector<int> widthsN = seed.hgemmCooperativeMatrix.SB == 0
        ? vector<int>{1} : vector<int>{1, 2, 4};
      for(int vwm: {1, 2, 4}) {
        for(int vwn: widthsN) {
          VulkanTuneParams config = seed;
          config.hgemmCooperativeMatrix.VWM = vwm;
          config.hgemmCooperativeMatrix.VWN = vwn;
          if(isValidCooperativeMatrixTuneParams(context, config.hgemmCooperativeMatrix))
            configs.push_back(config);
        }
      }
      return configs;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool,
      const TuningContext&
    ) {
      return {current};
    }

    static VkResult create(
      const TuningContext&,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      VkResult result = pipelines.createHgemmCooperativeMatrix(
        pipelines.hgemmCooperativeMatrix,
        config.hgemmCooperativeMatrix
      );
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.hgemmCooperativeMatrix);
      return result;
    }
  };

  struct HgemmCooperativeMatrixNHWCTunerImpl {
    static string name() { return "hgemmCooperativeMatrixNHWC"; }

    static bool isValid(const VulkanTuneParams& config) {
      return config.vulkan.canUseCooperativeMatrix &&
             config.vulkan.canUseFP16Storage && config.vulkan.canUseFP16Compute &&
             config.hgemmCooperativeMatrixNHWC.DB == 0 &&
             config.hgemmCooperativeMatrixNHWC.SA == 1 &&
             config.hgemmCooperativeMatrixNHWC.SB == 1 &&
             config.hgemmCooperativeMatrixNHWC.isValid();
    }

    static VulkanTuneParams reference(
      const VulkanTuneParams& current,
      const VulkanTuneParams& defaults,
      bool requireSharedStaging = true
    ) {
      VulkanTuneParams result = current;
      auto& p = result.hgemmCooperativeMatrixNHWC;
      p.MWG = p.MWARP;
      p.NWG = p.NWARP;
      p.KWG = p.KDIM;
      p.MWAVE = p.MWARP;
      p.NWAVE = p.NWARP;
      p.SA = requireSharedStaging ? 1 : 0;
      p.SB = requireSharedStaging ? 1 : 0;
      p.DB = 0;
      p.VWK = defaults.hgemmCooperativeMatrixNHWC.VWK;
      p.VWN = defaults.hgemmCooperativeMatrixNHWC.VWN;
      return result;
    }

    static vector<VulkanTuneParams> screeningCandidates(
      const VulkanTuneParams& current,
      const TuningContext& context,
      bool requireSharedStaging = true
    ) {
      vector<VulkanTuneParams> configs;
      size_t validPropertyCount = 0;
      for(const CooperativeMatrixTuneShape& shape: context.cooperativeMatrixTuneShapes) {
        const size_t previousConfigCount = configs.size();
        VulkanTuneParams config = current;
        auto& p = config.hgemmCooperativeMatrixNHWC;
        p.accType = shape.accType;
        p.MWARP = shape.MSize;
        p.NWARP = shape.NSize;
        p.KDIM = shape.KSize;
        p.subgroupSize = shape.subgroupSize;
        p.MWG = p.MWAVE = p.MWARP;
        p.NWG = p.NWAVE = p.NWARP;
        p.KWG = p.KDIM;
        p.SA = requireSharedStaging ? 1 : 0;
        p.SB = requireSharedStaging ? 1 : 0;
        p.DB = 0;
        for(int vwk: {1, 2, 4})
          for(int vwn: {1, 2, 4}) {
            p.VWK = vwk;
            p.VWN = vwn;
            if(isValidCooperativeMatrixTuneParams(context, p))
              configs.push_back(config);
          }
        if(configs.size() > previousConfigCount &&
           ++validPropertyCount >= VulkanTuner::COOPERATIVE_MATRIX_NON_FULL_PROPERTY_LIMIT && !context.full)
          break;
      }
      return configs;
    }

    static int accumulatorType(const VulkanTuneParams& config) { return config.hgemmCooperativeMatrixNHWC.accType; }
    static CooperativeMatrixTuneShape shape(const VulkanTuneParams& config) {
      const auto& p = config.hgemmCooperativeMatrixNHWC;
      return {p.accType, p.MWARP, p.NWARP, p.KDIM, p.subgroupSize};
    }

    static vector<VulkanTuneParams> subgroupCandidates(
      const VulkanTuneParams& seed,
      bool full,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      const vector<int> ratios = full ? vector<int>{1, 2, 4} : vector<int>{1, 2};
      for(int m: ratios)
        for(int n: ratios) {
          VulkanTuneParams c = seed;
          auto& p = c.hgemmCooperativeMatrixNHWC;
          if(!checkedMultiply(p.MWARP, m, p.MWAVE) || !checkedMultiply(p.NWARP, n, p.NWAVE))
            continue;
          p.MWG = p.MWAVE;
          p.NWG = p.NWAVE;
          if(isValidCooperativeMatrixTuneParams(context, p))
            configs.push_back(c);
        }
      return configs;
    }

    static vector<VulkanTuneParams> reuseCandidates(
      const VulkanTuneParams& seed,
      bool full,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      const vector<int> ratios = full ? vector<int>{1, 2, 4} : vector<int>{1, 2};
      for(int m: ratios)
        for(int n: ratios)
          for(int k: ratios) {
            VulkanTuneParams c = seed;
            auto& p = c.hgemmCooperativeMatrixNHWC;
            if(!checkedMultiply(p.MWAVE, m, p.MWG) || !checkedMultiply(p.NWAVE, n, p.NWG) || !checkedMultiply(p.KDIM, k, p.KWG))
              continue;
            if(isValidCooperativeMatrixTuneParams(context, p))
              configs.push_back(c);
          }
      return configs;
    }

    static vector<VulkanTuneParams> sharedMemoryCandidates(
      const VulkanTuneParams& seed,
      bool,
      const TuningContext& context,
      bool requireSharedStaging = true
    ) {
      vector<VulkanTuneParams> configs;
      for(int sa: {0, 1})
        for(int sb: {0, 1}) {
          if(requireSharedStaging && (sa != 1 || sb != 1))
            continue;
          VulkanTuneParams c = seed;
          c.hgemmCooperativeMatrixNHWC.SA = sa;
          c.hgemmCooperativeMatrixNHWC.SB = sb;
          if(isValidCooperativeMatrixTuneParams(context, c.hgemmCooperativeMatrixNHWC))
            configs.push_back(c);
        }
      return configs;
    }

    static vector<VulkanTuneParams> vectorCandidates(
      const VulkanTuneParams& seed,
      bool,
      const TuningContext& context,
      bool requireSharedStaging = true
    ) {
      vector<VulkanTuneParams> configs;
      for(int vwk: {1, 2, 4})
        for(int vwn: {1, 2, 4}) {
          VulkanTuneParams c = seed;
          c.hgemmCooperativeMatrixNHWC.VWK = vwk;
          c.hgemmCooperativeMatrixNHWC.VWN = vwn;
          // Vectorized cooperative loads require shared staging on the
          // corresponding operand. Do not let the earlier SA=SB=0 beam seed
          // suppress all VWK/VWN values above one.
          if(requireSharedStaging || vwk != 1)
            c.hgemmCooperativeMatrixNHWC.SA = 1;
          if(requireSharedStaging || vwn != 1)
            c.hgemmCooperativeMatrixNHWC.SB = 1;
          if(isValidCooperativeMatrixTuneParams(context, c.hgemmCooperativeMatrixNHWC))
            configs.push_back(c);
        }
      return configs;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool,
      const TuningContext&
    ) { return {current}; }

    static VkResult create(
      const TuningContext&,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      VkResult result = pipelines.createHgemmCooperativeMatrixNHWC(
        pipelines.hgemmCooperativeMatrixNHWC,
        config.hgemmCooperativeMatrixNHWC
      );
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.hgemmCooperativeMatrixNHWC);
      return result;
    }
  };

  template<>
  struct KeepsCurrentConfigFirst<HgemmCooperativeMatrixNHWCTunerImpl> { static constexpr bool value = true;
  };

  template<>
  struct UsesCooperativeMatrixShapeScreening<HgemmCooperativeMatrixNHWCTunerImpl> { static constexpr bool value = true;
  };

  template<>
  struct StopsOnReferenceImplFail<HgemmCooperativeMatrixNHWCTunerImpl> {
    static bool value(const VulkanTuneParams&) { return false; }
  };

  template<>
  struct KeepsCurrentConfigFirst<HgemmCooperativeMatrixTunerImpl> {
    static constexpr bool value = true;
  };

  template<>
  struct UsesCooperativeMatrixShapeScreening<HgemmCooperativeMatrixTunerImpl> {
    static constexpr bool value = true;
  };

  template<>
  struct StopsOnReferenceImplFail<HgemmCooperativeMatrixTunerImpl> {
    static bool value(const VulkanTuneParams&) { return false; }
  };

  void tuneTransformerPreNorm(const TuningContext& context, VulkanTuneParams& config) {
    if(!usesGenericNHWC(config) || !config.vulkan.shouldUseFP16Storage ||
       !config.vulkan.shouldUseFP16Compute || context.modelInfo.transformerProjections.empty())
      return;
    const int normCallsPerForward = context.modelInfo.transformerPlainNormCount + context.modelInfo.transformerFusedNormCount;
    if(!context.modelInfo.transformerUniformNormChannels || normCallsPerForward == 0)
      return;
    if(context.logger != nullptr)
      context.logger->write("Transformer pre-norm model counts: plain=" +
      to_string(context.modelInfo.transformerPlainNormCount) + " fused=" + to_string(context.modelInfo.transformerFusedNormCount));
    const int channels = VulkanTuner::getTransformerFFNInputChannelsForTuning(context.modelInfo);
    const auto& limits = context.device->info.properties.limits;
    const int channelAlignment = std::lcm(
      4,
      std::lcm(config.hgemmCooperativeMatrixNHWC.NWG, config.hgemmCooperativeMatrixNHWC.KWG)
    );
    if(channels % channelAlignment != 0)
      return; // Existing fused residual contract requires unpadded channels.
    int spatialAlignment = config.hgemmCooperativeMatrixNHWC.MWG;
    if(config.vulkan.shouldUseHgemmCooperativeMatrixNCHW)
      spatialAlignment = std::lcm(
        spatialAlignment,
        config.hgemmCooperativeMatrixNCHW.getRequiredSpatialAlignment()
      );
    if(config.vulkan.shouldUseTransformerDualGemmSwiGLU)
      spatialAlignment = std::lcm(
        spatialAlignment,
        config.transformerDualGemmSwiGLU.getRequiredSpatialAlignment()
      );
    const int spatial = vk_helper::roundUpToMultipleInt(context.nnXLen * context.nnYLen, spatialAlignment);
    using Params = TransformerRMSNormTuneParms;
    const auto describe = [](const Params& p) {
      return "MODE=" + to_string(p.MODE) + " WG_C=" + to_string(p.WG_C_SIZE) +
        " WG_XY=" + to_string(p.WG_XY_SIZE) + " CPT=" + to_string(p.C_PER_THREAD);
    };
    if(context.onlyMissing && tuningComplete("transformerPreNorm", context))
      return;
    logTuningStart(context, "transformerPreNorm");
    vector<Params> candidates{config.transformerPreNorm};
    for(int mode: {0,1,2})
      for(int c: {8,16,32,64,128,256})
        for(int xy: {1,2,4,8,16,32})
          for(int items: {1,2,4,8,16,32}) {
            Params p{c,xy,items,mode};
            const uint64_t invocations = static_cast<uint64_t>(c) * xy;
            if(!p.isValid() || invocations > limits.maxComputeWorkGroupInvocations ||
               invocations > limits.maxComputeWorkGroupSize[0] || invocations * sizeof(float) > limits.maxComputeSharedMemorySize ||
               c * items >= 2 * channels || (mode != 0 && c * items < channels) ||
               (mode == 2 && !canUseTransformerRMSNormSubgroup(context.device->info, p)))
              continue;
            if(none_of(candidates.begin(), candidates.end(), [&](const auto& other) { return describe(p) == describe(other); }))
              candidates.push_back(p);
          }

    ComputePipelines pipelines(
      context.device->device,
      context.device->info,
      nullptr,
      false
    );
    const auto measure = [&](
      const Params& p,
      int runs,
      bool reverse,
      double& seconds,
      bool masks
    ) {
      pipelines.destroyPipeline(pipelines.transformerResidualRmsNormNHWC);
      pipelines.destroyPipeline(pipelines.transformerRmsNormNHWC);
      pipelines.destroyPipeline(pipelines.addPointWise);
      VkResult result = pipelines.createTransformerRMSNorm(pipelines.transformerResidualRmsNormNHWC, p, config.vulkan, true, true);
      if(result == VK_SUCCESS)
        result = pipelines.createTransformerRMSNorm(pipelines.transformerRmsNormNHWC, p, config.vulkan, true);
      if(result == VK_SUCCESS)
        result = pipelines.createAddPointWise(pipelines.addPointWise, config.pointwise, config.vulkan);
      context.device->recordDeviceLoss(result, "pre-norm pipeline creation");
      requireTuningDevice(context, "transformerPreNorm", describe(p));
      if(result != VK_SUCCESS)
        return false;
      VulkanTuneParams measured = config;
      measured.rmsNorm = p; // Only this timer's dispatch geometry; production generic profile is unchanged.
      string error;
      bool valid = context.timer->measureTransformerResidualRMSNorm(
        measured,
        context,
        pipelines.transformerResidualRmsNormNHWC,
        pipelines.transformerRmsNormNHWC,
        pipelines.addPointWise,
        context.batchSize,
        spatial,
        0,
        error,
        &seconds,
        runs,
        reverse
      );
      requireTuningDevice(context, "transformerPreNorm", error);
      if(valid && masks)
        for(int batch: {1, context.batchSize})
          for(int s: {spatial, std::max(1, spatial - 1)})
            for(int mask = 0; mask < 4; mask++) {
              if(!context.timer->measureTransformerResidualRMSNorm(
                   measured,
                   context,
                   pipelines.transformerResidualRmsNormNHWC,
                   pipelines.transformerRmsNormNHWC,
                   pipelines.addPointWise,
                   batch,
                   s,
                   mask,
                   error
                 ))
                valid = false;
              requireTuningDevice(context, "transformerPreNorm", error);
            }
      if(!valid && context.logger != nullptr)
        context.logger->write("Transformer pre-norm rejected " + describe(p) + " " + error);
      return valid && isfinite(seconds) && seconds > 0.0;
    };

    vector<pair<double,Params>> passing;
    double bestScreenSeconds = numeric_limits<double>::infinity();
    for(size_t i = 0; i < candidates.size(); i++) {
      bool loggedResult = false;
      double seconds = 0.0;
      if(measure(candidates[i], 12, false, seconds, false)) {
        passing.emplace_back(seconds, candidates[i]);
        if(seconds < bestScreenSeconds) {
          bestScreenSeconds = seconds;
          VulkanTuneParams loggingConfig = config;
          loggingConfig.transformerPreNorm = candidates[i];
          loggedResult = logTuningResult(
            context, i + 1, candidates.size(), loggingConfig, "transformerPreNorm", normCallsPerForward / seconds
          );
        }
      }
      logTuningProgressIfNeeded(context, i + 1, candidates.size(), loggedResult);
    }

    sort(passing.begin(), passing.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    vector<Params> finalists{config.transformerPreNorm};
    // Recheck each execution path: short screening must not eliminate a new
    // reduction mode merely because clocks or compilation warmed up in order.
    for(int mode: {0,1,2})
      for(const auto& value: passing)
        if(value.second.MODE == mode && describe(value.second) != describe(finalists[0])) {
          finalists.push_back(value.second);
          break;
        }

    map<string,vector<double>> elapsed;
    for(int round = 0; round < 6; round++) {
      auto order = finalists;
      if(round % 2 != 0)
        reverse(order.begin(), order.end());
      for(const auto& p: order) {
        double seconds = 0.0;
        if(measure(p, 96, round % 2 != 0, seconds, round == 0))
          elapsed[describe(p)].push_back(seconds);
      }
    }

    const auto median = [](vector<double> v) {
      sort(v.begin(), v.end());
      return (v[2] + v[3]) * 0.5;
    };
    const auto& reference = elapsed[describe(finalists[0])];
    if(reference.size() != 6)
      throw StringError("Transformer pre-norm incumbent validation failed; tuning not saved");
    Params selected = finalists[0];
    double best = median(reference);
    for(const auto& p: finalists) {
      const auto& values = elapsed[describe(p)];
      if(VulkanTuner::isTransformerGemmUpgrade(reference, values) && median(values) < best) {
        selected = p;
        best = median(values);
        VulkanTuneParams loggingConfig = config;
        loggingConfig.transformerPreNorm = p;
        logTuningResult(
          context,
          static_cast<size_t>(&p - finalists.data()) + 1,
          finalists.size(),
          loggingConfig,
          "transformerPreNorm",
          normCallsPerForward / best
        );
      }
    }

    config.transformerPreNorm = selected;
    recordTuning("transformerPreNorm", context);
    if(context.logger != nullptr)
      context.logger->write("Transformer pre-norm selected " + describe(selected) +
      " us(norms/forward)=" + Global::doubleToString(best * 1e6) + " baseline_us=" + Global::doubleToString(median(reference) * 1e6));
  }

  void tuneTransformerHgemmNHWC(const TuningContext& context, VulkanTuneParams& config) {
    if(!config.transformerHgemmNHWC.isValid())
      config.transformerHgemmNHWC = config.hgemmCooperativeMatrixNHWC;
    if(!usesGenericNHWC(config) || context.modelInfo.transformerProjections.empty())
      return;
    const auto& layout = config.hgemmCooperativeMatrixNHWC;
    int spatialAlignment = layout.MWG;
    if(config.vulkan.shouldUseHgemmCooperativeMatrixNCHW)
      spatialAlignment = std::lcm(
        spatialAlignment,
        config.hgemmCooperativeMatrixNCHW.getRequiredSpatialAlignment()
      );
    if(config.vulkan.shouldUseTransformerDualGemmSwiGLU)
      spatialAlignment = std::lcm(
        spatialAlignment,
        config.transformerDualGemmSwiGLU.getRequiredSpatialAlignment()
      );
    const int physicalM = vk_helper::roundUpToMultipleInt(context.nnXLen * context.nnYLen, spatialAlignment);
    const int channelAlignment = std::lcm(layout.NWG, layout.KWG);
    // This first profile preserves the current global strides and packed weights.
    // Unaligned models retain the existing profile rather than changing padding.
    for(const auto& projection: context.modelInfo.transformerProjections) {
      if(config.vulkan.shouldUseTransformerDualGemmSwiGLU && projection.fusedFFNInput)
        continue;
      if(projection.inChannels % channelAlignment != 0 || projection.outChannels % channelAlignment != 0 ||
         (projection.packedOutput && config.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
          projection.outChannels % config.hgemmCooperativeMatrixNCHW.NWG != 0))
        return;
    }
    if(context.onlyMissing && tuningComplete("transformerHgemmNHWC", context))
      return;
    TuningContext projectionContext = context;
    projectionContext.transformerPhysicalLayout = &layout;
    projectionContext.transformerPhysicalSpatialSize = physicalM;
    projectionContext.transformerFusedFFN = config.vulkan.shouldUseTransformerDualGemmSwiGLU;
    logTuningStart(projectionContext, "transformerHgemmNHWC");
    const auto cases = getTunerGemmTuneCases("hgemmCooperativeMatrixNHWC", projectionContext);
    VulkanTuneParams reference = config;
    reference.hgemmCooperativeMatrixNHWC = config.transformerHgemmNHWC;
    vector<VulkanTuneParams> candidates{reference};
    for(const auto& shape: context.cooperativeMatrixTuneShapes) {
      // Preserve the existing accumulation precision in this selection experiment.
      if(shape.accType != layout.accType)
        continue;
      for(int m: {32, 64, 128})
        for(int n: {16, 32, 64})
          for(int k: {16, 32, 64})
            for(int sm: {1, 2})
              for(int sn: {1, 2}) {
                VulkanTuneParams candidate = config;
                auto& p = candidate.hgemmCooperativeMatrixNHWC;
                p.MWARP = shape.MSize;
                p.NWARP = shape.NSize;
                p.KDIM = shape.KSize;
                p.subgroupSize = shape.subgroupSize;
                p.MWG = m;
                p.NWG = n;
                p.KWG = k;
                p.MWAVE = p.MWARP * sm;
                p.NWAVE = p.NWARP * sn;
                p.SA = p.SB = 1;
                p.DB = 0;
                if(!isValidCooperativeMatrixTuneParams(context, p) || physicalM % m != 0)
                  continue;
                if(any_of(cases.begin(), cases.end(), [&](const GemmTuneCase& c) {
                  return c.inChannels % k != 0 || c.outChannels % n != 0;
                }))
                  continue;
                candidates.push_back(candidate);
              }
    }
    if(context.logger != nullptr) {
      context.logger->write("Transformer NHWC GEMM: fixed physical M=" + to_string(physicalM) +
        ", B=" + to_string(context.batchSize) + ", independent profile; convolution unchanged");
      for(const auto& c: cases)
        context.logger->write("Transformer projection K=" + to_string(c.inChannels) + ", N=" +
          to_string(c.outChannels) + ", dispatches/forward=" + to_string(static_cast<int>(c.weight)));
    }
    auto plan = makeMeasurementPlan("hgemmCooperativeMatrixNHWC", projectionContext);
    plan.warmupRuns = 8 * cases.size();
    plan.totalRuns = 40 * cases.size();
    vk_shader::ComputePipelines pipelines(
      context.device->device,
      context.device->info,
      nullptr,
      false
    );
    auto measurements = measureConfigs<HgemmCooperativeMatrixNHWCTunerImpl>(
      projectionContext, std::move(candidates), plan, nullptr, true, &pipelines
    );
    sort(measurements.values.begin(), measurements.values.end(),
      [](const auto& a, const auto& b) { return a.score > b.score;
      });
    vector<VulkanTuneParams> finalists{reference};
    for(const auto& value: measurements.values) {
      if(value.score > 0.0 && value.config != reference)
        finalists.push_back(value.config);
      if(finalists.size() == 4)
        break;
    }
    plan.warmupRuns = 20 * cases.size();
    plan.totalRuns = 116 * cases.size();
    projectionContext.preserveGemmMeasurementOrder = true;
    map<string, vector<double>> elapsed;
    for(int round = 0; round < 6; round++) {
      auto order = finalists;
      if(round % 2 != 0)
        reverse(order.begin(), order.end());
      const auto repeated = measureConfigs<HgemmCooperativeMatrixNHWCTunerImpl>(
        projectionContext,
        order,
        plan,
        nullptr,
        false,
        &pipelines
      );
      for(const auto& value: repeated.values) if(value.score > 0.0) {
        const string description = describeTuningParams("hgemmCooperativeMatrixNHWC", value.config);
        elapsed[description].push_back(1.0 / value.callsPerSecond);
      }
    }
    const auto medianSeconds = [](vector<double> seconds) {
      sort(seconds.begin(), seconds.end());
      return (seconds[2] + seconds[3]) * 0.5;
    };
    const string referenceDescription = describeTuningParams("hgemmCooperativeMatrixNHWC", reference);
    const auto& referenceSeconds = elapsed[referenceDescription];
    if(referenceSeconds.size() != 6)
      throw StringError("Transformer GEMM incumbent failed finalist validation; tuning was not saved");
    double bestSeconds = medianSeconds(referenceSeconds);
    VulkanTuneParams best = reference;
    for(const auto& candidate: finalists) {
      const string description = describeTuningParams("hgemmCooperativeMatrixNHWC", candidate);
      const auto& seconds = elapsed[description];
      const bool upgrade = VulkanTuner::isTransformerGemmUpgrade(referenceSeconds, seconds);
      if(seconds.size() != 6)
        continue;
      const double median = medianSeconds(seconds);
      if(upgrade && median < bestSeconds) {
        bestSeconds = median;
        best = candidate;
        logTuningResult(
          projectionContext,
          static_cast<size_t>(&candidate - finalists.data()) + 1,
          finalists.size(),
          candidate,
          "hgemmCooperativeMatrixNHWC",
          1.0 / median
        );
      }
    }
    config.transformerHgemmNHWC = best.hgemmCooperativeMatrixNHWC;
    recordTuning("transformerHgemmNHWC", context);
    if(context.logger != nullptr)
      context.logger->write("Transformer NHWC GEMM selected after6 rechecks, warmup20/timed96, minimum ratio1.02 and5/6 wins: " +
        describeTuningParams("hgemmCooperativeMatrixNHWC", best));
  }

  struct HgemmCooperativeMatrixNCHWTunerImpl {
    static string name() { return "hgemmCooperativeMatrixNCHW"; }

    static bool isValid(const VulkanTuneParams& config) {
      return config.vulkan.canUseCooperativeMatrix && config.hgemmCooperativeMatrixNCHW.isValid();
    }

    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.hgemmCooperativeMatrixNCHW.MWG = result.hgemmCooperativeMatrixNCHW.MWARP;
      result.hgemmCooperativeMatrixNCHW.NWG = result.hgemmCooperativeMatrixNCHW.NWARP;
      result.hgemmCooperativeMatrixNCHW.KWG = result.hgemmCooperativeMatrixNCHW.KDIM;
      result.hgemmCooperativeMatrixNCHW.MWAVE = result.hgemmCooperativeMatrixNCHW.MWARP;
      result.hgemmCooperativeMatrixNCHW.NWAVE = result.hgemmCooperativeMatrixNCHW.NWARP;
      result.hgemmCooperativeMatrixNCHW.SB = defaults.hgemmCooperativeMatrixNCHW.SB;
      result.hgemmCooperativeMatrixNCHW.VWM = defaults.hgemmCooperativeMatrixNCHW.VWM;
      result.hgemmCooperativeMatrixNCHW.VWN = defaults.hgemmCooperativeMatrixNCHW.VWN;
      return result;
    }

    static vector<VulkanTuneParams> screeningCandidates(
      const VulkanTuneParams& current,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      size_t validPropertyCount = 0;
      for(const CooperativeMatrixTuneShape& shape: context.cooperativeMatrixTuneShapes) {
        const size_t previousConfigCount = configs.size();
        VulkanTuneParams config = current;
        HGemmCooperativeMatrixNCHWTuneParams& params = config.hgemmCooperativeMatrixNCHW;
        params.accType = shape.accType;
        params.MWARP = shape.MSize;
        params.NWARP = shape.NSize;
        params.KDIM = shape.KSize;
        params.subgroupSize = shape.subgroupSize;
        params.MWG = shape.MSize;
        params.NWG = shape.NSize;
        params.KWG = shape.KSize;
        params.MWAVE = shape.MSize;
        params.NWAVE = shape.NSize;
        params.SB = 0;
        // Direct B loads do not use a vector width.
        params.VWN = 1;
        for(int vwm: {1, 2, 4}) {
          params.VWM = vwm;
          if(isValidCooperativeMatrixTuneParams(context, params))
            configs.push_back(config);
        }
        if(configs.size() > previousConfigCount) {
          validPropertyCount += 1;
          if(!context.full && validPropertyCount >= VulkanTuner::COOPERATIVE_MATRIX_NON_FULL_PROPERTY_LIMIT)
            break;
        }
      }
      return configs;
    }

    static int accumulatorType(const VulkanTuneParams& config) {
      return config.hgemmCooperativeMatrixNCHW.accType;
    }

    static CooperativeMatrixTuneShape shape(const VulkanTuneParams& config) {
      const HGemmCooperativeMatrixNCHWTuneParams& params = config.hgemmCooperativeMatrixNCHW;
      return {params.accType, params.MWARP, params.NWARP, params.KDIM, params.subgroupSize};
    }

    static vector<VulkanTuneParams> subgroupCandidates(
      const VulkanTuneParams& seed,
      bool full,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      const vector<int> ratios = full ? vector<int>{1, 2, 4} : vector<int>{1, 2};
      for(int subgroupM: ratios) {
        for(int subgroupN: ratios) {
          VulkanTuneParams config = seed;
          HGemmCooperativeMatrixNCHWTuneParams& params = config.hgemmCooperativeMatrixNCHW;
          if(!checkedMultiply(params.MWARP, subgroupM, params.MWAVE) ||
             !checkedMultiply(params.NWARP, subgroupN, params.NWAVE))
            continue;
          params.MWG = params.MWAVE;
          params.NWG = params.NWAVE;
          if(isValidCooperativeMatrixTuneParams(context, params))
            configs.push_back(config);
        }
      }
      return configs;
    }

    static vector<VulkanTuneParams> reuseCandidates(
      const VulkanTuneParams& seed,
      bool full,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      const vector<int> ratios = full ? vector<int>{1, 2, 4} : vector<int>{1, 2};
      for(int reuseM: ratios) {
        for(int reuseN: ratios) {
          for(int reuseK: ratios) {
            VulkanTuneParams config = seed;
            HGemmCooperativeMatrixNCHWTuneParams& params = config.hgemmCooperativeMatrixNCHW;
            if(!checkedMultiply(params.MWAVE, reuseM, params.MWG) ||
               !checkedMultiply(params.NWAVE, reuseN, params.NWG) ||
               !checkedMultiply(params.KDIM, reuseK, params.KWG))
              continue;
            if(isValidCooperativeMatrixTuneParams(context, params))
              configs.push_back(config);
          }
        }
      }
      return configs;
    }

    static vector<VulkanTuneParams> sharedMemoryCandidates(
      const VulkanTuneParams& seed,
      bool,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      for(int sb: {0, 1}) {
        VulkanTuneParams config = seed;
        config.hgemmCooperativeMatrixNCHW.SB = sb;
        if(sb == 0)
          config.hgemmCooperativeMatrixNCHW.VWN = 1;
        if(isValidCooperativeMatrixTuneParams(context, config.hgemmCooperativeMatrixNCHW))
          configs.push_back(config);
      }
      return configs;
    }

    static vector<VulkanTuneParams> vectorCandidates(
      const VulkanTuneParams& seed,
      bool,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      const vector<int> widthsN = seed.hgemmCooperativeMatrixNCHW.SB == 0
        ? vector<int>{1} : vector<int>{1, 2, 4};
      for(int vwm: {1, 2, 4}) {
        for(int vwn: widthsN) {
          VulkanTuneParams config = seed;
          config.hgemmCooperativeMatrixNCHW.VWM = vwm;
          config.hgemmCooperativeMatrixNCHW.VWN = vwn;
          if(isValidCooperativeMatrixTuneParams(context, config.hgemmCooperativeMatrixNCHW))
            configs.push_back(config);
        }
      }
      return configs;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool,
      const TuningContext&
    ) {
      return {current};
    }

    static VkResult create(
      const TuningContext&,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      VkResult result = pipelines.createHgemmCooperativeMatrixNCHW(
        pipelines.hgemmCooperativeMatrixNCHW,
        config.hgemmCooperativeMatrixNCHW
      );
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.hgemmCooperativeMatrixNCHW);
      return result;
    }
  };

  template<>
  struct KeepsCurrentConfigFirst<HgemmCooperativeMatrixNCHWTunerImpl> {
    static constexpr bool value = true;
  };

  template<>
  struct UsesCooperativeMatrixShapeScreening<HgemmCooperativeMatrixNCHWTunerImpl> {
    static constexpr bool value = true;
  };

  template<>
  struct StopsOnReferenceImplFail<HgemmCooperativeMatrixNCHWTunerImpl> {
    static bool value(const VulkanTuneParams&) { return false; }
  };

  template<>
  bool isValidTuningConfig<HgemmCooperativeMatrixTunerImpl>(
    const TuningContext& context,
    const VulkanTuneParams& config
  ) {
    return config.vulkan.canUseCooperativeMatrix &&
           config.vulkan.canUseFP16Storage &&
           config.vulkan.canUseFP16Compute &&
           isValidCooperativeMatrixTuneParams(context, config.hgemmCooperativeMatrix);
  }

  template<>
  bool isValidTuningConfig<HgemmCooperativeMatrixNHWCTunerImpl>(
    const TuningContext& context,
    const VulkanTuneParams& config
  ) {
    return config.vulkan.canUseCooperativeMatrix &&
           config.vulkan.canUseFP16Storage && config.vulkan.canUseFP16Compute &&
           config.hgemmCooperativeMatrixNHWC.DB == 0 &&
           isValidCooperativeMatrixTuneParams(context, config.hgemmCooperativeMatrixNHWC);
  }

  template<>
  bool isValidTuningConfig<HgemmCooperativeMatrixNCHWTunerImpl>(
    const TuningContext& context,
    const VulkanTuneParams& config
  ) {
    return config.vulkan.canUseCooperativeMatrix &&
           isValidCooperativeMatrixTuneParams(context, config.hgemmCooperativeMatrixNCHW);
  }

  struct XgemmTuner {
    static string name() { return "xgemm"; }

    static bool isValid(const VulkanTuneParams& config) { return config.xgemm.isValid(); }
    static XgemmTuneParams openCLReferenceParams() {
      XgemmTuneParams result;
      result.MDIMC = 1;
      result.NDIMC = 1;
      result.MWG = 8;
      result.NWG = 8;
      result.KWG = 8;
      result.KWI = 1;
      result.MDIMA = 1;
      result.NDIMB = 1;
      result.VWM = 1;
      result.VWN = 1;
      return result;
    }

    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams&) {
      VulkanTuneParams result = current;
      result.xgemm = openCLReferenceParams();
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool full,
      const TuningContext&
    ) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(
        configs,
        full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64},
        [](VulkanTuneParams& p, int v) { p.xgemm.MWG = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64},
        [](VulkanTuneParams& p, int v) { p.xgemm.NWG = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{8,16,32} : vector<int>{16,32},
        [](VulkanTuneParams& p, int v) { p.xgemm.KWG = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{1,2,8} : vector<int>{1,2},
        [](VulkanTuneParams& p, int v) { p.xgemm.KWI = v; }
      );
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.MDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.NDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.MDIMA = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.NDIMB = v; });
      // Vulkan provides width-specific targets for 1, 2, and 4; OpenCL's 8 has no target.
      addCandidates(
        configs,
        full ? vector<int>{1,2,4} : vector<int>{2,4},
        [](VulkanTuneParams& p, int v) { p.xgemm.VWM = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{1,2,4} : vector<int>{2,4},
        [](VulkanTuneParams& p, int v) { p.xgemm.VWN = v; }
      );
      configs.erase(
        remove_if(
          configs.begin(),
          configs.end(),
          [](const VulkanTuneParams& p) { return !p.xgemm.isValid(); }
        ),
        configs.end()
      );
      if(!full) {
        configs.erase(
          remove_if(
            configs.begin(),
            configs.end(),
            [](const VulkanTuneParams& p) { return !p.xgemm.isSimple(); }
          ),
          configs.end()
        );
      }
      VulkanTuneParams slightlyTunedConfig = current;
      slightlyTunedConfig.xgemm = openCLReferenceParams();
      slightlyTunedConfig.xgemm.MDIMC = 8;
      slightlyTunedConfig.xgemm.NDIMC = 8;
      slightlyTunedConfig.xgemm.MDIMA = 8;
      slightlyTunedConfig.xgemm.NDIMB = 8;
      VulkanTuneParams slightlyTunedConfig2 = slightlyTunedConfig;
      slightlyTunedConfig2.xgemm.MWG = 16;
      slightlyTunedConfig2.xgemm.NWG = 16;
      slightlyTunedConfig2.xgemm.KWG = 16;
      configs.insert(configs.begin(), slightlyTunedConfig2);
      configs.insert(configs.begin(), slightlyTunedConfig);
      configs.insert(configs.begin(), current);
      return configs;
    }

    static VkResult create(
      const TuningContext&,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      VkResult result = pipelines.createXgemmBatched(
        pipelines.xgemmBatchedFp32,
        config.xgemm,
        config.xgemm16,
        config.vulkan
      );
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.xgemmBatchedFp32);
      return result;
    }
  };

  template<>
  struct StopsOnReferenceImplFail<XgemmTuner> {
    static bool value(const VulkanTuneParams& config) {
      return config.vulkan.shouldUseFP16Storage && !config.vulkan.shouldUseFP16Compute;
    }
  };

  struct Xgemm16Tuner {
    static string name() { return "xgemm16"; }

    static bool isValid(const VulkanTuneParams& config) { return config.xgemm16.isValid(); }
    static XgemmTuneParams openCLReferenceParams() {
      XgemmTuneParams result;
      result.MDIMC = 1;
      result.NDIMC = 1;
      result.MWG = 8;
      result.NWG = 8;
      result.KWG = 8;
      result.KWI = 1;
      result.MDIMA = 1;
      result.NDIMB = 1;
      result.VWM = 1;
      result.VWN = 1;
      return result;
    }

    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams&) {
      VulkanTuneParams result = current;
      result.xgemm16 = openCLReferenceParams();
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool full,
      const TuningContext&
    ) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(
        configs,
        full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64},
        [](VulkanTuneParams& p, int v) { p.xgemm16.MWG = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64},
        [](VulkanTuneParams& p, int v) { p.xgemm16.NWG = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{8,16,32} : vector<int>{16,32},
        [](VulkanTuneParams& p, int v) { p.xgemm16.KWG = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{1,2,8} : vector<int>{1,2},
        [](VulkanTuneParams& p, int v) { p.xgemm16.KWI = v; }
      );
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.MDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.NDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.MDIMA = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.NDIMB = v; });
      addCandidates(
        configs,
        full ? vector<int>{1,2,4} : vector<int>{2,4},
        [](VulkanTuneParams& p, int v) { p.xgemm16.VWM = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{1,2,4} : vector<int>{2,4},
        [](VulkanTuneParams& p, int v) { p.xgemm16.VWN = v; }
      );
      configs.erase(
        remove_if(
          configs.begin(),
          configs.end(),
          [](const VulkanTuneParams& p) { return !p.xgemm16.isValid(); }
        ),
        configs.end()
      );
      if(!full) {
        configs.erase(
          remove_if(
            configs.begin(),
            configs.end(),
            [](const VulkanTuneParams& p) { return !p.xgemm16.isSimple(); }
          ),
          configs.end()
        );
      }

      VulkanTuneParams slightlyTunedConfig = current;
      slightlyTunedConfig.xgemm16 = openCLReferenceParams();
      slightlyTunedConfig.xgemm16.MDIMC = 8;
      slightlyTunedConfig.xgemm16.NDIMC = 8;
      slightlyTunedConfig.xgemm16.MDIMA = 8;
      slightlyTunedConfig.xgemm16.NDIMB = 8;
      VulkanTuneParams slightlyTunedConfig2 = slightlyTunedConfig;
      slightlyTunedConfig2.xgemm16.MWG = 16;
      slightlyTunedConfig2.xgemm16.NWG = 16;
      slightlyTunedConfig2.xgemm16.KWG = 16;
      configs.insert(configs.begin(), slightlyTunedConfig2);
      configs.insert(configs.begin(), slightlyTunedConfig);
      configs.insert(configs.begin(), current);
      return configs;
    }

    static VkResult create(
      const TuningContext&,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      VkResult result = pipelines.createXgemmBatched(
        pipelines.xgemmBatchedFp32,
        config.xgemm,
        config.xgemm16,
        config.vulkan
      );
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.xgemmBatchedFp32);
      return result;
    }
  };

  template<>
  struct StopsOnReferenceImplFail<Xgemm16Tuner> {
    static bool value(const VulkanTuneParams&) { return true; }
  };

  template<int ConvSize, bool InputTransform, bool UseNHWC>
  struct ConvTuner {
    static string name() {
      return string(ConvSize == 3 ? "conv3x3" : "conv5x5") +
        (InputTransform ? "InputTransform" : "OutputTransform");
    }

    static ConvTuneParams& params(VulkanTuneParams& config) { return ConvSize == 3 ? config.conv3x3 : config.conv5x5; }
    static const ConvTuneParams& params(const VulkanTuneParams& config) { return ConvSize == 3 ? config.conv3x3 : config.conv5x5; }

    static bool isValid(const VulkanTuneParams& config) {
      const ConvTuneParams& conv = params(config);
      return conv.isValid(ConvSize);
    }

    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      if(InputTransform) {
        if(UseNHWC) {
          params(result).inputTransformNHWCLocalXSize = params(defaults).inputTransformNHWCLocalXSize;
          params(result).inputTransformNHWCLocalYSize = params(defaults).inputTransformNHWCLocalYSize;
        }
        else {
          params(result).inputTransformLocalXSize = params(defaults).inputTransformLocalXSize;
          params(result).inputTransformLocalYSize = params(defaults).inputTransformLocalYSize;
        }
      }
      else {
        if(UseNHWC) {
          params(result).outputTransformNHWCLocalXSize = params(defaults).outputTransformNHWCLocalXSize;
          params(result).outputTransformNHWCLocalYSize = params(defaults).outputTransformNHWCLocalYSize;
        }
        else {
          params(result).outputTransformLocalXSize = params(defaults).outputTransformLocalXSize;
          params(result).outputTransformLocalYSize = params(defaults).outputTransformLocalYSize;
          params(result).outputTransformLocalZSize = params(defaults).outputTransformLocalZSize;
        }
      }
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool full,
      const TuningContext&
    ) {
      vector<VulkanTuneParams> configs = {current};
      if(InputTransform) {
        addCandidates(configs, vector<int>{1,2,4,8,16,32,64,128}, [](VulkanTuneParams& p, int v) {
          if(UseNHWC)
            params(p).inputTransformNHWCLocalXSize = v;
          else params(p).inputTransformLocalXSize = v;
        });
        addCandidates(configs, full ? vector<int>{1,2,4,8,16,32,64} : vector<int>{1,2,4,8,16,32}, [](VulkanTuneParams& p, int v) {
          if(UseNHWC)
            params(p).inputTransformNHWCLocalYSize = v;
          else params(p).inputTransformLocalYSize = v;
        });
      }
      else {
        addCandidates(configs, full ? vector<int>{1,2,4,8,16,32,64} : vector<int>{1,2,8,16,32}, [](VulkanTuneParams& p, int v) {
          if(UseNHWC)
            params(p).outputTransformNHWCLocalXSize = v;
          else params(p).outputTransformLocalXSize = v;
        });
        addCandidates(configs, full ? vector<int>{1,2,4,8,16,32,64} : vector<int>{1,2,4,16,32}, [](VulkanTuneParams& p, int v) {
          if(UseNHWC)
            params(p).outputTransformNHWCLocalYSize = v;
          else params(p).outputTransformLocalYSize = v;
        });
        if(!UseNHWC)
          addCandidates(
            configs,
            full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8,16},
            [](VulkanTuneParams& p, int v) { params(p).outputTransformLocalZSize = v; }
          );
      }
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }),
        configs.end()
      );
      configs.insert(configs.begin(), current);
      return configs;
    }

    static VkResult create(
      const TuningContext&,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      Pipeline& pipeline = InputTransform
        ? (UseNHWC
            ? (ConvSize == 3 ? pipelines.winogradInputTransform3x3NHWC : pipelines.winogradInputTransform5x5NHWC)
            : (ConvSize == 3 ? pipelines.winogradInputTransform3x3 : pipelines.winogradInputTransform5x5))
        : (UseNHWC
            ? (ConvSize == 3 ? pipelines.winogradOutputTransform3x3NHWC : pipelines.winogradOutputTransform5x5NHWC)
            : (ConvSize == 3 ? pipelines.winogradOutputTransform3x3 : pipelines.winogradOutputTransform5x5));
      VkResult result = InputTransform
        ? pipelines.createWinogradInputTransform(pipeline, params(config), ConvSize, config.vulkan, UseNHWC)
        : pipelines.createWinogradOutputTransform(pipeline, params(config), ConvSize, config.vulkan, UseNHWC);
      if(result == VK_SUCCESS) {
        pipeline.name += UseNHWC ? "_nhwc" : "_nchw";
        pipeline.name += ConvSize == 3 ? "_3x3" : "_5x5";
        targets.push_back(&pipeline);
      }
      return result;
    }
  };

  struct Conv3x3InputTuner : ConvTuner<3,true,false> {};
  struct Conv3x3OutputTuner : ConvTuner<3,false,false> {};
  struct Conv3x3InputNHWCTuner : ConvTuner<3,true,true> {};
  struct Conv3x3OutputNHWCTuner : ConvTuner<3,false,true> {};
  struct Conv5x5InputTuner : ConvTuner<5,true,false> {};
  struct Conv5x5OutputTuner : ConvTuner<5,false,false> {};
  struct Conv5x5InputNHWCTuner : ConvTuner<5,true,true> {};
  struct Conv5x5OutputNHWCTuner : ConvTuner<5,false,true> {};

  template<>
  struct KeepsCurrentConfigFirst<Conv3x3InputTuner> {
    static constexpr bool value = true;
  };

  template<>
  struct KeepsCurrentConfigFirst<Conv3x3OutputTuner> {
    static constexpr bool value = true;
  };

  template<>
  struct KeepsCurrentConfigFirst<Conv3x3InputNHWCTuner> {
    static constexpr bool value = true;
  };

  template<>
  struct KeepsCurrentConfigFirst<Conv3x3OutputNHWCTuner> {
    static constexpr bool value = true;
  };

  struct GPoolTuner {
    static string name() { return "gPool"; }

    static bool isValid(const VulkanTuneParams& config) { return config.gPool.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.gPool = defaults.gPool;
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool full,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(
        configs,
        full ? vector<int>{1,2,4,8,16,32,64} : vector<int>{1,2,4,8,16,32},
        [](VulkanTuneParams& p, int v) { p.gPool.XYSTRIDE = v; }
      );
      addCandidates(
        configs,
        powersOfTwoUpTo(std::min(full ? 64 : 32, std::max(1, context.modelInfo.gpoolNumChannels))),
        [](VulkanTuneParams& p, int v) { p.gPool.CHANNELSTRIDE = v; }
      );
      addCandidates(
        configs,
        powersOfTwoUpTo(std::min(4, std::max(1, context.batchSize))),
        [](VulkanTuneParams& p, int v) { p.gPool.BATCHSTRIDE = v; }
      );
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }),
        configs.end()
      );
      configs.insert(configs.begin(), current);
      return configs;
    }

    static VkResult create(
      const TuningContext&,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      VkResult result = pipelines.createGlobalPoolingChannelsFp32(
        pipelines.globalPoolingChannelsFp32,
        config.gPool,
        config.vulkan,
        usesGenericNHWC(config)
      );
      if(result != VK_SUCCESS)
        return result;
      targets.push_back(&pipelines.globalPoolingChannelsFp32);
      return result;
    }
  };

  template<>
  struct KeepsCurrentConfigFirst<GPoolTuner> {
    static constexpr bool value = true;
  };

  struct PointwiseTuner {
    static string name() { return "pointwise"; }

    static bool isValid(const VulkanTuneParams& config) { return config.pointwise.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.pointwise = defaults.pointwise;
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool full,
      const TuningContext&
    ) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(
        configs,
        full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8,16},
        [](VulkanTuneParams& p, int v) { p.pointwise.ELTS_PER_THREAD = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{32,64,128,256,512} : vector<int>{32,64,128,256},
        [](VulkanTuneParams& p, int v) { p.pointwise.LOCAL_SIZE = v; }
      );
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }),
        configs.end()
      );
      configs.insert(configs.begin(), current);
      return configs;
    }

    static VkResult create(
      const TuningContext& context,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      VkResult result = pipelines.createAddPointWise(pipelines.addPointWise, config.pointwise, config.vulkan);
      if(result != VK_SUCCESS)
        return result;
      targets.push_back(&pipelines.addPointWise);
      const bool hasSwiGLU = usesNonCooperativeNCHW(config) && !context.modelInfo.residualChannels.empty()
        ? !context.modelInfo.swiGLUChannels.empty() : context.modelInfo.transformerFFNChannels > 0;
      if(hasSwiGLU) {
        result = pipelines.createTransformerSwiGLU(
          pipelines.transformerSwiGLU,
          config.pointwise,
          config.vulkan
        );
        if(result == VK_SUCCESS)
          targets.push_back(&pipelines.transformerSwiGLU);
      }
      return result;
    }
  };

  template<>
  struct KeepsCurrentConfigFirst<PointwiseTuner> {
    static constexpr bool value = true;
  };

  struct AddChannelBiasesTuner {
    static string name() { return "addChannelBiases"; }

    static bool isValid(const VulkanTuneParams& config) { return config.addChannelBiases.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.addChannelBiases = defaults.addChannelBiases;
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool,
      const TuningContext&
    ) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(
        configs,
        vector<int>{1,2,4},
        [](VulkanTuneParams& p, int v) { p.addChannelBiases.XY_ELTS_PER_THREAD = v; }
      );
      addCandidates(
        configs,
        vector<int>{1,2,4,8},
        [](VulkanTuneParams& p, int v) { p.addChannelBiases.NC_ELTS_PER_THREAD = v; }
      );
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }),
        configs.end()
      );
      configs.insert(configs.begin(), current);
      return configs;
    }

    static VkResult create(
      const TuningContext&,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      VkResult result = pipelines.createAddChannelBias(
        pipelines.addChannelBias,
        config.addChannelBiases,
        config.vulkan,
        usesGenericNHWC(config)
      );
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.addChannelBias);
      return result;
    }
  };

  template<>
  struct KeepsCurrentConfigFirst<AddChannelBiasesTuner> {
    static constexpr bool value = true;
  };

  struct TransformerScaleDotProductTuner {
    static string name() { return "transformerScaleDotProduct"; }

    // Temporary diagnostic for the Tegra X1 scalar Attention device loss.
    static constexpr uint32_t tegraDebugSharedMemoryLimit = 32 * 1024;
    static bool useTegraDebugLimit(const TuningContext& context) {
      if(context.device == nullptr || context.device->info.deviceName.find("Tegra X1") == string::npos)
        return false;
      const char* debug = std::getenv("KATAGO_VULKAN_TEGRA_ATTN_DEBUG");
      return debug != nullptr && std::strcmp(debug, "1") == 0;
    }

    static bool isValid(const VulkanTuneParams& config) { return config.transformer.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.transformer = defaults.transformer;
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool full,
      const TuningContext& context
    ) {
      if(context.logger != nullptr && useTegraDebugLimit(context)) {
        const uint32_t deviceLimit = context.device->info.properties.limits.maxComputeSharedMemorySize;
        context.logger->write(
          "Vulkan Tegra X1 debug: scalar Attention shared memory limit " +
          to_string(std::min(deviceLimit, tegraDebugSharedMemoryLimit)) +
          " bytes (device limit " + to_string(deviceLimit) + " bytes)"
        );
      }
      VulkanTuneParams defaults;
      VulkanTuneParams naive = current;
      naive.transformer.USE_TILED_ATTN = 0;
      naive.transformer.ATTN_BLOCK_Q = defaults.transformer.ATTN_BLOCK_Q;
      naive.transformer.ATTN_BLOCK_KV = defaults.transformer.ATTN_BLOCK_KV;
      naive.transformer.Q_PER_THREAD = defaults.transformer.Q_PER_THREAD;
      VulkanTuneParams tiled = current;
      tiled.transformer.USE_TILED_ATTN = 1;
      vector<VulkanTuneParams> configs = {tiled};
      addCandidates(
        configs,
        full ? vector<int>{8,16,32,64,128,256} : vector<int>{8,16,32,64,128,256},
        [](VulkanTuneParams& p, int v) { p.transformer.ATTN_BLOCK_Q = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{8,16,32,64,128} : vector<int>{8,16,32,64,128},
        [](VulkanTuneParams& p, int v) { p.transformer.ATTN_BLOCK_KV = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{1,2,4,8} : vector<int>{1,2,4},
        [](VulkanTuneParams& p, int v) { p.transformer.Q_PER_THREAD = v; }
      );
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }),
        configs.end()
      );
      configs.insert(configs.begin(), naive);
      return configs;
    }

    static VkResult create(
      const TuningContext& context,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      if(context.modelInfo.transformerHeadDim <= 0 || context.modelInfo.transformerVHeadDim <= 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
      if(config.transformer.USE_TILED_ATTN) {
        VkResult result = pipelines.createTransformerScaleDotProduct(
          pipelines.transformerScaleDotProduct,
          config.transformer,
          context.modelInfo.transformerHeadDim,
          context.modelInfo.transformerVHeadDim,
          config.vulkan
        );
        if(result == VK_SUCCESS)
          targets.push_back(&pipelines.transformerScaleDotProduct);
        return result;
      }
      VkResult result = pipelines.createTransformerScaleDotProductNaive(
        pipelines.transformerScaleDotProductNaive,
        context.modelInfo.transformerHeadDim,
        context.modelInfo.transformerVHeadDim,
        config.vulkan
      );
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.transformerScaleDotProductNaive);
      return result;
    }
  };

  template<>
  bool isValidTuningConfig<TransformerScaleDotProductTuner>(
    const TuningContext& context, const VulkanTuneParams& config
  ) {
    if(context.device == nullptr)
      return false;
    const auto& model = context.modelInfo;
    auto limits = context.device->info.properties.limits;
    if(!context.attentionProbe && TransformerScaleDotProductTuner::useTegraDebugLimit(context))
      limits.maxComputeSharedMemorySize =
        std::min(limits.maxComputeSharedMemorySize, TransformerScaleDotProductTuner::tegraDebugSharedMemoryLimit);
    if(!config.transformer.isSupported(limits, model.transformerHeadDim, model.transformerVHeadDim))
      return false;

    const bool comparesPaths = comparesScalarAndCooperativeAttention(context, config);
    const bool useFP16Storage = config.vulkan.canUseFP16Storage && config.vulkan.canUseFP16Compute &&
      config.vulkan.shouldUseFP16Storage;
    const int logicalXYSize = std::max(1, context.nnXLen * context.nnYLen);
    const int seqLen = VulkanTuner::getPaddedNNXYLen(config, useFP16Storage, logicalXYSize);
    int maxHeads = model.transformerNumHeads;
    if(!comparesPaths)
      for(const auto& shape: model.attentionShapes)
        maxHeads = std::max(maxHeads, shape.heads);
    return config.transformer.getDispatchGroupCountX(seqLen) <= limits.maxComputeWorkGroupCount[0] &&
      static_cast<uint64_t>(std::max(1, context.batchSize)) * std::max(1, maxHeads) <= limits.maxComputeWorkGroupCount[1] &&
      limits.maxComputeWorkGroupCount[2] >= 1;
  }

  // Temporary scalar Attention probe. Use an explicit disposable -output with katago tuner.
  const char* getAttentionProbeCase(const VulkanDevice* device) {
    if(device == nullptr)
      return nullptr;
    if(const char* probeCase = std::getenv("KATAGO_VULKAN_ATTN_CASE"))
      return probeCase;
    return device->info.deviceName.find("Tegra X1") != string::npos
      ? std::getenv("KATAGO_VULKAN_TEGRA_ATTN_CASE")
      : nullptr;
  }

  void runAttentionProbe(
    const char* caseSpec, const VulkanDevice* device, int batchSize, int nnXLen, int nnYLen,
    const VulkanTuner::ModelInfoForTuning& modelInfo, Logger* logger, VulkanTuneParams& tunedConfig
  ) {
    const vector<string> values = Global::split(caseSpec, ',');
    VulkanTuneParams config = tunedConfig;
    auto& params = config.transformer;
    if(values.size() != 3 ||
       !Global::tryStringToInt(values[0], params.ATTN_BLOCK_Q) ||
       !Global::tryStringToInt(values[1], params.ATTN_BLOCK_KV) ||
       !Global::tryStringToInt(values[2], params.Q_PER_THREAD))
      throw StringError("Attention probe case must be Q,KV,Q_PER_THREAD");
    params.USE_TILED_ATTN = 1;
    config.vulkan = VulkanTuner::getHardwareParams(device->info);
    config.vulkan.shouldUseFP16Storage = false;
    config.vulkan.shouldUseFP16Compute = false;
    config.vulkan.shouldUseCooperativeMatrix = false;
    config.vulkan.shouldUseNHWC = false;

    const char* dummy = std::getenv("KATAGO_VULKAN_TEGRA_ATTN_DUMMY");
    const bool useDummy = dummy != nullptr && std::strcmp(dummy, "1") == 0;
    const char* tight = std::getenv("KATAGO_VULKAN_TEGRA_ATTN_TIGHT_BUFFERS");
    const char* fresh = std::getenv("KATAGO_VULKAN_TEGRA_ATTN_FRESH_COMMANDS");
    const char* dispatchOnly = std::getenv("KATAGO_VULKAN_ATTN_DISPATCH_ONLY");
    const bool skipBoth = dispatchOnly != nullptr && std::strcmp(dispatchOnly, "1") == 0;
    const char* skipTimestamps = std::getenv("KATAGO_VULKAN_ATTN_SKIP_TIMESTAMPS");
    const char* skipSnapshots = std::getenv("KATAGO_VULKAN_ATTN_SKIP_SNAPSHOTS");
    const char* probeBatch = std::getenv("KATAGO_VULKAN_TEGRA_ATTN_BATCH");
    if(probeBatch != nullptr) {
      int parsedBatch;
      if(!Global::tryStringToInt(probeBatch, parsedBatch) || parsedBatch < 1 || parsedBatch > batchSize)
        throw StringError("KATAGO_VULKAN_TEGRA_ATTN_BATCH must be between 1 and " + to_string(batchSize));
      batchSize = parsedBatch;
    }
    int dispatchBatchSize = batchSize;
    const char* probeDispatchBatch = std::getenv("KATAGO_VULKAN_TEGRA_ATTN_DISPATCH_BATCH");
    if(probeDispatchBatch != nullptr &&
       (!Global::tryStringToInt(probeDispatchBatch, dispatchBatchSize) ||
        dispatchBatchSize < 1 || dispatchBatchSize > batchSize))
      throw StringError("KATAGO_VULKAN_TEGRA_ATTN_DISPATCH_BATCH must be between 1 and the probe batch size");
    VulkanTimestampTimer timer(device);
    TuningContext context{device, batchSize, nnXLen, nnYLen, modelInfo, false, {}, logger, &timer, true};
    context.attentionProbe = true;
    context.tegraAttentionTightBuffers = tight != nullptr && std::strcmp(tight, "1") == 0;
    context.tegraAttentionFreshCommands = fresh != nullptr && std::strcmp(fresh, "1") == 0;
    context.attentionSkipTimestamps = skipBoth || (skipTimestamps != nullptr && std::strcmp(skipTimestamps, "1") == 0);
    context.attentionSkipSnapshots = skipBoth || (skipSnapshots != nullptr && std::strcmp(skipSnapshots, "1") == 0);
    using Tuner = TransformerScaleDotProductTuner;
    if(!isValidTuningConfig<Tuner>(context, config))
      throw StringError("Attention probe candidate violates the model or device limits");
    if(useDummy && device->dummyQueue == VK_NULL_HANDLE)
      throw StringError("Attention probe requested a dummy GPU workload without a second queue");

    if(logger != nullptr)
      logger->write(
        "Attention probe " + describeTuningParams(Tuner::name(), config) +
        " board=" + to_string(nnXLen) + "x" + to_string(nnYLen) + " batch=" + to_string(batchSize) +
        " dispatchBatch=" + to_string(dispatchBatchSize) +
        " dummy=" + to_string(useDummy) + " tightBuffers=" + to_string(context.tegraAttentionTightBuffers) +
        " freshCommands=" + to_string(context.tegraAttentionFreshCommands) +
        " timestamps=" + to_string(!context.attentionSkipTimestamps) +
        " snapshots=" + to_string(!context.attentionSkipSnapshots)
      );
    VulkanDummyThread dummyThread(device, logger);
    if(useDummy)
      dummyThread.start();
    auto plan = makeMeasurementPlan(Tuner::name(), context);
    plan.batchSizes = {dispatchBatchSize};
    const auto measurements = measureConfigs<Tuner>(context, {config}, plan, nullptr, !context.attentionSkipTimestamps);
    dummyThread.stopAndJoin();
    requireTuningDevice(context, "Attention probe");
    if(measurements.values.empty() || measurements.values.front().score <= 0.0)
      throw StringError("Attention probe failed");
    if(logger != nullptr)
      logger->write(
        "Attention probe PASS " + string(context.attentionSkipTimestamps ? "wall calls/sec=" : "calls/sec=") +
        Global::doubleToString(measurements.values.front().callsPerSecond)
      );
    tunedConfig = std::move(config);
  }

  template<bool NHWC>
  struct TransformerScaleDotProductCooperativeMatrixTuner {
    static string name() {
      return NHWC ? "transformerScaleDotProductCooperativeMatrixNHWC" :
                    "transformerScaleDotProductCooperativeMatrixNCHW";
    }

    static TransformerCooperativeMatrixTuneParams& params(VulkanTuneParams& config) {
      return NHWC ? config.transformerCooperativeMatrixNHWC : config.transformerCooperativeMatrixNCHW;
    }

    static const TransformerCooperativeMatrixTuneParams& params(const VulkanTuneParams& config) {
      return NHWC ? config.transformerCooperativeMatrixNHWC : config.transformerCooperativeMatrixNCHW;
    }

    static bool isValid(const VulkanTuneParams& config) { return params(config).isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      params(result) = params(defaults);
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool full,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs = {current};
      if(context.device == nullptr || context.modelInfo.transformerHeadDim <= 0 || context.modelInfo.transformerVHeadDim <= 0)
        return configs;
      vector<VulkanTuneParams> nativeConfigs;
      if(NHWC && context.device->info.canUseRegisterAttentionNHWC &&
         context.modelInfo.transformerHeadDim == 32 && context.modelInfo.transformerVHeadDim == 32) {
        for(int qTiles: {2, 4, 8})
          for(int kvStages: {2, 4, 8}) {
            VulkanTuneParams candidate = current;
            auto& p = params(candidate);
            p.ATTN_BLOCK_Q = 16 * qTiles;
            p.ATTN_BLOCK_KV = 16 * kvStages;
            p.COOP_M_SIZE = p.COOP_N_SIZE = p.COOP_K_SIZE = p.COOP_PV_N_SIZE = 16;
            p.COOP_SUBGROUP_SIZE = 32;
            p.COOP_Q_TILES_PER_WORKGROUP = qTiles;
            p.COOP_USE_NV2 = 1;
            nativeConfigs.push_back(candidate);
          }
      }
      for(const CooperativeMatrixTuneShape& shape: context.cooperativeMatrixTuneShapes) {
        if(shape.accType != 32 || shape.MSize > 256 || shape.NSize > 128 || shape.MSize > static_cast<int>(shape.subgroupSize))
          continue;
        vector<int> pvNSizes;
        for(const CooperativeMatrixTuneShape& pvShape: getCooperativeMatrixTuneShapes(context.device)) {
          if(pvShape.accType == shape.accType && pvShape.MSize == shape.MSize && pvShape.KSize == shape.NSize)
            pvNSizes.push_back(pvShape.NSize);
        }
        sort(pvNSizes.begin(), pvNSizes.end());
        pvNSizes.erase(unique(pvNSizes.begin(), pvNSizes.end()), pvNSizes.end());
        for(int pvNSize: pvNSizes) {
          for(int qTiles: {1, 2, 4, 8, 16}) {
            if(shape.MSize * qTiles > 256)
              break;
            for(int kvStages: (NHWC ? (full ? vector<int>{1, 2, 4, 8} : vector<int>{1, 2, 4}) : vector<int>{1})) {
              if(shape.NSize * kvStages > 128)
                continue;
              VulkanTuneParams candidate = current;
              TransformerCooperativeMatrixTuneParams& tuneParams = params(candidate);
              tuneParams.COOP_USE_NV2 = 0;
              tuneParams.ATTN_BLOCK_Q = shape.MSize * qTiles;
              tuneParams.ATTN_BLOCK_KV = shape.NSize * kvStages;
              tuneParams.COOP_ACC_TYPE = shape.accType;
              tuneParams.COOP_M_SIZE = shape.MSize;
              tuneParams.COOP_N_SIZE = shape.NSize;
              tuneParams.COOP_K_SIZE = shape.KSize;
              tuneParams.COOP_SUBGROUP_SIZE = shape.subgroupSize;
              tuneParams.COOP_Q_TILES_PER_WORKGROUP = qTiles;
              tuneParams.COOP_PV_N_SIZE = pvNSize;
              configs.push_back(candidate);
            }
          }
        }
      }
      configs.insert(configs.end(), nativeConfigs.begin(), nativeConfigs.end());
      configs.erase(remove_if(configs.begin(), configs.end(), [&context](const VulkanTuneParams& config) {
        return !isValidCooperativeMatrixConfig(
          context.device->info, params(config), context.modelInfo.transformerHeadDim,
          context.modelInfo.transformerVHeadDim, NHWC
        );
      }), configs.end());
      return configs;
    }

    static VkResult create(
      const TuningContext& context,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      if(context.modelInfo.transformerHeadDim <= 0 || context.modelInfo.transformerVHeadDim <= 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
      Pipeline& pipeline = NHWC ? pipelines.transformerScaleDotProductCooperativeNHWC : pipelines.transformerScaleDotProductCooperative;
      VkResult result = pipelines.createTransformerScaleDotProductCooperative(
        pipeline, params(config), context.modelInfo.transformerHeadDim, context.modelInfo.transformerVHeadDim,
        config.vulkan, NHWC
      );
      if(result == VK_SUCCESS)
        targets.push_back(&pipeline);
      return result;
    }
  };

  template<bool NHWC>
  bool isValidTransformerCooperativeConfig(
    const TuningContext& context,
    const VulkanTuneParams& config
  ) {
    const auto& params = TransformerScaleDotProductCooperativeMatrixTuner<NHWC>::params(config);
    return context.device != nullptr && config.vulkan.canUseCooperativeMatrix &&
           config.vulkan.canUseFP16Storage && config.vulkan.canUseFP16Compute &&
           isValidCooperativeMatrixConfig(
             context.device->info, params,
             context.modelInfo.transformerHeadDim, context.modelInfo.transformerVHeadDim, NHWC
           );
  }

  template<>
  bool isValidTuningConfig<TransformerScaleDotProductCooperativeMatrixTuner<false>>(
    const TuningContext& context, const VulkanTuneParams& config
  ) { return isValidTransformerCooperativeConfig<false>(context, config); }

  template<>
  bool isValidTuningConfig<TransformerScaleDotProductCooperativeMatrixTuner<true>>(
    const TuningContext& context, const VulkanTuneParams& config
  ) { return isValidTransformerCooperativeConfig<true>(context, config); }

  void validateTransformerAttentionMasks(const TuningContext& context, const VulkanTuneParams& config) {
    if(!isValidTransformerCooperativeConfig<true>(context, config))
      return;
    using Tuner = TransformerScaleDotProductCooperativeMatrixTuner<true>;
    vk_shader::ComputePipelines pipelines(
      context.device->device,
      context.device->info,
      nullptr,
      false
    );
    vector<const Pipeline*> targets;
    VkResult result = Tuner::create(context, config, pipelines, targets);
    if(result != VK_SUCCESS || targets.empty())
      throw StringError("Could not create NHWC attention mask-validation pipeline");
    for(int pattern: {1, 2, 3}) {
      TuningContext validation = context;
      validation.attentionValidationMask = pattern;
      vector<float> output, reference;
      double rate = 0.0;
      double errorProp = 0.0;
      string error;
      if(context.timer == nullptr || !context.timer->measure(targets, config, validation,
           makeMeasurementPlan(Tuner::name(), validation), rate, output, errorProp, error, &reference))
        throw StringError("NHWC attention mask validation failed: " + error);
      errorProp = VulkanTuner::computeErrorProp(reference, output);
      if(errorProp > VulkanTuner::COOPERATIVE_MATRIX_ROPE_ERROR_TOLERANCE)
        throw StringError("NHWC attention mask validation exceeded CPU error tolerance");
      if(context.logger != nullptr)
        context.logger->write("NHWC attention CPU mask check: pattern=" + to_string(pattern) +
          " (1=all-zero, 2=non-prefix, 3=prefix), ErrorProp=" + Global::doubleToString(errorProp));
    }
  }

  template<bool NHWC>
  double tuneTransformerCooperativeMatrix(
    const TuningContext& context,
    VulkanTuneParams& config,
    double scalarCallsPerSecond
  ) {
    if(!config.vulkan.canUseCooperativeMatrix || !config.vulkan.canUseFP16Storage ||
       !config.vulkan.canUseFP16Compute || !config.vulkan.shouldUseFP16Storage ||
       (NHWC && !config.vulkan.shouldUseFP16Compute)) {
      if(context.logger != nullptr)
        context.logger->write(
          "Skipping Vulkan " + TransformerScaleDotProductCooperativeMatrixTuner<NHWC>::name() +
          " tuning: cooperative-matrix FP16 attention is unavailable"
        );
      return 0.0;
    }

    VulkanTuneParams cooperativeConfig = config;
    cooperativeConfig.vulkan.shouldUseFP16Storage = true;
    cooperativeConfig.vulkan.shouldUseFP16Compute = true;
    const double callsPerSecond = runTuner<TransformerScaleDotProductCooperativeMatrixTuner<NHWC>>(
      context, cooperativeConfig
    );
    const bool selected = NHWC
      ? isfinite(callsPerSecond) && callsPerSecond > 0.0
      : VulkanTuner::isFastEnough(
        callsPerSecond,
        scalarCallsPerSecond,
        VulkanTuner::COOPERATIVE_MATRIX_MIN_THROUGHPUT_RATIO
      );
    if constexpr(NHWC) {
      if(selected) {
        validateTransformerAttentionMasks(context, config);
        validateTransformerAttentionMasks(context, cooperativeConfig);
      }
    }
    if(isfinite(callsPerSecond) && callsPerSecond > 0.0)
      TransformerScaleDotProductCooperativeMatrixTuner<NHWC>::params(config) =
        TransformerScaleDotProductCooperativeMatrixTuner<NHWC>::params(cooperativeConfig);
    if constexpr(NHWC)
      config.vulkan.shouldUseTransformerCooperativeMatrixNHWC = selected;
    else
      config.vulkan.shouldUseTransformerCooperativeMatrixNCHW = selected;
    if(context.logger != nullptr)
      context.logger->write(
        "Vulkan " + TransformerScaleDotProductCooperativeMatrixTuner<NHWC>::name() +
          " comparison: scalar=" + (isfinite(scalarCallsPerSecond)
            ? Global::strprintf("%.6g", scalarCallsPerSecond) : string("not_measured")) +
        " calls/s, cooperative=" + Global::strprintf("%.6g", callsPerSecond) +
        " calls/s, required_ratio=" +
        (NHWC ? "n/a" : Global::strprintf("%.2f", VulkanTuner::COOPERATIVE_MATRIX_MIN_THROUGHPUT_RATIO)) +
        ", selection_policy=" + (NHWC ? "valid_nhwc_candidate" : "scalar_ratio") +
        ", selected=" + (selected ? "true" : "false")
      );
    return callsPerSecond;
  }

  struct TransformerRMSNormTuner {
    static string name() { return "transformerRMSNorm"; }

    static bool isValid(const VulkanTuneParams& config) { return config.rmsNorm.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.rmsNorm = defaults.rmsNorm;
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool full,
      const TuningContext&
    ) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(
        configs,
        full ? vector<int>{32,64,128,256,512} : vector<int>{32,64,128,256},
        [](VulkanTuneParams& p, int v) { p.rmsNorm.WG_C_SIZE = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8,16},
        [](VulkanTuneParams& p, int v) { p.rmsNorm.WG_XY_SIZE = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{1,2,4,8,16} : vector<int>{1,2,4,8},
        [](VulkanTuneParams& p, int v) { p.rmsNorm.C_PER_THREAD = v; }
      );
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }),
        configs.end()
      );
      configs.insert(configs.begin(), current);
      return configs;
    }

    static VkResult create(
      const TuningContext&,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      VkResult result = pipelines.createTransformerRMSNorm(
        pipelines.transformerRmsNorm, config.rmsNorm, config.vulkan, usesGenericNHWC(config)
      );
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.transformerRmsNorm);
      return result;
    }
  };

  template<>
  struct KeepsCurrentConfigFirst<TransformerRMSNormTuner> {
    static constexpr bool value = true;
  };

  struct SpatialRMSNormTuner {
    static string name() { return "spatialRMSNorm"; }

    static bool isValid(const VulkanTuneParams& config) { return config.spatialRMSNorm.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.spatialRMSNorm = defaults.spatialRMSNorm;
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool full,
      const TuningContext&
    ) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(
        configs,
        full ? vector<int>{32,64,128,256,512,1024} : vector<int>{32,64,128,256,512},
        [](VulkanTuneParams& p, int v) { p.spatialRMSNorm.TILE_SIZE = v; }
      );
      addCandidates(
        configs,
        full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8,16},
        [](VulkanTuneParams& p, int v) { p.spatialRMSNorm.APPLY_ELTS_PER_THREAD = v; }
      );
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }),
        configs.end()
      );
      configs.insert(configs.begin(), current);
      return configs;
    }

    static VkResult create(
      const TuningContext&,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      const bool useNHWC = usesGenericNHWC(config);
      VkResult result = pipelines.createTransformerSpatialRMSNormSumSq(
        pipelines.transformerSpatialRMSNormSumSq, config.spatialRMSNorm, config.vulkan, useNHWC
      );
      if(result != VK_SUCCESS)
        return result;
      targets.push_back(&pipelines.transformerSpatialRMSNormSumSq);
      result = pipelines.createTransformerSpatialRMSNormReduce(
        pipelines.transformerSpatialRMSNormReduce,
        config.spatialRMSNorm,
        config.vulkan
      );
      if(result != VK_SUCCESS)
        return result;
      targets.push_back(&pipelines.transformerSpatialRMSNormReduce);
      result = pipelines.createTransformerSpatialRMSNormApply(
        pipelines.transformerSpatialRMSNormApply, config.spatialRMSNorm, config.vulkan, useNHWC
      );
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.transformerSpatialRMSNormApply);
      return result;
    }
  };

  template<>
  struct KeepsCurrentConfigFirst<SpatialRMSNormTuner> {
    static constexpr bool value = true;
  };

  void runNonGemmTuners(const TuningContext& context, VulkanTuneParams& config) {
    const auto plan = VulkanTuner::getNonGemmTuningPlan(context.modelInfo, config);
    if(plan.winograd && usesNonCooperativeNCHW(config) && context.modelInfo.maxConvChannels3x3 == 0) {
      runTuner<Conv5x5InputTuner>(context, config);
      runTuner<Conv5x5OutputTuner>(context, config);
    }
    else if(plan.winograd) {
      if(!config.vulkan.shouldUseNHWC) {
        VulkanTuneParams nchwConfig = config;
        runTuner<Conv3x3InputTuner>(context, nchwConfig);
        runTuner<Conv3x3OutputTuner>(context, nchwConfig);
        config.conv3x3.inputTransformLocalXSize = nchwConfig.conv3x3.inputTransformLocalXSize;
        config.conv3x3.inputTransformLocalYSize = nchwConfig.conv3x3.inputTransformLocalYSize;
        config.conv3x3.outputTransformLocalXSize = nchwConfig.conv3x3.outputTransformLocalXSize;
        config.conv3x3.outputTransformLocalYSize = nchwConfig.conv3x3.outputTransformLocalYSize;
        config.conv3x3.outputTransformLocalZSize = nchwConfig.conv3x3.outputTransformLocalZSize;
      }
      else if(config.vulkan.canUseCooperativeMatrix && config.vulkan.canUseFP16Storage &&
              config.vulkan.canUseFP16Compute && config.hgemmCooperativeMatrixNHWC.isValid()) {
        VulkanTuneParams nhwcConfig = config;
        nhwcConfig.vulkan.shouldUseNHWC = true;
        nhwcConfig.vulkan.shouldUseCooperativeMatrix = true;
        nhwcConfig.vulkan.shouldUseFP16Storage = true;
        nhwcConfig.vulkan.shouldUseFP16Compute = true;
        runTuner<Conv3x3InputNHWCTuner>(context, nhwcConfig);
        runTuner<Conv3x3OutputNHWCTuner>(context, nhwcConfig);
        config.conv3x3.inputTransformNHWCLocalXSize = nhwcConfig.conv3x3.inputTransformNHWCLocalXSize;
        config.conv3x3.inputTransformNHWCLocalYSize = nhwcConfig.conv3x3.inputTransformNHWCLocalYSize;
        config.conv3x3.outputTransformNHWCLocalXSize = nhwcConfig.conv3x3.outputTransformNHWCLocalXSize;
        config.conv3x3.outputTransformNHWCLocalYSize = nhwcConfig.conv3x3.outputTransformNHWCLocalYSize;
      }
      else if(context.logger != nullptr) {
        context.logger->write("Skipping Vulkan NHWC Winograd tuning because FP16 cooperative-matrix support is unavailable");
      }

      if(context.modelInfo.hasConv5x5) {
        if(config.vulkan.shouldUseNHWC) {
          config.conv5x5.inputTransformNHWCLocalXSize = config.conv3x3.inputTransformNHWCLocalXSize;
          config.conv5x5.inputTransformNHWCLocalYSize = config.conv3x3.inputTransformNHWCLocalYSize;
          config.conv5x5.outputTransformNHWCLocalXSize = config.conv3x3.outputTransformNHWCLocalXSize;
          config.conv5x5.outputTransformNHWCLocalYSize = config.conv3x3.outputTransformNHWCLocalYSize;
        }
        else {
          config.conv5x5.inputTransformLocalXSize = config.conv3x3.inputTransformLocalXSize;
          config.conv5x5.inputTransformLocalYSize = config.conv3x3.inputTransformLocalYSize;
          config.conv5x5.outputTransformLocalXSize = config.conv3x3.outputTransformLocalXSize;
          config.conv5x5.outputTransformLocalYSize = config.conv3x3.outputTransformLocalYSize;
          config.conv5x5.outputTransformLocalZSize = config.conv3x3.outputTransformLocalZSize;
        }
      }
      else if(context.logger != nullptr) {
        context.logger->write("Skipping Vulkan 5x5 Winograd tuning because the model has no 5x5 convolution");
      }
    }
    runTuner<GPoolTuner>(context, config);
    if(plan.attentionNHWC)
      tuneTransformerCooperativeMatrix<true>(context, config, numeric_limits<double>::quiet_NaN());
    else if(plan.attentionNCHW) {
      const double scalarCallsPerSecond = runTuner<TransformerScaleDotProductTuner>(context, config);
      tuneTransformerCooperativeMatrix<false>(context, config, scalarCallsPerSecond);
    }
    if(plan.rmsNorm)
      runTuner<TransformerRMSNormTuner>(context, config);
    runTuner<PointwiseTuner>(context, config);
    runTuner<AddChannelBiasesTuner>(context, config);
    if(plan.spatialRMSNorm)
      runTuner<SpatialRMSNormTuner>(context, config);
  }

  bool tuneXgemm16(
    const TuningContext& context,
    VulkanTuneParams& config,
    double fp32CallsPerSecond
  ) {
    if(!config.vulkan.canUseFP16Storage || !config.vulkan.canUseFP16Compute) {
      if(context.logger != nullptr)
        context.logger->write(
          "Skipping Vulkan xgemm16 tuning: FP16 storage or compute is unavailable, selected=false"
        );
      return false;
    }
    if(!isfinite(fp32CallsPerSecond) || fp32CallsPerSecond <= 0.0) {
      if(context.logger != nullptr)
        context.logger->write(
          "Skipping Vulkan xgemm16 tuning: FP32 xgemm tuning failed, selected=false"
        );
      return false;
    }

    VulkanTuneParams tunedConfig = config;
    tunedConfig.vulkan.shouldUseFP16Storage = true;
    tunedConfig.vulkan.shouldUseFP16Compute = true;
    const double fp16CallsPerSecond = runTuner<Xgemm16Tuner>(context, tunedConfig);
    if(!isfinite(fp16CallsPerSecond) || fp16CallsPerSecond <= 0.0) {
      config.xgemm16 = config.xgemm;
      if(context.logger != nullptr)
        context.logger->write(
          "Vulkan xgemm16 tuning failed, retaining xgemm parameters, selected=false"
        );
      return false;
    }

    config.xgemm16 = tunedConfig.xgemm16;
    const bool computeIsFastEnough = VulkanTuner::isFastEnough(
      fp16CallsPerSecond, fp32CallsPerSecond, VulkanTuner::FP16_COMPUTE_MIN_THROUGHPUT_RATIO
    );
    if(context.logger != nullptr) {
      context.logger->write(
        "Vulkan xgemm16 comparison: fp32=" + Global::strprintf("%.6g", fp32CallsPerSecond) +
        " calls/s, p16s16=" + Global::strprintf("%.6g", fp16CallsPerSecond) +
        " calls/s, required_ratio=" + Global::strprintf(
          "%.2f",
          VulkanTuner::FP16_COMPUTE_MIN_THROUGHPUT_RATIO
        ) +
        ", selected=" + (computeIsFastEnough ? "true" : "false")
      );
    }
    if(!computeIsFastEnough)
      return false;

    config.vulkan.shouldUseFP16Storage = true;
    config.vulkan.shouldUseFP16Compute = true;
    if(context.logger != nullptr)
      context.logger->write("Enabling Vulkan FP16 compute due to xgemm16 throughput");
    return true;
  }

  bool tuneXgemmStorage(
    const TuningContext& context,
    VulkanTuneParams& config,
    double fp32CallsPerSecond
  ) {
    if(!config.vulkan.canUseFP16Storage || !config.vulkan.canUseFP16Compute ||
       !isfinite(fp32CallsPerSecond) || fp32CallsPerSecond <= 0.0) {
      if(context.logger != nullptr)
        context.logger->write(
          "Skipping Vulkan xgemm storage tuning: FP16 capability or FP32 xgemm baseline unavailable, selected=false"
        );
      return false;
    }

    VulkanTuneParams tunedConfig = config;
    tunedConfig.vulkan.shouldUseFP16Storage = true;
    tunedConfig.vulkan.shouldUseFP16Compute = false;
    const double fp16StorageCallsPerSecond = runTuner<XgemmTuner>(context, tunedConfig);
    const bool storageIsFastEnough = VulkanTuner::isFastEnough(
      fp16StorageCallsPerSecond, fp32CallsPerSecond, VulkanTuner::FP16_STORAGE_MIN_THROUGHPUT_RATIO
    );
    if(context.logger != nullptr) {
      context.logger->write(
        "Vulkan xgemm storage comparison: fp32=" + Global::strprintf("%.6g", fp32CallsPerSecond) +
        " calls/s, p32s16=" + Global::strprintf("%.6g", fp16StorageCallsPerSecond) +
        " calls/s, required_ratio=" + Global::strprintf(
          "%.2f",
          VulkanTuner::FP16_STORAGE_MIN_THROUGHPUT_RATIO
        ) +
        ", selected=" + (storageIsFastEnough ? "true" : "false")
      );
    }
    if(!storageIsFastEnough)
      return false;

    config.xgemm = tunedConfig.xgemm;
    config.vulkan.shouldUseFP16Storage = true;
    return true;
  }

  bool tuneCooperativeMatrices(
    const TuningContext& context,
    VulkanTuneParams& config,
    bool canUseHgemmCooperativeMatrix,
    bool canUseNCHW
  ) {
    if(!canUseHgemmCooperativeMatrix || !config.vulkan.canUseFP16Storage ||
       !config.vulkan.canUseFP16Compute)
      return false;

    VulkanTuneParams cooperativeConfig = config;
    cooperativeConfig.vulkan.shouldUseFP16Storage = true;
    cooperativeConfig.vulkan.shouldUseFP16Compute = true;
    cooperativeConfig.vulkan.shouldUseCooperativeMatrix = true;
    const bool transformer = context.modelInfo.transformerHeadDim > 0 &&
      context.modelInfo.transformerVHeadDim > 0;
    cooperativeConfig.vulkan.shouldUseNHWC = transformer;
    if(transformer) {
      const double rate = runTuner<HgemmCooperativeMatrixNHWCTunerImpl>(context, cooperativeConfig);
      if(!isfinite(rate) || rate <= 0.0) {
        if(context.logger != nullptr)
          context.logger->write("NHWC cooperative GEMM validation failed; tuning scalar fallback workloads");
        return false;
      }
    }
    else {
      const bool hasWinograd = !getTunerGemmTuneCases("hgemmCooperativeMatrix", context).empty();
      if(hasWinograd) {
        const double rate = runTuner<HgemmCooperativeMatrixTunerImpl>(context, cooperativeConfig);
        if(!isfinite(rate) || rate <= 0.0) {
          if(context.logger != nullptr)
            context.logger->write("Winograd cooperative GEMM validation failed; tuning scalar fallback workloads");
          return false;
        }
      }
      bool useNCHW = false;
      if(canUseNCHW && !getTunerGemmTuneCases("hgemmCooperativeMatrixNCHW", context).empty()) {
        const double rate = runTuner<HgemmCooperativeMatrixNCHWTunerImpl>(context, cooperativeConfig);
        const auto cases = getTunerGemmTuneCases("hgemmCooperativeMatrixNCHW", context);
        const auto& p = cooperativeConfig.hgemmCooperativeMatrixNCHW;
        useNCHW = isfinite(rate) && rate > 0.0 && any_of(cases.begin(), cases.end(), [&](const GemmTuneCase& c) {
          return c.inChannels % p.KWG == 0 && c.outChannels % p.NWG == 0;
        });
      }
      if(!hasWinograd && !useNCHW)
        return false;
      cooperativeConfig.vulkan.shouldUseHgemmCooperativeMatrixNCHW = useNCHW;
    }
    config = cooperativeConfig;
    if(context.logger != nullptr)
      context.logger->write(string("Selected validated cooperative GEMM without a scalar throughput gate: ") +
        (transformer ? "NHWC" : "NCHW/Winograd"));
    return true;
  }

  struct TransformerDualGemmSwiGLUTuner {
    static string name() { return "transformerDualGemmSwiGLU"; }

    static bool isValid(const VulkanTuneParams& config) {
      return config.transformerDualGemmSwiGLU.isValid();
    }

    static VulkanTuneParams reference(
      const VulkanTuneParams& current,
      const VulkanTuneParams& defaults
    ) {
      VulkanTuneParams result = current;
      result.transformerDualGemmSwiGLU = defaults.transformerDualGemmSwiGLU;
      result.vulkan.shouldUseTransformerDualGemmSwiGLU = false;
      return result;
    }

    static vector<VulkanTuneParams> candidates(
      const VulkanTuneParams& current,
      bool full,
      const TuningContext& context
    ) {
      vector<VulkanTuneParams> configs;
      const int cSize = VulkanTuner::getTransformerFFNInputChannelsForTuning(context.modelInfo);
      const int ffnSize = context.modelInfo.transformerFFNChannels;
      if(cSize <= 0 || ffnSize <= 0 || ffnSize > numeric_limits<int>::max() / 2 ||
         context.cooperativeMatrixTuneShapes.empty())
        return configs;

      // Match the reference model's precision and expose the CUTLASS-like
      // 128x64 block / 64x32 per-subgroup reuse geometry, not just axis scaling.
      for(const auto& shape: context.cooperativeMatrixTuneShapes) {
        if(shape.accType != current.transformerHgemmNHWC.accType || shape.KSize % 4 != 0)
          continue;
        for(int m: (full ? vector<int>{64,128} : vector<int>{64}))
          for(int n: {32,64})
            for(int k: {32,64})
              for(int sm: {1,2})
                for(int sn: {1,2})
                  for(int stages: {1,2}) {
                    VulkanTuneParams config = current;
                    auto& q = config.transformerDualGemmSwiGLU;
                    q.accType = shape.accType;
                    q.MWARP = shape.MSize;
                    q.NWARP = shape.NSize;
                    q.KDIM = shape.KSize;
                    q.subgroupSize = shape.subgroupSize;
                    q.MWG = m;
                    q.NWG = n;
                    q.KWG = k;
                    q.MWAVE = shape.MSize * sm;
                    q.NWAVE = shape.NSize * sn;
                    q.VWM = q.VWN = 1;
                    q.SB = 1;
                    q.STAGES = stages;
                    if(!q.isValid() || cSize % k != 0 || ffnSize % n != 0 ||
                       !isValidCooperativeMatrixTuneParams(context, q))
                      continue;
                    config.vulkan.shouldUseFP16Storage = true;
                    config.vulkan.shouldUseFP16Compute = true;
                    config.vulkan.shouldUseTransformerDualGemmSwiGLU = false;
                    configs.push_back(config);
                  }
      }
      return configs;
    }

    static VkResult create(
      const TuningContext&,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      VkResult result = pipelines.createTransformerDualGemmSwiGLU(
        pipelines.transformerDualGemmSwiGLU, config.transformerDualGemmSwiGLU
      );
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.transformerDualGemmSwiGLU);
      return result;
    }
  };

  bool tuneTransformerDualGemmSwiGLU(
    const TuningContext& context,
    VulkanTuneParams& config
  ) {
    if(context.onlyMissing && tuningComplete("transformerDualGemmSwiGLU", context))
      return true;
    if(!context.onlyMissing)
      config.vulkan.shouldUseTransformerDualGemmSwiGLU = false;
    if(context.device == nullptr || VulkanTuner::getTransformerFFNInputChannelsForTuning(context.modelInfo) <= 0 ||
       !usesGenericNHWC(config) ||
       none_of(context.modelInfo.transformerProjections.begin(), context.modelInfo.transformerProjections.end(),
         [](const auto& projection) { return projection.fusedFFNInput; }) ||
       context.modelInfo.transformerFFNChannels <= 0 ||
       !config.vulkan.canUseCooperativeMatrix ||
       !config.vulkan.canUseFP16Storage || !config.vulkan.canUseFP16Compute ||
       !config.vulkan.shouldUseFP16Storage || !config.vulkan.shouldUseFP16Compute ||
       context.cooperativeMatrixTuneShapes.empty()) {
      if(context.logger != nullptr)
        context.logger->write(
          "Skipping Vulkan transformerDualGemmSwiGLU tuning: transformer, cooperative-matrix, "
          "FP16, or property prerequisites unavailable; selected=false"
        );
      return false;
    }

    // Keep immutable fixtures and GPU allocations until all candidates and rechecks finish.
    VulkanTimestampTimer timer(context.device);
    logTuningStart(context, "transformerDualGemmSwiGLU");
    const int cSize = VulkanTuner::getTransformerFFNInputChannelsForTuning(context.modelInfo);
    const int ffnSize = context.modelInfo.transformerFFNChannels;
    VulkanTuneParams legacyConfig = config;
    legacyConfig.vulkan.shouldUseTransformerDualGemmSwiGLU = false;
    const bool legacyNHWC = usesGenericNHWC(config);
    legacyConfig.vulkan.shouldUseNHWC = legacyNHWC;
    int productionSpatialAlignment = legacyNHWC ? config.hgemmCooperativeMatrixNHWC.MWG : 1;
    if(config.vulkan.shouldUseHgemmCooperativeMatrixNCHW)
      productionSpatialAlignment = std::lcm(
        productionSpatialAlignment,
        config.hgemmCooperativeMatrixNCHW.getRequiredSpatialAlignment()
      );
    const int baseSpatialAlignment = productionSpatialAlignment;
    if(context.onlyMissing)
      productionSpatialAlignment = std::lcm(
        productionSpatialAlignment, config.transformerDualGemmSwiGLU.getRequiredSpatialAlignment()
      );
    const int productionHW = vk_helper::roundUpToMultipleInt(
      context.nnXLen * context.nnYLen,
      productionSpatialAlignment
    );
    const HGemmCooperativeMatrixNCHWTuneParams& nchwParams = config.hgemmCooperativeMatrixNCHW;
    bool legacyNCHW =
      config.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
      config.vulkan.shouldUseFP16Storage && config.vulkan.shouldUseFP16Compute &&
      nchwParams.isValid() && cSize % nchwParams.KWG == 0;
    if(ffnSize > numeric_limits<int>::max() / 2) {
      if(context.logger != nullptr)
        context.logger->write("Skipping Vulkan transformerDualGemmSwiGLU tuning: packed FFN size overflow");
      return false;
    }
    const int logicalPackedOCSize = 2 * ffnSize;
    int packedOCSize = logicalPackedOCSize;
    if(legacyNCHW) {
      const int64_t roundedPackedOCSize =
        ((static_cast<int64_t>(logicalPackedOCSize) + nchwParams.NWG - 1) / nchwParams.NWG) * nchwParams.NWG;
      if(roundedPackedOCSize > numeric_limits<int>::max()) {
        if(context.logger != nullptr)
          context.logger->write("Skipping Vulkan transformerDualGemmSwiGLU tuning: packed FFN size overflow");
        return false;
      }
      packedOCSize = static_cast<int>(roundedPackedOCSize);
      legacyNCHW = packedOCSize % nchwParams.NWG == 0;
    }
    if(legacyNHWC) {
      packedOCSize = vk_helper::roundUpToMultipleInt(packedOCSize, config.hgemmCooperativeMatrixNHWC.NWG);
      legacyNCHW = false;
    }
    if(packedOCSize < logicalPackedOCSize) {
      if(context.logger != nullptr)
        context.logger->write("Skipping Vulkan transformerDualGemmSwiGLU tuning: packed FFN size overflow");
      return false;
    }

    vk_shader::ComputePipelines legacyPipelines(
      context.device->device,
      context.device->info,
      nullptr,
      false
    );
    VkResult result = VK_SUCCESS;
    Pipeline* legacyGemmPipeline = nullptr;
    if(legacyNHWC) {
      const auto& p = config.transformerHgemmNHWC;
      const auto& gemmParams = p.isValid() && productionHW % p.MWG == 0 &&
        cSize % p.KWG == 0 && packedOCSize % p.NWG == 0
        ? p : config.hgemmCooperativeMatrixNHWC;
      if(productionHW % gemmParams.MWG != 0 || cSize % gemmParams.KWG != 0 || packedOCSize % gemmParams.NWG != 0)
        return false;
      result = legacyPipelines.createHgemmCooperativeMatrixNHWC(
        legacyPipelines.hgemmCooperativeMatrixNHWC,
        gemmParams
      );
      legacyGemmPipeline = &legacyPipelines.hgemmCooperativeMatrixNHWC;
      if(context.logger != nullptr)
        context.logger->write("Transformer FFN comparator: production NHWC packed GEMM + NHWC SwiGLU, B=" +
          to_string(context.batchSize) + ", physical S=" + to_string(productionHW) + ", K=" +
          to_string(cSize) + ", N=" + to_string(packedOCSize));
    }
    else if(legacyNCHW) {
      result = legacyPipelines.createHgemmCooperativeMatrixNCHW(
        legacyPipelines.hgemmCooperativeMatrixNCHW, nchwParams
      );
      legacyGemmPipeline = &legacyPipelines.hgemmCooperativeMatrixNCHW;
    }
    else {
      result = legacyPipelines.createXgemmStridedBatched(
        legacyPipelines.xgemmStridedBatchedFp32, config.xgemmDirect, config.vulkan
      );
      legacyGemmPipeline = &legacyPipelines.xgemmStridedBatchedFp32;
    }
    if(result != VK_SUCCESS) {
      if(context.logger != nullptr)
        context.logger->write(
          "Vulkan transformerDualGemmSwiGLU legacy GEMM pipeline creation failed: " +
          vk_helper::vkErrorToString(result) + ", selected=false"
        );
      return false;
    }
    result = legacyPipelines.createTransformerSwiGLU(
      legacyPipelines.transformerSwiGLU, config.pointwise, config.vulkan
    );
    if(result != VK_SUCCESS) {
      if(context.logger != nullptr)
        context.logger->write(
          "Vulkan transformerDualGemmSwiGLU legacy SwiGLU pipeline creation failed: " +
          vk_helper::vkErrorToString(result) + ", selected=false"
        );
      return false;
    }

    vector<VulkanTuneParams> candidates = TransformerDualGemmSwiGLUTuner::candidates(
      config, context.full, context
    );
    if(candidates.empty()) {
      if(context.logger != nullptr)
        context.logger->write(
          "Skipping Vulkan transformerDualGemmSwiGLU tuning: no device-feasible candidates match model dimensions, selected=false"
        );
      return false;
    }

    const int logicalHW = std::max(1, context.nnXLen * context.nnYLen);
    const int legacySpatialAlignment = legacyNHWC ? productionSpatialAlignment :
      (legacyNCHW ? nchwParams.getRequiredSpatialAlignment() : 1);
    const int maxCandidates = static_cast<int>(candidates.size());
    double bestFusedCallsPerSecond = 0.0;
    double conservativeLegacyCallsPerSecond = 0.0;
    double bestErrorProp = numeric_limits<double>::quiet_NaN();
    TransformerDualGemmSwiGLUTuneParams bestCandidate;
    vk_shader::ComputePipelines candidatePipelines(
      context.device->device,
      context.device->info,
      nullptr,
      false
    );
    if((result = candidatePipelines.createNchwToNhwc(candidatePipelines.nchwToNhwc)) != VK_SUCCESS ||
       (result = candidatePipelines.createNhwcToNchw(candidatePipelines.nhwcToNchw)) != VK_SUCCESS) {
      if(context.logger != nullptr)
        context.logger->write(
          "Skipping Vulkan transformerDualGemmSwiGLU tuning: could not create NHWC conversion pipelines, " +
          vk_helper::vkErrorToString(result)
        );
      return false;
    }
    vector<TuningConfigMeasurement> passing;
    size_t candidateIndex = 0;
    for(const VulkanTuneParams& candidateConfig: candidates) {
      candidateIndex++;
      const TransformerDualGemmSwiGLUTuneParams& params = candidateConfig.transformerDualGemmSwiGLU;
      if((legacyNHWC && productionHW % params.getRequiredSpatialAlignment() != 0) ||
         (context.onlyMissing && vk_helper::roundUpToMultipleInt(
           context.nnXLen * context.nnYLen,
           std::lcm(baseSpatialAlignment, params.getRequiredSpatialAlignment())) != productionHW)) {
        logTuningProgressIfNeeded(context, candidateIndex, maxCandidates);
        continue;
      }
      vector<const Pipeline*> candidateTargets;
      result = TransformerDualGemmSwiGLUTuner::create(
        context, candidateConfig, candidatePipelines, candidateTargets
      );
      context.device->recordDeviceLoss(result, "fused FFN pipeline creation");
      requireTuningDevice(context, "transformerDualGemmSwiGLU");
      if(result != VK_SUCCESS) {
        if(!context.printOnlyOnImprovement && context.logger != nullptr)
          context.logger->write(
            "Vulkan transformerDualGemmSwiGLU candidate " + to_string(candidateIndex) + "/" +
            to_string(maxCandidates) + " pipeline rejected: " + vk_helper::vkErrorToString(result)
          );
        logTuningProgressIfNeeded(context, candidateIndex, maxCandidates);
        continue;
      }

      const int combinedAlignment = std::lcm(legacySpatialAlignment, params.getRequiredSpatialAlignment());
      const int hwSize = legacyNHWC ? productionHW : vk_helper::roundUpToMultipleInt(
        logicalHW,
        combinedAlignment
      );
      double fusedCallsPerSecond = 0.0;
      double legacyCallsPerSecond = 0.0;
      double errorProp = numeric_limits<double>::quiet_NaN();
      vector<float> readback;
      string error;
      const bool measured = context.timer != nullptr && timer.measureTransformerDualGemmSwiGLU(
        candidateConfig, legacyConfig, context,
        &candidatePipelines.transformerDualGemmSwiGLU,
        &candidatePipelines.nchwToNhwc,
        &candidatePipelines.nhwcToNchw,
        legacyGemmPipeline, &legacyPipelines.transformerSwiGLU,
        legacyNCHW, hwSize, packedOCSize,
        fusedCallsPerSecond, legacyCallsPerSecond, errorProp, readback, error
      );
      for(const Pipeline* pipeline: candidateTargets)
        candidatePipelines.destroyPipeline(*const_cast<Pipeline*>(pipeline));
      requireTuningDevice(context, "transformerDualGemmSwiGLU", error);
      if(!measured || !isfinite(fusedCallsPerSecond) || fusedCallsPerSecond <= 0.0 ||
         !isfinite(legacyCallsPerSecond) || legacyCallsPerSecond <= 0.0) {
        if(!context.printOnlyOnImprovement && context.logger != nullptr)
          context.logger->write(
            "Vulkan transformerDualGemmSwiGLU candidate " + to_string(candidateIndex) + "/" +
            to_string(maxCandidates) + " measurement failed: " +
            (error.empty() ? "invalid calls/sec" : error)
          );
        logTuningProgressIfNeeded(context, candidateIndex, maxCandidates);
        continue;
      }
      conservativeLegacyCallsPerSecond = std::max(conservativeLegacyCallsPerSecond, legacyCallsPerSecond);
      const bool numericallyValid = isfinite(errorProp) &&
        errorProp <= VulkanTuner::TRANSFORMER_DUAL_GEMM_ERROR_TOLERANCE;
      if(numericallyValid)
        passing.push_back({candidateConfig, fusedCallsPerSecond, fusedCallsPerSecond / legacyCallsPerSecond});
      const bool isBest = numericallyValid && fusedCallsPerSecond > bestFusedCallsPerSecond;
      if(isBest) {
        bestFusedCallsPerSecond = fusedCallsPerSecond;
        bestErrorProp = errorProp;
        bestCandidate = params;
      }
      const bool loggedResult = isBest && logTuningResult(
        context, candidateIndex, maxCandidates, candidateConfig, TransformerDualGemmSwiGLUTuner::name(), fusedCallsPerSecond
      );
      logTuningProgressIfNeeded(context, candidateIndex, maxCandidates, loggedResult);
    }

    sort(passing.begin(), passing.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    vector<TuningConfigMeasurement> finalists;
    for(int stages: {1,2}) {
      int count = 0;
      for(const auto& value: passing)
        if(value.config.transformerDualGemmSwiGLU.STAGES == stages && count++ < 2)
          finalists.push_back(value);
    }
    bool useDualGemm = false;
    double acceptedRate = 0.0;
    for(size_t finalist = 0; finalist < finalists.size(); finalist++) {
      const auto& candidate = finalists[finalist].config;
      const auto& q = candidate.transformerDualGemmSwiGLU;
      Pipeline& pipeline = candidatePipelines.transformerDualGemmSwiGLU;
      result = candidatePipelines.createTransformerDualGemmSwiGLU(pipeline, q);
      context.device->recordDeviceLoss(result, "fused FFN finalist pipeline creation");
      requireTuningDevice(context, "transformerDualGemmSwiGLU finalist");
      if(result != VK_SUCCESS)
        continue;
      vector<double> fusedSeconds, legacySeconds;
      double maxError = 0.0;
      const int alignment = std::lcm(legacySpatialAlignment, q.getRequiredSpatialAlignment());
      const int hwSize = legacyNHWC ? productionHW : vk_helper::roundUpToMultipleInt(logicalHW, alignment);
      for(int round = 0; round < 6; round++) {
        double fusedRate = 0.0, legacyRate = 0.0, error = 0.0;
        vector<float> output;
        string message;
        const bool measured = timer.measureTransformerDualGemmSwiGLU(candidate, legacyConfig, context,
          &pipeline, &candidatePipelines.nchwToNhwc, &candidatePipelines.nhwcToNchw,
          legacyGemmPipeline, &legacyPipelines.transformerSwiGLU, legacyNCHW, hwSize, packedOCSize,
          fusedRate, legacyRate, error, output, message, 96, round % 2 != 0);
        requireTuningDevice(context, "transformerDualGemmSwiGLU finalist", message);
        if(!measured || !isfinite(error) || error > VulkanTuner::TRANSFORMER_DUAL_GEMM_ERROR_TOLERANCE)
          break;
        fusedSeconds.push_back(1.0 / fusedRate);
        legacySeconds.push_back(1.0 / legacyRate);
        maxError = std::max(maxError, error);
      }
      candidatePipelines.destroyPipeline(pipeline);
      if(fusedSeconds.size() != 6)
        continue;
      const bool consistent = VulkanTuner::isTransformerGemmUpgrade(legacySeconds, fusedSeconds);
      sort(fusedSeconds.begin(), fusedSeconds.end());
      sort(legacySeconds.begin(), legacySeconds.end());
      const double fusedRate = 2.0 / (fusedSeconds[2] + fusedSeconds[3]);
      const double legacyRate = 2.0 / (legacySeconds[2] + legacySeconds[3]);
      const bool faster = consistent && VulkanTuner::isFastEnough(fusedRate, legacyRate,
        VulkanTuner::TRANSFORMER_DUAL_GEMM_MIN_THROUGHPUT_RATIO);
      if((context.onlyMissing || faster) && fusedRate > acceptedRate) {
        useDualGemm = true;
        acceptedRate = fusedRate;
        bestCandidate = q;
        bestFusedCallsPerSecond = fusedRate;
        conservativeLegacyCallsPerSecond = legacyRate;
        bestErrorProp = maxError;
        logTuningResult(
          context,
          finalist + 1,
          finalists.size(),
          candidate,
          TransformerDualGemmSwiGLUTuner::name(),
          fusedRate
        );
      }
    }
    if(useDualGemm) {
      config.transformerDualGemmSwiGLU = bestCandidate;
      config.vulkan.shouldUseTransformerDualGemmSwiGLU = true;
      recordTuning("transformerDualGemmSwiGLU", context);
    }
    if(context.logger != nullptr) {
      context.logger->write(
        "Vulkan transformerDualGemmSwiGLU comparison: fused=" +
        Global::strprintf("%.6g", bestFusedCallsPerSecond) + " calls/s, legacy packed-GEMM+SwiGLU=" +
        Global::strprintf("%.6g", conservativeLegacyCallsPerSecond) + " calls/s, error=" +
        Global::strprintf("%.6g", bestErrorProp) + ", required_ratio=" +
        Global::strprintf("%.2f", VulkanTuner::TRANSFORMER_DUAL_GEMM_MIN_THROUGHPUT_RATIO) +
        ", selected=" + (useDualGemm ? "true" : "false")
      );
    }
    return useDualGemm;
  }

  template<typename Tuner>
  bool completeTuner(
    const TuningContext& context,
    VulkanTuneParams& config,
    const function<void()>& checkpoint
  ) {
    if(tuningComplete(Tuner::name(), context))
      return false;
    runTuner<Tuner>(context, config);
    if(!tuningComplete(Tuner::name(), context))
      throw StringError("Vulkan tuning failed for required shader " + Tuner::name());
    if(Tuner::name() == "transformerScaleDotProductCooperativeMatrixNHWC")
      validateTransformerAttentionMasks(context, config);
    if(checkpoint)
      checkpoint();
    return true;
  }

  bool completeRequiredTuners(
    const TuningContext& originalContext,
    VulkanTuneParams& config,
    const function<void()>& checkpoint = {}
  ) {
    TuningContext context = originalContext;
    context.onlyMissing = true;
    context.selectedGemmConfig = &config;
    const VulkanParams requestedMode = config.vulkan;
    bool changed = false;
    const auto complete = [&](auto tuner) {
      changed |= completeTuner<decltype(tuner)>(context, config, checkpoint);
    };
    const auto hasCases = [&](const string& name) {
      return !VulkanTuner::getGemmTuningCases(name, context.modelInfo, config).empty();
    };
    const auto completeXgemm = [&]() {
      if(hasCases("xgemm")) {
        if(config.vulkan.shouldUseFP16Compute)
          complete(Xgemm16Tuner());
        else
          complete(XgemmTuner());
      }
    };
    if(config.vulkan.shouldUseCooperativeMatrix) {
      if(!usesGenericNHWC(config) && config.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
         hasCases("hgemmCooperativeMatrixNCHW"))
        complete(HgemmCooperativeMatrixNCHWTunerImpl());
      if(usesGenericNHWC(config)) {
        if(hasCases("hgemmCooperativeMatrixNHWC"))
          complete(HgemmCooperativeMatrixNHWCTunerImpl());
      }
      else if(hasCases("hgemmCooperativeMatrix"))
        complete(HgemmCooperativeMatrixTunerImpl());
    }
    else
      completeXgemm();
    if(hasCases("xgemmDirect"))
      complete(XgemmDirectTuner());
    if(hasCases("transformerXgemm") && !tuningComplete("transformerXgemm", context)) {
      tuneTransformerXgemm(context, config);
      if(!tuningComplete("transformerXgemm", context))
        throw StringError("Vulkan tuning failed for required shader transformerXgemm");
      changed = true;
      if(checkpoint)
        checkpoint();
    }
    if(config.vulkan.shouldUseCooperativeMatrix)
      completeXgemm();
    if(config.vulkan.shouldUseTransformerDualGemmSwiGLU &&
       !tuningComplete("transformerDualGemmSwiGLU", context)) {
      if(!tuneTransformerDualGemmSwiGLU(context, config) ||
         !tuningComplete("transformerDualGemmSwiGLU", context))
        throw StringError("Vulkan tuning failed for selected Transformer fused FFN");
      changed = true;
      if(checkpoint)
        checkpoint();
    }
    const auto completeCustom = [&](const string& name, auto tuneFunction) {
      if(tuningComplete(name, context))
        return;
      tuneFunction(context, config);
      if(tuningComplete(name, context)) {
        changed = true;
        if(checkpoint)
          checkpoint();
      }
    };
    completeCustom("transformerHgemmNHWC", tuneTransformerHgemmNHWC);
    const auto plan = VulkanTuner::getNonGemmTuningPlan(context.modelInfo, config);
    if(context.modelInfo.maxConvChannels3x3 > 0) {
      if(usesGenericNHWC(config)) {
        complete(Conv3x3InputNHWCTuner());
        complete(Conv3x3OutputNHWCTuner());
      }
      else {
        complete(Conv3x3InputTuner());
        complete(Conv3x3OutputTuner());
      }
    }
    if(context.modelInfo.hasConv5x5) {
      if(usesGenericNHWC(config)) {
        complete(Conv5x5InputNHWCTuner());
        complete(Conv5x5OutputNHWCTuner());
      }
      else {
        complete(Conv5x5InputTuner());
        complete(Conv5x5OutputTuner());
      }
    }
    complete(GPoolTuner());
    if(plan.attentionNHWC) {
      if(config.vulkan.shouldUseTransformerCooperativeMatrixNHWC)
        complete(TransformerScaleDotProductCooperativeMatrixTuner<true>());
      else
        throw StringError("Vulkan NHWC Attention requires its cooperative-matrix path");
    }
    else if(plan.attentionNCHW) {
      if(config.vulkan.shouldUseTransformerCooperativeMatrixNCHW)
        complete(TransformerScaleDotProductCooperativeMatrixTuner<false>());
      else
        complete(TransformerScaleDotProductTuner());
    }
    if(plan.rmsNorm)
      complete(TransformerRMSNormTuner());
    complete(PointwiseTuner());
    complete(AddChannelBiasesTuner());
    if(plan.spatialRMSNorm)
      complete(SpatialRMSNormTuner());
    completeCustom("transformerPreNorm", tuneTransformerPreNorm);
    config.vulkan = requestedMode;
    return changed;
  }

  void applyExecutionMode(
    VulkanTuneParams& config,
    const VulkanTuner::ModelInfoForTuning& model,
    enabled_t useFP16Mode,
    enabled_t useNHWCMode = enabled_t::Auto
  ) {
    auto& mode = config.vulkan;
    if(useFP16Mode != enabled_t::Auto) {
      mode.shouldUseFP16Storage = useFP16Mode == enabled_t::True && mode.canUseFP16Storage && mode.canUseFP16Compute;
      mode.shouldUseFP16Compute = useFP16Mode == enabled_t::True && mode.canUseFP16Compute;
    }
    mode.shouldUseCooperativeMatrix &= mode.canUseCooperativeMatrix &&
      mode.shouldUseFP16Storage && mode.shouldUseFP16Compute;
    if(useNHWCMode == enabled_t::True)
      mode.shouldUseNHWC = true;
    mode.shouldUseNHWC &= mode.shouldUseCooperativeMatrix &&
      model.transformerHeadDim > 0 && model.transformerVHeadDim > 0;
    mode.shouldUseHgemmCooperativeMatrixNCHW &= mode.canUseCooperativeMatrix &&
      mode.shouldUseFP16Storage && mode.shouldUseFP16Compute;
    mode.shouldUseTransformerDualGemmSwiGLU &= mode.shouldUseCooperativeMatrix && mode.shouldUseNHWC;
    mode.shouldUseTransformerCooperativeMatrixNCHW &= mode.canUseCooperativeMatrix && mode.shouldUseFP16Storage;
    mode.shouldUseTransformerCooperativeMatrixNHWC &= mode.canUseCooperativeMatrix &&
      mode.shouldUseFP16Storage && mode.shouldUseFP16Compute;
  }
}

bool VulkanTuner::HgemmCooperativeMatrixNCHWTuner::selectCooperativeMatrixProperties(
  const VulkanDevice* device,
  HGemmCooperativeMatrixNCHWTuneParams& params
) {
  return selectHgemmCooperativeMatrixProperties(device, params);
}

bool VulkanTuner::HgemmCooperativeMatrixTuner::selectCooperativeMatrixProperties(
  const VulkanDevice* device,
  HGemmCooperativeMatrixTuneParams& params
) {
  return selectHgemmCooperativeMatrixProperties(device, params);
}

void VulkanTuner::tune(
  const VulkanDevice* device,
  int batchSize,
  int nnXLen,
  int nnYLen,
  const ModelInfoForTuning& modelInfo,
  bool full,
  Logger* logger,
  VulkanTuneParams& tunedConfig,
  bool printOnlyOnImprovement,
  bool attentionOnly,
  bool transformerGemmOnly,
  bool transformerFFNOnly,
  bool transformerResidualOnly,
  bool transformerPreNormOnly
) {
  const auto hostStart = std::chrono::steady_clock::now();
  if(device == nullptr)
    throw StringError("VulkanTuner::tune: device is null");
  if(!tunedConfig.isValid())
    tunedConfig = VulkanTuneParams();
  if(const char* probeCase = getAttentionProbeCase(device)) {
    runAttentionProbe(probeCase, device, batchSize, nnXLen, nnYLen, modelInfo, logger, tunedConfig);
    return;
  }
  if(attentionOnly && (modelInfo.transformerHeadDim <= 0 || modelInfo.transformerVHeadDim <= 0))
    throw StringError("Vulkan attention-only tuning requires a transformer model");

  const VulkanParams hardwareParams = VulkanTuner::getHardwareParams(device->info);
  tunedConfig.vulkan.canUseFP16Storage = hardwareParams.canUseFP16Storage;
  tunedConfig.vulkan.canUseFP16Compute = hardwareParams.canUseFP16Compute;
  tunedConfig.vulkan.canUseCooperativeMatrix = hardwareParams.canUseCooperativeMatrix;
  tunedConfig.vulkan.canUseSubgroup = hardwareParams.canUseSubgroup;

  auto records = tunedConfig.tuningRecords;
  initializeTuningRecords(records);
  const auto recordGuard = makeScopeGuard([&]() { tunedConfig.tuningRecords = std::move(records); });
  VulkanTimestampTimer timer(device);
  VulkanDummyThread dummyThread(device, logger);
  // Projection tuning represents an otherwise idle, single-server inference GPU.
  if(!attentionOnly && !transformerGemmOnly && !transformerFFNOnly && !transformerPreNormOnly && !transformerResidualOnly)
    dummyThread.start();
  TuningContext context{
    device, batchSize, nnXLen, nnYLen, modelInfo, full,
    getCooperativeMatrixTuneShapes(device), logger, &timer, printOnlyOnImprovement
  };
  context.tuningRecords = &records;
  if(logger != nullptr) {
    logger->write(
      "Vulkan tuning capabilities: fp16Storage=" + string(tunedConfig.vulkan.canUseFP16Storage ? "true" : "false") +
      ", fp16Compute=" + string(tunedConfig.vulkan.canUseFP16Compute ? "true" : "false") +
      ", subgroup=" + string(tunedConfig.vulkan.canUseSubgroup ? "true" : "false") +
      ", cooperativeMatrix=" + string(tunedConfig.vulkan.canUseCooperativeMatrix ? "true" : "false")
    );
  }
  if(attentionOnly) {
    context.recheckAttentionFinalists = true;
    tunedConfig.vulkan.shouldUseFP16Storage = tunedConfig.vulkan.canUseFP16Storage;
    tunedConfig.vulkan.shouldUseFP16Compute = tunedConfig.vulkan.canUseFP16Compute;
    // NHWC requires cooperative attention; its CPU reference is checked by that tuner.
    tuneTransformerCooperativeMatrix<true>(context, tunedConfig, numeric_limits<double>::quiet_NaN());
    dummyThread.stopAndJoin();
    return;
  }
  if(transformerGemmOnly) {
    if(modelInfo.transformerProjections.empty() || !usesGenericNHWC(tunedConfig))
      throw StringError("Transformer GEMM-only tuning requires a production NHWC Transformer configuration");
    tuneTransformerHgemmNHWC(context, tunedConfig);
    dummyThread.stopAndJoin();
    return;
  }
  if(transformerFFNOnly) {
    tuneTransformerDualGemmSwiGLU(context, tunedConfig);
    dummyThread.stopAndJoin();
    return;
  }
  if(transformerPreNormOnly) {
    tuneTransformerPreNorm(context, tunedConfig);
    dummyThread.stopAndJoin();
    return;
  }
  if(transformerResidualOnly) {
    if(!usesGenericNHWC(tunedConfig) || !tunedConfig.vulkan.shouldUseFP16Storage ||
       !tunedConfig.vulkan.shouldUseFP16Compute || modelInfo.transformerProjections.empty())
      throw StringError(
        "Residual RMSNorm validation requires a production FP16 NHWC Transformer configuration"
      );
    const auto& layout = tunedConfig.hgemmCooperativeMatrixNHWC;
    const int channelAlignment = std::lcm(4, std::lcm(layout.NWG, layout.KWG));
    if(VulkanTuner::getTransformerFFNInputChannelsForTuning(modelInfo) % channelAlignment != 0)
      throw StringError("Residual RMSNorm requires Transformer channels without NHWC channel padding");
    VulkanTuneParams validation = tunedConfig;
    validation.rmsNorm = tunedConfig.transformerPreNorm;
    ComputePipelines pipelines(
      device->device,
      device->info,
      logger
    );
    VkResult result = pipelines.createTransformerRMSNorm(
      pipelines.transformerResidualRmsNormNHWC,
      tunedConfig.transformerPreNorm,
      tunedConfig.vulkan,
      true,
      true
    );
    if(result == VK_SUCCESS)
      result = pipelines.createTransformerRMSNorm(
        pipelines.transformerRmsNormNHWC,
        tunedConfig.transformerPreNorm,
        tunedConfig.vulkan,
        true
      );
    if(result == VK_SUCCESS)
      result = pipelines.createAddPointWise(
        pipelines.addPointWise,
        tunedConfig.pointwise,
        tunedConfig.vulkan
      );
    if(result != VK_SUCCESS)
      throw StringError("Could not create residual RMSNorm validation pipelines");
    int alignment = tunedConfig.hgemmCooperativeMatrixNHWC.MWG;
    if(tunedConfig.vulkan.shouldUseHgemmCooperativeMatrixNCHW)
      alignment = std::lcm(alignment, tunedConfig.hgemmCooperativeMatrixNCHW.getRequiredSpatialAlignment());
    if(tunedConfig.vulkan.shouldUseTransformerDualGemmSwiGLU)
      alignment = std::lcm(alignment, tunedConfig.transformerDualGemmSwiGLU.getRequiredSpatialAlignment());
    const int physicalS = vk_helper::roundUpToMultipleInt(nnXLen * nnYLen, alignment);
    string error;
    for(int batch: {1, batchSize})
      for(int spatial: {physicalS, std::max(1, physicalS - 1)})
        for(int mask = 0; mask < 4; mask++)
          if(!timer.measureTransformerResidualRMSNorm(validation, context,
            pipelines.transformerResidualRmsNormNHWC, pipelines.transformerRmsNormNHWC, pipelines.addPointWise,
            batch, spatial, mask, error))
            throw StringError("Residual RMSNorm validation failed: " + error);
    dummyThread.stopAndJoin();
    return;
  }
  const bool canUseHgemmCooperativeMatrix =
    tunedConfig.vulkan.canUseCooperativeMatrix &&
    VulkanTuner::HgemmCooperativeMatrixTuner::selectCooperativeMatrixProperties(
      device, tunedConfig.hgemmCooperativeMatrix
    );
  const bool canUseHgemmCooperativeMatrixNCHW =
    tunedConfig.vulkan.canUseCooperativeMatrix &&
    VulkanTuner::HgemmCooperativeMatrixNCHWTuner::selectCooperativeMatrixProperties(
      device, tunedConfig.hgemmCooperativeMatrixNCHW
    );
  tunedConfig.xgemmDirect.PADA = 1;
  tunedConfig.xgemmDirect.PADB = 1;
  tunedConfig.vulkan.shouldUseFP16Storage = false;
  tunedConfig.vulkan.shouldUseFP16Compute = false;
  tunedConfig.vulkan.shouldUseCooperativeMatrix = false;
  tunedConfig.vulkan.shouldUseNHWC = false;
  tunedConfig.vulkan.shouldUseHgemmCooperativeMatrixNCHW = false;
  tunedConfig.vulkan.shouldUseTransformerDualGemmSwiGLU = false;
  tunedConfig.vulkan.shouldUseTransformerCooperativeMatrixNCHW = false;
  tunedConfig.vulkan.shouldUseTransformerCooperativeMatrixNHWC = false;
  tunedConfig.vulkan.shouldUseSubgroup = false;
  if(canUseHgemmCooperativeMatrix) {
    // Unused CM pipelines are also built by createPipelines. Seed their
    // fragment types from a supported property even when only one layout is tuned.
    const auto& supported = tunedConfig.hgemmCooperativeMatrix;
    auto& nhwc = tunedConfig.hgemmCooperativeMatrixNHWC;
    nhwc.MWARP = supported.MWARP;
    nhwc.NWARP = supported.NWARP;
    nhwc.KDIM = supported.KDIM;
    nhwc.subgroupSize = supported.subgroupSize;
    nhwc.accType = supported.accType;
    nhwc.MWG = nhwc.MWAVE = nhwc.MWARP;
    nhwc.NWG = nhwc.NWAVE = nhwc.NWARP;
    nhwc.KWG = nhwc.KDIM;
    nhwc.SA = nhwc.SB = 1;
    nhwc.DB = 0;
    nhwc.VWK = nhwc.VWN = 1;
    tunedConfig.transformerHgemmNHWC = nhwc;
  }

  context.selectedGemmConfig = &tunedConfig;
  const bool usingCooperativeMatrix = tuneCooperativeMatrices(
    context,
    tunedConfig,
    canUseHgemmCooperativeMatrix,
    canUseHgemmCooperativeMatrixNCHW
  );
  if(usingCooperativeMatrix)
    runTuner<XgemmDirectTuner>(context, tunedConfig);
  if(!getTunerGemmTuneCases("xgemm", context).empty()) {
    const double xgemmCallsPerSecond = runTuner<XgemmTuner>(context, tunedConfig);
    tunedConfig.xgemm16 = tunedConfig.xgemm;
    tuneXgemm16(context, tunedConfig, xgemmCallsPerSecond);
    if(!tunedConfig.vulkan.shouldUseFP16Compute)
      tuneXgemmStorage(context, tunedConfig, xgemmCallsPerSecond);
  }
  else if(logger != nullptr)
    logger->write("Skipping xgemm/xgemm16 tuning: all model Winograd GEMMs use cooperative matrix");

  // Scalar GEMM tuning selects the precision used by the spatial direct GEMM.
  if(!usingCooperativeMatrix)
    runTuner<XgemmDirectTuner>(context, tunedConfig);
  tunedConfig.transformerXgemm = tunedConfig.xgemmDirect;

  runNonGemmTuners(context, tunedConfig);
  dummyThread.stopAndJoin();
  requireTuningDevice(context, "automatic tuning");
  tuneTransformerHgemmNHWC(context, tunedConfig);
  tuneTransformerDualGemmSwiGLU(context, tunedConfig);
  tunedConfig.transformerPreNorm = tunedConfig.rmsNorm;
  tuneTransformerPreNorm(context, tunedConfig);
  tunedConfig.vulkan.shouldUseNHWC =
    modelInfo.transformerHeadDim > 0 &&
    modelInfo.transformerVHeadDim > 0 &&
    tunedConfig.vulkan.shouldUseCooperativeMatrix;
  tuneTransformerXgemm(context, tunedConfig);

  if(logger != nullptr) {
    logger->write(
      "Vulkan tuning final selection: "
      "shouldUseFP16Storage=" + string(tunedConfig.vulkan.shouldUseFP16Storage ? "1" : "0") +
      ", shouldUseFP16Compute=" + string(tunedConfig.vulkan.shouldUseFP16Compute ? "1" : "0") +
      ", shouldUseCooperativeMatrix=" + string(tunedConfig.vulkan.shouldUseCooperativeMatrix ? "1" : "0") +
      ", shouldUseNHWC=" + string(tunedConfig.vulkan.shouldUseNHWC ? "1" : "0") +
      ", shouldUseHgemmCooperativeMatrixNCHW=" +
        string(tunedConfig.vulkan.shouldUseHgemmCooperativeMatrixNCHW ? "1" : "0") +
      ", shouldUseTransformerDualGemmSwiGLU=" +
        string(tunedConfig.vulkan.shouldUseTransformerDualGemmSwiGLU ? "1" : "0") +
      ", shouldUseTransformerCooperativeMatrixNCHW=" +
        string(tunedConfig.vulkan.shouldUseTransformerCooperativeMatrixNCHW ? "1" : "0") +
      ", shouldUseTransformerCooperativeMatrixNHWC=" +
        string(tunedConfig.vulkan.shouldUseTransformerCooperativeMatrixNHWC ? "1" : "0")
    );
  }
  dummyThread.stopAndJoin();
  if(logger != nullptr) {
    const double hostSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - hostStart).count();
    logger->write("Vulkan tuning total host time: " + Global::doubleToString(hostSeconds) + " sec");
  }
  completeRequiredTuners(context, tunedConfig);
}

void VulkanTuner::testTuningCache() {
  {
    VkPhysicalDeviceLimits limits{};
    limits.maxComputeWorkGroupSize[0] = 128;
    limits.maxComputeWorkGroupSize[1] = 1;
    limits.maxComputeWorkGroupSize[2] = 1;
    limits.maxComputeWorkGroupInvocations = 128;
    limits.maxComputeSharedMemorySize = 32768;

    TransformerTuneParams attention;
    attention.ATTN_BLOCK_KV = 128;
    testAssert(!attention.isSupported(limits, 32, 32));
    limits.maxComputeSharedMemorySize = 33280;
    testAssert(attention.isSupported(limits, 32, 32));
    attention.ATTN_BLOCK_Q = 256;
    testAssert(!attention.isSupported(limits, 32, 32));
    attention.USE_TILED_ATTN = 0;
    testAssert(attention.isSupported(limits, 32, 32));
    testAssert(attention.getDispatchGroupCountX(361) == 12);
  }
  {
    VulkanDevice device;
    testAssert(device.recordDeviceLoss(VK_ERROR_OUT_OF_DEVICE_MEMORY, "allocation") == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    testAssert(device.deviceLossStage.load() == nullptr);
    device.recordDeviceLoss(VK_ERROR_DEVICE_LOST, "first failure");
    device.recordDeviceLoss(VK_ERROR_DEVICE_LOST, "second failure");
    testAssert(string(device.deviceLossStage.load()) == "first failure");
    testAssert(vk_helper::submitCommandBuffers(&device, {}) == VK_ERROR_DEVICE_LOST);
  }

  ModelDesc desc;
  desc.trunk.trunkNumChannels = desc.trunk.initialMatMul.outChannels = 80;
  desc.trunk.initialConv.convXSize = desc.trunk.initialConv.convYSize = 5;
  desc.trunk.initialConv.inChannels = 22;
  desc.trunk.initialConv.outChannels = 80;
  desc.policyHead.p1Conv.outChannels = 24;
  desc.policyHead.g1Conv.outChannels = 16;
  auto* nested = new NestedBottleneckResidualBlockDesc();
  desc.trunk.blocks.emplace_back(NESTED_BOTTLENECK_BLOCK_KIND, make_unique_void(nested));
  nested->postConv.outChannels = 80;
  auto* ffn = new TransformerFFNDesc();
  nested->blocks.emplace_back(TRANSFORMER_FFN_BLOCK_KIND, make_unique_void(ffn));
  ffn->numChannels = ffn->preLN.numChannels = 40;
  ffn->ffnChannels = 64;
  ffn->useSwiGLU = true;
  const auto workloads = ModelInfoForTuning::ofDesc(desc);
  const auto residuals = getModelChannelCases("pointwise", workloads);
  testAssert((residuals == vector<int>{80, 40}));
  testAssert((getModelChannelCases("addChannelBiases", workloads) == vector<int>{80, 24}));
  testAssert((getModelChannelCases("gPool", workloads) == vector<int>{16}));
  testAssert((getModelChannelCases("transformerRMSNorm", workloads) == vector<int>{40}));
  testAssert((getModelChannelCases("conv5x5InputTransform", workloads) == vector<int>{22}));
  testAssert((getModelChannelCases("conv5x5OutputTransform", workloads) == vector<int>{80}));
  testAssert(getModelChannelCases("conv3x3InputTransform", workloads).empty());
  testAssert(workloadCaseForSlot(residuals, 0, 80) == 80);
  testAssert(workloadCaseForSlot(residuals, 1, 80) == 80);
  testAssert(workloadCaseForSlot(residuals, 2, 80) == 40);
  testAssert(workloadCaseForSlot(workloads.swiGLUChannels, 1, 80) == 64);

  ModelInfoForTuning model;
  model.modelVersion = 15;
  model.numInputChannels = 22;
  model.trunkNumChannels = model.midNumChannels = model.maxConvChannels3x3 = 64;
  model.gemmLayers = {
    {64, 64, 2.0, false, GemmTuneCase::Kind::Winograd3x3},
    {64, 32, 1.0, false, GemmTuneCase::Kind::Dense}
  };
  VulkanTuneParams config;
  testAssert(config.isValid());
  testAssert(getGemmTuningCases("xgemmDirect", model, config).empty());
  ModelInfoForTuning convAndDense = model;
  convAndDense.gemmLayers.push_back({64, 96, 3.0, false, GemmTuneCase::Kind::Spatial});
  const auto directCases = getGemmTuningCases("xgemmDirect", convAndDense, config);
  testAssert(directCases.size() == 1 && directCases[0].kind == GemmTuneCase::Kind::Spatial &&
    directCases[0].outChannels == 96 && directCases[0].weight == 3.0);
  initializeTuningRecords(config.tuningRecords);
  testAssert(config.tuningRecords.size() == tuningNames().size());
  TuningContext context{nullptr, 4, 19, 19, model, false, {}, nullptr, nullptr, true};
  context.selectedGemmConfig = &config;
  context.tuningRecords = &config.tuningRecords;
  const auto hgemmPlan = makeMeasurementPlan("hgemmCooperativeMatrix", context);
  testAssert(hgemmPlan.errorTolerance == 0.01);
  testAssert(computeCooperativeMatrixTuningScore(100.0, 0.01, hgemmPlan.errorTolerance) == 100.0);
  testAssert(computeCooperativeMatrixTuningScore(100.0, 0.010001, hgemmPlan.errorTolerance) == 0.0);
  testAssert(computeCooperativeMatrixTuningScore(
    100.0, numeric_limits<double>::quiet_NaN(), hgemmPlan.errorTolerance) == 0.0);
  testAssert(computeCooperativeMatrixTuningScore(
    100.0, numeric_limits<double>::infinity(), hgemmPlan.errorTolerance) == 0.0);
  testAssert(makeMeasurementPlan("hgemmCooperativeMatrixNHWC", context).errorTolerance == 0.002);
  testAssert(makeMeasurementPlan("transformerScaleDotProduct", context).errorTolerance ==
    COOPERATIVE_MATRIX_ROPE_ERROR_TOLERANCE);
  testAssert(!tuningComplete("xgemm", context));
  recordTuning("xgemm", context);
  testAssert(tuningComplete("xgemm", context));

  config.hgemmCooperativeMatrix.SA = 1;
  config.pointwise.LOCAL_SIZE = 128;
  model.gemmLayers[0].inChannels += 16;
  model.gemmLayers[0].weight += 1.0;
  context.batchSize = 1;
  context.nnXLen = context.nnYLen = 13;
  for(const char* name: {"xgemmDirect", "conv3x3InputTransform", "conv3x3OutputTransform",
                        "gPool", "pointwise", "addChannelBiases"})
    recordTuning(name, context);
  const VulkanTuneParams savedConfig = config;
  int checkpoints = 0;
  const auto checkpoint = [&]() noexcept { checkpoints++; };
  testAssert(!completeRequiredTuners(context, config, checkpoint));
  testAssert(checkpoints == 0 && config == savedConfig);
  config.tuningRecords["xgemm"].completed = false;
  testAssert(!tuningComplete("xgemm", context));
  recordTuning("xgemm", context);

  ModelInfoForTuning transformer;
  transformer.modelVersion = 16;
  transformer.midNumChannels = 64;
  transformer.transformerHeadDim = transformer.transformerVHeadDim = 16;
  transformer.transformerNumHeads = transformer.transformerNumKVHeads = 4;
  transformer.transformerProjections = {{64, 192, true}, {64, 64, false}};
  transformer.gemmLayers = {{64, 32, 1.0, false, GemmTuneCase::Kind::Dense}};
  VulkanTuneParams transformerConfig;
  auto& mode = transformerConfig.vulkan;
  mode.canUseFP16Storage = mode.canUseFP16Compute = mode.canUseCooperativeMatrix = true;
  mode.shouldUseFP16Storage = mode.shouldUseFP16Compute = true;
  mode.shouldUseCooperativeMatrix = mode.shouldUseNHWC = true;
  testAssert(transformerConfig.isValid());
  TuningContext transformerContext{nullptr, 4, 19, 19, transformer, false, {}, nullptr, nullptr, true};
  transformerContext.selectedGemmConfig = &transformerConfig;
  transformerContext.tuningRecords = &transformerConfig.tuningRecords;
  const auto nhwcCases = getGemmTuningCases("xgemmDirect", transformer, transformerConfig);
  testAssert(getGemmTuningCases("transformerXgemm", transformer, transformerConfig).empty());
  recordTuning("xgemmDirect", transformerContext);
  mode.shouldUseCooperativeMatrix = false;
  applyExecutionMode(transformerConfig, transformer, enabled_t::Auto);
  testAssert(!mode.shouldUseNHWC);
  testAssert(getGemmTuningCases("xgemmDirect", transformer, transformerConfig).size() == nhwcCases.size());
  const auto projectionCases = getGemmTuningCases("transformerXgemm", transformer, transformerConfig);
  testAssert(projectionCases.size() == 2 && projectionCases[0].inChannels == 64 &&
    projectionCases[0].outChannels == 192 && projectionCases[0].weight == 1.0);
  transformer.transformerProjections.push_back({64, 192, true});
  const auto repeatedCases = getGemmTuningCases("transformerXgemm", transformer, transformerConfig);
  testAssert(repeatedCases.size() == 2 && repeatedCases[0].weight == 2.0);
  const auto projectionPlan = makeMeasurementPlan("transformerXgemm", transformerContext);
  testAssert(projectionPlan.gemmCases.size() == 2 && projectionPlan.workloadWeights[0] == 2.0);
  testAssert(projectionPlan.warmupRuns == 2 && projectionPlan.totalRuns == 12);
  testAssert(!tuningComplete("transformerXgemm", transformerContext));
  recordTuning("transformerXgemm", transformerContext);
  testAssert(tuningComplete("xgemmDirect", transformerContext));
  testAssert(!tuningComplete("transformerScaleDotProduct", transformerContext));
  for(const char* name: {"gPool", "pointwise", "addChannelBiases",
                        "transformerScaleDotProduct", "transformerRMSNorm"})
    recordTuning(name, transformerContext);
  testAssert(!completeRequiredTuners(transformerContext, transformerConfig, checkpoint));
  testAssert(checkpoints == 0 && !mode.shouldUseCooperativeMatrix && !mode.shouldUseNHWC);
  testAssert(!transformerConfig.tuningRecords["xgemm"].completed);
  testAssert(!transformerConfig.tuningRecords["xgemm16"].completed);

  Rand rand;
  const string filename = "vulkan-tuning-cache-test-" + Global::uint64ToHexString(rand.nextUInt64()) + ".txt";
  const auto fileGuard = makeScopeGuard([&]() { FileUtils::tryRemoveFile(filename); });
  config.transformerXgemm.WGD = 16;
  VulkanTuneParams::save(filename, config);
  testAssert(FileUtils::readFile(filename).find(".conditions=") == string::npos);
  VulkanTuneParams loaded = VulkanTuneParams::load(filename);
  testAssert(loaded == config);
  VulkanTuneParams separate = config;
  separate.transformerXgemm.WGD *= 2;
  testAssert(separate != config);
  testAssert(loaded.tuningRecords.size() == config.tuningRecords.size());
  testAssert(loaded.tuningRecords.at("xgemm").completed);
  testAssert(!loaded.tuningRecords.at("xgemm16").completed);
  string legacy;
  for(const string& line: FileUtils::readFileLines(filename, '\n'))
    if(line.compare(0, 7, "tuning.") != 0)
      legacy += line + "\n";
  const auto writeCache = [&](const string& contents) {
    ofstream out;
    FileUtils::open(out, filename);
    out << contents;
    out.close();
    testAssert(!out.fail());
  };
  writeCache(legacy);
  loaded = VulkanTuneParams::load(filename);
  testAssert(loaded == config && loaded.tuningRecords.empty());
  string oldProfile;
  for(const string& line: Global::split(legacy, '\n'))
    if(line.compare(0, 17, "transformerXgemm.") != 0)
      oldProfile += line + "\n";
  writeCache(oldProfile + "tuning.xgemmDirect.completed=1\n");
  loaded = VulkanTuneParams::load(filename);
  VulkanTuneParams oldExpected = config;
  oldExpected.transformerXgemm = config.xgemmDirect;
  testAssert(loaded == oldExpected && loaded.tuningRecords.at("xgemmDirect").completed);
  testAssert(loaded.tuningRecords.find("transformerXgemm") == loaded.tuningRecords.end());
  writeCache(legacy + "tuning.xgemm.completed=1\ntuning.xgemm.conditions=obsolete\n"
    "tuning.xgemm16.completed=0\ntuning.xgemm16.conditions=obsolete\n"
    "tuning.orphan.conditions=ignored\n");
  loaded = VulkanTuneParams::load(filename);
  testAssert(loaded.tuningRecords.size() == 2);
  testAssert(loaded.tuningRecords.at("xgemm").completed);
  testAssert(!loaded.tuningRecords.at("xgemm16").completed);
  VulkanTuneParams::save(filename, loaded);
  testAssert(FileUtils::readFile(filename).find(".conditions=") == string::npos);
  for(const char* record: {
      "tuning.xgemm.completed=2\n", "tuning.xgemm.completed=\n", "tuning.xgemm.unknown=1\n"}) {
    writeCache(legacy + record);
    bool rejected = false;
    try {
      VulkanTuneParams::load(filename);
    }
    catch(const StringError&) {
      rejected = true;
    }
    testAssert(rejected);
  }
  mode.shouldUseCooperativeMatrix = mode.shouldUseNHWC = mode.shouldUseTransformerDualGemmSwiGLU = true;
  mode.shouldUseHgemmCooperativeMatrixNCHW = true;
  mode.shouldUseTransformerCooperativeMatrixNCHW = mode.shouldUseTransformerCooperativeMatrixNHWC = true;
  testAssert(transformerConfig.isValid());
  VulkanTuneParams::save(filename, transformerConfig);
  string manual = FileUtils::readFile(filename);
  const string setting = "vulkan.shouldUseCooperativeMatrix=1";
  const size_t position = manual.find(setting);
  testAssert(position != string::npos);
  manual.replace(position, setting.size(), "vulkan.shouldUseCooperativeMatrix=0");
  writeCache(manual);
  loaded = VulkanTuneParams::load(filename);
  testAssert(!loaded.vulkan.shouldUseCooperativeMatrix && !loaded.vulkan.shouldUseNHWC);
  testAssert(!loaded.vulkan.shouldUseTransformerDualGemmSwiGLU);
  testAssert(loaded.vulkan.shouldUseHgemmCooperativeMatrixNCHW);
  testAssert(loaded.vulkan.shouldUseTransformerCooperativeMatrixNCHW);
  testAssert(loaded.vulkan.shouldUseTransformerCooperativeMatrixNHWC);
  cout << "Vulkan tuning cache tests passed" << endl;
}

VulkanTuneParams VulkanTuner::loadOrCreate(
  const string& tunerFile,
  const string& homeDataDirOverride,
  const string& gpuName,
  int nnXLen,
  int nnYLen,
  const ModelInfoForTuning& modelInfo,
  const VulkanDeviceInfo& deviceInfo,
  Logger* logger
) {
  string filename = tunerFile;
  if(filename.empty()) {
    filename = defaultDirectory(true, homeDataDirOverride) + "/" +
               defaultFileName(gpuName, nnXLen, nnYLen, modelInfo);
  }

  try {
    VulkanTuneParams loaded = VulkanTuneParams::load(filename);
    const VulkanParams available = VulkanTuner::getHardwareParams(deviceInfo);
    if(loaded.vulkan.canUseFP16Storage != available.canUseFP16Storage ||
       loaded.vulkan.canUseFP16Compute != available.canUseFP16Compute ||
       loaded.vulkan.canUseCooperativeMatrix != available.canUseCooperativeMatrix ||
       loaded.vulkan.canUseSubgroup != available.canUseSubgroup)
      throw IOError("Vulkan tuning capabilities changed for " + filename);
    if((loaded.vulkan.shouldUseCooperativeMatrix &&
        (!isValidCooperativeMatrixConfig(deviceInfo, loaded.hgemmCooperativeMatrixNHWC) ||
          !isValidCooperativeMatrixConfig(deviceInfo, loaded.transformerHgemmNHWC))) ||
       (loaded.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
        !isValidCooperativeMatrixConfig(deviceInfo, loaded.hgemmCooperativeMatrixNCHW)) ||
      (loaded.vulkan.shouldUseTransformerDualGemmSwiGLU &&
        !isValidCooperativeMatrixConfig(deviceInfo, loaded.transformerDualGemmSwiGLU)) ||
      (loaded.vulkan.shouldUseTransformerCooperativeMatrixNCHW &&
        !isValidCooperativeMatrixConfig(
          deviceInfo, loaded.transformerCooperativeMatrixNCHW,
          modelInfo.transformerHeadDim, modelInfo.transformerVHeadDim
        )) ||
      ((loaded.vulkan.shouldUseTransformerCooperativeMatrixNHWC ||
        loaded.transformerCooperativeMatrixNHWC.COOP_USE_NV2 == 1) &&
        !isValidCooperativeMatrixConfig(
          deviceInfo, loaded.transformerCooperativeMatrixNHWC,
          modelInfo.transformerHeadDim, modelInfo.transformerVHeadDim, true
        )))
      throw IOError("Vulkan cooperative-matrix configuration is no longer valid for " + filename);
    if(logger != nullptr)
      logger->write("Loaded Vulkan tuning parameters from: " + filename);
    loaded.vulkan.shouldUseNHWC =
      modelInfo.transformerHeadDim > 0 &&
      modelInfo.transformerVHeadDim > 0 &&
      loaded.vulkan.shouldUseCooperativeMatrix;
    return loaded;
  } catch(const StringError&) {
  }

  VulkanTuneParams params;
  params.vulkan = VulkanTuner::getHardwareParams(deviceInfo);
  VulkanTuneParams::save(filename, params);
  if(logger != nullptr)
    logger->write("Saved default Vulkan tuning parameters to: " + filename);
  return params;
}

VulkanTuneParams VulkanTuner::loadOrAutoTune(
  const string& tunerFile,
  const string& homeDataDirOverride,
  const string& gpuName,
  int nnXLen,
  int nnYLen,
  const ModelInfoForTuning& modelInfo,
  const VulkanDevice* device,
  Logger* logger,
  bool* didAutoTune,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode
) {
  if(getAttentionProbeCase(device) != nullptr)
    throw StringError("Run the Attention probe with 'katago tuner -output <disposable file>'");
  if(didAutoTune != nullptr)
    *didAutoTune = false;
  string filename = tunerFile;
  if(filename.empty())
    filename = defaultDirectory(
      true,
      homeDataDirOverride
    ) + "/" + defaultFileName(gpuName, nnXLen, nnYLen, modelInfo);

  VulkanTuneParams params;
  bool cacheLoaded = false;
  try {
    params = VulkanTuneParams::load(filename);
    VulkanTuneParams& loaded = params;
    if(device != nullptr) {
      const VulkanParams available = VulkanTuner::getHardwareParams(device->info);
      if(loaded.vulkan.canUseFP16Storage != available.canUseFP16Storage ||
         loaded.vulkan.canUseFP16Compute != available.canUseFP16Compute ||
         loaded.vulkan.canUseCooperativeMatrix != available.canUseCooperativeMatrix ||
         loaded.vulkan.canUseSubgroup != available.canUseSubgroup) {
        throw IOError("Vulkan tuning capabilities changed for " + filename);
      }
       if((loaded.vulkan.shouldUseCooperativeMatrix &&
           (!isValidCooperativeMatrixConfig(device->info, loaded.hgemmCooperativeMatrixNHWC) ||
            !isValidCooperativeMatrixConfig(device->info, loaded.transformerHgemmNHWC))) ||
         (loaded.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
          !isValidCooperativeMatrixConfig(device->info, loaded.hgemmCooperativeMatrixNCHW)) ||
         (loaded.vulkan.shouldUseTransformerDualGemmSwiGLU &&
          !isValidCooperativeMatrixConfig(device->info, loaded.transformerDualGemmSwiGLU)) ||
        (loaded.vulkan.shouldUseTransformerCooperativeMatrixNCHW &&
          !isValidCooperativeMatrixConfig(
            device->info, loaded.transformerCooperativeMatrixNCHW,
            modelInfo.transformerHeadDim, modelInfo.transformerVHeadDim
          )) ||
        ((loaded.vulkan.shouldUseTransformerCooperativeMatrixNHWC ||
          loaded.transformerCooperativeMatrixNHWC.COOP_USE_NV2 == 1) &&
          !isValidCooperativeMatrixConfig(
            device->info, loaded.transformerCooperativeMatrixNHWC,
            modelInfo.transformerHeadDim, modelInfo.transformerVHeadDim, true
          ))) {
        throw IOError("Vulkan cooperative-matrix configuration is no longer valid for " + filename);
      }
    }
    if(logger != nullptr)
      logger->write("Loaded Vulkan tuning parameters from: " + filename);
    cacheLoaded = true;
  } catch(const StringError& e) {
    if(logger != nullptr)
      logger->write(
        "Ignoring Vulkan tuning cache " + filename + ": " + e.what()
      );
  }

  if(device == nullptr) {
    if(cacheLoaded)
      return params;
    throw StringError("VulkanTuner::loadOrAutoTune: device is null");
  }
  if(!cacheLoaded) {
    params.vulkan = VulkanTuner::getHardwareParams(device->info);
    tune(device, DEFAULT_BATCH_SIZE, nnXLen, nnYLen, modelInfo, false, logger, params);
  }
  const VulkanParams savedMode = params.vulkan;
  applyExecutionMode(params, modelInfo, useFP16Mode, useNHWCMode);
  auto records = params.tuningRecords;
  const size_t originalRecordCount = records.size();
  initializeTuningRecords(records);
  VulkanTimestampTimer timer(device);
  TuningContext context{
    device, DEFAULT_BATCH_SIZE, nnXLen, nnYLen, modelInfo, false,
    getCooperativeMatrixTuneShapes(device), logger, &timer, true
  };
  context.tuningRecords = &records;
  const auto save = [&]() {
    params.tuningRecords = records;
    VulkanTuneParams saved = params;
    saved.vulkan = savedMode;
    VulkanTuneParams::save(filename, saved);
  };
  const bool completed = completeRequiredTuners(context, params, save);
  requireTuningDevice(context, "required tuning");
  params.tuningRecords = records;
  if(!cacheLoaded || completed || records.size() != originalRecordCount)
    save();
  if(!cacheLoaded || completed) {
    if(didAutoTune != nullptr)
      *didAutoTune = true;
    if(logger != nullptr)
      logger->write("Completed required Vulkan tuning and saved results to: " + filename);
  }
  return params;
}

#endif
