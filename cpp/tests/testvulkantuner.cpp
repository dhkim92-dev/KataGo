#ifdef USE_VULKAN_BACKEND

#include "../tests/tests.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../external/half-2.2.0/include/half.hpp"
#include "../neuralnet/vulkantuner.h"

using namespace std;

namespace {
  void setCooperativeMatrixProperty(
    VkCooperativeMatrixPropertiesKHR& property,
    uint32_t mSize,
    uint32_t nSize,
    uint32_t kSize,
    VkComponentTypeKHR accumulatorType = VK_COMPONENT_TYPE_FLOAT16_KHR
  ) {
    property = {};
    property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    property.MSize = mSize;
    property.NSize = nSize;
    property.KSize = kSize;
    property.AType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    property.BType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    property.CType = accumulatorType;
    property.ResultType = accumulatorType;
    property.scope = VK_SCOPE_SUBGROUP_KHR;
  }

  VKAPI_ATTR VkResult VKAPI_CALL fakeCooperativeMatrixProperties(
    VkPhysicalDevice,
    uint32_t* propertyCount,
    VkCooperativeMatrixPropertiesKHR* properties
  ) {
    if(properties == nullptr) {
      *propertyCount = 1;
      return VK_SUCCESS;
    }
    if(*propertyCount == 0)
      return VK_INCOMPLETE;
    setCooperativeMatrixProperty(properties[0], 16, 16, 16);
    *propertyCount = 1;
    return VK_SUCCESS;
  }

  VKAPI_ATTR VkResult VKAPI_CALL fakeNonSubgroupCooperativeMatrixProperties(
    VkPhysicalDevice physicalDevice,
    uint32_t* propertyCount,
    VkCooperativeMatrixPropertiesKHR* properties
  ) {
    VkResult result = fakeCooperativeMatrixProperties(physicalDevice, propertyCount, properties);
    if(properties != nullptr && result == VK_SUCCESS)
      properties[0].scope = VK_SCOPE_DEVICE_KHR;
    return result;
  }

  VKAPI_ATTR VkResult VKAPI_CALL fakeSixteenByEightCooperativeMatrixProperties(
    VkPhysicalDevice physicalDevice,
    uint32_t* propertyCount,
    VkCooperativeMatrixPropertiesKHR* properties
  ) {
    VkResult result = fakeCooperativeMatrixProperties(physicalDevice, propertyCount, properties);
    if(properties != nullptr && result == VK_SUCCESS)
      properties[0].NSize = 8;
    return result;
  }

  VKAPI_ATTR VkResult VKAPI_CALL fakeNonPowerOfTwoCooperativeMatrixProperties(
    VkPhysicalDevice physicalDevice,
    uint32_t* propertyCount,
    VkCooperativeMatrixPropertiesKHR* properties
  ) {
    VkResult result = fakeCooperativeMatrixProperties(physicalDevice, propertyCount, properties);
    if(properties != nullptr && result == VK_SUCCESS)
      properties[0].NSize = 24;
    return result;
  }

  VKAPI_ATTR VkResult VKAPI_CALL fakeFloat32AccumulatorCooperativeMatrixProperties(
    VkPhysicalDevice physicalDevice,
    uint32_t* propertyCount,
    VkCooperativeMatrixPropertiesKHR* properties
  ) {
    VkResult result = fakeCooperativeMatrixProperties(physicalDevice, propertyCount, properties);
    if(properties != nullptr && result == VK_SUCCESS) {
      properties[0].CType = VK_COMPONENT_TYPE_FLOAT32_KHR;
      properties[0].ResultType = VK_COMPONENT_TYPE_FLOAT32_KHR;
    }
    return result;
  }

  int incompletePropertyDataQueries = 0;
  VKAPI_ATTR VkResult VKAPI_CALL fakeIncompleteCooperativeMatrixProperties(
    VkPhysicalDevice,
    uint32_t* propertyCount,
    VkCooperativeMatrixPropertiesKHR* properties
  ) {
    if(properties == nullptr) {
      *propertyCount = incompletePropertyDataQueries == 0 ? 1 : 4;
      return VK_SUCCESS;
    }
    if(incompletePropertyDataQueries++ == 0) {
      if(*propertyCount > 0)
        setCooperativeMatrixProperty(properties[0], 16, 16, 16);
      *propertyCount = std::min<uint32_t>(*propertyCount, 1);
      return VK_INCOMPLETE;
    }
    const uint32_t count = std::min<uint32_t>(*propertyCount, 4);
    if(count > 0) setCooperativeMatrixProperty(properties[0], 1, 1, 1);
    if(count > 1) setCooperativeMatrixProperty(properties[1], 8, 32, 16);
    if(count > 2) setCooperativeMatrixProperty(properties[2], 8, 32, 16);
    if(count > 3) setCooperativeMatrixProperty(properties[3], 32, 8, 16, VK_COMPONENT_TYPE_FLOAT32_KHR);
    *propertyCount = count;
    return VK_SUCCESS;
  }

  bool loadThrows(const string& filename) {
    try {
      (void)VulkanTuneParams::load(filename);
      return false;
    } catch(const StringError&) {
      return true;
    }
  }

  void writeText(const string& filename, const string& contents) {
    ofstream out;
    FileUtils::open(out, filename);
    out << contents;
    out.close();
  }
}  // namespace

void Tests::runVulkanTunerPersistenceTests() {
  cout << "Running Vulkan tuner persistence tests" << endl;
  const float nan = numeric_limits<float>::quiet_NaN();
  const float inf = numeric_limits<float>::infinity();
  testAssert(VulkanTuner::computeErrorProp({}, {}) == 0.0);
  testAssert(VulkanTuner::computeErrorProp({0.0f, 0.0f}, {0.0f, 0.0f}) == 0.0);
  testAssert(VulkanTuner::computeErrorProp({3.0f, 4.0f}, {3.0f, 4.0f}) == 0.0);
  testAssert(fabs(VulkanTuner::computeErrorProp({3.0f, 4.0f}, {3.0f, 5.0f}) - 0.2) < 1e-12);
  testAssert(fabs(VulkanTuner::computeErrorProp({30.0f, 40.0f}, {30.0f, 50.0f}) - 0.2) < 1e-12);
  testAssert(VulkanTuner::computeErrorProp({1.0f}, {}) == 1.0);
  testAssert(VulkanTuner::computeErrorProp({}, {1.0f}) == 1.0);
  testAssert(VulkanTuner::computeErrorProp({nan}, {1.0f}) == 1.0);
  testAssert(VulkanTuner::computeErrorProp({1.0f}, {nan}) == 1.0);
  testAssert(VulkanTuner::computeErrorProp({inf}, {inf}) == 1.0);
  testAssert(VulkanTuner::computeErrorProp({1.0f}, {-inf}) == 1.0);

  // Every rounded value differs, but FP16 rounding is a small numerical error.
  const vector<float> reference = {1.0003f, -2.0007f, 0.10003f};
  vector<float> rounded;
  for(float value: reference)
    rounded.push_back(static_cast<float>(half_float::half_cast<half_float::half>(value)));
  for(size_t i = 0; i < reference.size(); i++)
    testAssert(reference[i] != rounded[i]);
  const double halfError = VulkanTuner::computeErrorProp(reference, rounded);
  testAssert(halfError > 0.0 && halfError < 0.001);

  testAssert(VulkanTuner::computeTuningScore(100.0, 0.0, 0.005) == 100.0);
  testAssert(fabs(VulkanTuner::computeTuningScore(100.0, 0.005, 0.005) - 90.0 * (1.0 - sqrt(0.5))) < 1e-12);
  // The tolerance controls the penalty; only the larger hard cutoff rejects.
  testAssert(VulkanTuner::computeTuningScore(100.0, 0.006, 0.005) > 0.0);
  testAssert(VulkanTuner::computeTuningScore(100.0, 0.025, 0.005) > 0.0);
  testAssert(VulkanTuner::computeTuningScore(100.0, 0.025001, 0.005) == 0.0);
  testAssert(VulkanTuner::computeTuningScore(100.0, 0.5, 0.2) > 0.0);
  testAssert(VulkanTuner::computeTuningScore(100.0, 0.500001, 0.2) == 0.0);
  testAssert(VulkanTuner::computeTuningScore(100.0, halfError, 0.005) > 0.0);
  testAssert(VulkanTuner::computeTuningScore(100.0, nan, 0.005) == 0.0);
  testAssert(VulkanTuner::computeTuningScore(100.0, inf, 0.005) == 0.0);
  for(double badRate: {0.0, -1.0, static_cast<double>(nan), static_cast<double>(inf)}) {
    testAssert(VulkanTuner::computeTuningScore(badRate, 0.0, 0.005) == 0.0);
    testAssert(!VulkanTuner::isFastEnough(badRate, 100.0, 1.0));
    testAssert(!VulkanTuner::isFastEnough(100.0, badRate, 1.0));
  }
  // Optional paths use the adjustable VulkanTuner throughput thresholds.
  testAssert(VulkanTuner::isFastEnough(100.0, 100.0, 1.0));
  testAssert(!VulkanTuner::isFastEnough(99.999, 100.0, 1.0));
  for(double threshold: {
    VulkanTuner::FP16_COMPUTE_MIN_THROUGHPUT_RATIO,
    VulkanTuner::FP16_STORAGE_MIN_THROUGHPUT_RATIO,
    VulkanTuner::COOPERATIVE_MATRIX_MIN_THROUGHPUT_RATIO,
    VulkanTuner::COOPERATIVE_MATRIX_1X1_MIN_THROUGHPUT_RATIO
  }) {
    testAssert(VulkanTuner::isFastEnough(100.0 * threshold, 100.0, threshold));
    testAssert(!VulkanTuner::isFastEnough(100.0 * threshold - 0.001, 100.0, threshold));
  }
  testAssert(VulkanTuner::FP16_COMPUTE_MIN_THROUGHPUT_RATIO == 1.20);
  testAssert(VulkanTuner::FP16_STORAGE_MIN_THROUGHPUT_RATIO == 1.20);
  testAssert(VulkanTuner::COOPERATIVE_MATRIX_MIN_THROUGHPUT_RATIO == 0.90);
  testAssert(VulkanTuner::COOPERATIVE_MATRIX_1X1_MIN_THROUGHPUT_RATIO == 1.20);
  testAssert(VulkanTuner::COOPERATIVE_MATRIX_SHAPE_SCORE_RATIO == 0.95);
  testAssert(VulkanTuner::COOPERATIVE_MATRIX_MIN_SHAPES_PER_ACCUMULATOR == 2);

  testAssert(VulkanTuner::defaultDirectory(false, "tests/scratch") == "tests/scratch/vulkantuning");
  testAssert(
    VulkanTuner::defaultFileName("Apple M2", 19, 19, 96, 8) ==
    "tune" + to_string(VulkanTuner::TUNER_VERSION) + "_gpuAppleM2_x19_y19_c96_mv8.txt"
  );

  MakeDir::make("tests/scratch");
  const string filename = "tests/scratch/vulkantuner-roundtrip.txt";
  const string defaultsFilename = "tests/scratch/vulkantuner-defaults.txt";
  FileUtils::tryRemoveFile(defaultsFilename);
  VulkanTuneParams defaults;
  VulkanDeviceInfo deviceInfo = {};
  deviceInfo.storage16BitFeatures.storageBuffer16BitAccess = VK_TRUE;
  deviceInfo.shaderFloat16Int8Features.shaderFloat16 = VK_TRUE;
  deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix = VK_TRUE;
  deviceInfo.cooperativeMatrixPropertiesFn = fakeCooperativeMatrixProperties;
  deviceInfo.subgroupProperties.supportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
  deviceInfo.subgroupProperties.subgroupSize = 32;
  deviceInfo.subgroupSizeControlFeatures.computeFullSubgroups = VK_TRUE;
  deviceInfo.properties.limits.maxComputeWorkGroupSize[0] = 1024;
  deviceInfo.properties.limits.maxComputeWorkGroupSize[1] = 1024;
  deviceInfo.properties.limits.maxComputeWorkGroupSize[2] = 64;
  deviceInfo.properties.limits.maxComputeWorkGroupInvocations = 1024;
  deviceInfo.properties.limits.maxComputeSharedMemorySize = 32768;
  const VulkanParams hardwareParams = VulkanTuner::getHardwareParams(deviceInfo);
  testAssert(hardwareParams.canUseFP16Storage);
  testAssert(hardwareParams.canUseFP16Compute);
  testAssert(hardwareParams.canUseCooperativeMatrix);
  testAssert(hardwareParams.canUseSubgroup);
  VulkanDeviceInfo nonSubgroupDeviceInfo = deviceInfo;
  nonSubgroupDeviceInfo.cooperativeMatrixPropertiesFn = fakeNonSubgroupCooperativeMatrixProperties;
  testAssert(!VulkanTuner::getHardwareParams(nonSubgroupDeviceInfo).canUseCooperativeMatrix);
  VulkanDeviceInfo nonPowerOfTwoDeviceInfo = deviceInfo;
  nonPowerOfTwoDeviceInfo.cooperativeMatrixPropertiesFn = fakeNonPowerOfTwoCooperativeMatrixProperties;
  testAssert(!VulkanTuner::getHardwareParams(nonPowerOfTwoDeviceInfo).canUseCooperativeMatrix);
  VulkanDevice sixteenByEightDevice = {};
  sixteenByEightDevice.info = deviceInfo;
  sixteenByEightDevice.info.cooperativeMatrixPropertiesFn = fakeSixteenByEightCooperativeMatrixProperties;
  testAssert(VulkanTuner::getHardwareParams(sixteenByEightDevice.info).canUseCooperativeMatrix);
  HGemmCooperativeMatrixTuneParams expectedSixteenByEightParams;
  expectedSixteenByEightParams.MWARP = 16;
  expectedSixteenByEightParams.NWARP = 8;
  expectedSixteenByEightParams.KDIM = 16;
  expectedSixteenByEightParams.MWG = 16;
  expectedSixteenByEightParams.NWG = 8;
  expectedSixteenByEightParams.KWG = 16;
  expectedSixteenByEightParams.MWAVE = 16;
  expectedSixteenByEightParams.NWAVE = 8;
  testAssert(expectedSixteenByEightParams.isValid());
  HGemmCooperativeMatrixTuneParams sixteenByEightParams;
  testAssert(VulkanTuner::HgemmCooperativeMatrixTuner::selectCooperativeMatrixProperties(
    &sixteenByEightDevice, sixteenByEightParams
  ));
  testAssert(sixteenByEightParams.MWARP == 16);
  testAssert(sixteenByEightParams.NWARP == 8);
  testAssert(sixteenByEightParams.KDIM == 16);
  testAssert(sixteenByEightParams.isValid());
  VulkanDevice float32AccumulatorDevice = {};
  float32AccumulatorDevice.info = deviceInfo;
  float32AccumulatorDevice.info.cooperativeMatrixPropertiesFn = fakeFloat32AccumulatorCooperativeMatrixProperties;
  HGemmCooperativeMatrixTuneParams float32AccumulatorParams;
  float32AccumulatorParams.accType = 32;
  testAssert(VulkanTuner::HgemmCooperativeMatrixTuner::selectCooperativeMatrixProperties(
    &float32AccumulatorDevice, float32AccumulatorParams
  ));
  testAssert(float32AccumulatorParams.accType == 32);
  HGemmCooperativeMatrixNCHWTuneParams float32AccumulatorNCHWParams;
  float32AccumulatorNCHWParams.accType = 32;
  testAssert(VulkanTuner::HgemmCooperativeMatrixNCHWTuner::selectCooperativeMatrixProperties(
    &float32AccumulatorDevice, float32AccumulatorNCHWParams
  ));
  testAssert(float32AccumulatorNCHWParams.accType == 32);
  VulkanDevice incompletePropertyDevice = {};
  incompletePropertyDevice.info = deviceInfo;
  incompletePropertyDevice.info.cooperativeMatrixPropertiesFn = fakeIncompleteCooperativeMatrixProperties;
  incompletePropertyDataQueries = 0;
  HGemmCooperativeMatrixTuneParams oneByOneParams;
  oneByOneParams.MWARP = 1;
  oneByOneParams.NWARP = 1;
  oneByOneParams.KDIM = 1;
  testAssert(VulkanTuner::HgemmCooperativeMatrixTuner::selectCooperativeMatrixProperties(
    &incompletePropertyDevice, oneByOneParams
  ));
  testAssert(incompletePropertyDataQueries >= 2);
  testAssert(oneByOneParams.MWARP == 1 && oneByOneParams.NWARP == 1 && oneByOneParams.KDIM == 1);
  testAssert(oneByOneParams.VWM == 1 && oneByOneParams.VWN == 1);
  HGemmCooperativeMatrixTuneParams asymmetricParams;
  asymmetricParams.MWARP = 8;
  asymmetricParams.NWARP = 32;
  asymmetricParams.KDIM = 16;
  testAssert(VulkanTuner::HgemmCooperativeMatrixTuner::selectCooperativeMatrixProperties(
    &incompletePropertyDevice, asymmetricParams
  ));
  testAssert(asymmetricParams.MWARP == 8 && asymmetricParams.NWARP == 32 && asymmetricParams.KDIM == 16);
  HGemmCooperativeMatrixTuneParams asymmetricFloat32Params;
  asymmetricFloat32Params.accType = 32;
  asymmetricFloat32Params.MWARP = 32;
  asymmetricFloat32Params.NWARP = 8;
  asymmetricFloat32Params.KDIM = 16;
  testAssert(VulkanTuner::HgemmCooperativeMatrixTuner::selectCooperativeMatrixProperties(
    &incompletePropertyDevice, asymmetricFloat32Params
  ));
  testAssert(asymmetricFloat32Params.accType == 32 && asymmetricFloat32Params.MWARP == 32 &&
             asymmetricFloat32Params.NWARP == 8 && asymmetricFloat32Params.KDIM == 16);
  HGemmCooperativeMatrixTuneParams nonPowerOfTwoParams = sixteenByEightParams;
  nonPowerOfTwoParams.NWARP = 24;
  testAssert(!nonPowerOfTwoParams.isValid());
  HGemmCooperativeMatrixNCHWTuneParams thirtyTwoByEightParams;
  thirtyTwoByEightParams.MWARP = 32;
  thirtyTwoByEightParams.NWARP = 8;
  thirtyTwoByEightParams.KDIM = 16;
  thirtyTwoByEightParams.MWG = 32;
  thirtyTwoByEightParams.NWG = 8;
  thirtyTwoByEightParams.KWG = 16;
  thirtyTwoByEightParams.MWAVE = 32;
  thirtyTwoByEightParams.NWAVE = 8;
  testAssert(thirtyTwoByEightParams.isValid());
  testAssert(thirtyTwoByEightParams.getRequiredSpatialAlignment() == 32);
  HGemmCooperativeMatrixTuneParams multiSubgroupParams;
  multiSubgroupParams.MWG = 64;
  multiSubgroupParams.NWG = 64;
  multiSubgroupParams.KWG = 32;
  multiSubgroupParams.MWAVE = 32;
  multiSubgroupParams.NWAVE = 32;
  testAssert(multiSubgroupParams.isValid());
  testAssert(isValidCooperativeMatrixConfig(deviceInfo, multiSubgroupParams));
  HGemmCooperativeMatrixNCHWTuneParams multiSubgroupNCHWParams;
  multiSubgroupNCHWParams.MWG = 64;
  multiSubgroupNCHWParams.NWG = 64;
  multiSubgroupNCHWParams.KWG = 32;
  multiSubgroupNCHWParams.MWAVE = 32;
  multiSubgroupNCHWParams.NWAVE = 32;
  testAssert(multiSubgroupNCHWParams.isValid());
  testAssert(isValidCooperativeMatrixConfig(deviceInfo, multiSubgroupNCHWParams));
  HGemmCooperativeMatrixTuneParams kDimensionOneParams;
  kDimensionOneParams.MWARP = 4;
  kDimensionOneParams.NWARP = 4;
  kDimensionOneParams.KDIM = 1;
  kDimensionOneParams.MWG = 4;
  kDimensionOneParams.NWG = 4;
  kDimensionOneParams.KWG = 1;
  kDimensionOneParams.MWAVE = 4;
  kDimensionOneParams.NWAVE = 4;
  kDimensionOneParams.VWM = 4;
  kDimensionOneParams.VWN = 4;
  testAssert(kDimensionOneParams.isValid());
  HGemmCooperativeMatrixTuneParams nonPowerOfTwoRatioParams = multiSubgroupParams;
  nonPowerOfTwoRatioParams.MWAVE = 48;
  nonPowerOfTwoRatioParams.MWG = 96;
  testAssert(!nonPowerOfTwoRatioParams.isValid());
  VulkanDeviceInfo limitedDeviceInfo = deviceInfo;
  limitedDeviceInfo.properties.limits.maxComputeWorkGroupInvocations = 64;
  testAssert(!isValidCooperativeMatrixConfig(limitedDeviceInfo, multiSubgroupParams));
  VulkanDeviceInfo limitedSharedMemoryDeviceInfo = deviceInfo;
  limitedSharedMemoryDeviceInfo.properties.limits.maxComputeSharedMemorySize = 1024;
  HGemmCooperativeMatrixTuneParams sharedMemoryParams = multiSubgroupParams;
  sharedMemoryParams.SA = 1;
  testAssert(!isValidCooperativeMatrixConfig(limitedSharedMemoryDeviceInfo, sharedMemoryParams));
  testAssert(!hardwareParams.shouldUseFP16Storage);
  testAssert(!hardwareParams.shouldUseFP16Compute);
  defaults.vulkan.canUseFP16Storage = true;
  defaults.vulkan.canUseFP16Compute = true;
  defaults.vulkan.canUseCooperativeMatrix = true;
  defaults.vulkan.canUseSubgroup = true;
  testAssert(defaults.xgemmDirect.WGD == 8);
  testAssert(defaults.xgemmDirect.MDIMCD == 1);
  testAssert(defaults.xgemmDirect.NDIMCD == 1);
  testAssert(defaults.xgemmDirect.MDIMAD == 1);
  testAssert(defaults.xgemmDirect.NDIMBD == 1);
  testAssert(defaults.xgemmDirect.KWID == 2);
  testAssert(defaults.xgemmDirect.PADA == 1);
  testAssert(defaults.xgemmDirect.PADB == 1);
  testAssert(defaults.xgemmDirect.VWMD == 4);
  testAssert(defaults.xgemmDirect.VWND == 4);
  testAssert(defaults.xgemm.VWM == 4);
  testAssert(defaults.xgemm.VWN == 4);
  testAssert(defaults.xgemm.KWI == 1);
  testAssert(defaults.xgemm16.VWM == 4);
  testAssert(defaults.xgemm16.VWN == 4);
  testAssert(defaults.xgemm16.KWI == 1);
  testAssert(defaults.hgemmCooperativeMatrix.VWM == 4);
  testAssert(defaults.hgemmCooperativeMatrix.VWN == 4);
  testAssert(defaults.hgemmCooperativeMatrixNCHW.VWM == 4);
  testAssert(defaults.hgemmCooperativeMatrixNCHW.VWN == 4);
  testAssert(defaults.spatialRMSNorm.TILE_SIZE == 32);
  testAssert(defaults.spatialRMSNorm.APPLY_ELTS_PER_THREAD == 1);
  testAssert(defaults.rmsNorm.WG_C_SIZE == 64);
  testAssert(defaults.rmsNorm.WG_XY_SIZE == 1);
  testAssert(defaults.rmsNorm.C_PER_THREAD == 4);
  VulkanTuner::ModelInfoForTuning modelInfo;
  modelInfo.trunkNumChannels = 96;
  modelInfo.modelVersion = 8;
  testAssert(
    VulkanTuner::loadOrCreate(defaultsFilename, "", "", 19, 19, modelInfo, deviceInfo, nullptr) == defaults
  );
  const string deviceLimitedFilename = "tests/scratch/vulkantuner-device-limited.txt";
  VulkanTuneParams deviceLimitedConfig = defaults;
  deviceLimitedConfig.vulkan.shouldUseFP16Storage = true;
  deviceLimitedConfig.vulkan.shouldUseFP16Compute = true;
  deviceLimitedConfig.vulkan.shouldUseCooperativeMatrix = true;
  deviceLimitedConfig.hgemmCooperativeMatrix = multiSubgroupParams;
  VulkanTuneParams::save(deviceLimitedFilename, deviceLimitedConfig);
  const VulkanTuneParams revalidatedConfig = VulkanTuner::loadOrCreate(
    deviceLimitedFilename, "", "", 19, 19, modelInfo, limitedDeviceInfo, nullptr
  );
  testAssert(!revalidatedConfig.vulkan.shouldUseCooperativeMatrix);
  vector<string> defaultLines = FileUtils::readFileLines(defaultsFilename, '\n');
  testAssert(defaultLines[0] == "VERSION=" + to_string(VulkanTuner::TUNER_VERSION));
  testAssert(defaultLines[1] == "vulkan.canUseFP16Storage=1");
  testAssert(defaultLines[2] == "vulkan.canUseFP16Compute=1");
  testAssert(defaultLines[3] == "vulkan.canUseCooperativeMatrix=1");
  testAssert(defaultLines[4] == "vulkan.canUseSubgroup=1");
  testAssert(defaultLines[5] == "vulkan.shouldUseFP16Storage=0");
  testAssert(defaultLines[6] == "vulkan.shouldUseFP16Compute=0");
  testAssert(defaultLines[7] == "vulkan.shouldUseCooperativeMatrix=0");
  testAssert(defaultLines[8] == "vulkan.shouldUseHgemmCooperativeMatrixNCHW=0");
  testAssert(defaultLines[9] == "vulkan.shouldUseSubgroup=0");
  testAssert(defaultLines.size() == 101);
  const auto lineIndex = [&](const string& prefix) {
    for(size_t i = 0; i < defaultLines.size(); i++) {
      if(defaultLines[i].find(prefix) == 0)
        return i;
    }
    return defaultLines.size();
  };
  testAssert(lineIndex("p32s32.") == defaultLines.size());
  testAssert(lineIndex("p32s16.") == defaultLines.size());
  testAssert(lineIndex("p16s16.") == defaultLines.size());
  testAssert(lineIndex("xgemmDirect.WGD=") < lineIndex("xgemm.MWG="));
  testAssert(lineIndex("xgemm.MWG=") < lineIndex("xgemm16.MWG="));
  testAssert(lineIndex("xgemm16.MWG=") < lineIndex("hgemmCooperativeMatrix.MWG="));
  testAssert(lineIndex("hgemmCooperativeMatrix.MWG=") < lineIndex("hgemmCooperativeMatrixNCHW.MWG="));
  testAssert(lineIndex("hgemmCooperativeMatrix.accType=") < lineIndex("hgemmCooperativeMatrixNCHW.MWG="));
  testAssert(lineIndex("hgemmCooperativeMatrixNCHW.accType=") < lineIndex("conv3x3.inTileXSize="));
  testAssert(lineIndex("hgemmCooperativeMatrixNCHW.CType=") == defaultLines.size());
  testAssert(lineIndex("hgemmCooperativeMatrixNCHW.ResultType=") == defaultLines.size());
  testAssert(lineIndex("hgemmCooperativeMatrixNCHW.MWG=") < lineIndex("conv3x3.inTileXSize="));
  testAssert(lineIndex("conv5x5.inTileXSize=") < lineIndex("gPool.XYSTRIDE="));
  testAssert(lineIndex("gPool.XYSTRIDE=") < lineIndex("transformer.ATTN_BLOCK_Q="));
  testAssert(lineIndex("transformer.ATTN_BLOCK_Q=") < lineIndex("rmsNorm.WG_C_SIZE="));
  testAssert(lineIndex("rmsNorm.WG_C_SIZE=") < lineIndex("pointwise.ELTS_PER_THREAD="));
  testAssert(lineIndex("pointwise.ELTS_PER_THREAD=") < lineIndex("addChannelBiases.XY_ELTS_PER_THREAD="));
  testAssert(lineIndex("addChannelBiases.XY_ELTS_PER_THREAD=") < lineIndex("spatialRMSNorm.TILE_SIZE="));
  testAssert(VulkanTuneParams::load(defaultsFilename) == defaults);
  const string staleDefaultsFilename = "tests/scratch/vulkantuner-stale-defaults.txt";
  VulkanTuneParams staleDefaults = defaults;
  staleDefaults.vulkan.canUseFP16Storage = false;
  staleDefaults.vulkan.canUseFP16Compute = false;
  staleDefaults.vulkan.canUseCooperativeMatrix = false;
  staleDefaults.vulkan.canUseSubgroup = false;
  VulkanTuneParams::save(staleDefaultsFilename, staleDefaults);
  const VulkanTuneParams recreatedDefaults = VulkanTuner::loadOrCreate(
    staleDefaultsFilename, "", "", 19, 19, modelInfo, deviceInfo, nullptr
  );
  testAssert(recreatedDefaults.vulkan.canUseFP16Storage);
  testAssert(recreatedDefaults.vulkan.canUseFP16Compute);
  testAssert(recreatedDefaults.vulkan.canUseCooperativeMatrix);
  testAssert(recreatedDefaults.vulkan.canUseSubgroup);

  VulkanTuneParams params;
  params.conv3x3.inTileXSize = 4;
  params.conv3x3.inTileYSize = 4;
  params.conv3x3.outTileXSize = 2;
  params.conv3x3.outTileYSize = 2;
  params.conv3x3.inputTransformLocalXSize = 64;
  params.conv3x3.inputTransformLocalYSize = 4;
  params.conv5x5.outputTransformLocalXSize = 16;
  params.xgemm.KWG = 32;
  params.xgemm16.MWG = 64;
  params.xgemm16.NWG = 64;
  params.xgemm.VWM = 2;
  params.xgemm.VWN = 1;
  params.xgemm16.VWM = 1;
  params.xgemm16.VWN = 2;
  params.xgemmDirect.VWMD = 1;
  params.xgemmDirect.VWND = 2;
  params.addChannelBiases.XY_ELTS_PER_THREAD = 2;
  params.addChannelBiases.NC_ELTS_PER_THREAD = 8;
  params.pointwise.LOCAL_SIZE = 128;
  params.pointwise.ELTS_PER_THREAD = 2;
  params.gPool.XYSTRIDE = 16;
  params.gPool.CHANNELSTRIDE = 2;
  params.gPool.BATCHSTRIDE = 2;
  params.transformer.ATTN_BLOCK_Q = 64;
  params.transformer.ATTN_BLOCK_KV = 16;
  params.transformer.Q_PER_THREAD = 2;
  params.rmsNorm.WG_C_SIZE = 64;
  params.rmsNorm.WG_XY_SIZE = 4;
  params.rmsNorm.C_PER_THREAD = 2;
  params.spatialRMSNorm.TILE_SIZE = 64;
  params.spatialRMSNorm.APPLY_ELTS_PER_THREAD = 4;
  params.hgemmCooperativeMatrixNCHW.NWG = 32;
  params.hgemmCooperativeMatrixNCHW.KWG = 32;
  params.hgemmCooperativeMatrix.VWM = 2;
  params.hgemmCooperativeMatrix.VWN = 1;
  params.hgemmCooperativeMatrix.accType = 32;
  params.hgemmCooperativeMatrixNCHW.VWM = 1;
  params.hgemmCooperativeMatrixNCHW.VWN = 2;
  params.hgemmCooperativeMatrixNCHW.accType = 32;
  params.vulkan.canUseFP16Storage = true;
  params.vulkan.canUseFP16Compute = true;
  params.vulkan.canUseCooperativeMatrix = true;
  params.vulkan.shouldUseFP16Storage = true;
  params.vulkan.shouldUseFP16Compute = true;
  params.vulkan.shouldUseCooperativeMatrix = true;
  params.vulkan.shouldUseHgemmCooperativeMatrixNCHW = true;
  params.vulkan.canUseSubgroup = true;
  params.vulkan.shouldUseSubgroup = true;
  testAssert(params.isValid());
  params.vulkan.shouldUseCooperativeMatrix = false;
  testAssert(params.isValid());
  VulkanTuneParams::save(filename, params);
  VulkanTuneParams loaded = VulkanTuneParams::load(filename);
  testAssert(loaded == params);
  testAssert(loaded.xgemm16.MWG == 64);
  testAssert(loaded.hgemmCooperativeMatrix.accType == 32);
  testAssert(loaded.hgemmCooperativeMatrixNCHW.accType == 32);

  writeText(filename, "VERSION=999\n");
  testAssert(loadThrows(filename));
  writeText(filename, "VERSION=" + to_string(VulkanTuner::TUNER_VERSION) + "\nvulkan.canUseFP16Storage=not-an-int\n");
  testAssert(loadThrows(filename));

  VulkanTuneParams invalid = params;
  invalid.xgemm.MDIMC = 0;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.conv3x3.inTileXSize = 5;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.gPool.XYSTRIDE = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.xgemmDirect.MDIMAD = 16;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.xgemmDirect.MDIMCD = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.xgemm.VWM = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.hgemmCooperativeMatrix.VWM = 8;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.hgemmCooperativeMatrix.accType = 64;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.xgemmDirect.VWND = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.xgemmDirect.PADA = 2;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.xgemmDirect.PADB = 0;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.addChannelBiases.XY_ELTS_PER_THREAD = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.addChannelBiases.XY_ELTS_PER_THREAD = 8;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.addChannelBiases.NC_ELTS_PER_THREAD = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.addChannelBiases.NC_ELTS_PER_THREAD = 16;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.pointwise.ELTS_PER_THREAD = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.pointwise.LOCAL_SIZE = 16;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.pointwise.LOCAL_SIZE = 96;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.pointwise.LOCAL_SIZE = 1024;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.spatialRMSNorm.TILE_SIZE = 96;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.spatialRMSNorm.APPLY_ELTS_PER_THREAD = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.rmsNorm.WG_C_SIZE = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.rmsNorm.WG_C_SIZE = 16;
  invalid.rmsNorm.WG_XY_SIZE = 64;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.rmsNorm.C_PER_THREAD = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.rmsNorm.C_PER_THREAD = 64;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.transformer.ATTN_BLOCK_Q = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.transformer.ATTN_BLOCK_Q = 512;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.transformer.ATTN_BLOCK_KV = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.transformer.ATTN_BLOCK_KV = 256;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.transformer.Q_PER_THREAD = 3;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.transformer.Q_PER_THREAD = 16;
  testAssert(!invalid.isValid());
  bool saveThrew = false;
  try {
    VulkanTuneParams::save(filename, invalid);
  } catch(const StringError&) {
    saveThrew = true;
  }
  testAssert(saveThrew);

  FileUtils::tryRemoveFile(filename);
  FileUtils::tryRemoveFile(defaultsFilename);
  cout << "Vulkan tuner persistence tests passed" << endl;
}

#endif
