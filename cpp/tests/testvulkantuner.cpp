#ifdef USE_VULKAN_BACKEND

#include "../tests/tests.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../external/half-2.2.0/include/half.hpp"
#include "../neuralnet/vulkantuner.h"
#include "../neuralnet/vulkanbackend.h"

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

  VKAPI_ATTR VkResult VKAPI_CALL fakeSaturatingCooperativeMatrixProperties(
    VkPhysicalDevice physicalDevice,
    uint32_t* propertyCount,
    VkCooperativeMatrixPropertiesKHR* properties
  ) {
    VkResult result = fakeCooperativeMatrixProperties(physicalDevice, propertyCount, properties);
    if(properties != nullptr && result == VK_SUCCESS)
      properties[0].saturatingAccumulation = VK_TRUE;
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
#ifdef VK_BENCHMARK
  {
    // CPU-only checks: grouping must retain equal shader names in different layers.
    ComputeHandleInternal::BenchmarkInfos infos;
    infos[{"Layer A", "HGEMM"}] = {2, 2000000};
    infos[{"Layer A", "BIAS"}] = {1, 1000000};
    infos[{"Layer B,\"q\"", "HGEMM"}] = {4, 4000000};
    infos[{"Zero", "EMPTY"}] = {0, 0};
    ostringstream csv;
    ComputeHandleInternal::writeVulkanBenchmarkCSV(csv, infos, 1000);
    const string expected =
      "layer,pipeline,average_ms,calls_per_sec,calls_per_visit,total_ms,calls,execution_percent,layer_total_ms,layer_execution_percent,execution_percent_within_layer\n"
      "\"Layer B,\"\"q\"\"\",\"HGEMM\",1.000000,1000.000000,0.004000,4.000000,4,57.142857,4.000000,57.142857,100.000000\n"
      "\"Layer A\",\"HGEMM\",1.000000,1000.000000,0.002000,2.000000,2,28.571429,3.000000,42.857143,66.666667\n"
      "\"Layer A\",\"BIAS\",1.000000,1000.000000,0.001000,1.000000,1,14.285714,3.000000,42.857143,33.333333\n"
      "\"Zero\",\"EMPTY\",0.000000,0.000000,0.000000,0.000000,0,0.000000,0.000000,0.000000,0.000000\n";
    testAssert(csv.str() == expected);
    string layer;
    {
      VulkanBenchmarkLayerScope trunk(&layer, "Trunk");
      testAssert(layer == "Trunk");
      {
        VulkanBenchmarkLayerScope projection(&layer, "QKV");
        testAssert(layer == "Trunk/QKV");
      }
      testAssert(layer == "Trunk");
      VulkanBenchmarkLayerScope disabled(nullptr, "Disabled");
      testAssert(layer == "Trunk");
    }
    testAssert(layer.empty());
    cout << "Vulkan layer/shader benchmark CSV tests passed" << endl;
  }
#endif
  testAssert(VulkanTuner::TUNER_VERSION == 38);
  {
    using Case = VulkanTuner::GemmTuneCase;
    using Kind = Case::Kind;
    VulkanTuner::ModelInfoForTuning info;
    // tf2-b10c384 Dense shapes, including non-tile-aligned inputs and final heads.
    info.gemmLayers = {
      {19, 384, 1.0, false, Kind::Dense}, {144, 48, 1.0, false, Kind::Dense},
      {144, 48, 1.0, false, Kind::Dense}, {48, 2, 1.0, false, Kind::Dense},
      {288, 96, 1.0, false, Kind::Dense}, {96, 3, 1.0, false, Kind::Dense},
      {96, 6, 1.0, false, Kind::Dense},
      {192, 48, 1.0}, {48, 2, 1.0},
      {22, 384, 1.0, false, Kind::Winograd3x3},
      {384, 384, 1.0, false, Kind::Winograd5x5}
    };
    info.transformerProjections = {{192, 576, true}, {192, 192, false}};
    VulkanTuneParams config;
    config.vulkan.shouldUseFP16Storage = true;
    config.vulkan.shouldUseFP16Compute = true;
    config.vulkan.shouldUseCooperativeMatrix = true;
    config.vulkan.shouldUseNHWC = true;
    const auto dense = VulkanTuner::getGemmTuningCases("xgemmDirect", info, config);
    testAssert(dense.size() == 6);
    testAssert(dense[0].inChannels == 19 && dense[0].outChannels == 384);
    testAssert(dense[1].weight == 2.0);
    testAssert(all_of(dense.begin(), dense.end(), [](const Case& c) { return c.kind == Kind::Dense; }));
    ModelDesc desc;
    MatMulLayerDesc* denseDescs[] = {
      &desc.trunk.initialMatMul, &desc.policyHead.gpoolToBiasMul, &desc.policyHead.gpoolToPassMul,
      &desc.policyHead.gpoolToPassMul2, &desc.valueHead.v2Mul, &desc.valueHead.v3Mul, &desc.valueHead.sv3Mul
    };
    for(size_t i = 0; i < 7; i++) {
      denseDescs[i]->inChannels = info.gemmLayers[i].inChannels;
      denseDescs[i]->outChannels = info.gemmLayers[i].outChannels;
    }
    desc.trunk.initialConv.convXSize = desc.trunk.initialConv.convYSize = 3;
    desc.trunk.initialConv.inChannels = 22;
    desc.trunk.initialConv.outChannels = 384;
    const auto extracted = VulkanTuner::ModelInfoForTuning::ofDesc(desc);
    const auto extractedDense = VulkanTuner::getGemmTuningCases("xgemmDirect", extracted, config);
    testAssert(extracted.gemmLayers.size() == 8 && extractedDense.size() == dense.size());
    for(size_t i = 0; i < dense.size(); i++) {
      testAssert(extractedDense[i].inChannels == dense[i].inChannels);
      testAssert(extractedDense[i].outChannels == dense[i].outChannels);
      testAssert(extractedDense[i].weight == dense[i].weight);
    }
    testAssert(VulkanTuner::getGemmTuningCases("xgemm", info, config).empty());
    testAssert(VulkanTuner::getGemmTuningCases("xgemm16", info, config).empty());
    testAssert(VulkanTuner::getGemmTuningCases("hgemmCooperativeMatrixNHWC", info, config).size() == 4);

    // NCHW CM eligibility uses its own selected tiles, not the generic Winograd tiles.
    info.transformerProjections.clear();
    config.vulkan.shouldUseNHWC = false;
    config.vulkan.shouldUseHgemmCooperativeMatrixNCHW = true;
    config.hgemmCooperativeMatrix.NWG = 64;
    config.hgemmCooperativeMatrixNCHW.NWG = 16;
    config.hgemmCooperativeMatrixNCHW.KWG = 16;
    const auto fallback = VulkanTuner::getGemmTuningCases("xgemmDirect", info, config);
    testAssert(fallback.size() == 7);
    testAssert(fallback.back().inChannels == 48 && fallback.back().outChannels == 2);
    testAssert(fallback.back().kind == Kind::Spatial);
    testAssert(VulkanTuner::getGemmTuningCases("hgemmCooperativeMatrixNCHW", info, config).size() == 2);
    const auto winograd = VulkanTuner::getGemmTuningCases("hgemmCooperativeMatrix", info, config);
    testAssert(winograd.size() == 2 && winograd[1].kind == Kind::Winograd5x5);
    config.vulkan.shouldUseHgemmCooperativeMatrixNCHW = false;
    testAssert(VulkanTuner::getGemmTuningCases("xgemmDirect", info, config).size() == 8);

    config.vulkan.shouldUseCooperativeMatrix = false;
    const auto fp32 = VulkanTuner::getGemmTuningCases("xgemm", info, config);
    const auto fp16 = VulkanTuner::getGemmTuningCases("xgemm16", info, config);
    testAssert(fp32.size() == 2 && fp16.size() == 2);
    testAssert(fp32[0].kind == Kind::Winograd3x3 && fp32[1].kind == Kind::Winograd5x5);
  }
  testAssert(VulkanTuner::TRANSFORMER_DUAL_GEMM_MIN_THROUGHPUT_RATIO == 1.10);
  testAssert(VulkanTuner::TRANSFORMER_DUAL_GEMM_ERROR_TOLERANCE == 0.002);
  VulkanTuner::ModelInfoForTuning dualModelInfo;
  dualModelInfo.trunkNumChannels = 3;
  dualModelInfo.midNumChannels = 2;
  const int dualInputChannels = VulkanTuner::getTransformerFFNInputChannelsForTuning(dualModelInfo);
  testAssert(dualInputChannels == 2 && dualInputChannels != dualModelInfo.trunkNumChannels);
  const vector<float> packedDualFilter = VulkanTuner::packTransformerDualGemmSwiGLUFilter(
    {2.0f, -1.0f}, {0.5f, 3.0f}, dualInputChannels, 1, 4
  );
  testAssert(packedDualFilter == vector<float>({2.0f, 0.5f, 0.0f, 0.0f, -1.0f, 3.0f, 0.0f, 0.0f}));
  testAssert(VulkanTuner::packTransformerDualGemmSwiGLUFilterForNHWC(
    {1,2,3,4,5,6}, {7,8,9,10,11,12}, 3, 2) == vector<float>({1,3,5,2,4,6,7,9,11,8,10,12}));
  testAssert(sizeof(vk_shader::spec::TransformerDualGemmSwiGLUSpec) == 12 * sizeof(uint32_t));
  testAssert(offsetof(vk_shader::spec::TransformerDualGemmSwiGLUSpec, STAGES) == 11 * sizeof(uint32_t));
  const vector<float> dualReference = VulkanTuner::computeTransformerDualGemmSwiGLUReference(
    {1.0f, 2.0f, 3.0f, 4.0f}, packedDualFilter, 1, 2, dualInputChannels, 4, 1
  );
  testAssert(dualReference.size() == 2);
  testAssert(fabs(dualReference[0] - (-2.555f)) < 0.003f);
  testAssert(dualReference[1] == 0.0f);
  testAssert(VulkanTuner::isFastEnough(110.0, 100.0, VulkanTuner::TRANSFORMER_DUAL_GEMM_MIN_THROUGHPUT_RATIO));
  testAssert(!VulkanTuner::isFastEnough(109.99, 100.0, VulkanTuner::TRANSFORMER_DUAL_GEMM_MIN_THROUGHPUT_RATIO));
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

  // A lucky short sample or an incomplete finalist cannot replace the incumbent.
  const vector<double> incumbentTimes(6, 100.0);
  testAssert(!VulkanTuner::isTransformerGemmUpgrade(incumbentTimes, {99,99,99,99,99,99}));
  testAssert(!VulkanTuner::isTransformerGemmUpgrade(incumbentTimes, {90,90,90,90,101,101}));
  testAssert(VulkanTuner::isTransformerGemmUpgrade(incumbentTimes, {95,95,95,95,95,101}));
  testAssert(!VulkanTuner::isTransformerGemmUpgrade(incumbentTimes, {95,95,95,95,95}));
  testAssert(!VulkanTuner::isTransformerGemmUpgrade(incumbentTimes, {95,95,95,95,95,nan}));
  testAssert(!VulkanTuner::isTransformerGemmUpgrade(incumbentTimes, {95,95,95,95,95,0}));

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
  testAssert(VulkanTuner::computeCooperativeMatrixTuningScore(300.0, 0.0001, 0.002) == 300.0);
  testAssert(VulkanTuner::computeCooperativeMatrixTuningScore(230.0, 0.0, 0.002) == 230.0);
  testAssert(
    VulkanTuner::computeCooperativeMatrixTuningScore(300.0, 0.0001, 0.002) >
    VulkanTuner::computeCooperativeMatrixTuningScore(230.0, 0.0, 0.002)
  );
  testAssert(VulkanTuner::computeCooperativeMatrixTuningScore(300.0, 0.002001, 0.002) == 0.0);
  testAssert(VulkanTuner::computeCooperativeMatrixTuningScore(300.0, nan, 0.002) == 0.0);
  // Optional paths use the adjustable VulkanTuner throughput thresholds.
  testAssert(VulkanTuner::isFastEnough(100.0, 100.0, 1.0));
  testAssert(!VulkanTuner::isFastEnough(99.999, 100.0, 1.0));
  for(double threshold: {
    VulkanTuner::FP16_COMPUTE_MIN_THROUGHPUT_RATIO,
    VulkanTuner::FP16_STORAGE_MIN_THROUGHPUT_RATIO,
    VulkanTuner::COOPERATIVE_MATRIX_MIN_THROUGHPUT_RATIO
  }) {
    testAssert(VulkanTuner::isFastEnough(100.0 * threshold, 100.0, threshold));
    testAssert(!VulkanTuner::isFastEnough(100.0 * threshold - 0.001, 100.0, threshold));
  }
  testAssert(VulkanTuner::FP16_COMPUTE_MIN_THROUGHPUT_RATIO == 0.85);
  testAssert(VulkanTuner::FP16_STORAGE_MIN_THROUGHPUT_RATIO == 1.20);
  testAssert(VulkanTuner::COOPERATIVE_MATRIX_MIN_THROUGHPUT_RATIO == 1.10);
  testAssert(VulkanTuner::COOPERATIVE_MATRIX_SHAPE_SCORE_RATIO == 0.90);
  testAssert(VulkanTuner::COOPERATIVE_MATRIX_MIN_SHAPES_PER_ACCUMULATOR == 3);
  testAssert(VulkanTuner::COOPERATIVE_MATRIX_NON_FULL_PROPERTY_LIMIT == 2);

  testAssert(VulkanTuner::defaultDirectory(false, "tests/scratch") == "tests/scratch/vulkantuning");
  testAssert(
    VulkanTuner::defaultFileName("Apple M2", 19, 19, 96, 8) ==
    "tune" + to_string(VulkanTuner::TUNER_VERSION) + "_gpuAppleM2_x19_y19_c96_mv8.txt"
  );

  MakeDir::make("tests/scratch");
  const string filename = "tests/scratch/vulkantuner-roundtrip.txt";
  const string defaultsFilename = "tests/scratch/vulkantuner-defaults.txt";
  FileUtils::tryRemoveFile(defaultsFilename);
  VulkanTuneParams separateProjectionProfile;
  separateProjectionProfile.transformerHgemmNHWC.NWG = 64;
  testAssert(separateProjectionProfile != VulkanTuneParams());
  VulkanTuneParams::save("tests/scratch/vulkantuner-transformer-profile.txt", separateProjectionProfile);
  testAssert(VulkanTuneParams::load("tests/scratch/vulkantuner-transformer-profile.txt") == separateProjectionProfile);
  VulkanTuneParams defaults;
  VulkanDeviceInfo deviceInfo = {};
  deviceInfo.storage16BitFeatures.storageBuffer16BitAccess = VK_TRUE;
  deviceInfo.shaderFloat16Int8Features.shaderFloat16 = VK_TRUE;
  deviceInfo.vulkanMemoryModelFeatures.vulkanMemoryModel = VK_TRUE;
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
  VulkanDeviceInfo noVulkanMemoryModelDeviceInfo = deviceInfo;
  noVulkanMemoryModelDeviceInfo.vulkanMemoryModelFeatures.vulkanMemoryModel = VK_FALSE;
  testAssert(!VulkanTuner::getHardwareParams(noVulkanMemoryModelDeviceInfo).canUseCooperativeMatrix);
  VulkanDeviceInfo incompleteSubgroupDeviceInfo = deviceInfo;
  incompleteSubgroupDeviceInfo.subgroupSizeControlFeatures.computeFullSubgroups = VK_FALSE;
  testAssert(!VulkanTuner::getHardwareParams(incompleteSubgroupDeviceInfo).canUseCooperativeMatrix);
  testAssert(!vk_shader::tune::isValidCooperativeMatrixConfig(
    incompleteSubgroupDeviceInfo, defaults.hgemmCooperativeMatrix
  ));
  VulkanDeviceInfo nonSubgroupDeviceInfo = deviceInfo;
  nonSubgroupDeviceInfo.cooperativeMatrixPropertiesFn = fakeNonSubgroupCooperativeMatrixProperties;
  testAssert(!VulkanTuner::getHardwareParams(nonSubgroupDeviceInfo).canUseCooperativeMatrix);
  VulkanDeviceInfo nonPowerOfTwoDeviceInfo = deviceInfo;
  nonPowerOfTwoDeviceInfo.cooperativeMatrixPropertiesFn = fakeNonPowerOfTwoCooperativeMatrixProperties;
  testAssert(!VulkanTuner::getHardwareParams(nonPowerOfTwoDeviceInfo).canUseCooperativeMatrix);
  VulkanDeviceInfo saturatingDeviceInfo = deviceInfo;
  saturatingDeviceInfo.cooperativeMatrixPropertiesFn = fakeSaturatingCooperativeMatrixProperties;
  testAssert(!VulkanTuner::getHardwareParams(saturatingDeviceInfo).canUseCooperativeMatrix);
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
  VulkanDeviceInfo accumulatorMemoryLimitedDeviceInfo = float32AccumulatorDevice.info;
  accumulatorMemoryLimitedDeviceInfo.properties.limits.maxComputeSharedMemorySize = 1039;
  testAssert(!isValidCooperativeMatrixConfig(
    accumulatorMemoryLimitedDeviceInfo, float32AccumulatorParams
  ));
  testAssert(!isValidCooperativeMatrixConfig(
    accumulatorMemoryLimitedDeviceInfo, float32AccumulatorNCHWParams
  ));
  accumulatorMemoryLimitedDeviceInfo.properties.limits.maxComputeSharedMemorySize = 1040;
  testAssert(isValidCooperativeMatrixConfig(
    accumulatorMemoryLimitedDeviceInfo, float32AccumulatorParams
  ));
  testAssert(isValidCooperativeMatrixConfig(
    accumulatorMemoryLimitedDeviceInfo, float32AccumulatorNCHWParams
  ));
  TransformerCooperativeMatrixTuneParams attentionParams;
  attentionParams.ATTN_BLOCK_KV = 64;
  testAssert(attentionParams.isValid());
  VulkanDeviceInfo attentionMemoryLimitedDeviceInfo = float32AccumulatorDevice.info;
  attentionMemoryLimitedDeviceInfo.properties.limits.maxComputeSharedMemorySize = 24415;
  testAssert(!isValidCooperativeMatrixConfig(attentionMemoryLimitedDeviceInfo, attentionParams, 64, 64));
  attentionMemoryLimitedDeviceInfo.properties.limits.maxComputeSharedMemorySize = 24416;
  testAssert(isValidCooperativeMatrixConfig(attentionMemoryLimitedDeviceInfo, attentionParams, 64, 64));
  attentionParams.ATTN_BLOCK_KV = 8;
  testAssert(!attentionParams.isValid());
  attentionParams.ATTN_BLOCK_KV = 48;
  testAssert(!attentionParams.isValid());
  testAssert(offsetof(vk_shader::spec::ScaleDotProductCooperativeSpec, COOP_USE_NV2) == 12 * sizeof(uint32_t));
  testAssert(sizeof(vk_shader::spec::ScaleDotProductCooperativeSpec) == 13 * sizeof(uint32_t));
  TransformerCooperativeMatrixTuneParams nv2Attention;
  nv2Attention.ATTN_BLOCK_Q = 32;
  nv2Attention.ATTN_BLOCK_KV = 64;
  nv2Attention.COOP_Q_TILES_PER_WORKGROUP = 2;
  VulkanDeviceInfo nv2DeviceInfo = float32AccumulatorDevice.info;
  nv2DeviceInfo.canUseRegisterAttentionNHWC = true;
  nv2DeviceInfo.cooperativeMatrix2Properties.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize = 128;
  nv2DeviceInfo.cooperativeMatrix2Properties.cooperativeMatrixFlexibleDimensionsMaxDimension = 1024;
  nv2DeviceInfo.cooperativeMatrix2Properties.cooperativeMatrixWorkgroupScopeReservedSharedMemory = 8192;
  VkCooperativeMatrixFlexibleDimensionsPropertiesNV nv2Property = {};
  nv2Property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
  nv2Property.MGranularity = 32;
  nv2Property.NGranularity = 16;
  nv2Property.KGranularity = 16;
  nv2Property.AType = nv2Property.BType = VK_COMPONENT_TYPE_FLOAT16_KHR;
  nv2Property.CType = nv2Property.ResultType = VK_COMPONENT_TYPE_FLOAT32_KHR;
  nv2Property.scope = VK_SCOPE_WORKGROUP_KHR;
  nv2Property.workgroupInvocations = 128;
  nv2DeviceInfo.flexibleMatrixProperties = {nv2Property};
  nv2DeviceInfo.properties.limits.maxComputeWorkGroupSize[0] = 128;
  nv2DeviceInfo.properties.limits.maxComputeWorkGroupInvocations = 128;
  testAssert(!canUseRegisterAttentionNHWC(nv2DeviceInfo, nv2Attention, 32, 32));
  testAssert(nv2Attention.getLocalSizeX() == 64);
  nv2Attention.COOP_USE_NV2 = 1;
  testAssert(nv2Attention.getLocalSizeX() == 128);
  testAssert(canUseRegisterAttentionNHWC(nv2DeviceInfo, nv2Attention, 32, 32));
  nv2DeviceInfo.properties.limits.maxComputeSharedMemorySize = 22975;
  testAssert(!isValidCooperativeMatrixConfig(nv2DeviceInfo, nv2Attention, 32, 32, true));
  nv2DeviceInfo.properties.limits.maxComputeSharedMemorySize = 22976;
  testAssert(isValidCooperativeMatrixConfig(nv2DeviceInfo, nv2Attention, 32, 32, true));
  testAssert(!isValidCooperativeMatrixConfig(nv2DeviceInfo, nv2Attention, 32, 32, false));
  TransformerCooperativeMatrixTuneParams nv2WrongPv = nv2Attention;
  nv2WrongPv.COOP_PV_N_SIZE = 8;
  testAssert(!canUseRegisterAttentionNHWC(nv2DeviceInfo, nv2WrongPv, 32, 32));
  testAssert(!isValidCooperativeMatrixConfig(nv2DeviceInfo, nv2WrongPv, 32, 32, true));
  VulkanDeviceInfo khrOnlyAttentionDevice = nv2DeviceInfo;
  khrOnlyAttentionDevice.canUseRegisterAttentionNHWC = false;
  khrOnlyAttentionDevice.flexibleMatrixProperties.clear();
  testAssert(!isValidCooperativeMatrixConfig(khrOnlyAttentionDevice, nv2Attention, 32, 32, true));
  TransformerCooperativeMatrixTuneParams khrAttention = nv2Attention;
  khrAttention.COOP_USE_NV2 = 0;
  testAssert(isValidCooperativeMatrixConfig(khrOnlyAttentionDevice, khrAttention, 32, 32, true));
  testAssert(khrAttention.getLocalSizeX() == 64);
  nv2DeviceInfo.properties.limits.maxComputeWorkGroupInvocations = 127;
  testAssert(!isValidCooperativeMatrixConfig(nv2DeviceInfo, nv2Attention, 32, 32, true));
  nv2DeviceInfo.flexibleMatrixProperties[0].workgroupInvocations = 64;
  testAssert(!canUseRegisterAttentionNHWC(nv2DeviceInfo, nv2Attention, 32, 32));
  nv2Attention.COOP_USE_NV2 = 2;
  testAssert(!nv2Attention.isValid());
  TransformerDualGemmSwiGLUTuneParams dualGemmFloat32Params;
  dualGemmFloat32Params.accType = 32;
  VulkanDeviceInfo dualGemmMemoryLimitedDeviceInfo = float32AccumulatorDevice.info;
  dualGemmMemoryLimitedDeviceInfo.properties.limits.maxComputeSharedMemorySize = 527;
  testAssert(!isValidCooperativeMatrixConfig(dualGemmMemoryLimitedDeviceInfo, dualGemmFloat32Params));
  dualGemmMemoryLimitedDeviceInfo.properties.limits.maxComputeSharedMemorySize = 528;
  testAssert(isValidCooperativeMatrixConfig(dualGemmMemoryLimitedDeviceInfo, dualGemmFloat32Params));
  dualGemmFloat32Params.SB = 1;
  dualGemmMemoryLimitedDeviceInfo.properties.limits.maxComputeSharedMemorySize = 1583;
  testAssert(!isValidCooperativeMatrixConfig(dualGemmMemoryLimitedDeviceInfo, dualGemmFloat32Params));
  dualGemmMemoryLimitedDeviceInfo.properties.limits.maxComputeSharedMemorySize = 1584;
  testAssert(isValidCooperativeMatrixConfig(dualGemmMemoryLimitedDeviceInfo, dualGemmFloat32Params));
  dualGemmFloat32Params.STAGES = 2;
  dualGemmMemoryLimitedDeviceInfo.properties.limits.maxComputeSharedMemorySize = 3119;
  testAssert(!isValidCooperativeMatrixConfig(dualGemmMemoryLimitedDeviceInfo, dualGemmFloat32Params));
  dualGemmMemoryLimitedDeviceInfo.properties.limits.maxComputeSharedMemorySize = 3120;
  testAssert(isValidCooperativeMatrixConfig(dualGemmMemoryLimitedDeviceInfo, dualGemmFloat32Params));
  dualGemmFloat32Params.STAGES = 3;
  testAssert(!dualGemmFloat32Params.isValid());
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
  VulkanDeviceInfo twoDimensionalWorkgroupDeviceInfo = deviceInfo;
  twoDimensionalWorkgroupDeviceInfo.properties.limits.maxComputeWorkGroupSize[0] = 64;
  testAssert(isValidCooperativeMatrixConfig(twoDimensionalWorkgroupDeviceInfo, multiSubgroupParams));
  twoDimensionalWorkgroupDeviceInfo.properties.limits.maxComputeWorkGroupSize[1] = 1;
  testAssert(!isValidCooperativeMatrixConfig(twoDimensionalWorkgroupDeviceInfo, multiSubgroupParams));
  HGemmCooperativeMatrixNCHWTuneParams multiSubgroupNCHWParams;
  multiSubgroupNCHWParams.MWG = 64;
  multiSubgroupNCHWParams.NWG = 64;
  multiSubgroupNCHWParams.KWG = 32;
  multiSubgroupNCHWParams.MWAVE = 32;
  multiSubgroupNCHWParams.NWAVE = 32;
  testAssert(multiSubgroupNCHWParams.isValid());
  testAssert(isValidCooperativeMatrixConfig(deviceInfo, multiSubgroupNCHWParams));
  HGemmCooperativeMatrixTuneParams reportedGenericParams;
  reportedGenericParams.MWG = 16;
  reportedGenericParams.NWG = 32;
  reportedGenericParams.KWG = 16;
  reportedGenericParams.MWAVE = 16;
  reportedGenericParams.NWAVE = 16;
  reportedGenericParams.MWARP = 16;
  reportedGenericParams.NWARP = 8;
  reportedGenericParams.VWM = 2;
  reportedGenericParams.VWN = 1;
  reportedGenericParams.KDIM = 16;
  reportedGenericParams.subgroupSize = 32;
  reportedGenericParams.accType = 16;
  reportedGenericParams.SA = 0;
  reportedGenericParams.SB = 0;
  testAssert(reportedGenericParams.isValid());
  testAssert(isValidCooperativeMatrixConfig(
    sixteenByEightDevice.info, reportedGenericParams
  ));
  {
    const int M = 32;
    const int N = 64;
    vector<int> coverage(M * N, 0);
    const int subgroupCountM = reportedGenericParams.MWAVE / reportedGenericParams.MWARP;
    const int subgroupCountN = reportedGenericParams.NWAVE / reportedGenericParams.NWARP;
    for(int groupM = 0; groupM < M / reportedGenericParams.MWG; groupM++) {
      for(int groupN = 0; groupN < N / reportedGenericParams.NWG; groupN++) {
        for(int subgroup = 0; subgroup < subgroupCountM * subgroupCountN; subgroup++) {
          const int subgroupM = subgroup % subgroupCountM;
          const int subgroupN = subgroup / subgroupCountM;
          for(int bWave = 0; bWave < reportedGenericParams.NWG / reportedGenericParams.NWAVE; bWave++) {
            const int bOffset = bWave * reportedGenericParams.NWAVE + subgroupN * reportedGenericParams.NWARP;
            for(int aWave = 0; aWave < reportedGenericParams.MWG / reportedGenericParams.MWAVE; aWave++) {
              const int aOffset = aWave * reportedGenericParams.MWAVE + subgroupM * reportedGenericParams.MWARP;
              for(int n = 0; n < reportedGenericParams.NWARP; n++)
                for(int m = 0; m < reportedGenericParams.MWARP; m++)
                  coverage[(groupN * reportedGenericParams.NWG + bOffset + n) * M +
                           groupM * reportedGenericParams.MWG + aOffset + m]++;
            }
          }
        }
      }
    }
    testAssert(all_of(coverage.begin(), coverage.end(), [](int count) { return count == 1; }));
  }
  HGemmCooperativeMatrixNCHWTuneParams reportedNCHWParams;
  reportedNCHWParams.MWG = 16;
  reportedNCHWParams.NWG = 64;
  reportedNCHWParams.KWG = 32;
  reportedNCHWParams.MWAVE = 16;
  reportedNCHWParams.NWAVE = 32;
  reportedNCHWParams.MWARP = 16;
  reportedNCHWParams.NWARP = 16;
  reportedNCHWParams.VWM = 2;
  reportedNCHWParams.VWN = 4;
  reportedNCHWParams.KDIM = 16;
  reportedNCHWParams.subgroupSize = 32;
  reportedNCHWParams.accType = 16;
  reportedNCHWParams.SB = 0;
  testAssert(reportedNCHWParams.isValid());
  testAssert(reportedNCHWParams.getRequiredSpatialAlignment() == 16);
  testAssert(isValidCooperativeMatrixConfig(deviceInfo, reportedNCHWParams));
  {
    const int logicalNNXYLen = 19 * 19;
    const int alignment = reportedNCHWParams.getRequiredSpatialAlignment();
    const int paddedNNXYLen = ((logicalNNXYLen + alignment - 1) / alignment) * alignment;
    testAssert(paddedNNXYLen == 368);
    vector<int> coverage(reportedNCHWParams.NWG * paddedNNXYLen, 0);
    const int subgroupCountM = reportedNCHWParams.MWAVE / reportedNCHWParams.MWARP;
    const int subgroupCountN = reportedNCHWParams.NWAVE / reportedNCHWParams.NWARP;
    for(int groupM = 0; groupM < paddedNNXYLen / reportedNCHWParams.MWG; groupM++) {
      for(int subgroup = 0; subgroup < subgroupCountM * subgroupCountN; subgroup++) {
        const int subgroupM = subgroup % subgroupCountM;
        const int subgroupN = subgroup / subgroupCountM;
        for(int bWave = 0; bWave < reportedNCHWParams.NWG / reportedNCHWParams.NWAVE; bWave++) {
          const int bOffset = bWave * reportedNCHWParams.NWAVE + subgroupN * reportedNCHWParams.NWARP;
          for(int aWave = 0; aWave < reportedNCHWParams.MWG / reportedNCHWParams.MWAVE; aWave++) {
            const int aOffset = aWave * reportedNCHWParams.MWAVE + subgroupM * reportedNCHWParams.MWARP;
            for(int n = 0; n < reportedNCHWParams.NWARP; n++)
              for(int m = 0; m < reportedNCHWParams.MWARP; m++)
                coverage[(bOffset + n) * paddedNNXYLen +
                         groupM * reportedNCHWParams.MWG + aOffset + m]++;
          }
        }
      }
    }
    testAssert(all_of(coverage.begin(), coverage.end(), [](int count) { return count == 1; }));
  }
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
  testAssert(defaults.transformerDualGemmSwiGLU.VWM == 1);
  testAssert(defaults.transformerDualGemmSwiGLU.VWN == 1);
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
  const string dualGemmDeviceLimitedFilename = "tests/scratch/vulkantuner-dual-gemm-device-limited.txt";
  VulkanTuneParams dualGemmDeviceLimitedConfig = defaults;
  dualGemmDeviceLimitedConfig.vulkan.canUseFP16Storage = true;
  dualGemmDeviceLimitedConfig.vulkan.canUseFP16Compute = true;
  dualGemmDeviceLimitedConfig.vulkan.canUseCooperativeMatrix = true;
  dualGemmDeviceLimitedConfig.vulkan.shouldUseFP16Storage = true;
  dualGemmDeviceLimitedConfig.vulkan.shouldUseFP16Compute = true;
  dualGemmDeviceLimitedConfig.vulkan.shouldUseTransformerDualGemmSwiGLU = true;
  dualGemmDeviceLimitedConfig.transformerDualGemmSwiGLU.accType = 32;
  VulkanTuneParams::save(dualGemmDeviceLimitedFilename, dualGemmDeviceLimitedConfig);
  VulkanDeviceInfo dualGemmLimitedDeviceInfo = deviceInfo;
  dualGemmLimitedDeviceInfo.properties.limits.maxComputeSharedMemorySize = 2079;
  const VulkanTuneParams revalidatedDualGemmConfig = VulkanTuner::loadOrCreate(
    dualGemmDeviceLimitedFilename, "", "", 19, 19, modelInfo, dualGemmLimitedDeviceInfo, nullptr
  );
  testAssert(!revalidatedDualGemmConfig.vulkan.shouldUseTransformerDualGemmSwiGLU);
  vector<string> defaultLines = FileUtils::readFileLines(defaultsFilename, '\n');
  testAssert(defaultLines[0] == "VERSION=" + to_string(VulkanTuner::TUNER_VERSION));
  testAssert(defaultLines[1] == "vulkan.canUseFP16Storage=1");
  testAssert(defaultLines[2] == "vulkan.canUseFP16Compute=1");
  testAssert(defaultLines[3] == "vulkan.canUseCooperativeMatrix=1");
  testAssert(defaultLines[4] == "vulkan.canUseSubgroup=1");
  testAssert(defaultLines[5] == "vulkan.shouldUseFP16Storage=0");
  testAssert(defaultLines[6] == "vulkan.shouldUseFP16Compute=0");
  testAssert(defaultLines[7] == "vulkan.shouldUseCooperativeMatrix=0");
  testAssert(defaultLines[8] == "vulkan.shouldUseNHWC=0");
  testAssert(defaultLines[9] == "vulkan.shouldUseHgemmCooperativeMatrixNCHW=0");
  testAssert(defaultLines[10] == "vulkan.shouldUseSubgroup=0");
  testAssert(defaultLines[11] == "vulkan.shouldUseTransformerDualGemmSwiGLU=0");
  testAssert(defaultLines.size() == 181);
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
  testAssert(lineIndex("spatialRMSNorm.APPLY_ELTS_PER_THREAD=") < lineIndex("transformerDualGemmSwiGLU.MWG="));
  testAssert(lineIndex("transformerDualGemmSwiGLU.VWM=") < lineIndex("transformerDualGemmSwiGLU.SB="));
  testAssert(VulkanTuneParams::load(defaultsFilename) == defaults);
  testAssert(sizeof(vk_shader::spec::TransformerRMSNormSpec) == 8 * sizeof(uint32_t));
  testAssert(offsetof(vk_shader::spec::TransformerRMSNormSpec, MODE) == 7 * sizeof(uint32_t));
  VulkanDeviceInfo scalarDevice = {};
  TransformerRMSNormTuneParms subgroupNorm{32,4,8,2};
  testAssert(!canUseTransformerRMSNormSubgroup(scalarDevice, subgroupNorm));
  scalarDevice.subgroupProperties.subgroupSize = 32;
  scalarDevice.subgroupSizeProperties.maxComputeWorkgroupSubgroups = 32;
  scalarDevice.properties.limits.maxComputeWorkGroupSize[0] = 1024;
  scalarDevice.properties.limits.maxComputeWorkGroupInvocations = 1024;
  scalarDevice.properties.limits.maxComputeSharedMemorySize = 32768;
  scalarDevice.subgroupProperties.supportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
  scalarDevice.subgroupProperties.supportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
  scalarDevice.subgroupSizeControlFeatures.computeFullSubgroups = VK_TRUE;
  // No cooperative-matrix/TensorCore capability is needed for FP32 subgroup sums.
  testAssert(scalarDevice.cooperativeMatrixFeatures.cooperativeMatrix == VK_FALSE);
  testAssert(canUseTransformerRMSNormSubgroup(scalarDevice, subgroupNorm));
  scalarDevice.subgroupProperties.subgroupSize = 64;
  testAssert(!canUseTransformerRMSNormSubgroup(scalarDevice, subgroupNorm));
  subgroupNorm.WG_C_SIZE = 64;
  testAssert(canUseTransformerRMSNormSubgroup(scalarDevice, subgroupNorm));
  scalarDevice.subgroupSizeProperties.maxComputeWorkgroupSubgroups = 2;
  testAssert(!canUseTransformerRMSNormSubgroup(scalarDevice, subgroupNorm));
  scalarDevice.subgroupSizeProperties.maxComputeWorkgroupSubgroups = 32;
  scalarDevice.subgroupSizeControlFeatures.computeFullSubgroups = VK_FALSE;
  testAssert(!canUseTransformerRMSNormSubgroup(scalarDevice, subgroupNorm));
  scalarDevice.subgroupSizeControlFeatures.computeFullSubgroups = VK_TRUE;
  scalarDevice.subgroupProperties.supportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT;
  testAssert(!canUseTransformerRMSNormSubgroup(scalarDevice, subgroupNorm));
  string legacyPreNorm;
  for(const string& line: defaultLines)
    if(line.find("transformerPreNorm.") != 0)
      legacyPreNorm += line.find("VERSION=") == 0 ? "VERSION=37\n" : line + "\n";
  writeText("tests/scratch/vulkantuner-legacy-prenorm.txt", legacyPreNorm);
  testAssert(VulkanTuneParams::load("tests/scratch/vulkantuner-legacy-prenorm.txt") == defaults);
  VulkanTuneParams cachedNorm = defaults;
  cachedNorm.transformerPreNorm = {32,4,8,2};
  VulkanTuneParams::save("tests/scratch/vulkantuner-cached-prenorm.txt", cachedNorm);
  testAssert(VulkanTuneParams::load("tests/scratch/vulkantuner-cached-prenorm.txt") == cachedNorm);
  testAssert(cachedNorm != defaults);
  string legacyFFNText;
  for(const string& line: defaultLines)
    if(line.find("transformerDualGemmSwiGLU.STAGES=") != 0 && line.find("transformerPreNorm.") != 0)
      legacyFFNText += line.find("VERSION=") == 0 ? "VERSION=36\n" : line + "\n";
  writeText("tests/scratch/vulkantuner-legacy-ffn.txt", legacyFFNText);
  testAssert(VulkanTuneParams::load("tests/scratch/vulkantuner-legacy-ffn.txt") == defaults);
  VulkanTuneParams twoStageFFN = defaults;
  twoStageFFN.transformerDualGemmSwiGLU.STAGES = 2;
  testAssert(twoStageFFN != defaults);
  VulkanTuneParams::save("tests/scratch/vulkantuner-two-stage-ffn.txt", twoStageFFN);
  testAssert(VulkanTuneParams::load("tests/scratch/vulkantuner-two-stage-ffn.txt") == twoStageFFN);
  string legacyAttentionText;
  for(const string& line: defaultLines)
    if(line.find(".COOP_USE_NV2=") == string::npos)
      legacyAttentionText += line + "\n";
  const string legacyAttentionFilename = "tests/scratch/vulkantuner-legacy-attention.txt";
  writeText(legacyAttentionFilename, legacyAttentionText);
  testAssert(VulkanTuneParams::load(legacyAttentionFilename) == defaults);
  VulkanTuneParams nv2Roundtrip = defaults;
  nv2Roundtrip.transformerCooperativeMatrixNHWC.COOP_USE_NV2 = 1;
  testAssert(!(nv2Roundtrip == defaults));
  const string nv2RoundtripFilename = "tests/scratch/vulkantuner-nv2-attention.txt";
  VulkanTuneParams::save(nv2RoundtripFilename, nv2Roundtrip);
  testAssert(VulkanTuneParams::load(nv2RoundtripFilename) == nv2Roundtrip);
  nv2Roundtrip.transformerCooperativeMatrixNCHW.COOP_USE_NV2 = 1;
  testAssert(!nv2Roundtrip.isValid());
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
  params.transformerDualGemmSwiGLU.NWG = 32;
  params.transformerDualGemmSwiGLU.KWG = 32;
  params.transformerDualGemmSwiGLU.accType = 32;
  params.vulkan.canUseFP16Storage = true;
  params.vulkan.canUseFP16Compute = true;
  params.vulkan.canUseCooperativeMatrix = true;
  params.vulkan.shouldUseFP16Storage = true;
  params.vulkan.shouldUseFP16Compute = true;
  params.vulkan.shouldUseCooperativeMatrix = true;
  params.vulkan.shouldUseHgemmCooperativeMatrixNCHW = true;
  params.vulkan.shouldUseTransformerDualGemmSwiGLU = true;
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
  testAssert(loaded.transformerDualGemmSwiGLU.accType == 32);

  writeText(filename, "VERSION=999\n");
  testAssert(loadThrows(filename));
  writeText(filename, "VERSION=22\n");
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
  invalid.transformerDualGemmSwiGLU.VWM = 2;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.vulkan.shouldUseTransformerDualGemmSwiGLU = true;
  invalid.vulkan.shouldUseFP16Compute = false;
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
