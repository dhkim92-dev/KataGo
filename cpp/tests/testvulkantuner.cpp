#ifdef USE_VULKAN_BACKEND

#include "../tests/tests.h"

#include <cmath>
#include <fstream>
#include <limits>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../external/half-2.2.0/include/half.hpp"
#include "../neuralnet/vulkantuner.h"

using namespace std;

namespace {
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
  testAssert(VulkanTuner::shouldUseFP16ForModel(1.0, 0.5, 0.01));
  testAssert(!VulkanTuner::shouldUseFP16ForModel(1.0, 1.0, 0.01));
  testAssert(!VulkanTuner::shouldUseFP16ForModel(1.0, 0.5, 1.0));
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
  deviceInfo.subgroupProperties.supportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
  deviceInfo.subgroupSizeControlFeatures.computeFullSubgroups = VK_TRUE;
  const VulkanParams hardwareParams = VulkanTuner::getHardwareParams(deviceInfo);
  testAssert(hardwareParams.canUseFP16Storage);
  testAssert(hardwareParams.canUseFP16Compute);
  testAssert(hardwareParams.canUseCooperativeMatrix);
  testAssert(hardwareParams.canUseSubgroup);
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
  params.hgemmCooperativeMatrixNCHW.VWM = 1;
  params.hgemmCooperativeMatrixNCHW.VWN = 2;
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
  VulkanTuneParams::save(filename, params);
  VulkanTuneParams loaded = VulkanTuneParams::load(filename);
  testAssert(loaded == params);
  testAssert(loaded.xgemm16.MWG == 64);

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
