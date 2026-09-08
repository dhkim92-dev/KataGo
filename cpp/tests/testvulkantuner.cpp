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
  defaults.vulkan.canUseFP16Storage = true;
  defaults.vulkan.canUseFP16Compute = true;
  defaults.vulkan.canUseCooperativeMatrix = true;
  defaults.vulkan.canUseSubgroup = true;
  testAssert(defaults.p32s32.xgemmDirect.WGD == 8);
  testAssert(defaults.p32s32.xgemmDirect.MDIMCD == 1);
  testAssert(defaults.p32s32.xgemmDirect.NDIMCD == 1);
  testAssert(defaults.p32s32.xgemmDirect.MDIMAD == 1);
  testAssert(defaults.p32s32.xgemmDirect.NDIMBD == 1);
  testAssert(defaults.p32s32.xgemmDirect.KWID == 2);
  testAssert(defaults.p32s32.xgemmDirect.PADA == 1);
  testAssert(defaults.p32s32.xgemmDirect.PADB == 1);
  testAssert(defaults.p32s32.spatialRMSNorm.TILE_SIZE == 32);
  testAssert(defaults.p32s32.spatialRMSNorm.APPLY_ELTS_PER_THREAD == 1);
  testAssert(defaults.p32s32.rmsNorm.WG_C_SIZE == 64);
  testAssert(defaults.p32s32.rmsNorm.WG_XY_SIZE == 1);
  testAssert(defaults.p32s32.rmsNorm.C_PER_THREAD == 4);
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
  const auto lineIndex = [&](const string& prefix) {
    for(size_t i = 0; i < defaultLines.size(); i++) {
      if(defaultLines[i].find(prefix) == 0)
        return i;
    }
    return defaultLines.size();
  };
  testAssert(lineIndex("p32s32.xgemmDirect.WGD=") < lineIndex("p32s32.xgemm.MWG="));
  testAssert(lineIndex("p32s32.xgemm.MWG=") < lineIndex("p32s32.xgemm16.MWG="));
  testAssert(lineIndex("p32s32.xgemm16.MWG=") < lineIndex("p32s32.hgemmCooperativeMatrix.MWG="));
  testAssert(lineIndex("p32s32.hgemmCooperativeMatrix.MWG=") < lineIndex("p32s32.hgemmCooperativeMatrixNCHW.MWG="));
  testAssert(lineIndex("p32s32.hgemmCooperativeMatrixNCHW.MWG=") < lineIndex("p32s32.conv3x3.inTileXSize="));
  testAssert(lineIndex("p32s32.conv5x5.inTileXSize=") < lineIndex("p32s32.gPool.XYSTRIDE="));
  testAssert(lineIndex("p32s32.gPool.XYSTRIDE=") < lineIndex("p32s32.transformer.ATTN_BLOCK_Q="));
  testAssert(lineIndex("p32s32.transformer.ATTN_BLOCK_Q=") < lineIndex("p32s32.rmsNorm.WG_C_SIZE="));
  testAssert(lineIndex("p32s32.rmsNorm.WG_C_SIZE=") < lineIndex("p32s32.pointwise.ELTS_PER_THREAD="));
  testAssert(lineIndex("p32s32.pointwise.ELTS_PER_THREAD=") < lineIndex("p32s32.addChannelBiases.XY_ELTS_PER_THREAD="));
  testAssert(lineIndex("p32s32.addChannelBiases.XY_ELTS_PER_THREAD=") < lineIndex("p32s32.spatialRMSNorm.TILE_SIZE="));
  testAssert(lineIndex("p32s32.spatialRMSNorm.TILE_SIZE=") < lineIndex("p32s16.xgemmDirect.WGD="));
  testAssert(lineIndex("p32s16.spatialRMSNorm.TILE_SIZE=") < lineIndex("p16s16.xgemmDirect.WGD="));
  testAssert(VulkanTuneParams::load(defaultsFilename) == defaults);

  VulkanTuneParams params;
  auto& p32s32 = params.p32s32;
  p32s32.conv3x3.inTileXSize = 4;
  p32s32.conv3x3.inTileYSize = 4;
  p32s32.conv3x3.outTileXSize = 2;
  p32s32.conv3x3.outTileYSize = 2;
  p32s32.conv3x3.inputTransformLocalXSize = 64;
  p32s32.conv3x3.inputTransformLocalYSize = 4;
  p32s32.conv5x5.outputTransformLocalXSize = 16;
  p32s32.xgemm.KWG = 32;
  p32s32.addChannelBiases.XY_ELTS_PER_THREAD = 2;
  p32s32.addChannelBiases.NC_ELTS_PER_THREAD = 8;
  p32s32.pointwise.LOCAL_SIZE = 128;
  p32s32.pointwise.ELTS_PER_THREAD = 2;
  p32s32.gPool.XYSTRIDE = 16;
  p32s32.gPool.CHANNELSTRIDE = 2;
  p32s32.gPool.BATCHSTRIDE = 2;
  p32s32.transformer.ATTN_BLOCK_Q = 64;
  p32s32.transformer.ATTN_BLOCK_KV = 16;
  p32s32.transformer.Q_PER_THREAD = 2;
  p32s32.rmsNorm.WG_C_SIZE = 64;
  p32s32.rmsNorm.WG_XY_SIZE = 4;
  p32s32.rmsNorm.C_PER_THREAD = 2;
  p32s32.spatialRMSNorm.TILE_SIZE = 64;
  p32s32.spatialRMSNorm.APPLY_ELTS_PER_THREAD = 4;
  params.p32s16.xgemmDirect.KWID = 1;
  params.p32s16.xgemm.MWG = 64;
  params.p16s16.xgemm16.MWG = 64;
  params.p16s16.xgemm16.NWG = 64;
  params.vulkan.canUseFP16Storage = true;
  params.vulkan.canUseFP16Compute = true;
  params.vulkan.shouldUseFP16Storage = true;
  params.vulkan.shouldUseFP16Compute = true;
  params.vulkan.shouldUseHgemmCooperativeMatrixNCHW = true;
  params.vulkan.canUseSubgroup = true;
  params.vulkan.shouldUseSubgroup = true;
  params.activateProfile(PrecisionProfile::P16S16);
  testAssert(params.xgemm16.MWG == 64);
  params.activateProfile(PrecisionProfile::P32S16);
  testAssert(params.xgemmDirect.KWID == 1);
  params.activateProfile(PrecisionProfile::P16S16);
  testAssert(params.isValid());
  VulkanTuneParams::save(filename, params);
  VulkanTuneParams loaded = VulkanTuneParams::load(filename);
  testAssert(loaded == params);
  testAssert(loaded.configuredProfile() == PrecisionProfile::P16S16);
  testAssert(loaded.xgemm16.MWG == 64);
  loaded.activateProfile(PrecisionProfile::P32S16);
  testAssert(loaded.xgemmDirect.KWID == 1);

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
  invalid.xgemmDirect.MDIMCD = 4;
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
