#ifdef USE_VULKAN_BACKEND

#include "../tests/tests.h"

#include <fstream>

#include "../core/fileutils.h"
#include "../core/makedir.h"
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
  testAssert(VulkanTuneParams::load(defaultsFilename) == defaults);

  VulkanTuneParams params;
  params.conv3x3.inputTransformLocalXSize = 64;
  params.conv3x3.inputTransformLocalYSize = 4;
  params.conv5x5.outputTransformLocalXSize = 16;
  params.xgemm.KWG = 32;
  params.xgemmDirect.KWID = 1;
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
  params.vulkan.canUseFP16Storage = true;
  params.vulkan.canUseFP16Compute = true;
  params.vulkan.shouldUseFP16Storage = true;
  params.vulkan.shouldUseFP16Compute = true;
  params.vulkan.shouldUseHgemmCooperativeMatrixNCHW = true;
  params.vulkan.canUseSubgroup = true;
  params.vulkan.shouldUseSubgroup = true;
  testAssert(params.isValid());
  VulkanTuneParams::save(filename, params);
  testAssert(VulkanTuneParams::load(filename) == params);

  writeText(filename, "VERSION=999\n");
  testAssert(loadThrows(filename));
  writeText(filename, "VERSION=" + to_string(VulkanTuner::TUNER_VERSION) + "\nvulkan.canUseFP16Storage=not-an-int\n");
  testAssert(loadThrows(filename));

  VulkanTuneParams invalid = params;
  invalid.xgemm.MDIMC = 0;
  testAssert(!invalid.isValid());
  invalid = params;
  invalid.xgemmDirect.MDIMAD = 16;
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
