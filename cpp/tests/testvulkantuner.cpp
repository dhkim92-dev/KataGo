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
  testAssert(VulkanTuner::defaultFileName("Apple M2", 19, 19, 96, 8) == "tune1_gpuAppleM2_x19_y19_c96_mv8.txt");

  MakeDir::make("tests/scratch");
  const string filename = "tests/scratch/vulkantuner-roundtrip.txt";
  VulkanTuneParams params;
  params.conv3x3.inputTransformLocalXSize = 64;
  params.conv3x3.inputTransformLocalYSize = 4;
  params.conv5x5.outputTransformLocalXSize = 16;
  params.xgemm.KWG = 32;
  params.xgemmDirect.KWID = 1;
  testAssert(params.isValid());
  VulkanTuneParams::save(filename, params);
  testAssert(VulkanTuneParams::load(filename) == params);

  writeText(filename, "VERSION=2\n");
  testAssert(loadThrows(filename));
  writeText(filename, "VERSION=1\nxgemm.MDIMC=not-an-int\n");
  testAssert(loadThrows(filename));

  VulkanTuneParams invalid = params;
  invalid.xgemm.MDIMC = 0;
  testAssert(!invalid.isValid());
  bool saveThrew = false;
  try {
    VulkanTuneParams::save(filename, invalid);
  } catch(const StringError&) {
    saveThrew = true;
  }
  testAssert(saveThrew);

  FileUtils::tryRemoveFile(filename);
  cout << "Vulkan tuner persistence tests passed" << endl;
}

#endif
