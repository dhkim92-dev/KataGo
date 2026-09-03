#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"

#include <fstream>
#include <map>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../dataio/homedata.h"

using namespace std;
using namespace vk_shader;
using namespace vk_shader::tune;

namespace {
  const char* VERSION_LINE = "VERSION=3";

  bool isMultipleOf(uint64_t x, uint64_t y) {
    return y != 0 && x % y == 0;
  }

  void writeParam(ofstream& out, const char* name, uint32_t value) {
    out << name << "=" << value << "\n";
  }

  uint32_t getParam(const map<string, string>& values, const string& name, const string& filename) {
    auto iter = values.find(name);
    if(iter == values.end())
      throw IOError("VulkanTuneParams::load: missing parameter " + name + " in " + filename);
    int value;
    if(!Global::tryStringToInt(iter->second, value) || value < 0)
      throw IOError("VulkanTuneParams::load: invalid integer for " + name + " in " + filename);
    return static_cast<uint32_t>(value);
  }

  bool getBoolParam(const map<string, string>& values, const string& name, const string& filename) {
    uint32_t value = getParam(values, name, filename);
    if(value > 1)
      throw IOError("VulkanTuneParams::load: invalid boolean for " + name + " in " + filename);
    return value == 1;
  }

  VulkanParams makeVulkanParams(const VulkanDeviceInfo& deviceInfo) {
    VulkanParams params;
    params.canUseFP16Storage =
      deviceInfo.storage16BitFeatures.storageBuffer16BitAccess == VK_TRUE ||
      deviceInfo.storage16BitFeatures.uniformAndStorageBuffer16BitAccess == VK_TRUE;
    params.canUseFP16Compute = deviceInfo.shaderFloat16Int8Features.shaderFloat16 == VK_TRUE;
    params.canUseCooperativMatrix = deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE;
    params.canUseSubgroup =
      (deviceInfo.subgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
      deviceInfo.subgroupSizeControlFeatures.computeFullSubgroups == VK_TRUE;
    return params;
  }

}  // namespace

bool VulkanTuneParams::isValid() const {
  const auto validConv = [](const ConvTuneParams& p, uint32_t outTileSize) {
    if(p.inTileXSize != 6 || p.inTileYSize != 6 || p.outTileXSize != outTileSize || p.outTileYSize != outTileSize)
      return false;
    if(p.inputTransformLocalXSize == 0 || p.inputTransformLocalYSize == 0)
      return false;
    if(p.outputTransformLocalXSize == 0 || p.outputTransformLocalYSize == 0 || p.outputTransformLocalZSize == 0)
      return false;
    if(static_cast<uint64_t>(p.inputTransformLocalXSize) * p.inputTransformLocalYSize > 1024)
      return false;
    if(
      static_cast<uint64_t>(p.outputTransformLocalXSize) * p.outputTransformLocalYSize * p.outputTransformLocalZSize >
      1024)
      return false;
    return true;
  };
  if(!validConv(conv3x3, 4) || !validConv(conv5x5, 2))
    return false;

  if(
    xgemm.MDIMC == 0 || xgemm.NDIMC == 0 || xgemm.MWG == 0 || xgemm.NWG == 0 || xgemm.KWG == 0 || xgemm.MDIMA == 0 ||
    xgemm.NDIMB == 0)
    return false;
  const uint64_t xgemmWorkgroupSize = static_cast<uint64_t>(xgemm.MDIMC) * xgemm.NDIMC;
  if(xgemmWorkgroupSize > 1024)
    return false;
  if(
    !isMultipleOf(xgemm.MWG, static_cast<uint64_t>(xgemm.MDIMC) * 4) ||
    !isMultipleOf(xgemm.NWG, static_cast<uint64_t>(xgemm.NDIMC) * 4))
    return false;
  if(
    !isMultipleOf(xgemm.MWG, static_cast<uint64_t>(xgemm.MDIMA) * 4) ||
    !isMultipleOf(xgemm.NWG, static_cast<uint64_t>(xgemm.NDIMB) * 4))
    return false;
  if(!isMultipleOf(xgemm.KWG, 4))
    return false;
  if(!isMultipleOf(xgemm.KWG, xgemmWorkgroupSize / xgemm.MDIMA))
    return false;
  if(!isMultipleOf(xgemm.KWG, xgemmWorkgroupSize / xgemm.NDIMB))
    return false;

  if(
    xgemmDirect.WGD == 0 || xgemmDirect.MDIMCD == 0 || xgemmDirect.NDIMCD == 0 || xgemmDirect.MDIMAD == 0 ||
    xgemmDirect.NDIMBD == 0 || xgemmDirect.KWID == 0)
    return false;
  const uint64_t directWorkgroupSize = static_cast<uint64_t>(xgemmDirect.MDIMCD) * xgemmDirect.NDIMCD;
  if(directWorkgroupSize > 1024)
    return false;
  if(xgemmDirect.PADA > 1 || xgemmDirect.PADB > 1)
    return false;
  if(!isMultipleOf(xgemmDirect.WGD, xgemmDirect.KWID))
    return false;
  return true;
}

bool VulkanTuneParams::operator==(const VulkanTuneParams& other) const {
  return vulkan.canUseFP16Storage == other.vulkan.canUseFP16Storage &&
         vulkan.canUseFP16Compute == other.vulkan.canUseFP16Compute &&
         vulkan.canUseCooperativMatrix == other.vulkan.canUseCooperativMatrix &&
         vulkan.canUseSubgroup == other.vulkan.canUseSubgroup &&
         vulkan.shouldUseFP16Storage == other.vulkan.shouldUseFP16Storage &&
         vulkan.shouldUseFP16Compute == other.vulkan.shouldUseFP16Compute &&
         vulkan.shouldUseCooperativeMatrix == other.vulkan.shouldUseCooperativeMatrix &&
         vulkan.shouldUseSubgroup == other.vulkan.shouldUseSubgroup &&
         conv3x3.inTileYSize == other.conv3x3.inTileYSize && conv3x3.inTileXSize == other.conv3x3.inTileXSize &&
         conv3x3.outTileYSize == other.conv3x3.outTileYSize && conv3x3.outTileXSize == other.conv3x3.outTileXSize &&
         conv3x3.inputTransformLocalXSize == other.conv3x3.inputTransformLocalXSize &&
         conv3x3.inputTransformLocalYSize == other.conv3x3.inputTransformLocalYSize &&
         conv3x3.outputTransformLocalXSize == other.conv3x3.outputTransformLocalXSize &&
         conv3x3.outputTransformLocalYSize == other.conv3x3.outputTransformLocalYSize &&
         conv3x3.outputTransformLocalZSize == other.conv3x3.outputTransformLocalZSize &&
         conv5x5.inTileYSize == other.conv5x5.inTileYSize && conv5x5.inTileXSize == other.conv5x5.inTileXSize &&
         conv5x5.outTileYSize == other.conv5x5.outTileYSize && conv5x5.outTileXSize == other.conv5x5.outTileXSize &&
         conv5x5.inputTransformLocalXSize == other.conv5x5.inputTransformLocalXSize &&
         conv5x5.inputTransformLocalYSize == other.conv5x5.inputTransformLocalYSize &&
         conv5x5.outputTransformLocalXSize == other.conv5x5.outputTransformLocalXSize &&
         conv5x5.outputTransformLocalYSize == other.conv5x5.outputTransformLocalYSize &&
         conv5x5.outputTransformLocalZSize == other.conv5x5.outputTransformLocalZSize &&
         xgemm.MDIMC == other.xgemm.MDIMC && xgemm.NDIMC == other.xgemm.NDIMC && xgemm.MWG == other.xgemm.MWG &&
         xgemm.NWG == other.xgemm.NWG && xgemm.KWG == other.xgemm.KWG && xgemm.MDIMA == other.xgemm.MDIMA &&
         xgemm.NDIMB == other.xgemm.NDIMB && xgemmDirect.WGD == other.xgemmDirect.WGD &&
         xgemmDirect.MDIMCD == other.xgemmDirect.MDIMCD && xgemmDirect.NDIMCD == other.xgemmDirect.NDIMCD &&
         xgemmDirect.MDIMAD == other.xgemmDirect.MDIMAD && xgemmDirect.NDIMBD == other.xgemmDirect.NDIMBD &&
         xgemmDirect.KWID == other.xgemmDirect.KWID && xgemmDirect.PADA == other.xgemmDirect.PADA &&
         xgemmDirect.PADB == other.xgemmDirect.PADB;
}

void VulkanTuneParams::save(const string& filename, const VulkanTuneParams& config) {
  if(!config.isValid())
    throw StringError("VulkanTuneParams::save: refusing to save invalid parameters to " + filename);
  ofstream out;
  FileUtils::open(out, filename);
  out << VERSION_LINE << "\n";
  writeParam(out, "vulkan.canUseFP16Storage", config.vulkan.canUseFP16Storage);
  writeParam(out, "vulkan.canUseFP16Compute", config.vulkan.canUseFP16Compute);
  writeParam(out, "vulkan.canUseCooperativMatrix", config.vulkan.canUseCooperativMatrix);
  writeParam(out, "vulkan.canUseSubgroup", config.vulkan.canUseSubgroup);
  writeParam(out, "vulkan.shouldUseFP16Storage", config.vulkan.shouldUseFP16Storage);
  writeParam(out, "vulkan.shouldUseFP16Compute", config.vulkan.shouldUseFP16Compute);
  writeParam(out, "vulkan.shouldUseCooperativeMatrix", config.vulkan.shouldUseCooperativeMatrix);
  writeParam(out, "vulkan.shouldUseSubgroup", config.vulkan.shouldUseSubgroup);
#define WRITE_CONV(prefix, p) \
  writeParam(out, prefix ".inTileYSize", p.inTileYSize); \
  writeParam(out, prefix ".inTileXSize", p.inTileXSize); \
  writeParam(out, prefix ".outTileYSize", p.outTileYSize); \
  writeParam(out, prefix ".outTileXSize", p.outTileXSize); \
  writeParam(out, prefix ".inputTransformLocalXSize", p.inputTransformLocalXSize); \
  writeParam(out, prefix ".inputTransformLocalYSize", p.inputTransformLocalYSize); \
  writeParam(out, prefix ".outputTransformLocalXSize", p.outputTransformLocalXSize); \
  writeParam(out, prefix ".outputTransformLocalYSize", p.outputTransformLocalYSize); \
  writeParam(out, prefix ".outputTransformLocalZSize", p.outputTransformLocalZSize)
  WRITE_CONV("conv3x3", config.conv3x3);
  WRITE_CONV("conv5x5", config.conv5x5);
#undef WRITE_CONV
  writeParam(out, "xgemm.MDIMC", config.xgemm.MDIMC);
  writeParam(out, "xgemm.NDIMC", config.xgemm.NDIMC);
  writeParam(out, "xgemm.MWG", config.xgemm.MWG);
  writeParam(out, "xgemm.NWG", config.xgemm.NWG);
  writeParam(out, "xgemm.KWG", config.xgemm.KWG);
  writeParam(out, "xgemm.MDIMA", config.xgemm.MDIMA);
  writeParam(out, "xgemm.NDIMB", config.xgemm.NDIMB);
  writeParam(out, "xgemmDirect.WGD", config.xgemmDirect.WGD);
  writeParam(out, "xgemmDirect.MDIMCD", config.xgemmDirect.MDIMCD);
  writeParam(out, "xgemmDirect.NDIMCD", config.xgemmDirect.NDIMCD);
  writeParam(out, "xgemmDirect.MDIMAD", config.xgemmDirect.MDIMAD);
  writeParam(out, "xgemmDirect.NDIMBD", config.xgemmDirect.NDIMBD);
  writeParam(out, "xgemmDirect.KWID", config.xgemmDirect.KWID);
  writeParam(out, "xgemmDirect.PADA", config.xgemmDirect.PADA);
  writeParam(out, "xgemmDirect.PADB", config.xgemmDirect.PADB);
  out.close();
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
      if(line != VERSION_LINE)
        throw IOError("VulkanTuneParams::load: expected first line to be " + string(VERSION_LINE) + " in " + filename);
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
  if(values.size() != 41)
    throw IOError("VulkanTuneParams::load: unexpected number of parameters in " + filename);

  VulkanTuneParams config;
  config.vulkan.canUseFP16Storage = getBoolParam(values, "vulkan.canUseFP16Storage", filename);
  config.vulkan.canUseFP16Compute = getBoolParam(values, "vulkan.canUseFP16Compute", filename);
  config.vulkan.canUseCooperativMatrix = getBoolParam(values, "vulkan.canUseCooperativMatrix", filename);
  config.vulkan.canUseSubgroup = getBoolParam(values, "vulkan.canUseSubgroup", filename);
  config.vulkan.shouldUseFP16Storage = getBoolParam(values, "vulkan.shouldUseFP16Storage", filename);
  config.vulkan.shouldUseFP16Compute = getBoolParam(values, "vulkan.shouldUseFP16Compute", filename);
  config.vulkan.shouldUseCooperativeMatrix = getBoolParam(values, "vulkan.shouldUseCooperativeMatrix", filename);
  config.vulkan.shouldUseSubgroup = getBoolParam(values, "vulkan.shouldUseSubgroup", filename);
#define READ_CONV(prefix, p) \
  p.inTileYSize = getParam(values, prefix ".inTileYSize", filename); \
  p.inTileXSize = getParam(values, prefix ".inTileXSize", filename); \
  p.outTileYSize = getParam(values, prefix ".outTileYSize", filename); \
  p.outTileXSize = getParam(values, prefix ".outTileXSize", filename); \
  p.inputTransformLocalXSize = getParam(values, prefix ".inputTransformLocalXSize", filename); \
  p.inputTransformLocalYSize = getParam(values, prefix ".inputTransformLocalYSize", filename); \
  p.outputTransformLocalXSize = getParam(values, prefix ".outputTransformLocalXSize", filename); \
  p.outputTransformLocalYSize = getParam(values, prefix ".outputTransformLocalYSize", filename); \
  p.outputTransformLocalZSize = getParam(values, prefix ".outputTransformLocalZSize", filename)
  READ_CONV("conv3x3", config.conv3x3);
  READ_CONV("conv5x5", config.conv5x5);
#undef READ_CONV
  config.xgemm.MDIMC = getParam(values, "xgemm.MDIMC", filename);
  config.xgemm.NDIMC = getParam(values, "xgemm.NDIMC", filename);
  config.xgemm.MWG = getParam(values, "xgemm.MWG", filename);
  config.xgemm.NWG = getParam(values, "xgemm.NWG", filename);
  config.xgemm.KWG = getParam(values, "xgemm.KWG", filename);
  config.xgemm.MDIMA = getParam(values, "xgemm.MDIMA", filename);
  config.xgemm.NDIMB = getParam(values, "xgemm.NDIMB", filename);
  config.xgemmDirect.WGD = getParam(values, "xgemmDirect.WGD", filename);
  config.xgemmDirect.MDIMCD = getParam(values, "xgemmDirect.MDIMCD", filename);
  config.xgemmDirect.NDIMCD = getParam(values, "xgemmDirect.NDIMCD", filename);
  config.xgemmDirect.MDIMAD = getParam(values, "xgemmDirect.MDIMAD", filename);
  config.xgemmDirect.NDIMBD = getParam(values, "xgemmDirect.NDIMBD", filename);
  config.xgemmDirect.KWID = getParam(values, "xgemmDirect.KWID", filename);
  config.xgemmDirect.PADA = getParam(values, "xgemmDirect.PADA", filename);
  config.xgemmDirect.PADB = getParam(values, "xgemmDirect.PADB", filename);
  if(!config.isValid())
    throw IOError("VulkanTuneParams::load: parameters are invalid in " + filename);
  return config;
}

VulkanTuner::ModelInfoForTuning VulkanTuner::ModelInfoForTuning::ofDesc(const ModelDesc& desc) {
  return {desc.trunk.trunkNumChannels, desc.modelVersion};
}

string VulkanTuner::defaultDirectory(bool makeDir, const string& homeDataDirOverride) {
  string dir = HomeData::getHomeDataDir(true, homeDataDirOverride) + "/vulkantuning";
  if(makeDir)
    MakeDir::make(dir);
  return dir;
}

string
VulkanTuner::defaultFileName(const string& gpuName, int nnXLen, int nnYLen, int trunkNumChannels, int modelVersion) {
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
VulkanTuner::defaultFileName(const string& gpuName, int nnXLen, int nnYLen, const ModelInfoForTuning& modelInfo) {
  return defaultFileName(gpuName, nnXLen, nnYLen, modelInfo.trunkNumChannels, modelInfo.modelVersion);
}

VulkanTuneParams VulkanTuner::loadOrCreate(
  const string& tunerFile,
  const string& homeDataDirOverride,
  const string& gpuName,
  int nnXLen,
  int nnYLen,
  const ModelInfoForTuning& modelInfo,
  const VulkanDeviceInfo& deviceInfo,
  Logger* logger) {
  string filename = tunerFile;
  if(filename.empty()) {
    filename = defaultDirectory(true, homeDataDirOverride) + "/" +
               defaultFileName(gpuName, nnXLen, nnYLen, modelInfo);
  }

  try {
    VulkanTuneParams loaded = VulkanTuneParams::load(filename);
    if(logger != nullptr)
      logger->write("Loaded Vulkan tuning parameters from: " + filename);
    return loaded;
  } catch(const StringError&) {
  }

  VulkanTuneParams params;
  params.vulkan = makeVulkanParams(deviceInfo);
  VulkanTuneParams::save(filename, params);
  if(logger != nullptr)
    logger->write("Saved default Vulkan tuning parameters to: " + filename);
  return params;
}

#endif
