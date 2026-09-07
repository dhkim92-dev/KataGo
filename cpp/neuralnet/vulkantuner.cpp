#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"
#include "../neuralnet/vulkancompute.h"

#include <fstream>
#include <map>
#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/rand.h"
#include "../dataio/homedata.h"

using namespace std;
using namespace vk_shader;
using namespace vk_shader::tune;

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

double VulkanTuner::computeTuningScore(double callsPerSecond, double errorProp, double errorToleranceScale) {
  if(!isfinite(callsPerSecond) || callsPerSecond <= 0.0 || !isfinite(errorProp) ||
     errorProp > std::min(0.5, 5.0 * errorToleranceScale))
    return 0.0;
  double penalty = 1.0 - sqrt(errorProp / (errorProp + errorToleranceScale));
  if(errorProp > 1e-5)
    penalty *= 0.90;
  return callsPerSecond * penalty;
}

bool VulkanTuner::isFastEnough(double callsPerSecond, double baselineCallsPerSecond, double requiredRatio) {
  return isfinite(callsPerSecond) && isfinite(baselineCallsPerSecond) &&
         callsPerSecond > 0.0 && baselineCallsPerSecond > 0.0 &&
         callsPerSecond >= baselineCallsPerSecond * requiredRatio;
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
    params.canUseCooperativeMatrix = deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE;
    params.canUseSubgroup =
      (deviceInfo.subgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
      deviceInfo.subgroupSizeControlFeatures.computeFullSubgroups == VK_TRUE;
    return params;
  }

}  // namespace

bool AddPointWiseTuneParams::isValid() const {
  return LOCAL_SIZE > 0 && LOCAL_SIZE <= 1024 && ELTS_PER_THREAD > 0 && ELTS_PER_THREAD <= 32;
}

bool AddChannelBiasesNCHWTuneParams::isValid() const {
  return XY_ELTS_PER_THREAD > 0 && XY_ELTS_PER_THREAD <= 32 &&
         NC_ELTS_PER_THREAD > 0 && NC_ELTS_PER_THREAD <= 32;
}

bool GPoolTuneParams::isValid() const {
  if(XYSTRIDE <= 0 || CHANNELSTRIDE <= 0 || BATCHSTRIDE <= 0)
    return false;
  return static_cast<uint64_t>(XYSTRIDE) * CHANNELSTRIDE * BATCHSTRIDE <= 1024;
}

bool ConvTuneParams::isValid(uint32_t expectedOutTileSize) const {
  if(inTileXSize != 6 || inTileYSize != 6 || outTileXSize != expectedOutTileSize || outTileYSize != expectedOutTileSize)
    return false;
  if(inputTransformLocalXSize == 0 || inputTransformLocalYSize == 0 ||
     outputTransformLocalXSize == 0 || outputTransformLocalYSize == 0 || outputTransformLocalZSize == 0)
    return false;
  return static_cast<uint64_t>(inputTransformLocalXSize) * inputTransformLocalYSize <= 1024 &&
         static_cast<uint64_t>(outputTransformLocalXSize) * outputTransformLocalYSize * outputTransformLocalZSize <= 1024;
}

bool XgemmTuneParams::isValid() const {
  if(MDIMC == 0 || NDIMC == 0 || MWG == 0 || NWG == 0 || KWG == 0 || MDIMA == 0 || NDIMB == 0)
    return false;
  const uint64_t workgroupSize = static_cast<uint64_t>(MDIMC) * NDIMC;
  if(workgroupSize == 0 || workgroupSize > 1024)
    return false;
  return isMultipleOf(MWG, static_cast<uint64_t>(MDIMC) * 4) &&
         isMultipleOf(NWG, static_cast<uint64_t>(NDIMC) * 4) &&
         isMultipleOf(MWG, static_cast<uint64_t>(MDIMA) * 4) &&
         isMultipleOf(NWG, static_cast<uint64_t>(NDIMB) * 4) &&
         isMultipleOf(KWG, 4) &&
         isMultipleOf(KWG, workgroupSize / MDIMA) &&
         isMultipleOf(KWG, workgroupSize / NDIMB);
}

bool XgemmTuneParams::isSimple() const {
  return MDIMC == MDIMA && NDIMC == NDIMB && MWG == NWG;
}

bool XgemmDirectTuneParams::isValid() const {
  if(WGD == 0 || MDIMCD == 0 || NDIMCD == 0 || MDIMAD == 0 || NDIMBD == 0 || KWID == 0)
    return false;
  const uint64_t workgroupSize = static_cast<uint64_t>(MDIMCD) * NDIMCD;
  if(workgroupSize > 1024 || PADA > 1 || PADB > 1)
    return false;
  if(!isMultipleOf(WGD, KWID) ||
     !isMultipleOf(WGD, MDIMCD) || !isMultipleOf(WGD, NDIMCD) ||
     !isMultipleOf(WGD, static_cast<uint64_t>(MDIMAD) * 4) ||
     !isMultipleOf(WGD, static_cast<uint64_t>(NDIMBD) * 4) ||
     !isMultipleOf(workgroupSize, MDIMAD) || !isMultipleOf(workgroupSize, NDIMBD))
    return false;
  return isMultipleOf(WGD, workgroupSize / MDIMAD) &&
         isMultipleOf(WGD, workgroupSize / NDIMBD);
}

bool HGemmCooperativeMatrixTuneParams::isValid() const {
  if(MWARP <= 0 || NWARP <= 0 || KDIM <= 0 || subgroupSize == 0 ||
     MWG <= 0 || NWG <= 0 || KWG <= 0 || MWAVE <= 0 || NWAVE <= 0 ||
     SA < 0 || SA > 1 || SB < 0 || SB > 1)
    return false;
  const uint64_t localSizeX = static_cast<uint64_t>(MWAVE / MWARP) * subgroupSize;
  const uint64_t localSizeY = static_cast<uint64_t>(NWAVE / NWARP);
  if(localSizeX == 0 || localSizeY == 0 || localSizeX * localSizeY > 1024)
    return false;
  return isMultipleOf(MWG, MWAVE) && isMultipleOf(NWG, NWAVE) &&
         isMultipleOf(KWG, KDIM) && isMultipleOf(MWAVE, MWARP) &&
         isMultipleOf(NWAVE, NWARP) && isMultipleOf(MWG, 4) &&
         isMultipleOf(NWG, 4) && isMultipleOf(KWG, 4);
}

bool HGemmCooperativeMatrixTuneParams::isSimple() const {
  if(MWAVE != MWARP && MWAVE == MWG)
    return false;
  if(NWAVE != NWARP && NWAVE == NWG)
    return false;
  return MWG == NWG;
}

bool HGemmCooperativeMatrixNCHWTuneParams::isValid() const {
  if(MWARP <= 0 || NWARP <= 0 || KDIM <= 0 || subgroupSize == 0 ||
     MWG <= 0 || NWG <= 0 || KWG <= 0 ||
     MWAVE <= 0 || NWAVE <= 0 || SB < 0 || SB > 1 ||
     VWM != 4 || VWN != 4)
    return false;
  const uint64_t localSizeX = static_cast<uint64_t>(MWAVE / MWARP) * subgroupSize;
  const uint64_t localSizeY = static_cast<uint64_t>(NWAVE / NWARP);
  if(localSizeX == 0 || localSizeY == 0 || localSizeX * localSizeY > 1024)
    return false;
  if((CType != spec::HGemmCooperativeMatrixNCHWSpec::COMPONENT_TYPE_FLOAT16 &&
      CType != spec::HGemmCooperativeMatrixNCHWSpec::COMPONENT_TYPE_FLOAT32) ||
     (ResultType != spec::HGemmCooperativeMatrixNCHWSpec::COMPONENT_TYPE_FLOAT16 &&
      ResultType != spec::HGemmCooperativeMatrixNCHWSpec::COMPONENT_TYPE_FLOAT32) ||
     CType != ResultType)
    return false;
  if(!isMultipleOf(MWARP, 4) || !isMultipleOf(NWARP, 4))
    return false;
  if(!isMultipleOf(getRequiredCDivisor(), NWG) ||
     !isMultipleOf(getRequiredCDivisor(), KWG))
    return false;
  return isMultipleOf(MWG, MWAVE) && isMultipleOf(NWG, NWAVE) &&
         isMultipleOf(KWG, KDIM) && isMultipleOf(MWAVE, MWARP) &&
         isMultipleOf(NWAVE, NWARP);
}

int HGemmCooperativeMatrixNCHWTuneParams::getRequiredCDivisor() const {
  // Keep the Vulkan NCHW HGEMM channel contract identical to OpenCL.
  return 32;
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
  return ATTN_BLOCK_Q > 0 && ATTN_BLOCK_Q <= 1024 && ATTN_BLOCK_KV > 0 && Q_PER_THREAD > 0;
}

bool TransformerRMSNormTuneParms::isValid() const {
  return WG_C_SIZE > 0 && WG_XY_SIZE > 0 && C_PER_THREAD > 0 &&
         static_cast<uint64_t>(WG_C_SIZE) * WG_XY_SIZE <= 1024;
}

bool TransformerSpatialRmsNormTuneParams::isValid() const {
  return TILE_SIZE > 0 && TILE_SIZE <= 1024 && APPLY_ELTS_PER_THREAD > 0 && APPLY_ELTS_PER_THREAD <= 32;
}

bool VulkanTuneParams::isValid() const {
  return addChannelBiases.isValid() && pointwise.isValid() && gPool.isValid() &&
         conv3x3.isValid(4) && conv5x5.isValid(2) && hgemmCooperativeMatrix.isValid() &&
         hgemmCooperativeMatrixNCHW.isValid() &&
         xgemm.isValid() && xgemm16.isValid() && xgemmDirect.isValid() &&
         transformer.isValid() && rmsNorm.isValid() && spatialRMSNorm.isValid();
}

bool VulkanTuneParams::operator==(const VulkanTuneParams& other) const {
  return vulkan.canUseFP16Storage == other.vulkan.canUseFP16Storage &&
         vulkan.canUseFP16Compute == other.vulkan.canUseFP16Compute &&
         vulkan.canUseCooperativeMatrix == other.vulkan.canUseCooperativeMatrix &&
         vulkan.canUseSubgroup == other.vulkan.canUseSubgroup &&
         vulkan.shouldUseFP16Storage == other.vulkan.shouldUseFP16Storage &&
         vulkan.shouldUseFP16Compute == other.vulkan.shouldUseFP16Compute &&
         vulkan.shouldUseCooperativeMatrix == other.vulkan.shouldUseCooperativeMatrix &&
         vulkan.shouldUseHgemmCooperativeMatrixNCHW == other.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
         vulkan.shouldUseSubgroup == other.vulkan.shouldUseSubgroup &&
         addChannelBiases.XY_ELTS_PER_THREAD == other.addChannelBiases.XY_ELTS_PER_THREAD &&
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
         conv5x5.inTileYSize == other.conv5x5.inTileYSize && conv5x5.inTileXSize == other.conv5x5.inTileXSize &&
         conv5x5.outTileYSize == other.conv5x5.outTileYSize && conv5x5.outTileXSize == other.conv5x5.outTileXSize &&
         conv5x5.inputTransformLocalXSize == other.conv5x5.inputTransformLocalXSize &&
         conv5x5.inputTransformLocalYSize == other.conv5x5.inputTransformLocalYSize &&
         conv5x5.outputTransformLocalXSize == other.conv5x5.outputTransformLocalXSize &&
         conv5x5.outputTransformLocalYSize == other.conv5x5.outputTransformLocalYSize &&
         conv5x5.outputTransformLocalZSize == other.conv5x5.outputTransformLocalZSize &&
         hgemmCooperativeMatrix.MWARP == other.hgemmCooperativeMatrix.MWARP &&
         hgemmCooperativeMatrix.NWARP == other.hgemmCooperativeMatrix.NWARP &&
         hgemmCooperativeMatrix.KDIM == other.hgemmCooperativeMatrix.KDIM &&
         hgemmCooperativeMatrix.subgroupSize == other.hgemmCooperativeMatrix.subgroupSize &&
         hgemmCooperativeMatrix.MWG == other.hgemmCooperativeMatrix.MWG &&
         hgemmCooperativeMatrix.NWG == other.hgemmCooperativeMatrix.NWG &&
         hgemmCooperativeMatrix.KWG == other.hgemmCooperativeMatrix.KWG &&
         hgemmCooperativeMatrix.MWAVE == other.hgemmCooperativeMatrix.MWAVE &&
         hgemmCooperativeMatrix.NWAVE == other.hgemmCooperativeMatrix.NWAVE &&
         hgemmCooperativeMatrix.SA == other.hgemmCooperativeMatrix.SA &&
         hgemmCooperativeMatrix.SB == other.hgemmCooperativeMatrix.SB &&
         hgemmCooperativeMatrixNCHW.MWARP == other.hgemmCooperativeMatrixNCHW.MWARP &&
         hgemmCooperativeMatrixNCHW.NWARP == other.hgemmCooperativeMatrixNCHW.NWARP &&
         hgemmCooperativeMatrixNCHW.KDIM == other.hgemmCooperativeMatrixNCHW.KDIM &&
         hgemmCooperativeMatrixNCHW.subgroupSize == other.hgemmCooperativeMatrixNCHW.subgroupSize &&
         hgemmCooperativeMatrixNCHW.MWG == other.hgemmCooperativeMatrixNCHW.MWG &&
         hgemmCooperativeMatrixNCHW.NWG == other.hgemmCooperativeMatrixNCHW.NWG &&
         hgemmCooperativeMatrixNCHW.KWG == other.hgemmCooperativeMatrixNCHW.KWG &&
         hgemmCooperativeMatrixNCHW.MWAVE == other.hgemmCooperativeMatrixNCHW.MWAVE &&
         hgemmCooperativeMatrixNCHW.NWAVE == other.hgemmCooperativeMatrixNCHW.NWAVE &&
         hgemmCooperativeMatrixNCHW.CType == other.hgemmCooperativeMatrixNCHW.CType &&
         hgemmCooperativeMatrixNCHW.ResultType == other.hgemmCooperativeMatrixNCHW.ResultType &&
         hgemmCooperativeMatrixNCHW.SB == other.hgemmCooperativeMatrixNCHW.SB &&
         hgemmCooperativeMatrixNCHW.VWM == other.hgemmCooperativeMatrixNCHW.VWM &&
         hgemmCooperativeMatrixNCHW.VWN == other.hgemmCooperativeMatrixNCHW.VWN &&
         xgemm.MDIMC == other.xgemm.MDIMC && xgemm.NDIMC == other.xgemm.NDIMC && xgemm.MWG == other.xgemm.MWG &&
         xgemm.NWG == other.xgemm.NWG && xgemm.KWG == other.xgemm.KWG && xgemm.MDIMA == other.xgemm.MDIMA &&
         xgemm.NDIMB == other.xgemm.NDIMB &&
         xgemm16.MDIMC == other.xgemm16.MDIMC && xgemm16.NDIMC == other.xgemm16.NDIMC &&
         xgemm16.MWG == other.xgemm16.MWG && xgemm16.NWG == other.xgemm16.NWG &&
         xgemm16.KWG == other.xgemm16.KWG && xgemm16.MDIMA == other.xgemm16.MDIMA &&
         xgemm16.NDIMB == other.xgemm16.NDIMB && xgemmDirect.WGD == other.xgemmDirect.WGD &&
         xgemmDirect.MDIMCD == other.xgemmDirect.MDIMCD && xgemmDirect.NDIMCD == other.xgemmDirect.NDIMCD &&
         xgemmDirect.MDIMAD == other.xgemmDirect.MDIMAD && xgemmDirect.NDIMBD == other.xgemmDirect.NDIMBD &&
         xgemmDirect.KWID == other.xgemmDirect.KWID && xgemmDirect.PADA == other.xgemmDirect.PADA &&
         xgemmDirect.PADB == other.xgemmDirect.PADB &&
         transformer.ATTN_BLOCK_Q == other.transformer.ATTN_BLOCK_Q &&
         transformer.ATTN_BLOCK_KV == other.transformer.ATTN_BLOCK_KV &&
         transformer.Q_PER_THREAD == other.transformer.Q_PER_THREAD &&
         transformer.USE_TILED_ATTN == other.transformer.USE_TILED_ATTN &&
         rmsNorm.WG_C_SIZE == other.rmsNorm.WG_C_SIZE &&
         rmsNorm.WG_XY_SIZE == other.rmsNorm.WG_XY_SIZE &&
         rmsNorm.C_PER_THREAD == other.rmsNorm.C_PER_THREAD &&
         spatialRMSNorm.TILE_SIZE == other.spatialRMSNorm.TILE_SIZE &&
         spatialRMSNorm.APPLY_ELTS_PER_THREAD == other.spatialRMSNorm.APPLY_ELTS_PER_THREAD;
}

void VulkanTuneParams::save(const string& filename, const VulkanTuneParams& config) {
  if(!config.isValid())
    throw StringError("VulkanTuneParams::save: refusing to save invalid parameters to " + filename);
  ofstream out;
  FileUtils::open(out, filename);
  out << VERSION_LINE << "\n";
  writeParam(out, "vulkan.canUseFP16Storage", config.vulkan.canUseFP16Storage);
  writeParam(out, "vulkan.canUseFP16Compute", config.vulkan.canUseFP16Compute);
  writeParam(out, "vulkan.canUseCooperativeMatrix", config.vulkan.canUseCooperativeMatrix);
  writeParam(out, "vulkan.canUseSubgroup", config.vulkan.canUseSubgroup);
  writeParam(out, "vulkan.shouldUseFP16Storage", config.vulkan.shouldUseFP16Storage);
  writeParam(out, "vulkan.shouldUseFP16Compute", config.vulkan.shouldUseFP16Compute);
  writeParam(out, "vulkan.shouldUseCooperativeMatrix", config.vulkan.shouldUseCooperativeMatrix);
  writeParam(out, "vulkan.shouldUseHgemmCooperativeMatrixNCHW", config.vulkan.shouldUseHgemmCooperativeMatrixNCHW);
  writeParam(out, "vulkan.shouldUseSubgroup", config.vulkan.shouldUseSubgroup);

  writeParam(out, "xgemmDirect.WGD", config.xgemmDirect.WGD);
  writeParam(out, "xgemmDirect.MDIMCD", config.xgemmDirect.MDIMCD);
  writeParam(out, "xgemmDirect.NDIMCD", config.xgemmDirect.NDIMCD);
  writeParam(out, "xgemmDirect.MDIMAD", config.xgemmDirect.MDIMAD);
  writeParam(out, "xgemmDirect.NDIMBD", config.xgemmDirect.NDIMBD);
  writeParam(out, "xgemmDirect.KWID", config.xgemmDirect.KWID);
  writeParam(out, "xgemmDirect.PADA", config.xgemmDirect.PADA);
  writeParam(out, "xgemmDirect.PADB", config.xgemmDirect.PADB);

  writeParam(out, "xgemm.MWG", config.xgemm.MWG);
  writeParam(out, "xgemm.NWG", config.xgemm.NWG);
  writeParam(out, "xgemm.KWG", config.xgemm.KWG);
  writeParam(out, "xgemm.MDIMC", config.xgemm.MDIMC);
  writeParam(out, "xgemm.NDIMC", config.xgemm.NDIMC);
  writeParam(out, "xgemm.MDIMA", config.xgemm.MDIMA);
  writeParam(out, "xgemm.NDIMB", config.xgemm.NDIMB);

  writeParam(out, "xgemm16.MWG", config.xgemm16.MWG);
  writeParam(out, "xgemm16.NWG", config.xgemm16.NWG);
  writeParam(out, "xgemm16.KWG", config.xgemm16.KWG);
  writeParam(out, "xgemm16.MDIMC", config.xgemm16.MDIMC);
  writeParam(out, "xgemm16.NDIMC", config.xgemm16.NDIMC);
  writeParam(out, "xgemm16.MDIMA", config.xgemm16.MDIMA);
  writeParam(out, "xgemm16.NDIMB", config.xgemm16.NDIMB);

  writeParam(out, "hgemmCooperativeMatrix.MWG", config.hgemmCooperativeMatrix.MWG);
  writeParam(out, "hgemmCooperativeMatrix.NWG", config.hgemmCooperativeMatrix.NWG);
  writeParam(out, "hgemmCooperativeMatrix.KWG", config.hgemmCooperativeMatrix.KWG);
  writeParam(out, "hgemmCooperativeMatrix.MWAVE", config.hgemmCooperativeMatrix.MWAVE);
  writeParam(out, "hgemmCooperativeMatrix.NWAVE", config.hgemmCooperativeMatrix.NWAVE);
  writeParam(out, "hgemmCooperativeMatrix.MWARP", config.hgemmCooperativeMatrix.MWARP);
  writeParam(out, "hgemmCooperativeMatrix.NWARP", config.hgemmCooperativeMatrix.NWARP);
  writeParam(out, "hgemmCooperativeMatrix.SA", config.hgemmCooperativeMatrix.SA);
  writeParam(out, "hgemmCooperativeMatrix.SB", config.hgemmCooperativeMatrix.SB);
  writeParam(out, "hgemmCooperativeMatrix.KDIM", config.hgemmCooperativeMatrix.KDIM);
  writeParam(out, "hgemmCooperativeMatrix.subgroupSize", config.hgemmCooperativeMatrix.subgroupSize);

  writeParam(out, "hgemmCooperativeMatrixNCHW.MWG", config.hgemmCooperativeMatrixNCHW.MWG);
  writeParam(out, "hgemmCooperativeMatrixNCHW.NWG", config.hgemmCooperativeMatrixNCHW.NWG);
  writeParam(out, "hgemmCooperativeMatrixNCHW.KWG", config.hgemmCooperativeMatrixNCHW.KWG);
  writeParam(out, "hgemmCooperativeMatrixNCHW.MWAVE", config.hgemmCooperativeMatrixNCHW.MWAVE);
  writeParam(out, "hgemmCooperativeMatrixNCHW.NWAVE", config.hgemmCooperativeMatrixNCHW.NWAVE);
  writeParam(out, "hgemmCooperativeMatrixNCHW.MWARP", config.hgemmCooperativeMatrixNCHW.MWARP);
  writeParam(out, "hgemmCooperativeMatrixNCHW.NWARP", config.hgemmCooperativeMatrixNCHW.NWARP);
  writeParam(out, "hgemmCooperativeMatrixNCHW.VWM", config.hgemmCooperativeMatrixNCHW.VWM);
  writeParam(out, "hgemmCooperativeMatrixNCHW.VWN", config.hgemmCooperativeMatrixNCHW.VWN);
  writeParam(out, "hgemmCooperativeMatrixNCHW.SB", config.hgemmCooperativeMatrixNCHW.SB);
  writeParam(out, "hgemmCooperativeMatrixNCHW.KDIM", config.hgemmCooperativeMatrixNCHW.KDIM);
  writeParam(out, "hgemmCooperativeMatrixNCHW.subgroupSize", config.hgemmCooperativeMatrixNCHW.subgroupSize);
  writeParam(out, "hgemmCooperativeMatrixNCHW.CType", config.hgemmCooperativeMatrixNCHW.CType);
  writeParam(out, "hgemmCooperativeMatrixNCHW.ResultType", config.hgemmCooperativeMatrixNCHW.ResultType);

#define WRITE_CONV(prefix, p) \
  writeParam(out, prefix ".inTileXSize", p.inTileXSize); \
  writeParam(out, prefix ".inTileYSize", p.inTileYSize); \
  writeParam(out, prefix ".outTileXSize", p.outTileXSize); \
  writeParam(out, prefix ".outTileYSize", p.outTileYSize); \
  writeParam(out, prefix ".inputTransformLocalXSize", p.inputTransformLocalXSize); \
  writeParam(out, prefix ".inputTransformLocalYSize", p.inputTransformLocalYSize); \
  writeParam(out, prefix ".outputTransformLocalXSize", p.outputTransformLocalXSize); \
  writeParam(out, prefix ".outputTransformLocalYSize", p.outputTransformLocalYSize); \
  writeParam(out, prefix ".outputTransformLocalZSize", p.outputTransformLocalZSize)
  WRITE_CONV("conv3x3", config.conv3x3);
  WRITE_CONV("conv5x5", config.conv5x5);
#undef WRITE_CONV

  writeParam(out, "gPool.XYSTRIDE", config.gPool.XYSTRIDE);
  writeParam(out, "gPool.CHANNELSTRIDE", config.gPool.CHANNELSTRIDE);
  writeParam(out, "gPool.BATCHSTRIDE", config.gPool.BATCHSTRIDE);
  writeParam(out, "transformer.ATTN_BLOCK_Q", config.transformer.ATTN_BLOCK_Q);
  writeParam(out, "transformer.ATTN_BLOCK_KV", config.transformer.ATTN_BLOCK_KV);
  writeParam(out, "transformer.Q_PER_THREAD", config.transformer.Q_PER_THREAD);
  writeParam(out, "transformer.USE_TILED_ATTN", config.transformer.USE_TILED_ATTN);
  writeParam(out, "rmsNorm.WG_C_SIZE", config.rmsNorm.WG_C_SIZE);
  writeParam(out, "rmsNorm.WG_XY_SIZE", config.rmsNorm.WG_XY_SIZE);
  writeParam(out, "rmsNorm.C_PER_THREAD", config.rmsNorm.C_PER_THREAD);
  writeParam(out, "pointwise.ELTS_PER_THREAD", config.pointwise.ELTS_PER_THREAD);
  writeParam(out, "pointwise.LOCAL_SIZE", config.pointwise.LOCAL_SIZE);
  writeParam(out, "addChannelBiases.XY_ELTS_PER_THREAD", config.addChannelBiases.XY_ELTS_PER_THREAD);
  writeParam(out, "addChannelBiases.NC_ELTS_PER_THREAD", config.addChannelBiases.NC_ELTS_PER_THREAD);
  writeParam(out, "spatialRMSNorm.TILE_SIZE", config.spatialRMSNorm.TILE_SIZE);
  writeParam(out, "spatialRMSNorm.APPLY_ELTS_PER_THREAD", config.spatialRMSNorm.APPLY_ELTS_PER_THREAD);
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
  if(values.size() != 89 && values.size() != 90)
    throw IOError("VulkanTuneParams::load: unexpected number of parameters in " + filename);

  VulkanTuneParams config;
  config.vulkan.canUseFP16Storage = getBoolParam(values, "vulkan.canUseFP16Storage", filename);
  config.vulkan.canUseFP16Compute = getBoolParam(values, "vulkan.canUseFP16Compute", filename);
  config.vulkan.canUseCooperativeMatrix = getBoolParam(values, "vulkan.canUseCooperativeMatrix", filename);
  config.vulkan.canUseSubgroup = getBoolParam(values, "vulkan.canUseSubgroup", filename);
  config.vulkan.shouldUseFP16Storage = getBoolParam(values, "vulkan.shouldUseFP16Storage", filename);
  config.vulkan.shouldUseFP16Compute = getBoolParam(values, "vulkan.shouldUseFP16Compute", filename);
  config.vulkan.shouldUseCooperativeMatrix = getBoolParam(values, "vulkan.shouldUseCooperativeMatrix", filename);
  auto hgemmUseIter = values.find("vulkan.shouldUseHgemmCooperativeMatrixNCHW");
  if(hgemmUseIter != values.end())
    config.vulkan.shouldUseHgemmCooperativeMatrixNCHW = getBoolParam(values, "vulkan.shouldUseHgemmCooperativeMatrixNCHW", filename);
  config.vulkan.shouldUseSubgroup = getBoolParam(values, "vulkan.shouldUseSubgroup", filename);
  config.addChannelBiases.XY_ELTS_PER_THREAD = getParam(values, "addChannelBiases.XY_ELTS_PER_THREAD", filename);
  config.addChannelBiases.NC_ELTS_PER_THREAD = getParam(values, "addChannelBiases.NC_ELTS_PER_THREAD", filename);
  config.pointwise.LOCAL_SIZE = getParam(values, "pointwise.LOCAL_SIZE", filename);
  config.pointwise.ELTS_PER_THREAD = getParam(values, "pointwise.ELTS_PER_THREAD", filename);
  config.gPool.XYSTRIDE = getParam(values, "gPool.XYSTRIDE", filename);
  config.gPool.CHANNELSTRIDE = getParam(values, "gPool.CHANNELSTRIDE", filename);
  config.gPool.BATCHSTRIDE = getParam(values, "gPool.BATCHSTRIDE", filename);
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
  config.hgemmCooperativeMatrix.MWARP = getParam(values, "hgemmCooperativeMatrix.MWARP", filename);
  config.hgemmCooperativeMatrix.NWARP = getParam(values, "hgemmCooperativeMatrix.NWARP", filename);
  config.hgemmCooperativeMatrix.KDIM = getParam(values, "hgemmCooperativeMatrix.KDIM", filename);
  config.hgemmCooperativeMatrix.subgroupSize = getParam(values, "hgemmCooperativeMatrix.subgroupSize", filename);
  config.hgemmCooperativeMatrix.MWG = getParam(values, "hgemmCooperativeMatrix.MWG", filename);
  config.hgemmCooperativeMatrix.NWG = getParam(values, "hgemmCooperativeMatrix.NWG", filename);
  config.hgemmCooperativeMatrix.KWG = getParam(values, "hgemmCooperativeMatrix.KWG", filename);
  config.hgemmCooperativeMatrix.MWAVE = getParam(values, "hgemmCooperativeMatrix.MWAVE", filename);
  config.hgemmCooperativeMatrix.NWAVE = getParam(values, "hgemmCooperativeMatrix.NWAVE", filename);
  config.hgemmCooperativeMatrix.SA = getParam(values, "hgemmCooperativeMatrix.SA", filename);
  config.hgemmCooperativeMatrix.SB = getParam(values, "hgemmCooperativeMatrix.SB", filename);
  config.hgemmCooperativeMatrixNCHW.MWARP = getParam(values, "hgemmCooperativeMatrixNCHW.MWARP", filename);
  config.hgemmCooperativeMatrixNCHW.NWARP = getParam(values, "hgemmCooperativeMatrixNCHW.NWARP", filename);
  config.hgemmCooperativeMatrixNCHW.KDIM = getParam(values, "hgemmCooperativeMatrixNCHW.KDIM", filename);
  config.hgemmCooperativeMatrixNCHW.subgroupSize = getParam(values, "hgemmCooperativeMatrixNCHW.subgroupSize", filename);
  config.hgemmCooperativeMatrixNCHW.MWG = getParam(values, "hgemmCooperativeMatrixNCHW.MWG", filename);
  config.hgemmCooperativeMatrixNCHW.NWG = getParam(values, "hgemmCooperativeMatrixNCHW.NWG", filename);
  config.hgemmCooperativeMatrixNCHW.KWG = getParam(values, "hgemmCooperativeMatrixNCHW.KWG", filename);
  config.hgemmCooperativeMatrixNCHW.MWAVE = getParam(values, "hgemmCooperativeMatrixNCHW.MWAVE", filename);
  config.hgemmCooperativeMatrixNCHW.NWAVE = getParam(values, "hgemmCooperativeMatrixNCHW.NWAVE", filename);
  config.hgemmCooperativeMatrixNCHW.CType = getParam(values, "hgemmCooperativeMatrixNCHW.CType", filename);
  config.hgemmCooperativeMatrixNCHW.ResultType = getParam(values, "hgemmCooperativeMatrixNCHW.ResultType", filename);
  config.hgemmCooperativeMatrixNCHW.SB = getParam(values, "hgemmCooperativeMatrixNCHW.SB", filename);
  config.hgemmCooperativeMatrixNCHW.VWM = getParam(values, "hgemmCooperativeMatrixNCHW.VWM", filename);
  config.hgemmCooperativeMatrixNCHW.VWN = getParam(values, "hgemmCooperativeMatrixNCHW.VWN", filename);
  config.xgemm.MDIMC = getParam(values, "xgemm.MDIMC", filename);
  config.xgemm.NDIMC = getParam(values, "xgemm.NDIMC", filename);
  config.xgemm.MWG = getParam(values, "xgemm.MWG", filename);
  config.xgemm.NWG = getParam(values, "xgemm.NWG", filename);
  config.xgemm.KWG = getParam(values, "xgemm.KWG", filename);
  config.xgemm.MDIMA = getParam(values, "xgemm.MDIMA", filename);
  config.xgemm.NDIMB = getParam(values, "xgemm.NDIMB", filename);
  config.xgemm16.MDIMC = getParam(values, "xgemm16.MDIMC", filename);
  config.xgemm16.NDIMC = getParam(values, "xgemm16.NDIMC", filename);
  config.xgemm16.MWG = getParam(values, "xgemm16.MWG", filename);
  config.xgemm16.NWG = getParam(values, "xgemm16.NWG", filename);
  config.xgemm16.KWG = getParam(values, "xgemm16.KWG", filename);
  config.xgemm16.MDIMA = getParam(values, "xgemm16.MDIMA", filename);
  config.xgemm16.NDIMB = getParam(values, "xgemm16.NDIMB", filename);
  config.xgemmDirect.WGD = getParam(values, "xgemmDirect.WGD", filename);
  config.xgemmDirect.MDIMCD = getParam(values, "xgemmDirect.MDIMCD", filename);
  config.xgemmDirect.NDIMCD = getParam(values, "xgemmDirect.NDIMCD", filename);
  config.xgemmDirect.MDIMAD = getParam(values, "xgemmDirect.MDIMAD", filename);
  config.xgemmDirect.NDIMBD = getParam(values, "xgemmDirect.NDIMBD", filename);
  config.xgemmDirect.KWID = getParam(values, "xgemmDirect.KWID", filename);
  config.xgemmDirect.PADA = getParam(values, "xgemmDirect.PADA", filename);
  config.xgemmDirect.PADB = getParam(values, "xgemmDirect.PADB", filename);
  config.transformer.ATTN_BLOCK_Q = getParam(values, "transformer.ATTN_BLOCK_Q", filename);
  config.transformer.ATTN_BLOCK_KV = getParam(values, "transformer.ATTN_BLOCK_KV", filename);
  config.transformer.Q_PER_THREAD = getParam(values, "transformer.Q_PER_THREAD", filename);
  config.transformer.USE_TILED_ATTN = getParam(values, "transformer.USE_TILED_ATTN", filename);
  config.rmsNorm.WG_C_SIZE = getParam(values, "rmsNorm.WG_C_SIZE", filename);
  config.rmsNorm.WG_XY_SIZE = getParam(values, "rmsNorm.WG_XY_SIZE", filename);
  config.rmsNorm.C_PER_THREAD = getParam(values, "rmsNorm.C_PER_THREAD", filename);
  config.spatialRMSNorm.TILE_SIZE = getParam(values, "spatialRMSNorm.TILE_SIZE", filename);
  config.spatialRMSNorm.APPLY_ELTS_PER_THREAD = getParam(values, "spatialRMSNorm.APPLY_ELTS_PER_THREAD", filename);
  if(!config.isValid())
    throw IOError("VulkanTuneParams::load: parameters are invalid in " + filename);
  return config;
}

namespace {
  void findTransformerInfo(
    const vector<pair<int, unique_ptr_void>>& blocks,
    VulkanTuner::ModelInfoForTuning& modelInfo
  ) {
    for(const auto& block: blocks) {
      if(block.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
        const TransformerAttentionDesc* attn = static_cast<const TransformerAttentionDesc*>(block.second.get());
        modelInfo.transformerHeadDim = attn->qHeadDim;
        modelInfo.transformerVHeadDim = attn->vHeadDim;
        modelInfo.transformerNumHeads = attn->numHeads;
        modelInfo.transformerNumKVHeads = attn->numKVHeads;
      }
      else if(block.first == TRANSFORMER_FFN_BLOCK_KIND) {
        const TransformerFFNDesc* ffn = static_cast<const TransformerFFNDesc*>(block.second.get());
        modelInfo.transformerFFNChannels = ffn->ffnChannels;
      }
      else if(block.first == NESTED_BOTTLENECK_BLOCK_KIND) {
        const NestedBottleneckResidualBlockDesc* nested =
          static_cast<const NestedBottleneckResidualBlockDesc*>(block.second.get());
        findTransformerInfo(nested->blocks, modelInfo);
      }
    }
  }
}

VulkanTuner::ModelInfoForTuning VulkanTuner::ModelInfoForTuning::ofDesc(const ModelDesc& desc) {
  VulkanTuner::ModelInfoForTuning modelInfo;
  modelInfo.maxConvChannels1x1 = desc.maxConvChannels(1,1);
  modelInfo.maxConvChannels3x3 = desc.maxConvChannels(3,3);
  modelInfo.trunkNumChannels = desc.trunk.trunkNumChannels;
  modelInfo.midNumChannels = desc.trunk.midNumChannels;
  modelInfo.regularNumChannels = desc.trunk.regularNumChannels;
  modelInfo.gpoolNumChannels = desc.trunk.gpoolNumChannels;
  modelInfo.modelVersion = desc.modelVersion;
  findTransformerInfo(desc.trunk.blocks, modelInfo);
  return modelInfo;
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

namespace {
  bool selectHgemmCooperativeMatrixProperties(
    const VulkanDevice* device,
    HGemmCooperativeMatrixNCHWTuneParams& params
  ) {
    if(device == nullptr || device->info.cooperativeMatrixFeatures.cooperativeMatrix != VK_TRUE)
      return false;

    auto getProperties = device->info.cooperativeMatrixPropertiesFn;
    if(getProperties == nullptr)
      return false;

    uint32_t propertyCount = 0;
    VkResult result = getProperties(device->info.physicalDevice, &propertyCount, nullptr);
    if(result != VK_SUCCESS || propertyCount == 0)
      return false;

    vector<VkCooperativeMatrixPropertiesKHR> properties(propertyCount);
    for(VkCooperativeMatrixPropertiesKHR& property: properties) {
      property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
      property.pNext = nullptr;
    }
    result = getProperties(device->info.physicalDevice, &propertyCount, properties.data());
    if(result != VK_SUCCESS && result != VK_INCOMPLETE)
      return false;

    for(uint32_t i = 0; i < propertyCount; i++) {
      const VkCooperativeMatrixPropertiesKHR& property = properties[i];
      if(property.scope != VK_SCOPE_SUBGROUP_KHR ||
         property.AType != VK_COMPONENT_TYPE_FLOAT16_KHR ||
         property.BType != VK_COMPONENT_TYPE_FLOAT16_KHR ||
         (property.CType != VK_COMPONENT_TYPE_FLOAT16_KHR &&
          property.CType != VK_COMPONENT_TYPE_FLOAT32_KHR) ||
         property.CType != property.ResultType)
        continue;
      params.MWARP = static_cast<int>(property.MSize);
      params.NWARP = static_cast<int>(property.NSize);
      params.KDIM = static_cast<int>(property.KSize);
      params.CType = property.CType == VK_COMPONENT_TYPE_FLOAT16_KHR
        ? spec::HGemmCooperativeMatrixNCHWSpec::COMPONENT_TYPE_FLOAT16
        : spec::HGemmCooperativeMatrixNCHWSpec::COMPONENT_TYPE_FLOAT32;
      params.ResultType = property.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR
        ? spec::HGemmCooperativeMatrixNCHWSpec::COMPONENT_TYPE_FLOAT16
        : spec::HGemmCooperativeMatrixNCHWSpec::COMPONENT_TYPE_FLOAT32;
      params.subgroupSize = device->info.subgroupProperties.subgroupSize;
      if(!params.isValid()) {
        params.MWG = params.MWARP * 2;
        params.NWG = params.NWARP * 2;
        params.KWG = params.KDIM * 2;
        params.MWAVE = params.MWARP;
        params.NWAVE = params.NWARP;
        params.SB = 0;
      }
      if(params.isValid())
        return true;
    }
    return false;
  }

  bool selectHgemmCooperativeMatrixProperties(
    const VulkanDevice* device,
    HGemmCooperativeMatrixTuneParams& params
  ) {
    if(device == nullptr || device->info.cooperativeMatrixFeatures.cooperativeMatrix != VK_TRUE)
      return false;
    auto getProperties = device->info.cooperativeMatrixPropertiesFn;
    if(getProperties == nullptr)
      return false;

    uint32_t propertyCount = 0;
    VkResult result = getProperties(device->info.physicalDevice, &propertyCount, nullptr);
    if(result != VK_SUCCESS || propertyCount == 0)
      return false;
    vector<VkCooperativeMatrixPropertiesKHR> properties(propertyCount);
    for(VkCooperativeMatrixPropertiesKHR& property: properties) {
      property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
      property.pNext = nullptr;
    }
    result = getProperties(device->info.physicalDevice, &propertyCount, properties.data());
    if(result != VK_SUCCESS && result != VK_INCOMPLETE)
      return false;

    for(uint32_t i = 0; i < propertyCount; i++) {
      const VkCooperativeMatrixPropertiesKHR& property = properties[i];
      if(property.scope != VK_SCOPE_SUBGROUP_KHR ||
         property.AType != VK_COMPONENT_TYPE_FLOAT16_KHR ||
         property.BType != VK_COMPONENT_TYPE_FLOAT16_KHR ||
         property.CType != VK_COMPONENT_TYPE_FLOAT16_KHR ||
         property.ResultType != VK_COMPONENT_TYPE_FLOAT16_KHR)
        continue;
      params.MWARP = static_cast<int>(property.MSize);
      params.NWARP = static_cast<int>(property.NSize);
      params.KDIM = static_cast<int>(property.KSize);
      params.subgroupSize = device->info.subgroupProperties.subgroupSize;
      params.MWG = params.MWARP * 2;
      params.NWG = params.NWARP * 2;
      params.KWG = params.KDIM * 2;
      params.MWAVE = params.MWARP;
      params.NWAVE = params.NWARP;
      params.SA = 0;
      params.SB = 0;
      if(params.isValid())
        return true;
    }
    return false;
  }

  struct TuningContext {
    const VulkanDevice* device;
    int batchSize;
    int nnXLen;
    int nnYLen;
    const VulkanTuner::ModelInfoForTuning& modelInfo;
    bool full;
    Logger* logger;
  };

  struct GemmTuneCase {
    int inChannels;
    int outChannels;
    double weight;
  };

  vector<GemmTuneCase> getGemmTuneCases(
    const TuningContext& context,
    bool includeTransformerCases,
    bool use3x3
  ) {
    int maxConvChannels = use3x3 ? context.modelInfo.maxConvChannels3x3 : context.modelInfo.maxConvChannels1x1;
    maxConvChannels = std::max(context.modelInfo.trunkNumChannels, maxConvChannels);
    maxConvChannels = std::max(context.modelInfo.midNumChannels, maxConvChannels);
    maxConvChannels = std::max(context.modelInfo.regularNumChannels, maxConvChannels);
    maxConvChannels = std::max(context.modelInfo.gpoolNumChannels, maxConvChannels);

    vector<GemmTuneCase> cases = {
      {context.modelInfo.trunkNumChannels, context.modelInfo.midNumChannels, 0},
      {context.modelInfo.trunkNumChannels, context.modelInfo.midNumChannels, 1},
      {context.modelInfo.midNumChannels, context.modelInfo.trunkNumChannels, 1},
      {context.modelInfo.trunkNumChannels, context.modelInfo.regularNumChannels, 0.2},
      {context.modelInfo.trunkNumChannels, context.modelInfo.gpoolNumChannels, 0.2},
      {maxConvChannels, maxConvChannels, 1}
    };
    if(includeTransformerCases &&
       context.modelInfo.transformerHeadDim > 0 && context.modelInfo.transformerVHeadDim > 0 &&
       context.modelInfo.transformerNumHeads > 0 && context.modelInfo.transformerNumKVHeads > 0) {
      const int transformerQKC = context.modelInfo.transformerNumHeads * context.modelInfo.transformerHeadDim;
      const int transformerVC = context.modelInfo.transformerNumKVHeads * context.modelInfo.transformerVHeadDim;
      const int transformerFFNC = context.modelInfo.transformerFFNChannels;
      cases.push_back({context.modelInfo.midNumChannels, transformerQKC, 1});
      cases.push_back({transformerVC, context.modelInfo.midNumChannels, 1});
      cases.push_back({context.modelInfo.midNumChannels, transformerFFNC, 1});
      cases.push_back({transformerFFNC, context.modelInfo.midNumChannels, 1});
    }
    return cases;
  }

  size_t getWorkloadCaseCount(const string& tunerName, const TuningContext& context);

  vector<double> getWorkloadWeights(const string& tunerName, const TuningContext& context) {
    if(tunerName == "xgemm" || tunerName == "xgemm16") {
      vector<GemmTuneCase> cases = getGemmTuneCases(context, false, true);
      vector<double> weights;
      weights.reserve(cases.size());
      for(const GemmTuneCase& tuneCase: cases)
        weights.push_back(tuneCase.weight);
      return weights;
    }
    if(tunerName == "xgemmDirect" || tunerName == "hgemmCooperativeMatrixNCHW") {
      vector<GemmTuneCase> cases = getGemmTuneCases(context, true, false);
      vector<double> weights;
      weights.reserve(cases.size());
      for(const GemmTuneCase& tuneCase: cases)
        weights.push_back(tuneCase.weight);
      return weights;
    }
    if(tunerName == "hgemmCooperativeMatrix") {
      vector<GemmTuneCase> cases = getGemmTuneCases(context, false, true);
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

  size_t getWorkloadCaseCount(const string& tunerName, const TuningContext& context) {
    if(tunerName == "xgemmDirect" || tunerName == "hgemmCooperativeMatrixNCHW")
      return context.modelInfo.transformerHeadDim > 0 && context.modelInfo.transformerVHeadDim > 0 &&
             context.modelInfo.transformerNumHeads > 0 && context.modelInfo.transformerNumKVHeads > 0 ? 10 : 6;
    if(tunerName == "xgemm" || tunerName == "xgemm16" || tunerName == "hgemmCooperativeMatrix")
      return 6;
    if(tunerName == "transformerAttention")
      return 6;
    return 10;
  }

  vector<int> getTuningBatchSizes(const TuningContext& context) {
    return {std::max(1, context.batchSize)};
  }

  TuningMeasurementPlan makeMeasurementPlan(const string& tunerName, const TuningContext& context) {
    const bool isGemm = tunerName == "xgemmDirect" || tunerName == "xgemm" || tunerName == "xgemm16" ||
                        tunerName == "hgemmCooperativeMatrix" ||
                        tunerName == "hgemmCooperativeMatrixNCHW";
    const vector<int> batchSizes = getTuningBatchSizes(context);
    const bool direct = tunerName == "xgemmDirect" || tunerName == "hgemmCooperativeMatrixNCHW";
    const vector<GemmTuneCase> gemmCases = isGemm ?
      getGemmTuneCases(context, direct, !direct) : vector<GemmTuneCase>();
    const vector<double> workloadWeights = getWorkloadWeights(tunerName, context);
    const size_t workloadCaseCount = workloadWeights.size();
    if(isGemm) {
      const size_t totalRuns = 3 * workloadCaseCount * batchSizes.size();
      const double tolerance = tunerName == "xgemmDirect" ? 0.01 :
                               (tunerName == "xgemm" || tunerName == "xgemm16") ? 0.005 : 0.002;
      return {
        tunerName, totalRuns, std::min<size_t>(1, totalRuns - 1), tolerance, tolerance * 5.0,
        batchSizes, gemmCases, workloadWeights
      };
    }
    if(tunerName == "pointwise" || tunerName == "transformerRMSNorm" || tunerName == "spatialRMSNorm")
      return {tunerName, 20, 1, 0.05, 0.25, batchSizes, {}, workloadWeights};
    if(tunerName == "transformerAttention")
      return {tunerName, 12, 1, 0.005, 0.025, batchSizes, {}, workloadWeights};
    return {tunerName, 20, 1, 0.005, 0.025, batchSizes, {}, workloadWeights};
  }

  bool usesCpuReference(const string& tunerName) {
    return tunerName != "conv3x3InputTransform" && tunerName != "conv3x3OutputTransform" &&
           tunerName != "conv5x5InputTransform" && tunerName != "conv5x5OutputTransform";
  }

  bool validateReadback(
    const vector<float>& reference,
    const vector<float>& values,
    const TuningMeasurementPlan& plan,
    double& errorProp,
    string& error
  ) {
    if(reference.size() != values.size()) {
      errorProp = 1.0;
      error = "candidate output size mismatch: expected " + to_string(reference.size()) +
        ", got " + to_string(values.size());
      return false;
    }
    for(size_t i = 0; i < reference.size(); i++) {
      if(!isfinite(reference[i]) || !isfinite(values[i])) {
        errorProp = 1.0;
        error = "candidate output contains a non-finite value";
        return false;
      }
    }
    errorProp = VulkanTuner::computeErrorProp(reference, values);
    if(!isfinite(errorProp) || errorProp > std::min(0.5, 5.0 * plan.errorTolerance)) {
      errorProp = 1.0;
      error = "candidate readback exceeded the hard error cutoff";
      return false;
    }
    return true;
  }

  template<typename Setter>
  void addCandidates(vector<VulkanTuneParams>& configs, const vector<int>& values, const Setter& setter) {
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
    if(tunerName == "xgemmDirect") {
      add("WGD", config.xgemmDirect.WGD);
      add("MDIMCD", config.xgemmDirect.MDIMCD);
      add("NDIMCD", config.xgemmDirect.NDIMCD);
      add("MDIMAD", config.xgemmDirect.MDIMAD);
      add("NDIMBD", config.xgemmDirect.NDIMBD);
      add("KWID", config.xgemmDirect.KWID);
      add("PADA", config.xgemmDirect.PADA);
      add("PADB", config.xgemmDirect.PADB);
    }
    else if(tunerName == "xgemm" || tunerName == "xgemm16") {
      const XgemmTuneParams& params = tunerName == "xgemm" ? config.xgemm : config.xgemm16;
      add("MDIMC", params.MDIMC);
      add("NDIMC", params.NDIMC);
      add("MWG", params.MWG);
      add("NWG", params.NWG);
      add("KWG", params.KWG);
      add("MDIMA", params.MDIMA);
      add("NDIMB", params.NDIMB);
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
      add("SA", config.hgemmCooperativeMatrix.SA);
      add("SB", config.hgemmCooperativeMatrix.SB);
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
      add("CType", config.hgemmCooperativeMatrixNCHW.CType);
      add("ResultType", config.hgemmCooperativeMatrixNCHW.ResultType);
      add("SB", config.hgemmCooperativeMatrixNCHW.SB);
      add("VWM", config.hgemmCooperativeMatrixNCHW.VWM);
      add("VWN", config.hgemmCooperativeMatrixNCHW.VWN);
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
        add("inputTransformLocalXSize", conv.inputTransformLocalXSize);
        add("inputTransformLocalYSize", conv.inputTransformLocalYSize);
      }
      else {
        add("outputTransformLocalXSize", conv.outputTransformLocalXSize);
        add("outputTransformLocalYSize", conv.outputTransformLocalYSize);
        add("outputTransformLocalZSize", conv.outputTransformLocalZSize);
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
    else if(tunerName == "transformerAttention") {
      add("ATTN_BLOCK_Q", config.transformer.ATTN_BLOCK_Q);
      add("ATTN_BLOCK_KV", config.transformer.ATTN_BLOCK_KV);
      add("Q_PER_THREAD", config.transformer.Q_PER_THREAD);
      add("USE_TILED_ATTN", config.transformer.USE_TILED_ATTN);
    }
    else if(tunerName == "transformerRMSNorm") {
      add("WG_C_SIZE", config.rmsNorm.WG_C_SIZE);
      add("WG_XY_SIZE", config.rmsNorm.WG_XY_SIZE);
      add("C_PER_THREAD", config.rmsNorm.C_PER_THREAD);
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

  string describeTuningPipelines(const vector<const Pipeline*>& pipelines, const string& tunerName) {
    if(pipelines.empty())
      return tunerName;
    ostringstream out;
    for(const Pipeline* pipeline: pipelines) {
      if(out.tellp() > 0)
        out << ",";
      out << pipeline->name;
    }
    return out.str();
  }

  void logTuningResult(
    const TuningContext& context,
    size_t candidateIndex,
    size_t totalCandidates,
    const vector<const Pipeline*>& pipelines,
    const VulkanTuneParams& candidate,
    const string& tunerName,
    double callsPerSecond,
    double errorProp,
    bool isBest
  ) {
    const string pipelineNames = describeTuningPipelines(pipelines, tunerName);
    const string params = describeTuningParams(tunerName, candidate);
    ostringstream out;
    out << "Tuning " << pipelineNames << " " << (isBest ? "* " : "  ")
        << candidateIndex << "/" << totalCandidates
        << (candidateIndex == 0 ? " (reference)" : "")
        << " Calls/sec " << callsPerSecond
        << " ErrorProp " << errorProp
        << " " << params;
    writeTuningLog(context, out.str());
  }

  void logTuningFailure(
    const TuningContext& context,
    size_t candidateIndex,
    size_t totalCandidates,
    const vector<const Pipeline*>& pipelines,
    const string& tunerName,
    const string& error
  ) {
    const string pipelineNames = describeTuningPipelines(pipelines, tunerName);
    writeTuningLog(
      context,
      "Tuning " + pipelineNames + " " + to_string(candidateIndex) + "/" + to_string(totalCandidates) + " failed: " + error
    );
  }

  class VulkanTimestampTimer {
   public:
    explicit VulkanTimestampTimer(const VulkanDevice* device)
    : device(device), timestampPeriod(device->info.properties.limits.timestampPeriod) {}

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
    ) const {
      errorProp = numeric_limits<double>::quiet_NaN();
      if(!isUsable()) {
        error = "compute timestamps are not supported";
        return false;
      }
      if(pipelines.empty()) {
        error = "no pipeline was created";
        return false;
      }

      // Measure the same logical GEMMs for every tile/precision choice. Keep
      // each case separate so its output is checked before the next case.
      if(plan.gemmCases.size() > 1) {
        readback.clear();
        double weightedSeconds = 0.0;
        double totalWeight = 0.0;
        for(const GemmTuneCase& gemmCase: plan.gemmCases) {
          TuningMeasurementPlan casePlan = plan;
          casePlan.gemmCases = {gemmCase};
          casePlan.totalRuns = 4;
          casePlan.warmupRuns = 1;
          casePlan.workloadWeights = {1.0};
          vector<float> output;
          double rate = 0.0;
          if(!measure(pipelines, config, context, casePlan, rate, output, errorProp, error, cpuReference))
            return false;
          readback.insert(readback.end(), output.begin(), output.end());
          weightedSeconds += gemmCase.weight / rate;
          totalWeight += gemmCase.weight;
        }
        callsPerSecond = totalWeight / weightedSeconds;
        return true;
      }

      uint32_t descriptorCount = 0;
      for(const Pipeline* pipeline: pipelines)
        descriptorCount += pipeline->bindingCount;
      if(descriptorCount == 0) {
        error = "pipeline has no descriptor bindings";
        return false;
      }

      VkResult result = VK_SUCCESS;
      const size_t batchSize = static_cast<size_t>(std::max(1, context.batchSize));
      const size_t xySize = static_cast<size_t>(std::max(1, context.nnXLen * context.nnYLen));
      const XgemmTuneParams& xgemmParams =
        config.vulkan.shouldUseFP16Compute ? config.xgemm16 : config.xgemm;
      const size_t maxChannels = static_cast<size_t>(std::max({
        1, context.modelInfo.trunkNumChannels, context.modelInfo.midNumChannels,
        context.modelInfo.regularNumChannels, context.modelInfo.maxConvChannels1x1,
        context.modelInfo.maxConvChannels3x3, context.modelInfo.gpoolNumChannels,
        context.modelInfo.transformerFFNChannels,
        context.modelInfo.transformerNumHeads * context.modelInfo.transformerHeadDim,
        context.modelInfo.transformerNumKVHeads * context.modelInfo.transformerVHeadDim
      }));
      const bool isGemm = !plan.gemmCases.empty();
      const bool directGemm = plan.kernelName == "xgemmDirect" || plan.kernelName == "hgemmCooperativeMatrixNCHW";
      const bool cooperative = plan.kernelName == "hgemmCooperativeMatrix" || plan.kernelName == "hgemmCooperativeMatrixNCHW";
      const int tilesX = (context.nnXLen + config.conv3x3.outTileXSize - 1) / config.conv3x3.outTileXSize;
      const int tilesY = (context.nnYLen + config.conv3x3.outTileYSize - 1) / config.conv3x3.outTileYSize;
      const int logicalM = directGemm ? static_cast<int>(xySize) : static_cast<int>(batchSize) * tilesX * tilesY;
      const int logicalN = isGemm ? std::max(1, plan.gemmCases[0].outChannels) : static_cast<int>(maxChannels);
      const int logicalK = isGemm ? std::max(1, plan.gemmCases[0].inChannels) : static_cast<int>(maxChannels);
      const int gemmBatch = directGemm ? static_cast<int>(batchSize) :
        config.conv3x3.inTileXSize * config.conv3x3.inTileYSize;
      int gemmM = logicalM;
      int gemmN = logicalN;
      int gemmK = logicalK;
      if(cooperative && directGemm) {
        gemmM = vk_helper::roundUpToMultipleInt(logicalM, std::max(16, config.hgemmCooperativeMatrixNCHW.MWARP));
        const int align = config.hgemmCooperativeMatrixNCHW.getRequiredCDivisor();
        gemmN = vk_helper::roundUpToMultipleInt(logicalN, align);
        gemmK = vk_helper::roundUpToMultipleInt(logicalK, align);
      }
      else if(!directGemm) {
        gemmM = vk_helper::roundUpToMultipleInt(logicalM, cooperative ? config.hgemmCooperativeMatrix.MWG : xgemmParams.MWG);
        gemmN = vk_helper::roundUpToMultipleInt(logicalN, cooperative ? config.hgemmCooperativeMatrix.NWG : xgemmParams.NWG);
        gemmK = vk_helper::roundUpToMultipleInt(logicalK, cooperative ? config.hgemmCooperativeMatrix.KWG : xgemmParams.KWG);
      }
      const size_t maxTiles = batchSize * ((context.nnXLen + 1) / 2) * ((context.nnYLen + 1) / 2);
      const size_t paddedTiles = vk_helper::roundUpToMultiple(maxTiles, static_cast<size_t>(xgemmParams.MWG));
      const size_t paddedChannels = vk_helper::roundUpToMultiple(maxChannels, static_cast<size_t>(std::max(xgemmParams.KWG, xgemmParams.NWG)));
      const size_t scratchElements = isGemm ? static_cast<size_t>(gemmBatch) * std::max({
        static_cast<size_t>(gemmM) * gemmK, static_cast<size_t>(gemmN) * gemmK, static_cast<size_t>(gemmM) * gemmN
      }) : std::max(batchSize * maxChannels * xySize, paddedTiles * paddedChannels * 36);
      const size_t scratchBytes = vk_helper::roundUpToMultiple(std::max<size_t>(scratchElements, 4), size_t(4)) * sizeof(float);
      vector<VulkanBuffer*> tuningBuffers;
      VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
      VkQueryPool queryPool = VK_NULL_HANDLE;
      VkFence fence = VK_NULL_HANDLE;
      VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
      const auto cleanup = [&]() noexcept {
        if(commandBuffer != VK_NULL_HANDLE) {
          vkFreeCommandBuffers(device->device, device->commandPool, 1, &commandBuffer);
          commandBuffer = VK_NULL_HANDLE;
        }
        if(fence != VK_NULL_HANDLE) {
          vkDestroyFence(device->device, fence, nullptr);
          fence = VK_NULL_HANDLE;
        }
        if(queryPool != VK_NULL_HANDLE) {
          vkDestroyQueryPool(device->device, queryPool, nullptr);
          queryPool = VK_NULL_HANDLE;
        }
        if(descriptorPool != VK_NULL_HANDLE) {
          vkDestroyDescriptorPool(device->device, descriptorPool, nullptr);
          descriptorPool = VK_NULL_HANDLE;
        }
        for(VulkanBuffer* buffer: tuningBuffers)
          vk_helper::releaseVulkanBuffer(device, buffer);
        tuningBuffers.clear();
      };
      const auto cleanupGuard = makeScopeGuard(cleanup);

      const auto outputBinding = [](const Pipeline* pipeline) -> uint32_t {
        const string& name = pipeline->name;
        if(name.find("gemm") != string::npos || name.find("transformer_swiglu") == 0 ||
           name.find("transformer_spatial_rms_norm_sum_sq") == 0)
          return 2;
        if(name.find("transformer_scale_dot_product") == 0)
          return 3;
        if(name.find("add_pointwise") == 0 || name.find("add_channel_bias_nchw") == 0)
          return 0;
        return 1;
      };
      const auto halfBinding = [&](const Pipeline* pipeline, uint32_t binding) {
        const string& name = pipeline->name;
        if(name.find("hgemm_cooperative_matrix") == 0)
          return true;
        if(name.find("fp32") != string::npos)
          return false;
        if(name.find("global_pooling_channels") == 0)
          return binding == 0 || binding == 2;
        if(name.find("value_head_pool_channels") == 0 || name.find("sum_channels") == 0)
          return binding == 0;
        if(name.find("add_channel_bias_nchw") == 0)
          return binding == 0;
        if(name.find("transformer_rms_norm") == 0 || name.find("transformer_spatial_rms_norm_apply") == 0)
          return binding == 0 || binding == 1 || binding == 4;
        if(name.find("transformer_spatial_rms_norm_sum_sq") == 0)
          return binding < 2;
        if(name.find("transformer_spatial_rms_norm_reduce") == 0)
          return false;
        return config.vulkan.shouldUseFP16Storage;
      };
      vector<float> gemmInput, gemmFilter;
      vector<vector<float>> hostFloatBuffers;
      tuningBuffers.reserve(descriptorCount);
      hostFloatBuffers.reserve(descriptorCount);
      for(const Pipeline* pipeline: pipelines) {
        for(uint32_t binding = 0; binding < pipeline->bindingCount; binding++) {
          vector<float> data(scratchBytes / sizeof(float), 0.0f);
          Rand rand("VulkanTunerInput:" + to_string(binding));
          if(binding != outputBinding(pipeline)) {
            if(isGemm && binding < 2) {
              const int width = binding == 0 ? gemmM : gemmN;
              const int logicalWidth = binding == 0 ? logicalM : logicalN;
              const int batches = directGemm && binding == 1 ? 1 : gemmBatch;
              for(int n = 0; n < batches; n++)
                for(int k = 0; k < logicalK; k++)
                  for(int x = 0; x < logicalWidth; x++)
                    data[(static_cast<size_t>(n) * gemmK + k) * width + x] =
                      static_cast<float>(rand.nextDouble() - 0.5) / sqrtf(static_cast<float>(logicalK));
              if(cpuReference != nullptr) {
                if(binding == 0) gemmInput = data;
                else gemmFilter = data;
              }
            }
            else {
              for(float& value: data)
                value = static_cast<float>(rand.nextDouble());
              // Masks and their sums describe a fully valid board.
              const string& name = pipeline->name;
              const bool mask =
                (name.find("global_pooling_channels") == 0 && binding == 2) ||
                (name.find("transformer_scale_dot_product") == 0 && binding == 4) ||
                ((name.find("transformer_rms_norm") == 0 || name.find("transformer_spatial_rms_norm_apply") == 0) && binding == 4) ||
                (name.find("transformer_spatial_rms_norm_sum_sq") == 0 && binding == 1);
              const bool maskSum =
                (name.find("global_pooling_channels") == 0 && binding == 3) ||
                (name.find("value_head_pool_channels") == 0 && binding == 2) ||
                (name.find("transformer_spatial_rms_norm_apply") == 0 && binding == 5);
              if(mask || maskSum)
                std::fill(data.begin(), data.end(), mask ? 1.0f : static_cast<float>(xySize));
            }
          }
          hostFloatBuffers.push_back(data);
          vector<half_t> halfData;
          const void* initialData = data.data();
          size_t initialBytes = scratchBytes;
          if(halfBinding(pipeline, binding)) {
            halfData.resize(data.size());
            for(size_t j = 0; j < data.size(); j++)
              halfData[j] = half_float::half_cast<half_t>(data[j]);
            initialData = halfData.data();
            initialBytes = halfData.size() * sizeof(half_t);
          }
          VulkanBuffer* buffer = vk_helper::createDeviceBuffer(device, scratchBytes, false, &result);
          if(result != VK_SUCCESS || buffer == nullptr) {
            error = "could not allocate tuning buffer: " + vk_helper::vkErrorToString(result);
            return false;
          }
          tuningBuffers.push_back(buffer);
          vk_helper::copyHostToDeviceBuffer(device, initialData, buffer, initialBytes, true, &result);
          if(result != VK_SUCCESS) {
            error = "could not initialize tuning buffer: " + vk_helper::vkErrorToString(result);
            return false;
          }
        }
      }
      if(isGemm && cpuReference != nullptr) {
        // The reference uses logical float inputs, before half quantization.
        for(int n = 0; n < gemmBatch; n++)
          for(int y = 0; y < logicalN; y++)
            for(int x = 0; x < logicalM; x++) {
              double sum = 0.0;
              for(int k = 0; k < logicalK; k++)
                sum += static_cast<double>(gemmInput[(static_cast<size_t>(n) * gemmK + k) * gemmM + x]) *
                  gemmFilter[(static_cast<size_t>(directGemm ? 0 : n) * gemmK + k) * gemmN + y];
              cpuReference->push_back(static_cast<float>(sum));
            }
      }

      const auto appendCpuReference = [&](const Pipeline* pipeline, size_t firstBuffer) -> bool {
        const string& name = pipeline->name;
        const auto& buffer = [&](size_t binding) -> const vector<float>& { return hostFloatBuffers[firstBuffer + binding]; };
        const int cpuBatchSize = std::max(1, context.batchSize);
        const int cpuXYSize = std::max(1, context.nnXLen * context.nnYLen);
        const int cpuChannels = std::max(1, context.modelInfo.trunkNumChannels);

        if(name.find("transformer_spatial_rms_norm_sum_sq") == 0 ||
           name.find("transformer_spatial_rms_norm_reduce") == 0)
          return true;

        if(name.find("global_pooling_channels") == 0) {
          const vector<float>& input = buffer(0);
          const vector<float>& mask = buffer(2);
          const vector<float>& maskSum = buffer(3);
          const int channels = std::max(1, context.modelInfo.gpoolNumChannels);
          for(int n = 0; n < cpuBatchSize; n++) {
            const float divisor = maskSum[n];
            vector<float> means(channels);
            vector<float> maxima(channels);
            for(int c = 0; c < channels; c++) {
              float sum = 0.0f;
              float maximum = -1.0f;
              for(int xy = 0; xy < cpuXYSize; xy++) {
                const float value = input[(n * channels + c) * cpuXYSize + xy];
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
                sum += input[(n * channels + c) * cpuXYSize + xy];
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
          const size_t count = static_cast<size_t>(cpuBatchSize) * cpuChannels * cpuXYSize;
          for(size_t i = 0; i < count; i++)
            cpuReference->push_back(accum[i] + plan.totalRuns * value[i]);
          return true;
        }
        if(name.find("transformer_swiglu") == 0) {
          const vector<float>& input = buffer(0);
          const vector<float>& gate = buffer(1);
          const size_t count = static_cast<size_t>(cpuBatchSize) *
            std::max(cpuChannels, context.modelInfo.transformerFFNChannels) * cpuXYSize;
          for(size_t i = 0; i < count; i++)
            cpuReference->push_back(input[i] / (1.0f + expf(-input[i])) * gate[i]);
          return true;
        }
        if(name.find("add_channel_bias_nchw") == 0) {
          const vector<float>& accum = buffer(0);
          const vector<float>& bias = buffer(1);
          const size_t count = static_cast<size_t>(cpuBatchSize) * cpuChannels * cpuXYSize;
          for(size_t i = 0; i < count; i++)
            cpuReference->push_back(accum[i] + plan.totalRuns * bias[i / cpuXYSize]);
          return true;
        }
        if(name.find("transformer_scale_dot_product") == 0) {
          const vector<float>& query = buffer(0);
          const vector<float>& key = buffer(1);
          const vector<float>& value = buffer(2);
          const vector<float>& mask = buffer(4);
          const int heads = std::max(1, context.modelInfo.transformerNumHeads);
          const int kvHeads = std::max(1, context.modelInfo.transformerNumKVHeads);
          const int headDim = std::max(1, context.modelInfo.transformerHeadDim);
          const int vHeadDim = std::max(1, context.modelInfo.transformerVHeadDim);
          const float scale = 1.0f / sqrtf(static_cast<float>(headDim));
          for(int bh = 0; bh < cpuBatchSize * heads; bh++) {
            const int n = bh / heads;
            const int kvBase = n * kvHeads + (bh % heads) / (heads / kvHeads);
            vector<float> output(vHeadDim * cpuXYSize, 0.0f);
            for(int qPos = 0; qPos < cpuXYSize; qPos++) {
              if(mask[n * cpuXYSize + qPos] == 0.0f) {
                continue;
              }
              float runningMax = -1e30f;
              float runningSum = 0.0f;
              vector<float> accum(vHeadDim, 0.0f);
              for(int kPos = 0; kPos < cpuXYSize; kPos++) {
                if(mask[n * cpuXYSize + kPos] == 0.0f)
                  continue;
                float dot = 0.0f;
                for(int d = 0; d < headDim; d++)
                  dot += query[(bh * headDim + d) * cpuXYSize + qPos] *
                    key[(kvBase * headDim + d) * cpuXYSize + kPos];
                dot *= scale;
                const float nextMax = std::max(runningMax, dot);
                const float oldWeight = expf(runningMax - nextMax);
                const float weight = expf(dot - nextMax);
                for(int d = 0; d < vHeadDim; d++)
                  accum[d] = accum[d] * oldWeight + weight * value[(kvBase * vHeadDim + d) * cpuXYSize + kPos];
                runningSum = runningSum * oldWeight + weight;
                runningMax = nextMax;
              }
              for(int d = 0; d < vHeadDim; d++)
                output[d * cpuXYSize + qPos] = runningSum > 0.0f ? accum[d] / runningSum : 0.0f;
            }
            cpuReference->insert(cpuReference->end(), output.begin(), output.end());
          }
          return true;
        }
        if(name.find("transformer_rms_norm") == 0) {
          const vector<float>& input = buffer(0);
          const vector<float>& gamma = buffer(2);
          const vector<float>& beta = buffer(3);
          const vector<float>& mask = buffer(4);
          for(int n = 0; n < cpuBatchSize; n++) {
            vector<float> rms(cpuXYSize);
            for(int xy = 0; xy < cpuXYSize; xy++) {
              const float maskValue = mask[n * cpuXYSize + xy];
              float sumSq = 0.0f;
              for(int c = 0; c < cpuChannels; c++) {
                const float value = input[(n * cpuChannels + c) * cpuXYSize + xy] * maskValue;
                sumSq += value * value;
              }
              rms[xy] = 1.0f / sqrtf(sumSq / cpuChannels + 1e-6f);
            }
            for(int c = 0; c < cpuChannels; c++)
              for(int xy = 0; xy < cpuXYSize; xy++)
                cpuReference->push_back((input[(n * cpuChannels + c) * cpuXYSize + xy] * rms[xy] * gamma[c] + beta[c]) *
                  mask[n * cpuXYSize + xy]);
          }
          return true;
        }
        if(name.find("transformer_spatial_rms_norm_apply") == 0) {
          const vector<float>& input = buffer(0);
          const vector<float>& gamma = buffer(2);
          const vector<float>& beta = buffer(3);
          const vector<float>& mask = buffer(4);
          const vector<float>& maskSum = buffer(5);
          for(int n = 0; n < cpuBatchSize; n++) {
            float sumSq = 0.0f;
            for(int c = 0; c < cpuChannels; c++)
              for(int xy = 0; xy < cpuXYSize; xy++) {
                const float value = input[(n * cpuChannels + c) * cpuXYSize + xy] * mask[n * cpuXYSize + xy];
                sumSq += value * value;
              }
            const float rms = 1.0f / sqrtf(sumSq / (maskSum[n] * cpuChannels) + 1e-6f);
            for(int c = 0; c < cpuChannels; c++)
              for(int xy = 0; xy < cpuXYSize; xy++)
                cpuReference->push_back((input[(n * cpuChannels + c) * cpuXYSize + xy] * rms * gamma[c] + beta[c]) *
                  mask[n * cpuXYSize + xy]);
          }
          return true;
        }
        error = "no CPU reference implementation for " + name;
        return false;
      };
      if(!isGemm && cpuReference != nullptr) {
        size_t firstBuffer = 0;
        for(const Pipeline* pipeline: pipelines) {
          if(!appendCpuReference(pipeline, firstBuffer))
            return false;
          firstBuffer += pipeline->bindingCount;
        }
      }

      VkDescriptorPoolSize poolSize = {};
      poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      poolSize.descriptorCount = descriptorCount;
      VkDescriptorPoolCreateInfo descriptorPoolInfo = {};
      descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      descriptorPoolInfo.poolSizeCount = 1;
      descriptorPoolInfo.pPoolSizes = &poolSize;
      descriptorPoolInfo.maxSets = static_cast<uint32_t>(pipelines.size());
      result = vkCreateDescriptorPool(device->device, &descriptorPoolInfo, nullptr, &descriptorPool);
      if(result != VK_SUCCESS) {
        error = "could not create tuning descriptor pool: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }

      vector<VkDescriptorSet> descriptorSets;
      descriptorSets.reserve(pipelines.size());
      size_t bufferIndex = 0;
      for(const Pipeline* pipeline: pipelines) {
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
          VulkanBuffer* buffer = tuningBuffers[bufferIndex];
          if(plan.kernelName == "spatialRMSNorm") {
            if(pipeline->name.find("transformer_spatial_rms_norm_reduce") == 0 && binding == 0)
              buffer = tuningBuffers[2];
            if(pipeline->name.find("transformer_spatial_rms_norm_apply") == 0 && binding == 6)
              buffer = tuningBuffers[4];
          }
          writes.push_back(vk_helper::writeDescriptorSetBuffer(descriptorSet, binding, buffer));
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

      VkQueryPoolCreateInfo queryPoolInfo = {};
      queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
      queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
      queryPoolInfo.queryCount = static_cast<uint32_t>(2 * plan.timedRuns());
      result = vkCreateQueryPool(device->device, &queryPoolInfo, nullptr, &queryPool);
      if(result != VK_SUCCESS) {
        error = "could not create tuning query pool: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }

      VkFenceCreateInfo fenceInfo = {};
      fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      result = vkCreateFence(device->device, &fenceInfo, nullptr, &fence);
      if(result != VK_SUCCESS) {
        error = "could not create tuning fence: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }
      commandBuffer = vk_helper::allocateCommandBuffer(device, &result);
      if(result != VK_SUCCESS) {
        error = "could not allocate tuning command buffer: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }
      result = vk_helper::beginCommandBuffer(commandBuffer);
      if(result != VK_SUCCESS) {
        error = "could not begin tuning command buffer: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }

      const auto recordPipeline = [&](
        const Pipeline* pipeline,
        VkDescriptorSet descriptorSet,
        int runBatchSize
      ) {
        const int batchSize = std::max(1, runBatchSize);
        const int xySize = std::max(1, context.nnXLen * context.nnYLen);
        const int channels = std::max(1, context.modelInfo.trunkNumChannels);
        const auto dispatch = [&](uint32_t x, uint32_t y = 1, uint32_t z = 1) {
          vkCmdDispatch(commandBuffer, std::max(1u, x), std::max(1u, y), std::max(1u, z));
        };
        const auto push = [&](const auto& params) {
          vkCmdPushConstants(commandBuffer, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
        };
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
        vkCmdBindDescriptorSets(
          commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, 1, &descriptorSet, 0, nullptr
        );

        if(pipeline->name.find("hgemm_cooperative_matrix_nchw") == 0) {
          vk_shader::push::HGemmCooperativeMatrixNCHWParams params = {gemmK, gemmM, gemmN};
          push(params);
          dispatch(
            (gemmM + config.hgemmCooperativeMatrixNCHW.MWG - 1) / config.hgemmCooperativeMatrixNCHW.MWG,
            (gemmN + config.hgemmCooperativeMatrixNCHW.NWG - 1) / config.hgemmCooperativeMatrixNCHW.NWG,
            gemmBatch
          );
        }
        else if(pipeline->name.find("hgemm_cooperative_matrix_") == 0) {
          vk_shader::push::HGemmCooperativeMatrixParams params = {gemmM, gemmN, gemmK};
          push(params);
          dispatch(gemmM / config.hgemmCooperativeMatrix.MWG, gemmN / config.hgemmCooperativeMatrix.NWG, gemmBatch);
        }
        else if(pipeline->name.find("xgemm_batched") == 0) {
          vk_shader::push::XGEMMBatchedParams params = {
            static_cast<uint32_t>(gemmM), static_cast<uint32_t>(gemmN), static_cast<uint32_t>(gemmK),
            static_cast<uint32_t>(gemmM), static_cast<uint32_t>(gemmK),
            static_cast<uint32_t>(gemmN), static_cast<uint32_t>(gemmK),
            static_cast<uint32_t>(gemmM), static_cast<uint32_t>(gemmN)
          };
          push(params);
          dispatch(gemmM / xgemmParams.MWG, gemmN / xgemmParams.NWG, gemmBatch);
        }
        else if(pipeline->name.find("xgemm_strided_batched") == 0) {
          vk_shader::push::XgemmStridedBatchedFp32Params params = {
            static_cast<uint32_t>(gemmM), static_cast<uint32_t>(gemmN), static_cast<uint32_t>(gemmK),
            static_cast<uint32_t>(gemmM), static_cast<uint32_t>(gemmM * gemmK),
            static_cast<uint32_t>(gemmN), 0,
            static_cast<uint32_t>(gemmM), static_cast<uint32_t>(gemmM * gemmN), 0
          };
          push(params);
          dispatch(
            (gemmM + config.xgemmDirect.WGD - 1) / config.xgemmDirect.WGD,
            (gemmN + config.xgemmDirect.WGD - 1) / config.xgemmDirect.WGD, gemmBatch
          );
        }
        else if(pipeline->name.find("winograd_input_transform") == 0) {
          const vk_shader::tune::ConvTuneParams& convParams =
            pipeline->name.find("5x5") != string::npos ? config.conv5x5 : config.conv3x3;
          const int outTile = convParams.outTileXSize;
          const int tilesX = (context.nnXLen + outTile - 1) / outTile;
          const int tilesY = (context.nnYLen + outTile - 1) / outTile;
          const int paddedTiles = vk_helper::roundUpToMultipleInt(batchSize * tilesX * tilesY, xgemmParams.MWG);
          const int paddedChannels = vk_helper::roundUpToMultipleInt(channels, xgemmParams.KWG);
          vk_shader::push::WinogradInputTransformParams params = {
            batchSize,context.nnXLen,context.nnYLen,tilesX,tilesY,channels,paddedChannels,paddedTiles,xySize
          };
          push(params);
          dispatch(
            static_cast<uint32_t>((params.ntxtySizePadded + pipeline->localSizeX - 1) / pipeline->localSizeX),
            static_cast<uint32_t>((channels + pipeline->localSizeY - 1) / pipeline->localSizeY)
          );
        }
        else if(pipeline->name.find("winograd_output_transform") == 0) {
          const vk_shader::tune::ConvTuneParams& convParams =
            pipeline->name.find("5x5") != string::npos ? config.conv5x5 : config.conv3x3;
          const int outTile = convParams.outTileXSize;
          const int tilesX = (context.nnXLen + outTile - 1) / outTile;
          const int tilesY = (context.nnYLen + outTile - 1) / outTile;
          const int paddedTiles = vk_helper::roundUpToMultipleInt(batchSize * tilesX * tilesY, xgemmParams.MWG);
          const int paddedChannels = vk_helper::roundUpToMultipleInt(channels, xgemmParams.NWG);
          vk_shader::push::WinogradOutputTransformParams params = {
            batchSize,context.nnYLen,context.nnXLen,tilesY,tilesX,channels,paddedChannels,paddedTiles,xySize
          };
          push(params);
          dispatch(
            static_cast<uint32_t>((tilesX + pipeline->localSizeX - 1) / pipeline->localSizeX),
            static_cast<uint32_t>((tilesY + pipeline->localSizeY - 1) / pipeline->localSizeY),
            static_cast<uint32_t>((batchSize * channels + pipeline->localSizeZ - 1) / pipeline->localSizeZ)
          );
        }
        else if(pipeline->name.find("global_pooling_channels") == 0) {
          const int gpoolChannels = std::max(1, context.modelInfo.gpoolNumChannels);
          vk_shader::push::GlobalPoolingChannelsParams params = {batchSize,gpoolChannels,xySize};
          push(params);
          dispatch(
            1,
            static_cast<uint32_t>((gpoolChannels + pipeline->localSizeY - 1) / pipeline->localSizeY),
            static_cast<uint32_t>((batchSize + pipeline->localSizeZ - 1) / pipeline->localSizeZ)
          );
        }
        else if(pipeline->name.find("value_head_pool_channels") == 0) {
          const int gpoolChannels = std::max(1, context.modelInfo.gpoolNumChannels);
          vk_shader::push::ValueHeadPoolingChannelsParams params = {batchSize,gpoolChannels,xySize};
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
            static_cast<uint32_t>(xySize)
          };
          push(params);
          dispatch(1, 1, static_cast<uint32_t>((batchSize + pipeline->localSizeZ - 1) / pipeline->localSizeZ));
        }
        else if(pipeline->name.find("add_pointwise") == 0) {
          vk_shader::push::AddPointWiseParams params = {static_cast<uint32_t>(batchSize * channels * xySize)};
          push(params);
          dispatch((params.size + config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX - 1) /
                   (config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX));
        }
        else if(pipeline->name.find("add_channel_bias_nchw") == 0) {
          vk_shader::push::AddChannelBiasNCHWParams params = {static_cast<uint32_t>(batchSize * channels),static_cast<uint32_t>(xySize)};
          push(params);
          dispatch(
            (params.xySize + config.addChannelBiases.XY_ELTS_PER_THREAD * pipeline->localSizeX - 1) /
              (config.addChannelBiases.XY_ELTS_PER_THREAD * pipeline->localSizeX),
            (params.ncSize + config.addChannelBiases.NC_ELTS_PER_THREAD - 1) / config.addChannelBiases.NC_ELTS_PER_THREAD
          );
        }
        else if(pipeline->name.find("transformer_scale_dot_product") == 0) {
          const int heads = std::max(1, context.modelInfo.transformerNumHeads);
          const int kvHeads = std::max(1, context.modelInfo.transformerNumKVHeads);
          vk_shader::push::ScaleDotProductPushParam params = {xySize,heads,kvHeads,1.0f / sqrtf((float)std::max(1, context.modelInfo.transformerHeadDim))};
          push(params);
          if(config.transformer.USE_TILED_ATTN && pipeline->name.find("naive") == string::npos)
            dispatch((xySize + config.transformer.ATTN_BLOCK_Q * config.transformer.Q_PER_THREAD - 1) /
                       (config.transformer.ATTN_BLOCK_Q * config.transformer.Q_PER_THREAD), static_cast<uint32_t>(batchSize * heads));
          else
            dispatch((xySize + pipeline->localSizeX - 1) / pipeline->localSizeX, static_cast<uint32_t>(batchSize * heads));
        }
        else if(pipeline->name.find("transformer_rms_norm") == 0) {
          vk_shader::push::TransformerRMSNormPushParams params = {batchSize,channels,xySize,1e-6f};
          push(params);
          dispatch(
            static_cast<uint32_t>((xySize + config.rmsNorm.WG_XY_SIZE - 1) / config.rmsNorm.WG_XY_SIZE),
            static_cast<uint32_t>(batchSize)
          );
        }
        else if(pipeline->name.find("transformer_swiglu") == 0) {
          const int ffnChannels = std::max(channels, context.modelInfo.transformerFFNChannels);
          vk_shader::push::TransformerSwiGLUPushParams params = {batchSize * ffnChannels * xySize};
          push(params);
          dispatch((params.size + config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX - 1) /
                   (config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX));
        }
        else if(pipeline->name.find("transformer_spatial_rms_norm_sum_sq") == 0) {
          const vkcompute::SpatialRMSNormSizing sizing = vkcompute::computeSpatialRMSNormSizing(config.spatialRMSNorm.TILE_SIZE, channels * xySize);
          vk_shader::push::TransformerSpatialRMSNormSumSqPushParams params = {batchSize,channels,xySize,sizing.tilesPerGroupPass1};
          push(params);
          dispatch(static_cast<uint32_t>(sizing.numCHWWorkgroups), static_cast<uint32_t>(batchSize));
        }
        else if(pipeline->name.find("transformer_spatial_rms_norm_reduce") == 0) {
          const vkcompute::SpatialRMSNormSizing sizing = vkcompute::computeSpatialRMSNormSizing(config.spatialRMSNorm.TILE_SIZE, channels * xySize);
          vk_shader::push::TransformerSpatialRMSNormReducePushParams params = {batchSize,sizing.numCHWWorkgroups,sizing.tilesPerGroupPass2};
          push(params);
          dispatch(1, static_cast<uint32_t>(batchSize));
        }
        else if(pipeline->name.find("transformer_spatial_rms_norm_apply") == 0) {
          vk_shader::push::TransformerSpatialRMSNormApplyPushParams params = {batchSize,channels,xySize,1e-6f};
          push(params);
          dispatch(
            static_cast<uint32_t>((channels * xySize + config.spatialRMSNorm.APPLY_ELTS_PER_THREAD * pipeline->localSizeX - 1) /
                                  (config.spatialRMSNorm.APPLY_ELTS_PER_THREAD * pipeline->localSizeX)),
            static_cast<uint32_t>(batchSize)
          );
        }
        else {
          vector<uint32_t> params((pipeline->pushConstantSize + 3) / 4, 1);
          if(!params.empty())
            vkCmdPushConstants(commandBuffer, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pipeline->pushConstantSize, params.data());
          dispatch(1, 1, 1);
        }
      };

      const auto recordDispatches = [&](size_t repeat) {
        const int runBatchSize = plan.batchSizes.empty() ? std::max(1, context.batchSize) :
          plan.batchSizes[repeat % plan.batchSizes.size()];
        for(size_t i = 0; i < pipelines.size(); i++) {
          const Pipeline* pipeline = pipelines[i];
          recordPipeline(pipeline, descriptorSets[i], runBatchSize);
          if(i + 1 < pipelines.size())
            vk_helper::barrierCommandBuffer(commandBuffer);
        }
      };

      // The warm-up dispatch is deliberately outside the timestamp interval.
      for(size_t repeat = 0; repeat < plan.warmupRuns; repeat++)
        recordDispatches(repeat);
      const size_t timedRuns = plan.timedRuns();
      vkCmdResetQueryPool(commandBuffer, queryPool, 0, static_cast<uint32_t>(2 * timedRuns));
      for(size_t timedRepeat = 0; timedRepeat < timedRuns; timedRepeat++) {
        const uint32_t queryStart = static_cast<uint32_t>(2 * timedRepeat);
        vk_helper::barrierCommandBuffer(commandBuffer);
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, queryStart);
        recordDispatches(plan.warmupRuns + timedRepeat);
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, queryStart + 1);
      }
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
        cleanup();
        return false;
      }
      result = vk_helper::submitCommandBuffers(device, {commandBuffer}, fence);
      if(result != VK_SUCCESS) {
        error = "could not submit tuning command buffer: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }
      result = vkWaitForFences(device->device, 1, &fence, VK_TRUE, UINT64_MAX);
      if(result != VK_SUCCESS) {
        error = "could not wait for tuning fence: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }
      if(timedRuns == 0) {
        error = "tuning measurement plan has no timed runs";
        cleanup();
        return false;
      }
      vector<uint64_t> timestamps(2 * timedRuns, 0);
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
        const uint64_t start = timestamps[2 * timedRepeat];
        const uint64_t end = timestamps[2 * timedRepeat + 1];
        if(end <= start) {
          error = "could not read a valid tuning timestamp";
          cleanup();
          return false;
        }
        const double elapsedSeconds = (end - start) * timestampPeriod * 1e-9;
        const double weight = plan.weightForRun(plan.warmupRuns + timedRepeat);
        weightCounted += weight * pipelines.size();
        weightedTimeTaken += elapsedSeconds * weight;
      }
      if(weightCounted <= 0.0 || weightedTimeTaken <= 0.0) {
        error = "tuning measurement has no weighted runs";
        cleanup();
        return false;
      }
      callsPerSecond = weightCounted / weightedTimeTaken;

      // Compare only produced values. Input buffers and unused allocation
      // tails must not dilute the relative error; each output has its own type.
      readback.clear();
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
          count = vk_helper::roundUpToMultiple(tiles, static_cast<size_t>(xgemmParams.MWG)) *
            vk_helper::roundUpToMultiple(static_cast<size_t>(std::max(1, context.modelInfo.trunkNumChannels)),
                                         static_cast<size_t>(xgemmParams.KWG)) * conv.inTileXSize * conv.inTileYSize;
        }
        else if(name.find("global_pooling_channels") == 0 || name.find("value_head_pool_channels") == 0)
          count = batchSize * std::max(1, context.modelInfo.gpoolNumChannels) * 3;
        else if(name.find("sum_channels") == 0)
          count = batchSize;
        else if(name.find("transformer_scale_dot_product") == 0)
          count = batchSize * std::max(1, context.modelInfo.transformerNumHeads) *
            std::max(1, context.modelInfo.transformerVHeadDim) * xySize;
        else if(name.find("transformer_swiglu") == 0)
          count = batchSize * std::max(context.modelInfo.trunkNumChannels, context.modelInfo.transformerFFNChannels) * xySize;
        else if(name.find("transformer_spatial_rms_norm_sum_sq") == 0 ||
                name.find("transformer_spatial_rms_norm_reduce") == 0) {
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
            device, tuningBuffers[firstBuffer + binding], count * sizeof(half_t), halves.data(), true, &result
          );
          for(size_t j = 0; j < count; j++)
            output[j] = half_float::half_cast<float>(halves[j]);
        }
        else {
          vk_helper::copyDeviceBufferToHost(
            device, tuningBuffers[firstBuffer + binding], count * sizeof(float), output.data(), true, &result
          );
        }
        if(result != VK_SUCCESS) {
          error = "could not read tuning output: " + vk_helper::vkErrorToString(result);
          return false;
        }
        if(isGemm) {
          for(int n = 0; n < gemmBatch; n++)
            for(int y = 0; y < logicalN; y++)
              for(int x = 0; x < logicalM; x++)
                readback.push_back(output[(static_cast<size_t>(n) * gemmN + y) * gemmM + x]);
        }
        else
          readback.insert(readback.end(), output.begin(), output.end());
        firstBuffer += pipeline->bindingCount;
      }
      cleanup();
      return true;
    }


   private:
    const VulkanDevice* device;
    float timestampPeriod;
  };

  template<typename Tuner>
  double testAllConfigs(const TuningContext& context, VulkanTuneParams& currentConfig) {
    vector<VulkanTuneParams> configs = Tuner::candidates(currentConfig, context.full, context);
    VulkanTuneParams defaults;
    configs.insert(configs.begin(), Tuner::reference(currentConfig, defaults));
    dedupCandidates(configs);
    if(configs.size() > 2) {
      Rand rand("VulkanTuner:" + Tuner::name());
      for(size_t i = configs.size() - 1; i > 1; i--) {
        const size_t j = 1 + static_cast<size_t>(rand.nextUInt64(i));
        swap(configs[i], configs[j]);
      }
    }
    const size_t validCandidateCount = count_if(configs.begin(), configs.end(), Tuner::isValid);
    const TuningMeasurementPlan plan = makeMeasurementPlan(Tuner::name(), context);

    VulkanTimestampTimer timer(context.device);
    if(!timer.isUsable()) {
      if(context.logger != nullptr)
        context.logger->write("Skipping Vulkan tuner " + Tuner::name() + ": compute timestamps are unavailable");
      return 0.0;
    }

    bool found = false;
    double bestScore = 0.0;
    double bestCallsPerSecond = 0.0;
    vector<float> referenceReadback;
    size_t candidateIndex = 0;
    for(const VulkanTuneParams& candidate: configs) {
      if(!Tuner::isValid(candidate))
        continue;
      const size_t currentCandidateIndex = candidateIndex++;
      try {
        vk_shader::ComputePipelines pipelines(context.device->device, nullptr);
        vector<const Pipeline*> targets;
        VkResult result = Tuner::create(context, candidate, pipelines, targets);
        if(result != VK_SUCCESS) {
          logTuningFailure(
            context, currentCandidateIndex, validCandidateCount,
            targets, Tuner::name(),
            "pipeline creation failed: " + vk_helper::vkErrorToString(result)
          );
          continue;
        }
        double callsPerSecond = 0.0;
        vector<float> readback;
        vector<float> cpuReference;
        double errorProp = numeric_limits<double>::quiet_NaN();
        string error;
        const bool measured = timer.measure(
          targets, candidate, context, plan, callsPerSecond, readback, errorProp, error,
          referenceReadback.empty() && usesCpuReference(Tuner::name()) ? &cpuReference : nullptr
        );
        if(!measured) {
          logTuningFailure(
            context, currentCandidateIndex, validCandidateCount,
            targets, Tuner::name(),
            error.empty() ? "measurement failed" : error
          );
          continue;
        }
        if(!isfinite(callsPerSecond) || callsPerSecond <= 0.0) {
          logTuningFailure(
            context, currentCandidateIndex, validCandidateCount,
            targets, Tuner::name(),
            "measurement returned invalid calls/sec"
          );
          continue;
        }
        if(referenceReadback.empty() && !cpuReference.empty())
          referenceReadback = std::move(cpuReference);
        if(referenceReadback.empty()) {
          double referenceErrorProp = numeric_limits<double>::quiet_NaN();
          string referenceError;
          if(!validateReadback(
               readback, readback, plan, referenceErrorProp, referenceError
             )) {
            if(isfinite(referenceErrorProp))
              logTuningResult(context, currentCandidateIndex, validCandidateCount, targets, candidate, Tuner::name(), callsPerSecond, referenceErrorProp, false);
            else
              logTuningFailure(context, currentCandidateIndex, validCandidateCount, targets, Tuner::name(), referenceError);
            continue;
          }
          referenceReadback = readback;
          errorProp = 0.0;
        }
        else if(!validateReadback(
                  referenceReadback, readback, plan, errorProp, error
                )) {
          if(isfinite(errorProp))
            logTuningResult(context, currentCandidateIndex, validCandidateCount, targets, candidate, Tuner::name(), callsPerSecond, errorProp, false);
          else
            logTuningFailure(
              context, currentCandidateIndex, validCandidateCount,
              targets, Tuner::name(),
              error.empty() ? "output validation failed" : error
            );
          continue;
        }
        const double score = VulkanTuner::computeTuningScore(callsPerSecond, errorProp, plan.errorTolerance);
        logTuningResult(
          context, currentCandidateIndex, validCandidateCount, targets, candidate, Tuner::name(), callsPerSecond, errorProp,
          score > bestScore
        );
        if(score > bestScore) {
          bestScore = score;
          bestCallsPerSecond = callsPerSecond;
          currentConfig = candidate;
          found = true;
        }
      }
      catch(const StringError& e) {
        // A failed pipeline specialization is an invalid candidate, not a fatal tuning failure.
        logTuningFailure(context, currentCandidateIndex, validCandidateCount, vector<const Pipeline*>(), Tuner::name(), e.what());
      }
    }
    if(context.logger != nullptr) {
      context.logger->write(
        "Vulkan tuner " + Tuner::name() + (found ? " selected a measured candidate" : " retained the previous candidate")
      );
    }
    return found ? bestCallsPerSecond : 0.0;
  }

  template<typename Tuner>
  double runTuner(const TuningContext& context, VulkanTuneParams& currentConfig) {
    return testAllConfigs<Tuner>(context, currentConfig);
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
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.xgemmDirect = defaults.xgemmDirect;
      return result;
    }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{8,16,32,64} : vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.WGD = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.MDIMCD = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.NDIMCD = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.MDIMAD = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.NDIMBD = v; });
      addCandidates(configs, full ? vector<int>{2,8,16} : vector<int>{2,8}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.KWID = v; });
      addCandidates(configs, vector<int>{1}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.PADA = v; });
      addCandidates(configs, vector<int>{1}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.PADB = v; });
      VulkanTuneParams slightlyTunedConfig = current;
      slightlyTunedConfig.xgemmDirect = VulkanTuneParams().xgemmDirect;
      slightlyTunedConfig.xgemmDirect.MDIMCD = 8;
      slightlyTunedConfig.xgemmDirect.NDIMCD = 8;
      slightlyTunedConfig.xgemmDirect.MDIMAD = 8;
      slightlyTunedConfig.xgemmDirect.NDIMBD = 8;
      VulkanTuneParams slightlyTunedConfig2 = slightlyTunedConfig;
      slightlyTunedConfig2.xgemmDirect.WGD = 16;
      configs.insert(configs.begin(), slightlyTunedConfig2);
      configs.insert(configs.begin(), slightlyTunedConfig);
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createXgemmStridedBatched(pipelines.xgemmStridedBatchedFp32, config.xgemmDirect, config.vulkan);
      if(result == VK_SUCCESS) {
        targets.push_back(&pipelines.xgemmStridedBatchedFp32);
      }
      return result;
    }
  };

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
      result.hgemmCooperativeMatrix.MWG = defaults.hgemmCooperativeMatrix.MWG;
      result.hgemmCooperativeMatrix.NWG = defaults.hgemmCooperativeMatrix.NWG;
      result.hgemmCooperativeMatrix.KWG = defaults.hgemmCooperativeMatrix.KWG;
      result.hgemmCooperativeMatrix.MWAVE = defaults.hgemmCooperativeMatrix.MWAVE;
      result.hgemmCooperativeMatrix.NWAVE = defaults.hgemmCooperativeMatrix.NWAVE;
      result.hgemmCooperativeMatrix.SA = defaults.hgemmCooperativeMatrix.SA;
      result.hgemmCooperativeMatrix.SB = defaults.hgemmCooperativeMatrix.SB;
      return result;
    }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.MWG = v; });
      addCandidates(configs, full ? vector<int>{16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.NWG = v; });
      addCandidates(configs, vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.KWG = v; });
      addCandidates(configs, vector<int>{8,16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.MWAVE = v; });
      addCandidates(configs, vector<int>{8,16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.NWAVE = v; });
      addCandidates(configs, vector<int>{0,1}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.SA = v; });
      addCandidates(configs, vector<int>{0,1}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.SB = v; });
      if(!full) {
        configs.erase(
          remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) { return !p.hgemmCooperativeMatrix.isSimple(); }),
          configs.end()
        );
      }
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createHgemmCooperativeMatrix(pipelines.hgemmCooperativeMatrix, config.hgemmCooperativeMatrix);
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.hgemmCooperativeMatrix);
      return result;
    }
  };

  struct HgemmCooperativeMatrixNCHWTunerImpl {
    static string name() { return "hgemmCooperativeMatrixNCHW"; }
    static bool isValid(const VulkanTuneParams& config) {
      return config.vulkan.canUseCooperativeMatrix && config.hgemmCooperativeMatrixNCHW.isValid();
    }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.hgemmCooperativeMatrixNCHW.MWG = defaults.hgemmCooperativeMatrixNCHW.MWG;
      result.hgemmCooperativeMatrixNCHW.NWG = defaults.hgemmCooperativeMatrixNCHW.NWG;
      result.hgemmCooperativeMatrixNCHW.KWG = defaults.hgemmCooperativeMatrixNCHW.KWG;
      result.hgemmCooperativeMatrixNCHW.MWAVE = defaults.hgemmCooperativeMatrixNCHW.MWAVE;
      result.hgemmCooperativeMatrixNCHW.NWAVE = defaults.hgemmCooperativeMatrixNCHW.NWAVE;
      result.hgemmCooperativeMatrixNCHW.SB = defaults.hgemmCooperativeMatrixNCHW.SB;
      result.hgemmCooperativeMatrixNCHW.VWM = defaults.hgemmCooperativeMatrixNCHW.VWM;
      result.hgemmCooperativeMatrixNCHW.VWN = defaults.hgemmCooperativeMatrixNCHW.VWN;
      return result;
    }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.MWG = v; });
      addCandidates(configs, full ? vector<int>{16,32} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.NWG = v; });
      addCandidates(configs, vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.KWG = v; });
      addCandidates(configs, vector<int>{8,16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.MWAVE = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.NWAVE = v; });
      addCandidates(configs, vector<int>{0,1}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.SB = v; });
      if(!full) {
        configs.erase(
          remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) { return !p.hgemmCooperativeMatrixNCHW.isSimple(); }),
          configs.end()
        );
      }
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createHgemmCooperativeMatrixNCHW(pipelines.hgemmCooperativeMatrixNCHW, config.hgemmCooperativeMatrixNCHW);
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.hgemmCooperativeMatrixNCHW);
      return result;
    }
  };

  struct XgemmTuner {
    static string name() { return "xgemm"; }
    static bool isValid(const VulkanTuneParams& config) { return config.xgemm.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.xgemm = defaults.xgemm;
      return result;
    }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.xgemm.MWG = v; });
      addCandidates(configs, full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.xgemm.NWG = v; });
      addCandidates(configs, full ? vector<int>{8,16,32} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.KWG = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.MDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.NDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.MDIMA = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.NDIMB = v; });
      VulkanTuneParams slightlyTunedConfig = current;
      slightlyTunedConfig.xgemm = VulkanTuneParams().xgemm;
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
      if(!full) {
        configs.erase(
          remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) { return !p.xgemm.isSimple(); }),
          configs.end()
        );
      }
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createXgemmBatched(pipelines.xgemmBatchedFp32, config.xgemm, config.xgemm16, config.vulkan);
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.xgemmBatchedFp32);
      return result;
    }
  };

  struct Xgemm16Tuner {
    static string name() { return "xgemm16"; }
    static bool isValid(const VulkanTuneParams& config) { return config.xgemm16.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.xgemm16 = defaults.xgemm16;
      return result;
    }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.xgemm16.MWG = v; });
      addCandidates(configs, full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.xgemm16.NWG = v; });
      addCandidates(configs, full ? vector<int>{8,16,32} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.KWG = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.MDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.NDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.MDIMA = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.NDIMB = v; });

      VulkanTuneParams slightlyTunedConfig = current;
      slightlyTunedConfig.xgemm16 = VulkanTuneParams().xgemm16;
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
      if(!full) {
        configs.erase(
          remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) { return !p.xgemm16.isSimple(); }),
          configs.end()
        );
      }
      configs.insert(configs.begin(), current);
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createXgemmBatched(pipelines.xgemmBatchedFp32, config.xgemm, config.xgemm16, config.vulkan);
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.xgemmBatchedFp32);
      return result;
    }
  };

  template<int ConvSize, uint32_t OutTileSize, bool InputTransform>
  struct ConvTuner {
    static string name() {
      return string(ConvSize == 3 ? "conv3x3" : "conv5x5") +
        (InputTransform ? "InputTransform" : "OutputTransform");
    }
    static ConvTuneParams& params(VulkanTuneParams& config) { return ConvSize == 3 ? config.conv3x3 : config.conv5x5; }
    static const ConvTuneParams& params(const VulkanTuneParams& config) { return ConvSize == 3 ? config.conv3x3 : config.conv5x5; }
    static bool isValid(const VulkanTuneParams& config) { return params(config).isValid(OutTileSize); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      if(InputTransform) {
        params(result).inputTransformLocalXSize = params(defaults).inputTransformLocalXSize;
        params(result).inputTransformLocalYSize = params(defaults).inputTransformLocalYSize;
      }
      else {
        params(result).outputTransformLocalXSize = params(defaults).outputTransformLocalXSize;
        params(result).outputTransformLocalYSize = params(defaults).outputTransformLocalYSize;
        params(result).outputTransformLocalZSize = params(defaults).outputTransformLocalZSize;
      }
      return result;
    }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      if(InputTransform) {
        addCandidates(configs, vector<int>{1,2,4,8,16,32,64,128}, [](VulkanTuneParams& p, int v) { params(p).inputTransformLocalXSize = v; });
        addCandidates(configs, full ? vector<int>{1,2,4,8,16,32,64} : vector<int>{1,2,4,8,16,32}, [](VulkanTuneParams& p, int v) { params(p).inputTransformLocalYSize = v; });
      }
      else {
        addCandidates(configs, full ? vector<int>{1,2,4,8,16,32,64} : vector<int>{1,2,8,16,32}, [](VulkanTuneParams& p, int v) { params(p).outputTransformLocalXSize = v; });
        addCandidates(configs, full ? vector<int>{1,2,4,8,16,32,64} : vector<int>{1,2,4,16,32}, [](VulkanTuneParams& p, int v) { params(p).outputTransformLocalYSize = v; });
        addCandidates(configs, full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8,16}, [](VulkanTuneParams& p, int v) { params(p).outputTransformLocalZSize = v; });
      }
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      Pipeline& pipeline = InputTransform
        ? (ConvSize == 3 ? pipelines.winogradInputTransform3x3 : pipelines.winogradInputTransform5x5)
        : (ConvSize == 3 ? pipelines.winogradOutputTransform3x3 : pipelines.winogradOutputTransform5x5);
      VkResult result = InputTransform
        ? pipelines.createWinogradInputTransform(pipeline, params(config), ConvSize, config.vulkan)
        : pipelines.createWinogradOutputTransform(pipeline, params(config), ConvSize, config.vulkan);
      if(result == VK_SUCCESS) {
        pipeline.name += ConvSize == 3 ? "_3x3" : "_5x5";
        targets.push_back(&pipeline);
      }
      return result;
    }
  };

  struct Conv3x3InputTuner : ConvTuner<3,4,true> {};
  struct Conv3x3OutputTuner : ConvTuner<3,4,false> {};
  struct Conv5x5InputTuner : ConvTuner<5,2,true> {};
  struct Conv5x5OutputTuner : ConvTuner<5,2,false> {};

  struct GPoolTuner {
    static string name() { return "gPool"; }
    static bool isValid(const VulkanTuneParams& config) { return config.gPool.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.gPool = defaults.gPool; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext& context) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32,64} : vector<int>{1,2,4,8,16,32}, [](VulkanTuneParams& p, int v) { p.gPool.XYSTRIDE = v; });
      addCandidates(configs, powersOfTwoUpTo(std::min(full ? 64 : 32, std::max(1, context.modelInfo.gpoolNumChannels))), [](VulkanTuneParams& p, int v) { p.gPool.CHANNELSTRIDE = v; });
      addCandidates(configs, powersOfTwoUpTo(std::min(4, std::max(1, context.batchSize))), [](VulkanTuneParams& p, int v) { p.gPool.BATCHSTRIDE = v; });
      return configs;
    }
    static VkResult create(const TuningContext& context, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createGlobalPoolingChannelsFp32(pipelines.globalPoolingChannelsFp32, config.gPool, config.vulkan);
      if(result != VK_SUCCESS)
        return result;
      targets.push_back(&pipelines.globalPoolingChannelsFp32);

      const uint32_t localSizeY = static_cast<uint32_t>(std::min(
        config.gPool.CHANNELSTRIDE,
        static_cast<int>(vk_helper::powerOf2ify(std::max(1, context.modelInfo.gpoolNumChannels)))
      ));
      const uint32_t localSizeZ = static_cast<uint32_t>(std::min(
        config.gPool.BATCHSTRIDE,
        static_cast<int>(vk_helper::powerOf2ify(std::max(1, context.batchSize)))
      ));
      LocalDim valueHeadDim = {config.gPool.XYSTRIDE, static_cast<int>(localSizeY), static_cast<int>(localSizeZ)};
      Pipeline valueHeadPipeline;
      result = pipelines.createValueHeadPoolingChannels(valueHeadPipeline, config.gPool, localSizeY, localSizeZ, config.vulkan);
      if(result != VK_SUCCESS)
        return result;
      decltype(pipelines.valueHeadPoolingChannels)::iterator valueHeadIterator;
      bool valueHeadInserted = false;
      try {
        auto insertResult = pipelines.valueHeadPoolingChannels.emplace(valueHeadDim, std::move(valueHeadPipeline));
        valueHeadIterator = insertResult.first;
        valueHeadInserted = insertResult.second;
      }
      catch(...) {
        pipelines.destroyPipeline(valueHeadPipeline);
        throw;
      }
      if(!valueHeadInserted) {
        pipelines.destroyPipeline(valueHeadPipeline);
        return VK_ERROR_INITIALIZATION_FAILED;
      }
      valueHeadPipeline.pipeline = VK_NULL_HANDLE;
      valueHeadPipeline.layout = VK_NULL_HANDLE;
      valueHeadPipeline.descriptorSetLayout = VK_NULL_HANDLE;
      targets.push_back(&valueHeadIterator->second);

      LocalDim sumChannelsDim = {config.gPool.XYSTRIDE, 1, static_cast<int>(localSizeZ)};
      Pipeline sumChannelsPipeline;
      result = pipelines.createSumChannels(sumChannelsPipeline, config.gPool, localSizeZ, config.vulkan);
      if(result != VK_SUCCESS)
        return result;
      decltype(pipelines.sumChannels)::iterator sumChannelsIterator;
      bool sumChannelsInserted = false;
      try {
        auto insertResult = pipelines.sumChannels.emplace(sumChannelsDim, std::move(sumChannelsPipeline));
        sumChannelsIterator = insertResult.first;
        sumChannelsInserted = insertResult.second;
      }
      catch(...) {
        pipelines.destroyPipeline(sumChannelsPipeline);
        throw;
      }
      if(!sumChannelsInserted) {
        pipelines.destroyPipeline(sumChannelsPipeline);
        return VK_ERROR_INITIALIZATION_FAILED;
      }
      sumChannelsPipeline.pipeline = VK_NULL_HANDLE;
      sumChannelsPipeline.layout = VK_NULL_HANDLE;
      sumChannelsPipeline.descriptorSetLayout = VK_NULL_HANDLE;
      targets.push_back(&sumChannelsIterator->second);
      return result;
    }
  };

  struct PointwiseTuner {
    static string name() { return "pointwise"; }
    static bool isValid(const VulkanTuneParams& config) { return config.pointwise.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.pointwise = defaults.pointwise; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8,16}, [](VulkanTuneParams& p, int v) { p.pointwise.ELTS_PER_THREAD = v; });
      addCandidates(configs, full ? vector<int>{32,64,128,256,512} : vector<int>{32,64,128,256}, [](VulkanTuneParams& p, int v) { p.pointwise.LOCAL_SIZE = v; });
      return configs;
    }
    static VkResult create(const TuningContext& context, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createAddPointWise(pipelines.addPointWise, config.pointwise, config.vulkan);
      if(result != VK_SUCCESS) return result;
      targets.push_back(&pipelines.addPointWise);
      if(context.modelInfo.transformerFFNChannels > 0 &&
         context.modelInfo.transformerHeadDim > 0 && context.modelInfo.transformerVHeadDim > 0) {
        result = pipelines.createTransformerSwiGLU(pipelines.transformerSwiGLU, config.pointwise, config.vulkan);
        if(result == VK_SUCCESS) targets.push_back(&pipelines.transformerSwiGLU);
      }
      return result;
    }
  };

  struct AddChannelBiasesTuner {
    static string name() { return "addChannelBiases"; }
    static bool isValid(const VulkanTuneParams& config) { return config.addChannelBiases.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.addChannelBiases = defaults.addChannelBiases; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, vector<int>{1,2,4}, [](VulkanTuneParams& p, int v) { p.addChannelBiases.XY_ELTS_PER_THREAD = v; });
      addCandidates(configs, vector<int>{1,2,4,8}, [](VulkanTuneParams& p, int v) { p.addChannelBiases.NC_ELTS_PER_THREAD = v; });
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createAddChannelBiasNCHW(pipelines.addChannelBiasNCHW, config.addChannelBiases, config.vulkan);
      if(result == VK_SUCCESS) targets.push_back(&pipelines.addChannelBiasNCHW);
      return result;
    }
  };

  struct TransformerTuner {
    static string name() { return "transformerAttention"; }
    static bool isValid(const VulkanTuneParams& config) { return config.transformer.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.transformer = defaults.transformer; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, vector<int>{0,1}, [](VulkanTuneParams& p, int v) { p.transformer.USE_TILED_ATTN = v; });
      addCandidates(configs, full ? vector<int>{8,16,32,64,128,256} : vector<int>{16,32,64,128,256}, [](VulkanTuneParams& p, int v) { p.transformer.ATTN_BLOCK_Q = v; });
      addCandidates(configs, full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64,128}, [](VulkanTuneParams& p, int v) { p.transformer.ATTN_BLOCK_KV = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8} : vector<int>{1,2,4}, [](VulkanTuneParams& p, int v) { p.transformer.Q_PER_THREAD = v; });
      return configs;
    }
    static VkResult create(const TuningContext& context, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      if(context.modelInfo.transformerHeadDim <= 0 || context.modelInfo.transformerVHeadDim <= 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
      VkResult result;
      if(config.transformer.USE_TILED_ATTN) {
        result = pipelines.createTransformerScaleDotProduct(pipelines.transformerScaleDotProduct, config.transformer, context.modelInfo.transformerHeadDim, context.modelInfo.transformerVHeadDim, config.vulkan);
        if(result == VK_SUCCESS) targets.push_back(&pipelines.transformerScaleDotProduct);
      }
      else {
        result = pipelines.createTransformerScaleDotProductNaive(pipelines.transformerScaleDotProductNaive, context.modelInfo.transformerHeadDim, context.modelInfo.transformerVHeadDim, config.vulkan);
        if(result == VK_SUCCESS) targets.push_back(&pipelines.transformerScaleDotProductNaive);
      }
      return result;
    }
  };

  struct TransformerRMSNormTuner {
    static string name() { return "transformerRMSNorm"; }
    static bool isValid(const VulkanTuneParams& config) { return config.rmsNorm.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.rmsNorm = defaults.rmsNorm; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{32,64,128,256,512} : vector<int>{32,64,128,256}, [](VulkanTuneParams& p, int v) { p.rmsNorm.WG_C_SIZE = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8,16}, [](VulkanTuneParams& p, int v) { p.rmsNorm.WG_XY_SIZE = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8,16} : vector<int>{1,2,4,8}, [](VulkanTuneParams& p, int v) { p.rmsNorm.C_PER_THREAD = v; });
      return configs;
    }
    static VkResult create(const TuningContext& context, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      if(context.modelInfo.transformerHeadDim <= 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
      VkResult result = pipelines.createTransformerRMSNorm(pipelines.transformerRmsNorm, config.rmsNorm, config.vulkan);
      if(result == VK_SUCCESS) targets.push_back(&pipelines.transformerRmsNorm);
      return result;
    }
  };

  struct SpatialRMSNormTuner {
    static string name() { return "spatialRMSNorm"; }
    static bool isValid(const VulkanTuneParams& config) { return config.spatialRMSNorm.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.spatialRMSNorm = defaults.spatialRMSNorm; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{32,64,128,256,512,1024} : vector<int>{32,64,128,256,512}, [](VulkanTuneParams& p, int v) { p.spatialRMSNorm.TILE_SIZE = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8,16}, [](VulkanTuneParams& p, int v) { p.spatialRMSNorm.APPLY_ELTS_PER_THREAD = v; });
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createTransformerSpatialRMSNormSumSq(pipelines.transformerSpatialRMSNormSumSq, config.spatialRMSNorm, config.vulkan);
      if(result != VK_SUCCESS) return result;
      result = pipelines.createTransformerSpatialRMSNormReduce(pipelines.transformerSpatialRMSNormReduce, config.spatialRMSNorm, config.vulkan);
      if(result != VK_SUCCESS) return result;
      result = pipelines.createTransformerSpatialRMSNormApply(pipelines.transformerSpatialRMSNormApply, config.spatialRMSNorm, config.vulkan);
      if(result == VK_SUCCESS) {
        targets.push_back(&pipelines.transformerSpatialRMSNormSumSq);
        targets.push_back(&pipelines.transformerSpatialRMSNormReduce);
        targets.push_back(&pipelines.transformerSpatialRMSNormApply);
      }
      return result;
    }
  };

  void runNonGemmTuners(const TuningContext& context, VulkanTuneParams& config) {
    runTuner<Conv3x3InputTuner>(context, config);
    runTuner<Conv3x3OutputTuner>(context, config);
    runTuner<Conv5x5InputTuner>(context, config);
    runTuner<Conv5x5OutputTuner>(context, config);
    runTuner<GPoolTuner>(context, config);
    runTuner<PointwiseTuner>(context, config);
    runTuner<AddChannelBiasesTuner>(context, config);
    if(context.modelInfo.transformerHeadDim > 0 && context.modelInfo.transformerVHeadDim > 0) {
      runTuner<TransformerTuner>(context, config);
      runTuner<TransformerRMSNormTuner>(context, config);
      runTuner<SpatialRMSNormTuner>(context, config);
    }
  }

  bool tuneXgemm16(
    const TuningContext& context,
    VulkanTuneParams& config,
    double fp32CallsPerSecond
  ) {
    if(!config.vulkan.canUseFP16Storage || !config.vulkan.canUseFP16Compute) {
      if(context.logger != nullptr)
        context.logger->write("Skipping Vulkan xgemm16 tuning: FP16 storage or compute is unavailable");
      return false;
    }
    if(!isfinite(fp32CallsPerSecond) || fp32CallsPerSecond <= 0.0) {
      if(context.logger != nullptr)
        context.logger->write("Skipping Vulkan xgemm16 tuning: FP32 profile failed");
      return false;
    }

    VulkanTuneParams tunedConfig = config;
    tunedConfig.vulkan.shouldUseFP16Storage = true;
    tunedConfig.vulkan.shouldUseFP16Compute = true;
    const double fp16CallsPerSecond = runTuner<Xgemm16Tuner>(context, tunedConfig);
    if(!isfinite(fp16CallsPerSecond) || fp16CallsPerSecond <= 0.0) {
      config.xgemm16 = config.xgemm;
      if(context.logger != nullptr)
        context.logger->write("Vulkan xgemm16 tuning failed, retaining xgemm parameters");
      return false;
    }

    const bool computeIsFastEnough = VulkanTuner::isFastEnough(
      fp16CallsPerSecond, fp32CallsPerSecond, VulkanTuner::FP16_COMPUTE_MIN_THROUGHPUT_RATIO
    );
    if(context.logger != nullptr) {
      context.logger->write(
        "Vulkan xgemm16 comparison: fp32=" + Global::strprintf("%.6g", fp32CallsPerSecond) +
        " calls/s, p16s16=" + Global::strprintf("%.6g", fp16CallsPerSecond) +
        " calls/s, required_ratio=" + Global::strprintf("%.2f", VulkanTuner::FP16_COMPUTE_MIN_THROUGHPUT_RATIO)
      );
    }
    if(!computeIsFastEnough) {
      if(context.logger != nullptr)
        context.logger->write("Vulkan xgemm16 did not reach the FP32 baseline threshold, not enabling FP16 compute");
      return false;
    }
    config.xgemm16 = tunedConfig.xgemm16;
    config.vulkan.shouldUseFP16Storage = true;
    config.vulkan.shouldUseFP16Compute = true;
    if(context.logger != nullptr)
      context.logger->write("Enabling Vulkan FP16 compute due to better xgemm16 performance");
    return true;
  }

  bool tuneXgemmStorage(
    const TuningContext& context,
    VulkanTuneParams& config,
    double fp32CallsPerSecond
  ) {
    if(!config.vulkan.canUseFP16Storage || !config.vulkan.canUseFP16Compute ||
       !isfinite(fp32CallsPerSecond) || fp32CallsPerSecond <= 0.0)
      return false;

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
        " calls/s, p16s32=" + Global::strprintf("%.6g", fp16StorageCallsPerSecond) +
        " calls/s, required_ratio=" + Global::strprintf("%.2f", VulkanTuner::FP16_STORAGE_MIN_THROUGHPUT_RATIO) +
        ", selected=" + (storageIsFastEnough ? "true" : "false")
      );
    }
    if(!storageIsFastEnough)
      return false;
    config.xgemm = tunedConfig.xgemm;
    config.vulkan.shouldUseFP16Storage = true;
    return true;
  }

  void tuneCooperativeMatrices(
    const TuningContext& context,
    VulkanTuneParams& config,
    double xgemmDirectBaselineCallsPerSecond,
    double xgemmBaselineCallsPerSecond
  ) {
    if(!config.vulkan.canUseCooperativeMatrix || !config.vulkan.canUseFP16Storage ||
       !config.vulkan.canUseFP16Compute)
      return;

    VulkanTuneParams cooperativeConfig = config;
    cooperativeConfig.vulkan.shouldUseFP16Storage = true;
    cooperativeConfig.vulkan.shouldUseFP16Compute = false;
    const double hgemmCallsPerSecond = runTuner<HgemmCooperativeMatrixTunerImpl>(context, cooperativeConfig);
    const bool useHgemm = VulkanTuner::isFastEnough(
      hgemmCallsPerSecond, xgemmBaselineCallsPerSecond, VulkanTuner::COOPERATIVE_MATRIX_MIN_THROUGHPUT_RATIO
    );
    if(useHgemm) {
      config.hgemmCooperativeMatrix = cooperativeConfig.hgemmCooperativeMatrix;
      config.vulkan.shouldUseCooperativeMatrix = true;
    }
    if(context.logger != nullptr) {
      context.logger->write(
        "Vulkan hgemmCooperativeMatrix baseline comparison: xgemm=" +
        Global::strprintf("%.6g", xgemmBaselineCallsPerSecond) +
        " calls/s, hgemmCooperativeMatrix=" + Global::strprintf("%.6g", hgemmCallsPerSecond) +
        " calls/s, required_ratio=" + Global::strprintf("%.2f", VulkanTuner::COOPERATIVE_MATRIX_MIN_THROUGHPUT_RATIO) +
        ", selected=" + (useHgemm ? "true" : "false")
      );
    }

    cooperativeConfig = config;
    cooperativeConfig.vulkan.shouldUseFP16Storage = true;
    cooperativeConfig.vulkan.shouldUseFP16Compute = false;
    const double hgemmNCHWCallsPerSecond = runTuner<HgemmCooperativeMatrixNCHWTunerImpl>(context, cooperativeConfig);
    const bool useHgemmNCHW = VulkanTuner::isFastEnough(
      hgemmNCHWCallsPerSecond, xgemmDirectBaselineCallsPerSecond,
      VulkanTuner::COOPERATIVE_MATRIX_1X1_MIN_THROUGHPUT_RATIO
    );
    if(useHgemmNCHW) {
      config.hgemmCooperativeMatrixNCHW = cooperativeConfig.hgemmCooperativeMatrixNCHW;
      config.vulkan.shouldUseCooperativeMatrix = true;
      config.vulkan.shouldUseHgemmCooperativeMatrixNCHW = true;
    }
    if(config.vulkan.shouldUseCooperativeMatrix)
      config.vulkan.shouldUseFP16Storage = true;
    if(context.logger != nullptr) {
      context.logger->write(
        "Vulkan hgemmCooperativeMatrixNCHW baseline comparison: xgemmDirect=" +
        Global::strprintf("%.6g", xgemmDirectBaselineCallsPerSecond) +
        " calls/s, hgemmCooperativeMatrixNCHW=" + Global::strprintf("%.6g", hgemmNCHWCallsPerSecond) +
        " calls/s, required_ratio=" + Global::strprintf("%.2f", VulkanTuner::COOPERATIVE_MATRIX_1X1_MIN_THROUGHPUT_RATIO) +
        ", selected=" + (useHgemmNCHW ? "true" : "false")
      );
    }
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
  VulkanTuneParams& tunedConfig
) {
  if(device == nullptr)
    throw StringError("VulkanTuner::tune: device is null");
  if(!tunedConfig.isValid())
    tunedConfig = VulkanTuneParams();
  TuningContext context{device, batchSize, nnXLen, nnYLen, modelInfo, full, logger};
  if(logger != nullptr) {
    logger->write(
      "Vulkan tuning capabilities: fp16Storage=" + string(tunedConfig.vulkan.canUseFP16Storage ? "true" : "false") +
      ", fp16Compute=" + string(tunedConfig.vulkan.canUseFP16Compute ? "true" : "false") +
      ", subgroup=" + string(tunedConfig.vulkan.canUseSubgroup ? "true" : "false") +
      ", cooperativeMatrix=" + string(tunedConfig.vulkan.canUseCooperativeMatrix ? "true" : "false")
    );
  }
  if(tunedConfig.vulkan.canUseCooperativeMatrix &&
     !HgemmCooperativeMatrixTuner::selectCooperativeMatrixProperties(device, tunedConfig.hgemmCooperativeMatrix)) {
    tunedConfig.vulkan.canUseCooperativeMatrix = false;
  }
  if(tunedConfig.vulkan.canUseCooperativeMatrix) {
    if(HgemmCooperativeMatrixNCHWTuner::selectCooperativeMatrixProperties(device, tunedConfig.hgemmCooperativeMatrixNCHW)) {
      tunedConfig.hgemmCooperativeMatrix.MWARP = tunedConfig.hgemmCooperativeMatrixNCHW.MWARP;
      tunedConfig.hgemmCooperativeMatrix.NWARP = tunedConfig.hgemmCooperativeMatrixNCHW.NWARP;
      tunedConfig.hgemmCooperativeMatrix.KDIM = tunedConfig.hgemmCooperativeMatrixNCHW.KDIM;
      tunedConfig.hgemmCooperativeMatrix.subgroupSize = tunedConfig.hgemmCooperativeMatrixNCHW.subgroupSize;
    }
  }
  tunedConfig.vulkan.shouldUseFP16Storage = false;
  tunedConfig.vulkan.shouldUseFP16Compute = false;
  tunedConfig.vulkan.shouldUseCooperativeMatrix = false;
  tunedConfig.vulkan.shouldUseHgemmCooperativeMatrixNCHW = false;
  tunedConfig.vulkan.shouldUseSubgroup = false;
  double xgemmDirectBaselineCallsPerSecond = 0.0;
  double xgemmBaselineCallsPerSecond = 0.0;
  xgemmDirectBaselineCallsPerSecond = runTuner<XgemmDirectTuner>(context, tunedConfig);
  xgemmBaselineCallsPerSecond = runTuner<XgemmTuner>(context, tunedConfig);
  tunedConfig.xgemm16 = tunedConfig.xgemm;
  tuneXgemm16(context, tunedConfig, xgemmBaselineCallsPerSecond);
  if(!tunedConfig.vulkan.shouldUseFP16Compute)
    tuneXgemmStorage(context, tunedConfig, xgemmBaselineCallsPerSecond);
  tuneCooperativeMatrices(
    context, tunedConfig, xgemmDirectBaselineCallsPerSecond, xgemmBaselineCallsPerSecond
  );
  runNonGemmTuners(context, tunedConfig);
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

VulkanTuneParams VulkanTuner::loadOrAutoTune(
  const string& tunerFile,
  const string& homeDataDirOverride,
  const string& gpuName,
  int nnXLen,
  int nnYLen,
  const ModelInfoForTuning& modelInfo,
  const VulkanDevice* device,
  Logger* logger
) {
  string filename = tunerFile;
  if(filename.empty())
    filename = defaultDirectory(true, homeDataDirOverride) + "/" + defaultFileName(gpuName, nnXLen, nnYLen, modelInfo);

  try {
    VulkanTuneParams loaded = VulkanTuneParams::load(filename);
    if(device != nullptr) {
      const VulkanParams available = makeVulkanParams(device->info);
      if(loaded.vulkan.canUseFP16Storage != available.canUseFP16Storage ||
         loaded.vulkan.canUseFP16Compute != available.canUseFP16Compute ||
         (loaded.vulkan.canUseCooperativeMatrix && !available.canUseCooperativeMatrix) ||
         loaded.vulkan.canUseSubgroup != available.canUseSubgroup) {
        throw IOError("Vulkan tuning capabilities changed for " + filename);
      }
    }
    if(logger != nullptr)
      logger->write("Loaded Vulkan tuning parameters from: " + filename);
    return loaded;
  } catch(const StringError&) {
  }

  if(device == nullptr)
    throw StringError("VulkanTuner::loadOrAutoTune: device is null");
  VulkanTuneParams params;
  params.vulkan = makeVulkanParams(device->info);
  tune(device, DEFAULT_BATCH_SIZE, nnXLen, nnYLen, modelInfo, false, logger, params);
  VulkanTuneParams::save(filename, params);
  if(logger != nullptr)
    logger->write("Completed Vulkan tuning and saved results to: " + filename);
  return params;
}

#endif
