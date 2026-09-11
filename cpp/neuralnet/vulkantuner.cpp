#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"
#include "../neuralnet/vulkancompute.h"

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <map>
#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
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
  bool supportsCooperativeMatrixType(const VkCooperativeMatrixPropertiesKHR& property, int accType) {
    const VkComponentTypeKHR expectedType = accType == 16
      ? VK_COMPONENT_TYPE_FLOAT16_KHR
      : VK_COMPONENT_TYPE_FLOAT32_KHR;
    return property.scope == VK_SCOPE_SUBGROUP_KHR &&
           property.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
           property.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
           property.CType == expectedType &&
           property.ResultType == expectedType;
  }

  bool getSupportedCooperativeMatrixProperties(
    VkPhysicalDevice physicalDevice,
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getProperties,
    VkBool32 cooperativeMatrix,
    vector<VkCooperativeMatrixPropertiesKHR>& properties
  ) {
    if(cooperativeMatrix != VK_TRUE || getProperties == nullptr)
      return false;

    uint32_t propertyCount = 0;
    VkResult result = getProperties(physicalDevice, &propertyCount, nullptr);
    if(result != VK_SUCCESS || propertyCount == 0)
      return false;

    properties.resize(propertyCount);
    for(VkCooperativeMatrixPropertiesKHR& property: properties) {
      property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
      property.pNext = nullptr;
    }
    result = getProperties(physicalDevice, &propertyCount, properties.data());
    if(result != VK_SUCCESS && result != VK_INCOMPLETE)
      return false;
    properties.resize(propertyCount);

    properties.erase(
      remove_if(properties.begin(), properties.end(), [](const VkCooperativeMatrixPropertiesKHR& property) {
        return !supportsCooperativeMatrixType(property, 16) &&
               !supportsCooperativeMatrixType(property, 32);
      }),
      properties.end()
    );
    return !properties.empty();
  }

  bool supportsCooperativeMatrix(const VulkanDeviceInfo& deviceInfo) {
    vector<VkCooperativeMatrixPropertiesKHR> properties;
    return getSupportedCooperativeMatrixProperties(
      deviceInfo.physicalDevice,
      deviceInfo.cooperativeMatrixPropertiesFn,
      deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix,
      properties
    );
  }
}

VulkanParams VulkanTuner::getHardwareParams(const VulkanDeviceInfo& deviceInfo) {
  VulkanParams params;
  params.canUseFP16Storage =
    deviceInfo.storage16BitFeatures.storageBuffer16BitAccess == VK_TRUE ||
    deviceInfo.storage16BitFeatures.uniformAndStorageBuffer16BitAccess == VK_TRUE;
  params.canUseFP16Compute = deviceInfo.shaderFloat16Int8Features.shaderFloat16 == VK_TRUE;
  params.canUseCooperativeMatrix = supportsCooperativeMatrix(deviceInfo);
  params.canUseSubgroup =
    (deviceInfo.subgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
    deviceInfo.subgroupSizeControlFeatures.computeFullSubgroups == VK_TRUE;
  return params;
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

  void writeParam(ofstream& out, const string& name, uint32_t value) {
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
     outputTransformLocalXSize == 0 || outputTransformLocalYSize == 0 || outputTransformLocalZSize == 0)
    return false;
  return static_cast<uint64_t>(inputTransformLocalXSize) * inputTransformLocalYSize <= 1024 &&
         static_cast<uint64_t>(outputTransformLocalXSize) * outputTransformLocalYSize * outputTransformLocalZSize <= 1024;
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
  const uint64_t localSizeX = static_cast<uint64_t>(MWAVE / MWARP) * subgroupSize;
  const uint64_t localSizeY = static_cast<uint64_t>(NWAVE / NWARP);
  if(localSizeX == 0 || localSizeY == 0 || localSizeX * localSizeY > 1024)
    return false;
  return isMultipleOf(MWG, MWAVE) && isMultipleOf(NWG, NWAVE) &&
         isMultipleOf(KWG, KDIM) && isMultipleOf(MWAVE, MWARP) &&
         isMultipleOf(NWAVE, NWARP) && isMultipleOf(MWG, VWM) &&
         isMultipleOf(NWG, VWN) && isMultipleOf(KWG, VWM);
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
  const uint64_t localSizeX = static_cast<uint64_t>(MWAVE / MWARP) * subgroupSize;
  const uint64_t localSizeY = static_cast<uint64_t>(NWAVE / NWARP);
  if(localSizeX == 0 || localSizeY == 0 || localSizeX * localSizeY > 1024)
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

bool TransformerRMSNormTuneParms::isValid() const {
  if(WG_C_SIZE <= 0 || WG_C_SIZE > 1024) return false;
  if((WG_C_SIZE & (WG_C_SIZE - 1)) != 0) return false;
  if(WG_XY_SIZE <= 0 || WG_XY_SIZE > 32) return false;
  if((WG_XY_SIZE & (WG_XY_SIZE - 1)) != 0) return false;
  if(static_cast<uint64_t>(WG_C_SIZE) * WG_XY_SIZE > 1024) return false;
  if(C_PER_THREAD <= 0 || C_PER_THREAD > 32) return false;
  if((C_PER_THREAD & (C_PER_THREAD - 1)) != 0) return false;
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
         hgemmCooperativeMatrixNCHW.isValid() &&
         xgemm.isValid() && xgemm16.isValid() && xgemmDirect.isValid() &&
         transformer.isValid() && rmsNorm.isValid() && spatialRMSNorm.isValid();
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
         hgemmCooperativeMatrix.accType == other.hgemmCooperativeMatrix.accType &&
         hgemmCooperativeMatrix.SA == other.hgemmCooperativeMatrix.SA &&
         hgemmCooperativeMatrix.SB == other.hgemmCooperativeMatrix.SB &&
         hgemmCooperativeMatrix.VWM == other.hgemmCooperativeMatrix.VWM &&
         hgemmCooperativeMatrix.VWN == other.hgemmCooperativeMatrix.VWN &&
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
         xgemm.MDIMC == other.xgemm.MDIMC && xgemm.NDIMC == other.xgemm.NDIMC && xgemm.MWG == other.xgemm.MWG &&
         xgemm.NWG == other.xgemm.NWG && xgemm.KWG == other.xgemm.KWG && xgemm.KWI == other.xgemm.KWI && xgemm.MDIMA == other.xgemm.MDIMA &&
         xgemm.NDIMB == other.xgemm.NDIMB && xgemm.VWM == other.xgemm.VWM && xgemm.VWN == other.xgemm.VWN &&
         xgemm16.MDIMC == other.xgemm16.MDIMC && xgemm16.NDIMC == other.xgemm16.NDIMC &&
         xgemm16.MWG == other.xgemm16.MWG && xgemm16.NWG == other.xgemm16.NWG &&
         xgemm16.KWG == other.xgemm16.KWG && xgemm16.KWI == other.xgemm16.KWI && xgemm16.MDIMA == other.xgemm16.MDIMA &&
         xgemm16.NDIMB == other.xgemm16.NDIMB && xgemm16.VWM == other.xgemm16.VWM && xgemm16.VWN == other.xgemm16.VWN && xgemmDirect.WGD == other.xgemmDirect.WGD &&
         xgemmDirect.MDIMCD == other.xgemmDirect.MDIMCD && xgemmDirect.NDIMCD == other.xgemmDirect.NDIMCD &&
         xgemmDirect.MDIMAD == other.xgemmDirect.MDIMAD && xgemmDirect.NDIMBD == other.xgemmDirect.NDIMBD &&
         xgemmDirect.KWID == other.xgemmDirect.KWID && xgemmDirect.PADA == other.xgemmDirect.PADA &&
         xgemmDirect.PADB == other.xgemmDirect.PADB && xgemmDirect.VWMD == other.xgemmDirect.VWMD &&
         xgemmDirect.VWND == other.xgemmDirect.VWND &&
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

bool VulkanTuneParams::isValid() const {
  if(vulkan.shouldUseFP16Storage &&
     (!vulkan.canUseFP16Storage || !vulkan.canUseFP16Compute))
    return false;
  if(vulkan.shouldUseFP16Compute && !vulkan.canUseFP16Compute)
    return false;
  if(vulkan.shouldUseCooperativeMatrix && !vulkan.canUseCooperativeMatrix)
    return false;
  if(vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
     (!vulkan.canUseCooperativeMatrix || !vulkan.shouldUseCooperativeMatrix ||
      !vulkan.shouldUseFP16Storage || !vulkan.shouldUseFP16Compute))
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
         vulkan.shouldUseHgemmCooperativeMatrixNCHW == other.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
         vulkan.shouldUseSubgroup == other.vulkan.shouldUseSubgroup &&
         static_cast<const VulkanTuningProfile&>(*this) == static_cast<const VulkanTuningProfile&>(other);
}

namespace {
  void writeTuningProfile(ofstream& out, const string& prefix, const VulkanTuningProfile& profile) {
#define WRITE(name, value) writeParam(out, prefix.empty() ? name : prefix + "." name, value)
    WRITE("xgemmDirect.WGD", profile.xgemmDirect.WGD); WRITE("xgemmDirect.MDIMCD", profile.xgemmDirect.MDIMCD);
    WRITE("xgemmDirect.NDIMCD", profile.xgemmDirect.NDIMCD); WRITE("xgemmDirect.MDIMAD", profile.xgemmDirect.MDIMAD);
    WRITE("xgemmDirect.NDIMBD", profile.xgemmDirect.NDIMBD); WRITE("xgemmDirect.KWID", profile.xgemmDirect.KWID);
    WRITE("xgemmDirect.PADA", profile.xgemmDirect.PADA); WRITE("xgemmDirect.PADB", profile.xgemmDirect.PADB);
    WRITE("xgemmDirect.VWMD", profile.xgemmDirect.VWMD); WRITE("xgemmDirect.VWND", profile.xgemmDirect.VWND);
#define WRITE_XGEMM(name, params) \
    WRITE(name ".MWG", params.MWG); WRITE(name ".NWG", params.NWG); WRITE(name ".KWG", params.KWG); WRITE(name ".KWI", params.KWI); \
    WRITE(name ".MDIMC", params.MDIMC); WRITE(name ".NDIMC", params.NDIMC); \
    WRITE(name ".MDIMA", params.MDIMA); WRITE(name ".NDIMB", params.NDIMB); \
    WRITE(name ".VWM", params.VWM); WRITE(name ".VWN", params.VWN)
    WRITE_XGEMM("xgemm", profile.xgemm); WRITE_XGEMM("xgemm16", profile.xgemm16);
#undef WRITE_XGEMM
#define WRITE_HGEMM(name, params) \
    WRITE(name ".MWG", params.MWG); WRITE(name ".NWG", params.NWG); WRITE(name ".KWG", params.KWG); \
    WRITE(name ".MWAVE", params.MWAVE); WRITE(name ".NWAVE", params.NWAVE); \
    WRITE(name ".MWARP", params.MWARP); WRITE(name ".NWARP", params.NWARP); \
    WRITE(name ".VWM", params.VWM); WRITE(name ".VWN", params.VWN); \
    WRITE(name ".KDIM", params.KDIM); WRITE(name ".subgroupSize", params.subgroupSize); \
    WRITE(name ".accType", params.accType)
    WRITE_HGEMM("hgemmCooperativeMatrix", profile.hgemmCooperativeMatrix);
    WRITE("hgemmCooperativeMatrix.SA", profile.hgemmCooperativeMatrix.SA); WRITE("hgemmCooperativeMatrix.SB", profile.hgemmCooperativeMatrix.SB);
    WRITE_HGEMM("hgemmCooperativeMatrixNCHW", profile.hgemmCooperativeMatrixNCHW);
    WRITE("hgemmCooperativeMatrixNCHW.SB", profile.hgemmCooperativeMatrixNCHW.SB);
#undef WRITE_HGEMM
#define WRITE_CONV(name, params) \
    WRITE(name ".inTileXSize", params.inTileXSize); WRITE(name ".inTileYSize", params.inTileYSize); \
    WRITE(name ".outTileXSize", params.outTileXSize); WRITE(name ".outTileYSize", params.outTileYSize); \
    WRITE(name ".inputTransformLocalXSize", params.inputTransformLocalXSize); WRITE(name ".inputTransformLocalYSize", params.inputTransformLocalYSize); \
    WRITE(name ".outputTransformLocalXSize", params.outputTransformLocalXSize); WRITE(name ".outputTransformLocalYSize", params.outputTransformLocalYSize); WRITE(name ".outputTransformLocalZSize", params.outputTransformLocalZSize)
    WRITE_CONV("conv3x3", profile.conv3x3); WRITE_CONV("conv5x5", profile.conv5x5);
#undef WRITE_CONV
    WRITE("gPool.XYSTRIDE", profile.gPool.XYSTRIDE); WRITE("gPool.CHANNELSTRIDE", profile.gPool.CHANNELSTRIDE); WRITE("gPool.BATCHSTRIDE", profile.gPool.BATCHSTRIDE);
    WRITE("transformer.ATTN_BLOCK_Q", profile.transformer.ATTN_BLOCK_Q); WRITE("transformer.ATTN_BLOCK_KV", profile.transformer.ATTN_BLOCK_KV); WRITE("transformer.Q_PER_THREAD", profile.transformer.Q_PER_THREAD); WRITE("transformer.USE_TILED_ATTN", profile.transformer.USE_TILED_ATTN);
    WRITE("rmsNorm.WG_C_SIZE", profile.rmsNorm.WG_C_SIZE); WRITE("rmsNorm.WG_XY_SIZE", profile.rmsNorm.WG_XY_SIZE); WRITE("rmsNorm.C_PER_THREAD", profile.rmsNorm.C_PER_THREAD);
    WRITE("pointwise.ELTS_PER_THREAD", profile.pointwise.ELTS_PER_THREAD); WRITE("pointwise.LOCAL_SIZE", profile.pointwise.LOCAL_SIZE);
    WRITE("addChannelBiases.XY_ELTS_PER_THREAD", profile.addChannelBiases.XY_ELTS_PER_THREAD); WRITE("addChannelBiases.NC_ELTS_PER_THREAD", profile.addChannelBiases.NC_ELTS_PER_THREAD);
    WRITE("spatialRMSNorm.TILE_SIZE", profile.spatialRMSNorm.TILE_SIZE); WRITE("spatialRMSNorm.APPLY_ELTS_PER_THREAD", profile.spatialRMSNorm.APPLY_ELTS_PER_THREAD);
#undef WRITE
  }
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
  writeTuningProfile(out, "", static_cast<const VulkanTuningProfile&>(config));
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
  if(values.size() != 100)
    throw IOError("VulkanTuneParams::load: unexpected number of parameters in " + filename);

  const auto readProfile = [&](const string& prefix, VulkanTuningProfile& profile) {
    const auto read = [&](const string& name) {
      return getParam(values, prefix.empty() ? name : prefix + "." + name, filename);
    };
    profile.xgemmDirect.WGD = read("xgemmDirect.WGD"); profile.xgemmDirect.MDIMCD = read("xgemmDirect.MDIMCD"); profile.xgemmDirect.NDIMCD = read("xgemmDirect.NDIMCD"); profile.xgemmDirect.MDIMAD = read("xgemmDirect.MDIMAD"); profile.xgemmDirect.NDIMBD = read("xgemmDirect.NDIMBD"); profile.xgemmDirect.KWID = read("xgemmDirect.KWID"); profile.xgemmDirect.PADA = read("xgemmDirect.PADA"); profile.xgemmDirect.PADB = read("xgemmDirect.PADB"); profile.xgemmDirect.VWMD = read("xgemmDirect.VWMD"); profile.xgemmDirect.VWND = read("xgemmDirect.VWND");
#define READ_XGEMM(name, params) \
    params.MWG = read(name ".MWG"); params.NWG = read(name ".NWG"); params.KWG = read(name ".KWG"); params.KWI = read(name ".KWI"); \
    params.MDIMC = read(name ".MDIMC"); params.NDIMC = read(name ".NDIMC"); params.MDIMA = read(name ".MDIMA"); params.NDIMB = read(name ".NDIMB"); \
    params.VWM = read(name ".VWM"); params.VWN = read(name ".VWN")
    READ_XGEMM("xgemm", profile.xgemm); READ_XGEMM("xgemm16", profile.xgemm16);
#undef READ_XGEMM
#define READ_HGEMM(name, params) \
    params.MWG = read(name ".MWG"); params.NWG = read(name ".NWG"); params.KWG = read(name ".KWG"); \
    params.MWAVE = read(name ".MWAVE"); params.NWAVE = read(name ".NWAVE"); params.MWARP = read(name ".MWARP"); params.NWARP = read(name ".NWARP"); \
    params.VWM = read(name ".VWM"); params.VWN = read(name ".VWN"); params.KDIM = read(name ".KDIM"); params.subgroupSize = read(name ".subgroupSize"); \
    params.accType = read(name ".accType")
    READ_HGEMM("hgemmCooperativeMatrix", profile.hgemmCooperativeMatrix); profile.hgemmCooperativeMatrix.SA = read("hgemmCooperativeMatrix.SA"); profile.hgemmCooperativeMatrix.SB = read("hgemmCooperativeMatrix.SB");
    READ_HGEMM("hgemmCooperativeMatrixNCHW", profile.hgemmCooperativeMatrixNCHW); profile.hgemmCooperativeMatrixNCHW.SB = read("hgemmCooperativeMatrixNCHW.SB");
#undef READ_HGEMM
#define READ_CONV(name, params) \
    params.inTileXSize = read(name ".inTileXSize"); params.inTileYSize = read(name ".inTileYSize"); params.outTileXSize = read(name ".outTileXSize"); params.outTileYSize = read(name ".outTileYSize"); \
    params.inputTransformLocalXSize = read(name ".inputTransformLocalXSize"); params.inputTransformLocalYSize = read(name ".inputTransformLocalYSize"); params.outputTransformLocalXSize = read(name ".outputTransformLocalXSize"); params.outputTransformLocalYSize = read(name ".outputTransformLocalYSize"); params.outputTransformLocalZSize = read(name ".outputTransformLocalZSize")
    READ_CONV("conv3x3", profile.conv3x3); READ_CONV("conv5x5", profile.conv5x5);
#undef READ_CONV
    profile.gPool.XYSTRIDE = read("gPool.XYSTRIDE"); profile.gPool.CHANNELSTRIDE = read("gPool.CHANNELSTRIDE"); profile.gPool.BATCHSTRIDE = read("gPool.BATCHSTRIDE");
    profile.transformer.ATTN_BLOCK_Q = read("transformer.ATTN_BLOCK_Q"); profile.transformer.ATTN_BLOCK_KV = read("transformer.ATTN_BLOCK_KV"); profile.transformer.Q_PER_THREAD = read("transformer.Q_PER_THREAD"); profile.transformer.USE_TILED_ATTN = read("transformer.USE_TILED_ATTN");
    profile.rmsNorm.WG_C_SIZE = read("rmsNorm.WG_C_SIZE"); profile.rmsNorm.WG_XY_SIZE = read("rmsNorm.WG_XY_SIZE"); profile.rmsNorm.C_PER_THREAD = read("rmsNorm.C_PER_THREAD");
    profile.pointwise.ELTS_PER_THREAD = read("pointwise.ELTS_PER_THREAD"); profile.pointwise.LOCAL_SIZE = read("pointwise.LOCAL_SIZE");
    profile.addChannelBiases.XY_ELTS_PER_THREAD = read("addChannelBiases.XY_ELTS_PER_THREAD"); profile.addChannelBiases.NC_ELTS_PER_THREAD = read("addChannelBiases.NC_ELTS_PER_THREAD");
    profile.spatialRMSNorm.TILE_SIZE = read("spatialRMSNorm.TILE_SIZE"); profile.spatialRMSNorm.APPLY_ELTS_PER_THREAD = read("spatialRMSNorm.APPLY_ELTS_PER_THREAD");
  };

  VulkanTuneParams config;
  config.vulkan.canUseFP16Storage = getBoolParam(values, "vulkan.canUseFP16Storage", filename);
  config.vulkan.canUseFP16Compute = getBoolParam(values, "vulkan.canUseFP16Compute", filename);
  config.vulkan.canUseCooperativeMatrix = getBoolParam(values, "vulkan.canUseCooperativeMatrix", filename);
  config.vulkan.canUseSubgroup = getBoolParam(values, "vulkan.canUseSubgroup", filename);
  config.vulkan.shouldUseFP16Storage = getBoolParam(values, "vulkan.shouldUseFP16Storage", filename);
  config.vulkan.shouldUseFP16Compute = getBoolParam(values, "vulkan.shouldUseFP16Compute", filename);
  config.vulkan.shouldUseCooperativeMatrix = getBoolParam(values, "vulkan.shouldUseCooperativeMatrix", filename);
  config.vulkan.shouldUseHgemmCooperativeMatrixNCHW = getBoolParam(values, "vulkan.shouldUseHgemmCooperativeMatrixNCHW", filename);
  config.vulkan.shouldUseSubgroup = getBoolParam(values, "vulkan.shouldUseSubgroup", filename);
  readProfile("", static_cast<VulkanTuningProfile&>(config));
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
    HGemmCooperativeMatrixNCHWTuneParams& params,
    int accType
  ) {
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

    for(const VkCooperativeMatrixPropertiesKHR& property: properties) {
      if(!supportsCooperativeMatrixType(property, accType))
        continue;
      params.accType = accType;
      params.MWARP = static_cast<int>(property.MSize);
      params.NWARP = static_cast<int>(property.NSize);
      params.KDIM = static_cast<int>(property.KSize);
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

    for(const VkCooperativeMatrixPropertiesKHR& property: properties) {
      if(!supportsCooperativeMatrixType(property, accType))
        continue;
      params.accType = accType;
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
    Logger* logger;
    VulkanTimestampTimer* timer;
    bool printOnlyOnImprovement;
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
        tunerName, totalRuns, 0, tolerance, tolerance * 5.0,
        batchSizes, gemmCases, workloadWeights
      };
    }
    if(tunerName == "pointwise" || tunerName == "transformerRMSNorm" || tunerName == "spatialRMSNorm")
      return {tunerName, 20, 0, 0.05, 0.25, batchSizes, {}, workloadWeights};
    if(tunerName == "transformerAttention")
      return {tunerName, 12, 0, 0.005, 0.025, batchSizes, {}, workloadWeights};
    return {tunerName, 20, 0, 0.005, 0.025, batchSizes, {}, workloadWeights};
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

  template<typename Tuner>
  struct KeepsCurrentConfigFirst {
    static constexpr bool value = false;
  };

  template<typename Tuner>
  struct StopsOnReferenceImplFail {
    static bool value(const VulkanTuneParams&) { return false; }
  };

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
      add("VWMD", config.xgemmDirect.VWMD);
      add("VWND", config.xgemmDirect.VWND);
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

  void logTuningProgress(
    const TuningContext& context,
    size_t candidateIndex,
    size_t totalCandidates,
    const vector<const Pipeline*>& pipelines,
    const string& tunerName
  ) {
    const string pipelineNames = describeTuningPipelines(pipelines, tunerName);
    writeTuningLog(
      context,
      "Tuning " + pipelineNames + " " + to_string(candidateIndex) + "/" + to_string(totalCandidates) + " ..."
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
    ) {
      errorProp = numeric_limits<double>::quiet_NaN();
      if(!isUsable()) {
        error = "compute timestamps are not supported";
        return false;
      }
      if(pipelines.empty()) {
        error = "no pipeline was created";
        return false;
      }

      uint32_t descriptorCount = 0;
      for(const Pipeline* pipeline: pipelines)
        descriptorCount += pipeline->bindingCount;
      if(descriptorCount == 0) {
        error = "pipeline has no descriptor bindings";
        return false;
      }

      ReusableResources resources(device);
      activeResources = &resources;
      const auto activeResourcesGuard = makeScopeGuard([&]() { activeResources = nullptr; });

      VkResult result = VK_SUCCESS;
      const size_t batchSize = static_cast<size_t>(std::max(1, context.batchSize));
      const size_t logicalXYSize = static_cast<size_t>(std::max(1, context.nnXLen * context.nnYLen));
      const bool useNCHWCooperativeMatrix =
        config.vulkan.canUseCooperativeMatrix &&
        config.vulkan.shouldUseCooperativeMatrix &&
        config.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
        config.hgemmCooperativeMatrixNCHW.isValid();
      const bool usePaddedNCHWXY =
        plan.kernelName == "hgemmCooperativeMatrixNCHW" || useNCHWCooperativeMatrix;
      const bool usePaddedGpoolXY =
        plan.kernelName == "gPool" &&
        config.vulkan.canUseFP16Storage &&
        config.vulkan.canUseFP16Compute &&
        config.vulkan.shouldUseFP16Storage &&
        config.vulkan.shouldUseFP16Compute &&
        config.vulkan.canUseCooperativeMatrix &&
        config.vulkan.shouldUseCooperativeMatrix &&
        config.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
        config.hgemmCooperativeMatrixNCHW.isValid();
      const size_t xySize = usePaddedNCHWXY || usePaddedGpoolXY
        ? vk_helper::roundUpToMultiple(logicalXYSize, static_cast<size_t>(std::max(16, config.hgemmCooperativeMatrixNCHW.MWARP)))
        : logicalXYSize;
      const bool addChannelBiasesUsingFP16Storage =
        config.vulkan.canUseFP16Storage &&
        config.vulkan.canUseFP16Compute &&
        config.vulkan.shouldUseFP16Storage;
      const bool addChannelBiasesUsingFP16TensorCoresFor1x1 =
        config.vulkan.canUseFP16Storage &&
        config.vulkan.canUseFP16Compute &&
        config.vulkan.canUseCooperativeMatrix &&
        config.vulkan.shouldUseCooperativeMatrix &&
        config.vulkan.shouldUseFP16Storage &&
        config.vulkan.shouldUseFP16Compute &&
        config.vulkan.shouldUseHgemmCooperativeMatrixNCHW &&
        addChannelBiasesUsingFP16Storage;
      const size_t addChannelBiasesXYSize = addChannelBiasesUsingFP16TensorCoresFor1x1
        ? vk_helper::roundUpToMultiple(
            xySize, static_cast<size_t>(std::max(16, config.hgemmCooperativeMatrixNCHW.MWARP))
          )
        : xySize;
      const XgemmTuneParams& xgemmParams =
        config.vulkan.shouldUseFP16Compute ? config.xgemm16 : config.xgemm;
      const bool useHgemmCooperativeMatrixForPadding =
        config.vulkan.canUseCooperativeMatrix &&
        config.vulkan.shouldUseCooperativeMatrix &&
        config.vulkan.canUseFP16Storage &&
        config.vulkan.canUseFP16Compute &&
        config.vulkan.shouldUseFP16Storage &&
        config.vulkan.shouldUseFP16Compute &&
        config.hgemmCooperativeMatrix.isValid();
      const int activeMPaddingMult = useHgemmCooperativeMatrixForPadding
        ? config.hgemmCooperativeMatrix.MWG : xgemmParams.MWG;
      const int activeNPaddingMult = useHgemmCooperativeMatrixForPadding
        ? config.hgemmCooperativeMatrix.NWG : xgemmParams.NWG;
      const int activeKPaddingMult = useHgemmCooperativeMatrixForPadding
        ? config.hgemmCooperativeMatrix.KWG : xgemmParams.KWG;
      const size_t maxChannels = static_cast<size_t>(std::max({
        1, context.modelInfo.trunkNumChannels, context.modelInfo.midNumChannels,
        context.modelInfo.regularNumChannels, context.modelInfo.maxConvChannels1x1,
        context.modelInfo.maxConvChannels3x3, context.modelInfo.gpoolNumChannels,
        context.modelInfo.transformerFFNChannels,
        context.modelInfo.transformerNumHeads * context.modelInfo.transformerHeadDim,
        context.modelInfo.transformerNumKVHeads * context.modelInfo.transformerVHeadDim
      }));
      const size_t maxConvChannels = static_cast<size_t>(std::max({
        1, context.modelInfo.trunkNumChannels, context.modelInfo.midNumChannels,
        context.modelInfo.regularNumChannels, context.modelInfo.maxConvChannels3x3,
        context.modelInfo.gpoolNumChannels
      }));
      const bool isGemm = !plan.gemmCases.empty();
      const bool directGemm = plan.kernelName == "xgemmDirect" || plan.kernelName == "hgemmCooperativeMatrixNCHW";
      const bool cooperative = plan.kernelName == "hgemmCooperativeMatrix" || plan.kernelName == "hgemmCooperativeMatrixNCHW";
      const ConvTuneParams& transformConvParams =
        plan.kernelName.find("5x5") != string::npos ? config.conv5x5 : config.conv3x3;
      const int tilesX = (context.nnXLen + transformConvParams.outTileXSize - 1) / transformConvParams.outTileXSize;
      const int tilesY = (context.nnYLen + transformConvParams.outTileYSize - 1) / transformConvParams.outTileYSize;
      const int logicalM = directGemm ? static_cast<int>(logicalXYSize) : static_cast<int>(batchSize) * tilesX * tilesY;
      const int logicalN = isGemm ? std::max(1, std::accumulate(
        plan.gemmCases.begin(), plan.gemmCases.end(), 0,
        [](int maximum, const GemmTuneCase& gemmCase) { return std::max(maximum, gemmCase.outChannels); }
      )) : static_cast<int>(maxChannels);
      const int logicalK = isGemm ? std::max(1, std::accumulate(
        plan.gemmCases.begin(), plan.gemmCases.end(), 0,
        [](int maximum, const GemmTuneCase& gemmCase) { return std::max(maximum, gemmCase.inChannels); }
      )) : static_cast<int>(maxChannels);
      const int gemmBatch = directGemm ? static_cast<int>(batchSize) :
        config.conv3x3.inTileXSize * config.conv3x3.inTileYSize;
      struct GemmDimensions {
        int logicalM;
        int logicalN;
        int logicalK;
        int gemmM;
        int gemmN;
        int gemmK;
      };
      const auto getGemmDimensions = [&](int inChannels, int outChannels) {
        GemmDimensions dimensions = {
          logicalM, std::max(1, outChannels), std::max(1, inChannels), logicalM,
          std::max(1, outChannels), std::max(1, inChannels)
        };
        if(cooperative && directGemm) {
          dimensions.gemmM = vk_helper::roundUpToMultipleInt(
            dimensions.logicalM, std::max(16, config.hgemmCooperativeMatrixNCHW.MWARP)
          );
          const int align = config.hgemmCooperativeMatrixNCHW.getRequiredCDivisor();
          dimensions.gemmN = vk_helper::roundUpToMultipleInt(dimensions.logicalN, align);
          dimensions.gemmK = vk_helper::roundUpToMultipleInt(dimensions.logicalK, align);
        }
        else if(!directGemm) {
          dimensions.gemmM = vk_helper::roundUpToMultipleInt(
            dimensions.logicalM, cooperative ? config.hgemmCooperativeMatrix.MWG : xgemmParams.MWG
          );
          dimensions.gemmN = vk_helper::roundUpToMultipleInt(
            dimensions.logicalN, cooperative ? config.hgemmCooperativeMatrix.NWG : xgemmParams.NWG
          );
          dimensions.gemmK = vk_helper::roundUpToMultipleInt(
            dimensions.logicalK, cooperative ? config.hgemmCooperativeMatrix.KWG : xgemmParams.KWG
          );
        }
        return dimensions;
      };
      const GemmDimensions maxGemm = getGemmDimensions(logicalK, logicalN);
      const int gemmM = maxGemm.gemmM;
      const int gemmN = maxGemm.gemmN;
      const int gemmK = maxGemm.gemmK;
      const size_t maxTiles = batchSize * ((context.nnXLen + 1) / 2) * ((context.nnYLen + 1) / 2);
      const size_t paddedTiles = vk_helper::roundUpToMultiple(maxTiles, static_cast<size_t>(activeMPaddingMult));
      const size_t paddedChannels = vk_helper::roundUpToMultiple(
        maxChannels, static_cast<size_t>(std::max(activeKPaddingMult, activeNPaddingMult))
      );
      const size_t gemmElements = static_cast<size_t>(gemmBatch) * std::max({
        static_cast<size_t>(gemmM) * gemmK, static_cast<size_t>(gemmN) * gemmK, static_cast<size_t>(gemmM) * gemmN
      });
      const size_t attentionOutputElements = static_cast<size_t>(batchSize) *
        std::max(1, context.modelInfo.transformerNumHeads) *
        std::max(1, context.modelInfo.transformerVHeadDim) * xySize;
      const size_t transformElements = std::max({
        static_cast<size_t>(batchSize) * maxChannels * xySize,
        paddedTiles * paddedChannels * 36,
        attentionOutputElements
      });
      const size_t addChannelBiasesElements = batchSize *
        static_cast<size_t>(std::max(1, context.modelInfo.trunkNumChannels)) * addChannelBiasesXYSize;
      const size_t scratchElements = isGemm ? gemmElements :
        plan.kernelName == "addChannelBiases" ? addChannelBiasesElements : transformElements;
      const size_t scratchBytes = vk_helper::roundUpToMultiple(std::max<size_t>(scratchElements, 4), size_t(4)) * sizeof(float);
      vector<VulkanBuffer*>& tuningBuffers = resources.tuningBuffers;
      VulkanBuffer*& pointwiseAccumulatorInitialBuffer = resources.pointwiseAccumulatorInitialBuffer;
      VkDeviceSize pointwiseAccumulatorBytes = 0;
      vector<VulkanBuffer*>& pointwiseValidationBuffers = resources.pointwiseValidationBuffers;
      vector<VulkanBuffer*>& winogradInputValidationBuffers = resources.winogradInputValidationBuffers;
      vector<VulkanBuffer*>& winogradOutputValidationBuffers = resources.winogradOutputValidationBuffers;
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
      hostFloatBuffers.reserve(descriptorCount);
      struct PendingUpload {
        VulkanBuffer* stagingBuffer;
        VulkanBuffer* destinationBuffer;
        VkDeviceSize size;
      };
      vector<PendingUpload> pendingUploads;
      size_t stagingBufferIndex = 0;
      const auto queueUpload = [&](const void* data, VkDeviceSize size, VulkanBuffer* destination, const string& description) {
        if(stagingBufferIndex == resources.uploadStagingBuffers.size())
          resources.uploadStagingBuffers.push_back(nullptr);
        VulkanBuffer*& stagingBuffer = resources.uploadStagingBuffers[stagingBufferIndex++];
        if(!ensureStagingBuffer(stagingBuffer, size, result, error, description))
          return false;
        memset(stagingBuffer->allocationInfo.pMappedData, 0, static_cast<size_t>(stagingBuffer->requestedSize));
        memcpy(stagingBuffer->allocationInfo.pMappedData, data, static_cast<size_t>(size));
        result = vmaFlushAllocation(device->allocator, stagingBuffer->allocation, 0, stagingBuffer->requestedSize);
        if(result != VK_SUCCESS) {
          error = "could not prepare " + description + ": " + vk_helper::vkErrorToString(result);
          return false;
        }
        pendingUploads.push_back({stagingBuffer, destination, size});
        return true;
      };
      size_t tuningBufferIndex = 0;
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
              const string& name = pipeline->name;
              const bool winogradInputTransform =
                (plan.kernelName == "conv3x3InputTransform" || plan.kernelName == "conv5x5InputTransform") &&
                name.find("winograd_input_transform") == 0;
              const bool winogradOutputTransform =
                (plan.kernelName == "conv3x3OutputTransform" || plan.kernelName == "conv5x5OutputTransform") &&
                name.find("winograd_output_transform") == 0;
              const bool globalPoolingInput =
                plan.kernelName == "gPool" && name.find("global_pooling_channels") == 0 && binding == 0;
              const bool addPointwiseInput =
                plan.kernelName == "pointwise" && name.find("add_pointwise") == 0 && binding == 1;
              const bool swigluInput =
                plan.kernelName == "pointwise" && name.find("transformer_swiglu") == 0 && binding < 2;
              const bool addChannelBiasesInput =
                plan.kernelName == "addChannelBiases" && name.find("add_channel_bias_nchw") == 0 && binding == 1;
              const bool transformerAttentionInput =
                plan.kernelName == "transformerAttention" &&
                name.find("transformer_scale_dot_product") == 0 && binding < 3;
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
              if(winogradInputTransform && binding == 0) {
                const int inputChannels = static_cast<int>(maxConvChannels);
                for(size_t n = 0; n < batchSize; n++)
                  for(int c = 0; c < inputChannels; c++)
                    for(size_t xy = 0; xy < logicalXYSize; xy++)
                      data[(n * inputChannels + c) * logicalXYSize + xy] =
                        static_cast<float>(rand.nextDouble());
              }
              else if(winogradOutputTransform && binding == 0) {
                const size_t tileElements = static_cast<size_t>(transformConvParams.inTileXSize) * transformConvParams.inTileYSize;
                for(size_t tileElement = 0; tileElement < tileElements; tileElement++)
                  for(size_t channel = 0; channel < maxConvChannels; channel++)
                    for(size_t tile = 0; tile < maxTiles; tile++)
                      data[(tileElement * paddedChannels + channel) * paddedTiles + tile] =
                        static_cast<float>(rand.nextDouble());
              }
              else if(globalPoolingInput) {
                const int gpoolChannels = std::max(1, context.modelInfo.gpoolNumChannels);
                for(size_t n = 0; n < batchSize; n++)
                  for(int c = 0; c < gpoolChannels; c++)
                    for(size_t xy = 0; xy < logicalXYSize; xy++)
                      data[(n * gpoolChannels + c) * xySize + xy] =
                        static_cast<float>(rand.nextDouble());
              }
              else if(addPointwiseInput || swigluInput) {
                const size_t channels = addPointwiseInput
                  ? static_cast<size_t>(std::max(1, context.modelInfo.trunkNumChannels))
                  : static_cast<size_t>(std::max(context.modelInfo.trunkNumChannels, context.modelInfo.transformerFFNChannels));
                const size_t validElements = batchSize * channels * logicalXYSize;
                for(size_t i = 0; i < validElements; i++)
                  data[i] = static_cast<float>(rand.nextDouble());
              }
              else if(addChannelBiasesInput) {
                const size_t validBiases = batchSize * static_cast<size_t>(std::max(1, context.modelInfo.trunkNumChannels));
                for(size_t i = 0; i < validBiases; i++)
                  data[i] = static_cast<float>(rand.nextDouble());
              }
              else if(transformerAttentionInput) {
                const size_t heads = static_cast<size_t>(std::max(1, context.modelInfo.transformerNumHeads));
                const size_t kvHeads = static_cast<size_t>(std::max(1, context.modelInfo.transformerNumKVHeads));
                const size_t headDim = static_cast<size_t>(std::max(1, context.modelInfo.transformerHeadDim));
                const size_t vHeadDim = static_cast<size_t>(std::max(1, context.modelInfo.transformerVHeadDim));
                const size_t channels = binding == 0 ? heads : kvHeads;
                const size_t dimension = binding == 2 ? vHeadDim : headDim;
                const size_t validElements = batchSize * channels * dimension * logicalXYSize;
                for(size_t i = 0; i < validElements; i++)
                  data[i] = static_cast<float>(rand.nextDouble());
              }
              else if(transformerRMSNormInput) {
                const size_t channels = static_cast<size_t>(std::max(1, context.modelInfo.trunkNumChannels));
                const size_t validElements = batchSize * channels * logicalXYSize;
                for(size_t i = 0; i < validElements; i++)
                  data[i] = static_cast<float>(rand.nextDouble());
              }
              else if(transformerRMSNormGamma) {
                const size_t channels = static_cast<size_t>(std::max(1, context.modelInfo.trunkNumChannels));
                for(size_t i = 0; i < channels; i++)
                  data[i] = static_cast<float>(rand.nextDouble());
              }
              else if(spatialRMSNormInput) {
                const size_t channels = static_cast<size_t>(std::max(1, context.modelInfo.trunkNumChannels));
                const size_t validElements = batchSize * channels * logicalXYSize;
                for(size_t i = 0; i < validElements; i++)
                  data[i] = static_cast<float>(rand.nextDouble());
              }
              else if(spatialRMSNormGamma || spatialRMSNormBeta) {
                const size_t channels = static_cast<size_t>(std::max(1, context.modelInfo.trunkNumChannels));
                for(size_t i = 0; i < channels; i++)
                  data[i] = static_cast<float>(rand.nextDouble());
              }
              else if(!winogradInputTransform && !winogradOutputTransform) {
                for(float& value: data)
                  value = static_cast<float>(rand.nextDouble());
                // Masks and their sums describe a fully valid board.
                const bool mask =
                  (name.find("global_pooling_channels") == 0 && binding == 2) ||
                  (name.find("transformer_scale_dot_product") == 0 && binding == 4) ||
                  ((name.find("transformer_rms_norm") == 0 || name.find("transformer_spatial_rms_norm_apply") == 0) && binding == 4) ||
                  (name.find("transformer_spatial_rms_norm_sum_sq") == 0 && binding == 1);
                const bool maskSum =
                  (name.find("global_pooling_channels") == 0 && binding == 3) ||
                  (name.find("value_head_pool_channels") == 0 && binding == 2) ||
                  (name.find("transformer_spatial_rms_norm_apply") == 0 && binding == 5);
                if(mask) {
                  std::fill(data.begin(), data.end(), 0.0f);
                  const int maskBatchSize = std::max(1, context.batchSize);
                  for(int n = 0; n < maskBatchSize; n++)
                    std::fill(data.begin() + static_cast<size_t>(n) * xySize,
                              data.begin() + static_cast<size_t>(n) * xySize + logicalXYSize, 1.0f);
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
          if(tuningBufferIndex == tuningBuffers.size())
            tuningBuffers.push_back(nullptr);
          if(!ensureBuffer(tuningBuffers[tuningBufferIndex], scratchBytes, result, error, "tuning buffer"))
            return false;
          VulkanBuffer* buffer = tuningBuffers[tuningBufferIndex++];
          if(!queueUpload(initialData, initialBytes, buffer, "tuning buffer"))
            return false;

          // In-place add pipelines write binding 0. Keep an immutable copy of
          // their initialized input so every measured invocation has the same
          // input, as in the OpenCL tuner.
          if((pipeline->name.find("add_pointwise") == 0 ||
              pipeline->name.find("add_channel_bias_nchw") == 0) && binding == 0) {
            if(!ensureBuffer(pointwiseAccumulatorInitialBuffer, scratchBytes, result, error, "pointwise reset buffer"))
              return false;
            if(!queueUpload(initialData, initialBytes, pointwiseAccumulatorInitialBuffer, "pointwise reset buffer"))
              return false;
            pointwiseAccumulatorBytes = initialBytes;
          }
        }
      }
      if(isGemm && cpuReference != nullptr) {
        // The reference uses logical float inputs, before half quantization.
        for(const GemmTuneCase& gemmCase: plan.gemmCases) {
          const GemmDimensions dimensions = getGemmDimensions(gemmCase.inChannels, gemmCase.outChannels);
          for(int n = 0; n < gemmBatch; n++)
            for(int y = 0; y < dimensions.logicalN; y++)
              for(int x = 0; x < dimensions.logicalM; x++) {
                double sum = 0.0;
                for(int k = 0; k < dimensions.logicalK; k++)
                  sum += static_cast<double>(gemmInput[
                    (static_cast<size_t>(n) * dimensions.gemmK + k) * dimensions.gemmM + x
                  ]) * gemmFilter[
                    (static_cast<size_t>(directGemm ? 0 : n) * dimensions.gemmK + k) * dimensions.gemmN + y
                  ];
                cpuReference->push_back(static_cast<float>(sum));
              }
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
            cpuReference->push_back(accum[i] + value[i]);
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
          const size_t count = static_cast<size_t>(cpuBatchSize) * cpuChannels * addChannelBiasesXYSize;
          for(size_t i = 0; i < count; i++)
            cpuReference->push_back(accum[i] + bias[i / addChannelBiasesXYSize]);
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
        const int numValidationRepeats =
          plan.kernelName == "gPool" ? 10 :
          plan.kernelName == "transformerAttention" ? 6 :
          plan.kernelName == "pointwise" || plan.kernelName == "addChannelBiases" ||
          plan.kernelName == "transformerRMSNorm" ||
          plan.kernelName == "spatialRMSNorm" ? 10 : 1;
        if(plan.kernelName == "pointwise") {
          for(int repeat = 0; repeat < numValidationRepeats; repeat++) {
            size_t firstBuffer = 0;
            for(const Pipeline* pipeline: pipelines) {
              if(!appendCpuReference(pipeline, firstBuffer))
                return false;
              firstBuffer += pipeline->bindingCount;
            }
          }
        }
        else {
          size_t firstBuffer = 0;
          for(const Pipeline* pipeline: pipelines) {
            for(int repeat = 0; repeat < numValidationRepeats; repeat++) {
              if(!appendCpuReference(pipeline, firstBuffer))
                return false;
            }
            firstBuffer += pipeline->bindingCount;
          }
        }
      }

      if(plan.kernelName == "pointwise" || plan.kernelName == "addChannelBiases" ||
         plan.kernelName == "transformerRMSNorm") {
        const size_t validationCount = plan.kernelName == "pointwise" ? 10 * pipelines.size() : 10;
        if(!ensureBuffers(pointwiseValidationBuffers, validationCount, scratchBytes, result, error, "pointwise validation buffer"))
          return false;
      }
      else if(plan.kernelName == "spatialRMSNorm") {
        if(!ensureBuffers(spatialValidationBuffers, 10, scratchBytes, result, error, "spatial RMSNorm validation buffer"))
          return false;
      }
      else if(plan.kernelName.find("OutputTransform") != string::npos) {
        if(!ensureBuffers(winogradOutputValidationBuffers, 10, scratchBytes, result, error, "Winograd output validation buffer"))
          return false;
      }
      else if(plan.kernelName.find("InputTransform") != string::npos) {
        if(!ensureBuffers(winogradInputValidationBuffers, 10, scratchBytes, result, error, "Winograd input validation buffer"))
          return false;
      }
      else if(plan.kernelName == "gPool") {
        if(!ensureBuffers(gpoolValidationBuffers, 10, scratchBytes, result, error, "global pooling validation buffer"))
          return false;
      }
      else if(plan.kernelName == "transformerAttention") {
        if(!ensureBuffers(attentionValidationBuffers, 6, scratchBytes, result, error, "attention validation buffer"))
          return false;
      }
      else if(isGemm) {
        if(!ensureBuffers(gemmValidationBuffers, plan.gemmCases.size(), scratchBytes, result, error, "GEMM validation buffer"))
          return false;
      }

      if(!ensureDescriptorPool(
        descriptorCount, static_cast<uint32_t>(pipelines.size()), result, error
      ))
        return false;

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

      const size_t firstTimedPipeline = plan.kernelName == "spatialRMSNorm" ? 2 : 0;
      const size_t timedPipelineCount = plan.kernelName == "spatialRMSNorm" ? 1 : pipelines.size();
      if(!ensureQueryPool(
        static_cast<uint32_t>(2 * plan.timedRuns() * timedPipelineCount), result, error
      ))
        return false;
      if(!ensureCommandResources(result, error))
        return false;

      const auto recordPipeline = [&](
        VkCommandBuffer targetCommandBuffer,
        const Pipeline* pipeline,
        VkDescriptorSet descriptorSet,
        int runBatchSize,
        int runChannels,
        int runGemmM,
        int runGemmN,
        int runGemmK
      ) {
        const int batchSize = std::max(1, runBatchSize);
        const int logicalPipelineXYSize = std::max(1, context.nnXLen * context.nnYLen);
        const int pipelineXYSize = plan.kernelName == "gPool"
          ? static_cast<int>(xySize) : logicalPipelineXYSize;
        const int channels = std::max(1, runChannels);
        const auto dispatch = [&](uint32_t x, uint32_t y = 1, uint32_t z = 1) {
          vkCmdDispatch(targetCommandBuffer, std::max(1u, x), std::max(1u, y), std::max(1u, z));
        };
        const auto push = [&](const auto& params) {
          vkCmdPushConstants(targetCommandBuffer, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
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
        else if(pipeline->name.find("hgemm_cooperative_matrix_") == 0) {
          vk_shader::push::HGemmCooperativeMatrixParams params = {runGemmM, runGemmN, runGemmK};
          push(params);
          dispatch(runGemmM / config.hgemmCooperativeMatrix.MWG, runGemmN / config.hgemmCooperativeMatrix.NWG, gemmBatch);
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
        else if(pipeline->name.find("xgemm_strided_batched") == 0) {
          vk_shader::push::XgemmStridedBatchedFp32Params params = {
            static_cast<uint32_t>(runGemmM), static_cast<uint32_t>(runGemmN), static_cast<uint32_t>(runGemmK),
            static_cast<uint32_t>(runGemmM), static_cast<uint32_t>(runGemmM * runGemmK),
            static_cast<uint32_t>(runGemmN), 0,
            static_cast<uint32_t>(runGemmM), static_cast<uint32_t>(runGemmM * runGemmN), 0
          };
          push(params);
          dispatch(
            (runGemmM + config.xgemmDirect.WGD - 1) / config.xgemmDirect.WGD,
            (runGemmN + config.xgemmDirect.WGD - 1) / config.xgemmDirect.WGD, gemmBatch
          );
        }
        else if(pipeline->name.find("winograd_input_transform") == 0) {
          const vk_shader::tune::ConvTuneParams& convParams =
            pipeline->name.find("5x5") != string::npos ? config.conv5x5 : config.conv3x3;
          const int outTile = convParams.outTileXSize;
          const int tilesX = (context.nnXLen + outTile - 1) / outTile;
          const int tilesY = (context.nnYLen + outTile - 1) / outTile;
          const int paddedTiles = vk_helper::roundUpToMultipleInt(batchSize * tilesX * tilesY, activeMPaddingMult);
          const int paddedChannels = vk_helper::roundUpToMultipleInt(channels, activeKPaddingMult);
          vk_shader::push::WinogradInputTransformParams params = {
            batchSize,context.nnXLen,context.nnYLen,tilesX,tilesY,channels,paddedChannels,paddedTiles,pipelineXYSize
          };
          push(params);
          dispatch(
            static_cast<uint32_t>((params.ntxtySizePadded + pipeline->localSizeX - 1) / pipeline->localSizeX),
            static_cast<uint32_t>((paddedChannels + pipeline->localSizeY - 1) / pipeline->localSizeY)
          );
        }
        else if(pipeline->name.find("winograd_output_transform") == 0) {
          const vk_shader::tune::ConvTuneParams& convParams =
            pipeline->name.find("5x5") != string::npos ? config.conv5x5 : config.conv3x3;
          const int outTile = convParams.outTileXSize;
          const int tilesX = (context.nnXLen + outTile - 1) / outTile;
          const int tilesY = (context.nnYLen + outTile - 1) / outTile;
          const int paddedTiles = vk_helper::roundUpToMultipleInt(batchSize * tilesX * tilesY, activeMPaddingMult);
          const int paddedChannels = vk_helper::roundUpToMultipleInt(channels, activeNPaddingMult);
          vk_shader::push::WinogradOutputTransformParams params = {
            batchSize,context.nnYLen,context.nnXLen,tilesY,tilesX,channels,paddedChannels,paddedTiles,pipelineXYSize
          };
          push(params);
          dispatch(
            static_cast<uint32_t>((vk_helper::powerOf2ify(tilesX) + pipeline->localSizeX - 1) / pipeline->localSizeX),
            static_cast<uint32_t>((vk_helper::powerOf2ify(tilesY) + pipeline->localSizeY - 1) / pipeline->localSizeY),
            static_cast<uint32_t>((batchSize * channels + pipeline->localSizeZ - 1) / pipeline->localSizeZ)
          );
        }
        else if(pipeline->name.find("global_pooling_channels") == 0) {
          const int gpoolChannels = std::max(1, context.modelInfo.gpoolNumChannels);
          vk_shader::push::GlobalPoolingChannelsParams params = {batchSize,gpoolChannels,pipelineXYSize};
          push(params);
          dispatch(
            1,
            static_cast<uint32_t>((gpoolChannels + pipeline->localSizeY - 1) / pipeline->localSizeY),
            static_cast<uint32_t>((batchSize + pipeline->localSizeZ - 1) / pipeline->localSizeZ)
          );
        }
        else if(pipeline->name.find("value_head_pool_channels") == 0) {
          const int gpoolChannels = std::max(1, context.modelInfo.gpoolNumChannels);
          vk_shader::push::ValueHeadPoolingChannelsParams params = {batchSize,gpoolChannels,pipelineXYSize};
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
          dispatch(1, 1, static_cast<uint32_t>((batchSize + pipeline->localSizeZ - 1) / pipeline->localSizeZ));
        }
        else if(pipeline->name.find("add_pointwise") == 0) {
          vk_shader::push::AddPointWiseParams params = {static_cast<uint32_t>(batchSize * channels * pipelineXYSize)};
          push(params);
          dispatch((params.size + config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX - 1) /
                   (config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX));
        }
        else if(pipeline->name.find("add_channel_bias_nchw") == 0) {
          vk_shader::push::AddChannelBiasNCHWParams params = {
            static_cast<uint32_t>(batchSize * channels), static_cast<uint32_t>(addChannelBiasesXYSize)
          };
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
          vk_shader::push::ScaleDotProductPushParam params = {pipelineXYSize,heads,kvHeads,1.0f / sqrtf((float)std::max(1, context.modelInfo.transformerHeadDim))};
          push(params);
          if(config.transformer.USE_TILED_ATTN && pipeline->name.find("naive") == string::npos)
            dispatch((pipelineXYSize + config.transformer.ATTN_BLOCK_Q * config.transformer.Q_PER_THREAD - 1) /
                       (config.transformer.ATTN_BLOCK_Q * config.transformer.Q_PER_THREAD), static_cast<uint32_t>(batchSize * heads));
          else
            dispatch((pipelineXYSize + pipeline->localSizeX - 1) / pipeline->localSizeX, static_cast<uint32_t>(batchSize * heads));
        }
        else if(pipeline->name.find("transformer_rms_norm") == 0) {
          vk_shader::push::TransformerRMSNormPushParams params = {batchSize,channels,pipelineXYSize,1e-6f};
          push(params);
          dispatch(
            static_cast<uint32_t>((pipelineXYSize + config.rmsNorm.WG_XY_SIZE - 1) / config.rmsNorm.WG_XY_SIZE),
            static_cast<uint32_t>(batchSize)
          );
        }
        else if(pipeline->name.find("transformer_swiglu") == 0) {
          const int ffnChannels = std::max(channels, context.modelInfo.transformerFFNChannels);
          vk_shader::push::TransformerSwiGLUPushParams params = {batchSize * ffnChannels * pipelineXYSize};
          push(params);
          dispatch((params.size + config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX - 1) /
                   (config.pointwise.ELTS_PER_THREAD * pipeline->localSizeX));
        }
        else if(pipeline->name.find("transformer_spatial_rms_norm_sum_sq") == 0) {
          const vkcompute::SpatialRMSNormSizing sizing = vkcompute::computeSpatialRMSNormSizing(config.spatialRMSNorm.TILE_SIZE, channels * pipelineXYSize);
          vk_shader::push::TransformerSpatialRMSNormSumSqPushParams params = {batchSize,channels,pipelineXYSize,sizing.tilesPerGroupPass1};
          push(params);
          dispatch(static_cast<uint32_t>(sizing.numCHWWorkgroups), static_cast<uint32_t>(batchSize));
        }
        else if(pipeline->name.find("transformer_spatial_rms_norm_reduce") == 0) {
          const vkcompute::SpatialRMSNormSizing sizing = vkcompute::computeSpatialRMSNormSizing(config.spatialRMSNorm.TILE_SIZE, channels * pipelineXYSize);
          vk_shader::push::TransformerSpatialRMSNormReducePushParams params = {batchSize,sizing.numCHWWorkgroups,sizing.tilesPerGroupPass2};
          push(params);
          dispatch(1, static_cast<uint32_t>(batchSize));
        }
        else if(pipeline->name.find("transformer_spatial_rms_norm_apply") == 0) {
          vk_shader::push::TransformerSpatialRMSNormApplyPushParams params = {batchSize,channels,pipelineXYSize,1e-6f};
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
            vkCmdPushConstants(targetCommandBuffer, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pipeline->pushConstantSize, params.data());
          dispatch(1, 1, 1);
        }
      };

      const auto recordDispatches = [&](VkCommandBuffer targetCommandBuffer, size_t repeat, size_t firstPipeline, size_t pipelineCount) {
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
        const int runChannels =
          plan.kernelName.find("conv") != string::npos
          ? std::max(1, workload == 2 || workload == 5 || workload == 8 ? context.modelInfo.midNumChannels :
                         workload == 3 || workload == 6 || workload == 9 ? maxConvChannels :
                         context.modelInfo.trunkNumChannels)
          : std::max(1, context.modelInfo.trunkNumChannels);
        const size_t lastPipeline = std::min(pipelines.size(), firstPipeline + pipelineCount);
        for(size_t i = firstPipeline; i < lastPipeline; i++) {
          const Pipeline* pipeline = pipelines[i];
          recordPipeline(
            targetCommandBuffer, pipeline, descriptorSets[i], runBatchSize, runChannels,
            runGemm.gemmM, runGemm.gemmN, runGemm.gemmK
          );
          if(i + 1 < lastPipeline)
            vk_helper::barrierCommandBuffer(targetCommandBuffer);
        }
      };

      const bool resetsInPlaceAccumulator =
        pointwiseAccumulatorInitialBuffer != nullptr &&
        (pipelines[0]->name.find("add_pointwise") == 0 ||
         pipelines[0]->name.find("add_channel_bias_nchw") == 0);
      const auto resetInPlaceAccumulator = [&]() {
        if(!resetsInPlaceAccumulator)
          return;
        vk_helper::barrierCommandBufferForBuffer(
          commandBuffer, tuningBuffers[0],
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT
        );
        VkBufferCopy copyRegion = {};
        copyRegion.size = pointwiseAccumulatorBytes;
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

      const bool spatialRMSNorm = plan.kernelName == "spatialRMSNorm";
      result = vk_helper::beginCommandBuffer(commandBuffer);
      if(result != VK_SUCCESS) {
        error = "could not begin tuning command buffer: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }
      const auto zeroBuffer = [&](VulkanBuffer* buffer) {
        vkCmdFillBuffer(commandBuffer, buffer->buffer, 0, VK_WHOLE_SIZE, 0);
        vk_helper::barrierCommandBufferForBuffer(
          commandBuffer, buffer,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT
        );
      };
      for(VulkanBuffer* buffer: tuningBuffers)
        zeroBuffer(buffer);
      if(pointwiseAccumulatorInitialBuffer != nullptr)
        zeroBuffer(pointwiseAccumulatorInitialBuffer);
      for(VulkanBuffer* buffer: pointwiseValidationBuffers)
        zeroBuffer(buffer);
      for(VulkanBuffer* buffer: winogradInputValidationBuffers)
        zeroBuffer(buffer);
      for(VulkanBuffer* buffer: winogradOutputValidationBuffers)
        zeroBuffer(buffer);
      for(VulkanBuffer* buffer: gpoolValidationBuffers)
        zeroBuffer(buffer);
      for(VulkanBuffer* buffer: attentionValidationBuffers)
        zeroBuffer(buffer);
      for(VulkanBuffer* buffer: gemmValidationBuffers)
        zeroBuffer(buffer);
      for(VulkanBuffer* buffer: spatialValidationBuffers)
        zeroBuffer(buffer);
      for(const PendingUpload& upload: pendingUploads) {
        VkBufferCopy copyRegion = {};
        copyRegion.size = upload.size;
        vkCmdCopyBuffer(commandBuffer, upload.stagingBuffer->buffer, upload.destinationBuffer->buffer, 1, &copyRegion);
        vk_helper::barrierCommandBufferForBuffer(
          commandBuffer, upload.destinationBuffer,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT
        );
      }
      if(plan.kernelName == "transformerAttention") {
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
      // Zero-weight runs are timestamped too, matching OpenCL event handling.
      for(size_t repeat = 0; repeat < plan.warmupRuns; repeat++) {
        resetInPlaceAccumulator();
        if(spatialRMSNorm) {
          recordDispatches(commandBuffer, repeat, 0, 2);
          vk_helper::barrierCommandBuffer(commandBuffer);
        }
        recordDispatches(commandBuffer, repeat, firstTimedPipeline, timedPipelineCount);
      }
      const size_t timedRuns = plan.timedRuns();
      vkCmdResetQueryPool(
        commandBuffer, queryPool, 0, static_cast<uint32_t>(2 * timedRuns * timedPipelineCount)
      );
      const size_t spatialApplyFirstBuffer = spatialRMSNorm ?
        pipelines[0]->bindingCount + pipelines[1]->bindingCount : 0;
      for(size_t timedRepeat = 0; timedRepeat < timedRuns; timedRepeat++) {
        const uint32_t queryStart = static_cast<uint32_t>(2 * timedRepeat * timedPipelineCount);
        resetInPlaceAccumulator();
        vk_helper::barrierCommandBuffer(commandBuffer);
        if(spatialRMSNorm) {
          recordDispatches(commandBuffer, plan.warmupRuns + timedRepeat, 0, 2);
          vk_helper::barrierCommandBuffer(commandBuffer);
        }
        for(size_t timedPipeline = 0; timedPipeline < timedPipelineCount; timedPipeline++) {
          const uint32_t pipelineQueryStart = queryStart + static_cast<uint32_t>(2 * timedPipeline);
          vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, pipelineQueryStart);
          recordDispatches(
            commandBuffer, plan.warmupRuns + timedRepeat, firstTimedPipeline + timedPipeline, 1
          );
          vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, pipelineQueryStart + 1);
          if(timedPipeline + 1 < timedPipelineCount)
            vk_helper::barrierCommandBuffer(commandBuffer);
        }
        if(isGemm && timedRepeat < plan.gemmCases.size() && timedRepeat < gemmValidationBuffers.size()) {
          const Pipeline* gemmPipeline = pipelines[0];
          const uint32_t binding = outputBinding(gemmPipeline);
          const GemmDimensions dimensions = getGemmDimensions(
            plan.gemmCases[timedRepeat].inChannels, plan.gemmCases[timedRepeat].outChannels
          );
          VulkanBuffer* outputBuffer = tuningBuffers[binding];
          const VkDeviceSize outputBytes = static_cast<VkDeviceSize>(gemmBatch) * dimensions.gemmM * dimensions.gemmN *
            (halfBinding(gemmPipeline, binding) ? sizeof(half_t) : sizeof(float));
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          VkBufferCopy copyRegion = {};
          copyRegion.size = outputBytes;
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
          const size_t count = batchSize * std::max(1, context.modelInfo.gpoolNumChannels) * 3;
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
        else if(plan.kernelName.find("InputTransform") != string::npos &&
                timedRepeat < winogradInputValidationBuffers.size()) {
          const Pipeline* inputPipeline = pipelines[0];
          const uint32_t binding = outputBinding(inputPipeline);
          VulkanBuffer* outputBuffer = tuningBuffers[binding];
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          VkBufferCopy copyRegion = {};
          copyRegion.size = scratchBytes;
          vkCmdCopyBuffer(
            commandBuffer, outputBuffer->buffer, winogradInputValidationBuffers[timedRepeat]->buffer, 1, &copyRegion
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, winogradInputValidationBuffers[timedRepeat],
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT
          );
        }
        else if(plan.kernelName.find("OutputTransform") != string::npos &&
                timedRepeat < winogradOutputValidationBuffers.size()) {
          const uint32_t binding = outputBinding(pipelines[0]);
          VulkanBuffer* outputBuffer = tuningBuffers[binding];
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          VkBufferCopy copyRegion = {};
          copyRegion.size = scratchBytes;
          vkCmdCopyBuffer(
            commandBuffer, outputBuffer->buffer, winogradOutputValidationBuffers[timedRepeat]->buffer, 1, &copyRegion
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, winogradOutputValidationBuffers[timedRepeat],
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
          );
          vk_helper::barrierCommandBufferForBuffer(
            commandBuffer, outputBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT
          );
        }
        else if(plan.kernelName == "pointwise" && timedRepeat < 10) {
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
            copyRegion.size = scratchBytes;
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
        else if((plan.kernelName == "addChannelBiases" || plan.kernelName == "transformerRMSNorm") &&
                timedRepeat < pointwiseValidationBuffers.size()) {
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
        else if(plan.kernelName == "transformerAttention" && timedRepeat < attentionValidationBuffers.size()) {
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
      result = vkResetFences(device->device, 1, &fence);
      if(result != VK_SUCCESS) {
        error = "could not reset tuning fence: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }
      result = vk_helper::submitCommandBuffers(device, {commandBuffer}, fence);
      if(result != VK_SUCCESS) {
        error = "could not submit tuning command buffer: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }
      commandBufferSubmitted = true;
      result = vkWaitForFences(device->device, 1, &fence, VK_TRUE, UINT64_MAX);
      if(result != VK_SUCCESS) {
        error = "could not wait for tuning fence: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }
      commandBufferSubmitted = false;
      if(timedRuns == 0) {
        error = "tuning measurement plan has no timed runs";
        cleanup();
        return false;
      }
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
        const double weight = plan.weightForRun(plan.warmupRuns + timedRepeat);
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
          const size_t count = static_cast<size_t>(gemmBatch) * dimensions.gemmM * dimensions.gemmN;
          vector<float> output(count);
          if(useFP16) {
            vector<half_t> halves(count);
            vk_helper::copyDeviceBufferToHost(
              device, gemmValidationBuffers[caseIndex], count * sizeof(half_t), halves.data(), true, &result
            );
            for(size_t j = 0; j < count; j++)
              output[j] = half_float::half_cast<float>(halves[j]);
          }
          else {
            vk_helper::copyDeviceBufferToHost(
              device, gemmValidationBuffers[caseIndex], count * sizeof(float), output.data(), true, &result
            );
          }
          if(result != VK_SUCCESS) {
            error = "could not read GEMM validation output: " + vk_helper::vkErrorToString(result);
            return false;
          }
          for(int n = 0; n < gemmBatch; n++)
            for(int y = 0; y < dimensions.logicalN; y++)
              for(int x = 0; x < dimensions.logicalM; x++)
                readback.push_back(output[(static_cast<size_t>(n) * dimensions.gemmN + y) * dimensions.gemmM + x]);
        }
        cleanup();
        return true;
      }
      if(plan.kernelName == "gPool" && !gpoolValidationBuffers.empty()) {
        const Pipeline* gpoolPipeline = pipelines[0];
        const uint32_t binding = outputBinding(gpoolPipeline);
        const size_t count = batchSize * std::max(1, context.modelInfo.gpoolNumChannels) * 3;
        const bool useFP16 = halfBinding(gpoolPipeline, binding);
        for(VulkanBuffer* validationBuffer: gpoolValidationBuffers) {
          vector<float> output(count);
          if(useFP16) {
            vector<half_t> halves(count);
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(half_t), halves.data(), true, &result
            );
            for(size_t j = 0; j < count; j++)
              output[j] = half_float::half_cast<float>(halves[j]);
          }
          else {
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(float), output.data(), true, &result
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
      if(plan.kernelName.find("InputTransform") != string::npos && !winogradInputValidationBuffers.empty()) {
        const Pipeline* inputPipeline = pipelines[0];
        const uint32_t binding = outputBinding(inputPipeline);
        const ConvTuneParams& conv = inputPipeline->name.find("5x5") != string::npos ? config.conv5x5 : config.conv3x3;
        const size_t tiles = batchSize * ((context.nnXLen + conv.outTileXSize - 1) / conv.outTileXSize) *
          ((context.nnYLen + conv.outTileYSize - 1) / conv.outTileYSize);
        const size_t count = vk_helper::roundUpToMultiple(tiles, static_cast<size_t>(activeMPaddingMult)) *
          vk_helper::roundUpToMultiple(maxConvChannels, static_cast<size_t>(activeKPaddingMult)) *
          conv.inTileXSize * conv.inTileYSize;
        const bool useFP16 = halfBinding(inputPipeline, binding);
        for(VulkanBuffer* validationBuffer: winogradInputValidationBuffers) {
          vector<float> output(count);
          if(useFP16) {
            vector<half_t> halves(count);
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(half_t), halves.data(), true, &result
            );
            for(size_t j = 0; j < count; j++)
              output[j] = half_float::half_cast<float>(halves[j]);
          }
          else {
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(float), output.data(), true, &result
            );
          }
          if(result != VK_SUCCESS) {
            error = "could not read Winograd input validation output: " + vk_helper::vkErrorToString(result);
            return false;
          }
          readback.insert(readback.end(), output.begin(), output.end());
        }
        cleanup();
        return true;
      }
      if(plan.kernelName.find("OutputTransform") != string::npos && !winogradOutputValidationBuffers.empty()) {
        const Pipeline* outputPipeline = pipelines[0];
        const uint32_t binding = outputBinding(outputPipeline);
        const size_t count = batchSize * maxConvChannels * xySize;
        for(VulkanBuffer* validationBuffer: winogradOutputValidationBuffers) {
          vector<float> output(count);
          if(halfBinding(outputPipeline, binding)) {
            vector<half_t> halves(count);
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(half_t), halves.data(), true, &result
            );
            for(size_t j = 0; j < count; j++)
              output[j] = half_float::half_cast<float>(halves[j]);
          }
          else {
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(float), output.data(), true, &result
            );
          }
          if(result != VK_SUCCESS) {
            error = "could not read Winograd output validation output: " + vk_helper::vkErrorToString(result);
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
        const size_t count = batchSize * std::max(1, context.modelInfo.trunkNumChannels) * xySize;
        for(VulkanBuffer* validationBuffer: spatialValidationBuffers) {
          vector<float> output(count);
          if(halfBinding(applyPipeline, binding)) {
            vector<half_t> halves(count);
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(half_t), halves.data(), true, &result
            );
            for(size_t j = 0; j < count; j++)
              output[j] = half_float::half_cast<float>(halves[j]);
          }
          else {
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(float), output.data(), true, &result
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
      if(plan.kernelName == "pointwise" && !pointwiseValidationBuffers.empty()) {
        for(size_t repeat = 0; repeat < 10; repeat++) {
          for(size_t pipelineIndex = 0; pipelineIndex < pipelines.size(); pipelineIndex++) {
            const Pipeline* pipeline = pipelines[pipelineIndex];
            const uint32_t binding = outputBinding(pipeline);
            size_t count = batchSize * std::max(1, context.modelInfo.trunkNumChannels) * xySize;
            if(pipeline->name.find("transformer_swiglu") == 0)
              count = batchSize * std::max(context.modelInfo.trunkNumChannels, context.modelInfo.transformerFFNChannels) * xySize;
            const size_t validationIndex = repeat * pipelines.size() + pipelineIndex;
            vector<float> output(count);
            if(halfBinding(pipeline, binding)) {
              vector<half_t> halves(count);
              vk_helper::copyDeviceBufferToHost(
                device, pointwiseValidationBuffers[validationIndex], count * sizeof(half_t), halves.data(), true, &result
              );
              for(size_t j = 0; j < count; j++)
                output[j] = half_float::half_cast<float>(halves[j]);
            }
            else {
              vk_helper::copyDeviceBufferToHost(
                device, pointwiseValidationBuffers[validationIndex], count * sizeof(float), output.data(), true, &result
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
        size_t count = batchSize * std::max(1, context.modelInfo.trunkNumChannels) * xySize;
        if(pipelines[0]->name.find("add_channel_bias_nchw") == 0)
          count = batchSize * std::max(1, context.modelInfo.trunkNumChannels) * addChannelBiasesXYSize;
        else if(pipelines[0]->name.find("transformer_swiglu") == 0)
          count = batchSize * std::max(context.modelInfo.trunkNumChannels, context.modelInfo.transformerFFNChannels) * xySize;
        for(VulkanBuffer* validationBuffer: pointwiseValidationBuffers) {
          vector<float> output(count);
          if(halfBinding(pipelines[0], binding)) {
            vector<half_t> halves(count);
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(half_t), halves.data(), true, &result
            );
            for(size_t j = 0; j < count; j++)
              output[j] = half_float::half_cast<float>(halves[j]);
          }
          else {
            vk_helper::copyDeviceBufferToHost(
              device, validationBuffer, count * sizeof(float), output.data(), true, &result
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
      if(plan.kernelName == "transformerAttention" && !attentionValidationBuffers.empty()) {
        const Pipeline* attentionPipeline = pipelines[0];
        const uint32_t binding = outputBinding(attentionPipeline);
        const bool useFP16 = halfBinding(attentionPipeline, binding);
        const VkDeviceSize outputBytes = attentionOutputElements * (useFP16 ? sizeof(half_t) : sizeof(float));
        if(!readAttentionValidationBuffers(
          attentionValidationBuffers, outputBytes, attentionOutputElements, useFP16, readback, result, error
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
    struct ReusableResources {
      explicit ReusableResources(const VulkanDevice* device): device(device) {}

      ~ReusableResources() {
        if(commandBuffer != VK_NULL_HANDLE)
          vkFreeCommandBuffers(device->device, device->commandPool, 1, &commandBuffer);
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
        for(VulkanBuffer* buffer: winogradOutputValidationBuffers)
          vk_helper::releaseVulkanBuffer(device, buffer);
        for(VulkanBuffer* buffer: winogradInputValidationBuffers)
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
      vector<VulkanBuffer*> tuningBuffers;
      vector<VulkanBuffer*> uploadStagingBuffers;
      VulkanBuffer* pointwiseAccumulatorInitialBuffer = nullptr;
      VulkanBuffer* attentionReadbackBuffer = nullptr;
      vector<VulkanBuffer*> pointwiseValidationBuffers;
      vector<VulkanBuffer*> winogradInputValidationBuffers;
      vector<VulkanBuffer*> winogradOutputValidationBuffers;
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
      size_t outputElements,
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
      vkCmdFillBuffer(activeResources->commandBuffer, activeResources->attentionReadbackBuffer->buffer, 0, VK_WHOLE_SIZE, 0);
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
      result = vk_helper::submitCommandBuffers(device, {activeResources->commandBuffer}, activeResources->fence);
      if(result != VK_SUCCESS) {
        error = "could not submit attention readback command buffer: " + vk_helper::vkErrorToString(result);
        return false;
      }
      result = vkWaitForFences(device->device, 1, &activeResources->fence, VK_TRUE, UINT64_MAX);
      if(result != VK_SUCCESS) {
        error = "could not wait for attention readback: " + vk_helper::vkErrorToString(result);
        return false;
      }

      void* mappedData = nullptr;
      result = vmaMapMemory(device->allocator, activeResources->attentionReadbackBuffer->allocation, &mappedData);
      if(result != VK_SUCCESS) {
        error = "could not map attention readback buffer: " + vk_helper::vkErrorToString(result);
        return false;
      }
      readback.clear();
      readback.reserve(outputElements * validationBuffers.size());
      const unsigned char* raw = static_cast<const unsigned char*>(mappedData);
      for(size_t i = 0; i < validationBuffers.size(); i++) {
        const void* source = raw + stride * i;
        if(useFP16) {
          const half_t* halves = static_cast<const half_t*>(source);
          for(size_t j = 0; j < outputElements; j++)
            readback.push_back(half_float::half_cast<float>(halves[j]));
        }
        else {
          const float* floats = static_cast<const float*>(source);
          readback.insert(readback.end(), floats, floats + outputElements);
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
        activeResources->commandBuffer = vk_helper::allocateCommandBuffer(device, &result);
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
        logger->write("Dummy tuning thread starting");

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
        const int batchSize = 1;
        const int mSize = 97;
        const int kSize = 151;
        const size_t outputElements = static_cast<size_t>(mSize) * kSize;

        VkResult result = VK_SUCCESS;
        VkCommandPoolCreateInfo commandPoolInfo = {};
        commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolInfo.queueFamilyIndex = 0;
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
        AddPointWiseTuneParams pointwiseParams;
        VulkanParams vulkanParams;
        pipelines = make_unique<vk_shader::ComputePipelines>(device->device, nullptr);
        result = pipelines->createXgemmStridedBatched(
          pipelines->xgemmStridedBatchedFp32, xgemmParams, vulkanParams
        );
        if(result != VK_SUCCESS)
          throw StringError("could not create dummy XGEMM pipeline: " + vk_helper::vkErrorToString(result));
        result = pipelines->createAddPointWise(
          pipelines->addPointWise, pointwiseParams, vulkanParams
        );
        if(result != VK_SUCCESS)
          throw StringError("could not create dummy addPointWise pipeline: " + vk_helper::vkErrorToString(result));

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
          device, makeRandomVector(static_cast<size_t>(kSize) * kSize, 1.2 / kSize), false, &result
        );
        if(result != VK_SUCCESS || matrixA == nullptr)
          throw StringError("could not create dummy matrix A: " + vk_helper::vkErrorToString(result));
        matrixB = vk_helper::createReadOnlyBuffer(
          device, makeRandomVector(static_cast<size_t>(kSize) * kSize, 1.2 / kSize), false, &result
        );
        if(result != VK_SUCCESS || matrixB == nullptr)
          throw StringError("could not create dummy matrix B: " + vk_helper::vkErrorToString(result));
        matrixC = vk_helper::createReadOnlyBuffer(
          device, makeRandomVector(outputElements, 1.0), false, &result
        );
        if(result != VK_SUCCESS || matrixC == nullptr)
          throw StringError("could not create dummy matrix C: " + vk_helper::vkErrorToString(result));
        matrixD = vk_helper::createReadOnlyBuffer(
          device, makeRandomVector(outputElements, 1.0), false, &result
        );
        if(result != VK_SUCCESS || matrixD == nullptr)
          throw StringError("could not create dummy matrix D: " + vk_helper::vkErrorToString(result));
        buffer = vk_helper::createReadWriteBuffer(device, makeZeroVector(outputElements), false, &result);
        if(result != VK_SUCCESS || buffer == nullptr)
          throw StringError("could not create dummy buffer: " + vk_helper::vkErrorToString(result));
        buffer2 = vk_helper::createReadWriteBuffer(device, makeZeroVector(outputElements), false, &result);
        if(result != VK_SUCCESS || buffer2 == nullptr)
          throw StringError("could not create second dummy buffer: " + vk_helper::vkErrorToString(result));
        readbackBuffer = vk_helper::createReadbackBuffer(
          device, outputElements * sizeof(float), &result
        );
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
          throw StringError("could not allocate dummy XGEMM descriptor set: " + vk_helper::vkErrorToString(result));
        descriptorSetInfo.pSetLayouts = &pipelines->addPointWise.descriptorSetLayout;
        result = vkAllocateDescriptorSets(device->device, &descriptorSetInfo, &pointwiseDescriptorSet);
        if(result != VK_SUCCESS)
          throw StringError("could not allocate dummy pointwise descriptor set: " + vk_helper::vkErrorToString(result));

        const auto updateXgemmDescriptors = [&](VulkanBuffer* input, VulkanBuffer* other, VulkanBuffer* output) {
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
        const auto addReadWriteBarrier = [&](VulkanBuffer* target, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
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
        while(!shouldStop.load()) {
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
              static_cast<uint32_t>(mSize), static_cast<uint32_t>(kSize), static_cast<uint32_t>(kSize),
              static_cast<uint32_t>(mSize), 0,
              static_cast<uint32_t>(kSize), 0,
              static_cast<uint32_t>(mSize), 0, 0
            };
            vkCmdPushConstants(
              commandBuffer, pipelines->xgemmStridedBatchedFp32.layout,
              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params
            );
            vkCmdDispatch(
              commandBuffer,
              (mSize + xgemmParams.WGD - 1) / xgemmParams.WGD,
              (kSize + xgemmParams.WGD - 1) / xgemmParams.WGD,
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
            addReadWriteBarrier(buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
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
            VkBufferCopy copyRegion = {};
            copyRegion.size = outputElements * sizeof(float);
            vkCmdCopyBuffer(commandBuffer, buffer->buffer, readbackBuffer->buffer, 1, &copyRegion);
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
          VkSubmitInfo submitInfo = {};
          submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
          submitInfo.commandBufferCount = 1;
          submitInfo.pCommandBuffers = &commandBuffer;
          result = vkQueueSubmit(device->dummyQueue, 1, &submitInfo, fence);
          if(result != VK_SUCCESS) {
            reportFailure("could not submit dummy command buffer: " + vk_helper::vkErrorToString(result));
            break;
          }
          result = vkWaitForFences(device->device, 1, &fence, VK_TRUE, UINT64_MAX);
          if(result != VK_SUCCESS) {
            reportFailure("could not wait for dummy command buffer: " + vk_helper::vkErrorToString(result));
            break;
          }

          if(which <= 3)
            swap(buffer, buffer2);
          else if(which > 5) {
            result = vmaInvalidateAllocation(device->allocator, readbackBuffer->allocation, 0, VK_WHOLE_SIZE);
            if(result != VK_SUCCESS) {
              reportFailure("could not invalidate dummy readback buffer: " + vk_helper::vkErrorToString(result));
              break;
            }
            void* mappedData = nullptr;
            result = vmaMapMemory(device->allocator, readbackBuffer->allocation, &mappedData);
            if(result != VK_SUCCESS) {
              reportFailure("could not map dummy readback buffer: " + vk_helper::vkErrorToString(result));
              break;
            }
            memcpy(output.data(), mappedData, output.size() * sizeof(float));
            vmaUnmapMemory(device->allocator, readbackBuffer->allocation);
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

  template<typename Tuner>
  double testAllConfigs(const TuningContext& context, VulkanTuneParams& currentConfig) {
    vector<VulkanTuneParams> configs;
    configs = Tuner::candidates(currentConfig, context.full, context);
    VulkanTuneParams defaults;
    configs.insert(configs.begin(), Tuner::reference(currentConfig, defaults));
    dedupCandidates(configs);
    const size_t firstShuffledIndex = KeepsCurrentConfigFirst<Tuner>::value ? 2 : 1;
    if(configs.size() > 2) {
      Rand rand("VulkanTuner:" + Tuner::name());
      for(size_t i = configs.size() - 1; i > firstShuffledIndex; i--) {
        const size_t j = firstShuffledIndex + static_cast<size_t>(rand.nextUInt64(i - firstShuffledIndex + 1));
        swap(configs[i], configs[j]);
      }
    }
    const size_t candidateCount = count_if(configs.begin(), configs.end(), Tuner::isValid) +
      (!configs.empty() && !Tuner::isValid(configs.front()) ? 1 : 0);
    const TuningMeasurementPlan plan = makeMeasurementPlan(Tuner::name(), context);

    if(context.timer == nullptr || !context.timer->isUsable()) {
      if(context.logger != nullptr)
        context.logger->write("Skipping Vulkan tuner " + Tuner::name() + ": compute timestamps are unavailable");
      return 0.0;
    }
    VulkanTimestampTimer& timer = *context.timer;

    vk_shader::ComputePipelines pipelines(context.device->device, nullptr);
    vector<Pipeline*> previousTargets;
    bool found = false;
    double bestScore = 0.0;
    double bestCallsPerSecond = 0.0;
    vector<float> referenceReadback;
    size_t lastBestCandidateIndex = 0;
    size_t candidateIndex = 0;
    auto logProgressIfNeeded = [&](size_t currentCandidateIndex, const vector<const Pipeline*>& targets) {
      if(currentCandidateIndex % 20 == 0 && currentCandidateIndex >= lastBestCandidateIndex + 10)
        logTuningProgress(context, currentCandidateIndex, candidateCount, targets, Tuner::name());
    };
    for(size_t configIndex = 0; configIndex < configs.size(); configIndex++) {
      const VulkanTuneParams& candidate = configs[configIndex];
      const bool isReferenceCandidate = configIndex == 0;
      if(!isReferenceCandidate && !Tuner::isValid(candidate))
        continue;
      const size_t currentCandidateIndex = candidateIndex++;
      try {
        // The previous candidate has finished before the next pipeline is built.
        // Reuse the pipeline cache and avoid a device-idle wait for every candidate.
        for(Pipeline* pipeline: previousTargets)
          pipelines.destroyPipeline(*pipeline);
        previousTargets.clear();
        vector<const Pipeline*> targets;
        VkResult result = Tuner::create(context, candidate, pipelines, targets);
        if(result != VK_SUCCESS) {
          if(isReferenceCandidate) {
            logTuningFailure(
              context, currentCandidateIndex, candidateCount,
              targets, Tuner::name(),
              "pipeline creation failed: " + vk_helper::vkErrorToString(result)
            );
          }
          logProgressIfNeeded(currentCandidateIndex, targets);
          for(const Pipeline* pipeline: targets)
            pipelines.destroyPipeline(*const_cast<Pipeline*>(pipeline));
          if(StopsOnReferenceImplFail<Tuner>::value(candidate) && isReferenceCandidate)
            return 0.0;
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
        if(!measured) {
          if(isReferenceCandidate) {
            logTuningFailure(
              context, currentCandidateIndex, candidateCount,
              targets, Tuner::name(),
              error.empty() ? "measurement failed" : error
            );
          }
          logProgressIfNeeded(currentCandidateIndex, targets);
          if(StopsOnReferenceImplFail<Tuner>::value(candidate) && isReferenceCandidate)
            return 0.0;
          continue;
        }
        if(!isfinite(callsPerSecond) || callsPerSecond <= 0.0) {
          if(isReferenceCandidate) {
            logTuningFailure(
              context, currentCandidateIndex, candidateCount,
              targets, Tuner::name(),
              "measurement returned invalid calls/sec"
            );
          }
          logProgressIfNeeded(currentCandidateIndex, targets);
          if(StopsOnReferenceImplFail<Tuner>::value(candidate) && isReferenceCandidate)
            return 0.0;
          continue;
        }
        if(referenceReadback.empty() && !cpuReference.empty())
          referenceReadback = std::move(cpuReference);
        if(referenceReadback.empty()) {
          referenceReadback = readback;
          validateReadback(referenceReadback, readback, plan, errorProp);
        }
        else
          validateReadback(referenceReadback, readback, plan, errorProp);
        const double score = VulkanTuner::computeTuningScore(callsPerSecond, errorProp, plan.errorTolerance);
        const bool isBest = score > bestScore;
        if(!context.printOnlyOnImprovement || isBest) {
          logTuningResult(
            context, currentCandidateIndex, candidateCount, targets, candidate, Tuner::name(), callsPerSecond, errorProp,
            isBest
          );
        }
        if(isBest) {
          bestScore = score;
          bestCallsPerSecond = callsPerSecond;
          currentConfig = candidate;
          found = true;
          lastBestCandidateIndex = currentCandidateIndex;
        }
        logProgressIfNeeded(currentCandidateIndex, targets);
      }
      catch(const StringError& e) {
        // A failed pipeline specialization is an invalid candidate, not a fatal tuning failure.
        if(isReferenceCandidate)
          logTuningFailure(context, currentCandidateIndex, candidateCount, vector<const Pipeline*>(), Tuner::name(), e.what());
        logProgressIfNeeded(currentCandidateIndex, vector<const Pipeline*>());
        if(StopsOnReferenceImplFail<Tuner>::value(candidate) && isReferenceCandidate)
          return 0.0;
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
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{8,16,32,64} : vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.WGD = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.MDIMCD = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.NDIMCD = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.MDIMAD = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.NDIMBD = v; });
      addCandidates(configs, full ? vector<int>{2,8,16} : vector<int>{2,8}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.KWID = v; });
      addCandidates(configs, full ? vector<int>{1,2,4} : vector<int>{2,4}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.VWMD = v; });
      addCandidates(configs, full ? vector<int>{1,2,4} : vector<int>{2,4}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.VWND = v; });
      addCandidates(configs, vector<int>{1}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.PADA = v; });
      addCandidates(configs, vector<int>{1}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.PADB = v; });
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }),
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
      // Match OpenCL's untuned HGemmWmmaParams defaults. MWARP/NWARP/KDIM
      // and subgroupSize remain the hardware-selected Vulkan values.
      result.hgemmCooperativeMatrix.MWG = 16;
      result.hgemmCooperativeMatrix.NWG = 16;
      result.hgemmCooperativeMatrix.KWG = 16;
      result.hgemmCooperativeMatrix.MWAVE = 16;
      result.hgemmCooperativeMatrix.NWAVE = 16;
      result.hgemmCooperativeMatrix.SA = 0;
      result.hgemmCooperativeMatrix.SB = 0;
      result.hgemmCooperativeMatrix.VWM = defaults.hgemmCooperativeMatrix.VWM;
      result.hgemmCooperativeMatrix.VWN = defaults.hgemmCooperativeMatrix.VWN;
      return result;
    }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext& context) {
      vector<VulkanTuneParams> configs;
      for(int accType: {16, 32}) {
        VulkanTuneParams accConfig = current;
        if(selectHgemmCooperativeMatrixProperties(
             context.device, accConfig.hgemmCooperativeMatrix, accType
           ))
          configs.push_back(accConfig);
      }
      if(configs.empty())
        return configs;
      addCandidates(configs, full ? vector<int>{16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.MWG = v; });
      addCandidates(configs, full ? vector<int>{16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.NWG = v; });
      addCandidates(configs, vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.KWG = v; });
      addCandidates(configs, vector<int>{8,16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.MWAVE = v; });
      addCandidates(configs, vector<int>{8,16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.NWAVE = v; });
      addCandidates(configs, full ? vector<int>{1,2,4} : vector<int>{2,4}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.VWM = v; });
      addCandidates(configs, full ? vector<int>{1,2,4} : vector<int>{2,4}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.VWN = v; });
      addCandidates(configs, vector<int>{0,1}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.SA = v; });
      addCandidates(configs, vector<int>{0,1}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.SB = v; });
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) {
          return !p.hgemmCooperativeMatrix.isValid();
        }),
        configs.end()
      );
      if(!full) {
        configs.erase(
          remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) {
            return !p.hgemmCooperativeMatrix.isSimple();
          }),
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

  template<>
  struct KeepsCurrentConfigFirst<HgemmCooperativeMatrixTunerImpl> {
    static constexpr bool value = true;
  };

  template<>
  struct StopsOnReferenceImplFail<HgemmCooperativeMatrixTunerImpl> {
    static bool value(const VulkanTuneParams&) { return true; }
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
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext& context) {
      vector<VulkanTuneParams> configs;
      for(int accType: {16, 32}) {
        VulkanTuneParams accConfig = current;
        if(selectHgemmCooperativeMatrixProperties(
             context.device, accConfig.hgemmCooperativeMatrixNCHW, accType
           ))
          configs.push_back(accConfig);
      }
      if(configs.empty())
        return configs;
      addCandidates(configs, full ? vector<int>{16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.MWG = v; });
      addCandidates(configs, full ? vector<int>{16,32} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.NWG = v; });
      addCandidates(configs, vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.KWG = v; });
      addCandidates(configs, vector<int>{8,16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.MWAVE = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.NWAVE = v; });
      addCandidates(configs, full ? vector<int>{1,2,4} : vector<int>{2,4}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.VWM = v; });
      addCandidates(configs, full ? vector<int>{1,2,4} : vector<int>{2,4}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.VWN = v; });
      addCandidates(configs, vector<int>{0,1}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.SB = v; });
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) {
          return !p.hgemmCooperativeMatrixNCHW.isValid();
        }),
        configs.end()
      );
      if(!full) {
        configs.erase(
          remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) { return !p.hgemmCooperativeMatrixNCHW.isSimple(); }),
          configs.end()
        );
      }
      // Keep the incoming configuration at the same position as OpenCL's
      // explicit current-config insertion after candidate generation.
      configs.insert(configs.begin(), current);
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createHgemmCooperativeMatrixNCHW(pipelines.hgemmCooperativeMatrixNCHW, config.hgemmCooperativeMatrixNCHW);
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
  struct StopsOnReferenceImplFail<HgemmCooperativeMatrixNCHWTunerImpl> {
    static bool value(const VulkanTuneParams&) { return true; }
  };

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
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.xgemm.MWG = v; });
      addCandidates(configs, full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.xgemm.NWG = v; });
      addCandidates(configs, full ? vector<int>{8,16,32} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.KWG = v; });
      addCandidates(configs, full ? vector<int>{1,2,8} : vector<int>{1,2}, [](VulkanTuneParams& p, int v) { p.xgemm.KWI = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.MDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.NDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.MDIMA = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.NDIMB = v; });
      // Vulkan provides width-specific targets for 1, 2, and 4; OpenCL's 8 has no target.
      addCandidates(configs, full ? vector<int>{1,2,4} : vector<int>{2,4}, [](VulkanTuneParams& p, int v) { p.xgemm.VWM = v; });
      addCandidates(configs, full ? vector<int>{1,2,4} : vector<int>{2,4}, [](VulkanTuneParams& p, int v) { p.xgemm.VWN = v; });
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) { return !p.xgemm.isValid(); }),
        configs.end()
      );
      if(!full) {
        configs.erase(
          remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) { return !p.xgemm.isSimple(); }),
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
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createXgemmBatched(pipelines.xgemmBatchedFp32, config.xgemm, config.xgemm16, config.vulkan);
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
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.xgemm16.MWG = v; });
      addCandidates(configs, full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.xgemm16.NWG = v; });
      addCandidates(configs, full ? vector<int>{8,16,32} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.KWG = v; });
      addCandidates(configs, full ? vector<int>{1,2,8} : vector<int>{1,2}, [](VulkanTuneParams& p, int v) { p.xgemm16.KWI = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.MDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.NDIMC = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.MDIMA = v; });
      addCandidates(configs, vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.xgemm16.NDIMB = v; });
      addCandidates(configs, full ? vector<int>{1,2,4} : vector<int>{2,4}, [](VulkanTuneParams& p, int v) { p.xgemm16.VWM = v; });
      addCandidates(configs, full ? vector<int>{1,2,4} : vector<int>{2,4}, [](VulkanTuneParams& p, int v) { p.xgemm16.VWN = v; });
      configs.erase(
        remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) { return !p.xgemm16.isValid(); }),
        configs.end()
      );
      if(!full) {
        configs.erase(
          remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& p) { return !p.xgemm16.isSimple(); }),
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
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createXgemmBatched(pipelines.xgemmBatchedFp32, config.xgemm, config.xgemm16, config.vulkan);
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.xgemmBatchedFp32);
      return result;
    }
  };

  template<>
  struct StopsOnReferenceImplFail<Xgemm16Tuner> {
    static bool value(const VulkanTuneParams&) { return true; }
  };

  template<int ConvSize, bool InputTransform>
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
      configs.erase(remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }), configs.end());
      configs.insert(configs.begin(), current);
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

  struct Conv3x3InputTuner : ConvTuner<3,true> {};
  struct Conv3x3OutputTuner : ConvTuner<3,false> {};
  struct Conv5x5InputTuner : ConvTuner<5,true> {};
  struct Conv5x5OutputTuner : ConvTuner<5,false> {};

  template<>
  struct KeepsCurrentConfigFirst<Conv3x3InputTuner> {
    static constexpr bool value = true;
  };

  template<>
  struct KeepsCurrentConfigFirst<Conv3x3OutputTuner> {
    static constexpr bool value = true;
  };

  struct GPoolTuner {
    static string name() { return "gPool"; }
    static bool isValid(const VulkanTuneParams& config) { return config.gPool.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.gPool = defaults.gPool; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext& context) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32,64} : vector<int>{1,2,4,8,16,32}, [](VulkanTuneParams& p, int v) { p.gPool.XYSTRIDE = v; });
      addCandidates(configs, powersOfTwoUpTo(std::min(full ? 64 : 32, std::max(1, context.modelInfo.gpoolNumChannels))), [](VulkanTuneParams& p, int v) { p.gPool.CHANNELSTRIDE = v; });
      addCandidates(configs, powersOfTwoUpTo(std::min(4, std::max(1, context.batchSize))), [](VulkanTuneParams& p, int v) { p.gPool.BATCHSTRIDE = v; });
      configs.erase(remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }), configs.end());
      configs.insert(configs.begin(), current);
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createGlobalPoolingChannelsFp32(pipelines.globalPoolingChannelsFp32, config.gPool, config.vulkan);
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
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.pointwise = defaults.pointwise; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8,16}, [](VulkanTuneParams& p, int v) { p.pointwise.ELTS_PER_THREAD = v; });
      addCandidates(configs, full ? vector<int>{32,64,128,256,512} : vector<int>{32,64,128,256}, [](VulkanTuneParams& p, int v) { p.pointwise.LOCAL_SIZE = v; });
      configs.erase(remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }), configs.end());
      configs.insert(configs.begin(), current);
      return configs;
    }
    static VkResult create(const TuningContext& context, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createAddPointWise(pipelines.addPointWise, config.pointwise, config.vulkan);
      if(result != VK_SUCCESS) return result;
      targets.push_back(&pipelines.addPointWise);
      if(context.modelInfo.transformerFFNChannels > 0) {
        result = pipelines.createTransformerSwiGLU(pipelines.transformerSwiGLU, config.pointwise, config.vulkan);
        if(result == VK_SUCCESS) targets.push_back(&pipelines.transformerSwiGLU);
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
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.addChannelBiases = defaults.addChannelBiases; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, vector<int>{1,2,4}, [](VulkanTuneParams& p, int v) { p.addChannelBiases.XY_ELTS_PER_THREAD = v; });
      addCandidates(configs, vector<int>{1,2,4,8}, [](VulkanTuneParams& p, int v) { p.addChannelBiases.NC_ELTS_PER_THREAD = v; });
      configs.erase(remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }), configs.end());
      configs.insert(configs.begin(), current);
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createAddChannelBiasNCHW(pipelines.addChannelBiasNCHW, config.addChannelBiases, config.vulkan);
      if(result == VK_SUCCESS) targets.push_back(&pipelines.addChannelBiasNCHW);
      return result;
    }
  };

  template<>
  struct KeepsCurrentConfigFirst<AddChannelBiasesTuner> {
    static constexpr bool value = true;
  };

  struct TransformerTuner {
    static string name() { return "transformerAttention"; }
    static bool isValid(const VulkanTuneParams& config) { return config.transformer.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.transformer = defaults.transformer; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      VulkanTuneParams defaults;
      VulkanTuneParams naive = current;
      naive.transformer.USE_TILED_ATTN = 0;
      naive.transformer.ATTN_BLOCK_Q = defaults.transformer.ATTN_BLOCK_Q;
      naive.transformer.ATTN_BLOCK_KV = defaults.transformer.ATTN_BLOCK_KV;
      naive.transformer.Q_PER_THREAD = defaults.transformer.Q_PER_THREAD;
      VulkanTuneParams tiled = current;
      tiled.transformer.USE_TILED_ATTN = 1;
      vector<VulkanTuneParams> configs = {tiled};
      addCandidates(configs, full ? vector<int>{8,16,32,64,128,256} : vector<int>{16,32,64,128,256}, [](VulkanTuneParams& p, int v) { p.transformer.ATTN_BLOCK_Q = v; });
      addCandidates(configs, full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64,128}, [](VulkanTuneParams& p, int v) { p.transformer.ATTN_BLOCK_KV = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8} : vector<int>{1,2,4}, [](VulkanTuneParams& p, int v) { p.transformer.Q_PER_THREAD = v; });
      configs.erase(remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }), configs.end());
      configs.insert(configs.begin(), naive);
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
      configs.erase(remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }), configs.end());
      configs.insert(configs.begin(), current);
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

  template<>
  struct KeepsCurrentConfigFirst<TransformerRMSNormTuner> {
    static constexpr bool value = true;
  };

  struct SpatialRMSNormTuner {
    static string name() { return "spatialRMSNorm"; }
    static bool isValid(const VulkanTuneParams& config) { return config.spatialRMSNorm.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.spatialRMSNorm = defaults.spatialRMSNorm; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full, const TuningContext&) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{32,64,128,256,512,1024} : vector<int>{32,64,128,256,512}, [](VulkanTuneParams& p, int v) { p.spatialRMSNorm.TILE_SIZE = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8,16}, [](VulkanTuneParams& p, int v) { p.spatialRMSNorm.APPLY_ELTS_PER_THREAD = v; });
      configs.erase(remove_if(configs.begin(), configs.end(), [](const VulkanTuneParams& config) { return !isValid(config); }), configs.end());
      configs.insert(configs.begin(), current);
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createTransformerSpatialRMSNormSumSq(pipelines.transformerSpatialRMSNormSumSq, config.spatialRMSNorm, config.vulkan);
      if(result != VK_SUCCESS) return result;
      targets.push_back(&pipelines.transformerSpatialRMSNormSumSq);
      result = pipelines.createTransformerSpatialRMSNormReduce(pipelines.transformerSpatialRMSNormReduce, config.spatialRMSNorm, config.vulkan);
      if(result != VK_SUCCESS) return result;
      targets.push_back(&pipelines.transformerSpatialRMSNormReduce);
      result = pipelines.createTransformerSpatialRMSNormApply(pipelines.transformerSpatialRMSNormApply, config.spatialRMSNorm, config.vulkan);
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
    runTuner<Conv3x3InputTuner>(context, config);
    runTuner<Conv3x3OutputTuner>(context, config);
    config.conv5x5.inputTransformLocalXSize = config.conv3x3.inputTransformLocalXSize;
    config.conv5x5.inputTransformLocalYSize = config.conv3x3.inputTransformLocalYSize;
    config.conv5x5.outputTransformLocalXSize = config.conv3x3.outputTransformLocalXSize;
    config.conv5x5.outputTransformLocalYSize = config.conv3x3.outputTransformLocalYSize;
    config.conv5x5.outputTransformLocalZSize = config.conv3x3.outputTransformLocalZSize;
    runTuner<GPoolTuner>(context, config);
    const bool hasTransformerModel =
      context.modelInfo.transformerHeadDim > 0 && context.modelInfo.transformerVHeadDim > 0 &&
      context.modelInfo.transformerNumHeads > 0 && context.modelInfo.transformerNumKVHeads > 0;
    if(hasTransformerModel) {
      runTuner<TransformerTuner>(context, config);
      runTuner<TransformerRMSNormTuner>(context, config);
    }
    runTuner<PointwiseTuner>(context, config);
    runTuner<AddChannelBiasesTuner>(context, config);
    if(hasTransformerModel)
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
        " calls/s, required_ratio=" + Global::strprintf("%.2f", VulkanTuner::FP16_COMPUTE_MIN_THROUGHPUT_RATIO) +
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
    double xgemmBaselineCallsPerSecond,
    bool canUseHgemmCooperativeMatrix,
    bool canUseNCHW
  ) {
    if(!canUseHgemmCooperativeMatrix || !config.vulkan.canUseFP16Storage ||
       !config.vulkan.canUseFP16Compute) {
      if(context.logger != nullptr)
        context.logger->write(
          "Skipping Vulkan cooperative matrix tuning: capability or FP16 prerequisite unavailable, "
          "shouldUseCooperativeMatrix=false, shouldUseHgemmCooperativeMatrixNCHW=false"
        );
      return;
    }

    VulkanTuneParams cooperativeConfig = config;
    cooperativeConfig.vulkan.shouldUseFP16Storage = true;
    cooperativeConfig.vulkan.shouldUseFP16Compute = true;
    const double hgemmCallsPerSecond = runTuner<HgemmCooperativeMatrixTunerImpl>(context, cooperativeConfig);
    const bool useHgemm = VulkanTuner::isFastEnough(
      hgemmCallsPerSecond, xgemmBaselineCallsPerSecond, VulkanTuner::COOPERATIVE_MATRIX_MIN_THROUGHPUT_RATIO
    );
    if(useHgemm) {
      config.hgemmCooperativeMatrix = cooperativeConfig.hgemmCooperativeMatrix;
      config.vulkan.shouldUseCooperativeMatrix = true;
      config.vulkan.shouldUseFP16Compute = true;
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
    cooperativeConfig.vulkan.shouldUseFP16Compute = true;
    const double hgemmNCHWCallsPerSecond = canUseNCHW
      ? runTuner<HgemmCooperativeMatrixNCHWTunerImpl>(context, cooperativeConfig)
      : 0.0;
    const bool useHgemmNCHW = config.vulkan.shouldUseCooperativeMatrix && VulkanTuner::isFastEnough(
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
  VulkanTuneParams& tunedConfig,
  bool printOnlyOnImprovement
) {
  const auto hostStart = std::chrono::steady_clock::now();
  if(device == nullptr)
    throw StringError("VulkanTuner::tune: device is null");
  if(!tunedConfig.isValid())
    tunedConfig = VulkanTuneParams();
  const VulkanParams hardwareParams = VulkanTuner::getHardwareParams(device->info);
  tunedConfig.vulkan.canUseFP16Storage = hardwareParams.canUseFP16Storage;
  tunedConfig.vulkan.canUseFP16Compute = hardwareParams.canUseFP16Compute;
  tunedConfig.vulkan.canUseCooperativeMatrix = hardwareParams.canUseCooperativeMatrix;
  tunedConfig.vulkan.canUseSubgroup = hardwareParams.canUseSubgroup;
  VulkanTimestampTimer timer(device);
  VulkanDummyThread dummyThread(device, logger);
  dummyThread.start();
  TuningContext context{device, batchSize, nnXLen, nnYLen, modelInfo, full, logger, &timer, printOnlyOnImprovement};
  if(logger != nullptr) {
    logger->write(
      "Vulkan tuning capabilities: fp16Storage=" + string(tunedConfig.vulkan.canUseFP16Storage ? "true" : "false") +
      ", fp16Compute=" + string(tunedConfig.vulkan.canUseFP16Compute ? "true" : "false") +
      ", subgroup=" + string(tunedConfig.vulkan.canUseSubgroup ? "true" : "false") +
      ", cooperativeMatrix=" + string(tunedConfig.vulkan.canUseCooperativeMatrix ? "true" : "false")
    );
  }
  const bool canUseHgemmCooperativeMatrix =
    tunedConfig.vulkan.canUseCooperativeMatrix &&
    HgemmCooperativeMatrixTuner::selectCooperativeMatrixProperties(device, tunedConfig.hgemmCooperativeMatrix);
  const bool canUseHgemmCooperativeMatrixNCHW =
    canUseHgemmCooperativeMatrix &&
    HgemmCooperativeMatrixNCHWTuner::selectCooperativeMatrixProperties(
      device, tunedConfig.hgemmCooperativeMatrixNCHW
    );
  tunedConfig.xgemmDirect.PADA = 1;
  tunedConfig.xgemmDirect.PADB = 1;
  tunedConfig.vulkan.shouldUseFP16Storage = false;
  tunedConfig.vulkan.shouldUseFP16Compute = false;
  tunedConfig.vulkan.shouldUseCooperativeMatrix = false;
  tunedConfig.vulkan.shouldUseHgemmCooperativeMatrixNCHW = false;
  tunedConfig.vulkan.shouldUseSubgroup = false;
  const double xgemmDirectCallsPerSecond = runTuner<XgemmDirectTuner>(context, tunedConfig);
  const double xgemmCallsPerSecond = runTuner<XgemmTuner>(context, tunedConfig);
  tunedConfig.xgemm16 = tunedConfig.xgemm;
  tuneCooperativeMatrices(
    context, tunedConfig, xgemmDirectCallsPerSecond, xgemmCallsPerSecond,
    canUseHgemmCooperativeMatrix, canUseHgemmCooperativeMatrixNCHW
  );
  tuneXgemm16(context, tunedConfig, xgemmCallsPerSecond);
  if(!tunedConfig.vulkan.shouldUseFP16Compute)
    tuneXgemmStorage(context, tunedConfig, xgemmCallsPerSecond);
  runNonGemmTuners(context, tunedConfig);
  if(logger != nullptr) {
    logger->write(
      "Vulkan tuning final selection: "
      "shouldUseFP16Storage=" + string(tunedConfig.vulkan.shouldUseFP16Storage ? "1" : "0") +
      ", shouldUseFP16Compute=" + string(tunedConfig.vulkan.shouldUseFP16Compute ? "1" : "0") +
      ", shouldUseCooperativeMatrix=" + string(tunedConfig.vulkan.shouldUseCooperativeMatrix ? "1" : "0") +
      ", shouldUseHgemmCooperativeMatrixNCHW=" +
        string(tunedConfig.vulkan.shouldUseHgemmCooperativeMatrixNCHW ? "1" : "0")
    );
  }
  dummyThread.stopAndJoin();
  if(logger != nullptr) {
    const double hostSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - hostStart).count();
    logger->write("Vulkan tuning total host time: " + Global::doubleToString(hostSeconds) + " sec");
  }
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
    const VulkanParams available = VulkanTuner::getHardwareParams(deviceInfo);
    if(loaded.vulkan.canUseFP16Storage != available.canUseFP16Storage ||
       loaded.vulkan.canUseFP16Compute != available.canUseFP16Compute ||
       loaded.vulkan.canUseCooperativeMatrix != available.canUseCooperativeMatrix ||
       loaded.vulkan.canUseSubgroup != available.canUseSubgroup)
      throw IOError("Vulkan tuning capabilities changed for " + filename);
    if(logger != nullptr)
      logger->write("Loaded Vulkan tuning parameters from: " + filename);
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
  bool* didAutoTune
) {
  if(didAutoTune != nullptr)
    *didAutoTune = false;
  string filename = tunerFile;
  if(filename.empty())
    filename = defaultDirectory(true, homeDataDirOverride) + "/" + defaultFileName(gpuName, nnXLen, nnYLen, modelInfo);

  try {
    VulkanTuneParams loaded = VulkanTuneParams::load(filename);
    if(device != nullptr) {
      const VulkanParams available = VulkanTuner::getHardwareParams(device->info);
      if(loaded.vulkan.canUseFP16Storage != available.canUseFP16Storage ||
         loaded.vulkan.canUseFP16Compute != available.canUseFP16Compute ||
         loaded.vulkan.canUseCooperativeMatrix != available.canUseCooperativeMatrix ||
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
  params.vulkan = VulkanTuner::getHardwareParams(device->info);
  tune(device, DEFAULT_BATCH_SIZE, nnXLen, nnYLen, modelInfo, false, logger, params);
  VulkanTuneParams::save(filename, params);
  if(didAutoTune != nullptr)
    *didAutoTune = true;
  if(logger != nullptr)
    logger->write("Completed Vulkan tuning and saved results to: " + filename);
  return params;
}

#endif
