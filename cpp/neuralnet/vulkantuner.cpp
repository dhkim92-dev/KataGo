#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"
#include "../neuralnet/vulkancompute.h"

#include <fstream>
#include <map>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../dataio/homedata.h"

using namespace std;
using namespace vk_shader;
using namespace vk_shader::tune;

namespace {
  const string VERSION_LINE = Global::strprintf("VERSION=%d", VulkanTuner::TUNER_VERSION);

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
         xgemm.isValid() && xgemmDirect.isValid() &&
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
         xgemm.NDIMB == other.xgemm.NDIMB && xgemmDirect.WGD == other.xgemmDirect.WGD &&
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
  writeParam(out, "addChannelBiases.XY_ELTS_PER_THREAD", config.addChannelBiases.XY_ELTS_PER_THREAD);
  writeParam(out, "addChannelBiases.NC_ELTS_PER_THREAD", config.addChannelBiases.NC_ELTS_PER_THREAD);
  writeParam(out, "pointwise.LOCAL_SIZE", config.pointwise.LOCAL_SIZE);
  writeParam(out, "pointwise.ELTS_PER_THREAD", config.pointwise.ELTS_PER_THREAD);
  writeParam(out, "gPool.XYSTRIDE", config.gPool.XYSTRIDE);
  writeParam(out, "gPool.CHANNELSTRIDE", config.gPool.CHANNELSTRIDE);
  writeParam(out, "gPool.BATCHSTRIDE", config.gPool.BATCHSTRIDE);
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
  writeParam(out, "hgemmCooperativeMatrix.MWARP", config.hgemmCooperativeMatrix.MWARP);
  writeParam(out, "hgemmCooperativeMatrix.NWARP", config.hgemmCooperativeMatrix.NWARP);
  writeParam(out, "hgemmCooperativeMatrix.KDIM", config.hgemmCooperativeMatrix.KDIM);
  writeParam(out, "hgemmCooperativeMatrix.subgroupSize", config.hgemmCooperativeMatrix.subgroupSize);
  writeParam(out, "hgemmCooperativeMatrix.MWG", config.hgemmCooperativeMatrix.MWG);
  writeParam(out, "hgemmCooperativeMatrix.NWG", config.hgemmCooperativeMatrix.NWG);
  writeParam(out, "hgemmCooperativeMatrix.KWG", config.hgemmCooperativeMatrix.KWG);
  writeParam(out, "hgemmCooperativeMatrix.MWAVE", config.hgemmCooperativeMatrix.MWAVE);
  writeParam(out, "hgemmCooperativeMatrix.NWAVE", config.hgemmCooperativeMatrix.NWAVE);
  writeParam(out, "hgemmCooperativeMatrix.SA", config.hgemmCooperativeMatrix.SA);
  writeParam(out, "hgemmCooperativeMatrix.SB", config.hgemmCooperativeMatrix.SB);
  writeParam(out, "hgemmCooperativeMatrixNCHW.MWARP", config.hgemmCooperativeMatrixNCHW.MWARP);
  writeParam(out, "hgemmCooperativeMatrixNCHW.NWARP", config.hgemmCooperativeMatrixNCHW.NWARP);
  writeParam(out, "hgemmCooperativeMatrixNCHW.KDIM", config.hgemmCooperativeMatrixNCHW.KDIM);
  writeParam(out, "hgemmCooperativeMatrixNCHW.subgroupSize", config.hgemmCooperativeMatrixNCHW.subgroupSize);
  writeParam(out, "hgemmCooperativeMatrixNCHW.MWG", config.hgemmCooperativeMatrixNCHW.MWG);
  writeParam(out, "hgemmCooperativeMatrixNCHW.NWG", config.hgemmCooperativeMatrixNCHW.NWG);
  writeParam(out, "hgemmCooperativeMatrixNCHW.KWG", config.hgemmCooperativeMatrixNCHW.KWG);
  writeParam(out, "hgemmCooperativeMatrixNCHW.MWAVE", config.hgemmCooperativeMatrixNCHW.MWAVE);
  writeParam(out, "hgemmCooperativeMatrixNCHW.NWAVE", config.hgemmCooperativeMatrixNCHW.NWAVE);
  writeParam(out, "hgemmCooperativeMatrixNCHW.CType", config.hgemmCooperativeMatrixNCHW.CType);
  writeParam(out, "hgemmCooperativeMatrixNCHW.ResultType", config.hgemmCooperativeMatrixNCHW.ResultType);
  writeParam(out, "hgemmCooperativeMatrixNCHW.SB", config.hgemmCooperativeMatrixNCHW.SB);
  writeParam(out, "hgemmCooperativeMatrixNCHW.VWM", config.hgemmCooperativeMatrixNCHW.VWM);
  writeParam(out, "hgemmCooperativeMatrixNCHW.VWN", config.hgemmCooperativeMatrixNCHW.VWN);
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
  writeParam(out, "transformer.ATTN_BLOCK_Q", config.transformer.ATTN_BLOCK_Q);
  writeParam(out, "transformer.ATTN_BLOCK_KV", config.transformer.ATTN_BLOCK_KV);
  writeParam(out, "transformer.Q_PER_THREAD", config.transformer.Q_PER_THREAD);
  writeParam(out, "transformer.USE_TILED_ATTN", config.transformer.USE_TILED_ATTN);
  writeParam(out, "rmsNorm.WG_C_SIZE", config.rmsNorm.WG_C_SIZE);
  writeParam(out, "rmsNorm.WG_XY_SIZE", config.rmsNorm.WG_XY_SIZE);
  writeParam(out, "rmsNorm.C_PER_THREAD", config.rmsNorm.C_PER_THREAD);
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
  if(values.size() != 82 && values.size() != 83)
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

  void logTuningPipelines(
    const TuningContext& context,
    size_t candidateIndex,
    size_t totalCandidates,
    const vector<const Pipeline*>& pipelines
  ) {
    for(const Pipeline* pipeline: pipelines) {
      const string message = "(" + to_string(candidateIndex) + " / " + to_string(totalCandidates) + ") " + pipeline->name + "...";
      if(context.logger != nullptr)
        context.logger->write(message);
      if(context.logger == nullptr || (!context.logger->isLoggingToStdout() && !context.logger->isLoggingToStderr()))
        cerr << message << endl;
    }
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
      double& callsPerSecond,
      string& error
    ) const {
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

      VkResult result = VK_SUCCESS;
      const size_t batchSize = static_cast<size_t>(std::max(1, context.batchSize));
      const size_t xySize = static_cast<size_t>(std::max(1, context.nnXLen * context.nnYLen));
      const size_t maxChannels = static_cast<size_t>(std::max({
        1,
        context.modelInfo.trunkNumChannels,
        context.modelInfo.maxConvChannels1x1,
        context.modelInfo.maxConvChannels3x3,
        context.modelInfo.gpoolNumChannels,
        context.modelInfo.transformerFFNChannels
      }));
      const size_t maxTilesX = (static_cast<size_t>(std::max(1, context.nnXLen)) + 1) / 2;
      const size_t maxTilesY = (static_cast<size_t>(std::max(1, context.nnYLen)) + 1) / 2;
      const size_t paddedTiles = vk_helper::roundUpToMultiple(batchSize * maxTilesX * maxTilesY, static_cast<size_t>(config.xgemm.MWG));
      const size_t paddedChannels = vk_helper::roundUpToMultiple(maxChannels, static_cast<size_t>(std::max(config.xgemm.KWG, config.xgemm.NWG)));
      const size_t hgemmHWSize = vk_helper::roundUpToMultiple(
        xySize, static_cast<size_t>(std::max(16, config.hgemmCooperativeMatrixNCHW.MWARP))
      );
      const size_t hgemmChannelAlignment = static_cast<size_t>(std::max({
        32, config.hgemmCooperativeMatrixNCHW.KWG, config.hgemmCooperativeMatrixNCHW.KDIM, config.hgemmCooperativeMatrixNCHW.NWG
      }));
      const size_t hgemmCSize = vk_helper::roundUpToMultiple(
        maxChannels, hgemmChannelAlignment
      );
      const size_t hgemmOCSize = vk_helper::roundUpToMultiple(
        maxChannels, hgemmChannelAlignment
      );
      const size_t cooperativeTiles = vk_helper::roundUpToMultiple(
        batchSize * maxTilesX * maxTilesY,
        static_cast<size_t>(std::max(1, config.hgemmCooperativeMatrix.MWG))
      );
      const size_t cooperativeChannels = vk_helper::roundUpToMultiple(
        maxChannels,
        static_cast<size_t>(std::max({
          1, config.hgemmCooperativeMatrix.KWG, config.hgemmCooperativeMatrix.NWG
        }))
      );
      const size_t scratchElements = std::max({
        batchSize * maxChannels * xySize,
        paddedTiles * paddedChannels * 36,
        batchSize * hgemmCSize * hgemmHWSize,
        batchSize * hgemmOCSize * hgemmHWSize,
        cooperativeTiles * cooperativeChannels * 36
      });
      const size_t scratchBytes = std::max(static_cast<size_t>(4 * 1024 * 1024), scratchElements * sizeof(float));
      VulkanBuffer* scratch = vk_helper::createDeviceBuffer(device, scratchBytes, false, &result);
      if(result != VK_SUCCESS || scratch == nullptr) {
        error = "could not allocate tuning buffer: " + vk_helper::vkErrorToString(result);
        return false;
      }

      VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
      VkQueryPool queryPool = VK_NULL_HANDLE;
      VkFence fence = VK_NULL_HANDLE;
      VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
      const auto cleanup = [&]() {
        if(commandBuffer != VK_NULL_HANDLE)
          vkFreeCommandBuffers(device->device, device->commandPool, 1, &commandBuffer);
        if(fence != VK_NULL_HANDLE)
          vkDestroyFence(device->device, fence, nullptr);
        if(queryPool != VK_NULL_HANDLE)
          vkDestroyQueryPool(device->device, queryPool, nullptr);
        if(descriptorPool != VK_NULL_HANDLE)
          vkDestroyDescriptorPool(device->device, descriptorPool, nullptr);
        vk_helper::releaseVulkanBuffer(device, scratch);
      };

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
        for(uint32_t binding = 0; binding < pipeline->bindingCount; binding++)
          writes.push_back(vk_helper::writeDescriptorSetBuffer(descriptorSet, binding, scratch));
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
      queryPoolInfo.queryCount = 2;
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

      const auto recordPipeline = [&](const Pipeline* pipeline, VkDescriptorSet descriptorSet) {
        const int batchSize = std::max(1, context.batchSize);
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
          const int cSize = static_cast<int>(hgemmCSize);
          const int hwSize = static_cast<int>(hgemmHWSize);
          const int ocSize = static_cast<int>(hgemmOCSize);
          vk_shader::push::HGemmCooperativeMatrixNCHWParams params = {cSize, hwSize, ocSize};
          push(params);
          dispatch(
            static_cast<uint32_t>((hwSize + config.hgemmCooperativeMatrixNCHW.MWG - 1) / config.hgemmCooperativeMatrixNCHW.MWG),
            static_cast<uint32_t>((ocSize + config.hgemmCooperativeMatrixNCHW.NWG - 1) / config.hgemmCooperativeMatrixNCHW.NWG),
            static_cast<uint32_t>(batchSize)
          );
        }
        else if(pipeline->name.find("hgemm_cooperative_matrix_") == 0) {
          const int m = static_cast<int>(cooperativeTiles);
          const int n = static_cast<int>(cooperativeChannels);
          const int k = static_cast<int>(cooperativeChannels);
          const int tileBatch = 36;
          vk_shader::push::HGemmCooperativeMatrixParams params = {m, n, k};
          push(params);
          dispatch(
            static_cast<uint32_t>(m / config.hgemmCooperativeMatrix.MWG),
            static_cast<uint32_t>(n / config.hgemmCooperativeMatrix.NWG),
            static_cast<uint32_t>(tileBatch)
          );
        }
        else if(pipeline->name.find("xgemm_batched") == 0) {
          const uint32_t m = config.xgemm.MWG * 2;
          const uint32_t n = config.xgemm.NWG * 2;
          const uint32_t k = config.xgemm.KWG;
          vk_shader::push::XGEMMBatchedParams params = {m,n,k,m,k,n,k,m,n};
          push(params);
          dispatch(m / config.xgemm.MWG, n / config.xgemm.NWG, 4);
        }
        else if(pipeline->name.find("xgemm_direct_batched_tt") == 0) {
          const uint32_t size = config.xgemmDirect.WGD * 2;
          vk_shader::push::XgemmDirectBatchedTTParams params = {size,size,config.xgemmDirect.WGD,config.xgemmDirect.WGD,config.xgemmDirect.WGD,size,0,0,1};
          push(params);
          dispatch(2, 2, static_cast<uint32_t>(batchSize));
        }
        else if(pipeline->name.find("xgemm_strided_batched") == 0) {
          const uint32_t size = config.xgemmDirect.WGD * 2;
          vk_shader::push::XgemmStridedBatchedFp32Params params = {
            size,size,config.xgemmDirect.WGD,size,size * config.xgemmDirect.WGD,size,0,size,size * size,0
          };
          push(params);
          dispatch(2, 2, static_cast<uint32_t>(batchSize));
        }
        else if(pipeline->name.find("winograd_input_transform") == 0) {
          const vk_shader::tune::ConvTuneParams& convParams =
            pipeline->name.find("5x5") != string::npos ? config.conv5x5 : config.conv3x3;
          const int outTile = convParams.outTileXSize;
          const int tilesX = (context.nnXLen + outTile - 1) / outTile;
          const int tilesY = (context.nnYLen + outTile - 1) / outTile;
          const int paddedTiles = vk_helper::roundUpToMultipleInt(batchSize * tilesX * tilesY, config.xgemm.MWG);
          const int paddedChannels = vk_helper::roundUpToMultipleInt(channels, config.xgemm.KWG);
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
          const int paddedTiles = vk_helper::roundUpToMultipleInt(batchSize * tilesX * tilesY, config.xgemm.MWG);
          const int paddedChannels = vk_helper::roundUpToMultipleInt(channels, config.xgemm.NWG);
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

      const auto recordDispatches = [&]() {
        for(size_t i = 0; i < pipelines.size(); i++) {
          const Pipeline* pipeline = pipelines[i];
          recordPipeline(pipeline, descriptorSets[i]);
          if(i + 1 < pipelines.size())
            vk_helper::barrierCommandBuffer(commandBuffer);
        }
      };

      // The warm-up dispatch is deliberately outside the timestamp interval.
      recordDispatches();
      vkCmdResetQueryPool(commandBuffer, queryPool, 0, 2);
      vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 0);
      for(int repeat = 0; repeat < 8; repeat++)
        recordDispatches();
      vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 1);
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
      uint64_t timestamps[2] = {};
      result = vkGetQueryPoolResults(
        device->device, queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT
      );
      if(result != VK_SUCCESS || timestamps[1] <= timestamps[0]) {
        error = "could not read tuning timestamps: " + vk_helper::vkErrorToString(result);
        cleanup();
        return false;
      }
      const double elapsedSeconds = (timestamps[1] - timestamps[0]) * timestampPeriod * 1e-9;
      // Eight recorded repetitions are divided into individual pipeline calls.
      callsPerSecond = static_cast<double>(8 * pipelines.size()) / elapsedSeconds;
      cleanup();
      return true;
    }

   private:
    const VulkanDevice* device;
    float timestampPeriod;
  };

  template<typename Tuner>
  double runTuner(const TuningContext& context, VulkanTuneParams& currentConfig) {
    vector<VulkanTuneParams> configs = Tuner::candidates(currentConfig, context.full);
    VulkanTuneParams defaults;
    configs.insert(configs.begin(), Tuner::reference(currentConfig, defaults));
    dedupCandidates(configs);
    const size_t validCandidateCount = count_if(configs.begin(), configs.end(), Tuner::isValid);

    VulkanTimestampTimer timer(context.device);
    if(!timer.isUsable()) {
      if(context.logger != nullptr)
        context.logger->write("Skipping Vulkan tuner " + Tuner::name() + ": compute timestamps are unavailable");
      return 0.0;
    }

    bool found = false;
    double bestCallsPerSecond = 0.0;
    size_t candidateIndex = 0;
    for(const VulkanTuneParams& candidate: configs) {
      if(!Tuner::isValid(candidate))
        continue;
      candidateIndex += 1;
      try {
        vk_shader::ComputePipelines pipelines(context.device->device, nullptr);
        vector<const Pipeline*> targets;
        VkResult result = Tuner::create(context, candidate, pipelines, targets);
        if(result != VK_SUCCESS)
          continue;
        logTuningPipelines(context, candidateIndex, validCandidateCount, targets);
        double callsPerSecond = 0.0;
        string error;
        if(!timer.measure(targets, candidate, context, callsPerSecond, error) || !isfinite(callsPerSecond) || callsPerSecond <= 0.0)
          continue;
        if(callsPerSecond > bestCallsPerSecond) {
          bestCallsPerSecond = callsPerSecond;
          currentConfig = candidate;
          found = true;
        }
      }
      catch(const StringError&) {
        // A failed pipeline specialization is an invalid candidate, not a fatal tuning failure.
      }
    }
    if(context.logger != nullptr) {
      context.logger->write(
        "Vulkan tuner " + Tuner::name() + (found ? " selected a measured candidate" : " retained the previous candidate")
      );
    }
    return found ? bestCallsPerSecond : 0.0;
  }

  struct XgemmDirectTuner {
    static string name() { return "xgemmDirect"; }
    static bool isValid(const VulkanTuneParams& config) { return config.xgemmDirect.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      result.xgemmDirect = defaults.xgemmDirect;
      return result;
    }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{8,16,32,64} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.WGD = v; });
      addCandidates(configs, full ? vector<int>{4,8,16,32} : vector<int>{8,16}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.MDIMCD = v; });
      addCandidates(configs, full ? vector<int>{4,8,16,32} : vector<int>{8,16}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.NDIMCD = v; });
      addCandidates(configs, full ? vector<int>{4,8,16,32} : vector<int>{8,16}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.MDIMAD = v; });
      addCandidates(configs, full ? vector<int>{4,8,16,32} : vector<int>{8,16}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.NDIMBD = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8,16} : vector<int>{1,2}, [](VulkanTuneParams& p, int v) { p.xgemmDirect.KWID = v; });
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createXgemmDirectBatchedTT(pipelines.xgemmDirectBatchedTT, config.xgemmDirect, config.vulkan);
      if(result != VK_SUCCESS) return result;
      result = pipelines.createXgemmStridedBatched(pipelines.xgemmStridedBatchedFp32, config.xgemmDirect, config.vulkan);
      if(result == VK_SUCCESS) {
        targets.push_back(&pipelines.xgemmDirectBatchedTT);
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
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.MWG = v; });
      addCandidates(configs, full ? vector<int>{16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.NWG = v; });
      addCandidates(configs, full ? vector<int>{16,32,64} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.KWG = v; });
      addCandidates(configs, full ? vector<int>{8,16,32,64} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.MWAVE = v; });
      addCandidates(configs, full ? vector<int>{8,16,32,64} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrix.NWAVE = v; });
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
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.MWG = v; });
      addCandidates(configs, full ? vector<int>{16,32} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.NWG = v; });
      addCandidates(configs, full ? vector<int>{16,32,64} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.KWG = v; });
      addCandidates(configs, full ? vector<int>{8,16,32,64} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.MWAVE = v; });
      addCandidates(configs, full ? vector<int>{8,16,32} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.hgemmCooperativeMatrixNCHW.NWAVE = v; });
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
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.xgemm.MWG = v; });
      addCandidates(configs, full ? vector<int>{16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.xgemm.NWG = v; });
      addCandidates(configs, full ? vector<int>{8,16,32,64} : vector<int>{16,32}, [](VulkanTuneParams& p, int v) { p.xgemm.KWG = v; });
      addCandidates(configs, full ? vector<int>{4,8,16,32} : vector<int>{4,8}, [](VulkanTuneParams& p, int v) { p.xgemm.MDIMC = v; });
      addCandidates(configs, full ? vector<int>{4,8,16,32} : vector<int>{4,8}, [](VulkanTuneParams& p, int v) { p.xgemm.NDIMC = v; });
      addCandidates(configs, full ? vector<int>{4,8,16,32} : vector<int>{4,8}, [](VulkanTuneParams& p, int v) { p.xgemm.MDIMA = v; });
      addCandidates(configs, full ? vector<int>{4,8,16,32} : vector<int>{4,8}, [](VulkanTuneParams& p, int v) { p.xgemm.NDIMB = v; });
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      VkResult result = pipelines.createXgemmBatched(pipelines.xgemmBatchedFp32, config.xgemm, config.vulkan);
      if(result == VK_SUCCESS)
        targets.push_back(&pipelines.xgemmBatchedFp32);
      return result;
    }
  };

  template<int ConvSize, uint32_t OutTileSize>
  struct ConvTuner {
    static string name() { return ConvSize == 3 ? "conv3x3" : "conv5x5"; }
    static ConvTuneParams& params(VulkanTuneParams& config) { return ConvSize == 3 ? config.conv3x3 : config.conv5x5; }
    static const ConvTuneParams& params(const VulkanTuneParams& config) { return ConvSize == 3 ? config.conv3x3 : config.conv5x5; }
    static bool isValid(const VulkanTuneParams& config) { return params(config).isValid(OutTileSize); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) {
      VulkanTuneParams result = current;
      params(result) = params(defaults);
      return result;
    }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32} : vector<int>{2,4,8}, [](VulkanTuneParams& p, int v) { params(p).inputTransformLocalXSize = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8,16} : vector<int>{1,2,4}, [](VulkanTuneParams& p, int v) { params(p).inputTransformLocalYSize = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32} : vector<int>{4,8}, [](VulkanTuneParams& p, int v) { params(p).outputTransformLocalXSize = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8,16} : vector<int>{1,2}, [](VulkanTuneParams& p, int v) { params(p).outputTransformLocalYSize = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8,16} : vector<int>{1,2}, [](VulkanTuneParams& p, int v) { params(p).outputTransformLocalZSize = v; });
      return configs;
    }
    static VkResult create(const TuningContext&, const VulkanTuneParams& config, vk_shader::ComputePipelines& pipelines, vector<const Pipeline*>& targets) {
      Pipeline& input = ConvSize == 3 ? pipelines.winogradInputTransform3x3 : pipelines.winogradInputTransform5x5;
      Pipeline& output = ConvSize == 3 ? pipelines.winogradOutputTransform3x3 : pipelines.winogradOutputTransform5x5;
      VkResult result = pipelines.createWinogradInputTransform(input, params(config), ConvSize, config.vulkan);
      if(result != VK_SUCCESS) return result;
      input.name += ConvSize == 3 ? "_3x3" : "_5x5";
      result = pipelines.createWinogradOutputTransform(output, params(config), ConvSize, config.vulkan);
      if(result == VK_SUCCESS) {
        output.name += ConvSize == 3 ? "_3x3" : "_5x5";
        targets.push_back(&input);
        targets.push_back(&output);
      }
      return result;
    }
  };

  struct Conv3x3Tuner : ConvTuner<3,4> {};
  struct Conv5x5Tuner : ConvTuner<5,2> {};

  struct GPoolTuner {
    static string name() { return "gPool"; }
    static bool isValid(const VulkanTuneParams& config) { return config.gPool.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.gPool = defaults.gPool; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32,64} : vector<int>{8,16,32}, [](VulkanTuneParams& p, int v) { p.gPool.XYSTRIDE = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4}, [](VulkanTuneParams& p, int v) { p.gPool.CHANNELSTRIDE = v; });
      addCandidates(configs, vector<int>{1,2,4}, [](VulkanTuneParams& p, int v) { p.gPool.BATCHSTRIDE = v; });
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
      auto valueHeadInsert = pipelines.valueHeadPoolingChannels.emplace(valueHeadDim, std::move(valueHeadPipeline));
      targets.push_back(&valueHeadInsert.first->second);

      LocalDim sumChannelsDim = {config.gPool.XYSTRIDE, 1, static_cast<int>(localSizeZ)};
      Pipeline sumChannelsPipeline;
      result = pipelines.createSumChannels(sumChannelsPipeline, config.gPool, localSizeZ, config.vulkan);
      if(result != VK_SUCCESS)
        return result;
      auto sumChannelsInsert = pipelines.sumChannels.emplace(sumChannelsDim, std::move(sumChannelsPipeline));
      targets.push_back(&sumChannelsInsert.first->second);
      return result;
    }
  };

  struct PointwiseTuner {
    static string name() { return "pointwise"; }
    static bool isValid(const VulkanTuneParams& config) { return config.pointwise.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.pointwise = defaults.pointwise; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8}, [](VulkanTuneParams& p, int v) { p.pointwise.ELTS_PER_THREAD = v; });
      addCandidates(configs, full ? vector<int>{32,64,128,256,512} : vector<int>{32,64,128,256}, [](VulkanTuneParams& p, int v) { p.pointwise.LOCAL_SIZE = v; });
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

  struct AddChannelBiasesTuner {
    static string name() { return "addChannelBiases"; }
    static bool isValid(const VulkanTuneParams& config) { return config.addChannelBiases.isValid(); }
    static VulkanTuneParams reference(const VulkanTuneParams& current, const VulkanTuneParams& defaults) { VulkanTuneParams result = current; result.addChannelBiases = defaults.addChannelBiases; return result; }
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool) {
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
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full) {
      vector<VulkanTuneParams> configs = {current};
      VulkanTuneParams naive = current;
      naive.transformer.USE_TILED_ATTN = 0;
      addCandidates(configs, full ? vector<int>{8,16,32,64,128,256} : vector<int>{32,64,128}, [](VulkanTuneParams& p, int v) { p.transformer.ATTN_BLOCK_Q = v; p.transformer.USE_TILED_ATTN = 1; });
      addCandidates(configs, full ? vector<int>{8,16,32,64,128} : vector<int>{16,32,64}, [](VulkanTuneParams& p, int v) { p.transformer.ATTN_BLOCK_KV = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8} : vector<int>{1,2}, [](VulkanTuneParams& p, int v) { p.transformer.Q_PER_THREAD = v; });
      configs.push_back(naive);
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
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full) {
      vector<VulkanTuneParams> configs = {current};
      addCandidates(configs, full ? vector<int>{32,64,128,256,512} : vector<int>{32,64,128,256}, [](VulkanTuneParams& p, int v) { p.rmsNorm.WG_C_SIZE = v; });
      addCandidates(configs, full ? vector<int>{1,2,4,8,16,32} : vector<int>{1,2,4,8}, [](VulkanTuneParams& p, int v) { p.rmsNorm.WG_XY_SIZE = v; });
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
    static vector<VulkanTuneParams> candidates(const VulkanTuneParams& current, bool full) {
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

  void runOperationTuners(const TuningContext& context, VulkanTuneParams& config) {
    config.vulkan.shouldUseCooperativeMatrix = false;
    config.vulkan.shouldUseHgemmCooperativeMatrixNCHW = false;
    const double xgemmDirectBaselineCallsPerSecond = runTuner<XgemmDirectTuner>(context, config);
    const double xgemmBaselineCallsPerSecond = runTuner<XgemmTuner>(context, config);
    if(config.vulkan.canUseCooperativeMatrix &&
       config.vulkan.canUseFP16Storage &&
       config.vulkan.canUseFP16Compute &&
       config.vulkan.shouldUseFP16Storage) {
      const double hgemmCallsPerSecond = runTuner<HgemmCooperativeMatrixTunerImpl>(context, config);
      const bool hgemmIsFastEnough =
        isfinite(xgemmBaselineCallsPerSecond) && isfinite(hgemmCallsPerSecond) &&
        xgemmBaselineCallsPerSecond > 0.0 && hgemmCallsPerSecond > 0.0 &&
        hgemmCallsPerSecond / xgemmBaselineCallsPerSecond >= 0.90;
      config.vulkan.shouldUseCooperativeMatrix = hgemmIsFastEnough;
      if(context.logger != nullptr) {
        context.logger->write(
          "Vulkan hgemmCooperativeMatrix baseline comparison: xgemm=" +
          Global::strprintf("%.6g", xgemmBaselineCallsPerSecond) +
          " calls/s, hgemmCooperativeMatrix=" + Global::strprintf("%.6g", hgemmCallsPerSecond) +
          " calls/s, selected=" + (hgemmIsFastEnough ? "true" : "false")
        );
      }
    }
    if(config.vulkan.canUseCooperativeMatrix &&
       config.vulkan.canUseFP16Storage &&
       config.vulkan.canUseFP16Compute &&
       config.vulkan.shouldUseFP16Storage) {
      const double hgemmCallsPerSecond = runTuner<HgemmCooperativeMatrixNCHWTunerImpl>(context, config);
      const bool hgemmIsFastEnough =
        isfinite(xgemmDirectBaselineCallsPerSecond) && isfinite(hgemmCallsPerSecond) &&
        xgemmDirectBaselineCallsPerSecond > 0.0 && hgemmCallsPerSecond > 0.0 &&
        hgemmCallsPerSecond / xgemmDirectBaselineCallsPerSecond >= 0.90;
      config.vulkan.shouldUseHgemmCooperativeMatrixNCHW = hgemmIsFastEnough;
      if(context.logger != nullptr) {
        context.logger->write(
          "Vulkan hgemmCooperativeMatrixNCHW baseline comparison: xgemmDirect=" +
          Global::strprintf("%.6g", xgemmDirectBaselineCallsPerSecond) +
          " calls/s, hgemmCooperativeMatrixNCHW=" + Global::strprintf("%.6g", hgemmCallsPerSecond) +
          " calls/s, selected=" + (hgemmIsFastEnough ? "true" : "false")
        );
      }
    }
    runTuner<Conv3x3Tuner>(context, config);
    runTuner<Conv5x5Tuner>(context, config);
    runTuner<GPoolTuner>(context, config);
    runTuner<PointwiseTuner>(context, config);
    runTuner<AddChannelBiasesTuner>(context, config);
    if(context.modelInfo.transformerHeadDim > 0) {
      runTuner<TransformerTuner>(context, config);
      runTuner<TransformerRMSNormTuner>(context, config);
    }
    runTuner<SpatialRMSNormTuner>(context, config);
  }

  struct FP16ProfileTuner {
    static VkResult create(
      const TuningContext& context,
      const VulkanTuneParams& config,
      vk_shader::ComputePipelines& pipelines,
      vector<const Pipeline*>& targets
    ) {
      VkResult result = XgemmDirectTuner::create(context, config, pipelines, targets);
      if(result != VK_SUCCESS) return result;
      result = XgemmTuner::create(context, config, pipelines, targets);
      if(result != VK_SUCCESS) return result;
      result = Conv3x3Tuner::create(context, config, pipelines, targets);
      if(result != VK_SUCCESS) return result;
      result = Conv5x5Tuner::create(context, config, pipelines, targets);
      if(result != VK_SUCCESS) return result;
      result = GPoolTuner::create(context, config, pipelines, targets);
      if(result != VK_SUCCESS) return result;
      result = PointwiseTuner::create(context, config, pipelines, targets);
      if(result != VK_SUCCESS) return result;
      result = AddChannelBiasesTuner::create(context, config, pipelines, targets);
      if(result != VK_SUCCESS) return result;
      if(context.modelInfo.transformerHeadDim > 0) {
        result = TransformerTuner::create(context, config, pipelines, targets);
        if(result != VK_SUCCESS) return result;
        result = TransformerRMSNormTuner::create(context, config, pipelines, targets);
        if(result != VK_SUCCESS) return result;
      }
      return SpatialRMSNormTuner::create(context, config, pipelines, targets);
    }

    static bool measure(
      const TuningContext& context,
      const VulkanTuneParams& config,
      size_t candidateIndex,
      size_t totalCandidates,
      double& callsPerSecond,
      string& error
    ) {
      try {
        vk_shader::ComputePipelines pipelines(context.device->device, nullptr);
        vector<const Pipeline*> targets;
        const VkResult result = create(context, config, pipelines, targets);
        if(result != VK_SUCCESS) {
          error = "could not create FP16 profile pipelines: " + vk_helper::vkErrorToString(result);
          return false;
        }
        logTuningPipelines(context, candidateIndex, totalCandidates, targets);
        VulkanTimestampTimer timer(context.device);
        return timer.measure(targets, config, context, callsPerSecond, error) && isfinite(callsPerSecond) && callsPerSecond > 0.0;
      }
      catch(const StringError& e) {
        error = e.what();
        return false;
      }
    }
  };

  bool runFP16ProfileTuner(const TuningContext& context, VulkanTuneParams& config) {
    config.vulkan.shouldUseFP16Storage = false;
    config.vulkan.shouldUseFP16Compute = false;
    config.vulkan.shouldUseCooperativeMatrix = false;

    if(!config.vulkan.canUseFP16Storage || !config.vulkan.canUseFP16Compute) {
      if(context.logger != nullptr)
        context.logger->write("Skipping Vulkan FP16 profile tuning: FP16 storage or compute is unavailable");
      return false;
    }

    VulkanTimestampTimer timer(context.device);
    if(!timer.isUsable()) {
      if(context.logger != nullptr)
        context.logger->write("Skipping Vulkan FP16 profile tuning: compute timestamps are unavailable");
      return false;
    }

    VulkanTuneParams fp32Config = config;
    fp32Config.vulkan.shouldUseFP16Storage = false;
    fp32Config.vulkan.shouldUseFP16Compute = false;
    double fp32CallsPerSecond = 0.0;
    string error;
    if(!FP16ProfileTuner::measure(context, fp32Config, 1, 3, fp32CallsPerSecond, error)) {
      if(context.logger != nullptr)
        context.logger->write("Skipping Vulkan FP16 profile tuning: FP32 profile failed: " + error);
      return false;
    }

    VulkanTuneParams selectedConfig = fp32Config;
    double selectedCallsPerSecond = fp32CallsPerSecond;
    for(int useFP16Compute = 0; useFP16Compute <= 1; useFP16Compute++) {
      VulkanTuneParams candidate = config;
      candidate.vulkan.shouldUseFP16Storage = true;
      candidate.vulkan.shouldUseFP16Compute = useFP16Compute != 0;
      double candidateCallsPerSecond = 0.0;
      if(!FP16ProfileTuner::measure(context, candidate, useFP16Compute + 2, 3, candidateCallsPerSecond, error)) {
        if(context.logger != nullptr)
          context.logger->write(
            "Vulkan FP16 profile candidate was excluded: " +
            string(useFP16Compute != 0 ? "p16s16" : "p32s16") + ": " + error
          );
        continue;
      }
      if(candidateCallsPerSecond >= fp32CallsPerSecond * 1.2 && candidateCallsPerSecond > selectedCallsPerSecond) {
        selectedConfig = candidate;
        selectedCallsPerSecond = candidateCallsPerSecond;
      }
    }

    if(!selectedConfig.vulkan.shouldUseFP16Storage) {
      if(context.logger != nullptr)
        context.logger->write("Vulkan FP16 profiles did not reach the required 1.2x speedup over FP32");
      return false;
    }

    config = selectedConfig;
    if(context.logger != nullptr) {
      context.logger->write(
        "Vulkan FP16 profile selected: " +
        string(config.vulkan.shouldUseFP16Compute ? "p16s16" : "p32s16")
      );
    }
    return true;
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
  runOperationTuners(context, tunedConfig);
  if(runFP16ProfileTuner(context, tunedConfig)) {
    tunedConfig.vulkan.shouldUseCooperativeMatrix = tunedConfig.vulkan.canUseCooperativeMatrix;
    runOperationTuners(context, tunedConfig);
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
