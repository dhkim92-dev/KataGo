/**
 * @author dhkim92-dev
 * Vulkan API: 1.1
 * SPIR-V: 1.3
 * Reduce spatial channels into pooled statistics.
 */
#include "common.glsl"
#include "functions.glsl"

layout(constant_id=3) const int XYSTRIDE = 1;
layout(constant_id=4) const int CHANNELSTRIDE = 1;
layout(constant_id=5) const int LOCALSIZE_TOTAL = 2;
layout(constant_id=6) const int USE_NHWC = 0;

layout(push_constant) uniform GlobalPoolingChannelsParams {
  int nSize;
  int cSize;
  int xySize;
  int maskSpatialStride;
  int channelsPadded;
};

layout(set = 0, binding = 0) readonly buffer g_input_block {
    realstore g_input[];
};
layout(set = 0, binding = 1) buffer g_output_block {
    float g_output[];
};
layout(set = 0, binding = 2) readonly buffer g_mask_block {
    realstore mask[];
};
layout(set = 0, binding = 3) readonly buffer g_maskSum_block {
    float maskSums[];
};

shared float partialSums[LOCALSIZE_TOTAL];
shared float partialMaxes[LOCALSIZE_TOTAL];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void globalPoolingChannelsNCHW()
{
  const int xyBase = LocalId0();
  const int c = GlobalId1();
  const int n = GlobalId2();
  const int localId1 = LocalId1();
  const int localId2 = LocalId2();

  float sum = 0.0f;
  float _max = -1.0f;
  if(n < nSize && c < cSize) {
    //Sum up the elements that this group member is responsible for
    for(int xy = xyBase; xy < xySize; xy += XYSTRIDE) {
      int idx = (n * cSize + c) * xySize + xy;
      float v = LOAD(g_input,idx);
      sum += v;
      // Init to -1.0 above and + mask - 1.0 is because it will effectively make all padded space into -1.0
      // which is lower than the lowest value that any current activation function will produce.
      // so the max over all valid spaces will the same as the mask over all spaces including padding.
      // We're relying on all padded space being equal to 0 because this gpool only ever follows a BN+Activate with a mask.
      int maskIdx = n * maskSpatialStride + xy;
      float maskVal = LOAD(mask,maskIdx);
      _max = fmax(_max,v + (maskVal-1.0f));
    }
  }

  //Write to local memory for performing the reduction
  int localIdx = (localId2 * CHANNELSTRIDE + localId1) * XYSTRIDE + xyBase;
  partialSums[localIdx] = sum;
  partialMaxes[localIdx] = _max;

  //Parallel folding downward
  for(int span = XYSTRIDE / 2; span > 0; span /= 2) {
    barrier();

    if(xyBase < span) {
      partialSums[localIdx] += partialSums[localIdx + span];
      partialMaxes[localIdx] = fmax(partialMaxes[localIdx], partialMaxes[localIdx + span]);
    }
  }
  barrier();

  if(n < nSize && c < cSize && xyBase == 0) {
    float finalSum = partialSums[localIdx];
    float finalMax = partialMaxes[localIdx];

    float div = maskSums[n];
    float sqrtdiv = sqrt(div);
    float finalMean = finalSum/div;

    int outBase = n * cSize * 3 + c;
    g_output[outBase] = finalMean;
    g_output[outBase + cSize] = finalMean * (sqrtdiv - 14.0f) * 0.1f;
    g_output[outBase + cSize*2] = finalMax;
  }
}

void globalPoolingChannelsNHWC()
{
  const int xyBase = LocalId0();
  const int c = GlobalId1();
  const int n = GlobalId2();
  const int localId1 = LocalId1();
  const int localId2 = LocalId2();
  const int channelStride = int(channelsPadded);

  float sum = 0.0f;
  float _max = -1.0f;
  if(n < nSize && c < cSize) {
    // Sum up the elements that this group member is responsible for.
    for(int xy = xyBase; xy < xySize; xy += XYSTRIDE) {
      int idx = (n * xySize + xy) * channelStride + c;
      float v = LOAD(g_input,idx);
      sum += v;
      int maskIdx = n * maskSpatialStride + xy;
      float maskVal = LOAD(mask,maskIdx);
      _max = fmax(_max,v + (maskVal-1.0f));
    }
  }

  int localIdx = (localId2 * CHANNELSTRIDE + localId1) * XYSTRIDE + xyBase;
  partialSums[localIdx] = sum;
  partialMaxes[localIdx] = _max;

  for(int span = XYSTRIDE / 2; span > 0; span /= 2) {
    barrier();

    if(xyBase < span) {
      partialSums[localIdx] += partialSums[localIdx + span];
      partialMaxes[localIdx] = fmax(partialMaxes[localIdx], partialMaxes[localIdx + span]);
    }
  }
  barrier();

  if(n < nSize && c < cSize && xyBase == 0) {
    float finalSum = partialSums[localIdx];
    float finalMax = partialMaxes[localIdx];

    float div = maskSums[n];
    float sqrtdiv = sqrt(div);
    float finalMean = finalSum/div;

    int outBase = n * cSize * 3 + c;
    g_output[outBase] = finalMean;
    g_output[outBase + cSize] = finalMean * (sqrtdiv - 14.0f) * 0.1f;
    g_output[outBase + cSize*2] = finalMax;
  }
}

void main()
{
  if(USE_NHWC == 1)
    globalPoolingChannelsNHWC();
  else
    globalPoolingChannelsNCHW();
}
