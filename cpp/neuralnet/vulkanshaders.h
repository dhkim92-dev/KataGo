/**
 * @file vulkanshaders.h
 * @author Dohoon Kim(https://github.com/dhkim92-dev)
 * @details External symbols generated from SPIR-V shader binaries. 
 * These are produced by cmake/GenerateShaderSource.cmake during build. 
 * Symbol naming convention: _binary_<shader_name>_start, _binary_<shader_name>_end, _binary_<shader_name>_size
 */
#ifdef USE_VULKAN_BACKEND
#pragma once

#include <vector>
#include <cstddef>
#include <string>
#include "../neuralnet/activations.h"
#include "../neuralnet/vulkanhelpers.h"

extern "C" {
  // conv2d_fp32.glsl
  extern const unsigned char _binary_conv2d_fp32_start[];
  extern const unsigned char* _binary_conv2d_fp32_end;
  extern const size_t _binary_conv2d_fp32_size;

  // conv2d_tiled_bn_act_fp32.glsl
  extern const unsigned char _binary_conv2d_tiled_bn_act_3x3_fp32_start[];
  extern const unsigned char* _binary_conv2d_tiled_bn_act_3x3_fp32_end;
  extern const size_t _binary_conv2d_tiled_bn_act_3x3_fp32_size;

  // conv2d_tiled_bn_act_5x5_fp32.glsl
  extern const unsigned char _binary_conv2d_tiled_bn_act_5x5_fp32_start[];
  extern const unsigned char* _binary_conv2d_tiled_bn_act_5x5_fp32_end;
  extern const size_t _binary_conv2d_tiled_bn_act_5x5_fp32_size;

  // winograd_input_transform.glsl
  extern const unsigned char _binary_winograd_input_transform_start[];
  extern const unsigned char* _binary_winograd_input_transform_end;
  extern const size_t _binary_winograd_input_transform_size;

  extern const unsigned char _binary_winograd_input_transform_bnact_start[];
  extern const unsigned char* _binary_winograd_input_transform_bnact_end;
  extern const size_t _binary_winograd_input_transform_bnact_size;

  // winograd_output_transform.glsl
  extern const unsigned char _binary_winograd_output_transform_start[];
  extern const unsigned char* _binary_winograd_output_transform_end;
  extern const size_t _binary_winograd_output_transform_size;

  // add_pointwise_fp32.glsl
  extern const unsigned char _binary_add_pointwise_fp32_start[];
  extern const unsigned char* _binary_add_pointwise_fp32_end;
  extern const size_t _binary_add_pointwise_fp32_size;

  // matmul_fp32.glsl
  extern const unsigned char _binary_matmul_fp32_start[];
  extern const unsigned char* _binary_matmul_fp32_end;
  extern const size_t _binary_matmul_fp32_size;

  // xgemm_batched_fp32.glsl
  extern const unsigned char _binary_xgemm_batched_fp32_start[];
  extern const unsigned char* _binary_xgemm_batched_fp32_end;
  extern const size_t _binary_xgemm_batched_fp32_size;

  // batched_xgemm_direct_fp32.glsl
  extern const unsigned char _binary_batched_xgemm_direct_start[];
  extern const unsigned char* _binary_batched_xgemm_direct_end;
  extern const size_t _binary_batched_xgemm_direct_size;

  // xgemm_strided_batched_nn_fp32.glsl
  extern const unsigned char _binary_xgemm_strided_batched_nn_fp32_start[];
  extern const unsigned char* _binary_xgemm_strided_batched_nn_fp32_end;
  extern const size_t _binary_xgemm_strided_batched_nn_fp32_size;

  // bn_mask_identity_fp32.glsl
  extern const unsigned char _binary_bn_mask_identity_fp32_start[];
  extern const unsigned char* _binary_bn_mask_identity_fp32_end;
  extern const size_t _binary_bn_mask_identity_fp32_size;

  // bn_mask_relu_fp32.glsl
  extern const unsigned char _binary_bn_mask_relu_fp32_start[];
  extern const unsigned char* _binary_bn_mask_relu_fp32_end;
  extern const size_t _binary_bn_mask_relu_fp32_size;

  // bn_mask_mish_fp32.glsl
  extern const unsigned char _binary_bn_mask_mish_fp32_start[];
  extern const unsigned char* _binary_bn_mask_mish_fp32_end;
  extern const size_t _binary_bn_mask_mish_fp32_size;

  // bn_mask_mish_scale8_fp32.glsl
  extern const unsigned char _binary_bn_mask_mish_scale8_fp32_start[];
  extern const unsigned char* _binary_bn_mask_mish_scale8_fp32_end;
  extern const size_t _binary_bn_mask_mish_scale8_fp32_size;

  // bn_mask_silu_fp32.glsl
  extern const unsigned char _binary_bn_mask_silu_fp32_start[];
  extern const unsigned char* _binary_bn_mask_silu_fp32_end;
  extern const size_t _binary_bn_mask_silu_fp32_size;

  // global_pooling_channels_fp32.glsl
  extern const unsigned char _binary_global_pooling_channels_fp32_start[];
  extern const unsigned char* _binary_global_pooling_channels_fp32_end;
  extern const size_t _binary_global_pooling_channels_fp32_size;

  // value_head_pool_channels_fp32.glsl
  extern const unsigned char _binary_value_head_pool_channels_fp32_start[];
  extern const unsigned char* _binary_value_head_pool_channels_fp32_end;
  extern const size_t _binary_value_head_pool_channels_fp32_size;

  // sum_channels_fp32.glsl
  extern const unsigned char _binary_sum_channels_fp32_start[];
  extern const unsigned char* _binary_sum_channels_fp32_end;
  extern const size_t _binary_sum_channels_fp32_size;

  // add_channel_bias_nchw_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nchw_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nchw_fp32_end;
  extern const size_t _binary_add_channel_bias_nchw_fp32_size;

  // add_channel_bias_nc_identity_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nc_identity_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_identity_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_identity_fp32_size;

  // add_channel_bias_nc_relu_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nc_relu_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_relu_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_relu_fp32_size;

  // add_channel_bias_nc_mish_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nc_mish_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_mish_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_mish_fp32_size;

  // add_channel_bias_nc_mish_scale8_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nc_mish_scale8_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_mish_scale8_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_mish_scale8_fp32_size;

  // add_channel_bias_nc_silu_fp32.glsl
  extern const unsigned char _binary_add_channel_bias_nc_silu_fp32_start[];
  extern const unsigned char* _binary_add_channel_bias_nc_silu_fp32_end;
  extern const size_t _binary_add_channel_bias_nc_silu_fp32_size;

  // extract channel0_nchw_fp32.glsl
  extern const unsigned char _binary_extract_channel0_nchw_fp32_start[];
  extern const unsigned char* _binary_extract_channel0_nchw_fp32_end;
  extern const size_t _binary_extract_channel0_nchw_fp32_size;
}

/**
 * @brief SPIR-V shader data accessors.
 * Provides convenient access to embedded shader bytecode.
 * NCHW is the default data layout unless otherwise specified.
 */
namespace vk_shader {

  // conv2d_fp32 - Direct convolution layer
  extern const unsigned char* spirv_conv2d_fp32;
  extern size_t spirv_conv2d_fp32_size;

  extern const unsigned char* spirv_conv2d_tiled_bn_act_3x3_fp32;
  extern size_t spirv_conv2d_tiled_bn_act_3x3_fp32_size;

  extern const unsigned char* spirv_conv2d_tiled_bn_act_5x5_fp32;
  extern size_t spirv_conv2d_tiled_bn_act_5x5_fp32_size;

  extern const unsigned char* spirv_winograd_input_transform;
  extern size_t spirv_winograd_input_transform_size;

  extern const unsigned char* spirv_winograd_input_transform_bnact;
  extern size_t spirv_winograd_input_transform_bnact_size;

  extern const unsigned char* spirv_winograd_output_transform;
  extern size_t spirv_winograd_output_transform_size;

  // Convolution layers 3x3 with identity activation function
  // extern const unsigned char* spirv_conv2d_3x3_bn_fp32;
  // extern size_t spirv_conv2d_3x3_bn_fp32_size;

  // Convolution layers 3x3 with ReLU activation function
  // extern const unsigned char* spirv_conv2d_3x3_bn_relu_fp32;
  // extern size_t spirv_conv2d_3x3_bn_relu_fp32_size;

  // Convolution layers 3x3 with Mish activation function
  // extern const unsigned char* spirv_conv2d_3x3_bn_mish_fp32;
  // extern size_t spirv_conv2d_3x3_bn_mish_fp32_size;

  // Convolution layers 5x5 with identity activation function
  // extern const unsigned char* spirv_conv2d_5x5_bn_fp32;
  // extern size_t spirv_conv2d_5x5_bn_fp32_size;

  // Convolution layers 5x5 with ReLU activation function
  // extern const unsigned char* spirv_conv2d_5x5_bn_relu_fp32;
  // extern size_t spirv_conv2d_5x5_bn_relu_fp32_size;

  // Convolution layers 5x5 with Mish activation function
  // extern const unsigned char* spirv_conv2d_5x5_bn_mish_fp32;
  // extern size_t spirv_conv2d_5x5_bn_mish_fp32_size;

  // add_pointwise_fp32 - Element-wise add for skip connections
  extern const unsigned char* spirv_add_pointwise_fp32;
  extern size_t spirv_add_pointwise_fp32_size;

  // matmul_fp32 - Matrix multiplication
  extern const unsigned char* spirv_matmul_fp32;
  extern size_t spirv_matmul_fp32_size;

  extern const unsigned char* spirv_xgemm_batched_fp32;
  extern size_t spirv_xgemm_batched_fp32_size;

  // batched_xgemm_direct_fp32 - batched xgemm direct fp32
  extern const unsigned char* spirv_batched_xgemm_direct;
  extern size_t spirv_batched_xgemm_direct_size;

  // strided batched matmul fp32 - for 1x1 conv
  extern const unsigned char* spirv_xgemm_strided_batched_nn_fp32;;
  extern size_t spirv_xgemm_strided_batched_nn_fp32_size;

  // Future support, NCHW matmul Fp32
  // extern const unsigned char* spirv_matmul_tiled_chw_4x4x32_fp32;
  // extern size_t spirv_matmul_tiled_chw_4x4x32_fp32_size;

  // bn_mask_fp32 - Batch normalization with mask (identity activation)
  extern const unsigned char* spirv_bn_mask_identity_fp32;
  extern size_t spirv_bn_mask_identity_fp32_size;

  // bn_mask_relu_fp32 - Batch normalization with mask + ReLU
  extern const unsigned char* spirv_bn_mask_relu_fp32;
  extern size_t spirv_bn_mask_relu_fp32_size;

  // bn_mask_mish_fp32 - Batch normalization with mask + Mish
  extern const unsigned char* spirv_bn_mask_mish_fp32;
  extern size_t spirv_bn_mask_mish_fp32_size;

  // bn_mask_mish_scale8_fp32 - Batch normalization with mask + Mish + Scale8
  extern const unsigned char* spirv_bn_mask_mish_scale8_fp32;
  extern size_t spirv_bn_mask_mish_scale8_fp32_size;

  // bn_mask_silu_fp32 - Batch normalization with mask + silu
  extern const unsigned char* spirv_bn_mask_silu_fp32;
  extern size_t spirv_bn_mask_silu_fp32_size;

  // sum_channels_fp32 - Sum over channels
  extern const unsigned char* spirv_sum_channels_fp32;
  extern size_t spirv_sum_channels_fp32_size;

  // global_pooling_channels_fp32 - Global pooling with mask
  extern const unsigned char* spirv_global_pooling_channels_fp32;
  extern size_t spirv_global_pooling_channels_fp32_size;

  // value_head_pool_channels_fp32 - Value head pooling
  extern const unsigned char* spirv_value_head_pool_channels_fp32;
  extern size_t spirv_value_head_pool_channels_fp32_size;

  // add_channel_bias_nchw_fp32 - Add channel bias (identity)
  extern const unsigned char* spirv_add_channel_bias_nchw_fp32;
  extern size_t spirv_add_channel_bias_nchw_fp32_size;

  // add_channel_bias_nc_identity_fp32 - Add channel bias + Identity
  extern const unsigned char* spirv_add_channel_bias_nc_identity_fp32;
  extern size_t spirv_add_channel_bias_nc_identity_fp32_size;

  // add_channel_bias_nc_relu_fp32 - Add channel bias + ReLU
  extern const unsigned char* spirv_add_channel_bias_nc_relu_fp32;
  extern size_t spirv_add_channel_bias_nc_relu_fp32_size;

  // add_channel_bias_nc_mish_fp32 - Add channel bias + Mish
  extern const unsigned char* spirv_add_channel_bias_nc_mish_fp32;
  extern size_t spirv_add_channel_bias_nc_mish_fp32_size;

  // add_channel_bias_nc_mish_scale8_fp32 - Add channel bias + Mish + Scale8
  extern const unsigned char* spirv_add_channel_bias_nc_mish_scale8_fp32;
  extern size_t spirv_add_channel_bias_nc_mish_scale8_fp32_size;

  // add_channel_bias_nc_relu_fp32 - Add channel bias + ReLU
  extern const unsigned char* spirv_add_channel_bias_nc_silu_fp32;
  extern size_t spirv_add_channel_bias_nc_silu_fp32_size;

  // Extract channel 0 from NCHW fp32
  extern const unsigned char* spirv_extract_channel0_nchw_fp32;
  extern size_t spirv_extract_channel0_nchw_fp32_size;

  namespace spec {
    struct Conv2DSpec {
      uint32_t localSizeX = 16;
      uint32_t localSizeY = 8;
      uint32_t localSizeZ = 1;
    };

    struct Conv2DTiledBnAct3x3Spec {
      uint32_t localSizeX = 16;
      uint32_t localSizeY = 16;
      uint32_t localSizeZ = 1;
    };

    struct Conv2DTiledBnAct5x5Spec {
      uint32_t localSizeX = 16;
      uint32_t localSizeY = 16;
      uint32_t localSizeZ = 1;
    };

    struct AddPointWiseSpec {
      uint32_t localSizeX = 256;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
    };

    struct MatmulSpec {
      uint32_t localSizeX = 16;
      uint32_t localSizeY = 16;
      uint32_t localSizeZ = 1;
    };

    struct BatchNormMaskSpec {
      uint32_t localSizeX = 32;
      uint32_t localSizeY = 8;
      uint32_t localSizeZ = 1;
    };

    struct GlobalPoolingChannelsSpec {
      uint32_t localSizeX = 128;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
    };

    struct ValueHeadPoolingChannelsSpec {
      uint32_t localSizeX = 128;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
    };

    /**
     * @brief SumChannels Kernel Specialization Info
     * @param XYSTRIDE - power of two parallelism stride for reduction, should be gl_WorkGroupSize.x;
     * @param CHANNELSTRIDE - stride for channels, should be get gl_WorkGroupSize.y
     * @param LOCALSIZE_TOTAL = should be get gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z
     */
    struct SumChannelsSpec {
      uint32_t localSizeX = 32;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int XYSTRIDE;
      int CHANNELSTRIDE;
      int LOCALSIZE_TOTAL;
    };

    struct AddChannelBiasNCHWSpec {
      uint32_t localSizeX = 16;
      uint32_t localSizeY = 16;
      uint32_t localSizeZ = 1;
    };

    struct AddChannelBiasNCSpec {
      uint32_t localSizeX = 64;
      uint32_t localSizeY = 4;
      uint32_t localSizeZ = 1;
    };

    struct ExtractChannel0NCHWSpec {
      uint32_t localSizeX = 64;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
    };

    struct WinogradInputTransformBnActSpec {
      uint32_t localSizeX = 1;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int inTileYSize = 4;
      int inTileXSize = 4;
      int outTileYSize = 2;
      int outTileXSize = 2;
      int inTileYOffset = -1;
      int inTileXOffset = -1;
      int convY = 3;
      int convX = 3;
      int activation = 0; // 0: Identity, 1: ReLU, 2: Mish, 12: Mish + Scale8
    };

    struct WinogradInputTransformSpec {
      uint32_t localSizeX = 1;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int inTileYSize = 4;
      int inTileXSize = 4;
      int outTileYSize = 2;
      int outTileXSize = 2;
      int inTileYOffset = -1;
      int inTileXOffset = -1;
      int convY = 3;
      int convX = 3;
    };

    struct WinogradOutputTransformSpec {
      uint32_t localSizeX = 1;
      uint32_t localSizeY = 1;
      uint32_t localSizeZ = 1;
      int inTileYSize = 4;
      int inTileXSize = 4;
      int outTileYSize = 2;
      int outTileXSize = 2;
      int convY = 3;
      int convX = 3;
    };

    struct XGEMMBatchedSpec {
      uint32_t localSizeX = 16;
      uint32_t localSizeY = 16;
      uint32_t localSizeZ = 1;
      uint32_t MWG=64;
      uint32_t NWG=64;
      uint32_t KWG=32;
      uint32_t MDIMC=16;
      uint32_t NDIMC=16;
      uint32_t MDIMA=16;
      uint32_t NDIMB=16;
    };

    struct XgemmDirectSpec {
      uint32_t localSizeX = 16;
      uint32_t localSizeY = 16;
      uint32_t localSizeZ = 1;
      int WGD = 32;
      int MDIMCD = 8;
      int NDIMCD = 8;
      int MDIMAD = 16;
      int NDIMBD = 16;
      int KWID = 2;
      int PADA = 1;
      int PADB = 1;
    };
  };

  namespace push {
    struct Conv2DPushConstantParams {
      int nSize;
      int xSize;
      int ySize;
      int ocSize;
      int icSize;
      int filterXRadius;
      int filterYRadius;
      int xyStride;
    };

    struct Conv2DTiledBnActParams {
      uint32_t batchSize;
      uint32_t inChannels;
      uint32_t outChannels;
      uint32_t nnYLen;
      uint32_t nnXLen;
      uint32_t filterH;
      uint32_t filterW;
      uint32_t activation; // 0: Identity, 1: ReLU, 2: Mish, 3: Mish + Scale8
    };

    struct WinogradInputTransformParams {
      uint32_t batchSize;
      uint32_t nnYLen;
      uint32_t nnXLen;
      uint32_t numTilesY;
      uint32_t numTilesX;
      uint32_t inChannels;
      uint32_t inChannelsPadded;
      uint32_t ntxtySizePadded;
    };

    struct WinogradOutputTransformParams {
      int batchSize;
      int ySize;
      int xSize;
      int numTilesY;
      int numTilesX;
      int outChannels;
      int outChannelsPadded;
      int ntxtySizePadded;
    };

    /**
     * @brief Matmul pipeline Push Constant Parameters
     * @param M: rows of A and C, each batch
     * @param K: cols of A and rows of B, inChannels
     * @param N: cols of B and C, outChannels
     * @param numBatchElts: number of batches
     * @param cTranspose: whether output C is transposed or not
     */
    struct MatmulFp32Params {
      uint32_t M;  
      uint32_t K;  
      uint32_t N;
      uint32_t numBatchElts;
      uint32_t cTranspose; // Output Transpose
    };

    struct XGEMMBatchedParams{
      uint32_t M;  
      uint32_t N;  
      uint32_t K;
      uint32_t aOne;
      uint32_t aTwo;
      uint32_t bOne;
      uint32_t bTwo;
      uint32_t cOne;
      uint32_t cTwo;
    };

    struct BatchedXGEMMDirectParams {
      uint32_t M;  
      uint32_t N;
      uint32_t K;  
      uint32_t aLead;
      uint32_t bLead;
      uint32_t cLead;
      uint32_t aTranspose; // Input A Transpose
      uint32_t bTranspose; // Input B Transpose
      uint32_t cTranspose; // Output C Transpose
    };

    struct XgemmStridedBatchedFp32Params {
      uint32_t kSizeM;  
      uint32_t kSizeN;
      uint32_t kSizeK;  
      uint32_t aLead;
      uint32_t aStride;
      uint32_t bLead;
      uint32_t bStride;
      uint32_t cLead;
      uint32_t cStride;
      uint32_t cTranspose;
    };

    struct BatchedXgemmDirectFp32Params {
      uint32_t kSizeM;  
      uint32_t kSizeN;
      uint32_t kSizeK;  
      uint32_t aLead;
      uint32_t bLead;
      uint32_t cLead;
      uint32_t cTranspose; // Output C Transpose
    };

    /**
     * @brief Batch Normalization Mask Fp32 Push Constant Parameters 
     * @param batchSize
     * @param numChannels
     * @param nnYLen
     */
    struct BatchNormMaskParams {
      uint32_t batchSize;
      uint32_t numChannels;
      uint32_t nnXYLen;
    };

    /**
     * @brief Sum Channels Fp32 Push Constant Parameters
     * @param nSize - number of batch
     * @param cSize - number of channels
     * @param xySize - H*W spatial size
     */
    struct SumChannelsParams {
      uint32_t nSize;
      uint32_t cSize;
      uint32_t xySize;
    };

    /**
     * @brief MatBias Push Constant Parameters
     */
    struct MatBiasPushParams {
      uint32_t batchSize;
      uint32_t numChannels;
    };

    /**
      * @brief Global Pooling Channels Push Constant Parameters
      */
    struct GlobalPoolingChannelsParams {
      uint32_t batchSize;
      uint32_t gpoolChannels;
      uint32_t nnXYLen;
    };

    /**
      * @brief Value Head Pooling Channels Push Constant Parameters
      */
    struct ValueHeadPoolingChannelsParams {
      uint32_t batchSize;
      uint32_t gpoolChannels;
      uint32_t nnXYLen;
    };

    /**
      *  @brief Add Point Wise Push Constant Parameters
      **/
    struct AddPointWiseParams {
      uint32_t totalSize;
    };

    /**
     * @brief Add Channel Bias NCHW Push Constant Parameters
     */
    struct AddChannelBiasNCHWParams {
      uint32_t ncSize;
      uint32_t xySize;
    };

    /**
     * @brief Add Channel Bias NC Push Constant Parameters
     */
    struct AddChannelBiasNCParams {
      uint32_t nSize;
      uint32_t cSize;
    };

    /**
     * @brief Push parameters for ExtractChannel0NCHW shader
     * @param nSize: number of batches
     * @param cSize: number of input channels
     * @param xySize: H*W spatial size
     */
    struct ExtractChannel0NCHWParams {
      int nSize;  
      int cSize;
      int xySize;
    };

    struct NCHWPushConstantParams {
      uint32_t N; // Batch size
      uint32_t C; // Channels
      uint32_t H; // Height
      uint32_t W; // Width
    };
  };

  namespace tune {

    struct GPoolTuneParams {
      int XYSTRIDE=32;
      int CHANNELSTRIDE=1;
      int BATCHSTRIDE=2;
    };

    struct ConvTuneParams {
      uint32_t inTileYSize;
      uint32_t inTileXSize;
      uint32_t outTileYSize;
      uint32_t outTileXSize;
      uint32_t inputTransformLocalXSize;
      uint32_t inputTransformLocalYSize;
      uint32_t outputTransformLocalXSize;
      uint32_t outputTransformLocalYSize;
      uint32_t outputTransformLocalZSize;
    };

    struct XgemmTuneParams {
      uint32_t MDIMC;
      uint32_t NDIMC;
      uint32_t MWG;
      uint32_t NWG;
      uint32_t KWG;
      uint32_t MDIMA;
      uint32_t NDIMB;
    };

    struct XgemmDirectTuneParams {
      uint32_t WGD;
      uint32_t MDIMCD;
      uint32_t NDIMCD;
      uint32_t MDIMAD;
      uint32_t NDIMBD;
      uint32_t KWID;
      uint32_t PADA;
      uint32_t PADB;
    };

    struct VulkanTuneParams {
      GPoolTuneParams gPool;
      ConvTuneParams conv3x3;
      ConvTuneParams conv5x5;
      XgemmTuneParams xgemm;
      XgemmDirectTuneParams xgemmDirect;

      VulkanTuneParams(const VulkanTuneParams& other) = default;
      VulkanTuneParams& operator=(const VulkanTuneParams& other) = default;

      bool isValid() const;
      bool operator==(const VulkanTuneParams& other) const;
      bool operator!=(const VulkanTuneParams& other) const { return !(*this == other); }

      static void save(const std::string& filename, const VulkanTuneParams& config);
      static VulkanTuneParams load(const std::string& filename);

      VulkanTuneParams() {
        // conv3x3 = ConvTuneParams{
        //   .inTileYSize = 6,
        //   .inTileXSize = 6,
        //   .outTileYSize = 4,
        //   .outTileXSize = 4,
        //   .inputTransformLocalXSize = 4,
        //   .inputTransformLocalYSize = 2,
        //   .outputTransformLocalXSize = 8,
        //   .outputTransformLocalYSize = 2,
        //   .outputTransformLocalZSize = 2
        // };
        // conv5x5 = ConvTuneParams{
        //   .inTileYSize = 6,
        //   .inTileXSize = 6,
        //   .outTileYSize = 2,
        //   .outTileXSize = 2,
        //   .inputTransformLocalXSize = 4,
        //   .inputTransformLocalYSize = 2,
        //   .outputTransformLocalXSize = 8,
        //   .outputTransformLocalYSize = 2,
        //   .outputTransformLocalZSize = 2
        // };
        // xgemm = XgemmTuneParams{
        //   .MDIMC = 16,
        //   .NDIMC = 16,
        //   .MWG = 64,
        //   .NWG = 64,
        //   .KWG = 16,
        //   .MDIMA = 16,
        //   .NDIMB = 16
        // };

        // xgemmDirect = XgemmDirectTuneParams{
        //   .WGD = 32,
        //   .MDIMCD = 8,
        //   .NDIMCD = 8,
        //   .MDIMAD = 8,
        //   .NDIMBD = 8,
        //   .KWID = 2,
        //   .PADA = 1,
        //   .PADB = 1
        // };

        conv3x3 = ConvTuneParams();
        conv3x3.inTileYSize = 6;
        conv3x3.inTileXSize = 6;
        conv3x3.outTileYSize = 4;
        conv3x3.outTileXSize = 4;
        conv3x3.inputTransformLocalXSize = 128;
        conv3x3.inputTransformLocalYSize = 2;
        conv3x3.outputTransformLocalXSize = 8;
        conv3x3.outputTransformLocalYSize = 4;
        conv3x3.outputTransformLocalZSize = 8;

        conv5x5 = ConvTuneParams();
        conv5x5.inTileYSize = 6;
        conv5x5.inTileXSize = 6;
        conv5x5.outTileYSize = 2;
        conv5x5.outTileXSize = 2;
        conv5x5.inputTransformLocalXSize = 128;
        conv5x5.inputTransformLocalYSize = 2;
        conv5x5.outputTransformLocalXSize = 8;
        conv5x5.outputTransformLocalYSize = 2;
        conv5x5.outputTransformLocalZSize = 2;

        xgemm = XgemmTuneParams();
        xgemm.MDIMC = 16;
        xgemm.NDIMC = 16;
        xgemm.MWG = 64;
        xgemm.NWG = 64;
        xgemm.KWG = 16;
        xgemm.MDIMA = 16;
        xgemm.NDIMB = 16;

        xgemmDirect = XgemmDirectTuneParams();
        xgemmDirect.WGD = 32;
        xgemmDirect.MDIMCD = 8;
        xgemmDirect.NDIMCD = 8;
        xgemmDirect.MDIMAD = 8;
        xgemmDirect.NDIMBD = 8;
        xgemmDirect.KWID = 2;
        xgemmDirect.PADA = 1;
        xgemmDirect.PADB = 1;
      }
    };

  }

  struct ComputePipelines {
    // const VulkanTuneParams tuneParams;
    // const bool usingFP16Storage;
    // const bool usingFP16Compute;
    // const bool usingFP16TensorCores;
    VkDevice device;
    const tune::VulkanTuneParams tuneParams;
    VkPipelineCache cache;

    // In this code, assume that NCHW is default format if no postfix is given.

    // Conv2D pipelines
    Pipeline conv2dFp32; 
    Pipeline conv2dTiledBnAct3x3Fp32; // Conv2d + Tiled + BatchNorm + Activation fused pipeline
    Pipeline conv2dTiledBnAct5x5Fp32; // Conv2d + Tiled + BatchNorm + Activation fused pipeline

    Pipeline winogradInputTransform3x3;
    Pipeline winogradInputTransform5x5;

    Pipeline winogradInputTransform3x3_bnact_identity;
    Pipeline winogradInputTransform3x3_bnact_relu;
    Pipeline winogradInputTransform3x3_bnact_mish;
    Pipeline winogradInputTransform3x3_bnact_mish_scale8;
    Pipeline winogradInputTransform5x5_bnact_identity;
    Pipeline winogradInputTransform5x5_bnact_relu;
    Pipeline winogradInputTransform5x5_bnact_mish;
    Pipeline winogradInputTransform5x5_bnact_mish_scale8;

    Pipeline winogradOutputTransform3x3;
    Pipeline winogradOutputTransform5x5;

    Pipeline addPointWiseFp32;  // operation for skipping connections

    // Pipeline for matrix multiplication
    Pipeline matmulFp32; 
    Pipeline batchedXgemmDirect;
    Pipeline xgemmBatchedFp32;

    // note that conv1x1 can be implemented as matmul operation
    // Pipeline stridedBatchedMatmulFp32;
    Pipeline xgemmStridedBatchedFp32;

    // Batch Normalization pipelines
    // note that prediction phase does not need batch normalization operation separately
    // as the parameters can be folded into convolution scale and bias.
    Pipeline batchNormMaskIdentity;
    Pipeline batchNormMaskRelu;
    Pipeline batchNormMaskMish;
    Pipeline batchNormMaskMishScale8;
    Pipeline batchNormMaskSilu;

    // Pooling pipelines
    Pipeline globalPoolingChannelsFp32;
    Pipeline valueHeadPoolingChannelsFp32;
    
    // Element wise operations
    std::vector<Pipeline> sumChannels;

    Pipeline addChannelBiasNCHWFp32;
    Pipeline addChannelBiasNCIdentity;
    Pipeline addChannelBiasNCRelu;
    Pipeline addChannelBiasNCMish;
    Pipeline addChannelBiasNCMishScale8;
    Pipeline addChannelBiasNCSilu;
    Pipeline extractChannel0NCHWFp32;

    ComputePipelines(VkDevice device_, const tune::VulkanTuneParams& tuneParams_);
    ComputePipelines() = delete;
    ComputePipelines(const ComputePipelines&) = delete;
    ComputePipelines& operator=(const ComputePipelines&) = delete;

    ~ComputePipelines();

  private :
    void createPipelines();
    void destroyPipelines();
    void destroyPipeline(Pipeline& pipeline);
    void createPipeline(
      std::string pipelineName,
      const unsigned char* spirvBytes,
      size_t spirvSize,
      size_t bindingSize,
      uint32_t pushConstantSize,
      Pipeline &outPipeline,
      VkSpecializationInfo* specializationInfo = nullptr,
      uint32_t localSizeX = 1,
      uint32_t localSizeY = 1,
      uint32_t localSizeZ = 1
    );
    /**
     * @brief Create a Conv2d Fp32 object
    */
    // void createConv2dFp32();

    /**
     * @brief Create Conv2d Tiled Bn + Activation 3x3 Fp32 object
     */
    // void createConv2dTiledBnAct3x3Fp32();

    /**
     * @brief Create Conv2d Tiled Bn + Activation 5x5 Fp32 object
     */
    // void createConv2dTiledBnAct5x5Fp32();

    void createWinogradInputTransform();

    void createWinogradInputTransformBnAct();

    void createWinogradOutputTransform();

    /**
     * @brief Create a Conv2d3x3 Bn + Identity Activation fused Fp32 objects.
     */
    // void createConv2d3x3BnFp32();

    /**
     * @brief Create a Conv2d3x3 Bn + ReLU Activation fused Fp32 objects.
     */
    // void createConv2d3x3BnReluFp32();
    /**
     * @brief Create a Conv2d3x3 Bn Mish Fp32 object
     */
    // void createConv2d3x3BnMishFp32();

    /**
     * @brief Create a Conv2d5x5 Bn + Identity Activation fused Fp32 objects.
     */
    // void createConv2d5x5BnFp32();

    /**
     * @brief Create a Conv2d5x5 Bn + ReLU Activation fused Fp32 objects.
     */
    // void createConv2d5x5BnReluFp32();

    /**
     * @brief Create a Conv2d5x5 Bn + ReLU Activation fused Fp32 objects.
     */
    // void createConv2d5x5BnReluFp32();

    /**
     * @brief Create a Conv2d5x5 Bn Mish Fp32 object
     */
    // void createConv2d5x5BnMishFp32();

    /**
     * @brief Create a Add Point Wise Fp32 object
     */
    void createAddPointWiseFp32();

    /**
     * @brief Create a Matmul Fp32 object
     */
    void createMatmulFp32();

    /**
     * @brief Create a Batched XGEMM Direct Fp32 object
     */
    void createBatchedXgemmDirect();

    /**
     * @brief Create a XGEMM Batched Fp32 object
     */
    void createXGEMMBatchedFp32();

     /**
     * @brief Create a Strided Batched Matmul Fp32 object
     */
    void createXGEMMStridedBatchedFp32();

    /**
     * @brief Create a BatchNorm Mask Fp32 object
     */
    void createBatchNormMaskIdentity();

    /**
     * @brief Create a BatchNorm Mask + ReLU object
     */
    void createBatchNormMaskRelu();

    /**
     * @brief Create a BatchNorm Mask + Mish object
     */
    void createBatchNormMaskMish();

    /**
     * @brief Create a BatchNorm Mask + Mish + Scale8 object
     */
    void createBatchNormMaskMishScale8();

    void createBatchNormMaskSilu();

    /**
     * @brief Create a Global Average Pool Fp32 object
     */
    void createGlobalPoolingChannelsFp32();

    /**
     * @brief Create a Value Head Pool Channels Fp32 object
     */
    void createValueHeadPoolingChannelsFp32();

    /**
     * @brief Create a Sum Channels Fp32 object
     */
    void createSumChannelsFp32();

    /**
     * @brief Create a Add Channel Bias NCHW Fp32 object
     */
    void createAddChannelBiasNCHWFp32();

    /**
     * @brief Create a Add Channel Bias NC
     */
    void createAddChannelBiasNCIdentity();

    /**
     * @brief Create a Add Channel Bias NC + ReLU
     */
    void createAddChannelBiasNCRelu();

    /**
     * @brief Create a Add Channel Bias NC + Mish
     */
    void createAddChannelBiasNCMish();

    /**
     * @brief Create a Add Channel Bias NC + Mish + Scale8
     */
    void createAddChannelBiasNCMishScale8();

    /**
     * @brief Create a Add Channel Bias NC + Silu
     */
    void createAddChannelBiasNCSilu();
    
    /**
     * @brief Create a Extract Channel 0 NCHW Fp32 object
     */
    void createExtractChannel0NCHWFp32();
  };
}

#endif
