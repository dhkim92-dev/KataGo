/**
 * @file vulkanshaders.h
 * @author dhkim92-dev
 * @details External symbols generated from SPIR-V shader binaries. 
 * These are produced by cmake/GenerateShaderSource.cmake during build. 
 * Symbol naming convention: _binary_<shader_name>_start, _binary_<shader_name>_end, _binary_<shader_name>_size
 */
#ifdef USE_VULKAN_BACKEND
#pragma once

#include <vector>
#include <cstddef>
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkancompute.h"

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

  // winograd_input_transform_fp32.comp
  extern const unsigned char _binary_winograd_input_transform_start[];
  extern const unsigned char* _binary_winograd_input_transform_end;
  extern const size_t _binary_winograd_input_transform_size;

  extern const unsigned char _binary_winograd_input_transform_bnact_fp32_start[];
  extern const unsigned char* _binary_winograd_input_transform_bnact_fp32_end;
  extern const size_t _binary_winograd_input_transform_bnact_fp32_size;

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

  // bn_mask_idetity_fp32.glsl
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

// bn_mask_relu_fp32.glsl
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


  // Transformer kernels

  // RMSNorm
  extern const unsigned char _binary_rms_norm_fp32_start[];
  extern const unsigned char* _binary_rms_norm_fp32_end;
  extern const size_t _binary_rms_norm_fp32_size;

  // Spatial RMS Norm
  extern const unsigned char _binary_spatial_rms_norm_sum_sq_fp32_start[];
  extern const unsigned char* _binary_spatial_rms_norm_sum_sq_fp32_end;
  extern const size_t _binary_spatial_rms_norm_sum_sq_fp32_size;

  extern const unsigned char _binary_spatial_rms_norm_reduce_fp32_start[];
  extern const unsigned char* _binary_spatial_rms_norm_reduce_fp32_end;
  extern const size_t _binary_spatial_rms_norm_reduce_fp32_size;

  extern const unsigned char _binary_spatial_rms_norm_apply_fp32_start[];
  extern const unsigned char* _binary_spatial_rms_norm_apply_fp32_end;
  extern const size_t _binary_spatial_rms_norm_apply_fp32_size;

  // Attention
  extern const unsigned char _binary_scale_dot_product_attention_fp32_start[];
  extern const unsigned char* _binary_scale_dot_product_attention_fp32_end;
  extern const size_t _binary_scale_dot_product_attention_fp32_size;

  extern const unsigned char _binary_scale_dot_product_attention_naive_fp32_start[];
  extern const unsigned char* _binary_scale_dot_product_attention_naive_fp32_end;
  extern const size_t _binary_scale_dot_product_attention_naive_fp32_size;

  // Swiglu
  extern const unsigned char _binary_transformer_swiglu_fp32_start[];
  extern const unsigned char* _binary_transformer_swiglu_fp32_end;
  extern const size_t _binary_transformer_swiglu_fp32_size;

  // RoPE
  extern const unsigned char _binary_transformer_apply_rope_fp32_start[];
  extern const unsigned char* _binary_transformer_apply_rope_fp32_end;
  extern const size_t _binary_transformer_apply_rope_fp32_size;
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

  extern const unsigned char* spirv_winograd_input_transform_bnact_fp32;
  extern size_t spirv_winograd_input_transform_bnact_fp32_size;

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

  // bn_mask_identity_fp32 - Batch normalization with mask (identity activation)
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

  // bn_mask_silu_fp32 - Batch normalization with mask + SiLU
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

  // add_channel_bias_nc_silu_fp32 - Add channel bias + SiLU
  extern const unsigned char* spirv_add_channel_bias_nc_silu_fp32;
  extern size_t spirv_add_channel_bias_nc_silu_fp32_size;

  // Extract channel 0 from NCHW fp32
  extern const unsigned char* spirv_extract_channel0_nchw_fp32;
  extern size_t spirv_extract_channel0_nchw_fp32_size;

  // rms_norm_fp32
  extern const unsigned char* spirv_rms_norm_fp32;
  extern size_t spirv_rms_norm_fp32_size;

  // spatial_rms_norm_sum_sq_fp32
  extern const unsigned char* spirv_spatial_rms_norm_sum_sq_fp32;
  extern size_t spirv_spatial_rms_norm_sum_sq_fp32_size;

  // spatial_rms_norm_reduce_fp32
  extern const unsigned char* spirv_spatial_rms_norm_reduce_fp32;
  extern size_t spirv_spatial_rms_norm_reduce_fp32_size;

  // sptial_rms_norm_apply_fp32
  extern const unsigned char* spirv_spatial_rms_norm_apply_fp32;
  extern size_t spirv_spatial_rms_norm_apply_fp32_size;

  // scale_dot_product_attention_fp32
  extern const unsigned char* spirv_scale_dot_product_attention_fp32;
  extern size_t spirv_scale_dot_product_attention_fp32_size;

  // scale_dot_product_attention_naive_fp32
  extern const unsigned char* spirv_scale_dot_product_attention_naive_fp32;
  extern size_t spirv_scale_dot_product_attention_naive_fp32_size;

  // swiglu
  extern const unsigned char* spirv_transformer_swiglu_fp32;
  extern const size_t spirv_transformer_swiglu_fp32_size;

  // rope
  extern const unsigned char* spirv_transformer_apply_rope_fp32;
  extern const size_t spirv_transformer_apply_rope_fp32_size;

  namespace push {
    struct Conv2DPushConstantParams {
      uint32_t batchSize; // Batch size
      uint32_t inChannels; // Input channels
      uint32_t outChannels; // Output channels
      uint32_t nnYLen;
      uint32_t nnXLen;
      uint32_t filterH; // Filter height
      uint32_t filterW; // Filter width
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
     */
    struct SumChannelsParams {
      uint32_t batchSize;
      uint32_t numChannels;
      uint32_t nnXYLen;
    };

    /**
     * @brief MatBias Push Constant Parameters
     */
    struct MatBiasFp32Params {
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
      uint32_t size;
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

    struct ExtractChannel0NCHWParams {
      uint32_t nSize;
      uint32_t cSize;
      uint32_t xySize;
    };

    struct NCHWPushConstantParams {
      uint32_t N; // Batch size
      uint32_t C; // Channels
      uint32_t H; // Height
      uint32_t W; // Width
    };

    // RMSNormalization Push Constant Parameters for FP32 kernel
    struct RMSNormPushConstantParams {
      uint32_t nSize;
      uint32_t cSize;
      uint32_t xySize;
      float eps;
    };

    // Spatial RMSNormalization Sum Sq Push Constant Parameters for FP32 kernel
    struct SpatialRMSNormSumSqPushConstantParams {
      uint32_t nSize;
      int32_t cSize;
      int32_t xySize;
      int32_t tilesPerGroup;
    };

    // Spatial RMSNormalization Reduce Push Constant Parameters for FP32 kernel
    struct SpatialRMSNormReducePushConstantParams {
      int32_t nSize;
      int32_t numPartials;
      int32_t tilesPerGroup;
    };

    // Spatial RMSNormalization Apply Push Constant Parameters for FP32 kernel
    struct SpatialRMSNormApplyPushConstantParams {
      int32_t nSize;
      int32_t cSize;
      int32_t xySize;
      float scale;
    };

    // ScaleDotProductAttention Push Constant Parameters for FP32 kernel
    struct ScaleDotProductAttentionPushConstantParams {
      uint32_t seqLen;
      uint32_t numHeads;
      uint32_t numKVHeads;
      float scale;
    };

    // ScaleDotProductAttention Naive Push Constant Parameters for FP32 kernel
    struct ScaleDotProductAttentionNaivePushConstantParams {
      int seqLen;
      int numHeads;
      int numKVHeads;
      float scale;
    };

    // Transformer SwiGLU Push Constant Parameters for FP32 kernel
    struct TransformerSwiGLUPushContantParams {
      int size;
    };

    struct TransformerRoPEPushConstantParams {
      int nSize;
      int numBufHeads;
      int numKVHeads;
      int headDim;
      int xySize;
      int numPairs;
      int learnableRope;
    };
  }

  namespace spec {
    struct AddPointwiseSpec {
      uint32_t localSizeX;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1;
      int ELTS_PER_THREAD;
    };

    struct SumChannelsSpec {
      uint32_t localSizeX;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1;
      int XYSTRIDE=32;
      int CHANNELSTRIDE=1;
      int LOCALSIZE_TOTAL=1;
    };

    struct AddChannelBiasesNCHWSpec {
      uint32_t localSizeX=32;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1;
      int XY_ELTS_PER_THREAD = 1;
      int NC_ELTS_PER_THREAD = 1;
    };

    struct AddChannelBiasesNCSpec {
      uint32_t localSizeX=256;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1;
    };

    struct ExtractChannel0NCHWSpec {
      uint32_t localSizeX=64;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1;
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
      int activation = 0; // 0: Identity, 1: ReLU, 2: Mish, 12: Mish + Scale8 3: SiLU
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

    struct RMSNormSpec {
      uint32_t localSizeX;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1;
      int WG_C_SIZE=64;
      int WG_XY_SIZE=1;
      int C_PER_THREAD=4;
    };

    struct SpatialRMSNormSumSqSpec {
      uint32_t localSizeX;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1;
      int TILE_SIZE=32;
    };

    struct SpatialRMSNormReduceSpec {
      uint32_t localSizeX;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1;
      int TILE_SIZE=32;
    };

    struct SpatialRMSNormApplySpec {
      uint32_t localSizeX=32;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1;
      int APPLY_ELTS_PER_THREAD=1;
    };

    struct ScaleDotProductAttentionSpec {
      uint32_t localSizeX;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1; 
      int ATTN_BLOCK_Q=32;
      int ATTN_BLOCK_KV=32;
      int Q_PER_THREAD=1;
      int ATTN_HEAD_DIM = 1;
      int ATTN_V_HEAD_DIM=1;
    };

    struct ScaleDotProductAttentionNaiveSpec {
      uint32_t localSizeX=32;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1;
      int ATTN_HEAD_DIM=1;
      int ATTN_V_HEAD_DIM;
    };

    struct TransformerSwiGLUSpec {
      uint32_t localSizeX;
      uint32_t localSizeY;
      uint32_t localSizeZ;
      int ELTS_PER_THREAD=1;
    };

    struct TransformerApplyRoPESpec {
      uint32_t localSizeX=32;
      uint32_t localSizeY=1;
      uint32_t localSizeZ=1;
    };
  }

  /**
   * @brief Compute pipelines for various operations
   */
  struct ComputePipelines {
    // const VulkanTuneParams tuneParams;
    // const bool usingFP16Storage;
    // const bool usingFP16Compute;
    // const bool usingFP16TensorCores;
    VkDevice device;
    const VulkanTuneParams& tuneParams;
    VkPipelineCache cache;
    const bool useFP16;
    const bool useTensorcore;
    const int qHeadDim=-1;
    const int vHeadDim=-1;
    // const LoadedModel* loadedModel;
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
    Pipeline winogradInputTransform3x3_bnact_silu;
    Pipeline winogradInputTransform5x5_bnact_identity;
    Pipeline winogradInputTransform5x5_bnact_relu;
    Pipeline winogradInputTransform5x5_bnact_mish;
    Pipeline winogradInputTransform5x5_bnact_mish_scale8;
    Pipeline winogradInputTransform5x5_bnact_silu;

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
    Pipeline batchNormMaskIdentityFp32;
    Pipeline batchNormMaskReluFp32;
    Pipeline batchNormMaskMishFp32;
    Pipeline batchNormMaskMishScale8Fp32;
    Pipeline batchNormMaskSiluFp32;

    // Pooling pipelines
    Pipeline globalPoolingChannelsFp32;
    Pipeline valueHeadPoolingChannelsFp32;
    
    // Element wise operations
    std::map<int, Pipeline> sumChannelsFp32;

    Pipeline addChannelBiasNCHWFp32;
    Pipeline addChannelBiasNCIdentityFp32;
    Pipeline addChannelBiasNCReluFp32;
    Pipeline addChannelBiasNCMishFp32;
    Pipeline addChannelBiasNCMishScale8Fp32;
    Pipeline addChannelBiasNCSiluFp32;
    Pipeline extractChannel0NCHWFp32;

    // Transformer and attention kernels
    Pipeline rmsNormFp32;
    Pipeline spatialRmsNormSumSqFp32;
    Pipeline spatialRmsNormReduceFp32;
    Pipeline spatialRmsNormApplyFp32;
    // std::unordered_map<AttnDims ,Pipeline, AttnDimsHash> scaleDotProductAttentionFp32;
    Pipeline scaleDotProductAttentionFp32;
    Pipeline scaleDotProductAttentionNaiveFp32;
    // std::unordered_map<AttnDims, Pipeline, AttnDimsHash> scaleDotProductAttentionNaiveFp32;
    Pipeline transformerSwiGLUFp32;
    Pipeline transformerApplyRoPEFp32;

    ComputePipelines(
      VkDevice device_, 
      const VulkanTuneParams& tuneParams_,
      const bool useFP16_,
      const bool useTensorcore_,
      const int qHeadDim_,
      const int vHeadDim_
    );
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
      VkSpecializationInfo* specializationInfo = nullptr
    );
    /**
     * @brief Create a Conv2d Fp32 object
    */
    void createConv2dFp32();

    /**
     * @brief Create Conv2d Tiled Bn + Activation 3x3 Fp32 object
     */
    void createConv2dTiledBnAct3x3Fp32();

    /**
     * @brief Create Conv2d Tiled Bn + Activation 5x5 Fp32 object
     */
    void createConv2dTiledBnAct5x5Fp32();

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
    void createBatchNormMaskIdentityFp32();

    /**
     * @brief Create a BatchNorm Mask + ReLU Fp32 object
     */
    void createBatchNormMaskReluFp32();

    /**
     * @brief Create a BatchNorm Mask + Mish Fp32 object
     */
    void createBatchNormMaskMishFp32();

    /**
     * @brief Create a BatchNorm Mask + Mish + Scale8 Fp32 object
     */
    void createBatchNormMaskMishScale8Fp32();

    /**
     * @breif Create a BatchNorm Mask SiLU Fp32 object
     */
    void createBatchNormMaskSiluFp32();

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
     * @brief Create a Add Channel Bias NC Fp32 object
     */
    void createAddChannelBiasNCIdentityFp32();

    /**
     * @brief Create a Add Channel Bias NC + ReLU Fp32 object
     */
    void createAddChannelBiasNCReluFp32();

    /**
     * @brief Create a Add Channel Bias NC + Mish Fp32 object
     */
    void createAddChannelBiasNCMishFp32();

    /**
     * @brief Create a Add Channel Bias NC + Mish + Scale8 Fp32 object
     */
    void createAddChannelBiasNCMishScale8Fp32();

    /**
     * @brief Create a Add Channel Bias NC + SiLU Fp32 object
     */
    void createAddChannelBiasNCSiluFp32();
    
    /**
     * @brief Create a Extract Channel 0 NCHW Fp32 object
     */
    void createExtractChannel0NCHWFp32();

    /**
     * @brief Create RMSNorm Fp32 Kernel
     */
    void createRmsNormFp32();

    /**
     * @brief Create SpatialRMSNormSumSqFp32 Kernel
     */
    void createSpatialRMSNormSumSqFp32();

    /**
     * @brief Create SpatialRMSNormReduceFp32
     */
    void createSpatialRMSNormReduceFp32();

    /**
     * @brief Create SpatialRMSNormApplyFp32
     */
    void createSpatialRMSNormApplyFp32();

    /**
     * @brief Create scaleDotProductAttentionFp32
     */
    void createScaleDotProductAttentionFp32(int qHeadDim, int vHeadDim);

    /**
     * @brief Create scaleDotProductAttentionNaiveFp32
     */
    void createScaleDotProductAttentionNaiveFp32(int qHeadDim, int vHeadDim);

    /**
     * @brief Create TransfomerSwiGLUFp32
     */
    void createTransformerSwiGLUFp32();

    /**
     * @brief Create Transformer RoPEFP32
     */
    void createTransformerRoPEFp32();
  };
}



#endif
