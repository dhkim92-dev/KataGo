#ifndef __COMMON_HLSL__
#define __COMMON_HLSL__
/**
* @author dhkim92.dev@gmail.com
* @brief common.hlsl defined parameters
*/

/**
* @brief BatchNorm NCHW compute shader dispatch parameters
*/
#define BN_DISPATCH_X 16
#define BN_DISPATCH_Y 16
#define BN_DISPATCH_Z 1

/**
* @brief Add Channel Bias NCHW compute shader dispatch parameters
*/
#define ADD_CHANNELS_DISPATCH_X 16
#define ADD_CHANNELS_DISPATCH_Y 16
#define ADD_CHANNELS_DISPATCH_Z 1

/**
* @brief Convolution parameters
*/
#define CONV_TILE_N 8
#define CONV_TILE_M 8
#define CONV_TILE_K 16
#define CONV_DISPATCH_X 8
#define CONV_DISPATCH_Y 8
#define CONV_DISPATCH_Z 1

/**
* Extract Channel0 NCHW compute shader dispatch parameters
* EXTRACT_CHANNEL0_XYSTRIDE: number of threads processing spatial elements in x dimension
* EXTRACT_CHANNEL0_NSTRIDE: number of threads processing batch elements in y dimension
*/
#define EXTRACT_CHANNEL0_XYSTRIDE 64
#define EXTRACT_CHANNEL0_NSTRIDE 1
#define EXTRACT_CHANNEL0_DISPATCH_X EXTRACT_CHANNEL0_XYSTRIDE
#define EXTRACT_CHANNEL0_DISPATCH_Y EXTRACT_CHANNEL0_NSTRIDE

/**
* @brief Add Point Wise NCHW compute shader dispatch parameters
*/
#define ADD_POINTWISE_DISPATCH_X 256
#define ADD_POINTWISE_DISPATCH_Y 1
#define ADD_POINTWISE_DISPATCH_Z 1

/**
* Matmul parameters
*/
#define MATMUL_TILE_M 16
#define MATMUL_TILE_N 16
#define MATMUL_TILE_K 8
#define MATMUL_DISPATCH_X MATMUL_TILE_M
#define MATMUL_DISPATCH_Y MATMUL_TILE_N

/**
* @brief Direct Conv2D kernel parameters
*/
#define CONV_2D_TILE_N 8
#define CONV_2D_TILE_M 8
#define CONV_2D_TILE_K 16
#define CONV_2D_DISPATCH_X 8
#define CONV_2D_DISPATCH_Y 8

/**
* Pooling Layer params
*/
#define POOLING_XYSTRIDE 64
#define POOLING_NSTRIDE 1
#define POOLING_DISPATCH_X POOLING_XYSTRIDE
#define POOLING_DISPATCH_Y POOLING_NSTRIDE

/**
 * Sum Channels params
 */
#define SUM_CHANNELS_XYSTRIDE 64
#define SUM_CHANNELS_DISPATCH_X SUM_CHANNELS_XYSTRIDE
#define SUM_CHANNELS_DISPATCH_Y 1
#define SUM_CHANNELS_DISPATCH_Z 1

/**
 * Strided Batched Matmul params (for 1x1 conv)
 */
#define SBM_TILE_M 16
#define SBM_TILE_N 16
#define SBM_TILE_K 16
#define SBM_DISPATCH_X SBM_TILE_M
#define SBM_DISPATCH_Y SBM_TILE_N

#endif // __COMMON_HLSL__