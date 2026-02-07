#ifndef COMMON_GLSL_H
#define COMMON_GLSL_H
/**
* @author dhkim92.dev@gmail.com
* @brief common.glsl defined parameters
*/

#define SPECIALIZATION_CONSTANT(__id, __type, __name, __default) layout(constant_id = __id) const __type __name = __default

#define GroupId0() (int(gl_WorkGroupID.x))
#define GroupId1() (int(gl_WorkGroupID.y))
#define GroupId2() (int(gl_WorkGroupID.z))
#define LocalId0() (int(gl_LocalInvocationID.x))
#define LocalId1() (int(gl_LocalInvocationID.y))
#define LocalId2() (int(gl_LocalInvocationID.z))
#define LocalSize0() (int(gl_WorkGroupSize.x))
#define LocalSize1() (int(gl_WorkGroupSize.y))
#define LocalSize2() (int(gl_WorkGroupSize.z))
#define GlobalId0() (GroupId0() * LocalSize0 + LocalId0())
#define GlobalId1() (GroupId1() * LocalSize1 + LocalId1())
#define GlobalId2() (GroupId2() * LocalSize2 + LocalId2())

/**
* @brief BatchNorm NCHW compute shader dispatch parameters
* Thread mapping optimized for memory coalescing in NCHW layout:
*   x -> spatial (contiguous in memory)
*   y -> channel
*   z -> batch
* This ensures threads in a warp access contiguous memory locations.
*/
#define BN_DISPATCH_X 32
#define BN_DISPATCH_Y 8
#define BN_DISPATCH_Z 1

/**
* @brief Add Channel Bias NC compute shader dispatch parameters
* Uses 1D dispatch for NC tensors (no spatial dimension)
* Total elements = N * C, processed as flat array for better GPU utilization
*/
#define ADD_CHANNEL_BIAS_NC_THREADS 256
#define ADD_CHANNEL_BIAS_NC_DISPATCH_X ADD_CHANNEL_BIAS_NC_THREADS
#define ADD_CHANNEL_BIAS_NC_DISPATCH_Y 1
#define ADD_CHANNEL_BIAS_NC_DISPATCH_Z 1

/**
* @brief Add Channel Bias NCHW compute shader dispatch parameters
* Uses 2D dispatch: x=spatial (xy), y=batch*channel (nc)
* Both dimensions are typically large enough for good GPU utilization
*/
#define ADD_CHANNELS_DISPATCH_X 16
#define ADD_CHANNELS_DISPATCH_Y 16
#define ADD_CHANNELS_DISPATCH_Z 1

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
* Matmul parameters - Optimized with register tiling
* 
* Each thread computes MATMUL_WORK_M x MATMUL_WORK_N output elements.
* Workgroup has MATMUL_THREAD_M x MATMUL_THREAD_N threads.
* Therefore workgroup tile size:
*   MATMUL_TILE_M = MATMUL_THREAD_M * MATMUL_WORK_M
*   MATMUL_TILE_N = MATMUL_THREAD_N * MATMUL_WORK_N
* 
* This reduces global memory traffic by reusing values in registers.
* Total threads per workgroup: 16 * 16 = 256
* Work per thread: 4 * 4 = 16 output elements
* Workgroup output: 64 * 64 = 4096 elements (16x improvement over naive)
*/
#define MATMUL_THREAD_M 16   // Threads in M dimension
#define MATMUL_THREAD_N 16   // Threads in N dimension
#define MATMUL_WORK_M 4      // Work per thread in M dimension
#define MATMUL_WORK_N 4      // Work per thread in N dimension
#define MATMUL_TILE_M (MATMUL_THREAD_M * MATMUL_WORK_M)  // 64
#define MATMUL_TILE_N (MATMUL_THREAD_N * MATMUL_WORK_N)  // 64
#define MATMUL_TILE_K 16     // K tile size for shared memory blocking
#define MATMUL_SMEM_PAD 1    // Shared memory padding to avoid bank conflicts
#define MATMUL_DISPATCH_X MATMUL_THREAD_M
#define MATMUL_DISPATCH_Y MATMUL_THREAD_N


/**
* @brief Direct Conv2D kernel parameters
* TILE_X: number of output X positions per workgroup
* TILE_Y: number of output Y positions per workgroup
* TILE_OC: number of output channels per workgroup
* TILE_K: block size along K dimension (ic * fh * fw) for loop tiling
*/
#define CONV_2D_TILE_X 16
#define CONV_2D_TILE_Y 8
#define CONV_2D_TILE_OC 8
#define CONV_2D_TILE_K 16
#define CONV_2D_DISPATCH_X CONV_2D_TILE_X
#define CONV_2D_DISPATCH_Y CONV_2D_TILE_OC

/**
* @brief Fused BN->Act->Conv2D kernel parameters
* TILE_X: number of output X positions per workgroup
* TILE_Y: number of output Y positions per workgroup (each thread processes TILE_Y outputs)
* TILE_OC: number of output channels per workgroup
* TILE_K: block size along K dimension (ic * fh * fw) for loop tiling
*/
#define CONV_BNACT_TILE_X 16
#define CONV_BNACT_TILE_Y 8
#define CONV_BNACT_TILE_OC 16
#define CONV_BNACT_TILE_K 16
#define CONV_BNACT_MAX_FHFW 9  // Maximum filter size supported (3x3)
#define CONV_BNACT_DISPATCH_X CONV_BNACT_TILE_X
#define CONV_BNACT_DISPATCH_Y CONV_BNACT_TILE_OC
/**
* Pooling Layer params
* POOLING_XYSTRIDE: number of threads for parallel reduction over spatial dimension
* Increased from 64 to 128 for better GPU occupancy while maintaining power-of-2 for reduction
*/
#define POOLING_XYSTRIDE 128
#define POOLING_NSTRIDE 1
#define POOLING_DISPATCH_X POOLING_XYSTRIDE
#define POOLING_DISPATCH_Y POOLING_NSTRIDE

/**
 * Sum Channels params
 * SUM_CHANNELS_XYSTRIDE: number of threads for parallel reduction over spatial dimension
 * Increased from 64 to 128 for better GPU occupancy while maintaining power-of-2 for reduction
 */
#define SUM_CHANNELS_XYSTRIDE 128
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

#endif //