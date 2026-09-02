#ifndef COMMON_GLSL_H
#define COMMON_GLSL_H
/**
* @author dhkim92.dev@gmail.com
* @brief common.glsl defined parameters
*/

#define SPECIALIZATION_CONSTANT(__id, __type, __name, __default) layout(constant_id = __id) const __type __name = __default
#define SPEC(__id, __type, __name, __default) layout(constant_id = __id) const __type __name = __default

#define GroupId0() (int(gl_WorkGroupID.x))
#define GroupId1() (int(gl_WorkGroupID.y))
#define GroupId2() (int(gl_WorkGroupID.z))
#define LocalId0() (int(gl_LocalInvocationID.x))
#define LocalId1() (int(gl_LocalInvocationID.y))
#define LocalId2() (int(gl_LocalInvocationID.z))
#define LocalSize0() (int(gl_WorkGroupSize.x))
#define LocalSize1() (int(gl_WorkGroupSize.y))
#define LocalSize2() (int(gl_WorkGroupSize.z))
#define GlobalId0() (GroupId0() * LocalSize0() + LocalId0())
#define GlobalId1() (GroupId1() * LocalSize1() + LocalId1())
#define GlobalId2() (GroupId2() * LocalSize2() + LocalId2())

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
 * half float precision type definition
 */
 #ifndef PRECISION
  #define PRECISION 32
#endif

#if PRECISION == 16
  #extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
  #define real float16_t
  #define real4 f16vec4
  #define ZERO 0.0hf
  #define ONE 1.0hf
  #define HUNDRED 100.0hf
  #define FOURTEEN 14.0hf
  #define TEN 10.0hf
  #define EIGHT 8.0hf
  #define FIVE 5.0hf
  #define FOUR 4.0hf
  #define TWO 2.0hf
  #define HALF 0.5hf
  #define TWOP5 2.5hf
  #define SQRT8 2.82842712475hf
  #define SQRT2 1.41421356237hf
  #define SQRTHALF 0.70710678118hf
  #define SQRTEIGHTH 0.35355339059hf
  #define LOG1PEXPTHRESHOLD 20.0hf
  #define floatToReal(_r) float16_t(_r)
#elif PRECISION == 32 
  #define real float
  #define real4 vec4
  #define ZERO 0.0f
  #define ONE 1.0f
  #define HUNDRED 100.0f
  #define FOURTEEN 14.0f
  #define TEN 10.0f
  #define EIGHT 8.0f
  #define FIVE 5.0f
  #define FOUR 4.0f
  #define TWO 2.0f
  #define HALF 0.5f
  #define TWOP5 2.5f
  #define SQRT8 2.82842712475f
  #define SQRT2 1.41421356237f
  #define SQRTHALF 0.70710678118f
  #define SQRTEIGHTH 0.35355339059f
  #define LOG1PEXPTHRESHOLD 20.0f
  #define floatToReal(_r) float(_r)
#endif

#ifndef PRECISION_STORAGE
  #define PRECISION_STORAGE 32
#endif

#define ACTIVATION_TYPE_IDENTITY 0
#define ACTIVATION_TYPE_RELU 1
#define ACTIVATION_TYPE_MISH 2
#define ACTIVATION_TYPE_MISH_SCALE8 12
#define ACTIVATION_TYPE_SILU 3

#if PRECISION_STORAGE == 16
  #extension GL_EXT_shader_16bit_storage : enable
  #if PRECISION != 16
    #extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
  #endif
  #define realstore float16_t
  #define realstore4 f16vec4
  #if PRECISION == 16
    #define LOAD(__buf,__x) ((__buf)[(__x)])
    #define STORE(__buf,__x,__y) ((__buf)[(__x)] = (__y))
  #elif PRECISION == 32
    #define LOAD(__buf,__x) float((__buf)[(__x)])
    #define STORE(__buf,__x,__y) ((__buf)[(__x)] = float16_t(__y))
  #endif
#elif PRECISION_STORAGE == 32
  #define realstore float
  #define realstore4 vec4
  #if PRECISION == 16
    #define LOAD(__buf,__x) float16_t((__buf)[(__x)])
    #define STORE(__buf,__x,__y) ((__buf)[(__x)] = float(__y))
  #elif PRECISION == 32
    #define LOAD(__buf,__x) ((__buf)[(__x)])
    #define STORE(__buf,__x,__y) ((__buf)[(__x)] = (__y))
  #endif
#endif

// #define log1p(x) log(1.0 + exp(x))
#define log1p(x) log(1.0 + (x))
#define fmax(a, b) max(a, b)


real soft_plus(real x) {
  const float T = 20.0;
  if ( x > T ) {
    return x;
  } else if ( x < -T ) {
    return exp(x);
  } else {
    return log(exp(x) + ONE);
  }
}

#endif //
