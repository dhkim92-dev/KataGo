/**
* @author dhkim92.dev@gmail.com
* @brief activation.hlsl macro activation functions
*/
#ifndef __FUNCTIONS_HLSL__
#define __FUNCTIONS_HLSL__
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
#define RELU(x) max(0.0f, x)
// MISH with numerical stability: for large x, tanh(softplus(x)) ≈ 1, so mish(x) ≈ x
// This matches OpenCL's implementation which uses log1p(exp(x)) with threshold check
#define MISH(x) ((x) < LOG1PEXPTHRESHOLD ? (x) * tanh(log(1.0f + exp(x))) : (x))
// MISH_SCALE8: threshold is LOG1PEXPTHRESHOLD * 0.125 = 2.5
#define MISH_SCALE8(x) ((x) < 2.5f ? (x) * tanh(log(1.0f + exp((x) * 8.0f))) : (x))
#define IDENTITY(x) (x)
#endif // __FUNCTIONS_HLSL__
