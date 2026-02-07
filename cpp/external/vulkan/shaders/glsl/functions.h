/**
* @author dhkim92.dev@gmail.com
* @brief macro definitions
*/
#ifndef FUNCTIONS_GLSL_H
#define FUNCTIONS_GLSL_H
#define ZERO 0.0
#define ONE 1.0
#define HUNDRED 100.0
#define FOURTEEN 14.0
#define TEN 10.0
#define EIGHT 8.0
#define FIVE 5.0
#define FOUR 4.0
#define TWO 2.0
#define HALF 0.5
#define TWOP5 2.5
#define SQRT8 2.82842712475
#define SQRT2 1.41421356237
#define SQRTHALF 0.70710678118
#define SQRTEIGHTH 0.35355339059
#define LOG1PEXPTHRESHOLD 20.0
#define RELU(x) max(0.0, x)
// MISH with numerical stability: for large x, tanh(softplus(x)) ≈ 1, so mish(x) ≈ x
// This matches OpenCL's implementation which uses log1p(exp(x)) with threshold check
#define MISH(x) ((x) < LOG1PEXPTHRESHOLD ? (x) * tanh(log(1.0 + exp(x))) : (x))
// MISH_SCALE8: threshold is LOG1PEXPTHRESHOLD * 0.125 = 2.5
#define MISH_SCALE8(x) ((x) < 2.5 ? (x) * tanh(log(1.0 + exp((x) * 8.0))) : (x))
#define IDENTITY(x) (x)
#endif 
