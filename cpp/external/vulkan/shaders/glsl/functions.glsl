/**
* @author dhkim92.dev@gmail.com
* @brief macro definitions
*/


#define RELU(x) max(0.0, x)
// MISH with numerical stability: for large x, tanh(softplus(x)) ≈ 1, so mish(x) ≈ x
// This matches OpenCL's implementation which uses log1p(exp(x)) with threshold check
#define MISH(x) ((x) < LOG1PEXPTHRESHOLD ? (x) * tanh(log(1.0 + exp(x))) : (x))
// MISH_SCALE8: threshold is LOG1PEXPTHRESHOLD * 0.125 = 2.5
#define MISH_SCALE8(x) ((x) < 2.5 ? (x) * tanh(log(1.0 + exp((x) * 8.0))) : (x))
#define IDENTITY(x) (x)
