/**
* @author dhkim92.dev@gmail.com
* @brief activation.hlsl macro activation functions
*/
#ifndef __FUNCTIONS_HLSL__
#define __FUNCTIONS_HLSL__
#define RELU(x) max(0.0f, x)
#define MISH(x) (x * tanh(log(1.0f + exp(x))))
#define MISH_SCALE8(x) ((x) < 2.5f ? (x) * tanh(log(1.0f + exp((x) * 8.0f))) : (x))
#define IDENTITY(x) (x)
#endif // __FUNCTIONS_HLSL__
