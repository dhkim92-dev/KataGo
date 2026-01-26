/**
* @author dhkim92.dev@gmail.com
* @brief activation.hlsl macro activation functions
*/
#ifndef __FUNCTIONS_HLSL__
#define __FUNCTIONS_HLSL__
#define RELU(x) max(0.0f, x)
#define MISH(x) (x * tanh(log(1.0f + exp(x))))
#define IDENTITY(x) (x)
#endif // __FUNCTIONS_HLSL__
