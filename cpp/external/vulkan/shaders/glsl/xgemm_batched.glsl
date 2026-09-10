/**
* @author dhkim92.dev@gmail.com
* @brief Batched XGEMM Shader
* @details This shader performs batched matrix multiplication C = A * B for multiple batches.
* This file based on cpp/external/clblas/xgemm_batched.opencl 
* And translate version of CLBlas code. 
* Apache License Version 2.0
*/
// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)
// Or as an image (assuming column-major)
//       K
//    o-------o
//    |       |
//  N | [B^T] |
//    |       |
//    o-------o
//        K               N
//    o-------o        o-----o
//  M |  [A]  |      M | [C] |
//    |       |        |     |
//    o-------o        o-----o
#ifndef VWM
#define VWM 4
#endif
#ifndef VWN
#define VWN 4
#endif
#define STRM 0 
#define STRN 0
#define SA 1
#define SB 1
#define GEMMK 0
// layout(constant_id = 0) const uint LOCAL_SIZE_X = 8;
// layout(constant_id = 1) const uint LOCAL_SIZE_Y = 8;
// layout(constant_id = 2) const uint LOCAL_SIZE_Z = 1;
layout(constant_id = 3) const int _MWG = 64; // 64, 128
layout(constant_id = 4) const int _NWG = 64; ///
layout(constant_id = 5) const int _KWG = 32; // 8, 16
layout(constant_id = 6) const int _MDIMC = 16; //
layout(constant_id = 7) const int _NDIMC = 16; //
layout(constant_id = 8) const int _MDIMA = 16; // Re-shaped tile dimension of matrix A = KDIMA x MDIMA
layout(constant_id = 9) const int _NDIMB = 16; // Re-shaped tile dimension of matrix B =
layout(constant_id = 10) const int _KWI = 1; // Unroll factor for the K dimension

#include "common.glsl"

#if PRECISION == 16
  #define real2 f16vec2
#else
  #define real2 vec2
#endif
#if VWM == 1
  #define realM real
  #define realstoreM realstore
#elif VWM == 2
  #define realM real2
  #define realstoreM realstore2
#elif VWM == 4
  #define realM real4
  #define realstoreM realstore4
#endif
#if VWN == 1
  #define realN real
  #define realstoreN realstore
#elif VWN == 2
  #define realN real2
  #define realstoreN realstore2
#elif VWN == 4
  #define realN real4
  #define realstoreN realstore4
#endif

layout(push_constant) uniform BatchedXGEMMParams {
    int kSizeM;
    int kSizeN;
    int kSizeK;
    int a_one;
    int a_two;
    int b_one;
    int b_two;
    int c_one;
    int c_two;
};

layout(set = 0, binding = 0) readonly buffer MatA {
  realstoreM agm[];
};

layout(set = 0, binding = 1) readonly buffer MatB {
  realstoreN bgm[];
};

layout(set = 0, binding = 2) writeonly buffer MatC {
  realstoreM cgm[];
};

shared realstoreM alm[(_MWG * _KWG) / VWM];
shared realstoreN blm[(_NWG * _KWG) / VWN];

#define STRM 0
#define STRN 0

#include "xgemm.glsl"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
// layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;
void main() {
  const int batch = GroupId2();

  // Sets the offsets
  const int a_offset = batch * a_one * a_two;
  const int b_offset = batch * b_one * b_two;
  const int c_offset = batch * c_one * c_two;

  XgemmBody(kSizeM, kSizeN, kSizeK, a_offset, b_offset, c_offset);
}
