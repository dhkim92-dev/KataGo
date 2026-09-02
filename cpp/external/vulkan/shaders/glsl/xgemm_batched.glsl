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
#define VWM 4 
#define VWN 4
#define STRM 0 
#define STRN 0
#define SA 1
#define SB 1
#define GEMMK 0
#define KWI 1
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

#include "common.glsl"

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
  realstore4 agm[];
};

layout(set = 0, binding = 1) readonly buffer MatB {
  realstore4 bgm[];
};

layout(set = 0, binding = 2) writeonly buffer MatC {
  realstore4 cgm[];
};

shared realstore4 alm[(_MWG * _KWG) / VWM];
shared realstore4 blm[(_NWG * _KWG) / VWN];

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