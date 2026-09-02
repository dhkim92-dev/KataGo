
layout(constant_id = 3) const int _WGD = 8;
layout(constant_id = 4) const int _MDIMCD = 16;
layout(constant_id = 5) const int _NDIMCD = 16;
layout(constant_id = 6) const int _MDIMAD = 16;
layout(constant_id = 7) const int _NDIMBD = 16;
layout(constant_id = 8) const int _KWID = 1;
layout(constant_id = 9) const int _PADA = 1;
layout(constant_id = 10) const int _PADB = 1;


#include "common.glsl"

layout(set = 0, binding = 0) readonly buffer Agm {
    realstore4 agm[];
};

layout(set = 0, binding = 1) readonly buffer Bgm {
    realstore4 bgm[];
};

layout(set = 0, binding = 2) writeonly buffer Cgm {
    realstore4 cgm[];
};

shared realstore alm[_WGD * (_WGD + _PADA)];
shared realstore blm[_WGD * (_WGD + _PADB)];

layout(push_constant) uniform PushConsts {
  int kSizeM;
  int kSizeN;
  int kSizeK;
  int a_ld;
  int b_ld;
  int c_ld;
  int c_transpose;
};

#include "xgemm_direct.glsl"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  const int batch = GroupId2();
  const real arg_alpha = ONE;
  const real arg_beta = ZERO;
  const int a_offset = batch * kSizeM * kSizeK;
  const int b_offset = batch * kSizeN * kSizeK;
  const int c_offset = batch * kSizeM * kSizeN;
  const int a_conjugate = 0;
  const int b_conjugate = 0;
  XgemmDirect(kSizeM, kSizeN, kSizeK, 
              arg_alpha, arg_beta,
              a_offset, a_ld, 
              b_offset, b_ld,
              c_offset, c_ld,
              1, 1, c_transpose, a_conjugate, b_conjugate);

}
