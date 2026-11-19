/**
 * NCHW Transformer projection: C[b,n,m] = sum_k A[b,k,m] * B[k,n].
 * NN-only specialization of the CLBlast-derived direct GEMM tiling.
 * Shared-memory and register arithmetic retain the selected real precision.
 */
layout(constant_id = 3) const int _WGD = 8;
layout(constant_id = 4) const int _MDIMCD = 16;
layout(constant_id = 5) const int _NDIMCD = 16;
layout(constant_id = 6) const int _MDIMAD = 16;
layout(constant_id = 7) const int _NDIMBD = 16;
layout(constant_id = 8) const int _KWID = 1;
layout(constant_id = 9) const int _PADA = 1;
layout(constant_id = 10) const int _PADB = 1;

#include "common.glsl"
#include "xgemm_direct_types.glsl"

#define WGD _WGD
#define MDIMCD _MDIMCD
#define NDIMCD _NDIMCD
#define MDIMAD _MDIMAD
#define NDIMBD _NDIMBD
#define KWID _KWID
#define PADA _PADA
#define PADB _PADB
#define MWID (WGD/MDIMCD)
#define NWID (WGD/NDIMCD)
#define KDIMAD ((MDIMCD*NDIMCD)/MDIMAD)
#define KDIMBD ((MDIMCD*NDIMCD)/NDIMBD)
#define MWAD (WGD/MDIMAD)
#define KWAD (WGD/KDIMAD)
#define NWBD (WGD/NDIMBD)
#define KWBD (WGD/KDIMBD)

layout(set = 0, binding = 0) readonly buffer Agm {
  realstoreMD agm[];
};
layout(set = 0, binding = 1) readonly buffer Bgm {
  realstoreND bgm[];
};
layout(set = 0, binding = 2) writeonly buffer Cgm {
  realstore cgm[];
};

shared real alm[WGD * (WGD + PADA)];
shared real blm[WGD * (WGD + PADB)];

// Preserve the strided NN push ABI; this kernel always stores NCHW (c_transpose=0).
layout(push_constant) uniform PushConstants {
  int kSizeM;
  int kSizeN;
  int kSizeK;
  int a_ld;
  int a_stride;
  int b_ld;
  int b_stride;
  int c_ld;
  int c_stride;
  int c_transpose;
};

void loadProjectionA(const int offset, const int kwg, const bool fullTile) {
  int la0, la1;
  if (MDIMCD == MDIMAD) {
    la0 = LocalId0();
    la1 = LocalId1();
  } else {
    const int tid = LocalId0() + MDIMCD*LocalId1();
    la0 = tid % MDIMAD;
    la1 = tid / MDIMAD;
  }
  if (fullTile && a_ld % VWMD == 0 && offset % VWMD == 0) {
    #pragma unroll
    for (int mi = 0; mi < MWAD/VWMD; ++mi) {
      #pragma unroll
      for (int ki = 0; ki < KWAD; ++ki) {
        const int mg = mi + la0*(MWAD/VWMD);
        const int kg = ki + la1*KWAD;
        const int idm = mg + GroupId0()*(WGD/VWMD);
        const realMD value = LOADM(agm, (kg + kwg)*(a_ld/VWMD) + idm + offset/VWMD);
#if VWMD == 1
        alm[kg*(WGD + PADA) + mg] = value;
#elif VWMD == 2
        alm[kg*(WGD + PADA) + mg*VWMD + 0] = value.x;
        alm[kg*(WGD + PADA) + mg*VWMD + 1] = value.y;
#elif VWMD == 4
        alm[kg*(WGD + PADA) + mg*VWMD + 0] = value.x;
        alm[kg*(WGD + PADA) + mg*VWMD + 1] = value.y;
        alm[kg*(WGD + PADA) + mg*VWMD + 2] = value.z;
        alm[kg*(WGD + PADA) + mg*VWMD + 3] = value.w;
#endif
      }
    }
  } else {
    #pragma unroll
    for (int mi = 0; mi < MWAD; ++mi) {
      #pragma unroll
      for (int ki = 0; ki < KWAD; ++ki) {
        const int mg = mi + la0*MWAD;
        const int kg = ki + la1*KWAD;
        const int idm = mg + GroupId0()*WGD;
        const int idk = kg + kwg;
        real value = ZERO;
        if (fullTile || (idm < kSizeM && idk < kSizeK))
          value = LOADSINGLEM(agm, idk*a_ld + idm + offset);
        alm[kg*(WGD + PADA) + mg] = value;
      }
    }
  }
}

void loadProjectionB(const int offset, const int kwg, const bool fullTile) {
  int lb0, lb1;
  if (MDIMCD == NDIMBD) {
    lb0 = LocalId0();
    lb1 = LocalId1();
  } else {
    const int tid = LocalId0() + MDIMCD*LocalId1();
    lb0 = tid % NDIMBD;
    lb1 = tid / NDIMBD;
  }
  if (fullTile && b_ld % VWND == 0 && offset % VWND == 0) {
    #pragma unroll
    for (int ki = 0; ki < KWBD; ++ki) {
      #pragma unroll
      for (int ni = 0; ni < NWBD/VWND; ++ni) {
        const int ng = ni + lb0*(NWBD/VWND);
        const int kg = ki + lb1*KWBD;
        const int idn = ng + GroupId1()*(WGD/VWND);
        const realND value = LOADN(bgm, (kg + kwg)*(b_ld/VWND) + idn + offset/VWND);
#if VWND == 1
        blm[kg*(WGD + PADB) + ng] = value;
#elif VWND == 2
        blm[kg*(WGD + PADB) + ng*VWND + 0] = value.x;
        blm[kg*(WGD + PADB) + ng*VWND + 1] = value.y;
#elif VWND == 4
        blm[kg*(WGD + PADB) + ng*VWND + 0] = value.x;
        blm[kg*(WGD + PADB) + ng*VWND + 1] = value.y;
        blm[kg*(WGD + PADB) + ng*VWND + 2] = value.z;
        blm[kg*(WGD + PADB) + ng*VWND + 3] = value.w;
#endif
      }
    }
  } else {
    #pragma unroll
    for (int ki = 0; ki < KWBD; ++ki) {
      #pragma unroll
      for (int ni = 0; ni < NWBD; ++ni) {
        const int ng = ni + lb0*NWBD;
        const int kg = ki + lb1*KWBD;
        const int idn = ng + GroupId1()*WGD;
        const int idk = kg + kwg;
        real value = ZERO;
        if (fullTile || (idn < kSizeN && idk < kSizeK))
          value = LOADSINGLEN(bgm, idk*b_ld + idn + offset);
        blm[kg*(WGD + PADB) + ng] = value;
      }
    }
  }
}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  const int aOffset = a_stride*GroupId2();
  const int bOffset = b_stride*GroupId2();
  const int cOffset = c_stride*GroupId2();
  const int idm = LocalId0()*MWID + GroupId0()*WGD;
  const int idn = LocalId1()*NWID + GroupId1()*WGD;
  const bool fullTile = (idm < (kSizeM/WGD)*WGD) && (idn < (kSizeN/WGD)*WGD);
  real apd[MWID];
  real bpd[NWID];
  real cpd[NWID*MWID];
  for (int mi = 0; mi < MWID; ++mi) {
    #pragma unroll
    for (int ni = 0; ni < NWID; ++ni)
      cpd[ni*MWID + mi] = ZERO;
  }

  // Retain the direct kernel's separate complete/edge accumulation scopes for
  // FP16 code generation, including its scalar K remainder after each tile loop.
  if (fullTile) {
    int kwg = 0;
    for (; kwg < (kSizeK/WGD)*WGD; kwg += WGD) {
      loadProjectionA(aOffset, kwg, true);
      loadProjectionB(bOffset, kwg, true);
      barrier();
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int pit = 0; pit < KWID; ++pit) {
          const int kg = pwi + pit;
          #pragma unroll
          for (int mi = 0; mi < MWID; ++mi)
            apd[mi] = alm[kg*(WGD + PADA) + mi + LocalId0()*MWID];
          #pragma unroll
          for (int ni = 0; ni < NWID; ++ni)
            bpd[ni] = blm[kg*(WGD + PADB) + ni + LocalId1()*NWID];
          #pragma unroll
          for (int ni = 0; ni < NWID; ++ni) {
            #pragma unroll
            for (int mi = 0; mi < MWID; ++mi)
              cpd[ni*MWID + mi] += apd[mi]*bpd[ni];
          }
        }
      }
      barrier();
    }

    for (; kwg < kSizeK; ++kwg) {
      #pragma unroll
      for (int mi = 0; mi < MWID; ++mi)
        apd[mi] = LOADSINGLEM(agm, kwg*a_ld + idm + mi + aOffset);
      #pragma unroll
      for (int ni = 0; ni < NWID; ++ni)
        bpd[ni] = LOADSINGLEN(bgm, kwg*b_ld + idn + ni + bOffset);
      #pragma unroll
      for (int ni = 0; ni < NWID; ++ni) {
        #pragma unroll
        for (int mi = 0; mi < MWID; ++mi)
          cpd[ni*MWID + mi] += apd[mi]*bpd[ni];
      }
    }

    #pragma unroll
    for (int ni = 0; ni < NWID; ++ni) {
      #pragma unroll
      for (int mi = 0; mi < MWID; ++mi)
        cgm[(idn + ni)*c_ld + idm + mi + cOffset] = realstore(ONE*cpd[ni*MWID + mi]);
    }
  } else {
    int kwg = 0;
    for (; kwg < (kSizeK/WGD)*WGD; kwg += WGD) {
      loadProjectionA(aOffset, kwg, false);
      loadProjectionB(bOffset, kwg, false);
      barrier();
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int pit = 0; pit < KWID; ++pit) {
          const int kg = pwi + pit;
          #pragma unroll
          for (int mi = 0; mi < MWID; ++mi)
            apd[mi] = alm[kg*(WGD + PADA) + mi + LocalId0()*MWID];
          #pragma unroll
          for (int ni = 0; ni < NWID; ++ni)
            bpd[ni] = blm[kg*(WGD + PADB) + ni + LocalId1()*NWID];
          #pragma unroll
          for (int ni = 0; ni < NWID; ++ni) {
            #pragma unroll
            for (int mi = 0; mi < MWID; ++mi)
              cpd[ni*MWID + mi] += apd[mi]*bpd[ni];
          }
        }
      }
      barrier();
    }

    for (; kwg < kSizeK; ++kwg) {
      #pragma unroll
      for (int mi = 0; mi < MWID; ++mi) {
        apd[mi] = ZERO;
        if (idm + mi < kSizeM)
          apd[mi] = LOADSINGLEM(agm, kwg*a_ld + idm + mi + aOffset);
      }
      #pragma unroll
      for (int ni = 0; ni < NWID; ++ni) {
        bpd[ni] = ZERO;
        if (idn + ni < kSizeN)
          bpd[ni] = LOADSINGLEN(bgm, kwg*b_ld + idn + ni + bOffset);
      }
      #pragma unroll
      for (int ni = 0; ni < NWID; ++ni) {
        #pragma unroll
        for (int mi = 0; mi < MWID; ++mi)
          cpd[ni*MWID + mi] += apd[mi]*bpd[ni];
      }
    }

    #pragma unroll
    for (int ni = 0; ni < NWID; ++ni) {
      #pragma unroll
      for (int mi = 0; mi < MWID; ++mi) {
        if (idm + mi < kSizeM && idn + ni < kSizeN)
          cgm[(idn + ni)*c_ld + idm + mi + cOffset] = realstore(ONE*cpd[ni*MWID + mi]);
      }
    }
  }
}
