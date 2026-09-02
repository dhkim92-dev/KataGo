/**
* @author: dhkim92.dev
* This code is based on clblast project. and translate version for GLSL compute shader(vulkan)
* Ignore complex number.
* Apache License
*/
#include "common.glsl"

#ifndef WGD 
#define WGD _WGD 
#endif 

#ifndef MDIMCD
#define MDIMCD _MDIMCD
#endif

#ifndef NDIMCD
#define NDIMCD _NDIMCD
#endif

#ifndef MDIMAD
#define MDIMAD _MDIMAD
#endif  

#ifndef NDIMBD
#define NDIMBD _NDIMBD
#endif

#ifndef KWID
#define KWID _KWID
#endif

#ifndef VWMD
#define VWMD 4
#endif

#ifndef VWND 
#define VWND 4
#endif

#ifndef PADA
#define PADA _PADA
#endif 

#ifndef PADB
#define PADB _PADB
#endif


// Helper parameters based on the above tuning parameters
#define MWID (WGD/MDIMCD)                // Work per work-item (M-dimension)
#define NWID (WGD/NDIMCD)                // Work per work-item (N-dimension)
#define KDIMAD ((MDIMCD*NDIMCD)/(MDIMAD)) // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#define KDIMBD ((MDIMCD*NDIMCD)/(NDIMBD)) // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#define MWAD (WGD/MDIMAD)                // Amount of loads-per-thread for matrix A (M-dimension)
#define KWAD (WGD/KDIMAD)                // Amount of loads-per-thread for matrix A (K-dimension)
#define KWBD (WGD/KDIMBD)                // Amount of loads-per-thread for matrix B (K-dimension)
#define NWBD (WGD/NDIMBD)                // Amount of loads-per-thread for matrix B (N-dimension)

#if PRECISION_STORAGE == 16 && PRECISION == 32
    #define LOADGLOBAL(__buf,__x) real((__buf)[(__x)])
    #define LOADLOCAL(__buf,__x) real((__buf)[(__x)])
    #define STOREGLOBAL(__buf,__x,__val) ((__buf)[(__x)] = realstore(__val))
    #define STORELOCAL(__buf,__x,__val) ((__buf)[(__x)] = realstore(__val))
    #define SetToZeroStore(a) (a) = realstore(ZERO)
#else
    #define LOADGLOBAL(__buf,__x) ((__buf)[(__x)])
    #define LOADLOCAL(__buf,__x) ((__buf)[(__x)])
    #define STOREGLOBAL(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #define STORELOCAL(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #define SetToZeroStore(a) SetToZero(a)
#endif

#define SetToZero(a) a = ZERO

// Load a single element from global memory, __x means index of element that in real array, not index of real4 vector. This is used for loading the bias vector in the post-processing stage.
// __buf is vectorized buffer, only real4
#if PRECISION_STORAGE == 16 && PRECISION == 32
    #define LOADSINGLEGLOBAL(__buf,__x) real((__buf)[(__x)>>2][(__x)&0x3])
    #define LOADSINGLELOCAL(__buf,__x) real((__buf)[(__x)>>2][(__x)&0x3])
    #define STORESINGLEGLOBAL(__buf,__x,__val) ((__buf)[(__x)>>2][(__x)&0x3] = realstore(__val))
#else
    #define LOADSINGLEGLOBAL(__buf,__x) ((__buf)[(__x)>>2][(__x)&0x3])
    #define LOADSINGLELOCAL(__buf,__x) ((__buf)[(__x)>>2][(__x)&0x3])
    #define STORESINGLEGLOBAL(__buf,__x,__val) ((__buf)[(__x)>>2][(__x)&0x3] = (__val))
#endif

#define realMD real4
#define realstoreMD realstore4
#define realND real4
#define realstoreND realstore4

#if PRECISION_STORAGE == 16 && PRECISION == 32
    #define LOADGLOBALM(__buf,__x) real4((__buf)[(__x)])
    #define LOADLOCALM(__buf,__x) real4((__buf)[(__x)])
    #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = realstore4(__val))
#else
    #define LOADGLOBALM(__buf,__x) ((__buf)[(__x)])
    #define LOADLOCALM(__buf,__x) ((__buf)[(__x)])
    #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
#endif

#define realND real4
#define realstoreND realstore4

#if PRECISION_STORAGE == 16 && PRECISION == 32
    #define LOADGLOBALN(__buf,__x) real4((__buf)[(__x)])
    #define LOADLOCALN(__buf,__x) real4((__buf)[(__x)])
    #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = realstore4(__val))
#else
  #define LOADGLOBALN(__buf,__x) ((__buf)[(__x)])
  #define LOADLOCALN(__buf,__x) ((__buf)[(__x)])
  #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
#endif

#define Multiply(c, a, b) (c) = (a) * (b)
#define MultiplyAdd(c, a, b) (c) += (a) * (b)

real GlobalToPrivateDirectA(
    const int _mi,
    const int a_ld, const int a_offset, const int idm, const int idk,
    const int a_transpose,
    const int a_conjugate
) {
  // in opencl, agms means global memory for matrix A(not vectorized buffer)
  // in opengl, we consider it as an array of real4 vector. 
  const int a_index = (a_transpose == 1) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi);
  // real result = LOADGLOBAL(agms,a_index + a_offset);
  real result = LOADSINGLEGLOBAL(agm,a_index + a_offset);
  return result;
}

// Same as above, but now for the B input matrix
real GlobalToPrivateDirectB(
  const int _ni,
  const int b_ld, const int b_offset, const int idn, const int idk,
  const int b_transpose,
  const int b_conjugate
) {
  const int b_index = (b_transpose == 1) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni);
  real result = LOADSINGLEGLOBAL(bgm,b_index + b_offset);
  return result;
}

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix. This is the same as above but now includes a bounds check.
real GlobalToPrivateCheckedA(
  const int _mi,
  const int a_ld, const int a_offset, const int idm, const int idk,
  const int a_transpose,
  const int a_conjugate,
  const int kSizeM
)  {
  real result;
  if (idm + _mi < kSizeM) {
    const int a_index = (a_transpose == 1) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi);
    result = LOADSINGLEGLOBAL(agm,a_index + a_offset);
  }
  else {
    SetToZero(result);
  }
  return result;
}

// Same as above, but now for the B input matrix
real GlobalToPrivateCheckedB(
  const int _ni,
  const int b_ld, const int b_offset, const int idn, const int idk,
  const int b_transpose,
  const int b_conjugate,
  const int kSizeN) {
  real result;
  if (idn + _ni < kSizeN) {
    const int b_index = (b_transpose == 1) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni);
    result = LOADSINGLEGLOBAL(bgm,b_index + b_offset);
  }
  else {
    SetToZero(result);
  }
  return result;
}

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
real LocalToPrivateDirectA(const int _mi, const int kg, const int a_transpose) {
  const int mg = _mi + LocalId0()*MWID;
  const int index = (a_transpose == 1) ? mg*(WGD + PADA) + kg : kg*(WGD + PADA) + mg;
  return LOADLOCAL(alm,index);
}

// Same as above, but now for the B input matrix
real LocalToPrivateDirectB(const int _ni, const int kg, const int b_transpose) {
  const int ng = _ni + LocalId1()*NWID;
  const int index = (b_transpose == 1) ? ng*(WGD + PADB) + kg : kg*(WGD + PADB) + ng;
  return LOADLOCAL(blm,index);
}

void StoreResultsDirect(
  const real c_value,
  const int _mi, const int _ni, const int idm, const int idn,
  const real alpha, const real beta,
  const int c_ld, const int c_offset, const int c_transpose
) {
  // Determines the destination index
  int c_index = (c_transpose == 1) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);
  // The final multiplication with alpha (in case beta == 0)
  real result;
  // if (IsZero(beta)) {
    Multiply(result, alpha, c_value);
  // }
  // The final multiplication with alpha and the addition with beta*C
  // else {
    // real cloaded = LOADGLOBAL(cgm,c_index + c_offset);
    // AXPBY(result, alpha, c_value, beta, cloaded);
  // }
  // STORESINGLEGLOBAL(cgm, c_index + c_offset, result);
  STORESINGLEGLOBAL(cgm, c_index + c_offset, result);
  // cgm[(c_index + c_offset)/4][(c_index + c_offset) % 4] = result;
}

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
void StoreResultsChecked(
  const real c_value,
  const int _mi, const int _ni, const int idm, const int idn,
  const int kSizeM, const int kSizeN,
  const real alpha, const real beta,
  const int c_ld, const int c_offset, const int c_transpose
) {
  if ((idm + _mi) < kSizeM && (idn + _ni) < kSizeN) {

    // Deter_mines the destination index
    int c_index = (c_transpose == 1) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);

    // The final multiplication with alpha (in case beta == 0)
    real result;
    // if (IsZero(beta)) {
      Multiply(result, alpha, c_value);
    // }
    // The final multiplication with alpha and the addition with beta*C
    // else {
      // real cloaded = LOADGLOBAL(cgm,c_index + c_offset);
      // AXPBY(result, alpha, c_value, beta, cloaded);
    // }
    // STOREGLOBAL(cgm,c_index + c_offset,result);
    STORESINGLEGLOBAL(cgm, c_index + c_offset, result);
  }
}

void GlobalToLocalDirectA(const int a_ld, const int a_offset, const int kwg,
                          const int a_transpose, const int a_conjugate) {
  int la0, la1, tid;
  if (MDIMCD == MDIMAD) {
    la0 = LocalId0();
    la1 = LocalId1();
  } else {
    tid = LocalId0() + MDIMCD*LocalId1();
    la0 = tid % MDIMAD;
    la1 = tid / MDIMAD;
  }
  #pragma unroll
  for (int _mia = 0; _mia < MWAD/VWMD; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWAD; _kia += 1) {

      // Computes the indices for the global memory
      int mg = _mia + la0*(MWAD/VWMD);
      int kg = _kia + la1*KWAD;
      int idm = (a_transpose==1) ? mg + kwg/VWMD : mg + GroupId0()*(WGD/VWMD);
      int idk = (a_transpose==1) ? kg + GroupId0()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      const realMD avec = LOADGLOBALM(agm,idk*(a_ld/VWMD) + idm + (a_offset/VWMD));
        STORELOCAL(alm, kg*(WGD + PADA) + mg*VWMD + 0, avec.x);
        STORELOCAL(alm, kg*(WGD + PADA) + mg*VWMD + 1, avec.y);
        STORELOCAL(alm, kg*(WGD + PADA) + mg*VWMD + 2, avec.z);
        STORELOCAL(alm, kg*(WGD + PADA) + mg*VWMD + 3, avec.w);
    }
  }
}

// Same as above, but now for the B input matrix
void GlobalToLocalDirectB(
  const int b_ld, const int b_offset, const int kwg,
  const int b_transpose, const int b_conjugate
) {
  int lb0 = LocalId0();
  int lb1 = LocalId1();
  int tid = 0;
  if (MDIMCD == NDIMBD) {
    lb0 = LocalId0();
    lb1 = LocalId1();
  } else {
    tid = LocalId0() + MDIMCD*LocalId1();
    lb0 = tid % NDIMBD;
    lb1 = tid / NDIMBD;
  }
  #pragma unroll
  for (int _kib = 0; _kib < KWBD; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWBD/VWND; _nib += 1) {

      // Computes the indices for the global memory
      int ng = _nib + lb0*(NWBD/VWND);
      int kg = _kib + lb1*KWBD;
      int idn = (b_transpose==1) ? ng + kwg/VWND : ng + GroupId1()*(WGD/VWND);
      int idk = (b_transpose==1) ? kg + GroupId1()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      const realND bvec = LOADGLOBALN(bgm,idk*(b_ld/VWND) + idn + (b_offset/VWND));
      STORELOCAL(blm, kg*(WGD + PADB) + ng*VWND + 0, bvec.x);
      STORELOCAL(blm, kg*(WGD + PADB) + ng*VWND + 1, bvec.y);
      STORELOCAL(blm, kg*(WGD + PADB) + ng*VWND + 2, bvec.z);
      STORELOCAL(blm, kg*(WGD + PADB) + ng*VWND + 3, bvec.w);
    }
  }
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs doesn't
// use the vector data-types.
void GlobalToLocalScalarA(
  const int a_ld, const int a_offset, const int kwg,
  const int a_transpose, const int a_conjugate
) {
  int la0, la1, tid;
  if (MDIMCD == MDIMAD) {
    la0 = LocalId0();
    la1 = LocalId1();
  } else {
    tid = LocalId0() + MDIMCD*LocalId1();
    la0 = tid % MDIMAD;
    la1 = tid / MDIMAD;
  }
  #pragma unroll
  for (int _mia = 0; _mia < MWAD; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWAD; _kia += 1) {

      // Computes the indices for the global memory
      int mg = _mia + la0*MWAD;
      int kg = _kia + la1*KWAD;
      int idm = (a_transpose == 1) ? mg + kwg : mg + GroupId0()*WGD;
      int idk = (a_transpose == 1) ? kg + GroupId0()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      // real result = LOADGLOBAL(agms,idk*a_ld + idm + a_offset);
      real result = LOADSINGLEGLOBAL(agm,idk*a_ld + idm + a_offset);
      STORELOCAL(alm, kg*(WGD + PADA) + mg, result);
    }
  }
}

// Same as above, but now for the B input matrix
void GlobalToLocalScalarB(
  const int b_ld, const int b_offset, const int kwg,
  const int b_transpose, const int b_conjugate
) {
  int lb0, lb1, tid;
  if (MDIMCD == NDIMBD) {
    lb0 = LocalId0();
    lb1 = LocalId1();
  } else {
    tid = LocalId0() + MDIMCD*LocalId1();
    lb0 = tid % NDIMBD;
    lb1 = tid / NDIMBD;
  }
  #pragma unroll
  for (int _kib = 0; _kib < KWBD; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWBD; _nib += 1) {

      // Computes the indices for the global memory
      int ng = _nib + lb0*NWBD;
      int kg = _kib + lb1*KWBD;
      int idn = (b_transpose == 1) ? ng + kwg : ng + GroupId1()*WGD;
      int idk = (b_transpose == 1) ? kg + GroupId1()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      // real result = LOADGLOBAL(bgms,idk*b_ld + idn + b_offset);
      real result = LOADSINGLEGLOBAL(bgm,idk*b_ld + idn + b_offset);
      STORELOCAL(blm, kg*(WGD + PADB) + ng, result);
    }
  }
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs bounds
// checks and doesn't use the vector data-types.
void GlobalToLocalCheckedA(
  const int a_ld, const int a_offset, const int kwg,
  const int a_transpose, const int a_conjugate,
  const int kSizeM, const int kSizeK
) {
  int la0, la1, tid;

  if (MDIMCD == MDIMAD) {
    la0 = LocalId0();
    la1 = LocalId1();
  } else {
    tid = LocalId0() + MDIMCD*LocalId1();
    la0 = tid % MDIMAD;
    la1 = tid / MDIMAD;
  }
  #pragma unroll
  for (int _mia = 0; _mia < MWAD; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWAD; _kia += 1) {

      // Computes the indices for the global memory
      int mg = _mia + la0*MWAD;
      int kg = _kia + la1*KWAD;
      int idm = (a_transpose == 1) ? mg + kwg : mg + GroupId0()*WGD;
      int idk = (a_transpose == 1) ? kg + GroupId0()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      bool condition = (a_transpose==1) ? (idm < kSizeK) && (idk < kSizeM) :
                                      (idm < kSizeM) && (idk < kSizeK);
      if (condition) {
        // real result = LOADGLOBAL(agms,idk*a_ld + idm + a_offset);
        // if (a_conjugate) { COMPLEX_CONJUGATE(result); }
        real result = LOADSINGLEGLOBAL(agm,idk*a_ld + idm + a_offset);
        STORELOCAL(alm, kg*(WGD + PADA) + mg, result);
      }
      else {
        SetToZeroStore(alm[kg*(WGD + PADA) + mg]);
      }
    }
  }
}

// Same as above, but now for the B input matrix
void GlobalToLocalCheckedB(
  const int b_ld, const int b_offset, const int kwg,
  const int b_transpose, const int b_conjugate,
  const int kSizeN, const int kSizeK
) {
  int lb0, lb1, tid;
  if(MDIMCD == NDIMBD) {
    lb0 = LocalId0();
    lb1 = LocalId1();
  } else {
    tid = LocalId0() + MDIMCD*LocalId1();
    lb0 = tid % NDIMBD;
    lb1 = tid / NDIMBD;
  }
  #pragma unroll
  for (int _kib = 0; _kib < KWBD; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWBD; _nib += 1) {

      // Computes the indices for the global memory
      int ng = _nib + lb0*NWBD;
      int kg = _kib + lb1*KWBD;
      int idn = (b_transpose == 1) ? ng + kwg : ng + GroupId1()*WGD;
      int idk = (b_transpose == 1) ? kg + GroupId1()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      bool condition = (b_transpose == 1) ? (idn < kSizeK) && (idk < kSizeN) :
                                      (idn < kSizeN) && (idk < kSizeK);
      if (condition) {
        // real result = LOADGLOBAL(bgms,idk*b_ld + idn + b_offset);
        real result = LOADSINGLEGLOBAL(bgm,idk*b_ld + idn + b_offset);
        STORELOCAL(blm, kg*(WGD + PADB) + ng, result);
      }
      else {
        SetToZeroStore(blm[kg*(WGD + PADB) + ng]);
      }
    }
  }
}

void XgemmDirect(
  const int kSizeM, const int kSizeN, const int kSizeK,
  const real alpha,
  const real beta,
  const int a_offset, const int a_ld,
  const int b_offset, const int b_ld,
  const int c_offset, const int c_ld,
  const int a_transpose, const int b_transpose, const int c_transpose,
  const int a_conjugate, const int b_conjugate
) {
  // const real alpha = GetRealArg(arg_alpha);
  // const real beta = GetRealArg(arg_beta);

  // Extra pointers to scalar versions of global memory
  // const __global realstore* restrict agms = (const __global realstore* restrict) agm;
  // const __global realstore* restrict bgms = (const __global realstore* restrict) bgm;

  // Allocates workitem-private memory (registers)
  real apd[MWID];
  real bpd[NWID];
  real cpd[NWID * MWID];

  // Initializes the accumulation registers
  #pragma unroll
  for (int _mi = 0; _mi < MWID; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      SetToZero(cpd[_ni * MWID + _mi]);
    }
  }

  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of WGD by WGD.
  const int idm = LocalId0() * MWID + GroupId0() * WGD;
  const int idn = LocalId1() * NWID + GroupId1() * WGD;
  if ((idm < (kSizeM/WGD)*WGD) && (idn < (kSizeN/WGD)*WGD)) {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg += WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      if (a_ld % VWMD == 0 && a_offset % VWMD == 0) {
        GlobalToLocalDirectA(a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      else {
        GlobalToLocalScalarA(a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      if (b_ld % VWND == 0 && b_offset % VWND == 0) {
        GlobalToLocalDirectB(b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      else {
        GlobalToLocalScalarB(b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      barrier();

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;

          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LocalToPrivateDirectA(_mi, kg, a_transpose);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LocalToPrivateDirectB(_ni, kg, b_transpose);
          }

          // Performs the accumulation (Cpmd += Apmd * Bpmd)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier();
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GlobalToPrivateDirectA(_mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GlobalToPrivateDirectB(_ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate);
      }

      // Performs the accumulation (Cpmd += Apmd * Bpmd)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }

    // Stores a tile of results and performs the multiplication with alpha and beta
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        StoreResultsDirect(cpd[_ni * MWID + _mi], _mi, _ni, idm, idn,
                           alpha, beta, c_ld, c_offset, c_transpose);
      }
    }
  }

  // Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)
  else {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg+=WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      GlobalToLocalCheckedA(a_ld, a_offset, kwg, a_transpose, a_conjugate, kSizeM, kSizeK);
      GlobalToLocalCheckedB(b_ld, b_offset, kwg, b_transpose, b_conjugate, kSizeN, kSizeK);
      barrier();

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;

          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LocalToPrivateDirectA(_mi, kg, a_transpose);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LocalToPrivateDirectB(_ni, kg, b_transpose);
          }

          // Performs the accumulation (C += A * B)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier();
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GlobalToPrivateCheckedA(_mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate, kSizeM);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GlobalToPrivateCheckedB(_ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate, kSizeN);
      }

      // Performs the accumulation (C += A * B)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }

    // Stores a tile of results and performs the multiplication with alpha and beta
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        StoreResultsChecked(cpd[_ni * MWID + _mi], _mi, _ni, idm, idn, kSizeM, kSizeN,
                            alpha, beta, c_ld, c_offset, c_transpose);
      }
    }
  }
}
