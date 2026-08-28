#include "common.glsl"
#ifndef VWM
  #define VWM 4
#endif
#ifndef VWN
  #define VWN 4
#endif
#ifndef STRM
  #define STRM 0
#endif
#ifndef STRN
  #define STRN 0
#endif

#define MWG _MWG
#define NWG _NWG
#define KWG _KWG
#define MDIMC _MDIMC
#define NDIMC _NDIMC
#define MDIMA _MDIMA
#define NDIMB _NDIMB
#ifndef KREG
  #define KREG 1
#endif
#ifndef KWI
  #define KWI  1 // unroll factor for K dimension
#endif

#define MWI (MWG/MDIMC)               // Work per work-item (M-dimension)
#define NWI (NWG/NDIMC)               // Work per work-item (N-dimension)
#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG/MDIMA)               // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG/KDIMA)               // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG/KDIMB)               // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG/NDIMB)               // Amount of loads-per-thread for matrix B (N-dimension)

#define realM real4
#define realN real4
#define realstoreM realstore
#define realstoreN realstore

#if PRECISION_STORAGE == 16
  #define LOAD4(__buf, __x) real4(((__buf)[(__x)>>2]))
  #define LOADGLOBALM(__buf, __x) __buf[(__x)]
  #define LOADGLOBALN(__buf, __x) __buf[(__x)]
  #define STOREGLOBALM(__buf, __x, __y) __buf[(__x)] = __y
  #define STOREGLOBALN(__buf, __x, __y) __buf[(__x)] = __y
  #define LOADLOCALM(__buf, __x) __buf[(__x)]
  #define LOADLOCALN(__buf, __x) __buf[(__x)]
  #define STORELOCALM(__buf, __x, __y) __buf[(__x)] = __y
  #define STORELOCALN(__buf, __x, __y) __buf[(__x)] = __y
#else 
  #define LOADGLOBALM(__buf, __x) __buf[(__x)]
  #define LOADGLOBALN(__buf, __x) __buf[(__x)]
  #define STOREGLOBALM(__buf, __x, __y) __buf[(__x)] = __y
  #define STOREGLOBALN(__buf, __x, __y) __buf[(__x)] = __y
  #define LOADLOCALM(__buf, __x) __buf[(__x)]
  #define LOADLOCALN(__buf, __x) __buf[(__x)]
  #define STORELOCALM(__buf, __x, __y) __buf[(__x)] = __y
  #define STORELOCALN(__buf, __x, __y) __buf[(__x)] = __y
#endif

// return single real value from real4 buffer
#define LOADGLOBAL(__buf, __x) (__buf[(__x)>>2][(__x)&3])
#define STOREGLOBAL(__buf, __x, __y)  {           \
    real4 _tmp = __buf[(__x)>>2];          \
    _tmp[(__x)&3] = __y;                   \
    __buf[(__x)>>2] = _tmp;                \
}

realM InitAccRegisters() {
    return realM(ZERO);
}

void GlobalToLocalA(const int kSizeM, const int tid, const int kwg, const int baseA) {
  const int la0 = tid % MDIMA;
  const int la1 = tid / MDIMA;
  #pragma unroll
  for (int _mia = 0; _mia < MWA/VWM; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWA; _kia += 1) {
      // Computes the indices based on strided/non-strided access
      int mg = _mia + la0*(MWA/VWM);
      // Computes the indices for the global memory
      int kg = _kia + la1*KWA;
      int idm = mg + GroupId0() * (MWG/VWM);
      int idk = kg + kwg;

      // Loads the data from global memory (not transposed) into the local memory
      alm[kg*(MWG/VWM) + mg] = agm[baseA + idk*(kSizeM/VWM) + idm];
    }
  }
}

void GlobalToLocalB(const int kSizeN, const int tid, const int kwg, const int baseB) {
  const int lb0 = tid % NDIMB;
  const int lb1 = tid / NDIMB;
  #pragma unroll
  for (int _kib = 0; _kib < KWB; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {

      int ng = _nib + lb0*(NWB/VWN);

      int kg = _kib + lb1*KWB;
      int idn = ng + GroupId1() * (NWG/VWN);
      int idk = kg + kwg;

      blm[kg*(NWG/VWN) + ng] = bgm[baseB + idk*(kSizeN/VWN) + idn];
    }
  }
}

realM LocalToPrivateA(const int _mi, const int kg) {
  int mg = _mi + LocalId0()*(MWI/VWM);
  return LOADLOCALM(alm,kg*(MWG/VWM) + mg);
}

realN LocalToPrivateB(const int _ni, const int kg) {
  int ng = _ni + LocalId1()*(NWI/VWN);
  return LOADLOCALN(blm,kg*(NWG/VWN) + ng);
}

#define MUL_ADD_SCALAR(a, b, c) fma((a), (b), (c))
/**
* cvec : target vector to be updated
* avec : vector to be multiplied
* bval : scalar value to multiply
*/
// #define MultiplyAddVector(cvec, avec, bval) (cvec + avec * bval) 
#define MUL_ADD_SCALAR(a, b, c) fma((a), (b), (c))

#define MultiplyAddVector(cvec, avec, bval) \
  real4( MUL_ADD_SCALAR(avec.x, bval, cvec.x), \
          MUL_ADD_SCALAR(avec.y, bval, cvec.y), \
          MUL_ADD_SCALAR(avec.z, bval, cvec.z), \
          MUL_ADD_SCALAR(avec.w, bval, cvec.w) ) 

void StoreResults(realM _c_value, int _mi, int _ni, int _kSizeM, int _baseC) {
  int mg = _mi + LocalId0()*(MWI/VWM);
  int ng = _ni + LocalId1()*NWI;
  int idm = mg + GroupId0() * (MWG/VWM);
  int idn = ng + GroupId1() * NWG;
  int index = _baseC + idn*(_kSizeM/VWM) + idm;
  realM xval = _c_value;
  STOREGLOBALM(cgm, index, xval);
}


/**
* @brief Performs C += A*B where A, B, C are matrices and the multiplication is done in a tiled, batched manner.
* a_offset, b_offset, c_offset are the starting offsets for the A, B, C matrices for the current batch element.
* required to be divided by 4 (size of real4)
*/
void XgemmBody(
  const int kSizeM, const int kSizeN, const int kSizeK,
  const int a_offset, const int b_offset, const int c_offset
) {

  // Allocates workitem-private memory (registers)
  realM apm[MWI/VWM]; // MWI * 1
  realN bpm[NWI/VWN]; // 1 * NWI
  realM cpm[NWI*(MWI/VWM)]; // NWI * MWI
  volatile int tid = LocalId0() + MDIMC*LocalId1();

  const int baseA = a_offset / VWM;
  const int baseB = b_offset / VWN;
  const int baseC = c_offset / VWM;

  #pragma unroll
  for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
      cpm[_ni * (MWI/VWM) + _mi] = InitAccRegisters();
    }
  }

  // Loops over all workgroup tiles
  for (int kwg = 0; kwg < kSizeK; kwg += KWG * KREG) {

    // Loads data: off-chip --> local (matrix A)
    GlobalToLocalA(kSizeM, tid, kwg, baseA);
    // Loads data: off-chip --> local (matrix B)
    GlobalToLocalB(kSizeN, tid, kwg, baseB);
    barrier();

    // Loops over all workitem tiles, unrolled by a factor KWI
    for (int pwi = 0; pwi < KWG * KREG; pwi += KWI * KREG) {
      #pragma unroll
      for (int _pit = 0; _pit < KWI*KREG; _pit += KREG) {
        int kg = pwi + _pit;

        // Loads matrix A 
        #pragma unroll
        for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
          // Loads data: local --> private (matrix A)
          apm[_mi] = LocalToPrivateA(_mi, kg);
        }

        // Loads matrix B
        #pragma unroll
        for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
          bpm[_ni] = LocalToPrivateB(_ni, kg);
        }
        // Performs the accumulation (Cpm += Apm * Bpm)
        #pragma unroll
        for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
          #pragma unroll
          for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
            const realM aval = apm[_mi];
            cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].x);
            cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].y);
            cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[_ni].z);
            cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[_ni].w);
          }
        }
      }
    }
    barrier();
  }

  memoryBarrierBuffer();
  barrier();

  // Stores an MWG * NWG tile of results
  const int cld = kSizeM;
  #pragma unroll
  for (int _ni = 0; _ni < NWI; _ni += 1) {
    #pragma unroll
    for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
      StoreResults(cpm[_ni * (MWI/VWM) + _mi], _mi, _ni, cld, baseC);
    }
  }
}