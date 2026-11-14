#ifndef XGEMM_DIRECT_TYPES_GLSL_H
#define XGEMM_DIRECT_TYPES_GLSL_H

#if PRECISION == 16
  #define real2d f16vec2
#else
  #define real2d vec2
#endif

#if VWMD == 1
  #define realMD real
  #define realstoreMD realstore
#elif VWMD == 2
  #define realMD real2d
  #define realstoreMD realstore2
#elif VWMD == 4
  #define realMD real4
  #define realstoreMD realstore4
#else
  #error "VWMD must be 1, 2, or 4"
#endif

#if VWND == 1
  #define realND real
  #define realstoreND realstore
#elif VWND == 2
  #define realND real2d
  #define realstoreND realstore2
#elif VWND == 4
  #define realND real4
  #define realstoreND realstore4
#else
  #error "VWND must be 1, 2, or 4"
#endif

// The direct kernels receive A and B as vector arrays, just like OpenCL's
// realstoreMD and realstoreND pointers. The x arguments of LOADM/LOADN and
// STOREM/STOREN are vector-element indices.
#define LOADM(__buf, __x) realMD((__buf)[(__x)])
#define STOREM(__buf, __x, __val) ((__buf)[(__x)] = realstoreMD(__val))
#define LOADN(__buf, __x) realND((__buf)[(__x)])
#define STOREN(__buf, __x, __val) ((__buf)[(__x)] = realstoreND(__val))

// Scalar edge paths use scalar element indices. GLSL cannot cast a storage
// buffer to a scalar pointer, so convert the index to vector element/lane.
#if VWMD == 1
  #define LOADSINGLEM(__buf, __x) real((__buf)[(__x)])
#else
  #define LOADSINGLEM(__buf, __x) real((__buf)[(__x) / VWMD][(__x) % VWMD])
#endif

#if VWND == 1
  #define LOADSINGLEN(__buf, __x) real((__buf)[(__x)])
#else
  #define LOADSINGLEN(__buf, __x) real((__buf)[(__x) / VWND][(__x) % VWND])
#endif

#endif
