/**
* @author dhkim92.dev@gmail.com
* transform input to winograd domain
*/
#include "common.glsl"

layout(constant_id = 3) const int INTILE_YSIZE = 4;
layout(constant_id = 4) const int INTILE_XSIZE = 4;
layout(constant_id = 5) const int OUTTILE_YSIZE = 2;
layout(constant_id = 6) const int OUTTILE_XSIZE = 2;
layout(constant_id = 7) const int INTILE_YOFFSET = -1;
layout(constant_id = 8) const int INTILE_XOFFSET = -1;
layout(constant_id = 9) const int CONV_YSIZE = 3;
layout(constant_id = 10) const int CONV_XSIZE = 3;
layout(constant_id = 11) const int ACTIVATION = 0; // 0: identity, 1: relu 2. mish, 12. mish-scale8
    
layout(push_constant) uniform WinogradInputTransform{
  int nSize;
  int xSize;
  int ySize;
  int numTilesX;
  int numTilesY;
  int icSize;
  int icSizePadded;
  int ntxtySizePadded;
  int xyStride;
};

layout(set = 0, binding = 0) readonly buffer InputBuffer {
    realstore _input[];
};
layout(set = 0, binding = 1) writeonly buffer TransformedBuffer {
    realstore _transformed[];
};
layout(set = 0, binding = 2) readonly buffer ScaleBuffer {
  realstore scale[];
};
layout(set = 0, binding = 3) readonly buffer BiasBuffer {
  realstore bias[];
};
layout(set = 0, binding = 4) readonly buffer MaskBuffer {
  realstore mask[];
};


layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
// each work item process a tile in channel
void main()
{
  int id0 = GlobalId0();
  const int ntxty = id0;
  const int tileX = id0 % numTilesX;
  id0 = int(id0 / numTilesX);
  const int tileY = id0 % numTilesY;
  id0 = int(id0 / numTilesY);
  const int n = id0;
  const int ic = GlobalId1();
  const int nic = n * icSize + ic;
  const int xySize = xSize * ySize;

  #define INPUT(_nic,_xy) LOAD(_input,((_nic) * xyStride) + (_xy))
  #define WTILE(_y,_x) wTile[(_y)*INTILE_XSIZE + (_x)]

  real wTile[INTILE_XSIZE * INTILE_YSIZE];

  //Copy input into private tile
  for(int subY = 0; subY < INTILE_YSIZE; subY++) {
    int y = tileY * OUTTILE_YSIZE + subY + INTILE_YOFFSET;
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      int x = tileX * OUTTILE_XSIZE + subX + INTILE_XOFFSET;
      real value = ZERO;
      if(y >= 0 && y < ySize && x >= 0 && x < xSize && n < nSize && ic < icSize) {
        int xy = y * xSize + x;

        if ( ACTIVATION == 0 ) {
          // IDENTITY
          // value = (INPUT(nic,xy) * LOAD(scale, ic) + LOAD(bias,ic)) * LOAD(mask, n * xySize + xy) ;
          // value = fma(INPUT(nic,xy), LOAD(scale, ic), LOAD(bias,ic) * LOAD(mask, n * xySize + xy)) ;
          value = (INPUT(nic,xy) * LOAD(scale,ic) + LOAD(bias,ic)) * LOAD(mask, n * xyStride + xy);
        } else if (ACTIVATION == 1) {
          //RELU
          value = max(INPUT(nic,xy) * LOAD(scale,ic) + LOAD(bias,ic), ZERO) * LOAD(mask, n * xyStride + xy);
        } else if (ACTIVATION == 2) {
          float a = INPUT(nic,xy) * LOAD(scale,ic) + LOAD(bias,ic);
          // value = floatToReal(a * tanh(a < LOG1PEXPTHRESHOLD ? log1p(exp(a*8.0f))) * LOAD(mask, n * xyStride + xy));
          value = floatToReal(a * tanh(a < LOG1PEXPTHRESHOLD ? log1p(exp(a)) : a)) * LOAD(mask, n * xyStride + xy);
        } else if (ACTIVATION == 12) {
          // MISH_SCALE8
          float a = INPUT(nic,xy) * LOAD(scale,ic) + LOAD(bias,ic);
          value = floatToReal(a < (LOG1PEXPTHRESHOLD*0.125f) ? a * tanh(log1p(exp(a*8.0f))) : a) * LOAD(mask, n * xyStride + xy);
        } else if (ACTIVATION == 3) {
          float a = INPUT(nic,xy) * LOAD(scale,ic) + LOAD(bias,ic);
          value = floatToReal(a / (1.0f + exp(-a))) * LOAD(mask, n * xyStride + xy);
        }
      }
      WTILE(subY,subX) = value;
    }
  }

  if ( CONV_XSIZE == 3 && OUTTILE_XSIZE == 2 ) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      real z0 = WTILE(subY,0);
      real z1 = WTILE(subY,1);
      real z2 = WTILE(subY,2);
      real z3 = WTILE(subY,3);
      WTILE(subY,0) = z0 - z2;
      WTILE(subY,1) = z1 + z2;
      WTILE(subY,2) = z2 - z1;
      WTILE(subY,3) = z1 - z3;
    }
  } else if ( CONV_XSIZE == 3 && OUTTILE_XSIZE == 4 ) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      real z0 = WTILE(subY,0);
      real z1 = WTILE(subY,1);
      real z2 = WTILE(subY,2);
      real z3 = WTILE(subY,3);
      real z4 = WTILE(subY,4);
      real z5 = WTILE(subY,5);
      WTILE(subY,0) = FOUR*z0 - FIVE*z2 + z4;
      WTILE(subY,1) = - FOUR*z1 - FOUR*z2 + z3 + z4;
      WTILE(subY,2) =   FOUR*z1 - FOUR*z2 - z3 + z4;
      WTILE(subY,3) = - TWO*z1 - z2 + TWO*z3 + z4;
      WTILE(subY,4) =   TWO*z1 - z2 - TWO*z3 + z4;
      WTILE(subY,5) = FOUR*z1 - FIVE*z3 + z5;
    }
  } else if ( CONV_XSIZE== 5 && OUTTILE_XSIZE == 2 ) {
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      real z0 = WTILE(subY,0);
      real z1 = WTILE(subY,1);
      real z2 = WTILE(subY,2);
      real z3 = WTILE(subY,3);
      real z4 = WTILE(subY,4);
      real z5 = WTILE(subY,5);
      WTILE(subY,0) = FOUR*z0 - FIVE*z2 + z4;
      WTILE(subY,1) = - FOUR*z1 - FOUR*z2 + z3 + z4;
      WTILE(subY,2) =   FOUR*z1 - FOUR*z2 - z3 + z4;
      WTILE(subY,3) = - TWO*z1 - z2 + TWO*z3 + z4;
      WTILE(subY,4) =   TWO*z1 - z2 - TWO*z3 + z4;
      WTILE(subY,5) = FOUR*z1 - FIVE*z3 + z5;
    }
  } else {
    // 
  }

  if ( CONV_YSIZE == 3 && OUTTILE_YSIZE == 2 ) {
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      real z0 = WTILE(0,subX);
      real z1 = WTILE(1,subX);
      real z2 = WTILE(2,subX);
      real z3 = WTILE(3,subX);
      WTILE(0,subX) = z0 - z2;
      WTILE(1,subX) = z1 + z2;
      WTILE(2,subX) = z2 - z1;
      WTILE(3,subX) = z1 - z3;
    }
  } else if ( CONV_YSIZE == 3 && OUTTILE_YSIZE == 4 ) {
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      real z0 = WTILE(0,subX);
      real z1 = WTILE(1,subX);
      real z2 = WTILE(2,subX);
      real z3 = WTILE(3,subX);
      real z4 = WTILE(4,subX);
      real z5 = WTILE(5,subX);
      WTILE(0,subX) = FOUR*z0 - FIVE*z2 + z4;
      WTILE(1,subX) = - FOUR*z1 - FOUR*z2 + z3 + z4;
      WTILE(2,subX) =   FOUR*z1 - FOUR*z2 - z3 + z4;
      WTILE(3,subX) = - TWO*z1 - z2 + TWO*z3 + z4;
      WTILE(4,subX) =   TWO*z1 - z2 - TWO*z3 + z4;
      WTILE(5,subX) = FOUR*z1 - FIVE*z3 + z5;
    }
  } else if ( CONV_YSIZE == 5 && OUTTILE_YSIZE == 2 ) {
    for(int subX = 0; subX < INTILE_XSIZE; subX++) {
      real z0 = WTILE(0,subX);
      real z1 = WTILE(1,subX);
      real z2 = WTILE(2,subX);
      real z3 = WTILE(3,subX);
      real z4 = WTILE(4,subX);
      real z5 = WTILE(5,subX);
      WTILE(0,subX) = FOUR*z0 - FIVE*z2 + z4;
      WTILE(1,subX) = - FOUR*z1 - FOUR*z2 + z3 + z4;
      WTILE(2,subX) =   FOUR*z1 - FOUR*z2 - z3 + z4;
      WTILE(3,subX) = - TWO*z1 - z2 + TWO*z3 + z4;
      WTILE(4,subX) =   TWO*z1 - z2 - TWO*z3 + z4;
      WTILE(5,subX) = FOUR*z1 - FIVE*z3 + z5;
    }
  } else {
    // 
  }
  #define WRITETRANS(_suby,_subx,_ic,_ntile,_value) STORE(_transformed,(((_suby) * INTILE_XSIZE + (_subx))*icSizePadded + (_ic))*ntxtySizePadded + (_ntile),_value)

  if(ntxty < ntxtySizePadded && ic < icSizePadded) {
    //Copy private tile out to transformed output
    for(int subY = 0; subY < INTILE_YSIZE; subY++) {
      for(int subX = 0; subX < INTILE_XSIZE; subX++) {
        real result = WTILE(subY,subX);
        WRITETRANS(subY,subX,ic,ntxty,result);
        // WRITETRANS(subY,subX,ic,ntxty, LocalId );
      }
    }
  }
}