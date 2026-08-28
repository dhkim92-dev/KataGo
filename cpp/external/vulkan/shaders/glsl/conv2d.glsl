#include "common.glsl"
#include "functions.glsl"

//Spatial size of tile loaded into local memory, not counting filterRadius
#ifndef TILE_XSIZE
#define TILE_XSIZE 32
#endif
#ifndef TILE_YSIZE
#define TILE_YSIZE 4
#endif

//Channel depth of tile loaded into local memory
#ifndef TILE_CHANNELS
#define TILE_CHANNELS 4
#endif

// Buffers: set 0
// input: layout NCHW flattened
// filters: layout (oc, ic, fh, fw) flattened
// output: layout NCHW flattened

layout(constant_id=3) const int FILTER_XRADIUS = 1;
layout(constant_id=4) const int FILTER_YRADIUS = 1;

layout(set = 0, binding = 0) readonly buffer input_buf {
  realstore d_input[];
};
layout(set = 0, binding = 1) readonly buffer filter_buf {
  realstore d_filter[];
};
layout(set = 0, binding = 2) buffer output_buf{
  realstore d_output[];
};

layout(push_constant) uniform Conv2DParams{
  int nSize;
  int xSize;
  int ySize;
  int ocSize;
  int icSize;
  int filterXRadius;
  int filterYRadius;
  int xyStride;
};

shared float inputTile[
    TILE_CHANNELS *
    (TILE_YSIZE + 2 * FILTER_YRADIUS) *
    (TILE_XSIZE + 2 * FILTER_XRADIUS)
];

shared float outputTile[
    TILE_YSIZE * TILE_XSIZE
];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main()
{
  const int xBase = GroupId0() * TILE_XSIZE;
  const int yBase = GroupId1() * TILE_YSIZE;
  const int oc = GroupId2();

  const int lx = LocalId0();
  const int ly = LocalId1();
  const int lxSize = LocalSize0();
  const int lySize = LocalSize1();

  //The input tile is filterXRadius*2 or filterYRadius*2 larger than the tile size
  const int inputTileXSize = TILE_XSIZE + filterXRadius * 2;
  const int inputTileYSize = TILE_YSIZE + filterYRadius * 2;

  const int fxSize = (2 * filterXRadius + 1);
  const int fySize = (2 * filterYRadius + 1);

#define INPUT(_n,_ic,_y,_x) LOAD(d_input,((_n) * icSize + (_ic)) * xyStride + (_y) * xSize + (_x))
#define INPUTTILE(_ic,_ity,_itx) inputTile[((_ic) * inputTileYSize + (_ity)) * inputTileXSize + (_itx)]

#define FILTER(_oc,_ic,_y,_x) LOAD(d_filter,(((_oc) * icSize + (_ic)) * fySize + (_y)) * fxSize + (_x))

#define WRITEOUTPUT(_n,_oc,_y,_x,_value) STORE(d_output,((_n) * ocSize + (_oc)) * xyStride + (_y) * xSize + (_x),_value)
#define OUTPUTTILE(_oty,_otx) outputTile[(_oty) * TILE_XSIZE + (_otx)]

  for(int n = 0; n < nSize; n++) {
    real acc = ZERO;

    //Initialize outputTile. No need to sync for this tile since each thread only ever reads its own spots
    for(int oty = ly; oty<TILE_YSIZE; oty += lySize) {
      for(int otx = lx; otx<TILE_XSIZE; otx += lxSize) {
        OUTPUTTILE(oty,otx) = ZERO;
      }
    }

    //Walk over chunks of TILE_CHANNELS many input channels at a time
    for(int icBase = 0; icBase<icSize; icBase += TILE_CHANNELS) {

      //Copy input tile using local threads in parallel
      for(int dic = 0; dic<TILE_CHANNELS && icBase+dic < icSize; dic += 1) {
        for(int ity = ly; ity<inputTileYSize; ity += lySize) {
          int iy = ity+yBase-filterYRadius;
          for(int itx = lx; itx<inputTileXSize; itx += lxSize) {
            int ix = itx+xBase-filterXRadius;
            real inputValue = ZERO;
            if(iy >= 0 && iy < ySize && ix >= 0 && ix < xSize) {
              inputValue = INPUT(n,icBase+dic,iy,ix);
            }
            INPUTTILE(dic,ity,itx) = inputValue;
          }
        }
      }

      //Synchronize!
      barrier();

      //Accumulate this convolution block into output tile.
      //Iterate over the bits in this tile that the thread is responsible for
      for(int oty = ly; oty<TILE_YSIZE; oty += lySize) {
        for(int otx = lx; otx<TILE_XSIZE; otx += lxSize) {

          //And then perform the convolution to accumulate that bit
          real acc = ZERO;
          for(int dic = 0; dic<TILE_CHANNELS && icBase+dic < icSize; dic += 1) {
            for(int fy = 0; fy < fySize; fy++) {
              for(int fx = 0; fx < fxSize; fx++) {
                acc += INPUTTILE(dic,oty+fy,otx+fx) * FILTER(oc,icBase+dic,fy,fx);
              }
            }
          }
          OUTPUTTILE(oty,otx) += acc;
        }
      }
    } //close loop over input channel chunks

    //Now, write tile contents back into output
    for(int oty = ly; oty<TILE_YSIZE; oty += lySize) {
      int oy = yBase+oty;
      for(int otx = lx; otx<TILE_XSIZE; otx += lxSize) {
        int ox = xBase+otx;
        if(oy >= 0 && oy < ySize && ox >= 0 && ox < xSize) {
          real result = OUTPUTTILE(oty,otx);
          WRITEOUTPUT(n, oc, yBase+oty, xBase+otx, result);
        }
      }
    }
  } //Close loop over batch
}
