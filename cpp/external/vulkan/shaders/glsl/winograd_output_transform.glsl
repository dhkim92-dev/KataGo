/**
* @author dhkim92.dev@gmail.com
* @brief Winograd Output Recover Shader
* @details This shader performs A * input * A^T to recover the output from Winograd domain to spatial domain.
*          x-dimension means number of tiles in width axis
*          y-dimension means number of tiles in height axis
*          z-dimension means batch * output_channels
*          each thread 
* _input = [ InTileSizeY, InTileSizeX, OC, Batch, NumTileY, NumTIleX ] last two dimensions are padded
* _output = [ N, C, H, W ] 
*/

#include "common.glsl"

layout(constant_id=3) const int INTILE_YSIZE = 4;
layout(constant_id=4) const int INTILE_XSIZE = 4;
layout(constant_id=5) const int OUTTILE_YSIZE = 2;
layout(constant_id=6) const int OUTTILE_XSIZE = 2;
layout(constant_id=7) const int CONV_YSIZE = 3;
layout(constant_id=8) const int CONV_XSIZE = 3;

layout(set=0, binding = 0) readonly buffer InputBuffer {
    realstore _input[];
};

layout(set=0, binding = 1) writeonly buffer OutputBuffer {
    realstore _output[];
};

layout(push_constant) uniform WinogradOutputRecover{
    int batchSize;
    int ySize;
    int xSize;
    int numTilesY;
    int numTilesX;
    int outChannels;
    int outChannelsPadded;
    int ntxtySizePadded;
    int xyStride;
};

#define WTILE(_y, _x) wTile[(_y)*INTILE_XSIZE + (_x)]
#define TRANS(_suby, _subx, _oc, _ntile) LOAD(_input, ((((_suby) * INTILE_XSIZE + (_subx)) * outChannelsPadded + (_oc)) * ntxtySizePadded + (_ntile)))
// #define WRITEOUTPUT(_noc, _y, _x, _value) STORE(_output, (((_noc) * xyStride + (_y)) * xSize + (_x)), _value)
#define WRITEOUTPUT(_noc,_y,_x,_value) STORE(_output,(_noc) * xyStride + (_y) * xSize + (_x),_value)
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in; 

void main() {
    const int tileX = GlobalId0();
    const int tileY = GlobalId1();
    const int batchOutC = GlobalId2();
    const int batch = batchOutC / outChannels;
    const int outC = batchOutC % outChannels;
    const int nTiles = (batch * numTilesY + tileY) * numTilesX + tileX;

    real wTile[INTILE_YSIZE * INTILE_XSIZE];
    // load output tile from winograd domain to wTile

    if ( tileY < numTilesY && tileX < numTilesX && batch < batchSize) {
        for ( int subY = 0 ; subY < INTILE_YSIZE ; ++subY ) {
            for ( int subX = 0 ; subX < INTILE_XSIZE ; ++subX ) {
                WTILE(subY, subX) = TRANS(subY, subX, outC, nTiles);
            }
        }
    }

    // (debug block moved to end of function to avoid being overwritten by writes)

    if ( CONV_YSIZE == 3 && OUTTILE_YSIZE == 2 ) {
        // row transform
        for ( int subY = 0 ; subY < INTILE_YSIZE ; ++subY ) {
            real z0 = WTILE(subY, 0);
            real z1 = WTILE(subY, 1);
            real z2 = WTILE(subY, 2);
            real z3 = WTILE(subY, 3);
            WTILE(subY, 0) = z0 + z1 + z2;
            WTILE(subY, 1) = z1 - z2 - z3;
        }
        // col transform
        for ( int subX = 0 ; subX < OUTTILE_XSIZE ; ++subX ) {
            real z0 = WTILE(0, subX);
            real z1 = WTILE(1, subX);
            real z2 = WTILE(2, subX);
            real z3 = WTILE(3, subX);
            WTILE(0, subX) = z0 + z1 + z2;
            WTILE(1, subX) = z1 - z2 - z3;
        }
    } else if ( CONV_YSIZE == 3 && OUTTILE_YSIZE == 4 ) {
        // row transform
        for ( int subY = 0 ; subY < INTILE_YSIZE ; ++subY ) {
            real z0 = WTILE(subY, 0);
            real z1 = WTILE(subY, 1);
            real z2 = WTILE(subY, 2);
            real z3 = WTILE(subY, 3);
            real z4 = WTILE(subY, 4);
            real z5 = WTILE(subY, 5);
            WTILE(subY,0) = z0 + z1 + z2 + z3 + z4;
            WTILE(subY,1) = (z1 - z2) + TWO*(z3 - z4);
            WTILE(subY,2) = (z1 + z2) + FOUR*(z3 + z4);
            WTILE(subY,3) = (z1 - z2) + EIGHT*(z3 - z4) + z5;
        }
        // col transform
        for ( int subX = 0 ; subX < OUTTILE_XSIZE ; ++subX ) {
            real z0 = WTILE(0, subX);
            real z1 = WTILE(1, subX);
            real z2 = WTILE(2, subX);
            real z3 = WTILE(3, subX);
            real z4 = WTILE(4, subX);
            real z5 = WTILE(5, subX);
            WTILE(0, subX) = z0 + z1 + z2 + z3 + z4;
            WTILE(1, subX) = (z1 - z2) + TWO*(z3 - z4);
            WTILE(2, subX) = (z1 + z2) + FOUR*(z3 + z4);
            WTILE(3, subX) = (z1 - z2) + EIGHT*(z3 - z4) + z5;
        }
    } else if ( CONV_YSIZE == 5 && OUTTILE_YSIZE == 2 ) {
        //row transform
        for( int subY = 0; subY < INTILE_YSIZE; subY++) {
            real z0 = WTILE(subY,0);
            real z1 = WTILE(subY,1);
            real z2 = WTILE(subY,2);
            real z3 = WTILE(subY,3);
            real z4 = WTILE(subY,4);
            real z5 = WTILE(subY,5);
            WTILE(subY,0) = z0 + z1 + z2 + z3 + z4;
            WTILE(subY,1) = (z1 - z2) + TWO*(z3 - z4) + z5;
        }

        // col transform
        for( int subX = 0; subX < OUTTILE_XSIZE; subX++) {
            real z0 = WTILE(0,subX);
            real z1 = WTILE(1,subX);
            real z2 = WTILE(2,subX);
            real z3 = WTILE(3,subX);
            real z4 = WTILE(4,subX);
            real z5 = WTILE(5,subX);
            WTILE(0,subX) = z0 + z1 + z2 + z3 + z4;
            WTILE(1,subX) = (z1 - z2) + TWO*(z3 - z4) + z5;
        }
    }

    // write output

    for ( int subY = 0 ; subY < OUTTILE_YSIZE ; ++subY ) {
        int y = tileY * OUTTILE_YSIZE + subY;
        for ( int subX = 0 ; subX < OUTTILE_XSIZE ; ++subX ) {
            int x = tileX * OUTTILE_XSIZE + subX;

            if ( y >= 0 && y < ySize 
                 && x >= 0 && x < xSize 
                 && batch < batchSize 
                 && tileX < numTilesX 
                 && tileY < numTilesY ) {
                    real result = WTILE(subY, subX);
                    WRITEOUTPUT( batchOutC, y, x, result );
            }
        }
    }

    // DEBUG: write a small, deterministic diagnostic blob into the front of the
    // output buffer for verification. This is intentionally guarded so only a
    // single invocation writes these values to make it easy to paste and inspect
    // on the host. Remove or disable this block once verification is complete.
    // if (batch == 0 && outC == 0 && tileX == 0 && tileY == 0 && LocalId0() == 0 && LocalId1() == 0 && LocalId2() == 0) {
    //     // Linear stores into the output buffer (overwrites the first few output elements).
    //     // Values captured: batchSize, ySize, xSize, numTilesY, numTilesX, outChannels, outChannelsPadded, ntxtySizePadded, nTiles
    //     STORE(_output, 0, real(batchSize));
    //     STORE(_output, 1, real(ySize));
    //     STORE(_output, 2, real(xSize));
    //     STORE(_output, 3, real(numTilesY));
    //     STORE(_output, 4, real(numTilesX));
    //     STORE(_output, 5, real(outChannels));
    //     STORE(_output, 6, real(outChannelsPadded));
    //     STORE(_output, 7, real(ntxtySizePadded));
    //     STORE(_output, 8, real(nTiles));
    //     // Also record the raw transformed tile values (up to 16 entries) so
    //     // we can compare what the shader loaded from the transformed buffer.
    //     // Layout: row-major WTILE(y,x) for y=0..3, x=0..3 stored at indices 9..24.
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 1)  STORE(_output,  9, real(WTILE(0,0)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 2)  STORE(_output, 10, real(WTILE(0,1)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 3)  STORE(_output, 11, real(WTILE(0,2)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 4)  STORE(_output, 12, real(WTILE(0,3)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 5)  STORE(_output, 13, real(WTILE(1,0)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 6)  STORE(_output, 14, real(WTILE(1,1)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 7)  STORE(_output, 15, real(WTILE(1,2)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 8)  STORE(_output, 16, real(WTILE(1,3)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 9)  STORE(_output, 17, real(WTILE(2,0)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 10) STORE(_output, 18, real(WTILE(2,1)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 11) STORE(_output, 19, real(WTILE(2,2)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 12) STORE(_output, 20, real(WTILE(2,3)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 13) STORE(_output, 21, real(WTILE(3,0)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 14) STORE(_output, 22, real(WTILE(3,1)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 15) STORE(_output, 23, real(WTILE(3,2)));
    //     if (INTILE_XSIZE * INTILE_YSIZE >= 16) STORE(_output, 24, real(WTILE(3,3)));
    // }

}
