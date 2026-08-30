//Defines:
//TILE_SIZE - workgroup size (power of two)

#include "common.glsl"

layout(constant_id = 3) const int TILE_SIZE = 32;


layout(push_constant) uniform SpatialRMSNormSumSqParams{
    int nSize;
    int cSize;
    int xySize;
    int tilesPerGroup;
};

layout(set=0, binding=0) buffer readonly input_buffer {
    real d_input[];
};

layout(set=0, binding=1) buffer readonly mask_bffer {
    real mask[];
};

layout(set=0, binding=2) buffer writeonly output_buffer {
    float d_output[];
};

shared float partials[TILE_SIZE];

// workgroup [groupSize, batchSize]
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
    const int lid = int(gl_LocalInvocationID.x);
    const int n = int(gl_WorkGroupID.y);
    if(n >= nSize) return;

    const int groupIdx = int(gl_WorkGroupID.x);
    const int totalElems = cSize * xySize;
    const int chunkStart = groupIdx * TILE_SIZE * tilesPerGroup;

    float acc = 0.0f;

    for(int t = 0 ; t < tilesPerGroup ; t++) {
        int idx = chunkStart + t * TILE_SIZE + lid;
        if ( idx < totalElems ) {
            int c = idx / xySize;
            int xy = idx % xySize;
            float maskVal = LOAD(mask, n * xySize + xy);
            float val = LOAD(d_input, (n * cSize + c) * xySize + xy) * maskVal;
            acc += val * val;
        }
    }

    partials[lid] = acc;
    for(int span = TILE_SIZE / 2 ; span > 0 ; span /= 2) {
        barrier();
        if ( lid < span ) {
            partials[lid] += partials[lid + span];
        }
    }

    if ( lid == 0 ) {
        int numWorkgroups = int(gl_NumWorkGroups.x);
        d_output[n*numWorkgroups + groupIdx] = partials[0];
    }
}