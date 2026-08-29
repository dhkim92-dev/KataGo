//Defines:
//TILE_SIZE - workgroup size (power of two)

#include "common.glsl"

layout(constant_id = 4) const int TILE_SIZE = 32;

layout(push_constant) uniform SpatialRMSNormReduceParams{
    int nSize;
    int numPartials;
    int tilesPerGroup;
};

layout(set=0, binding=0) buffer readonly input_buffer {
    float d_input[];
};

layout(set=0, binding=1) buffer writeonly output_buffer {
    float d_output[];
};

shared float partials[TILE_SIZE];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
    const int lid = int(gl_LocalInvocationID.x);
    const int n = int(gl_WorkGroupID.y);
    if(n >= nSize) return;

    const int groupIdx = int(gl_WorkGroupID.x);
    const int chunkStart = groupIdx * TILE_SIZE * tilesPerGroup;

    float acc = 0.0f;
    for(int t = 0 ; t < tilesPerGroup ; t++) {
        int idx = chunkStart + t * TILE_SIZE + lid;
        if( idx < numPartials ) {
            acc += d_input[n * numPartials + idx];
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