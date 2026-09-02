/**
* Vulkan ported transformer RMSNorm kernel.
* Based on openclkernels.cpp, transformerRMSNorm kernel function
*/
#include "common.glsl"

layout(constant_id = 3) const int WG_C_SIZE = 64;
layout(constant_id = 4) const int WG_XY_SIZE = 1;
layout(constant_id = 5) const int C_PER_THREAD = 4;

layout(push_constant) uniform TransformerRMSNormParams {
    int nSize;
    int cSize;
    int xySize;
    float eps;
};

layout(set = 0, binding = 0) buffer readonly input_buffer {
    realstore d_input[];// NCHW
};

layout(set = 0, binding = 1) buffer writeonly output_buffer {
    realstore d_output[]; // NCHW
};

layout(set = 0, binding = 2) buffer readonly weight_buffer {
    float weight[]; // C(gamma)
};

layout(set = 0, binding = 3) buffer readonly beta_buffer {
    float beta[]; // (per-channel bias added after scaling)
};

layout(set = 0, binding = 4) buffer readonly mask_buffer {
    realstore mask[]; // NHW
};

shared float partials[WG_XY_SIZE * WG_C_SIZE];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
    const int lid = int(gl_LocalInvocationID.x);
    const int lid_xy = lid % WG_XY_SIZE;
    const int lid_c = lid / WG_XY_SIZE;
    const int xy = int(gl_WorkGroupID.x) * WG_XY_SIZE + lid_xy;
    const int n = int(gl_WorkGroupID.y);

    if ( n >= nSize) return;

    float maskVal = (xy < xySize) ? LOAD(mask, n*xySize + xy) : 0.0f;

    // Phase 1: Sum of squares for its channels;
    float acc = 0.0f;

    for ( int base = lid_c * C_PER_THREAD ; base < cSize ; base += WG_C_SIZE * C_PER_THREAD ) {
        for ( int dc = 0 ; dc < C_PER_THREAD ; dc++ ) {
            int c = base + dc;
            if ( c < cSize && xy < xySize ) {
                float val = LOAD(d_input, (n*cSize + c) * xySize + xy) * maskVal;
                acc += val*val;
            }
        }
    }

    // Phase 2: Reduce across C dimension in local memory

    partials[lid_xy * WG_C_SIZE + lid_c] = acc;
    for ( int span = WG_C_SIZE / 2 ; span > 0 ; span /= 2 ) {
        barrier();
        if ( lid_c < span ) {
            partials[lid_xy * WG_C_SIZE + lid_c] += partials[lid_xy * WG_C_SIZE + lid_c + span];
        }
    }
    barrier();

    float rms = inversesqrt(partials[lid_xy * WG_C_SIZE] / float(cSize) + eps);

    // Phase 3: Apply normalization
    for ( int base = lid_c * C_PER_THREAD ; base < cSize ; base += WG_C_SIZE * C_PER_THREAD ) {
        for ( int dc = 0 ; dc < C_PER_THREAD ; dc++ ) {
            int c = base + dc;
            if ( (c < cSize) && xy < xySize ) {
                float val = LOAD(d_input, (n * cSize + c) * xySize + xy);
                float result = (val * rms * weight[c] + beta[c]) * maskVal;
                STORE(d_output, (n * cSize + c) * xySize + xy, floatToReal(result));
            }
        }
    }
}
