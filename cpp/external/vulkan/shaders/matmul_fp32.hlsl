/**
* @author dhkim92.dev@gmail.com
*/
// Tiled MatMul HLSL for Vulkan
// Globals: input tensor (A), weight tensor (B), output tensor (C)
// Push constants match KatagoVulkan::MatmulFp32: M,K,N,numBatchElts,cTranspose

#include "common.h"
#include "functions.h"

// Threadgroup size should match TILE_M x TILE_N for this simple implementation

// Push constants - mapped by host as push constants
struct MatmulPushConstants  {
    uint M;            // rows of A and C
    uint K;            // cols of A and rows of B
    uint N;            // cols of B and C
    uint numBatchElts; // z-dimension: number of independent GEMMs
    uint cTranspose;   // if 1, write C transposed (col-major by M)
};

[[vk::push_constant]]
MatmulPushConstants params;

// Descriptor bindings: weights(B) read-only, input(A) read-write, output(C) read-write
// t0: weights (read-only), u0: input (read-write), u1: output (read-write)
// NOTE: Host packs weights as N x K (outChannels x inChannels) row-major: index = n*K + k
RWStructuredBuffer<float> gInput   : register(u0);
StructuredBuffer<float>  gWeights : register(t0);
RWStructuredBuffer<float> gOutput  : register(u1);

groupshared float Asub[MATMUL_TILE_M * MATMUL_TILE_K];
groupshared float Bsub[MATMUL_TILE_K * MATMUL_TILE_N];

[numthreads(MATMUL_DISPATCH_X, MATMUL_DISPATCH_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID,
          uint3 GTid : SV_GroupThreadID,
          uint3 GId  : SV_GroupID)
{
    // Group coordinates
    uint groupM = GId.x; // tile index along M axis
    uint groupN = GId.y; // tile index along N axis
    uint groupZ = GId.z; // instance index (0..numBatchElts-1)

    // Local thread coordinates inside group
    uint localRow = GTid.x; // 0..MATMUL_TILE_M-1
    uint localCol = GTid.y; // 0..MATMUL_TILE_N-1

    // Global row/col base for this tile
    uint rowBase = groupM * MATMUL_TILE_M;
    uint colBase = groupN * MATMUL_TILE_N;

    // Row and col this thread will compute within C
    uint row = rowBase + localRow;
    uint col = colBase + localCol;

    // Per-instance offsets assuming contiguous per-instance layout
    uint aInstanceStride = params.M * params.K; // elements per A instance
    uint bInstanceStride = params.K * params.N; // elements per B instance
    uint cInstanceStride = params.M * params.N; // elements per C instance

    uint aBase = groupZ * aInstanceStride;
    uint bBase = groupZ * bInstanceStride;
    uint cBase = groupZ * cInstanceStride;

    // Accumulator
    float acc = 0.0f;

    // Loop over K in blocks
    [loop]
    for (uint kk = 0; kk < params.K; kk += MATMUL_TILE_K) {
        // Load A sub-tile into shared memory
        // Each thread loads multiple elements covering A_tile of size TILE_M x TILE_K
        [unroll]
        for (uint kInner = 0; kInner < MATMUL_TILE_K; ++kInner) {
            uint aRow = rowBase + localRow;
            uint aCol = kk + kInner;
            float aVal = 0.0f;
            if (aRow < params.M && aCol < params.K) {
                uint aIndex = aBase + aRow * params.K + aCol; // A[row, k]
                aVal = gInput[aIndex];
            }
            // store at Asub[localRow * TILE_K + kInner]
            Asub[localRow * MATMUL_TILE_K + kInner] = aVal;
        }

        // Load B sub-tile into shared memory
        [unroll]
        for (uint kInner = 0; kInner < MATMUL_TILE_K; ++kInner) {
            uint bRow = kk + kInner; // k
            uint bCol = colBase + localCol;
            float bVal = 0.0f;
            if (bRow < params.K && bCol < params.N) {
                // gWeights is N x K (n,k) row-major; compute index = bBase + n*K + k
                uint bIndex = bBase + bCol * params.K + bRow; // B[n, k]
                bVal = gWeights[bIndex];
            }
            // store at Bsub[kInner * MATMUL_TILE_N + localCol]
            Bsub[kInner * MATMUL_TILE_N + localCol] = bVal;
        }

        GroupMemoryBarrierWithGroupSync();

        // Compute partial product for this tile
        [unroll]
        for (uint kInner = 0; kInner < MATMUL_TILE_K; ++kInner) {
            float aVal = Asub[localRow * MATMUL_TILE_K + kInner];
            float bVal = Bsub[kInner * MATMUL_TILE_N + localCol];
            acc += aVal * bVal;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write result
    if (row < params.M && col < params.N) {
        uint cIndex;
        if (params.cTranspose == 1) {
            // Write transposed: index = cBase + col * M + row
            cIndex = cBase + col * params.M + row;
        } else {
            // Normal row-major: index = cBase + row * N + col
            cIndex = cBase + row * params.N + col;
        }
        gOutput[cIndex] = acc;
    }
}
