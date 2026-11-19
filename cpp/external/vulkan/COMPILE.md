# Vulkan shader compilation

## Export policy

The backend uses Vulkan 1.1 as its minimum API. General shader variants are compiled with `--target-env=vulkan1.1 --target-spv=spv1.3`. The NV2 register-attention variant uses `--target-env=vulkan1.3 --target-spv=spv1.6` and `ATTENTION_USE_NV2=1`.

These export targets are verified by compilation and SPIR-V validation. Individual arithmetic operations can have lower theoretical minimum versions. Include-only GLSL files inherit the requirements of their consuming `.comp` wrapper. A device must also support every capability declared by the selected variant.

| Compiled family | Vulkan API | SPIR-V | Additional runtime requirements |
| --- | --- | --- | --- |
| General FP32 | 1.1 | 1.3 | Shader and device workgroup/shared-memory limits |
| FP16 storage/compute variants | 1.1 | 1.3 | Enabled 16-bit buffer storage and shaderFloat16; on API 1.1, enable VK_KHR_shader_float16_int8 |
| KHR HGEMM and dual GEMM/SwiGLU | 1.1 | 1.3 | VK_KHR_cooperative_matrix, enabled Vulkan Memory Model, FP16, compute BASIC subgroups, full subgroups and supported matrix shapes |
| KHR attention, original/block | 1.1 | 1.3 | KHR matrix requirements and attention shape/resource eligibility |
| KHR attention, pair | 1.1 | 1.3 | KHR matrix requirements and compute subgroup SHUFFLE support |
| RMSNorm subgroup variants | 1.1 | 1.3 | FP16, compute BASIC/ARITHMETIC subgroups, full subgroups and norm workgroup limits |
| NV2 register attention | 1.3 | 1.6 | KHR matrix requirements, VK_NV_cooperative_matrix2 workgroup scope, flexible dimensions, reductions, conversions, per-element operations and enabled Maintenance4 |

Module eligibility requires both FP16 storage and arithmetic for precision variants, matching the tuner capability policy.

Vulkan Memory Model is enabled through `VK_KHR_vulkan_memory_model` on API 1.1 and through core features on API 1.2+. Subgroup-size control/full subgroups use the extension path below API 1.3. The NV2 module uses `LocalSizeId`, which requires Maintenance4. Baseline modules use `LocalSize` and specialized `WorkgroupSize`.

## GLSL inventory

The API/SPIR-V columns describe export targets. The cooperative attention row lists KHR first and NV2 second. Precision wrappers and conditional subgroup branches add the requirements in the family table.

| GLSL source | Vulkan API | SPIR-V | Operation |
| --- | --- | --- | --- |
| [add_channel_bias.glsl](shaders/glsl/add_channel_bias.glsl) | 1.1 | 1.3 | Add per-channel bias to spatial tensors. |
| [add_channel_bias_nc.glsl](shaders/glsl/add_channel_bias_nc.glsl) | 1.1 | 1.3 | Add bias and activation to NC tensors. |
| [add_pointwise.glsl](shaders/glsl/add_pointwise.glsl) | 1.1 | 1.3 | Add tensors element by element. |
| [bn_mask.glsl](shaders/glsl/bn_mask.glsl) | 1.1 | 1.3 | Apply batch normalization, activation and masking. |
| [common.glsl](shaders/glsl/common.glsl) | 1.1 | 1.3 | Define shared precision types and dispatch constants. |
| [conv2d.glsl](shaders/glsl/conv2d.glsl) | 1.1 | 1.3 | Compute direct spatial convolution. |
| [extract_channel0.glsl](shaders/glsl/extract_channel0.glsl) | 1.1 | 1.3 | Extract the first spatial channel. |
| [functions.glsl](shaders/glsl/functions.glsl) | 1.1 | 1.3 | Define activation macros. |
| [global_pooling_channels.glsl](shaders/glsl/global_pooling_channels.glsl) | 1.1 | 1.3 | Reduce spatial channels into pooled statistics. |
| [hgemm_cooperative_matrix.glsl](shaders/glsl/hgemm_cooperative_matrix.glsl) | 1.1 | 1.3 | Multiply matrices with KHR cooperative matrices. |
| [hgemm_cooperative_matrix_nchw.glsl](shaders/glsl/hgemm_cooperative_matrix_nchw.glsl) | 1.1 | 1.3 | Multiply NCHW matrices with KHR cooperative matrices. |
| [hgemm_cooperative_matrix_nhwc.glsl](shaders/glsl/hgemm_cooperative_matrix_nhwc.glsl) | 1.1 | 1.3 | Multiply NHWC matrices with KHR cooperative matrices. |
| [sum_channels.glsl](shaders/glsl/sum_channels.glsl) | 1.1 | 1.3 | Sum spatial elements for each batch and channel. |
| [transformer_apply_rope.glsl](shaders/glsl/transformer_apply_rope.glsl) | 1.1 | 1.3 | Apply rotary position embeddings in place. |
| [transformer_dual_gemm_swiglu.glsl](shaders/glsl/transformer_dual_gemm_swiglu.glsl) | 1.1 | 1.3 | Fuse two cooperative matrix projections with SwiGLU. |
| [transformer_rms_norm.glsl](shaders/glsl/transformer_rms_norm.glsl) | 1.1 | 1.3 | Apply RMS normalization with optional residual addition. |
| [transformer_rope.glsl](shaders/glsl/transformer_rope.glsl) | 1.1 | 1.3 | Define rotary position embedding helpers. |
| [transformer_scale_dot_product.glsl](shaders/glsl/transformer_scale_dot_product.glsl) | 1.1 | 1.3 | Compute tiled scaled dot-product attention. |
| [transformer_scale_dot_product_cooperative.glsl](shaders/glsl/transformer_scale_dot_product_cooperative.glsl) | 1.1 / 1.3 | 1.3 / 1.6 | Compute attention with KHR or NV2 cooperative matrices. |
| [transformer_scale_dot_product_naive.glsl](shaders/glsl/transformer_scale_dot_product_naive.glsl) | 1.1 | 1.3 | Compute reference scaled dot-product attention. |
| [transformer_spatial_rms_norm_apply.glsl](shaders/glsl/transformer_spatial_rms_norm_apply.glsl) | 1.1 | 1.3 | Apply spatial RMS normalization from reduced sums. |
| [transformer_spatial_rms_norm_reduce.glsl](shaders/glsl/transformer_spatial_rms_norm_reduce.glsl) | 1.1 | 1.3 | Reduce partial spatial squared sums. |
| [transformer_spatial_rms_norm_sum_sq.glsl](shaders/glsl/transformer_spatial_rms_norm_sum_sq.glsl) | 1.1 | 1.3 | Accumulate partial spatial squared sums. |
| [transformer_swiglu.glsl](shaders/glsl/transformer_swiglu.glsl) | 1.1 | 1.3 | Apply the SwiGLU activation. |
| [value_head_pool_channels.glsl](shaders/glsl/value_head_pool_channels.glsl) | 1.1 | 1.3 | Pool channels for value-head outputs. |
| [winograd_input_transform.glsl](shaders/glsl/winograd_input_transform.glsl) | 1.1 | 1.3 | Transform input tiles into Winograd space. |
| [winograd_input_transform_bnact.glsl](shaders/glsl/winograd_input_transform_bnact.glsl) | 1.1 | 1.3 | Fuse normalization and activation with the Winograd input transform. |
| [winograd_output_transform.glsl](shaders/glsl/winograd_output_transform.glsl) | 1.1 | 1.3 | Transform Winograd results into spatial outputs. |
| [xgemm.glsl](shaders/glsl/xgemm.glsl) | 1.1 | 1.3 | Define tiled matrix multiplication helpers. |
| [xgemm_batched.glsl](shaders/glsl/xgemm_batched.glsl) | 1.1 | 1.3 | Multiply batched matrices. |
| [xgemm_direct.glsl](shaders/glsl/xgemm_direct.glsl) | 1.1 | 1.3 | Define direct matrix multiplication helpers. |
| [xgemm_direct_batched_tt.glsl](shaders/glsl/xgemm_direct_batched_tt.glsl) | 1.1 | 1.3 | Multiply transposed batched matrices directly. |
| [xgemm_direct_types.glsl](shaders/glsl/xgemm_direct_types.glsl) | 1.1 | 1.3 | Define precision-dependent direct GEMM vector types. |
| [xgemm_strided_batched_nn.glsl](shaders/glsl/xgemm_strided_batched_nn.glsl) | 1.1 | 1.3 | Multiply strided batches of non-transposed matrices. |

## Build and module selection

`cpp/CMakeLists.txt` assigns the baseline target to all ordinary `.comp` wrappers. It assigns the higher target only to `transformer_scale_dot_product_cooperative_nhwc_register_p32s16`. Its existing glslc compilation probe controls whether this optional NV2 variant is embedded. Header declarations remain guarded in C++ by Vulkan extension macros.

The KHR pair and NV2 register variants share `transformer_scale_dot_product_cooperative_nhwc.comp`. CMake defines `ATTENTION_USE_NV2=0` for pair and `1` for register. The separate block wrapper selects `ATTENTION_KV_BLOCK_SOFTMAX=1`. Unsupported NV capabilities are absent from KHR SPIR-V.

For each device, initialization enables only supported features/extensions. Module creation checks precision, matrix and subgroup requirements, then checks the SPIR-V header version against the effective device API. Unsupported variants retain null module handles. Pipeline selection and tuning additionally check matrix dimensions, subgroup size and resource limits. Old tuning caches are checked against current device capabilities before use.

The runtime SPIR-V core limits are 1.3 for Vulkan 1.1, 1.5 for Vulkan 1.2 and 1.6 for Vulkan 1.3+. Updating the shader target requires updating this document, the GLSL header and corresponding runtime eligibility together.

## Verification

Validate ordinary modules with `spirv-val --target-env vulkan1.1` and the NV2 register module with `spirv-val --target-env vulkan1.3`. Inspect binary header word 1 to confirm SPIR-V 1.3 and 1.6 respectively. The current wrapper inventory produces 330 ordinary modules and one optional NV2 module.

Compilation and SPIR-V validation establish toolchain compatibility. GPU correctness and performance require testing on the intended devices, including ARM Ubuntu devices.

## References

- [Vulkan API and SPIR-V version mapping](https://docs.vulkan.org/guide/latest/versions.html)
- [KHR Cooperative Matrix SPIR-V dependencies](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html)
- [Maintenance4 and LocalSizeId](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMaintenance4Features.html)
