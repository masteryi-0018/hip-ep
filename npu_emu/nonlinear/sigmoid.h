/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Sigmoid / tanh: quantized on the hardware tanh table, quantized on an interpolation table,
 * and direct bf16 / f16 / fp8 (E4M3FN) table variants. Elementwise on n contiguous elements.
 *
 * dq(x) = to_T(fma(x, dq_scale, fp32(-dq_zero_point * dq_scale)));
 * q(y) = sat(round(fma(y, q_scale, q_zero_point))), q_scale = 1 / s_out.
 * lut: 512 fp32 words, groups of 8 (offset, slope) pairs, each group stored twice (the first
 * copy is read). Index = floor(bf16(fma(x, k, 64))) clamped to [0, 127], k = 8 (sigmoid,
 * domain [-8, 8)) or 16 (tanh, [-4, 4)); LUT(x) = fma(x, slope[i], offset[i]), the slope
 * first rounded to f16 in the f16 variants.
 * fp8 is decoded in groups of 8 elements from the buffer start.
 */
#ifndef NPU_EMU_SIGMOID_H_
#define NPU_EMU_SIGMOID_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* v = dq(x) in compute_type (BF16 / F16); t = tanh(fp32(v * -0.5)) (hardware tanh table);
 * out = q(bf16(fma(t, -0.5, 0.5))), half away from zero.
 * a8: in_type / out_type I8 / U8; a16: I16 / U16. */
int npu_sigmoid_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                   npu_dtype out_type, int64_t n, npu_dtype compute_type,
                   float dq_zero_point, float dq_scale, float q_zero_point,
                   float q_scale);
/* The 16-bit variant of npu_sigmoid_a8. */
int npu_sigmoid_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                    npu_dtype out_type, int64_t n, npu_dtype compute_type,
                    float dq_zero_point, float dq_scale, float q_zero_point,
                    float q_scale);

/* out = q(tanh(dq(x))), hardware tanh table, half away from zero. Arguments as
 * npu_sigmoid_a8 / _a16. */
int npu_tanh_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                npu_dtype out_type, int64_t n, npu_dtype compute_type,
                float dq_zero_point, float dq_scale, float q_zero_point,
                float q_scale);
/* The 16-bit variant of npu_tanh_a8. */
int npu_tanh_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                 npu_dtype out_type, int64_t n, npu_dtype compute_type,
                 float dq_zero_point, float dq_scale, float q_zero_point,
                 float q_scale);

/* v = dq(x) in bf16; y = bf16(LUT(v)), 0 for v < -8 and 1 for v >= 8; out = q(y), half away
 * from zero. */
int npu_qdq_sigmoid_lut_a8(void* stream, const void* in, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n,
                           float dq_zero_point, float dq_scale,
                           float q_zero_point, float q_scale, const float* lut);
int npu_qdq_sigmoid_lut_a16(void* stream, const void* in, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n,
                            float dq_zero_point, float dq_scale,
                            float q_zero_point, float q_scale, const float* lut);

/* v = dq(x) in f16; y = f16(LUT(v)) clamped to [-1, 1]; out = q(y), half to even. */
int npu_qdq_tanh_lut_a8(void* stream, const void* in, npu_dtype in_type,
                        void* out, npu_dtype out_type, int64_t n,
                        float dq_zero_point, float dq_scale, float q_zero_point,
                        float q_scale, const float* lut);
int npu_qdq_tanh_lut_a16(void* stream, const void* in, npu_dtype in_type,
                         void* out, npu_dtype out_type, int64_t n,
                         float dq_zero_point, float dq_scale, float q_zero_point,
                         float q_scale, const float* lut);

/* type BF16 or F16 in and out. y = LUT(x) rounded half to even, 0 for x < -8, 1 for
 * x >= 8. */
int npu_elementwise_sigmoid_lut_bf16(void* stream, const void* in, void* out,
                                     npu_dtype type, int64_t n, const float* lut);

/* As the sigmoid one with the tanh domain: BF16 gives -1 for x < -4 and 1 for x >= 4, F16
 * clamps the interpolant to [-1, 1]. */
int npu_elementwise_tanh_lut_bf16(void* stream, const void* in, void* out,
                                  npu_dtype type, int64_t n, const float* lut);

/* fp8 in and out. v = fp32(x * ifm_scale); y = LUT(v) in fp32 clamped to [0, 1];
 * out = fp8(fp32(y * ofm_scale_recip)), half away, not saturated (above 448 -> NaN). */
int npu_elementwise_sigmoid_lut_fp8(void* stream, const void* in, void* out,
                                    int64_t n, const float* lut, float ifm_scale,
                                    float ofm_scale_recip);

/* fp8 in and out, compute_type BF16 / F16. v = compute_type(fp32(x * ifm_scale));
 * y = tanh LUT of v as npu_elementwise_tanh_lut_bf16, half away from zero;
 * out = fp8(clamp(y * compute_type(ofm_scale_recip), -448, 448)). */
int npu_elementwise_tanh_lut_fp8(void* stream, const void* in, void* out,
                                 int64_t n, npu_dtype compute_type,
                                 const float* lut, float ifm_scale,
                                 float ofm_scale_recip);

/* Test hook: out[i] = hardware tanh of the 16-bit code in[i] (BF16 / F16), as bf16 bits. */
int npu_test_sigmoid_tanh16(void* stream, const void* in, npu_dtype type,
                            void* out, int64_t n);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_SIGMOID_H_
