/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * exp / elu / hardsigmoid / sqrt on bf16 / fp16, fp8 (E4M3FN) and quantized
 * integers. Elementwise on n contiguous elements; fp8 inputs are decoded in
 * groups of 8 from the buffer start. Null buffers are rejected (also when n == 0);
 * n == 0 is a no-op.
 *
 * Notation: fp32(...) = one rounding (half to even) per mul / fused MAC;
 * exp2hw = the hardware exp2 table; log2e = 0x3fb8aa3b; t(...) = store to the
 * 16-bit float type, half to even; fp8_away / fp8_even = fp8 encode, half away
 * from zero / half to even, not saturated (above 448 -> NaN of that sign);
 * sat(rne(...)) = fp32 -> int half to even, saturated, NaN -> 0.
 */
#ifndef NPU_EMU_EXPSQRT_H_
#define NPU_EMU_EXPSQRT_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- bf16 / fp16 (in_type = BF16 or F16, the output has the same type) ---- */

/* out = t(exp2hw(fp32(log2e * x + 895.001))). */
int npu_elementwise_exp_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                             int64_t n);
/* x >= 0: out = t(x); else out = t(fp32(e * alpha - alpha)),
 * e = exp2hw(fp32(log2e * x + 895.001)). */
int npu_elementwise_elu_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                             int64_t n, float alpha);
/* out = t(min(max(fp32(x * alpha + beta), 0), 1)). */
int npu_elementwise_hardsigmoid_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                                     int64_t n, float alpha, float beta);
/* out = the hardware 16-bit sqrt of x; negative -> NaN. */
int npu_sqrt_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n);

/* ---- fp8 (E4M3FN in and out) ---- */

/* in_coeff = ifm_scale * log2(e), bias_coeff = 895.001 - log2(ofm_scale).
 * out = fp8_away(exp2hw(fp32(x * in_coeff + bias_coeff))). */
int npu_elementwise_exp_fp8(void* stream, const void* in, void* out, int64_t n,
                            float in_coeff, float bias_coeff);
/* positive_coeff = ifm_scale / ofm_scale, negative_coeff = alpha / ofm_scale.
 * r = fp32(x * ifm_scale); r >= 0: out = fp8_away(fp32(x * positive_coeff));
 * else out = fp8_away(fp32(e * negative_coeff - negative_coeff)),
 * e = exp2hw(fp32(log2e * r + 895.001)). */
int npu_elementwise_elu_fp8(void* stream, const void* in, void* out, int64_t n,
                            float ifm_scale, float positive_coeff, float negative_coeff);
/* output_slope = alpha * ifm_scale / ofm_scale, output_offset = beta / ofm_scale,
 * output_upper = 1 / ofm_scale.
 * out = fp8_away(min(max(fp32(x * output_slope + output_offset), 0), output_upper)). */
int npu_elementwise_hardsigmoid_fp8(void* stream, const void* in, void* out, int64_t n,
                                    float output_slope, float output_offset,
                                    float output_upper);
/* ifm_coeff = sqrt(ifm_scale) / ofm_scale.
 * y = fp16(fp16(invsqrt(x)) * x) for x > 0, x for x == 0, NaN for x < 0;
 * out = fp8_even(fp32(y * ifm_coeff)). */
int npu_elementwise_sqrt_fp8(void* stream, const void* in, void* out, int64_t n,
                             float ifm_coeff);

/* ---- quantized (a8: I8 / U8, a16: I16 / U16, in / out any combination) ---- */

/* c1 = s_in * log2(e), c0 = -s_in * zp_in * log2(e), c1_out = 1 / s_out, c0_out = zp_out.
 * p = min(fp32(x * c1 + c0), 127); e = exp2hw(fp32(p + 895.001)) with a bf16 result;
 * out = sat(rne(fp32(e * bf16(c1_out) + bf16(c0_out)))). */
int npu_elementwise_exp_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, float c1, float c0, float c1_out,
                           float c0_out);
/* The 16-bit variant of npu_elementwise_exp_a8. */
int npu_elementwise_exp_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, float c1, float c0, float c1_out,
                            float c0_out);
/* c1 = s_in / s_out, c0 = zp_out - c1 * zp_in, ifm_scale = s_in, neg_bias = -s_in * zp_in,
 * alpha_div_ofm_scale = alpha / s_out, neg_const = zp_out - alpha / s_out.
 * d = fp32(x * ifm_scale + neg_bias);
 * d > 0: v = fp32(x * c1 + c0);
 * else   v = fp32(e * alpha_div_ofm_scale + neg_const), e = exp2hw(fp32(log2e * d + 895.001));
 * out = sat(rne(v)). */
int npu_elementwise_elu_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, float c1, float c0, float ifm_scale,
                           float neg_bias, float alpha_div_ofm_scale, float neg_const);
/* The 16-bit variant of npu_elementwise_elu_a8. */
int npu_elementwise_elu_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, float c1, float c0,
                            float ifm_scale, float neg_bias, float alpha_div_ofm_scale,
                            float neg_const);
/* alpha_f = alpha * s_in / s_out, beta_f = beta / s_out + zp_out - alpha_f * zp_in,
 * lower_bound_i = floor(zp_out), upper_bound_i = floor(1 / s_out + zp_out), clipped to the
 * output range (-1 for a bound outside it).
 * out = sat(rne(min(max(fp32(x * alpha_f + beta_f), lower_bound_i), upper_bound_i))). */
int npu_hardsigmoid_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                       npu_dtype out_type, int64_t n, float alpha_f, float beta_f,
                       int32_t lower_bound_i, int32_t upper_bound_i);
/* The 16-bit variant of npu_hardsigmoid_a8. */
int npu_hardsigmoid_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                        npu_dtype out_type, int64_t n, float alpha_f, float beta_f,
                        int32_t lower_bound_i, int32_t upper_bound_i);
/* dq_zp = zp_in, dq_sc = s_in, q_zp = zp_out, q_sc = 1 / s_out.
 * h = fp16(fp32(x * dq_sc + fp32(-dq_zp * dq_sc))); s = hardware fp16 sqrt of h;
 * out = sat(rne(fp32(s * q_sc + q_zp))); h < 0 -> 0. */
int npu_qdq_sqrt_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                    npu_dtype out_type, int64_t n, float dq_zp, float dq_sc, float q_zp,
                    float q_sc);
/* The 16-bit variant of npu_qdq_sqrt_a8. */
int npu_qdq_sqrt_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                     npu_dtype out_type, int64_t n, float dq_zp, float dq_sc, float q_zp,
                     float q_sc);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_EXPSQRT_H_
