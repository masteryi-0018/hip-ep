/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * exp / elu / hardsigmoid / sqrt on bf16 / fp16, fp8 (E4M3FN) and quantized
 * integers. Elementwise on n contiguous elements; fp8 inputs are decoded in
 * groups of 8 from the buffer start.
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

/* ---- bf16 / fp16 (type = BF16 or F16, same type in and out) ---- */

/* out = t(exp2hw(fp32(log2e * x + 895.001))). */
int npu_elementwise_exp_bf16(void* stream, const void* in, npu_dtype type, void* out,
                             int64_t n);
/* x >= 0: out = t(x); else out = t(fp32(e * alpha - alpha)),
 * e = exp2hw(fp32(log2e * x + 895.001)). */
int npu_elementwise_elu_bf16(void* stream, const void* in, npu_dtype type, void* out,
                             int64_t n, float alpha);
/* out = t(min(max(fp32(x * alpha + beta), 0), 1)). */
int npu_elementwise_hardsigmoid_bf16(void* stream, const void* in, npu_dtype type, void* out,
                                     int64_t n, float alpha, float beta);
/* out = the hardware 16-bit sqrt of x; negative -> NaN. */
int npu_sqrt_bf16(void* stream, const void* in, npu_dtype type, void* out, int64_t n);

/* ---- fp8 (E4M3FN in and out) ---- */

/* in_coeff = ifm_scale * log2(e), bias = 895.001 - log2(ofm_scale).
 * out = fp8_away(exp2hw(fp32(x * in_coeff + bias))). */
int npu_elementwise_exp_fp8(void* stream, const void* in, void* out, int64_t n,
                            float in_coeff, float bias);
/* in_scale = ifm_scale, pos_scale = ifm_scale / ofm_scale, neg_scale = alpha / ofm_scale.
 * r = fp32(x * in_scale); r >= 0: out = fp8_away(fp32(x * pos_scale));
 * else out = fp8_away(fp32(e * neg_scale - neg_scale)),
 * e = exp2hw(fp32(log2e * r + 895.001)). */
int npu_elementwise_elu_fp8(void* stream, const void* in, void* out, int64_t n,
                            float in_scale, float pos_scale, float neg_scale);
/* slope = alpha * ifm_scale / ofm_scale, offset = beta / ofm_scale, upper = 1 / ofm_scale.
 * out = fp8_away(min(max(fp32(x * slope + offset), 0), upper)). */
int npu_elementwise_hardsigmoid_fp8(void* stream, const void* in, void* out, int64_t n,
                                    float slope, float offset, float upper);
/* scale = sqrt(ifm_scale) / ofm_scale.
 * y = fp16(fp16(invsqrt(x)) * x) for x > 0, x for x == 0, NaN for x < 0;
 * out = fp8_even(fp32(y * scale)). */
int npu_elementwise_sqrt_fp8(void* stream, const void* in, void* out, int64_t n, float scale);

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
 * lower = floor(zp_out), upper = floor(1 / s_out + zp_out), both within the output range.
 * out = sat(rne(min(max(fp32(x * alpha_f + beta_f), lower), upper))). */
int npu_hardsigmoid_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                       npu_dtype out_type, int64_t n, float alpha_f, float beta_f,
                       int32_t lower, int32_t upper);
/* The 16-bit variant of npu_hardsigmoid_a8. */
int npu_hardsigmoid_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                        npu_dtype out_type, int64_t n, float alpha_f, float beta_f,
                        int32_t lower, int32_t upper);
/* dq_zp = zp_in, dq_scale = s_in, q_zp = zp_out, q_scale = 1 / s_out.
 * h = fp16(fp32(x * dq_scale + fp32(-dq_zp * dq_scale))); s = hardware fp16 sqrt of h;
 * out = sat(rne(fp32(s * q_scale + q_zp))); h < 0 -> 0. */
int npu_qdq_sqrt_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                    npu_dtype out_type, int64_t n, float dq_scale, float dq_zp, float q_scale,
                    float q_zp);
int npu_qdq_sqrt_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                     npu_dtype out_type, int64_t n, float dq_scale, float dq_zp,
                     float q_scale, float q_zp);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_EXPSQRT_H_
