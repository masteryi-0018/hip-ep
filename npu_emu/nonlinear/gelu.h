/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * GeLU / Erf on bf16 / fp16, bfp16, fp8 (E4M3FN) and quantized integers. Elementwise on n
 * contiguous elements; fp8 is decoded in groups of 8 and bfp16 in groups of 64 elements from
 * the buffer start. Null buffers are rejected (also when n == 0); n == 0 is a no-op.
 *
 * fp32(...) = one rounding per mul / add / fused MAC; t(...) = store to the 16-bit type.
 * GeLU: xa = -|x|, v = fp32(0.750399888 * xa - 0.729944110), p = fp32(894.533813 - v * v),
 *   e = exp2hw(p) (the hardware exp2 table, subnormal -> +0), g = fp32(xa * e + max(x, -0)).
 * Erf: x2 = t(x*x), x3 = t(x2*x), x4 = t(x2*x2), x5 = t(x4*x) (half to even),
 *   a = fp32(fp32(fp32(1.125*x) - 0.376953125*x3) + 0.11279296875*x5), r = t(a) clamped to
 *   [-1, 1].
 */
#ifndef NPU_EMU_GELU_H_
#define NPU_EMU_GELU_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* in_type BF16 or F16 (also the output type). out = t(g), half away from zero; bf16
 * subnormal results -> signed zero. NaN in -> negative NaN, +-inf in -> positive NaN. */
int npu_gelu_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n);

/* bfp16 in and out, n % 512 == 0: n / 8 blocks of 9 bytes [exponent][8 x int8 mantissa],
 * value = m * 2^(e - 133). Each group of 64 elements is aligned to its largest exponent
 * (24-bit window, half to even), GeLU as npu_gelu_bf16 in bf16, re-encoded half away from
 * zero. */
int npu_gelu_bfp16(void* stream, const void* in, void* out, int64_t n);

/* fp8 in and out; t = compute_type, BF16 or F16.
 * x = t(fp32(v * ifm_scale)) (half away), y = GeLU of x as npu_gelu_bf16,
 * out = fp8(clamp(fp32(y * t(ofm_scale_recip)), +-448)), half away; NaN y -> fp8 NaN of its
 * sign. */
int npu_elementwise_gelu_fp8(void* stream, const void* in, void* out, int64_t n,
                             npu_dtype compute_type, float ifm_scale, float ofm_scale_recip);

/* in_type / out_type: I8 / U8 (a8) or I16 / U16 (a16).
 * dq_zp = ifm_zero_point, q_zp = ofm_zero_point, ka = -0.750399888 * ifm_scale,
 * c = ifm_scale / ofm_scale, kdsp = 894.533813 + log2(c). With qi = q - dq_zp:
 * v = fp32(|qi| * ka - 0.729944110), p = fp32(kdsp - v * v), e = exp2hw(p),
 * lo = fp32(q_zp - |qi| * e), out = sat(fp32(max(qi, 0) * c + lo)), half away from zero. */
int npu_qdq_gelu_exp2_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                         npu_dtype out_type, int64_t n, int32_t dq_zp, float q_zp, float ka,
                         float c, float kdsp);
/* The 16-bit variant of npu_qdq_gelu_exp2_a8. */
int npu_qdq_gelu_exp2_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                          npu_dtype out_type, int64_t n, int32_t dq_zp, float q_zp, float ka,
                          float c, float kdsp);

/* int8 in and out. rnd_id: 0 ceil, 1 nearest with ties toward zero, 2 away from zero;
 * srs = rounding right shift, saturated to [-127, 127].
 * A = q << a_shift, q4 = srs(A, q4_shift), R = max(srs(A, r_shift), 0),
 * B = q when up_path && a_shift == 1, else srs(A, b_shift);
 * m = max(K - |q4|, 0), h = min(P, m);
 * precision_enhanced (only when up_path && a_shift == 1):
 *   h = min(max(h, max(K + 1 - |q4|, 0) - m), |q4| + max(P - 2, 0));
 * out = max(B, R - h) in wrapping int8. Returns -1 unless rnd_id <= 2, the shifts are in
 * [0, 23] and precision_enhanced is 0 or 1. */
int npu_xint8_gelu(void* stream, const void* in, void* out, int64_t n, uint8_t up_path,
                   uint8_t rnd_id, uint8_t a_shift, uint8_t b_shift, uint8_t q4_shift,
                   uint8_t r_shift, int8_t K, int8_t P, int precision_enhanced);

/* in_type BF16 or F16 (also the output type). out = the Erf polynomial in t = in_type; bf16
 * subnormal inputs read as a signed zero. */
int npu_elementwise_erf_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                             int64_t n);

/* fp8 in and out. ifm_coeff = ifm_scale / ofm_scale, q3_coeff = -0.376953125 * ofm_scale^2,
 * q5_coeff = 0.11279296875 * ofm_scale^4, clamp_coeff = 1 / ofm_scale.
 * x = fp32(v * ifm_coeff), x2 = fp32(x * x), p35 = fp32(x2 * q5_coeff + q3_coeff),
 * p135 = fp32(x2 * p35 + 1.125), r = clamp(fp32(x * p135), +-clamp_coeff);
 * out = fp8(r), half away, not saturated (above 448 -> NaN). */
int npu_elementwise_erf_fp8(void* stream, const void* in, void* out, int64_t n,
                            float ifm_coeff, float q3_coeff, float q5_coeff, float clamp_coeff);

/* in_type / out_type: I8 / U8 (a8) or I16 / U16 (a16).
 * c1 = ifm_scale, c0 = -ifm_scale * ifm_zero_point, c1_out = 1 / ofm_scale,
 * c0_out = ofm_zero_point.
 * x = bf16(fp32(q * c1 + c0)), r = the Erf polynomial in bf16,
 * out = sat(rne(fp32(r * bf16(c1_out) + bf16(c0_out)))); NaN -> 0. */
int npu_elementwise_erf_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, float c1, float c0, float c1_out,
                           float c0_out);
/* The 16-bit variant of npu_elementwise_erf_a8. */
int npu_elementwise_erf_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, float c1, float c0, float c1_out,
                            float c0_out);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_GELU_H_
