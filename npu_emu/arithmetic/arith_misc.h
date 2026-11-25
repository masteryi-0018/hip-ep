/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Quantized elementwise division, leaky relu, prelu, square and batchnorm (a8 / a16).
 */
#ifndef NPU_EMU_ARITH_MISC_H_
#define NPU_EMU_ARITH_MISC_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Common rules. mac(x, c, a) = x * c + a rounded once to fp32 (half to even); an integer x
 * enters exactly. sat(rne(v)) rounds half to even and saturates to the output type; NaN -> 0.
 * Division is a * inv(b) with a reciprocal that depends on the element type:
 *   bf16: rne(1 / b); zero / subnormal b -> +-inf, |b| > 2^126 -> +-0;
 *   f16:  rne(1 / b), subnormals kept;
 *   fp32: the hardware reciprocal table (within 1 ulp of 1 / b).
 * Coefficients are fp32; s_x / zp_x are scales and zero points.
 * Tensors are dense: flat, [rows][C] or [H][W][C]; every size >= 0, an empty tensor is a
 * no-op. Types: I8 / U8 (a8), I16 / U16 (a16). Pointers may not be null. */

/* a and b share in_type; out_type may differ in sign. The zero points are passed biased:
 *   zero_point_coeff_1 = 12582912.0f + zp_a, zero_point_coeff_2 = 12582912.0f + zp_b
 *   da = bf16_rne((12582912.0f + a) - zero_point_coeff_1),  db likewise with b
 *   q  = da * inv_bf16(db);  out = sat(rne(fma(q, c0, c1)))
 * c0 = s_a / (s_b * s_out), c1 = zp_out. */
int npu_elementwise_div_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float zero_point_coeff_1, float zero_point_coeff_2);
/* The 16-bit variant of npu_elementwise_div_a8. */
int npu_elementwise_div_a16(void* stream, const void* a, npu_dtype in_type, const void* b,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float zero_point_coeff_1, float zero_point_coeff_2);

/* a has the full shape shape1 = {H, W, C}, b has shape2 (host arrays; each dimension 1 or
 * shape1's) and is broadcast to it; a, b and out share dtype. With zp1 / zp2 =
 * zero_point_coeff_1 / 2, da = a - zp1, db = b - zp2 in int32:
 *   a8:  da, db -> f16 (rne); db == 0 is replaced by f16(7e-5); q = da * inv_f16(db)
 *   a16: da, db -> fp32; db == 0 is replaced by 7e-5f; q = fp32(da * inv_f32(db))
 *   out = sat(rne(fma(q, ifm1_coeff, ifm2_coeff) * 2^-output_sft))
 * ifm1_coeff = floor(s_a / (s_b * s_out) * 2^15), ifm2_coeff = floor(zp_out * 2^15),
 * output_sft = 15 (accepted in [-64, 64]). a16 requires zp2 within dtype's range (else -1). */
int npu_qdq_elementwise_broadcasting3d_div_a8(void* stream, const void* a, npu_dtype dtype,
    const void* b, void* out, const int64_t* shape1, const int64_t* shape2, float ifm1_coeff,
    float ifm2_coeff, int32_t zero_point_coeff_1, int32_t zero_point_coeff_2, int output_sft);
int npu_qdq_elementwise_broadcasting3d_div_a16(void* stream, const void* a, npu_dtype dtype,
    const void* b, void* out, const int64_t* shape1, const int64_t* shape2, float ifm1_coeff,
    float ifm2_coeff, int32_t zero_point_coeff_1, int32_t zero_point_coeff_2, int output_sft);

/* activations */

/* out = sat(rne(max(mac(x, c1, c0), mac(x, c1_alpha, c0_alpha)))) with c1 = s_in / s_out,
 * c0 = zp_out - c1 * zp_in, c1_alpha = alpha * c1, c0_alpha = alpha * (c0 - zp_out) + zp_out. */
int npu_elementwise_leakyrelu_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                                 npu_dtype out_type, int64_t n, float c1, float c0,
                                 float c1_alpha, float c0_alpha);
/* The 16-bit variant of npu_elementwise_leakyrelu_a8. */
int npu_elementwise_leakyrelu_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                                  npu_dtype out_type, int64_t n, float c1, float c0,
                                  float c1_alpha, float c0_alpha);

/* in and out are [rows][C] (C >= 1); prelu_scaled_slope = device array of C
 * floats:
 *   d = x - ifm_zero_point;  s = d < 0 ? prelu_scaled_slope[c] : positive_scale
 *   out = sat(rne(mac(d, s, ofm_zero_point)))
 * positive_scale = s_in / s_out, prelu_scaled_slope[c] = slope[c] * positive_scale. */
int npu_elementwise_prelu_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                             npu_dtype out_type, int64_t rows, int64_t C, float positive_scale,
                             int32_t ifm_zero_point, float ofm_zero_point,
                             const void* prelu_scaled_slope);
/* The 16-bit variant of npu_elementwise_prelu_a8. */
int npu_elementwise_prelu_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                              npu_dtype out_type, int64_t rows, int64_t C, float positive_scale,
                              int32_t ifm_zero_point, float ofm_zero_point,
                              const void* prelu_scaled_slope);

/* k * (x - zp_in)^2 + zp_out: t = mac(x, a, b); out = sat(rne(mac(x, t, c))) with a = k,
 * b = -2 * k * zp_in, c = k * zp_in^2 + zp_out, k = s_in^2 / s_out. */
int npu_elementwise_square_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                              npu_dtype out_type, int64_t n, float a, float b, float c);
/* The 16-bit variant of npu_elementwise_square_a8. */
int npu_elementwise_square_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                               npu_dtype out_type, int64_t n, float a, float b, float c);

/* batch normalization */

/* in and out are [rows][C] of dtype (C >= 1); cw and cb = device arrays of C
 * floats; zp = ofm_zero_point:
 *   o = mac(x, cw[c], cb[c])
 *   activation 1 (relu):       o = max(o, zp)
 *   activation 3 (leaky relu): lc = fp32(zp * fp32(1 - relu_alpha));
 *                              o = max(o, fma(o, relu_alpha, lc))
 *   out = sat(rne(o)); activation 0: none. */
int npu_qdq_batchnorm2d_stream_a8(void* stream, const void* in, npu_dtype dtype, void* out,
                                  int64_t rows, int64_t C, const void* cw, const void* cb,
                                  float ofm_zero_point, int activation, float relu_alpha);
int npu_qdq_batchnorm2d_stream_a16(void* stream, const void* in, npu_dtype dtype, void* out,
                                   int64_t rows, int64_t C, const void* cw, const void* cb,
                                   float ofm_zero_point, int activation, float relu_alpha);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_ARITH_MISC_H_
