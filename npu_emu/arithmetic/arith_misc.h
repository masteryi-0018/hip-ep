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
 * Tensors are dense: flat, [rows][C] or [H][W][C]. Types: I8 / U8 (a8), I16 / U16 (a16). */

/* a and b share in_type; out_type may differ in sign. The zero points are passed biased:
 *   zp1_word = 12582912.0f + zp_a, zp2_word = 12582912.0f + zp_b
 *   da = bf16_rne((12582912.0f + a) - zp1_word),  db likewise with b
 *   q  = da * inv_bf16(db);  out = sat(rne(fma(q, c0, c1)))
 * c0 = s_a / (s_b * s_out), c1 = zp_out. */
int npu_elementwise_div_a8(void* stream, const void* a, const void* b,
                           npu_dtype in_type, void* out, npu_dtype out_type,
                           int64_t n, float c0, float c1, float zp1_word,
                           float zp2_word);
/* The 16-bit variant of npu_elementwise_div_a8. */
int npu_elementwise_div_a16(void* stream, const void* a, const void* b,
                            npu_dtype in_type, void* out, npu_dtype out_type,
                            int64_t n, float c0, float c1, float zp1_word,
                            float zp2_word);

/* a has the full shape shape1 = {H, W, C}, b has shape2 (host arrays; each dimension 1 or
 * shape1's) and is broadcast to it; a, b and out share dtype. With da = a - zp1, db = b - zp2
 * in int32:
 *   a8:  da, db -> f16 (rne); db == 0 is replaced by f16(7e-5); q = da * inv_f16(db)
 *   a16: da, db -> fp32; db == 0 is replaced by 7e-5f; q = fp32(da * inv_f32(db))
 *   out = sat(rne(fma(q, c0, c1) * 2^-output_shift))
 * c0 = floor(s_a / (s_b * s_out) * 2^15), c1 = floor(zp_out * 2^15), output_shift = 15.
 * a16 requires zp2 within dtype's range (else -1). */
int npu_qdq_elementwise_broadcasting3d_div_a8(void* stream, const void* a, const void* b,
    npu_dtype dtype, void* out, const int64_t* shape1, const int64_t* shape2, float c0, float c1,
    int32_t zp1, int32_t zp2, int output_shift);
int npu_qdq_elementwise_broadcasting3d_div_a16(void* stream, const void* a, const void* b,
    npu_dtype dtype, void* out, const int64_t* shape1, const int64_t* shape2, float c0, float c1,
    int32_t zp1, int32_t zp2, int output_shift);

/* activations */

/* out = sat(rne(max(mac(x, c1, c0), mac(x, c1_alpha, c0_alpha)))) with c1 = s_in / s_out,
 * c0 = zp_out - c1 * zp_in, c1_alpha = alpha * c1, c0_alpha = alpha * (c0 - zp_out) + zp_out. */
int npu_elementwise_leakyrelu_a8(void* stream, const void* in,
                                 npu_dtype in_type, void* out,
                                 npu_dtype out_type, int64_t n, float c1,
                                 float c0, float c1_alpha, float c0_alpha);
/* The 16-bit variant of npu_elementwise_leakyrelu_a8. */
int npu_elementwise_leakyrelu_a16(void* stream, const void* in,
                                  npu_dtype in_type, void* out,
                                  npu_dtype out_type, int64_t n, float c1,
                                  float c0, float c1_alpha, float c0_alpha);

/* in and out are [rows][C]; negative_scale = device array of C floats:
 *   d = x - in_zp;  s = d < 0 ? negative_scale[c] : positive_scale
 *   out = sat(rne(mac(d, s, out_zp)))
 * positive_scale = s_in / s_out, negative_scale[c] = slope[c] * positive_scale. */
int npu_elementwise_prelu_a8(void* stream, const void* in, npu_dtype in_type,
                             void* out, npu_dtype out_type, int64_t rows,
                             int64_t C, float positive_scale, int32_t in_zp,
                             float out_zp, const float* negative_scale);
/* The 16-bit variant of npu_elementwise_prelu_a8. */
int npu_elementwise_prelu_a16(void* stream, const void* in, npu_dtype in_type,
                              void* out, npu_dtype out_type, int64_t rows,
                              int64_t C, float positive_scale, int32_t in_zp,
                              float out_zp, const float* negative_scale);

/* k * (x - zp_in)^2 + zp_out: t = mac(x, a, b); out = sat(rne(mac(x, t, c))) with a = k,
 * b = -2 * k * zp_in, c = k * zp_in^2 + zp_out, k = s_in^2 / s_out. */
int npu_elementwise_square_a8(void* stream, const void* in, npu_dtype in_type,
                              void* out, npu_dtype out_type, int64_t n,
                              float a, float b, float c);
/* The 16-bit variant of npu_elementwise_square_a8. */
int npu_elementwise_square_a16(void* stream, const void* in,
                               npu_dtype in_type, void* out,
                               npu_dtype out_type, int64_t n, float a, float b,
                               float c);

/* batch normalization */

typedef enum {
  NPU_QDQ_BN_NONE = 0,
  NPU_QDQ_BN_RELU = 1,
  NPU_QDQ_BN_LRELU = 3,
} npu_qdq_bn_activation;

/* in and out are [rows][C] of dtype; cw and cb = device arrays of C floats:
 *   o = mac(x, cw[c], cb[c])
 *   RELU:  o = max(o, ofm_zp)
 *   LRELU: lc = fp32(ofm_zp * fp32(1 - alpha)); o = max(o, fma(o, alpha, lc))
 *   out = sat(rne(o)) */
int npu_qdq_batchnorm2d_stream_a8(void* stream, const void* in,
                                  npu_dtype dtype, void* out, int64_t rows,
                                  int64_t C, const float* cw, const float* cb,
                                  npu_qdq_bn_activation act, float ofm_zp,
                                  float alpha);
int npu_qdq_batchnorm2d_stream_a16(void* stream, const void* in,
                                   npu_dtype dtype, void* out, int64_t rows,
                                   int64_t C, const float* cw, const float* cb,
                                   npu_qdq_bn_activation act, float ofm_zp,
                                   float alpha);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_ARITH_MISC_H_
