/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Sine / cosine on bf16 / fp16, fp8 (E4M3FN) and quantized integers, as a polynomial and
 * as a table (LUT) variant. Elementwise on n contiguous elements; fp8 is decoded in groups of
 * 8 elements from the buffer start.
 *
 * fp32(...) = one rounding per mul / fused MAC. T = the compute type (BF16 / F16).
 * fp8 out = fp8(clamp(fp32(y * T(ofm_scale_recip)), +-448)).
 *
 * Polynomial core, bf16 (tp = bf16(2 / pi), pi2 = bf16(pi / 2), s0..s3 / c0..c3 = bf16 of
 * the sin / cos polynomial coefficients; subnormal x -> signed zero):
 *   z  = sat_int16(floor(bf16(fp32(x * tp + 0.5))));  zb = bf16(z)
 *   xa = fp32(x - pi2 * zb); xr = bf16(xa); r = bf16(xr * xr); r2 = bf16(r * r)
 *   sin = bf16(fp32(xa + bf16(r * xr) * bf16(fp32(r2 * bf16(fp32(s2 + s3 r))
 *                                                  + fp32(s0 + s1 r)))))
 *   cos = bf16(fp32(fp32(1 - 0.5 r) + r2 * bf16(fp32(r2 * bf16(fp32(c2 + c3 r))
 *                                                    + fp32(c0 + c1 r)))))
 * Polynomial core, fp16 (inv_pi_2 = fp16(0.6367f), pio2_hi = fp16(1.57f),
 * pio2_lo = fp16(0.0004838f), c3 / c5 / c7 = fp16(-0.1666f / 0.00833f / -0.0001984f),
 * c2 / c4 / c6 = fp16(-0.5f / 0.04166f / -0.001389f)):
 *   z = sat_int16(rne(fp32(x * inv_pi_2))); kh = fp16(z)
 *   r = fp16(fp32(fp32(x - pio2_hi kh) - pio2_lo kh)); r2 = fp16(r * r)
 *   sin = fp16(r * fp16(1 + r2 fp16(c3 + r2 fp16(c5 + r2 c7))))
 *   cos = fp16(1 + r2 fp16(c2 + r2 fp16(c4 + r2 c6)))
 * Quadrant: sin entries map z % 4 = 0, 1, 2, 3 to sin, cos, -sin, -cos; cos entries to
 * cos, -sin, -cos, sin.
 *
 * LUT core: lut = 2048 fp32 words (device) [64 groups][2 copies][8][offset, slope]; entry i
 * has its offset at lut[(i / 8) * 32 + (i % 8) * 2] and its slope next to it.
 *   ix = fp32(x * 0.31830987f); i = sat_int32(floor(256 ix)) & 511;
 *   fl = float(sat_int32(floor(ix))) (rounded toward -inf); rem = fp32(ix - fl);
 *   y = fp32(offset[i] + rem * slope[i]).
 * Sine and cosine differ only in their table. Null buffers and tables are rejected (also
 * when n == 0); n == 0 is a no-op.
 */
#ifndef NPU_EMU_SINCOS_H_
#define NPU_EMU_SINCOS_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* in_type BF16 or F16 (also the output type); the polynomial core, half to even. */
int npu_sin_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n);
/* The cosine variant of npu_sin_bf16. */
int npu_cos_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n);

/* fp8 in and out, T = compute_type. v = T(fp32(x * ifm_scale)); y = core(v); half to even.
 * ofm_scale_recip = 1 / ofm_scale. */
int npu_elementwise_sin_fp8(void* stream, const void* in, void* out, int64_t n,
                            npu_dtype compute_type, float ifm_scale, float ofm_scale_recip);
/* The cosine variant of npu_elementwise_sin_fp8. */
int npu_elementwise_cos_fp8(void* stream, const void* in, void* out, int64_t n,
                            npu_dtype compute_type, float ifm_scale, float ofm_scale_recip);

/* in_type BF16 or F16 (also the output type). out = in_type(LUT(x)), half to even. */
int npu_elementwise_sin_lut_bf16(void* stream, const void* in, npu_dtype in_type,
                                 const void* lut, void* out, int64_t n);
/* The cosine variant of npu_elementwise_sin_lut_bf16. */
int npu_elementwise_cos_lut_bf16(void* stream, const void* in, npu_dtype in_type,
                                 const void* lut, void* out, int64_t n);

/* fp8 in and out, T = compute_type. v = T(fp32(x * ifm_scale)); y = T(LUT(v)); conversions
 * round half away from zero. */
int npu_elementwise_sin_lut_fp8(void* stream, const void* in, const void* lut, void* out,
                                int64_t n, npu_dtype compute_type, float ifm_scale,
                                float ofm_scale_recip);
/* The cosine variant of npu_elementwise_sin_lut_fp8. */
int npu_elementwise_cos_lut_fp8(void* stream, const void* in, const void* lut, void* out,
                                int64_t n, npu_dtype compute_type, float ifm_scale,
                                float ofm_scale_recip);

/* in_type / out_type: I8 / U8 (a8) or I16 / U16 (a16); T = compute_type, BF16 / F16.
 * dq_zp = ifm_zero_point, dq_sc = ifm_scale, q_zp = ofm_zero_point, q_sc = 1 / ofm_scale.
 * v = T(fp32(q * dq_sc + fp32(-dq_zp * dq_sc))); y = core(v);
 * out = sat(rne(fp32(y * q_sc + q_zp))). */
int npu_elementwise_sin_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                           float dq_sc, float q_zp, float q_sc);
/* The 16-bit variant of npu_elementwise_sin_a8. */
int npu_elementwise_sin_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                            float dq_sc, float q_zp, float q_sc);
/* The cosine variant of npu_elementwise_sin_a8. */
int npu_elementwise_cos_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                           float dq_sc, float q_zp, float q_sc);
/* The 16-bit variant of npu_elementwise_cos_a8. */
int npu_elementwise_cos_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                            float dq_sc, float q_zp, float q_sc);

/* As above with the LUT core in fp32 (the dequantized value is not rounded):
 * v = fp32(q * dq_sc + fp32(-dq_zp * dq_sc)); y = LUT(v);
 * out = sat(rne(fp32(y * q_sc + q_zp))). */
int npu_elementwise_sin_lut_a8(void* stream, const void* in, npu_dtype in_type,
                               const void* lut, void* out, npu_dtype out_type, int64_t n,
                               float dq_zp, float dq_sc, float q_zp, float q_sc);
/* The 16-bit variant of npu_elementwise_sin_lut_a8. */
int npu_elementwise_sin_lut_a16(void* stream, const void* in, npu_dtype in_type,
                                const void* lut, void* out, npu_dtype out_type, int64_t n,
                                float dq_zp, float dq_sc, float q_zp, float q_sc);
/* The cosine variant of npu_elementwise_sin_lut_a8. */
int npu_elementwise_cos_lut_a8(void* stream, const void* in, npu_dtype in_type,
                               const void* lut, void* out, npu_dtype out_type, int64_t n,
                               float dq_zp, float dq_sc, float q_zp, float q_sc);
/* The 16-bit variant of npu_elementwise_cos_lut_a8. */
int npu_elementwise_cos_lut_a16(void* stream, const void* in, npu_dtype in_type,
                                const void* lut, void* out, npu_dtype out_type, int64_t n,
                                float dq_zp, float dq_sc, float q_zp, float q_sc);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_SINCOS_H_
