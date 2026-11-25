/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Bilinear and bicubic resize entry points.
 */
#ifndef NPU_EMU_RESIZE_H_
#define NPU_EMU_RESIZE_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The entry points take precomputed source indices and interpolation weights.
 * in [Hi][Wi][C], out [Ho][Wo][C]; idx_h[Ho] / idx_w[Wo] are absolute uint16 source
 * indices. Reads past the tensor are clamped to the edge. Hi, Wi, Ho, Wo <= 65535.
 *
 * Bilinear 4-tap: s[h][w][0..3] weight (y0, x0), (y0, x1), (y1, x0), (y1, x1),
 * y0 = idx_h[h], x0 = idx_w[w]; x1 = x0 + 1 if s[h][w][1] != 0, else x0;
 * y1 = y0 + 1 if s[h][w0][2] != 0, else y0, with w0 = w - w % sv_wo (sv_wo == 0
 * means Wo). acc = x00 * s0, then three MACs with s1, s2, s3 in that order.
 * Hi, Wi >= 1; Ho, Wo or C 0: nothing to do, returns 0 once the other arguments are valid.
 * Null pointers (also for empty work), negative sizes or sv_wo return -1. */

/* Weight layout of the 8-bit integer bilinear kernel. */
typedef enum {
  NPU_RESIZE_4TAP = 0,
  NPU_RESIZE_SEPARABLE = 1,
} npu_resize_route;

/* in / out of in_type (I8 / U8); exact integer accumulator. wts uint16:
 * NPU_RESIZE_4TAP: wts = s[Ho][Wo][4] Q15; Q = 15.
 * NPU_RESIZE_SEPARABLE: wts = qy[Ho][2] then qx[Wo][2]; sv_wo unused.
 *   tmp(x) = sat_int16(in[y0][x] * qy0 + in[y1][x] * qy1), y1 = y0 + (qy1 != 0);
 *   acc = tmp(x0) * qx0 + tmp(x0 + 1) * qx1; Q = 15 for I8, 14 for U8.
 * requantize == 0: out = sat(sign(acc) * (|acc| >> Q));
 * requantize != 0: out = sat(trunc(fp32(acc * c0 + fp32(c1 * 2^Q)) * 2^-Q)),
 *   c0 = s_in / s_out, c1 = zp_out - c0 * zp_in. */
int npu_resize_bilinear(void* stream, const void* in, npu_dtype in_type, const void* idx_h,
                        const void* idx_w, const void* wts, void* out, int64_t Hi,
                        int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                        npu_resize_route route, int64_t sv_wo, int requantize, float c0,
                        float c1);

/* in_type = I16 / U16: the 4-tap route of npu_resize_bilinear. */
int npu_resize_bilinear_a16(void* stream, const void* in, npu_dtype in_type, const void* idx_h,
                            const void* idx_w, const void* wts, void* out, int64_t Hi,
                            int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, int64_t sv_wo,
                            int requantize, float c0, float c1);

/* in, wts = s[Ho][Wo][4] and out of in_type (BF16 / F16). 4-tap in fp32, each MAC fused;
 * the result is truncated toward zero to in_type. Results below the smallest normal become
 * a signed zero; F16 magnitudes >= 2^16 saturate to 65504. */
int npu_resize_bilinear_bf16(void* stream, const void* in, npu_dtype in_type,
                             const void* idx_h, const void* idx_w, const void* wts, void* out,
                             int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                             int64_t sv_wo);

/* in / out fp8 (E4M3FN); wts = s[Ho][Wo][4] of wts_type (BF16 / F16), also the interior
 * type T. x = trunc_T(fp32(decode(in) * ifm_dequant_coeff)), decoded by rows of 8
 * channels (C % 8 == 0); 4-tap as above; out = fp8(fp32(acc * ofm_quant_coeff))
 * truncated toward zero, saturated to +-448. ofm_quant_coeff = 1 / ofm_scale. */
int npu_resize_bilinear_fp8(void* stream, const void* in, const void* idx_h,
                            const void* idx_w, const void* wts, npu_dtype wts_type, void* out,
                            int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                            int64_t sv_wo, float ifm_dequant_coeff, float ofm_quant_coeff);

/* BF16 only. idx_h[Ho] / idx_w[Wo] are the starts of each output's 4-sample
 * window; coef_h[Ho][4] / coef_w[Wo][4] are bf16 weights.
 *   t[h][x]   = rne(sum_k coef_h[h][k] * in[idx_h[h] + k][x]);
 *   out[h][w] = rne(sum_k coef_w[w][k] * t[h][idx_w[w] + k]);
 * each sum is a rounded product then three fused MACs in k order. Results below
 * the smallest normal become a signed zero. */
int npu_resize_cubic_bf16(void* stream, const void* in, const void* idx_h, const void* idx_w,
                          const void* coef_h, const void* coef_w, void* out, int64_t Hi,
                          int64_t Wi, int64_t C, int64_t Ho, int64_t Wo);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_RESIZE_H_
