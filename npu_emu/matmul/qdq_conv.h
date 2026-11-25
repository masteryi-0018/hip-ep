/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * QDQ convolution int8 x int8 / int16 x int8, with a fused residual add, and QDQ depthwise
 * convolution. Arithmetic conventions, coefficients and activation codes as in
 * matmul/qdq_gemm.h; ifm H x W x Ci, wts Kh x Kw x Ci x Co, out Ho x Wo x Co with
 * Ho = (H - Kh) / stride_h + 1 (no padding; the caller pads). H or W = 0 is an empty output
 * (returns 0, nothing launched; required pointers must still be non-null), otherwise H >= Kh
 * and W >= Kw.
 *
 * Special values: as qdq_gemm.h; the float words (c0 / c1 / c2, act0 / act1, slope, add words)
 * follow IEEE, the output conversion maps NaN to 0 and saturates the rest [default]. zp_w is
 * a weight zero point in the range of the weight type.
 */
#ifndef NPU_EMU_QDQ_CONV_H_
#define NPU_EMU_QDQ_CONV_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ifm I8 / U8, wts I8 / U8, out I8 / U8. c0 / c1 / c2: device arrays of Co floats, c1 null
 * for no activation sum. activation 0, 100, 3 or 9 (PRELU: act0 the pivot, slope Co floats;
 * slope null for the other codes).
 *   v = mac(conv, c2, c0);  v = fma(sum_ifm, c1, v)
 * sum_ifm adds the fp32 channel sum of each window pixel one tap at a time. k_opt = 0:
 * row-major. k_opt = 1 (a 1 x 1, 3 x 3 or 7 x 7 window with c1): column 0 down the window, the
 * other columns alternately into two partial sums; total (col0 + odd) + even. */
int npu_conv_qdq_int8x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                        npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h,
                        int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                        int stride_h, int stride_w, int k_opt, const float* c0,
                        const float* c1, const float* c2, int activation,
                        float act0, float act1, const float* slope);

/* As npu_conv_qdq_int8x8 with ifm I16 / U16 split into a high (signed for I16) and a low
 * byte, out I16 / U16:
 *   g = mac(conv_hi, 256, fp32(conv_lo));  v = fma(g, c2, c0);  v = fma(sum_ifm, c1, v) */
int npu_conv_qdq_int16x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                         npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h,
                         int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                         int stride_h, int stride_w, int k_opt, const float* c0,
                         const float* c1, const float* c2, int activation,
                         float act0, float act1, const float* slope);

/* The npu_conv_qdq_int8x8 accumulator v (c0 / c1 / c2 and k_opt as there), then the add of
 * ifm2 (Ho x Wo x Co, I8 / U8), out I8 / U8:
 *   if act_before_add: v = act(v)
 *   v = mac(ifm2, add_c1, fma(v, add_c0, add_c2))
 *   if !act_before_add: v = act(v)
 * activation 0, 100 or 3; act_before_add only with 3. */
int npu_qdq_conv_add_a8w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                          npu_dtype wts_type, const void* ifm2, npu_dtype ifm2_type, void* out,
                          npu_dtype out_type, int64_t h, int64_t w, int64_t ci, int64_t kh,
                          int64_t kw, int64_t co, int stride_h, int stride_w, int k_opt,
                          const float* c0, const float* c1, const float* c2, float add_c0,
                          float add_c1, float add_c2, int act_before_add, int activation,
                          float act0, float act1);

/* As npu_qdq_conv_add_a8w8 on the npu_conv_qdq_int16x8 accumulator, ifm2 and out I16 / U16;
 * each pixel's channel sum is fp32(256 * sum(hi) + sum(lo)). activation 0 or 3. */
int npu_qdq_conv_add_a16w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                           npu_dtype wts_type, const void* ifm2, npu_dtype ifm2_type,
                           void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t ci,
                           int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                           int k_opt, const float* c0, const float* c1, const float* c2,
                           float add_c0, float add_c1, float add_c2, int act_before_add,
                           int activation, float act0, float act1);

/* ifm, wts, ifm2 and out I8 / U8. No activation sum, no activation. The convolution is
 * requantized to the output type before the add:
 *   q = sat(rne(mac(conv, c2, c0)));  out = sat(rne(mac(ifm2, add_c1, mac(q, add_c0, add_c2)))) */
int npu_qdq_conv_q_add_a8w8(void* stream, const void* ifm, npu_dtype ifm_type,
                            const void* wts, npu_dtype wts_type, const void* ifm2,
                            npu_dtype ifm2_type, void* out, npu_dtype out_type, int64_t h,
                            int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                            int stride_h, int stride_w, const float* c0, const float* c2,
                            float add_c0, float add_c1, float add_c2);

/* Depthwise: ifm H x W x C (int8x8: I8 / U8; int16x8: I16 / U16 split into bytes), wts
 * Kh x Kw x C (I8 / U8), c0 / c2 device arrays of C floats, zp_w the weight zero point. S is
 * the window sum of the input (per byte plane for 16 bits), used when actv_sum is nonzero:
 *   8-bit:  acc = fp32(conv);  acc = mac(-zp_w * S, 1, acc)
 *   16-bit: acc = mac(conv_hi, 256, fp32(conv_lo));
 *           acc = mac(-zp_w * S_hi, 256, acc);  acc = mac(-zp_w * S_lo, 1, acc)
 *   v = fma(acc, c2, c0); activation 0, 100 or 3. out I8 / U8 (int8x8) or I16 / U16 (int16x8);
 * actv_sum 0 or 1. */
int npu_qdq_dwc_int8x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                       npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h, int64_t w,
                       int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w, int zp_w,
                       int actv_sum, const float* c0, const float* c2, int activation,
                       float act0, float act1);
/* As npu_qdq_dwc_int8x8 with ifm I16 / U16. */
int npu_qdq_dwc_int16x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                        npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h, int64_t w,
                        int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w, int zp_w,
                        int actv_sum, const float* c0, const float* c2, int activation,
                        float act0, float act1);

/* npu_qdq_dwc_int8x8 with a 3 x 3 window, stride 1 in both directions and the window sum
 * always applied. */
int npu_depthwise_conv_a8w8_s1(void* stream, const void* ifm, npu_dtype ifm_type,
                               const void* wts, npu_dtype wts_type, void* out,
                               npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                               const float* c0, const float* c2, int activation,
                               float act0, float act1);
/* npu_qdq_dwc_int8x8 with a 3 x 3 window, stride 2 in both directions and the window sum
 * always applied. */
int npu_depthwise_conv_a8w8_s2(void* stream, const void* ifm, npu_dtype ifm_type,
                               const void* wts, npu_dtype wts_type, void* out,
                               npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                               const float* c0, const float* c2, int activation,
                               float act0, float act1);
/* npu_qdq_dwc_int16x8 with a 3 x 3 window and stride 1 in both directions. */
int npu_depthwise_conv_a16w8_s1(void* stream, const void* ifm, npu_dtype ifm_type,
                                const void* wts, npu_dtype wts_type, void* out,
                                npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                                int actv_sum, const float* c0, const float* c2,
                                int activation, float act0, float act1);
/* npu_qdq_dwc_int16x8 with a 3 x 3 window and stride 2 in both directions. */
int npu_depthwise_conv_a16w8_s2(void* stream, const void* ifm, npu_dtype ifm_type,
                                const void* wts, npu_dtype wts_type, void* out,
                                npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                                int actv_sum, const float* c0, const float* c2,
                                int activation, float act0, float act1);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_QDQ_CONV_H_
