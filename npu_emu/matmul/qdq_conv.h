/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * QDQ convolution int8 x int8 / int16 x int8, with a fused residual add, and QDQ depthwise
 * convolution. Conventions at npu_activation in common/npu_emu.h.
 *
 * Special values: as qdq_gemm.h; the float words (c0 / c1 / c2, act0 / act1, slope, add words)
 * follow IEEE, the output conversion maps NaN to 0 and saturates the rest [default]. zp_w is
 * a weight zero point, within the 8-bit weight range.
 */
#ifndef NPU_EMU_QDQ_CONV_H_
#define NPU_EMU_QDQ_CONV_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ifm I8 / U8, wts I8 / U8, out I8 / U8. c0 / c1 / c2: device arrays of Co floats, c1 null
 * for no activation sum. slope: Co floats for PRELU, else null.
 *   v = mac(conv, c2, c0);  v = fma(sum_ifm, c1, v)
 * sum_ifm adds the fp32 channel sum of each window pixel one tap at a time. k_opt = 0:
 * row-major. k_opt = 1: column 0 down the window, the other columns alternately into two
 * partial sums; total (col0 + odd) + even, or even + odd for an even Kw. */
int npu_conv_qdq_int8x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                        npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h,
                        int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                        int stride_h, int stride_w, int k_opt, const float* c0,
                        const float* c1, const float* c2, npu_activation activation,
                        float act0, float act1, const float* slope);

/* As npu_conv_qdq_int8x8 with ifm I16 / U16 split into a high (signed for I16) and a low
 * byte, out I16 / U16 or 8-bit:
 *   g = mac(conv_hi, 256, fp32(conv_lo));  v = fma(g, c2, c0);  v = fma(sum_ifm, c1, v) */
int npu_conv_qdq_int16x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                         npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h,
                         int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                         int stride_h, int stride_w, int k_opt, const float* c0,
                         const float* c1, const float* c2, npu_activation activation,
                         float act0, float act1, const float* slope);

/* The npu_conv_qdq_int8x8 accumulator v (c0 / c1 / c2 and k_opt as there), then the add of
 * ifm2 (Ho x Wo x Co, 8- or 16-bit integer):
 *   if act_before_add: v = act(v)
 *   v = mac(ifm2, add_c1, fma(v, add_c0, add_c2))
 *   if !act_before_add: v = act(v)
 * activation: NONE, CLIP or LRELU. */
int npu_qdq_conv_add_a8w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                          npu_dtype wts_type, const void* ifm2, npu_dtype ifm2_type, void* out,
                          npu_dtype out_type, int64_t h, int64_t w, int64_t ci, int64_t kh,
                          int64_t kw, int64_t co, int stride_h, int stride_w, int k_opt,
                          const float* c0, const float* c1, const float* c2,
                          npu_activation activation, float act0, float act1,
                          int act_before_add, float add_c0, float add_c1, float add_c2);

/* As npu_qdq_conv_add_a8w8 on the npu_conv_qdq_int16x8 accumulator; each pixel's channel
 * sum is fp32(256 * sum(hi) + sum(lo)) and the window adds run row-major. */
int npu_qdq_conv_add_a16w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                           npu_dtype wts_type, const void* ifm2, npu_dtype ifm2_type,
                           void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t ci,
                           int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                           const float* c0, const float* c1, const float* c2,
                           npu_activation activation, float act0, float act1,
                           int act_before_add, float add_c0, float add_c1, float add_c2);

/* No activation sum. The convolution is requantized to the output type before the add:
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
 *   v = fma(acc, c2, c0); activation NONE, CLIP or LRELU. */
int npu_qdq_dwc_int8x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                       npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h, int64_t w,
                       int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w, int zp_w,
                       int actv_sum, const float* c0, const float* c2, npu_activation activation,
                       float act0, float act1);
/* As npu_qdq_dwc_int8x8 with ifm I16 / U16. */
int npu_qdq_dwc_int16x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                        npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h, int64_t w,
                        int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w, int zp_w,
                        int actv_sum, const float* c0, const float* c2, npu_activation activation,
                        float act0, float act1);

/* npu_qdq_dwc_int8x8 with a 3 x 3 window, stride 1 in both directions and the window sum
 * always applied. */
int npu_depthwise_conv_a8w8_s1(void* stream, const void* ifm, npu_dtype ifm_type,
                               const void* wts, npu_dtype wts_type, void* out,
                               npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                               const float* c0, const float* c2, npu_activation activation,
                               float act0, float act1);
/* npu_qdq_dwc_int8x8 with a 3 x 3 window, stride 2 in both directions and the window sum
 * always applied. */
int npu_depthwise_conv_a8w8_s2(void* stream, const void* ifm, npu_dtype ifm_type,
                               const void* wts, npu_dtype wts_type, void* out,
                               npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                               const float* c0, const float* c2, npu_activation activation,
                               float act0, float act1);
/* npu_qdq_dwc_int16x8 with a 3 x 3 window and stride 1 in both directions. */
int npu_depthwise_conv_a16w8_s1(void* stream, const void* ifm, npu_dtype ifm_type,
                                const void* wts, npu_dtype wts_type, void* out,
                                npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                                int actv_sum, const float* c0, const float* c2,
                                npu_activation activation, float act0, float act1);
/* npu_qdq_dwc_int16x8 with a 3 x 3 window and stride 2 in both directions. */
int npu_depthwise_conv_a16w8_s2(void* stream, const void* ifm, npu_dtype ifm_type,
                                const void* wts, npu_dtype wts_type, void* out,
                                npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                                int actv_sum, const float* c0, const float* c2,
                                npu_activation activation, float act0, float act1);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_QDQ_CONV_H_
