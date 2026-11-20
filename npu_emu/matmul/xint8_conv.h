/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * xint8 convolutions: conv, depthwise, depthwise + 1x1, conv + add and the conv with zero
 * points (qxint8). Integer arithmetic; only LReLU and GeLU go through fp32.
 *
 * Special values [default]: shift words take any int (a shift beyond the accumulator width
 * empties or saturates it); int64 sums saturate at the ends; the fp32 words (gelu_scale,
 * gelu_inv_scale) follow IEEE and the output conversion maps NaN to 0 and saturates the rest.
 */
#ifndef NPU_EMU_XINT8_CONV_H_
#define NPU_EMU_XINT8_CONV_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Common: products accumulate exactly in int32. UPS(b, s) = b << s saturated to int32.
 * SRS(a, s) is an arithmetic right shift by s that rounds half to even (half away from zero
 * where stated), then saturates to the output range; s <= 0 shifts left. Shifts, alpha and
 * clamps are the words the kernel reads, not ONNX scales: for power-of-two scales
 * shift_bias = log2(s_bias / (s_in * s_w)), shift_out = log2(s_out / (s_in * s_w)).
 * Output I8 saturates to [-128, 127], U8 to [0, 255]; max_value is the upper clamp,
 * max_value < 0 means none. ifm H x W x Ci, wts Kh x Kw x Ci x Co (depthwise Kh x Kw x C),
 * out Ho x Wo x Co, Ho = (H - Kh) / stride_h + 1 (no padding). */

/* ifm I8 / U8, wts I8, bias Co int16 or null, out I8 / U8. A = conv + UPS(bias, shift_bias).
 * lrelu = 0: out = SRS(min(A, max_value << shift_out), shift_out).
 * lrelu = 1: m = max(fp32(A), fp32(A * alpha)), alpha = alpha_q16 / 2^16,
 *   out = sat(rne(m / 2^shift_lrelu_pos)); shift_out and max_value unused.
 * A conv + GeLU chain is this entry with no activation followed by npu_qdq_gelu_exp2_a8 or
 * npu_xint8_gelu on the I8 output. */
int npu_xint8_conv(void* stream, const void* ifm, npu_dtype ifm_type, const int8_t* wts,
                   const int16_t* bias, void* out, npu_dtype out_type, int64_t h, int64_t w,
                   int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                   int shift_bias, int shift_out, int max_value, int lrelu, int alpha_q16,
                   int shift_lrelu_pos);
/* npu_xint8_conv with lrelu = 0, for the kernel that keeps weights or input in a buffer
 * across iterations. */
int npu_xint8_conv_pin_buffer(void* stream, const void* ifm, npu_dtype ifm_type,
                              const int8_t* wts, const int16_t* bias, void* out,
                              npu_dtype out_type, int64_t h, int64_t w, int64_t ci, int64_t kh,
                              int64_t kw, int64_t co, int stride_h, int stride_w, int shift_bias,
                              int shift_out, int max_value);
/* npu_xint8_conv with lrelu = 0, for the kernel that streams weights and keeps the input. */
int npu_xint8_conv_pin_stream(void* stream, const void* ifm, npu_dtype ifm_type,
                              const int8_t* wts, const int16_t* bias, void* out,
                              npu_dtype out_type, int64_t h, int64_t w, int64_t ci, int64_t kh,
                              int64_t kw, int64_t co, int stride_h, int stride_w, int shift_bias,
                              int shift_out, int max_value);
/* npu_xint8_conv with a 1 x 1 window, stride 1 and no activation: wts Ci x Co, out I8. */
int npu_xint8_conv1x1ci64co64(void* stream, const void* ifm, npu_dtype ifm_type,
                              const int8_t* wts, const int16_t* bias, int8_t* out, int64_t h,
                              int64_t w, int64_t ci, int64_t co, int shift_bias, int shift_out);

/* Per channel: out = min(SRS(dwconv + UPS(bias, shift_bias), shift_out), max_value). ifm
 * I8 / U8, wts I8 / U8 (Kh x Kw x C), bias C int16 or null. */
int npu_xint8_depthwise_conv(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                             npu_dtype wts_type, const int16_t* bias, void* out,
                             npu_dtype out_type, int64_t h, int64_t w, int64_t c, int64_t kh,
                             int64_t kw, int stride_h, int stride_w, int shift_bias,
                             int shift_out, int max_value);
/* npu_xint8_depthwise_conv with a 7 x 7 window and stride 1; its SRS rounds half away from
 * zero. */
int npu_xint8_dwc7x7_s1(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                        npu_dtype wts_type, const int16_t* bias, void* out, npu_dtype out_type,
                        int64_t h, int64_t w, int64_t c, int shift_bias, int shift_out,
                        int max_value);

/* A depthwise stage (I8 output, no clamp) feeding a 1x1 convolution (pw_wts Ci x Co, I8
 * output, no activation). dwc_bias / pw_bias: int16 arrays or null. */
int npu_xint8_dwc3x3_conv1x1_fused(void* stream, const int8_t* ifm, const int8_t* dwc_wts,
                                   const int16_t* dwc_bias, const int8_t* pw_wts,
                                   const int16_t* pw_bias, int8_t* out, int64_t h, int64_t w,
                                   int64_t ci, int64_t co, int64_t kh, int64_t kw, int stride_h,
                                   int stride_w, int dwc_shift_bias, int dwc_shift_out,
                                   int pw_shift_bias, int pw_shift_out);

/* ifm / wts / ifm2 I8, out I8 / U8. A = conv + UPS(bias, shift_bias).
 * lrelu = 0:
 *   q = clamp(SRS(A, shift_conv_q), -128, 127) << shift_conv_q; shift_conv_q < 0: q = A
 *   r = q + (ifm2 << shift_ifm2)
 *   out = clamp(SRS(min(r, max_value << shift_out), shift_out), lo, 127),
 *       lo = 0 with a max_value, else -128
 * lrelu = 1 (shift_conv_q, max_value unused):
 *   n = min(SRS16(A, shift_lrelu_neg), 0);  p = SRS_U16(A, shift_lrelu_pos)
 *   y = SRS8(UPS(p, shift_out) + alpha_q16 * n, shift_out)
 *   out = SRS8((y << shift_add_in) + (ifm2 << shift_ifm2), shift_add_res)
 * All SRS round half to even. */
int npu_xint8_conv_add(void* stream, const int8_t* ifm, const int8_t* wts, const int16_t* bias,
                       const int8_t* ifm2, void* out, npu_dtype out_type, int64_t h, int64_t w,
                       int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                       int shift_bias, int shift_conv_q, int shift_ifm2, int shift_out,
                       int max_value, int lrelu, int alpha_q16, int shift_lrelu_neg,
                       int shift_lrelu_pos, int shift_add_in, int shift_add_res);
/* npu_xint8_conv_add with lrelu = 0, no max_value and shift_conv_q >= 0; its I8 result c and
 * x0 (I8, Ho x Wo x Co) feed
 *   out = clamp(SRS(mac_coeff * c + (x0 << mac_shift_x0), mac_shift_out), -128, 127) */
int npu_xint8_conv_add_mac(void* stream, const int8_t* ifm, const int8_t* wts,
                           const int16_t* bias, const int8_t* ifm2, const int8_t* x0, int8_t* out,
                           int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                           int stride_h, int stride_w, int shift_bias, int shift_conv_q,
                           int shift_ifm2, int shift_out, int mac_coeff, int mac_shift_x0,
                           int mac_shift_out);

/* npu_xint8_conv with zero points. bias: packed int16 with the zero points folded in,
 * A = conv + bias * 2^bias_shift (a negative bias_shift rounds down). activation:
 * 0: out = SRS(A, shift) saturated to out_type (I8 / U8);
 * 1 / 2 (ReLU / ReLU6): then clamp to [zp_out, max_value];
 * 3 (LReLU), each fp32 op rounded once: c = fp32(A) - fp32(zp_out * 2^shift),
 *   v = max(c, c * fp32(alpha_q16 / 2^16)) + fp32(zp_out * 2^shift),
 *   out = sat(rne(v * 2^-shift));
 * 6 (GeLU): q = the activation-0 output, real = fp32(fp32(q - zp_out) * gelu_scale),
 *   out = sat(rne(fp32(fp32(gelu(real) * gelu_inv_scale) + zp_out))); gelu_scale = s_out,
 *   gelu_inv_scale = fp32(1 / s_out). */
int npu_qxint8_conv(void* stream, const void* ifm, npu_dtype ifm_type, const int8_t* wts,
                    const int16_t* bias, void* out, npu_dtype out_type, int64_t h, int64_t w,
                    int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                    int bias_shift, int shift, int activation, int zp_out, int max_value,
                    int alpha_q16, float gelu_scale, float gelu_inv_scale);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_XINT8_CONV_H_
