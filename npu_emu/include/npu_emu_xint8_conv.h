/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 xint8 convolution wrappers: xint8_conv
 * (also conv_pin_buffer / conv_pin_stream / conv1x1ci64co64, same
 * arithmetic), xint8_depthwise_conv, xint8_dwc7x7_s1, the fused
 * xint8_dwc3x3_conv1x1, xint8_conv_add / conv_q_add / conv_q_add_mac and
 * qxint8_conv. Conventions as in npu_emu.h.
 *
 * The arithmetic is integer: products accumulate exactly in int32 (acc32),
 * the int16 bias enters shifted left by shift_bias (UPS, saturating to int32),
 * and every requantization is an SRS: an arithmetic right shift of the signed
 * accumulator by `shift` bits that rounds half to even (the wrappers'
 * conv_even) or, where stated, half away from zero (symmetric_inf), then
 * saturates to the output range. A shift <= 0 shifts left without rounding.
 * Only LReLU (xint8_conv, qxint8_conv) goes through fp32.
 *
 * The shift / alpha / clamp arguments are the words the kernel reads (mllib
 * RTP q_shift_*, q_lrelu_alpha, q_max_value), not ONNX scales: for scales
 * that are powers of two, shift_bias = log2(s_bias / (s_in * s_w)) and
 * shift_out = log2(s_out / (s_in * s_w)) (the mllib YAMLs add 16 -
 * shift_lrelu_in to shift_out for LReLU).
 *
 * Output type I8: signed saturation [-128, 127]. U8: unsigned [0, 255], used
 * for ReLU / ReLU6 (the kernel's sign_O = 0), with max_value as the upper
 * clamp (127 for ReLU, min(6 / s_out, 127) for ReLU6); max_value < 0 means
 * no clamp.
 *
 * Tensors are logical and contiguous: input H x W x Ci, weights Kh x Kw x Ci x
 * Co (depthwise Kh x Kw x C), output Ho x Wo x Co with Ho = (H - Kh) /
 * stride_h + 1 (no padding). The NPU's sub-volume tiling, channel blocking
 * and weight / bias packing are not emulated. The GeLU chains (xint8_conv
 * activation_type 6 / 10, qxint8_conv 6) are not emulated: they end in the
 * nonlinear GeLU kernels; emulate the convolution with no activation to get
 * their int8 input. Unverified on the NPU: accumulators that overflow int32.
 */
#ifndef NPU_EMU_XINT8_CONV_H_
#define NPU_EMU_XINT8_CONV_H_

#include <stdint.h>

#include "npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* xint8_conv (mllib_biased_conv_int8x8): ifm I8 / U8, wts I8, bias device
 * array of Co int16 or null, output I8 / U8.
 *   A = conv + UPS(bias, shift_bias)
 * lrelu = 0: out = SRS(min(A, max_value << shift_out), shift_out), half to even.
 * lrelu = 1: m = max(fp32(A), fp32(A * alpha)), alpha = alpha_q16 / 2^16, each
 *   product rounded once, then out = RNE(m / 2^shift_lrelu_pos) saturated
 *   (one conv_even float -> int8 conversion); shift_out and max_value unused. */
int npu_xint8_conv(void* stream, const void* ifm, npu_dtype ifm_type, const int8_t* wts,
                   const int16_t* bias, void* out, npu_dtype out_type, int64_t h, int64_t w,
                   int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                   int shift_bias, int shift_out, int max_value, int lrelu, int alpha_q16,
                   int shift_lrelu_pos);

/* xint8_depthwise_conv (biased_depthwise_int8x8, conv_even) and
 * xint8_dwc7x7_s1 (symmetric_inf: round_away = 1): per channel
 *   out = SRS(dwconv + UPS(bias, shift_bias), shift_out), then min(., max_value).
 * ifm I8 / U8, wts I8 / U8 (Kh x Kw x C), bias C int16 or null. */
int npu_xint8_depthwise_conv(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                             npu_dtype wts_type, const int16_t* bias, void* out,
                             npu_dtype out_type, int64_t h, int64_t w, int64_t c, int64_t kh,
                             int64_t kw, int stride_h, int stride_w, int shift_bias,
                             int shift_out, int max_value, int round_away);

/* xint8_dwc3x3_conv1x1_fused: the depthwise stage (I8 output, conv_even, no
 * clamp) feeds a 1x1 convolution (pw_wts Ci x Co, I8 output, no activation:
 * the fused kernel instantiates the 1x1 stage without ReLU / LReLU).
 * dwc_bias / pw_bias: int16 arrays or null. */
int npu_xint8_dwc3x3_conv1x1(void* stream, const int8_t* ifm, const int8_t* dwc_wts,
                             const int16_t* dwc_bias, const int8_t* pw_wts,
                             const int16_t* pw_bias, int8_t* out, int64_t h, int64_t w,
                             int64_t ci, int64_t co, int64_t kh, int64_t kw, int stride_h,
                             int stride_w, int dwc_shift_bias, int dwc_shift_out,
                             int pw_shift_bias, int pw_shift_out);

/* xint8_conv_add / xint8_conv_q_add (mllib_biased_conv_add_int8x8), ifm / wts /
 * ifm2 I8, output I8 / U8. A = conv + UPS(bias, shift_bias), then
 * lrelu = 0 (activation_type 0 / 1 / 2), one pass:
 *   q = clamp(SRS(A, shift_conv_q), -128, 127) << shift_conv_q  (intermediate Q of
 *       conv_q_add, taken also when shift_conv_q = 0: it then only saturates A
 *       to int8); shift_conv_q < 0: q = A (conv_add has no intermediate Q)
 *   r = q + (ifm2 << shift_ifm2)
 *   out = clamp(SRS(min(r, max_value << shift_out), shift_out), lo, 127),
 *       lo = 0 with a max_value (ReLU / ReLU6), else -128
 * lrelu = 1 (activation_type 3), conv then add2d:
 *   n = min(SRS16(A, shift_lrelu_neg), 0);  p = SRS_U16(A, shift_lrelu_pos)
 *   y = SRS8(UPS(p, shift_out) + alpha_q16 * n, shift_out)
 *   out = SRS8((y << shift_add_in) + (ifm2 << shift_ifm2), shift_add_res)
 *   (shift_ifm2 is the add2d shift_in1; shift_conv_q, max_value unused).
 * x0 non-null (xint8_conv_q_add_mac): the result c of the above (as I8) feeds
 *   out = clamp(SRS(mac_coeff * c + (x0 << mac_shift_x0), mac_shift_out), -128, 127)
 * with x0 an I8 array of Ho x Wo x Co. All SRS round half to even. */
int npu_xint8_conv_add(void* stream, const int8_t* ifm, const int8_t* wts, const int16_t* bias,
                       const int8_t* ifm2, const int8_t* x0, void* out, npu_dtype out_type,
                       int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                       int stride_h, int stride_w, int shift_bias, int shift_conv_q,
                       int shift_ifm2, int shift_out, int max_value, int lrelu, int alpha_q16,
                       int shift_lrelu_neg, int shift_lrelu_pos, int shift_add_in,
                       int shift_add_res, int mac_coeff, int mac_shift_x0, int mac_shift_out);

/* qxint8_conv: xint8_conv with zero points. bias is the packed int16 the
 * kernel reads (mllib _corrected_bias: the bias, ifm and ofm zero points
 * folded in, pre-shifted by bias_shift), A = conv + bias * 2^bias_shift.
 * activation 0: out = SRS(A, shift) saturated to out_type (I8 / U8);
 * 1 / 2 (ReLU / ReLU6): then clamp to [zp_out, max_value];
 * 3 (LReLU), all fp32 ops rounded once: c = fp32(A) - fp32(zp_out * 2^shift),
 *   v = max(c, c * fp32(alpha_q16 / 2^16)) + fp32(zp_out * 2^shift),
 *   out = RNE(v * 2^-shift) saturated. */
int npu_qxint8_conv(void* stream, const void* ifm, npu_dtype ifm_type, const int8_t* wts,
                    const int16_t* bias, void* out, npu_dtype out_type, int64_t h, int64_t w,
                    int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                    int bias_shift, int shift, int activation, int zp_out, int max_value,
                    int alpha_q16);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_XINT8_CONV_H_
