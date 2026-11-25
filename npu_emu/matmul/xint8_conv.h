/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * xint8 convolutions: conv, depthwise, depthwise + 1x1, conv + add and the conv with zero
 * points (qxint8). Integer arithmetic; only LReLU and GeLU go through fp32.
 *
 * Special values [default]: int64 sums saturate at the ends; the fp32 words follow IEEE and
 * the output conversion maps NaN to 0 and saturates the rest.
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
 * where stated), then saturates to the output range; s <= 0 shifts left. The words are the
 * kernel's: shifts int8 [-128, 127], lrelu_alpha int16 (alpha = lrelu_alpha / 2^16),
 * max_value uint8, read in the output's signedness (an I8 output clamps at int8(max_value)).
 * For power-of-two scales shift_bias = log2(s_bias / (s_in * s_w)),
 * shift_out = log2(s_out / (s_in * s_w)).
 * ifm, wts, ifm2, x0 and out are int8 unless an out_type says otherwise (I8 / U8: the
 * output saturates to [-128, 127] / [0, 255]); bias int16 or null.
 * ifm H x W x Ci, wts Kh x Kw x Ci x Co (depthwise Kh x Kw x C), out Ho x Wo x Co,
 * Ho = (H - Kh) / stride_h + 1 (no padding); strides 1 or 2, Kw (and depthwise Kh) 1..15.
 * H or W = 0 is an empty output (returns 0, nothing launched); otherwise H >= Kh, W >= Kw.
 * Every entry returns -1 for arguments outside these ranges and for null required pointers
 * (all but the biases), also when the output is empty. */

/* A = conv + UPS(bias, shift_bias). activation:
 * 0: out = SRS(A, shift_out); 1 (ReLU) / 2 (ReLU6): then min(out, max_value);
 * 3 (LReLU): m = max(fp32(A), fp32(A * alpha)), out = sat(rne(m / 2^shift_lrelu_pos));
 * 6 / 10 (GeLU chains, out I8): the activation-0 result goes through npu_qdq_gelu_exp2_a8
 *   (zero points 0, ka = gelu_coeff_ka, c = gelu_coeff_c, kdsp = gelu_coeff_kdsp) / through
 *   npu_xint8_gelu (the xgelu_* words, precision enhanced). The coefficients of the other
 *   activations are ignored. */
int npu_xint8_conv(void* stream, const void* ifm, const void* wts, const void* bias, void* out,
                   npu_dtype out_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                   int64_t co, int stride_h, int stride_w, int lrelu_alpha, int max_value,
                   int shift_bias, int shift_lrelu_pos, int shift_out, int activation,
                   float gelu_coeff_ka, float gelu_coeff_c, float gelu_coeff_kdsp,
                   int xgelu_up_path, int xgelu_rnd_id, int xgelu_a_shift, int xgelu_b_shift,
                   int xgelu_q4_shift, int xgelu_r_shift, int xgelu_K, int xgelu_P);
/* npu_xint8_conv without LReLU (activation 0, 1, 2, 6, 10), for the kernel that keeps
 * weights or input in a buffer across iterations. */
int npu_xint8_conv_pin_buffer(void* stream, const void* ifm, const void* wts, const void* bias,
                              void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t ci,
                              int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                              int max_value, int shift_bias, int shift_out, int activation,
                              float gelu_coeff_ka, float gelu_coeff_c, float gelu_coeff_kdsp,
                              int xgelu_up_path, int xgelu_rnd_id, int xgelu_a_shift,
                              int xgelu_b_shift, int xgelu_q4_shift, int xgelu_r_shift,
                              int xgelu_K, int xgelu_P);
/* npu_xint8_conv_pin_buffer for the kernel that streams weights and keeps the input. */
int npu_xint8_conv_pin_stream(void* stream, const void* ifm, const void* wts, const void* bias,
                              void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t ci,
                              int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                              int max_value, int shift_bias, int shift_out, int activation,
                              float gelu_coeff_ka, float gelu_coeff_c, float gelu_coeff_kdsp,
                              int xgelu_up_path, int xgelu_rnd_id, int xgelu_a_shift,
                              int xgelu_b_shift, int xgelu_q4_shift, int xgelu_r_shift,
                              int xgelu_K, int xgelu_P);
/* npu_xint8_conv activation 0 with a 1 x 1 window, stride 1, Ci = Co = 64: wts Ci x Co. */
int npu_xint8_conv1x1ci64co64(void* stream, const void* ifm, const void* wts, const void* bias,
                              void* out, int64_t h, int64_t w, int64_t ci, int64_t co,
                              int shift_bias, int shift_out);
/* npu_xint8_conv with a 1 x 1 window, stride 1, Ci = 96, Co = 64 (wts Ci x Co) and
 * activation 0 or 10. */
int npu_spec_xint8_conv(void* stream, const void* ifm, const void* wts, const void* bias,
                        void* out, int64_t h, int64_t w, int64_t ci, int64_t co, int shift_bias,
                        int shift_out, int activation, int xgelu_up_path, int xgelu_rnd_id,
                        int xgelu_a_shift, int xgelu_b_shift, int xgelu_q4_shift,
                        int xgelu_r_shift, int xgelu_K, int xgelu_P);

/* Per channel: out = SRS(dwconv + UPS(bias, shift_bias), shift_out), then for activation
 * 1 (ReLU) / 2 (ReLU6) min(out, max_value). wts Kh x Kw x C. */
int npu_xint8_depthwise_conv(void* stream, const void* ifm, const void* wts, const void* bias,
                             void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t c,
                             int64_t kh, int64_t kw, int stride_h, int stride_w,
                             int shift_bias, int shift_out, int max_value, int activation);
/* npu_xint8_depthwise_conv with a 7 x 7 window and stride 1; its SRS rounds half away from
 * zero. */
int npu_xint8_dwc7x7_s1(void* stream, const void* ifm, const void* wts, const void* bias,
                        void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t c,
                        int shift_bias, int shift_out, int max_value, int activation);

/* A depthwise stage (dwc_wts Kh x Kw x Ci, I8 output, no clamp) feeding a 1x1 convolution
 * (pw_wts Ci x Co, I8 output, no activation). */
int npu_xint8_dwc3x3_conv1x1_fused(void* stream, const void* ifm, const void* dwc_wts,
                                   const void* dwc_bias, const void* pw_wts,
                                   const void* pw_bias, void* out, int64_t h, int64_t w,
                                   int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
                                   int stride_w, int dwc_shift_bias, int dwc_shift_out,
                                   int pw_shift_bias, int pw_shift_out);

/* ifm2 Ho x Wo x Co. A = conv + UPS(bias, shift_bias).
 * activation 0 / 1 (ReLU) / 2 (ReLU6):
 *   q = A without the intermediate Q (has_conv_output_q 0), else
 *     q = clamp(SRS(A, shift_in), -128, 127) << shift_in   (has_conv_output_q 1 or 2)
 *   r = q + (ifm2 << shift_in1)
 *   out = clamp(SRS(r, shift_out), -128, 127); 1 / 2: out = clamp(SRS(min(r, max_value <<
 *   shift_out), shift_out), 0, 127)
 * activation 3 (LReLU, has_conv_output_q 0, 0 <= shift_in <= 15):
 *   n = min(SRS16(A, shift_lrelu_neg), 0);  p = SRS_U16(A, shift_lrelu_pos)
 *   y = SRS8(UPS(p, shift_out) + lrelu_alpha * n, shift_out)
 *   out = SRS8((y << shift_in) + (ifm2 << shift_in1), shift_res)
 * All SRS round half to even. */
int npu_xint8_conv_add(void* stream, const void* ifm, const void* ifm2, const void* wts,
                       const void* bias, void* out, int64_t h, int64_t w, int64_t ci,
                       int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                       int lrelu_alpha, int max_value, int shift_bias, int shift_lrelu_pos,
                       int shift_lrelu_neg, int shift_out, int shift_in, int shift_in1,
                       int shift_res, int has_conv_output_q, int activation);
/* npu_xint8_conv_add with activation 0 and the intermediate Q; its result c and x0
 * (Ho x Wo x Co) feed
 *   out = clamp(SRS(mac_coeff * c + (x0 << mac_shift_x0), mac_shift_out), -128, 127)
 * mac_coeff int8, mac_shift_x0 / mac_shift_out uint8; with m = min(shift_in, shift_in1,
 * shift_out) the kernel needs m >= 0 and shift_in - m <= 7. */
int npu_xint8_conv_add_mac(void* stream, const void* ifm, const void* ifm2, const void* wts,
                           const void* bias, const void* x0, void* out, int64_t h, int64_t w,
                           int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
                           int stride_w, int shift_bias, int shift_out, int shift_in,
                           int shift_in1, int mac_coeff, int mac_shift_x0, int mac_shift_out);

/* npu_xint8_conv with zero points; ifm_type, out_type I8 / U8. bias: packed int16 with the
 * zero points folded in, A = conv + bias * 2^shift_bias (a negative shift_bias rounds down).
 * ofm_zero_point in the out_type range. activation:
 * 0: out = SRS(A, shift_out) saturated to out_type;
 * 1 / 2 (ReLU / ReLU6): then clamp to [ofm_zero_point, max_value];
 * 3 (LReLU), each fp32 op rounded once: O = output_zero_point_acc,
 *   v = max(fp32(A), fp32(A * alpha) + fp32((1 - alpha) * O)),
 *   out = sat(rne(v * 2^-shift_lrelu_pos));
 * 6 (GeLU): q = the activation-0 output,
 *   real = fp32(fp32(q - ofm_zero_point) * ofm_scale),
 *   out = sat(rne(fp32(fp32(gelu(real) * inv_ofm_scale) + ofm_zero_point))).
 * ofm_scale / inv_ofm_scale are read by activation 6 only, output_zero_point_acc by 3. */
int npu_qxint8_conv(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                    const void* bias, void* out, npu_dtype out_type, int64_t h, int64_t w,
                    int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                    int lrelu_alpha, int max_value, int shift_bias, int shift_lrelu_pos,
                    int shift_out, float ofm_scale, float inv_ofm_scale, int ofm_zero_point,
                    float output_zero_point_acc, int activation);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_XINT8_CONV_H_
