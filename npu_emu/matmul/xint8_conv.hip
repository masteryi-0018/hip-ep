/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for matmul/xint8_conv.h. One thread per output element.
 */
#include <hip/hip_runtime.h>

#include "common/act.h"
#include "common/npu_emu.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/nlf.h"
#include "matmul/xint8_conv.h"

using namespace npu_emu;

// The LReLU paths round every fp32 operation separately.

namespace {

// -s of a shift word; -INT32_MIN would overflow, and every shift this large acts alike. [default]
__device__ inline int neg_shift(int s) { return s == INT32_MIN ? INT32_MAX : -s; }

__device__ inline int64_t conv_at(const void* ifm, npu_dtype ifm_type, const int8_t* wts,
                                  int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                                  int64_t oy, int64_t ox, int64_t oc, int stride_h, int stride_w) {
  int64_t acc = 0;
  for (int64_t y = 0; y < kh; ++y)
    for (int64_t x = 0; x < kw; ++x) {
      const int64_t pix = ((oy * stride_h + y) * w + ox * stride_w + x) * ci;
      const int64_t wrow = (y * kw + x) * ci;
      for (int64_t c = 0; c < ci; ++c)
        acc += load_int(ifm, ifm_type, pix + c) * int64_t(wts[(wrow + c) * co + oc]);
    }
  return acc;
}

__device__ inline int64_t dwconv_at(const void* ifm, npu_dtype ifm_type, const void* wts,
                                    npu_dtype wts_type, int64_t w, int64_t c, int64_t kh,
                                    int64_t kw, int64_t oy, int64_t ox, int64_t oc, int stride_h,
                                    int stride_w) {
  int64_t acc = 0;
  for (int64_t y = 0; y < kh; ++y)
    for (int64_t x = 0; x < kw; ++x)
      acc += load_int(ifm, ifm_type, ((oy * stride_h + y) * w + ox * stride_w + x) * c + oc) *
             load_int(wts, wts_type, (y * kw + x) * c + oc);
  return acc;
}

// LReLU: max(fp32(A), fp32(A * alpha)), then one round-half-even float -> int
// conversion by 2^-shift.
__device__ inline int64_t lrelu_out(int64_t a, int alpha_q16, int shift, npu_dtype t) {
  const float alpha = float(alpha_q16) * 0x1p-16f;
  const float m = vmax(float(a), mac_exact(a, alpha, 0.0f));
  return float_to_int(m, t, neg_shift(shift), kHalfEven);
}

__global__ void xint8_conv_kernel(int64_t n, const void* ifm, npu_dtype ifm_type,
                                  const int8_t* wts, const int16_t* bias, void* out,
                                  npu_dtype out_type, int64_t w, int64_t ci, int64_t kh,
                                  int64_t kw, int64_t co, int64_t wo, int stride_h, int stride_w,
                                  int shift_bias, int shift_out, int max_value, int lrelu,
                                  int alpha_q16, int shift_lrelu_pos) {
  NPU_TID;
  const int64_t oc = i % co, ox = (i / co) % wo, oy = i / (co * wo);
  int64_t a = conv_at(ifm, ifm_type, wts, w, ci, kh, kw, co, oy, ox, oc, stride_h, stride_w) +
              (bias ? saturate(shift_round(bias[oc], neg_shift(shift_bias), kFloor), NPU_I32) : 0);
  int64_t lo, hi;
  int_range(out_type, &lo, &hi);
  int64_t r;
  if (lrelu) {
    r = lrelu_out(a, alpha_q16, shift_lrelu_pos, out_type);
  } else {
    if (max_value >= 0) a = min(a, shift_round(max_value, neg_shift(shift_out), kFloor));
    r = clamp64(shift_round(a, shift_out, kHalfEven), lo, hi);
  }
  store_int(out, out_type, i, r);
}

__global__ void xint8_dwc_kernel(int64_t n, const void* ifm, npu_dtype ifm_type, const void* wts,
                                 npu_dtype wts_type, const int16_t* bias, void* out,
                                 npu_dtype out_type, int64_t w, int64_t c, int64_t kh,
                                 int64_t kw, int64_t wo, int stride_h, int stride_w,
                                 int shift_bias, int shift_out, int max_value, int round_away) {
  NPU_TID;
  const int64_t oc = i % c, ox = (i / c) % wo, oy = i / (c * wo);
  const int64_t a =
      dwconv_at(ifm, ifm_type, wts, wts_type, w, c, kh, kw, oy, ox, oc, stride_h, stride_w) +
      (bias ? saturate(shift_round(bias[oc], neg_shift(shift_bias), kFloor), NPU_I32) : 0);
  int64_t lo, hi;
  int_range(out_type, &lo, &hi);
  int64_t r = clamp64(shift_round(a, shift_out, round_away != 0 ? kHalfAway : kHalfEven), lo, hi);
  if (max_value >= 0) r = min(r, int64_t(max_value));
  store_int(out, out_type, i, r);
}

__global__ void xint8_dwc_pw_kernel(int64_t n, const int8_t* ifm, const int8_t* dwc_wts,
                                    const int16_t* dwc_bias, const int8_t* pw_wts,
                                    const int16_t* pw_bias, int8_t* out, int64_t w, int64_t ci,
                                    int64_t co, int64_t kh, int64_t kw, int64_t wo, int stride_h,
                                    int stride_w, int dwc_shift_bias, int dwc_shift_out,
                                    int pw_shift_bias, int pw_shift_out) {
  NPU_TID;
  const int64_t oc = i % co, ox = (i / co) % wo, oy = i / (co * wo);
  int64_t a = pw_bias ? saturate(shift_round(pw_bias[oc], neg_shift(pw_shift_bias), kFloor), NPU_I32) : 0;
  for (int64_t c = 0; c < ci; ++c) {
    const int64_t b =
        dwc_bias ? saturate(shift_round(dwc_bias[c], neg_shift(dwc_shift_bias), kFloor), NPU_I32) : 0;
    const int64_t d =
        dwconv_at(ifm, NPU_I8, dwc_wts, NPU_I8, w, ci, kh, kw, oy, ox, c, stride_h, stride_w) + b;
    a += clamp64(shift_round(d, dwc_shift_out, kHalfEven), -128, 127) *
         int64_t(pw_wts[c * co + oc]);
  }
  out[i] = int8_t(clamp64(shift_round(a, pw_shift_out, kHalfEven), -128, 127));
}

struct ConvAdd {
  int shift_bias, shift_conv_q, shift_ifm2, shift_out, max_value, lrelu, alpha_q16;
  int shift_lrelu_neg, shift_lrelu_pos, shift_add_in, shift_add_res;
  int mac_coeff, mac_shift_x0, mac_shift_out;
};

__global__ void xint8_conv_add_kernel(int64_t n, const int8_t* ifm, const int8_t* wts,
                                      const int16_t* bias, const int8_t* ifm2, const int8_t* x0,
                                      void* out, npu_dtype out_type, int64_t w, int64_t ci,
                                      int64_t kh, int64_t kw, int64_t co, int64_t wo,
                                      int stride_h, int stride_w, ConvAdd p) {
  NPU_TID;
  const int64_t oc = i % co, ox = (i / co) % wo, oy = i / (co * wo);
  const int64_t a =
      conv_at(ifm, NPU_I8, wts, w, ci, kh, kw, co, oy, ox, oc, stride_h, stride_w) +
      (bias ? saturate(shift_round(bias[oc], neg_shift(p.shift_bias), kFloor), NPU_I32) : 0);
  // The result is int8 (ReLU / ReLU6: clamped below at 0) also when it feeds the mac stage.
  int64_t r;
  if (p.lrelu) {
    const int64_t neg =
        min(clamp64(shift_round(a, p.shift_lrelu_neg, kHalfEven), -32768, 32767), int64_t(0));
    const int64_t pos = clamp64(shift_round(a, p.shift_lrelu_pos, kHalfEven), 0, 65535);
    const int64_t m = saturate(shift_round(pos, neg_shift(p.shift_out), kFloor), NPU_I32) + p.alpha_q16 * neg;
    const int64_t y = clamp64(shift_round(m, p.shift_out, kHalfEven), -128, 127);
    const int64_t s = sat_add64(shift_round(y, neg_shift(p.shift_add_in), kFloor),
                              shift_round(ifm2[i], neg_shift(p.shift_ifm2), kFloor));
    r = clamp64(shift_round(s, p.shift_add_res, kHalfEven), -128, 127);
  } else {
    const int64_t q8 = clamp64(shift_round(a, p.shift_conv_q, kHalfEven), -128, 127);
    const int64_t q = p.shift_conv_q < 0 ? a : shift_round(q8, neg_shift(p.shift_conv_q), kFloor);
    int64_t s = sat_add64(q, shift_round(ifm2[i], neg_shift(p.shift_ifm2), kFloor));
    if (p.max_value >= 0) s = min(s, shift_round(p.max_value, neg_shift(p.shift_out), kFloor));
    r = clamp64(shift_round(s, p.shift_out, kHalfEven), p.max_value >= 0 ? 0 : -128, 127);
  }
  if (x0) {
    const int64_t m =
        sat_add64(int64_t(p.mac_coeff) * r, shift_round(x0[i], neg_shift(p.mac_shift_x0), kFloor));
    r = clamp64(shift_round(m, p.mac_shift_out, kHalfEven), -128, 127);
  }
  store_int(out, out_type, i, r);
}

__global__ void qxint8_conv_kernel(int64_t n, const void* ifm, npu_dtype ifm_type,
                                   const int8_t* wts, const int16_t* bias, void* out,
                                   npu_dtype out_type, int64_t w, int64_t ci, int64_t kh,
                                   int64_t kw, int64_t co, int64_t wo, int stride_h, int stride_w,
                                   int bias_shift, int shift, int activation, int zp_out,
                                   int max_value, int alpha_q16, float gelu_scale,
                                   float gelu_inv_scale) {
  NPU_TID;
  const int64_t oc = i % co, ox = (i / co) % wo, oy = i / (co * wo);
  const int64_t a = sat_add64(
      conv_at(ifm, ifm_type, wts, w, ci, kh, kw, co, oy, ox, oc, stride_h, stride_w),
      (bias ? shift_round(bias[oc], neg_shift(bias_shift), kFloor) : 0));
  int64_t lo, hi;
  int_range(out_type, &lo, &hi);
  int64_t r;
  if (activation == 3) {
    // max(A, alpha * A + (1 - alpha) * O), O = zp_out * 2^shift
    const float alpha = float(alpha_q16) * 0x1p-16f;
    const float offset = __fmul_rn(__fsub_rn(1.0f, alpha), ldexpf(float(zp_out), shift));
    const float v = vmax(float(a), __fadd_rn(mac_exact(a, alpha, 0.0f), offset));
    r = float_to_int(v * ldexpf(1.0f, neg_shift(shift)), out_type, 0, kHalfEven);
  } else {
    r = clamp64(shift_round(a, shift, kHalfEven), lo, hi);
    if (activation == 1 || activation == 2) r = clamp64(r, zp_out, max_value);
    if (activation == 6) {
      // dequantize the saturated conv output, GeLU in fp32, then mul by fp32(1 / s_out)
      // and add zp_out (two roundings), round half to even
      const float zp = float(zp_out);
      const float real = __fmul_rn(__fsub_rn(float(r), zp), gelu_scale);
      const float q = __fadd_rn(__fmul_rn(hw_gelu_f32(real), gelu_inv_scale), zp);
      r = float_to_int(q, out_type, 0, kHalfEven);
    }
  }
  store_int(out, out_type, i, r);
}

int xint8_conv(void* stream, const void* ifm, npu_dtype ifm_type, const int8_t* wts,
               const int16_t* bias, void* out, npu_dtype out_type, int64_t h, int64_t w,
               int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
               int shift_bias, int shift_out, int max_value, int lrelu, int alpha_q16,
               int shift_lrelu_pos) {
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  return launch(stream, ho * wo * co, xint8_conv_kernel, ifm, ifm_type, wts, bias, out, out_type,
                w, ci, kh, kw, co, wo, stride_h, stride_w, shift_bias, shift_out, max_value, lrelu,
                alpha_q16, shift_lrelu_pos);
}

int xint8_dwc(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
              npu_dtype wts_type, const int16_t* bias, void* out, npu_dtype out_type, int64_t h,
              int64_t w, int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w,
              int shift_bias, int shift_out, int max_value, int round_away) {
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  return launch(stream, ho * wo * c, xint8_dwc_kernel, ifm, ifm_type, wts, wts_type, bias, out,
                out_type, w, c, kh, kw, wo, stride_h, stride_w, shift_bias, shift_out, max_value,
                round_away);
}

int xint8_conv_add(void* stream, const int8_t* ifm, const int8_t* wts, const int16_t* bias,
                   const int8_t* ifm2, const int8_t* x0, void* out, npu_dtype out_type,
                   int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                   int stride_h, int stride_w, int shift_bias, int shift_conv_q,
                   int shift_ifm2, int shift_out, int max_value, int lrelu, int alpha_q16,
                   int shift_lrelu_neg, int shift_lrelu_pos, int shift_add_in,
                   int shift_add_res, int mac_coeff, int mac_shift_x0, int mac_shift_out) {
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  const ConvAdd p{shift_bias,      shift_conv_q,    shift_ifm2,   shift_out,
                  max_value,       lrelu,           alpha_q16,    shift_lrelu_neg,
                  shift_lrelu_pos, shift_add_in,    shift_add_res, mac_coeff,
                  mac_shift_x0,    mac_shift_out};
  return launch(stream, ho * wo * co, xint8_conv_add_kernel, ifm, wts, bias, ifm2, x0, out,
                out_type, w, ci, kh, kw, co, wo, stride_h, stride_w, p);
}

}  // namespace

extern "C" {

int npu_xint8_conv(void* stream, const void* ifm, npu_dtype ifm_type, const int8_t* wts,
                   const int16_t* bias, void* out, npu_dtype out_type, int64_t h, int64_t w,
                   int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                   int shift_bias, int shift_out, int max_value, int lrelu, int alpha_q16,
                   int shift_lrelu_pos) {
  return xint8_conv(stream, ifm, ifm_type, wts, bias, out, out_type, h, w, ci, kh, kw, co,
                    stride_h, stride_w, shift_bias, shift_out, max_value, lrelu, alpha_q16,
                    shift_lrelu_pos);
}

int npu_xint8_conv_pin_buffer(void* stream, const void* ifm, npu_dtype ifm_type,
                              const int8_t* wts, const int16_t* bias, void* out,
                              npu_dtype out_type, int64_t h, int64_t w, int64_t ci, int64_t kh,
                              int64_t kw, int64_t co, int stride_h, int stride_w, int shift_bias,
                              int shift_out, int max_value) {
  return xint8_conv(stream, ifm, ifm_type, wts, bias, out, out_type, h, w, ci, kh, kw, co,
                    stride_h, stride_w, shift_bias, shift_out, max_value, 0, 0, 0);
}

int npu_xint8_conv_pin_stream(void* stream, const void* ifm, npu_dtype ifm_type,
                              const int8_t* wts, const int16_t* bias, void* out,
                              npu_dtype out_type, int64_t h, int64_t w, int64_t ci, int64_t kh,
                              int64_t kw, int64_t co, int stride_h, int stride_w, int shift_bias,
                              int shift_out, int max_value) {
  return xint8_conv(stream, ifm, ifm_type, wts, bias, out, out_type, h, w, ci, kh, kw, co,
                    stride_h, stride_w, shift_bias, shift_out, max_value, 0, 0, 0);
}

int npu_xint8_conv1x1ci64co64(void* stream, const void* ifm, npu_dtype ifm_type,
                              const int8_t* wts, const int16_t* bias, int8_t* out, int64_t h,
                              int64_t w, int64_t ci, int64_t co, int shift_bias, int shift_out) {
  return xint8_conv(stream, ifm, ifm_type, wts, bias, out, NPU_I8, h, w, ci, 1, 1, co, 1, 1,
                    shift_bias, shift_out, -1, 0, 0, 0);
}

int npu_xint8_depthwise_conv(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                             npu_dtype wts_type, const int16_t* bias, void* out,
                             npu_dtype out_type, int64_t h, int64_t w, int64_t c, int64_t kh,
                             int64_t kw, int stride_h, int stride_w, int shift_bias,
                             int shift_out, int max_value) {
  return xint8_dwc(stream, ifm, ifm_type, wts, wts_type, bias, out, out_type, h, w, c, kh, kw,
                   stride_h, stride_w, shift_bias, shift_out, max_value, 0);
}

int npu_xint8_dwc7x7_s1(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                        npu_dtype wts_type, const int16_t* bias, void* out, npu_dtype out_type,
                        int64_t h, int64_t w, int64_t c, int shift_bias, int shift_out,
                        int max_value) {
  return xint8_dwc(stream, ifm, ifm_type, wts, wts_type, bias, out, out_type, h, w, c, 7, 7, 1, 1,
                   shift_bias, shift_out, max_value, 1);
}

int npu_xint8_dwc3x3_conv1x1_fused(void* stream, const int8_t* ifm, const int8_t* dwc_wts,
                                   const int16_t* dwc_bias, const int8_t* pw_wts,
                                   const int16_t* pw_bias, int8_t* out, int64_t h, int64_t w,
                                   int64_t ci, int64_t co, int64_t kh, int64_t kw, int stride_h,
                                   int stride_w, int dwc_shift_bias, int dwc_shift_out,
                                   int pw_shift_bias, int pw_shift_out) {
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  return launch(stream, ho * wo * co, xint8_dwc_pw_kernel, ifm, dwc_wts, dwc_bias, pw_wts,
                pw_bias, out, w, ci, co, kh, kw, wo, stride_h, stride_w, dwc_shift_bias,
                dwc_shift_out, pw_shift_bias, pw_shift_out);
}

int npu_xint8_conv_add(void* stream, const int8_t* ifm, const int8_t* wts, const int16_t* bias,
                       const int8_t* ifm2, void* out, npu_dtype out_type, int64_t h, int64_t w,
                       int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                       int shift_bias, int shift_conv_q, int shift_ifm2, int shift_out,
                       int max_value, int lrelu, int alpha_q16, int shift_lrelu_neg,
                       int shift_lrelu_pos, int shift_add_in, int shift_add_res) {
  return xint8_conv_add(stream, ifm, wts, bias, ifm2, nullptr, out, out_type, h, w, ci, kh, kw, co,
                        stride_h, stride_w, shift_bias, shift_conv_q, shift_ifm2, shift_out,
                        max_value, lrelu, alpha_q16, shift_lrelu_neg, shift_lrelu_pos,
                        shift_add_in, shift_add_res, 0, 0, 0);
}

int npu_xint8_conv_add_mac(void* stream, const int8_t* ifm, const int8_t* wts,
                           const int16_t* bias, const int8_t* ifm2, const int8_t* x0, int8_t* out,
                           int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                           int stride_h, int stride_w, int shift_bias, int shift_conv_q,
                           int shift_ifm2, int shift_out, int mac_coeff, int mac_shift_x0,
                           int mac_shift_out) {
  if (!x0 || shift_conv_q < 0) return -1;
  return xint8_conv_add(stream, ifm, wts, bias, ifm2, x0, out, NPU_I8, h, w, ci, kh, kw, co,
                        stride_h, stride_w, shift_bias, shift_conv_q, shift_ifm2, shift_out, -1, 0,
                        0, 0, 0, 0, 0, mac_coeff, mac_shift_x0, mac_shift_out);
}

int npu_qxint8_conv(void* stream, const void* ifm, npu_dtype ifm_type, const int8_t* wts,
                    const int16_t* bias, void* out, npu_dtype out_type, int64_t h, int64_t w,
                    int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                    int bias_shift, int shift, int activation, int zp_out, int max_value,
                    int alpha_q16, float gelu_scale, float gelu_inv_scale) {
  if (activation < 0 || activation > 6 || activation == 4 || activation == 5) return -1;
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  return launch(stream, ho * wo * co, qxint8_conv_kernel, ifm, ifm_type, wts, bias, out, out_type,
                w, ci, kh, kw, co, wo, stride_h, stride_w, bias_shift, shift, activation, zp_out,
                max_value, alpha_q16, gelu_scale, gelu_inv_scale);
}

}  // extern "C"
