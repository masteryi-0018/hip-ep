/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for include/npu_emu_xint8_conv.h. One thread per
 * output element, each recomputing its own reductions; these are
 * reference-speed kernels.
 */
#include <hip/hip_runtime.h>

#include "npu_emu.h"
#include "npu_emu_device.h"
#include "npu_emu_launch.h"
#include "npu_emu_xint8_conv.h"

using namespace npu_emu;

// The LReLU paths round every fp32 operation separately.
#pragma clang fp contract(off)

namespace {

// x * 2^s for s >= 0, x / 2^-s (floor) for s < 0.
__device__ inline int64_t shl(int64_t x, int s) { return s >= 0 ? x * (int64_t(1) << s) : x >> -s; }

// SRS of a signed accumulator (mllib DataConverter.shift_round on two's
// complement): arithmetic right shift by r rounding half to even, or half
// away from zero; r <= 0 shifts left.
__device__ inline int64_t srs(int64_t v, int r, bool away) {
  if (r <= 0) return v * (int64_t(1) << -r);
  if (r >= 63) return 0;
  const int64_t half = int64_t(1) << (r - 1);
  if (away) return (v + half - (v < 0)) >> r;
  return (v + half - 1 + ((v >> r) & 1)) >> r;
}

__device__ inline int64_t clamp64(int64_t v, int64_t lo, int64_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// UPS of the int16 bias into acc32 (saturating).
__device__ inline int64_t ups_bias(const int16_t* bias, int64_t c, int shift_bias) {
  return bias ? clamp64(shl(bias[c], shift_bias), INT32_MIN, INT32_MAX) : 0;
}

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

// mllib_biased_conv_int8x8 LReLU: max(fp32(A), fp32(A * alpha)), then one
// conv_even float -> int conversion by 2^-shift.
__device__ inline int64_t lrelu_out(int64_t a, int alpha_q16, int shift, npu_dtype t) {
  const float alpha = float(alpha_q16) * 0x1p-16f;
  const float m = fmaxf(float(a), float(double(a) * double(alpha)));
  return to_int_sat(m, t, -shift, kIntEven);
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
              ups_bias(bias, oc, shift_bias);
  int64_t lo, hi;
  int_range(out_type, &lo, &hi);
  int64_t r;
  if (lrelu) {
    r = lrelu_out(a, alpha_q16, shift_lrelu_pos, out_type);
  } else {
    if (max_value >= 0) a = min(a, shl(max_value, shift_out));
    r = clamp64(srs(a, shift_out, false), lo, hi);
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
      ups_bias(bias, oc, shift_bias);
  int64_t lo, hi;
  int_range(out_type, &lo, &hi);
  int64_t r = clamp64(srs(a, shift_out, round_away != 0), lo, hi);
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
  int64_t a = ups_bias(pw_bias, oc, pw_shift_bias);
  for (int64_t c = 0; c < ci; ++c) {
    const int64_t d = dwconv_at(ifm, NPU_I8, dwc_wts, NPU_I8, w, ci, kh, kw, oy, ox, c, stride_h,
                                stride_w) +
                      ups_bias(dwc_bias, c, dwc_shift_bias);
    a += clamp64(srs(d, dwc_shift_out, false), -128, 127) * int64_t(pw_wts[c * co + oc]);
  }
  out[i] = int8_t(clamp64(srs(a, pw_shift_out, false), -128, 127));
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
      ups_bias(bias, oc, p.shift_bias);
  // The conv_add result is int8 (ReLU / ReLU6: clamped below at 0) also when
  // it feeds the q_add_mac stage.
  int64_t r;
  if (p.lrelu) {
    const int64_t neg = min(clamp64(srs(a, p.shift_lrelu_neg, false), -32768, 32767), int64_t(0));
    const int64_t pos = clamp64(srs(a, p.shift_lrelu_pos, false), 0, 65535);
    const int64_t m = clamp64(shl(pos, p.shift_out), INT32_MIN, INT32_MAX) + p.alpha_q16 * neg;
    const int64_t y = clamp64(srs(m, p.shift_out, false), -128, 127);
    const int64_t s = shl(y, p.shift_add_in) + shl(ifm2[i], p.shift_ifm2);
    r = clamp64(srs(s, p.shift_add_res, false), -128, 127);
  } else {
    const int64_t q = p.shift_conv_q < 0
                          ? a
                          : shl(clamp64(srs(a, p.shift_conv_q, false), -128, 127), p.shift_conv_q);
    int64_t s = q + shl(ifm2[i], p.shift_ifm2);
    if (p.max_value >= 0) s = min(s, shl(p.max_value, p.shift_out));
    r = clamp64(srs(s, p.shift_out, false), p.max_value >= 0 ? 0 : -128, 127);
  }
  if (x0) {
    const int64_t m = int64_t(p.mac_coeff) * r + shl(x0[i], p.mac_shift_x0);
    r = clamp64(srs(m, p.mac_shift_out, false), -128, 127);
  }
  store_int(out, out_type, i, r);
}

__global__ void qxint8_conv_kernel(int64_t n, const void* ifm, npu_dtype ifm_type,
                                   const int8_t* wts, const int16_t* bias, void* out,
                                   npu_dtype out_type, int64_t w, int64_t ci, int64_t kh,
                                   int64_t kw, int64_t co, int64_t wo, int stride_h, int stride_w,
                                   int bias_shift, int shift, int activation, int zp_out,
                                   int max_value, int alpha_q16) {
  NPU_TID;
  const int64_t oc = i % co, ox = (i / co) % wo, oy = i / (co * wo);
  const int64_t a =
      conv_at(ifm, ifm_type, wts, w, ci, kh, kw, co, oy, ox, oc, stride_h, stride_w) +
      (bias ? int64_t(bias[oc]) * (int64_t(1) << bias_shift) : 0);
  int64_t lo, hi;
  int_range(out_type, &lo, &hi);
  int64_t r;
  if (activation == 3) {
    const float offset = ldexpf(float(zp_out), shift);
    const float c = float(a) - offset;
    const float v = fmaxf(c, c * (float(alpha_q16) * 0x1p-16f)) + offset;
    r = to_int_sat(v * ldexpf(1.0f, -shift), out_type, 0, kIntEven);
  } else {
    r = clamp64(srs(a, shift, false), lo, hi);
    if (activation == 1 || activation == 2) r = clamp64(r, zp_out, max_value);
  }
  store_int(out, out_type, i, r);
}

}  // namespace

extern "C" {

int npu_xint8_conv(void* stream, const void* ifm, npu_dtype ifm_type, const int8_t* wts,
                   const int16_t* bias, void* out, npu_dtype out_type, int64_t h, int64_t w,
                   int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                   int shift_bias, int shift_out, int max_value, int lrelu, int alpha_q16,
                   int shift_lrelu_pos) {
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  return launch(stream, ho * wo * co, xint8_conv_kernel, ifm, ifm_type, wts, bias, out, out_type,
                w, ci, kh, kw, co, wo, stride_h, stride_w, shift_bias, shift_out, max_value, lrelu,
                alpha_q16, shift_lrelu_pos);
}

int npu_xint8_depthwise_conv(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                             npu_dtype wts_type, const int16_t* bias, void* out,
                             npu_dtype out_type, int64_t h, int64_t w, int64_t c, int64_t kh,
                             int64_t kw, int stride_h, int stride_w, int shift_bias,
                             int shift_out, int max_value, int round_away) {
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  return launch(stream, ho * wo * c, xint8_dwc_kernel, ifm, ifm_type, wts, wts_type, bias, out,
                out_type, w, c, kh, kw, wo, stride_h, stride_w, shift_bias, shift_out, max_value,
                round_away);
}

int npu_xint8_dwc3x3_conv1x1(void* stream, const int8_t* ifm, const int8_t* dwc_wts,
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

int npu_qxint8_conv(void* stream, const void* ifm, npu_dtype ifm_type, const int8_t* wts,
                    const int16_t* bias, void* out, npu_dtype out_type, int64_t h, int64_t w,
                    int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                    int bias_shift, int shift, int activation, int zp_out, int max_value,
                    int alpha_q16) {
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  return launch(stream, ho * wo * co, qxint8_conv_kernel, ifm, ifm_type, wts, bias, out, out_type,
                w, ci, kh, kw, co, wo, stride_h, stride_w, bias_shift, shift, activation, zp_out,
                max_value, alpha_q16);
}

}  // extern "C"
