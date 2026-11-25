/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for matmul/xint8_conv.h. One thread per output element.
 */
#include <hip/hip_runtime.h>

#include <algorithm>

#include "common/act.h"
#include "common/check.h"
#include "common/npu_emu.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/nlf.h"
#include "common/float_cvt.h"
#include "common/quant.h"
#include "matmul/xint8_conv.h"
#include "nonlinear/gelu.h"

using namespace npu_emu;

// The LReLU paths round every fp32 operation separately.

namespace {

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

__device__ inline int64_t dwconv_at(const int8_t* ifm, const int8_t* wts, int64_t w, int64_t c,
                                    int64_t kh, int64_t kw, int64_t oy, int64_t ox, int64_t oc,
                                    int stride_h, int stride_w) {
  int64_t acc = 0;
  for (int64_t y = 0; y < kh; ++y)
    for (int64_t x = 0; x < kw; ++x)
      acc += int64_t(ifm[((oy * stride_h + y) * w + ox * stride_w + x) * c + oc]) *
             int64_t(wts[(y * kw + x) * c + oc]);
  return acc;
}

__device__ inline int64_t ups_bias(const int16_t* bias, int64_t oc, int shift_bias) {
  return bias ? srs(bias[oc], -shift_bias, NPU_I32, kFloor) : 0;
}

// LReLU: max(fp32(A), fp32(A * alpha)), then one round-half-even float -> int
// conversion by 2^-shift.
__device__ inline int64_t lrelu_out(int64_t a, int lrelu_alpha, int shift,
                                   npu_dtype out_type) {
  const float alpha = int_to_float(lrelu_alpha, -16, kHalfEven);
  const float m = vmax(int_f32(a), mac_exact(a, alpha, 0.0f));
  return float_to_int(m, out_type, -shift, kHalfEven);
}

__global__ void xint8_conv_kernel(int64_t n, const int8_t* ifm, const int8_t* wts,
                                  const int16_t* bias, void* out, npu_dtype out_type, int64_t w,
                                  int64_t ci, int64_t kh, int64_t kw, int64_t co, int64_t wo,
                                  int stride_h, int stride_w, int lrelu_alpha, int max_value,
                                  int shift_bias, int shift_lrelu_pos, int shift_out,
                                  int activation) {
  NPU_TID;
  const int64_t oc = i % co, ox = (i / co) % wo, oy = i / (co * wo);
  int64_t a = conv_at(ifm, NPU_I8, wts, w, ci, kh, kw, co, oy, ox, oc, stride_h, stride_w) +
              ups_bias(bias, oc, shift_bias);
  int64_t r;
  if (activation == 3) {
    r = lrelu_out(a, lrelu_alpha, shift_lrelu_pos, out_type);
  } else {
    // min before the monotone SRS equals the kernel's min after it
    if (activation == 1 || activation == 2)
      a = min(a, shift_round(max_value, -shift_out, kFloor));
    r = srs(a, shift_out, out_type);
  }
  store_int(out, out_type, i, r);
}

__global__ void xint8_dwc_kernel(int64_t n, const int8_t* ifm, const int8_t* wts,
                                 const int16_t* bias, void* out, npu_dtype out_type, int64_t w,
                                 int64_t c, int64_t kh, int64_t kw, int64_t wo, int stride_h,
                                 int stride_w, int shift_bias, int shift_out, int max_value,
                                 int clamp, int round_away) {
  NPU_TID;
  const int64_t oc = i % c, ox = (i / c) % wo, oy = i / (c * wo);
  const int64_t a = dwconv_at(ifm, wts, w, c, kh, kw, oy, ox, oc, stride_h, stride_w) +
                    ups_bias(bias, oc, shift_bias);
  int64_t r = srs(a, shift_out, out_type, round_away != 0 ? kHalfAway : kHalfEven);
  if (clamp) r = min(r, int64_t(max_value));
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
    const int64_t d = dwconv_at(ifm, dwc_wts, w, ci, kh, kw, oy, ox, c, stride_h, stride_w) +
                      ups_bias(dwc_bias, c, dwc_shift_bias);
    a += srs(d, dwc_shift_out, NPU_I8) * int64_t(pw_wts[c * co + oc]);
  }
  out[i] = int8_t(srs(a, pw_shift_out, NPU_I8));
}

struct ConvAdd {
  int lrelu_alpha, max_value, shift_bias, shift_lrelu_pos, shift_lrelu_neg, shift_out;
  int shift_in, shift_in1, shift_res, has_conv_output_q, activation;
  int mac_coeff, mac_shift_x0, mac_shift_out;
};

__global__ void xint8_conv_add_kernel(int64_t n, const int8_t* ifm, const int8_t* ifm2,
                                      const int8_t* wts, const int16_t* bias, const int8_t* x0,
                                      int8_t* out, int64_t w, int64_t ci, int64_t kh,
                                      int64_t kw, int64_t co, int64_t wo, int stride_h,
                                      int stride_w, ConvAdd p) {
  NPU_TID;
  const int64_t oc = i % co, ox = (i / co) % wo, oy = i / (co * wo);
  const int64_t a = conv_at(ifm, NPU_I8, wts, w, ci, kh, kw, co, oy, ox, oc, stride_h, stride_w) +
                    ups_bias(bias, oc, p.shift_bias);
  const int64_t i2 = shift_round(ifm2[i], -p.shift_in1, kFloor);
  // The result is int8 (ReLU / ReLU6: clamped below at 0) also when it feeds the mac stage.
  int64_t r;
  if (p.activation == 3) {
    const int64_t neg = min(srs(a, p.shift_lrelu_neg, NPU_I16), int64_t(0));
    const int64_t pos = srs(a, p.shift_lrelu_pos, NPU_U16);
    const int64_t m = srs(pos, -p.shift_out, NPU_I32, kFloor) + p.lrelu_alpha * neg;
    const int64_t y = srs(m, p.shift_out, NPU_I8);
    const int64_t s = sat_add64(shift_round(y, -p.shift_in, kFloor), i2);
    r = srs(s, p.shift_res, NPU_I8);
  } else {
    const bool clamp = p.activation != 0;
    const int64_t q8 = srs(a, p.shift_in, NPU_I8);
    const int64_t q = p.has_conv_output_q ? shift_round(q8, -p.shift_in, kFloor) : a;
    int64_t s = sat_add64(q, i2);
    if (clamp) s = min(s, shift_round(p.max_value, -p.shift_out, kFloor));
    r = clamp64(shift_round(s, p.shift_out, kHalfEven), clamp ? 0 : -128, 127);
  }
  if (x0) {
    const int64_t m =
        sat_add64(int64_t(p.mac_coeff) * r, shift_round(x0[i], -p.mac_shift_x0, kFloor));
    r = srs(m, p.mac_shift_out, NPU_I8);
  }
  out[i] = int8_t(r);
}

__global__ void qxint8_conv_kernel(int64_t n, const void* ifm, npu_dtype ifm_type,
                                   const int8_t* wts, const int16_t* bias, void* out,
                                   npu_dtype out_type, int64_t w, int64_t ci, int64_t kh,
                                   int64_t kw, int64_t co, int64_t wo, int stride_h, int stride_w,
                                   int lrelu_alpha, int max_value, int shift_bias,
                                   int shift_lrelu_pos, int shift_out, float ofm_scale,
                                   float inv_ofm_scale, int ofm_zero_point,
                                   float output_zero_point_acc, int activation) {
  NPU_TID;
  const int64_t oc = i % co, ox = (i / co) % wo, oy = i / (co * wo);
  const int64_t a = sat_add64(
      conv_at(ifm, ifm_type, wts, w, ci, kh, kw, co, oy, ox, oc, stride_h, stride_w),
      (bias ? shift_round(bias[oc], -shift_bias, kFloor) : 0));
  int64_t r;
  if (activation == 3) {
    // max(A, alpha * A + (1 - alpha) * O), O = output_zero_point_acc
    const float alpha = int_to_float(lrelu_alpha, -16, kHalfEven);
    const float offset = mul_q(sub_q(1.0f, alpha), output_zero_point_acc);
    const float v =
        vmax(int_f32(a), add_q(mac_exact(a, alpha, 0.0f), offset));
    r = requant_tail(mul_q(v, ldexpf(1.0f, -shift_lrelu_pos)), out_type);
  } else {
    r = srs(a, shift_out, out_type);
    if (activation == 1 || activation == 2) r = clamp64(r, ofm_zero_point, max_value);
    if (activation == 6) {
      // dequantize the saturated conv output, GeLU in fp32, then mul by fp32(1 / s_out)
      // and add the zero point (two roundings), round half to even
      const float zp = int_f32(ofm_zero_point);
      const float real = mul_q(sub_q(int_f32(r), zp), ofm_scale);
      r = requant_tail(add_q(mul_q(hw_gelu_f32(real), inv_ofm_scale), zp), out_type);
    }
  }
  store_int(out, out_type, i, r);
}

bool in_i8(int v) { return v >= -128 && v <= 127; }
bool in_u8(int v) { return v >= 0 && v <= 255; }
bool in_i16(int v) { return v >= -32768 && v <= 32767; }
bool stride_ok(int s) { return s == 1 || s == 2; }

// Output extent of a valid window; false for sizes, windows or strides the kernels reject.
// max_kh: the largest kernel height (0: any). H or W = 0 is an empty output (Ho = Wo = 0).
bool geometry(int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
              int stride_h, int stride_w, int64_t max_kh, int64_t* ho, int64_t* wo) {
  const bool empty = h == 0 || w == 0;
  if (h < 0 || w < 0 || ci < 1 || co < 1 || kh < 1 || kw < 1 || kw > 15 ||
      (max_kh && kh > max_kh) || (!empty && (h < kh || w < kw)) || !stride_ok(stride_h) ||
      !stride_ok(stride_w))
    return false;
  *ho = empty ? 0 : (h - kh) / stride_h + 1;
  *wo = empty ? 0 : (w - kw) / stride_w + 1;
  return true;
}

// The clamp the kernel compares in the output's signedness.
int clamp_word(int max_value, npu_dtype out_type) {
  return out_type == NPU_I8 ? int(int8_t(max_value)) : max_value;
}

struct Gelu {
  float ka, c, kdsp;
  int up_path, rnd_id, a_shift, b_shift, q4_shift, r_shift, K, P;
  bool ok() const {
    auto sh = [](int s) { return s >= 0 && s <= 23; };
    return (up_path == 0 || up_path == 1) && rnd_id >= 0 && rnd_id <= 2 && sh(a_shift) &&
           sh(b_shift) && sh(q4_shift) && sh(r_shift) && in_i8(K) && in_i8(P);
  }
};

// allowed: bit mask of the activation codes of the entry.
int xint8_conv(void* stream, const void* ifm, const void* wts, const void* bias, void* out,
               npu_dtype out_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
               int64_t co, int stride_h, int stride_w, int lrelu_alpha, int max_value,
               int shift_bias, int shift_lrelu_pos, int shift_out, int activation, unsigned allowed,
               const Gelu& g) {
  int64_t ho, wo;
  if (any_null(ifm, wts, out) || !is_i8(out_type) || activation < 0 || activation > 10 ||
      !((allowed >> activation) & 1) ||
      !geometry(h, w, ci, kh, kw, co, stride_h, stride_w, 0, &ho, &wo) ||
      !in_i16(lrelu_alpha) || !in_u8(max_value) || !in_i8(shift_bias) ||
      !in_i8(shift_lrelu_pos) || !in_i8(shift_out))
    return -1;
  const bool gelu = activation == 6 || activation == 10;
  if (gelu && (out_type != NPU_I8 || (activation == 10 && !g.ok()))) return -1;
  const int64_t n = ho * wo * co;
  if (n == 0) return 0;
  int rc = launch(stream, n, xint8_conv_kernel, static_cast<const int8_t*>(ifm),
                  static_cast<const int8_t*>(wts), static_cast<const int16_t*>(bias), out,
                  out_type, w, ci, kh, kw, co, wo, stride_h, stride_w, lrelu_alpha,
                  clamp_word(max_value, out_type), shift_bias, shift_lrelu_pos, shift_out,
                  gelu ? 0 : activation);
  // The GeLU chains run the standalone GeLU kernel in place on the int8 conv output.
  if (rc == 0 && activation == 6)
    rc = npu_qdq_gelu_exp2_a8(stream, out, NPU_I8, out, NPU_I8, n, 0, 0.0f, g.ka, g.c, g.kdsp);
  if (rc == 0 && activation == 10)
    rc = npu_xint8_gelu(stream, out, out, n, g.up_path, g.rnd_id, g.a_shift, g.b_shift,
                        g.q4_shift, g.r_shift, g.K, g.P, 1);
  return rc;
}

constexpr unsigned kActConv = 1u << 0 | 1u << 1 | 1u << 2 | 1u << 3 | 1u << 6 | 1u << 10;
constexpr unsigned kActPin = kActConv & ~(1u << 3);

int xint8_dwc(void* stream, const void* ifm, const void* wts, const void* bias, void* out,
              npu_dtype out_type, int64_t h, int64_t w, int64_t c, int64_t kh, int64_t kw,
              int stride_h, int stride_w, int shift_bias, int shift_out, int max_value,
              int activation, int round_away) {
  int64_t ho, wo;
  if (any_null(ifm, wts, out) || !is_i8(out_type) || activation < 0 || activation > 2 ||
      !geometry(h, w, c, kh, kw, c, stride_h, stride_w, 15, &ho, &wo) ||
      !in_i8(shift_bias) || !in_i8(shift_out) || !in_u8(max_value))
    return -1;
  return launch(stream, ho * wo * c, xint8_dwc_kernel, static_cast<const int8_t*>(ifm),
                static_cast<const int8_t*>(wts), static_cast<const int16_t*>(bias), out,
                out_type, w, c, kh, kw, wo, stride_h, stride_w, shift_bias, shift_out,
                clamp_word(max_value, out_type), int(activation != 0), round_away);
}

int xint8_conv_add(void* stream, const void* ifm, const void* ifm2, const void* wts,
                   const void* bias, const void* x0, void* out, int64_t h, int64_t w, int64_t ci,
                   int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                   const ConvAdd& p) {
  int64_t ho, wo;
  if (any_null(ifm, ifm2, wts, out) ||
      !geometry(h, w, ci, kh, kw, co, stride_h, stride_w, 0, &ho, &wo) ||
      !in_i16(p.lrelu_alpha) || !in_u8(p.max_value) || !in_i8(p.shift_bias) ||
      !in_i8(p.shift_lrelu_pos) || !in_i8(p.shift_lrelu_neg) || !in_i8(p.shift_out) ||
      !in_i8(p.shift_in) || !in_i8(p.shift_in1) || !in_i8(p.shift_res) ||
      p.has_conv_output_q < 0 || p.has_conv_output_q > 2 || p.activation < 0 ||
      p.activation > 3)
    return -1;
  // LReLU: only without the intermediate Q; shift_in scales by the uint16 1 << shift_in
  if (p.activation == 3 && (p.has_conv_output_q != 0 || p.shift_in < 0 || p.shift_in > 15))
    return -1;
  return launch(stream, ho * wo * co, xint8_conv_add_kernel, static_cast<const int8_t*>(ifm),
                static_cast<const int8_t*>(ifm2), static_cast<const int8_t*>(wts),
                static_cast<const int16_t*>(bias), static_cast<const int8_t*>(x0),
                static_cast<int8_t*>(out), w, ci, kh, kw, co, wo, stride_h, stride_w, p);
}

}  // namespace

extern "C" {

int npu_xint8_conv(void* stream, const void* ifm, const void* wts, const void* bias, void* out,
                   npu_dtype out_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                   int64_t co, int stride_h, int stride_w, int lrelu_alpha, int max_value,
                   int shift_bias, int shift_lrelu_pos, int shift_out, int activation,
                   float gelu_coeff_ka, float gelu_coeff_c, float gelu_coeff_kdsp,
                   int xgelu_up_path, int xgelu_rnd_id, int xgelu_a_shift, int xgelu_b_shift,
                   int xgelu_q4_shift, int xgelu_r_shift, int xgelu_K, int xgelu_P) {
  const Gelu g{gelu_coeff_ka,  gelu_coeff_c,  gelu_coeff_kdsp, xgelu_up_path, xgelu_rnd_id,
               xgelu_a_shift,  xgelu_b_shift, xgelu_q4_shift,  xgelu_r_shift, xgelu_K,
               xgelu_P};
  return xint8_conv(stream, ifm, wts, bias, out, out_type, h, w, ci, kh, kw, co, stride_h,
                    stride_w, lrelu_alpha, max_value, shift_bias, shift_lrelu_pos, shift_out,
                    activation, kActConv, g);
}

int npu_xint8_conv_pin_buffer(void* stream, const void* ifm, const void* wts, const void* bias,
                              void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t ci,
                              int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                              int max_value, int shift_bias, int shift_out, int activation,
                              float gelu_coeff_ka, float gelu_coeff_c, float gelu_coeff_kdsp,
                              int xgelu_up_path, int xgelu_rnd_id, int xgelu_a_shift,
                              int xgelu_b_shift, int xgelu_q4_shift, int xgelu_r_shift,
                              int xgelu_K, int xgelu_P) {
  const Gelu g{gelu_coeff_ka,  gelu_coeff_c,  gelu_coeff_kdsp, xgelu_up_path, xgelu_rnd_id,
               xgelu_a_shift,  xgelu_b_shift, xgelu_q4_shift,  xgelu_r_shift, xgelu_K,
               xgelu_P};
  return xint8_conv(stream, ifm, wts, bias, out, out_type, h, w, ci, kh, kw, co, stride_h,
                    stride_w, 0, max_value, shift_bias, 0, shift_out, activation, kActPin, g);
}

int npu_xint8_conv_pin_stream(void* stream, const void* ifm, const void* wts, const void* bias,
                              void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t ci,
                              int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                              int max_value, int shift_bias, int shift_out, int activation,
                              float gelu_coeff_ka, float gelu_coeff_c, float gelu_coeff_kdsp,
                              int xgelu_up_path, int xgelu_rnd_id, int xgelu_a_shift,
                              int xgelu_b_shift, int xgelu_q4_shift, int xgelu_r_shift,
                              int xgelu_K, int xgelu_P) {
  return npu_xint8_conv_pin_buffer(stream, ifm, wts, bias, out, out_type, h, w, ci, kh, kw, co,
                                   stride_h, stride_w, max_value, shift_bias, shift_out,
                                   activation, gelu_coeff_ka, gelu_coeff_c, gelu_coeff_kdsp,
                                   xgelu_up_path, xgelu_rnd_id, xgelu_a_shift, xgelu_b_shift,
                                   xgelu_q4_shift, xgelu_r_shift, xgelu_K, xgelu_P);
}

int npu_xint8_conv1x1ci64co64(void* stream, const void* ifm, const void* wts, const void* bias,
                              void* out, int64_t h, int64_t w, int64_t ci, int64_t co,
                              int shift_bias, int shift_out) {
  if (ci != 64 || co != 64) return -1;
  return xint8_conv(stream, ifm, wts, bias, out, NPU_I8, h, w, ci, 1, 1, co, 1, 1, 0, 0,
                    shift_bias, 0, shift_out, 0, 1u, Gelu{});
}

int npu_spec_xint8_conv(void* stream, const void* ifm, const void* wts, const void* bias,
                        void* out, int64_t h, int64_t w, int64_t ci, int64_t co, int shift_bias,
                        int shift_out, int activation, int xgelu_up_path, int xgelu_rnd_id,
                        int xgelu_a_shift, int xgelu_b_shift, int xgelu_q4_shift,
                        int xgelu_r_shift, int xgelu_K, int xgelu_P) {
  if (ci != 96 || co != 64) return -1;
  const Gelu g{0.0f,          0.0f,          0.0f,           xgelu_up_path, xgelu_rnd_id,
               xgelu_a_shift, xgelu_b_shift, xgelu_q4_shift, xgelu_r_shift, xgelu_K,
               xgelu_P};
  return xint8_conv(stream, ifm, wts, bias, out, NPU_I8, h, w, ci, 1, 1, co, 1, 1, 0, 0,
                    shift_bias, 0, shift_out, activation, 1u | 1u << 10, g);
}

int npu_xint8_depthwise_conv(void* stream, const void* ifm, const void* wts, const void* bias,
                             void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t c,
                             int64_t kh, int64_t kw, int stride_h, int stride_w,
                             int shift_bias, int shift_out, int max_value, int activation) {
  return xint8_dwc(stream, ifm, wts, bias, out, out_type, h, w, c, kh, kw, stride_h, stride_w,
                   shift_bias, shift_out, max_value, activation, 0);
}

int npu_xint8_dwc7x7_s1(void* stream, const void* ifm, const void* wts, const void* bias,
                        void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t c,
                        int shift_bias, int shift_out, int max_value, int activation) {
  return xint8_dwc(stream, ifm, wts, bias, out, out_type, h, w, c, 7, 7, 1, 1, shift_bias,
                   shift_out, max_value, activation, 1);
}

int npu_xint8_dwc3x3_conv1x1_fused(void* stream, const void* ifm, const void* dwc_wts,
                                   const void* dwc_bias, const void* pw_wts,
                                   const void* pw_bias, void* out, int64_t h, int64_t w,
                                   int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
                                   int stride_w, int dwc_shift_bias, int dwc_shift_out,
                                   int pw_shift_bias, int pw_shift_out) {
  int64_t ho, wo;
  if (any_null(ifm, dwc_wts, pw_wts, out) ||
      !geometry(h, w, ci, kh, kw, co, stride_h, stride_w, 15, &ho, &wo) ||
      !in_i8(dwc_shift_bias) || !in_i8(dwc_shift_out) || !in_i8(pw_shift_bias) ||
      !in_i8(pw_shift_out))
    return -1;
  return launch(stream, ho * wo * co, xint8_dwc_pw_kernel, static_cast<const int8_t*>(ifm),
                static_cast<const int8_t*>(dwc_wts), static_cast<const int16_t*>(dwc_bias),
                static_cast<const int8_t*>(pw_wts), static_cast<const int16_t*>(pw_bias),
                static_cast<int8_t*>(out), w, ci, co, kh, kw, wo, stride_h, stride_w,
                dwc_shift_bias, dwc_shift_out, pw_shift_bias, pw_shift_out);
}

int npu_xint8_conv_add(void* stream, const void* ifm, const void* ifm2, const void* wts,
                       const void* bias, void* out, int64_t h, int64_t w, int64_t ci,
                       int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                       int lrelu_alpha, int max_value, int shift_bias, int shift_lrelu_pos,
                       int shift_lrelu_neg, int shift_out, int shift_in, int shift_in1,
                       int shift_res, int has_conv_output_q, int activation) {
  const ConvAdd p{lrelu_alpha, max_value, shift_bias, shift_lrelu_pos, shift_lrelu_neg,
                  shift_out,   shift_in,  shift_in1,  shift_res,       has_conv_output_q,
                  activation,  0,         0,          0};
  return xint8_conv_add(stream, ifm, ifm2, wts, bias, nullptr, out, h, w, ci, kh, kw, co,
                        stride_h, stride_w, p);
}

int npu_xint8_conv_add_mac(void* stream, const void* ifm, const void* ifm2, const void* wts,
                           const void* bias, const void* x0, void* out, int64_t h, int64_t w,
                           int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
                           int stride_w, int shift_bias, int shift_out, int shift_in,
                           int shift_in1, int mac_coeff, int mac_shift_x0, int mac_shift_out) {
  // the kernel's rescaled adder: m = min of the three shifts, uint8 multiplier 2^(shift_in - m)
  const int m = std::min({shift_in, shift_in1, shift_out});
  if (!x0 || !in_i8(mac_coeff) || !in_u8(mac_shift_x0) || !in_u8(mac_shift_out) || m < 0 ||
      shift_in - m > 7)
    return -1;
  const ConvAdd p{0,         0,        shift_bias, 0, 0,         shift_out,    shift_in,
                  shift_in1, 0,        2,          0, mac_coeff, mac_shift_x0, mac_shift_out};
  return xint8_conv_add(stream, ifm, ifm2, wts, bias, x0, out, h, w, ci, kh, kw, co, stride_h,
                        stride_w, p);
}

int npu_qxint8_conv(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                    const void* bias, void* out, npu_dtype out_type, int64_t h, int64_t w,
                    int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                    int lrelu_alpha, int max_value, int shift_bias, int shift_lrelu_pos,
                    int shift_out, float ofm_scale, float inv_ofm_scale, int ofm_zero_point,
                    float output_zero_point_acc, int activation) {
  int64_t ho, wo, lo, hi;
  if (any_null(ifm, wts, out) || !is_i8(ifm_type) || !is_i8(out_type) ||
      !geometry(h, w, ci, kh, kw, co, stride_h, stride_w, 0, &ho, &wo) ||
      !in_i16(lrelu_alpha) || !in_u8(max_value) || !in_i8(shift_bias) ||
      !in_i8(shift_lrelu_pos) || !in_i8(shift_out) ||
      (!(activation >= 0 && activation <= 3) && activation != 6))
    return -1;
  int_range(out_type, &lo, &hi);
  if (ofm_zero_point < lo || ofm_zero_point > hi) return -1;
  return launch(stream, ho * wo * co, qxint8_conv_kernel, ifm, ifm_type,
                static_cast<const int8_t*>(wts), static_cast<const int16_t*>(bias), out, out_type,
                w, ci, kh, kw, co, wo, stride_h, stride_w, lrelu_alpha,
                clamp_word(max_value, out_type), shift_bias, shift_lrelu_pos, shift_out,
                ofm_scale, inv_ofm_scale, ofm_zero_point, output_zero_point_acc, activation);
}

}  // extern "C"
