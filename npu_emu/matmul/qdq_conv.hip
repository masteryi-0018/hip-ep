/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for matmul/qdq_conv.h. One thread per output element.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/act.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/quant.h"
#include "matmul/qdq_conv.h"

using namespace npu_emu;

namespace {

// Geometry and QDQ words of one convolution.
struct ConvQdq {
  const void* ifm;
  npu_dtype ifm_type;
  const void* wts;
  npu_dtype wts_type;
  int64_t w, ci, kh, kw, co, wo;
  int stride_h, stride_w, k_opt;
  // 0: exact per-pixel channel sum; 1: joined as fp32(256 * sum(hi) + sum(lo))
  int sum_join;
  const float *c0, *c1, *c2;
};

// fp32 QDQ accumulator of output element (oy, ox, oc), before the activation.
template <bool kSplit16>
__device__ float conv_qdq_value(const ConvQdq& p, int64_t oy, int64_t ox, int64_t oc) {
  const void* ifm = p.ifm;
  const void* wts = p.wts;
  const npu_dtype ifm_type = p.ifm_type, wts_type = p.wts_type;
  const int64_t w = p.w, ci = p.ci, kh = p.kh, kw = p.kw, co = p.co;
  const int stride_h = p.stride_h, stride_w = p.stride_w, k_opt = p.k_opt;
  int64_t conv_lo = 0, conv_hi = 0;
  // fp32 window sum of the channel sums, one add per tap: row-major, or (k_opt)
  // column 0 plus the other columns alternating between two partial sums.
  const bool split_cols = k_opt && kw > 1;
  const int64_t first = kw % 2;  // first column summed into s_lo
  float s_all = 0.0f, s_col0 = 0.0f, s_lo = 0.0f, s_hi = 0.0f;
  bool e_all = true, e_col0 = true, e_lo = true, e_hi = true;
  auto add = [](float& acc, bool& empty, float v) {
    acc = empty ? v : add_q(acc, v);
    empty = false;
  };
  for (int64_t y = 0; y < kh; ++y) {
    for (int64_t x = 0; x < kw; ++x) {
      const int64_t pix = ((oy * stride_h + y) * w + ox * stride_w + x) * ci;
      const int64_t wrow = (y * kw + x) * ci;
      int64_t ch_sum = 0, ch_lo = 0, ch_hi = 0;
      for (int64_t c = 0; c < ci; ++c) {
        const int64_t v = load_int(ifm, ifm_type, pix + c);
        const int64_t g = load_int(wts, wts_type, (wrow + c) * co + oc);
        ch_sum += v;
        ch_lo += v & 255;
        ch_hi += v >> 8;
        if (kSplit16) {
          conv_lo += (v & 255) * g;
          conv_hi += (v >> 8) * g;
        } else {
          conv_lo += v * g;
        }
      }
      const float cs =
          p.sum_join ? fma_q(int_f32(ch_hi), 256.0f, int_f32(ch_lo)) : int_f32(ch_sum);
      if (!split_cols) add(s_all, e_all, cs);
      else if (first && x == 0) add(s_col0, e_col0, cs);
      else if ((x - first) % 2 == 0) add(s_lo, e_lo, cs);
      else add(s_hi, e_hi, cs);
    }
  }
  const float sum_ifm =
      !split_cols ? s_all : first ? add_q(add_q(s_col0, s_lo), s_hi) : add_q(s_lo, s_hi);
  float v;
  if (kSplit16) {
    const float g = mac_exact(conv_hi, 256.0f, int_f32(conv_lo));
    v = fma_q(g, p.c2[oc], p.c0[oc]);
  } else {
    v = mac_exact(conv_lo, p.c2[oc], p.c0[oc]);
  }
  if (p.c1) v = fma_q(sum_ifm, p.c1[oc], v);
  return v;
}

template <bool kSplit16>
__global__ void conv_qdq_kernel(int64_t n, ConvQdq p, void* out, npu_dtype out_type,
                                int act, float act0, float act1, const float* slope) {
  NPU_TID;
  const int64_t oc = i % p.co, ox = (i / p.co) % p.wo, oy = i / (p.co * p.wo);
  const float v = conv_qdq_value<kSplit16>(p, oy, ox, oc);
  store_int(out, out_type, i,
            activate_out(v, act, act0, act1, slope ? slope[oc] : 0.0f, out_type));
}

// Shared checks of the convolutions; out_bits / ifm2 (when present) are fixed by the kernel.
// H or W = 0 is an empty output.
bool conv_args_ok(const void* ifm, npu_dtype ifm_type, const void* wts, npu_dtype wts_type,
                  void* out, npu_dtype out_type, int bits, int64_t h, int64_t w, int64_t ci,
                  int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                  const float* c0, const float* c2) {
  const bool empty = h == 0 || w == 0;
  return !any_null(ifm, wts, out, c0, c2) && is_int_bits(ifm_type, bits) && is_i8(wts_type) &&
         is_int_bits(out_type, bits) && stride_h >= 1 && stride_w >= 1 && kh >= 1 &&
         kw >= 1 && h >= 0 && w >= 0 && (empty || (h >= kh && w >= kw)) && ci >= 1 && co >= 1;
}

// Output pixels Ho * Wo of a window accepted by conv_args_ok; 0 when H or W is 0.
int64_t out_pixels(int64_t h, int64_t w, int64_t kh, int64_t kw, int stride_h, int stride_w,
                   int64_t* wo) {
  if (h == 0 || w == 0) return *wo = 0;
  *wo = (w - kw) / stride_w + 1;
  return ((h - kh) / stride_h + 1) * *wo;
}

// k_opt selects the kernel's window-sum order of the 3 x 3 / 7 x 7 (and the single-tap 1 x 1)
// specializations, which exist only with c1.
bool k_opt_ok(int k_opt, int64_t kh, int64_t kw, const float* c1) {
  return k_opt == 0 || (k_opt == 1 && c1 && kh == kw && (kh == 1 || kh == 3 || kh == 7));
}

template <bool kSplit16>
int conv_qdq(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
             npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t ci,
             int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w, int k_opt,
             const float* c0, const float* c1, const float* c2, int act, float act0,
             float act1, const float* slope) {
  if (!conv_args_ok(ifm, ifm_type, wts, wts_type, out, out_type, kSplit16 ? 16 : 8, h, w, ci,
                    kh, kw, co, stride_h, stride_w, c0, c2) ||
      !k_opt_ok(k_opt, kh, kw, c1) ||
      !(act == kQdqActNone || act == kQdqActClip || act == kQdqActLrelu || act == kQdqActPrelu) ||
      (act == kQdqActPrelu) != (slope != nullptr))
    return -1;
  int64_t wo;
  const int64_t pix = out_pixels(h, w, kh, kw, stride_h, stride_w, &wo);
  const ConvQdq p{ifm, ifm_type, wts, wts_type, w, ci, kh, kw, co, wo, stride_h, stride_w,
                  k_opt, 0, c0, c1, c2};
  return launch(stream, pix * co, conv_qdq_kernel<kSplit16>, p, out, out_type, act, act0,
                act1, slope);
}

// Residual add after the convolution: v = mac(ifm2, c1, fma(conv, c0, c2)).
struct AddQdq {
  const void* ifm2;
  npu_dtype ifm2_type;
  float c0, c1, c2;
  int act;
  float act0, act1;
  int act_before_add;
};

template <bool kSplit16>
__global__ void conv_qdq_add_kernel(int64_t n, ConvQdq p, AddQdq a, void* out,
                                    npu_dtype out_type) {
  NPU_TID;
  const int64_t oc = i % p.co, ox = (i / p.co) % p.wo, oy = i / (p.co * p.wo);
  float v = conv_qdq_value<kSplit16>(p, oy, ox, oc);
  if (a.act_before_add) v = activate_f32(v, a.act, a.act0, a.act1);
  v = mac_exact(load_int(a.ifm2, a.ifm2_type, i), a.c1, fma_q(v, a.c0, a.c2));
  store_int(out, out_type, i,
            activate_out(v, a.act_before_add ? kQdqActNone : a.act, a.act0, a.act1, 0.0f,
                         out_type));
}

// The convolution is requantized to the output type before the add.
__global__ void conv_qdq_q_add_kernel(int64_t n, ConvQdq p, AddQdq a, void* out,
                                      npu_dtype out_type) {
  NPU_TID;
  const int64_t oc = i % p.co, ox = (i / p.co) % p.wo, oy = i / (p.co * p.wo);
  const int64_t q = requant_tail(conv_qdq_value<false>(p, oy, ox, oc), out_type);
  store_int(out, out_type, i,
            requant(load_int(a.ifm2, a.ifm2_type, i), a.c1, dequant(q, a.c0, a.c2), out_type));
}

// Geometry and QDQ words of one depthwise convolution.
struct DwcQdq {
  const void* ifm;
  npu_dtype ifm_type;
  const void* wts;
  npu_dtype wts_type;
  int64_t w, c, kh, kw, wo;
  int stride_h, stride_w, zp_w, actv_sum;
  const float *c0, *c2;
  int act;
  float act0, act1;
};

__global__ void qdq_dwc_kernel(int64_t n, DwcQdq p, void* out, npu_dtype out_type) {
  NPU_TID;
  const int64_t oc = i % p.c, ox = (i / p.c) % p.wo, oy = i / (p.c * p.wo);
  const bool split16 = p.ifm_type == NPU_I16 || p.ifm_type == NPU_U16;
  int64_t conv = 0, conv_lo = 0, conv_hi = 0, s = 0, s_lo = 0, s_hi = 0;
  for (int64_t y = 0; y < p.kh; ++y)
    for (int64_t x = 0; x < p.kw; ++x) {
      const int64_t v = load_int(p.ifm, p.ifm_type,
                                 ((oy * p.stride_h + y) * p.w + ox * p.stride_w + x) * p.c + oc);
      const int64_t g = load_int(p.wts, p.wts_type, (y * p.kw + x) * p.c + oc);
      conv += v * g;
      conv_lo += (v & 255) * g;
      conv_hi += (v >> 8) * g;
      s += v;
      s_lo += v & 255;
      s_hi += v >> 8;
    }
  float acc;
  if (split16) {
    acc = mac_exact(conv_hi, 256.0f, int_f32(conv_lo));
    if (p.actv_sum) {
      acc = mac_exact(-p.zp_w * s_hi, 256.0f, acc);
      acc = mac_exact(-p.zp_w * s_lo, 1.0f, acc);
    }
  } else {
    acc = int_f32(conv);
    if (p.actv_sum) acc = mac_exact(-p.zp_w * s, 1.0f, acc);
  }
  store_int(out, out_type, i,
            activate_out(fma_q(acc, p.c2[oc], p.c0[oc]), p.act, p.act0, p.act1, 0.0f, out_type));
}

template <bool kSplit16>
int conv_qdq_add(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                 npu_dtype wts_type, const void* ifm2, npu_dtype ifm2_type, void* out,
                 npu_dtype out_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                 int64_t co, int stride_h, int stride_w, int k_opt, int sum_join, const float* c0,
                 const float* c1, const float* c2, float add_c0, float add_c1, float add_c2,
                 int act_before_add, int act, float act0, float act1) {
  const int bits = kSplit16 ? 16 : 8;
  // CLIP exists only on the 8-bit kernel; an activation before the add only as LRELU.
  const bool act_ok =
      act == kQdqActNone || act == kQdqActLrelu || (!kSplit16 && act == kQdqActClip);
  if (!conv_args_ok(ifm, ifm_type, wts, wts_type, out, out_type, bits, h, w, ci, kh, kw, co,
                    stride_h, stride_w, c0, c2) ||
      !ifm2 || !is_int_bits(ifm2_type, bits) || !k_opt_ok(k_opt, kh, kw, c1) || !act_ok ||
      (act_before_add != 0 && act_before_add != 1) || (act_before_add && act != kQdqActLrelu))
    return -1;
  int64_t wo;
  const int64_t pix = out_pixels(h, w, kh, kw, stride_h, stride_w, &wo);
  const ConvQdq p{ifm, ifm_type, wts, wts_type, w, ci, kh, kw, co, wo, stride_h, stride_w,
                  k_opt, sum_join, c0, c1, c2};
  const AddQdq a{ifm2, ifm2_type, add_c0, add_c1, add_c2, act, act0, act1, act_before_add};
  return launch(stream, pix * co, conv_qdq_add_kernel<kSplit16>, p, a, out, out_type);
}

// The weight zero point is held in one weight byte, read with the weights' signedness.
bool zp_w_ok(int zp_w, npu_dtype wts_type) {
  return wts_type == NPU_I8 ? zp_w >= -128 && zp_w <= 127 : zp_w >= 0 && zp_w <= 255;
}

int qdq_dwc(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
            npu_dtype wts_type, void* out, npu_dtype out_type, int bits, int64_t h, int64_t w,
            int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w, int zp_w, int actv_sum,
            const float* c0, const float* c2, int activation, float act0, float act1) {
  if (!conv_args_ok(ifm, ifm_type, wts, wts_type, out, out_type, bits, h, w, c, kh, kw, c,
                    stride_h, stride_w, c0, c2) ||
      !zp_w_ok(zp_w, wts_type) || (actv_sum != 0 && actv_sum != 1) ||
      !(activation == kQdqActNone || activation == kQdqActClip || activation == kQdqActLrelu))
    return -1;
  int64_t wo;
  const int64_t pix = out_pixels(h, w, kh, kw, stride_h, stride_w, &wo);
  const DwcQdq p{ifm, ifm_type, wts, wts_type, w, c, kh, kw, wo, stride_h, stride_w, zp_w,
                 actv_sum, c0, c2, activation, act0, act1};
  return launch(stream, pix * c, qdq_dwc_kernel, p, out, out_type);
}

}  // namespace

extern "C" {

int npu_conv_qdq_int8x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                        npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h,
                        int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                        int stride_h, int stride_w, int k_opt, const float* c0,
                        const float* c1, const float* c2, int activation,
                        float act0, float act1, const float* slope) {
  return conv_qdq<false>(stream, ifm, ifm_type, wts, wts_type, out, out_type, h, w, ci, kh, kw,
                         co, stride_h, stride_w, k_opt, c0, c1, c2, activation, act0, act1, slope);
}

int npu_conv_qdq_int16x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                         npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h,
                         int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                         int stride_h, int stride_w, int k_opt, const float* c0,
                         const float* c1, const float* c2, int activation,
                         float act0, float act1, const float* slope) {
  return conv_qdq<true>(stream, ifm, ifm_type, wts, wts_type, out, out_type, h, w, ci, kh, kw,
                        co, stride_h, stride_w, k_opt, c0, c1, c2, activation, act0, act1, slope);
}

int npu_qdq_conv_add_a8w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                          npu_dtype wts_type, const void* ifm2, npu_dtype ifm2_type, void* out,
                          npu_dtype out_type, int64_t h, int64_t w, int64_t ci, int64_t kh,
                          int64_t kw, int64_t co, int stride_h, int stride_w, int k_opt,
                          const float* c0, const float* c1, const float* c2, float add_c0,
                          float add_c1, float add_c2, int act_before_add, int activation,
                          float act0, float act1) {
  return conv_qdq_add<false>(stream, ifm, ifm_type, wts, wts_type, ifm2, ifm2_type, out,
                             out_type, h, w, ci, kh, kw, co, stride_h, stride_w, k_opt, 0, c0,
                             c1, c2, add_c0, add_c1, add_c2, act_before_add, activation, act0,
                             act1);
}

int npu_qdq_conv_add_a16w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                           npu_dtype wts_type, const void* ifm2, npu_dtype ifm2_type,
                           void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t ci,
                           int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                           int k_opt, const float* c0, const float* c1, const float* c2,
                           float add_c0, float add_c1, float add_c2, int act_before_add,
                           int activation, float act0, float act1) {
  return conv_qdq_add<true>(stream, ifm, ifm_type, wts, wts_type, ifm2, ifm2_type, out, out_type,
                            h, w, ci, kh, kw, co, stride_h, stride_w, k_opt, 1, c0, c1, c2,
                            add_c0, add_c1, add_c2, act_before_add, activation, act0, act1);
}

int npu_qdq_conv_q_add_a8w8(void* stream, const void* ifm, npu_dtype ifm_type,
                            const void* wts, npu_dtype wts_type, const void* ifm2,
                            npu_dtype ifm2_type, void* out, npu_dtype out_type, int64_t h,
                            int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                            int stride_h, int stride_w, const float* c0, const float* c2,
                            float add_c0, float add_c1, float add_c2) {
  if (!conv_args_ok(ifm, ifm_type, wts, wts_type, out, out_type, 8, h, w, ci, kh, kw, co,
                    stride_h, stride_w, c0, c2) ||
      !ifm2 || !is_i8(ifm2_type))
    return -1;
  int64_t wo;
  const int64_t pix = out_pixels(h, w, kh, kw, stride_h, stride_w, &wo);
  const ConvQdq p{ifm, ifm_type, wts, wts_type, w, ci, kh, kw, co, wo, stride_h, stride_w,
                  0, 0, c0, nullptr, c2};
  const AddQdq a{ifm2, ifm2_type, add_c0, add_c1, add_c2, kQdqActNone, 0.0f, 0.0f, 0};
  return launch(stream, pix * co, conv_qdq_q_add_kernel, p, a, out, out_type);
}

int npu_qdq_dwc_int8x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                       npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h, int64_t w,
                       int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w, int zp_w,
                       int actv_sum, const float* c0, const float* c2, int activation,
                       float act0, float act1) {
  return qdq_dwc(stream, ifm, ifm_type, wts, wts_type, out, out_type, 8, h, w, c, kh, kw,
                 stride_h, stride_w, zp_w, actv_sum, c0, c2, activation, act0, act1);
}

int npu_qdq_dwc_int16x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                        npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h, int64_t w,
                        int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w, int zp_w,
                        int actv_sum, const float* c0, const float* c2, int activation,
                        float act0, float act1) {
  return qdq_dwc(stream, ifm, ifm_type, wts, wts_type, out, out_type, 16, h, w, c, kh, kw,
                 stride_h, stride_w, zp_w, actv_sum, c0, c2, activation, act0, act1);
}

int npu_depthwise_conv_a8w8_s1(void* stream, const void* ifm, npu_dtype ifm_type,
                               const void* wts, npu_dtype wts_type, void* out,
                               npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                               const float* c0, const float* c2, int activation, float act0,
                               float act1) {
  return qdq_dwc(stream, ifm, ifm_type, wts, wts_type, out, out_type, 8, h, w, c, 3, 3, 1, 1,
                 zp_w, 1, c0, c2, activation, act0, act1);
}

int npu_depthwise_conv_a8w8_s2(void* stream, const void* ifm, npu_dtype ifm_type,
                               const void* wts, npu_dtype wts_type, void* out,
                               npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                               const float* c0, const float* c2, int activation, float act0,
                               float act1) {
  return qdq_dwc(stream, ifm, ifm_type, wts, wts_type, out, out_type, 8, h, w, c, 3, 3, 2, 2,
                 zp_w, 1, c0, c2, activation, act0, act1);
}

int npu_depthwise_conv_a16w8_s1(void* stream, const void* ifm, npu_dtype ifm_type,
                                const void* wts, npu_dtype wts_type, void* out,
                                npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                                int actv_sum, const float* c0, const float* c2, int activation,
                                float act0, float act1) {
  return qdq_dwc(stream, ifm, ifm_type, wts, wts_type, out, out_type, 16, h, w, c, 3, 3, 1, 1,
                 zp_w, actv_sum, c0, c2, activation, act0, act1);
}

int npu_depthwise_conv_a16w8_s2(void* stream, const void* ifm, npu_dtype ifm_type,
                                const void* wts, npu_dtype wts_type, void* out,
                                npu_dtype out_type, int64_t h, int64_t w, int64_t c, int zp_w,
                                int actv_sum, const float* c0, const float* c2, int activation,
                                float act0, float act1) {
  return qdq_dwc(stream, ifm, ifm_type, wts, wts_type, out, out_type, 16, h, w, c, 3, 3, 2, 2,
                 zp_w, actv_sum, c0, c2, activation, act0, act1);
}

}  // extern "C"
