/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for resize.h. One thread per output element.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/quant.h"
#include "interpolation/resize.h"

using namespace npu_emu;

namespace {

struct Geom {
  int64_t Hi, Wi, C, Ho, Wo;
};

// Element offset of pixel (y, x), channel c, coordinates clamped to the tensor.
__device__ inline int64_t at(const Geom& g, int64_t y, int64_t x, int64_t c) {
  return (clamp64(y, 0, g.Hi - 1) * g.Wi + clamp64(x, 0, g.Wi - 1)) * g.C + c;
}

// Pixel of a BF16 / F16 input, or of an fp8 input scaled and truncated to t when
// fp8 != nullptr.
struct FloatSrc {
  const void* in;
  npu_dtype t;
  const uint8_t* fp8;
  int64_t n;
  float dq;
  __device__ float operator()(int64_t k) const {
    // -0 decodes to +0; truncated, a result below the smallest normal of t becomes a signed
    // zero.
    if (fp8) return fp8_in(fp8, k, n, dq, t, kTrunc, t == NPU_F16 ? kFtzOut : kFtzIn);
    return load_float(in, t, k);
  }
};

struct IntArgs {
  Geom g;
  npu_dtype t;
  const uint16_t* ih;
  const uint16_t* iw;
  const uint16_t* wts;
  int separable;
  int64_t sv_wo;
  int requantize;
  float c0, c1q;
  int q;
};

__global__ void resize_bilinear_int_kernel(int64_t n, const void* in, void* out, IntArgs a) {
  NPU_TID;
  const Geom& g = a.g;
  const int64_t c = i % g.C, w = (i / g.C) % g.Wo, h = i / (g.C * g.Wo);
  const int64_t y0 = a.ih[h], x0 = a.iw[w];
  auto px = [&](int64_t y, int64_t x) { return load_int(in, a.t, at(g, y, x, c)); };
  int64_t acc;
  if (a.separable) {
    const int64_t qy0 = a.wts[2 * h], qy1 = a.wts[2 * h + 1];
    const int64_t qx0 = a.wts[2 * g.Ho + 2 * w], qx1 = a.wts[2 * g.Ho + 2 * w + 1];
    // The H pass reads row y0 + 1 only when its weight is non-zero.
    const int64_t y1 = y0 + (qy1 != 0);
    // H-pass result, saturated to int16.
    auto tmp = [&](int64_t x) {
      return saturate(px(y0, x) * qy0 + px(y1, x) * qy1, NPU_I16);
    };
    acc = tmp(x0) * qx0 + tmp(x0 + 1) * qx1;
  } else {
    const uint16_t* s = a.wts + (h * g.Wo + w) * 4;
    const int64_t w0 = w - w % a.sv_wo;
    const int64_t y1 = y0 + (a.wts[(h * g.Wo + w0) * 4 + 2] != 0);
    const int64_t x1 = x0 + (s[1] != 0);
    acc = px(y0, x0) * s[0] + px(y0, x1) * s[1] + px(y1, x0) * s[2] + px(y1, x1) * s[3];
  }
  int64_t r;
  if (a.requantize) {
    // One fused rounding, then scaled by 2^-Q toward zero.
    r = requant(acc, a.c0, a.c1q, a.t, -a.q, kTrunc);
  } else {
    // The magnitude is shifted (toward zero).
    r = srs(acc, a.q, a.t, kTrunc);
  }
  store_int(out, a.t, i, r);
}

struct FloatArgs {
  Geom g;
  npu_dtype t;  // interior / weight type
  const uint16_t* ih;
  const uint16_t* iw;
  const uint16_t* wts;  // raw bits of t
  int64_t sv_wo;
  int fp8_out;
  float oq;
};

__global__ void resize_bilinear_float_kernel(int64_t n, FloatSrc src, void* out, FloatArgs a) {
  NPU_TID;
  const Geom& g = a.g;
  const int64_t c = i % g.C, w = (i / g.C) % g.Wo, h = i / (g.C * g.Wo);
  auto wt = [&](int64_t k) { return decode_float(a.wts[k], a.t); };
  const int64_t base = (h * g.Wo + w) * 4;
  const float s0 = wt(base), s1 = wt(base + 1), s2 = wt(base + 2), s3 = wt(base + 3);
  const int64_t w0 = w - w % a.sv_wo;
  const int64_t y0 = a.ih[h], x0 = a.iw[w];
  // -0 counts as zero, NaN as non-zero.
  const int64_t y1 = y0 + (wt((h * g.Wo + w0) * 4 + 2) != 0.0f);
  const int64_t x1 = x0 + (s1 != 0.0f);
  float acc = mul_q(src(at(g, y0, x0, c)), s0);
  acc = fma_q(src(at(g, y0, x1, c)), s1, acc);
  acc = fma_q(src(at(g, y1, x0, c)), s2, acc);
  acc = fma_q(src(at(g, y1, x1, c)), s3, acc);
  if (a.fp8_out)
    static_cast<uint8_t*>(out)[i] = fp8_out(acc, a.oq, kTrunc);
  else
    store_float(out, a.t, i, acc, kTrunc, a.t == NPU_F16 ? kFtzOut : kFtzIn);
}

__global__ void resize_cubic_kernel(int64_t n, const uint16_t* in, uint16_t* out, Geom g,
                                    const uint16_t* ih, const uint16_t* iw, const uint16_t* cy,
                                    const uint16_t* cx) {
  NPU_TID;
  const int64_t c = i % g.C, w = (i / g.C) % g.Wo, h = i / (g.C * g.Wo);
  const int64_t y = ih[h], x = iw[w];
  auto px = [&](int64_t yy, int64_t xx) { return bf16_to_f32(in[at(g, yy, xx, c)]); };
  // Pass-1 value at column xx, rounded to bf16.
  auto t = [&](int64_t xx) {
    float acc = mul_q(px(y, xx), bf16_to_f32(cy[h * 4]));
    for (int k = 1; k < 4; ++k)
      acc = fma_q(px(y + k, xx), bf16_to_f32(cy[h * 4 + k]), acc);
    return round_float(acc, NPU_BF16, kHalfEven, kFtzIn);
  };
  float acc = mul_q(t(x), bf16_to_f32(cx[w * 4]));
  for (int k = 1; k < 4; ++k)
    acc = fma_q(t(x + k), bf16_to_f32(cx[w * 4 + k]), acc);
  out[i] = f32_to_bf16(acc, kHalfEven, kFtzIn);
}

// The kernels address uint16 dimensions and indices.
bool bad_geom(int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo) {
  return Hi < 1 || Wi < 1 || C < 0 || Ho < 0 || Wo < 0 || Hi > 65535 || Wi > 65535 ||
         Ho > 65535 || Wo > 65535;
}

int64_t sv_or_wo(int64_t sv_wo, int64_t Wo) { return sv_wo ? sv_wo : (Wo ? Wo : 1); }

int resize_bilinear_int(void* stream, const void* in, npu_dtype in_type, const void* idx_h,
                        const void* idx_w, const void* wts, void* out, int64_t Hi, int64_t Wi,
                        int64_t C, int64_t Ho, int64_t Wo, npu_resize_route route,
                        int64_t sv_wo, int requantize, float c0, float c1) {
  const bool a8 = is_i8(in_type);
  if (!is_small_int(in_type)) return -1;
  if (route != NPU_RESIZE_4TAP && route != NPU_RESIZE_SEPARABLE) return -1;
  if (route == NPU_RESIZE_SEPARABLE && !a8) return -1;  // separable is 8-bit only
  if (any_null(in, idx_h, idx_w, wts, out) || bad_geom(Hi, Wi, C, Ho, Wo) || sv_wo < 0)
    return -1;
  if (Ho == 0 || Wo == 0 || C == 0) return 0;
  IntArgs a;
  a.g = {Hi, Wi, C, Ho, Wo};
  a.t = in_type;
  a.ih = static_cast<const uint16_t*>(idx_h);
  a.iw = static_cast<const uint16_t*>(idx_w);
  a.wts = static_cast<const uint16_t*>(wts);
  a.separable = route == NPU_RESIZE_SEPARABLE;
  a.sv_wo = sv_or_wo(sv_wo, Wo);
  a.requantize = requantize != 0;
  a.q = a.separable ? (in_type == NPU_U8 ? 14 : 15) : 15;
  a.c0 = c0;
  // c1 in the accumulator's units, rounded to fp32.
  a.c1q = mul_q(c1, float(1u << a.q));
  return launch(stream, Ho * Wo * C, resize_bilinear_int_kernel, in, out, a);
}

}  // namespace

extern "C" {

int npu_resize_bilinear(void* stream, const void* in, npu_dtype in_type, const void* idx_h,
                        const void* idx_w, const void* wts, void* out, int64_t Hi,
                        int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                        npu_resize_route route, int64_t sv_wo, int requantize, float c0,
                        float c1) {
  if (!is_i8(in_type)) return -1;
  return resize_bilinear_int(stream, in, in_type, idx_h, idx_w, wts, out, Hi, Wi, C, Ho, Wo,
                             route, sv_wo, requantize, c0, c1);
}

int npu_resize_bilinear_a16(void* stream, const void* in, npu_dtype in_type, const void* idx_h,
                            const void* idx_w, const void* wts, void* out, int64_t Hi,
                            int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, int64_t sv_wo,
                            int requantize, float c0, float c1) {
  if (!is_i16(in_type)) return -1;
  return resize_bilinear_int(stream, in, in_type, idx_h, idx_w, wts, out, Hi, Wi, C, Ho, Wo,
                             NPU_RESIZE_4TAP, sv_wo, requantize, c0, c1);
}

int npu_resize_bilinear_bf16(void* stream, const void* in, npu_dtype in_type,
                             const void* idx_h, const void* idx_w, const void* wts, void* out,
                             int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                             int64_t sv_wo) {
  if (!is_half_float(in_type)) return -1;
  if (any_null(in, idx_h, idx_w, wts, out) || bad_geom(Hi, Wi, C, Ho, Wo) || sv_wo < 0)
    return -1;
  if (Ho == 0 || Wo == 0 || C == 0) return 0;
  FloatSrc src{in, in_type, nullptr, 0, 0.0f};
  FloatArgs a{{Hi, Wi, C, Ho, Wo}, in_type, static_cast<const uint16_t*>(idx_h),
              static_cast<const uint16_t*>(idx_w), static_cast<const uint16_t*>(wts),
              sv_or_wo(sv_wo, Wo), 0, 0.0f};
  return launch(stream, Ho * Wo * C, resize_bilinear_float_kernel, src, out, a);
}

int npu_resize_bilinear_fp8(void* stream, const void* in, const void* idx_h,
                            const void* idx_w, const void* wts, npu_dtype wts_type, void* out,
                            int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                            int64_t sv_wo, float ifm_dequant_coeff, float ofm_quant_coeff) {
  if (!is_half_float(wts_type)) return -1;
  if (any_null(in, idx_h, idx_w, wts, out) || bad_geom(Hi, Wi, C, Ho, Wo) || sv_wo < 0 || C % 8)
    return -1;
  if (Ho == 0 || Wo == 0 || C == 0) return 0;
  FloatSrc src{nullptr, wts_type, static_cast<const uint8_t*>(in), Hi * Wi * C,
               ifm_dequant_coeff};
  FloatArgs a{{Hi, Wi, C, Ho, Wo}, wts_type, static_cast<const uint16_t*>(idx_h),
              static_cast<const uint16_t*>(idx_w), static_cast<const uint16_t*>(wts),
              sv_or_wo(sv_wo, Wo), 1, ofm_quant_coeff};
  return launch(stream, Ho * Wo * C, resize_bilinear_float_kernel, src, out, a);
}

int npu_resize_cubic_bf16(void* stream, const void* in, const void* idx_h, const void* idx_w,
                          const void* coef_h, const void* coef_w, void* out, int64_t Hi,
                          int64_t Wi, int64_t C, int64_t Ho, int64_t Wo) {
  if (any_null(in, idx_h, idx_w, coef_h, coef_w, out) || bad_geom(Hi, Wi, C, Ho, Wo))
    return -1;
  if (Ho == 0 || Wo == 0 || C == 0) return 0;
  return launch(stream, Ho * Wo * C, resize_cubic_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), Geom{Hi, Wi, C, Ho, Wo},
                static_cast<const uint16_t*>(idx_h), static_cast<const uint16_t*>(idx_w),
                static_cast<const uint16_t*>(coef_h), static_cast<const uint16_t*>(coef_w));
}

}  // extern "C"
