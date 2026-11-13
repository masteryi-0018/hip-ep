/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for include/npu_emu_resize.h. One thread per output
 * element, each re-deriving the intermediate values it needs (the separable H
 * pass, the cubic pass-1 plane); these are reference-speed kernels.
 */
#include <hip/hip_runtime.h>

#include "npu_emu.h"
#include "npu_emu_device.h"
#include "npu_emu_launch.h"
#include "npu_emu_resize.h"

using namespace npu_emu;

// Every fp32 operation is rounded on its own; the MACs are written as fmaf.
#pragma clang fp contract(off)

namespace {

struct Geom {
  int64_t Hi, Wi, C, Ho, Wo;
};

__device__ inline int64_t clampi(int64_t v, int64_t hi) {
  return v < 0 ? 0 : (v > hi ? hi : v);
}

// Element offset of pixel (y, x), channel c; the coordinates are clamped to the
// tensor (the NPU reads a halo / pad pixel there, always with a zero weight).
__device__ inline int64_t at(const Geom& g, int64_t y, int64_t x, int64_t c) {
  return (clampi(y, g.Hi - 1) * g.Wi + clampi(x, g.Wi - 1)) * g.C + c;
}

// ---- float narrowing as the wrappers program the tile -----------------------
//
// mllib models these with DataConverter.f2bf(filter_denorm=True): a result (or
// an fp32 input) below the smallest normal of the target becomes a signed zero.
// The flush is unverified on the NPU (no aiesim case reaches it).

// fp32 -> bf16 toward zero (symmetric_floor).
__device__ inline uint16_t bf16_trunc(float v) {
  const uint32_t b = __float_as_uint(v);
  const uint16_t s = uint16_t((b >> 16) & 0x8000);
  const uint32_t e = (b >> 23) & 0xff;
  if (e == 0) return s;                                          // flush
  if (e == 0xff && (b & 0x7fffff)) return uint16_t((b >> 16) | 0x40);  // NaN stays NaN
  return uint16_t(b >> 16);
}

// fp32 -> f16 toward zero; |v| >= 2^16 saturates to 65504, |v| < 2^-14 -> +-0.
__device__ inline uint16_t f16_trunc(float v) {
  const uint32_t b = __float_as_uint(v);
  const uint16_t s = uint16_t((b >> 16) & 0x8000);
  const int e = int((b >> 23) & 0xff);
  if (e == 0xff) return s | ((b & 0x7fffff) ? 0x7e00 : 0x7c00);
  if (e >= 127 + 16) return s | 0x7bff;
  if (e < 127 - 14) return s;
  return s | uint16_t(((e - 127 + 15) << 10) | ((b & 0x7fffff) >> 13));
}

// The value of v truncated to t (BF16 / F16), as fp32.
__device__ inline float trunc16(float v, npu_dtype t) {
  return t == NPU_F16 ? f16_to_f32(f16_trunc(v)) : bf16_to_f32(bf16_trunc(v));
}

// fp32 -> bf16 half to even (conv_even), with the same flush.
__device__ inline float bf16_rne_flush(float v) {
  const uint16_t r = f32_to_bf16(v, kEven);
  if (((__float_as_uint(v) >> 23) & 0xff) == 0 || ((r >> 7) & 0xff) == 0)
    return __uint_as_float(uint32_t(r & 0x8000) << 16);
  return bf16_to_f32(r);
}

// fp32 -> E4M3FN toward zero with saturation: finite magnitudes past the
// largest code (448) give 448; subnormals are kept (mllib's fp8 converter does
// not filter them). NaN -> 0x7F, infinity -> 448 (both unverified).
__device__ inline uint8_t fp8_trunc_sat(float v) {
  const uint32_t b = __float_as_uint(v);
  const uint8_t s = uint8_t((b >> 24) & 0x80);
  const int e = int((b >> 23) & 0xff);
  const uint32_t m = b & 0x7fffff;
  if (e == 0xff) return s | (m ? 0x7f : 0x7e);
  const float a = fabsf(v);
  if (a >= 448.0f) return s | 0x7e;
  if (e < 127 - 6) {  // fp8 subnormal range: m * 2^-9, m = trunc(a * 2^9)
    return s | uint8_t(int(ldexpf(a, 9)));
  }
  return s | uint8_t(((e - 127 + 7) << 3) | (m >> 20));
}

// Pixel of an interior-type (BF16 / F16) input, or of an fp8 input run through
// the wrapper's convert_fp8_fp16 pass when fp8 != nullptr.
struct FloatSrc {
  const void* in;
  npu_dtype t;
  const uint8_t* fp8;
  int64_t n;
  float dq;
  __device__ float operator()(int64_t k) const {
    // The identity-matmul decode sums the row's products, so -0 comes out +0.
    if (fp8) return trunc16(__fmul_rn(fp8_to_f32_row(fp8, k, n) + 0.0f, dq), t);
    return load_float(in, t, k);
  }
};

// ---- integer bilinear -------------------------------------------------------

struct IntArgs {
  Geom g;
  npu_dtype t;
  const int32_t* ih;
  const int32_t* iw;
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
    // H-pass result, stored as int16 (to_vector<int16_t>(0) under saturation).
    auto tmp = [&](int64_t x) {
      const int64_t v = px(y0, x) * qy0 + px(y1, x) * qy1;
      return v < -32768 ? -32768 : (v > 32767 ? 32767 : v);
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
    // mac_elem_32(acc, c0, c1 * 2^Q), then to_vector_sign(-Q) toward zero.
    r = to_int_sat(mac_exact(acc, a.c0, a.c1q), a.t, -a.q, kIntTrunc);
  } else {
    // srs by Q under symmetric_floor: the magnitude is shifted (toward zero).
    int64_t lo, hi;
    int_range(a.t, &lo, &hi);
    r = acc < 0 ? -((-acc) >> a.q) : acc >> a.q;
    r = r < lo ? lo : (r > hi ? hi : r);
  }
  store_int(out, a.t, i, r);
}

// ---- float bilinear (bf16 / f16 / fp8) --------------------------------------

struct FloatArgs {
  Geom g;
  npu_dtype t;  // interior / weight type
  const int32_t* ih;
  const int32_t* iw;
  const uint16_t* wts;  // raw bits of t
  int64_t sv_wo;
  int fp8_out;
  float oq;
};

__global__ void resize_bilinear_float_kernel(int64_t n, FloatSrc src, void* out, FloatArgs a) {
  NPU_TID;
  const Geom& g = a.g;
  const int64_t c = i % g.C, w = (i / g.C) % g.Wo, h = i / (g.C * g.Wo);
  auto wt = [&](int64_t k) {
    return a.t == NPU_F16 ? f16_to_f32(a.wts[k]) : bf16_to_f32(a.wts[k]);
  };
  const int64_t base = (h * g.Wo + w) * 4;
  const float s0 = wt(base), s1 = wt(base + 1), s2 = wt(base + 2), s3 = wt(base + 3);
  const int64_t w0 = w - w % a.sv_wo;
  const int64_t y0 = a.ih[h], x0 = a.iw[w];
  // float(s) != 0: -0 counts as zero, NaN as non-zero.
  const int64_t y1 = y0 + (wt((h * g.Wo + w0) * 4 + 2) != 0.0f);
  const int64_t x1 = x0 + (s1 != 0.0f);
  float acc = __fmul_rn(src(at(g, y0, x0, c)), s0);
  acc = fmaf(src(at(g, y0, x1, c)), s1, acc);
  acc = fmaf(src(at(g, y1, x0, c)), s2, acc);
  acc = fmaf(src(at(g, y1, x1, c)), s3, acc);
  if (a.fp8_out) {
    static_cast<uint8_t*>(out)[i] = fp8_trunc_sat(__fmul_rn(acc, a.oq));
  } else {
    static_cast<uint16_t*>(out)[i] = a.t == NPU_F16 ? f16_trunc(acc) : bf16_trunc(acc);
  }
}

// ---- bicubic ----------------------------------------------------------------

__global__ void resize_cubic_kernel(int64_t n, const uint16_t* in, uint16_t* out, Geom g,
                                    const int32_t* ih, const int32_t* iw, const uint16_t* cy,
                                    const uint16_t* cx) {
  NPU_TID;
  const int64_t c = i % g.C, w = (i / g.C) % g.Wo, h = i / (g.C * g.Wo);
  const int64_t y = ih[h], x = iw[w];
  auto px = [&](int64_t yy, int64_t xx) { return bf16_to_f32(in[at(g, yy, xx, c)]); };
  // Pass-1 value at column xx, as stored in the bf16 scratch plane.
  auto t = [&](int64_t xx) {
    float acc = __fmul_rn(px(y, xx), bf16_to_f32(cy[h * 4]));
    for (int k = 1; k < 4; ++k) acc = fmaf(px(y + k, xx), bf16_to_f32(cy[h * 4 + k]), acc);
    return bf16_rne_flush(acc);
  };
  float acc = __fmul_rn(t(x), bf16_to_f32(cx[w * 4]));
  for (int k = 1; k < 4; ++k) acc = fmaf(t(x + k), bf16_to_f32(cx[w * 4 + k]), acc);
  out[i] = uint16_t(__float_as_uint(bf16_rne_flush(acc)) >> 16);
}

bool bad_geom(int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo) {
  return Hi < 1 || Wi < 1 || C < 0 || Ho < 0 || Wo < 0;
}

}  // namespace

extern "C" int npu_resize_bilinear(void* stream, const void* in, void* out, npu_dtype t,
                                   int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                                   const int32_t* idx_h, const int32_t* idx_w,
                                   const uint16_t* wts, npu_resize_route route,
                                   int64_t sv_wo, int requantize, float c0, float c1) {
  const bool a8 = t == NPU_I8 || t == NPU_U8;
  if (!a8 && t != NPU_I16 && t != NPU_U16) return -1;
  if (route != NPU_RESIZE_4TAP && route != NPU_RESIZE_SEPARABLE) return -1;
  if (route == NPU_RESIZE_SEPARABLE && !a8) return -1;  // separable is int8-only
  if (bad_geom(Hi, Wi, C, Ho, Wo) || sv_wo < 0) return -1;
  IntArgs a;
  a.g = {Hi, Wi, C, Ho, Wo};
  a.t = t;
  a.ih = idx_h;
  a.iw = idx_w;
  a.wts = wts;
  a.separable = route == NPU_RESIZE_SEPARABLE;
  a.sv_wo = sv_wo ? sv_wo : (Wo ? Wo : 1);
  a.requantize = requantize != 0;
  // Q_H = 8 for int8, 7 for uint8 (so the int16 H-pass result cannot overflow).
  a.q = a.separable ? (t == NPU_U8 ? 14 : 15) : 15;
  a.c0 = c0;
  // The kernel scales c1 into the accumulator's units: c1 * (1U << Q), fp32.
  a.c1q = c1 * float(1u << a.q);
  return launch(stream, Ho * Wo * C, resize_bilinear_int_kernel, in, out, a);
}

extern "C" int npu_resize_bilinear_bf16(void* stream, const void* in, void* out, npu_dtype t,
                                        int64_t Hi, int64_t Wi, int64_t C, int64_t Ho,
                                        int64_t Wo, const int32_t* idx_h,
                                        const int32_t* idx_w, const void* wts, int64_t sv_wo) {
  if (t != NPU_BF16 && t != NPU_F16) return -1;
  if (bad_geom(Hi, Wi, C, Ho, Wo) || sv_wo < 0) return -1;
  FloatSrc src{in, t, nullptr, 0, 0.0f};
  FloatArgs a{{Hi, Wi, C, Ho, Wo}, t, idx_h, idx_w, static_cast<const uint16_t*>(wts),
              sv_wo ? sv_wo : (Wo ? Wo : 1), 0, 0.0f};
  return launch(stream, Ho * Wo * C, resize_bilinear_float_kernel, src, out, a);
}

extern "C" int npu_resize_bilinear_fp8(void* stream, const void* in, void* out, npu_dtype t,
                                       int64_t Hi, int64_t Wi, int64_t C, int64_t Ho,
                                       int64_t Wo, const int32_t* idx_h,
                                       const int32_t* idx_w, const void* wts, int64_t sv_wo,
                                       float ifm_dequant_coeff, float ofm_quant_coeff) {
  if (t != NPU_BF16 && t != NPU_F16) return -1;
  if (bad_geom(Hi, Wi, C, Ho, Wo) || sv_wo < 0 || C % 8) return -1;
  FloatSrc src{nullptr, t, static_cast<const uint8_t*>(in), Hi * Wi * C, ifm_dequant_coeff};
  FloatArgs a{{Hi, Wi, C, Ho, Wo}, t, idx_h, idx_w, static_cast<const uint16_t*>(wts),
              sv_wo ? sv_wo : (Wo ? Wo : 1), 1, ofm_quant_coeff};
  return launch(stream, Ho * Wo * C, resize_bilinear_float_kernel, src, out, a);
}

extern "C" int npu_resize_cubic_bf16(void* stream, const void* in, void* out, int64_t Hi,
                                     int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                                     const int32_t* idx_h, const int32_t* idx_w,
                                     const void* coef_h, const void* coef_w) {
  if (bad_geom(Hi, Wi, C, Ho, Wo)) return -1;
  return launch(stream, Ho * Wo * C, resize_cubic_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), Geom{Hi, Wi, C, Ho, Wo}, idx_h, idx_w,
                static_cast<const uint16_t*>(coef_h), static_cast<const uint16_t*>(coef_w));
}
