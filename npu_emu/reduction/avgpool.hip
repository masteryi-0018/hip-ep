/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for reduction/avgpool.h. One thread per output
 * element.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/mmac.h"
#include "common/check.h"
#include "common/quant.h"
#include "reduction/avgpool.h"

using namespace npu_emu;

namespace {

// Window pooling geometry: in [H][W][C], out [Ho][Wo][C].
struct Pool {
  int64_t H, W, C, Ho, Wo;
  int kh, kw, sh, sw;
};

// An image with H or W = 0 has no output pixel; any other one holds at least one window.
bool make_pool(int64_t H, int64_t W, int64_t C, int kh, int kw, int sh, int sw, Pool* g) {
  const bool empty = H == 0 || W == 0;
  if (C < 0 || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || H < 0 || W < 0 ||
      (!empty && (H < kh || W < kw)))
    return false;
  const int64_t Ho = empty ? 0 : (H - kh) / sh + 1, Wo = empty ? 0 : (W - kw) / sw + 1;
  *g = Pool{H, W, C, Ho, Wo, kh, kw, sh, sw};
  return true;
}

struct Tap {
  int64_t base, c;  // element offset of the window's top-left at channel c
};

__device__ inline Tap window_of(const Pool& g, int64_t i) {
  const int64_t c = i % g.C, ow = (i / g.C) % g.Wo, oh = i / (g.C * g.Wo);
  return Tap{(oh * g.sh * g.W + ow * g.sw) * g.C + c, c};
}

__device__ inline int64_t tap_at(const Pool& g, const Tap& t, int y, int x) {
  return t.base + (int64_t(y) * g.W + x) * g.C;
}

// Two accumulator chains from +0, the taps alternating in window order. A bf16
// subnormal input reads as a zero of its sign; every fma result and the final sum
// flush a subnormal to a zero of its sign.
__global__ void avgpool_16_kernel(int64_t n, const uint16_t* in, npu_dtype t, uint16_t* out,
                                  Pool g, uint16_t factor_raw) {
  NPU_TID;
  const Tap w = window_of(g, i);
  const float f = decode_float(factor_raw, t);
  float acc[2] = {0.0f, 0.0f};
  int k = 0;
  for (int y = 0; y < g.kh; ++y)
    for (int x = 0; x < g.kw; ++x, ++k)
      acc[k & 1] = fma_ftz(load_float(in, t, tap_at(g, w, y, x), kFtzIn), f, acc[k & 1], kFtzOut);
  out[i] = uint16_t(encode_float(add_q(acc[0], acc[1]), t, kHalfEven, kFtzIn));
}

// Single-tap window: a plain copy.
__global__ void copy16_kernel(int64_t n, const uint16_t* in, uint16_t* out, Pool g) {
  NPU_TID;
  out[i] = in[window_of(g, i).base];
}

// Global pooling: the pixels of a channel block by block (W block fastest,
// then H block; row by row inside a block of sv_h x sv_w).
struct Global {
  int64_t H, W, C;
  int sv_h, sv_w;
};

bool make_global(int64_t H, int64_t W, int64_t C, int sv_h, int sv_w, Global* g) {
  if (H <= 0 || W <= 0 || C < 0 || sv_h <= 0 || sv_w <= 0 || H % sv_h || W % sv_w)
    return false;
  *g = Global{H, W, C, sv_h, sv_w};
  return true;
}

// Calls f(element offset) for every pixel of channel c in block order.
template <typename F>
__device__ inline void walk_tiles(const Global& g, int64_t c, F f) {
  for (int64_t th = 0; th < g.H; th += g.sv_h)
    for (int64_t tw = 0; tw < g.W; tw += g.sv_w)
      for (int64_t y = th; y < th + g.sv_h; ++y)
        for (int64_t x = tw; x < tw + g.sv_w; ++x) f((y * g.W + x) * g.C + c);
}

// fp32 add per pixel, then fma(acc, factor, 0); a bf16 subnormal input, a subnormal sum
// or product becomes a zero of its sign.
__global__ void globalavgpool_16_kernel(int64_t n, const uint16_t* in, npu_dtype t,
                                        uint16_t* out, Global g, float factor) {
  NPU_TID;
  float acc = 0.0f;
  walk_tiles(g, i, [&](int64_t k) { acc = ftz(add_q(acc, load_float(in, t, k, kFtzIn))); });
  out[i] = uint16_t(encode_float(fma_ftz(acc, factor, 0.0f, kFtzOut), t, kHalfEven));
}

// fp8 channel c of the pixel at element offset p - c (groups of 8 from the pixel start).
__device__ inline float load_fp8(const uint8_t* in, int64_t p, int64_t c, int64_t C) {
  return fp8_decode_identity(in + (p - c), c, C);
}

// Per pass of kh_sv rows a fresh MMAC sum, passes folded as pass + total (fp32).
__global__ void avgpool_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, Pool g,
                                   int kh_sv, const float* c0, const float* c2, bool act,
                                   float act_lo, float act_hi) {
  NPU_TID;
  const Tap w = window_of(g, i);
  float total = 0.0f;
  for (int p = 0; p < g.kh; p += kh_sv) {
    float s = 0.0f;
    for (int y = p; y < p + kh_sv; ++y)
      for (int x = 0; x < g.kw; ++x)
        s = mmac_add_bf16(s, load_fp8(in, tap_at(g, w, y, x), w.c, g.C));
    total = p == 0 ? s : add_q(s, total);
  }
  float v = fma_q(total, c2[w.c], c0[w.c]);
  if (act) v = vclamp(v, act_lo, act_hi);
  out[i] = f32_to_fp8(v, kHalfEven);
}

__global__ void globalavgpool_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, Global g,
                                         float c0, float c1) {
  NPU_TID;
  float acc = 0.0f;
  walk_tiles(g, i, [&](int64_t k) { acc = mmac_add_bf16(acc, load_fp8(in, k, i, g.C)); });
  out[i] = f32_to_fp8(fma_q(acc, c1, c0), kHalfEven);
}

struct Qdq {
  float c0, c1, act_lo, act_hi;
  int shift, act;
};

// fma(int32 sum, c1, c0), optional clamp, round half to even and saturate.
__device__ inline int64_t qdq_out(int64_t s, npu_dtype ot, const Qdq& q) {
  const int64_t x = wrap(s, NPU_I32);
  if (!q.act) return requant(x, q.c1, q.c0, ot, q.shift);
  const float v = activate_f32(dequant(x, q.c1, q.c0), kQdqActClip, q.act_lo, q.act_hi);
  return float_to_int(v, ot, q.shift, kHalfEven);
}

__global__ void qdq_avgpool_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                                   npu_dtype ot, Pool g, Qdq q) {
  NPU_TID;
  const Tap w = window_of(g, i);
  int64_t s = 0;
  for (int y = 0; y < g.kh; ++y)
    for (int x = 0; x < g.kw; ++x) s += load_int(in, it, tap_at(g, w, y, x));
  store_int(out, ot, i, qdq_out(s, ot, q));
}

__global__ void qdq_globalavgpool_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                                         npu_dtype ot, Global g, Qdq q) {
  NPU_TID;
  int64_t s = 0;
  walk_tiles(g, i, [&](int64_t k) { s += load_int(in, it, k); });
  store_int(out, ot, i, qdq_out(s, ot, q));
}

// int32 sum * weight, then shift-round-saturate: ties to even, symmetric in
// sign. The relu applies to signed outputs only.
__global__ void avgpool_xint8_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                                     npu_dtype ot, Pool g, uint16_t weight, int shift,
                                     bool relu) {
  NPU_TID;
  const Tap w = window_of(g, i);
  int64_t s = 0;
  for (int y = 0; y < g.kh; ++y)
    for (int x = 0; x < g.kw; ++x) s += load_int(in, it, tap_at(g, w, y, x));
  int64_t r = srs(wrap(s * int64_t(weight), NPU_I32), shift, ot);
  if (relu && ot == NPU_I8 && r < 0) r = 0;
  store_int(out, ot, i, r);
}

// The output type is the input type. Activation codes 0 (none), 1 / 2 (clip to
// [act_c0, act_c1]).
int qdq_pool(int bits, void* stream, const void* in, npu_dtype in_type, void* out, int64_t H,
             int64_t W, int64_t C, int kh, int kw, int sh, int sw, Qdq q) {
  Pool g;
  if (any_null(in, out) || !is_int_bits(in_type, bits) || !make_pool(H, W, C, kh, kw, sh, sw, &g) ||
      q.shift < -64 || q.shift > 64 || q.act < 0 || q.act > 2)
    return -1;
  return launch(stream, g.Ho * g.Wo * C, qdq_avgpool_kernel, in, in_type, out, in_type, g, q);
}

int qdq_global(int bits, void* stream, const void* in, npu_dtype in_type, void* out, int64_t H,
               int64_t W, int64_t C, Qdq q) {
  Global g;
  // The integer sum is order-free: any tiling gives the same result.
  if (any_null(in, out) || !is_int_bits(in_type, bits) || !make_global(H, W, C, 1, 1, &g) ||
      q.shift < -64 || q.shift > 64)
    return -1;
  return launch(stream, C, qdq_globalavgpool_kernel, in, in_type, out, in_type, g, q);
}

// The output type is the input type; -4 <= shift_res <= 23; one stride for both axes.
int xint8_pool(void* stream, const void* in, npu_dtype in_type, void* out, int64_t H, int64_t W,
               int64_t C, int kh, int kw, int sh, int sw, uint16_t weight, int8_t shift_res,
               bool relu) {
  Pool g;
  if (any_null(in, out) || !is_i8(in_type) || shift_res < -4 || shift_res > 23 || sh != sw ||
      !make_pool(H, W, C, kh, kw, sh, sw, &g))
    return -1;
  return launch(stream, g.Ho * g.Wo * C, avgpool_xint8_kernel, in, in_type, out, in_type, g,
                weight, int(shift_res), relu);
}

}  // namespace

extern "C" {

int npu_avgpool_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t H,
                     int64_t W, int64_t C, int kh, int kw, int sh, int sw, uint16_t factor_raw) {
  Pool g;
  if (any_null(in, out) || !is_half_float(in_type) || !make_pool(H, W, C, kh, kw, sh, sw, &g) ||
      int64_t(kh) * kw < 2)
    return -1;
  return launch(stream, g.Ho * g.Wo * C, avgpool_16_kernel, static_cast<const uint16_t*>(in),
                in_type, static_cast<uint16_t*>(out), g, factor_raw);
}

int npu_avgpool_bf16_c32(void* stream, const void* in, void* out, int64_t H, int64_t W,
                         int64_t C, int kh, int kw, int sh, int sw, uint16_t factor_raw) {
  Pool g;
  if (any_null(in, out) || !make_pool(H, W, C, kh, kw, sh, sw, &g)) return -1;
  const auto* src = static_cast<const uint16_t*>(in);
  auto* dst = static_cast<uint16_t*>(out);
  if (int64_t(kh) * kw == 1) return launch(stream, g.Ho * g.Wo * C, copy16_kernel, src, dst, g);
  return launch(stream, g.Ho * g.Wo * C, avgpool_16_kernel, src, NPU_BF16, dst, g, factor_raw);
}

int npu_globalavgpool_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                           int64_t H, int64_t W, int64_t C, int sv_h, int sv_w,
                           float factor_val) {
  Global g;
  if (any_null(in, out) || !is_half_float(in_type) || !make_global(H, W, C, sv_h, sv_w, &g))
    return -1;
  return launch(stream, C, globalavgpool_16_kernel, static_cast<const uint16_t*>(in), in_type,
                static_cast<uint16_t*>(out), g, factor_val);
}

int npu_globalavgpool_bf16_c32(void* stream, const void* in, void* out, int64_t H, int64_t W,
                               int64_t C, int sv_h, int sv_w, float factor) {
  return npu_globalavgpool_bf16(stream, in, NPU_BF16, out, H, W, C, sv_h, sv_w, factor);
}

int npu_avgpool_fp8(void* stream, const void* in, void* out, int64_t H, int64_t W, int64_t C,
                    int kh, int kw, int sh, int sw, int kh_sv, const void* c0, const void* c2,
                    int activation, float act_lo, float act_hi) {
  Pool g;
  if (any_null(in, out, c0, c2) || !make_pool(H, W, C, kh, kw, sh, sw, &g) || kh_sv <= 0 ||
      kh % kh_sv || activation < 0 || activation > 2)
    return -1;
  return launch(stream, g.Ho * g.Wo * C, avgpool_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), g, kh_sv, static_cast<const float*>(c0),
                static_cast<const float*>(c2), activation != 0, act_lo, act_hi);
}

int npu_globalavgpool_fp8(void* stream, const void* in, void* out, int64_t H, int64_t W,
                          int64_t C, int sv_h, int sv_w, float c0, float c1) {
  Global g;
  if (any_null(in, out) || !make_global(H, W, C, sv_h, sv_w, &g)) return -1;
  return launch(stream, C, globalavgpool_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), g, c0, c1);
}

int npu_qdq_avgpool_a8(void* stream, const void* in, npu_dtype in_type, void* out, int64_t H,
                       int64_t W, int64_t C, int kh, int kw, int sh, int sw, float c0, float c1,
                       int16_t shift_res, int activation, float act_c0, float act_c1) {
  return qdq_pool(8, stream, in, in_type, out, H, W, C, kh, kw, sh, sw,
                  Qdq{c0, c1, act_c0, act_c1, shift_res, activation});
}

int npu_qdq_avgpool_a16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t H,
                        int64_t W, int64_t C, int kh, int kw, int sh, int sw, float c0, float c1,
                        int16_t shift_res) {
  return qdq_pool(16, stream, in, in_type, out, H, W, C, kh, kw, sh, sw,
                  Qdq{c0, c1, 0.0f, 0.0f, shift_res, 0});
}

int npu_qdq_globalavgpool_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                             int64_t H, int64_t W, int64_t C, float c0, float c1,
                             int16_t shift_res) {
  return qdq_global(8, stream, in, in_type, out, H, W, C,
                    Qdq{c0, c1, 0.0f, 0.0f, shift_res, 0});
}

int npu_qdq_globalavgpool_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                              int64_t H, int64_t W, int64_t C, float c0, float c1,
                              int16_t shift_res) {
  return qdq_global(16, stream, in, in_type, out, H, W, C,
                    Qdq{c0, c1, 0.0f, 0.0f, shift_res, 0});
}

int npu_avgpool_xint8(void* stream, const void* in, npu_dtype in_type, void* out, int64_t H,
                      int64_t W, int64_t C, int kh, int kw, int sh, int sw, uint16_t weight,
                      int8_t shift_res) {
  return xint8_pool(stream, in, in_type, out, H, W, C, kh, kw, sh, sw, weight, shift_res, false);
}

int npu_avgpool_xint8_fused(void* stream, const void* in, npu_dtype in_type, void* out,
                            int64_t H, int64_t W, int64_t C, int kh, int kw, int sh, int sw,
                            uint16_t weight, int8_t shift_res, int activation) {
  if (activation != 0 && activation != 1) return -1;
  return xint8_pool(stream, in, in_type, out, H, W, C, kh, kw, sh, sw, weight, shift_res,
                    activation == 1);
}

}  // extern "C"
