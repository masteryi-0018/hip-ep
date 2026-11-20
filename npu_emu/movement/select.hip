/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Selection entry points: one thread per output element (per 8-channel fp8
 * row, per row for top-k).
 */
#include <hip/hip_runtime.h>

#include "movement/select.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"

using namespace npu_emu;

namespace {


__global__ void abs_half_kernel(int64_t n, const uint16_t* in, uint16_t* out) {
  NPU_TID;
  out[i] = in[i] & 0x7fff;
}

// 0 - x: both zeros give +0.
__global__ void neg_half_kernel(int64_t n, const uint16_t* in, npu_dtype type, uint16_t* out) {
  NPU_TID;
  out[i] = uint16_t(encode_float(sub_q(0.0f, load_float(in, type, i)), type, kHalfEven));
}

__global__ void clip_half_kernel(int64_t n, const uint16_t* in, npu_dtype type,
                                 uint16_t* out, uint16_t lo, uint16_t hi) {
  NPU_TID;
  out[i] = min16(max16(in[i], lo, type), hi, type);
}

enum Fp8Unary { kAbs, kNeg, kClip };

__global__ void fp8_unary_kernel(int64_t n, const uint8_t* in, uint8_t* out,
                                 Fp8Unary op, float scale, float lo, float hi) {
  NPU_TID;
  float v = fp8_decode_identity(in, i, n);
  if (op == kAbs) v = fabsf(v);
  v = mul_q(v, scale);
  if (op == kClip) v = vclamp(v, lo, hi);
  out[i] = f32_to_fp8(v, kHalfAway);
}

enum BinOp { kEq, kGt, kGe, kMax, kMin };

__global__ void bcast_half_kernel(int64_t n, const uint16_t* a, int abc,
                                  const uint16_t* b, int bbc, npu_dtype type,
                                  void* out, BinOp op, int64_t W, int64_t C) {
  NPU_TID;
  const int64_t c = i % C, w = (i / C) % W, h = i / (C * W);
  const BIdx ia = bidx(h, w, c, abc, W, C), ib = bidx(h, w, c, bbc, W, C);
  const uint16_t xa = a[ia.pixel + ia.c], xb = b[ib.pixel + ib.c];
  const float fa = load_float(&xa, type, 0), fb = load_float(&xb, type, 0);
  if (op == kEq || op == kGt || op == kGe) {
    const bool r = op == kEq ? veq(fa, fb) : op == kGt ? vgt(fa, fb) : vge(fa, fb);
    static_cast<int8_t*>(out)[i] = r ? 1 : 0;
    return;
  }
  static_cast<uint16_t*>(out)[i] = op == kMax ? max16(xa, xb, type) : min16(xa, xb, type);
}

__global__ void bcast_fp8_kernel(int64_t n, const uint8_t* a, int abc, float as,
                                 const uint8_t* b, int bbc, float bs, void* out,
                                 BinOp op, int64_t W, int64_t C) {
  NPU_TID;
  const int64_t c = i % C, w = (i / C) % W, h = i / (C * W);
  const BIdx ia = bidx(h, w, c, abc, W, C), ib = bidx(h, w, c, bbc, W, C);
  // fp8 rows are 8 consecutive channels of the operand's own pixel.
  const float fa = mul_q(fp8_decode_identity(a + ia.pixel, ia.c, ia.cn), as);
  const float fb = mul_q(fp8_decode_identity(b + ib.pixel, ib.c, ib.cn), bs);
  if (op == kEq || op == kGt || op == kGe) {
    const bool r = op == kEq ? veq(fa, fb) : op == kGt ? vgt(fa, fb) : vge(fa, fb);
    static_cast<int8_t*>(out)[i] = r ? 1 : 0;
    return;
  }
  store_float(out, NPU_FP8, i, op == kMax ? vmax(fa, fb) : vmin(fa, fb), kHalfAway);
}

// Comparable key: integer value, or decoded BF16 / F16 / FP8 (no row decode).
struct Key {
  bool is_float;
  int64_t iv;
  float fv;
};

__device__ inline Key load_key(const void* p, npu_dtype t, int64_t i) {
  if (t == NPU_FP8) return {true, 0, fp8_to_f32(static_cast<const uint8_t*>(p)[i])};
  if (is_half_float(t)) return {true, 0, load_float(p, t, i)};
  return {false, load_int(p, t, i), 0.0f};
}

// Strictly better (greater for max, less for min); NaN is never better.
__device__ inline bool better(const Key& x, const Key& m, bool is_max) {
  if (x.is_float) return is_max ? vgt(x.fv, m.fv) : vgt(m.fv, x.fv);
  return is_max ? x.iv > m.iv : x.iv < m.iv;
}

__device__ inline bool at_least(const Key& x, const Key& m, bool is_max) {
  if (x.is_float) return is_max ? vge(x.fv, m.fv) : vge(m.fv, x.fv);
  return is_max ? x.iv >= m.iv : x.iv <= m.iv;
}

__device__ inline void store_index(void* out, npu_dtype t, int64_t i, uint32_t idx) {
  switch (t) {
    case NPU_U8:
      static_cast<uint8_t*>(out)[i] = uint8_t(idx);
      break;
    case NPU_U16:
      static_cast<uint16_t*>(out)[i] = uint16_t(idx);
      break;
    case NPU_U32:
      static_cast<uint32_t*>(out)[i] = idx;
      break;
    case NPU_F32:
      static_cast<float*>(out)[i] = float(idx);
      break;
    default:
      static_cast<int64_t*>(out)[i] = int64_t(idx);
      break;
  }
}

// Lane L sees elements L, L + 64, ... and keeps the first index of its best
// (modulo `mod`); the result is the smallest wrapped index among the lanes
// holding the overall best.
__device__ inline uint32_t lane_scan(const void* in, npu_dtype t, int64_t base,
                                     int64_t step, int64_t len, bool is_max,
                                     Key init, uint32_t mod, Key* best_out) {
  Key best = init;
  bool have = false;
  for (int L = 0; L < 64 && L < len; ++L) {
    Key m = init;
    for (int64_t j = L; j < len; j += 64) {
      const Key x = load_key(in, t, base + j * step);
      if (better(x, m, is_max)) m = x;
    }
    if (!have || better(m, best, is_max)) best = m;
    have = true;
  }
  uint32_t res = 0xffffffffu;
  for (int L = 0; L < 64 && L < len; ++L) {
    Key m = init;
    uint32_t idx = 0;
    for (int64_t j = L; j < len; j += 64) {
      const Key x = load_key(in, t, base + j * step);
      if (better(x, m, is_max)) {
        m = x;
        idx = uint32_t((j - L) % mod);
      }
    }
    if (at_least(m, best, is_max)) {
      const uint32_t cand = (idx + uint32_t(L)) % mod;
      res = cand < res ? cand : res;
    }
  }
  if (best_out) *best_out = best;
  return res;
}

struct ArgParams {
  npu_dtype in_type, out_type;
  int64_t H, W, C;
  int axis;
  bool is_max;
  Key init;
  uint32_t mod;
};

__global__ void arg_reduce_kernel(int64_t n, const void* in, void* out, ArgParams p) {
  NPU_TID;
  uint32_t idx;
  if (p.axis == 2) {
    idx = lane_scan(in, p.in_type, i * p.C, 1, p.C, p.is_max, p.init, p.mod, nullptr);
  } else {
    int64_t base, step, len;
    if (p.axis == 1) {  // out (H, C)
      const int64_t h = i / p.C, c = i % p.C;
      base = h * p.W * p.C + c, step = p.C, len = p.W;
    } else {  // out (W, C)
      base = i, step = p.W * p.C, len = p.H;
    }
    Key m = p.init;
    idx = 0;
    for (int64_t j = 0; j < len; ++j) {
      const Key x = load_key(in, p.in_type, base + j * step);
      if (better(x, m, p.is_max)) {
        m = x;
        idx = uint32_t(j % p.mod);
      }
    }
  }
  store_index(out, p.out_type, i, idx);
}

int arg_reduce(void* stream, const void* in, npu_dtype in_type, void* out,
               npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis,
               bool is_max, Key init, int bits) {
  if (axis < 0 || axis > 2 || H <= 0 || W <= 0 || C <= 0) return -1;
  ArgParams p{in_type, out_type, H, W, C, axis, is_max, init, 1u << bits};
  const int64_t n = H * W * C / (axis == 0 ? H : axis == 1 ? W : C);
  return launch(stream, n, arg_reduce_kernel, in, out, p);
}

inline Key int_key(int64_t v) { return {false, v, 0.0f}; }
inline Key float_key(float v) { return {true, 0, v}; }

inline bool int_in(npu_dtype t, npu_dtype a, npu_dtype b) { return t == a || t == b; }
inline bool idx_out_wide(npu_dtype t) {
  return t == NPU_U8 || t == NPU_U16 || t == NPU_U32 || t == NPU_I64 || t == NPU_F32;
}
inline bool idx_out_narrow(npu_dtype t) { return t == NPU_U8 || t == NPU_U16; }

int64_t type_min(npu_dtype t) {
  int64_t lo, hi;
  int_range(t, &lo, &hi);
  return lo;
}
int64_t type_max(npu_dtype t) {
  int64_t lo, hi;
  int_range(t, &lo, &hi);
  return hi;
}

// Smallest positive normal / largest finite value.
inline float half_min_normal(npu_dtype t) { return t == NPU_F16 ? 0x1p-14f : 0x1p-126f; }
inline float half_max_finite(npu_dtype t) {
  return t == NPU_F16 ? 65504.0f : 0x1.fep127f;
}
inline uint16_t half_lowest_bits(npu_dtype t) { return t == NPU_F16 ? 0xfbff : 0xff7f; }
inline uint16_t half_max_bits(npu_dtype t) { return t == NPU_F16 ? 0x7bff : 0x7f7f; }

struct RedParams {
  npu_dtype type;
  int64_t H, W, C;
  int axes;
  bool is_max;
  uint16_t init;  // start value (bits)
};

__global__ void reduce_half_kernel(int64_t n, const uint16_t* in, uint16_t* out, RedParams p) {
  NPU_TID;
  const int64_t nc = p.axes & 1 ? 1 : p.C, nw = p.axes & 2 ? 1 : p.W;
  const int64_t oc = i % nc, ow = (i / nc) % nw, oh = i / (nc * nw);
  const int64_t hn = p.axes & 4 ? p.H : 1, wn = p.axes & 2 ? p.W : 1, cn = p.axes & 1 ? p.C : 1;
  uint16_t r = p.init;
  for (int64_t h = 0; h < hn; ++h)
    for (int64_t w = 0; w < wn; ++w)
      for (int64_t c = 0; c < cn; ++c) {
        const int64_t hh = p.axes & 4 ? h : oh, ww = p.axes & 2 ? w : ow, cc = p.axes & 1 ? c : oc;
        const uint16_t x = in[(hh * p.W + ww) * p.C + cc];
        r = p.is_max ? max16(r, x, p.type) : min16(r, x, p.type);
      }
  out[i] = r;
}

struct PoolParams {
  int64_t Wi, C, Ho, Wo;
  int kh, kw, sh, sw;
};

__global__ void maxpool_half_kernel(int64_t n, const uint16_t* in, npu_dtype type,
                                    uint16_t* out, PoolParams p) {
  NPU_TID;
  const int64_t c = i % p.C, wo = (i / p.C) % p.Wo, ho = i / (p.C * p.Wo);
  const int64_t y0 = ho * p.sh, x0 = wo * p.sw;
  uint16_t m = in[(y0 * p.Wi + x0) * p.C + c];
  for (int ty = 0; ty < p.kh; ++ty)
    for (int tx = 0; tx < p.kw; ++tx)
      m = max16(m, in[((y0 + ty) * p.Wi + x0 + tx) * p.C + c], type);
  out[i] = m;
}

// fp8 compare-and-select max: -0 below +0; NaN never replaces the running max.
__device__ inline bool fp8_gt(uint8_t a, uint8_t b) {
  return above(fp8_to_f32(a), fp8_to_f32(b));
}

// fma(v, c2, c0), optional clamp, fp8 half to even.
__device__ inline uint8_t fp8_requant(float v, float c2, float c0, int clip,
                                      float lo, float hi) {
  float r = fma_q(v, c2, c0);
  if (clip) r = vclamp(r, lo, hi);
  return f32_to_fp8(r, kHalfEven);
}

struct Fp8PoolParams {
  PoolParams pool;
  const float* c0;
  const float* c2;
  int64_t channels;
  int clip;
  float lo, hi;
};

// One thread per output pixel and 8-channel row.
__global__ void maxpool_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out,
                                   Fp8PoolParams q) {
  NPU_TID;
  const PoolParams& p = q.pool;
  const int64_t rows = (p.C + 7) / 8;
  const int64_t r = i % rows, wo = (i / rows) % p.Wo, ho = i / (rows * p.Wo);
  const int64_t y0 = ho * p.sh, x0 = wo * p.sw;
  const int64_t c0 = r * 8, cn = p.C - c0 < 8 ? p.C - c0 : 8;
  uint8_t m[8];
  for (int k = 0; k < cn; ++k) {
    m[k] = in[(y0 * p.Wi + x0) * p.C + c0 + k];
    for (int ty = 0; ty < p.kh; ++ty)
      for (int tx = 0; tx < p.kw; ++tx) {
        const uint8_t x = in[((y0 + ty) * p.Wi + x0 + tx) * p.C + c0 + k];
        if (fp8_gt(x, m[k])) m[k] = x;
      }
  }
  uint8_t* o = out + ((ho * p.Wo + wo) * p.C + c0);
  for (int k = 0; k < cn; ++k) {
    const float v = fp8_decode_identity(m, k, cn);
    const int64_t ch = q.channels == 1 ? 0 : c0 + k;
    o[k] = fp8_requant(v, q.c2[ch], q.c0[ch], q.clip, q.lo, q.hi);
  }
}

__global__ void globalmaxpool_half_kernel(int64_t n, const uint16_t* in, npu_dtype type,
                                          uint16_t* out, int64_t HW, uint16_t init) {
  NPU_TID;
  uint16_t m = init;
  for (int64_t j = 0; j < HW; ++j) m = max16(m, in[j * n + i], type);
  out[i] = m;
}

__global__ void globalmaxpool_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out,
                                         int64_t HW, int64_t C, float c0, float c1) {
  NPU_TID;
  const int64_t ch0 = i * 8, cn = C - ch0 < 8 ? C - ch0 : 8;
  uint8_t m[8];
  for (int k = 0; k < cn; ++k) {
    m[k] = in[ch0 + k];
    for (int64_t j = 1; j < HW; ++j) {
      const uint8_t x = in[j * C + ch0 + k];
      if (fp8_gt(x, m[k])) m[k] = x;
    }
  }
  for (int k = 0; k < cn; ++k)
    out[ch0 + k] = fp8_requant(fp8_decode_identity(m, k, cn), c1, c0, 0, 0.0f, 0.0f);
}

// `work` holds rows x Wp elements, the tail padded with the start value; each
// pick is overwritten there.
__global__ void topk_pad_kernel(int64_t n, const void* in, npu_dtype t, void* work,
                                int64_t W, int64_t Wp, uint64_t init_bits) {
  NPU_TID;
  const int64_t row = i / Wp, col = i % Wp;
  const int sz = t == NPU_I8 || t == NPU_U8 ? 1 : 2;
  const uint64_t v = col < W ? (sz == 1 ? static_cast<const uint8_t*>(in)[row * W + col]
                                        : static_cast<const uint16_t*>(in)[row * W + col])
                             : init_bits;
  if (sz == 1)
    static_cast<uint8_t*>(work)[i] = uint8_t(v);
  else
    static_cast<uint16_t*>(work)[i] = uint16_t(v);
}

struct TopkParams {
  npu_dtype type, index_type;
  int64_t Wp, K;
  bool largest;
  Key init;
  uint64_t init_bits;
  uint32_t mod;
};

// Value order of the running best: -0 below +0 for floats.
__device__ inline bool value_better(const Key& x, const Key& m, bool is_max) {
  if (x.is_float) return is_max ? above(x.fv, m.fv) : above(m.fv, x.fv);
  return is_max ? x.iv > m.iv : x.iv < m.iv;
}

__global__ void topk_kernel(int64_t n, void* work, void* out_val, void* out_idx, TopkParams p) {
  NPU_TID;
  const int64_t base = i * p.Wp;
  const int esz = p.type == NPU_I8 || p.type == NPU_U8 ? 1 : 2;
  for (int64_t k = 0; k < p.K; ++k) {
    Key best = p.init;
    int64_t best_j = -1;  // element holding the overall best
    uint32_t res = 0xffffffffu;
    // Per lane the index moves on a strict IEEE improvement, the value on the
    // -0 < +0 order.
    Key lane_val[64];
    uint32_t lane_idx[64];
    for (int L = 0; L < 64; ++L) {
      Key m = p.init;
      uint32_t idx = 0;
      int64_t src = -1;
      for (int64_t j = L; j < p.Wp; j += 64) {
        const Key x = load_key(work, p.type, base + j);
        if (better(x, m, p.largest)) idx = uint32_t((j - L) % p.mod);
        if (value_better(x, m, p.largest)) m = x, src = j;
      }
      lane_val[L] = m, lane_idx[L] = idx;
      if (value_better(m, best, p.largest)) best = m, best_j = src;
    }
    for (int L = 0; L < 64; ++L) {
      if (at_least(lane_val[L], best, p.largest)) {
        const uint32_t cand = (lane_idx[L] + uint32_t(L)) % p.mod;
        res = cand < res ? cand : res;
      }
    }
    uint64_t bits = p.init_bits;
    if (best_j >= 0)
      bits = esz == 1 ? static_cast<const uint8_t*>(work)[base + best_j]
                      : static_cast<const uint16_t*>(work)[base + best_j];
    const int64_t o = i * p.K + k;
    if (esz == 1)
      static_cast<uint8_t*>(out_val)[o] = uint8_t(bits);
    else
      static_cast<uint16_t*>(out_val)[o] = uint16_t(bits);
    if (p.index_type == NPU_U8)
      static_cast<uint8_t*>(out_idx)[o] = uint8_t(res);
    else
      static_cast<uint16_t*>(out_idx)[o] = uint16_t(res);
    if (int64_t(res) < p.Wp) {
      if (esz == 1)
        static_cast<uint8_t*>(work)[base + res] = uint8_t(p.init_bits);
      else
        static_cast<uint16_t*>(work)[base + res] = uint16_t(p.init_bits);
    }
  }
}

int topk(void* stream, const void* in, npu_dtype type, void* out_val, void* out_idx,
         npu_dtype index_type, int64_t rows, int64_t W, int64_t K, int largest) {
  if (rows <= 0 || W <= 0 || K <= 0) return rows == 0 || K == 0 ? 0 : -1;
  TopkParams p;
  p.type = type, p.index_type = index_type, p.K = K, p.largest = largest != 0;
  p.Wp = (W + 63) / 64 * 64;
  p.mod = index_type == NPU_U8 ? 256u : 65536u;
  if (is_half_float(type)) {
    p.init_bits = largest ? (type == NPU_F16 ? 0xfc00 : 0xff80) : (type == NPU_F16 ? 0x7c00 : 0x7f80);
    p.init = float_key(largest ? -INFINITY : INFINITY);
  } else {
    const int64_t v = largest ? type_min(type) : type_max(type);
    p.init_bits = uint64_t(v);
    p.init = int_key(v);
  }
  const size_t esz = type == NPU_I8 || type == NPU_U8 ? 1 : 2;
  hipStream_t s = static_cast<hipStream_t>(stream);
  void* work = nullptr;
  (void)hipGetLastError();
  int rc = int(hipMallocAsync(&work, size_t(rows * p.Wp) * esz, s));
  if (rc) return rc;
  rc = launch(stream, rows * p.Wp, topk_pad_kernel, in, type, work, W, p.Wp, p.init_bits);
  if (!rc) rc = launch(stream, rows, topk_kernel, work, out_val, out_idx, p);
  const int rc2 = int(hipFreeAsync(work, s));
  return rc ? rc : rc2;
}

}  // namespace

extern "C" {

int npu_elementwise_abs_bf16(void* stream, const void* in, npu_dtype type, void* out,
                             int64_t n) {
  if (!is_half_float(type)) return -1;
  return launch(stream, n, abs_half_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out));
}

int npu_elementwise_neg_bf16(void* stream, const void* in, npu_dtype type, void* out,
                             int64_t n) {
  if (!is_half_float(type)) return -1;
  return launch(stream, n, neg_half_kernel, static_cast<const uint16_t*>(in), type,
                static_cast<uint16_t*>(out));
}

int npu_elementwise_clip_bf16(void* stream, const void* in, npu_dtype type, void* out,
                              int64_t n, float lo, float hi) {
  if (!is_half_float(type)) return -1;
  return launch(stream, n, clip_half_kernel, static_cast<const uint16_t*>(in), type,
                static_cast<uint16_t*>(out), uint16_t(encode_float(lo, type, kHalfEven)),
                uint16_t(encode_float(hi, type, kHalfEven)));
}

int npu_elementwise_abs_fp8(void* stream, const void* in, void* out, int64_t n,
                            float scale) {
  return launch(stream, n, fp8_unary_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), kAbs, scale, 0.0f, 0.0f);
}

int npu_elementwise_neg_fp8(void* stream, const void* in, void* out, int64_t n,
                            float scale) {
  return launch(stream, n, fp8_unary_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), kNeg, scale, 0.0f, 0.0f);
}

int npu_elementwise_clip_fp8(void* stream, const void* in, void* out, int64_t n,
                             float scale, float lo, float hi) {
  return launch(stream, n, fp8_unary_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), kClip, scale, lo, hi);
}

// The full shape and both operands' broadcast bits; false if the shapes do not broadcast.
static bool bcast_args(const int64_t* sa, const int64_t* sb, int64_t* f, int* abc, int* bbc) {
  if (!bcast_shape(sa, sb, f)) return false;
  *abc = bcast_bits(f, sa);
  *bbc = bcast_bits(f, sb);
  return true;
}

static int bcast_half(void* stream, const void* a, const void* b, npu_dtype type, void* out,
                      const int64_t* sa, const int64_t* sb, BinOp op) {
  int64_t f[3];
  int abc, bbc;
  if (!is_half_float(type) || !bcast_args(sa, sb, f, &abc, &bbc)) return -1;
  return launch(stream, f[0] * f[1] * f[2], bcast_half_kernel, static_cast<const uint16_t*>(a),
                abc, static_cast<const uint16_t*>(b), bbc, type, out, op, f[1], f[2]);
}

static int bcast_fp8(void* stream, const void* a, float as, const void* b, float bs, void* out,
                     const int64_t* sa, const int64_t* sb, BinOp op) {
  int64_t f[3];
  int abc, bbc;
  if (!bcast_args(sa, sb, f, &abc, &bbc)) return -1;
  return launch(stream, f[0] * f[1] * f[2], bcast_fp8_kernel, static_cast<const uint8_t*>(a), abc,
                as, static_cast<const uint8_t*>(b), bbc, bs, out, op, f[1], f[2]);
}

int npu_elementwise_broadcasting3d_equal_bf16(void* stream, const void* a, const void* b,
    npu_dtype type, void* out, const int64_t* shape_a, const int64_t* shape_b) {
  return bcast_half(stream, a, b, type, out, shape_a, shape_b, kEq);
}
int npu_elementwise_broadcasting3d_greater_bf16(void* stream, const void* a, const void* b,
    npu_dtype type, void* out, const int64_t* shape_a, const int64_t* shape_b) {
  return bcast_half(stream, a, b, type, out, shape_a, shape_b, kGt);
}
int npu_elementwise_broadcasting3d_greater_or_equal_bf16(void* stream, const void* a, const void* b,
    npu_dtype type, void* out, const int64_t* shape_a, const int64_t* shape_b) {
  return bcast_half(stream, a, b, type, out, shape_a, shape_b, kGe);
}
int npu_elementwise_broadcasting3d_max_bf16(void* stream, const void* a, const void* b,
    npu_dtype type, void* out, const int64_t* shape_a, const int64_t* shape_b) {
  return bcast_half(stream, a, b, type, out, shape_a, shape_b, kMax);
}
int npu_elementwise_broadcasting3d_min_bf16(void* stream, const void* a, const void* b,
    npu_dtype type, void* out, const int64_t* shape_a, const int64_t* shape_b) {
  return bcast_half(stream, a, b, type, out, shape_a, shape_b, kMin);
}

int npu_elementwise_broadcasting3d_equal_fp8(void* stream, const void* a, float a_scale,
    const void* b, float b_scale, void* out, const int64_t* shape_a, const int64_t* shape_b) {
  return bcast_fp8(stream, a, a_scale, b, b_scale, out, shape_a, shape_b, kEq);
}
int npu_elementwise_broadcasting3d_greater_fp8(void* stream, const void* a, float a_scale,
    const void* b, float b_scale, void* out, const int64_t* shape_a, const int64_t* shape_b) {
  return bcast_fp8(stream, a, a_scale, b, b_scale, out, shape_a, shape_b, kGt);
}
int npu_elementwise_broadcasting3d_greater_or_equal_fp8(void* stream, const void* a, float a_scale,
    const void* b, float b_scale, void* out, const int64_t* shape_a, const int64_t* shape_b) {
  return bcast_fp8(stream, a, a_scale, b, b_scale, out, shape_a, shape_b, kGe);
}
int npu_elementwise_broadcasting3d_max_fp8(void* stream, const void* a, float a_scale,
    const void* b, float b_scale, void* out, const int64_t* shape_a, const int64_t* shape_b) {
  return bcast_fp8(stream, a, a_scale, b, b_scale, out, shape_a, shape_b, kMax);
}
int npu_elementwise_broadcasting3d_min_fp8(void* stream, const void* a, float a_scale,
    const void* b, float b_scale, void* out, const int64_t* shape_a, const int64_t* shape_b) {
  return bcast_fp8(stream, a, a_scale, b, b_scale, out, shape_a, shape_b, kMin);
}

int npu_arg_max_c64_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                       npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis) {
  if (!int_in(in_type, NPU_I8, NPU_U8) || !idx_out_wide(out_type)) return -1;
  return arg_reduce(stream, in, in_type, out, out_type, H, W, C, axis, true,
                    int_key(type_min(in_type)), 16);
}

int npu_arg_max_c64_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                        npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis) {
  if (!int_in(in_type, NPU_I16, NPU_U16) || !idx_out_narrow(out_type)) return -1;
  return arg_reduce(stream, in, in_type, out, out_type, H, W, C, axis, true,
                    int_key(type_min(in_type)), out_type == NPU_U8 ? 8 : 16);
}

int npu_arg_max_c64_a16_int64(void* stream, const void* in, npu_dtype in_type, void* out,
                              int64_t H, int64_t W, int64_t C, int axis) {
  if (!int_in(in_type, NPU_I16, NPU_U16)) return -1;
  return arg_reduce(stream, in, in_type, out, NPU_I64, H, W, C, axis, true,
                    int_key(type_min(in_type)), 16);
}

int npu_arg_max_c64_a32(void* stream, const void* in, npu_dtype in_type, void* out,
                        npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis) {
  if (!int_in(in_type, NPU_I32, NPU_U32) || !idx_out_wide(out_type)) return -1;
  return arg_reduce(stream, in, in_type, out, out_type, H, W, C, axis, true,
                    int_key(type_min(in_type)), 16);
}

int npu_arg_max_c64_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                         npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis) {
  if (!is_half_float(in_type) || !idx_out_narrow(out_type)) return -1;
  return arg_reduce(stream, in, in_type, out, out_type, H, W, C, axis, true,
                    float_key(half_min_normal(in_type)), out_type == NPU_U8 ? 8 : 16);
}

int npu_arg_max_c64_fp8(void* stream, const void* in, void* out, npu_dtype out_type,
                        int64_t H, int64_t W, int64_t C, int axis) {
  if (!idx_out_wide(out_type)) return -1;
  return arg_reduce(stream, in, NPU_FP8, out, out_type, H, W, C, axis, true,
                    float_key(-448.0f), 16);
}

int npu_arg_min_c64_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                       npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis) {
  if (!int_in(in_type, NPU_I8, NPU_U8) || !idx_out_narrow(out_type)) return -1;
  return arg_reduce(stream, in, in_type, out, out_type, H, W, C, axis, false,
                    int_key(type_max(in_type)), out_type == NPU_U8 ? 8 : 16);
}

int npu_arg_min_c64_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                        npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis) {
  if (!int_in(in_type, NPU_I16, NPU_U16) || !idx_out_narrow(out_type)) return -1;
  return arg_reduce(stream, in, in_type, out, out_type, H, W, C, axis, false,
                    int_key(type_max(in_type)), out_type == NPU_U8 ? 8 : 16);
}

int npu_arg_min_c64_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                         npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis) {
  if (!is_half_float(in_type) || !idx_out_narrow(out_type)) return -1;
  return arg_reduce(stream, in, in_type, out, out_type, H, W, C, axis, false,
                    float_key(half_max_finite(in_type)), out_type == NPU_U8 ? 8 : 16);
}

static int reduce_half(void* stream, const void* in, npu_dtype type, void* out, int64_t H,
                       int64_t W, int64_t C, int axes, bool is_max) {
  if (!is_half_float(type) || axes < 1 || axes > 7 || H <= 0 || W <= 0 || C <= 0) return -1;
  RedParams p{type, H, W, C, axes, is_max,
              is_max ? half_lowest_bits(type) : half_max_bits(type)};
  const int64_t n = (axes & 4 ? 1 : H) * (axes & 2 ? 1 : W) * (axes & 1 ? 1 : C);
  return launch(stream, n, reduce_half_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), p);
}

int npu_reduce_max_c64_bf16(void* stream, const void* in, npu_dtype type, void* out,
                            int64_t H, int64_t W, int64_t C, int axes) {
  return reduce_half(stream, in, type, out, H, W, C, axes, true);
}

int npu_reduce_min_c64_bf16(void* stream, const void* in, npu_dtype type, void* out,
                            int64_t H, int64_t W, int64_t C, int axes) {
  return reduce_half(stream, in, type, out, H, W, C, axes, false);
}

static bool pool_params(int64_t Hi, int64_t Wi, int64_t C, int kh, int kw, int sh, int sw,
                        PoolParams* p) {
  if (kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || Hi < kh || Wi < kw || C <= 0) return false;
  *p = {Wi, C, (Hi - kh) / sh + 1, (Wi - kw) / sw + 1, kh, kw, sh, sw};
  return true;
}

int npu_maxpool_bf16(void* stream, const void* in, npu_dtype type, void* out, int64_t Hi,
                     int64_t Wi, int64_t C, int kh, int kw, int sh, int sw) {
  PoolParams p;
  if (!is_half_float(type) || !pool_params(Hi, Wi, C, kh, kw, sh, sw, &p)) return -1;
  return launch(stream, p.Ho * p.Wo * C, maxpool_half_kernel,
                static_cast<const uint16_t*>(in), type, static_cast<uint16_t*>(out), p);
}

int npu_maxpool_bf16_c32(void* stream, const void* in, void* out, int64_t Hi, int64_t Wi,
                         int64_t C, int kh, int kw, int sh, int sw) {
  return npu_maxpool_bf16(stream, in, NPU_BF16, out, Hi, Wi, C, kh, kw, sh, sw);
}

int npu_maxpool_fp8(void* stream, const void* in, void* out, int64_t Hi, int64_t Wi,
                    int64_t C, int kh, int kw, int sh, int sw, const float* c0,
                    const float* c2, int64_t channels, int clip, float act_lo,
                    float act_hi) {
  Fp8PoolParams q;
  if (!pool_params(Hi, Wi, C, kh, kw, sh, sw, &q.pool) || (channels != 1 && channels != C))
    return -1;
  q.c0 = c0, q.c2 = c2, q.channels = channels, q.clip = clip, q.lo = act_lo, q.hi = act_hi;
  return launch(stream, q.pool.Ho * q.pool.Wo * ((C + 7) / 8), maxpool_fp8_kernel,
                static_cast<const uint8_t*>(in), static_cast<uint8_t*>(out), q);
}

int npu_globalmaxpool_bf16(void* stream, const void* in, npu_dtype type, void* out,
                           int64_t H, int64_t W, int64_t C) {
  if (!is_half_float(type) || H <= 0 || W <= 0) return -1;
  return launch(stream, C, globalmaxpool_half_kernel, static_cast<const uint16_t*>(in), type,
                static_cast<uint16_t*>(out), H * W,
                uint16_t(type == NPU_F16 ? 0x0400 : 0x0080));
}

int npu_globalmaxpool_fp8(void* stream, const void* in, void* out, int64_t H, int64_t W,
                          int64_t C, float c0, float c1) {
  if (H <= 0 || W <= 0) return -1;
  return launch(stream, (C + 7) / 8, globalmaxpool_fp8_kernel,
                static_cast<const uint8_t*>(in), static_cast<uint8_t*>(out), H * W, C, c0,
                c1);
}

int npu_topk2d_bf16(void* stream, const void* in, npu_dtype type, void* out_values,
                    void* out_indices, npu_dtype index_type, int64_t rows, int64_t W,
                    int64_t K, int largest) {
  if (!is_half_float(type) || !idx_out_narrow(index_type)) return -1;
  return topk(stream, in, type, out_values, out_indices, index_type, rows, W, K, largest);
}

int npu_topk2d_a8(void* stream, const void* in, npu_dtype type, void* out_values,
                  void* out_indices, npu_dtype index_type, int64_t rows, int64_t W, int64_t K,
                  int largest) {
  if (!int_in(type, NPU_I8, NPU_U8) || !idx_out_narrow(index_type)) return -1;
  return topk(stream, in, type, out_values, out_indices, index_type, rows, W, K, largest);
}

int npu_topk2d_a16(void* stream, const void* in, npu_dtype type, void* out_values,
                   void* out_indices, npu_dtype index_type, int64_t rows, int64_t W,
                   int64_t K, int largest) {
  if (!int_in(type, NPU_I16, NPU_U16) || index_type != NPU_U16) return -1;
  return topk(stream, in, type, out_values, out_indices, index_type, rows, W, K, largest);
}

}  // extern "C"
