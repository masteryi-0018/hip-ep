/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Flash attention kernels and entry points. Per KV block: scores (with masks), softmax (row
 * max, P, row sum, l) and PV with the O update; a last kernel rescales and converts O. Rows
 * whose Q block does not visit the KV block keep their state.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "attention/attention.h"
#include "common/block_fp.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/mmac.h"
#include "common/nlf.h"

using namespace npu_emu;

namespace {

enum Kind { kBf16 = 0, kQ16 = 1, kU8 = 2, kMx = 3 };

constexpr float kLog2e = 1.4453125f;       // bf16 0x3fb9
constexpr float kMaxBias = -895.00099f;    // folded into max * log2e
constexpr float kExp2Bias = 895.001f;      // input bias of the exp2 table

// The fp32 MACs flush subnormal operands and results to a signed zero.
constexpr int kFtz = kFtzIn | kFtzOut;

struct Fa {
  npu_flash_attention_cfg c;
  int kind;
  npu_dtype t16;    // the type of 16-bit integer tensors
  const void* q;
  const void* k;
  const void* v;
  const uint16_t* mask;
  void* out;
  float* s;         // [H][M][bc] scores, then P (values / codes / mx mantissas)
  uint8_t* pe;      // [H][M][bc / 8] mx exponents of P
  float* m;         // [H][M]
  float* l;         // [H][M]
  float* emd;       // [H][M]
  float* o;         // [H][M][dv]
  const uint32_t* inv_exc;
};

// Rounding of the alt_round-dependent conversions: ties away, or to even with alt_round.
__device__ inline Rnd rnd(const Fa& f) { return f.c.alt_round ? kHalfEven : kHalfAway; }

__device__ inline Rnd mode_s(const Fa& f) {
  if (f.kind == kU8) return kHalfEven;
  return (!f.c.variant_a && f.c.qdq <= 1) ? rnd(f) : kHalfEven;
}

__device__ inline int64_t nblocks(const Fa& f) { return f.c.kv_len / f.c.bc; }

// First Q position of Q block sc (causal) and the KV block range.
__device__ inline int64_t q_block_pos(const Fa& f, int64_t sc) {
  const int sh = f.c.q_shards;
  const int wrap = f.c.q_wrap > 0 ? f.c.q_wrap : 1;
  return f.c.q_offset + (sh > 0 ? (int64_t(f.c.q_shard) + int64_t(sh) * (sc % wrap)) * f.c.br : 0);
}

// [default] floor division: a negative q_offset must not round toward zero into block 0.
__device__ inline int64_t floor_div(int64_t a, int64_t b) { return a / b - (a % b < 0 ? 1 : 0); }

__device__ inline bool visits(const Fa& f, int64_t sc, int64_t blk) {
  if (f.c.mask_type != 1) return true;
  const int64_t qb = q_block_pos(f, sc), bc = f.c.bc;
  const int64_t end = floor_div(qb, bc) + 1;  // the block holding the first Q position
  int64_t begin = 0;
  const int64_t w = f.c.sliding_window_size;
  if (f.c.sliding && w > 0 && qb - w + 1 >= bc) begin = (qb - w + 1) / bc;
  return blk >= begin && blk < end;
}

__device__ inline int valid_keys(const Fa& f, int64_t blk) {
  const int v = f.c.kv_valid_last;
  return (blk == nblocks(f) - 1 && v > 0 && v < f.c.bc) ? v : f.c.bc;
}

// Row sum of the uint8 QDQ MatMuls over n values. With more than one 64-wide chunk and
// m_granularity >= 32: chunks 1 .. t-1, the last one counted twice.
__device__ inline int64_t act_sum_u8(const uint8_t* a, int64_t n, int m_gran) {
  const int64_t t = (n + 63) / 64;
  int64_t s = 0;
  if (t <= 1 || m_gran < 32) {
    for (int64_t i = 0; i < n; ++i) s += a[i];
    return s;
  }
  for (int64_t i = 64; i < n; ++i) s += a[i];
  for (int64_t i = (t - 1) * 64; i < n; ++i) s += a[i];
  return s;
}

// The same on P codes kept as floats.
__device__ inline int64_t act_sum_f(const float* a, int64_t n) {
  const int64_t t = (n + 63) / 64;
  int64_t s = 0;
  if (t <= 1) {
    for (int64_t i = 0; i < n; ++i) s += int64_t(a[i]);
    return s;
  }
  for (int64_t i = 64; i < n; ++i) s += int64_t(a[i]);
  for (int64_t i = (t - 1) * 64; i < n; ++i) s += int64_t(a[i]);
  return s;
}

// QDQ mode 2 int16x16 GEMM output for one row x column given the byte-split
// integer sums.
struct Sums16 {
  int64_t ll = 0, mix = 0, hh = 0, alo = 0, ahi = 0, wlo = 0, whi = 0;
  __device__ void add(int64_t a, int64_t w) {
    a &= 0xffff;
    w &= 0xffff;
    const int64_t al = a & 0xff, ah = a >> 8, wl = w & 0xff, wh = w >> 8;
    ll += al * wl;
    mix += al * wh + ah * wl;
    hh += ah * wh;
    alo += al;
    ahi += ah;
    wlo += wl;
    whi += wh;
  }
  __device__ float result(const float* c) const {
    const float a_sum = mac_exact(ahi, 256.0f, float(alo), kFtz);
    const float w_lo = mac_exact(wlo, c[3], c[0], kFtz);
    const float w_sum = mac_exact(whi, __fmul_rn(c[3], 256.0f), w_lo, kFtz);
    float acc = mac_exact(ll, c[2], w_sum, kFtz);
    acc = mac_exact(mix, __fmul_rn(c[2], 256.0f), acc, kFtz);
    acc = mac_exact(hh, __fmul_rn(c[2], 65536.0f), acc, kFtz);
    return fma_ftz(c[1], a_sum, acc, kFtz);
  }
};

__global__ void fa_init_kernel(int64_t n, Fa f) {
  NPU_TID;
  f.m[i] = -INFINITY;
  f.l[i] = 0.0f;
}

// One score of KV block blk: thread (h, row, key).
__global__ void fa_scores_kernel(int64_t n, Fa f, int64_t blk) {
  NPU_TID;
  const int64_t bc = f.c.bc, M = f.c.q_len, N = f.c.kv_len, d = f.c.head_dim;
  const int64_t col = i % bc, hr = i / bc, row = hr % M, h = hr / M;
  const int64_t sc = row / f.c.br;
  if (!visits(f, sc, blk)) return;
  const int64_t key = blk * bc + col;
  const float* qc = f.c.qk_c;
  float s;
  if (f.kind == kBf16) {
    const uint16_t* q = static_cast<const uint16_t*>(f.q) + (h * M + row) * d;
    const uint16_t* k = static_cast<const uint16_t*>(f.k) + (h * N + key) * d;
    uint16_t x[32], z[32];
    uint32_t acc = 0;
    for (int64_t k0 = 0; k0 < d; k0 += 32) {
      for (int p = 0; p < 32; ++p) {
        x[p] = k0 + p < d ? q[k0 + p] : 0;
        z[p] = k0 + p < d ? k[k0 + p] : 0;
      }
      acc = mmac_bf16(acc, x, 1, z, 1, 32);
    }
    // times the bf16 scale (16-bit vector multiply)
    const float qk = round_float(__uint_as_float(acc), NPU_BF16, kHalfAway, kFtzIn);
    s = mul16_f32(qk, bf16_to_f32(uint16_t(f.c.alpha_bf16)), NPU_BF16);
  } else if (f.kind == kQ16 && f.c.qdq == 1) {
    // Q / K dequantized to bf16 (ties away), then the bf16 MMAC. No softmax scale: it is
    // folded into the dequant coefficients (the kernel's masked QK path still multiplies by
    // its bf16 scalar, which is then 1.0).
    const int64_t qo = (h * M + row) * d, ko = (h * N + key) * d;
    uint16_t x[32], z[32];
    uint32_t acc = 0;
    for (int64_t k0 = 0; k0 < d; k0 += 32) {
      for (int p = 0; p < 32; ++p) {
        x[p] = z[p] = 0;
        if (k0 + p >= d) continue;
        const float xq = mac_exact(load_int(f.q, f.t16, qo + k0 + p), qc[0], qc[1], kFtz);
        const float zk = mac_exact(load_int(f.k, f.t16, ko + k0 + p), qc[2], qc[3], kFtz);
        x[p] = f32_to_bf16(xq, kHalfAway, kFtzIn);
        z[p] = f32_to_bf16(zk, kHalfAway, kFtzIn);
      }
      acc = mmac_bf16(acc, x, 1, z, 1, 32);
    }
    s = round_float(__uint_as_float(acc), NPU_BF16, kHalfAway, kFtzIn);
  } else if (f.kind == kQ16) {
    const int64_t qo = (h * M + row) * d, ko = (h * N + key) * d;
    Sums16 t;
    for (int64_t j = 0; j < d; ++j)
      t.add(load_int(f.q, f.t16, qo + j), load_int(f.k, f.t16, ko + j));
    s = round_float(t.result(qc), NPU_BF16, kHalfEven, kFtzIn);
  } else if (f.kind == kU8) {
    const uint8_t* q = static_cast<const uint8_t*>(f.q) + (h * M + row) * d;
    const uint8_t* k = static_cast<const uint8_t*>(f.k) + (h * N + key) * d;
    int64_t g = 0, w = 0;
    for (int64_t j = 0; j < d; ++j) {
      g += int64_t(q[j]) * k[j];
      w += k[j];
    }
    const float a = float(act_sum_u8(q, d, f.c.m_granularity));
    float acc;
    if (f.c.br == 1) {
      // scalar bias c0 + c1 * sum(Q) (plain fp32), then + c3 * sum(K), + c2 * QK
      const float bias = __fadd_rn(qc[0], __fmul_rn(qc[1], a));
      acc = mac_exact(g, qc[2], fma_ftz(float(w), qc[3], bias, kFtz), kFtz);
    } else {
      acc = mac_exact(g, qc[2], fma_ftz(float(w), qc[3], qc[0], kFtz), kFtz);
      acc = fma_ftz(qc[1], a, acc, kFtz);
    }
    s = round_float(acc, NPU_BF16, kHalfEven, kFtzIn);
  } else {
    // mx MMAC chain over d in steps of 64
    const int64_t rb = d / 8 * 9;
    const uint8_t* q = static_cast<const uint8_t*>(f.q) + (h * M + row) * rb;
    const uint8_t* k = static_cast<const uint8_t*>(f.k) + (h * N + key) * rb;
    uint32_t acc = 0;
    Bfp16Fields x[8], z[8];
    for (int64_t j0 = 0; j0 < d / 8; j0 += 8) {
      for (int b = 0; b < 8; ++b) {
        bfp16_fields(q + (j0 + b) * 9, &x[b]);
        bfp16_fields(k + (j0 + b) * 9, &z[b]);
      }
      acc = mmac_bfp16(acc, x, z);
    }
    s = round_float(__uint_as_float(acc), NPU_BF16, kHalfAway, kFtzIn);
  }

  // masks
  const Rnd ms = mode_s(f);
  const int nv = valid_keys(f, blk);
  if (f.c.mask_type == 1) {
    const int64_t qpos = q_block_pos(f, sc) + (f.c.q_shards > 0 ? row % f.c.br : 0);
    bool ok = key <= qpos && col < nv;
    if (f.c.sliding && f.c.sliding_window_size > 0)
      ok = ok && key >= qpos - f.c.sliding_window_size + 1;
    s = round_float(fma_ftz(s, 1.0f, ok ? 0.0f : -INFINITY, kFtz), NPU_BF16, ms, kFtzIn);
  } else if (nv < bc) {
    s = col < nv ? __fadd_rn(s, 0.0f) : -INFINITY;
  }
  if (f.c.mask_type == 2 && f.mask) {
    const uint16_t mc = f.mask[(h % f.c.mask_heads) * N + key];
    const float mv = f.kind == kBf16
                         ? bf16_to_f32(mc)
                         : round_float(mac_exact(mc, f.c.mask_c0, f.c.mask_c1, kFtz), NPU_BF16,
                                       ms, kFtzIn);
    s = round_float(fma_ftz(s, 1.0f, mv, kFtz), NPU_BF16, ms, kFtzIn);
  }
  f.s[i] = s;
}

// Row max, P, row sum and l of KV block blk: thread (h, row).
__global__ void fa_softmax_kernel(int64_t n, Fa f, int64_t blk) {
  NPU_TID;
  const int64_t bc = f.c.bc, M = f.c.q_len;
  const int64_t row = i % M;
  if (!visits(f, row / f.c.br, blk)) return;
  float* s = f.s + i * bc;
  const bool guard = f.c.sliding != 0;
  const bool qdq16 = f.kind == kQ16 && f.c.qdq == 2;
  const float alpha = bf16_to_f32(uint16_t(f.c.alpha_bf16));
  const float c =
      f.kind == kMx ? round_float(__fmul_rn(alpha, kLog2e), NPU_BF16, kHalfEven, kFtzIn) : kLog2e;
  float cur = s[0];
  for (int64_t j = 1; j < bc; ++j) cur = vmax(cur, s[j]);
  const float m_pre = f.m[i];
  float m_new, t;
  if (f.c.br == 1) {
    m_new = vmax(cur, m_pre);
    // scalar delta and scaling: double, then fp32
    float delta = __double2float_rn(__dsub_rn(double(m_pre), double(m_new)));
    if (guard && m_new == -INFINITY) delta = 0.0f;
    t = __double2float_rn(__dmul_rn(double(delta), double(c)));
  } else {
    m_new = vmax(cur, m_pre);
    float delta = fma_ftz(m_pre, 1.0f, -m_new, kFtz);
    if (guard && m_new == -INFINITY) delta = 0.0f;
    t = fma_ftz(delta, c, 0.0f, kFtz);
  }
  const float emd = hw_exp2_bf16(fma_ftz(t, 1.0f, kExp2Bias, kFtz));

  const float mm = (guard && f.kind != kMx && !qdq16 && m_new < -3.0e38f) ? 0.0f : m_new;
  const float ml = fma_ftz(c, mm, kMaxBias, kFtz);
  float acc[32];
  for (int64_t t0 = 0; t0 < bc; t0 += 64) {
    float p[64];
    for (int j = 0; j < 64; ++j) p[j] = hw_exp2_bf16(fma_ftz(s[t0 + j], c, -ml, kFtz));
    for (int j = 0; j < 32; ++j) {
      if (qdq16) {
        acc[j] = fma_ftz(fma_ftz(t0 == 0 ? 0.0f : acc[j], 1.0f, p[j], kFtz), 1.0f, p[32 + j], kFtz);
      } else if (f.c.br == 1 && f.kind != kMx) {
        const float e = fma_ftz(p[j], 1.0f, p[32 + j], kFtz);
        acc[j] = t0 == 0 ? e : fma_ftz(e, 1.0f, acc[j], kFtz);
      } else {
        acc[j] = t0 == 0 ? p[j] : fma_ftz(p[j], 1.0f, acc[j], kFtz);
        acc[j] = fma_ftz(acc[j], 1.0f, p[32 + j], kFtz);
      }
    }
    // P as the PV operand
    if (f.kind == kMx) {
      for (int b = 0; b < 8; ++b) {
        uint8_t blkb[9];
        encode_bfp16(p + 8 * b, kHalfEven, blkb);
        f.pe[(i * bc + t0) / 8 + b] = blkb[0];
        for (int u = 0; u < 8; ++u) s[t0 + 8 * b + u] = float(int8_t(blkb[1 + u]));
      }
    } else if (f.c.qdq == 2) {
      const npu_dtype pt = f.kind == kU8 ? NPU_U8 : NPU_U16;
      const float c0 = float(f.c.p_c0 & 0xffff);
      for (int j = 0; j < 64; ++j)
        s[t0 + j] = float(float_to_int(fma_ftz(p[j], f.c.p_c1, c0, kFtz), pt, 0, kHalfEven));
    } else {
      for (int j = 0; j < 64; ++j) s[t0 + j] = p[j];
    }
  }
  if (qdq16)
    for (int j = 0; j < 32; ++j) acc[j] = round_float(acc[j], NPU_BF16, kHalfEven, kFtzIn);
  const float rs = round_float(vaddr_f32_reduce(acc, 32), NPU_BF16, kHalfEven, kFtzIn);
  f.l[i] = round_float(fma_ftz(emd, f.l[i], rs, kFtz), NPU_BF16, f.c.br < 8 ? kHalfEven : rnd(f),
                       kFtzIn);
  f.m[i] = m_new;
  f.emd[i] = emd;
}

// PV of KV block blk and the O update: thread (h, row, column).
__global__ void fa_pv_kernel(int64_t n, Fa f, int64_t blk) {
  NPU_TID;
  const int64_t dv = f.c.v_head_dim, M = f.c.q_len, N = f.c.kv_len, bc = f.c.bc;
  const int64_t col = i % dv, hr = i / dv, row = hr % M, h = hr / M;
  if (!visits(f, row / f.c.br, blk)) return;
  const float* p = f.s + hr * bc;
  const int64_t k0 = blk * bc;
  const float* c = f.c.pv_c;
  float pv;
  if (f.kind == kMx) {
    // V^T block column `col`: as given, or V decoded and re-encoded along the keys
    const uint8_t* v = static_cast<const uint8_t*>(f.v);
    uint32_t acc = 0;
    Bfp16Fields x[8], z[8];
    for (int64_t j0 = 0; j0 < bc / 8; j0 += 8) {
      for (int b = 0; b < 8; ++b) {
        const int64_t j = j0 + b;  // block of 8 keys within the KV block
        x[b].e = f.pe[(hr * bc) / 8 + j];
        for (int u = 0; u < 8; ++u) x[b].m[u] = int(p[8 * j + u]);
        if (f.c.v_transposed) {
          bfp16_fields(v + (h * dv + col) * (N / 8 * 9) + (k0 / 8 + j) * 9, &z[b]);
        } else {
          float vals[8];
          for (int u = 0; u < 8; ++u)
            vals[u] = decode_bfp16(v + (h * N + k0 + 8 * j + u) * (dv / 8 * 9) + (col / 8) * 9,
                                   col % 8);
          uint8_t eb[9];
          encode_bfp16(vals, rnd(f), eb);
          bfp16_fields(eb, &z[b]);
        }
      }
      acc = mmac_bfp16(acc, x, z);
    }
    pv = __uint_as_float(acc);
  } else if (f.kind == kQ16 && f.c.qdq == 2) {
    Sums16 t;
    for (int64_t j = 0; j < bc; ++j)
      t.add(int64_t(p[j]), load_int(f.v, f.t16, (h * N + k0 + j) * dv + col));
    pv = t.result(c);
  } else if (f.kind == kU8 && f.c.qdq == 2) {
    const uint8_t* v = static_cast<const uint8_t*>(f.v) + (h * N + k0) * dv + col;
    const int64_t chunk = int64_t(f.c.k_tile) * 64;
    pv = 0.0f;
    for (int64_t j0 = 0; j0 < bc; j0 += chunk) {
      const int64_t len = bc - j0 < chunk ? bc - j0 : chunk;
      int64_t g = 0, vs = 0;
      for (int64_t j = j0; j < j0 + len; ++j) {
        g += int64_t(p[j]) * v[j * dv];
        vs += v[j * dv];
      }
      const float c0t = mac_exact(vs, c[3], j0 == 0 ? c[0] : 0.0f, kFtz);
      float acc = mac_exact(g, c[2], c0t, kFtz);
      acc = fma_ftz(c[1], float(act_sum_f(p + j0, len)), acc, kFtz);
      pv = j0 == 0 ? acc : fma_ftz(acc, 1.0f, pv, kFtz);
    }
  } else {
    // V as bf16 (bf16, or dequantized for QDQ mode 1)
    const Rnd v_rnd = (f.kind == kU8 || f.c.shuffle) ? rnd(f) : kHalfAway;
    auto vbits = [&](int64_t j) -> uint16_t {
      const int64_t idx = (h * N + k0 + j) * dv + col;
      if (f.kind == kBf16) return static_cast<const uint16_t*>(f.v)[idx];
      const int64_t x = f.kind == kU8 ? int64_t(static_cast<const uint8_t*>(f.v)[idx])
                                      : load_int(f.v, f.t16, idx);
      return f32_to_bf16(mac_exact(x, c[1], c[0], kFtz), v_rnd, kFtzIn);
    };
    if (f.c.br == 1) {
      // GEMV: four MAC chains over the keys, summed pairwise
      float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int64_t j = 0; j < bc; ++j)
        a[j & 3] = fma_ftz(bf16_to_f32(vbits(j)), p[j], a[j & 3], kFtz);
      pv = fma_ftz(fma_ftz(a[0], 1.0f, a[1], kFtz), 1.0f, fma_ftz(a[2], 1.0f, a[3], kFtz), kFtz);
    } else {
      uint16_t x[32], z[32];
      uint32_t acc = 0;
      for (int64_t j0 = 0; j0 < bc; j0 += 32) {
        for (int u = 0; u < 32; ++u) {
          x[u] = uint16_t(__float_as_uint(p[j0 + u]) >> 16);
          z[u] = vbits(j0 + u);
        }
        acc = mmac_bf16(acc, x, 1, z, 1, 32);
      }
      pv = __uint_as_float(acc);
    }
  }
  f.o[i] = fma_ftz(f.o[i], f.emd[hr], pv, kFtz);
}

// out = O * inv(l), converted: thread (h, row).
__global__ void fa_out_kernel(int64_t n, Fa f) {
  NPU_TID;
  const int64_t dv = f.c.v_head_dim;
  const float inv = hw_inv_f32(f.l[i], f.inv_exc);
  const float* o = f.o + i * dv;
  if (f.kind == kBf16) {
    uint16_t* out = static_cast<uint16_t*>(f.out) + i * dv;
    for (int64_t j = 0; j < dv; ++j)
      out[j] = f32_to_bf16(fma_ftz(o[j], inv, 0.0f, kFtz), rnd(f), kFtzIn);
    return;
  }
  if (f.kind == kMx) {
    uint8_t* out = static_cast<uint8_t*>(f.out) + i * (dv / 8 * 9);
    for (int64_t j0 = 0; j0 < dv; j0 += 8) {
      float vals[8];
      for (int u = 0; u < 8; ++u) vals[u] = fma_ftz(o[j0 + u], inv, 0.0f, kFtz);
      encode_bfp16(vals, rnd(f), out + j0 / 8 * 9);
    }
    return;
  }
  const float c1 = f.c.ofm_c1;
  const float scale = f.c.br == 1 ? __double2float_rn(__dmul_rn(double(c1), double(inv)))
                                  : fma_ftz(c1, inv, 0.0f, kFtz);
  const npu_dtype t = f.kind == kU8 ? NPU_U8 : (f.t16);
  for (int64_t j = 0; j < dv; ++j) {
    const float q = fma_ftz(o[j], scale, float(f.c.ofm_c0), kFtz);
    store_int(f.out, t, i * dv + j, float_to_int(q, t, 0, rnd(f)));
  }
}

bool cfg_ok(const npu_flash_attention_cfg* c, int kind) {
  if (!c || c->heads < 1 || c->br < 1 || c->bc < 64 || c->bc % 64 || c->q_len < c->br ||
      c->q_len % c->br || c->kv_len < c->bc || c->kv_len % c->bc || c->head_dim < 1 ||
      c->v_head_dim < 1 || c->mask_type < 0 || c->mask_type > 2 || c->kv_valid_last < 0 ||
      c->kv_valid_last > c->bc || c->k_tile < 1 || c->mask_heads < 1)
    return false;
  if (c->mask_type == 1 && c->q_shards < 0) return false;
  if (kind == kBf16 && c->qdq != 0) return false;
  if ((kind == kQ16 || kind == kU8) && c->qdq != 1 && c->qdq != 2) return false;
  if (kind == kMx && (c->qdq != 0 || c->head_dim % 64 || c->v_head_dim % 8 || c->mask_type == 2))
    return false;
  return true;
}

int run(void* stream, const npu_flash_attention_cfg* cfg, int kind, bool sgn, const void* q,
        const void* k, const void* v, const uint16_t* mask, void* out) {
  if (!q || !k || !v || !out || !cfg_ok(cfg, kind)) return -1;
  if (cfg->mask_type == 2 && !mask && kind != kMx) return -1;
  Fa f{};
  f.c = *cfg;
  f.kind = kind;
  f.t16 = sgn ? NPU_I16 : NPU_U16;
  f.q = q;
  f.k = k;
  f.v = v;
  f.mask = cfg->mask_type == 2 ? mask : nullptr;
  f.out = out;
  if (int rc = npu_emu_nlf_table(NPU_EMU_NLF_INV_F32, &f.inv_exc)) return rc;
  const int64_t rows = cfg->heads * cfg->q_len, bc = cfg->bc, dv = cfg->v_head_dim;
  const size_t bytes = size_t(rows) * (size_t(bc) * 4 + size_t(bc) / 8 + 12 + size_t(dv) * 4);
  hipStream_t st = static_cast<hipStream_t>(stream);
  char* work = nullptr;
  if (int rc = int(hipMallocAsync(reinterpret_cast<void**>(&work), bytes, st))) return rc;
  f.s = reinterpret_cast<float*>(work);
  f.o = f.s + rows * bc;
  f.m = f.o + rows * dv;
  f.l = f.m + rows;
  f.emd = f.l + rows;
  f.pe = reinterpret_cast<uint8_t*>(f.emd + rows);
  int rc = int(hipMemsetAsync(f.o, 0, size_t(rows * dv) * 4, st));
  if (!rc) rc = launch(stream, rows, fa_init_kernel, f);
  for (int64_t blk = 0; !rc && blk < cfg->kv_len / bc; ++blk) {
    rc = launch(stream, rows * bc, fa_scores_kernel, f, blk);
    if (!rc) rc = launch(stream, rows, fa_softmax_kernel, f, blk);
    if (!rc) rc = launch(stream, rows * dv, fa_pv_kernel, f, blk);
  }
  if (!rc) rc = launch(stream, rows, fa_out_kernel, f);
  const int rc2 = int(hipFreeAsync(work, st));
  return rc ? rc : rc2;
}

}  // namespace

extern "C" {

int npu_flash_attention_bf16(void* stream, const npu_flash_attention_cfg* cfg, const uint16_t* q,
                             const uint16_t* k, const uint16_t* v, const uint16_t* mask,
                             uint16_t* out) {
  return run(stream, cfg, kBf16, false, q, k, v, mask, out);
}

int npu_flash_attention_q16(void* stream, const npu_flash_attention_cfg* cfg, npu_dtype type,
                            const void* q, const void* k, const void* v, const uint16_t* mask,
                            void* out) {
  if (type != NPU_U16 && type != NPU_I16) return -1;
  return run(stream, cfg, kQ16, type == NPU_I16, q, k, v, mask, out);
}

int npu_flash_attention_u8(void* stream, const npu_flash_attention_cfg* cfg, const uint8_t* q,
                           const uint8_t* k, const uint8_t* v, const uint16_t* mask,
                           uint8_t* out) {
  return run(stream, cfg, kU8, false, q, k, v, mask, out);
}

int npu_flash_attention_mx(void* stream, const npu_flash_attention_cfg* cfg, const uint8_t* q,
                           const uint8_t* k, const uint8_t* v, uint8_t* out) {
  return run(stream, cfg, kMx, false, q, k, v, nullptr, out);
}

}  // extern "C"
