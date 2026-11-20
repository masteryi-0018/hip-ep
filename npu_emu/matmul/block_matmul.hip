/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * bfp16 / mx6 / fp8 matrix multiply and convolution. One thread per output
 * element (per output block for block-format outputs).
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "matmul/block_matmul.h"
#include "common/act.h"
#include "common/float_cvt.h"
#include "common/launch.h"
#include "common/minmax.h"
#include "common/mmac.h"
#include "common/nlf.h"
#include "common/mac.h"

using namespace npu_emu;

namespace {

constexpr int kBfpBytes = 9, kMx6Bytes = 12;

// 5 SiLU, 6 GELU, 7 Tanh on a bf16 value; the result is stored as bf16 half away.
__device__ inline float core_act_bf16(float v, int act) {
  if (act == 5) return round_float(hw_silu_f32(v), NPU_BF16, kHalfAway, kFtzIn);
  if (act == 6) return round_float(hw_gelu_f32(v), NPU_BF16, kHalfAway, kFtzIn);
  if (act == 7) return sigtanh_tanh_bf16(v);
  return v;
}

__device__ inline float activate_block(float v, int narrow, int act, float alpha) {
  if (!narrow) return v;
  v = round_float(v, NPU_BF16, kHalfAway, kFtzIn);
  if (act == 1) return vmax(v, 0.0f);
  if (act == 2) return vclamp(v, 0.0f, 6.0f);
  if (act == 3) return vmax(v, mul16_f32(v, alpha, NPU_BF16));
  return core_act_bf16(v, act);
}

// Output values per store: a packed block (bfp16 8, mx6 16), else 1.
__host__ __device__ inline int out_group(npu_dtype t) {
  return t == NPU_BFP16 ? 8 : (t == NPU_MX6 ? 16 : 1);
}

// The uint8 output cast (NPU_U8): u8 = sat(rne(fp32(v * scale_recip + zp))).
struct OutQ {
  float scale_recip, zp;
};

// Store g consecutive output values: bf16 / f16 / fp8 / uint8 elements, or one packed bfp16
// (g = 8) / mx6 (g = 16) block.
__device__ inline void store(void* out, int64_t first, const float* v, int g, npu_dtype t, Rnd rnd,
                             OutQ q = {}) {
  uint8_t* o = static_cast<uint8_t*>(out);
  if (t == NPU_BFP16) return encode_bfp16(v, rnd, o + first / g * kBfpBytes);
  if (t == NPU_MX6) return encode_mx6(v, rnd, o + first / g * kMx6Bytes);
  for (int j = 0; j < g; ++j) {
    if (t == NPU_U8)
      o[first + j] = uint8_t(float_to_int(fmaf(v[j], q.scale_recip, q.zp), NPU_U8, 0, kHalfEven));
    else
      store_float(out, t, first + j, v[j], rnd, t == NPU_FP8 ? 0 : kFtzIn);
  }
}

__device__ inline float f32(uint32_t b) { return __uint_as_float(b); }
__device__ inline uint32_t bits(float v) { return __float_as_uint(v); }

struct Gemm {
  const uint8_t* a;
  const uint8_t* bt;
  const float* bias;
  int64_t m, k, n;
};

__device__ uint32_t gemm_bfp16_acc(const Gemm& p, int64_t row, int64_t col, uint32_t acc,
                                   int64_t k0, int64_t k1) {
  const int64_t kb = p.k / 8;
  Bfp16Fields x[8], z[8];
  for (int64_t c = k0 / 8; c < k1 / 8; c += 8) {
    for (int b = 0; b < 8; ++b) {
      bfp16_fields(p.a + (row * kb + c + b) * kBfpBytes, &x[b]);
      bfp16_fields(p.bt + (col * kb + c + b) * kBfpBytes, &z[b]);
    }
    acc = mmac_bfp16(acc, x, z);
  }
  return acc;
}

__device__ uint32_t gemm_mx6_acc(const Gemm& p, int64_t row, int64_t col, uint32_t acc) {
  const int64_t kb = p.k / 16;
  Mx6Fields x[8], z[8];
  for (int64_t c = 0; c < kb; c += 8) {
    for (int b = 0; b < 8; ++b) {
      mx6_fields(p.a + (row * kb + c + b) * kMx6Bytes, &x[b]);
      mx6_fields(p.bt + (col * kb + c + b) * kMx6Bytes, &z[b]);
    }
    acc = mmac_mx6(acc, x, z);
  }
  return acc;
}

// Thread i computes output group i (g values along the output row).
__global__ void gemm_block_kernel(int64_t groups, Gemm p, void* out, int g, int mx6, int narrow,
                                  int act, float alpha, int transpose, npu_dtype t, OutQ q) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= groups) return;
  const int64_t cols = transpose ? p.m : p.n;
  const int64_t first = i * g, r = first / cols, c0 = first % cols;
  float v[16];
  for (int j = 0; j < g; ++j) {
    const int64_t row = transpose ? c0 + j : r, col = transpose ? r : c0 + j;
    const uint32_t acc0 = p.bias ? bits(p.bias[col]) : 0u;
    const uint32_t acc = mx6 ? gemm_mx6_acc(p, row, col, acc0) : gemm_bfp16_acc(p, row, col, acc0, 0, p.k);
    v[j] = activate_block(f32(acc), narrow, act, alpha);
  }
  store(out, first, v, g, t, kHalfAway, q);
}

// act(bf16(gate GEMM)) * bf16(up GEMM); the product is exact in fp32.
__global__ void gemm_gated_kernel(int64_t groups, Gemm gate, Gemm up, void* out, int g, int act,
                                  npu_dtype t) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= groups) return;
  const int64_t first = i * g, row = first / gate.n, c0 = first % gate.n;
  float v[16];
  for (int j = 0; j < g; ++j) {
    const int64_t col = c0 + j;
    const uint32_t ag = gemm_bfp16_acc(gate, row, col, gate.bias ? bits(gate.bias[col]) : 0u, 0,
                                       gate.k);
    const uint32_t au = gemm_bfp16_acc(up, row, col, up.bias ? bits(up.bias[col]) : 0u, 0, up.k);
    v[j] = mul_q(core_act_bf16(round_float(f32(ag), NPU_BF16, kHalfAway, kFtzIn), act),
                    round_float(f32(au), NPU_BF16, kHalfAway, kFtzIn));
  }
  store(out, first, v, g, t, kHalfAway);
}

struct Eltwise {
  const void* ifm2;
  int64_t m2, sv_m, sv_n, sv_k;
  int bfp, is_mul;
};

// acc + ifm2[row][col] as one identity block MMAC (ifm2 rows of n values, n % 64 == 0).
__device__ uint32_t e_block(const Eltwise& e, int64_t n, int64_t row, int64_t col, uint32_t acc) {
  const uint8_t* x2 = static_cast<const uint8_t*>(e.ifm2);
  const int64_t b = row * (n / 8) + col / 8, g0 = b & ~int64_t(7);
  const float a = decode_bfp16_identity(x2, g0, g0 + 8, b, col % 8);
  return bits(mmac_add_bfp16(x2, g0, g0 + 8, b, col % 8, a, f32(acc)));
}

__global__ void gemm_eltwise_kernel(int64_t groups, Gemm p, Eltwise e, void* out, int g,
                                    npu_dtype t) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= groups) return;
  const int64_t first = i * g, row = first / p.n, c0 = first % p.n;
  const int64_t r2 = e.m2 == 1 ? 0 : row;
  float v[8];
  for (int j = 0; j < g; ++j) {
    const int64_t col = c0 + j;
    const uint32_t bias = p.bias ? bits(p.bias[col]) : 0u;
    uint32_t acc;
    if (!e.bfp) {
      const float x2 = bf16_to_f32(static_cast<const uint16_t*>(e.ifm2)[r2 * p.n + col]);
      if (e.is_mul) {
        acc = gemm_bfp16_acc(p, row, col, bias, 0, p.k);
        v[j] = mul_q(round_float(f32(acc), NPU_BF16, kHalfAway, kFtzIn), x2);
        continue;
      }
      // seed: fp32(ifm2 + bias)
      acc = gemm_bfp16_acc(p, row, col, p.bias ? bits(add_q(x2, p.bias[col])) : bits(x2), 0, p.k);
    } else if ((row < e.sv_m && col < e.sv_n) || e.sv_k == 64) {
      acc = gemm_bfp16_acc(p, row, col, e_block(e, p.n, r2, col, bias), 0, p.k);
    } else {
      acc = gemm_bfp16_acc(p, row, col, bias, 0, 64);
      acc = gemm_bfp16_acc(p, row, col, e_block(e, p.n, r2, col, acc), 64, p.k);
    }
    v[j] = f32(acc);
  }
  store(out, first, v, g, t, kHalfAway);
}

struct Conv {
  const uint8_t* ifm;
  const uint8_t* wts;
  int64_t h, w, ci, kh, kw, co, ho, wo, kh_sv, ci_sv, co_sv;
  int sh, sw;
  const uint16_t* ifm16 = nullptr;  // fp8 convolution with an f16 activation (narrowed on load)
  uint16_t ifm_r16 = 0;             // its f16 reciprocal scale
};

// The in-place f16 -> E4M3 narrowing of an f16 activation: the 16-bit f16 product with the
// f16 reciprocal scale, clamped to +-448 in f16, then E4M3 half away (subnormals kept).
__device__ inline uint8_t narrow_f16_fp8(uint16_t x, uint16_t r) {
  const uint16_t p = min16(max16(0xdf00, mul16(x, r, NPU_F16), NPU_F16), 0x5f00, NPU_F16);
  return f32_to_fp8(f16_to_f32(p), kHalfAway);
}

// Folding of output column ox: its lane column, the first input column of its
// folded window and the window geometry.
struct Fold {
  int64_t c, x0, per, ng, cb_n, ktile;
};

__device__ inline Fold fold(const Conv& p, int64_t ox, int ktile) {
  const int64_t fwo = 64 / p.co_sv > 0 ? 64 / p.co_sv : 1;
  const int64_t fkw = p.kw + (fwo - 1) * p.sw;
  const int64_t per = p.ci_sv < ktile ? ktile / p.ci_sv : 1;
  const int64_t fkw_al = (fkw + per - 1) / per * per;
  return {ox % fwo, (ox / fwo) * fwo * p.sw, per, fkw_al / per,
          p.ci_sv >= ktile ? p.ci_sv / ktile : 1, ktile};
}

// Lane l of MMAC (channel group c0, channel block cb, group g): the input pixel
// column and channel, and the kernel column (valid in [0, kw)).
__device__ inline void lane(const Conv& p, const Fold& f, int64_t c0, int64_t cb, int64_t g, int l,
                            int64_t* x, int64_t* ch, int64_t* kx) {
  const int64_t fkx = p.ci_sv >= f.ktile ? g : g * f.per + l / p.ci_sv;
  *ch = p.ci_sv >= f.ktile ? c0 + cb * f.ktile + l : c0 + l % p.ci_sv;
  *x = f.x0 + fkx;
  *kx = fkx - f.c * p.sw;
}

__device__ uint32_t conv_bfp16_acc(const Conv& p, int64_t oy, int64_t ox, int64_t oc, uint32_t acc) {
  const Fold f = fold(p, ox, 64);
  const int64_t cib = p.ci / 8;
  Bfp16Fields x[8], z[8];
  for (int64_t kh0 = 0; kh0 < p.kh; kh0 += p.kh_sv)
    for (int64_t c0 = 0; c0 < p.ci; c0 += p.ci_sv)
      for (int64_t cb = 0; cb < f.cb_n; ++cb)
        for (int64_t y = kh0; y < kh0 + p.kh_sv; ++y)
          for (int64_t g = 0; g < f.ng; ++g) {
            for (int b = 0; b < 8; ++b) {
              int64_t xx, ch, kx;
              lane(p, f, c0, cb, g, b * 8, &xx, &ch, &kx);
              const int64_t iy = oy * p.sh + y;
              x[b] = Bfp16Fields{0, {}};
              if (xx < p.w) bfp16_fields(p.ifm + ((iy * p.w + xx) * cib + ch / 8) * kBfpBytes, &x[b]);
              // A padding weight block has exponent 1.
              z[b] = Bfp16Fields{1, {}};
              if (kx >= 0 && kx < p.kw)
                bfp16_fields(p.wts + (((y * p.kw + kx) * p.co + oc) * cib + ch / 8) * kBfpBytes, &z[b]);
            }
            acc = mmac_bfp16(acc, x, z);
          }
  return acc;
}

__device__ uint32_t conv_mx6_acc(const Conv& p, int64_t oy, int64_t ox, int64_t oc, uint32_t acc) {
  const Fold f = fold(p, ox, 128);
  const int64_t cib = p.ci / 16;
  Mx6Fields x[8], z[8];
  for (int64_t kh0 = 0; kh0 < p.kh; kh0 += p.kh_sv)
    for (int64_t c0 = 0; c0 < p.ci; c0 += p.ci_sv)
      for (int64_t cb = 0; cb < f.cb_n; ++cb)
        for (int64_t y = kh0; y < kh0 + p.kh_sv; ++y)
          for (int64_t g = 0; g < f.ng; ++g) {
            for (int b = 0; b < 8; ++b) {
              int64_t xx, ch, kx;
              lane(p, f, c0, cb, g, b * 16, &xx, &ch, &kx);
              const int64_t iy = oy * p.sh + y;
              x[b] = Mx6Fields{0, {}, {}};
              if (xx < p.w) mx6_fields(p.ifm + ((iy * p.w + xx) * cib + ch / 16) * kMx6Bytes, &x[b]);
              z[b] = Mx6Fields{1, {}, {}};
              if (kx >= 0 && kx < p.kw)
                mx6_fields(p.wts + (((y * p.kw + kx) * p.co + oc) * cib + ch / 16) * kMx6Bytes, &z[b]);
            }
            acc = mmac_mx6(acc, x, z);
          }
  return acc;
}

__device__ uint32_t conv_fp8_acc(const Conv& p, int64_t oy, int64_t ox, int64_t oc) {
  const Fold f = fold(p, ox, 64);
  uint32_t acc = 0;
  uint8_t x[64], z[64];
  for (int64_t kh0 = 0; kh0 < p.kh; kh0 += p.kh_sv)
    for (int64_t c0 = 0; c0 < p.ci; c0 += p.ci_sv)
      for (int64_t cb = 0; cb < f.cb_n; ++cb)
        for (int64_t y = kh0; y < kh0 + p.kh_sv; ++y)
          for (int64_t g = 0; g < f.ng; ++g) {
            for (int l = 0; l < 64; ++l) {
              int64_t xx, ch, kx;
              lane(p, f, c0, cb, g, l, &xx, &ch, &kx);
              const int64_t iy = oy * p.sh + y;
              const int64_t e = (iy * p.w + xx) * p.ci + ch;
              x[l] = xx >= p.w ? 0 : p.ifm16 ? narrow_f16_fp8(p.ifm16[e], p.ifm_r16) : p.ifm[e];
              z[l] = (kx >= 0 && kx < p.kw) ? p.wts[((y * p.kw + kx) * p.ci + ch) * p.co + oc] : 0;
            }
            acc = mmac_fp8(acc, x, 1, z, 1, 64);
          }
  return acc;
}

struct Residual {
  const void* ifm2;
  npu_dtype type;
  int64_t h2, w2, c2;
  float coeff;
};

__device__ inline float residual(const Residual& r, int64_t oy, int64_t ox, int64_t oc) {
  const int64_t j = ((r.h2 == 1 ? 0 : oy) * r.w2 + (r.w2 == 1 ? 0 : ox)) * r.c2 + (r.c2 == 1 ? 0 : oc);
  if (r.type == NPU_FP8) return fp8_to_f32(static_cast<const uint8_t*>(r.ifm2)[j]);
  const uint16_t h = static_cast<const uint16_t*>(r.ifm2)[j];
  return r.type == NPU_F16 ? f16_to_f32(h) : bf16_to_f32(h);
}

__global__ void conv_block_kernel(int64_t groups, Conv p, const float* bias, Residual res, void* out,
                                  int g, int mx6, int narrow, int act, float alpha, npu_dtype t) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= groups) return;
  const int64_t first = i * g, pix = first / p.co, oc0 = first % p.co;
  const int64_t oy = pix / p.wo, ox = pix % p.wo;
  float v[16];
  for (int j = 0; j < g; ++j) {
    const int64_t oc = oc0 + j;
    uint32_t acc0 = bias ? bits(bias[oc]) : 0u;
    if (res.ifm2) {
      const float r = residual(res, oy, ox, oc);
      acc0 = bias ? bits(add_q(r, bias[oc])) : bits(r);
    }
    const uint32_t acc = mx6 ? conv_mx6_acc(p, oy, ox, oc, acc0) : conv_bfp16_acc(p, oy, ox, oc, acc0);
    v[j] = res.ifm2 ? f32(acc) : activate_block(f32(acc), narrow, act, alpha);
  }
  store(out, first, v, g, t, res.ifm2 ? kHalfEven : kHalfAway);
}

__device__ inline float act_fp8(float v, int act, float a0, float a1, const float* slope, int64_t oc) {
  if (act == 1) return vmax(v, 0.0f);
  if (act == 2) return vclamp(v, 0.0f, a0);
  if (act == 3) return fma_q(vmin(v, 0.0f), a0, vmax(v, 0.0f));
  if (act == 8) return vclamp(v, a0, a1);
  if (act == 9) return v < 0.0f ? mul_q(v, slope[oc]) : v;
  return v;
}

__global__ void gemm_fp8_kernel(int64_t total, const uint8_t* a, const uint8_t* b, const float* c0,
                                const float* c2, void* out, int64_t k, int64_t n, npu_dtype t) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t row = i / n, col = i % n;
  uint32_t acc = 0;
  uint8_t x[64], z[64];
  for (int64_t k0 = 0; k0 < k; k0 += 64) {
    for (int l = 0; l < 64; ++l) {
      x[l] = k0 + l < k ? a[row * k + k0 + l] : 0;
      z[l] = k0 + l < k ? b[(k0 + l) * n + col] : 0;
    }
    acc = mmac_fp8(acc, x, 1, z, 1, 64);
  }
  const float v = fma_q(f32(acc), c2[col], c0[col]);
  store(out, i, &v, 1, t, kHalfAway);
}

__global__ void conv_fp8_kernel(int64_t total, Conv p, const float* c0, const float* c2,
                                Residual res, void* out, int act, float a0, float a1,
                                const float* slope, npu_dtype t) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t oc = i % p.co, ox = (i / p.co) % p.wo, oy = i / (p.co * p.wo);
  const float acc = f32(conv_fp8_acc(p, oy, ox, oc));
  float v;
  if (res.ifm2)
    v = fma_q(acc, c2[oc], fma_q(residual(res, oy, ox, oc), res.coeff, c0[oc]));
  else
    v = act_fp8(fma_q(acc, c2[oc], c0[oc]), act, a0, a1, slope, oc);
  store(out, i, &v, 1, t, kHalfAway);
}

__global__ void dwc_fp8_kernel(int64_t total, const uint8_t* ifm, const uint8_t* wts, const float* c0,
                               const float* c2, void* out, int64_t w, int64_t c, int64_t kh,
                               int64_t kw, int64_t wo, int sh, int sw, int act, float a0, npu_dtype t) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t oc = i % c, ox = (i / c) % wo, oy = i / (c * wo);
  float acc = 0.0f;
  for (int64_t y = 0; y < kh; ++y)
    for (int64_t x = 0; x < kw; ++x)
      acc = add_q(acc, mul_q(fp8_to_f32(ifm[((oy * sh + y) * w + ox * sw + x) * c + oc]),
                                   fp8_to_f32(wts[(y * kw + x) * c + oc])));
  float v = fma_q(acc, c2[oc], c0[oc]);
  if (act == 1) v = vmax(v, 0.0f);
  else if (act == 2) v = vclamp(v, 0.0f, a0);
  else if (act == 3) v = vmax(v, mul_q(v, a0));
  store(out, i, &v, 1, t, kHalfAway);
}

// Output types: bf16 or the entry's packed block format; the fp8 entries bf16 / f16 / fp8.
bool block_ofm_ok(npu_dtype t, npu_dtype packed) { return t == NPU_BF16 || t == packed; }
bool fp8_ofm_ok(npu_dtype t) { return t == NPU_BF16 || t == NPU_F16 || t == NPU_FP8; }

Conv make_conv(const uint8_t* ifm, const uint8_t* wts, int64_t h, int64_t w, int64_t ci, int64_t kh,
               int64_t kw, int64_t co, int sh, int sw, int64_t kh_sv, int64_t ci_sv, int64_t co_sv) {
  return {ifm, wts, h, w, ci, kh, kw, co, (h - kh) / sh + 1, (w - kw) / sw + 1, kh_sv, ci_sv, co_sv,
          sh, sw};
}

bool conv_ok(int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co, int sh, int sw,
             int64_t kh_sv, int64_t ci_sv, int64_t co_sv, int64_t blk) {
  return h >= kh && w >= kw && kh >= 1 && kw >= 1 && sh >= 1 && sw >= 1 && kh_sv >= 1 &&
         kh % kh_sv == 0 && ci_sv >= blk && ci_sv % blk == 0 && ci % ci_sv == 0 && co_sv >= 1 &&
         co_sv <= 64 && co % co_sv == 0;
}

int gemm_eltwise(void* stream, const uint8_t* a, const uint8_t* bt, const float* bias,
                 const void* ifm2, npu_dtype ifm2_type, int64_t m2, void* out, int64_t m,
                 int64_t k, int64_t n, int64_t sv_m, int64_t sv_n, int64_t sv_k, int is_mul,
                 npu_dtype ofm_type) {
  const int bfp = ifm2_type == NPU_BFP16;
  if (!a || !bt || !ifm2 || !out || m < 1 || n < 1 || k < 64 || k % 64 || (m2 != 1 && m2 != m) ||
      (ifm2_type != NPU_BF16 && !bfp) || (bfp && (is_mul || n % 64)) ||
      !block_ofm_ok(ofm_type, NPU_BFP16) || n % out_group(ofm_type))
    return -1;
  const int g = out_group(ofm_type);
  const Gemm p{a, bt, bias, m, k, n};
  const Eltwise e{ifm2, m2, sv_m, sv_n, sv_k, bfp, is_mul};
  return launch(stream, m * n / g, gemm_eltwise_kernel, p, e, out, g, ofm_type);
}

int conv_bfp16(void* stream, const uint8_t* ifm, const uint8_t* wts, const float* bias,
               const uint16_t* ifm2, int64_t h2, int64_t w2, int64_t c2, void* out, int64_t h,
               int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
               int stride_w, int64_t kh_sv, int64_t ci_sv, int64_t co_sv, int narrow, int act,
               float alpha, npu_dtype ofm_type) {
  if (!ifm || !wts || !out || !conv_ok(h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv, ci_sv, co_sv, 8) ||
      act < 0 || act > 3 || (act && !narrow) || (ifm2 && (narrow || act)) ||
      !block_ofm_ok(ofm_type, NPU_BFP16) || co % out_group(ofm_type))
    return -1;
  const Conv p = make_conv(ifm, wts, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv, ci_sv, co_sv);
  if (ifm2 && ((h2 != 1 && h2 != p.ho) || (w2 != 1 && w2 != p.wo) || (c2 != 1 && c2 != co))) return -1;
  const Residual r{ifm2, NPU_BF16, h2, w2, c2, 1.0f};
  const int g = out_group(ofm_type);
  return launch(stream, p.ho * p.wo * co / g, conv_block_kernel, p, bias, r, out, g, 0, narrow, act,
                scalar_cast(alpha, NPU_BF16, kHalfEven), ofm_type);
}

int conv_fp8(void* stream, const uint8_t* ifm, const uint16_t* ifm16, float ifm_scale_recip,
             const uint8_t* wts, const float* c0,
             const float* c2, const void* ifm2, npu_dtype ifm2_type, int64_t h2, int64_t w2,
             int64_t c2_dim, float ifm2_coeff, void* out, int64_t h, int64_t w, int64_t ci,
             int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w, int64_t kh_sv,
             int64_t ci_sv, int64_t co_sv, int act, float a0, float a1, const float* slope,
             npu_dtype ofm_type) {
  if ((!ifm && !ifm16) || !wts || !c0 || !c2 || !out ||
      !conv_ok(h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv, ci_sv, co_sv, 8) ||
      !(act == 0 || act == 1 || act == 2 || act == 3 || act == 8 || act == 9) ||
      (act == 9 && !slope) || (ifm2 && act) || !fp8_ofm_ok(ofm_type) ||
      (ifm2 && ifm2_type != NPU_FP8 && ifm2_type != NPU_BF16 && ifm2_type != NPU_F16))
    return -1;
  Conv p = make_conv(ifm, wts, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv, ci_sv, co_sv);
  p.ifm16 = ifm16;
  p.ifm_r16 = f32_to_f16(ifm_scale_recip, kHalfEven);
  if (ifm2 && ((h2 != 1 && h2 != p.ho) || (w2 != 1 && w2 != p.wo) || (c2_dim != 1 && c2_dim != co)))
    return -1;
  const Residual r{ifm2, ifm2_type, h2, w2, c2_dim, ifm2_coeff};
  return launch(stream, p.ho * p.wo * co, conv_fp8_kernel, p, c0, c2, r, out, act, a0, a1, slope,
                ofm_type);
}

}  // namespace

extern "C" {

int npu_gemm_bfp16(void* stream, const uint8_t* a, const uint8_t* bt, const float* bias, void* out,
                   int64_t m, int64_t k, int64_t n, int narrow, int act, float alpha, int transpose,
                   npu_dtype ofm_type, float ofm_scale_recip, float ofm_zp) {
  const int64_t cols = transpose ? m : n;
  const bool u8 = ofm_type == NPU_U8;
  if (!a || !bt || !out || m < 1 || n < 1 || k < 64 || k % 64 || act < 0 || act > 7 || act == 4 ||
      (act && !narrow) || (u8 && (!narrow || act)) || !(block_ofm_ok(ofm_type, NPU_BFP16) || u8) ||
      cols % out_group(ofm_type))
    return -1;
  const int g = out_group(ofm_type);
  const Gemm p{a, bt, bias, m, k, n};
  return launch(stream, m * n / g, gemm_block_kernel, p, out, g, 0, narrow, act,
                scalar_cast(alpha, NPU_BF16, kHalfEven), transpose, ofm_type,
                OutQ{ofm_scale_recip, ofm_zp});
}

int npu_gemm_gelu_mul_bfp16(void* stream, const uint8_t* a, const uint8_t* bt_gate,
                            const uint8_t* bt_up, const float* bias_gate, const float* bias_up,
                            void* out, int64_t m, int64_t k, int64_t n, int act,
                            npu_dtype ofm_type) {
  if (!a || !bt_gate || !bt_up || !out || m < 1 || n < 1 || k < 64 || k % 64 ||
      !(act == 0 || (act >= 5 && act <= 7)) || !block_ofm_ok(ofm_type, NPU_BFP16) ||
      n % out_group(ofm_type))
    return -1;
  const int g = out_group(ofm_type);
  const Gemm gate{a, bt_gate, bias_gate, m, k, n}, up{a, bt_up, bias_up, m, k, n};
  return launch(stream, m * n / g, gemm_gated_kernel, gate, up, out, g, act, ofm_type);
}

int npu_gemm_add_bfp16(void* stream, const uint8_t* a, const uint8_t* bt, const float* bias,
                       const void* ifm2, npu_dtype ifm2_type, int64_t m2, void* out, int64_t m,
                       int64_t k, int64_t n, int64_t sv_m, int64_t sv_n, int64_t sv_k,
                       npu_dtype ofm_type) {
  return gemm_eltwise(stream, a, bt, bias, ifm2, ifm2_type, m2, out, m, k, n, sv_m, sv_n, sv_k, 0,
                      ofm_type);
}

int npu_gemm_mul_bfp16(void* stream, const uint8_t* a, const uint8_t* bt, const float* bias,
                       const void* ifm2, npu_dtype ifm2_type, int64_t m2, void* out, int64_t m,
                       int64_t k, int64_t n, int64_t sv_m, int64_t sv_n, int64_t sv_k,
                       npu_dtype ofm_type) {
  return gemm_eltwise(stream, a, bt, bias, ifm2, ifm2_type, m2, out, m, k, n, sv_m, sv_n, sv_k, 1,
                      ofm_type);
}

int npu_conv_bfp16(void* stream, const uint8_t* ifm, const uint8_t* wts, const float* bias,
                   void* out, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                   int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv, int64_t co_sv,
                   int narrow, int act, float alpha, npu_dtype ofm_type) {
  return conv_bfp16(stream, ifm, wts, bias, nullptr, 1, 1, 1, out, h, w, ci, kh, kw, co, stride_h,
                    stride_w, kh_sv, ci_sv, co_sv, narrow, act, alpha, ofm_type);
}

int npu_conv_add_bfp16(void* stream, const uint8_t* ifm, const uint8_t* wts, const float* bias,
                       const uint16_t* ifm2, int64_t h2, int64_t w2, int64_t c2, void* out,
                       int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                       int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv, int64_t co_sv,
                       npu_dtype ofm_type) {
  if (!ifm2) return -1;
  return conv_bfp16(stream, ifm, wts, bias, ifm2, h2, w2, c2, out, h, w, ci, kh, kw, co, stride_h,
                    stride_w, kh_sv, ci_sv, co_sv, 0, 0, 0.0f, ofm_type);
}

int npu_gemm_mx6(void* stream, const uint8_t* a, const uint8_t* bt, const float* bias, void* out,
                 int64_t m, int64_t k, int64_t n, int narrow, int act, float alpha,
                 npu_dtype ofm_type) {
  if (!a || !bt || !out || m < 1 || n < 1 || k < 128 || k % 128 || act < 0 || act > 3 ||
      (act && !narrow) || !block_ofm_ok(ofm_type, NPU_MX6) || n % out_group(ofm_type))
    return -1;
  const int g = out_group(ofm_type);
  const Gemm p{a, bt, bias, m, k, n};
  return launch(stream, m * n / g, gemm_block_kernel, p, out, g, 1, narrow, act,
                scalar_cast(alpha, NPU_BF16, kHalfEven), 0, ofm_type, OutQ{});
}

int npu_conv_mx6(void* stream, const uint8_t* ifm, const uint8_t* wts, const float* bias, void* out,
                 int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
                 int stride_w, int64_t kh_sv, int64_t ci_sv, int64_t co_sv, int narrow, int act,
                 float alpha, npu_dtype ofm_type) {
  if (!ifm || !wts || !out ||
      !conv_ok(h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv, ci_sv, co_sv, 128) || act < 0 ||
      act > 3 || (act && !narrow) || !block_ofm_ok(ofm_type, NPU_MX6) || co % out_group(ofm_type))
    return -1;
  const Conv p = make_conv(ifm, wts, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv, ci_sv, co_sv);
  const Residual r{nullptr, NPU_BF16, 1, 1, 1, 1.0f};
  const int g = out_group(ofm_type);
  return launch(stream, p.ho * p.wo * co / g, conv_block_kernel, p, bias, r, out, g, 1, narrow, act,
                scalar_cast(alpha, NPU_BF16, kHalfEven), ofm_type);
}

int npu_gemm_fp8(void* stream, const uint8_t* a, const uint8_t* b, const float* c0, const float* c2,
                 void* out, int64_t m, int64_t k, int64_t n, npu_dtype ofm_type) {
  if (!a || !b || !c0 || !c2 || !out || m < 1 || k < 1 || n < 1 || !fp8_ofm_ok(ofm_type))
    return -1;
  return launch(stream, m * n, gemm_fp8_kernel, a, b, c0, c2, out, k, n, ofm_type);
}

int npu_conv_fp8(void* stream, const void* ifm, npu_dtype ifm_type, float ifm_scale_recip,
                 const uint8_t* wts, const float* c0, const float* c2, void* out, int64_t h,
                 int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
                 int stride_w, int64_t kh_sv, int64_t ci_sv, int64_t co_sv, int act, float a0,
                 float a1, const float* slope, npu_dtype ofm_type) {
  if (ifm_type != NPU_FP8 && ifm_type != NPU_F16) return -1;
  const bool f16 = ifm_type == NPU_F16;
  return conv_fp8(stream, f16 ? nullptr : static_cast<const uint8_t*>(ifm),
                  f16 ? static_cast<const uint16_t*>(ifm) : nullptr, ifm_scale_recip, wts, c0, c2,
                  nullptr, NPU_FP8, 1, 1, 1, 0.0f, out, h, w, ci, kh, kw, co, stride_h, stride_w,
                  kh_sv, ci_sv, co_sv, act, a0, a1, slope, ofm_type);
}

int npu_conv_add_fp8(void* stream, const uint8_t* ifm, const uint8_t* wts, const float* c0,
                     const float* c2, const void* ifm2, npu_dtype ifm2_type, int64_t h2,
                     int64_t w2, int64_t c2_dim, float ifm2_coeff, void* out, int64_t h, int64_t w,
                     int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                     int64_t kh_sv, int64_t ci_sv, int64_t co_sv, npu_dtype ofm_type) {
  if (!ifm2) return -1;
  return conv_fp8(stream, ifm, nullptr, 0.0f, wts, c0, c2, ifm2, ifm2_type, h2, w2, c2_dim,
                  ifm2_coeff, out, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv, ci_sv, co_sv, 0,
                  0.0f, 0.0f, nullptr, ofm_type);
}

int npu_depthwise_conv_fp8(void* stream, const uint8_t* ifm, const uint8_t* wts, const float* c0,
                           const float* c2, void* out, int64_t h, int64_t w, int64_t c, int64_t kh,
                           int64_t kw, int stride_h, int stride_w, int act, float a0,
                           npu_dtype ofm_type) {
  if (!ifm || !wts || !c0 || !c2 || !out || h < kh || w < kw || c < 1 || stride_h < 1 ||
      stride_w < 1 || act < 0 || act > 3 || !fp8_ofm_ok(ofm_type))
    return -1;
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  return launch(stream, ho * wo * c, dwc_fp8_kernel, ifm, wts, c0, c2, out, w, c, kh, kw, wo,
                stride_h, stride_w, act, a0, ofm_type);
}

}  // extern "C"
