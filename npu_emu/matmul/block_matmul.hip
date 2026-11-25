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
#include "common/bits.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/mmac.h"
#include "common/nlf.h"
#include "common/quant.h"

using namespace npu_emu;

namespace {

constexpr int kBfpBytes = 9, kMx6Bytes = 12;

// 4 Sigmoid, 5 SiLU, 6 GELU, 7 Tanh on a bf16 value; the result is stored as bf16 half away.
__device__ inline float core_act_bf16(float v, int act) {
  if (act == 4)
    return acc_to_bf16(hw_sigtanh_f32(v, false, kSigTanhK0, kSigmoidK1, kSigmoidK2));
  if (act == 5) return acc_to_bf16(hw_silu_f32(v));
  if (act == 6) return acc_to_bf16(hw_gelu_f32(v));
  if (act == 7) return sigtanh_tanh_bf16(v);
  return v;
}

__device__ inline float activate_block(float v, int narrow, int act, float alpha) {
  if (!narrow) return v;
  v = acc_to_bf16(v);
  return act <= 3 ? activate_bf16(v, act, alpha) : core_act_bf16(v, act);
}

// Output values per store: a packed block (bfp16 8, mx6 16), else 1.
__host__ __device__ inline int out_group(npu_dtype t) {
  return t == NPU_BFP16 ? 8 : (t == NPU_MX6 ? 16 : 1);
}

// The uint8 output cast (NPU_U8): u8 = sat(rne(fp32(v * quant_multiplier + zero_point))).
struct OutQ {
  float quant_multiplier, zero_point;
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
      o[first + j] = uint8_t(quant(v[j], q.quant_multiplier, q.zero_point, NPU_U8));
    else
      store_float(out, t, first + j, v[j], rnd, t == NPU_FP8 ? 0 : kFtzIn);
  }
}

struct Gemm {
  const uint8_t* a;
  const uint8_t* b;
  const float* bias;
  int64_t m, k, n;
};

__device__ uint32_t gemm_bfp16_acc(const Gemm& p, int64_t row, int64_t col, uint32_t acc,
                                   int64_t k0, int64_t k1) {
  const int64_t kb = p.k / 8;
  Bfp16Fields x[8], z[8];
  for (int64_t c = k0 / 8; c < k1 / 8; c += 8) {
    for (int j = 0; j < 8; ++j) {
      bfp16_fields(p.a + (row * kb + c + j) * kBfpBytes, &x[j]);
      bfp16_fields(p.b + ((c + j) * p.n + col) * kBfpBytes, &z[j]);
    }
    acc = mmac_bfp16(acc, x, z);
  }
  return acc;
}

__device__ uint32_t gemm_mx6_acc(const Gemm& p, int64_t row, int64_t col, uint32_t acc) {
  const int64_t kb = p.k / 16;
  Mx6Fields x[8], z[8];
  for (int64_t c = 0; c < kb; c += 8) {
    for (int j = 0; j < 8; ++j) {
      mx6_fields(p.a + (row * kb + c + j) * kMx6Bytes, &x[j]);
      mx6_fields(p.b + ((c + j) * p.n + col) * kMx6Bytes, &z[j]);
    }
    acc = mmac_mx6(acc, x, z);
  }
  return acc;
}

// Thread i computes output group i (g values along the output row).
__global__ void gemm_block_kernel(int64_t groups, Gemm p, void* out, int g, int mx6, int narrow,
                                  int act, float alpha, int transpose, npu_dtype ofm_type, OutQ q) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= groups) return;
  const int64_t cols = transpose ? p.m : p.n;
  const int64_t first = i * g, r = first / cols, c0 = first % cols;
  float v[16];
  for (int j = 0; j < g; ++j) {
    const int64_t row = transpose ? c0 + j : r, col = transpose ? r : c0 + j;
    const uint32_t acc0 = p.bias ? f32_bits(p.bias[col]) : 0u;
    const uint32_t acc = mx6 ? gemm_mx6_acc(p, row, col, acc0) : gemm_bfp16_acc(p, row, col, acc0, 0, p.k);
    v[j] = activate_block(bits_f32(acc), narrow, act, alpha);
  }
  store(out, first, v, g, ofm_type, kHalfAway, q);
}

// act(bf16(gate GEMM)) * bf16(up GEMM); the product is exact in fp32.
__global__ void gemm_gated_kernel(int64_t groups, Gemm gate, Gemm up, void* out, int g, int act,
                                  npu_dtype ofm_type) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= groups) return;
  const int64_t first = i * g, row = first / gate.n, c0 = first % gate.n;
  float v[16];
  for (int j = 0; j < g; ++j) {
    const int64_t col = c0 + j;
    const uint32_t ag = gemm_bfp16_acc(gate, row, col, gate.bias ? f32_bits(gate.bias[col]) : 0u, 0,
                                       gate.k);
    const uint32_t au = gemm_bfp16_acc(up, row, col, up.bias ? f32_bits(up.bias[col]) : 0u, 0, up.k);
    v[j] = mul_q(core_act_bf16(acc_to_bf16(bits_f32(ag)), act), acc_to_bf16(bits_f32(au)));
  }
  store(out, first, v, g, ofm_type, kHalfAway);
}

struct Eltwise {
  const void* ifm2;
  int64_t m2, sv_k;
  int bfp, is_mul;
};

// acc + ifm2[row][col] as one identity block MMAC (ifm2 rows of n values, n % 64 == 0).
__device__ uint32_t e_block(const Eltwise& e, int64_t n, int64_t row, int64_t col, uint32_t acc) {
  const uint8_t* x2 = static_cast<const uint8_t*>(e.ifm2);
  const int64_t b = row * (n / 8) + col / 8, g0 = b & ~int64_t(7);
  const float a = decode_bfp16_identity(x2, g0, g0 + 8, b, col % 8);
  return f32_bits(mmac_add_bfp16(x2, g0, g0 + 8, b, col % 8, a, bits_f32(acc)));
}

__global__ void gemm_eltwise_kernel(int64_t groups, Gemm p, Eltwise e, void* out, int g,
                                    npu_dtype ofm_type) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= groups) return;
  const int64_t first = i * g, row = first / p.n, c0 = first % p.n;
  const int64_t r2 = e.m2 == 1 ? 0 : row;
  float v[8];
  for (int j = 0; j < g; ++j) {
    const int64_t col = c0 + j;
    const uint32_t bias = p.bias ? f32_bits(p.bias[col]) : 0u;
    uint32_t acc;
    if (!e.bfp) {
      const float x2 = bf16_to_f32(static_cast<const uint16_t*>(e.ifm2)[r2 * p.n + col]);
      if (e.is_mul) {
        acc = gemm_bfp16_acc(p, row, col, bias, 0, p.k);
        v[j] = mul_q(acc_to_bf16(bits_f32(acc)), x2);
        continue;
      }
      // seed: fp32(ifm2 + bias)
      acc = gemm_bfp16_acc(p, row, col, p.bias ? f32_bits(add_q(x2, p.bias[col])) : f32_bits(x2), 0, p.k);
    } else if ((row < 64 && col < 64) || e.sv_k == 64) {  // the first 64 x 64 output tile
      acc = gemm_bfp16_acc(p, row, col, e_block(e, p.n, r2, col, bias), 0, p.k);
    } else {
      acc = gemm_bfp16_acc(p, row, col, bias, 0, 64);
      acc = gemm_bfp16_acc(p, row, col, e_block(e, p.n, r2, col, acc), 64, p.k);
    }
    v[j] = bits_f32(acc);
  }
  store(out, first, v, g, ofm_type, kHalfAway);
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
                bfp16_fields(p.wts + (((y * p.kw + kx) * cib + ch / 8) * p.co + oc) * kBfpBytes, &z[b]);
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
                mx6_fields(p.wts + (((y * p.kw + kx) * cib + ch / 16) * p.co + oc) * kMx6Bytes, &z[b]);
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
  npu_dtype ifm2_type;
  int64_t h2, w2, c2;
  float coeff;
};

__device__ inline float residual(const Residual& r, int64_t oy, int64_t ox, int64_t oc) {
  const int64_t j = ((r.h2 == 1 ? 0 : oy) * r.w2 + (r.w2 == 1 ? 0 : ox)) * r.c2 + (r.c2 == 1 ? 0 : oc);
  if (r.ifm2_type == NPU_FP8) return fp8_to_f32(static_cast<const uint8_t*>(r.ifm2)[j]);
  const uint16_t h = static_cast<const uint16_t*>(r.ifm2)[j];
  return r.ifm2_type == NPU_F16 ? f16_to_f32(h) : bf16_to_f32(h);
}

__global__ void conv_block_kernel(int64_t groups, Conv p, const float* bias, Residual res, void* out,
                                  int g, int mx6, int narrow, int act, float alpha,
                                  npu_dtype ofm_type) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= groups) return;
  const int64_t first = i * g, pix = first / p.co, oc0 = first % p.co;
  const int64_t oy = pix / p.wo, ox = pix % p.wo;
  float v[16];
  for (int j = 0; j < g; ++j) {
    const int64_t oc = oc0 + j;
    uint32_t acc0 = bias ? f32_bits(bias[oc]) : 0u;
    if (res.ifm2) {
      const float r = residual(res, oy, ox, oc);
      acc0 = bias ? f32_bits(add_q(r, bias[oc])) : f32_bits(r);
    }
    const uint32_t acc = mx6 ? conv_mx6_acc(p, oy, ox, oc, acc0) : conv_bfp16_acc(p, oy, ox, oc, acc0);
    v[j] = res.ifm2 ? bits_f32(acc) : activate_block(bits_f32(acc), narrow, act, alpha);
  }
  store(out, first, v, g, ofm_type, res.ifm2 ? kHalfEven : kHalfAway);
}

struct ActFp8 {
  int act;
  float coeff0, coeff1, relu6_max;
  const float* slope;
};

__device__ inline float act_fp8(float v, const ActFp8& a, int64_t oc) {
  const int act = a.act;
  const float* slope = a.slope;
  if (act == 1) return vmax(v, 0.0f);
  if (act == 2) return vclamp(v, 0.0f, a.relu6_max);
  if (act == 3) return fma_q(vmin(v, 0.0f), a.coeff0, vmax(v, 0.0f));
  if (act == 8) return vclamp(v, a.coeff0, a.coeff1);
  if (act == 9) return v < 0.0f ? mul_q(v, slope[oc]) : v;
  return v;
}

// The in-place narrowing of an f16 activation of the gemm: the exact fp32 product with the f16
// multiplier, clamped to +-448, then E4M3 half to even.
__device__ inline uint8_t quantize_f16_fp8(uint16_t x, float r) {
  return fp8_out(f16_to_f32(x), r, kHalfEven);
}

__global__ void gemm_fp8_kernel(int64_t total, const void* a, npu_dtype a_type, const uint8_t* b,
                                const float* c0, const float* c2, void* out, int64_t k, int64_t n,
                                float r, npu_dtype ofm_type) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t row = i / n, col = i % n;
  uint32_t acc = 0;
  uint8_t x[64], z[64];
  for (int64_t k0 = 0; k0 < k; k0 += 64) {
    for (int l = 0; l < 64; ++l) {
      const int64_t e = row * k + k0 + l;
      x[l] = k0 + l >= k ? 0
             : a_type == NPU_F16 ? quantize_f16_fp8(static_cast<const uint16_t*>(a)[e], r)
                                 : static_cast<const uint8_t*>(a)[e];
      z[l] = k0 + l < k ? b[(k0 + l) * n + col] : 0;
    }
    acc = mmac_fp8(acc, x, 1, z, 1, 64);
  }
  const float v = fma_q(bits_f32(acc), c2[col], c0[col]);
  store(out, i, &v, 1, ofm_type, kHalfAway);
}

__global__ void conv_fp8_kernel(int64_t total, Conv p, const float* c0, const float* c2,
                                Residual res, void* out, ActFp8 act, npu_dtype ofm_type) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t oc = i % p.co, ox = (i / p.co) % p.wo, oy = i / (p.co * p.wo);
  const float acc = bits_f32(conv_fp8_acc(p, oy, ox, oc));
  float v;
  if (res.ifm2)
    v = fma_q(acc, c2[oc], fma_q(residual(res, oy, ox, oc), res.coeff, c0[oc]));
  else
    v = act_fp8(fma_q(acc, c2[oc], c0[oc]), act, oc);
  store(out, i, &v, 1, ofm_type, kHalfAway);
}

__global__ void dwc_fp8_kernel(int64_t total, const uint8_t* ifm, const uint8_t* wts, const float* c0,
                               const float* c2, void* out, int64_t w, int64_t c, int64_t kh,
                               int64_t kw, int64_t wo, int sh, int sw, int act, float alpha,
                               float relu6_max, npu_dtype ofm_type) {
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
  else if (act == 2) v = vclamp(v, 0.0f, relu6_max);
  else if (act == 3) v = vmax(v, mul_q(v, alpha));
  store(out, i, &v, 1, ofm_type, kHalfAway);
}

// Output types: bf16 or the entry's packed block format; the fp8 entries bf16 / f16 / fp8.
bool block_ofm_ok(npu_dtype t, npu_dtype packed) { return t == NPU_BF16 || t == packed; }
bool fp8_ofm_ok(npu_dtype t) { return is_half_float(t) || t == NPU_FP8; }
bool stride_ok(int s) { return s == 1 || s == 2; }
const uint8_t* u8p(const void* p) { return static_cast<const uint8_t*>(p); }
const float* f32p(const void* p) { return static_cast<const float*>(p); }

Conv make_conv(const void* ifm, const void* wts, int64_t h, int64_t w, int64_t ci, int64_t kh,
               int64_t kw, int64_t co, int sh, int sw, int64_t kh_sv, int64_t ci_sv,
               int64_t co_sv) {
  return {u8p(ifm), u8p(wts), h, w, ci, kh, kw, co, h == 0 ? 0 : (h - kh) / sh + 1,
          w == 0 ? 0 : (w - kw) / sw + 1, kh_sv, ci_sv, co_sv, sh, sw};
}

// The tilings a convolution kernel takes (see the header).
enum ConvKind { kConvBfp16, kConvMx6, kConvFp8 };

bool conv_ok(ConvKind kind, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
             int sh, int sw, int64_t kh_sv, int64_t ci_sv, int64_t co_sv) {
  const int64_t kmax = kind == kConvFp8 ? 7 : 15;
  bool ci_sv_ok, co_sv_ok;
  if (kind == kConvBfp16) {
    ci_sv_ok = ci_sv >= 64 && ci_sv % 64 == 0;
    co_sv_ok = co_sv == 64;
  } else if (kind == kConvMx6) {
    ci_sv_ok = ci_sv == 128 || ci_sv == 256;
    co_sv_ok = co_sv == 64;
  } else {
    ci_sv_ok = ci_sv >= 8 && ci_sv <= 256 && (ci_sv & (ci_sv - 1)) == 0;
    co_sv_ok = co_sv == 8 || co_sv == 16 || co_sv == 32 || co_sv == 64;
  }
  // an empty image (h or w 0) has an empty output
  return ci_sv_ok && co_sv_ok && ci >= 1 && co >= 1 && h >= 0 && w >= 0 && (h == 0 || h >= kh) &&
         (w == 0 || w >= kw) && kh >= 1 && kw >= 1 && kw <= kmax &&
         stride_ok(sh) && stride_ok(sw) && kh_sv >= 1 && kh_sv <= kmax && kh % kh_sv == 0 &&
         ci % ci_sv == 0 && co % co_sv == 0;
}

// narrow / activation / output type rules of the bfp16 and mx6 kernels: the activation needs
// the bf16 rounding of narrow and a block-format output.
bool block_act_ok(int narrow, int activation, int max_act, npu_dtype ofm_type) {
  return activation >= 0 && activation <= max_act && (activation == 0 || narrow) &&
         (activation == 0 || ofm_type != NPU_BF16);
}

int gemm_eltwise(void* stream, const void* a, const void* b, const void* bias,
                 const void* ifm2, npu_dtype ifm2_type, int64_t m2, void* out, int64_t m,
                 int64_t k, int64_t n, int64_t sv_k, int is_mul, npu_dtype ofm_type) {
  const int bfp = ifm2_type == NPU_BFP16;
  if (any_null(a, b, ifm2, out) || m < 0 || n < 0 || k < 64 || k % 64 || (m2 != 1 && m2 != m) ||
      (ifm2_type != NPU_BF16 && !bfp) || (bfp && (is_mul || n % 64)) ||
      (!is_mul && (sv_k < 64 || sv_k > 4096 || sv_k % 64)) ||
      !block_ofm_ok(ofm_type, NPU_BFP16) || n % out_group(ofm_type))
    return -1;
  const int g = out_group(ofm_type);
  const Gemm p{u8p(a), u8p(b), f32p(bias), m, k, n};
  const Eltwise e{ifm2, m2, sv_k, bfp, is_mul};
  return launch(stream, m * n / g, gemm_eltwise_kernel, p, e, out, g, ofm_type);
}

int conv_block(void* stream, const void* ifm, const void* wts, const void* bias,
               const void* ifm2, int64_t h2, int64_t w2, int64_t c2, void* out,
               npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
               int64_t co, int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv, int narrow,
               int act, float alpha, int mx6) {
  const npu_dtype packed = mx6 ? NPU_MX6 : NPU_BFP16;
  if (any_null(ifm, wts, out) ||
      !conv_ok(mx6 ? kConvMx6 : kConvBfp16, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv,
               ci_sv, 64) ||
      !block_act_ok(narrow, act, 3, ofm_type) || !block_ofm_ok(ofm_type, packed) ||
      co % out_group(ofm_type))
    return -1;
  const Conv p = make_conv(ifm, wts, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv, ci_sv, 64);
  if (ifm2 && ((h2 != 1 && h2 != p.ho) || (w2 != 1 && w2 != p.wo) || (c2 != 1 && c2 != co)))
    return -1;
  const Residual r{ifm2, NPU_BF16, h2, w2, c2, 1.0f};
  const int g = out_group(ofm_type);
  return launch(stream, p.ho * p.wo * co / g, conv_block_kernel, p, f32p(bias), r, out, g, mx6,
                narrow, act, scalar_cast(alpha, NPU_BF16, kHalfEven), ofm_type);
}

int conv_fp8(void* stream, const void* ifm, npu_dtype ifm_type, uint16_t ifm_r16,
             const void* wts, const void* c0, const void* c2, const void* ifm2,
             npu_dtype ifm2_type, int64_t h2, int64_t w2, int64_t c2_dim, float ifm2_coeff,
             void* out, npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci, int64_t kh,
             int64_t kw, int64_t co, int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv,
             int64_t co_sv, const ActFp8& act) {
  const int a = act.act;
  if (any_null(ifm, wts, c0, c2, out) || (ifm_type != NPU_FP8 && ifm_type != NPU_F16) ||
      !conv_ok(kConvFp8, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv, ci_sv, co_sv) ||
      !(a == 0 || a == 1 || a == 2 || a == 3 || a == 8 || a == 9) || (a == 9 && !act.slope) ||
      !fp8_ofm_ok(ofm_type) ||
      (ifm2 && ifm2_type != NPU_FP8 && !is_half_float(ifm2_type)))
    return -1;
  Conv p = make_conv(ifm, wts, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv, ci_sv, co_sv);
  if (ifm_type == NPU_F16) {
    p.ifm = nullptr;
    p.ifm16 = static_cast<const uint16_t*>(ifm);
    p.ifm_r16 = ifm_r16;
  }
  if (ifm2 &&
      ((h2 != 1 && h2 != p.ho) || (w2 != 1 && w2 != p.wo) || (c2_dim != 1 && c2_dim != co)))
    return -1;
  const Residual r{ifm2, ifm2_type, h2, w2, c2_dim, ifm2_coeff};
  return launch(stream, p.ho * p.wo * co, conv_fp8_kernel, p, f32p(c0), f32p(c2), r, out, act,
                ofm_type);
}

}  // namespace

extern "C" {

int npu_gemm_bfp16(void* stream, const void* a, const void* b, const void* bias, void* out,
                   npu_dtype ofm_type, int64_t m, int64_t k, int64_t n, int transpose,
                   float quant_multiplier, float zero_point, int narrow, int activation,
                   float lrelu_alpha) {
  const int64_t cols = transpose ? m : n;
  const bool u8 = ofm_type == NPU_U8;
  if (any_null(a, b, out) || m < 0 || n < 0 || k < 64 || k % 64 ||
      !block_act_ok(narrow, activation, 7, ofm_type) || (u8 && (!narrow || activation)) ||
      !(block_ofm_ok(ofm_type, NPU_BFP16) || u8) || cols % out_group(ofm_type) ||
      (transpose && (bias || (activation >= 1 && activation <= 3) || ofm_type == NPU_BF16)))
    return -1;
  const int g = out_group(ofm_type);
  const Gemm p{u8p(a), u8p(b), f32p(bias), m, k, n};
  return launch(stream, m * n / g, gemm_block_kernel, p, out, g, 0, narrow, activation,
                scalar_cast(lrelu_alpha, NPU_BF16, kHalfEven), transpose, ofm_type,
                OutQ{quant_multiplier, zero_point});
}

int npu_gemm_gelu_mul_bfp16(void* stream, const void* a, const void* b_gate, const void* b_up,
                            const void* bias_gate, const void* bias_up, void* out, int64_t m,
                            int64_t k, int64_t n, int activation) {
  if (any_null(a, b_gate, b_up, out) || m < 0 || n < 0 || k < 64 || k % 64 || n % 8 ||
      !(activation == 0 || (activation >= 5 && activation <= 7)))
    return -1;
  const Gemm gate{u8p(a), u8p(b_gate), f32p(bias_gate), m, k, n};
  const Gemm up{u8p(a), u8p(b_up), f32p(bias_up), m, k, n};
  return launch(stream, m * n / 8, gemm_gated_kernel, gate, up, out, 8, activation, NPU_BFP16);
}

int npu_gemm_add_bfp16(void* stream, const void* a, const void* ifm2, npu_dtype ifm2_type,
                       const void* b, const void* bias, void* out, npu_dtype ofm_type,
                       int64_t m, int64_t k, int64_t n, int64_t m2, int64_t sv_k) {
  return gemm_eltwise(stream, a, b, bias, ifm2, ifm2_type, m2, out, m, k, n, sv_k, 0, ofm_type);
}

int npu_gemm_mul_bfp16(void* stream, const void* a, const void* ifm2, const void* b,
                       const void* bias, void* out, npu_dtype ofm_type, int64_t m, int64_t k,
                       int64_t n, int64_t m2) {
  return gemm_eltwise(stream, a, b, bias, ifm2, NPU_BF16, m2, out, m, k, n, 64, 1, ofm_type);
}

int npu_conv_bfp16(void* stream, const void* ifm, const void* wts, const void* bias, void* out,
                   npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                   int64_t co, int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv,
                   int narrow, int activation, float lrelu_alpha) {
  return conv_block(stream, ifm, wts, bias, nullptr, 1, 1, 1, out, ofm_type, h, w, ci, kh, kw, co,
                    stride_h, stride_w, kh_sv, ci_sv, narrow, activation, lrelu_alpha, 0);
}

int npu_conv_add_bfp16(void* stream, const void* ifm, const void* ifm2, const void* wts,
                       const void* bias, void* out, npu_dtype ofm_type, int64_t h, int64_t w,
                       int64_t ci, int64_t kh, int64_t kw, int64_t co, int64_t h2, int64_t w2,
                       int64_t c2, int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv) {
  if (!ifm2) return -1;
  return conv_block(stream, ifm, wts, bias, ifm2, h2, w2, c2, out, ofm_type, h, w, ci, kh, kw, co,
                    stride_h, stride_w, kh_sv, ci_sv, 0, 0, 0.0f, 0);
}

int npu_gemm_mx6(void* stream, const void* a, const void* b, const void* bias, void* out,
                 npu_dtype ofm_type, int64_t m, int64_t k, int64_t n, int narrow, int activation,
                 float lrelu_alpha) {
  if (any_null(a, b, out) || m < 0 || n < 0 || k < 128 || k % 128 ||
      !block_act_ok(narrow, activation, 3, ofm_type) || !block_ofm_ok(ofm_type, NPU_MX6) ||
      n % out_group(ofm_type))
    return -1;
  const int g = out_group(ofm_type);
  const Gemm p{u8p(a), u8p(b), f32p(bias), m, k, n};
  return launch(stream, m * n / g, gemm_block_kernel, p, out, g, 1, narrow, activation,
                scalar_cast(lrelu_alpha, NPU_BF16, kHalfEven), 0, ofm_type, OutQ{});
}

int npu_conv_mx6(void* stream, const void* ifm, const void* wts, const void* bias, void* out,
                 npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                 int64_t co, int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv, int narrow,
                 int activation, float lrelu_alpha) {
  return conv_block(stream, ifm, wts, bias, nullptr, 1, 1, 1, out, ofm_type, h, w, ci, kh, kw, co,
                    stride_h, stride_w, kh_sv, ci_sv, narrow, activation, lrelu_alpha, 1);
}

int npu_gemm_fp8(void* stream, const void* a, npu_dtype a_type, const void* b, const void* c0,
                 const void* c2, void* out, npu_dtype ofm_type, int64_t m, int64_t k, int64_t n,
                 float ifm_quant_mult) {
  if (any_null(a, b, c0, c2, out) || (a_type != NPU_FP8 && a_type != NPU_F16) || m < 0 ||
      k < 1 || n < 0 || !fp8_ofm_ok(ofm_type))
    return -1;
  // the fp32 multiplier is rounded to f16 on the core (half to even)
  const float r = scalar_cast(ifm_quant_mult, NPU_F16, kHalfEven);
  return launch(stream, m * n, gemm_fp8_kernel, a, a_type, u8p(b), f32p(c0), f32p(c2), out, k,
                n, r, ofm_type);
}

int npu_conv_fp8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                 const void* c0, const void* c2, const void* slope, void* out,
                 npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                 int64_t co, int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv,
                 int64_t co_sv, uint16_t ifm_scale_recip_f16, int activation,
                 float activation_coeff0, float activation_coeff1, float relu6_max) {
  const ActFp8 act{activation, activation_coeff0, activation_coeff1, relu6_max, f32p(slope)};
  return conv_fp8(stream, ifm, ifm_type, ifm_scale_recip_f16, wts, c0, c2, nullptr, NPU_FP8, 1,
                  1, 1, 0.0f, out, ofm_type, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv,
                  ci_sv, co_sv, act);
}

int npu_conv_add_fp8(void* stream, const void* ifm, const void* ifm2, npu_dtype ifm2_type,
                     const void* wts, const void* c0, const void* c2, void* out,
                     npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                     int64_t co, int64_t h2, int64_t w2, int64_t c2_dim, int stride_h,
                     int stride_w, int64_t kh_sv, int64_t ci_sv, int64_t co_sv,
                     float ifm2_coeff) {
  if (!ifm2) return -1;
  return conv_fp8(stream, ifm, NPU_FP8, 0, wts, c0, c2, ifm2, ifm2_type, h2, w2, c2_dim,
                  ifm2_coeff, out, ofm_type, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv,
                  ci_sv, co_sv, ActFp8{0, 0.0f, 0.0f, 0.0f, nullptr});
}

int npu_depthwise_conv_fp8(void* stream, const void* ifm, const void* wts, const void* c0,
                           const void* c2, void* out, npu_dtype ofm_type, int64_t h, int64_t w,
                           int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w,
                           int activation, float alpha, float relu6_max) {
  if (any_null(ifm, wts, c0, c2, out) || kh < 1 || kw < 1 || kh > 15 || kw > 15 || h < 0 ||
      w < 0 || (h > 0 && h < kh) || (w > 0 && w < kw) || c < 64 || c % 64 ||
      !stride_ok(stride_h) || !stride_ok(stride_w) || activation < 0 || activation > 3 ||
      !fp8_ofm_ok(ofm_type))
    return -1;
  const int64_t ho = h == 0 ? 0 : (h - kh) / stride_h + 1;
  const int64_t wo = w == 0 ? 0 : (w - kw) / stride_w + 1;
  return launch(stream, ho * wo * c, dwc_fp8_kernel, u8p(ifm), u8p(wts), f32p(c0), f32p(c2), out,
                w, c, kh, kw, wo, stride_h, stride_w, activation, alpha, relu6_max, ofm_type);
}

}  // extern "C"
