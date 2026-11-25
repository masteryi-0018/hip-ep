/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Layer norm, RMS norm and L2 norm kernels. One thread per row: the lane walk
 * of a row is sequential.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "common/bits.h"
#include "common/block_fp.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/mmac.h"
#include "common/nlf.h"
#include "common/quant.h"
#include "reduction/layernorm.h"

using namespace npu_emu;

namespace {

constexpr int kLanes = 32;

// Element kinds beyond npu_dtype: bfp16 blocks (8 values per 9 bytes).
enum Kind : int {
  kF32 = NPU_F32,
  kF16 = NPU_F16,
  kBF16 = NPU_BF16,
  kFP8 = NPU_FP8,
  kI8 = NPU_I8,
  kU8 = NPU_U8,
  kI16 = NPU_I16,
  kU16 = NPU_U16,
  kBFP16 = 100,
  kBFP16H = 101,  // bfp16 encoded from f16-rounded values
};

struct Src {
  const void* p;
  int kind;
  int64_t n;  // elements in the buffer (fp8 / bfp16 decode groups are buffer-aligned)
};

// Element i as fp32; fp8 is decoded in groups of 8 (-0 -> +0, NaN spreads).
__device__ inline float ld(const Src& s, int64_t i) {
  switch (s.kind) {
    case kFP8:
      return fp8_decode_identity(static_cast<const uint8_t*>(s.p), i, s.n);
    case kBFP16:  // the bfp16 -> bf16 cast, 64-element groups from the buffer start
      return bf16_to_f32(decode_bfp16_bf16(static_cast<const uint8_t*>(s.p), i >> 3, s.n >> 3,
                                           int(i & 7)));
    default:
      return load_float(s.p, npu_dtype(s.kind), i);
  }
}

__device__ inline int64_t ld_int(const Src& s, int64_t i) {
  return load_int(s.p, npu_dtype(s.kind), i);
}

// Output sink. bfp16 rounds to bf16 first; integer kinds take
// sat(rne(v * q_scale + q_zp)) when quantize is set, else sat(rne(v)).
struct Dst {
  void* p;
  int kind;
  bool quantize;
  float q_scale, q_zp;
};

__device__ inline void st_int(const Dst& d, int64_t i, float v) {
  const npu_dtype t = npu_dtype(d.kind);
  store_int(d.p, t, i,
            d.quantize ? quant(v, d.q_scale, d.q_zp, t) : float_to_int(v, t, 0, kHalfEven));
}

// Store out[i0 .. i0 + 8) (i0 % 8 == 0).
__device__ inline void st8(const Dst& d, int64_t i0, const float* v) {
  if (d.kind == kBFP16 || d.kind == kBFP16H) {
    const npu_dtype rt = d.kind == kBFP16 ? NPU_BF16 : NPU_F16;
    float r[8];
    for (int k = 0; k < 8; ++k) r[k] = round_float(v[k], rt, kHalfEven);
    encode_bfp16(r, kHalfEven, static_cast<uint8_t*>(d.p) + (i0 >> 3) * 9);
    return;
  }
  for (int k = 0; k < 8; ++k) {
    const int64_t i = i0 + k;
    if (d.kind == kFP8) {
      store_float(d.p, NPU_FP8, i, v[k], kHalfEven);
    } else if (is_small_int(npu_dtype(d.kind))) {
      st_int(d, i, v[k]);
    } else {
      store_float(d.p, npu_dtype(d.kind), i, v[k], kHalfEven);
    }
  }
}

// Store one element (rows whose C is not a multiple of 8; never bfp16).
__device__ inline void st1(const Dst& d, int64_t i, float v) {
  if (d.kind == kFP8) {
    store_float(d.p, NPU_FP8, i, v, kHalfEven);
  } else if (is_small_int(npu_dtype(d.kind))) {
    st_int(d, i, v);
  } else {
    store_float(d.p, npu_dtype(d.kind), i, v, kHalfEven);
  }
}

// x = dequant(q, scale, offset) rounded half to even to the compute type; the offset is the
// negated product -(zp * scale), not dq_offset's (-zp) * scale.
struct Dq {
  float scale, offset;
};
__device__ inline float dq(const Dq& c, int64_t q, int compute) {
  return round_float(dequant(q, c.scale, c.offset), npu_dtype(compute), kHalfEven);
}

// Reciprocal square root: bf16 exponent seed and two Newton steps.
__device__ inline float fast_rsqrt(float x) {
  x = vmax(x, 1.0e-5f);
  const int half = int(f32_to_bf16(x, kHalfEven)) >> 1;  // positive: floor == shift
  float y = bf16_to_f32(uint16_t(0x5f37 - half));
  const float h = mul_q(x, 0.5f);
  for (int k = 0; k < 2; ++k) {
    const float yy = mul_q(y, y);
    y = mul_q(y, msc_q(h, yy, 1.5f));
  }
  return y;
}

// Reciprocal square root: f16 seed 0x59bb - rne(bits / 2) from the clamped
// value, Newton steps on the unclamped one.
__device__ inline float rsqrt16(float v) {
  const float vs = vclamp(v, 6.2e-5f, 65000.0f);
  const int b = int(f32_to_f16(vs, kHalfEven));  // positive, < 0x7c00
  const int bh = int(shift_round(b, 1, kHalfEven));  // b / 2, half to even
  float y = f16_to_f32(uint16_t(0x59bb - bh));
  const float h = mul_q(v, 0.5f);
  for (int k = 0; k < 2; ++k) {
    const float y2 = mul_q(y, y);
    y = mul_q(y, msc_q(h, y2, 1.5f));
  }
  return y;
}

// ik_type: the dtype inv_std is rounded to before the normalization (F32 = kept).
// Statistics over channels [0, c_real) with divisor c_real; the rest of the
// row only takes part in the normalization.
__global__ void core_layernorm_kernel(int64_t n, Src in, Src alpha, Src beta, Dst out, int C,
                                     int c_real, float eps, int ik_type, NlfTables t) {
  NPU_TID;
  const int64_t base = i * C;
  const float inv_s = hw_inv_f32(float(c_real), t.inv);
  float ax[kLanes], ax2[kLanes];
  for (int l = 0; l < kLanes; ++l) ax[l] = ax2[l] = 0.0f;
  for (int c = 0; c < C; ++c) {
    const float x = c < c_real ? ld(in, base + c) : 0.0f;
    const int l = c & (kLanes - 1);
    ax[l] = fma_q(x, inv_s, ax[l]);
    ax2[l] = fma_q(x, x, ax2[l]);
  }
  for (int l = 0; l < kLanes; ++l) ax2[l] = mul_q(ax2[l], inv_s);
  const float mu = vaddr_f32_reduce(ax, kLanes);
  float var = vaddr_f32_reduce(ax2, kLanes, true, eps);
  var = msc_q(mu, mu, var);
  const float istd = hw_invsqrt_f32(var, t.isq);
  const float mis = negmul_q(istd, mu);
  const float ik = round_float(istd, npu_dtype(ik_type), kHalfEven);
  for (int c0 = 0; c0 < C; c0 += 8) {
    float v[8];
    for (int k = 0; k < 8; ++k) {
      const int c = c0 + k;
      const float t0 = fma_q(ld(in, base + c), ik, mis);
      v[k] = fma_q(t0, ld(alpha, c), ld(beta, c));
    }
    st8(out, base + c0, v);
  }
}

__global__ void rmsnorm_kernel(int64_t n, Src in, Src gamma, Dst out, int C, float inv_inner,
                               float eps, const uint32_t* isq) {
  NPU_TID;
  const int64_t base = i * C;
  float acc[kLanes];
  for (int l = 0; l < kLanes; ++l) acc[l] = 0.0f;
  for (int c = 0; c < C; ++c) {
    const float x = ld(in, base + c);
    acc[c & (kLanes - 1)] = fma_q(x, x, acc[c & (kLanes - 1)]);
  }
  const float s = vaddr_f32_reduce(acc, kLanes);
  // Unfused multiply and add.
  const float r = hw_invsqrt_f32(add_q(mul_q(s, inv_inner), eps), isq);
  for (int c0 = 0; c0 < C; c0 += 8) {
    float v[8];
    for (int k = 0; k < 8; ++k)
      v[k] = mul_q(mul_q(ld(in, base + c0 + k), r), ld(gamma, c0 + k));
    st8(out, base + c0, v);
  }
}

__global__ void l2norm_kernel(int64_t n, Src in, Src alpha, Src beta, Dst out, int C,
                              const uint32_t* isq) {
  NPU_TID;
  const int64_t base = i * C;
  float acc[kLanes];
  for (int l = 0; l < kLanes; ++l) acc[l] = 0.0f;
  for (int c = 0; c < C; ++c) {
    const float x = ld(in, base + c);
    acc[c & (kLanes - 1)] = fma_q(x, x, acc[c & (kLanes - 1)]);
  }
  const float r = hw_invsqrt_f32(vaddr_f32_reduce(acc, kLanes, true, 1e-12f), isq);
  for (int c0 = 0; c0 < C; c0 += 8) {
    float v[8];
    for (int k = 0; k < 8; ++k) {
      const int c = c0 + k;
      v[k] = fma_q(mul_q(ld(alpha, c), ld(in, base + c)), r, ld(beta, c));
    }
    st8(out, base + c0, v);
  }
}

// Quantized L2 norm, bf16 compute.
__global__ void qdq_l2norm_kernel(int64_t n, Src in, Dq dqc, Src alpha, Src beta, bool affine,
                                  Dst out, int C, const uint32_t* isq) {
  NPU_TID;
  const int64_t base = i * C;
  float acc[kLanes];
  for (int l = 0; l < kLanes; ++l) acc[l] = 0.0f;
  for (int c = 0; c < C; ++c) {
    const float x = dq(dqc, ld_int(in, base + c), kBF16);
    acc[c & (kLanes - 1)] = fma_q(x, x, acc[c & (kLanes - 1)]);
  }
  const float r = vmin(hw_invsqrt_f32(vaddr_f32_reduce(acc, kLanes), isq), 1e12f);
  for (int c0 = 0; c0 < C; c0 += 8) {
    float v[8];
    for (int k = 0; k < 8; ++k) {
      const int c = c0 + k;
      const float x = dq(dqc, ld_int(in, base + c), kBF16);
      const float y = affine ? fma_q(x, mul_q(ld(alpha, c), r), ld(beta, c)) : mul_q(x, r);
      v[k] = round_float(y, NPU_BF16, kHalfEven);
    }
    st8(out, base + c0, v);
  }
}

// How a row's layer-norm input is formed; cd() = rounded half to even to the compute type.
enum PreMode {
  kPreDq = 0,         // cd(dq(a))
  kPreAdd = 1,        // a, b floats: cd(a + b) / fp8 -> f16
  kPreDqAdd = 2,      // cd(cd(dq_a(a)) + cd(dq_b(b)))
  kPreDqAddT = 3,     // cd(cd(dq_a(a)) + b), b compute
  kPreAddAddC = 4,    // cd(A + fp32(dq_b(b) + k[c]))
};

struct Pre {
  int mode;
  Src a, b, k;  // k: per-channel add_const (kPreAddAddC)
  Dq da, db;
  int compute;  // BF16 / F16 / F32
};

__device__ inline float pre(const Pre& p, int64_t i, int c) {
  switch (p.mode) {
    case kPreDq:
      return dq(p.da, ld_int(p.a, i), p.compute);
    case kPreAdd:
      // One fp32 add, then rounded to the compute type.
      return round_float(add_q(ld(p.a, i), ld(p.b, i)), npu_dtype(p.compute), kHalfEven);
    case kPreDqAdd:
      return round_float(add_q(dq(p.da, ld_int(p.a, i), p.compute),
                                   dq(p.db, ld_int(p.b, i), p.compute)),
                         npu_dtype(p.compute), kHalfEven);
    case kPreDqAddT:
      return round_float(add_q(dq(p.da, ld_int(p.a, i), p.compute), ld(p.b, i)),
                         npu_dtype(p.compute), kHalfEven);
    default: {  // kPreAddAddC
      const float A =
          is_small_int(npu_dtype(p.a.kind)) ? dq(p.da, ld_int(p.a, i), p.compute) : ld(p.a, i);
      const float dc = add_q(dequant(ld_int(p.b, i), p.db.scale, p.db.offset), ld(p.k, c));
      return round_float(add_q(A, dc), npu_dtype(p.compute), kHalfEven);
    }
  }
}

enum Stats {
  kFused = 0,        // reduce-add of scaled lanes, eps in lane 16
  kBatched = 1,      // reduce-add of raw lane sums
  kBatchedFast = 2,  // as kBatched with fast_rsqrt
  kC8Fused = 3,
  kC8BatchedFast = 4,
};

struct Ln {
  int stats;
  float eps;
  int ik_type;  // dtype inv_std is rounded to
};

// mu and inv_std of one row whose values come from pre(); returns mis too.
__device__ inline void ln_stats(const Pre& p, int64_t base, int C, const Ln& ln, float inv_s,
                                const NlfTables& t, float* ik_out, float* mis_out) {
  float mu, istd;
  if (ln.stats == kC8Fused || ln.stats == kC8BatchedFast) {
    float sx[8], sq[8];
    for (int l = 0; l < 8; ++l) {
      const float x = pre(p, base + l, l);
      sx[l] = x;  // x * 1.0 into a cleared accumulator
      sq[l] = mul_q(x, x);
    }
    int c = 8;
    if (ln.stats == kC8Fused && C > 8) {
      // The second group is added with unfused squares.
      for (int l = 0; l < 8; ++l) {
        const float x = pre(p, base + 8 + l, 8 + l);
        sx[l] = add_q(sx[l], x);
        sq[l] = add_q(sq[l], mul_q(x, x));
      }
      c = 16;
    }
    for (; c < C; ++c) {
      const float x = pre(p, base + c, c);
      sx[c & 7] = add_q(sx[c & 7], x);
      sq[c & 7] = fma_q(x, x, sq[c & 7]);
    }
    // Tree ((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7)).
    auto fold = [](const float* v) {
      float a[4], b[2];
      for (int k = 0; k < 4; ++k) a[k] = add_q(v[k], v[k + 4]);
      for (int k = 0; k < 2; ++k) b[k] = add_q(a[k], a[k + 2]);
      return add_q(b[0], b[1]);
    };
    const float s1 = fold(sx), s2 = fold(sq);
    mu = mul_q(s1, inv_s);
    const float va = add_q(mul_q(s2, inv_s), ln.eps);
    const float var = msc_q(mu, mu, va);
    istd = ln.stats == kC8Fused ? hw_invsqrt_f32(var, t.isq) : fast_rsqrt(var);
    *mis_out = mul_q(neg_f32(mu), istd);
    *ik_out = round_float(istd, npu_dtype(ln.ik_type), kHalfEven);
    return;
  }
  float sx[kLanes], sq[kLanes];
  for (int l = 0; l < kLanes; ++l) sx[l] = sq[l] = 0.0f;
  for (int c = 0; c < C; ++c) {
    const float x = pre(p, base + c, c);
    const int l = c & (kLanes - 1);
    sx[l] = add_q(sx[l], x);
    sq[l] = fma_q(x, x, sq[l]);
  }
  if (ln.stats == kFused) {
    for (int l = 0; l < kLanes; ++l) {
      sx[l] = mul_q(sx[l], inv_s);
      sq[l] = fma_q(sq[l], inv_s, l == 16 ? ln.eps : 0.0f);
    }
    mu = vaddr_f32_reduce(sx, kLanes);
    const float var = msc_q(mu, mu, vaddr_f32_reduce(sq, kLanes));
    istd = hw_invsqrt_f32(var, t.isq);
    *mis_out = negmul_q(mu, istd);
  } else {
    mu = mul_q(vaddr_f32_reduce(sx, kLanes), inv_s);
    const float e2 = fma_q(vaddr_f32_reduce(sq, kLanes), inv_s, ln.eps);
    const float var = msc_q(mu, mu, e2);
    istd = ln.stats == kBatched ? hw_invsqrt_f32(var, t.isq) : fast_rsqrt(var);
    *mis_out = negmul_q(istd, mu);
  }
  *ik_out = round_float(istd, npu_dtype(ln.ik_type), kHalfEven);
}

// out_round: the dtype the result is rounded to before the store (F32 = none).
__global__ void layernorm_kernel(int64_t n, Pre p, Ln ln, Src alpha, Src beta, Dst out,
                                 int out_round, int C, NlfTables t) {
  NPU_TID;
  const int64_t base = i * C;
  const float inv_s = hw_inv_f32(float(C), t.inv);
  float ik, mis;
  ln_stats(p, base, C, ln, inv_s, t, &ik, &mis);
  const bool blocks = out.kind == kBFP16H;  // C % 8 == 0
  float v[8];
  for (int c = 0; c < C; ++c) {
    const float t0 = fma_q(pre(p, base + c, c), ik, mis);
    const float y = round_float(fma_q(t0, ld(alpha, c), ld(beta, c)), npu_dtype(out_round),
                                kHalfEven);
    if (!blocks) {
      st1(out, base + c, y);
      continue;
    }
    v[c & 7] = y;
    if ((c & 7) == 7) st8(out, base + c - 7, v);
  }
}

// Integer statistics on the raw codes.
__global__ void qln_integer_kernel(int64_t n, Src in, const float* alpha, const float* beta,
                                   Dst out, int C, float dq_scale, NlfTables t) {
  NPU_TID;
  const int64_t base = i * C;
  int64_t S = 0, Q = 0;
  for (int c = 0; c < C; ++c) {
    const int64_t q = ld_int(in, base + c);
    S += q;
    Q += q * q;
  }
  const float inv_c = hw_inv_f32(float(C), t.inv);
  const float eps2 = mul_q(1e-5f, hw_inv_f32(mul_q(dq_scale, dq_scale), t.inv));
  const float mu = mac_exact(S, inv_c, 0.0f);
  const float e2 = mac_exact(Q, inv_c, eps2);
  const float var = vmax(msc_q(mu, mu, e2), eps2);
  const float r = round_float(rsqrt16(var), NPU_F16, kHalfEven);
  const float mub = mac_exact(S, inv_c, 1536.0f);
  const float cc = negmul_q(mub, r);
  for (int c = 0; c < C; ++c) {
    // 1536 + q is exact in f16 for 8-bit q.
    const float x = float(1536 + ld_int(in, base + c));
    st1(out, base + c, fma_q(fma_q(x, r, cc), alpha[c], beta[c]));
  }
}

Src src(const void* p, int kind, int64_t n) { return Src{p, kind, n}; }
Dst dst(void* p, int kind) { return Dst{p, kind, false, 0.0f, 0.0f}; }
Dst qdst(void* p, npu_dtype kind, float q_scale, float q_zp) {
  return Dst{p, int(kind), true, q_scale, q_zp};
}
Dq dqc(float scale, float zp) { return Dq{scale, negmul_q(zp, scale)}; }

int core_impl(void* stream, const void* in, int in_kind, npu_dtype alpha_type, const void* alpha,
             const void* beta, void* out, int out_kind, int ik_type, int64_t rows, int C,
             int c_real = 0, float eps = 1e-12f) {
  if (rows < 0 || C <= 0 || C % 32 || any_null(in, alpha, beta, out)) return -1;
  if (c_real < 0 || c_real > C) return -1;
  if (rows == 0) return 0;
  NlfTables t;
  if (int rc = nlf_tables(&t)) return rc;
  const int64_t n = rows * C;
  return launch(stream, rows, core_layernorm_kernel, src(in, in_kind, n), src(alpha, alpha_type, C),
                src(beta, alpha_type, C), dst(out, out_kind), C, c_real ? c_real : C, eps,
                ik_type, t);
}

int rms_impl(void* stream, const void* in, int in_kind, npu_dtype gamma_type, const void* gamma,
             void* out, int out_kind, int64_t rows, int C, float inv_inner, float eps) {
  if (rows < 0 || C <= 0 || C % 32 || any_null(in, gamma, out)) return -1;
  if (rows == 0) return 0;
  // The kernel takes an all-zero epsilon word for 1e-12 (+0 only; -0 stays -0).
  if (f32_bits(eps) == 0) eps = 1e-12f;
  NlfTables t;
  if (int rc = nlf_tables(&t)) return rc;
  return launch(stream, rows, rmsnorm_kernel, src(in, in_kind, rows * C), src(gamma, gamma_type, C),
                dst(out, out_kind), C, inv_inner, eps, t.isq);
}

int qdq_l2_impl(bool a16, void* stream, const void* in, npu_dtype in_type, const void* alpha,
                const void* beta, void* out, npu_dtype out_type, int64_t rows, int C,
                float dq_scale, float dq_zp, float q_scale, float q_zp) {
  auto ok = [&](npu_dtype t) { return is_int_bits(t, a16 ? 16 : 8); };
  if (rows < 0 || C <= 0 || C % 32 || any_null(in, out) || !ok(in_type) || !ok(out_type))
    return -1;
  if ((alpha == nullptr) != (beta == nullptr)) return -1;
  if (rows == 0) return 0;
  NlfTables t;
  if (int rc = nlf_tables(&t)) return rc;
  return launch(stream, rows, qdq_l2norm_kernel, src(in, in_type, rows * C), dqc(dq_scale, dq_zp),
                src(alpha, kBF16, C), src(beta, kBF16, C), alpha != nullptr,
                qdst(out, out_type, q_scale, q_zp), C, t.isq);
}

// alpha_type: dtype of alpha / beta.
int ln_impl(void* stream, const Pre& p, npu_dtype alpha_type, const void* alpha, const void* beta,
            Dst out, int out_round, int64_t rows, int C, float eps, int stats) {
  if (rows < 0 || C <= 0 || any_null(alpha, beta, out.p)) return -1;
  if (stats != NPU_LN_STATS_FUSED && stats != NPU_LN_STATS_BATCHED) return -1;
  if (stats == NPU_LN_STATS_BATCHED && C != 64 && C != 96) return -1;
  if (rows == 0) return 0;
  NlfTables t;
  if (int rc = nlf_tables(&t)) return rc;
  const Ln ln{stats == NPU_LN_STATS_FUSED ? kFused : kBatched, eps, p.compute};
  return launch(stream, rows, layernorm_kernel, p, ln, src(alpha, alpha_type, C),
                src(beta, alpha_type, C), out, out_round, C, t);
}

}  // namespace

extern "C" {

int npu_layer_norm_bf16(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                        const void* beta, void* out, int64_t rows, int C, int C_real,
                        float epsilon) {
  if (C_real < 0 || C_real > C) return -1;
  if (C_real == 0 && epsilon != 1e-12f) return -1;
  if (C_real != 0 && in_type != NPU_BF16) return -1;
  if (C_real != 0)
    return core_impl(stream, in, in_type, in_type, alpha, beta, out, in_type, in_type, rows, C,
                     C_real, epsilon);
  if (in_type == NPU_FP8) {
    if (C % 64) return -1;
    return core_impl(stream, in, kFP8, NPU_F16, alpha, beta, out, kFP8, kF32, rows, C);
  }
  if (!is_half_float(in_type)) return -1;
  return core_impl(stream, in, in_type, in_type, alpha, beta, out, in_type, in_type, rows, C);
}

int npu_layer_norm_bfp16(void* stream, const void* in, const void* alpha, const void* beta,
                         void* out, int64_t rows, int C) {
  return core_impl(stream, in, kBFP16, NPU_BF16, alpha, beta, out, kBFP16, kBF16, rows, C);
}

int npu_layer_norm_bf16_bfp16(void* stream, const void* in, const void* alpha,
                              const void* beta, void* out, int64_t rows, int C) {
  return core_impl(stream, in, kBF16, NPU_BF16, alpha, beta, out, kBFP16, kBF16, rows, C);
}

int npu_rmsnorm_bf16(void* stream, const void* in, npu_dtype in_type, const void* gamma,
                     void* out, int64_t rows, int C, float inv_inner_dim, float epsilon) {
  if (in_type == NPU_FP8)
    return rms_impl(stream, in, kFP8, NPU_F16, gamma, out, kFP8, rows, C, inv_inner_dim,
                    epsilon);
  if (!is_half_float(in_type)) return -1;
  return rms_impl(stream, in, in_type, in_type, gamma, out, in_type, rows, C, inv_inner_dim,
                  epsilon);
}

int npu_rmsnorm_bfp16(void* stream, const void* in, const void* gamma, void* out,
                      int64_t rows, int C, float inv_inner_dim, float epsilon) {
  return rms_impl(stream, in, kBFP16, NPU_BF16, gamma, out, kBFP16, rows, C, inv_inner_dim,
                  epsilon);
}

int npu_l2_norm_bf16(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                     const void* beta, void* out, int64_t rows, int C) {
  const bool fp8 = in_type == NPU_FP8;
  if (!(fp8 || is_half_float(in_type)) || rows < 0 || C <= 0 || C % (fp8 ? 64 : 32) ||
      any_null(in, alpha, beta, out))
    return -1;
  if (rows == 0) return 0;
  NlfTables t;
  if (int rc = nlf_tables(&t)) return rc;
  const npu_dtype alpha_type = fp8 ? NPU_F16 : in_type;
  return launch(stream, rows, l2norm_kernel, src(in, in_type, rows * C), src(alpha, alpha_type, C),
                src(beta, alpha_type, C), dst(out, in_type), C, t.isq);
}

int npu_qdq_l2norm_a8(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                      const void* beta, void* out, npu_dtype out_type, int64_t rows, int C,
                      float dq_zp, float dq_sc, float q_zp, float q_sc) {
  return qdq_l2_impl(false, stream, in, in_type, alpha, beta, out, out_type, rows, C, dq_sc,
                     dq_zp, q_sc, q_zp);
}

int npu_qdq_l2norm_a16(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                       const void* beta, void* out, npu_dtype out_type, int64_t rows, int C,
                       float dq_zp, float dq_sc, float q_zp, float q_sc) {
  return qdq_l2_impl(true, stream, in, in_type, alpha, beta, out, out_type, rows, C, dq_sc,
                     dq_zp, q_sc, q_zp);
}

static int add_ln(void* stream, const void* a, const void* b, npu_dtype in_type,
                  const void* alpha, const void* beta, void* out, int64_t rows, int C, float eps,
                  int stats) {
  if (any_null(a, b)) return -1;
  const int64_t n = rows * C;
  Pre p{};
  p.mode = kPreAdd;
  if (in_type == NPU_FP8) {
    if (C % 8) return -1;
    // fp8: the sum is f16, the result is stored straight to fp8.
    p.a = src(a, kFP8, n);
    p.b = src(b, kFP8, n);
    p.compute = kF16;
    return ln_impl(stream, p, NPU_F16, alpha, beta, dst(out, kFP8), kF32, rows, C, eps, stats);
  }
  if (!is_half_float(in_type)) return -1;
  p.a = src(a, in_type, n);
  p.b = src(b, in_type, n);
  p.compute = in_type;
  return ln_impl(stream, p, in_type, alpha, beta, dst(out, in_type), kF32, rows, C, eps, stats);
}

int npu_add_layernorm_bf16(void* stream, const void* a, const void* b, npu_dtype in_type,
                           const void* alpha, const void* beta, void* out, int64_t rows,
                           int C, float epsilon, npu_ln_stats stats) {
  return add_ln(stream, a, b, in_type, alpha, beta, out, rows, C, epsilon, stats);
}

int npu_add_layernorm_const_bf16(void* stream, const void* a, const void* add_const,
                                 npu_dtype in_type, const void* alpha, const void* beta,
                                 void* out, int64_t rows, int C, float epsilon,
                                 npu_ln_stats stats) {
  return add_ln(stream, a, add_const, in_type, alpha, beta, out, rows, C, epsilon, stats);
}

int npu_qdq_add_layernorm(void* stream, const void* a, const void* b, npu_dtype in_type,
                          const void* alpha, const void* beta, npu_dtype compute_type,
                          void* out, npu_dtype out_type, int64_t rows, int C, float dq_zp,
                          float dq_sc, float q_zp, float q_sc, float dq2_zp, float dq2_sc,
                          float epsilon, npu_ln_stats stats) {
  if (any_null(a, b) || !is_small_int(in_type) || !is_small_int(out_type) ||
      !is_half_float(compute_type))
    return -1;
  Pre p{};
  p.mode = kPreDqAdd;
  p.a = src(a, in_type, rows * C);
  p.b = src(b, in_type, rows * C);
  p.da = dqc(dq_sc, dq_zp);
  p.db = dqc(dq2_sc, dq2_zp);
  p.compute = compute_type;
  return ln_impl(stream, p, compute_type, alpha, beta, qdst(out, out_type, q_sc, q_zp),
                 compute_type, rows, C, epsilon, stats);
}

int npu_qdq_add_layernorm_const(void* stream, const void* a, npu_dtype in_type,
                                const void* add_const, npu_dtype compute_type,
                                const void* alpha, const void* beta, void* out,
                                npu_dtype out_type, int64_t rows, int C, float dq_zp,
                                float dq_sc, float q_zp, float q_sc, float epsilon,
                                npu_ln_stats stats) {
  if (any_null(a, add_const) || !is_small_int(in_type) || !is_small_int(out_type) ||
      !is_half_float(compute_type))
    return -1;
  Pre p{};
  p.mode = kPreDqAddT;
  p.a = src(a, in_type, rows * C);
  p.b = src(add_const, compute_type, rows * C);
  p.da = dqc(dq_sc, dq_zp);
  p.compute = compute_type;
  return ln_impl(stream, p, compute_type, alpha, beta, qdst(out, out_type, q_sc, q_zp),
                 compute_type, rows, C, epsilon, stats);
}

int npu_qdq_add_add_const_layernorm(void* stream, const void* a, npu_dtype a_type,
                                    const void* b, npu_dtype b_type, const void* add_const,
                                    npu_dtype compute_type, const void* alpha,
                                    const void* beta, void* out, npu_dtype out_type,
                                    int64_t rows, int C, float dq_zp, float dq_sc, float q_zp,
                                    float q_sc, float dq2_zp, float dq2_sc, float epsilon,
                                    npu_ln_stats stats) {
  if (any_null(a, b, add_const) || !is_small_int(b_type) || !is_small_int(out_type)) return -1;
  if (!(is_half_float(compute_type) || compute_type == NPU_F32)) return -1;
  // An integer a is read with b's width and signedness: the types must agree.
  if (!(a_type == b_type || a_type == compute_type)) return -1;
  Pre p{};
  p.mode = kPreAddAddC;
  p.a = src(a, a_type, rows * C);
  p.b = src(b, b_type, rows * C);
  p.k = src(add_const, compute_type, C);
  p.da = dqc(dq_sc, dq_zp);
  p.db = dqc(dq2_sc, dq2_zp);
  p.compute = compute_type;
  return ln_impl(stream, p, compute_type, alpha, beta, qdst(out, out_type, q_sc, q_zp),
                 compute_type, rows, C, epsilon, stats);
}

int npu_qdq_layernorm(void* stream, const void* in, npu_dtype in_type, const void* alpha_,
                      const void* beta_, void* out, npu_dtype out_type, int64_t rows, int C,
                      npu_dtype compute_type, float dq_zp, float dq_sc,
                      npu_qln_variant variant) {
  const float dq_scale = dq_sc;
  const auto* alpha = static_cast<const float*>(alpha_);
  const auto* beta = static_cast<const float*>(beta_);
  if (rows < 0 || C <= 0 || any_null(in, alpha, beta, out) || !is_small_int(in_type) ||
      !is_small_int(out_type))
    return -1;
  if (!(is_half_float(compute_type) || compute_type == NPU_F32)) return -1;
  // C8 layouts serve C < 32; the integer statistics C 64 / 96 / 192 or 256..768 by 64.
  if ((variant == NPU_QLN_C8_FUSED || variant == NPU_QLN_C8_BATCHED) && C >= 32) return -1;
  if (variant == NPU_QLN_INTEGER && C != 64 && C != 96 && C != 192 &&
      (C % 64 || C < 256 || C > 768))
    return -1;
  int stats = 0;
  switch (variant) {
    case NPU_QLN_INTEGER:
      if (!is_i8(in_type) || !is_i8(out_type) || compute_type != NPU_F16) return -1;
      break;
    case NPU_QLN_FUSED:
      stats = kFused;
      break;
    case NPU_QLN_BATCHED:
      if (C != 64 && C != 96) return -1;
      stats = kBatchedFast;
      break;
    case NPU_QLN_C8_FUSED:
      if (C % 8) return -1;
      stats = kC8Fused;
      break;
    case NPU_QLN_C8_BATCHED:
      if (C != 8 && C != 16) return -1;
      stats = kC8BatchedFast;
      break;
    default:
      return -1;
  }
  if (rows == 0) return 0;
  NlfTables t;
  if (int rc = nlf_tables(&t)) return rc;
  const Src x = src(in, in_type, rows * C);
  if (variant == NPU_QLN_INTEGER)
    return launch(stream, rows, qln_integer_kernel, x, alpha, beta, dst(out, out_type), C,
                  dq_scale, t);
  Pre p{};
  p.mode = kPreDq;
  p.a = x;
  p.da = dqc(dq_scale, dq_zp);
  p.compute = compute_type;
  const Ln ln{stats, 1.0e-5f, compute_type};
  return launch(stream, rows, layernorm_kernel, p, ln, src(alpha, kF32, C), src(beta, kF32, C),
                dst(out, out_type), kF32, C, t);
}

int npu_layer_norm_a8_bfp16(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                            const void* beta, void* out, int64_t rows, int C, float dq_zp,
                            float dq_sc, npu_qln_variant variant) {
  const float dq_scale = dq_sc;
  if (rows < 0 || C <= 0 || C % 32 || any_null(in, alpha, beta, out) || !is_i8(in_type))
    return -1;
  if (variant != NPU_QLN_FUSED && variant != NPU_QLN_BATCHED) return -1;
  if (variant == NPU_QLN_BATCHED && (C != 64 && C != 96)) return -1;
  if (rows == 0) return 0;
  NlfTables t;
  if (int rc = nlf_tables(&t)) return rc;
  Pre p{};
  p.mode = kPreDq;
  p.a = src(in, in_type, rows * C);
  p.da = dqc(dq_scale, dq_zp);
  p.compute = NPU_F16;
  const Ln ln{variant == NPU_QLN_FUSED ? kFused : kBatchedFast, 1.0e-5f, NPU_F16};
  return launch(stream, rows, layernorm_kernel, p, ln, src(alpha, kF16, C), src(beta, kF16, C),
                dst(out, kBFP16H), kF32, C, t);
}

}  // extern "C"
