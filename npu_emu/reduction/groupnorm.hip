/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Group norm and instance norm: partial sums, per-channel / per-group
 * coefficients, elementwise apply.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "common/act.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/mmac.h"
#include "common/nlf.h"
#include "common/quant.h"
#include "reduction/groupnorm.h"
#include "reduction/groupnorm_device.h"

using namespace npu_emu;

namespace {

constexpr float kInEps = 1e-5f;  // instance-norm epsilon

// Element i as fp32; FP8 is decoded in groups of 8 (-0 -> +0, NaN spreads).
__device__ inline float load_any(const void* p, npu_dtype t, int64_t i, int64_t n) {
  if (t == NPU_FP8)
    return fp8_decode_identity(static_cast<const uint8_t*>(p), i, n);
  return load_float(p, t, i);
}

// Input tensor: dequantized (qdq) or decoded (float).
struct Src {
  const void* x;
  npu_dtype xt;
  int qdq;
  float sc, zpt;  // qdq: x = ct(dequant(q, sc, zpt))
  npu_dtype ct;
  int64_t n;  // elements (fp8 row decode)
  int mmac;   // statistics: x sums in the matrix unit
};

__device__ inline float src_load(const Src& s, int64_t i) {
  if (s.qdq) return round_float(dequant(load_int(s.x, s.xt, i), s.sc, s.zpt), s.ct, kHalfEven);
  return load_any(s.x, s.xt, i, s.n);
}

// Partial sums of one block and channel: two accumulators (even / odd pixels
// in row-major order), acc += x (fp32 add or MMAC) and acc2 = fma(x, x, acc2).
__global__ void in_block_sums_kernel(int64_t n, Src s, int64_t C, Plan p, float* part) {
  NPU_TID;
  const int64_t blk = i / C, ch = i % C;
  const int64_t bh = blk / p.nbw, bw = blk % p.nbw;
  float a[2] = {0.0f, 0.0f}, a2[2] = {0.0f, 0.0f};
  const int64_t cnt = p.sv_h * p.sv_w;
  for (int64_t k = 0; k < cnt; ++k) {
    const int64_t r = (bh * p.sv_h + k / p.sv_w) * p.W + bw * p.sv_w + k % p.sv_w;
    const float x = src_load(s, r * C + ch);
    a[k & 1] = s.mmac ? mmac_add_bf16(a[k & 1], x) : add_q(a[k & 1], x);
    a2[k & 1] = fma_q(x, x, a2[k & 1]);
  }
  float* o = part + i * 4;
  o[0] = a[0], o[1] = a[1], o[2] = a2[0], o[3] = a2[1];
}

// Where the per-channel statistics come from.
struct Stats {
  const float* part;  // block sums (Plan p), or null:
  const float* sum;   // given statistics
  const float* sumsq;
};

// Sum of cell (c, k): (s + acc_even) + acc_odd per block, over its steps.
__device__ inline void in_tile(const Plan& p, const float* part, int64_t C, int64_t ch, int c,
                               int k, float* t, float* t2) {
  float s = 0.0f, s2 = 0.0f;
  for (int j = 0; j < p.T; ++j) {
    const int64_t bh = c * p.cbh + k * p.kbh + (j / p.w_loop) * p.jbh;
    const int64_t bw = c * p.cbw + k * p.kbw + j % p.w_loop;
    const float* q = part + ((bh * p.nbw + bw) * C + ch) * 4;
    s = add_q(add_q(s, q[0]), q[1]);
    s2 = add_q(add_q(s2, q[2]), q[3]);
  }
  *t = s, *t2 = s2;
}

// The add_mul_add operands of channel ch.
struct Amad {
  const void* act;
  const void* add2;
  npu_dtype at;  // act / add2 type
  int qdq;
  float sc, zpt;  // qdq: dequantize to ct
  npu_dtype ct;
  const void* cst;
  npu_dtype cst_t;
  int64_t c;
};

__device__ inline float amad_load(const Amad& m, const void* p, int64_t ch) {
  if (m.qdq) return round_float(dequant(load_int(p, m.at, ch), m.sc, m.zpt), m.ct, kHalfEven);
  return load_any(p, m.at, ch, m.c);
}

// Statistics -> folded per-channel coefficients: AB[ch] = Alpha, AB[C + ch] = Beta.
__global__ void in_coeff_kernel(int64_t n, Plan p, Stats st, int64_t N, const void* alpha,
                                const void* beta, npu_dtype abt, int use_amad, Amad am,
                                const uint32_t* inv_exc, const uint32_t* isq_exc, float* AB) {
  NPU_TID;
  const int64_t C = n, ch = i;
  float S, S2;
  if (st.part) {
    float tot = 0.0f, tot2 = 0.0f;
    for (int c = 0; c < p.ncol; ++c) {
      float r, r2;
      in_tile(p, st.part, C, ch, c, p.nrow - 1, &r, &r2);
      for (int k = p.nrow - 2; k >= 0; --k) {
        float t, t2;
        in_tile(p, st.part, C, ch, c, k, &t, &t2);
        r = add_q(t, r);
        r2 = add_q(t2, r2);
      }
      if (c == 0)
        tot = r, tot2 = r2;
      else
        tot = add_q(r, tot), tot2 = add_q(r2, tot2);
    }
    S = tot, S2 = tot2;
  } else {
    S = st.sum[ch], S2 = st.sumsq[ch];
  }
  const float inv_s = hw_inv_f32(float(N), inv_exc);
  const float mu = mul_q(S, inv_s);
  const float ex2 = fma_q(S2, inv_s, kInEps);
  const float var = msc_q(mu, mu, ex2);
  const float inv_std = hw_invsqrt_f32(var, isq_exc);
  const float m = negmul_q(mu, inv_std);
  const float a = load_float(alpha, abt, ch), b = load_float(beta, abt, ch);
  float A = mul_q(inv_std, a), B = fma_q(m, a, b);
  if (use_amad) {
    const float add1 = add_q(amad_load(am, am.act, ch), load_float(am.cst, am.cst_t, ch));
    B = fma_q(B, add1, amad_load(am, am.add2, ch));
    A = mul_q(A, add1);
  }
  AB[ch] = A;
  AB[C + ch] = B;
}

// Activation in type t: 1 ReLU, 3 leaky max(y, t(y * slope_t)).
__device__ inline float act_typed(float y, int act, float slope_t, npu_dtype t) {
  if (act == 1) return vmax(y, 0.0f);
  if (act == 3) return vmax(y, round_float(mul_q(y, slope_t), t, kHalfEven));
  return y;
}

struct Out {
  void* o;
  void* o2;  // double_out: the activated output
  npu_dtype ot;
  int qdq;
  int act;
  float slope;  // qdq / 16-bit: rounded to the compute / output type
  float q_sc, q_zp, q2_sc, q2_zp;
};

__global__ void in_apply_kernel(int64_t n, Src s, int64_t C, const float* AB, Out o) {
  NPU_TID;
  const int64_t ch = i % C;
  const float y = fma_q(src_load(s, i), AB[ch], AB[C + ch]);
  if (o.qdq) {
    const float yt = round_float(y, s.ct, kHalfEven);
    if (o.o2) {
      store_int(o.o, o.ot, i, quant(yt, o.q_sc, o.q_zp, o.ot));
      const float ya = act_typed(yt, o.act, o.slope, s.ct);
      store_int(o.o2, o.ot, i, quant(ya, o.q2_sc, o.q2_zp, o.ot));
    } else {
      const float ya = act_typed(yt, o.act, o.slope, s.ct);
      store_int(o.o, o.ot, i, quant(ya, o.q_sc, o.q_zp, o.ot));
    }
    return;
  }
  if (o.ot == NPU_FP8) {
    // fp8: activated in fp32, narrowed once.
    float v = y;
    if (o.act == 1) v = vmax(v, 0.0f);
    if (o.act == 3) v = vmax(v, mul_q(v, o.slope));
    store_float(o.o, NPU_FP8, i, v, kHalfEven);
    return;
  }
  // Narrowed, then activated in the output type.
  const float v = act_typed(round_float(y, o.ot, kHalfEven), o.act, o.slope, o.ot);
  store_float(o.o, o.ot, i, v, kHalfEven);
}

bool is_io_float(npu_dtype t) { return t == NPU_BF16 || t == NPU_F16 || t == NPU_FP8; }
bool valid_act(int a) { return a == 0 || a == 1 || a == 3; }

// Plan for a linear spatial axis cut into chunks of sv_hw.
bool in_sd_plan(int split_mode, int64_t sv_hw, int64_t hw, Plan* p) {
  if (sv_hw <= 0 || hw <= 0 || hw % sv_hw) return false;
  const int64_t nq = hw / sv_hw;
  *p = Plan{};
  p->w_loop = 1;
  p->nbw = 1, p->nbh = nq, p->sv_h = 1, p->sv_w = sv_hw, p->W = sv_hw;
  switch (split_mode) {
    case 0:
      if (nq % 12) return false;
      p->ncol = 3, p->nrow = 4, p->T = int(nq / 12), p->cbh = nq / 3, p->kbh = 1, p->jbh = 4;
      return true;
    case 1:
      if (nq % 4) return false;
      p->ncol = 1, p->nrow = 4, p->T = int(nq / 4), p->kbh = 1, p->jbh = 4;
      return true;
    case 2:
      if (nq % 2) return false;
      p->ncol = 1, p->nrow = 2, p->T = int(nq / 2), p->kbh = 1, p->jbh = 2;
      return true;
    case 3:
      if (nq % 3) return false;
      p->ncol = 3, p->nrow = 1, p->T = int(nq / 3), p->cbh = nq / 3, p->jbh = 1;
      return true;
    case -1:
      p->ncol = 1, p->nrow = 1, p->T = int(nq), p->jbh = 1;
      return true;
    default:
      return false;
  }
}

// Instance norm: statistics (plan, or given sum / sumsq), coefficients, apply.
int in_run(void* stream, const Src& s, int64_t hw, int64_t C, const Plan* plan,
           const float* sum, const float* sumsq, int64_t N, const void* alpha,
           const void* beta, npu_dtype alpha_type, const Amad* am, const Out& o) {
  // hw is the reduction length of a plan; with given statistics it only counts output pixels.
  if (hw < (plan ? 1 : 0) || C < 0 || N <= 0 || any_null(s.x, o.o, alpha, beta)) return -1;
  if (am && any_null(am->act, am->add2, am->cst)) return -1;
  if (hw == 0 || C == 0) return 0;
  NlfTables nt;
  if (int rc = nlf_tables(&nt)) return rc;
  hipStream_t st = static_cast<hipStream_t>(stream);
  const int64_t nblk = plan ? plan->nbh * plan->nbw : 0;
  const size_t bytes = size_t(nblk * C * 4 + 2 * C) * sizeof(float);
  float* work = nullptr;
  if (int rc = int(hipMallocAsync(reinterpret_cast<void**>(&work), bytes, st))) return rc;
  float* AB = work + nblk * C * 4;
  int rc = 0;
  Stats stats{nullptr, sum, sumsq};
  Plan p = plan ? *plan : Plan{};
  if (plan) {
    rc = launch(stream, nblk * C, in_block_sums_kernel, s, C, p, work);
    stats.part = work;
  }
  Amad amad = am ? *am : Amad{};
  if (!rc)
    rc = launch(stream, C, in_coeff_kernel, p, stats, N, alpha, beta, alpha_type, am ? 1 : 0,
                amad, nt.inv, nt.isq, AB);
  if (!rc) rc = launch(stream, hw * C, in_apply_kernel, s, C, static_cast<const float*>(AB), o);
  const int rc2 = int(hipFreeAsync(work, st));
  return rc ? rc : rc2;
}

// The bf16 / f16 / fp8 instance norms: split_mode 0 / 1 / 2; alpha / beta (and add1_const)
// are ifm_type, F16 for FP8.
int in_bf16(void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
            const void* beta, const void* add1_act, const void* add2, const void* add1_const,
            bool amad, void* ofm, int64_t hw, int64_t c, int64_t sv_hw, int split_mode,
            int64_t N, int act, float slope) {
  if (!is_io_float(ifm_type) || !valid_act(act) || N <= 0 || N > hw || split_mode < 0 ||
      split_mode > 2)
    return -1;
  const npu_dtype alpha_type = ifm_type == NPU_FP8 ? NPU_F16 : ifm_type;
  Plan p;
  if (!in_sd_plan(split_mode, sv_hw, hw, &p)) return -1;
  const Src s{ifm, ifm_type, 0, 0.0f, 0.0f, NPU_F32, hw * c, ifm_type == NPU_FP8};
  Out o{};
  o.o = ofm, o.ot = ifm_type, o.act = act;
  o.slope = ifm_type == NPU_FP8 ? slope : scalar_cast(slope, ifm_type, kHalfEven);
  Amad am{add1_act, add2, ifm_type, 0, 0.0f, 0.0f, NPU_F32, add1_const, alpha_type, c};
  return in_run(stream, s, hw, c, &p, nullptr, nullptr, N, alpha, beta, alpha_type,
                amad ? &am : nullptr, o);
}

}  // namespace

namespace npu_emu {

int in_qdq(void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
           const void* beta, npu_dtype alpha_type, const void* add1_act, const void* add2,
           const void* add1_const, bool amad, void* ofm, void* ofm2, int64_t hw, int64_t c,
           npu_dtype compute_type, const Plan* plan, const float* sum, const float* sumsq,
           int64_t N, float dq_sc, float dq_zp, float q_sc, float q_zp, float q2_sc, float q2_zp,
           float dq2_sc, float dq2_zp, int act, float slope, bool sum_on_mmac) {
  if (!is_small_int(ifm_type) || !is_plain_float(compute_type) || !is_plain_float(alpha_type) ||
      !valid_act(act))
    return -1;
  const Src s{ifm, ifm_type, 1, dq_sc, dq_offset(dq_zp, dq_sc), compute_type, hw * c,
              sum_on_mmac};
  Out o{};
  o.o = ofm, o.o2 = ofm2, o.ot = ifm_type, o.qdq = 1, o.act = act;
  o.slope = compute_type == NPU_F32 ? slope : scalar_cast(slope, compute_type, kHalfEven);
  o.q_sc = q_sc, o.q_zp = q_zp, o.q2_sc = q2_sc, o.q2_zp = q2_zp;
  Amad am{add1_act, add2, ifm_type, 1, dq2_sc, dq_offset(dq2_zp, dq2_sc),
          compute_type, add1_const, compute_type, c};
  return in_run(stream, s, hw, c, plan, sum, sumsq, N, alpha, beta, alpha_type,
                amad ? &am : nullptr, o);
}

}  // namespace npu_emu

namespace {

// The quantized instance norms; alpha / beta (and add1_const) are compute_type. split_mode
// -1 / 3 only when all_splits.
int in_qdq_sd(void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
              const void* beta, npu_dtype compute_type, const void* add1_act, const void* add2,
              const void* add1_const, bool amad, void* ofm, void* ofm2, int64_t hw, int64_t c,
              int64_t sv_hw, int split_mode, bool all_splits, int64_t N, float dq_zp, float dq_sc,
              float q_zp, float q_sc, float q2_zp, float q2_sc, float dq2_zp, float dq2_sc, int act,
              float slope) {
  Plan p;
  if ((!all_splits && (split_mode < 0 || split_mode > 2)) ||
      !in_sd_plan(split_mode, sv_hw, hw, &p) || N <= 0 || N > hw)
    return -1;
  return in_qdq(stream, ifm, ifm_type, alpha, beta, compute_type, add1_act, add2, add1_const,
                amad, ofm, ofm2, hw, c, compute_type, &p, nullptr, nullptr, N, dq_sc, dq_zp, q_sc,
                q_zp, q2_sc, q2_zp, dq2_sc, dq2_zp, act, slope, compute_type == NPU_BF16);
}

// Per channel: even / odd row accumulators, then (inv_std, -mu*inv_std).
__global__ void stream_stats_kernel(int64_t n, Src s, int64_t hw, const uint32_t* inv_exc,
                                    const uint32_t* isq_exc, float* st) {
  NPU_TID;
  const int64_t C = n;
  float a[2] = {0.0f, 0.0f}, a2[2] = {0.0f, 0.0f};
  for (int64_t r = 0; r + 1 < hw; r += 2) {
    for (int h = 0; h < 2; ++h) {
      const float x = src_load(s, (r + h) * C + i);
      a[h] = add_q(a[h], x);
      a2[h] = fma_q(x, x, a2[h]);
    }
  }
  const float S = add_q(a[0], a[1]), S2 = add_q(a2[0], a2[1]);
  const float inv_s = hw_inv_f32(float(hw), inv_exc);
  const float mu = mul_q(S, inv_s);
  const float var = msc_q(mu, mu, fma_q(S2, inv_s, kInEps));
  const float inv_std = hw_invsqrt_f32(var, isq_exc);
  st[i] = inv_std;
  st[C + i] = negmul_q(mu, inv_std);
}

__global__ void stream_apply_kernel(int64_t n, Src s, int64_t C, const float* st,
                                    const void* alpha, const void* beta, npu_dtype abt, void* o,
                                    npu_dtype ot, float relu_alpha, float q_sc, float q_zp) {
  NPU_TID;
  const int64_t ch = i % C;
  const float t0 = fma_q(src_load(s, i), round_float(st[ch], NPU_BF16, kHalfEven), st[C + ch]);
  const float t1 = fma_q(t0, load_float(alpha, abt, ch), load_float(beta, abt, ch));
  if (s.qdq) {
    const float a = vmax(t1, mul_q(t1, relu_alpha));
    store_int(o, ot, i, quant(a, q_sc, q_zp, ot));
  } else {
    static_cast<uint16_t*>(o)[i] = f32_to_bf16(t1, kHalfEven);
  }
}

int in_stream(void* stream, const Src& s, const void* alpha, const void* beta,
              npu_dtype alpha_type, void* ofm, npu_dtype ofm_type, int64_t hw, int64_t c,
              float relu_alpha, float q_sc, float q_zp) {
  if (hw <= 0 || hw % 2 || c < 0 || !(alpha_type == NPU_BF16 || alpha_type == NPU_F32) ||
      any_null(s.x, ofm, alpha, beta))
    return -1;
  if (c == 0) return 0;
  NlfTables nt;
  if (int rc = nlf_tables(&nt)) return rc;
  hipStream_t st = static_cast<hipStream_t>(stream);
  float* work = nullptr;
  if (int rc = int(hipMallocAsync(reinterpret_cast<void**>(&work), size_t(2 * c) * 4, st)))
    return rc;
  int rc = launch(stream, c, stream_stats_kernel, s, hw, nt.inv, nt.isq, work);
  if (!rc)
    rc = launch(stream, hw * c, stream_apply_kernel, s, c, static_cast<const float*>(work),
                alpha, beta, alpha_type, ofm, ofm_type, relu_alpha, q_sc, q_zp);
  const int rc2 = int(hipFreeAsync(work, st));
  return rc ? rc : rc2;
}

// One chunk of sv elements of a group: two 32-lane accumulators (first / second
// half of the chunk), lanes joined, then one vector reduce-add each.
__global__ void gn_chunk_sums_kernel(int64_t n, Src s, int64_t l, int64_t sv, float* part) {
  NPU_TID;
  const int64_t nq = l / sv;
  const int64_t g = i / nq, q = i % nq;
  const int64_t base = g * l + q * sv, il = sv / 64;
  float acc[2][32], acc2[2][32];
  for (int h = 0; h < 2; ++h)
    for (int k = 0; k < 32; ++k) acc[h][k] = 0.0f, acc2[h][k] = 0.0f;
  for (int64_t r = 0; r < il; ++r)
    for (int h = 0; h < 2; ++h)
      for (int k = 0; k < 32; ++k) {
        const float x = src_load(s, base + h * (sv / 2) + r * 32 + k);
        acc[h][k] = add_q(acc[h][k], x);
        acc2[h][k] = fma_q(x, x, acc2[h][k]);
      }
  float v[32], v2[32];
  for (int k = 0; k < 32; ++k) {
    v[k] = add_q(acc[0][k], acc[1][k]);
    v2[k] = add_q(acc2[0][k], acc2[1][k]);
  }
  part[i * 2] = vaddr_f32_reduce(v, 32);
  part[i * 2 + 1] = vaddr_f32_reduce(v2, 32);
}

enum GnKind { kGnBf16Sd = 0, kGnQdqSd = 1, kGnQdq = 2 };

// Group statistics in the order given by kind, then (inv_std, -mu*inv_std).
__global__ void gn_coeff_kernel(int64_t n, const float* part, int64_t nq, int kind, int ncol,
                                int nrow, int64_t N, float eps, const uint32_t* inv_exc,
                                const uint32_t* isq_exc, float* st) {
  NPU_TID;
  const float* S = part + i * nq * 2;  // S[2q], S2 = S[2q + 1]
  float s = 0.0f, s2 = 0.0f;
  if (kind == kGnQdq) {
    for (int64_t q = 0; q < nq; ++q) s = add_q(S[2 * q], s), s2 = add_q(S[2 * q + 1], s2);
  } else if (kind == kGnBf16Sd) {
    // Per step: rows last to first, columns chained, then added to the running sum.
    const int64_t T = nq / (nrow * ncol), per = nq / ncol;
    for (int64_t j = 0; j < T; ++j) {
      float t = 0.0f, t2 = 0.0f;
      for (int c = 0; c < ncol; ++c) {
        const int64_t q0 = c * per + nrow * j;
        float r = S[2 * (q0 + nrow - 1)], r2 = S[2 * (q0 + nrow - 1) + 1];
        for (int k = nrow - 2; k >= 0; --k)
          r = add_q(S[2 * (q0 + k)], r), r2 = add_q(S[2 * (q0 + k) + 1], r2);
        if (c == 0)
          t = r, t2 = r2;
        else
          t = add_q(r, t), t2 = add_q(r2, t2);
      }
      s = add_q(s, t), s2 = add_q(s2, t2);
    }
  } else {
    // Each cell sums its chunks over the steps; then rows last to first, columns chained.
    const int64_t per = nq / ncol, T = per / nrow;
    for (int c = 0; c < ncol; ++c) {
      float r = 0.0f, r2 = 0.0f;
      for (int k = nrow - 1; k >= 0; --k) {
        float p = 0.0f, p2 = 0.0f;
        for (int64_t j = 0; j < T; ++j) {
          const int64_t q = c * per + j * nrow + k;
          p = add_q(p, S[2 * q]), p2 = add_q(p2, S[2 * q + 1]);
        }
        if (k == nrow - 1)
          r = p, r2 = p2;
        else
          r = add_q(p, r), r2 = add_q(p2, r2);
      }
      if (c == 0)
        s = r, s2 = r2;
      else
        s = add_q(r, s), s2 = add_q(r2, s2);
    }
  }
  const float inv_s = hw_inv_f32(float(N), inv_exc);
  const float mu = mul_q(s, inv_s);
  const float var = msc_q(mu, mu, fma_q(s2, inv_s, eps));
  const float inv_std = hw_invsqrt_f32(var, isq_exc);
  st[i * 2] = inv_std;
  st[i * 2 + 1] = negmul_q(mu, inv_std);
}

struct GnOut {
  void* o;
  npu_dtype ot;
  int kind, silu;
  float q_sc, q_zp;
};

__global__ void gn_apply_kernel(int64_t n, Src s, int64_t l, int64_t cg, const float* st,
                                const void* alpha, const void* beta, npu_dtype abt, GnOut o) {
  NPU_TID;
  const int64_t g = i / l, ch = g * cg + (i % l) % cg;
  const float x = src_load(s, i);
  const float inv_std = st[g * 2], m = st[g * 2 + 1];
  const float a = load_float(alpha, abt, ch), b = load_float(beta, abt, ch);
  if (o.kind == kGnQdqSd) {
    const float y = round_float(fma_q(x, mul_q(inv_std, a), fma_q(m, a, b)), s.ct, kHalfEven);
    const float v = o.silu ? round_float(hw_silu_f32(y), s.ct, kHalfEven, kFtzIn) : y;
    store_int(o.o, o.ot, i, quant(v, o.q_sc, o.q_zp, o.ot));
    return;
  }
  if (o.kind == kGnQdq) {
    const float t0 = fma_q(x, round_float(inv_std, NPU_BF16, kHalfEven), m);
    const float y = round_float(fma_q(t0, a, b), NPU_BF16, kHalfEven);
    store_int(o.o, o.ot, i, quant(y, o.q_sc, o.q_zp, o.ot));
    return;
  }
  // inv_std in the input type (fp32 for fp8).
  const float t0 = fma_q(x, o.ot == NPU_FP8 ? inv_std : round_float(inv_std, o.ot, kHalfEven), m);
  const float y = fma_q(t0, a, b);
  if (o.ot == NPU_FP8) {
    // fp8: SiLU on the fp32 value, narrowed once.
    const float v = o.silu ? ftz(hw_silu_f32(y)) : y;
    store_float(o.o, NPU_FP8, i, v, o.silu ? kHalfAway : kHalfEven);
    return;
  }
  float v = round_float(y, o.ot, kHalfEven);
  if (o.silu) v = round_float(hw_silu_f32(v), o.ot, kHalfAway, kFtzIn);
  store_float(o.o, o.ot, i, v, kHalfEven);
}

int gn_run(void* stream, const Src& s, int64_t g, int64_t l, int64_t cg, int64_t sv, int kind,
           int ncol, int nrow, int64_t N, float eps, const void* alpha, const void* beta,
           npu_dtype alpha_type, const GnOut& o) {
  if (g < 0 || l <= 0 || cg <= 0 || l % cg || sv <= 0 || sv % 64 || l % sv || N <= 0 ||
      N > l || !is_plain_float(alpha_type) || any_null(s.x, o.o, alpha, beta))
    return -1;
  const int64_t nq = l / sv;
  if (nq % (int64_t(ncol) * nrow)) return -1;
  if (g == 0) return 0;
  NlfTables nt;
  if (int rc = nlf_tables(&nt)) return rc;
  hipStream_t st = static_cast<hipStream_t>(stream);
  float* work = nullptr;
  const size_t bytes = size_t(g * nq * 2 + g * 2) * sizeof(float);
  if (int rc = int(hipMallocAsync(reinterpret_cast<void**>(&work), bytes, st))) return rc;
  float* coef = work + g * nq * 2;
  int rc = launch(stream, g * nq, gn_chunk_sums_kernel, s, l, sv, work);
  if (!rc)
    rc = launch(stream, g, gn_coeff_kernel, static_cast<const float*>(work), nq, kind, ncol,
                nrow, N, eps, nt.inv, nt.isq, coef);
  if (!rc)
    rc = launch(stream, g * l, gn_apply_kernel, s, l, cg, static_cast<const float*>(coef), alpha,
                beta, alpha_type, o);
  const int rc2 = int(hipFreeAsync(work, st));
  return rc ? rc : rc2;
}

}  // namespace

extern "C" {

int npu_instance_norm_sd_bf16(void* stream, const void* ifm, npu_dtype ifm_type,
                              const void* alpha, const void* beta, void* ofm, int64_t hw,
                              int64_t c, int64_t sv_hw, int split_mode, int64_t spatial_size,
                              int activation, float leakyRelu_alpha) {
  return in_bf16(stream, ifm, ifm_type, alpha, beta, nullptr, nullptr, nullptr, false, ofm, hw, c,
                 sv_hw, split_mode, spatial_size, activation, leakyRelu_alpha);
}

int npu_instance_norm_sd_opt_stage2_ifm_bf16(void* stream, const void* ifm, npu_dtype ifm_type,
                                             const void* alpha, const void* beta, void* ofm,
                                             int64_t hw, int64_t c, int64_t sv_hw,
                                             int split_mode, int64_t spatial_size,
                                             int activation, float leakyRelu_alpha) {
  return in_bf16(stream, ifm, ifm_type, alpha, beta, nullptr, nullptr, nullptr, false, ofm, hw, c,
                 sv_hw, split_mode, spatial_size, activation, leakyRelu_alpha);
}

int npu_instance_norm_add_mul_add_sd_bf16(void* stream, const void* ifm, npu_dtype ifm_type,
                                          const void* alpha, const void* beta,
                                          const void* add1_act, const void* add2,
                                          const void* add1_const, void* ofm, int64_t hw,
                                          int64_t c, int64_t sv_hw, int split_mode,
                                          int64_t spatial_size, int activation,
                                          float leakyRelu_alpha) {
  return in_bf16(stream, ifm, ifm_type, alpha, beta, add1_act, add2, add1_const, true, ofm, hw, c,
                 sv_hw, split_mode, spatial_size, activation, leakyRelu_alpha);
}

int npu_instance_norm_add_mul_add_sd_opt_stage2_ifm_bf16(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha, const void* beta,
    const void* add1_act, const void* add2, const void* add1_const, void* ofm, int64_t hw,
    int64_t c, int64_t sv_hw, int split_mode, int64_t spatial_size, int activation,
    float leakyRelu_alpha) {
  return in_bf16(stream, ifm, ifm_type, alpha, beta, add1_act, add2, add1_const, true, ofm, hw, c,
                 sv_hw, split_mode, spatial_size, activation, leakyRelu_alpha);
}

int npu_qdq_instance_norm_sd(void* stream, const void* ifm, npu_dtype ifm_type,
                             const void* alpha, const void* beta, npu_dtype compute_type,
                             void* ofm, int64_t hw, int64_t c, int64_t sv_hw, int split_mode,
                             int64_t spatial_size, float dq_zp, float dq_sc, float q_zp,
                             float q_sc, int activation, float leakyRelu_alpha) {
  return in_qdq_sd(stream, ifm, ifm_type, alpha, beta, compute_type, nullptr, nullptr, nullptr,
                   false, ofm, nullptr, hw, c, sv_hw, split_mode, true, spatial_size, dq_zp,
                   dq_sc, q_zp, q_sc, 0.0f, 0.0f, 0.0f, 0.0f, activation, leakyRelu_alpha);
}

int npu_qdq_instance_norm_sd_opt_stage2_ifm(void* stream, const void* ifm, npu_dtype ifm_type,
                                            const void* alpha, const void* beta,
                                            npu_dtype compute_type, void* ofm, int64_t hw,
                                            int64_t c, int64_t sv_hw, int split_mode,
                                            int64_t spatial_size, float dq_zp, float dq_sc,
                                            float q_zp, float q_sc, int activation,
                                            float leakyRelu_alpha) {
  return in_qdq_sd(stream, ifm, ifm_type, alpha, beta, compute_type, nullptr, nullptr, nullptr,
                   false, ofm, nullptr, hw, c, sv_hw, split_mode, false, spatial_size, dq_zp,
                   dq_sc, q_zp, q_sc, 0.0f, 0.0f, 0.0f, 0.0f, activation, leakyRelu_alpha);
}

int npu_qdq_instance_norm_sd_opt_stage2_ifm_double_out(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha, const void* beta,
    npu_dtype compute_type, void* ofm, void* ofm2, int64_t hw, int64_t c, int64_t sv_hw,
    int split_mode, int64_t spatial_size, float dq_zp, float dq_sc, float q_zp, float q_sc,
    float q2_zp, float q2_sc, int activation, float leakyRelu_alpha) {
  if (!ofm2) return -1;
  return in_qdq_sd(stream, ifm, ifm_type, alpha, beta, compute_type, nullptr, nullptr, nullptr,
                   false, ofm, ofm2, hw, c, sv_hw, split_mode, false, spatial_size, dq_zp, dq_sc,
                   q_zp, q_sc, q2_zp, q2_sc, 0.0f, 0.0f, activation, leakyRelu_alpha);
}

int npu_qdq_instance_norm_add_mul_add_sd(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha, const void* beta,
    npu_dtype compute_type, const void* add1_act, const void* add2, const void* add1_const,
    void* ofm, int64_t hw, int64_t c, int64_t sv_hw, int split_mode, int64_t spatial_size,
    float dq_zp, float dq_sc, float q_zp, float q_sc, float dq2_zp, float dq2_sc,
    int activation, float leakyRelu_alpha) {
  return in_qdq_sd(stream, ifm, ifm_type, alpha, beta, compute_type, add1_act, add2, add1_const,
                   true, ofm, nullptr, hw, c, sv_hw, split_mode, false, spatial_size, dq_zp,
                   dq_sc, q_zp, q_sc, 0.0f, 0.0f, dq2_zp, dq2_sc, activation, leakyRelu_alpha);
}

int npu_qdq_instance_norm_add_mul_add_sd_opt_stage2_ifm(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha, const void* beta,
    npu_dtype compute_type, const void* add1_act, const void* add2, const void* add1_const,
    void* ofm, int64_t hw, int64_t c, int64_t sv_hw, int split_mode, int64_t spatial_size,
    float dq_zp, float dq_sc, float q_zp, float q_sc, float dq2_zp, float dq2_sc,
    int activation, float leakyRelu_alpha) {
  return in_qdq_sd(stream, ifm, ifm_type, alpha, beta, compute_type, add1_act, add2, add1_const,
                   true, ofm, nullptr, hw, c, sv_hw, split_mode, false, spatial_size, dq_zp,
                   dq_sc, q_zp, q_sc, 0.0f, 0.0f, dq2_zp, dq2_sc, activation, leakyRelu_alpha);
}

int npu_instance_norm_stream_bf16(void* stream, const void* ifm, const void* alpha,
                                  const void* beta, npu_dtype alpha_type, void* ofm, int64_t hw,
                                  int64_t c) {
  const Src s{ifm, NPU_BF16, 0, 0.0f, 0.0f, NPU_BF16, hw * c};
  return in_stream(stream, s, alpha, beta, alpha_type, ofm, NPU_BF16, hw, c, 0.0f, 0.0f, 0.0f);
}

int npu_qdq_instance_norm_stream(void* stream, const void* ifm, npu_dtype ifm_type,
                                 const void* alpha, const void* beta, npu_dtype alpha_type,
                                 void* ofm, int64_t hw, int64_t c, float dq_sc, float q_sc,
                                 float dq_offset, float q_zp, float relu_alpha) {
  if (!is_small_int(ifm_type)) return -1;
  const Src s{ifm, ifm_type, 1, dq_sc, dq_offset, NPU_BF16, hw * c};
  return in_stream(stream, s, alpha, beta, alpha_type, ofm, ifm_type, hw, c, relu_alpha, q_sc,
                   q_zp);
}

int npu_groupnorm_sd_bf16(void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
                          const void* beta, void* ofm, int64_t g, int64_t l, int64_t cg,
                          int64_t sv, int64_t div_factor, int split_mode, float epsilon,
                          int activation) {
  if (!is_io_float(ifm_type) || (activation != 0 && activation != 5) || split_mode < 0 ||
      split_mode > 2)
    return -1;
  const Src s{ifm, ifm_type, 0, 0.0f, 0.0f, NPU_F32, g * l};
  const GnOut o{ofm, ifm_type, kGnBf16Sd, activation == 5, 0.0f, 0.0f};
  return gn_run(stream, s, g, l, cg, sv, kGnBf16Sd, split_mode == 0 ? 3 : 1,
                split_mode == 2 ? 2 : 4, div_factor, epsilon, alpha, beta,
                ifm_type == NPU_FP8 ? NPU_F16 : ifm_type, o);
}

int npu_qdq_groupnorm_sd(void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
                         const void* beta, npu_dtype compute_type, void* ofm, int64_t g,
                         int64_t l, int64_t cg, int64_t sv, int64_t div_factor, int split_mode,
                         float epsilon, float dq_zp, float dq_sc, float q_zp, float q_sc,
                         int activation) {
  if (!is_small_int(ifm_type) || !is_plain_float(compute_type) ||
      (activation != 0 && activation != 5) || split_mode < -1 || split_mode > 3)
    return -1;
  const int ncol = (split_mode == 0 || split_mode == 3) ? 3 : 1;
  const int nrow = split_mode == 0 || split_mode == 1 ? 4 : (split_mode == 2 ? 2 : 1);
  const Src s{ifm, ifm_type, 1, dq_sc, dq_offset(dq_zp, dq_sc), compute_type, g * l};
  const GnOut o{ofm, ifm_type, kGnQdqSd, activation == 5, q_sc, q_zp};
  return gn_run(stream, s, g, l, cg, sv, kGnQdqSd, ncol, nrow, div_factor, epsilon, alpha, beta,
                compute_type, o);
}

int npu_qdq_groupnorm(void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
                      const void* beta, void* ofm, int64_t g, int64_t l, int64_t cg, int64_t sv,
                      float epsilon, float dq_zp, float dq_sc, float q_zp, float q_sc) {
  if (!is_small_int(ifm_type)) return -1;
  const Src s{ifm, ifm_type, 1, dq_sc, dq_offset(dq_zp, dq_sc), NPU_BF16, g * l};
  const GnOut o{ofm, ifm_type, kGnQdq, 0, q_sc, q_zp};
  return gn_run(stream, s, g, l, cg, sv, kGnQdq, 1, 1, l, epsilon, alpha, beta, NPU_BF16, o);
}

int npu_qdq_ConvInstanceNormAddMulAdd_Stage2(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* sum, const void* sumsq,
    const void* alpha, const void* beta, npu_dtype compute_type, const void* add1_act,
    const void* add2, const void* add1_const, void* ofm, int64_t hw, int64_t c,
    int64_t spatial_size, float dq_zp, float dq_sc, float q_zp, float q_sc, float dq2_zp,
    float dq2_sc, int activation, float leakyRelu_alpha) {
  if (any_null(sum, sumsq) || !is_i16(ifm_type) ||
      (compute_type != NPU_BF16 && compute_type != NPU_F16))
    return -1;
  return in_qdq(stream, ifm, ifm_type, alpha, beta, compute_type, add1_act, add2, add1_const,
                true, ofm, nullptr, hw, c, compute_type, nullptr,
                static_cast<const float*>(sum), static_cast<const float*>(sumsq), spatial_size,
                dq_sc, dq_zp, q_sc, q_zp, 0.0f, 0.0f, dq2_sc, dq2_zp, activation,
                leakyRelu_alpha);
}

}  // extern "C"
