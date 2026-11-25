/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for matmul/qdq_gemm.h. One thread per output element.
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
#include "matmul/qdq_gemm.h"

using namespace npu_emu;

namespace {

// How the A operand is split and how its row sum enters the float chain.
enum GemmKind : int {
  kGemmI8 = 0,          // one int32 product
  kGemmI16Join = 1,     // lo / hi products, S from the byte-plane sums
  kGemmI16Exact = 2,    // lo / hi products, S = fp32(exact row sum)
  kGemmI16Product = 3,  // products joined to fp32 before the c2 MAC
};

// fp32(fp32(sum lo) + 256 * sum hi), or the exact sum rounded once.
__device__ inline float row_sum_f32(int64_t lo_sum, int64_t hi_sum, bool join) {
  return join ? mac_exact(hi_sum, 256.0f, int_f32(lo_sum)) : int_f32(lo_sum + 256 * hi_sum);
}

// The low nibble of an int4 weight byte: signed for I8, unsigned for U8.
__device__ inline int64_t int4_value(int64_t y, npu_dtype b_type) {
  return b_type == NPU_I8 ? ((y & 15) ^ 8) - 8 : y & 15;
}

// fp32 accumulator of output (row, col), before the activation; c1 / c3 may be null.
__device__ inline float gemm_qdq_acc(GemmKind kind, const void* a, npu_dtype a_type, const void* b,
                                     npu_dtype b_type, int64_t k, int64_t n, int64_t row,
                                     int64_t col, float c0, const float* c1, float c2,
                                     const float* c3, bool w4 = false) {
  int64_t g_lo = 0, g_hi = 0, s_lo = 0, s_hi = 0, colsum = 0;
  for (int64_t q = 0; q < k; ++q) {
    const int64_t x = load_int(a, a_type, row * k + q);
    int64_t y = load_int(b, b_type, q * n + col);
    if (w4) y = int4_value(y, b_type);
    if (kind == kGemmI8) {
      g_lo += x * y;
      s_lo += x;
    } else {
      g_lo += (x & 255) * y;
      g_hi += (x >> 8) * y;
      s_lo += x & 255;
      s_hi += x >> 8;
    }
    colsum += y;
  }
  float v = c3 ? mac_exact(colsum, *c3, c0) : c0;
  if (kind == kGemmI8) {
    v = mac_exact(g_lo, c2, v);
  } else if (kind == kGemmI16Product) {
    v = fma_q(mac_exact(g_hi, 256.0f, int_f32(g_lo)), c2, v);
  } else {
    v = mac_exact(g_lo, c2, v);
    v = mac_exact(g_hi, mul_q(c2, 256.0f), v);
  }
  if (c1) {
    const float sum = kind == kGemmI8 ? int_f32(s_lo)
                                      : row_sum_f32(s_lo, s_hi, kind != kGemmI16Exact);
    v = fma_q(sum, *c1, v);
  }
  return v;
}

// Coefficients of the QDQ GEMMs: per-column arrays (c0, c1, c2; c1 null for no activation
// sum, no c3 term) or the four scalars of the activation x activation kernels.
struct GemmCoef {
  bool vec;
  const float *c0, *c1, *c2;
  float s0, s1, s2, s3;
};

__global__ void gemm_qdq_kernel(int64_t n_out, GemmKind kind, bool w4, const void* a,
                                npu_dtype a_type, const void* b, npu_dtype b_type, void* out,
                                npu_dtype out_type, int64_t k, int64_t n, GemmCoef q, int act,
                                float act0, float act1) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  const int64_t col = i % n;
  float v;
  if (q.vec) {
    const float c1 = q.c1 ? q.c1[col] : 0.0f;
    v = gemm_qdq_acc(kind, a, a_type, b, b_type, k, n, i / n, col, q.c0[col],
                     q.c1 ? &c1 : nullptr, q.c2[col], nullptr, w4);
  } else {
    v = gemm_qdq_acc(kind, a, a_type, b, b_type, k, n, i / n, col, q.s0, &q.s1, q.s2, &q.s3, w4);
  }
  store_int(out, out_type, i, activate_out(v, act, act0, act1, 0.0f, out_type));
}

__global__ void gemm_hybrid_kernel(int64_t n_out, const void* a, npu_dtype a_type, const void* b,
                                   npu_dtype b_type, void* out, npu_dtype out_type, int64_t k,
                                   int64_t n, int64_t mode_a_cols, const float* c0_a,
                                   const float* c0_b, const float* c1, const float* c2, int act,
                                   float act0, float act1) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  const int64_t row = i / n, col = i % n;
  const bool mode_a = col < mode_a_cols;
  // mode A reads the low byte of a signed input as lo - 128
  const int64_t bias = a_type == NPU_I16 ? 128 : 0;
  int64_t g_lo = 0, g_hi = 0, s_lo = 0, s_hi = 0;
  for (int64_t q = 0; q < k; ++q) {
    const int64_t x = load_int(a, a_type, row * k + q);
    const int64_t y = load_int(b, b_type, q * n + col);
    const int64_t lo_a = (x & 255) - bias;
    g_lo += (mode_a ? lo_a : (x & 255)) * y;
    g_hi += (x >> 8) * y;
    s_lo += lo_a;
    s_hi += x >> 8;
  }
  float v = mac_exact(g_lo, c2[col], mode_a ? c0_a[col] : c0_b[col]);
  v = mac_exact(g_hi, mul_q(c2[col], 256.0f), v);
  v = fma_q(row_sum_f32(s_lo, s_hi, true), c1[col], v);
  store_int(out, out_type, i, activate_out(v, act, act0, act1, 0.0f, out_type));
}

__global__ void gemm_add_a8w8_kernel(int64_t n_out, const void* a, npu_dtype a_type,
                                     const void* b, npu_dtype b_type, const void* ifm3,
                                     npu_dtype ifm3_type, void* out, npu_dtype out_type,
                                     int64_t k, int64_t n, const float* c0, const float* c1,
                                     const float* c2, float add_c0, float add_c1, float add_c2,
                                     int kpass) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  const int64_t row = i / n, col = i % n;
  float f0 = c0[col], f1 = c1 ? c1[col] : 0.0f, f2 = c2[col];
  if (kpass) {
    // the add is folded into the coefficients
    f0 = fma_q(f0, add_c0, add_c2);
    f1 = fma_q(f1, add_c0, 0.0f);
    f2 = fma_q(f2, add_c0, 0.0f);
  }
  float g = gemm_qdq_acc(kGemmI8, a, a_type, b, b_type, k, n, row, col, f0, c1 ? &f1 : nullptr,
                         f2, nullptr);
  if (!kpass) g = fma_q(g, add_c0, add_c2);
  store_int(out, out_type, i, requant(load_int(ifm3, ifm3_type, i), add_c1, g, out_type));
}

__global__ void gemm_add_a16w8_kernel(int64_t n_out, const void* a, const void* b,
                                      const void* residual,
                                      void* out, int64_t k, int64_t n, const float* c0,
                                      const float* c1, const float* c2, float add_c0,
                                      float add_c1, float add_c2) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  const int64_t col = i % n;
  // The residual offset add_c0 * add_c2 is folded into c0, the accumulators' start value.
  const float c0f = add_q(c0[col], mul_q(add_c0, add_c2));
  const float g = gemm_qdq_acc(kGemmI16Exact, a, NPU_U16, b, NPU_I8, k, n, i / n, col, c0f,
                               c1 ? c1 + col : nullptr, c2[col], nullptr);
  store_int(out, NPU_U16, i, requant(load_int(residual, NPU_U16, i), add_c1, g, NPU_U16));
}

__global__ void gemm_qdq_int16x16_kernel(int64_t n, const void* a, npu_dtype a_type,
                                         const void* b, npu_dtype b_type, void* out,
                                         npu_dtype out_type, int64_t k, int64_t cols, float c0,
                                         float c1, float c2, float c3, int act,
                                         float act0, float act1) {
  NPU_TID;
  const int64_t row = i / cols, col = i % cols;
  int64_t lo = 0, mid = 0, hi = 0, wl = 0, wh = 0, asum = 0;
  for (int64_t q = 0; q < k; ++q) {
    const int64_t x = load_int(a, a_type, row * k + q);
    const int64_t y = load_int(b, b_type, q * cols + col);
    const int64_t xl = x & 255, xh = x >> 8, yl = y & 255, yh = y >> 8;
    lo += xl * yl;
    mid += xh * yl + xl * yh;
    hi += xh * yh;
    wl += yl;
    wh += yh;
    asum += x;
  }
  float v = mac_exact(wl, c3, c0);
  v = mac_exact(wh, mul_q(c3, 256.0f), v);
  v = mac_exact(lo, c2, v);
  v = mac_exact(mid, mul_q(c2, 256.0f), v);
  v = mac_exact(hi, mul_q(c2, 65536.0f), v);
  v = fma_q(int_f32(asum), c1, v);
  store_int(out, out_type, i, activate_out(v, act, act0, act1, 0.0f, out_type));
}

// M and N may be 0 (empty output, nothing launched); the reduction length K is at least 1.
bool gemm_sizes_ok(int64_t m, int64_t k, int64_t n) { return m >= 0 && k >= 1 && n >= 0; }

// Activations of the QDQ GEMM output stage; GELU only where the kernel has it.
bool gemm_act_ok(int act, bool gelu) {
  return act == kQdqActNone || act == kQdqActClip || act == kQdqActLrelu ||
         (gelu && act == kQdqActGelu);
}

int gemm_qdq(GemmKind kind, bool w4, bool gelu, void* stream, const void* a, npu_dtype a_type,
             const void* b, npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
             int64_t n, const GemmCoef& q, int act, float act0, float act1) {
  const int bits = kind == kGemmI8 ? 8 : 16;
  if (any_null(a, b, out) || !is_int_bits(a_type, bits) || !is_i8(b_type) ||
      !is_int_bits(out_type, bits) || (q.vec && any_null(q.c0, q.c2)) || !gemm_act_ok(act, gelu) ||
      !gemm_sizes_ok(m, k, n))
    return -1;
  return launch(stream, m * n, gemm_qdq_kernel, kind, w4, a, a_type, b, b_type, out, out_type, k,
                n, q, act, act0, act1);
}

int gemm_vec(GemmKind kind, bool w4, bool gelu, void* stream, const void* a, npu_dtype a_type,
             const void* b, npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
             int64_t n, const float* c0, const float* c1, const float* c2, int act, float act0,
             float act1) {
  const GemmCoef q{true, c0, c1, c2, 0.0f, 0.0f, 0.0f, 0.0f};
  return gemm_qdq(kind, w4, gelu, stream, a, a_type, b, b_type, out, out_type, m, k, n, q, act,
                  act0, act1);
}

int gemm_scalar(GemmKind kind, void* stream, const void* a, npu_dtype a_type, const void* b,
                npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k, int64_t n,
                float c0, float c1, float c2, float c3, int act, float act0, float act1) {
  const GemmCoef q{false, nullptr, nullptr, nullptr, c0, c1, c2, c3};
  return gemm_qdq(kind, false, false, stream, a, a_type, b, b_type, out, out_type, m, k, n, q,
                  act, act0, act1);
}

int gemm_hybrid(void* stream, const void* a, npu_dtype a_type, const void* b, npu_dtype b_type,
                void* out, npu_dtype out_type, int64_t m, int64_t k, int64_t n,
                int64_t mode_a_cols, const float* c0_a, const float* c0_b, const float* c1,
                const float* c2, int act, float act0, float act1) {
  if (any_null(a, b, out, c0_a, c0_b, c1, c2) || !is_i16(a_type) || !is_i8(b_type) ||
      !is_i16(out_type) || !gemm_act_ok(act, false) || !gemm_sizes_ok(m, k, n) ||
      mode_a_cols < 0 || mode_a_cols > n)
    return -1;
  return launch(stream, m * n, gemm_hybrid_kernel, a, a_type, b, b_type, out, out_type, k, n,
                mode_a_cols, c0_a, c0_b, c1, c2, act, act0, act1);
}

}  // namespace

extern "C" {

int npu_gemm_a8w8(void* stream, const void* a, npu_dtype a_type, const void* b, npu_dtype b_type,
                  void* out, npu_dtype out_type, int64_t m, int64_t k, int64_t n,
                  const float* c0, const float* c1, const float* c2, int activation,
                  float act0, float act1) {
  return gemm_vec(kGemmI8, false, true, stream, a, a_type, b, b_type, out, out_type, m, k, n, c0,
                  c1, c2, activation, act0, act1);
}

int npu_gemm_a8w8_stream(void* stream, const void* a, npu_dtype a_type, const void* b,
                         npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                         int64_t n, const float* c0, const float* c1, const float* c2,
                         int activation, float act0, float act1) {
  return gemm_vec(kGemmI8, false, true, stream, a, a_type, b, b_type, out, out_type, m, k, n, c0,
                  c1, c2, activation, act0, act1);
}

int npu_gemm_a8w4(void* stream, const void* a, npu_dtype a_type, const void* b, npu_dtype b_type,
                  void* out, npu_dtype out_type, int64_t m, int64_t k, int64_t n,
                  const float* c0, const float* c1, const float* c2, int activation,
                  float act0, float act1) {
  return gemm_vec(kGemmI8, true, false, stream, a, a_type, b, b_type, out, out_type, m, k, n, c0,
                  c1, c2, activation, act0, act1);
}

int npu_gemm_a8a8(void* stream, const void* a, npu_dtype a_type, const void* b, npu_dtype b_type,
                  void* out, npu_dtype out_type, int64_t m, int64_t k, int64_t n, float c0,
                  float c1, float c2, float c3, int activation, float act0, float act1) {
  return gemm_scalar(kGemmI8, stream, a, a_type, b, b_type, out, out_type, m, k, n, c0, c1, c2,
                     c3, activation, act0, act1);
}

int npu_gemm_qdq_a8a8_transpose(void* stream, const void* a, npu_dtype a_type, const void* b,
                                npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                                int64_t k, int64_t n, float c0, float c1, float c2, float c3) {
  return gemm_scalar(kGemmI8, stream, a, a_type, b, b_type, out, out_type, m, k, n, c0, c1, c2,
                     c3, kQdqActNone, 0.0f, 0.0f);
}

int npu_gemm_a16w8(void* stream, const void* a, npu_dtype a_type, const void* b,
                   npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                   int64_t n, const float* c0, const float* c1, const float* c2, int activation,
                   float act0, float act1) {
  return gemm_vec(kGemmI16Join, false, true, stream, a, a_type, b, b_type, out, out_type, m, k,
                  n, c0, c1, c2, activation, act0, act1);
}

int npu_gemm_a16w8_stream(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                          int64_t n, const float* c0, const float* c1, const float* c2,
                          int activation, float act0, float act1) {
  return gemm_vec(kGemmI16Exact, false, true, stream, a, a_type, b, b_type, out, out_type, m, k,
                  n, c0, c1, c2, activation, act0, act1);
}

int npu_gemm_a16a8(void* stream, const void* a, npu_dtype a_type, const void* b,
                   npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                   int64_t n, float c0, float c1, float c2, float c3, int activation, float act0,
                   float act1) {
  return gemm_scalar(kGemmI16Join, stream, a, a_type, b, b_type, out, out_type, m, k, n, c0, c1,
                     c2, c3, activation, act0, act1);
}

int npu_gemm_a16w4(void* stream, const void* a, npu_dtype a_type, const void* b,
                   npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                   int64_t n, const float* c0, const float* c1, const float* c2, int activation,
                   float act0, float act1) {
  return gemm_vec(kGemmI16Product, true, false, stream, a, a_type, b, b_type, out, out_type, m,
                  k, n, c0, c1, c2, activation, act0, act1);
}

int npu_gemm_a16w8_hybrid(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                          int64_t n, int64_t mode_a_cols, const float* c0_a, const float* c0_b,
                          const float* c1, const float* c2, int activation, float act0,
                          float act1) {
  return gemm_hybrid(stream, a, a_type, b, b_type, out, out_type, m, k, n, mode_a_cols, c0_a,
                     c0_b, c1, c2, activation, act0, act1);
}

int npu_gemm_a16w8_hybrid_channelwise(void* stream, const void* a, npu_dtype a_type,
                                      const void* b, npu_dtype b_type, void* out,
                                      npu_dtype out_type, int64_t m, int64_t k, int64_t n,
                                      int64_t mode_a_cols, const float* c0_a, const float* c0_b,
                                      const float* c1, const float* c2, int activation,
                                      float act0, float act1) {
  return gemm_hybrid(stream, a, a_type, b, b_type, out, out_type, m, k, n, mode_a_cols, c0_a,
                     c0_b, c1, c2, activation, act0, act1);
}

int npu_gemm_qdq_int16x16(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                          int64_t k, int64_t n, float c0, float c1, float c2, float c3,
                          int activation, float act0, float act1) {
  if (any_null(a, b, out) || !is_i16(a_type) || !is_i16(b_type) || !is_i16(out_type) ||
      !gemm_sizes_ok(m, k, n) || !gemm_act_ok(activation, false))
    return -1;
  return launch(stream, m * n, gemm_qdq_int16x16_kernel, a, a_type, b, b_type, out, out_type, k,
                n, c0, c1, c2, c3, activation, act0, act1);
}

int npu_gemm_qdq_a16a16_transpose(void* stream, const void* a, npu_dtype a_type, const void* b,
                                  npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                                  int64_t k, int64_t n, float c0, float c1, float c2, float c3,
                                  int activation, float act0, float act1) {
  return npu_gemm_qdq_int16x16(stream, a, a_type, b, b_type, out, out_type, m, k, n, c0, c1, c2,
                               c3, activation, act0, act1);
}

int npu_qdq_gemm_add_a8w8(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, const void* ifm3, npu_dtype ifm3_type, void* out,
                          npu_dtype out_type, int64_t m, int64_t k, int64_t n, const float* c0,
                          const float* c1, const float* c2, float add_c0, float add_c1,
                          float add_c2, int kpass) {
  if (any_null(a, b, ifm3, out, c0, c2) || !is_i8(a_type) || !is_i8(b_type) ||
      !is_i8(ifm3_type) || !is_i8(out_type) || !gemm_sizes_ok(m, k, n) ||
      (kpass != 0 && kpass != 1))
    return -1;
  return launch(stream, m * n, gemm_add_a8w8_kernel, a, a_type, b, b_type, ifm3, ifm3_type, out,
                out_type, k, n, c0, c1, c2, add_c0, add_c1, add_c2, kpass);
}

int npu_gemm_add_a16w8(void* stream, const void* a, const void* b, const void* residual,
                       void* out, int64_t m, int64_t k, int64_t n, const float* c0,
                       const float* c1, const float* c2, float add_c0, float add_c1,
                       float add_c2) {
  if (any_null(a, b, residual, out, c0, c2) || !gemm_sizes_ok(m, k, n)) return -1;
  return launch(stream, m * n, gemm_add_a16w8_kernel, a, b, residual, out, k, n, c0, c1, c2,
                add_c0, add_c1, add_c2);
}

}  // extern "C"
