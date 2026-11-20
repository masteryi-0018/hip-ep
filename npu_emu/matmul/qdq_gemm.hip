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
  return join ? mac_exact(hi_sum, 256.0f, float(lo_sum)) : float(lo_sum + 256 * hi_sum);
}

// fp32 accumulator of output (row, col), before the activation; c1 / c3 may be null.
__device__ inline float gemm_qdq_acc(GemmKind kind, const void* a, npu_dtype a_type, const void* b,
                                     npu_dtype b_type, int64_t k, int64_t n, int64_t row,
                                     int64_t col, float c0, const float* c1, float c2,
                                     const float* c3) {
  int64_t g_lo = 0, g_hi = 0, s_lo = 0, s_hi = 0, colsum = 0;
  for (int64_t q = 0; q < k; ++q) {
    const int64_t x = load_int(a, a_type, row * k + q);
    const int64_t y = load_int(b, b_type, q * n + col);
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
    v = fmaf(mac_exact(g_hi, 256.0f, float(g_lo)), c2, v);
  } else {
    v = mac_exact(g_lo, c2, v);
    v = mac_exact(g_hi, c2 * 256.0f, v);
  }
  if (c1) {
    const float sum = kind == kGemmI8 ? float(s_lo) : row_sum_f32(s_lo, s_hi, kind != kGemmI16Exact);
    v = fmaf(sum, *c1, v);
  }
  return v;
}

__global__ void gemm_qdq_kernel(int64_t n_out, GemmKind kind, const void* a, npu_dtype a_type,
                                const void* b, npu_dtype b_type, void* out, npu_dtype out_type,
                                int64_t k, int64_t n, const float* c0, const float* c1,
                                const float* c2, const float* c3, npu_activation act, float act0,
                                float act1) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  const int64_t col = i % n;
  const float v = gemm_qdq_acc(kind, a, a_type, b, b_type, k, n, i / n, col, c0[col],
                               c1 ? c1 + col : nullptr, c2[col], c3 ? c3 + col : nullptr);
  store_int(out, out_type, i, activate_out(v, act, act0, act1, 0.0f, out_type));
}

__global__ void gemm_hybrid_kernel(int64_t n_out, const void* a, npu_dtype a_type, const void* b,
                                   npu_dtype b_type, void* out, npu_dtype out_type, int64_t k,
                                   int64_t n, const float* c0_a, const float* c0_b,
                                   const float* c1, const float* c2, int64_t mode_a_cols,
                                   npu_activation act, float act0, float act1) {
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
  v = mac_exact(g_hi, c2[col] * 256.0f, v);
  v = fmaf(row_sum_f32(s_lo, s_hi, true), c1[col], v);
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
    f0 = fmaf(f0, add_c0, add_c2);
    f1 = fmaf(f1, add_c0, 0.0f);
    f2 = fmaf(f2, add_c0, 0.0f);
  }
  float g = gemm_qdq_acc(kGemmI8, a, a_type, b, b_type, k, n, row, col, f0, c1 ? &f1 : nullptr,
                         f2, nullptr);
  if (!kpass) g = fmaf(g, add_c0, add_c2);
  const float v = mac_exact(load_int(ifm3, ifm3_type, i), add_c1, g);
  store_int(out, out_type, i, activate_out(v, NPU_ACT_NONE, 0.0f, 0.0f, 0.0f, out_type));
}

__global__ void gemm_add_a16w8_kernel(int64_t n_out, const void* a, npu_dtype a_type,
                                      const void* b, npu_dtype b_type, const void* residual,
                                      void* out, int64_t k, int64_t n, const float* c0,
                                      const float* c1, const float* c2, float add_c0,
                                      float add_c1, float add_c2) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  const int64_t col = i % n;
  // The residual offset add_c0 * add_c2 is folded into c0, the accumulators' start value.
  const float c0f = __fadd_rn(c0[col], __fmul_rn(add_c0, add_c2));
  const float g = gemm_qdq_acc(kGemmI16Exact, a, a_type, b, b_type, k, n, i / n, col, c0f,
                               c1 ? c1 + col : nullptr, c2[col], nullptr);
  const float v = mac_exact(load_int(residual, NPU_U16, i), add_c1, g);
  store_int(out, NPU_U16, i, activate_out(v, NPU_ACT_NONE, 0.0f, 0.0f, 0.0f, NPU_U16));
}

__global__ void gemm_qdq_int16x16_kernel(int64_t n, const void* a, npu_dtype a_type,
                                         const void* b, npu_dtype b_type, void* out,
                                         npu_dtype out_type, int64_t k, int64_t cols, float c0,
                                         float c1, float c2, float c3, npu_activation act,
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
  v = mac_exact(wh, c3 * 256.0f, v);
  v = mac_exact(lo, c2, v);
  v = mac_exact(mid, c2 * 256.0f, v);
  v = mac_exact(hi, c2 * 65536.0f, v);
  v = fmaf(float(asum), c1, v);
  store_int(out, out_type, i, activate_out(v, act, act0, act1, 0.0f, out_type));
}

int gemm_qdq(GemmKind kind, void* stream, const void* a, npu_dtype a_type, const void* b,
             npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k, int64_t n,
             const float* c0, const float* c1, const float* c2, const float* c3,
             npu_activation act, float act0, float act1) {
  const bool a16 = kind != kGemmI8;
  if (!(a16 ? is_i16(a_type) : is_i8(a_type)) || !is_i8(b_type) ||
      !(a16 ? is_i16(out_type) : is_i8(out_type)) || !c0 || !c2 || act == NPU_ACT_PRELU ||
      (kind == kGemmI16Product && c3) || m < 0 || k < 0 || n < 0)
    return -1;
  return launch(stream, m * n, gemm_qdq_kernel, kind, a, a_type, b, b_type, out, out_type, k, n,
                c0, c1, c2, c3, act, act0, act1);
}

int gemm_hybrid(void* stream, const void* a, npu_dtype a_type, const void* b, npu_dtype b_type,
                void* out, npu_dtype out_type, int64_t m, int64_t k, int64_t n, const float* c0_a,
                const float* c0_b, const float* c1, const float* c2, int64_t mode_a_cols,
                npu_activation act, float act0, float act1) {
  if (!is_i16(a_type) || !is_i8(b_type) || !is_i16(out_type) || !c0_a || !c0_b || !c1 || !c2 ||
      act == NPU_ACT_PRELU || act == NPU_ACT_GELU || m < 0 || k < 0 || n < 0 || mode_a_cols < 0)
    return -1;
  return launch(stream, m * n, gemm_hybrid_kernel, a, a_type, b, b_type, out, out_type, k, n,
                c0_a, c0_b, c1, c2, mode_a_cols, act, act0, act1);
}

}  // namespace

extern "C" {

#define NPU_GEMM_QDQ_ENTRY(name, kind)                                                     \
  int name(NPU_GEMM_QDQ_ARGS) {                                                            \
    return gemm_qdq(kind, stream, a, a_type, b, b_type, out, out_type, m, k, n, c0, c1, c2, \
                    c3, activation, act0, act1);                                           \
  }
NPU_GEMM_QDQ_ENTRY(npu_gemm_a8w8, kGemmI8)
NPU_GEMM_QDQ_ENTRY(npu_gemm_a8w8_stream, kGemmI8)
NPU_GEMM_QDQ_ENTRY(npu_gemm_a8a8, kGemmI8)
NPU_GEMM_QDQ_ENTRY(npu_gemm_qdq_a8a8_transpose, kGemmI8)
NPU_GEMM_QDQ_ENTRY(npu_gemm_a8w4, kGemmI8)
NPU_GEMM_QDQ_ENTRY(npu_gemm_a16w8, kGemmI16Join)
NPU_GEMM_QDQ_ENTRY(npu_gemm_a16a8, kGemmI16Join)
NPU_GEMM_QDQ_ENTRY(npu_gemm_a16w8_stream, kGemmI16Exact)
NPU_GEMM_QDQ_ENTRY(npu_gemm_a16w4, kGemmI16Product)
#undef NPU_GEMM_QDQ_ENTRY

int npu_gemm_a16w8_hybrid(NPU_GEMM_HYBRID_ARGS) {
  return gemm_hybrid(stream, a, a_type, b, b_type, out, out_type, m, k, n, c0_a, c0_b, c1, c2,
                     mode_a_cols, activation, act0, act1);
}

int npu_gemm_a16w8_hybrid_channelwise(NPU_GEMM_HYBRID_ARGS) {
  return gemm_hybrid(stream, a, a_type, b, b_type, out, out_type, m, k, n, c0_a, c0_b, c1, c2,
                     mode_a_cols, activation, act0, act1);
}

int npu_gemm_qdq_int16x16(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                          int64_t k, int64_t n, float c0, float c1, float c2, float c3,
                          npu_activation activation, float act0, float act1) {
  if (!is_i16(a_type) || !is_i16(b_type) || !is_i16(out_type) || m < 0 || k < 0 || n < 0 ||
      activation == NPU_ACT_PRELU || activation == NPU_ACT_GELU)
    return -1;
  return launch(stream, m * n, gemm_qdq_int16x16_kernel, a, a_type, b, b_type, out, out_type, k,
                n, c0, c1, c2, c3, activation, act0, act1);
}

int npu_gemm_qdq_a16a16_transpose(void* stream, const void* a, npu_dtype a_type, const void* b,
                                  npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                                  int64_t k, int64_t n, float c0, float c1, float c2, float c3,
                                  npu_activation activation, float act0, float act1) {
  return npu_gemm_qdq_int16x16(stream, a, a_type, b, b_type, out, out_type, m, k, n, c0, c1, c2,
                               c3, activation, act0, act1);
}

int npu_qdq_gemm_add_a8w8(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, const void* ifm3, npu_dtype ifm3_type, void* out,
                          npu_dtype out_type, int64_t m, int64_t k, int64_t n, const float* c0,
                          const float* c1, const float* c2, float add_c0, float add_c1,
                          float add_c2, int kpass) {
  if (!is_i8(a_type) || !is_i8(b_type) || !is_i8(ifm3_type) || !is_i8(out_type) || !c0 || !c2 ||
      m < 0 || k < 0 || n < 0)
    return -1;
  return launch(stream, m * n, gemm_add_a8w8_kernel, a, a_type, b, b_type, ifm3, ifm3_type, out,
                out_type, k, n, c0, c1, c2, add_c0, add_c1, add_c2, kpass);
}

int npu_gemm_add_a16w8(void* stream, const void* a, npu_dtype a_type, const void* b,
                       npu_dtype b_type, const void* residual, void* out, int64_t m, int64_t k,
                       int64_t n, const float* c0, const float* c1, const float* c2,
                       float add_c0, float add_c1, float add_c2) {
  if (!is_i16(a_type) || !is_i8(b_type) || !c0 || !c2 || m < 0 || k < 0 || n < 0) return -1;
  return launch(stream, m * n, gemm_add_a16w8_kernel, a, a_type, b, b_type, residual, out, k, n,
                c0, c1, c2, add_c0, add_c1, add_c2);
}

}  // extern "C"
