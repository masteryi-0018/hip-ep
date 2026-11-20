/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * QDQ integer GEMMs (a8w8 / a8a8 / a8w4 / a16w8 / a16a8 / a16w4, stream, hybrid and
 * transpose variants), int16 x int16 and the GEMMs with a fused add. Conventions at
 * npu_activation in common/npu_emu.h.
 *
 * Special values: the integer accumulators are exact; the float words (c0..c3, act0, act1,
 * add_c*) follow IEEE through the multiply-adds, a NaN or infinity reaches the output only
 * through the float -> int conversion: NaN -> 0, saturated otherwise [default].
 */
#ifndef NPU_EMU_QDQ_GEMM_H_
#define NPU_EMU_QDQ_GEMM_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* c0 / c1 / c2 / c3: device arrays of N floats; c1 null for no activation-sum term, c3 null
 * for no weight-sum term. gemm = sum_k A B, rowsum = sum_k A, colsum = sum_k B.
 * activation: NONE, CLIP, LRELU or GELU. */
#define NPU_GEMM_QDQ_ARGS                                                                  \
  void *stream, const void *a, npu_dtype a_type, const void *b, npu_dtype b_type, void *out, \
      npu_dtype out_type, int64_t m, int64_t k, int64_t n, const float *c0, const float *c1, \
      const float *c2, const float *c3, npu_activation activation, float act0, float act1

/* A I8 / U8, B I8 / U8 (a8w4: int4 values), out I8 / U8.
 *   v = c3 ? mac(colsum, c3, c0) : c0;  v = mac(gemm, c2, v);  v = fma(fp32(rowsum), c1, v) */
int npu_gemm_a8w8(NPU_GEMM_QDQ_ARGS);
/* See npu_gemm_a8w8. */
int npu_gemm_a8w8_stream(NPU_GEMM_QDQ_ARGS);
/* See npu_gemm_a8w8. */
int npu_gemm_a8a8(NPU_GEMM_QDQ_ARGS);
int npu_gemm_qdq_a8a8_transpose(NPU_GEMM_QDQ_ARGS);
/* See npu_gemm_a8w8. */
int npu_gemm_a8w4(NPU_GEMM_QDQ_ARGS);

/* A I16 / U16 split into a low (unsigned) and a high byte, B I8 / U8, out I16 / U16.
 *   v = c3 ? mac(colsum, c3, c0) : c0
 *   v = mac(gemm_lo, c2, v);  v = mac(gemm_hi, 256 c2, v);  v = fma(S, c1, v)
 * S = mac(sum_k hi, 256, fp32(sum_k lo)); npu_gemm_a16w8_stream: S = fp32(rowsum). */
int npu_gemm_a16w8(NPU_GEMM_QDQ_ARGS);
/* See npu_gemm_a16w8. */
int npu_gemm_a16a8(NPU_GEMM_QDQ_ARGS);
/* See npu_gemm_a16w8. */
int npu_gemm_a16w8_stream(NPU_GEMM_QDQ_ARGS);

/* As npu_gemm_a16w8 (int4 weights), the products joined before the c2 MAC; c3 must be null:
 *   g = mac(gemm_hi, 256, fp32(gemm_lo));  v = fma(g, c2, c0);  v = fma(S, c1, v) */
int npu_gemm_a16w4(NPU_GEMM_QDQ_ARGS);

/* Columns [0, mode_a_cols) run in mode A, the rest in mode B. For I16 input mode A reads
 * the low byte as lo - 128, mode B as the unsigned lo; in both modes
 * S = mac(sum_k hi, 256, fp32(sum_k lo_A)), lo_A the mode-A low byte. c0_a / c0_b: the
 * per-mode c0, N floats each.
 *   v = mac(gemm_lo_mode, c2, c0_mode);  v = mac(gemm_hi, 256 c2, v);  v = fma(S, c1, v)
 * activation: NONE, CLIP or LRELU. */
#define NPU_GEMM_HYBRID_ARGS                                                               \
  void *stream, const void *a, npu_dtype a_type, const void *b, npu_dtype b_type, void *out, \
      npu_dtype out_type, int64_t m, int64_t k, int64_t n, const float *c0_a,               \
      const float *c0_b, const float *c1, const float *c2, int64_t mode_a_cols,             \
      npu_activation activation, float act0, float act1
/* The two-mode GEMM described above. */
int npu_gemm_a16w8_hybrid(NPU_GEMM_HYBRID_ARGS);
/* See npu_gemm_a16w8_hybrid. */
int npu_gemm_a16w8_hybrid_channelwise(NPU_GEMM_HYBRID_ARGS);

/* A and B I16 / U16, out I16 / U16. Operands are split into a signed high and an unsigned
 * low byte: lo = A_lo B_lo, mid = A_hi B_lo + A_lo B_hi, hi = A_hi B_hi; wl / wh the column
 * sums of B_lo / B_hi; asum = fp32(sum_k A).
 *   t = mac(wl, c3, c0);  t = mac(wh, 256 c3, t)
 *   v = mac(lo, c2, t);  v = mac(mid, 256 c2, v);  v = mac(hi, 65536 c2, v)
 *   v = fma(asum, c1, v)
 * activation: NONE, CLIP or LRELU. */
int npu_gemm_qdq_int16x16(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                          int64_t k, int64_t n, float c0, float c1, float c2, float c3,
                          npu_activation activation, float act0, float act1);

/* Same arithmetic as npu_gemm_qdq_int16x16. */
int npu_gemm_qdq_a16a16_transpose(void* stream, const void* a, npu_dtype a_type, const void* b,
                                  npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                                  int64_t k, int64_t n, float c0, float c1, float c2, float c3,
                                  npu_activation activation, float act0, float act1);

/* The 8-bit GEMM accumulator G (no activation) plus an I8 / U8 addend ifm3 (M x N), out
 * I8 / U8. kpass = 0: out = mac(ifm3, add_c1, fma(G, add_c0, add_c2)).
 * kpass = 1: the add is folded into the coefficients first, c0 = fma(c0, add_c0, add_c2),
 * c1 = fma(c1, add_c0, 0), c2 = fma(c2, add_c0, 0), then out = mac(ifm3, add_c1, G). */
int npu_qdq_gemm_add_a8w8(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, const void* ifm3, npu_dtype ifm3_type, void* out,
                          npu_dtype out_type, int64_t m, int64_t k, int64_t n, const float* c0,
                          const float* c1, const float* c2, float add_c0, float add_c1,
                          float add_c2, int kpass);

/* The npu_gemm_a16w8_stream accumulator G, started from c0' = fp32(c0 + fp32(add_c0 * add_c2))
 * instead of c0, plus a U16 residual (M x N), out U16: out = mac(residual, add_c1, G). */
int npu_gemm_add_a16w8(void* stream, const void* a, npu_dtype a_type, const void* b,
                       npu_dtype b_type, const void* residual, void* out, int64_t m, int64_t k,
                       int64_t n, const float* c0, const float* c1, const float* c2,
                       float add_c0, float add_c1, float add_c2);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_QDQ_GEMM_H_
