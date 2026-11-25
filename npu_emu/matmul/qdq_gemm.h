/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * QDQ integer GEMMs (a8w8 / a8a8 / a8w4 / a16w8 / a16a8 / a16w4, stream, hybrid and
 * transpose variants), int16 x int16 and the GEMMs with a fused add.
 *
 * Common to the QDQ integer kernels (also matmul/qdq_conv.h): integer sums are exact;
 * mac(x, c, a) = fp32(x * c + a) with the integer x exact, fma the same on an fp32 x; the result
 * is rounded half to even and saturated to the output type. The coefficients are the fp32
 * words the kernel receives, not ONNX scales: c2 = s_in * s_w / s_out, c1 = -c2 * zp_w
 * (activation-sum term), c3 = -c2 * zp_in (weight-sum term), c0 the constant term.
 * A is M x K, B is K x N, out M x N, all row-major. M or N = 0 is an empty output (returns 0,
 * nothing launched); K >= 1. Required pointers are rejected when null even then.
 *
 * activation (common/act.h codes): 0 none, 100 CLIP (clamp to [act0, act1]), 3 LRELU
 * (max(v, fma(v, act0, act1))), 6 GELU (fma(gelu(v), act0, act1) on the dequantized v;
 * act0 = 1 / ofm_scale, act1 = the ofm zero point). Each entry rejects the codes its
 * kernel lacks.
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

/* A I8 / U8, B I8 / U8, out I8 / U8. c0 / c1 / c2: device arrays of N floats, c1 null for no
 * activation-sum term. activation 0, 100, 3 or 6.
 *   v = mac(gemm, c2, c0);  v = fma(fp32(rowsum), c1, v)    gemm = sum_k A B, rowsum = sum_k A */
int npu_gemm_a8w8(void* stream, const void* a, npu_dtype a_type, const void* b, npu_dtype b_type,
                  void* out, npu_dtype out_type, int64_t m, int64_t k, int64_t n,
                  const float* c0, const float* c1, const float* c2, int activation,
                  float act0, float act1);
/* See npu_gemm_a8w8. */
int npu_gemm_a8w8_stream(void* stream, const void* a, npu_dtype a_type, const void* b,
                         npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                         int64_t n, const float* c0, const float* c1, const float* c2,
                         int activation, float act0, float act1);
/* As npu_gemm_a8w8 with int4 weights, one per byte: the low nibble of each B byte, signed for
 * b_type I8, unsigned for U8. activation 0, 100 or 3. */
int npu_gemm_a8w4(void* stream, const void* a, npu_dtype a_type, const void* b, npu_dtype b_type,
                  void* out, npu_dtype out_type, int64_t m, int64_t k, int64_t n,
                  const float* c0, const float* c1, const float* c2, int activation,
                  float act0, float act1);

/* A and B I8 / U8 (B an activation), out I8 / U8, four scalar coefficients. activation 0, 100
 * or 3. colsum = sum_k B.
 *   v = mac(colsum, c3, c0);  v = mac(gemm, c2, v);  v = fma(fp32(rowsum), c1, v) */
int npu_gemm_a8a8(void* stream, const void* a, npu_dtype a_type, const void* b, npu_dtype b_type,
                  void* out, npu_dtype out_type, int64_t m, int64_t k, int64_t n, float c0,
                  float c1, float c2, float c3, int activation, float act0, float act1);
/* npu_gemm_a8a8 without an activation. */
int npu_gemm_qdq_a8a8_transpose(void* stream, const void* a, npu_dtype a_type, const void* b,
                                npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                                int64_t k, int64_t n, float c0, float c1, float c2, float c3);

/* A I16 / U16 split into a low (unsigned) and a high byte, B I8 / U8, out I16 / U16; c0 / c1 /
 * c2 as npu_gemm_a8w8. activation 0, 100, 3 or 6.
 *   v = mac(gemm_lo, c2, c0);  v = mac(gemm_hi, 256 c2, v);  v = fma(S, c1, v)
 * S = mac(sum_k hi, 256, fp32(sum_k lo)); npu_gemm_a16w8_stream: S = fp32(rowsum). */
int npu_gemm_a16w8(void* stream, const void* a, npu_dtype a_type, const void* b,
                   npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                   int64_t n, const float* c0, const float* c1, const float* c2, int activation,
                   float act0, float act1);
/* See npu_gemm_a16w8. */
int npu_gemm_a16w8_stream(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                          int64_t n, const float* c0, const float* c1, const float* c2,
                          int activation, float act0, float act1);
/* As npu_gemm_a16w8 (S joined from the byte planes) with four scalar coefficients, B an
 * activation; activation 0, 100 or 3:
 *   v = mac(colsum, c3, c0);  v = mac(gemm_lo, c2, v);  v = mac(gemm_hi, 256 c2, v);
 *   v = fma(S, c1, v) */
int npu_gemm_a16a8(void* stream, const void* a, npu_dtype a_type, const void* b,
                   npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                   int64_t n, float c0, float c1, float c2, float c3, int activation, float act0,
                   float act1);
/* As npu_gemm_a16w8 with int4 weights (as npu_gemm_a8w4), the products joined before the c2
 * MAC; activation 0, 100 or 3:
 *   g = mac(gemm_hi, 256, fp32(gemm_lo));  v = fma(g, c2, c0);  v = fma(S, c1, v) */
int npu_gemm_a16w4(void* stream, const void* a, npu_dtype a_type, const void* b,
                   npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                   int64_t n, const float* c0, const float* c1, const float* c2, int activation,
                   float act0, float act1);

/* A I16 / U16, B I8 / U8, out I16 / U16. Columns [0, mode_a_cols) run in mode A, the rest in
 * mode B. For I16 input mode A reads the low byte as lo - 128, mode B as the unsigned lo; in
 * both modes S = mac(sum_k hi, 256, fp32(sum_k lo_A)), lo_A the mode-A low byte. c0_a / c0_b:
 * the per-mode c0, c1 / c2: N floats each. activation 0, 100 or 3.
 *   v = mac(gemm_lo_mode, c2, c0_mode);  v = mac(gemm_hi, 256 c2, v);  v = fma(S, c1, v) */
int npu_gemm_a16w8_hybrid(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, void* out, npu_dtype out_type, int64_t m, int64_t k,
                          int64_t n, int64_t mode_a_cols, const float* c0_a, const float* c0_b,
                          const float* c1, const float* c2, int activation, float act0,
                          float act1);
/* See npu_gemm_a16w8_hybrid. */
int npu_gemm_a16w8_hybrid_channelwise(void* stream, const void* a, npu_dtype a_type,
                                      const void* b, npu_dtype b_type, void* out,
                                      npu_dtype out_type, int64_t m, int64_t k, int64_t n,
                                      int64_t mode_a_cols, const float* c0_a, const float* c0_b,
                                      const float* c1, const float* c2, int activation,
                                      float act0, float act1);

/* A and B I16 / U16, out I16 / U16. Operands are split into a signed high and an unsigned
 * low byte: lo = A_lo B_lo, mid = A_hi B_lo + A_lo B_hi, hi = A_hi B_hi; wl / wh the column
 * sums of B_lo / B_hi; asum = fp32(sum_k A). activation 0, 100 or 3.
 *   t = mac(wl, c3, c0);  t = mac(wh, 256 c3, t)
 *   v = mac(lo, c2, t);  v = mac(mid, 256 c2, v);  v = mac(hi, 65536 c2, v)
 *   v = fma(asum, c1, v) */
int npu_gemm_qdq_int16x16(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                          int64_t k, int64_t n, float c0, float c1, float c2, float c3,
                          int activation, float act0, float act1);
/* Same arithmetic as npu_gemm_qdq_int16x16. */
int npu_gemm_qdq_a16a16_transpose(void* stream, const void* a, npu_dtype a_type, const void* b,
                                  npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                                  int64_t k, int64_t n, float c0, float c1, float c2, float c3,
                                  int activation, float act0, float act1);

/* The npu_gemm_a8w8 accumulator G (no activation) plus an I8 / U8 addend ifm3 (M x N), out
 * I8 / U8. kpass 0: out = mac(ifm3, add_c1, fma(G, add_c0, add_c2)).
 * kpass 1: the add is folded into the coefficients first, c0 = fma(c0, add_c0, add_c2),
 * c1 = fma(c1, add_c0, 0), c2 = fma(c2, add_c0, 0), then out = mac(ifm3, add_c1, G). */
int npu_qdq_gemm_add_a8w8(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, const void* ifm3, npu_dtype ifm3_type, void* out,
                          npu_dtype out_type, int64_t m, int64_t k, int64_t n, const float* c0,
                          const float* c1, const float* c2, float add_c0, float add_c1,
                          float add_c2, int kpass);

/* A U16, B I8, residual and out U16 (M x N). The npu_gemm_a16w8_stream accumulator G,
 * started from c0' = fp32(c0 + fp32(add_c0 * add_c2)) instead of c0, then
 * out = mac(residual, add_c1, G). */
int npu_gemm_add_a16w8(void* stream, const void* a, const void* b, const void* residual,
                       void* out, int64_t m, int64_t k, int64_t n, const float* c0,
                       const float* c1, const float* c2, float add_c0, float add_c1,
                       float add_c2);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_QDQ_GEMM_H_
