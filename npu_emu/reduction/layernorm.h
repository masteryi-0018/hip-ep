/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Layer norm, add + layer norm, RMS norm and L2 norm of the NPU, in bf16 /
 * f16 / fp8 / bfp16 and quantized forms.
 *
 * Every tensor is [rows][C], row-major; each row is normalized on its own.
 * rows may be 0 (nothing launched); C must be at least 1.
 * bfp16 tensors are blocks of 8 channels in 9 bytes (exponent, 8 int8
 * mantissas), C % 8 == 0. Per-channel weights are arrays of C elements.
 * Statistics run in 32 fp32 lanes (lane l sees channels l, l + 32, ...) joined
 * by one vector reduce-add. Rounding is half to even; float -> int saturates.
 * QDQ entries: x = cd(fma(q, dq_sc, fp32(-dq_zp * dq_sc))) with cd = round to
 * the compute type; out = sat(rne(fma(y, q_sc, q_zp))), q_sc = 1 / output
 * scale. (zero point, scale) pairs come in the kernel's word order.
 * Integer types are I8 / U8 / I16 / U16.
 * Special values: fp32 steps follow the NaN rules of common/mac.h [default]; the statistics
 * of a row with a NaN / infinity are NaN, so the whole row is NaN.
 */
#ifndef NPU_EMU_LAYERNORM_H_
#define NPU_EMU_LAYERNORM_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Statistics path of the add + layer norm entries (inv_s = 1 / C):
 *   FUSED:   mu = sum_l(fp32(sum_l * inv_s)), ex2 = sum_l(fma(sumsq_l, inv_s,
 *            l == 16 ? eps : 0)); var = fma(-mu, mu, ex2).
 *   BATCHED: mu = fp32(sum * inv_s); var = fma(-mu, mu, fma(sumsq, inv_s, eps)),
 *            hardware 1 / sqrt. Requires C = 64 or 96. */
typedef enum { NPU_LN_STATS_FUSED = 0, NPU_LN_STATS_BATCHED = 1 } npu_ln_stats;

/* y = alpha * (x - mu) / sqrt(var + eps) + beta. in_type BF16 / F16 (alpha,
 * beta the same type) or FP8 (alpha, beta F16). C % 32 == 0 (FP8: C % 64 == 0).
 * C_real == 0: mu and var over all C channels, epsilon must be 1e-12f.
 * 0 < C_real <= C (BF16 only): mu and var over channels [0, C_real) with
 * divisor C_real; channels from C_real on may hold anything (they are still
 * normalized, so their alpha / beta should be zero). */
int npu_layer_norm_bf16(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                        const void* beta, void* out, int64_t rows, int C, int C_real,
                        float epsilon);
/* bfp16 in / out, bf16 compute; alpha, beta bf16. C % 32 == 0. */
int npu_layer_norm_bfp16(void* stream, const void* in, const void* alpha, const void* beta,
                         void* out, int64_t rows, int C);
/* bf16 in, bfp16 out. */
int npu_layer_norm_bf16_bfp16(void* stream, const void* in, const void* alpha,
                              const void* beta, void* out, int64_t rows, int C);

/* y = x * r * gamma, r = 1 / sqrt(fp32(fp32(s * inv_inner_dim) + eps)), s = sum x^2,
 * eps = epsilon, or 1e-12f when epsilon is +0. inv_inner_dim = fp32(1 / C).
 * in_type BF16 / F16 (gamma the same type) or FP8 (gamma F16). C % 32 == 0. */
int npu_rmsnorm_bf16(void* stream, const void* in, npu_dtype in_type, const void* gamma,
                     void* out, int64_t rows, int C, float inv_inner_dim, float epsilon);
/* bfp16 in / out, bf16 compute; gamma bf16. */
int npu_rmsnorm_bfp16(void* stream, const void* in, const void* gamma, void* out,
                      int64_t rows, int C, float inv_inner_dim, float epsilon);

/* y = fma(fp32(alpha * x), r, beta), r = 1 / sqrt(sum x^2 + 1e-12). in_type
 * BF16 / F16 (alpha, beta the same type) or FP8 (alpha, beta F16).
 * C % 32 == 0 (FP8: C % 64 == 0). */
int npu_l2_norm_bf16(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                     const void* beta, void* out, int64_t rows, int C);
/* Quantized L2 norm, bf16 compute: r = min(1 / sqrt(sum x^2), 1e12);
 * y = bf16(fma(x, fp32(alpha * r), beta)), or bf16(x * r) when alpha and beta
 * (bf16) are both null. a8: in / out I8 or U8; a16: I16 or U16. C % 32 == 0. */
int npu_qdq_l2norm_a8(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                      const void* beta, void* out, npu_dtype out_type, int64_t rows, int C,
                      float dq_zp, float dq_sc, float q_zp, float q_sc);
int npu_qdq_l2norm_a16(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                       const void* beta, void* out, npu_dtype out_type, int64_t rows, int C,
                       float dq_zp, float dq_sc, float q_zp, float q_sc);

/* Layer norm of a + b (sum rounded once to in_type; fp8: to f16). in_type
 * (a, b and out) BF16 / F16 (alpha, beta the same type) or FP8 (alpha, beta
 * F16). */
int npu_add_layernorm_bf16(void* stream, const void* a, const void* b, npu_dtype in_type,
                           const void* alpha, const void* beta, void* out, int64_t rows,
                           int C, float epsilon, npu_ln_stats stats);
/* As above with b = add_const, a full [rows][C] tensor in in_type. */
int npu_add_layernorm_const_bf16(void* stream, const void* a, const void* add_const,
                                 npu_dtype in_type, const void* alpha, const void* beta,
                                 void* out, int64_t rows, int C, float epsilon,
                                 npu_ln_stats stats);

/* Quantized layer norm of v = cd(cd(dq(a)) + cd(dq2(b))); in_type applies to
 * both inputs; alpha / beta of compute_type (BF16 / F16). */
int npu_qdq_add_layernorm(void* stream, const void* a, const void* b, npu_dtype in_type,
                          const void* alpha, const void* beta, npu_dtype compute_type,
                          void* out, npu_dtype out_type, int64_t rows, int C, float dq_zp,
                          float dq_sc, float q_zp, float q_sc, float dq2_zp, float dq2_sc,
                          float epsilon, npu_ln_stats stats);
/* As above with v = cd(cd(dq(a)) + add_const), add_const [rows][C] of
 * compute_type (also the type of alpha / beta). */
int npu_qdq_add_layernorm_const(void* stream, const void* a, npu_dtype in_type,
                                const void* add_const, npu_dtype compute_type,
                                const void* alpha, const void* beta, void* out,
                                npu_dtype out_type, int64_t rows, int C, float dq_zp,
                                float dq_sc, float q_zp, float q_sc, float epsilon,
                                npu_ln_stats stats);
/* Quantized layer norm of v = cd(A + fp32(dq2(b) + add_const[c])), with
 * A = cd(dq(a)) for a_type == b_type (integer) or a itself for
 * a_type == compute_type (dq_zp / dq_sc unused). add_const has C values of
 * compute_type (BF16 / F16 / F32, also the type of alpha / beta). */
int npu_qdq_add_add_const_layernorm(void* stream, const void* a, npu_dtype a_type,
                                    const void* b, npu_dtype b_type, const void* add_const,
                                    npu_dtype compute_type, const void* alpha,
                                    const void* beta, void* out, npu_dtype out_type,
                                    int64_t rows, int C, float dq_zp, float dq_sc, float q_zp,
                                    float q_sc, float dq2_zp, float dq2_sc, float epsilon,
                                    npu_ln_stats stats);

/* Quantized layer norm with the output quantization folded into the weights:
 * alpha = fp32(alpha / out_scale), beta = fp32(beta / out_scale + out_zp), device
 * float arrays; out = sat(rne(y)). eps = 1e-5f. compute_type BF16 / F16 / F32.
 * variant:
 *   FUSED / BATCHED: the NPU_LN_STATS_* statistics, except that BATCHED
 *     (C = 64 or 96) uses a Newton reciprocal square root instead of the
 *     hardware one.
 *   C8_FUSED / C8_BATCHED: 8 lanes folded by an fp32 adder tree, C % 8 == 0,
 *     C < 32; C8_BATCHED (C = 8 or 16) uses the Newton reciprocal square root.
 *   INTEGER: integer statistics on the raw codes; in / out I8 / U8, compute
 *     F16, C = 64, 96, 192 or a multiple of 64 in [256, 768]; dq_zp unused. */
typedef enum {
  NPU_QLN_FUSED = 0,
  NPU_QLN_BATCHED = 1,
  NPU_QLN_C8_FUSED = 2,
  NPU_QLN_C8_BATCHED = 3,
  NPU_QLN_INTEGER = 4,
} npu_qln_variant;
int npu_qdq_layernorm(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                      const void* beta, void* out, npu_dtype out_type, int64_t rows, int C,
                      npu_dtype compute_type, float dq_zp, float dq_sc,
                      npu_qln_variant variant);
/* Layer norm of x = f16(dq(q)), q I8 / U8, f16 compute, alpha / beta f16,
 * eps = 1e-5f; y rounded to f16, then encoded to bfp16 from those f16 values.
 * variant NPU_QLN_FUSED or NPU_QLN_BATCHED (C = 64 or 96). C % 32 == 0. */
int npu_layer_norm_a8_bfp16(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                            const void* beta, void* out, int64_t rows, int C, float dq_zp,
                            float dq_sc, npu_qln_variant variant);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_LAYERNORM_H_
