/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Layer norm, add + layer norm, RMS norm and L2 norm of the NPU, in bf16 /
 * f16 / fp8 / bfp16 and quantized forms.
 *
 * Every tensor is [rows][C], row-major; each row is normalized on its own.
 * bfp16 tensors are blocks of 8 channels in 9 bytes (exponent, 8 int8
 * mantissas), C % 8 == 0. Per-channel weights are arrays of C elements.
 * Statistics run in 32 fp32 lanes (lane l sees channels l, l + 32, ...) joined
 * by one vector reduce-add. Rounding is half to even; float -> int saturates.
 * QDQ entries: x = cd(fma(q, scale, fp32(-zp * scale))) with cd = round to the
 * compute dtype; out = sat(rne(fma(y, q_scale, q_zp))).
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
 *   BATCHED: mu = fp32(sum * inv_s); var = fma(-mu, mu, fma(sumsq, inv_s, eps)).
 *            Requires C = 64 or 96. */
enum { NPU_LN_STATS_FUSED = 0, NPU_LN_STATS_BATCHED = 1 };

/* y = alpha * (x - mu) / sqrt(var + eps) + beta. dtype BF16 / F16 (alpha,
 * beta the same dtype) or FP8 (alpha, beta F16). C % 32 == 0 (FP8: C % 64 == 0).
 * C_real == 0: mu and var over all C channels, eps must be 1e-12f.
 * 0 < C_real <= C (BF16 only): mu and var over channels [0, C_real) with
 * divisor C_real; channels from C_real on may hold anything (they are still
 * normalized, so their alpha / beta should be zero). */
int npu_layer_norm_bf16(void* stream, const void* in, npu_dtype dtype, const void* alpha,
                        const void* beta, void* out, int64_t rows, int C, int C_real,
                        float eps);
/* bfp16 in / out, bf16 compute; alpha, beta bf16. C % 32 == 0. */
int npu_layer_norm_bfp16(void* stream, const void* in, const void* alpha, const void* beta,
                         void* out, int64_t rows, int C);
/* bf16 in, bfp16 out. */
int npu_layer_norm_bf16_bfp16(void* stream, const void* in, const void* alpha,
                              const void* beta, void* out, int64_t rows, int C);

/* y = x * r * gamma, r = 1 / sqrt(fp32(fp32(s * inv_inner_dim) + eps)), s = sum x^2.
 * inv_inner_dim = fp32(1 / C). dtype BF16 / F16 (gamma the same dtype) or FP8
 * (gamma F16). C % 32 == 0. */
int npu_rmsnorm_bf16(void* stream, const void* in, npu_dtype dtype, const void* gamma,
                     void* out, int64_t rows, int C, float inv_inner_dim, float eps);
/* bfp16 in / out, bf16 compute; gamma bf16. */
int npu_rmsnorm_bfp16(void* stream, const void* in, const void* gamma, void* out,
                      int64_t rows, int C, float inv_inner_dim, float eps);

/* y = fma(fp32(alpha * x), r, beta), r = 1 / sqrt(sum x^2 + 1e-12). dtype
 * BF16 / F16 (alpha, beta the same dtype) or FP8 (alpha, beta F16).
 * C % 32 == 0 (FP8: C % 64 == 0). */
int npu_l2_norm_bf16(void* stream, const void* in, npu_dtype dtype, const void* alpha,
                     const void* beta, void* out, int64_t rows, int C);
/* Quantized L2 norm, bf16 compute: r = min(1 / sqrt(sum x^2), 1e12);
 * y = bf16(fma(x, fp32(alpha * r), beta)), or bf16(x * r) when alpha and beta
 * (bf16) are both null. a8: in / out I8 or U8; a16: I16 or U16. C % 32 == 0. */
int npu_qdq_l2norm_a8(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                      const void* beta, void* out, npu_dtype out_type, int64_t rows, int C,
                      float dq_scale, float dq_zp, float q_scale, float q_zp);
int npu_qdq_l2norm_a16(void* stream, const void* in, npu_dtype in_type, const void* alpha,
                       const void* beta, void* out, npu_dtype out_type, int64_t rows, int C,
                       float dq_scale, float dq_zp, float q_scale, float q_zp);

/* Layer norm of a + b (sum rounded once to dtype; fp8: to f16). stats =
 * NPU_LN_STATS_*. dtype BF16 / F16 (alpha, beta the same dtype) or FP8 (alpha,
 * beta F16). */
int npu_add_layernorm_bf16(void* stream, const void* a, const void* b, npu_dtype dtype,
                           const void* alpha, const void* beta, void* out, int64_t rows,
                           int C, float eps, int stats);
/* As above with b = add_const, a full [rows][C] tensor in dtype. */
int npu_add_layernorm_const_bf16(void* stream, const void* a, const void* add_const,
                                 npu_dtype dtype, const void* alpha, const void* beta,
                                 void* out, int64_t rows, int C, float eps, int stats);

/* Quantized layer norm of v = cd(cd(dq_a(a)) + cd(dq_b(b))), compute BF16 /
 * F16 (alpha, beta compute dtype). in_type applies to both inputs. */
int npu_qdq_add_layernorm(void* stream, const void* a, const void* b, npu_dtype in_type,
                          npu_dtype compute, const void* alpha, const void* beta, void* out,
                          npu_dtype out_type, int64_t rows, int C, float a_scale, float a_zp,
                          float b_scale, float b_zp, float q_scale, float q_zp, float eps,
                          int stats);
/* As above with v = cd(cd(dq(a)) + add_const), add_const [rows][C] in compute
 * dtype. */
int npu_qdq_add_layernorm_const(void* stream, const void* a, npu_dtype in_type,
                                const void* add_const, npu_dtype compute, const void* alpha,
                                const void* beta, void* out, npu_dtype out_type, int64_t rows,
                                int C, float dq_scale, float dq_zp, float q_scale, float q_zp,
                                float eps, int stats);
/* Quantized layer norm of v = cd(A + fp32(dq_b(b) + add_const[c])), with
 * A = cd(dq_a(a)) for an integer a_type or a itself for a_type == compute
 * (a_scale / a_zp unused). add_const has C values in compute dtype. compute
 * BF16 / F16 / F32 (alpha, beta the same dtype). */
int npu_qdq_add_add_const_layernorm(void* stream, const void* a, npu_dtype a_type,
                                    const void* b, npu_dtype b_type, const void* add_const,
                                    npu_dtype compute, const void* alpha, const void* beta,
                                    void* out, npu_dtype out_type, int64_t rows, int C,
                                    float a_scale, float a_zp, float b_scale, float b_zp,
                                    float q_scale, float q_zp, float eps, int stats);

/* Quantized layer norm with the output quantization folded into the weights:
 * alpha = fp32(alpha / out_scale), beta = fp32(beta / out_scale + out_zp), fp32
 * arrays; out = sat(rne(y)). eps = 1e-5f. compute BF16 / F16 / F32. variant:
 *   FUSED / BATCHED: the NPU_LN_STATS_* paths; BATCHED (C = 64 or 96) uses a
 *     Newton reciprocal square root.
 *   C8_FUSED / C8_BATCHED: 8 lanes folded by an fp32 adder tree, C % 8 == 0;
 *     C8_BATCHED (C = 8 or 16) uses the Newton reciprocal square root.
 *   INTEGER: integer statistics on the raw codes; in / out I8 / U8, compute
 *     F16, C % 32 == 0; dq_zp unused. */
enum {
  NPU_QLN_FUSED = 0,
  NPU_QLN_BATCHED = 1,
  NPU_QLN_C8_FUSED = 2,
  NPU_QLN_C8_BATCHED = 3,
  NPU_QLN_INTEGER = 4,
};
int npu_qdq_layernorm(void* stream, const void* in, npu_dtype in_type, npu_dtype compute,
                      const float* alpha, const float* beta, void* out, npu_dtype out_type,
                      int64_t rows, int C, float dq_scale, float dq_zp, int variant);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_LAYERNORM_H_
