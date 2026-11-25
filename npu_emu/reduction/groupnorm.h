/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Group norm and instance norm of the NPU (bf16 / f16 / fp8 and quantized).
 *
 * Layouts are dense: instance norm (hw, c); group norm (g, l) with l = H*W*Cg
 * elements of a group in HWC order (alpha / beta have g*cg entries). hw / l
 * may include padding rows, which enter the sums but not the element count N
 * (spatial_size / div_factor, 0 < N <= hw / l).
 *
 * Statistics are fp32 partial sums of chunks of sv_hw / sv elements, combined
 * over a grid of columns x rows selected by split_mode:
 *   0: 3 x 4;  1: 1 x 4;  2: 1 x 2;  3: 3 x 1;  -1: 1 x 1
 * (rows of a column from last to first, then columns 0 -> 1 -> 2). The number
 * of chunks must be a multiple of columns * rows; other shapes return -1.
 * ifm_type is also the type of ofm (and ofm2). c (instance norm), g (group
 * norm) and hw of npu_qdq_ConvInstanceNormAddMulAdd_Stage2 may be 0: nothing is
 * launched; the other sizes (hw, l, cg, sv, N) must be at least 1.
 * split_mode -1 and 3 exist only for npu_qdq_instance_norm_sd and
 * npu_qdq_groupnorm_sd.
 *
 *   mu = S / N;  var = S2 / N + eps - mu * mu;  inv_std = 1 / sqrt(var)
 *   y = (x - mu) * inv_std * alpha + beta   (eps = 1e-5f unless epsilon is given)
 * QDQ entries: x = T(fma(q, dq_sc, fp32(-dq_zp * dq_sc))) with T the compute
 * type; out = sat(rne(fma(y, q_sc, q_zp))), q_sc = 1 / output scale.
 * add_mul_add entries: y = y * (add1_act + add1_const) + add2.
 * activation (codes of the kernels, not npu_activation): 0 none, 1 ReLU,
 * 3 leaky ReLU (leakyRelu_alpha), 5 SiLU (group norm only).
 * Special values: fp32 steps follow the NaN rules of common/mac.h [default].
 */
#ifndef NPU_EMU_GROUPNORM_H_
#define NPU_EMU_GROUPNORM_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Instance norm. ifm / ofm (hw, c) of ifm_type BF16 / F16 / FP8; alpha / beta
 * (c) ifm_type, F16 for FP8. split_mode 0 / 1 / 2; activation 0 / 1 / 3. Both
 * entries compute the same result. */
int npu_instance_norm_sd_bf16(void* stream, const void* ifm, npu_dtype ifm_type,
                              const void* alpha, const void* beta, void* ofm,
                              int64_t hw, int64_t c, int64_t sv_hw,
                              int split_mode, int64_t spatial_size,
                              int activation, float leakyRelu_alpha);
int npu_instance_norm_sd_opt_stage2_ifm_bf16(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
    const void* beta, void* ofm, int64_t hw, int64_t c, int64_t sv_hw,
    int split_mode, int64_t spatial_size, int activation, float leakyRelu_alpha);

/* As above with add_mul_add. add1_act / add2 (c each) are ifm_type,
 * add1_const (c) has the type of alpha. */
int npu_instance_norm_add_mul_add_sd_bf16(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
    const void* beta, const void* add1_act, const void* add2,
    const void* add1_const, void* ofm, int64_t hw, int64_t c, int64_t sv_hw,
    int split_mode, int64_t spatial_size, int activation, float leakyRelu_alpha);
int npu_instance_norm_add_mul_add_sd_opt_stage2_ifm_bf16(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
    const void* beta, const void* add1_act, const void* add2,
    const void* add1_const, void* ofm, int64_t hw, int64_t c, int64_t sv_hw,
    int split_mode, int64_t spatial_size, int activation, float leakyRelu_alpha);

/* Quantized instance norm. ifm / ofm of ifm_type I8 / U8 / I16 / U16; alpha /
 * beta of compute_type (F32 / BF16 / F16). split_mode -1..3 (sd) or 0..2
 * (opt_stage2_ifm); activation 0 / 1 / 3. */
int npu_qdq_instance_norm_sd(void* stream, const void* ifm, npu_dtype ifm_type,
                             const void* alpha, const void* beta,
                             npu_dtype compute_type, void* ofm, int64_t hw,
                             int64_t c, int64_t sv_hw, int split_mode,
                             int64_t spatial_size, float dq_zp, float dq_sc,
                             float q_zp, float q_sc, int activation,
                             float leakyRelu_alpha);
int npu_qdq_instance_norm_sd_opt_stage2_ifm(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
    const void* beta, npu_dtype compute_type, void* ofm, int64_t hw, int64_t c,
    int64_t sv_hw, int split_mode, int64_t spatial_size, float dq_zp,
    float dq_sc, float q_zp, float q_sc, int activation, float leakyRelu_alpha);

/* Two outputs: ofm = quantize(y) with (q_zp, q_sc), no activation;
 * ofm2 = quantize(act(y)) with (q2_zp, q2_sc). split_mode 0..2. */
int npu_qdq_instance_norm_sd_opt_stage2_ifm_double_out(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
    const void* beta, npu_dtype compute_type, void* ofm, void* ofm2, int64_t hw,
    int64_t c, int64_t sv_hw, int split_mode, int64_t spatial_size, float dq_zp,
    float dq_sc, float q_zp, float q_sc, float q2_zp, float q2_sc,
    int activation, float leakyRelu_alpha);

/* Quantized instance norm with add_mul_add. add1_act / add2 (c each, ifm_type)
 * are dequantized with (dq2_zp, dq2_sc); add1_const (c) is compute_type.
 * split_mode 0..2. */
int npu_qdq_instance_norm_add_mul_add_sd(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
    const void* beta, npu_dtype compute_type, const void* add1_act,
    const void* add2, const void* add1_const, void* ofm, int64_t hw, int64_t c,
    int64_t sv_hw, int split_mode, int64_t spatial_size, float dq_zp,
    float dq_sc, float q_zp, float q_sc, float dq2_zp, float dq2_sc,
    int activation, float leakyRelu_alpha);
int npu_qdq_instance_norm_add_mul_add_sd_opt_stage2_ifm(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* alpha,
    const void* beta, npu_dtype compute_type, const void* add1_act,
    const void* add2, const void* add1_const, void* ofm, int64_t hw, int64_t c,
    int64_t sv_hw, int split_mode, int64_t spatial_size, float dq_zp,
    float dq_sc, float q_zp, float q_sc, float dq2_zp, float dq2_sc,
    int activation, float leakyRelu_alpha);

/* Streaming instance norm, one accumulator pair for even / odd rows. ifm / ofm
 * BF16 (hw, c), hw even; alpha / beta of alpha_type BF16 / F32; N = hw. */
int npu_instance_norm_stream_bf16(void* stream, const void* ifm,
                                  const void* alpha, const void* beta,
                                  npu_dtype alpha_type, void* ofm, int64_t hw,
                                  int64_t c);

/* Quantized streaming instance norm, bf16 compute; the coefficients in the
 * kernel's word order. ifm / ofm I8 / U8 / I16 / U16;
 * x = bf16(fma(q, dq_sc, dq_offset)) with dq_offset = fp32(-zp * dq_sc);
 * out = sat(rne(fma(max(y, y * relu_alpha), q_sc, q_zp))). */
int npu_qdq_instance_norm_stream(void* stream, const void* ifm,
                                 npu_dtype ifm_type, const void* alpha,
                                 const void* beta, npu_dtype alpha_type, void* ofm,
                                 int64_t hw, int64_t c, float dq_sc, float q_sc,
                                 float dq_offset, float q_zp, float relu_alpha);

/* Group norm. x (g, l) of ifm_type BF16 / F16 / FP8; alpha / beta (g * cg)
 * ifm_type, F16 for FP8. sv = chunk size, sv % 64 == 0. split_mode 0 / 1 / 2
 * (4 rows unless 2). activation 0 or 5. */
int npu_groupnorm_sd_bf16(void* stream, const void* ifm, npu_dtype ifm_type,
                          const void* alpha, const void* beta, void* ofm,
                          int64_t g, int64_t l, int64_t cg, int64_t sv,
                          int64_t div_factor, int split_mode, float epsilon,
                          int activation);

/* Quantized group norm. x (g, l) of ifm_type I8 / U8 / I16 / U16; alpha / beta of
 * compute_type (F32 / BF16 / F16). split_mode -1..3; activation 0 or 5. */
int npu_qdq_groupnorm_sd(void* stream, const void* ifm, npu_dtype ifm_type,
                         const void* alpha, const void* beta,
                         npu_dtype compute_type, void* ofm, int64_t g,
                         int64_t l, int64_t cg, int64_t sv, int64_t div_factor,
                         int split_mode, float epsilon, float dq_zp,
                         float dq_sc, float q_zp, float q_sc, int activation);

/* Quantized group norm, bf16 compute and BF16 alpha / beta, chunks summed
 * sequentially; N = l. */
int npu_qdq_groupnorm(void* stream, const void* ifm, npu_dtype ifm_type,
                      const void* alpha, const void* beta, void* ofm, int64_t g,
                      int64_t l, int64_t cg, int64_t sv, float epsilon,
                      float dq_zp, float dq_sc, float q_zp, float q_sc);

/* npu_qdq_instance_norm_add_mul_add_sd with given statistics: sum / sumsq are
 * device float arrays of c entries, spatial_size the element count N.
 * ifm_type I16 / U16; compute_type BF16 / F16. */
int npu_qdq_ConvInstanceNormAddMulAdd_Stage2(
    void* stream, const void* ifm, npu_dtype ifm_type, const void* sum,
    const void* sumsq, const void* alpha, const void* beta,
    npu_dtype compute_type, const void* add1_act, const void* add2,
    const void* add1_const, void* ofm, int64_t hw, int64_t c,
    int64_t spatial_size, float dq_zp, float dq_sc, float q_zp, float q_sc,
    float dq2_zp, float dq2_sc, int activation, float leakyRelu_alpha);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_GROUPNORM_H_
