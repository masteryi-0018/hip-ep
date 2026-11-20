/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Group norm and instance norm of the NPU (bf16 / f16 / fp8 and quantized).
 *
 * Layouts are dense: instance norm (hw, c); group norm (g, l) with l = H*W*Cg
 * elements of a group in HWC order (alpha / beta have g*cg entries). hw / l
 * include spatial_pad padding rows, which enter the sums but not the count N.
 *
 * Statistics are fp32 partial sums of chunks of sv_hw / sv elements, combined
 * over a grid of columns x rows selected by split_mode:
 *   0: 3 x 4;  1: 1 x 4;  2: 1 x 2;  3: 3 x 1;  -1: 1 x 1
 * (rows of a column from last to first, then columns 0 -> 1 -> 2). The number
 * of chunks must be a multiple of columns * rows; other shapes return -1.
 *
 *   mu = S / N;  var = S2 / N + eps - mu * mu;  inv_std = 1 / sqrt(var)
 *   y = (x - mu) * inv_std * alpha + beta      (eps = 1e-5f unless a parameter)
 * QDQ entries: x = T(fma(q, dq_scale, fp32(-dq_zp * dq_scale))) with T the
 * compute type; out = sat(rne(fma(y, q_scale, q_zp))).
 * add_mul_add entries: y = y * (add1_act + add1_const) + add2.
 * activation: 0 none, 1 ReLU, 3 leaky ReLU (slope), 5 SiLU (group norm only).
 * Special values: fp32 steps follow the NaN rules of common/mac.h [default].
 */
#ifndef NPU_EMU_GROUPNORM_H_
#define NPU_EMU_GROUPNORM_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Instance norm. ifm / ofm io_type BF16 / F16 / FP8, (hw, c); alpha / beta
 * ab_type BF16 / F16 / F32 (c). activation 0 / 1 / 3. Both entries compute the
 * same result. */
int npu_instance_norm_sd_bf16(void* stream, const void* ifm, npu_dtype io_type,
                              const void* alpha, const void* beta, npu_dtype ab_type,
                              void* ofm, int64_t hw, int64_t c, int split_mode,
                              int64_t sv_hw, int64_t spatial_pad, int activation,
                              float slope);
/* See npu_instance_norm_sd_bf16. */
int npu_instance_norm_sd_opt_stage2_ifm_bf16(void* stream, const void* ifm,
                                             npu_dtype io_type, const void* alpha,
                                             const void* beta, npu_dtype ab_type, void* ofm,
                                             int64_t hw, int64_t c, int split_mode,
                                             int64_t sv_hw, int64_t spatial_pad,
                                             int activation, float slope);

/* As above with add_mul_add. add1_act / add2 (c each) are io_type,
 * add1_const (c) is ab_type. */
int npu_instance_norm_add_mul_add_sd_bf16(void* stream, const void* ifm, npu_dtype io_type,
                                          const void* alpha, const void* beta,
                                          npu_dtype ab_type, const void* add1_act,
                                          const void* add2, const void* add1_const,
                                          void* ofm, int64_t hw, int64_t c, int split_mode,
                                          int64_t sv_hw, int64_t spatial_pad,
                                          int activation, float slope);
/* See npu_instance_norm_add_mul_add_sd_bf16. */
int npu_instance_norm_add_mul_add_sd_opt_stage2_ifm_bf16(
    void* stream, const void* ifm, npu_dtype io_type, const void* alpha, const void* beta,
    npu_dtype ab_type, const void* add1_act, const void* add2, const void* add1_const,
    void* ofm, int64_t hw, int64_t c, int split_mode, int64_t sv_hw, int64_t spatial_pad,
    int activation, float slope);

/* Quantized instance norm. ifm / ofm io_type I8 / U8 / I16 / U16;
 * compute_type F32 / BF16 / F16; alpha / beta ab_type. activation 0 / 1 / 3. */
int npu_qdq_instance_norm_sd(void* stream, const void* ifm, npu_dtype io_type,
                             const void* alpha, const void* beta, npu_dtype ab_type, void* ofm,
                             int64_t hw, int64_t c, npu_dtype compute_type, int split_mode,
                             int64_t sv_hw, int64_t spatial_pad, float dq_scale, float dq_zp,
                             float q_scale, float q_zp, int activation, float slope);
/* See npu_qdq_instance_norm_sd. */
int npu_qdq_instance_norm_sd_opt_stage2_ifm(void* stream, const void* ifm, npu_dtype io_type,
                                            const void* alpha, const void* beta,
                                            npu_dtype ab_type, void* ofm, int64_t hw,
                                            int64_t c, npu_dtype compute_type, int split_mode,
                                            int64_t sv_hw, int64_t spatial_pad,
                                            float dq_scale, float dq_zp, float q_scale,
                                            float q_zp, int activation, float slope);

/* Two outputs: ofm = quantize(y) with (q_scale, q_zp), no activation;
 * ofm2 = quantize(act(y)) with (q2_scale, q2_zp). */
int npu_qdq_instance_norm_sd_opt_stage2_ifm_double_out(
    void* stream, const void* ifm, npu_dtype io_type, const void* alpha, const void* beta,
    npu_dtype ab_type, void* ofm, void* ofm2, int64_t hw, int64_t c, npu_dtype compute_type,
    int split_mode, int64_t sv_hw, int64_t spatial_pad, float dq_scale, float dq_zp,
    float q_scale, float q_zp, float q2_scale, float q2_zp, int activation, float slope);

/* Quantized instance norm with add_mul_add. add1_act / add2 (c each, io_type)
 * are dequantized with (dq2_scale, dq2_zp); add1_const (c) is compute_type. */
int npu_qdq_instance_norm_add_mul_add_sd(
    void* stream, const void* ifm, npu_dtype io_type, const void* alpha, const void* beta,
    npu_dtype ab_type, const void* add1_act, const void* add2, const void* add1_const,
    void* ofm, int64_t hw, int64_t c, npu_dtype compute_type, int split_mode, int64_t sv_hw,
    int64_t spatial_pad, float dq_scale, float dq_zp, float q_scale, float q_zp,
    float dq2_scale, float dq2_zp, int activation, float slope);
/* See npu_qdq_instance_norm_add_mul_add_sd. */
int npu_qdq_instance_norm_add_mul_add_sd_opt_stage2_ifm(
    void* stream, const void* ifm, npu_dtype io_type, const void* alpha, const void* beta,
    npu_dtype ab_type, const void* add1_act, const void* add2, const void* add1_const,
    void* ofm, int64_t hw, int64_t c, npu_dtype compute_type, int split_mode, int64_t sv_hw,
    int64_t spatial_pad, float dq_scale, float dq_zp, float q_scale, float q_zp,
    float dq2_scale, float dq2_zp, int activation, float slope);

/* Streaming instance norm, one accumulator pair for even / odd rows. ifm / ofm
 * BF16 (hw, c), hw even; alpha / beta BF16 / F32; N = hw. */
int npu_instance_norm_stream_bf16(void* stream, const void* ifm, const void* alpha,
                                  const void* beta, npu_dtype ab_type, void* ofm, int64_t hw,
                                  int64_t c);

/* Quantized streaming instance norm, bf16 compute. ifm / ofm I8 / U8 / I16 /
 * U16; x = bf16(fma(q, dq_scale, dq_offset)), dq_offset = fp32(-zp * scale);
 * out = sat(rne(fma(max(y, y * relu_alpha), q_scale, q_zp))). */
int npu_qdq_instance_norm_stream(void* stream, const void* ifm, npu_dtype io_type,
                                 const void* alpha, const void* beta, npu_dtype ab_type,
                                 void* ofm, int64_t hw, int64_t c, float dq_scale,
                                 float dq_offset, float q_scale, float q_zp, float relu_alpha);

/* Group norm. x (g, l) io_type BF16 / F16 / FP8, alpha / beta ab_type BF16 /
 * F16 (g * cg). sv = chunk size, sv % 64 == 0. split_mode 0 / 1 / 2 (4 rows
 * unless 2). activation 0 or 5. */
int npu_groupnorm_sd_bf16(void* stream, const void* ifm, npu_dtype io_type, const void* alpha,
                          const void* beta, npu_dtype ab_type, void* ofm, int64_t g,
                          int64_t l, int64_t cg, int split_mode, int64_t sv,
                          int64_t spatial_pad, float eps, int activation);

/* Quantized group norm. x (g, l) io_type I8 / U8 / I16 / U16; compute_type
 * F32 / BF16 / F16; alpha / beta ab_type. activation 0 or 5. */
int npu_qdq_groupnorm_sd(void* stream, const void* ifm, npu_dtype io_type, const void* alpha,
                         const void* beta, npu_dtype ab_type, void* ofm, int64_t g, int64_t l,
                         int64_t cg, npu_dtype compute_type, int split_mode, int64_t sv,
                         int64_t spatial_pad, float eps, float dq_scale, float dq_zp,
                         float q_scale, float q_zp, int activation);

/* Quantized group norm, bf16 compute, chunks summed sequentially; N = l.
 * alpha / beta BF16 / F16 / F32. */
int npu_qdq_groupnorm(void* stream, const void* ifm, npu_dtype io_type, const void* alpha,
                      const void* beta, npu_dtype ab_type, void* ofm, int64_t g, int64_t l,
                      int64_t cg, int64_t sv, float eps, float dq_scale, float dq_zp,
                      float q_scale, float q_zp);

/* npu_qdq_instance_norm_add_mul_add_sd with given statistics: sum / sumsq are
 * device arrays of c floats, n the element count N. */
int npu_qdq_ConvInstanceNormAddMulAdd_Stage2(
    void* stream, const void* ifm, npu_dtype io_type, const float* sum, const float* sumsq,
    const void* alpha, const void* beta, npu_dtype ab_type, const void* add1_act,
    const void* add2, const void* add1_const, void* ofm, int64_t hw, int64_t c, int64_t n,
    npu_dtype compute_type, float dq_scale, float dq_zp, float q_scale, float q_zp,
    float dq2_scale, float dq2_zp, int activation, float slope);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_GROUPNORM_H_
