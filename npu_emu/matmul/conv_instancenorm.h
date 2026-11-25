/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Quantized convolution fused with instance norm: the conv of matmul/qdq_conv.h
 * followed by the quantized instance norm of reduction/groupnorm.h. The special-value rules are
 * those of the two parts: the conv output is integer (NaN -> 0, saturated), the statistics of a
 * constant image have zero variance.
 */
#ifndef NPU_EMU_CONV_INSTANCENORM_H_
#define NPU_EMU_CONV_INSTANCENORM_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* conv_ifm (h, w, ci) I16 / U16 including the zero-point border, wts (kh, kw, ci, co)
 * I8 / U8, c0 / c2 device arrays of co floats: the npu_conv_qdq_int16x8 convolution without an
 * activation sum or activation, into ho x wo x co values of ofm_type (I16 / U16). Those are
 * normalized as by npu_qdq_instance_norm_sd in float16 (alpha / beta: co float16 values) and
 * written to ofm (ho * wo, co) of ofm_type: dequantized with dq_zp / dq_sc, the leaky ReLU
 * (activation 3) with slope, requantized with q_zp / q_sc (q_sc = 1 / ofm scale).
 * The statistics sum blocks of sv_h x sv_w pixels, h_loop x w_loop blocks per grid cell:
 *   split_mode 1: row k (0..3) starts at block (k*ho/4, 0)
 *   split_mode 0: column c, row k starts at column c*wo/3 + k*wo/12
 * N = cells * sv_h * sv_w * h_loop * w_loop - spatial_pad. h or w 0: an empty output, nothing
 * is launched. Every pointer is required. */
int npu_qdq_ConvInFusion_A16(void* stream, const void* conv_ifm, npu_dtype ifm_type,
                             const void* wts, npu_dtype wts_type, const void* alpha,
                             const void* beta, void* ofm, npu_dtype ofm_type, int64_t h,
                             int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                             int stride_h, int stride_w, int split_mode, int64_t sv_h,
                             int64_t sv_w, int64_t h_loop, int64_t w_loop, int64_t spatial_pad,
                             const float* c0, const float* c2, float dq_zp, float dq_sc,
                             float q_zp, float q_sc, float slope);

/* As above with the add_mul_add of npu_qdq_instance_norm_add_mul_add_sd: add1_act and add2
 * (co values of ofm_type, dequantized with dq_ifm2_zp / dq_ifm2_sc) and add1_const (co
 * float16 values). */
int npu_qdq_ConvInstanceNormAddMulAdd(
    void* stream, const void* conv_ifm, npu_dtype ifm_type, const void* wts, npu_dtype wts_type,
    const void* alpha, const void* beta, const void* add1_act, const void* add2,
    const void* add1_const, void* ofm, npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci,
    int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w, int split_mode, int64_t sv_h,
    int64_t sv_w, int64_t h_loop, int64_t w_loop, int64_t spatial_pad, const float* c0,
    const float* c2, float dq_zp, float dq_sc, float q_zp, float q_sc, float dq_ifm2_zp,
    float dq_ifm2_sc, float slope);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_CONV_INSTANCENORM_H_
