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

/* Conv arguments as npu_conv_qdq_int16x8: conv_ifm (h, w, ci) I16 / U16
 * including the zero-point border, wts (kh, kw, ci, co) I8 / U8, c0 / c1 / c2
 * device arrays of co floats (c1 may be null). The conv output (ho, wo, co) of
 * conv_out_type (I16 / U16) is normalized as by npu_qdq_instance_norm_sd; ofm is
 * (ho * wo, co) of conv_out_type. The statistics sum blocks of sv_h x sv_w
 * pixels, h_loop x w_loop blocks per grid cell:
 *   split_mode 1: row k (0..3) starts at block (k*ho/4, 0)
 *   split_mode 0: column c, row k starts at column c*wo/3 + k*wo/12
 * N = cells * sv_h * sv_w * h_loop * w_loop - spatial_pad. */
int npu_qdq_ConvInFusion_A16(void* stream, const void* conv_ifm, npu_dtype ifm_type,
                             const void* wts, npu_dtype wts_type, int64_t h, int64_t w,
                             int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
                             int stride_w, const float* c0, const float* c1, const float* c2,
                             npu_activation conv_act, float act0, float act1,
                             npu_dtype conv_out_type, const void* alpha, const void* beta,
                             npu_dtype ab_type, void* ofm, npu_dtype compute_type,
                             int split_mode, int64_t sv_h, int64_t sv_w, int64_t h_loop,
                             int64_t w_loop, int64_t spatial_pad, float dq_scale, float dq_zp,
                             float q_scale, float q_zp, int activation, float slope);

/* As above with the add_mul_add of npu_qdq_instance_norm_add_mul_add_sd
 * (add1_act / add2 of conv_out_type, add1_const compute_type). */
int npu_qdq_ConvInstanceNormAddMulAdd(
    void* stream, const void* conv_ifm, npu_dtype ifm_type, const void* wts, npu_dtype wts_type,
    int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
    int stride_w, const float* c0, const float* c1, const float* c2, npu_activation conv_act,
    float act0, float act1, npu_dtype conv_out_type, const void* alpha, const void* beta,
    npu_dtype ab_type, const void* add1_act, const void* add2, const void* add1_const,
    void* ofm, npu_dtype compute_type, int split_mode, int64_t sv_h, int64_t sv_w,
    int64_t h_loop, int64_t w_loop, int64_t spatial_pad, float dq_scale, float dq_zp,
    float q_scale, float q_zp, float dq2_scale, float dq2_zp, int activation, float slope);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_CONV_INSTANCENORM_H_
