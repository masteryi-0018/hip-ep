/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Internals of groupnorm.hip for kernels that fuse the instance norm: the block
 * plan of the instance-norm statistics and the quantized instance norm.
 */
#ifndef NPU_EMU_GROUPNORM_DEVICE_H_
#define NPU_EMU_GROUPNORM_DEVICE_H_

#include <stdint.h>

#include "common/npu_emu.h"

namespace npu_emu {

// Block plan of the statistics: the spatial axis is an image of width W cut
// into blocks of sv_h x sv_w pixels (nbw per block row). Cell (column c, row k)
// adds, for step j < T (ih = j / w_loop, iw = j % w_loop), block
//   (c*cbh + k*kbh + ih*jbh, c*cbw + k*kbw + iw).
struct Plan {
  int ncol, nrow, T, w_loop;
  int64_t cbh, kbh, jbh, cbw, kbw;
  int64_t nbw, nbh, sv_h, sv_w, W;
};

// Quantized instance norm of ifm (hw, c): statistics by `plan`, or the given
// sum / sumsq when plan is null; sum_on_mmac: the x sums of the plan are MMAC
// accumulations instead of fp32 adds.
int in_qdq(void* stream, const void* ifm, npu_dtype io, const void* alpha, const void* beta,
           npu_dtype abt, const void* add1_act, const void* add2, const void* add1_const,
           bool amad, void* ofm, void* ofm2, int64_t hw, int64_t c, npu_dtype ct,
           const Plan* plan, const float* sum, const float* sumsq, int64_t N, float dq_sc,
           float dq_zp, float q_sc, float q_zp, float q2_sc, float q2_zp, float dq2_sc,
           float dq2_zp, int act, float slope, bool sum_on_mmac = false);

}  // namespace npu_emu

#endif  // NPU_EMU_GROUPNORM_DEVICE_H_
