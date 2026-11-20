/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * The QDQ GridSample grid coordinate (npu_gridsample_qdq), shared by the nearest and the
 * bilinear GridSample kernels.
 */
#ifndef NPU_EMU_GRID_H_
#define NPU_EMU_GRID_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/npu_emu.h"

namespace npu_emu {

// Grid code -> fp32 pixel coordinate on an axis of `extent` pixels; each fp32 step rounds.
__device__ inline float qdq_grid_coord(int64_t code, const npu_gridsample_qdq& q, int64_t extent,
                                       int align_corners) {
  float g = __fmul_rn(__fsub_rn(float(code), float(q.grid_zp)), q.grid_scale);
  g = __fadd_rn(g, 1.0f);
  if (align_corners == 0) return fmaf(g, __fmul_rn(float(extent), 0.5f), -0.5f);  // one rounding
  return __fmul_rn(__fmul_rn(float(extent - 1), g), 0.5f);
}

}  // namespace npu_emu

#endif  // NPU_EMU_GRID_H_
