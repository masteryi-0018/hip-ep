/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernel launch shared by the npu_emu translation units.
 *
 * Special values: a negative work count is rejected (-1), a zero one returns 0 without
 * launching [default].
 */
#ifndef NPU_EMU_LAUNCH_H_
#define NPU_EMU_LAUNCH_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace npu_emu {

constexpr int kThreads = 256;

template <typename Kernel, typename... Args>
int launch(void* stream, int64_t work, Kernel k, Args... args) {
  if (work < 0) return -1;
  if (work == 0) return 0;
  const int64_t blocks = (work + kThreads - 1) / kThreads;
  if (blocks > int64_t(UINT32_MAX)) return -1;
  (void)hipGetLastError();
  hipLaunchKernelGGL(k, dim3(unsigned(blocks)), dim3(kThreads), 0,
                     static_cast<hipStream_t>(stream), work, args...);
  return int(hipGetLastError());
}

#define NPU_TID                                                  \
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; \
  if (i >= n) return

}  // namespace npu_emu

#endif  // NPU_EMU_LAUNCH_H_
