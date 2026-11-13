/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernel launch helpers shared by the npu_emu translation units.
 */
#ifndef NPU_EMU_LAUNCH_H_
#define NPU_EMU_LAUNCH_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "npu_emu.h"

namespace npu_emu {

constexpr int kThreads = 256;

template <typename Kernel, typename... Args>
int launch(void* stream, int64_t work, Kernel k, Args... args) {
  if (work <= 0) return 0;
  (void)hipGetLastError();
  const int64_t blocks = (work + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(k, dim3(unsigned(blocks)), dim3(kThreads), 0,
                     static_cast<hipStream_t>(stream), work, args...);
  return int(hipGetLastError());
}

#define NPU_TID                                                  \
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; \
  if (i >= n) return

inline bool is_float(npu_dtype t) { return t == NPU_F32 || t == NPU_F16 || t == NPU_BF16; }
inline bool is_small_int(npu_dtype t) {
  return t == NPU_I8 || t == NPU_U8 || t == NPU_I16 || t == NPU_U16;
}
inline bool is_32(npu_dtype t) { return t == NPU_I32 || t == NPU_U32; }
inline bool is_64(npu_dtype t) { return t == NPU_I64 || t == NPU_U64; }

}  // namespace npu_emu

#endif  // NPU_EMU_LAUNCH_H_
