/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernel launch helpers, dtype predicates and broadcast indexing shared by the npu_emu
 * translation units.
 */
#ifndef NPU_EMU_LAUNCH_H_
#define NPU_EMU_LAUNCH_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/npu_emu.h"

namespace npu_emu {

constexpr int kThreads = 256;

template <typename Kernel, typename... Args>
int launch(void* stream, int64_t work, Kernel k, Args... args) {
  if (work <= 0) return 0;
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

// F32 / F16 / BF16: the float types load_float reads as plain elements (not FP8).
__host__ __device__ inline bool is_plain_float(npu_dtype t) {
  return t == NPU_F32 || t == NPU_F16 || t == NPU_BF16;
}
__host__ __device__ inline bool is_small_int(npu_dtype t) {
  return t == NPU_I8 || t == NPU_U8 || t == NPU_I16 || t == NPU_U16;
}
__host__ __device__ inline bool is_i8(npu_dtype t) { return t == NPU_I8 || t == NPU_U8; }
__host__ __device__ inline bool is_i16(npu_dtype t) { return t == NPU_I16 || t == NPU_U16; }
__host__ __device__ inline bool is_int_bits(npu_dtype t, int bits) {
  return bits == 8 ? is_i8(t) : is_i16(t);
}
__host__ __device__ inline bool is_half_float(npu_dtype t) { return t == NPU_BF16 || t == NPU_F16; }
__host__ __device__ inline bool is_32(npu_dtype t) { return t == NPU_I32 || t == NPU_U32; }
__host__ __device__ inline bool is_64(npu_dtype t) { return t == NPU_I64 || t == NPU_U64; }

// HWC broadcasting. Shapes are host arrays {H, W, C}; every dimension of a broadcast operand
// is 1 or the full one.

// The full shape of two operands (per dimension the larger); false if they do not broadcast.
inline bool bcast_shape(const int64_t* a, const int64_t* b, int64_t* full) {
  if (!a || !b) return false;
  for (int d = 0; d < 3; ++d) {
    if (a[d] < 1 || b[d] < 1 || (a[d] != b[d] && a[d] != 1 && b[d] != 1)) return false;
    full[d] = a[d] > b[d] ? a[d] : b[d];
  }
  return true;
}

// Broadcast bits of an operand with dimensions bH x bW x bC (each 1 or the full one).
__host__ __device__ inline int dims_bits(int64_t bH, int64_t bW, int64_t bC) {
  return (bC == 1 ? 1 : 0) | (bW == 1 ? 2 : 0) | (bH == 1 ? 4 : 0);
}

// Broadcast bits of an operand of shape s within the full shape f: bit 0 / 1 / 2 = size 1
// along C / W / H; -1 if s does not broadcast to f.
inline int bcast_bits(const int64_t* f, const int64_t* s) {
  if (!f || !s) return -1;
  int bits = 0;
  for (int d = 0; d < 3; ++d) {
    if (s[d] != f[d] && s[d] != 1) return -1;
    if (s[d] == 1) bits |= 1 << (2 - d);
  }
  return bits;
}

// Element (h, w, c) of an operand broadcast along the axes in `bcast`
// (bit 0 / 1 / 2 = C / W / H).
struct BIdx {
  int64_t pixel;  // flat index of channel 0 of the pixel
  int64_t c;      // channel within the operand
  int64_t cn;     // channels of the operand
};
__host__ __device__ inline BIdx bidx(int64_t h, int64_t w, int64_t c, int bcast, int64_t W,
                                     int64_t C) {
  const int64_t wn = bcast & 2 ? 1 : W, cn = bcast & 1 ? 1 : C;
  const int64_t hh = bcast & 4 ? 0 : h, ww = bcast & 2 ? 0 : w;
  return {(hh * wn + ww) * cn, bcast & 1 ? 0 : c, cn};
}

}  // namespace npu_emu

#endif  // NPU_EMU_LAUNCH_H_
