/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * HWC broadcasting. Shapes are host arrays {H, W, C}; every dimension of a broadcast operand
 * is 1 or the full one.
 *
 * Special values: bcast_bits rejects a null shape (-1) and does not check the dimensions to be
 * positive; bcast_check also rejects a negative dimension and accepts a zero one (an empty
 * tensor) [default].
 */
#ifndef NPU_EMU_BCAST_H_
#define NPU_EMU_BCAST_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace npu_emu {

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

// bcast_bits for a full shape f with every dimension >= 0; -1 otherwise.
inline int bcast_check(const int64_t* f, const int64_t* s) {
  if (!f || f[0] < 0 || f[1] < 0 || f[2] < 0) return -1;
  return bcast_bits(f, s);
}

// Element (h, w, c) of an operand broadcast along the axes in `bcast`
// (bit 0 / 1 / 2 = C / W / H).
struct BIdx {
  int64_t pixel;  // flat index of channel 0 of the pixel
  int64_t c;      // channel within the operand
  int64_t cn;     // channels of the operand
  __host__ __device__ int64_t flat() const { return pixel + c; }
};
__host__ __device__ inline BIdx bidx(int64_t h, int64_t w, int64_t c, int bcast, int64_t W,
                                     int64_t C) {
  const int64_t wn = bcast & 2 ? 1 : W, cn = bcast & 1 ? 1 : C;
  const int64_t hh = bcast & 4 ? 0 : h, ww = bcast & 2 ? 0 : w;
  return {(hh * wn + ww) * cn, bcast & 1 ? 0 : c, cn};
}
// The same for flat index i of the full [H][W][C] tensor.
__host__ __device__ inline BIdx bidx_flat(int64_t i, int bcast, int64_t W, int64_t C) {
  return bidx(i / (C * W), (i / C) % W, i % C, bcast, W, C);
}

}  // namespace npu_emu

#endif  // NPU_EMU_BCAST_H_
