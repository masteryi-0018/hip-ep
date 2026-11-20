/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Vector compare and max / min: fp32 and 16-bit (bf16 / f16 codes).
 *
 * Special values ([measured] against the NPU, [unverified] otherwise): vmax / vmin order -0
 * below +0 and give 0x7fc00000 for a NaN operand [measured]; the 16-bit forms (canonical NaN,
 * equal values keep a), `above` and the compares' NaN rules carry these over [unverified].
 */
#ifndef NPU_EMU_MINMAX_H_
#define NPU_EMU_MINMAX_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/float_cvt.h"

namespace npu_emu {

// fp32 max / min: -0 orders below +0; a NaN operand gives 0x7fc00000.
__host__ __device__ inline float vmax(float a, float b) {
  if (a != a || b != b) return qnan();
  if (a == b) return f32_bits(a) >> 31 ? b : a;
  return a > b ? a : b;
}
__host__ __device__ inline float vmin(float a, float b) {
  if (a != a || b != b) return qnan();
  if (a == b) return f32_bits(a) >> 31 ? a : b;
  return a < b ? a : b;
}

// Vector compares (IEEE order: -0 == +0, a NaN compares false).
__host__ __device__ inline bool veq(float a, float b) { return a == b; }
__host__ __device__ inline bool vgt(float a, float b) { return a > b; }
__host__ __device__ inline bool vge(float a, float b) { return a >= b; }

// a strictly above b in the vmax order (-0 below +0); a NaN is never above and nothing is
// above it (a compare-and-select max keeps its running value on a NaN).
__host__ __device__ inline bool above(float a, float b) {
  return a == a && b == b && f32_bits(vmax(a, b)) != f32_bits(b);
}

// vmin(vmax(v, lo), hi).
__host__ __device__ inline float vclamp(float v, float lo, float hi) {
  return vmin(vmax(v, lo), hi);
}

// 16-bit max / min of t codes (BF16 / F16): -0 orders below +0, equal values keep a, a NaN
// operand gives the canonical NaN (0x7fc0 / 0x7e00).
__host__ __device__ inline uint16_t max16(uint16_t a, uint16_t b, npu_dtype t) {
  const float fa = decode_float(a, t), fb = decode_float(b, t);
  if (fa != fa || fb != fb) return t == NPU_F16 ? 0x7e00 : 0x7fc0;
  return f32_bits(vmax(fa, fb)) == f32_bits(fa) ? a : b;
}
__host__ __device__ inline uint16_t min16(uint16_t a, uint16_t b, npu_dtype t) {
  const float fa = decode_float(a, t), fb = decode_float(b, t);
  if (fa != fa || fb != fb) return t == NPU_F16 ? 0x7e00 : 0x7fc0;
  return f32_bits(vmin(fa, fb)) == f32_bits(fa) ? a : b;
}

}  // namespace npu_emu

#endif  // NPU_EMU_MINMAX_H_
