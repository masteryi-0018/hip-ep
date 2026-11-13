/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Device helpers of the arith_misc kernels: the AIE4 bf16 / fp16 reciprocal
 * units, computed exactly from integer division instead of a 64K-entry table.
 */
#ifndef NPU_EMU_ARITH_MISC_DEVICE_H_
#define NPU_EMU_ARITH_MISC_DEVICE_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "npu_emu_device.h"

namespace npu_emu {

// mag * 2^E (mag > 0) rounded half to even to an unsigned float code with
// `ebits` exponent bits, `mbits` mantissa bits and bias `bias`: subnormals
// kept, overflow gives infinity. The lowest bit of mag may be a sticky bit;
// mag must have more than mbits + 2 significant bits.
__device__ inline uint32_t round_to_code(uint64_t mag, int E, int ebits, int mbits, int bias) {
  const int L = 64 - __clzll(mag);
  const int top = L - 1 + E;  // exponent of the leading bit
  const int emin = 1 - bias;
  const bool sub = top < emin;
  const int ulp = (sub ? emin : top) - mbits;
  uint64_t q = round_shift(mag, ulp - E, kEven);
  if (sub) return uint32_t(q);  // q == 1 << mbits is the smallest normal: same code
  int biased = top + bias;
  if (q >> (mbits + 1)) {  // rounding carried into the next binade
    q >>= 1;
    ++biased;
  }
  const uint32_t emax = (1u << ebits) - 1;
  if (biased >= int(emax)) return emax << mbits;
  return (uint32_t(biased) << mbits) | uint32_t(q & ((1u << mbits) - 1));
}

// RNE(1 / y) as an unsigned code of the format above, for a finite nonzero
// fp32 y. 1/y = 2^62 / m * 2^(-62 - e) with y = m * 2^e: the integer quotient
// keeps at least 38 bits and the remainder becomes the sticky bit.
__device__ inline uint32_t recip_code(float y, int ebits, int mbits, int bias) {
  const uint32_t b = __float_as_uint(y);
  const int f = (b >> 23) & 0xff;
  const uint64_t m = f ? ((b & 0x7fffff) | 0x800000u) : (b & 0x7fffff);
  const int e = f ? f - 150 : -149;
  const uint64_t num = 1ull << 62;
  const uint64_t q = num / m, r = num % m;
  return round_to_code((q << 1) | (r != 0), -63 - e, ebits, mbits, bias);
}

// The AIE4 bf16 reciprocal (aie::inv / aie::div on bf16, mllib inv_lut.py
// inv_bf16_aie4): RNE(1/y), except that the unit treats bf16 subnormals as
// zero on both sides - a zero or subnormal y gives +-inf and a result that
// would be subnormal (|y| > 2^126) gives +-0. NaN stays NaN (quieted; the NPU
// payload is unverified).
__device__ inline uint16_t inv_bf16_aie4(uint16_t h) {
  const uint16_t s = h & 0x8000;
  const int e = (h >> 7) & 0xff;
  if (e == 0) return s | 0x7f80;
  if (e == 0xff) return (h & 0x7f) ? uint16_t(h | 0x40) : s;
  const uint32_t code = recip_code(bf16_to_f32(h), 8, 7, 127);
  if ((code >> 7) == 0) return s;  // subnormal result -> 0
  return s | uint16_t(code);
}

// The AIE4 fp16 reciprocal (inv_lut.py inv_fp16_aie4): RNE(1/y) in fp16 with
// subnormals; +-0 -> +-inf, +-inf -> +-0, NaN -> 0x7e00.
__device__ inline uint16_t inv_f16_aie4(uint16_t h) {
  const uint16_t s = h & 0x8000;
  const int e = (h >> 10) & 0x1f;
  if ((h & 0x7fff) == 0) return s | 0x7c00;
  if (e == 0x1f) return (h & 0x3ff) ? 0x7e00 : s;
  return s | uint16_t(recip_code(f16_to_f32(h), 5, 10, 15));
}

}  // namespace npu_emu

#endif  // NPU_EMU_ARITH_MISC_DEVICE_H_
