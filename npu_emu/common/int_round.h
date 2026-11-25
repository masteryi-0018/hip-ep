/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Rounding modes and the integer side of the datapath: shift-round, saturation, wrap,
 * shift-round-saturate, integer load / store.
 *
 * Special values: a shift amount of any int value is defined: a left shift of 64 or more
 * saturates, a right shift of 65 or more leaves only the sticky bit; sat_add64 saturates,
 * wrap_mul64 / wrap_add64 wrap; the int64 domain holds U64 values up to 2^63 - 1 only [default].
 */
#ifndef NPU_EMU_INT_ROUND_H_
#define NPU_EMU_INT_ROUND_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/npu_emu.h"

namespace npu_emu {

// Rounding modes. The kHalf* modes round to nearest and differ only on ties.
enum Rnd {
  kHalfEven = 0,      // ties to even
  kHalfAway = 1,      // ties away from zero
  kFloor = 2,         // toward -inf
  kCeil = 3,          // toward +inf
  kTrunc = 4,         // toward zero
  kAwayFromZero = 5,  // away from zero on any remainder
  kHalfDown = 6,      // ties toward -inf
  kHalfUp = 7,        // ties toward +inf
  kHalfZero = 8,      // ties toward zero
  kHalfOdd = 9,       // ties to odd
};

// Whether the truncated magnitude q of a value of sign `neg` is incremented: `odd` is the
// lowest kept bit of q, `half` the first dropped bit, `rest` any lower dropped bit.
__host__ __device__ inline bool round_up(Rnd rnd, bool neg, bool odd, bool half, bool rest) {
  switch (rnd) {
    case kFloor: return neg && (half || rest);
    case kCeil: return !neg && (half || rest);
    case kTrunc: return false;
    case kAwayFromZero: return half || rest;
    default: break;
  }
  if (!half) return false;
  if (rest) return true;
  switch (rnd) {
    case kHalfEven: return odd;
    case kHalfOdd: return !odd;
    case kHalfAway: return true;
    case kHalfZero: return false;
    case kHalfUp: return !neg;
    default: return neg;  // kHalfDown
  }
}

// Whether an overflow of a value of sign `neg` rounds to infinity (else to the largest
// finite value).
__host__ __device__ inline bool rounds_to_inf(Rnd rnd, bool neg) {
  if (rnd == kTrunc) return false;
  if (rnd == kFloor) return neg;
  if (rnd == kCeil) return !neg;
  return true;
}

// mag * 2^-s rounded with `rnd` for a value of sign `neg`. s <= 0 shifts left, saturating
// at 2^64 - 1.
__host__ __device__ inline uint64_t shift_round_mag(uint64_t mag, int s, Rnd rnd, bool neg) {
  if (s <= 0) {
    const int u = s < -64 ? 64 : -s;  // -INT_MIN would overflow
    if (u == 0 || mag == 0) return mag;
    if (u >= 64 || (mag >> (64 - u))) return ~0ull;
    return mag << u;
  }
  uint64_t q = 0;
  bool half, rest;
  if (s < 64) {
    q = mag >> s;
    half = (mag >> (s - 1)) & 1;
    rest = (mag & ((1ull << (s - 1)) - 1)) != 0;
  } else if (s == 64) {
    half = mag >> 63;
    rest = (mag & ~(1ull << 63)) != 0;
  } else {
    half = false;
    rest = mag != 0;
  }
  return q + (round_up(rnd, neg, q & 1, half, rest) ? 1 : 0);
}

// v * 2^-s rounded with `rnd`. s <= 0 shifts left, saturating to the int64 range.
__host__ __device__ inline int64_t shift_round(int64_t v, int s, Rnd rnd) {
  const bool neg = v < 0;
  const uint64_t q = shift_round_mag(neg ? 0 - uint64_t(v) : uint64_t(v), s, rnd, neg);
  if (neg) return q >= (1ull << 63) ? INT64_MIN : -int64_t(q);
  return q > uint64_t(INT64_MAX) ? INT64_MAX : int64_t(q);
}

// a + b saturated to the int64 range [default].
__host__ __device__ inline int64_t sat_add64(int64_t a, int64_t b) {
  int64_t r;
  if (__builtin_add_overflow(a, b, &r)) return a < 0 ? INT64_MIN : INT64_MAX;
  return r;
}

// a * b and a + b in two's complement, wrapping on overflow.
__host__ __device__ inline int64_t wrap_mul64(int64_t a, int64_t b) {
  return int64_t(uint64_t(a) * uint64_t(b));
}
__host__ __device__ inline int64_t wrap_add64(int64_t a, int64_t b) {
  return int64_t(uint64_t(a) + uint64_t(b));
}

// Range of the integer type t in the int64 domain of these helpers: U64 values of 2^63 and
// above are not representable (entries that need them handle the raw words).
__host__ __device__ inline void int_range(npu_dtype t, int64_t* lo, int64_t* hi) {
  switch (t) {
    case NPU_I8:
      *lo = -128, *hi = 127;
      break;
    case NPU_U8:
      *lo = 0, *hi = 255;
      break;
    case NPU_I16:
      *lo = -32768, *hi = 32767;
      break;
    case NPU_U16:
      *lo = 0, *hi = 65535;
      break;
    case NPU_I32:
      *lo = INT32_MIN, *hi = INT32_MAX;
      break;
    case NPU_U32:
      *lo = 0, *hi = 4294967295ll;
      break;
    case NPU_U64:
      *lo = 0, *hi = INT64_MAX;
      break;
    default:  // I64 (the helpers are only called with integer types)
      *lo = INT64_MIN, *hi = INT64_MAX;
      break;
  }
}

__host__ __device__ inline int64_t clamp64(int64_t v, int64_t lo, int64_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// v saturated to the range of t.
__host__ __device__ inline int64_t saturate(int64_t v, npu_dtype t) {
  int64_t lo, hi;
  int_range(t, &lo, &hi);
  return clamp64(v, lo, hi);
}

// v saturated symmetrically: a signed t clamps to [-max, max] (int8: [-127, 127]); an
// unsigned t as saturate.
__host__ __device__ inline int64_t saturate_sym(int64_t v, npu_dtype t) {
  int64_t lo, hi;
  int_range(t, &lo, &hi);
  return clamp64(v, lo < 0 ? -hi : lo, hi);
}

// v wrapped (two's complement) to the width of t.
__host__ __device__ inline int64_t wrap(int64_t v, npu_dtype t) {
  switch (t) {
    case NPU_I8:
      return int8_t(uint8_t(v));
    case NPU_U8:
      return uint8_t(v);
    case NPU_I16:
      return int16_t(uint16_t(v));
    case NPU_U16:
      return uint16_t(v);
    case NPU_I32:
      return int32_t(uint32_t(v));
    case NPU_U32:
      return uint32_t(v);
    default:
      return v;
  }
}

// v * 2^-s rounded with `rnd`, then saturated to t.
__host__ __device__ inline int64_t srs(int64_t v, int s, npu_dtype t, Rnd rnd = kHalfEven) {
  return saturate(shift_round(v, s, rnd), t);
}

// Element i of the integer type t; a U64 element is read as int64 (see int_range).
__device__ inline int64_t load_int(const void* p, npu_dtype t, int64_t i) {
  switch (t) {
    case NPU_I8:
      return static_cast<const int8_t*>(p)[i];
    case NPU_U8:
      return static_cast<const uint8_t*>(p)[i];
    case NPU_I16:
      return static_cast<const int16_t*>(p)[i];
    case NPU_U16:
      return static_cast<const uint16_t*>(p)[i];
    case NPU_I32:
      return static_cast<const int32_t*>(p)[i];
    case NPU_U32:
      return static_cast<const uint32_t*>(p)[i];
    default:
      return static_cast<const int64_t*>(p)[i];
  }
}

__device__ inline void store_int(void* p, npu_dtype t, int64_t i, int64_t v) {
  switch (t) {
    case NPU_I8:
    case NPU_U8:
      static_cast<uint8_t*>(p)[i] = uint8_t(v);
      break;
    case NPU_I16:
    case NPU_U16:
      static_cast<uint16_t*>(p)[i] = uint16_t(v);
      break;
    case NPU_I32:
    case NPU_U32:
      static_cast<uint32_t*>(p)[i] = uint32_t(v);
      break;
    default:
      static_cast<int64_t*>(p)[i] = v;
      break;
  }
}

}  // namespace npu_emu

#endif  // NPU_EMU_INT_ROUND_H_
