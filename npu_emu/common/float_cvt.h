/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Every float format conversion: fp32 <-> bf16 / f16 / fp8 (E4M3FN) through one encoder with
 * a rounding mode and flush / saturation flags, the decoders, float <-> integer, host scalar
 * casts.
 *
 * Special values ([measured] against the NPU, [unverified] otherwise): kFtzOut flushes after
 * rounding [measured, 16-bit results]; kFtzIn flushes fp32-subnormal operands [measured, bf16
 * conversion]; overflow follows the rounding direction and fp8 overflows to NaN unless kSat
 * [measured]; a NaN keeps its sign and top payload bits [unverified]; scalar_cast leaves inf
 * and NaN unchanged [unverified]; decoding a half-precision NaN keeps its sign and payload
 * bits (no quieting) [default]; float_to_int saturates out-of-range values and infinities
 * [measured] and gives 0 for NaN [unverified]; int_to_float keeps subnormals and overflows
 * per the rounding direction, for an exponent of any int value [default].
 */
#ifndef NPU_EMU_FLOAT_CVT_H_
#define NPU_EMU_FLOAT_CVT_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/bits.h"
#include "common/int_round.h"
#include "common/npu_emu.h"

namespace npu_emu {

// Conversion flags.
enum CvtFlag {
  kFtzIn = 1,   // an fp32-subnormal input (or decoded value) becomes a zero of its sign
  kFtzOut = 2,  // a result that rounds to a subnormal of the format (subnormals kept while
                // rounding) becomes a zero of its sign
  kSat = 4,     // overflow and infinity give the largest finite value (NaN stays NaN)
};

struct FloatFmt {
  int ebits, mbits, bias;
  bool fn;  // E4M3FN: no infinity, all-ones is NaN
};

// Format of the float type t (F32 / BF16 / F16 / FP8; only float types are valid here).
__host__ __device__ inline FloatFmt float_fmt(npu_dtype t) {
  switch (t) {
    case NPU_BF16:
      return {8, 7, 127, false};
    case NPU_F16:
      return {5, 10, 15, false};
    case NPU_FP8:
      return {4, 3, 7, true};
    default:  // F32
      return {8, 23, 127, false};
  }
}

// An fp32 subnormal becomes a zero of its sign.
__host__ __device__ inline float ftz(float v) {
  const uint32_t b = f32_bits(v);
  return (b & 0x7f800000u) ? v : bits_f32(b & 0x80000000u);
}

// mag * 2^E (mag > 0) rounded with `rnd` to the magnitude code of f for a value of sign
// `neg`, subnormals kept. The lowest bit of mag may be a sticky bit when mag has more than
// mbits + 2 significant bits. A result past the largest finite value is returned as a larger
// code (the exponent field stops one past its maximum); the caller picks infinity or
// saturation.
__host__ __device__ inline uint32_t encode_mag(uint64_t mag, int E, FloatFmt f, Rnd rnd,
                                               bool neg) {
  const int L = 64 - __builtin_clzll(mag);
  const int te = L - 1 + E + f.bias;  // biased exponent of the leading bit
  uint64_t q = shift_round_mag(mag, L - 1 - f.mbits + (te >= 1 ? 0 : 1 - te), rnd, neg);
  if (te < 1) return uint32_t(q);  // subnormal; q == 1 << mbits is the smallest normal
  int ee = te;
  if (q >= (2ull << f.mbits)) {  // rounding carried into the exponent
    q >>= 1;
    ++ee;
  }
  if (ee > (1 << f.ebits)) ee = 1 << f.ebits;
  return (uint32_t(ee) << f.mbits) | uint32_t(q & ((1u << f.mbits) - 1));
}

// fp32 -> code of t (F32 / BF16 / F16 / FP8) rounded with `rnd`, subnormals kept unless
// flushed (F32: the bits, after the flags).
// Overflow gives infinity (FP8: NaN), or the largest finite value when `rnd` rounds toward
// zero for the sign or with kSat. A NaN keeps its sign and top mantissa bits (at least
// one set); FP8 NaN is 0x7f | sign.
__host__ __device__ inline uint32_t encode_float(float x, npu_dtype t, Rnd rnd, int flags = 0) {
  const FloatFmt f = float_fmt(t);
  const uint32_t b = f32_bits(x);
  const bool neg = b >> 31;
  const uint32_t sbit = uint32_t(neg) << (f.ebits + f.mbits);
  const int e = (b >> 23) & 0xff;
  const uint32_t m = b & 0x7fffff;
  const uint32_t emax = (1u << f.ebits) - 1, inf = emax << f.mbits;
  const uint32_t nan = f.fn ? inf | ((1u << f.mbits) - 1) : inf | (1u << (f.mbits - 1));
  const uint32_t max_finite = f.fn ? nan - 1 : inf - 1;
  if (e == 0xff) {
    if (m && f.fn) return sbit | nan;
    if (m) {
      const uint32_t pay = m >> (23 - f.mbits);
      return sbit | inf | (pay ? pay : 1u);
    }
    if (flags & kSat) return sbit | max_finite;
    return sbit | (f.fn ? nan : inf);
  }
  if (e == 0 && (m == 0 || (flags & kFtzIn))) return sbit;
  const uint64_t sig = e ? (m | 0x800000u) : m;
  const uint32_t code = encode_mag(sig, (e ? e - 127 : -126) - 23, f, rnd, neg);
  if ((flags & kFtzOut) && code < (1u << f.mbits)) return sbit;
  if (code > max_finite) {
    if ((flags & kSat) || !rounds_to_inf(rnd, neg)) return sbit | max_finite;
    return sbit | (f.fn ? nan : inf);
  }
  return sbit | code;
}

// Code of t (BF16 / F16 / FP8 / F32) as fp32 (exact); kFtzIn flushes an fp32-subnormal
// value. FP8 NaN decodes to 0x7fc00000 | sign.
__host__ __device__ inline float decode_float(uint32_t c, npu_dtype t, int flags = 0) {
  float v;
  if (t == NPU_F32 || t == NPU_BF16) {
    v = bits_f32(t == NPU_F32 ? c : c << 16);
  } else if (t == NPU_F16) {
    const uint32_t s = uint32_t((c >> 15) & 1) << 31;
    const int e = (c >> 10) & 0x1f;
    const uint32_t m = c & 0x3ff;
    if (e == 0x1f)
      v = bits_f32(s | 0x7f800000u | (m << 13));
    else if (e == 0)
      v = bits_f32(s | f32_bits(float(m) * 0x1p-24f));
    else
      v = bits_f32(s | (uint32_t(e - 15 + 127) << 23) | (m << 13));
  } else {
    const int e = (c >> 3) & 0xf, m = c & 7;
    if (e == 0xf && m == 7)
      v = qnan();
    else if (e == 0)
      v = float(m) * 0x1p-9f;
    else
      v = float(8 + m) * bits_f32(uint32_t(e - 10 + 127) << 23);
    if (c & 0x80) v = neg_f32(v);
  }
  return flags & kFtzIn ? ftz(v) : v;
}

// v rounded to t (F32 / BF16 / F16 / FP8) and back as fp32.
__host__ __device__ inline float round_float(float v, npu_dtype t, Rnd rnd, int flags = 0) {
  return decode_float(encode_float(v, t, rnd, flags), t);
}

// The accumulator read-out to bf16: half away from zero, an fp32-subnormal value flushed.
__host__ __device__ inline float acc_to_bf16(float v) {
  return round_float(v, NPU_BF16, kHalfAway, kFtzIn);
}

// Smallest normal and largest finite value of the float type t, as fp32.
__host__ __device__ inline float min_normal(npu_dtype t) {
  return decode_float(1u << float_fmt(t).mbits, t);
}
__host__ __device__ inline float max_finite(npu_dtype t) {
  const FloatFmt f = float_fmt(t);
  const uint32_t top = ((1u << f.ebits) - 1) << f.mbits;  // infinity, or the NaN row of E4M3FN
  return decode_float(f.fn ? top + (1u << f.mbits) - 2 : top - 1, t);
}

// The 16-bit code that indexes a 16-bit function unit with an fp32 v: bf16 takes the top 16
// bits (no rounding), f16 rounds half to even.
__host__ __device__ inline uint16_t code16(float v, npu_dtype t) {
  return t == NPU_F16 ? uint16_t(encode_float(v, NPU_F16, kHalfEven)) : uint16_t(f32_bits(v) >> 16);
}

__host__ __device__ inline uint16_t f32_to_bf16(float x, Rnd rnd, int flags = 0) {
  return uint16_t(encode_float(x, NPU_BF16, rnd, flags));
}
__host__ __device__ inline uint16_t f32_to_f16(float x, Rnd rnd, int flags = 0) {
  return uint16_t(encode_float(x, NPU_F16, rnd, flags));
}
__host__ __device__ inline uint8_t f32_to_fp8(float x, Rnd rnd, int flags = 0) {
  return uint8_t(encode_float(x, NPU_FP8, rnd, flags));
}
__host__ __device__ inline float bf16_to_f32(uint16_t h, int flags = 0) {
  return decode_float(h, NPU_BF16, flags);
}
__host__ __device__ inline float f16_to_f32(uint16_t h, int flags = 0) {
  return decode_float(h, NPU_F16, flags);
}
__host__ __device__ inline float fp8_to_f32(uint8_t c, int flags = 0) {
  return decode_float(c, NPU_FP8, flags);
}

// Element i of type t (F32 / F16 / BF16) as fp32. FP8 is not accepted (a NaN is returned):
// a kernel decodes fp8 either directly (fp8_to_f32) or through the identity multiply
// (fp8_decode_identity in mmac.h), so the caller picks.
__device__ inline float load_float(const void* p, npu_dtype t, int64_t i, int flags = 0) {
  if (t == NPU_FP8) return qnan();
  if (t == NPU_F32) return decode_float(f32_bits(static_cast<const float*>(p)[i]), t, flags);
  return decode_float(static_cast<const uint16_t*>(p)[i], t, flags);
}

// v rounded to t (F32 / BF16 / F16 / FP8) and stored.
__device__ inline void store_float(void* p, npu_dtype t, int64_t i, float v, Rnd rnd,
                                   int flags = 0) {
  const uint32_t c = encode_float(v, t, rnd, flags);
  if (t == NPU_F32)
    static_cast<uint32_t*>(p)[i] = c;
  else if (t == NPU_FP8)
    static_cast<uint8_t*>(p)[i] = uint8_t(c);
  else
    static_cast<uint16_t*>(p)[i] = uint16_t(c);
}

// ---- float <-> integer ----

// v * 2^shift rounded with `rnd` and saturated to t. NaN -> 0. A scaling that underflows the
// double range leaves a zero.
__host__ __device__ inline int64_t float_to_int(float v, npu_dtype t, int shift, Rnd rnd) {
  if (v != v) return 0;
  const double x = ldexp(double(v), shift);
  const double a = fabs(x), q = floor(a), d = a - q;  // d is exact
  const bool half = d >= 0.5, rest = half ? d > 0.5 : d > 0.0;
  double r = q + (round_up(rnd, x < 0, fmod(q, 2.0) != 0.0, half, rest) ? 1.0 : 0.0);
  if (x < 0) r = -r;
  int64_t lo = INT64_MIN, hi = INT64_MAX;
  if (t == NPU_U64) lo = 0;
  if (t != NPU_I64 && t != NPU_U64) int_range(t, &lo, &hi);
  if (r <= double(lo)) return lo;
  if (r >= double(hi)) return hi;
  return int64_t(r);
}

// v * 2^e rounded with `rnd` to fp32 (subnormals kept; overflow per rounds_to_inf).
__host__ __device__ inline float int_to_float(int64_t v, int e, Rnd rnd) {
  if (v == 0) return 0.0f;
  const bool neg = v < 0;
  const uint64_t mag = neg ? 0 - uint64_t(v) : uint64_t(v);
  // beyond +-4096 every int64 has already overflowed / left only the sticky bit
  e = e < -4096 ? -4096 : (e > 4096 ? 4096 : e);
  uint32_t code = encode_mag(mag, e, float_fmt(NPU_F32), rnd, neg);
  if (code >= 0x7f800000u) code = rounds_to_inf(rnd, neg) ? 0x7f800000u : 0x7f7fffffu;
  return bits_f32(code | (neg ? 0x80000000u : 0u));
}

// The integer v as fp32, round half to even.
__host__ __device__ inline float int_f32(int64_t v) { return int_to_float(v, 0, kHalfEven); }

// ---- host scalar casts (also callable on the device) ----

// Scalar cast of a coefficient to t (BF16 / F16), returned as fp32: bf16 rounds half to even
// whatever `rnd`, f16 follows `rnd`. Inf and NaN are returned unchanged.
__host__ __device__ inline float scalar_cast(float v, npu_dtype t, Rnd rnd) {
  if ((f32_bits(v) & 0x7f800000u) == 0x7f800000u) return v;
  return decode_float(encode_float(v, t, t == NPU_F16 ? rnd : kHalfEven), t);
}

// The (int32) cast of a float coefficient: truncated toward zero, saturated, NaN -> 0
// [default].
__host__ __device__ inline int32_t scalar_cast_i32(float v) {
  return int32_t(float_to_int(v, NPU_I32, 0, kTrunc));
}

}  // namespace npu_emu

#endif  // NPU_EMU_FLOAT_CVT_H_
