/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * fp32 <-> bf16 / f16 / fp8 (E4M3FN) conversion: one encoder with a rounding mode and
 * flush / saturation flags, the decoders, the fp8 identity decode, host scalar casts.
 *
 * Special values ([measured] against the NPU, [unverified] otherwise): kFtzOut flushes after
 * rounding [measured, 16-bit results]; kFtzIn flushes fp32-subnormal operands [measured, bf16
 * conversion]; overflow follows the rounding direction and fp8 overflows to NaN unless kSat
 * [measured]; a NaN keeps its sign and top payload bits [unverified]; the identity-multiply
 * fp8 decode spreads a NaN over its group of 8 and turns -0 into +0 [measured]; scalar_cast
 * leaves inf and NaN unchanged [unverified]; decoding a half-precision NaN keeps its sign and
 * payload bits (no quieting) [default].
 */
#ifndef NPU_EMU_FLOAT_CVT_H_
#define NPU_EMU_FLOAT_CVT_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

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

__host__ __device__ inline uint32_t f32_bits(float x) { return __builtin_bit_cast(uint32_t, x); }
__host__ __device__ inline float bits_f32(uint32_t b) { return __builtin_bit_cast(float, b); }

__host__ __device__ inline float qnan() { return bits_f32(0x7fc00000u); }

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
// mbits + 2 significant bits. A result past the largest finite value is returned as is (a
// larger code); the caller picks infinity or saturation.
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
    if (c & 0x80) v = -v;
  }
  return flags & kFtzIn ? ftz(v) : v;
}

// v rounded to t (F32 / BF16 / F16 / FP8) and back as fp32.
__host__ __device__ inline float round_float(float v, npu_dtype t, Rnd rnd, int flags = 0) {
  return decode_float(encode_float(v, t, rnd, flags), t);
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
// (fp8_decode_identity), so the caller picks.
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

// fp8 values are decoded in rows of 8 consecutive elements: whether the row of element i
// (aligned to the buffer start, n elements) holds a NaN.
__device__ inline bool fp8_row_nan(const uint8_t* p, int64_t i, int64_t n) {
  const int64_t r = i & ~int64_t(7);
  for (int64_t k = r; k < r + 8 && k < n; ++k)
    if ((p[k] & 0x7f) == 0x7f) return true;
  return false;
}

// fp8 decode through the identity multiply: code c of a group with or without a NaN;
// a NaN group gives 0x7fc00000, -0 becomes +0.
__host__ __device__ inline float fp8_decode_identity(uint8_t c, bool group_nan) {
  return group_nan ? qnan() : fp8_to_f32(c) + 0.0f;
}
__device__ inline float fp8_decode_identity(const uint8_t* p, int64_t i, int64_t n) {
  return fp8_decode_identity(p[i], fp8_row_nan(p, i, n));
}

// ---- host scalar casts (also callable on the device) ----

// Scalar cast of a coefficient to t (BF16 / F16), returned as fp32: bf16 rounds half to even
// whatever `rnd`, f16 follows `rnd`. Inf and NaN are returned unchanged.
__host__ __device__ inline float scalar_cast(float v, npu_dtype t, Rnd rnd) {
  if ((f32_bits(v) & 0x7f800000u) == 0x7f800000u) return v;
  return decode_float(encode_float(v, t, t == NPU_F16 ? rnd : kHalfEven), t);
}


}  // namespace npu_emu

#endif  // NPU_EMU_FLOAT_CVT_H_
