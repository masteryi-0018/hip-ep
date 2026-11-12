/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Device helpers shared by the npu_emu kernels: rounding, float encode /
 * decode, saturating float -> int, and the block floating point codecs.
 */
#ifndef NPU_EMU_DEVICE_H_
#define NPU_EMU_DEVICE_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "npu_emu.h"

namespace npu_emu {

// The two rounding modes the AIE4 conversion wrappers select:
// conv_even = half to even, symmetric_inf = half away from zero. Both act on
// magnitudes (all formats here are sign-magnitude).
enum Rnd { kEven = 0, kAway = 1 };

// Shift an unsigned magnitude right by r bits with rounding. Shifts of 63 or
// more give 0, as mllib's DataConverter.shift_round does.
__device__ inline uint64_t round_shift(uint64_t v, int r, Rnd rnd) {
  if (r <= 0) return v << -r;
  if (r >= 63) return 0;
  const uint64_t half = 1ull << (r - 1);
  const uint64_t rem = v & ((1ull << r) - 1);
  uint64_t q = v >> r;
  if (rem > half || (rem == half && (rnd == kAway || (q & 1)))) ++q;
  return q;
}

// x * c + a rounded once to fp32 (half to even), with x an exact integer
// (|x| < 2^32). This is the NPU's mac_elem_32(int32, float, accfloat): the
// integer operand is not rounded to fp32 first (checked on aiesim outputs of
// requantize_a64 with 32-bit outputs). fmaf cannot express it for |x| > 2^24.
__device__ inline float mac_exact(int64_t x, float c, float a) {
  if (x == 0 || c == 0.0f || !isfinite(c) || !isfinite(a))
    return fmaf(float(x), c, a);
  auto split = [](float f, int64_t* m, int* e) {
    const uint32_t b = __float_as_uint(f);
    const int f_e = (b >> 23) & 0xff;
    const int64_t mag = f_e ? int64_t((b & 0x7fffff) | 0x800000) : int64_t(b & 0x7fffff);
    *m = (b >> 31) ? -mag : mag;
    *e = f_e ? f_e - 150 : -149;
  };
  int64_t mc, ma;
  int ec, ea;
  split(c, &mc, &ec);
  split(a, &ma, &ea);
  const int64_t p = x * mc;  // < 2^56
  if (ma == 0) ma = 0, ea = ec;
  // Common exponent E = max(ec, ea) - 64: the larger-exponent term shifts left
  // by 64 (fits in 121 bits); the other shifts left, or right with a sticky bit.
  // A right shift only happens when that term is < 2^-7 of the other, so the
  // sum keeps >= 63 significant bits and the dropped bits act as sticky.
  const int emax = ec > ea ? ec : ea;
  const int E = emax - 64;
  auto place = [&](int64_t m, int e, bool* sticky) -> __int128 {
    const int sh = e - E;  // >= 0 for the larger term
    if (sh >= 0) return __int128(m) << sh;
    if (-sh >= 127) {
      *sticky |= m != 0;
      return m < 0 ? -1 : 0;  // floor
    }
    const __int128 v = __int128(m);
    *sticky |= (v & ((__int128(1) << -sh) - 1)) != 0;
    return v >> -sh;  // arithmetic: floor
  };
  bool sticky = false;
  const __int128 S = place(p, ec, &sticky) + place(ma, ea, &sticky);
  // value = (S + frac) * 2^E with 0 <= frac < 1 when sticky
  const bool neg = S < 0;
  unsigned __int128 mag = neg ? (unsigned __int128)(-S) : (unsigned __int128)S;
  if (neg && sticky) mag -= 1;  // |S + frac| = (|S| - 1) + (1 - frac)
  if (mag == 0 && !sticky) return 0.0f;
  const uint64_t hi = uint64_t(mag >> 64), lo = uint64_t(mag);
  const int L = hi ? 128 - __clzll(hi) : (lo ? 64 - __clzll(lo) : 0);
  int r = L > 24 ? L - 24 : 0;
  const int top = L - 1 + E;  // exponent of the leading bit
  if (top < -126) r += -126 - top;  // subnormal result: fewer mantissa bits
  unsigned __int128 q = mag;
  if (r > 0) {
    const bool half_set = r <= 128 && ((mag >> (r - 1)) & 1);
    const bool below = r > 1 && (r - 1 >= 128 ? mag != 0 : (mag & ((unsigned __int128)1 << (r - 1)) - 1) != 0);
    q = r >= 128 ? 0 : mag >> r;
    const bool rest = below || sticky;
    if (half_set && (rest || (q & 1))) ++q;
  }
  const float v = ldexpf(float(uint64_t(q)), r + E);
  return neg ? -v : v;
}

// Encode an fp32 value into a sign-magnitude float with `ebits` exponent bits,
// `mbits` mantissa bits and exponent bias `bias`, with subnormals. `fn` selects
// the E4M3FN convention (no infinity; all-ones is NaN, overflow gives NaN);
// otherwise overflow gives infinity (IEEE f16 / bf16).
__device__ inline uint32_t encode_float(float x, int ebits, int mbits, int bias,
                                        bool fn, Rnd rnd) {
  const uint32_t b = __float_as_uint(x);
  const uint32_t sbit = (b >> 31) << (ebits + mbits);
  const int e = (b >> 23) & 0xff;
  const uint32_t m = b & 0x7fffff;
  const uint32_t emax = (1u << ebits) - 1;
  const uint32_t nan = fn ? ((emax << mbits) | ((1u << mbits) - 1))
                          : ((emax << mbits) | (1u << (mbits - 1)));
  if (e == 0xff) {
    if (fn) return sbit | nan;
    if (!m) return sbit | (emax << mbits);
    // NaN keeps its top mantissa bits, as numpy's float16 conversion does
    // (NaN handling on the NPU is unverified).
    const uint32_t pay = m >> (23 - mbits);
    return sbit | (emax << mbits) | (pay ? pay : 1u);
  }
  if (e == 0 && m == 0) return sbit;
  uint64_t sig = e ? (m | 0x800000u) : m;
  int u = e ? e - 127 : -126;
  while (!(sig & 0x800000u)) {  // normalize fp32 subnormals
    sig <<= 1;
    --u;
  }
  const int te = u + bias;
  const int shift = 23 - mbits + (te >= 1 ? 0 : 1 - te);
  uint64_t q = round_shift(sig, shift, rnd);
  uint32_t code;
  if (te >= 1) {
    int ee = te;
    if (q >= (2ull << mbits)) {  // rounding carried into the exponent
      q >>= 1;
      ++ee;
    }
    code = (uint32_t(ee) << mbits) | uint32_t(q & ((1u << mbits) - 1));
  } else {
    code = uint32_t(q);  // subnormal; q == 1 << mbits is the smallest normal
  }
  const uint32_t max_finite =
      fn ? (nan - 1) : (((emax - 1) << mbits) | ((1u << mbits) - 1));
  if (code > max_finite) return sbit | (fn ? nan : (emax << mbits));
  return sbit | code;
}

__device__ inline uint16_t f32_to_bf16(float x, Rnd rnd) {
  return uint16_t(encode_float(x, 8, 7, 127, false, rnd));
}
__device__ inline uint16_t f32_to_f16(float x, Rnd rnd) {
  return uint16_t(encode_float(x, 5, 10, 15, false, rnd));
}
__device__ inline uint8_t f32_to_fp8(float x, Rnd rnd) {
  return uint8_t(encode_float(x, 4, 3, 7, true, rnd));
}

__device__ inline float bf16_to_f32(uint16_t h) {
  return __uint_as_float(uint32_t(h) << 16);
}
__device__ inline float f16_to_f32(uint16_t h) {
  const uint32_t s = uint32_t(h >> 15) << 31;
  const int e = (h >> 10) & 0x1f;
  const uint32_t m = h & 0x3ff;
  if (e == 0x1f) return __uint_as_float(s | 0x7f800000u | (m << 13));
  if (e == 0) {
    const float v = ldexpf(float(m), -24);
    return s ? -v : v;
  }
  return __uint_as_float(s | (uint32_t(e - 15 + 127) << 23) | (m << 13));
}
__device__ inline float fp8_to_f32(uint8_t c) {
  const bool neg = c & 0x80;
  const int e = (c >> 3) & 0xf;
  const int m = c & 7;
  float v;
  if (e == 0xf && m == 7)
    v = __uint_as_float(0x7fc00000u);
  else if (e == 0)
    v = ldexpf(float(m), -9);
  else
    v = ldexpf(float(8 + m), e - 10);
  return neg ? -v : v;
}

// fp8 element i of a buffer as the AIE4 kernels decode it: mul_8x8_8x8 against
// an 8x8 identity matrix, i.e. each group of 8 consecutive elements (aligned to
// the buffer start) is a matrix row and every output sums its row's 8 products.
// NaN * 0 is NaN, so a NaN anywhere in the row makes all 8 outputs NaN. Checked
// on aiesim outputs of the fp8 compare and convert kernels.
__device__ inline float fp8_to_f32_row(const uint8_t* p, int64_t i, int64_t n) {
  const int64_t r = i & ~int64_t(7);
  for (int64_t k = r; k < r + 8 && k < n; ++k)
    if ((p[k] & 0x7f) == 0x7f) return __uint_as_float(0x7fc00000u);
  return fp8_to_f32(p[i]);
}

// Floating point element of type t (F32 / F16 / BF16) as fp32 (exact).
__device__ inline float load_float(const void* p, npu_dtype t, int64_t i) {
  switch (t) {
    case NPU_F32:
      return static_cast<const float*>(p)[i];
    case NPU_F16:
      return f16_to_f32(static_cast<const uint16_t*>(p)[i]);
    default:
      return bf16_to_f32(static_cast<const uint16_t*>(p)[i]);
  }
}

// Round fp32 to type t (F32 / F16 / BF16) and store.
__device__ inline void store_float(void* p, npu_dtype t, int64_t i, float v,
                                   Rnd rnd) {
  switch (t) {
    case NPU_F32:
      static_cast<float*>(p)[i] = v;
      break;
    case NPU_F16:
      static_cast<uint16_t*>(p)[i] = f32_to_f16(v, rnd);
      break;
    default:
      static_cast<uint16_t*>(p)[i] = f32_to_bf16(v, rnd);
      break;
  }
}

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

__device__ inline void int_range(npu_dtype t, int64_t* lo, int64_t* hi) {
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
    default:  // U32
      *lo = 0, *hi = 4294967295ll;
      break;
  }
}

// fp32 -> integer: round half to even, saturate to [lo, hi]. NaN -> 0
// (unverified on the NPU).
__device__ inline int64_t sat_rne(float v, int64_t lo, int64_t hi) {
  if (v != v) return 0;
  const double r = rint(double(v));  // v is exact in double; rint is RNE
  if (r <= double(lo)) return lo;
  if (r >= double(hi)) return hi;
  return int64_t(r);
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

// The NPU's to_vector_sign / to_v32int32 tail for small integer outputs:
// 8/16-bit outputs saturate to their own range; 32-bit outputs saturate to
// int32 and unsigned outputs then clamp at 0, so U32 tops out at 2^31 - 1.
__device__ inline int64_t requant_tail(float v, npu_dtype t) {
  if (t == NPU_I32 || t == NPU_U32) {
    const int64_t r = sat_rne(v, INT32_MIN, INT32_MAX);
    return t == NPU_U32 && r < 0 ? 0 : r;
  }
  int64_t lo, hi;
  int_range(t, &lo, &hi);
  return sat_rne(v, lo, hi);
}

// The AIE4 tile rounding modes used when a kernel converts fp32 to an integer
// (to_vector_sign / to_fixed): conv_even = nearest, ties to even;
// symmetric_floor = toward zero.
enum IntRnd { kIntEven = 0, kIntTrunc = 1 };

// to_vector_sign<t>(acc, shift) / to_fixed(v, shift) for 8/16/32-bit integer t:
// v * 2^shift (exact), rounded with `mode`, saturated to the range of t.
// NaN -> 0 (unverified on the NPU).
__device__ inline int64_t to_int_sat(float v, npu_dtype t, int shift, IntRnd mode) {
  int64_t lo, hi;
  int_range(t, &lo, &hi);
  if (v != v) return 0;
  const double x = ldexp(double(v), shift);
  const double r = mode == kIntEven ? rint(x) : trunc(x);
  if (r <= double(lo)) return lo;
  if (r >= double(hi)) return hi;
  return int64_t(r);
}

// ---- block floating point (mllib DataConverter.f2bfp_mladf, non-bfloat) ----
//
// For a block of `n` fp32 values: shared exponent me = max of the biased
// exponent fields (each clamped to >= 1); per subtile of `sub` values a shift
// s = min(min over the subtile of (me - e_i), max_shift); each mantissa is the
// 24-bit significand shifted right by me - e_i + 25 - mbits - s with rounding,
// saturated to 2^(mbits-1) - 1, and 0 when that shift exceeds 24. A set sign
// bit marks the value negative, including -0.0.
struct BlockOut {
  int mag[16];
  bool neg[16];
  int shift[8];
  int exp;
};

__device__ inline void encode_block(const float* v, int n, int mbits, int sub,
                                    int max_shift, Rnd rnd, BlockOut* o) {
  int e[16];
  uint32_t sig[16];
  int me = 0;
  for (int k = 0; k < n; ++k) {
    const uint32_t b = __float_as_uint(v[k]);
    const int f = (b >> 23) & 0xff;
    sig[k] = (f > 0 ? 0x800000u : 0u) | (b & 0x7fffff);
    e[k] = f > 1 ? f : 1;
    o->neg[k] = b >> 31;
    me = e[k] > me ? e[k] : me;
  }
  const int bm_max = (1 << (mbits - 1)) - 1;
  for (int s = 0; s < n / sub; ++s) {
    int mn = 1 << 30;
    for (int k = s * sub; k < (s + 1) * sub; ++k)
      mn = me - e[k] < mn ? me - e[k] : mn;
    o->shift[s] = mn < max_shift ? mn : max_shift;
  }
  for (int k = 0; k < n; ++k) {
    const int sh = me - e[k] + 25 - mbits - o->shift[k / sub];
    uint64_t q = round_shift(sig[k], sh, rnd);
    if (q > uint64_t(bm_max)) q = bm_max;
    o->mag[k] = sh > 24 ? 0 : int(q);
  }
  o->exp = me;
}

// bfp16: 8 values, 8-bit two's complement mantissas. A block whose mantissas
// are all zero gets exponent 0, as the NPU emits (the value is 0 either way).
__device__ inline void encode_bfp16(const float* v, Rnd rnd, uint8_t* blk) {
  BlockOut o;
  encode_block(v, 8, 8, 1, 0, rnd, &o);
  bool any = false;
  for (int k = 0; k < 8; ++k) {
    blk[1 + k] = uint8_t(int8_t(o.neg[k] ? -o.mag[k] : o.mag[k]));
    any |= o.mag[k] != 0;
  }
  blk[0] = any ? uint8_t(o.exp) : 0;
}

__device__ inline float decode_bfp16(const uint8_t* blk, int k) {
  return ldexpf(float(int8_t(blk[1 + k])), int(blk[0]) - 133);
}

// mx6: 16 values, 5-bit sign-magnitude mantissas, one shift bit per pair,
// packed little-endian: exp | shifts << 8 | m_k << (16 + 5k). Unlike bfp16, the
// exponent of an all-zero block follows mllib's model (unverified on the NPU).
__device__ inline void encode_mx6(const float* v, Rnd rnd, uint8_t* blk) {
  BlockOut o;
  encode_block(v, 16, 5, 2, 1, rnd, &o);
  uint32_t w[3] = {uint32_t(o.exp & 0xff), 0, 0};
  for (int s = 0; s < 8; ++s) w[0] |= uint32_t(o.shift[s] & 1) << (8 + s);
  for (int k = 0; k < 16; ++k) {
    const uint32_t code = (o.neg[k] ? 0x10u : 0u) | uint32_t(o.mag[k]);
    const int bit = 16 + 5 * k;
    w[bit / 32] |= code << (bit % 32);
    if (bit % 32 > 27) w[bit / 32 + 1] |= code >> (32 - bit % 32);
  }
  for (int j = 0; j < 12; ++j) blk[j] = uint8_t(w[j / 4] >> (8 * (j % 4)));
}

// Decoded mx6 values; zero magnitudes decode to +0 (the NPU outputs +0).
__device__ inline float decode_mx6(const uint8_t* blk, int k) {
  uint32_t w[3];
  for (int j = 0; j < 3; ++j)
    w[j] = uint32_t(blk[4 * j]) | uint32_t(blk[4 * j + 1]) << 8 |
           uint32_t(blk[4 * j + 2]) << 16 | uint32_t(blk[4 * j + 3]) << 24;
  const int exp = w[0] & 0xff;
  const int shift = (w[0] >> (8 + k / 2)) & 1;
  const int bit = 16 + 5 * k;
  uint32_t code = w[bit / 32] >> (bit % 32);
  if (bit % 32 > 27) code |= w[bit / 32 + 1] << (32 - bit % 32);
  code &= 0x1f;
  const int mag = code & 0xf;
  if (mag == 0) return 0.0f;
  const float x = ldexpf(float(mag), exp - shift - 130);
  return code & 0x10 ? -x : x;
}

}  // namespace npu_emu

#endif  // NPU_EMU_DEVICE_H_
