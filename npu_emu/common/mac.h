/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * fp32 multiply-add of the vector unit. Plain fp32 fma / mul / add are the GPU's IEEE
 * fmaf / __fmul_rn / __fadd_rn; this file adds the exact-integer MAC, the flushing MAC,
 * the 16-bit multiply and the vector reduce-add.
 *
 * Special values ([measured] against the NPU, [unverified] otherwise): the NaN results of the
 * multiply-adds are the GPU's IEEE ones [unverified]; mul16 flushes a subnormal result after
 * rounding [measured] and reads subnormal operands as zero [unverified]; vaddr_f32_reduce's
 * NaN / infinity rules [unverified]. mac_exact with x == 0 and an infinite c is invalid
 * (0xFFC00000), before a NaN addend; otherwise a NaN operand is returned quieted [default];
 * mul_q / add_q / sub_q / fma_q (golden: the same names, and mul / add / sub / fma) give the
 * first-NaN-operand rule; fma_ftz, mul16 and mul16_f32 use them, sqrt_q is the square root [default].
 */
#ifndef NPU_EMU_MAC_H_
#define NPU_EMU_MAC_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/float_cvt.h"

namespace npu_emu {

// Multiply / add / fma with a determined NaN: the first NaN operand quieted (sign and payload
// kept), else 0xFFC00000 for an invalid operation (0 * inf, inf - inf), 0 * inf before a NaN
// addend. The plain intrinsics leave the choice between two NaN operands, and the sign of a
// negated NaN, to the instruction selected, so a path that can see NaN operands uses these
// [default]. Same rule as golden_mac.mul / add / fma.
__host__ __device__ inline float quiet_nan(float v) {
  return __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, v) | 0x400000u);
}
__host__ __device__ inline float default_nan() { return __builtin_bit_cast(float, 0xffc00000u); }
__host__ __device__ inline float neg_f32(float x) {
  return __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, x) ^ 0x80000000u);
}
// The operands are tested before the operation: with the result tested first the compiler
// produced wrong values in a kernel that keeps two fma chains.
__host__ __device__ inline float mul_q(float a, float b) {
  if (a != a) return quiet_nan(a);
  if (b != b) return quiet_nan(b);
  const float r = a * b;
  return r != r ? default_nan() : r;
}
__host__ __device__ inline float add_q(float a, float b) {
  if (a != a) return quiet_nan(a);
  if (b != b) return quiet_nan(b);
  const float r = a + b;
  return r != r ? default_nan() : r;
}
// a - b: b is negated by flipping its sign bit (a NaN included), then added.
__host__ __device__ inline float sub_q(float a, float b) { return add_q(a, neg_f32(b)); }
__host__ __device__ inline float fma_q(float a, float b, float c) {
  if (a != a) return quiet_nan(a);
  if (b != b) return quiet_nan(b);
  if ((a == 0.0f && __builtin_isinf(b)) || (__builtin_isinf(a) && b == 0.0f)) return default_nan();
  if (c != c) return quiet_nan(c);
  const float r = fmaf(a, b, c);
  return r != r ? default_nan() : r;
}

// sqrt: a NaN is returned quieted, a negative operand gives 0xFFC00000 (-0 stays -0).
__device__ inline float sqrt_q(float a) {
  if (a != a) return quiet_nan(a);
  if (a < 0.0f) return default_nan();
  // __fsqrt_rn is not correctly rounded on the target (sqrt(1.5f) is 1 ulp low); the double sqrt
  // rounded once is exact for fp32.
  return __double2float_rn(__dsqrt_rn(double(a)));
}

// x * c + a rounded once to fp32 (half to even), x an exact integer of any int64 value.
// kFtzIn flushes c and a, kFtzOut the result.
__device__ inline float mac_exact(int64_t x, float c, float a, int flags = 0) {
  if (flags & kFtzIn) c = ftz(c), a = ftz(a);
  float r;
  if (x == 0 || c == 0.0f || !isfinite(c) || !isfinite(a)) {
    r = fma_q(float(x), c, a);
    return flags & kFtzOut ? ftz(r) : r;
  }
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
  const __int128 p = __int128(x) * mc;  // < 2^87
  if (ma == 0) ea = ec;
  // Common unit 2^E with E 120 below the leading bit of the larger term: that term is exact, the
  // other is shifted left or right with a sticky bit (it can only lose bits when it is so far
  // below that no cancellation reaches it).
  auto bit_len = [](unsigned __int128 v) {
    const uint64_t h = uint64_t(v >> 64), l = uint64_t(v);
    return h ? 128 - __clzll(h) : (l ? 64 - __clzll(l) : 0);
  };
  const int top_p = ec + bit_len(p < 0 ? (unsigned __int128)(-p) : (unsigned __int128)p) - 1;
  const int top_a = ma ? ea + bit_len(ma < 0 ? uint64_t(-ma) : uint64_t(ma)) - 1 : top_p;
  const int E = (top_p > top_a ? top_p : top_a) - 120;
  auto place = [&](__int128 m, int e, bool* sticky) -> __int128 {
    const int sh = e - E;  // >= 0 for the larger term
    if (sh >= 0) return m << sh;
    if (-sh >= 127) {
      *sticky |= m != 0;
      return m < 0 ? -1 : 0;  // floor
    }
    *sticky |= (m & ((__int128(1) << -sh) - 1)) != 0;
    return m >> -sh;  // arithmetic: floor
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
  int rs = L > 24 ? L - 24 : 0;
  const int top = L - 1 + E;  // exponent of the leading bit
  if (top < -126) rs += -126 - top;  // subnormal result: fewer mantissa bits
  unsigned __int128 q = mag;
  if (rs > 0) {
    const bool half_set = rs <= 128 && ((mag >> (rs - 1)) & 1);
    const bool below = rs > 1 && (rs - 1 >= 128 ? mag != 0 : (mag & ((unsigned __int128)1 << (rs - 1)) - 1) != 0);
    q = rs >= 128 ? 0 : mag >> rs;
    const bool rest = below || sticky;
    if (half_set && (rest || (q & 1))) ++q;
  }
  const float v = ldexpf(float(uint64_t(q)), rs + E);
  r = neg ? -v : v;
  return flags & kFtzOut ? ftz(r) : r;
}

// fp32 a * b + c, one rounding (half to even); kFtzIn flushes the operands, kFtzOut the
// result.
__device__ inline float fma_ftz(float a, float b, float c, int flags) {
  if (flags & kFtzIn) a = ftz(a), b = ftz(b), c = ftz(c);
  const float r = fma_q(a, b, c);
  return flags & kFtzOut ? ftz(r) : r;
}

// 16-bit vector multiply x * y of t codes (BF16 / F16): an operand subnormal in t reads as a
// zero of its sign, the product is rounded half to even, and a result below the smallest
// normal of t becomes a zero of its sign.
__device__ inline uint16_t mul16(uint16_t x, uint16_t y, npu_dtype t) {
  const FloatFmt f = float_fmt(t);
  const uint32_t emask = ((1u << f.ebits) - 1) << f.mbits, sbit = 1u << (f.ebits + f.mbits);
  auto operand = [&](uint32_t c) { return decode_float((c & emask) ? c : (c & sbit), t); };
  return uint16_t(encode_float(mul_q(operand(x), operand(y)), t, kHalfEven, kFtzOut));
}

// mul16 on fp32 values exact in t, as fp32.
__device__ inline float mul16_f32(float a, float b, npu_dtype t) {
  const uint16_t x = uint16_t(encode_float(a, t, kHalfEven));
  const uint16_t y = uint16_t(encode_float(b, t, kHalfEven));
  return decode_float(mul16(x, y, t), t);
}

// Vector reduce-add of v[0..n) (n <= 32) and, if has_s, the scalar s. Every
// operand is rounded half to even to a multiple of 2^(E - 157), E the largest
// exponent (a subnormal counts as exponent 1; operands 32 or more binades below
// E are dropped); the aligned integers are summed exactly and the sum is rounded
// once to 24 bits, half to even. A result below 2^-126 is a zero with the sign of
// the sum; an exact zero is +0; overflow gives the signed infinity. A NaN
// operand, or +inf with -inf, gives 0x7fffffff; else an infinite operand wins.
__device__ inline float vaddr_f32_reduce(const float* v, int n, bool has_s = false,
                                         float s = 0.0f) {
  const int cnt = n + (has_s ? 1 : 0);
  auto op = [&](int k) { return __float_as_uint(k < n ? v[k] : s); };
  int emax = 1;
  bool nan = false, pinf = false, ninf = false;
  for (int k = 0; k < cnt; ++k) {
    const uint32_t b = op(k);
    const int e = (b >> 23) & 0xff;
    if (e == 0xff) {
      if (b & 0x7fffff)
        nan = true;
      else if (b >> 31)
        ninf = true;
      else
        pinf = true;
    }
    emax = e > emax ? e : emax;
  }
  if (nan || (pinf && ninf)) return __uint_as_float(0x7fffffffu);
  if (pinf) return __uint_as_float(0x7f800000u);
  if (ninf) return __uint_as_float(0xff800000u);
  int64_t total = 0;
  for (int k = 0; k < cnt; ++k) {
    const uint32_t b = op(k);
    const int e = (b >> 23) & 0xff;
    const int d = emax - (e ? e : 1);
    if (d >= 32) continue;
    const int64_t mag = int64_t((e ? 0x800000u : 0u) | (b & 0x7fffff)) << 7;
    int64_t q = mag >> d;
    if (d > 0) {
      const int64_t rem = mag & ((int64_t(1) << d) - 1), half = int64_t(1) << (d - 1);
      if (rem > half || (rem == half && (q & 1))) ++q;
    }
    total += (b >> 31) ? -q : q;
  }
  if (total == 0) return 0.0f;
  // |total| < 2^38: one 24-bit RNE; the scaling is exact in double.
  const double val = ldexp(double(__ll2float_rn(total)), emax - 157);
  if (fabs(val) < 0x1p-126) return total < 0 ? -0.0f : 0.0f;
  return __double2float_rn(val);  // exact, or +-inf on overflow
}

}  // namespace npu_emu

#endif  // NPU_EMU_MAC_H_
