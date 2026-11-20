/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Device models of the NPU's function units: reciprocal, inverse square root, ln, exp2,
 * 16-bit sqrt / reciprocal / inverse square root, the tanh table and the table (LUT)
 * lookup.
 * hw_inv_f32 / hw_invsqrt_f32 take an exception bitmap in device memory
 * (npu_emu_nlf_table).
 *
 * Special values ([measured] against the NPU, [unverified] otherwise): exp2 matches on every
 * fp32 input [measured]; the zero / infinity / NaN results documented for the other units
 * [unverified].
 */
#ifndef NPU_EMU_NLF_H_
#define NPU_EMU_NLF_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/tanh_tables.h"

/* ---- host side (internal) ---- */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NPU_EMU_NLF_INV_F32 = 0,     /* hw_inv_f32: 2^23 bits, by fp32 mantissa */
  NPU_EMU_NLF_INVSQRT_F32 = 1, /* hw_invsqrt_f32: 2^24 bits, by domain index */
} npu_emu_nlf_table_id;

/* Device pointer to exception bitmap `id` on the current HIP device; decoded
 * and uploaded (synchronously) on the first call per device, then cached.
 * Thread-safe. Returns a hipError_t (0 on success). */
int npu_emu_nlf_table(int id, const uint32_t** dev_ptr);

#ifdef __cplusplus
}
#endif

/* ---- device side ---- */

namespace npu_emu {

// Bit i of a bitmap stored as little-endian 32-bit words.
__device__ inline bool nlf_bit(const uint32_t* bm, uint32_t i) {
  return (bm[i >> 5] >> (i & 31)) & 1u;
}

// fp32 reciprocal: RNE(1/x), except for the mantissas set in `exc`
// (NPU_EMU_NLF_INV_F32), where it is the other fp32 neighbour of 1/x. Zero /
// subnormal x -> signed inf; a result below 2^-126 -> signed zero; inf -> signed
// zero; NaN -> quiet NaN (payload kept).
__device__ inline float hw_inv_f32(float x, const uint32_t* exc) {
  const uint32_t b = __float_as_uint(x);
  const uint32_t sign = b & 0x80000000u;
  const uint32_t e = (b >> 23) & 0xff, m = b & 0x7fffff;
  if (e == 255) return __uint_as_float(m ? b | 0x400000u : sign);
  if (e == 0) return __uint_as_float(sign | 0x7f800000u);
  if (e >= 253 && !(e == 253 && m == 0)) return __uint_as_float(sign);
  uint32_t mag = __float_as_uint(__fdiv_rn(1.0f, x)) & 0x7fffffffu;
  if (nlf_bit(exc, m)) {
    // the product of two 24-bit significands is exact in double
    const bool above = __dmul_rn(double(__uint_as_float(mag)), double(fabsf(x))) > 1.0;
    mag = above ? mag - 1 : mag + 1;
  }
  return __uint_as_float(sign | mag);
}

// fp32 inverse square root. With x = m * 2^(2k), m in [1, 4) (index = bits(m) -
// bits(1.0)), the result is base(m) * 2^-k, base = fp32(1 / sqrt(double(m))),
// except for the indices set in `exc` (NPU_EMU_NLF_INVSQRT_F32), where it is the
// other fp32 neighbour of 1/sqrt(m). Zero / subnormal -> signed inf, NaN -> quiet
// NaN, +inf -> +0, negative -> 0xffc00000.
__device__ inline float hw_invsqrt_f32(float x, const uint32_t* exc) {
  const uint32_t b = __float_as_uint(x);
  const bool neg = b >> 31;
  const int e = (b >> 23) & 0xff;
  const uint32_t frac = b & 0x7fffff;
  if (e == 0) return __uint_as_float(neg ? 0xff800000u : 0x7f800000u);
  if (e == 255) return __uint_as_float(frac ? b | 0x400000u : (neg ? 0xffc00000u : 0u));
  if (neg) return __uint_as_float(0xffc00000u);
  // An even biased exponent is an odd power of two: borrow one so the
  // mantissa lands in [2, 4).
  const int borrow = 1 - (e & 1);
  const uint32_t index = (uint32_t(borrow) << 23) | frac;
  const float m = __uint_as_float(0x3f800000u + index);
  const float base = __double2float_rn(__ddiv_rn(1.0, __dsqrt_rn(double(m))));
  int32_t r = int32_t(__float_as_uint(base));
  if (nlf_bit(exc, index)) {
    // base < 1/sqrt(m) <=> Bi^2 * Mi < 2^71, Bi = base * 2^24, Mi = m * 2^23
    const uint32_t bb = __float_as_uint(base);  // base in (0.5, 1]
    const uint64_t Bi = uint64_t((bb & 0x7fffff) | 0x800000) << (((bb >> 23) & 0xff) - 126);
    const uint64_t Mi = uint64_t((__float_as_uint(m) & 0x7fffff) | 0x800000) << borrow;
    const unsigned __int128 p = (unsigned __int128)(Bi * Bi) * Mi;
    r += p < ((unsigned __int128)1 << 71) ? 1 : -1;
  }
  const int halved = (e - 127 - borrow) >> 1;  // exact: e - 127 - borrow is even
  return __uint_as_float(uint32_t(r - halved * (1 << 23)));
}

// Approximate ln(x). x = 2^e * m, m in [1, 2); two range reductions (sqrt(2),
// then 2^(+-1/4)) bring m near 1, t = m - 1, and
//   ln(x) ~ fma(t*t, fma(t, 1/3, -1/2), fma(e, ln2, t + offset)).
// Only the bit fields of x are used: sign, zero, inf and NaN are not special.
__device__ inline float hw_approx_ln(float x) {
  const float kSqrt2 = __uint_as_float(0x3fb504f3u);       // 1.41421356f
  const float kSqrt42 = __uint_as_float(0x3f9837f0u);      // 1.18920708f
  const float kSqrt42Inv = __uint_as_float(0x3f5744fdu);   // 0.84089641f
  const float kLn2Div4 = __uint_as_float(0x3e317218u);     // 0.17328679f
  const float kLn2 = __uint_as_float(0x3f317218u);         // 0.693147182f
  const float kThird = __uint_as_float(0x3eaaaaabu);       // 0.333333343f
  const uint32_t b = __float_as_uint(x);
  const int e_int = (b >> 23) & 0xff;
  float m = __uint_as_float((b & 0x7fffffu) | 0x3f800000u);
  const bool above = m > kSqrt2;
  if (above) m = __fmul_rn(m, 0.5f);
  float off = 0.0f;
  if (m > kSqrt42) {
    m = __fmul_rn(m, kSqrt42Inv);
    off = kLn2Div4;
  } else if (m < kSqrt42Inv) {
    m = __fmul_rn(m, kSqrt42);
    off = -kLn2Div4;
  }
  const float e = float(e_int - 127 + (above ? 1 : 0));
  const float t = __fsub_rn(m, 1.0f);
  const float p_low = fmaf(e, kLn2, __fadd_rn(t, off));
  const float p_high = fmaf(t, kThird, -0.5f);
  return fmaf(__fmul_rn(t, t), p_high, p_low);
}

// Offset and slope tables of the hardware exp2, per 32nd of an octave.
__device__ const int16_t kExp2Cnst[32] = {0,   39,  74,  110, 141, 172, 200, 225, 250, 270, 290,
                                          305, 320, 332, 340, 347, 352, 352, 350, 347, 340, 328,
                                          315, 299, 280, 257, 229, 200, 167, 131, 91,  47};
__device__ const int8_t kExp2Coeff[32] = {-10, -9, -9, -8, -8, -7, -6, -6, -5, -5, -4,
                                          -4,  -3, -2, -2, -1, 0,  0,  1,  2,  3,  3,
                                          4,   5,  6,  7,  7,  8,  9,  10, 11, 12};

// Piecewise-linear 2^x on the fp32 bit pattern of an already biased p
// (~ x + 895). bf16_out: the index gets a +16 forward rounding (cleared to a
// multiple of 16 when bits 4..11 are all ones) and the result keeps its upper 16
// bits. NaN p gives 0x7fc00000.
__device__ inline float hw_exp2_pwl(float p, bool bf16_out) {
  if (p != p) return qnan();
  const int64_t x = int64_t(__float_as_int(__fadd_rn(p, 0.0f)) >> 2) - (int64_t(0x111) << 20);
  const int sel = int((x >> 7) & 31);
  int64_t base = x & ~int64_t(1);
  if (bf16_out) {
    base += 16;
    if (((x >> 4) & 255) == 255) base &= ~int64_t(15);
  }
  int64_t approx = base + ((((x >> 2) & 31) * kExp2Coeff[sel]) >> 3) - kExp2Cnst[sel];
  if (x < (1 << 12) && x >= -(int64_t(11) << 12)) {
    const int64_t data = 4096 + (approx & 4095);
    const int sh = int(1 - (approx >> 12));  // 0..13 in this range
    approx = (data >> sh) | ((data & ((int64_t(1) << sh) - 1)) != 0);
  }
  approx = approx < 0 ? 0 : (approx > (int64_t(255) << 12) ? (int64_t(255) << 12) : approx);
  return __uint_as_float(uint32_t((approx << 11) & (bf16_out ? 0x7fff0000 : 0x7fffffff)));
}

// Hardware exp2 of an already biased p (~ x + 895), fp32 result. NaN p gives
// 0x7fc00000; results below 2^-126 stay subnormal.
__device__ inline float hw_exp2_bf20(float p) { return hw_exp2_pwl(p, false); }

// Hardware exp2 of an already biased p with bf16 output. NaN p gives 0x7fc00000.
__device__ inline float hw_exp2_bf16(float p) { return hw_exp2_pwl(p, true); }

// 16-bit square root on a t code (BF16 / F16): correctly rounded, except that f16
// (2 - 2^-10) * 4^k rounds up to the power of two. Negative inputs give -qNaN, NaN is
// quieted, bf16 subnormals read as signed zero.
__device__ inline uint16_t hw_sqrt16(uint16_t b, npu_dtype t) {
  const bool neg = b >> 15;
  if (t == NPU_F16) {
    const int e = (b >> 10) & 0x1f;
    const int m = b & 0x3ff;
    if (e == 0x1f) return m ? uint16_t(b | 0x200) : (neg ? 0xfe00 : 0x7c00);
    if (!(b & 0x7fff)) return b;
    if (neg) return 0xfe00;
    const uint16_t r =
        f32_to_f16(__double2float_rn(__dsqrt_rn(double(f16_to_f32(b)))), kHalfEven);
    return m == 0x3ff && e >= 2 && !(e & 1) ? uint16_t(r + 1) : r;
  }
  const int e = (b >> 7) & 0xff;
  const int m = b & 0x7f;
  if (e == 0xff) return m ? uint16_t(b | 0x40) : (neg ? 0xffc0 : 0x7f80);
  if (e == 0) return uint16_t(b & 0x8000);  // zero and subnormals: signed zero
  if (neg) return 0xffc0;
  return f32_to_bf16(__double2float_rn(__dsqrt_rn(double(bf16_to_f32(b)))), kHalfEven);
}

// RNE(1 / y) as an unsigned code of t (BF16 / F16), for a finite nonzero fp32 y; overflow
// gives infinity. 1/y = 2^62 / m * 2^(-62 - e) with y = m * 2^e: the integer quotient keeps
// at least 38 bits and the remainder becomes the sticky bit.
__device__ inline uint32_t recip_code(float y, npu_dtype t) {
  const uint32_t b = __float_as_uint(y);
  const int be = (b >> 23) & 0xff;
  const uint64_t m = be ? ((b & 0x7fffff) | 0x800000u) : (b & 0x7fffff);
  const int e = be ? be - 150 : -149;
  const uint64_t num = 1ull << 62;
  const uint64_t q = num / m, r = num % m;
  const FloatFmt f = float_fmt(t);
  const uint32_t code = encode_mag((q << 1) | (r != 0), -63 - e, f, kHalfEven, false);
  const uint32_t inf = ((1u << f.ebits) - 1) << f.mbits;
  return code < inf ? code : inf;
}

// 16-bit reciprocal on a t code (BF16 / F16): RNE(1 / y), NaN quieted. bf16: zero or
// subnormal y gives +-inf, a subnormal result +-0. f16: subnormals kept, +-0 -> +-inf,
// +-inf -> +-0.
__device__ inline uint16_t hw_inv16(uint16_t h, npu_dtype t) {
  const uint16_t s = h & 0x8000;
  if (t == NPU_F16) {
    const int e = (h >> 10) & 0x1f;
    if ((h & 0x7fff) == 0) return s | 0x7c00;
    if (e == 0x1f) return (h & 0x3ff) ? uint16_t(h | 0x200) : s;
    return s | uint16_t(recip_code(f16_to_f32(h), NPU_F16));
  }
  const int e = (h >> 7) & 0xff;
  if (e == 0) return s | 0x7f80;
  if (e == 0xff) return (h & 0x7f) ? uint16_t(h | 0x40) : s;
  const uint32_t code = recip_code(bf16_to_f32(h), NPU_BF16);
  if ((code >> 7) == 0) return s;  // subnormal result -> 0
  return s | uint16_t(code);
}

// 16-bit inverse square root on an f16 code: the fp32 inverse square root rounded half to
// even to f16 (no kernel uses a bf16 one).
__device__ inline uint16_t hw_invsqrt_f16(uint16_t h, const uint32_t* exc) {
  return f32_to_f16(hw_invsqrt_f32(f16_to_f32(h), exc), kHalfEven);
}

// ---- hardware tanh (table) ----

// tanh -> bf16 bits for the code c of type t (BF16 / F16). Below the tabulated slice the
// result is exact (bf16: the input; f16: the f16 value truncated to bf16), above it +-1;
// tanh is odd; every NaN code gives 0x7F81.
__device__ inline uint16_t tanh16(uint16_t c, npu_dtype t) {
  const bool f16 = t == NPU_F16;
  const uint32_t a = c & 0x7fffu, s = c & 0x8000u;
  const uint32_t lo = f16 ? 0x3800u : 0x3f00u, hi = f16 ? 0x4200u : 0x4040u;
  const uint32_t inf = f16 ? 0x7c00u : 0x7f80u;
  if (a > inf) return 0x7f81;
  if (a >= hi) return uint16_t(s | 0x3f80u);
  if (a >= lo) return uint16_t(s | (f16 ? kTanhF16[a - lo] : kTanhBf16[a - lo]));
  if (!f16) return c;
  return uint16_t(s | (f32_bits(f16_to_f32(uint16_t(a))) >> 16));
}

// tanh of an fp32 h for t (BF16 / F16): bf16 indexes by the top 16 bits, f16 by the f16
// code (round half to even).
__device__ inline float tanh_acc(float h, npu_dtype t) {
  const uint16_t c = t == NPU_F16 ? f32_to_f16(h, kHalfEven) : uint16_t(f32_bits(h) >> 16);
  return bf16_to_f32(tanh16(c, t));
}

// ---- table (LUT) lookup ----
//
// A table image holds groups of 8 (offset, slope) fp32 pairs, each group stored twice;
// entry i is read from the first copy.
__host__ __device__ inline void lut_entry(const float* lut, int i, float* off, float* sl) {
  const int k = (i / 8) * 32 + (i % 8) * 2;
  *off = lut[k];
  *sl = lut[k + 1];
}

// Entry index of a table of n entries: floor(bf16(fma(x, k, n / 2))) (bf16 rounding
// `rnd`) clamped to [0, n - 1]; NaN -> 0.
__device__ inline int lut_index(float x, float k, int n, Rnd rnd) {
  const float v = bf16_to_f32(f32_to_bf16(fma_q(x, k, float(n / 2)), rnd));
  if (!(v > 0.0f)) return 0;
  if (v >= float(n - 1)) return n - 1;
  return int(floorf(v));
}

// fma(x, slope, offset) of entry i; the slope is first rounded to slope_t (F32: unchanged)
// with `rnd`.
__device__ inline float lut_interp(float x, const float* lut, int i, npu_dtype slope_t, Rnd rnd) {
  float off, sl;
  lut_entry(lut, i, &off, &sl);
  return fma_q(x, round_float(sl, slope_t, rnd), off);
}

// Device pointers to both exception bitmaps (host side).
struct NlfTables {
  const uint32_t* inv;
  const uint32_t* isq;
};

inline int nlf_tables(NlfTables* t) {
  if (int rc = npu_emu_nlf_table(NPU_EMU_NLF_INV_F32, &t->inv)) return rc;
  return npu_emu_nlf_table(NPU_EMU_NLF_INVSQRT_F32, &t->isq);
}

}  // namespace npu_emu

#endif  // NPU_EMU_NLF_H_
