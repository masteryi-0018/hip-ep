/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Accumulate / normalise stages shared by the bf16 and the block floating
 * point / fp8 MMAC models.
 *
 * Special values ([measured] against the NPU, [unverified] otherwise): finite results follow
 * the instruction model [measured]; NaN / infinite operands, overflow and underflow, and the
 * fp32-add fallback of the single-value MMACs [unverified].
 */
#ifndef NPU_EMU_MMAC_H_
#define NPU_EMU_MMAC_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/block_fp.h"
#include "common/mac.h"

namespace npu_emu {

// Arithmetic shift right by s with one guard bit; *rup is the round-half-to-even
// increment, *frac whether nonzero bits were shifted out.
__device__ inline int64_t g_shift(int64_t v, int s, int width, int* rup, int* frac) {
  if (s > 62) s = 62;
  const int64_t w = v * 2;
  const int64_t e = w >> s;
  const int64_t r = e >> 1;
  const int lsb = int((e >> 1) & 1), grd = int(e & 1);
  const int ms = s < width + 1 ? s : width + 1;
  const int stk = (w & ((int64_t(1) << ms) - 1)) != 0;
  *rup = grd & (stk | lsb);
  *frac = stk | grd;
  return r;
}

// The fp32 accumulator as a signed 24-bit significand (subnormals flushed) and
// the exponent field.
__device__ inline int64_t acc_significand(uint32_t acc, int* e) {
  *e = int((acc >> 23) & 0xff);
  const int64_t m = *e ? ((int64_t(1) << 23) | (acc & 0x7fffff)) : 0;
  return (acc >> 31) ? -m : m;
}

// The signed sum `man` (|man| < 2^32, unit 2^(dexp - 158)) rounded half to even
// to fp32 bits; underflow (and a zero sum) flushes to +0.
__device__ inline uint32_t flp_nrm(int64_t man, int dexp) {
  const uint32_t sgn = man < 0 ? 1u : 0u;
  uint32_t m32 = uint32_t((man < 0 ? -man : man) & 0xffffffff);
  if (m32 == 0) return 0;
  const int zc = __clz(m32);
  m32 <<= zc;
  int exp = dexp - zc;
  const uint32_t stk = (m32 & 0x7f) != 0, grd = (m32 >> 7) & 1, lsb = (m32 >> 8) & 1;
  const uint32_t rup = grd & (stk | lsb);
  const uint32_t eup = ((m32 >> 7) & 0xffffff) == 0xffffff;
  const uint32_t mant = (((m32 >> 8) & 0x7fffff) + rup) & 0x7fffff;
  exp += int(eup);
  if (exp >= 255) return (sgn << 31) | 0x7f800000u;
  if (exp <= 0) return 0;
  return (sgn << 31) | (uint32_t(exp) << 23) | mant;
}

// Accumulate + normalise of the 8-block MMACs (bfp16 / mx6): the accumulator and
// the exact block sums s[b] (value s * 2^(ce - 266)) aligned to the largest
// exponent, each rounded half to even, summed.
__device__ inline uint32_t block_acc(uint32_t acc, const int64_t* s, const int* ce) {
  int acc_e;
  const int64_t a_m = acc_significand(acc, &acc_e);
  const int a_e = acc_e + 127;
  int exp = a_e;
  for (int b = 0; b < 8; ++b) exp = ce[b] > exp ? ce[b] : exp;
  int rup, frac;
  int64_t man = g_shift(a_m, exp - a_e, 25, &rup, &frac) + rup;
  for (int b = 0; b < 8; ++b) man += g_shift(s[b] * 2048, exp - ce[b], 30, &rup, &frac) + rup;
  return flp_nrm(man, exp - 0x77);
}

// bf16 signed significand with the hidden bit unless the exponent field is 0, and the
// exponent (a subnormal gets 1, a zero 0).
__device__ inline void bf16_sig(uint16_t b, int64_t* man, int* e) {
  int ex = (b >> 7) & 0xff;
  const int m = b & 0x7f;
  int64_t v = m | (ex ? 0x80 : 0);
  if (ex == 0 && m) ex = 1;
  *man = (b & 0x8000) ? -v : v;
  *e = ex;
}

// The fp32 accumulator acc plus two half sums hm[h] (product units at exponent hex[h]),
// each aligned to the largest exponent with one guard bit and rounded half to even, then
// normalised once (flp_nrm).
__device__ inline uint32_t merge_halves(uint32_t acc, const int64_t* hm, const int* hex) {
  int acc_e;
  const int64_t a_m = acc_significand(acc, &acc_e);
  const int a_ex = acc_e - 127;
  const int exp = (a_ex > hex[0] && a_ex > hex[1]) ? a_ex : (hex[0] > hex[1] ? hex[0] : hex[1]);
  int rup, fr;
  int64_t man = g_shift(a_m, exp - a_ex, 25, &rup, &fr) + rup;
  for (int h = 0; h < 2; ++h) {
    int sh = exp - hex[h];
    sh = sh >= 512 ? (256 | (sh & 0xff)) : (sh & 0x1ff);
    man += g_shift(hm[h], sh, 31, &rup, &fr) + rup;
  }
  return flp_nrm(man * 2, exp + 134);
}

// One bf16 MMAC: fp32 accumulator bits acc plus the 32 products x[p * sx] * z[p * sz] (lanes
// from n_valid are zero) in two halves of 16, each half aligned to its largest exponent.
__device__ inline uint32_t mmac_bf16(uint32_t acc, const uint16_t* x, int64_t sx,
                                     const uint16_t* z, int64_t sz, int n_valid) {
  int64_t pm[32], hm[2] = {0, 0};
  int pe[32], hmax[2] = {0, 0};
  for (int p = 0; p < 32; ++p) {
    pm[p] = 0;
    pe[p] = 0;
    if (p < n_valid) {
      int64_t xm, zm;
      int xe, ze;
      bf16_sig(x[p * sx], &xm, &xe);
      bf16_sig(z[p * sz], &zm, &ze);
      pm[p] = xm * zm;
      pe[p] = (xe == 0 || ze == 0) ? 0 : xe + ze;
    }
    if (pe[p] > hmax[p >> 4]) hmax[p >> 4] = pe[p];
  }
  for (int p = 0; p < 32; ++p) {
    int rup, frac;
    hm[p >> 4] += g_shift(pm[p] * 512, hmax[p >> 4] - pe[p], 26, &rup, &frac) + rup;
  }
  const int hex[2] = {hmax[0] - 254, hmax[1] - 254};
  return merge_halves(acc, hm, hex);
}

// Zero or a normal finite fp32. The single-value MMACs below are modelled only when
// both operands and the fp32 sum are plain; otherwise they keep the fp32 add.
__device__ inline bool mmac_plain(float v) {
  const uint32_t e = (__float_as_uint(v) >> 23) & 0xff;
  return v == 0.0f || (e != 0 && e != 0xff);
}

// acc + x in the matrix unit: one bf16 MMAC of x against a staged 1.0. x must be exact in
// bf16; otherwise (and for non-plain values, see above) the fp32 add is returned.
__device__ inline float mmac_add_bf16(float acc, float x) {
  const float f = add_q(acc, x);
  if (!mmac_plain(acc) || !mmac_plain(x) || !mmac_plain(f) || (__float_as_uint(x) & 0xffff))
    return f;
  const uint16_t xb = uint16_t(__float_as_uint(x) >> 16), one = 0x3f80;
  return __uint_as_float(mmac_bf16(__float_as_uint(acc), &xb, 0, &one, 0, 1));
}

// acc + value k of bfp16 block b (group [g0, min(g0 + 8, nblk))) as one identity block MMAC
// onto the fp32 accumulator; a = the decoded value.
__device__ inline float mmac_add_bfp16(const uint8_t* in, int64_t g0, int64_t nblk, int64_t b,
                                     int k, float a, float acc) {
  const float f = add_q(a, acc);
  if (!mmac_plain(a) || !mmac_plain(acc) || !mmac_plain(f)) return f;
  int64_t s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int ce[8];
  for (int j = 0; j < 8; ++j) ce[j] = (g0 + j < nblk ? in[(g0 + j) * 9] : 0) + 127;
  s[b - g0] = int64_t(int8_t(in[b * 9 + 1 + k])) * 64;  // times the staged 1.0 = 64 * 2^-6
  return __uint_as_float(block_acc(__float_as_uint(acc), s, ce));
}

// One bfp16 block MMAC: 8 blocks of x against 8 blocks of z; block b contributes the exact
// sum of its 8 int8 products at exponent x[b].e + z[b].e.
__device__ inline uint32_t mmac_bfp16(uint32_t acc, const Bfp16Fields* x, const Bfp16Fields* z) {
  int64_t s[8];
  int ce[8];
  for (int b = 0; b < 8; ++b) {
    int64_t t = 0;
    for (int j = 0; j < 8; ++j) t += int64_t(x[b].m[j]) * z[b].m[j];
    s[b] = t;
    ce[b] = x[b].e + z[b].e;
  }
  return block_acc(acc, s, ce);
}

// One mx6 block MMAC: per pair p the two products are summed and shifted left by
// 2 - (x.s[p] + z.s[p]); block sums are scaled by 16.
__device__ inline uint32_t mmac_mx6(uint32_t acc, const Mx6Fields* x, const Mx6Fields* z) {
  int64_t s[8];
  int ce[8];
  for (int b = 0; b < 8; ++b) {
    int64_t t = 0;
    for (int p = 0; p < 8; ++p) {
      const int64_t pair = int64_t(x[b].m[2 * p]) * z[b].m[2 * p] +
                           int64_t(x[b].m[2 * p + 1]) * z[b].m[2 * p + 1];
      t += pair << (2 - x[b].s[p] - z[b].s[p]);
    }
    s[b] = t * 16;
    ce[b] = x[b].e + z[b].e;
  }
  return block_acc(acc, s, ce);
}

// fp8 (E4M3) significand with the hidden bit unless the exponent field is 0, and the
// exponent (a subnormal gets 1, a zero 0).
__host__ __device__ inline void fp8_sig(uint8_t b, int* m, int* e) {
  int ex = (b >> 3) & 0xf;
  const int mm = b & 7;
  const int sig = mm | (ex ? 8 : 0);
  if (ex == 0 && mm) ex = 1;
  *m = (b & 0x80) ? -sig : sig;
  *e = ex;
}

// One fp8 MMAC: the accumulator plus the 64 products x[p * sx] * z[p * sz] (lanes from n_valid
// are zero) in two halves of 32, each product rounded half to even at its half's largest
// exponent.
__device__ inline uint32_t mmac_fp8(uint32_t acc, const uint8_t* x, int64_t sx, const uint8_t* z,
                                    int64_t sz, int n_valid) {
  int64_t pm[64], hm[2] = {0, 0};
  int pe[64], hmax[2] = {0, 0};
  for (int p = 0; p < 64; ++p) {
    pm[p] = 0;
    pe[p] = 0;
    if (p < n_valid) {
      int xm, xe, zm, ze;
      fp8_sig(x[p * sx], &xm, &xe);
      fp8_sig(z[p * sz], &zm, &ze);
      pm[p] = int64_t(xm) * zm;
      pe[p] = (xe == 0 || ze == 0) ? 0 : xe + ze;
    }
    if (pe[p] > hmax[p >> 5]) hmax[p >> 5] = pe[p];
  }
  for (int p = 0; p < 64; ++p) {
    int rup, frac;
    hm[p >> 5] += g_shift(pm[p] << 17, hmax[p >> 5] - pe[p], 31, &rup, &frac) + rup;
  }
  const int hex[2] = {hmax[0] - 14, hmax[1] - 14};
  return merge_halves(acc, hm, hex);
}

}  // namespace npu_emu

#endif  // NPU_EMU_MMAC_H_
