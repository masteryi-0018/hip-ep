/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Block floating point: bfp16 (9-byte blocks: exponent, 8 int8 mantissas) and mx6
 * (12-byte blocks: exponent, 8 pair shifts, 16 5-bit sign-magnitude mantissas).
 *
 * Special values ([measured] against the NPU, [unverified] otherwise): an all-zero bfp16 block
 * gets exponent 0, and the mx6 zero-block rule below [measured on cast / gemm]; the identity
 * decode's underflow to +0 and exponent-255 group NaN [measured on the bf16 cast]; encoding
 * NaN or infinite inputs [unverified].
 */
#ifndef NPU_EMU_BLOCK_FP_H_
#define NPU_EMU_BLOCK_FP_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/float_cvt.h"

namespace npu_emu {

// For a block of `n` fp32 values: shared exponent me = max of the biased exponent fields
// (each clamped to >= 1); per subtile of `sub` values a shift s = min(min over the subtile
// of (me - e_i), max_shift); each mantissa is the 24-bit significand shifted right by
// me - e_i + 25 - mbits - s with rounding, saturated to 2^(mbits-1) - 1, and 0 when that
// shift exceeds 24. A set sign bit marks the value negative, including -0.0.
struct BlockOut {
  int mag[16];
  bool neg[16];
  int shift[8];
  int exp;
};

__device__ inline void encode_block(const float* v, int n, int mbits, int sub, int max_shift,
                                    Rnd rnd, BlockOut* o) {
  int e[16];
  uint32_t sig[16];
  int me = 0;
  for (int k = 0; k < n; ++k) {
    const uint32_t b = f32_bits(v[k]);
    const int f = (b >> 23) & 0xff;
    sig[k] = (f > 0 ? 0x800000u : 0u) | (b & 0x7fffff);
    e[k] = f > 1 ? f : 1;
    o->neg[k] = b >> 31;
    me = e[k] > me ? e[k] : me;
  }
  const int bm_max = (1 << (mbits - 1)) - 1;
  for (int s = 0; s < n / sub; ++s) {
    int mn = 1 << 30;
    for (int k = s * sub; k < (s + 1) * sub; ++k) mn = me - e[k] < mn ? me - e[k] : mn;
    o->shift[s] = mn < max_shift ? mn : max_shift;
  }
  for (int k = 0; k < n; ++k) {
    const int sh = me - e[k] + 25 - mbits - o->shift[k / sub];
    uint64_t q = shift_round_mag(sig[k], sh, rnd, o->neg[k]);
    if (q > uint64_t(bm_max)) q = bm_max;
    o->mag[k] = sh > 24 ? 0 : int(q);
  }
  o->exp = me;
}

// bfp16: 8 values, 8-bit two's complement mantissas. An all-zero block gets exponent 0.
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

struct Bfp16Fields {
  int e;
  int m[8];
};

__host__ __device__ inline void bfp16_fields(const uint8_t* blk, Bfp16Fields* o) {
  o->e = blk[0];
  for (int k = 0; k < 8; ++k) o->m[k] = int8_t(blk[1 + k]);
}

// Value k of a bfp16 block, exactly: m * 2^(e - 133).
__device__ inline float decode_bfp16(const uint8_t* blk, int k) {
  return ldexpf(float(int8_t(blk[1 + k])), int(blk[0]) - 133);
}

// bfp16 identity-multiply decode of value k of block b in the 64-element group of blocks
// [g0, min(g0 + 8, nblk)): aligned to a 24-bit window under the group's largest exponent E,
// the mantissa of a block with exponent e is rounded half to even to a multiple of
// 2^(E - e - 17). A zero, or a result below 2^-126 (the multiply's underflow), is +0.
__device__ inline float decode_bfp16_identity(const uint8_t* in, int64_t g0, int64_t nblk,
                                              int64_t b, int k) {
  int emax = 0;
  for (int64_t j = g0; j < g0 + 8 && j < nblk; ++j) emax = in[j * 9] > emax ? in[j * 9] : emax;
  const int e = in[b * 9];
  int m = int8_t(in[b * 9 + 1 + k]);
  const int sh = emax - e - 17;
  if (sh > 8) {
    m = 0;  // |m| <= 128: below half of the window's unit
  } else if (sh > 0) {
    m = int(shift_round(m, sh, kHalfEven)) * (1 << sh);
  }
  const float v = ldexpf(float(m), e - 133);
  return fabsf(v) < 0x1p-126f ? 0.0f : v;
}

// decode_bfp16_identity with groups of 8 blocks from the buffer start; a block exponent 255
// in the group makes the whole group the NaN 0x7f810000.
__device__ inline float decode_bfp16_group(const uint8_t* in, int64_t b, int64_t nblk, int k) {
  const int64_t g0 = b & ~int64_t(7);
  for (int64_t j = g0; j < g0 + 8 && j < nblk; ++j)
    if (in[j * 9] == 255) return bits_f32(0x7f810000u);
  return decode_bfp16_identity(in, g0, nblk, b, k);
}

// decode_bfp16_group rounded half to even to bf16 (the bfp16 -> bf16 cast).
__device__ inline uint16_t decode_bfp16_bf16(const uint8_t* in, int64_t b, int64_t nblk, int k) {
  return f32_to_bf16(decode_bfp16_group(in, b, nblk, k), kHalfEven);
}

// mx6: 16 values, 5-bit sign-magnitude mantissas, one shift bit per pair,
// packed little-endian: exp | shifts << 8 | m_k << (16 + 5k). A block whose magnitudes and
// shifts are all zero and whose exponent is odd is written as 12 zero bytes.
__device__ inline void encode_mx6(const float* v, Rnd rnd, uint8_t* blk) {
  BlockOut o;
  encode_block(v, 16, 5, 2, 1, rnd, &o);
  bool zero = o.exp & 1;
  for (int k = 0; k < 16; ++k) zero &= o.mag[k] == 0;
  for (int s = 0; s < 8; ++s) zero &= o.shift[s] == 0;
  if (zero) {
    for (int j = 0; j < 12; ++j) blk[j] = 0;
    return;
  }
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

// Fields of an mx6 block: exponent, signed magnitudes, pair shifts.
struct Mx6Fields {
  int e;
  int m[16];
  int s[8];
};

__host__ __device__ inline void mx6_fields(const uint8_t* blk, Mx6Fields* o) {
  uint32_t w[3];
  for (int j = 0; j < 3; ++j)
    w[j] = uint32_t(blk[4 * j]) | uint32_t(blk[4 * j + 1]) << 8 | uint32_t(blk[4 * j + 2]) << 16 |
           uint32_t(blk[4 * j + 3]) << 24;
  o->e = w[0] & 0xff;
  for (int p = 0; p < 8; ++p) o->s[p] = (w[0] >> (8 + p)) & 1;
  for (int k = 0; k < 16; ++k) {
    const int bit = 16 + 5 * k;
    uint32_t code = w[bit / 32] >> (bit % 32);
    if (bit % 32 > 27) code |= w[bit / 32 + 1] << (32 - bit % 32);
    o->m[k] = (code & 0x10) ? -int(code & 0xf) : int(code & 0xf);
  }
}

// Value k of an mx6 block; zero magnitudes decode to +0.
__device__ inline float decode_mx6(const uint8_t* blk, int k) {
  Mx6Fields f;
  mx6_fields(blk, &f);
  if (f.m[k] == 0) return 0.0f;
  return ldexpf(float(f.m[k]), f.e - f.s[k / 2] - 130);
}

}  // namespace npu_emu

#endif  // NPU_EMU_BLOCK_FP_H_
