/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Sum / mean / product / L2 reductions and cumulative sum.
 *
 * Tensors are contiguous H x W x C; the output drops every reduced dimension
 * (axis_flags bit 0 = C, bit 1 = W, bit 2 = H, any non-zero combination).
 * Float reductions keep one fp32 accumulator per lane (channel c is lane c % 64
 * of group c / 64) and step through the reduced coordinates in the nest given
 * per function, outermost first (h = rows, w = columns, g = 64-channel groups),
 * rounding every step. Reducing C then adds lanes l and l + 32 and applies one
 * vector reduce-add to the 32 sums. Integer sums wrap modulo 2^32.
 *
 * Special values: fp32 steps follow the NaN rules of common/mac.h [default]; the vector
 * reduce-add keeps the rules of common/mac.h; float -> int conversions saturate and give 0 for
 * NaN (common/int_round.h); the int64 affine map of the integer sums wraps on overflow [default].
 */
#ifndef NPU_EMU_REDUCE_H_
#define NPU_EMU_REDUCE_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* acc = x * scale on the first step, then acc = fma(x, scale, acc); result
 * rounded half to even to dtype (BF16 / F16). scale = 1 (sum) or 1 / N (mean).
 * Nest per axis_flags: 1 g | 2 w | 3 g,w | 4 h | 5 g,h | 6 w,h | 7 g,h,w. */
int npu_reduce_sum_c64_bf16(void* stream, const void* in, npu_dtype dtype,
                            void* out, int64_t H, int64_t W, int64_t C,
                            int axis_flags, float scale);

/* E4M3FN in and out: acc = fma(x, scale, acc) from +0, clamped to [-448, 448],
 * rounded half to even. A NaN input makes its group of 8 channels NaN.
 * Nest as npu_reduce_sum_c64_bf16, except axis_flags 7: the tensor is cut into
 * h_tiles x w_tiles blocks (H % h_tiles == W % w_tiles == 0) and the nest is
 * H block, W block, g, h, w. h_tiles / w_tiles are ignored otherwise. */
int npu_reduce_sum_c64_fp8(void* stream, const void* in, void* out, int64_t H,
                           int64_t W, int64_t C, int axis_flags, float scale,
                           int64_t h_tiles, int64_t w_tiles);

/* Integer sum / mean, S = sum of the reduced inputs:
 *   axis_flags 7:  a8: v = c0 * wrap32(S) + c2;  a16: v = c0 * S + c2  (int64)
 *   otherwise:     v = wrap32((int32)c2 + c0 * S)
 *   out = sat(rne(v * 2^-c1)) to out_type.
 * a8: I8 / U8, 0 <= c0 <= 127; a16: I16 / U16, 0 <= c0 <= 255. 0 <= c1 <= 62. */
int npu_reduce_sum_c64_a8(void* stream, const void* in, npu_dtype in_type,
                          void* out, npu_dtype out_type, int64_t H, int64_t W,
                          int64_t C, int axis_flags, int c0, int c1,
                          int64_t c2);
int npu_reduce_sum_c64_a16(void* stream, const void* in, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t H,
                           int64_t W, int64_t C, int axis_flags, int c0,
                           int c1, int64_t c2);

/* out = sat(rne(fma(wrap32(S), c1, c0) * 2^-shift)), I32 / U32 in and out,
 * -31 <= shift <= 31. */
int npu_qdq_reduce_sum_c64_a32(void* stream, const void* in, npu_dtype in_type,
                               void* out, npu_dtype out_type, int64_t H,
                               int64_t W, int64_t C, int axis_flags, float c0,
                               float c1, int shift);

/* Product from 1, every step rounded to dtype (BF16 / F16). Reducing C ends
 * with the tree v[i] = v[i] * v[i + s], s = 32, 16, ..., 1. C % 64 == 0.
 * Nest: 1 g | 2 w | 3 w,g | 4 h | 5 h,g | 6 h,w | 7 h,w,g. */
int npu_reduce_prod_bf16(void* stream, const void* in, npu_dtype dtype,
                         void* out, int64_t H, int64_t W, int64_t C,
                         int axis_flags);

/* acc = fma(x, x, acc) from +0. Reducing C: fp32 sqrt of the lane sum, rounded
 * to dtype; otherwise acc is rounded to dtype and the hardware 16-bit sqrt
 * table applies. dtype: BF16 / F16; C % 64 == 0.
 * Nest: 1 g | 2 w | 3 w,g | 4 h | 5 h,g | 6 w,h | 7 h,w,g. */
int npu_reduce_l2_bf16(void* stream, const void* in, npu_dtype dtype,
                       void* out, int64_t H, int64_t W, int64_t C,
                       int axis_flags);

/* Dequantize, reduce in reduce_type (BF16 / F16), quantize:
 *   x'  = round(fma(x, dq_scale, fp32(-dq_zero_point * dq_scale)))
 *   y   = npu_reduce_prod_bf16 / npu_reduce_l2_bf16 of x'
 *   out = sat(rne(fma(y, q_scale, q_zero_point))) to out_type
 * The prod variants walk axis_flags 6 as w,h. in_type / out_type: I8 / U8 (a8)
 * or I16 / U16 (a16). C % 64 == 0. */
int npu_qdq_reduce_prod_a8(void* stream, const void* in, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t H, int64_t W,
                           int64_t C, int axis_flags, npu_dtype reduce_type,
                           float dq_scale, float dq_zero_point, float q_scale,
                           float q_zero_point);
/* The 16-bit variant of npu_qdq_reduce_prod_a8. */
int npu_qdq_reduce_prod_a16(void* stream, const void* in, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t H,
                            int64_t W, int64_t C, int axis_flags,
                            npu_dtype reduce_type, float dq_scale,
                            float dq_zero_point, float q_scale,
                            float q_zero_point);
/* See npu_qdq_reduce_prod_a8. */
int npu_qdq_reduce_l2_a8(void* stream, const void* in, npu_dtype in_type,
                         void* out, npu_dtype out_type, int64_t H, int64_t W,
                         int64_t C, int axis_flags, npu_dtype reduce_type,
                         float dq_scale, float dq_zero_point, float q_scale,
                         float q_zero_point);
/* The 16-bit variant of npu_qdq_reduce_l2_a8. */
int npu_qdq_reduce_l2_a16(void* stream, const void* in, npu_dtype in_type,
                          void* out, npu_dtype out_type, int64_t H, int64_t W,
                          int64_t C, int axis_flags, npu_dtype reduce_type,
                          float dq_scale, float dq_zero_point, float q_scale,
                          float q_zero_point);

/* Sum of the quantized product of ifm1 (H x W x C) and ifm2 (H2 x W2 x C2, each
 * dimension 1 or equal to ifm1's). Per step, i1 / i2 the integers, acc from +0:
 *   a8:  t = fma(i2, c0, c1); acc = fma(i2, c2, acc); acc = fma(i1, t, acc)
 *   a16: r = fp32(fp32(i1 * i2) * c0); r = fma(i1, c1, r); r = fma(i2, c2, r);
 *        acc = acc + r
 * out = sat(rne(acc + c3)) to out_type.
 * Nest: 1 g | 2 w | 3 g,w | 4 h | 5 h,g | 6 h,w | 7 h,w,g.
 * in_type (both inputs) / out_type: I8 / U8 (a8) or I16 / U16 (a16). */
int npu_qdq_elementwise_broadcasting3d_mul_reducesum_a8(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type,
    void* out, npu_dtype out_type, int64_t H, int64_t W, int64_t C, int64_t H2,
    int64_t W2, int64_t C2, int axis_flags, float c0, float c1, float c2,
    float c3);
int npu_qdq_elementwise_broadcasting3d_mul_reducesum_a16(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type,
    void* out, npu_dtype out_type, int64_t H, int64_t W, int64_t C, int64_t H2,
    int64_t W2, int64_t C2, int axis_flags, float c0, float c1, float c2,
    float c3);

/* H x W tensor, axis 0 = along H, 1 = along W. s = wrap32(running sum), from
 * the far end when reverse, excluding the current element when exclusive;
 * out = sat(rne(fma(s, c1, c0))). I8 / U8 / I16 / U16, same width in and out. */
int npu_qdq_cumsum(void* stream, const void* in, npu_dtype in_type, void* out,
                   npu_dtype out_type, int64_t H, int64_t W, int axis,
                   int exclusive, int reverse, float c0, float c1);

/* Test only: out[i] = the 16-bit sqrt table of npu_reduce_l2_bf16 applied to the
 * bit pattern in[i]; dtype BF16 / F16. */
int npu_test_reduce_sqrt16(void* stream, const void* in, npu_dtype dtype,
                           void* out, int64_t n);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_REDUCE_H_
