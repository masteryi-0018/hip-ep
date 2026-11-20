/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Floating-point elementwise arithmetic: bf16 / f16 / fp8 / bfp16 add, sub, mul, div,
 * mac_fused, mul_add, prelu, leaky relu, square and batchnorm.
 */
#ifndef NPU_EMU_ARITH_FLOAT_H_
#define NPU_EMU_ARITH_FLOAT_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Common rules. Arithmetic is fp32: fma(a, b, c) rounds once, fp32(a * b) is a rounded product.
 * Outputs round half to even (rne) unless noted; fp8 is E4M3FN, not saturated (overflow -> NaN).
 * max / min order -0 below +0.
 * "Row-decoded" fp8: -0 becomes +0 and a NaN spreads to the 8 consecutive channels of its row.
 * Tensors are dense HWC. shape1 = {H, W, C} (host array) is x1's, the full shape; x2 / x3 may
 * be broadcast: every dimension of shape2 / shape3 / shape23 is 1 or shape1's. `swap`
 * exchanges the operand roles. Coefficients are fp32.
 * bfp16: blocks of 8 channels in 9 bytes (exponent, 8 int8 mantissas, value = m * 2^(e - 133));
 * C % 8 == 0; a C-broadcast operand has one block per pixel and uses its element 0.
 * bfp16 operands are decoded by the identity multiply: in each 64-element group (8 blocks)
 * values are rounded half to even to multiples of 2^(E - 150), E the group's largest block
 * exponent. A pixel's channels form groups from channel 0; the block of a C-broadcast operand
 * is decoded in the group of the 8 operand blocks starting at it (fewer at the end). */

/* Fused activation, applied to the fp32 result before narrowing:
 *   RELU: max(v, act0)   CLIP: min(max(v, act0), act1)   LRELU: max(v, fp32(v * act0)) */
typedef enum {
  NPU_FLOAT_ACT_NONE = 0,
  NPU_FLOAT_ACT_RELU = 1,
  NPU_FLOAT_ACT_CLIP = 2,
  NPU_FLOAT_ACT_LRELU = 3,
} npu_float_act;

/* bf16 / f16 (type = NPU_BF16 or NPU_F16) */

/* Add / Sub: out = rne(act(fp32(s1 * x1 + s2 * x2))), s = -1 if negate1 / negate2 else +1. */
int npu_elementwise_broadcasting3d_add_bf16(void* stream, const void* x1, const void* x2,
    npu_dtype type, void* out, const int64_t* shape1, const int64_t* shape2, int negate1,
    int negate2, npu_float_act act, float act0, float act1);
/* out = rne(act(fp32(x1 * x2))). */
int npu_elementwise_broadcasting3d_mul_bf16(void* stream, const void* x1, const void* x2,
    npu_dtype type, void* out, const int64_t* shape1, const int64_t* shape2, npu_float_act act,
    float act0, float act1);
/* (num, den) = swap ? (x2, x1) : (x1, x2); out = rne(fp32(num * inv(den))). inv(y) = rne(1 / y);
 * bf16: zero / subnormal y -> +-inf, |1 / y| < 2^-126 -> +-0. */
int npu_elementwise_broadcasting3d_div_bf16(void* stream, const void* x1, const void* x2,
    npu_dtype type, void* out, const int64_t* shape1, const int64_t* shape2, int swap);
/* x2 (multiplier) and x3 (addend) share shape23; out = rne(act(fma(x1, x2, x3))). */
int npu_elementwise_broadcasting3d_mac_fused_bf16(void* stream, const void* x1, const void* x2,
    const void* x3, npu_dtype type, void* out, const int64_t* shape1, const int64_t* shape23,
    npu_float_act act, float act0, float act1);

/* bfp16: operands decoded, combined in fp32, re-encoded half to even. */

/* x2 + x1 as one block MMAC: x2 is the fp32 accumulator; x2 and every value of x1's group
 * are aligned to the larger of E and x2's exponent, rounded half to even, summed and rounded
 * to fp32 (NaN, infinite or subnormal operands or sums: fp32(x1 + x2)). */
int npu_elementwise_broadcasting3d_add_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2);
/* fp32(x1 - x2), or fp32(x2 - x1) if swap; C-broadcast without swap: x1 + (-x2) as add. */
int npu_elementwise_broadcasting3d_sub_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, int swap);
/* fp32(x1 * x2). */
int npu_elementwise_broadcasting3d_mul_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2);
/* fp32(num * inv(den)) with the bf16 inv; (num, den) = swap ? (x2, x1) : (x1, x2). */
int npu_elementwise_broadcasting3d_div_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, int swap);

/* x1 bfp16; x2 / x3 bf16 or bfp16 (x*_bfp16 = 1), H / W broadcast only; out bf16 or bfp16.
 * v = fma(x1, x2, x3). act_type 0 / 7 (tanh) / 100 (clip) selects the variant, activation
 * enables it:
 *   act_type 100, activation 100: v = clamp(rne_bf16(v), clip_min, clip_max)
 *   act_type 7: v = rne_bf16(v); activation 7: v = hardware tanh table of v, as bf16
 * out = rne(v). */
int npu_elementwise_broadcasting3d_mul_add_bfp16(void* stream, const void* x1, const void* x2,
    int x2_bfp16, const void* x3, int x3_bfp16, void* out, int out_bfp16, const int64_t* shape1,
    const int64_t* shape2, const int64_t* shape3, int act_type, int activation, float clip_min,
    float clip_max);

/* fp8 (E4M3FN) */

/* Add / Sub (Sub passes negated coefficients). c1 / c2 = signed input scales over the output
 * scale. x1 is row-decoded. Output rounds half away from zero.
 * Without C broadcast (x2 row-decoded): out = act(fma(x2, c2, fp32(x1 * c1)))
 * With C broadcast (x2 converted per value, -0 and NaN kept):
 *   NONE : fma(x1, c1, fp32(x2 * c2))
 *   RELU : max(fp32(fp32(x1 * c1) + fp32(x2 * c2)), act0)
 *   CLIP : clamp(fp32(fp32(x1 * c1) + table[code(x2)]), act0, act1)
 *   LRELU: q = fma(x1, table[256], table[code(x2)]); fma(|q|, table[257], q)
 * table: device array of 258 floats (256 per-code biases, then the two LRELU factors); needed
 * only for C broadcast with CLIP or LRELU, else may be null. */
int npu_elementwise_broadcasting3d_add_fp8(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, float c1, float c2,
    npu_float_act act, float act0, float act1, const float* table);
/* Output half away. Without C broadcast: fp32(fp32(x1 * x2) * scale), no row decode.
 * With C broadcast (both row-decoded): fp32(x1 * fp32(x2 * scale)). */
int npu_elementwise_broadcasting3d_mul_fp8(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, float scale);
/* Both row-decoded; (num, den) = swap ? (x2, x1) : (x1, x2);
 * out = fp32(fp32(num * scale) * inv(den)), half away; inv = the hardware fp32 reciprocal. */
int npu_elementwise_broadcasting3d_div_fp8(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, float scale, int swap);
/* Output half to even. act0 = alpha for LRELU. out = act(v) with
 * without C broadcast (all row-decoded): v = fma(x3, c4, fp32(x2 * fp32(x1 * c0)))
 * with C broadcast (x2 / x3 per-pixel scalars, -0 -> +0, NaN per value):
 *   v = fma(x1, fp32(x2 * c0), fp32(x3 * c4)) */
int npu_elementwise_broadcasting3d_mac_fused_fp8(void* stream, const void* x1, const void* x2,
    const void* x3, void* out, const int64_t* shape1, const int64_t* shape23, float c0, float c4,
    npu_float_act act, float act0, float act1);

/* unary / per-channel (n elements, flat) */

/* BF16 / F16: a = rne(alpha) in `type`; out = rne(max(x, x * a)). */
int npu_elementwise_leakyrelu_bf16(void* stream, const void* in, npu_dtype type, void* out,
                                   int64_t n, float alpha);
/* x row-decoded (rows = 8 consecutive elements);
 * out = fp8(max(fp32(x * pos), fp32(x * neg))), half away. */
int npu_elementwise_leakyrelu_fp8(void* stream, const void* in, void* out, int64_t n,
                                  float pos, float neg);
/* BF16 / F16: out = rne(fma(min(x, 0), D[i % C], x)); D = device array of C floats holding
 * fp32(slope - 1). */
int npu_elementwise_prelu_bf16(void* stream, const void* in, npu_dtype type, void* out,
                               int64_t n, const float* D, int64_t C);
/* x row-decoded; out = fp8(fma(min(x, 0), D[i % C], fp32(x * P))), half away;
 * D = device array of C floats holding fp32(fp32(slope - 1) * P). */
int npu_elementwise_prelu_fp8(void* stream, const void* in, void* out, int64_t n, float P,
                              const float* D, int64_t C);
/* v = fp32(x * x); F16: |v| < 2^-14 -> +0; out = rne(v). */
int npu_elementwise_square_bf16(void* stream, const void* in, npu_dtype type, void* out,
                                int64_t n);
/* x row-decoded; out = fp8(fp32(x * fp32(x * scale))), half away. */
int npu_elementwise_square_fp8(void* stream, const void* in, void* out, int64_t n,
                               float scale);
/* type BF16 / F16 / FP8; channel c = i % C. cw / cb: device arrays of C floats (values
 * representable in bf16 for BF16, f16 for F16 and FP8). acc = fma(x, cw[c], cb[c]);
 *   act 0: out = rne(acc)
 *   act 1 (RELU): 16-bit: max(rne(acc), +0) (-0 -> +0); FP8: rne(max(acc, 0))
 *   act 3 (LRELU): rne(max(acc, fp32(acc * alpha)))
 * FP8 inputs are row-decoded over 8 consecutive elements; FP8 output rounds half to even. */
int npu_batchnorm2d_stream_bf16(void* stream, const void* in, npu_dtype type, void* out,
                                int64_t n, int64_t C, const float* cw, const float* cb,
                                int act, float alpha);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_ARITH_FLOAT_H_
