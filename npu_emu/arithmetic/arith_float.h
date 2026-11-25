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
 * Tensors are dense HWC. shape1 = {H, W, C} (host array, every dimension >= 0; an empty
 * tensor is a no-op) is x1's, the full shape; x2 / x3 may be broadcast: every dimension of
 * shape2 / shape3 / shape23 is 1 or shape1's. Pointers may not be null unless noted.
 * Coefficients are fp32.
 * bfp16: blocks of 8 channels in 9 bytes (exponent, 8 int8 mantissas, value = m * 2^(e - 133));
 * C % 8 == 0; a C-broadcast operand has one block per pixel and uses its element 0.
 * bfp16 operands are decoded by the identity multiply: in each 64-element group (8 blocks)
 * values are rounded half to even to multiples of 2^(E - 150), E the group's largest block
 * exponent. A pixel's channels form groups from channel 0; the block of a C-broadcast operand
 * is decoded in the group of the 8 operand blocks starting at it (fewer at the end). */

/* Fused activation of add / mul / mac_fused (bf16 and fp8): `activation` takes the codes
 * 0 none, 1 relu, 2 relu6, 3 leaky relu (anything else: -1), applied to the fp32 result
 * before narrowing:
 *   1: max(v, activation_coeff0)   2: min(max(v, activation_coeff0), activation_coeff1)
 *   3: max(v, fp32(v * activation_coeff0)) */

/* bf16 / f16 (dtype = NPU_BF16 or NPU_F16, shared by every tensor) */

/* Add / Sub: out = rne(act(fp32(s1 * x1 + s2 * x2))), s1 = -1 if negate_flag, s2 = -1 if
 * is_sub_flag, else +1. */
int npu_elementwise_broadcasting3d_add_bf16(void* stream, const void* x1, npu_dtype dtype,
    const void* x2, void* out, const int64_t* shape1, const int64_t* shape2, int is_sub_flag,
    int negate_flag, int activation, float activation_coeff0, float activation_coeff1);
/* out = rne(act(fp32(x1 * x2))). */
int npu_elementwise_broadcasting3d_mul_bf16(void* stream, const void* x1, npu_dtype dtype,
    const void* x2, void* out, const int64_t* shape1, const int64_t* shape2, int activation,
    float activation_coeff0, float activation_coeff1);
/* (num, den) = swap_inputs ? (x2, x1) : (x1, x2); out = rne(fp32(num * inv(den))).
 * inv(y) = rne(1 / y); bf16: zero / subnormal y -> +-inf, |1 / y| < 2^-126 -> +-0. */
int npu_elementwise_broadcasting3d_div_bf16(void* stream, const void* x1, npu_dtype dtype,
    const void* x2, void* out, const int64_t* shape1, const int64_t* shape2, int swap_inputs);
/* x2 (multiplier) and x3 (addend) share shape23; out = rne(act(fma(x1, x2, x3))). */
int npu_elementwise_broadcasting3d_mac_fused_bf16(void* stream, const void* x1, npu_dtype dtype,
    const void* x2, const void* x3, void* out, const int64_t* shape1, const int64_t* shape23,
    int activation, float activation_coeff0, float activation_coeff1);

/* bfp16: operands decoded, combined in fp32, re-encoded half to even. */

/* x2 + x1 as one block MMAC: x2 is the fp32 accumulator; x2 and every value of x1's group
 * are aligned to the larger of E and x2's exponent, rounded half to even, summed and rounded
 * to fp32 (NaN, infinite or subnormal operands or sums: fp32(x1 + x2)). */
int npu_elementwise_broadcasting3d_add_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2);
/* fp32(x1 - x2), or fp32(x2 - x1) if swap_inputs; C-broadcast without swap: x1 + (-x2) as
 * add. */
int npu_elementwise_broadcasting3d_sub_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, int swap_inputs);
/* fp32(x1 * x2). */
int npu_elementwise_broadcasting3d_mul_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2);
/* fp32(num * inv(den)) with the bf16 inv; (num, den) = swap_inputs ? (x2, x1) : (x1, x2). */
int npu_elementwise_broadcasting3d_div_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, int swap_inputs);

/* x1 bfp16; x2_type / x3_type / out_type NPU_BF16 or NPU_BFP16; x2 / x3 H / W broadcast only.
 * v = fma(x1, x2, x3). activation_type 0 / 7 (tanh) / 100 (clip) selects the variant;
 * activation (0 or activation_type) enables it:
 *   activation_type 100, activation 100: v = clamp(rne_bf16(v), clip_min, clip_max)
 *   activation_type 7: v = rne_bf16(v); activation 7: v = hardware tanh table of v, as bf16
 * out = rne(v). clip_min / clip_max: bf16 bit patterns. */
int npu_elementwise_broadcasting3d_mul_add_bfp16(void* stream, const void* x1, const void* x2,
    npu_dtype x2_type, const void* x3, npu_dtype x3_type, void* out, npu_dtype out_type,
    const int64_t* shape1, const int64_t* shape2, const int64_t* shape3, int activation_type,
    int activation, uint16_t clip_min, uint16_t clip_max);

/* fp8 (E4M3FN) */

/* Add / Sub (Sub passes negated coefficients). ifm1_coeff / ifm2_coeff = signed input scales
 * over the output scale (c1 / c2 below). x1 is row-decoded. Output rounds half away from zero.
 * Without C broadcast (x2 row-decoded): out = act(fma(x2, c2, fp32(x1 * c1)))
 * With C broadcast (x2 converted per value, -0 and NaN kept):
 *   0: fma(x1, c1, fp32(x2 * c2))
 *   1: max(fp32(fp32(x1 * c1) + fp32(x2 * c2)), activation_coeff0)
 *   2: clamp(fp32(fp32(x1 * c1) + table[code(x2)]), activation_coeff0, activation_coeff1)
 *   3: q = fma(x1, table[256], table[code(x2)]); fma(|q|, table[257], q)
 * table: device array of 258 floats (256 per-code biases, then the two leaky relu factors);
 * required only for C broadcast with activation 2 or 3, else may be null. */
int npu_elementwise_broadcasting3d_add_fp8(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, float ifm1_coeff, float ifm2_coeff,
    int activation, float activation_coeff0, float activation_coeff1, const void* table);
/* Output half away. Without C broadcast: fp32(fp32(x1 * x2) * operation_coeff), no row
 * decode. With C broadcast (both row-decoded): fp32(x1 * fp32(x2 * operation_coeff)). */
int npu_elementwise_broadcasting3d_mul_fp8(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, float operation_coeff);
/* Both row-decoded; (num, den) = swap_inputs ? (x2, x1) : (x1, x2);
 * out = fp32(fp32(num * operation_coeff) * inv(den)), half away; inv = the hardware fp32
 * reciprocal. */
int npu_elementwise_broadcasting3d_div_fp8(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, float operation_coeff,
    int swap_inputs);
/* Output half to even. out = act(v) with
 * without C broadcast (all row-decoded): v = fma(x3, c4, fp32(x2 * fp32(x1 * c0)))
 * with C broadcast (x2 / x3 per-pixel scalars, -0 -> +0, NaN per value):
 *   v = fma(x1, fp32(x2 * c0), fp32(x3 * c4)) */
int npu_elementwise_broadcasting3d_mac_fused_fp8(void* stream, const void* x1, const void* x2,
    const void* x3, void* out, const int64_t* shape1, const int64_t* shape23, float c0, float c4,
    int activation, float activation_coeff0, float activation_coeff1);

/* unary / per-channel (n elements, flat; channel c = i % C, C >= 1) */

/* BF16 / F16: a = rne(alpha) in `dtype`; out = rne(max(x, x * a)). */
int npu_elementwise_leakyrelu_bf16(void* stream, const void* in, npu_dtype dtype, void* out,
                                   int64_t n, float alpha);
/* x row-decoded (rows = 8 consecutive elements);
 * out = fp8(max(fp32(x * positive_coeff), fp32(x * negative_coeff))), half away. */
int npu_elementwise_leakyrelu_fp8(void* stream, const void* in, void* out, int64_t n,
                                  float positive_coeff, float negative_coeff);
/* BF16 / F16: out = rne(fma(min(x, 0), D[c], x)); D = prelu_slope_delta, a device array of C
 * floats holding fp32(slope - 1). */
int npu_elementwise_prelu_bf16(void* stream, const void* in, npu_dtype dtype, void* out,
                               int64_t n, int64_t C, const void* prelu_slope_delta);
/* x row-decoded; out = fp8(fma(min(x, 0), D[c], fp32(x * positive_scale))), half away;
 * D = prelu_scaled_slope_delta, a device array of C floats holding
 * fp32(fp32(slope - 1) * positive_scale). */
int npu_elementwise_prelu_fp8(void* stream, const void* in, void* out, int64_t n, int64_t C,
                              float positive_scale, const void* prelu_scaled_slope_delta);
/* v = fp32(x * x); F16: |v| < 2^-14 -> +0; out = rne(v). */
int npu_elementwise_square_bf16(void* stream, const void* in, npu_dtype dtype, void* out,
                                int64_t n);
/* x row-decoded; out = fp8(fp32(x * fp32(x * ifm_coeff))), half away. */
int npu_elementwise_square_fp8(void* stream, const void* in, void* out, int64_t n,
                               float ifm_coeff);
/* dtype BF16 / F16 / FP8. cw / cb: device arrays of C 16-bit values, bf16 for BF16 and f16 for
 * F16 and FP8. acc = fma(x, cw[c], cb[c]);
 *   activation 0: out = rne(acc)
 *   activation 1 (relu): 16-bit: max(rne(acc), +0) (-0 -> +0); FP8: rne(max(acc, 0))
 *   activation 3 (leaky relu): rne(max(acc, fp32(acc * relu_alpha)))
 * FP8 inputs are row-decoded over 8 consecutive elements; FP8 output rounds half to even. */
int npu_batchnorm2d_stream_bf16(void* stream, const void* in, npu_dtype dtype, void* out,
                                int64_t n, int64_t C, const void* cw, const void* cb,
                                int activation, float relu_alpha);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_ARITH_FLOAT_H_
