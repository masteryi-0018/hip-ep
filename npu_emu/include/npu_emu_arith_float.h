/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 floating-point elementwise arithmetic
 * wrappers (arith_float group): bf16 / f16 / fp8 / bfp16 add, sub, mul, div,
 * mac_fused, mul_add, prelu, leaky relu, square and batchnorm2d_stream_bf16.
 * Conventions as in npu_emu.h. The arithmetic follows the kernel sources
 * (L1/include/aie4_misc/) and was checked byte for byte against aiesim outputs
 * of the mllib nightly regression unless marked "unverified".
 *
 * Rules shared by these kernels:
 *   - Arithmetic is fp32 (accfloat): a product of two 16-bit floats, of two
 *     fp8 values or of bfp16 mantissas is exact; every add, multiply by an fp32
 *     coefficient and multiply-add is rounded once to fp32 (half to even) at the
 *     points listed per function. "fma(a, b, c)" is one rounding, "fp32(a * b)"
 *     a rounded product.
 *   - Narrowing to the output type: bf16 / f16 / bfp16 round half to even
 *     (conv_even); fp8 rounds as the wrapper sets the tile: half away from zero
 *     (symmetric_inf) or half to even (conv_even), noted per function. fp8 is
 *     E4M3FN and is NOT saturated: magnitudes that round past 448 give NaN
 *     (overflow unverified on the NPU). bfp16 blocks are encoded as
 *     npu_cast_bf2bfp16 (an all-zero block gets exponent 0).
 *   - fp32 VMAX / VMIN (activations, prelu) order -0 below +0: min(-0, +0) is -0
 *     (aiesim: prelu with x = -0). NaN in a max / min gives the canonical NaN
 *     (unverified).
 *   - fp8 inputs that the kernel decodes with mul_8x8_8x8 (an 8x8 identity
 *     matmul) become fp32 with -0 turned into +0, and a NaN makes the 8 values of
 *     its row NaN. A row is 8 consecutive channels of one pixel (c / 8); a
 *     broadcast (C = 1) operand's row partners are NPU padding bytes and are
 *     taken as non-NaN here. Where a kernel converts an fp8 value directly
 *     (noted per function) -0 and NaN stay per element.
 *   - div: the kernels multiply by a reciprocal. bf16: inv(y) is the AIE4 bf16
 *     reciprocal table (mllib inv_lut.inv_bf16_aie4) = bf16 RNE of the fp32
 *     IEEE 1/y, except that a zero or subnormal y gives +-inf, a result below
 *     2^-126 gives +-0 and a NaN y gives y quieted; f16: f16 RNE of the fp32
 *     1/y; fp8: the fp32 IEEE 1/y.
 *   - NaN / infinity / overflow / subnormal inputs are otherwise unverified
 *     (no aiesim case has them); the kernels' fp32 arithmetic is taken as IEEE.
 *
 * Shapes: tensors are logical HWC (C innermost), the valid region densely
 * packed. In the 3D-broadcasting wrappers x1 is the full H x W x C operand
 * (the NPU's ifm1); x2 (and x3) may be broadcast: bit 0 / 1 / 2 of *_bcast set
 * means size 1 along C / W / H. The operands are in the NPU's order: the
 * wrapper's swap_inputs is passed as `swap` where the kernel uses it (x2 is then
 * the logical first operand); coefficients are the fp32 words the kernel reads
 * (RTP / coeff / wts buffers), not ONNX scales. bfp16 tensors are blocks of 8
 * channels (9 bytes: exponent, 8 int8 mantissas, value = m * 2^(e - 133)), so
 * C % 8 == 0 and C counts the padded channels; a C-broadcast bfp16 operand has
 * one block per pixel and uses its element 0.
 */
#ifndef NPU_EMU_ARITH_FLOAT_H_
#define NPU_EMU_ARITH_FLOAT_H_

#include <stdint.h>

#include "npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fused activation of the bf16 / fp8 binary wrappers, applied to the fp32
 * result before narrowing (mllib::ActivationType; CLIP is relu6 / clamp):
 *   RELU: max(v, act0)   CLIP: min(max(v, act0), act1)
 *   LRELU: max(v, fp32(v * act0))  (act0 = alpha) */
typedef enum {
  NPU_FLOAT_ACT_NONE = 0,
  NPU_FLOAT_ACT_RELU = 1,
  NPU_FLOAT_ACT_CLIP = 2,
  NPU_FLOAT_ACT_LRELU = 3,
} npu_float_act;

/* ---- bf16 / f16 3D broadcasting (type BF16 or F16) ------------------------ */

/* elementwise_broadcasting3d_add_bf16 (Add and Sub; Sub is the same kernel):
 * v = fp32(s1 * x1 + s2 * x2) with s1 = -1 if negate1 (RTP negate_flag),
 * s2 = -1 if negate2 (RTP is_sub_flag), else +1; out = rne(act(v)). */
int npu_elementwise_broadcasting3d_add_bf16(void* stream, const void* x1, const void* x2,
                                            int x2_bcast, npu_dtype type, void* out,
                                            int64_t H, int64_t W, int64_t C, int negate1,
                                            int negate2, npu_float_act act, float act0,
                                            float act1);
/* elementwise_broadcasting3d_mul_bf16: out = rne(act(fp32(x1 * x2))). */
int npu_elementwise_broadcasting3d_mul_bf16(void* stream, const void* x1, const void* x2,
                                            int x2_bcast, npu_dtype type, void* out,
                                            int64_t H, int64_t W, int64_t C,
                                            npu_float_act act, float act0, float act1);
/* elementwise_broadcasting3d_div_bf16: (num, den) = swap ? (x2, x1) : (x1, x2);
 * out = rne(fp32(num * inv(den))) with inv the bf16 / f16 reciprocal above. */
int npu_elementwise_broadcasting3d_div_bf16(void* stream, const void* x1, const void* x2,
                                            int x2_bcast, npu_dtype type, void* out,
                                            int64_t H, int64_t W, int64_t C, int swap);
/* elementwise_broadcasting3d_mac_fused_bf16: x2 (multiplier) and x3 (addend)
 * share x23_bcast; out = rne(act(fma(x1, x2, x3))). */
int npu_elementwise_broadcasting3d_mac_fused_bf16(void* stream, const void* x1,
                                                  const void* x2, const void* x3,
                                                  int x23_bcast, npu_dtype type, void* out,
                                                  int64_t H, int64_t W, int64_t C,
                                                  npu_float_act act, float act0, float act1);

/* ---- bfp16 3D broadcasting -------------------------------------------------
 * x1, x2 and out are bfp16 (see above); values are decoded exactly, combined
 * in fp32 and re-encoded half to even. */

/* elementwise_broadcasting3d_add_bfp16: fp32(x1 + x2). */
int npu_elementwise_broadcasting3d_add_bfp16(void* stream, const void* x1, const void* x2,
                                             int x2_bcast, void* out, int64_t H, int64_t W,
                                             int64_t C);
/* elementwise_broadcasting3d_sub_bfp16: fp32(x1 - x2), or fp32(x2 - x1) if swap. */
int npu_elementwise_broadcasting3d_sub_bfp16(void* stream, const void* x1, const void* x2,
                                             int x2_bcast, void* out, int64_t H, int64_t W,
                                             int64_t C, int swap);
/* elementwise_broadcasting3d_mul_bfp16: fp32(x1 * x2) (exact). */
int npu_elementwise_broadcasting3d_mul_bfp16(void* stream, const void* x1, const void* x2,
                                             int x2_bcast, void* out, int64_t H, int64_t W,
                                             int64_t C);
/* elementwise_broadcasting3d_div_bfp16: (num, den) = swap ? (x2, x1) : (x1, x2);
 * fp32(num * inv(den)) with the bf16 reciprocal table (operands are exact in
 * bf16), then bfp16 directly (no bf16 rounding of the quotient). */
int npu_elementwise_broadcasting3d_div_bfp16(void* stream, const void* x1, const void* x2,
                                             int x2_bcast, void* out, int64_t H, int64_t W,
                                             int64_t C, int swap);

/* elementwise_broadcasting3d_mul_add_bfp16: x1 bfp16 (H x W x C); x2 / x3 bf16
 * or bfp16 (x2_bfp16 / x3_bfp16 = 1), H / W broadcast only (bit 0 is not
 * supported, as in the kernel); out bf16 or bfp16 (out_bfp16 = 1).
 *   v = fma(x1, x2, x3)  (the bf16 x bf16 product is exact)
 * act_type is the compiled template (mllib::ActivationType 0 / 7 TANH / 100
 * CLIP), activation the runtime RTP word:
 *   act_type 100, activation 100: v = clamp(rne_bf16(v), clip_min, clip_max)
 *     (bf16 bounds, -0 below +0)
 *   act_type 7: v = rne_bf16(v) (the result goes through a bf16 chain buffer,
 *     also when activation is 0); activation 7: v = sigtanh(v), the SOL SigTanh
 *     kernel: p = fma(x^2, k2, fma(-|x|, k1, k0)) with k = (894.00097656,
 *     1.53878212, -0.49819374), e = exp2_bf20_hw(p) (subnormal -> 0),
 *     t = fma(-2, e, 1) with x's sign bit xor-ed in, rounded to bf16 half away
 *     from zero (subnormal -> 0).
 *   out = rne(v) to bf16 or bfp16. */
int npu_elementwise_broadcasting3d_mul_add_bfp16(
    void* stream, const void* x1, const void* x2, int x2_bfp16, int x2_bcast,
    const void* x3, int x3_bfp16, int x3_bcast, void* out, int out_bfp16, int64_t H,
    int64_t W, int64_t C, int act_type, int activation, float clip_min, float clip_max);

/* ---- fp8 3D broadcasting (E4M3FN) ------------------------------------------ */

/* elementwise_broadcasting3d_add_fp8 (Add and Sub; Sub passes negated
 * coefficients). c1 / c2 = coeff words 0 / 1 (ifm1 / ifm2 scale over ofm scale,
 * signed), act0 / act1 = words 2 / 3. x1 is row-decoded. Output half away from
 * zero. act: NONE, RELU, CLIP (the kernel's CLIP template, YAML relu6), LRELU.
 * Without C broadcast (x2 row-decoded):
 *   v = fma(x2, c2, fp32(x1 * c1)); out = act(v)
 * With C broadcast, x2 is converted directly (-0 and NaN kept per value):
 *   NONE : fma(x1, c1, fp32(x2 * c2))
 *   RELU : max(fp32(fp32(x1 * c1) + fp32(x2 * c2)), act0)
 *   CLIP : clamp(fp32(fp32(x1 * c1) + table[code(x2)]), act0, act1)
 *   LRELU: q = fma(x1, table[256], table[code(x2)]); fma(|q|, table[257], q)
 * table = coeff words 4..261 (device array of 258 floats: the YAML's per-code
 * biases, then A * c1 and B / A for LRELU); needed only for C broadcast with
 * CLIP or LRELU, else may be null. */
int npu_elementwise_broadcasting3d_add_fp8(void* stream, const void* x1, const void* x2,
                                           int x2_bcast, void* out, int64_t H, int64_t W,
                                           int64_t C, float c1, float c2, npu_float_act act,
                                           float act0, float act1, const float* table);
/* elementwise_broadcasting3d_mul_fp8: s = coeff word 0. Output half away.
 * Without C broadcast: fp32(fp32(x1 * x2) * s), the fp8 values multiplied
 * directly (no row decode: -0 and NaN per element). With C broadcast (both
 * row-decoded): fp32(x1 * fp32(x2 * s)). */
int npu_elementwise_broadcasting3d_mul_fp8(void* stream, const void* x1, const void* x2,
                                           int x2_bcast, void* out, int64_t H, int64_t W,
                                           int64_t C, float scale);
/* elementwise_broadcasting3d_div_fp8: both row-decoded; (num, den) = swap ?
 * (x2, x1) : (x1, x2); out = fp32(fp32(num * s) * inv(den)), half away, where
 * inv is the AIE4 faithfully rounded fp32 reciprocal (src/npu_emu_inv_f32_table.h;
 * 0 / Inf / NaN as IEEE). s = coeff word 0. */
int npu_elementwise_broadcasting3d_div_fp8(void* stream, const void* x1, const void* x2,
                                           int x2_bcast, void* out, int64_t H, int64_t W,
                                           int64_t C, float scale, int swap);
/* elementwise_broadcasting3d_mac_fused_fp8: c0 / c4 = coeff words 0 / 4.
 * Output half to even (the wrapper sets conv_even). act: RELU (act0 = coeff
 * word 5), CLIP (words 5, 6), LRELU (act0 = alpha, coeff byte 384).
 * Without C broadcast (all row-decoded): v = fma(x3, c4, fp32(x2 * fp32(x1 * c0)))
 * With C broadcast (x2 / x3 per-pixel scalars, -0 -> +0, NaN per value):
 *   v = fma(x1, fp32(x2 * c0), fp32(x3 * c4)) */
int npu_elementwise_broadcasting3d_mac_fused_fp8(void* stream, const void* x1,
                                                 const void* x2, const void* x3,
                                                 int x23_bcast, void* out, int64_t H,
                                                 int64_t W, int64_t C, float c0, float c4,
                                                 npu_float_act act, float act0, float act1);

/* ---- unary / per-channel (n elements, flat) -------------------------------- */

/* elementwise_leakyrelu_bf16 (BF16 / F16): a = rne(alpha) in `type`;
 * out = rne(max(x, x * a)) (x * a exact in fp32). */
int npu_elementwise_leakyrelu_bf16(void* stream, const void* in, npu_dtype type, void* out,
                                   int64_t n, float alpha);
/* elementwise_leakyrelu_fp8: x row-decoded (rows = 8 consecutive elements);
 * out = fp8(max(fp32(x * pos), fp32(x * neg))) half away. pos / neg = wts
 * words 0 / 1. */
int npu_elementwise_leakyrelu_fp8(void* stream, const void* in, void* out, int64_t n,
                                  float pos, float neg);
/* elementwise_prelu_bf16 (BF16 / F16): per channel c = i % C (innermost),
 * D = slope delta (wts words, fp32(slope - 1)), device array of C floats:
 * out = rne(fma(min(x, 0), D[c], x)). */
int npu_elementwise_prelu_bf16(void* stream, const void* in, npu_dtype type, void* out,
                               int64_t n, const float* D, int64_t C);
/* elementwise_prelu_fp8: x row-decoded; P = wts word 0, D = per-channel deltas
 * (wts words from byte 128, fp32(fp32(slope - 1) * P)), device array of C floats:
 * out = fp8(fma(min(x, 0), D[c], fp32(x * P))) half away. */
int npu_elementwise_prelu_fp8(void* stream, const void* in, void* out, int64_t n, float P,
                              const float* D, int64_t C);
/* elementwise_square_bf16: v = fp32(x * x) (exact); F16: |v| < 2^-14 -> +0
 * (the vector unit flushes subnormal f16 results); out = rne(v). The wrapper
 * does not set the rounding mode; half to even matches aiesim. */
int npu_elementwise_square_bf16(void* stream, const void* in, npu_dtype type, void* out,
                                int64_t n);
/* elementwise_square_fp8: x row-decoded; out = fp8(fp32(x * fp32(x * s))) half
 * away, s = coeff word 0. */
int npu_elementwise_square_fp8(void* stream, const void* in, void* out, int64_t n,
                               float scale);
/* batchnorm2d_stream_bf16: type BF16 / F16 / FP8; in / out n elements with
 * channel c = i % C. cw / cb: device arrays of C floats holding the wts values
 * (bf16 for BF16, f16 for F16 and FP8). acc = fma(x, cw[c], cb[c]);
 *   act 0: out = rne(acc)
 *   act 1 (RELU): 16-bit: max(rne(acc), +0) (-0 -> +0); FP8: rne(max(acc, 0))
 *   act 3 (LRELU): rne(max(acc, fp32(acc * alpha)))
 * FP8 inputs are row-decoded over 8 consecutive elements (the NPU stream rows
 * are 8 channels of a pixel for c_stride 32) and the output rounds half to
 * even. */
int npu_batchnorm2d_stream_bf16(void* stream, const void* in, npu_dtype type, void* out,
                                int64_t n, int64_t C, const float* cw, const float* cb,
                                int act, float alpha);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_ARITH_FLOAT_H_
