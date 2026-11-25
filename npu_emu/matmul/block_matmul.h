/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Block-floating (bfp16, mx6) and fp8 (E4M3FN) matrix multiply and convolution.
 *
 * MMACs into an fp32 accumulator:
 *   - bfp16: 64 products in 8 blocks of 8 along K; block sums are exact. The
 *     accumulator and the 8 block sums are aligned to the largest exponent
 *     (guard + sticky, each rounded half to even), summed and rounded to fp32.
 *   - mx6: 128 products in 8 blocks of 16; per pair of K the two products are
 *     summed and shifted left by 2 - (pair shift of x + of z); then as bfp16.
 *   - fp8: 64 products in two halves of 32, each aligned to its largest product
 *     exponent with every product rounded half to even; then accumulator and
 *     half sums as in the bf16 MMAC.
 * Results below the fp32 normal range give +0. Output conversions round half
 * away from zero unless noted; fp8 does not saturate.
 *
 * Layouts: gemm A is M x K, B is K x N; conv ifm is H x W x Ci, weights Kh x Kw x Ci x Co;
 * output Ho x Wo x Co (no padding). Blocks run along the reduction axis (K / Ci): an A or ifm
 * block is 8 (bfp16) / 16 (mx6) consecutive values of a row / pixel, and B is a K/8 x N grid
 * of blocks (mx6: K/16 x N), block (kb, n) holding rows 8 kb .. 8 kb + 7 of column n; conv
 * weights likewise a Kh x Kw x Ci/8 x Co grid (mx6: Ci/16). fp8 operands are plain arrays.
 * ofm_type: NPU_BF16, NPU_F16, NPU_FP8, or the packed block format of the inputs
 * (NPU_BFP16 / NPU_MX6, blocks along the last output axis). bias / c0 / c2 / slope: N / Co
 * fp32 values on the device; bias may be null.
 *
 * Convolutions run their MMACs over groups of kh_sv kernel rows and ci_sv channels with
 * 64 / co_sv output columns folded into one MMAC; kh_sv divides kh, ci_sv divides ci and
 * co_sv divides co. bfp16: ci_sv a multiple of 64, co_sv = 64, kh_sv and kw 1..15; mx6:
 * ci_sv 128 or 256, co_sv = 64, kh_sv and kw 1..15; fp8: ci_sv 8, 16, ..., 256, co_sv 8,
 * 16, 32 or 64, kh_sv and kw 1..7. Strides 1 or 2. Ci, Co >= 1. M, N, H and W may be 0 (an
 * empty output: nothing is launched; Ho = 0 when H is 0, Wo = 0 when W is 0), otherwise
 * H >= Kh and W >= Kw. Every entry returns -1 for a null pointer other than bias / slope (see
 * below) and for arguments outside the ranges stated.
 */
#ifndef NPU_EMU_BLOCK_MATMUL_H_
#define NPU_EMU_BLOCK_MATMUL_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* acc = bias (or 0), one MMAC per 64 K. narrow: the accumulator is rounded to bf16 before
 * activation 1 ReLU, 2 ReLU6, 3 LeakyReLU max(v, bf16(v * bf16(lrelu_alpha))), 4 Sigmoid,
 * 5 SiLU, 6 GELU, 7 Tanh (results stored as bf16); activation needs narrow. transpose: the
 * output is N x M (no bias, no activation 1 - 3). ofm_type NPU_BFP16, NPU_BF16 (no
 * activation, no transpose) or NPU_U8: the uint8 cast of the bf16 value (narrow,
 * activation 0), sat(rne(fp32(v * quant_multiplier + zero_point))); the two scalars are
 * read for NPU_U8 only. */
int npu_gemm_bfp16(void* stream, const void* a, const void* b, const void* bias, void* out,
                   npu_dtype ofm_type, int64_t m, int64_t k, int64_t n, int transpose,
                   float quant_multiplier, float zero_point, int narrow, int activation,
                   float lrelu_alpha);

/* out = act(bf16(a b_gate + bias_gate)) * bf16(a b_up + bias_up), stored as bfp16;
 * activation 0, 5, 6, 7 as in npu_gemm_bfp16. */
int npu_gemm_gelu_mul_bfp16(void* stream, const void* a, const void* b_gate, const void* b_up,
                            const void* bias_gate, const void* bias_up, void* out, int64_t m,
                            int64_t k, int64_t n, int activation);

/* GEMM plus ifm2: m2 x N (m2 = 1 broadcasts over M), bf16 (ifm2_type NPU_BF16) or bfp16
 * (NPU_BFP16, n a multiple of 64). bf16 ifm2: acc = fp32(ifm2 + bias), then the MMACs.
 * bfp16 ifm2: acc = bias, and ifm2 enters as one extra MMAC: before the first 64 K in the
 * first 64 x 64 output tile or when sv_k == 64, after it otherwise. sv_k: 64 .. 4096, a
 * multiple of 64. ofm_type NPU_BF16 or NPU_BFP16. */
int npu_gemm_add_bfp16(void* stream, const void* a, const void* ifm2, npu_dtype ifm2_type,
                       const void* b, const void* bias, void* out, npu_dtype ofm_type,
                       int64_t m, int64_t k, int64_t n, int64_t m2, int64_t sv_k);
/* GEMM times ifm2 (bf16, m2 x N, m2 = 1 broadcasts): out = bf16(acc) * ifm2; ofm_type
 * NPU_BF16 or NPU_BFP16. */
int npu_gemm_mul_bfp16(void* stream, const void* a, const void* ifm2, const void* b,
                       const void* bias, void* out, npu_dtype ofm_type, int64_t m, int64_t k,
                       int64_t n, int64_t m2);

/* acc = bias (or 0); narrow / activation (0 - 3) / lrelu_alpha as npu_gemm_bfp16; ofm_type
 * NPU_BFP16, or NPU_BF16 without activation. */
int npu_conv_bfp16(void* stream, const void* ifm, const void* wts, const void* bias, void* out,
                   npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                   int64_t co, int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv,
                   int narrow, int activation, float lrelu_alpha);

/* npu_conv_bfp16 plus ifm2 (h2 x w2 x c2 bf16, a dimension of 1 broadcasts):
 * acc = fp32(ifm2 + bias), no activation, and the output rounds half to even. */
int npu_conv_add_bfp16(void* stream, const void* ifm, const void* ifm2, const void* wts,
                       const void* bias, void* out, npu_dtype ofm_type, int64_t h, int64_t w,
                       int64_t ci, int64_t kh, int64_t kw, int64_t co, int64_t h2, int64_t w2,
                       int64_t c2, int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv);

/* As npu_gemm_bfp16 / npu_conv_bfp16 with mx6 operands (one MMAC per 128 K, activation
 * 0 - 3); ofm_type NPU_MX6, or NPU_BF16 without activation. */
int npu_gemm_mx6(void* stream, const void* a, const void* b, const void* bias, void* out,
                 npu_dtype ofm_type, int64_t m, int64_t k, int64_t n, int narrow, int activation,
                 float lrelu_alpha);
int npu_conv_mx6(void* stream, const void* ifm, const void* wts, const void* bias, void* out,
                 npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                 int64_t co, int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv, int narrow,
                 int activation, float lrelu_alpha);

/* a M x K of a_type NPU_FP8, or NPU_F16 narrowed on load: fp8_rne(clamp(fp32(x * r),
 * +-448)) with r = f16(ifm_quant_mult) (ifm_quant_mult read for NPU_F16 only); b K x N
 * fp8; acc from 0, one MMAC per 64 K (zero padded);
 * out = fp32(acc * c2[n] + c0[n]) as ofm_type (NPU_FP8, NPU_F16, NPU_BF16). */
int npu_gemm_fp8(void* stream, const void* a, npu_dtype a_type, const void* b, const void* c0,
                 const void* c2, void* out, npu_dtype ofm_type, int64_t m, int64_t k, int64_t n,
                 float ifm_quant_mult);

/* res = fp32(acc * c2[co] + c0[co]). activation 1 ReLU, 2 clamp to [0, relu6_max],
 * 3 LeakyReLU fp32(min(v, 0) * activation_coeff0 + max(v, 0)), 8 clamp to
 * [activation_coeff0, activation_coeff1], 9 PReLU (v < 0 ? fp32(v * slope[co]) : v; slope
 * may be null otherwise). ifm_type NPU_FP8, or NPU_F16: each activation is narrowed to E4M3
 * first, fp8_away(clamp(f16(x * r), +-448)) with r the f16 bits ifm_scale_recip_f16 and the
 * f16 product of the 16-bit multiply. */
int npu_conv_fp8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                 const void* c0, const void* c2, const void* slope, void* out,
                 npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                 int64_t co, int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv,
                 int64_t co_sv, uint16_t ifm_scale_recip_f16, int activation,
                 float activation_coeff0, float activation_coeff1, float relu6_max);

/* npu_conv_fp8 (fp8 ifm) plus ifm2 (h2 x w2 x c2_dim of ifm2_type NPU_FP8 / NPU_BF16 /
 * NPU_F16, a dimension of 1 broadcasts): res = fp32(acc * c2 + fp32(ifm2 * ifm2_coeff + c0)),
 * no activation. */
int npu_conv_add_fp8(void* stream, const void* ifm, const void* ifm2, npu_dtype ifm2_type,
                     const void* wts, const void* c0, const void* c2, void* out,
                     npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                     int64_t co, int64_t h2, int64_t w2, int64_t c2_dim, int stride_h,
                     int stride_w, int64_t kh_sv, int64_t ci_sv, int64_t co_sv,
                     float ifm2_coeff);

/* ifm H x W x C (C a multiple of 64), wts Kh x Kw x C (1..15 each); per channel one fp32
 * product and one fp32 add per tap (row-major), res = fp32(acc * c2[c] + c0[c]);
 * activation 1 ReLU, 2 clamp to [0, relu6_max], 3 LeakyReLU max(v, fp32(v * alpha)). */
int npu_depthwise_conv_fp8(void* stream, const void* ifm, const void* wts, const void* c0,
                           const void* c2, void* out, npu_dtype ofm_type, int64_t h, int64_t w,
                           int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w,
                           int activation, float alpha, float relu6_max);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_BLOCK_MATMUL_H_
