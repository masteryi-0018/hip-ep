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
 * Layouts: gemm A is M x K and B is given transposed, N x K (blocks along K);
 * conv ifm is H x W x Ci (blocks along Ci), weights Kh x Kw x Co x Ci for the
 * block formats and Kh x Kw x Ci x Co for fp8; output Ho x Wo x Co (no padding).
 * ofm_type: NPU_BF16, NPU_F16, NPU_FP8, or the packed block format of the inputs
 * (NPU_BFP16 / NPU_MX6, blocks along the last output axis). bias: Co / N fp32 or null.
 *
 * Convolutions run their MMACs over groups of kh_sv kernel rows and ci_sv
 * channels with 64 / co_sv output columns folded into one MMAC. kh_sv must
 * divide kh, ci_sv divide ci and be a multiple of 8 (bfp16, fp8) or 128 (mx6),
 * co_sv <= 64 divide co.
 */
#ifndef NPU_EMU_BLOCK_MATMUL_H_
#define NPU_EMU_BLOCK_MATMUL_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* acc = bias (or 0), one MMAC per 64 K. narrow: the accumulator is rounded to
 * bf16 before act: 1 ReLU, 2 ReLU6, 3 LeakyReLU max(v, bf16(v * bf16(alpha))),
 * 5 SiLU, 6 GELU, 7 Tanh (results stored as bf16). transpose: the output is
 * N x M. ofm_type NPU_BF16, NPU_BFP16, or NPU_U8: the uint8 cast of the bf16 value (narrow,
 * act 0), sat(rne(fp32(v * ofm_scale_recip + ofm_zp))); the two scalars are used for NPU_U8
 * only. */
int npu_gemm_bfp16(void* stream, const uint8_t* a, const uint8_t* bt, const float* bias,
                   void* out, int64_t m, int64_t k, int64_t n, int narrow, int act, float alpha,
                   int transpose, npu_dtype ofm_type, float ofm_scale_recip, float ofm_zp);

/* out = act(bf16(a bt_gate^T + bias_gate)) * bf16(a bt_up^T + bias_up), stored
 * as ofm_type (NPU_BF16 or NPU_BFP16); act 0, 5, 6, 7 as in npu_gemm_bfp16. */
int npu_gemm_gelu_mul_bfp16(void* stream, const uint8_t* a, const uint8_t* bt_gate,
                            const uint8_t* bt_up, const float* bias_gate, const float* bias_up,
                            void* out, int64_t m, int64_t k, int64_t n, int act,
                            npu_dtype ofm_type);

/* GEMM plus / times ifm2: m2 x N (m2 = 1 broadcasts over M), bf16 (ifm2_type
 * NPU_BF16) or bfp16 (NPU_BFP16, add only).
 * add, bf16 ifm2: acc = fp32(ifm2 + bias), then the MMACs. add, bfp16 ifm2:
 * acc = bias, and ifm2 enters as one extra MMAC: before the first 64 K when
 * (m < sv_m and n < sv_n) or sv_k == 64, after it otherwise. mul: ifm2 is
 * bf16; out = bf16(acc) * ifm2. ofm_type NPU_BF16 or NPU_BFP16. */
int npu_gemm_add_bfp16(void* stream, const uint8_t* a, const uint8_t* bt, const float* bias,
                       const void* ifm2, npu_dtype ifm2_type, int64_t m2, void* out, int64_t m,
                       int64_t k, int64_t n, int64_t sv_m, int64_t sv_n, int64_t sv_k,
                       npu_dtype ofm_type);
int npu_gemm_mul_bfp16(void* stream, const uint8_t* a, const uint8_t* bt, const float* bias,
                       const void* ifm2, npu_dtype ifm2_type, int64_t m2, void* out, int64_t m,
                       int64_t k, int64_t n, int64_t sv_m, int64_t sv_n, int64_t sv_k,
                       npu_dtype ofm_type);

/* acc = bias (or 0); narrow / act / alpha as npu_gemm_bfp16 (act 0 - 3); ofm_type NPU_BF16 or
 * NPU_BFP16. */
int npu_conv_bfp16(void* stream, const uint8_t* ifm, const uint8_t* wts, const float* bias,
                   void* out, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                   int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv, int64_t co_sv,
                   int narrow, int act, float alpha, npu_dtype ofm_type);

/* npu_conv_bfp16 plus ifm2 (h2 x w2 x c2 bf16, a dimension of 1 broadcasts):
 * acc = fp32(ifm2 + bias), no activation, and the output rounds half to even. */
int npu_conv_add_bfp16(void* stream, const uint8_t* ifm, const uint8_t* wts, const float* bias,
                       const uint16_t* ifm2, int64_t h2, int64_t w2, int64_t c2, void* out,
                       int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                       int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv, int64_t co_sv,
                       npu_dtype ofm_type);

/* As npu_gemm_bfp16 / npu_conv_bfp16 with mx6 operands (one MMAC per
 * 128 K); ofm_type NPU_BF16 or NPU_MX6. */
int npu_gemm_mx6(void* stream, const uint8_t* a, const uint8_t* bt, const float* bias, void* out,
                 int64_t m, int64_t k, int64_t n, int narrow, int act, float alpha,
                 npu_dtype ofm_type);
int npu_conv_mx6(void* stream, const uint8_t* ifm, const uint8_t* wts, const float* bias,
                 void* out, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                 int stride_h, int stride_w, int64_t kh_sv, int64_t ci_sv, int64_t co_sv,
                 int narrow, int act, float alpha, npu_dtype ofm_type);

/* a M x K, b K x N; acc from 0, one MMAC per 64 K (zero padded);
 * out = fp32(acc * c2[n] + c0[n]) as ofm_type (NPU_FP8, NPU_F16, NPU_BF16). */
int npu_gemm_fp8(void* stream, const uint8_t* a, const uint8_t* b, const float* c0,
                 const float* c2, void* out, int64_t m, int64_t k, int64_t n, npu_dtype ofm_type);

/* res = fp32(acc * c2[co] + c0[co]). act 1 ReLU, 2 clamp to [0, a0],
 * 3 LeakyReLU fp32(min(v, 0) * a0 + max(v, 0)), 8 clamp to [a0, a1], 9 PReLU
 * (v < 0 ? fp32(v * slope[co]) : v). ifm_type NPU_FP8, or NPU_F16: each activation is
 * narrowed to E4M3 first, fp8_away(clamp(f16(x * r), +-448)) with r = f16(ifm_scale_recip)
 * and the f16 product of the 16-bit multiply. */
int npu_conv_fp8(void* stream, const void* ifm, npu_dtype ifm_type, float ifm_scale_recip,
                 const uint8_t* wts, const float* c0, const float* c2, void* out, int64_t h,
                 int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
                 int stride_w, int64_t kh_sv, int64_t ci_sv, int64_t co_sv, int act, float a0,
                 float a1, const float* slope, npu_dtype ofm_type);

/* npu_conv_fp8 plus ifm2 (h2 x w2 x c2_dim of ifm2_type NPU_FP8 / NPU_BF16 / NPU_F16,
 * broadcast): res = fp32(acc * c2 + fp32(ifm2 * ifm2_coeff + c0)), no activation. */
int npu_conv_add_fp8(void* stream, const uint8_t* ifm, const uint8_t* wts, const float* c0,
                     const float* c2, const void* ifm2, npu_dtype ifm2_type, int64_t h2,
                     int64_t w2, int64_t c2_dim, float ifm2_coeff, void* out, int64_t h, int64_t w,
                     int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                     int64_t kh_sv, int64_t ci_sv, int64_t co_sv, npu_dtype ofm_type);

/* ifm H x W x C, wts Kh x Kw x C; per channel one fp32 product and one fp32
 * add per tap (row-major), res = fp32(acc * c2[c] + c0[c]); act 1 ReLU,
 * 2 clamp to [0, a0], 3 LeakyReLU max(v, fp32(v * a0)). */
int npu_depthwise_conv_fp8(void* stream, const uint8_t* ifm, const uint8_t* wts, const float* c0,
                           const float* c2, void* out, int64_t h, int64_t w, int64_t c, int64_t kh,
                           int64_t kw, int stride_h, int stride_w, int act, float a0,
                           npu_dtype ofm_type);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_BLOCK_MATMUL_H_
