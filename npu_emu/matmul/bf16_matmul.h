/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * bf16 matrix multiply and convolution on the 32-lane bf16 MMAC (fp32
 * accumulator); bf16 tensors are uint16 bit patterns.
 *
 * One MMAC adds 32 products to the accumulator: lanes 0-15 and 16-31 are each
 * aligned to their largest exponent (guard + sticky, half to even) and summed;
 * the accumulator and the two half sums are aligned, summed and rounded half
 * to even to fp32. Results below the normal range give +0.
 *
 * Tensors are logical and contiguous: A M x K, B K x N, output M x N; the
 * convolution input H x W x Ci, weights Kh x Kw x Ci x Co, output Ho x Wo x Co
 * (Ho = (H - Kh) / stride_h + 1, no padding). bias: N / Co values or null; the gemm bias is
 * bf16 or fp32 (bias_type), the convolution bias bf16. K, Ci, Co, Kh, Kw >= 1; M, N, H and W
 * may be 0 (an empty output: nothing is launched; Ho = 0 when H is 0, Wo = 0 when W is 0),
 * otherwise H >= Kh and W >= Kw. Every entry returns -1 for a null pointer other than bias and
 * for arguments outside the ranges stated.
 */
#ifndef NPU_EMU_BF16_MATMUL_H_
#define NPU_EMU_BF16_MATMUL_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* acc = bias (or 0), one MMAC per 32 consecutive K (zero-padded), bf16 (ties
 * away), then activation 0 none, 1 ReLU, 2 ReLU6. */
int npu_gemm_bf16x16(void* stream, const void* a, const void* b, const void* bias,
                     npu_dtype bias_type, void* out, int64_t m, int64_t k, int64_t n,
                     int activation);
/* npu_gemm_bf16x16 for the kernel that reads weights and bias from a buffer. */
int npu_gemm_bf16x16_buffer(void* stream, const void* a, const void* b, const void* bias,
                            npu_dtype bias_type, void* out, int64_t m, int64_t k, int64_t n,
                            int activation);
/* npu_gemm_bf16x16 for the kernel that keeps one input buffer across output tiles. */
int npu_gemm_bf16x16_ifm_reuse(void* stream, const void* a, const void* b, const void* bias,
                               npu_dtype bias_type, void* out, int64_t m, int64_t k, int64_t n,
                               int activation);
/* npu_gemm_bf16x16 with the output written transposed, N x M. */
int npu_gemm_bf16x16_transposed(void* stream, const void* a, const void* b, const void* bias,
                                npu_dtype bias_type, void* out, int64_t m, int64_t k, int64_t n,
                                int activation);

/* The npu_gemm_bf16x16 output v quantized to 16 bits and dequantized:
 * q = sat(rne(fp32(v * quant_mult + zero_point))) (one rounding), int16 (qdq_sign 1) or
 * uint16 (qdq_sign 0); out = rne_bf16(fp32(q * dequant_mult + fp32(-zero_point *
 * dequant_mult))). zero_point lies in the range of the 16-bit type. */
int npu_gemm_quant_dequant_bf16x16(void* stream, const void* a, const void* b, const void* bias,
                                   npu_dtype bias_type, void* out, int64_t m, int64_t k,
                                   int64_t n, float quant_mult, float dequant_mult,
                                   float zero_point, int qdq_sign, int activation);
/* npu_gemm_quant_dequant_bf16x16 for the kernel that keeps one input buffer across output
 * tiles. */
int npu_gemm_quant_dequant_bf16x16_ifm_reuse(void* stream, const void* a, const void* b,
                                             const void* bias, npu_dtype bias_type, void* out,
                                             int64_t m, int64_t k, int64_t n, float quant_mult,
                                             float dequant_mult, float zero_point, int qdq_sign,
                                             int activation);

/* acc = bias (or 0), MMACs ordered: groups of kh_sv kernel rows outermost,
 * then blocks of 32 input channels, the rows and the columns; with ci < 32 the
 * kw * ci values of a row are packed into the 32 lanes in order. Strides 1 or 2;
 * activation 0 / 1 / 2 as above. */
int npu_conv_bf16x16(void* stream, const void* ifm, const void* wts, const void* bias, void* out,
                     int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                     int stride_h, int stride_w, int64_t kh_sv, int activation);
/* npu_conv_bf16x16 for the kernel that reads weights from a buffer: strides 1, 2 or 4, and
 * activation 3 LeakyReLU: max(v, bf16(v * bf16(lrelu_alpha))), ties away. */
int npu_conv_bf16x16_buffer(void* stream, const void* ifm, const void* wts, const void* bias,
                            void* out, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                            int64_t co, int stride_h, int stride_w, int64_t kh_sv,
                            float lrelu_alpha, int activation);

/* Residual ifm2: bf16, h2 x w2 x Co broadcast to Ho x Wo x Co (h2 / w2 1 or full; the
 * post-add type needs the full shape). Strides 1 or 2. activation_type 0: acc =
 * fp32(residual + bias), MMACs, bf16 ties away (activation must be 0). activation_type 1:
 * npu_conv_bf16x16 with activation 0 / 1, then rne_bf16(fp32(v + residual)). */
int npu_conv_add_bf16x16(void* stream, const void* ifm, const void* ifm2, const void* wts,
                         const void* bias, void* out, int64_t h, int64_t w, int64_t ci,
                         int64_t kh, int64_t kw, int64_t co, int64_t h2, int64_t w2,
                         int stride_h, int stride_w, int64_t kh_sv, int activation_type,
                         int activation);

/* Per channel acc = bias (or 0), then over the window (row-major) one fp32
 * product and one fp32 add per tap; bf16 ties away; activation 1 / 2 on bf16;
 * 3 LeakyReLU bf16(max(v, fp32(bf16(alpha) * v))). Kh * Kw >= 2, strides 1 or 2. */
int npu_depthwise_conv_bf16x16(void* stream, const void* ifm, const void* wts, const void* bias,
                               void* out, int64_t h, int64_t w, int64_t c, int64_t kh,
                               int64_t kw, int stride_h, int stride_w, float alpha,
                               int activation);
/* npu_depthwise_conv_bf16x16 for the kernel that works on 32-channel blocks; the same
 * logical tensors and arithmetic. */
int npu_depthwise_conv_bf16x16_c32(void* stream, const void* ifm, const void* wts,
                                   const void* bias, void* out, int64_t h, int64_t w, int64_t c,
                                   int64_t kh, int64_t kw, int stride_h, int stride_w,
                                   float alpha, int activation);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_BF16_MATMUL_H_
