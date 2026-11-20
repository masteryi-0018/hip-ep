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
 * (Ho = (H - Kh) / stride_h + 1, no padding). bias: N / Co fp32 values or null.
 */
#ifndef NPU_EMU_BF16_MATMUL_H_
#define NPU_EMU_BF16_MATMUL_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* acc = bias (or 0), one MMAC per 32 consecutive K (zero-padded), bf16 (ties
 * away), then activation 0 none, 1 ReLU, 2 ReLU6. Weights and bias on streams. */
int npu_gemm_bf16x16(void* stream, const uint16_t* a, const uint16_t* b, const float* bias,
                     uint16_t* out, int64_t m, int64_t k, int64_t n, int activation);
/* npu_gemm_bf16x16 for the kernel that reads weights and bias from a buffer. */
int npu_gemm_bf16x16_buffer(void* stream, const uint16_t* a, const uint16_t* b,
                            const float* bias, uint16_t* out, int64_t m, int64_t k, int64_t n,
                            int activation);
/* npu_gemm_bf16x16 for the kernel that keeps one input buffer across output tiles. */
int npu_gemm_bf16x16_ifm_reuse(void* stream, const uint16_t* a, const uint16_t* b,
                               const float* bias, uint16_t* out, int64_t m, int64_t k, int64_t n,
                               int activation);
/* npu_gemm_bf16x16 with the output written transposed, N x M. */
int npu_gemm_bf16x16_transposed(void* stream, const uint16_t* a, const uint16_t* b,
                                const float* bias, uint16_t* out, int64_t m, int64_t k,
                                int64_t n, int activation);

/* The npu_gemm_bf16x16 output v quantized to 16 bits and dequantized:
 * q = sat(rne(fp32(fp32(v * inv_scale) + zp))), int16 (q_signed) or uint16;
 * out = rne_bf16(fp32(q * scale + fp32(-zp * scale))). */
int npu_gemm_quant_dequant_bf16x16(void* stream, const uint16_t* a, const uint16_t* b,
                                   const float* bias, uint16_t* out, int64_t m, int64_t k,
                                   int64_t n, int activation, int q_signed, float inv_scale,
                                   float scale, int zero_point);
/* npu_gemm_quant_dequant_bf16x16 for the kernel that keeps one input buffer across output
 * tiles. */
int npu_gemm_quant_dequant_bf16x16_ifm_reuse(void* stream, const uint16_t* a, const uint16_t* b,
                                             const float* bias, uint16_t* out, int64_t m,
                                             int64_t k, int64_t n, int activation, int q_signed,
                                             float inv_scale, float scale, int zero_point);

/* acc = bias (or 0), MMACs ordered: groups of kh_sv kernel rows outermost,
 * then blocks of 32 input channels, the rows and the columns; with ci < 32 the
 * kw * ci values of a row are packed into the 32 lanes in order. activation
 * 0 / 1 / 2 as above. Weights on a stream. */
int npu_conv_bf16x16(void* stream, const uint16_t* ifm, const uint16_t* wts, const float* bias,
                     uint16_t* out, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                     int64_t co, int stride_h, int stride_w, int64_t kh_sv, int activation);
/* npu_conv_bf16x16 for the kernel that reads weights from a buffer; it also has activation
 * 3 LeakyReLU: max(v, bf16(v * alpha_bf16)), ties away. */
int npu_conv_bf16x16_buffer(void* stream, const uint16_t* ifm, const uint16_t* wts,
                            const float* bias, uint16_t* out, int64_t h, int64_t w, int64_t ci,
                            int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                            int64_t kh_sv, int activation, float alpha);

/* Residual ifm2: bf16, h2 x w2 x c2 broadcast to Ho x Wo x Co (a dimension of
 * 1 broadcasts). type 0: acc = fp32(residual + bias), MMACs, bf16 ties away
 * (activation must be 0). type 1: npu_conv_bf16x16 with activation 0 / 1, then
 * rne_bf16(fp32(v + residual)). */
int npu_conv_add_bf16x16(void* stream, const uint16_t* ifm, const uint16_t* wts, const float* bias,
                         const uint16_t* ifm2, uint16_t* out, int64_t h, int64_t w, int64_t ci,
                         int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                         int64_t kh_sv, int64_t h2, int64_t w2, int64_t c2, int type,
                         int activation);

/* Per channel acc = bias (or 0), then over the window (row-major) one fp32
 * product and one fp32 add per tap; bf16 ties away; activation 1 / 2 on bf16;
 * 3 LeakyReLU bf16(max(v, fp32(alpha_bf16 * v))). */
int npu_depthwise_conv_bf16x16(void* stream, const uint16_t* ifm, const uint16_t* wts,
                               const float* bias, uint16_t* out, int64_t h, int64_t w, int64_t c,
                               int64_t kh, int64_t kw, int stride_h, int stride_w,
                               int activation, float alpha);
/* npu_depthwise_conv_bf16x16 for the kernel whose input and output are blocked by 32
 * channels. */
int npu_depthwise_conv_bf16x16_c32(void* stream, const uint16_t* ifm, const uint16_t* wts,
                                   const float* bias, uint16_t* out, int64_t h, int64_t w,
                                   int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w,
                                   int activation, float alpha);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_BF16_MATMUL_H_
