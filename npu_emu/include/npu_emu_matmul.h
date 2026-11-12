/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 QDQ matrix-multiply / convolution
 * wrappers with integer operands: gemm_qdq_int16x16, conv_qdq_int8x8 and
 * conv_qdq_int16x8. Conventions as in npu_emu.h.
 *
 * The products are accumulated in int32 MMAC accumulators (exact). The QDQ
 * output stage is a chain of accfloat MACs, each rounded once to fp32 with an
 * integer operand taken exactly, in the order given per function; then the
 * activation and a float -> integer conversion that rounds half to even and
 * saturates (to_vector_sign under conv_even).
 *
 * The coefficient arguments are the fp32 words the kernel reads, not ONNX
 * scales and zero points. Per channel (s_in, zp_in, s_w, zp_w, s_b, zp_b,
 * s_out, zp_out the ONNX parameters, K the reduction length):
 *   c2 = s_in * s_w / s_out
 *   c1 = -c2 * zp_w                         (activation-sum coefficient)
 *   c0 = zp_out + s_b / s_out * (bias - zp_b) - c2 * zp_in * (sum(w) - zp_w * K)
 * The mllib host code (flexml_metadata wts_pack_conv_a8w8_aie4, YAML coeff_*)
 * evaluates these in fp32 in its own order; pass the words it packs.
 *
 * Activation (npu_activation): NONE; CLIP clamps to [act0, act1] (relu, relu6
 * and clamp, bounds in the output's quantized domain); LRELU computes
 * max(v, v * act0 + act1) with one rounding; PRELU (convolutions only) uses a
 * per-channel slope and pivots on act0 = zp_out:
 *   v < zp_out ? v * slope + (zp_out - slope * zp_out) : v
 * where the offset is itself one MAC (slope * -zp_out + zp_out).
 *
 * Tensors are contiguous and logical: A is M x K, B is K x N, the output M x N
 * (row-major); the convolution input is H x W x Ci, the weights Kh x Kw x Ci x
 * Co, the output Ho x Wo x Co with Ho = (H - Kh) / stride_h + 1 (no padding).
 * The NPU's sub-volume tiling, C64 blocking and weight packing are not
 * emulated. Unverified on the NPU: shift_res != 0 (not taken), accumulators
 * that would overflow int32.
 */
#ifndef NPU_EMU_MATMUL_H_
#define NPU_EMU_MATMUL_H_

#include <stdint.h>

#include "npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NPU_ACT_NONE = 0,
  NPU_ACT_CLIP = 1,
  NPU_ACT_LRELU = 2,
  NPU_ACT_PRELU = 3,
} npu_activation;

/* gemm_qdq_int16x16 (also the arithmetic of gemm_qdq_a16a16_transpose):
 * A and B are I16 / U16, the output I16 / U16. Each 16-bit operand is split
 * into a signed high byte and an unsigned low byte, giving three int32 sums
 * lo = A_lo B_lo, mid = A_hi B_lo + A_lo B_hi, hi = A_hi B_hi, and per column
 * the byte sums wl = sum_k B_lo, wh = sum_k B_hi; asum = fp32(sum_k A) per row.
 *   t = mac(wl, c3, c0);  t = mac(wh, 256 c3, t)            (per column)
 *   v = mac(lo, c2, t);  v = mac(mid, 256 c2, v);  v = mac(hi, 65536 c2, v)
 *   v = fma(asum, c1, v)
 * c3 = -c2 * zp_in is the weight-sum coefficient; c0 here excludes the
 * weight-sum term. activation: NONE, CLIP or LRELU. */
int npu_gemm_qdq_int16x16(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                          int64_t k, int64_t n, float c0, float c1, float c2, float c3,
                          npu_activation activation, float act0, float act1);

/* conv_qdq_int8x8 (regression dirs conv_qdq_int8x8_wrapper and
 * conv_qdq_int8x8_channelwise_wrapper): ifm I8 / U8, wts I8 / U8, output
 * I8 / U8. c0 / c1 / c2 are device arrays of Co floats; c1 null when the
 * kernel has no activation sum (has_actv_sum = 0). conv is the exact int32
 * sum; sum_ifm the fp32 channel sum of each input pixel, added over the
 * Kh x Kw window one fp32 add at a time:
 *   v = mac(conv, c2, c0);  v = fma(sum_ifm, c1, v)
 * The order of the window adds matters once the sums pass 2^24 (16-bit
 * inputs). k_opt = 0: taps row-major (the generic path). k_opt = 1, the kernel
 * instance specialized for its window (mllib is_kh and is_kw nonzero: 3x3 and
 * 7x7 when the tiling allows) with an activation sum: column 0 summed down the
 * window, the odd and the even remaining columns summed kh-major into two
 * accumulators, total (col0 + odd) + even; for an even Kw, even + odd.
 * slope: device array of Co floats for PRELU, else null. */
int npu_conv_qdq_int8x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                        npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h,
                        int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                        int stride_h, int stride_w, int k_opt, const float* c0,
                        const float* c1, const float* c2, npu_activation activation,
                        float act0, float act1, const float* slope);

/* conv_qdq_int16x8 (conv_qdq_int16x8_wrapper and _channelwise_wrapper): as
 * conv_qdq_int8x8 with ifm I16 / U16 and output I16 / U16 (or 8-bit). The
 * input is split into its high (signed for I16) and low bytes; the two int32
 * convolutions are joined as fp32(fp32(lo) + 256 hi) before the c2 MAC:
 *   g = fma(hi, 256, fp32(lo));  v = fma(g, c2, c0);  v = fma(sum_ifm, c1, v)
 * k_opt as for conv_qdq_int8x8. */
int npu_conv_qdq_int16x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                         npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h,
                         int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                         int stride_h, int stride_w, int k_opt, const float* c0,
                         const float* c1, const float* c2, npu_activation activation,
                         float act0, float act1, const float* slope);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_MATMUL_H_
