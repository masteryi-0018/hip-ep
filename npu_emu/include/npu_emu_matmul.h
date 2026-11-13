/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 QDQ matrix-multiply / convolution
 * wrappers with integer operands: gemm_qdq_int16x16, conv_qdq_int8x8,
 * conv_qdq_int16x8 and the QDQ integer GEMMs (gemm_a8w8, a8a8, a8w4, a16w8,
 * a16a8, a16w4, their stream / hybrid / transpose variants and the fused
 * adds). Conventions as in npu_emu.h.
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

/* gemm_qdq_a16a16_transpose: the arithmetic of npu_gemm_qdq_int16x16 (the
 * transposes only move data). */
int npu_gemm_qdq_a16a16_transpose(void* stream, const void* a, npu_dtype a_type, const void* b,
                                  npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                                  int64_t k, int64_t n, float c0, float c1, float c2, float c3,
                                  npu_activation activation, float act0, float act1);

/* ---- QDQ integer GEMMs ------------------------------------------------------
 * A is M x K, B is K x N, the output M x N. c0 / c1 / c2 / c3 are device
 * arrays of N floats (a tensorwise wrapper repeats its scalar); c1 null when
 * the kernel has no activation-sum term, c3 null when it has no weight-sum
 * term. gemm = sum_k A B (exact int32), rowsum = sum_k A, colsum = sum_k B.
 * activation: NONE, CLIP or LRELU (GELU is not emulated). */

/* gemm_a8w8, gemm_a8w8_stream, gemm_a8a8, gemm_qdq_a8a8_transpose, gemm_a8w4:
 * A I8 / U8; B I8 / U8 (gemm_a8w4: int4 values held in I8 / U8); output I8 / U8.
 *   c0' = c3 ? mac(colsum, c3, c0) : c0      (sum_to_c0, rounded to fp32)
 *   v = mac(gemm, c2, c0');  v = fma(fp32(rowsum), c1, v) */
#define NPU_GEMM_QDQ_ARGS                                                                  \
  void *stream, const void *a, npu_dtype a_type, const void *b, npu_dtype b_type, void *out, \
      npu_dtype out_type, int64_t m, int64_t k, int64_t n, const float *c0, const float *c1, \
      const float *c2, const float *c3, npu_activation activation, float act0, float act1
int npu_gemm_a8w8(NPU_GEMM_QDQ_ARGS);
int npu_gemm_a8w8_stream(NPU_GEMM_QDQ_ARGS);
int npu_gemm_a8a8(NPU_GEMM_QDQ_ARGS);
int npu_gemm_qdq_a8a8_transpose(NPU_GEMM_QDQ_ARGS);
int npu_gemm_a8w4(NPU_GEMM_QDQ_ARGS);

/* gemm_a16w8, gemm_a16a8, gemm_a16w8_stream: A I16 / U16, B I8 / U8, output
 * I16 / U16. A is split into its low (unsigned) and high (signed for I16) bytes:
 *   v = c3 ? mac(colsum, c3, c0) : c0
 *   v = mac(gemm_lo, c2, v);  v = mac(gemm_hi, 256 c2, v);  v = fma(S, c1, v)
 * S = fp32(fp32(sum_k lo) + 256 sum_k hi) (the byte planes are summed apart);
 * gemm_a16w8_stream sums the whole value: S = fp32(rowsum). */
int npu_gemm_a16w8(NPU_GEMM_QDQ_ARGS);
int npu_gemm_a16a8(NPU_GEMM_QDQ_ARGS);
int npu_gemm_a16w8_stream(NPU_GEMM_QDQ_ARGS);

/* gemm_a16w4: as gemm_a16w8 (int4 weights held in I8 / U8), but the two
 * products are joined before the c2 MAC; c3 must be null:
 *   g = fma(gemm_hi, 256, fp32(gemm_lo));  v = fma(g, c2, c0);  v = fma(S, c1, v) */
int npu_gemm_a16w4(NPU_GEMM_QDQ_ARGS);

/* gemm_a16w8_hybrid / _hybrid_channelwise: the ifm-in-matrix kernel, two
 * modes. Columns [0, mode_a_cols) of every M block run in mode A (the first N
 * sub-volume, which also computes the activation sum), the rest in mode B.
 * For I16 input, mode A reads the low byte as lo - 128 (lo XOR 0x80 taken
 * signed) and mode B as the unsigned lo; both use S = fp32(fp32(sum_k lo_A) +
 * 256 sum_k hi), lo_A the mode-A low byte. c0_a / c0_b are the per-mode c0
 * words (mllib packs c0 compensated for the -128 bias into mode A);
 * c2 per column (tensorwise: repeated):
 *   v = mac(gemm_lo_mode, c2, c0_mode);  v = mac(gemm_hi, 256 c2, v);  v = fma(S, c1, v)
 * mode_a_cols = N runs every column in mode A. */
#define NPU_GEMM_HYBRID_ARGS                                                               \
  void *stream, const void *a, npu_dtype a_type, const void *b, npu_dtype b_type, void *out, \
      npu_dtype out_type, int64_t m, int64_t k, int64_t n, const float *c0_a,               \
      const float *c0_b, const float *c1, const float *c2, int64_t mode_a_cols,             \
      npu_activation activation, float act0, float act1
int npu_gemm_a16w8_hybrid(NPU_GEMM_HYBRID_ARGS);
int npu_gemm_a16w8_hybrid_channelwise(NPU_GEMM_HYBRID_ARGS);

/* qdq_gemm_add_a8w8: the gemm_a8w8 GEMM in fp32 (G, no activation) plus a
 * requantized int8 / uint8 addend ifm3 (M x N), output I8 / U8. add_c0 / add_c1
 * / add_c2 are the RTP words (add_c0 = 1, add_c1 = s3 / s_out, add_c2 =
 * -zp3 s3 / s_out, double rounded to float). kpass = 0 (il (1,1) (2,2) (1,4)):
 *   out = mac(ifm3, add_c1, fma(G, add_c0, add_c2))
 * kpass = 1 (il (3,3) (4,4)): the add is folded into the GEMM coefficients first,
 *   c0 = fma(c0, add_c0, add_c2), c1 = fma(c1, add_c0, 0), c2 = fma(c2, add_c0, 0),
 *   then out = mac(ifm3, add_c1, G). */
int npu_qdq_gemm_add_a8w8(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, const void* ifm3, npu_dtype ifm3_type, void* out,
                          npu_dtype out_type, int64_t m, int64_t k, int64_t n, const float* c0,
                          const float* c1, const float* c2, float add_c0, float add_c1,
                          float add_c2, int kpass);

/* gemm_add_a16w8: the gemm_a16w8_stream GEMM in fp32 (G, S = fp32(rowsum)) plus
 * a U16 residual (M x N), output U16:
 *   out = mac(residual, add_c1, fma(add_c2, add_c0, G)) */
int npu_gemm_add_a16w8(void* stream, const void* a, npu_dtype a_type, const void* b,
                       npu_dtype b_type, const void* residual, void* out, int64_t m, int64_t k,
                       int64_t n, const float* c0, const float* c1, const float* c2,
                       float add_c0, float add_c1, float add_c2);

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
