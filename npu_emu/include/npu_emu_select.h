/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 data-movement / selection wrappers
 * (select group): elementwise abs / neg / clip, 3D-broadcasting compare and
 * min / max, arg max / min, reduce max / min, max pooling and top-k.
 * Conventions as in npu_emu.h. Tensors are logical HWC (C innermost), without
 * the NPU's C64 tiling or padding.
 *
 * Rules shared by these kernels (checked on aiesim outputs unless marked
 * "unverified"):
 *   - bf16 / f16 VMAX / VMIN order -0 below +0: max(+0, -0) = +0 and
 *     min(+0, -0) = -0 whatever the operand order. Comparisons (gt / ge / eq)
 *     are IEEE: -0 == +0 and NaN compares false.
 *   - fp32 VMAX / VMIN / clamp (the fp8 kernels compute in fp32) return +0 for
 *     a zero result, also when the selected operand is -0.
 *   - fp8 inputs are decoded by an 8x8 identity matrix multiply: a NaN makes
 *     the 8 values of its row NaN. The row is 8 consecutive channels of one
 *     pixel (c / 8), i.e. 8 consecutive elements of a flat buffer. The arg max
 *     kernel compares raw fp8 codes and has no row effect.
 *   - Selection kernels (max pool, reduce, min / max) output the bits of the
 *     selected input element; NaN inputs to them are unverified unless noted.
 */
#ifndef NPU_EMU_SELECT_H_
#define NPU_EMU_SELECT_H_

#include <stdint.h>

#include "npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- elementwise unary ------------------------------------------------- */

/* elementwise_abs_bf16: clear the sign bit (BF16 / F16), so -0, inf and NaN
 * payloads pass bit-exactly. */
int npu_elementwise_abs_bf16(void* stream, const void* in, npu_dtype type,
                             void* out, int64_t n);
/* elementwise_neg_bf16: 0 - x in BF16 / F16. Both zeros give +0; subnormals
 * are kept. NaN gets its sign flipped (unverified). */
int npu_elementwise_neg_bf16(void* stream, const void* in, npu_dtype type,
                             void* out, int64_t n);
/* elementwise_clip_bf16: min(max(x, lo), hi) with lo / hi rounded to `type`
 * half to even (exact in every aiesim case; rounding unverified) and the bf16
 * zero order above. NaN x passes through (unverified). */
int npu_elementwise_clip_bf16(void* stream, const void* in, npu_dtype type,
                              void* out, int64_t n, float lo, float hi);
/* elementwise_abs_fp8: fp8(|row_decode(x)| * scale), the product rounded to
 * fp32, then to fp8 half away from zero (symmetric_inf), not saturated
 * (|v| >= 464 -> NaN). scale = the coeff word (ifm_scale / ofm_scale). No
 * aiesim output exists for this wrapper: unverified. */
int npu_elementwise_abs_fp8(void* stream, const void* in, void* out, int64_t n,
                            float scale);
/* elementwise_neg_fp8: as abs without the absolute value; the negation is in
 * scale (the coeff word is -ifm_scale / ofm_scale). */
int npu_elementwise_neg_fp8(void* stream, const void* in, void* out, int64_t n,
                            float scale);
/* elementwise_clip_fp8: fp8(clamp(row_decode(x) * scale, lo, hi)), half away
 * from zero; a zero result is +0. scale / lo / hi are the wts words
 * (ifm_scale / ofm_scale, f_min / ofm_scale, f_max / ofm_scale). */
int npu_elementwise_clip_fp8(void* stream, const void* in, void* out,
                             int64_t n, float scale, float lo, float hi);

/* ---- 3D broadcasting compare / min / max --------------------------------
 * out[h][w][c] = op(a[h][w][c], b[h][w][c]) over an H x W x C output. An
 * operand with bit 0 / 1 / 2 of its *_bcast set has size 1 along C / W / H
 * and is broadcast. a is the logical first operand (the NPU's ifm2 when
 * swap_inputs is set). Compare outputs are int8 0 / 1. */

/* bf16 / f16 (type BF16 / F16): IEEE compare; min / max return the canonical
 * NaN (bf16 0x7FC0, f16 0x7E00) if either operand is NaN, else the selected
 * operand's bits with the zero order above. */
int npu_elementwise_broadcasting3d_equal_bf16(void* stream, const void* a,
                                              int a_bcast, const void* b,
                                              int b_bcast, npu_dtype type,
                                              void* out, int64_t H, int64_t W,
                                              int64_t C);
int npu_elementwise_broadcasting3d_greater_bf16(void* stream, const void* a,
                                                int a_bcast, const void* b,
                                                int b_bcast, npu_dtype type,
                                                void* out, int64_t H,
                                                int64_t W, int64_t C);
int npu_elementwise_broadcasting3d_greater_or_equal_bf16(
    void* stream, const void* a, int a_bcast, const void* b, int b_bcast,
    npu_dtype type, void* out, int64_t H, int64_t W, int64_t C);
int npu_elementwise_broadcasting3d_max_bf16(void* stream, const void* a,
                                            int a_bcast, const void* b,
                                            int b_bcast, npu_dtype type,
                                            void* out, int64_t H, int64_t W,
                                            int64_t C);
int npu_elementwise_broadcasting3d_min_bf16(void* stream, const void* a,
                                            int a_bcast, const void* b,
                                            int b_bcast, npu_dtype type,
                                            void* out, int64_t H, int64_t W,
                                            int64_t C);

/* fp8: each operand is row-decoded (rows of 8 channels within a pixel of that
 * operand) and multiplied by its coeff word (a_scale / b_scale, the product
 * rounded to fp32); compare is IEEE on the products, so a NaN anywhere in a
 * row makes its 8 lanes compare false. min / max: canonical NaN if either
 * product is NaN, a zero result is +0, then fp8 half away from zero. On the
 * NPU a C-broadcast operand's row partners are the 7 padding bytes after it;
 * here they are taken as non-NaN. */
int npu_elementwise_broadcasting3d_equal_fp8(void* stream, const void* a,
                                             int a_bcast, float a_scale,
                                             const void* b, int b_bcast,
                                             float b_scale, void* out,
                                             int64_t H, int64_t W, int64_t C);
int npu_elementwise_broadcasting3d_greater_fp8(void* stream, const void* a,
                                               int a_bcast, float a_scale,
                                               const void* b, int b_bcast,
                                               float b_scale, void* out,
                                               int64_t H, int64_t W,
                                               int64_t C);
int npu_elementwise_broadcasting3d_greater_or_equal_fp8(
    void* stream, const void* a, int a_bcast, float a_scale, const void* b,
    int b_bcast, float b_scale, void* out, int64_t H, int64_t W, int64_t C);
int npu_elementwise_broadcasting3d_max_fp8(void* stream, const void* a,
                                           int a_bcast, float a_scale,
                                           const void* b, int b_bcast,
                                           float b_scale, void* out, int64_t H,
                                           int64_t W, int64_t C);
int npu_elementwise_broadcasting3d_min_fp8(void* stream, const void* a,
                                           int a_bcast, float a_scale,
                                           const void* b, int b_bcast,
                                           float b_scale, void* out, int64_t H,
                                           int64_t W, int64_t C);

/* ---- arg max / arg min (C64 kernels) --------------------------------------
 * in: H x W x C; axis 0 / 1 / 2 = H / W / C (one axis). out: the index tensor
 * with that axis removed, row-major, element type out_type.
 * The NPU scans 64 channel lanes; an index replaces the lane's best only on a
 * strict improvement, so ties keep the first index, and for axis C the result
 * is the smallest index among the lanes holding the overall best. Indices are
 * computed modulo 2^b (b = 8 for a uint8 index type of the bf16 / a16 / argmin
 * kernels, else 16) - for axis C the lane minimum is taken after the wrap.
 * The running best starts at a per-kernel init value that never counts as an
 * occurrence: a lane where no element beats it reports index 0. U32 / I64
 * outputs are the 16-bit index zero-extended, F32 is (float)index. NaN never
 * becomes a lane's best (unverified for bf16). */

/* arg_max_c64_a8: in I8 / U8; out U8 / U16 / U32 / I64 / F32. */
int npu_arg_max_c64_a8(void* stream, const void* in, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t H, int64_t W,
                       int64_t C, int axis);
/* arg_max_c64_a16: in I16 / U16; out U8 / U16 (index arithmetic in out_type). */
int npu_arg_max_c64_a16(void* stream, const void* in, npu_dtype in_type,
                        void* out, npu_dtype out_type, int64_t H, int64_t W,
                        int64_t C, int axis);
/* arg_max_c64_a16_int64: in I16 / U16; out I64. */
int npu_arg_max_c64_a16_int64(void* stream, const void* in, npu_dtype in_type,
                              void* out, int64_t H, int64_t W, int64_t C,
                              int axis);
/* arg_max_c64_a32: in I32 / U32; out U8 / U16 / U32 / I64 / F32. */
int npu_arg_max_c64_a32(void* stream, const void* in, npu_dtype in_type,
                        void* out, npu_dtype out_type, int64_t H, int64_t W,
                        int64_t C, int axis);
/* arg_max_c64_bf16: in BF16 / F16; out U8 / U16 (index arithmetic in
 * out_type). The running max starts at numeric_limits<T>::min(), the smallest
 * positive normal (bf16 0x0080, f16 0x0400), not -inf: a lane whose values
 * are all <= that value (e.g. all negative) reports index 0. This follows the
 * kernel source; no aiesim case reaches it (unverified). */
int npu_arg_max_c64_bf16(void* stream, const void* in, npu_dtype in_type,
                         void* out, npu_dtype out_type, int64_t H, int64_t W,
                         int64_t C, int axis);
/* arg_max_c64_fp8: compares the raw E4M3FN codes (-0 == +0, NaN never wins),
 * starting from -448; out U8 / U16 / U32 / I64 / F32. */
int npu_arg_max_c64_fp8(void* stream, const void* in, void* out,
                        npu_dtype out_type, int64_t H, int64_t W, int64_t C,
                        int axis);
/* arg_min_c64_a8 / _a16: in I8 / U8 resp. I16 / U16; out U8 / U16. */
int npu_arg_min_c64_a8(void* stream, const void* in, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t H, int64_t W,
                       int64_t C, int axis);
int npu_arg_min_c64_a16(void* stream, const void* in, npu_dtype in_type,
                        void* out, npu_dtype out_type, int64_t H, int64_t W,
                        int64_t C, int axis);
/* arg_min_c64_bf16: in BF16 / F16; out U8 / U16; the running min starts at
 * the largest finite value, so +inf never counts as an occurrence. */
int npu_arg_min_c64_bf16(void* stream, const void* in, npu_dtype in_type,
                         void* out, npu_dtype out_type, int64_t H, int64_t W,
                         int64_t C, int axis);

/* ---- reduce max / min ----------------------------------------------------
 * in: H x W x C (BF16 / F16); axes: bit 0 / 1 / 2 = reduce C / W / H (any
 * non-empty combination); out: the remaining axes, row-major. The result is
 * clamped by the kernel's start value: reduce_max gives at least the most
 * negative finite value (-inf -> 0xFF7F / 0xFBFF), reduce_min at most the
 * largest finite value. NaN inputs are skipped (unverified). */
int npu_reduce_max_c64_bf16(void* stream, const void* in, npu_dtype type,
                            void* out, int64_t H, int64_t W, int64_t C,
                            int axes);
int npu_reduce_min_c64_bf16(void* stream, const void* in, npu_dtype type,
                            void* out, int64_t H, int64_t W, int64_t C,
                            int axes);

/* ---- max pooling -----------------------------------------------------------
 * in: Hi x Wi x C; windows kh x kw at strides sh x sw, no padding;
 * out: Ho x Wo x C with Ho = (Hi - kh) / sh + 1, Wo = (Wi - kw) / sw + 1. The
 * window max starts from the first tap; a NaN first tap sticks and later NaN
 * taps are skipped (unverified). */

/* maxpool_bf16 (BF16 / F16) and maxpool_bf16_c32 (BF16): selected bits. No
 * aiesim output exists for maxpool_bf16_c32 (unverified; same arithmetic). */
int npu_maxpool_bf16(void* stream, const void* in, npu_dtype type, void* out,
                     int64_t Hi, int64_t Wi, int64_t C, int kh, int kw, int sh,
                     int sw);
int npu_maxpool_bf16_c32(void* stream, const void* in, void* out, int64_t Hi,
                         int64_t Wi, int64_t C, int kh, int kw, int sh,
                         int sw);
/* maxpool_fp8: m = window max of the fp8 codes (-0 below +0);
 * v = row_decode(m) over the 8-channel rows of the output pixel;
 * r = fma(v, c2[c], c0[c]) rounded once; with clip, r = clamp(r, act_lo,
 * act_hi) (a zero result becomes +0); out = fp8(r) half to even, not
 * saturated. c0 / c2 hold `channels` floats: 1 = tensor-wise, C = per
 * channel. */
int npu_maxpool_fp8(void* stream, const void* in, void* out, int64_t Hi,
                    int64_t Wi, int64_t C, int kh, int kw, int sh, int sw,
                    const float* c0, const float* c2, int64_t channels,
                    int clip, float act_lo, float act_hi);
/* globalmaxpool_bf16: out[c] = max over H x W of in[.][.][c] and the kernel's
 * start value, the smallest positive normal (bf16 0x0080, f16 0x0400): a
 * channel whose values are all <= it outputs 0x0080 / 0x0400 (kernel source;
 * no aiesim case reaches it, unverified). */
int npu_globalmaxpool_bf16(void* stream, const void* in, npu_dtype type,
                           void* out, int64_t H, int64_t W, int64_t C);
/* globalmaxpool_fp8: m = max over H x W of the fp8 codes; out[c] =
 * fp8(fma(row_decode(m)[c], c1, c0)) half to even. c0 / c1 = coeff words 0 /
 * 1. */
int npu_globalmaxpool_fp8(void* stream, const void* in, void* out, int64_t H,
                          int64_t W, int64_t C, float c0, float c1);

/* ---- top-k ---------------------------------------------------------------
 * in: rows x W; per row K rounds of: lane-wise arg max (min if !largest) as
 * in npu_arg_max_c64_* with axis C (64 lanes, first index on ties, index
 * modulo 2^b, b = 8 / 16 for index_type U8 / U16), emit the value and index,
 * then overwrite element [index] with the start value (-inf / +inf for bf16,
 * the type's min / max for integers). Rounds keep going after the valid
 * elements run out, and an overwritten element equal to a real one can be
 * picked again, as on the NPU. out_values: rows x K of the input type;
 * out_indices: rows x K of index_type. The integer wrappers run with
 * K = K_len_even. */
int npu_topk2d_bf16(void* stream, const void* in, npu_dtype type,
                    void* out_values, void* out_indices, npu_dtype index_type,
                    int64_t rows, int64_t W, int64_t K, int largest);
/* topk2d_a8: in I8 / U8; index_type U8 / U16. */
int npu_topk2d_a8(void* stream, const void* in, npu_dtype type,
                  void* out_values, void* out_indices, npu_dtype index_type,
                  int64_t rows, int64_t W, int64_t K, int largest);
/* topk2d_a16: in I16 / U16; index_type U16. */
int npu_topk2d_a16(void* stream, const void* in, npu_dtype type,
                   void* out_values, void* out_indices, npu_dtype index_type,
                   int64_t rows, int64_t W, int64_t K, int largest);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_SELECT_H_
