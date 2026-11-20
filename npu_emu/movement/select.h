/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Selection: abs / neg / clip, broadcasting compare and min / max, arg max /
 * min, reduce max / min, max pooling, top-k. Tensors are dense HWC.
 *
 * Elementwise, reduce and pooling min / max order -0 below +0 and give the
 * canonical NaN for a NaN operand (fp8 max pooling: see there);
 * compares are IEEE. row_decode: fp8 decode through the identity multiply: an
 * fp8 NaN makes the 8 consecutive elements of its row (c / 8) NaN, -0 gives +0.
 */
#ifndef NPU_EMU_SELECT_H_
#define NPU_EMU_SELECT_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* elementwise unary */

/* Clears the sign bit; BF16 / F16. */
int npu_elementwise_abs_bf16(void* stream, const void* in, npu_dtype type,
                             void* out, int64_t n);
/* 0 - x in BF16 / F16: both zeros give +0, a NaN flips its sign. */
int npu_elementwise_neg_bf16(void* stream, const void* in, npu_dtype type,
                             void* out, int64_t n);
/* min(max(x, lo), hi), lo / hi rounded to `type` half to even. */
int npu_elementwise_clip_bf16(void* stream, const void* in, npu_dtype type,
                              void* out, int64_t n, float lo, float hi);
/* out = fp8(|row_decode(x)| * scale), half away from zero, not saturated
 * (|v| >= 464 -> NaN). */
int npu_elementwise_abs_fp8(void* stream, const void* in, void* out, int64_t n,
                            float scale);
/* out = fp8(row_decode(x) * scale); the caller passes a negative scale. */
int npu_elementwise_neg_fp8(void* stream, const void* in, void* out, int64_t n,
                            float scale);
/* out = fp8(clamp(row_decode(x) * scale, lo, hi)), half away from zero. */
int npu_elementwise_clip_fp8(void* stream, const void* in, void* out,
                             int64_t n, float scale, float lo, float hi);

/* 3D broadcasting compare / min / max: out = op(a, b) over the full shape (per dimension the
 * larger of shape_a and shape_b, host arrays {H, W, C}; a dimension of 1 broadcasts). Compare
 * outputs are int8 0 / 1. */

/* BF16 / F16. min / max return the canonical NaN (0x7FC0 / 0x7E00) if either
 * operand is NaN, else the selected operand's bits. */
int npu_elementwise_broadcasting3d_equal_bf16(void* stream, const void* a, const void* b,
    npu_dtype type, void* out, const int64_t* shape_a, const int64_t* shape_b);
int npu_elementwise_broadcasting3d_greater_bf16(void* stream, const void* a, const void* b,
    npu_dtype type, void* out, const int64_t* shape_a, const int64_t* shape_b);
int npu_elementwise_broadcasting3d_greater_or_equal_bf16(void* stream, const void* a, const void* b,
    npu_dtype type, void* out, const int64_t* shape_a, const int64_t* shape_b);
int npu_elementwise_broadcasting3d_max_bf16(void* stream, const void* a, const void* b,
    npu_dtype type, void* out, const int64_t* shape_a, const int64_t* shape_b);
int npu_elementwise_broadcasting3d_min_bf16(void* stream, const void* a, const void* b,
    npu_dtype type, void* out, const int64_t* shape_a, const int64_t* shape_b);

/* fp8: op(row_decode(a) * a_scale, row_decode(b) * b_scale), products rounded
 * to fp32. min / max: canonical NaN if either product is NaN, then fp8 half
 * away from zero. A C-broadcast operand is not row-spread. */
int npu_elementwise_broadcasting3d_equal_fp8(void* stream, const void* a, float a_scale,
    const void* b, float b_scale, void* out, const int64_t* shape_a, const int64_t* shape_b);
int npu_elementwise_broadcasting3d_greater_fp8(void* stream, const void* a, float a_scale,
    const void* b, float b_scale, void* out, const int64_t* shape_a, const int64_t* shape_b);
int npu_elementwise_broadcasting3d_greater_or_equal_fp8(void* stream, const void* a, float a_scale,
    const void* b, float b_scale, void* out, const int64_t* shape_a, const int64_t* shape_b);
int npu_elementwise_broadcasting3d_max_fp8(void* stream, const void* a, float a_scale,
    const void* b, float b_scale, void* out, const int64_t* shape_a, const int64_t* shape_b);
int npu_elementwise_broadcasting3d_min_fp8(void* stream, const void* a, float a_scale,
    const void* b, float b_scale, void* out, const int64_t* shape_a, const int64_t* shape_b);

/* arg max / arg min over one axis of H x W x C (axis 0 / 1 / 2 = H / W / C);
 * out is the index tensor with that axis removed. Ties keep the first index.
 * Indices wrap modulo 2^b (b = 8 for a U8 index in the bf16 / a16 / arg min
 * entries, else 16); for axis C the minimum over 64 channel lanes is taken
 * after the wrap. A lane where no element beats the start value reports
 * index 0. U32 / I64 outputs are zero-extended, F32 is (float)index. NaN
 * never wins. */

/* in I8 / U8; out U8 / U16 / U32 / I64 / F32. */
int npu_arg_max_c64_a8(void* stream, const void* in, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t H, int64_t W,
                       int64_t C, int axis);
/* in I16 / U16; out U8 / U16. */
int npu_arg_max_c64_a16(void* stream, const void* in, npu_dtype in_type,
                        void* out, npu_dtype out_type, int64_t H, int64_t W,
                        int64_t C, int axis);
/* in I16 / U16; out I64. */
int npu_arg_max_c64_a16_int64(void* stream, const void* in, npu_dtype in_type,
                              void* out, int64_t H, int64_t W, int64_t C,
                              int axis);
/* in I32 / U32; out U8 / U16 / U32 / I64 / F32. */
int npu_arg_max_c64_a32(void* stream, const void* in, npu_dtype in_type,
                        void* out, npu_dtype out_type, int64_t H, int64_t W,
                        int64_t C, int axis);
/* in BF16 / F16; out U8 / U16. Start value is the smallest positive normal
 * (0x0080 / 0x0400). */
int npu_arg_max_c64_bf16(void* stream, const void* in, npu_dtype in_type,
                         void* out, npu_dtype out_type, int64_t H, int64_t W,
                         int64_t C, int axis);
/* Compares fp8 values (-0 == +0), start value -448; out U8 / U16 / U32 / I64 /
 * F32. */
int npu_arg_max_c64_fp8(void* stream, const void* in, void* out,
                        npu_dtype out_type, int64_t H, int64_t W, int64_t C,
                        int axis);
/* in I8 / U8 (a8) or I16 / U16 (a16); out U8 / U16. */
int npu_arg_min_c64_a8(void* stream, const void* in, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t H, int64_t W,
                       int64_t C, int axis);
int npu_arg_min_c64_a16(void* stream, const void* in, npu_dtype in_type,
                        void* out, npu_dtype out_type, int64_t H, int64_t W,
                        int64_t C, int axis);
/* in BF16 / F16; out U8 / U16. Start value is the largest finite value. */
int npu_arg_min_c64_bf16(void* stream, const void* in, npu_dtype in_type,
                         void* out, npu_dtype out_type, int64_t H, int64_t W,
                         int64_t C, int axis);

/* reduce max / min of H x W x C (BF16 / F16); axes bit 0 / 1 / 2 = reduce C /
 * W / H. The result is clamped to the finite range (-inf -> most negative
 * finite for max, +inf -> largest finite for min). */
int npu_reduce_max_c64_bf16(void* stream, const void* in, npu_dtype type,
                            void* out, int64_t H, int64_t W, int64_t C,
                            int axes);
int npu_reduce_min_c64_bf16(void* stream, const void* in, npu_dtype type,
                            void* out, int64_t H, int64_t W, int64_t C,
                            int axes);

/* max pooling: Hi x Wi x C, windows kh x kw at strides sh x sw, no padding;
 * Ho = (Hi - kh) / sh + 1, Wo = (Wi - kw) / sw + 1. */

/* BF16 / F16 (c32: BF16); output is the selected element's bits. */
int npu_maxpool_bf16(void* stream, const void* in, npu_dtype type, void* out,
                     int64_t Hi, int64_t Wi, int64_t C, int kh, int kw, int sh,
                     int sw);
int npu_maxpool_bf16_c32(void* stream, const void* in, void* out, int64_t Hi,
                         int64_t Wi, int64_t C, int kh, int kw, int sh,
                         int sw);
/* m = window max of the fp8 values (-0 below +0; a NaN first tap is kept,
 * later NaN taps are skipped); r = fma(row_decode(m),
 * c2[c], c0[c]); with clip, r = clamp(r, act_lo, act_hi); out = fp8(r) half to
 * even, not saturated. c0 / c2 hold `channels` floats (1 or C). */
int npu_maxpool_fp8(void* stream, const void* in, void* out, int64_t Hi,
                    int64_t Wi, int64_t C, int kh, int kw, int sh, int sw,
                    const float* c0, const float* c2, int64_t channels,
                    int clip, float act_lo, float act_hi);
/* out[c] = max over H x W of in[.][.][c] and the smallest positive normal
 * (0x0080 / 0x0400). */
int npu_globalmaxpool_bf16(void* stream, const void* in, npu_dtype type,
                           void* out, int64_t H, int64_t W, int64_t C);
/* m = max over H x W of the fp8 values; out[c] = fp8(fma(row_decode(m)[c], c1,
 * c0)) half to even. */
int npu_globalmaxpool_fp8(void* stream, const void* in, void* out, int64_t H,
                          int64_t W, int64_t C, float c0, float c1);

/* top-k of rows x W: K rounds per row of arg max (arg min if !largest) as
 * npu_arg_max_c64_* with axis C (index modulo 2^8 / 2^16 for index_type U8 /
 * U16), each emitting value and index and then overwriting element [index]
 * with the start value (-inf / +inf for bf16, the type's min / max for
 * integers). Rounds continue past the valid elements. out_values and
 * out_indices are rows x K. */
int npu_topk2d_bf16(void* stream, const void* in, npu_dtype type,
                    void* out_values, void* out_indices, npu_dtype index_type,
                    int64_t rows, int64_t W, int64_t K, int largest);
/* in I8 / U8; index_type U8 / U16. */
int npu_topk2d_a8(void* stream, const void* in, npu_dtype type,
                  void* out_values, void* out_indices, npu_dtype index_type,
                  int64_t rows, int64_t W, int64_t K, int largest);
/* in I16 / U16; index_type U16. */
int npu_topk2d_a16(void* stream, const void* in, npu_dtype type,
                   void* out_values, void* out_indices, npu_dtype index_type,
                   int64_t rows, int64_t W, int64_t K, int largest);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_SELECT_H_
