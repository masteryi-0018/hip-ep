/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Quantized selection / movement: abs / neg / clip / compare / max, where,
 * broadcast min, reduce, max pooling, concat, gather, nearest grid sample /
 * resize, and data movement with a fused requantization.
 *
 * mac(x, s, o) = x * s + o rounded once to fp32; rne = round half to even;
 * sat = saturate to the output type. Scales and offsets are the fp32
 * coefficients the kernel receives (s_in, zp_in, s_out, zp_out below are the
 * quantization parameters they are derived from). Tensors are dense row-major
 * (HWC for 3-D); integer types are I8 / U8 / I16 / U16.
 */
#ifndef NPU_EMU_QDQ_H_
#define NPU_EMU_QDQ_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* elementwise unary; in_type / out_type I8 / U8 (a8) or I16 / U16 (a16) */

/* out = sat(rne(max(mac(x, c1, fp32(c0 + ofm_zp)), mac(x, -c1, fp32(ofm_zp - c0)))))
 * with c1 = s_in / s_out, c0 = -c1 * zp_in, ofm_zp = zp_out. */
int npu_elementwise_abs_a8(void* stream, const void* in, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n, float c1,
                           float c0, float ofm_zp);
/* The 16-bit variant of npu_elementwise_abs_a8. */
int npu_elementwise_abs_a16(void* stream, const void* in, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c1,
                            float c0, float ofm_zp);
/* out = sat(rne(mac(x, c1, c0))), c1 = -s_in / s_out, c0 = zp_out - c1 * zp_in. */
int npu_elementwise_neg_a8(void* stream, const void* in, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n, float c1,
                           float c0);
/* The 16-bit variant of npu_elementwise_neg_a8. */
int npu_elementwise_neg_a16(void* stream, const void* in, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c1,
                            float c0);
/* out = sat(rne(min(max(mac(x, c1, c0), cmin), cmax))), c1 = s_in / s_out,
 * c0 = zp_out - c1 * zp_in; cmin / cmax are in the output's quantized domain. */
int npu_elementwise_clip_a8(void* stream, const void* in, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c1,
                            float c0, float cmin, float cmax);
/* The 16-bit variant of npu_elementwise_clip_a8. */
int npu_elementwise_clip_a16(void* stream, const void* in, npu_dtype in_type,
                             void* out, npu_dtype out_type, int64_t n,
                             float c1, float c0, float cmin, float cmax);

/* elementwise binary; same shape, no broadcasting */

/* out (int8 0 / 1) = fa == fb resp. fa > fb, with fa = mac(a, scale_a, zp_a),
 * fb = mac(b, scale_b, zp_b); scale_x = s_x, zp_x = -zp_x * s_x. a_type and
 * b_type may differ in sign. */
int npu_elementwise_equal_a8(void* stream, const void* a, npu_dtype a_type,
                             const void* b, npu_dtype b_type, void* out,
                             int64_t n, float scale_a, float scale_b,
                             float zp_a, float zp_b);
/* The 16-bit variant of npu_elementwise_equal_a8. */
int npu_elementwise_equal_a16(void* stream, const void* a, npu_dtype a_type,
                              const void* b, npu_dtype b_type, void* out,
                              int64_t n, float scale_a, float scale_b,
                              float zp_a, float zp_b);
/* a > b; otherwise npu_elementwise_equal_a8. */
int npu_elementwise_greater_a8(void* stream, const void* a, npu_dtype a_type,
                               const void* b, npu_dtype b_type, void* out,
                               int64_t n, float scale_a, float scale_b,
                               float zp_a, float zp_b);
/* The 16-bit variant of npu_elementwise_greater_a8. */
int npu_elementwise_greater_a16(void* stream, const void* a, npu_dtype a_type,
                                const void* b, npu_dtype b_type, void* out,
                                int64_t n, float scale_a, float scale_b,
                                float zp_a, float zp_b);
/* out = sat(rne(fma(max(fa, fb), scale_o, zp_o))), fa / fb as above,
 * scale_o = 1 / s_out, zp_o = zp_out. */
int npu_elementwise_max_a8(void* stream, const void* a, npu_dtype a_type,
                           const void* b, npu_dtype b_type, void* out,
                           npu_dtype out_type, int64_t n, float scale_a,
                           float scale_b, float scale_o, float zp_a, float zp_b,
                           float zp_o);
/* The 16-bit variant of npu_elementwise_max_a8. */
int npu_elementwise_max_a16(void* stream, const void* a, npu_dtype a_type,
                            const void* b, npu_dtype b_type, void* out,
                            npu_dtype out_type, int64_t n, float scale_a,
                            float scale_b, float scale_o, float zp_a,
                            float zp_b, float zp_o);

/* where / broadcast min */

/* x and out have shape = {H, W, C} (host array) of 1-byte (a8) or 2-byte (a16) elements,
 * copied bit for bit; cond is int8 of cond_shape (each dimension 1 or shape's); y_bits is
 * the scalar operand.
 *   scalar_position == 1: out = cond != 0 ? x : y
 *   scalar_position == 0: out = cond == 0 ? x : y */
int npu_elementwise_broadcasting3d_where_a8(void* stream, const void* cond, const void* x,
    void* out, const int64_t* shape, const int64_t* cond_shape, uint32_t y_bits,
    int scalar_position);
/* The 16-bit variant of npu_elementwise_broadcasting3d_where_a8. */
int npu_elementwise_broadcasting3d_where_a16(void* stream, const void* cond, const void* x,
    void* out, const int64_t* shape, const int64_t* cond_shape, uint32_t y_bits,
    int scalar_position);
/* The selection above (cond int8 for a8, int16 for a16), then
 * out = sat(rne(mac(sel, scale, offset))) for x and y alike; scale = s_in /
 * s_out, offset = zp_out - zp_in * scale. */
int npu_qdq_elementwise_broadcasting3d_where_a8(void* stream, const void* cond, const void* x,
    npu_dtype dtype, void* out, const int64_t* shape, const int64_t* cond_shape, int32_t y,
    int scalar_position, float scale, float offset);
int npu_qdq_elementwise_broadcasting3d_where_a16(void* stream, const void* cond, const void* x,
    npu_dtype dtype, void* out, const int64_t* shape, const int64_t* cond_shape, int32_t y,
    int scalar_position, float scale, float offset);
/* a has the full shape shape1 = {H, W, C}, b has shape2 (host arrays; each dimension 1 or
 * shape1's) and is broadcast to it;
 * out = sat(rne(fma(min(fa, fb), scale_o, zp_o))), coefficients as for
 * npu_elementwise_max_a8. */
int npu_qdq_elementwise_broadcasting3d_min_a8(void* stream, const void* a, const void* b,
    npu_dtype dtype, void* out, const int64_t* shape1, const int64_t* shape2, float scale_a,
    float scale_b, float scale_o, float zp_a, float zp_b, float zp_o);
int npu_qdq_elementwise_broadcasting3d_min_a16(void* stream, const void* a, const void* b,
    npu_dtype dtype, void* out, const int64_t* shape1, const int64_t* shape2, float scale_a,
    float scale_b, float scale_o, float zp_a, float zp_b, float zp_o);

/* reductions */

/* Integer max / min of in [H][W][C] over the axes in axis_flags (bit 0 = C,
 * bit 1 = W, bit 2 = H; 1..7); out has the reduced axes removed. */
int npu_reduce_max_c64_a8(void* stream, const void* in, npu_dtype dtype,
                          void* out, int64_t H, int64_t W, int64_t C,
                          int axis_flags);
int npu_reduce_max_c64_a16(void* stream, const void* in, npu_dtype dtype,
                           void* out, int64_t H, int64_t W, int64_t C,
                           int axis_flags);
int npu_reduce_min_c64_a8(void* stream, const void* in, npu_dtype dtype,
                          void* out, int64_t H, int64_t W, int64_t C,
                          int axis_flags);
int npu_reduce_min_c64_a16(void* stream, const void* in, npu_dtype dtype,
                           void* out, int64_t H, int64_t W, int64_t C,
                           int axis_flags);
/* The reduction above, then out = sat(rne(mac(r, scale, offset) * 2^shift));
 * scale = s_in / s_out, offset = zp_out - scale * zp_in, shift is signed. */
int npu_qdq_reduce_max_c64_a8(void* stream, const void* in, npu_dtype dtype,
                              void* out, int64_t H, int64_t W, int64_t C,
                              int axis_flags, float offset, float scale,
                              int shift);
int npu_qdq_reduce_max_c64_a16(void* stream, const void* in, npu_dtype dtype,
                               void* out, int64_t H, int64_t W, int64_t C,
                               int axis_flags, float offset, float scale,
                               int shift);
int npu_qdq_reduce_min_c64_a8(void* stream, const void* in, npu_dtype dtype,
                              void* out, int64_t H, int64_t W, int64_t C,
                              int axis_flags, float offset, float scale,
                              int shift);
int npu_qdq_reduce_min_c64_a16(void* stream, const void* in, npu_dtype dtype,
                               void* out, int64_t H, int64_t W, int64_t C,
                               int axis_flags, float offset, float scale,
                               int shift);

/* pooling */

/* m = max over kh x kw windows at strides sh x sw, no padding, of in [H][W][C];
 * out [Ho][Wo][C], Ho = (H - kh) / sh + 1, Wo = (W - kw) / sw + 1;
 * out = sat(rne(mac(m, scale, offset) * 2^shift)). */
int npu_qdq_maxpool_a8(void* stream, const void* in, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t H, int64_t W,
                       int64_t C, int kh, int kw, int sh, int sw, float offset,
                       float scale, int shift);
int npu_qdq_maxpool_a16(void* stream, const void* in, npu_dtype in_type,
                        void* out, npu_dtype out_type, int64_t H, int64_t W,
                        int64_t C, int kh, int kw, int sh, int sw, float offset,
                        float scale, int shift);
/* The same window max without requantization; dtype I8 / U8. */
int npu_generic_maxpool_a8(void* stream, const void* in, npu_dtype dtype,
                           void* out, int64_t H, int64_t W, int64_t C, int kh,
                           int kw, int sh, int sw);
/* Max over H and W of in [H][W][C]; out [C], requantized as npu_qdq_maxpool_a8. */
int npu_qdq_globalmaxpool_a8(void* stream, const void* in, npu_dtype in_type,
                             void* out, npu_dtype out_type, int64_t H,
                             int64_t W, int64_t C, float offset, float scale,
                             int shift);
int npu_qdq_globalmaxpool_a16(void* stream, const void* in, npu_dtype in_type,
                              void* out, npu_dtype out_type, int64_t H,
                              int64_t W, int64_t C, float offset, float scale,
                              int shift);

/* concat along C */

/* out[r] = a[r][0..ca) ++ b[r][0..cb), then sat(rne(mac(v, scale, offset)));
 * scale = s_in / s_out, offset = zp_out - zp_in * scale. */
int npu_qdq_concat_a8(void* stream, const void* a, const void* b,
                      npu_dtype in_type, void* out, npu_dtype out_type,
                      int64_t rows, int64_t ca, int64_t cb, float scale,
                      float offset);
int npu_qdq_concat_a16(void* stream, const void* a, const void* b,
                       npu_dtype in_type, void* out, npu_dtype out_type,
                       int64_t rows, int64_t ca, int64_t cb, float scale,
                       float offset);
/* Concat of n_in (1..6) tensors ins[k] = [rows][channels[k]], requantized as
 * npu_qdq_concat_a8. ins (device pointers) and channels are host arrays. */
int npu_qdq_concat_multi_c_fusion_a8(void* stream, int n_in,
                                     const void* const* ins, npu_dtype in_type,
                                     void* out, npu_dtype out_type,
                                     int64_t rows, const int64_t* channels,
                                     float scale, float offset);
int npu_qdq_concat_multi_c_fusion_a16(void* stream, int n_in,
                                      const void* const* ins,
                                      npu_dtype in_type, void* out,
                                      npu_dtype out_type, int64_t rows,
                                      const int64_t* channels, float scale,
                                      float offset);
/* As above with a (scale[k], offset[k]) per input (host arrays); in_type and
 * out_type have the same width. */
int npu_qdq_concat_multi_c_fusion(void* stream, int n_in,
                                  const void* const* ins, npu_dtype in_type,
                                  void* out, npu_dtype out_type, int64_t rows,
                                  const int64_t* channels, const float* scale,
                                  const float* offset);

/* gather / grid sample / resize */

/* out[i][k] = sat(rne(mac(in[idx[i][k]][k], scale, offset))); in [M_in][N],
 * idx int32 [M_out][N]. An index outside [0, M_in) leaves its output element
 * unwritten. */
int npu_qdq_gather_element_2din_2dindices_axis0(
    void* stream, const void* in, npu_dtype in_type, const void* idx,
    void* out, npu_dtype out_type, int64_t M_in, int64_t N, int64_t M_out,
    float scale, float offset);

/* Nearest GridSample of in [H][W][C] by grid [Ho][Wo][2] (x, y codes):
 *   g  = fp32(fp32(code - grid_zp) * grid_scale) + 1
 *   align_corners 0: ix = fma(gx, W * 0.5, -0.5)
 *   align_corners 1: ix = fp32((W - 1) * gx) * 0.5
 *   rx = sat_int16(rne(ix)); padding_mode 1 (border) clamps to [0, W - 1]
 *   v  = bf16_rne(mac(in[ry][rx][c], ifm_scale, fp32(-(float)ifm_zp * ifm_scale)))
 *        if 0 <= rx < W and 0 <= ry < H, else 0
 *   out = sat(rne(fma(v, ofm_scale, (float)ofm_zp)))
 * dtype I8 / U8 (a8, a8w8) or I16 / U16 (a16); grid codes U16 (a8, a16) or
 * grid_type I8 / U8 (a8w8). */
int npu_qdq_gridsample_nearest_a8(void* stream, const void* in,
                                  npu_dtype dtype, const void* grid, void* out,
                                  int64_t H, int64_t W, int64_t C, int64_t Ho,
                                  int64_t Wo, int align_corners,
                                  int padding_mode,
                                  const npu_gridsample_qdq* coeff);
int npu_qdq_gridsample_nearest_a16(void* stream, const void* in,
                                   npu_dtype dtype, const void* grid,
                                   void* out, int64_t H, int64_t W, int64_t C,
                                   int64_t Ho, int64_t Wo, int align_corners,
                                   int padding_mode,
                                   const npu_gridsample_qdq* coeff);
int npu_qdq_gridsample_nearest_a8w8(void* stream, const void* in,
                                    npu_dtype dtype, const void* grid,
                                    npu_dtype grid_type, void* out, int64_t H,
                                    int64_t W, int64_t C, int64_t Ho,
                                    int64_t Wo, int align_corners,
                                    int padding_mode,
                                    const npu_gridsample_qdq* coeff);
/* out[h][w][c] = in[src_y[h]][src_x[w]][c]; src_y [Ho] and src_x [Wo] are
 * device int32. With requantize != 0: out = sat(trunc(mac(v, scale, offset)))
 * (rounds toward zero). A source index outside the input leaves its output
 * elements unwritten. */
int npu_qdq_resize_nearest(void* stream, const void* in, npu_dtype dtype,
                           void* out, int64_t H, int64_t W, int64_t C,
                           int64_t Ho, int64_t Wo, const int32_t* src_y,
                           const int32_t* src_x, int requantize, float scale,
                           float offset);

/* data movement with fused requantization */

/* npu_generic_<op> of movement.h, then out = sat(rne(mac(v, c1, c0))) with
 * c1 = s_in / s_out, c0 = zp_out - c1 * zp_in. dtype I8 / U8 / I16 / U16, except
 * slice_16b_inner(_range) (I8 / I16) and slice_16b_c6463 (I16 / U16). */
int npu_qdq_reshape_innermost_requantize_fused(void* stream, const void* in,
                                               void* out, npu_dtype dtype,
                                               int64_t n, float c1, float c0);
int npu_qdq_slice_innermost_requantize_fused(void* stream, const void* in,
                                             void* out, npu_dtype dtype,
                                             int64_t rows, int64_t c,
                                             int64_t start, int64_t end,
                                             int64_t stride, float c1,
                                             float c0);
int npu_qdq_slice_16b_inner_range_requantize_fused(
    void* stream, const void* in, void* out, npu_dtype dtype, int64_t rows,
    int64_t c, int64_t start, int64_t end, float c1, float c0);
int npu_qdq_slice_16b_inner_requantize_fused(void* stream, const void* in,
                                             void* out0, void* out1,
                                             npu_dtype dtype, int64_t rows,
                                             int64_t c, int64_t split,
                                             float c1, float c0);
int npu_qdq_slice_16b_c6463_requantize_fused(void* stream, const void* in,
                                             void* out0, void* out1,
                                             npu_dtype dtype, int64_t rows,
                                             float c1, float c0);
int npu_qdq_transpose_k8n8_requantize_fused(void* stream, const void* in,
                                            void* out, npu_dtype dtype,
                                            int64_t K, int64_t N, float c1,
                                            float c0);
int npu_qdq_transpose_unaligned_requantize_fused(void* stream, const void* in,
                                                 void* out, npu_dtype dtype,
                                                 const int64_t* shape,
                                                 const int* perm, float c1,
                                                 float c0);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_QDQ_H_
