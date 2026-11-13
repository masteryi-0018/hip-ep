/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 quantized-domain selection / movement
 * wrappers (qdq group): elementwise abs / neg / clip / equal / greater / max,
 * where, broadcast min, reduce max / min, max pooling, concat, gather,
 * nearest grid sample and nearest resize, and the data-movement kernels with
 * a fused requantization. Conventions as in npu_emu.h.
 *
 * Requantization. Most kernels requantize their integer result with one
 * accfloat MAC and a float -> integer conversion:
 *
 *   v   = x * scale + offset   rounded once to fp32; an integer x enters
 *                              exactly (mac_elem_32(int32, float, accfloat))
 *   out = sat(round(v * 2^shift))   to the output type (to_vector_sign /
 *                              to_fixed); round = half to even (conv_even)
 *                              except npu_qdq_resize_nearest (toward zero)
 *
 * The scale / offset / shift arguments are the words the kernel reads from
 * its coefficient buffer (coeff32.txt), not ONNX scales and zero points. The
 * mllib host code computes them in double and stores them as fp32; the usual
 * formulas are given per function (s_in, zp_in, s_out, zp_out are the ONNX
 * quantization parameters). The buffers do not agree on naming: in some
 * kernels "c0" is the scale, in others the offset; the parameter names below
 * say which is which.
 *
 * Unless noted otherwise, tensors are contiguous, row-major and in logical
 * layout (HWC for the 3-D ones; the NPU's C64 blocking, sub-volume tiling and
 * channel padding are not emulated). Integer types are passed as npu_dtype
 * (I8 / U8 / I16 / U16). Float -> integer conversions saturate even though
 * the wrappers select saturation_mode::truncate (as for the conversion
 * kernels). Unverified on the NPU: shifts outside [-2, 2] and the points
 * marked "unverified".
 */
#ifndef NPU_EMU_QDQ_H_
#define NPU_EMU_QDQ_H_

#include <stdint.h>

#include "npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- A. elementwise unary (requantize) --------------------------------- */

/* elementwise_abs_a8 / _a16: |dequant(x)| requantized. The kernel cannot
 * fold the output zero point into an affine map (abs is not linear), so it
 * computes, with fp32 adds done once per call:
 *   p = mac(x,  c1, fp32(c0 + ofm_zp)),  q = mac(x, -c1, fp32(ofm_zp - c0))
 *   out = sat(rne(max(p, q)))
 * Coefficients: c1 = s_in / s_out, c0 = -c1 * zp_in, ofm_zp = zp_out.
 * in_type / out_type: I8 / U8 (a8) or I16 / U16 (a16), any combination. */
int npu_elementwise_abs_a8(void* stream, const void* in, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n, float c1,
                           float c0, float ofm_zp);
int npu_elementwise_abs_a16(void* stream, const void* in, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c1,
                            float c0, float ofm_zp);
/* elementwise_neg_a8 / _a16: out = sat(rne(mac(x, c1, c0))) with
 * c1 = -s_in / s_out, c0 = zp_out + zp_in * s_in / s_out. */
int npu_elementwise_neg_a8(void* stream, const void* in, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n, float c1,
                           float c0);
int npu_elementwise_neg_a16(void* stream, const void* in, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c1,
                            float c0);
/* elementwise_clip_a8 / _a16: v = mac(x, c1, c0) (the requantize map,
 * c1 = s_in / s_out, c0 = zp_out - c1 * zp_in); out = sat(rne(min(max(v,
 * cmin), cmax))) with cmin / cmax the clip bounds in the output's quantized
 * domain (min / s_out + zp_out, max / s_out + zp_out). cmin > cmax is
 * unverified. */
int npu_elementwise_clip_a8(void* stream, const void* in, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c1,
                            float c0, float cmin, float cmax);
int npu_elementwise_clip_a16(void* stream, const void* in, npu_dtype in_type,
                             void* out, npu_dtype out_type, int64_t n,
                             float c1, float c0, float cmin, float cmax);

/* ---- B. elementwise binary --------------------------------------------- */

/* elementwise_equal_a8 / _a16, elementwise_greater_a8 / _a16: compare in the
 * dequantized domain, NO requantization; out is int8 (I8 buffer), 0 or 1:
 *   fa = mac(a, scale_a, zp_a), fb = mac(b, scale_b, zp_b)  (fp32, one rounding)
 *   out = (fa == fb) / (fa > fb)
 * scale_x = s_x, zp_x = -zp_x * s_x. a_type / b_type: I8 / U8 (a8) or
 * I16 / U16 (a16); the two may differ in sign. Same shape, no broadcasting. */
int npu_elementwise_equal_a8(void* stream, const void* a, npu_dtype a_type,
                             const void* b, npu_dtype b_type, void* out,
                             int64_t n, float scale_a, float scale_b,
                             float zp_a, float zp_b);
int npu_elementwise_equal_a16(void* stream, const void* a, npu_dtype a_type,
                              const void* b, npu_dtype b_type, void* out,
                              int64_t n, float scale_a, float scale_b,
                              float zp_a, float zp_b);
int npu_elementwise_greater_a8(void* stream, const void* a, npu_dtype a_type,
                               const void* b, npu_dtype b_type, void* out,
                               int64_t n, float scale_a, float scale_b,
                               float zp_a, float zp_b);
int npu_elementwise_greater_a16(void* stream, const void* a, npu_dtype a_type,
                                const void* b, npu_dtype b_type, void* out,
                                int64_t n, float scale_a, float scale_b,
                                float zp_a, float zp_b);
/* elementwise_max_a8 / _a16: fa, fb as for equal; m = max(fa, fb);
 * out = sat(rne(fma(m, scale_o, zp_o))) with scale_o = 1 / s_out (fp32),
 * zp_o = zp_out. */
int npu_elementwise_max_a8(void* stream, const void* a, npu_dtype a_type,
                           const void* b, npu_dtype b_type, void* out,
                           npu_dtype out_type, int64_t n, float scale_a,
                           float scale_b, float scale_o, float zp_a, float zp_b,
                           float zp_o);
int npu_elementwise_max_a16(void* stream, const void* a, npu_dtype a_type,
                            const void* b, npu_dtype b_type, void* out,
                            npu_dtype out_type, int64_t n, float scale_a,
                            float scale_b, float scale_o, float zp_a,
                            float zp_b, float zp_o);

/* ---- C. where / broadcast min ------------------------------------------ */

/* elementwise_broadcasting3d_where_a8 / _a16: pure selection, NO
 * requantization. x and out are [H][W][C] of 1-byte (a8: int8 / uint8 / fp8)
 * or 2-byte (a16: int16 / uint16 / bf16) elements, copied bit for bit. cond
 * is int8 [cH][cW][cC], broadcast to [H][W][C] (each of cH / cW / cC is the
 * full extent or 1). y_bits is the scalar operand (its low 8 / 16 bits).
 *   scalar_position == 1 (Y is the scalar): out = cond != 0 ? x : y
 *   scalar_position == 0 (X is the scalar): out = cond == 0 ? x : y */
int npu_elementwise_broadcasting3d_where_a8(
    void* stream, const void* cond, const void* x, void* out, int64_t H,
    int64_t W, int64_t C, int64_t cH, int64_t cW, int64_t cC, uint32_t y_bits,
    int scalar_position);
int npu_elementwise_broadcasting3d_where_a16(
    void* stream, const void* cond, const void* x, void* out, int64_t H,
    int64_t W, int64_t C, int64_t cH, int64_t cW, int64_t cC, uint32_t y_bits,
    int scalar_position);
/* qdq_elementwise_broadcasting3d_where_a8 / _a16: the selection above (cond
 * int8 for a8, int16 for a16), then every selected value, including the
 * scalar y, is requantized: out = sat(rne(mac(sel, scale, offset))) with
 * scale = s_in / s_out, offset = zp_out - zp_in * scale (coeff words 0 and
 * 1). dtype (x, y and out): I8 / U8 (a8) or I16 / U16 (a16). */
int npu_qdq_elementwise_broadcasting3d_where_a8(
    void* stream, const void* cond, const void* x, npu_dtype dtype, void* out,
    int64_t H, int64_t W, int64_t C, int64_t cH, int64_t cW, int64_t cC,
    int32_t y, int scalar_position, float scale, float offset);
int npu_qdq_elementwise_broadcasting3d_where_a16(
    void* stream, const void* cond, const void* x, npu_dtype dtype, void* out,
    int64_t H, int64_t W, int64_t C, int64_t cH, int64_t cW, int64_t cC,
    int32_t y, int scalar_position, float scale, float offset);
/* qdq_elementwise_broadcasting3d_min_a8 / _a16: a is [H][W][C], b is
 * [bH][bW][bC] broadcast to it; fa = mac(a, scale_a, zp_a), fb = mac(b,
 * scale_b, zp_b); out = sat(rne(fma(min(fa, fb), scale_o, zp_o))).
 * Coefficients as for npu_elementwise_max_a8. dtype (a, b, out): I8 / U8 or
 * I16 / U16. */
int npu_qdq_elementwise_broadcasting3d_min_a8(
    void* stream, const void* a, const void* b, npu_dtype dtype, void* out,
    int64_t H, int64_t W, int64_t C, int64_t bH, int64_t bW, int64_t bC,
    float scale_a, float scale_b, float scale_o, float zp_a, float zp_b,
    float zp_o);
int npu_qdq_elementwise_broadcasting3d_min_a16(
    void* stream, const void* a, const void* b, npu_dtype dtype, void* out,
    int64_t H, int64_t W, int64_t C, int64_t bH, int64_t bW, int64_t bC,
    float scale_a, float scale_b, float scale_o, float zp_a, float zp_b,
    float zp_o);

/* ---- D. reductions ----------------------------------------------------- */

/* reduce_max_c64_a8 / _a16, reduce_min_c64_a8 / _a16: integer max / min of
 * in [H][W][C] over the axes in axis_flags (bit 0 = C, bit 1 = W, bit 2 = H;
 * 1..7), NO requantization. out is the reduced tensor in HWC order with the
 * reduced axes removed. dtype: I8 / U8 or I16 / U16 (in and out). The
 * kernels' keepdims only changes the physical output stride. */
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
/* qdq_reduce_{max,min}_c64_{a8,a16}: the reduction above (on the integers),
 * then out = sat(rne(mac(r, scale, offset) * 2^shift)) to dtype (the input
 * type; the kernel uses one sign flag for both). Coefficient buffer:
 * word 0 = offset (c0 = zp_out - scale * zp_in), word 1 = scale
 * (c1 = s_in / s_out), low 16 bits of word 2 = shift (int16; the kernel
 * passes it as uint16, which the hardware reads as a signed shift: -2 in
 * the regression gives 2^-2).
 * Logical channels only. On the NPU, qdq_reduce_max masks padded channels of
 * the last C64 block out of a C reduction, and qdq_reduce_min does not mask
 * them (its padding must hold neutral values). For W / H-only reductions with
 * padded channels, the max kernels apply that mask on every c_block_num-th
 * tap, which may hide valid channels; the regression has no such case
 * (unverified, not emulated). */
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

/* ---- E. pooling -------------------------------------------------------- */

/* qdq_maxpool_a8 / _a16: max over a kh x kw window with stride sh x sw, no
 * padding, of in [H][W][C]; out [Ho][Wo][C], Ho = (H - kh) / sh + 1,
 * Wo = (W - kw) / sw + 1; then out = sat(rne(mac(m, scale, offset) *
 * 2^shift)) to out_type. Coefficients as for npu_qdq_reduce_max_c64_a8
 * (word 0 offset, word 1 scale, word 2 shift). in_type / out_type: I8 / U8
 * (a8) or I16 / U16 (a16). */
int npu_qdq_maxpool_a8(void* stream, const void* in, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t H, int64_t W,
                       int64_t C, int kh, int kw, int sh, int sw, float offset,
                       float scale, int shift);
int npu_qdq_maxpool_a16(void* stream, const void* in, npu_dtype in_type,
                        void* out, npu_dtype out_type, int64_t H, int64_t W,
                        int64_t C, int kh, int kw, int sh, int sw, float offset,
                        float scale, int shift);
/* generic_maxpool_a8: the same window max, NO requantization (the wrapper's
 * weight buffer is unused). dtype: I8 / U8. */
int npu_generic_maxpool_a8(void* stream, const void* in, npu_dtype dtype,
                           void* out, int64_t H, int64_t W, int64_t C, int kh,
                           int kw, int sh, int sw);
/* qdq_globalmaxpool_a8 / _a16: max over H and W of in [H][W][C]; out [C]
 * requantized as npu_qdq_maxpool_a8. Not emulated: on the NPU (a8) the last
 * tap of a sub-volume whose H x W area is odd is compared as int8 even for
 * uint8 data (aie::max without the sign flag); the regression only has even
 * areas. */
int npu_qdq_globalmaxpool_a8(void* stream, const void* in, npu_dtype in_type,
                             void* out, npu_dtype out_type, int64_t H,
                             int64_t W, int64_t C, float offset, float scale,
                             int shift);
int npu_qdq_globalmaxpool_a16(void* stream, const void* in, npu_dtype in_type,
                              void* out, npu_dtype out_type, int64_t H,
                              int64_t W, int64_t C, float offset, float scale,
                              int shift);

/* ---- F. concat along C ------------------------------------------------- */

/* qdq_concat_a8 / _a16: out[r] = [a[r][0..ca), b[r][0..cb)] for r < rows,
 * then out = sat(rne(mac(v, scale, offset))) with scale = s_in / s_out
 * (coeff word 0, named c0 in the wrapper), offset = zp_out - zp_in * scale
 * (word 1, c1). Both inputs share one quantization. in_type / out_type:
 * I8 / U8 (a8) or I16 / U16 (a16). */
int npu_qdq_concat_a8(void* stream, const void* a, const void* b,
                      npu_dtype in_type, void* out, npu_dtype out_type,
                      int64_t rows, int64_t ca, int64_t cb, float scale,
                      float offset);
int npu_qdq_concat_a16(void* stream, const void* a, const void* b,
                       npu_dtype in_type, void* out, npu_dtype out_type,
                       int64_t rows, int64_t ca, int64_t cb, float scale,
                       float offset);
/* qdq_concat_multi_c_fusion_a8 / _a16: concat of n_in (1..6) tensors
 * ins[k] = [rows][channels[k]] along C, then one requantization as
 * npu_qdq_concat_a8 (scale = coeff word 0, offset = word 1). ins and
 * channels are host arrays of n_in entries (ins holds device pointers). */
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
/* qdq_concat_multi_c_fusion: as above, but each input k is requantized with
 * its own (scale[k], offset[k]) (coeff words 2k and 2k + 1: s_k / s_out and
 * zp_out - zp_k * s_k / s_out) before the concat. in_type / out_type: any of
 * I8 / U8 (8-bit build) or I16 / U16 (16-bit build), same width. scale and
 * offset are host arrays of n_in entries. */
int npu_qdq_concat_multi_c_fusion(void* stream, int n_in,
                                  const void* const* ins, npu_dtype in_type,
                                  void* out, npu_dtype out_type, int64_t rows,
                                  const int64_t* channels, const float* scale,
                                  const float* offset);

/* ---- G. gather / grid sample / resize ---------------------------------- */

/* qdq_gather_element_2din_2dindices_axis0: GatherElements on axis 0,
 *   out[i][k] = sat(rne(mac(in[idx[i][k]][k], scale, offset)))
 * in [M_in][N], idx int32 [M_out][N], out [M_out][N]. scale = s_in / s_out
 * (coeff word 0), offset = zp_out - scale * zp_in (word 1). in_type /
 * out_type: I8 / U8 / I16 / U16, any combination. An index outside
 * [0, M_in) (negative included: the kernel compares it as unsigned) leaves
 * that output element unwritten, as on the NPU. */
int npu_qdq_gather_element_2din_2dindices_axis0(
    void* stream, const void* in, npu_dtype in_type, const void* idx,
    void* out, npu_dtype out_type, int64_t M_in, int64_t N, int64_t M_out,
    float scale, float offset);

/* The coefficient buffer of the qdq_gridsample_nearest wrappers, as the
 * kernel reads it (GridSampleBf16QdqCoeffs): ofm_scale is the multiplier
 * 1 / s_out. */
typedef struct {
  int32_t grid_zp;
  float grid_scale;
  int32_t ifm_zp;
  float ifm_scale;
  int32_t ofm_zp;
  float ofm_scale;
} npu_gridsample_qdq;
/* qdq_gridsample_nearest_a8 / _a16 / _a8w8: GridSample (nearest) of
 * in [H][W][C] by grid [Ho][Wo][2] (x, y codes) into out [Ho][Wo][C]:
 *   g  = fp32(fp32(code - grid_zp) * grid_scale) + 1        (fp32 add)
 *   align_corners 0: ix = fma(gx, W * 0.5, -0.5)   (aie::mac, one rounding;
 *                    the regression does not tell it from two: unverified)
 *   align_corners 1: ix = fp32((W - 1) * gx) * 0.5
 *   rx = sat_int16(rne(ix)); padding_mode 1 (border) clamps to [0, W - 1];
 *   (same for y with H)
 *   v  = bf16_rne(mac(in[ry][rx][c], ifm_scale, fp32(-(float)ifm_zp *
 *        ifm_scale)))     if 0 <= rx < W and 0 <= ry < H, else 0
 *   out = sat(rne(fma(v, ofm_scale, (float)ofm_zp)))
 * The NPU streams the image in row bands and accumulates bands in bf16; one
 * band holds each pixel and the others add +-0, so the result is the same.
 * dtype (in and out): I8 / U8 (a8, a8w8) or I16 / U16 (a16). Grid codes:
 * U16 (a8, a16) or I8 / U8 (a8w8, grid_type). */
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
/* qdq_resize_nearest (A8 and A16 share this source): the source pixel of each
 * output comes from the host's index tables, src_y [Ho] and src_x [Wo]
 * (int32, device), as in the kernel's weight buffer:
 *   out[h][w][c] = in[src_y[h]][src_x[w]][c]
 * With requantize != 0: out = sat(trunc(mac(v, scale, offset))) - the wrapper
 * sets symmetric_floor, i.e. rounds toward zero; scale = coeff word 0
 * (s_in / s_out), offset = word 1 (zp_out - zp_in * scale). dtype (in and
 * out): I8 / U8 / I16 / U16. */
int npu_qdq_resize_nearest(void* stream, const void* in, npu_dtype dtype,
                           void* out, int64_t H, int64_t W, int64_t C,
                           int64_t Ho, int64_t Wo, const int32_t* src_y,
                           const int32_t* src_x, int requantize, float scale,
                           float offset);

/* ---- H. data movement with fused requantization ------------------------ */

/* qdq_<op>_requantize_fused: generic_<op> of npu_emu_movement.h (same
 * arguments, same logical tensors), then dqq<T, T> in place over the moved
 * elements:
 *   out = sat(rne(mac(v, c1, c0)))
 * with c1 = s_in / s_out (coeff word 0) and c0 = zp_out - c1 * zp_in
 * (word 1), both computed in double by the YAML and stored as fp32 - the
 * kernel reads the YAML's words, not ones the python model computes. Input
 * and output share dtype. dtype: I8 / U8 / I16 / U16, except
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
