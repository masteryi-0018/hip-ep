/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Data movement: concat, slice, copy, pad, transpose, gather, nearest resize.
 * Tensors are dense row-major; elements are copied bit for bit, so only the
 * element size of `dtype` matters. `dtype` types the input and the output.
 * Unless an entry says otherwise, dtype is one of U8 / I8 / FP8 / U16 / I16 /
 * BF16 / F16. Shapes, perms and pads are host arrays. Every pointer is
 * required. A call with an empty output (a zero row / element / pixel count)
 * returns 0 without launching; negative sizes return -1.
 */
#ifndef NPU_EMU_MOVEMENT_H_
#define NPU_EMU_MOVEMENT_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* concat / slice / copy */

/* out[r] = in1[r, s1 : s1 + n1] ++ in2[r, s2 : s2 + n2]; in1 [rows][c1],
 * in2 [rows][c2]. dtype may also be BFP16: sizes count values, c1, c2, n1, n2
 * are multiples of 8 and s1 = s2 = 0; 8-value blocks are copied unchanged. */
int npu_generic_concat(void* stream, const void* in1, npu_dtype dtype,
                       const void* in2, void* out, int64_t rows, int64_t c1,
                       int64_t s1, int64_t n1, int64_t c2, int64_t s2,
                       int64_t n2);
/* out[r] = in1[r] ++ in2[r]; in1 [rows][c1], in2 [rows][c2]. */
int npu_generic_concat_16b_inner(void* stream, const void* in1, npu_dtype dtype,
                                 const void* in2, void* out, int64_t rows,
                                 int64_t c1, int64_t c2);
/* out[r] = in1[r][0..62] ++ in2[r][0]; in1 [rows][63], in2 [rows][1]. */
int npu_generic_concat_16b_c6463(void* stream, const void* in1, npu_dtype dtype,
                                 const void* in2, void* out, int64_t rows);
/* out[r] = ins[0][r] ++ ... ++ ins[n_in-1][r]; ins[k] is [rows][channels[k]]
 * (device pointers in a host array), 2 <= n_in <= 6. */
int npu_generic_concat_16b_multi(void* stream, const void* const* ins,
                                 npu_dtype dtype, void* out, int64_t rows,
                                 int n_in, const int64_t* channels);
/* Same as npu_generic_concat_16b_multi. */
int npu_generic_concat_multi_c_fusion(void* stream, const void* const* ins,
                                      npu_dtype dtype, void* out, int64_t rows,
                                      int n_in, const int64_t* channels);
/* Per-row bfp16 concat of the first n1 values of in1 and the first n2 of in2;
 * rows are ceil(n/8) 9-byte blocks. The first 128*floor((n1-1)/128) values of
 * in1 (+64 if at least 64 remain) are copied block for block; later blocks are
 * decoded to bf16 and re-encoded (half to even), positions past n1 + n2 set
 * to 0. The decode aligns each 64-value group of an input row (8 blocks from
 * the row start) to its largest block exponent E: values rounded half to even
 * to multiples of 2^(E - 150), below 2^-126 +0, all NaN if a block exponent is
 * 255. */
int npu_concat_unaligned_bfp16(void* stream, const void* in1, const void* in2,
                               void* out, int64_t rows, int64_t n1,
                               int64_t n2);
/* out[r][j] = in[r][start + j*stride] for start + j*stride < end; in [rows][c],
 * stride 1 or 2. */
int npu_generic_slice_innermost(void* stream, const void* in, npu_dtype dtype,
                                void* out, int64_t rows, int64_t c,
                                int64_t start, int64_t end, int64_t stride);
/* out[r] = in[r][start:end]; in [rows][c]; dtype I8 / I16. */
int npu_generic_slice_16b_inner_range(void* stream, const void* in,
                                      npu_dtype dtype, void* out, int64_t rows,
                                      int64_t c, int64_t start, int64_t end);
/* out0[r] = in[r][0:split], out1[r] = in[r][split:c];
 * dtype I8 / U8 / I16 / U16. */
int npu_generic_slice_16b_inner(void* stream, const void* in, npu_dtype dtype,
                                void* out0, void* out1, int64_t rows,
                                int64_t c, int64_t split);
/* in [rows][64]; out0[r] = in[r][0:63], out1[r] = in[r][63];
 * dtype I16 / U16 / BF16 / F16. */
int npu_generic_slice_16b_c6463(void* stream, const void* in, npu_dtype dtype,
                                void* out0, void* out1, int64_t rows);
/* Copy of n elements (reshape / innermost pad / unpad do not move data). */
int npu_generic_mem_copy(void* stream, const void* in, npu_dtype dtype,
                         void* out, int64_t n);
int npu_generic_reshape_innermost(void* stream, const void* in, npu_dtype dtype,
                                  void* out, int64_t n);
int npu_generic_pad_innermost(void* stream, const void* in, npu_dtype dtype,
                              void* out, int64_t n);
int npu_generic_unpad_innermost(void* stream, const void* in, npu_dtype dtype,
                                void* out, int64_t n);
/* out[r][j] = in[r][j % cin], j < cout; cout % cin == 0;
 * dtype I8 / BF16 / F16. */
int npu_generic_expand_innermost(void* stream, const void* in, npu_dtype dtype,
                                 void* out, int64_t rows, int64_t cin,
                                 int64_t cout);

/* pad: dtype is not FP8; pad_value holds the bits of one element of type
 * dtype in its low 8 / 16 bits. An input extent may be 0 only when its pads
 * are 0. */

/* [H][W][C] -> [H+hs+he][W+ws+we][C], constant. */
int npu_generic_inplace_pad2d_c64(void* stream, const void* in, npu_dtype dtype,
                                  void* out, int64_t H, int64_t W, int64_t C,
                                  int64_t hs, int64_t he, int64_t ws,
                                  int64_t we, int32_t pad_value);
/* [H][W][C] padded by pads = {hs, he, ws, we, cs, ce}; mode 0 constant,
 * 1 reflect, 2 edge. Reflect / edge need every padded dimension larger than
 * twice its larger pad. */
int npu_generic_inplace_pad3d(void* stream, const void* in, npu_dtype dtype,
                              void* out, int64_t H, int64_t W, int64_t C,
                              const int64_t* pads, int mode,
                              int32_t pad_value);

/* transpose / space <-> depth */

/* 4D transpose: out = np.transpose(in.reshape(shape), perm). */
int npu_generic_transpose_unaligned(void* stream, const void* in,
                                    npu_dtype dtype, void* out,
                                    const int64_t* shape, const int* perm);
int npu_generic_transpose4d_aligned(void* stream, const void* in,
                                    npu_dtype dtype, void* out,
                                    const int64_t* shape, const int* perm);
/* [d0][d1][d2] -> [d0][d2][d1]; dtype BF16 / U16. */
int npu_generic_transpose4d_0132_RCR32(void* stream, const void* in,
                                       npu_dtype dtype, void* out, int64_t d0,
                                       int64_t d1, int64_t d2);
/* [K][N] -> [N][K]. */
int npu_generic_transpose_k8n8(void* stream, const void* in, npu_dtype dtype,
                               void* out, int64_t K, int64_t N);
/* 3D transpose: out = np.transpose(in, perm), perm one of (0, 2, 1),
 * (2, 0, 1), (1, 2, 0), (2, 1, 0); dtype I8 / U8 / I16 / U16. */
int npu_transpose3d_mmac(void* stream, const void* in, npu_dtype dtype,
                         void* out, const int64_t* shape, const int* perm);
/* [K][N] -> [N][K]; a8 is I8 / U8, a16 is I16 / U16. */
int npu_transpose_mmac_a8_n64k64(void* stream, const void* in, void* out,
                                 int64_t K, int64_t N);
/* See npu_transpose_mmac_a8_n64k64. */
int npu_transpose_mmac_a16_n32k64(void* stream, const void* in, void* out,
                                  int64_t K, int64_t N);
/* u8 in [batch][3][P] -> out [batch][P][8]: out[b][p][c] = in[b][c][p] for
 * c < 3, pad_value for c = 3..7. */
int npu_transpose_planar3_c8(void* stream, const void* in, void* out,
                             int64_t batch, int64_t P, uint8_t pad_value);
/* NCHW [N][C][H][W] -> [N][C/b^2][H*b][W*b]; mode 0 DCR: in channel
 * (bh*b + bw)*C/b^2 + c; mode 1 CRD: c*b^2 + bh*b + bw. */
int npu_generic_depth_to_space(void* stream, const void* in, npu_dtype dtype,
                               void* out, int64_t N, int64_t C, int64_t H,
                               int64_t W, int64_t b, int mode);
/* Same in NHWC: [N][H][W][C] -> [N][H*b][W*b][C/b^2]; b 2 or 4. */
int npu_generic_depth_to_space_wc64(void* stream, const void* in,
                                    npu_dtype dtype, void* out, int64_t N,
                                    int64_t H, int64_t W, int64_t C, int64_t b,
                                    int mode);
/* HWC [H][W][C] -> [H/b][W/b][C*b^2], out channel c*b^2 + bh*b + bw (CRD). */
int npu_generic_space_to_depth(void* stream, const void* in, npu_dtype dtype,
                               void* out, int64_t H, int64_t W, int64_t C,
                               int64_t b);
/* The same with b = 4. */
int npu_generic_space_to_depth_c4bs4(void* stream, const void* in,
                                     npu_dtype dtype, void* out, int64_t H,
                                     int64_t W, int64_t C, int64_t b);

/* gather / resize: an out-of-range index leaves its output elements
 * unwritten. data_type types data and out. */

/* data [d0][d1][d2] gathered along `axis` (0..2) with the uint32 indices
 * idx[0..n_idx-1]. */
int npu_generic_gather_3din_1dindices(void* stream, const void* data,
                                      npu_dtype data_type, const void* idx,
                                      void* out, int64_t d0, int64_t d1,
                                      int64_t d2, int64_t n_idx, int axis);
/* GatherElements axis 0: data [M][N], idx [Mi][N],
 * out[i][j] = data[idx[i][j]][j]. data_type is not FP8; idx_type I16 / U16 /
 * I32, signed indices are sign-extended to uint32 (a negative one is out of
 * range). */
int npu_generic_gather_element_2din_2dindices_axis0(
    void* stream, const void* data, npu_dtype data_type, const void* idx,
    npu_dtype idx_type, void* out, int64_t M, int64_t N, int64_t Mi);
/* GatherElements axis 1: data [M][N], idx [M][Ni],
 * out[i][j] = data[i][idx[i][j]]; types as above. */
int npu_generic_gather_element_2din_2dindices_innermost(
    void* stream, const void* data, npu_dtype data_type, const void* idx,
    npu_dtype idx_type, void* out, int64_t M, int64_t N, int64_t Ni);
/* HWC [Hi][Wi][C] -> [Ho][Wo][C], out[h][w] = in[idx_h[h]][idx_w[w]]; idx_h
 * [Ho] / idx_w [Wo] are device uint16 source-index tables. dtype is BF16 /
 * F16 / FP8. Hi (Wi) may be 0 only when Ho (Wo) is 0. For bfp16, C counts
 * values (C % 8 == 0) and blocks of 8 channels are copied unchanged. */
int npu_resize_nearest_bf16(void* stream, const void* in, npu_dtype dtype,
                            void* out, int64_t Hi, int64_t Wi, int64_t C,
                            int64_t Ho, int64_t Wo, const void* idx_h,
                            const void* idx_w);
int npu_resize_nearest_bfp16(void* stream, const void* in, void* out,
                             int64_t Hi, int64_t Wi, int64_t C, int64_t Ho,
                             int64_t Wo, const void* idx_h,
                             const void* idx_w);

/* not */

/* out = (in == 0) ? 1 : 0; dtype (input and output) I8 / U8. */
int npu_elementwise_not(void* stream, const void* in, npu_dtype dtype,
                        void* out, int64_t n);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_MOVEMENT_H_
