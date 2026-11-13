/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 data-movement / selection wrappers
 * (movement group). Conventions as in npu_emu.h.
 *
 * Tensors are the logical (valid-region, densely packed, row-major) tensors of
 * the mllib python reference model of each wrapper; the NPU's physical layout
 * (C64 / C8 blocking, sub-volume tiling, garbage padding) is not emulated.
 * Pure data movement copies the bits of each element, so only the element size
 * of `t` matters (F32 / F16 / BF16 / FP8 / I8 / U8 / I16 / U16 / I32 / U32 /
 * I64 / U64); NaN payloads, -0 and fp8 NaN groups pass through unchanged.
 * Aligned bfp16 data (whole 9-byte blocks) can be moved by passing NPU_I8 /
 * NPU_U8 and counting bytes, as the NPU's 8-bit kernels do.
 *
 * Shapes, permutations and pads are host arrays; tensors are device pointers.
 * Gather indices are read as the kernels read them: converted to uint32
 * (signed index types sign-extended, so a negative index becomes a huge
 * value). An index outside the gathered axis leaves its output elements
 * unwritten, as on the NPU (which keeps whatever its output buffer held).
 */
#ifndef NPU_EMU_MOVEMENT_H_
#define NPU_EMU_MOVEMENT_H_

#include <stdint.h>

#include "npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- A. concat / slice / copy ------------------------------------------- */

/* generic_concat: out[r] = in1[r, s1 : s1 + n1] ++ in2[r, s2 : s2 + n2] for
 * r < rows; in1 is [rows][c1], in2 is [rows][c2], out is [rows][n1 + n2]. */
int npu_generic_concat(void* stream, const void* in1, int64_t c1, int64_t s1,
                       int64_t n1, const void* in2, int64_t c2, int64_t s2,
                       int64_t n2, void* out, npu_dtype t, int64_t rows);
/* generic_concat_16b_inner: out[r] = in1[r] ++ in2[r]; in1 [rows][c1], in2
 * [rows][c2]. (The NPU also writes pad_value up to the next row_alignment
 * multiple; that is physical padding, not emulated.) */
int npu_generic_concat_16b_inner(void* stream, const void* in1, int64_t c1,
                                 const void* in2, int64_t c2, void* out,
                                 npu_dtype t, int64_t rows);
/* generic_concat_16b_c6463: out[r] = in1[r][0..62] ++ in2[r][0]; in1 is
 * [rows][63], in2 [rows][1], out [rows][64]. */
int npu_generic_concat_16b_c6463(void* stream, const void* in1, const void* in2,
                                 void* out, npu_dtype t, int64_t rows);
/* generic_concat_16b_multi: out[r] = ins[0][r] ++ ... ++ ins[k-1][r];
 * ins[j] is [rows][widths[j]] (device pointers in a host array), 2 <= k <= 6. */
int npu_generic_concat_16b_multi(void* stream, const void* const* ins,
                                 const int64_t* widths, int k, void* out,
                                 npu_dtype t, int64_t rows);
/* generic_concat_multi_c_fusion: same semantics as generic_concat_16b_multi. */
int npu_generic_concat_multi_c_fusion(void* stream, const void* const* ins,
                                      const int64_t* widths, int k, void* out,
                                      npu_dtype t, int64_t rows);
/* concat_unaligned_bfp16: per row, the bfp16 concatenation of the first n1
 * values of in1 and the first n2 values of in2, re-blocked from the row start.
 * in1 rows are ceil(n1/8) blocks, in2 rows ceil(n2/8) blocks, out rows
 * ceil((n1+n2)/8) blocks (9 bytes each). As the kernel does: the first
 * 128*floor((n1-1)/128) values of in1, plus 64 more if the remainder R =
 * n1 - that is >= 64, are copied block for block (not re-encoded); every later
 * block is re-encoded (half to even) from bf16 values decoded from the inputs
 * (decode rounds to bf16 half to even, which is exact unless the value is
 * below the bf16 normal range: unverified there), with positions past n1 + n2
 * set to 0. Re-encoding can round when the 8 values of a block come from
 * different source blocks. */
int npu_concat_unaligned_bfp16(void* stream, const void* in1, int64_t n1,
                               const void* in2, int64_t n2, void* out,
                               int64_t rows);
/* generic_slice_innermost: out[r][j] = in[r][start + j*stride] for
 * start + j*stride < end; in is [rows][c]. */
int npu_generic_slice_innermost(void* stream, const void* in, void* out,
                                npu_dtype t, int64_t rows, int64_t c,
                                int64_t start, int64_t end, int64_t stride);
/* generic_slice_16b_inner_range: out[r] = in[r][start:end]; in [rows][c]. */
int npu_generic_slice_16b_inner_range(void* stream, const void* in, void* out,
                                      npu_dtype t, int64_t rows, int64_t c,
                                      int64_t start, int64_t end);
/* generic_slice_16b_inner: out0[r] = in[r][0:split], out1[r] = in[r][split:c]. */
int npu_generic_slice_16b_inner(void* stream, const void* in, void* out0,
                                void* out1, npu_dtype t, int64_t rows,
                                int64_t c, int64_t split);
/* generic_slice_16b_c6463: in [rows][64]; out0 [rows][63] = in[r][0:63],
 * out1 [rows][1] = in[r][63]. */
int npu_generic_slice_16b_c6463(void* stream, const void* in, void* out0,
                                void* out1, npu_dtype t, int64_t rows);
/* generic_mem_copy, generic_reshape_innermost, generic_pad_innermost,
 * generic_unpad_innermost: a copy of n elements. The last three only re-block
 * the innermost axis of the NPU's padded layout (rows of ifm_valid_dim_3 ->
 * rows of ofm_valid_dim_3 over the same flat element sequence), which is the
 * identity on the logical tensor. */
int npu_generic_mem_copy(void* stream, const void* in, void* out, npu_dtype t,
                         int64_t n);
int npu_generic_reshape_innermost(void* stream, const void* in, void* out,
                                  npu_dtype t, int64_t n);
int npu_generic_pad_innermost(void* stream, const void* in, void* out,
                              npu_dtype t, int64_t n);
int npu_generic_unpad_innermost(void* stream, const void* in, void* out,
                                npu_dtype t, int64_t n);
/* generic_expand_innermost: out[r][j] = in[r][j % cin], j < cout (torch.tile
 * along the innermost axis; cin == 1 is a broadcast). cout % cin == 0. */
int npu_generic_expand_innermost(void* stream, const void* in, void* out,
                                 npu_dtype t, int64_t rows, int64_t cin,
                                 int64_t cout);

/* ---- B. pad ------------------------------------------------------------- */

/* Pad values are the raw bits of one element of type t (e.g. 0x4008 for
 * bf16 2.125), as the NPU receives them in its RTP. */

/* generic_inplace_pad2d_c64: [H][W][C] -> [H+hs+he][W+ws+we][C], constant. */
int npu_generic_inplace_pad2d_c64(void* stream, const void* in, void* out,
                                  npu_dtype t, int64_t H, int64_t W, int64_t C,
                                  int64_t hs, int64_t he, int64_t ws,
                                  int64_t we, uint64_t pad_bits);
/* generic_inplace_pad3d: [H][W][C] padded by pads = {hs, he, ws, we, cs, ce};
 * mode 0 constant (pad_bits), 1 reflect (numpy / ONNX "reflect"), 2 edge. */
int npu_generic_inplace_pad3d(void* stream, const void* in, void* out,
                              npu_dtype t, int64_t H, int64_t W, int64_t C,
                              const int64_t* pads, int mode, uint64_t pad_bits);

/* ---- C. transpose / space <-> depth -------------------------------------- */

/* generic_transpose_unaligned / generic_transpose4d_aligned: 4D transpose,
 * out = np.transpose(in.reshape(shape), perm) (ONNX perm). */
int npu_generic_transpose_unaligned(void* stream, const void* in, void* out,
                                    npu_dtype t, const int64_t* shape,
                                    const int* perm);
int npu_generic_transpose4d_aligned(void* stream, const void* in, void* out,
                                    npu_dtype t, const int64_t* shape,
                                    const int* perm);
/* generic_transpose4d_0132_RCR32: [d0][d1][d2] -> [d0][d2][d1]. */
int npu_generic_transpose4d_0132_RCR32(void* stream, const void* in, void* out,
                                       npu_dtype t, int64_t d0, int64_t d1,
                                       int64_t d2);
/* generic_transpose_k8n8: [K][N] -> [N][K]. (The NPU works on 8x8 tiles.) */
int npu_generic_transpose_k8n8(void* stream, const void* in, void* out,
                               npu_dtype t, int64_t K, int64_t N);
/* transpose3d_mmac: 3D transpose, out = np.transpose(in, perm), t 8 or 16 bit
 * integer. The kernel moves the data through integer identity-matrix
 * multiplies, which are exact. */
int npu_transpose3d_mmac(void* stream, const void* in, void* out, npu_dtype t,
                         const int64_t* shape, const int* perm);
/* transpose_mmac_a8_n64k64 (I8 / U8) / transpose_mmac_a16_n32k64 (I16 / U16):
 * [K][N] -> [N][K] through exact integer identity-matrix multiplies. (For a16
 * the NPU stores the low and high bytes of each 64-value K block as separate
 * planes; that is the physical layout and not emulated.) */
int npu_transpose_mmac_a8_n64k64(void* stream, const void* in, void* out,
                                 int64_t K, int64_t N);
int npu_transpose_mmac_a16_n32k64(void* stream, const void* in, void* out,
                                  int64_t K, int64_t N);
/* transpose_planar3_c8: u8 in [batch][3][P] (3 planes) -> out [batch][P][8]:
 * out[b][p][c] = in[b][c][p] for c < 3, pad_value for c = 3..7. */
int npu_transpose_planar3_c8(void* stream, const void* in, void* out,
                             int64_t batch, int64_t P, uint8_t pad_value);
/* generic_depth_to_space: NCHW [N][C][H][W] -> [N][C/b^2][H*b][W*b]; mode 0
 * DCR: in channel (bh*b + bw)*C/b^2 + c; mode 1 CRD: c*b^2 + bh*b + bw. */
int npu_generic_depth_to_space(void* stream, const void* in, void* out,
                               npu_dtype t, int64_t N, int64_t C, int64_t H,
                               int64_t W, int64_t b, int mode);
/* generic_depth_to_space_wc64: as generic_depth_to_space in NHWC,
 * [N][H][W][C] -> [N][H*b][W*b][C/b^2]. */
int npu_generic_depth_to_space_wc64(void* stream, const void* in, void* out,
                                    npu_dtype t, int64_t N, int64_t H,
                                    int64_t W, int64_t C, int64_t b, int mode);
/* generic_space_to_depth / generic_space_to_depth_c4bs4: HWC [H][W][C] ->
 * [H/b][W/b][C*b^2] with out channel c*b^2 + bh*b + bw (the "CRD" order,
 * as torch.pixel_unshuffle, not ONNX SpaceToDepth's DCR order). */
int npu_generic_space_to_depth(void* stream, const void* in, void* out,
                               npu_dtype t, int64_t H, int64_t W, int64_t C,
                               int64_t b);
int npu_generic_space_to_depth_c4bs4(void* stream, const void* in, void* out,
                                     npu_dtype t, int64_t H, int64_t W,
                                     int64_t C, int64_t b);

/* ---- D. gather / resize ------------------------------------------------ */

/* generic_gather_3din_1dindices: data [d0][d1][d2]; out = data gathered along
 * `axis` (0..2) with idx[0..n_idx-1] (idx_type U32 / I32 / U16 / I16). */
int npu_generic_gather_3din_1dindices(void* stream, const void* data,
                                      npu_dtype t, int64_t d0, int64_t d1,
                                      int64_t d2, int axis, const void* idx,
                                      npu_dtype idx_type, int64_t n_idx,
                                      void* out);
/* generic_gather_element_2din_2dindices_axis0: ONNX GatherElements axis 0,
 * data [M][N], idx [Mi][N], out[i][j] = data[idx[i][j]][j]. */
int npu_generic_gather_element_2din_2dindices_axis0(
    void* stream, const void* data, npu_dtype t, int64_t M, int64_t N,
    const void* idx, npu_dtype idx_type, int64_t Mi, void* out);
/* generic_gather_element_2din_2dindices_innermost: GatherElements axis 1,
 * data [M][N], idx [M][Ni], out[i][j] = data[i][idx[i][j]]. Negative indices
 * are not normalized (the kernel reads data[i] + idx). */
int npu_generic_gather_element_2din_2dindices_innermost(
    void* stream, const void* data, npu_dtype t, int64_t M, int64_t N,
    const void* idx, npu_dtype idx_type, int64_t Ni, void* out);
/* gather_nd_4x3_adf: ONNX GatherND, batch_dims 0. data has `rank` dims
 * `shape`; idx is [n_tuples][depth] (U16 / U32). Each tuple selects the row
 * r = sum_j uint32(idx[j]) * prod(shape[j+1 .. depth-1]) (uint32 arithmetic)
 * of data viewed as [prod(shape[0..depth-1])][row_len]; out is
 * [n_tuples][row_len]. Like the kernel, only r is range-checked, so a tuple
 * with a component past its own axis still selects row r. */
int npu_gather_nd_4x3_adf(void* stream, const void* data, npu_dtype t,
                          const int64_t* shape, int rank, const void* idx,
                          npu_dtype idx_type, int64_t n_tuples, int depth,
                          void* out);
/* resize_nearest_bf16 / resize_nearest_4x3 (t F16 / BF16 / FP8) and
 * resize_nearest_bfp16: HWC [Hi][Wi][C] -> [Ho][Wo][C],
 * out[h][w] = in[idx_h[h]][idx_w[w]]. idx_h / idx_w (device int32) are the
 * source index tables the host computes into the kernel's weights from the
 * ONNX coordinate_transformation_mode / nearest_mode; the kernel only gathers.
 * For bfp16, C counts values (C % 8 == 0) and blocks of 8 channels are copied
 * unchanged. */
int npu_resize_nearest_bf16(void* stream, const void* in, void* out,
                            npu_dtype t, int64_t Hi, int64_t Wi, int64_t C,
                            int64_t Ho, int64_t Wo, const int32_t* idx_h,
                            const int32_t* idx_w);
int npu_resize_nearest_4x3(void* stream, const void* in, void* out,
                           npu_dtype t, int64_t Hi, int64_t Wi, int64_t C,
                           int64_t Ho, int64_t Wo, const int32_t* idx_h,
                           const int32_t* idx_w);
int npu_resize_nearest_bfp16(void* stream, const void* in, void* out,
                             int64_t Hi, int64_t Wi, int64_t C, int64_t Ho,
                             int64_t Wo, const int32_t* idx_h,
                             const int32_t* idx_w);

/* ---- E. not ------------------------------------------------------------ */

/* elementwise_not: out = (in == 0) ? 1 : 0 (ONNX Not on 0/1 bytes; any
 * non-zero input gives 0). in_type / out_type I8 / U8. A compare, not a pure
 * move. */
int npu_elementwise_not(void* stream, const void* in, npu_dtype in_type,
                        void* out, npu_dtype out_type, int64_t n);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_MOVEMENT_H_
