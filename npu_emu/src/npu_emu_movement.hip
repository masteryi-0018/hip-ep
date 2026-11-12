/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for include/npu_emu_movement.h. Most wrappers are
 * one strided copy (copy_view_kernel) over a view of up to kMaxDims axes;
 * gathers, resize and the bfp16 concat have their own kernels.
 */
#include <hip/hip_runtime.h>

#include "npu_emu.h"
#include "npu_emu_device.h"
#include "npu_emu_launch.h"

using namespace npu_emu;

namespace {

constexpr int kMaxDims = 8;

// How an axis maps an out-of-range input coordinate (numpy.pad modes; kWrap is
// torch.tile).
enum Edge : int8_t { kDirect = 0, kConstant, kReflect, kEdge, kWrap };

// Output element i iterates `dim` (row-major). Along axis k the input
// coordinate is c = x * istep + ioff, mapped by edge[k] against ilen; offsets
// are in elements.
struct View {
  int nd;
  int64_t dim[kMaxDims];
  int64_t ostride[kMaxDims];
  int64_t istride[kMaxDims];
  int64_t istep[kMaxDims];
  int64_t ioff[kMaxDims];
  int64_t ilen[kMaxDims];
  int8_t edge[kMaxDims];
  int64_t obase, ibase;
};

// A view with dense row-major output strides over `dim`; input strides are left
// 0 for the caller to set.
View dense_view(int nd, const int64_t* dim) {
  View v{};
  v.nd = nd;
  int64_t s = 1;
  for (int k = nd - 1; k >= 0; --k) {
    v.dim[k] = dim[k];
    v.ostride[k] = s;
    v.istep[k] = 1;
    v.ilen[k] = dim[k];
    v.edge[k] = kDirect;
    s *= dim[k];
  }
  return v;
}

// Dense row-major strides of `shape`.
void strides_of(int nd, const int64_t* shape, int64_t* st) {
  int64_t s = 1;
  for (int k = nd - 1; k >= 0; --k) {
    st[k] = s;
    s *= shape[k];
  }
}

__device__ inline void copy_bytes(uint8_t* d, const uint8_t* s, int n) {
  for (int b = 0; b < n; ++b) d[b] = s[b];
}

__global__ void copy_view_kernel(int64_t n, const uint8_t* in, uint8_t* out,
                                 int es, View v, uint64_t fill) {
  NPU_TID;
  int64_t r = i, ip = v.ibase, op = v.obase;
  bool pad = false;
  for (int k = v.nd - 1; k >= 0; --k) {
    const int64_t x = r % v.dim[k];
    r /= v.dim[k];
    op += x * v.ostride[k];
    int64_t c = x * v.istep[k] + v.ioff[k];
    const int64_t L = v.ilen[k];
    switch (v.edge[k]) {
      case kConstant:
        pad |= c < 0 || c >= L;
        break;
      case kReflect:
        if (L == 1) {
          c = 0;
        } else {  // mirror without repeating the edge: period 2L - 2
          const int64_t p = 2 * L - 2;
          c = ((c % p) + p) % p;
          if (c >= L) c = p - c;
        }
        break;
      case kEdge:
        c = c < 0 ? 0 : (c >= L ? L - 1 : c);
        break;
      case kWrap:
        c = ((c % L) + L) % L;
        break;
      default:
        break;
    }
    ip += c * v.istride[k];
  }
  uint8_t* o = out + op * es;
  if (pad) {
    for (int b = 0; b < es; ++b) o[b] = uint8_t(fill >> (8 * b));
  } else {
    copy_bytes(o, in + ip * es, es);
  }
}

int copy_view(void* stream, const void* in, void* out, int es, const View& v,
              uint64_t fill = 0) {
  int64_t n = 1;
  for (int k = 0; k < v.nd; ++k) n *= v.dim[k];
  return launch(stream, n, copy_view_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), es, v, fill);
}

// Element size of a data type moved bit for bit, or 0 if unsupported.
int elem_size(npu_dtype t) {
  switch (t) {
    case NPU_FP8:
    case NPU_I8:
    case NPU_U8:
      return 1;
    case NPU_F16:
    case NPU_BF16:
    case NPU_I16:
    case NPU_U16:
      return 2;
    case NPU_F32:
    case NPU_I32:
    case NPU_U32:
      return 4;
    case NPU_I64:
    case NPU_U64:
      return 8;
    default:
      return 0;
  }
}

bool is_index_type(npu_dtype t) {
  return t == NPU_U16 || t == NPU_I16 || t == NPU_U32 || t == NPU_I32;
}

// An index as the kernels read it: (uint32_t)(index_t)x, sign-extending signed
// index types.
__device__ inline uint32_t load_index(const void* p, npu_dtype t, int64_t i) {
  switch (t) {
    case NPU_U16:
      return static_cast<const uint16_t*>(p)[i];
    case NPU_I16:
      return uint32_t(int32_t(static_cast<const int16_t*>(p)[i]));
    default:  // NPU_U32 / NPU_I32
      return static_cast<const uint32_t*>(p)[i];
  }
}

// out row (r, j) of a [rows][width] block placed at column `col` of an output
// with `ocols` columns, from input [rows][icols] starting at column `start`.
View rows_view(int64_t rows, int64_t width, int64_t icols, int64_t start,
               int64_t ocols, int64_t col) {
  const int64_t d[2] = {rows, width};
  View v = dense_view(2, d);
  v.ostride[0] = ocols;
  v.istride[0] = icols;
  v.istride[1] = 1;
  v.ioff[1] = start;
  v.obase = col;
  return v;
}

int concat_rows(void* stream, const void* const* ins, const int64_t* icols,
                const int64_t* starts, const int64_t* widths, int k, void* out,
                npu_dtype t, int64_t rows) {
  const int es = elem_size(t);
  if (!es || k < 1 || rows < 0) return -1;
  int64_t ocols = 0;
  for (int j = 0; j < k; ++j) {
    if (widths[j] < 0 || starts[j] < 0 || starts[j] + widths[j] > icols[j]) return -1;
    ocols += widths[j];
  }
  int64_t col = 0;
  for (int j = 0; j < k; ++j) {
    const int rc = copy_view(stream, ins[j], out, es,
                             rows_view(rows, widths[j], icols[j], starts[j], ocols, col));
    if (rc) return rc;
    col += widths[j];
  }
  return 0;
}

int multi_concat(void* stream, const void* const* ins, const int64_t* widths,
                 int k, void* out, npu_dtype t, int64_t rows) {
  if (k < 2 || k > 6) return -1;
  int64_t zero[6] = {0, 0, 0, 0, 0, 0};
  return concat_rows(stream, ins, widths, zero, widths, k, out, t, rows);
}

int copy_n(void* stream, const void* in, void* out, npu_dtype t, int64_t n) {
  const int es = elem_size(t);
  if (!es || n < 0) return -1;
  View v = dense_view(1, &n);
  v.istride[0] = 1;
  return copy_view(stream, in, out, es, v);
}

// out = transpose(in.reshape(shape), perm) for nd <= kMaxDims.
int transpose_nd(void* stream, const void* in, void* out, int es, int nd,
                 const int64_t* shape, const int* perm) {
  if (!es || nd < 1 || nd > kMaxDims) return -1;
  int64_t ist[kMaxDims], od[kMaxDims];
  bool seen[kMaxDims] = {};
  for (int k = 0; k < nd; ++k) {
    if (perm[k] < 0 || perm[k] >= nd || seen[perm[k]] || shape[k] < 0) return -1;
    seen[perm[k]] = true;
  }
  strides_of(nd, shape, ist);
  for (int k = 0; k < nd; ++k) od[k] = shape[perm[k]];
  View v = dense_view(nd, od);
  for (int k = 0; k < nd; ++k) v.istride[k] = ist[perm[k]];
  return copy_view(stream, in, out, es, v);
}

// ---- bfp16 concat ------------------------------------------------------------

// Value p of a row of bfp16 blocks, decoded and rounded to bf16 as the kernel
// does (mul by an identity matrix into accfloat, then to_vector<bfloat16>).
__device__ inline float bfp16_value_bf16(const uint8_t* row, int64_t p) {
  return bf16_to_f32(f32_to_bf16(decode_bfp16(row + 9 * (p / 8), int(p % 8)), kEven));
}

// One thread per output block (row-major over rows x out blocks).
__global__ void concat_bfp16_kernel(int64_t n, const uint8_t* in1, int64_t n1,
                                    const uint8_t* in2, int64_t n2, uint8_t* out,
                                    int64_t ob, int64_t raw_blocks) {
  NPU_TID;
  const int64_t r = i / ob, k = i % ob;
  const int64_t b1 = (n1 + 7) / 8, b2 = (n2 + 7) / 8;
  uint8_t* o = out + 9 * i;
  if (k < raw_blocks) {
    copy_bytes(o, in1 + 9 * (r * b1 + k), 9);
    return;
  }
  float v[8];
  for (int j = 0; j < 8; ++j) {
    const int64_t p = 8 * k + j;
    if (p < n1)
      v[j] = bfp16_value_bf16(in1 + 9 * r * b1, p);
    else if (p < n1 + n2)
      v[j] = bfp16_value_bf16(in2 + 9 * r * b2, p - n1);
    else
      v[j] = 0.0f;  // the kernel masks positions past the tail to zero
  }
  encode_bfp16(v, kEven, o);
}

// ---- gathers -----------------------------------------------------------------

// data [outer][len][inner] gathered along the middle axis: out [outer][n_idx][inner].
__global__ void gather_axis_kernel(int64_t n, const uint8_t* data, int es,
                                   int64_t len, int64_t inner, const void* idx,
                                   npu_dtype it, int64_t n_idx, uint8_t* out) {
  NPU_TID;
  const int64_t e = i % inner, j = (i / inner) % n_idx, o = i / (inner * n_idx);
  const uint32_t x = load_index(idx, it, j);
  if (x >= uint64_t(len)) return;  // never in any gather window: not written
  copy_bytes(out + i * es, data + ((o * len + x) * inner + e) * es, es);
}

// GatherElements on [M][N]: axis 0 reads data[idx][j], axis 1 data[i][idx].
__global__ void gather_elements_kernel(int64_t n, const uint8_t* data, int es,
                                       int64_t M, int64_t N, const void* idx,
                                       npu_dtype it, int64_t ocols, int axis,
                                       uint8_t* out) {
  NPU_TID;
  const int64_t r = i / ocols, c = i % ocols;
  const uint32_t x = load_index(idx, it, i);
  if (axis == 0) {
    if (x >= uint64_t(M)) return;
    copy_bytes(out + i * es, data + (int64_t(x) * N + c) * es, es);
  } else {
    if (x >= uint64_t(N)) return;
    copy_bytes(out + i * es, data + (r * N + x) * es, es);
  }
}

struct NdStrides {
  uint32_t s[4];
};

__global__ void gather_nd_kernel(int64_t n, const uint8_t* data, int es,
                                 int64_t nrows, int64_t row_len, const void* idx,
                                 npu_dtype it, int depth, NdStrides st,
                                 uint8_t* out) {
  NPU_TID;
  const int64_t t = i / row_len, e = i % row_len;
  uint32_t r = 0;  // the kernel linearizes in uint32 and range-checks only r
  for (int j = 0; j < depth; ++j) r += load_index(idx, it, t * depth + j) * st.s[j];
  if (r >= uint64_t(nrows)) return;
  copy_bytes(out + i * es, data + (int64_t(r) * row_len + e) * es, es);
}

// HWC resize: out[h][w][c] = in[ih[h]][iw[w]][c]; es bytes per channel.
__global__ void resize_gather_kernel(int64_t n, const uint8_t* in, uint8_t* out,
                                     int es, int64_t Hi, int64_t Wi, int64_t C,
                                     int64_t Wo, const int32_t* ih,
                                     const int32_t* iw) {
  NPU_TID;
  const int64_t c = i % C, w = (i / C) % Wo, h = i / (C * Wo);
  const int64_t y = ih[h], x = iw[w];
  if (y < 0 || y >= Hi || x < 0 || x >= Wi) return;
  copy_bytes(out + i * es, in + ((y * Wi + x) * C + c) * es, es);
}

int resize_nearest(void* stream, const void* in, void* out, int es, int64_t Hi,
                   int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                   const int32_t* idx_h, const int32_t* idx_w) {
  if (!es || Hi < 1 || Wi < 1 || C < 0 || Ho < 0 || Wo < 0) return -1;
  return launch(stream, Ho * Wo * C, resize_gather_kernel,
                static_cast<const uint8_t*>(in), static_cast<uint8_t*>(out), es,
                Hi, Wi, C, Wo, idx_h, idx_w);
}

// ---- not ---------------------------------------------------------------------

__global__ void not_kernel(int64_t n, const uint8_t* in, uint8_t* out) {
  NPU_TID;
  out[i] = in[i] == 0 ? 1 : 0;
}

// depth_to_space as a 6D transpose of the input viewed with the block axes
// split out. NCHW: DCR in [N][b][b][C'][H][W] -> (0, 3, 4, 1, 5, 2); CRD in
// [N][C'][b][b][H][W] -> (0, 1, 4, 2, 5, 3). Both give [N][C'][H][bh][W][bw].
int depth_to_space_nchw(void* stream, const void* in, void* out, int es,
                        int64_t N, int64_t C, int64_t H, int64_t W, int64_t b,
                        int mode) {
  if (!es || b < 1 || C % (b * b) || (mode != 0 && mode != 1)) return -1;
  const int64_t Cp = C / (b * b);
  if (mode == 0) {
    const int64_t s[6] = {N, b, b, Cp, H, W};
    const int p[6] = {0, 3, 4, 1, 5, 2};
    return transpose_nd(stream, in, out, es, 6, s, p);
  }
  const int64_t s[6] = {N, Cp, b, b, H, W};
  const int p[6] = {0, 1, 4, 2, 5, 3};
  return transpose_nd(stream, in, out, es, 6, s, p);
}

// NHWC: [N][H][W][C] with C split as above, to [N][H][bh][W][bw][C'].
int depth_to_space_nhwc(void* stream, const void* in, void* out, int es,
                        int64_t N, int64_t H, int64_t W, int64_t C, int64_t b,
                        int mode) {
  if (!es || b < 1 || C % (b * b) || (mode != 0 && mode != 1)) return -1;
  const int64_t Cp = C / (b * b);
  if (mode == 0) {
    const int64_t s[6] = {N, H, W, b, b, Cp};
    const int p[6] = {0, 1, 3, 2, 4, 5};
    return transpose_nd(stream, in, out, es, 6, s, p);
  }
  const int64_t s[6] = {N, H, W, Cp, b, b};
  const int p[6] = {0, 1, 4, 2, 5, 3};
  return transpose_nd(stream, in, out, es, 6, s, p);
}

// [H][W][C] viewed as [H/b][b][W/b][b][C] -> [H/b][W/b][C][bh][bw].
int space_to_depth(void* stream, const void* in, void* out, int es, int64_t H,
                   int64_t W, int64_t C, int64_t b) {
  if (!es || b < 1 || H % b || W % b) return -1;
  const int64_t s[5] = {H / b, b, W / b, b, C};
  const int p[5] = {0, 2, 4, 1, 3};
  return transpose_nd(stream, in, out, es, 5, s, p);
}

int pad_hwc(void* stream, const void* in, void* out, npu_dtype t, int64_t H,
            int64_t W, int64_t C, const int64_t* pads, int mode,
            uint64_t pad_bits) {
  const int es = elem_size(t);
  if (!es || H < 1 || W < 1 || C < 1 || mode < 0 || mode > 2) return -1;
  for (int k = 0; k < 6; ++k)
    if (pads[k] < 0) return -1;
  const int64_t in_dim[3] = {H, W, C};
  const int64_t od[3] = {H + pads[0] + pads[1], W + pads[2] + pads[3],
                         C + pads[4] + pads[5]};
  View v = dense_view(3, od);
  strides_of(3, in_dim, v.istride);
  const int8_t e = mode == 0 ? kConstant : (mode == 1 ? kReflect : kEdge);
  for (int k = 0; k < 3; ++k) {
    v.ioff[k] = -pads[2 * k];
    v.ilen[k] = in_dim[k];
    v.edge[k] = e;
  }
  return copy_view(stream, in, out, es, v, pad_bits);
}

}  // namespace

// ---- A. concat / slice / copy ----------------------------------------------

extern "C" int npu_generic_concat(void* stream, const void* in1, int64_t c1,
                                  int64_t s1, int64_t n1, const void* in2,
                                  int64_t c2, int64_t s2, int64_t n2, void* out,
                                  npu_dtype t, int64_t rows) {
  const void* ins[2] = {in1, in2};
  const int64_t ic[2] = {c1, c2}, st[2] = {s1, s2}, w[2] = {n1, n2};
  return concat_rows(stream, ins, ic, st, w, 2, out, t, rows);
}

extern "C" int npu_generic_concat_16b_inner(void* stream, const void* in1,
                                            int64_t c1, const void* in2,
                                            int64_t c2, void* out, npu_dtype t,
                                            int64_t rows) {
  const void* ins[2] = {in1, in2};
  const int64_t w[2] = {c1, c2}, st[2] = {0, 0};
  return concat_rows(stream, ins, w, st, w, 2, out, t, rows);
}

extern "C" int npu_generic_concat_16b_c6463(void* stream, const void* in1,
                                            const void* in2, void* out,
                                            npu_dtype t, int64_t rows) {
  const void* ins[2] = {in1, in2};
  const int64_t w[2] = {63, 1}, st[2] = {0, 0};
  return concat_rows(stream, ins, w, st, w, 2, out, t, rows);
}

extern "C" int npu_generic_concat_16b_multi(void* stream, const void* const* ins,
                                            const int64_t* widths, int k,
                                            void* out, npu_dtype t, int64_t rows) {
  return multi_concat(stream, ins, widths, k, out, t, rows);
}

extern "C" int npu_generic_concat_multi_c_fusion(void* stream,
                                                 const void* const* ins,
                                                 const int64_t* widths, int k,
                                                 void* out, npu_dtype t,
                                                 int64_t rows) {
  return multi_concat(stream, ins, widths, k, out, t, rows);
}

extern "C" int npu_concat_unaligned_bfp16(void* stream, const void* in1,
                                          int64_t n1, const void* in2,
                                          int64_t n2, void* out, int64_t rows) {
  if (n1 < 1 || n2 < 1 || rows < 0) return -1;
  // Phase 1 / 2 of concat_bfp16.hpp: ifm1_loads = (n1 - 1) / 128 whole v128
  // loads, then the first v64 of the last load when it is fully valid, are
  // pushed as loaded.
  const int64_t loads = (n1 - 1) / 128, R = n1 - 128 * loads;
  const int64_t raw = (128 * loads + (R >= 64 ? 64 : 0)) / 8;
  const int64_t ob = (n1 + n2 + 7) / 8;
  return launch(stream, rows * ob, concat_bfp16_kernel,
                static_cast<const uint8_t*>(in1), n1,
                static_cast<const uint8_t*>(in2), n2, static_cast<uint8_t*>(out),
                ob, raw);
}

extern "C" int npu_generic_slice_innermost(void* stream, const void* in,
                                           void* out, npu_dtype t, int64_t rows,
                                           int64_t c, int64_t start, int64_t end,
                                           int64_t stride) {
  const int es = elem_size(t);
  if (!es || stride < 1 || start < 0 || end > c || rows < 0) return -1;
  const int64_t w = end > start ? (end - start + stride - 1) / stride : 0;
  View v = rows_view(rows, w, c, start, w, 0);
  v.istep[1] = stride;
  return copy_view(stream, in, out, es, v);
}

extern "C" int npu_generic_slice_16b_inner_range(void* stream, const void* in,
                                                 void* out, npu_dtype t,
                                                 int64_t rows, int64_t c,
                                                 int64_t start, int64_t end) {
  return npu_generic_slice_innermost(stream, in, out, t, rows, c, start, end, 1);
}

extern "C" int npu_generic_slice_16b_inner(void* stream, const void* in,
                                           void* out0, void* out1, npu_dtype t,
                                           int64_t rows, int64_t c,
                                           int64_t split) {
  if (split < 0 || split > c) return -1;
  int rc = npu_generic_slice_innermost(stream, in, out0, t, rows, c, 0, split, 1);
  if (rc) return rc;
  return npu_generic_slice_innermost(stream, in, out1, t, rows, c, split, c, 1);
}

extern "C" int npu_generic_slice_16b_c6463(void* stream, const void* in,
                                           void* out0, void* out1, npu_dtype t,
                                           int64_t rows) {
  return npu_generic_slice_16b_inner(stream, in, out0, out1, t, rows, 64, 63);
}

extern "C" int npu_generic_mem_copy(void* stream, const void* in, void* out,
                                    npu_dtype t, int64_t n) {
  return copy_n(stream, in, out, t, n);
}

extern "C" int npu_generic_reshape_innermost(void* stream, const void* in,
                                             void* out, npu_dtype t, int64_t n) {
  return copy_n(stream, in, out, t, n);
}

extern "C" int npu_generic_pad_innermost(void* stream, const void* in, void* out,
                                         npu_dtype t, int64_t n) {
  return copy_n(stream, in, out, t, n);
}

extern "C" int npu_generic_unpad_innermost(void* stream, const void* in,
                                           void* out, npu_dtype t, int64_t n) {
  return copy_n(stream, in, out, t, n);
}

extern "C" int npu_generic_expand_innermost(void* stream, const void* in,
                                            void* out, npu_dtype t, int64_t rows,
                                            int64_t cin, int64_t cout) {
  const int es = elem_size(t);
  if (!es || cin < 1 || cout < 0 || cout % cin || rows < 0) return -1;
  View v = rows_view(rows, cout, cin, 0, cout, 0);
  v.ilen[1] = cin;
  v.edge[1] = kWrap;
  return copy_view(stream, in, out, es, v);
}

// ---- B. pad ----------------------------------------------------------------

extern "C" int npu_generic_inplace_pad2d_c64(void* stream, const void* in,
                                             void* out, npu_dtype t, int64_t H,
                                             int64_t W, int64_t C, int64_t hs,
                                             int64_t he, int64_t ws, int64_t we,
                                             uint64_t pad_bits) {
  const int64_t pads[6] = {hs, he, ws, we, 0, 0};
  return pad_hwc(stream, in, out, t, H, W, C, pads, 0, pad_bits);
}

extern "C" int npu_generic_inplace_pad3d(void* stream, const void* in, void* out,
                                         npu_dtype t, int64_t H, int64_t W,
                                         int64_t C, const int64_t* pads,
                                         int mode, uint64_t pad_bits) {
  return pad_hwc(stream, in, out, t, H, W, C, pads, mode, pad_bits);
}

// ---- C. transpose / space <-> depth ----------------------------------------

extern "C" int npu_generic_transpose_unaligned(void* stream, const void* in,
                                               void* out, npu_dtype t,
                                               const int64_t* shape,
                                               const int* perm) {
  return transpose_nd(stream, in, out, elem_size(t), 4, shape, perm);
}

extern "C" int npu_generic_transpose4d_aligned(void* stream, const void* in,
                                               void* out, npu_dtype t,
                                               const int64_t* shape,
                                               const int* perm) {
  return transpose_nd(stream, in, out, elem_size(t), 4, shape, perm);
}

extern "C" int npu_generic_transpose4d_0132_RCR32(void* stream, const void* in,
                                                  void* out, npu_dtype t,
                                                  int64_t d0, int64_t d1,
                                                  int64_t d2) {
  const int64_t s[3] = {d0, d1, d2};
  const int p[3] = {0, 2, 1};
  return transpose_nd(stream, in, out, elem_size(t), 3, s, p);
}

extern "C" int npu_generic_transpose_k8n8(void* stream, const void* in, void* out,
                                          npu_dtype t, int64_t K, int64_t N) {
  const int64_t s[2] = {K, N};
  const int p[2] = {1, 0};
  return transpose_nd(stream, in, out, elem_size(t), 2, s, p);
}

extern "C" int npu_transpose3d_mmac(void* stream, const void* in, void* out,
                                    npu_dtype t, const int64_t* shape,
                                    const int* perm) {
  if (t != NPU_I8 && t != NPU_U8 && t != NPU_I16 && t != NPU_U16) return -1;
  return transpose_nd(stream, in, out, elem_size(t), 3, shape, perm);
}

extern "C" int npu_transpose_mmac_a8_n64k64(void* stream, const void* in,
                                            void* out, int64_t K, int64_t N) {
  return npu_generic_transpose_k8n8(stream, in, out, NPU_I8, K, N);
}

extern "C" int npu_transpose_mmac_a16_n32k64(void* stream, const void* in,
                                             void* out, int64_t K, int64_t N) {
  return npu_generic_transpose_k8n8(stream, in, out, NPU_I16, K, N);
}

extern "C" int npu_transpose_planar3_c8(void* stream, const void* in, void* out,
                                        int64_t batch, int64_t P,
                                        uint8_t pad_value) {
  if (batch < 0 || P < 0) return -1;
  // out [batch][P][8] reads in [batch][3][P] transposed; channels 3..7 fall
  // outside the 3 input planes and take the pad value.
  const int64_t d[3] = {batch, P, 8};
  View v = dense_view(3, d);
  v.istride[0] = 3 * P;
  v.istride[1] = 1;
  v.istride[2] = P;
  v.ilen[2] = 3;
  v.edge[2] = kConstant;
  return copy_view(stream, in, out, 1, v, pad_value);
}

extern "C" int npu_generic_depth_to_space(void* stream, const void* in, void* out,
                                          npu_dtype t, int64_t N, int64_t C,
                                          int64_t H, int64_t W, int64_t b,
                                          int mode) {
  return depth_to_space_nchw(stream, in, out, elem_size(t), N, C, H, W, b, mode);
}

extern "C" int npu_generic_depth_to_space_wc64(void* stream, const void* in,
                                               void* out, npu_dtype t, int64_t N,
                                               int64_t H, int64_t W, int64_t C,
                                               int64_t b, int mode) {
  return depth_to_space_nhwc(stream, in, out, elem_size(t), N, H, W, C, b, mode);
}

extern "C" int npu_generic_space_to_depth(void* stream, const void* in, void* out,
                                          npu_dtype t, int64_t H, int64_t W,
                                          int64_t C, int64_t b) {
  return space_to_depth(stream, in, out, elem_size(t), H, W, C, b);
}

extern "C" int npu_generic_space_to_depth_c4bs4(void* stream, const void* in,
                                                void* out, npu_dtype t,
                                                int64_t H, int64_t W, int64_t C,
                                                int64_t b) {
  return space_to_depth(stream, in, out, elem_size(t), H, W, C, b);
}

// ---- D. gather / resize ----------------------------------------------------

extern "C" int npu_generic_gather_3din_1dindices(void* stream, const void* data,
                                                 npu_dtype t, int64_t d0,
                                                 int64_t d1, int64_t d2, int axis,
                                                 const void* idx,
                                                 npu_dtype idx_type,
                                                 int64_t n_idx, void* out) {
  const int es = elem_size(t);
  if (!es || !is_index_type(idx_type) || axis < 0 || axis > 2 || n_idx < 0)
    return -1;
  const int64_t d[3] = {d0, d1, d2};
  int64_t outer = 1, inner = 1;
  for (int k = 0; k < axis; ++k) outer *= d[k];
  for (int k = axis + 1; k < 3; ++k) inner *= d[k];
  return launch(stream, outer * n_idx * inner, gather_axis_kernel,
                static_cast<const uint8_t*>(data), es, d[axis], inner, idx,
                idx_type, n_idx, static_cast<uint8_t*>(out));
}

extern "C" int npu_generic_gather_element_2din_2dindices_axis0(
    void* stream, const void* data, npu_dtype t, int64_t M, int64_t N,
    const void* idx, npu_dtype idx_type, int64_t Mi, void* out) {
  const int es = elem_size(t);
  if (!es || !is_index_type(idx_type) || M < 0 || N < 0 || Mi < 0) return -1;
  return launch(stream, Mi * N, gather_elements_kernel,
                static_cast<const uint8_t*>(data), es, M, N, idx, idx_type, N, 0,
                static_cast<uint8_t*>(out));
}

extern "C" int npu_generic_gather_element_2din_2dindices_innermost(
    void* stream, const void* data, npu_dtype t, int64_t M, int64_t N,
    const void* idx, npu_dtype idx_type, int64_t Ni, void* out) {
  const int es = elem_size(t);
  if (!es || !is_index_type(idx_type) || M < 0 || N < 0 || Ni < 0) return -1;
  return launch(stream, M * Ni, gather_elements_kernel,
                static_cast<const uint8_t*>(data), es, M, N, idx, idx_type, Ni, 1,
                static_cast<uint8_t*>(out));
}

extern "C" int npu_gather_nd_4x3_adf(void* stream, const void* data, npu_dtype t,
                                     const int64_t* shape, int rank,
                                     const void* idx, npu_dtype idx_type,
                                     int64_t n_tuples, int depth, void* out) {
  const int es = elem_size(t);
  if (!es || (idx_type != NPU_U16 && idx_type != NPU_U32) || depth < 1 ||
      depth > 4 || depth > rank || n_tuples < 0)
    return -1;
  NdStrides st{};
  int64_t nrows = 1, row_len = 1;
  for (int j = depth - 1; j >= 0; --j) {
    st.s[j] = uint32_t(nrows);
    nrows *= shape[j];
  }
  for (int j = depth; j < rank; ++j) row_len *= shape[j];
  return launch(stream, n_tuples * row_len, gather_nd_kernel,
                static_cast<const uint8_t*>(data), es, nrows, row_len, idx,
                idx_type, depth, st, static_cast<uint8_t*>(out));
}

extern "C" int npu_resize_nearest_bf16(void* stream, const void* in, void* out,
                                       npu_dtype t, int64_t Hi, int64_t Wi,
                                       int64_t C, int64_t Ho, int64_t Wo,
                                       const int32_t* idx_h,
                                       const int32_t* idx_w) {
  if (t != NPU_F16 && t != NPU_BF16 && t != NPU_FP8) return -1;
  return resize_nearest(stream, in, out, elem_size(t), Hi, Wi, C, Ho, Wo, idx_h,
                        idx_w);
}

extern "C" int npu_resize_nearest_4x3(void* stream, const void* in, void* out,
                                      npu_dtype t, int64_t Hi, int64_t Wi,
                                      int64_t C, int64_t Ho, int64_t Wo,
                                      const int32_t* idx_h,
                                      const int32_t* idx_w) {
  if (t != NPU_F16 && t != NPU_BF16) return -1;
  return resize_nearest(stream, in, out, elem_size(t), Hi, Wi, C, Ho, Wo, idx_h,
                        idx_w);
}

extern "C" int npu_resize_nearest_bfp16(void* stream, const void* in, void* out,
                                        int64_t Hi, int64_t Wi, int64_t C,
                                        int64_t Ho, int64_t Wo,
                                        const int32_t* idx_h,
                                        const int32_t* idx_w) {
  if (C % 8) return -1;
  return resize_nearest(stream, in, out, 9, Hi, Wi, C / 8, Ho, Wo, idx_h, idx_w);
}

// ---- E. not ----------------------------------------------------------------

extern "C" int npu_elementwise_not(void* stream, const void* in,
                                   npu_dtype in_type, void* out,
                                   npu_dtype out_type, int64_t n) {
  if ((in_type != NPU_I8 && in_type != NPU_U8) ||
      (out_type != NPU_I8 && out_type != NPU_U8))
    return -1;
  return launch(stream, n, not_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out));
}
