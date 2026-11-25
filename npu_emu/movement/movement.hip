/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Data movement entry points; most are one strided copy over a View.
 */
#include <hip/hip_runtime.h>

#include "movement/movement.h"
#include "common/block_fp.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mmac.h"

using namespace npu_emu;

namespace {

constexpr int kMaxDims = 8;

// How an axis maps an out-of-range input coordinate.
enum Edge : int8_t { kDirect = 0, kConstant, kReflect, kEdge, kWrap };

// Along axis k the input coordinate is x * istep + ioff, mapped by edge[k]
// against ilen; offsets are in elements.
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

// Dense row-major output strides; input strides are left 0.
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
  if (n == 0) return 0;  // empty output: nothing to launch
  return launch(stream, n, copy_view_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), es, v, fill);
}

// Element size in bytes, or 0 if unsupported.
int elem_size(npu_dtype dtype) {
  switch (dtype) {
    case NPU_FP8:
    case NPU_I8:
    case NPU_U8:
      return 1;
    case NPU_F16:
    case NPU_BF16:
    case NPU_I16:
    case NPU_U16:
      return 2;
    default:
      return 0;
  }
}

// The 8- and 16-bit types of the generic movement kernels, optionally without FP8.
int es_of(npu_dtype dtype, bool fp8 = true) {
  return !fp8 && dtype == NPU_FP8 ? 0 : elem_size(dtype);
}

bool is_index_type(npu_dtype idx_type) {
  return idx_type == NPU_U16 || idx_type == NPU_I16 || idx_type == NPU_I32;
}

// Index as uint32; signed types are sign-extended.
__device__ inline uint32_t load_index(const void* p, npu_dtype idx_type, int64_t i) {
  return uint32_t(load_int(p, idx_type, i));
}

// [rows][width] from input column `start` to output column `col`.
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

// es-byte elements; all host arrays have k entries.
int concat_rows(void* stream, const void* const* ins, const int64_t* icols,
                const int64_t* starts, const int64_t* widths, int k, void* out, int es,
                int64_t rows) {
  if (!es || any_null(ins, icols, starts, widths, out) || k < 1 || rows < 0) return -1;
  int64_t ocols = 0;
  for (int j = 0; j < k; ++j) {
    if (!ins[j] || widths[j] < 0 || starts[j] < 0 || starts[j] + widths[j] > icols[j])
      return -1;
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

int multi_concat(void* stream, const void* const* ins, npu_dtype dtype, void* out, int64_t rows,
                 int n_in, const int64_t* channels) {
  if (n_in < 2 || n_in > 6) return -1;
  const int64_t zero[6] = {0, 0, 0, 0, 0, 0};
  return concat_rows(stream, ins, channels, zero, channels, n_in, out, es_of(dtype), rows);
}

int copy_n(void* stream, const void* in, npu_dtype dtype, void* out, int64_t n) {
  const int es = es_of(dtype);
  if (!es || any_null(in, out) || n < 0) return -1;
  View v = dense_view(1, &n);
  v.istride[0] = 1;
  return copy_view(stream, in, out, es, v);
}

int transpose_nd(void* stream, const void* in, void* out, int es, int nd,
                 const int64_t* shape, const int* perm) {
  if (!es || any_null(in, out, shape, perm) || nd < 1 || nd > kMaxDims) return -1;
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
      v[j] = bf16_to_f32(decode_bfp16_bf16(in1 + 9 * r * b1, p / 8, b1, int(p % 8)));
    else if (p < n1 + n2)
      v[j] = bf16_to_f32(decode_bfp16_bf16(in2 + 9 * r * b2, (p - n1) / 8, b2, int((p - n1) % 8)));
    else
      v[j] = 0.0f;
  }
  encode_bfp16(v, kHalfEven, o);
}

// data [outer][len][inner] gathered along the middle axis: out [outer][n_idx][inner].
__global__ void gather_axis_kernel(int64_t n, const uint8_t* data, int es,
                                   int64_t len, int64_t inner, const void* idx,
                                   npu_dtype idx_type, int64_t n_idx, uint8_t* out) {
  NPU_TID;
  const int64_t e = i % inner, j = (i / inner) % n_idx, o = i / (inner * n_idx);
  const uint32_t x = load_index(idx, idx_type, j);
  if (x >= uint64_t(len)) return;  // out of range: not written
  copy_bytes(out + i * es, data + ((o * len + x) * inner + e) * es, es);
}

// GatherElements on [M][N]: axis 0 reads data[idx][j], axis 1 data[i][idx].
__global__ void gather_elements_kernel(int64_t n, const uint8_t* data, int es,
                                       int64_t M, int64_t N, const void* idx,
                                       npu_dtype idx_type, int64_t ocols, int axis,
                                       uint8_t* out) {
  NPU_TID;
  const int64_t r = i / ocols, c = i % ocols;
  const uint32_t x = load_index(idx, idx_type, i);
  if (axis == 0) {
    if (x >= uint64_t(M)) return;
    copy_bytes(out + i * es, data + (int64_t(x) * N + c) * es, es);
  } else {
    if (x >= uint64_t(N)) return;
    copy_bytes(out + i * es, data + (r * N + x) * es, es);
  }
}

// HWC resize: out[h][w][c] = in[ih[h]][iw[w]][c]; es bytes per channel.
__global__ void resize_gather_kernel(int64_t n, const uint8_t* in, uint8_t* out,
                                     int es, int64_t Hi, int64_t Wi, int64_t C,
                                     int64_t Wo, const uint16_t* ih,
                                     const uint16_t* iw) {
  NPU_TID;
  const int64_t c = i % C, w = (i / C) % Wo, h = i / (C * Wo);
  const int64_t y = ih[h], x = iw[w];
  if (y >= Hi || x >= Wi) return;
  copy_bytes(out + i * es, in + ((y * Wi + x) * C + c) * es, es);
}

int resize_nearest(void* stream, const void* in, void* out, int es, int64_t Hi,
                   int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                   const void* idx_h, const void* idx_w) {
  // an input extent may be 0 only when the output extent is (nothing to read from)
  if (!es || any_null(in, out, idx_h, idx_w) || Hi < 0 || Wi < 0 || C < 0 || Ho < 0 || Wo < 0 ||
      (Hi == 0 && Ho) || (Wi == 0 && Wo))
    return -1;
  if (Ho == 0 || Wo == 0 || C == 0) return 0;  // empty output: nothing to launch
  return launch(stream, Ho * Wo * C, resize_gather_kernel,
                static_cast<const uint8_t*>(in), static_cast<uint8_t*>(out), es,
                Hi, Wi, C, Wo, static_cast<const uint16_t*>(idx_h),
                static_cast<const uint16_t*>(idx_w));
}

__global__ void not_kernel(int64_t n, const uint8_t* in, uint8_t* out) {
  NPU_TID;
  out[i] = in[i] == 0 ? 1 : 0;
}

// 6D transpose with the block axes split out. DCR: [N][b][b][C'][H][W], CRD:
// [N][C'][b][b][H][W]; both give [N][C'][H][bh][W][bw].
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

int pad_hwc(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
            int64_t W, int64_t C, const int64_t* pads, int mode,
            int32_t pad_value) {
  const int es = es_of(dtype, false);
  if (!es || any_null(in, out, pads) || H < 0 || W < 0 || C < 0 || mode < 0 || mode > 2) return -1;
  const int64_t in_dim[3] = {H, W, C};
  for (int k = 0; k < 3; ++k) {
    const int64_t s = pads[2 * k], e = pads[2 * k + 1];
    if (s < 0 || e < 0) return -1;
    // an empty axis cannot be padded (there is nothing to pad from)
    if (in_dim[k] == 0) {
      if (s || e) return -1;
      continue;
    }
    // reflect / edge: the padded extent exceeds twice the larger pad
    if (mode && in_dim[k] + s + e <= 2 * (s > e ? s : e)) return -1;
  }
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
  return copy_view(stream, in, out, es, v, uint64_t(uint32_t(pad_value)));
}

}  // namespace

extern "C" {

int npu_generic_concat(void* stream, const void* in1, npu_dtype dtype, const void* in2, void* out,
                       int64_t rows, int64_t c1, int64_t s1, int64_t n1, int64_t c2, int64_t s2,
                       int64_t n2) {
  const void* ins[2] = {in1, in2};
  if (dtype == NPU_BFP16) {  // whole 8-value blocks of 9 bytes
    if (c1 % 8 || c2 % 8 || n1 % 8 || n2 % 8 || s1 || s2) return -1;
    const int64_t ic[2] = {c1 / 8, c2 / 8}, st[2] = {0, 0}, w[2] = {n1 / 8, n2 / 8};
    return concat_rows(stream, ins, ic, st, w, 2, out, 9, rows);
  }
  const int64_t ic[2] = {c1, c2}, st[2] = {s1, s2}, w[2] = {n1, n2};
  return concat_rows(stream, ins, ic, st, w, 2, out, es_of(dtype), rows);
}

int npu_generic_concat_16b_inner(void* stream, const void* in1, npu_dtype dtype, const void* in2,
                                 void* out, int64_t rows, int64_t c1, int64_t c2) {
  const void* ins[2] = {in1, in2};
  const int64_t w[2] = {c1, c2}, st[2] = {0, 0};
  return concat_rows(stream, ins, w, st, w, 2, out, es_of(dtype), rows);
}

int npu_generic_concat_16b_c6463(void* stream, const void* in1, npu_dtype dtype, const void* in2,
                                 void* out, int64_t rows) {
  const void* ins[2] = {in1, in2};
  const int64_t w[2] = {63, 1}, st[2] = {0, 0};
  return concat_rows(stream, ins, w, st, w, 2, out, es_of(dtype), rows);
}

int npu_generic_concat_16b_multi(void* stream, const void* const* ins, npu_dtype dtype, void* out,
                                 int64_t rows, int n_in, const int64_t* channels) {
  return multi_concat(stream, ins, dtype, out, rows, n_in, channels);
}

int npu_generic_concat_multi_c_fusion(void* stream, const void* const* ins, npu_dtype dtype,
                                      void* out, int64_t rows, int n_in,
                                      const int64_t* channels) {
  return multi_concat(stream, ins, dtype, out, rows, n_in, channels);
}

int npu_concat_unaligned_bfp16(void* stream, const void* in1, const void* in2, void* out,
                               int64_t rows, int64_t n1, int64_t n2) {
  if (any_null(in1, in2, out) || n1 < 1 || n2 < 1 || rows < 0) return -1;
  // Leading blocks of in1 copied without re-encoding.
  const int64_t loads = (n1 - 1) / 128, R = n1 - 128 * loads;
  const int64_t raw = (128 * loads + (R >= 64 ? 64 : 0)) / 8;
  const int64_t ob = (n1 + n2 + 7) / 8;
  if (rows == 0) return 0;  // empty output: nothing to launch
  return launch(stream, rows * ob, concat_bfp16_kernel, static_cast<const uint8_t*>(in1), n1,
                static_cast<const uint8_t*>(in2), n2, static_cast<uint8_t*>(out), ob, raw);
}

}  // extern "C"

namespace {

int slice_rows(void* stream, const void* in, int es, void* out, int64_t rows, int64_t c,
               int64_t start, int64_t end, int64_t stride) {
  if (!es || any_null(in, out) || stride < 1 || stride > 2 || start < 0 || end > c || c < 0 ||
      rows < 0)
    return -1;
  const int64_t w = end > start ? (end - start + stride - 1) / stride : 0;
  View v = rows_view(rows, w, c, start, w, 0);
  v.istep[1] = stride;
  return copy_view(stream, in, out, es, v);
}

int slice_split(void* stream, const void* in, int es, void* out0, void* out1, int64_t rows,
                int64_t c, int64_t split) {
  if (split < 0 || split > c) return -1;
  if (any_null(in, out0, out1)) return -1;
  const int rc = slice_rows(stream, in, es, out0, rows, c, 0, split, 1);
  return rc ? rc : slice_rows(stream, in, es, out1, rows, c, split, c, 1);
}

bool is16b(npu_dtype dtype) { return elem_size(dtype) == 2; }

}  // namespace

extern "C" {

int npu_generic_slice_innermost(void* stream, const void* in, npu_dtype dtype, void* out,
                                int64_t rows, int64_t c, int64_t start, int64_t end,
                                int64_t stride) {
  return slice_rows(stream, in, es_of(dtype), out, rows, c, start, end, stride);
}

int npu_generic_slice_16b_inner_range(void* stream, const void* in, npu_dtype dtype, void* out,
                                      int64_t rows, int64_t c, int64_t start, int64_t end) {
  if (dtype != NPU_I8 && dtype != NPU_I16) return -1;
  return slice_rows(stream, in, es_of(dtype), out, rows, c, start, end, 1);
}

int npu_generic_slice_16b_inner(void* stream, const void* in, npu_dtype dtype, void* out0,
                                void* out1, int64_t rows, int64_t c, int64_t split) {
  if (!is_small_int(dtype)) return -1;
  return slice_split(stream, in, es_of(dtype), out0, out1, rows, c, split);
}

int npu_generic_slice_16b_c6463(void* stream, const void* in, npu_dtype dtype, void* out0,
                                void* out1, int64_t rows) {
  if (!is16b(dtype)) return -1;
  return slice_split(stream, in, 2, out0, out1, rows, 64, 63);
}

int npu_generic_mem_copy(void* stream, const void* in, npu_dtype dtype, void* out, int64_t n) {
  return copy_n(stream, in, dtype, out, n);
}

int npu_generic_reshape_innermost(void* stream, const void* in, npu_dtype dtype, void* out,
                                  int64_t n) {
  return copy_n(stream, in, dtype, out, n);
}

int npu_generic_pad_innermost(void* stream, const void* in, npu_dtype dtype, void* out, int64_t n) {
  return copy_n(stream, in, dtype, out, n);
}

int npu_generic_unpad_innermost(void* stream, const void* in, npu_dtype dtype, void* out,
                                int64_t n) {
  return copy_n(stream, in, dtype, out, n);
}

int npu_generic_expand_innermost(void* stream, const void* in, npu_dtype dtype, void* out,
                                 int64_t rows, int64_t cin, int64_t cout) {
  if ((dtype != NPU_I8 && !is_half_float(dtype)) || any_null(in, out) || cin < 1 || cout < 0 ||
      cout % cin || rows < 0)
    return -1;
  View v = rows_view(rows, cout, cin, 0, cout, 0);
  v.ilen[1] = cin;
  v.edge[1] = kWrap;
  return copy_view(stream, in, out, elem_size(dtype), v);
}

int npu_generic_inplace_pad2d_c64(void* stream, const void* in, npu_dtype dtype, void* out,
                                  int64_t H, int64_t W, int64_t C, int64_t hs, int64_t he,
                                  int64_t ws, int64_t we, int32_t pad_value) {
  const int64_t pads[6] = {hs, he, ws, we, 0, 0};
  return pad_hwc(stream, in, dtype, out, H, W, C, pads, 0, pad_value);
}

int npu_generic_inplace_pad3d(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                              int64_t W, int64_t C, const int64_t* pads, int mode,
                              int32_t pad_value) {
  return pad_hwc(stream, in, dtype, out, H, W, C, pads, mode, pad_value);
}

int npu_generic_transpose_unaligned(void* stream, const void* in, npu_dtype dtype, void* out,
                                    const int64_t* shape, const int* perm) {
  return transpose_nd(stream, in, out, es_of(dtype), 4, shape, perm);
}

int npu_generic_transpose4d_aligned(void* stream, const void* in, npu_dtype dtype, void* out,
                                    const int64_t* shape, const int* perm) {
  return transpose_nd(stream, in, out, es_of(dtype), 4, shape, perm);
}

int npu_generic_transpose4d_0132_RCR32(void* stream, const void* in, npu_dtype dtype, void* out,
                                       int64_t d0, int64_t d1, int64_t d2) {
  if (dtype != NPU_BF16 && dtype != NPU_U16) return -1;
  const int64_t s[3] = {d0, d1, d2};
  const int p[3] = {0, 2, 1};
  return transpose_nd(stream, in, out, 2, 3, s, p);
}

int npu_generic_transpose_k8n8(void* stream, const void* in, npu_dtype dtype, void* out, int64_t K,
                               int64_t N) {
  const int64_t s[2] = {K, N};
  const int p[2] = {1, 0};
  return transpose_nd(stream, in, out, es_of(dtype), 2, s, p);
}

int npu_transpose3d_mmac(void* stream, const void* in, npu_dtype dtype, void* out,
                         const int64_t* shape, const int* perm) {
  if (!is_small_int(dtype) || !perm) return -1;
  // The kernel moves the innermost axis: (0, 1, 2) and (1, 0, 2) are not supported.
  if (perm[2] == 2) return -1;
  return transpose_nd(stream, in, out, elem_size(dtype), 3, shape, perm);
}

int npu_transpose_mmac_a8_n64k64(void* stream, const void* in, void* out, int64_t K,
                                 int64_t N) {
  return npu_generic_transpose_k8n8(stream, in, NPU_I8, out, K, N);
}

int npu_transpose_mmac_a16_n32k64(void* stream, const void* in, void* out, int64_t K,
                                  int64_t N) {
  return npu_generic_transpose_k8n8(stream, in, NPU_I16, out, K, N);
}

int npu_transpose_planar3_c8(void* stream, const void* in, void* out, int64_t batch, int64_t P,
                             uint8_t pad_value) {
  if (any_null(in, out) || batch < 0 || P < 0) return -1;
  // Channels 3..7 fall outside the 3 input planes and take the pad value.
  const int64_t d[3] = {batch, P, 8};
  View v = dense_view(3, d);
  v.istride[0] = 3 * P;
  v.istride[1] = 1;
  v.istride[2] = P;
  v.ilen[2] = 3;
  v.edge[2] = kConstant;
  return copy_view(stream, in, out, 1, v, pad_value);
}

int npu_generic_depth_to_space(void* stream, const void* in, npu_dtype dtype, void* out, int64_t N,
                               int64_t C, int64_t H, int64_t W, int64_t b, int mode) {
  return depth_to_space_nchw(stream, in, out, es_of(dtype), N, C, H, W, b, mode);
}

int npu_generic_depth_to_space_wc64(void* stream, const void* in, npu_dtype dtype, void* out,
                                    int64_t N, int64_t H, int64_t W, int64_t C, int64_t b,
                                    int mode) {
  if (b != 2 && b != 4) return -1;
  return depth_to_space_nhwc(stream, in, out, es_of(dtype), N, H, W, C, b, mode);
}

int npu_generic_space_to_depth(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                               int64_t W, int64_t C, int64_t b) {
  return space_to_depth(stream, in, out, es_of(dtype), H, W, C, b);
}

int npu_generic_space_to_depth_c4bs4(void* stream, const void* in, npu_dtype dtype, void* out,
                                     int64_t H, int64_t W, int64_t C, int64_t b) {
  if (b != 4) return -1;
  return space_to_depth(stream, in, out, es_of(dtype), H, W, C, b);
}

int npu_generic_gather_3din_1dindices(void* stream, const void* data, npu_dtype data_type,
                                      const void* idx, void* out, int64_t d0, int64_t d1,
                                      int64_t d2, int64_t n_idx, int axis) {
  const int es = es_of(data_type);
  if (!es || any_null(data, idx, out) || axis < 0 || axis > 2 || d0 < 0 || d1 < 0 || d2 < 0 ||
      n_idx < 0)
    return -1;
  const int64_t d[3] = {d0, d1, d2};
  int64_t outer = 1, inner = 1;
  for (int k = 0; k < axis; ++k) outer *= d[k];
  for (int k = axis + 1; k < 3; ++k) inner *= d[k];
  if (outer * n_idx * inner == 0) return 0;  // empty output: nothing to launch
  return launch(stream, outer * n_idx * inner, gather_axis_kernel,
                static_cast<const uint8_t*>(data), es, d[axis], inner, idx, NPU_U32, n_idx,
                static_cast<uint8_t*>(out));
}

}  // extern "C"

namespace {

int gather_elements(void* stream, const void* data, npu_dtype data_type, const void* idx,
                    npu_dtype idx_type, void* out, int64_t M, int64_t N, int64_t on,
                    int axis) {
  const int es = es_of(data_type, false);
  if (!es || !is_index_type(idx_type) || any_null(data, idx, out) || M < 0 || N < 0 || on < 0)
    return -1;
  const int64_t rows = axis == 0 ? on : M, cols = axis == 0 ? N : on;
  if (rows * cols == 0) return 0;  // empty output: nothing to launch
  return launch(stream, rows * cols, gather_elements_kernel, static_cast<const uint8_t*>(data),
                es, M, N, idx, idx_type, cols, axis, static_cast<uint8_t*>(out));
}

}  // namespace

extern "C" {

int npu_generic_gather_element_2din_2dindices_axis0(void* stream, const void* data,
                                                    npu_dtype data_type, const void* idx,
                                                    npu_dtype idx_type, void* out, int64_t M,
                                                    int64_t N, int64_t Mi) {
  return gather_elements(stream, data, data_type, idx, idx_type, out, M, N, Mi, 0);
}

int npu_generic_gather_element_2din_2dindices_innermost(void* stream, const void* data,
                                                        npu_dtype data_type, const void* idx,
                                                        npu_dtype idx_type, void* out,
                                                        int64_t M, int64_t N, int64_t Ni) {
  return gather_elements(stream, data, data_type, idx, idx_type, out, M, N, Ni, 1);
}

int npu_resize_nearest_bf16(void* stream, const void* in, npu_dtype dtype, void* out, int64_t Hi,
                            int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, const void* idx_h,
                            const void* idx_w) {
  if (!is_half_float(dtype) && dtype != NPU_FP8) return -1;
  return resize_nearest(stream, in, out, elem_size(dtype), Hi, Wi, C, Ho, Wo, idx_h, idx_w);
}

int npu_resize_nearest_bfp16(void* stream, const void* in, void* out, int64_t Hi, int64_t Wi,
                             int64_t C, int64_t Ho, int64_t Wo, const void* idx_h,
                             const void* idx_w) {
  if (C % 8) return -1;
  return resize_nearest(stream, in, out, 9, Hi, Wi, C / 8, Ho, Wo, idx_h, idx_w);
}

int npu_elementwise_not(void* stream, const void* in, npu_dtype dtype, void* out, int64_t n) {
  if (!is_i8(dtype) || any_null(in, out) || n < 0) return -1;
  if (n == 0) return 0;  // empty output: nothing to launch
  return launch(stream, n, not_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out));
}

}  // extern "C"
