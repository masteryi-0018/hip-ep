/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Quantized selection / movement entry points; one thread per output element.
 */
#include <hip/hip_runtime.h>

#include "movement/movement.h"
#include "movement/qdq.h"
#include "common/bcast.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/grid.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/quant.h"

using namespace npu_emu;

namespace {

// |c1*x + c0| + zp as the larger of two MACs with pre-rounded addends.
__global__ void abs_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, float c1, float c0_plus_zp, float zp_minus_c0) {
  NPU_TID;
  const int64_t x = load_int(in, in_type, i);
  const float p = mac_exact(x, c1, c0_plus_zp);
  const float q = mac_exact(x, neg_f32(c1), zp_minus_c0);
  store_int(out, out_type, i, float_to_int(vmax(p, q), out_type, 0, kHalfEven));
}

__global__ void affine_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                              npu_dtype out_type, float scale, float offset) {
  NPU_TID;
  store_int(out, out_type, i, requant(load_int(in, in_type, i), scale, offset, out_type));
}

__global__ void clip_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, float c1, float c0, float lo, float hi) {
  NPU_TID;
  const float v = vclamp(mac_exact(load_int(in, in_type, i), c1, c0), lo, hi);
  store_int(out, out_type, i, float_to_int(v, out_type, 0, kHalfEven));
}

enum CmpOp { kEqual = 0, kGreater = 1 };

__global__ void compare_kernel(int64_t n, const void* a, npu_dtype a_type, const void* b,
                               npu_dtype b_type, int8_t* out, float sa, float sb, float za,
                               float zb, CmpOp op) {
  NPU_TID;
  const float fa = dequant(load_int(a, a_type, i), sa, za);
  const float fb = dequant(load_int(b, b_type, i), sb, zb);
  out[i] = int8_t(op == kEqual ? veq(fa, fb) : vgt(fa, fb));
}

// max / min of the dequantized a and b (b optionally broadcast), requantized.
struct MinMaxArgs {
  float sa, sb, so, za, zb, zo;
  int64_t W, C, bH, bW, bC;
  bool is_max, bcast;
};

__global__ void minmax_kernel(int64_t n, const void* a, npu_dtype a_type, const void* b,
                              npu_dtype b_type, void* out, npu_dtype out_type, MinMaxArgs g) {
  NPU_TID;
  int64_t j = i;
  if (g.bcast) {
    const int64_t c = i % g.C, w = (i / g.C) % g.W, h = i / (g.C * g.W);
    const BIdx k = bidx(h, w, c, dims_bits(g.bH, g.bW, g.bC), g.W, g.C);
    j = k.pixel + k.c;
  }
  const float fa = dequant(load_int(a, a_type, i), g.sa, g.za);
  const float fb = dequant(load_int(b, b_type, j), g.sb, g.zb);
  const float m = g.is_max ? vmax(fa, fb) : vmin(fa, fb);
  store_int(out, out_type, i, quant(m, g.so, g.zo, out_type));
}

struct WhereArgs {
  int64_t W, C, cH, cW, cC;
  int scalar_position;
};

__device__ inline bool where_take_x(const void* cond, npu_dtype cond_type, int64_t i, WhereArgs g) {
  const int64_t c = i % g.C, w = (i / g.C) % g.W, h = i / (g.C * g.W);
  const BIdx k = bidx(h, w, c, dims_bits(g.cH, g.cW, g.cC), g.W, g.C);
  const bool nz = load_int(cond, cond_type, k.pixel + k.c) != 0;
  return g.scalar_position ? nz : !nz;
}

// t: the unsigned type of the element width (raw bits move).
__global__ void where_kernel(int64_t n, const void* cond, const void* x, void* out, npu_dtype t,
                             uint32_t y_bits, WhereArgs g) {
  NPU_TID;
  store_int(out, t, i, where_take_x(cond, NPU_I8, i, g) ? load_int(x, t, i) : int64_t(y_bits));
}

__global__ void qdq_where_kernel(int64_t n, const void* cond, npu_dtype cond_type, const void* x,
                                 npu_dtype dtype, void* out, int32_t y, WhereArgs g, float scale,
                                 float offset) {
  NPU_TID;
  const int64_t v = where_take_x(cond, cond_type, i, g) ? load_int(x, dtype, i) : int64_t(y);
  store_int(out, dtype, i, requant(v, scale, offset, dtype));
}

struct ReduceArgs {
  int64_t H, W, C;
  int axis_flags;
  bool is_max, qdq;
  float offset, scale;
  int shift;
};

__global__ void reduce_kernel(int64_t n, const void* in, npu_dtype dtype, void* out,
                              ReduceArgs g) {
  NPU_TID;
  const bool rc = g.axis_flags & 1, rw = g.axis_flags & 2, rh = g.axis_flags & 4;
  const int64_t oC = rc ? 1 : g.C, oW = rw ? 1 : g.W;
  const int64_t oc = i % oC, ow = (i / oC) % oW, oh = i / (oC * oW);
  int64_t r = 0;
  bool first = true;
  for (int64_t h = rh ? 0 : oh; h < (rh ? g.H : oh + 1); ++h)
    for (int64_t w = rw ? 0 : ow; w < (rw ? g.W : ow + 1); ++w)
      for (int64_t c = rc ? 0 : oc; c < (rc ? g.C : oc + 1); ++c) {
        const int64_t v = load_int(in, dtype, (h * g.W + w) * g.C + c);
        if (first || (g.is_max ? v > r : v < r)) r = v;
        first = false;
      }
  store_int(out, dtype, i, g.qdq ? requant(r, g.scale, g.offset, dtype, g.shift) : r);
}

struct PoolArgs {
  int64_t W, C, Wo;
  int kh, kw, sh, sw;
  bool qdq;
  float offset, scale;
  int shift;
};

__global__ void maxpool_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                               npu_dtype out_type, PoolArgs g) {
  NPU_TID;
  const int64_t c = i % g.C, ow = (i / g.C) % g.Wo, oh = i / (g.C * g.Wo);
  int64_t m = load_int(in, in_type, (oh * g.sh * g.W + ow * g.sw) * g.C + c);
  for (int y = 0; y < g.kh; ++y)
    for (int x = 0; x < g.kw; ++x) {
      const int64_t v = load_int(in, in_type, ((oh * g.sh + y) * g.W + ow * g.sw + x) * g.C + c);
      m = v > m ? v : m;
    }
  store_int(out, out_type, i, g.qdq ? requant(m, g.scale, g.offset, out_type, g.shift) : m);
}

constexpr int kMaxConcat = 6;

struct ConcatArgs {
  const void* in[kMaxConcat];
  int64_t start[kMaxConcat + 1];  // first output channel of each input; [n_in] = total
  float scale[kMaxConcat], offset[kMaxConcat];
  int n_in;
};

__global__ void concat_kernel(int64_t n, ConcatArgs g, npu_dtype in_type, void* out,
                              npu_dtype out_type) {
  NPU_TID;
  const int64_t total = g.start[g.n_in];
  const int64_t r = i / total, c = i % total;
  int k = 0;
  while (c >= g.start[k + 1]) ++k;
  const int64_t ck = g.start[k + 1] - g.start[k];
  const int64_t v = load_int(g.in[k], in_type, r * ck + c - g.start[k]);
  store_int(out, out_type, i, requant(v, g.scale[k], g.offset[k], out_type));
}

__global__ void gather_axis0_kernel(int64_t n, const void* in, npu_dtype in_type,
                                    const int32_t* idx, void* out, npu_dtype out_type,
                                    int64_t M_in, int64_t N, float scale, float offset) {
  NPU_TID;
  const uint32_t row = uint32_t(idx[i]);  // unsigned compare
  if (row >= uint64_t(M_in)) return;     // out of range: not written
  const int64_t k = i % N;
  store_int(out, out_type, i,
            requant(load_int(in, in_type, int64_t(row) * N + k), scale, offset, out_type));
}

// Grid code -> rounded pixel coordinate; each fp32 step rounds.
__device__ inline int64_t grid_coord(int64_t code, const npu_gridsample_qdq& q, int64_t extent,
                                     int align_corners, int padding_mode) {
  const float v = qdq_grid_coord(code, q, extent, align_corners);
  const int64_t r = float_to_int(v, NPU_I16, 0, kHalfEven);
  return padding_mode == 1 ? clamp64(r, 0, extent - 1) : r;
}

struct GridArgs {
  int64_t H, W, C, Wo;
  int align_corners, padding_mode;
  npu_gridsample_qdq q;
  float ifm_offset;  // fp32(-(float)ifm_zp * ifm_scale)
};

__global__ void gridsample_nearest_kernel(int64_t n, const void* in, npu_dtype dtype,
                                          const void* grid, npu_dtype grid_type, void* out,
                                          GridArgs g) {
  NPU_TID;
  const int64_t c = i % g.C, p = i / g.C;  // p = output pixel
  const int64_t rx = grid_coord(load_int(grid, grid_type, 2 * p), g.q, g.W, g.align_corners,
                                g.padding_mode);
  const int64_t ry = grid_coord(load_int(grid, grid_type, 2 * p + 1), g.q, g.H, g.align_corners,
                                g.padding_mode);
  float v = 0.0f;
  if (rx >= 0 && rx < g.W && ry >= 0 && ry < g.H) {
    const float dq = dequant(load_int(in, dtype, (ry * g.W + rx) * g.C + c), g.q.ifm_scale,
                             g.ifm_offset);
    v = round_float(dq, NPU_BF16, kHalfEven);
  }
  store_int(out, dtype, i, quant(v, g.q.ofm_scale, float(g.q.ofm_zp), dtype));
}

__global__ void resize_nearest_kernel(int64_t n, const void* in, npu_dtype dtype, void* out,
                                      int64_t H, int64_t W, int64_t C, int64_t Wo,
                                      const uint16_t* sy, const uint16_t* sx, int requantize,
                                      float scale, float offset) {
  NPU_TID;
  const int64_t c = i % C, ow = (i / C) % Wo, oh = i / (C * Wo);
  const int64_t y = sy[oh], x = sx[ow];
  // [default] a source index outside the input leaves the output element unwritten.
  if (y >= H || x >= W) return;
  const int64_t v = load_int(in, dtype, (y * W + x) * C + c);
  store_int(out, dtype, i, requantize ? requant(v, scale, offset, dtype, 0, kTrunc) : v);
}

int unary(int bits, const void* in, npu_dtype in_type, const void* out, npu_dtype out_type,
          int64_t n) {
  return !any_null(in, out) && n >= 0 && is_int_bits(in_type, bits) && is_int_bits(out_type, bits)
             ? 0
             : -1;
}

int abs_impl(int bits, void* stream, const void* in, npu_dtype in_type, void* out,
             npu_dtype out_type, int64_t n, float c1, float c0, float zp) {
  if (unary(bits, in, in_type, out, out_type, n)) return -1;
  const float c0_plus_zp = add_q(c0, zp), zp_minus_c0 = sub_q(zp, c0);
  return launch(stream, n, abs_kernel, in, in_type, out, out_type, c1, c0_plus_zp, zp_minus_c0);
}

int neg_impl(int bits, void* stream, const void* in, npu_dtype in_type, void* out,
             npu_dtype out_type, int64_t n, float c1, float c0) {
  if (unary(bits, in, in_type, out, out_type, n)) return -1;
  return launch(stream, n, affine_kernel, in, in_type, out, out_type, c1, c0);
}

int clip_impl(int bits, void* stream, const void* in, npu_dtype in_type, void* out,
              npu_dtype out_type, int64_t n, float c1, float c0, float lo, float hi) {
  if (unary(bits, in, in_type, out, out_type, n)) return -1;
  return launch(stream, n, clip_kernel, in, in_type, out, out_type, c1, c0, lo, hi);
}

int compare_impl(int bits, void* stream, const void* a, npu_dtype a_type, const void* b,
                 npu_dtype b_type, void* out, int64_t n, float sa, float sb, float za, float zb,
                 CmpOp op) {
  if (!is_int_bits(a_type, bits) || !is_int_bits(b_type, bits) || any_null(a, b, out) || n < 0)
    return -1;
  return launch(stream, n, compare_kernel, a, a_type, b, b_type, static_cast<int8_t*>(out), sa,
                sb, za, zb, op);
}

int max_impl(int bits, void* stream, const void* a, npu_dtype dtype, const void* b, void* out,
             int64_t n, float sa, float sb, float so, float za, float zb, float zo) {
  if (!is_int_bits(dtype, bits) || any_null(a, b, out) || n < 0) return -1;
  MinMaxArgs g{sa, sb, so, za, zb, zo, 1, 1, 1, 1, 1, true, false};
  return launch(stream, n, minmax_kernel, a, dtype, b, dtype, out, dtype, g);
}

bool bcast_ok(int64_t full, int64_t b) { return b == full || b == 1; }

// Full shape s (an extent may be 0: empty output) and the shape b of an operand broadcast to
// it; false if invalid.
bool shapes_ok(const int64_t* s, const int64_t* b) {
  if (any_null(s, b)) return false;
  for (int d = 0; d < 3; ++d)
    if (s[d] < 0 || !bcast_ok(s[d], b[d])) return false;
  return true;
}

// Reduce extents: a reduced axis (axis_flags bit 0 / 1 / 2 = C / W / H) needs >= 1 element,
// a kept one may be empty.
bool reduce_dims_ok(int64_t H, int64_t W, int64_t C, int axis_flags) {
  return H >= (axis_flags & 4 ? 1 : 0) && W >= (axis_flags & 2 ? 1 : 0) &&
         C >= (axis_flags & 1 ? 1 : 0);
}

int where_impl(npu_dtype t, void* stream, const void* cond, const void* x, void* out,
               const int64_t* s, const int64_t* cs, uint32_t y_bits, int scalar_position) {
  if (any_null(cond, x, out) || !shapes_ok(s, cs) ||
      (scalar_position != 0 && scalar_position != 1))
    return -1;
  WhereArgs g{s[1], s[2], cs[0], cs[1], cs[2], scalar_position};
  return launch(stream, s[0] * s[1] * s[2], where_kernel, cond, x, out, t, y_bits, g);
}

int qdq_where_impl(int bits, void* stream, const void* cond, const void* x, npu_dtype dtype,
                   void* out, const int64_t* s, const int64_t* cs, int scalar_position, float c0,
                   float c1, int32_t y) {
  if (!is_int_bits(dtype, bits) || any_null(cond, x, out) || !shapes_ok(s, cs) ||
      (scalar_position != 0 && scalar_position != 1))
    return -1;
  // The scalar is the low bits of y, read as dtype.
  int32_t yv = y;
  switch (dtype) {
    case NPU_I8: yv = int8_t(y); break;
    case NPU_U8: yv = uint8_t(y); break;
    case NPU_I16: yv = int16_t(y); break;
    default: yv = uint16_t(y); break;
  }
  WhereArgs g{s[1], s[2], cs[0], cs[1], cs[2], scalar_position};
  return launch(stream, s[0] * s[1] * s[2], qdq_where_kernel, cond, bits == 8 ? NPU_I8 : NPU_I16,
                x, dtype, out, yv, g, c0, c1);
}

int min_impl(int bits, void* stream, const void* a, npu_dtype dtype, const void* b, void* out,
             const int64_t* s, const int64_t* bs, float sa, float sb, float so, float za,
             float zb, float zo) {
  if (!is_int_bits(dtype, bits) || any_null(a, b, out) || !shapes_ok(s, bs)) return -1;
  MinMaxArgs g{sa, sb, so, za, zb, zo, s[1], s[2], bs[0], bs[1], bs[2], false, true};
  return launch(stream, s[0] * s[1] * s[2], minmax_kernel, a, dtype, b, dtype, out, dtype, g);
}

int reduce_impl(int bits, void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                int64_t W, int64_t C, int axis_flags, bool is_max, bool qdq, float c0,
                float c1, int shift) {
  if (!is_int_bits(dtype, bits) || any_null(in, out) || axis_flags < 1 || axis_flags > 7 ||
      !reduce_dims_ok(H, W, C, axis_flags))
    return -1;
  ReduceArgs g{H, W, C, axis_flags, is_max, qdq, c0, c1, shift};
  const int64_t n = (axis_flags & 4 ? 1 : H) * (axis_flags & 2 ? 1 : W) * (axis_flags & 1 ? 1 : C);
  return launch(stream, n, reduce_kernel, in, dtype, out, g);
}

// kmax: the largest window / stride the kernel takes.
int pool_impl(int bits, void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
              int64_t W, int64_t C, int kh, int kw, int sh, int sw, int kmax, bool qdq,
              float c0, float c1, int shift) {
  if (!is_int_bits(dtype, bits) || any_null(in, out) || kh < 1 || kw < 1 || sh < 1 || sw < 1 ||
      kh > kmax || kw > kmax || sh > kmax || sw > kmax || kh > H || kw > W || C < 0)
    return -1;
  const int64_t Ho = (H - kh) / sh + 1, Wo = (W - kw) / sw + 1;
  PoolArgs g{W, C, Wo, kh, kw, sh, sw, qdq, c0, c1, shift};
  return launch(stream, Ho * Wo * C, maxpool_kernel, in, dtype, out, dtype, g);
}

int globalpool_impl(int bits, void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                    int64_t W, int64_t C, float c0, float c1, int shift) {
  if (H > INT32_MAX || W > INT32_MAX) return -1;
  return pool_impl(bits, stream, in, dtype, out, H, W, C, int(H), int(W), 1, 1, INT32_MAX, true,
                   c0, c1, shift);
}

// c0 = scale and c1 = offset per input (one pair for all unless per_input).
int concat_impl(int bits, void* stream, const void* const* ins, npu_dtype dtype, void* out,
                int64_t rows, int n_in, const int64_t* channels, const float* c0,
                const float* c1, bool per_input) {
  if (!is_int_bits(dtype, bits) || n_in < 2 || n_in > kMaxConcat ||
      any_null(ins, channels, out, c0, c1) || rows < 0)
    return -1;
  ConcatArgs g{};
  g.n_in = n_in;
  for (int k = 0; k < n_in; ++k) {
    if (channels[k] < 0 || !ins[k]) return -1;
    g.in[k] = ins[k];
    g.start[k + 1] = g.start[k] + channels[k];
    g.scale[k] = per_input ? c0[k] : c0[0];
    g.offset[k] = per_input ? c1[k] : c1[0];
  }
  if (rows == 0 || g.start[n_in] == 0) return 0;
  return launch(stream, rows * g.start[n_in], concat_kernel, g, dtype, out, dtype);
}

int gridsample_impl(void* stream, const void* in, npu_dtype dtype, const void* grid,
                    npu_dtype grid_type, void* out, int64_t H, int64_t W, int64_t C, int64_t Ho,
                    int64_t Wo, int align_corners, int padding_mode,
                    const npu_gridsample_qdq* coeff) {
  if (any_null(coeff, in, grid, out) || H <= 0 || W <= 0 || C < 0 || Ho < 0 || Wo < 0 ||
      (align_corners != 0 && align_corners != 1) || (padding_mode != 0 && padding_mode != 1))
    return -1;
  GridArgs g{H, W, C, Wo, align_corners, padding_mode, *coeff,
             dq_offset(float(coeff->ifm_zp), coeff->ifm_scale)};
  return launch(stream, Ho * Wo * C, gridsample_nearest_kernel, in, dtype, grid, grid_type, out, g);
}

// Requantizes the moved output in place.
int requant_inplace(int rc, void* stream, void* p, npu_dtype t, int64_t n, float c1,
                    float c0) {
  if (rc) return rc;
  return launch(stream, n, affine_kernel, static_cast<const void*>(p), t, p, t, c1, c0);
}

}  // namespace

extern "C" {

int npu_elementwise_abs_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, float c1, float c0, float ofm_zp) {
  return abs_impl(8, stream, in, in_type, out, out_type, n, c1, c0, ofm_zp);
}
int npu_elementwise_abs_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, float c1, float c0, float ofm_zp) {
  return abs_impl(16, stream, in, in_type, out, out_type, n, c1, c0, ofm_zp);
}
int npu_elementwise_neg_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, float c1, float c0) {
  return neg_impl(8, stream, in, in_type, out, out_type, n, c1, c0);
}
int npu_elementwise_neg_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, float c1, float c0) {
  return neg_impl(16, stream, in, in_type, out, out_type, n, c1, c0);
}
int npu_elementwise_clip_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, float c1, float c0, float cmin,
                            float cmax) {
  return clip_impl(8, stream, in, in_type, out, out_type, n, c1, c0, cmin, cmax);
}
int npu_elementwise_clip_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                             npu_dtype out_type, int64_t n, float c1, float c0, float cmin,
                             float cmax) {
  return clip_impl(16, stream, in, in_type, out, out_type, n, c1, c0, cmin, cmax);
}

int npu_elementwise_equal_a8(void* stream, const void* a, npu_dtype a_type, const void* b,
                             npu_dtype b_type, void* out, int64_t n, float scale_a,
                             float scale_b, float zp_a, float zp_b) {
  return compare_impl(8, stream, a, a_type, b, b_type, out, n, scale_a, scale_b, zp_a, zp_b,
                      kEqual);
}
int npu_elementwise_equal_a16(void* stream, const void* a, npu_dtype a_type, const void* b,
                              npu_dtype b_type, void* out, int64_t n, float scale_a,
                              float scale_b, float zp_a, float zp_b) {
  return compare_impl(16, stream, a, a_type, b, b_type, out, n, scale_a, scale_b, zp_a, zp_b,
                      kEqual);
}
int npu_elementwise_greater_a8(void* stream, const void* a, npu_dtype a_type, const void* b,
                               npu_dtype b_type, void* out, int64_t n, float scale_a,
                               float scale_b, float zp_a, float zp_b) {
  return compare_impl(8, stream, a, a_type, b, b_type, out, n, scale_a, scale_b, zp_a, zp_b,
                      kGreater);
}
int npu_elementwise_greater_a16(void* stream, const void* a, npu_dtype a_type, const void* b,
                                npu_dtype b_type, void* out, int64_t n, float scale_a,
                                float scale_b, float zp_a, float zp_b) {
  return compare_impl(16, stream, a, a_type, b, b_type, out, n, scale_a, scale_b, zp_a, zp_b,
                      kGreater);
}
int npu_elementwise_max_a8(void* stream, const void* a, npu_dtype dtype, const void* b,
                           void* out, int64_t n, float scale_a, float scale_b, float scale_o,
                           float zp_a, float zp_b, float zp_o) {
  return max_impl(8, stream, a, dtype, b, out, n, scale_a, scale_b, scale_o, zp_a, zp_b, zp_o);
}
int npu_elementwise_max_a16(void* stream, const void* a, npu_dtype dtype, const void* b,
                            void* out, int64_t n, float scale_a, float scale_b, float scale_o,
                            float zp_a, float zp_b, float zp_o) {
  return max_impl(16, stream, a, dtype, b, out, n, scale_a, scale_b, scale_o, zp_a, zp_b, zp_o);
}

int npu_elementwise_broadcasting3d_where_a8(void* stream, const void* cond, const void* x,
                                            void* out, const int64_t* shape,
                                            const int64_t* cond_shape, int scalar_position,
                                            int32_t y) {
  return where_impl(NPU_U8, stream, cond, x, out, shape, cond_shape, uint32_t(y) & 0xff,
                    scalar_position);
}
int npu_elementwise_broadcasting3d_where_a16(void* stream, const void* cond, const void* x,
                                             void* out, const int64_t* shape,
                                             const int64_t* cond_shape, int scalar_position,
                                             int32_t y) {
  return where_impl(NPU_U16, stream, cond, x, out, shape, cond_shape, uint32_t(y) & 0xffff,
                    scalar_position);
}
int npu_qdq_elementwise_broadcasting3d_where_a8(void* stream, const void* cond, const void* x,
                                                npu_dtype dtype, void* out,
                                                const int64_t* shape,
                                                const int64_t* cond_shape, int scalar_position,
                                                float c0, float c1, int32_t y) {
  return qdq_where_impl(8, stream, cond, x, dtype, out, shape, cond_shape, scalar_position, c0,
                        c1, y);
}
int npu_qdq_elementwise_broadcasting3d_where_a16(void* stream, const void* cond, const void* x,
                                                 npu_dtype dtype, void* out,
                                                 const int64_t* shape,
                                                 const int64_t* cond_shape,
                                                 int scalar_position, float c0, float c1,
                                                 int32_t y) {
  return qdq_where_impl(16, stream, cond, x, dtype, out, shape, cond_shape, scalar_position, c0,
                        c1, y);
}
int npu_qdq_elementwise_broadcasting3d_min_a8(void* stream, const void* a, npu_dtype dtype,
                                              const void* b, void* out, const int64_t* shape1,
                                              const int64_t* shape2, float scale_a,
                                              float scale_b, float scale_o, float zp_a,
                                              float zp_b, float zp_o) {
  return min_impl(8, stream, a, dtype, b, out, shape1, shape2, scale_a, scale_b, scale_o, zp_a,
                  zp_b, zp_o);
}
int npu_qdq_elementwise_broadcasting3d_min_a16(void* stream, const void* a, npu_dtype dtype,
                                               const void* b, void* out, const int64_t* shape1,
                                               const int64_t* shape2, float scale_a,
                                               float scale_b, float scale_o, float zp_a,
                                               float zp_b, float zp_o) {
  return min_impl(16, stream, a, dtype, b, out, shape1, shape2, scale_a, scale_b, scale_o, zp_a,
                  zp_b, zp_o);
}

int npu_reduce_max_c64_a8(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                          int64_t W, int64_t C, int axis_flags) {
  return reduce_impl(8, stream, in, dtype, out, H, W, C, axis_flags, true, false, 0, 0, 0);
}
int npu_reduce_max_c64_a16(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                           int64_t W, int64_t C, int axis_flags) {
  return reduce_impl(16, stream, in, dtype, out, H, W, C, axis_flags, true, false, 0, 0, 0);
}
int npu_reduce_min_c64_a8(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                          int64_t W, int64_t C, int axis_flags) {
  return reduce_impl(8, stream, in, dtype, out, H, W, C, axis_flags, false, false, 0, 0, 0);
}
int npu_reduce_min_c64_a16(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                           int64_t W, int64_t C, int axis_flags) {
  return reduce_impl(16, stream, in, dtype, out, H, W, C, axis_flags, false, false, 0, 0, 0);
}
int npu_qdq_reduce_max_c64_a8(void* stream, const void* in, npu_dtype dtype, void* out,
                              int64_t H, int64_t W, int64_t C, int axis_flags, float c0,
                              float c1, int16_t shift) {
  return reduce_impl(8, stream, in, dtype, out, H, W, C, axis_flags, true, true, c0, c1, shift);
}
int npu_qdq_reduce_max_c64_a16(void* stream, const void* in, npu_dtype dtype, void* out,
                               int64_t H, int64_t W, int64_t C, int axis_flags, float c0,
                               float c1, int16_t shift) {
  return reduce_impl(16, stream, in, dtype, out, H, W, C, axis_flags, true, true, c0, c1, shift);
}
int npu_qdq_reduce_min_c64_a8(void* stream, const void* in, npu_dtype dtype, void* out,
                              int64_t H, int64_t W, int64_t C, int axis_flags, float c0,
                              float c1, int16_t shift) {
  return reduce_impl(8, stream, in, dtype, out, H, W, C, axis_flags, false, true, c0, c1, shift);
}
int npu_qdq_reduce_min_c64_a16(void* stream, const void* in, npu_dtype dtype, void* out,
                               int64_t H, int64_t W, int64_t C, int axis_flags, float c0,
                               float c1, int16_t shift) {
  return reduce_impl(16, stream, in, dtype, out, H, W, C, axis_flags, false, true, c0, c1,
                     shift);
}

int npu_qdq_maxpool_a8(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                       int64_t W, int64_t C, int kh, int kw, int sh, int sw, float c0, float c1,
                       int16_t shift) {
  return pool_impl(8, stream, in, dtype, out, H, W, C, kh, kw, sh, sw, 32, true, c0, c1, shift);
}
int npu_qdq_maxpool_a16(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                        int64_t W, int64_t C, int kh, int kw, int sh, int sw, float c0,
                        float c1, int16_t shift) {
  return pool_impl(16, stream, in, dtype, out, H, W, C, kh, kw, sh, sw, 32, true, c0, c1,
                   shift);
}
int npu_generic_maxpool_a8(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                           int64_t W, int64_t C, int kh, int kw, int sh, int sw) {
  return pool_impl(8, stream, in, dtype, out, H, W, C, kh, kw, sh, sw, 15, false, 0, 0, 0);
}
int npu_qdq_globalmaxpool_a8(void* stream, const void* in, npu_dtype dtype, void* out,
                             int64_t H, int64_t W, int64_t C, float c0, float c1,
                             int16_t shift) {
  return globalpool_impl(8, stream, in, dtype, out, H, W, C, c0, c1, shift);
}
int npu_qdq_globalmaxpool_a16(void* stream, const void* in, npu_dtype dtype, void* out,
                              int64_t H, int64_t W, int64_t C, float c0, float c1,
                              int16_t shift) {
  return globalpool_impl(16, stream, in, dtype, out, H, W, C, c0, c1, shift);
}

int npu_qdq_concat_a8(void* stream, const void* a, npu_dtype dtype, const void* b, void* out,
                      int64_t rows, int64_t ca, int64_t cb, float c0, float c1) {
  const void* ins[2] = {a, b};
  const int64_t ch[2] = {ca, cb};
  return concat_impl(8, stream, ins, dtype, out, rows, 2, ch, &c0, &c1, false);
}
int npu_qdq_concat_a16(void* stream, const void* a, npu_dtype dtype, const void* b, void* out,
                       int64_t rows, int64_t ca, int64_t cb, float c0, float c1) {
  const void* ins[2] = {a, b};
  const int64_t ch[2] = {ca, cb};
  return concat_impl(16, stream, ins, dtype, out, rows, 2, ch, &c0, &c1, false);
}
int npu_qdq_concat_multi_c_fusion_a8(void* stream, const void* const* ins, npu_dtype dtype,
                                     void* out, int64_t rows, int n_in,
                                     const int64_t* channels, float c0, float c1) {
  return concat_impl(8, stream, ins, dtype, out, rows, n_in, channels, &c0, &c1, false);
}
int npu_qdq_concat_multi_c_fusion_a16(void* stream, const void* const* ins, npu_dtype dtype,
                                      void* out, int64_t rows, int n_in,
                                      const int64_t* channels, float c0, float c1) {
  return concat_impl(16, stream, ins, dtype, out, rows, n_in, channels, &c0, &c1, false);
}
int npu_qdq_concat_multi_c_fusion(void* stream, const void* const* ins, npu_dtype dtype,
                                  void* out, int64_t rows, int n_in, const int64_t* channels,
                                  const float* c0, const float* c1) {
  return concat_impl(is_i8(dtype) ? 8 : 16, stream, ins, dtype, out, rows, n_in, channels, c0, c1,
                     true);
}

int npu_qdq_gather_element_2din_2dindices_axis0(void* stream, const void* in,
                                                npu_dtype in_type, const void* idx, void* out,
                                                npu_dtype out_type, int64_t M_in, int64_t N,
                                                int64_t M_out, float c1, float c0) {
  if (!is_small_int(in_type) || !is_small_int(out_type) || any_null(in, idx, out) || N < 0 ||
      M_in < 0 || M_out < 0)
    return -1;
  return launch(stream, M_out * N, gather_axis0_kernel, in, in_type,
                static_cast<const int32_t*>(idx), out, out_type, M_in, N, c1, c0);
}
int npu_qdq_gridsample_nearest_a8(void* stream, const void* in, npu_dtype dtype,
                                  const void* grid, void* out, int64_t H, int64_t W, int64_t C,
                                  int64_t Ho, int64_t Wo, int align_corners, int padding_mode,
                                  const npu_gridsample_qdq* coeff) {
  if (!is_i8(dtype)) return -1;
  return gridsample_impl(stream, in, dtype, grid, NPU_U16, out, H, W, C, Ho, Wo, align_corners,
                         padding_mode, coeff);
}
int npu_qdq_gridsample_nearest_a16(void* stream, const void* in, npu_dtype dtype,
                                   const void* grid, void* out, int64_t H, int64_t W, int64_t C,
                                   int64_t Ho, int64_t Wo, int align_corners, int padding_mode,
                                   const npu_gridsample_qdq* coeff) {
  if (!is_i16(dtype)) return -1;
  return gridsample_impl(stream, in, dtype, grid, NPU_U16, out, H, W, C, Ho, Wo, align_corners,
                         padding_mode, coeff);
}
int npu_qdq_gridsample_nearest_a8w8(void* stream, const void* in, npu_dtype dtype,
                                    const void* grid, npu_dtype grid_type, void* out, int64_t H,
                                    int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                                    int align_corners, int padding_mode,
                                    const npu_gridsample_qdq* coeff) {
  if (!is_i8(dtype) || !is_i8(grid_type)) return -1;
  return gridsample_impl(stream, in, dtype, grid, grid_type, out, H, W, C, Ho, Wo,
                         align_corners, padding_mode, coeff);
}
int npu_qdq_resize_nearest(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                           int64_t W, int64_t C, int64_t Ho, int64_t Wo, const void* src_y,
                           const void* src_x, int requantize, float c0, float c1) {
  if (!is_small_int(dtype) || any_null(in, out, src_y, src_x) || H <= 0 || W <= 0 || C < 0 ||
      Ho < 0 || Wo < 0 || (requantize != 0 && requantize != 1))
    return -1;
  return launch(stream, Ho * Wo * C, resize_nearest_kernel, in, dtype, out, H, W, C, Wo,
                static_cast<const uint16_t*>(src_y), static_cast<const uint16_t*>(src_x),
                requantize, c0, c1);
}

int npu_qdq_reshape_innermost_requantize_fused(void* stream, const void* in, npu_dtype dtype,
                                               void* out, int64_t n, float c1, float c0) {
  if (!is_small_int(dtype)) return -1;
  return requant_inplace(npu_generic_reshape_innermost(stream, in, dtype, out, n), stream, out,
                         dtype, n, c1, c0);
}
int npu_qdq_slice_innermost_requantize_fused(void* stream, const void* in, npu_dtype dtype,
                                             void* out, int64_t rows, int64_t c, int64_t start,
                                             int64_t end, int64_t stride, float c1, float c0) {
  if (!is_small_int(dtype)) return -1;
  const int rc =
      npu_generic_slice_innermost(stream, in, dtype, out, rows, c, start, end, stride);
  if (rc) return rc;
  const int64_t w = end > start ? (end - start + stride - 1) / stride : 0;
  return requant_inplace(0, stream, out, dtype, rows * w, c1, c0);
}
int npu_qdq_slice_16b_inner_range_requantize_fused(void* stream, const void* in,
                                                   npu_dtype dtype, void* out, int64_t rows,
                                                   int64_t c, int64_t start, int64_t end,
                                                   float c1, float c0) {
  if (dtype != NPU_I8 && dtype != NPU_I16) return -1;
  const int rc = npu_generic_slice_16b_inner_range(stream, in, dtype, out, rows, c, start, end);
  return requant_inplace(rc, stream, out, dtype, end > start ? rows * (end - start) : 0, c1, c0);
}
int npu_qdq_slice_16b_inner_requantize_fused(void* stream, const void* in, npu_dtype dtype,
                                             void* out0, void* out1, int64_t rows, int64_t c,
                                             int64_t split, float c1, float c0) {
  if (dtype != NPU_I8 && dtype != NPU_I16) return -1;
  int rc = npu_generic_slice_16b_inner(stream, in, dtype, out0, out1, rows, c, split);
  rc = requant_inplace(rc, stream, out0, dtype, rows * split, c1, c0);
  return requant_inplace(rc, stream, out1, dtype, rows * (c - split), c1, c0);
}
int npu_qdq_slice_16b_c6463_requantize_fused(void* stream, const void* in, npu_dtype dtype,
                                             void* out0, void* out1, int64_t rows, float c1,
                                             float c0) {
  if (!is_i16(dtype)) return -1;
  int rc = npu_generic_slice_16b_c6463(stream, in, dtype, out0, out1, rows);
  rc = requant_inplace(rc, stream, out0, dtype, rows * 63, c1, c0);
  return requant_inplace(rc, stream, out1, dtype, rows, c1, c0);
}
int npu_qdq_transpose_k8n8_requantize_fused(void* stream, const void* in, npu_dtype dtype,
                                            void* out, int64_t K, int64_t N, float c1,
                                            float c0) {
  if (!is_small_int(dtype)) return -1;
  return requant_inplace(npu_generic_transpose_k8n8(stream, in, dtype, out, K, N), stream, out,
                         dtype, K * N, c1, c0);
}
int npu_qdq_transpose_unaligned_requantize_fused(void* stream, const void* in, npu_dtype dtype,
                                                 void* out, const int64_t* shape,
                                                 const int* perm, float c1, float c0) {
  if (!is_small_int(dtype) || !shape) return -1;
  const int rc = npu_generic_transpose_unaligned(stream, in, dtype, out, shape, perm);
  return requant_inplace(rc, stream, out, dtype, shape[0] * shape[1] * shape[2] * shape[3], c1,
                         c0);
}

}  // extern "C"
