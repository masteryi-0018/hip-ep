/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for include/npu_emu_qdq.h. One thread per output
 * element; these are reference-speed kernels.
 */
#include <hip/hip_runtime.h>

#include "npu_emu.h"
#include "npu_emu_device.h"
#include "npu_emu_launch.h"

using namespace npu_emu;

// The kernels round every fp32 operation separately; fused multiply-adds are
// written out (fmaf / mac_exact) where the NPU fuses.
#pragma clang fp contract(off)

namespace {

// mac_elem_32(int32 x, scale, offset) followed by to_vector_sign / to_fixed:
// the integer enters the MAC exactly and the fp32 result is rounded once.
__device__ inline int64_t requant(int64_t x, float scale, float offset, npu_dtype t,
                                  int shift = 0, IntRnd mode = kIntEven) {
  return to_int_sat(mac_exact(x, scale, offset), t, shift, mode);
}

// Index of a broadcast operand of shape [bH][bW][bC] for output (h, w, c).
__device__ inline int64_t bcast3(int64_t h, int64_t w, int64_t c, int64_t bH, int64_t bW,
                                 int64_t bC) {
  return ((bH == 1 ? 0 : h) * bW + (bW == 1 ? 0 : w)) * bC + (bC == 1 ? 0 : c);
}

// Raw element copy of 1 / 2 / 4-byte values.
__device__ inline uint32_t load_bits(const void* p, int bytes, int64_t i) {
  if (bytes == 1) return static_cast<const uint8_t*>(p)[i];
  if (bytes == 2) return static_cast<const uint16_t*>(p)[i];
  return static_cast<const uint32_t*>(p)[i];
}
__device__ inline void store_bits(void* p, int bytes, int64_t i, uint32_t v) {
  if (bytes == 1)
    static_cast<uint8_t*>(p)[i] = uint8_t(v);
  else if (bytes == 2)
    static_cast<uint16_t*>(p)[i] = uint16_t(v);
  else
    static_cast<uint32_t*>(p)[i] = v;
}

// ---- A. elementwise unary ----------------------------------------------------

// |c1*x + c0| + zp computed as the kernel does: the larger of two MACs whose
// addends c0 + zp and zp - c0 were rounded to fp32 beforehand.
__global__ void abs_kernel(int64_t n, const void* in, npu_dtype it, void* out, npu_dtype ot,
                           float c1, float c0_plus_zp, float zp_minus_c0) {
  NPU_TID;
  const int64_t x = load_int(in, it, i);
  const float p = mac_exact(x, c1, c0_plus_zp);
  const float q = mac_exact(x, -c1, zp_minus_c0);
  store_int(out, ot, i, to_int_sat(p > q ? p : q, ot, 0, kIntEven));
}

__global__ void affine_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                              npu_dtype ot, float scale, float offset) {
  NPU_TID;
  store_int(out, ot, i, requant(load_int(in, it, i), scale, offset, ot));
}

__global__ void clip_kernel(int64_t n, const void* in, npu_dtype it, void* out, npu_dtype ot,
                            float c1, float c0, float lo, float hi) {
  NPU_TID;
  float v = mac_exact(load_int(in, it, i), c1, c0);
  v = v < lo ? lo : v;  // aie::clamp
  v = v > hi ? hi : v;
  store_int(out, ot, i, to_int_sat(v, ot, 0, kIntEven));
}

// ---- B. elementwise binary ---------------------------------------------------

enum CmpOp { kEqual = 0, kGreater = 1 };

__global__ void compare_kernel(int64_t n, const void* a, npu_dtype at, const void* b,
                               npu_dtype bt, int8_t* out, float sa, float sb, float za,
                               float zb, CmpOp op) {
  NPU_TID;
  const float fa = mac_exact(load_int(a, at, i), sa, za);
  const float fb = mac_exact(load_int(b, bt, i), sb, zb);
  out[i] = int8_t(op == kEqual ? fa == fb : fa > fb);
}

// Dequantize a and b (b broadcast over [H][W][C] when bH / bW / bC are 1),
// take the max or min, requantize with a float MAC. H == 0 means no
// broadcasting (flat).
struct MinMaxArgs {
  float sa, sb, so, za, zb, zo;
  int64_t W, C, bH, bW, bC;
  bool is_max, bcast;
};

__global__ void minmax_kernel(int64_t n, const void* a, npu_dtype at, const void* b,
                              npu_dtype bt, void* out, npu_dtype ot, MinMaxArgs g) {
  NPU_TID;
  int64_t j = i;
  if (g.bcast) {
    const int64_t c = i % g.C, w = (i / g.C) % g.W, h = i / (g.C * g.W);
    j = bcast3(h, w, c, g.bH, g.bW, g.bC);
  }
  const float fa = mac_exact(load_int(a, at, i), g.sa, g.za);
  const float fb = mac_exact(load_int(b, bt, j), g.sb, g.zb);
  const float m = g.is_max ? (fa > fb ? fa : fb) : (fa < fb ? fa : fb);
  store_int(out, ot, i, to_int_sat(fmaf(m, g.so, g.zo), ot, 0, kIntEven));
}

// ---- C. where ----------------------------------------------------------------

struct WhereArgs {
  int64_t W, C, cH, cW, cC;
  int scalar_position;
};

__device__ inline bool where_take_x(const void* cond, npu_dtype ct, int64_t i, WhereArgs g) {
  const int64_t c = i % g.C, w = (i / g.C) % g.W, h = i / (g.C * g.W);
  const bool nz = load_int(cond, ct, bcast3(h, w, c, g.cH, g.cW, g.cC)) != 0;
  return g.scalar_position ? nz : !nz;
}

__global__ void where_kernel(int64_t n, const void* cond, const void* x, void* out, int bytes,
                             uint32_t y_bits, WhereArgs g) {
  NPU_TID;
  store_bits(out, bytes, i, where_take_x(cond, NPU_I8, i, g) ? load_bits(x, bytes, i) : y_bits);
}

__global__ void qdq_where_kernel(int64_t n, const void* cond, npu_dtype ct, const void* x,
                                 npu_dtype dt, void* out, int32_t y, WhereArgs g, float scale,
                                 float offset) {
  NPU_TID;
  const int64_t v = where_take_x(cond, ct, i, g) ? load_int(x, dt, i) : int64_t(y);
  store_int(out, dt, i, requant(v, scale, offset, dt));
}

// ---- D. reductions -----------------------------------------------------------

// Output (oh, ow, oc) of the reduced tensor; reduced axes have extent 1 in the
// output and are iterated here.
struct ReduceArgs {
  int64_t H, W, C;
  int axis_flags;
  bool is_max, qdq;
  float offset, scale;
  int shift;
};

__global__ void reduce_kernel(int64_t n, const void* in, npu_dtype dt, void* out,
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
        const int64_t v = load_int(in, dt, (h * g.W + w) * g.C + c);
        if (first || (g.is_max ? v > r : v < r)) r = v;
        first = false;
      }
  store_int(out, dt, i, g.qdq ? requant(r, g.scale, g.offset, dt, g.shift) : r);
}

// ---- E. pooling --------------------------------------------------------------

struct PoolArgs {
  int64_t W, C, Wo;
  int kh, kw, sh, sw;
  bool qdq;
  float offset, scale;
  int shift;
};

__global__ void maxpool_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                               npu_dtype ot, PoolArgs g) {
  NPU_TID;
  const int64_t c = i % g.C, ow = (i / g.C) % g.Wo, oh = i / (g.C * g.Wo);
  int64_t m = load_int(in, it, (oh * g.sh * g.W + ow * g.sw) * g.C + c);
  for (int y = 0; y < g.kh; ++y)
    for (int x = 0; x < g.kw; ++x) {
      const int64_t v = load_int(in, it, ((oh * g.sh + y) * g.W + ow * g.sw + x) * g.C + c);
      m = v > m ? v : m;
    }
  store_int(out, ot, i, g.qdq ? requant(m, g.scale, g.offset, ot, g.shift) : m);
}

// ---- F. concat ---------------------------------------------------------------

constexpr int kMaxConcat = 6;

struct ConcatArgs {
  const void* in[kMaxConcat];
  int64_t start[kMaxConcat + 1];  // first output channel of each input; [n_in] = total
  float scale[kMaxConcat], offset[kMaxConcat];
  int n_in;
};

__global__ void concat_kernel(int64_t n, ConcatArgs g, npu_dtype it, void* out, npu_dtype ot) {
  NPU_TID;
  const int64_t total = g.start[g.n_in];
  const int64_t r = i / total, c = i % total;
  int k = 0;
  while (c >= g.start[k + 1]) ++k;
  const int64_t ck = g.start[k + 1] - g.start[k];
  const int64_t v = load_int(g.in[k], it, r * ck + c - g.start[k]);
  store_int(out, ot, i, requant(v, g.scale[k], g.offset[k], ot));
}

// ---- G. gather / grid sample / resize ----------------------------------------

__global__ void gather_axis0_kernel(int64_t n, const void* in, npu_dtype it, const int32_t* idx,
                                    void* out, npu_dtype ot, int64_t M_in, int64_t N,
                                    float scale, float offset) {
  NPU_TID;
  const uint32_t row = uint32_t(idx[i]);  // unsigned compare, as the kernel
  if (row >= uint64_t(M_in)) return;     // no tile owns it: left unwritten
  const int64_t k = i % N;
  store_int(out, ot, i, requant(load_int(in, it, int64_t(row) * N + k), scale, offset, ot));
}

// Grid code -> rounded pixel coordinate, in the kernel's fp32 steps.
__device__ inline int64_t grid_coord(int64_t code, const npu_gridsample_qdq& q, int64_t extent,
                                     int align_corners, int padding_mode) {
  float f = (float(code) - float(q.grid_zp)) * q.grid_scale;
  f = f + 1.0f;
  float g;
  if (align_corners == 0)
    g = fmaf(f, float(extent) * 0.5f, -0.5f);  // aie::mac: one rounding
  else
    g = (float(extent - 1) * f) * 0.5f;
  int64_t r = to_int_sat(g, NPU_I16, 0, kIntEven);
  if (padding_mode == 1) r = r < 0 ? 0 : (r > extent - 1 ? extent - 1 : r);
  return r;
}

struct GridArgs {
  int64_t H, W, C, Wo;
  int align_corners, padding_mode;
  npu_gridsample_qdq q;
  float ifm_offset;  // fp32(-(float)ifm_zp * ifm_scale)
};

__global__ void gridsample_nearest_kernel(int64_t n, const void* in, npu_dtype dt,
                                          const void* grid, npu_dtype gt, void* out,
                                          GridArgs g) {
  NPU_TID;
  const int64_t c = i % g.C, p = i / g.C;  // p = output pixel
  const int64_t rx = grid_coord(load_int(grid, gt, 2 * p), g.q, g.W, g.align_corners,
                                g.padding_mode);
  const int64_t ry = grid_coord(load_int(grid, gt, 2 * p + 1), g.q, g.H, g.align_corners,
                                g.padding_mode);
  float v = 0.0f;
  if (rx >= 0 && rx < g.W && ry >= 0 && ry < g.H) {
    const float dq = mac_exact(load_int(in, dt, (ry * g.W + rx) * g.C + c), g.q.ifm_scale,
                               g.ifm_offset);
    v = bf16_to_f32(f32_to_bf16(dq, kEven));  // the bf16 band / accumulator
  }
  store_int(out, dt, i, to_int_sat(fmaf(v, g.q.ofm_scale, float(g.q.ofm_zp)), dt, 0, kIntEven));
}

__global__ void resize_nearest_kernel(int64_t n, const void* in, npu_dtype dt, void* out,
                                      int64_t W, int64_t C, int64_t Wo, const int32_t* sy,
                                      const int32_t* sx, int requantize, float scale,
                                      float offset) {
  NPU_TID;
  const int64_t c = i % C, ow = (i / C) % Wo, oh = i / (C * Wo);
  const int64_t v = load_int(in, dt, (int64_t(sy[oh]) * W + sx[ow]) * C + c);
  store_int(out, dt, i, requantize ? requant(v, scale, offset, dt, 0, kIntTrunc) : v);
}

// ---- argument checks -----------------------------------------------------------

bool is8(npu_dtype t) { return t == NPU_I8 || t == NPU_U8; }
bool is16(npu_dtype t) { return t == NPU_I16 || t == NPU_U16; }
bool width_ok(int bits, npu_dtype t) { return bits == 8 ? is8(t) : is16(t); }

int unary(int bits, const void* in, npu_dtype it, npu_dtype ot) {
  return in && width_ok(bits, it) && width_ok(bits, ot) ? 0 : -1;
}

int abs_impl(int bits, void* stream, const void* in, npu_dtype it, void* out, npu_dtype ot,
             int64_t n, float c1, float c0, float zp) {
  if (unary(bits, in, it, ot)) return -1;
  // The kernel forms both addends with scalar fp32 arithmetic.
  const float c0_plus_zp = c0 + zp, zp_minus_c0 = zp - c0;
  return launch(stream, n, abs_kernel, in, it, out, ot, c1, c0_plus_zp, zp_minus_c0);
}

int neg_impl(int bits, void* stream, const void* in, npu_dtype it, void* out, npu_dtype ot,
             int64_t n, float c1, float c0) {
  if (unary(bits, in, it, ot)) return -1;
  return launch(stream, n, affine_kernel, in, it, out, ot, c1, c0);
}

int clip_impl(int bits, void* stream, const void* in, npu_dtype it, void* out, npu_dtype ot,
              int64_t n, float c1, float c0, float lo, float hi) {
  if (unary(bits, in, it, ot)) return -1;
  return launch(stream, n, clip_kernel, in, it, out, ot, c1, c0, lo, hi);
}

int compare_impl(int bits, void* stream, const void* a, npu_dtype at, const void* b,
                 npu_dtype bt, void* out, int64_t n, float sa, float sb, float za, float zb,
                 CmpOp op) {
  if (!width_ok(bits, at) || !width_ok(bits, bt)) return -1;
  return launch(stream, n, compare_kernel, a, at, b, bt, static_cast<int8_t*>(out), sa, sb, za,
                zb, op);
}

int max_impl(int bits, void* stream, const void* a, npu_dtype at, const void* b, npu_dtype bt,
             void* out, npu_dtype ot, int64_t n, float sa, float sb, float so, float za,
             float zb, float zo) {
  if (!width_ok(bits, at) || !width_ok(bits, bt) || !width_ok(bits, ot)) return -1;
  MinMaxArgs g{sa, sb, so, za, zb, zo, 1, 1, 1, 1, 1, true, false};
  return launch(stream, n, minmax_kernel, a, at, b, bt, out, ot, g);
}

bool bcast_ok(int64_t full, int64_t b) { return b == full || b == 1; }

int where_impl(int bytes, void* stream, const void* cond, const void* x, void* out, int64_t H,
               int64_t W, int64_t C, int64_t cH, int64_t cW, int64_t cC, uint32_t y_bits,
               int scalar_position) {
  if (!bcast_ok(H, cH) || !bcast_ok(W, cW) || !bcast_ok(C, cC) || W <= 0 || C <= 0) return -1;
  WhereArgs g{W, C, cH, cW, cC, scalar_position != 0};
  return launch(stream, H * W * C, where_kernel, cond, x, out, bytes, y_bits, g);
}

int qdq_where_impl(int bits, void* stream, const void* cond, const void* x, npu_dtype dt,
                   void* out, int64_t H, int64_t W, int64_t C, int64_t cH, int64_t cW,
                   int64_t cC, int32_t y, int scalar_position, float scale, float offset) {
  if (!width_ok(bits, dt) || !bcast_ok(H, cH) || !bcast_ok(W, cW) || !bcast_ok(C, cC) ||
      W <= 0 || C <= 0)
    return -1;
  // The scalar arrives as the low bits of an int32 word, read as dtype.
  int32_t yv = y;
  switch (dt) {
    case NPU_I8: yv = int8_t(y); break;
    case NPU_U8: yv = uint8_t(y); break;
    case NPU_I16: yv = int16_t(y); break;
    default: yv = uint16_t(y); break;
  }
  WhereArgs g{W, C, cH, cW, cC, scalar_position != 0};
  return launch(stream, H * W * C, qdq_where_kernel, cond, bits == 8 ? NPU_I8 : NPU_I16, x, dt,
                out, yv, g, scale, offset);
}

int min_impl(int bits, void* stream, const void* a, const void* b, npu_dtype dt, void* out,
             int64_t H, int64_t W, int64_t C, int64_t bH, int64_t bW, int64_t bC, float sa,
             float sb, float so, float za, float zb, float zo) {
  if (!width_ok(bits, dt) || !bcast_ok(H, bH) || !bcast_ok(W, bW) || !bcast_ok(C, bC) ||
      W <= 0 || C <= 0)
    return -1;
  MinMaxArgs g{sa, sb, so, za, zb, zo, W, C, bH, bW, bC, false, true};
  return launch(stream, H * W * C, minmax_kernel, a, dt, b, dt, out, dt, g);
}

int reduce_impl(int bits, void* stream, const void* in, npu_dtype dt, void* out, int64_t H,
                int64_t W, int64_t C, int axis_flags, bool is_max, bool qdq, float offset,
                float scale, int shift) {
  if (!width_ok(bits, dt) || axis_flags < 1 || axis_flags > 7 || H <= 0 || W <= 0 || C <= 0)
    return -1;
  ReduceArgs g{H, W, C, axis_flags, is_max, qdq, offset, scale, shift};
  const int64_t n = (axis_flags & 4 ? 1 : H) * (axis_flags & 2 ? 1 : W) * (axis_flags & 1 ? 1 : C);
  return launch(stream, n, reduce_kernel, in, dt, out, g);
}

int pool_impl(int bits, void* stream, const void* in, npu_dtype it, void* out, npu_dtype ot,
              int64_t H, int64_t W, int64_t C, int kh, int kw, int sh, int sw, bool qdq,
              float offset, float scale, int shift) {
  if (!width_ok(bits, it) || !width_ok(bits, ot) || kh < 1 || kw < 1 || sh < 1 || sw < 1 ||
      kh > H || kw > W || C <= 0)
    return -1;
  const int64_t Ho = (H - kh) / sh + 1, Wo = (W - kw) / sw + 1;
  PoolArgs g{W, C, Wo, kh, kw, sh, sw, qdq, offset, scale, shift};
  return launch(stream, Ho * Wo * C, maxpool_kernel, in, it, out, ot, g);
}

int concat_impl(int bits, void* stream, int n_in, const void* const* ins, npu_dtype it,
                void* out, npu_dtype ot, int64_t rows, const int64_t* channels,
                const float* scale, const float* offset, bool per_input) {
  if (!width_ok(bits, it) || !width_ok(bits, ot) || n_in < 1 || n_in > kMaxConcat || !ins ||
      !channels)
    return -1;
  ConcatArgs g{};
  g.n_in = n_in;
  for (int k = 0; k < n_in; ++k) {
    if (channels[k] < 0) return -1;
    g.in[k] = ins[k];
    g.start[k + 1] = g.start[k] + channels[k];
    g.scale[k] = per_input ? scale[k] : scale[0];
    g.offset[k] = per_input ? offset[k] : offset[0];
  }
  if (g.start[n_in] <= 0) return 0;
  return launch(stream, rows * g.start[n_in], concat_kernel, g, it, out, ot);
}

int gridsample_impl(void* stream, const void* in, npu_dtype dt, const void* grid, npu_dtype gt,
                    void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                    int align_corners, int padding_mode, const npu_gridsample_qdq* coeff) {
  if (!coeff || H <= 0 || W <= 0 || C <= 0 || (align_corners != 0 && align_corners != 1) ||
      (padding_mode != 0 && padding_mode != 1))
    return -1;
  GridArgs g{H, W, C, Wo, align_corners, padding_mode, *coeff,
             -float(coeff->ifm_zp) * coeff->ifm_scale};
  return launch(stream, Ho * Wo * C, gridsample_nearest_kernel, in, dt, grid, gt, out, g);
}

// The fused wrappers run dqq<T, T> in place on the moved output; on the same
// stream the requantization sees the finished movement.
int requant_inplace(int rc, void* stream, void* p, npu_dtype t, int64_t n, float c1,
                    float c0) {
  if (rc) return rc;
  return launch(stream, n, affine_kernel, static_cast<const void*>(p), t, p, t, c1, c0);
}

}  // namespace

extern "C" {

// ---- A ------------------------------------------------------------------------

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

// ---- B ------------------------------------------------------------------------

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
int npu_elementwise_max_a8(void* stream, const void* a, npu_dtype a_type, const void* b,
                           npu_dtype b_type, void* out, npu_dtype out_type, int64_t n,
                           float scale_a, float scale_b, float scale_o, float zp_a, float zp_b,
                           float zp_o) {
  return max_impl(8, stream, a, a_type, b, b_type, out, out_type, n, scale_a, scale_b, scale_o,
                  zp_a, zp_b, zp_o);
}
int npu_elementwise_max_a16(void* stream, const void* a, npu_dtype a_type, const void* b,
                            npu_dtype b_type, void* out, npu_dtype out_type, int64_t n,
                            float scale_a, float scale_b, float scale_o, float zp_a,
                            float zp_b, float zp_o) {
  return max_impl(16, stream, a, a_type, b, b_type, out, out_type, n, scale_a, scale_b,
                  scale_o, zp_a, zp_b, zp_o);
}

// ---- C ------------------------------------------------------------------------

int npu_elementwise_broadcasting3d_where_a8(void* stream, const void* cond, const void* x,
                                            void* out, int64_t H, int64_t W, int64_t C,
                                            int64_t cH, int64_t cW, int64_t cC,
                                            uint32_t y_bits, int scalar_position) {
  return where_impl(1, stream, cond, x, out, H, W, C, cH, cW, cC, y_bits & 0xff,
                    scalar_position);
}
int npu_elementwise_broadcasting3d_where_a16(void* stream, const void* cond, const void* x,
                                             void* out, int64_t H, int64_t W, int64_t C,
                                             int64_t cH, int64_t cW, int64_t cC,
                                             uint32_t y_bits, int scalar_position) {
  return where_impl(2, stream, cond, x, out, H, W, C, cH, cW, cC, y_bits & 0xffff,
                    scalar_position);
}
int npu_qdq_elementwise_broadcasting3d_where_a8(void* stream, const void* cond, const void* x,
                                                npu_dtype dtype, void* out, int64_t H,
                                                int64_t W, int64_t C, int64_t cH, int64_t cW,
                                                int64_t cC, int32_t y, int scalar_position,
                                                float scale, float offset) {
  return qdq_where_impl(8, stream, cond, x, dtype, out, H, W, C, cH, cW, cC, y,
                        scalar_position, scale, offset);
}
int npu_qdq_elementwise_broadcasting3d_where_a16(void* stream, const void* cond, const void* x,
                                                 npu_dtype dtype, void* out, int64_t H,
                                                 int64_t W, int64_t C, int64_t cH, int64_t cW,
                                                 int64_t cC, int32_t y, int scalar_position,
                                                 float scale, float offset) {
  return qdq_where_impl(16, stream, cond, x, dtype, out, H, W, C, cH, cW, cC, y,
                        scalar_position, scale, offset);
}
int npu_qdq_elementwise_broadcasting3d_min_a8(void* stream, const void* a, const void* b,
                                              npu_dtype dtype, void* out, int64_t H, int64_t W,
                                              int64_t C, int64_t bH, int64_t bW, int64_t bC,
                                              float scale_a, float scale_b, float scale_o,
                                              float zp_a, float zp_b, float zp_o) {
  return min_impl(8, stream, a, b, dtype, out, H, W, C, bH, bW, bC, scale_a, scale_b, scale_o,
                  zp_a, zp_b, zp_o);
}
int npu_qdq_elementwise_broadcasting3d_min_a16(void* stream, const void* a, const void* b,
                                               npu_dtype dtype, void* out, int64_t H,
                                               int64_t W, int64_t C, int64_t bH, int64_t bW,
                                               int64_t bC, float scale_a, float scale_b,
                                               float scale_o, float zp_a, float zp_b,
                                               float zp_o) {
  return min_impl(16, stream, a, b, dtype, out, H, W, C, bH, bW, bC, scale_a, scale_b,
                  scale_o, zp_a, zp_b, zp_o);
}

// ---- D ------------------------------------------------------------------------

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
                              int64_t H, int64_t W, int64_t C, int axis_flags, float offset,
                              float scale, int shift) {
  return reduce_impl(8, stream, in, dtype, out, H, W, C, axis_flags, true, true, offset, scale,
                     shift);
}
int npu_qdq_reduce_max_c64_a16(void* stream, const void* in, npu_dtype dtype, void* out,
                               int64_t H, int64_t W, int64_t C, int axis_flags, float offset,
                               float scale, int shift) {
  return reduce_impl(16, stream, in, dtype, out, H, W, C, axis_flags, true, true, offset, scale,
                     shift);
}
int npu_qdq_reduce_min_c64_a8(void* stream, const void* in, npu_dtype dtype, void* out,
                              int64_t H, int64_t W, int64_t C, int axis_flags, float offset,
                              float scale, int shift) {
  return reduce_impl(8, stream, in, dtype, out, H, W, C, axis_flags, false, true, offset, scale,
                     shift);
}
int npu_qdq_reduce_min_c64_a16(void* stream, const void* in, npu_dtype dtype, void* out,
                               int64_t H, int64_t W, int64_t C, int axis_flags, float offset,
                               float scale, int shift) {
  return reduce_impl(16, stream, in, dtype, out, H, W, C, axis_flags, false, true, offset,
                     scale, shift);
}

// ---- E ------------------------------------------------------------------------

int npu_qdq_maxpool_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                       npu_dtype out_type, int64_t H, int64_t W, int64_t C, int kh, int kw,
                       int sh, int sw, float offset, float scale, int shift) {
  return pool_impl(8, stream, in, in_type, out, out_type, H, W, C, kh, kw, sh, sw, true, offset,
                   scale, shift);
}
int npu_qdq_maxpool_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                        npu_dtype out_type, int64_t H, int64_t W, int64_t C, int kh, int kw,
                        int sh, int sw, float offset, float scale, int shift) {
  return pool_impl(16, stream, in, in_type, out, out_type, H, W, C, kh, kw, sh, sw, true,
                   offset, scale, shift);
}
int npu_generic_maxpool_a8(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                           int64_t W, int64_t C, int kh, int kw, int sh, int sw) {
  return pool_impl(8, stream, in, dtype, out, dtype, H, W, C, kh, kw, sh, sw, false, 0, 0, 0);
}
int npu_qdq_globalmaxpool_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                             npu_dtype out_type, int64_t H, int64_t W, int64_t C, float offset,
                             float scale, int shift) {
  if (H > INT32_MAX || W > INT32_MAX) return -1;
  return pool_impl(8, stream, in, in_type, out, out_type, H, W, C, int(H), int(W), 1, 1, true,
                   offset, scale, shift);
}
int npu_qdq_globalmaxpool_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                              npu_dtype out_type, int64_t H, int64_t W, int64_t C, float offset,
                              float scale, int shift) {
  if (H > INT32_MAX || W > INT32_MAX) return -1;
  return pool_impl(16, stream, in, in_type, out, out_type, H, W, C, int(H), int(W), 1, 1, true,
                   offset, scale, shift);
}

// ---- F ------------------------------------------------------------------------

int npu_qdq_concat_a8(void* stream, const void* a, const void* b, npu_dtype in_type, void* out,
                      npu_dtype out_type, int64_t rows, int64_t ca, int64_t cb, float scale,
                      float offset) {
  const void* ins[2] = {a, b};
  const int64_t ch[2] = {ca, cb};
  return concat_impl(8, stream, 2, ins, in_type, out, out_type, rows, ch, &scale, &offset,
                     false);
}
int npu_qdq_concat_a16(void* stream, const void* a, const void* b, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t rows, int64_t ca, int64_t cb,
                       float scale, float offset) {
  const void* ins[2] = {a, b};
  const int64_t ch[2] = {ca, cb};
  return concat_impl(16, stream, 2, ins, in_type, out, out_type, rows, ch, &scale, &offset,
                     false);
}
int npu_qdq_concat_multi_c_fusion_a8(void* stream, int n_in, const void* const* ins,
                                     npu_dtype in_type, void* out, npu_dtype out_type,
                                     int64_t rows, const int64_t* channels, float scale,
                                     float offset) {
  return concat_impl(8, stream, n_in, ins, in_type, out, out_type, rows, channels, &scale,
                     &offset, false);
}
int npu_qdq_concat_multi_c_fusion_a16(void* stream, int n_in, const void* const* ins,
                                      npu_dtype in_type, void* out, npu_dtype out_type,
                                      int64_t rows, const int64_t* channels, float scale,
                                      float offset) {
  return concat_impl(16, stream, n_in, ins, in_type, out, out_type, rows, channels, &scale,
                     &offset, false);
}
int npu_qdq_concat_multi_c_fusion(void* stream, int n_in, const void* const* ins,
                                  npu_dtype in_type, void* out, npu_dtype out_type,
                                  int64_t rows, const int64_t* channels, const float* scale,
                                  const float* offset) {
  if (!scale || !offset) return -1;
  return concat_impl(is8(in_type) ? 8 : 16, stream, n_in, ins, in_type, out, out_type, rows,
                     channels, scale, offset, true);
}

// ---- G ------------------------------------------------------------------------

int npu_qdq_gather_element_2din_2dindices_axis0(void* stream, const void* in,
                                                npu_dtype in_type, const void* idx, void* out,
                                                npu_dtype out_type, int64_t M_in, int64_t N,
                                                int64_t M_out, float scale, float offset) {
  if (!(is8(in_type) || is16(in_type)) || !(is8(out_type) || is16(out_type)) || N <= 0 ||
      M_in < 0)
    return -1;
  return launch(stream, M_out * N, gather_axis0_kernel, in, in_type,
                static_cast<const int32_t*>(idx), out, out_type, M_in, N, scale, offset);
}
int npu_qdq_gridsample_nearest_a8(void* stream, const void* in, npu_dtype dtype,
                                  const void* grid, void* out, int64_t H, int64_t W, int64_t C,
                                  int64_t Ho, int64_t Wo, int align_corners, int padding_mode,
                                  const npu_gridsample_qdq* coeff) {
  if (!is8(dtype)) return -1;
  return gridsample_impl(stream, in, dtype, grid, NPU_U16, out, H, W, C, Ho, Wo, align_corners,
                         padding_mode, coeff);
}
int npu_qdq_gridsample_nearest_a16(void* stream, const void* in, npu_dtype dtype,
                                   const void* grid, void* out, int64_t H, int64_t W, int64_t C,
                                   int64_t Ho, int64_t Wo, int align_corners, int padding_mode,
                                   const npu_gridsample_qdq* coeff) {
  if (!is16(dtype)) return -1;
  return gridsample_impl(stream, in, dtype, grid, NPU_U16, out, H, W, C, Ho, Wo, align_corners,
                         padding_mode, coeff);
}
int npu_qdq_gridsample_nearest_a8w8(void* stream, const void* in, npu_dtype dtype,
                                    const void* grid, npu_dtype grid_type, void* out, int64_t H,
                                    int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                                    int align_corners, int padding_mode,
                                    const npu_gridsample_qdq* coeff) {
  if (!is8(dtype) || !is8(grid_type)) return -1;
  return gridsample_impl(stream, in, dtype, grid, grid_type, out, H, W, C, Ho, Wo,
                         align_corners, padding_mode, coeff);
}
int npu_qdq_resize_nearest(void* stream, const void* in, npu_dtype dtype, void* out, int64_t H,
                           int64_t W, int64_t C, int64_t Ho, int64_t Wo, const int32_t* src_y,
                           const int32_t* src_x, int requantize, float scale, float offset) {
  if (!(is8(dtype) || is16(dtype)) || H <= 0 || W <= 0 || C <= 0) return -1;
  return launch(stream, Ho * Wo * C, resize_nearest_kernel, in, dtype, out, W, C, Wo, src_y,
                src_x, requantize, scale, offset);
}

// ---- H ------------------------------------------------------------------------

int npu_qdq_reshape_innermost_requantize_fused(void* stream, const void* in, void* out,
                                               npu_dtype dtype, int64_t n, float c1, float c0) {
  if (!(is8(dtype) || is16(dtype))) return -1;
  return requant_inplace(npu_generic_reshape_innermost(stream, in, out, dtype, n), stream, out,
                         dtype, n, c1, c0);
}
int npu_qdq_slice_innermost_requantize_fused(void* stream, const void* in, void* out,
                                             npu_dtype dtype, int64_t rows, int64_t c,
                                             int64_t start, int64_t end, int64_t stride,
                                             float c1, float c0) {
  if (!(is8(dtype) || is16(dtype))) return -1;
  const int rc =
      npu_generic_slice_innermost(stream, in, out, dtype, rows, c, start, end, stride);
  const int64_t w = end > start && stride > 0 ? (end - start + stride - 1) / stride : 0;
  return requant_inplace(rc, stream, out, dtype, rows * w, c1, c0);
}
int npu_qdq_slice_16b_inner_range_requantize_fused(void* stream, const void* in, void* out,
                                                   npu_dtype dtype, int64_t rows, int64_t c,
                                                   int64_t start, int64_t end, float c1,
                                                   float c0) {
  if (dtype != NPU_I8 && dtype != NPU_I16) return -1;
  const int rc = npu_generic_slice_16b_inner_range(stream, in, out, dtype, rows, c, start, end);
  return requant_inplace(rc, stream, out, dtype, end > start ? rows * (end - start) : 0, c1, c0);
}
int npu_qdq_slice_16b_inner_requantize_fused(void* stream, const void* in, void* out0,
                                             void* out1, npu_dtype dtype, int64_t rows,
                                             int64_t c, int64_t split, float c1, float c0) {
  if (dtype != NPU_I8 && dtype != NPU_I16) return -1;
  int rc = npu_generic_slice_16b_inner(stream, in, out0, out1, dtype, rows, c, split);
  rc = requant_inplace(rc, stream, out0, dtype, rows * split, c1, c0);
  return requant_inplace(rc, stream, out1, dtype, rows * (c - split), c1, c0);
}
int npu_qdq_slice_16b_c6463_requantize_fused(void* stream, const void* in, void* out0,
                                             void* out1, npu_dtype dtype, int64_t rows,
                                             float c1, float c0) {
  if (!is16(dtype)) return -1;
  int rc = npu_generic_slice_16b_c6463(stream, in, out0, out1, dtype, rows);
  rc = requant_inplace(rc, stream, out0, dtype, rows * 63, c1, c0);
  return requant_inplace(rc, stream, out1, dtype, rows, c1, c0);
}
int npu_qdq_transpose_k8n8_requantize_fused(void* stream, const void* in, void* out,
                                            npu_dtype dtype, int64_t K, int64_t N, float c1,
                                            float c0) {
  if (!(is8(dtype) || is16(dtype))) return -1;
  return requant_inplace(npu_generic_transpose_k8n8(stream, in, out, dtype, K, N), stream, out,
                         dtype, K * N, c1, c0);
}
int npu_qdq_transpose_unaligned_requantize_fused(void* stream, const void* in, void* out,
                                                 npu_dtype dtype, const int64_t* shape,
                                                 const int* perm, float c1, float c0) {
  if (!(is8(dtype) || is16(dtype)) || !shape) return -1;
  const int rc = npu_generic_transpose_unaligned(stream, in, out, dtype, shape, perm);
  return requant_inplace(rc, stream, out, dtype, shape[0] * shape[1] * shape[2] * shape[3], c1,
                         c0);
}

}  // extern "C"
