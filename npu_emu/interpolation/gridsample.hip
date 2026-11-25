/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for gridsample.h. One thread per output element.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/grid.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/mmac.h"
#include "common/quant.h"
#include "interpolation/gridsample.h"

using namespace npu_emu;

namespace {

struct Shape {
  int64_t H, W, C;
};

// Clamped neighbour index k (0..3 = +0, +1, +W, +W+1) of the top-left index: the
// top-left index saturates to int16, the offset is added in int16 (wrapping),
// then the result is clamped to [0, H * W - 1].
__device__ inline int64_t neighbour(int fxc, int fyc, int k, const Shape& s) {
  const int64_t i0 = saturate(int64_t(fxc) + int64_t(fyc) * s.W, NPU_I16);
  const int64_t off = wrap(k == 0 ? 0 : k == 1 ? 1 : k == 2 ? s.W : s.W + 1, NPU_I16);
  return clamp64(wrap(i0 + off, NPU_I16), 0, s.H * s.W - 1);
}

// The four bilinear coefficients, all in T.
__device__ inline void coeffs_core(float gx, float gy, const Shape& s, npu_dtype t, int* fxc,
                                  int* fyc, float c[4]) {
  auto to_t = [t](float v) { return round_float(v, t, kHalfEven); };
  const int fx = int(float_to_int(gx, NPU_I16, 0, kFloor));
  const int fy = int(float_to_int(gy, NPU_I16, 0, kFloor));
  *fxc = fx < s.W - 1 ? fx : int(s.W - 1);
  *fyc = fy < s.H - 1 ? fy : int(s.H - 1);
  // The bounds are rounded to bf16, then to T.
  const float pw0 = to_t(round_float(float(s.W), NPU_BF16, kHalfEven));
  const float ph0 = to_t(round_float(float(s.H), NPU_BF16, kHalfEven));
  const float pw1 = to_t(pw0 - 1.0f), ph1 = to_t(ph0 - 1.0f);
  const bool x0 = gx > pw0 || gx < 0.0f, x1 = gx > pw1 || gx < -1.0f;
  const bool y0 = gy > ph0 || gy < 0.0f, y1 = gy > ph1 || gy < -1.0f;
  const float ffx = to_t(float(*fxc)), ffy = to_t(float(*fyc));
  const float t0x = to_t(add_q(to_t(sub_q(1.0f, gx)), ffx));
  const float t1x = to_t(sub_q(gx, ffx));
  const float t0y = to_t(add_q(to_t(sub_q(1.0f, gy)), ffy));
  const float t1y = to_t(sub_q(gy, ffy));
  // The x term is replaced by +0 before the multiply.
  c[0] = to_t(mul_q(x0 || y0 ? 0.0f : t0x, t0y));
  c[1] = to_t(mul_q(x1 || y0 ? 0.0f : t1x, t0y));
  c[2] = to_t(mul_q(x0 || y1 ? 0.0f : t0x, t1y));
  c[3] = to_t(mul_q(x1 || y1 ? 0.0f : t1x, t1y));
}

// 4-tap resize: fp32 mul then three fused MACs (exact T x T products).
template <typename Load>
__device__ inline float resize(int fxc, int fyc, const float c[4], const Shape& s, int64_t ch,
                               Load load) {
  float acc = mul_q(load(neighbour(fxc, fyc, 0, s) * s.C + ch), c[0]);
  for (int k = 1; k < 4; ++k)
    acc = fma_q(load(neighbour(fxc, fyc, k, s) * s.C + ch), c[k], acc);
  return acc;
}

__global__ void gridsample_bf16_kernel(int64_t n, const uint16_t* in, const uint16_t* grid,
                                       uint16_t* out, Shape s, npu_dtype t) {
  NPU_TID;
  const int64_t ch = i % s.C, p = i / s.C;
  const float gx = load_float(grid, t, 2 * p), gy = load_float(grid, t, 2 * p + 1);
  int fxc, fyc;
  float c[4];
  coeffs_core(gx, gy, s, t, &fxc, &fyc, c);
  const float acc = resize(fxc, fyc, c, s, ch, [&](int64_t e) { return load_float(in, t, e); });
  store_float(out, t, i, acc, kHalfEven);
}

// fp8 input decoded in groups of 8 (4 pixels x one channel pair, strided in the HWC buffer): a
// NaN makes its group NaN. Scaled in fp32, then rounded to T.
__device__ inline float fp8_in_pairs(const uint8_t* in, int64_t e, const Shape& s, float scale,
                                     npu_dtype t) {
  const int64_t pix = e / s.C, pair = (e % s.C) & ~int64_t(1);
  const int64_t p0 = pix & ~int64_t(3);
  bool group_nan = false;
  for (int64_t q = p0; q < p0 + 4 && q < s.H * s.W; ++q)
    group_nan |= fp8_row_nan(in + q * s.C + pair, 0, 2);
  return round_float(mul_q(fp8_decode_identity(in[e], group_nan), scale), t, kHalfEven);
}

__global__ void gridsample_fp8_kernel(int64_t n, const uint8_t* in, const uint16_t* grid,
                                      uint8_t* out, Shape s, npu_dtype t, float ifm_scale,
                                      float ofm_scale) {
  NPU_TID;
  const int64_t ch = i % s.C, p = i / s.C;
  const float gx = load_float(grid, t, 2 * p), gy = load_float(grid, t, 2 * p + 1);
  int fxc, fyc;
  float c[4];
  coeffs_core(gx, gy, s, t, &fxc, &fyc, c);
  const float acc = resize(fxc, fyc, c, s, ch,
                           [&](int64_t e) { return fp8_in_pairs(in, e, s, ifm_scale, t); });
  out[i] = fp8_out(acc, ofm_scale, kHalfEven);
}

struct QdqArgs {
  Shape s;
  int align_corners;
  npu_dtype t;  // interpolation type
  npu_gridsample_qdq q;
  float ifm_offset;
};

__global__ void qdq_gridsample_kernel(int64_t n, const void* in, npu_dtype dt,
                                      const uint16_t* grid, void* out, QdqArgs a) {
  NPU_TID;
  const Shape& s = a.s;
  const int64_t ch = i % s.C, p = i / s.C;
  const float gx = qdq_grid_coord(grid[2 * p], a.q, s.W, a.align_corners);
  const float gy = qdq_grid_coord(grid[2 * p + 1], a.q, s.H, a.align_corners);
  const int fx = int(float_to_int(gx, NPU_I16, 0, kFloor));
  const int fy = int(float_to_int(gy, NPU_I16, 0, kFloor));
  const int fxc = fx < s.W - 1 ? fx : int(s.W - 1);
  const int fyc = fy < s.H - 1 ? fy : int(s.H - 1);
  const int64_t nx = wrap(fx + 1, NPU_I16), ny = wrap(fy + 1, NPU_I16);  // int16 add
  const bool in_x0 = fx >= 0 && fx < s.W, in_x1 = nx >= 0 && nx < s.W;
  const bool in_y0 = fy >= 0 && fy < s.H, in_y1 = ny >= 0 && ny < s.H;
  const float t0x = add_q(sub_q(1.0f, gx), float(fxc));
  const float t1x = sub_q(gx, float(fxc));
  const float t0y = add_q(sub_q(1.0f, gy), float(fyc));
  const float t1y = sub_q(gy, float(fyc));
  float c[4];
  c[0] = in_x0 && in_y0 ? round_float(mul_q(t0x, t0y), a.t, kHalfEven) : 0.0f;
  c[1] = in_x1 && in_y0 ? round_float(mul_q(t1x, t0y), a.t, kHalfEven) : 0.0f;
  c[2] = in_x0 && in_y1 ? round_float(mul_q(t0x, t1y), a.t, kHalfEven) : 0.0f;
  c[3] = in_x1 && in_y1 ? round_float(mul_q(t1x, t1y), a.t, kHalfEven) : 0.0f;
  const float acc = resize(fxc, fyc, c, s, ch, [&](int64_t e) {
    return round_float(dequant(load_int(in, dt, e), a.q.ifm_scale, a.ifm_offset), a.t,
                       kHalfEven);
  });
  store_int(out, dt, i, quant(acc, a.q.ofm_scale, float(a.q.ofm_zp), dt));
}

bool shape_ok(int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo) {
  return H > 0 && W > 0 && H <= 32767 && W <= 32767 && H * W <= 32768 && C > 0 && C % 2 == 0 &&
         Ho >= 0 && Wo >= 0;
}

int qdq_impl(void* stream, const void* in, npu_dtype in_type, const void* grid, void* out,
             int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo, int align_corners,
             npu_dtype compute_type, const npu_gridsample_qdq* coeff) {
  if (any_null(in, grid, out, coeff) || !shape_ok(H, W, C, Ho, Wo) ||
      (align_corners != 0 && align_corners != 1))
    return -1;
  if (Ho == 0 || Wo == 0) return 0;
  QdqArgs a{{H, W, C}, align_corners, compute_type, *coeff,
            dq_offset(float(coeff->ifm_zp), coeff->ifm_scale)};
  return launch(stream, Ho * Wo * C, qdq_gridsample_kernel, in, in_type,
                static_cast<const uint16_t*>(grid), out, a);
}

// ---- grid from an int8 flow, fused with its ReduceL2 ----

// Pixel coordinate on one axis: Add + Q, Div + Q, denormalize.
__device__ inline int64_t flow_coord(int64_t T, int64_t f, int ka, int64_t mq, int msh, int64_t a,
                                     int64_t b, int shift, int64_t len, bool border) {
  const npu_dtype t = border ? NPU_I16 : NPU_I8;
  const int64_t e = srs(T * (int64_t(1) << ka) + f, ka, t);
  const int64_t r = srs(srs(e * mq, msh, t) * a + b, shift, NPU_I16);
  if (!border) return r;
  // The upper bound is the pixel of the largest int8 code.
  const int64_t g_hi = saturate(srs(127 * mq, msh, NPU_I16), NPU_I8);
  const int64_t hi = clamp64(srs(g_hi * a + b, shift, NPU_I16), 0, len - 1);
  return clamp64(r, 0, hi);
}

// round(sqrt(sq)) by a 7-step search on v = 2k - 1, then the mode epilogue.
__device__ inline int64_t flow_norm(int64_t fx, int64_t fy, const npu_gridsample_flow& p) {
  const int64_t sq = saturate(fx * fx + fy * fy, NPU_I16);
  const int cmp_sh = p.norm_mode == 2 ? 0 : 2;
  int64_t v = -1;
  for (int st = 6; st >= 0; --st) {
    const int64_t c = v + (2 << st);
    if (srs(c * c, cmp_sh, NPU_I16) < sq) v = c;
  }
  int64_t k = srs(v + 1, 1, NPU_I16);
  if (p.norm_mode == 0) return k;
  if (p.norm_mode == 2) {
    // The search ran on v^2 < sq: a tie sq == (v + 2)^2 goes to even.
    const int64_t t = v + 2 < 181 ? v + 2 : 181;
    if (saturate(t * t, NPU_I16) == sq && (v & 3) == 1) ++k;
  }
  return saturate(srs(k * p.n_mq, p.n_msh, NPU_I16), NPU_I8);
}

struct FlowArgs {
  int64_t H, W, Ho, Wo;
  bool border, f_up, fast;
  npu_gridsample_flow p;
  npu_gridsample_qdq q;
  float ifm_offset;
};

__global__ void gridsample_flow_kernel(int64_t n, const int8_t* in, const int8_t* fa,
                                       const int8_t* fc, const int8_t* tx, const int8_t* ty,
                                       int8_t* out, FlowArgs a) {
  NPU_TID;
  const npu_gridsample_flow& p = a.p;
  const int64_t h = i / a.Wo, w = i % a.Wo;
  const int64_t fi = a.f_up ? (h / 2) * (a.Wo / 2) + w / 2 : i;
  int64_t f[2];
  for (int k = 0; k < 2; ++k) {
    f[k] = fa[4 * fi + k];
    if (fc) {
      const int64_t m = saturate(srs(f[k] * p.p_mq, p.p_msh, NPU_I16), NPU_I8);
      f[k] = saturate(srs(m * p.p_am + int64_t(fc[4 * fi + k]) * p.p_ac, p.p_ash, NPU_I16),
                      NPU_I8);
    }
  }
  const int64_t rx = flow_coord(tx[w], f[0], p.ka, p.mq_x, p.msh_x, p.ax, p.bx, p.shift, a.W,
                                a.border);
  const int64_t ry = flow_coord(ty[h], f[1], p.ka, p.mq_y, p.msh_y, p.ay, p.by, p.shift, a.H,
                                a.border);
  const bool valid = rx >= 0 && rx < a.W && ry >= 0 && ry < a.H;
  const int64_t pix = clamp64(rx, 0, a.W - 1) + clamp64(ry, 0, a.H - 1) * a.W;
  int8_t* o = out + 8 * i;
  for (int k = 0; k < 4; ++k) {
    const int64_t x = in[4 * pix + k];
    if (a.fast) {
      o[k] = int8_t(srs(x, p.out_shift, NPU_I8));
    } else {
      const float v = round_float(
          mul_q(dequant(x, a.q.ifm_scale, a.ifm_offset), valid ? 1.0f : 0.0f), NPU_BF16,
          kHalfEven);
      o[k] = int8_t(quant(v, a.q.ofm_scale, float(int16_t(a.q.ofm_zp)), NPU_I8));
    }
  }
  const uint32_t nb = uint32_t(int32_t(flow_norm(f[0], f[1], p)));
  for (int k = 0; k < 4; ++k) o[4 + k] = int8_t(nb >> (8 * k));
}

bool fits16(int64_t v) { return v >= -32768 && v <= 32767; }
bool shift_ok(int64_t s) { return s >= 0 && s <= 31; }

int flow_impl(void* stream, const void* in, const void* fa, const void* fc, const void* tx,
              const void* ty, void* out, int64_t H, int64_t W, int64_t Ho, int64_t Wo,
              int padding_mode, int f_up, const npu_gridsample_qdq* coeff,
              const npu_gridsample_flow* p) {
  if (any_null(in, fa, tx, ty, out, coeff, p) || H < 1 || W < 1 || H > 32767 || W > 32767 ||
      Ho < 0 || (Ho > 0 && Ho < 32) || Ho % 8 != 0 || Wo < 0 || Wo % 4 != 0 ||
      (padding_mode != 0 && padding_mode != 1) ||
      (f_up != 0 && f_up != 1))
    return -1;
  for (int64_t m : {p->mq_x, p->mq_y, p->ax, p->ay, p->n_mq, p->p_mq, p->p_am, p->p_ac})
    if (!fits16(m)) return -1;
  for (int64_t s : {p->ka, p->msh_x, p->msh_y, p->shift, p->n_msh, p->p_msh, p->p_ash})
    if (!shift_ok(s)) return -1;
  if ((p->out_shift != -1 && !shift_ok(p->out_shift)) || p->norm_mode < 0 || p->norm_mode > 2)
    return -1;
  if (Ho == 0 || Wo == 0) return 0;
  // The zero point is negated in int16 (no -0, -32768 stays), unlike dq_offset.
  FlowArgs a{H, W, Ho, Wo, padding_mode == 1, f_up == 1, padding_mode == 1 && p->out_shift >= 0,
             *p, *coeff,
             mul_q(float(int16_t(-coeff->ifm_zp)), coeff->ifm_scale)};
  return launch(stream, Ho * Wo, gridsample_flow_kernel, static_cast<const int8_t*>(in),
                static_cast<const int8_t*>(fa), static_cast<const int8_t*>(fc),
                static_cast<const int8_t*>(tx), static_cast<const int8_t*>(ty),
                static_cast<int8_t*>(out), a);
}

}  // namespace

extern "C" {

int npu_qdq_Gridsample_ReduceL2_Fused_a8(void* stream, const void* in, const void* flow,
                                         const void* tx, const void* ty, void* out, int64_t H,
                                         int64_t W, int64_t Ho, int64_t Wo, int padding_mode,
                                         int f_up, const npu_gridsample_qdq* coeff,
                                         const npu_gridsample_flow* p) {
  return flow_impl(stream, in, flow, nullptr, tx, ty, out, H, W, Ho, Wo, padding_mode, f_up,
                   coeff, p);
}

int npu_qdq_Resize_Gridsample_ReduceL2_Fused_a8(void* stream, const void* in, const void* a,
                                                const void* c, const void* tx, const void* ty,
                                                void* out, int64_t H, int64_t W, int64_t Ho,
                                                int64_t Wo, int padding_mode, int f_up,
                                                const npu_gridsample_qdq* coeff,
                                                const npu_gridsample_flow* p) {
  if (!c) return -1;
  return flow_impl(stream, in, a, c, tx, ty, out, H, W, Ho, Wo, padding_mode, f_up, coeff, p);
}

int npu_gridsample_bf16(void* stream, const void* in, npu_dtype in_type, const void* grid,
                        void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo) {
  if (!is_half_float(in_type) || any_null(in, grid, out) || !shape_ok(H, W, C, Ho, Wo))
    return -1;
  if (Ho == 0 || Wo == 0) return 0;
  return launch(stream, Ho * Wo * C, gridsample_bf16_kernel, static_cast<const uint16_t*>(in),
                static_cast<const uint16_t*>(grid), static_cast<uint16_t*>(out),
                Shape{H, W, C}, in_type);
}

int npu_gridsample_fp8(void* stream, const void* in, const void* grid, npu_dtype grid_type,
                       void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                       float ifm_scale, float ofm_scale) {
  if (!is_half_float(grid_type) || any_null(in, grid, out) || !shape_ok(H, W, C, Ho, Wo))
    return -1;
  if (Ho == 0 || Wo == 0) return 0;
  return launch(stream, Ho * Wo * C, gridsample_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<const uint16_t*>(grid), static_cast<uint8_t*>(out),
                Shape{H, W, C}, grid_type, ifm_scale, ofm_scale);
}

int npu_qdq_gridsample_a8(void* stream, const void* in, npu_dtype in_type, const void* grid,
                          void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                          int align_corners, const npu_gridsample_qdq* coeff) {
  if (!is_i8(in_type)) return -1;
  return qdq_impl(stream, in, in_type, grid, out, H, W, C, Ho, Wo, align_corners, NPU_BF16,
                  coeff);
}

int npu_qdq_gridsample_a16(void* stream, const void* in, npu_dtype in_type, const void* grid,
                           void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                           int align_corners, npu_dtype compute_type,
                           const npu_gridsample_qdq* coeff) {
  if (!is_i16(in_type) || !is_half_float(compute_type)) return -1;
  return qdq_impl(stream, in, in_type, grid, out, H, W, C, Ho, Wo, align_corners, compute_type,
                  coeff);
}

}  // extern "C"
