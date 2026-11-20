/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for gridsample.h. One thread per output element.
 */
#include <hip/hip_runtime.h>
#include <math.h>

#include "common/npu_emu.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/grid.h"
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

// fp8 input decoded in groups of 8 (4 pixels x one channel pair): a NaN makes
// its group NaN. Scaled in fp32, then rounded to T.
__device__ inline float fp8_in(const uint8_t* in, int64_t e, const Shape& s, float scale,
                               npu_dtype t) {
  const int64_t pix = e / s.C, pair = (e % s.C) & ~int64_t(1);
  const int64_t p0 = pix & ~int64_t(3);
  bool group_nan = false;
  for (int64_t q = p0; q < p0 + 4 && q < s.H * s.W; ++q)
    for (int k = 0; k < 2; ++k) group_nan |= (in[q * s.C + pair + k] & 0x7f) == 0x7f;
  return round_float(mul_q(fp8_decode_identity(in[e], group_nan), scale), t, kHalfEven);
}

__global__ void gridsample_fp8_kernel(int64_t n, const uint8_t* in, const uint16_t* grid,
                                      uint8_t* out, Shape s, npu_dtype t, float ifm_scale,
                                      float ofm_recip) {
  NPU_TID;
  const int64_t ch = i % s.C, p = i / s.C;
  const float gx = load_float(grid, t, 2 * p), gy = load_float(grid, t, 2 * p + 1);
  int fxc, fyc;
  float c[4];
  coeffs_core(gx, gy, s, t, &fxc, &fyc, c);
  const float acc = resize(fxc, fyc, c, s, ch,
                           [&](int64_t e) { return fp8_in(in, e, s, ifm_scale, t); });
  out[i] = f32_to_fp8(mul_q(acc, ofm_recip), kHalfEven, kSat);
}

struct QdqArgs {
  Shape s;
  int align_corners;
  npu_dtype t;  // interpolation type
  npu_gridsample_qdq q;
  float ifm_offset;  // fp32(-(float)ifm_zp * ifm_scale)
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
    return round_float(mac_exact(load_int(in, dt, e), a.q.ifm_scale, a.ifm_offset), a.t,
                       kHalfEven);
  });
  const float q = fmaf(acc, a.q.ofm_scale, float(a.q.ofm_zp));
  store_int(out, dt, i, float_to_int(q, dt, 0, kHalfEven));
}

bool shape_ok(int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo) {
  return H > 0 && W > 0 && H <= 32767 && W <= 32767 && H * W <= 32768 && C > 0 && C % 2 == 0 &&
         Ho >= 0 && Wo >= 0;
}

int qdq_impl(void* stream, const void* in, npu_dtype dt, const void* grid, void* out,
             int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo, int align_corners,
             npu_dtype t, const npu_gridsample_qdq* coeff) {
  if (!coeff || !shape_ok(H, W, C, Ho, Wo) || (align_corners != 0 && align_corners != 1))
    return -1;
  QdqArgs a{{H, W, C}, align_corners, t, *coeff,
            -float(coeff->ifm_zp) * coeff->ifm_scale};
  return launch(stream, Ho * Wo * C, qdq_gridsample_kernel, in, dt,
                static_cast<const uint16_t*>(grid), out, a);
}

}  // namespace

extern "C" {

int npu_gridsample_bf16(void* stream, const void* in, npu_dtype dtype, const void* grid,
                        void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo) {
  if ((dtype != NPU_BF16 && dtype != NPU_F16) || !shape_ok(H, W, C, Ho, Wo)) return -1;
  return launch(stream, Ho * Wo * C, gridsample_bf16_kernel, static_cast<const uint16_t*>(in),
                static_cast<const uint16_t*>(grid), static_cast<uint16_t*>(out),
                Shape{H, W, C}, dtype);
}

int npu_gridsample_fp8(void* stream, const void* in, const void* grid, npu_dtype grid_type,
                       void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                       float ifm_scale, float ofm_recip) {
  if ((grid_type != NPU_BF16 && grid_type != NPU_F16) || !shape_ok(H, W, C, Ho, Wo)) return -1;
  return launch(stream, Ho * Wo * C, gridsample_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<const uint16_t*>(grid), static_cast<uint8_t*>(out),
                Shape{H, W, C}, grid_type, ifm_scale, ofm_recip);
}

int npu_qdq_gridsample_a8(void* stream, const void* in, npu_dtype dtype, const void* grid,
                          void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                          int align_corners, const npu_gridsample_qdq* coeff) {
  if (dtype != NPU_I8 && dtype != NPU_U8) return -1;
  return qdq_impl(stream, in, dtype, grid, out, H, W, C, Ho, Wo, align_corners, NPU_BF16, coeff);
}

int npu_qdq_gridsample_a16(void* stream, const void* in, npu_dtype dtype, const void* grid,
                           void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                           int align_corners, npu_dtype compute,
                           const npu_gridsample_qdq* coeff) {
  if ((dtype != NPU_I16 && dtype != NPU_U16) || (compute != NPU_BF16 && compute != NPU_F16))
    return -1;
  return qdq_impl(stream, in, dtype, grid, out, H, W, C, Ho, Wo, align_corners,
                  compute, coeff);
}

}  // extern "C"
