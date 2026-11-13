/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for include/npu_emu_gridsample.h. One thread per
 * output element, each recomputing its pixel's indices and coefficients;
 * these are reference-speed kernels.
 */
#include <hip/hip_runtime.h>
#include <math.h>

#include "npu_emu.h"
#include "npu_emu_device.h"
#include "npu_emu_gridsample.h"
#include "npu_emu_launch.h"

using namespace npu_emu;

// The kernels round every fp32 operation separately; fused multiply-adds are
// written out (fmaf / mac_exact) where the NPU fuses.
#pragma clang fp contract(off)

namespace {

// Round fp32 to the 2-byte interpolation type (half to even) and back.
__device__ inline float to_t(float v, bool f16) {
  return f16 ? f16_to_f32(f32_to_f16(v, kEven)) : bf16_to_f32(f32_to_bf16(v, kEven));
}

// aie::to_fixed_floor<int16> / to_fixed<int16> under rounding_mode::floor:
// floor, saturated to int16. NaN -> 0 (unverified).
__device__ inline int floor16(float g) {
  if (g != g) return 0;
  const float f = floorf(g);
  return f <= -32768.0f ? -32768 : (f >= 32767.0f ? 32767 : int(f));
}

struct Shape {
  int64_t H, W, C;
};

// Clamped neighbour index k (0..3 = +0, +1, +W, +W+1) of the top-left index
// i0, as grid_sample_gather_bf16_check computes it: the top-left index comes
// out of an acc32 MAC saturated to int16, the offsets are added in int16
// (wrapping), then clamped to [0, H * W - 1].
__device__ inline int64_t neighbour(int fxc, int fyc, int k, const Shape& s) {
  const int64_t a = int64_t(fxc) + int64_t(fyc) * s.W;
  const int16_t i0 = int16_t(a < -32768 ? -32768 : (a > 32767 ? 32767 : a));
  const int16_t off = int16_t(k == 0 ? 0 : k == 1 ? 1 : k == 2 ? s.W : s.W + 1);
  int16_t v = int16_t(i0 + off);
  const int16_t max_idx = int16_t(s.H * s.W - 1);
  if (v < 0) v = 0;
  if (v > max_idx) v = max_idx;
  return v;
}

// The four bilinear coefficients of the AIE_SOL index stage
// (grid_sample_index_bf16), all in T.
__device__ inline void coeffs_sol(float gx, float gy, const Shape& s, bool f16, int* fxc,
                                  int* fyc, float c[4]) {
  const int fx = floor16(gx), fy = floor16(gy);
  *fxc = fx < s.W - 1 ? fx : int(s.W - 1);
  *fyc = fy < s.H - 1 ? fy : int(s.H - 1);
  // pad_bound_w / _h are bfloat16 RTP fields widened through float to T.
  const float pw0 = to_t(bf16_to_f32(f32_to_bf16(float(s.W), kEven)), f16);
  const float ph0 = to_t(bf16_to_f32(f32_to_bf16(float(s.H), kEven)), f16);
  const float pw1 = to_t(pw0 - 1.0f, f16), ph1 = to_t(ph0 - 1.0f, f16);
  const bool x0 = gx > pw0 || gx < 0.0f, x1 = gx > pw1 || gx < -1.0f;
  const bool y0 = gy > ph0 || gy < 0.0f, y1 = gy > ph1 || gy < -1.0f;
  const float ffx = to_t(float(*fxc), f16), ffy = to_t(float(*fyc), f16);
  const float t0x = to_t(to_t(1.0f - gx, f16) + ffx, f16), t1x = to_t(gx - ffx, f16);
  const float t0y = to_t(to_t(1.0f - gy, f16) + ffy, f16), t1y = to_t(gy - ffy, f16);
  // The x term is replaced by +0 before the multiply (aie::select).
  c[0] = to_t((x0 || y0 ? 0.0f : t0x) * t0y, f16);
  c[1] = to_t((x1 || y0 ? 0.0f : t1x) * t0y, f16);
  c[2] = to_t((x0 || y1 ? 0.0f : t0x) * t1y, f16);
  c[3] = to_t((x1 || y1 ? 0.0f : t1x) * t1y, f16);
}

// 4-tap resize: fp32 mul then three fused MACs (exact T x T products).
template <typename Load>
__device__ inline float resize(int fxc, int fyc, const float c[4], const Shape& s, int64_t ch,
                               Load load) {
  float acc = load(neighbour(fxc, fyc, 0, s) * s.C + ch) * c[0];
  for (int k = 1; k < 4; ++k) acc = fmaf(load(neighbour(fxc, fyc, k, s) * s.C + ch), c[k], acc);
  return acc;
}

__global__ void gridsample_bf16_kernel(int64_t n, const uint16_t* in, const uint16_t* grid,
                                       uint16_t* out, Shape s, bool f16) {
  NPU_TID;
  const int64_t ch = i % s.C, p = i / s.C;
  const npu_dtype t = f16 ? NPU_F16 : NPU_BF16;
  const float gx = load_float(grid, t, 2 * p), gy = load_float(grid, t, 2 * p + 1);
  int fxc, fyc;
  float c[4];
  coeffs_sol(gx, gy, s, f16, &fxc, &fyc, c);
  const float acc = resize(fxc, fyc, c, s, ch, [&](int64_t e) { return load_float(in, t, e); });
  store_float(out, t, i, acc, kEven);
}

// convert_fp8_fp16: the identity mul_8x8_8x8 decodes 8 consecutive bytes of a
// channel pair's [H][W][2] slice together (4 pixels x 2 channels; the slice
// is converted from its start), then mul_elem_32 by the scale and a
// half-to-even conversion to T.
__device__ inline float fp8_in(const uint8_t* in, int64_t e, const Shape& s, float scale,
                               bool f16) {
  const int64_t pix = e / s.C, pair = (e % s.C) & ~int64_t(1);
  const int64_t p0 = pix & ~int64_t(3);
  for (int64_t q = p0; q < p0 + 4 && q < s.H * s.W; ++q)
    for (int k = 0; k < 2; ++k)
      if ((in[q * s.C + pair + k] & 0x7f) == 0x7f) return __uint_as_float(0x7fc00000u);
  // The 8-term dot product returns +0 for a -0 input.
  const float v = fp8_to_f32(in[e]) + 0.0f;
  return to_t(v * scale, f16);
}

__global__ void gridsample_fp8_kernel(int64_t n, const uint8_t* in, const uint16_t* grid,
                                      uint8_t* out, Shape s, bool f16, float ifm_scale,
                                      float ofm_recip) {
  NPU_TID;
  const int64_t ch = i % s.C, p = i / s.C;
  const npu_dtype t = f16 ? NPU_F16 : NPU_BF16;
  const float gx = load_float(grid, t, 2 * p), gy = load_float(grid, t, 2 * p + 1);
  int fxc, fyc;
  float c[4];
  coeffs_sol(gx, gy, s, f16, &fxc, &fyc, c);
  const float acc = resize(fxc, fyc, c, s, ch,
                           [&](int64_t e) { return fp8_in(in, e, s, ifm_scale, f16); });
  // to_vector<float8> under saturation: clamp to the largest finite value.
  float v = acc * ofm_recip;
  if (v == v) v = fminf(fmaxf(v, -448.0f), 448.0f);
  out[i] = f32_to_fp8(v, kEven);
}

struct QdqArgs {
  Shape s;
  int align_corners;
  bool f16;
  npu_gridsample_qdq q;
  float ifm_offset;  // fp32(-(float)ifm_zp * ifm_scale)
};

// gridsample_dequant_denorm_grid_float: grid code -> denormalized coordinate.
__device__ inline float qdq_coord(int64_t code, const QdqArgs& a, int64_t extent) {
  float g = (float(code) - float(a.q.grid_zp)) * a.q.grid_scale;
  g = g + 1.0f;
  if (a.align_corners == 0) return fmaf(g, float(extent) * 0.5f, -0.5f);  // aie::mac
  return (float(extent - 1) * g) * 0.5f;
}

__global__ void qdq_gridsample_kernel(int64_t n, const void* in, npu_dtype dt,
                                      const uint16_t* grid, void* out, QdqArgs a) {
  NPU_TID;
  const Shape& s = a.s;
  const int64_t ch = i % s.C, p = i / s.C;
  const float gx = qdq_coord(grid[2 * p], a, s.W), gy = qdq_coord(grid[2 * p + 1], a, s.H);
  // gridsample_index_floor / gridsample_coeff_float
  const int fx = floor16(gx), fy = floor16(gy);
  const int fxc = fx < s.W - 1 ? fx : int(s.W - 1);
  const int fyc = fy < s.H - 1 ? fy : int(s.H - 1);
  const int16_t nx = int16_t(fx + 1), ny = int16_t(fy + 1);  // int16 add
  const bool in_x0 = fx >= 0 && fx < s.W, in_x1 = nx >= 0 && nx < s.W;
  const bool in_y0 = fy >= 0 && fy < s.H, in_y1 = ny >= 0 && ny < s.H;
  const float t0x = (1.0f - gx) + float(fxc), t1x = gx - float(fxc);
  const float t0y = (1.0f - gy) + float(fyc), t1y = gy - float(fyc);
  float c[4];
  c[0] = in_x0 && in_y0 ? to_t(t0x * t0y, a.f16) : 0.0f;
  c[1] = in_x1 && in_y0 ? to_t(t1x * t0y, a.f16) : 0.0f;
  c[2] = in_x0 && in_y1 ? to_t(t0x * t1y, a.f16) : 0.0f;
  c[3] = in_x1 && in_y1 ? to_t(t1x * t1y, a.f16) : 0.0f;
  const float acc = resize(fxc, fyc, c, s, ch, [&](int64_t e) {
    return to_t(mac_exact(load_int(in, dt, e), a.q.ifm_scale, a.ifm_offset), a.f16);
  });
  // gridsample_resize_quant: one MAC with the output multiplier, then
  // to_fixed under conv_even + saturation.
  int64_t lo, hi;
  int_range(dt, &lo, &hi);
  store_int(out, dt, i, sat_rne(fmaf(acc, a.q.ofm_scale, float(a.q.ofm_zp)), lo, hi));
}

bool shape_ok(int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo) {
  return H > 0 && W > 0 && H <= 32767 && W <= 32767 && H * W <= 32768 && C > 0 && C % 2 == 0 &&
         Ho >= 0 && Wo >= 0;
}

int qdq_impl(void* stream, const void* in, npu_dtype dt, const void* grid, void* out,
             int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo, int align_corners,
             bool f16, const npu_gridsample_qdq* coeff) {
  if (!coeff || !shape_ok(H, W, C, Ho, Wo) || (align_corners != 0 && align_corners != 1))
    return -1;
  QdqArgs a{{H, W, C}, align_corners, f16, *coeff, -float(coeff->ifm_zp) * coeff->ifm_scale};
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
                Shape{H, W, C}, dtype == NPU_F16);
}

int npu_gridsample_fp8(void* stream, const void* in, const void* grid, npu_dtype grid_type,
                       void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                       float ifm_scale, float ofm_recip) {
  if ((grid_type != NPU_BF16 && grid_type != NPU_F16) || !shape_ok(H, W, C, Ho, Wo)) return -1;
  return launch(stream, Ho * Wo * C, gridsample_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<const uint16_t*>(grid), static_cast<uint8_t*>(out),
                Shape{H, W, C}, grid_type == NPU_F16, ifm_scale, ofm_recip);
}

int npu_qdq_gridsample_a8(void* stream, const void* in, npu_dtype dtype, const void* grid,
                          void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                          int align_corners, const npu_gridsample_qdq* coeff) {
  if (dtype != NPU_I8 && dtype != NPU_U8) return -1;
  return qdq_impl(stream, in, dtype, grid, out, H, W, C, Ho, Wo, align_corners, false, coeff);
}

int npu_qdq_gridsample_a16(void* stream, const void* in, npu_dtype dtype, const void* grid,
                           void* out, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo,
                           int align_corners, npu_dtype compute,
                           const npu_gridsample_qdq* coeff) {
  if ((dtype != NPU_I16 && dtype != NPU_U16) || (compute != NPU_BF16 && compute != NPU_F16))
    return -1;
  return qdq_impl(stream, in, dtype, grid, out, H, W, C, Ho, Wo, align_corners,
                  compute == NPU_F16, coeff);
}

}  // extern "C"
