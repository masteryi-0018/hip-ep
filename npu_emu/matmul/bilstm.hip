/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * bf16 LSTM direction. Per timestep one kernel computes the 4H gate
 * pre-activations and one updates the state; z, C and the H ping-pong live in
 * a scratch buffer.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/npu_emu.h"
#include "matmul/bilstm.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/launch.h"
#include "common/nlf.h"
#include "common/mac.h"
#include "common/mmac.h"
#include "common/bits.h"

using namespace npu_emu;

namespace {

constexpr int kChains = 4;

// W.x + b: one bf16 MMAC per tile of 32 inputs onto the accumulator, which starts from the
// bias; a short last tile has zero lanes.
__device__ inline float wx(const uint16_t* v, const uint16_t* row, uint16_t bias, int64_t k) {
  uint32_t acc = f32_bits(bf16_to_f32(bias));
  for (int64_t k0 = 0; k0 < k; k0 += 32)
    acc = mmac_bf16(acc, v + k0, 1, row + k0, 1, int(k - k0 < 32 ? k - k0 : 32));
  return bits_f32(acc);
}

// R.h + b: four fused fp32 chains over k mod 4, then the bias and the chains added in order.
__device__ inline float gemv(const uint16_t* v, const uint16_t* row, uint16_t bias, int64_t k) {
  float acc[kChains] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int64_t i = 0; i + kChains <= k; i += kChains)
    for (int q = 0; q < kChains; ++q)
      acc[q] = fma_q(bf16_to_f32(row[i + q]), bf16_to_f32(v[i + q]), acc[q]);
  float psum = bf16_to_f32(bias);
  for (int q = 0; q < kChains; ++q) psum = add_q(acc[q], psum);
  return psum;
}

// 128 segments of width 1/8 over [-8, 8], result rounded to f16.
__device__ inline float sigmoid_lut(float v, const float* lut) {
  const float y = lut_interp(v, lut, lut_index(v, 8.0f, 128, kHalfEven), NPU_F32, kHalfEven);
  return round_float(y, NPU_F16, kHalfEven);
}

// 2 * sigmoid(2 v) - 1 with f16 steps.
__device__ inline float tanh_lut(float v, const float* lut) {
  const float s = sigmoid_lut(round_float(mul_q(v, 2.0f), NPU_F16, kHalfEven), lut);
  const float s2 = round_float(mul_q(s, 2.0f), NPU_F16, kHalfEven);
  return round_float(sub_q(s2, 1.0f), NPU_F16, kHalfEven);
}

struct Lstm {
  const uint16_t *x, *w, *r, *b;
  const float* lut;
  uint16_t *y, *h[2];
  float *z, *c;
  int64_t in, hid;
};

__global__ void init_kernel(int64_t n, Lstm p, const uint16_t* h0, const uint16_t* c0) {
  NPU_TID;
  p.h[0][i] = h0[i];
  p.c[i] = bf16_to_f32(c0[i]);
}

__global__ void gates_kernel(int64_t n, Lstm p, int64_t t, int cur) {
  NPU_TID;
  const float zx = wx(p.x + t * p.in, p.w + i * p.in, p.b[i], p.in);
  const float zh = gemv(p.h[cur], p.r + i * p.hid, p.b[n + i], p.hid);
  p.z[i] = round_float(add_q(zx, zh), NPU_F16, kHalfEven);
}

__global__ void state_kernel(int64_t n, Lstm p, int64_t t, int cur) {
  NPU_TID;
  const float ig = sigmoid_lut(p.z[i], p.lut);
  const float og = sigmoid_lut(p.z[n + i], p.lut);
  const float fg = sigmoid_lut(p.z[2 * n + i], p.lut);
  const float gg = tanh_lut(p.z[3 * n + i], p.lut);
  const float fc = mul_q(fg, p.c[i]);
  const float ic = mul_q(ig, gg);
  const float c = add_q(fc, ic);
  const uint16_t h = f32_to_bf16(mul_q(og, tanh_lut(round_float(c, NPU_F16, kHalfEven), p.lut)), kHalfEven);
  p.c[i] = c;
  p.h[cur ^ 1][i] = h;
  p.y[t * n + i] = h;
}

__global__ void cout_kernel(int64_t n, const float* c, uint16_t* out) {
  NPU_TID;
  out[i] = f32_to_bf16(c[i], kHalfEven);
}

__global__ void gate_lut_kernel(int64_t n, const uint16_t* in, const float* lut, int is_tanh,
                                uint16_t* out) {
  NPU_TID;
  const float v = f16_to_f32(in[i]);
  out[i] = f32_to_f16(is_tanh ? tanh_lut(v, lut) : sigmoid_lut(v, lut), kHalfEven);
}

}  // namespace

extern "C" {

int npu_bidirectional_lstm_bf16(void* stream, const void* x_, const void* h0_, const void* c0_,
                                const void* w_, const void* r_, const void* b_, const void* lut_,
                                void* y_, void* c_out_, int64_t seq, int64_t input_size,
                                int64_t hidden, int reverse) {
  if (any_null(x_, h0_, c0_, w_, r_, b_, lut_, y_, c_out_)) return -1;
  if (seq < 0 || input_size <= 0 || hidden <= 0 || input_size % kChains || hidden % 16 ||
      (reverse != 0 && reverse != 1))
    return -1;
  const auto u16 = [](const void* p) { return static_cast<const uint16_t*>(p); };
  const uint16_t *x = u16(x_), *h0 = u16(h0_), *c0 = u16(c0_), *w = u16(w_), *r = u16(r_),
                 *b = u16(b_);
  const float* lut = static_cast<const float*>(lut_);
  uint16_t *y = static_cast<uint16_t*>(y_), *c_out = static_cast<uint16_t*>(c_out_);
  hipStream_t st = static_cast<hipStream_t>(stream);
  const int64_t g = 4 * hidden;
  char* work = nullptr;
  const size_t bytes = size_t(g + hidden) * 4 + size_t(2 * hidden) * 2;
  if (int rc = int(hipMallocAsync(reinterpret_cast<void**>(&work), bytes, st))) return rc;
  Lstm p{x, w, r, b, lut, y, {nullptr, nullptr}, nullptr, nullptr, input_size, hidden};
  p.z = reinterpret_cast<float*>(work);
  p.c = p.z + g;
  p.h[0] = reinterpret_cast<uint16_t*>(p.c + hidden);
  p.h[1] = p.h[0] + hidden;
  int rc = launch(stream, hidden, init_kernel, p, h0, c0);
  int cur = 0;
  for (int64_t s = 0; !rc && s < seq; ++s) {
    const int64_t t = reverse ? seq - 1 - s : s;
    rc = launch(stream, g, gates_kernel, p, t, cur);
    if (!rc) rc = launch(stream, hidden, state_kernel, p, t, cur);
    cur ^= 1;
  }
  if (!rc) rc = launch(stream, hidden, cout_kernel, static_cast<const float*>(p.c), c_out);
  const int rc2 = int(hipFreeAsync(work, st));
  return rc ? rc : rc2;
}

int npu_test_bilstm_gate_lut(void* stream, const void* in, const void* lut, int is_tanh,
                             void* out, int64_t n) {
  if (any_null(in, lut, out) || n < 0) return -1;
  return launch(stream, n, gate_lut_kernel, static_cast<const uint16_t*>(in),
                static_cast<const float*>(lut), is_tanh, static_cast<uint16_t*>(out));
}

}  // extern "C"
