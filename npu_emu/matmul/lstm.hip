/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Integer QDQ LSTM cell and grpb. One thread per hidden unit (LSTM) or per
 * row (grpb).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/npu_emu.h"
#include "common/act.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "matmul/lstm.h"
#include "common/bits.h"
#include "common/nlf.h"

using namespace npu_emu;

namespace {

__device__ inline int sbyte(uint32_t w, int k) { return int(int8_t(uint8_t(w >> (8 * k)))); }

struct LstmQdq {
  int64_t qx, qh;
  int xw, xs, hr, hs, xwq, hrq, lin, c0s, c1s, c2s, nl, oh_s, oc_s;
  int64_t c_zp, c_scale, oh_scale, oh_zp, oc_scale, oc_zp;
};

__device__ inline LstmQdq parse(const uint32_t* q, bool a16) {
  LstmQdq p;
  p.qx = q[0];
  p.qh = q[1];
  p.xw = sbyte(q[2], 0), p.xs = sbyte(q[2], 1), p.hr = sbyte(q[2], 2), p.hs = sbyte(q[2], 3);
  p.xwq = sbyte(q[3], 0), p.hrq = sbyte(q[3], 1), p.lin = sbyte(q[3], 2), p.c0s = sbyte(q[3], 3);
  p.c1s = sbyte(q[4], 0), p.c2s = sbyte(q[4], 1);
  p.c_zp = a16 ? int64_t(int16_t(q[4] >> 16)) : int64_t(int8_t(q[4] >> 16));
  p.c_scale = int16_t(q[5]);
  p.nl = sbyte(q[5], 2);
  p.oh_scale = int16_t(q[6]);
  p.oh_zp = a16 ? int64_t(uint16_t(q[6] >> 16)) : int64_t(int8_t(q[6] >> 16));
  p.oc_scale = int16_t(q[7]);
  p.oc_zp = a16 ? int64_t(uint16_t(q[7] >> 16)) : int64_t(int8_t(q[7] >> 16));
  p.oh_s = sbyte(q[8], 0), p.oc_s = sbyte(q[8], 1);
  return p;
}

template <typename TI, typename TW, typename TB, typename TO, bool A16>
__global__ void lstm_kernel(int64_t n, const TI* x, const TI* h, const TI* c, const TW* w,
                            const TW* r, const TB* b, const uint32_t* qdq, TO* h_out, TO* c_out,
                            int64_t len_x, int64_t len_h, float* gates) {
  NPU_TID;
  const LstmQdq p = parse(qdq, A16);
  // A16: unsigned 32-bit lanes; the sum is read back as int32.
  const npu_dtype lane = A16 ? NPU_U32 : NPU_I32;
  int64_t sx = 0, sh = 0;
  for (int64_t k = 0; k < len_x; ++k) sx += int64_t(x[k]);
  for (int64_t k = 0; k < len_h; ++k) sh += int64_t(h[k]);
  sx = srs(sx, p.xs, lane);
  sh = srs(sh, p.hs, lane);
  float y[4];
  for (int g = 0; g < 4; ++g) {
    const int64_t row = g * len_h + i;
    int64_t dx = 0, dh = 0;
    for (int64_t k = 0; k < len_x; ++k) dx += int64_t(w[row * len_x + k]) * int64_t(x[k]);
    for (int64_t k = 0; k < len_h; ++k) dh += int64_t(r[row * len_h + k]) * int64_t(h[k]);
    dx = wrap_mul64(srs(dx, p.xw, lane), p.qx);
    dh = wrap_mul64(srs(dh, p.hr, lane), p.qh);
    const int64_t xw = srs(dx, p.xwq, lane), hr = srs(dh, p.hrq, lane);
    // 32-bit lanes wrap.
    const int64_t lin = wrap(xw + hr + int64_t(b[row]) - (sx + sh), NPU_I32);
    y[g] = int_to_float(lin, -p.lin, kHalfEven);
    if (gates) gates[g * len_h + i] = y[g];
  }
  if (gates) return;
  const float vi = sigmoid_tanh(y[0], NPU_BF16, kHalfEven);
  const float vo = sigmoid_tanh(y[1], NPU_BF16, kHalfEven);
  const float vf = sigmoid_tanh(y[2], NPU_BF16, kHalfEven);
  const float vg = tanh_acc(y[3], NPU_BF16);
  // c - c_zp wraps in the lane type (int8 / int16).
  const int64_t cd = wrap(int64_t(c[i]) - p.c_zp, A16 ? NPU_I16 : NPU_I8);
  const int64_t cs = shift_round(cd, -p.c0s, kFloor);
  // cs saturates to the int64 range, so the product wraps. [default]
  const int64_t ci = srs(wrap_mul64(cs, p.c_scale), p.c1s, NPU_I32);
  const float cf = int_to_float(ci, -p.c2s, kHalfEven);
  const float C = fma_q(vf, vg, mul_q(vi, cf));
  const float H = mul_q(tanh_acc(C, NPU_BF16), vo);
  auto out_code = [&](float v, int64_t scale, int64_t zp, int s) {
    const int64_t zs = shift_round(zp, -s, kFloor);
    // zs saturates to the int64 range, so the sum wraps. [default]
    const int64_t q = wrap_add64(zs, scale * float_to_int(v, NPU_I32, p.nl, kHalfEven));
    return srs(q, s, A16 ? NPU_U16 : NPU_I8);
  };
  h_out[i] = TO(out_code(H, p.oh_scale, p.oh_zp, p.oh_s));
  c_out[i] = TO(out_code(C, p.oc_scale, p.oc_zp, p.oc_s));
}

// e = exp2(k0 - k1 |x| + k2 x^2); 1 - e when the sign bit of x is clear, else e.
__device__ inline float grpb_sigmoid(float x) {
  const float k0 = float(-1 + 895.001), k1 = 0.8083f, k2 = -0.1084f;
  const uint32_t xb = f32_bits(x);
  const float xa = bits_f32(xb | 0x80000000u);
  const uint32_t flip = (xb | 0x80000000u) ^ xb;  // the sign bit when x >= +0
  const float p01 = fma_q(k1, xa, k0);
  const float p012 = fma_q(mul_q(x, x), k2, p01);
  const float e = hw_exp2_bf20(p012);
  return add_q(flip ? 1.0f : 0.0f, bits_f32(f32_bits(e) | flip));
}

__global__ void grpb_kernel(int64_t n, const void* ifm, npu_dtype ifm_type, const void* wts,
                            npu_dtype wts_type, const float* c0, const float* c1, const float* c2,
                            float head, float lyr0, float lyr1, uint16_t* out, npu_dtype ofm_type,
                            int64_t k, float* stages) {
  NPU_TID;
  const bool a16 = is_i16(ifm_type);
  float s[2];
  for (int g = 0; g < 2; ++g) {
    // a16: the high byte keeps the input's sign, the low byte is unsigned.
    int64_t ghi = 0, glo = 0, shi = 0, slo = 0;
    for (int64_t kk = 0; kk < k; ++kk) {
      const int64_t a = load_int(ifm, ifm_type, i * k + kk);
      const int64_t ah = a16 ? (a >> 8) : a, al = a16 ? (a & 0xff) : 0;
      int64_t ws = 0;
      for (int j = 0; j < 4; ++j) ws += load_int(wts, wts_type, kk * 8 + 4 * g + j);
      ghi += ah * ws;
      glo += al * ws;
      shi += ah;
      slo += al;
    }
    // 32-bit accumulators wrap.
    ghi = wrap(ghi, NPU_I32), glo = wrap(glo, NPU_I32);
    float x;
    if (a16) {
      const float gm = mac_exact(ghi, 256.0f, mac_exact(glo, 1.0f, 0.0f));
      x = fma_q(gm, c2[i], c0[i]);
      if (c1) x = fma_q(c1[i], mac_exact(shi, 256.0f, mac_exact(slo, 1.0f, 0.0f)), x);
    } else {
      x = mac_exact(ghi, c2[i], c0[i]);
      if (c1) x = mac_exact(shi, c1[i], x);
    }
    s[g] = grpb_sigmoid(x);
    if (stages) stages[4 * i + g] = x, stages[4 * i + 2 + g] = s[g];
  }
  if (stages) return;
  const float r = fma_q(s[0], msc_q(s[1], head, lyr0), lyr1);
  out[i] = encode_float(r, ofm_type, kHalfEven, kFtzIn);
}

__global__ void grpb_sigmoid_kernel(int64_t n, const float* x, float* out) {
  NPU_TID;
  out[i] = grpb_sigmoid(x[i]);
}

const float* f32p(const void* p) { return static_cast<const float*>(p); }

int grpb(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts, npu_dtype wts_type,
         const void* c0, const void* c1, const void* c2, void* out, npu_dtype ofm_type,
         int64_t m, int64_t k, float head, float lyr0, float lyr1) {
  if (any_null(ifm, wts, c0, c2, out) || m < 0 || k < 1 || !is_small_int(ifm_type) ||
      !is_i8(wts_type) || !is_half_float(ofm_type))
    return -1;
  return launch(stream, m, grpb_kernel, ifm, ifm_type, wts, wts_type, f32p(c0), f32p(c1),
                f32p(c2), head, lyr0, lyr1, static_cast<uint16_t*>(out), ofm_type, k,
                static_cast<float*>(nullptr));
}

template <typename A, typename W, typename B, typename O, bool kA16>
int lstm(void* stream, const void* x, const void* h, const void* c, const void* w, const void* r,
         const void* b, const void* qdq, void* h_out, void* c_out, int64_t len_x, int64_t len_h,
         float* gates) {
  return launch(stream, len_h, lstm_kernel<A, W, B, O, kA16>, static_cast<const A*>(x),
                static_cast<const A*>(h), static_cast<const A*>(c), static_cast<const W*>(w),
                static_cast<const W*>(r), static_cast<const B*>(b),
                static_cast<const uint32_t*>(qdq), static_cast<O*>(h_out), static_cast<O*>(c_out),
                len_x, len_h, gates);
}

}  // namespace

extern "C" {

int npu_lstm_a8w8(void* stream, const void* x, const void* h, const void* c, const void* w,
                  const void* r, const void* b, const void* qdq, void* h_out, void* c_out,
                  int64_t len_x, int64_t len_h) {
  if (any_null(x, h, c, w, r, b, qdq, h_out, c_out) || len_x < 1 || len_h < 1) return -1;
  return lstm<int8_t, int8_t, int32_t, int8_t, false>(stream, x, h, c, w, r, b, qdq, h_out, c_out,
                                                       len_x, len_h, nullptr);
}

int npu_lstm_a16w8(void* stream, const void* x, const void* h, const void* c, const void* w,
                   const void* r, const void* b, const void* qdq, void* h_out, void* c_out,
                   int64_t len_x, int64_t len_h) {
  if (any_null(x, h, c, w, r, b, qdq, h_out, c_out) || len_x < 1 || len_h < 1) return -1;
  return lstm<uint16_t, uint8_t, uint32_t, uint16_t, true>(stream, x, h, c, w, r, b, qdq, h_out,
                                                           c_out, len_x, len_h, nullptr);
}

int npu_grpb_a8w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                  npu_dtype wts_type, const void* c0, const void* c1, const void* c2, void* out,
                  npu_dtype ofm_type, int64_t m, int64_t k, float scalar_head,
                  float scalar_lyr0, float scalar_lyr1) {
  if (!is_i8(ifm_type)) return -1;
  return grpb(stream, ifm, ifm_type, wts, wts_type, c0, c1, c2, out, ofm_type, m, k, scalar_head,
              scalar_lyr0, scalar_lyr1);
}

int npu_grpb_a16w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                   npu_dtype wts_type, const void* c0, const void* c1, const void* c2, void* out,
                   npu_dtype ofm_type, int64_t m, int64_t k, float scalar_head,
                   float scalar_lyr0, float scalar_lyr1) {
  if (!is_i16(ifm_type)) return -1;
  return grpb(stream, ifm, ifm_type, wts, wts_type, c0, c1, c2, out, ofm_type, m, k, scalar_head,
              scalar_lyr0, scalar_lyr1);
}

int npu_test_lstm_gates(void* stream, int a16, const void* x, const void* h, const void* w,
                        const void* r, const void* b, const void* qdq, void* gates,
                        int64_t len_x, int64_t len_h) {
  if (any_null(x, h, w, r, b, qdq, gates) || len_x < 1 || len_h < 1) return -1;
  float* g = static_cast<float*>(gates);
  // h stands in for c, which the gates do not read
  if (a16)
    return lstm<uint16_t, uint8_t, uint32_t, uint16_t, true>(stream, x, h, h, w, r, b, qdq,
                                                             nullptr, nullptr, len_x, len_h, g);
  return lstm<int8_t, int8_t, int32_t, int8_t, false>(stream, x, h, h, w, r, b, qdq, nullptr,
                                                       nullptr, len_x, len_h, g);
}

int npu_test_grpb_stages(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                         npu_dtype wts_type, const void* c0, const void* c1, const void* c2,
                         void* stages, int64_t m, int64_t k) {
  if (any_null(ifm, wts, c0, c2, stages) || m < 0 || k < 1 || !is_small_int(ifm_type) ||
      !is_i8(wts_type))
    return -1;
  return launch(stream, m, grpb_kernel, ifm, ifm_type, wts, wts_type, f32p(c0), f32p(c1),
                f32p(c2), 0.0f, 0.0f, 0.0f, static_cast<uint16_t*>(nullptr), NPU_F16, k,
                static_cast<float*>(stages));
}

int npu_test_grpb_sigmoid(void* stream, const void* x, void* out, int64_t n) {
  if (any_null(x, out) || n < 0) return -1;
  return launch(stream, n, grpb_sigmoid_kernel, f32p(x), static_cast<float*>(out));
}

}  // extern "C"
