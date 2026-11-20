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
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "matmul/lstm.h"
#include "common/nlf.h"

using namespace npu_emu;

namespace {

constexpr int64_t kI32Min = -(int64_t(1) << 31), kI32Max = (int64_t(1) << 31) - 1;
constexpr int64_t kU32Max = (int64_t(1) << 32) - 1;

// 64-bit product, wrapping.
__device__ inline int64_t mul64(int64_t a, int64_t b) { return int64_t(uint64_t(a) * uint64_t(b)); }

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

// bf16(0.5 - 0.5 * tanh(fp32(v * -0.5))), half to even.
__device__ inline float lstm_sigmoid(float v) {
  const float t = tanh_acc(__fmul_rn(v, -0.5f), NPU_BF16);
  return bf16_to_f32(f32_to_bf16(__fsub_rn(0.5f, __fmul_rn(t, 0.5f)), kHalfEven));
}

template <typename TI, typename TW, typename TB, typename TO, bool A16>
__global__ void lstm_kernel(int64_t n, const TI* x, const TI* h, const TI* c, const TW* w,
                            const TW* r, const TB* b, const uint32_t* qdq, TO* h_out, TO* c_out,
                            int64_t len_x, int64_t len_h, float* gates) {
  NPU_TID;
  const LstmQdq p = parse(qdq, A16);
  // A16: unsigned 32-bit lanes; the sum is read back as int32.
  const int64_t lo = A16 ? 0 : kI32Min, hi = A16 ? kU32Max : kI32Max;
  int64_t sx = 0, sh = 0;
  for (int64_t k = 0; k < len_x; ++k) sx += int64_t(x[k]);
  for (int64_t k = 0; k < len_h; ++k) sh += int64_t(h[k]);
  sx = clamp64(shift_round(sx, p.xs, kHalfEven), lo, hi);
  sh = clamp64(shift_round(sh, p.hs, kHalfEven), lo, hi);
  float y[4];
  for (int g = 0; g < 4; ++g) {
    const int64_t row = g * len_h + i;
    int64_t dx = 0, dh = 0;
    for (int64_t k = 0; k < len_x; ++k) dx += int64_t(w[row * len_x + k]) * int64_t(x[k]);
    for (int64_t k = 0; k < len_h; ++k) dh += int64_t(r[row * len_h + k]) * int64_t(h[k]);
    dx = mul64(clamp64(shift_round(dx, p.xw, kHalfEven), lo, hi), p.qx);
    dh = mul64(clamp64(shift_round(dh, p.hr, kHalfEven), lo, hi), p.qh);
    const int64_t xw = clamp64(shift_round(dx, p.xwq, kHalfEven), lo, hi);
    const int64_t hr = clamp64(shift_round(dh, p.hrq, kHalfEven), lo, hi);
    const uint32_t lin = uint32_t(xw) + uint32_t(hr) + uint32_t(b[row]) - (uint32_t(sx) + uint32_t(sh));
    y[g] = int_to_float(int32_t(lin), -p.lin, kHalfEven);
    if (gates) gates[g * len_h + i] = y[g];
  }
  if (gates) return;
  const float vi = lstm_sigmoid(y[0]), vo = lstm_sigmoid(y[1]), vf = lstm_sigmoid(y[2]);
  const float vg = tanh_acc(y[3], NPU_BF16);
  // c - c_zp wraps in the lane type (int8 / int16).
  const int64_t cd = A16 ? int64_t(int16_t(uint16_t(int64_t(int16_t(c[i])) - p.c_zp)))
                         : int64_t(int8_t(uint8_t(int64_t(c[i]) - p.c_zp)));
  const int64_t cs = shift_round(cd, -p.c0s, kFloor);
  // cs saturates to the int64 range, so the product wraps like mul64. [default]
  const int64_t ci = clamp64(shift_round(mul64(cs, p.c_scale), p.c1s, kHalfEven), kI32Min, kI32Max);
  const float cf = int_to_float(ci, -p.c2s, kHalfEven);
  const float C = fmaf(vf, vg, __fmul_rn(vi, cf));
  const float H = __fmul_rn(tanh_acc(C, NPU_BF16), vo);
  const int64_t olo = A16 ? 0 : -128, ohi = A16 ? 65535 : 127;
  auto quant = [&](float v, int64_t scale, int64_t zp, int s) {
    const int64_t zs = shift_round(zp, -s, kFloor);
    // zs saturates to the int64 range, so the sum wraps. [default]
    const int64_t q = int64_t(uint64_t(zs) + uint64_t(scale * float_to_int(v, NPU_I32, p.nl, kHalfEven)));
    return clamp64(shift_round(q, s, kHalfEven), olo, ohi);
  };
  h_out[i] = TO(quant(H, p.oh_scale, p.oh_zp, p.oh_s));
  c_out[i] = TO(quant(C, p.oc_scale, p.oc_zp, p.oc_s));
}

// e = exp2(k0 - k1 |x| + k2 x^2); 1 - e when the sign bit of x is clear, else e.
__device__ inline float grpb_sigmoid(float x) {
  const float k0 = float(-1 + 895.001), k1 = 0.8083f, k2 = -0.1084f;
  const uint32_t xb = __float_as_uint(x);
  const float xa = __uint_as_float(xb | 0x80000000u);
  const uint32_t flip = (xb | 0x80000000u) ^ xb;  // the sign bit when x >= +0
  const float p01 = fmaf(k1, xa, k0);
  const float p012 = fmaf(__fmul_rn(x, x), k2, p01);
  const float e = hw_exp2_bf20(p012);
  return __fadd_rn(flip ? 1.0f : 0.0f, __uint_as_float(__float_as_uint(e) | flip));
}

__global__ void grpb_kernel(int64_t n, const void* ifm, npu_dtype it, const void* wts,
                            npu_dtype wt, const float* c0, const float* c1, const float* c2,
                            float head, float lyr0, float lyr1, uint16_t* out, npu_dtype ot,
                            int64_t k, float* stages) {
  NPU_TID;
  const bool a16 = it == NPU_I16 || it == NPU_U16;
  float s[2];
  for (int g = 0; g < 2; ++g) {
    // a16: the high byte keeps the input's sign, the low byte is unsigned.
    int64_t ghi = 0, glo = 0, shi = 0, slo = 0;
    for (int64_t kk = 0; kk < k; ++kk) {
      const int64_t a = load_int(ifm, it, i * k + kk);
      const int64_t ah = a16 ? (a >> 8) : a, al = a16 ? (a & 0xff) : 0;
      int64_t ws = 0;
      for (int j = 0; j < 4; ++j) ws += load_int(wts, wt, kk * 8 + 4 * g + j);
      ghi += ah * ws;
      glo += al * ws;
      shi += ah;
      slo += al;
    }
    // 32-bit accumulators wrap.
    ghi = int32_t(uint32_t(ghi)), glo = int32_t(uint32_t(glo));
    float x;
    if (a16) {
      const float gm = mac_exact(ghi, 256.0f, mac_exact(glo, 1.0f, 0.0f));
      x = fmaf(gm, c2[i], c0[i]);
      if (c1) x = fmaf(c1[i], mac_exact(shi, 256.0f, mac_exact(slo, 1.0f, 0.0f)), x);
    } else {
      x = mac_exact(ghi, c2[i], c0[i]);
      if (c1) x = mac_exact(shi, c1[i], x);
    }
    s[g] = grpb_sigmoid(x);
    if (stages) stages[4 * i + g] = x, stages[4 * i + 2 + g] = s[g];
  }
  if (stages) return;
  const float r = fma_q(s[0], fma_q(-s[1], head, lyr0), lyr1);
  out[i] = encode_float(r, ot, kHalfEven, kFtzIn);
}

__global__ void grpb_sigmoid_kernel(int64_t n, const float* x, float* out) {
  NPU_TID;
  out[i] = grpb_sigmoid(x[i]);
}

int grpb(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts, npu_dtype wts_type,
         const float* c0, const float* c1, const float* c2, float head, float lyr0, float lyr1,
         uint16_t* out, npu_dtype ofm_type, int64_t m, int64_t k) {
  if (!ifm || !wts || !c0 || !c2 || !out || m < 1 || k < 1 ||
      (ifm_type != NPU_I8 && ifm_type != NPU_U8 && ifm_type != NPU_I16 && ifm_type != NPU_U16) ||
      (wts_type != NPU_I8 && wts_type != NPU_U8) || (ofm_type != NPU_F16 && ofm_type != NPU_BF16))
    return -1;
  return launch(stream, m, grpb_kernel, ifm, ifm_type, wts, wts_type, c0, c1, c2, head, lyr0, lyr1,
                out, ofm_type, k, static_cast<float*>(nullptr));
}

}  // namespace

extern "C" {

int npu_lstm_a8w8(void* stream, const int8_t* x, const int8_t* h, const int8_t* c,
                  const int8_t* w, const int8_t* r, const int32_t* b, const uint32_t* qdq,
                  int8_t* h_out, int8_t* c_out, int64_t len_x, int64_t len_h) {
  if (!x || !h || !c || !w || !r || !b || !qdq || !h_out || !c_out || len_x < 1 || len_h < 1)
    return -1;
  return launch(stream, len_h, lstm_kernel<int8_t, int8_t, int32_t, int8_t, false>, x, h, c, w, r,
                b, qdq, h_out, c_out, len_x, len_h, static_cast<float*>(nullptr));
}

int npu_lstm_a16w8(void* stream, const uint16_t* x, const uint16_t* h, const uint16_t* c,
                   const uint8_t* w, const uint8_t* r, const uint32_t* b, const uint32_t* qdq,
                   uint16_t* h_out, uint16_t* c_out, int64_t len_x, int64_t len_h) {
  if (!x || !h || !c || !w || !r || !b || !qdq || !h_out || !c_out || len_x < 1 || len_h < 1)
    return -1;
  return launch(stream, len_h, lstm_kernel<uint16_t, uint8_t, uint32_t, uint16_t, true>, x, h, c,
                w, r, b, qdq, h_out, c_out, len_x, len_h, static_cast<float*>(nullptr));
}

int npu_grpb_a8w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                  npu_dtype wts_type, const float* c0, const float* c1, const float* c2,
                  float head, float lyr0, float lyr1, uint16_t* out, npu_dtype ofm_type,
                  int64_t m, int64_t k) {
  if (ifm_type != NPU_I8 && ifm_type != NPU_U8) return -1;
  return grpb(stream, ifm, ifm_type, wts, wts_type, c0, c1, c2, head, lyr0, lyr1, out, ofm_type,
              m, k);
}

int npu_grpb_a16w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                   npu_dtype wts_type, const float* c0, const float* c1, const float* c2,
                   float head, float lyr0, float lyr1, uint16_t* out, npu_dtype ofm_type,
                   int64_t m, int64_t k) {
  if (ifm_type != NPU_I16 && ifm_type != NPU_U16) return -1;
  return grpb(stream, ifm, ifm_type, wts, wts_type, c0, c1, c2, head, lyr0, lyr1, out, ofm_type,
              m, k);
}

int npu_test_lstm_gates(void* stream, int a16, const void* x, const void* h, const void* w,
                        const void* r, const void* b, const uint32_t* qdq, float* gates,
                        int64_t len_x, int64_t len_h) {
  if (!x || !h || !w || !r || !b || !qdq || !gates || len_x < 1 || len_h < 1) return -1;
  if (a16)
    return launch(stream, len_h, lstm_kernel<uint16_t, uint8_t, uint32_t, uint16_t, true>,
                  static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(h),
                  static_cast<const uint16_t*>(h), static_cast<const uint8_t*>(w),
                  static_cast<const uint8_t*>(r), static_cast<const uint32_t*>(b), qdq,
                  static_cast<uint16_t*>(nullptr), static_cast<uint16_t*>(nullptr), len_x, len_h,
                  gates);
  return launch(stream, len_h, lstm_kernel<int8_t, int8_t, int32_t, int8_t, false>,
                static_cast<const int8_t*>(x), static_cast<const int8_t*>(h),
                static_cast<const int8_t*>(h), static_cast<const int8_t*>(w),
                static_cast<const int8_t*>(r), static_cast<const int32_t*>(b), qdq,
                static_cast<int8_t*>(nullptr), static_cast<int8_t*>(nullptr), len_x, len_h, gates);
}

int npu_test_grpb_stages(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                         npu_dtype wts_type, const float* c0, const float* c1, const float* c2,
                         float* stages, int64_t m, int64_t k) {
  if (!ifm || !wts || !c0 || !c2 || !stages || m < 1 || k < 1) return -1;
  return launch(stream, m, grpb_kernel, ifm, ifm_type, wts, wts_type, c0, c1, c2, 0.0f, 0.0f,
                0.0f, static_cast<uint16_t*>(nullptr), NPU_F16, k, stages);
}

int npu_test_grpb_sigmoid(void* stream, const float* x, float* out, int64_t n) {
  if (!x || !out) return -1;
  return launch(stream, n, grpb_sigmoid_kernel, x, out);
}

}  // extern "C"
