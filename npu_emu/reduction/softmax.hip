/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for reduction/softmax.h. One thread per row.
 */
#include "reduction/softmax.h"

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/nlf.h"

namespace {

using namespace npu_emu;

constexpr float kBias = 895.001f;                     // 0x445fc010
constexpr float kLog2e = 1.44269502162933349609375f;  // 0x3fb8aa3b
constexpr float kInvLn2Bf16 = 1.4453125f;             // bfloat16(1.44269504f)
constexpr float kInvLn2F16 = 1.4423828125f;           // float16(1.44269504f)
constexpr uint16_t kBf16Guard = 0xff32;               // bfloat16(-2.3660258e38f)

__device__ inline float neg_inf() { return __uint_as_float(0xff800000u); }

// One vector reduce-add over columns [c0, c0 + 32) of f (columns >= C add +0).
template <typename F>
__device__ inline float add_chunk(const F& f, int64_t c0, int64_t C, float acc) {
  float v[32];
  const int n = int(C - c0 < 32 ? C - c0 : 32);
  for (int k = 0; k < n; ++k) v[k] = f(c0 + k);
  return vaddr_f32_reduce(v, n, true, acc);
}

// Row sum with one accumulator, chunks in column order from +0.
template <typename F>
__device__ inline float chain_sum(const F& f, int64_t C) {
  float s = 0.0f;
  for (int64_t c0 = 0; c0 < C; c0 += 32) s = add_chunk(f, c0, C, s);
  return s;
}

// Bit c of a row's mask (bit c % 8 of byte c / 8); no mask = all valid.
__device__ inline bool valid(const uint8_t* mrow, int64_t c) {
  return !mrow || ((mrow[c >> 3] >> (c & 7)) & 1);
}

__global__ void softmax_bf16_kernel(int64_t n, const uint16_t* in, uint16_t* out, int64_t C,
                                    const uint32_t* inv_exc) {
  NPU_TID;
  const uint16_t* x = in + i * C;
  uint16_t* y = out + i * C;
  float rmax = neg_inf();
  for (int64_t c = 0; c < C; ++c) rmax = vmax(rmax, bf16_to_f32(x[c]));
  // A row headed by -inf / -max whose max is at most 0xff32 gets
  // 1 / (n * exp2(bias)) at its n maxima and 0 elsewhere.
  if ((x[0] == 0xff80 || x[0] == 0xff7f) && rmax <= bf16_to_f32(kBf16Guard)) {
    int64_t win = 0;
    for (int64_t c = 0; c < C; ++c) win += bf16_to_f32(x[c]) == rmax;
    const float s = mul_q(float(win), hw_exp2_bf20(kBias));
    const uint16_t inv = f32_to_bf16(hw_inv_f32(s, inv_exc), kHalfAway, kFtzIn);
    for (int64_t c = 0; c < C; ++c) y[c] = bf16_to_f32(x[c]) == rmax ? inv : 0;
    return;
  }
  float smax = neg_inf();
  for (int64_t c = 0; c < C; ++c) smax = vmax(smax, mul_q(bf16_to_f32(x[c]), kInvLn2Bf16));
  const float off = sub_q(kBias, smax);
  // The scaled input is rounded to bf16 (half away) before the exponent.
  auto e = [&](int64_t c) {
    const float sx = round_float(mul_q(bf16_to_f32(x[c]), kInvLn2Bf16), NPU_BF16, kHalfAway);
    return hw_exp2_bf20(add_q(sx, off));
  };
  const uint16_t inv = f32_to_bf16(hw_inv_f32(chain_sum(e, C), inv_exc), kHalfAway, kFtzIn);
  for (int64_t c = 0; c < C; ++c) y[c] = mul16(f32_to_bf16(e(c), kHalfAway), inv, NPU_BF16);
}

// nc chunks of cc columns; columns >= C are -inf padding.
__global__ void softmax_2pass_bf16_kernel(int64_t n, const uint16_t* in, uint16_t* out, int64_t C,
                                          int64_t cc, int64_t nc, const uint32_t* inv_exc) {
  NPU_TID;
  const uint16_t* x = in + i * C;
  uint16_t* y = out + i * C;
  const float alpha = hw_exp2_bf20(kBias);
  const float inv_alpha = hw_inv_f32(alpha, inv_exc);
  // x * c_bf16 is exact in fp32 unless it overflows; padding is -inf.
  auto s = [&](int64_t c) {
    return c < C ? mul_q(bf16_to_f32(x[c]), kInvLn2Bf16) : neg_inf();
  };
  float mx = neg_inf(), sum = 0.0f;
  for (int64_t k = 0; k < nc; ++k) {
    const int64_t c0 = k * cc;
    float cmax = neg_inf();
    for (int64_t c = c0; c < c0 + cc; ++c) {
      cmax = vmax(cmax, s(c));
    }
    const float m = vmax(cmax, mx);
    if (m > mx && mx != neg_inf() && sum != 0.0f)
      sum = mul_q(sum, mul_q(hw_exp2_bf20(sub_q(kBias, sub_q(m, mx))), inv_alpha));
    mx = m;
    const float off = mx == neg_inf() ? 0.0f : sub_q(kBias, mx);
    for (int64_t v0 = c0; v0 < c0 + cc; v0 += 32) {
      float e[32];
      for (int k2 = 0; k2 < 32; ++k2)
        e[k2] = hw_exp2_bf20(add_q(round_float(s(v0 + k2), NPU_BF16, kHalfAway), off));
      sum = vaddr_f32_reduce(e, 32, true, sum);
    }
  }
  if (mx == neg_inf()) {
    // 1 / (count * alpha) at the raw maxima, 0 elsewhere.
    float rmax = neg_inf();
    for (int64_t c = 0; c < C; ++c) {
      rmax = vmax(rmax, bf16_to_f32(x[c]));
    }
    int64_t win = 0;
    for (int64_t c = 0; c < C; ++c) win += bf16_to_f32(x[c]) == rmax;
    const uint16_t inv =
        f32_to_bf16(hw_inv_f32(mul_q(float(win), alpha), inv_exc), kHalfAway, kFtzIn);
    for (int64_t c = 0; c < C; ++c) y[c] = bf16_to_f32(x[c]) == rmax ? inv : 0;
    return;
  }
  const uint16_t inv = f32_to_bf16(hw_inv_f32(sum, inv_exc), kHalfAway, kFtzIn);
  const float off = sub_q(kBias, mx);
  for (int64_t c = 0; c < C; ++c) {
    const float e = hw_exp2_bf20(add_q(round_float(s(c), NPU_BF16, kHalfAway), off));
    y[c] = mul16(f32_to_bf16(e, kHalfAway), inv, NPU_BF16);
  }
}

__global__ void softmax_fp16x16_kernel(int64_t n, const uint16_t* in, const uint8_t* mask,
                                       uint16_t* out, npu_dtype t, int64_t C,
                                       const uint32_t* inv_exc) {
  NPU_TID;
  const uint16_t* x = in + i * C;
  const uint8_t* mrow = mask ? mask + i * (C / 8) : nullptr;
  uint16_t* y = out + i * C;
  const float k = t == NPU_F16 ? kInvLn2F16 : kInvLn2Bf16;
  auto xv = [&](int64_t c) { return valid(mrow, c) ? decode_float(x[c], t) : neg_inf(); };
  float mx = neg_inf();
  for (int64_t c = 0; c < C; ++c) mx = vmax(mx, xv(c));
  const float off = sub_q(kBias, mul_q(mx, k));
  auto e = [&](int64_t c) { return hw_exp2_bf20(fma_q(xv(c), k, off)); };
  const uint16_t inv =
      uint16_t(encode_float(hw_inv_f32(chain_sum(e, C), inv_exc), t, kHalfEven, kFtzIn));
  for (int64_t c = 0; c < C; ++c) y[c] = mul16(uint16_t(encode_float(e(c), t, kHalfEven)), inv, t);
}

__global__ void softmax_fp8_kernel(int64_t n, const uint8_t* in, const uint8_t* mask,
                                   uint8_t* out, int64_t C, float ifm_scale, float recip,
                                   const uint32_t* inv_exc) {
  NPU_TID;
  const uint8_t* mrow = mask ? mask + i * (C / 8) : nullptr;
  uint8_t* y = out + i * C;
  // fp8 -> bf16 (half away), then x / ln2, -inf where masked.
  auto s = [&](int64_t c) {
    if (!valid(mrow, c)) return neg_inf();
    const float v = fp8_decode_identity(in + i * C, c, C);
    const float b = round_float(mul_q(v, ifm_scale), NPU_BF16, kHalfAway, kFtzIn);
    return mul_q(b, kInvLn2Bf16);
  };
  float mx = neg_inf();
  for (int64_t c = 0; c < C; ++c) mx = vmax(mx, s(c));
  const float off = sub_q(kBias, mx);
  // The scaled input stays in fp32; the bf16 stores of exp2 and inv(sum) round half to even.
  auto e = [&](int64_t c) { return hw_exp2_bf20(add_q(s(c), off)); };
  const uint16_t inv = f32_to_bf16(hw_inv_f32(chain_sum(e, C), inv_exc), kHalfEven, kFtzIn);
  const float r = scalar_cast(recip, NPU_BF16, kHalfEven);  // scalar (bfloat16) cast
  for (int64_t c = 0; c < C; ++c) {
    const float p = bf16_to_f32(mul16(f32_to_bf16(e(c), kHalfEven), inv, NPU_BF16));
    const float q = mul_q(p, r);
    y[c] = f32_to_fp8(q == 0.0f ? 0.0f : q, kHalfAway, kSat);
  }
}

// Accumulator of 32-column chunk k: 64-column groups alternate even / odd in
// pairs, a leftover full group and the tail group go to even; low / high half
// of each group to lo / hi. Index 2 * odd + hi.
__device__ inline int lsm_acc(int64_t k, int64_t C) {
  const int64_t g = k >> 1, pairs = 2 * (C / 64 / 2);
  return 2 * int(g < pairs && (g & 1)) + int(k & 1);
}

// The four sums of f, added ((even_lo + even_hi) + odd_lo) + odd_hi.
template <typename F, typename A>
__device__ inline float sum4(const F& f, int64_t C, const A& acc) {
  float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int64_t c0 = 0; c0 < C; c0 += 32) {
    const int a = acc(c0 >> 5);
    s[a] = add_chunk(f, c0, C, s[a]);
  }
  return add_q(add_q(add_q(s[0], s[1]), s[2]), s[3]);
}

__global__ void logsoftmax_bf16_kernel(int64_t n, const uint16_t* in, uint16_t* out, npu_dtype t,
                                       int64_t C) {
  NPU_TID;
  const uint16_t* x = in + i * C;
  uint16_t* y = out + i * C;
  float rmax = neg_inf();
  for (int64_t c = 0; c < C; ++c) rmax = vmax(rmax, decode_float(x[c], t));
  const float off = sub_q(kBias, mul_q(rmax, kLog2e));
  auto e = [&](int64_t c) { return hw_exp2_bf20(fma_q(decode_float(x[c], t), kLog2e, off)); };
  const float sum = sum4(e, C, [&](int64_t k) { return lsm_acc(k, C); });
  const float o = sub_q(-rmax, hw_approx_ln(sum));
  for (int64_t c = 0; c < C; ++c)
    y[c] = encode_float(add_q(decode_float(x[c], t), o), t, kHalfEven);
}

__global__ void logsoftmax_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, int64_t C,
                                      float in_scale, float out_scale_inv) {
  NPU_TID;
  uint8_t* y = out + i * C;
  auto v = [&](int64_t c) { return fp8_decode_identity(in + i * C, c, C); };
  float rmax = neg_inf();
  for (int64_t c = 0; c < C; ++c) rmax = vmax(rmax, mul_q(v(c), in_scale));
  const float es = mul_q(in_scale, kLog2e);
  const float off = sub_q(kBias, mul_q(rmax, kLog2e));
  auto e = [&](int64_t c) { return hw_exp2_bf20(fma_q(v(c), es, off)); };
  const float sum = sum4(e, C, [&](int64_t k) { return lsm_acc(k, C); });
  const float io = mul_q(in_scale, out_scale_inv);
  const float o = mul_q(sub_q(-rmax, hw_approx_ln(sum)), out_scale_inv);
  for (int64_t c = 0; c < C; ++c) y[c] = f32_to_fp8(fma_q(v(c), io, o), kHalfEven);
}

struct QdqSoftmax {
  npu_dtype t;
  int path, shift;
  float inv_ln2, ofm_scale, zp_acc;
};

__global__ void qdq_softmax_kernel(int64_t n, const void* in, const uint8_t* mask, void* out,
                                   int64_t C, QdqSoftmax p, const uint32_t* inv_exc) {
  NPU_TID;
  const uint8_t* mrow = mask ? mask + i * (C / 8) : nullptr;
  auto q = [&](int64_t c) { return float(load_int(in, p.t, i * C + c)); };  // exact
  if (p.path == 2) {
    // v = fp32(q * inv_ln2), e = exp2(v - fp32(max(v) - bias)).
    auto v = [&](int64_t c) { return valid(mrow, c) ? mul_q(q(c), p.inv_ln2) : neg_inf(); };
    float mx = neg_inf();
    for (int64_t c = 0; c < C; ++c) mx = vmax(mx, v(c));
    const float t = sub_q(mx, kBias);
    auto e = [&](int64_t c) { return hw_exp2_bf20(sub_q(v(c), t)); };
    const float inv = hw_inv_f32(mul_q(chain_sum(e, C), p.ofm_scale), inv_exc);
    for (int64_t c = 0; c < C; ++c)
      store_int(out, p.t, i * C + c,
                float_to_int(fma_q(e(c), inv, p.zp_acc), p.t, p.shift, kHalfEven));
    return;
  }
  // Paths 0 / 1: integer max; e and the inverse are rounded to f16 (half to
  // even) before the output fma.
  int64_t lo, hi;
  int_range(p.t, &lo, &hi);
  int64_t qmax = lo;
  for (int64_t c = 0; c < C; ++c) {
    const int64_t v = load_int(in, p.t, i * C + c);
    if (valid(mrow, c) && v > qmax) qmax = v;
  }
  const float off = fma_q(float(qmax), -p.inv_ln2, kBias);
  auto e = [&](int64_t c) { return valid(mrow, c) ? hw_exp2_bf20(fma_q(q(c), p.inv_ln2, off)) : 0.0f; };
  float sum;
  if (p.path == 0) {
    // Even / odd 32-column blocks into two sums.
    float s[2] = {0.0f, 0.0f};
    for (int64_t c0 = 0; c0 < C; c0 += 32) s[(c0 >> 5) & 1] = add_chunk(e, c0, C, s[(c0 >> 5) & 1]);
    sum = add_q(s[0], s[1]);
  } else {
    sum = chain_sum(e, C);
  }
  const float inv =
      round_float(hw_inv_f32(mul_q(sum, p.ofm_scale), inv_exc), NPU_F16, kHalfEven);
  for (int64_t c = 0; c < C; ++c) {
    const float y = fma_q(round_float(e(c), NPU_F16, kHalfEven), inv, p.zp_acc);
    store_int(out, p.t, i * C + c, float_to_int(y, p.t, p.shift, kHalfEven));
  }
}

struct QdqLogSoftmax {
  npu_dtype it, ot;
  float ifm_scale, ofm_scale, zp;
};

__global__ void qdq_logsoftmax_kernel(int64_t n, const void* in, const uint8_t* mask, void* out,
                                      int64_t C, QdqLogSoftmax p, const uint32_t* inv_exc) {
  NPU_TID;
  const uint8_t* mrow = mask ? mask + i * (C / 8) : nullptr;
  auto q = [&](int64_t c) { return float(load_int(in, p.it, i * C + c)); };
  const float osi = hw_inv_f32(p.ofm_scale, inv_exc);
  const float inv_ln2 = mul_q(kLog2e, p.ifm_scale);
  const float io = mul_q(p.ifm_scale, osi);
  int64_t lo, hi;
  int_range(p.it, &lo, &hi);
  int64_t qmax = lo;
  for (int64_t c = 0; c < C; ++c) {
    const int64_t v = load_int(in, p.it, i * C + c);
    if (valid(mrow, c) && v > qmax) qmax = v;
  }
  const float off = fma_q(float(qmax), -inv_ln2, kBias);
  const float qio = fma_q(float(qmax), -io, p.zp);
  auto e = [&](int64_t c) { return valid(mrow, c) ? hw_exp2_bf20(fma_q(q(c), inv_ln2, off)) : 0.0f; };
  // Four sums: even / odd 64-column block x low / high 32 columns.
  const float sum = sum4(e, C, [](int64_t k) { return 2 * int((k >> 1) & 1) + int(k & 1); });
  const float nl = fma_q(hw_approx_ln(sum), -osi, qio);
  for (int64_t c = 0; c < C; ++c)
    store_int(out, p.ot, i * C + c,
              float_to_int(valid(mrow, c) ? fma_q(q(c), io, nl) : p.zp, p.ot, 0, kHalfEven));
}

int inv_table(const uint32_t** exc) { return npu_emu_nlf_table(NPU_EMU_NLF_INV_F32, exc); }

int qdq_softmax(void* stream, const void* in, const uint8_t* mask, void* out, npu_dtype t,
                int bits, int64_t rows, int64_t C, int path, int shift_res, float inv_ln2,
                float ofm_scale, float zp_acc) {
  if (!is_int_bits(t, bits) || rows < 0 || C <= 0 || C % 32 || path < 0 || path > 2 ||
      (path == 0 && C % 64) || (path == 2 && C != 32) || shift_res < -32 || shift_res > 32)
    return -1;
  const uint32_t* exc;
  if (int rc = inv_table(&exc)) return rc;
  return launch(stream, rows, qdq_softmax_kernel, in, mask, out, C,
                QdqSoftmax{t, path, shift_res, inv_ln2, ofm_scale, zp_acc}, exc);
}

int qdq_logsoftmax(void* stream, const void* in, npu_dtype in_type, const uint8_t* mask,
                   void* out, npu_dtype out_type, int bits, int64_t rows, int64_t C,
                   float ifm_scale, float ofm_scale, int32_t ofm_zero_point) {
  if (!is_int_bits(in_type, bits) || !is_int_bits(out_type, bits) || rows < 0 || C <= 0 || C % 128)
    return -1;
  const uint32_t* exc;
  if (int rc = inv_table(&exc)) return rc;
  return launch(stream, rows, qdq_logsoftmax_kernel, in, mask, out, C,
                QdqLogSoftmax{in_type, out_type, ifm_scale, ofm_scale, float(ofm_zero_point)},
                exc);
}

}  // namespace

extern "C" {

int npu_softmax_bf16(void* stream, const void* in, void* out, int64_t rows, int64_t cols) {
  if (rows < 0 || cols <= 0) return -1;
  const uint32_t* exc;
  if (int rc = inv_table(&exc)) return rc;
  return launch(stream, rows, softmax_bf16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), cols, exc);
}

int npu_softmax_2pass_bf16(void* stream, const void* in, void* out, int64_t rows, int64_t cols,
                           int64_t chunk_cols, int64_t num_chunks) {
  if (rows < 0 || cols <= 0 || chunk_cols < 128 || chunk_cols % 64 || num_chunks <= 0 ||
      cols > chunk_cols * num_chunks)
    return -1;
  const uint32_t* exc;
  if (int rc = inv_table(&exc)) return rc;
  return launch(stream, rows, softmax_2pass_bf16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), cols, chunk_cols, num_chunks, exc);
}

int npu_softmax_fp16x16(void* stream, const void* in, const void* mask, void* out,
                        npu_dtype dtype, int64_t rows, int64_t C) {
  if ((dtype != NPU_BF16 && dtype != NPU_F16) || rows < 0 || C <= 0 || C % 32) return -1;
  const uint32_t* exc;
  if (int rc = inv_table(&exc)) return rc;
  return launch(stream, rows, softmax_fp16x16_kernel, static_cast<const uint16_t*>(in),
                static_cast<const uint8_t*>(mask), static_cast<uint16_t*>(out),
                dtype, C, exc);
}

int npu_softmax_fp8(void* stream, const void* in, const void* mask, void* out, int64_t rows,
                    int64_t C, float ifm_scale, float ofm_scale_recip) {
  if (rows < 0 || C <= 0 || C % 64) return -1;
  const uint32_t* exc;
  if (int rc = inv_table(&exc)) return rc;
  return launch(stream, rows, softmax_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<const uint8_t*>(mask), static_cast<uint8_t*>(out), C, ifm_scale,
                ofm_scale_recip, exc);
}

int npu_logsoftmax_bf16(void* stream, const void* in, void* out, npu_dtype dtype, int64_t rows,
                        int64_t cols) {
  if ((dtype != NPU_BF16 && dtype != NPU_F16) || rows < 0 || cols <= 0) return -1;
  return launch(stream, rows, logsoftmax_bf16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), dtype, cols);
}

int npu_logsoftmax_fp8(void* stream, const void* in, void* out, int64_t rows, int64_t cols,
                       float in_scale, float out_scale_inv) {
  if (rows < 0 || cols <= 0 || !(in_scale > 0.0f)) return -1;
  return launch(stream, rows, logsoftmax_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), cols, in_scale, out_scale_inv);
}

int npu_qdq_softmax_a8(void* stream, const void* in, const void* mask, void* out,
                       npu_dtype dtype, int64_t rows, int64_t C, int path, int shift_res,
                       float inv_ln2, float ofm_scale, float zp_acc) {
  return qdq_softmax(stream, in, static_cast<const uint8_t*>(mask), out, dtype, 8, rows, C,
                     path, shift_res, inv_ln2, ofm_scale, zp_acc);
}

int npu_qdq_softmax_a16(void* stream, const void* in, const void* mask, void* out,
                        npu_dtype dtype, int64_t rows, int64_t C, int path, int shift_res,
                        float inv_ln2, float ofm_scale, float zp_acc) {
  return qdq_softmax(stream, in, static_cast<const uint8_t*>(mask), out, dtype, 16, rows, C,
                     path, shift_res, inv_ln2, ofm_scale, zp_acc);
}

int npu_qdq_logsoftmax_a8(void* stream, const void* in, npu_dtype in_type, const void* mask,
                          void* out, npu_dtype out_type, int64_t rows, int64_t C,
                          float ifm_scale, float ofm_scale, int32_t ofm_zero_point) {
  return qdq_logsoftmax(stream, in, in_type, static_cast<const uint8_t*>(mask), out, out_type,
                        8, rows, C, ifm_scale, ofm_scale, ofm_zero_point);
}

int npu_qdq_logsoftmax_a16(void* stream, const void* in, npu_dtype in_type, const void* mask,
                           void* out, npu_dtype out_type, int64_t rows, int64_t C,
                           float ifm_scale, float ofm_scale, int32_t ofm_zero_point) {
  return qdq_logsoftmax(stream, in, in_type, static_cast<const uint8_t*>(mask), out, out_type,
                        16, rows, C, ifm_scale, ofm_scale, ofm_zero_point);
}

}  // extern "C"
