/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for include/npu_emu_matmul.h. One thread per output
 * element, each recomputing its own reductions; these are reference-speed
 * kernels.
 */
#include <hip/hip_runtime.h>

#include "npu_emu.h"
#include "npu_emu_device.h"
#include "npu_emu_launch.h"
#include "npu_emu_matmul.h"

using namespace npu_emu;

// The kernels round every fp32 operation separately; fused multiply-adds are
// written out (fmaf / mac_exact) where the NPU fuses.
#pragma clang fp contract(off)

namespace {

// Activation on the fp32 accumulator (ChainedPReluPolicy /
// ChainedPerChannelPReluPolicy / the fused leaky relu), then to_vector_sign.
__device__ inline int64_t activate_out(float v, npu_activation act, float act0, float act1,
                                       float slope, npu_dtype t) {
  if (act == NPU_ACT_CLIP) {
    v = fminf(fmaxf(v, act0), act1);
  } else if (act == NPU_ACT_LRELU) {
    v = fmaxf(v, fmaf(v, act0, act1));
  } else if (act == NPU_ACT_PRELU) {
    const float offset = fmaf(slope, -act0, act0);
    v = v < act0 ? fmaf(v, slope, offset) : v;
  }
  int64_t lo, hi;
  int_range(t, &lo, &hi);
  return sat_rne(v, lo, hi);
}

__global__ void gemm_qdq_int16x16_kernel(int64_t n, const void* a, npu_dtype a_type,
                                         const void* b, npu_dtype b_type, void* out,
                                         npu_dtype out_type, int64_t k, int64_t cols, float c0,
                                         float c1, float c2, float c3, npu_activation act,
                                         float act0, float act1) {
  NPU_TID;
  const int64_t row = i / cols, col = i % cols;
  int64_t lo = 0, mid = 0, hi = 0, wl = 0, wh = 0, asum = 0;
  for (int64_t q = 0; q < k; ++q) {
    const int64_t x = load_int(a, a_type, row * k + q);
    const int64_t y = load_int(b, b_type, q * cols + col);
    const int64_t xl = x & 255, xh = x >> 8, yl = y & 255, yh = y >> 8;
    lo += xl * yl;
    mid += xh * yl + xl * yh;
    hi += xh * yh;
    wl += yl;
    wh += yh;
    asum += x;
  }
  float v = mac_exact(wl, c3, c0);
  v = mac_exact(wh, c3 * 256.0f, v);
  v = mac_exact(lo, c2, v);
  v = mac_exact(mid, c2 * 256.0f, v);
  v = mac_exact(hi, c2 * 65536.0f, v);
  v = fmaf(float(asum), c1, v);
  store_int(out, out_type, i, activate_out(v, act, act0, act1, 0.0f, out_type));
}

template <bool kSplit16>
__global__ void conv_qdq_kernel(int64_t n, const void* ifm, npu_dtype ifm_type, const void* wts,
                                npu_dtype wts_type, void* out, npu_dtype out_type, int64_t w,
                                int64_t ci, int64_t kh, int64_t kw, int64_t co, int64_t wo,
                                int stride_h, int stride_w, int k_opt, const float* c0,
                                const float* c1, const float* c2, npu_activation act, float act0,
                                float act1, const float* slope) {
  NPU_TID;
  const int64_t oc = i % co, ox = (i / co) % wo, oy = i / (co * wo);
  int64_t conv_lo = 0, conv_hi = 0;
  // fp32 window sums of the channel sums, one add per tap, in the kernel's
  // order (see the header): all taps row-major, or on the k_opt path column 0
  // (odd Kw) plus the remaining columns alternating between the two halves
  // (lo, hi) of one accumulator register.
  const bool split_cols = k_opt && kw > 1;
  const int64_t first = kw % 2;  // first column summed into the lo half
  float s_all = 0.0f, s_col0 = 0.0f, s_lo = 0.0f, s_hi = 0.0f;
  bool e_all = true, e_col0 = true, e_lo = true, e_hi = true;
  auto add = [](float& acc, bool& empty, float v) {
    acc = empty ? v : acc + v;
    empty = false;
  };
  for (int64_t y = 0; y < kh; ++y) {
    for (int64_t x = 0; x < kw; ++x) {
      const int64_t pix = ((oy * stride_h + y) * w + ox * stride_w + x) * ci;
      const int64_t wrow = (y * kw + x) * ci;
      int64_t ch_sum = 0;
      for (int64_t c = 0; c < ci; ++c) {
        const int64_t v = load_int(ifm, ifm_type, pix + c);
        const int64_t g = load_int(wts, wts_type, (wrow + c) * co + oc);
        ch_sum += v;
        if (kSplit16) {
          conv_lo += (v & 255) * g;
          conv_hi += (v >> 8) * g;
        } else {
          conv_lo += v * g;
        }
      }
      const float cs = float(ch_sum);
      if (!split_cols) add(s_all, e_all, cs);
      else if (first && x == 0) add(s_col0, e_col0, cs);
      else if ((x - first) % 2 == 0) add(s_lo, e_lo, cs);
      else add(s_hi, e_hi, cs);
    }
  }
  const float sum_ifm = !split_cols ? s_all : first ? (s_col0 + s_lo) + s_hi : s_lo + s_hi;
  float v;
  if (kSplit16) {
    const float g = mac_exact(conv_hi, 256.0f, float(conv_lo));
    v = fmaf(g, c2[oc], c0[oc]);
  } else {
    v = mac_exact(conv_lo, c2[oc], c0[oc]);
  }
  if (c1) v = fmaf(sum_ifm, c1[oc], v);
  store_int(out, out_type, i,
            activate_out(v, act, act0, act1, slope ? slope[oc] : 0.0f, out_type));
}

bool is_i16(npu_dtype t) { return t == NPU_I16 || t == NPU_U16; }
bool is_i8(npu_dtype t) { return t == NPU_I8 || t == NPU_U8; }

template <bool kSplit16>
int conv_qdq(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
             npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h, int64_t w, int64_t ci,
             int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w, int k_opt,
             const float* c0, const float* c1, const float* c2, npu_activation act, float act0,
             float act1, const float* slope) {
  if (!(kSplit16 ? is_i16(ifm_type) : is_i8(ifm_type)) || !is_i8(wts_type) ||
      !is_small_int(out_type) || !c0 || !c2 || (act == NPU_ACT_PRELU) != (slope != nullptr) ||
      stride_h < 1 || stride_w < 1 || kh < 1 || kw < 1 || h < kh || w < kw || ci < 1 || co < 1)
    return -1;
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  return launch(stream, ho * wo * co, conv_qdq_kernel<kSplit16>, ifm, ifm_type, wts, wts_type,
                out, out_type, w, ci, kh, kw, co, wo, stride_h, stride_w, k_opt, c0, c1, c2, act,
                act0, act1, slope);
}

}  // namespace

extern "C" {

int npu_gemm_qdq_int16x16(void* stream, const void* a, npu_dtype a_type, const void* b,
                          npu_dtype b_type, void* out, npu_dtype out_type, int64_t m,
                          int64_t k, int64_t n, float c0, float c1, float c2, float c3,
                          npu_activation activation, float act0, float act1) {
  if (!is_i16(a_type) || !is_i16(b_type) || !is_i16(out_type) || m < 0 || k < 0 || n < 0 ||
      activation == NPU_ACT_PRELU)
    return -1;
  return launch(stream, m * n, gemm_qdq_int16x16_kernel, a, a_type, b, b_type, out, out_type, k,
                n, c0, c1, c2, c3, activation, act0, act1);
}

int npu_conv_qdq_int8x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                        npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h,
                        int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                        int stride_h, int stride_w, int k_opt, const float* c0,
                        const float* c1, const float* c2, npu_activation activation,
                        float act0, float act1, const float* slope) {
  return conv_qdq<false>(stream, ifm, ifm_type, wts, wts_type, out, out_type, h, w, ci, kh, kw,
                         co, stride_h, stride_w, k_opt, c0, c1, c2, activation, act0, act1, slope);
}

int npu_conv_qdq_int16x8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                         npu_dtype wts_type, void* out, npu_dtype out_type, int64_t h,
                         int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                         int stride_h, int stride_w, int k_opt, const float* c0,
                         const float* c1, const float* c2, npu_activation activation,
                         float act0, float act1, const float* slope) {
  return conv_qdq<true>(stream, ifm, ifm_type, wts, wts_type, out, out_type, h, w, ci, kh, kw,
                        co, stride_h, stride_w, k_opt, c0, c1, c2, activation, act0, act1, slope);
}

}  // extern "C"
