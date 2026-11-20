/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * bf16 matrix multiply and convolution. One thread per output element,
 * running its own chain of MMACs.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "matmul/bf16_matmul.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/mmac.h"

using namespace npu_emu;

namespace {

// The accumulator (fp32 bits) to bf16, half away; fp32 subnormals -> signed zero.
__device__ inline uint16_t acc_bf16(uint32_t acc) {
  return f32_to_bf16(__uint_as_float(acc), kHalfAway, kFtzIn);
}

__device__ inline uint16_t activate_bf16(uint16_t v, int act, float alpha) {
  float x = bf16_to_f32(v);
  if (act == 1) return f32_to_bf16(vmax(x, 0.0f), kHalfAway, kFtzIn);
  if (act == 2) return f32_to_bf16(vclamp(x, 0.0f, 6.0f), kHalfAway, kFtzIn);
  if (act == 3) return f32_to_bf16(vmax(x, mul16_f32(x, alpha, NPU_BF16)), kHalfAway, kFtzIn);
  return v;
}

__device__ inline uint32_t acc_init(const float* bias, int64_t i) {
  return bias ? __float_as_uint(bias[i]) : 0u;
}

__device__ uint32_t gemm_acc(const uint16_t* a, const uint16_t* b, const float* bias, int64_t k,
                             int64_t n, int64_t row, int64_t col) {
  uint32_t acc = acc_init(bias, col);
  for (int64_t k0 = 0; k0 < k; k0 += 32) {
    const int nv = int(k - k0 < 32 ? k - k0 : 32);
    acc = mmac_bf16(acc, a + row * k + k0, 1, b + k0 * n + col, n, nv);
  }
  return acc;
}

__global__ void gemm_bf16_kernel(int64_t total, const uint16_t* a, const uint16_t* b,
                                 const float* bias, uint16_t* out, int64_t m, int64_t k,
                                 int64_t n, int act, int transpose) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t row = i / n, col = i % n;
  const uint32_t acc = gemm_acc(a, b, bias, k, n, row, col);
  out[transpose ? col * m + row : i] = activate_bf16(acc_bf16(acc), act, 0.0f);
}

__global__ void gemm_qdq_bf16_kernel(int64_t total, const uint16_t* a, const uint16_t* b,
                                     const float* bias, uint16_t* out, int64_t k, int64_t n,
                                     int act, int q_signed, float inv, float scale, int zp) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const uint32_t acc = gemm_acc(a, b, bias, k, n, i / n, i % n);
  const float v = bf16_to_f32(activate_bf16(acc_bf16(acc), act, 0.0f));
  // -float(zp), not float(-zp): negating INT_MIN overflows. [default]
  const int64_t q = float_to_int(fmaf(v, inv, float(zp)), q_signed ? NPU_I16 : NPU_U16, 0, kHalfEven);
  const float d = mac_exact(q, scale, __fmul_rn(-float(zp), scale));
  out[i] = f32_to_bf16(d, kHalfEven, kFtzIn);
}

struct ConvBf16 {
  const uint16_t* ifm;
  const uint16_t* wts;
  int64_t w, ci, kh, kw, co, wo, kh_sv;
  int stride_h, stride_w;
};

// The MMAC chain for output (oy, ox, oc).
__device__ uint32_t conv_acc(const ConvBf16& p, int64_t oy, int64_t ox, int64_t oc, uint32_t acc) {
  uint16_t xv[32], wv[32];
  for (int64_t k0 = 0; k0 < p.kh; k0 += p.kh_sv) {
    const int64_t k1 = k0 + p.kh_sv < p.kh ? k0 + p.kh_sv : p.kh;
    if (p.ci < 32) {
      for (int64_t y = k0; y < k1; ++y) {
        const int64_t row = p.kw * p.ci;
        for (int64_t c0 = 0; c0 < row; c0 += 32) {
          const int nv = int(row - c0 < 32 ? row - c0 : 32);
          for (int q = 0; q < nv; ++q) {
            const int64_t x = (c0 + q) / p.ci, c = (c0 + q) % p.ci;
            xv[q] = p.ifm[((oy * p.stride_h + y) * p.w + ox * p.stride_w + x) * p.ci + c];
            wv[q] = p.wts[((y * p.kw + x) * p.ci + c) * p.co + oc];
          }
          acc = mmac_bf16(acc, xv, 1, wv, 1, nv);
        }
      }
    } else {
      for (int64_t c0 = 0; c0 < p.ci; c0 += 32) {
        const int nv = int(p.ci - c0 < 32 ? p.ci - c0 : 32);
        for (int64_t y = k0; y < k1; ++y)
          for (int64_t x = 0; x < p.kw; ++x) {
            const int64_t pix = ((oy * p.stride_h + y) * p.w + ox * p.stride_w + x) * p.ci + c0;
            acc = mmac_bf16(acc, p.ifm + pix, 1, p.wts + ((y * p.kw + x) * p.ci + c0) * p.co + oc,
                            p.co, nv);
          }
      }
    }
  }
  return acc;
}

__global__ void conv_bf16_kernel(int64_t total, ConvBf16 p, const float* bias, uint16_t* out,
                                 int act, float alpha) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t oc = i % p.co, ox = (i / p.co) % p.wo, oy = i / (p.co * p.wo);
  const uint32_t acc = conv_acc(p, oy, ox, oc, acc_init(bias, oc));
  out[i] = activate_bf16(acc_bf16(acc), act, alpha);
}

__global__ void conv_add_bf16_kernel(int64_t total, ConvBf16 p, const float* bias,
                                     const uint16_t* ifm2, uint16_t* out, int64_t h2, int64_t w2,
                                     int64_t c2, int type, int act) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t oc = i % p.co, ox = (i / p.co) % p.wo, oy = i / (p.co * p.wo);
  const int64_t ry = h2 == 1 ? 0 : oy, rx = w2 == 1 ? 0 : ox, rc = c2 == 1 ? 0 : oc;
  const float res = bf16_to_f32(ifm2[(ry * w2 + rx) * c2 + rc]);
  if (type == 0) {
    // seed: fp32(residual + bias)
    const uint32_t seed = bias ? __float_as_uint(add_q(res, bias[oc])) : __float_as_uint(res);
    out[i] = acc_bf16(conv_acc(p, oy, ox, oc, seed));
  } else {
    float v = bf16_to_f32(acc_bf16(conv_acc(p, oy, ox, oc, acc_init(bias, oc))));
    if (act == 1) v = vmax(v, 0.0f);
    out[i] = f32_to_bf16(add_q(v, res), kHalfEven, kFtzIn);
  }
}

__global__ void dwc_bf16_kernel(int64_t total, const uint16_t* ifm, const uint16_t* wts,
                                const float* bias, uint16_t* out, int64_t w, int64_t c,
                                int64_t kh, int64_t kw, int64_t wo, int stride_h, int stride_w,
                                int act, float alpha) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t oc = i % c, ox = (i / c) % wo, oy = i / (c * wo);
  float acc = bias ? bias[oc] : 0.0f;
  for (int64_t y = 0; y < kh; ++y)
    for (int64_t x = 0; x < kw; ++x)
      acc = add_q(
          acc, mul_q(bf16_to_f32(ifm[((oy * stride_h + y) * w + ox * stride_w + x) * c + oc]),
                        bf16_to_f32(wts[(y * kw + x) * c + oc])));
  float v = round_float(acc, NPU_BF16, kHalfAway, kFtzIn);
  if (act == 1) v = vmax(v, 0.0f);
  else if (act == 2) v = vclamp(v, 0.0f, 6.0f);
  else if (act == 3) v = vmax(v, mul16_f32(alpha, v, NPU_BF16));
  out[i] = f32_to_bf16(v, kHalfAway, kFtzIn);
}

int gemm_bf16(void* stream, const uint16_t* a, const uint16_t* b, const float* bias,
              uint16_t* out, int64_t m, int64_t k, int64_t n, int activation, int transpose) {
  if (!a || !b || !out || m < 1 || k < 1 || n < 1 || activation < 0 || activation > 2) return -1;
  return launch(stream, m * n, gemm_bf16_kernel, a, b, bias, out, m, k, n, activation, transpose);
}

int gemm_qdq_bf16(void* stream, const uint16_t* a, const uint16_t* b, const float* bias,
                  uint16_t* out, int64_t m, int64_t k, int64_t n, int activation, int q_signed,
                  float inv_scale, float scale, int zero_point) {
  if (!a || !b || !out || m < 1 || k < 1 || n < 1 || activation < 0 || activation > 2) return -1;
  return launch(stream, m * n, gemm_qdq_bf16_kernel, a, b, bias, out, k, n, activation, q_signed,
                inv_scale, scale, zero_point);
}

// max_act: the largest activation code the kernel has.
int conv_bf16(void* stream, const uint16_t* ifm, const uint16_t* wts, const float* bias,
              uint16_t* out, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
              int64_t co, int stride_h, int stride_w, int64_t kh_sv, int activation, float alpha,
              int max_act) {
  if (!ifm || !wts || !out || h < kh || w < kw || ci < 1 || co < 1 || kh_sv < 1 ||
      stride_h < 1 || stride_w < 1 || activation < 0 || activation > max_act)
    return -1;
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  const ConvBf16 p{ifm, wts, w, ci, kh, kw, co, wo, kh_sv, stride_h, stride_w};
  return launch(stream, ho * wo * co, conv_bf16_kernel, p, bias, out, activation,
                scalar_cast(alpha, NPU_BF16, kHalfEven));
}

int dwc_bf16(void* stream, const uint16_t* ifm, const uint16_t* wts, const float* bias,
             uint16_t* out, int64_t h, int64_t w, int64_t c, int64_t kh, int64_t kw,
             int stride_h, int stride_w, int activation, float alpha) {
  if (!ifm || !wts || !out || h < kh || w < kw || c < 1 || stride_h < 1 || stride_w < 1 ||
      activation < 0 || activation > 3)
    return -1;
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  return launch(stream, ho * wo * c, dwc_bf16_kernel, ifm, wts, bias, out, w, c, kh, kw, wo,
                stride_h, stride_w, activation, scalar_cast(alpha, NPU_BF16, kHalfEven));
}

}  // namespace

extern "C" {

int npu_gemm_bf16x16(void* stream, const uint16_t* a, const uint16_t* b, const float* bias,
                     uint16_t* out, int64_t m, int64_t k, int64_t n, int activation) {
  return gemm_bf16(stream, a, b, bias, out, m, k, n, activation, 0);
}

int npu_gemm_bf16x16_buffer(void* stream, const uint16_t* a, const uint16_t* b,
                            const float* bias, uint16_t* out, int64_t m, int64_t k, int64_t n,
                            int activation) {
  return gemm_bf16(stream, a, b, bias, out, m, k, n, activation, 0);
}

int npu_gemm_bf16x16_ifm_reuse(void* stream, const uint16_t* a, const uint16_t* b,
                               const float* bias, uint16_t* out, int64_t m, int64_t k, int64_t n,
                               int activation) {
  return gemm_bf16(stream, a, b, bias, out, m, k, n, activation, 0);
}

int npu_gemm_bf16x16_transposed(void* stream, const uint16_t* a, const uint16_t* b,
                                const float* bias, uint16_t* out, int64_t m, int64_t k,
                                int64_t n, int activation) {
  return gemm_bf16(stream, a, b, bias, out, m, k, n, activation, 1);
}

int npu_gemm_quant_dequant_bf16x16(void* stream, const uint16_t* a, const uint16_t* b,
                                   const float* bias, uint16_t* out, int64_t m, int64_t k,
                                   int64_t n, int activation, int q_signed, float inv_scale,
                                   float scale, int zero_point) {
  return gemm_qdq_bf16(stream, a, b, bias, out, m, k, n, activation, q_signed, inv_scale, scale,
                       zero_point);
}

int npu_gemm_quant_dequant_bf16x16_ifm_reuse(void* stream, const uint16_t* a, const uint16_t* b,
                                             const float* bias, uint16_t* out, int64_t m,
                                             int64_t k, int64_t n, int activation, int q_signed,
                                             float inv_scale, float scale, int zero_point) {
  return gemm_qdq_bf16(stream, a, b, bias, out, m, k, n, activation, q_signed, inv_scale, scale,
                       zero_point);
}

int npu_conv_bf16x16(void* stream, const uint16_t* ifm, const uint16_t* wts, const float* bias,
                     uint16_t* out, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                     int64_t co, int stride_h, int stride_w, int64_t kh_sv, int activation) {
  return conv_bf16(stream, ifm, wts, bias, out, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv,
                   activation, 0.0f, 2);
}

int npu_conv_bf16x16_buffer(void* stream, const uint16_t* ifm, const uint16_t* wts,
                            const float* bias, uint16_t* out, int64_t h, int64_t w, int64_t ci,
                            int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                            int64_t kh_sv, int activation, float alpha) {
  return conv_bf16(stream, ifm, wts, bias, out, h, w, ci, kh, kw, co, stride_h, stride_w, kh_sv,
                   activation, alpha, 3);
}

int npu_conv_add_bf16x16(void* stream, const uint16_t* ifm, const uint16_t* wts, const float* bias,
                         const uint16_t* ifm2, uint16_t* out, int64_t h, int64_t w, int64_t ci,
                         int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
                         int64_t kh_sv, int64_t h2, int64_t w2, int64_t c2, int type,
                         int activation) {
  if (!ifm || !wts || !ifm2 || !out || h < kh || w < kw || ci < 1 || co < 1 || kh_sv < 1 ||
      stride_h < 1 || stride_w < 1 || (type != 0 && type != 1) ||
      (type == 0 && activation != 0) || (type == 1 && activation != 0 && activation != 1))
    return -1;
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  if ((h2 != 1 && h2 != ho) || (w2 != 1 && w2 != wo) || (c2 != 1 && c2 != co)) return -1;
  const ConvBf16 p{ifm, wts, w, ci, kh, kw, co, wo, kh_sv, stride_h, stride_w};
  return launch(stream, ho * wo * co, conv_add_bf16_kernel, p, bias, ifm2, out, h2, w2, c2, type,
                activation);
}

int npu_depthwise_conv_bf16x16(void* stream, const uint16_t* ifm, const uint16_t* wts,
                               const float* bias, uint16_t* out, int64_t h, int64_t w, int64_t c,
                               int64_t kh, int64_t kw, int stride_h, int stride_w,
                               int activation, float alpha) {
  return dwc_bf16(stream, ifm, wts, bias, out, h, w, c, kh, kw, stride_h, stride_w, activation,
                  alpha);
}

int npu_depthwise_conv_bf16x16_c32(void* stream, const uint16_t* ifm, const uint16_t* wts,
                                   const float* bias, uint16_t* out, int64_t h, int64_t w,
                                   int64_t c, int64_t kh, int64_t kw, int stride_h, int stride_w,
                                   int activation, float alpha) {
  return dwc_bf16(stream, ifm, wts, bias, out, h, w, c, kh, kw, stride_h, stride_w, activation,
                  alpha);
}

}  // extern "C"
