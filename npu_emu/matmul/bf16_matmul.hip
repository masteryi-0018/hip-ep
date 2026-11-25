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
#include "common/act.h"
#include "common/bits.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/mmac.h"
#include "common/quant.h"

using namespace npu_emu;

namespace {

// The accumulator (fp32 bits) to bf16, half away; fp32 subnormals -> signed zero.
__device__ inline uint16_t acc_bf16(uint32_t acc) {
  return f32_to_bf16(bits_f32(acc), kHalfAway, kFtzIn);
}

// The activated accumulator as a bf16 code.
__device__ inline uint16_t act_bf16(uint32_t acc, int act, float alpha) {
  return f32_to_bf16(activate_bf16(acc_to_bf16(bits_f32(acc)), act, alpha), kHalfAway, kFtzIn);
}

// fp32 bits of a bias element: bf16 widens exactly.
__device__ inline uint32_t acc_init(const void* bias, npu_dtype t, int64_t i) {
  if (!bias) return 0u;
  return t == NPU_BF16 ? uint32_t(static_cast<const uint16_t*>(bias)[i]) << 16
                       : static_cast<const uint32_t*>(bias)[i];
}

__device__ uint32_t gemm_acc(const uint16_t* a, const uint16_t* b, const void* bias,
                             npu_dtype bias_type, int64_t k, int64_t n, int64_t row, int64_t col) {
  uint32_t acc = acc_init(bias, bias_type, col);
  for (int64_t k0 = 0; k0 < k; k0 += 32) {
    const int nv = int(k - k0 < 32 ? k - k0 : 32);
    acc = mmac_bf16(acc, a + row * k + k0, 1, b + k0 * n + col, n, nv);
  }
  return acc;
}

__global__ void gemm_kernel(int64_t n_out, const uint16_t* a, const uint16_t* b,
                            const void* bias, npu_dtype bias_type, uint16_t* out, int64_t m,
                            int64_t k, int64_t n, int act, int transpose) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  const int64_t row = i / n, col = i % n;
  const uint32_t acc = gemm_acc(a, b, bias, bias_type, k, n, row, col);
  out[transpose ? col * m + row : i] = act_bf16(acc, act, 0.0f);
}

__global__ void gemm_quant_dequant_kernel(int64_t total, const uint16_t* a, const uint16_t* b,
                                          const void* bias, npu_dtype bias_type, uint16_t* out,
                                          int64_t k, int64_t n, float quant_mult,
                                          float dequant_mult, float zero_point, int qdq_sign,
                                          int act) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const uint32_t acc = gemm_acc(a, b, bias, bias_type, k, n, i / n, i % n);
  const float v = activate_bf16(acc_to_bf16(bits_f32(acc)), act, 0.0f);
  const int64_t q = quant(v, quant_mult, zero_point, qdq_sign ? NPU_I16 : NPU_U16);
  const float d = dequant(q, dequant_mult, dq_offset(zero_point, dequant_mult));
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

__global__ void conv_kernel(int64_t total, ConvBf16 p, const uint16_t* bias, uint16_t* out,
                                 int act, float alpha) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t oc = i % p.co, ox = (i / p.co) % p.wo, oy = i / (p.co * p.wo);
  const uint32_t acc = conv_acc(p, oy, ox, oc, acc_init(bias, NPU_BF16, oc));
  out[i] = act_bf16(acc, act, alpha);
}

__global__ void conv_add_kernel(int64_t total, ConvBf16 p, const uint16_t* bias,
                                const uint16_t* ifm2, uint16_t* out, int64_t h2, int64_t w2,
                                int type, int act) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t oc = i % p.co, ox = (i / p.co) % p.wo, oy = i / (p.co * p.wo);
  const int64_t ry = h2 == 1 ? 0 : oy, rx = w2 == 1 ? 0 : ox;
  const float res = bf16_to_f32(ifm2[(ry * w2 + rx) * p.co + oc]);
  if (type == 0) {
    // seed: fp32(residual + bias)
    const uint32_t seed = bias ? f32_bits(add_q(res, bf16_to_f32(bias[oc])))
                               : f32_bits(res);
    out[i] = acc_bf16(conv_acc(p, oy, ox, oc, seed));
  } else {
    const uint32_t acc = conv_acc(p, oy, ox, oc, acc_init(bias, NPU_BF16, oc));
    const float v = activate_bf16(acc_to_bf16(bits_f32(acc)), act, 0.0f);
    out[i] = f32_to_bf16(add_q(v, res), kHalfEven, kFtzIn);
  }
}

__global__ void depthwise_kernel(int64_t total, const uint16_t* ifm, const uint16_t* wts,
                                 const uint16_t* bias, uint16_t* out, int64_t w, int64_t c,
                                int64_t kh, int64_t kw, int64_t wo, int stride_h, int stride_w,
                                int act, float alpha) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t oc = i % c, ox = (i / c) % wo, oy = i / (c * wo);
  float acc = bias ? bf16_to_f32(bias[oc]) : 0.0f;
  for (int64_t y = 0; y < kh; ++y)
    for (int64_t x = 0; x < kw; ++x)
      acc = add_q(
          acc, mul_q(bf16_to_f32(ifm[((oy * stride_h + y) * w + ox * stride_w + x) * c + oc]),
                        bf16_to_f32(wts[(y * kw + x) * c + oc])));
  const float v = acc_to_bf16(acc);
  out[i] = f32_to_bf16(activate_bf16(v, act, alpha), kHalfAway, kFtzIn);
}

const uint16_t* u16(const void* p) { return static_cast<const uint16_t*>(p); }
bool bias_type_ok(npu_dtype t) { return t == NPU_BF16 || t == NPU_F32; }
// m / n may be 0 (no output); k is the reduction length.
bool sizes_ok(int64_t m, int64_t k, int64_t n) { return m >= 0 && k >= 1 && n >= 0; }
bool stride_ok(int s, int max_stride) { return s == 1 || s == 2 || (max_stride == 4 && s == 4); }

int gemm(void* stream, const void* a, const void* b, const void* bias, npu_dtype bias_type,
         void* out, int64_t m, int64_t k, int64_t n, int activation, int transpose) {
  if (any_null(a, b, out) || !bias_type_ok(bias_type) || !sizes_ok(m, k, n) || activation < 0 ||
      activation > 2)
    return -1;
  return launch(stream, m * n, gemm_kernel, u16(a), u16(b), bias, bias_type,
                static_cast<uint16_t*>(out), m, k, n, activation, transpose);
}

int gemm_quant_dequant(void* stream, const void* a, const void* b, const void* bias,
                       npu_dtype bias_type, void* out, int64_t m, int64_t k, int64_t n,
                       float quant_mult, float dequant_mult, float zero_point, int qdq_sign,
                       int activation) {
  const float lo = qdq_sign ? -32768.0f : 0.0f, hi = qdq_sign ? 32767.0f : 65535.0f;
  if (any_null(a, b, out) || !bias_type_ok(bias_type) || !sizes_ok(m, k, n) || activation < 0 ||
      activation > 2 || (qdq_sign != 0 && qdq_sign != 1) ||
      !(zero_point >= lo && zero_point <= hi))
    return -1;
  return launch(stream, m * n, gemm_quant_dequant_kernel, u16(a), u16(b), bias, bias_type,
                static_cast<uint16_t*>(out), k, n, quant_mult, dequant_mult, zero_point,
                qdq_sign, activation);
}

// Output extent (0 for an empty image, h or w 0); false for sizes, windows or strides the
// kernels reject.
bool conv_geometry(int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                   int stride_h, int stride_w, int max_stride, int64_t* ho, int64_t* wo) {
  if (ci < 1 || co < 1 || kh < 1 || kw < 1 || h < 0 || w < 0 || (h > 0 && h < kh) ||
      (w > 0 && w < kw) || !stride_ok(stride_h, max_stride) || !stride_ok(stride_w, max_stride))
    return false;
  *ho = h == 0 ? 0 : (h - kh) / stride_h + 1;
  *wo = w == 0 ? 0 : (w - kw) / stride_w + 1;
  return true;
}

// max_act: the largest activation code the kernel has.
int conv(void* stream, const void* ifm, const void* wts, const void* bias, void* out, int64_t h,
         int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w,
         int max_stride, int64_t kh_sv, float alpha, int activation, int max_act) {
  int64_t ho, wo;
  if (any_null(ifm, wts, out) || kh_sv < 1 || activation < 0 || activation > max_act ||
      !conv_geometry(h, w, ci, kh, kw, co, stride_h, stride_w, max_stride, &ho, &wo))
    return -1;
  const ConvBf16 p{u16(ifm), u16(wts), w, ci, kh, kw, co, wo, kh_sv, stride_h, stride_w};
  return launch(stream, ho * wo * co, conv_kernel, p, u16(bias), static_cast<uint16_t*>(out),
                activation, scalar_cast(alpha, NPU_BF16, kHalfEven));
}

int depthwise(void* stream, const void* ifm, const void* wts, const void* bias, void* out,
              int64_t h, int64_t w, int64_t c, int64_t kh, int64_t kw, int stride_h,
              int stride_w, float alpha, int activation) {
  int64_t ho, wo;
  if (any_null(ifm, wts, out) || kh * kw < 2 || activation < 0 || activation > 3 ||
      !conv_geometry(h, w, c, kh, kw, c, stride_h, stride_w, 2, &ho, &wo))
    return -1;
  return launch(stream, ho * wo * c, depthwise_kernel, u16(ifm), u16(wts), u16(bias),
                static_cast<uint16_t*>(out), w, c, kh, kw, wo, stride_h, stride_w, activation,
                scalar_cast(alpha, NPU_BF16, kHalfEven));
}

}  // namespace

extern "C" {

int npu_gemm_bf16x16(void* stream, const void* a, const void* b, const void* bias,
                     npu_dtype bias_type, void* out, int64_t m, int64_t k, int64_t n,
                     int activation) {
  return gemm(stream, a, b, bias, bias_type, out, m, k, n, activation, 0);
}

int npu_gemm_bf16x16_buffer(void* stream, const void* a, const void* b, const void* bias,
                            npu_dtype bias_type, void* out, int64_t m, int64_t k, int64_t n,
                            int activation) {
  return gemm(stream, a, b, bias, bias_type, out, m, k, n, activation, 0);
}

int npu_gemm_bf16x16_ifm_reuse(void* stream, const void* a, const void* b, const void* bias,
                               npu_dtype bias_type, void* out, int64_t m, int64_t k, int64_t n,
                               int activation) {
  return gemm(stream, a, b, bias, bias_type, out, m, k, n, activation, 0);
}

int npu_gemm_bf16x16_transposed(void* stream, const void* a, const void* b, const void* bias,
                                npu_dtype bias_type, void* out, int64_t m, int64_t k, int64_t n,
                                int activation) {
  return gemm(stream, a, b, bias, bias_type, out, m, k, n, activation, 1);
}

int npu_gemm_quant_dequant_bf16x16(void* stream, const void* a, const void* b, const void* bias,
                                   npu_dtype bias_type, void* out, int64_t m, int64_t k,
                                   int64_t n, float quant_mult, float dequant_mult,
                                   float zero_point, int qdq_sign, int activation) {
  return gemm_quant_dequant(stream, a, b, bias, bias_type, out, m, k, n, quant_mult,
                            dequant_mult, zero_point, qdq_sign, activation);
}

int npu_gemm_quant_dequant_bf16x16_ifm_reuse(void* stream, const void* a, const void* b,
                                             const void* bias, npu_dtype bias_type, void* out,
                                             int64_t m, int64_t k, int64_t n, float quant_mult,
                                             float dequant_mult, float zero_point, int qdq_sign,
                                             int activation) {
  return gemm_quant_dequant(stream, a, b, bias, bias_type, out, m, k, n, quant_mult,
                            dequant_mult, zero_point, qdq_sign, activation);
}

int npu_conv_bf16x16(void* stream, const void* ifm, const void* wts, const void* bias, void* out,
                     int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                     int stride_h, int stride_w, int64_t kh_sv, int activation) {
  return conv(stream, ifm, wts, bias, out, h, w, ci, kh, kw, co, stride_h, stride_w, 2, kh_sv,
              0.0f, activation, 2);
}

int npu_conv_bf16x16_buffer(void* stream, const void* ifm, const void* wts, const void* bias,
                            void* out, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
                            int64_t co, int stride_h, int stride_w, int64_t kh_sv,
                            float lrelu_alpha, int activation) {
  return conv(stream, ifm, wts, bias, out, h, w, ci, kh, kw, co, stride_h, stride_w, 4, kh_sv,
              lrelu_alpha, activation, 3);
}

int npu_conv_add_bf16x16(void* stream, const void* ifm, const void* ifm2, const void* wts,
                         const void* bias, void* out, int64_t h, int64_t w, int64_t ci,
                         int64_t kh, int64_t kw, int64_t co, int64_t h2, int64_t w2,
                         int stride_h, int stride_w, int64_t kh_sv, int activation_type,
                         int activation) {
  int64_t ho, wo;
  if (any_null(ifm, wts, ifm2, out) || kh_sv < 1 ||
      !conv_geometry(h, w, ci, kh, kw, co, stride_h, stride_w, 2, &ho, &wo) ||
      (activation_type != 0 && activation_type != 1) ||
      (activation_type == 0 && activation != 0) ||
      (activation_type == 1 && activation != 0 && activation != 1))
    return -1;
  if ((h2 != 1 && h2 != ho) || (w2 != 1 && w2 != wo) ||
      (activation_type == 1 && (h2 != ho || w2 != wo)))
    return -1;
  const ConvBf16 p{u16(ifm), u16(wts), w, ci, kh, kw, co, wo, kh_sv, stride_h, stride_w};
  return launch(stream, ho * wo * co, conv_add_kernel, p, u16(bias), u16(ifm2),
                static_cast<uint16_t*>(out), h2, w2, activation_type, activation);
}

int npu_depthwise_conv_bf16x16(void* stream, const void* ifm, const void* wts, const void* bias,
                               void* out, int64_t h, int64_t w, int64_t c, int64_t kh,
                               int64_t kw, int stride_h, int stride_w, float alpha,
                               int activation) {
  return depthwise(stream, ifm, wts, bias, out, h, w, c, kh, kw, stride_h, stride_w, alpha,
                   activation);
}

int npu_depthwise_conv_bf16x16_c32(void* stream, const void* ifm, const void* wts,
                                   const void* bias, void* out, int64_t h, int64_t w, int64_t c,
                                   int64_t kh, int64_t kw, int stride_h, int stride_w,
                                   float alpha, int activation) {
  return depthwise(stream, ifm, wts, bias, out, h, w, c, kh, kw, stride_h, stride_w, alpha,
                   activation);
}

}  // extern "C"
