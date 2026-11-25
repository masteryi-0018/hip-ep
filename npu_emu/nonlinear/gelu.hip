/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * GeLU / Erf kernels. One thread per element, except the bfp16 GeLU (one thread per
 * 8-value block).
 */
#include "nonlinear/gelu.h"

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#include "common/act.h"
#include "common/bits.h"
#include "common/block_fp.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/mmac.h"
#include "common/nlf.h"
#include "common/quant.h"

namespace {

using namespace npu_emu;

// Erf coefficients (exact in bf16 and fp16).
constexpr float kQ1 = 1.125f;
constexpr float kQ3 = -0.376953125f;
constexpr float kQ5 = 0.11279296875f;

// GeLU of a 16-bit x, stored half away from zero; bf16 subnormal results flush to a
// signed zero.
__device__ inline uint16_t gelu16(float x, npu_dtype ht) {
  return encode_float(hw_gelu_f32(x), ht, kHalfAway, kFtzIn);
}

__global__ void gelu_bf16_kernel(int64_t n, const uint16_t* in, uint16_t* out, npu_dtype ht) {
  NPU_TID;
  out[i] = gelu16(decode_float(in[i], ht), ht);
}

// One thread per bfp16 block; the decode aligns to the largest exponent of its 64-group.
__global__ void gelu_bfp16_kernel(int64_t n, const uint8_t* in, uint8_t* out) {
  NPU_TID;  // n = number of blocks
  float y[8];
  for (int k = 0; k < 8; ++k)
    y[k] = bf16_to_f32(gelu16(decode_bfp16_identity(in, i & ~int64_t(7), n, i, k), NPU_BF16));
  encode_bfp16(y, kHalfAway, out + i * 9);
}

__global__ void gelu_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, npu_dtype ht,
                                float ifm_scale, float recip) {
  NPU_TID;
  const float y = decode_float(gelu16(fp8_in(in, i, n, ifm_scale, ht, kHalfAway), ht), ht);
  // exact: two 16-bit significands
  out[i] = fp8_out(y, recip, kHalfAway);
}

__global__ void qdq_gelu_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                                npu_dtype out_type, int32_t zi, float zo, float ka, float c,
                                float kdsp) {
  NPU_TID;
  const int64_t qi = load_int(in, in_type, i) - zi;
  const int64_t aq = qi < 0 ? -qi : qi;
  const int64_t rq = qi > 0 ? qi : 0;
  const float v = mac_exact(aq, ka, kGeluK.kb);
  const float e = hw_exp2_bf20(msc_q(v, v, kdsp));
  const float lo = mac_exact(aq, neg_f32(e), zo);
  store_int(out, out_type, i, requant(rq, c, lo, out_type, 0, kHalfAway));
}

// v >> s with rounding mode rnd (0 ceil, 1 nearest ties toward zero, 2 away from zero),
// symmetric int8 saturation.
__device__ inline int srs_x8(int v, int s, int rnd) {
  const Rnd r = rnd == 0 ? kCeil : rnd == 1 ? kHalfZero : kAwayFromZero;
  return int(saturate_sym(shift_round(v, s, r), NPU_I8));
}

__global__ void xint8_gelu_kernel(int64_t n, const int8_t* in, int8_t* out, bool k1, bool pe,
                                  int rnd, int a_shift, int b_shift, int q4_shift, int r_shift,
                                  int K, int P) {
  NPU_TID;
  const int q = in[i];
  const int A = q * (1 << a_shift);
  const int q4 = srs_x8(A, q4_shift, rnd);
  int R = srs_x8(A, r_shift, rnd);
  R = R > 0 ? R : 0;
  const int B = k1 ? q : srs_x8(A, b_shift, rnd);
  const int aq4 = q4 < 0 ? -q4 : q4;
  const int md = K - aq4 > 0 ? K - aq4 : 0;
  int h = P < md ? P : md;
  if (pe && k1) {
    // min(max(min(P, m), max(K + 1 - |q4|, 0) - m), |q4| + max(P - 2, 0)) in int8
    const int k1v = int8_t(K + 1), p2 = P > 2 ? P - 2 : 0;
    const int md1 = k1v - aq4 > 0 ? k1v - aq4 : 0;
    const int up = int8_t(md1 - md), cap = int8_t(aq4 + p2);
    h = h > up ? h : up;
    h = h < cap ? h : cap;
  }
  const int8_t t = int8_t(R - h);  // wraps
  out[i] = int8_t(B > t ? B : t);
}

// The Erf polynomial in t (f16 or bf16, half to even) of an fp32 x that is already a
// t value.
__device__ inline uint16_t erf16(float x, npu_dtype ht) {
  auto t = [ht](float v) { return round_float(v, ht, kHalfEven); };
  const float x2 = t(mul_q(x, x));
  const float x3 = t(mul_q(x2, x));
  const float x4 = t(mul_q(x2, x2));
  const float x5 = t(mul_q(x4, x));
  const float a1 = mul_q(kQ1, x);
  const float a2 = add_q(a1, mul_q(kQ3, x3));
  const float a3 = add_q(a2, mul_q(kQ5, x5));
  return encode_float(vclamp(t(a3), -1.0f, 1.0f), ht, kHalfEven);
}

__global__ void erf_bf16_kernel(int64_t n, const uint16_t* in, uint16_t* out, npu_dtype ht) {
  NPU_TID;
  // bf16 subnormal inputs read as a signed zero
  out[i] = erf16(decode_float(in[i], ht, kFtzIn), ht);
}

__global__ void erf_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, float scale,
                               float q3, float q5, float lim) {
  NPU_TID;
  const float v = fp8_decode_identity(in, i, n);
  const float x = mul_q(v, scale);
  const float x2 = mul_q(x, x);
  const float p35 = fma_q(x2, q5, q3);
  const float p135 = fma_q(x2, p35, kQ1);
  float r = vclamp(mul_q(x, p135), neg_f32(lim), lim);  // NaN -> +NaN
  if (r == 0.0f) r = 0.0f;
  out[i] = f32_to_fp8(r, kHalfAway);
}

__global__ void erf_qdq_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                               npu_dtype out_type, float c1, float c0, float c1o, float c0o) {
  NPU_TID;
  const float xf = dequant(load_int(in, in_type, i), c1, c0);
  const uint16_t eb = erf16(round_float(xf, NPU_BF16, kHalfEven), NPU_BF16);
  // a NaN converts to 0
  const float y = add_q(mul_q(bf16_to_f32(eb), c1o), c0o);
  store_int(out, out_type, i, float_to_int(y, out_type, 0, kHalfEven));
}

int qdq_gelu(void* stream, const void* in, npu_dtype in_type, void* out, npu_dtype out_type,
             int bits, int64_t n, int32_t dq_zp, float q_zp, float ka, float c, float kdsp) {
  if (!qdq_args_ok(in, in_type, out, out_type, bits, n)) return -1;
  return launch(stream, n, qdq_gelu_kernel, in, in_type, out, out_type, dq_zp, q_zp, ka, c,
                kdsp);
}

int erf_qdq(void* stream, const void* in, npu_dtype in_type, void* out, npu_dtype out_type,
            int bits, int64_t n, float c1, float c0, float c1_out, float c0_out) {
  if (!qdq_args_ok(in, in_type, out, out_type, bits, n)) return -1;
  return launch(stream, n, erf_qdq_kernel, in, in_type, out, out_type, c1, c0,
                scalar_cast(c1_out, NPU_BF16, kHalfEven), scalar_cast(c0_out, NPU_BF16, kHalfEven));
}

}  // namespace

extern "C" {

int npu_gelu_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n) {
  if (!is_half_float(in_type) || n < 0 || any_null(in, out)) return -1;
  return launch(stream, n, gelu_bf16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), in_type);
}

int npu_gelu_bfp16(void* stream, const void* in, void* out, int64_t n) {
  // whole sub-volumes of 512 elements
  if (n < 0 || n % 512 || any_null(in, out)) return -1;
  return launch(stream, n / 8, gelu_bfp16_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out));
}

int npu_elementwise_gelu_fp8(void* stream, const void* in, void* out, int64_t n,
                             npu_dtype compute_type, float ifm_scale, float ofm_scale_recip) {
  if (!is_half_float(compute_type) || n < 0 || any_null(in, out)) return -1;
  const npu_dtype ht = compute_type;
  // the scale is cast half to even for bf16, half away for fp16
  return launch(stream, n, gelu_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), ht, ifm_scale,
                scalar_cast(ofm_scale_recip, ht, kHalfAway));
}

int npu_qdq_gelu_exp2_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                         npu_dtype out_type, int64_t n, int32_t dq_zp, float q_zp, float ka,
                         float c, float kdsp) {
  return qdq_gelu(stream, in, in_type, out, out_type, 8, n, dq_zp, q_zp, ka, c, kdsp);
}

int npu_qdq_gelu_exp2_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                          npu_dtype out_type, int64_t n, int32_t dq_zp, float q_zp, float ka,
                          float c, float kdsp) {
  return qdq_gelu(stream, in, in_type, out, out_type, 16, n, dq_zp, q_zp, ka, c, kdsp);
}

int npu_xint8_gelu(void* stream, const void* in, void* out, int64_t n, uint8_t up_path,
                   uint8_t rnd_id, uint8_t a_shift, uint8_t b_shift, uint8_t q4_shift,
                   uint8_t r_shift, int8_t K, int8_t P, int precision_enhanced) {
  auto sh_ok = [](uint8_t s) { return s <= 23; };
  if (n < 0 || any_null(in, out) || rnd_id > 2 || !sh_ok(a_shift) || !sh_ok(b_shift) ||
      !sh_ok(q4_shift) || !sh_ok(r_shift) || (precision_enhanced != 0 && precision_enhanced != 1))
    return -1;
  return launch(stream, n, xint8_gelu_kernel, static_cast<const int8_t*>(in),
                static_cast<int8_t*>(out), up_path != 0 && a_shift == 1, precision_enhanced != 0,
                int(rnd_id), int(a_shift), int(b_shift), int(q4_shift), int(r_shift), int(K),
                int(P));
}

int npu_elementwise_erf_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                             int64_t n) {
  if (!is_half_float(in_type) || n < 0 || any_null(in, out)) return -1;
  return launch(stream, n, erf_bf16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), in_type);
}

int npu_elementwise_erf_fp8(void* stream, const void* in, void* out, int64_t n,
                            float ifm_coeff, float q3_coeff, float q5_coeff, float clamp_coeff) {
  if (n < 0 || any_null(in, out)) return -1;
  return launch(stream, n, erf_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), ifm_coeff, q3_coeff, q5_coeff, clamp_coeff);
}

int npu_elementwise_erf_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, float c1, float c0, float c1_out,
                           float c0_out) {
  return erf_qdq(stream, in, in_type, out, out_type, 8, n, c1, c0, c1_out, c0_out);
}

int npu_elementwise_erf_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, float c1, float c0, float c1_out,
                            float c0_out) {
  return erf_qdq(stream, in, in_type, out, out_type, 16, n, c1, c0, c1_out, c0_out);
}

}  // extern "C"
