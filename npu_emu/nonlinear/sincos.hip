/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Sine / cosine kernels: polynomial and table variants on bf16 / fp16, fp8 and quantized
 * integers. One thread per element.
 */
#include "nonlinear/sincos.h"

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#include "common/check.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/nlf.h"
#include "common/quant.h"

namespace {

using namespace npu_emu;

__device__ inline float rb(float x) { return round_float(x, NPU_BF16, kHalfEven); }
__device__ inline float rh(float x) { return round_float(x, NPU_F16, kHalfEven); }

// The sin entry maps z % 4 to sin, cos, -sin, -cos; the cos entry to cos, -sin, -cos, sin.
// Negation is exact, +0 -> -0.
__device__ inline float quadrant(int64_t z, float sin_v, float cos_v, bool is_cos) {
  const bool odd = z & 1, neg = z & 2;
  if (!is_cos) {
    const float a = odd ? cos_v : sin_v;
    return neg ? mul_q(-1.0f, a) : a;
  }
  if (odd) return neg ? sin_v : mul_q(-1.0f, sin_v);
  return neg ? mul_q(-1.0f, cos_v) : cos_v;
}

// Polynomial sin / cos of a bf16 value; constants rounded to bf16, half to even.
__device__ float sincos_bf16(float ux, bool is_cos) {
  const float tp = rb(float(0x1.45f306dc9c883p-1));
  const float pi2 = rb(float(0x1.921fb54400000p0));
  const float s0 = rb(float(-0x1.5555555555555p-3)), s1 = rb(float(0x1.1111111110bb3p-7));
  const float s2 = rb(float(-0x1.a01a019e83e5cp-13)), s3 = rb(float(0x1.71de3796cde01p-19));
  const float c0 = rb(float(0x1.5555555555555p-5)), c1 = rb(float(-0x1.6c16c16c16967p-10));
  const float c2 = rb(float(0x1.A01A019F4EC91p-16)), c3 = rb(float(-0x1.27E4FA17F667Bp-22));
  ux = ftz(ux);
  // z = floor(bf16(fp32(ux * 2/pi + 0.5)))
  const int64_t z = float_to_int(rb(fma_q(ux, tp, 0.5f)), NPU_I16, 0, kFloor);
  const float zb = rb(float(z));
  const float xa = msc_q(pi2, zb, ux);
  const float x = rb(xa);
  const float r = rb(mul_q(x, x));
  const float r2 = rb(mul_q(r, r));
  const float t = fma_q(-0.5f, r, 1.0f);
  const float t1s = fma_q(s1, r, s0), t2s = fma_q(s3, r, s2);
  const float t1c = fma_q(c1, r, c0), t2c = fma_q(c3, r, c2);
  const float outs = fma_q(r2, rb(t2s), t1s);
  const float outc = fma_q(r2, rb(t2c), t1c);
  const float x3 = rb(mul_q(r, x));
  const float sin_v = rb(fma_q(x3, rb(outs), xa));
  const float cos_v = rb(fma_q(r2, rb(outc), t));
  return quadrant(z, sin_v, cos_v, is_cos);
}

// Polynomial sin / cos of an fp16 value; every MAC rounded to fp32, then to fp16.
__device__ float sincos_fp16(float ux, bool is_cos) {
  const float inv_pi_2 = rh(0.6367f), hi = rh(1.57f), lo = rh(0.0004838f);
  const float sc3 = rh(-0.1666f), sc5 = rh(0.00833f), sc7 = rh(-0.0001984f);
  const float cc2 = rh(-0.5f), cc4 = rh(0.04166f), cc6 = rh(-0.001389f);
  const int64_t z = float_to_int(mul_q(ux, inv_pi_2), NPU_I16, 0, kHalfEven);
  const float kh = rh(float(z));
  float ra = msc_q(hi, kh, ux);
  ra = msc_q(lo, kh, ra);
  const float r = rh(ra);
  const float r2 = rh(mul_q(r, r));
  float si = rh(fma_q(sc7, r2, sc5));
  si = rh(fma_q(si, r2, sc3));
  const float sp = rh(fma_q(si, r2, 1.0f));
  const float sin_r = rh(mul_q(r, sp));
  float ci = rh(fma_q(cc6, r2, cc4));
  ci = rh(fma_q(ci, r2, cc2));
  const float cos_r = rh(fma_q(ci, r2, 1.0f));
  return quadrant(z, sin_r, cos_r, is_cos);
}

__device__ inline float sincos_native(float ux, npu_dtype ht, bool is_cos) {
  return ht == NPU_F16 ? sincos_fp16(ux, is_cos) : sincos_bf16(ux, is_cos);
}

__device__ float lut_core(float x, const float* lut) {
  const float ix = ftz(mul_q(x, 0.31830987f));
  // ix * 256 is exact (or overflows to inf, which saturates too)
  const int idx = int(float_to_int(mul_q(ix, 256.0f), NPU_I32, 0, kFloor) & 511);
  const float fl = int_to_float(float_to_int(ix, NPU_I32, 0, kFloor), 0, kFloor);
  return lut_interp(fma_q(-1.0f, fl, ix), lut, idx, NPU_F32, kHalfEven);
}

__global__ void sincos16_kernel(int64_t n, const uint16_t* in, uint16_t* out, npu_dtype ht,
                                bool is_cos) {
  NPU_TID;
  const float x = decode_float(in[i], ht);
  out[i] = encode_float(sincos_native(x, ht, is_cos), ht, kHalfEven);
}

__global__ void sincos_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, npu_dtype ht,
                                  bool is_cos, float ifm_scale, float rc) {
  NPU_TID;
  const float x = fp8_in(in, i, n, ifm_scale, ht, kHalfEven);
  out[i] = fp8_out(sincos_native(x, ht, is_cos), rc, kHalfEven);
}

__global__ void lut16_kernel(int64_t n, const uint16_t* in, const float* lut, uint16_t* out,
                             npu_dtype ht) {
  NPU_TID;
  const float x = decode_float(in[i], ht);
  out[i] = encode_float(lut_core(x, lut), ht, kHalfEven);
}

__global__ void lut_fp8_kernel(int64_t n, const uint8_t* in, const float* lut, uint8_t* out,
                               npu_dtype ht, float ifm_scale, float rc) {
  NPU_TID;
  const float x = fp8_in(in, i, n, ifm_scale, ht, kHalfAway);
  out[i] = fp8_out(round_float(lut_core(x, lut), ht, kHalfAway), rc, kHalfAway);
}

__global__ void qdq_sincos_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                                  npu_dtype ot, npu_dtype ht, bool is_cos, float izp, float is,
                                  float ozp, float orc) {
  NPU_TID;
  const float d = dequant(load_int(in, it, i), is, dq_offset(izp, is));
  const float y = sincos_native(round_float(d, ht, kHalfEven), ht, is_cos);
  store_int(out, ot, i, quant(y, orc, ozp, ot));
}

__global__ void qdq_lut_kernel(int64_t n, const void* in, npu_dtype it, const float* lut,
                               void* out, npu_dtype ot, float izp, float is, float ozp,
                               float orc) {
  NPU_TID;
  const float d = dequant(load_int(in, it, i), is, dq_offset(izp, is));
  store_int(out, ot, i, quant(lut_core(d, lut), orc, ozp, ot));
}

// Every buffer is required, also when there is nothing to process.
int sincos16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n, bool is_cos) {
  if (!is_half_float(in_type) || n < 0 || any_null(in, out)) return -1;
  return launch(stream, n, sincos16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), in_type, is_cos);
}

int sincos_fp8(void* stream, const void* in, void* out, int64_t n, npu_dtype compute_type,
               float ifm_scale, float recip, bool is_cos) {
  if (!is_half_float(compute_type) || n < 0 || any_null(in, out)) return -1;
  return launch(stream, n, sincos_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), compute_type, is_cos, ifm_scale,
                scalar_cast(recip, compute_type, kHalfEven));
}

int lut16(void* stream, const void* in, npu_dtype in_type, const void* lut, void* out, int64_t n) {
  if (!is_half_float(in_type) || n < 0 || any_null(in, out, lut)) return -1;
  return launch(stream, n, lut16_kernel, static_cast<const uint16_t*>(in),
                static_cast<const float*>(lut), static_cast<uint16_t*>(out), in_type);
}

int lut_fp8(void* stream, const void* in, const void* lut, void* out, int64_t n,
            npu_dtype compute_type, float ifm_scale, float recip) {
  if (!is_half_float(compute_type) || n < 0 || any_null(in, out, lut)) return -1;
  return launch(stream, n, lut_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<const float*>(lut), static_cast<uint8_t*>(out), compute_type, ifm_scale,
                scalar_cast(recip, compute_type, kHalfAway));
}

int qdq_sincos(void* stream, const void* in, npu_dtype in_type, void* out, npu_dtype out_type,
               int bits, int64_t n, npu_dtype compute_type, float dq_zp, float dq_sc, float q_zp,
               float q_sc, bool is_cos) {
  if (!qdq_args_ok(in, in_type, out, out_type, bits, n) || !is_half_float(compute_type)) return -1;
  return launch(stream, n, qdq_sincos_kernel, in, in_type, out, out_type, compute_type, is_cos,
                dq_zp, dq_sc, q_zp, q_sc);
}

int qdq_lut(void* stream, const void* in, npu_dtype in_type, const void* lut, void* out,
            npu_dtype out_type, int bits, int64_t n, float dq_zp, float dq_sc, float q_zp,
            float q_sc) {
  if (!qdq_args_ok(in, in_type, out, out_type, bits, n) || !lut) return -1;
  return launch(stream, n, qdq_lut_kernel, in, in_type, static_cast<const float*>(lut), out,
                out_type, dq_zp, dq_sc, q_zp, q_sc);
}

}  // namespace

extern "C" {

int npu_sin_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n) {
  return sincos16(stream, in, in_type, out, n, false);
}

int npu_cos_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n) {
  return sincos16(stream, in, in_type, out, n, true);
}

int npu_elementwise_sin_fp8(void* stream, const void* in, void* out, int64_t n,
                            npu_dtype compute_type, float ifm_scale, float ofm_scale_recip) {
  return sincos_fp8(stream, in, out, n, compute_type, ifm_scale, ofm_scale_recip, false);
}

int npu_elementwise_cos_fp8(void* stream, const void* in, void* out, int64_t n,
                            npu_dtype compute_type, float ifm_scale, float ofm_scale_recip) {
  return sincos_fp8(stream, in, out, n, compute_type, ifm_scale, ofm_scale_recip, true);
}

int npu_elementwise_sin_lut_bf16(void* stream, const void* in, npu_dtype in_type,
                                 const void* lut, void* out, int64_t n) {
  return lut16(stream, in, in_type, lut, out, n);
}

int npu_elementwise_cos_lut_bf16(void* stream, const void* in, npu_dtype in_type,
                                 const void* lut, void* out, int64_t n) {
  return lut16(stream, in, in_type, lut, out, n);
}

int npu_elementwise_sin_lut_fp8(void* stream, const void* in, const void* lut, void* out,
                                int64_t n, npu_dtype compute_type, float ifm_scale,
                                float ofm_scale_recip) {
  return lut_fp8(stream, in, lut, out, n, compute_type, ifm_scale, ofm_scale_recip);
}

int npu_elementwise_cos_lut_fp8(void* stream, const void* in, const void* lut, void* out,
                                int64_t n, npu_dtype compute_type, float ifm_scale,
                                float ofm_scale_recip) {
  return lut_fp8(stream, in, lut, out, n, compute_type, ifm_scale, ofm_scale_recip);
}

int npu_elementwise_sin_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                           float dq_sc, float q_zp, float q_sc) {
  return qdq_sincos(stream, in, in_type, out, out_type, 8, n, compute_type, dq_zp, dq_sc, q_zp,
                    q_sc, false);
}

int npu_elementwise_sin_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                            float dq_sc, float q_zp, float q_sc) {
  return qdq_sincos(stream, in, in_type, out, out_type, 16, n, compute_type, dq_zp, dq_sc, q_zp,
                    q_sc, false);
}

int npu_elementwise_cos_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                           float dq_sc, float q_zp, float q_sc) {
  return qdq_sincos(stream, in, in_type, out, out_type, 8, n, compute_type, dq_zp, dq_sc, q_zp,
                    q_sc, true);
}

int npu_elementwise_cos_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                            float dq_sc, float q_zp, float q_sc) {
  return qdq_sincos(stream, in, in_type, out, out_type, 16, n, compute_type, dq_zp, dq_sc, q_zp,
                    q_sc, true);
}

int npu_elementwise_sin_lut_a8(void* stream, const void* in, npu_dtype in_type,
                               const void* lut, void* out, npu_dtype out_type, int64_t n,
                               float dq_zp, float dq_sc, float q_zp, float q_sc) {
  return qdq_lut(stream, in, in_type, lut, out, out_type, 8, n, dq_zp, dq_sc, q_zp, q_sc);
}

int npu_elementwise_sin_lut_a16(void* stream, const void* in, npu_dtype in_type,
                                const void* lut, void* out, npu_dtype out_type, int64_t n,
                                float dq_zp, float dq_sc, float q_zp, float q_sc) {
  return qdq_lut(stream, in, in_type, lut, out, out_type, 16, n, dq_zp, dq_sc, q_zp, q_sc);
}

int npu_elementwise_cos_lut_a8(void* stream, const void* in, npu_dtype in_type,
                               const void* lut, void* out, npu_dtype out_type, int64_t n,
                               float dq_zp, float dq_sc, float q_zp, float q_sc) {
  return qdq_lut(stream, in, in_type, lut, out, out_type, 8, n, dq_zp, dq_sc, q_zp, q_sc);
}

int npu_elementwise_cos_lut_a16(void* stream, const void* in, npu_dtype in_type,
                                const void* lut, void* out, npu_dtype out_type, int64_t n,
                                float dq_zp, float dq_sc, float q_zp, float q_sc) {
  return qdq_lut(stream, in, in_type, lut, out, out_type, 16, n, dq_zp, dq_sc, q_zp, q_sc);
}

}  // extern "C"
