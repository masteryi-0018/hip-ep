/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * SigTanh (sigmoid / tanh) and SiLU kernels. One thread per element (per bfp16 block of 8
 * for the bfp16 entries).
 */
#include "nonlinear/silu.h"

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#include "common/act.h"
#include "common/block_fp.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/nlf.h"

namespace {

using namespace npu_emu;

constexpr int kSigmoid = 0;
constexpr int kTanh = 1;

struct Qdq {
  npu_dtype in_t, out_t, ct;
  float dq_zp, dq_scale, q_zp, q_scale;
  int silu, act;
  float k0, k1, k2;
};

__global__ void sigtanh16_kernel(int64_t n, const uint16_t* x, uint16_t* y, npu_dtype t,
                                 int act) {
  NPU_TID;
  const bool th = act == kTanh;
  const float z = hw_sigtanh_f32(decode_float(x[i], t), th, kSigTanhK0,
                                 th ? kTanhK1 : kSigmoidK1, th ? kTanhK2 : kSigmoidK2);
  y[i] = encode_float(z, t, kHalfAway, kFtzIn);
}

__global__ void silu16_kernel(int64_t n, const uint16_t* x, uint16_t* y, npu_dtype t) {
  NPU_TID;
  y[i] = encode_float(hw_silu_f32(decode_float(x[i], t)), t, kHalfAway, kFtzIn);
}

// One thread per bfp16 block.
__global__ void bfp16_kernel(int64_t n, const uint8_t* in, uint8_t* out, int silu, int act) {
  NPU_TID;
  const float k1 = act == kTanh ? kTanhK1 : kSigmoidK1;
  const float k2 = act == kTanh ? kTanhK2 : kSigmoidK2;
  float v[8];
  for (int k = 0; k < 8; ++k) {
    // decode, then bf16 (exact but for the subnormal flush)
    const float x = round_float(decode_bfp16_identity(in, i & ~int64_t(7), n, i, k), NPU_BF16,
                                kHalfAway, kFtzIn);
    const float z = silu ? hw_silu_f32(x) : hw_sigtanh_f32(x, act == kTanh, kSigTanhK0, k1, k2);
    v[k] = round_float(z, NPU_BF16, kHalfAway, kFtzIn);
  }
  encode_bfp16(v, kHalfAway, out + i * 9);
}

__global__ void sigtanh_fp8_kernel(int64_t n, const uint8_t* x, uint8_t* y, int act, float k1,
                                   float k2, float recip) {
  NPU_TID;
  const float v = fp8_decode_identity(x, i, n);
  const float c = hw_sigtanh_f32(v, act == kTanh, kSigTanhK0, k1, k2);
  y[i] = f32_to_fp8(mul_q(c, recip), kHalfAway);
}

__global__ void silu_fp8_kernel(int64_t n, const uint8_t* x, uint8_t* y, npu_dtype ct,
                                float ifm_scale, float recip) {
  NPU_TID;
  // the scale is cast half to even for bf16, half away for f16
  const float recip_ct = round_float(recip, ct, ct == NPU_BF16 ? kHalfEven : kHalfAway, kFtzIn);
  const float d = mul_q(fp8_decode_identity(x, i, n), ifm_scale);
  const float v = round_float(d, ct, kHalfAway, kFtzIn);
  const float s = round_float(hw_silu_f32(v), ct, kHalfAway, kFtzIn);
  y[i] = f32_to_fp8(mul_q(s, recip_ct), kHalfAway, kSat);
}

__global__ void qdq_kernel(int64_t n, const void* x, void* y, Qdq p) {
  NPU_TID;
  // zp' = fp32(-zp * s), then one MAC with the exact integer
  const float zps = mul_q(-p.dq_zp, p.dq_scale);
  const float d =
      round_float(mac_exact(load_int(x, p.in_t, i), p.dq_scale, zps), p.ct, kHalfAway, kFtzIn);
  const float s = p.silu ? round_float(hw_silu_f32(d), p.ct, kHalfAway, kFtzIn)
                         : round_float(hw_sigtanh_f32(d, p.act == kTanh, p.k0, p.k1, p.k2),
                                       NPU_BF16, kHalfAway, kFtzIn);
  store_int(y, p.out_t, i, float_to_int(fma_q(s, p.q_scale, p.q_zp), p.out_t, 0, kHalfAway));
}

bool act_ok(int act) { return act == kSigmoid || act == kTanh; }
bool ints_ok(npu_dtype a, npu_dtype b, bool a16) {
  auto ok = [&](npu_dtype t) {
    return a16 ? (t == NPU_I16 || t == NPU_U16) : (t == NPU_I8 || t == NPU_U8);
  };
  return ok(a) && ok(b);
}

int qdq(void* stream, const void* in, npu_dtype in_t, void* out, npu_dtype out_t, int64_t n,
        bool a16, int silu, int act, npu_dtype ct, float dq_zp, float dq_scale, float q_zp,
        float q_scale, float k0, float k1, float k2) {
  if (n < 0 || !is_half_float(ct) || !ints_ok(in_t, out_t, a16) || (!silu && !act_ok(act)))
    return -1;
  Qdq p{in_t, out_t, ct, dq_zp, dq_scale, q_zp, q_scale, silu, act, k0, k1, k2};
  return launch(stream, n, qdq_kernel, in, out, p);
}

}  // namespace

extern "C" {

int npu_sigtanh_bf16(void* stream, const void* in, void* out, npu_dtype dtype, int64_t n,
                     int act) {
  if (n < 0 || !is_half_float(dtype) || !act_ok(act)) return -1;
  return launch(stream, n, sigtanh16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), dtype, act);
}

int npu_silu_bf16(void* stream, const void* in, void* out, npu_dtype dtype, int64_t n) {
  if (n < 0 || !is_half_float(dtype)) return -1;
  return launch(stream, n, silu16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), dtype);
}

int npu_sigtanh_bfp16(void* stream, const void* in, void* out, int64_t n, int act) {
  if (n < 0 || n % 8 || !act_ok(act)) return -1;
  return launch(stream, n / 8, bfp16_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), 0, act);
}

int npu_silu_bfp16(void* stream, const void* in, void* out, int64_t n) {
  if (n < 0 || n % 8) return -1;
  return launch(stream, n / 8, bfp16_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), 1, 0);
}

int npu_elementwise_sigtanh_fp8(void* stream, const void* in, void* out, int64_t n, int act,
                                float k1, float k2, float ofm_recip) {
  if (n < 0 || !act_ok(act)) return -1;
  return launch(stream, n, sigtanh_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), act, k1, k2, ofm_recip);
}

int npu_elementwise_silu_fp8(void* stream, const void* in, void* out, int64_t n, npu_dtype ct,
                             float ifm_scale, float ofm_recip) {
  if (n < 0 || !is_half_float(ct)) return -1;
  return launch(stream, n, silu_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), ct, ifm_scale, ofm_recip);
}

int npu_qdq_sigtanh_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                       npu_dtype out_type, int64_t n, int act, npu_dtype ct, float dq_zp,
                       float dq_scale, float q_zp, float q_scale, float k0, float k1, float k2) {
  return qdq(stream, in, in_type, out, out_type, n, false, 0, act, ct, dq_zp, dq_scale, q_zp,
             q_scale, k0, k1, k2);
}

int npu_qdq_sigtanh_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                        npu_dtype out_type, int64_t n, int act, npu_dtype ct, float dq_zp,
                        float dq_scale, float q_zp, float q_scale, float k0, float k1,
                        float k2) {
  return qdq(stream, in, in_type, out, out_type, n, true, 0, act, ct, dq_zp, dq_scale, q_zp,
             q_scale, k0, k1, k2);
}

int npu_qdq_silu_exp2_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                         npu_dtype out_type, int64_t n, npu_dtype ct, float dq_zp,
                         float dq_scale, float q_zp, float q_scale) {
  return qdq(stream, in, in_type, out, out_type, n, false, 1, 0, ct, dq_zp, dq_scale, q_zp,
             q_scale, 0.0f, 0.0f, 0.0f);
}

int npu_qdq_silu_exp2_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                          npu_dtype out_type, int64_t n, npu_dtype ct, float dq_zp,
                          float dq_scale, float q_zp, float q_scale) {
  return qdq(stream, in, in_type, out, out_type, n, true, 1, 0, ct, dq_zp, dq_scale, q_zp,
             q_scale, 0.0f, 0.0f, 0.0f);
}

}  // extern "C"
