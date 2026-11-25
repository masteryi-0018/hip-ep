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

constexpr int kSigmoid = 0;
constexpr int kTanh = 1;

struct Qdq {
  npu_dtype in_t, out_t, ct;
  float dq_zp, dq_sc, q_zp, q_sc;
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
    const float x = acc_to_bf16(decode_bfp16_identity(in, i & ~int64_t(7), n, i, k));
    const float z = silu ? hw_silu_f32(x) : hw_sigtanh_f32(x, act == kTanh, kSigTanhK0, k1, k2);
    v[k] = acc_to_bf16(z);
  }
  encode_bfp16(v, kHalfAway, out + i * 9);
}

__global__ void sigtanh_fp8_kernel(int64_t n, const uint8_t* x, uint8_t* y, int act, float k1,
                                   float k2, float recip) {
  NPU_TID;
  const float v = fp8_decode_identity(x, i, n);
  const float c = hw_sigtanh_f32(v, act == kTanh, kSigTanhK0, k1, k2);
  y[i] = fp8_out(c, recip, kHalfAway, 0);
}

__global__ void silu_fp8_kernel(int64_t n, const uint8_t* x, uint8_t* y, npu_dtype ct,
                                float ifm_scale, float recip) {
  NPU_TID;
  // the scale is cast half to even for bf16, half away for f16
  const float recip_ct = round_float(recip, ct, ct == NPU_BF16 ? kHalfEven : kHalfAway, kFtzIn);
  const float v = fp8_in(x, i, n, ifm_scale, ct, kHalfAway, kFtzIn);
  const float s = round_float(hw_silu_f32(v), ct, kHalfAway, kFtzIn);
  y[i] = fp8_out(s, recip_ct, kHalfAway);
}

__global__ void qdq_kernel(int64_t n, const void* x, void* y, Qdq p) {
  NPU_TID;
  const float d = round_float(dequant(load_int(x, p.in_t, i), p.dq_sc, dq_offset(p.dq_zp, p.dq_sc)),
                              p.ct, kHalfAway, kFtzIn);
  const float s = p.silu ? round_float(hw_silu_f32(d), p.ct, kHalfAway, kFtzIn)
                         : acc_to_bf16(hw_sigtanh_f32(d, p.act == kTanh, p.k0, p.k1, p.k2));
  store_int(y, p.out_t, i, quant(s, p.q_sc, p.q_zp, p.out_t, kHalfAway));
}

bool act_ok(int act) { return act == kSigmoid || act == kTanh; }

int qdq(void* stream, const void* in, npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
        int bits, int silu, int act, npu_dtype compute_type, float dq_zp, float dq_sc, float q_zp,
        float q_sc, float k0, float k1, float k2) {
  if (!qdq_args_ok(in, in_type, out, out_type, bits, n) || !is_half_float(compute_type) ||
      (!silu && !act_ok(act)))
    return -1;
  Qdq p{in_type, out_type, compute_type, dq_zp, dq_sc, q_zp, q_sc, silu, act, k0, k1, k2};
  return launch(stream, n, qdq_kernel, in, out, p);
}

// ---- fp32 QDQ SiLU ----

constexpr float kB0 = 895.0f;
// mode 0: e * h2(e) ~ log2(1 + e); P3(f) ~ 2^f on [0, 1)
constexpr float kH2[2] = {1.3720913054337833f, -0.3997208323665843f};
constexpr float kE3[4] = {0.9999278783798218f, 0.6957767605781555f, 0.22623352706432343f,
                          0.07790721952915192f};
// mode 1: h coefficients, gain, and log2(1 + x) = x * LP(x)
constexpr float kG3[3] = {1.4226349664583249f, -0.6029645309400076f, 0.18733279449559237f};
constexpr float kG3g = 0.9995263666916724f;
constexpr float kLP[7] = {1.4426963329315186f, -0.7213282585144043f, 0.4806240499019623f,
                          -0.3615472912788391f, 0.2982918620109558f, -0.24276652932167053f,
                          0.11998527497053146f};

struct SiluF32 {
  npu_dtype in_t, out_t;
  int mode;
  float zp_out;
  float s_k, b_k, ik;                // mode 0
  int32_t zi;                        // mode 1
  float C, K, A, cp0, cp1, cp2;      // mode 1
};

// Per-call constants, derived as the kernel does from the four coefficients.
SiluF32 silu_f32_params(npu_dtype in_type, npu_dtype out_type, int mode, float dq_zp, float s_in,
                        float q_zp, float inv) {
  SiluF32 p{};
  p.in_t = in_type;
  p.out_t = out_type;
  p.mode = mode;
  p.zp_out = q_zp;
  p.s_k = mul_q(s_in, inv);
  p.b_k = mul_q(dq_zp, neg_f32(p.s_k));
  p.ik = div_q(kLog2e, inv);
  p.zi = scalar_cast_i32(dq_zp);
  p.C = p.s_k;
  const float Cg = mul_q(p.C, kG3g);
  p.K = mul_q(s_in, -kLog2e);
  // A = 895 + log2(Cg): exponent from the bits, mantissa reduced to [0.75, 1.5)
  const uint32_t cb = f32_bits(Cg);
  int32_t E = int32_t((cb >> 23) & 0xFF) - 127;
  uint32_t mb = (cb & 0x7FFFFF) | 0x3F800000;
  if (cb & 0x400000) {
    mb = (cb & 0x7FFFFF) | 0x3F000000;
    E += 1;
  }
  const float vx = add_q(bits_f32(mb), -1.0f);
  float lp = kLP[6];
  for (int k = 5; k >= 0; --k) lp = fma_q(vx, lp, kLP[k]);
  p.A = add_q(float(895 + E), mul_q(lp, vx));
  const float r = div_q(1.0f, Cg);
  const float r2 = mul_q(r, r);
  p.cp0 = mul_q(kG3[0], r);
  p.cp1 = mul_q(kG3[1], r2);
  p.cp2 = mul_q(kG3[2], mul_q(r2, r));
  return p;
}

__global__ void silu_f32_kernel(int64_t n, const void* x, void* y, SiluF32 p) {
  NPU_TID;
  const int64_t q = load_int(x, p.in_t, i);
  float o;
  if (p.mode == 0) {
    const float xk = mac_exact(q, p.s_k, p.b_k);
    const float u = fma_q(p.ik, neg_f32(fabsf(xk)), kB0);
    const float un = bits_f32(f32_bits(u) & 0xFFFFC000u);
    const float f = sub_q(u, un);
    float pf = fma_q(f, kE3[3], kE3[2]);
    pf = fma_q(f, pf, kE3[1]);
    pf = fma_q(f, pf, kE3[0]);
    const float e = mul_q(hw_exp2_bf20(un), pf);
    const float h = fma_q(kH2[1], e, kH2[0]);
    const float s0 = hw_exp2_bf20(fma_q(neg_f32(e), h, kB0));
    const float er = fma_q(neg_f32(add_q(e, 1.0f)), s0, 1.0f);
    const float s1 = fma_q(s0, er, s0);
    o = fma_q(vmax(xk, mul_q(xk, e)), s1, p.zp_out);
  } else {
    const int64_t qi = q - p.zi;
    const int64_t aq = qi < 0 ? -qi : qi;
    const float arg = mac_exact(aq, p.K, p.A);
    const float o1 = mac_exact(qi > 0 ? qi : 0, p.C, p.zp_out);
    const float ep = hw_exp2_bf20(arg);
    const float h = fma_q(fma_q(p.cp2, ep, p.cp1), ep, p.cp0);
    const float sg = hw_exp2_bf20(fma_q(neg_f32(ep), h, arg));
    o = mac_exact(-aq, sg, o1);
  }
  store_int(y, p.out_t, i, float_to_int(o, p.out_t, 0, kHalfAway));
}

}  // namespace

extern "C" {

int npu_sigtanh_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n,
                     int act) {
  if (n < 0 || !is_half_float(in_type) || !act_ok(act) || any_null(in, out)) return -1;
  return launch(stream, n, sigtanh16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), in_type, act);
}

int npu_silu_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n) {
  if (n < 0 || !is_half_float(in_type) || any_null(in, out)) return -1;
  return launch(stream, n, silu16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), in_type);
}

// The bfp16 kernels run whole sub-volumes of 512 elements.
int npu_sigtanh_bfp16(void* stream, const void* in, void* out, int64_t n, int act) {
  if (n < 0 || n % 512 || !act_ok(act) || any_null(in, out)) return -1;
  return launch(stream, n / 8, bfp16_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), 0, act);
}

int npu_silu_bfp16(void* stream, const void* in, void* out, int64_t n) {
  if (n < 0 || n % 512 || any_null(in, out)) return -1;
  return launch(stream, n / 8, bfp16_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), 1, 0);
}

int npu_elementwise_sigtanh_fp8(void* stream, const void* in, void* out, int64_t n, int act,
                                float poly1_coeff, float poly2_coeff, float ofm_coeff) {
  if (n < 0 || !act_ok(act) || any_null(in, out)) return -1;
  return launch(stream, n, sigtanh_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), act, poly1_coeff, poly2_coeff, ofm_coeff);
}

int npu_elementwise_silu_fp8(void* stream, const void* in, void* out, int64_t n,
                             npu_dtype compute_type, float ifm_scale, float ofm_scale_recip) {
  if (n < 0 || !is_half_float(compute_type) || any_null(in, out)) return -1;
  return launch(stream, n, silu_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), compute_type, ifm_scale, ofm_scale_recip);
}

int npu_qdq_sigtanh_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                       npu_dtype out_type, int64_t n, int act, npu_dtype compute_type,
                       float dq_zp, float dq_sc, float q_zp, float q_sc, float k0, float k1,
                       float k2) {
  return qdq(stream, in, in_type, out, out_type, n, 8, 0, act, compute_type, dq_zp, dq_sc,
             q_zp, q_sc, k0, k1, k2);
}

int npu_qdq_sigtanh_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                        npu_dtype out_type, int64_t n, int act, npu_dtype compute_type,
                        float dq_zp, float dq_sc, float q_zp, float q_sc, float k0, float k1,
                        float k2) {
  return qdq(stream, in, in_type, out, out_type, n, 16, 0, act, compute_type, dq_zp, dq_sc,
             q_zp, q_sc, k0, k1, k2);
}

int npu_qdq_silu_exp2_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                         npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                         float dq_sc, float q_zp, float q_sc) {
  return qdq(stream, in, in_type, out, out_type, n, 8, 1, 0, compute_type, dq_zp, dq_sc, q_zp,
             q_sc, 0.0f, 0.0f, 0.0f);
}

int npu_qdq_silu_exp2_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                          npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                          float dq_sc, float q_zp, float q_sc) {
  return qdq(stream, in, in_type, out, out_type, n, 16, 1, 0, compute_type, dq_zp, dq_sc, q_zp,
             q_sc, 0.0f, 0.0f, 0.0f);
}

int npu_qdq_silu_fp32_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                          npu_dtype out_type, int64_t n, int mode, float dq_zp, float dq_sc,
                          float q_zp, float q_sc) {
  if (!qdq_args_ok(in, in_type, out, out_type, 16, n) || (mode != 0 && mode != 1)) return -1;
  return launch(stream, n, silu_f32_kernel, in, out,
                silu_f32_params(in_type, out_type, mode, dq_zp, dq_sc, q_zp, q_sc));
}

}  // extern "C"
