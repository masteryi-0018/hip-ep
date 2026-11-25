/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * exp / elu / hardsigmoid / sqrt on bf16 / fp16, fp8 and quantized integers.
 * One thread per element.
 */
#include "nonlinear/expsqrt.h"

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#include "common/bits.h"
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

// Hardware exp2 of fp32(log2e * x + bias).
__device__ inline float exp_hw(float x) { return hw_exp2_bf20(fma_q(kLog2e, x, kExp2Bias)); }

__global__ void exp_bf16_kernel(int64_t n, const uint16_t* in, uint16_t* out, npu_dtype ht) {
  NPU_TID;
  out[i] = encode_float(exp_hw(decode_float(in[i], ht)), ht, kHalfEven, kFtzIn);
}

__global__ void elu_bf16_kernel(int64_t n, const uint16_t* in, uint16_t* out, npu_dtype ht,
                                float alpha) {
  NPU_TID;
  const float x = decode_float(in[i], ht);
  if (x >= 0.0f) {  // +-0 included
    out[i] = encode_float(x, ht, kHalfEven, kFtzIn);
    return;
  }
  const float e = exp_hw(x);
  out[i] = encode_float(fma_q(e, alpha, neg_f32(alpha)), ht, kHalfEven, kFtzIn);
}

__global__ void hardsigmoid_bf16_kernel(int64_t n, const uint16_t* in, uint16_t* out, npu_dtype ht,
                                        float alpha, float beta) {
  NPU_TID;
  const float x = decode_float(in[i], ht);
  out[i] = encode_float(vclamp(fma_q(x, alpha, beta), 0.0f, 1.0f), ht, kHalfEven, kFtzIn);
}

__global__ void sqrt_bf16_kernel(int64_t n, const uint16_t* in, uint16_t* out, npu_dtype ht) {
  NPU_TID;
  out[i] = hw_sqrt16(in[i], ht);
}

__global__ void exp_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, float in_coeff,
                               float bias) {
  NPU_TID;
  const float x = fp8_decode_identity(in, i, n);
  // NaN x -> +qNaN -> 0x7f
  out[i] = f32_to_fp8(hw_exp2_bf20(fma_q(x, in_coeff, bias)), kHalfAway);
}

__global__ void elu_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, float in_scale,
                               float pos_scale, float neg_scale) {
  NPU_TID;
  const float x = fp8_decode_identity(in, i, n);
  const float r = mul_q(x, in_scale);
  float v;
  if (r >= 0.0f) {
    v = mul_q(x, pos_scale);
  } else {  // also NaN
    const float e = exp_hw(r);
    v = fma_q(e, neg_scale, neg_f32(neg_scale));
  }
  out[i] = f32_to_fp8(v, kHalfAway);
}

__global__ void hardsigmoid_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, float slope,
                                       float offset, float upper) {
  NPU_TID;
  const float x = fp8_decode_identity(in, i, n);
  out[i] = f32_to_fp8(vclamp(fma_q(x, slope, offset), 0.0f, upper), kHalfAway);
}

__global__ void sqrt_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, float scale,
                                const uint32_t* invsqrt_exc) {
  NPU_TID;
  const float x = fp8_decode_identity(in, i, n);  // exact in fp16
  float y;
  if (!(x >= 0.0f)) {
    y = default_nan();
  } else if (x == 0.0f) {
    y = x;
  } else {
    // 16-bit invsqrt; the product is rounded once to fp16
    const float inv = f16_to_f32(hw_invsqrt_f16(f32_to_f16(x, kHalfEven), invsqrt_exc));
    y = round_float(mul_q(inv, x), NPU_F16, kHalfEven);
  }
  out[i] = fp8_out(y, scale, kHalfEven, 0);
}

__global__ void exp_qdq_kernel(int64_t n, const void* in, npu_dtype it, void* out, npu_dtype ot,
                               float c1, float c0, float s_bf16, float z_bf16) {
  NPU_TID;
  const int64_t x = load_int(in, it, i);
  const float e = exp2_bf16(vmin(mac_exact(x, c1, c0), 127.0f));
  // bf16 * bf16 is exact in fp32; the MAC rounds once
  store_int(out, ot, i, quant(e, s_bf16, z_bf16, ot));
}

__global__ void elu_qdq_kernel(int64_t n, const void* in, npu_dtype it, void* out, npu_dtype ot,
                               float c1, float c0, float ifm_scale, float neg_bias, float a_div,
                               float neg_const) {
  NPU_TID;
  const int64_t x = load_int(in, it, i);
  const float d = dequant(x, ifm_scale, neg_bias);
  const float v = d > 0.0f ? mac_exact(x, c1, c0) : fma_q(exp_hw(d), a_div, neg_const);
  store_int(out, ot, i, requant_tail(v, ot));
}

__global__ void hardsigmoid_qdq_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                                       npu_dtype ot, float alpha_f, float beta_f, float lo,
                                       float hi) {
  NPU_TID;
  const float v = mac_exact(load_int(in, it, i), alpha_f, beta_f);
  store_int(out, ot, i, requant_tail(vclamp(v, lo, hi), ot));
}

__global__ void sqrt_qdq_kernel(int64_t n, const void* in, npu_dtype it, void* out, npu_dtype ot,
                                float dq_scale, float dq_off, float q_scale, float q_zp) {
  NPU_TID;
  const float d = dequant(load_int(in, it, i), dq_scale, dq_off);
  const float s = f16_to_f32(hw_sqrt16(f32_to_f16(d, kHalfEven), NPU_F16));
  // a negative input gives a NaN sqrt, which converts to 0
  store_int(out, ot, i, quant(s, q_scale, q_zp, ot));
}

bool f16_ok(const void* in, npu_dtype in_type, const void* out, int64_t n) {
  return is_half_float(in_type) && n >= 0 && !any_null(in, out);
}

bool fp8_ok(const void* in, const void* out, int64_t n) { return n >= 0 && !any_null(in, out); }

int exp_qdq(void* stream, const void* in, npu_dtype in_type, void* out, npu_dtype out_type,
            int bits, int64_t n, float c1, float c0, float c1_out, float c0_out) {
  if (!qdq_args_ok(in, in_type, out, out_type, bits, n)) return -1;
  // output coefficients to bf16, half to even
  return launch(stream, n, exp_qdq_kernel, in, in_type, out, out_type, c1, c0,
                scalar_cast(c1_out, NPU_BF16, kHalfEven), scalar_cast(c0_out, NPU_BF16, kHalfEven));
}

int elu_qdq(void* stream, const void* in, npu_dtype in_type, void* out, npu_dtype out_type,
            int bits, int64_t n, float c1, float c0, float ifm_scale, float neg_bias, float a_div,
            float neg_const) {
  if (!qdq_args_ok(in, in_type, out, out_type, bits, n)) return -1;
  return launch(stream, n, elu_qdq_kernel, in, in_type, out, out_type, c1, c0, ifm_scale, neg_bias,
                a_div, neg_const);
}

int hardsigmoid_qdq(void* stream, const void* in, npu_dtype in_type, void* out, npu_dtype out_type,
                    int bits, int64_t n, float alpha_f, float beta_f, int32_t lower,
                    int32_t upper) {
  if (!qdq_args_ok(in, in_type, out, out_type, bits, n)) return -1;
  int64_t lo, hi;
  int_range(out_type, &lo, &hi);
  if (lower < lo || lower > hi || upper < lo || upper > hi) return -1;
  // the bounds are within the 16-bit output range, so exact
  return launch(stream, n, hardsigmoid_qdq_kernel, in, in_type, out, out_type, alpha_f, beta_f,
                float(lower), float(upper));
}

int sqrt_qdq(void* stream, const void* in, npu_dtype in_type, void* out, npu_dtype out_type,
             int bits, int64_t n, float dq_zp, float dq_sc, float q_zp, float q_sc) {
  if (!qdq_args_ok(in, in_type, out, out_type, bits, n)) return -1;
  return launch(stream, n, sqrt_qdq_kernel, in, in_type, out, out_type, dq_sc,
                dq_offset(dq_zp, dq_sc), q_sc, q_zp);
}

}  // namespace

extern "C" {

int npu_elementwise_exp_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                             int64_t n) {
  if (!f16_ok(in, in_type, out, n)) return -1;
  return launch(stream, n, exp_bf16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), in_type);
}

int npu_elementwise_elu_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                             int64_t n, float alpha) {
  if (!f16_ok(in, in_type, out, n)) return -1;
  return launch(stream, n, elu_bf16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), in_type, alpha);
}

int npu_elementwise_hardsigmoid_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                                     int64_t n, float alpha, float beta) {
  if (!f16_ok(in, in_type, out, n)) return -1;
  return launch(stream, n, hardsigmoid_bf16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), in_type, alpha, beta);
}

int npu_sqrt_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n) {
  if (!f16_ok(in, in_type, out, n)) return -1;
  return launch(stream, n, sqrt_bf16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), in_type);
}

int npu_elementwise_exp_fp8(void* stream, const void* in, void* out, int64_t n,
                            float in_coeff, float bias_coeff) {
  if (!fp8_ok(in, out, n)) return -1;
  return launch(stream, n, exp_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), in_coeff, bias_coeff);
}

int npu_elementwise_elu_fp8(void* stream, const void* in, void* out, int64_t n,
                            float ifm_scale, float positive_coeff, float negative_coeff) {
  if (!fp8_ok(in, out, n)) return -1;
  return launch(stream, n, elu_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), ifm_scale, positive_coeff, negative_coeff);
}

int npu_elementwise_hardsigmoid_fp8(void* stream, const void* in, void* out, int64_t n,
                                    float output_slope, float output_offset,
                                    float output_upper) {
  if (!fp8_ok(in, out, n)) return -1;
  return launch(stream, n, hardsigmoid_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), output_slope, output_offset, output_upper);
}

int npu_elementwise_sqrt_fp8(void* stream, const void* in, void* out, int64_t n,
                             float ifm_coeff) {
  if (!fp8_ok(in, out, n)) return -1;
  if (n == 0) return 0;
  const uint32_t* exc;
  if (int rc = npu_emu_nlf_table(NPU_EMU_NLF_INVSQRT_F32, &exc)) return rc;
  return launch(stream, n, sqrt_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), ifm_coeff, exc);
}

int npu_elementwise_exp_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, float c1, float c0, float c1_out,
                           float c0_out) {
  return exp_qdq(stream, in, in_type, out, out_type, 8, n, c1, c0, c1_out, c0_out);
}

int npu_elementwise_exp_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, float c1, float c0, float c1_out,
                            float c0_out) {
  return exp_qdq(stream, in, in_type, out, out_type, 16, n, c1, c0, c1_out, c0_out);
}

int npu_elementwise_elu_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, float c1, float c0, float ifm_scale,
                           float neg_bias, float alpha_div_ofm_scale, float neg_const) {
  return elu_qdq(stream, in, in_type, out, out_type, 8, n, c1, c0, ifm_scale, neg_bias,
                 alpha_div_ofm_scale, neg_const);
}

int npu_elementwise_elu_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t n, float c1, float c0,
                            float ifm_scale, float neg_bias, float alpha_div_ofm_scale,
                            float neg_const) {
  return elu_qdq(stream, in, in_type, out, out_type, 16, n, c1, c0, ifm_scale, neg_bias,
                 alpha_div_ofm_scale, neg_const);
}

int npu_hardsigmoid_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                       npu_dtype out_type, int64_t n, float alpha_f, float beta_f,
                       int32_t lower_bound_i, int32_t upper_bound_i) {
  return hardsigmoid_qdq(stream, in, in_type, out, out_type, 8, n, alpha_f, beta_f,
                         lower_bound_i, upper_bound_i);
}

int npu_hardsigmoid_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                        npu_dtype out_type, int64_t n, float alpha_f, float beta_f,
                        int32_t lower_bound_i, int32_t upper_bound_i) {
  return hardsigmoid_qdq(stream, in, in_type, out, out_type, 16, n, alpha_f, beta_f,
                         lower_bound_i, upper_bound_i);
}

int npu_qdq_sqrt_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                    npu_dtype out_type, int64_t n, float dq_zp, float dq_sc, float q_zp,
                    float q_sc) {
  return sqrt_qdq(stream, in, in_type, out, out_type, 8, n, dq_zp, dq_sc, q_zp, q_sc);
}

int npu_qdq_sqrt_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                     npu_dtype out_type, int64_t n, float dq_zp, float dq_sc, float q_zp,
                     float q_sc) {
  return sqrt_qdq(stream, in, in_type, out, out_type, 16, n, dq_zp, dq_sc, q_zp, q_sc);
}

}  // extern "C"
