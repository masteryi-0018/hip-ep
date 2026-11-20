/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Sigmoid / tanh kernels. One thread per element.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/nlf.h"
#include "nonlinear/sigmoid.h"

namespace {

using namespace npu_emu;

constexpr int kLutEntries = 128;


// Dequantize into the compute type. |x| < 2^16, so float(x) is exact.
__device__ inline float dq(int64_t x, float zp, float s, npu_dtype ht, Rnd rnd) {
  return round_float(fma_q(float(x), s, mul_q(-zp, s)), ht, rnd, kFtzIn);
}

// Quantize: fma in fp32, rounded with rnd, saturated. NaN -> 0.
__device__ inline int64_t q(float y, float zp, float s, npu_dtype t, Rnd rnd) {
  return float_to_int(fma_q(y, s, zp), t, 0, rnd);
}

struct QdqCoeff {
  float dq_zp, dq_s, q_zp, q_s;
};

// fma(x, slope, offset) of the entry of x; the f16 variants round the slope to f16 first.
__device__ inline float interp(float x, const float* lut, float k, npu_dtype slope_t, Rnd rnd) {
  return lut_interp(x, lut, lut_index(x, k, kLutEntries, rnd), slope_t, rnd);
}

// Sigmoid table on a compute-type x: 0 / 1 outside [-8, 8).
__device__ inline float sigmoid_lut(float x, const float* lut, npu_dtype ht, Rnd rnd) {
  if (x != x) return qnan();
  if (x < -8.0f) return 0.0f;
  if (x >= 8.0f) return 1.0f;
  return round_float(interp(x, lut, 8.0f, ht == NPU_F16 ? NPU_F16 : NPU_F32, rnd), ht, rnd);
}

// Tanh table: bf16 selects -1 / 1 outside [-4, 4); f16 clamps the interpolant to [-1, 1].
__device__ inline float tanh_lut(float x, const float* lut, npu_dtype ht, Rnd rnd) {
  if (x != x) return qnan();
  if (ht != NPU_F16) {
    if (x < -4.0f) return -1.0f;
    if (x >= 4.0f) return 1.0f;
  }
  const npu_dtype slope_t = ht == NPU_F16 ? NPU_F16 : NPU_F32;
  const float o = round_float(interp(x, lut, 16.0f, slope_t, rnd), ht, rnd);
  return ht == NPU_F16 ? vclamp(o, -1.0f, 1.0f) : o;
}

enum QdqOp { kSigmoid = 0, kTanh = 1, kSigmoidLut = 2, kTanhLut = 3 };

__global__ void qdq_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int op, npu_dtype ht, QdqCoeff c,
                           const float* lut) {
  NPU_TID;
  const int64_t x = load_int(in, in_type, i);
  float y;
  Rnd rnd = kHalfAway;
  switch (op) {
    case kSigmoid: {
      const float v = dq(x, c.dq_zp, c.dq_s, ht, kHalfAway);
      const float t = tanh_acc(mul_q(v, -0.5f), ht);
      // 0.5 - t * 0.5 in one rounding, then bf16
      y = round_float(fma_q(t, -0.5f, 0.5f), NPU_BF16, kHalfAway);
      break;
    }
    case kTanh:
      y = tanh_acc(dq(x, c.dq_zp, c.dq_s, ht, kHalfAway), ht);
      break;
    case kSigmoidLut:
      y = sigmoid_lut(dq(x, c.dq_zp, c.dq_s, NPU_BF16, kHalfAway), lut, NPU_BF16, kHalfAway);
      break;
    default:  // kTanhLut: f16, half to even
      rnd = kHalfEven;
      y = tanh_lut(dq(x, c.dq_zp, c.dq_s, NPU_F16, kHalfEven), lut, NPU_F16, kHalfEven);
      break;
  }
  store_int(out, out_type, i, q(y, c.q_zp, c.q_s, out_type, rnd));
}

__global__ void lut16_kernel(int64_t n, const uint16_t* in, uint16_t* out, npu_dtype ht,
                             bool tanh_op, const float* lut) {
  NPU_TID;
  const float x = decode_float(in[i], ht);
  const float y = tanh_op ? tanh_lut(x, lut, ht, kHalfEven) : sigmoid_lut(x, lut, ht, kHalfEven);
  out[i] = encode_float(y, ht, kHalfEven);
}

__global__ void sigmoid_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, const float* lut,
                                   float s_in, float recip) {
  NPU_TID;
  const float x = mul_q(fp8_decode_identity(in, i, n), s_in);
  // fp32 interpolation (no 16-bit rounding)
  const float y = vclamp(interp(x, lut, 8.0f, NPU_F32, kHalfAway), 0.0f, 1.0f);
  // no +-448 clamp: overflow is NaN
  out[i] = f32_to_fp8(mul_q(y, recip), kHalfAway);
}

__global__ void tanh_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, npu_dtype ht,
                                const float* lut, float s_in, float r) {
  NPU_TID;
  // fp32 product, then the compute type, half away
  const float v = mul_q(fp8_decode_identity(in, i, n), s_in);
  const float x = round_float(v, ht, kHalfAway);
  const float y = tanh_lut(x, lut, ht, kHalfAway);
  // y * r is exact (two 16-bit values); saturated to +-448
  out[i] = f32_to_fp8(mul_q(y, r), kHalfAway, kSat);
}

__global__ void tanh16_kernel(int64_t n, const uint16_t* in, uint16_t* out, npu_dtype ht) {
  NPU_TID;
  out[i] = tanh16(in[i], ht);
}

int run_qdq(void* stream, const void* in, npu_dtype in_type, void* out, npu_dtype out_type,
            int64_t n, bool a16, int op, npu_dtype compute_type, QdqCoeff c, const float* lut) {
  const bool ok_io = a16 ? is_i16(in_type) && is_i16(out_type) : is_i8(in_type) && is_i8(out_type);
  if (!ok_io || n < 0 || !is_half_float(compute_type)) return -1;
  if ((op == kSigmoidLut || op == kTanhLut) && !lut && n > 0) return -1;
  return launch(stream, n, qdq_kernel, in, in_type, out, out_type, op, compute_type, c,
                lut);
}

}  // namespace

extern "C" {

#define NPU_QDQ_ENTRY(name, a16, op)                                                           \
  int name(void* stream, const void* in, npu_dtype in_type, void* out, npu_dtype out_type,     \
           int64_t n, npu_dtype compute_type, float dq_zero_point, float dq_scale,             \
           float q_zero_point, float q_scale) {                                                \
    return run_qdq(stream, in, in_type, out, out_type, n, a16, op, compute_type,               \
                   QdqCoeff{dq_zero_point, dq_scale, q_zero_point, q_scale}, nullptr);         \
  }
NPU_QDQ_ENTRY(npu_sigmoid_a8, false, kSigmoid)
NPU_QDQ_ENTRY(npu_sigmoid_a16, true, kSigmoid)
NPU_QDQ_ENTRY(npu_tanh_a8, false, kTanh)
NPU_QDQ_ENTRY(npu_tanh_a16, true, kTanh)
#undef NPU_QDQ_ENTRY

#define NPU_QDQ_LUT_ENTRY(name, a16, op)                                                       \
  int name(void* stream, const void* in, npu_dtype in_type, void* out, npu_dtype out_type,     \
           int64_t n, float dq_zero_point, float dq_scale, float q_zero_point, float q_scale,  \
           const float* lut) {                                                                 \
    return run_qdq(stream, in, in_type, out, out_type, n, a16, op,                             \
                   op == kTanhLut ? NPU_F16 : NPU_BF16,                                        \
                   QdqCoeff{dq_zero_point, dq_scale, q_zero_point, q_scale}, lut);             \
  }
NPU_QDQ_LUT_ENTRY(npu_qdq_sigmoid_lut_a8, false, kSigmoidLut)
NPU_QDQ_LUT_ENTRY(npu_qdq_sigmoid_lut_a16, true, kSigmoidLut)
NPU_QDQ_LUT_ENTRY(npu_qdq_tanh_lut_a8, false, kTanhLut)
NPU_QDQ_LUT_ENTRY(npu_qdq_tanh_lut_a16, true, kTanhLut)
#undef NPU_QDQ_LUT_ENTRY

int npu_elementwise_sigmoid_lut_bf16(void* stream, const void* in, void* out, npu_dtype type,
                                     int64_t n, const float* lut) {
  if (!is_half_float(type) || n < 0 || (!lut && n > 0)) return -1;
  return launch(stream, n, lut16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), type, false, lut);
}

int npu_elementwise_tanh_lut_bf16(void* stream, const void* in, void* out, npu_dtype type,
                                  int64_t n, const float* lut) {
  if (!is_half_float(type) || n < 0 || (!lut && n > 0)) return -1;
  return launch(stream, n, lut16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), type, true, lut);
}

int npu_elementwise_sigmoid_lut_fp8(void* stream, const void* in, void* out, int64_t n,
                                    const float* lut, float ifm_scale, float ofm_scale_recip) {
  if (n < 0 || (!lut && n > 0)) return -1;
  return launch(stream, n, sigmoid_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), lut, ifm_scale, ofm_scale_recip);
}

int npu_elementwise_tanh_lut_fp8(void* stream, const void* in, void* out, int64_t n,
                                 npu_dtype compute_type, const float* lut, float ifm_scale,
                                 float ofm_scale_recip) {
  if (!is_half_float(compute_type) || n < 0 || (!lut && n > 0)) return -1;
  const npu_dtype ht = compute_type;
  // the scale is cast half to even for bf16, half away for f16
  return launch(stream, n, tanh_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), ht, lut, ifm_scale,
                scalar_cast(ofm_scale_recip, ht, kHalfAway));
}

int npu_test_sigmoid_tanh16(void* stream, const void* in, npu_dtype type, void* out, int64_t n) {
  if (!is_half_float(type) || n < 0) return -1;
  return launch(stream, n, tanh16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), type);
}

}  // extern "C"
