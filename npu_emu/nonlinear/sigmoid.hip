/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Sigmoid / tanh kernels. One thread per element.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/act.h"
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
#include "nonlinear/sigmoid.h"

namespace {

using namespace npu_emu;

constexpr int kLutEntries = 128;

// Dequantize into the compute type.
__device__ inline float dq(int64_t x, float zp, float s, npu_dtype ht, Rnd rnd) {
  return round_float(dequant(x, s, dq_offset(zp, s)), ht, rnd, kFtzIn);
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
    case kSigmoid:
      y = sigmoid_tanh(dq(x, c.dq_zp, c.dq_s, ht, kHalfAway), ht, kHalfAway);
      break;
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
  // a NaN converts to 0
  store_int(out, out_type, i, quant(y, c.q_s, c.q_zp, out_type, rnd));
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
  out[i] = fp8_out(y, recip, kHalfAway, 0);
}

__global__ void tanh_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, npu_dtype ht,
                                const float* lut, float s_in, float r) {
  NPU_TID;
  const float x = fp8_in(in, i, n, s_in, ht, kHalfAway);
  // y * r is exact (two 16-bit values)
  out[i] = fp8_out(tanh_lut(x, lut, ht, kHalfAway), r, kHalfAway);
}

__global__ void tanh16_kernel(int64_t n, const uint16_t* in, uint16_t* out, npu_dtype ht) {
  NPU_TID;
  out[i] = tanh16(in[i], ht);
}

int run_qdq(void* stream, const void* in, npu_dtype in_type, const void* lut, void* out,
            npu_dtype out_type, int64_t n, int bits, int op, npu_dtype compute_type,
            QdqCoeff c) {
  // every buffer is required, also when there is nothing to process
  if (!is_int_bits(in_type, bits) || !is_int_bits(out_type, bits) || n < 0 ||
      !is_half_float(compute_type) || any_null(in, out))
    return -1;
  if ((op == kSigmoidLut || op == kTanhLut) && !lut) return -1;
  return launch(stream, n, qdq_kernel, in, in_type, out, out_type, op, compute_type, c,
                static_cast<const float*>(lut));
}

int run_lut16(void* stream, const void* in, npu_dtype in_type, const void* lut, void* out,
              int64_t n, bool tanh_op) {
  if (!is_half_float(in_type) || n < 0 || any_null(in, out, lut)) return -1;
  return launch(stream, n, lut16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), in_type, tanh_op, static_cast<const float*>(lut));
}

}  // namespace

extern "C" {

int npu_sigmoid_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                   npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                   float dq_sc, float q_zp, float q_sc) {
  return run_qdq(stream, in, in_type, nullptr, out, out_type, n, 8, kSigmoid, compute_type,
                 QdqCoeff{dq_zp, dq_sc, q_zp, q_sc});
}

int npu_sigmoid_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                    npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                    float dq_sc, float q_zp, float q_sc) {
  return run_qdq(stream, in, in_type, nullptr, out, out_type, n, 16, kSigmoid, compute_type,
                 QdqCoeff{dq_zp, dq_sc, q_zp, q_sc});
}

int npu_tanh_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                float dq_sc, float q_zp, float q_sc) {
  return run_qdq(stream, in, in_type, nullptr, out, out_type, n, 8, kTanh, compute_type,
                 QdqCoeff{dq_zp, dq_sc, q_zp, q_sc});
}

int npu_tanh_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                 npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                 float dq_sc, float q_zp, float q_sc) {
  return run_qdq(stream, in, in_type, nullptr, out, out_type, n, 16, kTanh, compute_type,
                 QdqCoeff{dq_zp, dq_sc, q_zp, q_sc});
}

// The table variants have a fixed inner type: bf16 for sigmoid, f16 for tanh.
int npu_qdq_sigmoid_lut_a8(void* stream, const void* in, npu_dtype in_type, const void* lut,
                           void* out, npu_dtype out_type, int64_t n, float dq_zp,
                           float dq_sc, float q_zp, float q_sc) {
  return run_qdq(stream, in, in_type, lut, out, out_type, n, 8, kSigmoidLut, NPU_BF16,
                 QdqCoeff{dq_zp, dq_sc, q_zp, q_sc});
}

int npu_qdq_sigmoid_lut_a16(void* stream, const void* in, npu_dtype in_type, const void* lut,
                            void* out, npu_dtype out_type, int64_t n, float dq_zp,
                            float dq_sc, float q_zp, float q_sc) {
  return run_qdq(stream, in, in_type, lut, out, out_type, n, 16, kSigmoidLut, NPU_BF16,
                 QdqCoeff{dq_zp, dq_sc, q_zp, q_sc});
}

int npu_qdq_tanh_lut_a8(void* stream, const void* in, npu_dtype in_type, const void* lut,
                        void* out, npu_dtype out_type, int64_t n, float dq_zp, float dq_sc,
                        float q_zp, float q_sc) {
  return run_qdq(stream, in, in_type, lut, out, out_type, n, 8, kTanhLut, NPU_F16,
                 QdqCoeff{dq_zp, dq_sc, q_zp, q_sc});
}

int npu_qdq_tanh_lut_a16(void* stream, const void* in, npu_dtype in_type, const void* lut,
                         void* out, npu_dtype out_type, int64_t n, float dq_zp, float dq_sc,
                         float q_zp, float q_sc) {
  return run_qdq(stream, in, in_type, lut, out, out_type, n, 16, kTanhLut, NPU_F16,
                 QdqCoeff{dq_zp, dq_sc, q_zp, q_sc});
}

int npu_elementwise_sigmoid_lut_bf16(void* stream, const void* in, npu_dtype in_type,
                                     const void* lut, void* out, int64_t n) {
  return run_lut16(stream, in, in_type, lut, out, n, false);
}

int npu_elementwise_tanh_lut_bf16(void* stream, const void* in, npu_dtype in_type,
                                  const void* lut, void* out, int64_t n) {
  return run_lut16(stream, in, in_type, lut, out, n, true);
}

int npu_elementwise_sigmoid_lut_fp8(void* stream, const void* in, const void* lut, void* out,
                                    int64_t n, float ifm_scale, float ofm_scale_recip) {
  if (n < 0 || any_null(in, out, lut)) return -1;
  return launch(stream, n, sigmoid_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), static_cast<const float*>(lut), ifm_scale,
                ofm_scale_recip);
}

int npu_elementwise_tanh_lut_fp8(void* stream, const void* in, const void* lut, void* out,
                                 int64_t n, npu_dtype compute_type, float ifm_scale,
                                 float ofm_scale_recip) {
  if (!is_half_float(compute_type) || n < 0 || any_null(in, out, lut)) return -1;
  const npu_dtype ht = compute_type;
  // the scale is cast half to even for bf16, half away for f16
  return launch(stream, n, tanh_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), ht, static_cast<const float*>(lut), ifm_scale,
                scalar_cast(ofm_scale_recip, ht, kHalfAway));
}

int npu_test_sigmoid_tanh16(void* stream, const void* in, npu_dtype in_type, void* out,
                            int64_t n) {
  if (!is_half_float(in_type) || n < 0 || any_null(in, out)) return -1;
  return launch(stream, n, tanh16_kernel, static_cast<const uint16_t*>(in),
                static_cast<uint16_t*>(out), in_type);
}

}  // extern "C"
