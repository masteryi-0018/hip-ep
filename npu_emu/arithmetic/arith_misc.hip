/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Quantized division, leaky relu, prelu, square and batchnorm, see arith_misc.h. One thread
 * per output element.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "arithmetic/arith_misc.h"
#include "common/bcast.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/nlf.h"
#include "common/quant.h"

using namespace npu_emu;

namespace {

constexpr int kActRelu = 1;
constexpr int kActLrelu = 3;

// float(12582912 + x) - zp_word is x - zp exactly (zp_word = 12582912.0f + zp).
__global__ void div_kernel(int64_t n, const void* a, const void* b, npu_dtype it, void* out,
                           npu_dtype ot, float c0, float c1, float zp1, float zp2) {
  NPU_TID;
  const float fa = sub_q(float(12582912 + load_int(a, it, i)), zp1);
  const float fb = sub_q(float(12582912 + load_int(b, it, i)), zp2);
  const uint16_t db = f32_to_bf16(fb, kHalfEven);
  // a * inv(b): exact in fp32. 0 * inf is NaN.
  const float q =
      mul_q(round_float(fa, NPU_BF16, kHalfEven), bf16_to_f32(hw_inv16(db, NPU_BF16)));
  store_int(out, ot, i, quant(q, c0, c1, ot));
}

struct BDivArgs {
  int64_t W, C;
  int bc;  // broadcast bits of the divisor
  float c0, c1;
  int32_t zp1, zp2;
  int shift;
};

// a8: x - zp rounded to f16, a zero divisor replaced by f16(7e-5), q = a * inv_f16(b) exact.
__global__ void bdiv_a8_kernel(int64_t n, const void* a, const void* b, npu_dtype dt, void* out,
                               BDivArgs g) {
  NPU_TID;
  const int64_t xa = load_int(a, dt, i) - g.zp1;
  const int64_t xb = load_int(b, dt, bidx_flat(i, g.bc, g.W, g.C).flat()) - g.zp2;
  uint16_t hb = f32_to_f16(float(xb), kHalfEven);
  if ((hb & 0x7fff) == 0) hb = 0x0496;  // f16(7e-5)
  const float q =
      mul_q(round_float(float(xa), NPU_F16, kHalfEven), f16_to_f32(hw_inv16(hb, NPU_F16)));
  store_int(out, dt, i, quant(q, g.c0, g.c1, dt, kHalfEven, -g.shift));
}

// a16: the same in fp32 with the hardware fp32 reciprocal; the product is rounded once.
__global__ void bdiv_a16_kernel(int64_t n, const void* a, const void* b, npu_dtype dt,
                                void* out, BDivArgs g, const uint32_t* exc) {
  NPU_TID;
  const float fa = float(load_int(a, dt, i) - g.zp1);
  const int64_t xb = load_int(b, dt, bidx_flat(i, g.bc, g.W, g.C).flat()) - g.zp2;
  // A zero divisor uses the correctly rounded 1 / 7e-5f.
  const float inv = xb == 0 ? float(1.0 / double(7e-5f)) : hw_inv_f32(float(xb), exc);
  store_int(out, dt, i, quant(mul_q(fa, inv), g.c0, g.c1, dt, kHalfEven, -g.shift));
}

// The max of two MACs, both rounded to fp32.
__global__ void leakyrelu_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                                 npu_dtype ot, float c1, float c0, float c1a, float c0a) {
  NPU_TID;
  const int64_t x = load_int(in, it, i);
  const float p = mac_exact(x, c1, c0), q = mac_exact(x, c1a, c0a);
  store_int(out, ot, i, float_to_int(vmax(p, q), ot, 0, kHalfEven));
}

// The slope is selected on the sign of x - zp_in.
__global__ void prelu_kernel(int64_t n, const void* in, npu_dtype it, void* out, npu_dtype ot,
                             int64_t C, float pos, int32_t in_zp, float out_zp,
                             const float* neg) {
  NPU_TID;
  const int64_t d = load_int(in, it, i) - in_zp;
  const float s = d < 0 ? neg[i % C] : pos;
  store_int(out, ot, i, requant(d, s, out_zp, ot));
}

// Horner form, two MACs with the integer x each time.
__global__ void square_kernel(int64_t n, const void* in, npu_dtype it, void* out, npu_dtype ot,
                              float a, float b, float c) {
  NPU_TID;
  const int64_t x = load_int(in, it, i);
  store_int(out, ot, i, requant(x, mac_exact(x, a, b), c, ot));
}

__global__ void batchnorm_kernel(int64_t n, const void* in, npu_dtype dt, void* out, int64_t C,
                                 const float* cw, const float* cb, int act, float zp,
                                 float alpha) {
  NPU_TID;
  const int64_t c = i % C;
  float o = mac_exact(load_int(in, dt, i), cw[c], cb[c]);
  if (act == kActRelu) {
    o = vmax(o, zp);
  } else if (act == kActLrelu) {
    // Two separately rounded scalar operations, then one fma.
    o = activate_f32(o, kQdqActLrelu, alpha, mul_q(zp, sub_q(1.0f, alpha)));
  }
  store_int(out, dt, i, float_to_int(o, dt, 0, kHalfEven));
}

int div_impl(int bits, void* stream, const void* a, npu_dtype in_type, const void* b, void* out,
             npu_dtype out_type, int64_t n, float c0, float c1, float zp1, float zp2) {
  if (any_null(a, b, out) || !is_int_bits(in_type, bits) || !is_int_bits(out_type, bits) || n < 0)
    return -1;
  return launch(stream, n, div_kernel, a, b, in_type, out, out_type, c0, c1, zp1, zp2);
}

int bdiv_impl(int bits, void* stream, const void* a, npu_dtype dtype, const void* b, void* out,
              const int64_t* shape1, const int64_t* shape2, float c0, float c1, int32_t zp1,
              int32_t zp2, int shift) {
  const int bc = bcast_check(shape1, shape2);
  if (any_null(a, b, out) || !is_int_bits(dtype, bits) || bc < 0 || shift < -64 || shift > 64)
    return -1;
  const int64_t H = shape1[0], W = shape1[1], C = shape1[2];
  // Divisors of at most 16 significant bits.
  if (bits == 16 && (dtype == NPU_I16 ? (zp2 < -32768 || zp2 > 32767) : (zp2 < 0 || zp2 > 65535)))
    return -1;
  if (H * W * C == 0) return 0;
  const BDivArgs g{W, C, bc, c0, c1, zp1, zp2, shift};
  if (bits == 8) return launch(stream, H * W * C, bdiv_a8_kernel, a, b, dtype, out, g);
  const uint32_t* exc;
  if (int rc = npu_emu_nlf_table(NPU_EMU_NLF_INV_F32, &exc)) return rc;
  return launch(stream, H * W * C, bdiv_a16_kernel, a, b, dtype, out, g, exc);
}

int leakyrelu_impl(int bits, void* stream, const void* in, npu_dtype in_type, void* out,
                   npu_dtype out_type, int64_t n, float c1, float c0, float c1a, float c0a) {
  if (any_null(in, out) || !is_int_bits(in_type, bits) || !is_int_bits(out_type, bits) || n < 0)
    return -1;
  return launch(stream, n, leakyrelu_kernel, in, in_type, out, out_type, c1, c0, c1a, c0a);
}

int prelu_impl(int bits, void* stream, const void* in, npu_dtype in_type, void* out,
               npu_dtype out_type, int64_t rows, int64_t C, float pos, int32_t in_zp, float out_zp,
               const void* neg) {
  if (any_null(in, out, neg) || !is_int_bits(in_type, bits) || !is_int_bits(out_type, bits) ||
      !per_channel_ok(rows, C))
    return -1;
  return launch(stream, rows * C, prelu_kernel, in, in_type, out, out_type, C, pos, in_zp, out_zp,
                static_cast<const float*>(neg));
}

int square_impl(int bits, void* stream, const void* in, npu_dtype in_type, void* out,
                npu_dtype out_type, int64_t n, float a, float b, float c) {
  if (any_null(in, out) || !is_int_bits(in_type, bits) || !is_int_bits(out_type, bits) || n < 0)
    return -1;
  return launch(stream, n, square_kernel, in, in_type, out, out_type, a, b, c);
}

int batchnorm_impl(int bits, void* stream, const void* in, npu_dtype dtype, void* out, int64_t rows,
                   int64_t C, const void* cw, const void* cb, float zp, int act, float alpha) {
  if (any_null(in, out, cw, cb) || !is_int_bits(dtype, bits) || !per_channel_ok(rows, C) ||
      !(act == 0 || act == kActRelu || act == kActLrelu))
    return -1;
  return launch(stream, rows * C, batchnorm_kernel, in, dtype, out, C,
                static_cast<const float*>(cw), static_cast<const float*>(cb), act, zp, alpha);
}

}  // namespace

extern "C" {

int npu_elementwise_div_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float zero_point_coeff_1, float zero_point_coeff_2) {
  return div_impl(8, stream, a, in_type, b, out, out_type, n, c0, c1, zero_point_coeff_1,
                  zero_point_coeff_2);
}
int npu_elementwise_div_a16(void* stream, const void* a, npu_dtype in_type, const void* b,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float zero_point_coeff_1, float zero_point_coeff_2) {
  return div_impl(16, stream, a, in_type, b, out, out_type, n, c0, c1, zero_point_coeff_1,
                  zero_point_coeff_2);
}

int npu_qdq_elementwise_broadcasting3d_div_a8(void* stream, const void* a, npu_dtype dtype,
    const void* b, void* out, const int64_t* shape1, const int64_t* shape2, float ifm1_coeff,
    float ifm2_coeff, int32_t zero_point_coeff_1, int32_t zero_point_coeff_2, int output_sft) {
  return bdiv_impl(8, stream, a, dtype, b, out, shape1, shape2, ifm1_coeff, ifm2_coeff,
                   zero_point_coeff_1, zero_point_coeff_2, output_sft);
}
int npu_qdq_elementwise_broadcasting3d_div_a16(void* stream, const void* a, npu_dtype dtype,
    const void* b, void* out, const int64_t* shape1, const int64_t* shape2, float ifm1_coeff,
    float ifm2_coeff, int32_t zero_point_coeff_1, int32_t zero_point_coeff_2, int output_sft) {
  return bdiv_impl(16, stream, a, dtype, b, out, shape1, shape2, ifm1_coeff, ifm2_coeff,
                   zero_point_coeff_1, zero_point_coeff_2, output_sft);
}

int npu_elementwise_leakyrelu_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                                 npu_dtype out_type, int64_t n, float c1, float c0,
                                 float c1_alpha, float c0_alpha) {
  return leakyrelu_impl(8, stream, in, in_type, out, out_type, n, c1, c0, c1_alpha, c0_alpha);
}
int npu_elementwise_leakyrelu_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                                  npu_dtype out_type, int64_t n, float c1, float c0,
                                  float c1_alpha, float c0_alpha) {
  return leakyrelu_impl(16, stream, in, in_type, out, out_type, n, c1, c0, c1_alpha, c0_alpha);
}

int npu_elementwise_prelu_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                             npu_dtype out_type, int64_t rows, int64_t C, float positive_scale,
                             int32_t ifm_zero_point, float ofm_zero_point,
                             const void* prelu_scaled_slope) {
  return prelu_impl(8, stream, in, in_type, out, out_type, rows, C, positive_scale,
                    ifm_zero_point, ofm_zero_point, prelu_scaled_slope);
}
int npu_elementwise_prelu_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                              npu_dtype out_type, int64_t rows, int64_t C, float positive_scale,
                              int32_t ifm_zero_point, float ofm_zero_point,
                              const void* prelu_scaled_slope) {
  return prelu_impl(16, stream, in, in_type, out, out_type, rows, C, positive_scale,
                    ifm_zero_point, ofm_zero_point, prelu_scaled_slope);
}

int npu_elementwise_square_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                              npu_dtype out_type, int64_t n, float a, float b, float c) {
  return square_impl(8, stream, in, in_type, out, out_type, n, a, b, c);
}
int npu_elementwise_square_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                               npu_dtype out_type, int64_t n, float a, float b, float c) {
  return square_impl(16, stream, in, in_type, out, out_type, n, a, b, c);
}

int npu_qdq_batchnorm2d_stream_a8(void* stream, const void* in, npu_dtype dtype, void* out,
                                  int64_t rows, int64_t C, const void* cw, const void* cb,
                                  float ofm_zero_point, int activation, float relu_alpha) {
  return batchnorm_impl(8, stream, in, dtype, out, rows, C, cw, cb, ofm_zero_point, activation,
                        relu_alpha);
}
int npu_qdq_batchnorm2d_stream_a16(void* stream, const void* in, npu_dtype dtype, void* out,
                                   int64_t rows, int64_t C, const void* cw, const void* cb,
                                   float ofm_zero_point, int activation, float relu_alpha) {
  return batchnorm_impl(16, stream, in, dtype, out, rows, C, cw, cb, ofm_zero_point,
                        activation, relu_alpha);
}

}  // extern "C"
