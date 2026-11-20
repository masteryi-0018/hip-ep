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
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/nlf.h"

using namespace npu_emu;

namespace {

bool bcast_ok(int64_t full, int64_t b) { return full > 0 && (b == full || b == 1); }

// float(12582912 + x) - zp_word is x - zp exactly (zp_word = 12582912.0f + zp).
__global__ void div_kernel(int64_t n, const void* a, const void* b, npu_dtype it, void* out,
                           npu_dtype ot, float c0, float c1, float zp1, float zp2) {
  NPU_TID;
  const float fa = float(12582912 + load_int(a, it, i)) - zp1;
  const float fb = float(12582912 + load_int(b, it, i)) - zp2;
  const uint16_t da = f32_to_bf16(fa, kHalfEven), db = f32_to_bf16(fb, kHalfEven);
  // a * inv(b): exact in fp32. 0 * inf is NaN.
  const float q = bf16_to_f32(da) * bf16_to_f32(hw_inv16(db, NPU_BF16));
  store_int(out, ot, i, float_to_int(fmaf(q, c0, c1), ot, 0, kHalfEven));
}

struct BDivArgs {
  int64_t W, C, bH, bW, bC;
  float c0, c1;
  int32_t zp1, zp2;
  int shift;
};

// Flat index of the broadcast operand [bH][bW][bC] for output element i of [H][W][C].
__device__ inline int64_t bcast_index(int64_t i, const BDivArgs& g) {
  const int64_t c = i % g.C, w = (i / g.C) % g.W, h = i / (g.C * g.W);
  const BIdx j = bidx(h, w, c, dims_bits(g.bH, g.bW, g.bC), g.W, g.C);
  return j.pixel + j.c;
}

// a8: x - zp rounded to f16, a zero divisor replaced by f16(7e-5), q = a * inv_f16(b) exact.
__global__ void bdiv_a8_kernel(int64_t n, const void* a, const void* b, npu_dtype dt, void* out,
                               BDivArgs g) {
  NPU_TID;
  const int64_t xa = load_int(a, dt, i) - g.zp1;
  const int64_t xb = load_int(b, dt, bcast_index(i, g)) - g.zp2;
  const uint16_t ha = f32_to_f16(float(xa), kHalfEven);
  uint16_t hb = f32_to_f16(float(xb), kHalfEven);
  if ((hb & 0x7fff) == 0) hb = 0x0496;  // f16(7e-5)
  const float q = f16_to_f32(ha) * f16_to_f32(hw_inv16(hb, NPU_F16));
  store_int(out, dt, i, float_to_int(fmaf(q, g.c0, g.c1), dt, -g.shift, kHalfEven));
}

// a16: the same in fp32 with the hardware fp32 reciprocal; the product is rounded once.
__global__ void bdiv_a16_kernel(int64_t n, const void* a, const void* b, npu_dtype dt,
                                void* out, BDivArgs g, const uint32_t* exc) {
  NPU_TID;
  const float fa = float(load_int(a, dt, i) - g.zp1);
  const int64_t xb = load_int(b, dt, bcast_index(i, g)) - g.zp2;
  // A zero divisor uses the correctly rounded 1 / 7e-5f.
  const float inv = xb == 0 ? float(1.0 / double(7e-5f)) : hw_inv_f32(float(xb), exc);
  const float q = fa * inv;
  store_int(out, dt, i, float_to_int(fmaf(q, g.c0, g.c1), dt, -g.shift, kHalfEven));
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
  store_int(out, ot, i, float_to_int(mac_exact(d, s, out_zp), ot, 0, kHalfEven));
}

// Horner form, two MACs with the integer x each time.
__global__ void square_kernel(int64_t n, const void* in, npu_dtype it, void* out, npu_dtype ot,
                              float a, float b, float c) {
  NPU_TID;
  const int64_t x = load_int(in, it, i);
  const float t = mac_exact(x, a, b);
  store_int(out, ot, i, float_to_int(mac_exact(x, t, c), ot, 0, kHalfEven));
}

__global__ void batchnorm_kernel(int64_t n, const void* in, npu_dtype dt, void* out, int64_t C,
                                 const float* cw, const float* cb, int act, float zp,
                                 float alpha) {
  NPU_TID;
  const int64_t c = i % C;
  float o = mac_exact(load_int(in, dt, i), cw[c], cb[c]);
  if (act == NPU_QDQ_BN_RELU) {
    o = vmax(o, zp);
  } else if (act == NPU_QDQ_BN_LRELU) {
    // Two separately rounded scalar operations, then one fma.
    const float lc = zp * (1.0f - alpha);
    const float t = fmaf(o, alpha, lc);
    o = vmax(o, t);
  }
  store_int(out, dt, i, float_to_int(o, dt, 0, kHalfEven));
}

int div_impl(int bits, void* stream, const void* a, const void* b, npu_dtype it, void* out,
             npu_dtype ot, int64_t n, float c0, float c1, float zp1, float zp2) {
  if (!is_int_bits(it, bits) || !is_int_bits(ot, bits) || n < 0) return -1;
  return launch(stream, n, div_kernel, a, b, it, out, ot, c0, c1, zp1, zp2);
}

int bdiv_impl(int bits, void* stream, const void* a, const void* b, npu_dtype dt, void* out,
              int64_t H, int64_t W, int64_t C, int64_t bH, int64_t bW, int64_t bC, float c0,
              float c1, int32_t zp1, int32_t zp2, int shift) {
  if (!is_int_bits(dt, bits) || H < 0 || !bcast_ok(W, bW) || !bcast_ok(C, bC) ||
      !(bH == H || bH == 1) || shift < -64 || shift > 64)
    return -1;
  // Divisors of at most 16 significant bits.
  if (bits == 16 && (dt == NPU_I16 ? (zp2 < -32768 || zp2 > 32767) : (zp2 < 0 || zp2 > 65535)))
    return -1;
  const BDivArgs g{W, C, bH, bW, bC, c0, c1, zp1, zp2, shift};
  if (bits == 8) return launch(stream, H * W * C, bdiv_a8_kernel, a, b, dt, out, g);
  const uint32_t* exc;
  if (int rc = npu_emu_nlf_table(NPU_EMU_NLF_INV_F32, &exc)) return rc;
  return launch(stream, H * W * C, bdiv_a16_kernel, a, b, dt, out, g, exc);
}

int leakyrelu_impl(int bits, void* stream, const void* in, npu_dtype it, void* out,
                   npu_dtype ot, int64_t n, float c1, float c0, float c1a, float c0a) {
  if (!is_int_bits(it, bits) || !is_int_bits(ot, bits) || n < 0) return -1;
  return launch(stream, n, leakyrelu_kernel, in, it, out, ot, c1, c0, c1a, c0a);
}

int prelu_impl(int bits, void* stream, const void* in, npu_dtype it, void* out, npu_dtype ot,
               int64_t rows, int64_t C, float pos, int32_t in_zp, float out_zp,
               const float* neg) {
  if (!is_int_bits(it, bits) || !is_int_bits(ot, bits) || rows < 0 || C <= 0 || !neg) return -1;
  return launch(stream, rows * C, prelu_kernel, in, it, out, ot, C, pos, in_zp, out_zp, neg);
}

int square_impl(int bits, void* stream, const void* in, npu_dtype it, void* out, npu_dtype ot,
                int64_t n, float a, float b, float c) {
  if (!is_int_bits(it, bits) || !is_int_bits(ot, bits) || n < 0) return -1;
  return launch(stream, n, square_kernel, in, it, out, ot, a, b, c);
}

int batchnorm_impl(int bits, void* stream, const void* in, npu_dtype dt, void* out,
                   int64_t rows, int64_t C, const float* cw, const float* cb,
                   npu_qdq_bn_activation act, float zp, float alpha) {
  if (!is_int_bits(dt, bits) || rows < 0 || C <= 0 || !cw || !cb ||
      (act != NPU_QDQ_BN_NONE && act != NPU_QDQ_BN_RELU && act != NPU_QDQ_BN_LRELU))
    return -1;
  return launch(stream, rows * C, batchnorm_kernel, in, dt, out, C, cw, cb, int(act), zp, alpha);
}

}  // namespace

extern "C" {

int npu_elementwise_div_a8(void* stream, const void* a, const void* b, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float zp1_word, float zp2_word) {
  return div_impl(8, stream, a, b, in_type, out, out_type, n, c0, c1, zp1_word, zp2_word);
}
int npu_elementwise_div_a16(void* stream, const void* a, const void* b, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float zp1_word, float zp2_word) {
  return div_impl(16, stream, a, b, in_type, out, out_type, n, c0, c1, zp1_word, zp2_word);
}

int npu_qdq_elementwise_broadcasting3d_div_a8(void* stream, const void* a, const void* b,
    npu_dtype dtype, void* out, const int64_t* shape1, const int64_t* shape2, float c0, float c1,
    int32_t zp1, int32_t zp2, int output_shift) {
  if (!shape1 || !shape2) return -1;
  const int64_t H = shape1[0], W = shape1[1], C = shape1[2];
  const int64_t bH = shape2[0], bW = shape2[1], bC = shape2[2];
  return bdiv_impl(8, stream, a, b, dtype, out, H, W, C, bH, bW, bC, c0, c1, zp1, zp2,
                   output_shift);
}
int npu_qdq_elementwise_broadcasting3d_div_a16(void* stream, const void* a, const void* b,
    npu_dtype dtype, void* out, const int64_t* shape1, const int64_t* shape2, float c0, float c1,
    int32_t zp1, int32_t zp2, int output_shift) {
  if (!shape1 || !shape2) return -1;
  const int64_t H = shape1[0], W = shape1[1], C = shape1[2];
  const int64_t bH = shape2[0], bW = shape2[1], bC = shape2[2];
  return bdiv_impl(16, stream, a, b, dtype, out, H, W, C, bH, bW, bC, c0, c1, zp1, zp2,
                   output_shift);
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
                             int32_t in_zp, float out_zp, const float* negative_scale) {
  return prelu_impl(8, stream, in, in_type, out, out_type, rows, C, positive_scale, in_zp,
                    out_zp, negative_scale);
}
int npu_elementwise_prelu_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                              npu_dtype out_type, int64_t rows, int64_t C,
                              float positive_scale, int32_t in_zp, float out_zp,
                              const float* negative_scale) {
  return prelu_impl(16, stream, in, in_type, out, out_type, rows, C, positive_scale, in_zp,
                    out_zp, negative_scale);
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
                                  int64_t rows, int64_t C, const float* cw, const float* cb,
                                  npu_qdq_bn_activation act, float ofm_zp, float alpha) {
  return batchnorm_impl(8, stream, in, dtype, out, rows, C, cw, cb, act, ofm_zp, alpha);
}
int npu_qdq_batchnorm2d_stream_a16(void* stream, const void* in, npu_dtype dtype, void* out,
                                   int64_t rows, int64_t C, const float* cw, const float* cb,
                                   npu_qdq_bn_activation act, float ofm_zp, float alpha) {
  return batchnorm_impl(16, stream, in, dtype, out, rows, C, cw, cb, act, ofm_zp, alpha);
}

}  // extern "C"
