/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for include/npu_emu_arith_misc.h. One thread per
 * output element; reference-speed kernels.
 */
#include <hip/hip_runtime.h>

#include "npu_emu.h"
#include "npu_emu_arith_misc.h"
#include "npu_emu_arith_misc_device.h"
#include "npu_emu_device.h"
#include "npu_emu_inv_f32_table.h"
#include "npu_emu_launch.h"

using namespace npu_emu;

// The kernels round every fp32 operation separately; fused multiply-adds are
// written out (fmaf / mac_exact) where the NPU fuses.
#pragma clang fp contract(off)

namespace {

bool is8(npu_dtype t) { return t == NPU_I8 || t == NPU_U8; }
bool is16(npu_dtype t) { return t == NPU_I16 || t == NPU_U16; }
bool width_ok(int bits, npu_dtype t) { return bits == 8 ? is8(t) : is16(t); }
bool bcast_ok(int64_t full, int64_t b) { return full > 0 && (b == full || b == 1); }

// fp32 -> integer with to_vector_sign / to_fixed under conv_even.
__device__ inline int64_t to_out(float v, npu_dtype t, int shift = 0) {
  return to_int_sat(v, t, shift, kIntEven);
}

// ---- A. division ---------------------------------------------------------------

// elementwise_div_a8.h / _a16.h. bitcast(x + 0x4B400000) is 12582912.0f + x
// exactly, and subtracting the pre-biased zero point word is exact too
// (Sterbenz), so the kernel gets x - zp in fp32 for one integer add.
__global__ void div_kernel(int64_t n, const void* a, const void* b, npu_dtype it, void* out,
                           npu_dtype ot, float c0, float c1, float zp1, float zp2) {
  NPU_TID;
  const float fa = float(12582912 + load_int(a, it, i)) - zp1;
  const float fb = float(12582912 + load_int(b, it, i)) - zp2;
  const uint16_t da = f32_to_bf16(fa, kEven), db = f32_to_bf16(fb, kEven);
  // aie::div(bf16, bf16) = a * inv(b); the product of two 8-bit significands
  // is exact in fp32. 0 * inf is NaN.
  const float q = bf16_to_f32(da) * bf16_to_f32(inv_bf16_aie4(db));
  store_int(out, ot, i, to_out(fmaf(q, c0, c1), ot));
}

struct BDivArgs {
  int64_t W, C, bH, bW, bC;
  float c0, c1;
  int32_t zp1, zp2;
  int shift;
};

// Flat index of the broadcast operand [bH][bW][bC] for output element i of
// [H][W][C].
__device__ inline int64_t bcast_index(int64_t i, const BDivArgs& g) {
  const int64_t c = i % g.C, w = (i / g.C) % g.W, h = i / (g.C * g.W);
  return ((g.bH == 1 ? 0 : h) * g.bW + (g.bW == 1 ? 0 : w)) * g.bC + (g.bC == 1 ? 0 : c);
}

// qdq_elementwise_broadcasting3d_div_a8.h: int32 x - zp -> fp16 (to_float
// rounds to nearest even), a zero divisor replaced by fp16(7e-5), then
// aie::div on fp16 = a * inv_fp16(b), exact in fp32.
__global__ void bdiv_a8_kernel(int64_t n, const void* a, const void* b, npu_dtype dt, void* out,
                               BDivArgs g) {
  NPU_TID;
  const int64_t xa = load_int(a, dt, i) - g.zp1;
  const int64_t xb = load_int(b, dt, bcast_index(i, g)) - g.zp2;
  const uint16_t ha = f32_to_f16(float(xa), kEven);
  uint16_t hb = f32_to_f16(float(xb), kEven);
  if ((hb & 0x7fff) == 0) hb = 0x0496;  // fp16(7e-5)
  const float q = f16_to_f32(ha) * f16_to_f32(inv_f16_aie4(hb));
  store_int(out, dt, i, to_out(fmaf(q, g.c0, g.c1), dt, -g.shift));
}

// qdq_elementwise_broadcasting3d_div_a16.h: the same in fp32 (x - zp is exact
// there; fp16 would overflow at +-65535) with the scalar fp32 ::inv, whose
// result is not correctly rounded; the fp32 product is rounded once.
__global__ void bdiv_a16_kernel(int64_t n, const void* a, const void* b, npu_dtype dt,
                                void* out, BDivArgs g) {
  NPU_TID;
  const float fa = float(load_int(a, dt, i) - g.zp1);
  const int64_t xb = load_int(b, dt, bcast_index(i, g)) - g.zp2;
  // ::inv(7e-5f) is the correctly rounded 1/7e-5f (aiesim).
  const float inv = xb == 0 ? float(1.0 / double(7e-5f)) : inv_f32_aie4(float(xb));
  const float q = fa * inv;
  store_int(out, dt, i, to_out(fmaf(q, g.c0, g.c1), dt, -g.shift));
}

// ---- B. activations ------------------------------------------------------------

// elementwise_leakyrelu.h: the max of two MACs, both rounded to fp32.
__global__ void leakyrelu_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                                 npu_dtype ot, float c1, float c0, float c1a, float c0a) {
  NPU_TID;
  const int64_t x = load_int(in, it, i);
  const float p = mac_exact(x, c1, c0), q = mac_exact(x, c1a, c0a);
  store_int(out, ot, i, to_out(p > q ? p : q, ot));
}

// elementwise_prelu_qdq.h: the slope is selected on the sign of x - zp_in.
__global__ void prelu_kernel(int64_t n, const void* in, npu_dtype it, void* out, npu_dtype ot,
                             int64_t C, float pos, int32_t in_zp, float out_zp,
                             const float* neg) {
  NPU_TID;
  const int64_t d = load_int(in, it, i) - in_zp;
  const float s = d < 0 ? neg[i % C] : pos;
  store_int(out, ot, i, to_out(mac_exact(d, s, out_zp), ot));
}

// elementwise_square.h: Horner form, two MACs with the integer x each time.
__global__ void square_kernel(int64_t n, const void* in, npu_dtype it, void* out, npu_dtype ot,
                              float a, float b, float c) {
  NPU_TID;
  const int64_t x = load_int(in, it, i);
  const float t = mac_exact(x, a, b);
  store_int(out, ot, i, to_out(mac_exact(x, t, c), ot));
}

// ---- C. batch normalization ----------------------------------------------------

// qdq_batchnorm2d_stream_a8.hpp / _a16.hpp.
__global__ void batchnorm_kernel(int64_t n, const void* in, npu_dtype dt, void* out, int64_t C,
                                 const float* cw, const float* cb, int act, float zp,
                                 float alpha) {
  NPU_TID;
  const int64_t c = i % C;
  float o = mac_exact(load_int(in, dt, i), cw[c], cb[c]);
  if (act == NPU_QDQ_BN_RELU) {
    o = o > zp ? o : zp;
  } else if (act == NPU_QDQ_BN_LRELU) {
    // Scalar fp32 operations on the core: two roundings, then one MAC.
    const float lc = zp * (1.0f - alpha);
    const float t = fmaf(o, alpha, lc);
    o = o > t ? o : t;
  }
  store_int(out, dt, i, to_out(o, dt));
}

int div_impl(int bits, void* stream, const void* a, const void* b, npu_dtype it, void* out,
             npu_dtype ot, int64_t n, float c0, float c1, float zp1, float zp2) {
  if (!width_ok(bits, it) || !width_ok(bits, ot) || n < 0) return -1;
  return launch(stream, n, div_kernel, a, b, it, out, ot, c0, c1, zp1, zp2);
}

int bdiv_impl(int bits, void* stream, const void* a, const void* b, npu_dtype dt, void* out,
              int64_t H, int64_t W, int64_t C, int64_t bH, int64_t bW, int64_t bC, float c0,
              float c1, int32_t zp1, int32_t zp2, int shift) {
  if (!width_ok(bits, dt) || H < 0 || !bcast_ok(W, bW) || !bcast_ok(C, bC) ||
      !(bH == H || bH == 1) || shift < -64 || shift > 64)
    return -1;
  // The fp32 reciprocal table covers divisors of at most 16 significant bits.
  if (bits == 16 && (dt == NPU_I16 ? (zp2 < -32768 || zp2 > 32767) : (zp2 < 0 || zp2 > 65535)))
    return -1;
  const BDivArgs g{W, C, bH, bW, bC, c0, c1, zp1, zp2, shift};
  return bits == 8 ? launch(stream, H * W * C, bdiv_a8_kernel, a, b, dt, out, g)
                   : launch(stream, H * W * C, bdiv_a16_kernel, a, b, dt, out, g);
}

int leakyrelu_impl(int bits, void* stream, const void* in, npu_dtype it, void* out,
                   npu_dtype ot, int64_t n, float c1, float c0, float c1a, float c0a) {
  if (!width_ok(bits, it) || !width_ok(bits, ot) || n < 0) return -1;
  return launch(stream, n, leakyrelu_kernel, in, it, out, ot, c1, c0, c1a, c0a);
}

int prelu_impl(int bits, void* stream, const void* in, npu_dtype it, void* out, npu_dtype ot,
               int64_t rows, int64_t C, float pos, int32_t in_zp, float out_zp,
               const float* neg) {
  if (!width_ok(bits, it) || !width_ok(bits, ot) || rows < 0 || C <= 0 || !neg) return -1;
  return launch(stream, rows * C, prelu_kernel, in, it, out, ot, C, pos, in_zp, out_zp, neg);
}

int square_impl(int bits, void* stream, const void* in, npu_dtype it, void* out, npu_dtype ot,
                int64_t n, float a, float b, float c) {
  if (!width_ok(bits, it) || !width_ok(bits, ot) || n < 0) return -1;
  return launch(stream, n, square_kernel, in, it, out, ot, a, b, c);
}

int batchnorm_impl(int bits, void* stream, const void* in, npu_dtype dt, void* out,
                   int64_t rows, int64_t C, const float* cw, const float* cb,
                   npu_qdq_bn_activation act, float zp, float alpha) {
  if (!width_ok(bits, dt) || rows < 0 || C <= 0 || !cw || !cb ||
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
                                              npu_dtype dtype, void* out, int64_t H, int64_t W,
                                              int64_t C, int64_t bH, int64_t bW, int64_t bC,
                                              float c0, float c1, int32_t zp1, int32_t zp2,
                                              int output_shift) {
  return bdiv_impl(8, stream, a, b, dtype, out, H, W, C, bH, bW, bC, c0, c1, zp1, zp2,
                   output_shift);
}
int npu_qdq_elementwise_broadcasting3d_div_a16(void* stream, const void* a, const void* b,
                                               npu_dtype dtype, void* out, int64_t H,
                                               int64_t W, int64_t C, int64_t bH, int64_t bW,
                                               int64_t bC, float c0, float c1, int32_t zp1,
                                               int32_t zp2, int output_shift) {
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
