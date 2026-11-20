/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Integer / QDQ elementwise arithmetic, see arith_int.h. One thread per output element.
 */
#include <hip/hip_runtime.h>

#include <cstring>

#include "common/npu_emu.h"
#include "arithmetic/arith_int.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/act.h"

using namespace npu_emu;

namespace {

// Logical HWC extents of ifm1 and of a broadcast operand (each dim 1 or equal).
struct Dims {
  int64_t h, w, c;
};

// Index into a broadcast operand of shape b for element i of an a-shaped tensor.
__device__ inline int64_t bcast_index(int64_t i, Dims a, Dims b) {
  const int64_t c = i % a.c, x = (i / a.c) % a.w, y = i / (a.c * a.w);
  const BIdx j = bidx(y, x, c, dims_bits(b.h, b.w, b.c), a.w, a.c);
  return j.pixel + j.c;
}

__device__ inline float relu_variants(float v, npu_arith_activation act, float a0, float a1) {
  return activate_f32(v, act == NPU_ARITH_ACT_CLIP    ? NPU_ACT_CLIP
                         : act == NPU_ARITH_ACT_LRELU ? NPU_ACT_LRELU
                                                      : NPU_ACT_NONE, a0, a1);
}

// Add: two MACs, ifm1 first.
__global__ void qdq_add_kernel(int64_t n, const void* a, const void* b, npu_dtype in_type,
                               void* out, npu_dtype out_type, Dims d1, Dims d2, float c0,
                               float c1, float c2, npu_arith_activation act, float a0,
                               float a1) {
  NPU_TID;
  const int64_t x1 = load_int(a, in_type, i);
  const int64_t x2 = load_int(b, in_type, bcast_index(i, d1, d2));
  float v = mac_exact(x2, c1, mac_exact(x1, c0, c2));
  store_int(out, out_type, i, float_to_int(relu_variants(v, act, a0, a1), out_type, 0, kHalfEven));
}

// Mul: Horner form, both ifm2 terms before the ifm1 MAC.
__global__ void qdq_mul_kernel(int64_t n, const void* a, const void* b, npu_dtype in_type,
                               void* out, npu_dtype out_type, Dims d1, Dims d2, float c0,
                               float c1, float c2, float c3, npu_arith_activation act, float a0,
                               float a1) {
  NPU_TID;
  const int64_t x1 = load_int(a, in_type, i);
  const int64_t x2 = load_int(b, in_type, bcast_index(i, d1, d2));
  const float t1 = mac_exact(x2, c2, c3);
  const float t2 = mac_exact(x2, c0, c1);
  const float v = mac_exact(x1, t2, t1);
  store_int(out, out_type, i, float_to_int(relu_variants(v, act, a0, a1), out_type, 0, kHalfEven));
}

// The C-broadcast route folds the scalar ifm2 into an intercept first; the other starts from
// ifm1, so the two round differently.
__global__ void bc3d_addsub_fused_kernel(int64_t n, const void* a, const void* b, npu_dtype t,
                                         void* out, Dims d1, Dims d2,
                                         npu_bc3d_addsub_fused_coeff k,
                                         npu_arith_activation act, const float* slope) {
  NPU_TID;
  const bool bcc = d2.c == 1;
  const int64_t x1 = load_int(a, t, i);
  const int64_t x2 = load_int(b, t, bcast_index(i, d1, d2));
  const float icpt = bcc ? mac_exact(x2, k.c1, k.c2) : 0.0f;
  const float r = bcc ? mac_exact(x1, k.c0, icpt) : mac_exact(x2, k.c1, mac_exact(x1, k.c0, k.c2));
  float v = r;
  switch (act) {
    case NPU_ARITH_ACT_RELU:
      v = vmax(r, k.w3);
      break;
    case NPU_ARITH_ACT_CLIP:
      v = vclamp(r, k.w3, k.w4);
      break;
    case NPU_ARITH_ACT_LRELU: {
      // bcc: leaky intercept fma(icpt, alpha, beta); else alpha is folded into the coefficients.
      const float l = bcc ? mac_exact(x1, k.w5, fmaf(icpt, k.w3, k.w4))
                          : mac_exact(x2, k.w6, mac_exact(x1, k.w5, k.w7));
      v = vmax(r, l);
      break;
    }
    case NPU_ARITH_ACT_PRELU:
      // r = A(c2 - zp_out); slope selected on its sign, then the zero point is added back.
      v = fmaf(r, r < 0.0f ? slope[i % d1.c] : 1.0f, k.w3);
      break;
    default:
      break;
  }
  store_int(out, t, i, float_to_int(v, t, 0, kHalfEven));
}

// Each route keeps its own form (Horner, zero-point shifted, or the integer product of LRELU a8).
__global__ void bc3d_mul_fused_kernel(int64_t n, const void* a, const void* b, npu_dtype t,
                                      void* out, Dims d1, Dims d2, npu_bc3d_mul_fused_coeff k,
                                      npu_arith_activation act, bool a16,
                                      const float* neg_scale) {
  NPU_TID;
  const bool bcc = d2.c == 1;
  const int64_t x1 = load_int(a, t, i);
  const int64_t x2 = load_int(b, t, bcast_index(i, d1, d2));
  const int64_t r1 = x1 - k.z1, r2 = x2 - k.z2;
  // A zero residual counts as positive.
  const bool neg = (r1 < 0) != (r2 < 0);
  const float F = mac_exact(x2, k.c0, k.c1);
  auto horner = [&] { return mac_exact(x1, F, mac_exact(x2, k.c2, k.c3)); };
  auto shifted = [&] { return mac_exact(r1, F, k.zy); };
  float v;
  switch (act) {
    case NPU_ARITH_ACT_RELU:
      v = vmax(bcc ? horner() : shifted(), k.act0);
      break;
    case NPU_ARITH_ACT_CLIP:
      v = vclamp(horner(), k.act0, k.act1);
      break;
    case NPU_ARITH_ACT_LRELU:
      if (!a16) {
        // a8: p = r1 r2 is formed exactly.
        const int64_t p = r1 * r2;
        v = bcc ? mac_exact(p, neg ? k.c1 : k.c0, k.zy)
                : mac_exact(p < 0 ? -p : p, k.c3, mac_exact(p, k.c2, k.zy));
      } else if (bcc) {
        const float leaky_f = F * k.act0;  // one rounding
        v = vmax(shifted(), mac_exact(r1, leaky_f, k.zy));
      } else {
        const float h = horner();
        v = vmax(h, fmaf(h, k.act0, k.act1));
      }
      break;
    case NPU_ARITH_ACT_PRELU: {
      const float scale = neg ? neg_scale[i % d1.c] : k.c0;
      v = mac_exact(r1, mac_exact(r2, scale, 0.0f), k.zy);
      break;
    }
    default:
      v = shifted();
      break;
  }
  store_int(out, t, i, float_to_int(v, t, 0, kHalfEven));
}

__global__ void bc3d_mac_fused_kernel(int64_t n, const void* a, npu_dtype ta, const void* b,
                                      npu_dtype tb, const void* c, npu_dtype tc, void* out,
                                      Dims d1, Dims d2, Dims d3, npu_bc3d_mac_fused_coeff k,
                                      npu_arith_activation act, bool addend_to_float) {
  NPU_TID;
  const bool bcc = d2.c == 1;
  const int64_t x1 = load_int(a, ta, i);
  const int64_t x2 = load_int(b, tb, bcast_index(i, d1, d2));
  const int64_t x3 = load_int(c, tc, bcast_index(i, d1, d3));
  float v;
  if (bcc && act == NPU_ARITH_ACT_NONE) {
    // Expanded form: zp1 folded into the constant with one fma.
    const float c3e = fmaf(float(k.zp1), -k.c1, k.c3);
    const float t1 = mac_exact(x2, k.c2, c3e);
    const float t2 = mac_exact(x2, k.c0, k.c1);
    v = mac_exact(x1, t2, mac_exact(x3, k.c4, t1));
  } else {
    const float f = mac_exact(x2, k.c0, k.c1);
    // addend_to_float: the scalar addend is rounded to fp32 before the MAC.
    const float w = bcc && addend_to_float ? fmaf(float(x3), k.c4, k.c3)
                                           : mac_exact(x3, k.c4, k.c3);
    v = mac_exact(x1 - k.zp1, f, w);
    if (act == NPU_ARITH_ACT_LRELU) v = vmax(v, fmaf(v, k.act0, k.act1));
  }
  int64_t q = float_to_int(v, ta, 0, kHalfEven);
  // RELU / CLIP clamp the quantized output against the quantized bounds.
  if (act == NPU_ARITH_ACT_RELU || act == NPU_ARITH_ACT_CLIP) {
    const int64_t lo = float_to_int(k.act0, ta, 0, kHalfEven);
    q = q < lo ? lo : q;
    if (act == NPU_ARITH_ACT_CLIP) {
      const int64_t hi = float_to_int(k.act1, ta, 0, kHalfEven);
      q = q > hi ? hi : q;
    }
  }
  store_int(out, ta, i, q);
}

// [default] The shifted terms saturate to int64 (a shift of 64 or more must not be a C shift),
// and so does their sum (sat_add64).
__global__ void xint_add_fused_kernel(int64_t n, const int8_t* a, const int8_t* b, int8_t* out,
                                      int s0, int s1, int os, bool clip, int64_t lo,
                                      int64_t hi) {
  NPU_TID;
  const int64_t v = sat_add64(shift_round(a[i], -s0, kHalfEven), shift_round(b[i], -s1, kHalfEven));
  int64_t q = clamp64(shift_round(v, os, kHalfEven), -128, 127);
  // [default] min(max(q, lo), hi), so crossed bounds give hi (as the fp32 clamp does)
  if (clip) {
    q = q < lo ? lo : q;
    q = q > hi ? hi : q;
  }
  out[i] = int8_t(q);
}

// Fixed point with the ifm2 scale folded into w.
__global__ void xint_mac_kernel(int64_t n, const void* a, npu_dtype ta, const void* b,
                                npu_dtype tb, const void* c, npu_dtype tc, void* out,
                                npu_dtype to, Dims d1, Dims d2, Dims d3, int32_t A, int32_t v,
                                int32_t p, int32_t t, bool scalar) {
  NPU_TID;
  const int64_t x1 = load_int(a, ta, i);
  const int64_t x2 = load_int(b, tb, bcast_index(i, d1, d2));
  const int64_t x3 = load_int(c, tc, bcast_index(i, d1, d3));
  int64_t w = shift_round(x2 * A, t, kHalfEven);
  // w saturates to int16; with a scalar ifm2 it wraps.
  w = scalar ? int64_t(int16_t(uint16_t(w))) : clamp64(w, -32768, 32767);
  int64_t lo, hi;
  int_range(to, &lo, &hi);
  store_int(out, to, i, clamp64(shift_round(x1 * w + x3 * v, p, kHalfEven), lo, hi));
}

bool get_dims(const int64_t* s, Dims* d) {
  if (!s || s[0] < 1 || s[1] < 1 || s[2] < 1) return false;
  *d = Dims{s[0], s[1], s[2]};
  return true;
}

bool bcast_ok(Dims a, Dims b) {
  return (b.h == 1 || b.h == a.h) && (b.w == 1 || b.w == a.w) && (b.c == 1 || b.c == a.c);
}

int flat_add(void* stream, const void* a, const void* b, npu_dtype it, void* out,
             npu_dtype ot, int64_t n, int bits, float c0, float c1, float c2,
             npu_arith_activation act, float a0, float a1) {
  if (!is_int_bits(it, bits) || !is_int_bits(ot, bits) || n < 0 ||
      (act != NPU_ARITH_ACT_NONE && act != NPU_ARITH_ACT_CLIP && act != NPU_ARITH_ACT_LRELU))
    return -1;
  const Dims d{1, 1, n > 0 ? n : 1};
  return launch(stream, n, qdq_add_kernel, a, b, it, out, ot, d, d, c0, c1, c2, act, a0, a1);
}

int flat_mul(void* stream, const void* a, const void* b, npu_dtype it, void* out,
             npu_dtype ot, int64_t n, int bits, float c0, float c1, float c2, float c3,
             npu_arith_activation act, float a0, float a1) {
  if (!is_int_bits(it, bits) || !is_int_bits(ot, bits) || n < 0 ||
      (act != NPU_ARITH_ACT_NONE && act != NPU_ARITH_ACT_CLIP && act != NPU_ARITH_ACT_LRELU))
    return -1;
  const Dims d{1, 1, n > 0 ? n : 1};
  return launch(stream, n, qdq_mul_kernel, a, b, it, out, ot, d, d, c0, c1, c2, c3, act, a0,
                a1);
}

int bc3d_binary(void* stream, const void* a, const void* b, npu_dtype it, void* out,
                npu_dtype ot, const int64_t* s1, const int64_t* s2, int bits, bool mul,
                float c0, float c1, float c2, float c3) {
  Dims d1, d2;
  if (!is_int_bits(it, bits) || !is_int_bits(ot, bits) || !get_dims(s1, &d1) ||
      !get_dims(s2, &d2) || !bcast_ok(d1, d2))
    return -1;
  const int64_t n = d1.h * d1.w * d1.c;
  if (mul)
    return launch(stream, n, qdq_mul_kernel, a, b, it, out, ot, d1, d2, c0, c1, c2, c3,
                  NPU_ARITH_ACT_NONE, 0.0f, 0.0f);
  return launch(stream, n, qdq_add_kernel, a, b, it, out, ot, d1, d2, c0, c1, c2,
                NPU_ARITH_ACT_NONE, 0.0f, 0.0f);
}

bool act_ok(npu_arith_activation act, bool prelu) {
  return act == NPU_ARITH_ACT_NONE || act == NPU_ARITH_ACT_RELU || act == NPU_ARITH_ACT_CLIP ||
         act == NPU_ARITH_ACT_LRELU || (prelu && act == NPU_ARITH_ACT_PRELU);
}

int bc3d_addsub_fused(void* stream, const void* a, const void* b, npu_dtype t, void* out,
                      const int64_t* s1, const int64_t* s2, int bits,
                      const npu_bc3d_addsub_fused_coeff* k, npu_arith_activation act,
                      const float* slope) {
  Dims d1, d2;
  if (!is_int_bits(t, bits) || !k || !act_ok(act, true) ||
      (act == NPU_ARITH_ACT_PRELU) != (slope != nullptr) || !get_dims(s1, &d1) ||
      !get_dims(s2, &d2) || !bcast_ok(d1, d2))
    return -1;
  return launch(stream, d1.h * d1.w * d1.c, bc3d_addsub_fused_kernel, a, b, t, out, d1, d2, *k,
                act, slope);
}

int bc3d_mul_fused(void* stream, const void* a, const void* b, npu_dtype t, void* out,
                   const int64_t* s1, const int64_t* s2, int bits,
                   const npu_bc3d_mul_fused_coeff* k, npu_arith_activation act,
                   const float* neg_scale) {
  Dims d1, d2;
  if (!is_int_bits(t, bits) || !k || !act_ok(act, true) ||
      (act == NPU_ARITH_ACT_PRELU) != (neg_scale != nullptr) || !get_dims(s1, &d1) ||
      !get_dims(s2, &d2) || !bcast_ok(d1, d2))
    return -1;
  return launch(stream, d1.h * d1.w * d1.c, bc3d_mul_fused_kernel, a, b, t, out, d1, d2, *k,
                act, bits == 16, neg_scale);
}

int bc3d_mac_fused(void* stream, const void* a, npu_dtype ta, const void* b, npu_dtype tb,
                   const void* c, npu_dtype tc, void* out, const int64_t* s1, const int64_t* s2,
                   const int64_t* s3, int a_bits, int w_bits, bool addend_to_float,
                   const npu_bc3d_mac_fused_coeff* k, npu_arith_activation act) {
  Dims d1, d2, d3;
  const bool c_ok = is_i8(tc) ? a_bits == 8 : (is_i16(tc) || tc == NPU_I32);
  if (!is_int_bits(ta, a_bits) || !is_int_bits(tb, w_bits) || !c_ok || !k || !act_ok(act, false) ||
      !get_dims(s1, &d1) || !get_dims(s2, &d2) || !get_dims(s3, &d3) || !bcast_ok(d1, d2) ||
      !bcast_ok(d1, d3) || d2.c != d3.c)
    return -1;
  return launch(stream, d1.h * d1.w * d1.c, bc3d_mac_fused_kernel, a, ta, b, tb, c, tc, out, d1,
                d2, d3, *k, act, addend_to_float);
}

// Exponent of a positive power of two, or -1000.
int pow2_exp(float f) {
  uint32_t b;
  memcpy(&b, &f, 4);
  const int e = (b >> 23) & 0xff;
  if ((b >> 31) || (b & 0x7fffff) || e == 0 || e == 0xff) return -1000;
  return e - 127;
}

}  // namespace

extern "C" {

int npu_elementwise_add_a8(void* stream, const void* a, const void* b, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float c2) {
  return flat_add(stream, a, b, in_type, out, out_type, n, 8, c0, c1, c2, NPU_ARITH_ACT_NONE,
                  0.0f, 0.0f);
}

int npu_elementwise_add_a16(void* stream, const void* a, const void* b, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float c2) {
  return flat_add(stream, a, b, in_type, out, out_type, n, 16, c0, c1, c2, NPU_ARITH_ACT_NONE,
                  0.0f, 0.0f);
}

int npu_elementwise_mul_a8(void* stream, const void* a, const void* b, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float c2, float c3) {
  return flat_mul(stream, a, b, in_type, out, out_type, n, 8, c0, c1, c2, c3,
                  NPU_ARITH_ACT_NONE, 0.0f, 0.0f);
}

int npu_elementwise_mul_a16(void* stream, const void* a, const void* b, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float c2, float c3) {
  return flat_mul(stream, a, b, in_type, out, out_type, n, 16, c0, c1, c2, c3,
                  NPU_ARITH_ACT_NONE, 0.0f, 0.0f);
}

int npu_elementwise_add_fused_a8(void* stream, const void* a, const void* b,
                                 npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                 float c0, float c1, float c2, npu_arith_activation activation,
                                 float act0, float act1) {
  return flat_add(stream, a, b, in_type, out, out_type, n, 8, c0, c1, c2, activation, act0,
                  act1);
}

int npu_elementwise_add_fused_a16(void* stream, const void* a, const void* b,
                                  npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2,
                                  npu_arith_activation activation, float act0, float act1) {
  return flat_add(stream, a, b, in_type, out, out_type, n, 16, c0, c1, c2, activation, act0,
                  act1);
}

int npu_elementwise_sub_fused_a8(void* stream, const void* a, const void* b,
                                 npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                 float c0, float c1, float c2, npu_arith_activation activation,
                                 float act0, float act1) {
  return flat_add(stream, a, b, in_type, out, out_type, n, 8, c0, c1, c2, activation, act0,
                  act1);
}

int npu_elementwise_sub_fused_a16(void* stream, const void* a, const void* b,
                                  npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2,
                                  npu_arith_activation activation, float act0, float act1) {
  return flat_add(stream, a, b, in_type, out, out_type, n, 16, c0, c1, c2, activation, act0,
                  act1);
}

int npu_elementwise_mul_fused_a8(void* stream, const void* a, const void* b,
                                 npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                 float c0, float c1, float c2, float c3,
                                 npu_arith_activation activation, float act0, float act1) {
  return flat_mul(stream, a, b, in_type, out, out_type, n, 8, c0, c1, c2, c3, activation, act0,
                  act1);
}

int npu_elementwise_mul_fused_a16(void* stream, const void* a, const void* b,
                                  npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2, float c3,
                                  npu_arith_activation activation, float act0, float act1) {
  return flat_mul(stream, a, b, in_type, out, out_type, n, 16, c0, c1, c2, c3, activation,
                  act0, act1);
}

int npu_elementwise_add_a8_xint(void* stream, const void* a, const void* b, void* out,
                                int64_t n, float c0, float c1, float c2) {
  return flat_add(stream, a, b, NPU_I8, out, NPU_I8, n, 8, c0, c1, c2, NPU_ARITH_ACT_NONE,
                  0.0f, 0.0f);
}

int npu_elementwise_add_fused_a8_xint(void* stream, const void* a, const void* b, void* out,
                                      int64_t n, float c0, float c1,
                                      npu_arith_activation activation, float act0,
                                      float act1) {
  const int e0 = pow2_exp(c0), e1 = pow2_exp(c1);
  if (n < 0 || e0 == -1000 || e1 == -1000 ||
      (activation != NPU_ARITH_ACT_NONE && activation != NPU_ARITH_ACT_CLIP))
    return -1;
  const int lower = e0 < e1 ? e0 : e1;
  const int base = lower < 0 ? lower : 0;
  return launch(stream, n, xint_add_fused_kernel, static_cast<const int8_t*>(a),
                static_cast<const int8_t*>(b), static_cast<int8_t*>(out), e0 - base, e1 - base,
                -base, activation == NPU_ARITH_ACT_CLIP,
                float_to_int(act0, NPU_I8, 0, kHalfEven), float_to_int(act1, NPU_I8, 0, kHalfEven));
}

int npu_elementwise_mac_a8(void* stream, const void* ifm1, npu_dtype ifm1_type,
                           const void* ifm2, npu_dtype ifm2_type, const void* ifm3,
                           npu_dtype ifm3_type, void* out, npu_dtype out_type,
                           const int64_t* shape1, const int64_t* shape2,
                           const int64_t* shape3, int32_t A, int32_t v, int32_t p,
                           int32_t t) {
  Dims d1, d2, d3;
  if (!is_i8(ifm1_type) || !is_i8(ifm2_type) || !is_i8(ifm3_type) || !is_i8(out_type) ||
      !get_dims(shape1, &d1) || !get_dims(shape2, &d2) || !get_dims(shape3, &d3) ||
      !bcast_ok(d1, d2) || !bcast_ok(d1, d3) || p < 0 || p > 31 || t < 0 || t > 31)
    return -1;
  const bool scalar = d2.h == 1 && d2.w == 1 && d2.c == 1;
  return launch(stream, d1.h * d1.w * d1.c, xint_mac_kernel, ifm1, ifm1_type, ifm2, ifm2_type,
                ifm3, ifm3_type, out, out_type, d1, d2, d3, A, v, p, t, scalar);
}

int npu_qdq_elementwise_broadcasting3d_add_a8(void* stream, const void* ifm1, const void* ifm2,
                                              npu_dtype in_type, void* out, npu_dtype out_type,
                                              const int64_t* shape1, const int64_t* shape2,
                                              float c0, float c1, float c2) {
  return bc3d_binary(stream, ifm1, ifm2, in_type, out, out_type, shape1, shape2, 8, false, c0,
                     c1, c2, 0.0f);
}

int npu_qdq_elementwise_broadcasting3d_add_a16(void* stream, const void* ifm1,
                                               const void* ifm2, npu_dtype in_type, void* out,
                                               npu_dtype out_type, const int64_t* shape1,
                                               const int64_t* shape2, float c0, float c1,
                                               float c2) {
  return bc3d_binary(stream, ifm1, ifm2, in_type, out, out_type, shape1, shape2, 16, false, c0,
                     c1, c2, 0.0f);
}

int npu_qdq_elementwise_broadcasting3d_mul_a8(void* stream, const void* ifm1, const void* ifm2,
                                              npu_dtype in_type, void* out, npu_dtype out_type,
                                              const int64_t* shape1, const int64_t* shape2,
                                              float c0, float c1, float c2, float c3) {
  return bc3d_binary(stream, ifm1, ifm2, in_type, out, out_type, shape1, shape2, 8, true, c0,
                     c1, c2, c3);
}

int npu_qdq_elementwise_broadcasting3d_mul_a16(void* stream, const void* ifm1,
                                               const void* ifm2, npu_dtype in_type, void* out,
                                               npu_dtype out_type, const int64_t* shape1,
                                               const int64_t* shape2, float c0, float c1,
                                               float c2, float c3) {
  return bc3d_binary(stream, ifm1, ifm2, in_type, out, out_type, shape1, shape2, 16, true, c0,
                     c1, c2, c3);
}

int npu_qdq_elementwise_broadcasting3d_add_fused_a8(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    npu_arith_activation activation, const float* slope) {
  return bc3d_addsub_fused(stream, ifm1, ifm2, in_type, out, shape1, shape2, 8, coeff,
                           activation, slope);
}

int npu_qdq_elementwise_broadcasting3d_add_fused_a16(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    npu_arith_activation activation, const float* slope) {
  return bc3d_addsub_fused(stream, ifm1, ifm2, in_type, out, shape1, shape2, 16, coeff,
                           activation, slope);
}

int npu_qdq_elementwise_broadcasting3d_sub_fused_a8(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    npu_arith_activation activation, const float* slope) {
  return bc3d_addsub_fused(stream, ifm1, ifm2, in_type, out, shape1, shape2, 8, coeff,
                           activation, slope);
}

int npu_qdq_elementwise_broadcasting3d_sub_fused_a16(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    npu_arith_activation activation, const float* slope) {
  return bc3d_addsub_fused(stream, ifm1, ifm2, in_type, out, shape1, shape2, 16, coeff,
                           activation, slope);
}

int npu_qdq_elementwise_broadcasting3d_mul_fused_a8(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_mul_fused_coeff* coeff,
    npu_arith_activation activation, const float* neg_scale) {
  return bc3d_mul_fused(stream, ifm1, ifm2, in_type, out, shape1, shape2, 8, coeff, activation,
                        neg_scale);
}

int npu_qdq_elementwise_broadcasting3d_mul_fused_a16(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_mul_fused_coeff* coeff,
    npu_arith_activation activation, const float* neg_scale) {
  return bc3d_mul_fused(stream, ifm1, ifm2, in_type, out, shape1, shape2, 16, coeff, activation,
                        neg_scale);
}

int npu_qdq_elementwise_broadcasting3d_mac_fused_a8w8(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const int64_t* shape3,
    const npu_bc3d_mac_fused_coeff* coeff, npu_arith_activation activation) {
  return bc3d_mac_fused(stream, ifm1, ifm1_type, ifm2, ifm2_type, ifm3, ifm3_type, out, shape1,
                        shape2, shape3, 8, 8, true, coeff, activation);
}

int npu_qdq_elementwise_broadcasting3d_mac_fused_a16w8(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const int64_t* shape3,
    const npu_bc3d_mac_fused_coeff* coeff, npu_arith_activation activation) {
  return bc3d_mac_fused(stream, ifm1, ifm1_type, ifm2, ifm2_type, ifm3, ifm3_type, out, shape1,
                        shape2, shape3, 16, 8, false, coeff, activation);
}

int npu_qdq_elementwise_broadcasting3d_mac_fused_a16w16(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const int64_t* shape3,
    const npu_bc3d_mac_fused_coeff* coeff, npu_arith_activation activation) {
  return bc3d_mac_fused(stream, ifm1, ifm1_type, ifm2, ifm2_type, ifm3, ifm3_type, out, shape1,
                        shape2, shape3, 16, 16, true, coeff, activation);
}

}  // extern "C"
