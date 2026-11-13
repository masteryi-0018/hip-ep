/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for include/npu_emu_arith_int.h. One thread per
 * output element.
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>

#include "npu_emu.h"
#include "npu_emu_arith_int.h"
#include "npu_emu_device.h"
#include "npu_emu_launch.h"

using namespace npu_emu;

// Every fp32 operation rounds separately; the NPU's fused MACs are written out
// (mac_exact / fmaf).
#pragma clang fp contract(off)

namespace {

// Logical HWC extents of ifm1 and of a broadcast operand (each dim 1 or equal).
struct Dims {
  int64_t h, w, c;
};

// Index into a broadcast operand of shape b for element i of an a-shaped tensor.
__device__ inline int64_t bcast_index(int64_t i, Dims a, Dims b) {
  const int64_t c = i % a.c, x = (i / a.c) % a.w, y = i / (a.c * a.w);
  return ((b.h == 1 ? 0 : y) * b.w + (b.w == 1 ? 0 : x)) * b.c + (b.c == 1 ? 0 : c);
}

__device__ inline int64_t sat_out(float v, npu_dtype t) {
  int64_t lo, hi;
  int_range(t, &lo, &hi);
  return sat_rne(v, lo, hi);
}

// aie::clamp / max on fp32 vectors.
__device__ inline float clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }

// impl_common.h apply_relu_variants, the activation of the flat fused kernels:
// CLIP clamps, LRELU takes max(v, mac_elem_32(v, alpha, beta)).
__device__ inline float relu_variants(float v, npu_arith_activation act, float a0, float a1) {
  if (act == NPU_ARITH_ACT_CLIP) return clampf(v, a0, a1);
  if (act == NPU_ARITH_ACT_LRELU) return fmaxf(v, fmaf(v, a0, a1));
  return v;
}

// The QDQ add kernels (flat, Broadcasting3d, Xint8): two MACs, ifm1 first.
__global__ void qdq_add_kernel(int64_t n, const void* a, const void* b, npu_dtype in_type,
                               void* out, npu_dtype out_type, Dims d1, Dims d2, float c0,
                               float c1, float c2, npu_arith_activation act, float a0,
                               float a1) {
  NPU_TID;
  const int64_t x1 = load_int(a, in_type, i);
  const int64_t x2 = load_int(b, in_type, bcast_index(i, d1, d2));
  float v = mac_exact(x2, c1, mac_exact(x1, c0, c2));
  store_int(out, out_type, i, sat_out(relu_variants(v, act, a0, a1), out_type));
}

// The QDQ mul kernels: Horner form, both ifm2 terms before the ifm1 MAC.
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
  store_int(out, out_type, i, sat_out(relu_variants(v, act, a0, a1), out_type));
}

// qdq_elementwise_broadcasting3d_add_fused_{a8,a16}.h. The channel-broadcast
// routes fold the scalar ifm2 into an intercept first, the others start from
// ifm1, so the two routes round differently.
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
      v = fmaxf(r, k.w3);
      break;
    case NPU_ARITH_ACT_CLIP:
      v = clampf(r, k.w3, k.w4);
      break;
    case NPU_ARITH_ACT_LRELU: {
      // bcc: the intercept is turned into the leaky one, fma(icpt, alpha, beta);
      // otherwise alpha is folded into all three coefficients.
      const float l = bcc ? mac_exact(x1, k.w5, fmaf(icpt, k.w3, k.w4))
                          : mac_exact(x2, k.w6, mac_exact(x1, k.w5, k.w7));
      v = fmaxf(r, l);
      break;
    }
    case NPU_ARITH_ACT_PRELU:
      // r is d = A(c2 - zp_out); the slope selection is on its sign, then one
      // MAC adds the output zero point back.
      v = fmaf(r, r < 0.0f ? slope[i % d1.c] : 1.0f, k.w3);
      break;
    default:
      break;
  }
  store_int(out, t, i, sat_out(v, t));
}

// qdq_elementwise_broadcasting3d_mul_fused_{a8,a16}.h: each route keeps its own
// form (Horner, zero-point shifted, or the integer-product forms of LRELU a8).
__global__ void bc3d_mul_fused_kernel(int64_t n, const void* a, const void* b, npu_dtype t,
                                      void* out, Dims d1, Dims d2, npu_bc3d_mul_fused_coeff k,
                                      npu_arith_activation act, bool a16,
                                      const float* neg_scale) {
  NPU_TID;
  const bool bcc = d2.c == 1;
  const int64_t x1 = load_int(a, t, i);
  const int64_t x2 = load_int(b, t, bcast_index(i, d1, d2));
  const int64_t r1 = x1 - k.z1, r2 = x2 - k.z2;
  // c0 > 0, so the sign of the product is the xor of the residual signs (a zero
  // residual counts as positive).
  const bool neg = (r1 < 0) != (r2 < 0);
  const float F = mac_exact(x2, k.c0, k.c1);
  auto horner = [&] { return mac_exact(x1, F, mac_exact(x2, k.c2, k.c3)); };
  auto shifted = [&] { return mac_exact(r1, F, k.zy); };
  float v;
  switch (act) {
    case NPU_ARITH_ACT_RELU:
      v = fmaxf(bcc ? horner() : shifted(), k.act0);
      break;
    case NPU_ARITH_ACT_CLIP:
      v = clampf(horner(), k.act0, k.act1);
      break;
    case NPU_ARITH_ACT_LRELU:
      if (!a16) {
        // a8 residuals fit int16, so the kernel forms p = r1 r2 exactly.
        const int64_t p = r1 * r2;
        v = bcc ? mac_exact(p, neg ? k.c1 : k.c0, k.zy)
                : mac_exact(p < 0 ? -p : p, k.c3, mac_exact(p, k.c2, k.zy));
      } else if (bcc) {
        const float leaky_f = F * k.act0;  // mul_elem_32: one rounding
        v = fmaxf(shifted(), mac_exact(r1, leaky_f, k.zy));
      } else {
        const float h = horner();
        v = fmaxf(h, fmaf(h, k.act0, k.act1));
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
  store_int(out, t, i, sat_out(v, t));
}

// qdq_elementwise_broadcasting3d_mac_fused_{a8w8,a16w8,a16w16}.h.
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
    // Expanded form: zp1 folded into the constant with one msc.
    const float c3e = fmaf(float(k.zp1), -k.c1, k.c3);
    const float t1 = mac_exact(x2, k.c2, c3e);
    const float t2 = mac_exact(x2, k.c0, k.c1);
    v = mac_exact(x1, t2, mac_exact(x3, k.c4, t1));
  } else {
    const float f = mac_exact(x2, k.c0, k.c1);
    // fix2float on the scalar addend of the activated bcc routes (A8W8,
    // A16W16) rounds int32 addends to fp32 before the MAC.
    const float w = bcc && addend_to_float ? fmaf(float(x3), k.c4, k.c3)
                                           : mac_exact(x3, k.c4, k.c3);
    v = mac_exact(x1 - k.zp1, f, w);
    if (act == NPU_ARITH_ACT_LRELU) v = fmaxf(v, fmaf(v, k.act0, k.act1));
  }
  int64_t q = sat_out(v, ta);
  // RELU / CLIP clamp the quantized output against the quantized bounds.
  if (act == NPU_ARITH_ACT_RELU || act == NPU_ARITH_ACT_CLIP) {
    const int64_t lo = sat_out(k.act0, ta);
    q = q < lo ? lo : q;
    if (act == NPU_ARITH_ACT_CLIP) {
      const int64_t hi = sat_out(k.act1, ta);
      q = q > hi ? hi : q;
    }
  }
  store_int(out, ta, i, q);
}

// Integer shift right by s with round half to even (the SRS under conv_even).
__device__ inline int64_t srs_even(int64_t v, int s) {
  if (s <= 0) return v << -s;
  const int64_t q = v >> s;  // floor
  const int64_t r = v - (q << s);
  const int64_t half = int64_t(1) << (s - 1);
  return q + ((r > half || (r == half && (q & 1))) ? 1 : 0);
}

__device__ inline int64_t clamp64(int64_t v, int64_t lo, int64_t hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

// elementwise_add_fused_a8_xint.h, both integer loops (they hold the same
// integer before the SRS).
__global__ void xint_add_fused_kernel(int64_t n, const int8_t* a, const int8_t* b, int8_t* out,
                                      int s0, int s1, int os, bool clip, int64_t lo,
                                      int64_t hi) {
  NPU_TID;
  const int64_t v = (int64_t(a[i]) << s0) + (int64_t(b[i]) << s1);
  int64_t q = clamp64(srs_even(v, os), -128, 127);
  if (clip) q = clamp64(q, lo, hi);
  out[i] = int8_t(q);
}

// elementwise_mac_a8.h: fixed point with the ifm2 scale folded into w.
__global__ void xint_mac_kernel(int64_t n, const void* a, npu_dtype ta, const void* b,
                                npu_dtype tb, const void* c, npu_dtype tc, void* out,
                                npu_dtype to, Dims d1, Dims d2, Dims d3, int32_t A, int32_t v,
                                int32_t p, int32_t t, bool scalar) {
  NPU_TID;
  const int64_t x1 = load_int(a, ta, i);
  const int64_t x2 = load_int(b, tb, bcast_index(i, d1, d2));
  const int64_t x3 = load_int(c, tc, bcast_index(i, d1, d3));
  int64_t w = srs_even(x2 * A, t);
  // The vector pre-scale saturates to int16; the scalar path casts (wraps).
  w = scalar ? int64_t(int16_t(uint16_t(w))) : clamp64(w, -32768, 32767);
  int64_t lo, hi;
  int_range(to, &lo, &hi);
  store_int(out, to, i, clamp64(srs_even(x1 * w + x3 * v, p), lo, hi));
}

bool is8(npu_dtype t) { return t == NPU_I8 || t == NPU_U8; }
bool is16(npu_dtype t) { return t == NPU_I16 || t == NPU_U16; }
bool width_ok(npu_dtype t, int bits) { return bits == 8 ? is8(t) : is16(t); }

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
  if (!width_ok(it, bits) || !width_ok(ot, bits) || n < 0 ||
      (act != NPU_ARITH_ACT_NONE && act != NPU_ARITH_ACT_CLIP && act != NPU_ARITH_ACT_LRELU))
    return -1;
  const Dims d{1, 1, n > 0 ? n : 1};
  return launch(stream, n, qdq_add_kernel, a, b, it, out, ot, d, d, c0, c1, c2, act, a0, a1);
}

int flat_mul(void* stream, const void* a, const void* b, npu_dtype it, void* out,
             npu_dtype ot, int64_t n, int bits, float c0, float c1, float c2, float c3,
             npu_arith_activation act, float a0, float a1) {
  if (!width_ok(it, bits) || !width_ok(ot, bits) || n < 0 ||
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
  if (!width_ok(it, bits) || !width_ok(ot, bits) || !get_dims(s1, &d1) || !get_dims(s2, &d2) ||
      !bcast_ok(d1, d2))
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
  if (!width_ok(t, bits) || !k || !act_ok(act, true) ||
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
  if (!width_ok(t, bits) || !k || !act_ok(act, true) ||
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
  const bool c_ok = is8(tc) ? a_bits == 8 : (is16(tc) || tc == NPU_I32);
  if (!width_ok(ta, a_bits) || !width_ok(tb, w_bits) || !c_ok || !k || !act_ok(act, false) ||
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

// float -> int8 as to_v32int8(broadcast(x), 0, 1): half to even, saturated.
int64_t host_sat8(float x) {
  if (x != x) return 0;
  const double r = nearbyint(double(x));  // default FE_TONEAREST: half to even
  return r <= -128 ? -128 : r >= 127 ? 127 : int64_t(r);
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
  // xint8_add_fused_detail::integer_scale
  const int lower = e0 < e1 ? e0 : e1;
  const int base = lower < 0 ? lower : 0;
  return launch(stream, n, xint_add_fused_kernel, static_cast<const int8_t*>(a),
                static_cast<const int8_t*>(b), static_cast<int8_t*>(out), e0 - base, e1 - base,
                -base, activation == NPU_ARITH_ACT_CLIP, host_sat8(act0), host_sat8(act1));
}

int npu_elementwise_mac_a8(void* stream, const void* ifm1, npu_dtype ifm1_type,
                           const void* ifm2, npu_dtype ifm2_type, const void* ifm3,
                           npu_dtype ifm3_type, void* out, npu_dtype out_type,
                           const int64_t* shape1, const int64_t* shape2,
                           const int64_t* shape3, int32_t A, int32_t v, int32_t p,
                           int32_t t) {
  Dims d1, d2, d3;
  if (!is8(ifm1_type) || !is8(ifm2_type) || !is8(ifm3_type) || !is8(out_type) ||
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
