/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Integer / QDQ elementwise arithmetic, see arith_int.h. One thread per output element.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "arithmetic/arith_int.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/bcast.h"
#include "common/bits.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/quant.h"

using namespace npu_emu;

namespace {

// W and C of ifm1 ([H][W][C]); the other operands carry their broadcast bits.
struct Dims {
  int64_t w, c;
};

// Activation codes of the QDQ entries.
constexpr int kActNone = 0;
constexpr int kActRelu = 1;
constexpr int kActRelu6 = 2;
constexpr int kActLrelu = 3;
constexpr int kActPrelu = 9;

// Flat fused activation: relu and relu6 both clamp to [a0, a1].
__device__ inline int relu_variants(int act) {
  return act == kActRelu || act == kActRelu6 ? kQdqActClip
         : act == kActLrelu                  ? kQdqActLrelu
                                             : kQdqActNone;
}

// Add: two MACs, ifm1 first.
__global__ void qdq_add_kernel(int64_t n, const void* a, const void* b, npu_dtype in_type,
                               void* out, npu_dtype out_type, Dims d, int bc, float c0,
                               float c1, float c2, int act, float a0, float a1) {
  NPU_TID;
  const int64_t x1 = load_int(a, in_type, i);
  const int64_t x2 = load_int(b, in_type, bidx_flat(i, bc, d.w, d.c).flat());
  const float v = mac_exact(x2, c1, mac_exact(x1, c0, c2));
  store_int(out, out_type, i, activate_out(v, relu_variants(act), a0, a1, 0.0f, out_type));
}

// Mul: Horner form, both ifm2 terms before the ifm1 MAC.
__global__ void qdq_mul_kernel(int64_t n, const void* a, const void* b, npu_dtype in_type,
                               void* out, npu_dtype out_type, Dims d, int bc, float c0,
                               float c1, float c2, float c3, int act, float a0, float a1) {
  NPU_TID;
  const int64_t x1 = load_int(a, in_type, i);
  const int64_t x2 = load_int(b, in_type, bidx_flat(i, bc, d.w, d.c).flat());
  const float t1 = mac_exact(x2, c2, c3);
  const float t2 = mac_exact(x2, c0, c1);
  const float v = mac_exact(x1, t2, t1);
  store_int(out, out_type, i, activate_out(v, relu_variants(act), a0, a1, 0.0f, out_type));
}

// The C-broadcast route folds the scalar ifm2 into an intercept first; the other starts from
// ifm1, so the two round differently.
__global__ void bc3d_addsub_fused_kernel(int64_t n, const void* a, const void* b, npu_dtype t,
                                         void* out, Dims d, int bc,
                                         npu_bc3d_addsub_fused_coeff k, int act,
                                         const float* slope) {
  NPU_TID;
  const bool bcc = bc & 1;
  const int64_t x1 = load_int(a, t, i);
  const int64_t x2 = load_int(b, t, bidx_flat(i, bc, d.w, d.c).flat());
  const float icpt = bcc ? mac_exact(x2, k.c1, k.c2) : 0.0f;
  const float r = bcc ? mac_exact(x1, k.c0, icpt) : mac_exact(x2, k.c1, mac_exact(x1, k.c0, k.c2));
  float v = r;
  switch (act) {
    case kActRelu:
      v = vmax(r, k.activation_coeff0);
      break;
    case kActRelu6:
      v = vclamp(r, k.activation_coeff0, k.activation_coeff1);
      break;
    case kActLrelu: {
      // bcc: leaky intercept fma(icpt, alpha, beta); else alpha is folded into the coefficients.
      const float l =
          bcc ? mac_exact(x1, k.lrelu_c0, fma_q(icpt, k.activation_coeff0, k.activation_coeff1))
              : mac_exact(x2, k.lrelu_c1, mac_exact(x1, k.lrelu_c0, k.lrelu_bias));
      v = vmax(r, l);
      break;
    }
    case kActPrelu:
      // r = A(c2 - zp_out); slope selected on its sign, then the zero point is added back.
      v = fma_q(r, r < 0.0f ? slope[i % d.c] : 1.0f, k.activation_coeff0);
      break;
    default:
      break;
  }
  store_int(out, t, i, float_to_int(v, t, 0, kHalfEven));
}

// Each route keeps its own form (Horner, zero-point shifted, or the integer product of LRELU a8).
__global__ void bc3d_mul_fused_kernel(int64_t n, const void* a, const void* b, npu_dtype t,
                                      void* out, Dims d, int bc, npu_bc3d_mul_fused_coeff k,
                                      int act, bool a16, const float* neg_scale) {
  NPU_TID;
  const bool bcc = bc & 1;
  const int64_t x1 = load_int(a, t, i);
  const int64_t x2 = load_int(b, t, bidx_flat(i, bc, d.w, d.c).flat());
  const int64_t r1 = x1 - k.ifm1_zero_point, r2 = x2 - k.ifm2_zero_point;
  const float zy = k.ofm_zero_point;
  // A zero residual counts as positive.
  const bool neg = (r1 < 0) != (r2 < 0);
  const float F = mac_exact(x2, k.c0, k.c1);
  auto horner = [&] { return mac_exact(x1, F, mac_exact(x2, k.c2, k.c3)); };
  auto shifted = [&] { return mac_exact(r1, F, zy); };
  float v;
  switch (act) {
    case kActRelu:
      v = vmax(bcc ? horner() : shifted(), k.activation_coeff0);
      break;
    case kActRelu6:
      v = vclamp(horner(), k.activation_coeff0, k.activation_coeff1);
      break;
    case kActLrelu:
      if (!a16) {
        // a8: p = r1 r2 is formed exactly.
        const int64_t p = r1 * r2;
        v = bcc ? mac_exact(p, neg ? k.c1 : k.c0, zy)
                : mac_exact(p < 0 ? -p : p, k.c3, mac_exact(p, k.c2, zy));
      } else if (bcc) {
        const float leaky_f = mul_q(F, k.activation_coeff0);  // one rounding
        v = vmax(shifted(), mac_exact(r1, leaky_f, zy));
      } else {
        v = activate_f32(horner(), kQdqActLrelu, k.activation_coeff0, k.activation_coeff1);
      }
      break;
    case kActPrelu: {
      const float scale = neg ? neg_scale[i % d.c] : k.c0;
      v = mac_exact(r1, mac_exact(r2, scale, 0.0f), zy);
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
                                      Dims d, int bc, npu_bc3d_mac_fused_coeff k, int act,
                                      bool addend_to_float) {
  NPU_TID;
  const bool bcc = bc & 1;
  const int64_t j = bidx_flat(i, bc, d.w, d.c).flat();
  const int64_t x1 = load_int(a, ta, i);
  const int64_t x2 = load_int(b, tb, j);
  const int64_t x3 = load_int(c, tc, j);
  float v;
  if (bcc && act == kActNone) {
    // Expanded form: zp1 folded into the constant with one multiply-subtract.
    const float c3e = msc_q(float(k.ifm1_zero_point), k.c1, k.c3);
    const float t1 = mac_exact(x2, k.c2, c3e);
    const float t2 = mac_exact(x2, k.c0, k.c1);
    v = mac_exact(x1, t2, mac_exact(x3, k.c4, t1));
  } else {
    const float f = mac_exact(x2, k.c0, k.c1);
    // addend_to_float: the scalar addend is rounded to fp32 before the MAC.
    const float w = bcc && addend_to_float ? fma_q(float(x3), k.c4, k.c3)
                                           : mac_exact(x3, k.c4, k.c3);
    v = mac_exact(x1 - k.ifm1_zero_point, f, w);
    if (act == kActLrelu)
      v = activate_f32(v, kQdqActLrelu, k.activation_coeff0, k.activation_coeff1);
  }
  int64_t q = float_to_int(v, ta, 0, kHalfEven);
  // relu / relu6 clamp the quantized output against the quantized bounds.
  if (act == kActRelu || act == kActRelu6) {
    const int64_t lo = float_to_int(k.activation_coeff0, ta, 0, kHalfEven);
    q = q < lo ? lo : q;
    if (act == kActRelu6) {
      const int64_t hi = float_to_int(k.activation_coeff1, ta, 0, kHalfEven);
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
  int64_t q = srs(v, os, NPU_I8);
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
                                npu_dtype to, Dims d, int bc2, int bc3, int32_t A, int32_t v,
                                int32_t p, int32_t t) {
  NPU_TID;
  const int64_t x1 = load_int(a, ta, i);
  const int64_t x2 = load_int(b, tb, bidx_flat(i, bc2, d.w, d.c).flat());
  const int64_t x3 = load_int(c, tc, bidx_flat(i, bc3, d.w, d.c).flat());
  // w saturates to int16; with a scalar ifm2 it wraps.
  const int64_t w = bc2 == 7 ? wrap(shift_round(x2 * A, t, kHalfEven), NPU_I16)
                             : srs(x2 * A, t, NPU_I16);
  store_int(out, to, i, srs(x1 * w + x3 * v, p, to));
}

// The flat add / sub (fused) entries: activation 0 none, 1 relu, 2 relu6, 3 leaky relu.
int flat_add(void* stream, const void* a, npu_dtype in_type, const void* b, void* out,
             npu_dtype out_type, int64_t n, int bits, float c0, float c1, float c2, int act,
             float a0, float a1) {
  if (any_null(a, b, out) || !is_int_bits(in_type, bits) || !is_int_bits(out_type, bits) || n < 0 ||
      act < kActNone || act > kActLrelu)
    return -1;
  return launch(stream, n, qdq_add_kernel, a, b, in_type, out, out_type, Dims{1, n}, 0, c0, c1, c2,
                act, a0, a1);
}

int flat_mul(void* stream, const void* a, npu_dtype in_type, const void* b, void* out,
             npu_dtype out_type, int64_t n, int bits, float c0, float c1, float c2, float c3,
             int act, float a0, float a1) {
  if (any_null(a, b, out) || !is_int_bits(in_type, bits) || !is_int_bits(out_type, bits) || n < 0 ||
      act < kActNone || act > kActLrelu)
    return -1;
  return launch(stream, n, qdq_mul_kernel, a, b, in_type, out, out_type, Dims{1, n}, 0, c0, c1, c2,
                c3, act, a0, a1);
}

int bc3d_binary(void* stream, const void* a, npu_dtype dtype, const void* b, void* out,
                const int64_t* s1, const int64_t* s2, int bits, bool mul, float c0, float c1,
                float c2, float c3) {
  const int bc = bcast_check(s1, s2);
  if (any_null(a, b, out) || !is_int_bits(dtype, bits) || bc < 0) return -1;
  const int64_t n = s1[0] * s1[1] * s1[2];
  const Dims d{s1[1], s1[2]};
  if (mul)
    return launch(stream, n, qdq_mul_kernel, a, b, dtype, out, dtype, d, bc, c0, c1, c2, c3,
                  kActNone, 0.0f, 0.0f);
  return launch(stream, n, qdq_add_kernel, a, b, dtype, out, dtype, d, bc, c0, c1, c2, kActNone,
                0.0f, 0.0f);
}

// The bc3d fused add / sub / mul: 0 none, 1 relu, 2 relu6, 3 leaky relu, 9 prelu (with its
// per-channel array, which is null otherwise).
bool fused_act_ok(int act, const void* prelu_array) {
  return ((act >= kActNone && act <= kActLrelu) || act == kActPrelu) &&
         (act == kActPrelu) == (prelu_array != nullptr);
}

int bc3d_addsub_fused(void* stream, const void* a, npu_dtype dtype, const void* b, void* out,
                      const int64_t* s1, const int64_t* s2, int bits,
                      const npu_bc3d_addsub_fused_coeff* k, int act, const void* slope) {
  const int bc = bcast_check(s1, s2);
  if (any_null(a, b, out, k) || !is_int_bits(dtype, bits) || !fused_act_ok(act, slope) || bc < 0)
    return -1;
  return launch(stream, s1[0] * s1[1] * s1[2], bc3d_addsub_fused_kernel, a, b, dtype, out,
                Dims{s1[1], s1[2]}, bc, *k, act, static_cast<const float*>(slope));
}

int bc3d_mul_fused(void* stream, const void* a, npu_dtype dtype, const void* b, void* out,
                   const int64_t* s1, const int64_t* s2, int bits,
                   const npu_bc3d_mul_fused_coeff* k, int act, const void* neg_scale) {
  const int bc = bcast_check(s1, s2);
  if (any_null(a, b, out, k) || !is_int_bits(dtype, bits) || !fused_act_ok(act, neg_scale) ||
      bc < 0)
    return -1;
  return launch(stream, s1[0] * s1[1] * s1[2], bc3d_mul_fused_kernel, a, b, dtype, out,
                Dims{s1[1], s1[2]}, bc, *k, act, bits == 16,
                static_cast<const float*>(neg_scale));
}

int bc3d_mac_fused(void* stream, const void* a, npu_dtype ifm1_type, const void* b,
                   npu_dtype ifm2_type, const void* c, npu_dtype ifm3_type, void* out,
                   const int64_t* s1, const int64_t* s23, int a_bits, int w_bits,
                   bool addend_to_float, const npu_bc3d_mac_fused_coeff* k, int act) {
  const int bc = bcast_check(s1, s23);
  const bool c_ok = is_i8(ifm3_type) ? a_bits == 8 : (is_i16(ifm3_type) || ifm3_type == NPU_I32);
  if (any_null(a, b, c, out, k) || !is_int_bits(ifm1_type, a_bits) ||
      !is_int_bits(ifm2_type, w_bits) || !c_ok || act < kActNone || act > kActLrelu || bc < 0)
    return -1;
  return launch(stream, s1[0] * s1[1] * s1[2], bc3d_mac_fused_kernel, a, ifm1_type, b, ifm2_type,
                c, ifm3_type, out, Dims{s1[1], s1[2]}, bc, *k, act, addend_to_float);
}

// Exponent of a positive power of two, or -1000.
int pow2_exp(float f) {
  const uint32_t b = f32_bits(f);
  const int e = (b >> 23) & 0xff;
  if ((b >> 31) || (b & 0x7fffff) || e == 0 || e == 0xff) return -1000;
  return e - 127;
}

}  // namespace

extern "C" {

int npu_elementwise_add_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float c2) {
  return flat_add(stream, a, in_type, b, out, out_type, n, 8, c0, c1, c2, kActNone, 0.0f, 0.0f);
}

int npu_elementwise_add_a16(void* stream, const void* a, npu_dtype in_type, const void* b,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float c2) {
  return flat_add(stream, a, in_type, b, out, out_type, n, 16, c0, c1, c2, kActNone, 0.0f, 0.0f);
}

int npu_elementwise_mul_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float c2, float c3) {
  return flat_mul(stream, a, in_type, b, out, out_type, n, 8, c0, c1, c2, c3, kActNone, 0.0f,
                  0.0f);
}

int npu_elementwise_mul_a16(void* stream, const void* a, npu_dtype in_type, const void* b,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float c2, float c3) {
  return flat_mul(stream, a, in_type, b, out, out_type, n, 16, c0, c1, c2, c3, kActNone, 0.0f,
                  0.0f);
}

int npu_elementwise_add_fused_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                                 void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                                 float c2, int activation, float c3, float c4) {
  return flat_add(stream, a, in_type, b, out, out_type, n, 8, c0, c1, c2, activation, c3, c4);
}

int npu_elementwise_add_fused_a16(void* stream, const void* a, npu_dtype in_type,
                                  const void* b, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2, int activation, float c3,
                                  float c4) {
  return flat_add(stream, a, in_type, b, out, out_type, n, 16, c0, c1, c2, activation, c3, c4);
}

int npu_elementwise_sub_fused_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                                 void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                                 float c2, int activation, float c3, float c4) {
  return flat_add(stream, a, in_type, b, out, out_type, n, 8, c0, c1, c2, activation, c3, c4);
}

int npu_elementwise_sub_fused_a16(void* stream, const void* a, npu_dtype in_type,
                                  const void* b, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2, int activation, float c3,
                                  float c4) {
  return flat_add(stream, a, in_type, b, out, out_type, n, 16, c0, c1, c2, activation, c3, c4);
}

int npu_elementwise_mul_fused_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                                 void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                                 float c2, float c3, int activation, float c4, float c5) {
  return flat_mul(stream, a, in_type, b, out, out_type, n, 8, c0, c1, c2, c3, activation, c4,
                  c5);
}

int npu_elementwise_mul_fused_a16(void* stream, const void* a, npu_dtype in_type,
                                  const void* b, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2, float c3, int activation,
                                  float c4, float c5) {
  return flat_mul(stream, a, in_type, b, out, out_type, n, 16, c0, c1, c2, c3, activation, c4,
                  c5);
}

int npu_elementwise_add_a8_xint(void* stream, const void* a, const void* b, void* out,
                                int64_t n, float c0, float c1, float c2) {
  return flat_add(stream, a, NPU_I8, b, out, NPU_I8, n, 8, c0, c1, c2, kActNone, 0.0f, 0.0f);
}

int npu_elementwise_add_fused_a8_xint(void* stream, const void* a, const void* b, void* out,
                                      int64_t n, float c0, float c1, int activation, float c3,
                                      float c4) {
  const int e0 = pow2_exp(c0), e1 = pow2_exp(c1);
  if (any_null(a, b, out) || n < 0 || e0 == -1000 || e1 == -1000 || activation < kActNone ||
      activation > kActRelu6)
    return -1;
  const int lower = e0 < e1 ? e0 : e1;
  const int base = lower < 0 ? lower : 0;
  // relu and relu6 both clamp to [c3, c4].
  return launch(stream, n, xint_add_fused_kernel, static_cast<const int8_t*>(a),
                static_cast<const int8_t*>(b), static_cast<int8_t*>(out), e0 - base, e1 - base,
                -base, activation != kActNone, float_to_int(c3, NPU_I8, 0, kHalfEven),
                float_to_int(c4, NPU_I8, 0, kHalfEven));
}

int npu_elementwise_mac_a8(void* stream, const void* ifm1, npu_dtype ifm1_type,
                           const void* ifm2, npu_dtype ifm2_type, const void* ifm3,
                           npu_dtype ifm3_type, void* out, npu_dtype out_type,
                           const int64_t* shape1, const int64_t* shape2,
                           const int64_t* shape3, int32_t fx_A, int32_t fx_v, int32_t fx_p,
                           int32_t fx_t) {
  const int bc2 = bcast_check(shape1, shape2), bc3 = bcast_check(shape1, shape3);
  if (any_null(ifm1, ifm2, ifm3, out) || !is_i8(ifm1_type) || !is_i8(ifm2_type) ||
      !is_i8(ifm3_type) || !is_i8(out_type) || bc2 < 0 || bc3 < 0 || fx_p < 0 || fx_p > 31 ||
      fx_t < 0 || fx_t > 31)
    return -1;
  return launch(stream, shape1[0] * shape1[1] * shape1[2], xint_mac_kernel, ifm1, ifm1_type, ifm2,
                ifm2_type, ifm3, ifm3_type, out, out_type, Dims{shape1[1], shape1[2]}, bc2, bc3,
                fx_A, fx_v, fx_p, fx_t);
}

int npu_qdq_elementwise_broadcasting3d_add_a8(void* stream, const void* ifm1, npu_dtype dtype,
                                              const void* ifm2, void* out,
                                              const int64_t* shape1, const int64_t* shape2,
                                              float c0, float c1, float c2) {
  return bc3d_binary(stream, ifm1, dtype, ifm2, out, shape1, shape2, 8, false, c0, c1, c2, 0.0f);
}

int npu_qdq_elementwise_broadcasting3d_add_a16(void* stream, const void* ifm1, npu_dtype dtype,
                                               const void* ifm2, void* out,
                                               const int64_t* shape1, const int64_t* shape2,
                                               float c0, float c1, float c2) {
  return bc3d_binary(stream, ifm1, dtype, ifm2, out, shape1, shape2, 16, false, c0, c1, c2,
                     0.0f);
}

int npu_qdq_elementwise_broadcasting3d_mul_a8(void* stream, const void* ifm1, npu_dtype dtype,
                                              const void* ifm2, void* out,
                                              const int64_t* shape1, const int64_t* shape2,
                                              float c0, float c1, float c2, float c3) {
  return bc3d_binary(stream, ifm1, dtype, ifm2, out, shape1, shape2, 8, true, c0, c1, c2, c3);
}

int npu_qdq_elementwise_broadcasting3d_mul_a16(void* stream, const void* ifm1, npu_dtype dtype,
                                               const void* ifm2, void* out,
                                               const int64_t* shape1, const int64_t* shape2,
                                               float c0, float c1, float c2, float c3) {
  return bc3d_binary(stream, ifm1, dtype, ifm2, out, shape1, shape2, 16, true, c0, c1, c2, c3);
}

int npu_qdq_elementwise_broadcasting3d_add_fused_a8(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    int activation, const void* slope) {
  return bc3d_addsub_fused(stream, ifm1, dtype, ifm2, out, shape1, shape2, 8, coeff, activation,
                           slope);
}

int npu_qdq_elementwise_broadcasting3d_add_fused_a16(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    int activation, const void* slope) {
  return bc3d_addsub_fused(stream, ifm1, dtype, ifm2, out, shape1, shape2, 16, coeff,
                           activation, slope);
}

int npu_qdq_elementwise_broadcasting3d_sub_fused_a8(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    int activation, const void* slope) {
  return bc3d_addsub_fused(stream, ifm1, dtype, ifm2, out, shape1, shape2, 8, coeff, activation,
                           slope);
}

int npu_qdq_elementwise_broadcasting3d_sub_fused_a16(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    int activation, const void* slope) {
  return bc3d_addsub_fused(stream, ifm1, dtype, ifm2, out, shape1, shape2, 16, coeff,
                           activation, slope);
}

int npu_qdq_elementwise_broadcasting3d_mul_fused_a8(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_mul_fused_coeff* coeff,
    int activation, const void* neg_scale) {
  return bc3d_mul_fused(stream, ifm1, dtype, ifm2, out, shape1, shape2, 8, coeff, activation,
                        neg_scale);
}

int npu_qdq_elementwise_broadcasting3d_mul_fused_a16(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_mul_fused_coeff* coeff,
    int activation, const void* neg_scale) {
  return bc3d_mul_fused(stream, ifm1, dtype, ifm2, out, shape1, shape2, 16, coeff, activation,
                        neg_scale);
}

int npu_qdq_elementwise_broadcasting3d_mac_fused_a8w8(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape23, const npu_bc3d_mac_fused_coeff* coeff,
    int activation) {
  return bc3d_mac_fused(stream, ifm1, ifm1_type, ifm2, ifm2_type, ifm3, ifm3_type, out, shape1,
                        shape23, 8, 8, true, coeff, activation);
}

int npu_qdq_elementwise_broadcasting3d_mac_fused_a16w8(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape23, const npu_bc3d_mac_fused_coeff* coeff,
    int activation) {
  return bc3d_mac_fused(stream, ifm1, ifm1_type, ifm2, ifm2_type, ifm3, ifm3_type, out, shape1,
                        shape23, 16, 8, false, coeff, activation);
}

int npu_qdq_elementwise_broadcasting3d_mac_fused_a16w16(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape23, const npu_bc3d_mac_fused_coeff* coeff,
    int activation) {
  return bc3d_mac_fused(stream, ifm1, ifm1_type, ifm2, ifm2_type, ifm3, ifm3_type, out, shape1,
                        shape23, 16, 16, true, coeff, activation);
}

}  // extern "C"
