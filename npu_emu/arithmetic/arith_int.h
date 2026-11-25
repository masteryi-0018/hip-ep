/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Integer / QDQ elementwise arithmetic: add, sub, mul, their fused-activation variants, the
 * 3D-broadcasting add / sub / mul / mac family and the xint8 add / mac kernels.
 */
#ifndef NPU_EMU_ARITH_INT_H_
#define NPU_EMU_ARITH_INT_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Common rules. mac(x, m, a) = x * m + a rounded once to fp32 (half to even); an integer x
 * enters exactly. The order of the mac steps is part of the result. sat(rne(v)) rounds half to
 * even and saturates to the output type.
 * Coefficients are fp32. With s / zp the scales and zero points of the inputs and the output:
 *   add: c0 = s1/s_out, c1 = s2/s_out, c2 = zp_out - zp1*s1/s_out - zp2*s2/s_out
 *   sub: c0 = s1/s_out, c1 = -s2/s_out, c2 = zp_out - zp1*s1/s_out + zp2*s2/s_out
 *   mul: c0 = s1*s2/s_out, c1 = -zp2*c0, c2 = -zp1*c0, c3 = zp1*zp2*c0 + zp_out
 * (products with c0 use its unrounded double value). Sub uses the add entries.
 * Activation bounds and slopes are in the output's quantized domain.
 * Flat entries take n >= 0 elements. Broadcasting entries take HWC tensors: shape1 = {H, W, C}
 * (host array of 3 int64, every dimension >= 0); every dimension of shape2 / shape3 / shape23
 * is 1 or equal to shape1's. An empty tensor is a no-op. Pointers may not be null unless
 * noted. */

/* flat add / sub / mul */

/* out = sat(rne(mac(b, c1, mac(a, c0, c2)))). in_type (both inputs): I8 / U8 (a8) or
 * I16 / U16 (a16); out_type has the same width, either signedness. */
int npu_elementwise_add_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float c2);
/* The 16-bit variant of npu_elementwise_add_a8. */
int npu_elementwise_add_a16(void* stream, const void* a, npu_dtype in_type, const void* b,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float c2);

/* t1 = mac(b, c2, c3); t2 = mac(b, c0, c1); out = sat(rne(mac(a, t2, t1))). Types as for add. */
int npu_elementwise_mul_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float c2, float c3);
/* The 16-bit variant of npu_elementwise_mul_a8. */
int npu_elementwise_mul_a16(void* stream, const void* a, npu_dtype in_type, const void* b,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float c2, float c3);

/* v as for add, then the activation, then sat(rne(.)). activation: 0 none; 1 relu and 2 relu6:
 * clamp(v, c3, c4); 3 leaky relu: max(v, fma(v, c3, c4)) with c3 = alpha,
 * c4 = (1 - alpha) * zp_out. */
int npu_elementwise_add_fused_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                                 void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                                 float c2, int activation, float c3, float c4);
/* The 16-bit variant of npu_elementwise_add_fused_a8. */
int npu_elementwise_add_fused_a16(void* stream, const void* a, npu_dtype in_type,
                                  const void* b, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2, int activation, float c3,
                                  float c4);
/* See npu_elementwise_add_fused_a8. */
int npu_elementwise_sub_fused_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                                 void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                                 float c2, int activation, float c3, float c4);
/* The 16-bit variant of npu_elementwise_sub_fused_a8. */
int npu_elementwise_sub_fused_a16(void* stream, const void* a, npu_dtype in_type,
                                  const void* b, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2, int activation, float c3,
                                  float c4);

/* v as for mul, activation as for add_fused with c4, c5 in place of c3, c4. */
int npu_elementwise_mul_fused_a8(void* stream, const void* a, npu_dtype in_type, const void* b,
                                 void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                                 float c2, float c3, int activation, float c4, float c5);
/* The 16-bit variant of npu_elementwise_mul_fused_a8. */
int npu_elementwise_mul_fused_a16(void* stream, const void* a, npu_dtype in_type,
                                  const void* b, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2, float c3, int activation,
                                  float c4, float c5);

/* xint8 */

/* I8 only; out = sat(rne(mac(b, c1, mac(a, c0, c2)))). */
int npu_elementwise_add_a8_xint(void* stream, const void* a, const void* b, void* out,
                                int64_t n, float c0, float c1, float c2);

/* I8 only, integer arithmetic. c0 and c1 must be powers of two 2^e0, 2^e1 (else -1). With
 * b0 = min(e0, e1, 0):
 *   v = (a << (e0 - b0)) + (b << (e1 - b0));  out = sat8(rne(v / 2^-b0))
 * activation 0 none, or 1 relu / 2 relu6: clamp to the bounds sat8(rne(c3)), sat8(rne(c4)) on
 * the int8 result. */
int npu_elementwise_add_fused_a8_xint(void* stream, const void* a, const void* b, void* out,
                                      int64_t n, float c0, float c1, int activation, float c3,
                                      float c4);

/* out = s_mul * ifm1 * ifm2 + s_add * ifm3 (zero points 0) in fixed point, with
 * fx_A = rne(s_mul * 2^(fx_p + fx_t)), fx_v = rne(s_add * 2^fx_p):
 *   w = srs16(ifm2 * fx_A, fx_t);  out = sat(rne((ifm1 * w + ifm3 * fx_v) / 2^fx_p))
 * srs16 rounds half to even and saturates to int16; it wraps instead when shape2 is {1, 1, 1}.
 * ifm2 / ifm3 broadcast over ifm1. Each input I8 / U8 (independent), out I8 / U8;
 * fx_p, fx_t in [0, 31]. */
int npu_elementwise_mac_a8(void* stream, const void* ifm1, npu_dtype ifm1_type,
                           const void* ifm2, npu_dtype ifm2_type, const void* ifm3,
                           npu_dtype ifm3_type, void* out, npu_dtype out_type,
                           const int64_t* shape1, const int64_t* shape2,
                           const int64_t* shape3, int32_t fx_A, int32_t fx_v, int32_t fx_p,
                           int32_t fx_t);

/* 3D-broadcasting add / sub / mul */

/* out = sat(rne(mac(ifm2, c1, mac(ifm1, c0, c2)))), ifm2 broadcast. dtype: ifm1, ifm2 and the
 * output. */
int npu_qdq_elementwise_broadcasting3d_add_a8(void* stream, const void* ifm1, npu_dtype dtype,
                                              const void* ifm2, void* out,
                                              const int64_t* shape1, const int64_t* shape2,
                                              float c0, float c1, float c2);
/* The 16-bit variant of npu_qdq_elementwise_broadcasting3d_add_a8. */
int npu_qdq_elementwise_broadcasting3d_add_a16(void* stream, const void* ifm1, npu_dtype dtype,
                                               const void* ifm2, void* out,
                                               const int64_t* shape1, const int64_t* shape2,
                                               float c0, float c1, float c2);

/* As npu_elementwise_mul_a8 with ifm2 broadcast; dtype as for add. */
int npu_qdq_elementwise_broadcasting3d_mul_a8(void* stream, const void* ifm1, npu_dtype dtype,
                                              const void* ifm2, void* out,
                                              const int64_t* shape1, const int64_t* shape2,
                                              float c0, float c1, float c2, float c3);
int npu_qdq_elementwise_broadcasting3d_mul_a16(void* stream, const void* ifm1, npu_dtype dtype,
                                               const void* ifm2, void* out,
                                               const int64_t* shape1, const int64_t* shape2,
                                               float c0, float c1, float c2, float c3);

/* 3D-broadcasting fused add / sub / mul. activation: 0 none, 1 relu, 2 relu6, 3 leaky relu,
 * 9 prelu. */

/* Coefficients of the fused add / sub (host pointer), the kernel's coefficient words. The
 * activation words depend on the activation:
 *   1: activation_coeff0 = lower bound;  2: activation_coeff0, activation_coeff1 = bounds;
 *   3: activation_coeff0 = alpha, activation_coeff1 = beta = (1 - alpha) zp_out,
 *      lrelu_c0 = fp32(alpha c0), lrelu_c1 = fp32(alpha c1), lrelu_bias = fp32(alpha c2 + beta);
 *   9: c2 holds fp32(c2 - zp_out) and activation_coeff0 = zp_out. */
typedef struct {
  float c0, c1, c2, activation_coeff0, activation_coeff1, lrelu_c0, lrelu_c1, lrelu_bias;
} npu_bc3d_addsub_fused_coeff;

/* dtype: ifm1, ifm2 and the output. With bcc = (ifm2 C == 1), a0 / a1 = activation_coeff0 / 1:
 *   bcc: A(bias) = mac(ifm1, c0, mac(ifm2, c1, bias))
 *   else A(bias) = mac(ifm2, c1, mac(ifm1, c0, bias))
 * 0: A(c2);  1: max(A(c2), a0);  2: clamp(A(c2), a0, a1);
 * 3: r = A(c2), max(r, l) with
 *   bcc: l = mac(ifm1, lrelu_c0, fma(mac(ifm2, c1, c2), a0, a1))
 *   else l = mac(ifm2, lrelu_c1, mac(ifm1, lrelu_c0, lrelu_bias))
 * 9: d = A(c2); out = fma(d, d < 0 ? slope[c] : 1, a0); slope = device array of C floats,
 * required for 9, else null. */
int npu_qdq_elementwise_broadcasting3d_add_fused_a8(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    int activation, const void* slope);
int npu_qdq_elementwise_broadcasting3d_add_fused_a16(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    int activation, const void* slope);
int npu_qdq_elementwise_broadcasting3d_sub_fused_a8(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    int activation, const void* slope);
int npu_qdq_elementwise_broadcasting3d_sub_fused_a16(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    int activation, const void* slope);

/* Coefficients of the fused mul (host pointer), the kernel's coefficient words: c0..c3 as for
 * mul; activation_coeff0 / 1: relu lower bound, relu6 bounds, a16 leaky relu alpha and
 * (1 - alpha) zp_out. a8 leaky relu holds fp32(alpha c0), fp32(c0 (1+alpha)/2) and
 * fp32(c0 (1-alpha)/2) in c1, c2, c3. */
typedef struct {
  float c0, c1, c2, c3, activation_coeff0, activation_coeff1;
  int32_t ifm1_zero_point;
  float ofm_zero_point;
  int32_t ifm2_zero_point;
} npu_bc3d_mul_fused_coeff;

/* With bcc = (ifm2 C == 1), z1 / z2 / zy = ifm1 / ifm2 / ofm zero points, a0 / a1 =
 * activation_coeff0 / 1, r1 = ifm1 - z1, r2 = ifm2 - z2, F = mac(ifm2, c0, c1),
 * B = mac(ifm2, c2, c3), H = mac(ifm1, F, B), S = mac(r1, F, zy):
 *   0: S;  1: max(bcc ? H : S, a0);  2: clamp(H, a0, a1)
 *   3 a8 (p = r1 r2): bcc: mac(p, (r1 < 0) != (r2 < 0) ? c1 : c0, zy)
 *                     else mac(|p|, c3, mac(p, c2, zy))
 *   3 a16: bcc: max(S, mac(r1, fp32(F a0), zy))
 *          else max(H, fma(H, a0, a1))
 *   9: k = (r1 < 0) != (r2 < 0) ? neg_scale[c] : c0;
 *      out = mac(r1, fp32(r2 k), zy)
 * neg_scale: device array of C floats, fp32(slope[c] * c0); required for 9, else null. */
int npu_qdq_elementwise_broadcasting3d_mul_fused_a8(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_mul_fused_coeff* coeff,
    int activation, const void* neg_scale);
int npu_qdq_elementwise_broadcasting3d_mul_fused_a16(
    void* stream, const void* ifm1, npu_dtype dtype, const void* ifm2, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_mul_fused_coeff* coeff,
    int activation, const void* neg_scale);

/* 3D-broadcasting fused mac */

/* Coefficients of the fused mac (host pointer), the kernel's coefficient words:
 * c0 = s1 s2 / s_out, c1 = -zp2 c0, c2 = -zp1 c0, c4 = s3 / s_out, c3 = zp_out - zp3 c4;
 * activation_coeff0 / 1: relu / relu6 bounds, or leaky relu alpha and (1 - alpha) zp_out. */
typedef struct {
  float c0, c1, c2, c3, c4, activation_coeff0, activation_coeff1;
  int32_t ifm1_zero_point;
} npu_bc3d_mac_fused_coeff;

/* out = (ifm1 - zp1)(ifm2 - zp2) s1 s2 / s_out + (ifm3 - zp3) s3 / s_out + zp_out; ifm2 and
 * ifm3 share shape23 and are broadcast over ifm1. With bcc = (ifm2 C == 1),
 * zp1 = ifm1_zero_point, a0 / a1 = activation_coeff0 / 1:
 *   bcc and activation 0: c3e = fp32(c3 - zp1 c1); t1 = mac(ifm2, c2, c3e);
 *                         t2 = mac(ifm2, c0, c1); t3 = mac(ifm3, c4, t1);
 *                         v = mac(ifm1, t2, t3)
 *   otherwise:            f = mac(ifm2, c0, c1); w = mac(x3, c4, c3);
 *                         v = mac(ifm1 - zp1, f, w)
 * where x3 = ifm3, converted to fp32 first on the activated bcc routes of a8w8 and a16w16.
 * activation 0 none, 1 relu, 2 relu6, 3 leaky relu. 3: v = max(v, fma(v, a0, a1)).
 * q = sat(rne(v)); 1: q = max(q, sat(rne(a0))); 2: also q = min(q, sat(rne(a1))).
 * ifm1 and out: ifm1_type (I8 / U8 for a8w8, I16 / U16 else); ifm2 I8 / U8 (a8w8, a16w8) or
 * I16 / U16 (a16w16); ifm3 I8 / U8 / I16 / U16 / I32, at least as wide as ifm1. */
int npu_qdq_elementwise_broadcasting3d_mac_fused_a8w8(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape23, const npu_bc3d_mac_fused_coeff* coeff,
    int activation);
int npu_qdq_elementwise_broadcasting3d_mac_fused_a16w8(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape23, const npu_bc3d_mac_fused_coeff* coeff,
    int activation);
int npu_qdq_elementwise_broadcasting3d_mac_fused_a16w16(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape23, const npu_bc3d_mac_fused_coeff* coeff,
    int activation);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_ARITH_INT_H_
