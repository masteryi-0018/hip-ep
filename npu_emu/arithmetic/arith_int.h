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
 * Flat entries take n elements. Broadcasting entries take HWC tensors: shape1 = {H, W, C}
 * (host array of 3 int64); every dimension of shape2 / shape3 is 1 or equal to shape1's. */
typedef enum {
  NPU_ARITH_ACT_NONE = 0,
  NPU_ARITH_ACT_RELU = 1,  /* max(v, act0) */
  NPU_ARITH_ACT_CLIP = 2,  /* min(max(v, act0), act1) */
  NPU_ARITH_ACT_LRELU = 3,
  NPU_ARITH_ACT_PRELU = 4,
} npu_arith_activation;

/* flat add / sub / mul */

/* out = sat(rne(mac(b, c1, mac(a, c0, c2)))). in_type (both inputs): I8 / U8 (a8) or
 * I16 / U16 (a16); out_type has the same width, either signedness. */
int npu_elementwise_add_a8(void* stream, const void* a, const void* b, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float c2);
/* The 16-bit variant of npu_elementwise_add_a8. */
int npu_elementwise_add_a16(void* stream, const void* a, const void* b, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float c2);

/* t1 = mac(b, c2, c3); t2 = mac(b, c0, c1); out = sat(rne(mac(a, t2, t1))). Types as for add. */
int npu_elementwise_mul_a8(void* stream, const void* a, const void* b, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float c2, float c3);
/* The 16-bit variant of npu_elementwise_mul_a8. */
int npu_elementwise_mul_a16(void* stream, const void* a, const void* b, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float c2, float c3);

/* v as for add, then the activation, then sat(rne(.)). NONE; CLIP: clamp(v, act0, act1);
 * LRELU: max(v, fma(v, act0, act1)) with act0 = alpha, act1 = (1 - alpha) * zp_out.
 * RELU and PRELU return -1. */
int npu_elementwise_add_fused_a8(void* stream, const void* a, const void* b,
                                 npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                 float c0, float c1, float c2, npu_arith_activation activation,
                                 float act0, float act1);
/* The 16-bit variant of npu_elementwise_add_fused_a8. */
int npu_elementwise_add_fused_a16(void* stream, const void* a, const void* b,
                                  npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2,
                                  npu_arith_activation activation, float act0, float act1);
/* See npu_elementwise_add_fused_a8. */
int npu_elementwise_sub_fused_a8(void* stream, const void* a, const void* b,
                                 npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                 float c0, float c1, float c2, npu_arith_activation activation,
                                 float act0, float act1);
/* The 16-bit variant of npu_elementwise_sub_fused_a8. */
int npu_elementwise_sub_fused_a16(void* stream, const void* a, const void* b,
                                  npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2,
                                  npu_arith_activation activation, float act0, float act1);

/* v as for mul, activation as for add_fused. */
int npu_elementwise_mul_fused_a8(void* stream, const void* a, const void* b,
                                 npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                 float c0, float c1, float c2, float c3,
                                 npu_arith_activation activation, float act0, float act1);
/* The 16-bit variant of npu_elementwise_mul_fused_a8. */
int npu_elementwise_mul_fused_a16(void* stream, const void* a, const void* b,
                                  npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2, float c3,
                                  npu_arith_activation activation, float act0, float act1);

/* xint8 */

/* I8 only; out = sat(rne(mac(b, c1, mac(a, c0, c2)))). */
int npu_elementwise_add_a8_xint(void* stream, const void* a, const void* b, void* out,
                                int64_t n, float c0, float c1, float c2);

/* I8 only, integer arithmetic. c0 and c1 must be powers of two 2^e0, 2^e1 (else -1). With
 * b0 = min(e0, e1, 0):
 *   v = (a << (e0 - b0)) + (b << (e1 - b0));  out = sat8(rne(v / 2^-b0))
 * activation NONE, or CLIP with bounds sat8(rne(act0)), sat8(rne(act1)) on the int8 result. */
int npu_elementwise_add_fused_a8_xint(void* stream, const void* a, const void* b, void* out,
                                      int64_t n, float c0, float c1,
                                      npu_arith_activation activation, float act0, float act1);

/* out = s_mul * ifm1 * ifm2 + s_add * ifm3 (zero points 0) in fixed point, with
 * A = rne(s_mul * 2^(p + t)), v = rne(s_add * 2^p):
 *   w = srs16(ifm2 * A, t);  out = sat(rne((ifm1 * w + ifm3 * v) / 2^p))
 * srs16 rounds half to even and saturates to int16; it wraps instead when shape2 is {1, 1, 1}.
 * ifm2 / ifm3 broadcast over ifm1. Each input I8 / U8 (independent), out I8 / U8. */
int npu_elementwise_mac_a8(void* stream, const void* ifm1, npu_dtype ifm1_type,
                           const void* ifm2, npu_dtype ifm2_type, const void* ifm3,
                           npu_dtype ifm3_type, void* out, npu_dtype out_type,
                           const int64_t* shape1, const int64_t* shape2,
                           const int64_t* shape3, int32_t A, int32_t v, int32_t p, int32_t t);

/* 3D-broadcasting add / sub / mul */

/* out = sat(rne(mac(ifm2, c1, mac(ifm1, c0, c2)))), ifm2 broadcast. in_type for both inputs;
 * out_type of the same width. */
int npu_qdq_elementwise_broadcasting3d_add_a8(void* stream, const void* ifm1, const void* ifm2,
                                              npu_dtype in_type, void* out, npu_dtype out_type,
                                              const int64_t* shape1, const int64_t* shape2,
                                              float c0, float c1, float c2);
/* The 16-bit variant of npu_qdq_elementwise_broadcasting3d_add_a8. */
int npu_qdq_elementwise_broadcasting3d_add_a16(void* stream, const void* ifm1,
                                               const void* ifm2, npu_dtype in_type, void* out,
                                               npu_dtype out_type, const int64_t* shape1,
                                               const int64_t* shape2, float c0, float c1,
                                               float c2);

/* As npu_elementwise_mul_a8 with ifm2 broadcast. */
int npu_qdq_elementwise_broadcasting3d_mul_a8(void* stream, const void* ifm1, const void* ifm2,
                                              npu_dtype in_type, void* out, npu_dtype out_type,
                                              const int64_t* shape1, const int64_t* shape2,
                                              float c0, float c1, float c2, float c3);
int npu_qdq_elementwise_broadcasting3d_mul_a16(void* stream, const void* ifm1,
                                               const void* ifm2, npu_dtype in_type, void* out,
                                               npu_dtype out_type, const int64_t* shape1,
                                               const int64_t* shape2, float c0, float c1,
                                               float c2, float c3);

/* 3D-broadcasting fused add / sub / mul */

/* Coefficients of the fused add / sub (host pointer). w3..w7 depend on the activation:
 *   RELU: w3 = lower bound;  CLIP: w3, w4 = bounds;
 *   LRELU: w3 = alpha, w4 = beta = (1 - alpha) zp_out, w5 = fp32(alpha c0),
 *          w6 = fp32(alpha c1), w7 = fp32(alpha c2 + beta);
 *   PRELU: c2 holds fp32(c2 - zp_out) and w3 = zp_out. */
typedef struct {
  float c0, c1, c2, w3, w4, w5, w6, w7;
} npu_bc3d_addsub_fused_coeff;

/* in_type is the type of ifm1, ifm2 and the output. With bcc = (ifm2 C == 1):
 *   bcc: A(bias) = mac(ifm1, c0, mac(ifm2, c1, bias))
 *   else A(bias) = mac(ifm2, c1, mac(ifm1, c0, bias))
 * NONE: A(c2);  RELU: max(A(c2), w3);  CLIP: clamp(A(c2), w3, w4);
 * LRELU: r = A(c2), max(r, l) with
 *   bcc: l = mac(ifm1, w5, fma(mac(ifm2, c1, c2), w3, w4))
 *   else l = mac(ifm2, w6, mac(ifm1, w5, w7))
 * PRELU: d = A(c2); out = fma(d, d < 0 ? slope[c] : 1, w3); slope = device array of C floats,
 * null unless PRELU. */
int npu_qdq_elementwise_broadcasting3d_add_fused_a8(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    npu_arith_activation activation, const float* slope);
int npu_qdq_elementwise_broadcasting3d_add_fused_a16(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    npu_arith_activation activation, const float* slope);
int npu_qdq_elementwise_broadcasting3d_sub_fused_a8(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    npu_arith_activation activation, const float* slope);
int npu_qdq_elementwise_broadcasting3d_sub_fused_a16(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_addsub_fused_coeff* coeff,
    npu_arith_activation activation, const float* slope);

/* Coefficients of the fused mul (host pointer): c0..c3 as for mul; act0 / act1: RELU lower
 * bound, CLIP bounds, LRELU a16 alpha and (1 - alpha) zp_out; z1 = zp1, z2 = zp2, zy = zp_out.
 * LRELU a8 uses c1 = fp32(alpha c0), c2 = fp32(c0 (1+alpha)/2), c3 = fp32(c0 (1-alpha)/2). */
typedef struct {
  float c0, c1, c2, c3, act0, act1;
  int32_t z1;
  float zy;
  int32_t z2;
} npu_bc3d_mul_fused_coeff;

/* With bcc = (ifm2 C == 1), r1 = ifm1 - z1, r2 = ifm2 - z2, F = mac(ifm2, c0, c1),
 * B = mac(ifm2, c2, c3), H = mac(ifm1, F, B), S = mac(r1, F, zy):
 *   NONE: S;  RELU: max(bcc ? H : S, act0);  CLIP: clamp(H, act0, act1)
 *   LRELU a8 (p = r1 r2): bcc: mac(p, (r1 < 0) != (r2 < 0) ? c1 : c0, zy)
 *                         else mac(|p|, c3, mac(p, c2, zy))
 *   LRELU a16: bcc: max(S, mac(r1, fp32(F act0), zy))
 *              else max(H, fma(H, act0, act1))
 *   PRELU: k = (r1 < 0) != (r2 < 0) ? neg_scale[c] : c0;
 *          out = mac(r1, fp32(r2 k), zy)
 * neg_scale: device array of C floats, fp32(slope[c] * c0); null unless PRELU. */
int npu_qdq_elementwise_broadcasting3d_mul_fused_a8(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_mul_fused_coeff* coeff,
    npu_arith_activation activation, const float* neg_scale);
int npu_qdq_elementwise_broadcasting3d_mul_fused_a16(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_mul_fused_coeff* coeff,
    npu_arith_activation activation, const float* neg_scale);

/* 3D-broadcasting fused mac */

/* Coefficients of the fused mac (host pointer): c0 = s1 s2 / s_out, c1 = -zp2 c0,
 * c2 = -zp1 c0, c4 = s3 / s_out, c3 = zp_out - zp3 c4; act0 / act1: RELU / CLIP bounds, or
 * LRELU alpha and (1 - alpha) zp_out. */
typedef struct {
  float c0, c1, c2, c3, c4, act0, act1;
  int32_t zp1;
} npu_bc3d_mac_fused_coeff;

/* out = (ifm1 - zp1)(ifm2 - zp2) s1 s2 / s_out + (ifm3 - zp3) s3 / s_out + zp_out, ifm2 and
 * ifm3 broadcast over ifm1 with equal C. With bcc = (ifm2 C == 1):
 *   bcc and NONE: c3e = fp32(c3 - zp1 c1); t1 = mac(ifm2, c2, c3e);
 *                 t2 = mac(ifm2, c0, c1); t3 = mac(ifm3, c4, t1);
 *                 v = mac(ifm1, t2, t3)
 *   otherwise:    f = mac(ifm2, c0, c1); w = mac(x3, c4, c3);
 *                 v = mac(ifm1 - zp1, f, w)
 * where x3 = ifm3, converted to fp32 first on the activated bcc routes of a8w8 and a16w16.
 * LRELU: v = max(v, fma(v, act0, act1)). q = sat(rne(v)); RELU: q = max(q, sat(rne(act0)));
 * CLIP: also q = min(q, sat(rne(act1))).
 * ifm1 and out: ifm1_type (I8 / U8 for a8w8, I16 / U16 else); ifm2 I8 / U8 (a8w8, a16w8) or
 * I16 / U16 (a16w16); ifm3 I8 / U8 / I16 / U16 / I32, at least as wide as ifm1.
 * activation: NONE, RELU, CLIP, LRELU. */
int npu_qdq_elementwise_broadcasting3d_mac_fused_a8w8(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const int64_t* shape3,
    const npu_bc3d_mac_fused_coeff* coeff, npu_arith_activation activation);
int npu_qdq_elementwise_broadcasting3d_mac_fused_a16w8(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const int64_t* shape3,
    const npu_bc3d_mac_fused_coeff* coeff, npu_arith_activation activation);
int npu_qdq_elementwise_broadcasting3d_mac_fused_a16w16(
    void* stream, const void* ifm1, npu_dtype ifm1_type, const void* ifm2,
    npu_dtype ifm2_type, const void* ifm3, npu_dtype ifm3_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const int64_t* shape3,
    const npu_bc3d_mac_fused_coeff* coeff, npu_arith_activation activation);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_ARITH_INT_H_
