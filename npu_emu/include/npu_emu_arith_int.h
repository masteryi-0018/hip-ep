/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 integer elementwise arithmetic wrappers
 * (arith_int group): QDQ add / sub / mul, their fused-activation variants,
 * the Broadcasting3d add / sub / mul / mac family and the Xint8 add / mac
 * kernels. Conventions as in npu_emu.h.
 *
 * Arithmetic. The QDQ kernels evaluate the dequantize-op-requantize chain as
 * a sequence of accfloat MACs, mac_elem_32(x, m, a) = x * m + a rounded once
 * to fp32 (half to even); an integer x enters exactly. The order of the MACs
 * differs per kernel and per route and is given per function: the result is
 * not the same as any other association order. A MAC whose x is an fp32
 * value is an fp32 fused multiply-add. The final float -> integer conversion
 * (to_vector_sign / to_v32int8 / to_v32int16) rounds half to even and
 * saturates to the output type, although the wrappers select
 * saturation_mode::truncate. A NaN accumulator gives 0 (unverified).
 *
 * Coefficients. The coefficient arguments are the fp32 words the kernel
 * reads (kernel_rtp.h for the plain flat kernels, the coeff buffer
 * coeff32.txt for the others), not ONNX scales and zero points; npu_emu does
 * not derive them. mllib's YAML computes them in double and stores them as
 * fp32, and an assignment that references another one uses its double value
 * (c1 = -zp2 * c0 multiplies by the unrounded c0). With s1, zp1, s2, zp2,
 * s_out, zp_out the ONNX parameters of ifm1, ifm2 and the output:
 *   add: c0 = s1/s_out, c1 = s2/s_out, c2 = zp_out - zp1*s1/s_out - zp2*s2/s_out
 *   sub: c0 = s1/s_out, c1 = -s2/s_out, c2 = zp_out - zp1*s1/s_out + zp2*s2/s_out
 *   mul: c0 = s1*s2/s_out, c1 = -zp2*c0, c2 = -zp1*c0, c3 = zp1*zp2*c0 + zp_out
 * Sub runs the add kernel with the sub coefficients (the Sub wrappers are
 * YAML specializations of the Add ones), so add entries serve both.
 *
 * Activations (fused wrappers, npu_arith_activation; bounds and slopes are in
 * the output's quantized domain): CLIP clamps to [act0, act1] (relu6, clamp,
 * and relu when the YAML gives it a finite upper bound); RELU only applies the
 * lower bound act0; LRELU and PRELU are evaluated per kernel, see each
 * function. Clamping an fp32 value and then rounding equals rounding and then
 * clamping to the rounded bounds (rounding is monotone), so the kernels that
 * clamp in the integer domain are covered by the same formula.
 *
 * Layout. Flat kernels take n contiguous elements. The Broadcasting3d kernels
 * take logical HWC tensors: ifm1 has shape shape1 = {H, W, C} (host array of
 * 3 int64), ifm2 (and ifm3) have shapes whose every dimension is 1 or equal
 * to ifm1's (numpy broadcasting); the output has ifm1's shape. The kernels
 * pick their route at run time from broadcast_c = (ifm2 C == 1), which
 * changes the MAC order. The NPU's sub-volume tiling, C64 padding and the
 * physical broadcast layouts are not emulated.
 */
#ifndef NPU_EMU_ARITH_INT_H_
#define NPU_EMU_ARITH_INT_H_

#include <stdint.h>

#include "npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NPU_ARITH_ACT_NONE = 0,
  NPU_ARITH_ACT_RELU = 1,  /* max(v, act0) */
  NPU_ARITH_ACT_CLIP = 2,  /* min(max(v, act0), act1) */
  NPU_ARITH_ACT_LRELU = 3,
  NPU_ARITH_ACT_PRELU = 4,
} npu_arith_activation;

/* ---- A. flat QDQ add / sub / mul --------------------------------------- */

/* elementwise_add_a8 / _a16 (also the Sub A8 / A16 wrappers, with the sub
 * coefficients): out = sat(rne(mac(b, c1, mac(a, c0, c2)))).
 * in_type is the type of both inputs, I8 / U8 (a8) or I16 / U16 (a16);
 * out_type of the same width, either signedness. */
int npu_elementwise_add_a8(void* stream, const void* a, const void* b, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float c2);
int npu_elementwise_add_a16(void* stream, const void* a, const void* b, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float c2);

/* elementwise_mul_a8 / _a16: the Horner form of
 * (a - zp1)(b - zp2) c0 + zp_out, i.e.
 *   t1 = mac(b, c2, c3);  t2 = mac(b, c0, c1);  out = sat(rne(mac(a, t2, t1)))
 * Types as for npu_elementwise_add_a8 / _a16. */
int npu_elementwise_mul_a8(void* stream, const void* a, const void* b, npu_dtype in_type,
                           void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                           float c2, float c3);
int npu_elementwise_mul_a16(void* stream, const void* a, const void* b, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t n, float c0, float c1,
                            float c2, float c3);

/* elementwise_add_fused_a8 / _a16 and elementwise_sub_fused_a8 / _a16 (one
 * kernel; the sub wrappers pass the sub coefficients): v as for
 * npu_elementwise_add_a8, then the activation of impl_common.h
 * apply_relu_variants, then sat(rne(.)). activation: NONE; CLIP with
 * act0 / act1 (coeff words 3 / 4; relu passes act1 = FLT_MAX/s_out + zp_out);
 * LRELU: v = max(v, fma(v, act0, act1)) with act0 = alpha and
 * act1 = (1 - alpha) * zp_out. RELU and PRELU are not instantiated (-1). */
int npu_elementwise_add_fused_a8(void* stream, const void* a, const void* b,
                                 npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                 float c0, float c1, float c2, npu_arith_activation activation,
                                 float act0, float act1);
int npu_elementwise_add_fused_a16(void* stream, const void* a, const void* b,
                                  npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2,
                                  npu_arith_activation activation, float act0, float act1);
int npu_elementwise_sub_fused_a8(void* stream, const void* a, const void* b,
                                 npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                 float c0, float c1, float c2, npu_arith_activation activation,
                                 float act0, float act1);
int npu_elementwise_sub_fused_a16(void* stream, const void* a, const void* b,
                                  npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2,
                                  npu_arith_activation activation, float act0, float act1);

/* elementwise_mul_fused_a8 / _a16: v as for npu_elementwise_mul_a8, then the
 * activation as for npu_elementwise_add_fused_a8 (coeff words 4 / 5). */
int npu_elementwise_mul_fused_a8(void* stream, const void* a, const void* b,
                                 npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                 float c0, float c1, float c2, float c3,
                                 npu_arith_activation activation, float act0, float act1);
int npu_elementwise_mul_fused_a16(void* stream, const void* a, const void* b,
                                  npu_dtype in_type, void* out, npu_dtype out_type, int64_t n,
                                  float c0, float c1, float c2, float c3,
                                  npu_arith_activation activation, float act0, float act1);

/* ---- B. Xint8 ---------------------------------------------------------- */

/* elementwise_add_a8_xint (Xint8_AddWrapper): I8 only;
 * out = sat(rne(mac(b, c1, mac(a, c0, c2)))). The YAML packs c2 = 0. */
int npu_elementwise_add_a8_xint(void* stream, const void* a, const void* b, void* out,
                                int64_t n, float c0, float c1, float c2);

/* elementwise_add_fused_a8_xint (Xint8_AddFusedWrapper): I8 only, pure
 * integer arithmetic. c0 = s1/s_out and c1 = s2/s_out must be powers of two
 * (the kernel uses only their exponents e0, e1; anything else gives -1).
 * With b0 = min(e0, e1, 0):
 *   v = (a << (e0 - b0)) + (b << (e1 - b0));  out = sat8(rne(v / 2^-b0))
 * activation NONE, or CLIP with bounds sat8(rne(act0)), sat8(rne(act1))
 * applied to the int8 result (the YAML's relu / relu6 use CLIP). */
int npu_elementwise_add_fused_a8_xint(void* stream, const void* a, const void* b, void* out,
                                      int64_t n, float c0, float c1,
                                      npu_arith_activation activation, float act0, float act1);

/* elementwise_mac_a8 (Xint8_MACWrapper): ofm = s_mul*ifm1*ifm2 + s_add*ifm3
 * (zero points 0) in fixed point, with the integer coefficients of the coeff
 * buffer: A = rne(s_mul * 2^(p+t)), v = rne(s_add * 2^p), output shift p and
 * the shift t (7) used to derive w from A:
 *   w = srs16(ifm2 * A, t);  out = sat(rne((ifm1 * w + ifm3 * v) / 2^p))
 * srs16 rounds half to even; it saturates to int16, except that when ifm2 is
 * a single value (shape {1, 1, 1}) the kernel's scalar path wraps instead
 * (reachable only with U8 ifm2; unverified). ifm2 / ifm3 broadcast over
 * ifm1 (shape1). Types: each input I8 / U8 (independent), out I8 / U8. */
int npu_elementwise_mac_a8(void* stream, const void* ifm1, npu_dtype ifm1_type,
                           const void* ifm2, npu_dtype ifm2_type, const void* ifm3,
                           npu_dtype ifm3_type, void* out, npu_dtype out_type,
                           const int64_t* shape1, const int64_t* shape2,
                           const int64_t* shape3, int32_t A, int32_t v, int32_t p, int32_t t);

/* ---- C. Broadcasting3d add / sub / mul (no activation) ----------------- */

/* qdq_elementwise_broadcasting3d_add_a8 / _a16 (also the Broadcasting3d Sub
 * wrappers, with the sub coefficients): ifm2 broadcast over ifm1, both
 * routes compute out = sat(rne(mac(ifm2, c1, mac(ifm1, c0, c2)))).
 * in_type for both inputs; out_type of the same width. */
int npu_qdq_elementwise_broadcasting3d_add_a8(void* stream, const void* ifm1, const void* ifm2,
                                              npu_dtype in_type, void* out, npu_dtype out_type,
                                              const int64_t* shape1, const int64_t* shape2,
                                              float c0, float c1, float c2);
int npu_qdq_elementwise_broadcasting3d_add_a16(void* stream, const void* ifm1,
                                               const void* ifm2, npu_dtype in_type, void* out,
                                               npu_dtype out_type, const int64_t* shape1,
                                               const int64_t* shape2, float c0, float c1,
                                               float c2);

/* qdq_elementwise_broadcasting3d_mul_a8 / _a16: as npu_elementwise_mul_a8
 * with ifm2 broadcast. The YAML's c1..c3 reuse the double c0:
 * c1 = -zp2*c0, c2 = -zp1*c0, c3 = zp_out + zp1*zp2*c0. */
int npu_qdq_elementwise_broadcasting3d_mul_a8(void* stream, const void* ifm1, const void* ifm2,
                                              npu_dtype in_type, void* out, npu_dtype out_type,
                                              const int64_t* shape1, const int64_t* shape2,
                                              float c0, float c1, float c2, float c3);
int npu_qdq_elementwise_broadcasting3d_mul_a16(void* stream, const void* ifm1,
                                               const void* ifm2, npu_dtype in_type, void* out,
                                               npu_dtype out_type, const int64_t* shape1,
                                               const int64_t* shape2, float c0, float c1,
                                               float c2, float c3);

/* ---- D. Broadcasting3d fused add / sub / mul --------------------------- */

/* The first 8 words of the add / sub fused coeff buffer (host pointer).
 * w3..w7 depend on the activation:
 *   RELU: w3 = lower bound;  CLIP: w3, w4 = bounds;
 *   LRELU: w3 = alpha, w4 = beta = (1 - alpha) zp_out, w5 = fp32(alpha c0),
 *          w6 = fp32(alpha c1), w7 = fp32(alpha c2 + beta) (double products of
 *          the unrounded c0 / c1 / c2);
 *   PRELU: c2 holds fp32(c2 - zp_out) and w3 = zp_out (the header of every
 *          per-channel-tile block; the slopes follow it in the buffer). */
typedef struct {
  float c0, c1, c2, w3, w4, w5, w6, w7;
} npu_bc3d_addsub_fused_coeff;

/* qdq_elementwise_broadcasting3d_add_fused_a8 / _a16 and
 * qdq_elementwise_broadcasting3d_sub_fused_a8 / _a16 (one kernel). in_type is
 * the type of ifm1, ifm2 and the output (I8 / U8 or I16 / U16). With
 * bcc = (ifm2 C == 1), the affine part is
 *   bcc: A(bias) = mac(ifm1, c0, mac(ifm2, c1, bias))
 *   else A(bias) = mac(ifm2, c1, mac(ifm1, c0, bias))
 * NONE: A(c2);  RELU: max(A(c2), w3);  CLIP: clamp(A(c2), w3, w4);
 * LRELU: r = A(c2), max(r, l) with
 *   bcc: l = mac(ifm1, w5, fma(mac(ifm2, c1, c2), w3, w4))
 *   else l = mac(ifm2, w6, mac(ifm1, w5, w7))
 * PRELU: d = A(c2); out = fma(d, d < 0 ? slope[c] : 1, w3), slope a device
 * array of C floats (the coeff payload), c the channel; null otherwise. */
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

/* The first 9 words of the mul fused coeff buffer (host pointer): c0..c3 as
 * for npu_qdq_elementwise_broadcasting3d_mul_a8; act0 / act1 the activation
 * words (RELU: lower bound; CLIP: bounds; LRELU a16: alpha and
 * beta = (1 - alpha) zp_out); z1 = zp1 and z2 = zp2 as int32; zy = zp_out.
 * For LRELU a8 the YAML repacks c1 = fp32(alpha c0), c2 = fp32(c0 (1+alpha)/2),
 * c3 = fp32(c0 (1-alpha)/2). */
typedef struct {
  float c0, c1, c2, c3, act0, act1;
  int32_t z1;
  float zy;
  int32_t z2;
} npu_bc3d_mul_fused_coeff;

/* qdq_elementwise_broadcasting3d_mul_fused_a8 / _a16. With bcc = (ifm2 C == 1),
 * r1 = ifm1 - z1, r2 = ifm2 - z2 (exact integers), F = mac(ifm2, c0, c1),
 * B = mac(ifm2, c2, c3), Horner H = mac(ifm1, F, B), shifted S = mac(r1, F, zy):
 *   NONE: S;  RELU: max(bcc ? H : S, act0);  CLIP: clamp(H, act0, act1)
 *   LRELU a8 (p = r1 r2): bcc: mac(p, (r1 < 0) != (r2 < 0) ? c1 : c0, zy)
 *                         else mac(|p|, c3, mac(p, c2, zy))
 *   LRELU a16: bcc: max(S, mac(r1, fp32(F act0), zy))
 *              else max(H, fma(H, act0, act1))
 *   PRELU: k = (r1 < 0) != (r2 < 0) ? neg_scale[c] : c0;
 *          out = mac(r1, fp32(r2 k), zy)
 * neg_scale: device array of C floats, fp32(slope[c] * c0) as the YAML packs
 * them; null unless PRELU. in_type as for the add fused functions. */
int npu_qdq_elementwise_broadcasting3d_mul_fused_a8(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_mul_fused_coeff* coeff,
    npu_arith_activation activation, const float* neg_scale);
int npu_qdq_elementwise_broadcasting3d_mul_fused_a16(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    const int64_t* shape1, const int64_t* shape2, const npu_bc3d_mul_fused_coeff* coeff,
    npu_arith_activation activation, const float* neg_scale);

/* ---- E. Broadcasting3d mac fused --------------------------------------- */

/* The first 8 words of the mac fused coeff buffer (host pointer), from the
 * ONNX parameters of ifm1..ifm3 and the output: c0 = s1 s2 / s_out,
 * c1 = -zp2 c0, c2 = -zp1 c0, c4 = s3 / s_out, c3 = zp_out - zp3 c4 (double c0
 * / c4 reused); act0 / act1: RELU / CLIP bounds (A8W8 caps the upper bound at
 * the output's qmax) or LRELU alpha and (1 - alpha) zp_out; zp1 as int32. */
typedef struct {
  float c0, c1, c2, c3, c4, act0, act1;
  int32_t zp1;
} npu_bc3d_mac_fused_coeff;

/* qdq_elementwise_broadcasting3d_mac_fused_a8w8 / _a16w8 / _a16w16:
 * ofm = (ifm1 - zp1)(ifm2 - zp2) s1 s2 / s_out + (ifm3 - zp3) s3 / s_out + zp_out
 * with ifm2 and ifm3 broadcast over ifm1; their C must agree (both 1 or both
 * C). With bcc = (ifm2 C == 1):
 *   bcc and NONE: c3e = fp32(c3 - zp1 c1); t1 = mac(ifm2, c2, c3e);
 *                 t2 = mac(ifm2, c0, c1); t3 = mac(ifm3, c4, t1);
 *                 v = mac(ifm1, t2, t3)
 *   otherwise:    f = mac(ifm2, c0, c1); w = mac(x3, c4, c3);
 *                 v = mac(ifm1 - zp1, f, w)
 * where x3 = ifm3, except on the activated bcc routes of A8W8 and A16W16,
 * which first convert ifm3 to fp32 (rounding int32 addends above 2^24).
 * LRELU: v = max(v, fma(v, act0, act1)). The output q = sat(rne(v)); RELU:
 * q = max(q, sat(rne(act0))); CLIP: also q = min(q, sat(rne(act1))).
 * Types: ifm1 and the output ifm1_type (I8 / U8 for a8w8, I16 / U16 else);
 * ifm2 I8 / U8 (a8w8, a16w8) or I16 / U16 (a16w16); ifm3 I8 / U8 / I16 /
 * U16 / I32, at least as wide as ifm1. activation: NONE, RELU, CLIP, LRELU. */
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
