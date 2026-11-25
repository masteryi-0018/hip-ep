/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Sigmoid / tanh (SigTanh) and SiLU on bf16 / f16, bfp16, fp8 (E4M3FN) and quantized
 * integers. Elementwise on n contiguous elements; fp8 is decoded in groups of 8 and bfp16 in
 * groups of 64 elements from the buffer start.
 *
 * fma / fp32(...) = one rounding per multiply / fused MAC; exp2 = the hardware exp2 table
 * (subnormal -> +0); narrowing to bf16 / f16 rounds half away from zero.
 * SigTanh core, act = 0 sigmoid / 1 tanh, coefficients (k0, k1, k2):
 *   e = exp2(fma(fp32(x * x), k2, fma(-|x|, k1, k0)))
 *   tanh:    fma(-2, e, 1) with the sign of x;
 *   sigmoid: fma(-1, e, 1) where x > 0, else e.
 *   Fixed coefficients: sigmoid (894.00097656, 0.76842356, -0.12469438),
 *   tanh (894.00097656, 1.53878212, -0.49819374).
 * SiLU core, xa = -|x|:
 *   v = fma(xa, 0.329241553, -1.227518206); e = exp2(fma(-v, v, 895.507801));
 *   z = fma(xa, e, max(x, -0)).
 * ct = compute_type; dq(x) = ct(fma(x, dq_sc, fp32(-dq_zp * dq_sc)));
 * q(y) = sat(round(fma(y, q_sc, q_zp))), half away from zero, q_sc = 1 / ofm_scale.
 * Null buffers are rejected (also when n == 0); n == 0 is a no-op.
 *
 * fp32 QDQ SiLU (no 16-bit intermediate), s = dq_sc, r = q_sc, exp2 as above,
 * mac(i, c, a) = i * c + a rounded once for an exact integer i:
 *   mode 0: xk = mac(x, fp32(s * r), fp32(dq_zp * -fp32(s * r))); u = fma(-|xk|, ik, 895),
 *     ik = fp32(log2(e) / r); n = exp2(u with the low 14 mantissa bits cleared),
 *     f = u - that; e = fp32(n * P3(f)) (Horner fma); s0 = exp2(fma(-e, h2(e), 895));
 *     s1 = fma(s0, fma(-(e + 1), s0, 1), s0); y = fma(max(xk, fp32(xk * e)), s1, q_zp).
 *   mode 1: qi = x - int(dq_zp); a = |qi|; C = fp32(s * r), Cg = fp32(C * g);
 *     arg = mac(a, fp32(s * -log2(e)), A), A = 895 + log2(Cg) by exponent and polynomial;
 *     ep = exp2(arg); h = fma(fma(c2', ep, c1'), ep, c0') with c_k' = c_k / Cg^(k+1) by
 *     products of fp32(1 / Cg); sg = exp2(fma(-ep, h, arg));
 *     y = mac(-a, sg, mac(max(qi, 0), C, q_zp)).
 *   out = sat(round(y)), half away from zero.
 */
#ifndef NPU_EMU_SILU_H_
#define NPU_EMU_SILU_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* in_type BF16 / F16 (also the output type). out = in_type(core(x)) with the fixed
 * coefficients. */
int npu_sigtanh_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n,
                     int act);

/* in_type BF16 / F16 (also the output type). out = in_type(silu(x)). */
int npu_silu_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n);

/* bfp16 in and out, n % 512 == 0: n / 8 blocks of 9 bytes [exponent][8 x int8 mantissa],
 * value = m * 2^(e - 133). out = bfp16(bf16(core(bf16(x)))). */
int npu_sigtanh_bfp16(void* stream, const void* in, void* out, int64_t n, int act);
/* The SiLU variant of npu_sigtanh_bfp16. */
int npu_silu_bfp16(void* stream, const void* in, void* out, int64_t n);

/* fp8 in and out. poly1_coeff = lit1 * ifm_scale, poly2_coeff = lit2 * ifm_scale^2 (lit =
 * the fixed coefficients), ofm_coeff = 1 / ofm_scale; k0 = 894.00097656.
 * out = fp8(fp32(core(x) * ofm_coeff)), half away, not saturated (above 448 -> NaN). */
int npu_elementwise_sigtanh_fp8(void* stream, const void* in, void* out, int64_t n, int act,
                                float poly1_coeff, float poly2_coeff, float ofm_coeff);

/* fp8 in and out, compute_type ct (BF16 / F16).
 * x = ct(fp32(v * ifm_scale)); y = ct(silu(x));
 * out = fp8(clamp(fp32(y * ct(ofm_scale_recip)), -448, 448)), half away. */
int npu_elementwise_silu_fp8(void* stream, const void* in, void* out, int64_t n,
                             npu_dtype compute_type, float ifm_scale, float ofm_scale_recip);

/* in_type / out_type: I8 / U8 (a8) or I16 / U16 (a16); compute_type BF16 / F16.
 * out = q(bf16(core(dq(x)))) with coefficients (k0, k1, k2). */
int npu_qdq_sigtanh_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                       npu_dtype out_type, int64_t n, int act, npu_dtype compute_type,
                       float dq_zp, float dq_sc, float q_zp, float q_sc, float k0, float k1,
                       float k2);
/* The 16-bit variant of npu_qdq_sigtanh_a8. */
int npu_qdq_sigtanh_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                        npu_dtype out_type, int64_t n, int act, npu_dtype compute_type,
                        float dq_zp, float dq_sc, float q_zp, float q_sc, float k0, float k1,
                        float k2);

/* As npu_qdq_sigtanh_a8 / _a16. out = q(ct(silu(dq(x)))). */
int npu_qdq_silu_exp2_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                         npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                         float dq_sc, float q_zp, float q_sc);
/* The 16-bit variant of npu_qdq_silu_exp2_a8. */
int npu_qdq_silu_exp2_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                          npu_dtype out_type, int64_t n, npu_dtype compute_type, float dq_zp,
                          float dq_sc, float q_zp, float q_sc);

/* in_type / out_type: I16 / U16; mode 0 (accurate) or 1 (GeLU form).
 * out = the fp32 QDQ SiLU above. */
int npu_qdq_silu_fp32_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                          npu_dtype out_type, int64_t n, int mode, float dq_zp, float dq_sc,
                          float q_zp, float q_sc);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_SILU_H_
