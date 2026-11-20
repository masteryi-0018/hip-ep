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
 * dq(x) = ct(fma(x, dq_scale, fp32(-dq_zp * dq_scale)));
 * q(y) = sat(round(fma(y, q_scale, q_zp))), half away from zero, q_scale = 1 / ofm_scale.
 */
#ifndef NPU_EMU_SILU_H_
#define NPU_EMU_SILU_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* dtype BF16 / F16 in and out. out = dtype(core(x)) with the fixed coefficients. */
int npu_sigtanh_bf16(void* stream, const void* in, void* out, npu_dtype dtype, int64_t n,
                     int act);

/* dtype BF16 / F16 in and out. out = dtype(silu(x)). */
int npu_silu_bf16(void* stream, const void* in, void* out, npu_dtype dtype, int64_t n);

/* bfp16 in and out, n % 8 == 0: n / 8 blocks of 9 bytes [exponent][8 x int8 mantissa],
 * value = m * 2^(e - 133). out = bfp16(bf16(core(bf16(x)))). */
int npu_sigtanh_bfp16(void* stream, const void* in, void* out, int64_t n, int act);
int npu_silu_bfp16(void* stream, const void* in, void* out, int64_t n);

/* fp8 in and out. k1 = lit1 * ifm_scale, k2 = lit2 * ifm_scale^2 (lit = the fixed
 * coefficients), ofm_recip = 1 / ofm_scale; k0 = 894.00097656.
 * out = fp8(fp32(core(x) * ofm_recip)), half away, not saturated (above 448 -> NaN). */
int npu_elementwise_sigtanh_fp8(void* stream, const void* in, void* out, int64_t n, int act,
                                float k1, float k2, float ofm_recip);

/* fp8 in and out, compute type ct (BF16 / F16).
 * x = ct(fp32(v * ifm_scale)); y = ct(silu(x));
 * out = fp8(clamp(fp32(y * ct(ofm_recip)), -448, 448)), half away. */
int npu_elementwise_silu_fp8(void* stream, const void* in, void* out, int64_t n, npu_dtype ct,
                             float ifm_scale, float ofm_recip);

/* in_type / out_type: I8 / U8 (a8) or I16 / U16 (a16); ct BF16 / F16.
 * out = q(bf16(core(dq(x)))) with coefficients (k0, k1, k2). */
int npu_qdq_sigtanh_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                       npu_dtype out_type, int64_t n, int act, npu_dtype ct, float dq_zp,
                       float dq_scale, float q_zp, float q_scale, float k0, float k1, float k2);
int npu_qdq_sigtanh_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                        npu_dtype out_type, int64_t n, int act, npu_dtype ct, float dq_zp,
                        float dq_scale, float q_zp, float q_scale, float k0, float k1,
                        float k2);

/* As npu_qdq_sigtanh_a8 / _a16. out = q(ct(silu(dq(x)))). */
int npu_qdq_silu_exp2_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                         npu_dtype out_type, int64_t n, npu_dtype ct, float dq_zp,
                         float dq_scale, float q_zp, float q_scale);
/* The 16-bit variant of npu_qdq_silu_exp2_a8. */
int npu_qdq_silu_exp2_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                          npu_dtype out_type, int64_t n, npu_dtype ct, float dq_zp,
                          float dq_scale, float q_zp, float q_scale);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_SILU_H_
