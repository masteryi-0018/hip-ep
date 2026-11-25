/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Cast, convert, quantize, dequantize and requantize entry points.
 */
#ifndef NPU_EMU_CONVERT_H_
#define NPU_EMU_CONVERT_H_

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Every entry returns -1, before any launch, for null pointers (also when n == 0), n < 0,
 * unsupported types and sizes that are not a multiple of the kernel's unit. n == 0 (or N or
 * K == 0) is valid and returns 0 without launching. */

/* f32 -> f16 / bf16, round half to even. */
int npu_convert_fp32_fp16(void* stream, const void* in, void* out,
                          npu_dtype out_type, int64_t n);
/* f16 / bf16 -> f32 (exact). */
int npu_convert_fp16_fp32(void* stream, const void* in, npu_dtype in_type,
                          void* out, int64_t n);
/* f16 -> bf16 or bf16 -> f16, round half to even. */
int npu_convert_fp16_bf16(void* stream, const void* in, npu_dtype in_type,
                          void* out, npu_dtype out_type, int64_t n);
/* bf16 -> bfp16, mantissas round half to even. n % 64 == 0. */
int npu_cast_bf2bfp16(void* stream, const void* in, void* out, int64_t n);
/* bf16 / f16 -> bfp16, mantissas round half away from zero. n % 8 == 0. */
int npu_convert_bf16_to_bfp16(void* stream, const void* in, npu_dtype in_type,
                              void* out, int64_t n);
/* bfp16 -> bf16. Each group of 64 values (8 blocks) is aligned to its largest block
 * exponent E: values are rounded half to even to multiples of 2^(E - 150), a result below
 * 2^-126 is +0, then rounded half to even to bf16; a block exponent 255 makes the whole
 * group 0x7f81. n % 64 == 0. */
int npu_cast_bfp16tobf(void* stream, const void* in, void* out, int64_t n);
/* bf16 -> mx6, mantissas round half away from zero. n % 64 == 0. */
int npu_cast_bftomx6(void* stream, const void* in, void* out, int64_t n);
/* mx6 -> bf16. n % 256 == 0. */
int npu_cast_mx6tobf(void* stream, const void* in, void* out, int64_t n);

/* fp8 is E4M3FN. */

/* out = fp8(fp32(x * scale_recip)), round half away from zero, not saturated
 * (|v| >= 464 -> NaN). */
int npu_convert_fp32_fp8(void* stream, const void* in, void* out, int64_t n,
                         float scale_recip);
/* f16 / bf16 -> fp8: r = 1.0f / scale rounded to in_type;
 * out = fp8(clamp(x * r, -448, 448)), round half to even. */
int npu_convert_fp16_fp8(void* stream, const void* in, npu_dtype in_type,
                         void* out, int64_t n, float scale);
/* out = fp32(decode(x) * scale). */
int npu_convert_fp8_fp32(void* stream, const void* in, void* out, int64_t n,
                         float scale);
/* out = fp32(decode(x) * scale) to f16 / bf16, round half away from zero. */
int npu_convert_fp8_fp16(void* stream, const void* in, void* out,
                         npu_dtype out_type, int64_t n, float scale);

/* fp8 -> fp8: out = fp8(clamp(fp32(x * ifm_coeff), -448, 448)), round half to even;
 * ifm_coeff = ifm_scale / ofm_scale. -0 decodes to +0; a NaN spreads over its row of 8. */
int npu_requantize_fp8(void* stream, const void* in, void* out, int64_t n,
                       float ifm_coeff);

/* out = sat(rne(fma(x, scale, zero_point))); `scale` is the multiplier (1/s).
 * in_type: F32 / F16 / BF16; out_type: I8 / U8 / I16 / U16. */
int npu_quant(void* stream, const void* in, npu_dtype in_type, void* out,
              npu_dtype out_type, int64_t n, float zero_point, float scale);
/* bfp16 -> bf16 (as npu_cast_bfp16tobf) -> sat(rne(fma(x, quant_multiplier, zero_point)));
 * NaN -> 0. Output u8 (a8) / u16 (a16). n % 64 == 0. */
int npu_cast_bfp16toa8(void* stream, const void* in, void* out, int64_t n,
                       float quant_multiplier, float zero_point);
/* As npu_cast_bfp16toa8 with a u16 output. */
int npu_cast_bfp16toa16(void* stream, const void* in, void* out, int64_t n,
                        float quant_multiplier, float zero_point);

/* c = fp32(-zero_point * scale); out = rne(fma(x, scale, c)) to out_type.
 * in_type: I8 / U8 / I16 / U16; out_type: F32 / F16 / BF16. */
int npu_dequant(void* stream, const void* in, npu_dtype in_type, void* out,
                npu_dtype out_type, int64_t n, float zero_point, float scale);
/* d = x - (int32)zero_point; fp32(d * scale) -> bf16 -> bfp16, both half to even.
 * Input u8 (a8) / u16 (a16). n % 64 == 0. */
int npu_cast_a8tobfp16(void* stream, const void* in, void* out, int64_t n,
                       float scale, float zero_point);
/* As npu_cast_a8tobfp16 with a u16 input. */
int npu_cast_a16tobfp16(void* stream, const void* in, void* out, int64_t n,
                        float scale, float zero_point);
/* zp[N][K/32] (u8) and scale[N][K/32] (bf16) per group of 32 along K, q[N][K] u4 values
 * (one per byte). out = bfp16(fp32((q - zp) * s)), half to even, as row-major [N][K/8]
 * blocks. N % 64 == 0, K % 64 == 0. */
int npu_dequant_w4_bfp16(void* stream, const void* zp, const void* scale,
                         const void* q, void* out, int64_t N, int64_t K);

/* out = sat(rne(fma(x, c1, c0))), c1 = s_in / s_out, c0 = zp_out - c1 * zp_in.
 * in_type / out_type: I8 / U8 / I16 / U16 / I32 / U32, not both 32-bit. 32-bit outputs
 * saturate to int32 first, so U32 tops out at 2^31 - 1. */
int npu_requantize(void* stream, const void* in, npu_dtype in_type, void* out,
                   npu_dtype out_type, int64_t n, float c1, float c0);
/* As npu_requantize with per-channel fp32 (M[c], B[c]), c = i % channels.
 * in_type I8 / U8; out_type I8 / U8 / I16 / U16 / I32 / U32. channels is 4, 8, 16 or a
 * multiple of 32 up to 65535; n % max(32, channels) == 0. */
int npu_requantize_a8_channelwise(void* stream, const void* in, npu_dtype in_type,
                                  void* out, npu_dtype out_type, int64_t n,
                                  int64_t channels, const void* M, const void* B);
/* As npu_requantize_a8_channelwise with in_type I16 / U16. */
int npu_requantize_a16_channelwise(void* stream, const void* in, npu_dtype in_type,
                                   void* out, npu_dtype out_type, int64_t n,
                                   int64_t channels, const void* M, const void* B);
/* Exactly one side is I64 / U64. 64 -> narrower uses the fp32 fields (fp32 MACs
 * over the hi / lo words); narrower -> 64 uses M, s, B:
 * out = round_half_even((x * M + B) / 2^s) in int64, U64 clamps < 0 to 0. */
typedef struct {
  float mult_f, bias_f, mult_f_x2p32, mult_f_x2p64;
  int32_t M, s;
  int64_t B;
} npu_a64_coeff;
int npu_requantize_a64(void* stream, const void* in, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t n,
                       const npu_a64_coeff* coeff);
/* int32 = sat(rne(fma(x, c1, c0))); in_type I8 / U8 / I16 / U16;
 * out_type I32, or I64 (sign-extended). */
int npu_cast_int32_int64(void* stream, const void* in, npu_dtype in_type,
                         void* out, npu_dtype out_type, int64_t n, float c1,
                         float c0);

/* q = npu_cast_bfp16toa16(quant_multiplier, zero_point);
 * out = fp32((q - (int32)zero_point) * scale). n % 128 == 0. */
int npu_cast_bfp16toqdqf32(void* stream, const void* in, void* out, int64_t n,
                           float quant_multiplier, float zero_point,
                           float scale);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_CONVERT_H_
