/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Bit-exact GPU emulation of the AIE4 (NPU) conversion kernels in mllib
 * (L2/src/wrapper/{cast,convert,quant,dequant,requantize}_*_wrapper.cpp).
 *
 * One entry point per mllib wrapper. Tensors use the NPU's real element types;
 * the arithmetic follows the kernel source, not mllib's python reference, which
 * differs from the NPU by up to 1 ULP. Every rule below was checked against
 * aiesim outputs of the mllib nightly regression unless marked "unverified".
 *
 * Common rules:
 *   - x * m + c is one fused multiply-add rounded once to fp32; an integer x
 *     enters exactly (it is not rounded to fp32 first, even above 2^24).
 *   - float -> integer rounds half to even and saturates to the output range.
 *   - Block floating point: bfp16 = blocks of 8 values, 9 bytes each:
 *     [shared exponent (uint8), 8 x int8 two's-complement mantissas];
 *     value = m * 2^(e - 133). mx6 = blocks of 16 values, 12 bytes each, a
 *     little-endian 96-bit word: bits 0-7 shared exponent, bits 8-15 one shift
 *     bit per pair of values, bits 16+5k..20+5k the 5-bit sign-magnitude
 *     mantissa of value k; value = +-mag * 2^(e - shift - 130).
 *   - fp8 is E4M3FN (bias 7, no infinity, 0x7F / 0xFF are NaN). The kernels
 *     decode fp8 by an 8x8 identity matrix multiply, so a NaN makes all 8
 *     values of its group (8 consecutive elements from the buffer start) NaN.
 *
 * All pointers are device pointers; n counts elements. Functions are
 * asynchronous on `stream` (a hipStream_t, may be null) and return a
 * hipError_t value (0 on success) or -1 for unsupported arguments.
 */
#ifndef NPU_EMU_H_
#define NPU_EMU_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NPU_F32 = 0,
  NPU_F16 = 1,
  NPU_BF16 = 2,
  NPU_FP8 = 3, /* E4M3FN */
  NPU_I8 = 4,
  NPU_U8 = 5,
  NPU_I16 = 6,
  NPU_U16 = 7,
  NPU_I32 = 8,
  NPU_U32 = 9,
  NPU_I64 = 10,
  NPU_U64 = 11,
} npu_dtype;

/* ---- A. format conversion (no scale) ---------------------------------- */

/* convert_fp32_fp16: f32 -> f16 / bf16, round half to even. */
int npu_convert_fp32_fp16(void* stream, const void* in, void* out,
                          npu_dtype out_type, int64_t n);
/* convert_fp16_fp32: f16 / bf16 -> f32 (exact). */
int npu_convert_fp16_fp32(void* stream, const void* in, npu_dtype in_type,
                          void* out, int64_t n);
/* convert_fp16_bf16: f16 -> bf16 or bf16 -> f16, round half to even. */
int npu_convert_fp16_bf16(void* stream, const void* in, npu_dtype in_type,
                          void* out, npu_dtype out_type, int64_t n);
/* cast_bf2bfp16: bf16 -> bfp16, mantissas round half to even. n % 8 == 0. */
int npu_cast_bf2bfp16(void* stream, const void* in, void* out, int64_t n);
/* convert_bf16_to_bfp16: bf16 / f16 -> bfp16, mantissas round half away from
 * zero (the wrapper sets symmetric_inf). Logical element order; the NPU's
 * K64 tiling is not emulated. n % 8 == 0. */
int npu_convert_bf16_to_bfp16(void* stream, const void* in, npu_dtype in_type,
                              void* out, int64_t n);
/* cast_bfp16tobf: bfp16 -> bf16. n % 8 == 0. */
int npu_cast_bfp16tobf(void* stream, const void* in, void* out, int64_t n);
/* cast_bftomx6: bf16 -> mx6, mantissas round half away from zero. n % 16 == 0. */
int npu_cast_bftomx6(void* stream, const void* in, void* out, int64_t n);
/* cast_mx6tobf: mx6 -> bf16. n % 16 == 0. */
int npu_cast_mx6tobf(void* stream, const void* in, void* out, int64_t n);

/* ---- B. fp8 (E4M3FN) with a scale ------------------------------------- */

/* convert_fp32_fp8: fp8(x * scale_recip), the product rounded to fp32, then
 * to fp8 rounding half away from zero, NOT saturated (|v| >= 464 -> NaN). */
int npu_convert_fp32_fp8(void* stream, const void* in, void* out, int64_t n,
                         float scale_recip);
/* convert_fp16_fp8: r = (in_type)(1.0f / scale) rounded to f16 / bf16; the
 * product x * r is exact in fp32; clamp to [-448, 448]; round half to even.
 * (1.0f / scale on the NPU is unverified for non power-of-two scales.) */
int npu_convert_fp16_fp8(void* stream, const void* in, npu_dtype in_type,
                         void* out, int64_t n, float scale);
/* convert_fp8_fp32: decode(x) * scale, rounded to fp32. */
int npu_convert_fp8_fp32(void* stream, const void* in, void* out, int64_t n,
                         float scale);
/* convert_fp8_fp16: decode(x) * scale rounded to fp32, then to f16 / bf16
 * rounding half away from zero (the wrapper sets symmetric_inf). */
int npu_convert_fp8_fp16(void* stream, const void* in, void* out,
                         npu_dtype out_type, int64_t n, float scale);

/* ---- C. quantize ------------------------------------------------------ */

/* quant (QuantA8 / QuantA16): out = sat(rne(fma(x, scale, zero_point))).
 * `scale` is the multiplier the NPU receives (1/s), `zero_point` a float.
 * in_type: F32 / F16 / BF16; out_type: I8 / U8 / I16 / U16. */
int npu_quant(void* stream, const void* in, npu_dtype in_type, void* out,
              npu_dtype out_type, int64_t n, float scale, float zero_point);
/* cast_bfp16toa8 / cast_bfp16toa16: bfp16 -> bf16 (exact) ->
 * sat_unsigned(rne(fma(x, quant_multiplier, zero_point))). out_type U8 / U16.
 * Unverified: no aiesim output was kept for the cases that differ. */
int npu_cast_bfp16toaq(void* stream, const void* in, void* out,
                       npu_dtype out_type, int64_t n, float quant_multiplier,
                       float zero_point);

/* ---- D. dequantize ---------------------------------------------------- */

/* dequant (DequantA8 / DequantA16): c = fp32(-zero_point * scale);
 * out = round_rne(fma(x, scale, c)) to out_type (F32 / F16 / BF16).
 * in_type: I8 / U8 / I16 / U16. The kernel's association order (x*s - zp*s)
 * is used; ONNX's s*(x - zp) gives the same results on every aiesim case so
 * far, but the two can differ (unverified which the NPU matches beyond them). */
int npu_dequant(void* stream, const void* in, npu_dtype in_type, void* out,
                npu_dtype out_type, int64_t n, float scale, float zero_point);
/* cast_a8tobfp16 / cast_a16tobfp16: d = x - (int32)zero_point (truncated);
 * v = fp32(d * scale) -> bf16 (half to even) -> bfp16 (half to even).
 * in_type U8 / U16. n % 8 == 0. */
int npu_cast_aqtobfp16(void* stream, const void* in, npu_dtype in_type,
                       void* out, int64_t n, float scale, float zero_point);
/* dequant_w4_bfp16: W[N][K] u4 values (one per byte, 0..15), per-group zero
 * points zp[N][K/32] (u8) and scales s[N][K/32] (bf16), group = 32 along K.
 * out = bfp16 of fp32((q - zp) * s) directly (no bf16 step), half to even;
 * blocks of 8 along K, row-major [N][K/8] blocks. K % 32 == 0. Logical
 * layout; the NPU's packed weight layout is not emulated. */
int npu_dequant_w4_bfp16(void* stream, const void* q, const void* zp,
                         const void* scale, void* out, int64_t N, int64_t K);

/* ---- E. requantize (integer -> integer) -------------------------------- */

/* requantize_a8 / _a16 / _a32: out = sat(rne(fma(x, c1, c0))) with the fp32
 * coefficients the NPU receives (c1 = s_in / s_out, c0 = zp_out - c1*zp_in).
 * in_type I8 / U8 / I16 / U16 / I32 / U32; out_type I8 / U8 / I16 / U16 / I32
 * / U32. 32-bit outputs saturate to int32 first, so U32 tops out at 2^31 - 1.
 * U32 input adds fma(-1 if x >= 2^31, -c1 * 2^32, .) like the kernel. */
int npu_requantize(void* stream, const void* in, npu_dtype in_type, void* out,
                   npu_dtype out_type, int64_t n, float c1, float c0);
/* requantize_a8_channelwise / _a16_channelwise: as npu_requantize with
 * per-channel fp32 (M[c], B[c]); c = i % channels (innermost axis).
 * in_type I8 / U8 / I16 / U16; out_type I8 / U8 / I16 / U16 / I32 / U32. */
int npu_requantize_channelwise(void* stream, const void* in, npu_dtype in_type,
                               void* out, npu_dtype out_type, int64_t n,
                               const float* M, const float* B,
                               int64_t channels);
/* requantize_a64: exactly one side is I64 / U64. Coefficients as the NPU's
 * coeff buffer: DOWN (64 -> small) uses mult_f, bias_f, mult_f_x2p32,
 * mult_f_x2p64 (three fp32 MACs over the hi / lo words); UP (small -> 64) uses
 * M, s, B: out = round_half_even((x*M + B) / 2^s) in int64, U64 clamps < 0. */
typedef struct {
  float mult_f, bias_f, mult_f_x2p32, mult_f_x2p64;
  int32_t M, s;
  int64_t B;
} npu_a64_coeff;
int npu_requantize_a64(void* stream, const void* in, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t n,
                       const npu_a64_coeff* coeff);
/* cast_int32_int64: in I8 / U8 / I16 / U16 -> int32 = sat(rne(fma(x, c1, c0)));
 * out_type I32, or I64 (the int32 result sign-extended). */
int npu_cast_int32_int64(void* stream, const void* in, npu_dtype in_type,
                         void* out, npu_dtype out_type, int64_t n, float c1,
                         float c0);

/* ---- F. quantize then dequantize --------------------------------------- */

/* cast_bfp16toqdqf32: q = npu_cast_bfp16toaq(U16, quant_multiplier,
 * zero_point); out = fp32((q - (int32)zero_point) * scale). */
int npu_cast_bfp16toqdqf32(void* stream, const void* in, void* out, int64_t n,
                           float quant_multiplier, float zero_point,
                           float scale);

#ifdef __cplusplus
}
#endif

/* Data-movement / selection wrappers. */
#include "npu_emu_movement.h"
#include "npu_emu_select.h"
#include "npu_emu_qdq.h"

/* QDQ integer matrix-multiply / convolution wrappers. */
#include "npu_emu_matmul.h"

/* xint8 (power-of-two scale) convolution, depthwise and conv + add wrappers. */
#include "npu_emu_xint8_conv.h"

/* Elementwise arithmetic: integer / qdq add, sub, mul, mac (npu_emu_arith_int.h),
 * integer division, prelu / leaky relu / square and qdq batchnorm
 * (npu_emu_arith_misc.h), and the bf16 / f16 / fp8 / bfp16 ones
 * (npu_emu_arith_float.h). */
#include "npu_emu_arith_int.h"
#include "npu_emu_arith_misc.h"
#include "npu_emu_arith_float.h"

/* Interpolation / sampling / position encoding: bilinear grid sample
 * (npu_emu_gridsample.h), bilinear / cubic resize (npu_emu_resize.h) and RoPE
 * (npu_emu_rope.h). */
#include "npu_emu_gridsample.h"
#include "npu_emu_resize.h"
#include "npu_emu_rope.h"

#endif  // NPU_EMU_H_
