/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 quantized elementwise arithmetic wrappers
 * of the arith_misc group: integer division (elementwise_div_a8 / _a16,
 * qdq_elementwise_broadcasting3d_div_a8 / _a16), the activations
 * elementwise_leakyrelu / prelu / square (_a8 / _a16) and
 * qdq_batchnorm2d_stream_a8 / _a16. Conventions as in npu_emu.h.
 *
 * Common rules (checked on the aiesim outputs of the mllib nightly builds
 * 2829 / 2831 / 2833 unless marked "unverified"):
 *   - mac(x, c, a) = x * c + a rounded once to fp32 (half to even). An integer
 *     x (mac_elem_32(int32, float, accfloat)) enters exactly.
 *   - The fp32 result is converted to the output integer type rounding half
 *     to even and saturating (to_vector_sign / to_fixed under conv_even),
 *     after a multiplication by 2^-output_shift where the kernel has one.
 *     NaN converts to 0 (unverified; only 0 / 0 in the divisions produces it).
 *   - Divisions are a * inv(b): the NPU has no divider. inv is a reciprocal
 *     unit whose result depends on the element type:
 *       bf16: RNE(1/b) in bf16, except that a subnormal or zero b gives +-inf
 *             and a result below 2^-126 (|b| > 2^126) gives +-0 (no bf16
 *             subnormals; mllib inv_lut.py inv_bf16_aie4);
 *       fp16: RNE(1/b) in fp16 with subnormals (inv_lut.py inv_fp16_aie4);
 *       fp32: the scalar ::inv, faithfully but not correctly rounded: RNE(1/b)
 *             corrected by -1 / 0 / +1 ulp from a table indexed by the
 *             significand of b (src/npu_emu_inv_f32_table.h).
 *     The product of two bf16 / fp16 values is exact in fp32; the fp32 product
 *     is rounded once.
 *   - The coefficient arguments are the fp32 words the kernel reads (runtime
 *     parameters or the coefficient / weight buffer), which the mllib YAML
 *     computes in double and stores as fp32. The usual formulas are given per
 *     function (s_x, zp_x are the ONNX scales and zero points).
 *
 * Tensors are contiguous and in logical layout: 1-D elementwise, [rows][C]
 * with the channel innermost, or [H][W][C] for the broadcasting kernels. The
 * NPU's sub-volume tiling, channel blocking and padding are not emulated.
 * Integer types are I8 / U8 (a8) or I16 / U16 (a16).
 */
#ifndef NPU_EMU_ARITH_MISC_H_
#define NPU_EMU_ARITH_MISC_H_

#include <stdint.h>

#include "npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- A. division ------------------------------------------------------- */

/* elementwise_div_a8 / _a16: a and b share in_type, out_type may differ in
 * sign. The kernel converts x - zp exactly through the fp32 "magic number"
 * 12582912.0f (0x4B400000), so it receives the zero points pre-biased:
 *   zp1_word = 12582912.0f + zp_a, zp2_word = 12582912.0f + zp_b
 *   da = bf16_rne((12582912.0f + a) - zp1_word),  db likewise with b
 *   q  = da * inv_bf16(db)                          (exact in fp32)
 *   out = sat(rne(fma(q, c0, c1)))
 * c0 = s_a / (s_b * s_out), c1 = zp_out (runtime words 2..5 are c0, c1,
 * zp1_word, zp2_word; the signedness bytes follow). A zero divisor
 * gives q = +-inf, which saturates; 0 / 0 gives NaN -> 0 (unverified). */
int npu_elementwise_div_a8(void* stream, const void* a, const void* b,
                           npu_dtype in_type, void* out, npu_dtype out_type,
                           int64_t n, float c0, float c1, float zp1_word,
                           float zp2_word);
int npu_elementwise_div_a16(void* stream, const void* a, const void* b,
                            npu_dtype in_type, void* out, npu_dtype out_type,
                            int64_t n, float c0, float c1, float zp1_word,
                            float zp2_word);

/* qdq_elementwise_broadcasting3d_div_a8 / _a16: a is [H][W][C], b is
 * [bH][bW][bC] broadcast to it (each of bH / bW / bC is the full extent or
 * 1); a, b and out share dtype. With da = a - zp1, db = b - zp2 in int32:
 *   a8:  da, db -> fp16 (rne); db == 0 is replaced by fp16(7e-5);
 *        q = da * inv_fp16(db)                       (exact in fp32)
 *   a16: da, db -> fp32 (exact); db == 0 is replaced by 7e-5f;
 *        q = fp32(da * inv_f32(db))
 *   out = sat(rne(fma(q, c0, c1) * 2^-output_shift))
 * Coefficient buffer: c0 = word 0 = floor(s_a / (s_b * s_out) * 2^15),
 * c1 = word 16 = floor(zp_out * 2^15) (integers stored as fp32), zp1 = word
 * 32 and zp2 = word 96 (int32); output_shift is the runtime output_sft (15).
 * The inv_f32 table covers divisors with at most 16 significant bits, so a16
 * requires zp2 within dtype's range (then |db| <= 65535); other zp2 return -1.
 * On the NPU a C == 1
 * divisor is read once per 32 (a16) / 64 (a8) output elements; with a's C
 * not a multiple of that the NPU result differs (unverified, not emulated). */
int npu_qdq_elementwise_broadcasting3d_div_a8(
    void* stream, const void* a, const void* b, npu_dtype dtype, void* out,
    int64_t H, int64_t W, int64_t C, int64_t bH, int64_t bW, int64_t bC,
    float c0, float c1, int32_t zp1, int32_t zp2, int output_shift);
int npu_qdq_elementwise_broadcasting3d_div_a16(
    void* stream, const void* a, const void* b, npu_dtype dtype, void* out,
    int64_t H, int64_t W, int64_t C, int64_t bH, int64_t bW, int64_t bC,
    float c0, float c1, int32_t zp1, int32_t zp2, int output_shift);

/* ---- B. activations ---------------------------------------------------- */

/* elementwise_leakyrelu_a8 / _a16:
 *   out = sat(rne(max(mac(x, c1, c0), mac(x, c1_alpha, c0_alpha))))
 * Coefficient buffer words 0..3: c1 = s_in / s_out, c0 = zp_out - c1 * zp_in,
 * c1_alpha = alpha * c1, c0_alpha = alpha * (c0 - zp_out) + zp_out. The
 * kernel always takes the max, so alpha > 1 gives max, not leaky relu. */
int npu_elementwise_leakyrelu_a8(void* stream, const void* in,
                                 npu_dtype in_type, void* out,
                                 npu_dtype out_type, int64_t n, float c1,
                                 float c0, float c1_alpha, float c0_alpha);
int npu_elementwise_leakyrelu_a16(void* stream, const void* in,
                                  npu_dtype in_type, void* out,
                                  npu_dtype out_type, int64_t n, float c1,
                                  float c0, float c1_alpha, float c0_alpha);

/* elementwise_prelu_a8 / _a16: in and out are [rows][C], negative_scale [C]
 * a device array (one slope per channel):
 *   d = x - in_zp (int32);  s = d < 0 ? negative_scale[c] : positive_scale
 *   out = sat(rne(mac(d, s, out_zp)))
 * Coefficient buffer: positive_scale = word 0 (fp32 s_in / s_out), in_zp =
 * word 1 (int32), out_zp = word 2 (fp32 zp_out), negative_scale from byte 128
 * (fp32 slope[c] * positive_scale). */
int npu_elementwise_prelu_a8(void* stream, const void* in, npu_dtype in_type,
                             void* out, npu_dtype out_type, int64_t rows,
                             int64_t C, float positive_scale, int32_t in_zp,
                             float out_zp, const float* negative_scale);
int npu_elementwise_prelu_a16(void* stream, const void* in, npu_dtype in_type,
                              void* out, npu_dtype out_type, int64_t rows,
                              int64_t C, float positive_scale, int32_t in_zp,
                              float out_zp, const float* negative_scale);

/* elementwise_square_a8 / _a16: k * (x - zp_in)^2 + zp_out in Horner form,
 *   t = mac(x, a, b);  out = sat(rne(mac(x, t, c)))
 * Coefficient buffer words 0..2: a = k, b = -2 * k * zp_in,
 * c = k * zp_in^2 + zp_out, with k = s_in^2 / s_out. */
int npu_elementwise_square_a8(void* stream, const void* in, npu_dtype in_type,
                              void* out, npu_dtype out_type, int64_t n,
                              float a, float b, float c);
int npu_elementwise_square_a16(void* stream, const void* in,
                               npu_dtype in_type, void* out,
                               npu_dtype out_type, int64_t n, float a, float b,
                               float c);

/* ---- C. batch normalization -------------------------------------------- */

/* The wrapper's activation_type (mllib::ActivationType values). */
typedef enum {
  NPU_QDQ_BN_NONE = 0,
  NPU_QDQ_BN_RELU = 1,
  NPU_QDQ_BN_LRELU = 3,
} npu_qdq_bn_activation;

/* qdq_batchnorm2d_stream_a8 / _a16: in and out are [rows][C] of dtype
 * (rows = H * W * N), cw and cb device arrays [C]:
 *   o = mac(x, cw[c], cb[c])
 *   RELU:  o = max(o, ofm_zp)
 *   LRELU: lc = fp32(ofm_zp * fp32(1 - alpha)); o = max(o, fma(o, alpha, lc))
 *   out = sat(rne(o))
 * Weight buffer: cw (cw_eff) and cb (cb_eff) are the per-channel fp32 words
 * the YAML packs, cw = bf16(gamma') * fp32(s_in / s_out) and
 * cb = bf16(beta') * fp32(1 / s_out) + fp32(bf16(gamma') * fp32(-zp_in * s_in
 * / s_out) + zp_out) with gamma' / beta' the folded batchnorm scale and shift
 * (each step rounded to fp32); ofm_zp and alpha are the two words after them
 * (relu_alpha; read only for LRELU). */
int npu_qdq_batchnorm2d_stream_a8(void* stream, const void* in,
                                  npu_dtype dtype, void* out, int64_t rows,
                                  int64_t C, const float* cw, const float* cb,
                                  npu_qdq_bn_activation act, float ofm_zp,
                                  float alpha);
int npu_qdq_batchnorm2d_stream_a16(void* stream, const void* in,
                                   npu_dtype dtype, void* out, int64_t rows,
                                   int64_t C, const float* cw, const float* cb,
                                   npu_qdq_bn_activation act, float ofm_zp,
                                   float alpha);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_ARITH_MISC_H_
