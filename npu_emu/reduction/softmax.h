/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Softmax / logsoftmax along the last axis of a contiguous [rows][C] tensor.
 *
 * Masks (where taken): one bit per element, [rows][C / 8] bytes, bit c % 8 of
 * byte c / 8; 1 = included, NULL = all included. A row with no included element
 * is evaluated with every x = -inf (NaN results) [default].
 * Special values: fp32 steps follow the NaN rules of common/mac.h [default]; a running
 * max is vmax (a NaN makes it NaN, -0 below +0).
 * Notation: exp2 = the hardware exp2 table applied to x * log2(e) + bias - max;
 * row sums are vector reduce-adds over 32 consecutive columns chained from +0;
 * inv = the hardware reciprocal; fp32(...) = one rounding.
 * bias = 895.001f, log2e = 0x3fb8aa3b, c_bf16 = 1.4453125, c_f16 = 1.4423828125.
 */
#ifndef NPU_EMU_SOFTMAX_H_
#define NPU_EMU_SOFTMAX_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* bf16 [rows][cols] -> bf16.
 *   s = fp32(x * c_bf16), off = fp32(bias - max(s)),
 *   e = exp2(fp32(bf16(s) + off)), out = bf16(bf16(e) * bf16(inv(sum(e)))),
 *   a subnormal result -> +-0.
 * Rows with x[0] = 0xff80 or 0xff7f and max(x) <= 0xff32: out =
 * bf16(inv(fp32(n * exp2(bias)))) at the n elements equal to the row max, 0
 * elsewhere. */
int npu_softmax_bf16(void* stream, const void* in, void* out, int64_t rows, int64_t cols);

/* bf16 [rows][cols] -> bf16, the row processed in num_chunks chunks of
 * chunk_cols columns (a multiple of 64, >= 128; cols <= chunk_cols *
 * num_chunks; columns >= cols count as -inf). The first pass keeps a running
 * max M and rescales the partial sum by exp2(bias - (M' - M)) * inv(exp2(bias))
 * when M grows; the second pass is as npu_softmax_bf16 with the final max.
 * Rows whose final M is -inf: out = bf16(inv(fp32(n * exp2(bias)))) at the n
 * elements equal to max(x), 0 elsewhere. */
int npu_softmax_2pass_bf16(void* stream, const void* in, void* out, int64_t rows, int64_t cols,
                           int64_t chunk_cols, int64_t num_chunks);

/* dtype BF16 or F16 (t), [rows][C] -> same type, C % 32 == 0, c = c_bf16 / c_f16.
 *   x = input, -inf where masked; off = fp32(bias - fp32(max(x) * c)),
 *   e = exp2(fp32(x * c + off)), out = t(t(e) * t(inv(sum(e)))), half to even,
 *   a subnormal result -> +-0. Masked elements give +0. */
int npu_softmax_fp16x16(void* stream, const void* in, const void* mask, void* out,
                        npu_dtype dtype, int64_t rows, int64_t C);

/* fp8 (E4M3FN) [rows][C] -> fp8, C % 64 == 0.
 *   v = bf16(fp32(x * ifm_scale)) (half away, subnormal -> 0; a NaN input makes
 *       its group of 8 columns NaN);
 *   y = the bf16 softmax of v (as npu_softmax_bf16, 16-bit stores half to even);
 *   out = fp8(clamp(fp32(y * bf16(ofm_scale_recip)), +-448)), half away,
 *       -0 -> +0. */
int npu_softmax_fp8(void* stream, const void* in, const void* mask, void* out, int64_t rows,
                    int64_t C, float ifm_scale, float ofm_scale_recip);

/* dtype BF16 or F16 (t), [rows][cols] -> same type.
 *   off = fp32(bias - fp32(max * log2e)), e = exp2(fp32(x * log2e + off)),
 *   sums over 64-column groups: pairs of full groups alternate even / odd, a
 *   leftover full group and the tail go to even; each group's low / high 32
 *   columns to lo / hi; sum = ((even_lo + even_hi) + odd_lo) + odd_hi;
 *   out = t(fp32(x + fp32(-max - ln(sum)))), half to even. */
int npu_logsoftmax_bf16(void* stream, const void* in, void* out, npu_dtype dtype, int64_t rows,
                        int64_t cols);

/* fp8 [rows][cols] -> fp8, in_scale > 0. v = decoded input (a NaN makes its
 * group of 8 columns NaN);
 *   max = max(fp32(v * in_scale)), off = fp32(bias - fp32(max * log2e)),
 *   e = exp2(fp32(v * fp32(in_scale * log2e) + off)), sums as npu_logsoftmax_bf16;
 *   out = fp8(fp32(v * fp32(in_scale * out_scale_inv)
 *                  + fp32(fp32(-max - ln(sum)) * out_scale_inv))),
 *   half to even; beyond 448 -> NaN. */
int npu_logsoftmax_fp8(void* stream, const void* in, void* out, int64_t rows, int64_t cols,
                       float in_scale, float out_scale_inv);

/* dtype I8 / U8 (a8) or I16 / U16 (a16) for input and output; C % 32 == 0.
 * path: 0 (C % 64 == 0), 1, or 2 (C == 32). -32 <= shift_res <= 32.
 *  path 0 / 1: off = fp32(bias - qmax * inv_ln2) (qmax = max over the mask),
 *    e = exp2(fp32(q * inv_ln2 + off)), 0 where masked; path 0 sums even / odd
 *    32-column blocks separately, then fp32(even + odd); e and
 *    inv(fp32(sum * ofm_scale)) are rounded to f16.
 *  path 2: v = fp32(q * inv_ln2) (-inf where masked),
 *    e = exp2(fp32(v - fp32(max(v) - bias))), inv(fp32(sum * ofm_scale)) in fp32.
 *  out = sat(rne(fp32(e * inv + zp_acc) * 2^shift_res)). */
int npu_qdq_softmax_a8(void* stream, const void* in, const void* mask, void* out,
                       npu_dtype dtype, int64_t rows, int64_t C, int path, int shift_res,
                       float inv_ln2, float ofm_scale, float zp_acc);
int npu_qdq_softmax_a16(void* stream, const void* in, const void* mask, void* out,
                        npu_dtype dtype, int64_t rows, int64_t C, int path, int shift_res,
                        float inv_ln2, float ofm_scale, float zp_acc);

/* in_type / out_type I8 / U8 (a8) or I16 / U16 (a16); C % 128 == 0.
 *   osi = inv(ofm_scale), inv_ln2 = fp32(log2e * ifm_scale),
 *   io = fp32(ifm_scale * osi); qmax = max over the mask;
 *   off = fp32(bias - qmax * inv_ln2), qio = fp32(zp - qmax * io);
 *   e = exp2(fp32(q * inv_ln2 + off)), 0 where masked; four sums: even / odd
 *   64-column block x low / high 32 columns, ((even_lo + even_hi) + odd_lo) +
 *   odd_hi; nl = fp32(qio - ln(sum) * osi);
 *   out = sat(rne(fp32(q * io + nl))); masked -> zero point. */
int npu_qdq_logsoftmax_a8(void* stream, const void* in, npu_dtype in_type, const void* mask,
                          void* out, npu_dtype out_type, int64_t rows, int64_t C,
                          float ifm_scale, float ofm_scale, int32_t ofm_zero_point);
int npu_qdq_logsoftmax_a16(void* stream, const void* in, npu_dtype in_type, const void* mask,
                           void* out, npu_dtype out_type, int64_t rows, int64_t C,
                           float ifm_scale, float ofm_scale, int32_t ofm_zero_point);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_SOFTMAX_H_
