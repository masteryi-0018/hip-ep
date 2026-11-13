/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 bilinear GridSample wrappers:
 * gridsample_bf16, gridsample_fp8 (AIE_SOL grid_sample_bf16 stages) and
 * qdq_gridsample_a8 / _a16 (L1 gridsample2d_bilinear_qdq.hpp). Conventions as
 * in npu_emu.h. Padding mode is always zeros (the only one the kernels
 * implement).
 *
 * Layout: in [H][W][C], grid [Ho][Wo][2] (x, y), out [Ho][Wo][C], all
 * contiguous. C must be even: the kernels process channel pairs. The NPU's
 * channel / width iterations and padded slices are not emulated, but the
 * kernels index with int16, so H * W must be <= 32768 (and H, W <= 32767);
 * other shapes return -1.
 *
 * Common NPU rules (T = the 2-byte interpolation type, bf16 or f16; every
 * conversion rounds half to even):
 *   - index: fx = floor(gx) saturated to int16 (NaN -> 0, unverified),
 *     fxc = min(fx, W - 1) (no lower clamp); same for y. The top-left index
 *     i0 = sat_int16(fxc + fyc * W). The four neighbours are i0 + {0, 1, W,
 *     W + 1} computed in int16 (wrapping), then clamped to [0, H * W - 1].
 *     A neighbour outside the image therefore reads some in-image pixel; its
 *     coefficient is zero instead.
 *   - resize: acc = fp32(v00 * c00), then three fp32 fused MACs with v01 c01,
 *     v10 c10, v11 c11 (T x T products are exact in fp32).
 *
 * gridsample_bf16 / gridsample_fp8 coefficients (all T arithmetic is the fp32
 * result rounded to T):
 *   ffx = T(fxc), t0x = T(T(1 - gx) + ffx), t1x = T(gx - ffx)   (same for y)
 *   pad_w = T(bf16(W)) (the RTP stores W as bfloat16), pad_w1 = T(pad_w - 1)
 *   column 0 is out when gx > pad_w  or gx < 0, column 1 when gx > pad_w1 or
 *   gx < -1; rows 0 / 1 likewise with gy and pad_h. The x term is zeroed
 *   (+0) before the product when its column or row is out:
 *   c00 = T(sel(t0x) * t0y), c01 = T(sel(t1x) * t0y),
 *   c10 = T(sel(t0x) * t1y), c11 = T(sel(t1x) * t1y)
 *
 * qdq_gridsample_a8 / _a16 run the coordinate math in fp32:
 *   g  = fp32(fp32(float(code) - float(grid_zp)) * grid_scale) + 1  (fp32)
 *   align_corners 0: gx = fma(g, fp32(W) * 0.5, -0.5)   (one rounding)
 *   align_corners 1: gx = fp32(fp32(W - 1) * g) * 0.5
 *   t0x = fp32(fp32(1 - gx) + float(fxc)), t1x = fp32(gx - float(fxc))
 *   c = T(fp32(tx * ty)), set to +0 when the unclamped neighbour (fx or
 *   int16(fx + 1), fy or int16(fy + 1)) lies outside [0, W) x [0, H).
 * align_corners 1 is not covered by the nightly regression (unverified).
 *
 * Unverified on the NPU: NaN / infinite grid values, fp16 overflow (results
 * past 65504 become infinity here), subnormal T values (kept, not flushed).
 */
#ifndef NPU_EMU_GRIDSAMPLE_H_
#define NPU_EMU_GRIDSAMPLE_H_

#include <stdint.h>

#include "npu_emu.h"
#include "npu_emu_qdq.h" /* npu_gridsample_qdq */

#ifdef __cplusplus
extern "C" {
#endif

/* gridsample_bf16: bilinear GridSample with T = dtype (BF16 or F16) for in,
 * grid and out. The grid holds denormalized pixel coordinates (the caller
 * applied align_corners). out = T(acc). */
int npu_gridsample_bf16(void* stream, const void* in, npu_dtype dtype,
                        const void* grid, void* out, int64_t H, int64_t W,
                        int64_t C, int64_t Ho, int64_t Wo);

/* gridsample_fp8: in and out FP8 (E4M3FN), grid of type grid_type (BF16 or
 * F16), which is also T. Coefficient words (coeff32.txt): ifm_scale (word
 * 0) and ofm_recip (word 1) = fp32(1.0 / ofm_scale) computed in double.
 *   v   = T(fp32(fp8(x) * ifm_scale)), x decoded in groups of 8 (a NaN makes
 *         its group NaN; -0 decodes to +0). The groups are 4 pixels x one
 *         channel pair: the kernel converts each channel pair's [H][W][2]
 *         slice from its start.
 *   out = fp8(fp32(acc * ofm_recip)): half to even, saturated to +-448 (NaN
 *         stays NaN, the sign of zero is kept). */
int npu_gridsample_fp8(void* stream, const void* in, const void* grid,
                       npu_dtype grid_type, void* out, int64_t H, int64_t W,
                       int64_t C, int64_t Ho, int64_t Wo, float ifm_scale,
                       float ofm_recip);

/* qdq_gridsample_a8 / _a16: in and out of type dtype (I8 / U8 for a8, I16 /
 * U16 for a16), grid U16 codes. coeff is the coefficient buffer as the kernel
 * reads it (coeff32.txt; ofm_scale is the multiplier fp32(1.0 / s_out)).
 *   v   = T(mac(x, ifm_scale, fp32(-float(ifm_zp) * ifm_scale)))  (x exact)
 *   out = sat(rne(fma(acc, ofm_scale, float(ofm_zp))))
 * T is BF16 for a8; compute (BF16 or F16) for a16. */
int npu_qdq_gridsample_a8(void* stream, const void* in, npu_dtype dtype,
                          const void* grid, void* out, int64_t H, int64_t W,
                          int64_t C, int64_t Ho, int64_t Wo,
                          int align_corners, const npu_gridsample_qdq* coeff);
int npu_qdq_gridsample_a16(void* stream, const void* in, npu_dtype dtype,
                           const void* grid, void* out, int64_t H, int64_t W,
                           int64_t C, int64_t Ho, int64_t Wo,
                           int align_corners, npu_dtype compute,
                           const npu_gridsample_qdq* coeff);

#ifdef __cplusplus
}
#endif

#endif /* NPU_EMU_GRIDSAMPLE_H_ */
