/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Bilinear GridSample entry points (zeros padding).
 */
#ifndef NPU_EMU_GRIDSAMPLE_H_
#define NPU_EMU_GRIDSAMPLE_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Layout: in [H][W][C], grid [Ho][Wo][2] (x, y), out [Ho][Wo][C]. C must be even
 * and H * W <= 32768; other shapes return -1. T is the 2-byte float type (BF16 /
 * F16); conversions round half to even. A neighbour outside the image has
 * coefficient 0. */

/* in, grid and out of type dtype (BF16 / F16). The grid holds pixel coordinates
 * (align_corners already applied). */
int npu_gridsample_bf16(void* stream, const void* in, npu_dtype dtype,
                        const void* grid, void* out, int64_t H, int64_t W,
                        int64_t C, int64_t Ho, int64_t Wo);

/* in / out fp8 (E4M3FN), grid of grid_type (BF16 / F16), which is also T.
 * v = T(fp32(decode(x) * ifm_scale)); out = fp8(sat(fp32(acc * ofm_recip))),
 * ofm_recip = 1 / ofm_scale. */
int npu_gridsample_fp8(void* stream, const void* in, const void* grid,
                       npu_dtype grid_type, void* out, int64_t H, int64_t W,
                       int64_t C, int64_t Ho, int64_t Wo, float ifm_scale,
                       float ofm_recip);

/* in / out of type dtype (I8 / U8 for a8, I16 / U16 for a16), grid U16 codes.
 * v = T(fma(x, ifm_scale, -ifm_zp * ifm_scale));
 * out = sat(rne(fma(acc, ofm_scale, ofm_zp))), ofm_scale = 1 / s_out.
 * T is BF16 for a8, `compute` (BF16 / F16) for a16. align_corners is 0 or 1. */
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
