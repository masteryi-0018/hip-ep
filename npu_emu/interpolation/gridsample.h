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

/* Layout: in [H][W][C], grid [Ho][Wo][2] (x, y), out [Ho][Wo][C]. C must be even and >= 2,
 * H, W >= 1 and H * W <= 32768; other shapes and null pointers (also for empty work) return
 * -1. Ho or Wo 0: nothing to do, returns 0. T is the 2-byte float type (BF16 /
 * F16); conversions round half to even. A neighbour outside the image has
 * coefficient 0. */

/* in, grid and out of in_type (BF16 / F16). The grid holds pixel coordinates
 * (align_corners already applied). */
int npu_gridsample_bf16(void* stream, const void* in, npu_dtype in_type,
                        const void* grid, void* out, int64_t H, int64_t W,
                        int64_t C, int64_t Ho, int64_t Wo);

/* in / out fp8 (E4M3FN), grid of grid_type (BF16 / F16), which is also T.
 * v = T(fp32(decode(x) * ifm_scale)); out = fp8(sat(fp32(acc * ofm_scale))),
 * ofm_scale = 1 / s_out. */
int npu_gridsample_fp8(void* stream, const void* in, const void* grid,
                       npu_dtype grid_type, void* out, int64_t H, int64_t W,
                       int64_t C, int64_t Ho, int64_t Wo, float ifm_scale,
                       float ofm_scale);

/* in / out of in_type (I8 / U8 for a8, I16 / U16 for a16), grid U16 codes.
 * v = T(fma(x, ifm_scale, -ifm_zp * ifm_scale));
 * out = sat(rne(fma(acc, ofm_scale, ofm_zp))), ofm_scale = 1 / s_out.
 * T is BF16 for a8, compute_type (BF16 / F16) for a16. align_corners is 0 or 1. */
int npu_qdq_gridsample_a8(void* stream, const void* in, npu_dtype in_type,
                          const void* grid, void* out, int64_t H, int64_t W,
                          int64_t C, int64_t Ho, int64_t Wo,
                          int align_corners, const npu_gridsample_qdq* coeff);
int npu_qdq_gridsample_a16(void* stream, const void* in, npu_dtype in_type,
                           const void* grid, void* out, int64_t H, int64_t W,
                           int64_t C, int64_t Ho, int64_t Wo,
                           int align_corners, npu_dtype compute_type,
                           const npu_gridsample_qdq* coeff);

/* Nearest GridSample whose grid is formed from an int8 flow f, fused with the
 * ReduceL2 of the flow. srs(v, s, T) = v * 2^-s rounded half to even, saturated
 * to T; clamp8 saturates to int8. Per output point (h, w), with fx / fy the
 * flow codes of the point and Tx = tx[w], Ty = ty[h]:
 *   e = srs((Tx << ka) + fx, ka, E), g = srs(e * mq_x, msh_x, E),
 *   rx = srs(g * ax + bx, shift, int16); ry likewise on the y axis.
 * Zeros padding: E = int8. Border: E = int16, rx clamped to [0, hx] with
 *   hx = clamp(srs(clamp8(srs(127 * mq_x, msh_x, int16)) * ax + bx, shift, int16),
 *              0, W - 1); ry likewise with H.
 * Image part, x = in[clamp(ry)][clamp(rx)][k]: border with out_shift >= 0:
 *   srs(x, out_shift, int8); else (bf16 requant, coeff fields ifm_zp, ifm_scale,
 *   ofm_zp, ofm_scale = 1 / s_out):
 *   v = bf16(fp32(x * ifm_scale + fp32(-ifm_zp * ifm_scale)) * valid),
 *   out = sat8(rne(fma(v, ofm_scale, ofm_zp))), valid = the point is inside; the zero
 *   points are read as int16.
 * Norm n of sq = sat16(fx^2 + fy^2): v the largest odd number <= 253 with
 *   srs(v^2, c, int16) < sq (c = 0 for norm_mode 2, else 2; v = -1 if none),
 *   k = (v + 1) / 2; mode 0: n = k; mode 2: k + 1 if (v + 2)^2 == sq and v % 4 == 1;
 *   modes 1 / 2: n = clamp8(srs(k * n_mq, n_msh, int16)).
 * Resize variant (pre): the flow is formed from a and c first,
 *   m = clamp8(srs(a * p_mq, p_msh, int16)), f = clamp8(srs(m * p_am + c * p_ac, p_ash, int16)).
 * The p_* fields are used by the resize variant only. */
typedef struct {
  int32_t ka, mq_x, msh_x, mq_y, msh_y;
  int32_t shift, ax, bx, ay, by;
  int32_t norm_mode, n_mq, n_msh;
  int32_t out_shift; /* -1: bf16 requant */
  int32_t p_mq, p_msh, p_am, p_ac, p_ash;
} npu_gridsample_flow;

/* Layout: in int8 [H][W][4]; flow int8 [Hf][Wf][4] with (fx, fy) in bytes 0 and 1
 * (bytes 2, 3 unused), [Hf][Wf] = [Ho][Wo], or [Ho / 2][Wo / 2] with f_up (each flow
 * point repeated 2 x 2); tx int8 [Wo], ty int8 [Ho]; out int8 [Ho][Wo][8] holding
 * per point the 4 image codes, then n as little-endian int32.
 * padding_mode 0 = zeros, 1 = border. Ho must be a multiple of 8 and at least 32,
 * Wo a multiple of 4 (Ho or Wo 0: nothing to do, returns 0), 1 <= H, W <= 32767; every
 * pointer is required; the multipliers (mq_*, ax, ay, n_mq, p_mq, p_am, p_ac) must fit
 * int16, shifts 0..31 (out_shift also -1), norm_mode 0..2; else -1. */
int npu_qdq_Gridsample_ReduceL2_Fused_a8(void* stream, const void* in, const void* flow,
                                         const void* tx, const void* ty, void* out, int64_t H,
                                         int64_t W, int64_t Ho, int64_t Wo, int padding_mode,
                                         int f_up, const npu_gridsample_qdq* coeff,
                                         const npu_gridsample_flow* p);

/* a and c in the layout of flow above; the flow is formed from them (pre). */
int npu_qdq_Resize_Gridsample_ReduceL2_Fused_a8(void* stream, const void* in, const void* a,
                                                const void* c, const void* tx, const void* ty,
                                                void* out, int64_t H, int64_t W, int64_t Ho,
                                                int64_t Wo, int padding_mode, int f_up,
                                                const npu_gridsample_qdq* coeff,
                                                const npu_gridsample_flow* p);

#ifdef __cplusplus
}
#endif

#endif /* NPU_EMU_GRIDSAMPLE_H_ */
