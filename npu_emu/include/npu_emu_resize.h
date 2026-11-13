/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 bilinear / bicubic resize wrappers:
 * resize_bilinear (A8, also the multi-core tile-replay cases), resize_bilinear_a16,
 * resize_bilinear_bf16, resize_bilinear_fp8 and resize_cubic_bf16.
 * Conventions as in npu_emu.h. The arithmetic follows the kernel sources
 * (L1/include/aie4_misc/resize_bilinear*_impl.hpp, resize_bilinear_fp8.hpp,
 * resize_cubic_bf16_impl.hpp) and was checked byte for byte against aiesim
 * outputs of the mllib nightly regression unless marked "unverified".
 *
 * The kernels compute no coordinates: the host packs, per output row / column,
 * a source index and the interpolation weights into the wts blob, and the core
 * only gathers and multiplies. These entries therefore take exactly what the
 * kernel reads -- the index tables and the weight words of the blob -- not the
 * ONNX coordinate_transformation_mode. The blob's indices are relative to each
 * sub-volume's input window; pass them made absolute (window origin added), so
 * that one table covers the whole tensor.
 *
 * Shapes: tensors are logical HWC, in = [Hi][Wi][C], out = [Ho][Wo][C], the
 * valid region densely packed (the NPU's C32 / C64 blocking, halo columns and
 * sub-volume tiling are not emulated). idx_h[Ho] / idx_w[Wo] are device int32
 * tables. A tap whose weight is zero contributes nothing, but which pixel the
 * kernel reads for it is part of the rules below (it matters for the float
 * kernels, where 0 * inf is NaN and the signs of zeros add); reads past the
 * tensor (the NPU reads a halo / pad pixel there) are clamped to the edge.
 *
 * Rules shared by the bilinear kernels (4-tap route):
 *   - Weights s[h][w][0..3] weight the pixels (y0, x0), (y0, x1), (y1, x0),
 *     (y1, x1) with y0 = idx_h[h], x0 = idx_w[w].
 *   - x1 = x0 + 1 if s[h][w][1] != 0, else x0 (per pixel).
 *   - y1 = y0 + 1 if s[h][w0][2] != 0, else y0, where w0 is the first output
 *     column of the NPU sub-volume containing w (w0 = w - w % sv_wo): the kernel
 *     decides the bottom row once per sub-volume row from that column. sv_wo is
 *     the sub-volume output width (ofm_sv_width); 0 means Wo.
 *   - acc = x00 * s0, then three MACs with s1, s2, s3 in that order.
 */
#ifndef NPU_EMU_RESIZE_H_
#define NPU_EMU_RESIZE_H_

#include <stdint.h>

#include "npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Route of the integer bilinear kernel: the wrapper dispatches on RTP word 17
 * (separable_gate == kResizeSeparableGateMagic, set by the YAML's
 * assignments.use_separable). The blob layout differs per route. */
typedef enum {
  NPU_RESIZE_4TAP = 0,      /* resize_bilinear_impl */
  NPU_RESIZE_SEPARABLE = 1, /* resize_bilinear_separable_impl (8-bit only) */
} npu_resize_route;

/* resize_bilinear (A8 wrapper, multi-core tile replay) and resize_bilinear_a16:
 * t = I8 / U8 (both routes) or I16 / U16 (4-tap only); int32 accumulator, exact.
 *
 * NPU_RESIZE_4TAP: wts = s[Ho][Wo][4] uint16 Q15 (the blob's s_table); taps as
 *   above; acc = sum of x * s (Q = 15).
 * NPU_RESIZE_SEPARABLE: wts = qy[Ho][2] then qx[Wo][2] (uint16; the blob's qy /
 *   qx tables, Q_H = 8 for I8 and 7 for U8, Q_W = 7). sv_wo is unused.
 *     H pass: tmp(x) = sat_int16(in[y0][x] * qy0 + in[y1][x] * qy1),
 *             y1 = y0 + 1 if qy1 != 0, else y0;
 *     W pass: acc = tmp(x0) * qx0 + tmp(x0 + 1) * qx1 (x0 + 1 always, the
 *             kernel loads it unconditionally); Q = Q_H + 7 (15 / 14).
 * Output (the wrapper sets symmetric_floor = toward zero, and saturation):
 *   requantize == 0: out = sat(sign(acc) * (|acc| >> Q));
 *   requantize != 0: v = fp32(acc * c0 + fp32(c1 * 2^Q)), one rounding with acc
 *                    exact (mac_elem_32); out = sat(trunc(v * 2^-Q)).
 * c0 / c1 are the two fp32 words at the head of the blob (c0 = s_in / s_out and
 * c1 = zp_out - c0 * zp_in, which the YAML evaluates in double and stores as
 * fp32). Weights are taken as given; with weights that do not sum to one the
 * int32 accumulator could overflow, which is unverified (computed exactly here).
 */
int npu_resize_bilinear(void* stream, const void* in, void* out, npu_dtype t,
                        int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                        const int32_t* idx_h, const int32_t* idx_w,
                        const uint16_t* wts, npu_resize_route route, int64_t sv_wo,
                        int requantize, float c0, float c1);

/* resize_bilinear_bf16: t = BF16 / F16; wts = s[Ho][Wo][4] in t (the blob's
 * s_table). 4-tap as above in fp32: the first product is rounded (exact for
 * 16-bit operands), each MAC is one fused rounding (half to even). The wrapper
 * sets symmetric_floor, so the result is truncated toward zero to t; BF16 /
 * F16 results below the smallest normal flush to a signed zero (unverified,
 * mllib's model filters denormals), F16 magnitudes >= 2^16 saturate to 65504. */
int npu_resize_bilinear_bf16(void* stream, const void* in, void* out, npu_dtype t,
                             int64_t Hi, int64_t Wi, int64_t C, int64_t Ho,
                             int64_t Wo, const int32_t* idx_h, const int32_t* idx_w,
                             const void* wts, int64_t sv_wo);

/* resize_bilinear_fp8: in / out fp8 (E4M3FN), interior type t = BF16 / F16
 * (the wrapper's compute dtype); wts = s[Ho][Wo][4] in t.
 *   1. convert_fp8_fp16: x = trunc_t(fp32(decode(in) * ifm_dequant_coeff)),
 *      fp8 decoded by rows of 8 channels (C % 8 == 0; a NaN makes its row NaN,
 *      -0 becomes +0), truncation and flushing as npu_resize_bilinear_bf16;
 *   2. the 4-tap bilinear of npu_resize_bilinear_bf16 on x, kept in fp32;
 *   3. out = fp8(fp32(acc * ofm_quant_coeff)) truncated toward zero, finite
 *      magnitudes past 448 saturate to 448 (saturation mode); fp8 subnormals
 *      are kept. NaN -> 0x7F with the sign, infinity -> +-448 (unverified).
 * ifm_dequant_coeff / ofm_quant_coeff are RTP words 19 / 20 (ifm_scale, and
 * 1.0 / ofm_scale evaluated in double, both stored as fp32). */
int npu_resize_bilinear_fp8(void* stream, const void* in, void* out, npu_dtype t,
                            int64_t Hi, int64_t Wi, int64_t C, int64_t Ho,
                            int64_t Wo, const int32_t* idx_h, const int32_t* idx_w,
                            const void* wts, int64_t sv_wo, float ifm_dequant_coeff,
                            float ofm_quant_coeff);

/* resize_cubic_bf16: BF16 only. idx_h[Ho] / idx_w[Wo] are the absolute starts
 * of each output's 4-sample window (always four in-range samples: the host
 * folds edge clamping into the weights); coef_h[Ho][4] / coef_w[Wo][4] bf16
 * weights (the blob's coef_y / coef_x).
 *   pass 1 (H): t[h][x] = rne(sum_k coef_h[h][k] * in[idx_h[h] + k][x]);
 *   pass 2 (W): out[h][w] = rne(sum_k coef_w[w][k] * t[h][idx_w[w] + k]);
 * each sum a rounded product then three fused MACs in k order, each pass
 * rounded half to even to bf16 (the wrapper sets conv_even); results below the
 * smallest normal flush to a signed zero (unverified, as mllib's model). */
int npu_resize_cubic_bf16(void* stream, const void* in, void* out, int64_t Hi,
                          int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                          const int32_t* idx_h, const int32_t* idx_w,
                          const void* coef_h, const void* coef_w);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_RESIZE_H_
