/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Rotary position embedding (RoPE) entry points.
 */
#ifndef NPU_EMU_ROPE_H_
#define NPU_EMU_ROPE_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* x and out are [rows][seq_len][head_dim]; cos and sin are [seq_len][half] tables
 * shared by all rows, half = rotary_dim / 2. rotate_half pairs xn = x[j] with
 * xr = x[j + half]; interleave pairs xn = x[2j] with xr = x[2j + 1].
 *   fused:   out_n = fma(-xr, s, fp32(xn * c)), out_r = fma(xr, c, fp32(xn * s))
 *   unfused: out_n = fp32(fp32(xn * c) - fp32(xr * s)),
 *            out_r = fp32(fp32(xn * s) + fp32(xr * c))
 * then rounded half to even to the 2-byte element type.
 *
 * rotary_emb_dim: 0 or head_dim rotates the full head. Otherwise it must be a
 * multiple of 32 with head_dim % 128 == 0; x[0 : rotary_emb_dim] is rotated and
 * the rest passes through. Other values return -1. */

/* x / out of `type` (BF16 / F16), cos / sin F32. Fused. head_dim % 4 == 0. */
int npu_rope_bf16_stream(void* stream, const void* x, npu_dtype type,
                         const float* cos, const float* sin, void* out,
                         int64_t rows, int64_t seq_len, int64_t head_dim,
                         int rotary_emb_dim);

/* x / out fp8 (E4M3FN), cos / sin F32; T = `type` (BF16 / F16), half to even.
 *   xd  = T(fp32(decode(x) * ifm_scale)), decoded by groups of 8 along head_dim
 *   y   = T(rotate(xd)), fused; the unrotated tail uses y = xd
 *   out = fp8(clamp(y * r, -448, 448)), r = T(1.0f / ofm_scale), half to even
 * head_dim % 64 == 0. */
int npu_rope_fp8_stream(void* stream, const void* x, npu_dtype type,
                        const float* cos, const float* sin, void* out,
                        int64_t rows, int64_t seq_len, int64_t head_dim,
                        int rotary_emb_dim, float ifm_scale, float ofm_scale);

/* x / out are bfp16 blocks (8 consecutive elements of a row per 9-byte block);
 * each 64-value group of a row (8 blocks) is decoded aligned to its largest block
 * exponent E: values rounded half to even to multiples of 2^(E - 150), below 2^-126 +0,
 * all NaN if a block exponent is 255. cos / sin of wts_type (BF16 / F32); interleave
 * selects the pairing (0 rotate_half, 1 interleave). Unfused when interleave or
 * head_dim == 128, fused otherwise. The result is rounded half to even to bf16, then
 * encoded to bfp16 with mantissas rounding half to even. Always rotates the full head.
 * head_dim % 128 == 0. */
int npu_rope_bfp16_stream(void* stream, const void* x, const void* cos,
                          const void* sin, npu_dtype wts_type, void* out,
                          int64_t rows, int64_t seq_len, int64_t head_dim,
                          int interleave);

#ifdef __cplusplus
}
#endif

#endif /* NPU_EMU_ROPE_H_ */
