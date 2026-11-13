/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * npu_emu entry points for the AIE4 rotary position embedding wrappers:
 * rope_bf16_stream, rope_fp8_stream and rope_bfp16_stream. Conventions as in
 * npu_emu.h.
 *
 * Layout. x and out are [rows][seq_len][head_dim] with rows = batch *
 * num_heads, contiguous (the NPU's {K, BATCH, M} IFM stream order, its
 * 128-padded head dimension and its row chunking are not emulated). cos and
 * sin are device tables [seq_len][half] shared by all rows, with half =
 * rotary_dim / 2: one value per rotated pair, as the kernels read them from
 * the wts stream (the host computes them; the kernel does no trigonometry).
 *
 * Pairing. rotate_half (GPT-NeoX) pairs xn = x[j] with xr = x[j + half] for
 * j < half; interleave (GPT-J, bfp16 only) pairs xn = x[2j] with xr =
 * x[2j + 1]. With c = cos[m][j], s = sin[m][j]:
 *   out_n = xn * c - xr * s        (written where xn was)
 *   out_r = xn * s + xr * c        (written where xr was)
 * computed in fp32 (accfloat) in one of two ways, depending on the kernel
 * path:
 *   fused:    out_n = fma(-xr, s, fp32(xn * c)),  out_r = fma(xr, c, fp32(xn * s))
 *             (aie::mul, then aie::msc / aie::mac: one rounding for the
 *             product, one for the multiply-add)
 *   unfused:  out_n = fp32(fp32(xn * c) - fp32(xr * s)),
 *             out_r = fp32(fp32(xn * s) + fp32(xr * c))
 *             (four aie::mul, then aie::sub / aie::add)
 * then rounded half to even (conv_even) to the 2-byte element type. The two
 * differ only when a product is inexact in fp32, i.e. with fp32 cos / sin.
 *
 * Partial RoPE (rotary_emb_dim, bf16 / fp8 only). 0 means the full head
 * (rotary_dim = head_dim). Otherwise only x[0 : rotary_dim] is rotated
 * (rotate_half within that prefix, half = rotary_dim / 2) and x[rotary_dim :]
 * passes through unchanged (same arithmetic as the full path otherwise). The
 * kernel's partial path is compiled with EmbDimIsMul32 = true and rotates
 * 16-lane chunks only, so it leaves the last rotary_half % 16 pairs
 * undefined; it also indexes the row in the 128-padded scratch, so it only
 * maps to the logical tensor when head_dim is a multiple of 128. The entries
 * therefore accept rotary_emb_dim in {0, head_dim} or a multiple of 32 below
 * head_dim with head_dim % 128 == 0, and return -1 otherwise. The YAMLs fix
 * rotary_emb_dim = 0, so no regression case exercises the partial path:
 * partial RoPE is unverified on the NPU.
 *
 * Unverified on the NPU: NaN / infinity inputs, f16 overflow, fp32 or bf16
 * subnormals (the GPU keeps them), partial RoPE.
 */
#ifndef NPU_EMU_ROPE_H_
#define NPU_EMU_ROPE_H_

#include <stdint.h>

#include "npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* rope_bf16_stream: x / out are BF16 or F16 (`type`), cos / sin F32. Fused
 * rotation (core_vec_body), output rounded half to even to `type`.
 * head_dim % 4 == 0. */
int npu_rope_bf16_stream(void* stream, const void* x, npu_dtype type,
                         const float* cos, const float* sin, void* out,
                         int64_t rows, int64_t seq_len, int64_t head_dim,
                         int rotary_emb_dim);

/* rope_fp8_stream: x / out are fp8 (E4M3FN); cos / sin F32; `type` (BF16 or
 * F16) is the kernel's internal 2-byte precision (the wrapper's dtype).
 *   xd  = T(fp32(decode(x) * ifm_scale))      decode by groups of 8 along
 *                                             head_dim (NaN spreads to the
 *                                             group, -0 decodes as +0)
 *   y   = T(rotate(xd))                       fused, as rope_bf16_stream
 *   out = fp8(clamp(y * r, -448, 448))        r = T(1.0f / ofm_scale), the
 *                                             product exact in fp32, fp8
 *                                             rounding half to even
 * with T() rounding half to even to `type`; the tail of a partial RoPE goes
 * through the same xd -> out path unrotated. ifm_scale and ofm_scale are the
 * RTP words (fp32). head_dim % 64 == 0. */
int npu_rope_fp8_stream(void* stream, const void* x, npu_dtype type,
                        const float* cos, const float* sin, void* out,
                        int64_t rows, int64_t seq_len, int64_t head_dim,
                        int rotary_emb_dim, float ifm_scale, float ofm_scale);

/* rope_bfp16_stream: x / out are bfp16 blocks (8 consecutive elements of a
 * row per 9-byte block, see npu_emu.h), head_dim % 128 == 0; cos / sin are
 * BF16 or F32 (`wts_type`, the wrapper's wts_dtype); `interleave` selects the
 * pairing (0 rotate_half, 1 interleave, a separate kernel binary each). The
 * decoded input is exact in bf16 and in fp32. The rotation is
 *   unfused  when interleave, or rotate_half with head_dim == 128 (the
 *            register-direct paths, rope_fma_fn);
 *   fused    for rotate_half with head_dim > 128 (the scratch two-stage
 *            path, core_vec_body).
 * (With BF16 cos / sin all products are exact and the paths agree.) The
 * result is rounded half to even to bf16, then encoded to bfp16 with
 * mantissas rounding half to even (to_v64mxint8_ebs8 under conv_even) and
 * saturating at 127. The kernel ignores rotary_emb_dim (core_row_fn always
 * rotates the full head), so there is no partial RoPE here. */
int npu_rope_bfp16_stream(void* stream, const void* x, const void* cos,
                          const void* sin, npu_dtype wts_type, void* out,
                          int64_t rows, int64_t seq_len, int64_t head_dim,
                          int interleave);

#ifdef __cplusplus
}
#endif

#endif /* NPU_EMU_ROPE_H_ */
