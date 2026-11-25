/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * bf16 multi-head flash attention: one pass of online softmax over key blocks. bf16 tensors
 * are uint16 bit patterns.
 *
 * Per (batch, head) and query row, over key blocks of sv_n columns, from mx = -inf, l = 0,
 * O = 0:
 *   S      = Q K^T: bf16 MMAC chain over 32-deep chunks of head_size -> bf16
 *   s      = bf16(fp32(S * sc + mask)), sc = bf16_rne(fp32(scale * 1.4423828125)),
 *            mask 0 or -inf
 *   mx_new = max(mx_old, max_j s_j)                  bf16 compare, -0 < +0
 *   P_j    = exp2(fp32(s_j - mx_new))
 *   rowsum = bf16(MMAC chain of P against ones over 32-column chunks)
 *   mx_exp = exp2(bf16(fp32(mx_old - mx_new)))
 *   l      = bf16(fp32(mx_exp * l + rowsum))
 *   O      = bf16(MMAC chain of P against V over 32-row chunks, from bf16_mul(O, mx_exp))
 * and at the end out = bf16_mul(O, bf16(inv(l))).
 * exp2 and inv are the hardware tables. Vector fp32 -> bf16 conversions round toward -inf;
 * fp32 MAC / add, bf16_mul, the MMAC and bf16(scale) round half to even; results below 2^-126
 * are signed zeros. Not modelled: NaN / Inf operands of the MMAC.
 */
#ifndef NPU_EMU_MHA_H_
#define NPU_EMU_MHA_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* q [batch][q_heads][q_len][head_size], k / v [batch][kv_heads][kv_len][head_size],
 * out [batch][q_len][q_heads][head_size]; query head h reads kv head h / (q_heads / kv_heads).
 * scaling_factor: the fp32 bit pattern of the scale.
 * mask_1d = 0: with c = mask_M_start % 65536 % sv_n and b = mask_M_start / sv_n, key blocks
 * before b are unmasked, block b masks its columns >= c, later blocks are fully masked.
 * mask_1d = 1 (causal): query row r masks key column j where
 * (mask_M_start + r) % 65536 < j % 65536.
 * batch, q_heads or q_len 0: nothing to do, returns 0 once the other arguments are valid.
 * Returns -1 for null pointers and unless q_len % 64 == 0, head_size == 64,
 * sv_n in {128, 192, 256}, kv_len % sv_n == 0, q_heads % kv_heads == 0 and mask_1d in {0, 1}.
 * sv_n = 64 is rejected: the kernel's loops over sv_n / 32 and sv_n / 64 declare a minimum
 * trip count of 4, which sv_n = 64 does not reach. */
int npu_mha_bf16_3strms(void* stream, const void* q, const void* k, const void* v, void* out,
                        int64_t batch, int64_t q_heads, int64_t kv_heads,
                        int64_t q_len, int64_t head_size, int64_t kv_len, int64_t sv_n,
                        int mask_1d, uint32_t mask_M_start, uint32_t scaling_factor);

#ifdef __cplusplus
}
#endif

#endif /* NPU_EMU_MHA_H_ */
