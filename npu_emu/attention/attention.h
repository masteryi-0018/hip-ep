/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Flash attention (online softmax over KV blocks) as the NPU computes it: bf16, uint16 / int16 /
 * uint8 QDQ and mxint8_ebs8.
 *
 * Logical contiguous tensors per head h < heads: Q [h][M][d], K [h][N][d], V [h][N][dv],
 * O [h][M][dv], mask [h % mask_heads][N]; M = q_len, N = kv_len.
 * Per Q block of br rows and KV block of bc keys: S = QK as bf16 plus masks, m' = max(m,
 * rowmax(S)), emd = exp2((m - m') * c), P = exp2((S - m') * c), l = emd * l + rowsum(P),
 * O = O * emd + PV; at the end out = O * inv(l). exp2 and inv are the hardware tables,
 * c = log2(e) as bf16 (mx: times the scale). Causal: a Q block visits the KV blocks from its
 * sliding window's first block up to the block that holds its first position (the state
 * entries: its last position). Grouped: the row at Q position p (as for causal) keeps the keys
 * [p / q_group_size * kv_group_size, + kv_group_size).
 * Not modelled: NaN / Inf inputs, rows whose scores are all -inf.
 */
#ifndef NPU_EMU_ATTENTION_H_
#define NPU_EMU_ATTENTION_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel parameters, named as in the kernel metadata. */
typedef struct {
  int32_t enable_qdq;               /* 0 (bf16 / mx), 1 (dequantized, bf16 math), 2 (integer
                                       GEMMs) */
  int32_t block_row_dim;            /* Br, rows per Q block: 1, 8, 32, 64 (mx also 2) */
  int32_t m_granularity;            /* 1, 8, 32, 64; q16 qdq 2: 16, u8 qdq 2: 64, mx also 2.
                                       Selects the u8 QK row-sum order */
  int32_t block_column_dim;         /* Bc, keys per KV block: a multiple of 64, <= 16320 */
  int32_t tile_k;                   /* 1..255; u8 qdq 2: PV in chunks of tile_k * 64 keys */
  int32_t k_seq_tail_value;         /* valid keys of the last KV block (0..Bc), 0 = all */
  int32_t mask_type;                /* 0 none, 1 causal, 2 mask stream, 7 grouped */
  int32_t enable_sliding_window;    /* 0 / 1: causal (mask_type 1) only, not with qdq 2 */
  int32_t enable_shuffle_transpose; /* 0 / 1: bf16, or q16 qdq 1; the V dequant rounding
                                       follows alt_round */
  int32_t mat_v_transpose;          /* mx: V given as V^T (1, 3); 3: also O * inv(l) rounded
                                       to bf16 before the encoding. bf16: 0, 1, 3; q16 / u8:
                                       0, 1 (V is logical) */
  int32_t variant_a;                /* 0 / 1; 1: masked scores round half to even */
  int32_t alt_round;                /* 0 / 1; 1: conversions round half to even, not ties
                                       away */
  int64_t sliding_window_size;      /* >= 0, < 2^32 */
  int64_t q_mask_offset;            /* causal position of row 0; 0..2^24 - 1 except for the
                                       state entries (relative to the first key) */
  int32_t q_shards;                 /* 0, 1, 2, 4, 6, 12. Causal: Q block sc starts at
                                       q_mask_offset + (q_shard + q_shards * (sc % q_mask_wrap))
                                       * Br; 0 = every row at q_mask_offset */
  int32_t q_shard;                  /* 0 <= q_shard < max(q_shards, 1) */
  int32_t q_mask_wrap;              /* 0..65535, 0 taken as 1 */
  uint16_t alpha_bf16;              /* softmax scale, bf16 bits */
  float matmul_qk_c[4];             /* matmul_qk_c0..c3 */
  float matmul_pv_c[4];             /* qdq 1: transpose_c0, transpose_c1 (V dequant);
                                       qdq 2: matmul_pv_c0..c3 */
  float p_c1;                       /* qdq 2: P quantization scale */
  uint16_t p_c0;                    /* qdq 2: P quantization offset */
  uint16_t ofm_c0;                  /* output offset, read as the output type */
  float ofm_c1;                     /* output scale */
  float mask_dequant_c0;            /* mask dequant: code * c0 + c1 */
  float mask_dequant_c1;
  int32_t q_group_size;             /* grouped mask: Q positions per group (> 0) */
  int32_t kv_group_size;            /* grouped mask: keys per group (> 0) */
} npu_flash_attention_cfg;

/* npu_flash_attention_*: one entry per data type family of the kernel flash_attention.
 * Sizes: heads, q_len (M, a multiple of Br), kv_len (N, a multiple of Bc), head_dim (d),
 * v_head_dim (dv), mask_heads (>= 1, heads of the mask). heads or q_len 0: nothing to do,
 * returns 0 once the other arguments are valid. Every pointer but mask is required (also for
 * empty work); mask is required for mask_type 2 and ignored otherwise. mask_type: bf16 / q16
 * 0, 1, 2; u8 0, 1, 2, 7; mx 0, 1, 7. Arguments outside the ranges above return -1. */

/* bf16 Q / K / V / O (bits); mask: bf16 values added as is. enable_qdq 0. */
int npu_flash_attention_bf16(void* stream, const void* q, const void* k, const void* v,
                             const void* mask, void* out, int64_t heads, int64_t q_len,
                             int64_t kv_len, int64_t head_dim, int64_t v_head_dim,
                             int64_t mask_heads, const npu_flash_attention_cfg* cfg);

/* q / k / v / out of dtype (U16, or I16 with enable_qdq 1 and mask_type 0), enable_qdq 1 or 2,
 * Br 1 (qdq 1 only), 32 or 64; m_granularity 16 for qdq 2. mask: u16 codes, dequantized as
 * bf16(fp32(code * mask_dequant_c0 + mask_dequant_c1)). */
int npu_flash_attention_q16(void* stream, const void* q, npu_dtype dtype, const void* k,
                            const void* v, const void* mask, void* out, int64_t heads,
                            int64_t q_len, int64_t kv_len, int64_t head_dim, int64_t v_head_dim,
                            int64_t mask_heads, const npu_flash_attention_cfg* cfg);

/* uint8 QDQ, enable_qdq 1 or 2; mask as npu_flash_attention_q16. */
int npu_flash_attention_u8(void* stream, const void* q, const void* k, const void* v,
                           const void* mask, void* out, int64_t heads, int64_t q_len,
                           int64_t kv_len, int64_t head_dim, int64_t v_head_dim,
                           int64_t mask_heads, const npu_flash_attention_cfg* cfg);

/* mxint8_ebs8, packed 9 bytes per block of 8 ([exponent, m0..m7]): Q [h][M][d/8 blocks],
 * K [h][N][d/8], V [h][N][dv/8] or (mat_v_transpose) [h][dv][N/8], O [h][M][dv/8]. The
 * exponents of all-zero Q / K blocks take part in the accumulation. head_dim % 64 == 0 (the
 * padded width), v_head_dim % 8 == 0, enable_qdq 0. */
int npu_flash_attention_mx(void* stream, const void* q, const void* k, const void* v, void* out,
                           int64_t heads, int64_t q_len, int64_t kv_len, int64_t head_dim,
                           int64_t v_head_dim, const npu_flash_attention_cfg* cfg);

/* npu_flash_attention_state_*: bf16 causal attention (enable_qdq 0, mask_type 1,
 * k_seq_tail_value 0) continuing a running state per Q row: m, l (bf16 bits) and the
 * numerator U = unnormalized O. KV blocks no row of a Q block sees leave its state as is; a
 * fully masked row keeps m = -inf, l = 0. Final: out = bf16(U * inv(l)), 0 for l = 0.
 * Positions are relative to the first key: q_mask_offset = (Q origin) - (KV origin). The
 * state buffers may alias. */

/* m_in / l_in [H][M], u_in [H][M][dv] bf16 (U widened to fp32); writes m_out, l_out and
 * out = output_final ? O : bf16(U). output_final is 0 or 1. */
int npu_flash_attention_state_io(void* stream, const void* q, const void* k, const void* v,
                                 const void* m_in, const void* l_in, const void* u_in,
                                 void* m_out, void* l_out, void* out, int64_t heads,
                                 int64_t q_len, int64_t kv_len, int64_t head_dim,
                                 int64_t v_head_dim, const npu_flash_attention_cfg* cfg,
                                 int output_final);

/* State kept by the caller across calls: m / l [H][M] bf16 bits, u [H][M][dv] fp32.
 * flags bit 1: start from m = -inf, l = U = 0; bit 0: write O to out (else out is untouched
 * and may be null). Other bits return -1. */
int npu_flash_attention_state_cache(void* stream, const void* q, const void* k, const void* v,
                                    void* m, void* l, void* u, void* out, int64_t heads,
                                    int64_t q_len, int64_t kv_len, int64_t head_dim,
                                    int64_t v_head_dim, const npu_flash_attention_cfg* cfg,
                                    int flags);

#ifdef __cplusplus
}
#endif

#endif /* NPU_EMU_ATTENTION_H_ */
