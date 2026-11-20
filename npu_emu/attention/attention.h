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
 * sliding window's first block up to the block that holds its first position.
 * Not modelled: NaN / Inf inputs, rows whose scores are all -inf.
 */
#ifndef NPU_EMU_ATTENTION_H_
#define NPU_EMU_ATTENTION_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Parameters. */
typedef struct {
  int64_t heads;
  int64_t q_len;               /* M rows per head, a multiple of br */
  int64_t kv_len;              /* N keys per head, a multiple of bc */
  int64_t head_dim;            /* d; mx: the padded width, % 64 == 0 */
  int64_t v_head_dim;          /* dv; mx: % 8 == 0 */
  int64_t mask_heads;          /* heads of the mask (>= 1) */
  int64_t sliding_window_size;
  int64_t q_offset;            /* causal position of row 0 */
  int32_t br;                  /* rows per Q block */
  int32_t m_granularity;       /* selects the u8 QK row-sum order */
  int32_t bc;                  /* keys per KV block, % 64 == 0 */
  int32_t k_tile;              /* u8 qdq 2: PV in chunks of k_tile * 64 keys */
  int32_t kv_valid_last;       /* valid keys of the last KV block, 0 = all */
  int32_t mask_type;           /* 0 none, 1 causal, 2 mask stream */
  int32_t sliding;             /* sliding window on */
  int32_t q_shards;            /* causal: Q block sc starts at q_offset + (q_shard + q_shards *
                                  (sc % q_wrap)) * br; 0 = every row at q_offset */
  int32_t q_shard;
  int32_t q_wrap;              /* 0 taken as 1 */
  int32_t qdq;                 /* 0 (bf16 / mx), 1 (dequantized, bf16 math), 2 (integer GEMMs) */
  int32_t variant_a;           /* 1: masked scores round half to even */
  int32_t shuffle;             /* q16 qdq 1: the V dequant rounding follows alt_round */
  int32_t alt_round;           /* 1: conversions round half to even instead of ties away */
  int32_t v_transposed;        /* mx: V given as V^T */
  uint32_t alpha_bf16;         /* softmax scale, bf16 bits */
  float qk_c[4];               /* QK QDQ coefficients c0..c3 */
  float pv_c[4];               /* qdq 1: V dequant c0, c1; qdq 2: PV c0..c3 */
  float p_c1;                  /* qdq 2: P quantization scale */
  int32_t p_c0;                /* qdq 2: P quantization offset */
  float ofm_c1;                /* output scale */
  int32_t ofm_c0;              /* output offset (u8: & 0xff) */
  float mask_c0, mask_c1;      /* mask dequant: code * mask_c0 + mask_c1 */
} npu_flash_attention_cfg;

/* npu_flash_attention_*: one entry per data type family of the kernel flash_attention. */

/* bf16 Q / K / V / O (bits); mask: bf16 values added as is (or null). qdq 0. */
int npu_flash_attention_bf16(void* stream, const npu_flash_attention_cfg* cfg, const uint16_t* q,
                             const uint16_t* k, const uint16_t* v, const uint16_t* mask,
                             uint16_t* out);

/* uint16 / int16 (type NPU_U16 / NPU_I16) QDQ, qdq 1 or 2; mask: u16 codes, dequantized as
 * bf16(fp32(code * mask_c0 + mask_c1)), or null. */
int npu_flash_attention_q16(void* stream, const npu_flash_attention_cfg* cfg, npu_dtype type,
                            const void* q, const void* k, const void* v, const uint16_t* mask,
                            void* out);

/* uint8 QDQ, qdq 1 or 2; mask as npu_flash_attention_q16. */
int npu_flash_attention_u8(void* stream, const npu_flash_attention_cfg* cfg, const uint8_t* q,
                           const uint8_t* k, const uint8_t* v, const uint16_t* mask,
                           uint8_t* out);

/* mxint8_ebs8, packed 9 bytes per block of 8 ([exponent, m0..m7]): Q [h][M][d/8 blocks],
 * K [h][N][d/8], V [h][N][dv/8] or (v_transposed) [h][dv][N/8], O [h][M][dv/8]. The exponents
 * of all-zero Q / K blocks take part in the accumulation. mask_type 0 / 1. */
int npu_flash_attention_mx(void* stream, const npu_flash_attention_cfg* cfg, const uint8_t* q,
                           const uint8_t* k, const uint8_t* v, uint8_t* out);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_ATTENTION_H_
