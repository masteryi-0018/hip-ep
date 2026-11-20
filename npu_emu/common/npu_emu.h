/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Shared types of the npu_emu C API. All pointers are device pointers; n
 * counts elements. Functions are asynchronous on `stream` (a hipStream_t, may
 * be null) and return a hipError_t value (0 on success) or -1 for unsupported
 * arguments.
 */
#ifndef NPU_EMU_H_
#define NPU_EMU_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NPU_F32 = 0,
  NPU_F16 = 1,
  NPU_BF16 = 2,
  NPU_FP8 = 3, /* E4M3FN */
  NPU_I8 = 4,
  NPU_U8 = 5,
  NPU_I16 = 6,
  NPU_U16 = 7,
  NPU_I32 = 8,
  NPU_U32 = 9,
  NPU_I64 = 10,
  NPU_U64 = 11,
  NPU_BFP16 = 12, /* packed blocks: 8 values in 9 bytes (block_fp.h) */
  NPU_MX6 = 13,   /* packed blocks: 16 values in 12 bytes (block_fp.h) */
} npu_dtype;

/* Coefficients of the qdq_gridsample_nearest entries; ofm_scale = 1 / s_out. */
typedef struct {
  int32_t grid_zp;
  float grid_scale;
  int32_t ifm_zp;
  float ifm_scale;
  int32_t ofm_zp;
  float ofm_scale;
} npu_gridsample_qdq;

/* Common to the QDQ integer kernels. Integer sums are exact; mac(x, c, a) = fp32(x * c + a)
 * with the integer x exact, fma the same on an fp32 x; the result is rounded half to even
 * and saturated to the output type. The coefficients are fp32 words, not ONNX scales:
 * c2 = s_in * s_w / s_out, c1 = -c2 * zp_w (activation-sum term), c3 = -c2 * zp_in
 * (weight-sum term), c0 the constant term.
 * GEMM: A is M x K, B is K x N, out M x N. Convolution: ifm H x W x Ci, wts Kh x Kw x Ci x Co,
 * out Ho x Wo x Co, Ho = (H - Kh) / stride_h + 1 (no padding).
 *
 * Activation on the accumulator v: CLIP clamps to [act0, act1]; LRELU is
 * max(v, fma(v, act0, act1)); PRELU (convolutions) is
 * v < act0 ? fma(v, slope, fma(slope, -act0, act0)) : v with a per-channel slope;
 * GELU (GEMMs) is fma(gelu(v), act0, act1) on dequantized coefficients. */
typedef enum {
  NPU_ACT_NONE = 0,
  NPU_ACT_CLIP = 1,
  NPU_ACT_LRELU = 2,
  NPU_ACT_PRELU = 3,
  NPU_ACT_GELU = 4,
} npu_activation;

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_H_
