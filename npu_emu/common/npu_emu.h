/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Shared types of the npu_emu C API. All pointers are device pointers; n
 * counts elements. Functions are asynchronous on `stream` (a hipStream_t, may
 * be null) and return a hipError_t value (0 on success) or -1 for unsupported
 * arguments. A null required pointer or a negative size is rejected (-1); a
 * call with no output elements returns 0 without launching. Optional pointers
 * (bias, ...) may be null.
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

/* Coefficients of the QDQ GridSample entries; ofm_scale = 1 / s_out. */
typedef struct {
  int32_t grid_zp;
  float grid_scale;
  int32_t ifm_zp;
  float ifm_scale;
  int32_t ofm_zp;
  float ofm_scale;
} npu_gridsample_qdq;

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_H_
