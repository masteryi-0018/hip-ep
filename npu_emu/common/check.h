/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Argument validation helpers of the entry points: dtype predicates and null checks.
 *
 * Special values: is_int_bits takes any width other than 8 as 16 [default].
 */
#ifndef NPU_EMU_CHECK_H_
#define NPU_EMU_CHECK_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/npu_emu.h"

namespace npu_emu {

// F32 / F16 / BF16: the float types load_float reads as plain elements (not FP8).
__host__ __device__ inline bool is_plain_float(npu_dtype t) {
  return t == NPU_F32 || t == NPU_F16 || t == NPU_BF16;
}
__host__ __device__ inline bool is_small_int(npu_dtype t) {
  return t == NPU_I8 || t == NPU_U8 || t == NPU_I16 || t == NPU_U16;
}
__host__ __device__ inline bool is_i8(npu_dtype t) { return t == NPU_I8 || t == NPU_U8; }
__host__ __device__ inline bool is_i16(npu_dtype t) { return t == NPU_I16 || t == NPU_U16; }
__host__ __device__ inline bool is_int_bits(npu_dtype t, int bits) {
  return bits == 8 ? is_i8(t) : is_i16(t);
}
__host__ __device__ inline bool is_half_float(npu_dtype t) { return t == NPU_BF16 || t == NPU_F16; }
__host__ __device__ inline bool is_32(npu_dtype t) { return t == NPU_I32 || t == NPU_U32; }
__host__ __device__ inline bool is_64(npu_dtype t) { return t == NPU_I64 || t == NPU_U64; }

// Whether any of the required pointers is null.
template <typename... P>
inline bool any_null(P... p) {
  return (... || (p == nullptr));
}

// Arguments of an elementwise quantized entry: integer input and output of `bits` bits and
// both buffers, required also when there is nothing to process.
inline bool qdq_args_ok(const void* in, npu_dtype in_type, const void* out, npu_dtype out_type,
                        int bits, int64_t n) {
  return is_int_bits(in_type, bits) && is_int_bits(out_type, bits) && n >= 0 && !any_null(in, out);
}

// n elements (or rows) over C per-channel coefficients.
inline bool per_channel_ok(int64_t n, int64_t C) { return n >= 0 && C >= 1; }

}  // namespace npu_emu

#endif  // NPU_EMU_CHECK_H_
