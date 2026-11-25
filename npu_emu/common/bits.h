/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * fp32 bit access and the NaN bit patterns shared by every model.
 *
 * Special values: qnan() is 0x7fc00000 (the NaN of the compare / select and table units),
 * default_nan() is 0xffc00000 (an invalid operation), neg_inf() is 0xff800000, quiet_nan(v)
 * sets the quiet bit of v and keeps its sign and payload, neg_f32 flips the sign bit of every
 * value, a NaN included [default].
 */
#ifndef NPU_EMU_BITS_H_
#define NPU_EMU_BITS_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace npu_emu {

__host__ __device__ inline uint32_t f32_bits(float x) { return __builtin_bit_cast(uint32_t, x); }
__host__ __device__ inline float bits_f32(uint32_t b) { return __builtin_bit_cast(float, b); }

__host__ __device__ inline float qnan() { return bits_f32(0x7fc00000u); }
__host__ __device__ inline float default_nan() { return bits_f32(0xffc00000u); }
__host__ __device__ inline float neg_inf() { return bits_f32(0xff800000u); }
__host__ __device__ inline float quiet_nan(float v) { return bits_f32(f32_bits(v) | 0x400000u); }

// An arithmetic negation leaves the sign of a negated NaN to the instruction selected.
__host__ __device__ inline float neg_f32(float x) { return bits_f32(f32_bits(x) ^ 0x80000000u); }

}  // namespace npu_emu

#endif  // NPU_EMU_BITS_H_
