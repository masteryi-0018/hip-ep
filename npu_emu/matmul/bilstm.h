/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * One direction of a bf16 LSTM with table-interpolated gate activations; a
 * bidirectional LSTM is two calls (forward and reverse) on the same x.
 */
#ifndef NPU_EMU_BILSTM_H_
#define NPU_EMU_BILSTM_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* x: seq x I, h0 / c0: H, w: 4H x I, r: 4H x H (gate order i, o, f, c), b: 8H
 * (the W biases, then the R biases), all bf16. lut: 512 fp32 words, (offset,
 * slope) pairs; segment s reads words (s / 8) * 32 + (s % 8) * 2 and the next.
 * y: seq x H bf16 in ascending timestep order (also for reverse), c_out: H bf16.
 * I must be a multiple of 4, H a multiple of 16.
 * sigmoid(v) = f16(fma(v, slope[s], offset[s])), s = clamp(floor(bf16(v * 128 +
 * 1024)), 0, 2032) >> 4; tanh(v) = f16(f16(2 * sigmoid(f16(2 v))) - 1).
 * z = f16(zx + zh); c = fp32(fp32(f * c) + fp32(i * g)), kept fp32 across
 * timesteps; h = bf16(o * tanh(f16(c))). Conversions round half to even. */
int npu_bidirectional_lstm_bf16(void* stream, const uint16_t* x, const uint16_t* h0,
                                const uint16_t* c0, const uint16_t* w, const uint16_t* r,
                                const uint16_t* b, const float* lut, uint16_t* y,
                                uint16_t* c_out, int64_t seq, int64_t input_size,
                                int64_t hidden, int reverse);

/* Test-only: the gate sigmoid (or tanh when is_tanh) on n f16 inputs; out f16. */
int npu_test_bilstm_gate_lut(void* stream, const uint16_t* in, const float* lut, int is_tanh,
                             uint16_t* out, int64_t n);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_BILSTM_H_
