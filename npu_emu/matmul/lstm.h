/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Integer QDQ LSTM cell (a8w8 / a16w8) and gated relative position bias (grpb).
 * Tensors are logical (no padding). srs = shift right rounding half to even,
 * saturating (a negative shift is a left shift).
 */
#ifndef NPU_EMU_LSTM_H_
#define NPU_EMU_LSTM_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One LSTM step. x: len_x, h / c: len_h; w: 4 x len_h x len_x, r: 4 x len_h x
 * len_h, gates in the order i, o, f, g; b: 4 x len_h, zero-point corrected.
 * qdq: 9 words (shifts are signed bytes):
 *   [0] qx, [1] qh, [2] bytes xw / xs / hr / hs shift, [3] bytes xwq / hrq /
 *   linear_out shift, c_shft0, [4] bytes c_shft1, c_shft2, then c_zp (a8w8:
 *   int8 byte 2; a16w8: int16 bits 16-31), [5] c_scale (int16) | nonlinear_out
 *   shift << 16, [6] / [7] output h / c scale (int16) | zero point << 16
 *   (a8w8: int8; a16w8: uint16), [8] bytes output h / c shift.
 * x, h, c, w, r, h_out, c_out int8, b int32, qdq uint32. Works in int32 lanes.
 * len_x, len_h >= 1. */
int npu_lstm_a8w8(void* stream, const void* x, const void* h, const void* c, const void* w,
                  const void* r, const void* b, const void* qdq, void* h_out, void* c_out,
                  int64_t len_x, int64_t len_h);
/* As npu_lstm_a8w8 on uint16 activations, uint8 weights and a uint32 b, in uint32 lanes. */
int npu_lstm_a16w8(void* stream, const void* x, const void* h, const void* c, const void* w,
                   const void* r, const void* b, const void* qdq, void* h_out, void* c_out,
                   int64_t len_x, int64_t len_h);

/* ifm: M x K (a8w8: NPU_I8 / NPU_U8; a16w8: NPU_I16 / NPU_U16), wts: K x 8
 * (NPU_I8 / NPU_U8), columns 4g .. 4g + 3 summed for gate g; c0, c1, c2: M fp32
 * coefficients on the device (c1 may be null: the kernel's path without the activation sum,
 * which the wrapper does not build). x_g = fp32(gemm_g * c2 + c0) [+ sum(ifm row) * c1],
 * s_g = sigmoid(x_g);
 * out = fp32(s0 * fp32(scalar_lyr0 - s1 * scalar_head) + scalar_lyr1) as ofm_type
 * NPU_F16 / NPU_BF16. K >= 1; M may be 0 (nothing is launched). */
int npu_grpb_a8w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                  npu_dtype wts_type, const void* c0, const void* c1, const void* c2, void* out,
                  npu_dtype ofm_type, int64_t m, int64_t k, float scalar_head,
                  float scalar_lyr0, float scalar_lyr1);
int npu_grpb_a16w8(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                   npu_dtype wts_type, const void* c0, const void* c1, const void* c2, void* out,
                   npu_dtype ofm_type, int64_t m, int64_t k, float scalar_head,
                   float scalar_lyr0, float scalar_lyr1);

/* Test-only: the 4 x len_h fp32 gate inputs of the LSTM entries (a16: a16w8). */
int npu_test_lstm_gates(void* stream, int a16, const void* x, const void* h, const void* w,
                        const void* r, const void* b, const void* qdq, void* gates,
                        int64_t len_x, int64_t len_h);
/* Test-only: per row 4 fp32 values (pre-activation of gate 0, gate 1, sigmoid
 * of gate 0, gate 1); the grpb sigmoid alone on n fp32 inputs. */
int npu_test_grpb_stages(void* stream, const void* ifm, npu_dtype ifm_type, const void* wts,
                         npu_dtype wts_type, const void* c0, const void* c1, const void* c2,
                         void* stages, int64_t m, int64_t k);
int npu_test_grpb_sigmoid(void* stream, const void* x, void* out, int64_t n);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_LSTM_H_
