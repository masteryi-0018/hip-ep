/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Average pooling and global average pooling: bf16 / f16, fp8 and integer.
 *
 * in is [H][W][C]; a window pooling's out is [Ho][Wo][C] with
 * Ho = (H - kh) / sh + 1, Wo = (W - kw) / sw + 1 (no padding), a global
 * pooling's out is [C]. Float sums are sequences of fp32 additions from +0, the
 * window row by row. A float global pooling adds the pixels block by block
 * (blocks of sv_h x sv_w, W block fastest, row by row inside a block;
 * H % sv_h == 0, W % sv_w == 0). Argument errors return -1.
 * Special values: fp32 steps follow the NaN rules of common/mac.h [default].
 */
#ifndef NPU_EMU_AVGPOOL_H_
#define NPU_EMU_AVGPOOL_H_

#include <stdint.h>

#include "common/npu_emu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* in / out of `type` (BF16 or F16). acc = fma(x, factor, acc) per tap, the taps
 * alternating between two accumulators that are added at the end; rounded half
 * to even to `type`. factor is rounded to `type`. fp32 subnormals (inputs and
 * intermediates) become zeros of their sign. kh * kw >= 2. */
int npu_avgpool_bf16(void* stream, const void* in, npu_dtype type, void* out,
                     int64_t H, int64_t W, int64_t C, int kh, int kw, int sh,
                     int sw, float factor);
/* BF16 only, as npu_avgpool_bf16; a 1 x 1 window copies the input. */
int npu_avgpool_bf16_c32(void* stream, const void* in, void* out, int64_t H,
                         int64_t W, int64_t C, int kh, int kw, int sh, int sw,
                         float factor);
/* in of `type` (BF16 or F16), out [C] = rne(fp32(s * factor)) with s the fp32
 * sum; a subnormal running sum becomes +0, a subnormal product a zero of its
 * sign. */
int npu_globalavgpool_bf16(void* stream, const void* in, npu_dtype type,
                           void* out, int64_t H, int64_t W, int64_t C,
                           int sv_h, int sv_w, float factor);
/* BF16 only, otherwise as npu_globalavgpool_bf16. */
int npu_globalavgpool_bf16_c32(void* stream, const void* in, void* out,
                               int64_t H, int64_t W, int64_t C, int sv_h,
                               int sv_w, float factor);

/* in / out FP8 (E4M3FN). The window sum is taken in passes of kh_sv rows
 * (kh % kh_sv == 0), each from +0, folded as total = pass + total. Then
 * v = fma(total, c2[c], c0[c]); activation != 0: v = min(max(v, act_lo),
 * act_hi); out = fp8(v) half to even, overflow gives NaN. c0 / c2: device
 * arrays [C]. A NaN input makes its group of 8 channels NaN. */
int npu_avgpool_fp8(void* stream, const void* in, void* out, int64_t H,
                    int64_t W, int64_t C, int kh, int kw, int sh, int sw,
                    int kh_sv, const float* c0, const float* c2,
                    int activation, float act_lo, float act_hi);
/* in FP8, out FP8 [C] = fp8(fma(s, c1, c0)) with s the fp32 sum. */
int npu_globalavgpool_fp8(void* stream, const void* in, void* out, int64_t H,
                          int64_t W, int64_t C, int sv_h, int sv_w, float c0,
                          float c1);

/* s = exact integer window sum, v = fma(s, c1, c0); a8 with activation != 0:
 * v = min(max(v, act_lo), act_hi); out = sat(rne(v * 2^shift)) to out_type.
 * in_type / out_type: I8 / U8 (a8) or I16 / U16 (a16). */
int npu_qdq_avgpool_a8(void* stream, const void* in, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t H, int64_t W,
                       int64_t C, int kh, int kw, int sh, int sw, float c0,
                       float c1, int shift, int activation, float act_lo,
                       float act_hi);
int npu_qdq_avgpool_a16(void* stream, const void* in, npu_dtype in_type,
                        void* out, npu_dtype out_type, int64_t H, int64_t W,
                        int64_t C, int kh, int kw, int sh, int sw, float c0,
                        float c1, int shift);
/* out [C] = sat(rne(fma(s, c1, c0) * 2^shift)) with s the exact integer sum
 * over H x W (int32). a16 reads U16 inputs as int16. */
int npu_qdq_globalavgpool_a8(void* stream, const void* in, npu_dtype in_type,
                             void* out, npu_dtype out_type, int64_t H,
                             int64_t W, int64_t C, float c0, float c1,
                             int shift);
int npu_qdq_globalavgpool_a16(void* stream, const void* in, npu_dtype in_type,
                              void* out, npu_dtype out_type, int64_t H,
                              int64_t W, int64_t C, float c0, float c1,
                              int shift);

/* out = sat(rne((integer window sum * weight) / 2^shift)), the product in
 * int32. weight <= 0xffff, 0 <= shift <= 31. in_type / out_type: I8 / U8. */
int npu_avgpool_xint8(void* stream, const void* in, npu_dtype in_type,
                      void* out, npu_dtype out_type, int64_t H, int64_t W,
                      int64_t C, int kh, int kw, int sh, int sw,
                      uint32_t weight, int shift);
/* npu_avgpool_xint8, then out = max(out, 0) when activation == 1 and out_type
 * is I8. activation: 0 or 1. */
int npu_avgpool_xint8_fused(void* stream, const void* in, npu_dtype in_type,
                            void* out, npu_dtype out_type, int64_t H,
                            int64_t W, int64_t C, int kh, int kw, int sh,
                            int sw, uint32_t weight, int shift,
                            int activation);

#ifdef __cplusplus
}
#endif

#endif  // NPU_EMU_AVGPOOL_H_
