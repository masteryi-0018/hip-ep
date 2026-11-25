/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Instruction sequences shared by the quantized kernels: dequantize, quantize, requantize,
 * the fp8 input / output scaling and the output activation of the QDQ integer kernels.
 *
 * Special values ([measured] against the NPU, [unverified] otherwise): the float steps are
 * mul_q / fma_q / mac_exact and follow their NaN rule [default]; the integer outputs saturate,
 * NaN -> 0 (float_to_int); requant_tail's u32 output tops out at 2^31 - 1 [measured]; fp8_in
 * takes the group NaN and the +0 of the identity decode [measured]; fp8_out saturates an
 * overflow or an infinity to the largest finite code unless called without kSat, a NaN stays
 * NaN [measured].
 */
#ifndef NPU_EMU_QUANT_H_
#define NPU_EMU_QUANT_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/act.h"
#include "common/bits.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/mmac.h"
#include "common/npu_emu.h"

namespace npu_emu {

// ---- dequantize / quantize / requantize ----

// Offset of the dequantize MAC: -zp * scale, the sign bit of zp flipped first.
__host__ __device__ inline float dq_offset(float zp, float scale) {
  return mul_q(neg_f32(zp), scale);
}

// x * scale + offset in one fp32 rounding; the caller rounds to its compute type. `flags`
// are the kFtz* flags of the MAC.
__device__ inline float dequant(int64_t x, float scale, float offset, int flags = 0) {
  return mac_exact(x, scale, offset, flags);
}

// v * scale + zp in one fp32 rounding (`flags`: the kFtz* flags of the MAC), times 2^shift,
// rounded with `rnd` and saturated to t.
__host__ __device__ inline int64_t quant(float v, float scale, float zp, npu_dtype t,
                                         Rnd rnd = kHalfEven, int shift = 0, int flags = 0) {
  return float_to_int(fma_ftz(v, scale, zp, flags), t, shift, rnd);
}

// x * c1 + c0 in one fp32 rounding, times 2^shift, rounded with `rnd` and saturated to t.
__device__ inline int64_t requant(int64_t x, float c1, float c0, npu_dtype t, int shift = 0,
                                  Rnd rnd = kHalfEven) {
  return float_to_int(mac_exact(x, c1, c0), t, shift, rnd);
}

// Integer output of the requantize tail: round half to even; 32-bit outputs saturate to
// int32, then U32 clamps at 0; narrower outputs saturate to their own range.
__host__ __device__ inline int64_t requant_tail(float v, npu_dtype t) {
  if (t == NPU_U32) {
    const int64_t r = float_to_int(v, NPU_I32, 0, kHalfEven);
    return r < 0 ? 0 : r;
  }
  return float_to_int(v, t, 0, kHalfEven);
}

// ---- fp8 input / output scaling ----

// Element i of an fp8 buffer of n through the identity decode, times scale, rounded to t.
__device__ inline float fp8_in(const uint8_t* p, int64_t i, int64_t n, float scale, npu_dtype t,
                               Rnd rnd, int flags = 0) {
  return round_float(mul_q(fp8_decode_identity(p, i, n), scale), t, rnd, flags);
}

// y * recip as an fp8 code, saturating; flags = 0 leaves an overflow to the rounding (NaN).
__host__ __device__ inline uint8_t fp8_out(float y, float recip, Rnd rnd, int flags = kSat) {
  return f32_to_fp8(mul_q(y, recip), rnd, flags);
}

// ---- QDQ output activation ----

// Activation codes of the QDQ integer kernels (the kernels' template codes; RELU, RELU6 and
// CLAMP all run as CLIP with host-computed bounds).
constexpr int kQdqActNone = 0, kQdqActLrelu = 3, kQdqActGelu = 6, kQdqActPrelu = 9,
              kQdqActClip = 100;

// Activation on the fp32 accumulator v:
//   CLIP:  clamp to [act0, act1]
//   LRELU: max(v, fma(v, act0, act1))
//   PRELU: v < act0 ? fma(v, slope, fma(slope, -act0, act0)) : v (per-channel slope)
//   GELU:  fma(gelu(v), act0, act1), v the dequantized value
// Any other code passes v through.
__device__ inline float activate_f32(float v, int act, float act0, float act1,
                                     float slope = 0.0f) {
  switch (act) {
    case kQdqActClip:
      return vclamp(v, act0, act1);
    case kQdqActLrelu:
      return vmax(v, fma_q(v, act0, act1));
    case kQdqActPrelu:
      return v < act0 ? fma_q(v, slope, fma_q(slope, neg_f32(act0), act0)) : v;
    case kQdqActGelu:
      return fma_q(hw_gelu_f32(v), act0, act1);
    default:
      return v;
  }
}

// activate_f32, then round half to even and saturate to t.
__device__ inline int64_t activate_out(float v, int act, float act0, float act1, float slope,
                                       npu_dtype t) {
  return float_to_int(activate_f32(v, act, act0, act1, slope), t, 0, kHalfEven);
}

}  // namespace npu_emu

#endif  // NPU_EMU_QUANT_H_
