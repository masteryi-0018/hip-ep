/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Activations built on the function units and the vector datapath: the SigTanh, GeLU and
 * SiLU cores, and the output activation of the QDQ integer kernels.
 *
 * Special values: NaN inputs are not special-cased; the results are those of the units and
 * the multiply-adds they are built from [unverified]; the 1 - e of SigTanh takes the NaN of e
 * (fma_q) [default].
 */
#ifndef NPU_EMU_ACT_H_
#define NPU_EMU_ACT_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/nlf.h"
#include "common/npu_emu.h"

namespace npu_emu {

// ---- float activation cores (built on the function units) ----

// Sigmoid / tanh up to the fp32 result:
//   e = exp2(fma(x * x, k2, fma(-|x|, k1, k0)))
//   tanh: copysign(fma(-2, e, 1), x);  sigmoid: x > 0 ? fma(-1, e, 1) : e.
// sigmoid(+-0) = e(0). NaN is not special-cased.
constexpr float kSigTanhK0 = 894.00097656f;
constexpr float kTanhK1 = 1.53878212f, kTanhK2 = -0.49819374f;
constexpr float kSigmoidK1 = 0.76842356f, kSigmoidK2 = -0.12469438f;

__device__ inline float hw_sigtanh_f32(float x, bool is_tanh, float k0, float k1, float k2) {
  const float p01 = fma_q(-fabsf(x), k1, k0);
  const float e = ftz(hw_exp2_bf20(fma_q(mul_q(x, x), k2, p01)));
  if (!is_tanh) return x > 0.0f ? fma_q(-1.0f, e, 1.0f) : e;
  const float t = fma_q(-2.0f, e, 1.0f);
  return signbit(x) ? __uint_as_float(__float_as_uint(t) ^ 0x80000000u) : t;
}

// The SigTanh tanh chain on a bf16 value; bf16 result rounded half away, an fp32-subnormal
// result -> signed zero.
__device__ inline float sigtanh_tanh_bf16(float x) {
  const float t = hw_sigtanh_f32(x, true, kSigTanhK0, kTanhK1, kTanhK2);
  return bf16_to_f32(f32_to_bf16(t, kHalfAway, kFtzIn));
}

// GeLU / SiLU up to the fp32 result:
//   z = fma(xa, e, vmax(x, -0)), xa = -|x|, v = fma(xa, ka, kb), e = exp2(fma(-v, v, kd))
// with a subnormal e flushed; GeLU for a finite x.
struct ExpActK {
  float ka, kb, kd;
};
constexpr ExpActK kGeluK{0.750399888f, -0.729944110f, 894.533813f};
constexpr ExpActK kSiluK{0.329241553f, -1.227518206f, 895.507801f};

__device__ inline float hw_exp_act_f32(float x, ExpActK k) {
  const float xa = -fabsf(x);
  const float v = fma_q(xa, k.ka, k.kb);
  const float e = ftz(hw_exp2_bf20(fma_q(-v, v, k.kd)));
  return fma_q(xa, e, vmax(x, -0.0f));
}
__device__ inline float hw_gelu_f32(float x) { return hw_exp_act_f32(x, kGeluK); }
__device__ inline float hw_silu_f32(float x) { return hw_exp_act_f32(x, kSiluK); }

// ---- QDQ output activation ----

// Activation on the fp32 accumulator: CLIP / LRELU / PRELU (per-channel slope) / GELU (v is
// the dequantized value); NONE passes v through.
__device__ inline float activate_f32(float v, npu_activation act, float act0, float act1,
                                     float slope = 0.0f) {
  switch (act) {
    case NPU_ACT_CLIP:
      return vclamp(v, act0, act1);
    case NPU_ACT_LRELU:
      return vmax(v, fma_q(v, act0, act1));
    case NPU_ACT_PRELU:
      return v < act0 ? fma_q(v, slope, fma_q(slope, -act0, act0)) : v;
    case NPU_ACT_GELU:
      return fma_q(hw_gelu_f32(v), act0, act1);
    default:
      return v;
  }
}

// activate_f32, then round half to even and saturate to t.
__device__ inline int64_t activate_out(float v, npu_activation act, float act0, float act1,
                                       float slope, npu_dtype t) {
  return float_to_int(activate_f32(v, act, act0, act1, slope), t, 0, kHalfEven);
}

}  // namespace npu_emu

#endif  // NPU_EMU_ACT_H_
