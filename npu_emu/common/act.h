/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Float activations built on the function units and the vector datapath: the SigTanh, GeLU
 * and SiLU cores, the tanh-unit sigmoid and the bf16 ReLU family.
 *
 * Special values: NaN inputs are not special-cased; the results are those of the units and
 * the multiply-adds they are built from [unverified]; the 1 - e of SigTanh takes the NaN of e
 * (fma_q) [default]; the ReLU family follows vmax / vclamp (a NaN gives 0x7fc00000, -0 orders
 * below +0) [measured].
 */
#ifndef NPU_EMU_ACT_H_
#define NPU_EMU_ACT_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "common/bits.h"
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
  const float p01 = fma_q(neg_f32(fabsf(x)), k1, k0);
  const float e = ftz(hw_exp2_bf20(fma_q(mul_q(x, x), k2, p01)));
  if (!is_tanh) return x > 0.0f ? fma_q(-1.0f, e, 1.0f) : e;
  const float t = fma_q(-2.0f, e, 1.0f);
  return signbit(x) ? neg_f32(t) : t;
}

// The SigTanh tanh chain on a bf16 value; bf16 result rounded half away, an fp32-subnormal
// result -> signed zero.
__device__ inline float sigtanh_tanh_bf16(float x) {
  const float t = hw_sigtanh_f32(x, true, kSigTanhK0, kTanhK1, kTanhK2);
  return acc_to_bf16(t);
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
  const float xa = neg_f32(fabsf(x));
  const float v = fma_q(xa, k.ka, k.kb);
  const float e = ftz(hw_exp2_bf20(msc_q(v, v, k.kd)));
  return fma_q(xa, e, vmax(x, -0.0f));
}
__device__ inline float hw_gelu_f32(float x) { return hw_exp_act_f32(x, kGeluK); }
__device__ inline float hw_silu_f32(float x) { return hw_exp_act_f32(x, kSiluK); }

// ---- tanh-unit sigmoid ----

// Sigmoid through the tanh table of t (BF16 / F16): 0.5 - 0.5 * tanh(-v / 2) in one rounding,
// then bf16 with `rnd`.
__device__ inline float sigmoid_tanh(float v, npu_dtype t, Rnd rnd) {
  return round_float(fma_q(tanh_acc(mul_q(v, -0.5f), t), -0.5f, 0.5f), NPU_BF16, rnd);
}

// ---- bf16 ReLU family ----

// Activation on a bf16 value v: 1 ReLU, 2 ReLU6, 3 leaky ReLU with the 16-bit multiply
// v * alpha; any other code passes v through.
__device__ inline float activate_bf16(float v, int act, float alpha) {
  switch (act) {
    case 1:
      return vmax(v, 0.0f);
    case 2:
      return vclamp(v, 0.0f, 6.0f);
    case 3:
      return vmax(v, mul16_f32(v, alpha, NPU_BF16));
    default:
      return v;
  }
}

}  // namespace npu_emu

#endif  // NPU_EMU_ACT_H_
