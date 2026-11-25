# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of act.h: the SigTanh, GeLU and SiLU cores, the tanh-unit sigmoid and the bf16
ReLU family."""

import numpy as np

from golden_bits import neg_f32, quiet
from golden_float_cvt import acc_to_bf16, ftz, round_float
from golden_mac import fma_q, msc_q, mul16_f32, mul_q
from golden_minmax import vclamp, vmax
from golden_nlf import hw_exp2_bf20, tanh_acc

f32 = np.float32


def _ftz_exp2(p):
    return ftz(hw_exp2_bf20(p))


# ---- float activation cores ----

GELU_K = (f32(0.750399888), f32(-0.729944110), f32(894.533813))  # ka, kb, kd
SILU_K = (f32(0.329241553), f32(-1.227518206), f32(895.507801))
SIGTANH_K0 = f32(894.00097656)
TANH_K1, TANH_K2 = f32(1.53878212), f32(-0.49819374)
SIGMOID_K1, SIGMOID_K2 = f32(0.76842356), f32(-0.12469438)


@quiet
def hw_sigtanh_f32(x, is_tanh, k0, k1, k2):
    """Sigmoid / tanh up to the fp32 result: e = exp2(fma_q(x * x, k2, fma_q(-|x|, k1, k0)));
    tanh: copysign(fma_q(-2, e, 1), x); sigmoid: x > 0 ? fma_q(-1, e, 1) : e."""
    x = np.asarray(x, f32)
    p01 = fma_q(neg_f32(np.abs(x)), f32(k1), f32(k0))
    e = _ftz_exp2(fma_q(mul_q(x, x), f32(k2), p01))
    if is_tanh:
        t = fma_q(f32(-2.0), e, f32(1.0))
        return np.where(np.signbit(x), neg_f32(t), t).astype(f32)
    return np.where(x > 0, fma_q(f32(-1.0), e, f32(1.0)), e).astype(f32)


@quiet
def hw_exp_act_f32(x, k):
    """GeLU / SiLU up to the fp32 result: fma_q(xa, e, vmax(x, -0)), xa = -|x|,
    v = fma_q(xa, ka, kb), e = exp2(fma_q(-v, v, kd)) with a subnormal e flushed."""
    x = np.asarray(x, f32)
    xa = neg_f32(np.abs(x))
    v = fma_q(xa, k[0], k[1])
    e = _ftz_exp2(msc_q(v, v, k[2]))
    return fma_q(xa, e, vmax(x, f32(-0.0)))


def hw_gelu_f32(x):
    return hw_exp_act_f32(x, GELU_K)


def hw_silu_f32(x):
    return hw_exp_act_f32(x, SILU_K)


def sigtanh_tanh_bf16(x):
    """The SigTanh tanh chain on bf16 values; bf16 result rounded half away, an
    fp32-subnormal result -> signed zero."""
    t = hw_sigtanh_f32(x, True, SIGTANH_K0, TANH_K1, TANH_K2)
    return acc_to_bf16(t)


# ---- tanh-unit sigmoid ----

def sigmoid_tanh(v, t, rnd):
    """Sigmoid through the tanh table of t ("bf16" / "f16"): 0.5 - 0.5 * tanh(-v / 2) in one
    rounding, then bf16 with rnd."""
    return round_float(fma_q(tanh_acc(mul_q(v, f32(-0.5)), t), f32(-0.5), f32(0.5)), "bf16", rnd)


# ---- bf16 ReLU family ----

def activate_bf16(v, act, alpha):
    """Activation on bf16 values v: 1 ReLU, 2 ReLU6, 3 leaky ReLU with the 16-bit multiply
    v * alpha; any other code passes v through."""
    v = np.asarray(v, f32)
    if act == 1:
        return vmax(v, f32(0.0))
    if act == 2:
        return vclamp(v, f32(0.0), f32(6.0))
    if act == 3:
        return vmax(v, mul16_f32(v, alpha, "bf16"))
    return v
