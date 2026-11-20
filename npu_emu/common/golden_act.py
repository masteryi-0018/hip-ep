# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of act.h: the SigTanh, GeLU and SiLU cores and the QDQ output activation."""

import numpy as np

from golden_float_cvt import ftz, kFtzIn, round_float
from golden_int_round import Rnd, float_to_int
from golden_mac import fma, fneg, mul
from golden_minmax import vclamp, vmax
from golden_nlf import hw_exp2_bf20, quiet

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
    """Sigmoid / tanh up to the fp32 result: e = exp2(fma(x * x, k2, fma(-|x|, k1, k0)));
    tanh: copysign(fma(-2, e, 1), x); sigmoid: x > 0 ? fma(-1, e, 1) : e."""
    x = np.asarray(x, f32)
    p01 = fma(fneg(np.abs(x)), f32(k1), f32(k0))
    e = _ftz_exp2(fma(mul(x, x), f32(k2), p01))
    if is_tanh:
        t = fma(f32(-2.0), e, f32(1.0))
        return np.where(np.signbit(x), fneg(t), t).astype(f32)
    return np.where(x > 0, fma(f32(-1.0), e, f32(1.0)), e).astype(f32)


@quiet
def hw_exp_act_f32(x, k):
    """GeLU / SiLU up to the fp32 result: fma(xa, e, vmax(x, -0)), xa = -|x|,
    v = fma(xa, ka, kb), e = exp2(fma(-v, v, kd)) with a subnormal e flushed."""
    x = np.asarray(x, f32)
    xa = fneg(np.abs(x))
    v = fma(xa, k[0], k[1])
    e = _ftz_exp2(fma(fneg(v), v, k[2]))
    return fma(xa, e, vmax(x, f32(-0.0)))


def hw_gelu_f32(x):
    return hw_exp_act_f32(x, GELU_K)


def hw_silu_f32(x):
    return hw_exp_act_f32(x, SILU_K)


def sigtanh_tanh_bf16(x):
    """The SigTanh tanh chain on bf16 values; bf16 result rounded half away, an
    fp32-subnormal result -> signed zero."""
    t = hw_sigtanh_f32(x, True, SIGTANH_K0, TANH_K1, TANH_K2)
    return round_float(t, "bf16", Rnd.kHalfAway, kFtzIn)


# ---- QDQ output activation ----

ACT_NONE, ACT_CLIP, ACT_LRELU, ACT_PRELU, ACT_GELU = 0, 1, 2, 3, 4


def activate_f32(v, act, act0, act1, slope=0.0):
    """Activation on the fp32 accumulator (see act.h); NONE passes v through."""
    v = np.asarray(v, f32)
    if act == ACT_CLIP:
        return vclamp(v, f32(act0), f32(act1))
    if act == ACT_LRELU:
        return vmax(v, fma(v, act0, act1))
    if act == ACT_PRELU:
        slope = np.asarray(slope, f32)
        return np.where(v < f32(act0), fma(v, slope, fma(slope, fneg(f32(act0)), act0)), v).astype(f32)
    if act == ACT_GELU:
        return fma(hw_gelu_f32(v), act0, act1)
    return v


def activate_out(v, act, act0, act1, slope, out_t):
    """activate_f32, then round half to even and saturate to out_t."""
    return float_to_int(activate_f32(v, act, act0, act1, slope), out_t)
