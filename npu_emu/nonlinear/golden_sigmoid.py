# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for sigmoid / tanh: quantized on the hardware tanh table, quantized on an
interpolation table, and the direct bf16 / f16 / fp8 table variants."""

import numpy as np

from golden_float_cvt import (encode_float, f32_to_fp8, fp8_decode_identity, kFtzIn, kSat,
                              load_float, round_float, scalar_cast)
from golden_int_round import Rnd, float_to_int
from golden_mac import fma, mac_exact, mul
from golden_minmax import vclamp
from golden_nlf import lut_index, lut_interp, tanh_acc

F32 = np.float32
BF16, F16 = "bf16", "f16"
LUT_ELEMENTS = 128


def dq(x, zp, scale, t, rnd):
    """Dequantize: fma(x, scale, fp32(-zp * scale)) -> t."""
    nzs = mul(-F32(zp), F32(scale))
    return round_float(mac_exact(x, F32(scale), nzs), t, rnd, kFtzIn)


def q(y, zp, scale, out_t, rnd):
    """Quantize: fma(y, scale, zp) in fp32, rounded (rnd) and saturated."""
    return float_to_int(fma(y, F32(scale), F32(zp)), out_t, 0, rnd)


def interp(x, lut, k, t, rnd):
    """fma(x, slope, offset) of the entry of x; f16 rounds the slope to f16 first."""
    lut = np.asarray(lut, F32)
    return lut_interp(x, lut, lut_index(x, k, LUT_ELEMENTS, rnd), "f16" if t == F16 else "f32", rnd)


def sigmoid_lut_core(x, t, rnd, lut):
    """Sigmoid table on t values x (fp32), result as t values."""
    o = round_float(interp(x, lut, 8.0, t, rnd), t, rnd)
    o = np.where(x < F32(-8), F32(0), o)
    o = np.where(x >= F32(8), F32(1), o)
    return np.where(np.isnan(x), F32(np.nan), o).astype(F32)


def tanh_lut_core(x, t, rnd, lut):
    """Tanh table: bf16 selects -1 / 1 outside [-4, 4), f16 clamps the interpolant to
    [-1, 1]; result as t values."""
    o = round_float(interp(x, lut, 16.0, t, rnd), t, rnd)
    if t == F16:
        o = vclamp(o, F32(-1), F32(1))
    else:
        o = np.where(x < F32(-4), F32(-1), o)
        o = np.where(x >= F32(4), F32(1), o)
    return np.where(np.isnan(x), F32(np.nan), o).astype(F32)


def qdq_sigmoid(x, out_t, t, dq_zp, dq_s, q_zp, q_s):
    """0.5 - tanh(-v / 2) / 2, half away from zero."""
    y = dq(x, dq_zp, dq_s, t, Rnd.kHalfAway)
    th = tanh_acc(mul(y, F32(-0.5)), t)
    o = round_float(fma(th, F32(-0.5), F32(0.5)), BF16, Rnd.kHalfAway)
    return q(o, q_zp, q_s, out_t, Rnd.kHalfAway)


def qdq_tanh(x, out_t, t, dq_zp, dq_s, q_zp, q_s):
    y = dq(x, dq_zp, dq_s, t, Rnd.kHalfAway)
    return q(tanh_acc(y, t), q_zp, q_s, out_t, Rnd.kHalfAway)


def qdq_sigmoid_lut(x, out_t, dq_zp, dq_s, q_zp, q_s, lut):
    """bf16 core, half away from zero."""
    y = dq(x, dq_zp, dq_s, BF16, Rnd.kHalfAway)
    return q(sigmoid_lut_core(y, BF16, Rnd.kHalfAway, lut), q_zp, q_s, out_t, Rnd.kHalfAway)


def qdq_tanh_lut(x, out_t, dq_zp, dq_s, q_zp, q_s, lut):
    """f16 core, half to even."""
    y = dq(x, dq_zp, dq_s, F16, Rnd.kHalfEven)
    return q(tanh_lut_core(y, F16, Rnd.kHalfEven, lut), q_zp, q_s, out_t, Rnd.kHalfEven)


def sigmoid_lut_16(bits, t, lut):
    """bf16 / f16 bits in and out, half to even."""
    y = sigmoid_lut_core(load_float(bits, t), t, Rnd.kHalfEven, lut)
    return encode_float(y, t).astype(np.uint16)


def tanh_lut_16(bits, t, lut):
    """bf16 / f16 bits in and out, half to even."""
    y = tanh_lut_core(load_float(bits, t), t, Rnd.kHalfEven, lut)
    return encode_float(y, t).astype(np.uint16)


def sigmoid_lut_fp8(codes, lut, s_in, recip):
    """fp32 interpolation clamped to [0, 1], times recip, to fp8 without a +-448 clamp."""
    xr = mul(fp8_decode_identity(codes), F32(s_in))
    o = vclamp(interp(xr, lut, 8.0, "f32", Rnd.kHalfAway), F32(0), F32(1))
    return f32_to_fp8(mul(o, F32(recip)), Rnd.kHalfAway)


def tanh_lut_fp8(codes, t, lut, s_in, recip):
    """Scale into t, tanh table core in t, times recip (exact), +-448 clamp. The recip is
    cast half to even for bf16, half away from zero for f16."""
    xr = round_float(mul(fp8_decode_identity(codes), F32(s_in)), t, Rnd.kHalfAway)
    o = tanh_lut_core(xr, t, Rnd.kHalfAway, lut)
    v = mul(o, scalar_cast(recip, t, Rnd.kHalfAway))
    return f32_to_fp8(v, Rnd.kHalfAway, kSat)
