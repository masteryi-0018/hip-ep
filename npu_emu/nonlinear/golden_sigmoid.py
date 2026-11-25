# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for sigmoid / tanh: quantized on the hardware tanh table, quantized on an
interpolation table, and the direct bf16 / f16 / fp8 table variants."""

import numpy as np

from golden_act import sigmoid_tanh
from golden_bits import qnan
from golden_float_cvt import encode_float, kFtzIn, load_float, round_float, scalar_cast
from golden_mmac import fp8_decode_identity
from golden_int_round import Rnd
from golden_mac import mul_q
from golden_minmax import vclamp
from golden_nlf import lut_index, lut_interp, tanh16, tanh_acc
from golden_quant import dequant, dq_offset, fp8_in, fp8_out, quant

F32 = np.float32
BF16, F16 = "bf16", "f16"
LUT_ELEMENTS = 128


def _dq(x, zp, scale, t, rnd):
    """Dequantize into the compute type t."""
    return round_float(dequant(x, F32(scale), dq_offset(zp, F32(scale))), t, rnd, kFtzIn)


def _interp(x, lut, k, t, rnd):
    """fma(x, slope, offset) of the entry of x; f16 rounds the slope to f16 first."""
    lut = np.asarray(lut, F32)
    return lut_interp(x, lut, lut_index(x, k, LUT_ELEMENTS, rnd), "f16" if t == F16 else "f32", rnd)


def _sigmoid_lut_core(x, t, rnd, lut):
    """Sigmoid table on t values x (fp32), result as t values."""
    o = round_float(_interp(x, lut, 8.0, t, rnd), t, rnd)
    o = np.where(x < F32(-8), F32(0), o)
    o = np.where(x >= F32(8), F32(1), o)
    return np.where(np.isnan(x), qnan(), o).astype(F32)


def _tanh_lut_core(x, t, rnd, lut):
    """Tanh table: bf16 selects -1 / 1 outside [-4, 4), f16 clamps the interpolant to
    [-1, 1]; result as t values."""
    o = round_float(_interp(x, lut, 16.0, t, rnd), t, rnd)
    if t == F16:
        o = vclamp(o, F32(-1), F32(1))
    else:
        o = np.where(x < F32(-4), F32(-1), o)
        o = np.where(x >= F32(4), F32(1), o)
    return np.where(np.isnan(x), qnan(), o).astype(F32)


def sigmoid_a8(x, out_t, t, dq_zp, dq_sc, q_zp, q_sc):
    """0.5 - tanh(-v / 2) / 2, half away from zero."""
    o = sigmoid_tanh(_dq(x, dq_zp, dq_sc, t, Rnd.kHalfAway), t, Rnd.kHalfAway)
    return quant(o, F32(q_sc), F32(q_zp), out_t, Rnd.kHalfAway)


def tanh_a8(x, out_t, t, dq_zp, dq_sc, q_zp, q_sc):
    y = _dq(x, dq_zp, dq_sc, t, Rnd.kHalfAway)
    return quant(tanh_acc(y, t), F32(q_sc), F32(q_zp), out_t, Rnd.kHalfAway)


def qdq_sigmoid_lut_a8(x, out_t, dq_zp, dq_sc, q_zp, q_sc, lut):
    """bf16 core, half away from zero."""
    y = _dq(x, dq_zp, dq_sc, BF16, Rnd.kHalfAway)
    o = _sigmoid_lut_core(y, BF16, Rnd.kHalfAway, lut)
    return quant(o, F32(q_sc), F32(q_zp), out_t, Rnd.kHalfAway)


def qdq_tanh_lut_a8(x, out_t, dq_zp, dq_sc, q_zp, q_sc, lut):
    """f16 core, half to even."""
    y = _dq(x, dq_zp, dq_sc, F16, Rnd.kHalfEven)
    o = _tanh_lut_core(y, F16, Rnd.kHalfEven, lut)
    return quant(o, F32(q_sc), F32(q_zp), out_t, Rnd.kHalfEven)


def elementwise_sigmoid_lut_bf16(bits, t, lut):
    """bf16 / f16 bits in and out, half to even."""
    y = _sigmoid_lut_core(load_float(bits, t), t, Rnd.kHalfEven, lut)
    return encode_float(y, t).astype(np.uint16)


def elementwise_tanh_lut_bf16(bits, t, lut):
    """bf16 / f16 bits in and out, half to even."""
    y = _tanh_lut_core(load_float(bits, t), t, Rnd.kHalfEven, lut)
    return encode_float(y, t).astype(np.uint16)


def elementwise_sigmoid_lut_fp8(codes, lut, s_in, recip):
    """fp32 interpolation clamped to [0, 1], times recip, to fp8 without a +-448 clamp."""
    xr = mul_q(fp8_decode_identity(codes), F32(s_in))
    o = vclamp(_interp(xr, lut, 8.0, "f32", Rnd.kHalfAway), F32(0), F32(1))
    return fp8_out(o, F32(recip), Rnd.kHalfAway, 0)


def elementwise_tanh_lut_fp8(codes, t, lut, s_in, recip):
    """Scale into t, tanh table core in t, times recip (exact), +-448 clamp. The recip is
    cast half to even for bf16, half away from zero for f16."""
    xr = fp8_in(codes, F32(s_in), t, Rnd.kHalfAway)
    o = _tanh_lut_core(xr, t, Rnd.kHalfAway, lut)
    return fp8_out(o, scalar_cast(recip, t, Rnd.kHalfAway), Rnd.kHalfAway)


# the 16-bit variants share the models
sigmoid_a16 = sigmoid_a8
tanh_a16 = tanh_a8
qdq_sigmoid_lut_a16 = qdq_sigmoid_lut_a8
qdq_tanh_lut_a16 = qdq_tanh_lut_a8


def test_sigmoid_tanh16(bits, t):
    """The test hook: hardware tanh of t (bf16 / f16) codes, as bf16 bits."""
    return tanh16(bits, t)


test_sigmoid_tanh16.__test__ = False  # a model, not a test
