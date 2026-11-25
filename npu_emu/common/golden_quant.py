# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of quant.h: dequantize, quantize, requantize, the fp8 input / output scaling
and the output activation of the QDQ integer kernels."""

import numpy as np

from golden_act import hw_gelu_f32
from golden_bits import neg_f32
from golden_float_cvt import f32_to_fp8, float_to_int, kSat, round_float
from golden_int_round import Rnd
from golden_mac import fma_ftz, fma_q, mac_exact, mul_q
from golden_minmax import vclamp, vmax
from golden_mmac import fp8_decode_identity

F32 = np.float32


def dq_offset(zp, scale):
    """Offset of the dequantize MAC: -zp * scale, the sign bit of zp flipped first."""
    return mul_q(neg_f32(np.asarray(zp, F32)), scale)


def dequant(x, scale, offset, flags=0):
    """x * scale + offset in one fp32 rounding; the caller rounds to its compute type. flags
    are the kFtz* flags of the MAC."""
    return mac_exact(x, scale, offset, flags)


def quant(v, scale, zp, t, rnd=Rnd.kHalfEven, shift=0, flags=0):
    """v * scale + zp in one fp32 rounding (flags: the kFtz* flags of the MAC), times 2^shift,
    rounded with rnd, saturated to t."""
    return float_to_int(fma_ftz(v, scale, zp, flags), t, shift, rnd)


def requant(x, c1, c0, t, shift=0, rnd=Rnd.kHalfEven):
    """x * c1 + c0 in one fp32 rounding, times 2^shift, rounded with rnd, saturated to t."""
    return float_to_int(mac_exact(x, c1, c0), t, shift, rnd)


def requant_tail(v, t):
    """Round half to even; 32-bit outputs saturate to int32, then u32 clamps at 0."""
    if t == "u32":
        return np.maximum(float_to_int(v, "i32"), 0)
    return float_to_int(v, t)


def fp8_in(c, scale, t, rnd, flags=0, row=None):
    """fp8 codes through the identity decode (groups of 8 from the start of each row of `row`
    elements), times scale, rounded to t."""
    return round_float(mul_q(fp8_decode_identity(c, row=row), scale), t, rnd, flags)


def fp8_out(y, recip, rnd, flags=kSat):
    """y * recip as fp8 codes, saturating; flags = 0 leaves an overflow to the rounding (NaN)."""
    return f32_to_fp8(mul_q(y, recip), rnd, flags)


# ---- QDQ output activation ----

# the QDQ integer kernels' activation codes
kQdqActNone, kQdqActLrelu, kQdqActGelu, kQdqActPrelu, kQdqActClip = 0, 3, 6, 9, 100


def activate_f32(v, act, act0, act1, slope=0.0):
    """Activation on the fp32 accumulator (see quant.h); other codes pass v through."""
    v = np.asarray(v, F32)
    if act == kQdqActClip:
        return vclamp(v, F32(act0), F32(act1))
    if act == kQdqActLrelu:
        return vmax(v, fma_q(v, act0, act1))
    if act == kQdqActPrelu:
        slope = np.asarray(slope, F32)
        return np.where(v < F32(act0), fma_q(v, slope, fma_q(slope, neg_f32(F32(act0)), act0)), v).astype(F32)
    if act == kQdqActGelu:
        return fma_q(hw_gelu_f32(v), act0, act1)
    return v


def activate_out(v, act, act0, act1, slope, out_t):
    """activate_f32, then round half to even and saturate to out_t."""
    return float_to_int(activate_f32(v, act, act0, act1, slope), out_t)
