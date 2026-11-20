# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for quantized division, leaky relu, prelu, square and batchnorm."""

import numpy as np

from golden_float_cvt import bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, round_float
from golden_int_round import float_to_int
from golden_mac import fma, mac_exact, mul
from golden_minmax import vmax
from golden_nlf import hw_inv16, hw_inv_f32

f32 = np.float32


def div(a, b, c0, c1, zp1_word, zp2_word, out_t):
    fa = (f32(12582912) + np.asarray(a, np.int64).astype(f32)) - f32(zp1_word)
    fb = (f32(12582912) + np.asarray(b, np.int64).astype(f32)) - f32(zp2_word)
    q = mul(round_float(fa, "bf16"), bf16_to_f32(hw_inv16(f32_to_bf16(fb), "bf16")))
    return float_to_int(fma(q, c0, c1), out_t)


def bcast(b, shape):
    return np.broadcast_to(np.asarray(b), shape)


def bdiv_a8(a, b, c0, c1, zp1, zp2, shift, t):
    """a [H][W][C], b broadcastable."""
    da = round_float((np.asarray(a, np.int64) - zp1).astype(f32), "f16")
    db = round_float((bcast(b, a.shape).astype(np.int64) - zp2).astype(f32), "f16")
    db = np.where(db == 0, f16_to_f32(0x0496), db)  # f16(7e-5)
    q = mul(da, f16_to_f32(hw_inv16(f32_to_f16(db), "f16")))
    return float_to_int(fma(q, c0, c1), t, -shift)


def bdiv_a16(a, b, c0, c1, zp1, zp2, shift, t):
    da = (np.asarray(a, np.int64) - zp1).astype(f32)
    db = (bcast(b, a.shape).astype(np.int64) - zp2).astype(f32)
    # A zero divisor uses the correctly rounded 1 / 7e-5f.
    inv = np.where(db == 0, f32(1.0 / np.float64(f32(7e-5))), hw_inv_f32(db))
    return float_to_int(fma(mul(da, inv), c0, c1), t, -shift)


def leakyrelu(x, c1, c0, c1a, c0a, t):
    return float_to_int(vmax(mac_exact(x, c1, c0), mac_exact(x, c1a, c0a)), t)


def prelu(x, pos, in_zp, out_zp, neg, t):
    """x [rows][C], neg [C]."""
    d = np.asarray(x, np.int64) - int(in_zp)
    s = np.where(d < 0, np.broadcast_to(np.asarray(neg, f32), d.shape), f32(pos))
    return float_to_int(mac_exact(d, s, out_zp), t)


def square(x, a, b, c, t):
    inner = mac_exact(x, a, b)
    return float_to_int(mac_exact(x, inner, c), t)


def batchnorm(x, cw, cb, act, zp, alpha, t):
    """x [rows][C]; act 0 none, 1 relu, 3 leaky relu."""
    o = mac_exact(x, cw, cb)
    if act == 1:
        o = vmax(o, f32(zp))
    elif act == 3:
        lc = f32(f32(zp) * f32(f32(1.0) - f32(alpha)))
        o = vmax(o, fma(o, alpha, lc))
    return float_to_int(o, t)
