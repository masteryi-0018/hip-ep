# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of minmax.h: vector compares, fp32 and 16-bit vector max / min."""

import numpy as np

from golden_bits import f32_bits
from golden_float_cvt import decode_float, fp8_to_f32

F32 = np.float32


def vmax(a, b):
    """fp32 max: -0 orders below +0; a NaN operand gives 0x7fc00000."""
    a, b = np.broadcast_arrays(np.asarray(a, F32), np.asarray(b, F32))
    r = np.where(a == b, np.where(np.signbit(a), b, a), np.where(a > b, a, b))
    return np.where(np.isnan(a) | np.isnan(b), F32(np.nan), r).astype(F32)


def vmin(a, b):
    """fp32 min: -0 orders below +0; a NaN operand gives 0x7fc00000."""
    a, b = np.broadcast_arrays(np.asarray(a, F32), np.asarray(b, F32))
    r = np.where(a == b, np.where(np.signbit(a), a, b), np.where(a < b, a, b))
    return np.where(np.isnan(a) | np.isnan(b), F32(np.nan), r).astype(F32)


def veq(a, b):
    """IEEE compares: -0 == +0, a NaN compares false."""
    return np.asarray(a, F32) == np.asarray(b, F32)


def vgt(a, b):
    return np.asarray(a, F32) > np.asarray(b, F32)


def vge(a, b):
    return np.asarray(a, F32) >= np.asarray(b, F32)


def above(a, b):
    """a strictly above b in the vmax order (-0 below +0); a NaN is never above and nothing is
    above it."""
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    return ~np.isnan(a) & ~np.isnan(b) & (f32_bits(vmax(a, b)) != f32_bits(b))


def vclamp(v, lo, hi):
    return vmin(vmax(v, lo), hi)


def _pick16(a, b, t, fn):
    a, b = np.broadcast_arrays(np.asarray(a, np.uint16), np.asarray(b, np.uint16))
    fa, fb = decode_float(a, t), decode_float(b, t)
    keep_a = fn(fa, fb).view(np.uint32) == fa.view(np.uint32)
    r = np.where(keep_a, a, b)
    nan = np.isnan(fa) | np.isnan(fb)
    return np.where(nan, np.uint16(0x7E00 if t == "f16" else 0x7FC0), r).astype(np.uint16)


def max16(a, b, t):
    """16-bit max of t ("bf16" / "f16") codes: -0 below +0, equal values keep a, a NaN operand
    gives 0x7fc0 / 0x7e00."""
    return _pick16(a, b, t, vmax)


def min16(a, b, t):
    return _pick16(a, b, t, vmin)


def fp8_max(a, b):
    """Compare-and-select max of fp8 codes, a the running value: b replaces it only when
    strictly above (-0 below +0); equal values and a NaN on either side keep a."""
    a, b = np.broadcast_arrays(np.asarray(a, np.uint8), np.asarray(b, np.uint8))
    return np.where(above(fp8_to_f32(b), fp8_to_f32(a)), b, a).astype(np.uint8)
