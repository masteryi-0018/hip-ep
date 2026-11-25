# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of bits.h: fp32 bit access and the NaN bit patterns shared by every model, and
the errstate decorator of the models."""

import functools

import numpy as np

F32, U32 = np.float32, np.uint32


def f32_bits(x):
    return np.asarray(x, F32).view(U32)


def bits_f32(b):
    return np.asarray(b, np.int64).astype(U32).view(F32)


def qnan():
    """0x7fc00000."""
    return F32(np.nan)


def default_nan():
    """0xffc00000, the NaN of an invalid operation."""
    return bits_f32(0xFFC00000)[()]


def neg_inf():
    """0xff800000."""
    return F32(-np.inf)


def quiet_nan(v):
    """v with the quiet bit set (sign and payload kept)."""
    return bits_f32(f32_bits(v) | U32(0x400000))


def neg_f32(x):
    """-x with the sign bit flipped for every value, a NaN included."""
    return bits_f32(f32_bits(x) ^ U32(0x80000000))


def quiet(fn):
    """inf / NaN operands are part of the domain: no numpy warnings."""
    @functools.wraps(fn)
    def w(*a, **k):
        with np.errstate(all="ignore"):
            return fn(*a, **k)
    return w
