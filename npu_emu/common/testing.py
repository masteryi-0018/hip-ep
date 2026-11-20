# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Helpers shared by the golden tests: type tables, exhaustive codes, calls, comparison."""

import numpy as np

from golden_float_cvt import f32_to_bf16, f32_to_f16
from golden_int_round import INT_RANGE
from npu import (NPU_BF16, NPU_F16, NPU_I8, NPU_I16, NPU_I32, NPU_I64, NPU_U8, NPU_U16, NPU_U32,
                 NPU_U64, Host, call, callx)

INT = {"i8": (NPU_I8, np.int8), "u8": (NPU_U8, np.uint8), "i16": (NPU_I16, np.int16),
       "u16": (NPU_U16, np.uint16)}
INT_ALL = {**INT, "i32": (NPU_I32, np.int32), "u32": (NPU_U32, np.uint32),
           "i64": (NPU_I64, np.int64), "u64": (NPU_U64, np.uint64)}
F16T = {"bf16": NPU_BF16, "f16": NPU_F16}
HALF = {"bf16": (NPU_BF16, "bf16"), "f16": (NPU_F16, "f16")}
ALL16 = np.arange(65536, dtype=np.uint32).astype(np.uint16)
ALL8 = np.arange(256, dtype=np.uint16).astype(np.uint8)


def shp(s):
    """A shape {H, W, C} as the host int64 array the broadcasting entries take."""
    return Host(np.array(s, np.int64))


def int_codes(t):
    """Every code of the integer type t."""
    lo, hi = INT_RANGE[t]
    return np.arange(lo, hi + 1, dtype=np.int64).astype(INT[t][1])


def rand_ints(rng, dt, shape, lo=None, hi=None):
    """Uniform integers of the numpy type dt in [lo, hi] (default: the whole type)."""
    info = np.iinfo(dt)
    return rng.integers(info.min if lo is None else lo, info.max if hi is None else hi, shape,
                        endpoint=True).astype(dt)


def half_values(rng, t, n, special=None):
    """n bf16 / f16 (t) codes: half wide random values, half a coarse grid (exact ties), zeros
    of either sign; with `special` (an array of codes), n // 8 positions get one of them."""
    wide = (rng.standard_normal(n) * np.exp2(rng.integers(-6, 7, n))).astype(np.float32)
    grid = (rng.integers(-12, 13, n) * np.float32(0.375)).astype(np.float32)
    v = np.where(rng.random(n) < 0.5, wide, grid)
    bits = f32_to_f16(v) if t == "f16" else f32_to_bf16(v)
    bits = np.where((v == 0) & (rng.random(n) < 0.5), bits | np.uint16(0x8000), bits)
    if special is not None:
        pos = rng.choice(n, n // 8, replace=False)
        bits[pos] = special[rng.integers(0, len(special), n // 8)]
    return bits.astype(np.uint16)


def fp8_values(rng, n, nan_rate=0.01):
    """Random E4M3 codes with some NaN (0x7F) and both zeros; n may be a shape."""
    c = rng.integers(0, 256, n).astype(np.uint8)
    c = np.where((c & 0x7F) == 0x7F, np.uint8(0x38), c)
    c = np.where(rng.random(n) < nan_rate, np.uint8(0x7F), c)
    c = np.where(rng.random(n) < 0.05, rng.choice(np.array([0x00, 0x80], np.uint8), n), c)
    return c.astype(np.uint8)


def run(name, *args, n, t):
    """Call an entry whose output is n elements of the integer type t (a key of INT_ALL)."""
    return call_n(name, *args, n=n, dt=INT_ALL[t][1])


def call_n(name, *args, n, dt, n2=0):
    """Call an entry whose output is n elements of the numpy type dt; a tuple n is the shape of
    the output. With n2, the entry has a second output of n2 elements and both are returned."""
    size = np.dtype(dt).itemsize
    if n2:
        a, b = callx(name, *args, outs=[int(np.prod(n)) * size, int(n2) * size])
        return a.view(dt), b.view(dt)
    out = call(name, *args, out_nbytes=int(np.prod(n)) * size, out_dtype=dt)
    return out.reshape(n) if isinstance(n, tuple) else out


def rejected(name, *args, nbytes):
    """True when the entry returns -1 for arguments that are otherwise valid (nbytes of output)."""
    try:
        call(name, *args, out_nbytes=nbytes, out_dtype=np.uint8)
    except RuntimeError as e:
        return str(e) == f"{name} failed: -1"
    return False


def assert_same(got, want):
    """Value comparison, want cast to the type of got."""
    got = np.asarray(got).reshape(-1)
    want = np.asarray(want).astype(got.dtype).reshape(-1)
    assert got.size == want.size, f"size {got.size} != {want.size}"
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{got.size} differ, first at {bad[:6].tolist()}: "
                           f"got {got[bad[:6]].tolist()} want {want[bad[:6]].tolist()}")


def same_bytes(got, want):
    """Byte comparison of two buffers."""
    got = np.ascontiguousarray(got).view(np.uint8).reshape(-1)
    want = np.ascontiguousarray(want).view(np.uint8).reshape(-1)
    assert got.size == want.size, f"size {got.size} != {want.size}"
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} bytes differ, first at {bad[:8].tolist()}: "
                           f"got {got[bad[:8]].tolist()} want {want[bad[:8]].tolist()}")


def same(got, want, what="", nan=None):
    """Bitwise comparison, want cast to the type of got. nan(a) -> bool mask: positions where
    both are NaN match."""
    got = np.ascontiguousarray(got).reshape(-1)
    want = np.ascontiguousarray(np.asarray(want).astype(got.dtype)).reshape(-1)
    assert got.size == want.size, f"{what}: size {got.size} != {want.size}"
    gb, wb = got.view(f"u{got.itemsize}"), want.view(f"u{want.itemsize}")
    bad = gb != wb
    if nan is not None:
        bad &= ~(nan(got) & nan(want))
    bad = np.nonzero(bad)[0]
    assert bad.size == 0, (f"{what}: {bad.size}/{got.size} differ, first {bad[:6].tolist()}: "
                           f"got {gb[bad[:6]].tolist()} want {wb[bad[:6]].tolist()}")


# ---- special-value tables (bit patterns; shared by every directory's golden tests) ----

# fp32: +-0, +-min subnormal, max subnormal, min normal, 1, max finite, +-inf, quiet / signalling
# NaNs of either sign with and without payload, and the default NaN.
F32_SPECIAL_BITS = np.array(
    [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000,
     0x80800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
     0x7FC00000, 0xFFC00000, 0x7FC00123, 0xFFC00456, 0x7F800001, 0xFF800001, 0x7FBFFFFF,
     0x7FFFFFFF, 0xFFFFFFFF], np.uint32)


def f32_special():
    """The fp32 special values as float32."""
    return F32_SPECIAL_BITS.view(np.float32).copy()


# 16-bit codes: +-0, +-min subnormal, max subnormal, min normal, 1, max finite, +-inf, quiet and
# signalling NaNs of either sign with payload.
BF16_SPECIAL = np.array(
    [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x3F80, 0xBF80, 0x7F7F,
     0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7FC5, 0xFFC7, 0x7F81, 0xFF81, 0x7FBF, 0x7FFF,
     0xFFFF], np.uint16)
F16_SPECIAL = np.array(
    [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400, 0x3C00, 0xBC00, 0x7BFF,
     0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7E05, 0xFE07, 0x7C01, 0xFC01, 0x7DFF, 0x7FFF,
     0xFFFF], np.uint16)
HALF_SPECIAL = {"bf16": BF16_SPECIAL, "f16": F16_SPECIAL}

# E4M3FN: +-0, min subnormal, max subnormal, min normal, 1, max finite, NaN of either sign.
FP8_SPECIAL = np.array([0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8, 0x7E, 0xFE,
                        0x7F, 0xFF], np.uint8)


def int_special(t):
    """Integer boundaries of the type t (a key of INT_ALL): 0, +-1, min, max and their
    neighbours, as int64 (u64 above 2^63 - 1 is not representable there)."""
    lo, hi = (0, 2**63 - 1) if t == "u64" else (np.iinfo(INT_ALL[t][1]).min,
                                                 np.iinfo(INT_ALL[t][1]).max)
    v = {0, 1, -1, lo, lo + 1, hi, hi - 1, (lo + hi) // 2}
    return np.array(sorted(x for x in v if lo <= x <= hi), np.int64)


# ---- shared tables with rounding / saturation boundaries (also used by golden_convert tests) ----

def bits32(*b):
    """fp32 values from bit patterns."""
    return np.array(b, np.uint32).view(np.float32)


# fp32: zeros, subnormals, extremes, inf, NaN of both signs with payloads, then values at the
# rounding and saturation boundaries of the 8 / 16 / 32-bit formats and integers.
F32_SPEC = np.concatenate([
    bits32(0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
           0x00800001, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F7FFFFE, 0x7F800000,
           0xFF800000, 0x7FC00000, 0xFFC00000, 0x7FC00001, 0xFFC12345, 0x7F800001, 0xFF800001,
           0x7FFFFFFF, 0xFFFFFFFF, 0x7FC0FFFF, 0x7F80FFFF, 0x7F818000, 0x7FBFFFFF),
    np.array([65504, 65520, 65536, 448, 464, 480, 2.0**-9, 2.0**-10, 2.0**-24, 2.0**-25,
              1.5 * 2.0**-25, 2.0**31, -2.0**31, 2.0**32, 2.0**63, 2.0**64, 0.5, 1.5, 2.5, -0.5,
              -1.5, 127.5, 128.5, -128.5, 255.5, 256.5, 32767.5, 32768.5, 65535.5, 65536.5,
              2147483520.0, 4294967296.0, -448, -464, -65520, 3.4e38, -3.4e38, 1e-39],
             np.float32)])

# 16-bit codes shared by bf16 and f16: zeros, subnormals, extremes, inf, NaN of both signs.
SP16 = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x03FF, 0x0080, 0x0400, 0x3C00, 0xBC00,
                 0x3F80, 0xBF80, 0x7BFF, 0xFBFF, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7C00, 0xFC00,
                 0x7FC0, 0xFFC0, 0x7E00, 0xFE00, 0x7F81, 0xFF81, 0x7C01, 0xFC01, 0x7FFF, 0xFFFF,
                 0x5F00, 0xDF00, 0x2000], np.uint16)

# Scalar coefficients: zeros, inf, NaN of both signs, subnormal, overflow, ordinary.
COEF = [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float(bits32(0x7FC00000)[0]),
        float(bits32(0xFFC00001)[0]), 1e-45, 3.4e38, 0.37, 3e9]
COEF_PAIRS = [(a, b) for a in COEF for b in COEF]


def x16_special(stride=97):
    """SP16 plus every `stride`-th 16-bit code."""
    return np.concatenate([SP16, ALL16[::stride]])
