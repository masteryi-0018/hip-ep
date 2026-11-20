# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of float_cvt.h: fp32 <-> bf16 / f16 / fp8 (E4M3FN) with rounding mode and
flags, fp8 row / identity decode, host scalar casts. Types are "f32", "bf16", "f16", "fp8"."""

import numpy as np

from golden_int_round import Rnd, rounds_to_inf, shift_round_mag

F32, I64, U32 = np.float32, np.int64, np.uint32
kFtzIn, kFtzOut, kSat = 1, 2, 4
FMT = {"f32": (8, 23, 127, False), "bf16": (8, 7, 127, False), "f16": (5, 10, 15, False),
       "fp8": (4, 3, 7, True)}


def f32_bits(x):
    return np.asarray(x, F32).view(U32)


def bits_f32(b):
    return np.asarray(b, np.int64).astype(U32).view(F32)


def qnan():
    return F32(np.nan)


def encode_mag(mag, E, t, rnd, neg):
    """mag * 2^E (0 < mag < 2^53) rounded with rnd to the magnitude code of t for a value of
    sign neg, subnormals kept; a result past the largest finite value is returned as is."""
    ebits, mbits, bias, fn = FMT[t]
    mag = np.asarray(mag, I64)
    L = np.frexp(mag.astype(np.float64))[1].astype(I64)  # bit length (exact below 2^53)
    te = L - 1 + np.asarray(E, I64) + bias
    q = shift_round_mag(mag.astype(np.uint64), L - 1 - mbits + np.where(te >= 1, 0, 1 - te),
                        rnd, neg).astype(I64)
    carry = (te >= 1) & (q >= (2 << mbits))
    q = np.where(carry, q >> 1, q)
    return np.where(te >= 1, ((te + carry) << mbits) | (q & ((1 << mbits) - 1)), q)


def encode_float(x, t, rnd=Rnd.kHalfEven, flags=0):
    """fp32 -> code of t (f32 / bf16 / f16 / fp8), see float_cvt.h."""
    ebits, mbits, bias, fn = FMT[t]
    b = f32_bits(x).astype(I64)
    neg = (b >> 31) == 1
    sbit = (b >> 31) << (ebits + mbits)
    e, m = (b >> 23) & 0xFF, b & 0x7FFFFF
    emax = (1 << ebits) - 1
    inf = emax << mbits
    nan = (inf | ((1 << mbits) - 1)) if fn else (inf | (1 << (mbits - 1)))
    max_finite = nan - 1 if fn else inf - 1
    sig = np.where(e > 0, m | 0x800000, np.maximum(m, 1))  # m = 0 is replaced below
    code = encode_mag(sig, np.where(e > 0, e - 127, -126) - 23, t, rnd, neg)
    sat = bool(flags & kSat) | ~rounds_to_inf(rnd, neg)
    code = np.where(code > max_finite, np.where(sat, max_finite, nan if fn else inf), code)
    if flags & kFtzOut:  # a subnormal result (after rounding) is flushed
        code = np.where(code >> mbits == 0, 0, code)
    zero = (e == 0) & ((m == 0) | bool(flags & kFtzIn))
    code = np.where(zero, 0, code)
    pay = m >> (23 - mbits)
    qn = nan if fn else inf | np.where(pay > 0, pay, 1)
    infc = max_finite if flags & kSat else (nan if fn else inf)
    code = np.where(e == 0xFF, np.where(m != 0, qn, infc), code)
    return (sbit | code).astype(U32)


def decode_float(c, t, flags=0):
    """Code of t (f32 / bf16 / f16 / fp8) as fp32; kFtzIn flushes an fp32-subnormal value."""
    c = np.asarray(c, I64)
    if t == "f32":
        v = bits_f32(c)
    elif t == "bf16":
        v = bits_f32(c << 16)
    elif t == "f16":
        # inf / NaN bits are built here: a hardware half -> single convert would quiet a NaN
        h = c & 0xFFFF
        sign = (h >> 15) << 31
        special = bits_f32(sign | 0x7F800000 | ((h & 0x3FF) << 13))
        v = np.where(((h >> 10) & 0x1F) == 0x1F, special,
                     h.astype(np.uint16).view(np.float16).astype(F32))
    else:
        e, m = (c >> 3) & 15, c & 7
        a = np.where(e == 0, m * 2.0 ** -9, (8 + m) * np.exp2(e - 10.0)).astype(F32)
        nan = bits_f32(0x7FC00000 | ((c & 0x80) << 24))
        v = np.where((e == 15) & (m == 7), nan, np.where(c & 0x80, -a, a)).astype(F32)
    v = np.asarray(v, F32)
    return ftz(v) if flags & kFtzIn else v


def ftz(v):
    """An fp32 subnormal becomes a zero of its sign."""
    b = f32_bits(v)
    return np.where((b & 0x7F800000) == 0, bits_f32(b & 0x80000000), np.asarray(v, F32)).astype(F32)


def round_float(v, t, rnd=Rnd.kHalfEven, flags=0):
    """v rounded to t and back as fp32."""
    return decode_float(encode_float(v, t, rnd, flags), t)


def f32_to_bf16(x, rnd=Rnd.kHalfEven, flags=0):
    return encode_float(x, "bf16", rnd, flags).astype(np.uint16)


def f32_to_f16(x, rnd=Rnd.kHalfEven, flags=0):
    return encode_float(x, "f16", rnd, flags).astype(np.uint16)


def f32_to_fp8(x, rnd=Rnd.kHalfEven, flags=0):
    return encode_float(x, "fp8", rnd, flags).astype(np.uint8)


def bf16_to_f32(h, flags=0):
    return decode_float(h, "bf16", flags)


def f16_to_f32(h, flags=0):
    return decode_float(h, "f16", flags)


def fp8_to_f32(c, flags=0):
    return decode_float(c, "fp8", flags)


def load_float(bits, t, flags=0):
    """Codes (f32 values for f32) of t as fp32; fp8 is not accepted (see float_cvt.h)."""
    if t == "fp8":
        raise ValueError("load_float: decode fp8 with fp8_to_f32 / fp8_decode_identity")
    if t == "f32":
        return decode_float(f32_bits(bits), t, flags)
    return decode_float(bits, t, flags)


def store_float(x, t, rnd=Rnd.kHalfEven, flags=0):
    """x rounded to t: fp32 values for f32, uint8 codes for fp8, else uint16 codes."""
    if t == "f32":
        return round_float(x, t, rnd, flags)
    c = encode_float(x, t, rnd, flags)
    return c.astype(np.uint8 if t == "fp8" else np.uint16)


def fp8_row_nan(c, row=None):
    """Whether each element's group of 8 (from the start of each row of `row` elements,
    default one row) holds a NaN code."""
    c = np.asarray(c, np.uint8)
    if c.size == 0:
        return np.zeros(c.shape, bool)
    n = row or c.size
    r = ((c & 0x7F) == 0x7F).reshape(-1, n)
    pad = np.concatenate([r, np.zeros((len(r), (-n) % 8), bool)], axis=1).reshape(len(r), -1, 8)
    g = np.repeat(pad.any(axis=2), 8, axis=1)[:, :n]
    return g.reshape(c.shape)


def fp8_decode_identity(c, group_nan=None, row=None):
    """fp8 decode through the identity multiply: a NaN group gives 0x7fc00000, -0 -> +0.
    group_nan: per-element flag; default the row groups of fp8_row_nan."""
    if group_nan is None:
        group_nan = fp8_row_nan(c, row)
    v = (fp8_to_f32(c) + F32(0.0)).astype(F32)
    return np.where(group_nan, qnan(), v).astype(F32)


def scalar_cast(v, t, rnd=Rnd.kHalfEven):
    """Scalar cast to t ("bf16": always half to even; "f16": rnd), as fp32; inf and NaN
    unchanged."""
    v = np.asarray(v, F32)
    r = round_float(v, t, rnd if t == "f16" else Rnd.kHalfEven)
    return np.where(np.isfinite(v), r, v).astype(F32)


