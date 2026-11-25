# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of float_cvt.h: fp32 <-> bf16 / f16 / fp8 (E4M3FN) with rounding mode and
flags, float <-> integer, host scalar casts. Types are "f32", "bf16", "f16", "fp8"."""

import numpy as np

from golden_bits import bits_f32, f32_bits, qnan
from golden_int_round import Rnd, int_range, round_up, rounds_to_inf, shift_round_mag

F32, I64, U32, U64 = np.float32, np.int64, np.uint32, np.uint64
kFtzIn, kFtzOut, kSat = 1, 2, 4
FMT = {"f32": (8, 23, 127, False), "bf16": (8, 7, 127, False), "f16": (5, 10, 15, False),
       "fp8": (4, 3, 7, True)}


def _bit_length(mag):
    """Bit length of uint64 values (0 for 0)."""
    L = np.minimum(np.frexp(mag.astype(np.float64))[1].astype(I64), 64)
    # the float64 of a wide value may round up to the next power of two
    low = (mag >> np.maximum(L - 1, 0).astype(U64)) == 0
    return np.where(mag == 0, 0, L - low)


def encode_mag(mag, E, t, rnd, neg):
    """mag * 2^E (0 < mag < 2^64) rounded with rnd to the magnitude code of t for a value of
    sign neg, subnormals kept; a result past the largest finite value is returned as a larger
    code (the exponent field stops one past its maximum)."""
    ebits, mbits, bias, fn = FMT[t]
    mag = np.asarray(mag).astype(U64)
    L = _bit_length(mag)
    te = L - 1 + np.asarray(E, I64) + bias
    q = shift_round_mag(mag, L - 1 - mbits + np.where(te >= 1, 0, 1 - te), rnd, neg).astype(I64)
    carry = (te >= 1) & (q >= (2 << mbits))
    q = np.where(carry, q >> 1, q)
    ee = np.minimum(te + carry, 1 << ebits)
    return np.where(te >= 1, (ee << mbits) | (q & ((1 << mbits) - 1)), q)


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


def acc_to_bf16(v):
    """The accumulator read-out to bf16: half away from zero, an fp32-subnormal value flushed."""
    return round_float(v, "bf16", Rnd.kHalfAway, kFtzIn)


def min_normal(t):
    """Smallest normal value of the float type t, as fp32."""
    return decode_float(1 << FMT[t][1], t)[()]


def max_finite(t):
    """Largest finite value of the float type t, as fp32."""
    ebits, mbits, bias, fn = FMT[t]
    top = ((1 << ebits) - 1) << mbits
    return decode_float(top + (1 << mbits) - 2 if fn else top - 1, t)[()]


def code16(v, t):
    """The 16-bit code that indexes a 16-bit function unit with an fp32 v: bf16 takes the top
    16 bits (no rounding), f16 rounds half to even."""
    if t == "f16":
        return encode_float(v, "f16").astype(np.uint16)
    return (f32_bits(v) >> U32(16)).astype(np.uint16)


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


def float_to_int(v, t, shift=0, rnd=Rnd.kHalfEven):
    """v * 2^shift (fp32 values, exact in float64) rounded with rnd, saturated to t; NaN -> 0."""
    with np.errstate(invalid="ignore", over="ignore"):
        # |shift| >= 1300 already overflows / underflows every fp32 in double: clamp so that any
        # Python int is accepted (np.ldexp takes a C int)
        x = np.ldexp(np.asarray(v, np.float64), np.clip(np.asarray(shift, object).astype(np.float64), -1300, 1300).astype(np.int64))
        a = np.abs(x)
        q = np.floor(a)
        d = a - q
        half = d >= 0.5
        rest = np.where(half, d > 0.5, d > 0)
        odd = np.fmod(q, 2.0) == 1
        r = q + round_up(rnd, x < 0, odd, half, rest)
        r = np.where(x < 0, -r, r)
        lo, hi = int_range(t)
        r = np.where(np.isnan(x), 0, r)
        out = np.where(r <= lo, lo, np.where(r >= hi, hi, 0)).astype(I64)
        inside = (r > lo) & (r < hi)
        out[inside] = r[inside].astype(I64)
    return out


def int_to_float(v, e, rnd):
    """int64 v * 2^e rounded with rnd to fp32 (subnormals kept; overflow per rounds_to_inf)."""
    v = np.asarray(v, I64)
    # beyond +-4096 every int64 has already overflowed / left only the sticky bit
    e = np.clip(np.broadcast_to(np.asarray(e, I64), v.shape), -4096, 4096)
    neg = v < 0
    with np.errstate(all="ignore"):  # uint64 wraparound of the two's complement is intended
        mag = np.where(neg, ~v.astype(U64) + U64(1), v.astype(U64))
    code = encode_mag(np.where(v == 0, U64(1), mag), e, "f32", rnd, neg)
    over = np.where(rounds_to_inf(rnd, neg), 0x7F800000, 0x7F7FFFFF)
    code = np.where(code >= 0x7F800000, over, code)
    r = bits_f32(code | (neg.astype(I64) << 31))
    return np.where(v == 0, F32(0), r).astype(F32)


def int_f32(v):
    """The integer v as fp32, round half to even."""
    return int_to_float(v, 0, Rnd.kHalfEven)


def scalar_cast(v, t, rnd=Rnd.kHalfEven):
    """Scalar cast to t ("bf16": always half to even; "f16": rnd), as fp32; inf and NaN
    unchanged."""
    v = np.asarray(v, F32)
    r = round_float(v, t, rnd if t == "f16" else Rnd.kHalfEven)
    return np.where(np.isfinite(v), r, v).astype(F32)


def scalar_cast_i32(v):
    """The (int32) cast of a float coefficient: truncated toward zero, saturated, NaN -> 0."""
    return float_to_int(v, "i32", 0, Rnd.kTrunc)
