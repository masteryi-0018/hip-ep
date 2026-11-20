# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of mac.h: IEEE fp32 fma / mul / add, the exact-integer MAC, the flushing
MAC, the 16-bit multiply and the vector reduce-add."""

from fractions import Fraction

import numpy as np

from golden_float_cvt import (FMT, bits_f32, decode_float, encode_float, f32_bits, ftz, kFtzIn,
                              kFtzOut)
from golden_int_round import Rnd

F32, F64 = np.float32, np.float64


def _fma64(a, b, c):
    """a * b + c rounded once to fp32; a float64 holding an fp32 or an integer below 2^29,
    b, c fp32, all finite."""
    with np.errstate(all="ignore"):
        return _fma64_body(a, b, c)


def _fma64_body(a, b, c):
    p = a * b.astype(F64)                         # exact: <= 29 + 24 bits
    c = c.astype(F64)
    s = p + c
    bb = s - p
    e = (p - (s - bb)) + (c - bb)                 # TwoSum: p + c == s + e exactly
    r = s.astype(F32)
    # s is the float64 nearest the exact sum, so rounding it to fp32 is correct
    # except when s lies exactly halfway between two fp32 values while e != 0.
    toward = np.where(s > r.astype(F64), np.inf, -np.inf).astype(F32)
    nb = np.nextafter(r, toward)
    mid = (r.astype(F64) + nb.astype(F64)) / 2 == s
    fix = mid & (e != 0) & (s != r.astype(F64))
    return np.where(fix, np.where(e > 0, np.maximum(r, nb), np.minimum(r, nb)), r).astype(F32)


DEFAULT_NAN = np.uint32(0xFFC00000)


def fneg(x):
    """-x with the sign bit flipped for every value, a NaN included."""
    return bits_f32(f32_bits(x) ^ np.uint32(0x80000000))


def _nan(r, ops, invalid):
    """The GPU NaN of a NaN result r: the first NaN operand, quieted; else (an invalid
    operation) 0xFFC00000. `invalid` marks a NaN made by the product before a NaN addend."""
    sel = np.full(r.shape, DEFAULT_NAN, np.uint32)
    for k in range(len(ops) - 1, -1, -1):
        b = f32_bits(ops[k])
        hit = np.isnan(ops[k])
        if k == 2:
            hit &= ~invalid
        sel = np.where(hit, b | np.uint32(0x400000), sel)
    return np.where(np.isnan(r), bits_f32(sel), r).astype(F32)


def fma(a, b, c):
    """fp32 a * b + c, one rounding (IEEE; NaN as the GPU)."""
    with np.errstate(all="ignore"):
        a, b, c = np.broadcast_arrays(np.asarray(a, F32), np.asarray(b, F32), np.asarray(c, F32))
        r = _fma(a, b, c)
        bad = ((a == 0) & np.isinf(b)) | (np.isinf(a) & (b == 0))
        return _nan(r, (a, b, c), bad)


def _fma(a, b, c):
    with np.errstate(all="ignore"):
        r = _fma64(a.astype(F64), b, c)
        # TwoSum is not meaningful for non-finite terms: IEEE result instead.
        bad = ~(np.isfinite(a) & np.isfinite(b) & np.isfinite(c)) | ~np.isfinite(r)
        ieee = (a.astype(F64) * b.astype(F64) + c.astype(F64)).astype(F32)
        r = np.where(bad, ieee, r)
        # An exact zero sum of non-zero terms is +0; signed zeros follow IEEE.
        p = a.astype(F64) * b.astype(F64)
        zero = (r == 0) & (p + c.astype(F64) == 0)
        both_neg = np.signbit(p) & np.signbit(c) & (p == 0) & (c == 0)
        return np.where(zero, np.where(both_neg, F32(-0.0), F32(0.0)), r).astype(F32)


# GPU names (mac.h mul_q / add_q / fma_q): the first NaN operand quieted, else 0xFFC00000.
def mul_q(a, b):
    return mul(a, b)


def add_q(a, b):
    return add(a, b)


def sub_q(a, b):
    return sub(a, b)


def fma_q(a, b, c):
    return fma(a, b, c)


def mul(a, b):
    with np.errstate(all="ignore"):
        a, b = np.broadcast_arrays(np.asarray(a, F32), np.asarray(b, F32))
        return _nan((a * b).astype(F32), (a, b), None)


def add(a, b):
    with np.errstate(all="ignore"):
        a, b = np.broadcast_arrays(np.asarray(a, F32), np.asarray(b, F32))
        return _nan((a + b).astype(F32), (a, b), None)


def sqrt(a):
    """fp32 square root: a NaN quieted (sign and payload kept), a negative operand 0xFFC00000."""
    with np.errstate(all="ignore"):
        a = np.asarray(a, F32)
        r = np.sqrt(a).astype(F32)
        r = np.where(a < 0, bits_f32(DEFAULT_NAN), r)
        return np.where(np.isnan(a), bits_f32(f32_bits(a) | np.uint32(0x400000)), r).astype(F32)


sqrt_q = sqrt


def sub(a, b):
    """a - b: b negated by flipping its sign bit (a NaN included), then added (NaN as add)."""
    return add(a, fneg(b))


def _rne_f32(q):
    """Fraction -> nearest fp32 value (ties to even)."""
    if q == 0:
        return 0.0
    sgn = -1 if q < 0 else 1
    a = abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    ulp = Fraction(2) ** (max(e, -126) - 23)
    n = a / ulp
    fl = n.numerator // n.denominator
    rem = n - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2):
        fl += 1
    return float(F32(sgn * fl * ulp))


def mac_exact(x, c, a, flags=0):
    """x * c + a rounded once to fp32, x an exact integer of any int64 value; kFtzIn flushes c
    and a, kFtzOut the result. Broadcasting."""
    x, c, a = np.broadcast_arrays(np.asarray(x, np.int64), np.asarray(c, F32), np.asarray(a, F32))
    shape = x.shape
    x, c, a = x.reshape(-1), c.reshape(-1), a.reshape(-1)
    if flags & kFtzIn:
        c, a = ftz(c), ftz(a)
    out = np.empty(x.shape, F32)
    plain = (x == 0) | (c == 0) | ~np.isfinite(c) | ~np.isfinite(a)
    with np.errstate(all="ignore"):
        xp, cp, ap = (v[plain].astype(F64) for v in (x, c, a))
        r = (xp * cp + ap).astype(F32)
        # the NaN bits are chosen here (as fma): 0 * inf before a NaN addend
        xf, cf, af = x[plain].astype(F32), c[plain], a[plain]
        out[plain] = _nan(r, (xf, cf, af), (xf == 0) & np.isinf(cf))
        small = ~plain & (x > -(1 << 29)) & (x < (1 << 29))  # np.abs(INT64_MIN) is negative
        out[small] = _fma64(x[small].astype(F64), c[small], a[small])
        for i in np.nonzero(~plain & ~small)[0]:
            out[i] = _rne_f32(int(x[i]) * Fraction(float(c[i])) + Fraction(float(a[i])))
    if flags & kFtzOut:
        out = ftz(out)
    return out.reshape(shape)


def fma_ftz(a, b, c, flags):
    """fp32 a * b + c, one rounding; kFtzIn flushes the operands, kFtzOut the result."""
    if flags & kFtzIn:
        a, b, c = ftz(a), ftz(b), ftz(c)
    r = fma(a, b, c)
    return ftz(r) if flags & kFtzOut else r


def mul16(x, y, t):
    """16-bit vector multiply of t ("bf16" / "f16") codes: an operand subnormal in t reads as
    a zero of its sign, the product is rounded half to even, and a result below the smallest
    normal of t becomes a zero of its sign."""
    ebits, mbits = FMT[t][:2]
    emask, sbit = ((1 << ebits) - 1) << mbits, 1 << (ebits + mbits)

    def operand(c):
        c = np.asarray(c, np.uint32)
        return decode_float(np.where(c & emask, c, c & sbit), t)

    return encode_float(mul(operand(x), operand(y)), t, Rnd.kHalfEven, kFtzOut).astype(np.uint16)


def mul16_f32(a, b, t):
    """mul16 on fp32 values exact in t, as fp32."""
    return decode_float(mul16(encode_float(a, t), encode_float(b, t), t), t)


def _rne_shift(v, d):
    """Nonnegative int64 v >> d (0 <= d <= 62), rounded half to even."""
    d = np.asarray(d, np.int64)
    q = v >> d
    rem = v - (q << d)
    half = np.where(d > 0, np.int64(1) << np.maximum(d - 1, 0), 0)
    return q + ((d > 0) & ((rem > half) | ((rem == half) & (q & 1 == 1))))


def vaddr_f32_reduce(lanes, scalar=None):
    """Vector reduce-add over the last axis (+ an optional scalar per row)."""
    v = np.asarray(lanes, F32)
    if scalar is not None:
        v = np.concatenate([v, np.asarray(scalar, F32).reshape(v.shape[:-1] + (1,))], -1)
    b = v.view(np.uint32).astype(np.int64)
    neg, e, m = b >> 31, (b >> 23) & 0xFF, b & 0x7FFFFF
    ee = np.maximum(e, 1)
    sig = np.where(e > 0, m | (1 << 23), m) << 7
    E = ee.max(-1, keepdims=True)
    d = E - ee
    a = np.where(d >= 32, 0, _rne_shift(sig, np.minimum(d, 62)))
    S = np.where(neg == 1, -a, a).sum(-1)                  # |S| < 2^38
    mag = np.abs(S)
    nb = np.zeros(mag.shape, np.int64)                      # bit length of |S|
    for k in range(40):
        nb += (mag >> k) > 0
    sh = np.maximum(nb - 24, 0)
    q = _rne_shift(mag, sh)                                 # <= 2^24
    expo = E[..., 0] - 157 + sh                             # |S| ~ q * 2^expo
    val = np.ldexp(q.astype(np.float64), expo)
    with np.errstate(over="ignore"):
        out = np.where(S < 0, -val, val).astype(F32)
    out = np.where(val < 2.0 ** -126, np.where(S < 0, F32(-0.0), F32(0.0)), out)
    isnan = ((e == 255) & (m != 0)).any(-1)
    pinf = ((e == 255) & (m == 0) & (neg == 0)).any(-1)
    ninf = ((e == 255) & (m == 0) & (neg == 1)).any(-1)
    out = np.where(pinf, F32(np.inf), np.where(ninf, F32(-np.inf), out))
    out = np.where(isnan | (pinf & ninf), np.uint32(0x7FFFFFFF).view(F32), out)
    return np.asarray(out, F32)
