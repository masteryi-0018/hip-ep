# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of nlf.h: function units, the tanh table and the table (LUT) lookup."""

import base64
import functools
import lzma
import os
import re

import numpy as np

from golden_float_cvt import bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, round_float
from golden_mac import fma

F32 = f32 = np.float32
HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "nlf_tables.h")
TANH_TABLES = os.path.join(HERE, "tanh_tables.h")
_cache = {}


def quiet(fn):
    """inf / NaN operands are part of the domain: no numpy warnings."""
    @functools.wraps(fn)
    def w(*a, **k):
        with np.errstate(all="ignore"):
            return fn(*a, **k)
    return w


def table(name):
    """Exception bitmap `name` (kInvF32ExcLzma / kInvSqrtF32ExcLzma) as bools."""
    if name not in _cache:
        src = open(TABLES).read()
        body = src[src.index(f"{name}[] ="):]
        body = body[:body.index(";")]
        blob = base64.b64decode("".join(re.findall(r'"([^"]*)"', body)))
        raw = np.frombuffer(lzma.decompress(blob), np.uint8)
        _cache[name] = np.unpackbits(raw, bitorder="little").astype(bool)
    return _cache[name]


# ---- vector reduce-add ----

# ---- reciprocal / inverse square root / ln ----

def hw_inv_f32(x):
    """fp32 reciprocal: RNE(1/x), the other neighbour at the bitmap's mantissas."""
    exc = table("kInvF32ExcLzma")
    x = np.asarray(x, f32)
    b = x.view(np.uint32)
    sign = b & np.uint32(0x80000000)
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    with np.errstate(all="ignore"):
        r = (1.0 / x.astype(np.float64)).astype(f32)  # 53 >= 2*24+2: no double rounding
        mag = r.view(np.uint32) & np.uint32(0x7FFFFFFF)
        above = np.abs(r).astype(np.float64) * np.abs(x).astype(np.float64) > 1.0
        mag = np.where(exc[m], np.where(above, mag - 1, mag + 1), mag).astype(np.uint32)
    out = sign | mag
    out = np.where(e == 0, sign | np.uint32(0x7F800000), out)
    out = np.where((e >= 253) & (e < 255) & ~((e == 253) & (m == 0)), sign, out)
    out = np.where(e == 255, np.where(m == 0, sign, b | np.uint32(0x400000)), out)
    return out.astype(np.uint32).view(f32)


def hw_invsqrt_f32(x):
    """fp32 inverse square root (see hw_invsqrt_f32 in nlf.h)."""
    exc = table("kInvSqrtF32ExcLzma")
    x = np.asarray(x, f32)
    b = x.view(np.uint32).astype(np.int64)
    e, frac, neg = (b >> 23) & 0xFF, b & 0x7FFFFF, (b >> 31) == 1
    borrow = 1 - (e & 1)
    idx = np.where(e == 0, 0, (borrow << 23) | frac)
    mm = (0x3F800000 + idx).astype(np.uint32).view(f32)
    base = (1.0 / np.sqrt(mm.astype(np.float64))).astype(f32)
    # base below 1/sqrt(m) <=> Bi^2 * Mi < 2^71 (Bi = base * 2^24, Mi = m * 2^23),
    # compared exactly by splitting Bi^2 into 24-bit halves.
    Bi = (base.astype(np.float64) * 2.0 ** 24).astype(np.int64)
    Mi = (mm.astype(np.float64) * 2.0 ** 23).astype(np.int64)
    B2 = Bi * Bi
    hi = (B2 >> 24) * Mi + (((B2 & 0xFFFFFF) * Mi) >> 24)
    below = hi < (1 << 47)
    delta = np.where(exc[idx], np.where(below, 1, -1), 0)
    half = (e - 127 - borrow) >> 1
    out = (base.view(np.uint32).astype(np.int64) + delta - (half << 23)) & 0xFFFFFFFF
    out = np.where(e == 0, np.where(neg, 0xFF800000, 0x7F800000), out)
    out = np.where((e == 255) & (frac != 0), b | 0x400000, out)
    out = np.where((e == 255) & (frac == 0), np.where(neg, 0xFFC00000, 0), out)
    out = np.where(neg & (e != 0) & (e != 255), 0xFFC00000, out)
    return out.astype(np.uint32).view(f32)


def hw_approx_ln(x):
    """Approximate ln on the bit fields of x."""
    x = np.asarray(x, f32)
    b = x.view(np.uint32)
    ei = ((b >> 23) & 0xFF).astype(np.int64)
    m = ((b & 0x7FFFFF) | 0x3F800000).astype(np.uint32).view(f32)
    above = m > f32(1.41421356)
    m = np.where(above, m * f32(0.5), m).astype(f32)
    hi = m > f32(1.18920708)
    lo = ~hi & (m < f32(0.84089641))
    m = np.where(hi, m * f32(0.84089641), np.where(lo, m * f32(1.18920708), m)).astype(f32)
    off = np.where(hi, f32(0.17328679), np.where(lo, -f32(0.17328679), f32(0))).astype(f32)
    e = (ei - 127 + above).astype(f32)
    t = (m - f32(1)).astype(f32)
    p_low = fma(e, f32(0.693147182), (t + off).astype(f32))
    p_high = fma(t, f32(0.333333343), f32(-0.5))
    return fma((t * t).astype(f32), p_high, p_low)


# ---- 16-bit units ----

def hw_sqrt16(b, t):
    """16-bit square root on t ("bf16" / "f16") codes."""
    b = np.asarray(b, np.uint16).astype(np.int64)
    neg = (b >> 15) == 1
    if t == "f16":
        e, m = (b >> 10) & 0x1F, b & 0x3FF
        with np.errstate(invalid="ignore"):
            r = np.sqrt(f16_to_f32(b).astype(np.float64)).astype(np.float16)
        r = r.view(np.uint16).astype(np.int64)
        r = np.where((m == 0x3FF) & (e >= 2) & (e % 2 == 0) & ~neg & (e < 31), r + 1, r)
        r = np.where(neg & ((b & 0x7FFF) != 0), 0xFE00, r)
        r = np.where((b & 0x7FFF) == 0, b, r)
        r = np.where(e == 0x1F, np.where(m != 0, b | 0x200, np.where(neg, 0xFE00, 0x7C00)), r)
        return r.astype(np.uint16)
    e, m = (b >> 7) & 0xFF, b & 0x7F
    with np.errstate(invalid="ignore"):
        s = np.sqrt(bf16_to_f32(b).astype(np.float64)).astype(f32)
    r = f32_to_bf16(np.where(neg, f32(0), s)).astype(np.int64)
    r = np.where(neg, 0xFFC0, r)
    r = np.where(e == 0, b & 0x8000, r)
    r = np.where(e == 0xFF, np.where(m != 0, b | 0x40, np.where(neg, 0xFFC0, 0x7F80)), r)
    return r.astype(np.uint16)


def hw_inv16(h, t):
    """16-bit reciprocal on t ("bf16" / "f16") codes: RNE(1 / y), NaN quieted. bf16: zero /
    subnormal y -> +-inf, a subnormal result -> +-0. f16: subnormals kept."""
    with np.errstate(all="ignore"):
        h = np.asarray(h, np.uint16)
        if t == "f16":
            y = h.view(np.float16).astype(np.float64)
            r = (1.0 / y).astype(np.float16).view(np.uint16)
            nan = (h & 0x7FFF) > 0x7C00
            return np.where(nan, h | np.uint16(0x200), r).astype(np.uint16)  # NaN: quieted
        y = bf16_to_f32(h)
        r = round_float((1.0 / y.astype(np.float64)).astype(f32), "bf16")
    r = np.where(np.abs(y) < f32(2.0 ** -126), np.copysign(f32(np.inf), y), r)
    r = np.where(np.abs(r) < f32(2.0 ** -126), np.copysign(f32(0), r), r)
    out = f32_to_bf16(r)
    nan = (h & 0x7FFF) > 0x7F80
    return np.where(nan, h | np.uint16(0x40), out).astype(np.uint16)  # NaN: quieted


def hw_invsqrt_f16(h):
    """16-bit inverse square root on f16 codes: the fp32 one rounded half to even to f16."""
    return f32_to_f16(hw_invsqrt_f32(f16_to_f32(h)))


# ---- exp2 ----

EXP2_CNST = np.array((0, 39, 74, 110, 141, 172, 200, 225, 250, 270, 290, 305, 320, 332, 340,
                      347, 352, 352, 350, 347, 340, 328, 315, 299, 280, 257, 229, 200, 167,
                      131, 91, 47), np.int64)
EXP2_COEFF = np.array((-10, -9, -9, -8, -8, -7, -6, -6, -5, -5, -4, -4, -3, -2, -2, -1, 0, 0,
                       1, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12), np.int64)


@quiet
def hw_exp2_pwl(p, bf16_out):
    """Piecewise-linear exp2 of a biased fp32 p; bf16_out keeps the upper 16 bits. NaN p
    gives 0x7fc00000."""
    p = np.asarray(p, f32)
    x = ((p + f32(0.0)).view(np.int32).astype(np.int64) >> 2) - (0x111 << 20)
    sel = (x >> 7) & 31
    fwd = (x & ~1) + (16 if bf16_out else 0)
    if bf16_out:
        fwd = np.where(((x >> 4) & 255) == 255, fwd & ~15, fwd)
    approx = fwd + ((((x >> 2) & 31) * EXP2_COEFF[sel]) >> 3) - EXP2_CNST[sel]
    dn = (x < (1 << 12)) & (x >= -(11 << 12))
    sh = np.where(dn, 1 - (approx >> 12), 0)
    data = 4096 + (approx & 4095)
    approx = np.where(dn, (data >> sh) | ((data & ((np.int64(1) << sh) - 1)) != 0), approx)
    mask = 0x7FFF0000 if bf16_out else 0x7FFFFFFF
    out = ((np.clip(approx, 0, 255 << 12) << 11) & mask).astype(np.uint32).view(f32)
    return np.where(np.isnan(p), f32(np.nan), out).astype(f32)


def hw_exp2_bf20(p):
    """Hardware exp2 of an already biased p; subnormals kept."""
    return hw_exp2_pwl(p, False)


def hw_exp2_bf16(p):
    return hw_exp2_pwl(p, True)


# ---- hardware tanh (table) ----

TANH_RANGE = {False: (0x3F00, 0x4040, 0x7F80), True: (0x3800, 0x4200, 0x7C00)}


def _tanh_table(t):
    """All 65536 results of tanh16 for t ("bf16" / "f16") codes, from tanh_tables.h."""
    f16 = t == "f16"
    key = ("tanh", f16)
    if key not in _cache:
        src = open(TANH_TABLES).read()
        name = "kTanhF16" if f16 else "kTanhBf16"
        body = src[src.index(f"{name}[] = {{"):]
        body = body[body.index("{") + 1:body.index("};")]
        tab = np.array([int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", body)], np.int64)
        lo, hi, inf = TANH_RANGE[f16]
        c = np.arange(1 << 16, dtype=np.int64)
        a, s = c & 0x7FFF, c & 0x8000
        small = (f32_bits16(f16_to_f32(a)) if f16 else a)
        mid = tab[np.clip(a - lo, 0, hi - lo - 1)]
        r = np.where(a < lo, small, np.where(a < hi, mid, 0x3F80)) | s
        _cache[key] = np.where(a > inf, 0x7F81, r).astype(np.uint16)
    return _cache[key]


def f32_bits16(v):
    """Upper 16 bits of fp32 values."""
    return (np.asarray(v, f32).view(np.uint32) >> 16).astype(np.int64)


def tanh16(c, t):
    """tanh -> bf16 bits of t ("bf16" / "f16") codes c."""
    return _tanh_table(t)[np.asarray(c, np.uint16).astype(np.int64)]


def tanh_acc(h, t):
    """tanh -> bf16 value of an fp32 h: bf16 indexes by the top 16 bits, f16 by the f16 code."""
    h = np.asarray(h, f32)
    c = f32_to_f16(h) if t == "f16" else f32_bits16(h)
    return bf16_to_f32(tanh16(c, t))


# ---- table (LUT) lookup ----

def lut_entry(lut, i):
    """(offset, slope) of entries i of a table image (groups of 8 pairs stored twice)."""
    i = np.asarray(i, np.int64)
    k = (i // 8) * 32 + (i % 8) * 2
    lut = np.asarray(lut, f32)
    return lut[k], lut[k + 1]


def lut_index(x, k, n, rnd):
    """floor(bf16(fma(x, k, n / 2))) clamped to [0, n - 1]; NaN -> 0."""
    v = round_float(fma(x, f32(k), f32(n // 2)), "bf16", rnd).astype(np.float64)
    v = np.where(np.isnan(v), 0, v)
    return np.clip(np.floor(v), 0, n - 1).astype(np.int64)


def lut_interp(x, lut, i, slope_t, rnd):
    """fma(x, slope, offset) of entries i; the slope is rounded to slope_t first."""
    off, sl = lut_entry(lut, i)
    return fma(x, round_float(sl, slope_t, rnd), off)
