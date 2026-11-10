# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for npu_emu: mllib's python model of each AIE4 conversion
wrapper, corrected to the NPU (fused fp32 MAC, half-to-even float -> int,
fp32 coefficients, the wrapper's rounding mode for block floating point).

Written independently of the HIP code where numpy allows it (np.float16 for
f16 rounding, exact float64 arithmetic plus TwoSum for the fp32 MAC)."""

from fractions import Fraction

import numpy as np

EVEN, AWAY = "even", "away"


# ---- scalar arithmetic ------------------------------------------------------

def fma32(a, b, c, exact_a=False):
    """fp32 fused multiply-add of fp32 arrays: a*b + c rounded once. With
    exact_a, a is taken as float64 as given (an integer below 2^29)."""
    a = np.asarray(a, np.float64) if exact_a else np.asarray(a, np.float32).astype(np.float64)
    b, c = (np.asarray(v, np.float32).astype(np.float64) for v in (b, c))
    p = a * b                                   # exact: 24 + 24 bits
    s = p + c
    bb = s - p
    e = (p - (s - bb)) + (c - bb)               # TwoSum: p + c == s + e exactly
    r = s.astype(np.float32)
    # s is the float64 nearest the exact sum, so rounding it to fp32 is correct
    # except when s lies exactly halfway between two fp32 values while e != 0.
    toward = np.where(s > r.astype(np.float64), np.inf, -np.inf).astype(np.float32)
    nb = np.nextafter(r, toward)
    mid = (r.astype(np.float64) + nb.astype(np.float64)) / 2 == s
    fix = mid & (e != 0) & (s != r.astype(np.float64))
    up = np.maximum(r, nb)
    dn = np.minimum(r, nb)
    return np.where(fix, np.where(e > 0, up, dn), r).astype(np.float32)


def _rne_f32(q):
    """Fraction -> nearest fp32 value (ties to even), returned as a python float."""
    if q == 0:
        return 0.0
    sgn = -1 if q < 0 else 1
    a = abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, -126)
    ulp = Fraction(2) ** (e - 23)
    n = a / ulp
    fl = n.numerator // n.denominator
    rem = n - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2):
        fl += 1
    return float(np.float32(sgn * fl * ulp))


def mac_int(x, c, a):
    """x * c + a rounded once to fp32, x an exact integer (the NPU's
    mac_elem_32(int32, float, accfloat)). Uses fma32 where x*c is exact in
    float64 (|x| < 2^29), exact rationals elsewhere."""
    x = np.asarray(x, np.int64)
    c = np.broadcast_to(np.asarray(c, np.float32), x.shape)
    a = np.broadcast_to(np.asarray(a, np.float32), x.shape)
    small = np.abs(x) < (1 << 29)
    out = np.empty(x.shape, np.float32)
    out[small] = fma32x(x[small].astype(np.float64), c[small], a[small])
    for i in np.nonzero(~small)[0]:
        out[i] = _rne_f32(int(x[i]) * Fraction(float(c[i])) + Fraction(float(a[i])))
    return out


def fma32x(x, b, c):
    """fma32 with x given exactly as float64 (|x| < 2^29 so x*b is exact)."""
    return fma32(x, b, c, exact_a=True)


def round_shift(v, r, mode):
    """Unsigned v >> r with rounding; r >= 63 gives 0."""
    v = np.asarray(v, np.uint64)
    r = np.broadcast_to(np.asarray(r, np.int64), v.shape)
    out = np.zeros(v.shape, np.uint64)
    ok = r < 63
    rr = np.where(ok, np.maximum(r, 1), 1).astype(np.uint64)
    q = v >> rr
    rem = v & ((np.uint64(1) << rr) - np.uint64(1))
    half = np.uint64(1) << (rr - np.uint64(1))
    up = (rem > half) | ((rem == half) & ((q & np.uint64(1)) == 1 if mode == EVEN else True))
    q = q + up.astype(np.uint64)
    q = np.where(r <= 0, v, q)
    return np.where(ok, q, out)


def rne_int(v, lo, hi):
    """fp32 -> integer, half to even (np.rint), saturated."""
    v = np.asarray(v, np.float64)
    return np.clip(np.where(np.isnan(v), 0, np.rint(v)), lo, hi).astype(np.int64)


# ---- floats -----------------------------------------------------------------

def bf16_bits_to_f32(b):
    return (np.asarray(b, np.uint32) << 16).view(np.float32)


def f16_bits_to_f32(b):
    return np.asarray(b, np.uint16).view(np.float16).astype(np.float32)


def encode_float(x, ebits, mbits, bias, fn, mode):
    """fp32 -> sign-magnitude float bits (subnormals; E4M3FN when fn)."""
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.int64)
    sbit = (b >> 31) << (ebits + mbits)
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    emax = (1 << ebits) - 1
    nan = ((emax << mbits) | ((1 << mbits) - 1)) if fn else ((emax << mbits) | (1 << (mbits - 1)))
    sig = np.where(e > 0, m | 0x800000, m)
    u = np.where(e > 0, e - 127, -126)
    # normalize fp32 subnormals
    lz = np.where(sig > 0, 23 - np.floor(np.log2(np.maximum(sig, 1))).astype(np.int64), 0)
    sig = sig << lz
    u = u - lz
    te = u + bias
    shift = 23 - mbits + np.where(te >= 1, 0, 1 - te)
    q = round_shift(sig.astype(np.uint64), shift, mode).astype(np.int64)
    carry = (te >= 1) & (q >= (2 << mbits))
    q = np.where(carry, q >> 1, q)
    ee = te + carry
    code = np.where(te >= 1, (ee << mbits) | (q & ((1 << mbits) - 1)), q)
    max_finite = (nan - 1) if fn else (((emax - 1) << mbits) | ((1 << mbits) - 1))
    over = fn and nan or (emax << mbits)
    code = np.where(code > max_finite, over, code)
    code = np.where((e == 0) & (m == 0), 0, code)
    pay = m >> (23 - mbits)
    qnan = (emax << mbits) | np.where(pay > 0, pay, 1)
    code = np.where(e == 0xFF, np.where(fn, nan, np.where(m != 0, qnan, emax << mbits)), code)
    return (sbit | code).astype(np.uint32)


def to_bf16(x, mode=EVEN):
    return encode_float(x, 8, 7, 127, False, mode).astype(np.uint16)


def to_f16(x, mode=EVEN):
    if mode == EVEN:
        return np.asarray(x, np.float32).astype(np.float16).view(np.uint16)
    return encode_float(x, 5, 10, 15, False, mode).astype(np.uint16)


def to_fp8(x, mode):
    return encode_float(x, 4, 3, 7, True, mode).astype(np.uint8)


def fp8_to_f32(c):
    c = np.asarray(c, np.int64)
    e = (c >> 3) & 15
    m = c & 7
    v = np.where(e == 0, m * 2.0 ** -9, (8 + m) * np.exp2(e - 10.0))
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return np.where(c & 0x80, -v, v).astype(np.float32)


def load_float(bits, t):
    return {"f32": lambda b: np.asarray(b, np.float32), "f16": f16_bits_to_f32,
            "bf16": bf16_bits_to_f32}[t](bits)


def store_float(x, t, mode=EVEN):
    if t == "f32":
        return np.asarray(x, np.float32)
    return to_f16(x, mode) if t == "f16" else to_bf16(x, mode)


# ---- block floating point (mllib DataConverter.f2bfp_mladf) -------------------

def encode_blocks(x, n, mbits, sub, max_shift, mode):
    """Returns (mag, neg, shift, exp) per block of n values."""
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.int64).reshape(-1, n)
    f = (b >> 23) & 0xFF
    sig = np.where(f > 0, 0x800000, 0) | (b & 0x7FFFFF)
    e = np.maximum(f, 1)
    me = e.max(axis=1, keepdims=True)
    diff = (me - e).reshape(-1, n // sub, sub).min(axis=2)
    shift = np.minimum(diff, max_shift)
    sh = me - e + 25 - mbits - np.repeat(shift, sub, axis=1)
    q = round_shift(sig.astype(np.uint64), sh, mode).astype(np.int64)
    q = np.minimum(q, (1 << (mbits - 1)) - 1)
    mag = np.where(sh > 24, 0, q)
    return mag, (b >> 31) == 1, shift, me[:, 0]


def to_bfp16(x, mode):
    mag, neg, _, me = encode_blocks(x, 8, 8, 1, 0, mode)
    m = np.where(neg, -mag, mag).astype(np.int8).view(np.uint8)
    exp = np.where(mag.any(axis=1), me, 0).astype(np.uint8)  # NPU: all-zero block -> 0
    return np.concatenate([exp[:, None], m], axis=1).reshape(-1)


def from_bfp16(blk):
    blk = np.asarray(blk, np.uint8).reshape(-1, 9)
    m = blk[:, 1:].view(np.int8).astype(np.float64)
    return (m * np.exp2(blk[:, :1].astype(np.float64) - 133)).reshape(-1).astype(np.float32)


def to_mx6(x, mode):
    mag, neg, shift, me = encode_blocks(x, 16, 5, 2, 1, mode)
    out = []
    for k in range(len(me)):
        w = int(me[k]) & 0xFF
        for s in range(8):
            w |= int(shift[k, s] & 1) << (8 + s)
        for j in range(16):
            w |= ((0x10 if neg[k, j] else 0) | int(mag[k, j])) << (16 + 5 * j)
        out.append(np.frombuffer(w.to_bytes(12, "little"), np.uint8))
    return np.concatenate(out)


def from_mx6(blk):
    vals = []
    for b in np.asarray(blk, np.uint8).reshape(-1, 12):
        w = int.from_bytes(bytes(b), "little")
        e = w & 0xFF
        for j in range(16):
            code = (w >> (16 + 5 * j)) & 0x1F
            sh = (w >> (8 + j // 2)) & 1
            mag = code & 0xF
            if mag == 0:
                vals.append(0.0)  # the NPU outputs +0 for a zero magnitude
            else:
                v = mag * 2.0 ** (e - sh - 130)
                vals.append(-v if code & 0x10 else v)
    return np.array(vals, np.float32)


# ---- per-wrapper models -------------------------------------------------------

INT_RANGE = {"i8": (-128, 127), "u8": (0, 255), "i16": (-32768, 32767),
             "u16": (0, 65535), "i32": (-2**31, 2**31 - 1), "u32": (0, 2**32 - 1)}


def requant_tail(v, t):
    if t in ("i32", "u32"):
        r = rne_int(v, -2**31, 2**31 - 1)
        return np.maximum(r, 0) if t == "u32" else r
    return rne_int(v, *INT_RANGE[t])


def quant(x, scale, zp, out_t):
    return rne_int(fma32(x, scale, zp), *INT_RANGE[out_t])


def dequant(x, scale, zp, out_t):
    c = np.float32(-np.float32(zp) * np.float32(scale))
    return store_float(mac_int(x, scale, c), out_t)


def requantize(x, in_t, c1, c0, out_t):
    x = np.asarray(x, np.int64)
    if in_t == "u32":
        raw = x.astype(np.uint32).view(np.int32).astype(np.int64)
        r = mac_int(raw, c1, c0)
        corr = np.float32(-(np.float32(c1) * np.float32(4294967296.0)))
        v = mac_int(raw >> 31, corr, r)
    else:
        v = mac_int(x, c1, c0)
    return requant_tail(v, out_t)


def requantize_cw(x, M, B, out_t):
    x = np.asarray(x, np.int64)
    c = np.arange(len(x)) % len(M)
    return requant_tail(mac_int(x, np.asarray(M)[c], np.asarray(B)[c]), out_t)


def a64_down(x, in_t, co, out_t):
    u = np.asarray(x).astype(np.uint64)
    lo = (u & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32).astype(np.int64)
    hi = (u >> np.uint64(32)).astype(np.uint32).view(np.int32).astype(np.int64)
    acc = mac_int(hi, co["mult_f_x2p32"], co["bias_f"])
    acc = mac_int(lo, co["mult_f"], acc)
    if in_t == "i64":
        lo_msb = np.where(lo < 0, -2**31, 0)
        acc = mac_int(lo_msb, np.float32(-np.float32(co["mult_f_x2p32"]) * np.float32(2.0**-31)), acc)
    else:
        acc = mac_int(np.where(lo < 0, -1, 0), -np.float32(co["mult_f_x2p32"]), acc)
        acc = mac_int(np.where(hi < 0, -1, 0), -np.float32(co["mult_f_x2p64"]), acc)
    return rne_int(acc, *INT_RANGE[out_t])


def a64_up(x, in_t, co, out_t):
    res = []
    for v in np.asarray(x, np.int64).tolist():
        if in_t in ("u8", "u16", "u32"):
            prod = (v & 0xFFFFFFFF) * (co["M"] & 0xFFFFFFFF)
        else:
            prod = v * co["M"]
        acc = (co["B"] + prod) & (2**64 - 1)
        acc = acc - 2**64 if acc >= 2**63 else acc
        r = acc
        if co["s"] > 0:
            r = acc >> co["s"]
            rem = acc - (r << co["s"])
            half = 1 << (co["s"] - 1)
            if rem > half or (rem == half and r & 1):
                r += 1
        if out_t == "u64" and r < 0:
            r = 0
        res.append(r)
    return np.array(res, np.int64)


def int32_int64(x, c1, c0):
    return rne_int(mac_int(x, c1, c0), -2**31, 2**31 - 1)


def fp32_fp8(x, scale_recip):
    return to_fp8(np.asarray(x, np.float32) * np.float32(scale_recip), AWAY)


def fp16_fp8(bits, in_t, scale):
    recip = np.float32(1.0) / np.float32(scale)
    r = load_float(to_bf16(recip) if in_t == "bf16" else to_f16(recip), in_t)
    v = load_float(bits, in_t) * np.float32(r)
    return to_fp8(np.clip(v, -448, 448).astype(np.float32), EVEN)


def fp8_to(c, scale, out_t):
    v = fp8_to_f32(c) * np.float32(scale)
    return store_float(v, out_t, AWAY if out_t != "f32" else EVEN)


def bfp16_to_unsigned(blk, mul, zp, hi):
    x = bf16_bits_to_f32(to_bf16(from_bfp16(blk)))
    return rne_int(fma32(x, mul, zp), 0, hi)


def aq_to_bfp16(x, scale, zp):
    d = (np.asarray(x, np.int64) - int(np.float32(zp))).astype(np.float32)
    v = bf16_bits_to_f32(to_bf16(d * np.float32(scale)))
    return to_bfp16(v, EVEN)


def w4_to_bfp16(q, zp, scale_bits, N, K):
    q = np.asarray(q, np.int64).reshape(N, K)
    g = np.repeat(np.arange(K // 32), 32)
    zpk = np.asarray(zp, np.int64).reshape(N, K // 32)[:, g]
    s = bf16_bits_to_f32(np.asarray(scale_bits).reshape(N, K // 32))[:, g]
    return to_bfp16(((q - zpk).astype(np.float32) * s).reshape(-1), EVEN)


def bfp16_qdq_f32(blk, mul, zp, scale):
    q = bfp16_to_unsigned(blk, mul, zp, 65535)
    return ((q - int(np.float32(zp))).astype(np.float32) * np.float32(scale)).astype(np.float32)
