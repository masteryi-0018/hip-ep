# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for select.h."""

import numpy as np

from golden_float_cvt import (f32_to_fp8, fp8_decode_identity, fp8_to_f32,
                              load_float, store_float)
from golden_int_round import Rnd
from golden_mac import fma, mul, sub
from golden_minmax import above, max16, min16, vclamp, veq, vge, vgt, vmax, vmin

BF16, F16 = "bf16", "f16"


MIN_NORMAL = {BF16: 0x0080, F16: 0x0400}
MAX_FINITE = {BF16: 0x7F7F, F16: 0x7BFF}
LOWEST = {BF16: 0xFF7F, F16: 0xFBFF}


def abs_half(bits):
    return np.asarray(bits, np.uint16) & np.uint16(0x7FFF)


def neg_half(bits, t):
    """0 - x: both zeros give +0, a NaN flips its sign."""
    return store_float(sub(np.float32(0), load_float(bits, t)), t)


def clip_half(bits, t, lo, hi):
    lob, hib = store_float(np.float32(lo), t), store_float(np.float32(hi), t)
    return min16(max16(bits, lob, t), hib, t)


def fp8_unary(codes, op, scale, lo=0.0, hi=0.0):
    """op: 'abs' | 'neg' | 'clip'; rounds half away from zero."""
    c = np.asarray(codes, np.uint8)
    v = fp8_decode_identity(c & 0x7F if op == "abs" else c)
    v = mul(v, np.float32(scale))
    if op == "clip":
        v = vclamp(v, np.float32(lo), np.float32(hi))
    return f32_to_fp8(v, Rnd.kHalfAway)


def bview(x, bc, H, W, C):
    """Broadcast an operand stored with shape (H|1, W|1, C|1) to (H, W, C)."""
    shape = (1 if bc & 4 else H, 1 if bc & 2 else W, 1 if bc & 1 else C)
    return np.broadcast_to(np.asarray(x).reshape(shape), (H, W, C))


def fp8_pixel_decode(codes, bc, H, W, C):
    """Row-decode an fp8 operand: rows of 8 channels within each pixel."""
    cn = 1 if bc & 1 else C
    v = fp8_decode_identity(np.asarray(codes, np.uint8).reshape(-1), row=cn)
    return bview(v, bc, H, W, C)


def compare(op, fa, fb):
    return {"equal": veq, "greater": vgt, "greater_or_equal": vge}[op](fa, fb).astype(np.int8)


def bcast_half(op, a, abc, b, bbc, t, H, W, C):
    ab, bb = bview(a, abc, H, W, C), bview(b, bbc, H, W, C)
    if op in ("max", "min"):
        return (max16 if op == "max" else min16)(ab, bb, t)
    return compare(op, load_float(ab, t), load_float(bb, t))


def bcast_fp8(op, a, abc, ascale, b, bbc, bscale, H, W, C):
    fa = mul(fp8_pixel_decode(a, abc, H, W, C), np.float32(ascale))
    fb = mul(fp8_pixel_decode(b, bbc, H, W, C), np.float32(bscale))
    if op in ("max", "min"):
        return f32_to_fp8((vmax if op == "max" else vmin)(fa, fb), Rnd.kHalfAway)
    return compare(op, fa, fb)


def arg_reduce(x, axis, is_max, init, bits):
    """x: float64 (H, W, C). Per lane (c % 64 for axis C) the first strict improvement
    over `init`; indices modulo 2^bits; for axis C the smallest wrapped index among
    the lanes holding the overall best."""
    with np.errstate(invalid="ignore"):
        x = np.asarray(x, np.float64)
    mod = 1 << bits

    def scan(v):  # v: (..., N) -> best value and first index per lane
        m = np.full(v.shape[:-1], init, np.float64)
        idx = np.zeros(v.shape[:-1], np.int64)
        for i in range(v.shape[-1]):
            upd = v[..., i] > m if is_max else v[..., i] < m
            m = np.where(upd, v[..., i], m)
            idx = np.where(upd, i, idx)
        return m, idx

    if axis != 2:
        _, idx = scan(np.moveaxis(x, axis, -1))
        return idx % mod
    H, W, C = x.shape
    nch = -(-C // 64)
    pad = nch * 64 - C
    xp = np.concatenate([x, np.full((H, W, pad), np.nan)], axis=2) if pad else x
    lanes = xp.reshape(H, W, nch, 64).transpose(0, 1, 3, 2)      # (H, W, lane, chunk)
    m, chunk = scan(lanes)
    present = np.arange(64) < C                                  # lanes with elements
    best = np.where(present, m, -np.inf if is_max else np.inf)
    best = best.max(axis=-1, keepdims=True) if is_max else best.min(axis=-1, keepdims=True)
    hit = present & ((m >= best) if is_max else (m <= best))
    cand = ((chunk * 64) % mod + np.arange(64)) % mod
    return np.where(hit, cand, mod).min(axis=-1)


def index_out(idx, out_t):
    """Indices as 'u8' / 'u16' / 'u32' / 'i64' (zero-extended) or 'f32'."""
    idx = np.asarray(idx, np.int64)
    if out_t == "f32":
        return idx.astype(np.float32)
    return {"u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "i64": np.int64}[out_t](idx & 0xFFFF)


def reduce_half(bits, t, axes, is_max):
    """axes bitmask: 1 = C, 2 = W, 4 = H; clamped by the start value, NaN skipped."""
    b = np.asarray(bits, np.uint16)
    ax = tuple(a for a, bit in ((0, 4), (1, 2), (2, 1)) if axes & bit)
    moved = np.moveaxis(b, ax, tuple(range(3 - len(ax), 3)))
    keep = moved.shape[:3 - len(ax)]
    flat = moved.reshape(keep + (-1,))
    init = np.uint16(LOWEST[t] if is_max else MAX_FINITE[t])
    r = np.full(keep, init, np.uint16)
    for j in range(flat.shape[-1]):
        r = (max16 if is_max else min16)(r, flat[..., j], t)
    return r


def windows(x, kh, kw, sh, sw):
    """(Ho, Wo, kh*kw, C) taps in row-major window order."""
    Hi, Wi, C = x.shape
    Ho, Wo = (Hi - kh) // sh + 1, (Wi - kw) // sw + 1
    taps = [x[ty:ty + (Ho - 1) * sh + 1:sh, tx:tx + (Wo - 1) * sw + 1:sw] for ty in range(kh)
            for tx in range(kw)]
    return np.stack(taps, axis=2)


def running_max(taps, step):
    """Running max over axis 2 from the first tap: m = step(m, tap)."""
    m = taps[:, :, 0]
    for k in range(1, taps.shape[2]):
        m = step(m, taps[:, :, k])
    return m


def max_half(t):
    return lambda m, x: max16(m, x, t)


def max_fp8(m, x):
    """fp8 running max step (-0 below +0); a NaN tap never replaces the max."""
    return np.where(above(fp8_to_f32(x), fp8_to_f32(m)), x, m)


def maxpool_half(bits, t, kh, kw, sh, sw):
    tp = windows(np.asarray(bits, np.uint16), kh, kw, sh, sw)
    return running_max(tp, max_half(t)).astype(np.uint16)


def fp8_requant(m, c0, c2, clip, lo, hi):
    """Row-decode the fp8 maxima, fma, clamp, round half to even."""
    H, W, C = m.shape
    v = fp8_pixel_decode(m.reshape(-1), 0, H, W, C)
    r = fma(v, np.float32(c2), np.float32(c0))
    if clip:
        r = vclamp(r, np.float32(lo), np.float32(hi))
    return f32_to_fp8(r)


def maxpool_fp8(codes, kh, kw, sh, sw, c0, c2, clip=0, lo=0.0, hi=0.0):
    tp = windows(np.asarray(codes, np.uint8), kh, kw, sh, sw)
    m = running_max(tp, max_fp8).astype(np.uint8)
    return fp8_requant(m, c0, c2, clip, lo, hi)


def globalmaxpool_half(bits, t):
    """Max over H x W per channel and the smallest positive normal."""
    b = np.asarray(bits, np.uint16)
    C = b.shape[-1]
    taps = np.concatenate([np.full((1, C), MIN_NORMAL[t], np.uint16), b.reshape(-1, C)])[None, None]
    return running_max(taps, max_half(t)).reshape(C).astype(np.uint16)


def globalmaxpool_fp8(codes, c0, c1):
    c = np.asarray(codes, np.uint8)
    H, W, C = c.shape
    tp = c.reshape(1, 1, H * W, C)
    m = running_max(tp, max_fp8).astype(np.uint8)
    return fp8_requant(m, c0, c1, 0, 0.0, 0.0).reshape(C)


def topk(vals, raw, W, K, largest, sentinel, sentinel_raw, bits):
    """vals: float64 (rows, W); raw: the same elements as stored. Returns (values, indices)."""
    rows = vals.shape[0]
    Wp = -(-W // 64) * 64
    mod = 1 << bits
    out_v = np.empty((rows, K), raw.dtype)
    out_i = np.empty((rows, K), np.int64)
    for r in range(rows):
        v = np.full(Wp, sentinel, np.float64)
        with np.errstate(invalid="ignore"):
            v[:W] = vals[r]
        rv = np.full(Wp, sentinel_raw, raw.dtype)
        rv[:W] = raw[r]
        lanes, lraw = v.reshape(-1, 64), rv.reshape(-1, 64)
        for k in range(K):
            m = np.full(64, sentinel)
            mr = np.full(64, sentinel_raw, raw.dtype)
            idx = np.zeros(64, np.int64)
            for i in range(lanes.shape[0]):
                x = lanes[i]
                up = x > m if largest else x < m                 # index: strict IEEE
                idx = np.where(up, (i * 64) % mod, idx)
                vu = above(x, m) if largest else above(m, x)             # value: zero order
                m, mr = np.where(vu, x, m), np.where(vu, lraw[i], mr)
            best, bestr = m[0], mr[0]
            for L in range(1, 64):
                if (above(m[L], best) if largest else above(best, m[L])):
                    best, bestr = m[L], mr[L]
            hit = m >= best if largest else m <= best
            res = int(np.where(hit, (idx + np.arange(64)) % mod, mod).min())
            out_v[r, k], out_i[r, k] = bestr, res
            if res < Wp:
                lanes.reshape(-1)[res] = sentinel
                lraw.reshape(-1)[res] = sentinel_raw
    return out_v, out_i
