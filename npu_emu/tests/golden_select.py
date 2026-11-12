# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the select-group entry points of npu_emu (selection,
compare and min / max wrappers), written from the AIE4 kernel sources and
checked against aiesim outputs. Logical HWC layouts throughout.

NPU rules modelled here (see include/npu_emu_select.h):
  - bf16 / f16 VMAX / VMIN order -0 below +0; compares are IEEE.
  - fp32 VMAX / VMIN / clamp give +0 for a zero result.
  - fp8 inputs decode through an 8x8 identity matmul: NaN spreads over its row
    of 8 channels (golden.fp8_to_f32_row on each pixel's channels)."""

import numpy as np

import golden as g

BF16, F16 = "bf16", "f16"


# ---- 16-bit floats -------------------------------------------------------------

def half_to_f64(bits, t):
    bits = np.asarray(bits, np.uint16)
    if t == F16:
        return bits.view(np.float16).astype(np.float64)
    return g.bf16_bits_to_f32(bits).astype(np.float64)


def half_round(x, t):
    """float -> 16-bit bits, half to even."""
    x = np.asarray(x, np.float32)
    return x.astype(np.float16).view(np.uint16) if t == F16 else g.to_bf16(x)


def zgt(a, b):
    """a > b with -0 below +0 (the AIE4 bf16 VMAX / VMIN order)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    zz = (a == 0) & (b == 0)
    return np.where(zz, ~np.signbit(a) & np.signbit(b), a > b)


CANON_NAN = {BF16: 0x7FC0, F16: 0x7E00}
MIN_NORMAL = {BF16: 0x0080, F16: 0x0400}     # numeric_limits<T>::min()
MAX_FINITE = {BF16: 0x7F7F, F16: 0x7BFF}
LOWEST = {BF16: 0xFF7F, F16: 0xFBFF}


# ---- elementwise unary ---------------------------------------------------------

def abs_half(bits):
    return np.asarray(bits, np.uint16) & np.uint16(0x7FFF)


def neg_half(bits):
    b = np.asarray(bits, np.uint16)
    return np.where((b & 0x7FFF) == 0, np.uint16(0), b ^ np.uint16(0x8000)).astype(np.uint16)


def clip_half(bits, t, lo, hi):
    b = np.asarray(bits, np.uint16)
    lob, hib = half_round(np.float32(lo), t), half_round(np.float32(hi), t)
    x = half_to_f64(b, t)
    r = np.where(zgt(half_to_f64(lob, t), x), lob, b)
    return np.where(zgt(half_to_f64(r, t), half_to_f64(hib, t)), hib, r).astype(np.uint16)


def fp8_unary(codes, op, scale, lo=0.0, hi=0.0):
    """op: 'abs' | 'neg' | 'clip'. Rounding half away from zero (symmetric_inf)."""
    c = np.asarray(codes, np.uint8)
    v = g.fp8_to_f32_row(c & 0x7F if op == "abs" else c)
    v = (v * np.float32(scale)).astype(np.float32)
    if op == "clip":
        v = np.where(np.float32(lo) > v, np.float32(lo), v)
        v = np.where(v > np.float32(hi), np.float32(hi), v).astype(np.float32)
        v = np.where(v == 0, np.float32(0), v).astype(np.float32)
    return g.to_fp8(v, g.AWAY)


# ---- broadcasting --------------------------------------------------------------

def bview(x, bc, H, W, C):
    """Broadcast an operand stored with shape (H|1, W|1, C|1) to (H, W, C)."""
    shape = (1 if bc & 4 else H, 1 if bc & 2 else W, 1 if bc & 1 else C)
    return np.broadcast_to(np.asarray(x).reshape(shape), (H, W, C))


def fp8_pixel_decode(codes, bc, H, W, C):
    """Row-decode an fp8 operand: rows of 8 channels within each of its pixels."""
    cn = 1 if bc & 1 else C
    x = np.asarray(codes, np.uint8).reshape(-1, cn)
    pad = (-cn) % 8
    xp = np.concatenate([x, np.zeros((x.shape[0], pad), np.uint8)], axis=1) if pad else x
    v = g.fp8_to_f32_row(xp.reshape(-1)).reshape(xp.shape)[:, :cn]
    return bview(v, bc, H, W, C)


def compare(op, fa, fb):
    return {"equal": fa == fb, "greater": fa > fb, "greater_or_equal": fa >= fb}[op].astype(np.int8)


def bcast_half(op, a, abc, b, bbc, t, H, W, C):
    ab, bb = bview(a, abc, H, W, C), bview(b, bbc, H, W, C)
    fa, fb = half_to_f64(ab, t), half_to_f64(bb, t)
    if op in ("max", "min"):
        pick_b = zgt(fb, fa) if op == "max" else zgt(fa, fb)
        r = np.where(pick_b, bb, ab)
        return np.where(np.isnan(fa) | np.isnan(fb), np.uint16(CANON_NAN[t]), r).astype(np.uint16)
    return compare(op, fa, fb)


def bcast_fp8(op, a, abc, ascale, b, bbc, bscale, H, W, C):
    fa = (fp8_pixel_decode(a, abc, H, W, C) * np.float32(ascale)).astype(np.float32)
    fb = (fp8_pixel_decode(b, bbc, H, W, C) * np.float32(bscale)).astype(np.float32)
    if op in ("max", "min"):
        r = np.where(fb > fa, fb, fa) if op == "max" else np.where(fb < fa, fb, fa)
        r = np.where(r == 0, np.float32(0), r)
        r = np.where(np.isnan(fa) | np.isnan(fb), np.float32(np.nan), r).astype(np.float32)
        return g.to_fp8(r.reshape(-1), g.AWAY).reshape(H, W, C)
    return compare(op, fa, fb)


# ---- arg max / arg min ---------------------------------------------------------

def arg_reduce(x, axis, is_max, init, bits):
    """x: float64 (H, W, C) values (NaN allowed). The C64 lane scan of the
    arg max / min kernels: per lane (c % 64 for axis C, every position for H / W)
    the first strict improvement over `init`; indices modulo 2^bits; for axis C
    the smallest wrapped index among the lanes holding the overall best."""
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
    """Indices to the output element type: 'u8' / 'u16' / 'u32' / 'i64' (low
    bits, zero-extended) or 'f32'."""
    idx = np.asarray(idx, np.int64)
    if out_t == "f32":
        return idx.astype(np.float32)
    return {"u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "i64": np.int64}[out_t](idx & 0xFFFF)


# ---- reduce ---------------------------------------------------------------------

def reduce_half(bits, t, axes, is_max):
    """axes bitmask: 1 = C, 2 = W, 4 = H. Result clamped by the start value;
    NaN skipped; zero order -0 < +0."""
    b = np.asarray(bits, np.uint16)
    ax = tuple(a for a, bit in ((0, 4), (1, 2), (2, 1)) if axes & bit)
    moved = np.moveaxis(b, ax, tuple(range(3 - len(ax), 3)))
    keep = moved.shape[:3 - len(ax)]
    flat = moved.reshape(keep + (-1,))
    init = np.uint16(LOWEST[t] if is_max else MAX_FINITE[t])
    r = np.full(keep, init, np.uint16)
    for j in range(flat.shape[-1]):
        x = flat[..., j]
        fx, fr = half_to_f64(x, t), half_to_f64(r, t)
        r = np.where(zgt(fx, fr) if is_max else zgt(fr, fx), x, r)
    return r


# ---- pooling --------------------------------------------------------------------

def windows(x, kh, kw, sh, sw):
    """(Ho, Wo, kh*kw, C) taps in row-major window order."""
    Hi, Wi, C = x.shape
    Ho, Wo = (Hi - kh) // sh + 1, (Wi - kw) // sw + 1
    taps = [x[ty:ty + (Ho - 1) * sh + 1:sh, tx:tx + (Wo - 1) * sw + 1:sw] for ty in range(kh)
            for tx in range(kw)]
    return np.stack(taps, axis=2)


def running_max(taps, vals):
    """Running max over axis 2 seeded by the first tap (zero order -0 < +0);
    NaN never replaces it. Returns the selected entries of `taps`."""
    m, mv = taps[:, :, 0], vals[:, :, 0]
    for k in range(1, taps.shape[2]):
        up = zgt(vals[:, :, k], mv)
        m, mv = np.where(up, taps[:, :, k], m), np.where(up, vals[:, :, k], mv)
    return m


def maxpool_half(bits, t, kh, kw, sh, sw):
    tp = windows(np.asarray(bits, np.uint16), kh, kw, sh, sw)
    return running_max(tp, half_to_f64(tp, t)).astype(np.uint16)


def fp8_requant(m, c0, c2, clip, lo, hi):
    """Row-decode the fp8 maxima (rows of 8 channels per pixel), fma, clamp, RNE."""
    H, W, C = m.shape
    v = fp8_pixel_decode(m.reshape(-1), 0, H, W, C)
    r = g.fma32(v, np.broadcast_to(np.float32(c2), v.shape), np.broadcast_to(np.float32(c0), v.shape))
    if clip:
        r = np.where(np.float32(lo) > r, np.float32(lo), r)
        r = np.where(r > np.float32(hi), np.float32(hi), r)
        r = np.where(r == 0, np.float32(0), r).astype(np.float32)
    return g.to_fp8(np.asarray(r, np.float32).reshape(-1), g.EVEN).reshape(H, W, C)


def maxpool_fp8(codes, kh, kw, sh, sw, c0, c2, clip=0, lo=0.0, hi=0.0):
    tp = windows(np.asarray(codes, np.uint8), kh, kw, sh, sw)
    m = running_max(tp, g.fp8_to_f32(tp).astype(np.float64)).astype(np.uint8)
    return fp8_requant(m, c0, c2, clip, lo, hi)


def globalmaxpool_half(bits, t):
    """Max over H x W per channel, seeded by numeric_limits<T>::min()."""
    b = np.asarray(bits, np.uint16)
    C = b.shape[-1]
    taps = np.concatenate([np.full((1, C), MIN_NORMAL[t], np.uint16), b.reshape(-1, C)])[None, None]
    return running_max(taps, half_to_f64(taps, t)).reshape(C).astype(np.uint16)


def globalmaxpool_fp8(codes, c0, c1):
    c = np.asarray(codes, np.uint8)
    H, W, C = c.shape
    tp = c.reshape(1, 1, H * W, C)
    m = running_max(tp, g.fp8_to_f32(tp).astype(np.float64)).astype(np.uint8)
    return fp8_requant(m, c0, c1, 0, 0.0, 0.0).reshape(C)


# ---- top-k ----------------------------------------------------------------------

def topk(vals, raw, W, K, largest, sentinel, sentinel_raw, bits):
    """vals: float64 (rows, W) values; raw: the same elements as stored bits /
    integers. Returns (raw values (rows, K), indices (rows, K))."""
    rows = vals.shape[0]
    Wp = -(-W // 64) * 64
    mod = 1 << bits
    out_v = np.empty((rows, K), raw.dtype)
    out_i = np.empty((rows, K), np.int64)
    for r in range(rows):
        v = np.full(Wp, sentinel, np.float64)
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
                vu = zgt(x, m) if largest else zgt(m, x)         # value: zero order
                m, mr = np.where(vu, x, m), np.where(vu, lraw[i], mr)
            best, bestr = m[0], mr[0]
            for L in range(1, 64):
                if (zgt(m[L], best) if largest else zgt(best, m[L])):
                    best, bestr = m[L], mr[L]
            hit = m >= best if largest else m <= best
            res = int(np.where(hit, (idx + np.arange(64)) % mod, mod).min())
            out_v[r, k], out_i[r, k] = bestr, res
            if res < Wp:
                lanes.reshape(-1)[res] = sentinel
                lraw.reshape(-1)[res] = sentinel_raw
    return out_v, out_i
