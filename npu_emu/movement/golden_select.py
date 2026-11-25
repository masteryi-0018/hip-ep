# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for select.h."""

import numpy as np

from golden_block_fp import encode_bfp16
from golden_float_cvt import (bf16_to_f32, encode_float, f32_to_fp8, fp8_to_f32, load_float,
                              max_finite, min_normal, store_float)
from golden_int_round import Rnd
from golden_mac import fma_q, mul_q, sub_q
from golden_minmax import above, fp8_max, max16, min16, vclamp, veq, vge, vgt, vmax, vmin
from golden_mmac import decode_bfp16_bf16, fp8_decode_identity
from golden_quant import activate_f32, kQdqActClip, kQdqActNone, quant

BF16, F16 = "bf16", "f16"


# start values of the half-float reductions, as codes
MIN_NORMAL = {t: int(encode_float(min_normal(t), t)) for t in (BF16, F16)}
MAX_FINITE = {t: int(encode_float(max_finite(t), t)) for t in (BF16, F16)}
LOWEST = {t: int(encode_float(-max_finite(t), t)) for t in (BF16, F16)}


def _abs_half(bits):
    return np.asarray(bits, np.uint16) & np.uint16(0x7FFF)


def _neg_half(bits, t):
    """0 - x: both zeros give +0, a NaN flips its sign."""
    return store_float(sub_q(np.float32(0), load_float(bits, t)), t)


def _clip_half(bits, t, lo, hi):
    lob, hib = store_float(np.float32(lo), t), store_float(np.float32(hi), t)
    return min16(max16(bits, lob, t), hib, t)


def _fp8_unary(codes, op, scale, lo=0.0, hi=0.0):
    """op: 'abs' | 'neg' | 'clip'; rounds half away from zero."""
    c = np.asarray(codes, np.uint8)
    v = fp8_decode_identity(c & 0x7F if op == "abs" else c)
    v = mul_q(v, np.float32(scale))
    if op == "clip":
        v = vclamp(v, np.float32(lo), np.float32(hi))
    return f32_to_fp8(v, Rnd.kHalfAway)


def _bview(x, bc, H, W, C):
    """Broadcast an operand stored with shape (H|1, W|1, C|1) to (H, W, C)."""
    shape = (1 if bc & 4 else H, 1 if bc & 2 else W, 1 if bc & 1 else C)
    return np.broadcast_to(np.asarray(x).reshape(shape), (H, W, C))


def _fp8_pixel_decode(codes, bc, H, W, C):
    """Row-decode an fp8 operand: rows of 8 channels within each pixel."""
    cn = 1 if bc & 1 else C
    v = fp8_decode_identity(np.asarray(codes, np.uint8).reshape(-1), row=cn)
    return _bview(v, bc, H, W, C)


def _compare(op, fa, fb):
    return {"equal": veq, "greater": vgt, "greater_or_equal": vge}[op](fa, fb).astype(np.int8)


def _bcast_half(op, a, abc, b, bbc, t, H, W, C):
    ab, bb = _bview(a, abc, H, W, C), _bview(b, bbc, H, W, C)
    if op in ("max", "min"):
        return (max16 if op == "max" else min16)(ab, bb, t)
    return _compare(op, load_float(ab, t), load_float(bb, t))


def _bcast_fp8(op, a, abc, ascale, b, bbc, bscale, H, W, C):
    fa = mul_q(_fp8_pixel_decode(a, abc, H, W, C), np.float32(ascale))
    fb = mul_q(_fp8_pixel_decode(b, bbc, H, W, C), np.float32(bscale))
    if op in ("max", "min"):
        return f32_to_fp8((vmax if op == "max" else vmin)(fa, fb), Rnd.kHalfAway)
    return _compare(op, fa, fb)


def _arg_reduce(x, axis, is_max, init, bits):
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


def _index_out(idx, out_t):
    """Indices as 'u8' / 'u16' / 'u32' / 'i64' (zero-extended) or 'f32'."""
    idx = np.asarray(idx, np.int64)
    if out_t == "f32":
        return idx.astype(np.float32)
    dt = {"u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "i64": np.int64}[out_t]
    return dt(idx & 0xFFFF)


def _reduce_half(bits, t, axes, is_max):
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


def _windows(x, kh, kw, sh, sw):
    """(Ho, Wo, kh*kw, C) taps in row-major window order."""
    Hi, Wi, C = x.shape
    Ho, Wo = (Hi - kh) // sh + 1, (Wi - kw) // sw + 1
    taps = [x[ty:ty + (Ho - 1) * sh + 1:sh, tx:tx + (Wo - 1) * sw + 1:sw] for ty in range(kh)
            for tx in range(kw)]
    return np.stack(taps, axis=2)


def _running_max(taps, step):
    """Running max over axis 2 from the first tap: m = step(m, tap)."""
    m = taps[:, :, 0]
    for k in range(1, taps.shape[2]):
        m = step(m, taps[:, :, k])
    return m


def _max_half(t):
    return lambda m, x: max16(m, x, t)


def _maxpool_half(bits, t, kh, kw, sh, sw):
    tp = _windows(np.asarray(bits, np.uint16), kh, kw, sh, sw)
    return _running_max(tp, _max_half(t)).astype(np.uint16)


def _fp8_requant(m, c0, c2, act, lo, hi):
    """Row-decode the fp8 maxima, fma, the activation, round half to even."""
    H, W, C = m.shape
    v = _fp8_pixel_decode(m.reshape(-1), 0, H, W, C)
    return f32_to_fp8(activate_f32(fma_q(v, np.float32(c2), np.float32(c0)), act, lo, hi))


def _maxpool_fp8(codes, kh, kw, sh, sw, c0, c2, act=kQdqActNone, lo=0.0, hi=0.0):
    tp = _windows(np.asarray(codes, np.uint8), kh, kw, sh, sw)
    m = _running_max(tp, fp8_max).astype(np.uint8)
    return _fp8_requant(m, c0, c2, act, lo, hi)


def _maxpool_bfp16(blk, Hi, Wi, C, u8=False, qm=1.0, zp=0.0):
    """2x2 stride-2 max pooling of bfp16 in HCW(WC)64; bfp16 out, or u8 if u8."""
    v = decode_bfp16_bf16(blk).reshape(Hi, C // 64, Wi, 64)
    t = [v[dy::2, :, dx::2] for dy in (0, 1) for dx in (0, 1)]
    m = bf16_to_f32(max16(max16(t[0], t[1], BF16), max16(t[2], t[3], BF16), BF16)).reshape(-1)
    if not u8:
        return encode_bfp16(m, Rnd.kHalfEven)
    return quant(m, np.float32(qm), np.float32(zp), "u8").astype(np.uint8)


def _globalmaxpool_half(bits, t):
    """Max over H x W per channel and the smallest positive normal."""
    b = np.asarray(bits, np.uint16)
    C = b.shape[-1]
    taps = np.concatenate([np.full((1, C), MIN_NORMAL[t], np.uint16), b.reshape(-1, C)])[None, None]
    return _running_max(taps, _max_half(t)).reshape(C).astype(np.uint16)


def _globalmaxpool_fp8(codes, c0, c1):
    c = np.asarray(codes, np.uint8)
    H, W, C = c.shape
    tp = c.reshape(1, 1, H * W, C)
    m = _running_max(tp, fp8_max).astype(np.uint8)
    return _fp8_requant(m, c0, c1, kQdqActNone, 0.0, 0.0).reshape(C)


def _topk(vals, raw, W, K, largest, sentinel, sentinel_raw, bits):
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


# ---- one model per entry; t is 'bf16' / 'f16', codes are raw bits

def elementwise_abs_bf16(bits):
    return _abs_half(bits)


def elementwise_neg_bf16(bits, t):
    return _neg_half(bits, t)


def elementwise_clip_bf16(bits, t, f_min, f_max):
    return _clip_half(bits, t, f_min, f_max)


def elementwise_abs_fp8(codes, ifm_coeff):
    return _fp8_unary(codes, "abs", ifm_coeff)


def elementwise_neg_fp8(codes, ifm_coeff):
    return _fp8_unary(codes, "neg", ifm_coeff)


def elementwise_clip_fp8(codes, input_coeff, output_min, output_max):
    return _fp8_unary(codes, "clip", input_coeff, output_min, output_max)


def _bits_of(shape, full):
    return sum(bit for bit, s, f in zip((4, 2, 1), shape, full) if s == 1 and f != 1)


def _bcast_half_entry(op):
    def model(a, b, t):
        full = np.maximum(a.shape, b.shape)
        return _bcast_half(op, a, _bits_of(a.shape, full), b, _bits_of(b.shape, full), t, *full)
    return model


def _bcast_fp8_entry(op):
    def model(a, b, ifm1_coeff, ifm2_coeff):
        full = np.maximum(a.shape, b.shape)
        return _bcast_fp8(op, a, _bits_of(a.shape, full), ifm1_coeff, b, _bits_of(b.shape, full),
                          ifm2_coeff, *full)
    return model


elementwise_broadcasting3d_equal_bf16 = _bcast_half_entry("equal")
elementwise_broadcasting3d_greater_bf16 = _bcast_half_entry("greater")
elementwise_broadcasting3d_greater_or_equal_bf16 = _bcast_half_entry("greater_or_equal")
elementwise_broadcasting3d_max_bf16 = _bcast_half_entry("max")
elementwise_broadcasting3d_min_bf16 = _bcast_half_entry("min")
elementwise_broadcasting3d_equal_fp8 = _bcast_fp8_entry("equal")
elementwise_broadcasting3d_greater_fp8 = _bcast_fp8_entry("greater")
elementwise_broadcasting3d_greater_or_equal_fp8 = _bcast_fp8_entry("greater_or_equal")
elementwise_broadcasting3d_max_fp8 = _bcast_fp8_entry("max")
elementwise_broadcasting3d_min_fp8 = _bcast_fp8_entry("min")

AXIS_OF = {4: 0, 2: 1, 1: 2}   # axis_flags -> numpy axis of (H, W, C)


def _arg_int(x, axis_flags, out_t, is_max, bits):
    info = np.iinfo(np.asarray(x).dtype)
    init = float(info.min if is_max else info.max)
    return _index_out(_arg_reduce(np.asarray(x, np.float64), AXIS_OF[axis_flags], is_max, init,
                                  bits), out_t)


def _arg_half(bits_in, t, axis_flags, out_t, is_max):
    init = load_float(np.uint16((MIN_NORMAL if is_max else MAX_FINITE)[t]), t)
    return _index_out(_arg_reduce(load_float(bits_in, t), AXIS_OF[axis_flags], is_max, init,
                                  8 if out_t == "u8" else 16), out_t)


def arg_max_c64_a8(x, axis_flags, out_t):
    return _arg_int(x, axis_flags, out_t, True, 16)


def arg_max_c64_a16(x, axis_flags, out_t):
    return _arg_int(x, axis_flags, out_t, True, 8 if out_t == "u8" else 16)


def arg_max_c64_a16_int64(x, axis_flags):
    return _arg_int(x, axis_flags, "i64", True, 16)


def arg_max_c64_a32(x, axis_flags, out_t):
    return _arg_int(x, axis_flags, out_t, True, 16)


def arg_max_c64_bf16(bits, t, axis_flags, out_t):
    return _arg_half(bits, t, axis_flags, out_t, True)


def arg_max_c64_fp8(codes, axis_flags, out_t):
    vals = fp8_to_f32(codes).astype(np.float64)
    init = -float(max_finite("fp8"))
    return _index_out(_arg_reduce(vals, AXIS_OF[axis_flags], True, init, 16), out_t)


def arg_min_c64_a8(x, axis_flags, out_t):
    return _arg_int(x, axis_flags, out_t, False, 8 if out_t == "u8" else 16)


arg_min_c64_a16 = arg_min_c64_a8


def arg_min_c64_bf16(bits, t, axis_flags, out_t):
    return _arg_half(bits, t, axis_flags, out_t, False)


def reduce_max_c64_bf16(bits, t, axis_flags):
    return _reduce_half(bits, t, axis_flags, True)


def reduce_min_c64_bf16(bits, t, axis_flags):
    return _reduce_half(bits, t, axis_flags, False)


def maxpool_bf16(bits, t, kh, kw, sh, sw):
    return _maxpool_half(bits, t, kh, kw, sh, sw)


def maxpool_bf16_c32(bits, kh, kw, sh, sw):
    return _maxpool_half(bits, BF16, kh, kw, sh, sw)


def maxpool_fp8(codes, kh, kw, sh, sw, c0, c2, activation=kQdqActNone, act_lo=0.0, act_hi=0.0):
    """activation kQdqActClip clamps to [act_lo, act_hi]; c0 / c2 scalars or per channel."""
    return _maxpool_fp8(codes, kh, kw, sh, sw, c0, c2, activation, act_lo, act_hi)


def maxpool_bfp16(blk, Hi, Wi, C, u8=False, quant_multiplier=1.0, zero_point=0.0):
    return _maxpool_bfp16(blk, Hi, Wi, C, u8, quant_multiplier, zero_point)


def globalmaxpool_bf16(bits, t):
    return _globalmaxpool_half(bits, t)


def globalmaxpool_fp8(codes, c0, c1):
    return _globalmaxpool_fp8(codes, c0, c1)


def topk2d_bf16(bits, t, K, largest, index_bits):
    """(values, indices) of rows x W codes."""
    x = np.asarray(bits, np.uint16)
    f16 = t == F16
    sraw = (0xFC00 if f16 else 0xFF80) if largest else (0x7C00 if f16 else 0x7F80)
    return _topk(load_float(x, t), x, x.shape[1], K, largest, -np.inf if largest else np.inf,
                 np.uint16(sraw), index_bits)


def topk2d_a8(x, K, largest, index_bits):
    x = np.asarray(x)
    info = np.iinfo(x.dtype)
    sent = info.min if largest else info.max
    return _topk(x.astype(np.float64), x, x.shape[1], K, largest, float(sent),
                 np.array(sent).astype(x.dtype), index_bits)


def topk2d_a16(x, K, largest):
    return topk2d_a8(x, K, largest, 16)
