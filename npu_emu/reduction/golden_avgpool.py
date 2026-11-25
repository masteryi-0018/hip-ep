# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for reduction/avgpool.h (in [H][W][C], out [Ho][Wo][C] or [C])."""

import numpy as np

from golden_float_cvt import f32_to_fp8, ftz, kFtzIn, kFtzOut, load_float, store_float
from golden_float_cvt import float_to_int
from golden_int_round import Rnd, srs, wrap
from golden_mac import add_q, fma_ftz, fma_q
from golden_minmax import vclamp
from golden_mmac import fp8_decode_identity, mmac_add_bf16
from golden_quant import activate_f32, dequant, kQdqActClip, requant

f32 = np.float32


def _out_hw(H, W, kh, kw, sh, sw):
    return (H - kh) // sh + 1, (W - kw) // sw + 1


def _taps(x, kh, kw, sh, sw):
    """The window taps row by row: one strided [Ho][Wo][C] view per tap."""
    H, W = x.shape[:2]
    Ho, Wo = _out_hw(H, W, kh, kw, sh, sw)
    for dy in range(kh):
        for dx in range(kw):
            yield x[dy:dy + (Ho - 1) * sh + 1:sh, dx:dx + (Wo - 1) * sw + 1:sw]


def _tile_order(x, sv_h, sv_w):
    """The pixels of a global pooling in block order (sv_h x sv_w blocks): [H*W][C]."""
    H, W = x.shape[:2]
    rows = []
    for th in range(0, H, sv_h):
        for tw in range(0, W, sv_w):
            for y in range(th, th + sv_h):
                for xx in range(tw, tw + sv_w):
                    rows.append(x[y, xx])
    return np.array(rows)


def _fsum(seq, start=None):
    """Sequential fp32 sum from +0."""
    acc = None
    for v in seq:
        v = np.asarray(v, f32)
        acc = np.add(np.zeros_like(v) if acc is None else acc, v, dtype=f32)
    return acc


def _msum(seq):
    """MMAC accumulation from +0."""
    acc = None
    for v in seq:
        v = np.asarray(v, f32)
        acc = mmac_add_bf16(np.zeros_like(v) if acc is None else acc, v)
    return acc


def avgpool_bf16(bits, t, kh, kw, sh, sw, factor_raw):
    """Two alternating fma chains, added and rounded to t; subnormals flush to signed zeros.
    factor_raw: the bits of the factor in t."""
    x = load_float(bits, t, kFtzIn)
    fac = load_float(np.uint16(factor_raw), t)
    acc = [None, None]
    for k, tap in enumerate(_taps(x, kh, kw, sh, sw)):
        a = acc[k % 2] if acc[k % 2] is not None else np.zeros(tap.shape, f32)
        acc[k % 2] = fma_ftz(tap, fac, a, kFtzOut)
    if acc[1] is None:
        acc[1] = np.zeros_like(acc[0])
    return store_float(add_q(acc[0], acc[1]), t, Rnd.kHalfEven, kFtzIn)


def avgpool_bf16_c32(bits, kh, kw, sh, sw, factor_raw):
    """avgpool_bf16 for bf16; a 1 x 1 window copies."""
    if kh * kw == 1:
        H, W = bits.shape[:2]
        Ho, Wo = _out_hw(H, W, kh, kw, sh, sw)
        return bits[:(Ho - 1) * sh + 1:sh, :(Wo - 1) * sw + 1:sw].copy()
    return avgpool_bf16(bits, "bf16", kh, kw, sh, sw, factor_raw)


def globalavgpool_bf16(bits, t, sv_h, sv_w, factor_val):
    """fp32 running sum, then fp32(s * factor_val); subnormals flush to signed zeros."""
    acc = None
    for v in _tile_order(load_float(bits, t, kFtzIn), sv_h, sv_w):
        acc = ftz(add_q(np.zeros_like(v) if acc is None else acc, v))
    return store_float(fma_ftz(acc, factor_val, 0.0, kFtzOut), t)


def globalavgpool_bf16_c32(bits, sv_h, sv_w, factor):
    return globalavgpool_bf16(bits, "bf16", sv_h, sv_w, factor)


def avgpool_fp8(codes, kh, kw, sh, sw, kh_sv, c0, c2, act=0, lo=0.0, hi=0.0):
    """Passes of kh_sv rows MMAC-summed from +0, folded as pass + total; fma, clamp, fp8 RNE."""
    x = fp8_decode_identity(codes, row=np.shape(codes)[-1])
    t = list(_taps(x, kh, kw, sh, sw))
    total = None
    for p in range(0, kh, kh_sv):
        s = _msum(t[p * kw:(p + kh_sv) * kw])
        total = s if total is None else add_q(s, total)
    C = codes.shape[-1]
    c0 = np.broadcast_to(np.asarray(c0, f32), (C,))
    c2 = np.broadcast_to(np.asarray(c2, f32), (C,))
    v = fma_q(total, c2, c0)
    if act:
        v = vclamp(v, lo, hi)
    return f32_to_fp8(v)


def globalavgpool_fp8(codes, sv_h, sv_w, c0, c1):
    s = _msum(_tile_order(fp8_decode_identity(codes, row=np.shape(codes)[-1]), sv_h, sv_w))
    return f32_to_fp8(fma_q(s, c1, c0))


def _qdq_avgpool(x, t, kh, kw, sh, sw, c0, c1, shift_res, act=0, act_c0=0.0, act_c1=0.0):
    s = sum(tap for tap in _taps(np.asarray(x).astype(np.int64), kh, kw, sh, sw))
    s = wrap(s, "i32")
    if not act:
        return requant(s, c1, c0, t, shift_res)
    v = activate_f32(dequant(s, c1, c0), kQdqActClip, act_c0, act_c1)
    return float_to_int(v, t, shift_res)


def qdq_avgpool_a8(x, t, kh, kw, sh, sw, c0, c1, shift_res, act=0, act_c0=0.0, act_c1=0.0):
    """t: the input and output type (i8 / u8)."""
    return _qdq_avgpool(x, t, kh, kw, sh, sw, c0, c1, shift_res, act, act_c0, act_c1)


def qdq_avgpool_a16(x, t, kh, kw, sh, sw, c0, c1, shift_res):
    return _qdq_avgpool(x, t, kh, kw, sh, sw, c0, c1, shift_res)


def _qdq_globalavgpool(x, t, c0, c1, shift_res):
    s = wrap(np.asarray(x).astype(np.int64).sum(axis=(0, 1)), "i32")
    return requant(s, c1, c0, t, shift_res)


def qdq_globalavgpool_a8(x, t, c0, c1, shift_res):
    return _qdq_globalavgpool(x, t, c0, c1, shift_res)


def qdq_globalavgpool_a16(x, t, c0, c1, shift_res):
    return _qdq_globalavgpool(x, t, c0, c1, shift_res)


def avgpool_xint8(x, t, kh, kw, sh, sw, weight, shift_res):
    """int32 sum * weight, shift with half to even (symmetric in sign; a negative shift_res
    shifts left), saturate to t."""
    s = sum(tap for tap in _taps(np.asarray(x, np.int64), kh, kw, sh, sw))
    return srs(wrap(s * int(weight), "i32"), shift_res, t)


def avgpool_xint8_fused(x, t, kh, kw, sh, sw, weight, shift_res, activation):
    r = avgpool_xint8(x, t, kh, kw, sh, sw, weight, shift_res)
    return np.maximum(r, 0) if activation == 1 and t == "i8" else r
