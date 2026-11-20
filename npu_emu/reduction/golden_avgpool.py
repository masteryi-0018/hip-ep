# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for reduction/avgpool.h (in [H][W][C], out [Ho][Wo][C] or [C])."""

import numpy as np

from golden_float_cvt import (f32_to_fp8, fp8_decode_identity, ftz, kFtzIn, kFtzOut, load_float,
                              round_float, store_float)
from golden_int_round import Rnd, float_to_int, saturate, shift_round, wrap
from golden_mac import add, fma, fma_ftz, mac_exact
from golden_minmax import vclamp
from golden_mmac import mmac_add_bf16

f32 = np.float32


def out_hw(H, W, kh, kw, sh, sw):
    return (H - kh) // sh + 1, (W - kw) // sw + 1


def taps(x, kh, kw, sh, sw):
    """The window taps row by row: one strided [Ho][Wo][C] view per tap."""
    H, W = x.shape[:2]
    Ho, Wo = out_hw(H, W, kh, kw, sh, sw)
    for dy in range(kh):
        for dx in range(kw):
            yield x[dy:dy + (Ho - 1) * sh + 1:sh, dx:dx + (Wo - 1) * sw + 1:sw]


def tile_order(x, sv_h, sv_w):
    """The pixels of a global pooling in block order (sv_h x sv_w blocks): [H*W][C]."""
    H, W = x.shape[:2]
    rows = []
    for th in range(0, H, sv_h):
        for tw in range(0, W, sv_w):
            for y in range(th, th + sv_h):
                for xx in range(tw, tw + sv_w):
                    rows.append(x[y, xx])
    return np.array(rows)


def fsum(seq, start=None):
    """Sequential fp32 sum from +0."""
    acc = None
    for v in seq:
        v = np.asarray(v, f32)
        acc = np.add(np.zeros_like(v) if acc is None else acc, v, dtype=f32)
    return acc


def msum(seq):
    """MMAC accumulation from +0."""
    acc = None
    for v in seq:
        v = np.asarray(v, f32)
        acc = mmac_add_bf16(np.zeros_like(v) if acc is None else acc, v)
    return acc


def avgpool_16(bits, t, kh, kw, sh, sw, factor, c32=False):
    """Two alternating fma chains, added and rounded to t; subnormals flush to signed zeros."""
    if c32 and kh * kw == 1:
        H, W = bits.shape[:2]
        Ho, Wo = out_hw(H, W, kh, kw, sh, sw)
        return bits[:(Ho - 1) * sh + 1:sh, :(Wo - 1) * sw + 1:sw].copy()
    x = load_float(bits, t, kFtzIn)
    fac = round_float(f32(factor), t)
    acc = [None, None]
    for k, tap in enumerate(taps(x, kh, kw, sh, sw)):
        a = acc[k % 2] if acc[k % 2] is not None else np.zeros(tap.shape, f32)
        acc[k % 2] = fma_ftz(tap, fac, a, kFtzOut)
    if acc[1] is None:
        acc[1] = np.zeros_like(acc[0])
    return store_float(add(acc[0], acc[1]), t, Rnd.kHalfEven, kFtzIn)


def globalavgpool_16(bits, t, sv_h, sv_w, factor):
    """fp32 running sum, then fp32(s * factor); subnormals flush to signed zeros."""
    acc = None
    for v in tile_order(load_float(bits, t, kFtzIn), sv_h, sv_w):
        acc = ftz(add(np.zeros_like(v) if acc is None else acc, v))
    return store_float(fma_ftz(acc, factor, 0.0, kFtzOut), t)


def avgpool_fp8(codes, kh, kw, sh, sw, kh_sv, c0, c2, act=0, lo=0.0, hi=0.0):
    """Passes of kh_sv rows MMAC-summed from +0, folded as pass + total; fma, clamp, fp8 RNE."""
    x = fp8_decode_identity(codes, row=np.shape(codes)[-1])
    t = list(taps(x, kh, kw, sh, sw))
    total = None
    for p in range(0, kh, kh_sv):
        s = msum(t[p * kw:(p + kh_sv) * kw])
        total = s if total is None else np.add(s, total, dtype=f32)
    C = codes.shape[-1]
    c0 = np.broadcast_to(np.asarray(c0, f32), (C,))
    c2 = np.broadcast_to(np.asarray(c2, f32), (C,))
    v = fma(total, c2, c0)
    if act:
        v = vclamp(v, lo, hi)
    return f32_to_fp8(v)


def globalavgpool_fp8(codes, sv_h, sv_w, c0, c1):
    s = msum(tile_order(fp8_decode_identity(codes, row=np.shape(codes)[-1]), sv_h, sv_w))
    return f32_to_fp8(fma(s, c1, c0))


def as_kernel_int(x, t, as_i16):
    """as_i16: the 16 bits are read as int16 whatever the type."""
    x = np.asarray(x)
    return x.view(np.int16).astype(np.int64) if as_i16 else x.astype(np.int64)


def qdq_avgpool(x, out_t, kh, kw, sh, sw, c0, c1, shift, act=0, lo=0.0, hi=0.0,
                as_i16=False):
    xi = as_kernel_int(x, None, as_i16)
    s = sum(tap for tap in taps(xi, kh, kw, sh, sw))
    v = mac_exact(wrap(s, "i32"), c1, c0)
    if act:
        v = vclamp(v, lo, hi)
    return float_to_int(v, out_t, shift)


def qdq_global(x, out_t, c0, c1, shift, as_i16=False):
    xi = as_kernel_int(x, None, as_i16)
    s = wrap(xi.sum(axis=(0, 1)), "i32")
    return float_to_int(mac_exact(s, c1, c0), out_t, shift)


def avgpool_xint8(x, out_t, kh, kw, sh, sw, weight, shift, relu=False):
    """int32 sum * weight, shift with half to even (symmetric in sign), saturate."""
    s = sum(tap for tap in taps(np.asarray(x, np.int64), kh, kw, sh, sw))
    r = saturate(shift_round(wrap(s * int(weight), "i32"), shift, Rnd.kHalfEven), out_t)
    if relu and out_t == "i8":
        r = np.maximum(r, 0)
    return r
