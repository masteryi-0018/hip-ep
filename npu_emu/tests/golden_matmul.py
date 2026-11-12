# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the matmul group of npu_emu (include/npu_emu_matmul.h):
exact integer products, then the kernels' chain of one-rounding fp32 MACs,
the activation and the rounding / saturating conversion. Written independently
of the HIP code. Logical layouts: A M x K, B K x N, ifm H x W x Ci, wts
Kh x Kw x Ci x Co."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import golden as g

f32 = np.float32
ACT_NONE, ACT_CLIP, ACT_LRELU, ACT_PRELU = 0, 1, 2, 3


def mac(x, c, a):
    """x * c + a rounded once to fp32, integer x exact; broadcasting."""
    x, c, a = np.broadcast_arrays(np.asarray(x, np.int64), np.asarray(c, f32), np.asarray(a, f32))
    return g.mac_int(x.reshape(-1), c.reshape(-1), a.reshape(-1)).reshape(x.shape)


def fma(x, c, a):
    """fp32 x * c + a, one rounding; broadcasting."""
    x, c, a = np.broadcast_arrays(np.asarray(x, f32), np.asarray(c, f32), np.asarray(a, f32))
    return g.fma32(x.reshape(-1), c.reshape(-1), a.reshape(-1)).reshape(x.shape)


def activate_out(v, out_t, act, act0, act1, slope=None):
    v = np.asarray(v, f32)
    if act == ACT_CLIP:
        v = np.minimum(np.maximum(v, f32(act0)), f32(act1))
    elif act == ACT_LRELU:
        v = np.maximum(v, fma(v, act0, act1))
    elif act == ACT_PRELU:
        slope = np.asarray(slope, f32)
        offset = fma(slope, -f32(act0), act0)
        v = np.where(v < f32(act0), fma(v, slope, offset), v)
    lo, hi = g.INT_RANGE[out_t]
    return g.rne_int(v, lo, hi)


def gemm_qdq_int16x16(a, b, out_t, c0, c1, c2, c3, act=ACT_NONE, act0=0.0, act1=0.0):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    al, ah, bl, bh = a & 255, a >> 8, b & 255, b >> 8
    lo, mid, hi = al @ bl, ah @ bl + al @ bh, ah @ bh
    c2, c3 = f32(c2), f32(c3)
    t = mac(bl.sum(0, keepdims=True), c3, c0)
    t = mac(bh.sum(0, keepdims=True), c3 * f32(256), t)
    v = mac(lo, c2, t)
    v = mac(mid, c2 * f32(256), v)
    v = mac(hi, c2 * f32(65536), v)
    v = fma(a.sum(1, keepdims=True).astype(f32), c1, v)
    return activate_out(v, out_t, act, act0, act1)


def conv_int(ifm, wts, sh, sw):
    kh, kw = wts.shape[:2]
    win = sliding_window_view(np.asarray(ifm, np.int64), (kh, kw), axis=(0, 1))[::sh, ::sw]
    return np.einsum("hwcyx,yxco->hwo", win, np.asarray(wts, np.int64), optimize=True)


def window_sum(ifm, kh, kw, sh, sw, k_opt=False):
    """fp32 per-pixel channel sums added over the window, one fp32 add per tap:
    row-major, or (k_opt) column 0 + the remaining columns alternating between
    two accumulators, kh-major."""
    ch = np.asarray(ifm, np.int64).sum(2).astype(f32)
    ho, wo = (ch.shape[0] - kh) // sh + 1, (ch.shape[1] - kw) // sw + 1

    def add(taps):
        acc = None
        for y, x in taps:
            tap = ch[y:y + sh * (ho - 1) + 1:sh, x:x + sw * (wo - 1) + 1:sw]
            acc = tap.copy() if acc is None else (acc + tap).astype(f32)
        return acc

    if not k_opt or kw < 2:
        return add([(y, x) for y in range(kh) for x in range(kw)])
    first = kw % 2
    lo = add([(y, x) for y in range(kh) for x in range(first, kw, 2)])
    hi = add([(y, x) for y in range(kh) for x in range(first + 1, kw, 2)])
    if first:
        return ((add([(y, 0) for y in range(kh)]) + lo).astype(f32) + hi).astype(f32)
    return (lo + hi).astype(f32)


def conv_qdq(ifm, wts, out_t, sh, sw, c0, c1, c2, split16, act=ACT_NONE, act0=0.0, act1=0.0,
             slope=None, k_opt=False):
    ifm = np.asarray(ifm, np.int64)
    if split16:
        lo, hi = conv_int(ifm & 255, wts, sh, sw), conv_int(ifm >> 8, wts, sh, sw)
        v = fma(mac(hi, f32(256), lo.astype(f32)), c2, c0)
    else:
        v = mac(conv_int(ifm, wts, sh, sw), c2, c0)
    if c1 is not None:
        s = window_sum(ifm, wts.shape[0], wts.shape[1], sh, sw, k_opt)
        v = fma(s[..., None], c1, v)
    return activate_out(v, out_t, act, act0, act1, slope)
