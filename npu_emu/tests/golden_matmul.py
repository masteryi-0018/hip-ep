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


# ---- QDQ integer GEMMs (gemm_a8w8 / a16w8 families) ------------------------------

I8, I16_JOIN, I16_EXACT, I16_PRODUCT = "i8", "i16_join", "i16_exact", "i16_product"


def row_sum(a, kind):
    """fp32 activation row sum as the kernel forms it."""
    a = np.asarray(a, np.int64)
    if kind in (I16_JOIN, I16_PRODUCT):
        lo = (a & 255).sum(1, keepdims=True)
        hi = (a >> 8).sum(1, keepdims=True)
        return mac(hi, f32(256), lo.astype(f32))
    return a.sum(1, keepdims=True).astype(f32)


def gemm_qdq_acc(a, b, kind, c0, c1=None, c2=None, c3=None):
    """fp32 accumulator of the QDQ GEMM entries; c0..c3 are per-column arrays."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    v = np.asarray(c0, f32)[None, :]
    if c3 is not None:
        v = mac(b.sum(0, keepdims=True), np.asarray(c3, f32)[None, :], v)
    c2 = np.asarray(c2, f32)[None, :]
    if kind == I8:
        v = mac(a @ b, c2, v)
    elif kind == I16_PRODUCT:
        g = mac((a >> 8) @ b, f32(256), ((a & 255) @ b).astype(f32))
        v = fma(g, c2, v)
    else:
        v = mac((a & 255) @ b, c2, v)
        v = mac((a >> 8) @ b, c2 * f32(256), v)
    if c1 is not None:
        v = fma(row_sum(a, kind), np.asarray(c1, f32)[None, :], v)
    return v


def gemm_qdq(a, b, out_t, kind, c0, c1, c2, c3, act=ACT_NONE, act0=0.0, act1=0.0):
    return activate_out(gemm_qdq_acc(a, b, kind, c0, c1, c2, c3), out_t, act, act0, act1)


def gemm_hybrid(a, b, out_t, c0_a, c0_b, c1, c2, mode_a_cols, act=ACT_NONE, act0=0.0, act1=0.0,
                signed=True):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    lo_u, hi = a & 255, a >> 8
    lo_a = lo_u - (128 if signed else 0)
    s = mac(hi.sum(1, keepdims=True), f32(256), lo_a.sum(1, keepdims=True).astype(f32))
    n = b.shape[1]
    mode_a = np.arange(n) < mode_a_cols
    c0 = np.where(mode_a, np.asarray(c0_a, f32), np.asarray(c0_b, f32))[None, :]
    c2 = np.asarray(c2, f32)[None, :]
    lo_prod = np.where(mode_a[None, :], lo_a @ b, lo_u @ b)
    v = mac(lo_prod, c2, c0)
    v = mac(hi @ b, c2 * f32(256), v)
    v = fma(s, np.asarray(c1, f32)[None, :], v)
    return activate_out(v, out_t, act, act0, act1)


def gemm_add_a8w8(a, b, ifm3, out_t, c0, c1, c2, add_c0, add_c1, add_c2, kpass):
    c0, c2 = np.asarray(c0, f32), np.asarray(c2, f32)
    c1 = None if c1 is None else np.asarray(c1, f32)
    if kpass:
        c0 = fma(c0, add_c0, add_c2)
        c1 = None if c1 is None else fma(c1, add_c0, f32(0))
        c2 = fma(c2, add_c0, f32(0))
    g = gemm_qdq_acc(a, b, I8, c0, c1, c2)
    if not kpass:
        g = fma(g, add_c0, add_c2)
    return activate_out(mac(np.asarray(ifm3, np.int64), add_c1, g), out_t, ACT_NONE, 0.0, 0.0)


def gemm_add_a16w8(a, b, residual, c0, c1, c2, add_c0, add_c1, add_c2):
    g = gemm_qdq_acc(a, b, I16_EXACT, c0, c1, c2)
    partial = fma(np.full(g.shape, add_c2, f32), add_c0, g)
    return activate_out(mac(np.asarray(residual, np.int64), add_c1, partial), "u16", ACT_NONE, 0.0, 0.0)
