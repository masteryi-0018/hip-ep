# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for matmul/qdq_gemm.h."""

import numpy as np

from golden_mac import add, fma, mac_exact, mul
from golden_act import ACT_NONE, activate_out, f32

I8, I16_JOIN, I16_EXACT, I16_PRODUCT = "i8", "i16_join", "i16_exact", "i16_product"


def row_sum(a, kind):
    """fp32 activation row sum as the kernel forms it."""
    a = np.asarray(a, np.int64)
    if kind in (I16_JOIN, I16_PRODUCT):
        lo = (a & 255).sum(1, keepdims=True)
        hi = (a >> 8).sum(1, keepdims=True)
        return mac_exact(hi, f32(256), lo.astype(f32))
    return a.sum(1, keepdims=True).astype(f32)


def gemm_qdq_acc(a, b, kind, c0, c1=None, c2=None, c3=None):
    """fp32 accumulator of the QDQ GEMM entries; c0..c3 are per-column arrays."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    v = np.asarray(c0, f32)[None, :]
    if c3 is not None:
        v = mac_exact(b.sum(0, keepdims=True), np.asarray(c3, f32)[None, :], v)
    c2 = np.asarray(c2, f32)[None, :]
    if kind == I8:
        v = mac_exact(a @ b, c2, v)
    elif kind == I16_PRODUCT:
        g = mac_exact((a >> 8) @ b, f32(256), ((a & 255) @ b).astype(f32))
        v = fma(g, c2, v)
    else:
        v = mac_exact((a & 255) @ b, c2, v)
        v = mac_exact((a >> 8) @ b, mul(c2, f32(256)), v)
    if c1 is not None:
        v = fma(row_sum(a, kind), np.asarray(c1, f32)[None, :], v)
    return v


def gemm_qdq(a, b, out_t, kind, c0, c1, c2, c3, act=ACT_NONE, act0=0.0, act1=0.0):
    return activate_out(gemm_qdq_acc(a, b, kind, c0, c1, c2, c3), act, act0, act1, 0.0, out_t)


def gemm_qdq_int16x16(a, b, out_t, c0, c1, c2, c3, act=ACT_NONE, act0=0.0, act1=0.0):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    al, ah, bl, bh = a & 255, a >> 8, b & 255, b >> 8
    lo, mid, hi = al @ bl, ah @ bl + al @ bh, ah @ bh
    c2, c3 = f32(c2), f32(c3)
    t = mac_exact(bl.sum(0, keepdims=True), c3, c0)
    t = mac_exact(bh.sum(0, keepdims=True), mul(c3, f32(256)), t)
    v = mac_exact(lo, c2, t)
    v = mac_exact(mid, mul(c2, f32(256)), v)
    v = mac_exact(hi, mul(c2, f32(65536)), v)
    v = fma(a.sum(1, keepdims=True).astype(f32), c1, v)
    return activate_out(v, act, act0, act1, 0.0, out_t)


def gemm_hybrid(a, b, out_t, c0_a, c0_b, c1, c2, mode_a_cols, act=ACT_NONE, act0=0.0, act1=0.0,
                signed=True):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    lo_u, hi = a & 255, a >> 8
    lo_a = lo_u - (128 if signed else 0)
    s = mac_exact(hi.sum(1, keepdims=True), f32(256), lo_a.sum(1, keepdims=True).astype(f32))
    n = b.shape[1]
    mode_a = np.arange(n) < mode_a_cols
    c0 = np.where(mode_a, np.asarray(c0_a, f32), np.asarray(c0_b, f32))[None, :]
    c2 = np.asarray(c2, f32)[None, :]
    lo_prod = np.where(mode_a[None, :], lo_a @ b, lo_u @ b)
    v = mac_exact(lo_prod, c2, c0)
    v = mac_exact(hi @ b, mul(c2, f32(256)), v)
    v = fma(s, np.asarray(c1, f32)[None, :], v)
    return activate_out(v, act, act0, act1, 0.0, out_t)


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
    v = mac_exact(np.asarray(ifm3, np.int64), add_c1, g)
    return activate_out(v, ACT_NONE, 0.0, 0.0, 0.0, out_t)


def gemm_add_a16w8(a, b, residual, c0, c1, c2, add_c0, add_c1, add_c2):
    c0 = add(np.asarray(c0, f32), mul(f32(add_c0), f32(add_c2)))  # folded into c0
    g = gemm_qdq_acc(a, b, I16_EXACT, c0, c1, c2)
    v = mac_exact(np.asarray(residual, np.int64), add_c1, g)
    return activate_out(v, ACT_NONE, 0.0, 0.0, 0.0, "u16")
