# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for matmul/qdq_gemm.h."""

import numpy as np

from golden_act import f32
from golden_float_cvt import int_f32
from golden_mac import add_q, fma_q, mac_exact, mul_q
from golden_quant import activate_out, kQdqActNone, requant

I8, I16_JOIN, I16_EXACT, I16_PRODUCT = "i8", "i16_join", "i16_exact", "i16_product"


def _row_sum(a, kind):
    """fp32 activation row sum as the kernel forms it."""
    a = np.asarray(a, np.int64)
    if kind in (I16_JOIN, I16_PRODUCT):
        lo = (a & 255).sum(1, keepdims=True)
        hi = (a >> 8).sum(1, keepdims=True)
        return mac_exact(hi, f32(256), int_f32(lo))
    return int_f32(a.sum(1, keepdims=True))


def int4(b):
    """The int4 weights of one-per-byte storage: the low nibble, signed for int8 storage."""
    b = np.asarray(b)
    v = b.astype(np.int64) & 15
    return ((v ^ 8) - 8) if b.dtype == np.int8 else v


def gemm_qdq_acc(a, b, kind, c0, c1=None, c2=None, c3=None):
    """fp32 accumulator of the QDQ GEMM entries; c0..c3 are per-column arrays or scalars."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    n = b.shape[1]
    col = lambda c: np.broadcast_to(np.asarray(c, f32), (n,))[None, :]  # noqa: E731
    v = col(c0)
    if c3 is not None:
        v = mac_exact(b.sum(0, keepdims=True), col(c3), v)
    c2 = col(c2)
    if kind == I8:
        v = mac_exact(a @ b, c2, v)
    elif kind == I16_PRODUCT:
        g = mac_exact((a >> 8) @ b, f32(256), int_f32((a & 255) @ b))
        v = fma_q(g, c2, v)
    else:
        v = mac_exact((a & 255) @ b, c2, v)
        v = mac_exact((a >> 8) @ b, mul_q(c2, f32(256)), v)
    if c1 is not None:
        v = fma_q(_row_sum(a, kind), col(c1), v)
    return v


def _gemm_qdq(a, b, out_t, kind, c0, c1, c2, c3, act, act0, act1):
    return activate_out(gemm_qdq_acc(a, b, kind, c0, c1, c2, c3), act, act0, act1, 0.0, out_t)


def gemm_a8w8(a, b, out_t, c0, c1, c2, act=kQdqActNone, act0=0.0, act1=0.0):
    return _gemm_qdq(a, b, out_t, I8, c0, c1, c2, None, act, act0, act1)


gemm_a8w8_stream = gemm_a8w8


def gemm_a8w4(a, b, out_t, c0, c1, c2, act=kQdqActNone, act0=0.0, act1=0.0):
    return _gemm_qdq(a, int4(b), out_t, I8, c0, c1, c2, None, act, act0, act1)


def gemm_a8a8(a, b, out_t, c0, c1, c2, c3, act=kQdqActNone, act0=0.0, act1=0.0):
    return _gemm_qdq(a, b, out_t, I8, c0, c1, c2, c3, act, act0, act1)


def gemm_qdq_a8a8_transpose(a, b, out_t, c0, c1, c2, c3):
    return gemm_a8a8(a, b, out_t, c0, c1, c2, c3)


def gemm_a16w8(a, b, out_t, c0, c1, c2, act=kQdqActNone, act0=0.0, act1=0.0):
    return _gemm_qdq(a, b, out_t, I16_JOIN, c0, c1, c2, None, act, act0, act1)


def gemm_a16w8_stream(a, b, out_t, c0, c1, c2, act=kQdqActNone, act0=0.0, act1=0.0):
    return _gemm_qdq(a, b, out_t, I16_EXACT, c0, c1, c2, None, act, act0, act1)


def gemm_a16a8(a, b, out_t, c0, c1, c2, c3, act=kQdqActNone, act0=0.0, act1=0.0):
    return _gemm_qdq(a, b, out_t, I16_JOIN, c0, c1, c2, c3, act, act0, act1)


def gemm_a16w4(a, b, out_t, c0, c1, c2, act=kQdqActNone, act0=0.0, act1=0.0):
    return _gemm_qdq(a, int4(b), out_t, I16_PRODUCT, c0, c1, c2, None, act, act0, act1)


def gemm_qdq_int16x16(a, b, out_t, c0, c1, c2, c3, act=kQdqActNone, act0=0.0, act1=0.0):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    al, ah, bl, bh = a & 255, a >> 8, b & 255, b >> 8
    lo, mid, hi = al @ bl, ah @ bl + al @ bh, ah @ bh
    c2, c3 = f32(c2), f32(c3)
    t = mac_exact(bl.sum(0, keepdims=True), c3, c0)
    t = mac_exact(bh.sum(0, keepdims=True), mul_q(c3, f32(256)), t)
    v = mac_exact(lo, c2, t)
    v = mac_exact(mid, mul_q(c2, f32(256)), v)
    v = mac_exact(hi, mul_q(c2, f32(65536)), v)
    v = fma_q(int_f32(a.sum(1, keepdims=True)), c1, v)
    return activate_out(v, act, act0, act1, 0.0, out_t)


gemm_qdq_a16a16_transpose = gemm_qdq_int16x16


def gemm_a16w8_hybrid(a, b, out_t, mode_a_cols, c0_a, c0_b, c1, c2, act=kQdqActNone, act0=0.0,
                      act1=0.0):
    signed = np.asarray(a).dtype == np.int16
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    lo_u, hi = a & 255, a >> 8
    lo_a = lo_u - (128 if signed else 0)
    s = mac_exact(hi.sum(1, keepdims=True), f32(256), int_f32(lo_a.sum(1, keepdims=True)))
    n = b.shape[1]
    mode_a = np.arange(n) < mode_a_cols
    c0 = np.where(mode_a, np.asarray(c0_a, f32), np.asarray(c0_b, f32))[None, :]
    c2 = np.asarray(c2, f32)[None, :]
    lo_prod = np.where(mode_a[None, :], lo_a @ b, lo_u @ b)
    v = mac_exact(lo_prod, c2, c0)
    v = mac_exact(hi @ b, mul_q(c2, f32(256)), v)
    v = fma_q(s, np.asarray(c1, f32)[None, :], v)
    return activate_out(v, act, act0, act1, 0.0, out_t)


gemm_a16w8_hybrid_channelwise = gemm_a16w8_hybrid


def qdq_gemm_add_a8w8(a, b, ifm3, out_t, c0, c1, c2, add_c0, add_c1, add_c2, kpass):
    c0, c2 = np.asarray(c0, f32), np.asarray(c2, f32)
    c1 = None if c1 is None else np.asarray(c1, f32)
    if kpass:
        c0 = fma_q(c0, add_c0, add_c2)
        c1 = None if c1 is None else fma_q(c1, add_c0, f32(0))
        c2 = fma_q(c2, add_c0, f32(0))
    g = gemm_qdq_acc(a, b, I8, c0, c1, c2)
    if not kpass:
        g = fma_q(g, add_c0, add_c2)
    return requant(np.asarray(ifm3, np.int64), add_c1, g, out_t)


def gemm_add_a16w8(a, b, residual, c0, c1, c2, add_c0, add_c1, add_c2):
    c0 = add_q(np.asarray(c0, f32), mul_q(f32(add_c0), f32(add_c2)))  # folded into c0
    g = gemm_qdq_acc(a, b, I16_EXACT, c0, c1, c2)
    return requant(np.asarray(residual, np.int64), add_c1, g, "u16")
