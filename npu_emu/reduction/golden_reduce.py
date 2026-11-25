# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for reduction/reduce.h (tensors H x W x C, reduced dimensions dropped)."""

import numpy as np

from golden_bits import quiet
from golden_float_cvt import f32_to_fp8, float_to_int, int_f32, kSat, load_float
from golden_float_cvt import round_float, store_float
from golden_int_round import Rnd, srs, wrap, wrap_add64, wrap_mul64
from golden_mac import add_q, fma_q, mul16_f32, mul_q, vaddr_f32_reduce
from golden_mmac import fp8_decode_identity
from golden_nlf import hw_sqrt16, sqrt_q
from golden_quant import dequant, dq_offset, quant, requant

f32 = np.float32

# Step nests per axis_flags, outermost first (g = 64-channel groups).
ORDER_SUM = {1: "g", 2: "w", 3: "gw", 4: "h", 5: "gh", 6: "wh", 7: "ghw"}
ORDER_PROD = {1: "g", 2: "w", 3: "wg", 4: "h", 5: "hg", 6: "hw", 7: "hwg"}
ORDER_QDQ_PROD = {**ORDER_PROD, 6: "wh"}
ORDER_L2 = {1: "g", 2: "w", 3: "wg", 4: "h", 5: "hg", 6: "wh", 7: "hwg"}
ORDER_MRS = {1: "g", 2: "w", 3: "gw", 4: "h", 5: "hg", 6: "hw", 7: "hwg"}


def _out_shape(H, W, C, axis):
    return tuple(n for n, b in ((H, 4), (W, 2), (C, 1)) if not axis & b)


def _steps(x, axis, order, pad=0):
    """x [H][W][C] -> (terms [S][oH][oW][L], (oH, oW)); L = 64 lanes if C is reduced, else C."""
    H, W, C = x.shape
    if axis & 1:
        G = -(-C // 64)
        xp = np.full((H, W, G * 64), pad, x.dtype)
        xp[..., :C] = x
        x4 = xp.reshape(H, W, G, 64)
        names = "hwgl"
    else:
        x4 = x
        names = "hwc"
    kept = [d for d, b in (("h", 4), ("w", 2)) if not axis & b]
    perm = [names.index(d) for d in order] + [names.index(d) for d in kept] + [len(names) - 1]
    t = x4.transpose(perm)
    S = int(np.prod([x4.shape[names.index(d)] for d in order]))
    oH, oW = (1 if axis & 4 else H), (1 if axis & 2 else W)
    return t.reshape(S, oH, oW, t.shape[-1]), (oH, oW)


def _finish_shape(v, oHW, axis, C):
    """[oH][oW][L] or [oH][oW] -> the logical output shape."""
    oH, oW = oHW
    shape = [n for n, b in ((oH, 4), (oW, 2)) if not axis & b]
    if not axis & 1:
        shape.append(C)
    return np.asarray(v).reshape(shape)


def _lanes(acc):
    """Lanes l and l + 32 added, then vector reduce-add over the last axis."""
    return vaddr_f32_reduce(add_q(acc[..., :32], acc[..., 32:]))


def reduce_sum_c64_bf16(bits, t, axis, scale):
    """bits [H][W][C] (bf16 / f16 patterns) -> bits."""
    x = load_float(bits, t)
    C = x.shape[2]
    T, ohw = _steps(x, axis, ORDER_SUM[axis])
    s = f32(scale)
    acc = mul_q(T[0], s)  # first step: the bare product
    for k in range(1, len(T)):
        acc = fma_q(T[k], s, acc)
    if axis & 1:
        acc = _lanes(acc)
    return _finish_shape(store_float(acc, t), ohw, axis, C)


def reduce_sum_c64_fp8(codes, h_tiles, w_tiles, axis, scale):
    """codes [H][W][C] -> codes."""
    x = fp8_decode_identity(codes, row=np.shape(codes)[-1])
    H, W, C = x.shape
    if axis == 7:
        G = -(-C // 64)
        xp = np.zeros((H, W, G * 64), f32)
        xp[..., :C] = x
        T = xp.reshape(h_tiles, H // h_tiles, w_tiles, W // w_tiles, G, 64)
        T = T.transpose(0, 2, 4, 1, 3, 5).reshape(-1, 1, 1, 64)
        ohw = (1, 1)
    else:
        T, ohw = _steps(x, axis, ORDER_SUM[axis])
    s = f32(scale)
    acc = np.zeros(T.shape[1:], f32)
    for k in range(len(T)):
        acc = fma_q(T[k], s, acc)
    if axis & 1:
        acc = _lanes(acc)
    return _finish_shape(f32_to_fp8(acc, Rnd.kHalfEven, kSat), ohw, axis, C)


def _sum_affine(x, axis, kind, out_t, c0, c1, c2):
    x = np.asarray(x).astype(np.int64)
    axes = tuple(a for a, b in ((0, 4), (1, 2), (2, 1)) if axis & b)
    S = x.sum(axis=axes)
    if axis == 7:  # int64 wrap-around [default]
        v = wrap_add64(wrap_mul64(c0, wrap(S, "i32") if kind == "a8" else S), c2)
    else:
        v = wrap(wrap(c2, "i32") + c0 * S, "i32")
    return srs(v, c1, out_t)


def reduce_sum_c64_a8(x, axis, out_t, c0, c1, c2):
    return _sum_affine(x, axis, "a8", out_t, c0, c1, c2)


def reduce_sum_c64_a16(x, axis, out_t, c0, c1, c2):
    return _sum_affine(x, axis, "a16", out_t, c0, c1, c2)


def qdq_reduce_sum_c64_a32(x, in_t, axis, out_t, c0, c1, shift_res):
    """x: 32-bit integers of type in_t (i32 / u32); the sum wraps to in_t."""
    x = np.asarray(x).astype(np.int64)
    axes = tuple(a for a, b in ((0, 4), (1, 2), (2, 1)) if axis & b)
    S = x.sum(axis=axes)
    return requant(wrap(S, in_t), c1, c0, out_t, -shift_res)


@quiet
def _prod16(x, t, axis, order):
    """fp32 values x (exact 16-bit) -> fp32 values (16-bit)."""
    C = x.shape[2]
    T, ohw = _steps(x, axis, order[axis])
    acc = np.ones(T.shape[1:], f32)
    for k in range(len(T)):
        acc = mul16_f32(acc, T[k], t)
    if axis & 1:
        v = acc
        while v.shape[-1] > 1:
            h = v.shape[-1] // 2
            v = mul16_f32(v[..., :h], v[..., h:], t)
        acc = v[..., 0]
    return _finish_shape(acc, ohw, axis, C)


def _l2_16(x, t, axis):
    """fp32 values x (exact 16-bit) -> fp32 values (16-bit)."""
    C = x.shape[2]
    T, ohw = _steps(x, axis, ORDER_L2[axis])
    acc = np.zeros(T.shape[1:], f32)
    for k in range(len(T)):
        acc = fma_q(T[k], T[k], acc)
    if axis & 1:
        r = round_float(sqrt_q(_lanes(acc)), t)
    else:
        r = load_float(hw_sqrt16(store_float(acc, t), t), t)
    return _finish_shape(r, ohw, axis, C)


def reduce_prod_bf16(bits, t, axis):
    return store_float(_prod16(load_float(bits, t), t, axis, ORDER_PROD), t)


def reduce_l2_bf16(bits, t, axis):
    return store_float(_l2_16(load_float(bits, t), t, axis), t)


@quiet
def _dq16(x, scale, zp, rt):
    """Dequantized and rounded to rt."""
    return round_float(dequant(x, scale, dq_offset(zp, scale)), rt)


@quiet
def _q_int(y, scale, zp, out_t):
    return quant(y, scale, zp, out_t)


def qdq_reduce_prod_a8(x, axis, dq_zp, dq_sc, q_zp, q_sc, out_t):
    return _q_int(_prod16(_dq16(x, dq_sc, dq_zp, "bf16"), "bf16", axis, ORDER_QDQ_PROD), q_sc,
                  q_zp, out_t)


qdq_reduce_prod_a16 = qdq_reduce_prod_a8


def qdq_reduce_l2_a8(x, axis, rt, dq_zp, dq_sc, q_zp, q_sc, out_t):
    """rt: the compute type."""
    return _q_int(_l2_16(_dq16(x, dq_sc, dq_zp, rt), rt, axis), q_sc, q_zp, out_t)


qdq_reduce_l2_a16 = qdq_reduce_l2_a8


def _mul_reducesum(x1, x2, axis, c, out_t, a16):
    c0, c1, c2, c3 = (f32(v) for v in c)
    x1 = np.asarray(x1).astype(np.int64)
    H, W, C = x1.shape
    x2 = np.broadcast_to(np.asarray(x2).astype(np.int64), x1.shape)
    T1, ohw = _steps(x1, axis, ORDER_MRS[axis])
    T2, _ = _steps(np.ascontiguousarray(x2), axis, ORDER_MRS[axis])
    acc = np.zeros(T1.shape[1:], f32)
    for i1, i2 in zip(T1, T2):
        f1, f2 = i1.astype(f32), i2.astype(f32)  # exact (16-bit)
        if a16:
            p = int_f32(i1 * i2)
            r = mul_q(p, c0)
            r = fma_q(f1, c1, r)
            r = fma_q(f2, c2, r)
            acc = add_q(acc, r)
        else:
            t = fma_q(f2, c0, c1)
            acc = fma_q(f2, c2, acc)
            acc = fma_q(f1, t, acc)
    v = _lanes(acc) if axis & 1 else acc
    v = add_q(v, c3)
    return _finish_shape(float_to_int(v, out_t), ohw, axis, C)


def qdq_elementwise_broadcasting3d_mul_reducesum_a8(x1, x2, axis, c, out_t):
    return _mul_reducesum(x1, x2, axis, c, out_t, False)


def qdq_elementwise_broadcasting3d_mul_reducesum_a16(x1, x2, axis, c, out_t):
    return _mul_reducesum(x1, x2, axis, c, out_t, True)


def qdq_cumsum(x, axis, exclusive, reverse, c0, c1, out_t):
    x = np.asarray(x).astype(np.int64)
    w = np.flip(x, axis) if reverse else x
    s = np.cumsum(w, axis)
    if exclusive:
        s = s - w
    if reverse:
        s = np.flip(s, axis)
    return requant(wrap(s, "i32"), c1, c0, out_t)
