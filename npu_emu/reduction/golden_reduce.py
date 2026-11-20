# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for reduction/reduce.h (tensors H x W x C, reduced dimensions dropped)."""

import numpy as np

from golden_float_cvt import f32_to_fp8, fp8_decode_identity, kSat, load_float, round_float
from golden_float_cvt import store_float
from golden_int_round import Rnd, float_to_int, saturate, shift_round, wrap
from golden_mac import add, fma, fneg, mac_exact, mul, mul16_f32, sqrt, vaddr_f32_reduce
from golden_nlf import hw_sqrt16, quiet

f32 = np.float32

# Step nests per axis_flags, outermost first (g = 64-channel groups).
ORDER_SUM = {1: "g", 2: "w", 3: "gw", 4: "h", 5: "gh", 6: "wh", 7: "ghw"}
ORDER_PROD = {1: "g", 2: "w", 3: "wg", 4: "h", 5: "hg", 6: "hw", 7: "hwg"}
ORDER_QDQ_PROD = {**ORDER_PROD, 6: "wh"}
ORDER_L2 = {1: "g", 2: "w", 3: "wg", 4: "h", 5: "hg", 6: "wh", 7: "hwg"}
ORDER_MRS = {1: "g", 2: "w", 3: "gw", 4: "h", 5: "hg", 6: "hw", 7: "hwg"}


def out_shape(H, W, C, axis):
    return tuple(n for n, b in ((H, 4), (W, 2), (C, 1)) if not axis & b)


def steps(x, axis, order, pad=0):
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


def finish_shape(v, oHW, axis, C):
    """[oH][oW][L] or [oH][oW] -> the logical output shape."""
    oH, oW = oHW
    shape = [n for n, b in ((oH, 4), (oW, 2)) if not axis & b]
    if not axis & 1:
        shape.append(C)
    return np.asarray(v).reshape(shape)


def lanes(acc):
    """Lanes l and l + 32 added, then vector reduce-add over the last axis."""
    return vaddr_f32_reduce(add(acc[..., :32], acc[..., 32:]))


def sum16(bits, t, axis, scale):
    """bits [H][W][C] (bf16 / f16 patterns) -> bits."""
    x = load_float(bits, t)
    C = x.shape[2]
    T, ohw = steps(x, axis, ORDER_SUM[axis])
    s = f32(scale)
    acc = mul(T[0], s)  # first step: the bare product
    for k in range(1, len(T)):
        acc = fma(T[k], s, acc)
    if axis & 1:
        acc = lanes(acc)
    return finish_shape(store_float(acc, t), ohw, axis, C)


def sum_fp8(codes, axis, scale, h_tiles=1, w_tiles=1):
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
        T, ohw = steps(x, axis, ORDER_SUM[axis])
    s = f32(scale)
    acc = np.zeros(T.shape[1:], f32)
    for k in range(len(T)):
        acc = fma(T[k], s, acc)
    if axis & 1:
        acc = lanes(acc)
    return finish_shape(f32_to_fp8(acc, Rnd.kHalfEven, kSat), ohw, axis, C)


def sum_int(x, axis, kind, in_t=None, out_t=None, c0=0, c1=0, c2=0, f0=0.0, f1=0.0, shift=0):
    """kind 'a8' / 'a16' use c0, c1, c2; kind 'a32' uses f0 = c0, f1 = c1, shift."""
    x = np.asarray(x).astype(np.int64)
    axes = tuple(a for a, b in ((0, 4), (1, 2), (2, 1)) if axis & b)
    S = x.sum(axis=axes)
    if kind == "a32":
        return float_to_int(mac_exact(wrap(S, "i32"), f1, f0), out_t, -shift)
    if axis == 7:  # int64 wrap-around [default]
        with np.errstate(all="ignore"):
            v = np.int64(c0) * np.asarray(wrap(S, "i32") if kind == "a8" else S, np.int64) \
                + np.int64(c2)
    else:
        v = wrap(wrap(c2, "i32") + c0 * S, "i32")
    return saturate(shift_round(v, c1, Rnd.kHalfEven), out_t)


@quiet
def prod16(x, t, axis, order):
    """fp32 values x (exact 16-bit) -> fp32 values (16-bit)."""
    C = x.shape[2]
    T, ohw = steps(x, axis, order[axis])
    acc = np.ones(T.shape[1:], f32)
    for k in range(len(T)):
        acc = mul16_f32(acc, T[k], t)
    if axis & 1:
        v = acc
        while v.shape[-1] > 1:
            h = v.shape[-1] // 2
            v = mul16_f32(v[..., :h], v[..., h:], t)
        acc = v[..., 0]
    return finish_shape(acc, ohw, axis, C)


def l2_16(x, t, axis):
    """fp32 values x (exact 16-bit) -> fp32 values (16-bit)."""
    C = x.shape[2]
    T, ohw = steps(x, axis, ORDER_L2[axis])
    acc = np.zeros(T.shape[1:], f32)
    for k in range(len(T)):
        acc = fma(T[k], T[k], acc)
    if axis & 1:
        r = round_float(sqrt(lanes(acc)), t)
    else:
        r = load_float(hw_sqrt16(store_float(acc, t), t), t)
    return finish_shape(r, ohw, axis, C)


def prod_bits(bits, t, axis):
    return store_float(prod16(load_float(bits, t), t, axis, ORDER_PROD), t)


def l2_bits(bits, t, axis):
    return store_float(l2_16(load_float(bits, t), t, axis), t)


@quiet
def dq16(x, scale, zp, rt):
    """fma(x, scale, fp32(-zp * scale)), rounded to rt."""
    z = mul(fneg(f32(zp)), f32(scale))
    return round_float(mac_exact(x, scale, z), rt)


@quiet
def q_int(y, scale, zp, out_t):
    return float_to_int(fma(y, scale, zp), out_t)


def qdq_prod(x, axis, rt, dq_s, dq_z, q_s, q_z, out_t):
    return q_int(prod16(dq16(x, dq_s, dq_z, rt), rt, axis, ORDER_QDQ_PROD), q_s, q_z, out_t)


def qdq_l2(x, axis, rt, dq_s, dq_z, q_s, q_z, out_t):
    return q_int(l2_16(dq16(x, dq_s, dq_z, rt), rt, axis), q_s, q_z, out_t)


def mul_reducesum(x1, x2, axis, c, out_t, a16):
    c0, c1, c2, c3 = (f32(v) for v in c)
    x1 = np.asarray(x1).astype(np.int64)
    H, W, C = x1.shape
    x2 = np.broadcast_to(np.asarray(x2).astype(np.int64), x1.shape)
    T1, ohw = steps(x1, axis, ORDER_MRS[axis])
    T2, _ = steps(np.ascontiguousarray(x2), axis, ORDER_MRS[axis])
    acc = np.zeros(T1.shape[1:], f32)
    for i1, i2 in zip(T1, T2):
        if a16:
            p = (i1 * i2).astype(np.float64).astype(f32)
            r = mul(p, c0)
            r = mac_exact(i1, c1, r)
            r = mac_exact(i2, c2, r)
            acc = add(acc, r)
        else:
            t = mac_exact(i2, c0, c1)
            acc = mac_exact(i2, c2, acc)
            acc = mac_exact(i1, t, acc)
    v = lanes(acc) if axis & 1 else acc
    v = add(v, c3)
    return finish_shape(float_to_int(v, out_t), ohw, axis, C)


def cumsum(x, axis, exclusive, reverse, c0, c1, out_t):
    x = np.asarray(x).astype(np.int64)
    w = np.flip(x, axis) if reverse else x
    s = np.cumsum(w, axis)
    if exclusive:
        s = s - w
    if reverse:
        s = np.flip(s, axis)
    return float_to_int(mac_exact(wrap(s, "i32"), c1, c0), out_t)
