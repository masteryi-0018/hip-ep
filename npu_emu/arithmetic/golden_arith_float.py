# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the floating-point elementwise arithmetic (bf16 / f16 / fp8 / bfp16)."""

import numpy as np

from golden_block_fp import decode_bfp16_identity, encode_bfp16
from golden_float_cvt import (bf16_to_f32, decode_float, f32_to_bf16, f32_to_f16, f32_to_fp8,
                              fp8_decode_identity, fp8_to_f32,
                              load_float, round_float, scalar_cast, store_float)
from golden_int_round import Rnd
from golden_mac import add, fma, mul
from golden_mac import mul16 as mul16_codes
from golden_minmax import vclamp, vmax, vmin
from golden_mmac import mmac_add_bfp16
from golden_act import sigtanh_tanh_bf16
from golden_nlf import hw_inv16, hw_inv_f32

BF16, F16 = "bf16", "f16"
F32 = np.float32

NONE, RELU, CLIP, LRELU = 0, 1, 2, 3



def inv16(y, t):
    """16-bit reciprocal of fp32-carried values: bf16 reads the upper 16 bits, f16 rounds half
    to even (see arith_float.hip)."""
    y = np.ascontiguousarray(y, F32)
    c = f32_to_f16(y) if t == F16 else (y.view(np.uint32) >> 16).astype(np.uint16)
    return decode_float(hw_inv16(c, t), t)

def activate(v, act, a0, a1):
    if act == RELU:
        return vmax(v, a0)
    if act == CLIP:
        return vclamp(v, a0, a1)
    if act == LRELU:
        return vmax(v, mul(v, a0))
    return np.asarray(v, F32)


def bshape(bc, H, W, C):
    return (1 if bc & 4 else H, 1 if bc & 2 else W, 1 if bc & 1 else C)


def expand(x, bc, H, W, C):
    """An operand of shape bshape(bc) as a full H x W x C array."""
    return np.broadcast_to(np.asarray(x).reshape(bshape(bc, H, W, C)), (H, W, C))


def add16(x1, x2, bc, t, H, W, C, neg1, neg2, act, a0, a1):
    a = load_float(x1, t).reshape(H, W, C)
    b = expand(load_float(x2, t), bc, H, W, C)
    v = add(-a if neg1 else a, -b if neg2 else b)
    return store_float(activate(v, act, a0, a1), t).reshape(-1)


def mul16(x1, x2, bc, t, H, W, C, act, a0, a1):
    a = load_float(x1, t).reshape(H, W, C)
    b = expand(load_float(x2, t), bc, H, W, C)
    return store_float(activate(mul(a, b), act, a0, a1), t).reshape(-1)


def div16(x1, x2, bc, t, H, W, C, swap):
    a = load_float(x1, t).reshape(H, W, C)
    b = expand(load_float(x2, t), bc, H, W, C)
    num, den = (b, a) if swap else (a, b)
    return store_float(mul(num, inv16(den, t)), t).reshape(-1)


def mac16(x1, x2, x3, bc, t, H, W, C, act, a0, a1):
    a = load_float(x1, t).reshape(H, W, C)
    b = expand(load_float(x2, t), bc, H, W, C)
    c = expand(load_float(x3, t), bc, H, W, C)
    return store_float(activate(fma(a, b, c), act, a0, a1), t).reshape(-1)


def bfp_values(blocks, shape):
    """Identity-multiply decode; each pixel's channels form 64-element groups from channel 0."""
    return decode_bfp16_identity(blocks, shape[-1] // 8).reshape(shape)


def bfp_operand(x2, bc, H, W, C):
    """A broadcast bfp16 operand as H x W x C values. C-broadcast: element 0 of the pixel's
    block, decoded in the group of the 8 operand blocks starting at it."""
    s = bshape(bc, H, W, C)
    if bc & 1:
        blk = np.asarray(x2, np.uint8).reshape(-1, 9)
        n = len(blk)
        pad = np.concatenate([blk, np.zeros((7, 9), np.uint8)])  # exponent 0: no effect
        win = pad[np.arange(n)[:, None] + np.arange(8)]
        v = decode_bfp16_identity(win, 8).reshape(n, 64)[:, :1].reshape(s[0], s[1], 1)
    else:
        v = bfp_values(x2, s)
    return np.broadcast_to(v, (H, W, C))


def bfp_bin(op, x1, x2, bc, H, W, C, swap=0):
    a = bfp_values(x1, (H, W, C))
    b = bfp_operand(x2, bc, H, W, C)
    if op == "add":
        v = mmac_add_bfp16(x1, a, b, C).reshape(H, W, C)
    elif op == "sub" and not swap and bc & 1:
        v = mmac_add_bfp16(x1, a, -b, C).reshape(H, W, C)
    elif op == "sub":
        v = add(b, -a) if swap else add(a, -b)
    elif op == "mul":
        v = mul(a, b)
    else:
        num, den = (b, a) if swap else (a, b)
        v = mul(num, inv16(den, BF16))
    return encode_bfp16(v.reshape(-1), Rnd.kHalfEven)


def mul_add(x1, x2, x2_bfp, bc2, x3, x3_bfp, bc3, out_bfp, H, W, C, act_type, activation,
            lo, hi):
    a = bfp_values(x1, (H, W, C))

    def operand(x, bfp, bc):
        s = bshape(bc, H, W, C)
        v = bfp_values(x, s) if bfp else bf16_to_f32(x).reshape(s)
        return np.broadcast_to(v, (H, W, C))

    v = fma(a, operand(x2, x2_bfp, bc2), operand(x3, x3_bfp, bc3))
    if act_type == 100 and activation == 100:
        v = vclamp(round_float(v, BF16), lo, hi)
    elif act_type == 7:
        v = round_float(v, BF16)
        if activation == 7:
            v = sigtanh_tanh_bf16(v)
    v = v.reshape(-1)
    return encode_bfp16(v, Rnd.kHalfEven) if out_bfp else f32_to_bf16(v)


def add_fp8(x1, x2, bc, H, W, C, c1, c2, act, a0, a1, table=None):
    a = fp8_decode_identity(np.reshape(x1, (H, W, C)), row=C)
    if not bc & 1:
        s = bshape(bc, H, W, C)
        b = np.broadcast_to(fp8_decode_identity(np.reshape(x2, s), row=s[-1]), (H, W, C))
        v = activate(fma(b, c2, mul(a, c1)), act, a0, a1)
    else:
        codes = expand(np.asarray(x2, np.uint8), bc, H, W, C)
        b = fp8_to_f32(codes)
        if act == RELU:
            v = vmax(add(mul(a, c1), mul(b, c2)), a0)
        elif act == CLIP:
            v = vclamp(add(mul(a, c1), F32(table)[codes]), a0, a1)
        elif act == LRELU:
            tb = np.asarray(table, F32)
            q = fma(a, tb[256], tb[codes])
            v = fma(np.abs(q), tb[257], q)
        else:
            v = fma(a, c1, mul(b, c2))
    return f32_to_fp8(v.reshape(-1), Rnd.kHalfAway)


def mul_fp8(x1, x2, bc, H, W, C, s):
    if not bc & 1:
        a = fp8_to_f32(np.asarray(x1, np.uint8)).reshape(H, W, C)
        b = fp8_to_f32(expand(np.asarray(x2, np.uint8), bc, H, W, C))
        v = mul(mul(a, b), s)
    else:
        a = fp8_decode_identity(np.reshape(x1, (H, W, C)), row=C)
        b = expand(fp8_to_f32(x2) + F32(0.0), bc, H, W, C)
        v = mul(a, mul(b, s))
    return f32_to_fp8(v.reshape(-1), Rnd.kHalfAway)


def div_fp8(x1, x2, bc, H, W, C, s, swap):
    a = fp8_decode_identity(np.reshape(x1, (H, W, C)), row=C)
    s2 = bshape(bc, H, W, C)
    b = np.broadcast_to(fp8_decode_identity(np.reshape(x2, s2), row=s2[-1]), (H, W, C))
    num, den = (b, a) if swap else (a, b)
    return f32_to_fp8(mul(mul(num, s), hw_inv_f32(den)).reshape(-1), Rnd.kHalfAway)


def mac_fp8(x1, x2, x3, bc, H, W, C, c0, c4, act, a0, a1):
    a = fp8_decode_identity(np.reshape(x1, (H, W, C)), row=C)
    s = bshape(bc, H, W, C)
    if bc & 1:
        b = expand(fp8_to_f32(x2) + F32(0.0), bc, H, W, C)
        d = expand(fp8_to_f32(x3) + F32(0.0), bc, H, W, C)
        v = fma(a, mul(b, c0), mul(d, c4))
    else:
        b = np.broadcast_to(fp8_decode_identity(np.reshape(x2, s), row=s[-1]), (H, W, C))
        d = np.broadcast_to(fp8_decode_identity(np.reshape(x3, s), row=s[-1]), (H, W, C))
        v = fma(d, c4, mul(b, mul(a, c0)))
    return f32_to_fp8(activate(v, act, a0, a1).reshape(-1), Rnd.kHalfEven)


def leakyrelu16(x, t, alpha):
    v = load_float(x, t)
    return store_float(vmax(v, mul(v, scalar_cast(F32(alpha), t))), t)


def leakyrelu_fp8(x, pos, neg):
    v = fp8_decode_identity(x)
    return f32_to_fp8(vmax(mul(v, pos), mul(v, neg)), Rnd.kHalfAway)


def prelu16(x, t, D, C):
    v = load_float(x, t)
    d = np.resize(F32(D), v.size)
    return store_float(fma(vmin(v, F32(0.0)), d, v), t)


def prelu_fp8(x, P, D, C):
    v = fp8_decode_identity(x)
    d = np.resize(F32(D), v.size)
    return f32_to_fp8(fma(vmin(v, F32(0.0)), d, mul(v, P)), Rnd.kHalfAway)


def square16(x, t):
    return mul16_codes(x, x, t)


def square_fp8(x, s):
    v = fp8_decode_identity(x)
    return f32_to_fp8(mul(v, mul(v, s)), Rnd.kHalfAway)


def batchnorm(x, t, C, cw, cb, act, alpha):
    """t in BF16 / F16 / 'fp8'."""
    if t == "fp8":
        v = fp8_decode_identity(x)
    else:
        v = load_float(x, t)
    n = v.size
    acc = fma(v, np.resize(F32(cw), n), np.resize(F32(cb), n))
    if act == 3:
        acc = vmax(acc, mul(acc, alpha))
    elif act == 1 and t == "fp8":
        acc = vmax(acc, F32(0.0))
    if t == "fp8":
        return f32_to_fp8(acc, Rnd.kHalfEven)
    if act == 1:
        acc = vmax(round_float(acc, t), F32(0.0))
    return store_float(acc, t)
