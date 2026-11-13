# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the arith_float entry points of npu_emu (bf16 / f16 /
fp8 / bfp16 elementwise arithmetic wrappers), written from the AIE4 kernel
sources and mllib's corrected python models (fp32_exact / inv_lut / SigTanh
model), checked against aiesim outputs. Logical HWC layouts; operands in the
NPU's order (x1 full, x2 / x3 broadcast by bit 0 / 1 / 2 = C / W / H).

NPU rules modelled here (see include/npu_emu_arith_float.h):
  - fp32 arithmetic, one rounding per add / product / fused multiply-add.
  - fp32 VMAX / VMIN order -0 below +0; NaN -> canonical NaN.
  - fp8 identity-matmul decode: -0 -> +0, NaN spreads over 8 channels of a pixel.
  - div multiplies by the AIE4 reciprocal (bf16 table / f16 RNE / fp32 faithful table)."""

import numpy as np

import golden as g
import golden_arith_misc as gm

BF16, F16 = "bf16", "f16"
NONE, RELU, CLIP, LRELU = 0, 1, 2, 3
F32 = np.float32


# ---- scalars / vectors -------------------------------------------------------------

def f32(x):
    return np.asarray(x, F32)


def mul(a, b):
    with np.errstate(all="ignore"):
        return (f32(a) * f32(b)).astype(F32)


def add(a, b):
    with np.errstate(all="ignore"):
        return (f32(a) + f32(b)).astype(F32)


def fma(a, b, c):
    with np.errstate(all="ignore"):
        a, b, c = np.broadcast_arrays(f32(a), f32(b), f32(c))
        r = g.fma32(a, b, c)
        # TwoSum is not meaningful for non-finite terms: IEEE result instead.
        bad = ~(np.isfinite(a) & np.isfinite(b) & np.isfinite(c)) | ~np.isfinite(r)
        ieee = (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(F32)
        r = np.where(bad, ieee, r)
        # An exact zero sum of non-zero terms is +0 (RNE); signed zeros follow IEEE.
        p = a.astype(np.float64) * b.astype(np.float64)
        zero = (r == 0) & (p + c.astype(np.float64) == 0)
        both_neg = np.signbit(p) & np.signbit(c) & (p == 0) & (c == 0)
        r = np.where(zero, np.where(both_neg, F32(-0.0), F32(0.0)), r)
        return r.astype(F32)


def _qnan(shape):
    return np.full(shape, np.nan, F32)


def vmax(a, b):
    a, b = np.broadcast_arrays(f32(a), f32(b))
    zz = (a == 0) & (b == 0)
    r = np.where(zz, np.where(np.signbit(a), b, a), np.where(a > b, a, b))
    return np.where(np.isnan(a) | np.isnan(b), _qnan(a.shape), r).astype(F32)


def vmin(a, b):
    a, b = np.broadcast_arrays(f32(a), f32(b))
    zz = (a == 0) & (b == 0)
    r = np.where(zz, np.where(np.signbit(a), a, b), np.where(a < b, a, b))
    return np.where(np.isnan(a) | np.isnan(b), _qnan(a.shape), r).astype(F32)


def activate(v, act, a0, a1):
    if act == RELU:
        return vmax(v, a0)
    if act == CLIP:
        return vmin(vmax(v, a0), a1)
    if act == LRELU:
        return vmax(v, mul(v, a0))
    return f32(v)


# ---- 16-bit floats ---------------------------------------------------------------------

def load16(bits, t):
    return g.f16_bits_to_f32(bits) if t == F16 else g.bf16_bits_to_f32(bits)


def round16(x, t):
    return g.to_f16(x) if t == F16 else g.to_bf16(x)


def rne16(x, t):
    return load16(round16(x, t), t)


def inv_bf16(y):
    """inv_lut.inv_bf16_aie4 as a rule (equal to the 65536-entry table)."""
    y = f32(y)
    b = y.view(np.uint32)
    e = (b >> 23) & 0xFF
    with np.errstate(all="ignore"):
        r = (F32(1.0) / y).astype(F32)
    out = g.bf16_bits_to_f32(g.to_bf16(r))
    out = np.where(np.abs(r) < F32(2.0 ** -126), np.copysign(F32(0.0), y), out)
    out = np.where(e == 0, np.copysign(F32(np.inf), y), out)
    quiet = (b | np.uint32(0x400000)).view(F32)
    return np.where((e == 0xFF) & ((b & 0x7FFFFF) != 0), quiet, out).astype(F32)


def inv16(y, t):
    if t == F16:
        with np.errstate(all="ignore"):
            return rne16((F32(1.0) / f32(y)).astype(F32), t)
    return inv_bf16(y)


# ---- broadcasting helpers -------------------------------------------------------------------

def bshape(bc, H, W, C):
    return (1 if bc & 4 else H, 1 if bc & 2 else W, 1 if bc & 1 else C)


def expand(x, bc, H, W, C):
    """An operand of shape bshape(bc) as a full H x W x C array."""
    return np.broadcast_to(np.asarray(x).reshape(bshape(bc, H, W, C)), (H, W, C))


def fp8_rows(codes, shape):
    """Identity-matmul decode of an H x W x C fp8 tensor: per pixel, rows of 8
    channels (NaN spreads), -0 -> +0."""
    C = shape[-1]
    c = np.asarray(codes, np.uint8).reshape(-1, C)
    pad = (-C) % 8
    v = g.fp8_to_f32(np.pad(c, ((0, 0), (0, pad)))).reshape(c.shape[0], -1, 8)
    v = np.where(np.isnan(v).any(axis=2, keepdims=True), F32(np.nan), v)
    return (v.reshape(c.shape[0], -1)[:, :C].reshape(shape) + F32(0.0)).astype(F32)


def fp8_plain(codes):
    return g.fp8_to_f32(codes)


# ---- bf16 / f16 broadcasting --------------------------------------------------------------

def add16(x1, x2, bc, t, H, W, C, neg1, neg2, act, a0, a1):
    a = load16(x1, t).reshape(H, W, C)
    b = expand(load16(x2, t), bc, H, W, C)
    v = add(-a if neg1 else a, -b if neg2 else b)
    return round16(activate(v, act, a0, a1), t).reshape(-1)


def mul16(x1, x2, bc, t, H, W, C, act, a0, a1):
    a = load16(x1, t).reshape(H, W, C)
    b = expand(load16(x2, t), bc, H, W, C)
    return round16(activate(mul(a, b), act, a0, a1), t).reshape(-1)


def div16(x1, x2, bc, t, H, W, C, swap):
    a = load16(x1, t).reshape(H, W, C)
    b = expand(load16(x2, t), bc, H, W, C)
    num, den = (b, a) if swap else (a, b)
    return round16(mul(num, inv16(den, t)), t).reshape(-1)


def mac16(x1, x2, x3, bc, t, H, W, C, act, a0, a1):
    a = load16(x1, t).reshape(H, W, C)
    b = expand(load16(x2, t), bc, H, W, C)
    c = expand(load16(x3, t), bc, H, W, C)
    return round16(activate(fma(a, b, c), act, a0, a1), t).reshape(-1)


# ---- bfp16 ---------------------------------------------------------------------------------------

def bfp_values(blocks, shape):
    return g.from_bfp16(blocks).reshape(shape)


def bfp_operand(x2, bc, H, W, C):
    """A broadcast bfp16 operand as H x W x C values; C-broadcast = element 0
    of the pixel's block."""
    s = bshape(bc, H, W, C)
    if bc & 1:
        v = g.from_bfp16(x2).reshape(s[0], s[1], 8)[:, :, :1]
    else:
        v = g.from_bfp16(x2).reshape(s)
    return np.broadcast_to(v, (H, W, C))


def bfp_bin(op, x1, x2, bc, H, W, C, swap=0):
    a = bfp_values(x1, (H, W, C))
    b = bfp_operand(x2, bc, H, W, C)
    if op == "add":
        v = add(a, b)
    elif op == "sub":
        v = add(b, -a) if swap else add(a, -b)
    elif op == "mul":
        v = mul(a, b)
    else:
        num, den = (b, a) if swap else (a, b)
        v = mul(num, inv_bf16(den))
    return g.to_bfp16(v.reshape(-1), g.EVEN)


# ---- mul_add bfp16 ----------------------------------------------------------------------------

TANH_K = (F32(894.00097656), F32(1.53878212), F32(-0.49819374))
_EXP2_CNST = np.array((0, 39, 74, 110, 141, 172, 200, 225, 250, 270, 290, 305, 320, 332, 340,
                       347, 352, 352, 350, 347, 340, 328, 315, 299, 280, 257, 229, 200, 167,
                       131, 91, 47), np.int64)
_EXP2_COEFF = np.array((-10, -9, -9, -8, -8, -7, -6, -6, -5, -5, -4, -4, -3, -2, -2, -1, 0, 0,
                        1, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12), np.int64)


def exp2_bf20(p):
    """mllib NlfModel.approx_exp2(bits_out=19, add_magic_const=False), then the
    srs denormal flush."""
    p = f32(p)
    x = ((p + F32(0.0)).view(np.int32).astype(np.int64) >> 2) - (0x111 << 20)
    sel = (x >> 7) & 31
    approx = (x & ~1) + ((((x >> 2) & 31) * _EXP2_COEFF[sel]) >> 3) - _EXP2_CNST[sel]
    dn = (x < (1 << 12)) & (x >= -(11 << 12))
    sh = np.where(dn, 1 - (approx >> 12), 0)
    data = 4096 + (approx & 4095)
    dn_val = (data >> sh) | ((data & ((np.int64(1) << sh) - 1)) != 0)
    approx = np.where(dn, dn_val, approx)
    bits = (np.clip(approx, 0, 255 << 12) << 11) & 0x7FFFFFFF
    out = bits.astype(np.uint32).view(F32)
    out = np.where((bits >> 23) == 0, F32(0.0), out)
    return np.where(np.isnan(p), F32(np.nan), out).astype(F32)


def sigtanh(x):
    """SOL SigTanh<TANH> on bf16 values; bf16 output, half away, subnormal -> 0."""
    x = f32(x)
    k0, k1, k2 = TANH_K
    p = fma(mul(x, x), k2, fma(-np.abs(x), k1, k0))
    t = fma(F32(-2.0), exp2_bf20(p), F32(1.0))
    t = np.where(np.signbit(x), -t, t).astype(F32)
    r = g.to_bf16(t, g.AWAY)
    flush = ((r >> 7) & 0xFF == 0) | ((t.view(np.uint32) >> 23) & 0xFF == 0)
    v = g.bf16_bits_to_f32(r)
    return np.where(flush, np.copysign(F32(0.0), v), v).astype(F32)


def to_bf16_vals(x):
    return g.bf16_bits_to_f32(g.to_bf16(x))


def mul_add(x1, x2, x2_bfp, bc2, x3, x3_bfp, bc3, out_bfp, H, W, C, act_type, activation,
            lo, hi):
    a = bfp_values(x1, (H, W, C))

    def operand(x, bfp, bc):
        s = bshape(bc, H, W, C)
        v = g.from_bfp16(x).reshape(s) if bfp else g.bf16_bits_to_f32(x).reshape(s)
        return np.broadcast_to(v, (H, W, C))

    v = fma(a, operand(x2, x2_bfp, bc2), operand(x3, x3_bfp, bc3))
    if act_type == 100 and activation == 100:
        v = vmin(vmax(to_bf16_vals(v), lo), hi)
    elif act_type == 7:
        v = to_bf16_vals(v)
        if activation == 7:
            v = sigtanh(v)
    v = v.reshape(-1)
    return g.to_bfp16(v, g.EVEN) if out_bfp else g.to_bf16(v)


# ---- fp8 broadcasting ------------------------------------------------------------------------

def add_fp8(x1, x2, bc, H, W, C, c1, c2, act, a0, a1, table=None):
    a = fp8_rows(x1, (H, W, C))
    if not bc & 1:
        s = bshape(bc, H, W, C)
        b = np.broadcast_to(fp8_rows(x2, s), (H, W, C))
        v = activate(fma(b, c2, mul(a, c1)), act, a0, a1)
    else:
        codes = expand(np.asarray(x2, np.uint8), bc, H, W, C)
        b = fp8_plain(codes)
        if act == RELU:
            v = vmax(add(mul(a, c1), mul(b, c2)), a0)
        elif act == CLIP:
            v = vmin(vmax(add(mul(a, c1), f32(table)[codes]), a0), a1)
        elif act == LRELU:
            tb = f32(table)
            q = fma(a, tb[256], tb[codes])
            v = fma(np.abs(q), tb[257], q)
        else:
            v = fma(a, c1, mul(b, c2))
    return g.to_fp8(v.reshape(-1), g.AWAY)


def mul_fp8(x1, x2, bc, H, W, C, s):
    if not bc & 1:
        a = fp8_plain(np.asarray(x1, np.uint8)).reshape(H, W, C)
        b = fp8_plain(expand(np.asarray(x2, np.uint8), bc, H, W, C))
        v = mul(mul(a, b), s)
    else:
        a = fp8_rows(x1, (H, W, C))
        b = expand(fp8_plain(x2) + F32(0.0), bc, H, W, C)
        v = mul(a, mul(b, s))
    return g.to_fp8(v.reshape(-1), g.AWAY)


def div_fp8(x1, x2, bc, H, W, C, s, swap):
    a = fp8_rows(x1, (H, W, C))
    b = np.broadcast_to(fp8_rows(x2, bshape(bc, H, W, C)), (H, W, C))
    num, den = (b, a) if swap else (a, b)
    with np.errstate(all="ignore"):
        r = np.array(F32(1.0) / den, F32)
    # aie::inv is the faithfully rounded fp32 reciprocal; 0 / Inf / NaN as IEEE.
    ok = np.isfinite(den) & (den != 0)
    r[ok] = gm.inv_f32(den[ok], gm.inv_f32_dev())
    return g.to_fp8(mul(mul(num, s), r).reshape(-1), g.AWAY)


def mac_fp8(x1, x2, x3, bc, H, W, C, c0, c4, act, a0, a1):
    a = fp8_rows(x1, (H, W, C))
    s = bshape(bc, H, W, C)
    if bc & 1:
        b = expand(fp8_plain(x2) + F32(0.0), bc, H, W, C)
        d = expand(fp8_plain(x3) + F32(0.0), bc, H, W, C)
        v = fma(a, mul(b, c0), mul(d, c4))
    else:
        b = np.broadcast_to(fp8_rows(x2, s), (H, W, C))
        d = np.broadcast_to(fp8_rows(x3, s), (H, W, C))
        v = fma(d, c4, mul(b, mul(a, c0)))
    return g.to_fp8(activate(v, act, a0, a1).reshape(-1), g.EVEN)


# ---- unary / per channel --------------------------------------------------------------------

def leakyrelu16(x, t, alpha):
    v = load16(x, t)
    return round16(vmax(v, mul(v, rne16(F32(alpha), t))), t)


def leakyrelu_fp8(x, pos, neg):
    v = g.fp8_to_f32_row(x) + F32(0.0)
    return g.to_fp8(vmax(mul(v, pos), mul(v, neg)), g.AWAY)


def prelu16(x, t, D, C):
    v = load16(x, t)
    d = np.resize(f32(D), v.size)
    return round16(fma(vmin(v, F32(0.0)), d, v), t)


def prelu_fp8(x, P, D, C):
    v = g.fp8_to_f32_row(x) + F32(0.0)
    d = np.resize(f32(D), v.size)
    return g.to_fp8(fma(vmin(v, F32(0.0)), d, mul(v, P)), g.AWAY)


def square16(x, t):
    v = load16(x, t)
    sq = mul(v, v)
    if t == F16:
        sq = np.where(np.abs(sq) < F32(2.0 ** -14), F32(0.0), sq)
    return round16(sq, t)


def square_fp8(x, s):
    v = g.fp8_to_f32_row(x) + F32(0.0)
    return g.to_fp8(mul(v, mul(v, s)), g.AWAY)


def batchnorm(x, t, C, cw, cb, act, alpha):
    """t in BF16 / F16 / 'fp8'."""
    if t == "fp8":
        v = g.fp8_to_f32_row(x) + F32(0.0)
    else:
        v = load16(x, t)
    n = v.size
    acc = fma(v, np.resize(f32(cw), n), np.resize(f32(cb), n))
    if act == 3:
        acc = vmax(acc, mul(acc, alpha))
    elif act == 1 and t == "fp8":
        acc = vmax(acc, F32(0.0))
    if t == "fp8":
        return g.to_fp8(acc, g.EVEN)
    if act == 1:
        acc = vmax(rne16(acc, t), F32(0.0))
    return round16(acc, t)
