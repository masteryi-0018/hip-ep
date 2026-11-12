# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the qdq group of npu_emu (include/npu_emu_qdq.h): the
mllib kernels' integer selection plus the exact requantization (one-rounding
fp32 MAC with the integer taken exactly, then round / saturate), written
independently of the HIP code. Tensors are logical HWC arrays."""

import numpy as np

import golden as g

INT_RANGE = g.INT_RANGE
f32 = np.float32


# ---- requantization ----------------------------------------------------------

def to_int(v, t, shift=0, mode="even"):
    """fp32 -> integer type t: v * 2^shift (exact in float64), rounded half to
    even (to_vector_sign under conv_even) or toward zero (symmetric_floor),
    saturated. NaN -> 0."""
    v = np.asarray(v, np.float64) * (2.0 ** shift)
    v = np.where(np.isnan(v), 0.0, v)
    r = np.rint(v) if mode == "even" else np.trunc(v)
    lo, hi = INT_RANGE[t]
    return np.clip(r, lo, hi).astype(np.int64)


def mac(x, scale, offset):
    """x * scale + offset rounded once to fp32, integer x exact (any shape)."""
    x = np.asarray(x, np.int64)
    return g.mac_int(x.reshape(-1), scale, offset).reshape(x.shape)


def fmac(x, scale, offset):
    """fp32 x * scale + offset, one rounding."""
    x = np.asarray(x, np.float32)
    return g.fma32(x.reshape(-1), scale, offset).reshape(x.shape)


def requant(x, scale, offset, t, shift=0, mode="even"):
    return to_int(mac(x, scale, offset), t, shift, mode)


# ---- A / B: elementwise ----------------------------------------------------------

def abs_(x, c1, c0, zp, t):
    p = mac(x, c1, f32(f32(c0) + f32(zp)))
    q = mac(x, -f32(c1), f32(f32(zp) - f32(c0)))
    return to_int(np.maximum(p, q), t)


def clip(x, c1, c0, lo, hi, t):
    v = mac(x, c1, c0)
    return to_int(np.minimum(np.maximum(v, f32(lo)), f32(hi)), t)


def compare(a, b, sa, sb, za, zb, op):
    fa, fb = mac(a, sa, za), mac(b, sb, zb)
    return (fa == fb if op == "equal" else fa > fb).astype(np.int8)


def minmax(a, b, sa, sb, so, za, zb, zo, t, op):
    """b broadcasts to a's shape (numpy rules)."""
    fa = mac(a, sa, za)
    fb = np.broadcast_to(mac(b, sb, zb), fa.shape)
    m = np.maximum(fa, fb) if op == "max" else np.minimum(fa, fb)
    return to_int(fmac(m, so, zo), t)


# ---- C: where ----------------------------------------------------------------------

def where(cond, x, y, scalar_position):
    c = np.broadcast_to(cond, x.shape) != 0
    if not scalar_position:
        c = ~c
    return np.where(c, x, np.asarray(y).astype(x.dtype))


# ---- D / E: reductions and pooling -------------------------------------------------

AXES = {1: (2,), 2: (1,), 3: (1, 2), 4: (0,), 5: (0, 2), 6: (0, 1), 7: (0, 1, 2)}


def reduce(x, axis_flags, op):
    x = np.asarray(x, np.int64)
    return (x.max if op == "max" else x.min)(axis=AXES[axis_flags])


def maxpool(x, kh, kw, sh, sw):
    x = np.asarray(x, np.int64)
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(0, 1))
    return win[::sh, ::sw].max(axis=(-2, -1))


# ---- F / G -------------------------------------------------------------------------

def concat(ins):
    return np.concatenate([np.asarray(a, np.int64) for a in ins], axis=-1)


def gather_axis0(x, idx, scale, offset, t, untouched):
    """Rows outside [0, M_in) keep `untouched` (the kernel does not write them)."""
    idx = np.asarray(idx, np.int64)
    ok = (idx >= 0) & (idx < x.shape[0])
    cols = np.broadcast_to(np.arange(idx.shape[1]), idx.shape)
    v = np.asarray(x, np.int64)[np.where(ok, idx, 0), cols]
    return np.where(ok, requant(v, scale, offset, t), untouched)


def grid_coord(code, zp, scale, extent, align_corners, padding_mode):
    f = (np.asarray(code, np.float32) - f32(zp)) * f32(scale)   # numpy fp32: one rounding each
    f = (f + f32(1.0)).astype(np.float32)
    if align_corners == 0:
        gc = fmac(f, f32(extent) * f32(0.5), f32(-0.5))
    else:
        gc = ((f32(extent - 1) * f).astype(np.float32) * f32(0.5)).astype(np.float32)
    r = to_int(gc, "i16")
    if padding_mode == 1:
        r = np.clip(r, 0, extent - 1)
    return r


def gridsample_nearest(x, grid, align_corners, padding_mode, q, t):
    """x [H][W][C] integers, grid [Ho][Wo][2] codes; q = (grid_zp, grid_scale,
    ifm_zp, ifm_scale, ofm_zp, ofm_scale)."""
    gzp, gs, izp, isc, ozp, osc = q
    H, W, _ = x.shape
    rx = grid_coord(grid[..., 0], gzp, gs, W, align_corners, padding_mode)
    ry = grid_coord(grid[..., 1], gzp, gs, H, align_corners, padding_mode)
    ok = (rx >= 0) & (rx < W) & (ry >= 0) & (ry < H)
    pix = np.asarray(x, np.int64)[np.where(ok, ry, 0), np.where(ok, rx, 0)]   # [Ho][Wo][C]
    off = f32(f32(-f32(izp)) * f32(isc))
    dq = g.bf16_bits_to_f32(g.to_bf16(mac(pix, isc, off)))
    v = np.where(ok[..., None], dq, f32(0.0)).astype(np.float32)
    return to_int(fmac(v, osc, f32(ozp)), t)


def resize_nearest(x, sy, sx, requantize, scale, offset, t):
    v = np.asarray(x, np.int64)[np.asarray(sy)[:, None], np.asarray(sx)[None, :]]
    return requant(v, scale, offset, t, 0, "trunc") if requantize else v
