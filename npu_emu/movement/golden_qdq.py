# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for qdq.h."""

import numpy as np

from golden_float_cvt import round_float
from golden_int_round import Rnd, float_to_int
from golden_mac import fma, mac_exact
from golden_minmax import vclamp, vmax, vmin
from golden_grid import qdq_grid_coord

f32 = np.float32


def requant(x, scale, offset, t, shift=0, rnd=Rnd.kHalfEven):
    """mac, then * 2^shift rounded with rnd, saturated to t; NaN -> 0."""
    return float_to_int(mac_exact(x, scale, offset), t, shift, rnd)


def abs_(x, c1, c0, zp, t):
    with np.errstate(all="ignore"):
        return _abs(x, c1, c0, zp, t)


def _abs(x, c1, c0, zp, t):
    p = mac_exact(x, c1, f32(f32(c0) + f32(zp)))
    q = mac_exact(x, -f32(c1), f32(f32(zp) - f32(c0)))
    return float_to_int(vmax(p, q), t)


def clip(x, c1, c0, lo, hi, t):
    return float_to_int(vclamp(mac_exact(x, c1, c0), f32(lo), f32(hi)), t)


def compare(a, b, sa, sb, za, zb, op):
    fa, fb = mac_exact(a, sa, za), mac_exact(b, sb, zb)
    return (fa == fb if op == "equal" else fa > fb).astype(np.int8)


def minmax(a, b, sa, sb, so, za, zb, zo, t, op):
    """b broadcasts to a's shape (numpy rules)."""
    fa = mac_exact(a, sa, za)
    fb = np.broadcast_to(mac_exact(b, sb, zb), fa.shape)
    m = vmax(fa, fb) if op == "max" else vmin(fa, fb)
    return float_to_int(fma(m, so, zo), t)


def where(cond, x, y, scalar_position):
    c = np.broadcast_to(cond, x.shape) != 0
    if not scalar_position:
        c = ~c
    return np.where(c, x, np.asarray(y).astype(x.dtype))


AXES = {1: (2,), 2: (1,), 3: (1, 2), 4: (0,), 5: (0, 2), 6: (0, 1), 7: (0, 1, 2)}


def reduce(x, axis_flags, op):
    x = np.asarray(x, np.int64)
    return (x.max if op == "max" else x.min)(axis=AXES[axis_flags])


def maxpool(x, kh, kw, sh, sw):
    x = np.asarray(x, np.int64)
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(0, 1))
    return win[::sh, ::sw].max(axis=(-2, -1))


def concat(ins):
    return np.concatenate([np.asarray(a, np.int64) for a in ins], axis=-1)


def gather_axis0(x, idx, scale, offset, t, untouched):
    """Rows outside [0, M_in) keep `untouched`."""
    idx = np.asarray(idx, np.int64)
    ok = (idx >= 0) & (idx < x.shape[0])
    cols = np.broadcast_to(np.arange(idx.shape[1]), idx.shape)
    v = np.asarray(x, np.int64)[np.where(ok, idx, 0), cols]
    return np.where(ok, requant(v, scale, offset, t), untouched)


def grid_coord(code, zp, scale, extent, align_corners, padding_mode):
    r = float_to_int(qdq_grid_coord(code, zp, scale, extent, align_corners), "i16")
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
    with np.errstate(all="ignore"):
        off = f32(f32(-f32(izp)) * f32(isc))
    dq = round_float(mac_exact(pix, isc, off), "bf16")
    v = np.where(ok[..., None], dq, f32(0.0)).astype(np.float32)
    return float_to_int(fma(v, osc, f32(ozp)), t)


def resize_nearest(x, sy, sx, requantize, scale, offset, t, untouched=0):
    """A source index outside the input leaves the output element `untouched` [default]."""
    x = np.asarray(x, np.int64)
    sy, sx = np.asarray(sy, np.int64), np.asarray(sx, np.int64)
    oky, okx = (sy >= 0) & (sy < x.shape[0]), (sx >= 0) & (sx < x.shape[1])
    v = x[np.where(oky, sy, 0)[:, None], np.where(okx, sx, 0)[None, :]]
    v = requant(v, scale, offset, t, 0, Rnd.kTrunc) if requantize else v
    return np.where((oky[:, None] & okx[None, :])[..., None], v, untouched)
