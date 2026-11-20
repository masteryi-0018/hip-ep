# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference model for gridsample.h."""

import numpy as np

from golden_float_cvt import f32_to_fp8, fp8_decode_identity, kSat, load_float, round_float
from golden_float_cvt import store_float
from golden_int_round import Rnd, float_to_int, saturate, wrap
from golden_mac import add, fma, mac_exact, mul, sub
from golden_grid import qdq_grid_coord

f32 = np.float32


def gather_index(fxc, fyc, H, W):
    """The four clamped neighbour indices: top-left saturated to int16, the
    offsets added in int16 (wrapping), clamped to [0, H*W - 1]."""
    i0 = saturate(fxc + fyc * W, "i16")
    out = []
    for off in (0, 1, W, W + 1):
        v = wrap(i0 + wrap(off, "i16"), "i16")
        out.append(np.clip(v, 0, H * W - 1))
    return out


def resize(v, idx, coeffs):
    """v [H*W][C] at T precision; fp32 mul + three fused MACs."""
    acc = None
    for nb, c in zip(idx, coeffs):
        x = v[nb]                                    # [Ho][Wo][C]
        cc = np.broadcast_to(c[..., None], x.shape)
        acc = mul(x, cc) if acc is None else fma(x, cc, acc)
    return acc


def coeffs_core(gx, gy, H, W, t):
    """(fxc, fyc, [c00, c01, c10, c11]) in T."""
    fxc = np.minimum(float_to_int(gx, "i16", 0, Rnd.kFloor), W - 1)
    fyc = np.minimum(float_to_int(gy, "i16", 0, Rnd.kFloor), H - 1)
    thr_w0 = round_float(round_float(f32(W), "bf16"), t)
    thr_h0 = round_float(round_float(f32(H), "bf16"), t)
    thr_w1 = round_float(thr_w0 - f32(1), t)
    thr_h1 = round_float(thr_h0 - f32(1), t)
    xo0 = (gx > thr_w0) | (gx < 0)
    xo1 = (gx > thr_w1) | (gx < -1)
    yo0 = (gy > thr_h0) | (gy < 0)
    yo1 = (gy > thr_h1) | (gy < -1)
    ffx, ffy = round_float(fxc.astype(f32), t), round_float(fyc.astype(f32), t)
    t0x = round_float(add(round_float(sub(f32(1), gx), t), ffx), t)
    t1x = round_float(sub(gx, ffx), t)
    t0y = round_float(add(round_float(sub(f32(1), gy), t), ffy), t)
    t1y = round_float(sub(gy, ffy), t)
    z = f32(0)
    cs = [round_float(mul(np.where(xo0 | yo0, z, t0x), t0y), t),
          round_float(mul(np.where(xo1 | yo0, z, t1x), t0y), t),
          round_float(mul(np.where(xo0 | yo1, z, t0x), t1y), t),
          round_float(mul(np.where(xo1 | yo1, z, t1x), t1y), t)]
    return fxc, fyc, cs


def _interp_core(v, grid, t):
    """v [H][W][C] (fp32 at T precision), grid [Ho][Wo][2] fp32 -> fp32 acc."""
    H, W, C = v.shape
    gx, gy = grid[..., 0].astype(f32), grid[..., 1].astype(f32)
    fxc, fyc, cs = coeffs_core(gx, gy, H, W, t)
    return resize(v.reshape(H * W, C), gather_index(fxc, fyc, H, W), cs)


def gridsample_bf16(x_bits, grid_bits, t):
    """x_bits [H][W][C], grid_bits [Ho][Wo][2] as uint16 of type t -> out bits."""
    acc = _interp_core(load_float(x_bits, t), load_float(grid_bits, t), t)
    return store_float(acc, t)


def fp8_ifm(x8, scale, t):
    """fp8 -> T per channel pair: decoded in groups of 8 (NaN spreads over the
    group, -0 -> +0), scaled in fp32, rounded to T."""
    H, W, C = x8.shape
    out = np.empty((H, W, C), f32)
    for c in range(0, C, 2):
        sl = np.ascontiguousarray(x8[..., c:c + 2]).reshape(-1)
        d = fp8_decode_identity(sl)
        out[..., c:c + 2] = round_float(mul(d, f32(scale)), t).reshape(H, W, 2)
    return out


def gridsample_fp8(x8, grid_bits, t, ifm_scale, ofm_recip):
    acc = _interp_core(fp8_ifm(x8, ifm_scale, t), load_float(grid_bits, t), t)
    return f32_to_fp8(mul(acc, f32(ofm_recip)), Rnd.kHalfEven, kSat)


def qdq_gridsample(x, grid, align_corners, q, t, out_t):
    """q = (grid_zp, grid_scale, ifm_zp, ifm_scale, ofm_zp, ofm_mult); t = compute
    type."""
    gzp, gs, izp, isc, ozp, omul = q
    H, W, C = x.shape
    off = f32(f32(-f32(izp)) * f32(isc))
    v = round_float(mac_exact(np.asarray(x, np.int64).reshape(-1), f32(isc), off), t)
    gx = qdq_grid_coord(grid[..., 0], gzp, gs, W, align_corners)
    gy = qdq_grid_coord(grid[..., 1], gzp, gs, H, align_corners)
    fx, fy = float_to_int(gx, "i16", 0, Rnd.kFloor), float_to_int(gy, "i16", 0, Rnd.kFloor)
    fxc, fyc = np.minimum(fx, W - 1), np.minimum(fy, H - 1)
    nx, ny = wrap(fx + 1, "i16"), wrap(fy + 1, "i16")
    ix0, ix1 = (fx >= 0) & (fx < W), (nx >= 0) & (nx < W)
    iy0, iy1 = (fy >= 0) & (fy < H), (ny >= 0) & (ny < H)
    ffx, ffy = fxc.astype(f32), fyc.astype(f32)
    t0x, t1x = add(sub(f32(1), gx), ffx), sub(gx, ffx)
    t0y, t1y = add(sub(f32(1), gy), ffy), sub(gy, ffy)
    z = f32(0)
    cs = [np.where(ix0 & iy0, round_float(mul(t0x, t0y), t), z),
          np.where(ix1 & iy0, round_float(mul(t1x, t0y), t), z),
          np.where(ix0 & iy1, round_float(mul(t0x, t1y), t), z),
          np.where(ix1 & iy1, round_float(mul(t1x, t1y), t), z)]
    acc = resize(v.reshape(H * W, C), gather_index(fxc, fyc, H, W), cs)
    return float_to_int(fma(acc, f32(omul), f32(ozp)), out_t)
