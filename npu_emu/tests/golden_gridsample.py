# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the gridsample group of npu_emu
(include/npu_emu_gridsample.h): mllib's corrected bit-exact models of the
AIE4 bilinear GridSample wrappers (GridSampleBf16Wrapper.py,
GridSampleFp8Wrapper.py, QDQ_GridSampleA8/A16Wrapper.py), vectorized over
the whole image and written independently of the HIP code. Tensors are
logical: in [H][W][C], grid [Ho][Wo][2], out [Ho][Wo][C]."""

import numpy as np

import golden as g

f32 = np.float32


def to_t(x, t):
    """fp32 -> T ('bf16' / 'f16'), half to even, returned as fp32."""
    x = np.asarray(x, f32)
    if t == "f16":
        return x.astype(np.float16).astype(f32)
    return g.bf16_bits_to_f32(g.to_bf16(x))


def floor16(v):
    """to_fixed_floor<int16>: floor saturated to int16 (NaN -> 0)."""
    v = np.asarray(v, np.float64)
    return np.clip(np.floor(np.where(np.isnan(v), 0.0, v)), -32768, 32767).astype(np.int64)


def gather_index(fxc, fyc, H, W):
    """The four clamped neighbour indices: top-left saturated to int16, the
    offsets added in int16 (wrapping), clamped to [0, H*W - 1]."""
    i0 = np.clip(fxc + fyc * W, -32768, 32767).astype(np.int16)
    out = []
    for off in (0, 1, W, W + 1):
        off16 = np.array(off, np.int64).astype(np.int16)    # broadcast_s16(W + 1) wraps
        v = (i0.astype(np.int32) + np.int32(off16)).astype(np.int16)
        out.append(np.clip(v.astype(np.int64), 0, H * W - 1))
    return out


def resize(v, idx, coeffs):
    """v [H*W][C] at T precision; fp32 mul + three fused MACs."""
    acc = None
    for nb, c in zip(idx, coeffs):
        x = v[nb]                                    # [Ho][Wo][C]
        cc = np.broadcast_to(c[..., None], x.shape)
        acc = (x * cc).astype(f32) if acc is None else g.fma32(x, cc, acc)
    return acc


def coeffs_sol(gx, gy, H, W, t):
    """grid_sample_index_bf16: (fxc, fyc, [c00, c01, c10, c11]) in T."""
    fxc = np.minimum(floor16(gx), W - 1)
    fyc = np.minimum(floor16(gy), H - 1)
    thr_w0 = to_t(g.bf16_bits_to_f32(g.to_bf16(f32(W))), t)
    thr_h0 = to_t(g.bf16_bits_to_f32(g.to_bf16(f32(H))), t)
    thr_w1 = to_t(thr_w0 - f32(1), t)
    thr_h1 = to_t(thr_h0 - f32(1), t)
    xo0 = (gx > thr_w0) | (gx < 0)
    xo1 = (gx > thr_w1) | (gx < -1)
    yo0 = (gy > thr_h0) | (gy < 0)
    yo1 = (gy > thr_h1) | (gy < -1)
    ffx, ffy = to_t(fxc.astype(f32), t), to_t(fyc.astype(f32), t)
    t0x = to_t(to_t(f32(1) - gx, t) + ffx, t)
    t1x = to_t(gx - ffx, t)
    t0y = to_t(to_t(f32(1) - gy, t) + ffy, t)
    t1y = to_t(gy - ffy, t)
    z = f32(0)
    cs = [to_t(np.where(xo0 | yo0, z, t0x) * t0y, t),
          to_t(np.where(xo1 | yo0, z, t1x) * t0y, t),
          to_t(np.where(xo0 | yo1, z, t0x) * t1y, t),
          to_t(np.where(xo1 | yo1, z, t1x) * t1y, t)]
    return fxc, fyc, cs


def _interp_sol(v, grid, t):
    """v [H][W][C] (fp32 at T precision), grid [Ho][Wo][2] fp32 -> fp32 acc."""
    H, W, C = v.shape
    gx, gy = grid[..., 0].astype(f32), grid[..., 1].astype(f32)
    fxc, fyc, cs = coeffs_sol(gx, gy, H, W, t)
    return resize(v.reshape(H * W, C), gather_index(fxc, fyc, H, W), cs)


def gridsample_bf16(x_bits, grid_bits, t):
    """x_bits [H][W][C], grid_bits [Ho][Wo][2] as uint16 of type t -> out bits."""
    acc = _interp_sol(g.load_float(x_bits, t), g.load_float(grid_bits, t), t)
    return g.store_float(acc, t)


def fp8_ifm(x8, scale, t):
    """convert_fp8_fp16 on each channel pair's [H][W][2] slice: groups of 8
    from the slice start (NaN spreads over the group, -0 -> +0), times the
    scale in fp32, rounded to T."""
    H, W, C = x8.shape
    out = np.empty((H, W, C), f32)
    for c in range(0, C, 2):
        sl = np.ascontiguousarray(x8[..., c:c + 2]).reshape(-1)
        d = g.fp8_to_f32_row(sl) + f32(0)
        out[..., c:c + 2] = to_t((d * f32(scale)).astype(f32), t).reshape(H, W, 2)
    return out


def gridsample_fp8(x8, grid_bits, t, ifm_scale, ofm_recip):
    acc = _interp_sol(fp8_ifm(x8, ifm_scale, t), g.load_float(grid_bits, t), t)
    v = (acc * f32(ofm_recip)).astype(f32)
    v = np.where(np.isnan(v), v, np.clip(v, f32(-448), f32(448))).astype(f32)
    return g.to_fp8(v, g.EVEN)


def qdq_coord(code, zp, scale, extent, align_corners):
    v = ((np.asarray(code, f32) - f32(zp)).astype(f32) * f32(scale)).astype(f32)
    v = (v + f32(1)).astype(f32)
    if align_corners == 0:
        return g.fma32(v, np.full(v.shape, f32(extent) * f32(0.5)), np.full(v.shape, f32(-0.5)))
    return ((f32(extent - 1) * v).astype(f32) * f32(0.5)).astype(f32)


def qdq_gridsample(x, grid, align_corners, q, t, out_t):
    """x [H][W][C] integers, grid [Ho][Wo][2] uint16 codes; q = (grid_zp,
    grid_scale, ifm_zp, ifm_scale, ofm_zp, ofm_mult); t = compute type;
    out_t = 'i8' / 'u8' / 'i16' / 'u16'."""
    gzp, gs, izp, isc, ozp, omul = q
    H, W, C = x.shape
    off = f32(f32(-f32(izp)) * f32(isc))
    xi = np.asarray(x, np.int64).reshape(-1)
    v = to_t(g.mac_int(xi, np.full(xi.shape, f32(isc)), np.full(xi.shape, off)), t)
    gx = qdq_coord(grid[..., 0], gzp, gs, W, align_corners)
    gy = qdq_coord(grid[..., 1], gzp, gs, H, align_corners)
    fx, fy = floor16(gx), floor16(gy)
    fxc, fyc = np.minimum(fx, W - 1), np.minimum(fy, H - 1)
    nx = (fx + 1).astype(np.int16).astype(np.int64)
    ny = (fy + 1).astype(np.int16).astype(np.int64)
    ix0, ix1 = (fx >= 0) & (fx < W), (nx >= 0) & (nx < W)
    iy0, iy1 = (fy >= 0) & (fy < H), (ny >= 0) & (ny < H)
    ffx, ffy = fxc.astype(f32), fyc.astype(f32)
    t0x = ((f32(1) - gx).astype(f32) + ffx).astype(f32)
    t1x = (gx - ffx).astype(f32)
    t0y = ((f32(1) - gy).astype(f32) + ffy).astype(f32)
    t1y = (gy - ffy).astype(f32)
    z = f32(0)
    cs = [np.where(ix0 & iy0, to_t((t0x * t0y).astype(f32), t), z),
          np.where(ix1 & iy0, to_t((t1x * t0y).astype(f32), t), z),
          np.where(ix0 & iy1, to_t((t0x * t1y).astype(f32), t), z),
          np.where(ix1 & iy1, to_t((t1x * t1y).astype(f32), t), z)]
    acc = resize(v.reshape(H * W, C), gather_index(fxc, fyc, H, W), cs)
    qv = g.fma32(acc, np.full(acc.shape, f32(omul)), np.full(acc.shape, f32(ozp)))
    lo, hi = g.INT_RANGE[out_t]
    return g.rne_int(qv, lo, hi)
