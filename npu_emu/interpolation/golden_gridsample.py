# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference model for gridsample.h."""

import numpy as np

from golden_float_cvt import float_to_int, load_float, round_float, store_float
from golden_grid import qdq_grid_coord
from golden_int_round import Rnd, clamp64, saturate, srs, wrap
from golden_mac import add_q, fma_q, mul_q, sub_q
from golden_quant import dequant, dq_offset, fp8_in, fp8_out, quant

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
        acc = mul_q(x, cc) if acc is None else fma_q(x, cc, acc)
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
    t0x = round_float(add_q(round_float(sub_q(f32(1), gx), t), ffx), t)
    t1x = round_float(sub_q(gx, ffx), t)
    t0y = round_float(add_q(round_float(sub_q(f32(1), gy), t), ffy), t)
    t1y = round_float(sub_q(gy, ffy), t)
    z = f32(0)
    cs = [round_float(mul_q(np.where(xo0 | yo0, z, t0x), t0y), t),
          round_float(mul_q(np.where(xo1 | yo0, z, t1x), t0y), t),
          round_float(mul_q(np.where(xo0 | yo1, z, t0x), t1y), t),
          round_float(mul_q(np.where(xo1 | yo1, z, t1x), t1y), t)]
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
        out[..., c:c + 2] = fp8_in(sl, f32(scale), t, Rnd.kHalfEven).reshape(H, W, 2)
    return out


def gridsample_fp8(x8, grid_bits, t, ifm_scale, ofm_scale):
    """ofm_scale = 1 / s_out."""
    acc = _interp_core(fp8_ifm(x8, ifm_scale, t), load_float(grid_bits, t), t)
    return fp8_out(acc, f32(ofm_scale), Rnd.kHalfEven)


def _qdq_gridsample(x, grid, align_corners, q, t, out_t):
    """q = (grid_zp, grid_scale, ifm_zp, ifm_scale, ofm_zp, ofm_mult); t = compute
    type."""
    gzp, gs, izp, isc, ozp, omul = q
    H, W, C = x.shape
    off = dq_offset(f32(izp), f32(isc))
    v = round_float(dequant(np.asarray(x, np.int64).reshape(-1), f32(isc), off), t)
    gx = qdq_grid_coord(grid[..., 0], gzp, gs, W, align_corners)
    gy = qdq_grid_coord(grid[..., 1], gzp, gs, H, align_corners)
    fx, fy = float_to_int(gx, "i16", 0, Rnd.kFloor), float_to_int(gy, "i16", 0, Rnd.kFloor)
    fxc, fyc = np.minimum(fx, W - 1), np.minimum(fy, H - 1)
    nx, ny = wrap(fx + 1, "i16"), wrap(fy + 1, "i16")
    ix0, ix1 = (fx >= 0) & (fx < W), (nx >= 0) & (nx < W)
    iy0, iy1 = (fy >= 0) & (fy < H), (ny >= 0) & (ny < H)
    ffx, ffy = fxc.astype(f32), fyc.astype(f32)
    t0x, t1x = add_q(sub_q(f32(1), gx), ffx), sub_q(gx, ffx)
    t0y, t1y = add_q(sub_q(f32(1), gy), ffy), sub_q(gy, ffy)
    z = f32(0)
    cs = [np.where(ix0 & iy0, round_float(mul_q(t0x, t0y), t), z),
          np.where(ix1 & iy0, round_float(mul_q(t1x, t0y), t), z),
          np.where(ix0 & iy1, round_float(mul_q(t0x, t1y), t), z),
          np.where(ix1 & iy1, round_float(mul_q(t1x, t1y), t), z)]
    acc = resize(v.reshape(H * W, C), gather_index(fxc, fyc, H, W), cs)
    return quant(acc, f32(omul), f32(ozp), out_t)


def qdq_gridsample_a8(x, grid, align_corners, q, out_t):
    """8-bit QDQ GridSample; the compute type is bf16."""
    return _qdq_gridsample(x, grid, align_corners, q, "bf16", out_t)


def qdq_gridsample_a16(x, grid, align_corners, q, t, out_t):
    """16-bit QDQ GridSample; t = compute type."""
    return _qdq_gridsample(x, grid, align_corners, q, t, out_t)


# ---- grid from an int8 flow, fused with its ReduceL2 ----

# Field order of npu_gridsample_flow.
FLOW_FIELDS = ("ka", "mq_x", "msh_x", "mq_y", "msh_y", "shift", "ax", "bx", "ay", "by",
               "norm_mode", "n_mq", "n_msh", "out_shift", "p_mq", "p_msh", "p_am", "p_ac",
               "p_ash")


def flow_coord(T, f, ka, mq, msh, a, b, shift, length, border):
    t = "i16" if border else "i8"
    e = srs(np.asarray(T, np.int64) * (1 << ka) + f, ka, t)
    r = srs(srs(e * mq, msh, t) * a + b, shift, "i16")
    if not border:
        return r
    g_hi = saturate(srs(127 * mq, msh, "i16"), "i8")
    hi = int(clamp64(srs(g_hi * a + b, shift, "i16"), 0, length - 1))
    return clamp64(r, 0, hi)


def flow_norm(fx, fy, p):
    sq = saturate(fx * fx + fy * fy, "i16")
    cmp_sh = 0 if p["norm_mode"] == 2 else 2
    v = np.full(sq.shape, -1, np.int64)
    for st in range(6, -1, -1):
        c = v + (2 << st)
        v = np.where(srs(c * c, cmp_sh, "i16") < sq, c, v)
    k = srs(v + 1, 1, "i16")
    if p["norm_mode"] == 0:
        return k
    if p["norm_mode"] == 2:
        t = np.minimum(v + 2, 181)
        k = np.where((saturate(t * t, "i16") == sq) & ((v & 3) == 1), k + 1, k)
    return saturate(srs(k * p["n_mq"], p["n_msh"], "i16"), "i8")


def _gridsample_reducel2_fused(x, fa, tx, ty, padding_mode, f_up, q, p, fc=None):
    """x int8 [H][W][4], fa (fc) int8 [Hf][Wf][4], tx [Wo], ty [Ho]; q = (grid_zp, grid_scale,
    ifm_zp, ifm_scale, ofm_zp, ofm_scale), p a dict of FLOW_FIELDS -> int8 [Ho][Wo][8]."""
    H, W, _ = x.shape
    Ho, Wo = len(ty), len(tx)
    f = np.asarray(fa, np.int64)[..., :2]
    if fc is not None:
        m = saturate(srs(f * p["p_mq"], p["p_msh"], "i16"), "i8")
        c = np.asarray(fc, np.int64)[..., :2]
        f = saturate(srs(m * p["p_am"] + c * p["p_ac"], p["p_ash"], "i16"), "i8")
    if f_up:
        f = f.repeat(2, axis=0).repeat(2, axis=1)
    fx, fy = f[..., 0], f[..., 1]
    border = padding_mode == 1
    tx = np.asarray(tx, np.int64)[None, :]
    ty = np.asarray(ty, np.int64)[:, None]
    rx = flow_coord(tx, fx, p["ka"], p["mq_x"], p["msh_x"], p["ax"], p["bx"], p["shift"], W,
                    border)
    ry = flow_coord(ty, fy, p["ka"], p["mq_y"], p["msh_y"], p["ay"], p["by"], p["shift"], H,
                    border)
    valid = (rx >= 0) & (rx < W) & (ry >= 0) & (ry < H)
    code = np.asarray(x, np.int64)[np.clip(ry, 0, H - 1), np.clip(rx, 0, W - 1)]  # [Ho][Wo][4]
    out = np.zeros((Ho, Wo, 8), np.int8)
    if border and p["out_shift"] >= 0:
        out[..., :4] = srs(code, p["out_shift"], "i8")
    else:
        _, _, izp, isc, ozp, osc = q
        # the zero point is negated in int16 (no -0, -32768 stays), unlike dq_offset
        off = mul_q(f32(np.int16(-izp)), f32(isc))
        v = mul_q(dequant(code, f32(isc), off), valid[..., None].astype(f32))
        v = round_float(v, "bf16")
        out[..., :4] = quant(v, f32(osc), f32(np.int16(ozp)), "i8")
    n = flow_norm(fx, fy, p).astype(np.int32)
    out[..., 4:] = np.ascontiguousarray(n).view(np.int8).reshape(Ho, Wo, 4)
    return out


def qdq_Gridsample_ReduceL2_Fused_a8(x, flow, tx, ty, padding_mode, f_up, q, p):
    return _gridsample_reducel2_fused(x, flow, tx, ty, padding_mode, f_up, q, p)


def qdq_Resize_Gridsample_ReduceL2_Fused_a8(x, a, c, tx, ty, padding_mode, f_up, q, p):
    """The flow is formed from a and c first."""
    return _gridsample_reducel2_fused(x, a, tx, ty, padding_mode, f_up, q, p, fc=c)
