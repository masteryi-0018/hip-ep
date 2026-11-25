# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for qdq.h."""

import numpy as np

from golden_bits import neg_f32
from golden_float_cvt import float_to_int, round_float
from golden_grid import qdq_grid_coord
from golden_int_round import Rnd
from golden_mac import add_q, mac_exact, sub_q
from golden_minmax import vclamp, veq, vgt, vmax, vmin
from golden_quant import dequant, dq_offset, quant, requant

f32 = np.float32


def _abs_safe(x, c1, c0, zp, t):
    with np.errstate(all="ignore"):
        return _abs(x, c1, c0, zp, t)


def _abs(x, c1, c0, zp, t):
    p = mac_exact(x, c1, add_q(f32(c0), f32(zp)))
    q = mac_exact(x, neg_f32(f32(c1)), sub_q(f32(zp), f32(c0)))
    return float_to_int(vmax(p, q), t)


def _clip(x, c1, c0, lo, hi, t):
    return float_to_int(vclamp(mac_exact(x, c1, c0), f32(lo), f32(hi)), t)


def _compare(a, b, sa, sb, za, zb, op):
    fa, fb = dequant(a, sa, za), dequant(b, sb, zb)
    return (veq(fa, fb) if op == "equal" else vgt(fa, fb)).astype(np.int8)


def _minmax(a, b, sa, sb, so, za, zb, zo, t, op):
    """b broadcasts to a's shape (numpy rules)."""
    fa = dequant(a, sa, za)
    fb = np.broadcast_to(dequant(b, sb, zb), fa.shape)
    m = vmax(fa, fb) if op == "max" else vmin(fa, fb)
    return quant(m, so, zo, t)


def _where(cond, x, y, scalar_position):
    c = np.broadcast_to(cond, x.shape) != 0
    if not scalar_position:
        c = ~c
    return np.where(c, x, np.asarray(y).astype(x.dtype))


_AXES = {1: (2,), 2: (1,), 3: (1, 2), 4: (0,), 5: (0, 2), 6: (0, 1), 7: (0, 1, 2)}


def _reduce(x, axis_flags, op):
    x = np.asarray(x, np.int64)
    return (x.max if op == "max" else x.min)(axis=_AXES[axis_flags])


def _maxpool(x, kh, kw, sh, sw):
    x = np.asarray(x, np.int64)
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(0, 1))
    return win[::sh, ::sw].max(axis=(-2, -1))


def _concat(ins):
    return np.concatenate([np.asarray(a, np.int64) for a in ins], axis=-1)


def _gather_axis0(x, idx, scale, offset, t, untouched):
    """Rows outside [0, M_in) keep `untouched`."""
    idx = np.asarray(idx, np.int64)
    ok = (idx >= 0) & (idx < x.shape[0])
    cols = np.broadcast_to(np.arange(idx.shape[1]), idx.shape)
    v = np.asarray(x, np.int64)[np.where(ok, idx, 0), cols]
    return np.where(ok, requant(v, scale, offset, t), untouched)


def _grid_coord(code, zp, scale, extent, align_corners, padding_mode):
    r = float_to_int(qdq_grid_coord(code, zp, scale, extent, align_corners), "i16")
    if padding_mode == 1:
        r = np.clip(r, 0, extent - 1)
    return r


def _gridsample_nearest(x, grid, align_corners, padding_mode, q, t):
    """x [H][W][C] integers, grid [Ho][Wo][2] codes; q = (grid_zp, grid_scale,
    ifm_zp, ifm_scale, ofm_zp, ofm_scale)."""
    gzp, gs, izp, isc, ozp, osc = q
    H, W, _ = x.shape
    rx = _grid_coord(grid[..., 0], gzp, gs, W, align_corners, padding_mode)
    ry = _grid_coord(grid[..., 1], gzp, gs, H, align_corners, padding_mode)
    ok = (rx >= 0) & (rx < W) & (ry >= 0) & (ry < H)
    pix = np.asarray(x, np.int64)[np.where(ok, ry, 0), np.where(ok, rx, 0)]   # [Ho][Wo][C]
    dq = round_float(dequant(pix, isc, dq_offset(f32(izp), isc)), "bf16")
    v = np.where(ok[..., None], dq, f32(0.0)).astype(np.float32)
    return quant(v, osc, f32(ozp), t)


def _resize_nearest(x, sy, sx, requantize, scale, offset, t, untouched=0):
    """A source index past the input leaves the output element `untouched` [default]."""
    x = np.asarray(x, np.int64)
    sy, sx = np.asarray(sy, np.int64), np.asarray(sx, np.int64)
    oky, okx = (sy >= 0) & (sy < x.shape[0]), (sx >= 0) & (sx < x.shape[1])
    v = x[np.where(oky, sy, 0)[:, None], np.where(okx, sx, 0)[None, :]]
    v = requant(v, scale, offset, t, 0, Rnd.kTrunc) if requantize else v
    return np.where((oky[:, None] & okx[None, :])[..., None], v, untouched)


# ---- one model per entry; t is the output type ('i8' / 'u8' / 'i16' / 'u16')

def elementwise_abs_a8(x, c1, c0, ofm_zp, t):
    return _abs_safe(x, c1, c0, ofm_zp, t)


def elementwise_neg_a8(x, c1, c0, t):
    return requant(x, c1, c0, t)


def elementwise_clip_a8(x, c1, c0, cmin, cmax, t):
    return _clip(x, c1, c0, cmin, cmax, t)


def elementwise_equal_a8(a, b, scale_a, scale_b, zp_a, zp_b):
    return _compare(a, b, scale_a, scale_b, zp_a, zp_b, "equal")


def elementwise_greater_a8(a, b, scale_a, scale_b, zp_a, zp_b):
    return _compare(a, b, scale_a, scale_b, zp_a, zp_b, "greater")


def elementwise_max_a8(a, b, scale_a, scale_b, scale_o, zp_a, zp_b, zp_o, t):
    return _minmax(a, b, scale_a, scale_b, scale_o, zp_a, zp_b, zp_o, t, "max")


elementwise_abs_a16, elementwise_neg_a16 = elementwise_abs_a8, elementwise_neg_a8
elementwise_clip_a16, elementwise_equal_a16 = elementwise_clip_a8, elementwise_equal_a8
elementwise_greater_a16, elementwise_max_a16 = elementwise_greater_a8, elementwise_max_a8


def elementwise_broadcasting3d_where_a8(cond, x, scalar_position, y):
    """x holds the raw element bits; y's low bits are the scalar."""
    return _where(cond, x, np.asarray(y).astype(np.int64) & ((1 << 8 * x.dtype.itemsize) - 1),
                  scalar_position)


elementwise_broadcasting3d_where_a16 = elementwise_broadcasting3d_where_a8


def qdq_elementwise_broadcasting3d_where_a8(cond, x, scalar_position, c0, c1, y, t):
    """c0 = scale, c1 = offset."""
    return requant(_where(cond, x, y, scalar_position), c0, c1, t)


qdq_elementwise_broadcasting3d_where_a16 = qdq_elementwise_broadcasting3d_where_a8


def qdq_elementwise_broadcasting3d_min_a8(a, b, scale_a, scale_b, scale_o, zp_a, zp_b, zp_o, t):
    return _minmax(a, b, scale_a, scale_b, scale_o, zp_a, zp_b, zp_o, t, "min")


qdq_elementwise_broadcasting3d_min_a16 = qdq_elementwise_broadcasting3d_min_a8


def reduce_max_c64_a8(x, axis_flags):
    return _reduce(x, axis_flags, "max")


def reduce_min_c64_a8(x, axis_flags):
    return _reduce(x, axis_flags, "min")


reduce_max_c64_a16, reduce_min_c64_a16 = reduce_max_c64_a8, reduce_min_c64_a8


def qdq_reduce_max_c64_a8(x, axis_flags, c0, c1, shift, t):
    """c0 = offset, c1 = scale."""
    return requant(_reduce(x, axis_flags, "max"), c1, c0, t, shift)


def qdq_reduce_min_c64_a8(x, axis_flags, c0, c1, shift, t):
    return requant(_reduce(x, axis_flags, "min"), c1, c0, t, shift)


qdq_reduce_max_c64_a16, qdq_reduce_min_c64_a16 = qdq_reduce_max_c64_a8, qdq_reduce_min_c64_a8


def qdq_maxpool_a8(x, kh, kw, sh, sw, c0, c1, shift, t):
    return requant(_maxpool(x, kh, kw, sh, sw), c1, c0, t, shift)


qdq_maxpool_a16 = qdq_maxpool_a8


def generic_maxpool_a8(x, kh, kw, sh, sw):
    return _maxpool(x, kh, kw, sh, sw)


def qdq_globalmaxpool_a8(x, c0, c1, shift, t):
    return requant(np.asarray(x, np.int64).max(axis=(0, 1)), c1, c0, t, shift)


qdq_globalmaxpool_a16 = qdq_globalmaxpool_a8


def qdq_concat_a8(a, b, c0, c1, t):
    """c0 = scale, c1 = offset."""
    return requant(_concat([a, b]), c0, c1, t)


qdq_concat_a16 = qdq_concat_a8


def qdq_concat_multi_c_fusion_a8(ins, c0, c1, t):
    return requant(_concat(ins), c0, c1, t)


qdq_concat_multi_c_fusion_a16 = qdq_concat_multi_c_fusion_a8


def qdq_concat_multi_c_fusion(ins, c0, c1, t):
    """Per input k: c0[k] = scale, c1[k] = offset."""
    return _concat([requant(x, s, o, t) for x, s, o in zip(ins, c0, c1)])


def qdq_gather_element_2din_2dindices_axis0(x, idx, c1, c0, t, untouched):
    return _gather_axis0(x, idx, c1, c0, t, untouched)


def qdq_gridsample_nearest_a8(x, grid, align_corners, padding_mode, q, t):
    return _gridsample_nearest(x, grid, align_corners, padding_mode, q, t)


qdq_gridsample_nearest_a16 = qdq_gridsample_nearest_a8w8 = qdq_gridsample_nearest_a8


def qdq_resize_nearest(x, src_y, src_x, requantize, c0, c1, t, untouched=0):
    """c0 = scale, c1 = offset."""
    return _resize_nearest(x, src_y, src_x, requantize, c0, c1, t, untouched)


def qdq_reshape_innermost_requantize_fused(x, c1, c0, t):
    return requant(x, c1, c0, t)


def qdq_slice_innermost_requantize_fused(x, start, end, stride, c1, c0, t):
    return requant(x[:, start:end:stride], c1, c0, t)


def qdq_slice_16b_inner_range_requantize_fused(x, start, end, c1, c0, t):
    return requant(x[:, start:end], c1, c0, t)


def qdq_slice_16b_inner_requantize_fused(x, split, c1, c0, t):
    return requant(x[:, :split], c1, c0, t), requant(x[:, split:], c1, c0, t)


def qdq_slice_16b_c6463_requantize_fused(x, c1, c0, t):
    return requant(x[:, :63], c1, c0, t), requant(x[:, 63:64], c1, c0, t)


def qdq_transpose_k8n8_requantize_fused(x, c1, c0, t):
    return requant(x.T, c1, c0, t)


def qdq_transpose_unaligned_requantize_fused(x, perm, c1, c0, t):
    return requant(np.transpose(x, perm), c1, c0, t)
