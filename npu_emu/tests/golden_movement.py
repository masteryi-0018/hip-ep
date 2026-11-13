# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the movement-group entry points (npu_emu_movement.h).

The references follow mllib's python model of each wrapper on the logical
tensor (np.concatenate / np.transpose / np.pad / np.take_along_axis ...), not
the HIP code. Gathers with an out-of-range index leave the output untouched:
the reference starts from the caller's fill pattern."""

import numpy as np

import golden as g
from npu import FILL


def concat_rows(parts):
    """parts: list of [rows][w] arrays."""
    return np.concatenate(parts, axis=1)


def slice_inner(x, start, end, stride=1):
    return x[:, start:end:stride]


def expand_inner(x, cout):
    return np.tile(x, (1, cout // x.shape[1]))


def pad3d(x, pads, mode, pad_bits):
    """x [H][W][C] as unsigned ints of the element width; numpy.pad."""
    pw = ((pads[0], pads[1]), (pads[2], pads[3]), (pads[4], pads[5]))
    if mode == 0:
        return np.pad(x, pw, mode="constant", constant_values=x.dtype.type(pad_bits))
    return np.pad(x, pw, mode="reflect" if mode == 1 else "edge")


def depth_to_space_nchw(x, b, mode):
    N, C, H, W = x.shape
    Cp = C // (b * b)
    if mode == 0:
        t = x.reshape(N, b, b, Cp, H, W).transpose(0, 3, 4, 1, 5, 2)
    else:
        t = x.reshape(N, Cp, b, b, H, W).transpose(0, 1, 4, 2, 5, 3)
    return t.reshape(N, Cp, H * b, W * b)


def depth_to_space_nhwc(x, b, mode):
    return depth_to_space_nchw(x.transpose(0, 3, 1, 2), b, mode).transpose(0, 2, 3, 1)


def space_to_depth_hwc(x, b):
    H, W, C = x.shape
    return (x.reshape(H // b, b, W // b, b, C).transpose(0, 2, 4, 1, 3)
            .reshape(H // b, W // b, C * b * b))


def planar3_c8(x, pad):
    """x [batch][3][P] u8 -> [batch][P][8]."""
    B, _, P = x.shape
    out = np.full((B, P, 8), pad, np.uint8)
    out[:, :, :3] = x.transpose(0, 2, 1)
    return out


def as_u32_index(idx):
    """Indices as the kernels read them: (uint32_t)(index_t)."""
    return np.asarray(idx).astype(np.int64).astype(np.uint32).astype(np.int64)


def gather_axis(data, axis, idx, out_init):
    """np.take along axis; out-of-range (incl. negative) indices keep out_init."""
    x = as_u32_index(idx)
    ok = x < data.shape[axis]
    got = np.take(data, np.where(ok, x, 0), axis=axis)
    shape = [1] * data.ndim
    shape[axis] = len(x)
    return np.where(ok.reshape(shape), got, out_init)


def gather_elements(data, idx, axis, out_init):
    x = as_u32_index(idx)
    ok = x < data.shape[axis]
    got = np.take_along_axis(data, np.where(ok, x, 0), axis=axis)
    return np.where(ok, got, out_init)


def gather_nd(data, idx, depth, out_init):
    """ONNX GatherND batch_dims 0 as the kernel linearizes it (uint32 sum of
    index * row stride; only the linear row is range-checked)."""
    shape = data.shape
    nrows = int(np.prod(shape[:depth]))
    rows = data.reshape(nrows, -1)
    t = as_u32_index(idx).reshape(-1, depth)
    st = [int(np.prod(shape[j + 1:depth])) for j in range(depth)]
    r = (t * np.array(st, np.int64)).sum(axis=1) & 0xFFFFFFFF
    ok = r < nrows
    got = rows[np.where(ok, r, 0)]
    return np.where(ok[:, None], got, out_init)


def resize_hwc(x, ih, iw):
    return x[np.asarray(ih)][:, np.asarray(iw)]


def onnx_nearest_index(out_len, in_len, coord_mode, nearest_mode, fp32=False):
    """Source index table of ONNX Resize nearest. fp32 follows the 4x3 metadata
    packer (float32 inverse scale), otherwise the double arithmetic of mllib's
    ResizeNearestBf16Wrapper model."""
    go = 0.5 if coord_mode in ("half_pixel", "pytorch_half_pixel") else 0.0
    adj = 1 if coord_mode == "align_corners" else 0
    out = np.arange(out_len)
    if fp32:
        inv = np.float32(np.float32(in_len - adj) / np.float32(out_len - adj))
        x = (out.astype(np.float32) + np.float32(go)) * inv - np.float32(go)
        x = np.minimum(np.maximum(np.float32(0), x), np.float32(in_len - 1)).astype(np.float32)
        h = np.float32(0.5)
    else:
        scale = (out_len - adj) / (in_len - adj)
        x = np.minimum(np.maximum(0, (out + go) / scale - go), in_len - 1)
        h = 0.5
    if nearest_mode == "ceil":
        r = np.ceil(x)
    elif nearest_mode == "floor":
        r = np.floor(x)
    elif nearest_mode == "round_prefer_ceil":
        r = np.floor(x + h)
    else:  # round_prefer_floor
        r = np.ceil(x - h)
    return r.astype(np.int32)


def bfp16_concat(in1, n1, in2, n2, rows):
    """concat_bfp16.hpp per row: the leading raw part of in1 copied block for
    block, the rest re-encoded from bf16-rounded decoded values."""
    b1, b2, bo = -(-n1 // 8), -(-n2 // 8), -(-(n1 + n2) // 8)
    in1 = np.asarray(in1, np.uint8).reshape(rows, b1 * 9)
    in2 = np.asarray(in2, np.uint8).reshape(rows, b2 * 9)
    loads = (n1 - 1) // 128
    R = n1 - 128 * loads
    raw = (128 * loads + (64 if R >= 64 else 0)) // 8
    out = np.empty((rows, bo * 9), np.uint8)
    for r in range(rows):
        v1 = g.bf16_bits_to_f32(g.to_bf16(g.from_bfp16(in1[r]), g.EVEN))[:n1]
        v2 = g.bf16_bits_to_f32(g.to_bf16(g.from_bfp16(in2[r]), g.EVEN))[:n2]
        v = np.zeros(bo * 8, np.float32)
        v[:n1 + n2] = np.concatenate([v1, v2])
        enc = g.to_bfp16(v, g.EVEN)
        enc[:raw * 9] = in1[r, :raw * 9]
        out[r] = enc
    return out.reshape(-1)
