# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for movement.h."""

import numpy as np

from golden_block_fp import encode_bfp16
from golden_mmac import decode_bfp16_bf16
from golden_float_cvt import bf16_to_f32
from golden_int_round import Rnd


def _concat_rows(parts):
    return np.concatenate(parts, axis=1)


def _copy(x):
    return np.array(x, copy=True)


def _pad3d(x, pads, mode, pad_value):
    """x [H][W][C] as unsigned ints of the element width; pad_value's low bits are the element."""
    pw = ((pads[0], pads[1]), (pads[2], pads[3]), (pads[4], pads[5]))
    if mode == 0:
        fill = x.dtype.type(int(pad_value) & ((1 << (8 * x.dtype.itemsize)) - 1))
        return np.pad(x, pw, mode="constant", constant_values=fill)
    return np.pad(x, pw, mode="reflect" if mode == 1 else "edge")


def _as_u32_index(idx):
    """Indices as uint32, signed types sign-extended."""
    return np.asarray(idx).astype(np.int64).astype(np.uint32).astype(np.int64)


def _gather_axis(data, axis, idx, out_init):
    """Out-of-range (incl. negative) indices keep out_init."""
    x = _as_u32_index(idx)
    ok = x < data.shape[axis]
    got = np.take(data, np.where(ok, x, 0), axis=axis)
    shape = [1] * data.ndim
    shape[axis] = len(x)
    return np.where(ok.reshape(shape), got, out_init)


def _gather_elements(data, idx, axis, out_init):
    x = _as_u32_index(idx)
    ok = x < data.shape[axis]
    got = np.take_along_axis(data, np.where(ok, x, 0), axis=axis)
    return np.where(ok, got, out_init)


def _resize_hwc(x, ih, iw, out_init):
    """Source indices at or past the input keep out_init."""
    ih, iw = np.asarray(ih, np.int64), np.asarray(iw, np.int64)
    ok = (ih < x.shape[0])[:, None] & (iw < x.shape[1])[None, :]
    got = x[np.minimum(ih, x.shape[0] - 1)][:, np.minimum(iw, x.shape[1] - 1)]
    return np.where(ok[..., None], got, out_init)


# ---- one model per entry (2-D [rows][c] row tensors, HWC images)

def generic_concat(in1, in2, s1, n1, s2, n2):
    return _concat_rows([in1[:, s1:s1 + n1], in2[:, s2:s2 + n2]])


def generic_concat_16b_inner(in1, in2):
    return _concat_rows([in1, in2])


def generic_concat_16b_c6463(in1, in2):
    return _concat_rows([in1[:, :63], in2[:, :1]])


def generic_concat_16b_multi(ins):
    return _concat_rows(ins)


generic_concat_multi_c_fusion = generic_concat_16b_multi


def generic_slice_innermost(x, start, end, stride=1):
    return x[:, start:end:stride]


def generic_slice_16b_inner_range(x, start, end):
    return x[:, start:end]


def generic_slice_16b_inner(x, split):
    return x[:, :split], x[:, split:]


def generic_slice_16b_c6463(x):
    return x[:, :63], x[:, 63:64]


generic_mem_copy = generic_reshape_innermost = _copy
generic_pad_innermost = generic_unpad_innermost = _copy


def generic_expand_innermost(x, cout):
    return np.tile(x, (1, cout // x.shape[1]))


def generic_inplace_pad2d_c64(x, hs, he, ws, we, pad_value):
    return _pad3d(x, (hs, he, ws, we, 0, 0), 0, pad_value)


def generic_inplace_pad3d(x, pads, mode, pad_value):
    return _pad3d(x, pads, mode, pad_value)


def generic_transpose_unaligned(x, perm):
    return np.transpose(x, perm)


generic_transpose4d_aligned = transpose3d_mmac = generic_transpose_unaligned


def generic_transpose4d_0132_RCR32(x):
    return np.swapaxes(x, -1, -2)


def generic_transpose_k8n8(x):
    return x.T


transpose_mmac_a8_n64k64 = transpose_mmac_a16_n32k64 = generic_transpose_k8n8


def transpose_planar3_c8(x, pad):
    """x [batch][3][P] u8 -> [batch][P][8]."""
    B, _, P = x.shape
    out = np.full((B, P, 8), pad, np.uint8)
    out[:, :, :3] = x.transpose(0, 2, 1)
    return out


def generic_depth_to_space(x, b, mode):
    N, C, H, W = x.shape
    Cp = C // (b * b)
    if mode == 0:
        t = x.reshape(N, b, b, Cp, H, W).transpose(0, 3, 4, 1, 5, 2)
    else:
        t = x.reshape(N, Cp, b, b, H, W).transpose(0, 1, 4, 2, 5, 3)
    return t.reshape(N, Cp, H * b, W * b)


def generic_depth_to_space_wc64(x, b, mode):
    return generic_depth_to_space(x.transpose(0, 3, 1, 2), b, mode).transpose(0, 2, 3, 1)


def generic_space_to_depth(x, b):
    H, W, C = x.shape
    return (x.reshape(H // b, b, W // b, b, C).transpose(0, 2, 4, 1, 3)
            .reshape(H // b, W // b, C * b * b))


generic_space_to_depth_c4bs4 = generic_space_to_depth


def generic_gather_3din_1dindices(data, idx, axis, out_init):
    return _gather_axis(data, axis, np.asarray(idx, np.uint32), out_init)


def generic_gather_element_2din_2dindices_axis0(data, idx, out_init):
    return _gather_elements(data, idx, 0, out_init)


def generic_gather_element_2din_2dindices_innermost(data, idx, out_init):
    return _gather_elements(data, idx, 1, out_init)


def resize_nearest_bf16(x, idx_h, idx_w, out_init=0):
    return _resize_hwc(x, idx_h, idx_w, out_init)


def resize_nearest_bfp16(x, idx_h, idx_w, out_init=0):
    """x [Hi][Wi][C/8 * 9] bfp16 blocks."""
    return _resize_hwc(x, idx_h, idx_w, out_init)


def elementwise_not(x):
    return (np.asarray(x) == 0).astype(np.uint8)


def onnx_nearest_index(out_len, in_len, coord_mode, nearest_mode):
    """Source index table of ONNX Resize nearest."""
    go = 0.5 if coord_mode in ("half_pixel", "pytorch_half_pixel") else 0.0
    adj = 1 if coord_mode == "align_corners" else 0
    out = np.arange(out_len)
    scale = (out_len - adj) / (in_len - adj)
    x = np.minimum(np.maximum(0, (out + go) / scale - go), in_len - 1)
    if nearest_mode == "ceil":
        r = np.ceil(x)
    elif nearest_mode == "floor":
        r = np.floor(x)
    elif nearest_mode == "round_prefer_ceil":
        r = np.floor(x + 0.5)
    else:  # round_prefer_floor
        r = np.ceil(x - 0.5)
    return r.astype(np.uint16)


def concat_unaligned_bfp16(in1, in2, rows, n1, n2):
    """Leading blocks of in1 copied as is, the rest re-encoded from bf16 values; rows are
    decoded per 64-value group."""
    b1, b2, bo = -(-n1 // 8), -(-n2 // 8), -(-(n1 + n2) // 8)
    in1 = np.asarray(in1, np.uint8).reshape(rows, b1 * 9)
    in2 = np.asarray(in2, np.uint8).reshape(rows, b2 * 9)
    loads = (n1 - 1) // 128
    R = n1 - 128 * loads
    raw = (128 * loads + (64 if R >= 64 else 0)) // 8
    out = np.empty((rows, bo * 9), np.uint8)
    for r in range(rows):
        v1 = bf16_to_f32(decode_bfp16_bf16(in1[r]))[:n1]
        v2 = bf16_to_f32(decode_bfp16_bf16(in2[r]))[:n2]
        v = np.zeros(bo * 8, np.float32)
        v[:n1 + n2] = np.concatenate([v1, v2])
        enc = encode_bfp16(v, Rnd.kHalfEven)
        enc[:raw * 9] = in1[r, :raw * 9]
        out[r] = enc
    return out.reshape(-1)
