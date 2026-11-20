# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference model for rope.h. x is [rows][seq][K]; cos / sin [seq][half]."""

import numpy as np

from golden_block_fp import decode_bfp16_group, encode_bfp16
from golden_float_cvt import (f32_to_fp8, fp8_decode_identity, kSat, load_float, round_float,
                              scalar_cast, store_float)
from golden_int_round import Rnd
from golden_mac import add, fma, mul, sub

f32 = np.float32


def rotate_half_pairs(x, K, rot):
    """(xn, xr) views of the rotated prefix: x[..., j] and x[..., j + rot/2]."""
    h = rot // 2
    return x[..., :h], x[..., h:rot]


def rot_pair(xn, xr, c, s, fused):
    """(out_n, out_r) in fp32: fused = product then fma, unfused = four rounded
    products then one rounded sub / add."""
    xn, xr, c, s = (np.asarray(v, f32) for v in (xn, xr, c, s))
    if fused:
        return fma(-xr, s, mul(xn, c)), fma(xr, c, mul(xn, s))
    return sub(mul(xn, c), mul(xr, s)), add(mul(xn, s), mul(xr, c))


def rope_fp32(x, cos, sin, rot, fused=True):
    """rotate_half RoPE of fp32 x [R][M][K] over the prefix rot; the tail is x."""
    x = np.asarray(x, f32)
    K = x.shape[-1]
    xn, xr = rotate_half_pairs(x, K, rot)
    c, s = cos[None], sin[None]
    o0, o1 = rot_pair(xn, xr, c, s, fused)
    return np.concatenate([o0, o1, x[..., rot:]], axis=-1)


def rope_16(bits, t, cos, sin, rot):
    """bits: uint16 [R][M][K] of type t; returns uint16 bits."""
    x = load_float(bits, t)
    out = store_float(rope_fp32(x, cos, sin, rot), t)
    out[..., rot:] = bits[..., rot:]  # the tail is a bit copy
    return out


def fp8_in(codes, ifm_scale, t):
    """fp8 rows decoded in groups of 8 (NaN spreads, -0 -> +0), scaled in fp32,
    rounded to t."""
    shape = codes.shape
    d = fp8_decode_identity(codes.reshape(-1)).reshape(shape)
    return round_float(mul(d, f32(ifm_scale)), t)


def rope_fp8(codes, t, cos, sin, rot, ifm_scale, ofm_scale):
    xd = fp8_in(codes, ifm_scale, t)
    y = round_float(rope_fp32(xd, cos, sin, rot), t)
    r = scalar_cast(f32(1.0) / f32(ofm_scale), t)
    return f32_to_fp8(mul(y, r), Rnd.kHalfEven, kSat)


def rope_bfp16(blk, R, M, K, cos, sin, interleave, fused=None):
    """blk: bfp16 bytes of [R][M][K]; cos / sin fp32; fused defaults to
    rotate_half with K > 128. x is decoded per 64-value group, as the NPU does."""
    if fused is None:
        fused = (not interleave) and K > 128
    x = decode_bfp16_group(blk).reshape(R, M, K)
    if interleave:
        o0, o1 = rot_pair(x[..., 0::2], x[..., 1::2], cos[None], sin[None], False)
        y = np.empty_like(x)
        y[..., 0::2] = o0
        y[..., 1::2] = o1
    else:
        y = rope_fp32(x, cos, sin, K, fused)
    return encode_bfp16(round_float(y, "bf16").reshape(-1), Rnd.kHalfEven)
