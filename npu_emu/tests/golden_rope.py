# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the rope group of npu_emu (include/npu_emu_rope.h),
after mllib's corrected RopeBf16/Fp8/Bfp16StreamWrapper models (fused
mul + msc / mac where the kernel uses core_vec_body, four rounded products and
one sub / add where it uses rope_fma_fn), written independently of the HIP
code. x is [rows][seq][K]; cos / sin are [seq][half]."""

import numpy as np

import golden as g

f32 = np.float32


def rne16(v, t):
    """Round fp32 half to even to t ("bf16" / "f16") and widen back."""
    return g.load_float(g.store_float(np.asarray(v, f32), t), t)


def rotate_half_pairs(x, K, rot):
    """(xn, xr) views of the rotated prefix: x[..., j] and x[..., j + rot/2]."""
    h = rot // 2
    return x[..., :h], x[..., h:rot]


def rot_pair(xn, xr, c, s, fused):
    """(out_n, out_r) in fp32: fused = mul then msc / mac (one rounding each),
    unfused = four rounded products then one rounded sub / add."""
    xn, xr, c, s = (np.asarray(v, f32) for v in (xn, xr, c, s))
    if fused:
        sh = np.broadcast_shapes(xn.shape, c.shape)
        b = lambda v: np.broadcast_to(v, sh).reshape(-1)
        o0 = g.fma32(b(-xr), b(s), b(xn * c)).reshape(sh)
        o1 = g.fma32(b(xr), b(c), b(xn * s)).reshape(sh)
        return o0, o1
    return (xn * c - xr * s).astype(f32), (xn * s + xr * c).astype(f32)


def rope_fp32(x, cos, sin, rot, fused=True):
    """rotate_half RoPE of fp32 x [R][M][K] over the prefix rot; the tail is x.
    Returns fp32 (not yet rounded to the output type)."""
    x = np.asarray(x, f32)
    K = x.shape[-1]
    xn, xr = rotate_half_pairs(x, K, rot)
    c, s = cos[None], sin[None]
    o0, o1 = rot_pair(xn, xr, c, s, fused)
    return np.concatenate([o0, o1, x[..., rot:]], axis=-1)


# ---- rope_bf16_stream ---------------------------------------------------------

def rope_16(bits, t, cos, sin, rot):
    """bits: uint16 [R][M][K] of type t; returns uint16 bits."""
    x = g.load_float(bits, t)
    out = g.store_float(rope_fp32(x, cos, sin, rot), t)
    out[..., rot:] = bits[..., rot:]  # the tail is a bit copy (NaN payloads, -0)
    return out


# ---- rope_fp8_stream ----------------------------------------------------------

def recip16(s, t):
    return f32(rne16(f32(1.0) / f32(s), t))


def fp8_in(codes, ifm_scale, t):
    """Kernel staging of fp8 rows: identity-matmul decode (rows of 8, NaN
    spreads, -0 -> +0), times ifm_scale in fp32, rounded to t."""
    shape = codes.shape
    d = g.fp8_to_f32_row(codes.reshape(-1)).reshape(shape) + f32(0.0)
    return rne16((d * f32(ifm_scale)).astype(f32), t)


def rope_fp8(codes, t, cos, sin, rot, ifm_scale, ofm_scale):
    xd = fp8_in(codes, ifm_scale, t)
    y = rne16(rope_fp32(xd, cos, sin, rot), t)
    v = (y * recip16(ofm_scale, t)).astype(f32)
    v = np.where(np.isnan(v), v, np.clip(v, -448, 448)).astype(f32)
    return g.to_fp8(v, g.EVEN)


# ---- rope_bfp16_stream ----------------------------------------------------------

def rope_bfp16(blk, R, M, K, cos, sin, interleave, fused=None):
    """blk: bfp16 bytes of [R][M][K]; cos / sin fp32 (already the wts values);
    fused defaults to the kernel's choice (rotate_half with K > 128)."""
    if fused is None:
        fused = (not interleave) and K > 128
    x = g.from_bfp16(blk).reshape(R, M, K)
    if interleave:
        o0, o1 = rot_pair(x[..., 0::2], x[..., 1::2], cos[None], sin[None], False)
        y = np.empty_like(x)
        y[..., 0::2] = o0
        y[..., 1::2] = o1
    else:
        y = rope_fp32(x, cos, sin, K, fused)
    y = g.bf16_bits_to_f32(g.to_bf16(y))
    return g.to_bfp16(y.reshape(-1), g.EVEN)
