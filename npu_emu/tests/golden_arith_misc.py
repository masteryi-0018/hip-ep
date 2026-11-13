# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the arith_misc group of npu_emu
(include/npu_emu_arith_misc.h), written from the mllib kernels and their
corrected golden models (fp32_exact.py, inv_lut.py), independently of the HIP
code. Tensors are logical; t is a key of golden.INT_RANGE."""

import os
import re

import numpy as np

import golden as g

f32 = np.float32
HERE = os.path.dirname(os.path.abspath(__file__))
INV_F32_TABLE = os.path.join(HERE, "..", "src", "npu_emu_inv_f32_table.h")


# ---- arithmetic -------------------------------------------------------------

def to_int(v, t, shift=0):
    """fp32 -> integer type t: v * 2^shift, half to even, saturated; NaN -> 0."""
    v = np.asarray(v, np.float64) * (2.0 ** shift)
    return g.rne_int(v, *g.INT_RANGE[t])


def mac(x, c, a):
    """x * c + a rounded once to fp32 with the integer x exact; c, a broadcast."""
    x = np.asarray(x, np.int64)
    c = np.broadcast_to(np.asarray(c, f32), x.shape).reshape(-1)
    a = np.broadcast_to(np.asarray(a, f32), x.shape).reshape(-1)
    return g.mac_int(x.reshape(-1), c, a).reshape(x.shape)


def fma(x, c, a):
    """fp32 x * c + a with one rounding; non-finite x gives x * c + a in IEEE."""
    x = np.asarray(x, f32)
    fin = np.isfinite(x)
    out = np.empty(x.shape, f32)
    c = np.broadcast_to(np.asarray(c, f32), x.shape)
    a = np.broadcast_to(np.asarray(a, f32), x.shape)
    out[fin] = g.fma32(x[fin], c[fin], a[fin])
    with np.errstate(invalid="ignore"):
        out[~fin] = x[~fin] * c[~fin] + a[~fin]
    return out


def bf16(x):
    """fp32 values rounded half to even to bf16, as fp32."""
    return g.bf16_bits_to_f32(g.to_bf16(np.asarray(x, f32)))


# ---- reciprocal units --------------------------------------------------------

def inv_bf16(y):
    """AIE4 bf16 reciprocal of bf16 values y (fp32 array): RNE(1/y), with bf16
    subnormals flushed - zero / subnormal y gives +-inf, a subnormal result +-0.
    Equals mllib inv_lut.py inv_bf16_aie4 on all 65536 codes (checked by
    test_inv_bf16_matches_mllib where the mllib tree is reachable)."""
    y = np.asarray(y, f32)
    with np.errstate(all="ignore"):
        r = bf16((1.0 / y.astype(np.float64)).astype(f32))
    tiny = np.abs(y) < f32(2.0 ** -126)
    r = np.where(tiny, np.copysign(f32(np.inf), y), r)
    r = np.where(np.abs(r) < f32(2.0 ** -126), np.copysign(f32(0), r), r)
    return r.astype(f32)


def inv_f16(h):
    """AIE4 fp16 reciprocal of fp16 values h (np.float16): RNE(1/h) in fp16."""
    with np.errstate(all="ignore"):
        return (1.0 / np.asarray(h, np.float16).astype(np.float64)).astype(np.float16)


def inv_f32_dev():
    """The -1 / 0 / +1 ulp corrections of the fp32 ::inv, from the header that
    holds them (None if it was removed)."""
    if not os.path.exists(INV_F32_TABLE):
        return None
    src = open(INV_F32_TABLE).read()
    body = src[src.index("kInvF32Dev[2048] = {"):]
    words = [int(w, 16) for w in re.findall(r"0x([0-9a-f]{8})u", body)][:2048]
    w = np.array(words, np.uint32)
    f = np.arange(32768)
    return ((w[f >> 4] >> (2 * (f & 15)).astype(np.uint32)) & 3).astype(np.int64) - 1


def inv_f32(x, dev):
    """AIE4 ::inv of nonzero fp32 x with at most 16 significant bits."""
    x = np.asarray(x, f32)
    bits = x.view(np.uint32)
    assert not np.any(bits & 0xFF) and np.all(x != 0)
    rne = (1.0 / x.astype(np.float64)).astype(f32).view(np.uint32).astype(np.int64)
    return (rne + dev[(bits >> 8) & 0x7FFF]).astype(np.uint32).view(f32)


# ---- per-wrapper models -------------------------------------------------------

def div(a, b, c0, c1, zp1_word, zp2_word, out_t):
    """elementwise_div_a8 / _a16."""
    fa = (f32(12582912) + np.asarray(a, np.int64).astype(f32)) - f32(zp1_word)
    fb = (f32(12582912) + np.asarray(b, np.int64).astype(f32)) - f32(zp2_word)
    with np.errstate(invalid="ignore"):
        q = (bf16(fa).astype(np.float64) * inv_bf16(bf16(fb))).astype(f32)
    return to_int(fma(q, c0, c1), out_t)


def bcast(b, shape):
    return np.broadcast_to(np.asarray(b), shape)


def bdiv_a8(a, b, c0, c1, zp1, zp2, shift, t):
    """qdq_elementwise_broadcasting3d_div_a8: a [H][W][C], b broadcastable."""
    da = (np.asarray(a, np.int64) - zp1).astype(f32).astype(np.float16)
    db = (bcast(b, a.shape).astype(np.int64) - zp2).astype(f32).astype(np.float16)
    db = np.where(db == 0, np.float16(7e-5), db)
    q = (da.astype(np.float64) * inv_f16(db).astype(np.float64)).astype(f32)
    return to_int(fma(q, c0, c1), t, -shift)


def bdiv_a16(a, b, c0, c1, zp1, zp2, shift, t, dev):
    """qdq_elementwise_broadcasting3d_div_a16 (needs the ::inv corrections)."""
    da = (np.asarray(a, np.int64) - zp1).astype(f32)
    db = (bcast(b, a.shape).astype(np.int64) - zp2).astype(f32)
    inv = np.full(db.shape, f32(1.0 / np.float64(f32(7e-5))), f32)
    nz = db != 0
    inv[nz] = inv_f32(db[nz], dev)
    q = (da.astype(np.float64) * inv.astype(np.float64)).astype(f32)
    return to_int(fma(q, c0, c1), t, -shift)


def leakyrelu(x, c1, c0, c1a, c0a, t):
    return to_int(np.maximum(mac(x, c1, c0), mac(x, c1a, c0a)), t)


def prelu(x, pos, in_zp, out_zp, neg, t):
    """x [rows][C], neg [C]."""
    d = np.asarray(x, np.int64) - int(in_zp)
    s = np.where(d < 0, np.broadcast_to(np.asarray(neg, f32), d.shape), f32(pos))
    return to_int(mac(d, s, out_zp), t)


def square(x, a, b, c, t):
    inner = mac(x, a, b)
    return to_int(mac(x, inner, c), t)


def batchnorm(x, cw, cb, act, zp, alpha, t):
    """x [rows][C]; act 0 none, 1 relu, 3 leaky relu."""
    o = mac(x, cw, cb)
    if act == 1:
        o = np.maximum(o, f32(zp))
    elif act == 3:
        lc = f32(f32(zp) * f32(f32(1.0) - f32(alpha)))
        o = np.maximum(o, fma(o, alpha, lc))
    return to_int(o, t)
