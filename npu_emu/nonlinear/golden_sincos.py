# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for sin / cos (polynomial and table variants) on bf16 / fp16, fp8 and
quantized integers. Values are fp32 arrays; bf16 / fp16 / fp8 data are raw bits."""

import numpy as np

from golden_float_cvt import (encode_float, f32_to_fp8, fp8_decode_identity, ftz, kSat, load_float,
                              round_float, scalar_cast)
from golden_int_round import Rnd, float_to_int, int_to_float, requant_tail
from golden_mac import fma, mul
from golden_nlf import lut_interp

F32 = np.float32
LUT_FLOATS = 2048


def f32(x):
    return np.asarray(x, F32)


def rb(x):
    return round_float(x, "bf16")


def rh(x):
    return round_float(x, "f16")


def quadrant(z, sin_v, cos_v, is_cos):
    odd = (z & 1) == 1
    neg = (z & 2) == 2
    if not is_cos:
        a = np.where(odd, cos_v, sin_v)
        return np.where(neg, mul(-1.0, a), a).astype(F32)
    return np.where(odd, np.where(neg, sin_v, mul(-1.0, sin_v)),
                    np.where(neg, mul(-1.0, cos_v), cos_v)).astype(F32)


def _b(v):
    return float(rb(F32(v)))


def sincos_bf16(ux, is_cos):
    tp = _b(float.fromhex("0x1.45f306dc9c883p-1"))
    npi2 = -_b(float.fromhex("0x1.921fb54400000p0"))
    s0, s1, s2, s3 = (_b(float.fromhex(h)) for h in (
        "-0x1.5555555555555p-3", "0x1.1111111110bb3p-7", "-0x1.a01a019e83e5cp-13",
        "0x1.71de3796cde01p-19"))
    c0, c1, c2, c3 = (_b(float.fromhex(h)) for h in (
        "0x1.5555555555555p-5", "-0x1.6c16c16c16967p-10", "0x1.A01A019F4EC91p-16",
        "-0x1.27E4FA17F667Bp-22"))
    ux = ftz(ux)
    z = float_to_int(rb(fma(ux, tp, 0.5)), "i16", 0, Rnd.kFloor)
    zb = rb(z.astype(F32))
    xa = fma(npi2, zb, ux)
    x = rb(xa)
    r = rb(mul(x, x))
    r2 = rb(mul(r, r))
    t = fma(-0.5, r, 1.0)
    t1s, t2s = fma(s1, r, s0), fma(s3, r, s2)
    t1c, t2c = fma(c1, r, c0), fma(c3, r, c2)
    outs = fma(r2, rb(t2s), t1s)
    outc = fma(r2, rb(t2c), t1c)
    x3 = rb(mul(r, x))
    sin_v = rb(fma(x3, rb(outs), xa))
    cos_v = rb(fma(r2, rb(outc), t))
    return quadrant(z, sin_v, cos_v, is_cos)


def sincos_fp16(ux, is_cos):
    h = {k: float(rh(F32(v))) for k, v in dict(
        inv=0.6367, hi=1.57, lo=0.0004838, s3=-0.1666, s5=0.00833, s7=-0.0001984,
        c2=-0.5, c4=0.04166, c6=-0.001389).items()}
    ux = f32(ux)
    z = float_to_int(mul(ux, h["inv"]), "i16")
    kh = rh(z.astype(F32))
    ra = fma(-h["hi"], kh, ux)
    ra = fma(-h["lo"], kh, ra)
    r = rh(ra)
    r2 = rh(mul(r, r))
    si = rh(fma(h["s7"], r2, h["s5"]))
    si = rh(fma(si, r2, h["s3"]))
    sp = rh(fma(si, r2, 1.0))
    sin_r = rh(mul(r, sp))
    ci = rh(fma(h["c6"], r2, h["c4"]))
    ci = rh(fma(ci, r2, h["c2"]))
    cos_r = rh(fma(ci, r2, 1.0))
    return quadrant(z, sin_r, cos_r, is_cos)


def sincos_native(x, t, is_cos):
    return sincos_fp16(x, is_cos) if t == "f16" else sincos_bf16(x, is_cos)


def lut_core(x, wts):
    ix = ftz(mul(x, F32(0.31830987)))
    idx = float_to_int(mul(ix, F32(256)), "i32", 0, Rnd.kFloor) & 511
    fl = int_to_float(float_to_int(ix, "i32", 0, Rnd.kFloor), 0, Rnd.kFloor)
    return lut_interp(fma(-1.0, fl, ix), f32(wts), idx, "f32", Rnd.kHalfEven)


def pack_lut(offset, slope):
    """Packing: [64 groups][2 copies][8][offset, slope]."""
    lg = np.stack([f32(offset), f32(slope)], axis=1).reshape(64, 1, 8, 2)
    return np.ascontiguousarray(np.repeat(lg, 2, axis=1).reshape(-1)).astype(F32)


def fit_lut(is_cos):
    """A packed table: a least-squares line per 1/256 of [0, 2) in units of pi (16384
    samples)."""
    x = np.arange(16384, dtype=F32) / F32(8192)
    y = (np.cos if is_cos else np.sin)(x * np.pi).astype(np.float64).reshape(512, -1)
    rem = (x - np.floor(x)).astype(F32).astype(np.float64).reshape(512, -1)
    lut = np.empty((512, 2))
    for i in range(512):
        phi = np.vstack((np.ones(y.shape[1]), rem[i]))
        lut[i] = np.linalg.inv(phi @ phi.T) @ phi @ y[i]
    return pack_lut(lut[:, 0].astype(F32), lut[:, 1].astype(F32))


def fp8_in(codes, scale, t, rnd):
    return round_float(mul(fp8_decode_identity(codes), F32(scale)), t, rnd)


def fp8_out(y, rc, rnd):
    return f32_to_fp8(mul(y, F32(rc)), rnd, kSat)


def sincos16(bits, t, is_cos):
    y = sincos_native(load_float(bits, t), t, is_cos)
    return encode_float(y, t).astype(np.uint16)


def sincos_fp8(codes, t, is_cos, ifm_scale, recip):
    x = fp8_in(codes, ifm_scale, t, Rnd.kHalfEven)
    rc = scalar_cast(recip, t, Rnd.kHalfEven)
    return fp8_out(sincos_native(x, t, is_cos), rc, Rnd.kHalfEven)


def lut16(bits, wts, t):
    y = lut_core(load_float(bits, t), wts)
    return encode_float(y, t).astype(np.uint16)


def lut_fp8(codes, wts, t, ifm_scale, recip):
    x = fp8_in(codes, ifm_scale, t, Rnd.kHalfAway)
    y = round_float(lut_core(x, wts), t, Rnd.kHalfAway)
    return fp8_out(y, scalar_cast(recip, t, Rnd.kHalfAway), Rnd.kHalfAway)


def dq(q, zp, s):
    return fma(np.asarray(q, np.int64).astype(F32), F32(s), mul(-F32(zp), F32(s)))


def q_out(y, zp, recip, out_t):
    return requant_tail(fma(y, F32(recip), F32(zp)), out_t)


def qdq_sincos(q, out_t, t, is_cos, izp, iscale, ozp, orc):
    v = round_float(dq(q, izp, iscale), t)
    return q_out(sincos_native(v, t, is_cos), ozp, orc, out_t)


def qdq_lut(q, wts, out_t, izp, iscale, ozp, orc):
    return q_out(lut_core(dq(q, izp, iscale), wts), ozp, orc, out_t)
