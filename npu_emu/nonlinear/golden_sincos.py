# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for sin / cos (polynomial and table variants) on bf16 / fp16, fp8 and
quantized integers. Values are fp32 arrays; bf16 / fp16 / fp8 data are raw bits."""

import numpy as np

from golden_float_cvt import encode_float, ftz, load_float, round_float, scalar_cast
from golden_float_cvt import float_to_int, int_to_float
from golden_int_round import Rnd
from golden_quant import dequant, dq_offset, fp8_in, fp8_out, quant
from golden_mac import fma_q, msc_q, mul_q
from golden_nlf import lut_interp

F32 = np.float32
LUT_FLOATS = 2048


def _f32(x):
    return np.asarray(x, F32)


def _rb(x):
    return round_float(x, "bf16")


def _rh(x):
    return round_float(x, "f16")


def _quadrant(z, sin_v, cos_v, is_cos):
    odd = (z & 1) == 1
    neg = (z & 2) == 2
    if not is_cos:
        a = np.where(odd, cos_v, sin_v)
        return np.where(neg, mul_q(-1.0, a), a).astype(F32)
    return np.where(odd, np.where(neg, sin_v, mul_q(-1.0, sin_v)),
                    np.where(neg, mul_q(-1.0, cos_v), cos_v)).astype(F32)


def _b(v):
    return float(_rb(F32(v)))


def _sincos_bf16(ux, is_cos):
    tp = _b(float.fromhex("0x1.45f306dc9c883p-1"))
    pi2 = _b(float.fromhex("0x1.921fb54400000p0"))
    s0, s1, s2, s3 = (_b(float.fromhex(h)) for h in (
        "-0x1.5555555555555p-3", "0x1.1111111110bb3p-7", "-0x1.a01a019e83e5cp-13",
        "0x1.71de3796cde01p-19"))
    c0, c1, c2, c3 = (_b(float.fromhex(h)) for h in (
        "0x1.5555555555555p-5", "-0x1.6c16c16c16967p-10", "0x1.A01A019F4EC91p-16",
        "-0x1.27E4FA17F667Bp-22"))
    ux = ftz(ux)
    z = float_to_int(_rb(fma_q(ux, tp, 0.5)), "i16", 0, Rnd.kFloor)
    zb = _rb(z.astype(F32))
    xa = msc_q(pi2, zb, ux)
    x = _rb(xa)
    r = _rb(mul_q(x, x))
    r2 = _rb(mul_q(r, r))
    t = fma_q(-0.5, r, 1.0)
    t1s, t2s = fma_q(s1, r, s0), fma_q(s3, r, s2)
    t1c, t2c = fma_q(c1, r, c0), fma_q(c3, r, c2)
    outs = fma_q(r2, _rb(t2s), t1s)
    outc = fma_q(r2, _rb(t2c), t1c)
    x3 = _rb(mul_q(r, x))
    sin_v = _rb(fma_q(x3, _rb(outs), xa))
    cos_v = _rb(fma_q(r2, _rb(outc), t))
    return _quadrant(z, sin_v, cos_v, is_cos)


def _sincos_fp16(ux, is_cos):
    h = {k: float(_rh(F32(v))) for k, v in dict(
        inv=0.6367, hi=1.57, lo=0.0004838, s3=-0.1666, s5=0.00833, s7=-0.0001984,
        c2=-0.5, c4=0.04166, c6=-0.001389).items()}
    ux = _f32(ux)
    z = float_to_int(mul_q(ux, h["inv"]), "i16")
    kh = _rh(z.astype(F32))
    ra = msc_q(h["hi"], kh, ux)
    ra = msc_q(h["lo"], kh, ra)
    r = _rh(ra)
    r2 = _rh(mul_q(r, r))
    si = _rh(fma_q(h["s7"], r2, h["s5"]))
    si = _rh(fma_q(si, r2, h["s3"]))
    sp = _rh(fma_q(si, r2, 1.0))
    sin_r = _rh(mul_q(r, sp))
    ci = _rh(fma_q(h["c6"], r2, h["c4"]))
    ci = _rh(fma_q(ci, r2, h["c2"]))
    cos_r = _rh(fma_q(ci, r2, 1.0))
    return _quadrant(z, sin_r, cos_r, is_cos)


def _sincos_native(x, t, is_cos):
    return _sincos_fp16(x, is_cos) if t == "f16" else _sincos_bf16(x, is_cos)


def _lut_core(x, lut):
    ix = ftz(mul_q(x, F32(0.31830987)))
    idx = float_to_int(mul_q(ix, F32(256)), "i32", 0, Rnd.kFloor) & 511
    fl = int_to_float(float_to_int(ix, "i32", 0, Rnd.kFloor), 0, Rnd.kFloor)
    return lut_interp(fma_q(-1.0, fl, ix), _f32(lut), idx, "f32", Rnd.kHalfEven)


def pack_lut(offset, slope):
    """Packing: [64 groups][2 copies][8][offset, slope]."""
    lg = np.stack([_f32(offset), _f32(slope)], axis=1).reshape(64, 1, 8, 2)
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


def _sincos16(bits, t, is_cos):
    y = _sincos_native(load_float(bits, t), t, is_cos)
    return encode_float(y, t).astype(np.uint16)


def _sincos_fp8(codes, t, is_cos, ifm_scale, recip):
    x = fp8_in(codes, F32(ifm_scale), t, Rnd.kHalfEven)
    rc = scalar_cast(recip, t, Rnd.kHalfEven)
    return fp8_out(_sincos_native(x, t, is_cos), rc, Rnd.kHalfEven)


def _lut16(bits, lut, t):
    y = _lut_core(load_float(bits, t), lut)
    return encode_float(y, t).astype(np.uint16)


def _lut_fp8(codes, lut, t, ifm_scale, recip):
    x = fp8_in(codes, F32(ifm_scale), t, Rnd.kHalfAway)
    y = round_float(_lut_core(x, lut), t, Rnd.kHalfAway)
    return fp8_out(y, scalar_cast(recip, t, Rnd.kHalfAway), Rnd.kHalfAway)


def _dq(q, zp, s):
    return dequant(np.asarray(q, np.int64), F32(s), dq_offset(zp, F32(s)))


def _qdq_sincos(q, out_t, t, is_cos, dq_zp, dq_sc, q_zp, q_sc):
    v = round_float(_dq(q, dq_zp, dq_sc), t)
    return quant(_sincos_native(v, t, is_cos), F32(q_sc), F32(q_zp), out_t)


def _qdq_lut(q, lut, out_t, dq_zp, dq_sc, q_zp, q_sc):
    return quant(_lut_core(_dq(q, dq_zp, dq_sc), lut), F32(q_sc), F32(q_zp), out_t)


# ---- the models of the entries ----

def sin_bf16(bits, t):
    return _sincos16(bits, t, False)


def cos_bf16(bits, t):
    return _sincos16(bits, t, True)


def elementwise_sin_fp8(codes, t, ifm_scale, ofm_scale_recip):
    return _sincos_fp8(codes, t, False, ifm_scale, ofm_scale_recip)


def elementwise_cos_fp8(codes, t, ifm_scale, ofm_scale_recip):
    return _sincos_fp8(codes, t, True, ifm_scale, ofm_scale_recip)


def elementwise_sin_a8(q, out_t, t, dq_zp, dq_sc, q_zp, q_sc):
    return _qdq_sincos(q, out_t, t, False, dq_zp, dq_sc, q_zp, q_sc)


def elementwise_cos_a8(q, out_t, t, dq_zp, dq_sc, q_zp, q_sc):
    return _qdq_sincos(q, out_t, t, True, dq_zp, dq_sc, q_zp, q_sc)


# sine and cosine differ only in their table; the 16-bit variants share the models
elementwise_sin_lut_bf16 = elementwise_cos_lut_bf16 = _lut16
elementwise_sin_lut_fp8 = elementwise_cos_lut_fp8 = _lut_fp8
elementwise_sin_a16 = elementwise_sin_a8
elementwise_cos_a16 = elementwise_cos_a8
elementwise_sin_lut_a8 = elementwise_sin_lut_a16 = _qdq_lut
elementwise_cos_lut_a8 = elementwise_cos_lut_a16 = _qdq_lut
