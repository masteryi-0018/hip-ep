# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for GeLU / Erf; functions take and return raw element bits / integers."""

import numpy as np

from golden_block_fp import decode_bfp16_identity, encode_bfp16
from golden_float_cvt import (bf16_to_f32, encode_float, f32_to_bf16, f32_to_fp8,
                              fp8_decode_identity, kFtzIn, kSat, load_float,
                              round_float, scalar_cast)
from golden_int_round import Rnd, float_to_int, saturate_sym, shift_round
from golden_mac import add, fma, mac_exact, mul
from golden_minmax import vclamp
from golden_act import GELU_K, hw_gelu_f32
from golden_nlf import hw_exp2_bf20

f32 = np.float32
Q1, Q3, Q5 = f32(1.125), f32(-0.376953125), f32(0.11279296875)


def gelu16(bits, t):
    """GeLU on t (bf16 / f16) bits -> t bits, half away from zero; bf16 subnormal results
    flush to a signed zero."""
    g = hw_gelu_f32(load_float(bits, t))
    return encode_float(g, t, Rnd.kHalfAway, kFtzIn).astype(np.uint16)


def gelu_bfp16(blk):
    x = decode_bfp16_identity(blk)
    y = bf16_to_f32(gelu16(f32_to_bf16(x), "bf16"))
    return encode_bfp16(y, Rnd.kHalfAway)


def gelu_fp8(codes, t, ifm_scale, recip):
    """fp8 codes -> fp8 codes; t = compute dtype."""
    x = mul(fp8_decode_identity(codes), ifm_scale)
    y = load_float(gelu16(encode_float(x, t, Rnd.kHalfAway), t), t)
    r = scalar_cast(recip, t, Rnd.kHalfAway)
    return f32_to_fp8(mul(y, r), Rnd.kHalfAway, kSat)


def qdq_gelu(q, zi, zo, ka, c, kdsp, out_t):
    """Integers -> integers of out_t ('i8' ...)."""
    qi = np.asarray(q, np.int64) - int(zi)
    aq, rq = np.abs(qi), np.maximum(qi, 0)
    v = mac_exact(aq, f32(ka), GELU_K[1])
    e = hw_exp2_bf20(fma(-v, v, f32(kdsp)))
    lo = mac_exact(aq, -e, f32(zo))
    return float_to_int(mac_exact(rq, f32(c), lo), out_t, 0, Rnd.kHalfAway)


def qdq_gelu_coeffs(ifm_scale, ifm_zp, ofm_scale, ofm_zp):
    """Coefficients of qdq_gelu (double, stored as fp32)."""
    import math
    cc = float(ifm_scale) * (1.0 / float(ofm_scale))
    return (int(ifm_zp), f32(float(ofm_zp)), f32(-0.750399888 * float(ifm_scale)), f32(cc),
            f32(894.533813 + math.log2(cc)))


def _srs_x8(x, s, rnd):
    """x >> s with rnd 0 ceil, 1 nearest ties toward zero, 2 away from zero; saturated."""
    r = (Rnd.kCeil, Rnd.kHalfZero, Rnd.kAwayFromZero)[rnd]
    return saturate_sym(shift_round(x, s, r), "i8")


def xint8_gelu(q, up_path, rnd, a_shift, b_shift, q4_shift, r_shift, K, P, pe=0):
    q = np.asarray(q, np.int64)
    A = q << a_shift
    q4 = _srs_x8(A, q4_shift, rnd)
    R = np.maximum(_srs_x8(A, r_shift, rnd), 0)
    k1 = up_path and a_shift == 1
    B = q if k1 else _srs_x8(A, b_shift, rnd)
    aq4 = np.abs(q4)
    m = np.maximum(K - aq4, 0)
    h = np.minimum(P, m)
    if pe and k1:
        w8 = lambda v: np.asarray(v).astype(np.int64).astype(np.int8).astype(np.int64)
        h = np.maximum(h, w8(np.maximum(w8(K + 1) - aq4, 0) - m))
        h = np.minimum(h, w8(aq4 + max(P - 2, 0)))
    t = (R - h).astype(np.int8).astype(np.int64)
    return np.maximum(B, t).astype(np.int8)


def erf_poly(x, t):
    """The Erf polynomial in t on fp32 x (already t values) -> t bits."""
    x = np.asarray(x, f32)
    x2 = round_float(mul(x, x), t)
    x3 = round_float(mul(x2, x), t)
    x4 = round_float(mul(x2, x2), t)
    x5 = round_float(mul(x4, x), t)
    a3 = add(add(mul(Q1, x), mul(Q3, x3)), mul(Q5, x5))
    return encode_float(vclamp(round_float(a3, t), f32(-1), f32(1)), t).astype(np.uint16)


def erf16(bits, t):
    return erf_poly(load_float(np.asarray(bits, np.uint16), t, kFtzIn), t)


def erf_fp8(codes, scale, q3, q5, lim):
    x = mul(fp8_decode_identity(codes), scale)
    x2 = mul(x, x)
    p135 = fma(x2, fma(x2, f32(q5), f32(q3)), Q1)
    r = vclamp(mul(x, p135), -f32(lim), f32(lim))  # NaN -> +NaN
    return f32_to_fp8(np.where(r == 0, f32(0), r), Rnd.kHalfAway)


def erf_fp8_coeffs(ifm_scale, ofm_scale):
    i, o = float(ifm_scale), float(ofm_scale)
    return f32(i / o), f32(-0.376953125 * o * o), f32(0.11279296875 * o * o * o * o), f32(1.0 / o)


def erf_qdq(q, c1, c0, c1_out, c0_out, out_t):
    x = mac_exact(np.asarray(q, np.int64), f32(c1), f32(c0))
    r = bf16_to_f32(erf_poly(round_float(x, "bf16"), "bf16"))
    y = add(mul(r, scalar_cast(c1_out, "bf16")), scalar_cast(c0_out, "bf16"))
    return float_to_int(y, out_t)


def erf_qdq_coeffs(ifm_scale, ifm_zp, ofm_scale, ofm_zp):
    return (f32(float(ifm_scale)), f32(0 - float(ifm_scale) * float(ifm_zp)),
            f32(1.0 / float(ofm_scale)), f32(float(ofm_zp)))
