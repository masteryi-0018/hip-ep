# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for GeLU / Erf; functions take and return raw element bits / integers."""

import numpy as np

from golden_bits import neg_f32
from golden_block_fp import encode_bfp16
from golden_mmac import decode_bfp16_identity
from golden_float_cvt import (bf16_to_f32, encode_float, f32_to_bf16, f32_to_fp8, kFtzIn,
                              load_float, round_float, scalar_cast)
from golden_mmac import fp8_decode_identity
from golden_float_cvt import float_to_int
from golden_int_round import Rnd, saturate_sym, shift_round, wrap
from golden_mac import add_q, fma_q, mac_exact, msc_q, mul_q
from golden_minmax import vclamp
from golden_act import GELU_K, hw_gelu_f32
from golden_nlf import hw_exp2_bf20
from golden_quant import dequant, fp8_in, fp8_out, requant

f32 = np.float32
Q1, Q3, Q5 = f32(1.125), f32(-0.376953125), f32(0.11279296875)


def gelu_bf16(bits, t):
    """GeLU on t (bf16 / f16) bits -> t bits, half away from zero; bf16 subnormal results
    flush to a signed zero."""
    g = hw_gelu_f32(load_float(bits, t))
    return encode_float(g, t, Rnd.kHalfAway, kFtzIn).astype(np.uint16)


def gelu_bfp16(blk):
    x = decode_bfp16_identity(blk)
    y = bf16_to_f32(gelu_bf16(f32_to_bf16(x), "bf16"))
    return encode_bfp16(y, Rnd.kHalfAway)


def elementwise_gelu_fp8(codes, t, ifm_scale, recip):
    """fp8 codes -> fp8 codes; t = compute dtype."""
    x = fp8_in(codes, ifm_scale, t, Rnd.kHalfAway)
    y = load_float(gelu_bf16(encode_float(x, t, Rnd.kHalfAway), t), t)
    return fp8_out(y, scalar_cast(recip, t, Rnd.kHalfAway), Rnd.kHalfAway)


def qdq_gelu_exp2_a8(q, dq_zp, q_zp, ka, c, kdsp, out_t):
    """Integers -> integers of out_t ('i8' ...)."""
    qi = np.asarray(q, np.int64) - int(dq_zp)
    aq, rq = np.abs(qi), np.maximum(qi, 0)
    v = mac_exact(aq, f32(ka), GELU_K[1])
    e = hw_exp2_bf20(msc_q(v, v, f32(kdsp)))
    lo = mac_exact(aq, neg_f32(e), f32(q_zp))
    return requant(rq, f32(c), lo, out_t, 0, Rnd.kHalfAway)


def qdq_gelu_coeffs(ifm_scale, ifm_zp, ofm_scale, ofm_zp):
    """Coefficients of qdq_gelu_exp2_a8 (double, stored as fp32)."""
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
        h = np.maximum(h, wrap(np.maximum(wrap(K + 1, "i8") - aq4, 0) - m, "i8"))
        h = np.minimum(h, wrap(aq4 + max(P - 2, 0), "i8"))
    t = wrap(R - h, "i8")
    return np.maximum(B, t).astype(np.int8)


def _erf_poly(x, t):
    """The Erf polynomial in t on fp32 x (already t values) -> t bits."""
    x = np.asarray(x, f32)
    x2 = round_float(mul_q(x, x), t)
    x3 = round_float(mul_q(x2, x), t)
    x4 = round_float(mul_q(x2, x2), t)
    x5 = round_float(mul_q(x4, x), t)
    a3 = add_q(add_q(mul_q(Q1, x), mul_q(Q3, x3)), mul_q(Q5, x5))
    return encode_float(vclamp(round_float(a3, t), f32(-1), f32(1)), t).astype(np.uint16)


def elementwise_erf_bf16(bits, t):
    return _erf_poly(load_float(np.asarray(bits, np.uint16), t, kFtzIn), t)


def elementwise_erf_fp8(codes, ifm_coeff, q3_coeff, q5_coeff, clamp_coeff):
    x = mul_q(fp8_decode_identity(codes), ifm_coeff)
    x2 = mul_q(x, x)
    p135 = fma_q(x2, fma_q(x2, f32(q5_coeff), f32(q3_coeff)), Q1)
    r = vclamp(mul_q(x, p135), neg_f32(f32(clamp_coeff)), f32(clamp_coeff))  # NaN -> +NaN
    return f32_to_fp8(np.where(r == 0, f32(0), r), Rnd.kHalfAway)


def erf_fp8_coeffs(ifm_scale, ofm_scale):
    i, o = float(ifm_scale), float(ofm_scale)
    return f32(i / o), f32(-0.376953125 * o * o), f32(0.11279296875 * o * o * o * o), f32(1.0 / o)


def elementwise_erf_a8(q, c1, c0, c1_out, c0_out, out_t):
    x = dequant(np.asarray(q, np.int64), f32(c1), f32(c0))
    r = bf16_to_f32(_erf_poly(round_float(x, "bf16"), "bf16"))
    y = add_q(mul_q(r, scalar_cast(c1_out, "bf16")), scalar_cast(c0_out, "bf16"))
    return float_to_int(y, out_t)


def erf_qdq_coeffs(ifm_scale, ifm_zp, ofm_scale, ofm_zp):
    return (f32(float(ifm_scale)), f32(0 - float(ifm_scale) * float(ifm_zp)),
            f32(1.0 / float(ofm_scale)), f32(float(ofm_zp)))


# the 16-bit variants share the models
qdq_gelu_exp2_a16 = qdq_gelu_exp2_a8
elementwise_erf_a16 = elementwise_erf_a8
