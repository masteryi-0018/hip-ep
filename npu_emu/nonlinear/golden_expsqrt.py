# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for exp / elu / hardsigmoid / sqrt on bf16 / fp16, fp8 and quantized integers."""

import numpy as np

from golden_float_cvt import (encode_float, f16_to_f32, f32_to_f16, f32_to_fp8, fp8_decode_identity,
                              kFtzIn, load_float, round_float, scalar_cast)
from golden_int_round import INT_RANGE, Rnd, requant_tail
from golden_mac import add, fma, mac_exact, mul
from golden_minmax import vclamp, vmin
from golden_nlf import hw_exp2_bf16, hw_exp2_bf20, hw_invsqrt_f16, hw_sqrt16

f32 = np.float32
LOG2E = f32(1.442695040888963)   # 0x3fb8aa3b
BIAS = f32(895.001)              # 0x445fc010


def exp_hw(x):
    return hw_exp2_bf20(fma(LOG2E, x, BIAS))


def store16(v, t):
    """Half-to-even store; an fp32 subnormal flushes to a signed zero."""
    return encode_float(v, t, Rnd.kHalfEven, kFtzIn).astype(np.uint16)


def exp16(bits, t):
    return store16(exp_hw(load_float(bits, t)), t)


def elu16(bits, t, alpha):
    x = load_float(bits, t)
    return store16(np.where(x >= 0, x, fma(exp_hw(x), alpha, -f32(alpha))), t)


def hardsigmoid16(bits, t, alpha, beta):
    return store16(vclamp(fma(load_float(bits, t), alpha, beta), f32(0), f32(1)), t)


def sqrt16(bits, t):
    return hw_sqrt16(bits, t)


def exp_fp8(codes, in_coeff, bias):
    return f32_to_fp8(hw_exp2_bf20(fma(fp8_decode_identity(codes), in_coeff, bias)), Rnd.kHalfAway)


def elu_fp8(codes, in_scale, pos_scale, neg_scale):
    x = fp8_decode_identity(codes)
    r = mul(x, in_scale)
    neg = fma(exp_hw(r), neg_scale, -f32(neg_scale))
    return f32_to_fp8(np.where(r >= 0, mul(x, pos_scale), neg), Rnd.kHalfAway)


def hardsigmoid_fp8(codes, slope, offset, upper):
    a = fma(fp8_decode_identity(codes), slope, offset)
    return f32_to_fp8(vclamp(a, f32(0), f32(upper)), Rnd.kHalfAway)


def sqrt_fp8(codes, scale):
    x = fp8_decode_identity(codes)
    inv = f16_to_f32(hw_invsqrt_f16(f32_to_f16(x)))
    y = round_float(mul(inv, x), "f16")   # product exact in fp32
    y = np.where(x == 0, x, y)
    y = np.where(x >= 0, y, np.uint32(0xFFC00000).view(f32))
    return f32_to_fp8(mul(y, scale), Rnd.kHalfEven)


def exp2_bf16out(p):
    """Hardware exp2 of fp32(p + 895.001) with a bf16 result."""
    return hw_exp2_bf16(add(p, BIAS))


def exp_qdq(x, ot, c1, c0, c1_out, c0_out):
    e = exp2_bf16out(vmin(mac_exact(x, c1, c0), f32(127.0)))
    return requant_tail(fma(e, scalar_cast(c1_out, "bf16"), scalar_cast(c0_out, "bf16")), ot)


def elu_qdq(x, ot, c1, c0, ifm_scale, neg_bias, a_div, neg_const):
    d = mac_exact(x, ifm_scale, neg_bias)
    neg = fma(exp_hw(d), a_div, neg_const)
    return requant_tail(np.where(d > 0, mac_exact(x, c1, c0), neg), ot)


def hardsigmoid_qdq(x, ot, alpha_f, beta_f, lower, upper):
    return requant_tail(vclamp(mac_exact(x, alpha_f, beta_f), f32(lower), f32(upper)), ot)


def sqrt_qdq(x, ot, dq_scale, dq_zp, q_scale, q_zp):
    zp_acc = mul(-f32(dq_zp), f32(dq_scale))
    h = f32_to_f16(mac_exact(x, dq_scale, zp_acc))
    s = f16_to_f32(hw_sqrt16(h, "f16"))
    return requant_tail(fma(s, q_scale, q_zp), ot)  # a NaN sqrt converts to 0


# ---- coefficients (evaluated in double, stored as fp32) ----

def coeff_exp_fp8(ifm_scale, ofm_scale):
    return f32(ifm_scale * 1.4426950408889634), f32(895.001 - np.log2(ofm_scale))


def coeff_elu_fp8(ifm_scale, ofm_scale, alpha):
    return f32(ifm_scale), f32(ifm_scale / ofm_scale), f32(alpha / ofm_scale)


def coeff_hardsigmoid_fp8(ifm_scale, ofm_scale, alpha, beta):
    return f32(alpha * ifm_scale / ofm_scale), f32(beta / ofm_scale), f32(1.0 / ofm_scale)


def coeff_sqrt_fp8(ifm_scale, ofm_scale):
    return f32((ifm_scale ** 0.5) / ofm_scale)


def coeff_exp_qdq(s_in, zp_in, s_out, zp_out):
    return (f32(s_in * 1.4426950408889634), f32(0 - s_in * zp_in * 1.4426950408889634),
            f32(1.0 / s_out), f32(zp_out))


def coeff_elu_qdq(s_in, zp_in, s_out, zp_out, alpha):
    c1 = s_in / s_out
    a_div = alpha / s_out
    return (f32(c1), f32(zp_out - c1 * zp_in), f32(s_in), f32(0.0 - s_in * zp_in), f32(a_div),
            f32(zp_out - a_div))


def coeff_hardsigmoid_qdq(s_in, zp_in, s_out, zp_out, alpha, beta, ot):
    lo, hi = INT_RANGE[ot]
    return (f32(alpha * s_in / s_out),
            f32(beta / s_out + zp_out - alpha * s_in * zp_in / s_out),
            int(max(lo, min(int(np.floor(zp_out)), hi))),
            int(max(lo, min(int(np.floor(1.0 / s_out + zp_out)), hi))))


def coeff_sqrt_qdq(s_in, zp_in, s_out, zp_out):
    return f32(s_in), f32(zp_in), f32(1.0 / s_out), f32(zp_out)
