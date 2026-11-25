# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for exp / elu / hardsigmoid / sqrt on bf16 / fp16, fp8 and quantized integers."""

import numpy as np

from golden_bits import default_nan, neg_f32
from golden_float_cvt import (encode_float, f16_to_f32, f32_to_f16, f32_to_fp8, kFtzIn, load_float,
                              round_float, scalar_cast)
from golden_mmac import fp8_decode_identity
from golden_int_round import INT_RANGE, Rnd
from golden_quant import dequant, dq_offset, fp8_out, quant, requant_tail
from golden_mac import fma_q, mac_exact, mul_q
from golden_minmax import vclamp, vmin
from golden_nlf import exp2_bf16, hw_exp2_bf20, hw_invsqrt_f16, hw_sqrt16, kExp2Bias, kLog2e

f32 = np.float32


def _exp_hw(x):
    return hw_exp2_bf20(fma_q(kLog2e, x, kExp2Bias))


def _store16(v, t):
    """Half-to-even store; an fp32 subnormal flushes to a signed zero."""
    return encode_float(v, t, Rnd.kHalfEven, kFtzIn).astype(np.uint16)


def elementwise_exp_bf16(bits, t):
    return _store16(_exp_hw(load_float(bits, t)), t)


def elementwise_elu_bf16(bits, t, alpha):
    x = load_float(bits, t)
    return _store16(np.where(x >= 0, x, fma_q(_exp_hw(x), alpha, neg_f32(f32(alpha)))), t)


def elementwise_hardsigmoid_bf16(bits, t, alpha, beta):
    return _store16(vclamp(fma_q(load_float(bits, t), alpha, beta), f32(0), f32(1)), t)


def sqrt_bf16(bits, t):
    return hw_sqrt16(bits, t)


def elementwise_exp_fp8(codes, in_coeff, bias_coeff):
    a = fma_q(fp8_decode_identity(codes), in_coeff, bias_coeff)
    return f32_to_fp8(hw_exp2_bf20(a), Rnd.kHalfAway)


def elementwise_elu_fp8(codes, ifm_scale, positive_coeff, negative_coeff):
    x = fp8_decode_identity(codes)
    r = mul_q(x, ifm_scale)
    neg = fma_q(_exp_hw(r), negative_coeff, neg_f32(f32(negative_coeff)))
    return f32_to_fp8(np.where(r >= 0, mul_q(x, positive_coeff), neg), Rnd.kHalfAway)


def elementwise_hardsigmoid_fp8(codes, output_slope, output_offset, output_upper):
    a = fma_q(fp8_decode_identity(codes), output_slope, output_offset)
    return f32_to_fp8(vclamp(a, f32(0), f32(output_upper)), Rnd.kHalfAway)


def elementwise_sqrt_fp8(codes, ifm_coeff):
    x = fp8_decode_identity(codes)
    inv = f16_to_f32(hw_invsqrt_f16(f32_to_f16(x)))
    y = round_float(mul_q(inv, x), "f16")   # product exact in fp32
    y = np.where(x == 0, x, y)
    y = np.where(x >= 0, y, default_nan())
    return fp8_out(y, ifm_coeff, Rnd.kHalfEven, 0)


def elementwise_exp_a8(x, ot, c1, c0, c1_out, c0_out):
    e = exp2_bf16(vmin(mac_exact(x, c1, c0), f32(127.0)))
    return quant(e, scalar_cast(c1_out, "bf16"), scalar_cast(c0_out, "bf16"), ot)


def elementwise_elu_a8(x, ot, c1, c0, ifm_scale, neg_bias, alpha_div_ofm_scale, neg_const):
    d = dequant(x, ifm_scale, neg_bias)
    neg = fma_q(_exp_hw(d), alpha_div_ofm_scale, neg_const)
    return requant_tail(np.where(d > 0, mac_exact(x, c1, c0), neg), ot)


def hardsigmoid_a8(x, ot, alpha_f, beta_f, lower_bound_i, upper_bound_i):
    v = vclamp(mac_exact(x, alpha_f, beta_f), f32(lower_bound_i), f32(upper_bound_i))
    return requant_tail(v, ot)


def qdq_sqrt_a8(x, ot, dq_zp, dq_sc, q_zp, q_sc):
    h = f32_to_f16(dequant(x, dq_sc, dq_offset(dq_zp, dq_sc)))
    s = f16_to_f32(hw_sqrt16(h, "f16"))
    return quant(s, q_sc, q_zp, ot)  # a NaN sqrt converts to 0


# the 16-bit variants share the models
elementwise_exp_a16 = elementwise_exp_a8
elementwise_elu_a16 = elementwise_elu_a8
hardsigmoid_a16 = hardsigmoid_a8
qdq_sqrt_a16 = qdq_sqrt_a8


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
    """(dq_zp, dq_sc, q_zp, q_sc)."""
    return f32(zp_in), f32(s_in), f32(zp_out), f32(1.0 / s_out)
