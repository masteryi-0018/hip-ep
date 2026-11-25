# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for quantized division, leaky relu, prelu, square and batchnorm."""

import numpy as np

import golden_quant as gq
from golden_float_cvt import (bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, float_to_int,
                              round_float)
from golden_mac import mac_exact, mul_q, sub_q
from golden_minmax import vmax
from golden_nlf import hw_inv16, hw_inv_f32

f32 = np.float32


def _div(a, b, c0, c1, zp1_word, zp2_word, out_t):
    fa = sub_q(f32(12582912) + np.asarray(a, np.int64).astype(f32), zp1_word)
    fb = sub_q(f32(12582912) + np.asarray(b, np.int64).astype(f32), zp2_word)
    q = mul_q(round_float(fa, "bf16"), bf16_to_f32(hw_inv16(f32_to_bf16(fb), "bf16")))
    return gq.quant(q, c0, c1, out_t)


def _bcast(b, shape):
    return np.broadcast_to(np.asarray(b), shape)


def _bdiv_a8(a, b, c0, c1, zp1, zp2, shift, t):
    """a [H][W][C], b broadcastable."""
    da = round_float((np.asarray(a, np.int64) - zp1).astype(f32), "f16")
    db = round_float((_bcast(b, a.shape).astype(np.int64) - zp2).astype(f32), "f16")
    db = np.where(db == 0, f16_to_f32(0x0496), db)  # f16(7e-5)
    q = mul_q(da, f16_to_f32(hw_inv16(f32_to_f16(db), "f16")))
    return gq.quant(q, c0, c1, t, shift=-shift)


def _bdiv_a16(a, b, c0, c1, zp1, zp2, shift, t):
    da = (np.asarray(a, np.int64) - zp1).astype(f32)
    db = (_bcast(b, a.shape).astype(np.int64) - zp2).astype(f32)
    # A zero divisor uses the correctly rounded 1 / 7e-5f.
    inv = np.where(db == 0, f32(1.0 / np.float64(f32(7e-5))), hw_inv_f32(db))
    return gq.quant(mul_q(da, inv), c0, c1, t, shift=-shift)


def _leakyrelu(x, c1, c0, c1a, c0a, t):
    return float_to_int(vmax(mac_exact(x, c1, c0), mac_exact(x, c1a, c0a)), t)


def _prelu(x, pos, in_zp, out_zp, neg, t):
    """x [rows][C], neg [C]."""
    d = np.asarray(x, np.int64) - int(in_zp)
    s = np.where(d < 0, np.broadcast_to(np.asarray(neg, f32), d.shape), f32(pos))
    return gq.requant(d, s, out_zp, t)


def _square(x, a, b, c, t):
    return gq.requant(x, mac_exact(x, a, b), c, t)


def _batchnorm(x, cw, cb, act, zp, alpha, t):
    """x [rows][C]; act 0 none, 1 relu, 3 leaky relu."""
    o = mac_exact(x, cw, cb)
    if act == 1:
        o = vmax(o, f32(zp))
    elif act == 3:
        o = gq.activate_f32(o, gq.kQdqActLrelu, alpha, mul_q(zp, sub_q(f32(1.0), alpha)))
    return float_to_int(o, t)


# ---- the entry points (named as the entries without the npu_ prefix; arguments in the entry's
# order without the stream and the output buffers; types as 'i8' / 'u8' / ...; shapes as
# (H, W, C)) ----

def elementwise_div_a8(a, in_type, b, out_type, n, c0, c1, zero_point_coeff_1,
                       zero_point_coeff_2):
    return _div(np.asarray(a)[:n], np.asarray(b)[:n], c0, c1, zero_point_coeff_1,
                zero_point_coeff_2, out_type)


elementwise_div_a16 = elementwise_div_a8


def qdq_elementwise_broadcasting3d_div_a8(a, dtype, b, shape1, shape2, ifm1_coeff, ifm2_coeff,
                                          zero_point_coeff_1, zero_point_coeff_2, output_sft):
    return _bdiv_a8(np.asarray(a).reshape(shape1), np.asarray(b).reshape(shape2), ifm1_coeff,
                    ifm2_coeff, zero_point_coeff_1, zero_point_coeff_2, output_sft, dtype)


def qdq_elementwise_broadcasting3d_div_a16(a, dtype, b, shape1, shape2, ifm1_coeff, ifm2_coeff,
                                           zero_point_coeff_1, zero_point_coeff_2, output_sft):
    return _bdiv_a16(np.asarray(a).reshape(shape1), np.asarray(b).reshape(shape2), ifm1_coeff,
                     ifm2_coeff, zero_point_coeff_1, zero_point_coeff_2, output_sft, dtype)


def elementwise_leakyrelu_a8(x, in_type, out_type, n, c1, c0, c1_alpha, c0_alpha):
    return _leakyrelu(np.asarray(x)[:n], c1, c0, c1_alpha, c0_alpha, out_type)


elementwise_leakyrelu_a16 = elementwise_leakyrelu_a8


def elementwise_prelu_a8(x, in_type, out_type, rows, C, positive_scale, ifm_zero_point,
                         ofm_zero_point, prelu_scaled_slope):
    return _prelu(np.asarray(x).reshape(rows, C), positive_scale, ifm_zero_point, ofm_zero_point,
                  prelu_scaled_slope, out_type)


elementwise_prelu_a16 = elementwise_prelu_a8


def elementwise_square_a8(x, in_type, out_type, n, a, b, c):
    return _square(np.asarray(x)[:n], a, b, c, out_type)


elementwise_square_a16 = elementwise_square_a8


def qdq_batchnorm2d_stream_a8(x, dtype, rows, C, cw, cb, ofm_zero_point, activation,
                              relu_alpha):
    return _batchnorm(np.asarray(x).reshape(rows, C), cw, cb, activation, ofm_zero_point,
                      relu_alpha, dtype)


qdq_batchnorm2d_stream_a16 = qdq_batchnorm2d_stream_a8
