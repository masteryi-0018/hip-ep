# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for SigTanh and SiLU on bf16 / f16, bfp16, fp8 and quantized integers."""

import numpy as np

from golden_block_fp import decode_bfp16_identity, encode_bfp16
from golden_float_cvt import (encode_float, f32_to_fp8, fp8_decode_identity, kFtzIn, kSat,
                              load_float, round_float)
from golden_int_round import Rnd, float_to_int
from golden_mac import fma, mac_exact, mul
from golden_act import (SIGMOID_K1, SIGMOID_K2, SIGTANH_K0, TANH_K1, TANH_K2, hw_sigtanh_f32,
                        hw_silu_f32)

F32 = np.float32
SIGMOID, TANH = 0, 1
FIXED_K = {SIGMOID: (SIGTANH_K0, SIGMOID_K1, SIGMOID_K2), TANH: (SIGTANH_K0, TANH_K1, TANH_K2)}


def to16(z, t, rnd=Rnd.kHalfAway):
    """fp32 -> bf16 / f16 bits; an fp32 subnormal flushes to a signed 0."""
    return encode_float(z, t, rnd, kFtzIn).astype(np.uint16)


def narrow(z, t, rnd=Rnd.kHalfAway):
    return round_float(z, t, rnd, kFtzIn)


def sigtanh(x, act, k):
    return hw_sigtanh_f32(x, act == TANH, *k)


def bfp16_identity_decode(blk):
    """bfp16 -> bf16 values: the identity-multiply decode, then bf16."""
    return narrow(decode_bfp16_identity(blk), "bf16")


def dq(x, zp, s, ct):
    """t(fma(x, s, fp32(-zp * s))), x exact."""
    zps = mul(-F32(zp), F32(s))
    return narrow(mac_exact(np.asarray(x, np.int64), F32(s), zps), ct)


def q(v, zp, s, out_t):
    return float_to_int(fma(v, F32(s), F32(zp)), out_t, 0, Rnd.kHalfAway)


def sigtanh_16(bits, t, act):
    return to16(sigtanh(load_float(bits, t), act, FIXED_K[act]), t)


def silu_16(bits, t):
    return to16(hw_silu_f32(load_float(bits, t)), t)


def bfp16_op(blk, silu, act=0):
    x = bfp16_identity_decode(blk)
    z = hw_silu_f32(x) if silu else sigtanh(x, act, FIXED_K[act])
    return encode_bfp16(narrow(z, "bf16"), Rnd.kHalfAway)


def sigtanh_fp8(codes, act, k1, k2, recip):
    c = sigtanh(fp8_decode_identity(codes), act, (SIGTANH_K0, k1, k2))
    return f32_to_fp8(mul(c, recip), Rnd.kHalfAway)


def silu_fp8(codes, ct, ifm_scale, recip):
    x = narrow(mul(fp8_decode_identity(codes), ifm_scale), ct)
    s = narrow(hw_silu_f32(x), ct)
    r = narrow(F32(recip), ct, Rnd.kHalfEven if ct == "bf16" else Rnd.kHalfAway)
    return f32_to_fp8(mul(s, r), Rnd.kHalfAway, kSat)


def qdq_sigtanh(x, out_t, act, ct, dq_zp, dq_scale, q_zp, q_scale, k):
    y = narrow(sigtanh(dq(x, dq_zp, dq_scale, ct), act, k), "bf16")
    return q(y, q_zp, q_scale, out_t)


def qdq_silu(x, out_t, ct, dq_zp, dq_scale, q_zp, q_scale):
    y = narrow(hw_silu_f32(dq(x, dq_zp, dq_scale, ct)), ct)
    return q(y, q_zp, q_scale, out_t)
