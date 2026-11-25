# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for SigTanh and SiLU on bf16 / f16, bfp16, fp8 and quantized integers."""

import numpy as np

from golden_block_fp import encode_bfp16
from golden_mmac import decode_bfp16_identity
from golden_float_cvt import acc_to_bf16, encode_float, kFtzIn, load_float, round_float
from golden_mmac import fp8_decode_identity
from golden_float_cvt import float_to_int, scalar_cast_i32
from golden_int_round import Rnd
from golden_bits import bits_f32, f32_bits, neg_f32
from golden_mac import add_q, div_q, fma_q, mac_exact, mul_q, sub_q
from golden_minmax import vmax
from golden_nlf import hw_exp2_bf20, kLog2e
from golden_quant import dequant, dq_offset, fp8_in, fp8_out, quant
from golden_act import (SIGMOID_K1, SIGMOID_K2, SIGTANH_K0, TANH_K1, TANH_K2, hw_sigtanh_f32,
                        hw_silu_f32)

F32 = np.float32
SIGMOID, TANH = 0, 1
FIXED_K = {SIGMOID: (SIGTANH_K0, SIGMOID_K1, SIGMOID_K2), TANH: (SIGTANH_K0, TANH_K1, TANH_K2)}


def _to16(z, t, rnd=Rnd.kHalfAway):
    """fp32 -> bf16 / f16 bits; an fp32 subnormal flushes to a signed 0."""
    return encode_float(z, t, rnd, kFtzIn).astype(np.uint16)


def _narrow(z, t, rnd=Rnd.kHalfAway):
    return round_float(z, t, rnd, kFtzIn)


def _sigtanh(x, act, k):
    return hw_sigtanh_f32(x, act == TANH, *k)


def _bfp16_identity_decode(blk):
    """bfp16 -> bf16 values: the identity-multiply decode, then bf16."""
    return acc_to_bf16(decode_bfp16_identity(blk))


def _dq(x, zp, s, ct):
    """Dequantize into the compute type ct."""
    return _narrow(dequant(np.asarray(x, np.int64), F32(s), dq_offset(zp, F32(s))), ct)


def sigtanh_bf16(bits, t, act):
    return _to16(_sigtanh(load_float(bits, t), act, FIXED_K[act]), t)


def silu_bf16(bits, t):
    return _to16(hw_silu_f32(load_float(bits, t)), t)


def _bfp16_op(blk, silu, act=0):
    x = _bfp16_identity_decode(blk)
    z = hw_silu_f32(x) if silu else _sigtanh(x, act, FIXED_K[act])
    return encode_bfp16(acc_to_bf16(z), Rnd.kHalfAway)


def sigtanh_bfp16(blk, act):
    return _bfp16_op(blk, False, act)


def silu_bfp16(blk):
    return _bfp16_op(blk, True)


def elementwise_sigtanh_fp8(codes, act, poly1_coeff, poly2_coeff, ofm_coeff):
    c = _sigtanh(fp8_decode_identity(codes), act, (SIGTANH_K0, poly1_coeff, poly2_coeff))
    return fp8_out(c, ofm_coeff, Rnd.kHalfAway, 0)


def elementwise_silu_fp8(codes, ct, ifm_scale, ofm_scale_recip):
    x = fp8_in(codes, ifm_scale, ct, Rnd.kHalfAway, kFtzIn)
    s = _narrow(hw_silu_f32(x), ct)
    r = _narrow(F32(ofm_scale_recip), ct, Rnd.kHalfEven if ct == "bf16" else Rnd.kHalfAway)
    return fp8_out(s, r, Rnd.kHalfAway)


def qdq_sigtanh_a8(x, out_t, act, ct, dq_zp, dq_sc, q_zp, q_sc, k):
    y = acc_to_bf16(_sigtanh(_dq(x, dq_zp, dq_sc, ct), act, k))
    return quant(y, F32(q_sc), F32(q_zp), out_t, Rnd.kHalfAway)


def qdq_silu_exp2_a8(x, out_t, ct, dq_zp, dq_sc, q_zp, q_sc):
    y = _narrow(hw_silu_f32(_dq(x, dq_zp, dq_sc, ct)), ct)
    return quant(y, F32(q_sc), F32(q_zp), out_t, Rnd.kHalfAway)


B0 = F32(895.0)
H2 = [F32(v) for v in (1.3720913054337833, -0.3997208323665843)]
E3 = [F32(v) for v in (0.9999278783798218, 0.6957767605781555, 0.22623352706432343,
                       0.07790721952915192)]
G3 = [F32(v) for v in (1.4226349664583249, -0.6029645309400076, 0.18733279449559237)]
G3G = F32(0.9995263666916724)
LP = [F32(v) for v in (1.4426963329315186, -0.7213282585144043, 0.4806240499019623,
                       -0.3615472912788391, 0.2982918620109558, -0.24276652932167053,
                       0.11998527497053146)]


def _log2_bias(cg):
    """895 + log2(cg): exponent from the bits, mantissa reduced to [0.75, 1.5)."""
    b = int(f32_bits(cg))
    e = ((b >> 23) & 0xFF) - 127
    mb = (b & 0x7FFFFF) | 0x3F800000
    if b & 0x400000:
        mb, e = (b & 0x7FFFFF) | 0x3F000000, e + 1
    vx = add_q(bits_f32(mb), F32(-1))
    lp = LP[6]
    for k in range(5, -1, -1):
        lp = fma_q(vx, lp, LP[k])
    return add_q(F32(895 + e), mul_q(lp, vx))


def qdq_silu_fp32_a16(x, out_t, mode, dq_zp, dq_sc, q_zp, q_sc):
    """fp32 QDQ SiLU; dq_sc = s_in, q_sc = 1 / s_out (fp32)."""
    x = np.asarray(x).astype(np.int64)
    s_in, inv, zp_out = F32(dq_sc), F32(q_sc), F32(q_zp)
    if mode == 0:
        s_k = mul_q(s_in, inv)
        xk = mac_exact(x, s_k, mul_q(F32(dq_zp), neg_f32(s_k)))
        u = fma_q(div_q(kLog2e, inv), neg_f32(np.abs(xk)), B0)
        un = bits_f32(f32_bits(u) & np.uint32(0xFFFFC000))
        f = sub_q(u, un)
        p = fma_q(f, E3[3], E3[2])
        p = fma_q(f, p, E3[1])
        p = fma_q(f, p, E3[0])
        e = mul_q(hw_exp2_bf20(un), p)
        s0 = hw_exp2_bf20(fma_q(neg_f32(e), fma_q(H2[1], e, H2[0]), B0))
        s1 = fma_q(s0, fma_q(neg_f32(add_q(e, F32(1))), s0, F32(1)), s0)
        o = fma_q(vmax(xk, mul_q(xk, e)), s1, zp_out)
    else:
        c = mul_q(s_in, inv)
        cg = mul_q(c, G3G)
        r = div_q(1.0, cg)
        r2 = mul_q(r, r)
        cp = (mul_q(G3[0], r), mul_q(G3[1], r2), mul_q(G3[2], mul_q(r2, r)))
        qi = x - int(scalar_cast_i32(F32(dq_zp)))
        aq = np.abs(qi)
        arg = mac_exact(aq, mul_q(s_in, neg_f32(kLog2e)), _log2_bias(cg))
        o1 = mac_exact(np.maximum(qi, 0), c, zp_out)
        ep = hw_exp2_bf20(arg)
        h = fma_q(fma_q(cp[2], ep, cp[1]), ep, cp[0])
        sg = hw_exp2_bf20(fma_q(neg_f32(ep), h, arg))
        o = mac_exact(-aq, sg, o1)
    return float_to_int(o, out_t, 0, Rnd.kHalfAway)


# the 16-bit variants share the models
qdq_sigtanh_a16 = qdq_sigtanh_a8
qdq_silu_exp2_a16 = qdq_silu_exp2_a8
