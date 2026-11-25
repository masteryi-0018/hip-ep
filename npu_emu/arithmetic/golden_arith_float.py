# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the floating-point elementwise arithmetic (bf16 / f16 / fp8 / bfp16)."""

import numpy as np

from golden_bits import neg_f32
from golden_block_fp import encode_bfp16
from golden_float_cvt import (bf16_to_f32, code16, decode_float, f32_to_bf16, f32_to_fp8,
                              fp8_to_f32, load_float, round_float,
                              scalar_cast, store_float)
from golden_int_round import Rnd
from golden_mac import add_q, fma_q, mul_q, sub_q
from golden_mac import mul16 as mul16_codes
from golden_minmax import vclamp, vmax, vmin
from golden_mmac import decode_bfp16_identity, fp8_decode_identity, mmac_add_bfp16
from golden_act import sigtanh_tanh_bf16
from golden_nlf import hw_inv16, hw_inv_f32

BF16, F16 = "bf16", "f16"
F32 = np.float32

# activation codes of add / mul / mac_fused
NONE, RELU, RELU6, LRELU = 0, 1, 2, 3


def _inv16(y, t):
    """16-bit reciprocal of fp32-carried values: bf16 reads the upper 16 bits, f16 rounds half
    to even (see arith_float.hip)."""
    return decode_float(hw_inv16(code16(y, t), t), t)

def _activate(v, act, a0, a1):
    if act == RELU:
        return vmax(v, a0)
    if act == RELU6:
        return vclamp(v, a0, a1)
    if act == LRELU:
        return vmax(v, mul_q(v, a0))
    return np.asarray(v, F32)


def _bshape(bc, H, W, C):
    return (1 if bc & 4 else H, 1 if bc & 2 else W, 1 if bc & 1 else C)


def _expand(x, bc, H, W, C):
    """An operand of shape _bshape(bc) as a full H x W x C array."""
    return np.broadcast_to(np.asarray(x).reshape(_bshape(bc, H, W, C)), (H, W, C))


def _add16(x1, x2, bc, t, H, W, C, neg1, neg2, act, a0, a1):
    a = load_float(x1, t).reshape(H, W, C)
    b = _expand(load_float(x2, t), bc, H, W, C)
    v = add_q(neg_f32(a) if neg1 else a, neg_f32(b) if neg2 else b)
    return store_float(_activate(v, act, a0, a1), t).reshape(-1)


def _mul16(x1, x2, bc, t, H, W, C, act, a0, a1):
    a = load_float(x1, t).reshape(H, W, C)
    b = _expand(load_float(x2, t), bc, H, W, C)
    return store_float(_activate(mul_q(a, b), act, a0, a1), t).reshape(-1)


def _div16(x1, x2, bc, t, H, W, C, swap):
    a = load_float(x1, t).reshape(H, W, C)
    b = _expand(load_float(x2, t), bc, H, W, C)
    num, den = (b, a) if swap else (a, b)
    return store_float(mul_q(num, _inv16(den, t)), t).reshape(-1)


def _mac16(x1, x2, x3, bc, t, H, W, C, act, a0, a1):
    a = load_float(x1, t).reshape(H, W, C)
    b = _expand(load_float(x2, t), bc, H, W, C)
    c = _expand(load_float(x3, t), bc, H, W, C)
    return store_float(_activate(fma_q(a, b, c), act, a0, a1), t).reshape(-1)


def _bfp_values(blocks, shape):
    """Identity-multiply decode; each pixel's channels form 64-element groups from channel 0."""
    return decode_bfp16_identity(blocks, shape[-1] // 8).reshape(shape)


def _bfp_operand(x2, bc, H, W, C):
    """A broadcast bfp16 operand as H x W x C values. C-broadcast: element 0 of the pixel's
    block, decoded in the group of the 8 operand blocks starting at it."""
    s = _bshape(bc, H, W, C)
    if bc & 1:
        blk = np.asarray(x2, np.uint8).reshape(-1, 9)
        n = len(blk)
        pad = np.concatenate([blk, np.zeros((7, 9), np.uint8)])  # exponent 0: no effect
        win = pad[np.arange(n)[:, None] + np.arange(8)]
        v = decode_bfp16_identity(win, 8).reshape(n, 64)[:, :1].reshape(s[0], s[1], 1)
    else:
        v = _bfp_values(x2, s)
    return np.broadcast_to(v, (H, W, C))


def _bfp_bin(op, x1, x2, bc, H, W, C, swap=0):
    a = _bfp_values(x1, (H, W, C))
    b = _bfp_operand(x2, bc, H, W, C)
    if op == "add":
        v = mmac_add_bfp16(x1, a, b, C).reshape(H, W, C)
    elif op == "sub" and not swap and bc & 1:
        v = mmac_add_bfp16(x1, a, neg_f32(b), C).reshape(H, W, C)
    elif op == "sub":
        v = sub_q(b, a) if swap else sub_q(a, b)
    elif op == "mul":
        v = mul_q(a, b)
    else:
        num, den = (b, a) if swap else (a, b)
        v = mul_q(num, _inv16(den, BF16))
    return encode_bfp16(v.reshape(-1), Rnd.kHalfEven)


def _mul_add(x1, x2, x2_bfp, bc2, x3, x3_bfp, bc3, out_bfp, H, W, C, act_type, activation,
            lo, hi):
    a = _bfp_values(x1, (H, W, C))

    def operand(x, bfp, bc):
        s = _bshape(bc, H, W, C)
        v = _bfp_values(x, s) if bfp else bf16_to_f32(x).reshape(s)
        return np.broadcast_to(v, (H, W, C))

    v = fma_q(a, operand(x2, x2_bfp, bc2), operand(x3, x3_bfp, bc3))
    if act_type == 100 and activation == 100:
        v = vclamp(round_float(v, BF16), lo, hi)
    elif act_type == 7:
        v = round_float(v, BF16)
        if activation == 7:
            v = sigtanh_tanh_bf16(v)
    v = v.reshape(-1)
    return encode_bfp16(v, Rnd.kHalfEven) if out_bfp else f32_to_bf16(v)


def _add_fp8(x1, x2, bc, H, W, C, c1, c2, act, a0, a1, table=None):
    a = fp8_decode_identity(np.reshape(x1, (H, W, C)), row=C)
    if not bc & 1:
        s = _bshape(bc, H, W, C)
        b = np.broadcast_to(fp8_decode_identity(np.reshape(x2, s), row=s[-1]), (H, W, C))
        v = _activate(fma_q(b, c2, mul_q(a, c1)), act, a0, a1)
    else:
        codes = _expand(np.asarray(x2, np.uint8), bc, H, W, C)
        b = fp8_to_f32(codes)
        if act == RELU:
            v = vmax(add_q(mul_q(a, c1), mul_q(b, c2)), a0)
        elif act == RELU6:
            v = vclamp(add_q(mul_q(a, c1), F32(table)[codes]), a0, a1)
        elif act == LRELU:
            tb = np.asarray(table, F32)
            q = fma_q(a, tb[256], tb[codes])
            v = fma_q(np.abs(q), tb[257], q)
        else:
            v = fma_q(a, c1, mul_q(b, c2))
    return f32_to_fp8(v.reshape(-1), Rnd.kHalfAway)


def _mul_fp8(x1, x2, bc, H, W, C, s):
    if not bc & 1:
        a = fp8_to_f32(np.asarray(x1, np.uint8)).reshape(H, W, C)
        b = fp8_to_f32(_expand(np.asarray(x2, np.uint8), bc, H, W, C))
        v = mul_q(mul_q(a, b), s)
    else:
        a = fp8_decode_identity(np.reshape(x1, (H, W, C)), row=C)
        b = _expand(add_q(fp8_to_f32(x2), F32(0.0)), bc, H, W, C)
        v = mul_q(a, mul_q(b, s))
    return f32_to_fp8(v.reshape(-1), Rnd.kHalfAway)


def _div_fp8(x1, x2, bc, H, W, C, s, swap):
    a = fp8_decode_identity(np.reshape(x1, (H, W, C)), row=C)
    s2 = _bshape(bc, H, W, C)
    b = np.broadcast_to(fp8_decode_identity(np.reshape(x2, s2), row=s2[-1]), (H, W, C))
    num, den = (b, a) if swap else (a, b)
    return f32_to_fp8(mul_q(mul_q(num, s), hw_inv_f32(den)).reshape(-1), Rnd.kHalfAway)


def _mac_fp8(x1, x2, x3, bc, H, W, C, c0, c4, act, a0, a1):
    a = fp8_decode_identity(np.reshape(x1, (H, W, C)), row=C)
    s = _bshape(bc, H, W, C)
    if bc & 1:
        b = _expand(add_q(fp8_to_f32(x2), F32(0.0)), bc, H, W, C)
        d = _expand(add_q(fp8_to_f32(x3), F32(0.0)), bc, H, W, C)
        v = fma_q(a, mul_q(b, c0), mul_q(d, c4))
    else:
        b = np.broadcast_to(fp8_decode_identity(np.reshape(x2, s), row=s[-1]), (H, W, C))
        d = np.broadcast_to(fp8_decode_identity(np.reshape(x3, s), row=s[-1]), (H, W, C))
        v = fma_q(d, c4, mul_q(b, mul_q(a, c0)))
    return f32_to_fp8(_activate(v, act, a0, a1).reshape(-1), Rnd.kHalfEven)


def _leakyrelu16(x, t, alpha):
    v = load_float(x, t)
    return store_float(_activate(v, LRELU, scalar_cast(F32(alpha), t), 0.0), t)


def _leakyrelu_fp8(x, pos, neg):
    v = fp8_decode_identity(x)
    return f32_to_fp8(vmax(mul_q(v, pos), mul_q(v, neg)), Rnd.kHalfAway)


def _prelu16(x, t, D, C):
    v = load_float(x, t)
    d = np.resize(F32(D), v.size)
    return store_float(fma_q(vmin(v, F32(0.0)), d, v), t)


def _prelu_fp8(x, P, D, C):
    v = fp8_decode_identity(x)
    d = np.resize(F32(D), v.size)
    return f32_to_fp8(fma_q(vmin(v, F32(0.0)), d, mul_q(v, P)), Rnd.kHalfAway)


def _square16(x, t):
    return mul16_codes(x, x, t)


def _square_fp8(x, s):
    v = fp8_decode_identity(x)
    return f32_to_fp8(mul_q(v, mul_q(v, s)), Rnd.kHalfAway)


def _batchnorm(x, t, C, cw, cb, act, alpha):
    """t in BF16 / F16 / 'fp8'; cw / cb fp32 values."""
    if t == "fp8":
        v = fp8_decode_identity(x)
    else:
        v = load_float(x, t)
    n = v.size
    acc = fma_q(v, np.resize(F32(cw), n), np.resize(F32(cb), n))
    if act == LRELU:
        acc = _activate(acc, LRELU, alpha, 0.0)
    elif act == RELU and t == "fp8":
        acc = _activate(acc, RELU, F32(0.0), 0.0)
    if t == "fp8":
        return f32_to_fp8(acc, Rnd.kHalfEven)
    if act == RELU:
        acc = vmax(round_float(acc, t), F32(0.0))
    return store_float(acc, t)


# ---- the entry points (named as the entries without the npu_ prefix; arguments in the entry's
# order without the stream and the output buffers; shapes as (H, W, C)) ----

def _bc(shape1, shape2):
    """Broadcast bits of shape2 in shape1: bit 0 / 1 / 2 = size 1 along C / W / H."""
    return sum(1 << (2 - d) for d in range(3) if shape2[d] == 1)


def elementwise_broadcasting3d_add_bf16(x1, type, x2, shape1, shape2, is_sub_flag, negate_flag,
                                        activation, activation_coeff0, activation_coeff1):
    return _add16(x1, x2, _bc(shape1, shape2), type, *shape1, negate_flag, is_sub_flag,
                  activation, activation_coeff0, activation_coeff1)


def elementwise_broadcasting3d_mul_bf16(x1, type, x2, shape1, shape2, activation,
                                        activation_coeff0, activation_coeff1):
    return _mul16(x1, x2, _bc(shape1, shape2), type, *shape1, activation, activation_coeff0,
                  activation_coeff1)


def elementwise_broadcasting3d_div_bf16(x1, type, x2, shape1, shape2, swap_inputs):
    return _div16(x1, x2, _bc(shape1, shape2), type, *shape1, swap_inputs)


def elementwise_broadcasting3d_mac_fused_bf16(x1, type, x2, x3, shape1, shape23, activation,
                                              activation_coeff0, activation_coeff1):
    return _mac16(x1, x2, x3, _bc(shape1, shape23), type, *shape1, activation,
                  activation_coeff0, activation_coeff1)


def elementwise_broadcasting3d_add_bfp16(x1, x2, shape1, shape2):
    return _bfp_bin("add", x1, x2, _bc(shape1, shape2), *shape1)


def elementwise_broadcasting3d_sub_bfp16(x1, x2, shape1, shape2, swap_inputs):
    return _bfp_bin("sub", x1, x2, _bc(shape1, shape2), *shape1, swap_inputs)


def elementwise_broadcasting3d_mul_bfp16(x1, x2, shape1, shape2):
    return _bfp_bin("mul", x1, x2, _bc(shape1, shape2), *shape1)


def elementwise_broadcasting3d_div_bfp16(x1, x2, shape1, shape2, swap_inputs):
    return _bfp_bin("div", x1, x2, _bc(shape1, shape2), *shape1, swap_inputs)


def elementwise_broadcasting3d_mul_add_bfp16(x1, x2, x2_type, x3, x3_type, out_type, shape1,
                                             shape2, shape3, activation_type, activation,
                                             clip_min, clip_max):
    """Types 'bf16' / 'bfp16'; clip_min / clip_max bf16 bit patterns. Output as bytes."""
    lo, hi = (bf16_to_f32(np.uint16(b)) for b in (clip_min, clip_max))
    out = _mul_add(x1, x2, x2_type == "bfp16", _bc(shape1, shape2), x3, x3_type == "bfp16",
                   _bc(shape1, shape3), out_type == "bfp16", *shape1, activation_type,
                   activation, lo, hi)
    return np.asarray(out).view(np.uint8)


def elementwise_broadcasting3d_add_fp8(x1, x2, shape1, shape2, ifm1_coeff, ifm2_coeff,
                                       activation, activation_coeff0, activation_coeff1,
                                       table):
    return _add_fp8(x1, x2, _bc(shape1, shape2), *shape1, ifm1_coeff, ifm2_coeff, activation,
                    activation_coeff0, activation_coeff1, table)


def elementwise_broadcasting3d_mul_fp8(x1, x2, shape1, shape2, operation_coeff):
    return _mul_fp8(x1, x2, _bc(shape1, shape2), *shape1, operation_coeff)


def elementwise_broadcasting3d_div_fp8(x1, x2, shape1, shape2, operation_coeff, swap_inputs):
    return _div_fp8(x1, x2, _bc(shape1, shape2), *shape1, operation_coeff, swap_inputs)


def elementwise_broadcasting3d_mac_fused_fp8(x1, x2, x3, shape1, shape23, c0, c4, activation,
                                             activation_coeff0, activation_coeff1):
    return _mac_fp8(x1, x2, x3, _bc(shape1, shape23), *shape1, c0, c4, activation,
                    activation_coeff0, activation_coeff1)


def elementwise_leakyrelu_bf16(x, type, n, alpha):
    return _leakyrelu16(np.asarray(x)[:n], type, alpha)


def elementwise_leakyrelu_fp8(x, n, positive_coeff, negative_coeff):
    return _leakyrelu_fp8(np.asarray(x)[:n], positive_coeff, negative_coeff)


def elementwise_prelu_bf16(x, type, n, C, prelu_slope_delta):
    return _prelu16(np.asarray(x)[:n], type, prelu_slope_delta, C)


def elementwise_prelu_fp8(x, n, C, positive_scale, prelu_scaled_slope_delta):
    return _prelu_fp8(np.asarray(x)[:n], positive_scale, prelu_scaled_slope_delta, C)


def elementwise_square_bf16(x, type, n):
    return _square16(np.asarray(x)[:n], type)


def elementwise_square_fp8(x, n, ifm_coeff):
    return _square_fp8(np.asarray(x)[:n], ifm_coeff)


def batchnorm2d_stream_bf16(x, type, n, C, cw, cb, activation, relu_alpha):
    """type BF16 / F16 / 'fp8'; cw / cb 16-bit codes, bf16 for BF16, else f16."""
    wt = BF16 if type == BF16 else F16
    return _batchnorm(np.asarray(x)[:n], type, C, load_float(cw, wt), load_float(cb, wt),
                      activation, relu_alpha)
