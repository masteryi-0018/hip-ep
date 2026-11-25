# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference model for convert.h: one function per entry point (without the npu_
prefix), taking the entry's arguments without pointers, sizes and output buffers."""

import numpy as np

import golden_quant as gq
from golden_bits import neg_f32
from golden_block_fp import decode_mx6, encode_bfp16, encode_mx6
from golden_float_cvt import (bf16_to_f32, f32_to_bf16, float_to_int, load_float, round_float,
                              scalar_cast, scalar_cast_i32,
                              store_float)
from golden_int_round import Rnd, shift_round
from golden_mac import div_q, mac_exact, mul_q, negmul_q
from golden_mmac import decode_bfp16_bf16, fp8_decode_identity


def convert_fp32_fp16(x, out_t):
    return store_float(np.asarray(x, np.float32), out_t)


def convert_fp16_fp32(bits, in_t):
    return load_float(bits, in_t)


def convert_fp16_bf16(bits, in_t, out_t):
    return store_float(load_float(bits, in_t), out_t)


def cast_bf2bfp16(bits):
    return encode_bfp16(bf16_to_f32(bits), Rnd.kHalfEven)


def convert_bf16_to_bfp16(bits, in_t):
    return encode_bfp16(load_float(bits, in_t), Rnd.kHalfAway)


def cast_bfp16tobf(blk):
    return decode_bfp16_bf16(blk)


def cast_bftomx6(bits):
    return encode_mx6(bf16_to_f32(bits), Rnd.kHalfAway)


def cast_mx6tobf(blk):
    return f32_to_bf16(decode_mx6(blk))


def convert_fp32_fp8(x, scale_recip):
    return gq.fp8_out(np.asarray(x, np.float32), np.float32(scale_recip), Rnd.kHalfAway, 0)


def convert_fp16_fp8(bits, in_t, scale):
    r = scalar_cast(div_q(1.0, scale), in_t)
    return gq.fp8_out(load_float(bits, in_t), np.float32(r), Rnd.kHalfEven)


def _fp8_to(c, scale, out_t):
    rnd = Rnd.kHalfAway if out_t != "f32" else Rnd.kHalfEven
    return store_float(gq.fp8_in(c, np.float32(scale), out_t, rnd), out_t, rnd)


def convert_fp8_fp32(c, scale):
    return _fp8_to(c, scale, "f32")


def convert_fp8_fp16(c, out_t, scale):
    return _fp8_to(c, scale, out_t)


def requantize_fp8(c, ifm_coeff):
    return gq.fp8_out(fp8_decode_identity(c), np.float32(ifm_coeff), Rnd.kHalfEven)


def quant(bits, in_t, out_t, zero_point, scale):
    return gq.quant(load_float(bits, in_t), scale, zero_point, out_t)


def _bfp16_to_unsigned(blk, mul, zp, out_t):
    return gq.quant(bf16_to_f32(decode_bfp16_bf16(blk)), mul, zp, out_t)


def cast_bfp16toa8(blk, quant_multiplier, zero_point):
    return _bfp16_to_unsigned(blk, quant_multiplier, zero_point, "u8")


def cast_bfp16toa16(blk, quant_multiplier, zero_point):
    return _bfp16_to_unsigned(blk, quant_multiplier, zero_point, "u16")


def dequant(x, in_t, out_t, zero_point, scale):
    return store_float(gq.dequant(x, scale, gq.dq_offset(zero_point, scale)), out_t)


def _aq_to_bfp16(x, scale, zp):
    d = (np.asarray(x, np.int64) - int(scalar_cast_i32(zp))).astype(np.float32)
    return encode_bfp16(round_float(mul_q(d, np.float32(scale)), "bf16"), Rnd.kHalfEven)


def cast_a8tobfp16(x, scale, zero_point):
    return _aq_to_bfp16(x, scale, zero_point)


def cast_a16tobfp16(x, scale, zero_point):
    return _aq_to_bfp16(x, scale, zero_point)


def dequant_w4_bfp16(zp, scale_bits, q, N, K):
    q = np.asarray(q, np.int64).reshape(N, K)
    g = np.repeat(np.arange(K // 32), 32)
    zpk = np.asarray(zp, np.int64).reshape(N, K // 32)[:, g]
    s = bf16_to_f32(np.asarray(scale_bits).reshape(N, K // 32))[:, g]
    return encode_bfp16(mul_q((q - zpk).astype(np.float32), s).reshape(-1), Rnd.kHalfEven)


def requantize(x, in_t, out_t, c1, c0):
    x = np.asarray(x, np.int64)
    if in_t == "u32":
        raw = x.astype(np.uint32).view(np.int32).astype(np.int64)
        r = mac_exact(raw, c1, c0)
        corr = negmul_q(np.float32(c1), np.float32(4294967296.0))
        v = mac_exact(raw >> 31, corr, r)
    else:
        v = mac_exact(x, c1, c0)
    return gq.requant_tail(v, out_t)


def _requantize_cw(x, out_t, channels, M, B):
    x = np.asarray(x, np.int64)
    c = np.arange(len(x)) % channels
    return gq.requant_tail(mac_exact(x, np.asarray(M)[c], np.asarray(B)[c]), out_t)


def requantize_a8_channelwise(x, in_t, out_t, channels, M, B):
    return _requantize_cw(x, out_t, channels, M, B)


def requantize_a16_channelwise(x, in_t, out_t, channels, M, B):
    return _requantize_cw(x, out_t, channels, M, B)


def _a64_down(x, in_t, out_t, co):
    u = np.asarray(x).astype(np.uint64)
    lo = (u & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32).astype(np.int64)
    hi = (u >> np.uint64(32)).astype(np.uint32).view(np.int32).astype(np.int64)
    acc = mac_exact(hi, co["mult_f_x2p32"], co["bias_f"])
    acc = mac_exact(lo, co["mult_f"], acc)
    if in_t == "i64":
        lo_msb = np.where(lo < 0, -2**31, 0)
        c = mul_q(neg_f32(np.float32(co["mult_f_x2p32"])), np.float32(2.0**-31))
        acc = mac_exact(lo_msb, c, acc)
    else:
        acc = mac_exact(np.where(lo < 0, -1, 0), neg_f32(np.float32(co["mult_f_x2p32"])), acc)
        acc = mac_exact(np.where(hi < 0, -1, 0), neg_f32(np.float32(co["mult_f_x2p64"])), acc)
    return float_to_int(acc, out_t)


def _a64_up(x, in_t, out_t, co):
    res = []
    for v in np.asarray(x, np.int64).tolist():
        if in_t in ("u8", "u16", "u32"):
            prod = (v & 0xFFFFFFFF) * (co["M"] & 0xFFFFFFFF)
        else:
            prod = v * co["M"]
        acc = (co["B"] + prod) & (2**64 - 1)
        acc = acc - 2**64 if acc >= 2**63 else acc
        r = int(shift_round(acc, co["s"], Rnd.kHalfEven)) if co["s"] > 0 else acc
        if out_t == "u64" and r < 0:
            r = 0
        res.append(r)
    return np.array(res, np.int64)


def requantize_a64(x, in_t, out_t, co):
    """co: dict of the npu_a64_coeff fields."""
    if in_t in ("i64", "u64"):
        return _a64_down(x, in_t, out_t, co)
    return _a64_up(x, in_t, out_t, co)


def cast_int32_int64(x, in_t, out_t, c1, c0):
    return gq.requant(x, c1, c0, "i32")


def cast_bfp16toqdqf32(blk, quant_multiplier, zero_point, scale):
    q = _bfp16_to_unsigned(blk, quant_multiplier, zero_point, "u16")
    return mul_q((q - int(scalar_cast_i32(zero_point))).astype(np.float32), np.float32(scale))
