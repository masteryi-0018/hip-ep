# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference model for convert.h."""

import numpy as np

from golden_block_fp import decode_bfp16_bf16, encode_bfp16
from golden_float_cvt import (bf16_to_f32, f32_to_fp8, fp8_decode_identity, kSat,
                              load_float, round_float, scalar_cast, store_float)
from golden_int_round import Rnd, float_to_int, requant_tail, shift_round
from golden_mac import fma, mac_exact, mul as fmul


def zp_int(zp):
    """int32 of a zero point: truncated toward zero, saturated, NaN -> 0 [default]."""
    return int(float_to_int(np.float32(zp), "i32", 0, Rnd.kTrunc))


def quant(x, scale, zp, out_t):
    return float_to_int(fma(x, scale, zp), out_t)


def dequant(x, scale, zp, out_t):
    c = fmul(-np.float32(zp), np.float32(scale))
    return store_float(mac_exact(x, scale, c), out_t)


def requantize(x, in_t, c1, c0, out_t):
    x = np.asarray(x, np.int64)
    if in_t == "u32":
        raw = x.astype(np.uint32).view(np.int32).astype(np.int64)
        r = mac_exact(raw, c1, c0)
        corr = -fmul(np.float32(c1), np.float32(4294967296.0))
        v = mac_exact(raw >> 31, corr, r)
    else:
        v = mac_exact(x, c1, c0)
    return requant_tail(v, out_t)


def requantize_cw(x, M, B, out_t):
    x = np.asarray(x, np.int64)
    c = np.arange(len(x)) % len(M)
    return requant_tail(mac_exact(x, np.asarray(M)[c], np.asarray(B)[c]), out_t)


def a64_down(x, in_t, co, out_t):
    u = np.asarray(x).astype(np.uint64)
    lo = (u & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32).astype(np.int64)
    hi = (u >> np.uint64(32)).astype(np.uint32).view(np.int32).astype(np.int64)
    acc = mac_exact(hi, co["mult_f_x2p32"], co["bias_f"])
    acc = mac_exact(lo, co["mult_f"], acc)
    if in_t == "i64":
        lo_msb = np.where(lo < 0, -2**31, 0)
        c = fmul(-np.float32(co["mult_f_x2p32"]), np.float32(2.0**-31))
        acc = mac_exact(lo_msb, c, acc)
    else:
        acc = mac_exact(np.where(lo < 0, -1, 0), -np.float32(co["mult_f_x2p32"]), acc)
        acc = mac_exact(np.where(hi < 0, -1, 0), -np.float32(co["mult_f_x2p64"]), acc)
    return float_to_int(acc, out_t)


def a64_up(x, in_t, co, out_t):
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


def int32_int64(x, c1, c0):
    return float_to_int(mac_exact(x, c1, c0), "i32")


def fp32_fp8(x, scale_recip):
    return f32_to_fp8(fmul(np.asarray(x, np.float32), np.float32(scale_recip)), Rnd.kHalfAway)


def fp16_fp8(bits, in_t, scale):
    with np.errstate(all="ignore"):
        r = scalar_cast(np.float32(1.0) / np.float32(scale), in_t)
    return f32_to_fp8(fmul(load_float(bits, in_t), np.float32(r)), flags=kSat)


def requantize_fp8(c, coeff):
    return f32_to_fp8(fmul(fp8_decode_identity(c), np.float32(coeff)), flags=kSat)


def fp8_to(c, scale, out_t):
    v = fmul(fp8_decode_identity(c), np.float32(scale))
    return store_float(v, out_t, Rnd.kHalfAway if out_t != "f32" else Rnd.kHalfEven)


def bfp16_to_unsigned(blk, mul, zp, out_t):
    return float_to_int(fma(bf16_to_f32(decode_bfp16_bf16(blk)), mul, zp), out_t)


def aq_to_bfp16(x, scale, zp):
    d = (np.asarray(x, np.int64) - zp_int(zp)).astype(np.float32)
    return encode_bfp16(round_float(fmul(d, np.float32(scale)), "bf16"), Rnd.kHalfEven)


def w4_to_bfp16(q, zp, scale_bits, N, K):
    q = np.asarray(q, np.int64).reshape(N, K)
    g = np.repeat(np.arange(K // 32), 32)
    zpk = np.asarray(zp, np.int64).reshape(N, K // 32)[:, g]
    s = bf16_to_f32(np.asarray(scale_bits).reshape(N, K // 32))[:, g]
    return encode_bfp16(fmul((q - zpk).astype(np.float32), s).reshape(-1), Rnd.kHalfEven)


def bfp16_qdq_f32(blk, mul, zp, scale):
    q = bfp16_to_unsigned(blk, mul, zp, "u16")
    return fmul((q - zp_int(zp)).astype(np.float32), np.float32(scale))
