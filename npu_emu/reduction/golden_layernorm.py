# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for reduction/layernorm.h; x is a (rows, C) fp32 array."""

import numpy as np

from golden_block_fp import decode_bfp16_bf16, encode_bfp16
from golden_float_cvt import (bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, f32_to_fp8,
                              fp8_decode_identity, load_float, round_float, store_float)
from golden_int_round import INT_RANGE, Rnd, float_to_int, shift_round
from golden_mac import add, fma, mac_exact, mul, vaddr_f32_reduce
from golden_minmax import vclamp, vmax, vmin
from golden_nlf import hw_inv_f32, hw_invsqrt_f32

f32 = np.float32
LANES = 32


def decode(bits, t):
    """Stored values of type t as fp32 (fp8: groups of 8, -0 -> +0; bfp16: the bfp16 -> bf16
    cast, 64-element groups from the buffer start)."""
    if t == "fp8":
        shape = np.shape(bits)
        return fp8_decode_identity(np.asarray(bits, np.uint8).reshape(-1)).reshape(shape)
    if t == "bfp16":
        return bf16_to_f32(decode_bfp16_bf16(bits))
    return load_float(bits, t)


def encode(x, t):
    """fp32 -> stored bits of type t, half to even (bfp16 via bf16 first)."""
    x = np.asarray(x, f32)
    if t == "fp8":
        return f32_to_fp8(x)
    if t == "bfp16":
        return encode_bfp16(round_float(x, "bf16").reshape(-1), Rnd.kHalfEven)
    return store_float(x, t)


def lane_steps(x, lanes=LANES):
    """(rows, C) -> list of (rows, lanes) steps, channel c in lane c % lanes,
    zero padded."""
    x = np.asarray(x, f32)
    rows, C = x.shape
    pad = (-C) % lanes
    if pad:
        x = np.concatenate([x, np.zeros((rows, pad), f32)], axis=1)
    return [x[:, j:j + lanes] for j in range(0, x.shape[1], lanes)]


def inv_c(C):
    return f32(hw_inv_f32(np.array([C], f32))[0])


def quantize(y, q_scale, q_zp, t):
    return float_to_int(fma(y, f32(q_scale), f32(q_zp)), t)


def dequant(q, scale, zp, compute):
    """fma(q, scale, fp32(-(zp * scale))) rounded to the compute type."""
    off = -mul(f32(zp), f32(scale))
    return round_float(mac_exact(np.asarray(q, np.int64), f32(scale), off), compute)


def core_layernorm(x, alpha, beta, ik_type, out_round, c_real=0, eps=1e-12):
    """x (rows, C) fp32; returns t1 rounded to out_round ('f32' = not rounded).
    c_real > 0: statistics over channels [0, c_real) only, divisor c_real; the
    other channels read as zero there and are normalized like the rest."""
    rows, C = x.shape
    c_real = c_real or C
    s = inv_c(c_real)
    xs = np.where(np.arange(C) < c_real, x, f32(0)).astype(f32)
    ax = np.zeros((rows, LANES), f32)
    ax2 = np.zeros((rows, LANES), f32)
    for v in lane_steps(xs):
        ax = fma(v, s, ax)
        ax2 = fma(v, v, ax2)
    mu = vaddr_f32_reduce(ax)
    var = vaddr_f32_reduce(mul(ax2, s), np.full(rows, f32(eps)))
    var = fma(-mu, mu, var)
    istd = hw_invsqrt_f32(var)
    mis = -mul(istd, mu)
    ik = round_float(istd, ik_type)
    t0 = fma(x, ik[:, None], mis[:, None])
    t1 = fma(t0, np.broadcast_to(alpha, x.shape), np.broadcast_to(beta, x.shape))
    return round_float(t1, out_round)


def rmsnorm(x, gamma, inv_inner, eps):
    acc = np.zeros((x.shape[0], LANES), f32)
    for v in lane_steps(x):
        acc = fma(v, v, acc)
    s = vaddr_f32_reduce(acc)
    r = hw_invsqrt_f32(add(mul(s, f32(inv_inner)), f32(eps)))
    return mul(mul(x, r[:, None]), gamma)


def l2norm(x, alpha, beta):
    acc = np.zeros((x.shape[0], LANES), f32)
    for v in lane_steps(x):
        acc = fma(v, v, acc)
    r = hw_invsqrt_f32(vaddr_f32_reduce(acc, np.full(x.shape[0], f32(1e-12))))
    return fma(mul(alpha, x), np.broadcast_to(r[:, None], x.shape),
                   np.broadcast_to(beta, x.shape))


def qdq_l2norm(q, dq_scale, dq_zp, alpha, beta, q_scale, q_zp, out_t):
    x = dequant(q, dq_scale, dq_zp, "bf16")
    acc = np.zeros((x.shape[0], LANES), f32)
    for v in lane_steps(x):
        acc = fma(v, v, acc)
    r = vmin(hw_invsqrt_f32(vaddr_f32_reduce(acc)), f32(1e12))
    if alpha is None:
        y = mul(x, r[:, None])
    else:
        y = fma(x, mul(alpha[None, :], r[:, None]), np.broadcast_to(beta, x.shape))
    return quantize(round_float(y, "bf16"), q_scale, q_zp, out_t)


def fast_rsqrt(v):
    """Reciprocal square root: bf16 seed, two Newton steps."""
    v = vmax(v, f32(1e-5))
    half = f32_to_bf16(v).astype(np.int64) >> 1
    y = bf16_to_f32((0x5F37 - half).astype(np.uint16))
    h = mul(v, f32(0.5))
    for _ in range(2):
        y = mul(y, fma(-h, mul(y, y), f32(1.5)))
    return y


def ln_stats(x, eps, stats, ik_type):
    """(ik, mis) per row. stats: 'fused' / 'batched' / 'batched_fast' /
    'c8_fused' / 'c8_batched_fast'."""
    rows, C = x.shape
    s = inv_c(C)
    if stats.startswith("c8"):
        steps = lane_steps(x, 8)
        sx = steps[0].copy()
        sq = mul(steps[0], steps[0])
        rest = steps[1:]
        if stats == "c8_fused" and rest:
            sx = add(sx, rest[0])
            sq = add(sq, mul(rest[0], rest[0]))
            rest = rest[1:]
        for v in rest:
            sx = add(sx, v)
            sq = fma(v, v, sq)

        def fold(v):
            a = add(v[:, :4], v[:, 4:])
            b = add(a[:, :2], a[:, 2:])
            return add(b[:, 0], b[:, 1])
        mu = mul(fold(sx), s)
        va = add(mul(fold(sq), s), f32(eps))
        var = fma(-mu, mu, va)
        istd = hw_invsqrt_f32(var) if stats == "c8_fused" else fast_rsqrt(var)
        return round_float(istd, ik_type), mul(-mu, istd)
    sx = np.zeros((rows, LANES), f32)
    sq = np.zeros((rows, LANES), f32)
    for v in lane_steps(x):
        sx = add(sx, v)
        sq = fma(v, v, sq)
    if stats == "fused":
        e = np.zeros(LANES, f32)
        e[16] = f32(eps)
        mu = vaddr_f32_reduce(mul(sx, s))
        var = fma(-mu, mu, vaddr_f32_reduce(fma(sq, s, np.broadcast_to(e, sq.shape))))
        istd = hw_invsqrt_f32(var)
        mis = -mul(mu, istd)
    else:
        mu = mul(vaddr_f32_reduce(sx), s)
        var = fma(-mu, mu, fma(vaddr_f32_reduce(sq), s, f32(eps)))
        istd = hw_invsqrt_f32(var) if stats == "batched" else fast_rsqrt(var)
        mis = -mul(istd, mu)
    return round_float(istd, ik_type), mis


def layernorm(x, alpha, beta, eps, stats, ik_type, out_round):
    ik, mis = ln_stats(x, eps, stats, ik_type)
    t0 = fma(x, np.broadcast_to(ik[:, None], x.shape), np.broadcast_to(mis[:, None], x.shape))
    t1 = fma(t0, np.broadcast_to(alpha, x.shape), np.broadcast_to(beta, x.shape))
    return round_float(t1, out_round)


def add_layernorm(a_bits, b_bits, dtype, alpha_bits, beta_bits, eps, stats):
    """Layer norm of a + b -> output bits."""
    wt = "f16" if dtype == "fp8" else dtype
    alpha, beta = decode(alpha_bits, wt), decode(beta_bits, wt)
    a, b = decode(a_bits, dtype), decode(b_bits, dtype)
    if dtype == "fp8":
        x = round_float(add(a, b), "f16")
        return encode(layernorm(x, alpha, beta, eps, stats, "f16", "f32"), "fp8")
    x = round_float(add(a, b), dtype)
    return encode(layernorm(x, alpha, beta, eps, stats, dtype, "f32"), dtype)


def qdq_layernorm(q, compute, alpha, beta, dq_scale, dq_zp, out_t, variant):
    """variant: 'fused' / 'batched' / 'c8_fused' / 'c8_batched' / 'integer'."""
    if variant == "integer":
        return qln_integer(q, alpha, beta, dq_scale, out_t)
    x = dequant(q, dq_scale, dq_zp, compute)
    stats = {"fused": "fused", "batched": "batched_fast", "c8_fused": "c8_fused",
             "c8_batched": "c8_batched_fast"}[variant]
    return float_to_int(layernorm(x, alpha, beta, 1e-5, stats, compute, "f32"), out_t)


def rsqrt16(v):
    """Reciprocal square root: f16 seed, two Newton steps."""
    v = np.asarray(v, f32)
    vs = vclamp(v, f32(6.2e-5), f32(65000.0))
    b = f32_to_f16(vs).astype(np.int64)
    bh = shift_round(b, 1, Rnd.kHalfEven)  # half to even
    y = f16_to_f32((0x59BB - bh).astype(np.uint16))
    h = mul(v, f32(0.5))
    for _ in range(2):
        y = mul(y, fma(-h, mul(y, y), f32(1.5)))
    return y


def qln_integer(q, alpha, beta, dq_scale, out_t):
    q = np.asarray(q, np.int64)
    rows, C = q.shape
    S = q.sum(axis=1)
    Q = (q * q).sum(axis=1)
    s = inv_c(C)
    eps2 = mul(f32(1e-5), hw_inv_f32(np.array([mul(f32(dq_scale), f32(dq_scale))], f32))[0])
    mu = mac_exact(S, s, f32(0))
    e2 = mac_exact(Q, s, eps2)
    var = vmax(fma(-mu, mu, e2), eps2)
    r = round_float(rsqrt16(var), "f16")
    c = -mul(mac_exact(S, s, f32(1536)), r)
    x = (q + 1536).astype(f32)
    t0 = fma(x, np.broadcast_to(r[:, None], x.shape), np.broadcast_to(c[:, None], x.shape))
    t1 = fma(t0, np.broadcast_to(alpha, x.shape), np.broadcast_to(beta, x.shape))
    return float_to_int(t1, out_t)


def qdq_ln_pre(mode, compute, a, b=None, k=None, a_t=None, da=None, db=None):
    """Layer-norm input (cd = rounded to the compute type): 'add': cd(cd(dq(a)) + cd(dq(b)));
    'const': cd(cd(dq(a)) + b); 'addc': cd(A + fp32(dq_b(b) + k)), A = cd(dq(a)) or a itself
    for a float a_t."""
    if mode == "add":
        return round_float(add(dequant(a, *da, compute), dequant(b, *db, compute)), compute)
    if mode == "const":
        return round_float(add(dequant(a, *da, compute), b), compute)
    A = dequant(a, *da, compute) if a_t in INT_RANGE else np.asarray(a, f32)
    dqb = dequant(b, *db, "f32")
    return round_float(add(A, add(dqb, np.broadcast_to(k, dqb.shape))), compute)
