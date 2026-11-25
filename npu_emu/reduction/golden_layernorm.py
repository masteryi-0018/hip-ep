# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for reduction/layernorm.h; x is a (rows, C) fp32 array."""

import numpy as np

from golden_bits import f32_bits, neg_f32
from golden_block_fp import encode_bfp16
from golden_float_cvt import (bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, f32_to_fp8,
                              float_to_int, load_float, round_float, store_float)
from golden_int_round import INT_RANGE, Rnd, shift_round
from golden_mac import add_q, fma_q, mac_exact, msc_q, mul_q, negmul_q, vaddr_f32_reduce
from golden_minmax import vclamp, vmax, vmin
from golden_mmac import decode_bfp16_bf16, fp8_decode_identity
from golden_nlf import hw_inv_f32, hw_invsqrt_f32
from golden_quant import dequant, quant

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


def _lane_steps(x, lanes=LANES):
    """(rows, C) -> list of (rows, lanes) steps, channel c in lane c % lanes,
    zero padded."""
    x = np.asarray(x, f32)
    rows, C = x.shape
    pad = (-C) % lanes
    if pad:
        x = np.concatenate([x, np.zeros((rows, pad), f32)], axis=1)
    return [x[:, j:j + lanes] for j in range(0, x.shape[1], lanes)]


def _inv_c(C):
    return f32(hw_inv_f32(np.array([C], f32))[0])


def _dq(q, scale, zp, compute):
    """dequant rounded to the compute type; the offset is the negated product -(zp * scale),
    not dq_offset's (-zp) * scale."""
    off = negmul_q(f32(zp), f32(scale))
    return round_float(dequant(np.asarray(q, np.int64), f32(scale), off), compute)


def _core_layernorm(x, alpha, beta, ik_type, out_round, c_real=0, eps=1e-12):
    """x (rows, C) fp32; returns t1 rounded to out_round ('f32' = not rounded).
    c_real > 0: statistics over channels [0, c_real) only, divisor c_real; the
    other channels read as zero there and are normalized like the rest."""
    rows, C = x.shape
    c_real = c_real or C
    s = _inv_c(c_real)
    xs = np.where(np.arange(C) < c_real, x, f32(0)).astype(f32)
    ax = np.zeros((rows, LANES), f32)
    ax2 = np.zeros((rows, LANES), f32)
    for v in _lane_steps(xs):
        ax = fma_q(v, s, ax)
        ax2 = fma_q(v, v, ax2)
    mu = vaddr_f32_reduce(ax)
    var = vaddr_f32_reduce(mul_q(ax2, s), np.full(rows, f32(eps)))
    var = msc_q(mu, mu, var)
    istd = hw_invsqrt_f32(var)
    mis = negmul_q(istd, mu)
    ik = round_float(istd, ik_type)
    t0 = fma_q(x, ik[:, None], mis[:, None])
    t1 = fma_q(t0, np.broadcast_to(alpha, x.shape), np.broadcast_to(beta, x.shape))
    return round_float(t1, out_round)


def _rmsnorm(x, gamma, inv_inner, eps):
    acc = np.zeros((x.shape[0], LANES), f32)
    for v in _lane_steps(x):
        acc = fma_q(v, v, acc)
    s = vaddr_f32_reduce(acc)
    r = hw_invsqrt_f32(add_q(mul_q(s, f32(inv_inner)), f32(eps)))
    return mul_q(mul_q(x, r[:, None]), gamma)


def _l2norm(x, alpha, beta):
    acc = np.zeros((x.shape[0], LANES), f32)
    for v in _lane_steps(x):
        acc = fma_q(v, v, acc)
    r = hw_invsqrt_f32(vaddr_f32_reduce(acc, np.full(x.shape[0], f32(1e-12))))
    return fma_q(mul_q(alpha, x), np.broadcast_to(r[:, None], x.shape),
                   np.broadcast_to(beta, x.shape))


def _qdq_l2norm(q, dq_scale, dq_zp, alpha, beta, q_scale, q_zp, out_t):
    x = _dq(q, dq_scale, dq_zp, "bf16")
    acc = np.zeros((x.shape[0], LANES), f32)
    for v in _lane_steps(x):
        acc = fma_q(v, v, acc)
    r = vmin(hw_invsqrt_f32(vaddr_f32_reduce(acc)), f32(1e12))
    if alpha is None:
        y = mul_q(x, r[:, None])
    else:
        y = fma_q(x, mul_q(alpha[None, :], r[:, None]), np.broadcast_to(beta, x.shape))
    return quant(round_float(y, "bf16"), q_scale, q_zp, out_t)


def _fast_rsqrt(v):
    """Reciprocal square root: bf16 seed, two Newton steps."""
    v = vmax(v, f32(1e-5))
    half = f32_to_bf16(v).astype(np.int64) >> 1
    y = bf16_to_f32((0x5F37 - half).astype(np.uint16))
    h = mul_q(v, f32(0.5))
    for _ in range(2):
        y = mul_q(y, msc_q(h, mul_q(y, y), f32(1.5)))
    return y


def _ln_stats(x, eps, stats, ik_type):
    """(ik, mis) per row. stats: 'fused' / 'batched' / 'batched_fast' /
    'c8_fused' / 'c8_batched_fast'."""
    rows, C = x.shape
    s = _inv_c(C)
    if stats.startswith("c8"):
        steps = _lane_steps(x, 8)
        sx = steps[0].copy()
        sq = mul_q(steps[0], steps[0])
        rest = steps[1:]
        if stats == "c8_fused" and rest:
            sx = add_q(sx, rest[0])
            sq = add_q(sq, mul_q(rest[0], rest[0]))
            rest = rest[1:]
        for v in rest:
            sx = add_q(sx, v)
            sq = fma_q(v, v, sq)

        def fold(v):
            a = add_q(v[:, :4], v[:, 4:])
            b = add_q(a[:, :2], a[:, 2:])
            return add_q(b[:, 0], b[:, 1])
        mu = mul_q(fold(sx), s)
        va = add_q(mul_q(fold(sq), s), f32(eps))
        var = msc_q(mu, mu, va)
        istd = hw_invsqrt_f32(var) if stats == "c8_fused" else _fast_rsqrt(var)
        return round_float(istd, ik_type), mul_q(neg_f32(mu), istd)
    sx = np.zeros((rows, LANES), f32)
    sq = np.zeros((rows, LANES), f32)
    for v in _lane_steps(x):
        sx = add_q(sx, v)
        sq = fma_q(v, v, sq)
    if stats == "fused":
        e = np.zeros(LANES, f32)
        e[16] = f32(eps)
        mu = vaddr_f32_reduce(mul_q(sx, s))
        var = msc_q(mu, mu, vaddr_f32_reduce(fma_q(sq, s, np.broadcast_to(e, sq.shape))))
        istd = hw_invsqrt_f32(var)
        mis = negmul_q(mu, istd)
    else:
        mu = mul_q(vaddr_f32_reduce(sx), s)
        var = msc_q(mu, mu, fma_q(vaddr_f32_reduce(sq), s, f32(eps)))
        istd = hw_invsqrt_f32(var) if stats == "batched" else _fast_rsqrt(var)
        mis = negmul_q(istd, mu)
    return round_float(istd, ik_type), mis


def _layernorm(x, alpha, beta, eps, stats, ik_type, out_round):
    ik, mis = _ln_stats(x, eps, stats, ik_type)
    t0 = fma_q(x, np.broadcast_to(ik[:, None], x.shape), np.broadcast_to(mis[:, None], x.shape))
    t1 = fma_q(t0, np.broadcast_to(alpha, x.shape), np.broadcast_to(beta, x.shape))
    return round_float(t1, out_round)


def _add_layernorm(a_bits, b_bits, dtype, alpha_bits, beta_bits, eps, stats):
    """Layer norm of a + b -> output bits."""
    wt = "f16" if dtype == "fp8" else dtype
    alpha, beta = decode(alpha_bits, wt), decode(beta_bits, wt)
    a, b = decode(a_bits, dtype), decode(b_bits, dtype)
    if dtype == "fp8":
        x = round_float(add_q(a, b), "f16")
        return encode(_layernorm(x, alpha, beta, eps, stats, "f16", "f32"), "fp8")
    x = round_float(add_q(a, b), dtype)
    return encode(_layernorm(x, alpha, beta, eps, stats, dtype, "f32"), dtype)


def _qdq_layernorm(q, compute, alpha, beta, dq_scale, dq_zp, out_t, variant):
    """variant: 'fused' / 'batched' / 'c8_fused' / 'c8_batched' / 'integer'."""
    if variant == "integer":
        return _qln_integer(q, alpha, beta, dq_scale, out_t)
    x = _dq(q, dq_scale, dq_zp, compute)
    stats = {"fused": "fused", "batched": "batched_fast", "c8_fused": "c8_fused",
             "c8_batched": "c8_batched_fast"}[variant]
    return float_to_int(_layernorm(x, alpha, beta, 1e-5, stats, compute, "f32"), out_t)


def layer_norm_a8_bfp16(q, alpha, beta, dq_zp, dq_sc, variant):
    """alpha / beta fp32 values; variant 'fused' / 'batched'; y rounded to f16, then bfp16
    bytes."""
    dq_scale = dq_sc
    x = _dq(q, dq_scale, dq_zp, "f16")
    stats = {"fused": "fused", "batched": "batched_fast"}[variant]
    y = _layernorm(x, alpha, beta, 1e-5, stats, "f16", "f16")
    return encode_bfp16(y.reshape(-1), Rnd.kHalfEven)


def _rsqrt16(v):
    """Reciprocal square root: f16 seed, two Newton steps."""
    v = np.asarray(v, f32)
    vs = vclamp(v, f32(6.2e-5), f32(65000.0))
    b = f32_to_f16(vs).astype(np.int64)
    bh = shift_round(b, 1, Rnd.kHalfEven)  # half to even
    y = f16_to_f32((0x59BB - bh).astype(np.uint16))
    h = mul_q(v, f32(0.5))
    for _ in range(2):
        y = mul_q(y, msc_q(h, mul_q(y, y), f32(1.5)))
    return y


def _qln_integer(q, alpha, beta, dq_scale, out_t):
    q = np.asarray(q, np.int64)
    rows, C = q.shape
    S = q.sum(axis=1)
    Q = (q * q).sum(axis=1)
    s = _inv_c(C)
    eps2 = mul_q(f32(1e-5), hw_inv_f32(np.array([mul_q(f32(dq_scale), f32(dq_scale))], f32))[0])
    mu = mac_exact(S, s, f32(0))
    e2 = mac_exact(Q, s, eps2)
    var = vmax(msc_q(mu, mu, e2), eps2)
    r = round_float(_rsqrt16(var), "f16")
    c = negmul_q(mac_exact(S, s, f32(1536)), r)
    x = (q + 1536).astype(f32)
    t0 = fma_q(x, np.broadcast_to(r[:, None], x.shape), np.broadcast_to(c[:, None], x.shape))
    t1 = fma_q(t0, np.broadcast_to(alpha, x.shape), np.broadcast_to(beta, x.shape))
    return float_to_int(t1, out_t)


def _qdq_ln_pre(mode, compute, a, b=None, k=None, a_t=None, da=None, db=None):
    """Layer-norm input (cd = rounded to the compute type): 'add': cd(cd(dq(a)) + cd(dq(b)));
    'const': cd(cd(dq(a)) + b); 'addc': cd(A + fp32(dq_b(b) + k)), A = cd(dq(a)) or a itself
    for a float a_t."""
    if mode == "add":
        return round_float(add_q(_dq(a, *da, compute), _dq(b, *db, compute)), compute)
    if mode == "const":
        return round_float(add_q(_dq(a, *da, compute), b), compute)
    A = _dq(a, *da, compute) if a_t in INT_RANGE else np.asarray(a, f32)
    dqb = _dq(b, *db, "f32")
    return round_float(add_q(A, add_q(dqb, np.broadcast_to(k, dqb.shape))), compute)


# Models named after the entries. Float tensors are bit patterns of their type t, weights
# (alpha, beta, gamma, add_const) bit patterns too unless noted; quantized entries take the
# weights as fp32 values.

def _wt(t):
    return "f16" if t == "fp8" else t


def layer_norm_bf16(x_bits, t, alpha, beta, C_real=0, epsilon=1e-12):
    y = _core_layernorm(decode(x_bits, t), decode(alpha, _wt(t)), decode(beta, _wt(t)),
                        "f32" if t == "fp8" else t, "f32", C_real, f32(epsilon))
    return encode(y, t)


def layer_norm_bfp16(x_bytes, alpha, beta, C):
    x = decode(x_bytes, "bfp16").reshape(-1, C)
    return encode(_core_layernorm(x, decode(alpha, "bf16"), decode(beta, "bf16"), "bf16",
                                  "bf16"), "bfp16")


def layer_norm_bf16_bfp16(x_bits, alpha, beta):
    return encode(_core_layernorm(decode(x_bits, "bf16"), decode(alpha, "bf16"),
                                  decode(beta, "bf16"), "bf16", "bf16"), "bfp16")


def _rms_eps(epsilon):
    """An all-zero epsilon word means 1e-12."""
    e = np.float32(epsilon)
    return f32(1e-12) if f32_bits(e) == 0 else e


def rmsnorm_bf16(x_bits, t, gamma, inv_inner_dim, epsilon):
    y = _rmsnorm(decode(x_bits, t), decode(gamma, _wt(t)), f32(inv_inner_dim),
                 _rms_eps(epsilon))
    return encode(y, t)


def rmsnorm_bfp16(x_bytes, gamma, C, inv_inner_dim, epsilon):
    x = decode(x_bytes, "bfp16").reshape(-1, C)
    return encode(_rmsnorm(x, decode(gamma, "bf16"), f32(inv_inner_dim), _rms_eps(epsilon)),
                  "bfp16")


def l2_norm_bf16(x_bits, t, alpha, beta):
    return encode(_l2norm(decode(x_bits, t), decode(alpha, _wt(t)), decode(beta, _wt(t))), t)


def qdq_l2norm_a8(q, alpha, beta, dq_zp, dq_sc, q_zp, q_sc, out_t):
    """alpha / beta fp32 values or both None."""
    return _qdq_l2norm(q, dq_sc, dq_zp, alpha, beta, q_sc, q_zp, out_t)


qdq_l2norm_a16 = qdq_l2norm_a8


def add_layernorm_bf16(a_bits, b_bits, t, alpha, beta, epsilon, stats):
    """stats 'fused' / 'batched'."""
    return _add_layernorm(a_bits, b_bits, t, alpha, beta, epsilon, stats)


add_layernorm_const_bf16 = add_layernorm_bf16


def qdq_add_layernorm(a, b, compute, alpha, beta, dq_zp, dq_sc, q_zp, q_sc, dq2_zp, dq2_sc,
                      epsilon, stats, out_t):
    """alpha / beta fp32 values."""
    x = _qdq_ln_pre("add", compute, a, b, da=(dq_sc, dq_zp), db=(dq2_sc, dq2_zp))
    y = _layernorm(x, alpha, beta, epsilon, stats, compute, compute)
    return quant(y, q_sc, q_zp, out_t)


def qdq_add_layernorm_const(a, add_const, compute, alpha, beta, dq_zp, dq_sc, q_zp, q_sc,
                            epsilon, stats, out_t):
    """add_const / alpha / beta fp32 values."""
    x = _qdq_ln_pre("const", compute, a, add_const, da=(dq_sc, dq_zp))
    y = _layernorm(x, alpha, beta, epsilon, stats, compute, compute)
    return quant(y, q_sc, q_zp, out_t)


def qdq_add_add_const_layernorm(a, a_t, b, add_const, compute, alpha, beta, dq_zp, dq_sc,
                                q_zp, q_sc, dq2_zp, dq2_sc, epsilon, stats, out_t):
    """a: integers, or fp32 values when a_t is the compute type; add_const / alpha / beta
    fp32 values."""
    x = _qdq_ln_pre("addc", compute, a, b, add_const, a_t, (dq_sc, dq_zp), (dq2_sc, dq2_zp))
    y = _layernorm(x, alpha, beta, epsilon, stats, compute, compute)
    return quant(y, q_sc, q_zp, out_t)


def qdq_layernorm(q, alpha, beta, out_t, compute, dq_zp, dq_sc, variant):
    """alpha / beta: the folded fp32 weights; variant 'fused' / 'batched' / 'c8_fused' /
    'c8_batched' / 'integer'."""
    return _qdq_layernorm(q, compute, alpha, beta, dq_sc, dq_zp, out_t, variant)
