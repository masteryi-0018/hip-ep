# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for reduction/groupnorm.h (group norm and instance norm)."""

import numpy as np

from golden_act import hw_silu_f32
from golden_float_cvt import f32_to_fp8, ftz, kFtzIn, load_float, round_float, store_float
from golden_int_round import Rnd
from golden_mac import add_q, fma_q, msc_q, mul_q, negmul_q, vaddr_f32_reduce
from golden_minmax import vmax
from golden_mmac import fp8_decode_identity, mmac_add_bf16
from golden_nlf import hw_inv_f32, hw_invsqrt_f32
from golden_quant import dequant, dq_offset, quant

f32 = np.float32
EPS = f32(1e-5)


def decode(bits, t):
    """Bits of type t ('bf16', 'f16', 'fp8', 'f32') as fp32; fp8 in groups of 8, -0 -> +0."""
    if t == "fp8":
        return fp8_decode_identity(np.asarray(bits).reshape(-1)).reshape(np.shape(bits))
    return load_float(bits, t)


def encode(v, t, mode=Rnd.kHalfEven):
    if t == "fp8":
        return f32_to_fp8(v, mode)
    return store_float(v, t, mode)


def dq(q, sc, zp, ct):
    """ct(dequant(q, sc, dq_offset(zp, sc))) with q exact."""
    return round_float(dequant(np.asarray(q, np.int64), f32(sc), dq_offset(zp, f32(sc))), ct)


def in_sd_stats(X, split_mode, sv_hw, mmac=False):
    """X (hw, C) -> (S, S2) in the order of split_mode; mmac: x sums in the matrix unit."""
    hw, C = X.shape
    nq = hw // sv_hw
    a = np.zeros((2, nq, C), f32)
    a2 = np.zeros((2, nq, C), f32)
    Xq = X.reshape(nq, sv_hw, C)
    for i in range(sv_hw):
        x = Xq[:, i]
        a[i & 1] = mmac_add_bf16(a[i & 1], x) if mmac else add_q(a[i & 1], x)
        a2[i & 1] = fma_q(x, x, a2[i & 1])

    def tile(chunks):
        p = np.zeros(C, f32)
        p2 = np.zeros(C, f32)
        for q in chunks:
            p = add_q(add_q(p, a[0, q]), a[1, q])
            p2 = add_q(add_q(p2, a2[0, q]), a2[1, q])
        return p, p2

    def cascade(parts):
        r, r2 = parts[0]
        for p, p2 in parts[1:]:
            r, r2 = add_q(p, r), add_q(p2, r2)
        return r, r2

    if split_mode == 0:
        per, T = nq // 3, nq // 12
        return cascade([cascade([tile([c * per + 4 * j + k for j in range(T)])
                                 for k in (3, 2, 1, 0)]) for c in range(3)])
    if split_mode == 1:
        return cascade([tile([4 * j + k for j in range(nq // 4)]) for k in (3, 2, 1, 0)])
    if split_mode == 2:
        return cascade([tile([2 * j + k for j in range(nq // 2)]) for k in (1, 0)])
    if split_mode == 3:
        T = nq // 3
        return cascade([tile([c * T + j for j in range(T)]) for c in range(3)])
    return tile(range(nq))


def norm_coeffs(S, S2, N, eps=EPS):
    """(inv_std, -mu*inv_std)."""
    inv_s = hw_inv_f32(np.array([N], f32))[0]
    mu = mul_q(S, inv_s)
    var = msc_q(mu, mu, fma_q(S2, inv_s, eps))
    inv_std = hw_invsqrt_f32(var)
    return inv_std, negmul_q(mu, inv_std)


def fold(inv_std, m, alpha, beta):
    return mul_q(inv_std, alpha), fma_q(m, alpha, beta)


def amad(A, B, act, cst, add2):
    add1 = add_q(act, cst)
    return mul_q(A, add1), fma_q(B, add1, add2)


def post_act(y, act, slope, t):
    """Activation on a value of type t."""
    if act == 1:
        return vmax(y, f32(0))
    if act == 3:
        return vmax(y, round_float(mul_q(y, round_float(f32(slope), t)), t))
    return y


def _in_bf16(x_bits, io, alpha, beta, split_mode, sv_hw, N, act, slope, amad_ops=None):
    """Float instance norm -> output bits; amad_ops = (act, const, add2) as fp32."""
    X = decode(x_bits, io)
    S, S2 = in_sd_stats(X, split_mode, sv_hw, io == "fp8")
    A, B = fold(*norm_coeffs(S, S2, N), alpha, beta)
    if amad_ops is not None:
        A, B = amad(A, B, *amad_ops)
    y = fma_q(X, A[None], B[None])
    if io == "fp8":
        if act == 1:
            y = vmax(y, f32(0))
        elif act == 3:
            y = vmax(y, mul_q(y, f32(slope)))
        return encode(y, "fp8")
    return encode(post_act(round_float(y, io), act, slope, io), io)


def in_qdq_tail(X, A, B, ct, io, act, slope, q_sc, q_zp, q2=None):
    y = round_float(fma_q(X, A[None], B[None]), ct)
    if q2 is not None:
        return quant(y, q_sc, q_zp, io), quant(post_act(y, act, slope, ct), q2[0], q2[1], io)
    return quant(post_act(y, act, slope, ct), q_sc, q_zp, io)


def _in_qdq(q, io, alpha, beta, ct, split_mode, sv_hw, N, dq_sc, dq_zp, q_sc, q_zp, act, slope,
            amad_ops=None, dq2=None, q2=None):
    """Quantized instance norm; amad_ops = (act_q, const, add2_q), dq2 = (scale, zp)."""
    X = dq(q, dq_sc, dq_zp, ct)
    S, S2 = in_sd_stats(X, split_mode, sv_hw, ct == "bf16")
    A, B = fold(*norm_coeffs(S, S2, N), alpha, beta)
    if amad_ops is not None:
        aq, cst, a2q = amad_ops
        A, B = amad(A, B, dq(aq, *dq2, ct), cst, dq(a2q, *dq2, ct))
    return in_qdq_tail(X, A, B, ct, io, act, slope, q_sc, q_zp, q2)


def _in_stage2(q, io, S, S2, N, alpha, beta, ct, dq_sc, dq_zp, q_sc, q_zp, act, slope,
              amad_ops, dq2):
    X = dq(q, dq_sc, dq_zp, ct)
    A, B = fold(*norm_coeffs(S, S2, N), alpha, beta)
    aq, cst, a2q = amad_ops
    A, B = amad(A, B, dq(aq, *dq2, ct), cst, dq(a2q, *dq2, ct))
    return in_qdq_tail(X, A, B, ct, io, act, slope, q_sc, q_zp)


def _stream_coeffs(X):
    """Even / odd row accumulators -> (inv_std, -mu*inv_std)."""
    acc = np.zeros((2, X.shape[1]), f32)
    acc2 = np.zeros_like(acc)
    for i in range(X.shape[0] // 2):
        x = X[2 * i:2 * i + 2]
        acc = add_q(acc, x)
        acc2 = fma_q(x, x, acc2)
    return norm_coeffs(add_q(acc[0], acc[1]), add_q(acc2[0], acc2[1]), X.shape[0])


def instance_norm_stream_bf16(x_bits, alpha, beta):
    """alpha / beta: fp32 values."""
    X = decode(x_bits, "bf16")
    inv_std, m = _stream_coeffs(X)
    t0 = fma_q(X, round_float(inv_std, "bf16")[None], m[None])
    return encode(fma_q(t0, alpha[None], beta[None]), "bf16")


def qdq_instance_norm_stream(q, io, alpha, beta, dq_sc, q_sc, dq_offset, q_zp, relu_alpha):
    X = round_float(dequant(np.asarray(q, np.int64), f32(dq_sc), f32(dq_offset)), "bf16")
    inv_std, m = _stream_coeffs(X)
    t1 = fma_q(fma_q(X, round_float(inv_std, "bf16")[None], m[None]), alpha[None], beta[None])
    return quant(vmax(t1, mul_q(t1, f32(relu_alpha))), q_sc, q_zp, io)


def _gn_tile_sums(X):
    """Chunk sums: X (..., sv) -> (S, S2)."""
    sv = X.shape[-1]
    Xr = X.reshape(X.shape[:-1] + (2, sv // 64, 32))
    acc = np.zeros(X.shape[:-1] + (2, 32), f32)
    acc2 = np.zeros_like(acc)
    for i in range(sv // 64):
        x = Xr[..., i, :]
        acc = add_q(acc, x)
        acc2 = fma_q(x, x, acc2)
    return vaddr_f32_reduce(add_q(acc[..., 0, :], acc[..., 1, :])), \
        vaddr_f32_reduce(add_q(acc2[..., 0, :], acc2[..., 1, :]))


def _gn_bf16_sd(x_bits, io, alpha, beta, cg, split_mode, sv, N, eps, act):
    """Float group norm: x (G, l) -> output bits."""
    X = decode(x_bits, io)
    G, l = X.shape
    nq = l // sv
    S, S2 = _gn_tile_sums(X.reshape(G, nq, sv))
    rows = 2 if split_mode == 2 else 4

    def cascade(q0):
        r, r2 = S[:, q0 + rows - 1], S2[:, q0 + rows - 1]
        for k in range(rows - 2, -1, -1):
            r, r2 = add_q(S[:, q0 + k], r), add_q(S2[:, q0 + k], r2)
        return r, r2

    s = np.zeros(G, f32)
    s2 = np.zeros(G, f32)
    cols = 3 if split_mode == 0 else 1
    for j in range(nq // (rows * cols)):
        t, t2 = cascade(rows * j)
        for c in range(1, cols):
            r, r2 = cascade(c * (nq // 3) + rows * j)
            t, t2 = add_q(r, t), add_q(r2, t2)
        s, s2 = add_q(s, t), add_q(s2, t2)
    inv_std, m = norm_coeffs(s, s2, N, f32(eps))
    a = np.tile(alpha.reshape(G, cg), (1, l // cg))
    b = np.tile(beta.reshape(G, cg), (1, l // cg))
    tinv = inv_std if io == "fp8" else round_float(inv_std, io)
    y = fma_q(fma_q(X, tinv[:, None], m[:, None]), a, b)
    if io == "fp8":
        if act == 5:
            z = ftz(hw_silu_f32(y))
            return encode(z, "fp8", Rnd.kHalfAway)
        return encode(y, "fp8")
    y = round_float(y, io)
    if act == 5:
        y = round_float(hw_silu_f32(y), io, Rnd.kHalfAway, kFtzIn)
    return encode(y, io)


def _gn_qdq_sd(q, io, alpha, beta, cg, ct, split_mode, sv, N, eps, dq_sc, dq_zp, q_sc, q_zp,
               act):
    X = dq(q, dq_sc, dq_zp, ct)
    G, l = X.shape
    nq = l // sv
    S, S2 = _gn_tile_sums(X.reshape(G, nq, sv))
    ncol = 3 if split_mode in (0, 3) else 1
    nrow = {0: 4, 1: 4, 2: 2}.get(split_mode, 1)
    per = nq // ncol
    P = np.zeros((G, ncol, nrow), f32)
    P2 = np.zeros_like(P)
    for j in range(per // nrow):
        idx = np.arange(ncol)[:, None] * per + j * nrow + np.arange(nrow)[None, :]
        P, P2 = add_q(P, S[:, idx]), add_q(P2, S2[:, idx])
    s = s2 = None
    for c in range(ncol):
        r, r2 = P[:, c, nrow - 1], P2[:, c, nrow - 1]
        for k in range(nrow - 2, -1, -1):
            r, r2 = add_q(P[:, c, k], r), add_q(P2[:, c, k], r2)
        s, s2 = (r, r2) if s is None else (add_q(r, s), add_q(r2, s2))
    inv_std, m = norm_coeffs(s, s2, N, f32(eps))
    A, B = fold(inv_std[:, None], m[:, None], alpha.reshape(G, cg), beta.reshape(G, cg))
    y = round_float(fma_q(X, np.tile(A, (1, l // cg)), np.tile(B, (1, l // cg))), ct)
    if act == 5:
        y = round_float(hw_silu_f32(y), ct, Rnd.kHalfEven, kFtzIn)
    return quant(y, q_sc, q_zp, io)


def qdq_groupnorm(q, io, alpha, beta, cg, sv, epsilon, dq_zp, dq_sc, q_zp, q_sc):
    """alpha / beta: fp32 values (bf16)."""
    eps = epsilon
    X = dq(q, dq_sc, dq_zp, "bf16")
    G, l = X.shape
    S, S2 = _gn_tile_sums(X.reshape(G, l // sv, sv))
    s = np.zeros(G, f32)
    s2 = np.zeros(G, f32)
    for j in range(l // sv):
        s, s2 = add_q(S[:, j], s), add_q(S2[:, j], s2)
    inv_std, m = norm_coeffs(s, s2, l, f32(eps))
    a = np.tile(alpha.reshape(G, cg), (1, l // cg))
    b = np.tile(beta.reshape(G, cg), (1, l // cg))
    t0 = fma_q(X, round_float(inv_std, "bf16")[:, None], m[:, None])
    y = round_float(fma_q(t0, a, b), "bf16")
    return quant(y, q_sc, q_zp, io)


# Models named after the entries: alpha / beta / add1_const are fp32 values; the add1_act / add2
# operands of the float entries too, those of the quantized entries are integers.

def instance_norm_sd_bf16(x_bits, io, alpha, beta, sv_hw, split_mode, spatial_size, act,
                          leakyRelu_alpha):
    return _in_bf16(x_bits, io, alpha, beta, split_mode, sv_hw, spatial_size, act,
                    leakyRelu_alpha)


instance_norm_sd_opt_stage2_ifm_bf16 = instance_norm_sd_bf16


def instance_norm_add_mul_add_sd_bf16(x_bits, io, alpha, beta, add1_act, add2, add1_const,
                                      sv_hw, split_mode, spatial_size, act, leakyRelu_alpha):
    return _in_bf16(x_bits, io, alpha, beta, split_mode, sv_hw, spatial_size, act,
                    leakyRelu_alpha, (add1_act, add1_const, add2))


instance_norm_add_mul_add_sd_opt_stage2_ifm_bf16 = instance_norm_add_mul_add_sd_bf16


def qdq_instance_norm_sd(q, io, alpha, beta, ct, sv_hw, split_mode, spatial_size, dq_zp, dq_sc,
                         q_zp, q_sc, act, leakyRelu_alpha):
    return _in_qdq(q, io, alpha, beta, ct, split_mode, sv_hw, spatial_size, dq_sc, dq_zp, q_sc,
                   q_zp, act, leakyRelu_alpha)


qdq_instance_norm_sd_opt_stage2_ifm = qdq_instance_norm_sd


def qdq_instance_norm_sd_opt_stage2_ifm_double_out(q, io, alpha, beta, ct, sv_hw, split_mode,
                                                   spatial_size, dq_zp, dq_sc, q_zp, q_sc,
                                                   q2_zp, q2_sc, act, leakyRelu_alpha):
    """(ofm, ofm2)."""
    return _in_qdq(q, io, alpha, beta, ct, split_mode, sv_hw, spatial_size, dq_sc, dq_zp, q_sc,
                   q_zp, act, leakyRelu_alpha, q2=(q2_sc, q2_zp))


def qdq_instance_norm_add_mul_add_sd(q, io, alpha, beta, ct, add1_act, add2, add1_const, sv_hw,
                                     split_mode, spatial_size, dq_zp, dq_sc, q_zp, q_sc, dq2_zp,
                                     dq2_sc, act, leakyRelu_alpha):
    return _in_qdq(q, io, alpha, beta, ct, split_mode, sv_hw, spatial_size, dq_sc, dq_zp, q_sc,
                   q_zp, act, leakyRelu_alpha, amad_ops=(add1_act, add1_const, add2),
                   dq2=(dq2_sc, dq2_zp))


qdq_instance_norm_add_mul_add_sd_opt_stage2_ifm = qdq_instance_norm_add_mul_add_sd


def groupnorm_sd_bf16(x_bits, io, alpha, beta, cg, sv, div_factor, split_mode, epsilon, act):
    return _gn_bf16_sd(x_bits, io, alpha, beta, cg, split_mode, sv, div_factor, epsilon, act)


def qdq_groupnorm_sd(q, io, alpha, beta, ct, cg, sv, div_factor, split_mode, epsilon, dq_zp,
                     dq_sc, q_zp, q_sc, act):
    return _gn_qdq_sd(q, io, alpha, beta, cg, ct, split_mode, sv, div_factor, epsilon, dq_sc,
                      dq_zp, q_sc, q_zp, act)


def qdq_ConvInstanceNormAddMulAdd_Stage2(q, io, S, S2, alpha, beta, ct, add1_act, add2,  # noqa: N802
                                         add1_const, spatial_size, dq_zp, dq_sc, q_zp, q_sc,
                                         dq2_zp, dq2_sc, act, leakyRelu_alpha):
    return _in_stage2(q, io, S, S2, spatial_size, alpha, beta, ct, dq_sc, dq_zp, q_sc, q_zp, act,
                      leakyRelu_alpha, (add1_act, add1_const, add2), (dq2_sc, dq2_zp))
