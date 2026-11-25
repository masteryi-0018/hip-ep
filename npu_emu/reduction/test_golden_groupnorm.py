# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""reduction/groupnorm.h entry points against golden_groupnorm.py, byte for byte."""

import numpy as np
import pytest

import golden_groupnorm as gg
from golden_int_round import INT_RANGE
from npu import I64, NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, OUT, OUT2, raw_rc
from npu import f32 as f
from testing import INT as IT
from testing import call_n, rejected
from testing import same_bytes as assert_same

RNG = np.random.default_rng(5150)
f32 = np.float32
FT = {"bf16": (NPU_BF16, np.uint16), "f16": (NPU_F16, np.uint16), "fp8": (NPU_FP8, np.uint8),
      "f32": (NPU_F32, np.float32)}


def code(t):
    return FT[t][0] if t in FT else IT[t][0]


def run(name, *args, n, t, n2=0):
    return call_n(name, *args, n=n, dt=(FT[t] if t in FT else IT[t])[1], n2=n2)


def float_tensor(shape, t, lo=-20.0, hi=20.0, special=True):
    """Bits of a float tensor; `special`: cancelling, constant and +-0 channels."""
    v = RNG.uniform(lo, hi, shape).astype(f32)
    if special and v.ndim == 2 and v.shape[1] >= 3:
        big = 400.0 if t == "fp8" else 3000.0
        v[:, 0] = np.where(np.arange(v.shape[0]) % 2, big, -big) + RNG.uniform(-1, 1, v.shape[0])
        v[:, 1] = f32(1.5)
        v[:, 2] = np.where(np.arange(v.shape[0]) % 3 == 0, f32(-0.0), f32(0.75))
    return gg.encode(v, t) if t != "f32" else v


def weights(n, t, lo, hi):
    v = RNG.uniform(lo, hi, n).astype(f32)
    return gg.encode(v, t) if t != "f32" else v


def int_tensor(shape, t, zp):
    """Integers around the zero point, with the extremes and a cancelling channel."""
    lo, hi = INT_RANGE[t]
    v = RNG.integers(lo, hi, shape, endpoint=True)
    if v.ndim == 2 and v.shape[1] >= 2:
        v[:, 0] = np.where(np.arange(v.shape[0]) % 2, hi, lo)
        v[:, 1] = int(zp)
    return v.astype(IT[t][1])


def sv_rows(split_mode, T):
    return {0: 12, 1: 4, 2: 2, 3: 3, -1: 1}[split_mode] * T


IN_SPLITS = [0, 1, 2, 3, -1]
BF16_SPLITS = [0, 1, 2]


def ab_of(io):
    """alpha / beta type of the float entries: io, f16 for fp8."""
    return "f16" if io in ("f16", "fp8") else "bf16"


@pytest.mark.parametrize("io", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("split", BF16_SPLITS)
@pytest.mark.parametrize("act,slope", [(0, 0.0), (1, 0.0), (3, f(0.0123)), (3, 1.0)])
@pytest.mark.parametrize("amad", [False, True])
def test_in_bf16(io, split, act, slope, amad):
    ab = ab_of(io)
    C, sv_hw, T, pad = 64, 6, 2, 5
    hw = sv_rows(split, T) * sv_hw
    N = hw - pad
    lo, hi = (-8.0, 8.0) if io == "fp8" else (-20.0, 20.0)
    x = float_tensor((hw, C), io, lo, hi)
    x[N:] = 0
    alpha, beta = weights(C, ab, -2, 2), weights(C, ab, -10, 10)
    al, be = gg.decode(alpha, ab), gg.decode(beta, ab)
    if amad:
        a1, a2 = weights(C, io, -10, 10), weights(C, io, -10, 10)
        cst = weights(C, ab, -10, 10)
        want = gg.instance_norm_add_mul_add_sd_bf16(
            x, io, al, be, gg.decode(a1, io), gg.decode(a2, io), gg.decode(cst, ab), sv_hw, split,
            N, act, slope)
        args = [x, code(io), alpha, beta, a1, a2, cst]
        names = ("npu_instance_norm_add_mul_add_sd_bf16",
                 "npu_instance_norm_add_mul_add_sd_opt_stage2_ifm_bf16")
    else:
        want = gg.instance_norm_sd_bf16(x, io, al, be, sv_hw, split, N, act, slope)
        args = [x, code(io), alpha, beta]
        names = ("npu_instance_norm_sd_bf16", "npu_instance_norm_sd_opt_stage2_ifm_bf16")
    for name in names:
        got = run(name, *args, OUT, I64(hw), I64(C), I64(sv_hw), split, I64(N), act, slope,
                  n=hw * C, t=io)
        assert_same(got.reshape(hw, C), want)


@pytest.mark.parametrize("io", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("act", [0, 3])
def test_in_bf16_specials(io, act):
    """Subnormals, NaN, infinity and -0 in single channels."""
    C, sv_hw, T = 32, 4, 1
    hw = 12 * T * sv_hw
    v = RNG.uniform(-4, 4, (hw, C)).astype(f32)
    tiny = {"bf16": 2.0 ** -130, "f16": 2.0 ** -20, "fp8": 2.0 ** -8}[io]
    v[:, 3] = tiny * RNG.integers(-3, 4, hw)
    v[5, 4] = np.nan
    v[7, 5] = np.inf
    v[:, 6] = f32(-0.0)
    x = gg.encode(v, io)
    ab = ab_of(io)
    alpha, beta = weights(C, ab, -2, 2), weights(C, ab, -10, 10)
    want = gg.instance_norm_sd_bf16(x, io, gg.decode(alpha, ab), gg.decode(beta, ab), sv_hw, 0,
                                    hw, act, f(0.25))
    got = run("npu_instance_norm_sd_bf16", x, code(io), alpha, beta, OUT, I64(hw), I64(C),
              I64(sv_hw), 0, I64(hw), act, f(0.25), n=hw * C, t=io)
    assert_same(got.reshape(hw, C), want)


@pytest.mark.parametrize("io", ["i8", "u16"])
def test_in_qdq_saturate(io):
    """Saturating and vanishing output scales, rounding ties."""
    C, sv_hw, hw = 32, 4, 48
    x = int_tensor((hw, C), io, 0)
    alpha, beta = weights(C, "bf16", -2, 2), weights(C, "bf16", -10, 10)
    al, be = gg.decode(alpha, "bf16"), gg.decode(beta, "bf16")
    for qs, qz in ((f(1000.3), 3.0), (f(1e-3), 0.5), (f(0.5), 0.5)):
        want = gg.qdq_instance_norm_sd(x, io, al, be, "bf16", sv_hw, 0, hw, 1.0, f(0.0123), qz,
                                       qs, 0, 0.0)
        got = run("npu_qdq_instance_norm_sd", x, code(io), alpha, beta, NPU_BF16, OUT, I64(hw),
                  I64(C), I64(sv_hw), 0, I64(hw), 1.0, f(0.0123), qz, qs, 0, 0.0,
                  n=hw * C, t=io)
        assert_same(got.reshape(hw, C), want.astype(IT[io][1]))


QDQ = [  # (dq_zp, dq_sc, q_zp, q_sc)
    (3.0, f(0.0123), 2.0, f(1 / 0.0371)),
    (2.0, f(0.2), 1.0, f(1 / 0.3)),
    (30502.0, f(0.000258591725), 968.0, f(1 / 0.00290053268)),
]


def qdq_for(io, k):
    zp, sc, qz, qs = QDQ[k]
    if io in ("i8", "u8"):
        zp, qz = (min(zp, 100.0), min(qz, 100.0))
        sc = f(sc * 64) if k == 2 else sc
    if io in ("i8", "i16"):
        zp, qz = zp - 2 * zp // 3, qz - 2 * qz // 3
    return zp, sc, qz, qs


@pytest.mark.parametrize("io", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ct", ["f32", "bf16", "f16"])
@pytest.mark.parametrize("split", IN_SPLITS)
@pytest.mark.parametrize("act,slope", [(0, 0.0), (1, 0.0), (3, f(0.0123))])
def test_in_qdq(io, ct, split, act, slope):
    C, sv_hw, T, pad = 32, 4, 2, 3
    hw = sv_rows(split, T) * sv_hw
    N = hw - pad
    k = (IN_SPLITS.index(split) + len(ct) + act) % len(QDQ)
    q = qdq_for(io, k)
    x = int_tensor((hw, C), io, q[0])
    x[N:] = int(q[0])
    alpha, beta = weights(C, ct, -2, 2), weights(C, ct, -10, 10)
    al, be = gg.decode(alpha, ct), gg.decode(beta, ct)
    want = gg.qdq_instance_norm_sd(x, io, al, be, ct, sv_hw, split, N, *q, act, slope)
    names = ["npu_qdq_instance_norm_sd"]
    if split in BF16_SPLITS:
        names.append("npu_qdq_instance_norm_sd_opt_stage2_ifm")
    for name in names:
        got = run(name, x, code(io), alpha, beta, code(ct), OUT, I64(hw), I64(C), I64(sv_hw),
                  split, I64(N), *q, act, slope, n=hw * C, t=io)
        assert_same(got.reshape(hw, C), want.astype(IT[io][1]))
    if split not in BF16_SPLITS:
        return
    # double out: plain then activated with the second quantizer
    q2 = (7.0 if io in ("u8", "u16") else -7.0, f(1 / 0.0517))
    wa, wb = gg.qdq_instance_norm_sd_opt_stage2_ifm_double_out(x, io, al, be, ct, sv_hw, split, N,
                                                               *q, *q2, act, slope)
    ga, gb = run("npu_qdq_instance_norm_sd_opt_stage2_ifm_double_out", x, code(io), alpha, beta,
                 code(ct), OUT, OUT2, I64(hw), I64(C), I64(sv_hw), split, I64(N), *q, *q2, act,
                 slope, n=hw * C, n2=hw * C, t=io)
    assert_same(ga.reshape(hw, C), wa.astype(IT[io][1]))
    assert_same(gb.reshape(hw, C), wb.astype(IT[io][1]))


@pytest.mark.parametrize("io", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ct", ["f32", "bf16", "f16"])
@pytest.mark.parametrize("split", [0, 1, 2])
@pytest.mark.parametrize("act,slope", [(0, 0.0), (3, f(0.01))])
def test_in_qdq_amad(io, ct, split, act, slope):
    C, sv_hw, T, pad = 32, 4, 1, 2
    hw = sv_rows(split, T) * sv_hw
    q = qdq_for(io, 1)
    dq2 = (3.0, f(0.4))
    x = int_tensor((hw, C), io, q[0])
    alpha, beta = weights(C, ct, -2, 2), weights(C, ct, -10, 10)
    lo, hi = (-10, 10) if io in ("i8", "i16") else (0, 10)
    a1 = RNG.integers(lo, hi, C).astype(IT[io][1])
    a2 = RNG.integers(lo, hi, C).astype(IT[io][1])
    cst = weights(C, ct, -10, 10)
    want = gg.qdq_instance_norm_add_mul_add_sd(x, io, gg.decode(alpha, ct), gg.decode(beta, ct),
                                               ct, a1, a2, gg.decode(cst, ct), sv_hw, split,
                                               hw - pad, *q, *dq2, act, slope)
    for name in ("npu_qdq_instance_norm_add_mul_add_sd",
                 "npu_qdq_instance_norm_add_mul_add_sd_opt_stage2_ifm"):
        got = run(name, x, code(io), alpha, beta, code(ct), a1, a2, cst, OUT, I64(hw), I64(C),
                  I64(sv_hw), split, I64(hw - pad), *q, *dq2, act, slope, n=hw * C, t=io)
        assert_same(got.reshape(hw, C), want.astype(IT[io][1]))


@pytest.mark.parametrize("ab", ["bf16", "f32"])
@pytest.mark.parametrize("hw", [64, 250])
def test_in_stream_bf16(ab, hw):
    C = 64
    x = float_tensor((hw, C), "bf16", -1, 1)
    alpha, beta = weights(C, ab, -2, 2), weights(C, ab, -2, 2)
    want = gg.instance_norm_stream_bf16(x, gg.decode(alpha, ab), gg.decode(beta, ab))
    got = run("npu_instance_norm_stream_bf16", x, alpha, beta, code(ab), OUT, I64(hw), I64(C),
              n=hw * C, t="bf16")
    assert_same(got.reshape(hw, C), want)


@pytest.mark.parametrize("io", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ab", ["bf16", "f32"])
@pytest.mark.parametrize("relu_alpha", [1.0, 0.0, f(0.1)])
def test_in_stream_qdq(io, ab, relu_alpha):
    C, hw = 32, 128
    zp, sc, qz, qs = qdq_for(io, 0)
    off = f(-zp * float(sc))
    x = int_tensor((hw, C), io, zp)
    alpha, beta = weights(C, ab, -2, 2), weights(C, ab, -2, 2)
    want = gg.qdq_instance_norm_stream(x, io, gg.decode(alpha, ab), gg.decode(beta, ab), sc, qs,
                                       off, qz, relu_alpha)
    got = run("npu_qdq_instance_norm_stream", x, code(io), alpha, beta, code(ab), OUT, I64(hw),
              I64(C), sc, qs, off, qz, relu_alpha, n=hw * C, t=io)
    assert_same(got.reshape(hw, C), want.astype(IT[io][1]))


def gn_tensor(G, l, t, lo, hi):
    """(G, l): group 0 cancelling, group 1 constant, the rest random."""
    v = RNG.uniform(lo, hi, (G, l)).astype(f32)
    big = 400.0 if t == "fp8" else 2.0 ** 12
    v[0] = np.where(np.arange(l) % 2, big, -big) + RNG.uniform(-1, 1, l).astype(f32)
    v[0, ::7] = f32(1.0 / 3.0)
    v[1] = f32(2.25)
    return gg.encode(v, t)


@pytest.mark.parametrize("io", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("split", [0, 1, 2])
@pytest.mark.parametrize("act", [0, 5])
def test_gn_bf16_sd(io, split, act):
    G, cg, sv, T, pad = 4, 4, 128, 2, 16
    rows, cols = (2 if split == 2 else 4), (3 if split == 0 else 1)
    l = rows * cols * T * sv
    x = gn_tensor(G, l, io, -8 if io == "fp8" else -10, 8 if io == "fp8" else 10)
    x[:, l - pad:] = 0
    ab = ab_of(io)
    alpha, beta = weights(G * cg, ab, -2, 2), weights(G * cg, ab, -10, 10)
    eps = f(1e-5)
    want = gg.groupnorm_sd_bf16(x, io, gg.decode(alpha, ab), gg.decode(beta, ab), cg, sv,
                                l - pad, split, eps, act)
    got = run("npu_groupnorm_sd_bf16", x, code(io), alpha, beta, OUT, I64(G), I64(l), I64(cg),
              I64(sv), I64(l - pad), split, eps, act, n=G * l, t=io)
    assert_same(got.reshape(G, l), want)


@pytest.mark.parametrize("io", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ct", ["f32", "bf16", "f16"])
@pytest.mark.parametrize("split", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("act", [0, 5])
def test_gn_qdq_sd(io, ct, split, act):
    G, cg, sv, T = 3, 8, 64, 2
    ncol = 3 if split in (0, 3) else 1
    nrow = {0: 4, 1: 4, 2: 2}.get(split, 1)
    l = ncol * nrow * T * sv
    q = qdq_for(io, 1 if split % 2 else 2)
    x = int_tensor((G, l), io, q[0])
    pad = 64 if split == 1 else 0
    if pad:
        x[:, l - pad:] = int(q[0])
    alpha, beta = weights(G * cg, ct, -2, 2), weights(G * cg, ct, -10, 10)
    eps = f(1e-5)
    want = gg.qdq_groupnorm_sd(x, io, gg.decode(alpha, ct), gg.decode(beta, ct), ct, cg, sv,
                               l - pad, split, eps, *q, act)
    got = run("npu_qdq_groupnorm_sd", x, code(io), alpha, beta, code(ct), OUT, I64(G), I64(l),
              I64(cg), I64(sv), I64(l - pad), split, eps, *q, act, n=G * l, t=io)
    assert_same(got.reshape(G, l), want.astype(IT[io][1]))


@pytest.mark.parametrize("io", ["i8", "u8", "i16", "u16"])
def test_gn_qdq(io):
    G, cg, sv, nq = 5, 6, 192, 4
    l = sv * nq
    q = qdq_for(io, 1)
    x = int_tensor((G, l), io, q[0])
    alpha, beta = weights(G * cg, "bf16", -2, 2), weights(G * cg, "bf16", -10, 10)
    eps = f(1e-5)
    want = gg.qdq_groupnorm(x, io, gg.decode(alpha, "bf16"), gg.decode(beta, "bf16"), cg, sv, eps,
                            *q)
    got = run("npu_qdq_groupnorm", x, code(io), alpha, beta, OUT, I64(G), I64(l), I64(cg),
              I64(sv), eps, *q, n=G * l, t=io)
    assert_same(got.reshape(G, l), want.astype(IT[io][1]))


@pytest.mark.parametrize("io", ["bf16", "fp8"])
def test_gn_specials(io):
    """A NaN in one group, subnormals in another."""
    G, cg, sv, l = 3, 4, 64, 256
    v = RNG.uniform(-4, 4, (G, l)).astype(f32)
    v[0, 17] = np.nan
    v[1] = (2.0 ** (-130 if io == "bf16" else -8)) * RNG.integers(-3, 4, l)
    x = gg.encode(v, io)
    ab = ab_of(io)
    alpha, beta = weights(G * cg, ab, -2, 2), weights(G * cg, ab, -10, 10)
    for act in (0, 5):
        want = gg.groupnorm_sd_bf16(x, io, gg.decode(alpha, ab), gg.decode(beta, ab), cg, sv, l,
                                    1, f(1e-5), act)
        got = run("npu_groupnorm_sd_bf16", x, code(io), alpha, beta, OUT, I64(G), I64(l),
                  I64(cg), I64(sv), I64(l), 1, f(1e-5), act, n=G * l, t=io)
        assert_same(got.reshape(G, l), want)


@pytest.mark.parametrize("io", ["bf16", "f16", "fp8"])
def test_gn_silu_sweep(io):
    """SiLU over a dense symmetric sweep (mean 0, alpha 1, beta 0)."""
    ab = ab_of(io)
    l = 1024
    base = np.linspace(-12, 12, l // 2).astype(f32)
    x = gg.encode(np.concatenate([base, -base]).reshape(1, l), io)
    one, zero = gg.encode(np.ones(1, f32), ab), gg.encode(np.zeros(1, f32), ab)
    want = gg.groupnorm_sd_bf16(x, io, np.ones(1, f32), np.zeros(1, f32), 1, 64, l, 1, f(1e-5), 5)
    got = run("npu_groupnorm_sd_bf16", x, code(io), one, zero, OUT, I64(1), I64(l), I64(1),
              I64(64), I64(l), 1, f(1e-5), 5, n=l, t=io)
    assert_same(got.reshape(1, l), want)


@pytest.mark.parametrize("io", ["i16", "u16"])
@pytest.mark.parametrize("ct", ["f16", "bf16"])
def test_conv_in_stage2(io, ct):
    hw, C = 96, 64
    q = qdq_for(io, 1)
    x = int_tensor((hw, C), io, q[0])
    X = gg.dq(x, q[1], q[0], ct)
    S, S2 = gg.in_sd_stats(X, 1, 8)
    alpha, beta = weights(C, ct, -2, 2), weights(C, ct, -10, 10)
    a1 = RNG.integers(0, 10, C).astype(IT[io][1])
    a2 = RNG.integers(0, 10, C).astype(IT[io][1])
    cst = weights(C, ct, -10, 10)
    dq2 = (1.0, f(0.3))
    N = hw - 3
    want = gg.qdq_ConvInstanceNormAddMulAdd_Stage2(
        x, io, S, S2, gg.decode(alpha, ct), gg.decode(beta, ct), ct, a1, a2, gg.decode(cst, ct),
        N, *q, *dq2, 3, f(0.02))
    got = run("npu_qdq_ConvInstanceNormAddMulAdd_Stage2", x, code(io), S, S2, alpha, beta,
              code(ct), a1, a2, cst, OUT, I64(hw), I64(C), I64(N), *q, *dq2, 3, f(0.02),
              n=hw * C, t=io)
    assert_same(got.reshape(hw, C), want.astype(IT[io][1]))


def cancel_chains(rng, nblk, sv, C, nbig, big, small):
    """(nblk * sv, C): each accumulator chain of a block holds nbig x +big, then values of
    `small()`, then nbig x -big, so that the low bits of the small values meet the
    rounding of a large running sum and the large terms cancel afterwards."""
    k = np.arange(sv // 2)
    sign = np.repeat(np.where(k < nbig, 1, np.where(k >= sv // 2 - nbig, -1, 0)), 2)
    v = np.where(sign[None, :, None] != 0, sign[None, :, None] * big, small((nblk, sv, C)))
    return v.reshape(nblk * sv, C)


def mmac_sums_differ(X, split, sv_hw):
    """The x sums accumulated in the matrix unit differ from fp32 adds on X."""
    return (gg.in_sd_stats(X, split, sv_hw, True)[0] != gg.in_sd_stats(X, split, sv_hw)[0]).any()


@pytest.mark.parametrize("split", [1, 2])
def test_in_bf16_fp8_mmac_sums(split):
    """fp8 x sums ride the identity MMAC: running sums of 2^15 drop a 2^-9 tie."""
    rng = np.random.default_rng(91)
    C, sv_hw = 32, 400
    nblk = sv_rows(split, 1)
    v = cancel_chains(rng, nblk, sv_hw, C, 76, 448.0, lambda s: (
        rng.integers(1, 16, s) * rng.choice([-1, 1], s) * 2.0 ** -9).astype(f32))
    x = gg.encode(v.astype(f32), "fp8")
    hw = x.shape[0]
    assert mmac_sums_differ(gg.decode(x, "fp8"), split, sv_hw)
    alpha = gg.encode(np.full(C, 4096.0, f32), "f16")
    beta = gg.encode(np.zeros(C, f32), "f16")
    want = gg.instance_norm_sd_bf16(x, "fp8", gg.decode(alpha, "f16"), gg.decode(beta, "f16"),
                                    sv_hw, split, hw, 0, 0.0)
    for name in ("npu_instance_norm_sd_bf16", "npu_instance_norm_sd_opt_stage2_ifm_bf16"):
        got = run(name, x, NPU_FP8, alpha, beta, OUT, I64(hw), I64(C), I64(sv_hw), split,
                  I64(hw), 0, 0.0, n=hw * C, t="fp8")
        assert_same(got.reshape(hw, C), want)


@pytest.mark.parametrize("split", [-1, 0, 1, 2, 3])
def test_in_qdq_bf16_mmac_sums(split):
    """bf16-compute x sums ride the identity MMAC; f16 compute keeps fp32 adds."""
    rng = np.random.default_rng(92)
    C, sv_hw, pad, io = 32, 64, 3, "i16"
    nblk = sv_rows(split, 2 if split in (-1, 2) else 1)
    x = cancel_chains(rng, nblk, sv_hw, C, 8, 32767,
                      lambda s: rng.integers(-7, 8, s)).astype(IT[io][1])
    hw = x.shape[0]
    zp, sc, qz, qs = 0.0, f(float.fromhex("0x1.fep-12")), 0.0, f(1e7)
    for ct in ("bf16", "f16"):
        alpha = weights(C, ct, -2, 2)
        beta = gg.encode(np.zeros(C, f32), ct)
        if ct == "bf16":
            assert mmac_sums_differ(gg.dq(x, sc, zp, ct), split, sv_hw)
        want = gg.qdq_instance_norm_sd(x, io, gg.decode(alpha, ct), gg.decode(beta, ct), ct,
                                       sv_hw, split, hw - pad, zp, sc, qz, qs, 0, 0.0)
        names = ["npu_qdq_instance_norm_sd"]
        if split in BF16_SPLITS:
            names.append("npu_qdq_instance_norm_sd_opt_stage2_ifm")
        for name in names:
            got = run(name, x, code(io), alpha, beta, code(ct), OUT, I64(hw), I64(C), I64(sv_hw),
                      split, I64(hw - pad), zp, sc, qz, qs, 0, 0.0, n=hw * C, t=io)
            assert_same(got.reshape(hw, C), want.astype(IT[io][1]))


ENTRIES = ["npu_instance_norm_sd_bf16", "npu_instance_norm_sd_opt_stage2_ifm_bf16",
           "npu_instance_norm_add_mul_add_sd_bf16",
           "npu_instance_norm_add_mul_add_sd_opt_stage2_ifm_bf16", "npu_qdq_instance_norm_sd",
           "npu_qdq_instance_norm_sd_opt_stage2_ifm",
           "npu_qdq_instance_norm_sd_opt_stage2_ifm_double_out",
           "npu_qdq_instance_norm_add_mul_add_sd",
           "npu_qdq_instance_norm_add_mul_add_sd_opt_stage2_ifm", "npu_instance_norm_stream_bf16",
           "npu_qdq_instance_norm_stream", "npu_groupnorm_sd_bf16", "npu_qdq_groupnorm_sd",
           "npu_qdq_groupnorm", "npu_qdq_ConvInstanceNormAddMulAdd_Stage2"]


def test_bad_args():
    """Every entry rejects an argument its kernel cannot run (each call otherwise valid)."""
    hw, C, sv = 48, 32, 4          # split_mode 0 / 1 / 2 / 3 / -1 all fit 12 chunks
    xb = np.zeros((hw, C), np.uint16)
    xi = np.zeros((hw, C), np.int16)
    w = np.zeros(C, np.uint16)
    wf = np.zeros(C, np.float32)
    ai = np.zeros(C, np.int16)
    q = (0.0, 1.0, 0.0, 1.0)
    shp = (I64(hw), I64(C), I64(sv))
    BF, F16, I16 = NPU_BF16, NPU_F16, NPU_I16
    in_bf = ["npu_instance_norm_sd_bf16", "npu_instance_norm_sd_opt_stage2_ifm_bf16"]
    amad_bf = ["npu_instance_norm_add_mul_add_sd_bf16",
               "npu_instance_norm_add_mul_add_sd_opt_stage2_ifm_bf16"]
    bad = []
    for name in in_bf:
        bad += [(name, xb, NPU_I8, w, w, OUT, *shp, 0, I64(hw), 0, 0.0),        # io type
                (name, xb, BF, w, w, OUT, *shp, 3, I64(hw), 0, 0.0),            # split_mode
                (name, xb, BF, w, w, OUT, *shp, -1, I64(hw), 0, 0.0),
                (name, xb, BF, w, w, OUT, *shp, 0, I64(hw + 1), 0, 0.0),        # N > hw
                (name, xb, BF, w, w, OUT, I64(40), I64(C), I64(sv), 0, I64(40), 0, 0.0),  # plan
                (name, xb, BF, None, w, OUT, *shp, 0, I64(hw), 0, 0.0),
                (name, xb, BF, w, w, OUT, *shp, 0, I64(hw), 5, 0.0)]            # activation
    for name in amad_bf:
        bad += [(name, xb, BF, w, w, w, None, w, OUT, *shp, 0, I64(hw), 0, 0.0),
                (name, xb, BF, w, w, w, w, w, OUT, *shp, -1, I64(hw), 0, 0.0)]
    qin = "npu_qdq_instance_norm_sd"
    bad += [(qin, xi, BF, w, w, BF, OUT, *shp, 0, I64(hw), *q, 0, 0.0),        # io type
            (qin, xi, I16, w, w, NPU_FP8, OUT, *shp, 0, I64(hw), *q, 0, 0.0),  # compute type
            (qin, xi, I16, w, w, BF, OUT, *shp, 4, I64(hw), *q, 0, 0.0),
            (qin, xi, I16, w, w, BF, OUT, *shp, 0, I64(0), *q, 0, 0.0),
            (qin + "_opt_stage2_ifm", xi, I16, w, w, BF, OUT, *shp, 3, I64(hw), *q, 0, 0.0),
            (qin + "_opt_stage2_ifm_double_out", xi, I16, w, w, BF, OUT, None, *shp, 0, I64(hw),
             *q, 0.0, 1.0, 0, 0.0),
            (qin + "_opt_stage2_ifm_double_out", xi, I16, w, w, BF, OUT, OUT, *shp, -1, I64(hw),
             *q, 0.0, 1.0, 0, 0.0)]
    for name in ("npu_qdq_instance_norm_add_mul_add_sd",
                 "npu_qdq_instance_norm_add_mul_add_sd_opt_stage2_ifm"):
        bad += [(name, xi, I16, w, w, BF, ai, ai, w, OUT, *shp, 3, I64(hw), *q, 0.0, 1.0, 0, 0.0),
                (name, xi, I16, w, w, BF, ai, ai, None, OUT, *shp, 0, I64(hw), *q, 0.0, 1.0, 0,
                 0.0)]
    bad += [("npu_instance_norm_stream_bf16", xb, w, w, BF, OUT, I64(7), I64(C)),     # odd hw
            ("npu_instance_norm_stream_bf16", xb, w, w, F16, OUT, I64(hw), I64(C)),  # alpha type
            ("npu_qdq_instance_norm_stream", xi, I16, w, w, F16, OUT, I64(hw), I64(C), 1.0, 1.0,
             0.0, 0.0, 0.0),
            ("npu_qdq_instance_norm_stream", xi, BF, w, w, BF, OUT, I64(hw), I64(C), 1.0, 1.0,
             0.0, 0.0, 0.0)]
    gl = (I64(2), I64(512), I64(4))
    bad += [("npu_groupnorm_sd_bf16", xb, BF, w, w, OUT, I64(2), I64(96), I64(4), I64(96),
             I64(96), 1, 1e-5, 0),                                                    # sv % 64
            ("npu_groupnorm_sd_bf16", xb, BF, w, w, OUT, *gl, I64(128), I64(512), 1, 1e-5, 2),
            ("npu_groupnorm_sd_bf16", xb, BF, w, w, OUT, *gl, I64(128), I64(512), 3, 1e-5, 0),
            ("npu_groupnorm_sd_bf16", xb, BF, w, w, OUT, *gl, I64(128), I64(513), 1, 1e-5, 0),
            ("npu_qdq_groupnorm_sd", xi, I16, w, w, BF, OUT, *gl, I64(128), I64(512), 4, 1e-5,
             *q, 0),
            ("npu_qdq_groupnorm_sd", xi, I16, w, w, NPU_I8, OUT, *gl, I64(128), I64(512), 1,
             1e-5, *q, 0),
            ("npu_qdq_groupnorm", xi, BF, w, w, OUT, *gl, I64(128), 1e-5, *q),
            ("npu_qdq_groupnorm", xi, I16, None, w, OUT, *gl, I64(128), 1e-5, *q)]
    st2 = "npu_qdq_ConvInstanceNormAddMulAdd_Stage2"
    bad += [(st2, xi, NPU_I8, wf, wf, w, w, BF, ai, ai, w, OUT, I64(hw), I64(C), I64(hw), *q,
             0.0, 1.0, 0, 0.0),                                                        # io i8
            (st2, xi, I16, wf, wf, wf, wf, NPU_F32, ai, ai, wf, OUT, I64(hw), I64(C), I64(hw),
             *q, 0.0, 1.0, 0, 0.0),                                                    # f32
            (st2, xi, I16, None, wf, w, w, BF, ai, ai, w, OUT, I64(hw), I64(C), I64(hw), *q,
             0.0, 1.0, 0, 0.0)]
    # negative sizes; a null pointer is rejected even when there is no output
    bad += [(in_bf[0], xb, BF, w, w, OUT, I64(hw), I64(-1), I64(sv), 0, I64(hw), 0, 0.0),
            ("npu_groupnorm_sd_bf16", xb, BF, w, w, OUT, I64(-2), I64(512), I64(4), I64(128),
             I64(512), 1, 1e-5, 0),
            (qin, None, I16, w, w, BF, OUT, I64(hw), I64(0), I64(sv), 0, I64(hw), *q, 0, 0.0),
            ("npu_instance_norm_stream_bf16", xb, w, None, BF, OUT, I64(hw), I64(0)),
            ("npu_qdq_groupnorm", xi, I16, w, w, None, I64(0), I64(512), I64(4), I64(128), 1e-5,
             *q)]
    for name, *args in bad:
        assert rejected(name, *args, nbytes=4 * hw * C), (name, args)
    covered = {b[0] for b in bad}
    assert covered == set(ENTRIES), set(ENTRIES) - covered
    # c == 0 / g == 0: no output element, nothing launched, success
    assert raw_rc(in_bf[0], xb, BF, w, w, xb, I64(hw), I64(0), I64(sv), 0, I64(hw), 0, 0.0) == 0
    assert raw_rc("npu_instance_norm_stream_bf16", xb, w, w, BF, xb, I64(hw), I64(0)) == 0
    assert raw_rc("npu_qdq_groupnorm", xi, I16, w, w, xi, I64(0), I64(512), I64(4), I64(128),
                  1e-5, *q) == 0
    # given statistics: hw only counts output pixels and may be 0; the pointers stay required
    assert raw_rc(st2, xi, I16, wf, wf, w, w, BF, ai, ai, w, xi, I64(0), I64(C), I64(hw), *q,
                  0.0, 1.0, 0, 0.0) == 0
    assert raw_rc(st2, xi, I16, wf, None, w, w, BF, ai, ai, w, xi, I64(0), I64(C), I64(hw), *q,
                  0.0, 1.0, 0, 0.0) == -1
    assert raw_rc(st2, xi, I16, wf, wf, w, w, BF, ai, ai, w, xi, I64(-1), I64(C), I64(hw), *q,
                  0.0, 1.0, 0, 0.0) == -1
