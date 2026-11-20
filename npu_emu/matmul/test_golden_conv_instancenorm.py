# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""matmul/conv_instancenorm.h entry points against golden_conv_instancenorm.py."""

import numpy as np
import pytest

import golden_conv_instancenorm as gc
import golden_groupnorm as gg
from golden_int_round import INT_RANGE
from npu import I64, NPU_I8, OUT
from test_golden_groupnorm import IT, RNG, assert_same, code, f, f32, run, weights

CONV_CASES = [  # (split, ho, wo, sv_h, sv_w, h_loop, w_loop, kh, stride)
    (1, 8, 6, 2, 3, 1, 2, 3, 1),
    (0, 4, 24, 2, 2, 2, 1, 3, 1),
    (1, 8, 4, 1, 4, 2, 1, 1, 1),
]


def conv_setup(case, it):
    split, ho, wo, sv_h, sv_w, h_loop, w_loop, kh, s = case
    ci, co = 8, 32
    h, w = (ho - 1) * s + kh, (wo - 1) * s + kh
    lo, hi = INT_RANGE[it]
    ifm = RNG.integers(max(lo, -3000), min(hi, 3000), (h, w, ci)).astype(IT[it][1])
    wts = RNG.integers(-64, 64, (kh, kh, ci, co)).astype(np.int8)
    c2 = (RNG.uniform(0.5, 2, co) * f(0.00737)).astype(f32)
    c0 = RNG.uniform(-50, 50, co).astype(f32)
    return ifm, wts, c0, c2


@pytest.mark.parametrize("case", range(len(CONV_CASES)))
@pytest.mark.parametrize("ct", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("amad", [False, True])
def test_conv_in(case, ct, amad):
    split, ho, wo, sv_h, sv_w, h_loop, w_loop, kh, s = CONV_CASES[case]
    it = "i16"
    ifm, wts, c0, c2 = conv_setup(CONV_CASES[case], it)
    h, w, ci = ifm.shape
    co = wts.shape[3]
    alpha, beta = weights(co, ct, -2, 2), weights(co, ct, -10, 10)
    dsc, dzp, qs, qz = f(1.6784559), 0.0, f(1 / 0.12429667), 0.0
    act, slope = 3, f(0.02)
    amad_ops = dq2 = None
    extra = []
    if amad:
        a1 = RNG.integers(-10, 10, co).astype(np.int16)
        a2 = RNG.integers(-10, 10, co).astype(np.int16)
        cst = weights(co, ct, -10, 10)
        amad_ops, dq2 = (a1, gg.decode(cst, ct), a2), (f(0.3), 0.0)
        extra = [a1, a2, cst]
    want, _ = gc.conv_in(ifm, wts, s, s, c0, None, c2, 0, 0.0, 0.0, it, gg.decode(alpha, ct),
                         gg.decode(beta, ct), ct, split, sv_h, sv_w, h_loop, w_loop, 0, dsc, dzp,
                         qs, qz, act, slope, amad_ops, dq2)
    name = "npu_qdq_ConvInstanceNormAddMulAdd" if amad else "npu_qdq_ConvInFusion_A16"
    tail = [dsc, dzp, qs, qz] + ([f(0.3), 0.0] if amad else []) + [act, slope]
    got = run(name, ifm, code(it), wts, NPU_I8, I64(h), I64(w), I64(ci), I64(kh), I64(kh),
              I64(co), s, s, c0, None, c2, 0, 0.0, 0.0, code(it), alpha, beta, code(ct), *extra,
              OUT, code(ct), split, I64(sv_h), I64(sv_w), I64(h_loop), I64(w_loop), I64(0),
              *tail, n=ho * wo * co, t=it)
    assert_same(got.reshape(ho * wo, co), want.astype(np.int16))


# ---- special values: the conv words (c0 / c1 / c2) and the norm words (alpha, beta, the
# constant, the scales and zero points, the slope) take NaN / inf / zeros / subnormals; a
# constant input gives a zero variance.
from test_golden_qdq_gemm import NS, SPECIAL  # noqa: E402


@pytest.mark.parametrize("ct", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("amad", [False, True])
def test_conv_in_special(ct, amad):
    case = CONV_CASES[0]
    split, ho, wo, sv_h, sv_w, h_loop, w_loop, kh, s = case
    it = "i16"
    ifm, wts, c0, c2 = conv_setup(case, it)
    h, w, ci = ifm.shape
    co = wts.shape[3]
    j = np.arange(co)
    lo, hi = INT_RANGE[it]
    ifm[0, 0], ifm[1, 1] = lo, hi
    wts[0, 0, 0, :], wts[1, 1, 1, :] = -128, 127

    def enc(v):
        return gg.encode(np.asarray(v, f32), ct) if ct != "f32" else np.asarray(v, f32)

    base = dict(c0=c0, c1=None, c2=c2, alpha=weights(co, ct, -2, 2), beta=weights(co, ct, -10, 10),
                cst=weights(co, ct, -10, 10), dsc=f(1.6784559), dzp=0.0, qs=f(1 / 0.12429667),
                qz=0.0, slope=f(0.02), ifm=ifm, act=3)

    def check(**kw):
        p = {**base, **kw}
        a1 = np.arange(co).astype(np.int16) % 7 - 3
        amad_ops = dq2 = None
        extra = []
        if amad:
            amad_ops, dq2 = (a1, gg.decode(p["cst"], ct), a1), (f(0.3), 0.0)
            extra = [a1, a1, p["cst"]]
        want, _ = gc.conv_in(p["ifm"], wts, s, s, p["c0"], p["c1"], p["c2"], 0, 0.0, 0.0, it,
                             gg.decode(p["alpha"], ct), gg.decode(p["beta"], ct), ct, split, sv_h,
                             sv_w, h_loop, w_loop, 0, p["dsc"], p["dzp"], p["qs"], p["qz"],
                             p["act"], p["slope"], amad_ops, dq2)
        name = "npu_qdq_ConvInstanceNormAddMulAdd" if amad else "npu_qdq_ConvInFusion_A16"
        tail = [p["dsc"], p["dzp"], p["qs"], p["qz"]] + ([f(0.3), 0.0] if amad else []) + \
            [p["act"], p["slope"]]
        got = run(name, p["ifm"], code(it), wts, NPU_I8, I64(h), I64(w), I64(ci), I64(kh), I64(kh),
                  I64(co), s, s, p["c0"], p["c1"], p["c2"], 0, 0.0, 0.0, code(it), p["alpha"],
                  p["beta"], code(ct), *extra, OUT, code(ct), split, I64(sv_h), I64(sv_w),
                  I64(h_loop), I64(w_loop), I64(0), *tail, n=ho * wo * co, t=it)
        assert_same(got.reshape(ho * wo, co), want.astype(np.int16))

    sp = lambda k, m=1: SPECIAL[(m * j + k) % NS]          # noqa: E731
    check()
    check(c0=sp(0), c2=sp(1, 3))                             # conv words: NaN / inf / zeros ...
    check(c1=sp(2, 5), c2=sp(4, 3))
    check(c0=sp(0, 5), c1=sp(3, 3), c2=base["c2"])
    check(alpha=enc(sp(0)), beta=enc(sp(3, 3)))
    check(alpha=enc(sp(1, 5)), beta=base["beta"], cst=enc(sp(2)))
    for i in range(NS):
        v = float(SPECIAL[i])
        for key in ("dsc", "dzp", "qs", "qz", "slope"):
            check(**{key: v})
        check(act=1, qs=v)
    const = np.full_like(ifm, 100)                            # constant input: zero variance
    check(ifm=const)
    check(ifm=const, c2=np.zeros(co, f32))
    check(ifm=const, c2=sp(0, 3), c0=sp(2, 7))
