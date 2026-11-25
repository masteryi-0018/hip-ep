# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""matmul/conv_instancenorm.h entry points against golden_conv_instancenorm.py."""

import numpy as np
import pytest

import golden_conv_instancenorm as gc
import golden_groupnorm as gg
from golden_int_round import INT_RANGE
from npu import I64, NPU_I8, NPU_I16, OUT, raw_rc
from qdq_testing import NS, SPECIAL
from testing import rejected
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


def call(amad, it, ifm, wts, alpha, beta, extra, case, c0, c2, dzp, dsc, qz, qs, dq2, slope):
    """Runs the entry (alpha / beta / the add constant as float16 bits) and its model."""
    split, ho, wo, sv_h, sv_w, h_loop, w_loop, kh, s = case
    h, w, ci = ifm.shape
    co = wts.shape[3]
    geo = (split, sv_h, sv_w, h_loop, w_loop, 0)
    al, be = gg.decode(alpha, "f16"), gg.decode(beta, "f16")
    if amad:
        a1, a2, cst = extra
        want = gc.qdq_ConvInstanceNormAddMulAdd(ifm, wts, al, be, a1, a2, gg.decode(cst, "f16"), it,
                                                s, s, *geo, c0, c2, dzp, dsc, qz, qs, *dq2, slope)
        got = run("npu_qdq_ConvInstanceNormAddMulAdd", ifm, code(it), wts, NPU_I8, alpha, beta, a1,
                  a2, cst, OUT, code(it), I64(h), I64(w), I64(ci), I64(kh), I64(kh), I64(co), s, s,
                  split, I64(sv_h), I64(sv_w), I64(h_loop), I64(w_loop), I64(0), c0, c2, dzp, dsc,
                  qz, qs, *dq2, slope, n=ho * wo * co, t=it)
    else:
        want = gc.qdq_ConvInFusion_A16(ifm, wts, al, be, it, s, s, *geo, c0, c2, dzp, dsc, qz, qs,
                                       slope)
        got = run("npu_qdq_ConvInFusion_A16", ifm, code(it), wts, NPU_I8, alpha, beta, OUT,
                  code(it), I64(h), I64(w), I64(ci), I64(kh), I64(kh), I64(co), s, s, split,
                  I64(sv_h), I64(sv_w), I64(h_loop), I64(w_loop), I64(0), c0, c2, dzp, dsc, qz, qs,
                  slope, n=ho * wo * co, t=it)
    assert_same(got.reshape(ho * wo, co), want.astype(np.int16))


@pytest.mark.parametrize("case", range(len(CONV_CASES)))
@pytest.mark.parametrize("it", ["i16", "u16"])
@pytest.mark.parametrize("amad", [False, True])
def test_conv_in(case, it, amad):
    ifm, wts, c0, c2 = conv_setup(CONV_CASES[case], it)
    co = wts.shape[3]
    alpha, beta = weights(co, "f16", -2, 2), weights(co, "f16", -10, 10)
    zp = 0.0 if it == "i16" else 32768.0
    c0 = (c0 + zp).astype(f32)  # conv outputs around the zero point
    extra = None
    lo = 0 if it == "u16" else -10
    if amad:
        a1 = RNG.integers(lo, lo + 20, co).astype(IT[it][1])
        a2 = RNG.integers(lo, lo + 20, co).astype(IT[it][1])
        extra = (a1, a2, weights(co, "f16", -10, 10))
    call(amad, it, ifm, wts, alpha, beta, extra, CONV_CASES[case], c0, c2, zp, f(1.6784559), zp,
         f(1 / 0.12429667), (float(lo), f(0.3)) if amad else (), f(0.02))


# ---- special values: the conv words (c0 / c2) and the norm words (alpha, beta, the
# constant, the scales and zero points, the slope) take NaN / inf / zeros / subnormals; a
# constant input gives a zero variance.
@pytest.mark.parametrize("amad", [False, True])
def test_conv_in_special(amad):
    case = CONV_CASES[0]
    it = "i16"
    ifm, wts, c0, c2 = conv_setup(case, it)
    co = wts.shape[3]
    j = np.arange(co)
    lo, hi = INT_RANGE[it]
    ifm[0, 0], ifm[1, 1] = lo, hi
    wts[0, 0, 0, :], wts[1, 1, 1, :] = -128, 127

    def enc(v):
        return gg.encode(np.asarray(v, f32), "f16")

    base = dict(c0=c0, c2=c2, alpha=weights(co, "f16", -2, 2), beta=weights(co, "f16", -10, 10),
                cst=weights(co, "f16", -10, 10), dsc=f(1.6784559), dzp=0.0, qs=f(1 / 0.12429667),
                qz=0.0, slope=f(0.02), ifm=ifm)

    def check(**kw):
        p = {**base, **kw}
        a1 = np.arange(co).astype(np.int16) % 7 - 3
        extra = (a1, a1, p["cst"]) if amad else None
        call(amad, it, p["ifm"], wts, p["alpha"], p["beta"], extra, case, p["c0"], p["c2"],
             p["dzp"], p["dsc"], p["qz"], p["qs"], (0.0, f(0.3)) if amad else (), p["slope"])

    sp = lambda k, m=1: SPECIAL[(m * j + k) % NS]          # noqa: E731
    check()
    check(c0=sp(0), c2=sp(1, 3))                             # conv words: NaN / inf / zeros ...
    check(c0=sp(0, 5), c2=sp(4, 3))
    check(alpha=enc(sp(0)), beta=enc(sp(3, 3)))
    check(alpha=enc(sp(1, 5)), beta=base["beta"], cst=enc(sp(2)))
    for i in range(NS):
        v = float(SPECIAL[i])
        for key in ("dsc", "dzp", "qs", "qz", "slope"):
            check(**{key: v})
    const = np.full_like(ifm, 100)                            # constant input: zero variance
    check(ifm=const)
    check(ifm=const, c2=np.zeros(co, f32))
    check(ifm=const, c2=sp(0, 3), c0=sp(2, 7))


def test_bad_args():
    """Every invalid argument is rejected before any allocation or launch."""
    case = CONV_CASES[0]
    split, ho, wo, sv_h, sv_w, h_loop, w_loop, kh, s = case
    ifm, wts, c0, c2 = conv_setup(case, "i16")
    h, w, ci = ifm.shape
    co = wts.shape[3]
    ab = weights(co, "f16", -2, 2)
    a1 = np.zeros(co, np.int16)
    ok = dict(it=code("i16"), wt=NPU_I8, ot=code("i16"), s=s, split=split, sv_h=sv_h, pad=0,
              c0=c0, kh=kh, alpha=ab, add2=a1)
    bad = [dict(it=NPU_I8), dict(wt=NPU_I16), dict(ot=NPU_I8), dict(s=0), dict(split=2),
           dict(split=-1), dict(sv_h=5), dict(sv_h=0), dict(pad=-1), dict(pad=10**6), dict(c0=None),
           dict(kh=h + 1), dict(alpha=None)]
    for amad in (False, True):
        for d in bad + ([dict(add2=None)] if amad else []):
            p = {**ok, **d}
            head = (ifm, p["it"], wts, p["wt"], p["alpha"], ab)
            tail = (OUT, p["ot"], I64(h), I64(w), I64(ci), I64(p["kh"]), I64(kh), I64(co), p["s"],
                    p["s"], p["split"], I64(p["sv_h"]), I64(sv_w), I64(h_loop), I64(w_loop),
                    I64(p["pad"]), p["c0"], c2, 0.0, 1.0, 0.0, 1.0)
            if amad:
                args = ("npu_qdq_ConvInstanceNormAddMulAdd", *head, a1, p["add2"], ab, *tail, 0.0,
                        1.0, 0.0)
            else:
                args = ("npu_qdq_ConvInFusion_A16", *head, *tail, 0.0)
            assert rejected(*args, nbytes=ho * wo * co * 2), (amad, d)
    # an empty image (h or w 0) is a no-op; negative sizes and null pointers are rejected
    for amad in (False, True):
        for hh, ww, a0, want in ((0, w, ab, 0), (h, 0, ab, 0), (-1, w, ab, -1), (h, -2, ab, -1),
                                 (0, w, None, -1)):
            head = (ifm, code("i16"), wts, NPU_I8, a0, ab)
            tail = (a1, code("i16"), I64(hh), I64(ww), I64(ci), I64(kh), I64(kh), I64(co), s, s,
                    split, I64(sv_h), I64(sv_w), I64(h_loop), I64(w_loop), I64(0), c0, c2, 0.0,
                    1.0, 0.0, 1.0)
            if amad:
                args = ("npu_qdq_ConvInstanceNormAddMulAdd", *head, a1, a1, ab, *tail, 0.0, 1.0,
                        0.0)
            else:
                args = ("npu_qdq_ConvInFusion_A16", *head, *tail, 0.0)
            assert raw_rc(*args) == want, (amad, hh, ww, a0 is None)
