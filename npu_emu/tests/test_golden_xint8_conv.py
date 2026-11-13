# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The xint8 convolution entry points (include/npu_emu_xint8_conv.h) against
the numpy reference in golden_xint8_conv.py, bit for bit, on random tensors:
input types, strides and kernel sizes, both SRS roundings, ReLU / ReLU6 /
LReLU, the conv_add paths with and without the q_add_mac stage, and small
shifts so that many accumulators land on a .5 tie."""

import numpy as np
import pytest

import golden_xint8_conv as gx
from npu import I64, NPU_I8, NPU_U8, OUT, call

RNG = np.random.default_rng(2031)
T = {np.int8: NPU_I8, np.uint8: NPU_U8}


def rnd(dt, shape, lo=None, hi=None):
    info = np.iinfo(dt)
    return RNG.integers(info.min if lo is None else lo, info.max if hi is None else hi, shape,
                        endpoint=True).astype(dt)


def run(name, *args, n, dt):
    return call(name, *args, out_nbytes=n * np.dtype(dt).itemsize, out_dtype=dt)


def assert_same(got, want):
    got = np.asarray(got).reshape(-1)
    want = np.asarray(want).astype(got.dtype).reshape(-1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{got.size} differ, first at {bad[:6].tolist()}: "
                           f"got {got[bad[:6]].tolist()} want {want[bad[:6]].tolist()}")


def ties(a, shift):
    """Fraction of accumulators exactly halfway between two outputs."""
    return float(np.mean((np.asarray(a, np.int64) & ((1 << shift) - 1)) == (1 << (shift - 1))))


GEOM = [(3, 3, 1), (3, 3, 2), (1, 1, 1), (5, 5, 1), (7, 7, 2), (2, 4, 1)]  # kh, kw, stride
ACT = [  # out dtype, max_value, lrelu
    (np.int8, -1, 0), (np.uint8, 127, 0), (np.uint8, 37, 0), (np.int8, -1, 1)]


@pytest.mark.parametrize("ifm_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("geom", GEOM)
@pytest.mark.parametrize("act", range(len(ACT)))
def test_conv(ifm_dt, geom, act):
    kh, kw, s = geom
    h, w, ci, co = 9 + kh, 8 + kw, 24, 16
    ifm = rnd(ifm_dt, (h, w, ci), *((-40, 40) if ifm_dt == np.int8 else (0, 60)))
    wts = rnd(np.int8, (kh, kw, ci, co), -6, 6)
    bias = rnd(np.int16, (co,), -3000, 3000)
    out_dt, mx, lr = ACT[act]
    for shift_bias, shift_out, alpha, slp in ((2, 5, 6553, 5), (0, 3, 26214, 2), (1, 0, 32768, -1)):
        args = (shift_bias, shift_out, mx, lr, alpha, slp)
        n = ((h - kh) // s + 1) * ((w - kw) // s + 1) * co
        got = run("npu_xint8_conv", ifm, T[ifm_dt], wts, bias, OUT, T[out_dt], I64(h), I64(w),
                  I64(ci), I64(kh), I64(kw), I64(co), s, s, *args, n=n, dt=out_dt)
        assert_same(got, gx.xint8_conv(ifm, wts, bias, out_dt, s, s, *args))
    if act == 0:
        a = gx.conv(ifm, wts, s, s) + gx.ups_bias(bias, co, 2)
        assert ties(a, 5) > 0.01


@pytest.mark.parametrize("dts", [(np.int8, np.int8, np.int8), (np.uint8, np.int8, np.uint8),
                                 (np.int8, np.uint8, np.int8)])
@pytest.mark.parametrize("geom", GEOM)
@pytest.mark.parametrize("away", [0, 1])
@pytest.mark.parametrize("mx", [-1, 37])
def test_depthwise(dts, geom, away, mx):
    ifm_dt, wts_dt, out_dt = dts
    if mx >= 0:
        out_dt = np.uint8
    kh, kw, s = geom
    h, w, c = 9 + kh, 10 + kw, 64
    ifm = rnd(ifm_dt, (h, w, c), *((-40, 40) if ifm_dt == np.int8 else (0, 60)))
    wts = rnd(wts_dt, (kh, kw, c), *((-6, 6) if wts_dt == np.int8 else (0, 9)))
    bias = rnd(np.int16, (c,), -2000, 2000)
    for shift_bias, shift_out in ((1, 3), (0, 1), (3, 6)):
        n = ((h - kh) // s + 1) * ((w - kw) // s + 1) * c
        got = run("npu_xint8_depthwise_conv", ifm, T[ifm_dt], wts, T[wts_dt], bias, OUT,
                  T[out_dt], I64(h), I64(w), I64(c), I64(kh), I64(kw), s, s, shift_bias, shift_out,
                  mx, away, n=n, dt=out_dt)
        assert_same(got, gx.xint8_depthwise(ifm, wts, bias, out_dt, s, s, shift_bias, shift_out,
                                            mx, away))


@pytest.mark.parametrize("ci,co,s", [(64, 64, 1), (128, 64, 1), (64, 32, 2)])
def test_dwc_pw(ci, co, s):
    h, w = 10, 18
    ifm = rnd(np.int8, (h, w, ci), -16, 15)
    dw, db = rnd(np.int8, (3, 3, ci), -4, 3), rnd(np.int16, (ci,), -64, 63)
    pw, pb = rnd(np.int8, (ci, co), -4, 3), rnd(np.int16, (co,), -64, 63)
    for args in ((0, 2, 0, 4), (1, 0, 0, 1), (0, 3, 2, 6)):
        n = ((h - 3) // s + 1) * ((w - 3) // s + 1) * co
        got = run("npu_xint8_dwc3x3_conv1x1", ifm, dw, db, pw, pb, OUT, I64(h), I64(w), I64(ci),
                  I64(co), I64(3), I64(3), s, s, *args, n=n, dt=np.int8)
        assert_same(got, gx.xint8_dwc_pw(ifm, dw, db, pw, pb, s, s, *args))


ADD = [  # out dtype, shift_conv_q, shift_ifm2, shift_out, max_value, lrelu, lrelu shifts / add shifts
    (np.int8, 3, 3, 4, -1, 0, (0, 0, 0, 0)),
    (np.int8, 0, 0, 2, -1, 0, (0, 0, 0, 0)),
    (np.int8, -1, 2, 4, -1, 0, (0, 0, 0, 0)),  # conv_add: no intermediate Q
    (np.uint8, -1, 3, 3, 23, 0, (0, 0, 0, 0)),
    (np.uint8, 2, 4, 4, 127, 0, (0, 0, 0, 0)),
    (np.uint8, 3, 3, 3, 23, 0, (0, 0, 0, 0)),
    (np.int8, 0, 1, 14, -1, 1, (4, 2, 0, 1)),  # lrelu: shift_out includes 16 - shift_lrelu_in
    (np.int8, 0, 0, 12, -1, 1, (2, -2, 1, 1)),
]


@pytest.mark.parametrize("cfg", range(len(ADD)))
@pytest.mark.parametrize("geom", [(3, 3, 1), (1, 1, 1), (3, 3, 2)])
@pytest.mark.parametrize("mac", [0, 1])
def test_conv_add(cfg, geom, mac):
    out_dt, scq, si2, so, mx, lr, (sln, slp, sai, sar) = ADD[cfg]
    kh, kw, s = geom
    h, w, ci, co = 8 + kh, 8 + kw, 32, 16
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    ifm, wts = rnd(np.int8, (h, w, ci), -25, 25), rnd(np.int8, (kh, kw, ci, co), -5, 5)
    bias, ifm2 = rnd(np.int16, (co,), -400, 400), rnd(np.int8, (ho, wo, co), -32, 32)
    x0 = rnd(np.int8, (ho, wo, co)) if mac else None
    if mac:
        out_dt = np.int8
    mc = (3, 2, 2) if mac else (0, 0, 0)
    args = (1, scq, si2, so, mx, lr, 6553, sln, slp, sai, sar, *mc)
    got = run("npu_xint8_conv_add", ifm, wts, bias, ifm2, x0, OUT, T[out_dt], I64(h), I64(w),
              I64(ci), I64(kh), I64(kw), I64(co), s, s, *args, n=ho * wo * co, dt=out_dt)
    assert_same(got, gx.xint8_conv_add(ifm, wts, bias, ifm2, x0, out_dt, s, s, *args))


@pytest.mark.parametrize("ifm_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("out_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("act", [0, 1, 2, 3])
@pytest.mark.parametrize("geom", [(3, 3, 1), (1, 1, 1), (3, 3, 2)])
def test_qxint8(ifm_dt, out_dt, act, geom):
    kh, kw, s = geom
    h, w, ci, co = 8 + kh, 9 + kw, 32, 16
    ifm, wts = rnd(ifm_dt, (h, w, ci)), rnd(np.int8, (kh, kw, ci, co), -20, 20)
    bias = rnd(np.int16, (co,), -2000, 2000)
    lo, hi = gx.rng_of(out_dt)
    zp = (lo + hi) // 2 + 3
    mx = hi if act != 2 else zp + 40
    for bias_shift, shift in ((3, 7), (0, 4), (5, 9)):
        n = ((h - kh) // s + 1) * ((w - kw) // s + 1) * co
        args = (bias_shift, shift, act, zp, mx, 6553)
        got = run("npu_qxint8_conv", ifm, T[ifm_dt], wts, bias, OUT, T[out_dt], I64(h), I64(w),
                  I64(ci), I64(kh), I64(kw), I64(co), s, s, *args, n=n, dt=out_dt)
        assert_same(got, gx.qxint8_conv(ifm, wts, bias, out_dt, s, s, *args))


def test_srs_ties():
    """The SRS rounds half to even (half away for dwc7x7) on both signs."""
    v = np.array([5, 6, 7, -5, -6, -7, 12, -12], np.int64)  # / 4: 1.25 1.5 1.75 ... 3 -3
    assert gx.srs(v, 2).tolist() == [1, 2, 2, -1, -2, -2, 3, -3]
    assert gx.srs(np.array([2, 6, -2, -6]), 2, away=True).tolist() == [1, 2, -1, -2]
    assert gx.srs(np.array([2, 6, -2, -6]), 2).tolist() == [0, 2, 0, -2]
    ifm = np.array([2, 6, -2, -6], np.int8).reshape(1, 4, 1)
    wts = np.ones((1, 1, 1), np.int8)
    for away, want in ((0, [0, 2, 0, -2]), (1, [1, 2, -1, -2])):
        got = run("npu_xint8_depthwise_conv", ifm, NPU_I8, wts, NPU_I8, None, OUT, NPU_I8, I64(1),
                  I64(4), I64(1), I64(1), I64(1), 1, 1, 0, 2, -1, away, n=4, dt=np.int8)
        assert got.tolist() == want
