# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The xint8 convolution entry points against golden_xint8_conv.py, bit for bit."""

import numpy as np
import pytest

import golden_xint8_conv as gx
from golden_int_round import Rnd, shift_round
from golden_mac import fma
from golden_act import hw_gelu_f32
from npu import I64, NPU_I8, NPU_U8, OUT
from testing import assert_same, rand_ints, rejected
from testing import call_n as run

RNG = np.random.default_rng(2031)
T = {np.int8: NPU_I8, np.uint8: NPU_U8}


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
    ifm = rand_ints(RNG, ifm_dt, (h, w, ci), *((-40, 40) if ifm_dt == np.int8 else (0, 60)))
    wts = rand_ints(RNG, np.int8, (kh, kw, ci, co), -6, 6)
    bias = rand_ints(RNG, np.int16, (co,), -3000, 3000)
    out_dt, mx, lr = ACT[act]
    for shift_bias, shift_out, alpha, slp in ((2, 5, 6553, 5), (0, 3, 26214, 2), (1, 0, 32768, -1)):
        args = (shift_bias, shift_out, mx, lr, alpha, slp)
        n = ((h - kh) // s + 1) * ((w - kw) // s + 1) * co
        head = (ifm, T[ifm_dt], wts, bias, OUT, T[out_dt], I64(h), I64(w), I64(ci), I64(kh),
                I64(kw), I64(co), s, s)
        want = gx.xint8_conv(ifm, wts, bias, out_dt, s, s, *args)
        assert_same(run("npu_xint8_conv", *head, *args, n=n, dt=out_dt), want)
        if not lr:  # the pinned kernels have no LeakyReLU
            for name in ("npu_xint8_conv_pin_buffer", "npu_xint8_conv_pin_stream"):
                assert_same(run(name, *head, *args[:3], n=n, dt=out_dt), want)
        if (kh, kw, s, act) == (1, 1, 1, 0):
            got = run("npu_xint8_conv1x1ci64co64", ifm, T[ifm_dt], wts, bias, OUT, I64(h), I64(w),
                      I64(ci), I64(co), shift_bias, shift_out, n=n, dt=np.int8)
            assert_same(got, want)
    if act == 0:
        a = gx.conv(ifm, wts, s, s) + gx.ups_bias(bias, co, 2)
        assert ties(a, 5) > 0.01


DWC_DTS = [(np.int8, np.int8, np.int8), (np.uint8, np.int8, np.uint8), (np.int8, np.uint8, np.int8)]


@pytest.mark.parametrize("dts", DWC_DTS)
@pytest.mark.parametrize("geom", GEOM + [(7, 7, 1)])
@pytest.mark.parametrize("mx", [-1, 37])
def test_depthwise(dts, geom, mx):
    """The 7 x 7 stride 1 geometry also runs the kernel of that size, which rounds half away."""
    ifm_dt, wts_dt, out_dt = dts
    if mx >= 0:
        out_dt = np.uint8
    kh, kw, s = geom
    h, w, c = 9 + kh, 10 + kw, 64
    ifm = rand_ints(RNG, ifm_dt, (h, w, c), *((-40, 40) if ifm_dt == np.int8 else (0, 60)))
    wts = rand_ints(RNG, wts_dt, (kh, kw, c), *((-6, 6) if wts_dt == np.int8 else (0, 9)))
    bias = rand_ints(RNG, np.int16, (c,), -2000, 2000)
    for shift_bias, shift_out in ((1, 3), (0, 1), (3, 6)):
        n = ((h - kh) // s + 1) * ((w - kw) // s + 1) * c
        head = (ifm, T[ifm_dt], wts, T[wts_dt], bias, OUT, T[out_dt], I64(h), I64(w), I64(c))
        got = run("npu_xint8_depthwise_conv", *head, I64(kh), I64(kw), s, s, shift_bias, shift_out,
                  mx, n=n, dt=out_dt)
        assert_same(got, gx.xint8_depthwise(ifm, wts, bias, out_dt, s, s, shift_bias, shift_out,
                                            mx, 0))
        if geom == (7, 7, 1):
            got = run("npu_xint8_dwc7x7_s1", *head, shift_bias, shift_out, mx, n=n, dt=out_dt)
            assert_same(got, gx.xint8_depthwise(ifm, wts, bias, out_dt, s, s, shift_bias,
                                                shift_out, mx, 1))


@pytest.mark.parametrize("ci,co,s", [(64, 64, 1), (128, 64, 1), (64, 32, 2)])
def test_dwc_pw(ci, co, s):
    h, w = 10, 18
    ifm = rand_ints(RNG, np.int8, (h, w, ci), -16, 15)
    dw, db = rand_ints(RNG, np.int8, (3, 3, ci), -4, 3), rand_ints(RNG, np.int16, (ci,), -64, 63)
    pw, pb = rand_ints(RNG, np.int8, (ci, co), -4, 3), rand_ints(RNG, np.int16, (co,), -64, 63)
    for args in ((0, 2, 0, 4), (1, 0, 0, 1), (0, 3, 2, 6)):
        n = ((h - 3) // s + 1) * ((w - 3) // s + 1) * co
        got = run("npu_xint8_dwc3x3_conv1x1_fused", ifm, dw, db, pw, pb, OUT, I64(h), I64(w),
                  I64(ci), I64(co), I64(3), I64(3), s, s, *args, n=n, dt=np.int8)
        assert_same(got, gx.xint8_dwc_pw(ifm, dw, db, pw, pb, s, s, *args))


ADD = [  # out dtype, shift_conv_q, shift_ifm2, shift_out, max_value, lrelu, lrelu shifts / add shifts
    (np.int8, 3, 3, 4, -1, 0, (0, 0, 0, 0)),
    (np.int8, 0, 0, 2, -1, 0, (0, 0, 0, 0)),
    (np.int8, -1, 2, 4, -1, 0, (0, 0, 0, 0)),  # conv_add: no intermediate Q
    (np.uint8, -1, 3, 3, 23, 0, (0, 0, 0, 0)),
    (np.uint8, 2, 4, 4, 127, 0, (0, 0, 0, 0)),
    (np.uint8, 3, 3, 3, 23, 0, (0, 0, 0, 0)),
    (np.int8, 0, 1, 14, -1, 1, (4, 2, 0, 1)),  # lrelu
    (np.int8, 0, 0, 12, -1, 1, (2, -2, 1, 1)),
]


@pytest.mark.parametrize("cfg", range(len(ADD)))
@pytest.mark.parametrize("geom", [(3, 3, 1), (1, 1, 1), (3, 3, 2)])
def test_conv_add(cfg, geom):
    """Configurations without activation and with an intermediate Q also run the mac kernel."""
    out_dt, scq, si2, so, mx, lr, (sln, slp, sai, sar) = ADD[cfg]
    kh, kw, s = geom
    h, w, ci, co = 8 + kh, 8 + kw, 32, 16
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    ifm = rand_ints(RNG, np.int8, (h, w, ci), -25, 25)
    wts = rand_ints(RNG, np.int8, (kh, kw, ci, co), -5, 5)
    bias = rand_ints(RNG, np.int16, (co,), -400, 400)
    ifm2 = rand_ints(RNG, np.int8, (ho, wo, co), -32, 32)
    geo = (I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s)
    args = (1, scq, si2, so, mx, lr, 6553, sln, slp, sai, sar)
    got = run("npu_xint8_conv_add", ifm, wts, bias, ifm2, OUT, T[out_dt], *geo, *args,
              n=ho * wo * co, dt=out_dt)
    assert_same(got, gx.xint8_conv_add(ifm, wts, bias, ifm2, None, out_dt, s, s, *args, 0, 0, 0))
    if mx >= 0 or lr:
        return
    x0, mc = rand_ints(RNG, np.int8, (ho, wo, co)), (3, 2, 2)
    mac = ("npu_xint8_conv_add_mac", ifm, wts, bias, ifm2, x0, OUT, *geo, 1, scq, si2, so, *mc)
    if scq < 0:  # the mac kernel needs the intermediate Q
        assert rejected(*mac, nbytes=ho * wo * co)
    else:
        assert_same(run(*mac, n=ho * wo * co, dt=np.int8),
                    gx.xint8_conv_add(ifm, wts, bias, ifm2, x0, np.int8, s, s, *args, *mc))


@pytest.mark.parametrize("ifm_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("out_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("act", [0, 1, 2, 3])
@pytest.mark.parametrize("geom", [(3, 3, 1), (1, 1, 1), (3, 3, 2)])
def test_qxint8(ifm_dt, out_dt, act, geom):
    kh, kw, s = geom
    h, w, ci, co = 8 + kh, 9 + kw, 32, 16
    ifm, wts = rand_ints(RNG, ifm_dt, (h, w, ci)), rand_ints(RNG, np.int8, (kh, kw, ci, co), -20, 20)
    bias = rand_ints(RNG, np.int16, (co,), -2000, 2000)
    lo, hi = gx.rng_of(out_dt)
    zp = (lo + hi) // 2 + 3
    mx = hi if act != 2 else zp + 40
    for bias_shift, shift in ((3, 7), (0, 4), (5, 9)):
        n = ((h - kh) // s + 1) * ((w - kw) // s + 1) * co
        args = (bias_shift, shift, act, zp, mx, 6553, 0.0, 0.0)
        got = run("npu_qxint8_conv", ifm, T[ifm_dt], wts, bias, OUT, T[out_dt], I64(h), I64(w),
                  I64(ci), I64(kh), I64(kw), I64(co), s, s, *args, n=n, dt=out_dt)
        assert_same(got, gx.qxint8_conv(ifm, wts, bias, out_dt, s, s, *args))


@pytest.mark.parametrize("out_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("scale", [0.05, 0.0123, 1 / 3])
def test_qxint8_gelu(out_dt, scale):
    """activation 6: GeLU on the dequantized conv output, then mul and add (two roundings)."""
    kh, kw, s = 3, 3, 1
    h, w, ci, co = 8 + kh, 9 + kw, 32, 16
    ifm, wts = rand_ints(RNG, np.int8, (h, w, ci)), rand_ints(RNG, np.int8, (kh, kw, ci, co), -20, 20)
    bias = rand_ints(RNG, np.int16, (co,), -2000, 2000)
    lo, hi = gx.rng_of(out_dt)
    zp = (lo + hi) // 2 + 3
    n = ((h - kh) // s + 1) * ((w - kw) // s + 1) * co
    args = (3, 7, 6, zp, hi, 0, float(np.float32(scale)), float(np.float32(1.0 / scale)))
    got = run("npu_qxint8_conv", ifm, T[np.int8], wts, bias, OUT, T[out_dt], I64(h), I64(w),
              I64(ci), I64(kh), I64(kw), I64(co), s, s, *args, n=n, dt=out_dt)
    assert_same(got, gx.qxint8_conv(ifm, wts, bias, out_dt, s, s, *args))


# (s_out, zp_out, conv output) where gelu * fp32(1 / s) + zp rounded twice and
# rounded once (one fma) quantize differently
GELU_Q_TIES = [(0.0018063593000510066, -84, 96), (0.0018000708444210996, -108, 84),
               (0.0034921921812486236, -71, 123), (0.004277373667325061, -66, 10)]


@pytest.mark.parametrize("scale,zp,r", GELU_Q_TIES)
def test_qxint8_gelu_two_roundings(scale, zp, r):
    """A 1x1 conv with weight 1 feeds r into the GeLU; mul then add, two roundings."""
    ifm = np.full((1, 1, 1), r, np.int8)
    wts = np.ones((1, 1, 1, 1), np.int8)
    args = (0, 0, 6, zp, 127, 0, float(np.float32(scale)), float(np.float32(1.0 / scale)))
    want = gx.qxint8_conv(ifm, wts, None, np.int8, 1, 1, *args)
    real = np.float32((np.float32(r) - np.float32(zp)) * np.float32(scale))
    fused = fma(hw_gelu_f32(np.array([real], np.float32)), np.float32(1.0 / scale), np.float32(zp))
    assert int(np.rint(np.float64(fused[0]))) != int(want.reshape(-1)[0])   # the data separates the two
    got = run("npu_qxint8_conv", ifm, T[np.int8], wts, None, OUT, T[np.int8], I64(1), I64(1), I64(1),
              I64(1), I64(1), I64(1), 1, 1, *args, n=1, dt=np.int8)
    assert_same(got, want)


def test_srs_ties():
    """SRS rounds half to even (or half away) on both signs."""
    v = np.array([5, 6, 7, -5, -6, -7, 12, -12], np.int64)  # / 4: 1.25 1.5 1.75 ... 3 -3
    assert shift_round(v, 2, Rnd.kHalfEven).tolist() == [1, 2, 2, -1, -2, -2, 3, -3]
    assert shift_round(np.array([2, 6, -2, -6]), 2, Rnd.kHalfAway).tolist() == [1, 2, -1, -2]
    assert shift_round(np.array([2, 6, -2, -6]), 2, Rnd.kHalfEven).tolist() == [0, 2, 0, -2]
    ifm, wts = np.zeros((7, 10, 1), np.int8), np.zeros((7, 7, 1), np.int8)
    ifm[3, 3:7, 0], wts[3, 3, 0] = [2, 6, -2, -6], 1  # the centre tap passes the four values
    head = (ifm, NPU_I8, wts, NPU_I8, None, OUT, NPU_I8, I64(7), I64(10), I64(1))
    got = run("npu_xint8_depthwise_conv", *head, I64(7), I64(7), 1, 1, 0, 2, -1, n=4, dt=np.int8)
    assert got.tolist() == [0, 2, 0, -2]
    got = run("npu_xint8_dwc7x7_s1", *head, 0, 2, -1, n=4, dt=np.int8)
    assert got.tolist() == [1, 2, -1, -2]


# accumulators where max(A, alpha * A + (1 - alpha) * O) and the centred
# max(A - O, alpha * (A - O)) + O quantize differently (alpha_q16 21845, zp 100, shift 20)
LRELU_SPLIT = [-41420987, -242750679, -365435891]


def test_qxint8_lrelu_expression():
    """1x1 conv over [127, 127, 127, 1] with per-channel weights and bias << 16 builds A."""
    alpha_q16, zp, shift = 21845, 100, 20
    co = len(LRELU_SPLIT)
    ifm = np.array([127, 127, 127, 1], np.int8).reshape(1, 1, 4)
    wts = np.zeros((1, 1, 4, co), np.int8)
    bias = np.zeros(co, np.int16)
    for j, a in enumerate(LRELU_SPLIT):
        bias[j] = round(a / 65536)
        rem = a - int(bias[j]) * 65536
        q = round(rem / 127)
        wts[0, 0, :3, j] = [q // 3, q // 3, q - 2 * (q // 3)]
        wts[0, 0, 3, j] = rem - 127 * q
    acc = gx.conv(ifm, wts, 1, 1) + bias.astype(np.int64) * 65536
    assert acc.reshape(-1).tolist() == LRELU_SPLIT
    args = (16, shift, 3, zp, 127, alpha_q16, 0.0, 0.0)
    want = gx.qxint8_conv(ifm, wts, bias, np.int8, 1, 1, *args)
    off = np.float32(np.ldexp(zp, shift))
    c = acc.astype(np.float32) - off
    centred = np.maximum(c, c * np.float32(alpha_q16 / 65536.0)) + off
    centred = np.rint((centred * np.float32(2.0**-shift)).astype(np.float64))
    assert (centred.reshape(-1) != want.reshape(-1)).all()   # the data separates the two
    got = run("npu_qxint8_conv", ifm, T[np.int8], wts, bias, OUT, T[np.int8], I64(1), I64(1),
              I64(4), I64(1), I64(1), I64(co), 1, 1, *args, n=co, dt=np.int8)
    assert_same(got, want)


# ---- special values: the words are integers (shifts, alpha, max_value, zero point), so the
# special inputs are the integer limits: operands at the type limits, shift words far outside
# the accumulator width (the saturating shift of common/int_round.h), sums that saturate at
# int64, and for the GeLU entry NaN / inf / zero scales.
SHIFTS = [-2**31, -70, -65, -64, -63, -33, -32, -31, -8, -1, 0, 1, 8, 15, 16, 31, 32, 33, 62, 63, 64,
          65, 70, 2**31 - 1]
MAXV = [-1, 0, 1, 127, 255, 2**31 - 1]
ALPHAS = [0, 1, -1, 6553, 65535, 65536, 2**30, 2**31 - 1, -2**31]
NSH = len(SHIFTS)


def limits(rng, dt, shape):
    """Random values with the first elements at the type limits."""
    x = rand_ints(rng, dt, shape)
    info = np.iinfo(dt)
    flat = x.reshape(-1)
    flat[:4] = [info.min, info.max, 0, info.min]
    return x


def same_at(got, want, what):
    try:
        assert_same(got, want)
    except AssertionError as e:
        raise AssertionError(f"{what}: {e}") from None


def pick(rng, table, n):
    return [table[i] for i in rng.integers(0, len(table), n)]


@pytest.mark.parametrize("ifm_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("out_dt", [np.int8, np.uint8])
def test_conv_special(ifm_dt, out_dt):
    rng = np.random.default_rng(2060)
    kh, kw, s = 3, 3, 1
    h, w, ci, co = 6, 6, 8, 16
    ifm, wts = limits(rng, ifm_dt, (h, w, ci)), limits(rng, np.int8, (kh, kw, ci, co))
    bias = limits(rng, np.int16, (co,))
    n = ((h - kh) // s + 1) * ((w - kw) // s + 1) * co
    head = (ifm, T[ifm_dt], wts, bias, OUT, T[out_dt], I64(h), I64(w), I64(ci), I64(kh), I64(kw),
            I64(co), s, s)
    seen = set()
    for t in range(3 * NSH):
        sb, so, sl = SHIFTS[t % NSH], SHIFTS[(7 * t + 3) % NSH], SHIFTS[(5 * t + 1) % NSH]
        mx, alpha = MAXV[t % len(MAXV)], ALPHAS[t % len(ALPHAS)]
        for lr in (0, 1):
            args = (sb, so, mx, lr, alpha, sl)
            want = gx.xint8_conv(ifm, wts, bias, out_dt, s, s, *args)
            same_at(run("npu_xint8_conv", *head, *args, n=n, dt=out_dt), want, args)
            seen.update(np.unique(want).tolist())
        want = gx.xint8_conv(ifm, wts, bias, out_dt, s, s, sb, so, mx, 0, 0, 0)
        for name in ("npu_xint8_conv_pin_buffer", "npu_xint8_conv_pin_stream"):
            same_at(run(name, *head, sb, so, mx, n=n, dt=out_dt), want, (name, sb, so, mx))
        want = gx.xint8_conv(ifm, wts[:1, :1], bias, np.int8, 1, 1, sb, so, -1, 0, 0, 0)
        got = run("npu_xint8_conv1x1ci64co64", ifm, T[ifm_dt], wts[0, 0], bias, OUT, I64(h), I64(w),
                  I64(ci), I64(co), sb, so, n=h * w * co, dt=np.int8)
        same_at(got, want, ('1x1', sb, so))
    assert len(seen) > 8


@pytest.mark.parametrize("dts", DWC_DTS)
def test_depthwise_special(dts):
    ifm_dt, wts_dt, out_dt = dts
    rng = np.random.default_rng(2061)
    kh = kw = 7
    h, w, c = 8, 9, 16
    ifm, wts = limits(rng, ifm_dt, (h, w, c)), limits(rng, wts_dt, (kh, kw, c))
    bias = limits(rng, np.int16, (c,))
    n = (h - kh + 1) * (w - kw + 1) * c
    head = (ifm, T[ifm_dt], wts, T[wts_dt], bias, OUT, T[out_dt], I64(h), I64(w), I64(c))
    for t in range(3 * NSH):
        sb, so, mx = SHIFTS[t % NSH], SHIFTS[(7 * t + 3) % NSH], MAXV[t % len(MAXV)]
        got = run("npu_xint8_depthwise_conv", *head, I64(kh), I64(kw), 1, 1, sb, so, mx, n=n,
                  dt=out_dt)
        same_at(got, gx.xint8_depthwise(ifm, wts, bias, out_dt, 1, 1, sb, so, mx, 0), (sb, so, mx))
        got = run("npu_xint8_dwc7x7_s1", *head, sb, so, mx, n=n, dt=out_dt)
        same_at(got, gx.xint8_depthwise(ifm, wts, bias, out_dt, 1, 1, sb, so, mx, 1), (sb, so, mx))


def test_dwc_pw_special():
    rng = np.random.default_rng(2062)
    h, w, ci, co = 6, 6, 16, 16
    ifm = limits(rng, np.int8, (h, w, ci))
    dw, db = limits(rng, np.int8, (3, 3, ci)), limits(rng, np.int16, (ci,))
    pw, pb = limits(rng, np.int8, (ci, co)), limits(rng, np.int16, (co,))
    n = (h - 2) * (w - 2) * co
    for t in range(3 * NSH):
        args = [SHIFTS[(k * t + 3 * k) % NSH] for k in (1, 3, 5, 7)]
        got = run("npu_xint8_dwc3x3_conv1x1_fused", ifm, dw, db, pw, pb, OUT, I64(h), I64(w),
                  I64(ci), I64(co), I64(3), I64(3), 1, 1, *args, n=n, dt=np.int8)
        same_at(got, gx.xint8_dwc_pw(ifm, dw, db, pw, pb, 1, 1, *args), args)


@pytest.mark.parametrize("out_dt", [np.int8, np.uint8])
def test_conv_add_special(out_dt):
    """Every shift word from the table: the shifted terms saturate at int64 and are added."""
    rng = np.random.default_rng(2063)
    kh = kw = 3
    h, w, ci, co = 5, 5, 8, 16
    ho, wo = h - kh + 1, w - kw + 1
    ifm, wts = limits(rng, np.int8, (h, w, ci)), limits(rng, np.int8, (kh, kw, ci, co))
    bias = limits(rng, np.int16, (co,))
    ifm2, x0 = limits(rng, np.int8, (ho, wo, co)), limits(rng, np.int8, (ho, wo, co))
    geo = (I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), 1, 1)
    n = ho * wo * co
    for t in range(6 * NSH):
        sb, scq, si2, so = pick(rng, SHIFTS, 4)
        sln, slp, sai, sar = pick(rng, SHIFTS, 4)
        mx, alpha = pick(rng, MAXV, 1)[0], pick(rng, ALPHAS, 1)[0]
        for lr in (0, 1):
            args = (sb, scq, si2, so, mx, lr, alpha, sln, slp, sai, sar)
            want = gx.xint8_conv_add(ifm, wts, bias, ifm2, None, out_dt, 1, 1, *args, 0, 0, 0)
            got = run("npu_xint8_conv_add", ifm, wts, bias, ifm2, OUT, T[out_dt], *geo, *args, n=n,
                      dt=out_dt)
            same_at(got, want, args)
        if scq >= 0:
            mc, mx0, mso = pick(rng, ALPHAS, 1)[0], pick(rng, SHIFTS, 1)[0], pick(rng, SHIFTS, 1)[0]
            mac = (sb, scq, si2, so, mc, mx0, mso)
            want = gx.xint8_conv_add(ifm, wts, bias, ifm2, x0, np.int8, 1, 1, sb, scq, si2, so, -1,
                                     0, 0, 0, 0, 0, 0, mc, mx0, mso)
            got = run("npu_xint8_conv_add_mac", ifm, wts, bias, ifm2, x0, OUT, *geo, *mac, n=n,
                      dt=np.int8)
            same_at(got, want, mac)


@pytest.mark.parametrize("ifm_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("out_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("act", [0, 1, 2, 3])
def test_qxint8_special(ifm_dt, out_dt, act):
    """bias_shift / shift over the table (a negative bias_shift shifts right, floor), zero point
    and max_value at the int limits (ReLU with zp_out > max_value included)."""
    rng = np.random.default_rng(2064)
    kh = kw = 3
    h, w, ci, co = 5, 5, 8, 16
    ifm, wts = limits(rng, ifm_dt, (h, w, ci)), limits(rng, np.int8, (kh, kw, ci, co))
    bias = limits(rng, np.int16, (co,))
    n = (h - kh + 1) * (w - kw + 1) * co
    zps = [0, 1, -1, 127, 128, 255, -128, 2**31 - 1, -2**31]
    for t in range(4 * NSH):
        bs, sh = SHIFTS[t % NSH], SHIFTS[(7 * t + 3) % NSH]
        zp, mx, alpha = zps[t % len(zps)], zps[(3 * t + 1) % len(zps)], ALPHAS[t % len(ALPHAS)]
        args = (bs, sh, act, zp, mx, alpha, 0.0, 0.0)
        got = run("npu_qxint8_conv", ifm, T[ifm_dt], wts, bias, OUT, T[out_dt], I64(h), I64(w),
                  I64(ci), I64(kh), I64(kw), I64(co), 1, 1, *args, n=n, dt=out_dt)
        same_at(got, gx.qxint8_conv(ifm, wts, bias, out_dt, 1, 1, *args), args)
        got = run("npu_qxint8_conv", ifm, T[ifm_dt], wts, None, OUT, T[out_dt], I64(h), I64(w),
                  I64(ci), I64(kh), I64(kw), I64(co), 1, 1, *args, n=n, dt=out_dt)
        same_at(got, gx.qxint8_conv(ifm, wts, None, out_dt, 1, 1, *args), ('nobias',) + args)


@pytest.mark.parametrize("out_dt", [np.int8, np.uint8])
def test_qxint8_gelu_special(out_dt):
    """GeLU with NaN / inf / zero / subnormal scales and the zero point at the int limits."""
    from test_golden_qdq_gemm import SPECIAL
    f32 = np.float32
    rng = np.random.default_rng(2065)
    kh = kw = 3
    h, w, ci, co = 5, 5, 8, 16
    ifm, wts = limits(rng, np.int8, (h, w, ci)), limits(rng, np.int8, (kh, kw, ci, co))
    bias = limits(rng, np.int16, (co,))
    n = (h - kh + 1) * (w - kw + 1) * co
    for i, sc in enumerate(SPECIAL):
        for inv in (SPECIAL[(5 * i + 2) % len(SPECIAL)], f32(1) / f32(sc) if sc else f32(0), f32(1.0)):
            for zp in (0, 127, -128, 2**31 - 1, -2**31):
                for sh in (7, 0, -3):
                    args = (3, sh, 6, zp, 127, 0, float(sc), float(inv))
                    got = run("npu_qxint8_conv", ifm, T[np.int8], wts, bias, OUT, T[out_dt], I64(h),
                              I64(w), I64(ci), I64(kh), I64(kw), I64(co), 1, 1, *args, n=n, dt=out_dt)
                    same_at(got, gx.qxint8_conv(ifm, wts, bias, out_dt, 1, 1, *args), args)
    for sh in (150, -150, 2**31 - 1, -2**31):       # zero-point offset and 2^-shift in fp32 under/overflow
        for zp in (0, 1, -1, 2**31 - 1, -2**31):
            for alpha in (0, 21845, 65536, 2**31 - 1):
                args = (0, sh, 3, zp, 127, alpha, 0.0, 0.0)
                got = run("npu_qxint8_conv", ifm, T[np.int8], wts, bias, OUT, T[out_dt], I64(h),
                          I64(w), I64(ci), I64(kh), I64(kw), I64(co), 1, 1, *args, n=n, dt=out_dt)
                same_at(got, gx.qxint8_conv(ifm, wts, bias, out_dt, 1, 1, *args), args)
