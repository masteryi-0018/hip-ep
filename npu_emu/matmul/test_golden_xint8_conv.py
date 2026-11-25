# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The xint8 convolution entry points against golden_xint8_conv.py, bit for bit."""

import numpy as np
import pytest

import golden_xint8_conv as gx
from golden_int_round import Rnd, shift_round
from golden_mac import fma_q
from golden_act import hw_gelu_f32
from npu import I64, NPU_F32, NPU_I8, NPU_I16, NPU_I32, NPU_U8, OUT, raw_rc
from testing import assert_same, rand_ints
from testing import call_n as run

RNG = np.random.default_rng(2031)
T = {np.int8: NPU_I8, np.uint8: NPU_U8}


def ties(a, shift):
    """Fraction of accumulators exactly halfway between two outputs."""
    return float(np.mean((np.asarray(a, np.int64) & ((1 << shift) - 1)) == (1 << (shift - 1))))


GEOM = [(3, 3, 1), (3, 3, 2), (1, 1, 1), (5, 5, 1), (7, 7, 2), (2, 4, 1)]  # kh, kw, stride
ACT = [  # out dtype, max_value, activation
    (np.int8, 0, 0), (np.uint8, 127, 2), (np.uint8, 37, 1), (np.int8, 0, 3)]
NO_GELU = (0.0, 0.0, 0.0) + (0,) * 8
# xgelu words: up_path, rnd_id, a_shift, b_shift, q4_shift, r_shift, K, P
XGELU = [(1, 1, 1, 1, 3, 0, 20, 5), (0, 2, 0, 2, 4, 1, 37, 9), (1, 0, 2, 0, 5, 0, 12, 3)]


def ofm(h, w, kh, kw, s):
    return ((h - kh) // s + 1) * ((w - kw) // s + 1)


@pytest.mark.parametrize("geom", GEOM)
@pytest.mark.parametrize("act", range(len(ACT)))
def test_conv(geom, act):
    kh, kw, s = geom
    h, w, ci, co = 9 + kh, 8 + kw, 24, 16
    ifm = rand_ints(RNG, np.int8, (h, w, ci), -40, 40)
    wts = rand_ints(RNG, np.int8, (kh, kw, ci, co), -6, 6)
    bias = rand_ints(RNG, np.int16, (co,), -3000, 3000)
    out_dt, mx, a = ACT[act]
    for shift_bias, shift_out, alpha, slp in ((2, 5, 6553, 5), (0, 3, 26214, 2), (1, 0, 32767, -1)):
        n = ofm(h, w, kh, kw, s) * co
        head = (ifm, wts, bias, OUT, T[out_dt], I64(h), I64(w), I64(ci), I64(kh), I64(kw),
                I64(co), s, s)
        args = (alpha, mx, shift_bias, slp, shift_out, a)
        want = gx.xint8_conv(ifm, wts, bias, out_dt, s, s, *args)
        assert_same(run("npu_xint8_conv", *head, *args, *NO_GELU, n=n, dt=out_dt), want)
        if a != 3:  # the pinned kernels have no LeakyReLU
            for name in ("npu_xint8_conv_pin_buffer", "npu_xint8_conv_pin_stream"):
                pin = (mx, shift_bias, shift_out, a)
                assert_same(run(name, *head, *pin, *NO_GELU, n=n, dt=out_dt), want)
                model = getattr(gx, name[4:])
                assert_same(want, model(ifm, wts, bias, out_dt, s, s, *pin))
    if a == 0:
        acc = gx._conv(ifm, wts, s, s) + gx._ups_bias(bias, co, 2)
        assert ties(acc, 5) > 0.01


@pytest.mark.parametrize("act", [6, 10])
@pytest.mark.parametrize("k", range(3))
def test_conv_gelu(act, k):
    """The GeLU chains: the int8 conv result through the GeLU kernel."""
    import golden_gelu as gg
    kh, kw, s = 3, 3, 1
    h, w, ci, co = 10, 9, 16, 32
    ifm = rand_ints(RNG, np.int8, (h, w, ci), -40, 40)
    wts = rand_ints(RNG, np.int8, (kh, kw, ci, co), -6, 6)
    bias = rand_ints(RNG, np.int16, (co,), -3000, 3000)
    sc = (0.05, 0.0123, 1 / 3)[k]
    ka, c, kdsp = (float(v) for v in gg.qdq_gelu_coeffs(sc, 0, sc / 2, 0)[2:])
    gelu = (ka, c, kdsp) + XGELU[k]
    n = ofm(h, w, kh, kw, s) * co
    head = (ifm, wts, bias, OUT, T[np.int8], I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co),
            s, s)
    want = gx.xint8_conv(ifm, wts, bias, np.int8, s, s, 0, 0, 1, 0, 6, act, gelu)
    assert_same(run("npu_xint8_conv", *head, 0, 0, 1, 0, 6, act, *gelu, n=n, dt=np.int8), want)
    for name in ("npu_xint8_conv_pin_buffer", "npu_xint8_conv_pin_stream"):
        assert_same(run(name, *head, 0, 1, 6, act, *gelu, n=n, dt=np.int8), want)
    plain = gx.xint8_conv(ifm, wts, bias, np.int8, s, s, 0, 0, 1, 0, 6, 0)
    assert np.any(plain != want)


@pytest.mark.parametrize("act", [0, 10])
def test_conv1x1(act):
    h, w = 7, 9
    for ci, name in ((64, "npu_xint8_conv1x1ci64co64"), (96, "npu_spec_xint8_conv")):
        if act and ci == 64:
            continue
        ifm = rand_ints(RNG, np.int8, (h, w, ci), -40, 40)
        wts = rand_ints(RNG, np.int8, (ci, 64), -6, 6)
        bias = rand_ints(RNG, np.int16, (64,), -3000, 3000)
        for shift_bias, shift_out in ((2, 5), (0, 3), (1, 0)):
            head = (ifm, wts, bias, OUT, I64(h), I64(w), I64(ci), I64(64), shift_bias, shift_out)
            if ci == 64:
                got = run(name, *head, n=h * w * 64, dt=np.int8)
                want = gx.xint8_conv1x1ci64co64(ifm, wts, bias, shift_bias, shift_out)
            else:
                got = run(name, *head, act, *XGELU[0], n=h * w * 64, dt=np.int8)
                want = gx.spec_xint8_conv(ifm, wts, bias, shift_bias, shift_out, act, XGELU[0])
            assert_same(got, want)


DWC_OUT = [np.int8, np.uint8]


@pytest.mark.parametrize("out_dt", DWC_OUT)
@pytest.mark.parametrize("geom", GEOM + [(7, 7, 1)])
@pytest.mark.parametrize("mx,act", [(0, 0), (37, 2), (127, 1)])
def test_depthwise(out_dt, geom, mx, act):
    """The 7 x 7 stride 1 geometry also runs the kernel of that size, which rounds half away."""
    kh, kw, s = geom
    h, w, c = 9 + kh, 10 + kw, 64
    ifm = rand_ints(RNG, np.int8, (h, w, c), -40, 40)
    wts = rand_ints(RNG, np.int8, (kh, kw, c), -6, 6)
    bias = rand_ints(RNG, np.int16, (c,), -2000, 2000)
    for shift_bias, shift_out in ((1, 3), (0, 1), (3, 6)):
        n = ofm(h, w, kh, kw, s) * c
        head = (ifm, wts, bias, OUT, T[out_dt], I64(h), I64(w), I64(c))
        got = run("npu_xint8_depthwise_conv", *head, I64(kh), I64(kw), s, s, shift_bias, shift_out,
                  mx, act, n=n, dt=out_dt)
        assert_same(got, gx.xint8_depthwise_conv(ifm, wts, bias, out_dt, s, s, shift_bias,
                                                 shift_out, mx, act))
        if geom == (7, 7, 1):
            got = run("npu_xint8_dwc7x7_s1", *head, shift_bias, shift_out, mx, act, n=n, dt=out_dt)
            assert_same(got, gx.xint8_dwc7x7_s1(ifm, wts, bias, out_dt, shift_bias, shift_out, mx,
                                                act))


@pytest.mark.parametrize("ci,co,s", [(64, 64, 1), (128, 64, 1), (64, 32, 2)])
def test_dwc_pw(ci, co, s):
    h, w = 10, 18
    ifm = rand_ints(RNG, np.int8, (h, w, ci), -16, 15)
    dw, db = rand_ints(RNG, np.int8, (3, 3, ci), -4, 3), rand_ints(RNG, np.int16, (ci,), -64, 63)
    pw, pb = rand_ints(RNG, np.int8, (ci, co), -4, 3), rand_ints(RNG, np.int16, (co,), -64, 63)
    for args in ((0, 2, 0, 4), (1, 0, 0, 1), (0, 3, 2, 6)):
        n = ofm(h, w, 3, 3, s) * co
        got = run("npu_xint8_dwc3x3_conv1x1_fused", ifm, dw, db, pw, pb, OUT, I64(h), I64(w),
                  I64(ci), I64(3), I64(3), I64(co), s, s, *args, n=n, dt=np.int8)
        assert_same(got, gx.xint8_dwc3x3_conv1x1_fused(ifm, dw, db, pw, pb, s, s, *args))


ADD = [  # has_conv_output_q, shift_in, shift_in1, shift_out, max_value, activation,
    # shift_lrelu_neg, shift_lrelu_pos, shift_res
    (1, 3, 3, 4, 0, 0, 0, 0, 0),
    (2, 0, 0, 2, 0, 0, 0, 0, 0),
    (0, 0, 2, 4, 0, 0, 0, 0, 0),  # conv_add: no intermediate Q
    (0, 0, 3, 3, 23, 2, 0, 0, 0),
    (1, 2, 4, 4, 127, 1, 0, 0, 0),
    (1, 3, 3, 3, 23, 2, 0, 0, 0),
    (0, 0, 1, 14, 0, 3, 4, 2, 1),  # lrelu
    (0, 1, 0, 12, 0, 3, 2, -2, 1),
]


@pytest.mark.parametrize("cfg", range(len(ADD)))
@pytest.mark.parametrize("geom", [(3, 3, 1), (1, 1, 1), (3, 3, 2)])
def test_conv_add(cfg, geom):
    """Configurations without activation and with an intermediate Q also run the mac kernel."""
    q, sin, sin1, so, mx, act, sln, slp, sres = ADD[cfg]
    kh, kw, s = geom
    h, w, ci, co = 8 + kh, 8 + kw, 32, 16
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    ifm = rand_ints(RNG, np.int8, (h, w, ci), -25, 25)
    wts = rand_ints(RNG, np.int8, (kh, kw, ci, co), -5, 5)
    bias = rand_ints(RNG, np.int16, (co,), -400, 400)
    ifm2 = rand_ints(RNG, np.int8, (ho, wo, co), -32, 32)
    geo = (I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s)
    args = (6553, mx, 1, slp, sln, so, sin, sin1, sres, q, act)
    got = run("npu_xint8_conv_add", ifm, ifm2, wts, bias, OUT, *geo, *args, n=ho * wo * co,
              dt=np.int8)
    assert_same(got, gx.xint8_conv_add(ifm, ifm2, wts, bias, s, s, *args))
    if act or not q:
        return
    x0, mc = rand_ints(RNG, np.int8, (ho, wo, co)), (3, 2, 2)
    got = run("npu_xint8_conv_add_mac", ifm, ifm2, wts, bias, x0, OUT, *geo, 1, so, sin, sin1,
              *mc, n=ho * wo * co, dt=np.int8)
    assert_same(got, gx.xint8_conv_add_mac(ifm, ifm2, wts, bias, x0, s, s, 1, so, sin, sin1, *mc))


def qx_args(lrelu_alpha, max_value, shift_bias, shift, zp, act, scale=0.0, inv=0.0):
    """npu_qxint8_conv words for one output shift: shift_lrelu_pos = shift_out = shift and
    output_zero_point_acc = zp * 2^shift."""
    with np.errstate(over="ignore"):
        ozpa = float(np.float32(np.ldexp(np.float64(zp), shift)))
    return (lrelu_alpha, max_value & 0xFF, shift_bias, shift, shift, float(scale), float(inv), zp,
            ozpa, act)


@pytest.mark.parametrize("ifm_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("out_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("act", [0, 1, 2, 3])
@pytest.mark.parametrize("geom", [(3, 3, 1), (1, 1, 1), (3, 3, 2)])
def test_qxint8(ifm_dt, out_dt, act, geom):
    kh, kw, s = geom
    h, w, ci, co = 8 + kh, 9 + kw, 32, 16
    ifm, wts = rand_ints(RNG, ifm_dt, (h, w, ci)), rand_ints(RNG, np.int8, (kh, kw, ci, co), -20, 20)
    bias = rand_ints(RNG, np.int16, (co,), -2000, 2000)
    lo, hi = gx._rng_of(out_dt)
    zp = (lo + hi) // 2 + 3
    mx = hi if act != 2 else zp + 40
    for bias_shift, shift in ((3, 7), (0, 4), (5, 9)):
        n = ofm(h, w, kh, kw, s) * co
        args = qx_args(6553, mx, bias_shift, shift, zp, act)
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
    lo, hi = gx._rng_of(out_dt)
    zp = (lo + hi) // 2 + 3
    n = ofm(h, w, kh, kw, s) * co
    args = qx_args(0, hi, 3, 7, zp, 6, np.float32(scale), np.float32(1.0 / scale))
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
    args = qx_args(0, 127, 0, 0, zp, 6, np.float32(scale), np.float32(1.0 / scale))
    want = gx.qxint8_conv(ifm, wts, None, np.int8, 1, 1, *args)
    real = np.float32((np.float32(r) - np.float32(zp)) * np.float32(scale))
    fused = fma_q(hw_gelu_f32(np.array([real], np.float32)), np.float32(1.0 / scale), np.float32(zp))
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
    head = (ifm, wts, None, OUT, NPU_I8, I64(7), I64(10), I64(1))
    got = run("npu_xint8_depthwise_conv", *head, I64(7), I64(7), 1, 1, 0, 2, 0, 0, n=4, dt=np.int8)
    assert got.tolist() == [0, 2, 0, -2]
    got = run("npu_xint8_dwc7x7_s1", *head, 0, 2, 0, 0, n=4, dt=np.int8)
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
    acc = gx._conv(ifm, wts, 1, 1) + bias.astype(np.int64) * 65536
    assert acc.reshape(-1).tolist() == LRELU_SPLIT
    args = qx_args(alpha_q16, 127, 16, shift, zp, 3)
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
# special inputs are the integer limits: operands at the type limits, shift words at the ends
# of their int8 / uint8 fields (far outside the accumulator width), sums that saturate at
# int64, and for the GeLU entry NaN / inf / zero scales.
SHIFTS = [-128, -70, -65, -64, -63, -33, -32, -31, -8, -1, 0, 1, 8, 15, 16, 31, 32, 33, 62, 63,
          64, 65, 70, 127]
USHIFTS = [0, 1, 8, 15, 16, 31, 32, 33, 62, 63, 64, 65, 70, 127, 200, 255]
MAXV = [0, 1, 37, 127, 128, 200, 255]
ALPHAS = [0, 1, -1, 6553, 21845, 32767, -32768]
COEF8 = [0, 1, -1, 3, 127, -128]
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


@pytest.mark.parametrize("out_dt", [np.int8, np.uint8])
def test_conv_special(out_dt):
    rng = np.random.default_rng(2060)
    kh, kw, s = 3, 3, 1
    h, w, ci, co = 6, 6, 8, 16
    ifm, wts = limits(rng, np.int8, (h, w, ci)), limits(rng, np.int8, (kh, kw, ci, co))
    bias = limits(rng, np.int16, (co,))
    n = ofm(h, w, kh, kw, s) * co
    head = (ifm, wts, bias, OUT, T[out_dt], I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co),
            s, s)
    seen = set()
    for t in range(3 * NSH):
        sb, so, sl = SHIFTS[t % NSH], SHIFTS[(7 * t + 3) % NSH], SHIFTS[(5 * t + 1) % NSH]
        mx, alpha = MAXV[t % len(MAXV)], ALPHAS[t % len(ALPHAS)]
        for act in (0, 1, 2, 3):
            args = (alpha, mx, sb, sl, so, act)
            want = gx.xint8_conv(ifm, wts, bias, out_dt, s, s, *args)
            same_at(run("npu_xint8_conv", *head, *args, *NO_GELU, n=n, dt=out_dt), want, args)
            seen.update(np.unique(want).tolist())
            if act != 3:
                for name in ("npu_xint8_conv_pin_buffer", "npu_xint8_conv_pin_stream"):
                    got = run(name, *head, mx, sb, so, act, *NO_GELU, n=n, dt=out_dt)
                    same_at(got, want, (name,) + args)
    assert len(seen) > 8


def test_conv1x1_special():
    rng = np.random.default_rng(2066)
    h, w = 3, 4
    for ci, name in ((64, "npu_xint8_conv1x1ci64co64"), (96, "npu_spec_xint8_conv")):
        ifm, wts = limits(rng, np.int8, (h, w, ci)), limits(rng, np.int8, (ci, 64))
        bias = limits(rng, np.int16, (64,))
        for t in range(NSH):
            sb, so = SHIFTS[t], SHIFTS[(7 * t + 3) % NSH]
            head = (ifm, wts, bias, OUT, I64(h), I64(w), I64(ci), I64(64), sb, so)
            extra = (0,) + (0,) * 8 if ci == 96 else ()
            got = run(name, *head, *extra, n=h * w * 64, dt=np.int8)
            same_at(got, gx.xint8_conv1x1ci64co64(ifm, wts, bias, sb, so), (name, sb, so))


@pytest.mark.parametrize("out_dt", DWC_OUT)
def test_depthwise_special(out_dt):
    rng = np.random.default_rng(2061)
    kh = kw = 7
    h, w, c = 8, 9, 16
    ifm, wts = limits(rng, np.int8, (h, w, c)), limits(rng, np.int8, (kh, kw, c))
    bias = limits(rng, np.int16, (c,))
    n = (h - kh + 1) * (w - kw + 1) * c
    head = (ifm, wts, bias, OUT, T[out_dt], I64(h), I64(w), I64(c))
    for t in range(3 * NSH):
        sb, so, mx = SHIFTS[t % NSH], SHIFTS[(7 * t + 3) % NSH], MAXV[t % len(MAXV)]
        for act in (0, 2):
            got = run("npu_xint8_depthwise_conv", *head, I64(kh), I64(kw), 1, 1, sb, so, mx, act,
                      n=n, dt=out_dt)
            same_at(got, gx.xint8_depthwise_conv(ifm, wts, bias, out_dt, 1, 1, sb, so, mx, act),
                    (sb, so, mx, act))
            got = run("npu_xint8_dwc7x7_s1", *head, sb, so, mx, act, n=n, dt=out_dt)
            same_at(got, gx.xint8_dwc7x7_s1(ifm, wts, bias, out_dt, sb, so, mx, act),
                    (sb, so, mx, act))


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
                  I64(ci), I64(3), I64(3), I64(co), 1, 1, *args, n=n, dt=np.int8)
        same_at(got, gx.xint8_dwc3x3_conv1x1_fused(ifm, dw, db, pw, pb, 1, 1, *args), args)


def test_conv_add_special():
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
        sb, sin, sin1, so = pick(rng, SHIFTS, 4)
        sln, slp, sres = pick(rng, SHIFTS, 3)
        mx, alpha = pick(rng, MAXV, 1)[0], pick(rng, ALPHAS, 1)[0]
        for act in (0, 1, 2, 3):
            q = 0 if act == 3 else int(rng.integers(0, 3))
            s_in = int(rng.integers(0, 16)) if act == 3 else sin
            args = (alpha, mx, sb, slp, sln, so, s_in, sin1, sres, q, act)
            want = gx.xint8_conv_add(ifm, ifm2, wts, bias, 1, 1, *args)
            got = run("npu_xint8_conv_add", ifm, ifm2, wts, bias, OUT, *geo, *args, n=n,
                      dt=np.int8)
            same_at(got, want, args)
        # the mac kernel: m = min(shift_in, shift_in1, shift_out) >= 0, shift_in - m <= 7
        msin1, mso = pick(rng, USHIFTS[:14], 2)
        msin = min(min(msin1, mso) + int(rng.integers(0, 8)), 127)
        mac = (sb, mso, msin, msin1, pick(rng, COEF8, 1)[0], *pick(rng, USHIFTS, 2))
        want = gx.xint8_conv_add_mac(ifm, ifm2, wts, bias, x0, 1, 1, *mac)
        got = run("npu_xint8_conv_add_mac", ifm, ifm2, wts, bias, x0, OUT, *geo, *mac, n=n,
                  dt=np.int8)
        same_at(got, want, mac)


@pytest.mark.parametrize("ifm_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("out_dt", [np.int8, np.uint8])
@pytest.mark.parametrize("act", [0, 1, 2, 3])
def test_qxint8_special(ifm_dt, out_dt, act):
    """shift_bias / shift over the table (a negative shift_bias shifts right, floor), zero point
    and max_value at the limits of their fields (ReLU with ofm_zero_point > max_value
    included)."""
    rng = np.random.default_rng(2064)
    kh = kw = 3
    h, w, ci, co = 5, 5, 8, 16
    ifm, wts = limits(rng, ifm_dt, (h, w, ci)), limits(rng, np.int8, (kh, kw, ci, co))
    bias = limits(rng, np.int16, (co,))
    n = (h - kh + 1) * (w - kw + 1) * co
    lo, hi = gx._rng_of(out_dt)
    zps = [lo, hi, lo + 1, hi - 1, (lo + hi) // 2, 0, 1] if lo else [0, 1, 127, 128, 254, 255]
    for t in range(4 * NSH):
        bs, sh = SHIFTS[t % NSH], SHIFTS[(7 * t + 3) % NSH]
        zp, mx, alpha = zps[t % len(zps)], MAXV[(3 * t + 1) % len(MAXV)], ALPHAS[t % len(ALPHAS)]
        args = qx_args(alpha, mx, bs, sh, zp, act)
        for b in (bias, None):
            got = run("npu_qxint8_conv", ifm, T[ifm_dt], wts, b, OUT, T[out_dt], I64(h), I64(w),
                      I64(ci), I64(kh), I64(kw), I64(co), 1, 1, *args, n=n, dt=out_dt)
            same_at(got, gx.qxint8_conv(ifm, wts, b, out_dt, 1, 1, *args), (b is None,) + args)


@pytest.mark.parametrize("out_dt", [np.int8, np.uint8])
def test_qxint8_gelu_special(out_dt):
    """GeLU with NaN / inf / zero / subnormal scales and the zero point at the field limits;
    LReLU with the zero-point offset and 2^-shift under / overflowing fp32."""
    from test_golden_qdq_gemm import SPECIAL
    f32 = np.float32
    rng = np.random.default_rng(2065)
    kh = kw = 3
    h, w, ci, co = 5, 5, 8, 16
    ifm, wts = limits(rng, np.int8, (h, w, ci)), limits(rng, np.int8, (kh, kw, ci, co))
    bias = limits(rng, np.int16, (co,))
    n = (h - kh + 1) * (w - kw + 1) * co
    lo, hi = gx._rng_of(out_dt)
    head = (ifm, T[np.int8], wts, bias, OUT, T[out_dt], I64(h), I64(w), I64(ci), I64(kh),
            I64(kw), I64(co), 1, 1)
    for i, sc in enumerate(SPECIAL):
        with np.errstate(all="ignore"):
            invs = (SPECIAL[(5 * i + 2) % len(SPECIAL)], f32(1) / f32(sc) if sc else f32(0),
                    f32(1.0))
        for inv in invs:
            for zp in (0, lo, hi):
                for sh in (7, 0, -3):
                    args = qx_args(0, 127, 3, sh, zp, 6, sc, inv)
                    got = run("npu_qxint8_conv", *head, *args, n=n, dt=out_dt)
                    same_at(got, gx.qxint8_conv(ifm, wts, bias, out_dt, 1, 1, *args), args)
    for sh in (127, -128, 100, -100):
        for zp in (0, 1, lo, hi):
            for alpha in (0, 21845, 32767, -32768):
                args = qx_args(alpha, 127, 0, sh, zp, 3)
                got = run("npu_qxint8_conv", *head, *args, n=n, dt=out_dt)
                same_at(got, gx.qxint8_conv(ifm, wts, bias, out_dt, 1, 1, *args), args)


def test_bad_args():
    """Every entry returns -1 for arguments its kernel cannot take, before touching memory."""
    p = np.zeros(1, np.int8)   # placeholder: these calls must return before any access
    I = I64  # noqa: E741  (a short alias keeps the argument tables compact)
    def conv(**k):
        return ("npu_xint8_conv", p, p, None, p, k.get("ot", NPU_I8), I(4), I(4), I(2),
                I(k.get("kh", 3)), I(3), I(2), k.get("s", 1), 1, k.get("al", 0),
                k.get("mx", 0), k.get("sb", 0), 0, 0, k.get("act", 0), 0.0, 0.0, 0.0,
                *k.get("xg", (0,) * 8))
    for k in (dict(ot=NPU_I16), dict(ot=NPU_F32), dict(s=0), dict(s=3), dict(kh=5),
              dict(kh=0), dict(al=32768), dict(mx=256), dict(mx=-1), dict(sb=128), dict(act=4),
              dict(act=11), dict(act=6, ot=NPU_U8), dict(act=10, xg=(2, 0, 0, 0, 0, 0, 0, 0)),
              dict(act=10, xg=(1, 0, 24, 0, 0, 0, 0, 0)),
              dict(act=10, xg=(1, 0, 0, 0, 0, 0, 128, 0))):
        assert raw_rc(*conv(**k)) == -1, k
    assert raw_rc("npu_xint8_conv", None, p, None, p, NPU_I8, I(4), I(4), I(2), I(3), I(3), I(2),
                  1, 1, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, *(0,) * 8) == -1
    for name in ("npu_xint8_conv_pin_buffer", "npu_xint8_conv_pin_stream"):
        assert raw_rc(name, p, p, None, p, NPU_I8, I(4), I(4), I(2), I(3), I(3), I(2), 1, 1, 0,
                      0, 0, 3, 0.0, 0.0, 0.0, *(0,) * 8) == -1   # no LReLU
        assert raw_rc(name, p, p, None, p, NPU_I8, I(1), I(4), I(2), I(3), I(3), I(2), 1, 1, 0,
                      0, 0, 0, 0.0, 0.0, 0.0, *(0,) * 8) == -1   # h < kh
    assert raw_rc("npu_xint8_conv1x1ci64co64", p, p, None, p, I(2), I(2), I(32), I(64), 0,
                  0) == -1
    assert raw_rc("npu_spec_xint8_conv", p, p, None, p, I(2), I(2), I(96), I(64), 0, 0, 6,
                  *(0,) * 8) == -1
    assert raw_rc("npu_spec_xint8_conv", p, p, None, p, I(2), I(2), I(64), I(64), 0, 0, 0,
                  *(0,) * 8) == -1
    def dw(name, **k):
        return ((name, p, p, None, p, k.get("ot", NPU_I8), I(k.get("h", 8)), I(8), I(4))
                + ((I(3), I(3), k.get("s", 1), 1) if "depthwise" in name else ())
                + (k.get("sb", 0), 0, k.get("mx", 0), k.get("act", 0)))
    for name in ("npu_xint8_depthwise_conv", "npu_xint8_dwc7x7_s1"):
        for k in (dict(ot=NPU_I32), dict(h=2), dict(sb=-129), dict(mx=300), dict(act=3)):
            assert raw_rc(*dw(name, **k)) == -1, (name, k)
    assert raw_rc(*dw("npu_xint8_depthwise_conv", s=0)) == -1
    assert raw_rc("npu_xint8_dwc3x3_conv1x1_fused", p, p, None, p, None, p, I(4), I(4), I(2),
                  I(3), I(3), I(2), 0, 1, 0, 0, 0, 0) == -1
    assert raw_rc("npu_xint8_dwc3x3_conv1x1_fused", p, p, None, None, None, p, I(4), I(4), I(2),
                  I(3), I(3), I(2), 1, 1, 0, 0, 0, 0) == -1
    def add(**k):
        return ("npu_xint8_conv_add", p, k.get("i2", p), p, None, p, I(4), I(4), I(2),
                I(3), I(3), I(2), 1, 1, 0, 0, 0, 0, 0, 0, k.get("sin", 0), 0, 0,
                k.get("q", 0), k.get("act", 0))
    for k in (dict(i2=None), dict(q=3), dict(q=-1), dict(act=4), dict(act=3, q=1),
              dict(act=3, sin=16), dict(act=3, sin=-1), dict(sin=200)):
        assert raw_rc(*add(**k)) == -1, k
    def mac(x0=p, sin=0, sin1=0, so=0, mc=0, sx=0):
        return ("npu_xint8_conv_add_mac", p, p, p, None, x0, p, I(4), I(4), I(2), I(3), I(3), I(2),
                1, 1, 0, so, sin, sin1, mc, sx, 0)
    for k in (dict(x0=None), dict(sin=8), dict(sin=-1), dict(mc=128), dict(sx=-1), dict(sx=256)):
        assert raw_rc(*mac(**k)) == -1, k
    def qx(**k):
        return ("npu_qxint8_conv", p, k.get("it", NPU_I8), p, None, p, k.get("ot", NPU_I8),
                I(4), I(4), I(2), I(3), I(3), I(2), k.get("s", 1), 1, 0, 0, 0, 0, 0, 0.0,
                0.0, k.get("zp", 0), 0.0, k.get("act", 0))
    for k in (dict(it=NPU_I16), dict(ot=NPU_F32), dict(s=0), dict(act=4), dict(act=10),
              dict(zp=128), dict(ot=NPU_U8, zp=-1)):
        assert raw_rc(*qx(**k)) == -1, k
    _sizes(p)


def _sizes(p):
    """H or W = 0 is an empty output: 0 without a launch once every other argument is valid;
    negative sizes and null required pointers stay -1, also when empty."""
    I, g8 = I64, (0,) * 8  # noqa: E741

    def calls(ifm=p, out=p, h=0, w=4):
        hw = (I(h), I(w))
        return [
            ("npu_xint8_conv", ifm, p, None, out, NPU_I8, *hw, I(2), I(3), I(3), I(2), 1, 1, 0, 0,
             0, 0, 0, 0, 0.0, 0.0, 0.0, *g8),
            ("npu_xint8_conv", ifm, p, None, out, NPU_I8, *hw, I(2), I(3), I(3), I(2), 1, 1, 0, 0,
             0, 0, 0, 6, 0.0, 0.0, 0.0, *g8),
            ("npu_xint8_conv_pin_buffer", ifm, p, None, out, NPU_I8, *hw, I(2), I(3), I(3), I(2),
             1, 1, 0, 0, 0, 0, 0.0, 0.0, 0.0, *g8),
            ("npu_xint8_conv_pin_stream", ifm, p, None, out, NPU_I8, *hw, I(2), I(3), I(3), I(2),
             1, 1, 0, 0, 0, 0, 0.0, 0.0, 0.0, *g8),
            ("npu_xint8_conv1x1ci64co64", ifm, p, None, out, *hw, I(64), I(64), 0, 0),
            ("npu_spec_xint8_conv", ifm, p, None, out, *hw, I(96), I(64), 0, 0, 0, *g8),
            ("npu_xint8_depthwise_conv", ifm, p, None, out, NPU_I8, *hw, I(4), I(3), I(3), 1, 1,
             0, 0, 0, 0),
            ("npu_xint8_dwc7x7_s1", ifm, p, None, out, NPU_I8, *hw, I(4), 0, 0, 0, 0),
            ("npu_xint8_dwc3x3_conv1x1_fused", ifm, p, None, p, None, out, *hw, I(2), I(3), I(3),
             I(2), 1, 1, 0, 0, 0, 0),
            ("npu_xint8_conv_add", ifm, p, p, None, out, *hw, I(2), I(3), I(3), I(2), 1, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0),
            ("npu_xint8_conv_add_mac", ifm, p, p, None, p, out, *hw, I(2), I(3), I(3), I(2), 1, 1,
             0, 0, 0, 0, 0, 0, 0),
            ("npu_qxint8_conv", ifm, NPU_I8, p, None, out, NPU_I8, *hw, I(2), I(3), I(3), I(2), 1,
             1, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0.0, 0),
        ]

    for args in calls():
        assert raw_rc(*args) == 0, args[0]
    for args in calls(h=8, w=0):
        assert raw_rc(*args) == 0, args[0]
    for bad in (dict(h=-1), dict(h=8, w=-1), dict(ifm=None), dict(out=None)):
        for args in calls(**bad):
            assert raw_rc(*args) == -1, (args[0], bad)
