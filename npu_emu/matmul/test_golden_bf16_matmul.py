# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The bf16 matmul / convolution entry points against the numpy reference, bit for bit."""

import numpy as np
import pytest

import golden_bf16_matmul as gb
from golden_float_cvt import bf16_to_f32, f32_to_bf16, kFtzIn
from golden_int_round import Rnd
from golden_mmac import bf16_bits
from npu import I64, OUT
from testing import assert_same, call_n, rejected

RNG = np.random.default_rng(2041)


def bf16(shape, spread=6):
    """bf16 bits of random values with exponents spread over +-spread octaves."""
    v = RNG.standard_normal(shape) * np.exp2(RNG.integers(-spread, spread + 1, shape))
    return bf16_bits(v.astype(np.float32))


@pytest.mark.parametrize("mkn", [(8, 32, 16), (5, 100, 7), (16, 256, 64), (3, 17, 9)])
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("act", [0, 1, 2])
@pytest.mark.parametrize("spread", [2, 12])
def test_gemm(mkn, bias, act, spread):
    m, k, n = mkn
    a, b = bf16((m, k), spread), bf16((k, n), spread)
    bi = (RNG.standard_normal(n) * 4).astype(np.float32) if bias else None
    want = gb.gemm(a, b, bi, act).reshape(m, n)
    for name in ("npu_gemm_bf16x16", "npu_gemm_bf16x16_buffer", "npu_gemm_bf16x16_ifm_reuse"):
        assert_same(call_n(name, a, b, bi, OUT, I64(m), I64(k), I64(n), act, n=m * n, dt=np.uint16), want)
    got = call_n("npu_gemm_bf16x16_transposed", a, b, bi, OUT, I64(m), I64(k), I64(n), act, n=m * n, dt=np.uint16)
    assert_same(got, want.T)


def test_mmac_cancellation():
    """Large terms that cancel inside one half leave the small terms."""
    x = np.full((1, 32), bf16_bits(np.float32(1.0)), np.uint16)
    w = bf16_bits(np.array([2.0**20, -2.0**20, 3.0, 2.0**-8] * 8, np.float32)).reshape(32, 1)
    bi = np.array([1e-3], np.float32)
    got = call_n("npu_gemm_bf16x16", x, w, bi, OUT, I64(1), I64(32), I64(1), 0, n=1, dt=np.uint16)
    assert_same(got, gb.gemm(x, w, bi, 0))


@pytest.mark.parametrize("signed,scale,zp", [(0, 37.3, 0), (1, 0.0123, 5), (0, 0.5, 30000)])
def test_gemm_qdq(signed, scale, zp):
    m, k, n = 6, 96, 20
    a, b = bf16((m, k), 3), bf16((k, n), 3)
    bi = (RNG.standard_normal(n)).astype(np.float32)
    inv = float(np.float32(1.0 / scale))
    want = gb.gemm_qdq(a, b, bi, 1, signed, inv, scale, zp)
    for name in ("npu_gemm_quant_dequant_bf16x16", "npu_gemm_quant_dequant_bf16x16_ifm_reuse"):
        got = call_n(name, a, b, bi, OUT, I64(m), I64(k), I64(n), 1, signed, inv,
                     float(np.float32(scale)), zp, n=m * n, dt=np.uint16)
        assert_same(got, want)


CONV = [  # kh, kw, ci, co, stride, kh_sv
    (3, 3, 32, 16, 1, 3), (3, 3, 64, 8, 2, 1), (1, 1, 96, 16, 1, 1), (3, 3, 16, 8, 1, 1),
    (5, 5, 16, 8, 1, 5), (3, 3, 4, 8, 1, 3), (2, 4, 48, 8, 2, 2), (1, 3, 8, 16, 1, 1)]


@pytest.mark.parametrize("geom", CONV)
@pytest.mark.parametrize("act", [0, 1, 2, 3])
def test_conv(geom, act):
    kh, kw, ci, co, s, kh_sv = geom
    h, w = (3 - 1) * s + kh, (4 - 1) * s + kw
    ifm, wts = bf16((h, w, ci), 6), bf16((kh, kw, ci, co), 6)
    bi = (RNG.standard_normal(co) * 2).astype(np.float32)
    want = gb.conv(ifm, wts, bi, s, s, kh_sv, act, 0.1015625)
    args = (ifm, wts, bi, OUT, I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s,
            I64(kh_sv), act)
    assert_same(call_n("npu_conv_bf16x16_buffer", *args, 0.1015625, n=want.size, dt=np.uint16), want)
    if act == 3:  # only the buffer kernel has LeakyReLU
        assert rejected("npu_conv_bf16x16", *args, nbytes=want.size * 2)
    else:
        assert_same(call_n("npu_conv_bf16x16", *args, n=want.size, dt=np.uint16), want)


@pytest.mark.parametrize("geom", CONV[:5])
@pytest.mark.parametrize("typ,act", [(0, 0), (1, 0), (1, 1)])
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("bshape", ["full", "chan", "pixel"])
def test_conv_add(geom, typ, act, bias, bshape):
    kh, kw, ci, co, s, kh_sv = geom
    ho, wo = 3, 4
    h, w = (ho - 1) * s + kh, (wo - 1) * s + kw
    ifm, wts = bf16((h, w, ci), 6), bf16((kh, kw, ci, co), 6)
    shp = {"full": (ho, wo, co), "chan": (1, 1, co), "pixel": (ho, wo, 1)}[bshape]
    ifm2 = bf16(shp, 4)
    bi = (RNG.standard_normal(co) * 2).astype(np.float32) if bias else None
    want = gb.conv_add(ifm, wts, bi, ifm2, s, s, kh_sv, typ, act)
    got = call_n("npu_conv_add_bf16x16", ifm, wts, bi, ifm2, OUT, I64(h), I64(w), I64(ci), I64(kh),
                 I64(kw), I64(co), s, s, I64(kh_sv), I64(shp[0]), I64(shp[1]), I64(shp[2]), typ, act,
                 n=want.size, dt=np.uint16)
    assert_same(got, want)


@pytest.mark.parametrize("kk,s", [((3, 3), 1), ((3, 3), 2), ((5, 5), 1), ((7, 7), 2), ((1, 1), 1)])
@pytest.mark.parametrize("act", [0, 1, 2, 3])
@pytest.mark.parametrize("bias", [0, 1])
def test_depthwise(kk, s, act, bias):
    kh, kw = kk
    c = 64
    h, w = (4 - 1) * s + kh, (5 - 1) * s + kw
    ifm, wts = bf16((h, w, c), 6), bf16((kh, kw, c), 6)
    bi = bf16_to_f32(bf16((c,), 2)) if bias else None
    want = gb.depthwise(ifm, wts, bi, s, s, act, 0.1015625)
    for name in ("npu_depthwise_conv_bf16x16", "npu_depthwise_conv_bf16x16_c32"):
        got = call_n(name, ifm, wts, bi, OUT, I64(h), I64(w), I64(c), I64(kh), I64(kw), s, s, act,
                     0.1015625, n=want.size, dt=np.uint16)
        assert_same(got, want)


def test_depthwise_bias_first():
    """The accumulator starts at the bias: 257 + 49 taps of -2^-16 stays 257."""
    kh = kw = 7
    c = 64
    ifm = np.full((kh, kw, c), 0x3B80, np.uint16)              # 2^-8
    wts = np.full((kh, kw, c), 0xBB80, np.uint16)              # -2^-8
    bi = np.full(c, 257.0, np.float32)
    want = gb.depthwise(ifm, wts, bi, 1, 1, 0)
    assert (want == 0x4381).all()                              # bf16 258
    got = call_n("npu_depthwise_conv_bf16x16", ifm, wts, bi, OUT, I64(kh), I64(kw), I64(c), I64(kh),
                 I64(kw), 1, 1, 0, 0.0, n=want.size, dt=np.uint16)
    assert_same(got, want)


# (inv_scale, zero point, bf16 bits) where v * inv + zp rounded once and twice
# quantize differently
QDQ_FMA = [(1 / 0.0234375, 5, [0xBEEA, 0xBF49, 0xBE34]), (1 / 0.0234375, -7, [0x3F61, 0x3F01, 0x3F49])]


@pytest.mark.parametrize("inv,zp,bits", QDQ_FMA)
def test_gemm_qdq_one_rounding(inv, zp, bits):
    """A K = 1 product with 1.0 feeds v into the quantize, one MAC."""
    m = len(bits)
    a = np.array(bits, np.uint16).reshape(m, 1)
    b = np.array([[0x3F80]], np.uint16)
    inv, scale = np.float32(inv), np.float32(1 / inv)
    want = gb.gemm_qdq(a, b, None, 0, 1, inv, scale, zp)
    v = bf16_to_f32(a.reshape(-1))
    two = np.rint(((v * inv).astype(np.float32) + np.float32(zp)).astype(np.float32).astype(np.float64))
    zt = np.float64(np.float32(np.float32(-zp) * scale))
    old = f32_to_bf16((two * np.float64(scale) + zt).astype(np.float32), Rnd.kHalfEven, kFtzIn)
    assert (old != want.reshape(-1)).all()   # the data separates the two
    for name in ("npu_gemm_quant_dequant_bf16x16", "npu_gemm_quant_dequant_bf16x16_ifm_reuse"):
        got = call_n(name, a, b, None, OUT, I64(m), I64(1), I64(1), 0, 1, float(inv), float(scale), zp,
                     n=m, dt=np.uint16)
        assert_same(got, want)


TIE_ALPHA = float(np.uint32(0x3DD08000).view(np.float32))   # halfway between two bf16
AWAY_ALPHA = float(np.uint32(0x3DD10000).view(np.float32))  # the tie rounded away


def test_conv_lrelu_alpha_tie():
    """The alpha is cast to bf16 half to even."""
    kh, kw, ci, co, s, kh_sv = CONV[0]
    h, w = 2 + kh, 3 + kw
    ifm, wts = bf16((h, w, ci), 6), bf16((kh, kw, ci, co), 6)
    bi = (RNG.standard_normal(co) * 2).astype(np.float32)
    want = gb.conv(ifm, wts, bi, s, s, kh_sv, 3, TIE_ALPHA)
    assert (want != gb.conv(ifm, wts, bi, s, s, kh_sv, 3, AWAY_ALPHA)).any()
    got = call_n("npu_conv_bf16x16_buffer", ifm, wts, bi, OUT, I64(h), I64(w), I64(ci), I64(kh),
                 I64(kw), I64(co), s, s, I64(kh_sv), 3, TIE_ALPHA, n=want.size, dt=np.uint16)
    assert_same(got, want)


def test_depthwise_lrelu_alpha_tie():
    kh = kw = 3
    c = 64
    h, w = 4 + kh, 5 + kw
    ifm, wts = bf16((h, w, c), 6), bf16((kh, kw, c), 6)
    want = gb.depthwise(ifm, wts, None, 1, 1, 3, TIE_ALPHA)
    assert (want != gb.depthwise(ifm, wts, None, 1, 1, 3, AWAY_ALPHA)).any()
    for name in ("npu_depthwise_conv_bf16x16", "npu_depthwise_conv_bf16x16_c32"):
        got = call_n(name, ifm, wts, None, OUT, I64(h), I64(w), I64(c), I64(kh), I64(kw), 1, 1, 3,
                     TIE_ALPHA, n=want.size, dt=np.uint16)
        assert_same(got, want)


# ---- special values: NaN (quiet, signalling, payload), +-inf, +-0, subnormals, largest finite ----

SPECIAL_BF16 = np.array([0x7FC0, 0xFFC0, 0x7FC1, 0x7F81, 0xFF81, 0x7F80, 0xFF80, 0x0000, 0x8000,
                         0x007F, 0x807F, 0x0001, 0x8001, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80], np.uint16)
SPECIAL_F32 = np.array([np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-40, -1e-40, 1e-45, 3.4028235e38,
                        -3.4028235e38, 1.0, -1.0], np.float32)
SPECIAL_F32 = np.concatenate([SPECIAL_F32, np.array([0x7FC12345, 0xFFA00001], np.uint32).view(np.float32)])


def sp_bf16(shape, rate=0.3):
    """Random bf16 codes with a share of the special ones."""
    v = bf16(shape, 3).reshape(-1)
    pos = RNG.random(v.size) < rate
    v[pos] = SPECIAL_BF16[RNG.integers(0, SPECIAL_BF16.size, int(pos.sum()))]
    return v.reshape(shape)


def sp_f32(n, rate=0.4):
    """fp32 values (bias / coefficients) with a share of the special ones."""
    v = (RNG.standard_normal(n) * 4).astype(np.float32)
    pos = RNG.random(n) < rate
    v[pos] = SPECIAL_F32[RNG.integers(0, SPECIAL_F32.size, int(pos.sum()))]
    return v


@pytest.mark.parametrize("mkn", [(8, 32, 16), (5, 70, 9)])
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("act", [0, 1, 2])
@pytest.mark.parametrize("rep", range(3))
def test_special_gemm(mkn, bias, act, rep):
    m, k, n = mkn
    a, b = sp_bf16((m, k)), sp_bf16((k, n))
    bi = sp_f32(n) if bias else None
    want = gb.gemm(a, b, bi, act).reshape(m, n)
    for name in ("npu_gemm_bf16x16", "npu_gemm_bf16x16_buffer", "npu_gemm_bf16x16_ifm_reuse"):
        assert_same(call_n(name, a, b, bi, OUT, I64(m), I64(k), I64(n), act, n=m * n, dt=np.uint16), want)
    got = call_n("npu_gemm_bf16x16_transposed", a, b, bi, OUT, I64(m), I64(k), I64(n), act, n=m * n, dt=np.uint16)
    assert_same(got, want.T)


INT_MIN, INT_MAX = -(1 << 31), (1 << 31) - 1
F32_MAX = float(np.float32(3.4028235e38))
NANV = [float(v) for v in np.array([0x7FC00000, 0xFFC00000, 0xFFC12345, 0x7F800001], np.uint32).view(np.float32)]
# (inv_scale, scale, zero point): NaN / inf / zero / subnormal coefficients and extreme zero points
QDQ_SPECIAL = [(NANV[0], 1.0, 0), (NANV[2], NANV[1], 0), (NANV[3], 2.0, 1), (np.inf, 1.0, 3), (-np.inf, 2.0, -3), (0.0, np.inf, 0), (1.0, -np.inf, 5),
               (1.0, NANV[2], 7), (-0.0, 0.5, 0), (1e-40, 1e-40, 1), (F32_MAX, F32_MAX, 100),
               (1.0, 1.0, INT_MIN), (1.0, 1.0, INT_MAX), (0.5, 3.0, -32768), (0.5, 3.0, 65535)]


@pytest.mark.parametrize("signed", [0, 1])
@pytest.mark.parametrize("coef", QDQ_SPECIAL, ids=lambda c: "-".join(f"{v:g}" for v in c))
def test_special_gemm_qdq(signed, coef):
    inv, scale, zp = coef
    m, k, n = 6, 40, 11
    a, b = sp_bf16((m, k)), sp_bf16((k, n))
    bi = sp_f32(n)
    want = gb.gemm_qdq(a, b, bi, 1, signed, inv, scale, zp)
    for name in ("npu_gemm_quant_dequant_bf16x16", "npu_gemm_quant_dequant_bf16x16_ifm_reuse"):
        got = call_n(name, a, b, bi, OUT, I64(m), I64(k), I64(n), 1, signed, float(np.float32(inv)),
                     float(np.float32(scale)), zp, n=m * n, dt=np.uint16)
        assert_same(got, want)


ALPHA_SPECIAL = [*NANV, np.inf, -np.inf, 0.0, -0.0, 1e-40, -3e-39, F32_MAX, 0.1015625]


@pytest.mark.parametrize("geom", CONV[:6])
@pytest.mark.parametrize("act,alpha", [(0, 0.0), (1, 0.0), (2, 0.0)] + [(3, al) for al in ALPHA_SPECIAL],
                         ids=lambda v: f"{v:g}")
def test_special_conv(geom, act, alpha):
    kh, kw, ci, co, s, kh_sv = geom
    h, w = (3 - 1) * s + kh, (4 - 1) * s + kw
    ifm, wts = sp_bf16((h, w, ci), 0.15), sp_bf16((kh, kw, ci, co), 0.15)
    bi = sp_f32(co)
    want = gb.conv(ifm, wts, bi, s, s, kh_sv, act, alpha)
    args = (ifm, wts, bi, OUT, I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s, I64(kh_sv), act)
    assert_same(call_n("npu_conv_bf16x16_buffer", *args, alpha, n=want.size, dt=np.uint16), want)
    if act != 3:
        assert_same(call_n("npu_conv_bf16x16", *args, n=want.size, dt=np.uint16), want)


@pytest.mark.parametrize("geom", CONV[:5])
@pytest.mark.parametrize("typ,act", [(0, 0), (1, 0), (1, 1)])
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("bshape", ["full", "chan", "pixel"])
def test_special_conv_add(geom, typ, act, bias, bshape):
    kh, kw, ci, co, s, kh_sv = geom
    ho, wo = 3, 4
    h, w = (ho - 1) * s + kh, (wo - 1) * s + kw
    ifm, wts = sp_bf16((h, w, ci), 0.15), sp_bf16((kh, kw, ci, co), 0.15)
    shp = {"full": (ho, wo, co), "chan": (1, 1, co), "pixel": (ho, wo, 1)}[bshape]
    ifm2 = sp_bf16(shp)
    bi = sp_f32(co) if bias else None
    want = gb.conv_add(ifm, wts, bi, ifm2, s, s, kh_sv, typ, act)
    got = call_n("npu_conv_add_bf16x16", ifm, wts, bi, ifm2, OUT, I64(h), I64(w), I64(ci), I64(kh),
                 I64(kw), I64(co), s, s, I64(kh_sv), I64(shp[0]), I64(shp[1]), I64(shp[2]), typ, act,
                 n=want.size, dt=np.uint16)
    assert_same(got, want)


@pytest.mark.parametrize("kk,s", [((3, 3), 1), ((5, 5), 1), ((1, 1), 1), ((3, 3), 2)])
@pytest.mark.parametrize("act,alpha", [(0, 0.0), (1, 0.0), (2, 0.0)] + [(3, al) for al in ALPHA_SPECIAL],
                         ids=lambda v: f"{v:g}")
@pytest.mark.parametrize("bias", [0, 1])
def test_special_depthwise(kk, s, act, alpha, bias):
    kh, kw = kk
    c = 64
    h, w = (4 - 1) * s + kh, (5 - 1) * s + kw
    ifm, wts = sp_bf16((h, w, c), 0.15), sp_bf16((kh, kw, c), 0.15)
    bi = sp_f32(c) if bias else None
    want = gb.depthwise(ifm, wts, bi, s, s, act, alpha)
    for name in ("npu_depthwise_conv_bf16x16", "npu_depthwise_conv_bf16x16_c32"):
        got = call_n(name, ifm, wts, bi, OUT, I64(h), I64(w), I64(c), I64(kh), I64(kw), s, s, act, alpha,
                     n=want.size, dt=np.uint16)
        assert_same(got, want)
