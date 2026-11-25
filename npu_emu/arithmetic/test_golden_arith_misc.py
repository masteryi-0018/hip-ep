# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Quantized div / leaky relu / prelu / square / batchnorm against the numpy reference."""

import numpy as np
import pytest

import golden_arith_misc as m
from golden_float_cvt import round_float
from npu import I64, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT, raw_rc
from npu import f32 as f
from testing import INT as T
from testing import assert_same, rand_ints, run, shp

RNG = np.random.default_rng(3031)


def types(bits):
    return ["i8", "u8"] if bits == 8 else ["i16", "u16"]


def pairs(bits):
    return [(a, b) for a in types(bits) for b in types(bits)]


def values(t, n=1 << 14):
    """Every value of an 8-bit type; the ends plus a random sample of a 16-bit one."""
    info = np.iinfo(T[t][1])
    if info.bits == 8:
        return np.arange(info.min, info.max + 1).astype(T[t][1])
    edge = np.array([info.min, info.min + 1, 0, 1, info.max - 1, info.max], np.int64)
    edge = edge[(edge >= info.min) & (edge <= info.max)]
    return np.concatenate([edge, RNG.integers(info.min, info.max, n, endpoint=True)]).astype(
        T[t][1])


def zps(t):
    info = np.iinfo(T[t][1])
    mid = (int(info.min) + int(info.max) + 1) // 2
    return [0, info.min, info.max, mid + 3, mid - 7]


# (s_in, s_out) pairs: non powers of two, ratios with many ties, one large ratio.
SCALES = [(0.1, 0.6), (0.6, 0.1), (0.02, 0.05), (0.003, 0.0071), (0.25, 0.75), (1.7, 0.01)]


def div_coeffs():
    """(c0, c1): c0 = s_a / (s_b * s_out) in double, c1 = zp_out; then ties."""
    out = [(f(sa / (sb * so)), f(zo)) for sa, sb, so, zo in
           [(0.02, 0.01, 0.05, 20), (0.1, 0.6, 0.3, -3), (0.003, 0.7, 0.0071, 100),
            (0.6, 0.1, 0.1, 0)]]
    return out + [(0.5, 0.0), (0.5, 0.5), (1.5, -0.5), (f(1 / 3), 0.5)]


@pytest.mark.parametrize("ci", range(len(div_coeffs())))
def test_div_a8(ci):
    c0, c1 = div_coeffs()[ci]
    for it, ot in pairs(8):
        v = values(it)
        a, b = (x.reshape(-1).astype(T[it][1]) for x in np.meshgrid(v, v))
        for z1, z2 in [(0, 0), (10, -5), (zps(it)[3], zps(it)[1])]:
            w1, w2 = f(12582912.0 + z1), f(12582912.0 + z2)
            got = run("npu_elementwise_div_a8", a, T[it][0], b, OUT, T[ot][0], I64(a.size),
                      c0, c1, w1, w2, n=a.size, t=ot)
            assert_same(got, m.elementwise_div_a8(a, it, b, ot, a.size, c0, c1, w1, w2))


@pytest.mark.parametrize("ci", range(len(div_coeffs())))
def test_div_a16(ci):
    c0, c1 = div_coeffs()[ci]
    for it, ot in pairs(16):
        info = np.iinfo(T[it][1])
        b = np.arange(info.min, info.max + 1).astype(T[it][1])   # every divisor
        a = rand_ints(RNG, T[it][1], b.size)
        a[:6] = [info.min, info.max, 0, 1, info.max, info.min]
        for z1, z2 in [(0, 0), (100, -50), (zps(it)[1], zps(it)[2]), (zps(it)[2], zps(it)[1])]:
            w1, w2 = f(12582912.0 + z1), f(12582912.0 + z2)
            got = run("npu_elementwise_div_a16", a, T[it][0], b, OUT, T[ot][0], I64(a.size),
                      c0, c1, w1, w2, n=a.size, t=ot)
            assert_same(got, m.elementwise_div_a16(a, it, b, ot, a.size, c0, c1, w1, w2))


BSHAPES = [(3, 5, 64), (1, 1, 64), (3, 5, 1), (1, 5, 64), (3, 1, 64), (3, 1, 1), (1, 1, 1)]


def bdiv_coeffs():
    """(c0, c1, shift): floor(C * 2^15) words with shift 15, then plain coefficients with
    shift 0 (ties)."""
    out = []
    for sa, sb, so, zo in [(0.001, 0.001, 0.001, 8), (0.0079, 0.0023, 5.35e-05, 40487),
                           (0.1, 0.6, 0.3, -3), (0.02, 0.01, 0.05, 20)]:
        out.append((f(np.floor(sa / (sb * so) * 2 ** 15)), f(np.floor(zo * 2 ** 15)), 15))
    return out + [(0.5, 0.5, 0), (f(1 / 3), -0.5, 0), (3.0, 0.0, 1)]


def bdiv_case(bits, t, bshape, H=3, W=5, C=64):
    a = rand_ints(RNG, T[t][1], (H, W, C))
    b = rand_ints(RNG, T[t][1], bshape)
    return a, b


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("ci", range(len(bdiv_coeffs())))
def test_bdiv_shapes(bits, ci):
    c0, c1, sh = bdiv_coeffs()[ci]
    for t in types(bits):
        for bs in BSHAPES:
            a, b = bdiv_case(bits, t, bs)
            b.reshape(-1)[:3] = [0, 0, 1][:b.size]     # zero divisors with zp2 = 0
            # a16 zp2 must lie in the dtype's range
            for z1, z2 in [(0, 0), (10, 5 if t.startswith("u") else -5), (zps(t)[3], zps(t)[4])]:
                name = f"qdq_elementwise_broadcasting3d_div_a{bits}"
                got = run(f"npu_{name}", a, T[t][0], b, OUT, shp(a.shape), shp(bs), c0, c1, z1,
                          z2, sh, n=a.size, t=t)
                want = getattr(m, name)(a, t, b, a.shape, bs, c0, c1, z1, z2, sh)
                assert_same(got, want)


@pytest.mark.parametrize("bits", [8, 16])
def test_bdiv_all_divisors(bits):
    """Every (a8: every pair of) value, so every reciprocal the kernel can use."""
    for t in types(bits):
        info = np.iinfo(T[t][1])
        if bits == 8:
            v = np.arange(info.min, info.max + 1)
            a, b = (x.astype(T[t][1]).reshape(256, 256, 1) for x in np.meshgrid(v, v))
        else:
            b = np.arange(info.min, info.max + 1).astype(T[t][1]).reshape(256, 256, 1)
            a = rand_ints(RNG, T[t][1], b.shape)
        for c0, c1, sh in bdiv_coeffs()[:3]:
            for z1, z2 in [(0, 0), (zps(t)[1], zps(t)[2]), (zps(t)[2], zps(t)[1])]:
                name = f"qdq_elementwise_broadcasting3d_div_a{bits}"
                got = run(f"npu_{name}", a, T[t][0], b, OUT, shp(a.shape), shp(b.shape), c0, c1,
                          z1, z2, sh, n=a.size, t=t)
                want = getattr(m, name)(a, t, b, a.shape, b.shape, c0, c1, z1, z2, sh)
                assert_same(got, want)


def leaky_coeffs(si, so, zi, zo, alpha):
    """c1, c0, c1_alpha, c0_alpha (double, then fp32)."""
    c1 = si / so
    c0 = zo - c1 * zi
    return f(c1), f(c0), f(alpha * c1), f(alpha * (c0 - zo) + zo)


LEAKY = [leaky_coeffs(si, so, zi, zo, al) for (si, so), (zi, zo), al in
         [(SCALES[0], (3, -2), 0.1), (SCALES[1], (0, 0), 0.2), (SCALES[2], (-10, 7), 0.01),
          (SCALES[3], (143, 3), 0.3), (SCALES[4], (5, 5), 0.5), (SCALES[5], (0, 100), 1.5)]] + \
        [(0.5, 0.5, 0.25, 0.5), (1.0, 0.0, -0.5, 0.0)]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("ci", range(len(LEAKY)))
def test_leakyrelu(bits, ci):
    c = LEAKY[ci]
    for it, ot in pairs(bits):
        x = values(it)
        got = run(f"npu_elementwise_leakyrelu_a{bits}", x, T[it][0], OUT, T[ot][0], I64(x.size),
                  *c, n=x.size, t=ot)
        assert_same(got, getattr(m, f"elementwise_leakyrelu_a{bits}")(x, it, ot, x.size, *c))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("si,so", SCALES)
def test_prelu(bits, si, so):
    pos = f(si / so)
    C = 64
    slopes = np.concatenate([[-1.0, -0.25, 0.0, 0.5, 1.5, 2.0, 0.1, 0.3],
                             RNG.uniform(-2, 2, C - 8)])
    neg = (slopes * np.float64(pos)).astype(np.float32)
    for it, ot in pairs(bits):
        x = values(it, 1 << 13)
        rows = x.size // C
        x = np.ascontiguousarray(x[:rows * C].reshape(rows, C))
        for zi, zo in [(0, 0.0), (5, -3.0), (zps(it)[3], f(zps(ot)[4])), (-7, 0.5)]:
            got = run(f"npu_elementwise_prelu_a{bits}", x, T[it][0], OUT, T[ot][0], I64(rows),
                      I64(C), pos, zi, f(zo), neg, n=x.size, t=ot)
            assert_same(got, getattr(m, f"elementwise_prelu_a{bits}")(x, it, ot, rows, C, pos, zi,
                                                                      zo, neg))


def square_coeffs(si, so, zi, zo):
    k = si * si / so
    return f(1.0 * k), f(-2.0 * k * zi), f(k * zi * zi + zo)


SQUARE = [square_coeffs(si, so, zi, zo) for (si, so), (zi, zo) in
          zip(SCALES, [(0, 0), (-20, -5), (3, 7), (100, 0), (-1, 2), (0, -100)])] + \
         [(0.5, 0.5, 0.5), (f(1 / 3), -1.5, 0.25), (0.0078125, 0.0, 0.0)]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("ci", range(len(SQUARE)))
def test_square(bits, ci):
    c = SQUARE[ci]
    for it, ot in pairs(bits):
        x = values(it)
        got = run(f"npu_elementwise_square_a{bits}", x, T[it][0], OUT, T[ot][0], I64(x.size),
                  *c, n=x.size, t=ot)
        assert_same(got, getattr(m, f"elementwise_square_a{bits}")(x, it, ot, x.size, *c))


def bn_coeffs(C, si, so, zi, zo):
    """bf16 gamma / beta folded with fp32 scale words, each step rounded to fp32."""
    gamma = round_float(RNG.uniform(-2, 2, C).astype(np.float32), "bf16").astype(np.float64)
    beta = round_float(RNG.uniform(-5, 5, C).astype(np.float32), "bf16").astype(np.float64)
    gamma[:4] = [0.5, -0.5, 1.0, 0.0]
    w_scale, b_scale, cross_scale = f(si / so), f(1.0 / so), f(-zi * si / so)
    cw = (gamma * w_scale).astype(np.float32)
    cross = (gamma * cross_scale + f(zo)).astype(np.float32)
    cb = (beta * b_scale + cross.astype(np.float64)).astype(np.float32)
    return cw, cb


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("act,alpha", [(0, 1.0), (1, 1.0), (3, 0.1), (3, 0.2), (3, 0.01),
                                       (3, 1.5)])
def test_batchnorm(bits, act, alpha):
    C = 96
    for t in types(bits):
        for (si, so), (zi, zo) in zip(SCALES, [(10, 10), (0, 0), (-5, 3), (128, 100),
                                                (3, -2), (0, 0)]):
            if t.startswith("u"):
                zi, zo = abs(zi), abs(zo)
            cw, cb = bn_coeffs(C, si, so, zi, zo)
            x = values(t, 1 << 13)
            rows = x.size // C
            x = np.ascontiguousarray(x[:rows * C].reshape(rows, C))
            got = run(f"npu_qdq_batchnorm2d_stream_a{bits}", x, T[t][0], OUT, I64(rows), I64(C),
                      cw, cb, f(zo), act, f(alpha), n=x.size, t=t)
            assert_same(got, getattr(m, f"qdq_batchnorm2d_stream_a{bits}")(
                x, t, rows, C, cw, cb, zo, act, alpha))


# One device buffer for every pointer of the invalid calls: they must fail before any launch.
P = np.zeros(4096, np.uint8)
S1 = shp((1, 1, 1))


def _bad_args():
    """Each entry of arith_misc.h with argument sets it must reject (-1)."""
    out = []
    for b, t, u, w in ((8, NPU_I8, NPU_U8, NPU_I16), (16, NPU_I16, NPU_U16, NPU_I8)):
        div = f"npu_elementwise_div_a{b}"
        bdiv = f"npu_qdq_elementwise_broadcasting3d_div_a{b}"
        leaky = f"npu_elementwise_leakyrelu_a{b}"
        prelu = f"npu_elementwise_prelu_a{b}"
        square = f"npu_elementwise_square_a{b}"
        bn = f"npu_qdq_batchnorm2d_stream_a{b}"
        out += [(div, P, w, P, P, t, I64(1), 1.0, 0.0, 0.0, 0.0),
                (div, P, t, P, P, t, I64(-1), 1.0, 0.0, 0.0, 0.0),
                (div, P, t, P, None, t, I64(0), 1.0, 0.0, 0.0, 0.0),
                (div, P, t, P, P, w, I64(1), 1.0, 0.0, 0.0, 0.0),
                (div, P, t, None, P, u, I64(1), 1.0, 0.0, 0.0, 0.0),
                (bdiv, P, t, P, P, shp((2, 2, 4)), shp((1, 2, 3)), 1.0, 0.0, 0, 0, 0),
                (bdiv, P, t, P, P, S1, S1, 1.0, 0.0, 0, 0, 65),
                (bdiv, P, w, P, P, S1, S1, 1.0, 0.0, 0, 0, 0),
                (bdiv, P, t, P, None, S1, S1, 1.0, 0.0, 0, 0, 0),
                (leaky, P, w, P, t, I64(1), 1.0, 0.0, 1.0, 0.0),
                (leaky, P, t, None, t, I64(1), 1.0, 0.0, 1.0, 0.0),
                (prelu, P, t, P, t, I64(1), I64(0), 1.0, 0, 0.0, P),
                (prelu, P, t, P, t, I64(0), I64(0), 1.0, 0, 0.0, P),   # C >= 1 even if empty
                (prelu, P, t, P, t, I64(-1), I64(1), 1.0, 0, 0.0, P),
                (prelu, P, t, P, t, I64(0), I64(1), 1.0, 0, 0.0, None),
                (bn, P, t, P, I64(-1), I64(1), P, P, 0.0, 0, 0.0),
                (prelu, P, t, P, t, I64(1), I64(1), 1.0, 0, 0.0, None),
                (square, P, w, P, t, I64(1), 1.0, 0.0, 0.0),
                (square, None, t, P, t, I64(1), 1.0, 0.0, 0.0),
                (bn, P, t, P, I64(1), I64(1), P, P, 0.0, 2, 0.0),
                (bn, P, t, P, I64(1), I64(1), None, P, 0.0, 0, 0.0),
                (bn, P, t, P, I64(1), I64(1), P, None, 0.0, 0, 0.0),
                (bn, P, w, P, I64(1), I64(1), P, P, 0.0, 0, 0.0)]
    # the fp32 reciprocal table covers |b - zp2| <= 65535 only
    out.append(("npu_qdq_elementwise_broadcasting3d_div_a16", P, NPU_U16, P, P, S1, S1, 1.0, 0.0,
                0, -1, 0))
    return out


BAD_ARGS = _bad_args()


def test_bad_args():
    for args in BAD_ARGS:
        assert raw_rc(*args) == -1, args
    # empty tensors are a no-op
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_div_a8", P, NPU_I8, P, P, shp((2, 3, 0)),
                  shp((1, 1, 1)), 1.0, 0.0, 0, 0, 0) == 0
    assert raw_rc("npu_elementwise_prelu_a8", P, NPU_I8, P, NPU_I8, I64(0), I64(3), 1.0, 0, 0.0,
                  P) == 0
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_div_a16", P, NPU_I16, P, P, shp((0, 3, 4)),
                  shp((1, 1, 1)), 1.0, 0.0, 0, 0, 0) == 0
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_div_a16", P, NPU_I16, P, None,
                  shp((0, 3, 4)), shp((1, 1, 1)), 1.0, 0.0, 0, 0, 0) == -1
