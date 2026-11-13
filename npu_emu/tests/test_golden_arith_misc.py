# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The arith_misc entry points (include/npu_emu_arith_misc.h) against the
numpy reference in golden_arith_misc.py, bit for bit. 8-bit inputs are
exhaustive (all value pairs for the divisions), the 16-bit divisions see
every divisor; coefficients are computed the way the mllib YAML does (double,
stored as fp32) from scales that are not powers of two (0.1 / 0.6 style ratios
with many .5 ties), plus hand-made ties, saturation and zero divisors."""

import os
import sys

import numpy as np
import pytest

import golden_arith_misc as m
from npu import I64, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT, raw_rc
from npu import call as callx

RNG = np.random.default_rng(3031)
T = {"i8": (NPU_I8, np.int8), "u8": (NPU_U8, np.uint8), "i16": (NPU_I16, np.int16),
     "u16": (NPU_U16, np.uint16)}
DEV = m.inv_f32_dev()


def f(x):
    return float(np.float32(x))


def run(name, *args, n, t):
    dt = T[t][1]
    return callx(name, *args, out_nbytes=n * np.dtype(dt).itemsize, out_dtype=dt)


def assert_same(got, want):
    got = np.asarray(got).reshape(-1)
    want = np.asarray(want).astype(got.dtype).reshape(-1)
    assert got.size == want.size
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{got.size} differ, first at {bad[:6].tolist()}: "
                           f"got {got[bad[:6]].tolist()} want {want[bad[:6]].tolist()}")


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


def rnd(t, shape):
    info = np.iinfo(T[t][1])
    return RNG.integers(info.min, info.max, shape, endpoint=True).astype(T[t][1])


def zps(t):
    info = np.iinfo(T[t][1])
    mid = (int(info.min) + int(info.max) + 1) // 2
    return [0, info.min, info.max, mid + 3, mid - 7]


# (s_in, s_out) pairs: non powers of two, ratios with many ties, one large ratio.
SCALES = [(0.1, 0.6), (0.6, 0.1), (0.02, 0.05), (0.003, 0.0071), (0.25, 0.75), (1.7, 0.01)]


# ---- A. division ------------------------------------------------------------------

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
            got = run("npu_elementwise_div_a8", a, b, T[it][0], OUT, T[ot][0], I64(a.size),
                      c0, c1, w1, w2, n=a.size, t=ot)
            assert_same(got, m.div(a, b, c0, c1, w1, w2, ot))


@pytest.mark.parametrize("ci", range(len(div_coeffs())))
def test_div_a16(ci):
    c0, c1 = div_coeffs()[ci]
    for it, ot in pairs(16):
        info = np.iinfo(T[it][1])
        b = np.arange(info.min, info.max + 1).astype(T[it][1])   # every divisor
        a = rnd(it, b.size)
        a[:6] = [info.min, info.max, 0, 1, info.max, info.min]
        for z1, z2 in [(0, 0), (100, -50), (zps(it)[1], zps(it)[2]), (zps(it)[2], zps(it)[1])]:
            w1, w2 = f(12582912.0 + z1), f(12582912.0 + z2)
            got = run("npu_elementwise_div_a16", a, b, T[it][0], OUT, T[ot][0], I64(a.size),
                      c0, c1, w1, w2, n=a.size, t=ot)
            assert_same(got, m.div(a, b, c0, c1, w1, w2, ot))


BSHAPES = [(3, 5, 64), (1, 1, 64), (3, 5, 1), (1, 5, 64), (3, 1, 64), (3, 1, 1), (1, 1, 1)]


def bdiv_coeffs():
    """(c0, c1, shift): the YAML's floor(C * 2^15) words with shift 15, then
    plain coefficients with shift 0 (ties)."""
    out = []
    for sa, sb, so, zo in [(0.001, 0.001, 0.001, 8), (0.0079, 0.0023, 5.35e-05, 40487),
                           (0.1, 0.6, 0.3, -3), (0.02, 0.01, 0.05, 20)]:
        out.append((f(np.floor(sa / (sb * so) * 2 ** 15)), f(np.floor(zo * 2 ** 15)), 15))
    return out + [(0.5, 0.5, 0), (f(1 / 3), -0.5, 0), (3.0, 0.0, 1)]


def bdiv_case(bits, t, bshape, H=3, W=5, C=64):
    a = rnd(t, (H, W, C))
    b = rnd(t, bshape)
    return a, b


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("ci", range(len(bdiv_coeffs())))
def test_bdiv_shapes(bits, ci):
    if bits == 16 and DEV is None:
        pytest.skip("src/npu_emu_inv_f32_table.h removed")
    c0, c1, sh = bdiv_coeffs()[ci]
    for t in types(bits):
        for bs in BSHAPES:
            a, b = bdiv_case(bits, t, bs)
            b.reshape(-1)[:3] = [0, 0, 1][:b.size]     # zero divisors with zp2 = 0
            # a16 zp2 must lie in the dtype's range (the fp32 reciprocal table)
            for z1, z2 in [(0, 0), (10, 5 if t.startswith("u") else -5), (zps(t)[3], zps(t)[4])]:
                got = run(f"npu_qdq_elementwise_broadcasting3d_div_a{bits}", a, b, T[t][0], OUT,
                          *[I64(v) for v in a.shape + bs], c0, c1, z1, z2, sh,
                          n=a.size, t=t)
                want = (m.bdiv_a8(a, b, c0, c1, z1, z2, sh, t) if bits == 8 else
                        m.bdiv_a16(a, b, c0, c1, z1, z2, sh, t, DEV))
                assert_same(got, want)


@pytest.mark.parametrize("bits", [8, 16])
def test_bdiv_all_divisors(bits):
    """Every (a8: every pair of) value, so every reciprocal the kernel can use."""
    if bits == 16 and DEV is None:
        pytest.skip("src/npu_emu_inv_f32_table.h removed")
    for t in types(bits):
        info = np.iinfo(T[t][1])
        if bits == 8:
            v = np.arange(info.min, info.max + 1)
            a, b = (x.astype(T[t][1]).reshape(256, 256, 1) for x in np.meshgrid(v, v))
        else:
            b = np.arange(info.min, info.max + 1).astype(T[t][1]).reshape(256, 256, 1)
            a = rnd(t, b.shape)
        for c0, c1, sh in bdiv_coeffs()[:3]:
            for z1, z2 in [(0, 0), (zps(t)[1], zps(t)[2]), (zps(t)[2], zps(t)[1])]:
                got = run(f"npu_qdq_elementwise_broadcasting3d_div_a{bits}", a, b, T[t][0], OUT,
                          *[I64(v) for v in a.shape + b.shape], c0, c1, z1, z2, sh,
                          n=a.size, t=t)
                want = (m.bdiv_a8(a, b, c0, c1, z1, z2, sh, t) if bits == 8 else
                        m.bdiv_a16(a, b, c0, c1, z1, z2, sh, t, DEV))
                assert_same(got, want)


def test_inv_bf16_matches_mllib():
    """The bf16 / fp16 reciprocal models equal mllib's inv_lut.py tables on
    every code (only where MLLIB_OPERATORS_COMMON points at mllib's
    internal/models/python/restructured/operators/common)."""
    p = os.environ.get("MLLIB_OPERATORS_COMMON", "")
    if not p or not os.path.exists(os.path.join(p, "inv_lut.py")):
        pytest.skip("mllib inv_lut.py not reachable")
    sys.path.insert(0, p)
    try:
        import inv_lut
    finally:
        sys.path.remove(p)
    y = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    assert np.array_equal(m.inv_bf16(y).view(np.uint32),
                          inv_lut.bf16_inv_lut().view(np.uint32))
    h = np.arange(65536, dtype=np.uint16).view(np.float16)
    want = inv_lut.inv_fp16_aie4(h).astype(np.float16)
    ok = ~np.isnan(h)
    assert np.array_equal(m.inv_f16(h)[ok].view(np.uint16), want[ok].view(np.uint16))


# ---- B. activations ---------------------------------------------------------------

def leaky_coeffs(si, so, zi, zo, alpha):
    """The YAML's c1, c0, c1_alpha, c0_alpha (double, then fp32)."""
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
        assert_same(got, m.leakyrelu(x, *c, ot))


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
            assert_same(got, m.prelu(x, pos, zi, zo, neg, ot))


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
        assert_same(got, m.square(x, *c, ot))


# ---- C. batch normalization ---------------------------------------------------------

def bn_coeffs(C, si, so, zi, zo):
    """effective_coefficients() of the mllib model: bf16 gamma / beta folded
    with fp32 scale words, each step rounded to fp32."""
    gamma = m.bf16(RNG.uniform(-2, 2, C).astype(np.float32)).astype(np.float64)
    beta = m.bf16(RNG.uniform(-5, 5, C).astype(np.float32)).astype(np.float64)
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
                      cw, cb, act, f(zo), f(alpha), n=x.size, t=t)
            assert_same(got, m.batchnorm(x, cw, cb, act, zo, alpha, t))


# ---- argument checks ------------------------------------------------------------------

def test_bad_arguments():
    z = None
    assert raw_rc("npu_elementwise_div_a8", z, z, NPU_I16, z, NPU_I8, I64(1), 1.0, 0.0, 0.0,
                  0.0) == -1
    assert raw_rc("npu_elementwise_div_a16", z, z, NPU_I16, z, NPU_U8, I64(1), 1.0, 0.0, 0.0,
                  0.0) == -1
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_div_a8", z, z, NPU_I8, z, I64(2), I64(2),
                  I64(4), I64(1), I64(2), I64(3), 1.0, 0.0, 0, 0, 0) == -1
    # the fp32 reciprocal table covers |b - zp2| <= 65535 only
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_div_a16", z, z, NPU_U16, z, I64(1),
                  I64(1), I64(1), I64(1), I64(1), I64(1), 1.0, 0.0, 0, -1, 0) == -1
    assert raw_rc("npu_elementwise_leakyrelu_a8", z, NPU_U16, z, NPU_U8, I64(1), 1.0, 0.0, 1.0,
                  0.0) == -1
    assert raw_rc("npu_elementwise_prelu_a16", z, NPU_I16, z, NPU_I16, I64(1), I64(0), 1.0, 0,
                  0.0, z) == -1
    assert raw_rc("npu_elementwise_square_a16", z, NPU_I8, z, NPU_I16, I64(1), 1.0, 0.0,
                  0.0) == -1
    assert raw_rc("npu_qdq_batchnorm2d_stream_a8", z, NPU_I8, z, I64(1), I64(1), z, z, 2, 0.0,
                  0.0) == -1
