# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""GeLU / Erf entry points against golden_gelu.py, bit for bit."""

import numpy as np
import pytest

import golden_gelu as gg
from golden_int_round import INT_RANGE
from npu import I64, NPU_BF16, NPU_I8, NPU_I16, NPU_U8, NPU_FP8, OUT, call, raw_rc
from testing import ALL8, ALL16, F16T, INT
from testing import assert_same as check

f32 = np.float32
RNG = np.random.default_rng(20261115)


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_gelu_bf16_all(t):
    got = call("npu_gelu_bf16", ALL16, F16T[t], OUT, I64(65536), out_nbytes=2 * 65536,
               out_dtype=np.uint16)
    check(got, gg.gelu_bf16(ALL16, t))


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_erf_bf16_all(t):
    got = call("npu_elementwise_erf_bf16", ALL16, F16T[t], OUT, I64(65536),
               out_nbytes=2 * 65536, out_dtype=np.uint16)
    check(got, gg.elementwise_erf_bf16(ALL16, t))


def test_gelu_erf_spot():
    """NaN codes and the sign of zero."""
    x = np.array([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x0001, 0x8001], np.uint16)
    got = call("npu_gelu_bf16", x, NPU_BF16, OUT, I64(8), out_nbytes=16, out_dtype=np.uint16)
    check(got, [0x0000, 0x8000, 0xFFC0, 0xFFC0, 0xFFC0, 0xFFC1, 0x0000, 0x8000])
    got = call("npu_elementwise_erf_bf16", x, NPU_BF16, OUT, I64(8), out_nbytes=16,
               out_dtype=np.uint16)
    # -0 and subnormals give +0; +-inf and NaN give the canonical NaN of the clamp
    check(got, [0x0000, 0x0000, 0x7FC0, 0x7FC0, 0x7FC0, 0x7FC0, 0x0000, 0x0000])


def bfp16_blocks(nblk, emin, emax):
    e = RNG.integers(emin, emax + 1, nblk)
    m = RNG.integers(-128, 128, (nblk, 8))
    m[RNG.random(nblk) < 0.05] = 0
    m[RNG.random((nblk, 8)) < 0.03] = -128
    return np.concatenate([e[:, None], m & 0xFF], axis=1).astype(np.uint8).reshape(-1)


@pytest.mark.parametrize("emin,emax", [(120, 135), (100, 156), (0, 30), (1, 254), (126, 128)])
def test_gelu_bfp16(emin, emax):
    nblk = 8 * 512
    blk = bfp16_blocks(nblk, emin, emax)
    # groups with one dominant exponent: the others lose bits in the window
    blk.reshape(-1, 9)[::8, 0] = np.minimum(blk.reshape(-1, 9)[::8, 0].astype(int) + 20, 254)
    n = nblk * 8
    got = call("npu_gelu_bfp16", blk, OUT, I64(n), out_nbytes=n * 9 // 8, out_dtype=np.uint8)
    check(got, gg.gelu_bfp16(blk))


def test_gelu_bfp16_window_ties():
    """E - e = 17 + s with mantissas on the rounding ties of the window."""
    rows = []
    for s in range(0, 10):
        for m in (-128, -96, -65, -64, -3, -1, 1, 3, 64, 65, 96, 127):
            e = 110
            grp = np.zeros((8, 9), np.int64)
            grp[:, 0] = e
            grp[0, 0] = e + 17 + s
            grp[0, 1:] = 64
            grp[1:, 1:] = m
            rows.append(grp)
    blk = (np.concatenate(rows) & 0xFF).astype(np.uint8).reshape(-1)
    n = blk.size // 9 * 8
    got = call("npu_gelu_bfp16", blk, OUT, I64(n), out_nbytes=blk.size, out_dtype=np.uint8)
    check(got, gg.gelu_bfp16(blk))


FP8_SCALES = [(1.0, 1.0), (0.37, 0.013), (3.7, 0.0017), (0.011, 3.3), (1.1, 0.9),
              (1.00390625, 1 / 1.00390625), (1.01171875, 1 / 1.01171875), (2.0, 1 / 300.0),
              (200.0, 0.37), (0.013, 0.0071), (1 + 2 ** -11, 1 / (1 + 2 ** -11)),
              (1 + 3 * 2 ** -11, 1 / (1 + 3 * 2 ** -11)), (2.0 ** -20, 2.0 ** -10),
              (1e4, 1.0), (0.5, 4.0)]


def fp8_input(with_nan):
    """Every code 16 times, shuffled; with_nan keeps the NaN codes."""
    c = np.tile(ALL8, 16)
    if not with_nan:
        c = c[(c & 0x7F) != 0x7F]
    RNG.shuffle(c)
    return np.ascontiguousarray(c)


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("sc", FP8_SCALES)
@pytest.mark.parametrize("with_nan", [False, True])
def test_gelu_fp8(t, sc, with_nan):
    x = fp8_input(with_nan)
    i, recip = f32(sc[0]), f32(1.0 / sc[1])
    got = call("npu_elementwise_gelu_fp8", x, OUT, I64(x.size), F16T[t], float(i), float(recip),
               out_nbytes=x.size, out_dtype=np.uint8)
    check(got, gg.elementwise_gelu_fp8(x, t, i, recip))


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_gelu_fp8_recip_ties(t):
    """1 / ofm_scale on a tie of the compute type."""
    x = fp8_input(False)
    ulp = 2.0 ** -7 if t == "bf16" else 2.0 ** -10
    for k in (1, 3, 5):
        recip = f32(1 + k * ulp / 2)
        got = call("npu_elementwise_gelu_fp8", x, OUT, I64(x.size), F16T[t], 1.0, float(recip),
                   out_nbytes=x.size, out_dtype=np.uint8)
        check(got, gg.elementwise_gelu_fp8(x, t, f32(1), recip))


@pytest.mark.parametrize("sc", FP8_SCALES)
@pytest.mark.parametrize("with_nan", [False, True])
def test_erf_fp8(sc, with_nan):
    x = fp8_input(with_nan)
    co = gg.erf_fp8_coeffs(*sc)
    got = call("npu_elementwise_erf_fp8", x, OUT, I64(x.size), *[float(v) for v in co],
               out_nbytes=x.size, out_dtype=np.uint8)
    check(got, gg.elementwise_erf_fp8(x, *co))


def test_fp8_tail_group():
    """n not a multiple of 8: the last NaN group is cut at n."""
    x = np.array([0x38, 0x7F, 0x40, 0x10, 0x00, 0x80, 0x48, 0x30, 0x38, 0x7F, 0x20], np.uint8)
    got = call("npu_elementwise_gelu_fp8", x, OUT, I64(x.size), NPU_BF16, 1.0, 1.0,
               out_nbytes=x.size, out_dtype=np.uint8)
    check(got, gg.elementwise_gelu_fp8(x, "bf16", f32(1), f32(1)))
    co = gg.erf_fp8_coeffs(1.0, 0.5)
    got = call("npu_elementwise_erf_fp8", x, OUT, I64(x.size), *[float(v) for v in co],
               out_nbytes=x.size, out_dtype=np.uint8)
    check(got, gg.elementwise_erf_fp8(x, *co))


def all_ints(t):
    lo, hi = INT_RANGE[t]
    return np.arange(lo, hi + 1).astype(INT[t][1])


QGELU = [  # (ifm_scale, ifm_zp, ofm_scale, ofm_zp)
    (0.00608058413490653, -30, 0.008470021188259125, 30), (0.00037, 1234, 0.00091, -321),
    (0.047, 77, 0.019, 3), (0.1, 0, 0.1, 0), (0.031, -128, 1 / 257.0, -128),
    (0.0003, 100, 6e-5, 25024), (2.0 ** -7, 0, 2.0 ** -8, 0), (1.7, 5, 0.013, 0)]


@pytest.mark.parametrize("wid", ["a8", "a16"])
@pytest.mark.parametrize("sgn", ["ss", "su", "us", "uu"])
@pytest.mark.parametrize("p", QGELU)
def test_qdq_gelu(wid, sgn, p):
    base = "8" if wid == "a8" else "16"
    ti = ("i" if sgn[0] == "s" else "u") + base
    to = ("i" if sgn[1] == "s" else "u") + base
    x = all_ints(ti)
    zi, zo, ka, c, kdsp = gg.qdq_gelu_coeffs(*p)
    got = call(f"npu_qdq_gelu_exp2_{wid}", x, INT[ti][0], OUT, INT[to][0], I64(x.size), zi,
               float(zo), float(ka), float(c), float(kdsp),
               out_nbytes=x.size * np.dtype(INT[to][1]).itemsize, out_dtype=INT[to][1])
    check(got, gg.qdq_gelu_exp2_a8(x, zi, zo, ka, c, kdsp, to))


QERF = [  # (ifm_scale, ifm_zp, ofm_scale, ofm_zp); 257 / 259 / 16448 put c1_out on bf16 ties
    (0.024, 0, 0.008, 0), (0.031, 3, 1 / 257.0, -5), (0.027, 120, 1 / 259.0, 100),
    (0.0003, 100, 1 / 16448.0, 25024), (0.0002, 30000, 1 / 16576.0, 25152),
    (0.0001, 0, 4e-5, 0), (0.37, -7, 0.011, 1), (1e-3, 0, 2.0 ** -7, 0)]


@pytest.mark.parametrize("wid", ["a8", "a16"])
@pytest.mark.parametrize("sgn", ["ss", "su", "us", "uu"])
@pytest.mark.parametrize("p", QERF)
def test_qdq_erf(wid, sgn, p):
    base = "8" if wid == "a8" else "16"
    ti = ("i" if sgn[0] == "s" else "u") + base
    to = ("i" if sgn[1] == "s" else "u") + base
    x = all_ints(ti)
    co = gg.erf_qdq_coeffs(*p)
    got = call(f"npu_elementwise_erf_{wid}", x, INT[ti][0], OUT, INT[to][0], I64(x.size),
               *[float(v) for v in co],
               out_nbytes=x.size * np.dtype(INT[to][1]).itemsize, out_dtype=INT[to][1])
    check(got, gg.elementwise_erf_a8(x, *co, to))


def test_qdq_erf_large_x():
    """Huge dequantized values: x^5 / x^3 overflow (clamps), and inf - inf (NaN -> 0)."""
    x = all_ints("i16")
    for c1 in (1e4, 1e9):
        co = (f32(c1), f32(0), f32(100.0), f32(3.0))
        got = call("npu_elementwise_erf_a16", x, NPU_I16, OUT, NPU_I16, I64(x.size),
                   *[float(v) for v in co], out_nbytes=2 * x.size, out_dtype=np.int16)
        check(got, gg.elementwise_erf_a8(x, *co, "i16"))


# Parameter tables, indexed (ei - 1) * 5 + (eo - 1).
PW_RND = [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          2, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
PW_S = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 3, 2, 0, 0, 0, 3, 3, 0, 0, 0, 5,
        4, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
PW_K = [0, 3, 4, 5, 6, 4, 6, 8, 5, 9, 4, 11, 15, 5, 9, 7, 21, 29, 5, 9, 13, 41, 57, 8, 9, 25,
        81, 113, 5, 9, 50, 65, 65, 67, 69, 99, 65, 65, 66, 69]
PW_P = [0, 1, 1, 2, 5, 1, 1, 1, 3, 5, 1, 1, 1, 3, 5, 1, 1, 1, 3, 5, 1, 1, 1, 2, 5, 1, 1, 1, 3,
        4, 1, 1, 1, 3, 5, 1, 1, 1, 2, 5]


def xint8_params(ei, eo):
    cell = (ei - 1) * 5 + (eo - 1)
    k = eo - ei
    a = max(k, 0)
    return [int(k >= 1), PW_RND[cell], a, a + 1 - k, a + PW_S[cell], max(-k, 0), PW_K[cell],
            PW_P[cell]]


@pytest.mark.parametrize("ei", range(1, 9))
@pytest.mark.parametrize("eo", range(1, 6))
@pytest.mark.parametrize("pe", [0, 1])
def test_xint8_gelu_cells(ei, eo, pe):
    x = np.tile(ALL8.view(np.int8), 2)
    p = xint8_params(ei, eo) + [pe]
    got = call("npu_xint8_gelu", x, OUT, I64(x.size), *p, out_nbytes=x.size, out_dtype=np.int8)
    check(got, gg.xint8_gelu(x, *p))


def test_xint8_gelu_precision_enhanced_differs():
    """precision_enhanced changes the result (ifm 2^-4, ofm 2^-5)."""
    x = ALL8.view(np.int8)
    p = xint8_params(4, 5)
    assert np.any(gg.xint8_gelu(x, *p, 1) != gg.xint8_gelu(x, *p, 0))


def test_xint8_gelu_random_params():
    x = ALL8.view(np.int8).copy()
    for _ in range(300):
        p = [int(RNG.integers(0, 2)), int(RNG.integers(0, 3))] + \
            [int(v) for v in RNG.integers(0, 8, 4)] + [int(RNG.integers(0, 128)),
                                                       int(RNG.integers(0, 64)),
                                                       int(RNG.integers(0, 2))]
        got = call("npu_xint8_gelu", x, OUT, I64(x.size), *p, out_nbytes=x.size,
                   out_dtype=np.int8)
        check(got, gg.xint8_gelu(x, *p))


def test_bad_args():
    P = np.zeros(64, np.uint8)  # a valid device buffer
    n = I64(64)
    for name in ("npu_gelu_bf16", "npu_elementwise_erf_bf16"):
        assert raw_rc(name, None, NPU_I8, None, n) == -1
        assert raw_rc(name, None, NPU_BF16, None, I64(-1)) == -1
        assert raw_rc(name, None, NPU_BF16, None, n) == -1
        assert raw_rc(name, None, NPU_BF16, None, I64(0)) == -1
        assert raw_rc(name, P, NPU_BF16, P, I64(0)) == 0
    # whole sub-volumes of 512 elements only
    for bad in (65, 64, 448, -512):
        assert raw_rc("npu_gelu_bfp16", None, None, I64(bad)) == -1
    assert raw_rc("npu_gelu_bfp16", None, None, I64(512)) == -1
    assert raw_rc("npu_gelu_bfp16", None, None, I64(0)) == -1
    assert raw_rc("npu_gelu_bfp16", P, P, I64(0)) == 0
    assert raw_rc("npu_elementwise_gelu_fp8", None, None, n, NPU_FP8, 1.0, 1.0) == -1
    assert raw_rc("npu_elementwise_gelu_fp8", None, None, n, NPU_BF16, 1.0, 1.0) == -1
    assert raw_rc("npu_elementwise_gelu_fp8", None, None, I64(-1), NPU_BF16, 1.0, 1.0) == -1
    assert raw_rc("npu_elementwise_erf_fp8", None, None, I64(-1), 1.0, 0.0, 0.0, 1.0) == -1
    assert raw_rc("npu_elementwise_erf_fp8", None, None, n, 1.0, 0.0, 0.0, 1.0) == -1
    for bits, good, bad in ((8, NPU_I8, NPU_I16), (16, NPU_I16, NPU_U8)):
        for name, co in ((f"npu_qdq_gelu_exp2_a{bits}", (0, 0.0, 0.0, 0.0, 0.0)),
                         (f"npu_elementwise_erf_a{bits}", (0.0, 0.0, 0.0, 0.0))):
            assert raw_rc(name, None, bad, None, good, n, *co) == -1
            assert raw_rc(name, None, good, None, bad, n, *co) == -1
            assert raw_rc(name, None, good, None, good, I64(-1), *co) == -1
            assert raw_rc(name, None, good, None, good, n, *co) == -1
            assert raw_rc(name, None, good, None, good, I64(0), *co) == -1
            assert raw_rc(name, P, good, P, good, I64(0), *co) == 0
    ok = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # up_path, rnd_id, 4 shifts, K, P, precision_enhanced
    assert raw_rc("npu_xint8_gelu", None, None, I64(0), *ok) == -1
    assert raw_rc("npu_xint8_gelu", P, P, I64(0), *ok) == 0
    assert raw_rc("npu_xint8_gelu", None, None, n, *ok) == -1
    for k, v in ((1, 3), (2, 24), (3, 24), (4, 24), (5, 24), (8, 2), (8, -1)):
        p = list(ok)
        p[k] = v
        assert raw_rc("npu_xint8_gelu", P, P, I64(0), *p) == -1
