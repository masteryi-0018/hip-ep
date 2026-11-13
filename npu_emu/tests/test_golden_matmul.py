# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The matmul-group entry points (include/npu_emu_matmul.h) against the numpy
reference in golden_matmul.py, bit for bit, on random tensors: every operand
type, the activations, odd and even kernel widths on both window-sum orders,
16-bit inputs whose window sums pass 2^24, and coefficients with ties."""

import numpy as np
import pytest

import golden_matmul as gm
from npu import I64, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT, call

RNG = np.random.default_rng(2028)
T = {"i8": (NPU_I8, np.int8), "u8": (NPU_U8, np.uint8), "i16": (NPU_I16, np.int16),
     "u16": (NPU_U16, np.uint16)}
f32 = np.float32


def rnd(t, shape, lo=None, hi=None):
    info = np.iinfo(T[t][1])
    return RNG.integers(info.min if lo is None else lo, info.max if hi is None else hi, shape,
                        endpoint=True).astype(T[t][1])


def run(name, *args, n, t):
    dt = T[t][1]
    return call(name, *args, out_nbytes=n * np.dtype(dt).itemsize, out_dtype=dt)


def assert_same(got, want):
    got = np.asarray(got).reshape(-1)
    want = np.asarray(want).astype(got.dtype).reshape(-1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{got.size} differ, first at {bad[:6].tolist()}: "
                           f"got {got[bad[:6]].tolist()} want {want[bad[:6]].tolist()}")


ACTS = [(gm.ACT_NONE, 0.0, 0.0), (gm.ACT_CLIP, -3.5, 1000.5), (gm.ACT_LRELU, 0.1015625, -3.59375),
        (gm.ACT_LRELU, 1.5, 2.0)]


@pytest.mark.parametrize("at,bt,ot", [("i16", "i16", "i16"), ("u16", "i16", "u16"),
                                      ("i16", "u16", "i16"), ("u16", "u16", "u16")])
@pytest.mark.parametrize("act", range(len(ACTS)))
def test_gemm(at, bt, ot, act):
    m, k, n = 32, 192, 64
    a, b = rnd(at, (m, k)), rnd(bt, (k, n))
    # c2 = 1/6-ish scale gives many ties; c3 / c1 / c0 as from a zero point
    c2 = f32(1.0 / 6 / 2**12)
    c0, c1, c3 = f32(-12.5), f32(-c2 * 3), f32(-c2 * 7)
    kind, a0, a1 = ACTS[act]
    got = run("npu_gemm_qdq_int16x16", a, T[at][0], b, T[bt][0], OUT, T[ot][0], I64(m), I64(k),
              I64(n), float(c0), float(c1), float(c2), float(c3), kind, a0, a1, n=m * n, t=ot)
    assert_same(got, gm.gemm_qdq_int16x16(a, b, ot, c0, c1, c2, c3, kind, a0, a1))


CONV = [  # (kh, kw, stride, k_opt)
    (3, 3, 1, 1), (3, 3, 2, 1), (3, 3, 1, 0), (7, 7, 2, 1), (1, 1, 1, 1), (2, 2, 1, 1),
    (4, 4, 2, 1), (5, 5, 1, 1), (3, 1, 1, 0)]


@pytest.mark.parametrize("split16", [False, True], ids=["int8x8", "int16x8"])
@pytest.mark.parametrize("geom", range(len(CONV)))
@pytest.mark.parametrize("act", range(len(ACTS) + 1))
def test_conv(split16, geom, act):
    kh, kw, s, k_opt = CONV[geom]
    ho, wo, ci, co = 3, 4, 64 if split16 else 32, 48
    h, w = (ho - 1) * s + kh, (wo - 1) * s + kw
    it, ot = ("u16", "u16") if split16 else ("i8", "u8")
    # 16-bit inputs near the top of the range, so the window sums pass 2^24
    ifm = rnd(it, (h, w, ci), 40000, 65535) if split16 else rnd(it, (h, w, ci))
    wts = rnd("u8" if split16 else "i8", (kh, kw, ci, co))
    c2 = (RNG.uniform(1e-5, 4e-5, co) if split16 else RNG.uniform(0.002, 0.02, co)).astype(f32)
    c1 = (-c2 * RNG.integers(100, 200, co)).astype(f32)
    c0 = (RNG.uniform(-1e5, 1e5, co) if split16 else RNG.uniform(-200, 200, co)).astype(f32)
    slope = None
    if act == len(ACTS):  # PRELU, pivot on the ofm zero point
        kind, a0, a1 = gm.ACT_PRELU, 3000.0 if split16 else 100.0, 0.0
        slope = RNG.uniform(-0.5, 1.5, co).astype(f32)
    else:
        kind, a0, a1 = ACTS[act]
    name = "npu_conv_qdq_int16x8" if split16 else "npu_conv_qdq_int8x8"
    want = gm.conv_qdq(ifm, wts, ot, s, s, c0, c1, c2, split16, kind, a0, a1, slope,
                       k_opt=bool(k_opt))
    got = run(name, ifm, T[it][0], wts, T["u8" if split16 else "i8"][0], OUT, T[ot][0], I64(h),
              I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s, k_opt, c0, c1, c2, kind, a0, a1,
              slope, n=want.size, t=ot)
    assert_same(got, want)


def test_conv_no_actv_sum():
    ifm, wts = rnd("i8", (5, 6, 32)), rnd("i8", (3, 3, 32, 16))
    c0, c2 = RNG.uniform(-50, 50, 16).astype(f32), RNG.uniform(0.01, 0.02, 16).astype(f32)
    want = gm.conv_qdq(ifm, wts, "i8", 1, 1, c0, None, c2, False)
    got = run("npu_conv_qdq_int8x8", ifm, NPU_I8, wts, NPU_I8, OUT, NPU_I8, I64(5), I64(6),
              I64(32), I64(3), I64(3), I64(16), 1, 1, 1, c0, None, c2, gm.ACT_NONE, 0.0, 0.0, None,
              n=want.size, t="i8")
    assert_same(got, want)


def test_k_opt_order_matters():
    """The reference's two window-sum orders differ on these inputs, so the
    k_opt tests above do discriminate."""
    ifm = rnd("u16", (5, 5, 64), 60000, 65535).astype(np.int64)
    a = gm.window_sum(ifm, 3, 3, 1, 1, False)
    b = gm.window_sum(ifm, 3, 3, 1, 1, True)
    assert np.any(a != b)


# ---- QDQ integer GEMMs --------------------------------------------------------------

GEMM_ENTRIES = [  # (entry, kind, a types, b types, out types, has c3)
    ("npu_gemm_a8w8", gm.I8, ("i8", "u8"), ("i8",), ("i8", "u8"), False),
    ("npu_gemm_a8w8_stream", gm.I8, ("u8",), ("i8",), ("u8",), False),
    ("npu_gemm_a8a8", gm.I8, ("i8", "u8"), ("i8", "u8"), ("i8",), True),
    ("npu_gemm_qdq_a8a8_transpose", gm.I8, ("u8",), ("u8",), ("u8",), True),
    ("npu_gemm_a8w4", gm.I8, ("i8",), ("i4",), ("i8",), False),
    ("npu_gemm_a16w8", gm.I16_JOIN, ("i16", "u16"), ("i8", "u8"), ("i16", "u16"), False),
    ("npu_gemm_a16a8", gm.I16_JOIN, ("i16",), ("i8",), ("i16",), True),
    ("npu_gemm_a16w8_stream", gm.I16_EXACT, ("i16", "u16"), ("i8",), ("u16",), False),
    ("npu_gemm_a16w4", gm.I16_PRODUCT, ("u16",), ("i4",), ("i16",), False),
]


def gemm_operands(at, bt, m, k, n, small=False):
    """Random operands; small: values in [-4, 4] (non-negative for unsigned types) so that
    with the ties coefficients the outputs stay in range and often land on .5."""
    if small:
        lo = 0 if at.startswith("u") else -4
        a = RNG.integers(lo, 5, (m, k)).astype(T[at][1])
        b = RNG.integers(0 if bt.startswith("u") else -4, 5, (k, n)).astype(np.int8 if bt == "i4" else T[bt][1])
        return a, b
    a = rnd(at, (m, k))
    b = RNG.integers(-8, 8, (k, n)).astype(np.int8) if bt == "i4" else rnd(bt, (k, n))
    return a, b


def gemm_coeffs(kind, n, ties):
    """Per-column c0..c3. ties: power-of-two scales, so many outputs land on .5."""
    if ties:
        # with small operands every term is a multiple of 1/4: many exact .5 outputs
        c2 = np.full(n, 0.25, f32)
        c0 = RNG.integers(-20, 20, n).astype(f32)
        return c0, (-c2 * RNG.integers(-3, 4, n)).astype(f32), c2, (-c2 * RNG.integers(-3, 4, n)).astype(f32)
    else:
        c2 = (RNG.uniform(1e-4, 4e-3, n) if kind == gm.I8 else RNG.uniform(1e-7, 2e-5, n)).astype(f32)
        c0 = RNG.uniform(-300, 300, n).astype(f32)
    c1 = (-c2 * RNG.integers(-20, 20, n)).astype(f32)
    c3 = (-c2 * RNG.integers(-20, 20, n)).astype(f32)
    return c0, c1, c2, c3


@pytest.mark.parametrize("entry,kind,ats,bts,ots,has_c3", GEMM_ENTRIES, ids=[e[0] for e in GEMM_ENTRIES])
@pytest.mark.parametrize("ties", [False, True], ids=["rand", "ties"])
@pytest.mark.parametrize("act", range(len(ACTS)))
def test_gemm_qdq(entry, kind, ats, bts, ots, has_c3, ties, act):
    m, k, n = 16, 192, 48
    for at in ats:
        for bt in bts:
            ot = ots[(ats.index(at) + bts.index(bt)) % len(ots)]
            a, b = gemm_operands(at, bt, m, k, n, small=ties)
            c0, c1, c2, c3 = gemm_coeffs(kind, n, ties)
            c3 = c3 if has_c3 else None
            kind_, a0, a1 = ACTS[act]
            if ot in ("i8", "u8") and kind_ == gm.ACT_CLIP:
                a1 = 100.5
            want = gm.gemm_qdq(a, b, ot, kind, c0, c1, c2, c3, kind_, a0, a1)
            btype = T["i8"][0] if bt == "i4" else T[bt][0]
            got = run(entry, a, T[at][0], b, btype, OUT, T[ot][0], I64(m), I64(k), I64(n), c0, c1,
                      c2, c3, kind_, a0, a1, n=m * n, t=ot)
            assert_same(got, want)


@pytest.mark.parametrize("entry", ["npu_gemm_a16w8_hybrid", "npu_gemm_a16w8_hybrid_channelwise"])
@pytest.mark.parametrize("at", ["i16", "u16"])
@pytest.mark.parametrize("act", range(len(ACTS)))
def test_gemm_hybrid(entry, at, act):
    m, k, n, sv_n = 32, 256, 128, 64
    ties = act % 2 == 1
    a, b = gemm_operands(at, "i8", m, k, n, small=ties)
    c0_a, c1, c2, _ = gemm_coeffs(gm.I16_JOIN, n, ties)
    c0_b = (c0_a + RNG.uniform(-5, 5, n)).astype(f32)
    if entry.endswith("_hybrid"):
        c2 = np.full(n, c2[0], f32)
    kind_, a0, a1 = ACTS[act]
    for cols in (sv_n, n):
        want = gm.gemm_hybrid(a, b, "i16", c0_a, c0_b, c1, c2, cols, kind_, a0, a1, signed=at == "i16")
        got = run(entry, a, T[at][0], b, T["i8"][0], OUT, T["i16"][0], I64(m), I64(k), I64(n), c0_a,
                  c0_b, c1, c2, I64(cols), kind_, a0, a1, n=m * n, t="i16")
        assert_same(got, want)


@pytest.mark.parametrize("kpass", [0, 1])
@pytest.mark.parametrize("ties", [False, True], ids=["rand", "ties"])
def test_gemm_add_a8w8(kpass, ties):
    m, k, n = 16, 256, 64
    a, b = gemm_operands("u8", "i8", m, k, n, small=ties)
    ifm3 = rnd("u8", (m, n))
    c0, c1, c2, _ = gemm_coeffs(gm.I8, n, ties)
    add = (f32(1.0), f32(0.75 if ties else 0.329383 / 0.439177), f32(-96.0 if ties else -128 * 0.329383 / 0.439177))
    want = gm.gemm_add_a8w8(a, b, ifm3, "u8", c0, c1, c2, *add, kpass)
    got = run("npu_qdq_gemm_add_a8w8", a, T["u8"][0], b, T["i8"][0], ifm3, T["u8"][0], OUT, T["u8"][0],
              I64(m), I64(k), I64(n), c0, c1, c2, *(float(x) for x in add), kpass, n=m * n, t="u8")
    assert_same(got, want)


@pytest.mark.parametrize("ties", [False, True], ids=["rand", "ties"])
def test_gemm_add_a16w8(ties):
    m, k, n = 16, 128, 64
    a, b = gemm_operands("u16", "i8", m, k, n, small=ties)
    res = rnd("u16", (m, n))
    c0, c1, c2, _ = gemm_coeffs(gm.I16_EXACT, n, ties)
    c0 = (c0 + 30000).astype(f32)
    add = (f32(1.0), f32(0.5 if ties else 0.731), f32(-512.5 if ties else -1234.567))
    want = gm.gemm_add_a16w8(a, b, res, c0, c1, c2, *add)
    got = run("npu_gemm_add_a16w8", a, T["u16"][0], b, T["i8"][0], res, OUT, I64(m), I64(k), I64(n),
              c0, c1, c2, *(float(x) for x in add), n=m * n, t="u16")
    assert_same(got, want)


def test_gemm_ties_are_hit():
    """The ties coefficients really put outputs on .5, so half-even is exercised."""
    a, b = gemm_operands("i8", "i8", 16, 192, 48, small=True)
    c0, c1, c2, _ = gemm_coeffs(gm.I8, 48, True)
    v = gm.gemm_qdq_acc(a, b, gm.I8, c0, c1, c2).astype(np.float64)
    assert np.sum((v - np.floor(v) == 0.5) & (np.abs(v) < 127)) > 50
