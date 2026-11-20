# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The QDQ integer GEMM entry points against golden_qdq_gemm.py, bit for bit."""

import numpy as np
import pytest

import golden_act as ga
import golden_qdq_gemm as gm
from npu import I64, OUT
from qdq_testing import ACTS, RNG, T, assert_same, f32, rnd, run

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
            if ot in ("i8", "u8") and kind_ == ga.ACT_CLIP:
                a1 = 100.5
            want = gm.gemm_qdq(a, b, ot, kind, c0, c1, c2, c3, kind_, a0, a1)
            btype = T["i8"][0] if bt == "i4" else T[bt][0]
            got = run(entry, a, T[at][0], b, btype, OUT, T[ot][0], I64(m), I64(k), I64(n), c0, c1,
                      c2, c3, kind_, a0, a1, n=m * n, t=ot)
            assert_same(got, want)


GELU_ENTRIES = [e for e in GEMM_ENTRIES if e[0] in ("npu_gemm_a8w8", "npu_gemm_a8w8_stream",
                                                       "npu_gemm_a16w8", "npu_gemm_a16w8_stream")]


@pytest.mark.parametrize("entry,kind,ats,bts,ots,has_c3", GELU_ENTRIES, ids=[e[0] for e in GELU_ENTRIES])
@pytest.mark.parametrize("scale", [0.0123, 0.5, 1 / 3])
def test_gemm_qdq_gelu(entry, kind, ats, bts, ots, has_c3, scale):
    """GELU on the dequantized accumulator (real values in [-6, 6], column 0 = +-0),
    then one quantizing mac."""
    m, k, n = 16, 192, 48
    for at in ats:
        for bt in bts:
            ot = ots[(ats.index(at) + bts.index(bt)) % len(ots)]
            a, b = gemm_operands(at, bt, m, k, n)
            c0, c1, c2, _ = gemm_coeffs(kind, n, False)
            acc = gm.gemm_qdq_acc(a, b, kind, np.zeros(n, f32), c1, c2, None)
            span = np.abs(acc).max() / 6 + 1e-30
            c1, c2 = (c1 / span).astype(f32), (c2 / span).astype(f32)
            c0 = RNG.uniform(-1, 1, n).astype(f32)
            c0[0], c1[0], c2[0] = -0.0, 0.0, 0.0             # column 0: v = +-0
            a0 = float(f32(1.0 / scale))
            a1 = {"i8": 3.0, "u8": 128.0, "i16": -5.0, "u16": 32768.0}[ot]
            want = gm.gemm_qdq(a, b, ot, kind, c0, c1, c2, None, ga.ACT_GELU, a0, a1)
            btype = T["i8"][0] if bt == "i4" else T[bt][0]
            got = run(entry, a, T[at][0], b, btype, OUT, T[ot][0], I64(m), I64(k), I64(n), c0, c1,
                      c2, None, ga.ACT_GELU, a0, a1, n=m * n, t=ot)
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
    """The ties coefficients put outputs on .5."""
    a, b = gemm_operands("i8", "i8", 16, 192, 48, small=True)
    c0, c1, c2, _ = gemm_coeffs(gm.I8, 48, True)
    v = gm.gemm_qdq_acc(a, b, gm.I8, c0, c1, c2).astype(np.float64)
    assert np.sum((v - np.floor(v) == 0.5) & (np.abs(v) < 127)) > 50


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


def test_gelu_rejected_outside_qdq_gemm():
    """Only the QDQ integer GEMMs take GELU."""
    a = rnd("i16", (4, 8)); b = rnd("i16", (8, 4))
    with pytest.raises(RuntimeError):
        run("npu_gemm_qdq_int16x16", a, T["i16"][0], b, T["i16"][0], OUT, T["i16"][0], I64(4), I64(8),
            I64(4), 0.0, 0.0, 1.0, 0.0, ga.ACT_GELU, 1.0, 0.0, n=16, t="i16")


# ---- special values: the float words (c0..c3, act0, act1, add_c*) take NaN / inf / zeros /
# subnormals; operands sit at the integer type limits; the output is integer, so NaN -> 0 and
# +-inf saturates ([default] float -> int conversion).
SPECIAL = np.array([0x7FC00000, 0xFFC12345, 0x7F800001, 0x7F800000, 0xFF800000, 0, 0x80000000,
                    1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000],
                   np.uint32).view(f32)
NS = len(SPECIAL)


def special_cols(n, which, kind, rng=None):
    """Per-column c0..c3: normal base words, with `which` replaced by the special table (a
    cross product of the two words over the columns), or (mixed) every word drawn at random
    from the table and the base."""
    c2 = np.full(n, 0.01 if kind == gm.I8 else 1e-5, f32)
    c0, c1, c3 = np.full(n, 0.5, f32), (-3 * c2).astype(f32), (-5 * c2).astype(f32)
    j = np.arange(n)
    words = {"c0": c0, "c1": c1, "c2": c2, "c3": c3}
    if which == "mixed":
        pool = np.concatenate([SPECIAL, np.array([0.5, -0.01, 0.01, 3.0], f32)])
        for w in words.values():
            w[:] = pool[rng.integers(0, len(pool), n)]
    else:
        for r, name in enumerate(which.split("+")):
            words[name][:] = SPECIAL[(j // NS**r) % NS]
    return c0, c1, c2, c3


def special_operands(at, bt, m, k, n):
    """Row 0 / 1 at the lowest / highest value of the A type, row 2 zero, the rest random."""
    a, b = gemm_operands(at, bt, m, k, n)
    info = np.iinfo(a.dtype)
    a[0], a[1], a[2] = info.min, info.max, 0
    return a, b


COEF_SETS = ["c0+c2", "c1+c3", "mixed"]


@pytest.mark.parametrize("entry,kind,ats,bts,ots,has_c3", GEMM_ENTRIES, ids=[e[0] for e in GEMM_ENTRIES])
def test_gemm_qdq_special(entry, kind, ats, bts, ots, has_c3):
    m, k, n = 4, 16, NS * NS
    rng = np.random.default_rng(2040)
    gelu = entry in [e[0] for e in GELU_ENTRIES]
    seen = set()
    for at in ats:
        for bt in bts:
            ot = ots[(ats.index(at) + bts.index(bt)) % len(ots)]
            a, b = special_operands(at, bt, m, k, n)
            btype = T["i8"][0] if bt == "i4" else T[bt][0]

            def check(cols, act, a0, a1):
                c0, c1, c2, c3 = cols
                c3 = c3 if has_c3 else None
                want = gm.gemm_qdq(a, b, ot, kind, c0, c1, c2, c3, act, a0, a1)
                got = run(entry, a, T[at][0], b, btype, OUT, T[ot][0], I64(m), I64(k), I64(n), c0,
                          c1, c2, c3, act, a0, a1, n=m * n, t=ot)
                assert_same(got, want)
                seen.update(np.unique(want).tolist())

            for which in COEF_SETS:
                cols = special_cols(n, which, kind, rng)
                check(cols, ga.ACT_NONE, 0.0, 0.0)
                check(cols, ga.ACT_LRELU, 0.1015625, -3.59375)
            base = special_cols(n, "c0", kind)
            acts = [ga.ACT_CLIP, ga.ACT_LRELU] + ([ga.ACT_GELU] if gelu else [])
            for act in acts:
                for i in range(NS):
                    a0, a1 = float(SPECIAL[i]), float(SPECIAL[(5 * i + 2) % NS])
                    if act == ga.ACT_CLIP and ot in ("i8", "u8"):
                        a1 = a1 if np.isnan(a1) or abs(a1) > 1e3 else 100.5
                    check(base, act, a0, a1)
    assert len(seen) > 2        # the outputs do not collapse to 0 and the two limits


@pytest.mark.parametrize("entry", ["npu_gemm_a16w8_hybrid", "npu_gemm_a16w8_hybrid_channelwise"])
@pytest.mark.parametrize("at", ["i16", "u16"])
def test_gemm_hybrid_special(entry, at):
    m, k, n = 4, 16, NS * NS
    rng = np.random.default_rng(2041)
    a, b = special_operands(at, "i8", m, k, n)
    for which in COEF_SETS:
        c0_a, c1, c2, c3 = special_cols(n, which, gm.I16_JOIN, rng)
        c0_b = np.roll(c0_a, 7) if which != "c1+c3" else c3
        for kind_, a0, a1 in [(ga.ACT_NONE, 0.0, 0.0), (ga.ACT_CLIP, float(SPECIAL[0]), 1000.5),
                              (ga.ACT_CLIP, -3.5, float(SPECIAL[3])),
                              (ga.ACT_LRELU, float(SPECIAL[3]), float(SPECIAL[4])),
                              (ga.ACT_LRELU, float(SPECIAL[6]), float(SPECIAL[0]))]:
            for cols in (n // 3, n):
                want = gm.gemm_hybrid(a, b, "i16", c0_a, c0_b, c1, c2, cols, kind_, a0, a1,
                                      signed=at == "i16")
                got = run(entry, a, T[at][0], b, T["i8"][0], OUT, T["i16"][0], I64(m), I64(k),
                          I64(n), c0_a, c0_b, c1, c2, I64(cols), kind_, a0, a1, n=m * n, t="i16")
                assert_same(got, want)


@pytest.mark.parametrize("entry", ["npu_gemm_qdq_int16x16", "npu_gemm_qdq_a16a16_transpose"])
@pytest.mark.parametrize("at,bt,ot", [("i16", "i16", "i16"), ("u16", "u16", "u16")])
def test_gemm_int16x16_special(entry, at, bt, ot):
    """One special word at a time (the others normal), then all four from the table."""
    m, k, n = 4, 16, 8
    a, b = rnd(at, (m, k)), rnd(bt, (k, n))
    a[0], a[1], a[2] = np.iinfo(a.dtype).min, np.iinfo(a.dtype).max, 0
    b[:, 0], b[:, 1] = np.iinfo(b.dtype).max, np.iinfo(b.dtype).min
    base = [-12.5, -2.5e-5, 1.5e-4, -3.5e-4]
    sets = []
    for w in range(4):
        for i in range(NS):
            c = list(base)
            c[w] = float(SPECIAL[i])
            sets.append(c)
    sets += [[float(SPECIAL[i]), float(SPECIAL[(3 * i + 1) % NS]), float(SPECIAL[(5 * i + 2) % NS]),
              float(SPECIAL[(7 * i + 3) % NS])] for i in range(NS)]
    for c0, c1, c2, c3 in sets:
        for kind_, a0, a1 in [(ga.ACT_NONE, 0.0, 0.0), (ga.ACT_CLIP, -3.5, 1000.5),
                              (ga.ACT_LRELU, float(SPECIAL[0]), -3.5),
                              (ga.ACT_LRELU, 0.1015625, float(SPECIAL[4]))]:
            want = gm.gemm_qdq_int16x16(a, b, ot, f32(c0), f32(c1), f32(c2), f32(c3), kind_, a0, a1)
            got = run(entry, a, T[at][0], b, T[bt][0], OUT, T[ot][0], I64(m), I64(k), I64(n), c0, c1,
                      c2, c3, kind_, a0, a1, n=m * n, t=ot)
            assert_same(got, want)


@pytest.mark.parametrize("kpass", [0, 1])
def test_gemm_add_a8w8_special(kpass):
    m, k, n = 4, 16, NS * NS
    rng = np.random.default_rng(2042)
    a, b = special_operands("u8", "i8", m, k, n)
    ifm3 = rnd("u8", (m, n))
    ifm3[0], ifm3[1] = 0, 255
    base = (1.0, 0.75, -96.0)
    sets = [[float(SPECIAL[i]) if w == j else base[j] for j in range(3)]
            for w in range(3) for i in range(NS)]
    sets += [[float(SPECIAL[i]), float(SPECIAL[(3 * i + 1) % NS]), float(SPECIAL[(5 * i + 2) % NS])]
             for i in range(NS)]
    for which in ["normal", "c0+c2", "c1+c3", "mixed"]:
        c0, c1, c2, _ = special_cols(n, "c0" if which == "normal" else which, gm.I8, rng)
        for add in (sets if which != "mixed" else sets[::5]):
            want = gm.gemm_add_a8w8(a, b, ifm3, "u8", c0, c1, c2, *(f32(x) for x in add), kpass)
            got = run("npu_qdq_gemm_add_a8w8", a, T["u8"][0], b, T["i8"][0], ifm3, T["u8"][0], OUT,
                      T["u8"][0], I64(m), I64(k), I64(n), c0, c1, c2, *add, kpass, n=m * n, t="u8")
            assert_same(got, want)


def test_gemm_add_a16w8_special():
    m, k, n = 4, 16, NS * NS
    rng = np.random.default_rng(2043)
    a, b = special_operands("u16", "i8", m, k, n)
    res = rnd("u16", (m, n))
    res[0], res[1] = 0, 65535
    base = (1.0, 0.5, -512.5)
    sets = [[float(SPECIAL[i]) if w == j else base[j] for j in range(3)]
            for w in range(3) for i in range(NS)]
    sets += [[float(SPECIAL[i]), float(SPECIAL[(3 * i + 1) % NS]), float(SPECIAL[(5 * i + 2) % NS])]
             for i in range(NS)]
    for which in ["c0", "c0+c2", "c1+c3", "mixed"]:
        c0, c1, c2, _ = special_cols(n, which, gm.I16_EXACT, rng)
        for add in (sets if which != "mixed" else sets[::5]):
            want = gm.gemm_add_a16w8(a, b, res, c0, c1, c2, *(f32(x) for x in add))
            got = run("npu_gemm_add_a16w8", a, T["u16"][0], b, T["i8"][0], res, OUT, I64(m), I64(k),
                      I64(n), c0, c1, c2, *add, n=m * n, t="u16")
            assert_same(got, want)
