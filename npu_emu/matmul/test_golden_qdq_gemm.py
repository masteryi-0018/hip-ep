# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The QDQ integer GEMM entry points against golden_qdq_gemm.py, bit for bit."""

import numpy as np
import pytest

import golden_act as ga
import golden_quant as gq
import golden_qdq_gemm as gm
from npu import I64, NPU_I8, NPU_I16, NPU_I32, NPU_U8, NPU_U16, OUT, raw_rc
from qdq_testing import ACTS, NS, RNG, SPECIAL, T, assert_same, f32, rejected, rnd, run

VEC_ENTRIES = [  # (entry, kind, a types, b types, out types); per-column c0 / c1 / c2
    ("npu_gemm_a8w8", gm.I8, ("i8", "u8"), ("i8", "u8"), ("i8", "u8")),
    ("npu_gemm_a8w8_stream", gm.I8, ("u8",), ("i8",), ("u8",)),
    ("npu_gemm_a8w4", gm.I8, ("i8",), ("i4", "u4"), ("i8",)),
    ("npu_gemm_a16w8", gm.I16_JOIN, ("i16", "u16"), ("i8", "u8"), ("i16", "u16")),
    ("npu_gemm_a16w8_stream", gm.I16_EXACT, ("i16", "u16"), ("i8",), ("u16",)),
    ("npu_gemm_a16w4", gm.I16_PRODUCT, ("u16",), ("i4",), ("i16",)),
]
SCALAR_ENTRIES = [  # (entry, kind, a types, b types, out types); four scalar coefficients
    ("npu_gemm_a8a8", gm.I8, ("i8", "u8"), ("i8", "u8"), ("i8",)),
    ("npu_gemm_qdq_a8a8_transpose", gm.I8, ("u8",), ("u8",), ("u8",)),
    ("npu_gemm_a16a8", gm.I16_JOIN, ("i16",), ("i8",), ("i16",)),
]
GELU = ("npu_gemm_a8w8", "npu_gemm_a8w8_stream", "npu_gemm_a16w8", "npu_gemm_a16w8_stream")
STORE = {"i4": "i8", "u4": "u8"}  # int4 weights travel one per byte


def model(entry):
    return getattr(gm, entry[len("npu_"):])


def has_act(entry):
    return entry != "npu_gemm_qdq_a8a8_transpose"


def combos(ats, bts, ots):
    for at in ats:
        for bt in bts:
            yield at, bt, ots[(ats.index(at) + bts.index(bt)) % len(ots)]


def gemm_operands(at, bt, m, k, n, small=False):
    """Random operands; small: values in [-4, 4] (non-negative for unsigned types) so that
    with the ties coefficients the outputs stay in range and often land on .5."""
    if small:
        lo = 0 if at.startswith("u") else -4
        a = RNG.integers(lo, 5, (m, k)).astype(T[at][1])
        b = RNG.integers(0 if bt.startswith("u") else -4, 5, (k, n)).astype(T[STORE.get(bt, bt)][1])
        return a, b
    a = rnd(at, (m, k))
    if bt == "i4":
        b = RNG.integers(-8, 8, (k, n)).astype(np.int8)
    elif bt == "u4":
        b = RNG.integers(0, 16, (k, n)).astype(np.uint8)
    else:
        b = rnd(bt, (k, n))
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


def vec_args(entry, at, bt, ot, a, b, m, k, n, c0, c1, c2, act, a0, a1):
    return (entry, a, T[at][0], b, T[STORE.get(bt, bt)][0], OUT, T[ot][0], I64(m), I64(k), I64(n),
            c0, c1, c2, act, a0, a1)


def scalar_args(entry, at, bt, ot, a, b, m, k, n, c, act, a0, a1):
    tail = (act, a0, a1) if has_act(entry) else ()
    return (entry, a, T[at][0], b, T[bt][0], OUT, T[ot][0], I64(m), I64(k), I64(n),
            *(float(x) for x in c), *tail)


@pytest.mark.parametrize("entry,kind,ats,bts,ots", VEC_ENTRIES, ids=[e[0] for e in VEC_ENTRIES])
@pytest.mark.parametrize("ties", [False, True], ids=["rand", "ties"])
@pytest.mark.parametrize("act", range(len(ACTS)))
@pytest.mark.parametrize("asum", [True, False], ids=["c1", "no_c1"])
def test_gemm_vec(entry, kind, ats, bts, ots, ties, act, asum):
    m, k, n = 16, 192, 48
    for at, bt, ot in combos(ats, bts, ots):
        a, b = gemm_operands(at, bt, m, k, n, small=ties)
        c0, c1, c2, _ = gemm_coeffs(kind, n, ties)
        c1 = c1 if asum else None
        kind_, a0, a1 = ACTS[act]
        if ot in ("i8", "u8") and kind_ == gq.kQdqActClip:
            a1 = 100.5
        want = model(entry)(a, b, ot, c0, c1, c2, kind_, a0, a1)
        got = run(*vec_args(entry, at, bt, ot, a, b, m, k, n, c0, c1, c2, kind_, a0, a1),
                  n=m * n, t=ot)
        assert_same(got, want)


SCALAR_CASES = [(*e, act) for e in SCALAR_ENTRIES for act in range(len(ACTS) if has_act(e[0]) else 1)]


@pytest.mark.parametrize("entry,kind,ats,bts,ots,act", SCALAR_CASES,
                         ids=[f"{e[0]}-{e[-1]}" for e in SCALAR_CASES])
@pytest.mark.parametrize("ties", [False, True], ids=["rand", "ties"])
def test_gemm_scalar(entry, kind, ats, bts, ots, act, ties):
    m, k, n = 16, 192, 48
    for at, bt, ot in combos(ats, bts, ots):
        a, b = gemm_operands(at, bt, m, k, n, small=ties)
        c = [f32(x[0]) for x in gemm_coeffs(kind, 1, ties)]
        kind_, a0, a1 = ACTS[act]
        if ot in ("i8", "u8") and kind_ == gq.kQdqActClip:
            a1 = 100.5
        want = (model(entry)(a, b, ot, *c, kind_, a0, a1) if has_act(entry)
                else model(entry)(a, b, ot, *c))
        got = run(*scalar_args(entry, at, bt, ot, a, b, m, k, n, c, kind_, a0, a1), n=m * n, t=ot)
        assert_same(got, want)


def test_gemm_int4_bytes():
    """int4 weights: only the low nibble of each byte is read (signed for I8)."""
    m, k, n = 8, 64, 32
    for entry, at, ot in (("npu_gemm_a8w4", "i8", "i8"), ("npu_gemm_a16w4", "u16", "i16")):
        kind = gm.I8 if at == "i8" else gm.I16_PRODUCT
        for bt in ("i8", "u8"):
            a, b = gemm_operands(at, bt, m, k, n)
            c0, c1, c2, _ = gemm_coeffs(kind, n, False)
            want = model(entry)(a, b, ot, c0, c1, c2)
            got = run(*vec_args(entry, at, bt, ot, a, b, m, k, n, c0, c1, c2, gq.kQdqActNone, 0.0,
                                0.0), n=m * n, t=ot)
            assert_same(got, want)


GELU_ENTRIES = [e for e in VEC_ENTRIES if e[0] in GELU]


@pytest.mark.parametrize("entry,kind,ats,bts,ots", GELU_ENTRIES, ids=[e[0] for e in GELU_ENTRIES])
@pytest.mark.parametrize("scale", [0.0123, 0.5, 1 / 3])
def test_gemm_qdq_gelu(entry, kind, ats, bts, ots, scale):
    """GELU on the dequantized accumulator (real values in [-6, 6], column 0 = +-0),
    then one quantizing mac."""
    m, k, n = 16, 192, 48
    for at, bt, ot in combos(ats, bts, ots):
        a, b = gemm_operands(at, bt, m, k, n)
        c0, c1, c2, _ = gemm_coeffs(kind, n, False)
        acc = gm.gemm_qdq_acc(a, b, kind, np.zeros(n, f32), c1, c2, None)
        span = np.abs(acc).max() / 6 + 1e-30
        c1, c2 = (c1 / span).astype(f32), (c2 / span).astype(f32)
        c0 = RNG.uniform(-1, 1, n).astype(f32)
        c0[0], c1[0], c2[0] = -0.0, 0.0, 0.0             # column 0: v = +-0
        a0 = float(f32(1.0 / scale))
        a1 = {"i8": 3.0, "u8": 128.0, "i16": -5.0, "u16": 32768.0}[ot]
        want = model(entry)(a, b, ot, c0, c1, c2, gq.kQdqActGelu, a0, a1)
        got = run(*vec_args(entry, at, bt, ot, a, b, m, k, n, c0, c1, c2, gq.kQdqActGelu, a0, a1),
                  n=m * n, t=ot)
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
        want = model(entry)(a, b, "i16", cols, c0_a, c0_b, c1, c2, kind_, a0, a1)
        got = run(entry, a, T[at][0], b, T["i8"][0], OUT, T["i16"][0], I64(m), I64(k), I64(n),
                  I64(cols), c0_a, c0_b, c1, c2, kind_, a0, a1, n=m * n, t="i16")
        assert_same(got, want)


@pytest.mark.parametrize("kpass", [0, 1])
@pytest.mark.parametrize("ties", [False, True], ids=["rand", "ties"])
def test_gemm_add_a8w8(kpass, ties):
    m, k, n = 16, 256, 64
    a, b = gemm_operands("u8", "i8", m, k, n, small=ties)
    ifm3 = rnd("u8", (m, n))
    c0, c1, c2, _ = gemm_coeffs(gm.I8, n, ties)
    add = (f32(1.0), f32(0.75 if ties else 0.329383 / 0.439177), f32(-96.0 if ties else -128 * 0.329383 / 0.439177))
    want = gm.qdq_gemm_add_a8w8(a, b, ifm3, "u8", c0, c1, c2, *add, kpass)
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
    got = run("npu_gemm_add_a16w8", a, b, res, OUT, I64(m), I64(k), I64(n), c0, c1, c2,
              *(float(x) for x in add), n=m * n, t="u16")
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


# ---- special values: the float words (c0..c3, act0, act1, add_c*) take NaN / inf / zeros /
# subnormals (qdq_testing.SPECIAL); operands sit at the integer type limits; the output is
# integer, so NaN -> 0 and +-inf saturates ([default] float -> int conversion).


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


@pytest.mark.parametrize("entry,kind,ats,bts,ots", VEC_ENTRIES, ids=[e[0] for e in VEC_ENTRIES])
def test_gemm_vec_special(entry, kind, ats, bts, ots):
    m, k, n = 4, 16, NS * NS
    rng = np.random.default_rng(2040)
    seen = set()
    for at, bt, ot in combos(ats, bts, ots):
        a, b = special_operands(at, bt, m, k, n)

        def check(cols, act, a0, a1):
            c0, c1, c2, _ = cols
            want = model(entry)(a, b, ot, c0, c1, c2, act, a0, a1)
            got = run(*vec_args(entry, at, bt, ot, a, b, m, k, n, c0, c1, c2, act, a0, a1),
                      n=m * n, t=ot)
            assert_same(got, want)
            seen.update(np.unique(want).tolist())

        for which in COEF_SETS:
            cols = special_cols(n, which, kind, rng)
            check(cols, gq.kQdqActNone, 0.0, 0.0)
            check(cols, gq.kQdqActLrelu, 0.1015625, -3.59375)
        base = special_cols(n, "c0", kind)
        acts = [gq.kQdqActClip, gq.kQdqActLrelu] + ([gq.kQdqActGelu] if entry in GELU else [])
        for act in acts:
            for i in range(NS):
                a0, a1 = float(SPECIAL[i]), float(SPECIAL[(5 * i + 2) % NS])
                if act == gq.kQdqActClip and ot in ("i8", "u8"):
                    a1 = a1 if np.isnan(a1) or abs(a1) > 1e3 else 100.5
                check(base, act, a0, a1)
    assert len(seen) > 2        # the outputs do not collapse to 0 and the two limits


def scalar_sets():
    """One special word at a time (the others normal), then all four from the table."""
    base = [-12.5, -2.5e-5, 1.5e-4, -3.5e-4]
    sets = []
    for w in range(4):
        for i in range(NS):
            c = list(base)
            c[w] = float(SPECIAL[i])
            sets.append(c)
    return sets + [[float(SPECIAL[i]), float(SPECIAL[(3 * i + 1) % NS]),
                    float(SPECIAL[(5 * i + 2) % NS]), float(SPECIAL[(7 * i + 3) % NS])]
                   for i in range(NS)]


SCALAR_ACTS = [(gq.kQdqActNone, 0.0, 0.0), (gq.kQdqActClip, -3.5, 100.5),
               (gq.kQdqActLrelu, float(SPECIAL[0]), -3.5),
               (gq.kQdqActLrelu, 0.1015625, float(SPECIAL[4]))]


@pytest.mark.parametrize("entry,kind,ats,bts,ots", SCALAR_ENTRIES,
                         ids=[e[0] for e in SCALAR_ENTRIES])
def test_gemm_scalar_special(entry, kind, ats, bts, ots):
    m, k, n = 4, 16, 8
    for at, bt, ot in combos(ats, bts, ots):
        a, b = special_operands(at, bt, m, k, n)
        b[:, 0], b[:, 1] = np.iinfo(b.dtype).max, np.iinfo(b.dtype).min
        for c in scalar_sets():
            for act, a0, a1 in SCALAR_ACTS if has_act(entry) else SCALAR_ACTS[:1]:
                cf = [f32(x) for x in c]
                want = (model(entry)(a, b, ot, *cf, act, a0, a1) if has_act(entry)
                        else model(entry)(a, b, ot, *cf))
                got = run(*scalar_args(entry, at, bt, ot, a, b, m, k, n, c, act, a0, a1),
                          n=m * n, t=ot)
                assert_same(got, want)


@pytest.mark.parametrize("entry", ["npu_gemm_a16w8_hybrid", "npu_gemm_a16w8_hybrid_channelwise"])
@pytest.mark.parametrize("at", ["i16", "u16"])
def test_gemm_hybrid_special(entry, at):
    m, k, n = 4, 16, NS * NS
    rng = np.random.default_rng(2041)
    a, b = special_operands(at, "i8", m, k, n)
    for which in COEF_SETS:
        c0_a, c1, c2, c3 = special_cols(n, which, gm.I16_JOIN, rng)
        c0_b = np.roll(c0_a, 7) if which != "c1+c3" else c3
        for kind_, a0, a1 in [(gq.kQdqActNone, 0.0, 0.0),
                              (gq.kQdqActClip, float(SPECIAL[0]), 1000.5),
                              (gq.kQdqActClip, -3.5, float(SPECIAL[3])),
                              (gq.kQdqActLrelu, float(SPECIAL[3]), float(SPECIAL[4])),
                              (gq.kQdqActLrelu, float(SPECIAL[6]), float(SPECIAL[0]))]:
            for cols in (n // 3, n):
                want = model(entry)(a, b, "i16", cols, c0_a, c0_b, c1, c2, kind_, a0, a1)
                got = run(entry, a, T[at][0], b, T["i8"][0], OUT, T["i16"][0], I64(m), I64(k),
                          I64(n), I64(cols), c0_a, c0_b, c1, c2, kind_, a0, a1, n=m * n, t="i16")
                assert_same(got, want)


@pytest.mark.parametrize("entry", ["npu_gemm_qdq_int16x16", "npu_gemm_qdq_a16a16_transpose"])
@pytest.mark.parametrize("at,bt,ot", [("i16", "i16", "i16"), ("u16", "u16", "u16")])
def test_gemm_int16x16_special(entry, at, bt, ot):
    """One special word at a time (the others normal), then all four from the table."""
    m, k, n = 4, 16, 8
    a, b = rnd(at, (m, k)), rnd(bt, (k, n))
    a[0], a[1], a[2] = np.iinfo(a.dtype).min, np.iinfo(a.dtype).max, 0
    b[:, 0], b[:, 1] = np.iinfo(b.dtype).max, np.iinfo(b.dtype).min
    for c0, c1, c2, c3 in scalar_sets():
        for kind_, a0, a1 in [(gq.kQdqActNone, 0.0, 0.0), (gq.kQdqActClip, -3.5, 1000.5),
                              (gq.kQdqActLrelu, float(SPECIAL[0]), -3.5),
                              (gq.kQdqActLrelu, 0.1015625, float(SPECIAL[4]))]:
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
            want = gm.qdq_gemm_add_a8w8(a, b, ifm3, "u8", c0, c1, c2, *(f32(x) for x in add), kpass)
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
            got = run("npu_gemm_add_a16w8", a, b, res, OUT, I64(m), I64(k), I64(n), c0, c1, c2,
                      *add, n=m * n, t="u16")
            assert_same(got, want)


def _bad_vec(entry, at, bt, ot, **kw):
    """Argument sets of a per-column entry, each with one invalid argument."""
    m, k, n = 4, 8, 4
    a, b = gemm_operands(at, bt, m, k, n)
    c = np.ones(n, f32)
    ok = dict(at=at, bt=STORE.get(bt, bt), ot=ot, m=m, c0=c, c2=c, act=gq.kQdqActNone)
    bad = [dict(at="i32"), dict(bt="i16"), dict(ot="i16" if ot in ("i8", "u8") else "i8"),
           dict(m=-1), dict(c0=None), dict(c2=None), dict(act=gq.kQdqActPrelu), dict(act=1),
           dict(act=2), dict(act=8)]
    if entry not in GELU:
        bad.append(dict(act=gq.kQdqActGelu))
    for d in bad:
        p = {**ok, **d}
        at_ = NPU_I32 if p["at"] == "i32" else T[p["at"]][0]
        yield (entry, a, at_, b, T[p["bt"]][0], OUT, T[p["ot"]][0], I64(p["m"]), I64(k), I64(n),
               p["c0"], c, p["c2"], p["act"], 0.0, 0.0), m * n * 2


def test_bad_args():
    for entry, kind, ats, bts, ots in VEC_ENTRIES:
        for args, nb in _bad_vec(entry, ats[0], bts[0], ots[0]):
            assert rejected(*args, nbytes=nb), (entry, args[2:])
        assert raw_rc(entry, None, T[ats[0]][0], None, T[STORE.get(bts[0], bts[0])][0], None,
                      T[ots[0]][0], I64(4), I64(8), I64(4), None, None, None, 0, 0.0, 0.0) == -1
    for entry, kind, ats, bts, ots in SCALAR_ENTRIES:
        a, b = gemm_operands(ats[0], bts[0], 4, 8, 4)
        tail = (gq.kQdqActNone, 0.0, 0.0) if has_act(entry) else ()
        good = [T[ats[0]][0], T[bts[0]][0], T[ots[0]][0], I64(4)]
        bad = [[NPU_I32] + good[1:], good[:1] + [T["i16"][0]] + good[2:],
               good[:2] + [T["i16" if ots[0] in ("i8", "u8") else "i8"][0], good[3]],
               good[:3] + [I64(-1)]]
        if has_act(entry):
            for act in (gq.kQdqActGelu, gq.kQdqActPrelu, 1):
                assert rejected(entry, a, good[0], b, good[1], OUT, good[2], good[3], I64(8),
                                I64(4), 0.0, 0.0, 1.0, 0.0, act, 0.0, 0.0, nbytes=64)
        for at_, bt_, ot_, m_ in bad:
            assert rejected(entry, a, at_, b, bt_, OUT, ot_, m_, I64(8), I64(4), 0.0, 0.0, 1.0, 0.0,
                            *tail, nbytes=64), (entry, at_, bt_, ot_)
        assert raw_rc(entry, None, good[0], None, good[1], None, good[2], I64(4), I64(8), I64(4),
                      0.0, 0.0, 1.0, 0.0, *tail) == -1
    # hybrid: mode_a_cols outside [0, N], null c0_b / c1, GELU
    a, b = gemm_operands("i16", "i8", 4, 8, 4)
    c = np.ones(4, f32)
    for entry in ("npu_gemm_a16w8_hybrid", "npu_gemm_a16w8_hybrid_channelwise"):
        for cols, c0b, c1, act in ((5, c, c, 0), (-1, c, c, 0), (2, None, c, 0), (2, c, None, 0),
                                   (2, c, c, gq.kQdqActGelu)):
            assert rejected(entry, a, NPU_I16, b, NPU_I8, OUT, NPU_I16, I64(4), I64(8), I64(4),
                            I64(cols), c, c0b, c1, c, act, 0.0, 0.0, nbytes=32)
        assert rejected(entry, a, NPU_I16, b, NPU_I16, OUT, NPU_I16, I64(4), I64(8), I64(4),
                        I64(2), c, c, c, c, 0, 0.0, 0.0, nbytes=32)
    # int16 x int16: 8-bit operand, GELU
    a16, b16 = rnd("i16", (4, 8)), rnd("i16", (8, 4))
    for entry in ("npu_gemm_qdq_int16x16", "npu_gemm_qdq_a16a16_transpose"):
        for bt, act in ((NPU_I8, 0), (NPU_I16, gq.kQdqActGelu), (NPU_I16, gq.kQdqActPrelu)):
            assert rejected(entry, a16, NPU_I16, b16, bt, OUT, NPU_I16, I64(4), I64(8), I64(4), 0.0,
                            0.0, 1.0, 0.0, act, 0.0, 0.0, nbytes=32)
    # gemm + add: 16-bit addend, kpass 2, null c2; a16w8 add: zero size, null residual
    a8, b8, i3 = rnd("u8", (4, 8)), rnd("i8", (8, 4)), rnd("u8", (4, 4))
    for it3, kp, c2 in ((NPU_U16, 0, c), (NPU_U8, 2, c), (NPU_U8, 0, None)):
        assert rejected("npu_qdq_gemm_add_a8w8", a8, NPU_U8, b8, NPU_I8, i3, it3, OUT, NPU_U8,
                        I64(4), I64(8), I64(4), c, None, c2, 1.0, 1.0, 0.0, kp, nbytes=16)
    au, res = rnd("u16", (4, 8)), rnd("u16", (4, 4))
    assert rejected("npu_gemm_add_a16w8", au, b8, res, OUT, I64(4), I64(0), I64(4), c, None, c,
                    1.0, 1.0, 0.0, nbytes=32)
    assert raw_rc("npu_gemm_add_a16w8", None, None, None, None, I64(4), I64(8), I64(4), None, None,
                  None, 1.0, 1.0, 0.0) == -1
    _empty_work()


def _empty_work():
    """M or N = 0 is an empty output: 0 without a launch once every other argument is valid;
    null required pointers and K = 0 are still rejected."""
    p, c = np.zeros(64, np.uint8), np.ones(4, f32)
    for entry, kind, ats, bts, ots in VEC_ENTRIES:
        ty = (T[ats[0]][0], T[STORE.get(bts[0], bts[0])][0], T[ots[0]][0])
        for m, k, n in ((0, 8, 4), (4, 8, 0)):
            assert raw_rc(entry, p, ty[0], p, ty[1], p, ty[2], I64(m), I64(k), I64(n), c, None, c,
                          0, 0.0, 0.0) == 0, entry
        assert raw_rc(entry, p, ty[0], p, ty[1], None, ty[2], I64(0), I64(8), I64(4), c, None, c,
                      0, 0.0, 0.0) == -1, entry
        assert raw_rc(entry, p, ty[0], p, ty[1], p, ty[2], I64(0), I64(8), I64(4), None, None, c,
                      0, 0.0, 0.0) == -1, entry
        assert raw_rc(entry, p, ty[0], p, ty[1], p, ty[2], I64(0), I64(0), I64(4), c, None, c,
                      0, 0.0, 0.0) == -1, entry
    for entry, kind, ats, bts, ots in SCALAR_ENTRIES:
        tail = (gq.kQdqActNone, 0.0, 0.0) if has_act(entry) else ()
        ty = (T[ats[0]][0], T[bts[0]][0], T[ots[0]][0])
        assert raw_rc(entry, p, ty[0], p, ty[1], p, ty[2], I64(0), I64(8), I64(4), 0.0, 0.0, 1.0,
                      0.0, *tail) == 0, entry
        assert raw_rc(entry, None, ty[0], p, ty[1], p, ty[2], I64(0), I64(8), I64(4), 0.0, 0.0,
                      1.0, 0.0, *tail) == -1, entry
    for entry in ("npu_gemm_a16w8_hybrid", "npu_gemm_a16w8_hybrid_channelwise"):
        assert raw_rc(entry, p, NPU_I16, p, NPU_I8, p, NPU_I16, I64(0), I64(8), I64(4), I64(2), c,
                      c, c, c, 0, 0.0, 0.0) == 0
        assert raw_rc(entry, p, NPU_I16, p, NPU_I8, p, NPU_I16, I64(0), I64(8), I64(4), I64(2), c,
                      None, c, c, 0, 0.0, 0.0) == -1
    for entry in ("npu_gemm_qdq_int16x16", "npu_gemm_qdq_a16a16_transpose"):
        assert raw_rc(entry, p, NPU_I16, p, NPU_I16, p, NPU_I16, I64(4), I64(8), I64(0), 0.0, 0.0,
                      1.0, 0.0, 0, 0.0, 0.0) == 0
    assert raw_rc("npu_qdq_gemm_add_a8w8", p, NPU_U8, p, NPU_I8, p, NPU_U8, p, NPU_U8, I64(0),
                  I64(8), I64(4), c, None, c, 1.0, 1.0, 0.0, 0) == 0
    assert raw_rc("npu_qdq_gemm_add_a8w8", p, NPU_U8, p, NPU_I8, None, NPU_U8, p, NPU_U8, I64(0),
                  I64(8), I64(4), c, None, c, 1.0, 1.0, 0.0, 0) == -1
    assert raw_rc("npu_gemm_add_a16w8", p, p, p, p, I64(4), I64(8), I64(0), c, None, c, 1.0, 1.0,
                  0.0) == 0
    assert raw_rc("npu_gemm_add_a16w8", p, p, None, p, I64(4), I64(8), I64(0), c, None, c, 1.0,
                  1.0, 0.0) == -1
    assert raw_rc("npu_gemm_add_a16w8", p, p, p, p, I64(-1), I64(8), I64(4), c, None, c, 1.0, 1.0,
                  0.0) == -1
