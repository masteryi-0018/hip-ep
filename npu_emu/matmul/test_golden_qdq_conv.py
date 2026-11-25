# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The QDQ conv, conv + add and depthwise entry points against golden_qdq_conv.py, bit for bit."""

import numpy as np
import pytest

import golden_act as ga
import golden_quant as gq
import golden_qdq_conv as gm
from npu import I64, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT, raw_rc
from qdq_testing import ACTS, NS, RNG, SPECIAL, T, assert_same, f32, rejected, rnd, run

def model(entry):
    return getattr(gm, entry[len("npu_"):])


CONV = [  # (kh, kw, stride, k_opt); k_opt only on a 3 x 3 / 7 x 7 window
    (3, 3, 1, 1), (3, 3, 2, 1), (3, 3, 1, 0), (7, 7, 2, 1), (1, 1, 1, 0), (2, 2, 1, 0),
    (4, 4, 2, 0), (5, 5, 1, 0), (3, 1, 1, 0), (7, 7, 1, 0)]


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
        kind, a0, a1 = gq.kQdqActPrelu, 3000.0 if split16 else 100.0, 0.0
        slope = RNG.uniform(-0.5, 1.5, co).astype(f32)
    else:
        kind, a0, a1 = ACTS[act]
    name = "npu_conv_qdq_int16x8" if split16 else "npu_conv_qdq_int8x8"
    want = model(name)(ifm, wts, ot, s, s, k_opt, c0, c1, c2, kind, a0, a1, slope)
    got = run(name, ifm, T[it][0], wts, T["u8" if split16 else "i8"][0], OUT, T[ot][0], I64(h),
              I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s, k_opt, c0, c1, c2, kind, a0, a1,
              slope, n=want.size, t=ot)
    assert_same(got, want)


def test_conv_no_actv_sum():
    ifm, wts = rnd("i8", (5, 6, 32)), rnd("i8", (3, 3, 32, 16))
    c0, c2 = RNG.uniform(-50, 50, 16).astype(f32), RNG.uniform(0.01, 0.02, 16).astype(f32)
    want = gm.conv_qdq_int8x8(ifm, wts, "i8", 1, 1, 0, c0, None, c2)
    got = run("npu_conv_qdq_int8x8", ifm, NPU_I8, wts, NPU_I8, OUT, NPU_I8, I64(5), I64(6),
              I64(32), I64(3), I64(3), I64(16), 1, 1, 0, c0, None, c2, gq.kQdqActNone, 0.0, 0.0,
              None, n=want.size, t="i8")
    assert_same(got, want)


def test_k_opt_order_matters():
    """The two window-sum orders differ on these inputs."""
    ifm = rnd("u16", (5, 5, 64), 60000, 65535).astype(np.int64)
    a = gm.window_sum(ifm, 3, 3, 1, 1, False)
    b = gm.window_sum(ifm, 3, 3, 1, 1, True)
    assert np.any(a != b)


CHAIN_ACTS = [(gq.kQdqActNone, 0.0, 0.0), (gq.kQdqActClip, -3.5, 100.5),
              (gq.kQdqActLrelu, 0.1015625, -3.59375)]


def add_act_ok(split16, kind, before):
    """The conv + add kernels: CLIP only on the 8-bit one, an activation before the add only
    as LRELU."""
    return (not split16 or kind != gq.kQdqActClip) and (not before or kind == gq.kQdqActLrelu)


ADD_CASES = [(split16, act, before) for split16 in (False, True) for act in range(len(CHAIN_ACTS))
             for before in (0, 1) if add_act_ok(split16, CHAIN_ACTS[act][0], before)]


def conv_add(split16, ifm, it, wts, wt, ifm2, i2t, ot, h, w, ci, kh, kw, co, s, k_opt, c0, c1, c2,
             add, before, kind, a0, a1, n):
    name = "npu_qdq_conv_add_a16w8" if split16 else "npu_qdq_conv_add_a8w8"
    return run(name, ifm, T[it][0], wts, wt, ifm2, i2t, OUT, T[ot][0], I64(h), I64(w), I64(ci),
               I64(kh), I64(kw), I64(co), s, s, k_opt, c0, c1, c2, *(float(x) for x in add),
               before, kind, a0, a1, n=n, t=ot)


@pytest.mark.parametrize("split16,act,before", ADD_CASES,
                         ids=[f"{'a16w8' if c[0] else 'a8w8'}-{c[1]}-{c[2]}" for c in ADD_CASES])
@pytest.mark.parametrize("geom", [0, 1, 4, 6])
def test_conv_add(split16, geom, act, before):
    kh, kw, s, k_opt = CONV[geom]
    ho, wo, ci, co = 3, 4, 64 if split16 else 32, 48
    h, w = (ho - 1) * s + kh, (wo - 1) * s + kw
    it, ot = ("u16", "i16") if split16 else ("i8", "u8")
    ifm = rnd(it, (h, w, ci), 40000, 65535) if split16 else rnd(it, (h, w, ci))
    wts = rnd("u8" if split16 else "i8", (kh, kw, ci, co))
    ifm2 = rnd("u16" if split16 else "i8", (ho, wo, co))
    c2 = (RNG.uniform(1e-5, 4e-5, co) if split16 else RNG.uniform(0.002, 0.02, co)).astype(f32)
    c1 = (-c2 * RNG.integers(100, 200, co)).astype(f32)
    c0 = (RNG.uniform(-1e5, 1e5, co) if split16 else RNG.uniform(-200, 200, co)).astype(f32)
    # add words with a nondyadic c1 and a .5 offset, so many outputs land on ties
    ac0, ac1, ac2 = f32(1.0), f32(0.375 if split16 else 0.75), f32(-12.5)
    kind, a0, a1 = CHAIN_ACTS[act]
    m = gm.qdq_conv_add_a16w8 if split16 else gm.qdq_conv_add_a8w8
    want = m(ifm, wts, ifm2, ot, s, s, k_opt, c0, c1, c2, ac0, ac1, ac2, before, kind, a0, a1)
    got = conv_add(split16, ifm, it, wts, NPU_U8 if split16 else NPU_I8, ifm2,
                   NPU_U16 if split16 else NPU_I8, ot, h, w, ci, kh, kw, co, s, k_opt, c0, c1, c2,
                   (ac0, ac1, ac2), before, kind, a0, a1, want.size)
    assert_same(got, want)


@pytest.mark.parametrize("ot", ["i8", "u8"])
@pytest.mark.parametrize("geom", [0, 1, 4])
def test_conv_q_add(ot, geom):
    kh, kw, s, _ = CONV[geom]
    ho, wo, ci, co = 3, 4, 32, 48
    h, w = (ho - 1) * s + kh, (wo - 1) * s + kw
    ifm, wts, ifm2 = rnd("i8", (h, w, ci)), rnd("i8", (kh, kw, ci, co)), rnd(ot, (ho, wo, co))
    c2 = RNG.uniform(0.002, 0.02, co).astype(f32)
    c0 = RNG.uniform(-50, 150, co).astype(f32)
    ac0, ac1, ac2 = f32(0.625), f32(0.375), f32(3.5)
    want = gm.qdq_conv_q_add_a8w8(ifm, wts, ifm2, ot, s, s, c0, c2, ac0, ac1, ac2)
    got = run("npu_qdq_conv_q_add_a8w8", ifm, NPU_I8, wts, NPU_I8, ifm2, T[ot][0], OUT,
              T[ot][0], I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s, c0, c2,
              float(ac0), float(ac1), float(ac2), n=want.size, t=ot)
    assert_same(got, want)


DWC = [(3, 3, 1), (3, 3, 2), (5, 5, 1), (7, 7, 1), (11, 11, 2), (1, 1, 1)]  # kh, kw, stride


@pytest.mark.parametrize("types", [("i8", "i8", "i8"), ("u8", "u8", "u8"), ("i8", "u8", "i8"),
                                   ("i16", "i8", "i16"), ("u16", "u8", "u16")])
@pytest.mark.parametrize("geom", range(len(DWC)))
@pytest.mark.parametrize("act", range(len(CHAIN_ACTS)))
@pytest.mark.parametrize("actv_sum", [0, 1])
def test_qdq_dwc(types, geom, act, actv_sum):
    it, wt, ot = types
    kh, kw, s = DWC[geom]
    ho, wo, c = 4, 5, 64
    h, w = (ho - 1) * s + kh, (wo - 1) * s + kw
    split16 = it in ("i16", "u16")
    ifm, wts = rnd(it, (h, w, c)), rnd(wt, (kh, kw, c))
    zp_w = 128 if wt == "u8" else -7
    c2 = (RNG.uniform(1e-6, 4e-6, c) if split16 else RNG.uniform(0.002, 0.02, c)).astype(f32)
    c0 = (RNG.uniform(-2e4, 2e4, c) if split16 else RNG.uniform(-100, 200, c)).astype(f32)
    kind, a0, a1 = CHAIN_ACTS[act]
    fn = gm.qdq_dwc_int16x8 if split16 else gm.qdq_dwc_int8x8
    want = fn(ifm, wts, ot, s, s, zp_w, actv_sum, c0, c2, kind, a0, a1)
    got = run("npu_qdq_dwc_int16x8" if split16 else "npu_qdq_dwc_int8x8", ifm, T[it][0], wts,
              T[wt][0], OUT, T[ot][0], I64(h), I64(w), I64(c), I64(kh), I64(kw), s, s, zp_w, actv_sum,
              c0, c2, kind, a0, a1, n=want.size, t=ot)
    assert_same(got, want)


@pytest.mark.parametrize("a16", [False, True], ids=["a8w8", "a16w8"])
@pytest.mark.parametrize("s", [1, 2])
@pytest.mark.parametrize("act", range(len(CHAIN_ACTS)))
def test_depthwise_conv_s(a16, s, act):
    """The 3 x 3 fixed-stride entries; each rejects the input width of the other family."""
    it, wt = ("u16", "u8") if a16 else ("i8", "i8")
    ho, wo, c = 4, 5, 64
    h, w = (ho - 1) * s + 3, (wo - 1) * s + 3
    ifm, wts = rnd(it, (h, w, c)), rnd(wt, (3, 3, c))
    zp_w = 128 if wt == "u8" else -7
    c2 = (RNG.uniform(1e-6, 4e-6, c) if a16 else RNG.uniform(0.002, 0.02, c)).astype(f32)
    c0 = (RNG.uniform(-2e4, 2e4, c) if a16 else RNG.uniform(-100, 200, c)).astype(f32)
    kind, a0, a1 = CHAIN_ACTS[act]
    name = f"npu_depthwise_conv_a{16 if a16 else 8}w8_s{s}"
    want = model(name)(ifm, wts, it, zp_w, *((1,) if a16 else ()), c0, c2, kind, a0, a1)
    sums = (1,) if a16 else ()  # the 8-bit kernels always apply the window sum
    args = (wts, T[wt][0], OUT, T[it][0], I64(h), I64(w), I64(c), zp_w, *sums, c0, c2, kind, a0, a1)
    got = run(name, ifm, T[it][0], *args, n=want.size, t=it)
    assert_same(got, want)
    other = "i8" if a16 else "u16"
    assert rejected(name, ifm.astype(T[other][1]), T[other][0], *args, nbytes=want.nbytes)


# ---- special values: the float words (c0 / c1 / c2, act0 / act1, slope, add words) take NaN /
# inf / zeros / subnormals, operands sit at the integer limits; integer outputs, so NaN -> 0 and
# +-inf saturates ([default] float -> int conversion).
import golden_qdq_gemm as gg  # noqa: E402
from test_golden_qdq_gemm import COEF_SETS, special_cols  # noqa: E402

ACT_SETS = [(gq.kQdqActNone, 0.0, 0.0), (gq.kQdqActClip, float(SPECIAL[0]), 100.5),
            (gq.kQdqActClip, -3.5, float(SPECIAL[3])),
            (gq.kQdqActClip, float(SPECIAL[4]), float(SPECIAL[3])),
            (gq.kQdqActLrelu, float(SPECIAL[0]), 2.0),
            (gq.kQdqActLrelu, float(SPECIAL[3]), float(SPECIAL[4])),
            (gq.kQdqActLrelu, float(SPECIAL[6]), float(SPECIAL[0])),
            (gq.kQdqActLrelu, 0.1015625, float(SPECIAL[3]))]


def special_ifm(it, shape):
    x = rnd(it, shape)
    info = np.iinfo(x.dtype)
    x[0, 0], x[1, 1] = info.min, info.max
    return x


def add_sets(base):
    s = [[float(SPECIAL[i]) if w == j else base[j] for j in range(3)] for w in range(3) for i in range(NS)]
    return s + [[float(SPECIAL[i]), float(SPECIAL[(3 * i + 1) % NS]), float(SPECIAL[(5 * i + 2) % NS])]
                for i in range(NS)]


@pytest.mark.parametrize("split16", [False, True], ids=["int8x8", "int16x8"])
def test_conv_special(split16):
    kh = kw = 3
    ho = wo = 2
    ci, co = 16, NS * NS
    h, w = ho + kh - 1, wo + kw - 1
    it, ot = ("u16", "i16") if split16 else ("i8", "u8")
    rng = np.random.default_rng(2050)
    ifm = special_ifm(it, (h, w, ci))
    wts = rnd("u8" if split16 else "i8", (kh, kw, ci, co))
    name = "npu_conv_qdq_int16x8" if split16 else "npu_conv_qdq_int8x8"
    wt = NPU_U8 if split16 else NPU_I8

    def check(c0, c1, c2, kind, a0, a1, slope=None):
        k_opt = int(c1 is not None)
        want = model(name)(ifm, wts, ot, 1, 1, k_opt, c0, c1, c2, kind, a0, a1, slope)
        got = run(name, ifm, T[it][0], wts, wt, OUT, T[ot][0], I64(h), I64(w), I64(ci), I64(kh),
                  I64(kw), I64(co), 1, 1, k_opt, c0, c1, c2, kind, a0, a1, slope, n=want.size, t=ot)
        assert_same(got, want)

    kind = gg.I8 if not split16 else gg.I16_JOIN
    for which in COEF_SETS:
        c0, c1, c2, _ = special_cols(co, which, kind, rng)
        for k_, a0, a1 in ACT_SETS[:1] + ACT_SETS[4:5]:
            check(c0, c1, c2, k_, a0, a1)
        check(c0, None, c2, gq.kQdqActNone, 0.0, 0.0)
    c0, c1, c2, _ = special_cols(co, "c0", kind)
    for k_, a0, a1 in ACT_SETS:
        check(c0, c1, c2, k_, a0, a1)
    # PRELU: the slope per channel, the pivot act0
    slope = SPECIAL[np.arange(co) % NS].copy()
    for a0 in (SPECIAL[:NS].tolist() + [100.0]):
        check(c0, c1, c2, gq.kQdqActPrelu, float(a0), 0.0, slope)


@pytest.mark.parametrize("split16", [False, True], ids=["a8w8", "a16w8"])
@pytest.mark.parametrize("before", [0, 1])
def test_conv_add_special(split16, before):
    kh = kw = 3
    ho = wo = 2
    ci, co = 16, NS * NS
    h, w = ho + kh - 1, wo + kw - 1
    it, ot = ("u16", "i16") if split16 else ("i8", "u8")
    rng = np.random.default_rng(2051)
    ifm = special_ifm(it, (h, w, ci))
    wts = rnd("u8" if split16 else "i8", (kh, kw, ci, co))
    ifm2 = rnd("u16" if split16 else "i8", (ho, wo, co))
    ifm2[0, 0], ifm2[1, 1] = (0, 65535) if split16 else (-128, 127)

    def check(c0, c1, c2, kind, a0, a1, add):
        if not add_act_ok(split16, kind, before):
            return
        k_opt = int(c1 is not None)
        m = gm.qdq_conv_add_a16w8 if split16 else gm.qdq_conv_add_a8w8
        want = m(ifm, wts, ifm2, ot, 1, 1, k_opt, c0, c1, c2, *(f32(x) for x in add), before, kind,
                 a0, a1)
        got = conv_add(split16, ifm, it, wts, NPU_U8 if split16 else NPU_I8, ifm2,
                       NPU_U16 if split16 else NPU_I8, ot, h, w, ci, kh, kw, co, 1, k_opt, c0, c1,
                       c2, add, before, kind, a0, a1, want.size)
        assert_same(got, want)

    kind = gg.I16_JOIN if split16 else gg.I8
    base_add = (1.0, 0.375, -12.5)
    c0, c1, c2, _ = special_cols(co, "c0", kind)
    for add in add_sets(base_add):
        check(c0, c1, c2, gq.kQdqActNone, 0.0, 0.0, add)
    for which in COEF_SETS:
        c0, c1, c2, _ = special_cols(co, which, kind, rng)
        for k_, a0, a1 in ACT_SETS[:1] + ACT_SETS[4:5]:
            check(c0, c1, c2, k_, a0, a1, base_add)
    c0, c1, c2, _ = special_cols(co, "c0", kind)
    for k_, a0, a1 in ACT_SETS:
        check(c0, c1, c2, k_, a0, a1, base_add)


@pytest.mark.parametrize("ot", ["i8", "u8"])
def test_conv_q_add_special(ot):
    kh = kw = 3
    ho = wo = 2
    ci, co = 16, NS * NS
    h, w = ho + kh - 1, wo + kw - 1
    rng = np.random.default_rng(2052)
    ifm, wts = special_ifm("i8", (h, w, ci)), rnd("i8", (kh, kw, ci, co))
    ifm2 = rnd(ot, (ho, wo, co))
    ifm2[0, 0], ifm2[1, 1] = (-128, 127) if ot == "i8" else (0, 255)

    def check(c0, c2, add):
        want = gm.qdq_conv_q_add_a8w8(ifm, wts, ifm2, ot, 1, 1, c0, c2, *(f32(x) for x in add))
        got = run("npu_qdq_conv_q_add_a8w8", ifm, NPU_I8, wts, NPU_I8, ifm2, T[ot][0], OUT, T[ot][0],
                  I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), 1, 1, c0, c2, *add,
                  n=want.size, t=ot)
        assert_same(got, want)

    c0, _, c2, _ = special_cols(co, "c0", gg.I8)
    for add in add_sets((0.625, 0.375, 3.5)):
        check(c0, c2, add)
    for which in COEF_SETS:
        c0, _, c2, _ = special_cols(co, which, gg.I8, rng)
        check(c0, c2, (0.625, 0.375, 3.5))


@pytest.mark.parametrize("types", [("i8", "i8", "i8"), ("u8", "u8", "u8"), ("i16", "i8", "i16"),
                                   ("u16", "u8", "u16")])
@pytest.mark.parametrize("actv_sum", [0, 1])
def test_qdq_dwc_special(types, actv_sum):
    it, wt, ot = types
    kh = kw = 3
    ho = wo = 2
    c = NS * NS
    h, w = ho + kh - 1, wo + kw - 1
    split16 = it in ("i16", "u16")
    rng = np.random.default_rng(2053)
    ifm, wts = special_ifm(it, (h, w, c)), rnd(wt, (kh, kw, c))
    zp_w = 128 if wt == "u8" else -7
    fn = gm.qdq_dwc_int16x8 if split16 else gm.qdq_dwc_int8x8
    kind = gg.I16_JOIN if split16 else gg.I8

    def check(c0, c2, k_, a0, a1, entry="npu_qdq_dwc_int16x8" if split16 else "npu_qdq_dwc_int8x8"):
        want = fn(ifm, wts, ot, 1, 1, zp_w, actv_sum, c0, c2, k_, a0, a1)
        got = run(entry, ifm, T[it][0], wts, T[wt][0], OUT, T[ot][0], I64(h), I64(w), I64(c),
                  I64(kh), I64(kw), 1, 1, zp_w, actv_sum, c0, c2, k_, a0, a1, n=want.size, t=ot)
        assert_same(got, want)

    for which in COEF_SETS:
        c0, _, c2, _ = special_cols(c, which, kind, rng)
        check(c0, c2, gq.kQdqActNone, 0.0, 0.0)
        check(c0, c2, gq.kQdqActLrelu, 0.1015625, -3.59375)
    c0, _, c2, _ = special_cols(c, "c0", kind)
    for k_, a0, a1 in ACT_SETS:
        check(c0, c2, k_, a0, a1)


@pytest.mark.parametrize("a16", [False, True], ids=["a8w8", "a16w8"])
@pytest.mark.parametrize("s", [1, 2])
def test_depthwise_conv_s_special(a16, s):
    it, wt = ("u16", "u8") if a16 else ("i8", "i8")
    ho, wo, c = 2, 2, NS * NS
    h, w = (ho - 1) * s + 3, (wo - 1) * s + 3
    rng = np.random.default_rng(2054)
    ifm, wts = special_ifm(it, (h, w, c)), rnd(wt, (3, 3, c))
    zp_w = 128 if wt == "u8" else -7
    name = f"npu_depthwise_conv_a{16 if a16 else 8}w8_s{s}"
    sums = (1,) if a16 else ()
    kind = gg.I16_JOIN if a16 else gg.I8
    for which in COEF_SETS:
        c0, _, c2, _ = special_cols(c, which, kind, rng)
        for k_, a0, a1 in ACT_SETS[:1] + ACT_SETS[4:7]:
            want = model(name)(ifm, wts, it, zp_w, *sums, c0, c2, k_, a0, a1)
            got = run(name, ifm, T[it][0], wts, T[wt][0], OUT, T[it][0], I64(h), I64(w), I64(c), zp_w,
                      *sums, c0, c2, k_, a0, a1, n=want.size, t=it)
            assert_same(got, want)


def test_bad_args():
    h = w = 5
    ci = co = 8
    c = np.ones(co, f32)
    for split16 in (False, True):
        it, ot, ot_bad = ("i16", "i16", "i8") if split16 else ("i8", "i8", "i16")
        name = "npu_conv_qdq_int16x8" if split16 else "npu_conv_qdq_int8x8"
        ifm, wts = rnd(it, (h, w, ci)), rnd("i8", (3, 3, ci, co))
        ok = dict(ot=ot, k=3, s=1, k_opt=0, c1=c, act=gq.kQdqActNone, slope=None)
        for d in (dict(ot=ot_bad), dict(s=0), dict(k=6), dict(k_opt=2), dict(k_opt=1, c1=None),
                  dict(k=2, k_opt=1), dict(act=gq.kQdqActGelu), dict(act=1),
                  dict(act=gq.kQdqActPrelu), dict(slope=c)):
            p = {**ok, **d}
            assert rejected(name, ifm, T[it][0], wts, NPU_I8, OUT, T[p["ot"]][0], I64(h), I64(w),
                            I64(ci), I64(p["k"]), I64(p["k"]), I64(co), p["s"], p["s"], p["k_opt"],
                            c, p["c1"], c, p["act"], 0.0, 0.0, p["slope"], nbytes=1024), (name, d)
        assert raw_rc(name, None, T[it][0], None, NPU_I8, None, T[ot][0], I64(h), I64(w), I64(ci),
                      I64(3), I64(3), I64(co), 1, 1, 0, None, None, None, 0, 0.0, 0.0, None) == -1
        # conv + add: ifm2 / out width, CLIP before the add, CLIP on 16 bits, act_before_add 2
        add = "npu_qdq_conv_add_a16w8" if split16 else "npu_qdq_conv_add_a8w8"
        ifm2 = rnd(it, (3, 3, co))
        ok = dict(i2t=it, ot=ot, before=0, act=gq.kQdqActNone)
        bads = [dict(i2t=ot_bad), dict(ot=ot_bad), dict(before=1, act=gq.kQdqActNone),
                dict(before=1, act=gq.kQdqActClip), dict(before=2, act=gq.kQdqActLrelu),
                dict(act=gq.kQdqActPrelu)] + ([dict(act=gq.kQdqActClip)] if split16 else [])
        for d in bads:
            p = {**ok, **d}
            assert rejected(add, ifm, T[it][0], wts, NPU_I8, ifm2, T[p["i2t"]][0], OUT,
                            T[p["ot"]][0], I64(h), I64(w), I64(ci), I64(3), I64(3), I64(co), 1, 1,
                            0, c, c, c, 1.0, 1.0, 0.0, p["before"], p["act"], 0.0, 0.0,
                            nbytes=1024), (add, d)
        # depthwise: out width, zp_w outside the weight range, actv_sum 2, GELU
        dw, dwts = rnd(it, (h, w, co)), rnd("i8", (3, 3, co))
        name = "npu_qdq_dwc_int16x8" if split16 else "npu_qdq_dwc_int8x8"
        for wt, o, zp, asum, act in ((NPU_I8, ot_bad, 0, 1, 0), (NPU_I8, ot, 128, 1, 0),
                                     (NPU_U8, ot, -1, 1, 0), (NPU_U8, ot, 256, 1, 0),
                                     (NPU_I8, ot, 0, 2, 0), (NPU_I8, ot, 0, 1, gq.kQdqActGelu)):
            assert rejected(name, dw, T[it][0], dwts, wt, OUT, T[o][0], I64(h), I64(w),
                            I64(co), I64(3), I64(3), 1, 1, zp, asum, c, c, act, 0.0, 0.0,
                            nbytes=1024)
        for s_ in (1, 2):
            name = f"npu_depthwise_conv_a{16 if split16 else 8}w8_s{s_}"
            sums = (1,) if split16 else ()
            assert rejected(name, dw, T[it][0], dwts, NPU_I8, OUT, T[ot_bad][0], I64(h),
                            I64(w), I64(co), 0, *sums, c, c, 0, 0.0, 0.0, nbytes=1024)
            assert rejected(name, dw, T[it][0], dwts, NPU_U8, OUT, T[ot][0], I64(h),
                            I64(w), I64(co), 300, *sums, c, c, 0, 0.0, 0.0, nbytes=1024)
    # requantize + add: 16-bit ifm2 / out, null ifm2
    ifm, wts = rnd("i8", (h, w, ci)), rnd("i8", (3, 3, ci, co))
    for i2, i2t, o in ((rnd("i16", (3, 3, co)), NPU_I16, NPU_I8), (rnd("i8", (3, 3, co)), NPU_I8,
                                                                    NPU_I16), (None, NPU_I8, NPU_I8)):
        assert rejected("npu_qdq_conv_q_add_a8w8", ifm, NPU_I8, wts, NPU_I8, i2, i2t, OUT, o, I64(h),
                        I64(w), I64(ci), I64(3), I64(3), I64(co), 1, 1, c, c, 1.0, 1.0, 0.0,
                        nbytes=1024)
    _sizes()


def _sizes():
    """H or W = 0 is an empty output: 0 without a launch once every other argument is valid;
    negative sizes, 0 < H < Kh and null required pointers stay -1, also when empty."""
    p, c = np.zeros(64, np.uint8), np.ones(8, f32)

    def call(name, ifm=p, out=p, c0=c, h=0, w=5):
        it, ot = (NPU_I16, NPU_I16) if "16" in name else (NPU_I8, NPU_I8)
        hw = (I64(h), I64(w))
        if name.startswith("npu_conv_qdq"):
            args = (ifm, it, p, NPU_I8, out, ot, *hw, I64(8), I64(3), I64(3), I64(8), 1, 1, 0, c0,
                    None, c, 0, 0.0, 0.0, None)
        elif name == "npu_qdq_conv_q_add_a8w8":
            args = (ifm, it, p, NPU_I8, p, it, out, ot, *hw, I64(8), I64(3), I64(3), I64(8), 1, 1,
                    c0, c, 1.0, 1.0, 0.0)
        elif "conv_add" in name:
            args = (ifm, it, p, NPU_I8, p, it, out, ot, *hw, I64(8), I64(3), I64(3), I64(8), 1, 1,
                    0, c0, None, c, 1.0, 1.0, 0.0, 0, 0, 0.0, 0.0)
        elif "dwc" in name:
            args = (ifm, it, p, NPU_I8, out, ot, *hw, I64(8), I64(3), I64(3), 1, 1, 0, 1, c0, c, 0,
                    0.0, 0.0)
        else:
            sums = (1,) if "a16" in name else ()
            args = (ifm, it, p, NPU_I8, out, ot, *hw, I64(8), 0, *sums, c0, c, 0, 0.0, 0.0)
        return raw_rc(name, *args)

    for name in ("npu_conv_qdq_int8x8", "npu_conv_qdq_int16x8", "npu_qdq_conv_add_a8w8",
                 "npu_qdq_conv_add_a16w8", "npu_qdq_conv_q_add_a8w8", "npu_qdq_dwc_int8x8",
                 "npu_qdq_dwc_int16x8", "npu_depthwise_conv_a8w8_s1", "npu_depthwise_conv_a8w8_s2",
                 "npu_depthwise_conv_a16w8_s1", "npu_depthwise_conv_a16w8_s2"):
        assert call(name) == 0, name
        assert call(name, h=5, w=0) == 0, name
        assert call(name, h=-1, w=5) == -1, name
        assert call(name, h=5, w=-1) == -1, name
        assert call(name, h=2, w=5) == -1, name
        assert call(name, ifm=None) == -1, name
        assert call(name, out=None) == -1, name
        assert call(name, c0=None) == -1, name
