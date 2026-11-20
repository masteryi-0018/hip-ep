# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The QDQ conv, conv + add and depthwise entry points against golden_qdq_conv.py, bit for bit."""

import numpy as np
import pytest

import golden_act as ga
import golden_qdq_conv as gm
from npu import I64, NPU_I8, NPU_U8, NPU_U16, OUT
from qdq_testing import ACTS, RNG, T, assert_same, f32, rejected, rnd, run

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
        kind, a0, a1 = ga.ACT_PRELU, 3000.0 if split16 else 100.0, 0.0
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
              I64(32), I64(3), I64(3), I64(16), 1, 1, 1, c0, None, c2, ga.ACT_NONE, 0.0, 0.0, None,
              n=want.size, t="i8")
    assert_same(got, want)


def test_k_opt_order_matters():
    """The two window-sum orders differ on these inputs."""
    ifm = rnd("u16", (5, 5, 64), 60000, 65535).astype(np.int64)
    a = gm.window_sum(ifm, 3, 3, 1, 1, False)
    b = gm.window_sum(ifm, 3, 3, 1, 1, True)
    assert np.any(a != b)


CHAIN_ACTS = [(ga.ACT_NONE, 0.0, 0.0), (ga.ACT_CLIP, -3.5, 100.5),
              (ga.ACT_LRELU, 0.1015625, -3.59375)]


@pytest.mark.parametrize("split16", [False, True], ids=["a8w8", "a16w8"])
@pytest.mark.parametrize("geom", [0, 1, 4, 6])
@pytest.mark.parametrize("act", range(len(CHAIN_ACTS)))
@pytest.mark.parametrize("before", [0, 1])
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
    want = gm.conv_qdq_add(ifm, wts, ifm2, ot, s, s, c0, c1, c2, split16, kind, a0, a1, before,
                           ac0, ac1, ac2, k_opt=bool(k_opt))
    if split16:
        got = run("npu_qdq_conv_add_a16w8", ifm, T[it][0], wts, NPU_U8, ifm2, NPU_U16, OUT,
                  T[ot][0], I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s, c0, c1, c2,
                  kind, a0, a1, before, float(ac0), float(ac1), float(ac2), n=want.size, t=ot)
    else:
        got = run("npu_qdq_conv_add_a8w8", ifm, T[it][0], wts, NPU_I8, ifm2, NPU_I8, OUT,
                  T[ot][0], I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s, k_opt, c0,
                  c1, c2, kind, a0, a1, before, float(ac0), float(ac1), float(ac2), n=want.size,
                  t=ot)
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
    want = gm.conv_qdq_q_add(ifm, wts, ifm2, ot, s, s, c0, c2, ac0, ac1, ac2)
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
    fn = gm.qdq_dwc16 if split16 else gm.qdq_dwc
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
    want = (gm.qdq_dwc16 if a16 else gm.qdq_dwc)(ifm, wts, it, s, s, zp_w, 1, c0, c2, kind, a0, a1)
    name = f"npu_depthwise_conv_a{16 if a16 else 8}w8_s{s}"
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
from test_golden_qdq_gemm import COEF_SETS, NS, SPECIAL, special_cols  # noqa: E402

ACT_SETS = [(ga.ACT_NONE, 0.0, 0.0), (ga.ACT_CLIP, float(SPECIAL[0]), 100.5),
            (ga.ACT_CLIP, -3.5, float(SPECIAL[3])), (ga.ACT_CLIP, float(SPECIAL[4]), float(SPECIAL[3])),
            (ga.ACT_LRELU, float(SPECIAL[0]), 2.0), (ga.ACT_LRELU, float(SPECIAL[3]), float(SPECIAL[4])),
            (ga.ACT_LRELU, float(SPECIAL[6]), float(SPECIAL[0])), (ga.ACT_LRELU, 0.1015625, float(SPECIAL[3]))]


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
        want = gm.conv_qdq(ifm, wts, ot, 1, 1, c0, c1, c2, split16, kind, a0, a1, slope, k_opt=True)
        got = run(name, ifm, T[it][0], wts, wt, OUT, T[ot][0], I64(h), I64(w), I64(ci), I64(kh),
                  I64(kw), I64(co), 1, 1, 1, c0, c1, c2, kind, a0, a1, slope, n=want.size, t=ot)
        assert_same(got, want)

    kind = gg.I8 if not split16 else gg.I16_JOIN
    for which in COEF_SETS:
        c0, c1, c2, _ = special_cols(co, which, kind, rng)
        for k_, a0, a1 in ACT_SETS[:1] + ACT_SETS[4:5]:
            check(c0, c1, c2, k_, a0, a1)
        check(c0, None, c2, ga.ACT_NONE, 0.0, 0.0)
    c0, c1, c2, _ = special_cols(co, "c0", kind)
    for k_, a0, a1 in ACT_SETS:
        check(c0, c1, c2, k_, a0, a1)
    # PRELU: the slope per channel, the pivot act0
    slope = SPECIAL[np.arange(co) % NS].copy()
    for a0 in (SPECIAL[:NS].tolist() + [100.0]):
        check(c0, c1, c2, ga.ACT_PRELU, float(a0), 0.0, slope)


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
        want = gm.conv_qdq_add(ifm, wts, ifm2, ot, 1, 1, c0, c1, c2, split16, kind, a0, a1, before,
                               *(f32(x) for x in add), k_opt=True)
        if split16:
            got = run("npu_qdq_conv_add_a16w8", ifm, T[it][0], wts, NPU_U8, ifm2, NPU_U16, OUT,
                      T[ot][0], I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), 1, 1, c0, c1, c2,
                      kind, a0, a1, before, *add, n=want.size, t=ot)
        else:
            got = run("npu_qdq_conv_add_a8w8", ifm, T[it][0], wts, NPU_I8, ifm2, NPU_I8, OUT,
                      T[ot][0], I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), 1, 1, 1, c0, c1,
                      c2, kind, a0, a1, before, *add, n=want.size, t=ot)
        assert_same(got, want)

    kind = gg.I16_JOIN if split16 else gg.I8
    base_add = (1.0, 0.375, -12.5)
    c0, c1, c2, _ = special_cols(co, "c0", kind)
    for add in add_sets(base_add):
        check(c0, c1, c2, ga.ACT_NONE, 0.0, 0.0, add)
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
        want = gm.conv_qdq_q_add(ifm, wts, ifm2, ot, 1, 1, c0, c2, *(f32(x) for x in add))
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
    fn = gm.qdq_dwc16 if split16 else gm.qdq_dwc
    kind = gg.I16_JOIN if split16 else gg.I8

    def check(c0, c2, k_, a0, a1, entry="npu_qdq_dwc_int16x8" if split16 else "npu_qdq_dwc_int8x8"):
        want = fn(ifm, wts, ot, 1, 1, zp_w, actv_sum, c0, c2, k_, a0, a1)
        got = run(entry, ifm, T[it][0], wts, T[wt][0], OUT, T[ot][0], I64(h), I64(w), I64(c),
                  I64(kh), I64(kw), 1, 1, zp_w, actv_sum, c0, c2, k_, a0, a1, n=want.size, t=ot)
        assert_same(got, want)

    for which in COEF_SETS:
        c0, _, c2, _ = special_cols(c, which, kind, rng)
        check(c0, c2, ga.ACT_NONE, 0.0, 0.0)
        check(c0, c2, ga.ACT_LRELU, 0.1015625, -3.59375)
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
            want = (gm.qdq_dwc16 if a16 else gm.qdq_dwc)(ifm, wts, it, s, s, zp_w, 1, c0, c2, k_, a0, a1)
            got = run(name, ifm, T[it][0], wts, T[wt][0], OUT, T[it][0], I64(h), I64(w), I64(c), zp_w,
                      *sums, c0, c2, k_, a0, a1, n=want.size, t=it)
            assert_same(got, want)
