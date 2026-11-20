# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""select.h entry points against the numpy reference, bit for bit."""

import numpy as np
import pytest

import golden_float_cvt as gf
import golden_select as gs
import npu
from npu import I64, OUT, OUT2, NPU_F32, NPU_I64, NPU_U8, NPU_U16, NPU_U32
from testing import ALL8, ALL16, HALF, shp
from testing import INT_ALL as INT
from testing import call_n as go
from testing import same as assert_same

RNG = np.random.default_rng(2027)
IDX = {"u8": (NPU_U8, np.uint8), "u16": (NPU_U16, np.uint16), "u32": (NPU_U32, np.uint32),
       "i64": (NPU_I64, np.int64), "f32": (NPU_F32, np.float32)}


def call2(name, *args, outs):
    """Two outputs (OUT, OUT2); outs = ((count, dtype), (count, dtype))."""
    res = npu.callx(name, *args, outs=[int(c) * np.dtype(d).itemsize for c, d in outs])
    return [r.view(d) for r, (_, d) in zip(res, outs)]


def half_special(t):
    """Signed zeros, start values, extremes, NaN, ties."""
    f16 = t == gs.F16
    b = [0x0000, 0x8000, 0x0001, 0x8001, 0x3C00 if f16 else 0x3F80, 0xBC00 if f16 else 0xBF80,
         0x7C00 if f16 else 0x7F80, 0xFC00 if f16 else 0xFF80, 0x7E00 if f16 else 0x7FC0,
         0xFE55 if f16 else 0xFFC5, 0x7BFF if f16 else 0x7F7F, 0xFBFF if f16 else 0xFF7F,
         0x0400 if f16 else 0x0080, 0x8400 if f16 else 0x8080, 0x03FF if f16 else 0x007F]
    return np.array(b, np.uint16)


def half_values(t, n, nan=True, few=False):
    """n bit patterns: small set of values (many ties) or wide random, plus specials."""
    if few:
        v = RNG.integers(-3, 4, n).astype(np.float32) * np.float32(0.5)
        bits = gf.store_float(v, t)
        bits = np.where((v == 0) & (RNG.random(n) < 0.5), bits | np.uint16(0x8000), bits)
    else:
        v = (RNG.standard_normal(n) * 8).astype(np.float32)
        bits = gf.store_float(v, t)
    sp = half_special(t)
    if not nan:
        sp = sp[~np.isnan(gf.load_float(sp, t))]
    k = min(n // 4, 4 * len(sp))
    pos = RNG.choice(n, k, replace=False)
    bits[pos] = sp[RNG.integers(0, len(sp), k)]
    return bits.astype(np.uint16)


def fp8_values(n, nan_rate=0.01, few=False):
    if few:
        c = RNG.choice(np.array([0x00, 0x80, 0x38, 0xB8, 0x40, 0xC0, 0x7E, 0xFE], np.uint8), n)
    else:
        c = RNG.integers(0, 256, n).astype(np.uint8)
        c = np.where((c & 0x7F) == 0x7F, np.uint8(0x30), c)
    nan = RNG.random(n) < nan_rate
    return np.where(nan, RNG.choice(np.array([0x7F, 0xFF], np.uint8), n), c).astype(np.uint8)


def int_values(t, n, few=False):
    info = np.iinfo(INT[t][1])
    if few:
        v = RNG.integers(-2, 3, n) + (0 if info.min < 0 else 2)
        v = np.where(RNG.random(n) < 0.05, info.max, np.where(RNG.random(n) < 0.05, info.min, v))
        return v.astype(INT[t][1])
    v = RNG.integers(info.min, info.max, n, dtype=INT[t][1], endpoint=True)
    edge = np.array([info.min, info.max, 0], dtype=INT[t][1])
    v[RNG.choice(n, min(n, 6), replace=False)] = edge[RNG.integers(0, 3, min(n, 6))]
    return v


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("op", ["abs", "neg"])
def test_abs_neg_half(op, t):
    nt, ht = HALF[t]
    got = go(f"npu_elementwise_{op}_bf16", ALL16, nt, OUT, I64(65536), dt=np.uint16, n=65536)
    assert_same(got, gs.abs_half(ALL16) if op == "abs" else gs.neg_half(ALL16, ht), op)


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("lo,hi", [(-0.3125, 0.625), (0.0, 1.0), (-1.0, -0.0), (-0.0, 0.0),
                                   (-1.7, 2.3), (0.1, 0.1), (-1e30, 1e30)])
def test_clip_half(t, lo, hi):
    nt, ht = HALF[t]
    got = go("npu_elementwise_clip_bf16", ALL16, nt, OUT, I64(65536), lo, hi, dt=np.uint16,
             n=65536)
    assert_same(got, gs.clip_half(ALL16, ht, lo, hi), "clip")


def fp8_all_rows():
    """Every code in every lane position, with rows that hold a NaN and rows that do not."""
    base = np.concatenate([RNG.permutation(ALL8) for _ in range(8)])
    clean = np.where((base & 0x7F) == 0x7F, np.uint8(0x41), base)
    return np.concatenate([base, clean, fp8_values(4093)])


@pytest.mark.parametrize("op,scale", [("abs", 1.0), ("abs", 0.37), ("abs", 300.0),
                                      ("neg", -1.0), ("neg", -0.5), ("neg", -0.0137),
                                      ("neg", -97.0)])
def test_fp8_abs_neg(op, scale):
    x = fp8_all_rows()
    got = go(f"npu_elementwise_{op}_fp8", x, OUT, I64(len(x)), scale, dt=np.uint8, n=len(x))
    assert_same(got, gs.fp8_unary(x, op, scale), op)


@pytest.mark.parametrize("scale,lo,hi", [(2.0, -6.0, 8.0), (0.5, -1.0, 2.0), (1.0, 0.0, 448.0),
                                         (0.3, -0.0, 0.0), (-1.5, -3.0, 3.0), (100.0, -500.0, 500.0)])
def test_fp8_clip(scale, lo, hi):
    x = fp8_all_rows()
    got = go("npu_elementwise_clip_fp8", x, OUT, I64(len(x)), scale, lo, hi, dt=np.uint8,
             n=len(x))
    assert_same(got, gs.fp8_unary(x, "clip", scale, lo, hi), "clip")


BC = [(0, 0), (0, 1), (0, 2), (0, 4), (0, 3), (0, 7), (1, 0), (6, 0), (5, 2), (7, 7)]
OPS = ["equal", "greater", "greater_or_equal", "max", "min"]


def bshape(bc, H, W, C):
    return (1 if bc & 4 else H, 1 if bc & 2 else W, 1 if bc & 1 else C)


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("op", OPS)
def test_bcast_half(op, t):
    nt, ht = HALF[t]
    H, W, C = 3, 5, 70
    for abc, bbc in BC:
        sa, sb = bshape(abc, H, W, C), bshape(bbc, H, W, C)
        few = abc == bbc
        a = half_values(ht, int(np.prod(sa)), few=few).reshape(sa)
        b = half_values(ht, int(np.prod(sb)), few=few).reshape(sb)
        if abc == bbc == 0:   # every pair of specials
            sp = half_special(ht)
            pa, pb = np.meshgrid(sp, sp)
            a.reshape(-1)[:pa.size] = pa.reshape(-1)
            b.reshape(-1)[:pb.size] = pb.reshape(-1)
        out_dt = np.uint16 if op in ("max", "min") else np.int8
        f = np.maximum(sa, sb)  # the full shape (both operands size 1 along an axis: 1)
        got = go(f"npu_elementwise_broadcasting3d_{op}_bf16", a, b, nt, OUT, shp(sa), shp(sb),
                 dt=out_dt, n=int(np.prod(f)))
        assert_same(got, gs.bcast_half(op, a, abc, b, bbc, ht, *f), f"{op} bc {abc},{bbc}")


@pytest.mark.parametrize("op", OPS)
def test_bcast_fp8(op):
    H, W, C = 3, 4, 77
    for (abc, bbc), (sa_, sb_) in zip(BC, [(1.0, 1.0), (0.5, 0.25), (0.37, 1.3), (2.0, 0.5),
                                          (1.0, 0.75), (0.1, 0.1), (3.0, 3.0), (0.5, 1.0),
                                          (1.0, 1.0), (256.0, 1.0)]):
        sa, sb = bshape(abc, H, W, C), bshape(bbc, H, W, C)
        few = abc == bbc
        a = fp8_values(int(np.prod(sa)), 0.02, few).reshape(sa)
        b = fp8_values(int(np.prod(sb)), 0.02, few).reshape(sb)
        out_dt = np.uint8 if op in ("max", "min") else np.int8
        f = np.maximum(sa, sb)
        got = go(f"npu_elementwise_broadcasting3d_{op}_fp8", a, sa_, b, sb_, OUT, shp(sa), shp(sb),
                 dt=out_dt, n=int(np.prod(f)))
        assert_same(got, gs.bcast_fp8(op, a, abc, sa_, b, bbc, sb_, *f),
                    f"{op} bc {abc},{bbc}")


# entry: (input kind, input types, output types, is_max, index bits: None = 16 / 'out')
ARGS = {
    "arg_max_c64_a8": ("int", ["i8", "u8"], ["u8", "u16", "u32", "i64", "f32"], True, None),
    "arg_max_c64_a16": ("int", ["i16", "u16"], ["u8", "u16"], True, "out"),
    "arg_max_c64_a16_int64": ("int", ["i16", "u16"], ["i64"], True, None),
    "arg_max_c64_a32": ("int", ["i32", "u32"], ["u8", "u16", "u32", "i64", "f32"], True, None),
    "arg_max_c64_bf16": ("half", ["bf16", "f16"], ["u8", "u16"], True, "out"),
    "arg_max_c64_fp8": ("fp8", ["fp8"], ["u8", "u16", "u32", "i64", "f32"], True, None),
    "arg_min_c64_a8": ("int", ["i8", "u8"], ["u8", "u16"], False, "out"),
    "arg_min_c64_a16": ("int", ["i16", "u16"], ["u8", "u16"], False, "out"),
    "arg_min_c64_bf16": ("half", ["bf16", "f16"], ["u8", "u16"], False, "out"),
}
SHAPES = [(2, 3, 300, True), (300, 2, 64, True), (2, 290, 128, True), (3, 5, 37, False),
          (4, 3, 640, False)]


@pytest.mark.parametrize("w", list(ARGS))
def test_arg(w):
    kind, ins, outs, is_max, bits = ARGS[w]
    for it in ins:
        for ot in outs:
            for H, W, C, few in SHAPES:
                n = H * W * C
                if kind == "int":
                    nt, _ = INT[it]
                    x = int_values(it, n, few).reshape(H, W, C)
                    vals = x.astype(np.float64)
                    info = np.iinfo(x.dtype)
                    init = float(info.min if is_max else info.max)
                elif kind == "half":
                    nt, ht = HALF[it]
                    x = half_values(ht, n, few=few).reshape(H, W, C)
                    if few:   # lanes whose values never beat the start value
                        x[0] = gf.store_float(-np.abs(gf.load_float(x[0], ht)), ht)
                    vals = gf.load_float(x, ht)
                    init = gf.load_float(np.uint16((gs.MIN_NORMAL if is_max else gs.MAX_FINITE)[ht]), ht)
                else:
                    x = fp8_values(n, 0.02, few).reshape(H, W, C)
                    vals = gf.fp8_to_f32(x).astype(np.float64)
                    init = -448.0
                nb = 16 if bits is None else (8 if ot == "u8" else 16)
                for axis in range(3):
                    no = n // (H, W, C)[axis]
                    pre = () if kind == "fp8" else (nt,)
                    post = () if w == "arg_max_c64_a16_int64" else (IDX[ot][0],)
                    got = go(f"npu_{w}", x, *pre, OUT, *post, I64(H), I64(W), I64(C), axis,
                             dt=IDX[ot][1], n=no)
                    want = gs.index_out(gs.arg_reduce(vals, axis, is_max, init, nb), ot)
                    assert_same(got, want, f"{it}->{ot} {H}x{W}x{C} axis {axis}")


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("is_max", [True, False])
def test_reduce(t, is_max):
    nt, ht = HALF[t]
    for (H, W, C), few in [((3, 4, 70), True), ((5, 2, 64), False), ((1, 1, 200), True)]:
        x = half_values(ht, H * W * C, few=few).reshape(H, W, C)
        x[0, 0, :5] = 0xFC00 if ht == gs.F16 else 0xFF80   # -inf: below the start value
        x[-1, -1] = 0x7C00 if ht == gs.F16 else 0x7F80     # a +inf pixel
        for axes in range(1, 8):
            n = (1 if axes & 4 else H) * (1 if axes & 2 else W) * (1 if axes & 1 else C)
            name = "npu_reduce_max_c64_bf16" if is_max else "npu_reduce_min_c64_bf16"
            got = go(name, x, nt, OUT, I64(H), I64(W), I64(C), axes, dt=np.uint16, n=n)
            assert_same(got, gs.reduce_half(x, ht, axes, is_max), f"{H}x{W}x{C} axes {axes}")
    # every value at or beyond the start value
    x = np.full((2, 2, 8), 0xFC00 if ht == gs.F16 else 0xFF80, np.uint16)
    if not is_max:
        x[:] = 0x7C00 if ht == gs.F16 else 0x7F80
    name = "npu_reduce_max_c64_bf16" if is_max else "npu_reduce_min_c64_bf16"
    got = go(name, x, nt, OUT, I64(2), I64(2), I64(8), 7, dt=np.uint16, n=1)
    assert_same(got, [(gs.LOWEST if is_max else gs.MAX_FINITE)[ht]], "start value")


POOL = [(9, 9, 70, 3, 3, 1, 1), (8, 12, 64, 2, 2, 2, 2), (7, 11, 16, 3, 2, 2, 3),
        (13, 13, 8, 13, 13, 1, 1), (6, 6, 5, 1, 2, 1, 2)]


@pytest.mark.parametrize("w,t", [("maxpool_bf16", "bf16"), ("maxpool_bf16", "f16"),
                                 ("maxpool_bf16_c32", "bf16")])
def test_maxpool_half(w, t):
    nt, ht = HALF[t]
    for Hi, Wi, C, kh, kw, sh, sw in POOL:
        for few in (False, True):
            x = half_values(ht, Hi * Wi * C, few=few).reshape(Hi, Wi, C)
            Ho, Wo = (Hi - kh) // sh + 1, (Wi - kw) // sw + 1
            pre = (nt,) if w == "maxpool_bf16" else ()
            got = go(f"npu_{w}", x, *pre, OUT, I64(Hi), I64(Wi), I64(C), kh, kw, sh, sw,
                     dt=np.uint16, n=Ho * Wo * C)
            assert_same(got, gs.maxpool_half(x, ht, kh, kw, sh, sw), f"{Hi}x{Wi}x{C} k{kh}x{kw}")


@pytest.mark.parametrize("cw", [False, True])
@pytest.mark.parametrize("clip", [0, 1])
def test_maxpool_fp8(cw, clip):
    for Hi, Wi, C, kh, kw, sh, sw in POOL:
        for few in (False, True):
            x = fp8_values(Hi * Wi * C, 0.002, few).reshape(Hi, Wi, C)
            Ho, Wo = (Hi - kh) // sh + 1, (Wi - kw) // sw + 1
            nc = C if cw else 1
            c2 = (RNG.uniform(0.01, 2.0, nc) * RNG.choice([-1, 1], nc)).astype(np.float32)
            c0 = RNG.uniform(-8, 8, nc).astype(np.float32)
            c0[::3] = 0.0
            c0[1::5] = -c2[1::5] * np.float32(1.5)    # zero results from cancellation
            lo, hi = (0.0, 6.0) if clip else (0.0, 0.0)
            got = go("npu_maxpool_fp8", x, OUT, I64(Hi), I64(Wi), I64(C), kh, kw, sh, sw, c0, c2,
                     I64(nc), clip, lo, hi, dt=np.uint8, n=Ho * Wo * C)
            want = gs.maxpool_fp8(x, kh, kw, sh, sw, c0 if cw else c0[0], c2 if cw else c2[0],
                                  clip, lo, hi)
            assert_same(got, want, f"{Hi}x{Wi}x{C} k{kh}x{kw}")


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_globalmaxpool_half(t):
    nt, ht = HALF[t]
    for H, W, C in [(4, 4, 70), (1, 3, 64), (7, 5, 9)]:
        x = half_values(ht, H * W * C, few=True).reshape(H, W, C)
        x[:, :, 0] = gf.store_float(-np.abs(gf.load_float(x[:, :, 0], ht)), ht)  # all <= 0
        x[:, :, 1] = 0x8000                                                      # all -0
        got = go("npu_globalmaxpool_bf16", x, nt, OUT, I64(H), I64(W), I64(C), dt=np.uint16, n=C)
        assert_same(got, gs.globalmaxpool_half(x, ht), f"{H}x{W}x{C}")


def test_globalmaxpool_fp8():
    for H, W, C in [(4, 4, 70), (8, 8, 64), (1, 2, 13)]:
        for c0, c1 in [(0.0, 1.0), (-7.07, 0.0148), (6.93, -0.0148), (1.0, 300.0)]:
            x = fp8_values(H * W * C, 0.003, few=C < 20).reshape(H, W, C)
            got = go("npu_globalmaxpool_fp8", x, OUT, I64(H), I64(W), I64(C), c0, c1,
                     dt=np.uint8, n=C)
            assert_same(got, gs.globalmaxpool_fp8(x, c0, c1), f"{H}x{W}x{C}")


TOPK = {"topk2d_bf16": (["bf16", "f16"], ["u8", "u16"]), "topk2d_a8": (["i8", "u8"], ["u8", "u16"]),
        "topk2d_a16": (["i16", "u16"], ["u16"])}


@pytest.mark.parametrize("w", list(TOPK))
@pytest.mark.parametrize("largest", [1, 0])
def test_topk(w, largest):
    ins, idxs = TOPK[w]
    for it in ins:
        for ix in idxs:
            for rows, W, K, few in [(3, 100, 7, False), (2, 64, 70, True), (2, 300, 12, True),
                                    (1, 7, 9, False)]:
                if it in HALF:
                    nt, ht = HALF[it]
                    x = half_values(ht, rows * W, nan=False, few=few).reshape(rows, W)
                    vals = gf.load_float(x, ht)
                    sent = -np.inf if largest else np.inf
                    f16 = ht == gs.F16
                    sraw = (0xFC00 if f16 else 0xFF80) if largest else (0x7C00 if f16 else 0x7F80)
                    dt = np.uint16
                else:
                    nt, dt = INT[it]
                    x = int_values(it, rows * W, few).reshape(rows, W)
                    vals = x.astype(np.float64)
                    info = np.iinfo(dt)
                    sent = float(info.min if largest else info.max)
                    sraw = int(sent)
                nti, idt = IDX[ix]
                gv, gi = call2(f"npu_{w}", x, nt, OUT, OUT2, nti, I64(rows), I64(W), I64(K),
                               largest, outs=((rows * K, dt), (rows * K, idt)))
                rv, ri = gs.topk(vals, x, W, K, largest, sent, np.array(sraw).astype(dt),
                                 8 if ix == "u8" else 16)
                what = f"{it} {ix} {rows}x{W} K{K}"
                assert_same(gv, rv, what + " values")
                assert_same(gi, ri.astype(idt), what + " indices")
