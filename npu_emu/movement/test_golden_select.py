# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""select.h entry points against the numpy reference, bit for bit."""

import numpy as np
import pytest

import golden_float_cvt as gf
import golden_select as gs
import npu
from npu import (I64, OUT, OUT2, NPU_BF16, NPU_BFP16, NPU_F16, NPU_F32, NPU_I8, NPU_I16, NPU_I32,
                 NPU_I64, NPU_U8, NPU_U16, NPU_U32, Out, raw_rc)
from testing import ALL8, ALL16, HALF, shp
from testing import INT_ALL as INT
from testing import call_n as go
from testing import rejected
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
    want = gs.elementwise_abs_bf16(ALL16) if op == "abs" else gs.elementwise_neg_bf16(ALL16, ht)
    assert_same(got, want, op)


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("lo,hi", [(-0.3125, 0.625), (0.0, 1.0), (-1.0, -0.0), (-0.0, 0.0),
                                   (-1.7, 2.3), (0.1, 0.1), (-1e30, 1e30)])
def test_clip_half(t, lo, hi):
    nt, ht = HALF[t]
    got = go("npu_elementwise_clip_bf16", ALL16, nt, OUT, I64(65536), lo, hi, dt=np.uint16,
             n=65536)
    assert_same(got, gs.elementwise_clip_bf16(ALL16, ht, lo, hi), "clip")


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
    assert_same(got, getattr(gs, f"elementwise_{op}_fp8")(x, scale), op)


@pytest.mark.parametrize("scale,lo,hi", [(2.0, -6.0, 8.0), (0.5, -1.0, 2.0), (1.0, 0.0, 448.0),
                                         (0.3, -0.0, 0.0), (-1.5, -3.0, 3.0), (100.0, -500.0, 500.0)])
def test_fp8_clip(scale, lo, hi):
    x = fp8_all_rows()
    got = go("npu_elementwise_clip_fp8", x, OUT, I64(len(x)), scale, lo, hi, dt=np.uint8,
             n=len(x))
    assert_same(got, gs.elementwise_clip_fp8(x, scale, lo, hi), "clip")


BC = [(0, 0), (0, 1), (0, 2), (0, 4), (0, 3), (0, 7), (1, 0), (6, 0), (5, 0), (7, 7)]
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
        name = f"elementwise_broadcasting3d_{op}_bf16"
        got = go(f"npu_{name}", a, nt, b, OUT, shp(sa), shp(sb), dt=out_dt, n=int(np.prod(f)))
        assert_same(got, getattr(gs, name)(a, b, ht), f"{op} bc {abc},{bbc}")


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
        name = f"elementwise_broadcasting3d_{op}_fp8"
        got = go(f"npu_{name}", a, b, OUT, shp(sa), shp(sb), sa_, sb_, dt=out_dt,
                 n=int(np.prod(f)))
        assert_same(got, getattr(gs, name)(a, b, sa_, sb_), f"{op} bc {abc},{bbc}")


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
                for flags in (4, 2, 1):
                    axis = gs.AXIS_OF[flags]
                    no = n // (H, W, C)[axis]
                    pre = () if kind == "fp8" else (nt,)
                    post = () if w == "arg_max_c64_a16_int64" else (IDX[ot][0],)
                    got = go(f"npu_{w}", x, *pre, OUT, *post, I64(H), I64(W), I64(C), flags,
                             dt=IDX[ot][1], n=no)
                    want = gs._index_out(gs._arg_reduce(vals, axis, is_max, init, nb), ot)
                    if kind == "half":
                        model = getattr(gs, w)(x, ht, flags, ot)
                    elif w == "arg_max_c64_a16_int64":
                        model = getattr(gs, w)(x, flags)
                    else:
                        model = getattr(gs, w)(x, flags, ot)
                    assert_same(model, want, "model")
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
            model = gs.reduce_max_c64_bf16 if is_max else gs.reduce_min_c64_bf16
            assert_same(got, model(x, ht, axes), f"{H}x{W}x{C} axes {axes}")
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
            want = (gs.maxpool_bf16(x, ht, kh, kw, sh, sw) if w == "maxpool_bf16" else
                    gs.maxpool_bf16_c32(x, kh, kw, sh, sw))
            assert_same(got, want, f"{Hi}x{Wi}x{C} k{kh}x{kw}")


@pytest.mark.parametrize("cw", [False, True])
@pytest.mark.parametrize("act", [0, 100])
def test_maxpool_fp8(cw, act):
    for Hi, Wi, C, kh, kw, sh, sw in POOL:
        for few in (False, True):
            x = fp8_values(Hi * Wi * C, 0.002, few).reshape(Hi, Wi, C)
            Ho, Wo = (Hi - kh) // sh + 1, (Wi - kw) // sw + 1
            nc = C if cw else 1
            c2 = (RNG.uniform(0.01, 2.0, nc) * RNG.choice([-1, 1], nc)).astype(np.float32)
            c0 = RNG.uniform(-8, 8, nc).astype(np.float32)
            c0[::3] = 0.0
            c0[1::5] = -c2[1::5] * np.float32(1.5)    # zero results from cancellation
            lo, hi = (0.0, 6.0) if act else (0.0, 0.0)
            got = go("npu_maxpool_fp8", x, OUT, I64(Hi), I64(Wi), I64(C), kh, kw, sh, sw, c0, c2,
                     I64(nc), act, lo, hi, dt=np.uint8, n=Ho * Wo * C)
            want = gs.maxpool_fp8(x, kh, kw, sh, sw, c0 if cw else c0[0], c2 if cw else c2[0],
                                  act, lo, hi)
            assert_same(got, want, f"{Hi}x{Wi}x{C} k{kh}x{kw}")


def pixel_blocks(npix):
    """npix pixels of 8 bfp16 blocks: per-pixel exponent spreads of 0..30 (decode alignment),
    all-zero blocks, extreme mantissas and exponents."""
    nb = npix * 8
    e = np.repeat(RNG.integers(110, 140, npix), 8) + (
        RNG.random(nb) * np.repeat(RNG.integers(0, 31, npix), 8)).astype(np.int64)
    m = RNG.integers(-128, 128, (nb, 8))
    k = max(1, nb // 16)
    m[RNG.integers(0, nb, k)] = 0
    m[RNG.integers(0, nb, k), RNG.integers(0, 8, k)] = RNG.choice([-128, 127, -1, 1], k)
    e[RNG.integers(0, nb, k)] = RNG.choice([0, 1, 2, 253, 254, 255], k)
    return np.concatenate([np.clip(e, 0, 255)[:, None].astype(np.uint8),
                           m.astype(np.int8).view(np.uint8)], axis=1).reshape(-1)


@pytest.mark.parametrize("u8", [False, True])
def test_maxpool_bfp16(u8):
    for Hi, Wi, C in [(2, 2, 64), (6, 8, 64), (4, 12, 192), (8, 4, 128)]:
        blk = pixel_blocks(Hi * Wi * C // 64)
        n = Hi // 2 * (Wi // 2) * C
        for qm, zp in ([(26.954178, 117.0), (4.0, 128.0), (0.3, 3.0)] if u8 else [(1.0, 0.0)]):
            got = go("npu_maxpool_bfp16", blk, OUT, NPU_U8 if u8 else NPU_BFP16, I64(Hi),
                     I64(Wi), I64(C), qm, zp, dt=np.uint8, n=n if u8 else n // 8 * 9)
            want = gs.maxpool_bfp16(blk, Hi, Wi, C, u8, qm, zp)
            assert_same(got, want, f"{Hi}x{Wi}x{C} u8={u8}")


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_globalmaxpool_half(t):
    nt, ht = HALF[t]
    for H, W, C in [(4, 4, 70), (1, 3, 64), (7, 5, 9)]:
        x = half_values(ht, H * W * C, few=True).reshape(H, W, C)
        x[:, :, 0] = gf.store_float(-np.abs(gf.load_float(x[:, :, 0], ht)), ht)  # all <= 0
        x[:, :, 1] = 0x8000                                                      # all -0
        got = go("npu_globalmaxpool_bf16", x, nt, OUT, I64(H), I64(W), I64(C), dt=np.uint16, n=C)
        assert_same(got, gs.globalmaxpool_bf16(x, ht), f"{H}x{W}x{C}")


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
            for rows, W, K, few in [(3, 100, 7, False), (2, 64, 64, True), (2, 300, 12, True),
                                    (1, 7, 7, False), (2, 70, 70, True)]:
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
                ia = () if w == "topk2d_a16" else (nti,)
                gv, gi = call2(f"npu_{w}", x, nt, OUT, OUT2, *ia, I64(rows), I64(W), I64(K),
                               largest, outs=((rows * K, dt), (rows * K, idt)))
                rv, ri = gs._topk(vals, x, W, K, largest, sent, np.array(sraw).astype(dt),
                                  8 if ix == "u8" else 16)
                if w == "topk2d_bf16":
                    mv, mi = gs.topk2d_bf16(x, ht, K, largest, 8 if ix == "u8" else 16)
                elif w == "topk2d_a8":
                    mv, mi = gs.topk2d_a8(x, K, largest, 8 if ix == "u8" else 16)
                else:
                    mv, mi = gs.topk2d_a16(x, K, largest)
                assert_same(mv, rv, "model values")
                assert_same(mi.astype(idt), ri.astype(idt), "model indices")
                what = f"{it} {ix} {rows}x{W} K{K}"
                assert_same(gv, rv, what + " values")
                assert_same(gi, ri.astype(idt), what + " indices")


def test_bad_args():
    """Every entry rejects (-1) an argument set its kernel cannot run."""
    _empty_work_cases()
    n = None
    d = np.zeros(512, np.uint8)
    nb = 512
    out = Out()
    s3 = shp((2, 2, 2))
    for op in ("abs", "neg"):
        assert rejected(f"npu_elementwise_{op}_bf16", d, NPU_I16, out, I64(4), nbytes=nb)
        assert raw_rc(f"npu_elementwise_{op}_bf16", n, NPU_BF16, d, I64(4)) == -1
        assert raw_rc(f"npu_elementwise_{op}_fp8", d, n, I64(4), 1.0) == -1
        assert rejected(f"npu_elementwise_{op}_fp8", d, out, I64(-1), 1.0, nbytes=nb)
    assert rejected("npu_elementwise_clip_bf16", d, NPU_BF16, out, I64(4), 1.0, -1.0, nbytes=nb)
    assert rejected("npu_elementwise_clip_bf16", d, NPU_BF16, out, I64(4), float("nan"), 1.0,
                    nbytes=nb)
    assert rejected("npu_elementwise_clip_bf16", d, NPU_I8, out, I64(4), 0.0, 1.0, nbytes=nb)
    assert rejected("npu_elementwise_clip_fp8", d, out, I64(4), 1.0, 2.0, 1.0, nbytes=nb)
    assert raw_rc("npu_elementwise_clip_fp8", n, d, I64(4), 1.0, 0.0, 1.0) == -1
    for op in OPS:
        e = f"npu_elementwise_broadcasting3d_{op}_bf16"
        assert rejected(e, d, NPU_U16, d, out, s3, s3, nbytes=nb)
        # each operand broadcast along a different axis: neither has the full shape
        assert rejected(e, d, NPU_BF16, d, out, shp((2, 1, 2)), shp((1, 2, 2)), nbytes=nb)
        assert raw_rc(e, d, NPU_BF16, d, d, n, s3) == -1
        e = f"npu_elementwise_broadcasting3d_{op}_fp8"
        assert rejected(e, d, d, out, shp((2, 1, 2)), shp((1, 2, 2)), 1.0, 1.0, nbytes=nb)
        assert rejected(e, d, d, out, shp((2, 3, 2)), s3, 1.0, 1.0, nbytes=nb)
        assert raw_rc(e, d, n, d, s3, s3, 1.0, 1.0) == -1
    for w in ARGS:
        kind = ARGS[w][0]
        nt = INT[ARGS[w][1][0]][0] if kind == "int" else NPU_BF16
        pre = () if kind == "fp8" else (nt,)
        post = () if w == "arg_max_c64_a16_int64" else (NPU_U16,)
        for flags in (0, 3, 7):   # one axis only, as a bit flag
            assert rejected(f"npu_{w}", d, *pre, out, *post, I64(2), I64(2), I64(2), flags,
                            nbytes=nb)
        assert raw_rc(f"npu_{w}", d, *pre, n, *post, I64(2), I64(2), I64(2), 1) == -1
        if kind != "fp8":
            assert rejected(f"npu_{w}", d, NPU_I32 if kind != "int" or "a32" not in w else NPU_I8,
                            out, *post, I64(2), I64(2), I64(2), 1, nbytes=nb)
    assert rejected("npu_arg_max_c64_a16", d, NPU_I16, out, NPU_U32, I64(2), I64(2), I64(2), 1,
                    nbytes=nb)
    for op in ("max", "min"):
        e = f"npu_reduce_{op}_c64_bf16"
        assert rejected(e, d, NPU_BF16, out, I64(2), I64(2), I64(2), 8, nbytes=nb)
        assert rejected(e, d, NPU_I16, out, I64(2), I64(2), I64(2), 1, nbytes=nb)
        assert raw_rc(e, n, NPU_BF16, d, I64(2), I64(2), I64(2), 1) == -1
    assert rejected("npu_maxpool_bf16", d, NPU_BF16, out, I64(16), I64(16), I64(1), 16, 2, 1, 1,
                    nbytes=nb)                                             # kh > 15
    assert rejected("npu_maxpool_bf16", d, NPU_U16, out, I64(4), I64(4), I64(1), 2, 2, 1, 1,
                    nbytes=nb)
    assert rejected("npu_maxpool_bf16_c32", d, out, I64(4), I64(4), I64(1), 2, 2, 1, 16, nbytes=nb)
    c = np.ones(1, np.float32)
    e = "npu_maxpool_fp8"
    assert rejected(e, d, out, I64(4), I64(4), I64(1), 2, 2, 1, 1, c, c, I64(1), 1, 0.0, 6.0,
                    nbytes=nb)                                             # activation 1
    assert rejected(e, d, out, I64(40), I64(40), I64(1), 33, 2, 1, 1, c, c, I64(1), 0, 0.0, 0.0,
                    nbytes=nb)
    assert rejected(e, d, out, I64(4), I64(4), I64(3), 2, 2, 1, 1, c, c, I64(2), 0, 0.0, 0.0,
                    nbytes=nb)                                             # channels
    assert raw_rc(e, d, d, I64(4), I64(4), I64(1), 2, 2, 1, 1, n, c, I64(1), 0, 0.0, 0.0) == -1
    assert rejected("npu_maxpool_bfp16", d, out, NPU_I8, I64(2), I64(2), I64(64), 1.0, 0.0,
                    nbytes=nb)
    assert rejected("npu_maxpool_bfp16", d, out, NPU_U8, I64(3), I64(2), I64(64), 1.0, 0.0,
                    nbytes=nb)
    assert rejected("npu_globalmaxpool_bf16", d, NPU_BF16, out, I64(0), I64(2), I64(2), nbytes=nb)
    assert rejected("npu_globalmaxpool_bf16", d, NPU_I8, out, I64(1), I64(2), I64(2), nbytes=nb)
    assert raw_rc("npu_globalmaxpool_fp8", d, n, I64(1), I64(2), I64(2), 1.0, 0.0) == -1
    assert rejected("npu_globalmaxpool_fp8", d, out, I64(1), I64(-2), I64(2), 1.0, 0.0, nbytes=nb)
    for w, t, ix in (("topk2d_bf16", NPU_BF16, (NPU_U8,)), ("topk2d_a8", NPU_I8, (NPU_U16,)),
                     ("topk2d_a16", NPU_I16, ())):
        e = f"npu_{w}"
        assert raw_rc(e, d, t, d, d, *ix, I64(2), I64(8), I64(9), 1) == -1   # K > W
        assert raw_rc(e, d, t, d, d, *ix, I64(2), I64(8), I64(4), 2) == -1   # largest
        assert raw_rc(e, d, NPU_I32, d, d, *ix, I64(2), I64(8), I64(4), 1) == -1
        assert raw_rc(e, d, t, n, d, *ix, I64(2), I64(8), I64(4), 1) == -1
    assert raw_rc("npu_topk2d_a8", d, NPU_I8, d, d, NPU_U32, I64(2), I64(8), I64(4), 1) == -1


def _empty_work_cases():
    """A call with 0 output elements returns 0 after validating every other argument; a null
    required pointer is rejected even then, and a negative size always."""
    n, d, z = None, np.zeros(64, np.uint8), I64(0)
    s0, e1 = shp((2, 0, 2)), shp((1, 1, 1))
    for op in ("abs", "neg"):
        assert raw_rc(f"npu_elementwise_{op}_bf16", d, NPU_BF16, d, z) == 0
        assert raw_rc(f"npu_elementwise_{op}_bf16", d, NPU_BF16, n, z) == -1
        assert raw_rc(f"npu_elementwise_{op}_bf16", d, NPU_BF16, d, I64(-1)) == -1
        assert raw_rc(f"npu_elementwise_{op}_fp8", d, d, z, 1.0) == 0
        assert raw_rc(f"npu_elementwise_{op}_fp8", n, d, z, 1.0) == -1
    assert raw_rc("npu_elementwise_clip_bf16", d, NPU_BF16, d, z, 0.0, 1.0) == 0
    assert raw_rc("npu_elementwise_clip_bf16", n, NPU_BF16, d, z, 0.0, 1.0) == -1
    assert raw_rc("npu_elementwise_clip_fp8", d, d, z, 1.0, 0.0, 1.0) == 0
    assert raw_rc("npu_elementwise_clip_fp8", d, n, z, 1.0, 0.0, 1.0) == -1
    assert raw_rc("npu_elementwise_clip_fp8", d, d, I64(-1), 1.0, 0.0, 1.0) == -1
    for op in OPS:
        e = f"npu_elementwise_broadcasting3d_{op}_bf16"
        assert raw_rc(e, d, NPU_BF16, d, d, s0, e1) == 0
        assert raw_rc(e, d, NPU_BF16, d, d, e1, s0) == 0
        assert raw_rc(e, d, NPU_BF16, n, d, s0, e1) == -1
        assert raw_rc(e, d, NPU_BF16, d, d, s0, shp((2, 2, 2))) == -1   # 2 against 0
        assert raw_rc(e, d, NPU_BF16, d, d, shp((2, -1, 2)), e1) == -1
        e = f"npu_elementwise_broadcasting3d_{op}_fp8"
        assert raw_rc(e, d, d, d, s0, s0, 1.0, 1.0) == 0
        assert raw_rc(e, d, d, n, s0, s0, 1.0, 1.0) == -1
    for w in ARGS:
        kind = ARGS[w][0]
        nt = INT[ARGS[w][1][0]][0] if kind == "int" else NPU_BF16
        pre = () if kind == "fp8" else (nt,)
        post = () if w == "arg_max_c64_a16_int64" else (NPU_U16,) if kind != "fp8" else (NPU_U8,)
        e = f"npu_{w}"
        assert raw_rc(e, d, *pre, d, *post, z, I64(2), I64(2), 1) == 0      # H kept
        assert raw_rc(e, d, *pre, d, *post, I64(2), I64(2), z, 1) == -1     # C reduced
        assert raw_rc(e, n, *pre, d, *post, z, I64(2), I64(2), 1) == -1
        assert raw_rc(e, d, *pre, d, *post, I64(-1), I64(2), I64(2), 1) == -1
    for op in ("max", "min"):
        e = f"npu_reduce_{op}_c64_bf16"
        assert raw_rc(e, d, NPU_F16, d, I64(2), z, I64(2), 5) == 0          # W kept
        assert raw_rc(e, d, NPU_F16, d, I64(2), z, I64(2), 2) == -1         # W reduced
        assert raw_rc(e, d, NPU_F16, n, I64(2), z, I64(2), 5) == -1
        assert raw_rc(e, d, NPU_F16, d, I64(2), I64(-1), I64(2), 5) == -1
    assert raw_rc("npu_maxpool_bf16", d, NPU_BF16, d, I64(4), I64(4), z, 2, 2, 1, 1) == 0
    assert raw_rc("npu_maxpool_bf16", d, NPU_BF16, n, I64(4), I64(4), z, 2, 2, 1, 1) == -1
    assert raw_rc("npu_maxpool_bf16", d, NPU_BF16, d, I64(4), I64(4), I64(-1), 2, 2, 1, 1) == -1
    assert raw_rc("npu_maxpool_bf16_c32", d, d, I64(4), I64(4), z, 2, 2, 1, 1) == 0
    assert raw_rc("npu_maxpool_bf16_c32", n, d, I64(4), I64(4), z, 2, 2, 1, 1) == -1
    c = np.ones(1, np.float32)
    e = "npu_maxpool_fp8"
    assert raw_rc(e, d, d, I64(4), I64(4), z, 2, 2, 1, 1, c, c, I64(1), 0, 0.0, 0.0) == 0
    assert raw_rc(e, d, d, I64(4), I64(4), z, 2, 2, 1, 1, c, c, z, 0, 0.0, 0.0) == -1  # channels
    assert raw_rc(e, d, d, I64(4), I64(4), z, 2, 2, 1, 1, c, n, I64(1), 0, 0.0, 0.0) == -1
    e = "npu_maxpool_bfp16"
    assert raw_rc(e, d, d, NPU_U8, I64(2), I64(2), z, 1.0, 0.0) == 0
    assert raw_rc(e, d, n, NPU_U8, I64(2), I64(2), z, 1.0, 0.0) == -1
    assert raw_rc(e, d, d, NPU_U8, I64(2), I64(2), I64(-64), 1.0, 0.0) == -1
    assert raw_rc("npu_globalmaxpool_bf16", d, NPU_BF16, d, I64(2), I64(2), z) == 0
    assert raw_rc("npu_globalmaxpool_bf16", d, NPU_BF16, n, I64(2), I64(2), z) == -1
    assert raw_rc("npu_globalmaxpool_fp8", d, d, I64(2), I64(2), z, 1.0, 0.0) == 0
    assert raw_rc("npu_globalmaxpool_fp8", n, d, I64(2), I64(2), z, 1.0, 0.0) == -1
    for w, t, ix in (("topk2d_bf16", NPU_BF16, (NPU_U8,)), ("topk2d_a8", NPU_I8, (NPU_U16,)),
                     ("topk2d_a16", NPU_I16, ())):
        e = f"npu_{w}"
        assert raw_rc(e, d, t, d, d, *ix, z, I64(8), I64(4), 1) == 0
        assert raw_rc(e, d, t, d, d, *ix, I64(2), I64(8), z, 1) == 0
        assert raw_rc(e, d, t, d, n, *ix, z, I64(8), I64(4), 1) == -1
        assert raw_rc(e, d, t, d, d, *ix, I64(-1), I64(8), I64(4), 1) == -1
