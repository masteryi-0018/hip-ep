# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""qdq.h entry points against the numpy reference, bit for bit."""

import ctypes

import numpy as np
import pytest

import golden_qdq as q
import npu
from npu import FILL, I64, OUT, DevList, Host, raw_rc
from npu import call as callx
from testing import INT_ALL as T
from testing import assert_same, rand_ints, rejected, run, shp

RNG = np.random.default_rng(2027)


def S16(v):
    """An int16_t argument."""
    return Host(ctypes.c_int16(v))


def f(x):
    return float(np.float32(x))


def values(t, n=1 << 14):
    """Every value of an 8-bit type, a random sample plus the ends of a 16-bit one."""
    dt = T[t][1]
    info = np.iinfo(dt)
    if info.bits == 8:
        return np.arange(info.min, info.max + 1).astype(dt)
    edge = np.array([info.min, info.min + 1, -1 if info.min else 1, 0, 1, info.max - 1,
                     info.max]).astype(dt)
    return np.concatenate([edge, RNG.integers(info.min, info.max, n, endpoint=True).astype(dt)])


# (scale, offset): ties, single-rounding cases, saturation, negative scale, random.
COEFFS = [(0.5, 0.0), (0.5, 0.5), (-0.5, 3.0), (f(1 + 2.0 ** -23), f(-(2.0 ** 15) - 0.5)),
          (f(1 / 3), 0.5), (300.0, -7.0), (f(0.0123), f(77.7)), (1.0, 0.0)] + \
         [(f(np.exp2(RNG.uniform(-8, 3)) * RNG.choice([-1, 1])), f(RNG.uniform(-300, 300)))
          for _ in range(4)]


def pairs(bits):
    s = ["i8", "u8"] if bits == 8 else ["i16", "u16"]
    return [(a, b) for a in s for b in s]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("ci", range(len(COEFFS)))
def test_abs(bits, ci):
    c1, c0 = COEFFS[ci]
    for it, ot in pairs(bits):
        x = values(it)
        for zp in (0.0, -5.0, f(123.25)):
            got = run(f"npu_elementwise_abs_a{bits}", x, T[it][0], OUT, T[ot][0], I64(len(x)),
                      c1, c0, zp, n=len(x), t=ot)
            assert_same(got, getattr(q, f"elementwise_abs_a{bits}")(x, c1, c0, zp, ot))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("ci", range(len(COEFFS)))
def test_neg(bits, ci):
    c1, c0 = COEFFS[ci]
    for it, ot in pairs(bits):
        x = values(it)
        got = run(f"npu_elementwise_neg_a{bits}", x, T[it][0], OUT, T[ot][0], I64(len(x)),
                  c1, c0, n=len(x), t=ot)
        assert_same(got, getattr(q, f"elementwise_neg_a{bits}")(x, c1, c0, ot))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("ci", range(len(COEFFS)))
def test_clip(bits, ci):
    c1, c0 = COEFFS[ci]
    for it, ot in pairs(bits):
        x = values(it)
        for lo, hi in ((-11.5, 11.5), (f(-0.1), 200.25), (-1e9, 1e9)):
            got = run(f"npu_elementwise_clip_a{bits}", x, T[it][0], OUT, T[ot][0],
                      I64(len(x)), c1, c0, f(lo), f(hi), n=len(x), t=ot)
            assert_same(got, getattr(q, f"elementwise_clip_a{bits}")(x, c1, c0, lo, hi, ot))


def binary_inputs(at, bt):
    """All 8-bit pairs, or random 16-bit pairs with many equal dequantized values."""
    if np.dtype(T[at][1]).itemsize == 1:
        a, b = np.meshgrid(values(at), values(bt), indexing="ij")
        return a.reshape(-1), b.reshape(-1)
    a, b = rand_ints(RNG, T[at][1], 1 << 16), rand_ints(RNG, T[bt][1], 1 << 16)
    b[: 1 << 14] = a[: 1 << 14].astype(np.int64).astype(T[bt][1])   # equal codes
    return a, b


QPAIRS = [(0.5, 0.5, 0.0, 0.0), (f(0.0060805), f(0.0060805), f(0.10945), f(-0.17025)),
          (0.25, 0.5, -4.0, 2.0), (f(1 / 3), f(2 / 3), 0.0, 0.0)]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("op", ["equal", "greater"])
@pytest.mark.parametrize("qi", range(len(QPAIRS)))
def test_compare(bits, op, qi):
    sa, sb, za, zb = QPAIRS[qi]
    for at, bt in pairs(bits):
        a, b = binary_inputs(at, bt)
        got = run(f"npu_elementwise_{op}_a{bits}", a, T[at][0], b, T[bt][0], OUT, I64(len(a)),
                  sa, sb, za, zb, n=len(a), t="i8")
        assert_same(got, getattr(q, f"elementwise_{op}_a{bits}")(a, b, sa, sb, za, zb))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("qi", range(len(QPAIRS)))
def test_max(bits, qi):
    sa, sb, za, zb = QPAIRS[qi]
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        a, b = binary_inputs(t, t)
        for so, zo in ((2.0, 0.5), (f(118.06), 8.0), (f(1 / 0.0123), -3.0)):
            got = run(f"npu_elementwise_max_a{bits}", a, T[t][0], b, OUT, I64(len(a)), sa, sb, so,
                      za, zb, zo, n=len(a), t=t)
            assert_same(got, getattr(q, f"elementwise_max_a{bits}")(a, b, sa, sb, so, za, zb, zo,
                                                                     t))


SHAPES = [((5, 7, 70), (5, 7, 70)), ((5, 7, 70), (1, 7, 70)), ((5, 7, 70), (5, 7, 1)),
          ((5, 7, 70), (5, 1, 70)), ((3, 4, 33), (1, 1, 1))]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("si", range(len(SHAPES)))
@pytest.mark.parametrize("pos", [0, 1])
def test_where(bits, si, pos):
    shape, cshape = SHAPES[si]
    cond = RNG.integers(-2, 3, cshape).astype(np.int8)
    x = RNG.integers(0, 1 << bits, shape).astype(np.uint8 if bits == 8 else np.uint16)
    y = int(RNG.integers(0, 1 << bits)) | 0x2BCD0000   # only the low bits are the scalar
    got = callx(f"npu_elementwise_broadcasting3d_where_a{bits}", cond, x, OUT,
                shp(shape), shp(cshape), pos, y, out_nbytes=x.nbytes, out_dtype=x.dtype)
    assert_same(got, getattr(q, f"elementwise_broadcasting3d_where_a{bits}")(cond, x, pos, y))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("si", range(len(SHAPES)))
@pytest.mark.parametrize("pos", [0, 1])
def test_qdq_where(bits, si, pos):
    shape, cshape = SHAPES[si]
    ct = np.int8 if bits == 8 else np.int16
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        cond = RNG.integers(-2, 3, cshape).astype(ct)
        cond[..., :1] = np.array(0x100 if bits == 16 else 0, ct)   # nonzero only in the high byte
        x = rand_ints(RNG, T[t][1], shape)
        y = int(rand_ints(RNG, T[t][1], 1)[0])
        for scale, offset in COEFFS[:6]:
            got = run(f"npu_qdq_elementwise_broadcasting3d_where_a{bits}", cond, x, T[t][0], OUT,
                      shp(shape), shp(cshape), pos, scale, offset, y, n=x.size, t=t)
            want = getattr(q, f"qdq_elementwise_broadcasting3d_where_a{bits}")(
                cond, x, pos, scale, offset, y, t)
            assert_same(got, want)


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("si", range(len(SHAPES)))
def test_qdq_min(bits, si):
    shape, bshape = SHAPES[si]
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        a, b = rand_ints(RNG, T[t][1], shape), rand_ints(RNG, T[t][1], bshape)
        for sa, sb, za, zb in QPAIRS:
            so, zo = f(1 / 0.0123), 5.0
            got = run(f"npu_qdq_elementwise_broadcasting3d_min_a{bits}", a, T[t][0], b, OUT,
                      shp(shape), shp(bshape), sa, sb, so, za, zb, zo, n=a.size, t=t)
            want = getattr(q, f"qdq_elementwise_broadcasting3d_min_a{bits}")(
                a, b, sa, sb, so, za, zb, zo, t)
            assert_same(got, want)


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("axis", range(1, 8))
def test_reduce(bits, op, axis):
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        x = rand_ints(RNG, T[t][1], (4, 6, 70))
        want = getattr(q, f"reduce_{op}_c64_a{bits}")(x, axis)
        got = run(f"npu_reduce_{op}_c64_a{bits}", x, T[t][0], OUT, I64(4), I64(6), I64(70), axis,
                  n=want.size, t=t)
        assert_same(got, want)
        for (scale, offset), shift in zip(COEFFS[:6], (0, 1, -1, 0, 2, -2)):
            got = run(f"npu_qdq_reduce_{op}_c64_a{bits}", x, T[t][0], OUT, I64(4), I64(6),
                      I64(70), axis, offset, scale, S16(shift), n=want.size, t=t)
            want_q = getattr(q, f"qdq_reduce_{op}_c64_a{bits}")(x, axis, offset, scale, shift, t)
            assert_same(got, want_q)


POOLS = [(3, 3, 1, 1), (2, 2, 2, 2), (6, 3, 6, 3), (3, 2, 2, 1), (1, 1, 1, 1)]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("pi", range(len(POOLS)))
def test_maxpool(bits, pi):
    kh, kw, sh, sw = POOLS[pi]
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        x = rand_ints(RNG, T[t][1], (13, 11, 65))
        m = q.generic_maxpool_a8(x, kh, kw, sh, sw)
        for (scale, offset), shift in zip(COEFFS[:5], (0, -2, 1, 0, 2)):
            got = run(f"npu_qdq_maxpool_a{bits}", x, T[t][0], OUT, I64(13), I64(11), I64(65), kh,
                      kw, sh, sw, offset, scale, S16(shift), n=m.size, t=t)
            want = getattr(q, f"qdq_maxpool_a{bits}")(x, kh, kw, sh, sw, offset, scale, shift, t)
            assert_same(got, want)
        if bits == 8:
            got = run("npu_generic_maxpool_a8", x, T[t][0], OUT, I64(13), I64(11), I64(65),
                      kh, kw, sh, sw, n=m.size, t=t)
            assert_same(got, m)


@pytest.mark.parametrize("bits", [8, 16])
def test_globalmaxpool(bits):
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        x = rand_ints(RNG, T[t][1], (7, 9, 130))
        for (scale, offset), shift in zip(COEFFS, (0, -2, 1, 0, 2, 0, 0, -1)):
            got = run(f"npu_qdq_globalmaxpool_a{bits}", x, T[t][0], OUT, I64(7), I64(9), I64(130),
                      offset, scale, S16(shift), n=130, t=t)
            want = getattr(q, f"qdq_globalmaxpool_a{bits}")(x, offset, scale, shift, t)
            assert_same(got, want)


@pytest.mark.parametrize("bits", [8, 16])
def test_qdq_concat(bits):
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        a, b = rand_ints(RNG, T[t][1], (9, 70)), rand_ints(RNG, T[t][1], (9, 33))
        for scale, offset in COEFFS:
            got = run(f"npu_qdq_concat_a{bits}", a, T[t][0], b, OUT, I64(9), I64(70), I64(33),
                      scale, offset, n=9 * 103, t=t)
            assert_same(got, getattr(q, f"qdq_concat_a{bits}")(a, b, scale, offset, t))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("n_in", [2, 3, 6])
def test_qdq_concat_multi(bits, n_in):
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        ch = RNG.integers(1, 40, n_in)
        ins = [rand_ints(RNG, T[t][1], (5, c)) for c in ch]
        chans = Host((ctypes.c_int64 * n_in)(*[int(c) for c in ch]))
        n = 5 * int(ch.sum())
        for scale, offset in COEFFS[:6]:
            got = run(f"npu_qdq_concat_multi_c_fusion_a{bits}", DevList(ins), T[t][0], OUT, I64(5),
                      n_in, chans, scale, offset, n=n, t=t)
            want = getattr(q, f"qdq_concat_multi_c_fusion_a{bits}")(ins, scale, offset, t)
            assert_same(got, want)
        sc = [COEFFS[k % len(COEFFS)] for k in range(n_in)]
        c0 = [c[0] for c in sc]
        c1 = [c[1] for c in sc]
        got = run("npu_qdq_concat_multi_c_fusion", DevList(ins), T[t][0], OUT, I64(5), n_in, chans,
                  Host((ctypes.c_float * n_in)(*c0)), Host((ctypes.c_float * n_in)(*c1)), n=n,
                  t=t)
        assert_same(got, q.qdq_concat_multi_c_fusion(ins, c0, c1, t))


def test_qdq_gather():
    for it in ("i8", "u8", "i16", "u16"):
        for ot in ("i8", "u8", "i16", "u16"):
            x = rand_ints(RNG, T[it][1], (13, 53))
            idx = RNG.integers(0, 13, (19, 53)).astype(np.int32)
            idx[0, :4] = [-1, 13, 1 << 30, -(1 << 31)]       # out of range: not written
            for scale, offset in COEFFS[:6]:
                got = run("npu_qdq_gather_element_2din_2dindices_axis0", x, T[it][0], idx, OUT,
                          T[ot][0], I64(13), I64(53), I64(19), scale, offset,
                          n=idx.size, t=ot)
                untouched = np.frombuffer(bytes([FILL]) * np.dtype(T[ot][1]).itemsize,
                                          T[ot][1])[0]
                assert_same(got, q.qdq_gather_element_2din_2dindices_axis0(x, idx, scale, offset,
                                                                           ot, untouched))


class GridQdq(ctypes.Structure):
    _fields_ = [("grid_zp", ctypes.c_int32), ("grid_scale", ctypes.c_float),
                ("ifm_zp", ctypes.c_int32), ("ifm_scale", ctypes.c_float),
                ("ofm_zp", ctypes.c_int32), ("ofm_scale", ctypes.c_float)]


GRIDQ = [(28526, f(0.00010520511568756774), 3, f(0.00014300664770416915), 5, f(1 / 0.0001342372)),
         (128, f(1 / 127), 128, f(0.0123), 128, f(1 / 0.0151)),
         (0, f(1 / 2000), -7, f(0.5), 2, 1.0)]


@pytest.mark.parametrize("variant", ["a8", "a16", "a8w8"])
@pytest.mark.parametrize("align", [0, 1])
@pytest.mark.parametrize("pad", [0, 1])
@pytest.mark.parametrize("gi", range(len(GRIDQ)))
def test_gridsample(variant, align, pad, gi):
    H, W, C, Ho, Wo = 9, 13, 4, 7, 11
    gq = GRIDQ[gi]
    for t in (["i16", "u16"] if variant == "a16" else ["i8", "u8"]):
        x = rand_ints(RNG, T[t][1], (H, W, C))
        if variant == "a8w8":
            gt = "u8" if gq[0] == 128 else "i8"
            grid = rand_ints(RNG, T[gt][1], (Ho, Wo, 2))
            args = (grid, T[gt][0])
        else:
            grid = RNG.integers(0, 65536, (Ho, Wo, 2)).astype(np.uint16)
            # codes near the dequantized range [-1.2, 1.2] so most samples land inside
            span = int(1.2 / gq[1])
            grid[:Ho // 2] = np.clip(gq[0] + RNG.integers(-span, span, (Ho // 2, Wo, 2)), 0, 65535)
            args = (grid,)
        cq = GridQdq(*gq)
        got = run(f"npu_qdq_gridsample_nearest_{variant}", x, T[t][0], *args, OUT, I64(H), I64(W),
                  I64(C), I64(Ho), I64(Wo), align, pad, Host(ctypes.byref(cq)),
                  n=Ho * Wo * C, t=t)
        want = getattr(q, f"qdq_gridsample_nearest_{variant}")(x, grid.astype(np.int64), align,
                                                               pad, gq, t)
        assert_same(got, want)


@pytest.mark.parametrize("requantize", [0, 1])
def test_qdq_resize_nearest(requantize):
    for t in ("i8", "u8", "i16", "u16"):
        x = rand_ints(RNG, T[t][1], (5, 6, 70))
        sy = RNG.integers(0, 5, 11).astype(np.uint16)
        sx = RNG.integers(0, 6, 9).astype(np.uint16)
        for scale, offset in COEFFS[:6] + [(0.6, 0.0), (0.6, -0.5)]:
            got = run("npu_qdq_resize_nearest", x, T[t][0], OUT, I64(5), I64(6), I64(70), I64(11),
                      I64(9), sy, sx, requantize, scale, offset, n=11 * 9 * 70, t=t)
            assert_same(got, q.qdq_resize_nearest(x, sy, sx, requantize, scale, offset, t))


FUSED_COEFFS = COEFFS[:6] + [(f(1 / 6), f(64 - 128 / 6))]


def fused_types(op):
    if op in ("slice_16b_inner", "slice_16b_inner_range"):
        return ["i8", "i16"]
    return ["i16", "u16"] if op == "slice_16b_c6463" else ["i8", "u8", "i16", "u16"]


@pytest.mark.parametrize("t", fused_types("reshape_innermost"))
def test_qdq_reshape_innermost_requantize_fused(t):
    x = rand_ints(RNG, T[t][1], (1001,))
    for c1, c0 in FUSED_COEFFS:
        got = run("npu_qdq_reshape_innermost_requantize_fused", x, T[t][0], OUT, I64(x.size), c1,
                  c0, n=x.size, t=t)
        assert_same(got, q.qdq_reshape_innermost_requantize_fused(x, c1, c0, t))


@pytest.mark.parametrize("t", fused_types("slice_innermost"))
def test_qdq_slice_innermost_requantize_fused(t):
    x = rand_ints(RNG, T[t][1], (7, 50))
    for start, end, stride in [(3, 47, 2), (0, 50, 1), (10, 11, 1)]:
        for c1, c0 in FUSED_COEFFS:
            want = q.qdq_slice_innermost_requantize_fused(x, start, end, stride, c1, c0, t)
            got = run("npu_qdq_slice_innermost_requantize_fused", x, T[t][0], OUT, I64(7), I64(50),
                      I64(start), I64(end), I64(stride), c1, c0, n=want.size, t=t)
            assert_same(got, want)


@pytest.mark.parametrize("t", fused_types("slice_16b_inner_range"))
def test_qdq_slice_16b_inner_range_requantize_fused(t):
    x = rand_ints(RNG, T[t][1], (9, 70))
    for c1, c0 in FUSED_COEFFS:
        got = run("npu_qdq_slice_16b_inner_range_requantize_fused", x, T[t][0], OUT, I64(9),
                  I64(70), I64(5), I64(66), c1, c0, n=9 * 61, t=t)
        assert_same(got, q.qdq_slice_16b_inner_range_requantize_fused(x, 5, 66, c1, c0, t))


def two_outputs(name, x, t, *args, n0, n1):
    es = np.dtype(T[t][1]).itemsize
    o0, o1 = npu.callx(name, x, T[t][0], npu.Out(0), npu.Out(1), *args,
                       outs=[n0 * es, n1 * es])
    return np.concatenate([o0.view(T[t][1]), o1.view(T[t][1])])


@pytest.mark.parametrize("t", fused_types("slice_16b_inner"))
def test_qdq_slice_16b_inner_requantize_fused(t):
    x = rand_ints(RNG, T[t][1], (9, 70))
    for c1, c0 in FUSED_COEFFS:
        got = two_outputs("npu_qdq_slice_16b_inner_requantize_fused", x, t, I64(9), I64(70),
                          I64(24), c1, c0, n0=9 * 24, n1=9 * 46)
        w0, w1 = q.qdq_slice_16b_inner_requantize_fused(x, 24, c1, c0, t)
        assert_same(got, np.concatenate([w0.reshape(-1), w1.reshape(-1)]))


@pytest.mark.parametrize("t", fused_types("slice_16b_c6463"))
def test_qdq_slice_16b_c6463_requantize_fused(t):
    x = rand_ints(RNG, T[t][1], (11, 64))
    for c1, c0 in FUSED_COEFFS:
        got = two_outputs("npu_qdq_slice_16b_c6463_requantize_fused", x, t, I64(11), c1, c0,
                          n0=11 * 63, n1=11)
        w0, w1 = q.qdq_slice_16b_c6463_requantize_fused(x, c1, c0, t)
        assert_same(got, np.concatenate([w0.reshape(-1), w1.reshape(-1)]))


@pytest.mark.parametrize("t", fused_types("transpose_k8n8"))
def test_qdq_transpose_k8n8_requantize_fused(t):
    x = rand_ints(RNG, T[t][1], (16, 40))
    for c1, c0 in FUSED_COEFFS:
        got = run("npu_qdq_transpose_k8n8_requantize_fused", x, T[t][0], OUT, I64(16), I64(40),
                  c1, c0, n=x.size, t=t)
        assert_same(got, q.qdq_transpose_k8n8_requantize_fused(x, c1, c0, t))


@pytest.mark.parametrize("t", fused_types("transpose_unaligned"))
def test_qdq_transpose_unaligned_requantize_fused(t):
    shape, perm = (2, 3, 5, 7), (2, 0, 3, 1)
    x = rand_ints(RNG, T[t][1], shape)
    for c1, c0 in FUSED_COEFFS:
        got = run("npu_qdq_transpose_unaligned_requantize_fused", x, T[t][0], OUT,
                  Host(np.array(shape, np.int64)), Host(np.array(perm, np.int32)), c1, c0,
                  n=x.size, t=t)
        assert_same(got, q.qdq_transpose_unaligned_requantize_fused(x, perm, c1, c0, t))


def test_bad_args():
    """Every entry rejects (-1) an argument set its kernel cannot run."""
    _empty_work_cases()
    from npu import NPU_BF16, NPU_I8, NPU_I16, NPU_I32, NPU_U8, NPU_U16, Out
    n = None
    d = np.zeros(256, np.uint8)
    nb = 256
    out = Out()
    s3 = shp((2, 2, 2))
    for bits, t, wrong in ((8, NPU_I8, NPU_I16), (16, NPU_I16, NPU_I8)):
        for op in ("abs", "neg", "clip"):
            co = {"abs": (1.0, 0.0, 0.0), "neg": (1.0, 0.0), "clip": (1.0, 0.0, 0.0, 1.0)}[op]
            e = f"npu_elementwise_{op}_a{bits}"
            assert rejected(e, d, wrong, out, t, I64(4), *co, nbytes=nb)
            assert raw_rc(e, n, t, d, t, I64(4), *co) == -1
            assert rejected(e, d, t, out, t, I64(-1), *co, nbytes=nb)
        for op in ("equal", "greater"):
            e = f"npu_elementwise_{op}_a{bits}"
            assert rejected(e, d, t, d, wrong, out, I64(4), 1., 1., 0., 0., nbytes=nb)
            assert raw_rc(e, d, t, n, t, d, I64(4), 1., 1., 0., 0.) == -1
        e = f"npu_elementwise_max_a{bits}"
        assert rejected(e, d, wrong, d, out, I64(4), 1., 1., 1., 0., 0., 0., nbytes=nb)
        assert raw_rc(e, d, t, d, n, I64(4), 1., 1., 1., 0., 0., 0.) == -1
        e = f"npu_elementwise_broadcasting3d_where_a{bits}"
        assert rejected(e, d, d, out, s3, s3, 2, 0, nbytes=nb)              # scalar_position
        assert rejected(e, d, d, out, s3, shp((2, 3, 2)), 1, 0, nbytes=nb)
        assert rejected(e, d, d, out, shp((-2, 2, 2)), s3, 1, 0, nbytes=nb)
        assert raw_rc(e, d, d, d, n, s3, 1, 0) == -1
        e = f"npu_qdq_elementwise_broadcasting3d_where_a{bits}"
        assert rejected(e, d, d, wrong, out, s3, s3, 1, 1.0, 0.0, 0, nbytes=nb)
        assert rejected(e, d, d, t, out, s3, s3, -1, 1.0, 0.0, 0, nbytes=nb)
        assert raw_rc(e, n, d, t, d, s3, s3, 1, 1.0, 0.0, 0) == -1
        e = f"npu_qdq_elementwise_broadcasting3d_min_a{bits}"
        assert rejected(e, d, wrong, d, out, s3, s3, 1., 1., 1., 0., 0., 0., nbytes=nb)
        assert rejected(e, d, t, d, out, shp((1, 2, 2)), s3, 1., 1., 1., 0., 0., 0., nbytes=nb)
        for op in ("max", "min"):
            e = f"npu_reduce_{op}_c64_a{bits}"
            assert rejected(e, d, t, out, I64(2), I64(2), I64(2), 8, nbytes=nb)
            assert rejected(e, d, wrong, out, I64(2), I64(2), I64(2), 1, nbytes=nb)
            e = f"npu_qdq_reduce_{op}_c64_a{bits}"
            assert rejected(e, d, t, out, I64(2), I64(2), I64(2), 0, 0.0, 1.0, S16(0), nbytes=nb)
            assert raw_rc(e, d, t, n, I64(2), I64(2), I64(2), 1, 0.0, 1.0, S16(0)) == -1
        e = f"npu_qdq_maxpool_a{bits}"
        assert rejected(e, d, wrong, out, I64(4), I64(4), I64(2), 2, 2, 1, 1, 0.0, 1.0, S16(0),
                        nbytes=nb)
        assert rejected(e, d, t, out, I64(40), I64(40), I64(1), 33, 2, 1, 1, 0.0, 1.0, S16(0),
                        nbytes=nb)                                         # kh > 32
        assert rejected(e, d, t, out, I64(40), I64(40), I64(1), 2, 2, 1, 33, 0.0, 1.0, S16(0),
                        nbytes=nb)
        e = f"npu_qdq_globalmaxpool_a{bits}"
        assert rejected(e, d, wrong, out, I64(2), I64(2), I64(2), 0.0, 1.0, S16(0), nbytes=nb)
        assert rejected(e, d, t, out, I64(0), I64(2), I64(2), 0.0, 1.0, S16(0), nbytes=nb)
        e = f"npu_qdq_concat_a{bits}"
        assert rejected(e, d, wrong, d, out, I64(1), I64(2), I64(2), 1.0, 0.0, nbytes=nb)
        assert rejected(e, d, t, d, out, I64(1), I64(-1), I64(2), 1.0, 0.0, nbytes=nb)
        e = f"npu_qdq_concat_multi_c_fusion_a{bits}"
        ch = Host(np.ones(7, np.int64))
        assert rejected(e, DevList([d]), t, out, I64(1), 1, ch, 1.0, 0.0, nbytes=nb)  # n_in 1
        assert rejected(e, DevList([d] * 7), t, out, I64(1), 7, ch, 1.0, 0.0, nbytes=nb)
        assert rejected(e, DevList([d, d]), wrong, out, I64(1), 2, ch, 1.0, 0.0, nbytes=nb)
        assert raw_rc(e, DevList([d, d]), t, d, I64(1), 2, n, 1.0, 0.0) == -1
    c = Host((ctypes.c_float * 2)(1.0, 1.0))
    ch = Host(np.ones(2, np.int64))
    e = "npu_qdq_concat_multi_c_fusion"
    assert rejected(e, DevList([d]), NPU_I8, out, I64(1), 1, ch, c, c, nbytes=nb)
    assert rejected(e, DevList([d, d]), NPU_BF16, out, I64(1), 2, ch, c, c, nbytes=nb)
    assert raw_rc(e, DevList([d, d]), NPU_I8, d, I64(1), 2, ch, n, c) == -1
    assert rejected("npu_generic_maxpool_a8", d, NPU_I16, out, I64(4), I64(4), I64(1), 2, 2, 1, 1,
                    nbytes=nb)
    assert rejected("npu_generic_maxpool_a8", d, NPU_I8, out, I64(20), I64(20), I64(1), 16, 2, 1, 1,
                    nbytes=nb)                                             # kh > 15
    e = "npu_qdq_gather_element_2din_2dindices_axis0"
    assert rejected(e, d, NPU_I32, d, out, NPU_I8, I64(2), I64(2), I64(2), 1.0, 0.0, nbytes=nb)
    assert rejected(e, d, NPU_I8, d, out, NPU_I8, I64(2), I64(2), I64(-2), 1.0, 0.0, nbytes=nb)
    assert raw_rc(e, d, NPU_I8, n, d, NPU_I8, I64(2), I64(2), I64(2), 1.0, 0.0) == -1
    gq = GridQdq(0, 1.0, 0, 1.0, 0, 1.0)
    for v, t, extra in (("a8", NPU_I8, ()), ("a16", NPU_I16, ()), ("a8w8", NPU_I8, (NPU_U8,))):
        e = f"npu_qdq_gridsample_nearest_{v}"
        # both output sizes negative would give positive work
        assert rejected(e, d, t, d, *extra, out, I64(2), I64(2), I64(1), I64(-2), I64(-2), 0, 0,
                        Host(ctypes.byref(gq)), nbytes=nb)
        assert rejected(e, d, t, d, *extra, out, I64(2), I64(2), I64(1), I64(1), I64(1), 2, 0,
                        Host(ctypes.byref(gq)), nbytes=nb)
        assert raw_rc(e, d, t, d, *extra, d, I64(2), I64(2), I64(1), I64(1), I64(1), 0, 0,
                      n) == -1
    tab = np.zeros(4, np.uint16)
    e = "npu_qdq_resize_nearest"
    assert rejected(e, d, NPU_I8, out, I64(2), I64(2), I64(1), I64(-2), I64(-2), tab, tab, 0, 1.0,
                    0.0, nbytes=nb)
    assert rejected(e, d, NPU_I8, out, I64(2), I64(2), I64(1), I64(1), I64(1), tab, tab, 2, 1.0,
                    0.0, nbytes=nb)
    assert rejected(e, d, NPU_I32, out, I64(2), I64(2), I64(1), I64(1), I64(1), tab, tab, 0, 1.0,
                    0.0, nbytes=nb)
    assert raw_rc(e, d, NPU_I8, d, I64(2), I64(2), I64(1), I64(1), I64(1), n, tab, 0, 1.0,
                  0.0) == -1
    assert rejected("npu_qdq_reshape_innermost_requantize_fused", d, NPU_BF16, out, I64(4), 1.0,
                    0.0, nbytes=nb)
    assert rejected("npu_qdq_slice_innermost_requantize_fused", d, NPU_I8, out, I64(1), I64(8),
                    I64(0), I64(8), I64(3), 1.0, 0.0, nbytes=nb)          # stride 3
    assert rejected("npu_qdq_slice_innermost_requantize_fused", d, NPU_I8, out, I64(1), I64(8),
                    I64(0), I64(8), I64(0), 1.0, 0.0, nbytes=nb)
    assert rejected("npu_qdq_slice_16b_inner_range_requantize_fused", d, NPU_U8, out, I64(1),
                    I64(8), I64(0), I64(4), 1.0, 0.0, nbytes=nb)
    assert raw_rc("npu_qdq_slice_16b_inner_requantize_fused", d, NPU_U16, d, d, I64(1), I64(8),
                  I64(4), 1.0, 0.0) == -1
    assert raw_rc("npu_qdq_slice_16b_c6463_requantize_fused", d, NPU_I8, d, d, I64(1), 1.0,
                  0.0) == -1
    assert rejected("npu_qdq_transpose_k8n8_requantize_fused", d, NPU_I32, out, I64(8), I64(8), 1.0,
                    0.0, nbytes=nb)
    assert raw_rc("npu_qdq_transpose_unaligned_requantize_fused", d, NPU_I8, d,
                  Host(np.array([2, 2, 2, 2], np.int64)), n, 1.0, 0.0) == -1


def _empty_work_cases():
    """A call with 0 output elements returns 0 after validating every other argument; a null
    required pointer is rejected even then, and a negative size always."""
    from npu import NPU_I8, NPU_I16, NPU_U8
    n, d, z = None, np.zeros(64, np.uint8), I64(0)
    gq = Host(ctypes.byref(GridQdq(0, 1.0, 0, 1.0, 0, 1.0)))
    e1, s0 = shp((1, 1, 1)), shp((2, 0, 2))
    for bits, t in ((8, NPU_I8), (16, NPU_I16)):
        for op, co in (("abs", (1.0, 0.0, 0.0)), ("neg", (1.0, 0.0)),
                       ("clip", (1.0, 0.0, 0.0, 1.0))):
            e = f"npu_elementwise_{op}_a{bits}"
            assert raw_rc(e, d, t, d, t, z, *co) == 0
            assert raw_rc(e, d, t, n, t, z, *co) == -1
            assert raw_rc(e, n, t, d, t, z, *co) == -1
        for op in ("equal", "greater"):
            e = f"npu_elementwise_{op}_a{bits}"
            assert raw_rc(e, d, t, d, t, d, z, 1., 1., 0., 0.) == 0
            assert raw_rc(e, d, t, n, t, d, z, 1., 1., 0., 0.) == -1
            assert raw_rc(e, d, t, d, t, d, I64(-1), 1., 1., 0., 0.) == -1
        e = f"npu_elementwise_max_a{bits}"
        assert raw_rc(e, d, t, d, d, z, 1., 1., 1., 0., 0., 0.) == 0
        assert raw_rc(e, n, t, d, d, z, 1., 1., 1., 0., 0., 0.) == -1
        e = f"npu_elementwise_broadcasting3d_where_a{bits}"
        assert raw_rc(e, d, d, d, s0, e1, 1, 0) == 0
        assert raw_rc(e, d, d, d, s0, s0, 1, 0) == 0
        assert raw_rc(e, n, d, d, s0, e1, 1, 0) == -1
        assert raw_rc(e, d, d, d, s0, shp((1, 2, 1)), 1, 0) == -1   # 2 does not broadcast to 0
        e = f"npu_qdq_elementwise_broadcasting3d_where_a{bits}"
        assert raw_rc(e, d, d, t, d, s0, e1, 1, 1.0, 0.0, 0) == 0
        assert raw_rc(e, d, n, t, d, s0, e1, 1, 1.0, 0.0, 0) == -1
        e = f"npu_qdq_elementwise_broadcasting3d_min_a{bits}"
        assert raw_rc(e, d, t, d, d, s0, e1, 1., 1., 1., 0., 0., 0.) == 0
        assert raw_rc(e, d, t, d, n, s0, e1, 1., 1., 1., 0., 0., 0.) == -1
        assert raw_rc(e, d, t, d, d, shp((2, -1, 2)), e1, 1., 1., 1., 0., 0., 0.) == -1
        for op in ("max", "min"):
            for e, extra in ((f"npu_reduce_{op}_c64_a{bits}", ()),
                             (f"npu_qdq_reduce_{op}_c64_a{bits}", (0.0, 1.0, S16(0)))):
                assert raw_rc(e, d, t, d, z, I64(2), I64(2), 1, *extra) == 0   # H kept
                assert raw_rc(e, d, t, d, z, I64(2), I64(2), 4, *extra) == -1  # H reduced
                assert raw_rc(e, d, t, n, z, I64(2), I64(2), 1, *extra) == -1
                assert raw_rc(e, d, t, d, I64(-1), I64(2), I64(2), 1, *extra) == -1
        e = f"npu_qdq_maxpool_a{bits}"
        assert raw_rc(e, d, t, d, I64(4), I64(4), z, 2, 2, 1, 1, 0.0, 1.0, S16(0)) == 0
        assert raw_rc(e, n, t, d, I64(4), I64(4), z, 2, 2, 1, 1, 0.0, 1.0, S16(0)) == -1
        assert raw_rc(e, d, t, d, I64(4), I64(4), I64(-1), 2, 2, 1, 1, 0.0, 1.0, S16(0)) == -1
        e = f"npu_qdq_globalmaxpool_a{bits}"
        assert raw_rc(e, d, t, d, I64(2), I64(2), z, 0.0, 1.0, S16(0)) == 0
        assert raw_rc(e, d, t, n, I64(2), I64(2), z, 0.0, 1.0, S16(0)) == -1
        e = f"npu_qdq_concat_a{bits}"
        assert raw_rc(e, d, t, d, d, z, I64(2), I64(2), 1.0, 0.0) == 0
        assert raw_rc(e, d, t, d, d, I64(3), z, z, 1.0, 0.0) == 0
        assert raw_rc(e, d, t, n, d, z, I64(2), I64(2), 1.0, 0.0) == -1
        assert raw_rc(e, d, t, d, d, I64(-1), I64(2), I64(2), 1.0, 0.0) == -1
        e = f"npu_qdq_concat_multi_c_fusion_a{bits}"
        ch = Host(np.ones(2, np.int64))
        assert raw_rc(e, DevList([d, d]), t, d, z, 2, ch, 1.0, 0.0) == 0
        assert raw_rc(e, DevList([d, d]), t, n, z, 2, ch, 1.0, 0.0) == -1
    c = Host((ctypes.c_float * 2)(1.0, 1.0))
    ch = Host(np.ones(2, np.int64))
    assert raw_rc("npu_qdq_concat_multi_c_fusion", DevList([d, d]), NPU_I8, d, z, 2, ch, c, c) == 0
    assert raw_rc("npu_qdq_concat_multi_c_fusion", DevList([d, d]), NPU_I8, d, z, 2, ch, c,
                  n) == -1
    e = "npu_generic_maxpool_a8"
    assert raw_rc(e, d, NPU_U8, d, I64(4), I64(4), z, 2, 2, 1, 1) == 0
    assert raw_rc(e, d, NPU_U8, n, I64(4), I64(4), z, 2, 2, 1, 1) == -1
    e = "npu_qdq_gather_element_2din_2dindices_axis0"
    assert raw_rc(e, d, NPU_I8, d, d, NPU_I8, I64(2), z, I64(2), 1.0, 0.0) == 0
    assert raw_rc(e, d, NPU_I8, d, d, NPU_I8, I64(2), I64(2), z, 1.0, 0.0) == 0
    assert raw_rc(e, d, NPU_I8, d, n, NPU_I8, I64(2), I64(2), z, 1.0, 0.0) == -1
    assert raw_rc(e, d, NPU_I8, d, d, NPU_I8, I64(2), I64(-2), I64(2), 1.0, 0.0) == -1
    for v, t, extra in (("a8", NPU_I8, ()), ("a16", NPU_I16, ()), ("a8w8", NPU_I8, (NPU_U8,))):
        e = f"npu_qdq_gridsample_nearest_{v}"
        assert raw_rc(e, d, t, d, *extra, d, I64(2), I64(2), z, I64(1), I64(1), 0, 0, gq) == 0
        assert raw_rc(e, d, t, d, *extra, d, I64(2), I64(2), I64(1), z, I64(3), 0, 0, gq) == 0
        assert raw_rc(e, d, t, n, *extra, d, I64(2), I64(2), I64(1), z, I64(3), 0, 0, gq) == -1
        assert raw_rc(e, d, t, d, *extra, d, z, I64(2), I64(1), z, I64(3), 0, 0, gq) == -1  # H
    tab = np.zeros(4, np.uint16)
    e = "npu_qdq_resize_nearest"
    assert raw_rc(e, d, NPU_I8, d, I64(2), I64(2), z, I64(1), I64(1), tab, tab, 0, 1.0, 0.0) == 0
    assert raw_rc(e, d, NPU_I8, d, I64(2), I64(2), I64(1), z, I64(1), tab, tab, 1, 1.0, 0.0) == 0
    assert raw_rc(e, d, NPU_I8, d, I64(2), I64(2), z, I64(1), I64(1), tab, n, 0, 1.0, 0.0) == -1
    assert raw_rc(e, d, NPU_I8, d, I64(2), I64(2), I64(-1), I64(1), I64(1), tab, tab, 0, 1.0,
                  0.0) == -1
    assert raw_rc("npu_qdq_reshape_innermost_requantize_fused", d, NPU_I8, d, z, 1.0, 0.0) == 0
    assert raw_rc("npu_qdq_reshape_innermost_requantize_fused", d, NPU_I8, n, z, 1.0, 0.0) == -1
    assert raw_rc("npu_qdq_reshape_innermost_requantize_fused", d, NPU_I8, d, I64(-1), 1.0,
                  0.0) == -1
