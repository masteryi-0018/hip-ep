# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The qdq-group entry points (include/npu_emu_qdq.h) against the numpy
reference in golden_qdq.py, bit for bit. 8-bit inputs are exhaustive where
the op is elementwise; coefficients mix random values with ties (x * 0.5),
values that need the single-rounding MAC, saturation and shifts."""

import ctypes

import numpy as np
import pytest

import golden_qdq as q
import npu
from npu import (FILL, I64, OUT, U32, DevList, Host, NPU_I8, NPU_I16, NPU_I32, NPU_U8, NPU_U16)
from npu import call as callx

RNG = np.random.default_rng(2027)
T = {"i8": (NPU_I8, np.int8), "u8": (NPU_U8, np.uint8), "i16": (NPU_I16, np.int16),
     "u16": (NPU_U16, np.uint16), "i32": (NPU_I32, np.int32)}
def run(name, *args, n, t):
    """Call `name` with an output of n elements of type t (a key of T)."""
    dt = T[t][1]
    return callx(name, *args, out_nbytes=n * np.dtype(dt).itemsize, out_dtype=dt)


def f(x):
    return float(np.float32(x))


def assert_same(got, want):
    got = np.asarray(got).reshape(-1)
    want = np.asarray(want).astype(got.dtype).reshape(-1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{got.size} differ, first at {bad[:6].tolist()}: "
                           f"got {got[bad[:6]].tolist()} want {want[bad[:6]].tolist()}")


def values(t, n=1 << 14):
    """Every value of an 8-bit type, a random sample plus the ends of a 16-bit one."""
    dt = T[t][1]
    info = np.iinfo(dt)
    if info.bits == 8:
        return np.arange(info.min, info.max + 1).astype(dt)
    edge = np.array([info.min, info.min + 1, -1 if info.min else 1, 0, 1, info.max - 1,
                     info.max]).astype(dt)
    return np.concatenate([edge, RNG.integers(info.min, info.max, n, endpoint=True).astype(dt)])


def rnd(t, shape):
    info = np.iinfo(T[t][1])
    return RNG.integers(info.min, info.max, shape, endpoint=True).astype(T[t][1])


# (scale, offset): ties at x * 0.5, ties at x * 0.5 + offset, values where a
# two-rounding MAC differs (1 + 2^-23 against a large offset), saturation,
# negative scale and random.
COEFFS = [(0.5, 0.0), (0.5, 0.5), (-0.5, 3.0), (f(1 + 2.0 ** -23), f(-(2.0 ** 15) - 0.5)),
          (f(1 / 3), 0.5), (300.0, -7.0), (f(0.0123), f(77.7)), (1.0, 0.0)] + \
         [(f(np.exp2(RNG.uniform(-8, 3)) * RNG.choice([-1, 1])), f(RNG.uniform(-300, 300)))
          for _ in range(4)]


def pairs(bits):
    s = ["i8", "u8"] if bits == 8 else ["i16", "u16"]
    return [(a, b) for a in s for b in s]


# ---- A ----------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("ci", range(len(COEFFS)))
def test_abs(bits, ci):
    c1, c0 = COEFFS[ci]
    for it, ot in pairs(bits):
        x = values(it)
        for zp in (0.0, -5.0, f(123.25)):
            got = run(f"npu_elementwise_abs_a{bits}", x, T[it][0], OUT, T[ot][0], I64(len(x)),
                      c1, c0, zp, n=len(x), t=ot)
            assert_same(got, q.abs_(x, c1, c0, zp, ot))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("ci", range(len(COEFFS)))
def test_neg(bits, ci):
    c1, c0 = COEFFS[ci]
    for it, ot in pairs(bits):
        x = values(it)
        got = run(f"npu_elementwise_neg_a{bits}", x, T[it][0], OUT, T[ot][0], I64(len(x)),
                  c1, c0, n=len(x), t=ot)
        assert_same(got, q.requant(x, c1, c0, ot))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("ci", range(len(COEFFS)))
def test_clip(bits, ci):
    c1, c0 = COEFFS[ci]
    for it, ot in pairs(bits):
        x = values(it)
        for lo, hi in ((-11.5, 11.5), (f(-0.1), 200.25), (-1e9, 1e9)):
            got = run(f"npu_elementwise_clip_a{bits}", x, T[it][0], OUT, T[ot][0],
                      I64(len(x)), c1, c0, f(lo), f(hi), n=len(x), t=ot)
            assert_same(got, q.clip(x, c1, c0, lo, hi, ot))


# ---- B ----------------------------------------------------------------------------

def binary_inputs(at, bt):
    """All 8-bit pairs, or random 16-bit pairs with many equal dequantized values."""
    if np.dtype(T[at][1]).itemsize == 1:
        a, b = np.meshgrid(values(at), values(bt), indexing="ij")
        return a.reshape(-1), b.reshape(-1)
    a, b = rnd(at, 1 << 16), rnd(bt, 1 << 16)
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
        assert_same(got, q.compare(a, b, sa, sb, za, zb, op))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("qi", range(len(QPAIRS)))
def test_max(bits, qi):
    sa, sb, za, zb = QPAIRS[qi]
    for at, bt in pairs(bits):
        a, b = binary_inputs(at, bt)
        for so, zo in ((2.0, 0.5), (f(118.06), 8.0), (f(1 / 0.0123), -3.0)):
            ot = at
            got = run(f"npu_elementwise_max_a{bits}", a, T[at][0], b, T[bt][0], OUT, T[ot][0],
                      I64(len(a)), sa, sb, so, za, zb, zo, n=len(a), t=ot)
            assert_same(got, q.minmax(a, b, sa, sb, so, za, zb, zo, ot, "max"))


# ---- C ----------------------------------------------------------------------------

SHAPES = [((5, 7, 70), (5, 7, 70)), ((5, 7, 70), (1, 7, 70)), ((5, 7, 70), (5, 7, 1)),
          ((5, 7, 70), (5, 1, 70)), ((3, 4, 33), (1, 1, 1))]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("si", range(len(SHAPES)))
@pytest.mark.parametrize("pos", [0, 1])
def test_where(bits, si, pos):
    shape, cshape = SHAPES[si]
    cond = RNG.integers(-2, 3, cshape).astype(np.int8)
    x = RNG.integers(0, 1 << bits, shape).astype(np.uint8 if bits == 8 else np.uint16)
    y = int(RNG.integers(0, 1 << bits))
    got = callx(f"npu_elementwise_broadcasting3d_where_a{bits}", cond, x, OUT,
                *[I64(v) for v in shape + cshape], U32(y | 0xABCD0000), pos,
                out_nbytes=x.nbytes, out_dtype=x.dtype)
    assert_same(got, q.where(cond, x, y, pos))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("si", range(len(SHAPES)))
@pytest.mark.parametrize("pos", [0, 1])
def test_qdq_where(bits, si, pos):
    shape, cshape = SHAPES[si]
    ct = np.int8 if bits == 8 else np.int16
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        cond = RNG.integers(-2, 3, cshape).astype(ct)
        cond[..., :1] = np.array(0x100 if bits == 16 else 0, ct)   # nonzero only in the high byte
        x = rnd(t, shape)
        y = int(rnd(t, 1)[0])
        for scale, offset in COEFFS[:6]:
            got = run(f"npu_qdq_elementwise_broadcasting3d_where_a{bits}", cond, x, T[t][0], OUT,
                      *[I64(v) for v in shape + cshape], y, pos, scale, offset,
                      n=x.size, t=t)
            assert_same(got, q.requant(q.where(cond, x, y, pos), scale, offset, t))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("si", range(len(SHAPES)))
def test_qdq_min(bits, si):
    shape, bshape = SHAPES[si]
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        a, b = rnd(t, shape), rnd(t, bshape)
        for sa, sb, za, zb in QPAIRS:
            so, zo = f(1 / 0.0123), 5.0
            got = run(f"npu_qdq_elementwise_broadcasting3d_min_a{bits}", a, b, T[t][0], OUT,
                      *[I64(v) for v in shape + bshape], sa, sb, so, za, zb, zo,
                      n=a.size, t=t)
            assert_same(got, q.minmax(a, b, sa, sb, so, za, zb, zo, t, "min"))


# ---- D ----------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("axis", range(1, 8))
def test_reduce(bits, op, axis):
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        x = rnd(t, (4, 6, 70))
        want = q.reduce(x, axis, op)
        got = run(f"npu_reduce_{op}_c64_a{bits}", x, T[t][0], OUT, I64(4), I64(6), I64(70), axis,
                  n=want.size, t=t)
        assert_same(got, want)
        for (scale, offset), shift in zip(COEFFS[:6], (0, 1, -1, 0, 2, -2)):
            got = run(f"npu_qdq_reduce_{op}_c64_a{bits}", x, T[t][0], OUT, I64(4), I64(6),
                      I64(70), axis, offset, scale, shift, n=want.size, t=t)
            assert_same(got, q.requant(want, scale, offset, t, shift))


# ---- E ----------------------------------------------------------------------------

POOLS = [(3, 3, 1, 1), (2, 2, 2, 2), (6, 3, 6, 3), (3, 2, 2, 1), (1, 1, 1, 1)]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("pi", range(len(POOLS)))
def test_maxpool(bits, pi):
    kh, kw, sh, sw = POOLS[pi]
    for it, ot in pairs(bits):
        x = rnd(it, (13, 11, 65))
        m = q.maxpool(x, kh, kw, sh, sw)
        for (scale, offset), shift in zip(COEFFS[:5], (0, -2, 1, 0, 2)):
            got = run(f"npu_qdq_maxpool_a{bits}", x, T[it][0], OUT, T[ot][0], I64(13), I64(11),
                      I64(65), kh, kw, sh, sw, offset, scale, shift, n=m.size, t=ot)
            assert_same(got, q.requant(m, scale, offset, ot, shift))
        if bits == 8 and it == ot:
            got = run("npu_generic_maxpool_a8", x, T[it][0], OUT, I64(13), I64(11), I64(65),
                      kh, kw, sh, sw, n=m.size, t=it)
            assert_same(got, m)


@pytest.mark.parametrize("bits", [8, 16])
def test_globalmaxpool(bits):
    for it, ot in pairs(bits):
        x = rnd(it, (7, 9, 130))
        m = np.asarray(x, np.int64).max(axis=(0, 1))
        for (scale, offset), shift in zip(COEFFS, (0, -2, 1, 0, 2, 0, 0, -1)):
            got = run(f"npu_qdq_globalmaxpool_a{bits}", x, T[it][0], OUT, T[ot][0], I64(7),
                      I64(9), I64(130), offset, scale, shift, n=m.size, t=ot)
            assert_same(got, q.requant(m, scale, offset, ot, shift))


# ---- F ----------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [8, 16])
def test_qdq_concat(bits):
    for it, ot in pairs(bits):
        a, b = rnd(it, (9, 70)), rnd(it, (9, 33))
        for scale, offset in COEFFS:
            got = run(f"npu_qdq_concat_a{bits}", a, b, T[it][0], OUT, T[ot][0], I64(9), I64(70),
                      I64(33), scale, offset, n=9 * 103, t=ot)
            assert_same(got, q.requant(q.concat([a, b]), scale, offset, ot))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("n_in", [1, 3, 6])
def test_qdq_concat_multi(bits, n_in):
    for it, ot in pairs(bits):
        ch = RNG.integers(1, 40, n_in)
        ins = [rnd(it, (5, c)) for c in ch]
        chans = Host((ctypes.c_int64 * n_in)(*[int(c) for c in ch]))
        n = 5 * int(ch.sum())
        for scale, offset in COEFFS[:6]:
            got = run(f"npu_qdq_concat_multi_c_fusion_a{bits}", n_in, DevList(ins), T[it][0], OUT,
                      T[ot][0], I64(5), chans, scale, offset, n=n, t=ot)
            assert_same(got, q.requant(q.concat(ins), scale, offset, ot))
        sc = [COEFFS[k % len(COEFFS)] for k in range(n_in)]
        s = Host((ctypes.c_float * n_in)(*[c[0] for c in sc]))
        o = Host((ctypes.c_float * n_in)(*[c[1] for c in sc]))
        got = run("npu_qdq_concat_multi_c_fusion", n_in, DevList(ins), T[it][0], OUT, T[ot][0],
                  I64(5), chans, s, o, n=n, t=ot)
        want = q.concat([q.requant(x, c[0], c[1], ot) for x, c in zip(ins, sc)])
        assert_same(got, want)


# ---- G ----------------------------------------------------------------------------

def test_qdq_gather():
    for it in ("i8", "u8", "i16", "u16"):
        for ot in ("i8", "u8", "i16", "u16"):
            x = rnd(it, (13, 53))
            idx = RNG.integers(0, 13, (19, 53)).astype(np.int32)
            idx[0, :4] = [-1, 13, 1 << 30, -(1 << 31)]       # out of range: not written
            for scale, offset in COEFFS[:6]:
                got = run("npu_qdq_gather_element_2din_2dindices_axis0", x, T[it][0], idx, OUT,
                          T[ot][0], I64(13), I64(53), I64(19), scale, offset,
                          n=idx.size, t=ot)
                untouched = np.frombuffer(bytes([FILL]) * np.dtype(T[ot][1]).itemsize,
                                          T[ot][1])[0]
                assert_same(got, q.gather_axis0(x, idx, scale, offset, ot, untouched))


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
        x = rnd(t, (H, W, C))
        if variant == "a8w8":
            gt = "u8" if gq[0] == 128 else "i8"
            grid = rnd(gt, (Ho, Wo, 2))
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
        assert_same(got, q.gridsample_nearest(x, grid.astype(np.int64), align, pad, gq, t))


@pytest.mark.parametrize("requantize", [0, 1])
def test_qdq_resize_nearest(requantize):
    for t in ("i8", "u8", "i16", "u16"):
        x = rnd(t, (5, 6, 70))
        sy = RNG.integers(0, 5, 11).astype(np.int32)
        sx = RNG.integers(0, 6, 9).astype(np.int32)
        for scale, offset in COEFFS[:6] + [(0.6, 0.0), (0.6, -0.5)]:
            got = run("npu_qdq_resize_nearest", x, T[t][0], OUT, I64(5), I64(6), I64(70), I64(11),
                      I64(9), sy, sx, requantize, scale, offset, n=11 * 9 * 70, t=t)
            assert_same(got, q.resize_nearest(x, sy, sx, requantize, scale, offset, t))


# ---- H ----------------------------------------------------------------------------
# Data movement then requantization in place; the movement itself is covered by
# test_golden_movement.py, so the expected result is numpy indexing + q.requant.

FUSED_COEFFS = COEFFS[:6] + [(f(1 / 6), f(64 - 128 / 6))]


def fused_types(op):
    if op in ("slice_16b_inner", "slice_16b_inner_range"):
        return ["i8", "i16"]
    return ["i16", "u16"] if op == "slice_16b_c6463" else ["i8", "u8", "i16", "u16"]


@pytest.mark.parametrize("t", fused_types("reshape_innermost"))
def test_qdq_reshape_innermost_requantize_fused(t):
    x = rnd(t, (1001,))
    for c1, c0 in FUSED_COEFFS:
        got = run("npu_qdq_reshape_innermost_requantize_fused", x, OUT, T[t][0], I64(x.size), c1,
                  c0, n=x.size, t=t)
        assert_same(got, q.requant(x, c1, c0, t))


@pytest.mark.parametrize("t", fused_types("slice_innermost"))
def test_qdq_slice_innermost_requantize_fused(t):
    x = rnd(t, (7, 50))
    for start, end, stride in [(3, 47, 3), (0, 50, 1), (10, 11, 1)]:
        want = x[:, start:end:stride]
        for c1, c0 in FUSED_COEFFS:
            got = run("npu_qdq_slice_innermost_requantize_fused", x, OUT, T[t][0], I64(7), I64(50),
                      I64(start), I64(end), I64(stride), c1, c0, n=want.size, t=t)
            assert_same(got, q.requant(want, c1, c0, t))


@pytest.mark.parametrize("t", fused_types("slice_16b_inner_range"))
def test_qdq_slice_16b_inner_range_requantize_fused(t):
    x = rnd(t, (9, 70))
    for c1, c0 in FUSED_COEFFS:
        got = run("npu_qdq_slice_16b_inner_range_requantize_fused", x, OUT, T[t][0], I64(9),
                  I64(70), I64(5), I64(66), c1, c0, n=9 * 61, t=t)
        assert_same(got, q.requant(x[:, 5:66], c1, c0, t))


def two_outputs(name, x, t, *args, n0, n1):
    es = np.dtype(T[t][1]).itemsize
    o0, o1 = npu.callx(name, x, npu.Out(0), npu.Out(1), T[t][0], *args,
                       outs=[n0 * es, n1 * es])
    return np.concatenate([o0.view(T[t][1]), o1.view(T[t][1])])


@pytest.mark.parametrize("t", fused_types("slice_16b_inner"))
def test_qdq_slice_16b_inner_requantize_fused(t):
    x = rnd(t, (9, 70))
    for c1, c0 in FUSED_COEFFS:
        got = two_outputs("npu_qdq_slice_16b_inner_requantize_fused", x, t, I64(9), I64(70),
                          I64(24), c1, c0, n0=9 * 24, n1=9 * 46)
        want = np.concatenate([x[:, :24].reshape(-1), x[:, 24:].reshape(-1)])
        assert_same(got, q.requant(want, c1, c0, t))


@pytest.mark.parametrize("t", fused_types("slice_16b_c6463"))
def test_qdq_slice_16b_c6463_requantize_fused(t):
    x = rnd(t, (11, 64))
    for c1, c0 in FUSED_COEFFS:
        got = two_outputs("npu_qdq_slice_16b_c6463_requantize_fused", x, t, I64(11), c1, c0,
                          n0=11 * 63, n1=11)
        want = np.concatenate([x[:, :63].reshape(-1), x[:, 63]])
        assert_same(got, q.requant(want, c1, c0, t))


@pytest.mark.parametrize("t", fused_types("transpose_k8n8"))
def test_qdq_transpose_k8n8_requantize_fused(t):
    x = rnd(t, (16, 40))
    for c1, c0 in FUSED_COEFFS:
        got = run("npu_qdq_transpose_k8n8_requantize_fused", x, OUT, T[t][0], I64(16), I64(40),
                  c1, c0, n=x.size, t=t)
        assert_same(got, q.requant(x.T, c1, c0, t))


@pytest.mark.parametrize("t", fused_types("transpose_unaligned"))
def test_qdq_transpose_unaligned_requantize_fused(t):
    shape, perm = (2, 3, 5, 7), (2, 0, 3, 1)
    x = rnd(t, shape)
    for c1, c0 in FUSED_COEFFS:
        got = run("npu_qdq_transpose_unaligned_requantize_fused", x, OUT, T[t][0],
                  Host(np.array(shape, np.int64)), Host(np.array(perm, np.int32)), c1, c0,
                  n=x.size, t=t)
        assert_same(got, q.requant(np.transpose(x, perm), c1, c0, t))
