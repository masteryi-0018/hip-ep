# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Resize entry points (include/npu_emu_resize.h) against the numpy reference
in golden_resize.py, bit for bit: up / down scales on every coordinate mode with
the packers' weight tables, plus random weight tables that hit the kernels'
tap-selection rules (zero s1 / s2 with non-zero s3, a sub-volume whose first
column decides the bottom row), int16 H-pass saturation, output saturation,
accumulators past 2^29, non power-of-two requantize coefficients (0.1 / 0.6),
and float inputs with signed zeros, NaN, infinities, subnormals and overflow."""

import numpy as np
import pytest

import golden as g
import golden_arith_float as ga
import golden_resize as gr
from npu import (I64, NPU_BF16, NPU_F16, NPU_FP8, NPU_I8, NPU_I16, NPU_U8, NPU_U16, NPU_F32,
                 OUT, call, raw_rc)

RNG = np.random.default_rng(2031)
F32 = np.float32
INT = {"i8": (NPU_I8, np.int8), "u8": (NPU_U8, np.uint8), "i16": (NPU_I16, np.int16),
       "u16": (NPU_U16, np.uint16)}
HALF = {"bf16": (NPU_BF16, ga.BF16), "f16": (NPU_F16, ga.F16)}
# (Hi, Wi, Ho, Wo, coordinate mode): up, down, mixed, and integer ratios.
SHAPES = [(5, 7, 13, 16, "half_pixel"), (12, 11, 5, 4, "half_pixel"),
          (6, 9, 12, 6, "align_corners"), (4, 8, 12, 24, "asymmetric"),
          (7, 5, 7, 12, "half_pixel")]


def half_values(t, n, special=True):
    """Half wide random values, half a coarse grid (exact ties, zero sums);
    with `special`, 1/8 of them signed zeros, +-1, infinities, NaN, the largest
    finite, the smallest normal and subnormals."""
    f16 = t == ga.F16
    wide = (RNG.standard_normal(n) * np.exp2(RNG.integers(-6, 7, n))).astype(F32)
    grid = (RNG.integers(-12, 13, n) * F32(0.375)).astype(F32)
    v = np.where(RNG.random(n) < 0.5, wide, grid)
    bits = ga.round16(v, t)
    bits = np.where((v == 0) & (RNG.random(n) < 0.5), bits | np.uint16(0x8000), bits)
    if special:
        sp = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x3C00 if f16 else 0x3F80,
                       0xBC00 if f16 else 0xBF80, 0x7C00 if f16 else 0x7F80,
                       0xFC00 if f16 else 0xFF80, 0x7E00 if f16 else 0x7FC0,
                       0x7BFF if f16 else 0x7F7F, 0xFBFF if f16 else 0xFF7F,
                       0x0400 if f16 else 0x0080, 0x03FF if f16 else 0x007F], np.uint16)
        pos = RNG.choice(n, n // 8, replace=False)
        bits[pos] = sp[RNG.integers(0, len(sp), n // 8)]
    return bits.astype(np.uint16)


def fp8_values(n, nan_rate=0.01):
    """Random E4M3 codes with some NaN (0x7F) and both zeros."""
    c = RNG.integers(0, 256, n).astype(np.uint8)
    c = np.where((c & 0x7F) == 0x7F, np.uint8(0x38), c)
    c = np.where(RNG.random(n) < nan_rate, np.uint8(0x7F), c)
    c = np.where(RNG.random(n) < 0.05, RNG.choice(np.array([0x00, 0x80], np.uint8), n), c)
    return c.astype(np.uint8)


def same(got, want, what=""):
    got = np.ascontiguousarray(got).reshape(-1)
    want = np.ascontiguousarray(np.asarray(want).astype(got.dtype)).reshape(-1)
    assert got.size == want.size, f"{what}: size {got.size} != {want.size}"
    gb, wb = got.view(f"u{got.itemsize}"), want.view(f"u{want.itemsize}")
    bad = np.nonzero(gb != wb)[0]
    assert bad.size == 0, (f"{what}: {bad.size}/{got.size} differ, first {bad[:6].tolist()}: "
                           f"got {gb[bad[:6]].tolist()} want {wb[bad[:6]].tolist()}")


def axes(Hi, Wi, Ho, Wo, mode):
    return gr.bilinear_axis(Hi, Ho, mode), gr.bilinear_axis(Wi, Wo, mode)


def random_q15(Ho, Wo):
    """Random 4-tap Q15 weights summing to at most 2^15, with zeros injected so
    that s1 == 0 / s2 == 0 occur next to non-zero s3."""
    cut = np.sort(RNG.integers(0, 1 << 15, (Ho, Wo, 3)), axis=2)
    s = np.diff(np.concatenate([np.zeros((Ho, Wo, 1), np.int64), cut,
                                np.full((Ho, Wo, 1), 1 << 15)], 2), axis=2)
    s = np.where(RNG.random((Ho, Wo, 4)) < 0.25, 0, s)
    return s.astype(np.uint16)


def int_input(t, shape):
    info = np.iinfo(INT[t][1])
    x = RNG.integers(info.min, info.max, shape, endpoint=True)
    # some pixels at the extremes so saturation and |acc| >= 2^29 occur
    x = np.where(RNG.random(shape) < 0.2, RNG.choice([info.min, info.max], shape), x)
    return x.astype(INT[t][1])


def run_int(x, ih, iw, wts, t, route, sv_wo, req, c0, c1):
    Hi, Wi, C = x.shape
    Ho, Wo = len(ih), len(iw)
    nt, dt = INT[t]
    got = call("npu_resize_bilinear", x, OUT, nt, I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo),
               np.asarray(ih, np.int32), np.asarray(iw, np.int32), np.asarray(wts, np.uint16),
               route, I64(sv_wo), req, float(c0), float(c1),
               out_nbytes=Ho * Wo * C * np.dtype(dt).itemsize, out_dtype=dt)
    want = gr.bilinear_int(x, ih, iw, wts, route, sv_wo, req, c0, c1, dt)
    same(got, want, f"{t} route={route} req={req}")


# (requantize, c0, c1): plain shift; ratios with many ties; saturating gain.
QDQ = [(0, 1.0, 0.0), (1, float(F32(0.1 / 0.6)), float(F32(-3.3))),
       (1, float(F32(0.375 / 0.25)), float(F32(23 - 1.5 * -19))), (1, 3.7, -11.0)]


@pytest.mark.parametrize("t,route", [(t, gr.TAP4) for t in INT] + [("i8", gr.SEP), ("u8", gr.SEP)])
@pytest.mark.parametrize("shape", range(len(SHAPES)))
@pytest.mark.parametrize("qdq", range(len(QDQ)))
def test_bilinear_int_tables(t, route, shape, qdq):
    Hi, Wi, Ho, Wo, mode = SHAPES[shape]
    (ih, wy0, wy1), (iw, wx0, wx1) = axes(Hi, Wi, Ho, Wo, mode)
    if route == gr.SEP:
        wts = gr.sep_table(wy0, wy1, wx0, wx1, 7 if t == "u8" else 8)
    else:
        wts = gr.q15_table(wy0, wy1, wx0, wx1)
    x = int_input(t, (Hi, Wi, 24))
    req, c0, c1 = QDQ[qdq]
    run_int(x, ih, iw, wts, t, route, 6, req, c0, c1)


@pytest.mark.parametrize("t", list(INT))
@pytest.mark.parametrize("qdq", range(len(QDQ)))
@pytest.mark.parametrize("sv_wo", [0, 4, 5])
def test_bilinear_int_4tap_random(t, qdq, sv_wo):
    Hi, Wi, Ho, Wo = 9, 10, 11, 14
    ih = RNG.integers(0, Hi - 1, Ho)
    iw = RNG.integers(0, Wi - 1, Wo)
    wts = random_q15(Ho, Wo)
    x = int_input(t, (Hi, Wi, 16))
    req, c0, c1 = QDQ[qdq]
    run_int(x, ih, iw, wts, t, gr.TAP4, sv_wo, req, c0, c1)


@pytest.mark.parametrize("t", ["i8", "u8"])
@pytest.mark.parametrize("qdq", range(len(QDQ)))
def test_bilinear_int_separable_random(t, qdq):
    """qy pairs up to 700 overflow the int16 H-pass (saturated); zero qy1 /
    qx1 and the right-edge x0 + 1 == Wi read."""
    Hi, Wi, Ho, Wo = 8, 9, 13, 12
    ih = RNG.integers(0, Hi - 1, Ho)
    iw = RNG.integers(0, Wi, Wo)  # includes Wi - 1 (the +1 tap is past the edge)
    qy = RNG.integers(0, 350, (Ho, 2))
    qx = RNG.integers(0, 129, (Wo, 2))
    qy[RNG.random(Ho) < 0.3, 1] = 0
    qx[RNG.random(Wo) < 0.3, 1] = 0
    qx[iw == Wi - 1, 1] = 0
    wts = np.concatenate([qy.reshape(-1), qx.reshape(-1)]).astype(np.uint16)
    x = int_input(t, (Hi, Wi, 16))
    req, c0, c1 = QDQ[qdq]
    run_int(x, ih, iw, wts, t, gr.SEP, 0, req, c0, c1)


# ---- bf16 / f16 --------------------------------------------------------------------------

def half_input(tt, shape, special=True):
    return half_values(tt, int(np.prod(shape)), special).reshape(shape)


def random_half_weights(Ho, Wo, tt):
    """Random weights (wide range, so sums overflow / underflow), zeros of both
    signs injected, a few subnormal / NaN weights."""
    w = (RNG.standard_normal((Ho, Wo, 4)) * np.exp2(RNG.integers(-3, 3, (Ho, Wo, 4)))).astype(F32)
    bits = ga.round16(w, tt).astype(np.uint16)
    z = RNG.random((Ho, Wo, 4))
    bits = np.where(z < 0.15, 0, bits)
    bits = np.where((z >= 0.15) & (z < 0.2), 0x8000, bits)
    bits = np.where((z >= 0.2) & (z < 0.21), 0x0001, bits)
    return bits.astype(np.uint16)


def run_half(xbits, ih, iw, sbits, sv_wo, t):
    Hi, Wi, C = xbits.shape
    Ho, Wo = len(ih), len(iw)
    nt, tt = HALF[t]
    got = call("npu_resize_bilinear_bf16", xbits, OUT, nt, I64(Hi), I64(Wi), I64(C), I64(Ho),
               I64(Wo), np.asarray(ih, np.int32), np.asarray(iw, np.int32),
               np.asarray(sbits, np.uint16), I64(sv_wo),
               out_nbytes=Ho * Wo * C * 2, out_dtype=np.uint16)
    same(got, gr.bilinear_bf16(xbits, ih, iw, sbits, sv_wo, tt), t)


@pytest.mark.parametrize("t", list(HALF))
@pytest.mark.parametrize("shape", range(len(SHAPES)))
@pytest.mark.parametrize("special", [False, True])
def test_bilinear_bf16_tables(t, shape, special):
    Hi, Wi, Ho, Wo, mode = SHAPES[shape]
    (ih, wy0, wy1), (iw, wx0, wx1) = axes(Hi, Wi, Ho, Wo, mode)
    s = gr.float_table(wy0, wy1, wx0, wx1, HALF[t][1])
    run_half(half_input(HALF[t][1], (Hi, Wi, 32), special), ih, iw, s, 8, t)


@pytest.mark.parametrize("t", list(HALF))
@pytest.mark.parametrize("sv_wo", [0, 3])
def test_bilinear_bf16_random(t, sv_wo):
    Hi, Wi, Ho, Wo = 7, 8, 10, 9
    ih = RNG.integers(0, Hi - 1, Ho)
    iw = RNG.integers(0, Wi - 1, Wo)
    tt = HALF[t][1]
    run_half(half_input(tt, (Hi, Wi, 24)), ih, iw, random_half_weights(Ho, Wo, tt), sv_wo, t)


def test_bilinear_f16_saturate_and_flush():
    """Sums past 65504 saturate (not inf); sums below 2^-14 flush to signed 0."""
    Hi, Wi, Ho, Wo = 2, 2, 2, 2
    x = np.array([0x7BFF, 0x7BFF, 0x0400, 0x8400] * 2, np.uint16).reshape(Hi, Wi, 2)
    s = np.tile(np.array([0x3C00, 0x3C00, 0x0000, 0x0000], np.uint16), (Ho, Wo, 1))
    s[1] = np.array([0x3800, 0x0000, 0x0000, 0x0000], np.uint16)
    run_half(x, [0, 0], [0, 0], s, 0, "f16")


# ---- fp8 ------------------------------------------------------------------------------------

@pytest.mark.parametrize("t", list(HALF))
@pytest.mark.parametrize("shape", range(len(SHAPES)))
@pytest.mark.parametrize("dq,oq", [(F32(0.1), F32(1.0 / 0.6)), (F32(0.05), F32(20.0)),
                                   (F32(3.0), F32(37.5)), (F32(1e-30), F32(1e30)),
                                   (F32(2.0 ** -12), F32(0.6))])
def test_bilinear_fp8(t, shape, dq, oq):
    Hi, Wi, Ho, Wo, mode = SHAPES[shape]
    (ih, wy0, wy1), (iw, wx0, wx1) = axes(Hi, Wi, Ho, Wo, mode)
    tt = HALF[t][1]
    s = gr.float_table(wy0, wy1, wx0, wx1, tt)
    if shape % 2:
        s = random_half_weights(Ho, Wo, tt)
    C = 32
    codes = fp8_values(Hi * Wi * C).reshape(Hi, Wi, C)
    got = call("npu_resize_bilinear_fp8", codes, OUT, HALF[t][0], I64(Hi), I64(Wi), I64(C),
               I64(Ho), I64(Wo), np.asarray(ih, np.int32), np.asarray(iw, np.int32), s,
               I64(4), float(dq), float(oq), out_nbytes=Ho * Wo * C, out_dtype=np.uint8)
    same(got, gr.bilinear_fp8(codes, ih, iw, s, 4, tt, dq, oq), t)


# ---- bicubic ---------------------------------------------------------------------------------

CUBIC = [(8, 8, 16, 16, "half_pixel"), (12, 20, 5, 7, "half_pixel"),
         (6, 10, 18, 24, "asymmetric"), (8, 9, 4, 48, "align_corners"), (4, 5, 9, 4, "half_pixel")]


def run_cubic(x, ih, iw, cy, cx):
    Hi, Wi, C = x.shape
    Ho, Wo = len(ih), len(iw)
    got = call("npu_resize_cubic_bf16", x, OUT, I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo),
               np.asarray(ih, np.int32), np.asarray(iw, np.int32), cy, cx,
               out_nbytes=Ho * Wo * C * 2, out_dtype=np.uint16)
    same(got, gr.cubic_bf16(x, ih, iw, cy, cx), "cubic")


@pytest.mark.parametrize("shape", range(len(CUBIC)))
@pytest.mark.parametrize("special", [False, True])
def test_cubic_tables(shape, special):
    Hi, Wi, Ho, Wo, mode = CUBIC[shape]
    ih, cy = gr.cubic_axis(Hi, Ho, mode)
    iw, cx = gr.cubic_axis(Wi, Wo, mode)
    run_cubic(half_input(ga.BF16, (Hi, Wi, 32), special), ih, iw, cy, cx)


def test_cubic_random():
    Hi, Wi, Ho, Wo = 9, 11, 7, 13
    ih = RNG.integers(0, Hi - 3, Ho)
    iw = RNG.integers(0, Wi - 3, Wo)
    w = lambda n: random_half_weights(n, 1, ga.BF16).reshape(n, 4)
    run_cubic(half_input(ga.BF16, (Hi, Wi, 16)), ih, iw, w(Ho), w(Wo))


# ---- argument checks ---------------------------------------------------------------------------

def test_bad_args():
    assert raw_rc("npu_resize_bilinear", None, None, NPU_F32, I64(2), I64(2), I64(1), I64(1),
                  I64(1), None, None, None, 0, I64(0), 0, 1.0, 0.0) == -1
    assert raw_rc("npu_resize_bilinear", None, None, NPU_I16, I64(2), I64(2), I64(1), I64(1),
                  I64(1), None, None, None, gr.SEP, I64(0), 0, 1.0, 0.0) == -1
    assert raw_rc("npu_resize_bilinear", None, None, NPU_I8, I64(2), I64(2), I64(1), I64(1),
                  I64(1), None, None, None, 2, I64(0), 0, 1.0, 0.0) == -1
    assert raw_rc("npu_resize_bilinear", None, None, NPU_I8, I64(0), I64(2), I64(1), I64(1),
                  I64(1), None, None, None, 0, I64(0), 0, 1.0, 0.0) == -1
    assert raw_rc("npu_resize_bilinear_bf16", None, None, NPU_FP8, I64(2), I64(2), I64(1),
                  I64(1), I64(1), None, None, None, I64(0)) == -1
    assert raw_rc("npu_resize_bilinear_fp8", None, None, NPU_BF16, I64(2), I64(2), I64(12),
                  I64(1), I64(1), None, None, None, I64(0), 1.0, 1.0) == -1
    assert raw_rc("npu_resize_bilinear_fp8", None, None, NPU_FP8, I64(2), I64(2), I64(8),
                  I64(1), I64(1), None, None, None, I64(0), 1.0, 1.0) == -1
    assert raw_rc("npu_resize_cubic_bf16", None, None, I64(0), I64(2), I64(1), I64(1), I64(1),
                  None, None, None, None) == -1
    # empty output: nothing launched
    assert raw_rc("npu_resize_cubic_bf16", None, None, I64(4), I64(4), I64(1), I64(0), I64(0),
                  None, None, None, None) == 0
