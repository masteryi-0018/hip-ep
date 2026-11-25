# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""resize.h entry points against the numpy reference model, bit for bit."""

import numpy as np
import pytest

import golden_resize as gr
from golden_float_cvt import store_float
from npu import (I64, NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT,
                 call, raw_rc)
import testing
from testing import HALF, INT, same

RNG = np.random.default_rng(2031)
F32 = np.float32
# (Hi, Wi, Ho, Wo, coordinate mode)
SHAPES = [(5, 7, 13, 16, "half_pixel"), (12, 11, 5, 4, "half_pixel"),
          (6, 9, 12, 6, "align_corners"), (4, 8, 12, 24, "asymmetric"),
          (7, 5, 7, 12, "half_pixel")]


# Special 16-bit codes mixed into the inputs.
SPECIAL = {t: np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x3C00 if f16 else 0x3F80,
                        0xBC00 if f16 else 0xBF80, 0x7C00 if f16 else 0x7F80,
                        0xFC00 if f16 else 0xFF80, 0x7E00 if f16 else 0x7FC0,
                        0x7BFF if f16 else 0x7F7F, 0xFBFF if f16 else 0xFF7F,
                        0x0400 if f16 else 0x0080, 0x03FF if f16 else 0x007F], np.uint16)
           for t, f16 in (("bf16", False), ("f16", True))}


def fp8_values(n, nan_rate=0.01):
    return testing.fp8_values(RNG, n, nan_rate)


def axes(Hi, Wi, Ho, Wo, mode):
    return gr.bilinear_axis(Hi, Ho, mode), gr.bilinear_axis(Wi, Wo, mode)


def random_q15(Ho, Wo):
    """Random 4-tap Q15 weights summing to at most 2^15, with zeros injected."""
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
    a16 = np.dtype(dt).itemsize == 2  # the 16-bit kernel is 4-tap only and takes no route
    got = call("npu_resize_bilinear_a16" if a16 else "npu_resize_bilinear", x, nt,
               np.asarray(ih, np.uint16), np.asarray(iw, np.uint16), np.asarray(wts, np.uint16),
               OUT, I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo), *(() if a16 else (route,)),
               I64(sv_wo), req, float(c0), float(c1),
               out_nbytes=Ho * Wo * C * np.dtype(dt).itemsize, out_dtype=dt)
    if a16:
        want = gr.resize_bilinear_a16(x, ih, iw, wts, sv_wo, req, c0, c1, dt)
    else:
        want = gr.resize_bilinear(x, ih, iw, wts, route, sv_wo, req, c0, c1, dt)
    same(got, want, f"{t} route={route} req={req}")


# (requantize, c0, c1)
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


def half_input(tt, shape, special=True):
    n = int(np.prod(shape))
    return testing.half_values(RNG, tt, n, SPECIAL[tt] if special else None).reshape(shape)


def random_half_weights(Ho, Wo, tt):
    """Random weights with zeros of both signs and a few subnormals."""
    w = (RNG.standard_normal((Ho, Wo, 4)) * np.exp2(RNG.integers(-3, 3, (Ho, Wo, 4)))).astype(F32)
    bits = store_float(w, tt).astype(np.uint16)
    z = RNG.random((Ho, Wo, 4))
    bits = np.where(z < 0.15, 0, bits)
    bits = np.where((z >= 0.15) & (z < 0.2), 0x8000, bits)
    bits = np.where((z >= 0.2) & (z < 0.21), 0x0001, bits)
    return bits.astype(np.uint16)


def run_half(xbits, ih, iw, sbits, sv_wo, t):
    Hi, Wi, C = xbits.shape
    Ho, Wo = len(ih), len(iw)
    nt, tt = HALF[t]
    got = call("npu_resize_bilinear_bf16", xbits, nt, np.asarray(ih, np.uint16),
               np.asarray(iw, np.uint16), np.asarray(sbits, np.uint16), OUT, I64(Hi), I64(Wi),
               I64(C), I64(Ho), I64(Wo), I64(sv_wo), out_nbytes=Ho * Wo * C * 2,
               out_dtype=np.uint16)
    same(got, gr.resize_bilinear_bf16(xbits, ih, iw, sbits, sv_wo, tt), t)


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
    Hi, Wi, Ho, Wo = 2, 2, 2, 2
    x = np.array([0x7BFF, 0x7BFF, 0x0400, 0x8400] * 2, np.uint16).reshape(Hi, Wi, 2)
    s = np.tile(np.array([0x3C00, 0x3C00, 0x0000, 0x0000], np.uint16), (Ho, Wo, 1))
    s[1] = np.array([0x3800, 0x0000, 0x0000, 0x0000], np.uint16)
    run_half(x, [0, 0], [0, 0], s, 0, "f16")


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
    got = call("npu_resize_bilinear_fp8", codes, np.asarray(ih, np.uint16),
               np.asarray(iw, np.uint16), s, HALF[t][0], OUT, I64(Hi), I64(Wi), I64(C),
               I64(Ho), I64(Wo), I64(4), float(dq), float(oq), out_nbytes=Ho * Wo * C,
               out_dtype=np.uint8)
    same(got, gr.resize_bilinear_fp8(codes, ih, iw, s, 4, tt, dq, oq), t)


CUBIC = [(8, 8, 16, 16, "half_pixel"), (12, 20, 5, 7, "half_pixel"),
         (6, 10, 18, 24, "asymmetric"), (8, 9, 4, 48, "align_corners"), (4, 5, 9, 4, "half_pixel")]


def run_cubic(x, ih, iw, cy, cx):
    Hi, Wi, C = x.shape
    Ho, Wo = len(ih), len(iw)
    got = call("npu_resize_cubic_bf16", x, np.asarray(ih, np.uint16), np.asarray(iw, np.uint16),
               cy, cx, OUT, I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo),
               out_nbytes=Ho * Wo * C * 2, out_dtype=np.uint16)
    same(got, gr.resize_cubic_bf16(x, ih, iw, cy, cx), "cubic")


@pytest.mark.parametrize("shape", range(len(CUBIC)))
@pytest.mark.parametrize("special", [False, True])
def test_cubic_tables(shape, special):
    Hi, Wi, Ho, Wo, mode = CUBIC[shape]
    ih, cy = gr.cubic_axis(Hi, Ho, mode)
    iw, cx = gr.cubic_axis(Wi, Wo, mode)
    run_cubic(half_input("bf16", (Hi, Wi, 32), special), ih, iw, cy, cx)


def test_cubic_random():
    Hi, Wi, Ho, Wo = 9, 11, 7, 13
    ih = RNG.integers(0, Hi - 3, Ho)
    iw = RNG.integers(0, Wi - 3, Wo)
    def w(n):
        return random_half_weights(n, 1, "bf16").reshape(n, 4)
    run_cubic(half_input("bf16", (Hi, Wi, 16)), ih, iw, w(Ho), w(Wo))


def test_bad_args():
    z = np.zeros(64, np.uint16)
    tail = (I64(2), I64(2), I64(8), I64(1), I64(1))
    # npu_resize_bilinear: in, in_type, idx_h, idx_w, wts, out, sizes, route, sv_wo, req, c0, c1
    def bil(t, *, x=z, route=0, sv=0, sizes=tail):
        return raw_rc("npu_resize_bilinear", x, t, z, z, z, z, *sizes, route, I64(sv), 0, 1.0,
                      0.0)
    assert bil(NPU_I8) == 0
    assert bil(NPU_F32) == -1
    for route in (0, gr.SEP):  # 16 bits belong to npu_resize_bilinear_a16
        assert bil(NPU_I16, route=route) == -1
    assert bil(NPU_I8, route=2) == -1
    assert bil(NPU_I8, sv=-1) == -1
    assert bil(NPU_I8, x=None) == -1
    assert bil(NPU_I8, sizes=(I64(0), I64(2), I64(8), I64(1), I64(1))) == -1
    assert bil(NPU_I8, sizes=(I64(2), I64(2), I64(0), I64(1), I64(1))) == 0  # C == 0: empty
    assert bil(NPU_I8, sizes=(I64(2), I64(2), I64(-1), I64(1), I64(1))) == -1
    assert bil(NPU_I8, sizes=(I64(2), I64(2), I64(8), I64(-1), I64(1))) == -1
    assert bil(NPU_I8, sizes=(I64(2), I64(2), I64(8), I64(1), I64(-1))) == -1
    # empty output: valid, nothing launched; the pointers stay required
    empty = (I64(2), I64(2), I64(8), I64(0), I64(1))
    assert bil(NPU_I8, sizes=empty) == 0
    assert bil(NPU_U8, route=gr.SEP, sizes=empty) == 0
    assert bil(NPU_I8, sizes=empty, x=None) == -1
    assert bil(NPU_F32, sizes=empty) == -1
    assert bil(NPU_I8, sizes=(I64(65536), I64(2), I64(8), I64(1), I64(1))) == -1
    def a16(t, *, w=z):
        return raw_rc("npu_resize_bilinear_a16", z, t, z, z, w, z, *tail, I64(0),
                      0, 1.0, 0.0)
    assert a16(NPU_U16) == 0
    assert a16(NPU_I8) == -1
    assert a16(NPU_I16, w=None) == -1
    assert raw_rc("npu_resize_bilinear_a16", z, NPU_I16, z, z, z, z, I64(2), I64(2), I64(8),
                  I64(1), I64(0), I64(0), 0, 1.0, 0.0) == 0
    def bf(t, *, ih=z):
        return raw_rc("npu_resize_bilinear_bf16", z, t, ih, z, z, z, *tail, I64(0))
    assert bf(NPU_F16) == 0
    assert bf(NPU_FP8) == -1
    assert bf(NPU_BF16, ih=None) == -1
    assert raw_rc("npu_resize_bilinear_bf16", z, NPU_BF16, z, z, z, z, I64(2), I64(2), I64(8),
                  I64(0), I64(0), I64(0)) == 0
    def fp8(t, *, C=8, o=z):
        return raw_rc("npu_resize_bilinear_fp8", z, z, z, z, t, o, I64(2),
                      I64(2), I64(C), I64(1), I64(1), I64(0), 1.0, 1.0)
    assert fp8(NPU_BF16) == 0
    assert fp8(NPU_BF16, C=12) == -1
    assert fp8(NPU_FP8) == -1
    assert fp8(NPU_F16, o=None) == -1
    assert raw_rc("npu_resize_bilinear_fp8", z, z, z, z, NPU_BF16, z, I64(2), I64(2), I64(8),
                  I64(0), I64(3), I64(0), 1.0, 1.0) == 0
    def cub(*, Hi=4, cx=z, Ho=0):
        return raw_rc("npu_resize_cubic_bf16", z, z, z, z, cx, z, I64(Hi),
                      I64(4), I64(1), I64(Ho), I64(0))
    assert cub() == 0  # empty output: nothing launched
    assert cub(Hi=0) == -1
    assert cub(cx=None) == -1
    assert cub(Ho=-1) == -1


# ---- special values, bit for bit -----------------------------------------------------------

SP16 = {t: np.concatenate([SPECIAL[t], np.array([0x7F81 if t == "bf16" else 0x7C01,
                                                 0xFFBF if t == "bf16" else 0xFDFF,
                                                 0x7FFF, 0xFFFF], np.uint16)]) for t in SPECIAL}
SP8 = np.array([0x00, 0x80, 0x01, 0x81, 0x38, 0xB8, 0x7E, 0xFE, 0x7F, 0xFF, 0x08, 0x77], np.uint8)
IDX_EDGE = np.array([0, 1, 65535, 65534, 32768, 1000, 2, 3], np.uint16)
COEF_SPECIAL = [(1, 0.0), (0, 0.0), (1, 1e30), (1, -1e30), (1, np.inf), (1, -np.inf),
                (1, np.nan), (np.inf, 0.0), (np.nan, 0.0), (-0.0, -0.0), (1e30, 1e30),
                (1e-45, -1e-45), (-1.0, 0.5)]


def edge_idx(n, hi):
    """n indices drawn from the in-range and extreme uint16 values."""
    return np.concatenate([IDX_EDGE, RNG.integers(0, hi, n)])[:n].astype(np.uint16)


@pytest.mark.parametrize("t,route", [(t, gr.TAP4) for t in INT] + [("i8", gr.SEP), ("u8", gr.SEP)])
@pytest.mark.parametrize("ci", range(len(COEF_SPECIAL)))
@pytest.mark.parametrize("sv_wo", [0, 3])
def test_bilinear_int_special(t, route, ci, sv_wo):
    """Extreme indices (clamped), saturated weights, degenerate requantize constants."""
    Hi, Wi, Ho, Wo = 4, 5, 9, 11
    info = np.iinfo(INT[t][1])
    x = RNG.choice(np.array([info.min, info.max, 0, 1], INT[t][1]), (Hi, Wi, 8))
    ih, iw = edge_idx(Ho, Hi), edge_idx(Wo, Wi)
    if route == gr.SEP:
        wts = RNG.choice(np.array([0, 1, 65535, 129, 350, 32768], np.uint16),
                         2 * Ho + 2 * Wo)
    else:
        wts = RNG.choice(np.array([0, 1, 65535, 32768, 32767], np.uint16), Ho * Wo * 4)
    c0, c1 = COEF_SPECIAL[ci]
    for req in (0, 1):
        run_int(x, ih, iw, wts, t, route, sv_wo, req, c0, c1)


@pytest.mark.parametrize("t", list(HALF))
@pytest.mark.parametrize("sv_wo", [0, 3])
def test_bilinear_bf16_special(t, sv_wo):
    """Every pixel and weight a special code; extreme indices."""
    Hi, Wi, Ho, Wo = 5, 6, 12, 13
    tt = HALF[t][1]
    x = RNG.choice(SP16[tt], (Hi, Wi, 24))
    w = RNG.choice(SP16[tt], (Ho, Wo, 4)).astype(np.uint16)
    run_half(x, edge_idx(Ho, Hi), edge_idx(Wo, Wi), w, sv_wo, t)


@pytest.mark.parametrize("t", list(HALF))
@pytest.mark.parametrize("coef", [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (np.inf, 1.0),
                                  (1.0, np.inf), (np.nan, 1.0), (1.0, np.nan), (-0.0, -0.0),
                                  (1e30, 1e30), (1e-45, 1e-45), (1e30, 1e-30)])
def test_bilinear_fp8_special(t, coef):
    Hi, Wi, Ho, Wo, C = 5, 6, 12, 13, 16
    tt = HALF[t][1]
    x = RNG.choice(SP8, (Hi, Wi, C))
    x = np.where(RNG.random((Hi, Wi, C)) < 0.3, RNG.integers(0, 256, (Hi, Wi, C)),
                 x).astype(np.uint8)
    s = RNG.choice(SP16[tt], (Ho, Wo, 4)).astype(np.uint16)
    ih, iw = edge_idx(Ho, Hi), edge_idx(Wo, Wi)
    dq, oq = coef
    got = call("npu_resize_bilinear_fp8", x, ih, iw, s, HALF[t][0], OUT, I64(Hi), I64(Wi),
               I64(C), I64(Ho), I64(Wo), I64(4), float(dq), float(oq), out_nbytes=Ho * Wo * C,
               out_dtype=np.uint8)
    same(got, gr.resize_bilinear_fp8(x, ih, iw, s, 4, tt, F32(dq), F32(oq)), t)


def test_cubic_special():
    Hi, Wi, Ho, Wo = 6, 7, 9, 10
    x = RNG.choice(SP16["bf16"], (Hi, Wi, 8))
    cy = RNG.choice(SP16["bf16"], (Ho, 4)).astype(np.uint16)
    cx = RNG.choice(SP16["bf16"], (Wo, 4)).astype(np.uint16)
    run_cubic(x, edge_idx(Ho, Hi), edge_idx(Wo, Wi), cy, cx)
