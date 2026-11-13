# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The gridsample-group entry points (include/npu_emu_gridsample.h) against
the numpy reference in golden_gridsample.py, bit for bit. Grids mix random
coordinates with the edges of every mask and clamp (integers, -0, -1, W - 1,
W, past the int16 range, inf, NaN); shapes include a width bfloat16 cannot
hold (the RTP pad bound rounds) and H * W = 32768 (the int16 neighbour
offsets wrap); scales are not powers of two."""

import ctypes

import numpy as np
import pytest

import golden as g
import golden_gridsample as gs
from npu import (I64, NPU_BF16, NPU_F16, NPU_I8, NPU_I16, NPU_U8, NPU_U16, NPU_FP8, OUT, Host,
                 raw_rc)
from npu import call as callx

RNG = np.random.default_rng(2031)
f32 = np.float32
FT = {"bf16": NPU_BF16, "f16": NPU_F16}
IT = {"i8": (NPU_I8, np.int8), "u8": (NPU_U8, np.uint8), "i16": (NPU_I16, np.int16),
      "u16": (NPU_U16, np.uint16)}

# (H, W, C, Ho, Wo)
SHAPES = [(5, 7, 2, 6, 9), (9, 13, 4, 7, 11), (3, 257, 2, 8, 8), (128, 256, 2, 4, 8),
          (1, 1, 2, 3, 3), (2, 3, 6, 5, 4)]


class GridQdq(ctypes.Structure):
    _fields_ = [("grid_zp", ctypes.c_int32), ("grid_scale", ctypes.c_float),
                ("ifm_zp", ctypes.c_int32), ("ifm_scale", ctypes.c_float),
                ("ofm_zp", ctypes.c_int32), ("ofm_scale", ctypes.c_float)]


def ff(x):
    return float(f32(x))


def bits(x, t):
    return g.store_float(np.asarray(x, f32), t)


def assert_same(got, want):
    got = np.asarray(got).reshape(-1)
    want = np.asarray(want).astype(got.dtype).reshape(-1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{got.size} differ, first at {bad[:6].tolist()}: "
                           f"got {got[bad[:6]].tolist()} want {want[bad[:6]].tolist()}")


def coords(n, extent, specials):
    """n coordinates along an axis of `extent` pixels (fp32)."""
    k = n // 3
    a = RNG.uniform(-2.5, extent + 1.5, n)
    a[:k] = RNG.integers(-2, extent + 2, k) + RNG.choice([0.0, 0.5, 0.25, 0.75], k)
    e = np.array([0.0, -0.0, -1.0, -1.0 - 2.0 ** -6, -2.0 ** -9, 0.5, extent - 1.0,
                  extent - 1.0 + 2.0 ** -7, extent, extent + 0.5, extent - 0.5, -0.5,
                  40000.0, -40000.0, 1e9, np.inf, -np.inf] + ([np.nan] if specials else []))
    m = min(len(e), n)
    pos = RNG.choice(n, m, replace=False)
    a[pos] = e[:m]
    return a.astype(f32)


def float_grid(H, W, Ho, Wo, t, specials):
    """[Ho][Wo][2] grid bits of type t; the last rows target the bottom-right
    corner, where the clamped indices and int16 offsets matter most."""
    n = Ho * Wo
    gx, gy = coords(n, W, specials), coords(n, H, specials)
    r = n // 4
    gx[-r:] = RNG.uniform(W - 2, W + 1, r)
    gy[-r:] = RNG.uniform(H - 2, H + 1, r)
    return bits(np.stack([gx, gy], -1).reshape(Ho, Wo, 2), t)


def float_ifm(shape, t, nan):
    x = RNG.normal(0, 3, shape).astype(f32)
    flat = x.reshape(-1)
    big = 3e38 if t == "bf16" else 60000.0
    e = [0.0, -0.0, 1e-40 if t == "bf16" else 3e-6, -2.0 ** -20, big, -big, 1.0, -1.0]
    pos = RNG.choice(flat.size, min(len(e), flat.size), replace=False)
    flat[pos] = e[:len(pos)]
    if nan:
        flat[RNG.integers(0, flat.size)] = np.nan
        # pixel 0 is where clamped (and int16-wrapped) out-of-image neighbours
        # read: inf * +0 makes their zero coefficient visible
        flat[:shape[-1]] = np.inf
    return bits(x, t)


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("si", range(len(SHAPES)))
@pytest.mark.parametrize("specials", [False, True])
def test_gridsample_bf16(t, si, specials):
    H, W, C, Ho, Wo = SHAPES[si]
    x = float_ifm((H, W, C), t, specials)
    grid = float_grid(H, W, Ho, Wo, t, specials)
    got = callx("npu_gridsample_bf16", x, FT[t], grid, OUT, I64(H), I64(W), I64(C), I64(Ho),
                I64(Wo), out_nbytes=Ho * Wo * C * 2, out_dtype=np.uint16)
    assert_same(got, gs.gridsample_bf16(x, grid, t))


# (ifm_scale, ofm_recip): powers of two, ties-heavy ratios, saturation past 448.
FP8_SCALES = [(0.03125, 32.0), (0.1, ff(1 / 0.6)), (0.6, ff(1 / 0.1)), (ff(3.7), 0.01),
              (40.0, 1.0), (ff(1 / 3), 3.0)]


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("si", range(len(SHAPES)))
@pytest.mark.parametrize("ci", range(len(FP8_SCALES)))
def test_gridsample_fp8(t, si, ci):
    H, W, C, Ho, Wo = SHAPES[si]
    s_in, recip = FP8_SCALES[ci]
    x = RNG.integers(0, 256, (H, W, C)).astype(np.uint8)
    nan = ci == 1
    if not nan:
        x[(x & 0x7F) == 0x7F] = 0x80  # keep NaN to one case: it spreads over groups of 8
    else:
        x.reshape(-1)[RNG.integers(0, x.size)] = 0x7F
    e = [0x80, 0x00, 0x7E, 0xFE]  # -0, +0, +-448
    x.reshape(-1)[:min(4, x.size)] = e[:x.size]
    grid = float_grid(H, W, Ho, Wo, t, nan)
    got = callx("npu_gridsample_fp8", x, grid, FT[t], OUT, I64(H), I64(W), I64(C), I64(Ho),
                I64(Wo), s_in, recip, out_nbytes=Ho * Wo * C, out_dtype=np.uint8)
    assert_same(got, gs.gridsample_fp8(x, grid, t, s_in, recip))


# (grid_zp, grid_scale, ifm_zp, ifm_scale, ofm_zp, ofm_mult)
GRIDQ = [(28526, ff(0.00010520511568756774), 3, ff(0.00014300664770416915), 5,
          ff(1 / 0.0001342372)),
         (128, ff(1 / 127), 128, ff(0.0123), 128, ff(1 / 0.0151)),
         (0, ff(1 / 2000), -7, 0.5, 2, 1.0),
         (32768, ff(1 / 30000), 31633, ff(0.1), 33234, ff(1 / 0.6)),
         (-5, ff(0.6), 0, ff(0.6), -3, ff(1 / 0.1))]


def qdq_grid(q, Ho, Wo):
    grid = RNG.integers(0, 65536, (Ho, Wo, 2)).astype(np.uint16)
    span = max(int(1.3 / q[1]), 2)  # codes near the dequantized range [-1.3, 1.3]
    k = (Ho * 3) // 4
    grid[:k] = np.clip(q[0] + RNG.integers(-span, span, (k, Wo, 2)), 0, 65535)
    grid.reshape(-1)[:4] = [0, 65535, np.clip(q[0], 0, 65535), np.clip(q[0] - 1, 0, 65535)]
    return grid


def int_ifm(t, shape):
    info = np.iinfo(IT[t][1])
    x = RNG.integers(info.min, info.max, shape, endpoint=True).astype(IT[t][1])
    x.reshape(-1)[:2] = [info.min, info.max]
    return x


@pytest.mark.parametrize("variant", ["a8", "a16_bf16", "a16_f16"])
@pytest.mark.parametrize("align", [0, 1])
@pytest.mark.parametrize("si", range(len(SHAPES)))
@pytest.mark.parametrize("qi", range(len(GRIDQ)))
def test_qdq_gridsample(variant, align, si, qi):
    H, W, C, Ho, Wo = SHAPES[si]
    q = GRIDQ[qi]
    for t in (["i8", "u8"] if variant == "a8" else ["i16", "u16"]):
        x = int_ifm(t, (H, W, C))
        grid = qdq_grid(q, Ho, Wo)
        cq = GridQdq(*q)
        nt, dt = IT[t]
        if variant == "a8":
            name, tail, ct = "npu_qdq_gridsample_a8", (align,), "bf16"
        else:
            ct = variant[4:]
            name, tail = "npu_qdq_gridsample_a16", (align, FT[ct])
        got = callx(name, x, nt, grid, OUT, I64(H), I64(W), I64(C), I64(Ho), I64(Wo), *tail,
                    Host(ctypes.byref(cq)), out_nbytes=Ho * Wo * C * np.dtype(dt).itemsize,
                    out_dtype=dt)
        assert_same(got, gs.qdq_gridsample(x, grid, align, q, ct, t))


def test_bad_arguments():
    z = None
    cq = GridQdq(0, 1.0, 0, 1.0, 0, 1.0)
    shp = lambda H, W, C: (I64(H), I64(W), I64(C), I64(1), I64(1))  # noqa: E731
    # odd C, H * W past the int16 index range, wrong types
    assert raw_rc("npu_gridsample_bf16", z, NPU_BF16, z, z, *shp(2, 2, 3)) == -1
    assert raw_rc("npu_gridsample_bf16", z, NPU_BF16, z, z, *shp(129, 256, 2)) == -1
    assert raw_rc("npu_gridsample_bf16", z, NPU_FP8, z, z, *shp(2, 2, 2)) == -1
    assert raw_rc("npu_gridsample_fp8", z, z, NPU_I16, z, *shp(2, 2, 2), 1.0, 1.0) == -1
    assert raw_rc("npu_qdq_gridsample_a8", z, NPU_I16, z, z, *shp(2, 2, 2), 0,
                  Host(ctypes.byref(cq))) == -1
    assert raw_rc("npu_qdq_gridsample_a8", z, NPU_I8, z, z, *shp(2, 2, 2), 2,
                  Host(ctypes.byref(cq))) == -1
    assert raw_rc("npu_qdq_gridsample_a8", z, NPU_I8, z, z, *shp(2, 2, 2), 0, z) == -1
    assert raw_rc("npu_qdq_gridsample_a16", z, NPU_U16, z, z, *shp(2, 2, 2), 0, NPU_FP8,
                  Host(ctypes.byref(cq))) == -1
    assert raw_rc("npu_qdq_gridsample_a16", z, NPU_U8, z, z, *shp(2, 2, 2), 0, NPU_BF16,
                  Host(ctypes.byref(cq))) == -1
