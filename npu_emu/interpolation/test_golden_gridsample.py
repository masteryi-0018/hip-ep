# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""gridsample.h entry points against the numpy reference model, bit for bit."""

import ctypes

import numpy as np
import pytest

import golden_gridsample as gs
from golden_float_cvt import store_float
from npu import (I64, NPU_BF16, NPU_F16, NPU_FP8, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT, Host,
                 raw_rc)
from npu import call as callx
from npu import f32 as ff
from testing import F16T as FT
from testing import INT as IT
from testing import assert_same

RNG = np.random.default_rng(2031)
f32 = np.float32

# (H, W, C, Ho, Wo)
SHAPES = [(5, 7, 2, 6, 9), (9, 13, 4, 7, 11), (3, 257, 2, 8, 8), (128, 256, 2, 4, 8),
          (1, 1, 2, 3, 3), (2, 3, 6, 5, 4)]


class GridQdq(ctypes.Structure):
    _fields_ = [("grid_zp", ctypes.c_int32), ("grid_scale", ctypes.c_float),
                ("ifm_zp", ctypes.c_int32), ("ifm_scale", ctypes.c_float),
                ("ofm_zp", ctypes.c_int32), ("ofm_scale", ctypes.c_float)]


def bits(x, t):
    return store_float(np.asarray(x, f32), t)


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
    """[Ho][Wo][2] grid bits of type t; the last rows target the bottom-right corner."""
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
        # out-of-image neighbours read pixel 0 with a zero coefficient
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


# (ifm_scale, ofm_scale = 1 / s_out)
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
        x[(x & 0x7F) == 0x7F] = 0x80  # NaN in one case only
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
        if variant == "a8":
            want = gs.qdq_gridsample_a8(x, grid, align, q, t)
        else:
            want = gs.qdq_gridsample_a16(x, grid, align, q, ct, t)
        assert_same(got, want)


# ---- special values, bit for bit -------------------------------------------------------

SP16 = {"bf16": [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F,
                 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFBF, 0x4020, 0xC020, 0x3F00],
        "f16": [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF,
                0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0xFDFF, 0x4100, 0xC100, 0x3800]}
SP8 = np.array([0x00, 0x80, 0x01, 0x81, 0x38, 0xB8, 0x7E, 0xFE, 0x7F, 0xFF, 0x08, 0x77],
               np.uint8)


def special_grid(t, Ho, Wo):
    """[Ho][Wo][2] grid of special coordinate codes: every (gx, gy) pair of the table, cycled."""
    sp = np.array(SP16[t], np.uint16)
    n = Ho * Wo
    k = np.arange(n)
    gx, gy = sp[k % len(sp)], sp[(k // len(sp) + k) % len(sp)]
    return np.stack([gx, gy], -1).reshape(Ho, Wo, 2)


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("si", range(len(SHAPES)))
def test_gridsample_special_bits(t, si):
    """Specials (NaN with payload and sign, inf, subnormals, +-0) in pixels and coordinates."""
    H, W, C, _, _ = SHAPES[si]
    Ho, Wo = 19, 19
    x = RNG.choice(np.array(SP16[t], np.uint16), (H, W, C))
    x = np.where(RNG.random((H, W, C)) < 0.4, bits(RNG.normal(0, 2, (H, W, C)), t), x)
    grid = special_grid(t, Ho, Wo)
    got = callx("npu_gridsample_bf16", x, FT[t], grid, OUT, I64(H), I64(W), I64(C), I64(Ho),
                I64(Wo), out_nbytes=Ho * Wo * C * 2, out_dtype=np.uint16)
    assert_same(got, gs.gridsample_bf16(x, grid, t))


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (np.inf, 1.0),
                                    (1.0, np.inf), (np.nan, 1.0), (1.0, np.nan), (-0.0, -0.0),
                                    (1e30, 1e30), (1e-45, 1e-45)])
def test_gridsample_fp8_special(t, scales):
    H, W, C, _, _ = SHAPES[0]
    Ho, Wo = 19, 19
    x = RNG.choice(SP8, (H, W, C))
    x = np.where(RNG.random((H, W, C)) < 0.4, RNG.integers(0, 256, (H, W, C)), x).astype(np.uint8)
    grid = special_grid(t, Ho, Wo)
    got = callx("npu_gridsample_fp8", x, grid, FT[t], OUT, I64(H), I64(W), I64(C), I64(Ho),
                I64(Wo), float(scales[0]), float(scales[1]), out_nbytes=Ho * Wo * C,
                out_dtype=np.uint8)
    assert_same(got, gs.gridsample_fp8(x, grid, t, *scales))


# (grid_zp, grid_scale, ifm_zp, ifm_scale, ofm_zp, ofm_mult)
GRIDQ_SPECIAL = [(0, 0.0, 0, 1.0, 0, 1.0), (0, np.inf, 0, 1.0, 0, 1.0),
                 (0, np.nan, 0, 1.0, 0, 1.0), (0, ff(1 / 100), -2**31, 1.0, 0, 1.0),
                 (65535, ff(1 / 100), 2**31 - 1, 1e30, 0, 1.0),
                 (-2**31, 1.0, 0, np.inf, 0, 1.0), (2**31 - 1, 1.0, 0, np.nan, 0, 1.0),
                 (0, ff(1 / 100), 0, 1.0, 2**31 - 1, np.inf),
                 (0, ff(1 / 100), 0, 1.0, -2**31, np.nan), (0, ff(1 / 100), 0, 1.0, 0, 0.0),
                 (0, ff(1 / 100), 0, 0.0, 0, 1e30), (100, -ff(1 / 100), 0, 1.0, 0, -1.0)]


@pytest.mark.parametrize("variant", ["a8", "a16_bf16", "a16_f16"])
@pytest.mark.parametrize("align", [0, 1])
@pytest.mark.parametrize("qi", range(len(GRIDQ_SPECIAL)))
def test_qdq_gridsample_special(variant, align, qi):
    """Degenerate coefficients and integer extremes in codes, zero points and pixels."""
    H, W, C, Ho, Wo = SHAPES[0]
    q = GRIDQ_SPECIAL[qi]
    for t in (["i8", "u8"] if variant == "a8" else ["i16", "u16"]):
        info = np.iinfo(IT[t][1])
        x = RNG.choice(np.array([info.min, info.max, 0, 1, -1 if info.min else 1], IT[t][1]),
                       (H, W, C))
        grid = RNG.choice(np.array([0, 1, 65535, 65534, 32768, 32767, 100, 99, 101], np.uint16),
                          (Ho, Wo, 2))
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
        if variant == "a8":
            want = gs.qdq_gridsample_a8(x, grid, align, q, t)
        else:
            want = gs.qdq_gridsample_a16(x, grid, align, q, ct, t)
        assert_same(got, want)


# ---- grid from an int8 flow, fused with its ReduceL2 ----------------------------------

class GridFlow(ctypes.Structure):
    _fields_ = [(k, ctypes.c_int32) for k in gs.FLOW_FIELDS]


def flow_params(rng, H, W, kind):
    """Random npu_gridsample_flow fields; kind 0: the closed forms of a power-of-two grid
    scale (realistic coordinates), 1: arbitrary values in range."""
    if kind == 0:
        t = int(rng.integers(5, 9))
        ka = int(rng.integers(0, 3))
        ac = int(rng.integers(0, 2))
        p = dict(ka=ka, shift=t + 1, ax=W - ac, ay=H - ac,
                 bx=(1 << t) * (W - 1), by=(1 << t) * (H - 1))
        for ax in "xy":
            r = rng.uniform(0.5, 3.0) / (1 << ka)
            msh = 14 - int(np.ceil(np.log2(r)))
            p["mq_" + ax], p["msh_" + ax] = int(np.floor(r * 2.0 ** msh + 0.5)), msh
    else:
        p = dict(ka=int(rng.integers(0, 8)), shift=int(rng.integers(0, 16)),
                 ax=int(rng.integers(-32768, 32768)), ay=int(rng.integers(-32768, 32768)),
                 bx=int(rng.integers(-2**22, 2**22)), by=int(rng.integers(-2**22, 2**22)))
        for ax in "xy":
            p["mq_" + ax] = int(rng.integers(-32768, 32768))
            p["msh_" + ax] = int(rng.integers(0, 20))
    p.update(norm_mode=int(rng.integers(0, 3)), n_mq=int(rng.integers(1, 1 << 14)),
             n_msh=int(rng.integers(0, 16)), out_shift=int(rng.integers(-1, 8)),
             p_mq=int(rng.integers(1, 1 << 14)), p_msh=int(rng.integers(0, 16)),
             p_am=int(rng.integers(0, 1 << 14)), p_ac=int(rng.integers(0, 1 << 14)),
             p_ash=int(rng.integers(0, 16)))
    if p["norm_mode"] != 1 and rng.random() < 0.5:
        p.update(n_mq=1, n_msh=0)
    if rng.random() < 0.3:  # identity Mul
        p.update(p_mq=1 << min(p["p_msh"], 14), p_msh=min(p["p_msh"], 14))
    return p


# (H, W, Ho, Wo)
FLOW_SHAPES = [(32, 32, 32, 32), (5, 300, 40, 12), (300, 7, 32, 4), (1, 1, 32, 8),
               (144, 144, 48, 16)]
FLOW_Q = [(0, 0.0078125, 0, 0.0078125, 0, 32.0), (0, 0.0078125, 0, 0.0078125, 0, 64.0),
          (0, 0.5, 3, ff(0.013), -5, ff(1 / 0.021)), (0, 1.0, -128, ff(0.37), 127, ff(1 / 0.11))]


@pytest.mark.parametrize("pre", [False, True])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("f_up", [0, 1])
@pytest.mark.parametrize("si", range(len(FLOW_SHAPES)))
@pytest.mark.parametrize("kind", [0, 1])
def test_qdq_gridsample_reducel2_fused(pre, padding, f_up, si, kind):
    H, W, Ho, Wo = FLOW_SHAPES[si]
    rng = np.random.default_rng([si, kind, f_up, padding, pre])
    for trial in range(4):
        p = flow_params(rng, H, W, kind)
        q = FLOW_Q[trial]
        x = rng.integers(-128, 128, (H, W, 4)).astype(np.int8)
        fs = (Ho // 2, Wo // 2, 4) if f_up else (Ho, Wo, 4)
        fa = rng.integers(-128, 128, fs).astype(np.int8)
        fa.reshape(-1, 4)[:4, :2] = [[-128, -128], [127, 127], [-128, 0], [0, 0]]
        fc = rng.integers(-128, 128, fs).astype(np.int8)
        tx = rng.integers(-128, 128, Wo).astype(np.int8)
        ty = rng.integers(-128, 128, Ho).astype(np.int8)
        if kind == 0:  # a skip-zero base grid, as the models use
            tx = np.where(np.arange(Wo) < Wo // 2, np.arange(Wo) - Wo // 2,
                          np.arange(Wo) - Wo // 2 + 1).astype(np.int8)
        args = (x, fa, fc) if pre else (x, fa)
        name = ("npu_qdq_Resize_Gridsample_ReduceL2_Fused_a8" if pre
                else "npu_qdq_Gridsample_ReduceL2_Fused_a8")
        got = callx(name, *args, tx, ty, OUT, I64(H), I64(W), I64(Ho), I64(Wo), padding, f_up,
                    Host(ctypes.byref(GridQdq(*q))),
                    Host(ctypes.byref(GridFlow(*(p[k] for k in gs.FLOW_FIELDS)))),
                    out_nbytes=Ho * Wo * 8, out_dtype=np.int8)
        if pre:
            want = gs.qdq_Resize_Gridsample_ReduceL2_Fused_a8(x, fa, fc, tx, ty, padding, f_up,
                                                              q, p)
        else:
            want = gs.qdq_Gridsample_ReduceL2_Fused_a8(x, fa, tx, ty, padding, f_up, q, p)
        assert_same(got, want.reshape(-1))


def test_reducel2_norm_exhaustive():
    """Every (fx, fy) int8 pair, every norm mode."""
    H = W = 1
    Ho, Wo = 256, 256
    v = np.arange(-128, 128, dtype=np.int8)
    fa = np.zeros((Ho, Wo, 4), np.int8)
    fa[..., 0], fa[..., 1] = v[None, :], v[:, None]
    x = np.zeros((H, W, 4), np.int8)
    tx, ty = np.zeros(Wo, np.int8), np.zeros(Ho, np.int8)
    for mode, n_mq, n_msh in [(0, 1, 0), (1, 11585, 14), (2, 1, 0), (2, 3, 1)]:
        p = dict.fromkeys(gs.FLOW_FIELDS, 0)
        p.update(mq_x=1, mq_y=1, ax=1, ay=1, norm_mode=mode, n_mq=n_mq, n_msh=n_msh,
                 out_shift=0)
        q = FLOW_Q[0]
        got = callx("npu_qdq_Gridsample_ReduceL2_Fused_a8", x, fa, tx, ty, OUT, I64(H), I64(W),
                    I64(Ho), I64(Wo), 1, 0, Host(ctypes.byref(GridQdq(*q))),
                    Host(ctypes.byref(GridFlow(*(p[k] for k in gs.FLOW_FIELDS)))),
                    out_nbytes=Ho * Wo * 8, out_dtype=np.int8)
        want = gs.qdq_Gridsample_ReduceL2_Fused_a8(x, fa, tx, ty, 1, 0, q, p)
        assert_same(got, want.reshape(-1))
        if mode == 0:  # round(sqrt) of the exact square, clamped to 127
            n = got.reshape(Ho, Wo, 8)[..., 4:].copy().view(np.int32)[..., 0]
            ref = np.minimum(np.rint(np.sqrt(v.astype(float)[None, :] ** 2
                                             + v.astype(float)[:, None] ** 2)), 127)
            assert np.array_equal(n, ref)


# ---- argument checks -----------------------------------------------------------------

def test_bad_args():
    z = np.zeros(8192, np.uint8)
    cq = Host(ctypes.byref(GridQdq(0, 1.0, 0, 1.0, 0, 1.0)))
    shp = lambda H, W, C, Ho=1, Wo=1: (I64(H), I64(W), I64(C), I64(Ho), I64(Wo))  # noqa: E731
    bf = lambda t, H=2, W=2, C=2, x=z, Ho=1, Wo=1: raw_rc(  # noqa: E731
        "npu_gridsample_bf16", x, t, z, z, *shp(H, W, C, Ho, Wo))
    assert bf(NPU_BF16) == 0
    # empty output: valid, nothing launched; pointers, types and C still checked
    assert bf(NPU_BF16, Ho=0) == 0
    assert bf(NPU_F16, Wo=0) == 0
    assert bf(NPU_BF16, Ho=0, x=None) == -1
    assert bf(NPU_FP8, Ho=0) == -1
    assert bf(NPU_BF16, Ho=0, C=0) == -1
    assert bf(NPU_BF16, Ho=-1) == -1
    assert bf(NPU_BF16, Wo=-1) == -1
    # odd C, H * W past the int16 index range, bad types, null pointers
    assert bf(NPU_BF16, C=3) == -1
    assert bf(NPU_BF16, H=129, W=256) == -1
    assert bf(NPU_FP8) == -1
    assert bf(NPU_BF16, x=None) == -1
    fp8 = lambda t, C=2, g=z: raw_rc("npu_gridsample_fp8", z, g, t, z,  # noqa: E731
                                     *shp(2, 2, C), 1.0, 1.0)
    assert fp8(NPU_F16) == 0
    assert fp8(NPU_I16) == -1
    assert fp8(NPU_BF16, C=3) == -1
    assert fp8(NPU_BF16, g=None) == -1
    assert raw_rc("npu_gridsample_fp8", z, z, NPU_BF16, z, *shp(2, 2, 2, 0, 0), 1.0, 1.0) == 0
    assert raw_rc("npu_gridsample_fp8", z, None, NPU_BF16, z, *shp(2, 2, 2, 0, 0), 1.0,
                  1.0) == -1
    a8 = lambda t, align=0, c=cq, o=z: raw_rc("npu_qdq_gridsample_a8", z, t, z, o,  # noqa: E731
                                              *shp(2, 2, 2), align, c)
    assert a8(NPU_U8) == 0
    assert a8(NPU_I16) == -1
    assert a8(NPU_I8, align=2) == -1
    assert a8(NPU_I8, c=None) == -1
    assert a8(NPU_I8, o=None) == -1
    assert raw_rc("npu_qdq_gridsample_a8", z, NPU_I8, z, z, *shp(2, 2, 2, 0, 4), 0, cq) == 0
    assert raw_rc("npu_qdq_gridsample_a8", z, NPU_I8, z, z, *shp(2, 2, 2, 0, 4), 0, None) == -1
    a16 = lambda t, ct, x=z: raw_rc("npu_qdq_gridsample_a16", x, t, z, z,  # noqa: E731
                                    *shp(2, 2, 2), 0, ct, cq)
    assert a16(NPU_I16, NPU_F16) == 0
    assert a16(NPU_U16, NPU_FP8) == -1
    assert a16(NPU_U8, NPU_BF16) == -1
    assert a16(NPU_U16, NPU_BF16, x=None) == -1
    assert raw_rc("npu_qdq_gridsample_a16", z, NPU_U16, z, z, *shp(2, 2, 2, 3, 0), 0, NPU_BF16,
                  cq) == 0
    assert raw_rc("npu_qdq_gridsample_a16", z, NPU_U16, z, z, *shp(2, 2, 2, 3, 0), 0, NPU_FP8,
                  cq) == -1

    good = dict.fromkeys(gs.FLOW_FIELDS, 0)

    def rc(H=32, W=32, Ho=32, Wo=32, pad=1, up=0, ty=z, q=cq, **kw):
        p = dict(good, **kw)
        fp = Host(ctypes.byref(GridFlow(*(p[k] for k in gs.FLOW_FIELDS))))
        return raw_rc("npu_qdq_Gridsample_ReduceL2_Fused_a8", z, z, z, ty, z, I64(H), I64(W),
                      I64(Ho), I64(Wo), pad, up, q, fp)
    assert rc() == 0
    # empty output: valid, nothing launched; the rest still checked
    assert rc(Ho=0) == 0
    assert rc(Wo=0) == 0
    assert rc(Ho=0, ty=None) == -1
    assert rc(Ho=0, shift=32) == -1
    for bad in [dict(Ho=24), dict(Ho=36), dict(Wo=6), dict(H=0), dict(W=32768), dict(pad=2),
                dict(Ho=-8), dict(Wo=-4), dict(Ho=8), dict(Wo=0, Ho=4),
                dict(up=2), dict(mq_x=32768), dict(ax=-32769), dict(ka=-1), dict(shift=32),
                dict(out_shift=-2), dict(norm_mode=3), dict(ty=None), dict(q=None)]:
        assert rc(**bad) == -1, bad
    fp = Host(ctypes.byref(GridFlow(*(good[k] for k in gs.FLOW_FIELDS))))
    pre = lambda c: raw_rc("npu_qdq_Resize_Gridsample_ReduceL2_Fused_a8", z, z, c, z,  # noqa: E731
                           z, z, I64(32), I64(32), I64(32), I64(32), 1, 0, cq, fp)
    assert pre(z) == 0
    assert pre(None) == -1
    assert raw_rc("npu_qdq_Resize_Gridsample_ReduceL2_Fused_a8", z, z, z, z, z, z, I64(32),
                  I64(32), I64(0), I64(32), 1, 0, cq, fp) == 0
    assert raw_rc("npu_qdq_Resize_Gridsample_ReduceL2_Fused_a8", z, z, None, z, z, z, I64(32),
                  I64(32), I64(0), I64(32), 1, 0, cq, fp) == -1
