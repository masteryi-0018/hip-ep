# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""sin / cos entry points against golden_sincos.py, bit for bit."""

import numpy as np
import pytest

from golden_float_cvt import scalar_cast
from golden_int_round import Rnd
import golden_sincos as gs
from npu import I64, NPU_BF16, NPU_F16, NPU_FP8, NPU_I8, NPU_I16, NPU_U8, OUT, raw_rc
from testing import ALL16, INT
from testing import assert_same as check
from testing import call_n as go

F32 = np.float32
NAT = {"bf16": (NPU_BF16, "bf16"), "f16": (NPU_F16, "f16")}
TABLES = {}


def table(is_cos):
    if is_cos not in TABLES:
        TABLES[is_cos] = gs.fit_lut(is_cos)
    return TABLES[is_cos]


def fp8_inputs():
    """Every fp8 code twice: in groups of 8 without a NaN code, and in natural order."""
    codes = np.arange(256, dtype=np.uint8)
    clean = codes[(codes & 0x7F) != 0x7F]
    clean = np.concatenate([clean, clean[:(-len(clean)) % 8]])
    return np.concatenate([clean, codes])


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("fn", ["sin", "cos"])
def test_direct_all_codes(fn, t):
    nt, ht = NAT[t]
    got = go(f"npu_{fn}_bf16", ALL16, nt, OUT, I64(65536), n=65536, dt=np.uint16)
    check(got, getattr(gs, f"{fn}_bf16")(ALL16, ht))


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("fn", ["sin", "cos"])
def test_lut_all_codes(fn, t):
    nt, ht = NAT[t]
    w = table(fn == "cos")
    got = go(f"npu_elementwise_{fn}_lut_bf16", ALL16, nt, w, OUT, I64(65536), n=65536,
             dt=np.uint16)
    check(got, getattr(gs, f"elementwise_{fn}_lut_bf16")(ALL16, w, ht))


def test_lut_random_table():
    """A table of arbitrary floats (signs, NaN, inf, subnormals)."""
    rng = np.random.default_rng(7)
    w = rng.standard_normal(gs.LUT_FLOATS).astype(F32) * F32(3)
    w[::97] = np.inf
    w[5::131] = np.nan
    w[7::61] = F32(1e-40)
    for t in ("bf16", "f16"):
        nt, ht = NAT[t]
        got = go("npu_elementwise_sin_lut_bf16", ALL16, nt, w, OUT, I64(65536), n=65536,
                 dt=np.uint16)
        check(got, gs.elementwise_sin_lut_bf16(ALL16, w, ht))


# (ifm_scale, ofm_scale_recip): plain, non power-of-two, bf16 / fp16 ties after the scale
# and in the recip cast, large scales.
FP8_SCALES = [(1.0, 1.0), (0.3, float(F32(1 / 0.0123))), (1.00390625, 1.00390625),
              (1.0, 1.00048828125), (0.75, 600.0), (5000.0, 0.01), (2.0 ** -9, 3.0)]


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("fn", ["sin", "cos"])
@pytest.mark.parametrize("sc", FP8_SCALES, ids=lambda s: f"{s[0]:g}_{s[1]:g}")
def test_fp8(fn, t, sc):
    nt, ht = NAT[t]
    x = fp8_inputs()
    n = x.size
    s, r = sc
    got = go(f"npu_elementwise_{fn}_fp8", x, OUT, I64(n), nt, s, r, n=n, dt=np.uint8)
    check(got, getattr(gs, f"elementwise_{fn}_fp8")(x, ht, s, r))


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("fn", ["sin", "cos"])
@pytest.mark.parametrize("sc", FP8_SCALES, ids=lambda s: f"{s[0]:g}_{s[1]:g}")
def test_lut_fp8(fn, t, sc):
    nt, ht = NAT[t]
    x = fp8_inputs()
    n = x.size
    s, r = sc
    w = table(fn == "cos")
    got = go(f"npu_elementwise_{fn}_lut_fp8", x, w, OUT, I64(n), nt, s, r, n=n, dt=np.uint8)
    check(got, getattr(gs, f"elementwise_{fn}_lut_fp8")(x, w, ht, s, r))


def test_scalar_cast():
    """Scalar cast: bf16 half to even under both modes, fp16 by the mode."""
    tie_b, tie_h = 1.00390625, 1.00048828125
    assert scalar_cast(tie_b, "bf16", Rnd.kHalfAway) == 1.0
    assert scalar_cast(tie_h, "f16", Rnd.kHalfEven) == 1.0
    assert scalar_cast(tie_h, "f16", Rnd.kHalfAway) == 1.0009765625


def all_codes(t):
    info = np.iinfo(INT[t][1])
    return np.arange(info.min, info.max + 1, dtype=np.int64).astype(INT[t][1])


def qdq_params(bits):
    """(dq_zp, dq_sc, q_zp, q_sc)."""
    if bits == 8:
        return [(0.0, 0.0371, 0.0, float(F32(1 / 0.0078125))),
                (-3.0, 0.4137, 5.0, float(F32(1 / 0.0091))),     # |x| ~ 50: range reduction
                (128.0, 0.01, 128.0, 300.0)]                      # saturation
    return [(0.0, 0.00040086842, 22.0, float(F32(1 / 0.0004007))),
            (24.0, 0.0371, -7.0, float(F32(1 / 3.05e-5))),        # |x| ~ 1200, saturation
            (32768.0, 1.7, 32768.0, 1000.0)]                      # beyond the int16 reduction


def qdq_types(bits):
    s, u = ("i8", "u8") if bits == 8 else ("i16", "u16")
    return [(s, s), (u, s), (s, u), (u, u)]


@pytest.mark.parametrize("ct", ["bf16", "f16"])
@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("fn", ["sin", "cos"])
def test_qdq(fn, bits, ct):
    cnt, ht = NAT[ct]
    for it, ot in qdq_types(bits):
        x = all_codes(it)
        n = x.size
        for izp, isc, ozp, orc in qdq_params(bits):
            got = go(f"npu_elementwise_{fn}_a{bits}", x, INT[it][0], OUT, INT[ot][0], I64(n),
                     cnt, izp, isc, ozp, orc, n=n, dt=INT[ot][1])
            check(got, getattr(gs, f"elementwise_{fn}_a{bits}")(x, ot, ht, izp, isc, ozp, orc))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("fn", ["sin", "cos"])
def test_qdq_lut(fn, bits):
    w = table(fn == "cos")
    for it, ot in qdq_types(bits):
        x = all_codes(it)
        n = x.size
        for izp, isc, ozp, orc in qdq_params(bits):
            got = go(f"npu_elementwise_{fn}_lut_a{bits}", x, INT[it][0], w, OUT, INT[ot][0],
                     I64(n), izp, isc, ozp, orc, n=n, dt=INT[ot][1])
            check(got, getattr(gs, f"elementwise_{fn}_lut_a{bits}")(x, w, ot, izp, isc, ozp, orc))


def test_golden_nan_rules():
    """IEEE NaNs: an input NaN passes with its sign and payload (sin, cos, LUT); fp8 NaN
    groups decode to +NaN."""
    nan_p, nan_n = np.array([0x7F85, 0xFF85], np.uint16), np.array([0x7C05, 0xFC05], np.uint16)
    assert gs.sin_bf16(nan_p, "bf16").tolist() == [0x7FC5, 0xFFC5]
    assert gs.cos_bf16(nan_p, "bf16").tolist() == [0x7FC5, 0xFFC5]
    assert gs.sin_bf16(nan_n, "f16").tolist() == [0x7E05, 0xFE05]
    assert gs.cos_bf16(nan_n, "f16").tolist() == [0x7E05, 0xFE05]
    w = table(True)
    assert w[1] < 0  # cos slope[0] < 0
    assert gs.elementwise_cos_lut_bf16(nan_p, w, "bf16").tolist() == [0x7FC5, 0xFFC5]
    nan8 = np.array([0x7F, 1, 2, 3, 4, 5, 6, 7], np.uint8)
    assert set(gs.elementwise_sin_fp8(nan8, "bf16", 1.0, 1.0).tolist()) == {0x7F}
    assert set(gs.elementwise_sin_fp8(nan8, "f16", 1.0, 1.0).tolist()) == {0x7F}


def test_bad_args():
    P = np.zeros(64, np.uint8)  # a valid device buffer
    n = I64(64)
    w = table(False)
    x16 = np.zeros(64, np.uint16)
    x8 = np.zeros(64, np.uint8)
    for fn in ("sin", "cos"):
        name = f"npu_{fn}_bf16"
        assert raw_rc(name, None, NPU_FP8, None, n) == -1
        assert raw_rc(name, None, NPU_BF16, None, I64(-1)) == -1
        assert raw_rc(name, None, NPU_BF16, None, n) == -1
        assert raw_rc(name, None, NPU_BF16, None, I64(0)) == -1
        assert raw_rc(name, P, NPU_BF16, P, I64(0)) == 0
        name = f"npu_elementwise_{fn}_fp8"
        assert raw_rc(name, None, None, n, NPU_FP8, 1.0, 1.0) == -1
        assert raw_rc(name, None, None, I64(-1), NPU_BF16, 1.0, 1.0) == -1
        assert raw_rc(name, None, None, n, NPU_BF16, 1.0, 1.0) == -1
        assert raw_rc(name, None, None, I64(0), NPU_F16, 1.0, 1.0) == -1
        assert raw_rc(name, P, P, I64(0), NPU_F16, 1.0, 1.0) == 0
        name = f"npu_elementwise_{fn}_lut_bf16"
        assert raw_rc(name, None, NPU_I16, w, None, n) == -1
        assert raw_rc(name, None, NPU_BF16, w, None, I64(-1)) == -1
        assert raw_rc(name, x16, NPU_BF16, None, x16, n) == -1
        assert raw_rc(name, None, NPU_BF16, w, None, n) == -1
        assert raw_rc(name, None, NPU_BF16, None, None, I64(0)) == -1
        assert raw_rc(name, P, NPU_BF16, P, P, I64(0)) == 0
        name = f"npu_elementwise_{fn}_lut_fp8"
        assert raw_rc(name, None, w, None, n, NPU_I8, 1.0, 1.0) == -1
        assert raw_rc(name, None, w, None, I64(-1), NPU_BF16, 1.0, 1.0) == -1
        assert raw_rc(name, x8, None, x8, n, NPU_BF16, 1.0, 1.0) == -1
        assert raw_rc(name, None, None, None, I64(0), NPU_BF16, 1.0, 1.0) == -1
        assert raw_rc(name, P, P, P, I64(0), NPU_BF16, 1.0, 1.0) == 0
        for bits, good, bad in ((8, NPU_I8, NPU_I16), (16, NPU_I16, NPU_U8)):
            co = (0.0, 1.0, 0.0, 1.0)
            name = f"npu_elementwise_{fn}_a{bits}"
            assert raw_rc(name, None, bad, None, good, n, NPU_BF16, *co) == -1
            assert raw_rc(name, None, good, None, bad, n, NPU_BF16, *co) == -1
            assert raw_rc(name, None, good, None, good, n, NPU_FP8, *co) == -1
            assert raw_rc(name, None, good, None, good, I64(-1), NPU_BF16, *co) == -1
            assert raw_rc(name, None, good, None, good, n, NPU_BF16, *co) == -1
            assert raw_rc(name, None, good, None, good, I64(0), NPU_F16, *co) == -1
            assert raw_rc(name, P, good, P, good, I64(0), NPU_F16, *co) == 0
            name = f"npu_elementwise_{fn}_lut_a{bits}"
            x = np.zeros(64, np.int8 if bits == 8 else np.int16)
            assert raw_rc(name, None, bad, w, None, good, n, *co) == -1
            assert raw_rc(name, None, good, w, None, bad, n, *co) == -1
            assert raw_rc(name, None, good, w, None, good, I64(-1), *co) == -1
            assert raw_rc(name, x, good, None, x, good, n, *co) == -1
            assert raw_rc(name, None, good, None, None, good, I64(0), *co) == -1
            assert raw_rc(name, P, good, P, P, good, I64(0), *co) == 0
