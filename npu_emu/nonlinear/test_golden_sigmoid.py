# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""sigmoid / tanh entry points against golden_sigmoid.py, bit for bit."""

import numpy as np
import pytest

from golden_float_cvt import fp8_to_f32, round_float
from golden_int_round import INT_RANGE, Rnd
from golden_nlf import tanh16
import golden_sigmoid as gs
from npu import I64, OUT, NPU_BF16, NPU_F16, NPU_FP8, NPU_I8, NPU_I16, NPU_U8, call, raw_rc
from npu import f32 as f
from testing import F16T as HALF
from testing import INT
from testing import same_bytes as assert_same

F32 = np.float32
RNG = np.random.default_rng(4242)


def image(offset, slope):
    """Table image (512 fp32) of 128 (offset, slope) pairs."""
    w = np.stack([np.asarray(offset, F32), np.asarray(slope, F32)], 1).reshape(16, 1, 8, 2)
    return np.repeat(w, 2, axis=1).reshape(-1).astype(F32)


def synthetic_lut(kind):
    """Random slopes / offsets with full fp32 mantissas."""
    sl = RNG.uniform(1e-3, 0.6, 128).astype(F32)
    off = RNG.uniform(-1.0, 1.0, 128).astype(F32)
    if kind == "sigmoid":
        off = (off + 1) / 2
    return image(off, sl)


def luts(kind):
    out = [("synthetic", synthetic_lut(kind))]
    return out


@pytest.mark.parametrize("t", [gs.BF16, gs.F16])
def test_tanh16_exhaustive(t):
    x = np.arange(1 << 16, dtype=np.uint16)
    got = call("npu_test_sigmoid_tanh16", x, HALF[t], OUT, I64(len(x)), out_nbytes=2 * len(x),
               out_dtype=np.uint16)
    assert_same(got, tanh16(x, t))


def codes(t):
    lo, hi = INT_RANGE[t]
    return np.arange(lo, hi + 1, dtype=np.int64)


# (dq zero point, dq scale, q zero point, q scale): non power-of-two scales, bf16 / f16
# ties in the dq, integer ties in the q, saturation, dq overflow to inf, subnormals.
QDQ8 = [(3.0, f(0.09543454647064209), 0.0, f(1 / 0.003921241965144873)),
        (-7.0, f(0.0313), 5.0, 256.0),
        (0.0, 1.00390625, -3.0, 0.5),
        (0.0, 1.00048828125, 0.0, 2.0 ** -7),
        (128.0, f(0.051), 128.0, f(1 / 0.0039)),
        (2.5, f(0.37), 0.0, 1e5),
        (0.0, 2.0 ** -140, 0.0, 2.0 ** 130),
        (0.0, 2.0 ** -20, 1.0, 3.0),
        # dq ties that reach the output
        (0.0, 2.0 ** -12 * (1 + 2.0 ** -8), -100.0, 1.5 * 2.0 ** 19),
        (0.0, 2.0 ** -9 * (1 + 2.0 ** -8), -120.0, 2.0 ** 15)]
QDQ16 = QDQ8 + [(0.0, 2.0 ** -12 * (1 + 2.0 ** -11), 0.0, 2.0 ** 14),
                (0.0, 2.0 ** -10 * (1 + 2.0 ** -8), 0.0, 2.0 ** 12),
                (0.0, 2.0 ** -13 * (1 + 2.0 ** -11), -30000.0, 2.0 ** 18),(0.0, 3.0e36, 0.0, 1000.0), (0.0, 4.0, 0.0, 1000.0),
                (-1234.0, f(0.000305), 17.0, f(1 / 3.0518e-05)),
                (0.0, 2.0 ** -12, 0.0, 32768.0)]


def qdq_params(a16):
    sets = QDQ16 if a16 else QDQ8
    io = (("i16", "i16"), ("u16", "u16"), ("i16", "u16")) if a16 else \
        (("i8", "i8"), ("u8", "u8"), ("u8", "i8"))
    return [(it, ot, c) for it, ot in io for c in sets]


@pytest.mark.parametrize("op", ["sigmoid", "tanh"])
@pytest.mark.parametrize("a16", [False, True], ids=["a8", "a16"])
@pytest.mark.parametrize("t", [gs.BF16, gs.F16])
def test_qdq(op, a16, t):
    fn = gs.sigmoid_a8 if op == "sigmoid" else gs.tanh_a8
    name = f"npu_{op}_{'a16' if a16 else 'a8'}"
    for it, ot, co in qdq_params(a16):
        x = codes(it)
        (in_t, idt), (out_t, odt) = INT[it], INT[ot]
        got = call(name, x.astype(idt), in_t, OUT, out_t, I64(len(x)), HALF[t], *co,
                   out_nbytes=len(x) * np.dtype(odt).itemsize, out_dtype=odt)
        assert_same(got, fn(x, ot, t, *co).astype(odt))


@pytest.mark.parametrize("a16", [False, True], ids=["a8", "a16"])
@pytest.mark.parametrize("t", [gs.BF16, gs.F16])
def test_qdq_tanh_sweep(a16, t):
    """Power-of-two dq scales: every compute-type code within reach of the integer inputs."""
    name = f"npu_tanh_{'a16' if a16 else 'a8'}"
    it, ot = ("i16", "i16") if a16 else ("i8", "i8")
    x = codes(it)
    (in_t, idt), (out_t, odt) = INT[it], INT[ot]
    ks = range(-140, 120, 3) if t == gs.BF16 else range(-31, 16)
    for k in ks:
        co = (0.0, 2.0 ** k, 0.0, 2.0 ** 15 if a16 else 127.0)
        got = call(name, x.astype(idt), in_t, OUT, out_t, I64(len(x)), HALF[t], *co,
                   out_nbytes=len(x) * np.dtype(odt).itemsize, out_dtype=odt)
        assert_same(got, gs.tanh_a8(x, ot, t, *co).astype(odt))


@pytest.mark.parametrize("op", ["sigmoid", "tanh"])
@pytest.mark.parametrize("a16", [False, True], ids=["a8", "a16"])
def test_qdq_lut(op, a16):
    fn = gs.qdq_sigmoid_lut_a8 if op == "sigmoid" else gs.qdq_tanh_lut_a8
    name = f"npu_qdq_{op}_lut_{'a16' if a16 else 'a8'}"
    extra = [(0.0, f(8.0 / 127), 0.0, 127.0), (0.0, f(4.0 / 127), 0.0, 127.0),
             (0.0, 2.0 ** -5, 0.0, 2.0 ** 9), (0.0, 2.0 ** -4, 0.0, 2.0 ** 13)]
    for lname, lut in luts(op):
        for it, ot, co in qdq_params(a16) + [(p[0], p[1], c) for p in qdq_params(a16)[:1]
                                              for c in extra]:
            x = codes(it)
            (in_t, idt), (out_t, odt) = INT[it], INT[ot]
            got = call(name, x.astype(idt), in_t, lut, OUT, out_t, I64(len(x)), *co,
                       out_nbytes=len(x) * np.dtype(odt).itemsize, out_dtype=odt)
            assert_same(got, fn(x, ot, *co, lut).astype(odt))


@pytest.mark.parametrize("op", ["sigmoid", "tanh"])
@pytest.mark.parametrize("t", [gs.BF16, gs.F16])
def test_lut16_exhaustive(op, t):
    fn = gs.elementwise_sigmoid_lut_bf16 if op == "sigmoid" else gs.elementwise_tanh_lut_bf16
    x = np.arange(1 << 16, dtype=np.uint16)
    for lname, lut in luts(op):
        got = call(f"npu_elementwise_{op}_lut_bf16", x, HALF[t], lut, OUT, I64(len(x)),
                   out_nbytes=2 * len(x), out_dtype=np.uint16)
        assert_same(got, fn(x, t, lut))


def fp8_inputs():
    """All 256 codes in order, shuffled, and the finite codes only."""
    x = np.arange(256, dtype=np.uint8)
    fin = x[(x & 0x7F) != 0x7F]
    return [x, RNG.permutation(x), np.concatenate([fin, fin[::-1]])]


# (ifm_scale, ofm_scale_recip): powers of two, non powers of two, bf16 / f16 ties in the
# cast of the reciprocal, fp8 overflow, f16 overflow of the reciprocal, tiny scales.
FP8_SCALES = [(1.0, 8.0), (0.5, 4.0), (0.25, 2.0), (f(0.3), f(1 / 0.07)),
              (1.00390625, f(1 / 0.9961089494163424)), (1.00048828125, f(1 / 0.9995119570522206)),
              (f(0.0137), 1.00390625), (2.0 ** -6, 1.00048828125), (f(3.1), f(3.0 + 2.0 ** -9)),
              (1.0, 500.0), (2.0 ** 7, 1000.0), (1.0, 7e4), (1.0, 2.0 ** -10), (f(1e-3), f(0.013)),
              (1.0, 2.0 ** -16 * 3.0)]


@pytest.mark.parametrize("op", ["sigmoid", "tanh"])
def test_fp8(op):
    cts = [None] if op == "sigmoid" else [gs.BF16, gs.F16]
    for lname, lut in luts(op):
        for x in fp8_inputs():
            for s_in, recip in FP8_SCALES:
                for t in cts:
                    if op == "sigmoid":
                        got = call("npu_elementwise_sigmoid_lut_fp8", x, lut, OUT, I64(len(x)),
                                   s_in, recip, out_nbytes=len(x), out_dtype=np.uint8)
                        want = gs.elementwise_sigmoid_lut_fp8(x, lut, s_in, recip)
                    else:
                        got = call("npu_elementwise_tanh_lut_fp8", x, lut, OUT, I64(len(x)),
                                   HALF[t], s_in, recip, out_nbytes=len(x), out_dtype=np.uint8)
                        want = gs.elementwise_tanh_lut_fp8(x, t, lut, s_in, recip)
                    assert_same(got, want)


def below_ties(t):
    """128 compute-type values one ulp below an fp8 tie in [0.25, 1), both signs."""
    v = fp8_to_f32(np.arange(0x28, 0x38))  # fp8 values in [0.25, 1)
    ties = (v[:-1] + v[1:]) / 2
    ulp = np.where(ties >= 0.5, 2.0 ** -8, 2.0 ** -9) if t == gs.BF16 else \
        np.where(ties >= 0.5, 2.0 ** -11, 2.0 ** -12)
    y = (ties - ulp).astype(F32)
    y = np.resize(np.concatenate([y, -y]), 128)
    assert np.array_equal(round_float(y, t), y)
    return y


def test_scalar_cast(monkeypatch):
    """The cast of the reciprocal: f16 rounds ties away from zero, bf16 to even."""
    x = np.arange(256, dtype=np.uint8)
    for t, r in ((gs.F16, 1.0 + 2.0 ** -11), (gs.BF16, 1.0 + 2.0 ** -8)):
        lut = image(below_ties(t), np.zeros(128))
        want = gs.elementwise_tanh_lut_fp8(x, t, lut, 2.0 ** -3, f(r))
        got = call("npu_elementwise_tanh_lut_fp8", x, lut, OUT, I64(len(x)), HALF[t], 2.0 ** -3,
                   f(r), out_nbytes=len(x), out_dtype=np.uint8)
        assert_same(got, want)
        with monkeypatch.context() as m:  # the other rounding would be visible
            m.setattr(gs, "scalar_cast", lambda v, t, rnd: round_float(
                F32(v), t, Rnd.kHalfEven if t == gs.F16 else Rnd.kHalfAway))
            assert np.any(gs.elementwise_tanh_lut_fp8(x, t, lut, 2.0 ** -3, f(r)) != want)


def test_bad_args():
    P = np.zeros(64, np.uint8)  # a valid device buffer
    n = I64(4)
    lut = synthetic_lut("sigmoid")
    co = (0.0, 1.0, 0.0, 1.0)
    for bits, good, bad in ((8, NPU_I8, NPU_I16), (16, NPU_I16, NPU_U8)):
        for op in ("sigmoid", "tanh"):
            name = f"npu_{op}_a{bits}"
            assert raw_rc(name, None, bad, None, good, n, NPU_BF16, *co) == -1
            assert raw_rc(name, None, good, None, bad, n, NPU_BF16, *co) == -1
            assert raw_rc(name, None, good, None, good, n, NPU_FP8, *co) == -1
            assert raw_rc(name, None, good, None, good, I64(-1), NPU_BF16, *co) == -1
            assert raw_rc(name, None, good, None, good, n, NPU_BF16, *co) == -1
            assert raw_rc(name, None, good, None, good, I64(0), NPU_BF16, *co) == -1
            assert raw_rc(name, P, good, P, good, I64(0), NPU_BF16, *co) == 0
            name = f"npu_qdq_{op}_lut_a{bits}"
            assert raw_rc(name, None, bad, lut, None, good, n, *co) == -1
            assert raw_rc(name, None, good, lut, None, bad, n, *co) == -1
            assert raw_rc(name, None, good, lut, None, good, I64(-1), *co) == -1
            assert raw_rc(name, None, good, lut, None, good, n, *co) == -1
            assert raw_rc(name, None, good, None, None, good, I64(0), *co) == -1
            assert raw_rc(name, P, good, P, P, good, I64(0), *co) == 0
            # a null table with buffers
            x = np.zeros(4, np.int8 if bits == 8 else np.int16)
            assert raw_rc(name, x, good, None, x, good, n, *co) == -1
    x16 = np.zeros(4, np.uint16)
    for op in ("sigmoid", "tanh"):
        name = f"npu_elementwise_{op}_lut_bf16"
        assert raw_rc(name, None, NPU_FP8, lut, None, n) == -1
        assert raw_rc(name, None, NPU_BF16, lut, None, I64(-1)) == -1
        assert raw_rc(name, x16, NPU_BF16, None, x16, n) == -1
        assert raw_rc(name, None, NPU_BF16, lut, None, n) == -1
        assert raw_rc(name, None, NPU_BF16, None, None, I64(0)) == -1
        assert raw_rc(name, P, NPU_BF16, P, P, I64(0)) == 0
    x8 = np.zeros(4, np.uint8)
    assert raw_rc("npu_elementwise_sigmoid_lut_fp8", None, lut, None, I64(-1), 1.0, 1.0) == -1
    assert raw_rc("npu_elementwise_sigmoid_lut_fp8", x8, None, x8, n, 1.0, 1.0) == -1
    assert raw_rc("npu_elementwise_sigmoid_lut_fp8", None, None, None, I64(0), 1.0, 1.0) == -1
    assert raw_rc("npu_elementwise_sigmoid_lut_fp8", P, P, P, I64(0), 1.0, 1.0) == 0
    assert raw_rc("npu_elementwise_tanh_lut_fp8", None, lut, None, n, NPU_I8, 1.0, 1.0) == -1
    assert raw_rc("npu_elementwise_tanh_lut_fp8", x8, None, x8, n, NPU_BF16, 1.0, 1.0) == -1
    assert raw_rc("npu_elementwise_tanh_lut_fp8", None, lut, None, n, NPU_BF16, 1.0, 1.0) == -1
    assert raw_rc("npu_elementwise_tanh_lut_fp8", P, None, P, I64(0), NPU_F16, 1.0, 1.0) == -1
    assert raw_rc("npu_elementwise_tanh_lut_fp8", P, P, P, I64(0), NPU_F16, 1.0, 1.0) == 0
    assert raw_rc("npu_test_sigmoid_tanh16", None, NPU_FP8, None, n) == -1
    assert raw_rc("npu_test_sigmoid_tanh16", None, NPU_BF16, None, n) == -1
    assert raw_rc("npu_test_sigmoid_tanh16", None, NPU_BF16, P, I64(0)) == -1
    assert raw_rc("npu_test_sigmoid_tanh16", P, NPU_BF16, P, I64(-1)) == -1
    assert raw_rc("npu_test_sigmoid_tanh16", P, NPU_BF16, P, I64(0)) == 0
