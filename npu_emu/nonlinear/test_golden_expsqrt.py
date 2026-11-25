# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""exp / elu / hardsigmoid / sqrt entry points against golden_expsqrt.py, bit for bit."""

import numpy as np
import pytest

import golden_expsqrt as ge
from golden_float_cvt import f32_to_fp8
from golden_mmac import fp8_decode_identity
from golden_int_round import Rnd
from npu import I64, NPU_BF16, NPU_FP8, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT, call, raw_rc
from testing import ALL16, F16T, INT
from testing import assert_same as check
from testing import int_codes as ints

f32 = np.float32
RNG = np.random.default_rng(20261115)


def go16(name, x, t, *coeff):
    return call(name, x, F16T[t], OUT, I64(len(x)), *[float(c) for c in coeff],
                out_nbytes=2 * len(x), out_dtype=np.uint16)


def go8(name, x, *coeff):
    return call(name, x, OUT, I64(len(x)), *[float(c) for c in coeff], out_nbytes=len(x),
                out_dtype=np.uint8)


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_exp16_exhaustive(t):
    check(go16("npu_elementwise_exp_bf16", ALL16, t), ge.elementwise_exp_bf16(ALL16, t))


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("alpha", [1.0, 0.37, 1.6732632, 0.0, 600.0, -0.5, 3e-39])
def test_elu16_exhaustive(t, alpha):
    a = f32(alpha)
    check(go16("npu_elementwise_elu_bf16", ALL16, t, a), ge.elementwise_elu_bf16(ALL16, t, a))


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("alpha,beta", [(0.2, 0.5), (1 / 6, 0.5), (0.37, 0.29), (0.0, 0.5),
                                        (-0.25, 0.5), (1.0, 0.0), (3.0, -1.0), (1e-30, 0.0)])
def test_hardsigmoid16_exhaustive(t, alpha, beta):
    a, b = f32(alpha), f32(beta)
    check(go16("npu_elementwise_hardsigmoid_bf16", ALL16, t, a, b),
          ge.elementwise_hardsigmoid_bf16(ALL16, t, a, b))


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_sqrt16_exhaustive(t):
    check(go16("npu_sqrt_bf16", ALL16, t), ge.sqrt_bf16(ALL16, t))


def fp8_inputs():
    """Every code alone in a group of 8, random groups with NaN codes, a ragged tail of 5."""
    alone = np.repeat(np.arange(256, dtype=np.uint8), 8)
    mixed = RNG.integers(0, 256, 8 * 64).astype(np.uint8)
    mixed[RNG.integers(0, mixed.size, 12)] = 0x7F
    mixed[RNG.integers(0, mixed.size, 12)] = 0xFF
    tail = np.array([0x38, 0x80, 0x00, 0xB8, 0x7F], np.uint8)
    return np.concatenate([alone, mixed, tail])


FP8_X = fp8_inputs()
# (ifm_scale, ofm_scale)
FP8_SCALES = [(1.0, 1.0), (1.0625, 1.0), (0.37, 0.29), (0.37, 0.0029), (1.0, 0.01),
              (2.375, 1.0), (0.0166015625, 0.015625), (3.625, 1.7), (1.0, 300.0)]


@pytest.mark.parametrize("si,so", FP8_SCALES)
def test_exp_fp8(si, so):
    c = ge.coeff_exp_fp8(si, so)
    check(go8("npu_elementwise_exp_fp8", FP8_X, *c), ge.elementwise_exp_fp8(FP8_X, *c))


@pytest.mark.parametrize("si,so", FP8_SCALES)
@pytest.mark.parametrize("alpha", [1.0, 0.83, 600.0, -0.5])
def test_elu_fp8(si, so, alpha):
    c = ge.coeff_elu_fp8(si, so, alpha)
    check(go8("npu_elementwise_elu_fp8", FP8_X, *c), ge.elementwise_elu_fp8(FP8_X, *c))


@pytest.mark.parametrize("si,so", FP8_SCALES)
@pytest.mark.parametrize("alpha,beta", [(0.2, 0.5), (1.0, 0.0), (1.5, 0.25), (-0.3, 0.5)])
def test_hardsigmoid_fp8(si, so, alpha, beta):
    c = ge.coeff_hardsigmoid_fp8(si, so, alpha, beta)
    check(go8("npu_elementwise_hardsigmoid_fp8", FP8_X, *c),
          ge.elementwise_hardsigmoid_fp8(FP8_X, *c))


@pytest.mark.parametrize("si,so", FP8_SCALES + [(1.0, 0.9411764705882353),
                                                (1.0, 0.017543859649122806), (0.37, 0.0291)])
def test_sqrt_fp8(si, so):
    c = ge.coeff_sqrt_fp8(si, so)
    check(go8("npu_elementwise_sqrt_fp8", FP8_X, c), ge.elementwise_sqrt_fp8(FP8_X, c))


def test_fp8_ties():
    """Output ties: away from zero for hardsigmoid, to even for sqrt."""
    x = np.repeat(np.arange(0, 0x78, dtype=np.uint8), 8)
    v = fp8_decode_identity(x) * f32(1.5)
    assert np.any(f32_to_fp8(v, Rnd.kHalfAway) != f32_to_fp8(v, Rnd.kHalfEven))
    c = (f32(1.5), f32(0.0), f32(448.0))
    check(go8("npu_elementwise_hardsigmoid_fp8", x, *c), ge.elementwise_hardsigmoid_fp8(x, *c))
    check(go8("npu_elementwise_sqrt_fp8", x, f32(1.5)), ge.elementwise_sqrt_fp8(x, f32(1.5)))


# (s_in, zp_in, s_out, zp_out) per width
QP = {8: [(0.024, 0, 0.1, -128), (0.0371, 17, 0.00731, 3), (0.004008684190921485, 24,
                                                              0.004007086390629411, 22),
          (0.0234375, 1, 0.008470021, 0), (0.25, -3, 0.75, 100)],
      16: [(0.0001, 32768, 0.9961089494163424, 257), (0.00013, -77, 0.001, 259),
           (0.00040086841909214854, 24, 0.0004007086390629411, 22),
           (0.0009765625, 0, 0.00123, 0), (0.0137, -123, 0.0123, -5),
           (0.00015258789, -32768, 0.00015258789, -32768)]}


def qdq_types(bits):
    names = ("i8", "u8") if bits == 8 else ("i16", "u16")
    return [(a, b) for a in names for b in names]


def go_qdq(name, x, it, ot, *coeff):
    n = len(x)
    args = [c if isinstance(c, int) else float(c) for c in coeff]
    return call(name, x, INT[it][0], OUT, INT[ot][0], I64(n), *args,
                out_nbytes=n * np.dtype(INT[ot][1]).itemsize, out_dtype=INT[ot][1])


@pytest.mark.parametrize("bits", [8, 16])
def test_exp_qdq(bits):
    for it, ot in qdq_types(bits):
        x = ints(it)
        for p in QP[bits]:
            c = ge.coeff_exp_qdq(*p)
            check(go_qdq(f"npu_elementwise_exp_a{bits}", x, it, ot, *c),
                  ge.elementwise_exp_a8(x, ot, *c))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("alpha", [1.0, 0.37, 2.5])
def test_elu_qdq(bits, alpha):
    for it, ot in qdq_types(bits):
        x = ints(it)
        for p in QP[bits]:
            c = ge.coeff_elu_qdq(*p, alpha)
            check(go_qdq(f"npu_elementwise_elu_a{bits}", x, it, ot, *c),
                  ge.elementwise_elu_a8(x, ot, *c))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("alpha,beta", [(0.2, 0.5), (0.166666667, 0.5), (1.0, 0.25)])
def test_hardsigmoid_qdq(bits, alpha, beta):
    for it, ot in qdq_types(bits):
        x = ints(it)
        for p in QP[bits]:
            c = ge.coeff_hardsigmoid_qdq(*p, alpha, beta, ot)
            check(go_qdq(f"npu_hardsigmoid_a{bits}", x, it, ot, *c),
                  ge.hardsigmoid_a8(x, ot, *c))


@pytest.mark.parametrize("bits", [8, 16])
def test_sqrt_qdq(bits):
    extra = [(f32(2.0 ** -20), 0, 0.001, 0), (8.0, 0, 0.5, 3)]  # fp16 subnormal / overflow
    for it, ot in qdq_types(bits):
        x = ints(it)
        for p in QP[bits] + extra:
            c = ge.coeff_sqrt_qdq(*p)
            check(go_qdq(f"npu_qdq_sqrt_a{bits}", x, it, ot, *c), ge.qdq_sqrt_a8(x, ot, *c))


def test_exp_qdq_exp2_cases():
    """x = 0 and c0 = p make the exponent p; c1_out = 2^(14 - floor(log2 e)) moves e to
    [2^14, 2^15), so the int16 output shows every bit of e."""
    p = np.concatenate([np.linspace(-100, 127, 6000, dtype=f32),
                        RNG.uniform(-100, 127, 2000).astype(f32), [f32(127.0), f32(200.0)]])
    e = ge.exp2_bf16(np.minimum(p, f32(127.0)))
    assert np.all(np.isfinite(e)) and np.all(e > 0)
    k = 14 - np.floor(np.log2(e.astype(np.float64))).astype(np.int64)
    x = np.zeros(32, np.int16)
    for pv, kv in zip(p, k):
        c = (f32(1.0), f32(pv), f32(2.0 ** int(kv)), f32(0.0))
        got = go_qdq("npu_elementwise_exp_a16", x, "i16", "i16", *c)
        want = ge.elementwise_exp_a8(x.astype(np.int64), "i16", *c)
        assert 1 << 14 <= int(want[0]) < 1 << 15
        check(got, want)


def test_bad_args():
    P = np.zeros(64, np.uint8)  # a valid device buffer
    n = I64(64)
    x16 = np.zeros(64, np.uint16)
    # bad dtype, negative n, null buffers with n > 0
    for name, co in (("npu_elementwise_exp_bf16", ()), ("npu_elementwise_elu_bf16", (1.0,)),
                     ("npu_elementwise_hardsigmoid_bf16", (0.2, 0.5)), ("npu_sqrt_bf16", ())):
        assert raw_rc(name, None, NPU_FP8, None, n, *co) == -1
        assert raw_rc(name, None, NPU_BF16, None, I64(-1), *co) == -1
        assert raw_rc(name, x16, NPU_BF16, None, n, *co) == -1
        assert raw_rc(name, None, NPU_BF16, None, I64(0), *co) == -1
        assert raw_rc(name, P, NPU_BF16, P, I64(0), *co) == 0
    for name, co in (("npu_elementwise_exp_fp8", (1.0, 1.0)),
                     ("npu_elementwise_elu_fp8", (1.0, 1.0, 1.0)),
                     ("npu_elementwise_hardsigmoid_fp8", (1.0, 0.0, 1.0)),
                     ("npu_elementwise_sqrt_fp8", (1.0,))):
        assert raw_rc(name, None, None, I64(-1), *co) == -1
        assert raw_rc(name, np.zeros(64, np.uint8), None, n, *co) == -1
        assert raw_rc(name, None, None, I64(0), *co) == -1
        assert raw_rc(name, P, P, I64(0), *co) == 0
    qdq = (("exp", (1.0, 0.0, 1.0, 0.0)), ("elu", (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)),
           ("hardsigmoid", (1.0, 0.0, 0, 1)), ("qdq_sqrt", (0.0, 1.0, 0.0, 1.0)))
    for op, co in qdq:
        for bits, good, bad in ((8, NPU_I8, NPU_I16), (16, NPU_U16, NPU_U8)):
            name = (f"npu_elementwise_{op}_a{bits}" if op in ("exp", "elu") else
                    f"npu_{op}_a{bits}")
            assert raw_rc(name, None, bad, None, good, n, *co) == -1
            assert raw_rc(name, None, good, None, bad, n, *co) == -1
            assert raw_rc(name, None, good, None, good, I64(-1), *co) == -1
            assert raw_rc(name, None, good, None, good, n, *co) == -1
            assert raw_rc(name, None, good, None, good, I64(0), *co) == -1
            assert raw_rc(name, P, good, P, good, I64(0), *co) == 0
    # hardsigmoid bounds outside the output range
    for bits, ot, lo, hi in ((8, NPU_I8, -128, 127), (8, NPU_U8, 0, 255),
                             (16, NPU_I16, -32768, 32767), (16, NPU_U16, 0, 65535)):
        name = f"npu_hardsigmoid_a{bits}"
        it = NPU_I8 if bits == 8 else NPU_I16
        assert raw_rc(name, None, it, None, ot, I64(0), 1.0, 0.0, lo, hi) == -1
        assert raw_rc(name, P, it, P, ot, I64(0), 1.0, 0.0, lo, hi) == 0
        assert raw_rc(name, P, it, P, ot, I64(0), 1.0, 0.0, lo - 1, hi) == -1
        assert raw_rc(name, P, it, P, ot, I64(0), 1.0, 0.0, lo, hi + 1) == -1
