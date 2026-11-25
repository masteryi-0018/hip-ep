# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""reduction/reduce.h entry points against golden_reduce.py, byte for byte."""

import numpy as np
import pytest

import golden_reduce as r
from golden_float_cvt import f32_to_fp8, kSat, store_float
from golden_int_round import INT_RANGE, Rnd
from npu import I64, NPU_BF16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, NPU_I32, NPU_U8
from npu import OUT, call, raw_rc
from npu import f32 as f
from testing import F16T, rejected
from testing import INT_ALL as INT
from testing import same_bytes as assert_same

f32 = np.float32
RNG = np.random.default_rng(15114)
AXES = range(1, 8)
SHAPES = [(3, 4, 64), (2, 5, 200), (4, 3, 7), (1, 1, 129), (5, 2, 128)]
SHAPES64 = [(3, 4, 64), (2, 5, 128), (1, 1, 192), (4, 3, 64)]


def nout(shape, axis):
    return int(np.prod(r._out_shape(*shape, axis)))


def floats16(shape, t, kind):
    """16-bit patterns: 'rand', 'cancel' (large +- lanes), 'special' (inf / NaN)."""
    x = (RNG.standard_normal(shape) * 3).astype(f32)
    if kind == "cancel":
        big = RNG.integers(0, 2, shape).astype(bool)
        sign = np.where(RNG.integers(0, 2, shape) == 1, 1, -1)
        x = np.where(big, sign * f32(2 ** 20 if t == "bf16" else 2 ** 14), x * 0.01).astype(f32)
    b = store_float(x, t).astype(np.uint16)
    m = RNG.random(shape)
    sub = RNG.integers(1, 0x7F if t == "bf16" else 0x3FF, shape).astype(np.uint16)
    b = np.where(m < 0.04, 0x8000, b)
    b = np.where((m >= 0.04) & (m < 0.07), sub | np.where(m < 0.055, 0x8000, 0), b)
    if kind == "special":
        nan, inf = (0x7FC0, 0x7F80) if t == "bf16" else (0x7E00, 0x7C00)
        b = np.where(m > 0.995, nan, b)
        b = np.where((m > 0.99) & (m <= 0.995), inf | np.where(m > 0.9925, 0x8000, 0), b)
    return b.astype(np.uint16)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("kind", ["rand", "cancel", "special"])
def test_sum16(axis, t, kind):
    for shape in SHAPES:
        x = floats16(shape, t, kind)
        for scale in (1.0, f(1 / (shape[0] * 7)), f(0.3)):
            got = call("npu_reduce_sum_c64_bf16", x, F16T[t], OUT, *[I64(v) for v in shape],
                       axis, scale, out_nbytes=2 * nout(shape, axis), out_dtype=np.uint16)
            assert_same(got, r.reduce_sum_c64_bf16(x, t, axis, scale))


def test_vaddr_distinguishes():
    """The vector reduce-add differs from both an exact sum and an fp32 adder tree."""
    v = np.zeros(64, f32)
    v[:32] = f32(3 * 2.0 ** -12)
    v[0], v[2] = 2.0 ** 20, -(2.0 ** 20)
    b = store_float(v, "bf16").reshape(1, 1, 64)
    got = call("npu_reduce_sum_c64_bf16", b, NPU_BF16, OUT, I64(1), I64(1), I64(64), 1, 1.0,
               out_nbytes=2, out_dtype=np.uint16)
    assert_same(got, r.reduce_sum_c64_bf16(b, "bf16", 1, 1.0))
    lanes = v[:32]
    tree = lanes
    while tree.size > 1:
        tree = (tree[0::2] + tree[1::2]).astype(f32)
    assert int(got[0]) == int(store_float(f32(30 * 2.0 ** -10), "bf16"))
    assert int(got[0]) != int(store_float(f32(90 * 2.0 ** -12), "bf16"))
    assert int(got[0]) != int(store_float(tree[0], "bf16"))


def fp8_codes(shape, nan=False):
    c = f32_to_fp8((RNG.standard_normal(shape) * 4).astype(f32), Rnd.kHalfEven, kSat)
    m = RNG.random(shape)
    c = np.where(m < 0.03, 0x80, c)          # -0
    c = np.where((m >= 0.03) & (m < 0.06), RNG.integers(1, 8, shape) | 0x80 * (m < 0.045), c)
    c = np.where((m >= 0.06) & (m < 0.08), 0x7E | 0x80 * (m < 0.07), c)  # +-448
    if nan:
        c = np.where(m > 0.995, 0x7F, c)
    return c.astype(np.uint8)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("nan", [False, True])
def test_sum_fp8(axis, nan):
    for shape in SHAPES + [(4, 6, 70)]:
        x = fp8_codes(shape, nan)
        tiles = [(1, 1)]
        if axis == 7:
            tiles += [(h, w) for h in (1, 2, shape[0]) for w in (1, 3, shape[1])
                      if shape[0] % h == 0 and shape[1] % w == 0]
        for scale in (f(0.017857142857142856 / 0.019244817), f(1 / 3), 30.0):
            for ht, wt in tiles:
                got = call("npu_reduce_sum_c64_fp8", x, OUT, *[I64(v) for v in shape], I64(ht),
                           I64(wt), axis, scale, out_nbytes=nout(shape, axis),
                           out_dtype=np.uint8)
                assert_same(got, r.reduce_sum_c64_fp8(x, ht, wt, axis, scale))


def ints(t, shape, extreme=False):
    lo, hi = INT_RANGE[t]
    if extreme:
        return RNG.choice([lo, hi, lo + 1, hi - 1], shape).astype(INT[t][1])
    return RNG.integers(lo, hi + 1, shape, dtype=np.int64).astype(INT[t][1])


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("bits", [8, 16])
def test_sum_int(axis, bits):
    names = ("i8", "u8") if bits == 8 else ("i16", "u16")
    cmax = 127 if bits == 8 else 255
    for shape in SHAPES:
        for it in names:
            for ot in names:
                for extreme in (False, True):
                    x = ints(it, shape, extreme)
                    for c0, c1, c2 in ((1, 0, 0), (cmax, 13, -123456789), (77, 5, 3 << 33),
                                       (100, 20, 2**31 + 12345), (3, 1, -(2**40) - 7)):
                        got = call(f"npu_reduce_sum_c64_a{bits}", x, INT[it][0], OUT, INT[ot][0],
                                   *[I64(v) for v in shape], axis, c0, c1, I64(c2),
                                   out_nbytes=nout(shape, axis) * bits // 8,
                                   out_dtype=INT[ot][1])
                        want = getattr(r, f"reduce_sum_c64_a{bits}")(x, axis, ot, c0, c1, c2)
                        assert_same(got, want.astype(INT[ot][1]))


@pytest.mark.parametrize("axis", AXES)
def test_sum_int_wrap(axis):
    """Sums beyond int32 wrap (not for a16 with axis_flags 7)."""
    shape = (64, 64, 128)
    for bits, t in ((8, "u8"), (16, "u16"), (16, "i16")):
        x = np.full(shape, INT_RANGE[t][1], INT[t][1])
        c0 = 127 if bits == 8 else 255
        got = call(f"npu_reduce_sum_c64_a{bits}", x, INT[t][0], OUT, INT[t][0],
                   *[I64(v) for v in shape], axis, c0, 30, I64(-5),
                   out_nbytes=nout(shape, axis) * bits // 8, out_dtype=INT[t][1])
        want = getattr(r, f"reduce_sum_c64_a{bits}")(x, axis, t, c0, 30, -5)
        assert_same(got, want.astype(INT[t][1]))


@pytest.mark.parametrize("axis", AXES)
def test_sum_a32(axis):
    for shape in SHAPES:
        for it in ("i32", "u32"):
            for ot in ("i32", "u32"):
                for scale in ("small", "big"):
                    x = ints(it, shape) if scale == "big" else \
                        RNG.integers(-1000, 1000, shape).astype(INT[it][1])
                    for c0, c1, sh in ((0.0, 1.0, 0), (f(17.25), f(1 / 3), 0), (f(-3.5), f(0.7), 2),
                                       (f(1e9), f(3e5), 1), (f(0.5), f(0.37), -1),
                                       (f(-7.5), f(2.3), -3), (f(0.25), f(0.9), 31)):
                        got = call("npu_qdq_reduce_sum_c64_a32", x, INT[it][0], OUT, INT[ot][0],
                                   *[I64(v) for v in shape], axis, c0, c1, sh,
                                   out_nbytes=4 * nout(shape, axis), out_dtype=INT[ot][1])
                        want = r.qdq_reduce_sum_c64_a32(x, it, axis, ot, c0, c1, sh)
                        assert_same(got, want.astype(INT[ot][1]))


def prod_inputs(shape, t, kind):
    """Values near 1, plus +-0 / subnormal / inf / NaN ('special') or wide range ('wide')."""
    x = (1 + RNG.standard_normal(shape) * 0.2).astype(f32)
    x = np.where(RNG.random(shape) < 0.5, -x, x).astype(f32)
    b = store_float(x, t)
    if kind == "special":
        m = RNG.random(shape)
        b = np.where(m < 0.01, 0x8000, b)
        b = np.where((m > 0.01) & (m < 0.02), 0x0003, b)
        nan, inf = (0x7FC0, 0x7F80) if t == "bf16" else (0x7E00, 0x7C00)
        b = np.where(m > 0.995, nan, b)
        b = np.where((m > 0.99) & (m <= 0.995), inf, b)
    if kind == "wide":   # overflow / underflow of the running product
        b = store_float((RNG.standard_normal(shape) * 300).astype(f32), t)
    return b.astype(np.uint16)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("kind", ["rand", "special", "wide"])
def test_prod16(axis, t, kind):
    for shape in SHAPES64:
        x = prod_inputs(shape, t, kind)
        got = call("npu_reduce_prod_bf16", x, F16T[t], OUT, *[I64(v) for v in shape], axis,
                   out_nbytes=2 * nout(shape, axis), out_dtype=np.uint16)
        assert_same(got, r.reduce_prod_bf16(x, t, axis))


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("kind", ["rand", "cancel", "special"])
def test_l2_16(axis, t, kind):
    for shape in SHAPES64:
        x = floats16(shape, t, kind)
        got = call("npu_reduce_l2_bf16", x, F16T[t], OUT, *[I64(v) for v in shape], axis,
                   out_nbytes=2 * nout(shape, axis), out_dtype=np.uint16)
        assert_same(got, r.reduce_l2_bf16(x, t, axis))


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_sqrt16_exhaustive(t):
    x = np.arange(1 << 16, dtype=np.uint16)
    got = call("npu_test_nlf16", {"bf16": 0, "f16": 1}[t], x, OUT, I64(len(x)),
               out_nbytes=2 * len(x), out_dtype=np.uint16)
    assert_same(got, r.hw_sqrt16(x, t))


# (dq_zp, dq_sc, q_zp, q_sc)
QDQ = [(2.0, 0.01, 4.0, f(1 / 1.319809797)), (-2.0, f(0.0003051515990363106), 17.0, f(1 / 0.0074)),
       (0.0, 0.5, 0.0, 1.0), (128.0, f(0.0371), 100.0, f(1 / 0.29))]


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("op", ["prod", "l2"])
@pytest.mark.parametrize("bits", [8, 16])
def test_qdq_prod_l2(axis, op, bits):
    names = ("i8", "u8") if bits == 8 else ("i16", "u16")
    for shape in SHAPES64[:3]:
        for it in names:
            x = ints(it, shape)
            for rt in ("bf16",) if op == "prod" else ("bf16", "f16"):
                rta, rtm = ((), ()) if op == "prod" else ((F16T[rt],), (rt,))
                for ot in names:
                    for dq_z, dq_s, q_z, q_s in QDQ:
                        if op == "prod" and dq_s > 0.02:
                            continue  # products of large values only saturate
                        got = call(f"npu_qdq_reduce_{op}_a{bits}", x, INT[it][0], OUT,
                                   INT[ot][0], *[I64(v) for v in shape], axis, *rta,
                                   dq_z, dq_s, q_z, q_s,
                                   out_nbytes=nout(shape, axis) * bits // 8,
                                   out_dtype=INT[ot][1])
                        fn = getattr(r, f"qdq_reduce_{op}_a{bits}")
                        want = fn(x, axis, *rtm, dq_z, dq_s, q_z, q_s, ot)
                        assert_same(got, want.astype(INT[ot][1]))


MRS_SHAPES = [((3, 4, 64), (3, 4, 64)), ((2, 5, 100), (1, 5, 100)), ((4, 3, 70), (4, 3, 1)),
              ((2, 3, 128), (2, 1, 128)), ((3, 2, 33), (1, 1, 1))]
MRS_C = [(f(0.0004 * 4.8e-6 / 0.0004), f(13 * 4.8e-6), f(-24 * 4.8e-6), 22.0),
         (f(0.3), f(-1.7), f(2.9), f(5.5)), (f(1 / 3), 0.0, 0.0, 0.0),
         (f(0.01), f(-0.13), f(0.07), f(1e4 / 3))]


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("bits", [8, 16])
def test_mul_reducesum(axis, bits):
    names = ("i8", "u8") if bits == 8 else ("i16", "u16")
    for s1, s2 in MRS_SHAPES:
        for it in names:
            a, b = ints(it, s1), ints(it, s2)
            for ot in names:
                for c in MRS_C:
                    got = call(f"npu_qdq_elementwise_broadcasting3d_mul_reducesum_a{bits}", a, b,
                               INT[it][0], OUT, INT[ot][0], *[I64(v) for v in s1 + s2], axis,
                               *c, out_nbytes=nout(s1, axis) * bits // 8, out_dtype=INT[ot][1])
                    fn = getattr(r, f"qdq_elementwise_broadcasting3d_mul_reducesum_a{bits}")
                    want = fn(a, b, axis, c, ot)
                    assert_same(got, want.astype(INT[ot][1]))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("exclusive", [0, 1])
@pytest.mark.parametrize("reverse", [0, 1])
def test_cumsum(axis, bits, exclusive, reverse):
    names = ("i8", "u8") if bits == 8 else ("i16", "u16")
    for shape in ((7, 130), (40, 64), (3, 1)):
        for it in names:
            x = ints(it, shape)
            for ot in names:
                for c0, c1 in ((0.0, 1.0), (f(3.7), f(0.0625)), (f(-12.25), f(1 / 3)),
                               (f(100.5), f(0.0123))):
                    got = call("npu_qdq_cumsum", x, INT[it][0], OUT, INT[ot][0], I64(shape[0]),
                               I64(shape[1]), axis, exclusive, reverse, c0, c1,
                               out_nbytes=x.size * bits // 8, out_dtype=INT[ot][1])
                    want = r.qdq_cumsum(x, axis, exclusive, reverse, c0, c1, ot)
                    assert_same(got, want.astype(INT[ot][1]))


def test_cumsum_wrap():
    x = np.full((70000, 1), 32767, np.int16)
    got = call("npu_qdq_cumsum", x, NPU_I16, OUT, NPU_I16, I64(70000), I64(1), 0, 0, 0,
               0.0, f(1 / 65536), out_nbytes=x.nbytes, out_dtype=np.int16)
    assert_same(got, r.qdq_cumsum(x, 0, 0, 0, 0.0, f(1 / 65536), "i16").astype(np.int16))


ENTRIES = ["npu_reduce_sum_c64_bf16", "npu_reduce_sum_c64_fp8", "npu_reduce_sum_c64_a8",
           "npu_reduce_sum_c64_a16", "npu_qdq_reduce_sum_c64_a32", "npu_reduce_prod_bf16",
           "npu_reduce_l2_bf16", "npu_qdq_reduce_prod_a8", "npu_qdq_reduce_prod_a16",
           "npu_qdq_reduce_l2_a8", "npu_qdq_reduce_l2_a16",
           "npu_qdq_elementwise_broadcasting3d_mul_reducesum_a8",
           "npu_qdq_elementwise_broadcasting3d_mul_reducesum_a16", "npu_qdq_cumsum"]


def test_bad_args():
    """Every entry rejects an argument its kernel cannot run (each call otherwise valid)."""
    s = (I64(2), I64(2), I64(64))
    x8, x16, x32 = (np.zeros((2, 2, 64), d) for d in (np.int8, np.int16, np.int32))
    q = (0.0, 1.0, 0.0, 1.0)
    mrs = "npu_qdq_elementwise_broadcasting3d_mul_reducesum_a"
    bad = [
        ("npu_reduce_sum_c64_bf16", x16, NPU_I8, OUT, *s, 1, 1.0),
        ("npu_reduce_sum_c64_bf16", x16, NPU_BF16, OUT, *s, 0, 1.0),
        ("npu_reduce_sum_c64_bf16", x16, NPU_BF16, OUT, *s, 8, 1.0),
        ("npu_reduce_sum_c64_bf16", None, NPU_BF16, OUT, *s, 1, 1.0),
        ("npu_reduce_sum_c64_fp8", x8, OUT, *s, I64(3), I64(1), 7, 1.0),
        ("npu_reduce_sum_c64_fp8", x8, OUT, I64(2), I64(0), I64(64), I64(1), I64(1), 2, 1.0),
        ("npu_reduce_sum_c64_a8", x8, NPU_I8, OUT, NPU_I16, *s, 1, 1, 0, I64(0)),
        ("npu_reduce_sum_c64_a8", x8, NPU_I8, OUT, NPU_I8, *s, 1, 128, 0, I64(0)),
        ("npu_reduce_sum_c64_a16", x16, NPU_I16, OUT, NPU_I16, *s, 1, 1, 63, I64(0)),
        ("npu_reduce_sum_c64_a16", x16, NPU_I16, OUT, NPU_I16, *s, 1, 256, 0, I64(0)),
        ("npu_qdq_reduce_sum_c64_a32", x32, NPU_I32, OUT, NPU_I32, *s, 1, 0.0, 1.0, 32),
        ("npu_qdq_reduce_sum_c64_a32", x32, NPU_I32, OUT, NPU_I32, *s, 1, 0.0, 1.0, -32),
        ("npu_qdq_reduce_sum_c64_a32", x32, NPU_I32, OUT, NPU_I16, *s, 1, 0.0, 1.0, 0),
        ("npu_qdq_reduce_sum_c64_a32", x32, NPU_I16, OUT, NPU_I32, *s, 1, 0.0, 1.0, 0),
        ("npu_reduce_prod_bf16", x16, NPU_BF16, OUT, I64(1), I64(1), I64(65), 1),
        ("npu_reduce_l2_bf16", x16, NPU_U8, OUT, *s, 1),
        ("npu_qdq_reduce_prod_a8", x8, NPU_I16, OUT, NPU_I8, *s, 1, *q),
        ("npu_qdq_reduce_prod_a16", x16, NPU_I16, OUT, NPU_I16, I64(2), I64(2), I64(32), 1, *q),
        ("npu_qdq_reduce_l2_a8", x8, NPU_I8, OUT, NPU_I8, *s, 1, NPU_F32, *q),
        ("npu_qdq_reduce_l2_a16", x16, NPU_I16, OUT, NPU_U8, *s, 1, NPU_BF16, *q),
        (mrs + "8", x8, x8, NPU_I8, OUT, NPU_I8, *s, I64(2), I64(3), I64(64), 1, *q),
        (mrs + "16", x16, None, NPU_I16, OUT, NPU_I16, *s, *s, 1, *q),
        (mrs + "16", x16, x16, NPU_I8, OUT, NPU_I16, *s, *s, 1, *q),
        ("npu_qdq_cumsum", x8, NPU_I8, OUT, NPU_I16, I64(2), I64(2), 0, 0, 0, 0.0, 1.0),
        ("npu_qdq_cumsum", x8, NPU_I8, OUT, NPU_I8, I64(2), I64(2), 2, 0, 0, 0.0, 1.0),
        # negative sizes; a reduced dimension needs an element; a null pointer is rejected even
        # when there is no output
        ("npu_reduce_sum_c64_bf16", x16, NPU_BF16, OUT, I64(-2), I64(2), I64(64), 1, 1.0),
        ("npu_reduce_prod_bf16", x16, NPU_BF16, OUT, I64(2), I64(2), I64(0), 1),
        ("npu_qdq_reduce_l2_a8", None, NPU_I8, OUT, NPU_I8, I64(0), I64(2), I64(64), 1,
         NPU_BF16, *q),
        ("npu_qdq_cumsum", x8, NPU_I8, OUT, NPU_I8, I64(0), I64(2), 0, 0, 0, 0.0, 1.0),
        ("npu_qdq_cumsum", x8, NPU_I8, None, NPU_I8, I64(2), I64(0), 0, 0, 0, 0.0, 1.0),
    ]
    for name, *args in bad:
        assert rejected(name, *args, nbytes=1024), (name, args)
    covered = {b[0] for b in bad}
    assert covered == set(ENTRIES), set(ENTRIES) - covered
    # an empty kept dimension: no output element, nothing launched, success
    assert raw_rc("npu_reduce_sum_c64_bf16", x16, NPU_BF16, x16, I64(0), I64(2), I64(64), 1,
                  1.0) == 0
    assert raw_rc("npu_reduce_sum_c64_a8", x8, NPU_I8, x8, NPU_I8, I64(2), I64(2), I64(0), 6, 1,
                  0, I64(0)) == 0
    assert raw_rc("npu_qdq_reduce_l2_a8", x8, NPU_I8, x8, NPU_I8, I64(2), I64(0), I64(64), 1,
                  NPU_BF16, *q) == 0
    assert raw_rc("npu_qdq_cumsum", x8, NPU_I8, x8, NPU_I8, I64(2), I64(0), 0, 0, 0, 0.0,
                  1.0) == 0
    # product / L2 keep whole 64-channel groups, at least one, also when C is kept
    assert raw_rc("npu_reduce_prod_bf16", x16, NPU_BF16, x16, I64(2), I64(2), I64(0), 6) == -1
