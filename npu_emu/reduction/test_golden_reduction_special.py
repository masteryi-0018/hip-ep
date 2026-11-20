# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Special values (NaN, infinities, +-0, subnormals, extremes) through every reduction entry,
compared byte for byte with the numpy references."""

import numpy as np
import pytest

import golden_avgpool as ga
import golden_groupnorm as gg
import golden_layernorm as gl
import golden_reduce as gr
import golden_softmax as gs
from golden_block_fp import encode_bfp16
from golden_float_cvt import store_float
from golden_int_round import INT_RANGE, Rnd
from npu import (I64, NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT,
                 OUT2, U32, call)
from npu import f32 as f
from testing import F16T, INT, call_n
from testing import same_bytes as same

f32 = np.float32
RNG = np.random.default_rng(20261119)
INF, NAN = float("inf"), float("nan")

# Special codes: +-0, smallest / largest subnormal, smallest normal, +-max, +-inf, quiet and
# signalling NaN of both signs, +-1.
SP = {
    "bf16": [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80,
             0x7FC0, 0xFFC0, 0x7F81, 0xFFA1, 0x3F80, 0xBF80],
    "f16": [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00,
            0x7E00, 0xFE00, 0x7C01, 0xFD55, 0x3C00, 0xBC00],
    "fp8": [0x00, 0x80, 0x01, 0x81, 0x07, 0x08, 0x7E, 0xFE, 0x7F, 0xFF, 0x38, 0xB8],
}
STORE = {"bf16": np.uint16, "f16": np.uint16, "fp8": np.uint8}
CODE = {"bf16": NPU_BF16, "f16": NPU_F16, "fp8": NPU_FP8, "f32": NPU_F32}
COEF = [0.0, -0.0, INF, -INF, NAN, 1e30, -1e30, 3.4e38, 1e-40, f(1 / 3)]


def base_values(t, shape, sd=2.0):
    v = (RNG.standard_normal(shape) * sd).astype(f32)
    if t == "fp8":
        v = np.clip(v, -448, 448)
    return gg.encode(v, t) if t != "f32" else v


def sprinkle(t, R, shape_tail, sd=2.0):
    """[R] + shape_tail codes of type t: the first rows are one special each, then rows with
    two specials in random places, then rows of mostly specials."""
    sp = np.array(SP[t], STORE[t])
    x = base_values(t, (R,) + tuple(shape_tail), sd).copy()
    flat = x.reshape(R, -1)
    n = flat.shape[1]
    k = len(sp)
    for r in range(R):
        if r < k:
            flat[r] = sp[r]
        elif r < 3 * k:
            flat[r, RNG.integers(0, n, 2)] = sp[[r % k, (r * 7 + 3) % k]]
        elif r < 4 * k:
            m = RNG.random(n) < 0.3
            flat[r, m] = sp[RNG.integers(0, k, int(m.sum()))]
        else:
            m = RNG.random(n) < 0.05
            flat[r, m] = sp[RNG.integers(0, k, int(m.sum()))]
    return x


def nrows(t):
    return 5 * len(SP[t])


# ---------------------------------------------------------------------------- reduce.h

def nout(shape, axis):
    return int(np.prod(gr.out_shape(*shape, axis)))


R_SHAPES = [(3, 4, 64), (2, 3, 70), (1, 1, 130)]
R_AXES = range(1, 8)


@pytest.mark.parametrize("axis", R_AXES)
@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_golden_special_sum16(axis, t):
    for shape in R_SHAPES:
        H, W, C = shape
        xs = sprinkle(t, H * 8, shape[1:]).reshape(8, H, W, C)
        for scale in (1.0, f(1 / 7), 0.0, -0.0, INF, NAN, 1e30):
            for blk in range(8):
                got = call("npu_reduce_sum_c64_bf16", xs[blk], F16T[t], OUT,
                           *[I64(v) for v in shape], axis, scale,
                           out_nbytes=2 * nout(shape, axis), out_dtype=np.uint16)
                same(got, gr.sum16(xs[blk], t, axis, scale))


@pytest.mark.parametrize("axis", R_AXES)
def test_golden_special_sum_fp8(axis):
    for shape in R_SHAPES:
        H, W, C = shape
        x = sprinkle("fp8", H * 6, shape[1:]).reshape(6, H, W, C)
        tiles = [(1, 1)] + ([(H, 1)] if axis == 7 else [])
        for scale in (1.0, f(1 / 3), 0.0, -0.0, INF, NAN, 1e30, -1e-30):
            for blk in range(6):
                for ht, wt in tiles:
                    got = call("npu_reduce_sum_c64_fp8", x[blk], OUT, *[I64(v) for v in shape],
                               axis, scale, I64(ht), I64(wt), out_nbytes=nout(shape, axis),
                               out_dtype=np.uint8)
                    same(got, gr.sum_fp8(x[blk], axis, scale, ht, wt))


def int_extremes(t, shape):
    lo, hi = INT_RANGE[t]
    return RNG.choice([lo, hi, lo + 1, hi - 1, 0, -1 if lo < 0 else 1], shape).astype(INT[t][1])


@pytest.mark.parametrize("axis", R_AXES)
@pytest.mark.parametrize("bits", [8, 16])
def test_golden_special_sum_int(axis, bits):
    """Extreme sums and offsets: the int64 affine map must wrap, not overflow."""
    names = ("i8", "u8") if bits == 8 else ("i16", "u16")
    cmax = 127 if bits == 8 else 255
    i64 = (1 << 63) - 1
    for shape in R_SHAPES[:2]:
        for it in names:
            for ot in names:
                x = int_extremes(it, shape)
                for c0, c1, c2 in ((cmax, 0, i64), (cmax, 0, -i64 - 1), (cmax, 62, i64),
                                   (cmax, 62, -i64 - 1), (0, 0, i64), (1, 0, 1 << 62),
                                   (cmax, 1, -(1 << 62)), (cmax, 0, 0)):
                    got = call(f"npu_reduce_sum_c64_a{bits}", x, INT[it][0], OUT, INT[ot][0],
                               *[I64(v) for v in shape], axis, c0, c1, I64(c2),
                               out_nbytes=nout(shape, axis) * bits // 8, out_dtype=INT[ot][1])
                    want = gr.sum_int(x, axis, f"a{bits}", it, ot, c0, c1, c2)
                    same(got, want.astype(INT[ot][1]))


@pytest.mark.parametrize("axis", R_AXES)
def test_golden_special_sum_a32(axis):
    for shape in R_SHAPES[:2]:
        for it in ("i32", "u32"):
            lo, hi = (-2**31, 2**31 - 1) if it == "i32" else (0, 2**32 - 1)
            x = RNG.choice([lo, hi, lo + 1, hi - 1, 0, 1], shape).astype(INT_ALL[it])
            for ot in ("i32", "u32"):
                for c0, c1 in ((0.0, INF), (INF, 0.0), (NAN, 1.0), (1.0, NAN), (-INF, INF),
                               (1e30, 1e30), (-1e30, 3e38), (-0.0, -0.0), (0.0, 1.0)):
                    for sh in (-31, 0, 31):
                        got = call("npu_qdq_reduce_sum_c64_a32", x, INT_NPU[it], OUT, INT_NPU[ot],
                                   *[I64(v) for v in shape], axis, c0, c1, sh,
                                   out_nbytes=4 * nout(shape, axis), out_dtype=INT_ALL[ot])
                        want = gr.sum_int(x, axis, "a32", it, ot, f0=c0, f1=c1, shift=sh)
                        same(got, want.astype(INT_ALL[ot]))


INT_ALL = {"i32": np.int32, "u32": np.uint32}
INT_NPU = {"i32": 8, "u32": 9}


@pytest.mark.parametrize("axis", R_AXES)
@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_golden_special_prod_l2(axis, t):
    for shape in [(3, 4, 64), (2, 3, 128)]:
        H, W, C = shape
        x = sprinkle(t, H * 8, shape[1:]).reshape(8, H, W, C)
        for blk in range(8):
            got = call("npu_reduce_prod_bf16", x[blk], F16T[t], OUT, *[I64(v) for v in shape],
                       axis, out_nbytes=2 * nout(shape, axis), out_dtype=np.uint16)
            same(got, gr.prod_bits(x[blk], t, axis))
            got = call("npu_reduce_l2_bf16", x[blk], F16T[t], OUT, *[I64(v) for v in shape],
                       axis, out_nbytes=2 * nout(shape, axis), out_dtype=np.uint16)
            same(got, gr.l2_bits(x[blk], t, axis))


def test_golden_special_prod_overflow():
    """Running products that overflow, underflow, and meet 0 * inf."""
    for t in ("bf16", "f16"):
        big = {"bf16": 0x7F00, "f16": 0x7800}[t]
        for fill in ([big, 0x0000], [big, 0x7F80 if t == "bf16" else 0x7C00],
                     [big, 0x0001], [big], [0x0001, 0x0001], [0x7F80 if t == "bf16" else 0x7C00,
                                                              0x8000]):
            for axis in (1, 4, 7):
                x = RNG.choice(np.array(fill, np.uint16), (2, 3, 64))
                got = call("npu_reduce_prod_bf16", x, F16T[t], OUT, I64(2), I64(3), I64(64),
                           axis, out_nbytes=2 * nout((2, 3, 64), axis), out_dtype=np.uint16)
                same(got, gr.prod_bits(x, t, axis))


QDQ_SPECIAL = [
    (0.0, 0.0, 1.0, 0.0), (INF, 0.0, 1.0, 0.0), (NAN, 0.0, 1.0, 0.0), (1e30, 0.0, 1.0, 0.0),
    (f(0.01), INF, 1.0, 0.0), (f(0.01), NAN, 1.0, 0.0), (f(0.01), 1e30, 1.0, 0.0),
    (f(0.01), 2.0, 0.0, 0.0), (f(0.01), 2.0, INF, 0.0), (f(0.01), 2.0, NAN, 0.0),
    (f(0.01), 2.0, 1e30, 0.0), (f(0.01), 2.0, 1.0, INF), (f(0.01), 2.0, 1.0, -INF),
    (f(0.01), 2.0, 1.0, NAN), (f(0.01), 2.0, 1.0, 3e9), (f(0.01), 2.0, -INF, INF),
]


@pytest.mark.parametrize("axis", [1, 4, 6, 7])
@pytest.mark.parametrize("op", ["prod", "l2"])
@pytest.mark.parametrize("bits", [8, 16])
def test_golden_special_qdq_prod_l2(axis, op, bits):
    names = ("i8", "u8") if bits == 8 else ("i16", "u16")
    shape = (3, 2, 64)
    for it in names:
        x = int_extremes(it, shape)
        for rt in ("bf16", "f16"):
            for ot in names:
                for dq_s, dq_z, q_s, q_z in QDQ_SPECIAL:
                    got = call(f"npu_qdq_reduce_{op}_a{bits}", x, INT[it][0], OUT, INT[ot][0],
                               *[I64(v) for v in shape], axis, F16T[rt], dq_s, dq_z, q_s, q_z,
                               out_nbytes=nout(shape, axis) * bits // 8, out_dtype=INT[ot][1])
                    fn = gr.qdq_prod if op == "prod" else gr.qdq_l2
                    same(got, fn(x, axis, rt, dq_s, dq_z, q_s, q_z, ot).astype(INT[ot][1]))


@pytest.mark.parametrize("axis", [1, 4, 6, 7])
@pytest.mark.parametrize("bits", [8, 16])
def test_golden_special_mul_reducesum(axis, bits):
    names = ("i8", "u8") if bits == 8 else ("i16", "u16")
    s1, s2 = (3, 4, 64), (3, 1, 64)
    for it in names:
        a, b = int_extremes(it, s1), int_extremes(it, s2)
        for ot in names:
            for c in ((1.0, 0.0, 0.0, 0.0), (INF, 0.0, 0.0, 0.0), (NAN, 0.0, 0.0, 0.0),
                      (0.0, INF, 0.0, 0.0), (0.0, 0.0, NAN, 0.0), (0.0, 0.0, 0.0, INF),
                      (0.0, 0.0, 0.0, NAN), (1e30, 1e30, 1e30, -1e30), (-INF, 1.0, INF, 0.0),
                      (3e38, 3e38, 3e38, 0.0), (-0.0, -0.0, -0.0, -0.0), (0.0, 0.0, 0.0, 3e9)):
                got = call(f"npu_qdq_elementwise_broadcasting3d_mul_reducesum_a{bits}", a, b,
                           INT[it][0], OUT, INT[ot][0], *[I64(v) for v in s1 + s2], axis, *c,
                           out_nbytes=nout(s1, axis) * bits // 8, out_dtype=INT[ot][1])
                same(got, gr.mul_reducesum(a, b, axis, c, ot, bits == 16).astype(INT[ot][1]))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("excl,rev", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_golden_special_cumsum(axis, bits, excl, rev):
    names = ("i8", "u8") if bits == 8 else ("i16", "u16")
    for it in names:
        x = int_extremes(it, (9, 70))
        for ot in names:
            for c0, c1 in ((0.0, INF), (INF, 0.0), (NAN, 1.0), (1.0, NAN), (-INF, INF),
                           (1e30, 1e30), (-0.0, -0.0), (0.0, 0.0), (0.0, 3e38), (3e38, 1.0)):
                got = call("npu_qdq_cumsum", x, INT[it][0], OUT, INT[ot][0], I64(9), I64(70),
                           axis, excl, rev, c0, c1, out_nbytes=x.size * bits // 8,
                           out_dtype=INT[ot][1])
                same(got, gr.cumsum(x, axis, excl, rev, c0, c1, ot).astype(INT[ot][1]))


# ---------------------------------------------------------------------------- softmax.h

def mask_bytes(R, C, kind):
    if kind is None:
        return None, np.ones((R, C), bool)
    if kind == "empty_rows":  # some rows with no included element, some with one
        m = RNG.random((R, C)) < 0.6
        m[::3] = False
        m[1::3, RNG.integers(0, C)] = True
    else:
        m = RNG.random((R, C)) < 0.6
        m[:, 0] = True
    return np.packbits(m, axis=1, bitorder="little"), m


@pytest.mark.parametrize("C", [5, 64, 100])
def test_golden_special_softmax_bf16(C):
    R = nrows("bf16")
    x = sprinkle("bf16", R, (C,))
    got = call_n("npu_softmax_bf16", x, OUT, I64(R), I64(C), n=(R, C), dt=np.uint16)
    same(got, gs.softmax_bf16(x))


@pytest.mark.parametrize("C,cc,nc", [(5, 128, 1), (100, 128, 3), (300, 128, 3)])
def test_golden_special_softmax_2pass(C, cc, nc):
    R = nrows("bf16")
    x = sprinkle("bf16", R, (C,))
    got = call_n("npu_softmax_2pass_bf16", x, OUT, I64(R), I64(C), I64(cc), I64(nc), n=(R, C),
                 dt=np.uint16)
    same(got, gs.softmax_2pass_bf16(x, cc, nc))


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("C", [32, 96])
@pytest.mark.parametrize("mk", [None, "random", "empty_rows"])
def test_golden_special_softmax_fp16x16(t, C, mk):
    R = nrows(t)
    x = sprinkle(t, R, (C,))
    mb, m = mask_bytes(R, C, mk)
    got = call_n("npu_softmax_fp16x16", x, mb, OUT, F16T[t], I64(R), I64(C), n=(R, C),
                 dt=np.uint16)
    same(got, gs.softmax_fp16x16(x, m, t))


@pytest.mark.parametrize("C", [64, 128])
@pytest.mark.parametrize("mk", [None, "random", "empty_rows"])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.37, 1 / 0.013), (0.0, 1.0), (INF, 1.0),
                                    (NAN, 1.0), (1.0, 0.0), (1.0, INF), (1.0, NAN),
                                    (1e30, 1e30), (-1.0, -1.0), (1e-40, 1e-40)])
def test_golden_special_softmax_fp8(C, mk, scales):
    R = nrows("fp8")
    x = sprinkle("fp8", R, (C,))
    mb, m = mask_bytes(R, C, mk)
    s, r = (float(f32(v)) for v in scales)
    got = call_n("npu_softmax_fp8", x, mb, OUT, I64(R), I64(C), s, r, n=(R, C), dt=np.uint8)
    same(got, gs.softmax_fp8(x, m, s, r))


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("C", [32, 64, 130, 192])
def test_golden_special_logsoftmax(t, C):
    R = nrows(t)
    x = sprinkle(t, R, (C,))
    got = call_n("npu_logsoftmax_bf16", x, OUT, F16T[t], I64(R), I64(C), n=(R, C), dt=np.uint16)
    same(got, gs.logsoftmax_bf16(x, t))


@pytest.mark.parametrize("C", [32, 64, 130])
@pytest.mark.parametrize("scales", [(1.0, 4.0), (0.37, 1 / 0.11), (INF, 1.0), (1e30, 1.0),
                                    (1e-40, 1.0), (1.0, 0.0), (1.0, INF), (1.0, NAN),
                                    (1.0, -1.0), (1e30, 1e30)])
def test_golden_special_logsoftmax_fp8(C, scales):
    R = nrows("fp8")
    x = sprinkle("fp8", R, (C,))
    s, r = (float(f32(v)) for v in scales)
    got = call_n("npu_logsoftmax_fp8", x, OUT, I64(R), I64(C), s, r, n=(R, C), dt=np.uint8)
    same(got, gs.logsoftmax_fp8(x, s, r))


def int_rows(t, R, C):
    lo, hi = INT_RANGE[t]
    return RNG.choice([lo, hi, lo + 1, hi - 1, 0, (lo + hi) // 2], (R, C)).astype(INT[t][1])


QS_COEFFS = [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (INF, 1.0, 0.0), (NAN, 1.0, 0.0),
             (-1.0, 1.0, 0.0), (1e30, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, INF, 0.0),
             (1.0, NAN, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, INF), (1.0, 1.0, NAN),
            (1.0, 1.0, -INF), (1.0, 1.0, 3e9), (1.0, 1e-40, 0.0), (1e-40, 1.0, 0.0)]


@pytest.mark.parametrize("t", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("path,C", [(0, 64), (1, 32), (2, 32)])
@pytest.mark.parametrize("mk", [None, "empty_rows"])
@pytest.mark.parametrize("shift_res", [-32, 0, 32])
def test_golden_special_qdq_softmax(t, path, C, mk, shift_res):
    bits = 8 if t in ("i8", "u8") else 16
    R = 12
    x = int_rows(t, R, C)
    mb, m = mask_bytes(R, C, mk)
    for il, os_, zp in QS_COEFFS:
        got = call_n(f"npu_qdq_softmax_a{bits}", x, mb, OUT, INT[t][0], I64(R), I64(C), path,
                     shift_res, il, os_, zp, n=(R, C), dt=INT[t][1])
        same(got, gs.qdq_softmax(x, m, t, path, shift_res, il, os_, zp).astype(INT[t][1]))


@pytest.mark.parametrize("ti,to", [("i8", "i8"), ("u8", "u8"), ("i16", "u16"), ("u16", "i16")])
@pytest.mark.parametrize("mk", [None, "empty_rows"])
def test_golden_special_qdq_logsoftmax(ti, to, mk):
    bits = 8 if ti in ("i8", "u8") else 16
    R, C = 12, 128
    x = int_rows(ti, R, C)
    mb, m = mask_bytes(R, C, mk)
    for s_in, s_out, zp in ((1.0, 1.0, 0), (0.0, 1.0, 0), (INF, 1.0, 0), (NAN, 1.0, 0),
                            (-1.0, 1.0, 0), (1e30, 1.0, 0), (1.0, 0.0, 0), (1.0, INF, 0),
                            (1.0, NAN, 0), (1.0, -1.0, 0), (1.0, 1e-40, 0), (1e-40, 1.0, 0),
                            (1.0, 1.0, INT_RANGE[to][1]), (1.0, 1.0, INT_RANGE[to][0])):
        got = call_n(f"npu_qdq_logsoftmax_a{bits}", x, INT[ti][0], mb, OUT, INT[to][0], I64(R),
                     I64(C), s_in, s_out, zp, n=(R, C), dt=INT[to][1])
        same(got, gs.qdq_logsoftmax(x, m, ti, to, s_in, s_out, zp).astype(INT[to][1]))


# ---------------------------------------------------------------------------- helpers

def dense(t, shape, p=0.25):
    """Random values of type t with a fraction p replaced by specials."""
    if t == "f32":
        return wsp_f32(int(np.prod(shape)), p).reshape(shape)
    sp = np.array(SP[t], STORE[t])
    x = base_values(t, shape).copy()
    m = RNG.random(shape) < p
    x[m] = sp[RNG.integers(0, len(sp), int(m.sum()))]
    return x


def wsp_f32(n, p):
    v = RNG.uniform(-3, 3, n).astype(f32)
    sp = np.array([0.0, -0.0, INF, -INF, NAN, 1e-40, 3e38], f32)
    m = RNG.random(n) < p
    v[m] = sp[RNG.integers(0, len(sp), int(m.sum()))]
    return v


def wsp(t, n, p=0.12):
    """Per-channel weights of type t (f32: float32) with some specials."""
    if t == "f32":
        return wsp_f32(n, p)
    return dense(t, (n,), p)


def fdec(t, bits):
    return gg.decode(bits, t) if t != "f32" else bits


# ---------------------------------------------------------------------------- layernorm.h

def lrun(name, *args, n, t):
    dt = {"bf16": np.uint16, "f16": np.uint16, "fp8": np.uint8}.get(t) or INT[t][1]
    return call_n(name, *args, n=n, dt=dt)


@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("C", [64, 96])
def test_golden_special_layer_norm(t, C):
    if t == "fp8" and C % 64:
        pytest.skip("fp8 C % 64")
    R = nrows(t)
    x = sprinkle(t, R, (C,))
    wt = "f16" if t == "fp8" else t
    for al, be in ((wsp(wt, C, 0.0), wsp(wt, C, 0.0)), (wsp(wt, C), wsp(wt, C))):
        got = lrun("npu_layer_norm_bf16", x, CODE[t], al, be, OUT, I64(R), C, 0, 1e-12,
                   n=(R, C), t=t)
        want = gl.core_layernorm(gl.decode(x, t), gl.decode(al, wt), gl.decode(be, wt),
                                 "f32" if t == "fp8" else t, "f32")
        same(got, gl.encode(want, t))


@pytest.mark.parametrize("eps", [0.0, -1.0, INF, NAN, 1e-5, -0.0])
@pytest.mark.parametrize("c_real", [1, 40, 64])
def test_golden_special_layer_norm_real(eps, c_real):
    C, R = 64, nrows("bf16")
    x = sprinkle("bf16", R, (C,))
    al, be = wsp("bf16", C, 0.0), wsp("bf16", C)
    got = lrun("npu_layer_norm_bf16", x, NPU_BF16, al, be, OUT, I64(R), C, c_real, eps,
               n=(R, C), t="bf16")
    want = gl.core_layernorm(gl.decode(x, "bf16"), gl.decode(al, "bf16"), gl.decode(be, "bf16"),
                             "bf16", "f32", c_real, f32(eps))
    same(got, gl.encode(want, "bf16"))


@pytest.mark.parametrize("C", [64, 128])
@pytest.mark.parametrize("io", ["bfp16", "bf16_bfp16"])
def test_golden_special_layer_norm_bfp16(C, io):
    R = nrows("bf16")
    xb = sprinkle("bf16", R, (C,))
    xf = gl.decode(xb, "bf16")
    al, be = wsp("bf16", C), wsp("bf16", C)
    if io == "bfp16":
        x = encode_bfp16(xf.reshape(-1), Rnd.kHalfEven)
        got = lrun("npu_layer_norm_bfp16", x, al, be, OUT, I64(R), C, n=R * C // 8 * 9, t="fp8")
        xd = gl.decode(x, "bfp16").reshape(R, C)
    else:
        x = xb
        got = lrun("npu_layer_norm_bf16_bfp16", x, al, be, OUT, I64(R), C, n=R * C // 8 * 9,
                   t="fp8")
        xd = xf
    want = gl.core_layernorm(xd, gl.decode(al, "bf16"), gl.decode(be, "bf16"), "bf16", "bf16")
    same(got, gl.encode(want, "bfp16"))


RMS_COEF = [(1 / 64, 1e-5), (0.0, 0.0), (0.0, 1e-5), (NAN, 1e-5), (INF, 1e-5), (1 / 64, -1.0),
            (1 / 64, NAN), (1 / 64, INF), (-1.0, 1e-5), (1e30, 1e30), (1 / 64, -0.0)]


@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("bfp", [False, True])
def test_golden_special_rmsnorm(t, bfp):
    if bfp and t != "bf16":
        pytest.skip("bfp16 is bf16 compute")
    C, R = 64, nrows(t)
    x = sprinkle(t, R, (C,))
    wt = "f16" if t == "fp8" else t
    gm = wsp(wt, C)
    for inv, eps in RMS_COEF:
        inv, eps = float(f32(inv)), float(f32(eps))
        if bfp:
            xb = encode_bfp16(gl.decode(x, t).reshape(-1), Rnd.kHalfEven)
            got = lrun("npu_rmsnorm_bfp16", xb, gm, OUT, I64(R), C, inv, eps, n=R * C // 8 * 9,
                       t="fp8")
            want = gl.rmsnorm(gl.decode(xb, "bfp16").reshape(R, C), gl.decode(gm, wt),
                              f32(inv), f32(eps))
            same(got, gl.encode(want, "bfp16"))
        else:
            got = lrun("npu_rmsnorm_bf16", x, CODE[t], gm, OUT, I64(R), C, inv, eps, n=(R, C),
                       t=t)
            want = gl.rmsnorm(gl.decode(x, t), gl.decode(gm, wt), f32(inv), f32(eps))
            same(got, gl.encode(want, t))


@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
def test_golden_special_l2_norm(t):
    C, R = 64, nrows(t)
    x = sprinkle(t, R, (C,))
    wt = "f16" if t == "fp8" else t
    al, be = wsp(wt, C), wsp(wt, C)
    got = lrun("npu_l2_norm_bf16", x, CODE[t], al, be, OUT, I64(R), C, n=(R, C), t=t)
    want = gl.l2norm(gl.decode(x, t), gl.decode(al, wt), gl.decode(be, wt))
    same(got, gl.encode(want, t))


QL2 = [(f(0.0123), 3.0, f(1 / 0.0371), 2.0), (0.0, 0.0, 1.0, 0.0), (INF, 0.0, 1.0, 0.0),
       (NAN, 0.0, 1.0, 0.0), (1e30, 0.0, 1.0, 0.0), (1.0, INF, 1.0, 0.0), (1.0, NAN, 1.0, 0.0),
       (1.0, 1e30, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, INF, 0.0), (1.0, 0.0, NAN, 0.0),
       (1.0, 0.0, 1e30, 0.0), (1.0, 0.0, 1.0, INF), (1.0, 0.0, 1.0, -INF), (1.0, 0.0, 1.0, NAN),
       (1.0, 0.0, 1.0, 3e9)]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("affine", [True, False])
def test_golden_special_qdq_l2norm(bits, affine):
    C, R = 64, 12
    types = ["i8", "u8"] if bits == 8 else ["i16", "u16"]
    al = wsp("bf16", C) if affine else None
    be = wsp("bf16", C) if affine else None
    for it in types:
        for ot in types:
            q = int_extremes(it, (R, C))
            q[0] = 0
            q[1] = INT_RANGE[it][1]
            for s, zp, qs, qz in QL2:
                got = lrun(f"npu_qdq_l2norm_a{bits}", q, INT[it][0], al, be, OUT, INT[ot][0],
                           I64(R), C, s, zp, qs, qz, n=(R, C), t=ot)
                want = gl.qdq_l2norm(q, f32(s), f32(zp),
                                     None if al is None else gl.decode(al, "bf16"),
                                     None if be is None else gl.decode(be, "bf16"),
                                     f32(qs), f32(qz), ot)
                same(got, want.astype(INT[ot][1]))


LN_EPS = [1e-5, 0.0, -1.0, INF, NAN, 1e-40]


@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("stats", ["fused", "batched"])
@pytest.mark.parametrize("entry", ["npu_add_layernorm_bf16", "npu_add_layernorm_const_bf16"])
def test_golden_special_add_layernorm(t, stats, entry):
    C, R = 64, nrows(t)
    a, b = sprinkle(t, R, (C,)), dense(t, (R, C), 0.3)
    wt = "f16" if t == "fp8" else t
    al, be = wsp(wt, C), wsp(wt, C)
    for eps in LN_EPS:
        got = lrun(entry, a, b, CODE[t], al, be, OUT, I64(R), C, eps,
                   {"fused": 0, "batched": 1}[stats], n=(R, C), t=t)
        same(got, gl.add_layernorm(a, b, t, al, be, f32(eps), stats))


QSC = [(f(0.0123), 3.0), (0.0, 0.0), (INF, 0.0), (NAN, 0.0), (1e30, 0.0), (1.0, INF),
       (1.0, NAN), (1.0, 1e30), (-1.0, 0.0), (1e-40, 0.0)]
QT = ["i8", "u8", "i16", "u16"]


@pytest.mark.parametrize("compute", ["bf16", "f16"])
@pytest.mark.parametrize("stats", ["fused", "batched"])
def test_golden_special_qdq_add_layernorm(compute, stats):
    C, R = 64, 16
    al, be = wsp(compute, C), wsp(compute, C)
    st = {"fused": 0, "batched": 1}[stats]
    for k, it in enumerate(QT):
        ot = QT[(k + 1) % 4]
        a, b = int_extremes(it, (R, C)), int_extremes(it, (R, C))
        bc = dense(compute, (R, C), 0.3)
        for i, (da, db) in enumerate(zip(QSC, QSC[3:] + QSC[:3])):
            qs, qz = QSC[(i + 2) % len(QSC)]
            eps = LN_EPS[i % len(LN_EPS)]
            got = lrun("npu_qdq_add_layernorm", a, b, INT[it][0], CODE[compute], al, be, OUT,
                       INT[ot][0], I64(R), C, *map(float, (*da, *db, qs, qz, eps)), st,
                       n=(R, C), t=ot)
            x = gl.qdq_ln_pre("add", compute, a, b, da=da, db=db)
            y = gl.layernorm(x, gl.decode(al, compute), gl.decode(be, compute), f32(eps), stats,
                             compute, compute)
            same(got, gl.quantize(y, qs, qz, ot).astype(INT[ot][1]))
            got = lrun("npu_qdq_add_layernorm_const", a, INT[it][0], bc, CODE[compute], al, be,
                       OUT, INT[ot][0], I64(R), C, *map(float, (*da, qs, qz, eps)), st,
                       n=(R, C), t=ot)
            x = gl.qdq_ln_pre("const", compute, a, gl.decode(bc, compute), da=da)
            y = gl.layernorm(x, gl.decode(al, compute), gl.decode(be, compute), f32(eps), stats,
                             compute, compute)
            same(got, gl.quantize(y, qs, qz, ot).astype(INT[ot][1]))


@pytest.mark.parametrize("compute", ["bf16", "f16", "f32"])
@pytest.mark.parametrize("stats", ["fused", "batched"])
@pytest.mark.parametrize("float_a", [False, True])
def test_golden_special_qdq_add_add_const_layernorm(compute, stats, float_a):
    C, R = 64, 16
    al, be = wsp(compute, C), wsp(compute, C)
    kc = wsp(compute, C)
    st = {"fused": 0, "batched": 1}[stats]
    for k, bt in enumerate(QT):
        ot = QT[(k + 2) % 4]
        at = compute if float_a else bt
        a = dense(compute, (R, C), 0.3) if float_a else int_extremes(bt, (R, C))
        b = int_extremes(bt, (R, C))
        for i, (da, db) in enumerate(zip(QSC, QSC[3:] + QSC[:3])):
            qs, qz = QSC[(i + 2) % len(QSC)]
            eps = LN_EPS[i % len(LN_EPS)]
            got = lrun("npu_qdq_add_add_const_layernorm", a, CODE[at] if float_a else INT[bt][0],
                       b, INT[bt][0], kc, CODE[compute], al, be, OUT, INT[ot][0], I64(R), C,
                       *map(float, (*da, *db, qs, qz, eps)), st, n=(R, C), t=ot)
            av = fdec(compute, a) if float_a else a
            x = gl.qdq_ln_pre("addc", compute, av, b, fdec(compute, kc), at, da, db)
            y = gl.layernorm(x, fdec(compute, al), fdec(compute, be), f32(eps), stats, compute,
                             compute)
            same(got, gl.quantize(y, qs, qz, ot).astype(INT[ot][1]))


QLN = {"fused": 0, "batched": 1, "c8_fused": 2, "c8_batched": 3, "integer": 4}


@pytest.mark.parametrize("variant,C", [("fused", 64), ("batched", 64), ("c8_fused", 24),
                                       ("c8_batched", 16), ("integer", 64)])
@pytest.mark.parametrize("compute", ["bf16", "f16", "f32"])
def test_golden_special_qdq_layernorm(variant, C, compute):
    if variant == "integer" and compute != "f16":
        pytest.skip("integer statistics are f16 compute")
    R = 16
    for k, it in enumerate(QT[:2] if variant == "integer" else QT):
        ot = QT[(k + 3) % 4] if variant != "integer" else QT[k ^ 1]
        q = int_extremes(it, (R, C))
        q[0] = INT_RANGE[it][0]
        q[1] = INT_RANGE[it][1]
        for i, (s, zp) in enumerate(QSC):
            a, b = wsp("f32", C), wsp("f32", C)
            got = lrun("npu_qdq_layernorm", q, INT[it][0], CODE.get(compute, NPU_F32), a, b, OUT,
                       INT[ot][0], I64(R), C, s, zp, QLN[variant], n=(R, C), t=ot)
            want = gl.qdq_layernorm(q, compute, a, b, f32(s), f32(zp), ot, variant)
            same(got, want.astype(INT[ot][1]))


# ---------------------------------------------------------------------------- groupnorm.h

GCODE = {"bf16": NPU_BF16, "f16": NPU_F16, "fp8": NPU_FP8, "f32": NPU_F32,
         **{k: v[0] for k, v in INT.items()}}
GSTORE = {"bf16": np.uint16, "f16": np.uint16, "fp8": np.uint8, "f32": np.float32,
          **{k: v[1] for k, v in INT.items()}}


def grun(name, *args, n, t, n2=0):
    return call_n(name, *args, n=n, dt=GSTORE[t], n2=n2)


def chan_special(t, hw, C):
    """(hw, C): channel c holds the c-th special flavour (rows of `sprinkle` transposed)."""
    return np.ascontiguousarray(sprinkle(t, C, (hw,)).T)


@pytest.mark.parametrize("io", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("act,slope", [(0, 0.0), (1, 0.0), (3, f(0.0123)), (3, NAN), (3, INF),
                                       (3, -0.0)])
@pytest.mark.parametrize("amad", [False, True])
@pytest.mark.parametrize("split", [0, -1])
def test_golden_special_instance_norm(io, act, slope, amad, split):
    C, sv_hw = 32, 4
    hw = (12 if split == 0 else 1) * sv_hw * 2
    x = chan_special(io, hw, C)
    ab = "f16" if io == "f16" else "bf16"
    alpha, beta = wsp(ab, C), wsp(ab, C)
    args = [x, GCODE[io], alpha, beta, GCODE[ab]]
    amad_ops = None
    if amad:
        a1, a2, cst = wsp(io, C), wsp(io, C), wsp(ab, C)
        args += [a1, a2, cst]
        amad_ops = (gg.decode(a1, io), gg.decode(cst, ab), gg.decode(a2, io))
    want = gg.in_bf16(x, io, gg.decode(alpha, ab), gg.decode(beta, ab), split, sv_hw, 0, act,
                      slope, amad_ops)
    base = "npu_instance_norm_add_mul_add_sd" if amad else "npu_instance_norm_sd"
    for name in (base + "_bf16", base + "_opt_stage2_ifm_bf16"):
        got = grun(name, *args, OUT, I64(hw), I64(C), split, I64(sv_hw), I64(0), act, slope,
                   n=hw * C, t=io)
        same(got.reshape(hw, C), want)


QDQ_IN = [(f(0.0123), 3.0, f(1 / 0.0371), 2.0), (0.0, 0.0, 1.0, 0.0), (INF, 0.0, 1.0, 0.0),
          (NAN, 0.0, 1.0, 0.0), (1e30, 0.0, 1.0, 0.0), (1.0, INF, 1.0, 0.0),
          (1.0, NAN, 1.0, 0.0), (1.0, 1e30, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0),
          (1.0, 0.0, INF, 0.0), (1.0, 0.0, NAN, 0.0), (1.0, 0.0, 1e30, 0.0),
          (1.0, 0.0, 1.0, INF), (1.0, 0.0, 1.0, -INF), (1.0, 0.0, 1.0, NAN),
          (1.0, 0.0, 1.0, 3e9)]


@pytest.mark.parametrize("io", ["i8", "u16"])
@pytest.mark.parametrize("ct", ["f32", "bf16", "f16"])
@pytest.mark.parametrize("act,slope", [(0, 0.0), (1, 0.0), (3, f(0.0123)), (3, NAN), (3, INF)])
def test_golden_special_qdq_instance_norm(io, ct, act, slope):
    C, sv_hw, split, hw = 32, 4, 1, 16
    x = int_extremes(io, (hw, C))
    x[:, 0] = 0
    x[:, 1] = INT_RANGE[io][1]
    alpha, beta = wsp(ct, C), wsp(ct, C)
    al, be = fdec(ct, alpha), fdec(ct, beta)
    for sc, zp, qs, qz in QDQ_IN:
        want = gg.in_qdq(x, io, al, be, ct, split, sv_hw, 0, sc, zp, qs, qz, act, slope)
        got = grun("npu_qdq_instance_norm_sd", x, GCODE[io], alpha, beta, GCODE[ct], OUT,
                   I64(hw), I64(C), GCODE[ct], split, I64(sv_hw), I64(0), sc, zp, qs, qz, act,
                   slope, n=hw * C, t=io)
        same(got.reshape(hw, C), want.astype(GSTORE[io]))
        q2 = (qs, qz)
        wa, wb = gg.in_qdq(x, io, al, be, ct, split, sv_hw, 0, sc, zp, qs, qz, act, slope,
                           q2=q2)
        ga_, gb_ = grun("npu_qdq_instance_norm_sd_opt_stage2_ifm_double_out", x, GCODE[io],
                        alpha, beta, GCODE[ct], OUT, OUT2, I64(hw), I64(C), GCODE[ct], split,
                        I64(sv_hw), I64(0), sc, zp, qs, qz, q2[0], q2[1], act, slope,
                        n=hw * C, n2=hw * C, t=io)
        same(ga_.reshape(hw, C), wa.astype(GSTORE[io]))
        same(gb_.reshape(hw, C), wb.astype(GSTORE[io]))


@pytest.mark.parametrize("io", ["i8", "u16"])
@pytest.mark.parametrize("ct", ["f32", "bf16", "f16"])
def test_golden_special_qdq_instance_norm_amad(io, ct):
    C, sv_hw, split, hw = 32, 4, 2, 8
    x = int_extremes(io, (hw, C))
    alpha, beta, cst = wsp(ct, C), wsp(ct, C), wsp(ct, C)
    a1, a2 = int_extremes(io, (C,)), int_extremes(io, (C,))
    for i, (sc, zp, qs, qz) in enumerate(QDQ_IN):
        sc2, zp2 = QDQ_IN[(i + 3) % len(QDQ_IN)][:2]
        want = gg.in_qdq(x, io, fdec(ct, alpha), fdec(ct, beta), ct, split, sv_hw, 0, sc, zp, qs,
                         qz, 3, f(0.01), amad_ops=(a1, fdec(ct, cst), a2), dq2=(sc2, zp2))
        got = grun("npu_qdq_instance_norm_add_mul_add_sd", x, GCODE[io], alpha, beta, GCODE[ct],
                   a1, a2, cst, OUT, I64(hw), I64(C), GCODE[ct], split, I64(sv_hw), I64(0), sc,
                   zp, qs, qz, sc2, zp2, 3, f(0.01), n=hw * C, t=io)
        same(got.reshape(hw, C), want.astype(GSTORE[io]))


@pytest.mark.parametrize("ab", ["bf16", "f32"])
def test_golden_special_instance_norm_stream(ab):
    C, hw = 32, 64
    x = chan_special("bf16", hw, C)
    alpha, beta = wsp(ab, C), wsp(ab, C)
    want = gg.stream_bf16(x, fdec(ab, alpha), fdec(ab, beta))
    got = grun("npu_instance_norm_stream_bf16", x, alpha, beta, GCODE[ab], OUT, I64(hw), I64(C),
               n=hw * C, t="bf16")
    same(got.reshape(hw, C), want)


@pytest.mark.parametrize("io", ["i8", "u16"])
@pytest.mark.parametrize("ab", ["bf16", "f32"])
@pytest.mark.parametrize("relu_alpha", [1.0, 0.0, NAN, INF, -INF])
def test_golden_special_qdq_instance_norm_stream(io, ab, relu_alpha):
    C, hw = 32, 64
    x = int_extremes(io, (hw, C))
    alpha, beta = wsp(ab, C), wsp(ab, C)
    for sc, off, qs, qz in QDQ_IN:
        want = gg.stream_qdq(x, io, fdec(ab, alpha), fdec(ab, beta), sc, off, qs, qz, relu_alpha)
        got = grun("npu_qdq_instance_norm_stream", x, GCODE[io], alpha, beta, GCODE[ab], OUT,
                   I64(hw), I64(C), sc, off, qs, qz, relu_alpha, n=hw * C, t=io)
        same(got.reshape(hw, C), want.astype(GSTORE[io]))


@pytest.mark.parametrize("io", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("act", [0, 5])
@pytest.mark.parametrize("eps", [f(1e-5), 0.0, -1.0, INF, NAN])
def test_golden_special_groupnorm_sd(io, act, eps):
    cg, sv, l, split = 2, 64, 256, 1
    G = 12 if io == "fp8" else 16
    x = sprinkle(io, G, (l,))
    ab = "f16" if io == "f16" else "bf16"
    alpha, beta = wsp(ab, G * cg), wsp(ab, G * cg)
    want = gg.gn_bf16_sd(x, io, gg.decode(alpha, ab), gg.decode(beta, ab), cg, split, sv, 0, eps,
                         act)
    got = grun("npu_groupnorm_sd_bf16", x, GCODE[io], alpha, beta, GCODE[ab], OUT, I64(G),
               I64(l), I64(cg), split, I64(sv), I64(0), eps, act, n=G * l, t=io)
    same(got.reshape(G, l), want)


@pytest.mark.parametrize("io", ["i8", "u16"])
@pytest.mark.parametrize("ct", ["f32", "bf16", "f16"])
def test_golden_special_qdq_groupnorm(io, ct):
    G, cg, sv, split = 3, 4, 64, 1
    l = 4 * sv
    x = int_extremes(io, (G, l))
    x[0] = 0
    x[1] = INT_RANGE[io][1]
    alpha, beta = wsp(ct, G * cg), wsp(ct, G * cg)
    for i, (sc, zp, qs, qz) in enumerate(QDQ_IN):
        eps = [f(1e-5), 0.0, -1.0, INF, NAN][i % 5]
        want = gg.gn_qdq_sd(x, io, fdec(ct, alpha), fdec(ct, beta), cg, ct, split, sv, 0, eps,
                            sc, zp, qs, qz, 5 if i % 2 else 0)
        got = grun("npu_qdq_groupnorm_sd", x, GCODE[io], alpha, beta, GCODE[ct], OUT, I64(G),
                   I64(l), I64(cg), GCODE[ct], split, I64(sv), I64(0), eps, sc, zp, qs, qz,
                   5 if i % 2 else 0, n=G * l, t=io)
        same(got.reshape(G, l), want.astype(GSTORE[io]))
    ab = "bf16" if ct == "f16" else ct
    al, be = wsp(ab, G * cg), wsp(ab, G * cg)
    for i, (sc, zp, qs, qz) in enumerate(QDQ_IN):
        eps = [f(1e-5), 0.0, -1.0, INF, NAN][i % 5]
        want = gg.gn_qdq(x, io, fdec(ab, al), fdec(ab, be), cg, sv, eps, sc, zp, qs, qz)
        got = grun("npu_qdq_groupnorm", x, GCODE[io], al, be, GCODE[ab], OUT, I64(G), I64(l),
                   I64(cg), I64(sv), eps, sc, zp, qs, qz, n=G * l, t=io)
        same(got.reshape(G, l), want.astype(GSTORE[io]))


@pytest.mark.parametrize("io", ["i16", "u16"])
@pytest.mark.parametrize("ct", ["f16", "bf16", "f32"])
def test_golden_special_conv_in_stage2(io, ct):
    hw, C = 96, 32
    x = int_extremes(io, (hw, C))
    alpha, beta, cst = wsp(ct, C), wsp(ct, C), wsp(ct, C)
    a1, a2 = int_extremes(io, (C,)), int_extremes(io, (C,))
    sp = np.array([0.0, -0.0, INF, -INF, NAN, 1e30, -1.0, 1e-40, 3e38], f32)
    S = RNG.choice(sp, C).astype(f32)
    S2 = RNG.choice(sp, C).astype(f32)
    for N in (hw - 3, 1):
        for i, (sc, zp, qs, qz) in enumerate(QDQ_IN):
            dq2 = QDQ_IN[(i + 5) % len(QDQ_IN)][:2]
            want = gg.in_stage2(x, io, S, S2, N, fdec(ct, alpha), fdec(ct, beta), ct, sc, zp, qs,
                                qz, 3, f(0.02), (a1, fdec(ct, cst), a2), dq2)
            got = grun("npu_qdq_ConvInstanceNormAddMulAdd_Stage2", x, GCODE[io], S, S2, alpha,
                       beta, GCODE[ct], a1, a2, cst, OUT, I64(hw), I64(C), I64(N), GCODE[ct], sc,
                       zp, qs, qz, dq2[0], dq2[1], 3, f(0.02), n=hw * C, t=io)
            same(got.reshape(hw, C), want.astype(GSTORE[io]))


# ---------------------------------------------------------------------------- avgpool.h

def s3(H, W, C):
    return I64(H), I64(W), I64(C)


POOLS = [(6, 6, 16, 3, 3, 1, 2), (4, 4, 8, 2, 2, 2, 2), (5, 7, 8, 5, 1, 1, 1)]
POOL_FACTORS = [f(1 / 9), 0.0, -0.0, INF, NAN, 1e30, 1e-40, -1.0]
GLOBALS = [(8, 12, 16, 4, 3), (5, 5, 8, 5, 1), (4, 4, 8, 4, 4)]


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("w", POOLS)
def test_golden_special_avgpool16(t, w):
    H, W, C, kh, kw, sh, sw = w
    Ho, Wo = ga.out_hw(H, W, kh, kw, sh, sw)
    for p in (0.1, 0.4):
        x = dense(t, (H, W, C), p)
        for factor in POOL_FACTORS:
            for entry in ("npu_avgpool_bf16", "npu_avgpool_bf16_c32"):
                if entry.endswith("c32"):
                    if t != "bf16":
                        continue
                    got = call(entry, x, OUT, *s3(H, W, C), kh, kw, sh, sw, factor,
                               out_nbytes=2 * Ho * Wo * C, out_dtype=np.uint16)
                else:
                    got = call(entry, x, F16T[t], OUT, *s3(H, W, C), kh, kw, sh, sw, factor,
                               out_nbytes=2 * Ho * Wo * C, out_dtype=np.uint16)
                same(got, ga.avgpool_16(x, t, kh, kw, sh, sw, factor))


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("gl_", GLOBALS)
def test_golden_special_globalavgpool16(t, gl_):
    H, W, C, sv_h, sv_w = gl_
    for p in (0.05, 0.3):
        x = dense(t, (H, W, C), p)
        for factor in POOL_FACTORS:
            got = call("npu_globalavgpool_bf16", x, F16T[t], OUT, *s3(H, W, C), sv_h, sv_w,
                       factor, out_nbytes=2 * C, out_dtype=np.uint16)
            same(got, ga.globalavgpool_16(x, t, sv_h, sv_w, factor))
            if t == "bf16":
                got = call("npu_globalavgpool_bf16_c32", x, OUT, *s3(H, W, C), sv_h, sv_w,
                           factor, out_nbytes=2 * C, out_dtype=np.uint16)
                same(got, ga.globalavgpool_16(x, t, sv_h, sv_w, factor))


@pytest.mark.parametrize("w", POOLS)
def test_golden_special_avgpool_fp8(w):
    H, W, C, kh, kw, sh, sw = w
    Ho, Wo = ga.out_hw(H, W, kh, kw, sh, sw)
    sp = np.array([0.0, -0.0, INF, -INF, NAN, 1e30, -1e30, 1e-40, f(1 / 9)], f32)
    for p in (0.05, 0.3):
        x = dense("fp8", (H, W, C), p)
        for kh_sv in sorted({1, kh}):
            for k in range(len(sp)):
                c0 = np.roll(sp, k)[np.arange(C) % len(sp)]
                c2 = np.roll(sp, 2 * k + 1)[np.arange(C) % len(sp)]
                for act, lo, hi in ((0, 0.0, 0.0), (1, -1.5, 40.0), (1, NAN, 1.0), (1, 0.0, NAN),
                                    (1, INF, -INF), (1, -INF, INF), (1, 1.0, -1.0), (1, -0.0, 0.0)):
                    got = call("npu_avgpool_fp8", x, OUT, *s3(H, W, C), kh, kw, sh, sw, kh_sv,
                               c0, c2, act, lo, hi, out_nbytes=Ho * Wo * C, out_dtype=np.uint8)
                    same(got, ga.avgpool_fp8(x, kh, kw, sh, sw, kh_sv, c0, c2, act, lo, hi))


@pytest.mark.parametrize("gl_", GLOBALS)
def test_golden_special_globalavgpool_fp8(gl_):
    H, W, C, sv_h, sv_w = gl_
    for p in (0.05, 0.3):
        x = dense("fp8", (H, W, C), p)
        for c0 in (0.0, -0.0, INF, -INF, NAN, 1e30, f(3.25)):
            for c1 in (0.0, INF, NAN, 1e30, f(1 / 9), -1e30):
                got = call("npu_globalavgpool_fp8", x, OUT, *s3(H, W, C), sv_h, sv_w, c0, c1,
                           out_nbytes=C, out_dtype=np.uint8)
                same(got, ga.globalavgpool_fp8(x, sv_h, sv_w, c0, c1))


QPOOL = [(0.0, f(0.5), 0), (INF, f(0.5), 0), (NAN, f(0.5), 0), (-INF, INF, 0), (0.0, INF, 0),
         (0.0, NAN, 0), (1e30, 1e30, 0), (-1e30, 1e30, 31), (0.0, 1.0, 64), (0.0, 1.0, -64),
         (0.5, 1e30, 64), (3e38, 3e38, -64), (-0.0, -0.0, 0), (f(0.5), f(1 / 9), 2)]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("w", POOLS)
def test_golden_special_qdq_avgpool(bits, w):
    H, W, C, kh, kw, sh, sw = w
    Ho, Wo = ga.out_hw(H, W, kh, kw, sh, sw)
    names = ("i8", "u8") if bits == 8 else ("i16", "u16")
    for it in names:
        for ot in names:
            x = int_extremes(it, (H, W, C))
            for c0, c1, shift in QPOOL:
                if bits == 8:
                    for act, lo, hi in ((0, 0.0, 0.0), (1, -3.5, 40.25), (1, NAN, 1.0),
                                        (1, 0.0, NAN), (1, INF, -INF), (1, -INF, INF)):
                        got = call("npu_qdq_avgpool_a8", x, INT[it][0], OUT, INT[ot][0],
                                   *s3(H, W, C), kh, kw, sh, sw, c0, c1, shift, act, lo, hi,
                                   out_nbytes=Ho * Wo * C, out_dtype=INT[ot][1])
                        same(got, ga.qdq_avgpool(x, ot, kh, kw, sh, sw, c0, c1, shift, act, lo,
                                                 hi).astype(INT[ot][1]))
                else:
                    got = call("npu_qdq_avgpool_a16", x, INT[it][0], OUT, INT[ot][0],
                               *s3(H, W, C), kh, kw, sh, sw, c0, c1, shift,
                               out_nbytes=2 * Ho * Wo * C, out_dtype=INT[ot][1])
                    same(got, ga.qdq_avgpool(x, ot, kh, kw, sh, sw, c0, c1, shift
                                             ).astype(INT[ot][1]))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("gl_", GLOBALS[:2])
def test_golden_special_qdq_globalavgpool(bits, gl_):
    H, W, C = gl_[:3]
    names = ("i8", "u8") if bits == 8 else ("i16", "u16")
    for it in names:
        for ot in names:
            x = int_extremes(it, (H, W, C))
            for c0, c1, shift in QPOOL:
                got = call(f"npu_qdq_globalavgpool_a{bits}", x, INT[it][0], OUT, INT[ot][0],
                           *s3(H, W, C), c0, c1, shift, out_nbytes=C * bits // 8,
                           out_dtype=INT[ot][1])
                same(got, ga.qdq_global(x, ot, c0, c1, shift, as_i16=bits == 16
                                        ).astype(INT[ot][1]))


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("w", POOLS)
def test_golden_special_avgpool_xint8(w, fused):
    H, W, C, kh, kw, sh, sw = w
    Ho, Wo = ga.out_hw(H, W, kh, kw, sh, sw)
    for t in ("i8", "u8"):
        x = int_extremes(t, (H, W, C))
        for wt, shift in ((0, 0), (1, 0), (0xFFFF, 0), (0xFFFF, 31), (0xFFFF, 1), (1, 31),
                          (3641, 15)):
            if fused:
                for act in (0, 1):
                    got = call("npu_avgpool_xint8_fused", x, INT[t][0], OUT, INT[t][0],
                               *s3(H, W, C), kh, kw, sh, sw, U32(wt), shift, act,
                               out_nbytes=Ho * Wo * C, out_dtype=INT[t][1])
                    same(got, ga.avgpool_xint8(x, t, kh, kw, sh, sw, wt, shift, act == 1
                                               ).astype(INT[t][1]))
            else:
                got = call("npu_avgpool_xint8", x, INT[t][0], OUT, INT[t][0], *s3(H, W, C), kh,
                           kw, sh, sw, U32(wt), shift, out_nbytes=Ho * Wo * C,
                           out_dtype=INT[t][1])
                same(got, ga.avgpool_xint8(x, t, kh, kw, sh, sw, wt, shift).astype(INT[t][1]))
