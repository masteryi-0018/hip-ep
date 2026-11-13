# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The arith_int entry points (include/npu_emu_arith_int.h) against the numpy
reference in golden_arith_int.py, bit for bit: every operand type, both
Broadcasting3d routes (channel broadcast or not) and the other broadcast
shapes, every activation, coefficients from non power-of-two scales (0.1 /
0.6 style ratios) and dyadic ones that make exact .5 ties, and inputs at the
type limits so the outputs saturate."""

import numpy as np
import pytest

import golden_arith_int as ga
from npu import (I64, NPU_I8, NPU_I16, NPU_I32, NPU_U8, NPU_U16, OUT, Host, call, raw_rc)

RNG = np.random.default_rng(2031)
f32 = np.float32
T = {"i8": (NPU_I8, np.int8), "u8": (NPU_U8, np.uint8), "i16": (NPU_I16, np.int16),
     "u16": (NPU_U16, np.uint16), "i32": (NPU_I32, np.int32)}


def rnd(t, shape):
    """Random values with the type limits and their neighbours mixed in."""
    info = np.iinfo(T[t][1])
    x = RNG.integers(info.min, info.max, shape, endpoint=True)
    edge = np.array([info.min, info.min + 1, info.max - 1, info.max, 0, 1, -1])
    edge = edge[(edge >= info.min) & (edge <= info.max)]
    m = RNG.random(shape) < 0.1
    x[m] = RNG.choice(edge, m.sum())
    return x.astype(T[t][1])


def run(name, *args, n, t):
    dt = T[t][1]
    return call(name, *args, out_nbytes=n * np.dtype(dt).itemsize, out_dtype=dt)


def assert_same(got, want):
    got = np.asarray(got).reshape(-1).astype(np.int64)
    want = np.asarray(want).reshape(-1).astype(np.int64)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{got.size} differ, first at {bad[:6].tolist()}: "
                           f"got {got[bad[:6]].tolist()} want {want[bad[:6]].tolist()}")


def zp(t):
    lo, hi = {"i8": (-128, 127), "u8": (0, 255), "i16": (-32768, 32767),
              "u16": (0, 65535)}[t]
    return int(RNG.integers(lo, hi + 1))


# Quantization parameter sets (s1, s2, s_out): non power-of-two ratios with
# many near-ties, dyadic ratios with exact ties, and a large output range.
SCALES = [(0.1, 0.6, 0.3), (0.004, 0.004, 0.004), (0.125, 0.25, 0.5), (0.02, 0.03, 0.7),
          (1.332358e-05, 4.788564e-06, 1.4e-08)]


def add_coeffs(t, si, sub=False):
    s1, s2, so = SCALES[si]
    z1, z2, zo = zp(t), zp(t), zp(t)
    c0 = s1 / so
    c1 = -(s2 / so) if sub else s2 / so
    c2 = zo - z1 * s1 / so + (z2 * s2 / so if sub else -(z2 * s2 / so))
    return (f32(c0), f32(c1), f32(c2)), (z1, z2, zo)


def mul_coeffs(t, si):
    s1, s2, so = SCALES[si]
    z1, z2, zo = zp(t), zp(t), zp(t)
    c0 = s1 * s2 / so
    return (f32(c0), f32(-z2 * c0), f32(-z1 * c0), f32(zo + z1 * z2 * c0)), (z1, z2, zo, c0)


def act_words(act, zo, so, t, alpha=0.1015625):
    """act0 / act1 as the fused YAMLs compute them (fp32)."""
    with np.errstate(over="ignore"):
        if act == ga.ACT_RELU:
            return f32(zo), f32(3.4028235e38 / so + zo)
        if act == ga.ACT_CLIP:
            return f32(zo), f32(6.0 / so + zo)
        if act == ga.ACT_LRELU:
            return f32(alpha), f32((1 - alpha) * zo)
    return f32(0), f32(0)


BITS = {8: ["i8", "u8"], 16: ["i16", "u16"]}


# ---- flat add / mul and the fused variants ----------------------------------------

@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("it", [0, 1])
@pytest.mark.parametrize("ot", [0, 1])
@pytest.mark.parametrize("si", range(len(SCALES)))
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_flat(bits, it, ot, si, op):
    it, ot = BITS[bits][it], BITS[bits][ot]
    n = 4099
    a, b = rnd(it, n), rnd(it, n)
    if op == "mul":
        c, _ = mul_coeffs(ot, si)
        want = ga.mul(a, b, ot, *c)
        got = run(f"npu_elementwise_mul_a{bits}", a, b, T[it][0], OUT, T[ot][0], I64(n),
                  *map(float, c), n=n, t=ot)
    else:
        c, _ = add_coeffs(ot, si, op == "sub")
        want = ga.add(a, b, ot, *c)
        got = run(f"npu_elementwise_add_a{bits}", a, b, T[it][0], OUT, T[ot][0], I64(n),
                  *map(float, c), n=n, t=ot)
    assert_same(got, want)


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("t", [0, 1])
@pytest.mark.parametrize("si", [0, 2, 3])
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
@pytest.mark.parametrize("act", [ga.ACT_NONE, ga.ACT_RELU, ga.ACT_CLIP, ga.ACT_LRELU])
def test_flat_fused(bits, t, si, op, act):
    t = BITS[bits][t]
    n = 2048
    a, b = rnd(t, n), rnd(t, n)
    so = SCALES[si][2]
    # relu runs as CLIP with a huge upper bound in the flat fused kernels
    kind = ga.ACT_CLIP if act == ga.ACT_RELU else act
    if op == "mul":
        c, (_, _, zo, _) = mul_coeffs(t, si)
    else:
        c, (_, _, zo) = add_coeffs(t, si, op == "sub")
    a0, a1 = act_words(act, zo, so, t)
    if op == "mul":
        want = ga.mul(a, b, t, *c, kind, a0, a1)
        name = f"npu_elementwise_mul_fused_a{bits}"
    else:
        want = ga.add(a, b, t, *c, kind, a0, a1)
        name = f"npu_elementwise_{op}_fused_a{bits}"
    got = run(name, a, b, T[t][0], OUT, T[t][0], I64(n), *map(float, c), kind, float(a0),
              float(a1), n=n, t=t)
    assert_same(got, want)


# ---- Xint8 ------------------------------------------------------------------------

@pytest.mark.parametrize("si", range(len(SCALES)))
def test_xint_add(si):
    n = 3000
    a, b = rnd("i8", n), rnd("i8", n)
    s1, s2, so = SCALES[si]
    c = (f32(s1 / so), f32(s2 / so), f32(0))
    got = run("npu_elementwise_add_a8_xint", a, b, OUT, I64(n), *map(float, c), n=n, t="i8")
    assert_same(got, ga.add(a, b, "i8", *c))


@pytest.mark.parametrize("e", [(0, 1, 0), (-1, -3, -2), (2, -4, 0), (-7, -7, -7), (3, 3, 3)])
@pytest.mark.parametrize("act", [ga.ACT_NONE, ga.ACT_CLIP])
def test_xint_add_fused(e, act):
    """c0 = 2^e0, c1 = 2^e1 (e[2] only labels the case); CLIP bounds with ties."""
    n = 4096
    a, b = rnd("i8", n), rnd("i8", n)
    c0, c1 = f32(2.0 ** e[0]), f32(2.0 ** e[1])
    a0, a1 = (f32(-20.5), f32(90.5)) if act == ga.ACT_CLIP else (f32(0), f32(0))
    got = run("npu_elementwise_add_fused_a8_xint", a, b, OUT, I64(n), float(c0), float(c1), act,
              float(a0), float(a1), n=n, t="i8")
    assert_same(got, ga.add_fused_xint(a, b, c0, c1, act, a0, a1))
    # the kernel's integer result equals the float model on these coefficients
    assert_same(got, ga.add(a, b, "i8", c0, c1, f32(0), ga.ACT_CLIP if act else 0, a0, a1))


def fixed_coeffs(s_mul, s_add, t=7):
    p = int(np.floor(min(np.log2(32767.0 / (s_mul * 2 ** t)), np.log2(32767.0 / s_add))))
    return int(np.rint(s_mul * 2 ** (p + t))), int(np.rint(s_add * 2 ** p)), p, t


MAC_SHAPES = [((5, 7, 64), (5, 7, 64), (5, 7, 64)), ((5, 7, 64), (1, 1, 64), (5, 7, 64)),
              ((5, 7, 64), (5, 7, 1), (5, 7, 64)), ((5, 7, 64), (1, 1, 1), (5, 7, 64)),
              ((4, 6, 32), (1, 6, 32), (4, 1, 32))]


@pytest.mark.parametrize("shape", range(len(MAC_SHAPES)))
@pytest.mark.parametrize("types", [("i8", "i8", "i8", "i8"), ("u8", "u8", "i8", "u8")])
@pytest.mark.parametrize("scales", [(0.02, 0.03, 0.05, 0.1), (0.1, 0.6, 0.3, 0.7)])
def test_xint_mac(shape, types, scales):
    s1, s2, s3 = MAC_SHAPES[shape]
    t1, t2, t3, to = types
    x1, x2, x3 = rnd(t1, s1), rnd(t2, s2), rnd(t3, s3)
    A, v, p, t = fixed_coeffs(f32(scales[0] * scales[1] / scales[3]),
                              f32(scales[2] / scales[3]))
    if s2 == (1, 1, 1):
        want = ga.mac_xint_scalar(x1, x2.reshape(-1)[0], x3, to, A, v, p, t)
    else:
        want = ga.mac_xint(x1, x2, x3, to, A, v, p, t)
    got = run("npu_elementwise_mac_a8", x1, T[t1][0], x2, T[t2][0], x3, T[t3][0], OUT, T[to][0],
              Host(np.array(s1, np.int64)), Host(np.array(s2, np.int64)),
              Host(np.array(s3, np.int64)), A, v, p, t, n=x1.size, t=to)
    assert_same(got, want)


def test_xint_mac_scalar_wraps():
    """A uint8 scalar ifm2 can push w past int16; the scalar path wraps it."""
    x1, x3 = rnd("i8", (2, 3, 32)), rnd("i8", (2, 3, 32))
    x2 = np.array([[[250]]], np.uint8)
    A, v, p, t = 30000, 100, 9, 7
    one = Host(np.array((1, 1, 1), np.int64))
    s = Host(np.array(x1.shape, np.int64))
    got = run("npu_elementwise_mac_a8", x1, NPU_I8, x2, NPU_U8, x3, NPU_I8, OUT, NPU_I8, s, one,
              s, A, v, p, t, n=x1.size, t="i8")
    assert_same(got, ga.mac_xint_scalar(x1, 250, x3, "i8", A, v, p, t))
    assert not np.array_equal(got, ga.mac_xint(x1, x2, x3, "i8", A, v, p, t).reshape(-1))


# ---- Broadcasting3d -----------------------------------------------------------------

BC_SHAPES = [((4, 6, 96), (4, 6, 96)), ((4, 6, 96), (4, 6, 1)), ((4, 6, 96), (1, 1, 96)),
             ((4, 6, 96), (1, 1, 1)), ((4, 6, 96), (1, 6, 1)), ((3, 5, 40), (3, 1, 40))]


def shp(s):
    return Host(np.array(s, np.int64))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("it", [0, 1])
@pytest.mark.parametrize("ot", [0, 1])
@pytest.mark.parametrize("shape", range(len(BC_SHAPES)))
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_bc3d(bits, it, ot, shape, op):
    it, ot = BITS[bits][it], BITS[bits][ot]
    s1, s2 = BC_SHAPES[shape]
    a, b = rnd(it, s1), rnd(it, s2)
    si = shape % len(SCALES)
    if op == "mul":
        c, _ = mul_coeffs(ot, si)
        want = ga.mul(a, b, ot, *c)
    else:
        c, _ = add_coeffs(ot, si, op == "sub")
        want = ga.add(a, b, ot, *c)
    name = f"npu_qdq_elementwise_broadcasting3d_{'mul' if op == 'mul' else 'add'}_a{bits}"
    got = run(name, a, b, T[it][0], OUT, T[ot][0], shp(s1), shp(s2), *map(float, c), n=a.size,
              t=ot)
    assert_same(got, want)


def addsub_fused_words(t, si, sub, act, alpha=0.1015625):
    s1, s2, so = SCALES[si]
    z1, z2, zo = zp(t), zp(t), zp(t)
    c0 = s1 / so
    c1 = -(s2 / so) if sub else s2 / so
    c2 = zo - z1 * s1 / so + (z2 * s2 / so if sub else -(z2 * s2 / so))
    a0, a1 = act_words(act, zo, so, t, alpha)
    if act == ga.ACT_PRELU:
        return [f32(c0), f32(c1), f32(c2 - zo), f32(zo), 0, 0, 0, 0]
    beta = (1 - alpha) * zo
    return [f32(c0), f32(c1), f32(c2), a0, a1, f32(c0 * alpha), f32(c1 * alpha),
            f32(c2 * alpha + beta)]


def words(vals, ints=()):
    """A coeff struct as a host array of 32-bit words; `ints` lists int32 fields."""
    w = np.zeros(len(vals), np.uint32)
    for k, v in enumerate(vals):
        w[k] = np.int64(v) & 0xFFFFFFFF if k in ints else np.array([v], f32).view(np.uint32)[0]
    return Host(w)


FUSED_SHAPES = [((3, 5, 64), (3, 5, 1)), ((3, 5, 64), (3, 5, 64)), ((3, 5, 64), (1, 1, 1)),
                ((3, 5, 64), (1, 5, 64)), ((2, 4, 96), (1, 1, 96))]
ACTS = [ga.ACT_NONE, ga.ACT_RELU, ga.ACT_CLIP, ga.ACT_LRELU, ga.ACT_PRELU]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("t", [0, 1])
@pytest.mark.parametrize("shape", range(len(FUSED_SHAPES)))
@pytest.mark.parametrize("op", ["add", "sub"])
@pytest.mark.parametrize("act", ACTS)
def test_bc3d_addsub_fused(bits, t, shape, op, act):
    t = BITS[bits][t]
    s1, s2 = FUSED_SHAPES[shape]
    a, b = rnd(t, s1), rnd(t, s2)
    k = addsub_fused_words(t, (shape + act) % len(SCALES), op == "sub", act)
    slope = RNG.uniform(-0.6, 1.4, s1[2]).astype(f32) if act == ga.ACT_PRELU else None
    want = ga.addsub_fused(a, b, t, k, act, slope)
    got = run(f"npu_qdq_elementwise_broadcasting3d_{op}_fused_a{bits}", a, b, T[t][0], OUT,
              shp(s1), shp(s2), words(k), act, slope, n=a.size, t=t)
    assert_same(got, want)


def mul_fused_words(t, si, act, a16, alpha=0.1015625):
    s1, s2, so = SCALES[si]
    z1, z2, zo = zp(t), zp(t), zp(t)
    c0 = s1 * s2 / so
    c = [c0, -z2 * c0, -z1 * c0, zo + z1 * z2 * c0]
    if act == ga.ACT_LRELU and not a16:
        c = [c0, alpha * c0, c0 * (1 + alpha) / 2, c0 * (1 - alpha) / 2]
    a0, a1 = act_words(act, zo, so, t, alpha)
    return [f32(v) for v in c] + [a0, a1, z1, f32(zo), z2], f32(c0)


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("t", [0, 1])
@pytest.mark.parametrize("shape", range(len(FUSED_SHAPES)))
@pytest.mark.parametrize("act", ACTS)
def test_bc3d_mul_fused(bits, t, shape, act):
    t = BITS[bits][t]
    s1, s2 = FUSED_SHAPES[shape]
    a, b = rnd(t, s1), rnd(t, s2)
    k, c0 = mul_fused_words(t, (shape + act) % len(SCALES), act, bits == 16)
    ns = None
    if act == ga.ACT_PRELU:
        ns = (RNG.uniform(-0.6, 1.4, s1[2]).astype(f32) * c0).astype(f32)
    want = ga.mul_fused(a, b, t, k, act, bits == 16, ns)
    got = run(f"npu_qdq_elementwise_broadcasting3d_mul_fused_a{bits}", a, b, T[t][0], OUT,
              shp(s1), shp(s2), words(k, ints=(6, 8)), act, ns, n=a.size, t=t)
    assert_same(got, want)


MAC_VARIANTS = {"a8w8": (8, 8, True), "a16w8": (16, 8, False), "a16w16": (16, 16, True)}
MACF_SHAPES = [((3, 5, 64), (3, 5, 1), (3, 5, 1)), ((3, 5, 64), (3, 5, 64), (3, 5, 64)),
               ((3, 5, 64), (1, 1, 1), (1, 1, 1)), ((3, 5, 64), (1, 1, 64), (3, 5, 64))]


@pytest.mark.parametrize("variant", list(MAC_VARIANTS))
@pytest.mark.parametrize("signs", [(0, 0, 0), (1, 1, 1), (1, 0, 1)])
@pytest.mark.parametrize("ct", ["same", "i32"])
@pytest.mark.parametrize("shape", range(len(MACF_SHAPES)))
@pytest.mark.parametrize("act", [ga.ACT_NONE, ga.ACT_RELU, ga.ACT_CLIP, ga.ACT_LRELU])
def test_bc3d_mac_fused(variant, signs, ct, shape, act):
    abits, wbits, to_float = MAC_VARIANTS[variant]
    ta, tw = BITS[abits][signs[0]], BITS[wbits][signs[1]]
    tc = BITS[abits][signs[2]] if ct == "same" else "i32"
    s1, s2, s3 = MACF_SHAPES[shape]
    a, b, c = rnd(ta, s1), rnd(tw, s2), rnd(tc, s3)
    si = (shape + act) % len(SCALES)
    sa, sb, so = SCALES[si]
    sc = 1e-6 * so if tc == "i32" else 0.07
    z1, z2, z3, zo = zp(ta), zp(tw), zp(tc) if tc != "i32" else -12345, zp(ta)
    c0 = sa * sb / so
    c4 = sc / so
    a0, a1 = act_words(act, zo, so, ta)
    k = [f32(c0), f32(-z2 * c0), f32(-z1 * c0), f32(zo - z3 * c4), f32(c4), a0, a1, z1]
    want = ga.mac_fused(a, b, c, ta, k, act, to_float)
    got = run(f"npu_qdq_elementwise_broadcasting3d_mac_fused_{variant}", a, T[ta][0], b, T[tw][0],
              c, T[tc][0], OUT, shp(s1), shp(s2), shp(s3), words(k, ints=(7,)), act, n=a.size,
              t=ta)
    assert_same(got, want)


def test_mac_fused_addend_rounding_matters():
    """With int32 addends above 2^24 the fix2float route differs, so the bcc
    tests above discriminate between the two forms."""
    a = rnd("i16", (2, 2, 64))
    b = rnd("i8", (2, 2, 1))
    c = np.full((2, 2, 1), (1 << 26) + 3, np.int32)
    # c3 cancels the addend, leaving the 3 that fix2float rounds away
    k = [f32(1e-3), f32(0.01), f32(0), f32(-2.0 ** 26), f32(1), f32(0.1), f32(5), 7]
    assert np.any(ga.mac_fused(a, b, c, "i16", k, ga.ACT_LRELU, True) !=
                  ga.mac_fused(a, b, c, "i16", k, ga.ACT_LRELU, False))


# ---- argument validation --------------------------------------------------------------

def test_rejects():
    s3 = Host(np.array((2, 3, 4), np.int64))
    bad = Host(np.array((2, 3, 2), np.int64))
    one = Host(np.array((1, 1, 1), np.int64))
    k8 = Host(np.zeros(8, np.uint32))
    k9 = Host(np.zeros(9, np.uint32))
    assert raw_rc("npu_elementwise_add_a8", None, None, NPU_I16, None, NPU_I8, I64(4),
                  1.0, 1.0, 0.0) == -1
    assert raw_rc("npu_elementwise_add_fused_a8", None, None, NPU_I8, None, NPU_I8, I64(4),
                  1.0, 1.0, 0.0, ga.ACT_PRELU, 0.0, 0.0) == -1
    assert raw_rc("npu_elementwise_add_fused_a8_xint", None, None, None, I64(4), 0.3, 1.0,
                  ga.ACT_NONE, 0.0, 0.0) == -1
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_add_a8", None, None, NPU_I8, None, NPU_I8,
                  s3, bad, 1.0, 1.0, 0.0) == -1
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_add_fused_a8", None, None, NPU_I8, None,
                  s3, one, k8, ga.ACT_PRELU, None) == -1
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_mul_fused_a16", None, None, NPU_I8, None,
                  s3, one, k9, ga.ACT_NONE, None) == -1
    # ifm2 and ifm3 must agree on channel broadcasting
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_mac_fused_a16w8", None, NPU_I16, None,
                  NPU_I8, None, NPU_I16, None, s3, one, s3, k8, ga.ACT_NONE) == -1
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_mac_fused_a16w8", None, NPU_I16, None,
                  NPU_I8, None, NPU_I8, None, s3, one, one, k8, ga.ACT_NONE) == -1
