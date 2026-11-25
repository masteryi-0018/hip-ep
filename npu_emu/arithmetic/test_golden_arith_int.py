# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Integer / QDQ arithmetic entry points against the numpy reference, bit for bit."""

import numpy as np
import pytest

import golden_arith_int as ga
from npu import I64, NPU_I8, NPU_I16, NPU_I32, NPU_U8, NPU_U16, OUT, Host, raw_rc
from testing import INT_ALL as T
from testing import assert_same, run, shp

RNG = np.random.default_rng(2031)
f32 = np.float32


def rnd(t, shape):
    """Random values with the type limits and their neighbours mixed in."""
    info = np.iinfo(T[t][1])
    x = RNG.integers(info.min, info.max, shape, endpoint=True)
    edge = np.array([info.min, info.min + 1, info.max - 1, info.max, 0, 1, -1])
    edge = edge[(edge >= info.min) & (edge <= info.max)]
    m = RNG.random(shape) < 0.1
    x[m] = RNG.choice(edge, m.sum())
    return x.astype(T[t][1])


def zp(t):
    lo, hi = {"i8": (-128, 127), "u8": (0, 255), "i16": (-32768, 32767),
              "u16": (0, 65535)}[t]
    return int(RNG.integers(lo, hi + 1))


# (s1, s2, s_out): non power-of-two ratios, dyadic ratios with exact ties, a large range.
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
    """act0 / act1 of the fused variants (fp32)."""
    with np.errstate(over="ignore"):
        if act == ga.ACT_RELU:
            return f32(zo), f32(3.4028235e38 / so + zo)
        if act == ga.ACT_RELU6:
            return f32(zo), f32(6.0 / so + zo)
        if act == ga.ACT_LRELU:
            return f32(alpha), f32((1 - alpha) * zo)
    return f32(0), f32(0)


BITS = {8: ["i8", "u8"], 16: ["i16", "u16"]}


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
    else:
        c, _ = add_coeffs(ot, si, op == "sub")
    name = f"elementwise_{'mul' if op == 'mul' else 'add'}_a{bits}"
    want = getattr(ga, name)(a, it, b, ot, n, *c)
    got = run(f"npu_{name}", a, T[it][0], b, OUT, T[ot][0], I64(n), *map(float, c), n=n, t=ot)
    assert_same(got, want)


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("t", [0, 1])
@pytest.mark.parametrize("si", [0, 2, 3])
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
@pytest.mark.parametrize("act", [ga.ACT_NONE, ga.ACT_RELU, ga.ACT_RELU6, ga.ACT_LRELU])
def test_flat_fused(bits, t, si, op, act):
    t = BITS[bits][t]
    n = 2048
    a, b = rnd(t, n), rnd(t, n)
    so = SCALES[si][2]
    if op == "mul":
        c, (_, _, zo, _) = mul_coeffs(t, si)
    else:
        c, (_, _, zo) = add_coeffs(t, si, op == "sub")
    a0, a1 = act_words(act, zo, so, t)
    name = f"elementwise_{op}_fused_a{bits}"
    want = getattr(ga, name)(a, t, b, t, n, *c, act, a0, a1)
    got = run(f"npu_{name}", a, T[t][0], b, OUT, T[t][0], I64(n), *map(float, c), act, float(a0),
              float(a1), n=n, t=t)
    assert_same(got, want)


@pytest.mark.parametrize("si", range(len(SCALES)))
def test_xint_add(si):
    n = 3000
    a, b = rnd("i8", n), rnd("i8", n)
    s1, s2, so = SCALES[si]
    c = (f32(s1 / so), f32(s2 / so), f32(0))
    got = run("npu_elementwise_add_a8_xint", a, b, OUT, I64(n), *map(float, c), n=n, t="i8")
    assert_same(got, ga.elementwise_add_a8_xint(a, b, n, *c))


@pytest.mark.parametrize("e", [(0, 1, 0), (-1, -3, -2), (2, -4, 0), (-7, -7, -7), (3, 3, 3)])
@pytest.mark.parametrize("act", [ga.ACT_NONE, ga.ACT_RELU, ga.ACT_RELU6])
def test_xint_add_fused(e, act):
    """c0 = 2^e0, c1 = 2^e1 (e[2] only labels the case); clamp bounds with ties."""
    n = 4096
    a, b = rnd("i8", n), rnd("i8", n)
    c0, c1 = f32(2.0 ** e[0]), f32(2.0 ** e[1])
    a0, a1 = (f32(-20.5), f32(90.5)) if act else (f32(0), f32(0))
    got = run("npu_elementwise_add_fused_a8_xint", a, b, OUT, I64(n), float(c0), float(c1), act,
              float(a0), float(a1), n=n, t="i8")
    assert_same(got, ga.elementwise_add_fused_a8_xint(a, b, n, c0, c1, act, a0, a1))
    # the integer result equals the float model on these coefficients
    assert_same(got, ga._add(a, b, "i8", c0, c1, f32(0), act, a0, a1))


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
    want = ga.elementwise_mac_a8(x1, t1, x2, t2, x3, t3, to, s1, s2, s3, A, v, p, t)
    got = run("npu_elementwise_mac_a8", x1, T[t1][0], x2, T[t2][0], x3, T[t3][0], OUT, T[to][0],
              Host(np.array(s1, np.int64)), Host(np.array(s2, np.int64)),
              Host(np.array(s3, np.int64)), A, v, p, t, n=x1.size, t=to)
    assert_same(got, want)


def test_xint_mac_scalar_wraps():
    """A uint8 scalar ifm2 can push w past int16; it then wraps."""
    x1, x3 = rnd("i8", (2, 3, 32)), rnd("i8", (2, 3, 32))
    x2 = np.array([[[250]]], np.uint8)
    A, v, p, t = 30000, 100, 9, 7
    one = Host(np.array((1, 1, 1), np.int64))
    s = Host(np.array(x1.shape, np.int64))
    got = run("npu_elementwise_mac_a8", x1, NPU_I8, x2, NPU_U8, x3, NPU_I8, OUT, NPU_I8, s, one,
              s, A, v, p, t, n=x1.size, t="i8")
    assert_same(got, ga.elementwise_mac_a8(x1, "i8", x2, "u8", x3, "i8", "i8", x1.shape, (1, 1, 1),
                                           x3.shape, A, v, p, t))
    assert not np.array_equal(got, ga._mac_xint(x1, x2, x3, "i8", A, v, p, t).reshape(-1))


BC_SHAPES = [((4, 6, 96), (4, 6, 96)), ((4, 6, 96), (4, 6, 1)), ((4, 6, 96), (1, 1, 96)),
             ((4, 6, 96), (1, 1, 1)), ((4, 6, 96), (1, 6, 1)), ((3, 5, 40), (3, 1, 40))]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("t", [0, 1])
@pytest.mark.parametrize("shape", range(len(BC_SHAPES)))
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_bc3d(bits, t, shape, op):
    t = BITS[bits][t]
    s1, s2 = BC_SHAPES[shape]
    a, b = rnd(t, s1), rnd(t, s2)
    si = shape % len(SCALES)
    if op == "mul":
        c, _ = mul_coeffs(t, si)
    else:
        c, _ = add_coeffs(t, si, op == "sub")
    name = f"qdq_elementwise_broadcasting3d_{'mul' if op == 'mul' else 'add'}_a{bits}"
    want = getattr(ga, name)(a, t, b, s1, s2, *c)
    got = run(f"npu_{name}", a, T[t][0], b, OUT, shp(s1), shp(s2), *map(float, c), n=a.size, t=t)
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
ACTS = [ga.ACT_NONE, ga.ACT_RELU, ga.ACT_RELU6, ga.ACT_LRELU, ga.ACT_PRELU]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("t", [0, 1])
@pytest.mark.parametrize("shape", range(len(FUSED_SHAPES)))
@pytest.mark.parametrize("op", ["add", "sub"])
@pytest.mark.parametrize("act", ACTS)
def test_bc3d_addsub_fused(bits, t, shape, op, act):
    t = BITS[bits][t]
    s1, s2 = FUSED_SHAPES[shape]
    a, b = rnd(t, s1), rnd(t, s2)
    k = addsub_fused_words(t, (shape + ACTS.index(act)) % len(SCALES), op == "sub", act)
    slope = RNG.uniform(-0.6, 1.4, s1[2]).astype(f32) if act == ga.ACT_PRELU else None
    name = f"qdq_elementwise_broadcasting3d_{op}_fused_a{bits}"
    want = getattr(ga, name)(a, t, b, s1, s2, k, act, slope)
    got = run(f"npu_{name}", a, T[t][0], b, OUT, shp(s1), shp(s2), words(k), act, slope,
              n=a.size, t=t)
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
    k, c0 = mul_fused_words(t, (shape + ACTS.index(act)) % len(SCALES), act, bits == 16)
    ns = None
    if act == ga.ACT_PRELU:
        ns = (RNG.uniform(-0.6, 1.4, s1[2]).astype(f32) * c0).astype(f32)
    name = f"qdq_elementwise_broadcasting3d_mul_fused_a{bits}"
    want = getattr(ga, name)(a, t, b, s1, s2, k, act, ns)
    got = run(f"npu_{name}", a, T[t][0], b, OUT, shp(s1), shp(s2), words(k, ints=(6, 8)), act,
              ns, n=a.size, t=t)
    assert_same(got, want)


MAC_VARIANTS = {"a8w8": (8, 8, True), "a16w8": (16, 8, False), "a16w16": (16, 16, True)}
# (shape1, shape23)
MACF_SHAPES = [((3, 5, 64), (3, 5, 1)), ((3, 5, 64), (3, 5, 64)), ((3, 5, 64), (1, 1, 1)),
               ((3, 5, 64), (1, 1, 64)), ((3, 5, 64), (3, 1, 64))]


@pytest.mark.parametrize("variant", list(MAC_VARIANTS))
@pytest.mark.parametrize("signs", [(0, 0, 0), (1, 1, 1), (1, 0, 1)])
@pytest.mark.parametrize("ct", ["same", "i32"])
@pytest.mark.parametrize("shape", range(len(MACF_SHAPES)))
@pytest.mark.parametrize("act", [ga.ACT_NONE, ga.ACT_RELU, ga.ACT_RELU6, ga.ACT_LRELU])
def test_bc3d_mac_fused(variant, signs, ct, shape, act):
    abits, wbits, _ = MAC_VARIANTS[variant]
    ta, tw = BITS[abits][signs[0]], BITS[wbits][signs[1]]
    tc = BITS[abits][signs[2]] if ct == "same" else "i32"
    s1, s23 = MACF_SHAPES[shape]
    a, b, c = rnd(ta, s1), rnd(tw, s23), rnd(tc, s23)
    si = (shape + act) % len(SCALES)
    sa, sb, so = SCALES[si]
    sc = 1e-6 * so if tc == "i32" else 0.07
    z1, z2, z3, zo = zp(ta), zp(tw), zp(tc) if tc != "i32" else -12345, zp(ta)
    c0 = sa * sb / so
    c4 = sc / so
    a0, a1 = act_words(act, zo, so, ta)
    k = [f32(c0), f32(-z2 * c0), f32(-z1 * c0), f32(zo - z3 * c4), f32(c4), a0, a1, z1]
    name = f"qdq_elementwise_broadcasting3d_mac_fused_{variant}"
    want = getattr(ga, name)(a, ta, b, tw, c, tc, s1, s23, k, act)
    got = run(f"npu_{name}", a, T[ta][0], b, T[tw][0], c, T[tc][0], OUT, shp(s1), shp(s23),
              words(k, ints=(7,)), act, n=a.size, t=ta)
    assert_same(got, want)


def test_mac_fused_addend_rounding_matters():
    """With int32 addends above 2^24 the two addend conversions differ."""
    a = rnd("i16", (2, 2, 64))
    b = rnd("i8", (2, 2, 1))
    c = np.full((2, 2, 1), (1 << 26) + 3, np.int32)
    # c3 cancels the addend, leaving the 3 that the fp32 conversion rounds away
    k = [f32(1e-3), f32(0.01), f32(0), f32(-2.0 ** 26), f32(1), f32(0.1), f32(5), 7]
    assert np.any(ga._mac_fused(a, b, c, "i16", k, ga.ACT_LRELU, True) !=
                  ga._mac_fused(a, b, c, "i16", k, ga.ACT_LRELU, False))


# One device buffer for every pointer of the invalid calls: they must fail before any launch.
P = np.zeros(4096, np.uint8)
S3, BAD, ONE = shp((2, 3, 4)), shp((2, 3, 2)), shp((1, 1, 1))
K8, K9 = Host(np.zeros(8, np.uint32)), Host(np.zeros(9, np.uint32))
FLAT = [f"npu_elementwise_{op}_a{b}" for op in ("add", "mul") for b in (8, 16)]
FLAT_FUSED = [f"npu_elementwise_{op}_fused_a{b}" for op in ("add", "sub", "mul") for b in (8, 16)]
BC3D = [f"npu_qdq_elementwise_broadcasting3d_{op}_a{b}" for op in ("add", "mul") for b in (8, 16)]
BC3D_FUSED = [f"npu_qdq_elementwise_broadcasting3d_{op}_fused_a{b}"
              for op in ("add", "sub", "mul") for b in (8, 16)]
MACF = {"a8w8": (NPU_I8, NPU_I8), "a16w8": (NPU_I16, NPU_I8), "a16w16": (NPU_I16, NPU_I16)}


def _coeffs(name):
    return (1.0,) * (4 if "_mul" in name else 3)


def _bad_args():
    """Each entry of arith_int.h with argument sets it must reject (-1)."""
    out = []
    for name in FLAT:
        t = NPU_I8 if name.endswith("8") else NPU_I16
        out += [(name, P, NPU_I32, P, P, t, I64(4), *_coeffs(name)),
                (name, P, t, None, P, t, I64(4), *_coeffs(name)),
                (name, P, t, P, P, t, I64(-1), *_coeffs(name))]
    for name in FLAT_FUSED:
        t, u = (NPU_I8, NPU_U8) if name.endswith("8") else (NPU_I16, NPU_U16)
        out += [(name, P, t, P, P, u, I64(4), *_coeffs(name), 9, 0.0, 0.0),
                (name, P, t, P, None, t, I64(4), *_coeffs(name), 0, 0.0, 0.0),
                (name, P, t, P, P, NPU_I32, I64(4), *_coeffs(name), 0, 0.0, 0.0)]
    out += [("npu_elementwise_add_a8_xint", P, P, None, I64(4), 1.0, 1.0, 0.0),
            ("npu_elementwise_add_a8_xint", P, P, P, I64(-4), 1.0, 1.0, 0.0),
            ("npu_elementwise_add_fused_a8_xint", P, P, P, I64(4), 0.3, 1.0, 0, 0.0, 0.0),
            ("npu_elementwise_add_fused_a8_xint", P, P, P, I64(4), 1.0, 1.0, 3, 0.0, 0.0),
            ("npu_elementwise_add_fused_a8_xint", None, P, P, I64(4), 1.0, 1.0, 0, 0.0, 0.0),
            ("npu_elementwise_mac_a8", P, NPU_I8, P, NPU_I16, P, NPU_I8, P, NPU_I8, S3, S3, S3,
             1, 1, 0, 0),
            ("npu_elementwise_mac_a8", P, NPU_I8, P, NPU_I8, P, NPU_I8, P, NPU_I8, S3, S3, S3,
             1, 1, 32, 0),
            ("npu_elementwise_mac_a8", P, NPU_I8, P, NPU_I8, P, NPU_I8, P, NPU_I8, S3, BAD, S3,
             1, 1, 0, 0)]
    for name in BC3D:
        t = NPU_I8 if name.endswith("8") else NPU_I16
        out += [(name, P, t, P, P, S3, BAD, *_coeffs(name)),
                (name, P, NPU_U16 if t == NPU_I8 else NPU_U8, P, P, S3, S3, *_coeffs(name)),
                (name, P, t, P, None, S3, S3, *_coeffs(name))]
    for name in BC3D_FUSED:
        t = NPU_I8 if name.endswith("8") else NPU_I16
        k = K9 if "_mul" in name else K8
        out += [(name, P, t, P, P, S3, ONE, k, 9, None),    # prelu needs its array
                (name, P, t, P, P, S3, ONE, k, 0, P),       # which is null otherwise
                (name, P, t, P, P, S3, ONE, k, 4, None),
                (name, P, t, P, P, S3, ONE, None, 0, None),
                (name, P, t, None, P, S3, ONE, k, 0, None),
                (name, P, NPU_U8 if t == NPU_I16 else NPU_U16, P, P, S3, ONE, k, 0, None)]
    for v, (ta, tw) in MACF.items():
        name = f"npu_qdq_elementwise_broadcasting3d_mac_fused_{v}"
        out += [(name, P, ta, P, tw, P, NPU_I32, P, S3, BAD, K8, 0),
                (name, P, ta, P, tw, P, NPU_I32, P, S3, ONE, K8, 9),
                (name, P, ta, P, tw, P, NPU_I32, P, S3, ONE, None, 0),
                (name, P, ta, P, tw, None, NPU_I32, P, S3, ONE, K8, 0),
                (name, P, ta, P, NPU_I32, P, NPU_I32, P, S3, ONE, K8, 0)]
    return out


BAD_ARGS = _bad_args()


def test_bad_args():
    for args in BAD_ARGS:
        assert raw_rc(*args) == -1, args
    # the a16 addends must be at least 16-bit
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_mac_fused_a16w8", P, NPU_I16, P, NPU_I8, P,
                  NPU_I8, P, S3, ONE, K8, 0) == -1
    # empty tensors are a no-op
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_add_a8", P, NPU_I8, P, P, shp((2, 0, 4)),
                  shp((1, 1, 4)), 1.0, 1.0, 0.0) == 0
    assert raw_rc("npu_elementwise_add_fused_a8", P, NPU_I8, P, P, NPU_I8, I64(0), 1.0, 1.0, 0.0,
                  1, 0.0, 0.0) == 0
    # ... but its pointers are still required
    assert raw_rc("npu_elementwise_add_fused_a8", P, NPU_I8, None, P, NPU_I8, I64(0), 1.0, 1.0,
                  0.0, 1, 0.0, 0.0) == -1
    assert raw_rc("npu_qdq_elementwise_broadcasting3d_mul_fused_a16", P, NPU_I16, P, P,
                  shp((2, 0, 4)), ONE, None, 0, None) == -1
