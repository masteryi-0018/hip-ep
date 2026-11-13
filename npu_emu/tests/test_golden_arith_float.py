# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""arith_float entry points against the numpy reference (golden_arith_float.py),
bit for bit. 16-bit unary inputs are exhaustive (all 65536 patterns), fp8
unary inputs cover all 256 codes (with and without NaN rows); the binary
wrappers mix random values, values on a coarse grid (exact ties and zero sums),
signed zeros, infinities, NaN, subnormals and overflow, over every broadcast
mode, with non power-of-two coefficients (ratios such as 0.1 / 0.6)."""

import numpy as np
import pytest

import golden as g
import golden_arith_float as ga
import npu
from npu import I64, OUT, NPU_BF16, NPU_F16, NPU_FP8, call

RNG = np.random.default_rng(2028)
HALF = {"bf16": (NPU_BF16, ga.BF16), "f16": (NPU_F16, ga.F16)}
ALL16 = np.arange(65536, dtype=np.uint32).astype(np.uint16)
BCASTS = (0, 1, 2, 4, 3, 5, 6, 7)
F32 = np.float32
COEFFS = (float(F32(0.1 / 0.6)), float(F32(0.6)), float(F32(-0.3 / 0.7)), 1.0, 0.5)


def go(name, *args, dtype, n):
    return call(name, *args, out_nbytes=int(n) * np.dtype(dtype).itemsize, out_dtype=dtype)


def same(got, want, what=""):
    got = np.ascontiguousarray(got).reshape(-1)
    want = np.ascontiguousarray(np.asarray(want).astype(got.dtype)).reshape(-1)
    assert got.size == want.size, f"{what}: size {got.size} != {want.size}"
    gb, wb = got.view(f"u{got.itemsize}"), want.view(f"u{want.itemsize}")
    bad = np.nonzero(gb != wb)[0]
    assert bad.size == 0, (f"{what}: {bad.size}/{got.size} differ, first {bad[:6].tolist()}: "
                           f"got {gb[bad[:6]].tolist()} want {wb[bad[:6]].tolist()}")


# ---- input generators ----------------------------------------------------------------

def half_special(t):
    f16 = t == ga.F16
    return np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x3C00 if f16 else 0x3F80,
                     0xBC00 if f16 else 0xBF80, 0x7C00 if f16 else 0x7F80,
                     0xFC00 if f16 else 0xFF80, 0x7E00 if f16 else 0x7FC0,
                     0xFE55 if f16 else 0xFFC5, 0x7BFF if f16 else 0x7F7F,
                     0xFBFF if f16 else 0xFF7F, 0x0400 if f16 else 0x0080,
                     0x03FF if f16 else 0x007F, 0x3555 if f16 else 0x3EAB], np.uint16)


def half_values(t, n, special=True):
    """Half random-wide values, half a coarse grid (exact ties, zero sums)."""
    wide = (RNG.standard_normal(n) * np.exp2(RNG.integers(-6, 7, n))).astype(F32)
    grid = (RNG.integers(-12, 13, n) * F32(0.375)).astype(F32)
    v = np.where(RNG.random(n) < 0.5, wide, grid)
    bits = ga.round16(v, t)
    bits = np.where((v == 0) & (RNG.random(n) < 0.5), bits | np.uint16(0x8000), bits)
    if special:
        sp = half_special(t)
        k = n // 8
        pos = RNG.choice(n, k, replace=False)
        bits[pos] = sp[RNG.integers(0, len(sp), k)]
    return bits.astype(np.uint16)


def fp8_values(n, nan_rate=0.01):
    c = RNG.integers(0, 256, n).astype(np.uint8)
    c = np.where((c & 0x7F) == 0x7F, np.uint8(0x38), c)
    c = np.where(RNG.random(n) < nan_rate, np.uint8(0x7F), c)
    c = np.where(RNG.random(n) < 0.05, RNG.choice(np.array([0x00, 0x80], np.uint8), n), c)
    return c.astype(np.uint8)


def bfp_values(n_vals, zero_rate=0.0):
    v = (RNG.standard_normal(n_vals) * np.exp2(RNG.integers(-8, 8, n_vals // 8).repeat(8))
         ).astype(F32)
    v = np.where(RNG.random(n_vals) < 0.2, (RNG.integers(-8, 9, n_vals) * F32(0.25)), v)
    v = np.where(RNG.random(n_vals) < zero_rate, F32(0.0), v).astype(F32)
    return g.to_bfp16(v, g.EVEN)


SHAPES = ((3, 5, 24), (2, 3, 64), (1, 1, 40))


# ---- bf16 / f16 broadcasting ------------------------------------------------------------

@pytest.mark.parametrize("t", list(HALF))
@pytest.mark.parametrize("bc", BCASTS)
def test_add_mul_div_half(t, bc):
    nt, ht = HALF[t]
    for H, W, C in SHAPES:
        x1 = half_values(ht, H * W * C)
        x2 = half_values(ht, int(np.prod(ga.bshape(bc, H, W, C))))
        dims = (I64(H), I64(W), I64(C))
        n = H * W * C
        for n1, n2 in ((0, 0), (0, 1), (1, 0)):
            for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.0, 6.0), (3, 0.1015625, 0.0),
                                (3, 0.3, 0.0)):
                got = go("npu_elementwise_broadcasting3d_add_bf16", x1, x2, bc, nt, OUT, *dims,
                         n1, n2, act, a0, a1, dtype=np.uint16, n=n)
                same(got, ga.add16(x1, x2, bc, ht, H, W, C, n1, n2, act, a0, a1),
                     f"add {n1}{n2} act{act}")
        for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, -1.5, 6.0), (3, 0.2, 0.0)):
            got = go("npu_elementwise_broadcasting3d_mul_bf16", x1, x2, bc, nt, OUT, *dims,
                     act, a0, a1, dtype=np.uint16, n=n)
            same(got, ga.mul16(x1, x2, bc, ht, H, W, C, act, a0, a1), f"mul act{act}")
        for sw in (0, 1):
            got = go("npu_elementwise_broadcasting3d_div_bf16", x1, x2, bc, nt, OUT, *dims, sw,
                     dtype=np.uint16, n=n)
            same(got, ga.div16(x1, x2, bc, ht, H, W, C, sw), f"div swap{sw}")


@pytest.mark.parametrize("t", list(HALF))
def test_div_half_all_divisors(t):
    """Every 16-bit divisor (the reciprocal table / rule), numerators 1 and 3."""
    nt, ht = HALF[t]
    for num in (1.0, 3.0, -0.0):
        x1 = np.full(65536, ga.round16(F32(num), ht), np.uint16)
        got = go("npu_elementwise_broadcasting3d_div_bf16", x1, ALL16, 0, nt, OUT, I64(1),
                 I64(256), I64(256), 0, dtype=np.uint16, n=65536)
        same(got, ga.div16(x1, ALL16, 0, ht, 1, 256, 256, 0), f"div {num}")


def test_inv_bf16_matches_mllib_table():
    """The reciprocal rule equals mllib's inv_bf16_aie4 table (copied values for
    the special classes: zero, subnormal, overflow, NaN)."""
    y = g.bf16_bits_to_f32(ALL16)
    r = ga.inv_bf16(y).view(np.uint32) >> 16
    for b, want in ((0x0000, 0x7F80), (0x8000, 0xFF80), (0x0001, 0x7F80), (0x8001, 0xFF80),
                    (0x7F80, 0x0000), (0xFF80, 0x8000), (0x7F7F, 0x0000), (0x7F00, 0x0000),
                    (0x7E80, 0x0080), (0x7E00, 0x0100), (0x7F81, 0x7FC1), (0xFFC0, 0xFFC0),
                    (0x3F80, 0x3F80), (0x4040, 0x3EAB)):
        assert r[b] == want, hex(b)


@pytest.mark.parametrize("t", list(HALF))
@pytest.mark.parametrize("bc", BCASTS)
def test_mac_fused_half(t, bc):
    nt, ht = HALF[t]
    for H, W, C in SHAPES:
        n = H * W * C
        m = int(np.prod(ga.bshape(bc, H, W, C)))
        x1, x2, x3 = half_values(ht, n), half_values(ht, m), half_values(ht, m)
        for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.0, 6.0), (3, 0.1015625, 0.0)):
            got = go("npu_elementwise_broadcasting3d_mac_fused_bf16", x1, x2, x3, bc, nt, OUT,
                     I64(H), I64(W), I64(C), act, a0, a1, dtype=np.uint16, n=n)
            same(got, ga.mac16(x1, x2, x3, bc, ht, H, W, C, act, a0, a1), f"mac act{act}")


# ---- bfp16 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("bc", BCASTS)
def test_bfp16_binary(bc):
    for H, W, C in SHAPES:
        nbl = H * W * C // 8
        x1 = bfp_values(H * W * C, zero_rate=0.05)
        s = ga.bshape(bc, H, W, C)
        x2 = bfp_values(s[0] * s[1] * (8 if bc & 1 else s[2]), zero_rate=0.0)
        dims = (I64(H), I64(W), I64(C))
        for op in ("add", "sub", "mul", "div"):
            for sw in ((0, 1) if op in ("sub", "div") else (0,)):
                extra = (sw,) if op in ("sub", "div") else ()
                got = go(f"npu_elementwise_broadcasting3d_{op}_bfp16", x1, x2, bc, OUT, *dims,
                         *extra, dtype=np.uint8, n=nbl * 9)
                same(got, ga.bfp_bin(op, x1, x2, bc, H, W, C, sw), f"{op} swap{sw}")


@pytest.mark.parametrize("bc2,bc3", [(0, 0), (2, 0), (4, 6), (6, 2)])
@pytest.mark.parametrize("act_type,activation", [(0, 0), (100, 0), (100, 100), (7, 0), (7, 7)])
def test_mul_add_bfp16(bc2, bc3, act_type, activation):
    H, W, C = 3, 4, 64
    n = H * W * C
    x1 = bfp_values(n)
    lo, hi = -2.0, 1.5
    for b2 in (0, 1):
        for b3 in (0, 1):
            m2 = int(np.prod(ga.bshape(bc2, H, W, C)))
            m3 = int(np.prod(ga.bshape(bc3, H, W, C)))
            x2 = bfp_values(m2) if b2 else half_values(ga.BF16, m2, special=False)
            x3 = bfp_values(m3) if b3 else half_values(ga.BF16, m3, special=False)
            for ob in (0, 1):
                got = go("npu_elementwise_broadcasting3d_mul_add_bfp16", x1, x2, b2, bc2, x3,
                         b3, bc3, OUT, ob, I64(H), I64(W), I64(C), act_type, activation, lo,
                         hi, dtype=np.uint8, n=n // 8 * 9 if ob else n * 2)
                ref = ga.mul_add(x1, x2, b2, bc2, x3, b3, bc3, ob, H, W, C, act_type,
                                 activation, lo, hi)
                same(got, np.asarray(ref).view(np.uint8), f"b{b2}{b3} out{ob}")


def test_sigtanh_all_bf16():
    """SigTanh on every bf16 value (through mul_add: x1 = 1, x2 = x, x3 = 0)."""
    H, W, C = 1, 256, 256
    x1 = g.to_bfp16(np.ones(H * W * C, F32), g.EVEN)
    x3 = np.zeros(H * W * C, np.uint16)
    got = go("npu_elementwise_broadcasting3d_mul_add_bfp16", x1, ALL16, 0, 0, x3, 0, 0, OUT,
             0, I64(H), I64(W), I64(C), 7, 7, 0.0, 0.0, dtype=np.uint16, n=65536)
    same(got, ga.mul_add(x1, ALL16, 0, 0, x3, 0, 0, 0, H, W, C, 7, 7, 0.0, 0.0), "sigtanh")
    # The fit against tanh: finite inputs stay within 2^-7 absolute.
    v = g.bf16_bits_to_f32(got)
    x = g.bf16_bits_to_f32(ALL16)
    ok = np.isfinite(x)
    assert np.max(np.abs(v[ok] - np.tanh(x[ok].astype(np.float64)))) < 2 ** -7


# ---- fp8 -------------------------------------------------------------------------------------

def table(c1, c2, act, alpha):
    """The YAML coeff table of add/sub fp8: [code] = fp32(decode(code) * c2 * A),
    then A * c1 and B / A (A = (1 + alpha) / 2, B = |1 - alpha| / 2; A = 1 for CLIP)."""
    d = g.fp8_to_f32(np.arange(256)).astype(np.float64)
    a = 1.0 if act == ga.CLIP else (1.0 + alpha) * 0.5
    t = (d * c2 * a).astype(F32)
    t = np.where(np.isnan(t), F32(0.0), t)
    return np.concatenate([t, np.array([F32(c1 * a), F32(abs(1 - alpha) / 2 / a)], F32)])


@pytest.mark.parametrize("bc", BCASTS)
def test_add_fp8(bc):
    for H, W, C in SHAPES:
        n = H * W * C
        m = int(np.prod(ga.bshape(bc, H, W, C)))
        x1, x2 = fp8_values(n), fp8_values(m)
        for c1, c2 in ((COEFFS[0], COEFFS[1]), (COEFFS[0], -COEFFS[1]), (-COEFFS[2], COEFFS[0])):
            for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.0, 12.0),
                                (3, 0.1015625, 0.0)):
                tb = table(c1, c2, act, a0) if act in (2, 3) else None
                got = go("npu_elementwise_broadcasting3d_add_fp8", x1, x2, bc, OUT, I64(H),
                         I64(W), I64(C), c1, c2, act, a0, a1, tb, dtype=np.uint8, n=n)
                same(got, ga.add_fp8(x1, x2, bc, H, W, C, c1, c2, act, a0, a1, tb),
                     f"add c{c1},{c2} act{act}")


@pytest.mark.parametrize("bc", BCASTS)
def test_mul_div_fp8(bc):
    for H, W, C in SHAPES:
        n = H * W * C
        m = int(np.prod(ga.bshape(bc, H, W, C)))
        x1, x2 = fp8_values(n), fp8_values(m)
        for s in COEFFS:
            got = go("npu_elementwise_broadcasting3d_mul_fp8", x1, x2, bc, OUT, I64(H), I64(W),
                     I64(C), s, dtype=np.uint8, n=n)
            same(got, ga.mul_fp8(x1, x2, bc, H, W, C, s), f"mul {s}")
            for sw in (0, 1):
                got = go("npu_elementwise_broadcasting3d_div_fp8", x1, x2, bc, OUT, I64(H),
                         I64(W), I64(C), s, sw, dtype=np.uint8, n=n)
                same(got, ga.div_fp8(x1, x2, bc, H, W, C, s, sw), f"div {s} swap{sw}")


def test_div_fp8_reciprocal():
    """aie::inv(1.75) is one ulp below RNE(1 / 1.75): 1.859375 / 1.75 lands just
    below the fp8 tie 1.0625 (-> 1.0, 0x38) instead of on it (-> 1.125, 0x39)."""
    x1, x2 = np.full(8, 0x38, np.uint8), np.full(8, 0x3E, np.uint8)   # 1.0, 1.75
    got = go("npu_elementwise_broadcasting3d_div_fp8", x1, x2, 0, OUT, I64(1), I64(1), I64(8),
             1.859375, 0, dtype=np.uint8, n=8)
    same(got, np.full(8, 0x38, np.uint8), "div 1.0 * 1.859375 / 1.75")
    same(got, ga.div_fp8(x1, x2, 0, 1, 1, 8, 1.859375, 0), "golden")


@pytest.mark.parametrize("bc", BCASTS)
def test_mac_fused_fp8(bc):
    for H, W, C in SHAPES:
        n = H * W * C
        m = int(np.prod(ga.bshape(bc, H, W, C)))
        x1, x2, x3 = fp8_values(n), fp8_values(m), fp8_values(m)
        for c0, c4 in ((COEFFS[0], COEFFS[1]), (COEFFS[2], COEFFS[0])):
            for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.0, 24.0),
                                (3, 0.1015625, 0.0)):
                got = go("npu_elementwise_broadcasting3d_mac_fused_fp8", x1, x2, x3, bc, OUT,
                         I64(H), I64(W), I64(C), c0, c4, act, a0, a1, dtype=np.uint8, n=n)
                same(got, ga.mac_fp8(x1, x2, x3, bc, H, W, C, c0, c4, act, a0, a1),
                     f"mac act{act}")


# ---- unary / per channel ------------------------------------------------------------------------

@pytest.mark.parametrize("t", list(HALF))
def test_unary_half_exhaustive(t):
    nt, ht = HALF[t]
    for alpha in (0.01, 0.1015625, 0.3, 0.1 / 0.6):
        got = go("npu_elementwise_leakyrelu_bf16", ALL16, nt, OUT, I64(65536), alpha,
                 dtype=np.uint16, n=65536)
        same(got, ga.leakyrelu16(ALL16, ht, alpha), f"leakyrelu {alpha}")
    got = go("npu_elementwise_square_bf16", ALL16, nt, OUT, I64(65536), dtype=np.uint16,
             n=65536)
    same(got, ga.square16(ALL16, ht), "square")
    for C in (64, 96):
        D = (RNG.standard_normal(C) * 0.7 - 1.0).astype(F32)
        D[:4] = (F32(0.0), F32(-1.0), F32(0.25 - 1), F32(0.1 - 1))
        got = go("npu_elementwise_prelu_bf16", ALL16, nt, OUT, I64(65536), D, I64(C),
                 dtype=np.uint16, n=65536)
        same(got, ga.prelu16(ALL16, ht, D, C), f"prelu C{C}")


def fp8_all(nan_rows):
    """All 256 codes, each in every row position; NaN rows optional."""
    c = np.concatenate([(np.arange(256) + k) % 256 for k in range(8)]).astype(np.uint8)
    if not nan_rows:
        c = np.where((c & 0x7F) == 0x7F, np.uint8(0x00), c)
    return c.astype(np.uint8)


@pytest.mark.parametrize("nan_rows", [False, True])
def test_unary_fp8(nan_rows):
    x = fp8_all(nan_rows)
    n = x.size
    for pos, neg in ((COEFFS[0], F32(COEFFS[0] * 0.01)), (2.0, 0.02), (COEFFS[2], 0.5)):
        got = go("npu_elementwise_leakyrelu_fp8", x, OUT, I64(n), float(pos), float(neg),
                 dtype=np.uint8, n=n)
        same(got, ga.leakyrelu_fp8(x, pos, neg), "leakyrelu")
    for s in COEFFS + (1e-3, 64.0):
        got = go("npu_elementwise_square_fp8", x, OUT, I64(n), s, dtype=np.uint8, n=n)
        same(got, ga.square_fp8(x, s), f"square {s}")
    for P in (COEFFS[0], 2.0):
        C = 64
        D = ((RNG.standard_normal(C) * 0.7 - 1.0) * P).astype(F32)
        D[0] = F32(0.0)
        D[1] = F32(-P)
        got = go("npu_elementwise_prelu_fp8", x, OUT, I64(n), float(P), D, I64(C),
                 dtype=np.uint8, n=n)
        same(got, ga.prelu_fp8(x, P, D, C), f"prelu {P}")


@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("act", [0, 1, 3])
def test_batchnorm(t, act):
    C = 64
    if t == "fp8":
        x = np.concatenate([fp8_all(True), fp8_values(4096)])
        nt, wt = NPU_FP8, ga.F16
    else:
        nt, wt = HALF[t]
        x = np.concatenate([ALL16, half_values(wt, 4096)])
    n = x.size - x.size % C
    x = np.ascontiguousarray(x[:n])
    cw = ga.load16(ga.round16((RNG.standard_normal(C) * 0.6).astype(F32), wt), wt)
    cb = ga.load16(ga.round16((RNG.standard_normal(C) * 3).astype(F32), wt), wt)
    cw[:3] = (F32(0.0), F32(-0.0), F32(1.0))
    cb[:3] = (F32(-0.0), F32(0.0), F32(-0.0))
    cw, cb = np.ascontiguousarray(cw), np.ascontiguousarray(cb)
    for alpha in (0.1015625, 0.1 / 0.6):
        dt = np.uint8 if t == "fp8" else np.uint16
        got = go("npu_batchnorm2d_stream_bf16", x, nt, OUT, I64(n), I64(C), cw, cb, act,
                 float(F32(alpha)), dtype=dt, n=n)
        same(got, ga.batchnorm(x, t if t == "fp8" else wt, C, cw, cb, act, F32(alpha)),
             f"bn {t} act{act}")


# ---- argument checks ----------------------------------------------------------------------------

def test_bad_arguments():
    rc = npu.raw_rc
    nul = None
    assert rc("npu_elementwise_broadcasting3d_add_bf16", nul, nul, 0, NPU_FP8, nul, I64(1),
              I64(1), I64(8), 0, 0, 0, 0.0, 0.0) == -1
    assert rc("npu_elementwise_broadcasting3d_mul_bf16", nul, nul, 8, NPU_BF16, nul, I64(1),
              I64(1), I64(8), 0, 0.0, 0.0) == -1
    assert rc("npu_elementwise_broadcasting3d_add_bfp16", nul, nul, 0, nul, I64(1), I64(1),
              I64(12)) == -1
    assert rc("npu_elementwise_broadcasting3d_mul_add_bfp16", nul, nul, 0, 1, nul, 0, 0, nul,
              0, I64(1), I64(1), I64(8), 0, 0, 0.0, 0.0) == -1
    assert rc("npu_elementwise_broadcasting3d_mul_add_bfp16", nul, nul, 0, 0, nul, 0, 0, nul,
              0, I64(1), I64(1), I64(8), 5, 0, 0.0, 0.0) == -1
    assert rc("npu_elementwise_broadcasting3d_add_fp8", nul, nul, 1, nul, I64(1), I64(1),
              I64(8), 1.0, 1.0, 3, 0.1, 0.0, nul) == -1
    assert rc("npu_elementwise_broadcasting3d_mac_fused_fp8", nul, nul, nul, 0, nul, I64(1),
              I64(1), I64(8), 1.0, 1.0, 9, 0.0, 0.0) == -1
    assert rc("npu_elementwise_prelu_bf16", nul, NPU_BF16, nul, I64(8), nul, I64(0)) == -1
    assert rc("npu_batchnorm2d_stream_bf16", nul, NPU_BF16, nul, I64(8), I64(8), nul, nul, 2,
              0.0) == -1
    # Empty tensors are a no-op.
    assert rc("npu_elementwise_square_fp8", nul, nul, I64(0), 1.0) == 0
