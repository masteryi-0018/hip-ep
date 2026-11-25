# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Floating-point arithmetic entry points against the numpy reference, bit for bit."""

import numpy as np
import pytest

import golden_arith_float as ga
from golden_block_fp import decode_bfp16, encode_bfp16
from golden_float_cvt import bf16_to_f32, fp8_to_f32, store_float
from golden_int_round import Rnd
import npu
from npu import I64, OUT, NPU_BF16, NPU_BFP16, NPU_F16, NPU_FP8, NPU_I8, NPU_I16, NPU_U16
import testing
from testing import ALL16, HALF, half_values, same, shp
from testing import call_n as go

RNG = np.random.default_rng(2028)
BCASTS = (0, 1, 2, 4, 3, 5, 6, 7)
F32 = np.float32
COEFFS = (float(F32(0.1 / 0.6)), float(F32(0.6)), float(F32(-0.3 / 0.7)), 1.0, 0.5)


def half_special(t):
    f16 = t == ga.F16
    return np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x3C00 if f16 else 0x3F80,
                     0xBC00 if f16 else 0xBF80, 0x7C00 if f16 else 0x7F80,
                     0xFC00 if f16 else 0xFF80, 0x7E00 if f16 else 0x7FC0,
                     0xFE55 if f16 else 0xFFC5, 0x7BFF if f16 else 0x7F7F,
                     0xFBFF if f16 else 0xFF7F, 0x0400 if f16 else 0x0080,
                     0x03FF if f16 else 0x007F, 0x3555 if f16 else 0x3EAB], np.uint16)


def fp8_values(n, nan_rate=0.01):
    return testing.fp8_values(RNG, n, nan_rate)


def bfp_values(n_vals, zero_rate=0.0):
    v = (RNG.standard_normal(n_vals) * np.exp2(RNG.integers(-8, 8, n_vals // 8).repeat(8))
         ).astype(F32)
    v = np.where(RNG.random(n_vals) < 0.2, (RNG.integers(-8, 9, n_vals) * F32(0.25)), v)
    v = np.where(RNG.random(n_vals) < zero_rate, F32(0.0), v).astype(F32)
    return encode_bfp16(v, Rnd.kHalfEven)


SHAPES = ((3, 5, 24), (2, 3, 64), (1, 1, 40))


def tshapes(H, W, C, *bcs):
    """The shapes of x1 (H x W x C) and of operands broadcast along bcs."""
    return ((H, W, C),) + tuple(ga._bshape(bc, H, W, C) for bc in bcs)


def shapes(H, W, C, *bcs):
    """tshapes as the host shape arrays the entries take."""
    return tuple(shp(s) for s in tshapes(H, W, C, *bcs))


@pytest.mark.parametrize("t", list(HALF))
@pytest.mark.parametrize("bc", BCASTS)
def test_add_mul_div_half(t, bc):
    nt, ht = HALF[t]
    for H, W, C in SHAPES:
        x1 = half_values(RNG, ht, H * W * C, half_special(ht))
        x2 = half_values(RNG, ht, int(np.prod(ga._bshape(bc, H, W, C))), half_special(ht))
        dims, sh = shapes(H, W, C, bc), tshapes(H, W, C, bc)
        n = H * W * C
        for sub, neg in ((0, 0), (0, 1), (1, 0)):
            for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.0, 6.0), (3, 0.1015625, 0.0),
                                (3, 0.3, 0.0)):
                got = go("npu_elementwise_broadcasting3d_add_bf16", x1, nt, x2, OUT, *dims,
                         sub, neg, act, a0, a1, dt=np.uint16, n=n)
                same(got, ga.elementwise_broadcasting3d_add_bf16(x1, ht, x2, *sh, sub, neg, act,
                                                                 a0, a1),
                     f"add {sub}{neg} act{act}")
        for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, -1.5, 6.0), (3, 0.2, 0.0)):
            got = go("npu_elementwise_broadcasting3d_mul_bf16", x1, nt, x2, OUT, *dims,
                     act, a0, a1, dt=np.uint16, n=n)
            same(got, ga.elementwise_broadcasting3d_mul_bf16(x1, ht, x2, *sh, act, a0, a1),
                 f"mul act{act}")
        for sw in (0, 1):
            got = go("npu_elementwise_broadcasting3d_div_bf16", x1, nt, x2, OUT, *dims, sw,
                     dt=np.uint16, n=n)
            same(got, ga.elementwise_broadcasting3d_div_bf16(x1, ht, x2, *sh, sw), f"div swap{sw}")


@pytest.mark.parametrize("t", list(HALF))
def test_div_half_all_divisors(t):
    """Every 16-bit divisor."""
    nt, ht = HALF[t]
    for num in (1.0, 3.0, -0.0):
        x1 = np.full(65536, store_float(F32(num), ht), np.uint16)
        got = go("npu_elementwise_broadcasting3d_div_bf16", x1, nt, ALL16, OUT,
                 *shapes(1, 256, 256, 0), 0, dt=np.uint16, n=65536)
        same(got, ga.elementwise_broadcasting3d_div_bf16(x1, ht, ALL16, *tshapes(1, 256, 256, 0),
                                                         0), f"div {num}")


def test_inv_bf16_table():
    """Reciprocal special classes: zero, subnormal, overflow, NaN."""
    y = bf16_to_f32(ALL16)
    r = ga._inv16(y, "bf16").view(np.uint32) >> 16
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
        m = int(np.prod(ga._bshape(bc, H, W, C)))
        sp = half_special(ht)
        x1, x2, x3 = (half_values(RNG, ht, k, sp) for k in (n, m, m))
        for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.0, 6.0), (3, 0.1015625, 0.0)):
            got = go("npu_elementwise_broadcasting3d_mac_fused_bf16", x1, nt, x2, x3, OUT,
                     *shapes(H, W, C, bc), act, a0, a1, dt=np.uint16, n=n)
            same(got, ga.elementwise_broadcasting3d_mac_fused_bf16(
                x1, ht, x2, x3, *tshapes(H, W, C, bc), act, a0, a1), f"mac act{act}")


@pytest.mark.parametrize("bc", BCASTS)
def test_bfp16_binary(bc):
    for H, W, C in SHAPES:
        nbl = H * W * C // 8
        x1 = bfp_values(H * W * C, zero_rate=0.05)
        s = ga._bshape(bc, H, W, C)
        x2 = bfp_values(s[0] * s[1] * (8 if bc & 1 else s[2]), zero_rate=0.0)
        dims = shapes(H, W, C, bc)
        for op in ("add", "sub", "mul", "div"):
            name = f"elementwise_broadcasting3d_{op}_bfp16"
            for sw in ((0, 1) if op in ("sub", "div") else (0,)):
                extra = (sw,) if op in ("sub", "div") else ()
                got = go(f"npu_{name}", x1, x2, OUT, *dims, *extra, dt=np.uint8, n=nbl * 9)
                same(got, getattr(ga, name)(x1, x2, *tshapes(H, W, C, bc), *extra),
                     f"{op} swap{sw}")


@pytest.mark.parametrize("bc2,bc3", [(0, 0), (2, 0), (4, 6), (6, 2)])
@pytest.mark.parametrize("act_type,activation", [(0, 0), (100, 0), (100, 100), (7, 0), (7, 7)])
def test_mul_add_bfp16(bc2, bc3, act_type, activation):
    H, W, C = 3, 4, 64
    n = H * W * C
    x1 = bfp_values(n)
    lo, hi = 0xC000, 0x3FC0   # bf16 -2.0, 1.5
    for b2 in (0, 1):
        for b3 in (0, 1):
            m2 = int(np.prod(ga._bshape(bc2, H, W, C)))
            m3 = int(np.prod(ga._bshape(bc3, H, W, C)))
            x2 = bfp_values(m2) if b2 else half_values(RNG, ga.BF16, m2)
            x3 = bfp_values(m3) if b3 else half_values(RNG, ga.BF16, m3)
            for ob in (0, 1):
                got = go("npu_elementwise_broadcasting3d_mul_add_bfp16", x1, x2, BFP[b2], x3,
                         BFP[b3], OUT, BFP[ob], *shapes(H, W, C, bc2, bc3), act_type,
                         activation, lo, hi, dt=np.uint8, n=n // 8 * 9 if ob else n * 2)
                ref = ga.elementwise_broadcasting3d_mul_add_bfp16(
                    x1, x2, BFPT[b2], x3, BFPT[b3], BFPT[ob], *tshapes(H, W, C, bc2, bc3),
                    act_type, activation, lo, hi)
                same(got, ref, f"b{b2}{b3} out{ob}")


# npu_dtype / model type of a mul_add operand: 0 bf16, 1 bfp16
BFP = (NPU_BF16, NPU_BFP16)
BFPT = ("bf16", "bfp16")


def wide_blocks(nblk, span, top):
    """Random bfp16 blocks with exponent top or 18 .. span below it (some all-zero)."""
    b = RNG.integers(0, 256, (nblk, 9)).astype(np.uint8)
    b[:, 0] = np.where(RNG.random(nblk) < 0.25, top, top - RNG.integers(18, span + 1, nblk))
    b[RNG.random(nblk) < 0.05] = 0
    return b.reshape(-1)


def exact_decode(blk, row=None):
    return decode_bfp16(blk)


def exact_bin(op, x1, x2, bc, H, W, C, sw):
    """Exactly decoded operands combined in fp32."""
    a = decode_bfp16(x1).reshape(H, W, C)
    s = ga._bshape(bc, H, W, C)
    b = decode_bfp16(x2).reshape(s[0], s[1], -1)[:, :, :1 if bc & 1 else None]
    b = np.broadcast_to(b, (H, W, C))
    if op == "add":
        v = ga.add_q(a, b)
    elif op == "sub":
        v = ga.add_q(b, -a) if sw else ga.add_q(a, -b)
    elif op == "mul":
        v = ga.mul_q(a, b)
    else:
        num, den = (b, a) if sw else (a, b)
        v = ga.mul_q(num, ga._inv16(den, "bf16"))
    return encode_bfp16(v.reshape(-1), Rnd.kHalfEven)


@pytest.mark.parametrize("span", [20, 30])
def test_bfp16_binary_wide_groups(span):
    """Exponents spread over more than 17 / 25 inside a 64-element group: the identity-multiply
    decode rounds the small blocks; add is one block MMAC onto x2."""
    changed = set()
    for bc in BCASTS:
        for H, W, C in ((2, 3, 128), (3, 4, 80), (1, 9, 64)):
            x1 = wide_blocks(H * W * C // 8, span, 140)
            s = ga._bshape(bc, H, W, C)
            x2 = wide_blocks(s[0] * s[1] * (1 if bc & 1 else s[2] // 8), span, 134)
            dims = shapes(H, W, C, bc)
            for op in ("add", "sub", "mul", "div"):
                name = f"elementwise_broadcasting3d_{op}_bfp16"
                for sw in ((0, 1) if op in ("sub", "div") else (0,)):
                    extra = (sw,) if op in ("sub", "div") else ()
                    got = go(f"npu_{name}", x1, x2, OUT, *dims, *extra, dt=np.uint8,
                             n=H * W * C // 8 * 9)
                    want = getattr(ga, name)(x1, x2, *tshapes(H, W, C, bc), *extra)
                    same(got, want, f"{op} swap{sw} bc{bc} C{C}")
                    if (exact_bin(op, x1, x2, bc, H, W, C, sw) != want).any():
                        changed.add((op, sw, bc & 1))
            a, b = ga._bfp_values(x1, (H, W, C)), ga._bfp_operand(x2, bc, H, W, C)
            if (ga._bfp_bin("add", x1, x2, bc, H, W, C) !=
                    encode_bfp16(ga.add_q(a, b).reshape(-1), Rnd.kHalfEven)).any():
                changed.add(("mac", bc & 1))
    ops = [("add", 0), ("sub", 0), ("sub", 1), ("mul", 0), ("div", 0), ("div", 1)]
    assert changed == {(op, sw, c) for op, sw in ops for c in (0, 1)} | {("mac", 0), ("mac", 1)}


@pytest.mark.parametrize("span", [20, 30])
def test_mul_add_bfp16_wide_groups(span, monkeypatch):
    H, W, C = 3, 4, 128
    n = H * W * C
    x1 = wide_blocks(n // 8, span, 140)
    for bc2, bc3 in ((0, 0), (2, 4)):
        m2 = int(np.prod(ga._bshape(bc2, H, W, C)))
        m3 = int(np.prod(ga._bshape(bc3, H, W, C)))
        for b2, b3 in ((1, 1), (0, 1), (1, 0)):
            x2 = wide_blocks(m2 // 8, span, 134) if b2 else half_values(RNG, ga.BF16, m2)
            x3 = wide_blocks(m3 // 8, span, 134) if b3 else half_values(RNG, ga.BF16, m3)
            for ob in (0, 1):
                args = (x1, x2, BFPT[b2], x3, BFPT[b3], BFPT[ob], *tshapes(H, W, C, bc2, bc3),
                        0, 0, 0, 0)
                got = go("npu_elementwise_broadcasting3d_mul_add_bfp16", x1, x2, BFP[b2], x3,
                         BFP[b3], OUT, BFP[ob], *shapes(H, W, C, bc2, bc3), 0, 0, 0, 0,
                         dt=np.uint8, n=n // 8 * 9 if ob else n * 2)
                want = ga.elementwise_broadcasting3d_mul_add_bfp16(*args)
                same(got, want, f"b{b2}{b3} out{ob}")
                with monkeypatch.context() as m:
                    m.setattr(ga, "decode_bfp16_identity", exact_decode)
                    assert (ga.elementwise_broadcasting3d_mul_add_bfp16(*args) != want).any()


def test_sigtanh_all_bf16():
    """The tanh path on every bf16 value (through mul_add: x1 = 1, x2 = x, x3 = 0)."""
    H, W, C = 1, 256, 256
    x1 = encode_bfp16(np.ones(H * W * C, F32), Rnd.kHalfEven)
    x3 = np.zeros(H * W * C, np.uint16)
    got = go("npu_elementwise_broadcasting3d_mul_add_bfp16", x1, ALL16, NPU_BF16, x3, NPU_BF16,
             OUT, NPU_BF16, *shapes(H, W, C, 0, 0), 7, 7, 0, 0, dt=np.uint16, n=65536)
    same(got, ga.elementwise_broadcasting3d_mul_add_bfp16(
        x1, ALL16, "bf16", x3, "bf16", "bf16", *tshapes(H, W, C, 0, 0), 7, 7, 0, 0).view(
            np.uint16), "sigtanh")
    # Finite inputs stay within 2^-7 of tanh.
    v = bf16_to_f32(got)
    x = bf16_to_f32(ALL16)
    ok = np.isfinite(x)
    assert np.max(np.abs(v[ok] - np.tanh(x[ok].astype(np.float64)))) < 2 ** -7


def table(c1, c2, act, alpha):
    """Coefficient table of add / sub fp8: [code] = fp32(decode(code) * c2 * A), then A * c1
    and B / A (A = (1 + alpha) / 2, B = |1 - alpha| / 2; A = 1 for CLIP)."""
    d = fp8_to_f32(np.arange(256)).astype(np.float64)
    a = 1.0 if act == ga.RELU6 else (1.0 + alpha) * 0.5
    t = (d * c2 * a).astype(F32)
    t = np.where(np.isnan(t), F32(0.0), t)
    return np.concatenate([t, np.array([F32(c1 * a), F32(abs(1 - alpha) / 2 / a)], F32)])


@pytest.mark.parametrize("bc", BCASTS)
def test_add_fp8(bc):
    for H, W, C in SHAPES:
        n = H * W * C
        m = int(np.prod(ga._bshape(bc, H, W, C)))
        x1, x2 = fp8_values(n), fp8_values(m)
        for c1, c2 in ((COEFFS[0], COEFFS[1]), (COEFFS[0], -COEFFS[1]), (-COEFFS[2], COEFFS[0])):
            for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.0, 12.0),
                                (3, 0.1015625, 0.0)):
                tb = table(c1, c2, act, a0) if act in (2, 3) else None
                got = go("npu_elementwise_broadcasting3d_add_fp8", x1, x2, OUT,
                         *shapes(H, W, C, bc), c1, c2, act, a0, a1, tb, dt=np.uint8, n=n)
                same(got, ga.elementwise_broadcasting3d_add_fp8(
                    x1, x2, *tshapes(H, W, C, bc), c1, c2, act, a0, a1, tb),
                    f"add c{c1},{c2} act{act}")


@pytest.mark.parametrize("bc", BCASTS)
def test_mul_div_fp8(bc):
    for H, W, C in SHAPES:
        n = H * W * C
        m = int(np.prod(ga._bshape(bc, H, W, C)))
        x1, x2 = fp8_values(n), fp8_values(m)
        for s in COEFFS:
            sh = tshapes(H, W, C, bc)
            got = go("npu_elementwise_broadcasting3d_mul_fp8", x1, x2, OUT, *shapes(H, W, C, bc),
                     s, dt=np.uint8, n=n)
            same(got, ga.elementwise_broadcasting3d_mul_fp8(x1, x2, *sh, s), f"mul {s}")
            for sw in (0, 1):
                got = go("npu_elementwise_broadcasting3d_div_fp8", x1, x2, OUT,
                         *shapes(H, W, C, bc), s, sw, dt=np.uint8, n=n)
                same(got, ga.elementwise_broadcasting3d_div_fp8(x1, x2, *sh, s, sw),
                     f"div {s} swap{sw}")


def test_div_fp8_reciprocal():
    """The hardware reciprocal of 1.75 is one ulp below RNE(1 / 1.75): the quotient falls
    below the fp8 tie 1.0625."""
    x1, x2 = np.full(8, 0x38, np.uint8), np.full(8, 0x3E, np.uint8)   # 1.0, 1.75
    got = go("npu_elementwise_broadcasting3d_div_fp8", x1, x2, OUT, *shapes(1, 1, 8, 0),
             1.859375, 0, dt=np.uint8, n=8)
    same(got, np.full(8, 0x38, np.uint8), "div 1.0 * 1.859375 / 1.75")
    same(got, ga.elementwise_broadcasting3d_div_fp8(x1, x2, (1, 1, 8), (1, 1, 8), 1.859375, 0),
         "golden")


@pytest.mark.parametrize("bc", BCASTS)
def test_mac_fused_fp8(bc):
    for H, W, C in SHAPES:
        n = H * W * C
        m = int(np.prod(ga._bshape(bc, H, W, C)))
        x1, x2, x3 = fp8_values(n), fp8_values(m), fp8_values(m)
        for c0, c4 in ((COEFFS[0], COEFFS[1]), (COEFFS[2], COEFFS[0])):
            for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.0, 24.0),
                                (3, 0.1015625, 0.0)):
                got = go("npu_elementwise_broadcasting3d_mac_fused_fp8", x1, x2, x3, OUT,
                         *shapes(H, W, C, bc), c0, c4, act, a0, a1, dt=np.uint8, n=n)
                same(got, ga.elementwise_broadcasting3d_mac_fused_fp8(
                    x1, x2, x3, *tshapes(H, W, C, bc), c0, c4, act, a0, a1), f"mac act{act}")


@pytest.mark.parametrize("t", list(HALF))
def test_unary_half_exhaustive(t):
    nt, ht = HALF[t]
    for alpha in (0.01, 0.1015625, 0.3, 0.1 / 0.6):
        got = go("npu_elementwise_leakyrelu_bf16", ALL16, nt, OUT, I64(65536), alpha,
                 dt=np.uint16, n=65536)
        same(got, ga.elementwise_leakyrelu_bf16(ALL16, ht, 65536, alpha), f"leakyrelu {alpha}")
    got = go("npu_elementwise_square_bf16", ALL16, nt, OUT, I64(65536), dt=np.uint16,
             n=65536)
    same(got, ga.elementwise_square_bf16(ALL16, ht, 65536), "square")
    for C in (64, 96):
        D = (RNG.standard_normal(C) * 0.7 - 1.0).astype(F32)
        D[:4] = (F32(0.0), F32(-1.0), F32(0.25 - 1), F32(0.1 - 1))
        got = go("npu_elementwise_prelu_bf16", ALL16, nt, OUT, I64(65536), I64(C), D,
                 dt=np.uint16, n=65536)
        same(got, ga.elementwise_prelu_bf16(ALL16, ht, 65536, C, D), f"prelu C{C}")


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
                 dt=np.uint8, n=n)
        same(got, ga.elementwise_leakyrelu_fp8(x, n, pos, neg), "leakyrelu")
    for s in COEFFS + (1e-3, 64.0):
        got = go("npu_elementwise_square_fp8", x, OUT, I64(n), s, dt=np.uint8, n=n)
        same(got, ga.elementwise_square_fp8(x, n, s), f"square {s}")
    for P in (COEFFS[0], 2.0):
        C = 64
        D = ((RNG.standard_normal(C) * 0.7 - 1.0) * P).astype(F32)
        D[0] = F32(0.0)
        D[1] = F32(-P)
        got = go("npu_elementwise_prelu_fp8", x, OUT, I64(n), I64(C), float(P), D,
                 dt=np.uint8, n=n)
        same(got, ga.elementwise_prelu_fp8(x, n, C, P, D), f"prelu {P}")


@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("act", [0, 1, 3])
def test_batchnorm(t, act):
    C = 64
    if t == "fp8":
        x = np.concatenate([fp8_all(True), fp8_values(4096)])
        nt, wt = NPU_FP8, ga.F16
    else:
        nt, wt = HALF[t]
        x = np.concatenate([ALL16, half_values(RNG, wt, 4096, half_special(wt))])
    n = x.size - x.size % C
    x = np.ascontiguousarray(x[:n])
    cw = store_float((RNG.standard_normal(C) * 0.6).astype(F32), wt)
    cb = store_float((RNG.standard_normal(C) * 3).astype(F32), wt)
    cw[:3] = store_float(np.array([0.0, -0.0, 1.0], F32), wt)
    cb[:3] = store_float(np.array([-0.0, 0.0, -0.0], F32), wt)
    cw, cb = np.ascontiguousarray(cw, np.uint16), np.ascontiguousarray(cb, np.uint16)
    for alpha in (0.1015625, 0.1 / 0.6):
        dt = np.uint8 if t == "fp8" else np.uint16
        got = go("npu_batchnorm2d_stream_bf16", x, nt, OUT, I64(n), I64(C), cw, cb, act,
                 float(F32(alpha)), dt=dt, n=n)
        same(got, ga.batchnorm2d_stream_bf16(x, t if t == "fp8" else wt, n, C, cw, cb, act,
                                             F32(alpha)), f"bn {t} act{act}")


# One device buffer for every pointer of the invalid calls: they must fail before any launch.
P = np.zeros(4096, np.uint8)
S8, S16 = shp((1, 1, 8)), shp((1, 1, 16))

# Each entry of arith_float.h with argument sets it must reject (-1).
BAD_ARGS = [
    ("npu_elementwise_broadcasting3d_add_bf16", P, NPU_FP8, P, P, S8, S8, 0, 0, 0, 0.0, 0.0),
    ("npu_elementwise_broadcasting3d_add_bf16", None, NPU_BF16, P, P, S8, S8, 0, 0, 0, 0.0, 0.0),
    ("npu_elementwise_broadcasting3d_add_bf16", P, NPU_BF16, P, P, S8, S8, 0, 0, 4, 0.0, 0.0),
    ("npu_elementwise_broadcasting3d_mul_bf16", P, NPU_BF16, P, P, S8, shp((1, 1, 2)), 0, 0.0,
     0.0),   # not broadcastable
    ("npu_elementwise_broadcasting3d_mul_bf16", P, NPU_BF16, P, P, shp((1, -1, 8)), S8, 0, 0.0,
     0.0),
    ("npu_elementwise_broadcasting3d_div_bf16", P, NPU_I16, P, P, S8, S8, 0),
    ("npu_elementwise_broadcasting3d_div_bf16", P, NPU_F16, P, None, S8, S8, 0),
    ("npu_elementwise_broadcasting3d_mac_fused_bf16", P, NPU_BF16, P, None, P, S8, S8, 0, 0.0,
     0.0),
    ("npu_elementwise_broadcasting3d_mac_fused_bf16", P, NPU_BF16, P, P, P, S8, S8, -1, 0.0, 0.0),
    ("npu_elementwise_broadcasting3d_add_bfp16", P, P, P, shp((1, 1, 12)), shp((1, 1, 12))),
    ("npu_elementwise_broadcasting3d_add_bfp16", P, P, P, S8, None),
    ("npu_elementwise_broadcasting3d_sub_bfp16", P, P, None, S8, S8, 0),
    ("npu_elementwise_broadcasting3d_mul_bfp16", P, None, P, S8, S8),
    ("npu_elementwise_broadcasting3d_div_bfp16", P, P, P, S16, shp((1, 1, 8)), 0),
    ("npu_elementwise_broadcasting3d_mul_add_bfp16", P, P, NPU_BF16, P, NPU_BF16, P, NPU_BF16,
     S8, shp((1, 1, 1)), S8, 0, 0, 0, 0),   # C broadcast
    ("npu_elementwise_broadcasting3d_mul_add_bfp16", P, P, NPU_BF16, P, NPU_BF16, P, NPU_BF16,
     S8, S8, S8, 5, 0, 0, 0),
    ("npu_elementwise_broadcasting3d_mul_add_bfp16", P, P, NPU_BF16, P, NPU_BF16, P, NPU_BF16,
     S8, S8, S8, 7, 100, 0, 0),   # activation must be 0 or activation_type
    ("npu_elementwise_broadcasting3d_mul_add_bfp16", P, P, 1, P, NPU_BF16, P, NPU_BF16,
     S8, S8, S8, 0, 0, 0, 0),
    ("npu_elementwise_broadcasting3d_mul_add_bfp16", P, P, NPU_BF16, P, NPU_BFP16, P, NPU_F16,
     S8, S8, S8, 0, 0, 0, 0),
    ("npu_elementwise_broadcasting3d_add_fp8", P, P, P, S8, shp((1, 1, 1)), 1.0, 1.0, 3, 0.1,
     0.0, None),   # the table is required
    ("npu_elementwise_broadcasting3d_add_fp8", P, P, P, S8, S8, 1.0, 1.0, 4, 0.0, 0.0, None),
    ("npu_elementwise_broadcasting3d_mul_fp8", None, P, P, S8, S8, 1.0),
    ("npu_elementwise_broadcasting3d_div_fp8", P, P, P, S8, shp((2, 1, 8)), 1.0, 0),
    ("npu_elementwise_broadcasting3d_mac_fused_fp8", P, P, P, P, S8, S8, 1.0, 1.0, 9, 0.0, 0.0),
    ("npu_elementwise_broadcasting3d_mac_fused_fp8", P, P, None, P, S8, S8, 1.0, 1.0, 0, 0.0,
     0.0),
    ("npu_elementwise_leakyrelu_bf16", P, NPU_FP8, P, I64(8), 0.1),
    ("npu_elementwise_leakyrelu_bf16", P, NPU_BF16, P, I64(-1), 0.1),
    ("npu_elementwise_leakyrelu_fp8", P, None, I64(8), 1.0, 0.1),
    ("npu_elementwise_prelu_bf16", P, NPU_BF16, P, I64(8), I64(0), P),
    ("npu_elementwise_prelu_bf16", P, NPU_BF16, P, I64(0), I64(0), P),   # C >= 1 even if empty
    ("npu_elementwise_prelu_bf16", P, NPU_BF16, P, I64(8), I64(8), None),
    ("npu_elementwise_prelu_fp8", P, P, I64(8), I64(8), 1.0, None),
    ("npu_elementwise_prelu_fp8", P, P, I64(8), I64(-8), 1.0, P),
    ("npu_elementwise_square_bf16", P, NPU_U16, P, I64(8)),
    ("npu_elementwise_square_fp8", None, P, I64(8), 1.0),
    ("npu_batchnorm2d_stream_bf16", P, NPU_BF16, P, I64(8), I64(8), P, P, 2, 0.0),
    ("npu_batchnorm2d_stream_bf16", P, NPU_BF16, P, I64(8), I64(8), None, P, 0, 0.0),
    ("npu_batchnorm2d_stream_bf16", P, NPU_BF16, P, I64(8), I64(8), P, None, 0, 0.0),
    ("npu_batchnorm2d_stream_bf16", P, NPU_I8, P, I64(8), I64(8), P, P, 0, 0.0),
]


def test_bad_args():
    for args in BAD_ARGS:
        assert npu.raw_rc(*args) == -1, args
    # Empty tensors are a no-op, but the pointers are still required.
    assert npu.raw_rc("npu_elementwise_square_fp8", P, P, I64(0), 1.0) == 0
    assert npu.raw_rc("npu_elementwise_square_fp8", P, None, I64(0), 1.0) == -1
    assert npu.raw_rc("npu_elementwise_prelu_fp8", P, P, I64(0), I64(8), 1.0, P) == 0
    assert npu.raw_rc("npu_elementwise_prelu_fp8", P, P, I64(0), I64(8), 1.0, None) == -1
    assert npu.raw_rc("npu_elementwise_broadcasting3d_div_fp8", P, P, P, shp((0, 1, 8)),
                      shp((1, 1, 8)), 1.0, 0) == 0
    assert npu.raw_rc("npu_elementwise_broadcasting3d_div_fp8", P, None, P, shp((0, 1, 8)),
                      shp((1, 1, 8)), 1.0, 0) == -1
    assert npu.raw_rc("npu_elementwise_broadcasting3d_add_bf16", P, NPU_BF16, P, P,
                      shp((0, 3, 8)), shp((1, 3, 8)), 0, 0, 0, 0.0, 0.0) == 0
