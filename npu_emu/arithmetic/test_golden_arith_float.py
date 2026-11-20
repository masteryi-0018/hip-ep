# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Floating-point arithmetic entry points against the numpy reference, bit for bit."""

import numpy as np
import pytest

import golden_arith_float as ga
from golden_block_fp import decode_bfp16, encode_bfp16
from golden_float_cvt import bf16_to_f32, fp8_to_f32, load_float, store_float
from golden_int_round import Rnd
import npu
from npu import I64, OUT, NPU_BF16, NPU_FP8
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


def shapes(H, W, C, *bcs):
    """The host shape arrays of x1 (H x W x C) and of operands broadcast along bcs."""
    return (shp((H, W, C)),) + tuple(shp(ga.bshape(bc, H, W, C)) for bc in bcs)


@pytest.mark.parametrize("t", list(HALF))
@pytest.mark.parametrize("bc", BCASTS)
def test_add_mul_div_half(t, bc):
    nt, ht = HALF[t]
    for H, W, C in SHAPES:
        x1 = half_values(RNG, ht, H * W * C, half_special(ht))
        x2 = half_values(RNG, ht, int(np.prod(ga.bshape(bc, H, W, C))), half_special(ht))
        dims = shapes(H, W, C, bc)
        n = H * W * C
        for n1, n2 in ((0, 0), (0, 1), (1, 0)):
            for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.0, 6.0), (3, 0.1015625, 0.0),
                                (3, 0.3, 0.0)):
                got = go("npu_elementwise_broadcasting3d_add_bf16", x1, x2, nt, OUT, *dims,
                         n1, n2, act, a0, a1, dt=np.uint16, n=n)
                same(got, ga.add16(x1, x2, bc, ht, H, W, C, n1, n2, act, a0, a1),
                     f"add {n1}{n2} act{act}")
        for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, -1.5, 6.0), (3, 0.2, 0.0)):
            got = go("npu_elementwise_broadcasting3d_mul_bf16", x1, x2, nt, OUT, *dims,
                     act, a0, a1, dt=np.uint16, n=n)
            same(got, ga.mul16(x1, x2, bc, ht, H, W, C, act, a0, a1), f"mul act{act}")
        for sw in (0, 1):
            got = go("npu_elementwise_broadcasting3d_div_bf16", x1, x2, nt, OUT, *dims, sw,
                     dt=np.uint16, n=n)
            same(got, ga.div16(x1, x2, bc, ht, H, W, C, sw), f"div swap{sw}")


@pytest.mark.parametrize("t", list(HALF))
def test_div_half_all_divisors(t):
    """Every 16-bit divisor."""
    nt, ht = HALF[t]
    for num in (1.0, 3.0, -0.0):
        x1 = np.full(65536, store_float(F32(num), ht), np.uint16)
        got = go("npu_elementwise_broadcasting3d_div_bf16", x1, ALL16, nt, OUT,
                 *shapes(1, 256, 256, 0), 0, dt=np.uint16, n=65536)
        same(got, ga.div16(x1, ALL16, 0, ht, 1, 256, 256, 0), f"div {num}")


def test_inv_bf16_table():
    """Reciprocal special classes: zero, subnormal, overflow, NaN."""
    y = bf16_to_f32(ALL16)
    r = ga.inv16(y, "bf16").view(np.uint32) >> 16
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
        sp = half_special(ht)
        x1, x2, x3 = (half_values(RNG, ht, k, sp) for k in (n, m, m))
        for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (2, 0.0, 6.0), (3, 0.1015625, 0.0)):
            got = go("npu_elementwise_broadcasting3d_mac_fused_bf16", x1, x2, x3, nt, OUT,
                     *shapes(H, W, C, bc), act, a0, a1, dt=np.uint16, n=n)
            same(got, ga.mac16(x1, x2, x3, bc, ht, H, W, C, act, a0, a1), f"mac act{act}")


@pytest.mark.parametrize("bc", BCASTS)
def test_bfp16_binary(bc):
    for H, W, C in SHAPES:
        nbl = H * W * C // 8
        x1 = bfp_values(H * W * C, zero_rate=0.05)
        s = ga.bshape(bc, H, W, C)
        x2 = bfp_values(s[0] * s[1] * (8 if bc & 1 else s[2]), zero_rate=0.0)
        dims = shapes(H, W, C, bc)
        for op in ("add", "sub", "mul", "div"):
            for sw in ((0, 1) if op in ("sub", "div") else (0,)):
                extra = (sw,) if op in ("sub", "div") else ()
                got = go(f"npu_elementwise_broadcasting3d_{op}_bfp16", x1, x2, OUT, *dims,
                         *extra, dt=np.uint8, n=nbl * 9)
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
            x2 = bfp_values(m2) if b2 else half_values(RNG, ga.BF16, m2)
            x3 = bfp_values(m3) if b3 else half_values(RNG, ga.BF16, m3)
            for ob in (0, 1):
                got = go("npu_elementwise_broadcasting3d_mul_add_bfp16", x1, x2, b2, x3, b3, OUT,
                         ob, *shapes(H, W, C, bc2, bc3), act_type, activation, lo, hi,
                         dt=np.uint8, n=n // 8 * 9 if ob else n * 2)
                ref = ga.mul_add(x1, x2, b2, bc2, x3, b3, bc3, ob, H, W, C, act_type,
                                 activation, lo, hi)
                same(got, np.asarray(ref).view(np.uint8), f"b{b2}{b3} out{ob}")


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
    s = ga.bshape(bc, H, W, C)
    b = decode_bfp16(x2).reshape(s[0], s[1], -1)[:, :, :1 if bc & 1 else None]
    b = np.broadcast_to(b, (H, W, C))
    if op == "add":
        v = ga.add(a, b)
    elif op == "sub":
        v = ga.add(b, -a) if sw else ga.add(a, -b)
    elif op == "mul":
        v = ga.mul(a, b)
    else:
        num, den = (b, a) if sw else (a, b)
        v = ga.mul(num, ga.inv16(den, "bf16"))
    return encode_bfp16(v.reshape(-1), Rnd.kHalfEven)


@pytest.mark.parametrize("span", [20, 30])
def test_bfp16_binary_wide_groups(span):
    """Exponents spread over more than 17 / 25 inside a 64-element group: the identity-multiply
    decode rounds the small blocks; add is one block MMAC onto x2."""
    changed = set()
    for bc in BCASTS:
        for H, W, C in ((2, 3, 128), (3, 4, 80), (1, 9, 64)):
            x1 = wide_blocks(H * W * C // 8, span, 140)
            s = ga.bshape(bc, H, W, C)
            x2 = wide_blocks(s[0] * s[1] * (1 if bc & 1 else s[2] // 8), span, 134)
            dims = shapes(H, W, C, bc)
            for op in ("add", "sub", "mul", "div"):
                for sw in ((0, 1) if op in ("sub", "div") else (0,)):
                    extra = (sw,) if op in ("sub", "div") else ()
                    got = go(f"npu_elementwise_broadcasting3d_{op}_bfp16", x1, x2, OUT, *dims,
                             *extra, dt=np.uint8, n=H * W * C // 8 * 9)
                    want = ga.bfp_bin(op, x1, x2, bc, H, W, C, sw)
                    same(got, want, f"{op} swap{sw} bc{bc} C{C}")
                    if (exact_bin(op, x1, x2, bc, H, W, C, sw) != want).any():
                        changed.add((op, sw, bc & 1))
            a, b = ga.bfp_values(x1, (H, W, C)), ga.bfp_operand(x2, bc, H, W, C)
            if (ga.bfp_bin("add", x1, x2, bc, H, W, C) !=
                    encode_bfp16(ga.add(a, b).reshape(-1), Rnd.kHalfEven)).any():
                changed.add(("mac", bc & 1))
    ops = [("add", 0), ("sub", 0), ("sub", 1), ("mul", 0), ("div", 0), ("div", 1)]
    assert changed == {(op, sw, c) for op, sw in ops for c in (0, 1)} | {("mac", 0), ("mac", 1)}


@pytest.mark.parametrize("span", [20, 30])
def test_mul_add_bfp16_wide_groups(span, monkeypatch):
    H, W, C = 3, 4, 128
    n = H * W * C
    x1 = wide_blocks(n // 8, span, 140)
    for bc2, bc3 in ((0, 0), (2, 4)):
        m2 = int(np.prod(ga.bshape(bc2, H, W, C)))
        m3 = int(np.prod(ga.bshape(bc3, H, W, C)))
        for b2, b3 in ((1, 1), (0, 1), (1, 0)):
            x2 = wide_blocks(m2 // 8, span, 134) if b2 else half_values(RNG, ga.BF16, m2)
            x3 = wide_blocks(m3 // 8, span, 134) if b3 else half_values(RNG, ga.BF16, m3)
            for ob in (0, 1):
                args = (x1, x2, b2, bc2, x3, b3, bc3, ob, H, W, C, 0, 0, 0.0, 0.0)
                got = go("npu_elementwise_broadcasting3d_mul_add_bfp16", x1, x2, b2, x3, b3, OUT,
                         ob, *shapes(H, W, C, bc2, bc3), *args[11:], dt=np.uint8,
                         n=n // 8 * 9 if ob else n * 2)
                want = np.asarray(ga.mul_add(*args))
                same(got, want.view(np.uint8), f"b{b2}{b3} out{ob}")
                with monkeypatch.context() as m:
                    m.setattr(ga, "decode_bfp16_identity", exact_decode)
                    assert (np.asarray(ga.mul_add(*args)) != want).any()


def test_sigtanh_all_bf16():
    """The tanh path on every bf16 value (through mul_add: x1 = 1, x2 = x, x3 = 0)."""
    H, W, C = 1, 256, 256
    x1 = encode_bfp16(np.ones(H * W * C, F32), Rnd.kHalfEven)
    x3 = np.zeros(H * W * C, np.uint16)
    got = go("npu_elementwise_broadcasting3d_mul_add_bfp16", x1, ALL16, 0, x3, 0, OUT, 0,
             *shapes(H, W, C, 0, 0), 7, 7, 0.0, 0.0, dt=np.uint16, n=65536)
    same(got, ga.mul_add(x1, ALL16, 0, 0, x3, 0, 0, 0, H, W, C, 7, 7, 0.0, 0.0), "sigtanh")
    # Finite inputs stay within 2^-7 of tanh.
    v = bf16_to_f32(got)
    x = bf16_to_f32(ALL16)
    ok = np.isfinite(x)
    assert np.max(np.abs(v[ok] - np.tanh(x[ok].astype(np.float64)))) < 2 ** -7


def table(c1, c2, act, alpha):
    """Coefficient table of add / sub fp8: [code] = fp32(decode(code) * c2 * A), then A * c1
    and B / A (A = (1 + alpha) / 2, B = |1 - alpha| / 2; A = 1 for CLIP)."""
    d = fp8_to_f32(np.arange(256)).astype(np.float64)
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
                got = go("npu_elementwise_broadcasting3d_add_fp8", x1, x2, OUT,
                         *shapes(H, W, C, bc), c1, c2, act, a0, a1, tb, dt=np.uint8, n=n)
                same(got, ga.add_fp8(x1, x2, bc, H, W, C, c1, c2, act, a0, a1, tb),
                     f"add c{c1},{c2} act{act}")


@pytest.mark.parametrize("bc", BCASTS)
def test_mul_div_fp8(bc):
    for H, W, C in SHAPES:
        n = H * W * C
        m = int(np.prod(ga.bshape(bc, H, W, C)))
        x1, x2 = fp8_values(n), fp8_values(m)
        for s in COEFFS:
            got = go("npu_elementwise_broadcasting3d_mul_fp8", x1, x2, OUT, *shapes(H, W, C, bc),
                     s, dt=np.uint8, n=n)
            same(got, ga.mul_fp8(x1, x2, bc, H, W, C, s), f"mul {s}")
            for sw in (0, 1):
                got = go("npu_elementwise_broadcasting3d_div_fp8", x1, x2, OUT,
                         *shapes(H, W, C, bc), s, sw, dt=np.uint8, n=n)
                same(got, ga.div_fp8(x1, x2, bc, H, W, C, s, sw), f"div {s} swap{sw}")


def test_div_fp8_reciprocal():
    """The hardware reciprocal of 1.75 is one ulp below RNE(1 / 1.75): the quotient falls
    below the fp8 tie 1.0625."""
    x1, x2 = np.full(8, 0x38, np.uint8), np.full(8, 0x3E, np.uint8)   # 1.0, 1.75
    got = go("npu_elementwise_broadcasting3d_div_fp8", x1, x2, OUT, *shapes(1, 1, 8, 0),
             1.859375, 0, dt=np.uint8, n=8)
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
                got = go("npu_elementwise_broadcasting3d_mac_fused_fp8", x1, x2, x3, OUT,
                         *shapes(H, W, C, bc), c0, c4, act, a0, a1, dt=np.uint8, n=n)
                same(got, ga.mac_fp8(x1, x2, x3, bc, H, W, C, c0, c4, act, a0, a1),
                     f"mac act{act}")


@pytest.mark.parametrize("t", list(HALF))
def test_unary_half_exhaustive(t):
    nt, ht = HALF[t]
    for alpha in (0.01, 0.1015625, 0.3, 0.1 / 0.6):
        got = go("npu_elementwise_leakyrelu_bf16", ALL16, nt, OUT, I64(65536), alpha,
                 dt=np.uint16, n=65536)
        same(got, ga.leakyrelu16(ALL16, ht, alpha), f"leakyrelu {alpha}")
    got = go("npu_elementwise_square_bf16", ALL16, nt, OUT, I64(65536), dt=np.uint16,
             n=65536)
    same(got, ga.square16(ALL16, ht), "square")
    for C in (64, 96):
        D = (RNG.standard_normal(C) * 0.7 - 1.0).astype(F32)
        D[:4] = (F32(0.0), F32(-1.0), F32(0.25 - 1), F32(0.1 - 1))
        got = go("npu_elementwise_prelu_bf16", ALL16, nt, OUT, I64(65536), D, I64(C),
                 dt=np.uint16, n=65536)
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
                 dt=np.uint8, n=n)
        same(got, ga.leakyrelu_fp8(x, pos, neg), "leakyrelu")
    for s in COEFFS + (1e-3, 64.0):
        got = go("npu_elementwise_square_fp8", x, OUT, I64(n), s, dt=np.uint8, n=n)
        same(got, ga.square_fp8(x, s), f"square {s}")
    for P in (COEFFS[0], 2.0):
        C = 64
        D = ((RNG.standard_normal(C) * 0.7 - 1.0) * P).astype(F32)
        D[0] = F32(0.0)
        D[1] = F32(-P)
        got = go("npu_elementwise_prelu_fp8", x, OUT, I64(n), float(P), D, I64(C),
                 dt=np.uint8, n=n)
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
        x = np.concatenate([ALL16, half_values(RNG, wt, 4096, half_special(wt))])
    n = x.size - x.size % C
    x = np.ascontiguousarray(x[:n])
    cw = load_float(store_float((RNG.standard_normal(C) * 0.6).astype(F32), wt), wt)
    cb = load_float(store_float((RNG.standard_normal(C) * 3).astype(F32), wt), wt)
    cw[:3] = (F32(0.0), F32(-0.0), F32(1.0))
    cb[:3] = (F32(-0.0), F32(0.0), F32(-0.0))
    cw, cb = np.ascontiguousarray(cw), np.ascontiguousarray(cb)
    for alpha in (0.1015625, 0.1 / 0.6):
        dt = np.uint8 if t == "fp8" else np.uint16
        got = go("npu_batchnorm2d_stream_bf16", x, nt, OUT, I64(n), I64(C), cw, cb, act,
                 float(F32(alpha)), dt=dt, n=n)
        same(got, ga.batchnorm(x, t if t == "fp8" else wt, C, cw, cb, act, F32(alpha)),
             f"bn {t} act{act}")


def test_bad_arguments():
    rc = npu.raw_rc
    nul = None
    one8 = shapes(1, 1, 8, 0)
    assert rc("npu_elementwise_broadcasting3d_add_bf16", nul, nul, NPU_FP8, nul, *one8, 0, 0, 0,
              0.0, 0.0) == -1
    assert rc("npu_elementwise_broadcasting3d_mul_bf16", nul, nul, NPU_BF16, nul, shp((1, 1, 8)),
              shp((1, 1, 2)), 0, 0.0, 0.0) == -1   # not broadcastable
    assert rc("npu_elementwise_broadcasting3d_add_bfp16", nul, nul, nul,
              *shapes(1, 1, 12, 0)) == -1
    assert rc("npu_elementwise_broadcasting3d_mul_add_bfp16", nul, nul, 0, nul, 0, nul, 0,
              *shapes(1, 1, 8, 1, 0), 0, 0, 0.0, 0.0) == -1
    assert rc("npu_elementwise_broadcasting3d_mul_add_bfp16", nul, nul, 0, nul, 0, nul, 0,
              *shapes(1, 1, 8, 0, 0), 5, 0, 0.0, 0.0) == -1
    assert rc("npu_elementwise_broadcasting3d_add_fp8", nul, nul, nul, *shapes(1, 1, 8, 1),
              1.0, 1.0, 3, 0.1, 0.0, nul) == -1
    assert rc("npu_elementwise_broadcasting3d_mac_fused_fp8", nul, nul, nul, nul, *one8, 1.0,
              1.0, 9, 0.0, 0.0) == -1
    assert rc("npu_elementwise_prelu_bf16", nul, NPU_BF16, nul, I64(8), nul, I64(0)) == -1
    assert rc("npu_batchnorm2d_stream_bf16", nul, NPU_BF16, nul, I64(8), I64(8), nul, nul, 2,
              0.0) == -1
    # Empty tensors are a no-op.
    assert rc("npu_elementwise_square_fp8", nul, nul, I64(0), 1.0) == 0
