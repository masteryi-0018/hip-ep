# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""reduction/avgpool.h entry points against golden_avgpool.py, bit for bit."""

import numpy as np
import pytest

import golden_avgpool as ga
from golden_float_cvt import bf16_to_f32, f32_to_bf16, f32_to_f16, f32_to_fp8, fp8_to_f32
from golden_int_round import float_to_int
from npu import I64, NPU_BF16, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT, U32, call, raw_rc
from npu import f32 as f
from testing import F16T as HALF
from testing import INT, fp8_values, rand_ints, same

RNG = np.random.default_rng(2041)
F32 = np.float32

# (H, W, C, kh, kw, sh, sw)
WINDOWS = [(9, 11, 64, 3, 3, 1, 2), (8, 8, 32, 2, 2, 2, 2), (15, 15, 16, 13, 13, 1, 1),
           (7, 9, 24, 3, 1, 2, 1), (6, 6, 16, 1, 2, 1, 1), (10, 12, 12, 5, 4, 3, 2)]
# (H, W, C, sv_h, sv_w)
GLOBALS = [(8, 12, 64, 4, 3), (5, 5, 32, 5, 1), (16, 16, 24, 8, 8), (6, 4, 16, 1, 4)]


def half_values(t, shape, special=True, cancel=True):
    """Random 16-bit floats; `cancel` adds large +-2^k values, `special` adds boundary values."""
    n = int(np.prod(shape))
    wide = (RNG.standard_normal(n) * np.exp2(RNG.integers(-12, 12, n))).astype(F32)
    grid = (RNG.integers(-24, 25, n) * F32(0.125)).astype(F32)
    v = np.where(RNG.random(n) < 0.5, wide, grid)
    if cancel:
        big = RNG.random(n) < 0.06
        v = np.where(big, RNG.choice([-1, 1], n) * np.exp2(RNG.integers(10, 15, n)), v)
    bits = (f32_to_f16(v) if t == "f16" else f32_to_bf16(v)).astype(np.uint16)
    if special:
        f16 = t == "f16"
        sp = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x0040 if not f16 else 0x0200,
                       0x7C00 if f16 else 0x7F80, 0xFC00 if f16 else 0xFF80,
                       0x7E00 if f16 else 0x7FC0, 0x7BFF if f16 else 0x7F7F,
                       0x0400 if f16 else 0x0080, 0x03FF if f16 else 0x007F], np.uint16)
        pos = RNG.choice(n, max(1, n // 40), replace=False)
        bits[pos] = sp[RNG.integers(0, len(sp), pos.size)]
    return bits.reshape(shape)


def s3(H, W, C):
    return I64(H), I64(W), I64(C)


def out_shape(w):
    H, W, C, kh, kw, sh, sw = w
    return ga.out_hw(H, W, kh, kw, sh, sw) + (C,)


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("w", WINDOWS)
def test_avgpool_bf16(t, w):
    H, W, C, kh, kw, sh, sw = w
    x = half_values(t, (H, W, C))
    for factor in (1.0 / (kh * kw), f(RNG.uniform(0.01, 3.0))):
        got = call("npu_avgpool_bf16", x, HALF[t], OUT, I64(H), I64(W), I64(C), kh, kw, sh, sw, f(factor),
                   out_nbytes=2 * int(np.prod(out_shape(w))), out_dtype=np.uint16)
        same(got, ga.avgpool_16(x, t, kh, kw, sh, sw, factor))


@pytest.mark.parametrize("w", WINDOWS)
def test_avgpool_bf16_c32(w):
    H, W, C, kh, kw, sh, sw = w
    x = half_values("bf16", (H, W, C))
    for factor in (1.0 / (kh * kw), f(RNG.uniform(0.01, 3.0))):
        got = call("npu_avgpool_bf16_c32", x, OUT, I64(H), I64(W), I64(C), kh, kw, sh, sw, f(factor),
                   out_nbytes=2 * int(np.prod(out_shape(w))), out_dtype=np.uint16)
        same(got, ga.avgpool_16(x, "bf16", kh, kw, sh, sw, factor, c32=True))


def test_avgpool_bf16_order_matters():
    """Cancelling taps: the two-chain order differs from one fp32 chain."""
    H, W, C, kh, kw = 3, 3, 8, 3, 3
    v = np.zeros((H, W, C), F32)
    v[0, 0], v[0, 1], v[0, 2] = 2.0 ** 24, 1.0, -(2.0 ** 24)   # chain 0 / 1 / 0
    v[1, 0] = 2.0 ** -4
    x = f32_to_bf16(v)
    got = call("npu_avgpool_bf16", x, NPU_BF16, OUT, I64(H), I64(W), I64(C), kh, kw, 1, 1, 1.0,
               out_nbytes=2 * C, out_dtype=np.uint16)
    want = ga.avgpool_16(x, "bf16", kh, kw, 1, 1, 1.0)
    same(got, want)
    one_chain = f32_to_bf16(ga.fsum(ga.taps(bf16_to_f32(x), kh, kw, 1, 1)))
    assert not np.array_equal(want, one_chain)


def tiny_values(t, shape):
    """Smallest 16-bit float values (bf16: around the fp32 normal limit), both signs."""
    n = int(np.prod(shape))
    if t == "f16":
        bits = RNG.integers(0, 0x0800, n).astype(np.uint16)
    else:
        bits = RNG.integers(0, 0x0200, n).astype(np.uint16)
    bits |= RNG.choice(np.array([0, 0x8000], np.uint16), n)
    return bits.reshape(shape)


def test_avgpool_bf16_subnormal_example():
    """Taps 0x8139, 0x80c5 with factor 0.5 give 0x80b9 (0x80c5 * 0.5 becomes -0)."""
    x = np.ascontiguousarray(np.stack([np.full(8, 0x8139), np.full(8, 0x80C5)])
                             .astype(np.uint16).reshape(1, 2, 8))
    for name, args in (("npu_avgpool_bf16", (NPU_BF16,)), ("npu_avgpool_bf16_c32", ())):
        got = call(name, x, *args, OUT, *s3(1, 2, 8), 1, 2, 1, 2, 0.5, out_nbytes=16,
                   out_dtype=np.uint16)
        assert (got == 0x80B9).all(), got


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_avgpool_bf16_subnormal(t):
    """Subnormal inputs, products, chain steps and chain sums (signed zeros)."""
    H, W, C = 6, 8, 64
    x = tiny_values(t, (H, W, C))
    for kh, kw, factor in ((1, 2, 0.5), (2, 2, 0.25), (3, 3, 1.0 / 9), (2, 3, 2.0 ** -20)):
        Ho, Wo = ga.out_hw(H, W, kh, kw, 1, 1)
        got = call("npu_avgpool_bf16", x, HALF[t], OUT, *s3(H, W, C), kh, kw, 1, 1, f(factor),
                   out_nbytes=2 * Ho * Wo * C, out_dtype=np.uint16)
        want = ga.avgpool_16(x, t, kh, kw, 1, 1, factor)
        same(got, want)
        if t == "bf16":
            c32 = call("npu_avgpool_bf16_c32", x, OUT, *s3(H, W, C), kh, kw, 1, 1, f(factor),
                       out_nbytes=2 * want.size, out_dtype=np.uint16)
            same(c32, ga.avgpool_16(x, t, kh, kw, 1, 1, factor, c32=True))
    assert (want == 0x8000).any() or t == "f16"


@pytest.mark.parametrize("entry", ["bf16", "f16", "c32"])
def test_globalavgpool_bf16_subnormal(entry):
    """Subnormal running sums and final products (-> signed zero)."""
    H, W, C, sv_h, sv_w = 4, 6, 64, 2, 3
    t = "f16" if entry == "f16" else "bf16"
    x = tiny_values(t, (H, W, C))
    for factor in (f(1.0 / (H * W)), 0.5, f(2.0 ** -20)):
        if entry == "c32":
            got = call("npu_globalavgpool_bf16_c32", x, OUT, *s3(H, W, C), sv_h, sv_w, factor,
                       out_nbytes=2 * C, out_dtype=np.uint16)
        else:
            got = call("npu_globalavgpool_bf16", x, HALF[t], OUT, *s3(H, W, C), sv_h, sv_w,
                       factor, out_nbytes=2 * C, out_dtype=np.uint16)
        same(got, ga.globalavgpool_16(x, t, sv_h, sv_w, factor))


@pytest.mark.parametrize("entry", ["bf16", "f16", "c32"])
@pytest.mark.parametrize("gl", GLOBALS)
def test_globalavgpool_bf16(entry, gl):
    H, W, C, sv_h, sv_w = gl
    t = "f16" if entry == "f16" else "bf16"
    x = half_values(t, (H, W, C))
    for factor in (f(1.0 / (H * W)), f(RNG.uniform(0.001, 2.0))):
        if entry == "c32":
            got = call("npu_globalavgpool_bf16_c32", x, OUT, I64(H), I64(W), I64(C), sv_h, sv_w, factor,
                       out_nbytes=2 * C, out_dtype=np.uint16)
        else:
            got = call("npu_globalavgpool_bf16", x, HALF[t], OUT, I64(H), I64(W), I64(C), sv_h, sv_w, factor,
                       out_nbytes=2 * C, out_dtype=np.uint16)
        same(got, ga.globalavgpool_16(x, t, sv_h, sv_w, factor))


def tile_sensitive(H, W, C, sv_w):
    """Input whose block-order sum loses a 1 that the row-major sum keeps."""
    v = np.zeros((H, W, C), F32)
    v[0, 0], v[1, 0], v[0, sv_w] = 2.0 ** 24, 1.0, -(2.0 ** 24)
    return v


def test_globalavgpool_tiling_matters():
    H, W, C, sv_h, sv_w = 4, 4, 8, 2, 2
    x = f32_to_bf16(tile_sensitive(H, W, C, sv_w))
    res = [call("npu_globalavgpool_bf16", x, NPU_BF16, OUT, I64(H), I64(W), I64(C), a, b, 1.0,
                out_nbytes=2 * C, out_dtype=np.uint16) for a, b in ((sv_h, sv_w), (H, W))]
    same(res[0], ga.globalavgpool_16(x, "bf16", sv_h, sv_w, 1.0))
    same(res[1], ga.globalavgpool_16(x, "bf16", H, W, 1.0))
    assert (bf16_to_f32(res[0]) == 0).all() and (bf16_to_f32(res[1]) == 1).all()


def fp8_coeffs(C, channelwise):
    if channelwise:
        c2 = (RNG.uniform(0.002, 0.3, C) * RNG.choice([-1, 1], C)).astype(F32)
        c0 = RNG.uniform(-20, 20, C).astype(F32)
        c0[: C // 4] = 0
        return c0, c2
    return np.full(C, f(-1.75), F32), np.full(C, f(1.0 / 9 * 0.37), F32)


@pytest.mark.parametrize("channelwise", [False, True])
@pytest.mark.parametrize("w", WINDOWS)
def test_avgpool_fp8(w, channelwise):
    H, W, C, kh, kw, sh, sw = w
    x = fp8_values(RNG, (H, W, C), 0.002)
    c0, c2 = fp8_coeffs(C, channelwise)
    n = int(np.prod(out_shape(w)))
    for kh_sv in sorted({1, kh} | ({kh // 3} if kh % 3 == 0 and kh > 3 else set())):
        if kh % kh_sv:
            continue
        for act, lo, hi in ((0, 0.0, 0.0), (1, -1.5, f(3.4028235e38)), (1, 0.0, f(6 / 0.07))):
            got = call("npu_avgpool_fp8", x, OUT, I64(H), I64(W), I64(C), kh, kw, sh, sw, kh_sv, c0, c2,
                       act, f(lo), f(hi), out_nbytes=n, out_dtype=np.uint8)
            same(got, ga.avgpool_fp8(x, kh, kw, sh, sw, kh_sv, c0, c2, act, lo, hi))


def test_avgpool_fp8_psum_matters():
    """Two passes of 80 rows keep 2 * 2^-9 that a single row-by-row sum loses."""
    H, W, C, kh, kw = 160, 1, 8, 160, 1
    v = np.zeros((H, W, C), F32)
    v[:80], v[80], v[81] = 448.0, 2.0 ** -9, 2.0 ** -9
    x = f32_to_fp8(v)
    c0, c2 = np.full(C, -35840.0, F32), np.ones(C, F32)
    res = [call("npu_avgpool_fp8", x, OUT, *s3(H, W, C), kh, kw, 1, 1, kh_sv, c0, c2, 0, 0.0,
                0.0, out_nbytes=C, out_dtype=np.uint8) for kh_sv in (kh, 80)]
    for r, kh_sv in zip(res, (kh, 80)):
        same(r, ga.avgpool_fp8(x, kh, kw, 1, 1, kh_sv, c0, c2))
    assert (res[0] == 0).all() and (fp8_to_f32(res[1]) == 2.0 ** -8).all()


def mmac_tie_column(n_big):
    """2^-8, n_big x 448, 2^-9: the last tap is half an ulp of an odd running sum, which
    the MMAC accumulation drops and an fp32 add rounds up."""
    v = np.zeros(n_big + 2, F32)
    v[0], v[1:n_big + 1], v[-1] = 2.0 ** -8, 448.0, 2.0 ** -9
    return v


def test_avgpool_fp8_mmac_sum():
    H, W, C, n = 82, 1, 8, 80
    x = f32_to_fp8(np.repeat(mmac_tie_column(n)[:, None, None], C, axis=2))
    c0, c2 = np.full(C, -448.0 * n, F32), np.ones(C, F32)
    got = call("npu_avgpool_fp8", x, OUT, *s3(H, W, C), H, 1, 1, 1, H, c0, c2, 0, 0.0, 0.0,
               out_nbytes=C, out_dtype=np.uint8)
    same(got, ga.avgpool_fp8(x, H, 1, 1, 1, H, c0, c2))
    assert (fp8_to_f32(got) == 2.0 ** -8).all()


def test_globalavgpool_fp8_mmac_sum():
    H, W, C, n = 82, 1, 8, 80
    x = f32_to_fp8(np.repeat(mmac_tie_column(n)[:, None, None], C, axis=2))
    got = call("npu_globalavgpool_fp8", x, OUT, *s3(H, W, C), 41, 1, f(-448.0 * n), f(1.0),
               out_nbytes=C, out_dtype=np.uint8)
    same(got, ga.globalavgpool_fp8(x, 41, 1, f(-448.0 * n), f(1.0)))
    assert (fp8_to_f32(got) == 2.0 ** -8).all()


def test_avgpool_fp8_specials():
    """NaN groups of 8 channels, overflow to NaN, half-to-even ties, -0."""
    H, W, C = 2, 2, 16
    x = np.full((H, W, C), 0x38, np.uint8)       # 1.0
    x[0, 0, 3] = 0x7F                            # NaN: channels 0..7 of out[0][0]
    x[1, 1, 12] = 0xFF
    c2 = np.array([1.0, 500.0, 1.0625, 2.0 ** -12] * 4, F32)  # 500 -> NaN, 1.0625 a tie
    c0 = np.array([0.0, 0.0, 0.0, -(2.0 ** -10) - 2.0 ** -12] * 4, F32)
    got = call("npu_avgpool_fp8", x, OUT, I64(H), I64(W), I64(C), 1, 1, 1, 1, 1, c0, c2, 0, 0.0, 0.0,
               out_nbytes=H * W * C, out_dtype=np.uint8)
    same(got, ga.avgpool_fp8(x, 1, 1, 1, 1, 1, c0, c2))
    got = call("npu_avgpool_fp8", x, OUT, I64(H), I64(W), I64(C), 2, 2, 1, 1, 1, c0, c2, 0, 0.0, 0.0,
               out_nbytes=C, out_dtype=np.uint8)
    same(got, ga.avgpool_fp8(x, 2, 2, 1, 1, 1, c0, c2))


@pytest.mark.parametrize("gl", GLOBALS)
def test_globalavgpool_fp8(gl):
    H, W, C, sv_h, sv_w = gl
    x = fp8_values(RNG, (H, W, C), 0.002)
    for c0, c1 in ((0.0, f(1.0 / (H * W))), (f(3.25), f(0.37 / (H * W))), (f(-7.5), f(5.0))):
        got = call("npu_globalavgpool_fp8", x, OUT, I64(H), I64(W), I64(C), sv_h, sv_w, c0, c1,
                   out_nbytes=C, out_dtype=np.uint8)
        same(got, ga.globalavgpool_fp8(x, sv_h, sv_w, c0, c1))


# (c0, c1, shift)
def qdq_coeffs(taps):
    return [(0.0, f(0.5), 0), (f(0.5), f(0.5 / taps), 1), (f(-3.7), f(1.0 / taps), -2),
            (f(12.25), f(0.0123 / taps * 300), 2), (f(100.0), f(-0.37 / taps), 0),
            (f(-17.0), f(37.0 / taps), -1)]


def pairs(bits):
    s = ["i8", "u8"] if bits == 8 else ["i16", "u16"]
    return [(a, b) for a in s for b in s]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("w", WINDOWS)
def test_qdq_avgpool(bits, w):
    H, W, C, kh, kw, sh, sw = w
    n = int(np.prod(out_shape(w)))
    for it, ot in pairs(bits):
        x = rand_ints(RNG, INT[it][1], (H, W, C))
        for c0, c1, shift in qdq_coeffs(kh * kw):
            if bits == 8:
                for act, lo, hi in ((0, 0.0, 0.0), (1, f(-3.5), f(40.25))):
                    got = call("npu_qdq_avgpool_a8", x, INT[it][0], OUT, INT[ot][0], I64(H), I64(W), I64(C),
                               kh, kw, sh, sw, c0, c1, shift, act, lo, hi,
                               out_nbytes=n, out_dtype=INT[ot][1])
                    same(got, ga.qdq_avgpool(x, ot, kh, kw, sh, sw, c0, c1, shift, act, lo, hi))
            else:
                got = call("npu_qdq_avgpool_a16", x, INT[it][0], OUT, INT[ot][0], I64(H), I64(W), I64(C),
                           kh, kw, sh, sw, c0, c1, shift, out_nbytes=2 * n, out_dtype=INT[ot][1])
                same(got, ga.qdq_avgpool(x, ot, kh, kw, sh, sw, c0, c1, shift))


def test_qdq_avgpool_a16_u16_unsigned():
    """U16 inputs are summed as unsigned: 40000 stays 40000."""
    x = np.full((2, 2, 8), 40000, np.uint16)
    got = call("npu_qdq_avgpool_a16", x, NPU_U16, OUT, NPU_U16, *s3(2, 2, 8), 2, 2, 1, 1, 0.0, 0.25, 0,
               out_nbytes=16, out_dtype=np.uint16)
    assert (got == 40000).all()


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("gl", GLOBALS)
def test_qdq_globalavgpool(bits, gl):
    H, W, C = gl[:3]
    for it, ot in pairs(bits):
        x = rand_ints(RNG, INT[it][1], (H, W, C))
        for c0, c1, shift in qdq_coeffs(H * W):
            name = f"npu_qdq_globalavgpool_a{bits}"
            got = call(name, x, INT[it][0], OUT, INT[ot][0], I64(H), I64(W), I64(C), c0, c1, shift,
                       out_nbytes=C * bits // 8, out_dtype=INT[ot][1])
            same(got, ga.qdq_global(x, ot, c0, c1, shift, as_i16=bits == 16))


def test_qdq_mac_single_rounding():
    """sum * c1 + c0 is rounded once (a two-rounding evaluation gives another result)."""
    x = np.full((1, 3, 8), 100, np.int8)
    c1, c0 = f(1 + 2.0 ** -23), f(-(300 + 2.0 ** -15) + 0.5)
    got = call("npu_qdq_globalavgpool_a8", x, NPU_I8, OUT, NPU_I8, *s3(1, 3, 8), c0, c1, 0,
               out_nbytes=8, out_dtype=np.int8)
    want = ga.qdq_global(x, "i8", c0, c1, 0)
    same(got, want)
    two = float_to_int(F32(F32(300 * np.float64(c1)) + c0), "i8")
    assert two != want[0]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("w", WINDOWS)
def test_avgpool_xint8(w, fused):
    H, W, C, kh, kw, sh, sw = w
    n = int(np.prod(out_shape(w)))
    for t in ("i8", "u8"):
        x = rand_ints(RNG, INT[t][1], (H, W, C))
        weight = int(round(2 ** 15 / (kh * kw)))
        for wt, shift in ((weight, 15), (weight, 14), (weight, 16), (1 << 14, 15), (3, 0),
                          (65535, 20)):
            if fused:
                for act in (0, 1):
                    got = call("npu_avgpool_xint8_fused", x, INT[t][0], OUT, INT[t][0], I64(H), I64(W), I64(C),
                               kh, kw, sh, sw, U32(wt), shift, act,
                               out_nbytes=n, out_dtype=INT[t][1])
                    same(got, ga.avgpool_xint8(x, t, kh, kw, sh, sw, wt, shift, act == 1))
            else:
                got = call("npu_avgpool_xint8", x, INT[t][0], OUT, INT[t][0], I64(H), I64(W), I64(C),
                           kh, kw, sh, sw, U32(wt), shift, out_nbytes=n, out_dtype=INT[t][1])
                same(got, ga.avgpool_xint8(x, t, kh, kw, sh, sw, wt, shift))


def test_bad_args():
    z = None
    assert raw_rc("npu_avgpool_bf16", z, NPU_I8, z, *s3(4, 4, 8), 2, 2, 1, 1, 0.25) == -1
    assert raw_rc("npu_avgpool_bf16", z, NPU_BF16, z, *s3(4, 4, 8), 1, 1, 1, 1, 1.0) == -1
    assert raw_rc("npu_avgpool_bf16", z, NPU_BF16, z, *s3(4, 4, 8), 5, 2, 1, 1, 0.1) == -1
    assert raw_rc("npu_avgpool_bf16_c32", z, z, *s3(4, 4, 8), 2, 2, 0, 1, 0.25) == -1
    assert raw_rc("npu_globalavgpool_bf16", z, NPU_BF16, z, *s3(6, 4, 8), 4, 4, 0.1) == -1
    assert raw_rc("npu_globalavgpool_fp8", z, z, *s3(6, 4, 8), 3, 3, 0.0, 0.1) == -1
    assert raw_rc("npu_avgpool_fp8", z, z, *s3(6, 6, 8), 3, 3, 1, 1, 2, z, z, 0, 0.0, 0.0) == -1
    assert raw_rc("npu_qdq_avgpool_a8", z, NPU_I16, z, NPU_I8, *s3(4, 4, 8), 2, 2, 1, 1, 0.0, 1.0, 0,
                  0, 0.0, 0.0) == -1
    assert raw_rc("npu_qdq_avgpool_a16", z, NPU_I16, z, NPU_U8, *s3(4, 4, 8), 2, 2, 1, 1, 0.0, 1.0,
                  0) == -1
    assert raw_rc("npu_qdq_globalavgpool_a8", z, NPU_I8, z, NPU_I16, *s3(4, 4, 8), 0.0, 1.0, 0) == -1
    assert raw_rc("npu_avgpool_xint8", z, NPU_I8, z, NPU_I8, *s3(4, 4, 8), 2, 2, 1, 1, U32(1 << 16),
                  15) == -1
    assert raw_rc("npu_avgpool_xint8", z, NPU_I8, z, NPU_I8, *s3(4, 4, 8), 2, 2, 1, 1, U32(8192),
                  -1) == -1
    assert raw_rc("npu_avgpool_xint8_fused", z, NPU_I8, z, NPU_I8, *s3(4, 4, 8), 2, 2, 1, 1,
                  U32(8192), 15, 2) == -1
