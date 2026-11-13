# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Every npu_emu entry point against the numpy reference, bit for bit.
16-bit and 8-bit float inputs are exhaustive; the rest mixes random values with
ties, saturation, special values and subnormals."""

import numpy as np
import pytest

import golden as g
from npu import (I64, OUT, A64Coeff, NPU_BF16, NPU_F16, NPU_F32, NPU_I8, NPU_I16,
                 NPU_I32, NPU_I64, NPU_U8, NPU_U16, NPU_U32, NPU_U64, call, f32)

RNG = np.random.default_rng(2026)
ALL16 = np.arange(65536, dtype=np.uint32).astype(np.uint16)
ALL8 = np.arange(256, dtype=np.uint16).astype(np.uint8)
INT_T = {"i8": (NPU_I8, np.int8), "u8": (NPU_U8, np.uint8), "i16": (NPU_I16, np.int16),
         "u16": (NPU_U16, np.uint16), "i32": (NPU_I32, np.int32), "u32": (NPU_U32, np.uint32),
         "i64": (NPU_I64, np.int64), "u64": (NPU_U64, np.uint64)}
FLT_T = {"f32": (NPU_F32, np.float32), "f16": (NPU_F16, np.uint16), "bf16": (NPU_BF16, np.uint16)}


def f32_values(n=1 << 16):
    """fp32 inputs: wide-range random, values on bf16 / f16 / fp8 half-way
    points, specials and subnormals."""
    r = (RNG.standard_normal(n) * np.exp2(RNG.integers(-30, 30, n))).astype(np.float32)
    bits = RNG.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
    ties = ((bits & 0xFFFF0000) | 0x8000).view(np.float32)            # bf16 ties
    ties16 = ((bits & 0xFFFFE000) | 0x1000).view(np.float32)          # f16 ties
    spec = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-45, -1e-45, 1e-39, 3.4e38,
                     65504, 65520, 65536, 448, 464, 465, -464, 480, 2.0**-9, 2.0**-10,
                     2.0**-24, 2.0**-25, 1.5 * 2.0**-25], np.float32)
    v = np.concatenate([r, ties, ties16, spec])
    return v[np.isfinite(v) | np.isnan(v)]


def float_input(t):
    if t == "f32":
        return f32_values()
    return ALL16


def ints(t, n=1 << 14):
    if t in ("i8", "u8", "i16", "u16"):
        info = np.iinfo(INT_T[t][1])
        return np.arange(info.min, info.max + 1, dtype=np.int64).astype(INT_T[t][1])
    info = np.iinfo(INT_T[t][1])
    edge = [info.min, info.min + 1, -1 if info.min < 0 else 0, 0, 1,
            info.max - 1, info.max, 2**24 + 1, 2**31 - 1, 2**32 - 1, 2**32, 2**63 - 1]
    edge = np.array([v for v in edge if info.min <= v <= info.max], dtype=object)
    if t == "u64":
        r = RNG.integers(0, 2**63, n, dtype=np.uint64) * np.uint64(2) + RNG.integers(0, 2, n, dtype=np.uint64)
    else:
        r = RNG.integers(info.min, info.max, n, dtype=INT_T[t][1], endpoint=True)
    return np.concatenate([edge.astype(INT_T[t][1]), r])


def pad(a, k):
    return np.concatenate([a, a[: (-len(a)) % k]]) if len(a) % k else a


def assert_same(got, want, x=None):
    got = np.asarray(got)
    want = np.asarray(want).astype(got.dtype)
    if got.dtype == np.uint8 or got.shape != want.shape:
        bad = np.nonzero(got.view(np.uint8) != want.view(np.uint8))[0]
    else:
        bad = np.nonzero(got.view(f"u{got.itemsize}") != want.view(f"u{got.itemsize}"))[0]
    if bad.size:
        show = ", ".join(
            f"[{i}] x={x[i] if x is not None and i < len(x) else '?'!r} got={got[i]!r} want={want[i]!r}"
            for i in bad[:6])
        raise AssertionError(f"{bad.size} differing elements: {show}")


# ---- A ----------------------------------------------------------------------

@pytest.mark.parametrize("ot", ["f16", "bf16"])
def test_convert_fp32_fp16(ot):
    x = f32_values()
    got = call("npu_convert_fp32_fp16", x, OUT, FLT_T[ot][0], I64(len(x)),
               out_nbytes=2 * len(x), out_dtype=np.uint16)
    assert_same(got, g.store_float(x, ot), x)


@pytest.mark.parametrize("it", ["f16", "bf16"])
def test_convert_fp16_fp32(it):
    got = call("npu_convert_fp16_fp32", ALL16, FLT_T[it][0], OUT, I64(65536),
               out_nbytes=4 * 65536, out_dtype=np.float32)
    assert_same(got, g.load_float(ALL16, it))


@pytest.mark.parametrize("it,ot", [("f16", "bf16"), ("bf16", "f16")])
def test_convert_fp16_bf16(it, ot):
    got = call("npu_convert_fp16_bf16", ALL16, FLT_T[it][0], OUT, FLT_T[ot][0], I64(65536),
               out_nbytes=2 * 65536, out_dtype=np.uint16)
    assert_same(got, g.store_float(g.load_float(ALL16, it), ot), ALL16)


def finite16(t):
    x = ALL16
    v = g.load_float(x, t)
    return x[np.isfinite(v)]


def test_cast_bf2bfp16():
    x = pad(finite16("bf16"), 8)
    got = call("npu_cast_bf2bfp16", x, OUT, I64(len(x)), out_nbytes=len(x) // 8 * 9,
               out_dtype=np.uint8)
    assert_same(got, g.to_bfp16(g.bf16_bits_to_f32(x), g.EVEN))


@pytest.mark.parametrize("it", ["bf16", "f16"])
def test_convert_bf16_to_bfp16(it):
    x = pad(RNG.permutation(finite16(it)), 8)
    got = call("npu_convert_bf16_to_bfp16", x, FLT_T[it][0], OUT, I64(len(x)),
               out_nbytes=len(x) // 8 * 9, out_dtype=np.uint8)
    assert_same(got, g.to_bfp16(g.load_float(x, it), g.AWAY))


def bfp16_blocks(nblk=8192):
    b = RNG.integers(0, 256, (nblk, 9), dtype=np.uint16).astype(np.uint8)
    b[:, 0] = RNG.integers(1, 255, nblk)
    return b.reshape(-1)


def test_cast_bfp16tobf():
    blk = bfp16_blocks()
    n = len(blk) // 9 * 8
    got = call("npu_cast_bfp16tobf", blk, OUT, I64(n), out_nbytes=2 * n, out_dtype=np.uint16)
    assert_same(got, g.to_bf16(g.from_bfp16(blk)))


def test_cast_bftomx6():
    x = pad(RNG.permutation(finite16("bf16")), 16)[: 16 * 2048]
    got = call("npu_cast_bftomx6", x, OUT, I64(len(x)), out_nbytes=len(x) // 16 * 12,
               out_dtype=np.uint8)
    assert_same(got, g.to_mx6(g.bf16_bits_to_f32(x), g.AWAY))


def test_cast_mx6tobf():
    blk = RNG.integers(0, 256, (2048, 12), dtype=np.uint16).astype(np.uint8)
    blk[:, 0] = RNG.integers(20, 235, 2048)
    blk = blk.reshape(-1)
    n = len(blk) // 12 * 16
    got = call("npu_cast_mx6tobf", blk, OUT, I64(n), out_nbytes=2 * n, out_dtype=np.uint16)
    assert_same(got, g.to_bf16(g.from_mx6(blk)), g.from_mx6(blk))


# ---- B ----------------------------------------------------------------------

@pytest.mark.parametrize("scale_recip", [1.0, 0.125, 3.0, 0.37])
def test_convert_fp32_fp8(scale_recip):
    x = f32_values()
    got = call("npu_convert_fp32_fp8", x, OUT, I64(len(x)), f32(scale_recip),
               out_nbytes=len(x), out_dtype=np.uint8)
    assert_same(got, g.fp32_fp8(x, scale_recip))


@pytest.mark.parametrize("it", ["f16", "bf16"])
@pytest.mark.parametrize("scale", [1.0, 4.0, 0.25, 3.0])
def test_convert_fp16_fp8(it, scale):
    x = ALL16
    got = call("npu_convert_fp16_fp8", x, FLT_T[it][0], OUT, I64(65536), f32(scale),
               out_nbytes=65536, out_dtype=np.uint8)
    assert_same(got, g.fp16_fp8(x, it, scale), x)


@pytest.mark.parametrize("scale", [1.0, 0.5, 3.0, 1e-3])
def test_convert_fp8_fp32(scale):
    got = call("npu_convert_fp8_fp32", ALL8, OUT, I64(256), f32(scale), out_nbytes=1024,
               out_dtype=np.float32)
    assert_same(got, g.fp8_to(ALL8, scale, "f32"))


@pytest.mark.parametrize("ot", ["f16", "bf16"])
@pytest.mark.parametrize("scale", [1.0, 0.5, 3.0, 1e-3, 300.0])
def test_convert_fp8_fp16(ot, scale):
    got = call("npu_convert_fp8_fp16", ALL8, OUT, FLT_T[ot][0], I64(256), f32(scale),
               out_nbytes=512, out_dtype=np.uint16)
    assert_same(got, g.fp8_to(ALL8, scale, ot))


# ---- C ----------------------------------------------------------------------

@pytest.mark.parametrize("it", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("ot", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("scale,zp", [(1.0, 0.0), (127.5, 128.0), (64.0, -64.0),
                                      (32767.5, 32768.0), (0.37, 3.0)])
def test_quant(it, ot, scale, zp):
    x = float_input(it)
    xf = g.load_float(x, it)
    keep = np.isfinite(xf)
    x, xf = x[keep], xf[keep]
    got = call("npu_quant", x, FLT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), f32(scale),
               f32(zp), out_nbytes=len(x) * np.dtype(INT_T[ot][1]).itemsize,
               out_dtype=INT_T[ot][1])
    assert_same(got, g.quant(xf, scale, zp, ot))


@pytest.mark.parametrize("ot", ["u8", "u16"])
@pytest.mark.parametrize("mul,zp", [(1.0, 0.0), (127.5, 128.0), (0.37, 3.0)])
def test_cast_bfp16toaq(ot, mul, zp):
    blk = bfp16_blocks()
    blk.reshape(-1, 9)[:, 0] = RNG.integers(118, 140, len(blk) // 9)
    n = len(blk) // 9 * 8
    got = call("npu_cast_bfp16toaq", blk, OUT, INT_T[ot][0], I64(n), f32(mul), f32(zp),
               out_nbytes=n * np.dtype(INT_T[ot][1]).itemsize, out_dtype=INT_T[ot][1])
    assert_same(got, g.bfp16_to_unsigned(blk, mul, zp, 255 if ot == "u8" else 65535))


# ---- D ----------------------------------------------------------------------

@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ot", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("scale,zp", [(1.0, 0.0), (0.0078125, 128.0), (0.37, 3.0),
                                      (3.0517578e-05, -32768.0), (1.7, 11.0)])
def test_dequant(it, ot, scale, zp):
    x = ints(it)
    got = call("npu_dequant", x, INT_T[it][0], OUT, FLT_T[ot][0], I64(len(x)), f32(scale),
               f32(zp), out_nbytes=len(x) * (4 if ot == "f32" else 2),
               out_dtype=FLT_T[ot][1])
    assert_same(got, g.dequant(x, scale, zp, ot))


@pytest.mark.parametrize("it", ["u8", "u16"])
@pytest.mark.parametrize("scale,zp", [(1.0, 0.0), (0.0078125, 128.0), (0.37, 3.7)])
def test_cast_aqtobfp16(it, scale, zp):
    x = pad(RNG.permutation(ints(it)), 8)
    got = call("npu_cast_aqtobfp16", x, INT_T[it][0], OUT, I64(len(x)), f32(scale), f32(zp),
               out_nbytes=len(x) // 8 * 9, out_dtype=np.uint8)
    assert_same(got, g.aq_to_bfp16(x, scale, zp))


def test_dequant_w4_bfp16():
    N, K = 64, 256
    q = RNG.integers(0, 16, N * K).astype(np.uint8)
    zp = RNG.integers(0, 16, N * K // 32).astype(np.uint8)
    s = g.to_bf16(RNG.uniform(-0.25, 0.25, N * K // 32).astype(np.float32))
    got = call("npu_dequant_w4_bfp16", q, zp, s, OUT, I64(N), I64(K),
               out_nbytes=N * K // 8 * 9, out_dtype=np.uint8)
    assert_same(got, g.w4_to_bfp16(q, zp, s, N, K))


# ---- E ----------------------------------------------------------------------

REQ_COEFF = [(0.7625, -7782.4), (1.15475, 0.3), (33333.332, 0.0), (1e-5, 5.5), (2.0, 1.0)]


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16", "i32", "u32"])
@pytest.mark.parametrize("ot", ["i8", "u8", "i16", "u16", "i32", "u32"])
@pytest.mark.parametrize("c1,c0", REQ_COEFF)
def test_requantize(it, ot, c1, c0):
    x = ints(it)
    got = call("npu_requantize", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), f32(c1),
               f32(c0), out_nbytes=len(x) * np.dtype(INT_T[ot][1]).itemsize,
               out_dtype=INT_T[ot][1])
    assert_same(got, g.requantize(x, it, c1, c0, ot))


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ot", ["i8", "u16", "i32", "u32"])
def test_requantize_channelwise(it, ot):
    x = ints(it)
    C = 96
    M = RNG.uniform(-3, 3, C).astype(np.float32)
    B = RNG.uniform(-100, 100, C).astype(np.float32)
    got = call("npu_requantize_channelwise", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)),
               M, B, I64(C), out_nbytes=len(x) * np.dtype(INT_T[ot][1]).itemsize,
               out_dtype=INT_T[ot][1])
    assert_same(got, g.requantize_cw(x, M, B, ot))


def a64_coeff(c1, c0, s):
    co = dict(mult_f=f32(c1), bias_f=f32(c0), mult_f_x2p32=f32(c1 * 2.0**32),
              mult_f_x2p64=f32(c1 * 2.0**64), M=int(round(c1 * 2**s)), s=s,
              B=int(round(c0 * 2**s)))
    return co, A64Coeff(**co)


@pytest.mark.parametrize("it", ["i64", "u64"])
@pytest.mark.parametrize("ot", ["i8", "u8", "i16", "u16", "i32", "u32"])
@pytest.mark.parametrize("c1,c0", [(1e-9, 3.0), (2.5e-12, -100.0), (1.0, 0.0)])
def test_requantize_a64_down(it, ot, c1, c0):
    x = ints(it)
    co, cs = a64_coeff(c1, c0, 0)
    got = call("npu_requantize_a64", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), cs,
               out_nbytes=len(x) * np.dtype(INT_T[ot][1]).itemsize, out_dtype=INT_T[ot][1])
    assert_same(got, g.a64_down(x, it, co, ot))


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16", "i32", "u32"])
@pytest.mark.parametrize("ot", ["i64", "u64"])
@pytest.mark.parametrize("c1,c0,s", [(3.0, -5.0, 0), (1.37, 2.5, 20), (1000.25, -7.0, 8)])
def test_requantize_a64_up(it, ot, c1, c0, s):
    x = ints(it, 4096)
    co, cs = a64_coeff(c1, c0, s)
    got = call("npu_requantize_a64", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), cs,
               out_nbytes=8 * len(x), out_dtype=np.int64)
    assert_same(got, g.a64_up(x, it, co, ot))


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ot", ["i32", "i64"])
def test_cast_int32_int64(it, ot):
    x = ints(it)
    c1, c0 = 0.0056104352697730064 / 0.0003, -100.0
    got = call("npu_cast_int32_int64", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), f32(c1),
               f32(c0), out_nbytes=len(x) * (4 if ot == "i32" else 8),
               out_dtype=np.int32 if ot == "i32" else np.int64)
    assert_same(got, g.int32_int64(x, c1, c0))


# ---- F ----------------------------------------------------------------------

@pytest.mark.parametrize("mul,zp,scale", [(1.0, 0.0, 1.0), (127.5, 32768.0, 0.0078125),
                                          (0.37, 3.0, 2.7)])
def test_cast_bfp16toqdqf32(mul, zp, scale):
    blk = bfp16_blocks()
    blk.reshape(-1, 9)[:, 0] = RNG.integers(118, 145, len(blk) // 9)
    n = len(blk) // 9 * 8
    got = call("npu_cast_bfp16toqdqf32", blk, OUT, I64(n), f32(mul), f32(zp), f32(scale),
               out_nbytes=4 * n, out_dtype=np.float32)
    assert_same(got, g.bfp16_qdq_f32(blk, mul, zp, scale))
