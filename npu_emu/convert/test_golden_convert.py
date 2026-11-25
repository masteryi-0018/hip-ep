# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""convert.h entry points against the numpy reference model, bit for bit."""

import ctypes

import numpy as np
import pytest

import golden_block_fp as gb
import golden_mmac as gmm
import golden_convert as g
import golden_float_cvt as gf
from npu import (I64, OUT, NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, NPU_I32,
                 NPU_I64, NPU_U8, NPU_U32, NPU_U64, Host, call, f32, raw_rc)
from testing import (ALL8, ALL16, COEF, COEF_PAIRS, F32_SPEC, SP16, bits32, x16_special)
from testing import INT_ALL as INT_T

RNG = np.random.default_rng(2026)
FLT_T = {"f32": (NPU_F32, np.float32), "f16": (NPU_F16, np.uint16), "bf16": (NPU_BF16, np.uint16)}


class A64Coeff(ctypes.Structure):
    _fields_ = [("mult_f", ctypes.c_float), ("bias_f", ctypes.c_float),
                ("mult_f_x2p32", ctypes.c_float), ("mult_f_x2p64", ctypes.c_float),
                ("M", ctypes.c_int32), ("s", ctypes.c_int32), ("B", ctypes.c_int64)]


def f32_values(n=1 << 16):
    """Random fp32 values plus rounding ties, specials and subnormals."""
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
        u = got.view(f"u{got.itemsize}"), want.view(f"u{got.itemsize}")
        show = ", ".join(
            f"[{i}] x={x[i] if x is not None and i < len(x) else '?'!r} got={int(u[0][i]):#x} "
            f"want={int(u[1][i]):#x}" for i in bad[:6])
        raise AssertionError(f"{bad.size} differing elements: {show}")


@pytest.mark.parametrize("ot", ["f16", "bf16"])
def test_convert_fp32_fp16(ot):
    x = f32_values()
    got = call("npu_convert_fp32_fp16", x, OUT, FLT_T[ot][0], I64(len(x)),
               out_nbytes=2 * len(x), out_dtype=np.uint16)
    assert_same(got, g.convert_fp32_fp16(x, ot), x)


@pytest.mark.parametrize("it", ["f16", "bf16"])
def test_convert_fp16_fp32(it):
    got = call("npu_convert_fp16_fp32", ALL16, FLT_T[it][0], OUT, I64(65536),
               out_nbytes=4 * 65536, out_dtype=np.float32)
    assert_same(got, g.convert_fp16_fp32(ALL16, it))


@pytest.mark.parametrize("it,ot", [("f16", "bf16"), ("bf16", "f16")])
def test_convert_fp16_bf16(it, ot):
    got = call("npu_convert_fp16_bf16", ALL16, FLT_T[it][0], OUT, FLT_T[ot][0], I64(65536),
               out_nbytes=2 * 65536, out_dtype=np.uint16)
    assert_same(got, g.convert_fp16_bf16(ALL16, it, ot), ALL16)


def finite16(t):
    x = ALL16
    v = gf.load_float(x, t)
    return x[np.isfinite(v)]


def test_cast_bf2bfp16():
    x = pad(finite16("bf16"), 64)
    got = call("npu_cast_bf2bfp16", x, OUT, I64(len(x)), out_nbytes=len(x) // 8 * 9,
               out_dtype=np.uint8)
    assert_same(got, g.cast_bf2bfp16(x))


@pytest.mark.parametrize("it", ["bf16", "f16"])
def test_convert_bf16_to_bfp16(it):
    x = pad(RNG.permutation(finite16(it)), 8)
    got = call("npu_convert_bf16_to_bfp16", x, FLT_T[it][0], OUT, I64(len(x)),
               out_nbytes=len(x) // 8 * 9, out_dtype=np.uint8)
    assert_same(got, g.convert_bf16_to_bfp16(x, it))


def bfp16_blocks(nblk=8192):
    b = RNG.integers(0, 256, (nblk, 9), dtype=np.uint16).astype(np.uint8)
    b[:, 0] = RNG.integers(1, 255, nblk)
    return b.reshape(-1)


def test_cast_bfp16tobf():
    blk = bfp16_blocks()
    n = len(blk) // 9 * 8
    got = call("npu_cast_bfp16tobf", blk, OUT, I64(n), out_nbytes=2 * n, out_dtype=np.uint16)
    assert_same(got, g.cast_bfp16tobf(blk))


def bfp16_edge_blocks():
    """64-value groups whose block exponents differ widely (spans 17, 18, 24, 25, 26, > 100),
    groups with an exponent-255 block, groups of tiny exponents, exponents 0..255 in order,
    random exponents and mantissas."""
    spans = [0, 16, 17, 18, 20, 24, 25, 26, 40, 120, 253]
    groups = []
    for top in (140, 200, 254):
        for s in spans:
            e = np.full(8, top, np.int64)
            e[1::2] = max(top - s, 0)
            groups.append(e)
    for e0 in (0, 1, 3, 8, 9, 10):
        groups.append(np.full(8, e0) + np.arange(8) % 2)
    for k in range(8):
        e = RNG.integers(100, 160, 8)
        e[k] = 255
        groups.append(e)
    groups.append(np.arange(512) % 256)
    groups.append(RNG.integers(0, 256, 8 * 64))
    groups.append(RNG.integers(0, 255, 8 * 64))
    groups.append(np.array([150, 120, 133, 150, 120, 133, 140, 141]))  # even group count
    e = np.concatenate(groups).astype(np.uint8)
    m = RNG.integers(-128, 128, (len(e), 8)).astype(np.int8)
    m[: 8 * 3 * len(spans), :] = np.array([127, -128, 1, -1, 64, -65, 3, -3], np.int8)
    return np.concatenate([e[:, None], m.view(np.uint8)], axis=1).reshape(-1)


def test_bfp16_group_decode_values():
    blk = np.zeros((24, 9), np.uint8)
    blk[:8, 0] = [150, 132, 124, 133, 0, 0, 0, 0]
    blk[8:16, 0] = [0, 1, 0, 0, 0, 0, 0, 0]
    blk[16:, 0] = [150, 120, 0, 0, 0, 0, 0, 255]
    blk[:, 1:] = np.array([1, 3, 127, -5, 2, -128, 0, 0], np.int8).view(np.uint8)
    v = gmm.decode_bfp16_group(blk).reshape(3, 8, 8)
    assert v[0, 0, 0] == 2.0 ** 17
    assert v[0, 1, 1] == 2.0 and v[0, 1, 2] == 64.0  # window 2^-1: 1.5 -> 2, 63.5 -> 64
    assert not v[0, 2].any() and v[0, 3, 3] == -5.0  # span 26: 0; span 17: exact
    assert v[1, 0, 5] == -(2.0 ** -126) and v[1, 1, 2] == np.float32(127 * 2.0 ** -132)
    small = v[1].reshape(-1)[[2, 3, 8, 11]]  # 127, -5 * 2^-133 and 1, -5 * 2^-132
    assert (small.view(np.uint32) == 0).all()  # below 2^-126: +0
    assert (v[2].view(np.uint32) == 0x7F810000).all()
    assert (gmm.decode_bfp16_bf16(blk)[128:] == 0x7F81).all()


def test_cast_bfp16tobf_edge():
    blk = bfp16_edge_blocks()
    n = len(blk) // 9 * 8
    got = call("npu_cast_bfp16tobf", blk, OUT, I64(n), out_nbytes=2 * n, out_dtype=np.uint16)
    want = g.cast_bfp16tobf(blk)
    assert_same(got, want)
    assert (want == 0x7F81).sum() >= 64 * 9 and (want != gf.f32_to_bf16(gb.decode_bfp16(blk))).any()


def test_cast_bftomx6():
    x = pad(RNG.permutation(finite16("bf16")), 16)[: 16 * 2048]
    got = call("npu_cast_bftomx6", x, OUT, I64(len(x)), out_nbytes=len(x) // 16 * 12,
               out_dtype=np.uint8)
    assert_same(got, g.cast_bftomx6(x))


def test_cast_mx6tobf():
    blk = RNG.integers(0, 256, (2048, 12), dtype=np.uint16).astype(np.uint8)
    blk[:, 0] = RNG.integers(20, 235, 2048)
    blk = blk.reshape(-1)
    n = len(blk) // 12 * 16
    got = call("npu_cast_mx6tobf", blk, OUT, I64(n), out_nbytes=2 * n, out_dtype=np.uint16)
    assert_same(got, g.cast_mx6tobf(blk), gb.decode_mx6(blk))


@pytest.mark.parametrize("scale_recip", [1.0, 0.125, 3.0, 0.37])
def test_convert_fp32_fp8(scale_recip):
    x = f32_values()
    got = call("npu_convert_fp32_fp8", x, OUT, I64(len(x)), f32(scale_recip),
               out_nbytes=len(x), out_dtype=np.uint8)
    assert_same(got, g.convert_fp32_fp8(x, scale_recip))


@pytest.mark.parametrize("it", ["f16", "bf16"])
@pytest.mark.parametrize("scale", [1.0, 4.0, 0.25, 3.0])
def test_convert_fp16_fp8(it, scale):
    x = ALL16
    got = call("npu_convert_fp16_fp8", x, FLT_T[it][0], OUT, I64(65536), f32(scale),
               out_nbytes=65536, out_dtype=np.uint8)
    assert_same(got, g.convert_fp16_fp8(x, it, scale), x)


@pytest.mark.parametrize("scale", [1.0, 0.5, 3.0, 1e-3])
def test_convert_fp8_fp32(scale):
    got = call("npu_convert_fp8_fp32", ALL8, OUT, I64(256), f32(scale), out_nbytes=1024,
               out_dtype=np.float32)
    assert_same(got, g.convert_fp8_fp32(ALL8, scale))


@pytest.mark.parametrize("ot", ["f16", "bf16"])
@pytest.mark.parametrize("scale", [1.0, 0.5, 3.0, 1e-3, 300.0])
def test_convert_fp8_fp16(ot, scale):
    got = call("npu_convert_fp8_fp16", ALL8, OUT, FLT_T[ot][0], I64(256), f32(scale),
               out_nbytes=512, out_dtype=np.uint16)
    assert_same(got, g.convert_fp8_fp16(ALL8, ot, scale))


@pytest.mark.parametrize("coeff", [1.0, 0.5, 1.5, 3.0, 0.37, 1e-3, 300.0, 1 / 3])
def test_requantize_fp8(coeff):
    x = np.tile(ALL8[(ALL8 & 0x7F) != 0x7F], 4)
    got = call("npu_requantize_fp8", x, OUT, I64(len(x)), f32(coeff), out_nbytes=len(x),
               out_dtype=np.uint8)
    assert_same(got, g.requantize_fp8(x, f32(coeff)))


@pytest.mark.parametrize("it", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("ot", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("scale,zp", [(1.0, 0.0), (127.5, 128.0), (64.0, -64.0),
                                      (32767.5, 32768.0), (0.37, 3.0)])
def test_quant(it, ot, scale, zp):
    x = float_input(it)
    x = x[np.isfinite(gf.load_float(x, it))]
    got = call("npu_quant", x, FLT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), f32(zp),
               f32(scale), out_nbytes=len(x) * np.dtype(INT_T[ot][1]).itemsize,
               out_dtype=INT_T[ot][1])
    assert_same(got, g.quant(x, it, ot, zp, scale))


@pytest.mark.parametrize("ot", ["u8", "u16"])
@pytest.mark.parametrize("mul,zp", [(1.0, 0.0), (127.5, 128.0), (0.37, 3.0)])
def test_cast_bfp16toaq(ot, mul, zp):
    blk = bfp16_blocks()
    blk.reshape(-1, 9)[:, 0] = RNG.integers(118, 140, len(blk) // 9)
    n = len(blk) // 9 * 8
    name = "cast_bfp16toa" + ot[1:]
    got = call("npu_" + name, blk, OUT, I64(n), f32(mul), f32(zp),
               out_nbytes=n * np.dtype(INT_T[ot][1]).itemsize, out_dtype=INT_T[ot][1])
    assert_same(got, getattr(g, name)(blk, mul, zp))


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ot", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("scale,zp", [(1.0, 0.0), (0.0078125, 128.0), (0.37, 3.0),
                                      (3.0517578e-05, -32768.0), (1.7, 11.0)])
def test_dequant(it, ot, scale, zp):
    x = ints(it)
    got = call("npu_dequant", x, INT_T[it][0], OUT, FLT_T[ot][0], I64(len(x)), f32(zp),
               f32(scale), out_nbytes=len(x) * (4 if ot == "f32" else 2),
               out_dtype=FLT_T[ot][1])
    assert_same(got, g.dequant(x, it, ot, zp, scale))


@pytest.mark.parametrize("it", ["u8", "u16"])
@pytest.mark.parametrize("scale,zp", [(1.0, 0.0), (0.0078125, 128.0), (0.37, 3.7)])
def test_cast_aqtobfp16(it, scale, zp):
    x = pad(RNG.permutation(ints(it)), 64)
    name = f"cast_a{it[1:]}tobfp16"
    got = call("npu_" + name, x, OUT, I64(len(x)), f32(scale), f32(zp),
               out_nbytes=len(x) // 8 * 9, out_dtype=np.uint8)
    assert_same(got, getattr(g, name)(x, scale, zp))


def test_dequant_w4_bfp16():
    N, K = 64, 256
    q = RNG.integers(0, 16, N * K).astype(np.uint8)
    zp = RNG.integers(0, 16, N * K // 32).astype(np.uint8)
    s = gf.f32_to_bf16(RNG.uniform(-0.25, 0.25, N * K // 32).astype(np.float32))
    got = call("npu_dequant_w4_bfp16", zp, s, q, OUT, I64(N), I64(K),
               out_nbytes=N * K // 8 * 9, out_dtype=np.uint8)
    assert_same(got, g.dequant_w4_bfp16(zp, s, q, N, K))


REQ_COEFF = [(0.7625, -7782.4), (1.15475, 0.3), (33333.332, 0.0), (1e-5, 5.5), (2.0, 1.0)]


REQ_PAIRS = [(a, b) for a in INT_T if a[1:] in ("8", "16", "32") for b in INT_T
             if b[1:] in ("8", "16", "32") and not (a[1:] == b[1:] == "32")]


@pytest.mark.parametrize("it,ot", REQ_PAIRS)
@pytest.mark.parametrize("c1,c0", REQ_COEFF)
def test_requantize(it, ot, c1, c0):
    x = ints(it)
    got = call("npu_requantize", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), f32(c1),
               f32(c0), out_nbytes=len(x) * np.dtype(INT_T[ot][1]).itemsize,
               out_dtype=INT_T[ot][1])
    assert_same(got, g.requantize(x, it, ot, c1, c0))


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ot", ["i8", "u16", "i32", "u32"])
def test_requantize_channelwise(it, ot):
    C = 96
    x = pad(ints(it), C)
    M = RNG.uniform(-3, 3, C).astype(np.float32)
    B = RNG.uniform(-100, 100, C).astype(np.float32)
    name = f"requantize_a{it[1:]}_channelwise"
    got = call("npu_" + name, x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), I64(C), M, B,
               out_nbytes=len(x) * np.dtype(INT_T[ot][1]).itemsize, out_dtype=INT_T[ot][1])
    assert_same(got, getattr(g, name)(x, it, ot, C, M, B))


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
    assert_same(got, g.requantize_a64(x, it, ot, co))


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16", "i32", "u32"])
@pytest.mark.parametrize("ot", ["i64", "u64"])
@pytest.mark.parametrize("c1,c0,s", [(3.0, -5.0, 0), (1.37, 2.5, 20), (1000.25, -7.0, 8)])
def test_requantize_a64_up(it, ot, c1, c0, s):
    x = ints(it, 4096)
    co, cs = a64_coeff(c1, c0, s)
    got = call("npu_requantize_a64", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), cs,
               out_nbytes=8 * len(x), out_dtype=np.int64)
    assert_same(got, g.requantize_a64(x, it, ot, co))


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ot", ["i32", "i64"])
def test_cast_int32_int64(it, ot):
    x = ints(it)
    c1, c0 = 0.0056104352697730064 / 0.0003, -100.0
    got = call("npu_cast_int32_int64", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), f32(c1),
               f32(c0), out_nbytes=len(x) * (4 if ot == "i32" else 8),
               out_dtype=np.int32 if ot == "i32" else np.int64)
    assert_same(got, g.cast_int32_int64(x, it, ot, c1, c0))


@pytest.mark.parametrize("mul,zp,scale", [(1.0, 0.0, 1.0), (127.5, 32768.0, 0.0078125),
                                          (0.37, 3.0, 2.7)])
def test_cast_bfp16toqdqf32(mul, zp, scale):
    blk = bfp16_blocks()
    blk.reshape(-1, 9)[:, 0] = RNG.integers(118, 145, len(blk) // 9)
    n = len(blk) // 9 * 8
    got = call("npu_cast_bfp16toqdqf32", blk, OUT, I64(n), f32(mul), f32(zp), f32(scale),
               out_nbytes=4 * n, out_dtype=np.float32)
    assert_same(got, g.cast_bfp16toqdqf32(blk, mul, zp, scale))


@pytest.mark.parametrize("ot,mul,zp", [("u8", 1.0, 0.0), ("u8", 2.0 ** 120, 128.0),
                                       ("u16", 2.0 ** 125, 32768.0), ("u16", 0.37, 3.0)])
def test_cast_bfp16toaq_edge(ot, mul, zp):
    blk = bfp16_edge_blocks()
    n = len(blk) // 9 * 8
    name = "cast_bfp16toa" + ot[1:]
    got = call("npu_" + name, blk, OUT, I64(n), f32(mul), f32(zp),
               out_nbytes=n * np.dtype(INT_T[ot][1]).itemsize, out_dtype=INT_T[ot][1])
    want = getattr(g, name)(blk, mul, zp)
    assert_same(got, want)
    assert (want[gmm.decode_bfp16_bf16(blk) == 0x7F81] == 0).all()  # a NaN group quantizes to 0


@pytest.mark.parametrize("mul,zp,scale", [(1.0, 0.0, 1.0), (2.0 ** 125, 32768.0, 2.0 ** -125)])
def test_cast_bfp16toqdqf32_edge(mul, zp, scale):
    blk = bfp16_edge_blocks()
    n = len(blk) // 9 * 8
    got = call("npu_cast_bfp16toqdqf32", blk, OUT, I64(n), f32(mul), f32(zp), f32(scale),
               out_nbytes=4 * n, out_dtype=np.float32)
    assert_same(got, g.cast_bfp16toqdqf32(blk, mul, zp, scale))


# ---- special values: NaN (both signs, payloads), inf, +-0, subnormals, overflow, integer edges.
# Everything is compared bit for bit, NaN patterns included. A signaling NaN cannot be passed as
# a scalar argument (the float -> double -> float round trip quiets it), so scalar NaNs are quiet.

def fbits(v):
    return hex(int(np.float32(v).view(np.uint32)))


def i_special(t, n=256):
    info = np.iinfo(INT_T[t][1])
    edge = [info.min, info.min + 1, -1 if info.min < 0 else 0, 0, 1, info.max - 1, info.max,
            2**24 + 1, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**63 - 1, -2**31, -2**31 - 1]
    edge = np.array([v for v in edge if info.min <= v <= info.max], dtype=object)
    r = RNG.integers(info.min, info.max, n, dtype=INT_T[t][1], endpoint=True)
    return np.concatenate([edge.astype(INT_T[t][1]), r])


def special_halves(t, nblk, per):
    """nblk blocks of `per` 16-bit codes; per block a different share of special codes."""
    rate = RNG.choice([0.0, 0.1, 0.5, 1.0], nblk)[:, None]
    base = RNG.permutation(finite16(t))[: nblk * per].reshape(nblk, per)
    sp = SP16[RNG.integers(0, len(SP16), (nblk, per))]
    same = np.repeat(SP16[RNG.integers(0, len(SP16), (nblk, 1))], per, axis=1)
    pick = RNG.random((nblk, per)) < rate
    x = np.where(pick, sp, base)
    x[::7] = same[::7]
    return x.reshape(-1).astype(np.uint16)


def f32_all_nan_patterns():
    """NaN patterns of both signs: every top 16 bits, several low payloads."""
    top = np.arange(0x7F80, 0x8000, dtype=np.uint32)
    top = np.concatenate([top, top | 0x8000])
    out = [(top << 16) | low for low in (0, 1, 0x8000, 0xFFFF, 0x2000, 0x1FFF, 0x7FFF)]
    b = np.concatenate(out)
    return b[(b & 0x7FFFFF) != 0].view(np.float32)


@pytest.mark.parametrize("ot", ["f16", "bf16"])
def test_convert_fp32_fp16_special(ot):
    x = np.concatenate([F32_SPEC, f32_all_nan_patterns(), -F32_SPEC])
    got = call("npu_convert_fp32_fp16", x, OUT, FLT_T[ot][0], I64(len(x)),
               out_nbytes=2 * len(x), out_dtype=np.uint16)
    assert_same(got, g.convert_fp32_fp16(x, ot), x)


@pytest.mark.parametrize("it", ["bf16", "f16"])
def test_convert_bfp16_special(it):
    x = special_halves(it, 2048, 8)
    got = call("npu_convert_bf16_to_bfp16", x, FLT_T[it][0], OUT, I64(len(x)),
               out_nbytes=len(x) // 8 * 9, out_dtype=np.uint8)
    assert_same(got, g.convert_bf16_to_bfp16(x, it))
    if it == "bf16":
        got = call("npu_cast_bf2bfp16", x, OUT, I64(len(x)), out_nbytes=len(x) // 8 * 9,
                   out_dtype=np.uint8)
        assert_same(got, g.cast_bf2bfp16(x))


def test_cast_bftomx6_special():
    x = special_halves("bf16", 2048, 16)
    got = call("npu_cast_bftomx6", x, OUT, I64(len(x)), out_nbytes=len(x) // 16 * 12,
               out_dtype=np.uint8)
    assert_same(got, g.cast_bftomx6(x))


def test_cast_mx6tobf_special():
    """Every block exponent 0..255 (underflow / overflow of the decode), any field bits."""
    blk = RNG.integers(0, 256, (4096, 12), dtype=np.uint16).astype(np.uint8)
    blk[:, 0] = np.arange(4096) % 256
    blk[::5, 4:] = 0xFF
    blk[1::5, 4:] = 0
    blk = blk.reshape(-1)
    n = len(blk) // 12 * 16
    got = call("npu_cast_mx6tobf", blk, OUT, I64(n), out_nbytes=2 * n, out_dtype=np.uint16)
    assert_same(got, g.cast_mx6tobf(blk), gb.decode_mx6(blk))


SCALES = [c for c in COEF if c != 1.0] + [1.0, 2.0**-100, 2.0**100]


@pytest.mark.parametrize("scale_recip", SCALES)
def test_convert_fp32_fp8_special(scale_recip):
    x = np.concatenate([F32_SPEC, f32_all_nan_patterns()[::31]])
    got = call("npu_convert_fp32_fp8", x, OUT, I64(len(x)), f32(scale_recip),
               out_nbytes=len(x), out_dtype=np.uint8)
    assert_same(got, g.convert_fp32_fp8(x, scale_recip), x)


@pytest.mark.parametrize("it", ["f16", "bf16"])
@pytest.mark.parametrize("scale", SCALES)
def test_convert_fp16_fp8_special(it, scale):
    x = ALL16
    got = call("npu_convert_fp16_fp8", x, FLT_T[it][0], OUT, I64(65536), f32(scale),
               out_nbytes=65536, out_dtype=np.uint8)
    assert_same(got, g.convert_fp16_fp8(x, it, scale), x)


def fp8_rows(n=2048):
    """fp8 codes with NaN codes scattered (one NaN per 8-row spreads over the row)."""
    return np.concatenate([ALL8, fp8_values_nan(n)])


def fp8_values_nan(n):
    c = RNG.integers(0, 256, n).astype(np.uint8)
    c = np.where(RNG.random(n) < 0.03, np.where(RNG.random(n) < 0.5, 0x7F, 0xFF), c)
    return c.astype(np.uint8)


@pytest.mark.parametrize("scale", SCALES)
def test_convert_fp8_fp32_special(scale):
    x = fp8_rows()
    got = call("npu_convert_fp8_fp32", x, OUT, I64(len(x)), f32(scale), out_nbytes=4 * len(x),
               out_dtype=np.float32)
    assert_same(got, g.convert_fp8_fp32(x, scale))


@pytest.mark.parametrize("ot", ["f16", "bf16"])
@pytest.mark.parametrize("scale", SCALES)
def test_convert_fp8_fp16_special(ot, scale):
    x = fp8_rows()
    got = call("npu_convert_fp8_fp16", x, OUT, FLT_T[ot][0], I64(len(x)), f32(scale),
               out_nbytes=2 * len(x), out_dtype=np.uint16)
    assert_same(got, g.convert_fp8_fp16(x, ot, scale))


@pytest.mark.parametrize("coeff", SCALES)
def test_requantize_fp8_special(coeff):
    x = fp8_rows()
    got = call("npu_requantize_fp8", x, OUT, I64(len(x)), f32(coeff), out_nbytes=len(x),
               out_dtype=np.uint8)
    assert_same(got, g.requantize_fp8(x, f32(coeff)))


@pytest.mark.parametrize("it", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("ot", ["i8", "u8", "i16", "u16"])
def test_quant_special(it, ot):
    x = F32_SPEC if it == "f32" else x16_special()
    for scale, zp in COEF_PAIRS:
        got = call("npu_quant", x, FLT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), f32(zp),
                   f32(scale), out_nbytes=len(x) * np.dtype(INT_T[ot][1]).itemsize,
                   out_dtype=INT_T[ot][1])
        try:
            assert_same(got, g.quant(x, it, ot, zp, scale), x)
        except AssertionError as e:
            raise AssertionError(f"scale={scale!r} zp={zp!r}: {e}")


@pytest.mark.parametrize("ot", ["u8", "u16"])
def test_cast_bfp16toaq_special(ot):
    blk = bfp16_edge_blocks()
    n = len(blk) // 9 * 8
    name = "cast_bfp16toa" + ot[1:]
    for mul, zp in COEF_PAIRS:
        got = call("npu_" + name, blk, OUT, I64(n), f32(mul), f32(zp),
                   out_nbytes=n * np.dtype(INT_T[ot][1]).itemsize, out_dtype=INT_T[ot][1])
        try:
            assert_same(got, getattr(g, name)(blk, mul, zp))
        except AssertionError as e:
            raise AssertionError(f"mul={mul!r} zp={zp!r}: {e}")


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ot", ["f32", "f16", "bf16"])
def test_dequant_special(it, ot):
    x = i_special(it, 512)
    for scale, zp in COEF_PAIRS:
        got = call("npu_dequant", x, INT_T[it][0], OUT, FLT_T[ot][0], I64(len(x)), f32(zp),
                   f32(scale), out_nbytes=len(x) * (4 if ot == "f32" else 2),
                   out_dtype=FLT_T[ot][1])
        try:
            assert_same(got, g.dequant(x, it, ot, zp, scale), x)
        except AssertionError as e:
            raise AssertionError(f"scale={fbits(scale)} zp={fbits(zp)}: {e}")


ZP_EDGE = COEF + [2.0**31, -2.0**31, 2.0**31 - 128, 2.0**32, -2.0**32, 3.7, -3.7, 65535.5, 1e30]


@pytest.mark.parametrize("it", ["u8", "u16"])
def test_cast_aqtobfp16_special(it):
    x = pad(i_special(it, 248), 64)
    name = f"cast_a{it[1:]}tobfp16"
    for scale, zp in [(s, z) for s in COEF[:12:2] + [0.37] for z in ZP_EDGE]:
        got = call("npu_" + name, x, OUT, I64(len(x)), f32(scale), f32(zp),
                   out_nbytes=len(x) // 8 * 9, out_dtype=np.uint8)
        try:
            assert_same(got, getattr(g, name)(x, scale, zp))
        except AssertionError as e:
            raise AssertionError(f"scale={scale!r} zp={zp!r}: {e}")


def test_dequant_w4_bfp16_special():
    N, K = 64, 256
    q = RNG.integers(0, 256, N * K).astype(np.uint8)
    zp = RNG.integers(0, 256, N * K // 32).astype(np.uint8)
    q[: K * 8] = np.repeat(zp[: K * 8 // 32], 32)  # q == zp: a zero difference
    s = SP16[RNG.integers(0, len(SP16), N * K // 32)]
    s[::3] = gf.f32_to_bf16(RNG.uniform(-0.25, 0.25, len(s[::3])).astype(np.float32))
    got = call("npu_dequant_w4_bfp16", zp, s, q, OUT, I64(N), I64(K), out_nbytes=N * K // 8 * 9,
               out_dtype=np.uint8)
    assert_same(got, g.dequant_w4_bfp16(zp, s, q, N, K))


@pytest.mark.parametrize("it,ot", REQ_PAIRS)
def test_requantize_special(it, ot):
    x = i_special(it, 64)
    for c1, c0 in COEF_PAIRS + [(2.0**100, 0.0), (-2.0**100, 1.0), (2.0**97, -2.0**97)]:
        got = call("npu_requantize", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), f32(c1),
                   f32(c0), out_nbytes=len(x) * np.dtype(INT_T[ot][1]).itemsize,
                   out_dtype=INT_T[ot][1])
        try:
            assert_same(got, g.requantize(x, it, ot, c1, c0))
        except AssertionError as e:
            raise AssertionError(f"c1={c1!r} c0={c0!r}: {e}")


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ot", ["i8", "u16", "i32", "u32"])
def test_requantize_channelwise_special(it, ot):
    x = pad(i_special(it, 512), 32)
    M = np.array(COEF + [-2.0**100], np.float32)
    B = np.roll(np.array(COEF + [3e9], np.float32), 5)
    M = np.concatenate([M, bits32(0x7FC12345, 0xFFC00000, 0x7F800001)])  # NaN payloads
    B = np.concatenate([B, bits32(0x7F800001, 0xFFC12345, 0x7FC00000)])
    name = f"requantize_a{it[1:]}_channelwise"
    got = call("npu_" + name, x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)), I64(len(M)), M, B,
               out_nbytes=len(x) * np.dtype(INT_T[ot][1]).itemsize, out_dtype=INT_T[ot][1])
    assert_same(got, getattr(g, name)(x, it, ot, len(M), M, B))


A64_SPEC = [float("inf"), float("-inf"), float(bits32(0x7FC00000)[0]), 0.0, -0.0, 1e-45, 3.4e38,
            1.0]


@pytest.mark.parametrize("it", ["i64", "u64"])
@pytest.mark.parametrize("ot", ["i8", "u8", "i16", "u16", "i32", "u32"])
def test_requantize_a64_down_special(it, ot):
    x = i_special(it, 64)
    for k in range(len(A64_SPEC)):
        f = [A64_SPEC[(k + j) % len(A64_SPEC)] for j in range(4)]
        co = dict(mult_f=f32(f[0]), bias_f=f32(f[1]), mult_f_x2p32=f32(f[2]),
                  mult_f_x2p64=f32(f[3]), M=0, s=0, B=0)
        got = call("npu_requantize_a64", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)),
                   A64Coeff(**co), out_nbytes=len(x) * np.dtype(INT_T[ot][1]).itemsize,
                   out_dtype=INT_T[ot][1])
        try:
            assert_same(got, g.requantize_a64(x, it, ot, co))
        except AssertionError as e:
            raise AssertionError(f"fields={f!r}: {e}")


I32_EDGE, I64_EDGE = [0, 1, -1, 2**31 - 1, -2**31, 3, -3], [0, 1, -1, 2**63 - 1, -2**63, 2**40]
S_EDGE = [0, 1, 7, 31, 32, 33, 63, 64, 65, 127, 1000, 2**30, 2**31 - 1, -1, -64, -2**31]


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16", "i32", "u32"])
@pytest.mark.parametrize("ot", ["i64", "u64"])
def test_requantize_a64_up_special(it, ot):
    x = i_special(it, 24)
    combos = [(m, s, b) for m in I32_EDGE for s in S_EDGE[::2] for b in I64_EDGE[::2]]
    combos += [(M, s, 0) for M in (2**31 - 1, -2**31) for s in S_EDGE]
    combos += [(3, s, B) for s in S_EDGE for B in (2**63 - 1, -2**63)]
    for M, s, B in combos:
        co = dict(mult_f=0.0, bias_f=0.0, mult_f_x2p32=0.0, mult_f_x2p64=0.0, M=M, s=s, B=B)
        got = call("npu_requantize_a64", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)),
                   A64Coeff(**co), out_nbytes=8 * len(x), out_dtype=np.int64)
        try:
            assert_same(got, g.requantize_a64(x, it, ot, co))
        except AssertionError as e:
            raise AssertionError(f"M={M} s={s} B={B}: {e}")


@pytest.mark.parametrize("it", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("ot", ["i32", "i64"])
def test_cast_int32_int64_special(it, ot):
    x = i_special(it, 512)
    for c1, c0 in COEF_PAIRS + [(2.0**31, 0.0), (-2.0**31, 0.0), (2.0**31, 2.0**31)]:
        got = call("npu_cast_int32_int64", x, INT_T[it][0], OUT, INT_T[ot][0], I64(len(x)),
                   f32(c1), f32(c0), out_nbytes=len(x) * (4 if ot == "i32" else 8),
                   out_dtype=np.int32 if ot == "i32" else np.int64)
        try:
            assert_same(got, g.cast_int32_int64(x, it, ot, c1, c0))
        except AssertionError as e:
            raise AssertionError(f"c1={c1!r} c0={c0!r}: {e}")


def test_cast_bfp16toqdqf32_special():
    blk = bfp16_edge_blocks()
    n = len(blk) // 9 * 8
    for mul, zp, scale in ([(m, z, 1.0) for m in (1.0, float("inf"), COEF[6], 0.0, 3.4e38)
                            for z in ZP_EDGE] +
                           [(1.0, 0.0, s) for s in COEF] + [(1.0, 3.0, s) for s in COEF]):
        got = call("npu_cast_bfp16toqdqf32", blk, OUT, I64(n), f32(mul), f32(zp), f32(scale),
                   out_nbytes=4 * n, out_dtype=np.float32)
        try:
            assert_same(got, g.cast_bfp16toqdqf32(blk, mul, zp, scale))
        except AssertionError as e:
            raise AssertionError(f"mul={mul!r} zp={zp!r} scale={scale!r}: {e}")


def test_bad_args():
    """Every entry rejects null pointers (also with no work), negative sizes, sizes off the kernel
    unit and unsupported types before touching memory, and returns 0 without launching for an
    empty output; the non-null pointers here are host addresses."""
    p = Host(np.zeros(64, np.uint8))
    co = A64Coeff(1.0, 0.0, 0.0, 0.0, 1, 0, 0)
    n = I64(256)
    bad = [
        ("npu_convert_fp32_fp16", (None, p, NPU_F16, n)),
        ("npu_convert_fp32_fp16", (p, p, NPU_F32, n)),
        ("npu_convert_fp32_fp16", (p, p, NPU_F16, I64(-1))),
        ("npu_convert_fp16_fp32", (p, NPU_FP8, p, n)),
        ("npu_convert_fp16_fp32", (p, NPU_F16, None, n)),
        ("npu_convert_fp16_bf16", (p, NPU_F16, p, NPU_F16, n)),
        ("npu_convert_fp16_bf16", (p, NPU_BF16, p, NPU_F16, I64(-1))),
        ("npu_cast_bf2bfp16", (p, p, I64(32))),
        ("npu_cast_bf2bfp16", (p, None, n)),
        ("npu_convert_bf16_to_bfp16", (p, NPU_F32, p, n)),
        ("npu_convert_bf16_to_bfp16", (p, NPU_BF16, p, I64(12))),
        ("npu_cast_bfp16tobf", (p, p, I64(8))),
        ("npu_cast_bfp16tobf", (None, p, n)),
        ("npu_cast_bftomx6", (p, p, I64(16))),
        ("npu_cast_bftomx6", (p, p, I64(-64))),
        ("npu_cast_mx6tobf", (p, p, I64(64))),
        ("npu_cast_mx6tobf", (p, None, n)),
        ("npu_convert_fp32_fp8", (None, p, n, 1.0)),
        ("npu_convert_fp32_fp8", (p, p, I64(-8), 1.0)),
        ("npu_convert_fp16_fp8", (p, NPU_F32, p, n, 1.0)),
        ("npu_convert_fp8_fp32", (p, None, n, 1.0)),
        ("npu_convert_fp8_fp16", (p, p, NPU_F32, n, 1.0)),
        ("npu_requantize_fp8", (None, p, n, 1.0)),
        ("npu_requantize_fp8", (p, p, I64(-1), 1.0)),
        ("npu_quant", (p, NPU_I8, p, NPU_U8, n, 0.0, 1.0)),
        ("npu_quant", (p, NPU_F32, p, NPU_I32, n, 0.0, 1.0)),
        ("npu_quant", (p, NPU_F32, None, NPU_U8, n, 0.0, 1.0)),
        ("npu_cast_bfp16toa8", (p, p, I64(8), 1.0, 0.0)),
        ("npu_cast_bfp16toa8", (None, p, n, 1.0, 0.0)),
        ("npu_cast_bfp16toa16", (p, p, I64(72), 1.0, 0.0)),
        ("npu_cast_bfp16toa16", (p, None, n, 1.0, 0.0)),
        ("npu_dequant", (p, NPU_I32, p, NPU_F32, n, 0.0, 1.0)),
        ("npu_dequant", (p, NPU_U8, p, NPU_FP8, n, 0.0, 1.0)),
        ("npu_dequant", (p, NPU_U8, p, NPU_F32, I64(-1), 0.0, 1.0)),
        ("npu_cast_a8tobfp16", (p, p, I64(8), 1.0, 0.0)),
        ("npu_cast_a8tobfp16", (None, p, n, 1.0, 0.0)),
        ("npu_cast_a16tobfp16", (p, p, I64(-64), 1.0, 0.0)),
        ("npu_cast_a16tobfp16", (p, None, n, 1.0, 0.0)),
        ("npu_dequant_w4_bfp16", (p, p, p, p, I64(64), I64(32))),
        ("npu_dequant_w4_bfp16", (p, p, p, p, I64(32), I64(64))),
        ("npu_dequant_w4_bfp16", (p, p, p, p, I64(-64), I64(64))),
        ("npu_dequant_w4_bfp16", (p, p, p, p, I64(64), I64(-64))),
        ("npu_dequant_w4_bfp16", (p, None, p, p, I64(64), I64(64))),
        ("npu_requantize", (p, NPU_I32, p, NPU_U32, n, 1.0, 0.0)),
        ("npu_requantize", (p, NPU_U32, p, NPU_I32, n, 1.0, 0.0)),
        ("npu_requantize", (p, NPU_I64, p, NPU_I8, n, 1.0, 0.0)),
        ("npu_requantize", (p, NPU_I8, p, NPU_F32, n, 1.0, 0.0)),
        ("npu_requantize", (None, NPU_I8, p, NPU_I8, n, 1.0, 0.0)),
        ("npu_requantize_a8_channelwise", (p, NPU_I16, p, NPU_I8, n, I64(32), p, p)),
        ("npu_requantize_a8_channelwise", (p, NPU_I8, p, NPU_I64, n, I64(32), p, p)),
        ("npu_requantize_a8_channelwise", (p, NPU_I8, p, NPU_I8, n, I64(0), p, p)),
        ("npu_requantize_a8_channelwise", (p, NPU_I8, p, NPU_I8, n, I64(2), p, p)),
        ("npu_requantize_a8_channelwise", (p, NPU_I8, p, NPU_I8, n, I64(48), p, p)),
        ("npu_requantize_a8_channelwise", (p, NPU_I8, p, NPU_I8, I64(48), I64(16), p, p)),
        ("npu_requantize_a8_channelwise", (p, NPU_I8, p, NPU_I8, I64(96), I64(64), p, p)),
        ("npu_requantize_a8_channelwise", (p, NPU_I8, p, NPU_I8, n, I64(32), None, p)),
        ("npu_requantize_a16_channelwise", (p, NPU_U8, p, NPU_I8, n, I64(32), p, p)),
        ("npu_requantize_a16_channelwise", (p, NPU_I16, p, NPU_I8, n, I64(65536), p, p)),
        ("npu_requantize_a16_channelwise", (p, NPU_I16, p, NPU_I8, n, I64(32), p, None)),
        ("npu_requantize_a64", (p, NPU_I64, p, NPU_U64, n, co)),
        ("npu_requantize_a64", (p, NPU_I32, p, NPU_U32, n, co)),
        ("npu_requantize_a64", (p, NPU_I64, p, NPU_F32, n, co)),
        ("npu_requantize_a64", (p, NPU_I64, p, NPU_I8, n, None)),
        ("npu_requantize_a64", (p, NPU_I8, None, NPU_I64, n, co)),
        ("npu_cast_int32_int64", (p, NPU_I32, p, NPU_I64, n, 1.0, 0.0)),
        ("npu_cast_int32_int64", (p, NPU_I8, p, NPU_U32, n, 1.0, 0.0)),
        ("npu_cast_int32_int64", (p, NPU_I8, p, NPU_I64, I64(-1), 1.0, 0.0)),
        ("npu_cast_bfp16toqdqf32", (p, p, I64(64), 1.0, 0.0, 1.0)),
        ("npu_cast_bfp16toqdqf32", (None, p, n, 1.0, 0.0, 1.0)),
    ]
    z = I64(0)
    empty = [  # 0 elements: valid, nothing launched
        ("npu_convert_fp32_fp16", (p, p, NPU_BF16, z)),
        ("npu_convert_fp16_fp32", (p, NPU_F16, p, z)),
        ("npu_convert_fp16_bf16", (p, NPU_BF16, p, NPU_F16, z)),
        ("npu_cast_bf2bfp16", (p, p, z)),
        ("npu_convert_bf16_to_bfp16", (p, NPU_BF16, p, z)),
        ("npu_cast_bfp16tobf", (p, p, z)),
        ("npu_cast_bftomx6", (p, p, z)),
        ("npu_cast_mx6tobf", (p, p, z)),
        ("npu_convert_fp32_fp8", (p, p, z, 1.0)),
        ("npu_convert_fp16_fp8", (p, NPU_F16, p, z, 1.0)),
        ("npu_convert_fp8_fp32", (p, p, z, 1.0)),
        ("npu_convert_fp8_fp16", (p, p, NPU_BF16, z, 1.0)),
        ("npu_requantize_fp8", (p, p, z, 1.0)),
        ("npu_quant", (p, NPU_F32, p, NPU_U8, z, 0.0, 1.0)),
        ("npu_cast_bfp16toa8", (p, p, z, 1.0, 0.0)),
        ("npu_cast_bfp16toa16", (p, p, z, 1.0, 0.0)),
        ("npu_dequant", (p, NPU_U8, p, NPU_F32, z, 0.0, 1.0)),
        ("npu_cast_a8tobfp16", (p, p, z, 1.0, 0.0)),
        ("npu_cast_a16tobfp16", (p, p, z, 1.0, 0.0)),
        ("npu_dequant_w4_bfp16", (p, p, p, p, z, I64(64))),
        ("npu_dequant_w4_bfp16", (p, p, p, p, I64(64), z)),
        ("npu_requantize", (p, NPU_I8, p, NPU_I8, z, 1.0, 0.0)),
        ("npu_requantize_a8_channelwise", (p, NPU_I8, p, NPU_I8, z, I64(32), p, p)),
        ("npu_requantize_a16_channelwise", (p, NPU_I16, p, NPU_I8, z, I64(32), p, p)),
        ("npu_requantize_a64", (p, NPU_I64, p, NPU_I8, z, co)),
        ("npu_cast_int32_int64", (p, NPU_I8, p, NPU_I64, z, 1.0, 0.0)),
        ("npu_cast_bfp16toqdqf32", (p, p, z, 1.0, 0.0, 1.0)),
    ]
    # a null required pointer is rejected even with no work; so are bad types / structure
    bad += [(name, tuple(None if a is p and k == 0 else a for k, a in enumerate(args)))
            for name, args in empty]
    bad += [("npu_requantize_a8_channelwise", (p, NPU_I8, p, NPU_I8, z, I64(32), p, None)),
            ("npu_requantize_a8_channelwise", (p, NPU_I8, p, NPU_I8, z, I64(2), p, p)),
            ("npu_requantize_a64", (p, NPU_I64, p, NPU_I8, z, None)),
            ("npu_quant", (p, NPU_I8, p, NPU_U8, z, 0.0, 1.0)),
            ("npu_dequant_w4_bfp16", (p, p, p, p, z, I64(32)))]
    rc = [(name, k, raw_rc(name, *args)) for k, (name, args) in enumerate(bad)]
    assert [r for r in rc if r[2] != -1] == []
    rc = [(name, k, raw_rc(name, *args)) for k, (name, args) in enumerate(empty)]
    assert [r for r in rc if r[2] != 0] == []
    models = {k for k, v in vars(g).items()
              if callable(v) and not k.startswith("_") and v.__module__ == g.__name__}
    assert {name[4:] for name, _ in bad} == models  # one model per entry, named like it
