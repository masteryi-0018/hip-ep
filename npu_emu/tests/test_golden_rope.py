# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The rope-group entry points (include/npu_emu_rope.h) against the numpy
reference in golden_rope.py, bit for bit. Inputs mix random values with
+-0, NaN, infinity, subnormals and values that overflow f16; cos / sin are
full-precision fp32 (so fused and unfused rotations differ) plus exact values
and NaN; fp8 scales include non powers of two (0.1, 0.6)."""

import numpy as np
import pytest

import golden as g
import golden_rope as r
import npu
from npu import I64, NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, OUT

RNG = np.random.default_rng(6061)
f32 = np.float32
T16 = {"bf16": NPU_BF16, "f16": NPU_F16}


def canon(v, kind):
    """NaN bit patterns folded to one (the sign / payload of a NaN after
    several fp32 operations is not specified, and is unverified on the NPU)."""
    v = np.asarray(v).reshape(-1).copy()
    if kind == "bf16":
        v[(v & 0x7FFF) > 0x7F80] = 0x7FC0
    elif kind == "f16":
        v[(v & 0x7FFF) > 0x7C00] = 0x7E00
    elif kind == "fp8":
        v[(v & 0x7F) == 0x7F] = 0x7F
    return v


def assert_same(got, want, kind=None):
    got = np.asarray(got).reshape(-1)
    want = np.asarray(want).astype(got.dtype).reshape(-1)
    if kind:
        got, want = canon(got, kind), canon(want, kind)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{got.size} differ, first at {bad[:6].tolist()}: "
                           f"got {got[bad[:6]].tolist()} want {want[bad[:6]].tolist()}")


def tables(M, h, special=True, nan=True):
    """fp32 cos / sin [M][h]: cos / sin of random angles (full 24-bit
    mantissas), with exact values (0, +-1, 0.5, -0), 0.1 / 0.6 and NaN mixed in."""
    a = RNG.uniform(-np.pi, np.pi, (M, h))
    c, s = np.cos(a).astype(f32), np.sin(a).astype(f32)
    if special:
        sp = np.array([0.0, -0.0, 1.0, -1.0, 0.5, 0.1, 0.6] + [np.nan] * nan, f32)
        for t in (c, s):
            idx = RNG.integers(0, t.size, max(1, t.size // 64))
            t.reshape(-1)[idx] = RNG.choice(sp, idx.size)
    return c, s


def x16(shape, t, special=True):
    """Random t bits in [-10, 10] (some huge), with specials sprinkled in."""
    v = RNG.uniform(-10, 10, shape).astype(f32)
    v[RNG.random(shape) < 0.02] *= f32(6000.0)  # f16 overflow after the add
    bits = g.store_float(v, t)
    if special:
        sp = {"bf16": [0x0000, 0x8000, 0x7FC0, 0xFFC0, 0x7F80, 0xFF80, 0x0001, 0x8005,
                       0x7F7F, 0x3F80],
              "f16": [0x0000, 0x8000, 0x7E00, 0xFE00, 0x7C00, 0xFC00, 0x0001, 0x8003,
                      0x7BFF, 0x3C00]}[t]
        flat = bits.reshape(-1)
        idx = RNG.integers(0, flat.size, max(1, flat.size // 50))
        flat[idx] = RNG.choice(np.array(sp, np.uint16), idx.size)
    return bits


SHAPES16 = [(3, 5, 64, 0), (2, 4, 96, 0), (1, 3, 128, 0), (2, 2, 256, 0), (1, 2, 4, 0),
            (2, 3, 128, 32), (1, 4, 256, 64), (2, 2, 256, 192), (1, 2, 128, 128)]


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("R,M,K,rot", SHAPES16)
def test_rope_bf16_stream(t, R, M, K, rot):
    rk = rot or K
    bits = x16((R, M, K), t)
    c, s = tables(M, rk // 2)
    got = npu.call("npu_rope_bf16_stream", bits, T16[t], c, s, OUT, I64(R), I64(M), I64(K),
                   rot, out_nbytes=bits.nbytes, out_dtype=np.uint16)
    assert_same(got, r.rope_16(bits, t, c, s, rk), t)


def crafted(level, n=64):
    """Pairs (xn, xr, c, s) for which the fused and the unfused rotation give a
    different out_n after rounding to bf16 ("bf16") or, alone in its block, to
    bfp16 ("bfp16"). xn / xr are int8 * 2^-6, so exact in bf16 and in bfp16;
    c / s are fp32 cos / sin of random angles. Random pairs hit this about
    once in 2^15, too rarely for test-sized tensors."""
    rng = np.random.default_rng(11)
    N = 1 << 22
    xn, xr = (rng.integers(-127, 128, N).astype(f32) * f32(2.0 ** -6) for _ in range(2))
    a = rng.uniform(-np.pi, np.pi, N)
    c, s = np.cos(a).astype(f32), np.sin(a).astype(f32)
    fu = g.to_bf16(r.rot_pair(xn, xr, c, s, True)[0])
    un = g.to_bf16(r.rot_pair(xn, xr, c, s, False)[0])
    hit = fu != un
    if level == "bfp16":
        def code(b):
            v = np.zeros((N, 8), f32)
            v[:, 0] = g.bf16_bits_to_f32(b)
            return g.to_bfp16(v.reshape(-1), g.EVEN).reshape(N, 9)
        hit &= (code(fu) != code(un)).any(axis=1)
    k = np.nonzero(hit)[0][:n]
    assert k.size >= 8, k.size
    return xn[k], xr[k], c[k], s[k]


def test_rope_bf16_fused_matters():
    """bf16 rows whose element 0 tells the fused from the unfused rotation."""
    xn, xr, cn, sn = crafted("bf16")
    M, K = len(xn), 128
    bits = x16((1, M, K), "bf16", special=False)
    bits[0, :, 0], bits[0, :, 64] = g.to_bf16(xn), g.to_bf16(xr)
    c, s = tables(M, 64, special=False)
    c[:, 0], s[:, 0] = cn, sn
    x = g.load_float(bits, "bf16")
    fu = g.store_float(r.rope_fp32(x, c, s, K, True), "bf16")
    un = g.store_float(r.rope_fp32(x, c, s, K, False), "bf16")
    assert (fu != un)[0, :, 0].all()
    got = npu.call("npu_rope_bf16_stream", bits, NPU_BF16, c, s, OUT, I64(1), I64(M), I64(K),
                   0, out_nbytes=bits.nbytes, out_dtype=np.uint16)
    assert_same(got, fu)


# ---- fp8 ----------------------------------------------------------------------

SCALES = [(0.125, 0.0625), (0.1, 0.6), (0.6, 0.1), (3.0, 0.01), (0.0123, 7.0), (1.0, 1.0)]


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("si", range(len(SCALES)))
@pytest.mark.parametrize("R,M,K,rot", [(2, 3, 64, 0), (1, 4, 128, 0), (2, 2, 256, 0),
                                       (2, 3, 128, 64), (1, 2, 256, 96)])
def test_rope_fp8_stream(t, si, R, M, K, rot):
    ifm_scale, ofm_scale = (float(f32(v)) for v in SCALES[si])
    rk = rot or K
    codes = RNG.integers(0, 256, (R, M, K)).astype(np.uint8)
    # NaN codes are rare so most groups of 8 survive; +-0 and subnormals often.
    codes[codes == 0x7F] = 0x00
    codes[codes == 0xFF] = 0x80
    flat = codes.reshape(-1)
    for code, frac in ((0x7F, 0.002), (0xFF, 0.002), (0x80, 0.02), (0x01, 0.01), (0x7E, 0.01)):
        idx = RNG.integers(0, flat.size, max(1, int(flat.size * frac)))
        flat[idx] = code
    c, s = tables(M, rk // 2)
    got = npu.call("npu_rope_fp8_stream", codes, T16[t], c, s, OUT, I64(R), I64(M), I64(K), rot,
                   ifm_scale, ofm_scale, out_nbytes=codes.nbytes, out_dtype=np.uint8)
    assert_same(got, r.rope_fp8(codes, t, c, s, rk, ifm_scale, ofm_scale), "fp8")


def test_rope_fp8_zero_sign():
    """-0 decodes as +0: a row of -0 with cos = 1, sin = -0 rotates to +0."""
    codes = np.full((1, 1, 64), 0x80, np.uint8)
    c = np.ones((1, 32), f32)
    s = np.full((1, 32), -0.0, f32)
    got = npu.call("npu_rope_fp8_stream", codes, NPU_BF16, c, s, OUT, I64(1), I64(1), I64(64), 0,
                   1.0, 1.0, out_nbytes=64, out_dtype=np.uint8)
    want = r.rope_fp8(codes, "bf16", c, s, 64, 1.0, 1.0)
    assert_same(got, want)
    assert (want == 0).all()


# ---- bfp16 --------------------------------------------------------------------

def bfp16_blocks(n_blocks, special=True):
    """Random bfp16 blocks: exponents around 2^0, all mantissas, plus zero
    blocks, -128 mantissas and tiny / huge exponents."""
    e = RNG.integers(122, 140, n_blocks).astype(np.uint8)
    m = RNG.integers(-128, 128, (n_blocks, 8)).astype(np.int8)
    if special:
        k = max(1, n_blocks // 20)
        m[RNG.integers(0, n_blocks, k)] = 0
        m[RNG.integers(0, n_blocks, k), RNG.integers(0, 8, k)] = -128
        e[RNG.integers(0, n_blocks, k)] = RNG.choice(np.array([1, 2, 254], np.uint8), k)
    return np.concatenate([e[:, None], m.view(np.uint8)], axis=1).reshape(-1)


@pytest.mark.parametrize("wt", ["bf16", "f32"])
@pytest.mark.parametrize("interleave", [0, 1])
@pytest.mark.parametrize("R,M,K", [(2, 3, 128), (1, 4, 256), (2, 2, 384)])
def test_rope_bfp16_stream(wt, interleave, R, M, K):
    blk = bfp16_blocks(R * M * K // 8)
    c, s = tables(M, K // 2, nan=False)  # a NaN's sign would decide the bfp16 bytes
    if wt == "bf16":
        cw, sw = g.to_bf16(c), g.to_bf16(s)
        cf, sf = g.bf16_bits_to_f32(cw), g.bf16_bits_to_f32(sw)
        wtype = NPU_BF16
    else:
        cw, sw, cf, sf, wtype = c, s, c, s, NPU_F32
    got = npu.call("npu_rope_bfp16_stream", blk, cw, sw, wtype, OUT, I64(R), I64(M), I64(K),
                   interleave, out_nbytes=blk.nbytes, out_dtype=np.uint8)
    assert_same(got, r.rope_bfp16(blk, R, M, K, cf, sf, interleave))


@pytest.mark.parametrize("K", [128, 256])
def test_rope_bfp16_path_matters(K):
    """With fp32 cos / sin, rows whose element 0 (alone in its block) gives
    different bfp16 bytes fused and unfused: tests the K == 128 / K > 128 split."""
    xn, xr, cn, sn = crafted("bfp16")
    M = len(xn)
    x = np.zeros((M, K), f32)
    x[:, 0], x[:, K // 2] = xn, xr
    blk = g.to_bfp16(x.reshape(-1), g.EVEN)
    assert (g.from_bfp16(blk) == x.reshape(-1)).all()
    c, s = tables(M, K // 2, special=False)
    c[:, 0], s[:, 0] = cn, sn
    fu = r.rope_bfp16(blk, 1, M, K, c, s, 0, fused=True)
    un = r.rope_bfp16(blk, 1, M, K, c, s, 0, fused=False)
    assert (fu.reshape(M, -1)[:, :9] != un.reshape(M, -1)[:, :9]).any(axis=1).all()
    got = npu.call("npu_rope_bfp16_stream", blk, c, s, NPU_F32, OUT, I64(1), I64(M), I64(K), 0,
                   out_nbytes=blk.nbytes, out_dtype=np.uint8)
    assert_same(got, fu if K > 128 else un)


# ---- argument checks -------------------------------------------------------------

def test_rope_bad_args():
    z = None
    rc = npu.raw_rc
    assert rc("npu_rope_bf16_stream", z, NPU_F32, z, z, z, I64(1), I64(1), I64(64), 0) == -1
    assert rc("npu_rope_bf16_stream", z, NPU_BF16, z, z, z, I64(1), I64(1), I64(66), 0) == -1
    # partial RoPE: multiple of 32 below a 128-multiple head_dim only
    assert rc("npu_rope_bf16_stream", z, NPU_BF16, z, z, z, I64(1), I64(1), I64(128), 48) == -1
    assert rc("npu_rope_bf16_stream", z, NPU_BF16, z, z, z, I64(1), I64(1), I64(96), 32) == -1
    assert rc("npu_rope_bf16_stream", z, NPU_BF16, z, z, z, I64(1), I64(1), I64(128), 160) == -1
    assert rc("npu_rope_bf16_stream", z, NPU_BF16, z, z, z, I64(0), I64(1), I64(128), 0) == 0
    assert rc("npu_rope_fp8_stream", z, NPU_BF16, z, z, z, I64(1), I64(1), I64(96), 0,
              1.0, 1.0) == -1
    assert rc("npu_rope_fp8_stream", z, NPU_FP8, z, z, z, I64(1), I64(1), I64(64), 0,
              1.0, 1.0) == -1
    assert rc("npu_rope_bfp16_stream", z, z, z, NPU_F16, z, I64(1), I64(1), I64(128), 0) == -1
    assert rc("npu_rope_bfp16_stream", z, z, z, NPU_BF16, z, I64(1), I64(1), I64(64), 0) == -1
    assert rc("npu_rope_bfp16_stream", z, z, z, NPU_BF16, z, I64(1), I64(1), I64(128), 2) == -1

