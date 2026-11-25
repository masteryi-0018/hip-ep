# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""rope.h entry points against the numpy reference model, bit for bit."""

import numpy as np
import pytest

import golden_rope as r
from golden_block_fp import decode_bfp16, encode_bfp16
from golden_mmac import decode_bfp16_group
from golden_float_cvt import bf16_to_f32, f32_to_bf16, load_float, store_float
from golden_int_round import Rnd
import npu
from npu import I64, NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, OUT
from testing import F16T as T16

RNG = np.random.default_rng(6061)
f32 = np.float32


def assert_same(got, want):
    got = np.asarray(got).reshape(-1)
    want = np.asarray(want).astype(got.dtype).reshape(-1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{got.size} differ, first at {bad[:6].tolist()}: "
                           f"got {got[bad[:6]].tolist()} want {want[bad[:6]].tolist()}")


def tables(M, h, special=True, nan=True):
    """fp32 cos / sin [M][h] of random angles, with exact values and NaN mixed in."""
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
    bits = store_float(v, t)
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
    assert_same(got, r.rope_bf16_stream(bits, t, c, s, rk))


def crafted(level, n=64):
    """Pairs (xn, xr, c, s) whose fused and unfused out_n differ after rounding to
    bf16 ("bf16") or, alone in its block, to bfp16 ("bfp16")."""
    rng = np.random.default_rng(11)
    N = 1 << 22
    xn, xr = (rng.integers(-127, 128, N).astype(f32) * f32(2.0 ** -6) for _ in range(2))
    a = rng.uniform(-np.pi, np.pi, N)
    c, s = np.cos(a).astype(f32), np.sin(a).astype(f32)
    fu = f32_to_bf16(r.rot_pair(xn, xr, c, s, True)[0])
    un = f32_to_bf16(r.rot_pair(xn, xr, c, s, False)[0])
    hit = fu != un
    if level == "bfp16":
        def code(b):
            v = np.zeros((N, 8), f32)
            v[:, 0] = bf16_to_f32(b)
            return encode_bfp16(v.reshape(-1), Rnd.kHalfEven).reshape(N, 9)
        hit &= (code(fu) != code(un)).any(axis=1)
    k = np.nonzero(hit)[0][:n]
    assert k.size >= 8, k.size
    return xn[k], xr[k], c[k], s[k]


def test_rope_bf16_fused_matters():
    xn, xr, cn, sn = crafted("bf16")
    M, K = len(xn), 128
    bits = x16((1, M, K), "bf16", special=False)
    bits[0, :, 0], bits[0, :, 64] = f32_to_bf16(xn), f32_to_bf16(xr)
    c, s = tables(M, 64, special=False)
    c[:, 0], s[:, 0] = cn, sn
    x = load_float(bits, "bf16")
    fu = store_float(r.rope_fp32(x, c, s, K, True), "bf16")
    un = store_float(r.rope_fp32(x, c, s, K, False), "bf16")
    assert (fu != un)[0, :, 0].all()
    got = npu.call("npu_rope_bf16_stream", bits, NPU_BF16, c, s, OUT, I64(1), I64(M), I64(K),
                   0, out_nbytes=bits.nbytes, out_dtype=np.uint16)
    assert_same(got, fu)


SCALES = [(0.125, 0.0625), (0.1, 0.6), (0.6, 0.1), (3.0, 0.01), (0.0123, 7.0), (1.0, 1.0)]


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("si", range(len(SCALES)))
@pytest.mark.parametrize("R,M,K,rot", [(2, 3, 64, 0), (1, 4, 128, 0), (2, 2, 256, 0),
                                       (2, 3, 128, 64), (1, 2, 256, 96)])
def test_rope_fp8_stream(t, si, R, M, K, rot):
    ifm_scale, ofm_scale = (float(f32(v)) for v in SCALES[si])
    rk = rot or K
    codes = RNG.integers(0, 256, (R, M, K)).astype(np.uint8)
    codes[codes == 0x7F] = 0x00
    codes[codes == 0xFF] = 0x80
    flat = codes.reshape(-1)
    for code, frac in ((0x7F, 0.002), (0xFF, 0.002), (0x80, 0.02), (0x01, 0.01), (0x7E, 0.01)):
        idx = RNG.integers(0, flat.size, max(1, int(flat.size * frac)))
        flat[idx] = code
    c, s = tables(M, rk // 2)
    got = npu.call("npu_rope_fp8_stream", codes, c, s, OUT, I64(R), I64(M), I64(K),
                   T16[t], rot,
                   ifm_scale, ofm_scale, out_nbytes=codes.nbytes, out_dtype=np.uint8)
    assert_same(got, r.rope_fp8_stream(codes, t, c, s, rk, ifm_scale, ofm_scale))


def test_rope_fp8_zero_sign():
    codes = np.full((1, 1, 64), 0x80, np.uint8)
    c = np.ones((1, 32), f32)
    s = np.full((1, 32), -0.0, f32)
    got = npu.call("npu_rope_fp8_stream", codes, c, s, OUT, I64(1), I64(1), I64(64),
                   NPU_BF16, 0,
                   1.0, 1.0, out_nbytes=64, out_dtype=np.uint8)
    want = r.rope_fp8_stream(codes, "bf16", c, s, 64, 1.0, 1.0)
    assert_same(got, want)
    assert (want == 0).all()


def bfp16_blocks(n_blocks, special=True):
    """Random bfp16 blocks, with zero blocks, -128 mantissas and extreme exponents."""
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
    c, s = tables(M, K // 2)
    if wt == "bf16":
        cw, sw = f32_to_bf16(c), f32_to_bf16(s)
        cf, sf = bf16_to_f32(cw), bf16_to_f32(sw)
        wtype = NPU_BF16
    else:
        cw, sw, cf, sf, wtype = c, s, c, s, NPU_F32
    got = npu.call("npu_rope_bfp16_stream", blk, cw, sw, wtype, OUT, I64(R), I64(M), I64(K),
                   interleave, 0, out_nbytes=blk.nbytes, out_dtype=np.uint8)
    assert_same(got, r.rope_bfp16_stream(blk, R, M, K, cf, sf, interleave))


@pytest.mark.parametrize("wt", ["bf16", "f32"])
@pytest.mark.parametrize("interleave", [0, 1])
def test_rope_bfp16_group_spans(wt, interleave):
    """Block exponents 0..30 below the largest of their 64-value group, and tiny groups."""
    R, M, K = 1, 4, 256
    nb = R * M * K // 8
    e = (140 - RNG.integers(0, 31, nb)).astype(np.uint8)
    e[RNG.integers(0, nb, nb // 4)] = 141
    e[-8:] = RNG.integers(0, 12, 8)
    m = RNG.integers(-128, 128, (nb, 8)).astype(np.int8)
    blk = np.concatenate([e[:, None], m.view(np.uint8)], axis=1).reshape(-1)
    c, s = tables(M, K // 2)
    if wt == "bf16":
        cw, sw = f32_to_bf16(c), f32_to_bf16(s)
        cf, sf, wtype = bf16_to_f32(cw), bf16_to_f32(sw), NPU_BF16
    else:
        cw, sw, cf, sf, wtype = c, s, c, s, NPU_F32
    want = r.rope_bfp16_stream(blk, R, M, K, cf, sf, interleave)
    got = npu.call("npu_rope_bfp16_stream", blk, cw, sw, wtype, OUT, I64(R), I64(M), I64(K),
                   interleave, 0, out_nbytes=blk.nbytes, out_dtype=np.uint8)
    assert_same(got, want)
    assert (decode_bfp16_group(blk) != decode_bfp16(blk)).any()


@pytest.mark.parametrize("K", [128, 256])
def test_rope_bfp16_path_matters(K):
    xn, xr, cn, sn = crafted("bfp16")
    M = len(xn)
    x = np.zeros((M, K), f32)
    x[:, 0], x[:, K // 2] = xn, xr
    blk = encode_bfp16(x.reshape(-1), Rnd.kHalfEven)
    assert (decode_bfp16(blk) == x.reshape(-1)).all()
    c, s = tables(M, K // 2, special=False)
    c[:, 0], s[:, 0] = cn, sn
    fu = r.rope_bfp16_stream(blk, 1, M, K, c, s, 0, fused=True)
    un = r.rope_bfp16_stream(blk, 1, M, K, c, s, 0, fused=False)
    assert (fu.reshape(M, -1)[:, :9] != un.reshape(M, -1)[:, :9]).any(axis=1).all()
    got = npu.call("npu_rope_bfp16_stream", blk, c, s, NPU_F32, OUT, I64(1), I64(M), I64(K), 0, 0,
                   out_nbytes=blk.nbytes, out_dtype=np.uint8)
    assert_same(got, fu if K > 128 else un)


def test_bad_args():
    z = np.zeros(256, np.uint16)
    rc = npu.raw_rc
    def bf(t, K, rot, *, x=z, rows=1):
        return rc("npu_rope_bf16_stream", x, t, z, z, z, I64(rows),
                  I64(1), I64(K), rot)
    assert bf(NPU_BF16, 64, 0) == 0
    assert bf(NPU_F32, 64, 0) == -1
    assert bf(NPU_BF16, 66, 0) == -1
    # partial RoPE: multiple of 32 below a 128-multiple head_dim only
    assert bf(NPU_BF16, 128, 48) == -1
    assert bf(NPU_BF16, 96, 32) == -1
    assert bf(NPU_BF16, 128, 160) == -1
    assert bf(NPU_BF16, 128, 0, rows=0) == 0
    assert bf(NPU_BF16, 128, 0, rows=-1) == -1
    assert bf(NPU_BF16, 64, 0, x=None) == -1
    assert bf(NPU_BF16, 64, 0, x=None, rows=0) == -1  # pointers are required for empty work too
    assert bf(NPU_F32, 64, 0, rows=0) == -1
    assert rc("npu_rope_bf16_stream", z, NPU_BF16, z, z, z, I64(1), I64(0), I64(64), 0) == 0
    assert rc("npu_rope_bf16_stream", z, NPU_BF16, z, z, z, I64(1), I64(-1), I64(64), 0) == -1
    assert rc("npu_rope_bf16_stream", z, NPU_BF16, z, z, z, I64(1), I64(1), I64(0), 0) == -1
    def f8(t, K, *, x=z, rot=0):
        return rc("npu_rope_fp8_stream", x, z, z, z, I64(1), I64(1), I64(K),
                  t, rot, 1.0, 1.0)
    assert f8(NPU_F16, 64) == 0
    assert f8(NPU_BF16, 96) == -1
    assert f8(NPU_FP8, 64) == -1
    assert f8(NPU_BF16, 128, rot=48) == -1
    assert f8(NPU_BF16, 64, x=None) == -1
    f8e = lambda *, x=z, rows=0, seq=0: rc(  # noqa: E731
        "npu_rope_fp8_stream", x, z, z, z, I64(rows), I64(seq), I64(64), NPU_BF16, 0, 1.0, 1.0)
    assert f8e() == 0
    assert f8e(x=None) == -1
    assert f8e(rows=-1) == -1
    def b16(t, K, il, *, rot=0, c=z):
        return rc("npu_rope_bfp16_stream", z, c, z, t, z, I64(1),
                  I64(1), I64(K), il, rot)
    assert b16(NPU_BF16, 128, 1) == 0
    assert b16(NPU_F16, 128, 0) == -1
    assert b16(NPU_BF16, 64, 0) == -1
    assert b16(NPU_BF16, 128, 2) == -1
    assert b16(NPU_BF16, 128, 0, rot=64) == -1
    assert b16(NPU_BF16, 128, 0, c=None) == -1
    b16e = lambda *, c=z, rows=1, seq=0: rc(  # noqa: E731
        "npu_rope_bfp16_stream", z, c, z, NPU_BF16, z, I64(rows), I64(seq), I64(128), 0, 0)
    assert b16e() == 0
    assert b16e(rows=0, seq=1) == 0
    assert b16e(c=None) == -1
    assert b16e(seq=-1) == -1



# ---- special values, bit for bit (no NaN canonicalisation) ----------------------------

SP16 = {"bf16": [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F,
                 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFBF],
        "f16": [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF,
                0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0xFDFF]}
SP32 = np.array([0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x80000001, 0x00800000,
                 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
                 0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FC12345, 0xFFA00000],
                np.uint32).view(f32)
SP8 = np.array([0x00, 0x80, 0x01, 0x81, 0x38, 0xB8, 0x7E, 0xFE, 0x7F, 0xFF, 0x08, 0x77],
               np.uint8)


def cross(*lists):
    """Flat index arrays of the full cross product of the lists."""
    g = np.meshgrid(*[np.arange(len(v)) for v in lists], indexing="ij")
    return [np.asarray(v)[i.reshape(-1)] for v, i in zip(lists, g)]


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_rope_bf16_special_cross(t):
    """Every combination of special xn, xr, cos, sin (head_dim 4: two pairs per row)."""
    sp16 = np.array(SP16[t], np.uint16)
    xn, xr, c, s = cross(sp16, sp16, SP32[:14], SP32[:14])
    M = len(xn)
    bits = np.zeros((1, M, 4), np.uint16)
    bits[0, :, 0], bits[0, :, 2] = xn, xr
    bits[0, :, 1], bits[0, :, 3] = np.roll(xn, 7), np.roll(xr, 3)
    ct, st = np.zeros((M, 2), f32), np.zeros((M, 2), f32)
    ct[:, 0], st[:, 0] = c, s
    ct[:, 1], st[:, 1] = np.roll(c, 5), np.roll(s, 11)
    got = npu.call("npu_rope_bf16_stream", bits, T16[t], ct, st, OUT, I64(1), I64(M), I64(4), 0,
                   out_nbytes=bits.nbytes, out_dtype=np.uint16)
    assert_same(got, r.rope_bf16_stream(bits, t, ct, st, 4))


@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.0, 0.0), (1e30, 1e-30), (np.inf, np.inf),
                                    (np.nan, 2.0), (1.0, np.nan), (-0.0, -0.0), (1e-45, 1e-45)])
@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_rope_fp8_special_cross(t, scales):
    """Special fp8 codes x special cos / sin x degenerate scales (ofm_scale 0 gives an
    infinite reciprocal)."""
    ifm_scale, ofm_scale = scales
    xn, xr, c, s = cross(SP8, SP8, SP32[:13], SP32[:13])
    n = len(xn)
    M = -(-n // 32)
    pad = M * 32 - n
    xn, xr, c, s = (np.concatenate([a, a[:pad]]) for a in (xn, xr, c, s))
    codes = np.zeros((1, M, 64), np.uint8)
    codes[0, :, :32], codes[0, :, 32:] = xn.reshape(M, 32), xr.reshape(M, 32)
    ct, st = c.reshape(M, 32), s.reshape(M, 32)
    got = npu.call("npu_rope_fp8_stream", codes, ct, st, OUT, I64(1), I64(M), I64(64),
                   T16[t], 0,
                   float(ifm_scale), float(ofm_scale), out_nbytes=codes.nbytes,
                   out_dtype=np.uint8)
    assert_same(got, r.rope_fp8_stream(codes, t, ct, st, 64, ifm_scale, ofm_scale))


@pytest.mark.parametrize("wt", ["bf16", "f32"])
@pytest.mark.parametrize("interleave", [0, 1])
@pytest.mark.parametrize("K", [128, 256])
def test_rope_bfp16_special(wt, interleave, K):
    """Exponent-255 blocks (NaN groups), zero blocks, -128 mantissas, with special cos / sin."""
    R, M = 2, 32
    nb = R * M * K // 8
    e = RNG.integers(120, 135, nb).astype(np.uint8)
    e[RNG.integers(0, nb, nb // 16)] = 255
    e[RNG.integers(0, nb, nb // 16)] = 0
    m = RNG.integers(-128, 128, (nb, 8)).astype(np.int8)
    m[RNG.integers(0, nb, nb // 16)] = 0
    m[RNG.integers(0, nb, nb // 16), RNG.integers(0, 8, nb // 16)] = -128
    blk = np.concatenate([e[:, None], m.view(np.uint8)], axis=1).reshape(-1)
    c = RNG.choice(SP32, (M, K // 2)).astype(f32)
    s = RNG.choice(SP32, (M, K // 2)).astype(f32)
    if wt == "bf16":
        cw, sw = f32_to_bf16(c), f32_to_bf16(s)
        cf, sf, wtype = bf16_to_f32(cw), bf16_to_f32(sw), NPU_BF16
    else:
        cw, sw, cf, sf, wtype = c, s, c, s, NPU_F32
    got = npu.call("npu_rope_bfp16_stream", blk, cw, sw, wtype, OUT, I64(R), I64(M), I64(K),
                   interleave, 0, out_nbytes=blk.nbytes, out_dtype=np.uint8)
    assert_same(got, r.rope_bfp16_stream(blk, R, M, K, cf, sf, interleave))


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("rot", [32, 64, 96])
def test_rope_partial_special(t, rot):
    """Specials in the rotated prefix and in the copied tail of a partial RoPE."""
    R, M, K = 2, 6, 128
    bits = RNG.choice(np.array(SP16[t], np.uint16), (R, M, K))
    c = RNG.choice(SP32, (M, rot // 2)).astype(f32)
    s = RNG.choice(SP32, (M, rot // 2)).astype(f32)
    got = npu.call("npu_rope_bf16_stream", bits, T16[t], c, s, OUT, I64(R), I64(M), I64(K), rot,
                   out_nbytes=bits.nbytes, out_dtype=np.uint16)
    assert_same(got, r.rope_bf16_stream(bits, t, c, s, rot))
    codes = RNG.choice(SP8, (R, M, K))
    got = npu.call("npu_rope_fp8_stream", codes, c, s, OUT, I64(R), I64(M), I64(K),
                   T16[t], rot,
                   0.5, 0.25, out_nbytes=codes.nbytes, out_dtype=np.uint8)
    assert_same(got, r.rope_fp8_stream(codes, t, c, s, rot, 0.5, 0.25))
