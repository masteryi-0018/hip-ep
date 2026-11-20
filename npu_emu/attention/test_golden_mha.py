# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""npu_mha_bf16_3strms against golden_mha.py, bit for bit."""

import numpy as np
import pytest

import golden_mha as gm
from golden_float_cvt import bf16_to_f32, f32_to_bf16, kFtzIn
from golden_int_round import Rnd
from npu import I64, OUT, U32, Host, call, raw_rc

RNG = np.random.default_rng(15194)
D = 64


def bf16(x):
    """float32 -> bf16 bits, round half to even."""
    return f32_to_bf16(np.asarray(x, np.float32), Rnd.kHalfEven, kFtzIn)


def rand(shape, lo=-1.0, hi=1.0):
    return bf16(RNG.uniform(lo, hi, shape))


def spread(shape, octaves):
    v = RNG.standard_normal(shape) * np.exp2(RNG.integers(-octaves, octaves + 1, shape))
    return bf16(v)


def f32w(x):
    return int(np.array([x], np.float32).view(np.uint32)[0])


DEFAULT_SCALE = f32w(1 / np.sqrt(D))


def run(q, k, v, sv_n, causal, start, scale=DEFAULT_SCALE):
    B, H, M, _ = q.shape
    Hk, N = k.shape[1], k.shape[2]
    got = call("npu_mha_bf16_3strms", q, k, v, OUT, I64(B), I64(H), I64(Hk), I64(M), I64(D),
               I64(N), I64(sv_n), causal, U32(start), U32(scale),
               out_nbytes=B * M * H * D * 2, out_dtype=np.uint16).reshape(B, M, H, D)
    want = gm.mha(q, k, v, sv_n, scale, causal, start)
    return got, want


def assert_same(got, want, rows=None):
    got, want = np.asarray(got), np.asarray(want)
    if rows is not None:
        got, want = got[:, rows], want[:, rows]
    got, want = got.reshape(-1), want.reshape(-1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{got.size} differ, first at {bad[:6].tolist()}: "
                           f"got {[hex(x) for x in got[bad[:6]]]} want {[hex(x) for x in want[bad[:6]]]}")


def qkv(B, H, Hk, M, N, gen=rand):
    return gen((B, H, M, D)), gen((B, Hk, N, D)), gen((B, Hk, N, D))


# ---- random, both mask modes -------------------------------------------------------

@pytest.mark.parametrize("sv_n,N", [(256, 256), (128, 512), (192, 384), (128, 128), (256, 1024)])
@pytest.mark.parametrize("causal", [0, 1])
def test_random(sv_n, N, causal):
    start = N if not causal else int(RNG.integers(0, N))
    got, want = run(*qkv(1, 1, 1, 64, N), sv_n, causal, start)
    assert_same(got, want)


@pytest.mark.parametrize("start", [1, 31, 100, 127, 128, 129, 255, 256, 300, 511, 512, 1000,
                                   65536 + 300, 70000])
def test_mask_1d(start):
    """mask_M_start inside, on and past block boundaries, and past 65535."""
    got, want = run(*qkv(1, 1, 1, 64, 512), 128, 0, start)
    assert_same(got, want)


@pytest.mark.parametrize("start", [0, 5, 63, 64, 200, 448, 1000, 32767, 65535 - 40, 65536 + 7])
def test_causal(start):
    """Causal with a row offset; the row id wraps at 65536."""
    got, want = run(*qkv(1, 1, 1, 128, 512), 128, 1, start)
    assert_same(got, want)


@pytest.mark.parametrize("B,H,Hk,M", [(2, 1, 1, 64), (1, 4, 2, 64), (2, 2, 1, 128), (1, 3, 3, 192)])
@pytest.mark.parametrize("causal", [0, 1])
def test_batch_heads(B, H, Hk, M, causal):
    got, want = run(*qkv(B, H, Hk, M, 256), 128, causal, 200)
    assert_same(got, want)


# ---- constructed -------------------------------------------------------------------

def test_wide_spread():
    """Scores over many octaves: most P underflow."""
    got, want = run(*qkv(1, 1, 1, 64, 512, lambda s: spread(s, 6)), 128, 0, 512)
    assert_same(got, want)


@pytest.mark.parametrize("scale", [f32w(1.0), f32w(8.0), f32w(1e-3), f32w(0.0), f32w(-0.125),
                                   f32w(1e-38), f32w(3e38)])
def test_scales(scale):
    got, want = run(*qkv(1, 1, 1, 64, 256), 128, 0, 200, scale)
    assert_same(got, want)


def test_ties_and_zero():
    """All-zero score rows, exact ties of nonzero scores, V = 0 columns."""
    q, k, v = qkv(1, 1, 1, 64, 256)
    q[0, 0, :16] = 0
    q[0, 0, 16:32] = 0x8000  # -0
    k[0, 0, 128:] = k[0, 0, :128]
    v[0, 0, :, :8] = 0
    got, want = run(q, k, v, 128, 0, 256)
    assert_same(got, want)


def test_negative_scores_and_cancel():
    """Large +- products that cancel in one MMAC half; scores of both signs."""
    q = np.tile(bf16([256.0, -256.0, 1.0, 2.0**-7] * 16), (1, 1, 64, 1)).astype(np.uint16)
    q[0, 0, :, 2] = rand((64,))
    k, v = rand((1, 1, 256, D)), rand((1, 1, 256, D), -4, 4)
    got, want = run(q, k, v, 128, 1, 100)
    assert_same(got, want)


def test_one_hot_v():
    """V = identity blocks: O reads out P column by column."""
    q, k, _ = qkv(1, 1, 1, 64, 256)
    v = np.zeros((1, 1, 256, D), np.uint16)
    for j in range(256):
        v[0, 0, j, j % D] = 0x3F80
    got, want = run(q, k, v, 256, 0, 256)
    assert_same(got, want)


def test_big_values():
    """|Q|, |K| near 16: scores of hundreds, P mostly 0, l close to 1."""
    got, want = run(*qkv(1, 1, 1, 64, 256, lambda s: rand(s, -16, 16)), 128, 0, 256)
    assert_same(got, want)


def test_jump():
    """The max rises a lot in a later block: mx_exp underflows, O and l restart."""
    q, k, v = qkv(1, 1, 1, 64, 512)
    k[0, 0, 384:] = bf16(bf16_to_f32(k[0, 0, 384:]) * 8)
    q = bf16(bf16_to_f32(q) * 4)
    got, want = run(q, k, v, 128, 0, 512)
    assert_same(got, want)


# ---- argument checks -------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["M", "head", "sv64", "sv512", "N", "group", "mask", "null"])
def test_rejects(bad):
    x = Host(np.zeros(16, np.uint16))
    p = dict(B=1, H=2, Hk=1, M=64, Dh=64, N=256, sv=128, mask=0)
    p.update({"M": dict(M=96), "head": dict(Dh=128), "sv64": dict(sv=64), "sv512": dict(sv=512),
              "N": dict(N=320), "group": dict(H=3, Hk=2), "mask": dict(mask=2), "null": {}}[bad])
    ptr = None if bad == "null" else x
    rc = raw_rc("npu_mha_bf16_3strms", ptr, x, x, x, I64(p["B"]), I64(p["H"]), I64(p["Hk"]),
                I64(p["M"]), I64(p["Dh"]), I64(p["N"]), I64(p["sv"]), p["mask"], U32(0), U32(0))
    assert rc == -1


# ---- special values, bit for bit -------------------------------------------------------

SPB = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F,
                0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFBF, 0x4B00, 0xCB00], np.uint16)


@pytest.mark.parametrize("causal", [0, 1])
@pytest.mark.parametrize("rate", [0.002, 0.05])
def test_special_inputs(causal, rate):
    """NaN, inf, subnormal and signed-zero entries in Q, K, V."""
    q, k, v = qkv(1, 2, 1, 64, 256)
    for x in (q, k, v):
        m = RNG.random(x.shape) < rate
        x[m] = RNG.choice(SPB, int(m.sum()))
    got, want = run(q, k, v, 128, causal, 100)
    assert_same(got, want)


@pytest.mark.parametrize("scale", [f32w(np.inf), f32w(-np.inf), f32w(np.nan), f32w(-0.0),
                                   0x7FA00001, 0xFFC12345, 0x00000001])
def test_special_scales(scale):
    got, want = run(*qkv(1, 1, 1, 64, 256), 128, 0, 200, scale)
    assert_same(got, want)


@pytest.mark.parametrize("start", [0, 1, 128, 65535])
@pytest.mark.parametrize("causal", [0, 1])
def test_masked_rows(start, causal):
    """mask_M_start 0 masks every column (all -inf scores)."""
    got, want = run(*qkv(1, 1, 1, 64, 384), 128, causal, start)
    assert_same(got, want)
