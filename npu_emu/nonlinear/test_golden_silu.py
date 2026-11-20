# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""SigTanh / SiLU entry points against golden_silu.py, bit for bit."""

import numpy as np
import pytest

from golden_block_fp import encode_bfp16
from golden_float_cvt import bf16_to_f32
from golden_int_round import Rnd
import golden_silu as gs
from npu import (I64, NPU_BF16, NPU_F16, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT, call, f32,
                 raw_rc)
from testing import ALL8, ALL16, INT, int_codes, same
from testing import F16T as HALF

RNG = np.random.default_rng(4242)
F32 = np.float32


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("act", [gs.SIGMOID, gs.TANH])
def test_sigtanh_bf16_all(t, act):
    got = call("npu_sigtanh_bf16", ALL16, OUT, HALF[t], I64(ALL16.size), act,
               out_nbytes=2 * ALL16.size, out_dtype=np.uint16)
    same(got, gs.sigtanh_16(ALL16, t, act), f"sigtanh {t} act {act}")


def test_sigtanh_bf16_zero():
    """sigmoid(+-0) = e(0) = 0.5 + 2^-12 for both zeros."""
    z = np.array([0x0000, 0x8000], np.uint16)
    got = call("npu_sigtanh_bf16", z, OUT, NPU_F16, I64(2), gs.SIGMOID, out_nbytes=4,
               out_dtype=np.uint16)
    assert got.tolist() == [0x3801, 0x3801]


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_silu_bf16_all(t):
    got = call("npu_silu_bf16", ALL16, OUT, HALF[t], I64(ALL16.size),
               out_nbytes=2 * ALL16.size, out_dtype=np.uint16)
    same(got, gs.silu_16(ALL16, t), f"silu {t}")


def test_silu_nan_codes():
    """SiLU(+-inf) and NaN inputs give IEEE NaNs: -|x| makes them negative, a payload is kept."""
    x = np.array([0x7C00, 0xFC00, 0x7E00, 0xFE01], np.uint16)
    got = call("npu_silu_bf16", x, OUT, NPU_F16, I64(4), out_nbytes=8, out_dtype=np.uint16)
    assert got.tolist() == [0xFE00, 0xFE00, 0xFE00, 0xFE01]
    x = np.array([0x7F80, 0xFF80, 0x7FC0, 0xFF81], np.uint16)
    got = call("npu_silu_bf16", x, OUT, NPU_BF16, I64(4), out_nbytes=8, out_dtype=np.uint16)
    assert got.tolist() == [0xFFC0, 0xFFC0, 0xFFC0, 0xFFC1]


def bfp16_blocks(nvec):
    """nvec 64-element vectors (8 blocks each): exponents with a per-vector spread of
    0..40, all-zero blocks, extreme mantissas, exponents 0..3 and 250..255."""
    nb = nvec * 8
    m = RNG.integers(-128, 128, (nb, 8))
    base = RNG.integers(100, 140, nvec)
    spread = RNG.integers(0, 41, nvec)
    e = np.repeat(base, 8) + (RNG.random(nb) * np.repeat(spread, 8)).astype(np.int64)
    k = nb // 8
    e[:k] = RNG.integers(0, 4, k)                      # tiny / subnormal decodes
    e[k:2 * k] = RNG.integers(250, 256, k)             # huge (255: overflow to inf)
    m[2 * k:2 * k + k // 2] = 0                        # all-zero blocks
    m[3 * k:4 * k] = RNG.choice([-128, 127, -127, 64, -64, 96, -96, 1, -1], (k, 8))
    blk = np.concatenate([np.clip(e, 0, 255)[:, None].astype(np.uint8),
                          m.astype(np.int8).view(np.uint8)], axis=1)
    return blk.reshape(-1)


def all_bf16_as_bfp16():
    """Every finite bf16 value as the first value of an otherwise zero block."""
    codes = ALL16[((ALL16 >> 7) & 0xFF) < 0xFF]
    v = bf16_to_f32(codes)
    blk = encode_bfp16(np.repeat(v, 8) * np.tile(np.array([1, 0, 0, 0, 0, 0, 0, 0], F32),
                                                 v.size), Rnd.kHalfEven)
    return blk


@pytest.mark.parametrize("op", ["sigmoid", "tanh", "silu"])
@pytest.mark.parametrize("kind", ["random", "allbf16"])
def test_bfp16(op, kind):
    blk = bfp16_blocks(4096) if kind == "random" else all_bf16_as_bfp16()
    n = blk.size // 9 * 8
    if op == "silu":
        got = call("npu_silu_bfp16", blk, OUT, I64(n), out_nbytes=blk.size,
                   out_dtype=np.uint8)
        want = gs.bfp16_op(blk, True)
    else:
        act = gs.TANH if op == "tanh" else gs.SIGMOID
        got = call("npu_sigtanh_bfp16", blk, OUT, I64(n), act, out_nbytes=blk.size,
                   out_dtype=np.uint8)
        want = gs.bfp16_op(blk, False, act)
    same(got, want, f"{op} bfp16 {kind}")


def test_bfp16_alignment():
    """A block 18+ exponents below its vector maximum loses mantissa bits; 17 below does
    not."""
    blk = np.zeros((16, 9), np.uint8)
    blk[:, 0] = 120
    blk[0, 0] = 140                                      # vector 0 max: 140
    blk[1, 0] = 122                                      # 18 below: 1 bit lost
    blk[2, 0] = 123                                      # 17 below: exact
    blk[1, 1:] = np.array([1, 3, 5, -1, -3, 127, -128, 2], np.int8).view(np.uint8)
    blk[2, 1:] = blk[1, 1:]
    x = gs.bfp16_identity_decode(blk.reshape(-1)).reshape(16, 8)
    assert (x[1] * F32(2.0 ** 11)).tolist() == [0, 4, 4, 0, -4, 128, -128, 2]
    assert (x[2] * F32(2.0 ** 10)).tolist() == [1, 3, 5, -1, -3, 127, -128, 2]


FP8_CODES = np.concatenate([ALL8[(ALL8 & 0x7F) != 0x7F]] * 16)          # 4064, no NaN
FP8_NAN = np.concatenate([ALL8, RNG.permutation(ALL8)])                  # NaN groups


def sigtanh_fp8_coeffs(act, ifm_scale, ofm_scale):
    lit1, lit2 = (0.76842356, -0.12469438) if act == gs.SIGMOID else (1.53878212, -0.49819374)
    return f32(lit1 * ifm_scale), f32(lit2 * ifm_scale * ifm_scale), f32(1.0 / ofm_scale)


@pytest.mark.parametrize("act", [gs.SIGMOID, gs.TANH])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.5, 0.5), (0.3, 0.47058823529411764),
                                    (0.7, 0.002127659574468085), (0.011, 3.3),
                                    (2.7, 0.0037)])
@pytest.mark.parametrize("codes", ["all", "nan"])
def test_sigtanh_fp8(act, scales, codes):
    c = FP8_CODES if codes == "all" else FP8_NAN
    k1, k2, r = sigtanh_fp8_coeffs(act, *scales)
    got = call("npu_elementwise_sigtanh_fp8", c, OUT, I64(c.size), act, k1, k2, r,
               out_nbytes=c.size, out_dtype=np.uint8)
    same(got, gs.sigtanh_fp8(c, act, F32(k1), F32(k2), F32(r)), f"sigtanh fp8 {scales}")


@pytest.mark.parametrize("ct", ["bf16", "f16"])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.5, 0.5), (1.00390625, 0.30117647058823527),
                                    (1.00048828125, 0.6664497233973316),
                                    (0.3, 0.2997658079625293), (3.7, 0.0017),
                                    (150.0, 0.001), (0.011, 3.3)])
@pytest.mark.parametrize("codes", ["all", "nan"])
def test_silu_fp8(ct, scales, codes):
    c = FP8_CODES if codes == "all" else FP8_NAN
    s, r = f32(scales[0]), f32(1.0 / scales[1])
    got = call("npu_elementwise_silu_fp8", c, OUT, I64(c.size), HALF[ct], s, r,
               out_nbytes=c.size, out_dtype=np.uint8)
    same(got, gs.silu_fp8(c, ct, F32(s), F32(r)), f"silu fp8 {ct} {scales}")


def test_silu_fp8_recip_cast():
    """ofm_recip is cast half to even for bf16, half away for f16."""
    one = np.array([0x38], np.uint8)  # fp8 1.0
    for ct, rc, want_r in (("bf16", 1 + 2 ** -8 + 2 ** -9, 1 + 2 ** -7),
                           ("f16", 1.5 + 2 ** -11, 1.5 + 2 ** -10)):
        rnd = Rnd.kHalfEven if ct == "bf16" else Rnd.kHalfAway
        assert gs.narrow(F32(rc), ct, rnd)[()] == F32(want_r)
        got = call("npu_elementwise_silu_fp8", one, OUT, I64(1), HALF[ct], 1.0, f32(rc),
                   out_nbytes=1, out_dtype=np.uint8)
        same(got, gs.silu_fp8(one, ct, F32(1.0), F32(rc)), f"recip {ct}")


# (dq_zp, dq_scale, q_zp, ofm_scale) per input width
QP8 = [(1, 0.0434375, 0, 0.008470021), (128, 0.1034375, 128, 0.008470021),
       (-128, 0.00608058413490653, -128, 0.008470021188259125), (3, 0.3, 7, 1 / 61.0),
       (0, 0.0625, 0, 1 / 256.0), (-17, 0.75, 100, 0.002)]
QP16 = [(1, 0.000183, 0, 0.008470021), (32768, 0.000183, 32768, 0.008470021),
        (22401, 3.76372663595248e-05, 32768, 3.0517578125e-05),
        (-30, 0.00608058413490653, 30, 0.008470021188259125),
        (0, 0.00146484375, 0, 0.0009765625), (30658, 0.0005181971937417984, 994,
                                              0.00028002934413962066),
        (0, 0.01, 0, 1e-5), (-1000, 0.003, 12, 3e-6)]


def qdq_args(t_in, t_out, p):
    zp, s, qzp, os_ = p
    return f32(zp), f32(s), f32(qzp), f32(1.0 / os_)


@pytest.mark.parametrize("a16", [False, True])
@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("ct", ["bf16", "f16"])
@pytest.mark.parametrize("act", [gs.SIGMOID, gs.TANH])
def test_qdq_sigtanh(a16, signed, ct, act):
    t = ("i16" if signed else "u16") if a16 else ("i8" if signed else "u8")
    to = t
    x = int_codes(t)
    for p in (QP16 if a16 else QP8):
        dzp, ds, qzp, qs = qdq_args(t, to, p)
        # arbitrary fp32 coefficients near the fixed ones
        k = (f32(894.00097656), f32(0.76842356 * 1.1 if act == 0 else 1.53878212 * 0.9),
             f32(-0.12469438 * 1.3 if act == 0 else -0.49819374 * 1.05))
        name = "npu_qdq_sigtanh_a16" if a16 else "npu_qdq_sigtanh_a8"
        got = call(name, x, INT[t][0], OUT, INT[to][0], I64(x.size), act, HALF[ct], dzp, ds,
                   qzp, qs, *k, out_nbytes=x.nbytes, out_dtype=INT[to][1])
        want = gs.qdq_sigtanh(x, to, act, ct, F32(dzp), F32(ds), F32(qzp), F32(qs),
                              tuple(F32(v) for v in k))
        same(got, want, f"qdq_sigtanh {t} {ct} {p}")


@pytest.mark.parametrize("a16", [False, True])
@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("ct", ["bf16", "f16"])
def test_qdq_silu(a16, signed, ct):
    t = ("i16" if signed else "u16") if a16 else ("i8" if signed else "u8")
    to = t
    x = int_codes(t)
    for p in (QP16 if a16 else QP8):
        dzp, ds, qzp, qs = qdq_args(t, to, p)
        name = "npu_qdq_silu_exp2_a16" if a16 else "npu_qdq_silu_exp2_a8"
        got = call(name, x, INT[t][0], OUT, INT[to][0], I64(x.size), HALF[ct], dzp, ds, qzp,
                   qs, out_nbytes=x.nbytes, out_dtype=INT[to][1])
        want = gs.qdq_silu(x, to, ct, F32(dzp), F32(ds), F32(qzp), F32(qs))
        same(got, want, f"qdq_silu {t} {ct} {p}")


def test_qdq_mixed_sign():
    """Input and output signedness are independent."""
    x = int_codes("i8")
    got = call("npu_qdq_silu_exp2_a8", x, NPU_I8, OUT, NPU_U8, I64(x.size), NPU_BF16,
               f32(-3), f32(0.05), f32(128), f32(1 / 0.01), out_nbytes=x.size,
               out_dtype=np.uint8)
    same(got, gs.qdq_silu(x, "u8", "bf16", F32(-3), F32(0.05), F32(128), F32(1 / 0.01)))


def test_q_round_half_away():
    """q rounds half away from zero: y = 0.5 / -0.5 / 1.5 / 2.5 with q_scale 1."""
    x = np.array([1, -1, 3, 5, -5], np.int8)
    got = call("npu_qdq_silu_exp2_a8", x, NPU_I8, OUT, NPU_I8, I64(x.size), NPU_BF16, f32(0),
               f32(64.0), f32(0), f32(1.0 / 64), out_nbytes=x.size, out_dtype=np.int8)
    same(got, gs.qdq_silu(x, "i8", "bf16", F32(0), F32(64.0), F32(0), F32(1.0 / 64)))


def test_bad_args():
    assert raw_rc("npu_sigtanh_bf16", None, None, NPU_I8, I64(8), 0) == -1
    assert raw_rc("npu_sigtanh_bf16", None, None, NPU_BF16, I64(8), 2) == -1
    assert raw_rc("npu_silu_bf16", None, None, NPU_I16, I64(8)) == -1
    assert raw_rc("npu_sigtanh_bfp16", None, None, I64(12), 0) == -1
    assert raw_rc("npu_silu_bfp16", None, None, I64(-8)) == -1
    assert raw_rc("npu_elementwise_sigtanh_fp8", None, None, I64(8), 3, 1.0, 1.0, 1.0) == -1
    assert raw_rc("npu_elementwise_silu_fp8", None, None, I64(8), NPU_I8, 1.0, 1.0) == -1
    assert raw_rc("npu_qdq_sigtanh_a8", None, NPU_I16, None, NPU_I8, I64(8), 0, NPU_BF16,
                  0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0) == -1
    assert raw_rc("npu_qdq_sigtanh_a16", None, NPU_I16, None, NPU_I8, I64(8), 0, NPU_BF16,
                  0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0) == -1
    assert raw_rc("npu_qdq_silu_exp2_a8", None, NPU_I8, None, NPU_I8, I64(8), NPU_I8,
                  0.0, 1.0, 0.0, 1.0) == -1
    assert raw_rc("npu_qdq_silu_exp2_a16", None, NPU_U8, None, NPU_U16, I64(8), NPU_F16,
                  0.0, 1.0, 0.0, 1.0) == -1
