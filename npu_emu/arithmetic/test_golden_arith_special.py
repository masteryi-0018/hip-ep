# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Special values (NaN, infinities, signed zeros, subnormals, extreme coefficients, integer
limits) through every arithmetic entry point, bit for bit against the numpy reference."""

import numpy as np
import pytest

import golden_arith_float as gf
import golden_arith_int as gi
import golden_arith_misc as gm
from golden_block_fp import encode_bfp16
from golden_int_round import Rnd
from npu import I64, NPU_BF16, NPU_BFP16, NPU_FP8, OUT, Host, raw_rc
from testing import ALL16, HALF, same, shp
from testing import INT_ALL as T
from testing import call_n as go
from testing import run

RNG = np.random.default_rng(4041)
F32 = np.float32

# fp32 coefficient specials: signed zeros, subnormals, largest finite, infinities, quiet NaNs
# (positive, negative, with payload), 2^24, +-2^31, ordinary values.
SPC = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000,
                0x3F800000, 0xBF800000, 0x40490FDB, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000,
                0xFF800000, 0x7FC00000, 0xFFC00000, 0x7FC12345, 0xFFC54321, 0x4B800000,
                0x4F000000, 0xCF000000, 0x3E4CCCCD, 0x33800000, 0xBF000000],
               np.uint32).view(F32)

NEG_NAN = float(SPC[14])


def coef(special=0.6):
    """A random fp32 coefficient: a special with probability `special`, else an ordinary one."""
    if RNG.random() < special:
        return float(SPC[RNG.integers(len(SPC))])
    return float(F32(RNG.standard_normal() * 10.0 ** RNG.integers(-3, 4)))


def codes16(t):
    """bf16 / f16 codes: zeros, subnormals, extremes, infinities, NaNs (quiet, signalling,
    payload), ordinary values."""
    if t == "f16":
        c = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400, 0x3C00, 0xBC00,
             0x3555, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7E55, 0xFE55, 0x7C01,
             0xFC01, 0x4248, 0x0200]
    else:
        c = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x3F80, 0xBF80,
             0x3EAB, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7FC5, 0xFFC5, 0x7F81,
             0xFF81, 0x4049, 0x0040]
    return np.array(c, np.uint16)


def pick(table, n):
    return np.ascontiguousarray(table[RNG.integers(0, len(table), n)])


FP8_ALL = np.arange(256, dtype=np.uint16).astype(np.uint8)
# Every code, with the NaNs kept out of most rows: each aligned group of 8 holding a NaN is
# decoded as a NaN group.
FP8_NO_NAN = FP8_ALL[(FP8_ALL & 0x7F) != 0x7F]
BCASTS = (0, 1, 2, 4, 7)
ACTS = ((0, 0.0, 0.0), (1, 0.0, 0.0), (1, float("nan"), 0.0), (3, NEG_NAN, 0.0), (1, -0.0, 0.0),
        (2, float("-inf"), float("inf")), (2, 6.0, -6.0), (2, 0.0, float("nan")),
        (2, float("nan"), 1.0), (3, 0.1015625, 0.0), (3, float("inf"), 0.0),
        (3, float("-inf"), 0.0), (3, float("nan"), 0.0), (3, -0.0, 0.0), (3, 1e-40, 0.0))


# ---------------------------------------------------------------- 16-bit float

@pytest.mark.parametrize("t", list(HALF))
@pytest.mark.parametrize("bc", BCASTS)
def test_special_half_binary(t, bc):
    nt, ht = HALF[t]
    sp = codes16(ht)
    H, W, C = 2, 3, 24
    n = H * W * C
    m = int(np.prod(gf._bshape(bc, H, W, C)))
    for _ in range(3):
        x1, x2, x3 = pick(sp, n), pick(sp, m), pick(sp, m)
        sh = ((H, W, C), gf._bshape(bc, H, W, C))
        dims = tuple(shp(x) for x in sh)
        for sub, neg in ((0, 0), (1, 0), (0, 1), (1, 1)):
            for act, a0, a1 in ACTS:
                got = go("npu_elementwise_broadcasting3d_add_bf16", x1, nt, x2, OUT, *dims,
                         sub, neg, act, a0, a1, dt=np.uint16, n=n)
                same(got, gf.elementwise_broadcasting3d_add_bf16(x1, ht, x2, *sh, sub, neg, act,
                                                                 a0, a1),
                     f"add {sub}{neg} act{act} {a0} {a1}")
        for act, a0, a1 in ACTS:
            got = go("npu_elementwise_broadcasting3d_mul_bf16", x1, nt, x2, OUT, *dims, act,
                     a0, a1, dt=np.uint16, n=n)
            same(got, gf.elementwise_broadcasting3d_mul_bf16(x1, ht, x2, *sh, act, a0, a1),
                 f"mul act{act}")
            got = go("npu_elementwise_broadcasting3d_mac_fused_bf16", x1, nt, x2, x3, OUT,
                     *dims, act, a0, a1, dt=np.uint16, n=n)
            same(got, gf.elementwise_broadcasting3d_mac_fused_bf16(x1, ht, x2, x3, *sh, act, a0,
                                                                   a1), f"mac act{act}")
        for sw in (0, 1):
            got = go("npu_elementwise_broadcasting3d_div_bf16", x1, nt, x2, OUT, *dims, sw,
                     dt=np.uint16, n=n)
            same(got, gf.elementwise_broadcasting3d_div_bf16(x1, ht, x2, *sh, sw),
                 f"div swap{sw}")


@pytest.mark.parametrize("t", list(HALF))
def test_special_half_cross(t):
    """Every pair of specials, one pair per element."""
    nt, ht = HALF[t]
    sp = codes16(ht)
    k = len(sp)
    x1, x2 = np.repeat(sp, k), np.tile(sp, k)
    sh = ((1, k, k), (1, k, k))
    dims = tuple(shp(x) for x in sh)
    for sub, neg in ((0, 0), (1, 1)):
        got = go("npu_elementwise_broadcasting3d_add_bf16", x1, nt, x2, OUT, *dims, sub, neg, 0,
                 0.0, 0.0, dt=np.uint16, n=k * k)
        same(got, gf.elementwise_broadcasting3d_add_bf16(x1, ht, x2, *sh, sub, neg, 0, 0.0, 0.0),
             f"add {sub}{neg}")
    got = go("npu_elementwise_broadcasting3d_mul_bf16", x1, nt, x2, OUT, *dims, 0, 0.0, 0.0,
             dt=np.uint16, n=k * k)
    same(got, gf.elementwise_broadcasting3d_mul_bf16(x1, ht, x2, *sh, 0, 0.0, 0.0), "mul")
    for sw in (0, 1):
        got = go("npu_elementwise_broadcasting3d_div_bf16", x1, nt, x2, OUT, *dims, sw,
                 dt=np.uint16, n=k * k)
        same(got, gf.elementwise_broadcasting3d_div_bf16(x1, ht, x2, *sh, sw), f"div swap{sw}")
    x3 = np.tile(sp, k)[::-1].copy()
    got = go("npu_elementwise_broadcasting3d_mac_fused_bf16", x1, nt, x2, x3, OUT, *dims, 0,
             0.0, 0.0, dt=np.uint16, n=k * k)
    same(got, gf.elementwise_broadcasting3d_mac_fused_bf16(x1, ht, x2, x3, *sh, 0, 0.0, 0.0),
         "mac")


# ---------------------------------------------------------------- bfp16

def special_blocks(nblk, zero_rate=0.1):
    """bfp16 blocks with extreme exponents (0, 1, 254, 255), mantissas -128 / 127 / 0 / 1 and
    all-zero blocks."""
    b = np.where(RNG.random((nblk, 9)) < 0.5,
                 RNG.choice(np.array([0, 1, 127, 128, 255, 129], np.uint8), (nblk, 9)),
                 RNG.integers(0, 256, (nblk, 9)).astype(np.uint8)).astype(np.uint8)
    b[:, 0] = RNG.choice(np.array([0, 1, 2, 120, 133, 150, 200, 253, 254, 255], np.uint8), nblk)
    b[RNG.random(nblk) < zero_rate] = 0
    return b.reshape(-1)


@pytest.mark.parametrize("bc", BCASTS)
def test_special_bfp16_binary(bc):
    H, W, C = 2, 3, 64
    nbl = H * W * C // 8
    s = gf._bshape(bc, H, W, C)
    sh = ((H, W, C), s)
    dims = tuple(shp(x) for x in sh)
    for _ in range(4):
        x1 = special_blocks(nbl)
        # a C-broadcast operand is one block per pixel
        x2 = special_blocks(s[0] * s[1] * (1 if bc & 1 else s[2] // 8))
        for op in ("add", "sub", "mul", "div"):
            name = f"elementwise_broadcasting3d_{op}_bfp16"
            for sw in ((0, 1) if op in ("sub", "div") else (0,)):
                extra = (sw,) if op in ("sub", "div") else ()
                got = go(f"npu_{name}", x1, x2, OUT, *dims, *extra, dt=np.uint8, n=nbl * 9)
                same(got, getattr(gf, name)(x1, x2, *sh, *extra), f"{op} swap{sw}")


# npu_dtype / model type of a mul_add operand: 0 bf16, 1 bfp16
BFP = (NPU_BF16, NPU_BFP16)
BFPT = ("bf16", "bfp16")


@pytest.mark.parametrize("bc2,bc3", [(0, 0), (2, 4), (6, 0)])
@pytest.mark.parametrize("act_type,activation", [(0, 0), (100, 100), (7, 0), (7, 7)])
def test_special_mul_add_bfp16(bc2, bc3, act_type, activation):
    H, W, C = 2, 3, 64
    n = H * W * C
    sp = codes16("bf16")
    # bf16 clip bounds: (-2, 1.5), (NaN, 1), (1, -inf), (-inf, inf)
    for lo, hi in ((0xC000, 0x3FC0), (0x7FC0, 0x3F80), (0x3F80, 0xFF80), (0xFF80, 0x7F80)):
        x1 = special_blocks(n // 8)
        m2 = int(np.prod(gf._bshape(bc2, H, W, C)))
        m3 = int(np.prod(gf._bshape(bc3, H, W, C)))
        for b2, b3 in ((0, 0), (1, 1), (0, 1), (1, 0)):
            x2 = special_blocks(m2 // 8) if b2 else pick(sp, m2)
            x3 = special_blocks(m3 // 8) if b3 else pick(sp, m3)
            sh = ((H, W, C), gf._bshape(bc2, H, W, C), gf._bshape(bc3, H, W, C))
            for ob in (0, 1):
                got = go("npu_elementwise_broadcasting3d_mul_add_bfp16", x1, x2, BFP[b2], x3,
                         BFP[b3], OUT, BFP[ob], *(shp(x) for x in sh), act_type, activation, lo,
                         hi, dt=np.uint8, n=n // 8 * 9 if ob else n * 2)
                ref = gf.elementwise_broadcasting3d_mul_add_bfp16(
                    x1, x2, BFPT[b2], x3, BFPT[b3], BFPT[ob], *sh, act_type, activation, lo, hi)
                same(got, ref, f"b{b2}{b3} out{ob} {lo:#x} {hi:#x}")


def test_special_mul_add_tanh_all_bf16():
    """The tanh table on every bf16 code, NaNs and infinities included."""
    H, W, C = 1, 256, 256
    x1 = encode_bfp16(np.ones(H * W * C, F32), Rnd.kHalfEven)
    x3 = np.zeros(H * W * C, np.uint16)
    sh = ((H, W, C),) * 3
    for ob in (0, 1):
        got = go("npu_elementwise_broadcasting3d_mul_add_bfp16", x1, ALL16, NPU_BF16, x3,
                 NPU_BF16, OUT, BFP[ob], *(shp(x) for x in sh), 7, 7, 0, 0,
                 dt=np.uint8, n=65536 // 8 * 9 if ob else 65536 * 2)
        ref = gf.elementwise_broadcasting3d_mul_add_bfp16(x1, ALL16, "bf16", x3, "bf16",
                                                          BFPT[ob], *sh, 7, 7, 0, 0)
        same(got, ref, f"tanh out{ob}")


# ---------------------------------------------------------------- fp8

def fp8_codes(n, nan_rate=0.0):
    c = FP8_NO_NAN[RNG.integers(0, len(FP8_NO_NAN), n)]
    if nan_rate:
        c = np.where(RNG.random(n) < nan_rate, RNG.choice(np.array([0x7F, 0xFF], np.uint8), n),
                     c)
    return np.ascontiguousarray(c.astype(np.uint8))


def fp8_table(c1, c2, act, alpha):
    """add_fp8 table for coefficient specials: per-code biases (NaN entries zeroed) then the
    two LRELU factors."""
    with np.errstate(all="ignore"):
        d = gf.fp8_to_f32(np.arange(256)).astype(np.float64)
        a = 1.0 if act == gf.RELU6 else (1.0 + alpha) * 0.5
        t = (d * c2 * a).astype(F32)
        t = np.where(np.isnan(t), F32(0.0), t)
        tail = np.array([F32(c1 * a), F32(abs(1 - alpha) / 2 / a)], F32)
    return np.concatenate([t, tail])


FP8_COEFFS = (1.0, -1.0, 0.0, -0.0, float("inf"), float("-inf"), float("nan"), NEG_NAN, 1e-40,
              3.0e38, 0.1 / 0.6, 448.0)


@pytest.mark.parametrize("bc", (0, 1, 2, 7))
@pytest.mark.parametrize("nan_rate", [0.0, 0.05])
def test_special_add_fp8(bc, nan_rate):
    H, W, C = 2, 3, 32
    n = H * W * C
    m = int(np.prod(gf._bshape(bc, H, W, C)))
    for _ in range(2):
        x1, x2 = fp8_codes(n, nan_rate), fp8_codes(m, nan_rate)
        for c1, c2 in [(RNG.choice(FP8_COEFFS), RNG.choice(FP8_COEFFS)) for _ in range(6)]:
            c1, c2 = float(F32(c1)), float(F32(c2))
            for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (1, float("nan"), 0.0),
                                (2, float("-inf"), 12.0), (2, 0.0, float("nan")),
                                (3, 0.1015625, 0.0), (3, float("inf"), 0.0)):
                tb = fp8_table(c1, c2, act, a0) if act in (2, 3) else None
                got = go("npu_elementwise_broadcasting3d_add_fp8", x1, x2, OUT,
                         shp((H, W, C)), shp(gf._bshape(bc, H, W, C)), c1, c2, act, a0, a1, tb,
                         dt=np.uint8, n=n)
                same(got, gf.elementwise_broadcasting3d_add_fp8(
                    x1, x2, (H, W, C), gf._bshape(bc, H, W, C), c1, c2, act, a0, a1, tb),
                    f"add c{c1},{c2} act{act} {a0}")


def _fp8_operands(bc, nan_rate):
    H, W, C = 2, 3, 32
    n = H * W * C
    m = int(np.prod(gf._bshape(bc, H, W, C)))
    sh = ((H, W, C), gf._bshape(bc, H, W, C))
    return sh, n, fp8_codes(n, nan_rate), fp8_codes(m, nan_rate), fp8_codes(m, nan_rate)


@pytest.mark.parametrize("bc", (0, 1, 2, 7))
@pytest.mark.parametrize("nan_rate", [0.0, 0.05])
def test_special_mul_fp8(bc, nan_rate):
    for _ in range(2):
        sh, n, x1, x2, _x3 = _fp8_operands(bc, nan_rate)
        for s in FP8_COEFFS:
            s = float(F32(s))
            got = go("npu_elementwise_broadcasting3d_mul_fp8", x1, x2, OUT, *map(shp, sh), s,
                     dt=np.uint8, n=n)
            same(got, gf.elementwise_broadcasting3d_mul_fp8(x1, x2, *sh, s), f"mul {s}")


@pytest.mark.parametrize("bc", (0, 1, 2, 7))
@pytest.mark.parametrize("nan_rate", [0.0, 0.05])
def test_special_div_fp8(bc, nan_rate):
    for _ in range(2):
        sh, n, x1, x2, _x3 = _fp8_operands(bc, nan_rate)
        for s in FP8_COEFFS:
            s = float(F32(s))
            for sw in (0, 1):
                got = go("npu_elementwise_broadcasting3d_div_fp8", x1, x2, OUT, *map(shp, sh), s,
                         sw, dt=np.uint8, n=n)
                same(got, gf.elementwise_broadcasting3d_div_fp8(x1, x2, *sh, s, sw),
                     f"div {s} swap{sw}")


@pytest.mark.parametrize("bc", (0, 1, 2, 7))
@pytest.mark.parametrize("nan_rate", [0.0, 0.05])
def test_special_mac_fp8(bc, nan_rate):
    for _ in range(2):
        sh, n, x1, x2, x3 = _fp8_operands(bc, nan_rate)
        for c0, c4 in [(float(F32(RNG.choice(FP8_COEFFS))), float(F32(RNG.choice(FP8_COEFFS))))
                       for _ in range(6)]:
            for act, a0, a1 in ((0, 0.0, 0.0), (1, 0.0, 0.0), (1, float("nan"), 0.0),
                                (2, float("-inf"), 24.0), (3, 0.1015625, 0.0),
                                (3, float("inf"), 0.0)):
                got = go("npu_elementwise_broadcasting3d_mac_fused_fp8", x1, x2, x3, OUT,
                         *map(shp, sh), c0, c4, act, a0, a1, dt=np.uint8, n=n)
                same(got, gf.elementwise_broadcasting3d_mac_fused_fp8(x1, x2, x3, *sh, c0, c4, act,
                                                                      a0, a1),
                     f"mac {c0},{c4} act{act}")


def test_special_div_fp8_all_pairs():
    """Every (num, den) code pair outside NaN rows, both swaps."""
    k = len(FP8_NO_NAN)
    x1, x2 = np.repeat(FP8_NO_NAN, k), np.tile(FP8_NO_NAN, k)
    for sw in (0, 1):
        got = go("npu_elementwise_broadcasting3d_div_fp8", x1, x2, OUT, shp((1, k, k)),
                 shp((1, k, k)), 1.0, sw, dt=np.uint8, n=k * k)
        same(got, gf.elementwise_broadcasting3d_div_fp8(x1, x2, (1, k, k), (1, k, k), 1.0, sw),
             f"div swap{sw}")


# ---------------------------------------------------------------- unary / batchnorm

@pytest.mark.parametrize("t", list(HALF))
def test_special_unary_half(t):
    nt, ht = HALF[t]
    for alpha in [coef(1.0) for _ in range(12)] + [0.0, -0.0, float("inf"), float("-inf"),
                                                   float("nan"), 3.0e38, 1e-40]:
        got = go("npu_elementwise_leakyrelu_bf16", ALL16, nt, OUT, I64(65536), alpha,
                 dt=np.uint16, n=65536)
        same(got, gf.elementwise_leakyrelu_bf16(ALL16, ht, 65536, alpha), f"leakyrelu {alpha}")
    C = 64
    for _ in range(3):
        D = np.array([coef(0.5) for _ in range(C)], F32)
        got = go("npu_elementwise_prelu_bf16", ALL16, nt, OUT, I64(65536), I64(C), D,
                 dt=np.uint16, n=65536)
        same(got, gf.elementwise_prelu_bf16(ALL16, ht, 65536, C, D), "prelu")
    got = go("npu_elementwise_square_bf16", ALL16, nt, OUT, I64(65536), dt=np.uint16, n=65536)
    same(got, gf.elementwise_square_bf16(ALL16, ht, 65536), "square")


@pytest.mark.parametrize("nan_rows", [False, True])
def test_special_unary_fp8(nan_rows):
    # all codes in every row position, including 0xFF
    x = np.concatenate([(np.arange(256) + k) % 256 for k in range(8)]).astype(np.uint8)
    if not nan_rows:
        x = np.where((x & 0x7F) == 0x7F, np.uint8(0x00), x).astype(np.uint8)
    n = x.size
    for _ in range(8):
        pos, neg = coef(0.7), coef(0.7)
        got = go("npu_elementwise_leakyrelu_fp8", x, OUT, I64(n), pos, neg, dt=np.uint8, n=n)
        same(got, gf.elementwise_leakyrelu_fp8(x, n, F32(pos), F32(neg)), f"leakyrelu {pos} {neg}")
        s = coef(0.7)
        got = go("npu_elementwise_square_fp8", x, OUT, I64(n), s, dt=np.uint8, n=n)
        same(got, gf.elementwise_square_fp8(x, n, F32(s)), f"square {s}")
        P, C = coef(0.5), 64
        D = np.array([coef(0.5) for _ in range(C)], F32)
        got = go("npu_elementwise_prelu_fp8", x, OUT, I64(n), I64(C), P, D, dt=np.uint8, n=n)
        same(got, gf.elementwise_prelu_fp8(x, n, C, F32(P), D), f"prelu {P}")


@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("act", [0, 1, 3])
def test_special_batchnorm(t, act):
    C = 64
    if t == "fp8":
        x = np.concatenate([FP8_ALL, fp8_codes(1024, 0.02)])
        nt, wt = NPU_FP8, "f16"
    else:
        nt, wt = HALF[t]
        x = np.concatenate([ALL16, pick(codes16(wt), 4096)])
    n = x.size - x.size % C
    x = np.ascontiguousarray(x[:n])
    for _ in range(4):
        cw, cb = pick(codes16(wt), C), pick(codes16(wt), C)
        alpha = float(F32(RNG.choice([0.1015625, 0.0, -0.0, np.inf, -np.inf, np.nan, 1e-40])))
        dt = np.uint8 if t == "fp8" else np.uint16
        got = go("npu_batchnorm2d_stream_bf16", x, nt, OUT, I64(n), I64(C), cw, cb, act, alpha,
                 dt=dt, n=n)
        same(got, gf.batchnorm2d_stream_bf16(x, t if t == "fp8" else wt, n, C, cw, cb, act,
                                             F32(alpha)), f"bn {t} act{act} {alpha}")


# ---------------------------------------------------------------- integer, helpers

BITS = {8: ["i8", "u8"], 16: ["i16", "u16"]}


def edge_ints(t, n):
    """Values of the integer type t, the limits and their neighbours over-represented."""
    info = np.iinfo(T[t][1])
    x = RNG.integers(info.min, info.max, n, endpoint=True)
    edge = np.array([info.min, info.min + 1, info.max - 1, info.max, 0, 1, -1, 127, 128, -128])
    edge = edge[(edge >= info.min) & (edge <= info.max)]
    m = RNG.random(n) < 0.3
    x[m] = RNG.choice(edge, m.sum())
    return x.astype(T[t][1])


I32_EDGE = [-2 ** 31, 2 ** 31 - 1, 2 ** 24 + 1, -2 ** 24 - 1]


def zp_wide(t):
    """A zero point of the type's limits, or of int32 limits where the entry allows it."""
    return int(RNG.choice(I32_EDGE)) if RNG.random() < 0.5 else zp_edge(t)


def zp_edge(t):
    info = np.iinfo(T[t][1])
    return int(RNG.choice([info.min, info.max, 0, 1, info.min + 1, info.max - 1]))


def words(vals, ints=()):
    w = np.zeros(len(vals), np.uint32)
    for k, v in enumerate(vals):
        w[k] = np.int64(v) & 0xFFFFFFFF if k in ints else np.array([v], F32).view(np.uint32)[0]
    return Host(w)


TRIALS = 24


# ---------------------------------------------------------------- integer, flat

@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("op", ["add", "mul"])
def test_special_flat(bits, op):
    n = 2051
    for _ in range(TRIALS):
        it, ot = RNG.choice(BITS[bits]), RNG.choice(BITS[bits])
        a, b = edge_ints(it, n), edge_ints(it, n)
        c = [coef() for _ in range(4 if op == "mul" else 3)]
        name = f"elementwise_{op}_a{bits}"
        got = run(f"npu_{name}", a, T[it][0], b, OUT, T[ot][0], I64(n), *c, n=n, t=ot)
        assert (got == getattr(gi, name)(a, it, b, ot, n, *c)).all(), f"{op} {c}"
        for act, a0, a1 in ((gi.ACT_RELU6, coef(), coef()), (gi.ACT_RELU6, -3.5, 4.5),
                            (gi.ACT_RELU, coef(), coef()), (gi.ACT_LRELU, coef(), coef())):
            names = (f"elementwise_{op}_fused_a{bits}",) + (
                (f"elementwise_sub_fused_a{bits}",) if op == "add" else ())
            for name in names:
                got = run(f"npu_{name}", a, T[it][0], b, OUT, T[ot][0], I64(n), *c, act, a0, a1,
                          n=n, t=ot)
                want = getattr(gi, name)(a, it, b, ot, n, *c, act, a0, a1)
                assert (got == want).all(), f"{name} {c} {act} {a0} {a1}"


def test_special_flat_xint_add():
    n = 2051
    for _ in range(TRIALS):
        a, b = edge_ints("i8", n), edge_ints("i8", n)
        c = [coef() for _ in range(3)]
        got = run("npu_elementwise_add_a8_xint", a, b, OUT, I64(n), *c, n=n, t="i8")
        assert (got == gi.elementwise_add_a8_xint(a, b, n, *c)).all(), c


# Exponents of c0, c1 for the integer xint add: shifts up to 253 bits and beyond.
XINT_E = [(0, 0), (1, -1), (-126, -126), (127, 127), (127, -126), (-126, 127), (63, 0), (0, 63),
          (64, 0), (0, 64), (70, 70), (-100, -3), (30, 30), (62, -2), (100, 100), (-1, -127 + 1)]


@pytest.mark.parametrize("e", XINT_E)
@pytest.mark.parametrize("act", [gi.ACT_NONE, gi.ACT_RELU, gi.ACT_RELU6])
def test_special_xint_add_fused(e, act):
    n = 4096
    a, b = edge_ints("i8", n), edge_ints("i8", n)
    c0, c1 = float(F32(2.0 ** e[0])), float(F32(2.0 ** e[1]))
    for a0, a1 in ((-20.5, 90.5), (float("nan"), 5.0), (float("-inf"), float("inf")),
                   (1e30, -1e30), (-0.0, 0.0)) if act else ((0.0, 0.0),):
        got = run("npu_elementwise_add_fused_a8_xint", a, b, OUT, I64(n), c0, c1, act, a0, a1,
                  n=n, t="i8")
        want = gi.elementwise_add_fused_a8_xint(a, b, n, c0, c1, act, a0, a1)
        assert (got == want).all(), (e, act, a0, a1)


@pytest.mark.parametrize("types", [("i8", "i8", "i8", "i8"), ("u8", "u8", "i8", "u8"),
                                   ("i8", "u8", "u8", "i8")])
@pytest.mark.parametrize("shape", [((3, 5, 32),) * 3, ((3, 5, 32), (1, 1, 1), (3, 5, 32)),
                                   ((3, 5, 32), (1, 1, 32), (3, 5, 1))])
def test_special_xint_mac(types, shape):
    s1, s2, s3 = shape
    t1, t2, t3, to = types
    i32 = [-2 ** 31, 2 ** 31 - 1, 0, 1, -1, 32767, -32768, 65535, 100000]
    for _ in range(TRIALS):
        x1, x2, x3 = edge_ints(t1, s1), edge_ints(t2, s2), edge_ints(t3, s3)
        A, v = int(RNG.choice(i32)), int(RNG.choice(i32))
        p, t = int(RNG.choice([0, 1, 15, 30, 31])), int(RNG.choice([0, 1, 7, 15, 30, 31]))
        got = run("npu_elementwise_mac_a8", x1, T[t1][0], x2, T[t2][0], x3, T[t3][0], OUT,
                  T[to][0], shp(s1), shp(s2), shp(s3), A, v, p, t, n=x1.size, t=to)
        want = gi.elementwise_mac_a8(x1, t1, x2, t2, x3, t3, to, s1, s2, s3, A, v, p, t)
        assert (got == want.reshape(-1)).all(), (A, v, p, t)


# ---------------------------------------------------------------- integer, 3D broadcasting

SH = [((3, 5, 40), (3, 5, 40)), ((3, 5, 40), (3, 5, 1)), ((3, 5, 40), (1, 1, 40)),
      ((3, 5, 40), (1, 1, 1))]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("shape", range(len(SH)))
@pytest.mark.parametrize("op", ["add", "mul"])
def test_special_bc3d(bits, shape, op):
    s1, s2 = SH[shape]
    name = f"qdq_elementwise_broadcasting3d_{op}_a{bits}"
    for _ in range(TRIALS // 2):
        t = RNG.choice(BITS[bits])
        a, b = edge_ints(t, s1), edge_ints(t, s2)
        c = [coef() for _ in range(4 if op == "mul" else 3)]
        got = run(f"npu_{name}", a, T[t][0], b, OUT, shp(s1), shp(s2), *c, n=a.size, t=t)
        assert (got == getattr(gi, name)(a, t, b, s1, s2, *c).reshape(-1)).all(), c


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("shape", range(len(SH)))
@pytest.mark.parametrize("op", ["add", "sub"])
@pytest.mark.parametrize("act", [gi.ACT_NONE, gi.ACT_RELU, gi.ACT_RELU6, gi.ACT_LRELU,
                                 gi.ACT_PRELU])
def test_special_bc3d_addsub_fused(bits, shape, op, act):
    s1, s2 = SH[shape]
    for _ in range(TRIALS // 3):
        t = RNG.choice(BITS[bits])
        a, b = edge_ints(t, s1), edge_ints(t, s2)
        k = [coef() for _ in range(8)]
        slope = np.array([coef(0.5) for _ in range(s1[2])], F32) if act == gi.ACT_PRELU else None
        name = f"qdq_elementwise_broadcasting3d_{op}_fused_a{bits}"
        want = getattr(gi, name)(a, t, b, s1, s2, k, act, slope)
        got = run(f"npu_{name}", a, T[t][0], b, OUT, shp(s1), shp(s2), words(k), act, slope,
                  n=a.size, t=t)
        assert (got == want.reshape(-1)).all(), (k, act)


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("shape", range(len(SH)))
@pytest.mark.parametrize("act", [gi.ACT_NONE, gi.ACT_RELU, gi.ACT_RELU6, gi.ACT_LRELU,
                                 gi.ACT_PRELU])
def test_special_bc3d_mul_fused(bits, shape, act):
    s1, s2 = SH[shape]
    for _ in range(TRIALS // 3):
        t = RNG.choice(BITS[bits])
        a, b = edge_ints(t, s1), edge_ints(t, s2)
        k = [coef() for _ in range(6)] + [zp_edge(t), coef(), zp_edge(t)]
        ns = np.array([coef(0.5) for _ in range(s1[2])], F32) if act == gi.ACT_PRELU else None
        name = f"qdq_elementwise_broadcasting3d_mul_fused_a{bits}"
        want = getattr(gi, name)(a, t, b, s1, s2, k, act, ns)
        got = run(f"npu_{name}", a, T[t][0], b, OUT, shp(s1), shp(s2), words(k, ints=(6, 8)), act,
                  ns, n=a.size, t=t)
        assert (got == want.reshape(-1)).all(), (k, act)


MAC_VARIANTS = {"a8w8": (8, 8, True), "a16w8": (16, 8, False), "a16w16": (16, 16, True)}
# (shape1, shape23)
MACF_SHAPES = [((3, 5, 40), (3, 5, 1)), ((3, 5, 40), (3, 5, 40)), ((3, 5, 40), (1, 1, 1)),
               ((3, 5, 40), (1, 1, 40))]


@pytest.mark.parametrize("variant", list(MAC_VARIANTS))
@pytest.mark.parametrize("ct", ["same", "i32"])
@pytest.mark.parametrize("shape", range(len(MACF_SHAPES)))
@pytest.mark.parametrize("act", [gi.ACT_NONE, gi.ACT_RELU, gi.ACT_RELU6, gi.ACT_LRELU])
def test_special_bc3d_mac_fused(variant, ct, shape, act):
    abits, wbits, _ = MAC_VARIANTS[variant]
    s1, s23 = MACF_SHAPES[shape]
    name = f"qdq_elementwise_broadcasting3d_mac_fused_{variant}"
    for _ in range(TRIALS // 3):
        ta, tw = RNG.choice(BITS[abits]), RNG.choice(BITS[wbits])
        tc = RNG.choice(BITS[abits]) if ct == "same" else "i32"
        a, b, c = edge_ints(ta, s1), edge_ints(tw, s23), edge_ints(tc, s23)
        k = [coef() for _ in range(7)] + [zp_wide(ta)]
        want = getattr(gi, name)(a, ta, b, tw, c, tc, s1, s23, k, act)
        got = run(f"npu_{name}", a, T[ta][0], b, T[tw][0], c, T[tc][0], OUT, shp(s1), shp(s23),
                  words(k, ints=(7,)), act, n=a.size, t=ta)
        assert (got == want.reshape(-1)).all(), (k, act)


# ---------------------------------------------------------------- integer, misc

def pairs(bits):
    return [(a, b) for a in BITS[bits] for b in BITS[bits]]


@pytest.mark.parametrize("bits", [8, 16])
def test_special_div(bits):
    n = 4099
    for _ in range(TRIALS):
        it, ot = pairs(bits)[RNG.integers(4)]
        a, b = edge_ints(it, n), edge_ints(it, n)
        c0, c1 = coef(), coef()
        z1, z2 = zp_edge(it), zp_edge(it)
        w1 = float(F32(12582912.0 + z1)) if RNG.random() < 0.7 else coef(1.0)
        w2 = float(F32(12582912.0 + z2)) if RNG.random() < 0.7 else coef(1.0)
        name = f"elementwise_div_a{bits}"
        got = run(f"npu_{name}", a, T[it][0], b, OUT, T[ot][0], I64(n), c0, c1, w1, w2, n=n, t=ot)
        assert (got == getattr(gm, name)(a, it, b, ot, n, c0, c1, w1, w2)).all(), (c0, c1, w1, w2)


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("bs", [(3, 5, 40), (1, 1, 40), (3, 5, 1), (1, 1, 1)])
def test_special_bdiv(bits, bs):
    for _ in range(TRIALS // 2):
        t = RNG.choice(BITS[bits])
        a, b = edge_ints(t, (3, 5, 40)), edge_ints(t, bs)
        c0, c1 = coef(), coef()
        z1, z2 = zp_wide(t), (zp_wide(t) if bits == 8 else zp_edge(t))
        sh = int(RNG.choice([0, 15, 64, -64, 1, -1, 31, 33]))
        name = f"qdq_elementwise_broadcasting3d_div_a{bits}"
        got = run(f"npu_{name}", a, T[t][0], b, OUT, shp(a.shape), shp(bs), c0, c1, z1, z2, sh,
                  n=a.size, t=t)
        want = getattr(gm, name)(a, t, b, a.shape, bs, c0, c1, z1, z2, sh)
        assert (got == want.reshape(-1)).all(), (c0, c1, z1, z2, sh)


@pytest.mark.parametrize("bits", [8, 16])
def test_special_leakyrelu_square(bits):
    n = 4099
    for _ in range(TRIALS):
        it, ot = pairs(bits)[RNG.integers(4)]
        x = edge_ints(it, n)
        c = [coef() for _ in range(4)]
        got = run(f"npu_elementwise_leakyrelu_a{bits}", x, T[it][0], OUT, T[ot][0], I64(n), *c,
                  n=n, t=ot)
        assert (got == getattr(gm, f"elementwise_leakyrelu_a{bits}")(x, it, ot, n, *c)).all(), c
        got = run(f"npu_elementwise_square_a{bits}", x, T[it][0], OUT, T[ot][0], I64(n), *c[:3],
                  n=n, t=ot)
        want = getattr(gm, f"elementwise_square_a{bits}")(x, it, ot, n, *c[:3])
        assert (got == want).all(), c[:3]


@pytest.mark.parametrize("bits", [8, 16])
def test_special_prelu(bits):
    C, rows = 40, 101
    for _ in range(TRIALS):
        it, ot = pairs(bits)[RNG.integers(4)]
        x = edge_ints(it, (rows, C))
        pos, zo = coef(), coef()
        neg = np.array([coef(0.5) for _ in range(C)], F32)
        zi = zp_wide(it)
        got = run(f"npu_elementwise_prelu_a{bits}", x, T[it][0], OUT, T[ot][0], I64(rows),
                  I64(C), pos, zi, zo, neg, n=x.size, t=ot)
        want = getattr(gm, f"elementwise_prelu_a{bits}")(x, it, ot, rows, C, pos, zi, zo, neg)
        assert (got == want.reshape(-1)).all(), (pos, zi, zo)


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("act", [0, 1, 3])
def test_special_qdq_batchnorm(bits, act):
    C, rows = 40, 101
    for _ in range(TRIALS):
        t = RNG.choice(BITS[bits])
        x = edge_ints(t, (rows, C))
        cw = np.array([coef(0.5) for _ in range(C)], F32)
        cb = np.array([coef(0.5) for _ in range(C)], F32)
        zo, alpha = coef(), coef()
        name = f"qdq_batchnorm2d_stream_a{bits}"
        got = run(f"npu_{name}", x, T[t][0], OUT, I64(rows), I64(C), cw, cb, zo, act, alpha,
                  n=x.size, t=t)
        want = getattr(gm, name)(x, t, rows, C, cw, cb, zo, act, alpha)
        assert (got == want.reshape(-1)).all(), (zo, alpha)


@pytest.mark.parametrize("act", [gi.ACT_NONE, gi.ACT_RELU, gi.ACT_RELU6, gi.ACT_LRELU,
                                 gi.ACT_PRELU])
@pytest.mark.parametrize("bits", [8, 16])
def test_special_bc3d_mul_fused_extreme_zero_points(bits, act):
    """Zero points of int32 extremes: the residuals and their product stay exact."""
    ext = [-2 ** 31, 2 ** 31 - 1, 2 ** 20, -2 ** 20, 65535, -65536]
    for s1, s2 in SH:
        t = BITS[bits][0]
        a, b = edge_ints(t, s1), edge_ints(t, s2)
        for _ in range(4):
            k = [coef(0.2) for _ in range(6)] + [int(RNG.choice(ext)), coef(0.2),
                                                  int(RNG.choice(ext))]
            ns = np.array([coef(0.3) for _ in range(s1[2])], F32) if act == gi.ACT_PRELU else None
            name = f"qdq_elementwise_broadcasting3d_mul_fused_a{bits}"
            want = getattr(gi, name)(a, t, b, s1, s2, k, act, ns)
            got = run(f"npu_{name}", a, T[t][0], b, OUT, shp(s1), shp(s2), words(k, ints=(6, 8)),
                      act, ns, n=a.size, t=t)
            assert (got == want.reshape(-1)).all(), (k, act)


def test_bad_args():
    """Every entry of the three headers rejects its invalid argument sets."""
    from test_golden_arith_float import BAD_ARGS as bad_float
    from test_golden_arith_int import BAD_ARGS as bad_int
    from test_golden_arith_misc import BAD_ARGS as bad_misc
    for args in bad_float + bad_int + bad_misc:
        assert raw_rc(*args) == -1, args
    p, i16, u16 = np.zeros(64, np.uint8), T["i16"][0], T["u16"][0]
    # negative sizes; null pointers even for empty tensors; empty tensors are a no-op
    assert raw_rc("npu_elementwise_square_bf16", p, NPU_BF16, p, I64(-8)) == -1
    assert raw_rc("npu_elementwise_square_bf16", None, NPU_BF16, p, I64(0)) == -1
    assert raw_rc("npu_elementwise_square_bf16", p, NPU_BF16, p, I64(0)) == 0
    assert raw_rc("npu_elementwise_mul_a16", p, i16, p, p, u16, I64(-1), 1.0, 0.0, 0.0, 0.0) == -1
    assert raw_rc("npu_elementwise_mul_a16", p, i16, p, None, u16, I64(0), 1.0, 0.0, 0.0, 0.0) == -1
    assert raw_rc("npu_elementwise_mul_a16", p, i16, p, p, u16, I64(0), 1.0, 0.0, 0.0, 0.0) == 0
