# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The C models of common/ against their numpy mirrors, bit for bit: exhaustive 16-bit codes,
structured fp32 samples (every sign / exponent field with rounding-boundary mantissas,
specials, random), never an exhaustive fp32 sweep."""

import numpy as np
import pytest

import golden_block_fp as gb
import golden_float_cvt as gc
import golden_int_round as gi
import golden_mac as gm
import golden_minmax as gx
import golden_mmac as gmm
import golden_act as ga
import golden_nlf as m
from golden_int_round import Rnd
from npu import I64, OUT, call

f32 = np.float32
RNG = np.random.default_rng(5150)
ROWS = 100_000
OP_INV, OP_INVSQRT, OP_LN = 0, 1, 2


# ---- helpers ----

def bits(x):
    return np.asarray(x, f32).view(np.uint32).reshape(-1)


def assert_same(got, want, x=None):
    got, want = bits(got), bits(want)
    assert got.size == want.size
    bad = np.nonzero(got != want)[0]
    msg = f"{bad.size}/{got.size} differ, first at {bad[:4].tolist()}: "
    msg += f"got {[hex(v) for v in got[bad[:4]]]} want {[hex(v) for v in want[bad[:4]]]}"
    if x is not None and bad.size:
        msg += f" x {[hex(v) for v in bits(x)[bad[:4]]]}"
    assert bad.size == 0, msg


def fbits(b):
    return np.asarray(b, np.uint32).view(f32)


def gpu_vaddr(lanes, scalar=None):
    lanes = np.ascontiguousarray(lanes, f32)
    rows, n = lanes.shape
    s = None if scalar is None else np.ascontiguousarray(scalar, f32)
    return call("npu_test_vaddr_f32_reduce", lanes, n, s, OUT, I64(rows),
                out_nbytes=rows * 4, out_dtype=f32)


def gpu_unary(op, x):
    x = np.ascontiguousarray(x, f32).reshape(-1)
    return call("npu_test_nlf_unary", op, x, OUT, I64(x.size), out_nbytes=x.size * 4,
                out_dtype=f32)


def make(e, mant=None, neg=None, shape=None):
    """fp32 from biased exponent / mantissa / sign arrays (random where None)."""
    e = np.asarray(e, np.int64)
    shape = shape or e.shape
    e = np.broadcast_to(e, shape)
    mant = RNG.integers(0, 1 << 23, shape) if mant is None else np.broadcast_to(mant, shape)
    neg = RNG.integers(0, 2, shape) if neg is None else np.broadcast_to(neg, shape)
    return fbits((neg.astype(np.int64) << 31) | (e << 23) | mant)


# ---- reduce-add ----

def lanes_random(r):
    return make(RNG.integers(100, 155, (r, 32)))


def lanes_cancel(r):
    """+-2^20 (and +-2^20 * (1 + k ulp)) that cancel, around small values."""
    x = make(RNG.integers(127 - 30, 127 + 12, (r, 32)))
    big = make(127 + 20, mant=RNG.integers(0, 4, (r, 1)) * RNG.integers(0, 2, (r, 1)),
               neg=np.zeros((r, 1), np.int64), shape=(r, 1))
    for k in range(r):
        p = RNG.permutation(32)[:4]
        x[k, p[0]] = big[k, 0]
        x[k, p[1]] = -big[k, 0]
        x[k, p[2]] = f32(2.0 ** 20) * (1 if RNG.integers(2) else -1)
        if RNG.integers(2):
            x[k, p[3]] = -x[k, p[2]]
    return x


def lanes_span(span):
    def f(r):
        top = RNG.integers(span + 1, 255, (r, 1))
        return make(top - RNG.integers(0, span + 1, (r, 32)))
    return f


def lanes_subnormal(r):
    return make(np.zeros((r, 32), np.int64))


def lanes_min_normal(r):
    """Exponent fields 0..3 with cancellation: sums land around 2^-126."""
    x = make(RNG.integers(0, 4, (r, 32)))
    x[: r // 2, 16:] = -x[: r // 2, :16] * (1 + RNG.integers(0, 2, (r // 2, 16)) * f32(2.0 ** -22))
    return x


def lanes_zeros(r):
    x = make(RNG.integers(1, 255, (r, 32)))
    z = RNG.random((r, 32))
    x[z < 0.45] = f32(0.0)
    x[(z >= 0.45) & (z < 0.9)] = f32(-0.0)
    x[: r // 4] = f32(0.0)
    x[r // 4: r // 2] = f32(-0.0)
    x[r // 2: r // 2 + 1000, 5] = fbits(1)       # one subnormal among zeros
    return x


def lanes_special(r):
    x = lanes_random(r)
    sp = fbits(np.array([0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFFFFFFF,
                         0x7FFFFFFF], np.uint32))
    for n_sp in (1, 2):
        pick = RNG.random((r, 32)) < (0.02 * n_sp)
        x[pick] = sp[RNG.integers(0, sp.size, pick.sum())]
    return x


def lanes_large(r):
    return make(RNG.integers(230, 255, (r, 32)),
                neg=(RNG.random((r, 32)) < 0.2).astype(np.int64))


CLASSES = {"random": lanes_random, "cancel": lanes_cancel, "span30": lanes_span(30),
           "span40": lanes_span(40), "span120": lanes_span(120), "subnormal": lanes_subnormal,
           "min_normal": lanes_min_normal, "zeros": lanes_zeros, "special": lanes_special,
           "large": lanes_large}


def scalar_like(lanes):
    """A scalar per row: a lane value, its negation, a tiny / huge value, +-0,
    or a special value."""
    r = lanes.shape[0]
    pick = RNG.integers(0, 32, r)
    s = lanes[np.arange(r), pick].copy()
    k = RNG.integers(0, 7, r)
    s[k == 1] = -s[k == 1]
    s[k == 2] = make(RNG.integers(0, 40, (k == 2).sum()))
    s[k == 3] = make(RNG.integers(160, 255, (k == 3).sum()))
    s[k == 4] = f32(0.0)
    s[k == 5] = f32(-0.0)
    s[k == 6] = fbits(np.array([0x7F800000, 0xFF800000, 0x7FC00001], np.uint32))[
        RNG.integers(0, 3, (k == 6).sum())]
    return s


@pytest.mark.parametrize("with_scalar", [False, True])
@pytest.mark.parametrize("cls", list(CLASSES))
def test_vaddr(cls, with_scalar):
    lanes = CLASSES[cls](ROWS)
    s = scalar_like(lanes) if with_scalar else None
    got = gpu_vaddr(lanes, s)
    assert_same(got, gm.vaddr_f32_reduce(lanes, s))


@pytest.mark.parametrize("n", [1, 7, 16, 31])
def test_vaddr_short_vectors(n):
    """Fewer lanes = the 32-lane op padded with +0."""
    lanes = np.concatenate([lanes_cancel(20000), lanes_special(20000), lanes_zeros(20000)])
    s = scalar_like(lanes)
    pad = np.zeros_like(lanes)
    pad[:, :n] = lanes[:, :n]
    assert_same(gpu_vaddr(lanes[:, :n], s), gm.vaddr_f32_reduce(pad, s))


# ---- reciprocal ----

INV_EXPS = [1, 2, 63, 126, 127, 128, 190, 252, 253, 254]
ALL_M = np.arange(1 << 23, dtype=np.int64)


def all_mantissas(e):
    return np.concatenate([make(e, ALL_M, 0, ALL_M.shape), make(e, ALL_M, 1, ALL_M.shape)])


def specials():
    v = np.array([0, 0x80000000, 1, 0x807FFFFF, 0x7F800000, 0xFF800000, 0x7F800001,
                  0x7FC00000, 0xFFFFFFFF, 0x7F7FFFFF, 0xFF7FFFFF, 0x00800000, 0x80800000,
                  0x3F800000, 0xBF800000, 0x7E800000, 0x7E800001, 0x7EFFFFFF], np.uint32)
    return np.concatenate([fbits(v), make(0, shape=(4096,)), make(255, shape=(4096,))])


@pytest.mark.parametrize("e", INV_EXPS)
def test_inv_all_mantissas(e):
    x = all_mantissas(e)
    got = gpu_unary(OP_INV, x)
    assert_same(got, m.hw_inv_f32(x), x)


def test_inv_special_and_random():
    x = np.concatenate([specials(), fbits(RNG.integers(0, 1 << 32, 1 << 22, dtype=np.uint64)
                                          .astype(np.uint32))])
    got = gpu_unary(OP_INV, x)
    assert_same(got, m.hw_inv_f32(x), x)


# ---- inverse square root ----

@pytest.mark.parametrize("e", [1, 2, 3, 4, 100, 101, 127, 128, 253, 254])
def test_invsqrt_all_mantissas(e):
    """e = 127 / 128 together are the whole [1, 4) domain of the table."""
    x = all_mantissas(e)
    got = gpu_unary(OP_INVSQRT, x)
    assert_same(got, m.hw_invsqrt_f32(x), x)


def half_inputs():
    h = np.arange(1 << 16, dtype=np.uint32)
    return np.concatenate([gc.bf16_to_f32(h), gc.f16_to_f32(h)])


def test_invsqrt_half_special_random():
    x = np.concatenate([half_inputs(), specials(),
                        fbits(RNG.integers(0, 1 << 32, 1 << 22, dtype=np.uint64).astype(np.uint32))])
    got = gpu_unary(OP_INVSQRT, x)
    assert_same(got, m.hw_invsqrt_f32(x), x)


# ---- approx_ln ----

@pytest.mark.parametrize("e", [0, 1, 126, 127, 128, 254, 255])
def test_approx_ln_all_mantissas(e):
    x = all_mantissas(e)
    got = gpu_unary(OP_LN, x)
    assert_same(got, m.hw_approx_ln(x), x)


def test_approx_ln_half_random():
    x = np.concatenate([half_inputs(), specials(),
                        fbits(RNG.integers(0, 1 << 32, 1 << 22, dtype=np.uint64).astype(np.uint32))])
    got = gpu_unary(OP_LN, x)
    assert_same(got, m.hw_approx_ln(x), x)


def test_bad_args():
    from npu import raw_rc
    assert raw_rc("npu_test_vaddr_f32_reduce", None, 33, None, None, I64(1)) == -1
    assert raw_rc("npu_test_vaddr_f32_reduce", None, 0, None, None, I64(1)) == -1
    assert raw_rc("npu_test_nlf_unary", 12, None, None, I64(1)) == -1


# ==== int_round / float_cvt / mac / minmax / block_fp / mmac / nlf additions ====

T = {"f32": 0, "f16": 1, "bf16": 2, "fp8": 3, "i8": 4, "u8": 5, "i16": 6, "u16": 7,
     "i32": 8, "u32": 9, "i64": 10, "u64": 11}
ALL_RND = list(Rnd)
kFtzIn, kFtzOut, kSat = gc.kFtzIn, gc.kFtzOut, gc.kSat
ALL16 = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
U64 = np.uint64


def same(got, want, x=None):
    """Bitwise equality of two arrays of the same item size."""
    got, want = np.ascontiguousarray(got).reshape(-1), np.ascontiguousarray(want).reshape(-1)
    assert got.size == want.size, f"size {got.size} != {want.size}"
    u = f"u{got.itemsize}"
    gb_, wb = got.view(u), want.astype(got.dtype).view(u)
    bad = np.nonzero(gb_ != wb)[0]
    msg = (f"{bad.size}/{got.size} differ, first at {bad[:4].tolist()}: got "
           f"{[hex(v) for v in gb_[bad[:4]]]} want {[hex(v) for v in wb[bad[:4]]]}")
    if x is not None and bad.size:
        xs = np.ascontiguousarray(x).reshape(-1)
        msg += f" x {[hex(v) for v in xs.view(f'u{xs.itemsize}')[bad[:4]]]}"
    assert bad.size == 0, msg


def gpu(name, *args, n, dt):
    return call(name, *args, out_nbytes=n * np.dtype(dt).itemsize, out_dtype=dt)


MANTS = np.array([0, 1, 2, 3, 0x7F, 0x80, 0x81, 0xFFF, 0x1000, 0x1001, 0x1FFF, 0x2000, 0x3000,
                  0x7FFF, 0x8000, 0x8001, 0xFFFF, 0x10000, 0x17FFF, 0x18000, 0x18001, 0x7FFFF,
                  0x80000, 0x80001, 0xFFFFF, 0x100000, 0x17FFFF, 0x180000, 0x3FFFFF, 0x400000,
                  0x400001, 0x7E0000, 0x7F0000, 0x7F8000, 0x7FC000, 0x7FE000, 0x7FF000, 0x7FFFFF],
                 np.int64)


def f32_struct(n_random=1 << 19, seed=1):
    """Every sign / exponent field with boundary mantissas (ties and carries of bf16, f16 and
    fp8 rounding) and random ones, plus random fp32 bit patterns."""
    rng = np.random.default_rng(seed)
    se = np.arange(512, dtype=np.int64)[:, None]
    mant = np.concatenate([np.broadcast_to(MANTS, (512, MANTS.size)),
                           rng.integers(0, 1 << 23, (512, 26))], axis=1)
    b = ((se >> 8) << 31) | ((se & 255) << 23) | mant
    rnd = rng.integers(0, 1 << 32, n_random, dtype=np.uint64)
    return np.concatenate([b.reshape(-1).astype(np.uint32), rnd.astype(np.uint32)]).view(f32)


F32S = f32_struct()


def int_struct(rng, n):
    """int64 values with ties around every shift: k * 2^s + {0, +-1, 2^(s-1), ...}."""
    s = rng.integers(1, 62, n)
    k = rng.integers(-(1 << 20), 1 << 20, n)
    off = np.choose(rng.integers(0, 5, n), [np.zeros(n, np.int64), np.ones(n, np.int64),
                                            -np.ones(n, np.int64), np.int64(1) << (s - 1),
                                            (np.int64(1) << (s - 1)) + 1])
    v = (k << np.minimum(s, 40)) + off
    big = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, n // 4, dtype=np.int64)
    ext = np.array([0, 1, -1, 2**62, -2**62, 2**63 - 1, -2**63, 2**31, -2**31, 2**32 - 1],
                   np.int64)
    return np.concatenate([v, big, ext])


# ---- int_round ----

@pytest.mark.parametrize("rnd", ALL_RND)
def test_shift_round(rnd):
    rng = np.random.default_rng(int(rnd) + 11)
    v = int_struct(rng, 200_000)
    s = rng.integers(-8, 70, v.size).astype(np.int32)
    s[: v.size // 2] = rng.integers(1, 42, v.size // 2)
    got = gpu("npu_test_int_round", 0, v, s, int(rnd), 0, OUT, I64(v.size), n=v.size, dt=np.int64)
    same(got, gi.shift_round(v, s, rnd), v)
    for neg in (0, 1):
        got = gpu("npu_test_int_round", 1, v, s, int(rnd), neg, OUT, I64(v.size), n=v.size,
                  dt=np.int64)
        same(got, gi.shift_round_mag(v.astype(U64), s, rnd, bool(neg)).astype(np.int64), v)


@pytest.mark.parametrize("t", ["i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"])
def test_saturate_wrap(t):
    v = int_struct(np.random.default_rng(3), 100_000)
    s = np.zeros(v.size, np.int32)
    for op, fn in ((2, gi.saturate), (3, gi.wrap), (4, gi.saturate_sym)):
        got = gpu("npu_test_int_round", op, v, s, 0, T[t], OUT, I64(v.size), n=v.size,
                  dt=np.int64)
        same(got, fn(v, t), v)


def to_int_inputs():
    k = np.arange(-70000, 70000, dtype=np.float64)
    ties = np.concatenate([k + 0.5, k, k + 0.25, k + 0.75]).astype(f32)
    edges = np.array([2**31 - 0.5, 2**31, -2**31 - 0.5, 2**32 - 0.5, 2**32, 1e30, -1e30,
                      np.inf, -np.inf, np.nan], f32)
    return np.concatenate([F32S[:300_000], ties, edges])


@pytest.mark.parametrize("rnd", ALL_RND)
@pytest.mark.parametrize("t,shift", [("i8", 0), ("u16", 0), ("i32", 0), ("u32", 0), ("i16", -9),
                                     ("i16", 7)])
def test_float_to_int(rnd, t, shift):
    x = to_int_inputs()
    got = gpu("npu_test_float_to_int", 0, x, T[t], shift, int(rnd), OUT, I64(x.size), n=x.size,
              dt=np.int64)
    same(got, gc.float_to_int(x, t, shift, rnd), x)


@pytest.mark.parametrize("t", ["i8", "u8", "i16", "u16", "i32", "u32"])
def test_requant_tail(t):
    x = to_int_inputs()
    got = gpu("npu_test_float_to_int", 1, x, T[t], 0, 0, OUT, I64(x.size), n=x.size, dt=np.int64)
    same(got, gq.requant_tail(x, t), x)


@pytest.mark.parametrize("rnd", ALL_RND)
@pytest.mark.parametrize("e", [0, -30, -149, -170, 60, 100, 104, 110])
def test_int_to_float(rnd, e):
    rng = np.random.default_rng(int(rnd) * 100 + e + 1000)
    p = np.int64(1) << np.arange(63, dtype=np.int64)
    v = np.concatenate([p, p + 1, p - 1, -p, -p - 1, -p + 1, int_struct(rng, 100_000),
                        rng.integers(-(1 << 31), 1 << 31, 100_000)])
    got = gpu("npu_test_int_to_float", v, e, int(rnd), OUT, I64(v.size), n=v.size, dt=f32)
    same(got, gc.int_to_float(v, e, rnd), v)


# ---- float_cvt ----

def gpu_encode(x, t, rnd, flags):
    return gpu("npu_test_encode_float", x, T[t], int(rnd), flags, OUT, I64(x.size), n=x.size,
               dt=np.uint32)


@pytest.mark.parametrize("rnd", ALL_RND)
@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
def test_encode_float(t, rnd):
    same(gpu_encode(F32S, t, rnd, 0), gc.encode_float(F32S, t, rnd, 0), F32S)


@pytest.mark.parametrize("flags", [kFtzIn, kFtzOut, kSat, kFtzIn | kFtzOut | kSat])
@pytest.mark.parametrize("rnd", [Rnd.kHalfEven, Rnd.kHalfAway, Rnd.kTrunc, Rnd.kFloor])
@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
def test_encode_float_flags(t, rnd, flags):
    same(gpu_encode(F32S, t, rnd, flags), gc.encode_float(F32S, t, rnd, flags), F32S)


@pytest.mark.parametrize("flags", [0, kFtzIn])
@pytest.mark.parametrize("t", ["bf16", "f16", "fp8", "f32"])
def test_decode_float(t, flags):
    c = {"fp8": np.arange(256, dtype=np.uint32), "f32": F32S.view(np.uint32)}.get(
        t, ALL16.astype(np.uint32))
    got = gpu("npu_test_decode_float", c, T[t], flags, OUT, I64(c.size), n=c.size, dt=f32)
    same(got, gc.decode_float(c, t, flags), c)


@pytest.mark.parametrize("flags", [0, kFtzIn, kFtzOut])
@pytest.mark.parametrize("t,rnd", [("f32", Rnd.kHalfEven), ("bf16", Rnd.kHalfAway),
                                   ("f16", Rnd.kTrunc), ("fp8", Rnd.kHalfEven)])
def test_round_float(t, rnd, flags):
    got = gpu("npu_test_float_misc", 0, F32S, T[t], int(rnd), flags, OUT, I64(F32S.size),
              n=F32S.size, dt=f32)
    same(got, gc.round_float(F32S, t, rnd, flags), F32S)


@pytest.mark.parametrize("rnd", [Rnd.kHalfEven, Rnd.kHalfAway])
@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_scalar_cast(t, rnd):
    got = gpu("npu_test_float_misc", 1, F32S, T[t], int(rnd), 0, OUT, I64(F32S.size),
              n=F32S.size, dt=f32)
    same(got, gc.scalar_cast(F32S, t, rnd), F32S)


@pytest.mark.parametrize("n", [1, 13, 64, 1003])
def test_fp8_decode(n):
    rng = np.random.default_rng(n)
    c = np.concatenate([rng.integers(0, 256, n).astype(np.uint8) for _ in range(50)])
    c[rng.random(c.size) < 0.02] = 0x7F
    c[rng.random(c.size) < 0.05] = 0x80
    got = gpu("npu_test_fp8_decode", c, OUT, I64(c.size), n=c.size, dt=f32)
    same(got, gmm.fp8_decode_identity(c), c)


# ---- mac ----

def triples(n, seed):
    rng = np.random.default_rng(seed)
    pick = [F32S[rng.integers(0, F32S.size, n)] for _ in range(3)]
    sub = rng.random((3, n)) < 0.1  # extra subnormals and zeros
    for k in range(3):
        pick[k] = np.where(sub[k], (rng.integers(0, 1 << 23, n) | (rng.integers(0, 2, n) << 31))
                           .astype(np.uint32).view(f32), pick[k])
    # products near the addend: cancellation
    pick[2][: n // 4] = -(pick[0][: n // 4] * pick[1][: n // 4])
    return pick


@pytest.mark.parametrize("flags", [0, kFtzIn, kFtzOut, kFtzIn | kFtzOut])
def test_fma_ftz(flags):
    a, b, c = triples(1_000_000, 7 + flags)
    with np.errstate(all="ignore"):
        got = gpu("npu_test_mac", 0, a, b, c, None, flags, OUT, I64(a.size), n=a.size, dt=f32)
        same(got, gm.fma_ftz(a, b, c, flags), a)


@pytest.mark.parametrize("n", [8, 11, 16, 0])
def test_fma_nan(n):
    """NaN operands (signs, payloads) against finite, zero and infinite ones, in arrays of
    several lengths (numpy's own NaN choice depends on the length)."""
    v = np.array([0x7FC00000, 0xFFC00000, 0x7FC00123, 0xFFC00456, 0x7F800001, 0xFF800001,
                  0x3FC00000, 0xC0000000, 0, 0x80000000, 0x7F800000, 0xFF800000], np.uint32)
    a, b, c = (x.reshape(-1).view(f32) for x in np.meshgrid(v, v, v, indexing="ij"))
    for s in range(0, a.size, n or a.size):
        x, y, z = (np.ascontiguousarray(t[s:s + (n or a.size)]) for t in (a, b, c))
        got = gpu("npu_test_mac", 0, x, y, z, None, 0, OUT, I64(x.size), n=x.size, dt=f32)
        same(got, gm.fma_q(x, y, z), x)


@pytest.mark.parametrize("flags", [0, kFtzIn | kFtzOut])
def test_mac_exact(flags):
    rng = np.random.default_rng(17 + flags)
    _, b, c = triples(500_000, 9)
    x = rng.integers(-(1 << 28), 1 << 28, b.size)
    x[:200] = rng.integers(-(1 << 32), 1 << 32, 200)
    x[200:300] = 0
    x[300:304] = [(1 << 32) - 1, 1 - (1 << 32), 1 << 31, -(1 << 31)]  # around 2^32
    got = gpu("npu_test_mac", 1, None, b, c, x, flags, OUT, I64(x.size), n=x.size, dt=f32)
    same(got, gm.mac_exact(x, b, c, flags), x)


def test_ftz():
    got = gpu("npu_test_mac", 2, F32S, None, None, None, 0, OUT, I64(F32S.size), n=F32S.size,
              dt=f32)
    same(got, gc.ftz(F32S), F32S)


def code_pairs(nb=64, seed=5):
    """Every 16-bit code against nb codes (specials, subnormals, random)."""
    rng = np.random.default_rng(seed)
    b = np.concatenate([np.array([0, 0x8000, 1, 0x8001, 0x7F, 0x80, 0x3F80, 0xBF80, 0x7F80, 0xFF80,
                                  0x7F81, 0x7FC0, 0xFFFF, 0x7C00, 0x7E00, 0x3C00], np.uint16),
                        rng.integers(0, 1 << 16, nb - 16).astype(np.uint16)])
    return np.repeat(ALL16, nb), np.tile(b, ALL16.size)


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_mul16(t):
    x, y = code_pairs()
    got = gpu("npu_test_mul16", x, y, T[t], OUT, I64(x.size), n=x.size, dt=np.uint16)
    same(got, gm.mul16(x, y, t), x)


# ---- minmax ----

@pytest.mark.parametrize("op", [0, 1, 2])
def test_minmax(op):
    a, b, c = triples(1_000_000, 21)
    b[:1000], c[:1000] = a[:1000], -a[:1000]                         # equal and +-0 pairs
    a[1000:2000], b[1000:2000] = f32(0.0), f32(-0.0)
    got = gpu("npu_test_minmax", op, a, b, c, OUT, I64(a.size), n=a.size, dt=f32)
    want = gx.vmax(a, b) if op == 0 else gx.vmin(a, b) if op == 1 else gx.vclamp(a, b, c)
    same(got, want, a)


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("op", [0, 1, 2])
def test_minmax16(op, t):
    a, b = code_pairs(seed=6)
    got = gpu("npu_test_minmax16", op, a, b, T[t], OUT, I64(a.size), n=a.size, dt=np.uint16)
    if op == 2:
        want = gx.above(gc.decode_float(a, t), gc.decode_float(b, t)).astype(np.uint16)
    else:
        want = (gx.max16 if op == 0 else gx.min16)(a, b, t)
    same(got, want, a)


# ---- block_fp ----

def bfp16_blocks(nblk, seed, nan_rate=0.002):
    rng = np.random.default_rng(seed)
    e = rng.integers(100, 160, nblk)
    e[rng.random(nblk) < 0.05] = 0
    e[rng.random(nblk) < 0.05] = rng.integers(0, 256, 1)[0]
    e[rng.random(nblk) < nan_rate] = 255
    m = rng.integers(-128, 128, (nblk, 8)).astype(np.int8).view(np.uint8)
    m[rng.random((nblk, 8)) < 0.05] = 0
    return np.concatenate([e.astype(np.uint8)[:, None], m], axis=1).reshape(-1)


@pytest.mark.parametrize("op", [0, 1, 2, 3, 4])
def test_block_decode(op):
    nblk = 4003
    if op == 3:
        blk = np.random.default_rng(3).integers(0, 256, nblk * 12).astype(np.uint8)
    else:
        blk = bfp16_blocks(nblk, op)
    per = 16 if op == 3 else 8
    got = gpu("npu_test_block_decode", op, blk, I64(nblk), OUT, n=nblk * per, dt=np.uint32)
    want = [gb.decode_bfp16, gmm.decode_bfp16_identity, gmm.decode_bfp16_group, gb.decode_mx6,
            gmm.decode_bfp16_bf16][op](blk)
    if op == 4:
        want = want.astype(np.uint32)
    same(got, np.asarray(want).view(np.uint32) if op != 4 else want)


@pytest.mark.parametrize("rnd", [Rnd.kHalfEven, Rnd.kHalfAway, Rnd.kTrunc, Rnd.kCeil])
@pytest.mark.parametrize("op", [0, 1])
def test_block_encode(op, rnd):
    rng = np.random.default_rng(int(rnd) + 40 * op)
    per = 16 if op else 8
    nblk = 20_000 if op else 100_000
    v = F32S[rng.integers(0, F32S.size, nblk * per)]
    v[: nblk * per // 2] = (rng.standard_normal(nblk * per // 2) * 3).astype(f32)
    nb = 12 if op else 9
    got = gpu("npu_test_block_encode", op, v, int(rnd), OUT, I64(nblk), n=nblk * nb, dt=np.uint8)
    same(got, (gb.encode_mx6 if op else gb.encode_bfp16)(v, rnd))


# ---- mmac ----

@pytest.mark.parametrize("op", [0, 1, 2, 3])
def test_mmac(op):
    rng = np.random.default_rng(60 + op)
    n = 20_000
    acc = F32S[rng.integers(0, F32S.size, n)].view(np.uint32).copy()
    acc[: n // 2] = (rng.standard_normal(n // 2).astype(f32) * f32(64)).view(np.uint32)
    if op == 0:
        x, z = bfp16_blocks(n * 8, 1, 0), bfp16_blocks(n * 8, 2, 0)
        xm, xe = gb.bfp16_fields(x, n)
        zm, ze = gb.bfp16_fields(z, n)
        want = gmm.mmac_bfp16(acc, xm, xe, zm, ze)
    elif op == 1:
        x, z = (rng.integers(0, 256, n * 96).astype(np.uint8) for _ in range(2))
        for p in (x, z):  # block exponents in a useful range
            p.reshape(-1, 12)[:, 0] = rng.integers(110, 140, n * 8)
        xm, xe, xs = gb.mx6_fields(x, n)
        zm, ze, zs = gb.mx6_fields(z, n)
        want = gmm.mmac_mx6(acc, xm, xs, xe, zm, zs, ze)
    elif op == 2:
        x, z = (rng.integers(0, 256, n * 64).astype(np.uint8) for _ in range(2))
        want = gmm.mmac_fp8(acc, x.reshape(n, 64), z.reshape(n, 64))
    else:
        x = (rng.standard_normal(n * 32).astype(f32).view(np.uint32) >> 16).astype(np.uint16)
        z = rng.integers(0, 1 << 16, n * 32).astype(np.uint16)
        want = gmm.mmac_bf16(acc, x.reshape(n, 32), z.reshape(n, 32))
        x, z = x.view(np.uint8), z.view(np.uint8)
    got = gpu("npu_test_mmac", op, acc, x, z, OUT, I64(n), n=n, dt=np.uint32)
    same(got, want, acc)


# ---- nlf additions ----

NLF_UNARY = {
    3: m.hw_exp2_bf20, 4: m.hw_exp2_bf16, 5: ga.hw_gelu_f32, 6: ga.hw_silu_f32,
    7: lambda x: ga.hw_sigtanh_f32(x, False, ga.SIGTANH_K0, ga.SIGMOID_K1, ga.SIGMOID_K2),
    8: lambda x: ga.hw_sigtanh_f32(x, True, ga.SIGTANH_K0, ga.TANH_K1, ga.TANH_K2),
    9: ga.sigtanh_tanh_bf16, 10: lambda x: m.tanh_acc(x, "bf16"),
    11: lambda x: m.tanh_acc(x, "f16")}


@pytest.mark.parametrize("op", list(NLF_UNARY))
def test_nlf_unary(op):
    x = np.concatenate([half_inputs(), F32S])
    if op <= 4:  # exp2 takes a biased argument around 895
        x = np.concatenate([x, (np.random.default_rng(op).random(1 << 20) * 400 + 700).astype(f32)])
    same(gpu_unary(op, x), NLF_UNARY[op](x), x)


NLF16 = {0: lambda c: m.hw_sqrt16(c, "bf16"), 1: lambda c: m.hw_sqrt16(c, "f16"),
         2: lambda c: m.hw_inv16(c, "bf16"), 3: lambda c: m.hw_inv16(c, "f16"),
         4: m.hw_invsqrt_f16, 5: lambda c: m.tanh16(c, "bf16"), 6: lambda c: m.tanh16(c, "f16")}


@pytest.mark.parametrize("op", list(NLF16))
def test_nlf16(op):
    got = gpu("npu_test_nlf16", op, ALL16, OUT, I64(ALL16.size), n=ALL16.size, dt=np.uint16)
    same(got, NLF16[op](ALL16), ALL16)


@pytest.mark.parametrize("slope_t", ["f32", "f16"])
@pytest.mark.parametrize("rnd", [Rnd.kHalfEven, Rnd.kHalfAway])
@pytest.mark.parametrize("k", [8.0, 16.0])
def test_lut(k, rnd, slope_t):
    rng = np.random.default_rng(int(k) + int(rnd))
    lut = rng.standard_normal(512).astype(f32)
    x = np.concatenate([F32S[:200_000], (rng.random(1 << 20) * 40 - 20).astype(f32)])
    idx = gpu("npu_test_lut", 0, x, lut, float(k), 128, T[slope_t], int(rnd), OUT, I64(x.size),
              n=x.size, dt=f32)
    want = m.lut_index(x, k, 128, rnd)
    same(idx, want.astype(f32), x)
    got = gpu("npu_test_lut", 1, x, lut, float(k), 128, T[slope_t], int(rnd), OUT, I64(x.size),
              n=x.size, dt=f32)
    same(got, m.lut_interp(x, lut, want, slope_t, rnd), x)


def exp2_bf20_ref(p):
    """Independent model of the exp2 unit (fraction / integer fields of the biased input)."""
    b = np.asarray(p, f32).view(np.uint32).astype(np.int64)
    sign, expo, prefix = (b >> 31) & 1, (b >> 23) & 0xFF, (b >> 22) & 1
    integer, fraction, rem = (b >> 14) & 0xFF, (b >> 3) & 0x7FF, b & 7
    zero_m = (prefix == 0) & (integer == 0) & (fraction == 0) & (rem == 0)
    sel = (fraction >> 6) & 0x1F
    a = ((fraction << 1) - m.EXP2_CNST[sel]
         + (((fraction >> 1) & 0x1F) * m.EXP2_COEFF[sel] >> 3)) & 0xFFF
    ctrl = np.where((integer == 0) | (prefix == 0), ((4 - (integer & 0xF)) & 0xF) >> 1, 0)
    dn = (integer == 0) | (prefix == 0)
    odd = (1 << 10) | (((a >> 3) & 0x1FF) << 1) | ((a & 7) != 0)
    even = (1 << 11) | (((a >> 2) & 0x3FF) << 1) | ((a & 3) != 0)
    a = np.where(dn & (integer & 1 == 1), odd, np.where(dn, even, a))
    a = np.where(ctrl & 1, (((a >> 3) & 0x1FF) << 1) | ((a & 7) != 0), a)
    a = np.where(ctrl & 4, np.where(ctrl & 2 == 0, (((a >> 5) & 0x7F) << 1) | ((a & 0x1F) != 0),
                                    (((a >> 9) & 7) << 1) | ((a & 0x1FF) != 0)), a)
    nan = (expo == 0xFF) & ~zero_m
    low = (expo == 0x88) & (prefix == 0) & (integer < 0xF5)
    under = ~nan & ((expo < 0x88) | (sign == 1) | low)
    over = ~nan & ~under & ((expo > 0x88) | ((prefix == 1) & (integer == 0xFF)))
    normal = ~nan & ~under & ~over
    oe = np.where(normal & (prefix == 0), 0, integer)
    out = (oe << 23) | (a << 11)
    out = np.where(normal & ((prefix == 1) | (integer >= 0xF5)), out, 0)
    out = np.where(nan, 0x7FC00000, np.where(over, 0x7F800000, out))
    return out.astype(np.uint32).view(f32)


def test_exp2_independent_model():
    p = np.concatenate([F32S, (np.random.default_rng(9).random(1 << 20) * 400 + 700).astype(f32)])
    same(m.hw_exp2_bf20(p), exp2_bf20_ref(p), p)


def test_new_bad_args():
    from npu import raw_rc
    assert raw_rc("npu_test_int_round", 6, None, None, 0, 0, None, I64(1)) == -1
    assert raw_rc("npu_test_encode_float", None, 2, 10, 0, None, I64(1)) == -1
    assert raw_rc("npu_test_nlf16", 7, None, None, I64(1)) == -1
    assert raw_rc("npu_test_mmac", 4, None, None, None, None, I64(1)) == -1
    assert raw_rc("npu_test_block_decode", 5, None, I64(1), None) == -1
    assert raw_rc("npu_test_mul16", None, None, 3, None, I64(1)) == -1


# ==== special values: GPU == golden on the special tables, plus the default rules ====

from testing import (BF16_SPECIAL, F16_SPECIAL, FP8_SPECIAL, HALF_SPECIAL,  # noqa: E402
                     f32_special, int_special)

SP = f32_special()
SP_BITS = SP.view(np.uint32)
ALL_FLAGS = kFtzIn | kFtzOut | kSat
INT_TYPES = ["i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"]
WIDE = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.0**31, -2.0**31, 2.0**32, 2.0**63, -2.0**63,
                 2.0**64, 1e30, -1e30, 3.4028235e38, -3.4028235e38, 1e-45, -1e-45], f32)


def cross(*tables):
    """Every combination of the tables, as flat arrays (one per table)."""
    return [g.reshape(-1).copy() for g in np.meshgrid(*tables, indexing="ij")]


@pytest.mark.parametrize("rnd", ALL_RND)
def test_shift_round_extreme_shifts(rnd):
    """A shift amount of any int32 is defined: left shifts of 64 or more saturate, right shifts
    of 65 or more keep only the sticky bit."""
    rng = np.random.default_rng(int(rnd) + 300)
    sh = np.array([-2**31, -1000, -65, -64, -63, 63, 64, 65, 1000, 2**31 - 1], np.int32)
    v = np.repeat(int_struct(rng, 4000), sh.size)
    s = np.tile(sh, v.size // sh.size)
    got = gpu("npu_test_int_round", 0, v, s, int(rnd), 0, OUT, I64(v.size), n=v.size, dt=np.int64)
    same(got, gi.shift_round(v, s, rnd), v)
    for neg in (0, 1):
        got = gpu("npu_test_int_round", 1, v, s, int(rnd), neg, OUT, I64(v.size), n=v.size,
                  dt=np.int64)
        same(got, gi.shift_round_mag(v.astype(U64), s, rnd, bool(neg)).astype(np.int64), v)


@pytest.mark.parametrize("t", INT_TYPES)
def test_saturate_wrap_boundaries(t):
    v = np.concatenate([int_special(t), int_special("i64")])
    s = np.zeros(v.size, np.int32)
    for op, fn in ((2, gi.saturate), (3, gi.wrap), (4, gi.saturate_sym)):
        got = gpu("npu_test_int_round", op, v, s, 0, T[t], OUT, I64(v.size), n=v.size,
                  dt=np.int64)
        same(got, fn(v, t), v)


@pytest.mark.parametrize("rnd", ALL_RND)
@pytest.mark.parametrize("t", INT_TYPES)
def test_float_to_int_special(t, rnd):
    """NaN -> 0, infinities and out-of-range values saturate, for every integer type and for
    shifts that overflow or underflow the double scaling."""
    x = np.concatenate([SP, WIDE, -SP])
    for shift in (0, -2000, 2000, -64, 63, 64, 7):
        got = gpu("npu_test_float_to_int", 0, x, T[t], shift, int(rnd), OUT, I64(x.size),
                  n=x.size, dt=np.int64)
        want = gc.float_to_int(x, t, shift, rnd)
        same(got, want, x)
    lo, hi = gi.int_range(t)
    got = gc.float_to_int(np.array([np.nan, np.inf, -np.inf, 3.4028235e38, -3.4028235e38], f32), t)
    assert got.tolist() == [0, hi, lo, hi, lo]


@pytest.mark.parametrize("t", INT_TYPES)
def test_requant_tail_special(t):
    x = np.concatenate([SP, WIDE])
    got = gpu("npu_test_float_to_int", 1, x, T[t], 0, 0, OUT, I64(x.size), n=x.size, dt=np.int64)
    same(got, gq.requant_tail(x, t), x)


@pytest.mark.parametrize("rnd", ALL_RND)
@pytest.mark.parametrize("e", [-1000, -300, 300, 1000])
def test_int_to_float_extreme_exponents(rnd, e):
    v = np.concatenate([int_special(t) for t in INT_TYPES] + [-int_special("i64")[1:]])
    got = gpu("npu_test_int_to_float", v, e, int(rnd), OUT, I64(v.size), n=v.size, dt=f32)
    same(got, gc.int_to_float(v, e, rnd), v)


@pytest.mark.parametrize("flags", [0, kFtzIn, kFtzOut, kSat, ALL_FLAGS])
@pytest.mark.parametrize("rnd", ALL_RND)
@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
def test_encode_float_special(t, rnd, flags):
    same(gpu_encode(SP, t, rnd, flags), gc.encode_float(SP, t, rnd, flags), SP)


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_encode_float_special_rules(t):
    """[default] NaN stays NaN (sign kept), inf stays inf, zeros keep their sign; kSat turns
    only inf into the largest finite value."""
    pos = np.array([0, 0x7F800000, 0x7FC00000, 0x7F800001, 0x7FC00123], np.uint32)
    inf = 0x7F80 if t == "bf16" else 0x7C00
    for neg in (0, 1):
        x = (pos | (np.uint32(neg) << 31)).view(f32)
        c = gc.encode_float(x, t).astype(np.int64)
        sbit = neg << 15
        assert c[0] == sbit
        assert c[1] == sbit | inf
        assert ((c[2:] & 0x7FFF) > inf).all() and ((c[2:] >> 15) == neg).all()
        c = gc.encode_float(x, t, flags=kSat).astype(np.int64)
        assert c[1] == sbit | (inf - 1)
        assert ((c[2:] & 0x7FFF) > inf).all()


def test_encode_fp8_special_rules():
    x = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001], np.uint32)
    x = x.view(f32)
    assert gc.encode_float(x, "fp8").tolist() == [0x00, 0x80, 0x7F, 0xFF, 0x7F, 0xFF]
    assert gc.encode_float(x, "fp8", flags=kSat).tolist() == [0x00, 0x80, 0x7E, 0xFE, 0x7F, 0xFF]


@pytest.mark.parametrize("flags", [0, kFtzIn])
def test_decode_float_half_special(flags):
    """[default] The decode of a 16-bit NaN keeps its sign and payload bits (no quieting)."""
    for t, tab in HALF_SPECIAL.items():
        c = np.concatenate([tab, ALL16[::97]]).astype(np.uint32)
        got = gpu("npu_test_decode_float", c, T[t], flags, OUT, I64(c.size), n=c.size, dt=f32)
        want = gc.decode_float(c, t, flags)
        same(got, want, c)
        w = want.view(np.uint32)
        if t == "bf16":
            nan = (c & 0x7FFF) > 0x7F80
            assert (w[nan] == c[nan] << 16).all()
        else:
            nan = (c & 0x7FFF) > 0x7C00
            assert (w[nan] == (((c[nan] >> 15) << 31) | 0x7F800000 | ((c[nan] & 0x3FF) << 13))).all()


def test_decode_fp8_nan_sign():
    c = np.arange(256, dtype=np.uint32)
    got = gpu("npu_test_decode_float", c, T["fp8"], 0, OUT, I64(c.size), n=c.size, dt=f32)
    same(got, gc.decode_float(c, "fp8"), c)
    assert gc.decode_float(np.array([0x7F, 0xFF]), "fp8").view(np.uint32).tolist() == [
        0x7FC00000, 0xFFC00000]


@pytest.mark.parametrize("flags", [0, kFtzIn, kFtzOut])
@pytest.mark.parametrize("t,rnd", [("f32", Rnd.kHalfEven), ("bf16", Rnd.kHalfAway),
                                   ("f16", Rnd.kTrunc), ("fp8", Rnd.kHalfEven)])
def test_round_float_special(t, rnd, flags):
    got = gpu("npu_test_float_misc", 0, SP, T[t], int(rnd), flags, OUT, I64(SP.size), n=SP.size,
              dt=f32)
    same(got, gc.round_float(SP, t, rnd, flags), SP)


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_scalar_cast_special(t):
    got = gpu("npu_test_float_misc", 1, SP, T[t], 0, 0, OUT, I64(SP.size), n=SP.size, dt=f32)
    same(got, gc.scalar_cast(SP, t), SP)
    nonfinite = ~np.isfinite(SP)
    assert (gc.scalar_cast(SP, t).view(np.uint32)[nonfinite] == SP_BITS[nonfinite]).all()


@pytest.mark.parametrize("n", [1, 14, 100])
def test_fp8_decode_special(n):
    c = np.tile(FP8_SPECIAL, n)
    got = gpu("npu_test_fp8_decode", c, OUT, I64(c.size), n=c.size, dt=f32)
    same(got, gmm.fp8_decode_identity(c), c)


@pytest.mark.parametrize("flags", [0, kFtzIn | kFtzOut])
def test_fma_special_table(flags):
    a, b, c = cross(SP, SP, SP)
    got = gpu("npu_test_mac", 0, a, b, c, None, flags, OUT, I64(a.size), n=a.size, dt=f32)
    same(got, gm.fma_ftz(a, b, c, flags), a)


def test_fma_invalid_rules():
    """[default] 0 * inf and inf - inf give 0xFFC00000, also before a NaN addend; a NaN operand
    is returned quieted with its sign and payload; x + (-x) is +0, (-0) + (-0) is -0."""
    z, inf, nan, one = f32(0), f32(np.inf), fbits(0xFF800001), f32(1)
    nn = fbits(np.array([0x7FC00123], np.uint32))[0]
    assert bits(gm.fma_q(z, inf, one)).tolist() == [0xFFC00000]
    assert bits(gm.fma_q(z, inf, nan)).tolist() == [0xFFC00000]
    assert bits(gm.fma_q(one, inf, -inf)).tolist() == [0xFFC00000]
    assert bits(gm.fma_q(nan, one, one)).tolist() == [0xFFC00001]
    assert bits(gm.fma_q(one, nn, nan)).tolist() == [0x7FC00123]
    assert bits(gm.fma_q(one, one, -one)).tolist() == [0]
    assert bits(gm.fma_q(-z, one, -z)).tolist() == [0x80000000]


@pytest.mark.parametrize("flags", [0, kFtzIn | kFtzOut])
def test_mac_exact_special_table(flags):
    xs = np.array([0, 1, -1, 12345, (1 << 32) - 1, 1 - (1 << 32), 1 << 29], np.int64)
    x, c, a = cross(xs, SP, SP)
    got = gpu("npu_test_mac", 1, None, c, a, x, flags, OUT, I64(x.size), n=x.size, dt=f32)
    same(got, gm.mac_exact(x, c, a, flags), x)


def test_mac_exact_nan_rules():
    """[default] x == 0 with an infinite coefficient is invalid (0xFFC00000, before a NaN
    addend); a NaN coefficient or addend comes back quieted."""
    x = np.array([0, 0, 5, 5, 0], np.int64)
    c = fbits(np.array([0x7F800000, 0x7F800000, 0x7F800001, 0x3F800000, 0xFF800000], np.uint32))
    a = fbits(np.array([0x3F800000, 0xFFC00005, 0x7FC00007, 0x7F800001, 0x3F800000], np.uint32))
    assert bits(gm.mac_exact(x, c, a)).tolist() == [0xFFC00000, 0xFFC00000, 0x7FC00001,
                                                    0x7FC00001, 0xFFC00000]
    got = gpu("npu_test_mac", 1, None, c, a, x, 0, OUT, I64(x.size), n=x.size, dt=f32)
    same(got, gm.mac_exact(x, c, a), x)


def test_minmax_special_table():
    a, b, c = cross(SP, SP, SP)
    for op in (0, 1, 2):
        got = gpu("npu_test_minmax", op, a, b, c, OUT, I64(a.size), n=a.size, dt=f32)
        want = gx.vmax(a, b) if op == 0 else gx.vmin(a, b) if op == 1 else gx.vclamp(a, b, c)
        same(got, want, a)


def test_minmax_rules():
    pz, nz, nan = f32(0.0), f32(-0.0), fbits(np.array([0xFF800001], np.uint32))[0]
    assert bits(gx.vmax(nz, pz)).tolist() == bits(gx.vmax(pz, nz)).tolist() == [0]
    assert bits(gx.vmin(nz, pz)).tolist() == bits(gx.vmin(pz, nz)).tolist() == [0x80000000]
    assert bits(gx.vmax(nan, f32(1))).tolist() == bits(gx.vmin(f32(1), nan)).tolist() == [
        0x7FC00000]
    assert not gx.veq(nan, nan) and not gx.vgt(nan, f32(1)) and not gx.vge(f32(1), nan)
    assert gx.veq(pz, nz)


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_minmax16_special_table(t):
    a, b = cross(HALF_SPECIAL[t], HALF_SPECIAL[t])
    for op in (0, 1, 2):
        got = gpu("npu_test_minmax16", op, a, b, T[t], OUT, I64(a.size), n=a.size, dt=np.uint16)
        if op == 2:
            want = gx.above(gc.decode_float(a, t), gc.decode_float(b, t)).astype(np.uint16)
        else:
            want = (gx.max16 if op == 0 else gx.min16)(a, b, t)
        same(got, want, a)


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_mul16_special_table(t):
    x, y = cross(HALF_SPECIAL[t], HALF_SPECIAL[t])
    got = gpu("npu_test_mul16", x, y, T[t], OUT, I64(x.size), n=x.size, dt=np.uint16)
    same(got, gm.mul16(x, y, t), x)


@pytest.mark.parametrize("with_scalar", [False, True])
def test_vaddr_special_table(with_scalar):
    rng = np.random.default_rng(77)
    r = 20_000
    lanes = np.where(rng.random((r, 32)) < 0.25, SP[rng.integers(0, SP.size, (r, 32))],
                     np.where(rng.random((r, 32)) < 0.5, f32(0.0), make(rng.integers(120, 135, (r, 32)))))
    s = SP[rng.integers(0, SP.size, r)] if with_scalar else None
    assert_same(gpu_vaddr(lanes, s), gm.vaddr_f32_reduce(lanes, s))


@pytest.mark.parametrize("op", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
def test_nlf_unary_special_table(op):
    x = np.concatenate([SP, gc.bf16_to_f32(BF16_SPECIAL), gc.f16_to_f32(F16_SPECIAL)])
    want = {0: m.hw_inv_f32, 1: m.hw_invsqrt_f32, 2: m.hw_approx_ln}.get(op) or NLF_UNARY[op]
    assert_same(gpu_unary(op, x), want(x), x)


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_nlf16_special_rules(t):
    """[default] 16-bit reciprocal / square root of a NaN code is that NaN quieted."""
    tab = HALF_SPECIAL[t]
    nan = (tab & 0x7FFF) > (0x7F80 if t == "bf16" else 0x7C00)
    q = np.uint16(0x40 if t == "bf16" else 0x200)
    for fn in (m.hw_inv16, m.hw_sqrt16):
        assert (fn(tab, t)[nan] == (tab[nan] | q)).all()


@pytest.mark.parametrize("rnd", [Rnd.kHalfEven, Rnd.kTrunc, Rnd.kCeil])
@pytest.mark.parametrize("op", [0, 1])
def test_block_encode_special_table(op, rnd):
    rng = np.random.default_rng(int(rnd) + 90 + op)
    per, nb = (16, 12) if op else (8, 9)
    nblk = 20_000
    v = np.where(rng.random(nblk * per) < 0.5, SP[rng.integers(0, SP.size, nblk * per)],
                 (rng.standard_normal(nblk * per) * 3).astype(f32))
    got = gpu("npu_test_block_encode", op, v, int(rnd), OUT, I64(nblk), n=nblk * nb, dt=np.uint8)
    same(got, (gb.encode_mx6 if op else gb.encode_bfp16)(v, rnd))


@pytest.mark.parametrize("op", [0, 1, 2, 4])
def test_block_decode_extreme_blocks(op):
    """Exponents 0 / 1 / 254 / 255 with extreme mantissas, in groups of 8."""
    rng = np.random.default_rng(op + 500)
    nblk = 4000
    e = rng.choice(np.array([0, 1, 2, 253, 254, 255], np.uint8), nblk)
    mant = rng.choice(np.array([-128, -127, -1, 0, 1, 127], np.int8), (nblk, 8)).view(np.uint8)
    blk = np.concatenate([e[:, None], mant], axis=1).reshape(-1)
    got = gpu("npu_test_block_decode", op, blk, I64(nblk), OUT, n=nblk * 8, dt=np.uint32)
    want = [gb.decode_bfp16, gmm.decode_bfp16_identity, gmm.decode_bfp16_group, None,
            gmm.decode_bfp16_bf16][op](blk)
    same(got, want.astype(np.uint32) if op == 4 else np.asarray(want).view(np.uint32))


def test_block_decode_mx6_extreme_blocks():
    """Exponent 255 with magnitude 15 overflows to infinity."""
    rng = np.random.default_rng(600)
    nblk = 4000
    blk = rng.integers(0, 256, nblk * 12).astype(np.uint8).reshape(nblk, 12)
    blk[:, 0] = rng.choice(np.array([0, 1, 130, 254, 255], np.uint8), nblk)
    blk[: nblk // 4, 2:] = 0xEF  # magnitudes 15 (sign bits set as well)
    blk[nblk // 4: nblk // 2, 2:] = 0x0F
    blk = blk.reshape(-1)
    got = gpu("npu_test_block_decode", 3, blk, I64(nblk), OUT, n=nblk * 16, dt=np.uint32)
    same(got, gb.decode_mx6(blk).view(np.uint32))


@pytest.mark.parametrize("op", [0, 1, 2, 3])
def test_mmac_special(op):
    rng = np.random.default_rng(700 + op)
    n = 5000
    acc = SP_BITS[rng.integers(0, SP.size, n)].copy()
    ebyte = np.array([0, 1, 254, 255], np.uint8)
    if op == 0:
        mk = lambda: np.concatenate([rng.choice(ebyte, (n * 8, 1)),
                                     rng.choice(np.array([-128, -127, 0, 127], np.int8),
                                                (n * 8, 8)).view(np.uint8)], axis=1).reshape(-1)
        x, z = mk(), mk()
        xm, xe = gb.bfp16_fields(x, n)
        zm, ze = gb.bfp16_fields(z, n)
        want = gmm.mmac_bfp16(acc, xm, xe, zm, ze)
    elif op == 1:
        x, z = (rng.integers(0, 256, n * 96).astype(np.uint8) for _ in range(2))
        for p in (x, z):
            p.reshape(-1, 12)[:, 0] = rng.choice(ebyte, n * 8)
        xm, xe, xs = gb.mx6_fields(x, n)
        zm, ze, zs = gb.mx6_fields(z, n)
        want = gmm.mmac_mx6(acc, xm, xs, xe, zm, zs, ze)
    elif op == 2:
        x, z = (rng.choice(FP8_SPECIAL, n * 64) for _ in range(2))
        want = gmm.mmac_fp8(acc, x.reshape(n, 64), z.reshape(n, 64))
    else:
        x, z = (rng.choice(BF16_SPECIAL, n * 32) for _ in range(2))
        want = gmm.mmac_bf16(acc, x.reshape(n, 32), z.reshape(n, 32))
        x, z = x.view(np.uint8), z.view(np.uint8)
    got = gpu("npu_test_mmac", op, acc, x, z, OUT, I64(n), n=n, dt=np.uint32)
    same(got, want, acc)


@pytest.mark.parametrize("op,fn", [(3, gm.mul_q), (4, gm.add_q)])
def test_mul_add_q_special_table(op, fn):
    """mul_q / add_q: the first NaN operand (two NaNs of different sign / payload included)."""
    a, b = cross(SP, SP)
    got = gpu("npu_test_mac", op, a, b, None, None, 0, OUT, I64(a.size), n=a.size, dt=f32)
    same(got, fn(a, b), a)
    nan_a = np.isnan(a) & np.isnan(b)
    assert (got.view(np.uint32)[nan_a] == (a.view(np.uint32)[nan_a] | 0x400000)).all()


def test_fma_q_special_table():
    a, b, c = cross(SP, SP, SP)
    got = gpu("npu_test_mac", 5, a, b, c, None, 0, OUT, I64(a.size), n=a.size, dt=f32)
    same(got, gm.fma_q(a, b, c), a)


@pytest.mark.parametrize("is_tanh", [0, 1])
def test_sigtanh_special_coefficients(is_tanh):
    """[default] A NaN e from a non-finite coefficient comes out of 1 - e with the sign of e."""
    x, k0, k1 = cross(SP, SP, SP)
    got = gpu("npu_test_mac", 6, x, k0, k1, None, is_tanh, OUT, I64(x.size), n=x.size, dt=f32)
    k2 = ga.TANH_K2 if is_tanh else ga.SIGMOID_K2
    same(got, ga.hw_sigtanh_f32(x, bool(is_tanh), k0, k1, k2), x)


def test_sub_q_special_table():
    a, b = cross(SP, SP)
    got = gpu("npu_test_mac", 7, a, b, None, None, 0, OUT, I64(a.size), n=a.size, dt=f32)
    same(got, gm.sub_q(a, b), a)
    both = np.isnan(a) & np.isnan(b)  # the first NaN operand: a's, quieted
    assert (got.view(np.uint32)[both] == (a.view(np.uint32)[both] | 0x400000)).all()


def test_sat_add64():
    v = np.concatenate([int_struct(np.random.default_rng(9), 50_000), int_special("i64")])
    s = np.zeros(v.size, np.int32)
    got = gpu("npu_test_int_round", 5, v, s, 0, 0, OUT, I64(v.size), n=v.size, dt=np.int64)
    same(got, gi.sat_add64(v, v[(np.arange(v.size) * 7 + 3) % v.size]), v)
    big = np.iinfo(np.int64).max
    assert gi.sat_add64(big, 1) == big and gi.sat_add64(-big - 1, -1) == -big - 1
    assert gi.sat_add64(big, -big) == 0


def test_clamp64_lo_above_hi():
    """As the C helper: below lo gives lo, else hi."""
    v = np.array([-5, 0, 3, 10], np.int64)
    want = [lo if x < lo else hi if x > hi else x for x in v.tolist() for lo, hi in [(4, 2)]]
    assert gi.clamp64(v, 4, 2).tolist() == want


@pytest.mark.parametrize("rnd", [Rnd.kHalfEven, Rnd.kCeil, Rnd.kTrunc])
def test_float_to_int_huge_shifts(rnd):
    """The shift is an int: any amount saturates / underflows like the GPU (no exception)."""
    x = np.concatenate([SP, WIDE])
    for shift in (-2**31, 2**31 - 1, -5000, 5000, -1300, 1300, -1201, 1200):
        got = gpu("npu_test_float_to_int", 0, x, T["i32"], shift, int(rnd), OUT, I64(x.size),
                  n=x.size, dt=np.int64)
        same(got, gc.float_to_int(x, "i32", shift, rnd), x)
    assert gc.float_to_int(f32(1), "i32", 2**40).tolist() == 2**31 - 1
    assert gc.float_to_int(f32(1), "i32", -2**40).tolist() == 0


def test_mac_exact_wide_x():
    """x of any int64 value is exact (the product needs up to 87 bits)."""
    rng = np.random.default_rng(31)
    _, c, a = triples(200_000, 33)
    x = rng.integers(-(1 << 62), 1 << 62, c.size)
    x[:6] = [np.iinfo(np.int64).min, np.iinfo(np.int64).max, 1 << 38, -(1 << 39), 1 << 40, 1 << 55]
    x[6:1000] = rng.integers(-(1 << 63), 1 << 63, 994, dtype=np.int64)
    for flags in (0, kFtzIn | kFtzOut):
        got = gpu("npu_test_mac", 1, None, c, a, x, flags, OUT, I64(x.size), n=x.size, dt=f32)
        same(got, gm.mac_exact(x, c, a, flags), x)


def test_sqrt_q_special_table():
    x = np.concatenate([SP, -SP, WIDE])
    got = gpu("npu_test_mac", 8, x, None, None, None, 0, OUT, I64(x.size), n=x.size, dt=f32)
    same(got, m.sqrt_q(x), x)
    assert bits(m.sqrt_q(np.array([-1.0, -0.0, 4.0], f32))).tolist() == [0xFFC00000, 0x80000000,
                                                                           0x40000000]


# ==== negated multiply-adds, shift-round-saturate, shared sequences, single-value MMACs ====

import golden_grid as gg  # noqa: E402
import golden_quant as gq  # noqa: E402
from testing import fp8_values  # noqa: E402

I32MIN, I32MAX = -2**31, 2**31 - 1


def as_bits(v):
    """fp32 values as the int64 an entry returns their bits in."""
    return np.asarray(v, f32).view(np.uint32).astype(np.int64)


def moderate(n, seed):
    """Operands of a quantize step: values, scales and zero points of ordinary size mixed with
    the structured fp32 samples."""
    rng = np.random.default_rng(seed)
    v, a, b = triples(n, seed)
    k = n // 2
    v[:k] = (rng.standard_normal(k) * 100).astype(f32)
    v[: k // 2] = (rng.integers(-2000, 2000, k // 2) * f32(0.125)).astype(f32)  # exact ties
    a[:k] = rng.choice(np.array([1.0, 0.5, 0.37, 2.0, 0.0078125, -1.0], f32), k)
    b[:k] = rng.choice(np.array([0.0, -0.0, 128.0, -128.0, 0.5, 3.0], f32), k)
    return v, a, b


@pytest.mark.parametrize("rnd", ALL_RND)
@pytest.mark.parametrize("t", ["i8", "u16", "i32", "i64"])
def test_srs(t, rnd):
    rng = np.random.default_rng(int(rnd) + 900)
    v = np.concatenate([int_struct(rng, 50_000), int_special(t), int_special("i64")])
    s = rng.integers(-8, 70, v.size).astype(np.int32)
    s[: v.size // 2] = rng.integers(1, 42, v.size // 2)
    got = gpu("npu_test_srs", v, s, T[t], int(rnd), OUT, I64(v.size), n=v.size, dt=np.int64)
    same(got, gi.srs(v, s, t, rnd), v)


def test_negmul_q_msc_q():
    """negmul_q = -(a * b), msc_q = c - a * b in one rounding; NaN operands as mul_q / fma_q
    with the sign bit flipped after / before."""
    for a, b, c in (triples(500_000, 41), cross(SP, SP, SP)):
        got = gpu("npu_test_mac2", 0, a, b, c, OUT, I64(a.size), n=a.size, dt=f32)
        same(got, gm.negmul_q(a, b), a)
        got = gpu("npu_test_mac2", 1, a, b, c, OUT, I64(a.size), n=a.size, dt=f32)
        same(got, gm.msc_q(a, b, c), a)
    one, nan = f32(1), fbits(np.array([0x7F800001], np.uint32))[0]
    assert bits(gm.negmul_q(nan, one)).tolist() == [0xFFC00001]
    assert bits(gm.msc_q(nan, one, one)).tolist() == [0xFFC00001]
    assert bits(gm.msc_q(one, one, nan)).tolist() == [0x7FC00001]
    assert bits(gm.msc_q(one, one, one)).tolist() == [0]


def test_scalar_cast_i32():
    x = np.concatenate([to_int_inputs(), SP, WIDE])
    got = gpu("npu_test_quant", 4, None, x, None, None, 0, 0, 0, OUT, I64(x.size), n=x.size,
              dt=np.int64)
    same(got, gc.scalar_cast_i32(x), x)
    r = gc.scalar_cast_i32(np.array([np.nan, np.inf, -np.inf, 2.9, -2.9, 3e9], f32))
    assert r.tolist() == [0, I32MAX, I32MIN, 2, -2, I32MAX]


@pytest.mark.parametrize("rnd", [Rnd.kHalfEven, Rnd.kHalfAway, Rnd.kTrunc, Rnd.kFloor])
@pytest.mark.parametrize("t,shift", [("i8", 0), ("u8", 0), ("i16", 0), ("u16", 3), ("i32", -4)])
def test_quant(t, shift, rnd):
    v, a, b = moderate(300_000, 50 + int(rnd))
    got = gpu("npu_test_quant", 0, None, v, a, b, T[t], int(rnd), shift, OUT, I64(v.size),
              n=v.size, dt=np.int64)
    same(got, gq.quant(v, a, b, t, rnd, shift), v)


def test_quant_special_table():
    v, a, b = cross(SP, SP, SP)
    for t in ("i8", "u16", "i32"):
        got = gpu("npu_test_quant", 0, None, v, a, b, T[t], 0, 0, OUT, I64(v.size), n=v.size,
                  dt=np.int64)
        same(got, gq.quant(v, a, b, t), v)


def requant_inputs(seed):
    rng = np.random.default_rng(seed)
    _, a, b = moderate(200_000, seed)
    x = rng.integers(-40000, 70000, a.size)
    x[:1000] = rng.integers(-(1 << 62), 1 << 62, 1000)
    x[1000:1100] = 0
    return x, a, b


@pytest.mark.parametrize("rnd", [Rnd.kHalfEven, Rnd.kTrunc])
@pytest.mark.parametrize("t,shift", [("i8", 0), ("u8", 0), ("u16", 0), ("i16", 2), ("i32", 0)])
def test_requant(t, shift, rnd):
    x, a, b = requant_inputs(60 + int(rnd))
    got = gpu("npu_test_quant", 1, x, None, a, b, T[t], int(rnd), shift, OUT, I64(x.size),
              n=x.size, dt=np.int64)
    same(got, gq.requant(x, a, b, t, shift, rnd), x)


def test_dequant():
    x, a, b = requant_inputs(70)
    got = gpu("npu_test_quant", 2, x, None, a, b, 0, 0, 0, OUT, I64(x.size), n=x.size,
              dt=np.int64)
    same(got, as_bits(gq.dequant(x, a, b)), x)
    xs = np.array([0, 1, -1, 255, -32768, 1 << 40], np.int64)
    x, a, b = cross(xs, SP, SP)
    got = gpu("npu_test_quant", 2, x, None, a, b, 0, 0, 0, OUT, I64(x.size), n=x.size,
              dt=np.int64)
    same(got, as_bits(gq.dequant(x, a, b)), x)


def test_dq_offset():
    """-zp * scale with the sign bit of zp flipped first: a NaN zp comes back with the other
    sign, quieted."""
    for zp, sc in (cross(SP, SP), triples(200_000, 80)[:2]):
        got = gpu("npu_test_quant", 3, None, zp, sc, None, 0, 0, 0, OUT, I64(zp.size), n=zp.size,
                  dt=np.int64)
        same(got, as_bits(gq.dq_offset(zp, sc)), zp)
    nan = fbits(np.array([0x7F800001], np.uint32))[0]
    assert bits(gq.dq_offset(nan, f32(2))).tolist() == [0xFFC00001]
    assert bits(gq.dq_offset(f32(0), f32(2))).tolist() == [0x80000000]


@pytest.mark.parametrize("flags", [0, kFtzIn | kFtzOut])
@pytest.mark.parametrize("t,rnd", [("bf16", Rnd.kHalfEven), ("bf16", Rnd.kHalfAway),
                                   ("f16", Rnd.kHalfEven), ("f32", Rnd.kHalfEven)])
def test_fp8_in(t, rnd, flags):
    rng = np.random.default_rng(int(rnd) + flags + 90)
    c = np.concatenate([fp8_values(rng, 20_003), np.tile(FP8_SPECIAL, 9),
                        np.arange(256, dtype=np.uint16).astype(np.uint8)])
    for scale in (1.0, 0.37, 2.0 ** -130, 3e38, float("inf"), float(fbits(0xFFC00001))):
        got = gpu("npu_test_fp8_io", 0, c, None, scale, T[t], int(rnd), flags, OUT, I64(c.size),
                  n=c.size, dt=np.uint32)
        same(got, gq.fp8_in(c, f32(scale), t, rnd, flags).view(np.uint32), c)


@pytest.mark.parametrize("rnd", [Rnd.kHalfEven, Rnd.kHalfAway, Rnd.kTrunc])
def test_fp8_out(rnd):
    y = np.concatenate([F32S, SP, WIDE])
    for recip in (1.0, 2.7, 2.0 ** -20, float("inf"), float(fbits(0x7F800001))):
        got = gpu("npu_test_fp8_io", 1, None, y, recip, 0, int(rnd), 0, OUT, I64(y.size),
                  n=y.size, dt=np.uint32)
        same(got, gq.fp8_out(y, f32(recip), rnd).astype(np.uint32), y)
    r = gq.fp8_out(np.array([np.inf, -np.inf, 1e9, np.nan], f32), f32(1), Rnd.kHalfEven)
    assert r.tolist() == [0x7E, 0xFE, 0x7E, 0x7F]


def act_call(op, v, a, b, s, act, t, rnd):
    return gpu("npu_test_act", op, v, a, b, s, act, T[t], int(rnd), OUT, I64(v.size), n=v.size,
               dt=np.int64)


@pytest.mark.parametrize("rnd", [Rnd.kHalfEven, Rnd.kHalfAway])
@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_sigmoid_tanh(t, rnd):
    x = np.concatenate([half_inputs(), F32S, SP,
                        (np.random.default_rng(3).standard_normal(1 << 18) * 4).astype(f32)])
    same(act_call(0, x, None, None, None, 0, t, rnd), as_bits(ga.sigmoid_tanh(x, t, rnd)), x)


def test_sigmoid_tanh_two_step_form():
    """0.5 - fp32(t * 0.5) is the single-rounding fma(t, -0.5, 0.5) for every non-NaN input:
    the half of a tanh table value is exact."""
    x = np.concatenate([half_inputs(), F32S, SP])
    x = x[~np.isnan(x)]
    with np.errstate(all="ignore"):
        t = m.tanh_acc((x * f32(-0.5)).astype(f32), "bf16")
        two = gc.round_float((f32(0.5) - (t * f32(0.5)).astype(f32)).astype(f32), "bf16",
                             Rnd.kHalfEven)
    same(two, ga.sigmoid_tanh(x, "bf16", Rnd.kHalfEven), x)


@pytest.mark.parametrize("act", [0, 1, 2, 3, 4])
def test_activate_bf16(act):
    v = gc.bf16_to_f32(ALL16)
    alphas = np.concatenate([np.array([0.0, -0.0, 0.01, 0.1015625, 1.0, -0.5, 3e38, 1e-39], f32),
                             fbits(np.array([0x7F800000, 0xFF800000, 0x7FC00000, 0xFF800001],
                                            np.uint32))])
    v, al = cross(v, alphas)
    same(act_call(1, v, al, None, None, act, "bf16", 0), as_bits(ga.activate_bf16(v, act, al)), v)


def test_exp2_bf16():
    x = np.concatenate([F32S, SP, half_inputs(),
                        (np.random.default_rng(4).random(1 << 20) * 400 - 200).astype(f32)])
    same(act_call(2, x, None, None, None, 0, "bf16", 0), as_bits(m.exp2_bf16(x)), x)
    assert m.kExp2Bias.view(np.uint32) == 0x445FC010 and m.kLog2e.view(np.uint32) == 0x3FB8AA3B


QDQ_ACTS = [gq.kQdqActNone, gq.kQdqActClip, gq.kQdqActLrelu, gq.kQdqActPrelu, gq.kQdqActGelu, 55]


def activate_inputs():
    v, a, b = moderate(200_000, 95)
    s = triples(200_000, 96)[0]
    s[:100_000] = np.random.default_rng(97).choice(np.array([0.25, -0.5, 0.0, 1.5], f32), 100_000)
    sp = cross(SP, SP, SP, SP[::3])
    return [np.concatenate([p, q]) for p, q in zip((v, a, b, s), sp)]


@pytest.mark.parametrize("act", QDQ_ACTS)
def test_activate_f32(act):
    v, a, b, s = activate_inputs()
    same(act_call(3, v, a, b, s, act, "i8", 0), as_bits(gq.activate_f32(v, act, a, b, s)), v)


@pytest.mark.parametrize("t", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("act", QDQ_ACTS)
def test_activate_out(act, t):
    v, a, b, s = activate_inputs()
    same(act_call(4, v, a, b, s, act, t, 0), gq.activate_out(v, act, a, b, s, t), v)


def test_mmac_add_bf16():
    rng = np.random.default_rng(110)
    n = 400_000
    acc = F32S[rng.integers(0, F32S.size, n)].copy()
    acc[: n // 2] = (rng.standard_normal(n // 2) * 64).astype(f32)
    x = (rng.integers(0, 1 << 16, n).astype(np.uint32) << 16).view(f32).copy()   # exact in bf16
    x[: n // 4] = gc.round_float((rng.standard_normal(n // 4) * 64).astype(f32), "bf16")
    x[n // 4: n // 4 + 5000] = F32S[rng.integers(0, F32S.size, 5000)]              # not exact
    x[-n // 8:] = -acc[-n // 8:]                                                  # cancellation
    for a, b in ((acc, x), cross(SP, SP)):
        got = gpu("npu_test_mmac_add", 0, a, b, None, I64(0), OUT, I64(a.size), n=a.size,
                  dt=np.uint32)
        same(got, gmm.mmac_add_bf16(a, b).view(np.uint32), a)


@pytest.mark.parametrize("nblk", [8, 61, 4003])
def test_mmac_add_bfp16(nblk):
    rng = np.random.default_rng(120 + nblk)
    blk = bfp16_blocks(nblk, 7 + nblk)
    n = nblk * 8
    a = gmm.decode_bfp16_identity(blk)
    acc = F32S[rng.integers(0, F32S.size, n)].copy()
    acc[: n // 2] = (rng.standard_normal(n // 2) * 2.0 ** rng.integers(-30, 30, n // 2)).astype(f32)
    acc[n // 2: n // 2 + n // 8] = -a[n // 2: n // 2 + n // 8]
    got = gpu("npu_test_mmac_add", 1, acc, None, blk, I64(nblk), OUT, I64(n), n=n, dt=np.uint32)
    same(got, gmm.mmac_add_bfp16(blk, a, acc, n).view(np.uint32), acc)


def test_mmac_add_bfp16_outside_group():
    """A block outside the group adds nothing: the accumulator alone goes through the block
    accumulate / normalise of that group's exponents."""
    rng = np.random.default_rng(130)
    nblk = 203
    blk = bfp16_blocks(nblk, 131)
    n = nblk * 8
    acc = F32S[rng.integers(0, F32S.size, n)].copy()
    e = blk.reshape(-1, 9)[:, 0].astype(np.int64)
    e = np.concatenate([e, np.zeros((-nblk) % 8 + 8, np.int64)]).reshape(-1, 8)
    ce = np.repeat(e[1:], 64, axis=0)[:n] + 127          # the group after each value's own
    got = gpu("npu_test_mmac_add", 2, acc, None, blk, I64(nblk), OUT, I64(n), n=n, dt=np.uint32)
    same(got, gmm.block_acc(acc.view(np.uint32), np.zeros((n, 8), np.int64), ce), acc)


def test_bfp16_identity_is_block_mmac():
    """decode_bfp16_identity is the bfp16 MMAC of the value against a staged 1.0 onto a zero
    accumulator (the other lanes zero)."""
    nblk = 4000
    for seed in (140, 141):
        blk = bfp16_blocks(nblk, seed, 0.02 if seed == 141 else 0)
        if seed == 141:
            blk.reshape(-1, 9)[:, 0] = np.random.default_rng(seed).choice(
                np.array([0, 1, 2, 120, 137, 253, 254, 255], np.uint8), nblk)
        xm, xe = gb.bfp16_fields(blk, nblk // 8)                 # (groups, 64), (groups, 8)
        g = xm.shape[0]
        one = np.zeros((g, 64, 64), np.int64)
        one[:, np.arange(64), np.arange(64)] = xm                # value j alone in MMAC j
        want = gmm.mmac_bfp16(np.zeros((g, 64), np.uint32), one, xe[:, None, :],
                              np.full((g, 64, 64), 64, np.int64), np.full((g, 1, 8), 127, np.int64))
        got = gpu("npu_test_block_decode", 1, blk, I64(nblk), OUT, n=nblk * 8, dt=np.uint32)
        same(got, want)
        same(gmm.decode_bfp16_identity(blk).view(np.uint32), want)


def test_fp8_identity_is_mmac():
    """fp8_decode_identity of a non-NaN code is the fp8 MMAC of that code against 1.0 (0x38)
    onto a zero accumulator, whatever its lane."""
    codes = np.array([c for c in range(256) if c & 0x7F != 0x7F], np.uint8)
    want = gpu("npu_test_fp8_decode", codes, OUT, I64(codes.size), n=codes.size, dt=f32)
    same(want, gmm.fp8_decode_identity(codes), codes)
    for lane in (0, 13, 31, 32, 63):
        x = np.zeros((codes.size, 64), np.uint8)
        z = np.zeros((codes.size, 64), np.uint8)
        x[:, lane], z[:, lane] = codes, 0x38
        acc = np.zeros(codes.size, np.uint32)
        got = gpu("npu_test_mmac", 2, acc, x.reshape(-1), z.reshape(-1), OUT, I64(codes.size),
                  n=codes.size, dt=np.uint32)
        same(got, want.view(np.uint32), codes)
        same(gmm.mmac_fp8(acc, x, z), want.view(np.uint32), codes)


@pytest.mark.parametrize("align", [0, 1])
@pytest.mark.parametrize("zp,scale", [(0, 2.0 / 255), (128, 0.0078125), (-7, 0.37),
                                      (3, float("inf")), (0, float("nan")), (5, 0.0), (1, 3e38)])
def test_qdq_grid_coord(zp, scale, align):
    rng = np.random.default_rng(150)
    code = np.concatenate([np.arange(-40000, 40000, dtype=np.int64),
                           rng.integers(I32MIN, I32MAX, 20_000), int_special("i32"),
                           int_special("u16")])
    for extent in (1, 2, 7, 640, 100_000):
        got = gpu("npu_test_grid", code, zp, float(f32(scale)), I64(extent), align, OUT,
                  I64(code.size), n=code.size, dt=f32)
        same(got, gg.qdq_grid_coord(code, zp, scale, extent, align), code)


@pytest.mark.parametrize("rnd", ALL_RND)
@pytest.mark.parametrize("e", [-100_000, -5000, -1500, 1500, 5000, 100_000, I32MIN, I32MAX])
def test_int_to_float_huge_exponents(rnd, e):
    """The exponent is an int: any amount overflows / leaves the sticky bit like the model."""
    rng = np.random.default_rng(int(rnd) + 160)
    v = np.concatenate([int_special(t) for t in INT_TYPES] + [-int_special("i64")[1:],
                                                             int_struct(rng, 2000)])
    got = gpu("npu_test_int_to_float", v, e, int(rnd), OUT, I64(v.size), n=v.size, dt=f32)
    same(got, gc.int_to_float(v, int(np.clip(e, -100_000, 100_000)), rnd), v)


def test_sqrt_q_structured():
    x = np.concatenate([F32S, half_inputs()])
    got = gpu("npu_test_mac", 8, x, None, None, None, 0, OUT, I64(x.size), n=x.size, dt=f32)
    same(got, m.sqrt_q(x), x)


def test_shared_bad_args():
    from npu import raw_rc
    assert raw_rc("npu_test_srs", None, None, 4, 10, None, I64(1)) == -1
    assert raw_rc("npu_test_mac2", 2, None, None, None, None, I64(1)) == -1
    assert raw_rc("npu_test_mmac_add", 3, None, None, None, I64(1), None, I64(8)) == -1
    assert raw_rc("npu_test_mmac_add", 1, None, None, None, I64(1), None, I64(7)) == -1
    assert raw_rc("npu_test_act", 5, None, None, None, None, 0, 2, 0, None, I64(1)) == -1
    assert raw_rc("npu_test_quant", 5, None, None, None, None, 4, 0, 0, None, I64(1)) == -1
    assert raw_rc("npu_test_fp8_io", 2, None, None, 1.0, 2, 0, 0, None, I64(1)) == -1
    assert raw_rc("npu_test_grid", None, 0, 1.0, I64(1), 0, None, I64(-1)) == -1


# ==== flagged quantize steps, fp8 max, format limits, wrapping int64, divide, 16-bit index ====

import golden_bits as gbits  # noqa: E402


def misc(op, x=None, a=None, b=None, c=None, t="f32", rnd=0, shift=0, flags=0, n=None):
    n = next(v for v in (x, a) if v is not None).size if n is None else n
    return gpu("npu_test_misc", op, x, a, b, c, T[t], int(rnd), shift, flags, OUT, I64(n), n=n,
               dt=np.int64)


@pytest.mark.parametrize("flags", [0, kSat])
@pytest.mark.parametrize("rnd", [Rnd.kHalfEven, Rnd.kHalfAway, Rnd.kTrunc])
def test_fp8_out_flags(rnd, flags):
    y = np.concatenate([F32S[:200_000], SP, WIDE])
    for recip in (1.0, 2.7, 2.0 ** -20, float("inf"), float(fbits(0x7F800001))):
        r = np.full(y.size, recip, f32)
        same(misc(0, a=y, b=r, rnd=rnd, flags=flags), gq.fp8_out(y, r, rnd, flags).astype(I64), y)
    r = gq.fp8_out(np.array([np.inf, -np.inf, 1e9, -1e9, 448.0, np.nan], f32), f32(1),
                   Rnd.kHalfEven, 0)
    assert r.tolist() == [0x7F, 0xFF, 0x7F, 0xFF, 0x7E, 0x7F]


@pytest.mark.parametrize("flags", [0, kFtzIn, kFtzOut, kFtzIn | kFtzOut])
def test_dequant_flags(flags):
    x, a, b = requant_inputs(71 + flags)
    same(misc(1, x, a, b, flags=flags), as_bits(gq.dequant(x, a, b, flags)), x)
    xs = np.array([0, 1, -1, 255, -32768, 1 << 40], np.int64)
    x, a, b = cross(xs, SP, SP)
    same(misc(1, x, a, b, flags=flags), as_bits(gq.dequant(x, a, b, flags)), x)


@pytest.mark.parametrize("flags", [0, kFtzIn, kFtzOut, kFtzIn | kFtzOut])
@pytest.mark.parametrize("t,shift,rnd", [("u8", 0, Rnd.kHalfEven), ("u16", 0, Rnd.kHalfAway),
                                         ("i16", 3, Rnd.kFloor), ("i32", -4, Rnd.kHalfEven)])
def test_quant_flags(t, shift, rnd, flags):
    for v, a, b in (moderate(200_000, 60 + flags), cross(SP, SP, SP)):
        same(misc(2, None, v, a, b, t=t, rnd=rnd, shift=shift, flags=flags),
             gq.quant(v, a, b, t, rnd, shift, flags), v)


def test_fp8_max():
    """Every pair of codes: the running value stays unless the other is strictly above."""
    x = np.arange(1 << 16, dtype=np.int64)
    a, b = (x & 255).astype(np.uint8), (x >> 8).astype(np.uint8)
    want = gx.fp8_max(a, b)
    same(misc(3, x), want.astype(I64), x)
    fa, fb = gc.fp8_to_f32(a), gc.fp8_to_f32(b)
    assert np.array_equal(want == b, gx.above(fb, fa) | (a == b))
    assert gx.fp8_max(0x7F, 0x7E) == 0x7F and gx.fp8_max(0x10, 0x7F) == 0x10  # NaN: a stays
    assert gx.fp8_max(0x80, 0x00) == 0x00 and gx.fp8_max(0x00, 0x80) == 0x00  # -0 below +0


@pytest.mark.parametrize("t,lo,hi", [("f32", 0x00800000, 0x7F7FFFFF),
                                     ("bf16", 0x00800000, 0x7F7F0000),
                                     ("f16", 0x38800000, 0x477FE000),
                                     ("fp8", 0x3C800000, 0x43E00000)])
def test_min_normal_max_finite(t, lo, hi):
    x = np.zeros(4, np.int64)
    assert set(misc(4, x, t=t).tolist()) == {lo} and set(misc(5, x, t=t).tolist()) == {hi}
    assert bits(gc.min_normal(t)).tolist() == [lo] and bits(gc.max_finite(t)).tolist() == [hi]
    if t != "f32":  # the codes next to them are a subnormal and an infinity / NaN
        assert gc.encode_float(gc.max_finite(t), t) + 1 == gc.encode_float(f32(np.inf), t)
        assert (gc.encode_float(gc.min_normal(t), t) - 1) >> gc.FMT[t][1] == 0


def test_int_f32():
    v = np.concatenate([int_struct(np.random.default_rng(21), 200_000), int_special("i64")])
    same(misc(6, v), as_bits(gc.int_f32(v)), v)
    same(as_bits(gc.int_f32(v)), as_bits(gc.int_to_float(v, 0, Rnd.kHalfEven)), v)
    small = v[np.abs(v) < (1 << 24)]
    assert np.array_equal(gc.int_f32(small).astype(np.int64), small)


def test_neg_inf():
    assert set(misc(7, np.zeros(4, np.int64)).tolist()) == {0xFF800000}
    assert bits(gbits.neg_inf()).tolist() == [0xFF800000]


def test_acc_to_bf16():
    x = np.concatenate([F32S, SP, WIDE])
    want = gc.acc_to_bf16(x)
    same(misc(8, a=x), as_bits(want), x)
    same(as_bits(want), as_bits(gc.round_float(x, "bf16", Rnd.kHalfAway, kFtzIn)), x)
    assert bits(gc.acc_to_bf16(fbits(np.array([0x007FFFFF, 0x807FFFFF, 0x3F808000], np.uint32)))
                ).tolist() == [0, 0x80000000, 0x3F810000]


def test_wrap_mul_add64():
    v = np.concatenate([int_struct(np.random.default_rng(22), 200_000), int_special("i64")])
    w = v[(np.arange(v.size) * 7 + 3) % v.size]
    same(misc(9, v), gi.wrap_mul64(v, w), v)
    same(misc(10, v), gi.wrap_add64(v, w), v)
    ref_mul = [((int(p) * int(q) + 2**63) % 2**64) - 2**63 for p, q in zip(v[:5000], w[:5000])]
    ref_add = [((int(p) + int(q) + 2**63) % 2**64) - 2**63 for p, q in zip(v[:5000], w[:5000])]
    assert gi.wrap_mul64(v[:5000], w[:5000]).tolist() == ref_mul
    assert gi.wrap_add64(v[:5000], w[:5000]).tolist() == ref_add
    big = np.iinfo(np.int64).max
    assert gi.wrap_add64(big, 1) == -big - 1 and gi.wrap_mul64(big, 2) == -2


def test_div_q():
    for a, b in (cross(SP, SP), triples(300_000, 91)[:2], moderate(300_000, 92)[:2]):
        same(misc(11, a=a, b=b), as_bits(gm.div_q(a, b)), a)
    nan, inf = fbits(np.array([0x7F800001], np.uint32))[0], f32(np.inf)
    d = gm.div_q(np.array([0, inf, nan, 1, 1, -1, 1], f32),
                 np.array([0, -inf, 2, nan, 0, 0, inf], f32))
    assert bits(d).tolist() == [0xFFC00000, 0xFFC00000, 0x7FC00001, 0x7FC00001, 0x7F800000,
                                0xFF800000, 0]


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_code16(t):
    x = np.concatenate([F32S, SP, WIDE])
    want = gc.code16(x, t)
    same(misc(12, a=x, t=t), want.astype(I64), x)
    if t == "bf16":
        assert np.array_equal(want, (bits(x) >> 16).astype(np.uint16))
    else:
        assert np.array_equal(want, gc.f32_to_f16(x, Rnd.kHalfEven))


def test_misc_bad_args():
    from npu import raw_rc
    assert raw_rc("npu_test_misc", 13, None, None, None, None, 0, 0, 0, 0, None, I64(1)) == -1
    assert raw_rc("npu_test_misc", 0, None, None, None, None, 0, 10, 0, 0, None, I64(1)) == -1
