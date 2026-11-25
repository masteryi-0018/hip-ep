# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""reduction/layernorm.h entry points against golden_layernorm.py, bit for bit (bfp16: two
blocks with exponent 255 match)."""

import numpy as np
import pytest

import golden_layernorm as m
import testing
from golden_block_fp import decode_bfp16, encode_bfp16
from golden_float_cvt import f32_to_bf16
from golden_int_round import INT_RANGE, Rnd
from npu import (I64, OUT, NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, NPU_U8,
                 NPU_U16, call)

f32 = np.float32
RNG = np.random.default_rng(4242)
DT = {"bf16": NPU_BF16, "f16": NPU_F16, "fp8": NPU_FP8, "f32": NPU_F32,
      "i8": NPU_I8, "u8": NPU_U8, "i16": NPU_I16, "u16": NPU_U16}
STORE = {"bf16": np.uint16, "f16": np.uint16, "fp8": np.uint8, "f32": np.float32,
         "i8": np.int8, "u8": np.uint8, "i16": np.int16, "u16": np.uint16}
FLOAT_BITS = {"bf16": (np.uint16, 7, 8), "f16": (np.uint16, 10, 5), "fp8": (np.uint8, 3, 4)}
SCALES = (f32(0.0123), f32(1 / 3), f32(0.5), f32(7.25), f32(0.000731))
LN_EPS = (f32(1e-12), f32(1e-5), f32(0.1))


def run(name, *args, nbytes, dtype):
    return call(name, *args, out_nbytes=int(nbytes), out_dtype=dtype)


def same(got, want, t="int", what=""):
    """Bitwise; for bfp16, two blocks with exponent 255 match."""
    if t == "bfp16":
        g, w = np.array(got, np.uint8).reshape(-1, 9), np.array(want, np.uint8).reshape(-1, 9)
        both = (g[:, 0] == 255) & (w[:, 0] == 255)
        g[both], w[both] = 0, 0
        got, want = g.reshape(-1), w.reshape(-1)
    return testing.same(got, want, what)


def rows_f32(rows, C, nan=True):
    """(rows, C) fp32 values, one class per row."""
    x = np.empty((rows, C), f32)
    kind = RNG.integers(0, 9, rows)
    for r in range(rows):
        k = kind[r]
        sc = f32(np.exp2(RNG.integers(-6, 7)))
        if k == 0:      # wide random
            v = RNG.standard_normal(C) * sc
        elif k == 1:    # mean far above the spread
            v = RNG.choice([-1, 1]) * sc * 64 + RNG.standard_normal(C) * sc / 8
        elif k == 2:    # +-2^k lanes that cancel, plus small values
            big = np.exp2(RNG.integers(4, 12))
            v = np.where(np.arange(C) % 32 < 16, big, -big) + RNG.standard_normal(C) * sc / 16
            v[RNG.integers(0, C)] += big / 2
        elif k == 3:    # constant row
            v = np.full(C, RNG.standard_normal() * sc)
        elif k == 4:    # coarse grid: exact ties
            v = RNG.integers(-12, 13, C) * 0.375
        elif k == 5:    # signed zeros with a few values
            v = np.where(RNG.random(C) < 0.5, 0.0, -0.0)
            v[RNG.integers(0, C, 3)] = RNG.standard_normal(3)
        elif k == 6:    # tiny values (bf16 / f16 subnormal range)
            v = RNG.standard_normal(C) * np.exp2(RNG.integers(-136, -100))
        elif k == 7:    # large values
            v = RNG.standard_normal(C) * np.exp2(RNG.integers(20, 60))
        else:           # two-valued row
            v = np.where(RNG.random(C) < 0.5, 1.0, 3.0) * sc
        x[r] = v
    if nan and rows >= 64:
        x[RNG.integers(0, rows, 2), RNG.integers(0, C, 2)] = np.nan
    return x


def float_bits(x, t):
    """fp32 values -> bits of t; fp8 is clipped to +-448."""
    x = np.asarray(x, f32)
    if t == "fp8":
        x = np.where(np.isnan(x), x, np.clip(x, -448, 448)).astype(f32)
    return m.encode(x, t).astype(STORE[t])


def weights(C, t, lo=-4.0, hi=4.0):
    w = RNG.uniform(lo, hi, C).astype(f32)
    w[: C // 8] = (RNG.integers(-6, 7, C // 8) * 0.25).astype(f32)
    return float_bits(w, t)


def ints(rows, C, t):
    lo, hi = INT_RANGE[t]
    q = RNG.integers(lo, hi, (rows, C), endpoint=True)
    kind = RNG.integers(0, 4, rows)
    for r in np.nonzero(kind == 1)[0]:          # constant row
        q[r] = RNG.integers(lo, hi, endpoint=True)
    for r in np.nonzero(kind == 2)[0]:          # narrow row near one value
        c = RNG.integers(lo + 2, hi - 2, endpoint=True)
        q[r] = c + RNG.integers(-2, 3, C)
    return q.astype(STORE[t])


def nbytes(rows, C, t):
    return rows * C * np.dtype(STORE[t]).itemsize


@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("C", [64, 128, 256, 512, 768])
def test_layer_norm_bf16(t, C):
    if t == "fp8":
        x = RNG.integers(0, 256, (512, C)).astype(np.uint8)
        x = np.where((x & 0x7F) == 0x7F, np.uint8(0x38), x)
        x[:256] = float_bits(rows_f32(256, C, nan=False) * f32(4), "fp8")
        x[300, 5] = 0x7F
    else:
        x = float_bits(rows_f32(512, C), t)
    wt = "f16" if t == "fp8" else t
    al, be = weights(C, wt), weights(C, wt)
    got = run("npu_layer_norm_bf16", x, DT[t], al, be, OUT, I64(x.shape[0]), C, 0, 1e-12,
              nbytes=x.size * x.itemsize, dtype=STORE[t])
    same(got, m.layer_norm_bf16(x, t, al, be), t, f"layer_norm_bf16 {t} C{C}")


@pytest.mark.parametrize("C,c_real,eps", [(256, 96, 1e-5), (384, 257, 1e-3), (256, 20, 1e-5),
                                          (256, 256, 1e-5), (64, 1, 0.1), (512, 333, 1e-12),
                                          (128, 64, 1e-6)])
def test_layer_norm_bf16_real(C, c_real, eps):
    """Statistics over the first c_real channels; the padding holds large values,
    infinities and NaNs that must stay out of them."""
    rows = 256
    xf = rows_f32(rows, C)
    pad = C - c_real
    xf[:, c_real:] = RNG.uniform(-1000.0, 1000.0, (rows, pad)).astype(f32)
    if pad:
        xf[3, c_real] = np.inf
        xf[7, C - 1] = np.nan
    x = float_bits(xf, "bf16")
    al, be = weights(C, "bf16"), weights(C, "bf16")
    if C > 128:  # zero weight padding, as the caller supplies it
        al[c_real:] = 0
        be[c_real:] = 0
    got = run("npu_layer_norm_bf16", x, NPU_BF16, al, be, OUT, I64(rows), C, c_real, eps,
              nbytes=x.size * 2, dtype=np.uint16)
    want = m.layer_norm_bf16(x, "bf16", al, be, c_real, eps)
    same(got, want, "bf16", f"layer_norm_bf16 C{C} real{c_real} eps{eps}")


@pytest.mark.parametrize("C", [64, 256, 512])
@pytest.mark.parametrize("io", ["bfp16", "bf16_bfp16"])
def test_layer_norm_bfp16(C, io):
    rows = 384
    xf = rows_f32(rows, C, nan=False)
    al, be = weights(C, "bf16"), weights(C, "bf16")
    if io == "bfp16":
        x = encode_bfp16(xf.reshape(-1), Rnd.kHalfEven)
        got = run("npu_layer_norm_bfp16", x, al, be, OUT, I64(rows), C,
                  nbytes=rows * C // 8 * 9, dtype=np.uint8)
        want = m.layer_norm_bfp16(x, al, be, C)
    else:
        x = float_bits(xf, "bf16")
        got = run("npu_layer_norm_bf16_bfp16", x, al, be, OUT, I64(rows), C,
                  nbytes=rows * C // 8 * 9, dtype=np.uint8)
        want = m.layer_norm_bf16_bfp16(x, al, be)
    same(got, want, "bfp16", f"layer_norm {io} C{C}")


def wide_rows(rows, C, span):
    """bfp16 rows: in every 64-element group blocks 0 / 1 hold opposite large values (exponent
    140), the other blocks exponents 18 .. span below."""
    b = RNG.integers(-127, 128, (rows, C // 64, 8, 9))
    b[:, :, 1, 1:] = -b[:, :, 0, 1:]
    b[:, :, :2, 0] = 140
    b[:, :, 2:, 0] = 140 - RNG.integers(18, span + 1, (rows, C // 64, 6))
    return b.astype(np.int8).view(np.uint8).reshape(-1)


def exact_cast(blk):
    return f32_to_bf16(decode_bfp16(blk), Rnd.kHalfEven)


@pytest.mark.parametrize("span", [20, 30])
def test_bfp16_wide_groups(span, monkeypatch):
    """Exponents spread over more than 17 / 25 inside a 64-element group: the identity-multiply
    decode rounds the small blocks."""
    rows, C = 64, 128
    x = wide_rows(rows, C, span)
    al, be = weights(C, "bf16"), np.zeros(C, np.uint16)
    gm = weights(C, "bf16")
    inv, eps = f32(1.0 / C), f32(1e-5)

    def refs():
        return m.layer_norm_bfp16(x, al, be, C), m.rmsnorm_bfp16(x, gm, C, inv, eps)
    ln, rms = refs()
    got = run("npu_layer_norm_bfp16", x, al, be, OUT, I64(rows), C,
              nbytes=rows * C // 8 * 9, dtype=np.uint8)
    same(got, ln, "bfp16", f"layer_norm wide{span}")
    got = run("npu_rmsnorm_bfp16", x, gm, OUT, I64(rows), C, float(inv), float(eps),
              nbytes=rows * C // 8 * 9, dtype=np.uint8)
    same(got, rms, "bfp16", f"rmsnorm wide{span}")
    with monkeypatch.context() as mp:
        mp.setattr(m, "decode_bfp16_bf16", exact_cast)
        old_ln, old_rms = refs()
    assert (old_ln != ln).any() and (old_rms != rms).any()


@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("C", [64, 96, 384, 512])
@pytest.mark.parametrize("eps", [f32(1e-5), f32(1e-6), f32(0.37), f32(0.0)])
def test_rmsnorm(t, C, eps):
    rows = 256
    xf = rows_f32(rows, C, nan=t != "fp8")
    if t == "fp8":
        xf = np.clip(xf, -448, 448)
    x = float_bits(xf, t)
    wt = "f16" if t == "fp8" else t
    gm = weights(C, wt)
    inv = f32(1.0 / C)
    got = run("npu_rmsnorm_bf16", x, DT[t], gm, OUT, I64(rows), C, float(inv), float(eps),
              nbytes=nbytes(rows, C, t), dtype=STORE[t])
    same(got, m.rmsnorm_bf16(x, t, gm, inv, eps), t, f"rmsnorm {t} C{C}")
    if t == "bf16":
        xb = encode_bfp16(xf.reshape(-1), Rnd.kHalfEven)
        got = run("npu_rmsnorm_bfp16", xb, gm, OUT, I64(rows), C, float(inv), float(eps),
                  nbytes=rows * C // 8 * 9, dtype=np.uint8)
        same(got, m.rmsnorm_bfp16(xb, gm, C, inv, eps), "bfp16", f"rmsnorm_bfp16 C{C}")


@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("C", [64, 256, 512])
def test_l2_norm_bf16(t, C):
    rows = 512
    xf = rows_f32(rows, C)
    x = float_bits(xf, t)
    wt = "f16" if t == "fp8" else t
    al, be = weights(C, wt), weights(C, wt)
    got = run("npu_l2_norm_bf16", x, DT[t], al, be, OUT, I64(rows), C,
              nbytes=nbytes(rows, C, t), dtype=STORE[t])
    same(got, m.l2_norm_bf16(x, t, al, be), t, f"l2_norm {t} C{C}")


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("C", [64, 256])
def test_qdq_l2norm(bits, affine, C):
    rows = 512
    types = ["i8", "u8"] if bits == 8 else ["i16", "u16"]
    al = weights(C, "bf16") if affine else None
    be = weights(C, "bf16") if affine else None
    for k, (it, ot) in enumerate((a, b) for a in types for b in types):
        q = ints(rows, C, it)
        s, zp = SCALES[k % len(SCALES)], f32(RNG.integers(-20, 140))
        qs, qz = f32(1 / SCALES[(k + 2) % len(SCALES)]), f32(RNG.integers(-10, 100))
        got = run(f"npu_qdq_l2norm_a{bits}", q, DT[it], al, be, OUT, DT[ot], I64(rows), C,
                  float(zp), float(s), float(qz), float(qs),
                  nbytes=nbytes(rows, C, ot), dtype=STORE[ot])
        want = getattr(m, f"qdq_l2norm_a{bits}")(
            q, None if al is None else m.decode(al, "bf16"),
            None if be is None else m.decode(be, "bf16"), zp, s, qz, qs, ot)
        same(got, want, "int", f"qdq_l2norm a{bits} {it}->{ot}")


STATS = {"fused": 0, "batched": 1}


@pytest.mark.parametrize("t", ["bf16", "f16", "fp8"])
@pytest.mark.parametrize("C", [48, 64, 96, 268, 512])
@pytest.mark.parametrize("stats", ["fused", "batched"])
@pytest.mark.parametrize("entry", ["npu_add_layernorm_bf16", "npu_add_layernorm_const_bf16"])
def test_add_layernorm(t, C, stats, entry):
    if stats == "batched" and C not in (64, 96):
        pytest.skip("the three-pass form exists for C = 64 / 96 only")
    if t == "fp8" and C % 8:
        pytest.skip("fp8 C % 8")
    rows = 256
    a = float_bits(rows_f32(rows, C) / 2, t)
    b = float_bits(rows_f32(rows, C) / 2, t)
    wt = "f16" if t == "fp8" else t
    al, be = weights(C, wt), weights(C, wt)
    eps = LN_EPS[C % 3]
    got = run(entry, a, b, DT[t], al, be, OUT, I64(rows), C, float(eps), STATS[stats],
              nbytes=nbytes(rows, C, t), dtype=STORE[t])
    want = getattr(m, entry[4:])(a, b, t, al, be, eps, stats)
    same(got, want, t, f"{entry} {t} C{C} {stats}")


QT = ["i8", "u8", "i16", "u16"]


@pytest.mark.parametrize("compute", ["bf16", "f16"])
@pytest.mark.parametrize("C", [64, 96, 300, 512])
@pytest.mark.parametrize("stats", ["fused", "batched"])
def test_qdq_add_layernorm(compute, C, stats):
    if stats == "batched" and C not in (64, 96):
        pytest.skip("the three-pass form exists for C = 64 / 96 only")
    rows = 192
    al, be = weights(C, compute), weights(C, compute)
    for k, it in enumerate(QT):
        ot = QT[(k + 1) % 4]
        a, b = ints(rows, C, it), ints(rows, C, it)
        da = (f32(RNG.integers(-30, 30)), SCALES[k])          # (zp, scale)
        db = (f32(RNG.integers(-30, 30)), SCALES[(k + 1) % 5])
        q = (f32(RNG.integers(0, 50)), f32(1 / SCALES[(k + 3) % 5]))
        eps = LN_EPS[k % 3]
        alf, bef = m.decode(al, compute), m.decode(be, compute)
        got = run("npu_qdq_add_layernorm", a, b, DT[it], al, be, DT[compute], OUT, DT[ot],
                  I64(rows), C, *map(float, (*da, *q, *db, eps)), STATS[stats],
                  nbytes=nbytes(rows, C, ot), dtype=STORE[ot])
        want = m.qdq_add_layernorm(a, b, compute, alf, bef, *da, *q, *db, eps, stats, ot)
        same(got, want, "int", f"qdq_add_layernorm {it}->{ot} {compute}")

        # the const form: b is a compute-dtype tensor
        bc = float_bits(rows_f32(rows, C, nan=False) / 4, compute)
        got = run("npu_qdq_add_layernorm_const", a, DT[it], bc, DT[compute], al, be, OUT,
                  DT[ot], I64(rows), C, *map(float, (*da, *q, eps)), STATS[stats],
                  nbytes=nbytes(rows, C, ot), dtype=STORE[ot])
        want = m.qdq_add_layernorm_const(a, m.decode(bc, compute), compute, alf, bef, *da, *q,
                                         eps, stats, ot)
        same(got, want, "int", f"qdq_add_layernorm_const {it}->{ot} {compute}")


@pytest.mark.parametrize("compute", ["bf16", "f16", "f32"])
@pytest.mark.parametrize("C", [64, 96, 384])
@pytest.mark.parametrize("stats", ["fused", "batched"])
@pytest.mark.parametrize("float_a", [False, True])
def test_qdq_add_add_const_layernorm(compute, C, stats, float_a):
    if stats == "batched" and C not in (64, 96):
        pytest.skip("the three-pass form exists for C = 64 / 96 only")
    rows = 192
    al, be = weights(C, compute), weights(C, compute)
    kc = float_bits(RNG.standard_normal(C).astype(f32) * f32(3), compute)
    for k, bt in enumerate(QT):
        ot = QT[(k + 2) % 4]
        at = compute if float_a else bt
        a = float_bits(rows_f32(rows, C, nan=False), compute) if float_a else ints(rows, C, bt)
        b = ints(rows, C, bt)
        da = (f32(RNG.integers(-30, 30)), SCALES[(k + 2) % 5])  # (zp, scale)
        db = (f32(RNG.integers(-30, 30)), SCALES[k])
        q = (f32(RNG.integers(0, 50)), f32(1 / SCALES[(k + 1) % 5]))
        eps = LN_EPS[(k + 1) % 3]
        got = run("npu_qdq_add_add_const_layernorm", a, DT[at], b, DT[bt], kc, DT[compute], al,
                  be, OUT, DT[ot], I64(rows), C, *map(float, (*da, *q, *db, eps)),
                  STATS[stats], nbytes=nbytes(rows, C, ot), dtype=STORE[ot])
        av = m.decode(a, compute) if float_a else a
        want = m.qdq_add_add_const_layernorm(av, at, b, m.decode(kc, compute), compute,
                                             m.decode(al, compute), m.decode(be, compute), *da,
                                             *q, *db, eps, stats, ot)
        same(got, want, "int", f"qdq_add_add_const {at}+{bt}->{ot} {compute}")


QLN = {"fused": 0, "batched": 1, "c8_fused": 2, "c8_batched": 3, "integer": 4}


def folded(C, ofm_scale, ofm_zp):
    """Folded fp32 weights: alpha / s, beta / s + zp (double, then fp32)."""
    alpha = RNG.uniform(-3, 3, C).astype(f32)
    beta = RNG.uniform(-2, 2, C).astype(f32)
    qs = np.float64(f32(1 / ofm_scale))
    a = (alpha.astype(np.float64) * qs).astype(f32)
    b = (beta.astype(np.float64) * qs + np.float64(f32(ofm_zp))).astype(f32)
    return a, b


@pytest.mark.parametrize("variant,C", [("fused", 48), ("fused", 64), ("fused", 300),
                                       ("fused", 768), ("batched", 64), ("batched", 96),
                                       ("c8_fused", 8), ("c8_fused", 16), ("c8_fused", 24),
                                       ("c8_batched", 8),
                                       ("c8_batched", 16)])
@pytest.mark.parametrize("compute", ["bf16", "f16", "f32"])
def test_qdq_layernorm(variant, C, compute):
    rows = 256
    for k, it in enumerate(QT):
        ot = QT[(k + 3) % 4]
        q = ints(rows, C, it)
        s, zp = SCALES[k], f32(RNG.integers(-30, 30))
        lo, hi = INT_RANGE[ot]
        osc, ozp = [0.02, 0.37, 0.004, 1.3][k], float((lo + hi) // 2 + RNG.integers(-5, 5))
        a, b = folded(C, osc, ozp)
        got = run("npu_qdq_layernorm", q, DT[it], a, b, OUT, DT[ot], I64(rows), C, DT[compute],
                  float(zp), float(s), QLN[variant], nbytes=nbytes(rows, C, ot),
                  dtype=STORE[ot])
        want = m.qdq_layernorm(q, a, b, ot, compute, zp, s, variant)
        same(got, want, "int", f"qdq_layernorm {variant} {it}->{ot} {compute} C{C}")


@pytest.mark.parametrize("C", [64, 96, 192, 256, 384, 768])
def test_qdq_layernorm_integer(C):
    rows = 512
    for k, (it, ot) in enumerate((a, b) for a in ("i8", "u8") for b in ("i8", "u8")):
        q = ints(rows, C, it)
        s = SCALES[k]
        lo, hi = INT_RANGE[ot]
        a, b = folded(C, [0.05, 0.11, 0.013, 0.9][k], float((lo + hi) // 2))
        got = run("npu_qdq_layernorm", q, DT[it], a, b, OUT, DT[ot], I64(rows), C, NPU_F16,
                  0.0, float(s), QLN["integer"], nbytes=nbytes(rows, C, ot), dtype=STORE[ot])
        same(got, m.qdq_layernorm(q, a, b, ot, "f16", 0.0, s, "integer"), "int",
             f"qdq_layernorm integer {it}->{ot} C{C}")


@pytest.mark.parametrize("variant,C", [("fused", 32), ("fused", 64), ("fused", 192),
                                       ("fused", 768), ("batched", 64), ("batched", 96)])
def test_layer_norm_a8_bfp16(variant, C):
    rows = 256
    for k, it in enumerate(("i8", "u8")):
        q = ints(rows, C, it)
        s, zp = SCALES[2 * k + 1], f32(RNG.integers(-30, 30) + (128 if it == "u8" else 0))
        al, be = weights(C, "f16"), weights(C, "f16", -2.0, 2.0)
        got = run("npu_layer_norm_a8_bfp16", q, DT[it], al, be, OUT, I64(rows), C, float(zp),
                  float(s), QLN[variant], nbytes=rows * C // 8 * 9, dtype=np.uint8)
        want = m.layer_norm_a8_bfp16(q, m.decode(al, "f16"), m.decode(be, "f16"), zp, s,
                                     variant)
        same(got, want, "bfp16", f"layer_norm_a8_bfp16 {variant} {it} C{C}")


ENTRIES = ["npu_layer_norm_bf16", "npu_layer_norm_bfp16", "npu_layer_norm_bf16_bfp16",
           "npu_rmsnorm_bf16", "npu_rmsnorm_bfp16", "npu_l2_norm_bf16", "npu_qdq_l2norm_a8",
           "npu_qdq_l2norm_a16", "npu_add_layernorm_bf16", "npu_add_layernorm_const_bf16",
           "npu_qdq_add_layernorm", "npu_qdq_add_layernorm_const",
           "npu_qdq_add_add_const_layernorm", "npu_qdq_layernorm", "npu_layer_norm_a8_bfp16"]


def test_bad_args():
    """Every entry rejects an argument its kernel cannot run; all other arguments are valid
    (real tensors), so each case isolates one check."""
    R, C = 2, 64
    h = np.zeros((R, C), np.uint16)                 # 16-bit floats
    b8 = np.zeros((R, C), np.uint8)                 # 8-bit codes / fp8
    i16 = np.zeros((R, C), np.int16)
    w = np.zeros(C, np.uint16)
    wf = np.zeros(C, np.float32)
    bfp = np.zeros(R * C // 8 * 9, np.uint8)
    q = (0.0, 1.0, 0.0, 1.0)
    BF, F16, I8, I16 = NPU_BF16, NPU_F16, NPU_I8, NPU_I16
    bad = [
        ("npu_layer_norm_bf16", h, I8, w, w, OUT, I64(R), C, 0, 1e-12),              # type
        ("npu_layer_norm_bf16", h, BF, w, w, OUT, I64(R), 48, 0, 1e-12),             # C % 32
        ("npu_layer_norm_bf16", h, BF, w, w, OUT, I64(R), C, 65, 1e-5),              # C_real
        ("npu_layer_norm_bf16", h, BF, w, w, OUT, I64(R), C, -1, 1e-5),
        ("npu_layer_norm_bf16", h, F16, w, w, OUT, I64(R), C, 32, 1e-5),             # bf16 only
        ("npu_layer_norm_bf16", h, BF, w, w, OUT, I64(R), C, 0, 1e-5),               # eps
        ("npu_layer_norm_bf16", b8, NPU_FP8, w, w, OUT, I64(R), 32, 0, 1e-12),       # fp8 C % 64
        ("npu_layer_norm_bf16", h, BF, None, w, OUT, I64(R), C, 0, 1e-12),
        ("npu_layer_norm_bfp16", bfp, w, w, OUT, I64(R), 48),
        ("npu_layer_norm_bfp16", bfp, w, None, OUT, I64(R), C),
        ("npu_layer_norm_bf16_bfp16", h, w, w, OUT, I64(R), 40),
        ("npu_rmsnorm_bf16", h, NPU_F32, w, OUT, I64(R), C, 0.1, 0.1),
        ("npu_rmsnorm_bf16", h, BF, None, OUT, I64(R), C, 0.1, 0.1),
        ("npu_rmsnorm_bfp16", bfp, w, OUT, I64(R), 40, 0.1, 0.1),
        ("npu_l2_norm_bf16", b8, NPU_FP8, w, w, OUT, I64(R), 96),
        ("npu_l2_norm_bf16", h, BF, w, w, OUT, I64(R), 48),
        ("npu_qdq_l2norm_a8", i16, I16, w, w, OUT, I8, I64(R), C, *q),               # a8 types
        ("npu_qdq_l2norm_a8", b8, I8, w, None, OUT, I8, I64(R), C, *q),              # one weight
        ("npu_qdq_l2norm_a16", i16, I16, w, w, OUT, I8, I64(R), C, *q),
        ("npu_add_layernorm_bf16", h, h, BF, w, w, OUT, I64(R), 128, 0.1, 1),        # batched C
        ("npu_add_layernorm_bf16", h, h, BF, w, w, OUT, I64(R), C, 0.1, 7),          # stats
        ("npu_add_layernorm_const_bf16", h, None, BF, w, w, OUT, I64(R), C, 0.1, 0),
        ("npu_add_layernorm_const_bf16", h, h, NPU_F32, w, w, OUT, I64(R), C, 0.1, 0),
        ("npu_qdq_add_layernorm", b8, b8, I8, w, w, NPU_F32, OUT, I8, I64(R), C, *q, 0.0, 1.0,
         0.1, 0),                                                                     # compute
        ("npu_qdq_add_layernorm", b8, b8, NPU_FP8, w, w, BF, OUT, I8, I64(R), C, *q, 0.0, 1.0,
         0.1, 0),
        ("npu_qdq_add_layernorm_const", b8, I8, h, NPU_F32, w, w, OUT, I8, I64(R), C, *q, 0.1,
         0),
        ("npu_qdq_add_layernorm_const", b8, I8, h, BF, w, w, OUT, I8, I64(R), C, *q, 0.1, 3),
        ("npu_qdq_add_add_const_layernorm", i16, I16, b8, I8, w, BF, w, w, OUT, I8, I64(R), C,
         *q, 0.0, 1.0, 0.1, 0),                                                       # a != b
        ("npu_qdq_add_add_const_layernorm", b8, NPU_U8, b8, I8, w, BF, w, w, OUT, I8, I64(R),
         C, *q, 0.0, 1.0, 0.1, 0),
        ("npu_qdq_add_add_const_layernorm", h, F16, b8, I8, w, BF, w, w, OUT, I8, I64(R), C,
         *q, 0.0, 1.0, 0.1, 0),                                                       # float a
        ("npu_qdq_layernorm", b8, I8, wf, wf, OUT, I8, I64(R), C, NPU_F16, 0.0, 1.0, 9),
        ("npu_qdq_layernorm", i16, I16, wf, wf, OUT, I8, I64(R), C, NPU_F16, 0.0, 1.0, 4),
        ("npu_qdq_layernorm", b8, I8, wf, wf, OUT, I8, I64(R), 128 + 32, NPU_F16, 0.0, 1.0, 4),
        ("npu_qdq_layernorm", b8, I8, wf, wf, OUT, I8, I64(R), 40, NPU_F16, 0.0, 1.0, 2),
        ("npu_qdq_layernorm", b8, I8, wf, wf, OUT, I8, I64(R), C, NPU_FP8, 0.0, 1.0, 0),
        ("npu_layer_norm_a8_bfp16", i16, I16, w, w, OUT, I64(R), C, 0.0, 1.0, 0),
        ("npu_layer_norm_a8_bfp16", b8, I8, w, w, OUT, I64(R), 128, 0.0, 1.0, 1),
        ("npu_layer_norm_a8_bfp16", b8, I8, w, w, OUT, I64(R), C, 0.0, 1.0, 2),
        # negative sizes; with 0 rows every other argument is still checked
        ("npu_layer_norm_bf16", h, BF, w, w, OUT, I64(-1), C, 0, 1e-12),
        ("npu_qdq_add_layernorm", b8, b8, I8, w, w, BF, OUT, I8, I64(-2), C, *q, 0.0, 1.0, 0.1,
         0),
        ("npu_rmsnorm_bf16", h, BF, None, OUT, I64(0), C, 0.1, 0.1),
        ("npu_qdq_layernorm", None, I8, wf, wf, OUT, I8, I64(0), C, NPU_F16, 0.0, 1.0, 0),
        ("npu_qdq_layernorm", i16, I16, wf, wf, OUT, I16, I64(0), C, NPU_F16, 0.0, 1.0, 4),
        ("npu_layer_norm_bf16", h, BF, w, w, OUT, I64(0), 0, 0, 1e-12),            # C >= 1
    ]
    for name, *args in bad:
        assert testing.rejected(name, *args, nbytes=4 * R * 160), (name, args)
    covered = {b[0] for b in bad}
    assert covered == set(ENTRIES), set(ENTRIES) - covered
    # empty tensors are a no-op
    x = np.zeros(1, np.uint16)
    assert run("npu_layer_norm_bf16", x, NPU_BF16, x, x, OUT, I64(0), 64, 0, 1e-12, nbytes=2,
               dtype=np.uint16)[0] == 0xA5A5
