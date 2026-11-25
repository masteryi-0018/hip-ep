# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Special values in every nonlinear entry: NaN / inf / zero / subnormal / extreme inputs, and the
same values as float coefficients and table words, GPU against the golden functions, by bits.

Each case sweeps one float parameter at a time over the special fp32 values, keeping the others
at a sane default, then sets all of them to each special at once."""

import numpy as np
import pytest

import golden_expsqrt as ge
import golden_gelu as gg
import golden_silu as gsi
import golden_sigmoid as gsg
import golden_sincos as gsc
from golden_int_round import INT_RANGE
from golden_nlf import tanh16
from npu import I64, NPU_BF16, NPU_I8, OUT, call, raw_rc
from testing import ALL8, ALL16, F16T, INT
from testing import same_bytes

F32 = np.float32
RNG = np.random.default_rng(20261119)


def _f(bits):
    return np.uint32(bits).view(F32)


# +-0, +-inf, quiet NaNs (default, negative with payload, payload), subnormals, extremes.
SPECIAL = [("+0", _f(0)), ("-0", _f(0x80000000)), ("+inf", _f(0x7F800000)),
           ("-inf", _f(0xFF800000)), ("qnan", _f(0x7FC00000)), ("-qnan1", _f(0xFFC00001)),
           ("qnan_pl", _f(0x7FC12345)), ("minsub", _f(1)), ("-maxsub", _f(0x807FFFFF)),
           ("maxfin", _f(0x7F7FFFFF)), ("-maxfin", _f(0xFF7FFFFF)), ("minnorm", _f(0x00800000)),
           ("1e-30", F32(1e-30)), ("-1e30", F32(-1e30))]

CODES16 = {
    "bf16": [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFF81, 0x7FFF, 0xFFFF,
             0x7FC5, 0xFFC5, 0x0001, 0x007F, 0x8001, 0x807F, 0x0080, 0x8080, 0x7F7F, 0xFF7F,
             0x3F80, 0xBF80, 0x4000, 0xC000],
    "f16": [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0xFC01, 0x7FFF, 0xFFFF,
            0x7E05, 0xFE05, 0x0001, 0x03FF, 0x8001, 0x83FF, 0x0400, 0x8400, 0x7BFF, 0xFBFF,
            0x3C00, 0xBC00, 0x4000, 0xC000]}


def x16(t):
    rnd = RNG.integers(0, 1 << 16, 3000).astype(np.uint16)
    return np.concatenate([np.array(CODES16[t], np.uint16), rnd])


def xfp8():
    return np.concatenate([ALL8, RNG.permutation(ALL8), np.repeat(np.uint8(0x7F), 8),
                           np.repeat(np.uint8(0xFF), 8), np.array([0x38, 0x7F, 0x80], np.uint8)])


def xint(t):
    dt = INT[t][1]
    info = np.iinfo(dt)
    if info.bits == 8:
        return np.arange(info.min, info.max + 1, dtype=np.int64).astype(dt)
    ext = np.array([info.min, info.min + 1, -1, 0, 1, info.max - 1, info.max], np.int64)
    return np.concatenate([ext, RNG.integers(info.min, info.max + 1, 2500)]).astype(dt)


def xbfp16(nblk=2048):
    e = RNG.integers(0, 256, nblk)
    e[::5] = RNG.choice([0, 1, 2, 126, 127, 128, 133, 253, 254, 255], len(e[::5]))
    m = RNG.integers(-128, 128, (nblk, 8))
    m[::7] = 0
    m[RNG.random((nblk, 8)) < 0.05] = -128
    m[RNG.random((nblk, 8)) < 0.05] = 127
    return np.concatenate([e[:, None], m & 0xFF], axis=1).astype(np.uint8).reshape(-1)


def fl(*v):
    return [F32(a) for a in v]


class Case:
    """gpu(x, p) / gold(x, p) take a parameter list p of fp32 scalars; ints (not swept as fp32)
    are closed over."""

    def __init__(self, name, x, gpu, gold, defaults):
        self.name, self.x, self.gpu, self.gold, self.defaults = name, x, gpu, gold, defaults

    def __repr__(self):
        return self.name


def run_case(c, params):
    got = np.ascontiguousarray(c.gpu(c.x, [float(v) for v in params]))
    want = np.ascontiguousarray(np.asarray(c.gold(c.x, [F32(v) for v in params])))
    got = got.reshape(-1)
    want = want.astype(got.dtype).reshape(-1)
    gb, wb = got.view(f"u{got.itemsize}"), want.view(f"u{want.itemsize}")
    diff = gb != wb
    bad = np.nonzero(diff)[0]
    if bad.size:
        i = bad[0]
        return (f"{bad.size}/{got.size} differ, first at {i} (x={c.x[i // max(1, got.size // c.x.size)]}): "
                f"got {int(gb[i]):#x} want {int(wb[i]):#x}")
    return None


def sweep(c):
    fails = []
    err = run_case(c, c.defaults)
    if err:
        fails.append(f"default: {err}")
    for k in range(len(c.defaults)):
        for lab, v in SPECIAL:
            p = list(c.defaults)
            p[k] = v
            err = run_case(c, p)
            if err:
                fails.append(f"param {k} = {lab}: {err}")
    for lab, v in SPECIAL:
        if len(c.defaults) > 1:
            err = run_case(c, [v] * len(c.defaults))
            if err:
                fails.append(f"all params = {lab}: {err}")
    assert not fails, f"{c.name}\n" + "\n".join(fails[:12]) + f"\n({len(fails)} failing)"


CASES = []


def add(*a):
    CASES.append(Case(*a))


# ---------------------------------------------------------------- expsqrt

for t in ("bf16", "f16"):
    x = x16(t)
    n = x.size

    def go16(name, t=t, x=x):
        return lambda xx, p: call(name, xx, F16T[t], OUT, I64(len(xx)), *p, out_nbytes=2 * len(xx),
                                  out_dtype=np.uint16)

    add(f"exp16_{t}", x, go16("npu_elementwise_exp_bf16"),
        lambda xx, p, t=t: ge.elementwise_exp_bf16(xx, t), [])
    add(f"elu16_{t}", x, go16("npu_elementwise_elu_bf16"),
        lambda xx, p, t=t: ge.elementwise_elu_bf16(xx, t, p[0]), fl(1.0))
    add(f"hardsigmoid16_{t}", x, go16("npu_elementwise_hardsigmoid_bf16"),
        lambda xx, p, t=t: ge.elementwise_hardsigmoid_bf16(xx, t, *p), fl(0.2, 0.5))
    add(f"sqrt16_{t}", x, go16("npu_sqrt_bf16"), lambda xx, p, t=t: ge.sqrt_bf16(xx, t), [])


def go8(name):
    return lambda xx, p: call(name, xx, OUT, I64(len(xx)), *p, out_nbytes=len(xx),
                              out_dtype=np.uint8)


X8 = xfp8()
add("exp_fp8", X8, go8("npu_elementwise_exp_fp8"), lambda xx, p: ge.elementwise_exp_fp8(xx, *p),
    fl(*ge.coeff_exp_fp8(1.0, 1.0)))
add("elu_fp8", X8, go8("npu_elementwise_elu_fp8"), lambda xx, p: ge.elementwise_elu_fp8(xx, *p),
    fl(*ge.coeff_elu_fp8(1.0, 1.0, 1.0)))
add("hardsigmoid_fp8", X8, go8("npu_elementwise_hardsigmoid_fp8"),
    lambda xx, p: ge.elementwise_hardsigmoid_fp8(xx, *p),
    fl(*ge.coeff_hardsigmoid_fp8(1.0, 1.0, 0.2, 0.5)))
add("sqrt_fp8", X8, go8("npu_elementwise_sqrt_fp8"), lambda xx, p: ge.elementwise_sqrt_fp8(xx, *p),
    fl(ge.coeff_sqrt_fp8(1.0, 1.0)))


def goq(name, it, ot, pre=()):
    """Integer in / out: args (x, in_type, OUT, out_type, n, <pre>, *p)."""
    return lambda xx, p: call(name, xx, INT[it][0], OUT, INT[ot][0], I64(len(xx)), *pre,
                              *p, out_nbytes=len(xx) * np.dtype(INT[ot][1]).itemsize,
                              out_dtype=INT[ot][1])


QTYPES = {8: [("i8", "u8"), ("u8", "i8")], 16: [("i16", "u16"), ("u16", "i16")]}

for bits in (8, 16):
    for it, ot in QTYPES[bits]:
        x = xint(it)
        tag = f"a{bits}_{it}_{ot}"
        add(f"exp_{tag}", x, goq(f"npu_elementwise_exp_a{bits}", it, ot),
            lambda xx, p, ot=ot: ge.elementwise_exp_a8(xx, ot, *p),
            fl(*ge.coeff_exp_qdq(0.024, 3, 0.1, 5)))
        add(f"elu_{tag}", x, goq(f"npu_elementwise_elu_a{bits}", it, ot),
            lambda xx, p, ot=ot: ge.elementwise_elu_a8(xx, ot, *p),
            fl(*ge.coeff_elu_qdq(0.024, 3, 0.1, 5, 0.8)))
        add(f"hardsigmoid_{tag}", x,
            lambda xx, p, bits=bits, it=it, ot=ot: call(
                f"npu_hardsigmoid_a{bits}", xx, INT[it][0], OUT, INT[ot][0], I64(len(xx)),
                p[0], p[1], 0, 100, out_nbytes=len(xx) * np.dtype(INT[ot][1]).itemsize,
                out_dtype=INT[ot][1]),
            lambda xx, p, ot=ot: ge.hardsigmoid_a8(xx, ot, p[0], p[1], 0, 100), fl(0.05, 3.0))
        add(f"qdq_sqrt_{tag}", x, goq(f"npu_qdq_sqrt_a{bits}", it, ot),
            lambda xx, p, ot=ot: ge.qdq_sqrt_a8(xx, ot, *p), fl(17, 0.0371, 3, 1 / 0.00731))

# ---------------------------------------------------------------- gelu
for t in ("bf16", "f16"):
    x = x16(t)
    add(f"gelu16_{t}", x, lambda xx, p, t=t: call(
        "npu_gelu_bf16", xx, F16T[t], OUT, I64(len(xx)), out_nbytes=2 * len(xx),
        out_dtype=np.uint16), lambda xx, p, t=t: gg.gelu_bf16(xx, t), [])
    add(f"erf16_{t}", x, lambda xx, p, t=t: call(
        "npu_elementwise_erf_bf16", xx, F16T[t], OUT, I64(len(xx)), out_nbytes=2 * len(xx),
        out_dtype=np.uint16), lambda xx, p, t=t: gg.elementwise_erf_bf16(xx, t), [])
    add(f"gelu_fp8_{t}", X8, lambda xx, p, t=t: call(
        "npu_elementwise_gelu_fp8", xx, OUT, I64(len(xx)), F16T[t], *p, out_nbytes=len(xx),
        out_dtype=np.uint8), lambda xx, p, t=t: gg.elementwise_gelu_fp8(xx, t, *p), fl(1.0, 1.0))
BFP = xbfp16()
add("gelu_bfp16", BFP, lambda xx, p: call("npu_gelu_bfp16", xx, OUT, I64(len(xx) // 9 * 8),
                                          out_nbytes=len(xx), out_dtype=np.uint8),
    lambda xx, p: gg.gelu_bfp16(xx), [])
add("erf_fp8", X8, go8("npu_elementwise_erf_fp8"), lambda xx, p: gg.elementwise_erf_fp8(xx, *p),
    fl(*gg.erf_fp8_coeffs(1.0, 1.0)))

for bits in (8, 16):
    for it, ot in QTYPES[bits]:
        x = xint(it)
        tag = f"a{bits}_{it}_{ot}"
        zi, zo, ka, c, kd = gg.qdq_gelu_coeffs(0.047, 7, 0.019, 3)
        add(f"qdq_gelu_{tag}", x,
            lambda xx, p, bits=bits, it=it, ot=ot, zi=zi: call(
                f"npu_qdq_gelu_exp2_a{bits}", xx, INT[it][0], OUT, INT[ot][0], I64(len(xx)),
                zi, *p, out_nbytes=len(xx) * np.dtype(INT[ot][1]).itemsize,
                out_dtype=INT[ot][1]),
            lambda xx, p, ot=ot, zi=zi: gg.qdq_gelu_exp2_a8(xx, zi, *p, ot), fl(zo, ka, c, kd))
        add(f"erf_qdq_{tag}", x, goq(f"npu_elementwise_erf_a{bits}", it, ot),
            lambda xx, p, ot=ot: gg.elementwise_erf_a8(xx, *p, ot),
            fl(*gg.erf_qdq_coeffs(0.024, 0, 0.008, 3)))
        for zi_ in (-2**31, 2**31 - 1):
            add(f"qdq_gelu_{tag}_zi{zi_}", x,
                lambda xx, p, bits=bits, it=it, ot=ot, zi=zi_: call(
                    f"npu_qdq_gelu_exp2_a{bits}", xx, INT[it][0], OUT, INT[ot][0], I64(len(xx)),
                    zi, *p, out_nbytes=len(xx) * np.dtype(INT[ot][1]).itemsize,
                    out_dtype=INT[ot][1]),
                lambda xx, p, ot=ot, zi=zi_: gg.qdq_gelu_exp2_a8(xx, zi, *p, ot), fl(zo, ka, c, kd))

# ---------------------------------------------------------------- sigmoid / tanh


def image(offset, slope):
    w = np.stack([np.asarray(offset, F32), np.asarray(slope, F32)], 1).reshape(16, 1, 8, 2)
    return np.repeat(w, 2, axis=1).reshape(-1).astype(F32)


def lut128(sigmoid):
    sl = RNG.uniform(1e-3, 0.6, 128).astype(F32)
    off = RNG.uniform(-1.0, 1.0, 128).astype(F32)
    return image((off + 1) / 2 if sigmoid else off, sl)


LUT_S, LUT_T = lut128(True), lut128(False)

for ct in ("bf16", "f16"):
    for a16 in (False, True):
        bits = 16 if a16 else 8
        for it, ot in QTYPES[bits]:
            x = xint(it)
            for op, fn in (("sigmoid", gsg.sigmoid_a8), ("tanh", gsg.tanh_a8)):
                add(f"{op}_a{bits}_{it}_{ot}_{ct}", x,
                    goq(f"npu_{op}_a{bits}", it, ot, (F16T[ct],)),
                    lambda xx, p, fn=fn, ot=ot, ct=ct: fn(xx, ot, ct, *p),
                    fl(3.0, 0.0954, 0.0, 255.0))
for a16 in (False, True):
    bits = 16 if a16 else 8
    for it, ot in QTYPES[bits]:
        x = xint(it)
        for op, fn, lut in (("sigmoid", gsg.qdq_sigmoid_lut_a8, LUT_S),
                            ("tanh", gsg.qdq_tanh_lut_a8, LUT_T)):
            add(f"qdq_{op}_lut_a{bits}_{it}_{ot}", x,
                lambda xx, p, op=op, bits=bits, it=it, ot=ot, lut=lut: call(
                    f"npu_qdq_{op}_lut_a{bits}", xx, INT[it][0], lut, OUT, INT[ot][0],
                    I64(len(xx)), *p, out_nbytes=len(xx) * np.dtype(INT[ot][1]).itemsize,
                    out_dtype=INT[ot][1]),
                lambda xx, p, fn=fn, ot=ot, lut=lut: fn(xx, ot, *p, lut),
                fl(3.0, 0.0954, 0.0, 255.0))
for t in ("bf16", "f16"):
    x = x16(t)
    add(f"tanh16_{t}", x, lambda xx, p, t=t: call(
        "npu_test_sigmoid_tanh16", xx, F16T[t], OUT, I64(len(xx)), out_nbytes=2 * len(xx),
        out_dtype=np.uint16), lambda xx, p, t=t: tanh16(xx, t), [])
    for op, fn, lut in (("sigmoid", gsg.elementwise_sigmoid_lut_bf16, LUT_S),
                        ("tanh", gsg.elementwise_tanh_lut_bf16, LUT_T)):
        add(f"{op}_lut16_{t}", x, lambda xx, p, op=op, t=t, lut=lut: call(
            f"npu_elementwise_{op}_lut_bf16", xx, F16T[t], lut, OUT, I64(len(xx)),
            out_nbytes=2 * len(xx), out_dtype=np.uint16),
            lambda xx, p, fn=fn, t=t, lut=lut: fn(xx, t, lut), [])
add("sigmoid_lut_fp8", X8, lambda xx, p: call(
    "npu_elementwise_sigmoid_lut_fp8", xx, LUT_S, OUT, I64(len(xx)), *p, out_nbytes=len(xx),
    out_dtype=np.uint8), lambda xx, p: gsg.elementwise_sigmoid_lut_fp8(xx, LUT_S, *p),
    fl(1.0, 8.0))
for t in ("bf16", "f16"):
    add(f"tanh_lut_fp8_{t}", X8, lambda xx, p, t=t: call(
        "npu_elementwise_tanh_lut_fp8", xx, LUT_T, OUT, I64(len(xx)), F16T[t], *p,
        out_nbytes=len(xx), out_dtype=np.uint8),
        lambda xx, p, t=t: gsg.elementwise_tanh_lut_fp8(xx, t, LUT_T, *p), fl(1.0, 8.0))

# ---------------------------------------------------------------- silu / sigtanh
SIG, TAN = gsi.SIGMOID, gsi.TANH
for t in ("bf16", "f16"):
    x = x16(t)
    for act in (SIG, TAN):
        add(f"sigtanh16_{t}_{act}", x, lambda xx, p, t=t, act=act: call(
            "npu_sigtanh_bf16", xx, F16T[t], OUT, I64(len(xx)), act, out_nbytes=2 * len(xx),
            out_dtype=np.uint16), lambda xx, p, t=t, act=act: gsi.sigtanh_bf16(xx, t, act), [])
    add(f"silu16_{t}", x, lambda xx, p, t=t: call(
        "npu_silu_bf16", xx, F16T[t], OUT, I64(len(xx)), out_nbytes=2 * len(xx),
        out_dtype=np.uint16), lambda xx, p, t=t: gsi.silu_bf16(xx, t), [])
    add(f"silu_fp8_{t}", X8, lambda xx, p, t=t: call(
        "npu_elementwise_silu_fp8", xx, OUT, I64(len(xx)), F16T[t], *p, out_nbytes=len(xx),
        out_dtype=np.uint8), lambda xx, p, t=t: gsi.elementwise_silu_fp8(xx, t, *p), fl(1.0, 1.0))
for act in (SIG, TAN):
    add(f"sigtanh_bfp16_{act}", BFP, lambda xx, p, act=act: call(
        "npu_sigtanh_bfp16", xx, OUT, I64(len(xx) // 9 * 8), act, out_nbytes=len(xx),
        out_dtype=np.uint8), lambda xx, p, act=act: gsi.sigtanh_bfp16(xx, act), [])
    lit1, lit2 = (0.76842356, -0.12469438) if act == SIG else (1.53878212, -0.49819374)
    add(f"sigtanh_fp8_{act}", X8, lambda xx, p, act=act: call(
        "npu_elementwise_sigtanh_fp8", xx, OUT, I64(len(xx)), act, *p, out_nbytes=len(xx),
        out_dtype=np.uint8), lambda xx, p, act=act: gsi.elementwise_sigtanh_fp8(xx, act, *p),
        fl(lit1, lit2, 1.0))
add("silu_bfp16", BFP, lambda xx, p: call("npu_silu_bfp16", xx, OUT, I64(len(xx) // 9 * 8),
                                          out_nbytes=len(xx), out_dtype=np.uint8),
    lambda xx, p: gsi.silu_bfp16(xx), [])
for ct in ("bf16", "f16"):
    for bits in (8, 16):
        for it, ot in QTYPES[bits]:
            x = xint(it)
            add(f"qdq_silu_a{bits}_{it}_{ot}_{ct}", x,
                goq(f"npu_qdq_silu_exp2_a{bits}", it, ot, (F16T[ct],)),
                lambda xx, p, ot=ot, ct=ct: gsi.qdq_silu_exp2_a8(xx, ot, ct, *p),
                fl(3.0, 0.0954, 0.0, 255.0))
            for act in (SIG, TAN):
                lit1, lit2 = (0.76842356, -0.12469438) if act == SIG else (1.53878212, -0.49819374)
                add(f"qdq_sigtanh_a{bits}_{it}_{ot}_{ct}_{act}", x,
                    lambda xx, p, bits=bits, it=it, ot=ot, ct=ct, act=act: call(
                        f"npu_qdq_sigtanh_a{bits}", xx, INT[it][0], OUT, INT[ot][0],
                        I64(len(xx)), act, F16T[ct], *p,
                        out_nbytes=len(xx) * np.dtype(INT[ot][1]).itemsize,
                        out_dtype=INT[ot][1]),
                    lambda xx, p, ot=ot, ct=ct, act=act: gsi.qdq_sigtanh_a8(
                        xx, ot, act, ct, *p[:4], tuple(p[4:])),
                    fl(3.0, 0.0954, 0.0, 255.0, 894.00097656, lit1, lit2))

for mode in (0, 1):
    for it in ("i16", "u16"):
        x = ALL16.view(INT[it][1])           # every 16-bit input code
        for ot in ("i16", "u16"):
            add(f"qdq_silu_fp32_a16_{it}_{ot}_m{mode}", x,
                goq("npu_qdq_silu_fp32_a16", it, ot, (mode,)),
                lambda xx, p, ot=ot, mode=mode: gsi.qdq_silu_fp32_a16(xx, ot, mode, *p),
                fl(3.0, 0.00156915537, 361.0, 1 / 0.00077118905))

# ---------------------------------------------------------------- sin / cos
W = {False: gsc.fit_lut(False), True: gsc.fit_lut(True)}
for fn in ("sin", "cos"):
    cs = fn == "cos"
    for t in ("bf16", "f16"):
        x = x16(t)
        add(f"{fn}16_{t}", x, lambda xx, p, fn=fn, t=t: call(
            f"npu_{fn}_bf16", xx, F16T[t], OUT, I64(len(xx)), out_nbytes=2 * len(xx),
            out_dtype=np.uint16), lambda xx, p, t=t, fn=fn: getattr(gsc, f"{fn}_bf16")(xx, t), [])
        add(f"{fn}_lut16_{t}", x, lambda xx, p, fn=fn, t=t, cs=cs: call(
            f"npu_elementwise_{fn}_lut_bf16", xx, F16T[t], W[cs], OUT, I64(len(xx)),
            out_nbytes=2 * len(xx), out_dtype=np.uint16),
            lambda xx, p, t=t, cs=cs: gsc.elementwise_sin_lut_bf16(xx, W[cs], t), [])
        add(f"{fn}_fp8_{t}", X8, lambda xx, p, fn=fn, t=t: call(
            f"npu_elementwise_{fn}_fp8", xx, OUT, I64(len(xx)), F16T[t], *p, out_nbytes=len(xx),
            out_dtype=np.uint8),
            lambda xx, p, t=t, fn=fn: getattr(gsc, f"elementwise_{fn}_fp8")(xx, t, *p),
            fl(1.0, 1.0))
        add(f"{fn}_lut_fp8_{t}", X8, lambda xx, p, fn=fn, t=t, cs=cs: call(
            f"npu_elementwise_{fn}_lut_fp8", xx, W[cs], OUT, I64(len(xx)), F16T[t], *p,
            out_nbytes=len(xx), out_dtype=np.uint8),
            lambda xx, p, t=t, cs=cs: gsc.elementwise_sin_lut_fp8(xx, W[cs], t, *p), fl(1.0, 1.0))
        for bits in (8, 16):
            for it, ot in QTYPES[bits]:
                x = xint(it)
                add(f"{fn}_a{bits}_{it}_{ot}_{t}", x, lambda xx, p, fn=fn, bits=bits, it=it, ot=ot,
                    t=t: call(f"npu_elementwise_{fn}_a{bits}", xx, INT[it][0], OUT, INT[ot][0],
                              I64(len(xx)), F16T[t], *p,
                              out_nbytes=len(xx) * np.dtype(INT[ot][1]).itemsize,
                              out_dtype=INT[ot][1]),
                    lambda xx, p, ot=ot, t=t, fn=fn, bits=bits: getattr(
                        gsc, f"elementwise_{fn}_a{bits}")(xx, ot, t, *p),
                    fl(3.0, 0.0371, 5.0, 127.0))
    for bits in (8, 16):
        for it, ot in QTYPES[bits]:
            x = xint(it)
            add(f"{fn}_lut_a{bits}_{it}_{ot}", x, lambda xx, p, fn=fn, bits=bits, it=it, ot=ot,
                cs=cs: call(f"npu_elementwise_{fn}_lut_a{bits}", xx, INT[it][0], W[cs], OUT,
                            INT[ot][0], I64(len(xx)), *p,
                            out_nbytes=len(xx) * np.dtype(INT[ot][1]).itemsize,
                            out_dtype=INT[ot][1]),
                lambda xx, p, ot=ot, cs=cs: gsc.elementwise_sin_lut_a8(xx, W[cs], ot, *p),
                fl(3.0, 0.0371, 5.0, 127.0))


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_special_values(case):
    sweep(case)


# ---------------------------------------------------------------- table words


def poke(lut, group_len, field, sv, every):
    """lut with the offset (field 0) or slope (field 1) of every `every`-th entry set to sv."""
    w = lut.copy()
    for i in range(0, group_len, every):
        k = (i // 8) * 32 + (i % 8) * 2 + field
        w[k] = sv
        w[k + 16] = sv  # second copy; never read
    return w


@pytest.mark.parametrize("field", [0, 1])
@pytest.mark.parametrize("every", [1, 3])
def test_lut_words_special(field, every):
    """NaN / inf / subnormal words in the sigmoid / tanh / sincos tables."""
    fails = []
    for name, mk, glut, n in LUT_CASES:
        for lab, v in SPECIAL:
            tab = poke(glut, n, field, v, every)
            err = run_case(mk(tab), mk(tab).defaults)
            if err:
                fails.append(f"{name} field {field} every {every} {lab}: {err}")
    assert not fails, "\n".join(fails[:12]) + f"\n({len(fails)} failing)"


def _lut_cases():
    out = []
    for t in ("bf16", "f16"):
        x = x16(t)
        out.append((f"sigmoid_lut16_{t}", lambda tab, t=t, x=x: Case(
            "s", x, lambda xx, p: call("npu_elementwise_sigmoid_lut_bf16", xx, F16T[t], tab, OUT,
                                       I64(len(xx)), out_nbytes=2 * len(xx),
                                       out_dtype=np.uint16),
            lambda xx, p: gsg.elementwise_sigmoid_lut_bf16(xx, t, tab), []), LUT_S, 128))
        out.append((f"tanh_lut16_{t}", lambda tab, t=t, x=x: Case(
            "s", x, lambda xx, p: call("npu_elementwise_tanh_lut_bf16", xx, F16T[t], tab, OUT,
                                       I64(len(xx)), out_nbytes=2 * len(xx),
                                       out_dtype=np.uint16),
            lambda xx, p: gsg.elementwise_tanh_lut_bf16(xx, t, tab), []), LUT_T, 128))
        out.append((f"tanh_lut_fp8_{t}", lambda tab, t=t: Case(
            "s", X8, lambda xx, p: call("npu_elementwise_tanh_lut_fp8", xx, tab, OUT, I64(len(xx)),
                                        F16T[t], 1.0, 8.0, out_nbytes=len(xx),
                                        out_dtype=np.uint8),
            lambda xx, p: gsg.elementwise_tanh_lut_fp8(xx, t, tab, 1.0, 8.0), []), LUT_T, 128))
    out.append(("sigmoid_lut_fp8", lambda tab: Case(
        "s", X8, lambda xx, p: call("npu_elementwise_sigmoid_lut_fp8", xx, tab, OUT, I64(len(xx)),
                                    1.0, 8.0, out_nbytes=len(xx), out_dtype=np.uint8),
        lambda xx, p: gsg.elementwise_sigmoid_lut_fp8(xx, tab, 1.0, 8.0), []), LUT_S, 128))
    xi = xint("i8")
    for op, fn, lut in (("sigmoid", gsg.qdq_sigmoid_lut_a8, LUT_S),
                        ("tanh", gsg.qdq_tanh_lut_a8, LUT_T)):
        out.append((f"qdq_{op}_lut_a8", lambda tab, op=op, fn=fn: Case(
            "s", xi, lambda xx, p: call(f"npu_qdq_{op}_lut_a8", xx, NPU_I8, tab, OUT, NPU_I8,
                                        I64(len(xx)), 3.0, 0.0954, 0.0, 127.0,
                                        out_nbytes=len(xx), out_dtype=np.int8),
            lambda xx, p: fn(xx, "i8", 3.0, 0.0954, 0.0, 127.0, tab), []), lut, 128))
    for fn in ("sin", "cos"):
        cs = fn == "cos"
        x = x16("bf16")
        out.append((f"{fn}_lut16", lambda tab, fn=fn, x=x: Case(
            "s", x, lambda xx, p: call(f"npu_elementwise_{fn}_lut_bf16", xx, NPU_BF16, tab, OUT,
                                       I64(len(xx)), out_nbytes=2 * len(xx), out_dtype=np.uint16),
            lambda xx, p: gsc.elementwise_sin_lut_bf16(xx, tab, "bf16"), []), W[cs], 512))
        out.append((f"{fn}_lut_a8", lambda tab, fn=fn: Case(
            "s", xi, lambda xx, p: call(f"npu_elementwise_{fn}_lut_a8", xx, NPU_I8, tab, OUT,
                                        NPU_I8, I64(len(xx)), 3.0, 0.0371, 5.0, 127.0,
                                        out_nbytes=len(xx), out_dtype=np.int8),
            lambda xx, p: gsc.elementwise_sin_lut_a8(xx, tab, "i8", 3.0, 0.0371, 5.0, 127.0),
            []), W[cs], 512))
    return out


LUT_CASES = _lut_cases()


# ---------------------------------------------------------------- integer parameters


@pytest.mark.parametrize("a_shift,b_shift,q4,r", [(0, 0, 0, 0), (23, 23, 23, 23), (0, 23, 0, 23),
                                                 (23, 0, 23, 0), (1, 23, 5, 0)])
@pytest.mark.parametrize("K,P", [(127, 127), (-128, -128), (-128, 127), (127, -128), (0, 0)])
@pytest.mark.parametrize("up,pe", [(0, 0), (1, 1)])
def test_xint8_gelu_extreme(a_shift, b_shift, q4, r, K, P, up, pe):
    x = np.arange(-128, 128).astype(np.int8)
    for rnd in (0, 1, 2):
        p = [up, rnd, a_shift, b_shift, q4, r, K, P, pe]
        got = call("npu_xint8_gelu", x, OUT, I64(x.size), *p, out_nbytes=x.size, out_dtype=np.int8)
        same_bytes(got, gg.xint8_gelu(x, *p))


@pytest.mark.parametrize("bits", [8, 16])
def test_hardsigmoid_bounds_extreme(bits):
    """Bounds at the ends of the output range, also reversed."""
    for it, ot in QTYPES[bits]:
        x = xint(it)
        lo_r, hi_r = INT_RANGE[ot]
        for lo, hi in ((lo_r, hi_r), (hi_r, lo_r), (0, 0), (lo_r, lo_r), (hi_r, hi_r),
                       (max(lo_r, -1), 1)):
            got = call(f"npu_hardsigmoid_a{bits}", x, INT[it][0], OUT, INT[ot][0], I64(len(x)),
                       0.05, 3.0, lo, hi, out_nbytes=len(x) * np.dtype(INT[ot][1]).itemsize,
                       out_dtype=INT[ot][1])
            want = ge.hardsigmoid_a8(x, ot, F32(0.05), F32(3.0), lo, hi)
            same_bytes(got, want.astype(got.dtype))


def test_bad_args():
    # special coefficients do not change the argument checks
    nan, inf = float(_f(0x7FC00000)), float(_f(0x7F800000))
    P = np.zeros(64, np.uint16)  # a valid device buffer
    for name, args in (("npu_elementwise_exp_fp8", (nan, inf)),
                       ("npu_elementwise_elu_bf16", (nan,))):
        pre = (NPU_BF16,) if name.endswith("bf16") else ()
        assert raw_rc(name, None, *pre, P, I64(0), *args) == -1
        assert raw_rc(name, P, *pre, None, I64(64), *args) == -1
        assert raw_rc(name, P, *pre, P, I64(-1), *args) == -1
        assert raw_rc(name, P, *pre, P, I64(0), *args) == 0
