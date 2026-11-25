# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The LSTM and grpb entry points against the numpy reference, bit for bit."""

import numpy as np
import pytest

import golden_lstm as gl
from golden_act import sigmoid_tanh
from golden_float_cvt import round_float
from golden_int_round import Rnd
from npu import (I64, NPU_BF16, NPU_F16, NPU_F32, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT, OUT2,
                 call, callx, raw_rc)

RNG = np.random.default_rng(1503)


def lstm_case(a16, lx, lh, small, wide=False):
    r = RNG.integers
    if a16:
        hi = 16 if small else 65536
        x, h, c = (r(0, hi, n).astype(np.uint16) for n in (lx, lh, lh))
        w, rr = (r(0, 16 if small else 256, s).astype(np.uint8) for s in ((4, lh, lx), (4, lh, lh)))
        b = r(0, 1 << 20, (4, lh)).astype(np.uint32)
        zp = dict(c_zp=int(r(-32768, 32768)), oh_zp=int(r(0, 65536)), oc_zp=int(r(0, 65536)))
    else:
        lim = 8 if small else 128
        x, h, c = (r(-lim, lim, n).astype(np.int8) for n in (lx, lh, lh))
        w, rr = (r(-lim, lim, s).astype(np.int8) for s in ((4, lh, lx), (4, lh, lh)))
        b = r(-(1 << 12), 1 << 12, (4, lh)).astype(np.int32)
        zp = dict(c_zp=int(r(-128, 128)), oh_zp=int(r(-128, 128)), oc_zp=int(r(-128, 128)))
    # Gate inputs of a few units; output scales that sometimes saturate.
    big = 0 if small else 6
    p = dict(qx=int(r(1, 4)), qh=int(r(1, 4)), xw=int(r(0, 4)) + big, xs=int(r(0, 8)), hr=int(r(0, 4)) + big,
             hs=int(r(0, 8)), xwq=int(r(-4, 3)), hrq=int(r(-4, 3)), lin=int(r(4, 10)), c0s=int(r(0, 4)),
             c1s=int(r(0, 8)), c2s=int(r(4, 12)), c_scale=int(r(-300, 300)), nl=int(r(6, 12)),
             oh_scale=int(r(-4000, 4000)), oc_scale=int(r(-4000, 4000)), oh_s=int(r(4, 14)), oc_s=int(r(4, 14)), **zp)
    if wide:
        # Gate sums above 2^24, so the fp32 conversion rounds.
        b = (r(-(1 << 30), 1 << 30, (4, lh)) if not a16 else r(0, 1 << 31, (4, lh))).astype(b.dtype)
        p["lin"] = int(r(26, 30))
    return x, h, c, w, rr, b, gl._header(a16, p)


@pytest.mark.parametrize("a16", [False, True])
@pytest.mark.parametrize("dims", [(64, 16), (320, 64), (128, 48)])
@pytest.mark.parametrize("small", [True, False])
@pytest.mark.parametrize("rep", range(4))
def test_qdq_lstm(a16, dims, small, rep):
    lx, lh = dims
    x, h, c, w, r, b, q = lstm_case(a16, lx, lh, small)
    name = "npu_lstm_a16w8" if a16 else "npu_lstm_a8w8"
    nb = lh * (2 if a16 else 1)
    got_h, got_c = callx(name, x, h, c, w, r, b, q, OUT, OUT2, I64(lx), I64(lh), outs=[nb, nb])
    dt = np.uint16 if a16 else np.int8
    want_h, want_c = (gl.lstm_a16w8 if a16 else gl.lstm_a8w8)(x, h, c, w, r, b, q)
    for got, want, what in ((got_h, want_h, "h"), (got_c, want_c, "c")):
        got = got.view(dt).astype(np.int64)
        bad = np.nonzero(got != want)[0]
        assert bad.size == 0, f"{what}: {bad.size}/{lh} differ, first at {bad[:8].tolist()}"


@pytest.mark.parametrize("a16", [False, True])
@pytest.mark.parametrize("rep", range(3))
def test_qdq_lstm_wide(a16, rep):
    lx, lh = 128, 64
    x, h, c, w, r, b, q = lstm_case(a16, lx, lh, True, wide=True)
    name = "npu_lstm_a16w8" if a16 else "npu_lstm_a8w8"
    nb = lh * (2 if a16 else 1)
    got_h, got_c = callx(name, x, h, c, w, r, b, q, OUT, OUT2, I64(lx), I64(lh), outs=[nb, nb])
    dt = np.uint16 if a16 else np.int8
    want_h, want_c = (gl.lstm_a16w8 if a16 else gl.lstm_a8w8)(x, h, c, w, r, b, q)
    assert (got_h.view(dt).astype(np.int64) == want_h).all() and (got_c.view(dt).astype(np.int64) == want_c).all()


@pytest.mark.parametrize("a16", [False, True])
@pytest.mark.parametrize("wide", [False, True])
def test_lstm_gates(a16, wide):
    lx, lh = 128, 64
    x, h, c, w, r, b, q = lstm_case(a16, lx, lh, True, wide=wide)
    got = call("npu_test_lstm_gates", int(a16), x, h, w, r, b, q, OUT, I64(lx), I64(lh),
               out_nbytes=16 * lh, out_dtype=np.float32)
    want = gl._lstm_gates(x, h, w, r, b, q, a16)
    bad = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0]
    assert bad.size == 0, f"{bad.size}/{4 * lh} differ"
    if wide:
        assert (np.abs(want) * 2.0 ** gl._parse(q, a16)["lin"] > 2 ** 24).mean() > 0.5


def test_lstm_sigmoid_tie():
    """A bf16 tie rounds half to even."""
    t = np.float32(-0.98828125)
    assert sigmoid_tanh(np.float32(0), "bf16", Rnd.kHalfEven)[()] == np.float32(0.5)
    d = np.float32(0.5) - t * np.float32(0.5)
    assert d == np.float32(0.994140625)
    assert round_float(d, "bf16") == np.float32(0.9921875)


GRPB_IN = {"i8": (NPU_I8, np.int8, False), "u8": (NPU_U8, np.uint8, False),
           "i16": (NPU_I16, np.int16, True), "u16": (NPU_U16, np.uint16, True)}


@pytest.mark.parametrize("it", list(GRPB_IN))
@pytest.mark.parametrize("wt", ["i8", "u8"])
@pytest.mark.parametrize("mk", [(16, 64), (64, 96), (7, 200)])
@pytest.mark.parametrize("ofm", ["f16", "bf16"])
@pytest.mark.parametrize("actv_sum", [True, False])
def test_grpb(it, wt, mk, ofm, actv_sum):
    m, k = mk
    t, dt, a16 = GRPB_IN[it]
    info = np.iinfo(dt)
    ifm = RNG.integers(info.min, int(info.max) + 1, (m, k)).astype(dt)
    ifm[0] = 0                                     # x = c0 exactly
    w_t, w_dt = (NPU_I8, np.int8) if wt == "i8" else (NPU_U8, np.uint8)
    wi = np.iinfo(w_dt)
    wts = RNG.integers(wi.min, int(wi.max) + 1, (k, 8)).astype(w_dt)
    gsum = np.abs(ifm.astype(np.float64) @ wts.astype(np.float64)).max() * 4 + 1
    c2 = (RNG.standard_normal(m) * np.exp2(RNG.integers(1, 6, m)) / gsum).astype(np.float32)
    c0 = (RNG.standard_normal(m) * 2).astype(np.float32)
    c0[0] = -0.0 if ofm == "f16" else 0.0
    c1 = (RNG.standard_normal(m) / (k * float(info.max))).astype(np.float32) if actv_sum else None
    if actv_sum:
        c1[0] = -1.0
    head, lyr0, lyr1 = (float(np.float32(v)) for v in RNG.standard_normal(3) * 3)
    ot = NPU_F16 if ofm == "f16" else NPU_BF16
    got = call("npu_grpb_a16w8" if a16 else "npu_grpb_a8w8", ifm, t, wts, w_t, c0, c1, c2, OUT,
               ot, I64(m), I64(k), head, lyr0, lyr1, out_nbytes=2 * m, out_dtype=np.uint16)
    want = (gl.grpb_a16w8 if a16 else gl.grpb_a8w8)(ifm, wts, c0, c1, c2, ofm, head, lyr0, lyr1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, f"{bad.size}/{m} differ, first at {bad[:8].tolist()}"


def test_grpb_sigmoid_sweep():
    x = np.concatenate([np.linspace(-40, 40, 200001, dtype=np.float32),
                        RNG.standard_normal(100000).astype(np.float32) * 4,
                        np.ldexp(np.float32(1), np.arange(-149, 20)).astype(np.float32),
                        -np.ldexp(np.float32(1), np.arange(-149, 20)).astype(np.float32),
                        np.array([0.0, -0.0], np.float32)]).astype(np.float32)
    got = call("npu_test_grpb_sigmoid", x, OUT, I64(x.size), out_nbytes=4 * x.size, out_dtype=np.float32)
    want = gl._grpb_sigmoid(x)
    bad = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0]
    assert bad.size == 0, f"{bad.size}/{x.size} differ, first x {x[bad[:4]].tolist()}"


@pytest.mark.parametrize("it", ["i16", "u16"])
def test_grpb_preact_large_lo(it):
    """Low-byte sums near 2^27: fp32(gemm_lo) rounds before the join."""
    m, k = 1024, 1000
    t, dt, a16 = GRPB_IN[it]
    hi = RNG.integers(-128 if it == "i16" else 0, 128 if it == "i16" else 256, (m, k))
    ifm = ((hi << 8) | RNG.integers(200, 256, (m, k))).astype(np.int64).astype(dt)
    wts = RNG.integers(200, 256, (k, 8)).astype(np.uint8)
    c0, c1, c2 = (RNG.standard_normal(m).astype(np.float32) for _ in range(3))
    got = call("npu_test_grpb_stages", ifm, t, wts, NPU_U8, c0, c1, c2, OUT, I64(m), I64(k),
               out_nbytes=16 * m, out_dtype=np.float32).reshape(m, 4)
    x = gl._grpb_preact(ifm, a16, wts, c0, c1, c2)
    bad = np.nonzero(got[:, :2].view(np.uint32) != x.view(np.uint32))
    assert bad[0].size == 0, f"{bad[0].size}/{2 * m} pre-activations differ"


@pytest.mark.parametrize("it", list(GRPB_IN))
@pytest.mark.parametrize("k", [64, 200, 1000])
def test_grpb_preact(it, k):
    m = 64
    t, dt, a16 = GRPB_IN[it]
    info = np.iinfo(dt)
    ifm = RNG.integers(info.min, int(info.max) + 1, (m, k)).astype(dt)
    wts = RNG.integers(-128, 128, (k, 8)).astype(np.int8)
    c0, c1, c2 = (RNG.standard_normal(m).astype(np.float32) for _ in range(3))
    got = call("npu_test_grpb_stages", ifm, t, wts, NPU_I8, c0, c1, c2, OUT, I64(m), I64(k),
               out_nbytes=16 * m, out_dtype=np.float32).reshape(m, 4)
    x = gl._grpb_preact(ifm, a16, wts, c0, c1, c2)
    bad = np.nonzero(got[:, :2].view(np.uint32) != x.view(np.uint32))
    assert bad[0].size == 0, f"{bad[0].size}/{2 * m} pre-activations differ"
    s = gl._grpb_sigmoid(x)
    assert (got[:, 2:].view(np.uint32) == s.view(np.uint32)).all()


@pytest.mark.parametrize("ofm,tie,want", [("f16", 1 + 2.0 ** -11, 0x3C00), ("f16", 1 + 3 * 2.0 ** -11, 0x3C02),
                                          ("bf16", 1 + 2.0 ** -8, 0x3F80), ("bf16", 1 + 3 * 2.0 ** -8, 0x3F82)])
def test_grpb_output_tie(ofm, tie, want):
    """head = lyr0 = 0 makes out = lyr1: an f16 / bf16 tie rounds to even."""
    m, k = 16, 64
    ifm = RNG.integers(-128, 128, (m, k)).astype(np.int8)
    wts = RNG.integers(-128, 128, (k, 8)).astype(np.int8)
    c = np.ones(m, np.float32)
    got = call("npu_grpb_a8w8", ifm, NPU_I8, wts, NPU_I8, c, c, c, OUT,
               NPU_F16 if ofm == "f16" else NPU_BF16, I64(m), I64(k), 0.0, 0.0,
               float(np.float32(tie)), out_nbytes=2 * m, out_dtype=np.uint16)
    assert (got == want).all(), hex(int(got[0]))


def test_grpb_sigmoid_sign():
    x = np.array([0.0, -0.0, 1.5, -1.5, 30.0, -30.0], np.float32)
    s = gl._grpb_sigmoid(x)
    assert s[0] == np.float32(1) - s[1] and s[1] > 0
    assert 0.8 < s[2] < 0.85 and 0.15 < s[3] < 0.2 and s[4] == 1 and 0 < s[5] < 1e-30


# ---- special values: extreme shifts / scales / zero points / operands of the integer LSTM (NaN and
# inf arise from int_to_float overflow), NaN / inf / subnormal / -0 coefficients of grpb ----

SHIFTS = [-128, -127, -64, -9, -1, 0, 1, 9, 31, 32, 63, 64, 65, 126, 127]
SCALES = [-32768, -32767, -1, 0, 1, 32767]
QWORDS = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]


def extreme_case(a16, lx, lh):
    r = RNG.integers
    pick = lambda xs: int(xs[r(0, len(xs))])
    if a16:
        x, h, c = (np.where(r(0, 3, n) == 0, 0, np.where(r(0, 2, n) == 0, 65535, r(0, 65536, n))).astype(np.uint16)
                   for n in (lx, lh, lh))
        w, rr = (np.where(r(0, 3, s) == 0, 0, np.where(r(0, 2, s) == 0, 255, r(0, 256, s))).astype(np.uint8)
                 for s in ((4, lh, lx), (4, lh, lh)))
        b = RNG.choice(np.array([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF], np.uint32), (4, lh))
        zp = dict(c_zp=pick([-32768, -1, 0, 32767]), oh_zp=pick([0, 1, 32768, 65535]), oc_zp=pick([0, 65535]))
    else:
        x, h, c = (RNG.choice(np.array([-128, -127, -1, 0, 1, 127], np.int8), n) for n in (lx, lh, lh))
        w, rr = (RNG.choice(np.array([-128, -1, 0, 1, 127], np.int8), s) for s in ((4, lh, lx), (4, lh, lh)))
        b = RNG.choice(np.array([INT_MIN, -1, 0, 1, INT_MAX], np.int32), (4, lh))
        zp = dict(c_zp=pick([-128, -1, 0, 127]), oh_zp=pick([-128, 0, 127]), oc_zp=pick([-128, 127]))
    p = dict(qx=pick(QWORDS), qh=pick(QWORDS), xw=pick(SHIFTS), xs=pick(SHIFTS), hr=pick(SHIFTS), hs=pick(SHIFTS),
             xwq=pick(SHIFTS), hrq=pick(SHIFTS), lin=pick(SHIFTS), c0s=pick(SHIFTS), c1s=pick(SHIFTS),
             c2s=pick(SHIFTS), c_scale=pick(SCALES), nl=pick(SHIFTS), oh_scale=pick(SCALES),
             oc_scale=pick(SCALES), oh_s=pick(SHIFTS), oc_s=pick(SHIFTS), **zp)
    return x, h, c, w, rr, b, gl._header(a16, p)


INT_MIN, INT_MAX = -(1 << 31), (1 << 31) - 1


@pytest.mark.parametrize("a16", [False, True])
@pytest.mark.parametrize("rep", range(40))
def test_special_qdq_lstm(a16, rep):
    lx, lh = 24, 16
    x, h, c, w, r, b, q = extreme_case(a16, lx, lh)
    name = "npu_lstm_a16w8" if a16 else "npu_lstm_a8w8"
    nb = lh * (2 if a16 else 1)
    got_h, got_c = callx(name, x, h, c, w, r, b, q, OUT, OUT2, I64(lx), I64(lh), outs=[nb, nb])
    dt = np.uint16 if a16 else np.int8
    want_h, want_c = (gl.lstm_a16w8 if a16 else gl.lstm_a8w8)(x, h, c, w, r, b, q)
    assert (got_h.view(dt).astype(np.int64) == want_h).all(), "h"
    assert (got_c.view(dt).astype(np.int64) == want_c).all(), "c"
    gates = call("npu_test_lstm_gates", int(a16), x, h, w, r, b, q, OUT, I64(lx), I64(lh),
                 out_nbytes=16 * lh, out_dtype=np.float32)
    assert (gates.view(np.uint32) == gl._lstm_gates(x, h, w, r, b, q, a16).view(np.uint32)).all(), "gates"


SPECIAL_F32 = np.array([np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-40, -1e-40, 1e-45, 3.4028235e38,
                        -3.4028235e38, 1.0, -1.0], np.float32)
SPECIAL_F32 = np.concatenate([SPECIAL_F32, np.array([0x7FC12345, 0xFFA00001], np.uint32).view(np.float32)])


def sp_f32(n, rate=0.4):
    v = (RNG.standard_normal(n) * 2).astype(np.float32)
    pos = RNG.random(n) < rate
    v[pos] = SPECIAL_F32[RNG.integers(0, SPECIAL_F32.size, int(pos.sum()))]
    return v


@pytest.mark.parametrize("it", list(GRPB_IN))
@pytest.mark.parametrize("wt", ["i8", "u8"])
@pytest.mark.parametrize("ofm", ["f16", "bf16"])
@pytest.mark.parametrize("actv_sum", [True, False])
@pytest.mark.parametrize("rep", range(4))
def test_special_grpb(it, wt, ofm, actv_sum, rep):
    m, k = 32, 48
    t, dt, a16 = GRPB_IN[it]
    info = np.iinfo(dt)
    ifm = RNG.choice(np.array([info.min, info.max, 0, 1], dt), (m, k))
    w_t, w_dt = (NPU_I8, np.int8) if wt == "i8" else (NPU_U8, np.uint8)
    wts = RNG.choice(np.array([np.iinfo(w_dt).min, np.iinfo(w_dt).max, 0, 1], w_dt), (k, 8))
    c0, c2 = sp_f32(m), sp_f32(m)
    c1 = sp_f32(m) if actv_sum else None
    head, lyr0, lyr1 = (float(v) for v in SPECIAL_F32[RNG.integers(0, SPECIAL_F32.size, 3)])
    ot = NPU_F16 if ofm == "f16" else NPU_BF16
    name = "npu_grpb_a16w8" if a16 else "npu_grpb_a8w8"
    got = call(name, ifm, t, wts, w_t, c0, c1, c2, OUT, ot, I64(m), I64(k), head, lyr0, lyr1,
               out_nbytes=2 * m, out_dtype=np.uint16)
    want = (gl.grpb_a16w8 if a16 else gl.grpb_a8w8)(ifm, wts, c0, c1, c2, ofm, head, lyr0, lyr1)
    assert (got == want).all(), f"{(got != want).sum()}/{m} differ"
    st = call("npu_test_grpb_stages", ifm, t, wts, w_t, c0, c1, c2, OUT, I64(m), I64(k),
              out_nbytes=16 * m, out_dtype=np.float32).reshape(m, 4)
    x = gl._grpb_preact(ifm, a16, wts, c0, c1, c2)
    assert (st[:, :2].view(np.uint32) == x.view(np.uint32)).all(), "pre-activation"
    assert (st[:, 2:].view(np.uint32) == gl._grpb_sigmoid(x).view(np.uint32)).all(), "sigmoid"


def test_special_grpb_sigmoid():
    x = np.concatenate([SPECIAL_F32, np.array([0x7F800001, 0xFF800001, 0x00000001, 0x80000001, 0x007FFFFF,
                                               0x7F7FFFFF, 0xFF7FFFFF], np.uint32).view(np.float32)])
    got = call("npu_test_grpb_sigmoid", x, OUT, I64(x.size), out_nbytes=4 * x.size, out_dtype=np.float32)
    want = gl._grpb_sigmoid(x)
    bad = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0]
    assert bad.size == 0, f"x {x[bad].tolist()} got {got.view(np.uint32)[bad].tolist()} " \
                          f"want {want.view(np.uint32)[bad].tolist()}"


@pytest.mark.parametrize("a16", [False, True])
@pytest.mark.parametrize("c0s,oh_s,scale", [(127, 127, 32767), (100, 100, -32768), (-127, 64, 1), (64, -127, -1)])
def test_special_qdq_lstm_int64_wrap(a16, c0s, oh_s, scale):
    """c and the output zero point shifted left out of the int64 range saturate, and the products
    and sums they enter wrap."""
    lx, lh = 16, 16
    x, h, c, w, r, b, _ = lstm_case(a16, lx, lh, True)
    c[:] = 3
    zp = dict(c_zp=1, oh_zp=65535 if a16 else 127, oc_zp=65535 if a16 else -128)
    p = dict(qx=1, qh=1, xw=2, xs=1, hr=2, hs=1, xwq=0, hrq=0, lin=8, c0s=c0s, c1s=3, c2s=8, c_scale=scale,
             nl=9, oh_scale=scale, oc_scale=-scale if scale != -32768 else 32767, oh_s=oh_s, oc_s=oh_s, **zp)
    q = gl._header(a16, p)
    name = "npu_lstm_a16w8" if a16 else "npu_lstm_a8w8"
    nb = lh * (2 if a16 else 1)
    got_h, got_c = callx(name, x, h, c, w, r, b, q, OUT, OUT2, I64(lx), I64(lh), outs=[nb, nb])
    dt = np.uint16 if a16 else np.int8
    want_h, want_c = (gl.lstm_a16w8 if a16 else gl.lstm_a8w8)(x, h, c, w, r, b, q)
    assert (got_h.view(dt).astype(np.int64) == want_h).all(), "h"
    assert (got_c.view(dt).astype(np.int64) == want_c).all(), "c"


def test_bad_args():
    """Every entry returns -1 for arguments it cannot take, before touching memory."""
    p = np.zeros(64, np.uint8)   # placeholder: these calls must return before any access
    for name in ("npu_lstm_a8w8", "npu_lstm_a16w8"):
        for lx, lh, x in ((0, 4, p), (4, 0, p), (4, 4, None)):
            assert raw_rc(name, x, p, p, p, p, p, p, p, p, I64(lx), I64(lh)) == -1, (name, lx, lh)
        assert raw_rc(name, p, p, p, p, p, p, None, p, p, I64(4), I64(4)) == -1
    for name, it in (("npu_grpb_a8w8", NPU_I8), ("npu_grpb_a16w8", NPU_I16)):
        other = NPU_I16 if it == NPU_I8 else NPU_I8
        for ifm_t, w_t, ot, m, c0 in ((other, NPU_I8, NPU_F16, 2, p), (NPU_F32, NPU_I8, NPU_F16, 2, p),
                                      (it, NPU_I16, NPU_F16, 2, p), (it, NPU_I8, NPU_F32, 2, p),
                                      (it, NPU_I8, NPU_F16, -1, p), (it, NPU_I8, NPU_F16, 2, None),
                                      (it, NPU_I8, NPU_F16, 0, None)):
            assert raw_rc(name, p, ifm_t, p, w_t, c0, None, p, p, ot, I64(m), I64(4), 0.0, 0.0,
                          0.0) == -1, (name, ifm_t, w_t, ot, m)
        assert raw_rc(name, p, it, p, NPU_I8, p, None, p, p, NPU_F16, I64(2), I64(0), 0.0, 0.0,
                      0.0) == -1, name   # k is the reduction length
        # zero rows: an empty output, nothing launched
        assert raw_rc(name, p, it, p, NPU_I8, p, None, p, p, NPU_F16, I64(0), I64(4), 0.0, 0.0,
                      0.0) == 0, name
    for ifm_t, w_t in ((NPU_F32, NPU_I8), (NPU_I8, NPU_U16), (NPU_BF16, NPU_U8)):
        assert raw_rc("npu_test_grpb_stages", p, ifm_t, p, w_t, p, None, p, p, I64(2),
                      I64(4)) == -1, (ifm_t, w_t)
    assert raw_rc("npu_test_lstm_gates", 0, p, p, p, p, p, None, p, I64(4), I64(4)) == -1
    assert raw_rc("npu_test_grpb_stages", p, NPU_I8, p, NPU_I8, p, None, p, p, I64(-1),
                  I64(4)) == -1
    assert raw_rc("npu_test_grpb_stages", p, NPU_I8, p, NPU_I8, p, None, p, p, I64(0),
                  I64(4)) == 0
    assert raw_rc("npu_test_grpb_sigmoid", None, p, I64(4)) == -1
    assert raw_rc("npu_test_grpb_sigmoid", p, None, I64(0)) == -1
    assert raw_rc("npu_test_grpb_sigmoid", p, p, I64(-1)) == -1
    assert raw_rc("npu_test_grpb_sigmoid", p, p, I64(0)) == 0
