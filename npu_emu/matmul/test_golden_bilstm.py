# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""npu_bidirectional_lstm_bf16 against the numpy reference, bit for bit."""

import numpy as np
import pytest

import golden_bilstm as gl
from golden_float_cvt import bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16
from npu import I64, OUT, OUT2, call, callx, raw_rc

RNG = np.random.default_rng(1607)


def bf16(v):
    return f32_to_bf16(np.asarray(v, np.float32))


def random_lut():
    """The default table with perturbed offsets and slopes."""
    words = gl._default_lut().reshape(-1, 2)
    off = words[:, 0] + np.float32(2.0 ** -6) * RNG.uniform(-1, 1, words.shape[0]).astype(np.float32)
    sl = words[:, 1] * RNG.uniform(0.9, 1.1, words.shape[0]).astype(np.float32)
    return np.stack([off, sl], 1).reshape(-1).astype(np.float32)


LUTS = {"default": gl._default_lut(), "random": random_lut()}


@pytest.mark.parametrize("lut", list(LUTS))
@pytest.mark.parametrize("is_tanh", [0, 1])
def test_gate_lut_exhaustive(lut, is_tanh):
    codes = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    codes = codes[(codes & 0x7C00) != 0x7C00]  # finite f16
    v = f16_to_f32(codes)
    fn = gl._tanh_lut if is_tanh else gl._sigmoid_lut
    want = f32_to_f16(fn(v, LUTS[lut]))
    got = call("npu_test_bilstm_gate_lut", codes, LUTS[lut], is_tanh, OUT, I64(codes.size),
               out_nbytes=2 * codes.size, out_dtype=np.uint16)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, f"{bad.size} differ, first f16 {codes[bad[0]]:#06x}"


CASES = [
    # seq, input, hidden, reverse, weight scale, lut
    (1, 4, 16, 0, 1.0, "default"),
    (3, 8, 16, 1, 1.0, "default"),
    (5, 64, 32, 0, 0.5, "default"),
    (5, 64, 32, 1, 0.5, "random"),
    (9, 20, 48, 0, 2.0, "default"),
    (9, 20, 48, 1, 2.0, "random"),
    (4, 256, 16, 0, 0.25, "default"),
    (12, 36, 32, 1, 3.0, "default"),
    (7, 128, 48, 0, 1.0, "random"),
    (2, 12, 16, 1, 8.0, "default"),
]


def operands(seq, n_in, hid, scale):
    x = bf16(RNG.standard_normal((seq, n_in)))
    h0 = bf16(0.5 * RNG.standard_normal(hid))
    c0 = bf16(RNG.standard_normal(hid) * np.exp2(RNG.integers(-3, 4, hid)))
    w = bf16(scale / np.sqrt(n_in) * RNG.standard_normal((4 * hid, n_in)))
    r = bf16(scale / np.sqrt(hid) * RNG.standard_normal((4 * hid, hid)))
    b = bf16(RNG.standard_normal(8 * hid))
    return x, h0, c0, w, r, b


@pytest.mark.parametrize("seq,n_in,hid,rev,scale,lut", CASES)
def test_bilstm_random(seq, n_in, hid, rev, scale, lut):
    x, h0, c0, w, r, b = operands(seq, n_in, hid, scale)
    f = bf16_to_f32
    y, c = gl.bidirectional_lstm_bf16(f(x), f(h0), f(c0), f(w), f(r), f(b), LUTS[lut], bool(rev))
    assert np.all(np.isfinite(f(y))) and np.all(np.isfinite(f(c)))
    gy, gc = callx("npu_bidirectional_lstm_bf16", x, h0, c0, w, r, b, LUTS[lut], OUT, OUT2,
                   I64(seq), I64(n_in), I64(hid), rev, outs=[2 * seq * hid, 2 * hid])
    np.testing.assert_array_equal(gy.view(np.uint16).reshape(seq, hid), y)
    np.testing.assert_array_equal(gc.view(np.uint16), c)


# Seeds whose outputs depend on rounding f * c before the add.
@pytest.mark.parametrize("seed", [7448, 7896, 8134])
def test_bilstm_cell_rounding(seed):
    rng = np.random.default_rng(seed)
    seq, n_in, hid = 3, 4, 16
    x = bf16(rng.standard_normal((seq, n_in)))
    h0 = bf16(rng.standard_normal(hid))
    c0 = bf16(4 * rng.standard_normal(hid))
    w = bf16(rng.standard_normal((4 * hid, n_in)))
    r = bf16(rng.standard_normal((4 * hid, hid)) / 4)
    b = bf16(rng.standard_normal(8 * hid))
    f = bf16_to_f32
    y, c = gl.bidirectional_lstm_bf16(f(x), f(h0), f(c0), f(w), f(r), f(b), LUTS["default"], False)
    gy, gc = callx("npu_bidirectional_lstm_bf16", x, h0, c0, w, r, b, LUTS["default"], OUT, OUT2,
                   I64(seq), I64(n_in), I64(hid), 0, outs=[2 * seq * hid, 2 * hid])
    np.testing.assert_array_equal(gy.view(np.uint16).reshape(seq, hid), y)
    np.testing.assert_array_equal(gc.view(np.uint16), c)


@pytest.mark.parametrize("rev", [0, 1])
def test_bilstm_wx_matrix_mac(rev):
    """W.x is one matrix MAC per 32 inputs, not fp32 chains: inputs 0 and 4 give products
    +-2^40 that cancel, and the 32-input tile aligns the bias and the other products to
    that exponent, so W.x + b is 0 - as if x and the W bias were zero. Chains over k mod 4
    would keep the small products and the bias."""
    rng = np.random.default_rng(5)
    seq, n_in, hid = 3, 64, 16
    x = rng.standard_normal((seq, n_in))
    w = rng.standard_normal((4 * hid, n_in))
    x[:, [0, 4, 32, 36]] = 2.0 ** 20
    w[:, [0, 32]], w[:, [4, 36]] = 2.0 ** 20, -(2.0 ** 20)
    x, w = bf16(x), bf16(w)
    h0, c0 = bf16(0.5 * rng.standard_normal(hid)), bf16(rng.standard_normal(hid))
    r = bf16(rng.standard_normal((4 * hid, hid)) / 4)
    b = bf16(rng.standard_normal(8 * hid))
    b_r = b.copy()
    b_r[:4 * hid] = 0
    f = bf16_to_f32
    y, c = gl.bidirectional_lstm_bf16(f(x) * 0, f(h0), f(c0), f(w), f(r), f(b_r),
                                      LUTS["default"], bool(rev))
    gy, gc = callx("npu_bidirectional_lstm_bf16", x, h0, c0, w, r, b, LUTS["default"], OUT, OUT2,
                   I64(seq), I64(n_in), I64(hid), rev, outs=[2 * seq * hid, 2 * hid])
    np.testing.assert_array_equal(gy.view(np.uint16).reshape(seq, hid), y)
    np.testing.assert_array_equal(gc.view(np.uint16), c)


def test_bad_args():
    p = np.zeros(64, np.uint16)
    lut = LUTS["default"]
    for n_in, hid, seq, rev in ((6, 16, 1, 0), (8, 24, 1, 0), (0, 16, 1, 0), (8, 16, -1, 0),
                                (8, 16, 1, 2), (8, 16, 1, -1)):
        assert raw_rc("npu_bidirectional_lstm_bf16", p, p, p, p, p, p, lut, p, p, I64(seq),
                      I64(n_in), I64(hid), rev) == -1, (n_in, hid, seq, rev)
    for k in range(9):   # each required pointer
        args = [p] * 6 + [lut] + [p] * 2
        args[k] = None
        assert raw_rc("npu_bidirectional_lstm_bf16", *args, I64(1), I64(8), I64(16), 0) == -1, k
    assert raw_rc("npu_test_bilstm_gate_lut", None, lut, 0, p, I64(4)) == -1
    assert raw_rc("npu_test_bilstm_gate_lut", p, lut, 0, None, I64(0)) == -1   # also when empty
    assert raw_rc("npu_test_bilstm_gate_lut", p, lut, 0, p, I64(-1)) == -1
    assert raw_rc("npu_test_bilstm_gate_lut", p, lut, 0, p, I64(0)) == 0
    # seq 0: y is empty; a null y is still rejected
    assert raw_rc("npu_bidirectional_lstm_bf16", p, p, p, p, p, p, lut, None, p, I64(0), I64(8),
                  I64(16), 0) == -1


# ---- special values: NaN / inf / -0 / subnormal operands and table entries ----

SPECIAL_BF16 = np.array([0x7FC0, 0xFFC0, 0x7FC1, 0x7F81, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x007F, 0x0001,
                         0x7F7F, 0xFF7F, 0x3F80], np.uint16)


def sp_bf16(v, rate):
    v = bf16(v).reshape(-1).copy()
    pos = RNG.random(v.size) < rate
    v[pos] = SPECIAL_BF16[RNG.integers(0, SPECIAL_BF16.size, int(pos.sum()))]
    return v


@pytest.mark.parametrize("lut", list(LUTS))
@pytest.mark.parametrize("is_tanh", [0, 1])
def test_special_gate_lut(lut, is_tanh):
    """Every f16 code, NaN and infinity included."""
    codes = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    v = f16_to_f32(codes)
    fn = gl._tanh_lut if is_tanh else gl._sigmoid_lut
    want = f32_to_f16(fn(v, LUTS[lut]))
    got = call("npu_test_bilstm_gate_lut", codes, LUTS[lut], is_tanh, OUT, I64(codes.size),
               out_nbytes=2 * codes.size, out_dtype=np.uint16)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, f"{bad.size} differ, first f16 {codes[bad[:4]].tolist()}"


@pytest.mark.parametrize("is_tanh", [0, 1])
@pytest.mark.parametrize("entry", [np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-40, 3.4028235e38])
def test_special_gate_lut_entries(is_tanh, entry):
    lut = LUTS["default"].copy()
    lut[RNG.choice(lut.size, 24, replace=False)] = entry
    codes = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    fn = gl._tanh_lut if is_tanh else gl._sigmoid_lut
    want = f32_to_f16(fn(f16_to_f32(codes), lut))
    got = call("npu_test_bilstm_gate_lut", codes, lut, is_tanh, OUT, I64(codes.size),
               out_nbytes=2 * codes.size, out_dtype=np.uint16)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, f"{bad.size} differ, first f16 {codes[bad[:4]].tolist()}"


@pytest.mark.parametrize("rate", [0.02, 0.1, 0.5])
@pytest.mark.parametrize("rev", [0, 1])
@pytest.mark.parametrize("lut", list(LUTS))
@pytest.mark.parametrize("rep", range(3))
def test_specialbidirectional_lstm_bf16(rate, rev, lut, rep):
    seq, n_in, hid = 4, 12, 16
    x = sp_bf16(RNG.standard_normal((seq, n_in)), rate).reshape(seq, n_in)
    h0, c0 = sp_bf16(0.5 * RNG.standard_normal(hid), rate), sp_bf16(RNG.standard_normal(hid), rate)
    w = sp_bf16(RNG.standard_normal((4 * hid, n_in)) / 3, rate).reshape(4 * hid, n_in)
    r = sp_bf16(RNG.standard_normal((4 * hid, hid)) / 4, rate).reshape(4 * hid, hid)
    b = sp_bf16(RNG.standard_normal(8 * hid), rate)
    f = bf16_to_f32
    y, c = gl.bidirectional_lstm_bf16(f(x), f(h0), f(c0), f(w), f(r), f(b), LUTS[lut], bool(rev))
    gy, gc = callx("npu_bidirectional_lstm_bf16", x, h0, c0, w, r, b, LUTS[lut], OUT, OUT2,
                   I64(seq), I64(n_in), I64(hid), rev, outs=[2 * seq * hid, 2 * hid])
    np.testing.assert_array_equal(gy.view(np.uint16).reshape(seq, hid), y)
    np.testing.assert_array_equal(gc.view(np.uint16), c)
