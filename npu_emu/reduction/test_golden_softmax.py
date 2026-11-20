# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""reduction/softmax.h entry points against golden_softmax.py, byte for byte."""

import numpy as np
import pytest

import golden_softmax as gs
from golden_float_cvt import bf16_to_f32, f32_to_fp8, store_float
from golden_int_round import INT_RANGE
from npu import I64, NPU_BF16, NPU_I8, NPU_I16, NPU_U16, OUT, raw_rc
from testing import F16T, INT
from testing import assert_same as check
from testing import call_n as go

f32 = np.float32
RNG = np.random.default_rng(20261113)


def rows16(t, R, C, kind):
    """[R][C] bf16 / f16 bits of a given flavour."""
    if kind == "uniform8":
        v = RNG.uniform(-8, 8, (R, C))
    elif kind == "wide":
        v = RNG.uniform(-128, 127, (R, C))
    elif kind == "dominant":
        # one large element over many much smaller ones
        v = RNG.uniform(-14, -6, (R, C))
        v[np.arange(R), RNG.integers(0, C, R)] = RNG.uniform(0, 2, R)
    elif kind == "ties":
        # few distinct values: many equal maxima
        v = RNG.integers(-6, 3, (R, C)) * 0.75
    elif kind == "tiny":
        v = RNG.uniform(-1e-3, 1e-3, (R, C)) * (RNG.random((R, C)) < 0.5)
    else:  # "steps": exponentials spanning the subnormal range
        v = -np.abs(RNG.normal(0, 40, (R, C)))
    return store_float(v.astype(f32), t)


def mask_of(R, C, kind):
    """[R][C/8] mask bytes or None."""
    if kind == "none":
        return None
    if kind == "random":
        m = RNG.random((R, C)) < 0.7
    elif kind == "single":
        m = np.zeros((R, C), bool)
    elif kind == "partial":
        m = np.arange(C)[None, :] < RNG.integers(1, C + 1, (R, 1))
    else:  # "alternating"
        m = np.broadcast_to(np.arange(C) % 2 == 0, (R, C)).copy()
    m[np.arange(R), RNG.integers(0, C, R) if kind != "partial" else 0] = True
    return np.packbits(m, axis=1, bitorder="little")


def unpack(mb, R, C):
    if mb is None:
        return np.ones((R, C), bool)
    return np.unpackbits(mb, axis=1, bitorder="little")[:, :C].astype(bool)


@pytest.mark.parametrize("C", [1, 5, 31, 32, 33, 64, 100, 497, 512, 1031])
@pytest.mark.parametrize("kind", ["uniform8", "wide", "dominant", "ties", "tiny", "steps"])
def test_softmax_bf16(C, kind):
    R = 24
    x = rows16("bf16", R, C, kind)
    check(go("npu_softmax_bf16", x, OUT, I64(R), I64(C), n=(R, C), dt=np.uint16),
          gs.softmax_bf16(x))


def test_softmax_bf16_extreme_rows():
    """Rows headed by -inf / -max (0xff80 / 0xff7f), plus +-0, subnormals, huge negatives."""
    C = 70
    rows = []
    for head in (0xFF80, 0xFF7F):
        for fill in ([0xFF7F, 0xFF40], [0xFF80], [0xFF32, 0xFF33, 0xFF7F, 0xFF80],
                     [0xFF31, 0xFF7F], [0x3F80, 0xC000, 0xFF80], [0x0000, 0x8000]):
            r = RNG.choice(np.array(fill, np.uint16), C)
            r[0] = head
            rows.append(r)
    rows.append(np.full(C, 0xFF80, np.uint16))
    rows.append(np.full(C, 0xFF7F, np.uint16))
    rows.append(RNG.choice(np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F], np.uint16), C))
    rows.append(RNG.choice(np.array([0xC2F0, 0xFF00, 0x7F00], np.uint16), C))
    x = np.stack(rows)
    R = len(rows)
    check(go("npu_softmax_bf16", x, OUT, I64(R), I64(C), n=(R, C), dt=np.uint16),
          gs.softmax_bf16(x))


# (cols, chunk_cols, num_chunks)
LAYOUTS_2P = [(5, 128, 1), (128, 128, 1), (100, 128, 3), (1000, 512, 2), (1025, 512, 3),
              (4000, 2048, 2), (7936, 7936, 1), (2000, 128, 16)]


def go_2pass(x, cc, nc):
    R, C = x.shape
    return go("npu_softmax_2pass_bf16", x, OUT, I64(R), I64(C), I64(cc), I64(nc),
              n=(R, C), dt=np.uint16)


def rescale_rows(R, C, cc):
    """Each chunk raises the running max, a little or by a lot."""
    v = RNG.uniform(-4, 4, (R, C))
    step = np.where(np.arange(R) % 2 == 0, RNG.uniform(0.01, 0.6, R), RNG.uniform(2, 40, R))
    v += (np.arange(C) // cc)[None, :] * step[:, None]
    return store_float(v.astype(f32), "bf16")


@pytest.mark.parametrize("C,cc,nc", LAYOUTS_2P)
@pytest.mark.parametrize("kind", ["uniform8", "wide", "dominant", "ties", "tiny", "steps", "rescale"])
def test_softmax_2pass_bf16(C, cc, nc, kind):
    R = 24
    x = rescale_rows(R, C, cc) if kind == "rescale" else rows16("bf16", R, C, kind)
    check(go_2pass(x, cc, nc), gs.softmax_2pass_bf16(x, cc, nc))


@pytest.mark.parametrize("order", ["ascending", "descending", "shuffled"])
def test_softmax_2pass_bf16_all_codes(order):
    """Every bf16 code except NaN and those whose x / ln2 overflows to +inf."""
    codes = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    with np.errstate(over="ignore", invalid="ignore"):
        s = bf16_to_f32(codes) * gs.INV_LN2_BF16
    codes = codes[~np.isnan(s) & (s != np.inf)]
    codes = codes[np.argsort(bf16_to_f32(codes), kind="stable")]
    if order == "descending":
        codes = codes[::-1]
    C = 500
    codes = np.concatenate([codes, codes[:(-len(codes)) % C]]).reshape(-1, C)
    if order == "shuffled":
        codes = RNG.permuted(codes.reshape(-1)).reshape(-1, C)
    codes = np.ascontiguousarray(codes)
    check(go_2pass(codes, 128, 4), gs.softmax_2pass_bf16(codes, 128, 4))


def test_softmax_2pass_bf16_extreme_rows():
    """x / ln2 overflowing to -inf (x <= 0xff32) in some or all chunks."""
    cc, nc = 128, 3
    C = cc * nc - 20
    big = [0xFF80, 0xFF7F, 0xFF40, 0xFF32, 0xFF31]
    rows = []
    for _ in range(40):
        r = RNG.choice(np.array(big, np.uint16), C)
        for k in range(nc):
            if RNG.random() < 0.3:  # a finite chunk
                seg = slice(k * cc, min(C, (k + 1) * cc))
                r[seg] = store_float(RNG.uniform(-6, 6, r[seg].size).astype(f32), "bf16")
        rows.append(r)
    rows.append(np.full(C, 0xFF80, np.uint16))
    rows.append(np.full(C, 0xFF7F, np.uint16))
    x = np.stack(rows)
    check(go_2pass(x, cc, nc), gs.softmax_2pass_bf16(x, cc, nc))


@pytest.mark.parametrize("C", [5, 100, 512, 1031, 4000])
@pytest.mark.parametrize("kind", ["uniform8", "wide", "dominant", "steps"])
def test_softmax_2pass_bf16_one_chunk_is_softmax_bf16(C, kind):
    """With the row in one chunk the result equals npu_softmax_bf16."""
    R = 24
    x = rows16("bf16", R, C, kind)
    cc = -(-C // 64) * 64
    cc = max(cc, 128)
    check(go_2pass(x, cc, 1), go("npu_softmax_bf16", x, OUT, I64(R), I64(C), n=(R, C),
                                 dt=np.uint16))


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("C", [32, 64, 96, 288, 1024])
@pytest.mark.parametrize("kind", ["uniform8", "wide", "dominant", "ties", "steps"])
@pytest.mark.parametrize("mk", ["none", "random", "partial", "single", "alternating"])
def test_softmax_fp16x16(t, C, kind, mk):
    R = 12
    x = rows16(t, R, C, kind)
    mb = mask_of(R, C, mk)
    check(go("npu_softmax_fp16x16", x, mb, OUT, F16T[t], I64(R), I64(C), n=(R, C),
             dt=np.uint16),
          gs.softmax_fp16x16(x, unpack(mb, R, C), t))


def fp8_codes(R, C, kind):
    """fp8 codes without NaN (0x7f / 0xff)."""
    if kind == "all":
        c = RNG.integers(0, 256, (R, C)).astype(np.uint8)
    elif kind == "small":
        c = f32_to_fp8(RNG.uniform(-4, 4, (R, C)).astype(f32))
    else:  # "dominant"
        c = f32_to_fp8(RNG.uniform(-12, -4, (R, C)).astype(f32))
        c[np.arange(R), RNG.integers(0, C, R)] = f32_to_fp8(RNG.uniform(0, 3, R).astype(f32))
    return np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8)


@pytest.mark.parametrize("C", [64, 128, 512])
@pytest.mark.parametrize("kind", ["all", "small", "dominant"])
@pytest.mark.parametrize("mk", ["none", "random", "partial"])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.37, 1 / 0.013), (0.5, 1000.3), (0.011, 300.7),
                                    (3.3, 1 / 3.0), (1.0, 1.00390625)])  # last: a bf16 tie
def test_softmax_fp8(C, kind, mk, scales):
    R = 10
    x = fp8_codes(R, C, kind)
    mb = mask_of(R, C, mk)
    s, r = (float(f32(v)) for v in scales)
    check(go("npu_softmax_fp8", x, mb, OUT, I64(R), I64(C), s, r, n=(R, C), dt=np.uint8),
          gs.softmax_fp8(x, unpack(mb, R, C), s, r))


LSM_COLS = [1, 31, 32, 63, 64, 65, 128, 129, 192, 256, 320, 640, 705, 706, 1089]


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("C", LSM_COLS)
@pytest.mark.parametrize("kind", ["uniform8", "wide", "dominant", "ties", "tiny", "steps"])
def test_logsoftmax_bf16(t, C, kind):
    R = 8
    x = rows16(t, R, C, kind)
    check(go("npu_logsoftmax_bf16", x, OUT, F16T[t], I64(R), I64(C), n=(R, C), dt=np.uint16),
          gs.logsoftmax_bf16(x, t))


@pytest.mark.parametrize("C", LSM_COLS + [7680])
@pytest.mark.parametrize("kind", ["all", "small", "dominant"])
@pytest.mark.parametrize("scales", [(1.0, 4.0), (0.5, 4.0), (0.37, 1 / 0.11), (2.7, 1 / 9.3),
                                    (0.013, 50.3)])
def test_logsoftmax_fp8(C, kind, scales):
    R = 6
    x = fp8_codes(R, C, kind)
    s, r = (float(f32(v)) for v in scales)
    check(go("npu_logsoftmax_fp8", x, OUT, I64(R), I64(C), s, r, n=(R, C), dt=np.uint8),
          gs.logsoftmax_fp8(x, s, r))


def test_logsoftmax_fp8_all_minimum():
    R, C = 3, 960
    x = np.full((R, C), 0xFE, np.uint8)  # -448
    for s, r in ((0.5, 4.0), (1.0, 1 / 0.013)):
        s, r = float(f32(s)), float(f32(r))
        check(go("npu_logsoftmax_fp8", x, OUT, I64(R), I64(C), s, r, n=(R, C), dt=np.uint8),
              gs.logsoftmax_fp8(x, s, r))


def qvals(t, R, C, kind):
    lo, hi = INT_RANGE[t]
    if kind == "full":
        q = RNG.integers(lo, hi + 1, (R, C))
    elif kind == "narrow":
        mid = RNG.integers(lo, hi + 1, (R, 1))
        q = np.clip(mid + RNG.integers(-20, 21, (R, C)), lo, hi)
    else:  # "edges"
        q = RNG.choice(np.array([lo, lo + 1, hi - 1, hi, (lo + hi) // 2]), (R, C))
    return q.astype(INT[t][1])


def qdq_coeffs(ifm_scale, ofm_scale, ozp, shift_res):
    """Coefficients evaluated in double, stored as fp32."""
    return (float(f32(1.4426950216293335 * ifm_scale)), float(f32(ofm_scale)),
            float(f32(ozp * 2.0 ** -shift_res)))


QDQ_SCALES = {  # (ifm_scale, ofm_scale, ofm_zp) per output type
    "u8": [(0.0037717584, 0.0039215689, 0), (0.0731, 0.0041, 3)],
    "i8": [(0.01, 5e-05, 0), (0.0217, 0.0079, -128)],
    "u16": [(0.0020431830, 1.5259022e-05, 0), (0.00093, 1.7e-05, 1)],
    "i16": [(0.0002, 0.0001, 0), (0.0013, 3.1e-05, -32768)],
}


@pytest.mark.parametrize("t", ["i8", "u8", "i16", "u16"])
@pytest.mark.parametrize("path,C", [(0, 64), (0, 320), (0, 512), (1, 32), (1, 96), (1, 224),
                                    (2, 32)])
@pytest.mark.parametrize("kind", ["full", "narrow", "edges"])
@pytest.mark.parametrize("mk", ["none", "random", "partial"])
@pytest.mark.parametrize("si", [0, 1])
@pytest.mark.parametrize("shift_res", [0, 2])
def test_qdq_softmax(t, path, C, kind, mk, si, shift_res):
    R = 10
    bits = 8 if t in ("i8", "u8") else 16
    x = qvals(t, R, C, kind)
    mb = mask_of(R, C, mk)
    il, os_, zp = qdq_coeffs(*QDQ_SCALES[t][si], shift_res)
    check(go(f"npu_qdq_softmax_a{bits}", x, mb, OUT, INT[t][0], I64(R), I64(C), path, shift_res,
             il, os_, zp, n=(R, C), dt=INT[t][1]),
          gs.qdq_softmax(x, unpack(mb, R, C), t, path, shift_res, il, os_, zp))


@pytest.mark.parametrize("ti,to", [("i8", "i8"), ("u8", "u8"), ("i8", "u8"), ("u8", "i8"),
                                   ("i16", "i16"), ("u16", "u16"), ("i16", "u16"),
                                   ("u16", "i16")])
@pytest.mark.parametrize("C", [128, 256, 640])
@pytest.mark.parametrize("kind", ["full", "narrow", "edges"])
@pytest.mark.parametrize("mk", ["none", "random", "partial"])
@pytest.mark.parametrize("sc", [(0.03125, 1.0, 0), (0.0625, 0.25, 2), (0.0371, 0.0913, 77),
                                (0.000857645, 0.000635821, -1)])
def test_qdq_logsoftmax(ti, to, C, kind, mk, sc):
    R = 8
    bits = 8 if ti in ("i8", "u8") else 16
    x = qvals(ti, R, C, kind)
    mb = mask_of(R, C, mk)
    s_in, s_out, zp = sc
    lo, hi = INT_RANGE[to]
    zp = int(np.clip(zp if zp >= 0 else hi, lo, hi))
    s_in, s_out = float(f32(s_in)), float(f32(s_out))
    check(go(f"npu_qdq_logsoftmax_a{bits}", x, INT[ti][0], mb, OUT, INT[to][0], I64(R), I64(C),
             s_in, s_out, zp, n=(R, C), dt=INT[to][1]),
          gs.qdq_logsoftmax(x, unpack(mb, R, C), ti, to, s_in, s_out, zp))


@pytest.mark.parametrize("args", [
    ("npu_softmax_bf16", None, None, I64(4), I64(0)),
    ("npu_softmax_2pass_bf16", None, None, I64(4), I64(0), I64(128), I64(1)),
    ("npu_softmax_2pass_bf16", None, None, I64(4), I64(64), I64(64), I64(1)),
    ("npu_softmax_2pass_bf16", None, None, I64(4), I64(64), I64(160), I64(1)),
    ("npu_softmax_2pass_bf16", None, None, I64(4), I64(129), I64(128), I64(1)),
    ("npu_softmax_2pass_bf16", None, None, I64(4), I64(64), I64(128), I64(0)),
    ("npu_softmax_fp16x16", None, None, None, NPU_I8, I64(4), I64(32)),
    ("npu_softmax_fp16x16", None, None, None, NPU_BF16, I64(4), I64(48)),
    ("npu_softmax_fp8", None, None, None, I64(4), I64(96), 1.0, 1.0),
    ("npu_logsoftmax_bf16", None, None, NPU_I16, I64(4), I64(64)),
    ("npu_logsoftmax_fp8", None, None, I64(4), I64(64), 0.0, 1.0),
    ("npu_qdq_softmax_a8", None, None, None, NPU_I16, I64(4), I64(64), 0, 0, 1.0, 1.0, 0.0),
    ("npu_qdq_softmax_a16", None, None, None, NPU_I8, I64(4), I64(64), 0, 0, 1.0, 1.0, 0.0),
    ("npu_qdq_softmax_a8", None, None, None, NPU_I8, I64(4), I64(96), 0, 0, 1.0, 1.0, 0.0),
    ("npu_qdq_softmax_a8", None, None, None, NPU_I8, I64(4), I64(64), 2, 0, 1.0, 1.0, 0.0),
    ("npu_qdq_softmax_a8", None, None, None, NPU_I8, I64(4), I64(64), 3, 0, 1.0, 1.0, 0.0),
    ("npu_qdq_logsoftmax_a8", None, NPU_I8, None, None, NPU_I16, I64(4), I64(128), 1.0, 1.0, 0),
    ("npu_qdq_logsoftmax_a16", None, NPU_I16, None, None, NPU_U16, I64(4), I64(192), 1.0, 1.0, 0),
])
def test_bad_args(args):
    assert raw_rc(*args) == -1
