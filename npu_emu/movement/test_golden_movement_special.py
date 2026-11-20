# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Special values (NaN, inf, subnormal, +-0, out of range) through the movement entries: bit
for bit against the numpy reference."""

import ctypes
import struct

import numpy as np
import pytest

import golden_float_cvt as gf
import golden_movement as gm
import golden_qdq as q
import golden_select as gs
import npu
from npu import FILL, I64, OUT, OUT2, U32, U64, DevList, Host, Out
from npu import NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, NPU_I32, NPU_I64
from npu import NPU_U8, NPU_U16, NPU_U32
from testing import (ALL16, ALL8, F32_SPECIAL_BITS, FP8_SPECIAL, HALF, HALF_SPECIAL,
                     INT_ALL as T, call_n, rand_ints, same, shp)

RNG = np.random.default_rng(2031)


def fb(bits):
    """fp32 with the given bit pattern, as a Python float (a NaN keeps sign and payload)."""
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# fp32 coefficient specials: +-0, +-1, subnormals, min normal, max finite, +-inf, NaNs.
SC = [0.0, -0.0, 1.0, -1.0, 0.5, fb(1), fb(0x007FFFFF), fb(0x00800000), fb(0x7F7FFFFF),
      fb(0xFF7FFFFF), fb(0x7F800000), fb(0xFF800000), fb(0x7FC00000), fb(0xFFC00000),
      fb(0x7FC12345), 300.0]
SO = [(s, o) for s in SC for o in SC]
SHIFTS = [0, 1, -1, 31, -31, 200, -200, 1100, -1100]


def sample(n, seed):
    r = np.random.default_rng(seed)
    return [SO[k] for k in r.choice(len(SO), n, replace=False)] + [(fb(0x7FC00000), 0.0),
                                                                  (fb(0x7F800000), 0.0),
                                                                  (0.0, fb(0xFF800000))]


SO_SMALL = sample(20, 5)


def vals(t, n=300):
    """The ends and the middle of the type plus a random sample."""
    info = np.iinfo(T[t][1])
    edge = [info.min, info.min + 1, -1 if info.min else 1, 0, 1, info.max - 1, info.max]
    if info.bits == 8:
        return np.arange(info.min, info.max + 1).astype(T[t][1])
    return np.concatenate([np.array(edge), RNG.integers(info.min, info.max, n,
                                                        endpoint=True)]).astype(T[t][1])


def pairs(bits):
    s = ["i8", "u8"] if bits == 8 else ["i16", "u16"]
    return [(a, b) for a in s for b in s]


def run(name, *args, n, t):
    return call_n(name, *args, n=n, dt=T[t][1])


# ---------------------------------------------------------------- half-float select

QNAN = {"bf16": np.uint16(0x7FC0), "f16": np.uint16(0x7E00)}
NAN_FREE_FILL = {"bf16": np.uint16(0x7F7F), "f16": np.uint16(0x7BFF)}


def hs(t):
    return HALF_SPECIAL[t]


def hf(t, bits):
    """A half code as the fp32 value a coefficient takes."""
    return float(gf.load_float(np.uint16(bits), t))


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_sp_clip_half(t):
    nt, ht = HALF[t]
    for lo in SC + [hf(ht, b) for b in HALF_SPECIAL[t]]:
        for hi in (SC[:6] + SC[8:14]):
            got = call_n("npu_elementwise_clip_bf16", ALL16, nt, OUT, I64(65536), lo, hi,
                         dt=np.uint16, n=65536)
            same(got, gs.clip_half(ALL16, ht, lo, hi), f"clip {lo} {hi}")


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("op", ["equal", "greater", "greater_or_equal", "max", "min"])
def test_sp_bcast_half(op, t):
    """Every code (incl. signaling NaN) against every special, both operand orders."""
    nt, ht = HALF[t]
    sp = hs(ht)
    out_dt = np.uint16 if op in ("max", "min") else np.int8
    for k in range(len(sp)):
        full = ALL16.reshape(1, 256, 256)
        other = np.full((1, 256, 256), sp[k], np.uint16)
        for a, b in ((full, other), (other, full)):
            got = call_n(f"npu_elementwise_broadcasting3d_{op}_bf16", a, b, nt, OUT,
                         shp(a.shape), shp(b.shape), dt=out_dt, n=a.size)
            same(got, gs.bcast_half(op, a, 0, b, 0, ht, 1, 256, 256), f"{op} {sp[k]:#x}")


def fp8_rows(n=4096):
    base = np.concatenate([RNG.permutation(ALL8) for _ in range(4)])
    sp = RNG.choice(FP8_SPECIAL, n)
    return np.concatenate([base, sp])


@pytest.mark.parametrize("scale", SC)
@pytest.mark.parametrize("op", ["abs", "neg"])
def test_sp_fp8_abs_neg(op, scale):
    x = fp8_rows()
    got = call_n(f"npu_elementwise_{op}_fp8", x, OUT, I64(len(x)), scale, dt=np.uint8, n=len(x))
    same(got, gs.fp8_unary(x, op, scale), f"{op} {scale}")


@pytest.mark.parametrize("scale", SC)
def test_sp_fp8_clip(scale):
    x = fp8_rows()
    for lo, hi in [(-6.0, 8.0), (0.0, 0.0), (-0.0, 0.0), (fb(0x7FC00000), 1.0),
                   (-1.0, fb(0x7FC00000)), (fb(0xFF800000), fb(0x7F800000)),
                   (fb(0x7F800000), fb(0xFF800000)), (-500.0, 500.0), (fb(1), fb(0x007FFFFF))]:
        got = call_n("npu_elementwise_clip_fp8", x, OUT, I64(len(x)), scale, lo, hi,
                     dt=np.uint8, n=len(x))
        same(got, gs.fp8_unary(x, "clip", scale, lo, hi), f"clip {scale} {lo} {hi}")


@pytest.mark.parametrize("op", ["equal", "greater", "greater_or_equal", "max", "min"])
def test_sp_bcast_fp8(op):
    H, W, C = 2, 3, 40
    out_dt = np.uint8 if op in ("max", "min") else np.int8
    for sa in SC:
        for sb in SC[::3] + [SC[12], SC[10]]:
            a = RNG.choice(FP8_SPECIAL, (H, W, C)).astype(np.uint8)
            b = RNG.choice(np.concatenate([FP8_SPECIAL, ALL8[:40]]), (H, W, C)).astype(np.uint8)
            got = call_n(f"npu_elementwise_broadcasting3d_{op}_fp8", a, sa, b, sb, OUT,
                         shp((H, W, C)), shp((H, W, C)), dt=out_dt, n=a.size)
            same(got, gs.bcast_fp8(op, a, 0, sa, b, 0, sb, H, W, C), f"{op} {sa} {sb}")


@pytest.mark.parametrize("w,t", [("maxpool_bf16", "bf16"), ("maxpool_bf16", "f16")])
def test_sp_maxpool_half(w, t):
    nt, ht = HALF[t]
    Hi, Wi, C = 6, 6, 20
    x = RNG.choice(hs(ht), (Hi, Wi, C))
    got = call_n(f"npu_{w}", x, nt, OUT, I64(Hi), I64(Wi), I64(C), 2, 2, 1, 1, dt=np.uint16,
                 n=5 * 5 * C)
    same(got, gs.maxpool_half(x, ht, 2, 2, 1, 1), "maxpool")


def test_sp_maxpool_fp8():
    Hi, Wi, C = 6, 6, 20
    x = RNG.choice(FP8_SPECIAL, (Hi, Wi, C)).astype(np.uint8)
    for c0 in SC[::2]:
        for c2 in SC:
            for clip, lo, hi in ((0, 0.0, 0.0), (1, -1.0, fb(0x7FC00000)), (1, fb(0xFF800000), 6.0)):
                c0a, c2a = np.array([c0], np.float32), np.array([c2], np.float32)
                got = call_n("npu_maxpool_fp8", x, OUT, I64(Hi), I64(Wi), I64(C), 2, 2, 1, 1,
                             c0a, c2a, I64(1), clip, lo, hi, dt=np.uint8, n=5 * 5 * C)
                same(got, gs.maxpool_fp8(x, 2, 2, 1, 1, c0, c2, clip, lo, hi), f"{c0} {c2}")


def test_sp_globalmaxpool_fp8():
    H, W, C = 3, 3, 21
    x = RNG.choice(FP8_SPECIAL, (H, W, C)).astype(np.uint8)
    for c0 in SC:
        for c1 in SC:
            got = call_n("npu_globalmaxpool_fp8", x, OUT, I64(H), I64(W), I64(C), c0, c1,
                         dt=np.uint8, n=C)
            same(got, gs.globalmaxpool_fp8(x, c0, c1), f"{c0} {c1}")


@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_sp_globalmaxpool_half(t):
    nt, ht = HALF[t]
    H, W, C = 3, 4, 40
    x = RNG.choice(hs(ht), (H, W, C))
    got = call_n("npu_globalmaxpool_bf16", x, nt, OUT, I64(H), I64(W), I64(C), dt=np.uint16, n=C)
    same(got, gs.globalmaxpool_half(x, ht), "globalmaxpool")


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("is_max", [True, False])
def test_sp_reduce_half(t, is_max):
    nt, ht = HALF[t]
    H, W, C = 3, 4, 70
    x = RNG.choice(hs(ht), (H, W, C))
    name = "npu_reduce_max_c64_bf16" if is_max else "npu_reduce_min_c64_bf16"
    for axes in range(1, 8):
        n = (1 if axes & 4 else H) * (1 if axes & 2 else W) * (1 if axes & 1 else C)
        got = call_n(name, x, nt, OUT, I64(H), I64(W), I64(C), axes, dt=np.uint16, n=n)
        same(got, gs.reduce_half(x, ht, axes, is_max), f"axes {axes}")


ARG_ENTRIES = [("arg_max_c64_bf16", True), ("arg_min_c64_bf16", False)]


@pytest.mark.parametrize("t", ["bf16", "f16"])
@pytest.mark.parametrize("name,is_max", ARG_ENTRIES)
def test_sp_arg_half(name, is_max, t):
    nt, ht = HALF[t]
    H, W, C = 3, 4, 70
    x = RNG.choice(hs(ht), (H, W, C))
    x[:, :, 64:] = NAN_FREE_FILL[ht]                       # fill past one lane block with one value
    x[0, 0, :] = QNAN[ht]                        # an all-NaN row: no element wins
    vals_ = gf.load_float(x, ht)
    init = float(gf.load_float(np.uint16((gs.MIN_NORMAL if is_max else gs.MAX_FINITE)[ht]), ht))
    for ot, ou in (("u8", np.uint8), ("u16", np.uint16)):
        for axis in range(3):
            n = H * W * C // (H, W, C)[axis]
            got = call_n(f"npu_{name}", x, nt, OUT, npu.NPU_U8 if ot == "u8" else NPU_U16,
                         I64(H), I64(W), I64(C), axis, dt=ou, n=n)
            want = gs.index_out(gs.arg_reduce(vals_, axis, is_max, init, 8 if ot == "u8" else 16),
                                ot)
            same(got, want, f"{ot} axis {axis}")


@pytest.mark.parametrize("largest", [1, 0])
@pytest.mark.parametrize("t", ["bf16", "f16"])
def test_sp_topk_half(t, largest):
    """NaN (and infinities, zeros) in the rows."""
    nt, ht = HALF[t]
    f16 = ht == gs.F16
    for rows, W, K in [(2, 100, 9), (1, 64, 70), (2, 20, 30)]:
        x = RNG.choice(hs(ht), (rows, W))
        x[0, :] = QNAN[ht]                       # all NaN
        vals_ = gf.load_float(x, ht)
        sent = -np.inf if largest else np.inf
        sraw = (0xFC00 if f16 else 0xFF80) if largest else (0x7C00 if f16 else 0x7F80)
        res = npu.callx("npu_topk2d_bf16", x, nt, OUT, OUT2, NPU_U16, I64(rows), I64(W), I64(K),
                        largest, outs=[rows * K * 2, rows * K * 2])
        gv, gi = res[0].view(np.uint16), res[1].view(np.uint16)
        rv, ri = gs.topk(vals_, x, W, K, largest, sent, np.array(sraw).astype(np.uint16), 16)
        same(gv, rv, "values")
        same(gi, ri.astype(np.uint16), "indices")


@pytest.mark.parametrize("name,is_max", [("arg_max_c64_fp8", True)])
def test_sp_arg_fp8(name, is_max):
    H, W, C = 3, 4, 70
    x = RNG.choice(FP8_SPECIAL, (H, W, C)).astype(np.uint8)
    x[0, 0, :] = 0x7F
    vals_ = gf.fp8_to_f32(x).astype(np.float64)
    for axis in range(3):
        n = H * W * C // (H, W, C)[axis]
        got = call_n(f"npu_{name}", x, OUT, NPU_U16, I64(H), I64(W), I64(C), axis, dt=np.uint16,
                     n=n)
        same(got, gs.index_out(gs.arg_reduce(vals_, axis, is_max, -448.0, 16), "u16"), "arg fp8")


# ---------------------------------------------------------------- qdq

@pytest.mark.parametrize("bits", [8, 16])
def test_sp_abs_neg_clip(bits):
    for it, ot in pairs(bits)[:2] + pairs(bits)[3:]:
        x = vals(it)
        for c1, c0 in SO:
            got = run(f"npu_elementwise_neg_a{bits}", x, T[it][0], OUT, T[ot][0], I64(len(x)),
                      c1, c0, n=len(x), t=ot)
            same(got, q.requant(x, c1, c0, ot), f"neg {c1} {c0}")
        for c1, c0 in SO_SMALL:
            for zp in (0.0, fb(0x7F800000), fb(0x7FC00000), -5.0):
                got = run(f"npu_elementwise_abs_a{bits}", x, T[it][0], OUT, T[ot][0],
                          I64(len(x)), c1, c0, zp, n=len(x), t=ot)
                same(got, q.abs_(x, c1, c0, zp, ot), f"abs {c1} {c0} {zp}")
            for lo, hi in ((-11.5, 11.5), (fb(0x7FC00000), 9.0), (-9.0, fb(0x7FC00000)),
                           (fb(0xFF800000), fb(0x7F800000)), (fb(0x7F800000), fb(0xFF800000)),
                           (-0.0, 0.0), (fb(1), fb(0x007FFFFF))):
                got = run(f"npu_elementwise_clip_a{bits}", x, T[it][0], OUT, T[ot][0],
                          I64(len(x)), c1, c0, lo, hi, n=len(x), t=ot)
                same(got, q.clip(x, c1, c0, lo, hi, ot), f"clip {c1} {c0} {lo} {hi}")


@pytest.mark.parametrize("bits", [8, 16])
def test_sp_compare_max(bits):
    for at, bt in pairs(bits)[:2] + pairs(bits)[3:]:
        a = vals(at)[:200]
        b = np.roll(vals(bt), 7)[:200]
        for k in range(len(SO_SMALL)):
            sa, za = SO_SMALL[k]
            sb, zb = SO_SMALL[(k * 7 + 3) % len(SO_SMALL)]
            for op in ("equal", "greater"):
                got = run(f"npu_elementwise_{op}_a{bits}", a, T[at][0], b, T[bt][0], OUT,
                          I64(len(a)), sa, sb, za, zb, n=len(a), t="i8")
                same(got, q.compare(a, b, sa, sb, za, zb, op), f"{op}")
            for so, zo in ((2.0, 0.5), (fb(0x7F800000), 0.0), (fb(0x7FC00000), 1.0),
                           (0.0, fb(0xFF800000)), (fb(0x007FFFFF), fb(0x7F7FFFFF))):
                got = run(f"npu_elementwise_max_a{bits}", a, T[at][0], b, T[bt][0], OUT,
                          T[at][0], I64(len(a)), sa, sb, so, za, zb, zo, n=len(a), t=at)
                same(got, q.minmax(a, b, sa, sb, so, za, zb, zo, at, "max"), "max")


@pytest.mark.parametrize("bits", [8, 16])
def test_sp_qdq_min(bits):
    shape, bshape = (3, 4, 33), (1, 4, 1)
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        a, b = rand_ints(RNG, T[t][1], shape), rand_ints(RNG, T[t][1], bshape)
        for k in range(len(SO_SMALL)):
            sa, za = SO_SMALL[k]
            sb, zb = SO_SMALL[(k * 5 + 1) % len(SO_SMALL)]
            for so, zo in ((2.0, 0.5), (fb(0x7F800000), 0.0), (fb(0x7FC00000), 1.0),
                           (0.0, fb(0xFF800000)), (fb(0x007FFFFF), fb(0x7F7FFFFF))):
                got = run(f"npu_qdq_elementwise_broadcasting3d_min_a{bits}", a, b, T[t][0], OUT,
                          shp(shape), shp(bshape), sa, sb, so, za, zb, zo, n=a.size, t=t)
                same(got, q.minmax(a, b, sa, sb, so, za, zb, zo, t, "min"), "min")


@pytest.mark.parametrize("bits", [8, 16])
def test_sp_qdq_where_reduce_pool(bits):
    shape, cshape = (3, 4, 33), (3, 1, 33)
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        ct = np.int8 if bits == 8 else np.int16
        cond = RNG.integers(-2, 3, cshape).astype(ct)
        x = vals(t, 400)[:3 * 4 * 33].reshape(shape) if bits == 16 else \
            rand_ints(RNG, T[t][1], shape)
        y = int(rand_ints(RNG, T[t][1], 1)[0])
        red = q.reduce(x, 5, "max")
        mp = q.maxpool(x, 2, 2, 1, 1)
        gm_ = np.asarray(x, np.int64).max(axis=(0, 1))
        for (scale, offset), shift in zip(SO_SMALL * 2, SHIFTS * 5):
            got = run(f"npu_qdq_elementwise_broadcasting3d_where_a{bits}", cond, x, T[t][0], OUT,
                      shp(shape), shp(cshape), y, 1, scale, offset, n=x.size, t=t)
            same(got, q.requant(q.where(cond, x, y, 1), scale, offset, t), f"where {scale} {offset}")
            for op in ("max", "min"):
                want = q.reduce(x, 5, op)
                got = run(f"npu_qdq_reduce_{op}_c64_a{bits}", x, T[t][0], OUT, I64(3), I64(4),
                          I64(33), 5, offset, scale, shift, n=want.size, t=t)
                same(got, q.requant(want, scale, offset, t, shift), f"reduce {op} {shift}")
            got = run(f"npu_qdq_maxpool_a{bits}", x, T[t][0], OUT, T[t][0], I64(3), I64(4),
                      I64(33), 2, 2, 1, 1, offset, scale, shift, n=mp.size, t=t)
            same(got, q.requant(mp, scale, offset, t, shift), f"maxpool {shift}")
            got = run(f"npu_qdq_globalmaxpool_a{bits}", x, T[t][0], OUT, T[t][0], I64(3), I64(4),
                      I64(33), offset, scale, shift, n=gm_.size, t=t)
            same(got, q.requant(gm_, scale, offset, t, shift), f"globalmaxpool {shift}")


@pytest.mark.parametrize("bits", [8, 16])
def test_sp_qdq_concat_gather_resize(bits):
    for t in (["i8", "u8"] if bits == 8 else ["i16", "u16"]):
        a, b = rand_ints(RNG, T[t][1], (4, 20)), rand_ints(RNG, T[t][1], (4, 9))
        x = rand_ints(RNG, T[t][1], (5, 6, 7))
        g_in = rand_ints(RNG, T[t][1], (13, 21))
        idx = RNG.integers(0, 13, (7, 21)).astype(np.int32)
        idx[0, :5] = [-1, 13, 1 << 30, -(1 << 31), (1 << 31) - 1]
        sy = np.array([0, 4, -1, 5, 1 << 30, -(1 << 31), 2], np.int32)
        sx = np.array([0, 5, 6, -1, 3, (1 << 31) - 1], np.int32)
        for scale, offset in SO_SMALL:
            got = run(f"npu_qdq_concat_a{bits}", a, b, T[t][0], OUT, T[t][0], I64(4), I64(20),
                      I64(9), scale, offset, n=4 * 29, t=t)
            same(got, q.requant(q.concat([a, b]), scale, offset, t), f"concat {scale} {offset}")
            ch = Host((ctypes.c_int64 * 2)(20, 9))
            got = run(f"npu_qdq_concat_multi_c_fusion_a{bits}", 2, DevList([a, b]), T[t][0], OUT,
                      T[t][0], I64(4), ch, scale, offset, n=4 * 29, t=t)
            same(got, q.requant(q.concat([a, b]), scale, offset, t), "concat multi")
            s_ = Host((ctypes.c_float * 2)(scale, offset))
            o_ = Host((ctypes.c_float * 2)(offset, scale))
            got = run("npu_qdq_concat_multi_c_fusion", 2, DevList([a, b]), T[t][0], OUT, T[t][0],
                      I64(4), ch, s_, o_, n=4 * 29, t=t)
            same(got, q.concat([q.requant(a, scale, offset, t), q.requant(b, offset, scale, t)]),
                 "concat per input")
            got = run("npu_qdq_gather_element_2din_2dindices_axis0", g_in, T[t][0], idx, OUT,
                      T[t][0], I64(13), I64(21), I64(7), scale, offset, n=idx.size, t=t)
            untouched = np.frombuffer(bytes([FILL]) * np.dtype(T[t][1]).itemsize, T[t][1])[0]
            same(got, q.gather_axis0(g_in, idx, scale, offset, t, untouched), "gather")
            for rq in (0, 1):
                got = run("npu_qdq_resize_nearest", x, T[t][0], OUT, I64(5), I64(6), I64(7),
                          I64(7), I64(6), sy, sx, rq, scale, offset, n=7 * 6 * 7, t=t)
                same(got, q.resize_nearest(x, sy, sx, rq, scale, offset, t, untouched), f"resize {rq}")


class GridQdq(ctypes.Structure):
    _fields_ = [("grid_zp", ctypes.c_int32), ("grid_scale", ctypes.c_float),
                ("ifm_zp", ctypes.c_int32), ("ifm_scale", ctypes.c_float),
                ("ofm_zp", ctypes.c_int32), ("ofm_scale", ctypes.c_float)]


@pytest.mark.parametrize("variant", ["a8", "a16", "a8w8"])
@pytest.mark.parametrize("align", [0, 1])
@pytest.mark.parametrize("pad", [0, 1])
def test_sp_gridsample(variant, align, pad):
    H, W, C, Ho, Wo = 5, 6, 3, 4, 5
    gq_list = [(0, fb(0x7F800000), 0, 1.0, 0, 1.0), (7, fb(0x7FC00000), 3, 0.5, 1, 2.0),
               (0, fb(0xFF800000), -2, 1.0, 5, 1.0), (0, 0.0, 0, 1.0, 0, 1.0),
               (0, fb(0x7F7FFFFF), 0, 1.0, 0, 1.0), (0, fb(1), 0, 1.0, 0, 1.0),
               (128, 1 / 127, 0, fb(0x7F800000), 0, 1.0), (128, 1 / 127, 0, fb(0x7FC00000), 0, 1.0),
               (128, 1 / 127, 5, 1.0, 0, fb(0x7F800000)), (128, 1 / 127, 5, 1.0, 0, fb(0x7FC00000)),
               (128, 1 / 127, 5, fb(0x7F800000), 0, 1.0), (128, 1 / 127, 5, fb(0x007FFFFF), 0, 0.0),
               (128, 1 / 127, 0, 0.0, 0, fb(0x7F800000)), (128, 1 / 127, 3, fb(0xFF800000), 0, 1.0)]
    for t in (["i16", "u16"] if variant == "a16" else ["i8", "u8"]):
        x = vals(t)[:H * W * C].reshape(H, W, C) if variant == "a16" else \
            rand_ints(RNG, T[t][1], (H, W, C))
        for gq in gq_list:
            if variant == "a8w8":
                gt = "u8"
                grid = rand_ints(RNG, T[gt][1], (Ho, Wo, 2))
                args = (grid, T[gt][0])
            else:
                grid = RNG.integers(0, 65536, (Ho, Wo, 2)).astype(np.uint16)
                grid[:2] = RNG.integers(0, 256, (2, Wo, 2)) + gq[0]
                args = (grid,)
            cq = GridQdq(*gq)
            got = run(f"npu_qdq_gridsample_nearest_{variant}", x, T[t][0], *args, OUT, I64(H),
                      I64(W), I64(C), I64(Ho), I64(Wo), align, pad, Host(ctypes.byref(cq)),
                      n=Ho * Wo * C, t=t)
            gqf = (gq[0], np.float32(cq.grid_scale), gq[2], np.float32(cq.ifm_scale), gq[4],
                   np.float32(cq.ofm_scale))
            same(got, q.gridsample_nearest(x, grid.astype(np.int64), align, pad, gqf, t),
                 f"grid {gq}")


def test_sp_qdq_fused():
    for t in ("i8", "u8", "i16", "u16"):
        x = np.resize(vals(t), 420)
        for c1, c0 in SO_SMALL:
            got = run("npu_qdq_reshape_innermost_requantize_fused", x, OUT, T[t][0], I64(x.size),
                      c1, c0, n=x.size, t=t)
            same(got, q.requant(x, c1, c0, t), f"reshape {c1} {c0}")
            y = x.reshape(6, 70)
            got = run("npu_qdq_slice_innermost_requantize_fused", y, OUT, T[t][0], I64(6), I64(70),
                      I64(3), I64(60), I64(4), c1, c0, n=6 * 15, t=t)
            same(got, q.requant(y[:, 3:60:4], c1, c0, t), "slice")
            got = run("npu_qdq_transpose_k8n8_requantize_fused", y, OUT, T[t][0], I64(6), I64(70),
                      c1, c0, n=y.size, t=t)
            same(got, q.requant(y.T, c1, c0, t), "transpose")


# ---------------------------------------------------------------- movement

SPECIAL = {NPU_F32: F32_SPECIAL_BITS, NPU_BF16: (F32_SPECIAL_BITS >> 16).astype(np.uint16),
           NPU_F16: HALF_SPECIAL["f16"], NPU_FP8: FP8_SPECIAL}
WIDTH = {NPU_F32: np.uint32, NPU_BF16: np.uint16, NPU_F16: np.uint16, NPU_FP8: np.uint8}
FT = [NPU_F32, NPU_BF16, NPU_F16, NPU_FP8]


def spec(shape, t):
    return RNG.choice(SPECIAL[t], shape).astype(WIDTH[t])


def mv(name, *args, want):
    got = call_n(name, *args, n=tuple(want.shape), dt=want.dtype)
    same(got, want, name)


@pytest.mark.parametrize("t", FT)
def test_sp_copy_slice_concat_expand(t):
    x = spec((9, 40), t)
    mv("npu_generic_mem_copy", x, OUT, t, I64(x.size), want=x)
    mv("npu_generic_reshape_innermost", x, OUT, t, I64(x.size), want=x)
    mv("npu_generic_slice_innermost", x, OUT, t, I64(9), I64(40), I64(3), I64(37), I64(2),
       want=gm.slice_inner(x, 3, 37, 2))
    b = spec((9, 7), t)
    mv("npu_generic_concat_16b_inner", x, I64(40), b, I64(7), OUT, t, I64(9),
       want=gm.concat_rows([x, b]))
    e = spec((9, 4), t)
    mv("npu_generic_expand_innermost", e, OUT, t, I64(9), I64(4), I64(20), want=gm.expand_inner(e, 20))


@pytest.mark.parametrize("t", FT)
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_sp_pad(t, mode):
    x = spec((4, 5, 6), t)
    pads = (1, 2, 2, 1, 3, 2)
    for bits in SPECIAL[t][:6]:
        mv("npu_generic_inplace_pad3d", x, OUT, t, I64(4), I64(5), I64(6),
           Host(np.array(pads, np.int64)), mode, U64(int(bits)), want=gm.pad3d(x, pads, mode, int(bits)))


@pytest.mark.parametrize("t", FT)
def test_sp_transpose_space_gather_resize(t):
    x = spec((3, 5, 7, 4), t)
    p = [0, 2, 3, 1]
    mv("npu_generic_transpose_unaligned", x, OUT, t, Host(np.array(x.shape, np.int64)),
       Host(np.array(p, np.int32)), want=np.transpose(x, p))
    s = spec((6, 8, 3), t)
    mv("npu_generic_space_to_depth", s, OUT, t, I64(6), I64(8), I64(3), I64(2),
       want=gm.space_to_depth_hwc(s, 2))
    d = spec((5, 6, 7), t)
    idx = np.array([0, 4, 5, -1, 3, 1 << 20, 2, -(1 << 31)], np.int32)
    shape = list(d.shape)
    fill = np.frombuffer(bytes([FILL]) * d.dtype.itemsize, d.dtype)[0]
    shape[1] = len(idx)
    mv("npu_generic_gather_3din_1dindices", d, t, I64(5), I64(6), I64(7), 1, idx, NPU_I32,
       I64(len(idx)), OUT, want=gm.gather_axis(d, 1, idx, fill))
    idx16 = np.array([0, 4, 5, -1, 3, 32767, 2, -32768], np.int16)
    mv("npu_generic_gather_3din_1dindices", d, t, I64(5), I64(6), I64(7), 0, idx16, NPU_I16,
       I64(len(idx16)), OUT, want=gm.gather_axis(d, 0, idx16, fill))
    idx32 = np.array([0, 4, 5, 1 << 31, 3, 0xFFFFFFFF, 2, 0x80000005], np.uint32)
    mv("npu_generic_gather_3din_1dindices", d, t, I64(5), I64(6), I64(7), 2, idx32, NPU_U32,
       I64(len(idx32)), OUT, want=gm.gather_axis(d, 2, idx32, fill))
    ge = spec((6, 9), t)
    gi = RNG.integers(-2, 9, (6, 12)).astype(np.int32)
    gi[0, :4] = [-(1 << 31), (1 << 31) - 1, 9, -1]
    mv("npu_generic_gather_element_2din_2dindices_innermost", ge, t, I64(6), I64(9), gi, NPU_I32,
       I64(12), OUT, want=gm.gather_elements(ge, gi, 1, fill))
    gi0 = RNG.integers(-2, 8, (10, 9)).astype(np.int32)
    gi0[0, :4] = [-(1 << 31), (1 << 31) - 1, 6, -1]
    mv("npu_generic_gather_element_2din_2dindices_axis0", ge, t, I64(6), I64(9), gi0, NPU_I32,
       I64(10), OUT, want=gm.gather_elements(ge, gi0, 0, fill))
    if t != NPU_F32:
        r = spec((7, 9, 16), t)
        ih = np.array([0, 3, -1, 6, 7, 1 << 30, -(1 << 31), 2], np.int32)
        iw = np.array([8, 0, 9, -1, 4], np.int32)
        ok = ((ih >= 0) & (ih < 7))[:, None] & ((iw >= 0) & (iw < 9))[None, :]
        want = np.where(ok[..., None], gm.resize_hwc(r, np.where(ih >= 0, np.minimum(ih, 6), 0),
                                                     np.where(iw >= 0, np.minimum(iw, 8), 0)), fill)
        mv("npu_resize_nearest_bf16", r, OUT, t, I64(7), I64(9), I64(16), I64(8), I64(5), ih, iw,
           want=want)


def test_sp_resize_bfp16_oob():
    Hi, Wi, C = 4, 5, 16
    x = RNG.integers(0, 256, (Hi, Wi, C // 8 * 9), dtype=np.uint8)
    ih = np.array([0, 3, -1, 4, 2], np.int32)
    iw = np.array([4, 5, 0, -7], np.int32)
    ok = ((ih >= 0) & (ih < Hi))[:, None] & ((iw >= 0) & (iw < Wi))[None, :]
    base = gm.resize_hwc(x, np.clip(ih, 0, Hi - 1), np.clip(iw, 0, Wi - 1))
    want = np.where(ok[..., None], base, np.uint8(FILL))
    mv("npu_resize_nearest_bfp16", x, OUT, I64(Hi), I64(Wi), I64(C), I64(5), I64(4), ih, iw,
       want=want)
