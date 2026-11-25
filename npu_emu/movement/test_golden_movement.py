# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""movement.h entry points against the numpy reference, bit for bit."""

import numpy as np
import pytest

import golden_movement as gm
from golden_block_fp import encode_bfp16
from golden_int_round import Rnd
from npu import FILL, DevList, Host, I64, Out, callx, raw_rc
from npu import (NPU_BF16, NPU_BFP16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, NPU_I32,
                 NPU_I64, NPU_U8, NPU_U16, NPU_U32)
from testing import call_n, rejected

RNG = np.random.default_rng(7)
# npu_dtype -> unsigned storage of the same width (movement only copies bits)
WIDTH = {NPU_I8: np.uint8, NPU_U8: np.uint8, NPU_FP8: np.uint8, NPU_I16: np.uint16,
         NPU_U16: np.uint16, NPU_F16: np.uint16, NPU_BF16: np.uint16}
TYPES = [NPU_I8, NPU_FP8, NPU_BF16, NPU_U16]


def rand(shape, t):
    dt = WIDTH[t]
    return RNG.integers(0, np.iinfo(dt).max, size=shape, dtype=dt, endpoint=True)


def run(name, *args, out_like):
    """Single output shaped like out_like (array or (shape, dtype))."""
    shape, dt = (out_like.shape, out_like.dtype) if isinstance(out_like, np.ndarray) else out_like
    return call_n(name, *args, n=tuple(shape), dt=dt)


def same(got, want):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    bad = np.nonzero(got.reshape(-1) != want.reshape(-1))[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} differ, first at {bad[:6].tolist()}: "
                           f"got {got.reshape(-1)[bad[:6]].tolist()} "
                           f"want {want.reshape(-1)[bad[:6]].tolist()}")


def fill_of(dt):
    return np.frombuffer(bytes([FILL]) * np.dtype(dt).itemsize, dt)[0]


def two(name, *args, w0, w1):
    """An entry with outputs Out(0), Out(1) shaped like w0, w1."""
    es = w0.dtype.itemsize
    o0, o1 = callx(name, *args, outs=[w0.size * es, w1.size * es])
    return o0.view(w0.dtype).reshape(w0.shape), o1.view(w1.dtype).reshape(w1.shape)


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("c1,s1,n1,c2,s2,n2", [(24, 0, 23, 48, 2, 40), (64, 0, 63, 4, 0, 1),
                                                (256, 1, 255, 748, 2, 745), (5, 5, 0, 3, 0, 3)])
def test_generic_concat(t, c1, s1, n1, c2, s2, n2):
    rows = 37
    a, b = rand((rows, c1), t), rand((rows, c2), t)
    want = gm.generic_concat(a, b, s1, n1, s2, n2)
    same(run("npu_generic_concat", a, t, b, Out(), I64(rows), I64(c1), I64(s1), I64(n1), I64(c2),
             I64(s2), I64(n2), out_like=want), want)


@pytest.mark.parametrize("c1,n1,c2,n2", [(64, 64, 16, 8), (24, 16, 40, 40), (8, 0, 8, 8)])
def test_generic_concat_bfp16(c1, n1, c2, n2):
    """Values counted in 8-value blocks of 9 bytes."""
    rows = 9
    a = RNG.integers(0, 256, (rows, c1 // 8 * 9), dtype=np.uint8)
    b = RNG.integers(0, 256, (rows, c2 // 8 * 9), dtype=np.uint8)
    want = gm.generic_concat(a, b, 0, n1 // 8 * 9, 0, n2 // 8 * 9)
    same(run("npu_generic_concat", a, NPU_BFP16, b, Out(), I64(rows), I64(c1), I64(0), I64(n1),
             I64(c2), I64(0), I64(n2), out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
def test_generic_concat_16b_inner(t):
    a, b = rand((50, 13), t), rand((50, 70), t)
    want = gm.generic_concat_16b_inner(a, b)
    same(run("npu_generic_concat_16b_inner", a, t, b, Out(), I64(50), I64(13), I64(70),
             out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
def test_generic_concat_16b_c6463(t):
    a, b = rand((40, 63), t), rand((40, 1), t)
    want = gm.generic_concat_16b_c6463(a, b)
    same(run("npu_generic_concat_16b_c6463", a, t, b, Out(), I64(40), out_like=want), want)


@pytest.mark.parametrize("name", ["generic_concat_16b_multi", "generic_concat_multi_c_fusion"])
@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("widths", [[3, 5], [64, 1, 17], [1, 2, 3, 4, 5, 6], [0, 7, 9]])
def test_generic_concat_multi(name, t, widths):
    rows = 29
    parts = [rand((rows, w), t) for w in widths]
    want = getattr(gm, name)(parts)
    same(run(f"npu_{name}", DevList(parts), t, Out(), I64(rows), len(widths),
             Host(np.array(widths, np.int64)), out_like=want), want)


def bfp16_rows(rows, n, normalized):
    """rows of ceil(n/8) bfp16 blocks: random bytes or encoded random floats."""
    nb = -(-n // 8)
    if not normalized:
        return RNG.integers(0, 256, rows * nb * 9, dtype=np.uint8)
    x = (RNG.standard_normal(rows * nb * 8) * np.exp2(RNG.integers(-20, 20, rows * nb * 8)))
    return encode_bfp16(x.astype(np.float32), Rnd.kHalfEven)


@pytest.mark.parametrize("n1,n2", [(1, 1), (7, 9), (63, 1), (64, 64), (65, 3), (100, 200),
                                   (127, 1), (128, 128), (129, 255), (192, 64), (255, 745),
                                   (300, 7), (8, 8), (512, 512)])
@pytest.mark.parametrize("normalized", [False, True])
def test_concat_unaligned_bfp16(n1, n2, normalized):
    rows = 5
    a, b = bfp16_rows(rows, n1, normalized), bfp16_rows(rows, n2, normalized)
    want = gm.concat_unaligned_bfp16(a, b, rows, n1, n2)
    got = callx("npu_concat_unaligned_bfp16", a, b, Out(), I64(rows), I64(n1), I64(n2),
                outs=[want.size])[0]
    same(got, want)


@pytest.mark.parametrize("n1,n2", [(20, 981), (63, 1), (100, 200), (129, 255), (255, 745)])
def test_concat_unaligned_bfp16_group_spans(n1, n2):
    """Block exponents up to 30 below the largest of their 64-value group, a tiny group and
    an exponent-255 block in the decoded part."""
    rows = 3

    def blocks(n):
        nb = -(-n // 8)
        e = (140 - RNG.integers(0, 31, (rows, nb))).astype(np.uint8)
        e[:, -1] = RNG.choice(np.array([0, 3, 141, 255], np.uint8), rows)
        m = RNG.integers(-128, 128, (rows, nb, 8)).astype(np.int8)
        return np.concatenate([e[..., None], m.view(np.uint8)], axis=2).reshape(-1)

    a, b = blocks(n1), blocks(n2)
    want = gm.concat_unaligned_bfp16(a, b, rows, n1, n2)
    got = callx("npu_concat_unaligned_bfp16", a, b, Out(), I64(rows), I64(n1), I64(n2),
                outs=[want.size])[0]
    same(got, want)


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("c,start,end,stride", [(1000, 456, 1000, 1), (729, 2, 621, 2),
                                                (64, 10, 10, 1), (33, 0, 33, 2), (17, 1, 16, 2)])
def test_generic_slice_innermost(t, c, start, end, stride):
    x = rand((23, c), t)
    want = gm.generic_slice_innermost(x, start, end, stride)
    same(run("npu_generic_slice_innermost", x, t, Out(), I64(23), I64(c), I64(start), I64(end),
             I64(stride), out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_I16])
def test_generic_slice_16b_inner_range(t):
    x = rand((31, 512), t)
    want = gm.generic_slice_16b_inner_range(x, 16, 193)
    same(run("npu_generic_slice_16b_inner_range", x, t, Out(), I64(31), I64(512), I64(16),
             I64(193), out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_U8, NPU_I16, NPU_U16])
@pytest.mark.parametrize("c,split", [(64, 16), (100, 63), (64, 64), (5, 0)])
def test_generic_slice_16b_inner(t, c, split):
    x = rand((19, c), t)
    w0, w1 = gm.generic_slice_16b_inner(x, split)
    o0, o1 = two("npu_generic_slice_16b_inner", x, t, Out(0), Out(1), I64(19), I64(c),
                 I64(split), w0=w0, w1=w1)
    same(o0, w0)
    same(o1, w1)


@pytest.mark.parametrize("t", [NPU_I16, NPU_U16, NPU_BF16, NPU_F16])
def test_generic_slice_16b_c6463(t):
    x = rand((21, 64), t)
    w0, w1 = gm.generic_slice_16b_c6463(x)
    o0, o1 = two("npu_generic_slice_16b_c6463", x, t, Out(0), Out(1), I64(21), w0=w0, w1=w1)
    same(o0, w0)
    same(o1, w1)


@pytest.mark.parametrize("name", ["generic_mem_copy", "generic_reshape_innermost",
                                  "generic_pad_innermost", "generic_unpad_innermost"])
@pytest.mark.parametrize("t", TYPES)
def test_copy(name, t):
    x = rand(1001, t)
    want = getattr(gm, name)(x)
    same(run(f"npu_{name}", x, t, Out(), I64(x.size), out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_BF16, NPU_F16])
@pytest.mark.parametrize("cin,cout", [(1, 256), (127, 635), (3, 3), (2, 8)])
def test_generic_expand_innermost(t, cin, cout):
    x = rand((11, cin), t)
    want = gm.generic_expand_innermost(x, cout)
    same(run("npu_generic_expand_innermost", x, t, Out(), I64(11), I64(cin), I64(cout),
             out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_BF16, NPU_U16])
def test_generic_inplace_pad2d_c64(t):
    x = rand((7, 9, 64), t)
    pv = int(RNG.integers(-(1 << 31), 1 << 31))
    want = gm.generic_inplace_pad2d_c64(x, 1, 3, 2, 1, pv)
    same(run("npu_generic_inplace_pad2d_c64", x, t, Out(), I64(7), I64(9), I64(64), I64(1),
             I64(3), I64(2), I64(1), pv, out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_BF16, NPU_U16])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("pads", [(0, 1, 3, 0, 4, 5), (2, 2, 1, 1, 1, 0), (3, 3, 0, 2, 0, 0),
                                  (5, 6, 0, 0, 9, 4)])
def test_generic_inplace_pad3d(t, mode, pads):
    x = rand((4, 5, 6), t)
    pv = int(RNG.integers(-(1 << 31), 1 << 31))
    want = gm.generic_inplace_pad3d(x, pads, mode, pv)
    same(run("npu_generic_inplace_pad3d", x, t, Out(), I64(4), I64(5), I64(6),
             Host(np.array(pads, np.int64)), mode, pv, out_like=want), want)


PERMS4 = ["0123", "0132", "0213", "0231", "0312", "0321", "1023", "1302", "2013", "3210"]


@pytest.mark.parametrize("name", ["generic_transpose_unaligned", "generic_transpose4d_aligned"])
@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("perm", PERMS4)
def test_transpose4d(name, t, perm):
    shape = (3, 5, 7, 4)
    p = [int(c) for c in perm]
    x = rand(shape, t)
    want = getattr(gm, name)(x, p)
    same(run(f"npu_{name}", x, t, Out(), Host(np.array(shape, np.int64)),
             Host(np.array(p, np.int32)), out_like=want), want)


@pytest.mark.parametrize("t", [NPU_BF16, NPU_U16])
def test_generic_transpose4d_0132_RCR32(t):
    x = rand((3, 16, 24), t)
    want = gm.generic_transpose4d_0132_RCR32(x)
    same(run("npu_generic_transpose4d_0132_RCR32", x, t, Out(), I64(3), I64(16), I64(24),
             out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
def test_generic_transpose_k8n8(t):
    x = rand((40, 24), t)
    want = gm.generic_transpose_k8n8(x)
    same(run("npu_generic_transpose_k8n8", x, t, Out(), I64(40), I64(24), out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_U8, NPU_I16, NPU_U16])
@pytest.mark.parametrize("perm", [(0, 2, 1), (2, 0, 1), (1, 2, 0), (2, 1, 0)])
def test_transpose3d_mmac(t, perm):
    x = rand((4, 6, 10), t)
    want = gm.transpose3d_mmac(x, perm)
    same(run("npu_transpose3d_mmac", x, t, Out(), Host(np.array(x.shape, np.int64)),
             Host(np.array(perm, np.int32)), out_like=want), want)


@pytest.mark.parametrize("name,t", [("transpose_mmac_a8_n64k64", NPU_I8),
                                    ("transpose_mmac_a16_n32k64", NPU_I16)])
def test_transpose_mmac(name, t):
    x = rand((128, 96), t)
    want = getattr(gm, name)(x)
    same(run(f"npu_{name}", x, Out(), I64(128), I64(96), out_like=want), want)


@pytest.mark.parametrize("pad", [0, 128, 255])
def test_transpose_planar3_c8(pad):
    x = rand((3, 3, 192), NPU_U8)
    want = gm.transpose_planar3_c8(x, pad)
    same(run("npu_transpose_planar3_c8", x, Out(), I64(3), I64(192), pad, out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("b", [2, 4])
@pytest.mark.parametrize("mode", [0, 1])
def test_generic_depth_to_space(t, b, mode):
    x = rand((2, 3 * b * b, 5, 3), t)
    want = gm.generic_depth_to_space(x, b, mode)
    same(run("npu_generic_depth_to_space", x, t, Out(), *[I64(d) for d in x.shape], I64(b), mode,
             out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("b", [2, 4])
@pytest.mark.parametrize("mode", [0, 1])
def test_generic_depth_to_space_wc64(t, b, mode):
    x = rand((2, 5, 3, 3 * b * b), t)
    want = gm.generic_depth_to_space_wc64(x, b, mode)
    same(run("npu_generic_depth_to_space_wc64", x, t, Out(), *[I64(d) for d in x.shape], I64(b),
             mode, out_like=want), want)


@pytest.mark.parametrize("name,b", [("generic_space_to_depth", 2), ("generic_space_to_depth", 4),
                                    ("generic_space_to_depth_c4bs4", 4)])
@pytest.mark.parametrize("t", TYPES)
def test_space_to_depth(name, t, b):
    x = rand((3 * b, 5 * b, 7), t)
    want = getattr(gm, name)(x, b)
    same(run(f"npu_{name}", x, t, Out(), *[I64(d) for d in x.shape], I64(b), out_like=want),
         want)


IDX_T = {NPU_U16: np.uint16, NPU_I16: np.int16, NPU_U32: np.uint32, NPU_I32: np.int32}


def indices(shape, hi, it):
    """In-range indices plus out-of-range and negative ones (for signed types)."""
    x = RNG.integers(0, hi, size=shape).astype(np.int64)
    flat = x.reshape(-1)
    flat[::7] = hi + RNG.integers(0, 5, size=flat[::7].shape)
    if it in (NPU_I16, NPU_I32):
        flat[3::11] = -RNG.integers(1, 4, size=flat[3::11].shape)
    return x.astype(IDX_T[it])


@pytest.mark.parametrize("t", [NPU_U8, NPU_F16, NPU_FP8])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_generic_gather_3din_1dindices(t, axis):
    x = rand((5, 6, 7), t)
    idx = indices(9, x.shape[axis], NPU_U32)
    want = gm.generic_gather_3din_1dindices(x, idx, axis, fill_of(x.dtype))
    same(run("npu_generic_gather_3din_1dindices", x, t, idx, Out(), I64(5), I64(6), I64(7), I64(9),
             axis, out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_BF16, NPU_U16])
@pytest.mark.parametrize("it", [NPU_I16, NPU_I32, NPU_U16])
def test_generic_gather_element_axis0(t, it):
    x = rand((13, 17), t)
    idx = indices((21, 17), 13, it)
    want = gm.generic_gather_element_2din_2dindices_axis0(x, idx, fill_of(x.dtype))
    same(run("npu_generic_gather_element_2din_2dindices_axis0", x, t, idx, it, Out(), I64(13),
             I64(17), I64(21), out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_BF16, NPU_U16])
@pytest.mark.parametrize("it", [NPU_I16, NPU_I32, NPU_U16])
def test_generic_gather_element_innermost(t, it):
    x = rand((13, 17), t)
    idx = indices((13, 30), 17, it)
    want = gm.generic_gather_element_2din_2dindices_innermost(x, idx, fill_of(x.dtype))
    same(run("npu_generic_gather_element_2din_2dindices_innermost", x, t, idx, it, Out(), I64(13),
             I64(17), I64(30), out_like=want), want)


@pytest.mark.parametrize("t", [NPU_BF16, NPU_F16, NPU_FP8])
@pytest.mark.parametrize("coord", ["half_pixel", "asymmetric", "align_corners"])
@pytest.mark.parametrize("nearest", ["floor", "round_prefer_ceil"])
def test_resize_nearest(t, coord, nearest):
    Hi, Wi, C, Ho, Wo = 7, 9, 16, 15, 4
    x = rand((Hi, Wi, C), t)
    ih = gm.onnx_nearest_index(Ho, Hi, coord, nearest)
    iw = gm.onnx_nearest_index(Wo, Wi, coord, nearest)
    want = gm.resize_nearest_bf16(x, ih, iw)
    same(run("npu_resize_nearest_bf16", x, t, Out(), I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo),
             ih, iw, out_like=want), want)


def test_resize_nearest_bfp16():
    Hi, Wi, C, Ho, Wo = 5, 6, 128, 10, 12
    x = RNG.integers(0, 256, (Hi, Wi, C // 8 * 9), dtype=np.uint8)
    ih = gm.onnx_nearest_index(Ho, Hi, "asymmetric", "floor")
    iw = gm.onnx_nearest_index(Wo, Wi, "half_pixel", "round_prefer_floor")
    want = gm.resize_nearest_bfp16(x, ih, iw)
    same(run("npu_resize_nearest_bfp16", x, Out(), I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo),
             ih, iw, out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_U8])
def test_elementwise_not(t):
    x = np.arange(256, dtype=np.uint16).astype(np.uint8)
    x = np.concatenate([x, np.zeros(100, np.uint8), RNG.integers(0, 2, 1000, dtype=np.uint8)])
    want = gm.elementwise_not(x)
    same(run("npu_elementwise_not", x, t, Out(), I64(x.size), out_like=want), want)


def test_bad_args():
    """Every entry rejects (-1) an argument set its kernel cannot run."""
    n = None
    one = np.zeros(64, np.uint8)
    sh4 = Host(np.array([2, 2, 2, 2], np.int64))
    p4 = Host(np.array([0, 1, 3, 2], np.int32))
    sh3 = Host(np.array([2, 2, 2], np.int64))
    w2 = Host(np.array([1, 1], np.int64))
    # null pointers
    assert raw_rc("npu_generic_mem_copy", n, NPU_I8, n, I64(1)) == -1
    assert raw_rc("npu_generic_transpose_unaligned", one, NPU_I8, one, n, p4) == -1
    assert raw_rc("npu_generic_transpose4d_aligned", one, NPU_I8, one, sh4, n) == -1
    assert raw_rc("npu_generic_concat_16b_multi", n, NPU_I8, one, I64(1), 2, w2) == -1
    assert raw_rc("npu_generic_concat_multi_c_fusion", DevList([one, one]), NPU_I8, one, I64(1),
                  2, n) == -1
    assert raw_rc("npu_generic_inplace_pad3d", one, NPU_I8, one, I64(1), I64(1), I64(1), n, 0,
                  0) == -1
    # types no wrapper has
    for name in ("generic_mem_copy", "generic_reshape_innermost", "generic_pad_innermost",
                 "generic_unpad_innermost"):
        for t in (NPU_F32, NPU_I32, NPU_I64, NPU_BFP16):
            assert rejected(f"npu_{name}", one, t, Out(), I64(4), nbytes=64)
    nb = 64
    assert rejected("npu_generic_concat", one, NPU_F32, one, Out(), I64(1), I64(2), I64(0),
                    I64(2), I64(2), I64(0), I64(2), nbytes=nb)
    assert rejected("npu_generic_concat", one, NPU_BFP16, one, Out(), I64(1), I64(16), I64(0),
                    I64(12), I64(8), I64(0), I64(8), nbytes=nb)          # not whole blocks
    assert rejected("npu_generic_concat", one, NPU_BFP16, one, Out(), I64(1), I64(16), I64(8),
                    I64(8), I64(8), I64(0), I64(8), nbytes=nb)           # start != 0
    assert rejected("npu_generic_concat", one, NPU_I8, one, Out(), I64(1), I64(4), I64(2),
                    I64(3), I64(4), I64(0), I64(1), nbytes=nb)           # past the row
    assert rejected("npu_generic_concat_16b_inner", one, NPU_I32, one, Out(), I64(1), I64(1),
                    I64(1), nbytes=nb)
    assert rejected("npu_generic_concat_16b_c6463", one, NPU_F32, one, Out(), I64(1), nbytes=nb)
    w7 = Host(np.array([1] * 7, np.int64))
    assert rejected("npu_generic_concat_16b_multi", DevList([one] * 7), NPU_I8, Out(), I64(1), 7,
                    w7, nbytes=nb)
    assert rejected("npu_generic_concat_multi_c_fusion", DevList([one]), NPU_I8, Out(), I64(1), 1,
                    w7, nbytes=nb)
    assert rejected("npu_concat_unaligned_bfp16", one, one, Out(), I64(1), I64(0), I64(8),
                    nbytes=nb)
    assert rejected("npu_generic_slice_innermost", one, NPU_I8, Out(), I64(1), I64(9), I64(0),
                    I64(9), I64(3), nbytes=nb)                           # stride 3
    assert rejected("npu_generic_slice_16b_inner_range", one, NPU_U8, Out(), I64(1), I64(9),
                    I64(0), I64(4), nbytes=nb)
    assert raw_rc("npu_generic_slice_16b_inner", one, NPU_BF16, one, one, I64(1), I64(8),
                  I64(4)) == -1
    assert raw_rc("npu_generic_slice_16b_inner", one, NPU_I8, one, one, I64(1), I64(8),
                  I64(9)) == -1
    assert raw_rc("npu_generic_slice_16b_c6463", one, NPU_I8, one, one, I64(1)) == -1
    assert rejected("npu_generic_expand_innermost", one, NPU_U8, Out(), I64(1), I64(2), I64(4),
                    nbytes=nb)
    assert rejected("npu_generic_expand_innermost", one, NPU_I8, Out(), I64(1), I64(3), I64(4),
                    nbytes=nb)
    assert rejected("npu_generic_inplace_pad2d_c64", one, NPU_FP8, Out(), I64(1), I64(1), I64(1),
                    I64(0), I64(0), I64(0), I64(0), 0, nbytes=nb)
    pads = Host(np.array([0, 0, 0, 0, 3, 0], np.int64))                 # C: 1 + 3 <= 2 * 3
    assert rejected("npu_generic_inplace_pad3d", one, NPU_I8, Out(), I64(1), I64(1), I64(1),
                    pads, 1, 0, nbytes=nb)
    assert rejected("npu_generic_inplace_pad3d", one, NPU_I8, Out(), I64(1), I64(1), I64(1),
                    Host(np.zeros(6, np.int64)), 3, 0, nbytes=nb)
    assert rejected("npu_generic_transpose_unaligned", one, NPU_I8, Out(), sh4,
                    Host(np.array([0, 1, 1, 3], np.int32)), nbytes=nb)
    assert rejected("npu_generic_transpose4d_aligned", one, NPU_U32, Out(), sh4, p4, nbytes=nb)
    assert rejected("npu_generic_transpose4d_0132_RCR32", one, NPU_I16, Out(), I64(1), I64(2),
                    I64(2), nbytes=nb)
    assert rejected("npu_generic_transpose_k8n8", one, NPU_F32, Out(), I64(2), I64(2), nbytes=nb)
    assert rejected("npu_transpose3d_mmac", one, NPU_I8, Out(), sh3,
                    Host(np.array([1, 0, 2], np.int32)), nbytes=nb)       # keeps the inner axis
    assert rejected("npu_transpose3d_mmac", one, NPU_FP8, Out(), sh3,
                    Host(np.array([0, 2, 1], np.int32)), nbytes=nb)
    assert rejected("npu_transpose_mmac_a8_n64k64", one, Out(), I64(-1), I64(2), nbytes=nb)
    assert rejected("npu_transpose_mmac_a16_n32k64", one, Out(), I64(2), I64(-2), nbytes=nb)
    assert rejected("npu_transpose_planar3_c8", one, Out(), I64(-1), I64(2), 0, nbytes=nb)
    assert rejected("npu_generic_depth_to_space", one, NPU_I8, Out(), I64(1), I64(3), I64(1),
                    I64(1), I64(2), 0, nbytes=nb)
    assert rejected("npu_generic_depth_to_space_wc64", one, NPU_I8, Out(), I64(1), I64(1),
                    I64(1), I64(9), I64(3), 0, nbytes=nb)                # b 3
    assert rejected("npu_generic_space_to_depth", one, NPU_I8, Out(), I64(3), I64(2), I64(1),
                    I64(2), nbytes=nb)
    assert rejected("npu_generic_space_to_depth_c4bs4", one, NPU_I8, Out(), I64(2), I64(2),
                    I64(4), I64(2), nbytes=nb)                           # b 2
    # two negative dims would give positive work with negative index math
    assert rejected("npu_generic_gather_3din_1dindices", one, NPU_I8, one, Out(), I64(-2), I64(2),
                    I64(-3), I64(2), 1, nbytes=nb)
    assert rejected("npu_generic_gather_3din_1dindices", one, NPU_I8, one, Out(), I64(2), I64(2),
                    I64(2), I64(2), 3, nbytes=nb)
    for name in ("axis0", "innermost"):
        e = f"npu_generic_gather_element_2din_2dindices_{name}"
        assert rejected(e, one, NPU_I8, one, NPU_U32, Out(), I64(2), I64(2), I64(2), nbytes=nb)
        assert rejected(e, one, NPU_FP8, one, NPU_I32, Out(), I64(2), I64(2), I64(2), nbytes=nb)
    tab = np.zeros(4, np.uint16)
    assert rejected("npu_resize_nearest_bf16", one, NPU_I8, Out(), I64(1), I64(1), I64(1), I64(1),
                    I64(1), tab, tab, nbytes=nb)
    assert rejected("npu_resize_nearest_bf16", one, NPU_BF16, Out(), I64(1), I64(1), I64(1),
                    I64(-1), I64(-1), tab, tab, nbytes=nb)
    assert raw_rc("npu_resize_nearest_bf16", one, NPU_BF16, one, I64(1), I64(1), I64(1), I64(1),
                  I64(1), n, tab) == -1
    assert rejected("npu_resize_nearest_bfp16", one, Out(), I64(1), I64(1), I64(12), I64(1),
                    I64(1), tab, tab, nbytes=nb)
    assert rejected("npu_elementwise_not", one, NPU_I16, Out(), I64(4), nbytes=nb)
    assert rejected("npu_elementwise_not", one, NPU_I8, Out(), I64(-1), nbytes=nb)
    # negative sizes
    assert raw_rc("npu_generic_mem_copy", one, NPU_I8, one, I64(-1)) == -1
    assert raw_rc("npu_generic_concat_16b_inner", one, NPU_I8, one, one, I64(-1), I64(1),
                  I64(1)) == -1
    assert raw_rc("npu_generic_slice_innermost", one, NPU_I8, one, I64(1), I64(-2), I64(0),
                  I64(-3), I64(1)) == -1                                 # c < 0
    assert raw_rc("npu_generic_inplace_pad3d", one, NPU_I8, one, I64(-1), I64(1), I64(1),
                  Host(np.zeros(6, np.int64)), 0, 0) == -1
    # a required pointer is rejected even when the work is empty
    z = I64(0)
    assert raw_rc("npu_generic_mem_copy", n, NPU_I8, one, z) == -1
    assert raw_rc("npu_generic_mem_copy", one, NPU_I8, n, z) == -1
    assert raw_rc("npu_generic_concat_16b_inner", one, NPU_I8, n, one, z, I64(1), I64(1)) == -1
    assert raw_rc("npu_resize_nearest_bf16", one, NPU_BF16, one, I64(1), I64(1), I64(1), z, z,
                  tab, n) == -1
    assert raw_rc("npu_elementwise_not", n, NPU_I8, one, z) == -1
    # an empty axis cannot be padded or resized from
    assert raw_rc("npu_generic_inplace_pad2d_c64", one, NPU_I8, one, z, I64(1), I64(1), I64(1),
                  z, z, z, 0) == -1
    assert raw_rc("npu_resize_nearest_bf16", one, NPU_BF16, one, z, I64(1), I64(1), I64(1),
                  I64(1), tab, tab) == -1


def test_empty_work():
    """A call with an empty output returns 0 without touching memory."""
    one = np.zeros(64, np.uint8)
    z, i1 = I64(0), I64(1)
    tab = np.zeros(4, np.uint16)
    zero6 = Host(np.zeros(6, np.int64))
    sh4 = Host(np.array([2, 0, 2, 2], np.int64))
    p4 = Host(np.array([0, 1, 3, 2], np.int32))
    calls = [
        ("npu_generic_concat", one, NPU_I8, one, one, z, I64(4), z, I64(4), I64(4), z, I64(4)),
        ("npu_generic_concat", one, NPU_BFP16, one, one, z, I64(8), z, I64(8), I64(8), z, I64(8)),
        ("npu_generic_concat_16b_inner", one, NPU_I16, one, one, z, I64(2), I64(3)),
        ("npu_generic_concat_16b_c6463", one, NPU_I16, one, one, z),
        ("npu_generic_concat_16b_multi", DevList([one, one]), NPU_I8, one, z, 2,
         Host(np.array([1, 2], np.int64))),
        ("npu_generic_concat_multi_c_fusion", DevList([one, one]), NPU_I8, one, z, 2,
         Host(np.array([1, 2], np.int64))),
        ("npu_concat_unaligned_bfp16", one, one, one, z, I64(5), I64(7)),
        ("npu_generic_slice_innermost", one, NPU_I8, one, z, I64(9), z, I64(9), I64(2)),
        ("npu_generic_slice_16b_inner_range", one, NPU_I16, one, z, I64(9), z, I64(4)),
        ("npu_generic_slice_16b_inner", one, NPU_I8, one, one, z, I64(8), I64(3)),
        ("npu_generic_slice_16b_c6463", one, NPU_BF16, one, one, z),
        ("npu_generic_mem_copy", one, NPU_I8, one, z),
        ("npu_generic_reshape_innermost", one, NPU_BF16, one, z),
        ("npu_generic_pad_innermost", one, NPU_FP8, one, z),
        ("npu_generic_unpad_innermost", one, NPU_U16, one, z),
        ("npu_generic_expand_innermost", one, NPU_I8, one, z, I64(2), I64(4)),
        ("npu_generic_inplace_pad2d_c64", one, NPU_I8, one, i1, z, I64(64), i1, i1, z, z, 0),
        ("npu_generic_inplace_pad3d", one, NPU_I8, one, i1, i1, z, zero6, 1, 0),
        ("npu_generic_transpose_unaligned", one, NPU_I8, one, sh4, p4),
        ("npu_generic_transpose4d_aligned", one, NPU_BF16, one, sh4, p4),
        ("npu_generic_transpose4d_0132_RCR32", one, NPU_BF16, one, z, I64(2), I64(2)),
        ("npu_generic_transpose_k8n8", one, NPU_I8, one, I64(8), z),
        ("npu_transpose3d_mmac", one, NPU_I8, one, Host(np.array([0, 2, 2], np.int64)),
         Host(np.array([0, 2, 1], np.int32))),
        ("npu_transpose_mmac_a8_n64k64", one, one, z, I64(64)),
        ("npu_transpose_mmac_a16_n32k64", one, one, I64(64), z),
        ("npu_transpose_planar3_c8", one, one, z, I64(4), 0),
        ("npu_generic_depth_to_space", one, NPU_I8, one, z, I64(4), i1, i1, I64(2), 0),
        ("npu_generic_depth_to_space_wc64", one, NPU_I8, one, z, i1, i1, I64(4), I64(2), 1),
        ("npu_generic_space_to_depth", one, NPU_I8, one, z, I64(2), i1, I64(2)),
        ("npu_generic_space_to_depth_c4bs4", one, NPU_I8, one, I64(4), z, i1, I64(4)),
        ("npu_generic_gather_3din_1dindices", one, NPU_I8, one, one, I64(2), I64(2), I64(2), z,
         1),
        ("npu_generic_gather_element_2din_2dindices_axis0", one, NPU_I8, one, NPU_I16, one,
         I64(2), I64(2), z),
        ("npu_generic_gather_element_2din_2dindices_innermost", one, NPU_I8, one, NPU_U16, one,
         z, I64(2), I64(2)),
        ("npu_resize_nearest_bf16", one, NPU_BF16, one, I64(2), I64(2), I64(4), z, I64(3), tab,
         tab),
        ("npu_resize_nearest_bf16", one, NPU_F16, one, z, z, I64(4), z, z, tab, tab),
        ("npu_resize_nearest_bfp16", one, one, I64(2), I64(2), I64(8), I64(3), z, tab, tab),
        ("npu_elementwise_not", one, NPU_U8, one, z),
    ]
    for args in calls:
        assert raw_rc(*args) == 0, args[0]
