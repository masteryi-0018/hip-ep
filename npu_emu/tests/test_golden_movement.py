# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Movement-group entry points against the numpy reference, bit for bit.
Inputs are random bit patterns (so NaN payloads, -0 and fp8 NaN groups are
covered and must pass through unchanged); shapes include odd sizes, empty
slices and the bfp16 concat's phase boundaries."""

import numpy as np
import pytest

import golden as g
import golden_movement as gm
from npu import FILL, DevList, Host, I64, Out, U64, callx, raw_rc
from npu import (NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, NPU_I32,
                 NPU_I64, NPU_U8, NPU_U16, NPU_U32)

RNG = np.random.default_rng(7)
# npu_dtype -> unsigned storage of the same width (movement only copies bits)
WIDTH = {NPU_I8: np.uint8, NPU_U8: np.uint8, NPU_FP8: np.uint8, NPU_I16: np.uint16,
         NPU_U16: np.uint16, NPU_F16: np.uint16, NPU_BF16: np.uint16,
         NPU_F32: np.uint32, NPU_I32: np.uint32, NPU_U32: np.uint32, NPU_I64: np.uint64}
TYPES = [NPU_I8, NPU_BF16, NPU_F32, NPU_I64]


def rand(shape, t):
    dt = WIDTH[t]
    return RNG.integers(0, np.iinfo(dt).max, size=shape, dtype=dt, endpoint=True)


def run(name, *args, out_like):
    """Single output shaped like out_like (array or (shape, dtype))."""
    shape, dt = (out_like.shape, out_like.dtype) if isinstance(out_like, np.ndarray) else out_like
    nb = int(np.prod(shape)) * np.dtype(dt).itemsize
    return callx(name, *args, outs=[nb])[0].view(dt).reshape(shape)


def same(got, want):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    bad = np.nonzero(got.reshape(-1) != want.reshape(-1))[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} differ, first at {bad[:6].tolist()}: "
                           f"got {got.reshape(-1)[bad[:6]].tolist()} "
                           f"want {want.reshape(-1)[bad[:6]].tolist()}")


def fill_of(dt):
    return np.frombuffer(bytes([FILL]) * np.dtype(dt).itemsize, dt)[0]


# ---- A. concat / slice / copy -------------------------------------------------

@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("c1,s1,n1,c2,s2,n2", [(24, 0, 23, 48, 2, 40), (64, 0, 63, 4, 0, 1),
                                                (256, 1, 255, 748, 2, 745), (5, 5, 0, 3, 0, 3)])
def test_generic_concat(t, c1, s1, n1, c2, s2, n2):
    rows = 37
    a, b = rand((rows, c1), t), rand((rows, c2), t)
    want = gm.concat_rows([a[:, s1:s1 + n1], b[:, s2:s2 + n2]])
    same(run("npu_generic_concat", a, I64(c1), I64(s1), I64(n1), b, I64(c2), I64(s2), I64(n2),
             Out(), t, I64(rows), out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
def test_generic_concat_16b_inner(t):
    a, b = rand((50, 13), t), rand((50, 70), t)
    want = gm.concat_rows([a, b])
    same(run("npu_generic_concat_16b_inner", a, I64(13), b, I64(70), Out(), t, I64(50),
             out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
def test_generic_concat_16b_c6463(t):
    a, b = rand((40, 63), t), rand((40, 1), t)
    want = gm.concat_rows([a, b])
    same(run("npu_generic_concat_16b_c6463", a, b, Out(), t, I64(40), out_like=want), want)


@pytest.mark.parametrize("name", ["npu_generic_concat_16b_multi",
                                  "npu_generic_concat_multi_c_fusion"])
@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("widths", [[3, 5], [64, 1, 17], [1, 2, 3, 4, 5, 6], [0, 7, 9]])
def test_generic_concat_multi(name, t, widths):
    rows = 29
    parts = [rand((rows, w), t) for w in widths]
    want = gm.concat_rows(parts)
    same(run(name, DevList(parts), Host(np.array(widths, np.int64)), len(widths), Out(), t,
             I64(rows), out_like=want), want)


def test_concat_multi_rejects_bad_count():
    w = Host(np.array([1] * 7, np.int64))
    assert raw_rc("npu_generic_concat_16b_multi", None, w, 7, None, NPU_I8, I64(1)) == -1
    assert raw_rc("npu_generic_concat_multi_c_fusion", None, w, 1, None, NPU_I8, I64(1)) == -1


def bfp16_rows(rows, n, normalized):
    """rows of ceil(n/8) bfp16 blocks: random bytes (unnormalized mantissas,
    any exponent) or encoded random floats."""
    nb = -(-n // 8)
    if not normalized:
        return RNG.integers(0, 256, rows * nb * 9, dtype=np.uint8)
    x = (RNG.standard_normal(rows * nb * 8) * np.exp2(RNG.integers(-20, 20, rows * nb * 8)))
    return g.to_bfp16(x.astype(np.float32), g.EVEN)


@pytest.mark.parametrize("n1,n2", [(1, 1), (7, 9), (63, 1), (64, 64), (65, 3), (100, 200),
                                   (127, 1), (128, 128), (129, 255), (192, 64), (255, 745),
                                   (300, 7), (8, 8), (512, 512)])
@pytest.mark.parametrize("normalized", [False, True])
def test_concat_unaligned_bfp16(n1, n2, normalized):
    rows = 5
    a, b = bfp16_rows(rows, n1, normalized), bfp16_rows(rows, n2, normalized)
    want = gm.bfp16_concat(a, n1, b, n2, rows)
    got = callx("npu_concat_unaligned_bfp16", a, I64(n1), b, I64(n2), Out(), I64(rows),
                outs=[want.size])[0]
    same(got, want)


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("c,start,end,stride", [(1000, 456, 1000, 1), (729, 2, 621, 2),
                                                (64, 10, 10, 1), (33, 0, 33, 3), (17, 1, 16, 5)])
def test_generic_slice_innermost(t, c, start, end, stride):
    x = rand((23, c), t)
    want = gm.slice_inner(x, start, end, stride)
    same(run("npu_generic_slice_innermost", x, Out(), t, I64(23), I64(c), I64(start), I64(end),
             I64(stride), out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
def test_generic_slice_16b_inner_range(t):
    x = rand((31, 512), t)
    want = gm.slice_inner(x, 16, 193)
    same(run("npu_generic_slice_16b_inner_range", x, Out(), t, I64(31), I64(512), I64(16),
             I64(193), out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("c,split", [(64, 16), (100, 63), (64, 64), (5, 0)])
def test_generic_slice_16b_inner(t, c, split):
    x = rand((19, c), t)
    w0, w1 = x[:, :split], x[:, split:]
    es = x.dtype.itemsize
    o0, o1 = callx("npu_generic_slice_16b_inner", x, Out(0), Out(1), t, I64(19), I64(c),
                   I64(split), outs=[w0.size * es, w1.size * es])
    same(o0.view(x.dtype).reshape(w0.shape), w0)
    same(o1.view(x.dtype).reshape(w1.shape), w1)


@pytest.mark.parametrize("t", TYPES)
def test_generic_slice_16b_c6463(t):
    x = rand((21, 64), t)
    es = x.dtype.itemsize
    o0, o1 = callx("npu_generic_slice_16b_c6463", x, Out(0), Out(1), t, I64(21),
                   outs=[21 * 63 * es, 21 * es])
    same(o0.view(x.dtype).reshape(21, 63), x[:, :63])
    same(o1.view(x.dtype).reshape(21, 1), x[:, 63:])


@pytest.mark.parametrize("name", ["npu_generic_mem_copy", "npu_generic_reshape_innermost",
                                  "npu_generic_pad_innermost", "npu_generic_unpad_innermost"])
@pytest.mark.parametrize("t", TYPES)
def test_copy(name, t):
    x = rand(1001, t)
    same(run(name, x, Out(), t, I64(x.size), out_like=x), x)


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("cin,cout", [(1, 256), (127, 635), (3, 3), (2, 8)])
def test_generic_expand_innermost(t, cin, cout):
    x = rand((11, cin), t)
    want = gm.expand_inner(x, cout)
    same(run("npu_generic_expand_innermost", x, Out(), t, I64(11), I64(cin), I64(cout),
             out_like=want), want)


# ---- B. pad -------------------------------------------------------------------

@pytest.mark.parametrize("t", [NPU_I8, NPU_BF16, NPU_F32])
def test_generic_inplace_pad2d_c64(t):
    x = rand((7, 9, 64), t)
    bits = int(rand((), t))
    want = gm.pad3d(x, (1, 3, 2, 1, 0, 0), 0, bits)
    same(run("npu_generic_inplace_pad2d_c64", x, Out(), t, I64(7), I64(9), I64(64), I64(1),
             I64(3), I64(2), I64(1), U64(bits), out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_BF16, NPU_F32])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("pads", [(0, 1, 3, 0, 22, 5), (2, 2, 1, 1, 1, 0), (3, 3, 0, 2, 0, 0),
                                  (5, 6, 0, 0, 9, 4)])
def test_generic_inplace_pad3d(t, mode, pads):
    x = rand((4, 5, 6), t)
    bits = int(rand((), t))
    want = gm.pad3d(x, pads, mode, bits)
    same(run("npu_generic_inplace_pad3d", x, Out(), t, I64(4), I64(5), I64(6),
             Host(np.array(pads, np.int64)), mode, U64(bits), out_like=want), want)


# ---- C. transpose / space <-> depth ---------------------------------------------

PERMS4 = ["0123", "0132", "0213", "0231", "0312", "0321", "1023", "1302", "2013", "3210"]


@pytest.mark.parametrize("name", ["npu_generic_transpose_unaligned",
                                  "npu_generic_transpose4d_aligned"])
@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("perm", PERMS4)
def test_transpose4d(name, t, perm):
    shape = (3, 5, 7, 4)
    p = [int(c) for c in perm]
    x = rand(shape, t)
    want = np.transpose(x, p)
    same(run(name, x, Out(), t, Host(np.array(shape, np.int64)), Host(np.array(p, np.int32)),
             out_like=want), want)


def test_transpose_rejects_bad_perm():
    s = Host(np.array([2, 2, 2, 2], np.int64))
    assert raw_rc("npu_generic_transpose_unaligned", None, None, NPU_I8, s,
                  Host(np.array([0, 1, 1, 3], np.int32))) == -1


@pytest.mark.parametrize("t", TYPES)
def test_generic_transpose4d_0132_RCR32(t):
    x = rand((3, 16, 24), t)
    want = x.transpose(0, 2, 1)
    same(run("npu_generic_transpose4d_0132_RCR32", x, Out(), t, I64(3), I64(16), I64(24),
             out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
def test_generic_transpose_k8n8(t):
    x = rand((40, 24), t)
    same(run("npu_generic_transpose_k8n8", x, Out(), t, I64(40), I64(24), out_like=x.T), x.T)


@pytest.mark.parametrize("t", [NPU_I8, NPU_U8, NPU_I16, NPU_U16])
@pytest.mark.parametrize("perm", [(0, 2, 1), (2, 0, 1), (1, 2, 0), (2, 1, 0)])
def test_transpose3d_mmac(t, perm):
    x = rand((4, 6, 10), t)
    want = np.transpose(x, perm)
    same(run("npu_transpose3d_mmac", x, Out(), t, Host(np.array(x.shape, np.int64)),
             Host(np.array(perm, np.int32)), out_like=want), want)


@pytest.mark.parametrize("name,t", [("npu_transpose_mmac_a8_n64k64", NPU_I8),
                                    ("npu_transpose_mmac_a16_n32k64", NPU_I16)])
def test_transpose_mmac(name, t):
    x = rand((128, 96), t)
    same(run(name, x, Out(), I64(128), I64(96), out_like=x.T), x.T)


@pytest.mark.parametrize("pad", [0, 128, 255])
def test_transpose_planar3_c8(pad):
    x = rand((3, 3, 192), NPU_U8)
    want = gm.planar3_c8(x, pad)
    same(run("npu_transpose_planar3_c8", x, Out(), I64(3), I64(192), pad, out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("b", [2, 4])
@pytest.mark.parametrize("mode", [0, 1])
def test_generic_depth_to_space(t, b, mode):
    x = rand((2, 3 * b * b, 5, 3), t)
    want = gm.depth_to_space_nchw(x, b, mode)
    same(run("npu_generic_depth_to_space", x, Out(), t, *[I64(d) for d in x.shape], I64(b), mode,
             out_like=want), want)


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("b", [2, 4])
@pytest.mark.parametrize("mode", [0, 1])
def test_generic_depth_to_space_wc64(t, b, mode):
    x = rand((2, 5, 3, 3 * b * b), t)
    want = gm.depth_to_space_nhwc(x, b, mode)
    same(run("npu_generic_depth_to_space_wc64", x, Out(), t, *[I64(d) for d in x.shape], I64(b),
             mode, out_like=want), want)


@pytest.mark.parametrize("name", ["npu_generic_space_to_depth",
                                  "npu_generic_space_to_depth_c4bs4"])
@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("b", [2, 4])
def test_space_to_depth(name, t, b):
    x = rand((3 * b, 5 * b, 7), t)
    want = gm.space_to_depth_hwc(x, b)
    same(run(name, x, Out(), t, *[I64(d) for d in x.shape], I64(b), out_like=want), want)


# ---- D. gather / resize ----------------------------------------------------------

IDX_T = {NPU_U16: np.uint16, NPU_I16: np.int16, NPU_U32: np.uint32, NPU_I32: np.int32}


def indices(shape, hi, it):
    """In-range indices plus out-of-range and negative ones (for signed types)."""
    x = RNG.integers(0, hi, size=shape).astype(np.int64)
    flat = x.reshape(-1)
    flat[::7] = hi + RNG.integers(0, 5, size=flat[::7].shape)
    if it in (NPU_I16, NPU_I32):
        flat[3::11] = -RNG.integers(1, 4, size=flat[3::11].shape)
    return x.astype(IDX_T[it])


@pytest.mark.parametrize("t", [NPU_U8, NPU_F16, NPU_F32])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("it", [NPU_U32, NPU_I32, NPU_U16, NPU_I16])
def test_generic_gather_3din_1dindices(t, axis, it):
    x = rand((5, 6, 7), t)
    idx = indices(9, x.shape[axis], it)
    shape = list(x.shape)
    shape[axis] = 9
    want = gm.gather_axis(x, axis, idx, fill_of(x.dtype))
    same(run("npu_generic_gather_3din_1dindices", x, t, I64(5), I64(6), I64(7), axis, idx, it,
             I64(9), Out(), out_like=(tuple(shape), x.dtype)), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_BF16, NPU_F32])
@pytest.mark.parametrize("it", [NPU_I16, NPU_I32, NPU_U16])
def test_generic_gather_element_axis0(t, it):
    x = rand((13, 17), t)
    idx = indices((21, 17), 13, it)
    want = gm.gather_elements(x, idx, 0, fill_of(x.dtype))
    same(run("npu_generic_gather_element_2din_2dindices_axis0", x, t, I64(13), I64(17), idx, it,
             I64(21), Out(), out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_BF16, NPU_F32])
@pytest.mark.parametrize("it", [NPU_I16, NPU_I32, NPU_U16])
def test_generic_gather_element_innermost(t, it):
    x = rand((13, 17), t)
    idx = indices((13, 30), 17, it)
    want = gm.gather_elements(x, idx, 1, fill_of(x.dtype))
    same(run("npu_generic_gather_element_2din_2dindices_innermost", x, t, I64(13), I64(17), idx,
             it, I64(30), Out(), out_like=want), want)


@pytest.mark.parametrize("t", [NPU_I8, NPU_BF16])
@pytest.mark.parametrize("it", [NPU_U16, NPU_U32])
@pytest.mark.parametrize("shape,depth", [((1, 48, 48, 34), 3), ((6, 5, 4), 1), ((6, 5, 4), 2),
                                         ((3, 4, 5, 6), 4), ((2, 3, 4, 5, 6), 2)])
def test_gather_nd_4x3_adf(t, it, shape, depth):
    x = rand(shape, t)
    nt = 23
    idx = np.stack([RNG.integers(0, shape[j] + 1, nt) for j in range(depth)], axis=1)
    idx[::5, depth - 1] += shape[depth - 1] * 3  # past its axis, maybe still a valid row
    idx = idx.astype(IDX_T[it])
    row = int(np.prod(shape[depth:]))
    want = gm.gather_nd(x, idx, depth, fill_of(x.dtype))
    same(run("npu_gather_nd_4x3_adf", x, t, Host(np.array(shape, np.int64)), len(shape), idx, it,
             I64(nt), depth, Out(), out_like=((nt, row), x.dtype)), want)


@pytest.mark.parametrize("name,t", [("npu_resize_nearest_bf16", NPU_BF16),
                                    ("npu_resize_nearest_bf16", NPU_FP8),
                                    ("npu_resize_nearest_4x3", NPU_F16)])
@pytest.mark.parametrize("coord", ["half_pixel", "asymmetric", "align_corners"])
@pytest.mark.parametrize("nearest", ["floor", "round_prefer_ceil"])
def test_resize_nearest(name, t, coord, nearest):
    Hi, Wi, C, Ho, Wo = 7, 9, 16, 15, 4
    x = rand((Hi, Wi, C), t)
    ih = gm.onnx_nearest_index(Ho, Hi, coord, nearest)
    iw = gm.onnx_nearest_index(Wo, Wi, coord, nearest)
    want = gm.resize_hwc(x, ih, iw)
    same(run(name, x, Out(), t, I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo), ih, iw,
             out_like=want), want)


def test_resize_nearest_bfp16():
    Hi, Wi, C, Ho, Wo = 5, 6, 128, 10, 12
    x = RNG.integers(0, 256, (Hi, Wi, C // 8 * 9), dtype=np.uint8)
    ih = gm.onnx_nearest_index(Ho, Hi, "asymmetric", "floor")
    iw = gm.onnx_nearest_index(Wo, Wi, "half_pixel", "round_prefer_floor")
    want = gm.resize_hwc(x, ih, iw)
    same(run("npu_resize_nearest_bfp16", x, Out(), I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo),
             ih, iw, out_like=want), want)


# ---- E. not ------------------------------------------------------------------------

@pytest.mark.parametrize("it,ot", [(NPU_I8, NPU_I8), (NPU_U8, NPU_U8), (NPU_I8, NPU_U8)])
def test_elementwise_not(it, ot):
    x = np.arange(256, dtype=np.uint16).astype(np.uint8)
    x = np.concatenate([x, np.zeros(100, np.uint8), RNG.integers(0, 2, 1000, dtype=np.uint8)])
    want = np.logical_not(x).astype(np.uint8)
    same(run("npu_elementwise_not", x, it, Out(), ot, I64(x.size), out_like=want), want)
