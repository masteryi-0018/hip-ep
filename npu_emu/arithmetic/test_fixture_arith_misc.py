# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Quantized div, leaky relu, prelu, square and batchnorm against NPU reference outputs."""

import numpy as np
import pytest

from fixture import TYPES as INT
from fixture import cases, data, ofm, rtp_words, run_unary, words
from fixture import testcase as tc
from npu import I64, NPU_I8, NPU_I16, OUT
from testing import call_n as go
from testing import shp


def fl(w):
    return [float(v) for v in np.asarray(w, np.uint32).view(np.float32)]


def phys3(x, valid):
    return np.ascontiguousarray(x[:valid[0], :valid[1], :valid[2]])


def run_div(case, t, bits):
    it, idt = INT[t["ifm_dtype"]]
    ot, odt = INT[t["ofm_dtype"]]
    a = data(case, "ifm32.txt", idt)
    b = data(case, "wts32.txt", idt)
    r = rtp_words(case)
    c0, c1, zp1, zp2 = fl(r[2:6])   # after the two loop-count words
    sig = int(r[6])
    assert (sig & 1) == (it in (NPU_I8, NPU_I16)) and ((sig >> 8) & 1) == (ot in (NPU_I8, NPU_I16))
    got = go(f"npu_elementwise_div_a{bits}", a, it, b, OUT, ot, I64(a.size), c0, c1, zp1, zp2,
             n=a.size, dt=odt)
    return got, data(case, ofm(case), odt)[:a.size]


def run_bdiv(case, t, bits):
    it = [t["ifm1_height_iteration"], t["ifm1_width_iteration"], t["ifm1_channel_iteration"]]
    dims = ("height", "width", "channel")
    av = [t[f"ifm1_valid_full_{d}"] for d in dims]
    a_s = [t[f"ifm1_physical_sv_{d}"] for d in dims]
    bv = [t[f"ifm2_valid_full_{d}"] for d in dims]
    b_s = [t[f"ifm2_physical_sv_{d}"] for d in dims]
    ap = [s * i for s, i in zip(a_s, it)]
    bp = [s * i if v > 1 else s for s, i, v in zip(b_s, it, bv)]
    nt, dt = INT[t["dtype"]]
    a = phys3(data(case, "ifm32.txt", dt).reshape(ap), av)
    b = phys3(data(case, "wts32.txt", dt)[:int(np.prod(bp))].reshape(bp), bv)
    want = phys3(data(case, ofm(case), dt).reshape(ap), av)
    w = words(case)
    c0, c1 = fl(w[[0, 16]])
    assert np.all(w[16:32] == w[16]) and np.all(w[32:64] == w[32]) and np.all(w[96:128] == w[96])
    zp1, zp2 = (int(v) for v in w[[32, 96]].view(np.int32))
    got = go(f"npu_qdq_elementwise_broadcasting3d_div_a{bits}", a, nt, b, OUT, shp(av), shp(bv),
             c0, c1, zp1, zp2, int(t["output_sft"]), n=a.size, dt=dt)
    return got, want


def run_prelu(case, t, bits):
    it, idt = INT[t["ifm_dtype"]]
    ot, odt = INT[t["ofm_dtype"]]
    C = t["ifm_sv_channel"]
    x = data(case, "ifm32.txt", idt)
    w = words(case)
    pos, out_zp = fl(w[[0, 2]])
    in_zp = int(w[1:2].view(np.int32)[0])
    neg = np.ascontiguousarray(w[32:32 + C].view(np.float32))   # from byte 128
    rows = x.size // C
    got = go(f"npu_elementwise_prelu_a{bits}", x, it, OUT, ot, I64(rows), I64(C), pos, in_zp,
             out_zp, neg, n=x.size, dt=odt)
    return got, data(case, ofm(case), odt)[:x.size]


def run_batchnorm(case, t, bits):
    nt, dt = INT[t["dtype"]]
    C, cs, K = t["ifm_sv_depth"], t["c_stride"], t["wts_cb_offset"]
    H = t["ifm_sv_height"] * t["ifm_height_iteration"]
    W = t["ifm_sv_width"] * t["ifm_width_iteration"]
    N = t["ifm_sv_batch"]
    assert K == C * 32 // cs
    w = words(case, "wts32.txt")
    # c_stride-channel groups are repeated up to 32 lanes: keep one copy.
    cw = np.ascontiguousarray(w[:K].view(np.float32).reshape(C // cs, 32 // cs, cs)[:, 0, :])
    cb = np.ascontiguousarray(w[K:2 * K].view(np.float32).reshape(C // cs, 32 // cs, cs)[:, 0, :])
    zp, alpha = fl(w[2 * K:2 * K + 2])

    def logical(v):   # [H][C/cs][W][N][cs] -> [H][W][N][C]
        return np.ascontiguousarray(v[:H * C * W * N].reshape(H, C // cs, W, N, cs)
                                    .transpose(0, 2, 3, 1, 4).reshape(H * W * N, C))

    x = logical(data(case, "ifm32.txt", dt))
    got = go(f"npu_qdq_batchnorm2d_stream_a{bits}", x, nt, OUT, I64(H * W * N), I64(C),
             cw.reshape(-1), cb.reshape(-1), zp, int(t["activation_type"]), alpha,
             n=x.size, dt=dt)
    return got, logical(data(case, ofm(case), dt))


def runner(d):
    w = d.removesuffix("_wrapper")
    bits = 16 if w.endswith("a16") else 8
    if "broadcasting3d_div" in w:
        return lambda c, t: run_bdiv(c, t, bits)
    if "elementwise_div" in w:
        return lambda c, t: run_div(c, t, bits)
    for op in ("leakyrelu", "square"):
        if f"elementwise_{op}" in w:
            return lambda c, t: run_unary(c, t, op, bits, {"leakyrelu": 4, "square": 3}[op])
    if "elementwise_prelu" in w:
        return lambda c, t: run_prelu(c, t, bits)
    if "batchnorm2d_stream" in w:
        return lambda c, t: run_batchnorm(c, t, bits)
    raise KeyError(d)


DIRS = [f"qdq_batchnorm2d_stream_a{b}_wrapper" for b in (8, 16)] + \
       [f"qdq_elementwise_broadcasting3d_div_a{b}_wrapper" for b in (8, 16)] + \
       [f"qdq_elementwise_{op}_a{b}_wrapper" for op in ("div", "leakyrelu", "prelu", "square")
        for b in (8, 16)]

CASES = cases(DIRS)


@pytest.mark.skipif(not CASES, reason="no fixture cases")
@pytest.mark.parametrize("wrapper,case", CASES)
def test_fixture_arith_misc(wrapper, case):
    t = tc(case)
    got, want = runner(wrapper)(case, t)
    got = np.asarray(got).reshape(-1)
    want = np.ascontiguousarray(want).reshape(-1).astype(got.dtype)
    assert got.size == want.size, f"size {got.size} != {want.size}"
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} elements differ from sim, first at "
                           f"{bad[:8].tolist()}: got {got[bad[:8]].tolist()} "
                           f"want {want[bad[:8]].tolist()}")
