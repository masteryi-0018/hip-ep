# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The matmul-group entry points (include/npu_emu_matmul.h) against mllib
nightly aiesim outputs, byte for byte. Inputs are decoded from the NPU's
physical layouts (gemm: C64-blocked matrices, byte-split weights; conv: HWC
activations and the packed weight stream) to logical tensors; the
coefficients are the words the kernel reads (coeff32.txt for gemm, the qdq /
clip / slope blocks of the packed wts32.txt for conv)."""

import json
import os

import numpy as np
import pytest

from aiesim_common import assert_bytes, cases, data, ty, words
from npu import I64, OUT, call

GEMM_DIRS = ["gemm_qdq_a16a16_wrapper"]
CONV_DIRS = ["conv_qdq_int8x8_wrapper", "conv_qdq_int8x8_channelwise_wrapper",
             "conv_qdq_int16x8_wrapper", "conv_qdq_int16x8_channelwise_wrapper"]
ACT_NONE, ACT_CLIP, ACT_LRELU, ACT_PRELU = 0, 1, 2, 3


def act_kind(t):
    """mllib activation_type -> npu_activation."""
    a = t.get("activation_type", 0)
    return {0: ACT_NONE, 1: ACT_CLIP, 2: ACT_CLIP, 8: ACT_CLIP, 100: ACT_CLIP, 3: ACT_LRELU,
            9: ACT_PRELU}[a]


def k_opt(t):
    """The kernel instance sums the activations on its k_opt path."""
    return int(bool(t.get("is_kh") and t.get("is_kw") and t["has_actv_sum"]))


def run(name, *args, n, t):
    dt = ty(t)[1]
    return call(name, *args, out_nbytes=n * np.dtype(dt).itemsize, out_dtype=dt)


# ---- gemm ----------------------------------------------------------------------

def gemm_inputs(case, t):
    m = t["sv_m_dim"] * t["m_iteration"]
    k = t["sv_k_dim"] * t["k_iteration"]
    n = t["sv_n_dim"] * t["n_iteration"]
    adt, bdt, odt = ty(t["ifm_dtype"])[1], ty(t["wts_dtype"])[1], ty(t["ofm_dtype"])[1]
    # A: [K/64][M][64]
    a = data(case, "ifm32.txt", adt)[:m * k].reshape(k // 64, m, 64).transpose(1, 0, 2)
    a = np.ascontiguousarray(a).reshape(m, k)
    # B^T: [K/64][N][64]; split: each row of 64 is 64 low bytes then 64 high bytes
    raw = data(case, "wts32.txt")[:n * k * 2].reshape(k // 64, n, 128)
    if t.get("is_split", 0):
        raw = np.stack([raw[..., :64], raw[..., 64:]], axis=-1).reshape(k // 64, n, 128)
    bt = raw.copy().view(bdt).reshape(k // 64, n, 64).transpose(1, 0, 2).reshape(n, k)
    b = np.ascontiguousarray(bt.T)
    want = data(case, "ofm32.txt", odt)[:m * n].reshape(n // 64, m, 64).transpose(1, 0, 2)
    return a, b, np.ascontiguousarray(want).reshape(m, n)


@pytest.mark.parametrize("d,case", cases(GEMM_DIRS))
def test_gemm(d, case):
    t = json.load(open(os.path.join(case, "testcase.json")))
    a, b, want = gemm_inputs(case, t)
    w = words(case).view(np.float32)
    act = act_kind(t)
    act0, act1 = (float(w[4]), float(w[5])) if act != ACT_NONE else (0.0, 0.0)
    m, k = a.shape
    n = b.shape[1]
    got = run("npu_gemm_qdq_int16x16", a, ty(t["ifm_dtype"])[0], b, ty(t["wts_dtype"])[0], OUT,
              ty(t["ofm_dtype"])[0], I64(m), I64(k), I64(n), float(w[0]), float(w[1]),
              float(w[2]), float(w[3]), act, act0, act1, n=m * n, t=t["ofm_dtype"])
    assert_bytes(got.reshape(m, n), want)


# ---- conv ----------------------------------------------------------------------

def _ceil_to(n, d):
    return d * -(-n // d)


def _pack_sv(t, sv):
    """mllib's transpose_wts_sv: one [Ksh][Kw][Ci_sv][Co_sv] weight sub-volume
    to the kernel's packed order (Kw folded for 64 // Co_sv output pixels)."""
    ksh, kw, ci_sv, co_sv = sv.shape
    folded_wo = 64 // co_sv
    folded_kw = kw + (folded_wo - 1) * t["stride_width"]
    fkw = _ceil_to(folded_kw, 64 // ci_sv) if ci_sv < 64 else folded_kw
    o = np.zeros((ksh, fkw, ci_sv, 64), sv.dtype)
    for c in range(folded_wo):
        sw = t["stride_width"]
        o[:, c * sw:c * sw + kw, :, co_sv * c:co_sv * (c + 1)] = sv
    if ci_sv < 64:
        return o.reshape(ksh, fkw // (64 // ci_sv), 64, 64).transpose(0, 1, 3, 2)
    return o.reshape(ksh, fkw, ci_sv // 64, 64, 64).transpose(2, 0, 1, 4, 3)


def conv_inputs(case, t):
    """Logical ifm / wts, the per-channel c0 / c1 / c2, the clip and slope
    words, and the expected output, from a conv case."""
    kh = t["kernel_sv_height"] * t["kernel_sv_height_iteration"]
    kw, sh, sw = t["kernel_width"], t["stride_height"], t["stride_width"]
    ci = t["ifm_sv_depth"] * t["ifm_depth_iteration"]
    co_sv = t["ofm_sv_depth"]
    co = co_sv * t["ofm_depth_iteration"]
    ho = t["ofm_sv_height"] * t["ofm_height_iteration"]
    wo = t["ofm_sv_width"] * t["ofm_width_iteration"]
    h, w = (ho - 1) * sh + kh, (wo - 1) * sw + kw
    idt, wdt, odt = (ty(t[k])[1] for k in ("ifm_dtype", "wts_dtype", "ofm_dtype"))
    ifm = data(case, "ifm32.txt", idt)[:h * w * ci].reshape(h, w, ci)
    # Pack an index tensor the way mllib packs the weights to find where each
    # logical weight sits in the stream; the qdq / clip / slope blocks follow
    # each output-depth iteration.
    ksh, ci_sv = t["kernel_sv_height"], t["ifm_sv_depth"]
    idx = np.arange(1, kh * kw * ci * co + 1, dtype=np.int64).reshape(kh, kw, ci, co)
    stream = data(case, "wts32.txt")
    rows = 3 if t["has_actv_sum"] else 2
    act = act_kind(t)
    wts = np.zeros(kh * kw * ci * co + 1, wdt)
    c = np.zeros((3, co), np.float32)
    clip = np.zeros(2, np.float32)
    slope = np.zeros(co, np.float32)
    pos = 0
    for o in range(co // co_sv):
        for kh_it in range(kh // ksh):
            for ci_it in range(ci // ci_sv):
                sv = idx[ksh * kh_it:ksh * (kh_it + 1), :, ci_sv * ci_it:ci_sv * (ci_it + 1),
                         co_sv * o:co_sv * (o + 1)]
                p = _pack_sv(t, sv).reshape(-1)
                seg = stream[pos:pos + p.size].view(wdt)
                wts[p[p > 0]] = seg[p > 0]
                pos += p.size
        q = stream[pos:pos + rows * 256].view(np.float32).reshape(rows, 64)[:, :co_sv]
        c[0, co_sv * o:co_sv * (o + 1)] = q[0]
        c[2, co_sv * o:co_sv * (o + 1)] = q[rows - 1]
        if rows == 3:
            c[1, co_sv * o:co_sv * (o + 1)] = q[1]
        pos += rows * 256
        if act != ACT_NONE:
            clip = stream[pos:pos + 8].view(np.float32).copy()
            pos += 64
        if act == ACT_PRELU:
            slope[co_sv * o:co_sv * (o + 1)] = stream[pos:pos + 256].view(np.float32)[:co_sv]
            pos += 256
    wts = wts[1:].reshape(kh, kw, ci, co)
    want = data(case, "ofm32.txt", odt)[:ho * wo * co].reshape(ho, wo, co)
    return ifm, wts, c, clip, slope, want


@pytest.mark.parametrize("d,case", cases(CONV_DIRS))
def test_conv(d, case):
    t = json.load(open(os.path.join(case, "testcase.json")))
    ifm, wts, c, clip, slope, want = conv_inputs(case, t)
    act = act_kind(t)
    name = "npu_conv_qdq_int16x8" if "int16" in d else "npu_conv_qdq_int8x8"
    h, w, ci = ifm.shape
    kh, kw, _, co = wts.shape
    got = run(name, ifm, ty(t["ifm_dtype"])[0], wts, ty(t["wts_dtype"])[0], OUT,
              ty(t["ofm_dtype"])[0], I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co),
              t["stride_height"], t["stride_width"], k_opt(t), c[0],
              c[1] if t["has_actv_sum"] else None,
              c[2], act, float(clip[0]), float(clip[1]),
              slope if act == ACT_PRELU else None, n=want.size, t=t["ofm_dtype"])
    assert_bytes(got.reshape(want.shape), want)
