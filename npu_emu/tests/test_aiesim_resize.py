# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The resize-group entry points (include/npu_emu_resize.h) against mllib
nightly aiesim outputs, byte for byte.

Tensors are decoded from the NPU's HCWC layout (h, C / gran blocks, w, gran
channels; gran = 32, or 64 for the integer kernels when the sub-volume depth is
above 32) to logical HWC. The weights are the words the kernel reads: the
per-sub-volume blobs of wts32.txt, packed by the YAML (index tables, then the
weight tables; the integer blobs start with the fp32 c0 / c1 words). The blob
indices are relative to each sub-volume's input window; the window origins are
recomputed here with the rules of the mllib packers and tiling
(ResizeBilinearA8Wrapper._first_tap_base / win_org, ResizeBilinearBf16Wrapper
rel_off, ResizeCubicBf16Wrapper sv_tile_origin), so the tables passed to the
entry points are absolute.

The multi-core tile-replay cases (resize_bilinear_multicore_a8) deliver one
padded input window per tile; the full image is reassembled from them."""

import json
import math
import os
import re

import numpy as np
import pytest

from aiesim_common import assert_bytes, cases, data, ty
from npu import I64, OUT, call

A8_DIRS = ["resize_a8_wrapper", "resize_bilinear_a16_wrapper"]
MC_DIRS = ["resize_bilinear_multicore_a8"]
BF16_DIRS = ["resize_bilinear_bf16_wrapper"]
FP8_DIRS = ["resize_bilinear_fp8_wrapper"]
CUBIC_DIRS = ["resize_cubic_bf16_wrapper"]
SEP_GATE = 0x52534550  # kResizeSeparableGateMagic: RTP word 17 selects separable
F32 = np.float32


def ceil_to(n, d):
    return d * -(-n // d)


def hcwc(case, name, dtype, H, W, C, gran):
    """[H][C/gran][W][gran] -> [H][W][C]."""
    a = data(case, name, dtype)[:H * W * C].reshape(H, C // gran, W, gran)
    return np.ascontiguousarray(a.transpose(0, 2, 1, 3).reshape(H, W, C))


def rtp_words(case):
    """generated_rtps[] of kernel_rtp.h as uint32, or None."""
    p = os.path.join(case, "kernel_rtp.h")
    if not os.path.exists(p):
        return None
    m = re.search(r"generated_rtps\[\d+\]\s*=\s*\{([^}]*)\}", open(p).read())
    return np.array([int(v, 16) for v in m.group(1).split(",") if v.strip()], np.uint32)


# ---- coordinate rules of the mllib packers (window origins only) ----------------

def a8_policy(t):
    """(grid_offset, scale_adjust) of ResizeBilinearA8Wrapper._coord_policy."""
    return (0.0, 1.0) if t.get("coordinate_transformation_mode") == "align_corners" else (0.5, 0.0)


def a8_source_coord(o, n_in, n_out, scale, pol):
    grid, adj = pol
    if adj:
        return 0.0 if n_out <= 1 else float(o) * (n_in - 1) / (n_out - 1)
    return (o + grid) / scale - grid


def a8_first_tap_base(off, scale, n_in, mode, pol):
    """ResizeBilinearA8Wrapper._first_tap_base: the blob's index origin."""
    v = a8_source_coord(off, n_in, round(n_in * scale), scale, pol)
    if mode == "reflect":
        if n_in > 1:
            period = 2.0 * (n_in - 1)
            vr = v - period * math.floor(v / period)
            if vr > n_in - 1:
                vr = period - vr
        else:
            vr = 0.0
        f = min(max(0, int(math.floor(vr))), n_in - 1)
        return n_in - 2 if f >= n_in - 1 else f
    if mode == "zero_pad":
        f = int(math.floor(v))
        return 0 if f < 0 else (n_in - 2 if f + 1 > n_in - 1 else f)
    return int(min(max(0.0, v), n_in - 1))


def a8_axis_scale(n_out, n_in, pol):
    return (n_out - pol[1]) / (n_in - pol[1])


# ---- integer bilinear (A8 / A16 single core) ------------------------------------

def blob_tables(u16, Ho, Wo, separable):
    """(hy, wx, weights) of one blob (after the c0 / c1 words)."""
    n_idx = ceil_to(Ho + Wo, 4)
    hy, wx = u16[:Ho].astype(np.int64), u16[Ho:Ho + Wo].astype(np.int64)
    if separable:
        qy = u16[n_idx:n_idx + 2 * Ho].reshape(Ho, 2)
        qx = u16[n_idx + 2 * Ho:n_idx + 2 * Ho + 2 * Wo].reshape(Wo, 2)
        return hy, wx, (qy, qx)
    return hy, wx, u16[n_idx:n_idx + 4 * Ho * Wo].reshape(Ho, Wo, 4)


def put(table, sl, value, what):
    """table[sl] = value, asserting agreement where already set."""
    cur = table[sl]
    set_ = cur >= 0 if cur.dtype.kind == "i" else ~np.isnan(cur)
    assert np.all(cur[set_] == np.asarray(value)[set_]), f"blobs disagree on {what}"
    table[sl] = value


def a8_case(case, t):
    Ho_sv, Wo_sv = t["ofm_sv_height"], t["ofm_sv_width"]
    SH, SW = t["ifm_height_iteration"], t["ifm_width_iteration"]
    depth = t["depth"]
    C = depth * t["depth_iteration"]
    Hi, Wi = t["ifm_sv_height"] * SH, t["ifm_sv_width"] * SW
    Ho, Wo = Ho_sv * SH, Wo_sv * SW
    gran = 32 if depth <= 32 else 64
    nt, dt = ty(t["dtype"])
    separable = t.get("separable_gate", 0) == SEP_GATE
    n_idx = ceil_to(Ho_sv + Wo_sv, 4)
    blob = 8 + 2 * n_idx + (4 * (Ho_sv + Wo_sv) if separable else 8 * Ho_sv * Wo_sv)
    raw = data(case, "wts32.txt")
    assert raw.size >= SH * SW * blob
    pol = a8_policy(t)
    mode = t.get("boundary_mode", "clamp")
    ifh, ifw = t["ifm_full_height"], t["ifm_full_width"]
    sc_h = a8_axis_scale(t["ofm_full_height"], ifh, pol)
    sc_w = a8_axis_scale(t["ofm_full_width"], ifw, pol)
    ih, iw = np.full(Ho, -1, np.int64), np.full(Wo, -1, np.int64)
    if separable:
        qy, qx = np.full((Ho, 2), -1, np.int64), np.full((Wo, 2), -1, np.int64)
    else:
        s = np.full((Ho, Wo, 4), -1, np.int64)
    c01 = set()
    for sh in range(SH):
        for sw in range(SW):
            b = raw[(sh * SW + sw) * blob:(sh * SW + sw + 1) * blob]
            c01.add(bytes(b[:8]))
            off_h = t["ofm_sv_offset_height"] + sh * Ho_sv
            off_w = t["ofm_sv_offset_width"] + sw * Wo_sv
            base_h = a8_first_tap_base(off_h, sc_h, ifh, mode, pol)
            base_w = a8_first_tap_base(off_w, sc_w, ifw, mode, pol)
            hy, wx, w = blob_tables(b[8:].copy().view(np.uint16), Ho_sv, Wo_sv, separable)
            rows, cols = slice(sh * Ho_sv, (sh + 1) * Ho_sv), slice(sw * Wo_sv, (sw + 1) * Wo_sv)
            put(ih, rows, hy + base_h, "rows")
            put(iw, cols, wx + base_w, "columns")
            if separable:
                put(qy, rows, w[0], "qy")
                put(qx, cols, w[1], "qx")
            else:
                s[rows, cols] = w
    assert len(c01) == 1
    c0, c1 = np.frombuffer(c01.pop(), F32)
    wts = np.concatenate([qy.reshape(-1), qx.reshape(-1)]) if separable else s.reshape(-1)
    x = hcwc(case, "ifm32.txt", dt, Hi, Wi, C, gran)
    want = hcwc(case, "ofm32.txt", dt, Ho, Wo, C, gran)
    return dict(x=x, ih=ih, iw=iw, wts=wts.astype(np.uint16), route=int(separable),
                sv_wo=Wo_sv, req=int(t.get("requantize", 0)), c0=float(c0), c1=float(c1),
                nt=nt, dt=dt, want=want)


def mc_case(case, t):
    """Tile replay: K row-major tiles, one padded input window and one blob each."""
    Ho_sv, Wo_sv = t["ofm_sv_height"], t["ofm_sv_width"]
    Hi, Wi = t["ifm_full_height"], t["ifm_full_width"]
    Ho, Wo = t["ofm_full_height"], t["ofm_full_width"]
    TH, TW = Ho // Ho_sv, Wo // Wo_sv
    assert TH * Ho_sv == Ho and TW * Wo_sv == Wo
    C = t["depth"]
    gran = 32 if C <= 32 else 64
    nt, dt = ty(t["dtype"])
    iph, ipw = t["ifm_sv_height"] + 2, t["ifm_sv_width"] + 2
    pol = a8_policy(t)
    mode = t.get("boundary_mode", "clamp")
    sH, sW = a8_axis_scale(Ho, Hi, pol), a8_axis_scale(Wo, Wi, pol)

    def win_org(off, s, N, pad):
        rel = a8_first_tap_base(off, s, N, mode, pol)
        return rel if rel + pad <= N else max(0, N - pad)

    n_idx = ceil_to(Ho_sv + Wo_sv, 4)
    blob = 8 + 2 * n_idx + 4 * (Ho_sv + Wo_sv)
    raw = data(case, "wts32.txt")
    win_bytes = iph * ipw * C * np.dtype(dt).itemsize
    ifm = data(case, "ifm32.txt")
    x = np.zeros((Hi, Wi, C), dt)
    ih, iw = np.full(Ho, -1, np.int64), np.full(Wo, -1, np.int64)
    qy, qx = np.full((Ho, 2), -1, np.int64), np.full((Wo, 2), -1, np.int64)
    c01 = set()
    for k in range(TH * TW):
        th, tw = divmod(k, TW)
        oh0 = win_org(th * Ho_sv, sH, Hi, iph)
        ow0 = win_org(tw * Wo_sv, sW, Wi, ipw)
        win = ifm[k * win_bytes:(k + 1) * win_bytes].copy().view(dt)
        win = win.reshape(iph, C // gran, ipw, gran).transpose(0, 2, 1, 3).reshape(iph, ipw, C)
        hh, ww = min(iph, Hi - oh0), min(ipw, Wi - ow0)
        put_img = x[oh0:oh0 + hh, ow0:ow0 + ww]
        put_img[...] = win[:hh, :ww]
        b = raw[k * blob:(k + 1) * blob]
        c01.add(bytes(b[:8]))
        hy, wx, (by, bx) = blob_tables(b[8:].copy().view(np.uint16), Ho_sv, Wo_sv, True)
        rows, cols = slice(th * Ho_sv, (th + 1) * Ho_sv), slice(tw * Wo_sv, (tw + 1) * Wo_sv)
        put(ih, rows, hy + oh0, "rows")
        put(iw, cols, wx + ow0, "columns")
        put(qy, rows, by, "qy")
        put(qx, cols, bx, "qx")
    assert len(c01) == 1
    c0, c1 = np.frombuffer(c01.pop(), F32)
    want = hcwc(case, "ofm32.txt", dt, Ho, Wo, C, gran)
    wts = np.concatenate([qy.reshape(-1), qx.reshape(-1)]).astype(np.uint16)
    return dict(x=x, ih=ih, iw=iw, wts=wts, route=1, sv_wo=Wo_sv,
                req=int(t.get("requantize", 0)), c0=float(c0), c1=float(c1), nt=nt, dt=dt,
                want=want)


def run_int(p):
    Hi, Wi, C = p["x"].shape
    Ho, Wo = len(p["ih"]), len(p["iw"])
    return call("npu_resize_bilinear", p["x"], OUT, p["nt"], I64(Hi), I64(Wi), I64(C), I64(Ho),
                I64(Wo), p["ih"].astype(np.int32), p["iw"].astype(np.int32), p["wts"], p["route"],
                I64(p["sv_wo"]), p["req"], p["c0"], p["c1"],
                out_nbytes=Ho * Wo * C * np.dtype(p["dt"]).itemsize, out_dtype=p["dt"])


@pytest.mark.parametrize("d,case", cases(A8_DIRS))
def test_resize_bilinear(d, case):
    p = a8_case(case, json.load(open(os.path.join(case, "testcase.json"))))
    assert_bytes(run_int(p), p["want"])


@pytest.mark.parametrize("d,case", cases(MC_DIRS))
def test_resize_bilinear_multicore(d, case):
    p = mc_case(case, json.load(open(os.path.join(case, "testcase.json"))))
    assert_bytes(run_int(p), p["want"])


# ---- bf16 / f16 / fp8 bilinear ----------------------------------------------------

COORD = {"half_pixel": (0.5, 0.0), "asymmetric": (0.0, 0.0), "align_corners": (0.0, 1.0)}


def bf16_rel_off(off, n_in, n_out, mode):
    """int(raw[0]) of ResizeBilinearBf16Wrapper._pack_one_sv: the clamped
    coordinate of the sub-volume's first output, stored as fp32, truncated."""
    grid, adj = COORD[mode]
    scale = (n_out - adj) / (n_in - adj)
    v = min(max(0.0, (off + grid) / scale - grid), n_in - 1)
    return int(F32(v))


def bf16_case(case, t, fp8):
    Ho_sv, Wo_sv = t["ofm_sv_height"], t["ofm_sv_width"]
    SH, SW = t["height_iteration"], t["width_iteration"]
    C = t["sv_depth"] * t["depth_iteration"]
    Hi, Wi = t["ifm_sv_height"] * SH, t["ifm_sv_width"] * SW
    Ho, Wo = Ho_sv * SH, Wo_sv * SW
    mode = t.get("coordinate_transformation_mode", "half_pixel")
    nt = ty(t["compute_dtype"] if fp8 else t["dtype"])[0]
    n_idx = ceil_to(Ho_sv + Wo_sv, 8)
    blob = 2 * n_idx + 8 * Ho_sv * Wo_sv
    raw = data(case, "wts32.txt")
    assert raw.size >= SH * SW * blob
    ih, iw = np.full(Ho, -1, np.int64), np.full(Wo, -1, np.int64)
    s = np.zeros((Ho, Wo, 4), np.uint16)
    for sh in range(SH):
        for sw in range(SW):
            u16 = raw[(sh * SW + sw) * blob:(sh * SW + sw + 1) * blob].copy().view(np.uint16)
            rows, cols = slice(sh * Ho_sv, (sh + 1) * Ho_sv), slice(sw * Wo_sv, (sw + 1) * Wo_sv)
            put(ih, rows, u16[:Ho_sv] + bf16_rel_off(sh * Ho_sv, Hi, Ho, mode), "rows")
            put(iw, cols, u16[Ho_sv:Ho_sv + Wo_sv] + bf16_rel_off(sw * Wo_sv, Wi, Wo, mode),
                "columns")
            s[rows, cols] = u16[n_idx:n_idx + 4 * Ho_sv * Wo_sv].reshape(Ho_sv, Wo_sv, 4)
    dt = np.uint8 if fp8 else np.uint16
    p = dict(x=hcwc(case, "ifm32.txt", dt, Hi, Wi, C, 32), ih=ih, iw=iw, s=s, sv_wo=Wo_sv,
             nt=nt, dt=dt, want=hcwc(case, "ofm32.txt", dt, Ho, Wo, C, 32))
    if fp8:
        w = rtp_words(case)
        if w is not None:
            p["dq"], p["oq"] = (float(v) for v in w[19:21].view(F32))
        else:  # as the YAML evaluates them
            p["dq"] = float(F32(t["ifm_scale"]))
            p["oq"] = float(F32(1.0 / float(t["ofm_scale"])))
    return p


def run_float(p, fp8):
    Hi, Wi, C = p["x"].shape
    Ho, Wo = len(p["ih"]), len(p["iw"])
    args = [p["x"], OUT, p["nt"], I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo),
            p["ih"].astype(np.int32), p["iw"].astype(np.int32), p["s"], I64(p["sv_wo"])]
    if fp8:
        args += [p["dq"], p["oq"]]
    return call("npu_resize_bilinear_fp8" if fp8 else "npu_resize_bilinear_bf16", *args,
                out_nbytes=Ho * Wo * C * np.dtype(p["dt"]).itemsize, out_dtype=p["dt"])


@pytest.mark.parametrize("d,case", cases(BF16_DIRS))
def test_resize_bilinear_bf16(d, case):
    p = bf16_case(case, json.load(open(os.path.join(case, "testcase.json"))), False)
    assert_bytes(run_float(p, False), p["want"])


@pytest.mark.parametrize("d,case", cases(FP8_DIRS))
def test_resize_bilinear_fp8(d, case):
    p = bf16_case(case, json.load(open(os.path.join(case, "testcase.json"))), True)
    assert_bytes(run_float(p, True), p["want"])


# ---- bicubic ------------------------------------------------------------------------

CUBIC_MODES = ("half_pixel", "pytorch_half_pixel", "asymmetric", "align_corners")


def cubic_origin(t0, n_in, n_out, mode, staged):
    """ResizeCubicBf16Wrapper.sv_tile_origin."""
    grid, adj = COORD["half_pixel" if mode == "pytorch_half_pixel" else mode]
    scale = (n_out - adj) / (n_in - adj)
    src = (t0 + grid) / scale - grid
    start = min(max(int(np.floor(src)) - 1, 0), n_in - 4)
    return min(max(start, 0), max(n_in - staged, 0))


def cubic_case(case, t):
    Ho_sv, Wo_sv = t["ofm_sv_height"], t["ofm_sv_width"]
    SH, SW = t["height_iteration"], t["width_iteration"]
    C = t["sv_depth"] * t["depth_iteration"]
    Hi, Wi = t["ifm_full_height"], t["ifm_full_width"]
    Ho, Wo = t["ofm_full_height"], t["ofm_full_width"]
    assert Ho == Ho_sv * SH and Wo == Wo_sv * SW
    mode = CUBIC_MODES[int(t["co_trans_mode"])]
    st_h, st_w = t["ifm_sv_height"], (t["ifm_sv_width"] + 1) // 2 * 2
    n_idx = ceil_to(Ho_sv + Wo_sv, 16)
    n_cy = ceil_to(4 * Ho_sv, 16)
    blob = 2 * (n_idx + n_cy + 4 * Wo_sv)
    raw = data(case, "wts32.txt")
    assert raw.size >= SH * SW * blob
    ih, iw = np.full(Ho, -1, np.int64), np.full(Wo, -1, np.int64)
    cy, cx = np.full((Ho, 4), -1, np.int64), np.full((Wo, 4), -1, np.int64)
    for sh in range(SH):
        for sw in range(SW):
            u16 = raw[(sh * SW + sw) * blob:(sh * SW + sw + 1) * blob].copy().view(np.uint16)
            rows, cols = slice(sh * Ho_sv, (sh + 1) * Ho_sv), slice(sw * Wo_sv, (sw + 1) * Wo_sv)
            put(ih, rows, u16[:Ho_sv] + cubic_origin(sh * Ho_sv, Hi, Ho, mode, st_h), "rows")
            put(iw, cols, u16[Ho_sv:Ho_sv + Wo_sv] + cubic_origin(sw * Wo_sv, Wi, Wo, mode, st_w),
                "columns")
            put(cy, rows, u16[n_idx:n_idx + 4 * Ho_sv].reshape(Ho_sv, 4), "coef_y")
            put(cx, cols, u16[n_idx + n_cy:n_idx + n_cy + 4 * Wo_sv].reshape(Wo_sv, 4), "coef_x")
    return dict(x=hcwc(case, "ifm32.txt", np.uint16, Hi, Wi, C, 32), ih=ih, iw=iw,
                cy=cy.astype(np.uint16), cx=cx.astype(np.uint16),
                want=hcwc(case, "ofm32.txt", np.uint16, Ho, Wo, C, 32))


def run_cubic(p):
    Hi, Wi, C = p["x"].shape
    Ho, Wo = len(p["ih"]), len(p["iw"])
    return call("npu_resize_cubic_bf16", p["x"], OUT, I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo),
                p["ih"].astype(np.int32), p["iw"].astype(np.int32), p["cy"], p["cx"],
                out_nbytes=Ho * Wo * C * 2, out_dtype=np.uint16)


@pytest.mark.parametrize("d,case", cases(CUBIC_DIRS))
def test_resize_cubic_bf16(d, case):
    p = cubic_case(case, json.load(open(os.path.join(case, "testcase.json"))))
    assert_bytes(run_cubic(p), p["want"])
