# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The qdq-group entry points against the NPU: each mllib regression case's
inputs are converted from the NPU's physical layout (C64 blocking, padded
sub-volumes) to the logical layout, run through the entry point, and the
result is compared byte for byte with the aiesim output (data/ofm32.txt) on
the region the regression grades (tensor_descr.h's valid shape where the case
has one, otherwise every output element the kernel defines)."""

import ctypes
import json
import os

import numpy as np
import pytest

from aiesim_common import cases, data, words
from npu import I64, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT
from npu import DevList, Host, Out, U32
from npu import call as callx
from npu import callx as callx_n
from test_golden_qdq import GridQdq

INT = {"int8": (NPU_I8, np.int8), "uint8": (NPU_U8, np.uint8),
       "int16": (NPU_I16, np.int16), "uint16": (NPU_U16, np.uint16)}


def signed(bits, s):
    return {8: ("uint8", "int8"), 16: ("uint16", "int16")}[bits][int(bool(s))]


def go(name, *args, n, dt):
    """Call `name` with an n-element output of numpy dtype dt."""
    return callx(name, *args, out_nbytes=n * np.dtype(dt).itemsize, out_dtype=dt)


def coeffs(case, k, name="coeff32.txt"):
    return [float(v) for v in words(case, name)[:k].view(np.float32)]


def shift_of(case, name="coeff32.txt"):
    """The int16 shift in the low half of word 2."""
    return int(words(case, name)[2:3].view(np.int16)[0])


def phys3(x, valid):
    """The valid [H][W][C] corner of a padded HWC tensor."""
    return np.ascontiguousarray(x[:valid[0], :valid[1], :valid[2]])


def from_c64(x, H, W, C, gc=64):
    """[H][C/gc][W][gc] blocked -> [H][W][C]."""
    return np.ascontiguousarray(x.reshape(H, C // gc, W, gc).transpose(0, 2, 1, 3).reshape(H, W, C))


# Each runner returns (got, want) over the graded region.

def run_unary(case, t, op, bits):
    it, idt = INT[t["ifm_dtype"]]
    ot, odt = INT[t["ofm_dtype"]]
    x = data(case, "ifm32.txt", idt)
    want = data(case, "ofm32.txt", odt)
    n = len(x)
    k = {"abs": 3, "neg": 2, "clip": 4}[op]
    got = go(f"npu_elementwise_{op}_a{bits}", x, it, OUT, ot, I64(n), *coeffs(case, k),
             n=n, dt=odt)
    return got, want[:n]


def run_binary(case, t, op, bits):
    if op == "max":
        a_t = b_t = t["ifm_dtype"]
        ot, odt = INT[t["ofm_dtype"]]
    else:
        a_t, b_t = t["ifm1_dtype"], t["ifm2_dtype"]
        ot, odt = NPU_I8, np.int8
    at, adt = INT[a_t]
    bt, bdt = INT[b_t]
    a = data(case, "ifm32.txt", adt)
    b = data(case, "wts32.txt", bdt)
    want = data(case, "ofm32.txt", odt)
    if op == "max":
        got = go(f"npu_elementwise_max_a{bits}", a, at, b, bt, OUT, ot, I64(len(a)),
                 *coeffs(case, 6), n=len(a), dt=odt)
    else:
        got = go(f"npu_elementwise_{op}_a{bits}", a, at, b, bt, OUT, I64(len(a)),
                 *coeffs(case, 4), n=len(a), dt=odt)
    return got, want[:len(a)]


def where_shapes(t):
    it = [t["X_height_iteration"], t["X_width_iteration"], t["X_channel_iteration"]]
    xv = [t["X_valid_full_height"], t["X_valid_full_width"], t["X_valid_full_channel"]]
    xs = [t["X_physical_sv_height"], t["X_physical_sv_width"], t["X_physical_sv_channel"]]
    cv = [t["condition_valid_full_height"], t["condition_valid_full_width"],
          t["condition_valid_full_channel"]]
    cs = [t["condition_physical_sv_height"], t["condition_physical_sv_width"],
          t["condition_physical_sv_channel"]]
    xp = [s * i for s, i in zip(xs, it)]
    cp = [s * i if v > 1 else s for s, i, v in zip(cs, it, cv)]
    return xv, xp, cv, cp


def run_where(case, t, bits, qdq):
    xv, xp, cv, cp = where_shapes(t)
    dt = INT[t["dtype"] if qdq else t["ifm_dtype"]][1]
    cdt = np.int16 if qdq and bits == 16 else np.int8
    cond = phys3(data(case, "ifm32.txt", cdt).reshape(cp), cv)
    x = phys3(data(case, "wts32.txt", dt).reshape(xp), xv)
    want = phys3(data(case, "ofm32.txt", dt).reshape(xp), xv)
    shp = [I64(v) for v in xv + cv]
    pos = t["scalar_position"]
    if qdq:
        scale, offset = coeffs(case, 2)
        y = int(words(case)[2:3].view(np.int32)[0])
        got = go(f"npu_qdq_elementwise_broadcasting3d_where_a{bits}", cond, x, INT[t["dtype"]][0],
                 OUT, *shp, y, pos, scale, offset, n=x.size, dt=dt)
    else:
        y = int(data(case, "ifm2_32.txt", np.uint8 if bits == 8 else np.uint16)[0])
        got = go(f"npu_elementwise_broadcasting3d_where_a{bits}", cond, x, OUT, *shp, U32(y), pos,
                 n=x.size, dt=dt)
    return got, want


def run_min(case, t, bits):
    it = [t["ifm1_height_iteration"], t["ifm1_width_iteration"], t["ifm1_channel_iteration"]]
    av = [t[f"ifm1_valid_full_{d}"] for d in ("height", "width", "channel")]
    a_s = [t[f"ifm1_physical_sv_{d}"] for d in ("height", "width", "channel")]
    bv = [t[f"ifm2_valid_full_{d}"] for d in ("height", "width", "channel")]
    b_s = [t[f"ifm2_physical_sv_{d}"] for d in ("height", "width", "channel")]
    ap = [s * i for s, i in zip(a_s, it)]
    bp = [s * i if v > 1 else s for s, i, v in zip(b_s, it, bv)]
    nt, dt = INT[t["dtype"]]
    a = phys3(data(case, "ifm32.txt", dt).reshape(ap), av)
    b = phys3(data(case, "wts32.txt", dt).reshape(bp), bv)
    want = phys3(data(case, "ofm32.txt", dt).reshape(ap), av)
    got = go(f"npu_qdq_elementwise_broadcasting3d_min_a{bits}", a, b, nt, OUT,
             *[I64(v) for v in av + bv], *coeffs(case, 6), n=a.size, dt=dt)
    return got, want


def run_reduce(case, t, op, bits, qdq):
    nt, dt = INT[t["dtype"]]
    H = t["ifm_sv_height"] * t["ifm_height_iteration"]
    W = t["ifm_sv_width"] * t["ifm_width_iteration"]
    Cp = t["ifm_sv_depth"] * t["ifm_depth_iteration"]
    C = t.get("ifm_sv_depth_real", t["ifm_sv_depth"]) * t["ifm_depth_iteration"]
    x = np.ascontiguousarray(from_c64(data(case, "ifm32.txt", dt), H, W, Cp)[..., :C])
    ax = t["axis_flags"]
    oH, oW = (1 if ax & 4 else H), (1 if ax & 2 else W)
    want = data(case, "ofm32.txt", dt)
    if ax & 1:   # one value per (h, w), 4-byte slots with keepdims (and always for 7)
        n = oH * oW
        if (t["keepdims"] and ax in (1, 3, 5)) or ax == 7:
            want = want[::4 // np.dtype(dt).itemsize]
        want = want[:n]
    else:
        n = oH * oW * C
        want = from_c64(want[:oH * oW * Cp], oH, oW, Cp)[..., :C]
    args = (x, nt, OUT, I64(H), I64(W), I64(C), ax)
    if qdq:
        offset, scale = coeffs(case, 2)
        got = go(f"npu_qdq_reduce_{op}_c64_a{bits}", *args, offset, scale, shift_of(case),
                 n=n, dt=dt)
    else:
        got = go(f"npu_reduce_{op}_c64_a{bits}", *args, n=n, dt=dt)
    return got, want


def run_maxpool(case, t, bits, kind):
    it, idt = INT[t["ifm_dtype"]]
    ot, odt = INT[t.get("ofm_dtype", t["ifm_dtype"])]
    gc = 64 // int(t.get("granX", 1))
    C = t["ofm_sv_depth"] * t["ifm_depth_iteration"]
    if kind == "global":
        Hi = t["ifm_sv_height"] * t["ifm_height_iteration"]
        Wi = t["ifm_sv_width"] * t["ifm_width_iteration"]
        H = W = 1
    else:
        H = t["ofm_sv_height"] * t["ofm_height_iteration"]
        W = t["ofm_sv_width"] * t["ofm_width_iteration"]
        kh, kw, sh, sw = (t["kernel_height"], t["kernel_width"], t["stride_height"],
                          t["stride_width"])
        Hi, Wi = (H - 1) * sh + kh, (W - 1) * sw + kw
    x = from_c64(data(case, "ifm32.txt", idt), Hi, Wi, C, gc)
    want = from_c64(data(case, "ofm32.txt", odt)[:H * W * C], H, W, C, gc)
    if kind == "generic":
        got = go("npu_generic_maxpool_a8", x, it, OUT, I64(Hi), I64(Wi), I64(C), kh, kw, sh, sw,
                 n=want.size, dt=odt)
        return got, want
    offset, scale = coeffs(case, 2)
    if kind == "global":
        got = go(f"npu_qdq_globalmaxpool_a{bits}", x, it, OUT, ot, I64(Hi), I64(Wi), I64(C),
                 offset, scale, shift_of(case), n=want.size, dt=odt)
    else:
        got = go(f"npu_qdq_maxpool_a{bits}", x, it, OUT, ot, I64(Hi), I64(Wi), I64(C), kh, kw,
                 sh, sw, offset, scale, shift_of(case), n=want.size, dt=odt)
    return got, want


def run_concat(case, t, bits):
    flags = t["sign_flags"]
    it, idt = INT[signed(bits, flags & 1)]
    ot, odt = INT[signed(bits, flags & 2)]
    rows = (t["ifm1_physical_sv_X"] * t["ifm1_physical_sv_Y"] * t["ifm1_physical_sv_B"] *
            t["ifm_X_iteration"] * t["ifm_Y_iteration"] * t["ifm_B_iteration"])
    a = data(case, "ifm32.txt", idt).reshape(rows, t["ifm1_physical_sv_C"])
    b = data(case, "wts32.txt", idt).reshape(rows, t["ifm2_physical_sv_C"])
    a = np.ascontiguousarray(a[:, t["in1_start_idx"]:t["in1_start_idx"] + t["ifm1_valid_sv_C"]])
    b = np.ascontiguousarray(b[:, t["in2_start_idx"]:t["in2_start_idx"] + t["ifm2_valid_sv_C"]])
    ca, cb = a.shape[1], b.shape[1]
    want = data(case, "ofm32.txt", odt).reshape(rows, t["ofm_physical_sv_C"])[:, :ca + cb]
    got = go(f"npu_qdq_concat_a{bits}", a, b, it, OUT, ot, I64(rows), I64(ca), I64(cb),
             *coeffs(case, 2), n=rows * (ca + cb), dt=odt)
    return got, want


def run_concat_multi(case, t, bits):
    if bits:   # _a8 / _a16: one coefficient pair, sign_flags
        flags = t["sign_flags"]
        sn, so = flags & 1, flags & 2
    else:
        bits = 8 if "8" in t["ifm_dtype_mapped"] else 16
        sn, so = t["sign_N"], t["sign_O"]
    it, idt = INT[signed(bits, sn)]
    ot, odt = INT[signed(bits, so)]
    rows = (t["ifm_physical_sv_X"] * t["ifm_physical_sv_Y"] * t["ifm_physical_sv_B"] *
            t["ifm_X_iteration"] * t["ifm_Y_iteration"] * t["ifm_B_iteration"])
    n_in = t["concat_volumes"]
    flat = data(case, "ifm32.txt", idt)
    ins, pos = [], 0
    for k in range(n_in):
        pc, vc = t[f"ifm{k}_physical_sv_C"], t[f"ifm{k}_valid_sv_C"]
        ins.append(np.ascontiguousarray(flat[pos:pos + rows * pc].reshape(rows, pc)[:, :vc]))
        pos += rows * pc
    ch = [x.shape[1] for x in ins]
    total = sum(ch)
    want = data(case, "ofm32.txt", odt).reshape(rows, t["ofm_physical_sv_C"])[:, :total]
    chans = Host((ctypes.c_int64 * n_in)(*ch))
    if "sign_flags" in t:
        got = go(f"npu_qdq_concat_multi_c_fusion_a{bits}", n_in, DevList(ins), it, OUT, ot,
                 I64(rows), chans, *coeffs(case, 2), n=rows * total, dt=odt)
    else:
        c = coeffs(case, 2 * n_in)
        s = Host((ctypes.c_float * n_in)(*c[0::2]))
        o = Host((ctypes.c_float * n_in)(*c[1::2]))
        got = go("npu_qdq_concat_multi_c_fusion", n_in, DevList(ins), it, OUT, ot, I64(rows),
                 chans, s, o, n=rows * total, dt=odt)
    return got, want


def run_gather(case, t):
    it, idt = INT[t["ifm_dtype"]]
    ot, odt = INT[t["ofm_dtype"]]
    Np = t["ifm_physical_sv_N"] * t["ifm_N_iteration"]
    Mp = t["ifm_physical_sv_M"] * t["ifm_M_iteration"]
    Mwp = t["wts_physical_sv_M"] * t["wts_M_iteration"]
    M, N, Mo = t["ifm_valid_full_M"], t["ifm_valid_full_N"], t["wts_valid_full_M"]
    x = np.ascontiguousarray(data(case, "ifm32.txt", idt).reshape(Mp, Np)[:M, :N])
    idx = np.ascontiguousarray(data(case, "wts32.txt", np.int32).reshape(Mwp, Np)[:Mo, :N])
    want = data(case, "ofm32.txt", odt).reshape(Mwp, Np)[:Mo, :N]
    got = go("npu_qdq_gather_element_2din_2dindices_axis0", x, it, idx, OUT, ot, I64(M), I64(N),
             I64(Mo), *coeffs(case, 2), n=Mo * N, dt=odt)
    return got, want


def run_gridsample(case, t, variant):
    nt, dt = INT[t["ifm_dtype"]]
    H, W, C, Ho, Wo = t["in_h"], t["in_w"], t["in_c"], t["out_h"], t["out_w"]
    cin, ci = t["sv_in_c"], t["c_iteration"]
    align = 32 if variant == "a16" else 64
    flat = data(case, "ifm32.txt", dt)
    chunks, pos = [], 0
    for _ in range(ci):
        chunks.append(flat[pos:pos + H * W * cin].reshape(H, W, cin))
        pos += -(-H * W * cin // align) * align
    x = np.ascontiguousarray(np.concatenate(chunks, axis=-1))
    out = data(case, "ofm32.txt", dt)
    if variant == "a16":
        want = out[:Ho * Wo * C].reshape(Ho, Wo, C)
    else:
        sw, wi = t["sv_out_w"], t["w_iteration"]
        want = np.empty((Ho, Wo, C), dt)
        pos = 0
        for c in range(ci):
            for w in range(wi):
                n = Ho * sw * cin
                want[:, w * sw:(w + 1) * sw, c * cin:(c + 1) * cin] = \
                    out[pos:pos + n].reshape(Ho, sw, cin)
                pos += n
    w = words(case)
    co = GridQdq(*[int(w[k:k + 1].view(np.int32)[0]) if k % 2 == 0 else
                   float(w[k:k + 1].view(np.float32)[0]) for k in range(6)])
    if variant == "a8w8":
        gt, gdt = INT[t["wts_dtype"]]
        grid = np.ascontiguousarray(data(case, "wts32.txt", gdt)[:Ho * Wo * 4].reshape(-1, 4)[:, :2])
        gargs = (grid, gt)
    else:
        gargs = (np.ascontiguousarray(data(case, "wts32.txt", np.uint16)[:Ho * Wo * 2]),)
    got = go(f"npu_qdq_gridsample_nearest_{variant}", x, nt, *gargs, OUT, I64(H), I64(W), I64(C),
             I64(Ho), I64(Wo), t["align_corners"], t["padding_mode"], Host(ctypes.byref(co)),
             n=Ho * Wo * C, dt=dt)
    return got, want


def nearest_index(n_out, n_in):
    """ONNX Resize nearest, half_pixel + round_prefer_floor (the wrapper's
    defaults), clamped to the input."""
    x = np.maximum((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0)
    r = np.where(x - np.floor(x) == 0.5, np.floor(x), np.round(x))
    return np.minimum(r, n_in - 1).astype(np.int32)


def run_resize(case, t):
    nt, dt = INT[t["dtype"]]
    Hi, Wi = t["ifm_full_height"], t["ifm_full_width"]
    Ho, Wo = t["ofm_full_height"], t["ofm_full_width"]
    C = t["depth"] * t["depth_iteration"]
    x = from_c64(data(case, "ifm32.txt", dt), Hi, Wi, C)
    want = from_c64(data(case, "ofm32.txt", dt)[:Ho * Wo * C], Ho, Wo, C)
    sy, sx = nearest_index(Ho, Hi), nearest_index(Wo, Wi)
    w = words(case, "wts32.txt")
    if t["ifm_height_iteration"] == 1 and t["ifm_width_iteration"] == 1:
        table = w[2:].view(np.uint16)
        assert (table[:Ho] == sy).all() and \
            (table[t["wts_x_offset"]:t["wts_x_offset"] + Wo] == sx).all(), "index tables"
    scale, offset = (float(v) for v in w[:2].view(np.float32))
    got = go("npu_qdq_resize_nearest", x, nt, OUT, I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo),
             sy, sx, t["requantize"], scale, offset, n=Ho * Wo * C, dt=dt)
    return got, want


# ---- H: data movement with fused requantization -------------------------------
# The inputs and graded regions are those of the generic_* movement wrappers
# (test_aiesim_movement.py); c1 / c0 are coeff32.txt words 0 and 1.

def fused_dtype(t):
    return INT[t.get("ifm_dtype", t.get("dtype"))]


def run_fused_reshape(case, t):
    tt, dt = fused_dtype(t)
    phys = [t[f"ifm_physical_dim_{i}"] * t[f"ifm_dim_{i}_iter"] for i in range(4)]
    vin, vout, pout = t["ifm_valid_dim_3"], t["ofm_valid_dim_3"], t["ofm_physical_dim_3"]
    x = data(case, "ifm32.txt", dt)[:int(np.prod(phys))].reshape(-1, phys[3])[:, :vin]
    x = np.ascontiguousarray(x).reshape(-1)
    rows = x.size // vout
    got = go("npu_qdq_reshape_innermost_requantize_fused", x, OUT, tt, I64(x.size),
             *coeffs(case, 2), n=x.size, dt=dt)
    want = data(case, "ofm32.txt", dt)[:rows * pout].reshape(rows, pout)[:, :vout]
    return got, want


def sv_full(t, pre):
    return [t[f"{pre}_physical_sv_{d}"] * t[f"ifm_{d}_iteration"]
            for d in ("batch", "height", "width", "channel")]


def run_fused_slice_innermost(case, t):
    tt, dt = fused_dtype(t)
    dims = ("batch", "height", "width", "channel")
    phys = sv_full(t, "ifm")
    valid = [t[f"ifm_valid_sv_{d}"] if t[f"ifm_{d}_iteration"] == 1 else phys[i]
             for i, d in enumerate(dims)]
    valid[3] = t["ifm_valid_sv_channel"]
    x = data(case, "ifm32.txt", dt)[:int(np.prod(phys))].reshape(phys)
    x = np.ascontiguousarray(x[:valid[0], :valid[1], :valid[2], :valid[3]])
    rows = valid[0] * valid[1] * valid[2]
    s, e, st = t["slice_inner_start"], t["slice_inner_end"], t.get("slice_inner_stride", 1)
    wo = len(range(s, e, st))
    got = go("npu_qdq_slice_innermost_requantize_fused", x.reshape(rows, valid[3]), OUT, tt,
             I64(rows), I64(valid[3]), I64(s), I64(e), I64(st), *coeffs(case, 2),
             n=rows * wo, dt=dt)
    ophys = [valid[i] + t[f"ofm_physical_sv_{d}"] - t[f"ofm_valid_sv_{d}"]
             for i, d in enumerate(dims[:3])] + [wo + t["ofm_physical_sv_channel"] -
                                                 t["ofm_valid_sv_channel"]]
    want = data(case, "ofm32.txt", dt)[:int(np.prod(ophys))].reshape(ophys)
    return got, want[:valid[0], :valid[1], :valid[2], :wo]


def run_fused_slice_16b_inner_range(case, t):
    tt, dt = fused_dtype(t)
    n, h, w, c = sv_full(t, "ifm")
    rows, cv = n * h * w, t["ifm_valid_sv_channel"]
    s, e, co = t["slice_inner_start"], t["slice_inner_end"], t["ofm_physical_sv_channel"]
    x = np.ascontiguousarray(data(case, "ifm32.txt", dt)[:rows * c].reshape(rows, c)[:, :cv])
    got = go("npu_qdq_slice_16b_inner_range_requantize_fused", x, OUT, tt, I64(rows), I64(cv),
             I64(s), I64(e), *coeffs(case, 2), n=rows * (e - s), dt=dt)
    return got, data(case, "ofm32.txt", dt)[:rows * co].reshape(rows, co)[:, :e - s]


def run_fused_slice_16b_inner(case, t):
    tt, dt = fused_dtype(t)
    n, h, w, c = sv_full(t, "ifm")
    rows, cv, sp = n * h * w, t["ifm_valid_sv_channel"], t["slice_inner"]
    p0, p1 = t["ofm0_physical_sv_channel"], t["ofm1_physical_sv_channel"]
    x = np.ascontiguousarray(data(case, "ifm32.txt", dt)[:rows * c].reshape(rows, c)[:, :cv])
    es = np.dtype(dt).itemsize
    o0, o1 = callx_n("npu_qdq_slice_16b_inner_requantize_fused", x, Out(0), Out(1), tt,
                     I64(rows), I64(cv), I64(sp), *coeffs(case, 2),
                     outs=[rows * sp * es, rows * (cv - sp) * es])
    want = data(case, "ofm32.txt", dt)[:rows * (p0 + p1)]
    got = np.concatenate([o0.view(dt), o1.view(dt)])
    want = np.concatenate([want[:rows * p0].reshape(rows, p0)[:, :sp].reshape(-1),
                           want[rows * p0:].reshape(rows, p1)[:, :cv - sp].reshape(-1)])
    return got, want


def run_fused_slice_16b_c6463(case, t):
    tt, dt = fused_dtype(t)
    n, h, w, _ = sv_full(t, "ifm")
    rows = n * h * w
    x = data(case, "ifm32.txt", dt)[:rows * 64].reshape(rows, 64)
    es = np.dtype(dt).itemsize
    o0, o1 = callx_n("npu_qdq_slice_16b_c6463_requantize_fused", x, Out(0), Out(1), tt,
                     I64(rows), *coeffs(case, 2), outs=[rows * 63 * es, rows * es])
    return np.concatenate([o0.view(dt), o1.view(dt)]), data(case, "ofm32.txt", dt)[:rows * 64]


def run_fused_transpose_k8n8(case, t):
    tt, dt = fused_dtype(t)
    K = t["ifm_sv_K_physical"] * t["ifm_K_iteration"]
    N = t["ifm_sv_N_physical"] * t["ifm_N_iteration"]
    Kv, Nv = t["ifm_K_full_valid"], t["ifm_N_full_valid"]
    # (K/8, N/8, 8, 8) tiles; only the valid tiles are graded (tensor_descr.h)
    tiles = data(case, "ifm32.txt", dt)[:K * N].reshape(K // 8, N // 8, 8, 8)
    x = np.ascontiguousarray(tiles[:Kv // 8, :Nv // 8].transpose(0, 2, 1, 3)).reshape(Kv, Nv)
    got = go("npu_qdq_transpose_k8n8_requantize_fused", x, OUT, tt, I64(Kv), I64(Nv),
             *coeffs(case, 2), n=Kv * Nv, dt=dt)
    # out tile (kb, nb) holds out[nb*8 + j][kb*8 + i] at [j][i]
    got = got.reshape(Nv // 8, 8, Kv // 8, 8).transpose(2, 0, 1, 3)
    want = data(case, "ofm32.txt", dt)[:K * N].reshape(K // 8, N // 8, 8, 8)
    return np.ascontiguousarray(got), want[:Kv // 8, :Nv // 8]


# The wrapper's perm code -> axis order, innermost first (QDQ_TransposeUnaligned...py)
PERM_CODE = [(3, 2, 1, 0), (3, 2, 0, 1), (3, 1, 2, 0), (3, 1, 0, 2), (3, 0, 2, 1), (3, 0, 1, 2),
             (2, 3, 1, 0), (2, 3, 0, 1), (2, 1, 3, 0), (2, 1, 0, 3), (2, 0, 3, 1), (2, 0, 1, 3),
             (1, 3, 2, 0), (1, 3, 0, 2), (1, 2, 3, 0), (1, 2, 0, 3), (1, 0, 3, 2), (1, 0, 2, 3),
             (0, 3, 2, 1), (0, 3, 1, 2), (0, 2, 3, 1), (0, 2, 1, 3), (0, 1, 3, 2), (0, 1, 2, 3)]


def run_fused_transpose_unaligned(case, t):
    tt, dt = fused_dtype(t)
    axes = ("outer", "mid_o", "mid_i", "inner")
    start = [t[f"Di_{a}_start"] for a in axes]
    full = [t[f"Di_{a}_iter"] * t[f"Di_{a}_sv_physical"] for a in axes]
    valid = [t[f"Di_{a}_full_valid"] for a in axes]
    perm = PERM_CODE[t["perm"]][::-1]
    phys = [s + f for s, f in zip(start, full)]
    x = data(case, "ifm32.txt", dt)[:int(np.prod(phys))].reshape(phys)
    x = np.ascontiguousarray(x[tuple(slice(s, s + v) for s, v in zip(start, valid))])
    got = go("npu_qdq_transpose_unaligned_requantize_fused", x, OUT, tt,
             Host(np.array(valid, np.int64)), Host(np.array(perm, np.int32)), *coeffs(case, 2),
             n=x.size, dt=dt)
    # output axis k comes from input axis perm[k], with that axis's start and garbage
    ostart = [start[p] for p in perm]
    ophys = [start[p] + full[p] for p in perm]
    want = data(case, "ofm32.txt", dt)[:int(np.prod(ophys))].reshape(ophys)
    return got, want[tuple(slice(s, s + valid[p]) for s, p in zip(ostart, perm))]


FUSED = {"reshape_innermost": run_fused_reshape, "slice_innermost": run_fused_slice_innermost,
         "slice_16b_inner_range": run_fused_slice_16b_inner_range,
         "slice_16b_inner": run_fused_slice_16b_inner,
         "slice_16b_c6463": run_fused_slice_16b_c6463,
         "transpose_k8n8": run_fused_transpose_k8n8,
         "transpose_unaligned": run_fused_transpose_unaligned}


def runner(d):
    """Regression dir name -> runner(case, testcase)."""
    w = d.removesuffix("_wrapper")
    if w.endswith("_requantize_fused"):
        return FUSED[w.removeprefix("qdq_").removesuffix("_requantize_fused")]
    bits = 16 if "a16" in w and "a16a8" not in w else 8
    for op in ("abs", "neg", "clip"):
        if w.startswith(f"qdq_elementwise_{op}_"):
            return lambda c, t: run_unary(c, t, op, bits)
    for op in ("equal", "greater", "max"):
        if w.startswith(f"qdq_elementwise_{op}_"):
            return lambda c, t: run_binary(c, t, op, bits)
    if "broadcasting3d_where" in w:
        return lambda c, t: run_where(c, t, bits, w.startswith("qdq_"))
    if "broadcasting3d_min" in w:
        return lambda c, t: run_min(c, t, bits)
    if "reduce_" in w:
        op = "max" if "reduce_max" in w else "min"
        return lambda c, t: run_reduce(c, t, op, bits, w.startswith("qdq_"))
    if "globalmaxpool" in w:
        return lambda c, t: run_maxpool(c, t, bits, "global")
    if w == "generic_maxpool_a8":
        return lambda c, t: run_maxpool(c, t, 8, "generic")
    if "maxpool" in w:
        return lambda c, t: run_maxpool(c, t, bits, "qdq")
    if "concat_multi_c_fusion" in w:
        b = 0 if w == "qdq_concat_multi_c_fusion" else bits
        return lambda c, t: run_concat_multi(c, t, b)
    if "concat" in w:
        return lambda c, t: run_concat(c, t, bits)
    if "gather_element" in w:
        return run_gather
    if "gridsample_nearest" in w:
        v = w.rsplit("_", 1)[1]
        return lambda c, t: run_gridsample(c, t, v)
    if "resize_nearest" in w:
        return run_resize
    raise KeyError(d)


DIRS = [f"qdq_elementwise_{op}_a{b}_wrapper" for op in
        ("abs", "neg", "clip", "equal", "greater", "max") for b in (8, 16)] + \
       [f"{p}elementwise_broadcasting3d_where_a{b}_wrapper" for p in ("", "qdq_") for b in (8, 16)] + \
       [f"qdq_elementwise_broadcasting3d_min_a{b}_wrapper" for b in (8, 16)] + \
       [f"{p}reduce_{op}_c64_a{b}_wrapper" for p in ("", "qdq_") for op in ("max", "min")
        for b in (8, 16)] + \
       [f"qdq_maxpool_a{b}_wrapper" for b in (8, 16)] + ["generic_maxpool_a8_wrapper"] + \
       [f"qdq_globalmaxpool_a{b}_wrapper" for b in (8, 16)] + \
       [f"qdq_concat_a{b}_wrapper" for b in (8, 16)] + \
       [f"qdq_concat_multi_c_fusion{s}_wrapper" for s in ("_a8", "_a16", "")] + \
       [f"qdq_gather_element_2din_2dindices_axis0_{s}_wrapper" for s in
        ("a8a8", "a8a16", "a16a8", "a16a16")] + \
       [f"qdq_gridsample_nearest_{v}_wrapper" for v in ("a8", "a16", "a8w8")] + \
       [f"qdq_resize_nearest_a{b}_wrapper" for b in (8, 16)] + \
       [f"qdq_{op}_requantize_fused_wrapper" for op in FUSED]

CASES = cases(DIRS)


@pytest.mark.skipif(not CASES, reason="no mllib regression outputs reachable")
@pytest.mark.parametrize("wrapper,case", CASES)
def test_aiesim_qdq(wrapper, case):
    t = json.load(open(os.path.join(case, "testcase.json")))
    got, want = runner(wrapper)(case, t)
    got = np.asarray(got).reshape(-1)
    want = np.ascontiguousarray(want).reshape(-1).astype(got.dtype)
    assert got.size == want.size, f"size {got.size} != {want.size}"
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} elements differ from aiesim, first at "
                           f"{bad[:8].tolist()}: got {got[bad[:8]].tolist()} "
                           f"want {want[bad[:8]].tolist()}")
