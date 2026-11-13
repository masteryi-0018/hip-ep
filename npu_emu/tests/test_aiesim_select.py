# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Select-group entry points against the NPU: each mllib regression case's
inputs are converted from the NPU's physical layout (C64-tiled HCgWC64 for the
pooling / reduction kernels, padded HWC for the broadcasting ones) to the
logical tensors, run through npu_emu, and compared byte for byte with the
aiesim output (data/ofm32.txt) over the valid region. The numpy reference in
golden_select.py is checked against the same output.

No aiesim output was kept for abs_fp8 and maxpool_bf16_c32; those are covered
by test_golden_select.py only."""

import json
import os

import numpy as np
import pytest

import golden as g
import golden_select as gs
from npu import (I64, OUT, NPU_BF16, NPU_F16, NPU_F32, NPU_I8, NPU_I16, NPU_I32,
                 NPU_I64, NPU_U8, NPU_U16, NPU_U32, call)
from aiesim_common import assert_bytes, cases, data, tensor_descr
from npu import OUT2
from test_golden_select import call2

INT = {"int8": (NPU_I8, np.int8), "uint8": (NPU_U8, np.uint8), "int16": (NPU_I16, np.int16),
       "uint16": (NPU_U16, np.uint16), "int32": (NPU_I32, np.int32),
       "uint32": (NPU_U32, np.uint32)}
HALF = {"bfloat16": (NPU_BF16, gs.BF16), "float16": (NPU_F16, gs.F16)}
IDX = {"uint8": (NPU_U8, np.uint8, "u8"), "uint16": (NPU_U16, np.uint16, "u16"),
       "uint32": (NPU_U32, np.uint32, "u32"), "int64": (NPU_I64, np.int64, "i64"),
       "float": (NPU_F32, np.float32, "f32")}


def tc(case):
    return json.load(open(os.path.join(case, "testcase.json")))


def go(name, *args, dtype, n):
    return call(name, *args, out_nbytes=n * np.dtype(dtype).itemsize, out_dtype=dtype)


def check(got, want, ref=None):
    assert_bytes(got, want)
    if ref is not None:
        assert_bytes(np.asarray(ref).astype(np.asarray(got).dtype), want)


# ---- elementwise unary ----------------------------------------------------------

def run_unary(case, w):
    t = tc(case)
    if w in ("abs_bf16", "neg_bf16", "clip_bf16"):
        nt, ht = HALF[t["ifm_dtype"]]
        x = data(case, "ifm32.txt", np.uint16)
        want = data(case, "ofm32.txt", np.uint16)[:len(x)]
        if w == "clip_bf16":
            lo, hi = (float(v) for v in data(case, "wts32.txt", np.float32)[:2])
            got = go("npu_elementwise_clip_bf16", x, nt, OUT, I64(len(x)), lo, hi,
                     dtype=np.uint16, n=len(x))
            ref = gs.clip_half(x, ht, lo, hi)
        else:
            got = go(f"npu_elementwise_{w}", x, nt, OUT, I64(len(x)), dtype=np.uint16, n=len(x))
            ref = gs.abs_half(x) if w == "abs_bf16" else gs.neg_half(x)
        return check(got, want, ref)
    x = data(case, "ifm32.txt")
    n = tensor_descr(case)[0][0] if tensor_descr(case) else len(x)
    want = data(case, "ofm32.txt")[:n]
    if w == "clip_fp8":
        s, lo, hi = (float(v) for v in data(case, "wts32.txt", np.float32)[:3])
        got = go("npu_elementwise_clip_fp8", x, OUT, I64(len(x)), s, lo, hi, dtype=np.uint8,
                 n=len(x))
        ref = gs.fp8_unary(x, "clip", s, lo, hi)
    else:
        s = float(data(case, "coeff32.txt", np.float32)[0])
        got = go(f"npu_elementwise_{w}", x, OUT, I64(len(x)), s, dtype=np.uint8, n=len(x))
        ref = gs.fp8_unary(x, w.split("_")[0], s)
    return check(got[:n], want, ref[:n])


# ---- 3D broadcasting ------------------------------------------------------------

def bcast_inputs(case, dt):
    """Valid logical operands (a, a_bcast, b, b_bcast, swapped) and the output
    shape. The NPU always puts the full tensor on ifm1."""
    t = tc(case)
    dims = ("height", "width", "channel")
    it = [t[f"ifm1_{d}_iteration"] for d in dims]
    p1 = [t[f"ifm1_physical_sv_{d}"] * i for d, i in zip(dims, it)]
    v1 = [t[f"ifm1_valid_full_{d}"] for d in dims]
    v2 = [t[f"ifm2_valid_full_{d}"] for d in dims]
    p2 = [t[f"ifm2_physical_sv_{d}"] * (i if v != 1 else 1) for d, i, v in zip(dims, it, v2)]
    x1 = data(case, "ifm32.txt", dt)[:np.prod(p1)].reshape(p1)
    x2 = data(case, "wts32.txt", dt)[:np.prod(p2)].reshape(p2)
    bc2 = sum(bit for bit, v, u in zip((4, 2, 1), v2, v1) if v == 1 and u != 1)
    s2 = [1 if bit & bc2 else v for bit, v in zip((4, 2, 1), v1)]
    a = np.ascontiguousarray(x1[:v1[0], :v1[1], :v1[2]])
    b = np.ascontiguousarray(x2[:s2[0], :s2[1], :s2[2]])
    return a, 0, b, bc2, bool(t["swap_inputs"]), v1


def run_bcast(case, w):
    op, ty = w.rsplit("_", 1)
    fp8 = ty == "fp8"
    t = tc(case)
    a, abc, b, bbc, swap, (H, W, C) = bcast_inputs(case, np.uint8 if fp8 else np.uint16)
    out_dt = np.int8 if op in ("equal", "greater", "greater_or_equal") else (
        np.uint8 if fp8 else np.uint16)
    phys = tensor_descr(case)[1]
    want = data(case, "ofm32.txt", out_dt)[:np.prod(phys)].reshape(phys)[:H, :W, :C]
    name = f"npu_elementwise_broadcasting3d_{w}"
    if fp8:
        co = data(case, "coeff32.txt", np.float32)
        sa, sb = float(co[0]), float(co[1])
        if swap:
            a, abc, sa, b, bbc, sb = b, bbc, sb, a, abc, sa
        got = go(name, a, abc, sa, b, bbc, sb, OUT, I64(H), I64(W), I64(C), dtype=out_dt,
                 n=H * W * C)
        ref = gs.bcast_fp8(op, a, abc, sa, b, bbc, sb, H, W, C)
    else:
        nt, ht = HALF[t.get("ifm_dtype", "bfloat16")]
        if swap:
            a, abc, b, bbc = b, bbc, a, abc
        got = go(name, a, abc, b, bbc, nt, OUT, I64(H), I64(W), I64(C), dtype=out_dt,
                 n=H * W * C)
        ref = gs.bcast_half(op, a, abc, b, bbc, ht, H, W, C)
    return check(got, want, ref)


# ---- arg max / arg min ----------------------------------------------------------

def hcgwc64_to_hwc(flat, H, W, C):
    return flat[:H * W * C].reshape(H, C // 64, W, 64).transpose(0, 2, 1, 3).reshape(H, W, C)


ARG = {  # wrapper: (kind, is_max, index bits: None = 16, 'out' = by out type)
    "arg_max_c64_a8": ("int", True, None), "arg_max_c64_a16": ("int", True, "out"),
    "arg_max_c64_a16_int64": ("int", True, None), "arg_max_c64_a32": ("int", True, None),
    "arg_max_c64_bf16": ("half", True, "out"), "arg_max_c64_fp8": ("fp8", True, None),
    "arg_min_c64_a8": ("int", False, "out"), "arg_min_c64_a16": ("int", False, "out"),
    "arg_min_c64_bf16": ("half", False, "out"),
}


def run_arg(case, w):
    kind, is_max, bits = ARG[w]
    t = tc(case)
    H = t["ifm_sv_height"] * t["ifm_height_iteration"]
    W = t["ifm_sv_width"] * t["ifm_width_iteration"]
    C = t["ifm_sv_depth"] * t["ifm_depth_iteration"]
    axis = {1: 2, 2: 1, 4: 0}[t["axis_flags"]]
    nt_out, odt, okey = IDX[t.get("dtype_out", "int64")]
    n = H * W * C // (H, W, C)[axis]
    if kind == "fp8":
        x = hcgwc64_to_hwc(data(case, "ifm32.txt"), H, W, C)
        vals, init = g.fp8_to_f32(x).astype(np.float64), -448.0
        args = (x, OUT, nt_out)
    elif kind == "half":
        nt, ht = HALF[t["dtype_in"]]
        x = hcgwc64_to_hwc(data(case, "ifm32.txt", np.uint16), H, W, C)
        vals = gs.half_to_f64(x, ht)
        init = gs.half_to_f64(np.uint16((gs.MIN_NORMAL if is_max else gs.MAX_FINITE)[ht]), ht)
        args = (x, nt, OUT, nt_out)
    else:
        nt, dt = INT[t["dtype_in"]]
        x = hcgwc64_to_hwc(data(case, "ifm32.txt", dt), H, W, C)
        vals = x.astype(np.float64)
        info = np.iinfo(dt)
        init = float(info.min if is_max else info.max)
        args = (x, nt, OUT, nt_out)
    if w == "arg_max_c64_a16_int64":
        args = args[:2] + (OUT,)
    nb = 16 if bits is None else (8 if okey == "u8" else 16)
    got = go(f"npu_{w}", *args, I64(H), I64(W), I64(C), axis, dtype=odt, n=n)
    want = data(case, "ofm32.txt", odt)[:n]
    if axis == 0:  # the NPU writes (C/64, W, 64)
        want = want.reshape(C // 64, W, 64).transpose(1, 0, 2).reshape(-1)
    ref = gs.index_out(gs.arg_reduce(vals, axis, is_max, init, nb), okey).reshape(-1)
    return check(got, want, ref)


# ---- reduce ---------------------------------------------------------------------

def run_reduce(case, w):
    t = tc(case)
    nt, ht = HALF[t["dtype"]]
    H = t["ifm_sv_height"] * t["ifm_height_iteration"]
    W = t["ifm_sv_width"] * t["ifm_width_iteration"]
    C = t["ifm_sv_depth"] * t["ifm_depth_iteration"]
    Cr = t.get("ifm_sv_depth_real", t["ifm_sv_depth"]) * t["ifm_depth_iteration"]
    x = np.ascontiguousarray(hcgwc64_to_hwc(data(case, "ifm32.txt", np.uint16), H, W, C)[:, :, :Cr])
    a = t["axis_flags"]
    shape = tuple(s for s, bit in ((H, 4), (W, 2), (Cr, 1)) if not a & bit)
    n = int(np.prod(shape))
    y = data(case, "ofm32.txt", np.uint16)
    if a == 2:      # (H, C) with the padded channels
        want = y[:H * C].reshape(H, C)[:, :Cr]
    elif a == 4:    # (C/64, W, 64)
        want = y[:W * C].reshape(C // 64, W, 64).transpose(1, 0, 2).reshape(W, C)[:, :Cr]
    elif a == 6:
        want = y[:C][:Cr]
    else:
        want = y[:n]
    got = go(f"npu_{w}", x, nt, OUT, I64(H), I64(W), I64(Cr), a, dtype=np.uint16, n=n)
    ref = gs.reduce_half(x, ht, a, w.startswith("reduce_max"))
    return check(got, np.ascontiguousarray(want).reshape(-1), ref.reshape(-1))


# ---- pooling --------------------------------------------------------------------

def run_maxpool(case, w):
    t = tc(case)
    H = t["ofm_sv_height"] * t["ofm_height_iteration"]
    W = t["ofm_sv_width"] * t["ofm_width_iteration"]
    C = t["ofm_sv_depth"] * t["ifm_depth_iteration"]
    kh, kw, sh, sw = t["kernel_height"], t["kernel_width"], t["stride_height"], t["stride_width"]
    Hi, Wi = (H - 1) * sh + kh, (W - 1) * sw + kw
    gc = 64 // t.get("granX", 1)

    def unpack(flat, h, w_):
        return flat[:h * w_ * C].reshape(h, C // gc, w_, gc).transpose(0, 2, 1, 3).reshape(h, w_, C)

    if w == "maxpool_bf16":
        nt, ht = HALF[t["ifm_dtype"]]
        x = np.ascontiguousarray(unpack(data(case, "ifm32.txt", np.uint16), Hi, Wi))
        got = go("npu_maxpool_bf16", x, nt, OUT, I64(Hi), I64(Wi), I64(C), kh, kw, sh, sw,
                 dtype=np.uint16, n=H * W * C)
        want = unpack(data(case, "ofm32.txt", np.uint16), H, W)
        return check(got, want.reshape(-1), gs.maxpool_half(x, ht, kh, kw, sh, sw).reshape(-1))
    x = np.ascontiguousarray(unpack(data(case, "ifm32.txt"), Hi, Wi))
    if t["vector_coeff"]:
        wt = data(case, "wts32.txt", np.float32).reshape(-1, 128)
        ch = np.arange(C)
        sv = t["ofm_sv_depth"]
        blk = (ch // sv) * (sv // gc) + (ch % sv) // gc
        c0 = np.ascontiguousarray(wt[blk, ch % gc])
        c2 = np.ascontiguousarray(wt[blk, 64 + ch % gc])
    else:
        c0 = np.array([t["quant_c0"]], np.float32)
        c2 = np.array([t["quant_c2"]], np.float32)
    clip = int(t["activation_type_templates"] == 100)
    lo, hi = float(np.float32(t["act_lo"])), float(np.float32(t["act_hi"]))
    got = go("npu_maxpool_fp8", x, OUT, I64(Hi), I64(Wi), I64(C), kh, kw, sh, sw, c0, c2,
             I64(len(c0)), clip, lo, hi, dtype=np.uint8, n=H * W * C)
    want = unpack(data(case, "ofm32.txt"), H, W)
    ref = gs.maxpool_fp8(x, kh, kw, sh, sw, c0, c2, clip, lo, hi)
    return check(got, want.reshape(-1), ref.reshape(-1))


def run_globalmaxpool(case, w):
    t = tc(case)
    H = t["ifm_sv_height"] * t["ifm_height_iteration"]
    W = t["ifm_sv_width"] * t["ifm_width_iteration"]
    C = t["ofm_sv_depth"] * t["ifm_depth_iteration"]
    if w == "globalmaxpool_bf16":
        nt, ht = HALF[t["ifm_dtype"]]
        x = np.ascontiguousarray(hcgwc64_to_hwc(data(case, "ifm32.txt", np.uint16), H, W, C))
        got = go("npu_globalmaxpool_bf16", x, nt, OUT, I64(H), I64(W), I64(C), dtype=np.uint16,
                 n=C)
        return check(got, data(case, "ofm32.txt", np.uint16)[:C], gs.globalmaxpool_half(x, ht))
    x = np.ascontiguousarray(hcgwc64_to_hwc(data(case, "ifm32.txt"), H, W, C))
    co = data(case, "coeff32.txt", np.float32)
    c0, c1 = float(co[0]), float(co[1])
    got = go("npu_globalmaxpool_fp8", x, OUT, I64(H), I64(W), I64(C), c0, c1, dtype=np.uint8,
             n=C)
    return check(got, data(case, "ofm32.txt")[:C], gs.globalmaxpool_fp8(x, c0, c1))


# ---- top-k ----------------------------------------------------------------------

def run_topk(case, w):
    t = tc(case)
    din = t["dtype_in"]
    K = t.get("K_len_even", t["K_len"])   # the integer kernels run K_len_even rounds
    Hs, it = t["ifm_sv_height"], t["ifm_height_iteration"]
    Wp, Wr, largest = t["ifm_sv_width"], t["ifm_sv_width_real"], t["largest"]
    nti, idt, _ = IDX[t["dtype_index_out"]]
    if din in HALF:
        nt, ht = HALF[din]
        dt = np.uint16
        x = data(case, "ifm32.txt", dt)[:Hs * it * Wp].reshape(-1, Wp)[:, :Wr]
        vals = gs.half_to_f64(x, ht)
        sent = -np.inf if largest else np.inf
        sent_raw = (0xFF80 if largest else 0x7F80) if ht == gs.BF16 else (0xFC00 if largest else 0x7C00)
    else:
        nt, dt = INT[din]
        x = data(case, "ifm32.txt", dt)[:Hs * it * Wp].reshape(-1, Wp)[:, :Wr]
        vals = x.astype(np.float64)
        info = np.iinfo(dt)
        sent = float(info.min if largest else info.max)
        sent_raw = int(sent)
    x = np.ascontiguousarray(x)
    rows = Hs * it
    gv, gi = call2(f"npu_{w}", x, nt, OUT, OUT2, nti, I64(rows), I64(Wr), I64(K), largest,
                   outs=((rows * K, dt), (rows * K, idt)))
    y = data(case, "ofm32.txt")
    chunk = len(y) // it
    off = t["ofm_index_offset"]
    vb, ib = rows // it * K * np.dtype(dt).itemsize, rows // it * K * np.dtype(idt).itemsize
    want_v = np.concatenate([y[j * chunk:j * chunk + vb] for j in range(it)])
    want_i = np.concatenate([y[j * chunk + off:j * chunk + off + ib] for j in range(it)])
    rv, ri = gs.topk(vals, x, Wr, K, largest, sent, np.array(sent_raw).astype(dt),
                     8 if idt == np.uint8 else 16)
    check(gv, want_v, rv.reshape(-1))
    check(gi, want_i, ri.astype(idt).reshape(-1))


# ---- registry -------------------------------------------------------------------

# src wrapper -> (regression dir, runner)
WRAPPERS = {
    "elementwise_abs_bf16": ("abs_bf16_wrapper", run_unary),
    "elementwise_neg_bf16": ("neg_bf16_wrapper", run_unary),
    "elementwise_clip_bf16": ("clip_bf16_wrapper", run_unary),
    "elementwise_abs_fp8": ("abs_fp8_wrapper", run_unary),
    "elementwise_neg_fp8": ("neg_fp8_wrapper", run_unary),
    "elementwise_clip_fp8": ("clip_fp8_wrapper", run_unary),
    "maxpool_bf16": ("maxpool_bf16_wrapper", run_maxpool),
    "maxpool_bf16_c32": ("maxpool_bf16_c32_wrapper", run_maxpool),
    "maxpool_fp8": ("maxpool_fp8_wrapper", run_maxpool),
    "globalmaxpool_bf16": ("globalmaxpool_bf16_wrapper", run_globalmaxpool),
    "globalmaxpool_fp8": ("globalmaxpool_fp8_wrapper", run_globalmaxpool),
    "reduce_max_c64_bf16": ("reduce_max_c64_bf16_wrapper", run_reduce),
    "reduce_min_c64_bf16": ("reduce_min_c64_bf16_wrapper", run_reduce),
    "topk2d_a8": ("topk2d_a8_wrapper", run_topk),
    "topk2d_a16": ("topk2d_a16_wrapper", run_topk),
    "topk2d_bf16": ("topk2d_bf16_wrapper", run_topk),
}
WRAPPERS.update({w: (f"{w}_wrapper", run_arg) for w in ARG})
for _op in ("equal", "greater", "greater_or_equal", "max", "min"):
    for _ty in ("bf16", "fp8"):
        WRAPPERS[f"elementwise_broadcasting3d_{_op}_{_ty}"] = (
            f"elementwise_broadcasting3d_{_op}_{_ty}_wrapper", run_bcast)

_DIR = {d: w for w, (d, _) in WRAPPERS.items()}
CASES = cases(sorted(_DIR))


@pytest.mark.skipif(not CASES, reason="no mllib regression outputs reachable")
@pytest.mark.parametrize("wdir,case", CASES)
def test_aiesim_select(wdir, case):
    w = _DIR[wdir]
    runner = WRAPPERS[w][1]
    key = w.removeprefix("elementwise_").removeprefix("broadcasting3d_") \
        if runner in (run_unary, run_bcast) else w
    runner(case, key)
