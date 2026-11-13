# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""arith_float entry points against the NPU: each mllib regression case's
inputs are converted from the NPU's physical layout (padded HWC sub-volume
grids, bfp16 blocks, the batchnorm YCXNC stream) to the logical tensors, run
through npu_emu with the coefficient words of the case (testcase RTP values,
coeff32.txt, wts32.txt, kernel_rtp.h), and compared byte for byte with the
aiesim output (data/ofm32.txt) over the valid region. The numpy reference in
golden_arith_float.py is checked against the same output."""

import json
import os
import re

import numpy as np
import pytest

import golden_arith_float as ga
from npu import I64, OUT, NPU_BF16, NPU_F16, NPU_FP8, call
from aiesim_common import assert_bytes, cases, data

HALF = {"bfloat16": (NPU_BF16, ga.BF16), "float16": (NPU_F16, ga.F16)}
DIMS = ("height", "width", "channel")


def tc(case):
    return json.load(open(os.path.join(case, "testcase.json")))


def rtp_words(case):
    txt = open(os.path.join(case, "kernel_rtp.h")).read()
    body = re.search(r"generated_rtps\[\d+\]\s*=\s*\{([^}]*)\}", txt).group(1)
    return np.array([int(w, 16) for w in body.replace(",", " ").split()], np.uint32)


def floats(case, name):
    return data(case, name, np.float32)


def ceil8(c):
    return (c + 7) // 8 * 8


class Op:
    """One aiesim case: entry name, its arguments, the output size, the aiesim
    bytes and the numpy reference, both over the compared region."""

    def __init__(self, name, args, nbytes, dtype, want, ref, view=None):
        self.name, self.args, self.nbytes, self.dtype = name, args, nbytes, dtype
        self.want, self.ref = want, ref
        self.view = view or (lambda x: x)


# ---- physical layouts -------------------------------------------------------------

def phys(t, k):
    """Physical shape of operand k (1 = ifm1): broadcast dims keep one sub-volume."""
    it = [t[f"ifm1_{d}_iteration"] for d in DIMS]
    if k == 1:
        return [t[f"ifm1_physical_sv_{d}"] * i for d, i in zip(DIMS, it)]
    return [t[f"ifm{k}_physical_sv_{d}"] * (i if t[f"ifm{k}_valid_full_{d}"] != 1 else 1)
            for d, i in zip(DIMS, it)]


def valid(t, k):
    return [t[f"ifm{k}_valid_full_{d}"] for d in DIMS]


def bcast_bits(t, k):
    v1, vk = valid(t, 1), valid(t, k)
    return sum(bit for bit, a, b in zip((4, 2, 1), v1, vk) if b == 1 and a != 1)


def elems(case, fname, t, k, dt, c_keep=None):
    """Operand k as its logical (valid / broadcast) array of elements."""
    p = phys(t, k)
    x = data(case, fname, dt)[:np.prod(p)].reshape(p)
    v1 = valid(t, 1)
    bc = bcast_bits(t, 1) if k == 1 else bcast_bits(t, k)
    keep = [1 if bc & bit else v for bit, v in zip((4, 2, 1), v1)]
    if c_keep is not None and not bc & 1:
        keep[2] = c_keep
    return np.ascontiguousarray(x[:keep[0], :keep[1], :keep[2]]), bc


def blocks(case, fname, t, k, cv8):
    """bfp16 operand k: (h, w, blocks, 9) bytes; C-broadcast keeps one block."""
    p = phys(t, k)
    x = data(case, fname)[:p[0] * p[1] * p[2] // 8 * 9].reshape(p[0], p[1], p[2] // 8, 9)
    v1 = valid(t, 1)
    bc = 0 if k == 1 else bcast_bits(t, k)
    hk = 1 if bc & 4 else v1[0]
    wk = 1 if bc & 2 else v1[1]
    nb = 1 if bc & 1 else cv8 // 8
    return np.ascontiguousarray(x[:hk, :wk, :nb]).reshape(-1), bc


def out_view(case, t, dt):
    p = phys(t, 1)
    H, W, C = valid(t, 1)
    return data(case, "ofm32.txt", dt)[:np.prod(p)].reshape(p)[:H, :W, :C].reshape(-1)


# ---- runners ----------------------------------------------------------------------

def run_bin16(case, w):
    t = tc(case)
    nt, ht = HALF[t.get("ifm_dtype", "bfloat16")]
    H, W, C = valid(t, 1)
    x1, _ = elems(case, "ifm32.txt", t, 1, np.uint16)
    x2, bc = elems(case, "wts32.txt", t, 2, np.uint16)
    want = out_view(case, t, np.uint16)
    act = int(t.get("activation_type", 0))
    a0, a1 = float(np.float32(t.get("activation_coeff0", 0.0))), float(
        np.float32(t.get("activation_coeff1", 0.0)))
    dims = (I64(H), I64(W), I64(C))
    if w == "add_bf16":
        n1, n2 = int(t.get("negate_flag", 0)), int(t.get("is_sub_flag", 0))
        args = (x1, x2, bc, nt, OUT) + dims + (n1, n2, act, a0, a1)
        ref = ga.add16(x1, x2, bc, ht, H, W, C, n1, n2, act, a0, a1)
    elif w == "mul_bf16":
        args = (x1, x2, bc, nt, OUT) + dims + (act, a0, a1)
        ref = ga.mul16(x1, x2, bc, ht, H, W, C, act, a0, a1)
    else:
        sw = int(t["swap_inputs"])
        args = (x1, x2, bc, nt, OUT) + dims + (sw,)
        ref = ga.div16(x1, x2, bc, ht, H, W, C, sw)
    return Op(f"npu_elementwise_broadcasting3d_{w}", args, H * W * C * 2, np.uint16, want, ref)


def run_mac16(case, w):
    t = tc(case)
    nt, ht = HALF[t.get("ifm_dtype", "bfloat16")]
    H, W, C = valid(t, 1)
    x1, _ = elems(case, "ifm32.txt", t, 1, np.uint16)
    x2, bc = elems(case, "ifm2_32.txt", t, 2, np.uint16)
    x3, bc3 = elems(case, "wts32.txt", t, 3, np.uint16)
    assert bc == bc3
    act = int(t.get("activation_type", 0))
    a0, a1 = (float(np.float32(t.get(k, 0.0))) for k in ("activation_coeff0", "activation_coeff1"))
    args = (x1, x2, x3, bc, nt, OUT, I64(H), I64(W), I64(C), act, a0, a1)
    ref = ga.mac16(x1, x2, x3, bc, ht, H, W, C, act, a0, a1)
    return Op("npu_elementwise_broadcasting3d_mac_fused_bf16", args, H * W * C * 2, np.uint16,
              out_view(case, t, np.uint16), ref)


def run_bfp(case, w):
    t = tc(case)
    H, W, C = valid(t, 1)
    C8 = ceil8(C)
    x1, _ = blocks(case, "ifm32.txt", t, 1, C8)
    x2, bc = blocks(case, "wts32.txt", t, 2, C8)
    op = w.split("_")[0]
    sw = int(t.get("swap_inputs", 0))
    extra = (sw,) if op in ("sub", "div") else ()
    args = (x1, x2, bc, OUT, I64(H), I64(W), I64(C8)) + extra
    ref = ga.bfp_bin(op, x1, x2, bc, H, W, C8, sw)
    want, _ = blocks(case, "ofm32.txt", t, 1, C8)
    return Op(f"npu_elementwise_broadcasting3d_{w}", args, H * W * C8 // 8 * 9, np.uint8,
              want, ref)


def run_mul_add(case, w):
    t = tc(case)
    H, W, C = valid(t, 1)
    C8 = ceil8(C)
    x1, _ = blocks(case, "ifm32.txt", t, 1, C8)
    ops = []
    for k, f in ((2, "ifm2_32.txt"), (3, "wts32.txt")):
        if t[f"ifm{k}_dtype"] == "bfp16":
            x, bc = blocks(case, f, t, k, C8)
            ops.append((x, 1, bc))
        else:
            x, bc = elems(case, f, t, k, np.uint16, c_keep=C8)
            ops.append((x, 0, bc))
    (x2, b2, bc2), (x3, b3, bc3) = ops
    out_bfp = int(t["ofm_dtype"] == "bfp16")
    act_type, activation = int(t["activation_type"]), int(t["activation"])
    rw = rtp_words(case)
    # clip_min / clip_max: the bf16 RTP halves at byte 52 / 54.
    lo, hi = (float(np.array([(int(rw[13]) >> sh & 0xFFFF) << 16], np.uint32).view(np.float32)[0])
              for sh in (0, 16))
    args = (x1, x2, b2, bc2, x3, b3, bc3, OUT, out_bfp, I64(H), I64(W), I64(C8), act_type,
            activation, lo, hi)
    ref = ga.mul_add(x1, x2, b2, bc2, x3, b3, bc3, out_bfp, H, W, C8, act_type, activation,
                     lo, hi)
    if out_bfp:
        want, _ = blocks(case, "ofm32.txt", t, 1, C8)
        return Op("npu_elementwise_broadcasting3d_mul_add_bfp16", args, H * W * C8 // 8 * 9,
                  np.uint8, want, ref)
    p = phys(t, 1)
    want = data(case, "ofm32.txt", np.uint16)[:np.prod(p)].reshape(p)[:H, :W, :C].reshape(-1)

    def view(x):
        return np.asarray(x).reshape(H, W, C8)[:, :, :C].reshape(-1)
    return Op("npu_elementwise_broadcasting3d_mul_add_bfp16", args, H * W * C8 * 2, np.uint16,
              want, ref, view)


def run_bin_fp8(case, w):
    t = tc(case)
    H, W, C = valid(t, 1)
    x1, _ = elems(case, "ifm32.txt", t, 1, np.uint8)
    x2, bc = elems(case, "wts32.txt", t, 2, np.uint8)
    co = floats(case, "coeff32.txt")
    dims = (I64(H), I64(W), I64(C))
    if w in ("add_fp8",):
        tmpl = int(t.get("activation_type_templates", t.get("activation_type", 0)))
        act = ga.CLIP if tmpl == 100 else tmpl
        c1, c2 = float(co[0]), float(co[1])
        a0, a1 = (float(co[2]), float(co[3])) if len(co) > 3 else (0.0, 0.0)
        table = np.ascontiguousarray(co[4:262]) if len(co) >= 262 else None
        args = (x1, x2, bc, OUT) + dims + (c1, c2, act, a0, a1, table)
        ref = ga.add_fp8(x1, x2, bc, H, W, C, c1, c2, act, a0, a1, table)
    elif w == "mul_fp8":
        args = (x1, x2, bc, OUT) + dims + (float(co[0]),)
        ref = ga.mul_fp8(x1, x2, bc, H, W, C, float(co[0]))
    else:
        sw = int(t["swap_inputs"])
        args = (x1, x2, bc, OUT) + dims + (float(co[0]), sw)
        ref = ga.div_fp8(x1, x2, bc, H, W, C, float(co[0]), sw)
    return Op(f"npu_elementwise_broadcasting3d_{w}", args, H * W * C, np.uint8,
              out_view(case, t, np.uint8), ref)


def run_mac_fp8(case, w):
    t = tc(case)
    H, W, C = valid(t, 1)
    x1, _ = elems(case, "ifm32.txt", t, 1, np.uint8)
    x2, bc = elems(case, "wts32.txt", t, 2, np.uint8)
    x3, bc3 = elems(case, "ifm2_32.txt", t, 3, np.uint8)
    assert bc == bc3
    co = floats(case, "coeff32.txt")
    act = int(t.get("activation_type", 0))
    a0, a1 = 0.0, 0.0
    if act in (ga.RELU, ga.CLIP):
        a0 = float(co[5])
        a1 = float(co[6]) if act == ga.CLIP else 0.0
    elif act == ga.LRELU:
        a0 = float(co[96])
    c0, c4 = float(co[0]), float(co[4])
    args = (x1, x2, x3, bc, OUT, I64(H), I64(W), I64(C), c0, c4, act, a0, a1)
    ref = ga.mac_fp8(x1, x2, x3, bc, H, W, C, c0, c4, act, a0, a1)
    return Op("npu_elementwise_broadcasting3d_mac_fused_fp8", args, H * W * C, np.uint8,
              out_view(case, t, np.uint8), ref)


def flat_n(t):
    return t["ifm_sv_size"] * t.get("ifm_sv_iteration", 1)


def run_unary(case, w):
    t = tc(case)
    n = flat_n(t)
    if w.endswith("fp8"):
        x = data(case, "ifm32.txt")[:n]
        want = data(case, "ofm32.txt")[:n]
        if w == "leakyrelu_fp8":
            pos, neg = (float(v) for v in floats(case, "wts32.txt")[:2])
            args, ref = (x, OUT, I64(n), pos, neg), ga.leakyrelu_fp8(x, pos, neg)
        elif w == "square_fp8":
            s = float(floats(case, "coeff32.txt")[0])
            args, ref = (x, OUT, I64(n), s), ga.square_fp8(x, s)
        else:
            C = int(t["ifm_sv_channel"])
            wt = floats(case, "wts32.txt")
            P, D = float(wt[0]), np.ascontiguousarray(wt[32:32 + C])
            args, ref = (x, OUT, I64(n), P, D, I64(C)), ga.prelu_fp8(x, P, D, C)
        return Op(f"npu_elementwise_{w}", args, n, np.uint8, want, ref)
    nt, ht = HALF[t["ifm_dtype"]]
    x = data(case, "ifm32.txt", np.uint16)[:n]
    want = data(case, "ofm32.txt", np.uint16)[:n]
    if w == "leakyrelu_bf16":
        a = float(floats(case, "wts32.txt")[0])
        args, ref = (x, nt, OUT, I64(n), a), ga.leakyrelu16(x, ht, a)
    elif w == "square_bf16":
        args, ref = (x, nt, OUT, I64(n)), ga.square16(x, ht)
    else:
        C = int(t["ifm_sv_channel"])
        D = np.ascontiguousarray(floats(case, "wts32.txt")[:C])
        args, ref = (x, nt, OUT, I64(n), D, I64(C)), ga.prelu16(x, ht, D, C)
    return Op(f"npu_elementwise_{w}", args, n * 2, np.uint16, want, ref)


def run_batchnorm(case, w):
    t = tc(case)
    dt = t["dtype"]
    N, C, cs = t["ifm_sv_batch"], t["ifm_sv_depth"], int(t["c_stride"])
    H = t["ifm_sv_height"] * t["ifm_height_iteration"]
    Wd = t["ifm_sv_width"] * t["ifm_width_iteration"]
    n = N * H * Wd * C
    fp8 = dt == "float8"
    sdt = np.uint8 if fp8 else np.uint16

    def to_logical(s):  # stream [H][C/cs][W][N][cs] -> [H][W][N][C]
        return np.ascontiguousarray(
            s[:n].reshape(H, C // cs, Wd, N, cs).transpose(0, 2, 3, 1, 4)).reshape(-1)

    x = to_logical(data(case, "ifm32.txt", sdt))
    want = to_logical(data(case, "ofm32.txt", sdt))
    wts = data(case, "wts32.txt")
    npk = C * 32 // cs
    wt16 = wts[:4 * npk].view(np.uint16)
    wdt = "f16" if dt in ("float16", "float8") else "bf16"
    unpack = lambda v: np.ascontiguousarray(  # noqa: E731
        ga.load16(v.reshape(C // cs, 32 // cs, cs)[:, 0, :].reshape(-1), wdt))
    cw, cb = unpack(wt16[:npk]), unpack(wt16[npk:2 * npk])
    alpha = float(wts[4 * npk:4 * npk + 4].view(np.float32)[0])
    act = int(t["activation_type"])
    nt = {"bfloat16": NPU_BF16, "float16": NPU_F16, "float8": NPU_FP8}[dt]
    gt = "fp8" if fp8 else ("f16" if dt == "float16" else "bf16")
    args = (x, nt, OUT, I64(n), I64(C), cw, cb, act, alpha)
    ref = ga.batchnorm(x, gt, C, cw, cb, act, alpha)
    return Op("npu_batchnorm2d_stream_bf16", args, n * np.dtype(sdt).itemsize, sdt, want, ref)


# ---- registry -------------------------------------------------------------------------

# regression dir -> (entry key, runner)
DIRS = {
    "elementwise_broadcasting3d_add_bf16_wrapper": ("add_bf16", run_bin16),
    "elementwise_broadcasting3d_sub_bf16_wrapper": ("add_bf16", run_bin16),
    "elementwise_broadcasting3d_mul_bf16_wrapper": ("mul_bf16", run_bin16),
    "elementwise_broadcasting3d_div_bf16_wrapper": ("div_bf16", run_bin16),
    "elementwise_broadcasting3d_mac_fused_bf16_wrapper": ("mac_fused_bf16", run_mac16),
    "elementwise_broadcasting3d_add_bfp16_wrapper": ("add_bfp16", run_bfp),
    "elementwise_broadcasting3d_sub_bfp16_wrapper": ("sub_bfp16", run_bfp),
    "elementwise_broadcasting3d_mul_bfp16_wrapper": ("mul_bfp16", run_bfp),
    "elementwise_broadcasting3d_div_bfp16_wrapper": ("div_bfp16", run_bfp),
    "elementwise_broadcasting3d_mul_add_bfp16_wrapper": ("mul_add_bfp16", run_mul_add),
    "elementwise_broadcasting3d_add_fp8_wrapper": ("add_fp8", run_bin_fp8),
    "elementwise_broadcasting3d_sub_fp8_wrapper": ("add_fp8", run_bin_fp8),
    "elementwise_broadcasting3d_mul_fp8_wrapper": ("mul_fp8", run_bin_fp8),
    "elementwise_broadcasting3d_div_fp8_wrapper": ("div_fp8", run_bin_fp8),
    "elementwise_broadcasting3d_mac_fused_fp8_wrapper": ("mac_fused_fp8", run_mac_fp8),
    "leakyrelu_bf16_wrapper": ("leakyrelu_bf16", run_unary),
    "leakyrelu_fp8_wrapper": ("leakyrelu_fp8", run_unary),
    "prelu_bf16_wrapper": ("prelu_bf16", run_unary),
    "prelu_fp8_wrapper": ("prelu_fp8", run_unary),
    "square_bf16_wrapper": ("square_bf16", run_unary),
    "square_fp8_wrapper": ("square_fp8", run_unary),
    "batchnorm2d_stream_bf16_wrapper": ("batchnorm2d_stream_bf16", run_batchnorm),
}

CASES = cases(sorted(DIRS))


def prepare(wdir, case):
    key, runner = DIRS[wdir]
    return runner(case, key)


@pytest.mark.skipif(not CASES, reason="no mllib regression outputs reachable")
@pytest.mark.parametrize("wdir,case", CASES)
def test_aiesim_arith_float(wdir, case):
    op = prepare(wdir, case)
    got = call(op.name, *op.args, out_nbytes=op.nbytes, out_dtype=op.dtype)
    assert_bytes(op.view(got), op.want)
    assert_bytes(op.view(np.asarray(op.ref).astype(op.dtype)), op.want)
