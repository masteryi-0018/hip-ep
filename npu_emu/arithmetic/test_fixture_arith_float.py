# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Floating-point arithmetic entry points and the numpy model against NPU reference outputs."""

import numpy as np
import pytest

import golden_arith_float as ga
from npu import I64, OUT, NPU_BF16, NPU_BFP16, NPU_F16, NPU_FP8, call
from testing import shp
from fixture import HALF, assert_bytes, cases, data, ofm, rtp_words
from fixture import testcase as tc

DIMS = ("height", "width", "channel")


def floats(case, name):
    return data(case, name, np.float32)


def ceil8(c):
    return (c + 7) // 8 * 8


class Op:
    """One sim case: entry name, arguments, output size, sim bytes and the numpy reference."""

    def __init__(self, name, args, nbytes, dtype, want, ref, view=None):
        self.name, self.args, self.nbytes, self.dtype = name, args, nbytes, dtype
        self.want, self.ref = want, ref
        self.view = view or (lambda x: x)


def phys(t, k):
    """Physical shape of operand k (1 = ifm1): broadcast dims are not repeated."""
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
    return data(case, ofm(case), dt)[:np.prod(p)].reshape(p)[:H, :W, :C].reshape(-1)


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
    sh = ((H, W, C), ga._bshape(bc, H, W, C))
    if w == "add_bf16":
        params = (int(t.get("is_sub_flag", 0)), int(t.get("negate_flag", 0)), act, a0, a1)
    elif w == "mul_bf16":
        params = (act, a0, a1)
    else:
        params = (int(t["swap_inputs"]),)
    name = f"elementwise_broadcasting3d_{w}"
    args = (x1, nt, x2, OUT, *map(shp, sh), *params)
    ref = getattr(ga, name)(x1, ht, x2, *sh, *params)
    return Op(f"npu_{name}", args, H * W * C * 2, np.uint16, want, ref)


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
    sh = ((H, W, C), ga._bshape(bc, H, W, C))
    args = (x1, nt, x2, x3, OUT, *map(shp, sh), act, a0, a1)
    ref = ga.elementwise_broadcasting3d_mac_fused_bf16(x1, ht, x2, x3, *sh, act, a0, a1)
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
    sh = ((H, W, C8), ga._bshape(bc, H, W, C8))
    args = (x1, x2, OUT, *map(shp, sh)) + extra
    ref = getattr(ga, f"elementwise_broadcasting3d_{w}")(x1, x2, *sh, *extra)
    want, _ = blocks(case, ofm(case), t, 1, C8)
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
            ops.append((x, "bfp16", bc))
        else:
            x, bc = elems(case, f, t, k, np.uint16, c_keep=C8)
            ops.append((x, "bf16", bc))
    (x2, t2, bc2), (x3, t3, bc3) = ops
    out_bfp = int(t["ofm_dtype"] == "bfp16")
    to = "bfp16" if out_bfp else "bf16"
    act_type, activation = int(t["activation_type"]), int(t["activation"])
    rw = rtp_words(case)
    # clip_min / clip_max: the bf16 halves of word 13.
    lo, hi = (int(rw[13]) >> s & 0xFFFF for s in (0, 16))
    sh = ((H, W, C8), ga._bshape(bc2, H, W, C8), ga._bshape(bc3, H, W, C8))
    nty = {"bf16": NPU_BF16, "bfp16": NPU_BFP16}
    args = (x1, x2, nty[t2], x3, nty[t3], OUT, nty[to], *map(shp, sh), act_type, activation, lo,
            hi)
    ref = ga.elementwise_broadcasting3d_mul_add_bfp16(x1, x2, t2, x3, t3, to, *sh, act_type,
                                                      activation, lo, hi)
    if out_bfp:
        want, _ = blocks(case, ofm(case), t, 1, C8)
        return Op("npu_elementwise_broadcasting3d_mul_add_bfp16", args, H * W * C8 // 8 * 9,
                  np.uint8, want, ref)
    ref = ref.view(np.uint16)
    p = phys(t, 1)
    want = data(case, ofm(case), np.uint16)[:np.prod(p)].reshape(p)[:H, :W, :C].reshape(-1)

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
    sh = ((H, W, C), ga._bshape(bc, H, W, C))
    if w == "add_fp8":
        act = int(t.get("activation_type", 0))
        c1, c2 = float(co[0]), float(co[1])
        a0, a1 = (float(co[2]), float(co[3])) if len(co) > 3 else (0.0, 0.0)
        table = np.ascontiguousarray(co[4:262]) if len(co) >= 262 else None
        params = (c1, c2, act, a0, a1, table)
    elif w == "mul_fp8":
        params = (float(co[0]),)
    else:
        params = (float(co[0]), int(t["swap_inputs"]))
    name = f"elementwise_broadcasting3d_{w}"
    args = (x1, x2, OUT, *map(shp, sh), *params)
    ref = getattr(ga, name)(x1, x2, *sh, *params)
    return Op(f"npu_{name}", args, H * W * C, np.uint8, out_view(case, t, np.uint8), ref)


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
    if act in (ga.RELU, ga.RELU6):
        a0 = float(co[5])
        a1 = float(co[6]) if act == ga.RELU6 else 0.0
    elif act == ga.LRELU:
        a0 = float(co[96])
    c0, c4 = float(co[0]), float(co[4])
    sh = ((H, W, C), ga._bshape(bc, H, W, C))
    args = (x1, x2, x3, OUT, *map(shp, sh), c0, c4, act, a0, a1)
    ref = ga.elementwise_broadcasting3d_mac_fused_fp8(x1, x2, x3, *sh, c0, c4, act, a0, a1)
    return Op("npu_elementwise_broadcasting3d_mac_fused_fp8", args, H * W * C, np.uint8,
              out_view(case, t, np.uint8), ref)


def flat_n(t):
    return t["ifm_sv_size"] * t.get("ifm_sv_iteration", 1)


def run_unary(case, w):
    t = tc(case)
    n = flat_n(t)
    if w.endswith("fp8"):
        x = data(case, "ifm32.txt")[:n]
        want = data(case, ofm(case))[:n]
        if w == "leakyrelu_fp8":
            params = tuple(float(v) for v in floats(case, "wts32.txt")[:2])
        elif w == "square_fp8":
            params = (float(floats(case, "coeff32.txt")[0]),)
        else:
            C = int(t["ifm_sv_channel"])
            wt = floats(case, "wts32.txt")
            params = (I64(C), float(wt[0]), np.ascontiguousarray(wt[32:32 + C]))
        ref = getattr(ga, f"elementwise_{w}")(x, n, *params)
        return Op(f"npu_elementwise_{w}", (x, OUT, I64(n), *params), n, np.uint8, want, ref)
    nt, ht = HALF[t["ifm_dtype"]]
    x = data(case, "ifm32.txt", np.uint16)[:n]
    want = data(case, ofm(case), np.uint16)[:n]
    if w == "leakyrelu_bf16":
        params = (float(floats(case, "wts32.txt")[0]),)
    elif w == "square_bf16":
        params = ()
    else:
        C = int(t["ifm_sv_channel"])
        params = (I64(C), np.ascontiguousarray(floats(case, "wts32.txt")[:C]))
    ref = getattr(ga, f"elementwise_{w}")(x, ht, n, *params)
    return Op(f"npu_elementwise_{w}", (x, nt, OUT, I64(n), *params), n * 2, np.uint16, want, ref)


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
    want = to_logical(data(case, ofm(case), sdt))
    wts = data(case, "wts32.txt")
    npk = C * 32 // cs
    wt16 = wts[:4 * npk].view(np.uint16)
    # the 16-bit cw / cb, c_stride lanes repeated to 32: keep one copy
    unpack = lambda v: np.ascontiguousarray(  # noqa: E731
        v.reshape(C // cs, 32 // cs, cs)[:, 0, :].reshape(-1))
    cw, cb = unpack(wt16[:npk]), unpack(wt16[npk:2 * npk])
    alpha = float(wts[4 * npk:4 * npk + 4].view(np.float32)[0])
    act = int(t["activation_type"])
    nt = {"bfloat16": NPU_BF16, "float16": NPU_F16, "float8": NPU_FP8}[dt]
    gt = "fp8" if fp8 else ("f16" if dt == "float16" else "bf16")
    args = (x, nt, OUT, I64(n), I64(C), cw, cb, act, alpha)
    ref = ga.batchnorm2d_stream_bf16(x, gt, n, C, cw, cb, act, alpha)
    return Op("npu_batchnorm2d_stream_bf16", args, n * np.dtype(sdt).itemsize, sdt, want, ref)


# dump dir -> (entry key, runner)
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


@pytest.mark.skipif(not CASES, reason="no fixture cases")
@pytest.mark.parametrize("wdir,case", CASES)
def test_fixture_arith_float(wdir, case):
    op = prepare(wdir, case)
    got = call(op.name, *op.args, out_nbytes=op.nbytes, out_dtype=op.dtype)
    assert_bytes(op.view(got), op.want)
    assert_bytes(op.view(np.asarray(op.ref).astype(op.dtype)), op.want)
