# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""reduction/reduce.h entry points against NPU reference outputs, on the logical output."""

import numpy as np
import pytest

from fixture import TYPES as T
from fixture import cases, data, fword, ofm, tensor_descr, words
from fixture import testcase as tc
from npu import I64, NPU_BF16, OUT
from testing import call_n as go


def from_c64(x, H, W, C):
    """[H][C/64][W][64] -> [H][W][C]."""
    return np.ascontiguousarray(x[:H * W * C].reshape(H, C // 64, W, 64).transpose(0, 2, 1, 3)
                                .reshape(H, W, C))


def shape_of(t):
    H = t["ifm_sv_height"] * t["ifm_height_iteration"]
    W = t["ifm_sv_width"] * t["ifm_width_iteration"]
    Cp = t["ifm_sv_depth"] * t["ifm_depth_iteration"]
    C = t.get("ifm_sv_depth_real", t["ifm_sv_depth"]) * t["ifm_depth_iteration"]
    return H, W, Cp, C


def c64_output(raw, t, H, W, Cp, C, slot):
    """Logical output from the raw dump (`slot` = element spacing with keepdims)."""
    ax = t["axis_flags"]
    oH, oW = (1 if ax & 4 else H), (1 if ax & 2 else W)
    if ax & 1:
        if t.get("keepdims", 0) and ax in (1, 3, 5):
            raw = raw[::slot]
        return raw[:oH * oW]
    return from_c64(raw, oH, oW, Cp)[..., :C]


def run_sum(case, t, kind):
    H, W, Cp, C = shape_of(t)
    ax = t["axis_flags"]
    nt, dt = T[t["dtype"]]
    x = np.ascontiguousarray(from_c64(data(case, "ifm32.txt", dt), H, W, Cp)[..., :C])
    want = c64_output(data(case, ofm(case), dt), t, H, W, Cp, C, 4 // np.dtype(dt).itemsize
                      if kind != "a32" else 2)
    shp = (I64(H), I64(W), I64(C), ax)
    w = words(case, "wts32.txt")
    if kind == "bf16":
        got = go("npu_reduce_sum_c64_bf16", x, nt, OUT, *shp, fword(case, 0, "wts32.txt"),
                 n=want.size, dt=dt)
    elif kind == "fp8":
        got = go("npu_reduce_sum_c64_fp8", x, OUT, *shp[:3], I64(t["ifm_height_iteration"]),
                 I64(t["ifm_width_iteration"]), ax, fword(case, 0, "wts32.txt"),
                 n=want.size, dt=dt)
    elif kind == "a32":
        shift_res = int(w[2:3].view(np.int16)[0])  # the low 16 bits the kernel reads
        got = go("npu_qdq_reduce_sum_c64_a32", x, nt, OUT, nt, *shp,
                 fword(case, 0, "wts32.txt"), fword(case, 1, "wts32.txt"), shift_res,
                 n=want.size, dt=dt)
    else:
        c2 = int(w[2:4].view(np.int64)[0])
        got = go(f"npu_reduce_sum_c64_{kind}", x, nt, OUT, nt, *shp, int(w[0] & 0xFF),
                 int(w[1] & 0xFF), I64(c2), n=want.size, dt=dt)
    return got, want


def run_prod_l2(case, t, op, qdq):
    H, W, Cp, C = shape_of(t)
    it, idt = T[t["ifm_dtype"]]
    ot, odt = T[t["ofm_dtype"]]
    x = np.ascontiguousarray(from_c64(data(case, "ifm32.txt", idt), H, W, C))
    want = c64_output(data(case, ofm(case), odt), t, H, W, C, C, 1)
    shp = (I64(H), I64(W), I64(C), t["axis_flags"])
    if not qdq:
        name = f"npu_reduce_{op}_bf16"
        return go(name, x, it, OUT, *shp, n=want.size, dt=odt), want
    co = words(case, "coeff32.txt").view(np.float32)
    coeffs = [float(co[k]) for k in (0, 64, 128, 192)]  # dq_zp, dq_sc, q_zp, q_sc
    bits = 8 if np.dtype(idt).itemsize == 1 else 16
    compute = (T[t["reduce_dtype"]][0],)
    if op == "prod":  # bf16 only
        assert compute == (NPU_BF16,), t["reduce_dtype"]
        compute = ()
    got = go(f"npu_qdq_reduce_{op}_a{bits}", x, it, OUT, ot, *shp, *compute, *coeffs,
             n=want.size, dt=odt)
    return got, want


def run_mrs(case, t, bits):
    dims = ("height", "width", "channel")
    it = [t[f"ifm1_{d}_iteration"] for d in dims]
    av = [t[f"ifm1_valid_full_{d}"] for d in dims]
    bv = [t[f"ifm2_valid_full_{d}"] for d in dims]
    ap = [t[f"ifm1_physical_sv_{d}"] * i for d, i in zip(dims, it)]
    bp = [t[f"ifm2_physical_sv_{d}"] * (i if v > 1 else 1) for d, i, v in zip(dims, it, bv)]
    nt, dt = T[t["dtype"]]
    a = np.ascontiguousarray(data(case, "ifm32.txt", dt)[:np.prod(ap)].reshape(ap)
                             [:av[0], :av[1], :av[2]])
    b = np.ascontiguousarray(data(case, "wts32.txt", dt)[:np.prod(bp)].reshape(bp)
                             [:bv[0], :bv[1], :bv[2]])
    valid, phys, _ = tensor_descr(case)
    want = data(case, ofm(case), dt)[:np.prod(phys)].reshape(phys)
    want = want[:valid[0], :valid[1], :valid[2]]
    c = [float(v) for v in words(case, "coeff32.txt")[:4].view(np.float32)]
    got = go(f"npu_qdq_elementwise_broadcasting3d_mul_reducesum_a{bits}", a, b, nt, OUT, nt,
             *[I64(v) for v in av + bv], t["axis_flags"], *c, n=want.size, dt=dt)
    return got, want


def run_cumsum(case, t):
    H = t["ifm_height"] * t["ofm_height_iteration"]
    W = t["ifm_width"] * t["ofm_width_iteration"]
    it, idt = T[t["ifm_dtype"]]
    ot, odt = T[t["ofm_dtype"]]
    x = np.ascontiguousarray(data(case, "ifm32.txt", idt)[:H * W])
    want = data(case, ofm(case), odt)[:H * W]
    got = go("npu_qdq_cumsum", x, it, OUT, ot, I64(H), I64(W), t["axis"], t["exclusive"],
             t["reverse"], fword(case, 0, "wts32.txt"), fword(case, 1, "wts32.txt"),
             n=H * W, dt=odt)
    return got, want


RUNNERS = {
    "reduce_sum_c64_bf16_wrapper": lambda c, t: run_sum(c, t, "bf16"),
    "reduce_sum_c64_fp8_wrapper": lambda c, t: run_sum(c, t, "fp8"),
    "reduce_sum_c64_a8_wrapper": lambda c, t: run_sum(c, t, "a8"),
    "reduce_sum_c64_a16_wrapper": lambda c, t: run_sum(c, t, "a16"),
    "qdq_reduce_sum_c64_a32_wrapper": lambda c, t: run_sum(c, t, "a32"),
    "reduce_prod_c64_bf16_wrapper": lambda c, t: run_prod_l2(c, t, "prod", False),
    "reduce_l2_c64_bf16_wrapper": lambda c, t: run_prod_l2(c, t, "l2", False),
    "qdq_reduce_prod_c64_a8_wrapper": lambda c, t: run_prod_l2(c, t, "prod", True),
    "qdq_reduce_prod_c64_a16_wrapper": lambda c, t: run_prod_l2(c, t, "prod", True),
    "qdq_reduce_l2_c64_a8_wrapper": lambda c, t: run_prod_l2(c, t, "l2", True),
    "qdq_reduce_l2_c64_a16_wrapper": lambda c, t: run_prod_l2(c, t, "l2", True),
    "qdq_elementwise_broadcasting3d_mul_reducesum_a8_wrapper": lambda c, t: run_mrs(c, t, 8),
    "qdq_elementwise_broadcasting3d_mul_reducesum_a16_wrapper": lambda c, t: run_mrs(c, t, 16),
    "qdq_cumsum_axis0_wrapper": run_cumsum,
    "qdq_cumsum_axis1_wrapper": run_cumsum,
}

CASES = cases(list(RUNNERS))


def flat(t):
    """testcase.json with the parameter-group prefixes dropped."""
    return {k.split(".", 1)[1] if k.startswith("Kernel") else k: v for k, v in t.items()}


@pytest.mark.skipif(not CASES, reason="no fixture cases")
@pytest.mark.parametrize("wrapper,case", CASES)
def test_fixture_reduce(wrapper, case):
    t = flat(tc(case))
    got, want = RUNNERS[wrapper](case, t)
    got = np.asarray(got).reshape(-1)
    want = np.ascontiguousarray(want).reshape(-1).astype(got.dtype)
    assert got.size == want.size, f"size {got.size} != {want.size}"
    bad = np.nonzero((got.view(np.uint8).reshape(got.size, -1) !=
                      want.view(np.uint8).reshape(want.size, -1)).any(1))[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} elements differ from sim, first at "
                           f"{bad[:8].tolist()}: got {got[bad[:8]].tolist()} "
                           f"want {want[bad[:8]].tolist()}")
