# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The gridsample-group entry points (include/npu_emu_gridsample.h) against
the NPU: each mllib regression case's inputs are converted from the DDR
layout the wrapper's test bench writes (one [H][W][2] slice per channel pair,
padded to the stream / conversion granule) to the logical layout, run
through the entry point, and the result is compared byte for byte with the
aiesim output (data/ofm32.txt) on every output element."""

import ctypes
import json
import os

import numpy as np
import pytest

from aiesim_common import cases, data, words
from npu import I64, NPU_BF16, NPU_F16, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT, Host
from npu import call as callx

FLOAT = {"bfloat16": (NPU_BF16, "bf16"), "float16": (NPU_F16, "f16")}
INT = {"int8": (NPU_I8, np.int8, "i8"), "uint8": (NPU_U8, np.uint8, "u8"),
       "int16": (NPU_I16, np.int16, "i16"), "uint16": (NPU_U16, np.uint16, "u16")}


class GridQdq(ctypes.Structure):
    """npu_gridsample_qdq (GridSampleBf16QdqCoeffs)."""
    _fields_ = [("grid_zp", ctypes.c_int32), ("grid_scale", ctypes.c_float),
                ("ifm_zp", ctypes.c_int32), ("ifm_scale", ctypes.c_float),
                ("ofm_zp", ctypes.c_int32), ("ofm_scale", ctypes.c_float)]


def ifm_slices(case, t, dt, align):
    """ifm32.txt: per channel pair an [H][W][2] slice padded to `align`
    elements -> [H][W][C]."""
    H, W, ci = t["in_h"], t["in_w"], t["c_iteration"]
    flat = data(case, "ifm32.txt", dt)
    n, step = H * W * 2, -(-H * W * 2 // align) * align
    return np.ascontiguousarray(np.concatenate(
        [flat[k * step:k * step + n].reshape(H, W, 2) for k in range(ci)], axis=-1))


def ofm_slices(case, t, dt, align):
    """ofm32.txt written as one flat slice per kernel call (channel pair
    outer, width iteration inner), each [Ho][sv_out_w][2] padded to `align`
    elements -> [Ho][Wo][C]."""
    Ho, Wo, C = t["out_h"], t["out_w"], t["in_c"]
    sw, wi, ci = t["sv_out_w"], t["w_iteration"], t["c_iteration"]
    flat = data(case, "ofm32.txt", dt)
    n = Ho * sw * 2
    step = -(-n // align) * align
    out = np.empty((Ho, Wo, C), dt)
    pos = 0
    for c in range(ci):
        for w in range(wi):
            out[:, w * sw:(w + 1) * sw, 2 * c:2 * c + 2] = flat[pos:pos + n].reshape(Ho, sw, 2)
            pos += step
    return out


def ofm_hwc(case, t, dt):
    Ho, Wo, C = t["out_h"], t["out_w"], t["in_c"]
    return data(case, "ofm32.txt", dt)[:Ho * Wo * C].reshape(Ho, Wo, C)


def grid_u16(case, t):
    Ho, Wo = t["out_h"], t["out_w"]
    return np.ascontiguousarray(data(case, "wts32.txt", np.uint16)[:Ho * Wo * 2].reshape(Ho, Wo, 2))


def qdq_coeff(case):
    w = words(case)
    return tuple(int(w[k:k + 1].view(np.int32)[0]) if k % 2 == 0 else
                 float(w[k:k + 1].view(np.float32)[0]) for k in range(6))


# Each loader returns (entry point, args before OUT, args after OUT, golden
# call, want); `args` are what the entry point takes, `golden` a function of
# the golden_gridsample module giving the same result.

def load_bf16(case, t):
    nt, tn = FLOAT[t["ifm_dtype"]]
    x = ifm_slices(case, t, np.uint16, 32)
    grid = grid_u16(case, t)
    want = ofm_hwc(case, t, np.uint16)
    return ("npu_gridsample_bf16", (x, nt, grid), (), (),
            lambda gs: gs.gridsample_bf16(x, grid, tn), want)


def load_fp8(case, t):
    nt, tn = FLOAT[t["wts_dtype"]]
    x = ifm_slices(case, t, np.uint8, 64)
    grid = grid_u16(case, t)
    s_in, recip = (float(v) for v in words(case)[:2].view(np.float32))
    want = ofm_slices(case, t, np.uint8, 64)
    return ("npu_gridsample_fp8", (x, grid, nt), (s_in, recip), (),
            lambda gs: gs.gridsample_fp8(x, grid, tn, s_in, recip), want)


def load_qdq(case, t, a16):
    nt, dt, tn = INT[t["ifm_dtype"]]
    x = ifm_slices(case, t, dt, 32 if a16 else 64)
    grid = grid_u16(case, t)
    q = qdq_coeff(case)
    al = t["align_corners"]
    want = ofm_hwc(case, t, dt) if a16 else ofm_slices(case, t, dt, 1)
    ct = FLOAT[t.get("compute_dtype", "bfloat16")] if a16 else FLOAT["bfloat16"]
    name = "npu_qdq_gridsample_a16" if a16 else "npu_qdq_gridsample_a8"
    tail = (al, ct[0]) if a16 else (al,)
    return (name, (x, nt, grid), tail, q,
            lambda gs: gs.qdq_gridsample(x, grid, al, q, ct[1], tn), want)


LOADERS = {"gridsample_bf16_wrapper": load_bf16, "gridsample_fp8_wrapper": load_fp8,
           "qdq_gridsample_a16_wrapper": lambda c, t: load_qdq(c, t, True),
           "qdq_gridsample_a8_wrapper": lambda c, t: load_qdq(c, t, False)}

CASES = cases(list(LOADERS))


def load(wrapper, case):
    t = json.load(open(os.path.join(case, "testcase.json")))
    return t, LOADERS[wrapper](case, t)


@pytest.mark.skipif(not CASES, reason="no mllib regression outputs reachable")
@pytest.mark.parametrize("wrapper,case", CASES)
def test_aiesim_gridsample(wrapper, case):
    t, (name, pre, post, q, _, want) = load(wrapper, case)
    H, W, C, Ho, Wo = t["in_h"], t["in_w"], t["in_c"], t["out_h"], t["out_w"]
    args = list(pre) + [OUT, I64(H), I64(W), I64(C), I64(Ho), I64(Wo)] + list(post)
    if q:
        args.append(Host(ctypes.byref(GridQdq(*q))))
    got = callx(name, *args, out_nbytes=want.nbytes, out_dtype=want.dtype)
    got, want = got.reshape(-1), want.reshape(-1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} elements differ from aiesim, first at "
                           f"{bad[:8].tolist()}: got {got[bad[:8]].tolist()} "
                           f"want {want[bad[:8]].tolist()}")
