# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The QDQ integer GEMM entry points against fixtures exported from mllib's
golden runs (NPU_EMU_MATMUL_FIXTURES, one .npz per nightly case): logical A / B
(and the add operand), the coefficient words the kernel reads, the parameters,
and the golden output. Only cases whose golden ofm32_ref.txt is byte-identical
to the nightly aiesim ofm32.txt are exported, so matching the fixture output
matches aiesim. Exported by logs/class7a/fix_run.py (mllib PR #15102)."""

import glob
import os

import numpy as np
import pytest

from npu import (I64, NPU_I8, NPU_I16, NPU_U8, NPU_U16, OUT, call)

ROOT = os.environ.get("NPU_EMU_MATMUL_FIXTURES",
                      "/wrk/xcohdnobkup3/yigao102/shared/npu-emu/fixtures/matmul_a")
TYPES = {"int8": (NPU_I8, np.int8), "uint8": (NPU_U8, np.uint8), "int16": (NPU_I16, np.int16),
         "uint16": (NPU_U16, np.uint16)}
ACT_NONE, ACT_CLIP, ACT_LRELU = 0, 1, 2
f32 = np.float32


def act_kind(t):
    return {0: ACT_NONE, 1: ACT_CLIP, 2: ACT_CLIP, 8: ACT_CLIP, 100: ACT_CLIP, 3: ACT_LRELU}.get(int(t))


def ty(a):
    return TYPES[np.asarray(a).dtype.name][0]


def col(v, n):
    return np.ascontiguousarray(np.broadcast_to(np.asarray(v, f32).reshape(-1), (n,)))


def params(z):
    """The model call's parameters, saved as the repr of a dict of scalars (np.float32 etc.)."""
    return eval(str(z["param"]), {"__builtins__": {}, "np": np})


def coeffs(z, n, kind):
    """(c0, c1, c2, c3) per column from the model call's coefficient rows / scalars."""
    rows = z["coeff"].reshape(-1, z["coeff"].shape[-1]) if z["coeff"].ndim else np.zeros((0, n), f32)
    sc = list(z["scalars"])
    nr = rows.shape[0]
    get = lambda i: col(rows[i], n) if i < nr else col(sc[i - nr], n)
    p = params(z)
    if kind == "int8x8":
        asum = int(p.get("has_actv_sum", 0))
        c2 = get(2) if asum else get(1)
        return get(0), get(1) if asum else None, c2, get(3) if asum > 1 else None
    ncf = nr + len(sc)
    if kind == "int16x4":
        return get(0), get(1) if nr > 2 else None, get(2) if nr > 2 else get(1), None
    c2 = get(2) if ncf > 2 else get(1)
    return get(0), get(1) if ncf > 2 else None, c2, get(3) if ncf > 3 else None


ENTRY = {  # golden class -> npu_emu entry
    "QDQ_GemmA8W8Wrapper": "npu_gemm_a8w8", "QDQ_GemmA8W8ChannelwiseWrapper": "npu_gemm_a8w8",
    "QDQ_GemmA8W8StreamWrapper": "npu_gemm_a8w8_stream",
    "QDQ_GemmA8W8ChannelwiseStreamWrapper": "npu_gemm_a8w8_stream",
    "QDQ_GemmA8A8Wrapper": "npu_gemm_a8a8", "QDQ_GemmA8A8TransposeWrapper": "npu_gemm_qdq_a8a8_transpose",
    "QDQ_GemmA8W4Wrapper": "npu_gemm_a8w4",
    "QDQ_GemmA16W8Wrapper": "npu_gemm_a16w8", "QDQ_GemmA16W8ChannelwiseWrapper": "npu_gemm_a16w8",
    "QDQ_GemmA16W8StreamWrapper": "npu_gemm_a16w8_stream",
    "QDQ_GemmA16W8StreamChannelwiseWrapper": "npu_gemm_a16w8_stream",
    "QDQ_GemmA16A8Wrapper": "npu_gemm_a16a8", "QDQ_GemmA16W4Wrapper": "npu_gemm_a16w4",
}


def fixtures():
    return [pytest.param(f, id=os.path.relpath(f, ROOT)[:-4]) for f in sorted(glob.glob(f"{ROOT}/*/*.npz"))]


@pytest.mark.parametrize("path", fixtures())
def test_fixture(path):
    z = np.load(path, allow_pickle=True)
    kind, cls = str(z["kind"]), str(z["cls"])
    act = act_kind(z["activation_type"])
    if act is None:
        pytest.skip(f"activation_type {int(z['activation_type'])} (GELU) is not emulated")
    a, b, want = z["matA"], z["matB"], np.asarray(z["ofm"])
    m, k = a.shape
    n = b.shape[1]
    ot, odt = TYPES[str(z["ofm_dtype"])]
    a0, a1 = float(z["act0"]), float(z["act1"])
    if kind == "hybrid":
        c2 = col(z["coeff_a"][2], n) if int(z["channelwise_c2"]) else col(z["coeff_a"][2][0], n)
        name = "npu_gemm_a16w8_hybrid_channelwise" if "Channelwise" in cls else "npu_gemm_a16w8_hybrid"
        cols = int(z["sv_n"]) if int(z["hybrid"]) else n
        args = (a, ty(a), b, ty(b), OUT, ot, I64(m), I64(k), I64(n), col(z["coeff_a"][0], n),
                col(z["coeff_b"][0], n), col(z["coeff_a"][1], n), c2, I64(cols), act, a0, a1)
    elif kind == "add_a8w8":
        c = z["coeff"]
        name = "npu_qdq_gemm_add_a8w8"
        args = (a, ty(a), b, ty(b), z["ifm3"], ty(z["ifm3"]), OUT, ot, I64(m), I64(k), I64(n),
                col(c[0], n), col(c[1], n) if int(z["has_actv_sum"]) else None, col(c[2], n),
                *(float(x) for x in z["add_c"]), int(z["kpass"]))
    elif kind == "add_a16w8":
        c = z["coeff"]
        name = "npu_gemm_add_a16w8"
        args = (a, ty(a), b, ty(b), np.ascontiguousarray(z["ifm2"].astype(np.uint16)), OUT, I64(m), I64(k),
                I64(n), col(c[0], n), col(c[1], n), col(c[2], n), *(float(x) for x in z["add_c"]))
    elif kind.endswith("int16x16_model"):
        sc = [float(x) for x in z["scalars"]]
        name = "npu_gemm_qdq_a16a16_transpose"
        args = (a, ty(a), b, ty(b), OUT, ot, I64(m), I64(k), I64(n), *sc[:4], act, a0, a1)
    else:
        name = ENTRY[cls]
        c0, c1, c2, c3 = coeffs(z, n, "int8x8" if "int8x" in kind else
                                "int16x4" if "int16x4" in kind else "int16x8")
        args = (a, ty(a), b, ty(b), OUT, ot, I64(m), I64(k), I64(n), c0, c1, c2, c3, act, a0, a1)
    got = call(name, *args, out_nbytes=m * n * np.dtype(odt).itemsize, out_dtype=odt).reshape(m, n)
    want = want.astype(odt).reshape(-1, n)
    got = got[:want.shape[0]]
    bad = np.nonzero(got != want)
    assert bad[0].size == 0, (f"{name}: {bad[0].size}/{want.size} differ, first at "
                              f"{list(zip(*bad))[:4]}: got {got[bad][:4]} want {want[bad][:4]}")
