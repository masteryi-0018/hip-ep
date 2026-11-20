# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The QDQ integer GEMM entry points against the reference models of the model repository."""

from types import SimpleNamespace

import numpy as np
import pytest

import fixture as fm
from fixture import TYPES
from npu import I64, OUT, call

DIRS = ["gemm_add_a16w8_channelwise_wrapper", "gemm_add_a16w8_wrapper",
        "gemm_qdq_a16a16_transpose_wrapper", "gemm_qdq_a16a16_wrapper", "gemm_qdq_a16a8_wrapper",
        "gemm_qdq_a16w4_wrapper", "gemm_qdq_a16w8_channelwise_wrapper",
        "gemm_qdq_a16w8_hybrid_channelwise_wrapper", "gemm_qdq_a16w8_hybrid_wrapper",
        "gemm_qdq_a16w8_stream_channelwise_wrapper", "gemm_qdq_a16w8_stream_wrapper",
        "gemm_qdq_a16w8_wrapper", "gemm_qdq_a8a8_transpose_wrapper", "gemm_qdq_a8a8_wrapper",
        "gemm_qdq_a8w4_wrapper", "gemm_qdq_a8w8_channelwise_stream_wrapper",
        "gemm_qdq_a8w8_channelwise_wrapper", "gemm_qdq_a8w8_stream_wrapper",
        "gemm_qdq_a8w8_wrapper", "qdq_gemm_add_a8w8_wrapper"]
MODELS = tuple(f"activated_mmult_qdq_{k}_model" for k in ("int8x8", "int8x4", "int16x8", "int16x4",
                                                            "int16x16"))
HOOKS = {"*Gemm*": ("compute_bit_exact",) + MODELS}
PARAMS = ("has_actv_sum", "ofm_scale", "ofm_zero_point")


def extract(d, seen):
    """Logical operands, coefficients and expected output of the model object."""
    s = fm.last(seen)
    cls, data = type(s).__name__, s.data
    a0, a1 = s.gen_clip_coeff() if hasattr(s, "gen_clip_coeff") else (0.0, 0.0)
    fx = dict(cls=cls, matA=data["matA"], matB=data["matB"], ofm=data["ofm_0"],
              ofm_dtype=str(s.ofm_dtype), activation_type=getattr(s, "activation_type", 0),
              act0=np.float32(a0), act1=np.float32(a1))
    if cls.startswith("QDQ_GemmA16W8Hybrid"):
        fx.update(kind="hybrid", coeff_a=s._compensated_coeff(compensate=True),
                  coeff_b=s._compensated_coeff(compensate=False), sv_n=s.sv_n_dim,
                  hybrid=int(s.HYBRID_MODE), channelwise_c2=int(s.channelwise_c2()))
    elif cls == "QDQ_GemmAddA8W8Wrapper":
        fx.update(kind="add_a8w8", coeff=s.kernel_qdq_coeff(), ifm3=data["ifm3"],
                  add_c=np.array([s.add_c0, s.add_c1, s.add_c2], np.float32),
                  kpass=int(s.kpass_add()), has_actv_sum=s.has_actv_sum)
    elif cls.startswith("QDQ_GemmAddA16W8"):
        fx.update(kind="add_a16w8", coeff=s.compute_qdq_param(matB_with_coeff=True)[0],
                  ifm2=data["ifm2"], add_c=np.array([s.add_c0, s.add_c1, s.add_c2], np.float32))
    else:
        name = next((n for n in MODELS if n in seen), None)
        if name is None:
            raise fm.Skip("the reference model does not call a recorded kernel model")
        a = fm.args(seen, name)
        scalars = None if name.endswith("int16x4_model") else a[3]
        fx.update(kind=name, coeff=a[2],
                  scalars=np.asarray(scalars if scalars is not None else [], np.float32))
        for k in PARAMS:
            if hasattr(a[-1], k):
                fx["p_" + k] = getattr(a[-1], k)
    return fx


ACT_NONE, ACT_CLIP, ACT_LRELU, ACT_GELU = 0, 1, 2, 4
f32 = np.float32


def act_kind(t):
    return {0: ACT_NONE, 1: ACT_CLIP, 2: ACT_CLIP, 8: ACT_CLIP, 100: ACT_CLIP, 3: ACT_LRELU,
            6: ACT_GELU}.get(int(t))


def ty(a):
    return TYPES[np.asarray(a).dtype.name][0]


def params(z):
    return SimpleNamespace(**{k: z["p_" + k].item() for k in PARAMS if "p_" + k in z})


def coeffs(z, n, kind):
    """(c0, c1, c2, c3) per column from the coefficient rows / scalars."""
    rows = (z["coeff"].reshape(-1, z["coeff"].shape[-1]) if "coeff" in z and z["coeff"].ndim
            else np.zeros((0, n), f32))
    sc = list(z["scalars"])
    nr = rows.shape[0]
    get = lambda i: fm.col(rows[i], n) if i < nr else fm.col(sc[i - nr], n)
    p = params(z)
    if kind == "int8x8":
        asum = int(getattr(p, "has_actv_sum", 0))
        c2 = get(2) if asum else get(1)
        return get(0), get(1) if asum else None, c2, get(3) if asum > 1 else None
    ncf = nr + len(sc)
    if kind == "int16x4":
        return get(0), get(1) if nr > 2 else None, get(2) if nr > 2 else get(1), None
    c2 = get(2) if ncf > 2 else get(1)
    return get(0), get(1) if ncf > 2 else None, c2, get(3) if ncf > 3 else None


ENTRY = {  # fixture class -> entry
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


@pytest.mark.parametrize("d,case", fm.cases(DIRS, __file__))
def test_fixture(d, case):
    z = fm.load(case)
    kind, cls = str(z["kind"]), str(z["cls"])
    act = act_kind(z["activation_type"])
    a, b, want = z["matA"], z["matB"], np.asarray(z["ofm"])
    m, k = a.shape
    n = b.shape[1]
    ot, odt = TYPES[str(z["ofm_dtype"])]
    a0, a1 = float(z["act0"]), float(z["act1"])
    if act == ACT_GELU:
        # quantized with fp32(1 / ofm_scale) (computed in double) and the zero point
        p = params(z)
        a0, a1 = float(f32(1.0 / float(p.ofm_scale))), float(f32(p.ofm_zero_point))
    if kind == "hybrid":
        c2 = fm.col(z["coeff_a"][2], n) if int(z["channelwise_c2"]) else fm.col(z["coeff_a"][2][0], n)
        name = "npu_gemm_a16w8_hybrid_channelwise" if "Channelwise" in cls else "npu_gemm_a16w8_hybrid"
        cols = int(z["sv_n"]) if int(z["hybrid"]) else n
        args = (a, ty(a), b, ty(b), OUT, ot, I64(m), I64(k), I64(n), fm.col(z["coeff_a"][0], n),
                fm.col(z["coeff_b"][0], n), fm.col(z["coeff_a"][1], n), c2, I64(cols), act, a0, a1)
    elif kind == "add_a8w8":
        c = z["coeff"]
        name = "npu_qdq_gemm_add_a8w8"
        args = (a, ty(a), b, ty(b), z["ifm3"], ty(z["ifm3"]), OUT, ot, I64(m), I64(k), I64(n),
                fm.col(c[0], n), fm.col(c[1], n) if int(z["has_actv_sum"]) else None, fm.col(c[2], n),
                *(float(x) for x in z["add_c"]), int(z["kpass"]))
    elif kind == "add_a16w8":
        c = z["coeff"]
        name = "npu_gemm_add_a16w8"
        args = (a, ty(a), b, ty(b), np.ascontiguousarray(z["ifm2"].astype(np.uint16)), OUT, I64(m), I64(k),
                I64(n), fm.col(c[0], n), fm.col(c[1], n), fm.col(c[2], n), *(float(x) for x in z["add_c"]))
    elif kind.endswith("int16x16_model"):
        sc = [float(x) for x in z["scalars"]]
        name = ("npu_gemm_qdq_a16a16_transpose" if "Transpose" in cls else
                "npu_gemm_qdq_int16x16")
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
