# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The bf16 gemm / conv entry points against the reference models of the model repository."""

import numpy as np
import pytest

import fixture as fm
from npu import I64, OUT, call

DIRS = ["conv_add_bf16x16_wrapper", "conv_bf16x16_buffer_wrapper", "conv_bf16x16_wrapper",
        "depthwise_conv_bf16x16_c32_wrapper", "depthwise_conv_bf16x16_wrapper",
        "gemm_bf16x16_buffer_wrapper", "gemm_bf16x16_ifm_reuse_wrapper",
        "gemm_bf16x16_transposed_wrapper", "gemm_bf16x16_wrapper",
        "gemm_quant_dequant_bf16x16_ifm_reuse_wrapper", "gemm_quant_dequant_bf16x16_wrapper"]
HOOKS = {"*Bf16x16*": ("compute_bit_exact",)}
f32 = np.float32


def extract(d, seen):
    """Logical operands (bf16 bits), the fp32 bias, parameters and expected output."""
    s = fm.last(seen)
    cls, data = type(s).__name__, s.data
    act = int(getattr(s, "activation", 0))
    alpha = f32(getattr(s, "leakyRelu_alpha", 0.0))
    if cls.startswith("Gemm"):
        bias = np.asarray(data["bias"], f32).reshape(-1) if (s.do_bias and "bias" in data) else None
        fx = dict(a=fm.bf16_bits(data["matA"]), b=fm.bf16_bits(data["matB"]))
        if cls.startswith("GemmQuantDequant"):
            inv = f32(np.array([1.0 / s.ofm_scale], dtype=f32)[0])
            fx.update(fl=np.array([inv, f32(s.ofm_scale)], f32),
                      scal=np.array([act, int(s.qdq_dtype == "int16"), int(s.ofm_zero_point)]))
        else:
            fx.update(scal=np.array([act]))
    elif cls.startswith("Conv"):
        sh, sw, kh_sv = int(s.stride_height), int(s.stride_width), int(s.kernel_sv_height)
        fx = dict(ifm=fm.bf16_bits(data["ifm"]), wts=fm.bf16_bits(data["wts"]))
        if cls.startswith("ConvAdd"):
            with_bias = bool(s.has_bias and getattr(s, "do_bias", 1))
            bias = np.asarray(data["bias"], f32).reshape(-1) if with_bias else None
            fx.update(ifm2=fm.bf16_bits(np.asarray(data["ifm2"], f32).reshape(s._ifm2_logical_shape())),
                      scal=np.array([sh, sw, kh_sv, int(s.activation_type), act]))
        else:
            bias = np.asarray(data["bias"], f32).reshape(-1) if getattr(s, "do_bias", 1) else None
            fx.update(scal=np.array([sh, sw, kh_sv, act]), fl=np.array([alpha], f32))
    else:
        with_bias = s.do_bias and data.get("bias") is not None
        bias = np.asarray(data["bias"], f32).reshape(-1) if with_bias else None
        fx = dict(ifm=fm.bf16_bits(data["ifm"]), wts=fm.bf16_bits(data["wts"]),
                  scal=np.array([int(s.stride_height), int(s.stride_width), act]),
                  fl=np.array([alpha], f32))
    # the entry is named after the kernel the directory builds
    fx.update(entry=d[:-len("_wrapper")], bias=bias, cls=cls,
              ofm=fm.bf16_bits(np.asarray(data.get("ofm_0", data.get("ofm")), f32)))
    return fx


@pytest.mark.parametrize("d,case", fm.cases(DIRS, __file__))
def test_fixture(d, case):
    z = fm.load(case)
    entry = str(z["entry"])
    scal = [int(v) for v in z["scal"]]
    fl = [float(v) for v in z["fl"]] if "fl" in z else []
    bias = z["bias"] if "bias" in z else None
    want = z["ofm"]
    n = want.size
    if entry.startswith("gemm"):
        a, b = z["a"], z["b"]
        (m, k), nn = a.shape, b.shape[1]
        args = (a, b, bias, OUT, I64(m), I64(k), I64(nn), scal[0])
        if entry.startswith("gemm_quant_dequant"):
            act, signed, zp = scal
            args += (signed, fl[0], fl[1], zp)
    elif entry.startswith("conv"):
        ifm, wts = z["ifm"], z["wts"]
        h, w, ci = ifm.shape
        kh, kw, _, co = wts.shape
        args = (ifm, wts, bias)
        if entry == "conv_add_bf16x16":
            sh, sw, kh_sv, typ, act = scal
            h2, w2, c2 = z["ifm2"].shape
            tail = (I64(h2), I64(w2), I64(c2), typ, act)
            args += (z["ifm2"],)
        else:
            sh, sw, kh_sv, act = scal
            tail = (act, fl[0]) if entry == "conv_bf16x16_buffer" else (act,)
        args += (OUT, I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), sh, sw, I64(kh_sv), *tail)
    else:
        ifm, wts = z["ifm"], z["wts"]
        h, w, c = ifm.shape
        kh, kw, _ = wts.shape
        sh, sw, act = scal
        args = (ifm, wts, bias, OUT, I64(h), I64(w), I64(c), I64(kh), I64(kw), sh, sw, act, fl[0])
    got = call("npu_" + entry, *args, out_nbytes=n * 2, out_dtype=np.uint16)
    got, want = np.asarray(got).reshape(-1), want.reshape(-1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{z['cls']}: {bad.size}/{n} differ, first at {bad[:6].tolist()}: "
                           f"got {[hex(x) for x in got[bad[:6]]]} want {[hex(x) for x in want[bad[:6]]]}")
