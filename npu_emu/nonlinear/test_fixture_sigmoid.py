# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""sigmoid / tanh entry points and golden_sigmoid.py against NPU reference outputs."""

import numpy as np
import pytest

import golden_sigmoid as gs
from npu import I64, OUT, call
from fixture import HALF, INT, assert_bytes, cases, data, ofm, tensor_descr
from fixture import testcase as tc

QDQ = {"qdq_sigmoid_a8_wrapper": ("npu_sigmoid_a8", gs.sigmoid_a8),
       "qdq_sigmoid_a16_wrapper": ("npu_sigmoid_a16", gs.sigmoid_a16),
       "qdq_tanh_a8_wrapper": ("npu_tanh_a8", gs.tanh_a8),
       "qdq_tanh_a16_wrapper": ("npu_tanh_a16", gs.tanh_a16)}
QDQ_LUT = {
    "qdq_sigmoid_lut_a8_wrapper": ("npu_qdq_sigmoid_lut_a8", gs.qdq_sigmoid_lut_a8),
    "qdq_sigmoid_lut_a16_wrapper": ("npu_qdq_sigmoid_lut_a16", gs.qdq_sigmoid_lut_a16),
    "qdq_tanh_lut_a8_wrapper": ("npu_qdq_tanh_lut_a8", gs.qdq_tanh_lut_a8),
    "qdq_tanh_lut_a16_wrapper": ("npu_qdq_tanh_lut_a16", gs.qdq_tanh_lut_a16)}
LUT16 = {
    "sigmoid_lut_bf16_wrapper": ("npu_elementwise_sigmoid_lut_bf16",
                                 gs.elementwise_sigmoid_lut_bf16),
    "tanh_lut_bf16_wrapper": ("npu_elementwise_tanh_lut_bf16", gs.elementwise_tanh_lut_bf16)}
FP8 = ("sigmoid_lut_fp8_wrapper", "tanh_lut_fp8_wrapper")
DIRS = list(QDQ) + list(QDQ_LUT) + list(LUT16) + list(FP8)


def build(d, case):
    """(entry, args, nbytes, want bytes, reference) for one case."""
    t = tc(case)
    n = int(t["ifm_sv_size"]) * int(t["ifm_sv_iteration"])
    if d in QDQ or d in QDQ_LUT:
        it, idt, ik = INT[t["ifm_dtype"]]
        ot, odt, ok = INT[t["ofm_dtype"]]
        x = data(case, "ifm32.txt", idt)[:n]
        wname = "coeff32.txt" if d in QDQ else "wts32.txt"
        w = data(case, wname, np.float32)
        co = [float(w[k]) for k in (0, 64, 128, 192)]
        want = data(case, ofm(case), odt)[:n]
        if d in QDQ:
            name, fn = QDQ[d]
            ct, ck = HALF[t["compute_dtype"]]
            args = [x, it, OUT, ot, I64(n), ct, *co]
            ref = fn(x.astype(np.int64), ok, ck, *co)
        else:
            name, fn = QDQ_LUT[d]
            lut = w[256:768].copy()
            args = [x, it, lut, OUT, ot, I64(n), *co]
            ref = fn(x.astype(np.int64), ok, *co, lut)
        return name, args, n * np.dtype(odt).itemsize, want, ref.astype(odt)
    w = data(case, "wts32.txt", np.float32)
    lut = w[:512].copy()
    if d in LUT16:
        name, fn = LUT16[d]
        ht, hk = HALF[t["ifm_dtype"]]
        x = data(case, "ifm32.txt", np.uint16)[:n]
        want = data(case, ofm(case), np.uint16)[:n]
        return name, [x, ht, lut, OUT, I64(n)], 2 * n, want, fn(x, hk, lut)
    x = data(case, "ifm32.txt", np.uint8)[:n]
    want = data(case, ofm(case), np.uint8)[:n]
    s_in, recip = float(w[512]), float(w[513])
    if d == "sigmoid_lut_fp8_wrapper":
        return ("npu_elementwise_sigmoid_lut_fp8", [x, lut, OUT, I64(n), s_in, recip], n, want,
                gs.elementwise_sigmoid_lut_fp8(x, lut, s_in, recip))
    ct, ck = HALF[t["compute_dtype"]]
    return ("npu_elementwise_tanh_lut_fp8", [x, lut, OUT, I64(n), ct, s_in, recip], n, want,
            gs.elementwise_tanh_lut_fp8(x, ck, lut, s_in, recip))


def check(d, case):
    descr = tensor_descr(case)
    assert descr is None or descr[0] == descr[1], f"partial valid region not modelled: {descr}"
    name, args, nbytes, want, ref = build(d, case)
    assert_bytes(ref, want)
    got = call(name, *args, out_nbytes=nbytes, out_dtype=np.uint8)
    assert_bytes(got, want)


@pytest.mark.parametrize("d,case", cases(DIRS))
def test_fixture_sigmoid(d, case):
    check(d, case)
