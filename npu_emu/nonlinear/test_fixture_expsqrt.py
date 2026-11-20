# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""exp, elu, hardsigmoid, sqrt entry points and golden_expsqrt.py against NPU reference outputs."""

import numpy as np
import pytest

import golden_expsqrt as ge
from fixture import HALF as F16T
from fixture import INT, assert_bytes, cases, data, ofm, words
from fixture import testcase as tc
from npu import I64, OUT, call


def nelem(t):
    return t["ifm_sv_size"] * t["ifm_sv_iteration"]


def fwords(case, name, k):
    return [np.float32(v) for v in words(case, name)[:k].view(np.float32)]


def ld_16(op, nwts):
    def ld(case, t):
        nt, tt = F16T[t["ifm_dtype"]]
        n = nelem(t)
        x = data(case, "ifm32.txt", np.uint16)[:n].copy()
        want = data(case, ofm(case), np.uint16)[:n]
        c = fwords(case, "wts32.txt", nwts) if nwts else []
        name = "npu_sqrt_bf16" if op == "sqrt" else f"npu_elementwise_{op}_bf16"
        ref = {"exp": ge.exp16, "elu": ge.elu16, "hardsigmoid": ge.hardsigmoid16,
               "sqrt": ge.sqrt16}[op]
        return (lambda: ref(x, tt, *c),
                lambda: call(name, x, nt, OUT, I64(n), *[float(v) for v in c],
                             out_nbytes=2 * n, out_dtype=np.uint16),
                want)
    return ld


def ld_fp8(op, fname, ncoeff):
    def ld(case, t):
        n = nelem(t)
        x = data(case, "ifm32.txt")[:n].copy()
        want = data(case, ofm(case))[:n]
        c = fwords(case, fname, ncoeff)
        ref = {"exp": ge.exp_fp8, "elu": ge.elu_fp8, "hardsigmoid": ge.hardsigmoid_fp8,
               "sqrt": ge.sqrt_fp8}[op]
        return (lambda: ref(x, *c),
                lambda: call(f"npu_elementwise_{op}_fp8", x, OUT, I64(n),
                             *[float(v) for v in c], out_nbytes=n, out_dtype=np.uint8),
                want)
    return ld


def qdq_coeff(case, op):
    w = words(case)
    fl = w.view(np.float32)
    if op == "exp":
        return [fl[k] for k in range(4)]
    if op == "elu":
        return [fl[k] for k in range(6)]
    if op == "hardsigmoid":
        return [fl[0], fl[1], int(w[2:3].view(np.int32)[0]), int(w[3:4].view(np.int32)[0])]
    # sqrt: 64 copies each of dq_zp, dq_scale, q_zp, q_scale
    return [fl[64], fl[0], fl[192], fl[128]]


QDQ_NAME = {"exp": "npu_elementwise_exp_a{}", "elu": "npu_elementwise_elu_a{}",
            "hardsigmoid": "npu_hardsigmoid_a{}", "sqrt": "npu_qdq_sqrt_a{}"}
QDQ_REF = {"exp": ge.exp_qdq, "elu": ge.elu_qdq, "hardsigmoid": ge.hardsigmoid_qdq,
           "sqrt": ge.sqrt_qdq}


def ld_qdq(op, bits):
    def ld(case, t):
        it, idt, _ = INT[t["ifm_dtype"]]
        ot, odt, ott = INT[t["ofm_dtype"]]
        n = nelem(t)
        x = data(case, "ifm32.txt", idt)[:n].copy()
        want = data(case, ofm(case), odt)[:n]
        c = qdq_coeff(case, op)
        args = [v if isinstance(v, int) else float(v) for v in c]
        return (lambda: QDQ_REF[op](x.astype(np.int64), ott, *c).astype(odt),
                lambda: call(QDQ_NAME[op].format(bits), x, it, OUT, ot, I64(n), *args,
                             out_nbytes=n * np.dtype(odt).itemsize, out_dtype=odt),
                want)
    return ld


LOADERS = {
    "exp_bf16_wrapper": ld_16("exp", 0),
    "elu_bf16_wrapper": ld_16("elu", 1),
    "hardsigmoid_bf16_wrapper": ld_16("hardsigmoid", 2),
    "sqrt_bf16_wrapper": ld_16("sqrt", 0),
    "exp_fp8_wrapper": ld_fp8("exp", "coeff32.txt", 2),
    "elu_fp8_wrapper": ld_fp8("elu", "wts32.txt", 3),
    "hardsigmoid_fp8_wrapper": ld_fp8("hardsigmoid", "wts32.txt", 3),
    "sqrt_fp8_wrapper": ld_fp8("sqrt", "coeff32.txt", 1),
    "qdq_elementwise_exp_a8_wrapper": ld_qdq("exp", 8),
    "qdq_elementwise_exp_a16_wrapper": ld_qdq("exp", 16),
    "qdq_elementwise_elu_a8_wrapper": ld_qdq("elu", 8),
    "qdq_elementwise_elu_a16_wrapper": ld_qdq("elu", 16),
    "qdq_hardsigmoid_a8_wrapper": ld_qdq("hardsigmoid", 8),
    "qdq_hardsigmoid_a16_wrapper": ld_qdq("hardsigmoid", 16),
    "qdq_sqrt_a8_wrapper": ld_qdq("sqrt", 8),
    "qdq_sqrt_a16_wrapper": ld_qdq("sqrt", 16),
}


def run(d, case):
    golden, entry, want = LOADERS[d](case, tc(case))
    assert_bytes(golden(), want)
    assert_bytes(entry(), want)


@pytest.mark.parametrize("d,case", cases(sorted(LOADERS)))
def test_fixture(d, case):
    run(d, case)
