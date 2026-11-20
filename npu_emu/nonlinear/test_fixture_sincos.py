# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""sin / cos entry points and golden_sincos.py against NPU reference outputs."""

import numpy as np
import pytest

import golden_sincos as gs
from fixture import INT, assert_bytes, cases, data, ofm
from fixture import testcase as tc
from npu import I64, NPU_BF16, NPU_F16, OUT
from testing import call_n as go

NAT = {"bfloat16": (NPU_BF16, "bf16"), "float16": (NPU_F16, "f16")}


def nelem(t):
    return t["ifm_sv_size"] * t["ifm_sv_iteration"]


def fwords(case, name):
    return data(case, name, np.float32)


def ld_direct(case, t, is_cos):
    nt, ht = NAT[t["ifm_dtype"]]
    n = nelem(t)
    x = data(case, "ifm32.txt", np.uint16)[:n].copy()
    want = data(case, ofm(case), np.uint16)[:n]
    name = "npu_cos_bf16" if is_cos else "npu_sin_bf16"
    return (lambda: gs.sincos16(x, ht, is_cos),
            lambda: go(name, x, OUT, nt, I64(n), n=n, dt=np.uint16),
            want)


def ld_fp8(case, t, is_cos):
    nt, ht = NAT[t["compute_dtype"]]
    n = nelem(t)
    x = data(case, "ifm32.txt")[:n].copy()
    want = data(case, ofm(case))[:n]
    s, r = (float(v) for v in fwords(case, "coeff32.txt")[:2])
    name = f"npu_elementwise_{'cos' if is_cos else 'sin'}_fp8"
    return (lambda: gs.sincos_fp8(x, ht, is_cos, s, r),
            lambda: go(name, x, OUT, nt, I64(n), s, r, n=n, dt=np.uint8),
            want)


def ld_lut16(case, t, is_cos):
    nt, ht = NAT[t["ifm_dtype"]]
    n = nelem(t)
    x = data(case, "ifm32.txt", np.uint16)[:n].copy()
    want = data(case, ofm(case), np.uint16)[:n]
    w = fwords(case, "wts32.txt")[:gs.LUT_FLOATS].copy()
    name = f"npu_elementwise_{'cos' if is_cos else 'sin'}_lut_bf16"
    return (lambda: gs.lut16(x, w, ht),
            lambda: go(name, x, w, OUT, nt, I64(n), n=n, dt=np.uint16),
            want)


def ld_lut_fp8(case, t, is_cos):
    nt, ht = NAT[t["compute_dtype"]]
    n = nelem(t)
    x = data(case, "ifm32.txt")[:n].copy()
    want = data(case, ofm(case))[:n]
    wa = fwords(case, "wts32.txt")
    w = wa[:gs.LUT_FLOATS].copy()
    s, r = float(wa[2048]), float(wa[2049])
    name = f"npu_elementwise_{'cos' if is_cos else 'sin'}_lut_fp8"
    return (lambda: gs.lut_fp8(x, w, ht, s, r),
            lambda: go(name, x, w, OUT, nt, I64(n), s, r, n=n, dt=np.uint8),
            want)


def ld_qdq(case, t, is_cos, bits, lut):
    it, idt, _ = INT[t["ifm_dtype"]]
    ot, odt, ott = INT[t["ofm_dtype"]]
    n = nelem(t)
    x = data(case, "ifm32.txt", idt)[:n].copy()
    want = data(case, ofm(case), odt)[:n]
    fn = "cos" if is_cos else "sin"
    if lut:
        wa = fwords(case, "wts32.txt")
        w = wa[:gs.LUT_FLOATS].copy()
        izp, isc, ozp, orc = (float(wa[gs.LUT_FLOATS + 64 * k]) for k in range(4))
        return (lambda: gs.qdq_lut(x, w, ott, izp, isc, ozp, orc).astype(odt),
                lambda: go(f"npu_elementwise_{fn}_lut_a{bits}", x, it, w, OUT, ot, I64(n),
                           izp, isc, ozp, orc, n=n, dt=odt),
                want)
    ct, ht = NAT[t["compute_dtype"]]
    c = fwords(case, "coeff32.txt")
    izp, isc, ozp, orc = (float(c[64 * k]) for k in range(4))
    return (lambda: gs.qdq_sincos(x, ott, ht, is_cos, izp, isc, ozp, orc).astype(odt),
            lambda: go(f"npu_elementwise_{fn}_a{bits}", x, it, OUT, ot, ct, I64(n),
                       izp, isc, ozp, orc, n=n, dt=odt),
            want)


LOADERS = {}
for _fn, _cos in (("sin", False), ("cos", True)):
    LOADERS[f"{_fn}_bf16_wrapper"] = lambda c, t, k=_cos: ld_direct(c, t, k)
    LOADERS[f"{_fn}_fp8_wrapper"] = lambda c, t, k=_cos: ld_fp8(c, t, k)
    LOADERS[f"{_fn}_lut_bf16_wrapper"] = lambda c, t, k=_cos: ld_lut16(c, t, k)
    LOADERS[f"{_fn}_lut_fp8_wrapper"] = lambda c, t, k=_cos: ld_lut_fp8(c, t, k)
    for _b in (8, 16):
        LOADERS[f"qdq_elementwise_{_fn}_a{_b}_wrapper"] = \
            lambda c, t, k=_cos, b=_b: ld_qdq(c, t, k, b, False)
        LOADERS[f"qdq_elementwise_{_fn}_lut_a{_b}_wrapper"] = \
            lambda c, t, k=_cos, b=_b: ld_qdq(c, t, k, b, True)


def load(d, case):
    return LOADERS[d](case, tc(case))


@pytest.mark.parametrize("d,case", cases(sorted(LOADERS)))
def test_fixture(d, case):
    golden, entry, want = load(d, case)
    assert_bytes(golden(), want)
    assert_bytes(entry(), want)
