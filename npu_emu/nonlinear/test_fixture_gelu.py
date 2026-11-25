# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""GeLU / Erf entry points and golden_gelu.py against NPU reference outputs."""

import numpy as np
import pytest

import golden_gelu as gg
from fixture import HALF as F16T
from fixture import INT, assert_bytes, cases, data, ofm, words
from fixture import testcase as tc
from npu import OUT, I64, call

DIRS = ["gelu_bf16_wrapper", "gelu_bfp16_wrapper", "gelu_fp8_wrapper", "qdq_gelu_a8_wrapper",
        "qdq_gelu_a16_wrapper", "xint8_gelu_wrapper", "erf_bf16_wrapper", "erf_fp8_wrapper",
        "qdq_elementwise_erf_a8_wrapper", "qdq_elementwise_erf_a16_wrapper"]


def spec(d, c):
    """(entry, args, out_nbytes, reference bytes, sim bytes) of one case."""
    j = tc(c)
    n = int(j["ifm_sv_size"]) * int(j.get("ifm_sv_iteration", 1))
    want = data(c, ofm(c))
    if d in ("gelu_bf16_wrapper", "erf_bf16_wrapper"):
        t, tn = F16T[j["ifm_dtype"]]
        x = data(c, "ifm32.txt", np.uint16)[:n]
        if d == "gelu_bf16_wrapper":
            return ("npu_gelu_bf16", [x, t, OUT, I64(n)], 2 * n, gg.gelu_bf16(x, tn), want)
        return ("npu_elementwise_erf_bf16", [x, t, OUT, I64(n)], 2 * n,
                gg.elementwise_erf_bf16(x, tn), want)
    if d == "gelu_bfp16_wrapper":
        x = data(c, "ifm32.txt")[:n * 9 // 8]
        return ("npu_gelu_bfp16", [x, OUT, I64(n)], n * 9 // 8, gg.gelu_bfp16(x), want)
    if d == "gelu_fp8_wrapper":
        x = data(c, "ifm32.txt")[:n]
        w = words(c).view(np.float32)
        t, tn = F16T[j["compute_dtype"]]
        return ("npu_elementwise_gelu_fp8", [x, OUT, I64(n), t, float(w[0]), float(w[1])], n,
                gg.elementwise_gelu_fp8(x, tn, w[0], w[1]), want)
    if d == "erf_fp8_wrapper":
        x = data(c, "ifm32.txt")[:n]
        w = words(c).view(np.float32)[:4]
        return ("npu_elementwise_erf_fp8", [x, OUT, I64(n)] + [float(v) for v in w], n,
                gg.elementwise_erf_fp8(x, *w), want)
    if d.startswith("qdq_elementwise_erf"):
        ti, tdi, _ = INT[j["ifm_dtype"]]
        to, tdo, tno = INT[j["ofm_dtype"]]
        x = data(c, "ifm32.txt", tdi)[:n]
        w = words(c).view(np.float32)[:4]
        name = "npu_elementwise_erf_a8" if "a8" in d else "npu_elementwise_erf_a16"
        return (name, [x, ti, OUT, to, I64(n)] + [float(v) for v in w],
                n * np.dtype(tdo).itemsize, gg.elementwise_erf_a8(x, *w, tno).astype(tdo), want)
    if d.startswith("qdq_gelu"):
        ti, tdi, _ = INT[j["ifm_dtype"]]
        to, tdo, tno = INT[j["ofm_dtype"]]
        x = data(c, "ifm32.txt", tdi)[:n]
        w = words(c)
        zi = int(w[0:1].view(np.int32)[0])
        zo, ka, cc, kdsp = (float(v) for v in w[1:5].view(np.float32))
        name = "npu_qdq_gelu_exp2_a8" if "a8" in d else "npu_qdq_gelu_exp2_a16"
        return (name, [x, ti, OUT, to, I64(n), zi, zo, ka, cc, kdsp],
                n * np.dtype(tdo).itemsize,
                getattr(gg, name[4:])(x, zi, zo, ka, cc, kdsp, tno).astype(tdo), want)
    if d == "xint8_gelu_wrapper":
        p = {k.split(".")[-1]: int(v) for k, v in j.items() if k.startswith("KernelGeLUParam.")}
        x = data(c, "ifm32.txt", np.int8)[:n]
        args = [p["up_path"], p["rnd_id"], p["a_shift"], p["b_shift"], p["q4_shift"],
                p["r_shift"], p["K"], p["P"], int(j.get("precision_enhanced", 0))]
        return ("npu_xint8_gelu", [x, OUT, I64(n)] + args, n, gg.xint8_gelu(x, *args), want)
    raise ValueError(d)


@pytest.mark.parametrize("d,c", cases(DIRS))
def test_fixture_gelu(d, c):
    name, args, nb, gold, want = spec(d, c)
    got = call(name, *args, out_nbytes=nb, out_dtype=np.uint8)
    want = np.asarray(want).view(np.uint8)[:nb]
    assert_bytes(np.asarray(gold).view(np.uint8), want)
    assert_bytes(got, want)
