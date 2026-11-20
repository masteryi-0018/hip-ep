# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""SigTanh / SiLU entry points against NPU reference outputs."""

import numpy as np
import pytest

from fixture import NPU_TYPE as HALF
from fixture import TYPES as INT
from fixture import assert_bytes, cases, data, ofm, tensor_descr, words
from fixture import testcase as tc
from npu import I64, OUT, call


def size(t):
    return t["ifm_sv_size"] * t["ifm_sv_iteration"]


def coeffs(case, idx):
    w = words(case, "coeff32.txt").view(np.float32)
    return [float(w[i]) for i in idx]


def valid(case, n):
    td = tensor_descr(case)
    if td is not None:
        assert int(np.prod(td[0])) == n, f"valid shape {td[0]} != {n} elements"
    return n


def run_sigtanh_bf16(case, t):
    n = size(t)
    x = data(case, "ifm32.txt", np.uint16)[:n]
    got = call("npu_sigtanh_bf16", x, OUT, HALF[t["ifm_dtype"]], I64(n), t["act_func"],
               out_nbytes=2 * n, out_dtype=np.uint16)
    return got, data(case, ofm(case), np.uint16)[:n]


def run_silu_bf16(case, t):
    n = size(t)
    x = data(case, "ifm32.txt", np.uint16)[:n]
    got = call("npu_silu_bf16", x, OUT, HALF[t["ifm_dtype"]], I64(n), out_nbytes=2 * n,
               out_dtype=np.uint16)
    return got, data(case, ofm(case), np.uint16)[:n]


def run_sigtanh_bfp16(case, t):
    n = valid(case, size(t))
    nb = n // 8 * 9
    got = call("npu_sigtanh_bfp16", data(case, "ifm32.txt")[:nb], OUT, I64(n), t["act_func"],
               out_nbytes=nb, out_dtype=np.uint8)
    return got, data(case, ofm(case))[:nb]


def run_silu_bfp16(case, t):
    n = valid(case, size(t))
    nb = n // 8 * 9
    got = call("npu_silu_bfp16", data(case, "ifm32.txt")[:nb], OUT, I64(n), out_nbytes=nb,
               out_dtype=np.uint8)
    return got, data(case, ofm(case))[:nb]


def run_sigtanh_fp8(case, t):
    n = valid(case, size(t))
    k1, k2, r = coeffs(case, (0, 1, 2))
    got = call("npu_elementwise_sigtanh_fp8", data(case, "ifm32.txt")[:n], OUT, I64(n),
               t["act_func"], k1, k2, r, out_nbytes=n, out_dtype=np.uint8)
    return got, data(case, ofm(case))[:n]


def run_silu_fp8(case, t):
    n = valid(case, size(t))
    s, r = coeffs(case, (0, 1))
    got = call("npu_elementwise_silu_fp8", data(case, "ifm32.txt")[:n], OUT, I64(n),
               HALF[t["compute_dtype"]], s, r, out_nbytes=n, out_dtype=np.uint8)
    return got, data(case, ofm(case))[:n]


def qdq_runner(name, sigtanh):
    def run(case, t):
        n = size(t)
        ti, to = INT[t["ifm_dtype"]], INT[t["ofm_dtype"]]
        x = data(case, "ifm32.txt", ti[1])[:n]
        dq_zp, dq_s, q_zp, q_s = coeffs(case, (0, 64, 128, 192))
        extra = [t["act_func"]] if sigtanh else []
        poly = coeffs(case, (256, 257, 258)) if sigtanh else []
        got = call(name, x, ti[0], OUT, to[0], I64(n), *extra, HALF[t["compute_dtype"]], dq_zp,
                   dq_s, q_zp, q_s, *poly, out_nbytes=n * np.dtype(to[1]).itemsize,
                   out_dtype=to[1])
        return got, data(case, ofm(case), to[1])[:n]
    return run


RUNNERS = {
    "sigtanh_bf16_wrapper": run_sigtanh_bf16,
    "silu_bf16_wrapper": run_silu_bf16,
    "sigtanh_bfp16_wrapper": run_sigtanh_bfp16,
    "silu_bfp16_wrapper": run_silu_bfp16,
    "sigtanh_fp8_wrapper": run_sigtanh_fp8,
    "silu_fp8_wrapper": run_silu_fp8,
    "qdq_sigtanh_a8_wrapper": qdq_runner("npu_qdq_sigtanh_a8", True),
    "qdq_sigtanh_a16_wrapper": qdq_runner("npu_qdq_sigtanh_a16", True),
    "qdq_silu_a8_wrapper": qdq_runner("npu_qdq_silu_exp2_a8", False),
    "qdq_silu_a16_wrapper": qdq_runner("npu_qdq_silu_exp2_a16", False),
}


@pytest.mark.parametrize("d,case", cases(sorted(RUNNERS)))
def test_fixture_silu(d, case):
    t = tc(case)
    got, want = RUNNERS[d](case, t)
    assert_bytes(got, want)
