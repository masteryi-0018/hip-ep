# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""rope.h entry points against NPU reference outputs, byte for byte."""

import numpy as np
import pytest

from fixture import cases, data, ofm
from fixture import testcase as tc
from npu import I64, NPU_BF16, NPU_F16, NPU_F32, OUT
from npu import call as callx

WRAPPERS = ["rope_bf16_stream_wrapper", "rope_fp8_stream_wrapper", "rope_bfp16_stream_wrapper"]
CASES = cases(WRAPPERS)
T16 = {"bfloat16": NPU_BF16, "float16": NPU_F16}


def shape(t):
    return t["batch"] * t["num_heads"], t["seq_len"], t["head_dim"]


def logical(x, R, M, row):
    """[seq][rows][row] -> [rows][seq][row]."""
    return np.ascontiguousarray(x[:M * R * row].reshape(M, R, row).transpose(1, 0, 2))


def tables(case, M, K, dt, width):
    """[cos; sin], 2 * seq_len rows of `width` values; the first K / 2 of each
    row are returned."""
    w = data(case, "wts32.txt", dt)[:2 * M * width].reshape(2 * M, width)
    return (np.ascontiguousarray(w[:M, :K // 2]), np.ascontiguousarray(w[M:, :K // 2]))


def run_16(case, t):
    R, M, K = shape(t)
    x = logical(data(case, "ifm32.txt", np.uint16), R, M, K)
    c, s = tables(case, M, K, np.float32, K)
    got = callx("npu_rope_bf16_stream", x, T16[t["dtype"]], c, s, OUT, I64(R), I64(M), I64(K),
                int(t.get("rotary_emb_dim", 0)), out_nbytes=x.nbytes, out_dtype=np.uint16)
    return got, data(case, ofm(case), np.uint16)[:x.size]


def run_fp8(case, t):
    R, M, K = shape(t)
    x = logical(data(case, "ifm32.txt", np.uint8), R, M, K)
    c, s = tables(case, M, K, np.float32, K)
    got = callx("npu_rope_fp8_stream", x, c, s, OUT, I64(R), I64(M), I64(K), T16[t["dtype"]],
                int(t.get("rotary_emb_dim", 0)), float(np.float32(t["ifm_scale"])),
                float(np.float32(t["ofm_scale"])), out_nbytes=x.nbytes, out_dtype=np.uint8)
    return got, data(case, ofm(case), np.uint8)[:x.size]


def run_bfp16(case, t):
    R, M, K = shape(t)
    x = logical(data(case, "ifm32.txt", np.uint8), R, M, K // 8 * 9)
    wbf = t["wts_dtype"] == "bfloat16"
    c, s = tables(case, M, K, np.uint16 if wbf else np.float32, K // 2)
    got = callx("npu_rope_bfp16_stream", x, c, s, NPU_BF16 if wbf else NPU_F32, OUT, I64(R),
                I64(M), I64(K), int(t.get("interleave_mode", 0)), int(t.get("rotary_emb_dim", 0)),
                out_nbytes=x.nbytes,
                out_dtype=np.uint8)
    return got, data(case, ofm(case), np.uint8)[:x.size]


RUNNERS = {"rope_bf16_stream_wrapper": run_16, "rope_fp8_stream_wrapper": run_fp8,
           "rope_bfp16_stream_wrapper": run_bfp16}


@pytest.mark.skipif(not CASES, reason="no fixture cases")
@pytest.mark.parametrize("wrapper,case", CASES)
def test_fixture_rope(wrapper, case):
    t = tc(case)
    got, want = RUNNERS[wrapper](case, t)
    got = np.asarray(got).reshape(-1)
    want = np.ascontiguousarray(want).reshape(-1).astype(got.dtype)
    assert got.size == want.size, f"size {got.size} != {want.size}"
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} elements differ from sim, first at "
                           f"{bad[:8].tolist()}: got {got[bad[:8]].tolist()} "
                           f"want {want[bad[:8]].tolist()}")
