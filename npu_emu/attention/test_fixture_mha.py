# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""npu_mha_bf16_3strms and golden_mha.py against NPU reference outputs, byte for byte."""

import numpy as np
import pytest

import golden_mha as gm
from fixture import assert_bytes, cases, data, ofm
from fixture import testcase as tc
from npu import I64, OUT, U32, Host, call, raw_rc


CASES = cases(["mha_bf16_3strms_wrapper"])


def load(case):
    t = tc(case)
    B, H, Hk, M, D, N = (int(t[k]) for k in ("batch_size", "q_num_heads", "kv_num_heads",
                                             "q_sequence_length", "head_size", "kv_sequence_length"))
    q = data(case, "matQ.txt", np.uint16)[:B * H * M * D].reshape(B, H, M, D)
    k = data(case, "matK.txt", np.uint16)[:B * Hk * N * D].reshape(B, Hk, N, D)
    v = data(case, "matV.txt", np.uint16)[:B * Hk * N * D].reshape(B, Hk, N, D)
    want = data(case, ofm(case), np.uint16)[:B * M * H * D].reshape(B, M, H, D)
    sv_n = int(t.get("sv_n", N))
    mask_1d = int(t.get("mask_1d", 0))
    start = int(t.get("mask_M_start", N))
    scale = int(t.get("scaling_factor", np.array([1 / np.sqrt(D)], np.float32).view(np.uint32)[0]))
    return dict(q=q, k=k, v=v, want=want, B=B, H=H, Hk=Hk, M=M, D=D, N=N, sv_n=sv_n,
                mask_1d=mask_1d, start=start, scale=scale)


def entry_args(c):
    return (c["q"], c["k"], c["v"], OUT, I64(c["B"]), I64(c["H"]), I64(c["Hk"]), I64(c["M"]),
            I64(c["D"]), I64(c["N"]), I64(c["sv_n"]), c["mask_1d"], U32(c["start"]),
            U32(c["scale"]))


@pytest.mark.parametrize("d,case", CASES)
def test_entry(d, case):
    c = load(case)
    if c["sv_n"] == 64:
        out = Host(np.zeros(1, np.uint16))  # never written: the check fails first
        assert raw_rc("npu_mha_bf16_3strms", *[out if a is OUT else a for a in entry_args(c)]) == -1
        pytest.skip("sv_n = 64 is rejected")
    got = call("npu_mha_bf16_3strms", *entry_args(c), out_nbytes=c["want"].nbytes,
               out_dtype=np.uint16)
    assert_bytes(got, c["want"])


@pytest.mark.parametrize("d,case", CASES)
def test_golden(d, case):
    c = load(case)
    if c["sv_n"] == 64:
        pytest.skip("sv_n = 64 is not supported")
    got = gm.mha(c["q"], c["k"], c["v"], c["sv_n"], c["scale"], c["mask_1d"], c["start"])
    assert_bytes(got, c["want"])
