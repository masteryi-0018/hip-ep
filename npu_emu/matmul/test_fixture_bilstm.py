# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""npu_bidirectional_lstm_bf16 against the reference model of the model repository."""

import sys

import numpy as np
import pytest

import fixture as fm
from npu import I64, OUT, OUT2, callx

DIRS = ["bidirectional_lstm_bf16_wrapper"]
HOOKS = {"BidirectionalLSTMBf16Wrapper": ("compute_bit_exact",)}


def extract(d, seen):
    """Logical operands (bf16 bits), the table and expected output of the model object."""
    s = fm.last(seen)
    m, data = sys.modules[type(s).__module__], s.data
    hid, n_in = int(s.hidden_size), int(s.ifm_sv_input_size)
    seq = int(s.ifm_sv_seq) * int(s.ifm_seq_iteration)
    bits = m.bf16_bits
    want = bits(data["ofm"])
    return dict(x=bits(data["ifm"].reshape(seq, n_in)), h0=bits(data["ifm3"][:hid]),
                c0=bits(data["ifm3"][hid:]), w=bits(data["W"]), r=bits(data["R"]), b=bits(data["B"]),
                lut=np.asarray(m.lut_words(), np.uint32).view(np.float32),
                dims=np.array([seq, n_in, hid, int(s.reverse)]), y=want[:seq * hid],
                c_out=want[seq * hid:])


@pytest.mark.parametrize("d,case", fm.cases(DIRS, __file__))
def test_fixture(d, case):
    z = fm.load(case)
    seq, n_in, hid, rev = (int(v) for v in z["dims"])
    gy, gc = callx("npu_bidirectional_lstm_bf16", z["x"], z["h0"], z["c0"], z["w"], z["r"],
                   z["b"], z["lut"], OUT, OUT2, I64(seq), I64(n_in), I64(hid), rev,
                   outs=[2 * seq * hid, 2 * hid])
    np.testing.assert_array_equal(gy.view(np.uint16), z["y"])
    np.testing.assert_array_equal(gc.view(np.uint16), z["c_out"])
