# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The LSTM and grpb entry points against the reference models of the model repository."""

import numpy as np
import pytest

import fixture as fm
from fixture import NPU_TYPE as T
from npu import I64, NPU_BF16, NPU_F16, OUT, OUT2, call, callx

DIRS = ["grpb_a16w8_wrapper", "grpb_a8w8_wrapper", "qdq_lstm_a16w8_wrapper",
        "qdq_lstm_a8w8_wrapper"]
HOOKS = {"QDQ_LSTM*": ("compute_bit_exact",), "QDQ_GRPB*": ("compute_bit_exact",)}


def extract(d, seen):
    """Logical operands, coefficients and expected output of the model object."""
    s = fm.last(seen)
    data = s.data
    if hasattr(s, "ifm_sv_hidden"):
        a16 = s.ifm_dtype == "uint16"
        x, h, c = (np.asarray(data[k]).reshape(-1) for k in ("x", "h0", "c0"))
        w, r = np.asarray(data["w"]), np.asarray(data["r"])
        b = np.asarray(s._correct_bias_full(np.asarray(data["b"]), w, r))
        dt = (np.uint16, np.uint8, np.uint32) if a16 else (np.int8, np.int8, np.int32)
        return dict(entry=d[len("qdq_"):-len("_wrapper")], cls=type(s).__name__,
                    x=x[:s.ifm_sv_x].astype(dt[0]), h=h.astype(dt[0]), c=c.astype(dt[0]),
                    w=w.astype(dt[1]), r=r.astype(dt[1]), b=b.astype(dt[2]),
                    qdq=np.asarray(s._prepare_qdq_params(), np.uint32)[:9],
                    h_out=np.asarray(data["hout"]).reshape(-1).astype(dt[0]),
                    c_out=np.asarray(data["cout"]).reshape(-1).astype(dt[0]))
    coeff = np.asarray(s.compute_qdq_param(), np.float32)
    f16 = s.ofm_dtype == "float16"
    out = np.asarray(data["ofm_0"], np.float32).reshape(-1)
    return dict(entry=d[:-len("_wrapper")], cls=type(s).__name__,
                ifm=np.asarray(data["matA"]).astype(fm.TYPES[s.ifm_dtype][1]),
                ifm_t=s.ifm_dtype, wts=np.asarray(data["matB"]).astype(fm.TYPES[s.wts_dtype][1]),
                wts_t=s.wts_dtype, c0=coeff[0], c1=coeff[1] if s.has_actv_sum == 1 else None,
                c2=coeff[-1], sc=np.array([s.scalar_head, s.scalar_lyr0, s.scalar_lyr1], np.float32),
                ofm="f16" if f16 else "bf16",
                out=out.astype(np.float16).view(np.uint16) if f16 else
                fm.bf16_bits(out))


@pytest.mark.parametrize("d,case", fm.cases(DIRS, __file__))
def test_fixture(d, case):
    z = fm.load(case)
    entry = str(z["entry"])
    if entry in ("grpb_a8w8", "grpb_a16w8"):
        m, k = z["ifm"].shape
        head, lyr0, lyr1 = (float(v) for v in z["sc"])
        got = call("npu_" + entry, z["ifm"], T[str(z["ifm_t"])], z["wts"], T[str(z["wts_t"])],
                   z["c0"], z["c1"] if "c1" in z else None, z["c2"], OUT,
                   NPU_F16 if str(z["ofm"]) == "f16" else NPU_BF16, I64(m), I64(k), head, lyr0,
                   lyr1, out_nbytes=2 * m, out_dtype=np.uint16)
        pairs = [(got, z["out"], "out")]
    else:
        lx, lh = z["x"].size, z["h"].size
        nb = z["h_out"].nbytes
        gh, gc = callx("npu_" + entry, z["x"], z["h"], z["c"], z["w"], z["r"], z["b"], z["qdq"], OUT, OUT2,
                       I64(lx), I64(lh), outs=[nb, nb])
        pairs = [(gh.view(z["h_out"].dtype), z["h_out"], "h"), (gc.view(z["c_out"].dtype), z["c_out"], "c")]
    for got, want, what in pairs:
        bad = np.nonzero(np.asarray(got).reshape(-1) != want.reshape(-1))[0]
        assert bad.size == 0, f"{z['cls']} {what}: {bad.size}/{want.size} differ, first at {bad[:8].tolist()}"
