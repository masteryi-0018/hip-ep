# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The QDQ conv, conv + add and depthwise entry points against the reference models of the
model repository."""

import re
import sys

import numpy as np
import pytest

import fixture as fm
from fixture import NPU_TYPE as T
from fixture import SHORT
from npu import I64, OUT, call
from testing import INT as OT

DIRS = ["conv_qdq_int8x8_wrapper", "conv_qdq_int8x8_channelwise_wrapper",
        "conv_qdq_int16x8_wrapper", "conv_qdq_int16x8_channelwise_wrapper",
        "depthwise_conv_a16w8_s1_wrapper", "depthwise_conv_a16w8_s2_wrapper",
        "depthwise_conv_a8w8_s1_wrapper", "depthwise_conv_a8w8_s2_wrapper",
        "qdq_conv_add_a16w8_wrapper", "qdq_conv_add_a8w8_channelwise_lrelu_add_wrapper",
        "qdq_conv_add_a8w8_channelwise_wrapper", "qdq_conv_add_a8w8_lrelu_add_wrapper",
        "qdq_conv_add_a8w8_wrapper", "qdq_conv_q_add_a8w8_channelwise_wrapper",
        "qdq_conv_q_add_a8w8_wrapper", "qdq_dwc_int16x8_wrapper", "qdq_dwc_int8x8_wrapper"]
HOOKS = {"QDQ_*Conv*": ("compute_bit_exact", "_qdq_conv_compute"),
         "QDQ_DWC*": ("compute_bit_exact",)}
# extract() stores the activation as an index (the cached tensors keep this encoding):
# 0 none, 1 CLIP (RELU / RELU6 / CLAMP), 2 LRELU, 3 PRELU; CODE maps it to the kernel code.
ACT_PRELU = 3
KIND = {0: 0, 1: 1, 2: 1, 8: 1, 100: 1, 3: 2, 9: ACT_PRELU}
CODE = {0: 0, 1: 100, 2: 3, 3: 9}


def extract(d, seen):
    """Logical operands, coefficients and expected output of the model object."""
    s = fm.last(seen)
    cls, data = type(s).__name__, s.data
    ofm = np.asarray(data.get("ofm_0", data.get("ofm")))
    act = int(s.activation_type)
    kind = KIND[act]
    ifm, wts = np.asarray(data["ifm"]), np.asarray(data["wts"])
    sh, sw = int(s.stride_height), int(s.stride_width)
    if "DepthwiseConv" in cls or "DWC" in cls:
        packed_dwc_coeff = sys.modules["qdq_dwc_exact"].packed_dwc_coeff
        qd = cls.startswith("QDQ_DWC")
        packed = s._packed_wts_svs()
        c0, c2, a0, a1 = packed_dwc_coeff(packed, s.ifm_sv_depth, 64 if qd else 32, act > 0)
        zp_w = int(np.frombuffer(packed[0][:1] if qd else packed[0][8:9], s.wts_dtype)[0])
        actv = int(bool(s.has_actv_sum)) if qd else 1
        fx = dict(c0=c0, c2=c2,
                  scal=np.array([sh, sw, zp_w, actv, kind]),
                  act=np.array([a0, a1], np.float32))
    elif "ConvAdd" in cls:
        co = wts.shape[3]
        clip = s.gen_clip_coeff()
        k_opt = int(sys.modules["qdq_conv_exact"].uses_k_opt(s))
        before = int(getattr(s, "act_before_add", 0))
        fx = dict(ifm2=np.asarray(data["ifm2"]), c0=fm.col(s.C0, co), c2=fm.col(s.C2, co),
                  c1=fm.col(s.C1, co) if s.has_actv_sum else None,
                  scal=np.array([sh, sw, k_opt, kind, before]),
                  act=np.array([clip[0], clip[1]], np.float32),
                  adds=np.array([s.add_c0, s.add_c1, s.add_c2], np.float32))
    elif "ConvQAdd" in cls:
        co = wts.shape[3]
        fx = dict(ifm2=np.asarray(data["ifm2"]), c0=fm.col(s.C0, co), c2=fm.col(s.C2, co),
                  scal=np.array([sh, sw]),
                  adds=np.array([s.add_c0, s.add_c1, s.add_c2], np.float32))
    else:
        _, ifm, wts, c2, c1, c0 = fm.args(seen, "_qdq_conv_compute")[:6]
        ifm, wts = np.asarray(ifm), np.asarray(wts)
        co = wts.shape[3]
        clip = s.gen_clip_coeff()
        k_opt = int(sys.modules["qdq_conv_exact"].uses_k_opt(s))
        fx = dict(c0=fm.col(c0, co),
                  c1=fm.col(c1, co) if s.has_actv_sum else None, c2=fm.col(c2, co),
                  scal=np.array([sh, sw, k_opt, kind]),
                  act=np.array([clip[0], clip[1]], np.float32),
                  slope=fm.col(data["prelu_slope"], co) if kind == ACT_PRELU else None)
    # the entry is named after the kernel the directory builds
    entry = re.sub("_channelwise|_lrelu_add", "", d[:-len("_wrapper")])
    fx.update(entry=entry, ifm=ifm, wts=wts, out_t=SHORT[np.dtype(s.ofm_dtype).name], ofm=ofm,
              cls=cls)
    return fx


@pytest.mark.parametrize("d,case", fm.cases(DIRS, __file__))
def test_fixture(d, case):
    z = fm.load(case)
    entry = str(z["entry"])
    ot, odt = OT[str(z["out_t"])]
    ifm, wts = z["ifm"], z["wts"]
    h, w = ifm.shape[:2]
    scal = [int(v) for v in z["scal"]]
    want = z["ofm"]
    if entry.startswith(("qdq_dwc_", "depthwise_conv_")):
        sh, sw, zp_w, actv, kind = scal
        kind = CODE[kind]
        kh, kw, c = wts.shape
        a0, a1 = (float(v) for v in z["act"])
        geom = (I64(kh), I64(kw), sh, sw)
        if entry.startswith("depthwise_conv_"):  # 3 x 3, the stride is in the name
            assert (kh, kw, sh, sw) == (3, 3, int(entry[-1]), int(entry[-1]))
            geom = ()
        sums = (actv,)
        if entry.startswith("depthwise_conv_a8w8"):  # the window sum is always applied
            assert actv == 1
            sums = ()
        args = (ifm, T[ifm.dtype.name], wts, T[wts.dtype.name], OUT, ot, I64(h), I64(w), I64(c),
                *geom, zp_w, *sums, z["c0"], z["c2"], kind, a0, a1)
    elif entry in ("qdq_conv_add_a8w8", "qdq_conv_add_a16w8"):
        sh, sw, k_opt, kind, before = scal
        kind = CODE[kind]
        kh, kw, ci, co = wts.shape
        a0, a1 = (float(v) for v in z["act"])
        c1 = z["c1"] if "c1" in z else None
        head = (ifm, T[ifm.dtype.name], wts, T[wts.dtype.name], z["ifm2"], T[z["ifm2"].dtype.name],
                OUT, ot, I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), sh, sw)
        args = head + (k_opt, z["c0"], c1, z["c2"], *(float(v) for v in z["adds"]), before, kind,
                       a0, a1)
    elif entry == "qdq_conv_q_add_a8w8":
        sh, sw = scal
        kh, kw, ci, co = wts.shape
        args = (ifm, T[ifm.dtype.name], wts, T[wts.dtype.name], z["ifm2"], T[z["ifm2"].dtype.name],
                OUT, ot, I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), sh, sw, z["c0"],
                z["c2"], *(float(v) for v in z["adds"]))
    elif entry in ("conv_qdq_int8x8", "conv_qdq_int16x8"):
        sh, sw, k_opt, kind = scal
        kind = CODE[kind]
        kh, kw, ci, co = wts.shape
        a0, a1 = (float(v) for v in z["act"])
        args = (ifm, T[ifm.dtype.name], wts, T[wts.dtype.name], OUT, ot, I64(h), I64(w), I64(ci),
                I64(kh), I64(kw), I64(co), sh, sw, k_opt, z["c0"], fm.opt(z, "c1"), z["c2"], kind,
                a0, a1, fm.opt(z, "slope"))
    else:
        raise AssertionError(f"unknown entry {entry}")
    got = call("npu_" + entry, *args, out_nbytes=want.size * np.dtype(odt).itemsize, out_dtype=odt)
    got, want = np.asarray(got).reshape(-1), want.astype(odt).reshape(-1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{entry}: {bad.size}/{want.size} differ, first at {bad[:4].tolist()}: "
                           f"got {got[bad[:4]].tolist()} want {want[bad[:4]].tolist()}")
