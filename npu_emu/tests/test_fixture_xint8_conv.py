# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The xint8 convolution entry points against fixtures exported from mllib's
golden runs (NPU_EMU_XINT8_FIXTURES, one .npz per nightly case): the entry,
its logical tensors and the shift / clamp / alpha words the kernel reads, and
the golden output. Only cases whose golden ofm32_ref.txt is byte-identical to
the nightly aiesim ofm32.txt are exported, so matching the fixture output
matches aiesim. Exported by logs/class7c/fix_xint8.py (mllib PR #15123)."""

import glob
import os

import numpy as np
import pytest

from npu import I64, NPU_I8, NPU_U8, OUT, call

ROOT = os.environ.get("NPU_EMU_XINT8_FIXTURES",
                      "/wrk/xcohdnobkup3/yigao102/shared/npu-emu/fixtures/xint8_conv")
T = {"int8": NPU_I8, "uint8": NPU_U8}


def fixtures():
    return [pytest.param(f, id=os.path.relpath(f, ROOT)[:-4]) for f in sorted(glob.glob(f"{ROOT}/*/*.npz"))]


def opt(z, k):
    return z[k] if k in z else None


@pytest.mark.parametrize("path", fixtures())
def test_fixture(path):
    z = np.load(path, allow_pickle=False)
    entry, out_dt = str(z["entry"]), str(z["out_dt"])
    ifm, wts = z["ifm"], z["wts"]
    sh, sw = (int(v) for v in z["stride"])
    scal = [int(v) for v in z["scal"]]
    want = z["ofm"]
    h, w = ifm.shape[:2]
    if entry == "xint8_conv":
        kh, kw, ci, co = wts.shape
        args = (ifm, T[ifm.dtype.name], wts, opt(z, "bias"), OUT, T[out_dt], I64(h), I64(w),
                I64(ci), I64(kh), I64(kw), I64(co), sh, sw, *scal)
    elif entry == "xint8_depthwise_conv":
        kh, kw, co = wts.shape
        args = (ifm, T[ifm.dtype.name], wts, T[wts.dtype.name], opt(z, "bias"), OUT, T[out_dt],
                I64(h), I64(w), I64(co), I64(kh), I64(kw), sh, sw, *scal)
    elif entry == "xint8_dwc3x3_conv1x1":
        kh, kw, ci = wts.shape
        co = z["pw_wts"].shape[1]
        args = (ifm, wts, opt(z, "bias"), z["pw_wts"], opt(z, "pw_bias"), OUT, I64(h), I64(w),
                I64(ci), I64(co), I64(kh), I64(kw), sh, sw, *scal)
    elif entry == "xint8_conv_add":
        kh, kw, ci, co = wts.shape
        args = (ifm, wts, opt(z, "bias"), z["ifm2"], opt(z, "x0"), OUT, T[out_dt], I64(h), I64(w),
                I64(ci), I64(kh), I64(kw), I64(co), sh, sw, *scal)
    elif entry == "qxint8_conv":
        kh, kw, ci, co = wts.shape
        args = (ifm, T[ifm.dtype.name], wts, z["bias"], OUT, T[out_dt], I64(h), I64(w), I64(ci),
                I64(kh), I64(kw), I64(co), sh, sw, *scal)
    else:
        raise AssertionError(f"unknown entry {entry}")
    dt = np.dtype(out_dt)
    got = call("npu_" + entry, *args, out_nbytes=want.size * dt.itemsize, out_dtype=dt)
    got, want = np.asarray(got).reshape(-1), want.astype(dt).reshape(-1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{entry}: {bad.size}/{want.size} differ, first at {bad[:4].tolist()}: "
                           f"got {got[bad[:4]].tolist()} want {want[bad[:4]].tolist()}")
