# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""reduction/softmax.h entry points and golden_softmax.py against NPU reference outputs."""

import numpy as np
import pytest

import golden_softmax as gs
from fixture import HALF as F16T
from fixture import INT, assert_bytes, cases, data, ofm, words
from fixture import testcase as tc
from npu import I64, OUT
from testing import call_n as go

def unblock(x, H, C, W, g):
    """[H][C/g][W][g] -> [H*W][C]."""
    return np.ascontiguousarray(x[:H * C * W].reshape(H, C // g, W, g)
                                .transpose(0, 2, 1, 3).reshape(H * W, C))


def mask_bits(m, C):
    """[rows][C/8] bytes (bit c % 8 of byte c / 8) -> bool [rows][C]."""
    return np.unpackbits(m, axis=-1, bitorder="little")[:, :C].astype(bool)


def fcoeff(case, k):
    return [float(v) for v in words(case)[:k].view(np.float32)]


# Each loader returns (golden(), entry(), want) on the logical region.

def ld_softmax_bf16(case, t):
    it, H, W, V = t["ifm_height_iteration"], t["ifm_sv_height"], t["ifm_sv_width"], t["valid_cols"]

    def tiles(name):
        x = data(case, name, np.uint16)[:it * H * W]
        x = x.reshape(it, W // 64, H, 64).transpose(0, 2, 1, 3).reshape(it * H, W)
        return np.ascontiguousarray(x[:, :V])

    x, want = tiles("ifm32.txt"), tiles(ofm(case))
    R = it * H
    return (lambda: gs.softmax_bf16(x),
            lambda: go("npu_softmax_bf16", x, OUT, I64(R), I64(V),
                       n=R * V, dt=np.uint16).reshape(R, V),
            want)


def ld_softmax_2pass_bf16(case, t):
    it, H, cc, nc = (t["ifm_height_iteration"], t["ifm_sv_height"], t["ifm_sv_width"],
                     t["ifm_width_iteration"])
    V = t["ifm_sv_width_real"]
    W = cc * nc

    def tiles(name):
        x = data(case, name, np.uint16)[:it * H * W]
        x = x.reshape(it, W // 64, H, 64).transpose(0, 2, 1, 3).reshape(it * H, W)
        return np.ascontiguousarray(x[:, :V])

    x, want = tiles("ifm32.txt"), tiles(ofm(case))
    R = it * H
    return (lambda: gs.softmax_2pass_bf16(x, cc, nc),
            lambda: go("npu_softmax_2pass_bf16", x, OUT, I64(R), I64(V), I64(cc), I64(nc),
                       n=R * V, dt=np.uint16).reshape(R, V),
            want)


def ld_softmax_fp16x16(case, t):
    nt, tt = F16T[t["dtype"]]
    H = t["ifm_sv_height"] * t["ifm_height_iteration"]
    W = t["ifm_sv_width"] * t["ifm_width_iteration"]
    C = t["ifm_sv_depth"]
    Cp = -(-C // 64) * 64
    x = unblock(data(case, "ifm32.txt", np.uint16), H, Cp, W, 64)[:, :C].copy()
    want = unblock(data(case, ofm(case), np.uint16), H, Cp, W, 64)[:, :C].copy()
    mw = unblock(data(case, "wts32.txt", np.uint16), H, Cp // 16, W, 4)[:, :C // 16]
    mb = np.ascontiguousarray(mw).view(np.uint8)
    R = H * W
    return (lambda: gs.softmax_fp16x16(x, mask_bits(mb, C), tt),
            lambda: go("npu_softmax_fp16x16", x, mb, OUT, nt, I64(R), I64(C),
                       n=R * C, dt=np.uint16).reshape(R, C),
            want)


def ld_softmax_fp8(case, t):
    H = t["ifm_sv_height"] * t["ifm_height_iteration"]
    W = t["ifm_sv_width"] * t["ifm_width_iteration"]
    C = t["ifm_sv_depth"] * t["ifm_depth_iteration"]
    assert t["ifm_depth_iteration"] == 1
    x = unblock(data(case, "ifm32.txt"), H, C, W, 64)
    want = unblock(data(case, ofm(case)), H, C, W, 64)
    mb = unblock(data(case, "wts32.txt"), H, C // 8, W, 8)
    s, r = fcoeff(case, 2)
    R = H * W
    return (lambda: gs.softmax_fp8(x, mask_bits(mb, C), s, r),
            lambda: go("npu_softmax_fp8", x, mb, OUT, I64(R), I64(C), s, r,
                       n=R * C, dt=np.uint8).reshape(R, C),
            want)


def ld_logsoftmax_bf16(case, t):
    nt, tt = F16T[t["dtype"]]
    R = t["ifm_height_iteration"] * t["ifm_sv_height"] * t["ifm_sv_width"]
    C = t["ifm_sv_depth"]
    V = t["valid_cols"] or C
    x = np.ascontiguousarray(data(case, "ifm32.txt", np.uint16)[:R * C].reshape(R, C)[:, :V])
    want = data(case, ofm(case), np.uint16)[:R * C].reshape(R, C)[:, :V]
    return (lambda: gs.logsoftmax_bf16(x, tt),
            lambda: go("npu_logsoftmax_bf16", x, OUT, nt, I64(R), I64(V),
                       n=R * V, dt=np.uint16).reshape(R, V),
            want)


def ld_logsoftmax_fp8(case, t):
    R = t["ifm_height_iteration"] * t["ifm_sv_height"] * t["ifm_sv_width"]
    C = t["ifm_sv_depth"]
    V = t["valid_cols"] or C
    x = np.ascontiguousarray(data(case, "ifm32.txt")[:R * C].reshape(R, C)[:, :V])
    want = data(case, ofm(case))[:R * C].reshape(R, C)[:, :V]
    s, r = fcoeff(case, 2)
    return (lambda: gs.logsoftmax_fp8(x, s, r),
            lambda: go("npu_logsoftmax_fp8", x, OUT, I64(R), I64(V), s, r,
                       n=R * V, dt=np.uint8).reshape(R, V),
            want)


def _qdq_shape(t):
    assert t["depth_iteration"] == 1
    return (t["sv_height"] * t["height_iteration"], t["sv_width"] * t["width_iteration"],
            t["sv_depth"])


def ld_qdq_softmax(case, t, bits):
    nt, dt, tt = INT[t["ifm_dtype"]]
    assert t["ofm_dtype"] == t["ifm_dtype"]
    H, W, C = _qdq_shape(t)
    x = unblock(data(case, "ifm32.txt", dt), H, C, W, 32)
    want = unblock(data(case, ofm(case), dt), H, C, W, 32)
    mb = unblock(data(case, "wts32.txt"), H, C // 8, W, 4)
    il, os_, zp = fcoeff(case, 3)
    path, sh = t["path"], t["shift_res"]
    R = H * W
    return (lambda: gs.qdq_softmax(x, mask_bits(mb, C), tt, path, sh, il, os_, zp).astype(dt),
            lambda: go(f"npu_qdq_softmax_a{bits}", x, mb, OUT, nt, I64(R), I64(C), path, sh,
                       il, os_, zp, n=R * C, dt=dt).reshape(R, C),
            want)


def ld_qdq_logsoftmax(case, t, bits):
    it, idt, itt = INT[t["ifm_dtype"]]
    ot, odt, ott = INT[t["ofm_dtype"]]
    H, W, C = _qdq_shape(t)
    x = unblock(data(case, "ifm32.txt", idt), H, C, W, 64)
    want = unblock(data(case, ofm(case), odt), H, C, W, 64)
    mb = unblock(data(case, "wts32.txt"), H, C // 8, W, 8)
    w = words(case)
    ozp = int(w[1:2].view(np.int32)[0])
    s_in, s_out = (float(v) for v in w[2:4].view(np.float32))
    R = H * W
    return (lambda: gs.qdq_logsoftmax(x, mask_bits(mb, C), itt, ott, s_in, s_out, ozp).astype(odt),
            lambda: go(f"npu_qdq_logsoftmax_a{bits}", x, it, mb, OUT, ot, I64(R), I64(C),
                       s_in, s_out, ozp, n=R * C, dt=odt).reshape(R, C),
            want)


LOADERS = {
    "softmax_bf16_wrapper": ld_softmax_bf16,
    "softmax_2pass_bf16_wrapper": ld_softmax_2pass_bf16,
    "softmax_fp16x16_wrapper": ld_softmax_fp16x16,
    "softmax_fp8_wrapper": ld_softmax_fp8,
    "logsoftmax_bf16_wrapper": ld_logsoftmax_bf16,
    "logsoftmax_fp8_wrapper": ld_logsoftmax_fp8,
    "qdq_softmax_a8_wrapper": lambda c, t: ld_qdq_softmax(c, t, 8),
    "qdq_softmax_a16_wrapper": lambda c, t: ld_qdq_softmax(c, t, 16),
    "qdq_logsoftmax_a8_wrapper": lambda c, t: ld_qdq_logsoftmax(c, t, 8),
    "qdq_logsoftmax_a16_wrapper": lambda c, t: ld_qdq_logsoftmax(c, t, 16),
}


def load(d, case):
    return LOADERS[d](case, tc(case))


@pytest.mark.parametrize("d,case", cases(sorted(LOADERS)))
def test_fixture(d, case):
    golden, entry, want = load(d, case)
    assert_bytes(golden(), want)
    assert_bytes(entry(), want)
