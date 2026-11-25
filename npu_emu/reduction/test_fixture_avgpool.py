# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""reduction/avgpool.h entry points against NPU reference outputs, byte for byte."""

import os

import numpy as np
import pytest

from fixture import NPU_TYPE as HALF
from fixture import assert_bytes, cases, data, ofm, words
from fixture import TYPES as INT
from fixture import testcase as tc
from npu import I64, OUT, call

def s3(H, W, C):
    return I64(H), I64(W), I64(C)


def f32(x):
    return float(np.float32(x))


def from_blocks(x, H, W, C, gc=64):
    """[H][C / gc][W][gc] -> [H][W][C]."""
    return np.ascontiguousarray(
        x[:H * W * C].reshape(H, C // gc, W, gc).transpose(0, 2, 1, 3).reshape(H, W, C))


def window_geom(t):
    """(H, W, C) of the input, (Ho, Wo) of the output, window and strides."""
    Ho = t["ofm_sv_height"] * t["ofm_height_iteration"]
    Wo = t["ofm_sv_width"] * t["ofm_width_iteration"]
    C = t["ofm_sv_depth"] * t["ifm_depth_iteration"]
    kh, kw, sh, sw = t["kernel_height"], t["kernel_width"], t["stride_height"], t["stride_width"]
    return ((Ho - 1) * sh + kh, (Wo - 1) * sw + kw, C), (Ho, Wo), (kh, kw, sh, sw)


def global_geom(t):
    return (t["ifm_sv_height"] * t["ifm_height_iteration"],
            t["ifm_sv_width"] * t["ifm_width_iteration"],
            t["ofm_sv_depth"] * t["ifm_depth_iteration"])


def coeff_words(case, k):
    """The first k coefficient words as floats and the int16 shift in word 3."""
    name = "wts32.txt" if os.path.exists(os.path.join(case, "data", "wts32.txt")) else "coeff32.txt"
    w = words(case, name)
    return [float(v) for v in w[:k].view(np.float32)], int(w[3:4].view(np.int16)[0])


# Runners return (got, want).

def run_avgpool_bf16(case, t):
    (H, W, C), (Ho, Wo), win = window_geom(t)
    x = from_blocks(data(case, "ifm32.txt", np.uint16), H, W, C)
    want = from_blocks(data(case, ofm(case), np.uint16), Ho, Wo, C)
    got = call("npu_avgpool_bf16", x, HALF[t["ifm_dtype"]], OUT, *s3(H, W, C), *win,
               int(t["factor_raw"]),
               out_nbytes=2 * want.size, out_dtype=np.uint16)
    return got, want


def run_avgpool_bf16_c32(case, t):
    (H, W, C), (Ho, Wo), win = window_geom(t)
    x = from_blocks(data(case, "ifm32.txt", np.uint16), H, W, C, 32)
    want = from_blocks(data(case, ofm(case), np.uint16), Ho, Wo, C, 32)
    got = call("npu_avgpool_bf16_c32", x, OUT, *s3(H, W, C), *win, int(t["factor_raw"]),
               out_nbytes=2 * want.size, out_dtype=np.uint16)
    return got, want


def fp8_coeffs(case, t, C):
    """(c0[C], c2[C]) from the scalar parameters or the per-channel stream."""
    if t["vector_coeff"] == 0:
        return np.full(C, f32(t["quant_c0"]), np.float32), np.full(C, f32(t["quant_c2"]),
                                                                   np.float32)
    gc = 64 // t["granX"]
    blk = words(case, "wts32.txt").view(np.float32).reshape(-1, 128)
    assert blk.shape[0] == C // gc
    return (np.ascontiguousarray(blk[:, :gc]).reshape(-1),
            np.ascontiguousarray(blk[:, 64:64 + gc]).reshape(-1))


def run_avgpool_fp8(case, t):
    (H, W, C), (Ho, Wo), (kh, kw, sh, sw) = window_geom(t)
    gc = 64 // t["granX"]
    x = from_blocks(data(case, "ifm32.txt"), H, W, C, gc)
    want = from_blocks(data(case, ofm(case)), Ho, Wo, C, gc)
    c0, c2 = fp8_coeffs(case, t, C)
    kh_sv = t["Ky_g"]
    assert kh_sv * t["psum_iter"] == kh
    got = call("npu_avgpool_fp8", x, OUT, *s3(H, W, C), kh, kw, sh, sw, kh_sv, c0, c2,
               t["activation_type"],
               f32(t["act_lo"]), f32(t["act_hi"]), out_nbytes=want.size, out_dtype=np.uint8)
    return got, want


def run_globalavgpool_bf16(case, t):
    H, W, C = global_geom(t)
    x = from_blocks(data(case, "ifm32.txt", np.uint16), H, W, C)
    want = data(case, ofm(case), np.uint16)[:C]
    got = call("npu_globalavgpool_bf16", x, HALF[t["ifm_dtype"]], OUT, *s3(H, W, C),
               t["ifm_sv_height"], t["ifm_sv_width"], f32(t["factor_val"]),
               out_nbytes=2 * C, out_dtype=np.uint16)
    return got, want


def run_globalavgpool_bf16_c32(case, t):
    H, W, C = global_geom(t)
    x = from_blocks(data(case, "ifm32.txt", np.uint16), H, W, C, 32)
    want = data(case, ofm(case), np.uint16)[:C]
    got = call("npu_globalavgpool_bf16_c32", x, OUT, *s3(H, W, C), t["ifm_sv_height"],
               t["ifm_sv_width"], f32(t["factor"]), out_nbytes=2 * C, out_dtype=np.uint16)
    return got, want


def run_globalavgpool_fp8(case, t):
    H, W, C = global_geom(t)
    x = from_blocks(data(case, "ifm32.txt"), H, W, C)
    want = data(case, ofm(case))[:C]
    (c0, c1), _ = coeff_words(case, 2)
    got = call("npu_globalavgpool_fp8", x, OUT, *s3(H, W, C), t["ifm_sv_height"],
               t["ifm_sv_width"], c0, c1, out_nbytes=C, out_dtype=np.uint8)
    return got, want


def qdq_type(t):
    assert t.get("ofm_dtype", t["ifm_dtype"]) == t["ifm_dtype"]
    return INT[t["ifm_dtype"]]


def run_qdq_avgpool(case, t, bits):
    (H, W, C), (Ho, Wo), win = window_geom(t)
    it, dt = qdq_type(t)
    gc = 64 // t.get("granX", 1)
    x = from_blocks(data(case, "ifm32.txt", dt), H, W, C, gc)
    want = from_blocks(data(case, ofm(case), dt), Ho, Wo, C, gc)
    nb = want.size * np.dtype(dt).itemsize
    if bits == 8:
        (c0, c1, _, _, act_c0, act_c1), shift_res = coeff_words(case, 6)
        got = call("npu_qdq_avgpool_a8", x, it, OUT, *s3(H, W, C), *win, c0, c1, shift_res,
                   t.get("activation_type", 0), act_c0, act_c1, out_nbytes=nb, out_dtype=dt)
    else:
        (c0, c1), shift_res = coeff_words(case, 2)
        got = call("npu_qdq_avgpool_a16", x, it, OUT, *s3(H, W, C), *win, c0, c1, shift_res,
                   out_nbytes=nb, out_dtype=dt)
    return got, want


def run_qdq_global(case, t, bits):
    H, W, C = global_geom(t)
    it, dt = qdq_type(t)
    x = from_blocks(data(case, "ifm32.txt", dt), H, W, C)
    want = data(case, ofm(case), dt)[:C]
    (c0, c1), shift_res = coeff_words(case, 2)
    got = call(f"npu_qdq_globalavgpool_a{bits}", x, it, OUT, *s3(H, W, C), c0, c1, shift_res,
               out_nbytes=C * np.dtype(dt).itemsize, out_dtype=dt)
    return got, want


def run_xint8(case, t, fused):
    (H, W, C), (Ho, Wo), (kh, kw, sh, sw) = window_geom(t)
    it, dt = qdq_type(t)
    x = from_blocks(data(case, "ifm32.txt", dt), H, W, C)
    want = from_blocks(data(case, ofm(case), dt), Ho, Wo, C)
    wts = data(case, "wts32.txt", np.uint16)  # one equal weight per output position
    args = [x, it, OUT, *s3(H, W, C), kh, kw, sh, sw, int(wts[0]), t["rtp_shift_res"]]
    if fused:
        args.append(t.get("activation_type", 0))
    got = call("npu_avgpool_xint8_fused" if fused else "npu_avgpool_xint8", *args,
               out_nbytes=want.size, out_dtype=dt)
    return got, want


RUNNERS = {
    "avgpool_bf16_wrapper": run_avgpool_bf16,
    "avgpool_bf16_c32_wrapper": run_avgpool_bf16_c32,
    "avgpool_fp8_wrapper": run_avgpool_fp8,
    "globalavgpool_bf16_wrapper": run_globalavgpool_bf16,
    "globalavgpool_bf16_c32_wrapper": run_globalavgpool_bf16_c32,
    "globalavgpool_fp8_wrapper": run_globalavgpool_fp8,
    "qdq_avgpool_a8_wrapper": lambda c, t: run_qdq_avgpool(c, t, 8),
    "qdq_avgpool_a16_wrapper": lambda c, t: run_qdq_avgpool(c, t, 16),
    "qdq_globalavgpool_a8_wrapper": lambda c, t: run_qdq_global(c, t, 8),
    "qdq_globalavgpool_a16_wrapper": lambda c, t: run_qdq_global(c, t, 16),
    "xint8_avgpool_wrapper": lambda c, t: run_xint8(c, t, False),
    "xint8_avgpool_fused_wrapper": lambda c, t: run_xint8(c, t, True),
}


@pytest.mark.parametrize("d,case", cases(sorted(RUNNERS)))
def test_fixture_avgpool(d, case):
    t = tc(case)
    got, want = RUNNERS[d](case, t)
    assert_bytes(got, want)
