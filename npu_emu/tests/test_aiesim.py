# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""npu_emu against the NPU itself: feed each mllib regression case's input
(data/ifm32.txt) through the matching entry point and compare with the aiesim
output (data/ofm32.txt) byte for byte.

Cases come from an mllib nightly regression tree, NPU_EMU_AIESIM_ROOT
(.../Vitis_Libraries/mllib/L2/regression/wrapper); by default the newest
nightly under /wrk/mllib_regr. Every hex file is one 32-bit word per line,
little-endian, so it reads as a plain byte stream. Skipped when no tree is
reachable. convert_bf16_to_bfp16 (K64-tiled data) and dequant_w4_bfp16
(packed weights) store layouts this library does not emulate."""

import glob
import json
import os
import re

import numpy as np
import pytest

from npu import (I64, OUT, A64Coeff, NPU_BF16, NPU_F16, NPU_F32, NPU_I8, NPU_I16,
                 NPU_I32, NPU_I64, NPU_U8, NPU_U16, NPU_U32, NPU_U64, call)

from aiesim_common import ROOT, cases as _cases, fword, stream, ty, words  # noqa: E402


# Each runner returns (output bytes, expected bytes).
def run(case, wrapper):
    t = json.load(open(os.path.join(case, "testcase.json")))
    ifm = lambda dt: stream(os.path.join(case, "data/ifm32.txt"), dt)  # noqa: E731
    want = stream(os.path.join(case, "data/ofm32.txt"), np.uint8)
    nb = len(want)

    def go(name, *args, out_dtype=np.uint8):
        return call(name, *args, out_nbytes=nb, out_dtype=out_dtype).view(np.uint8)

    if wrapper == "convert_fp32_fp16":
        x = ifm(np.float32)
        return go("npu_convert_fp32_fp16", x, OUT, ty(t["ofm_dtype"])[0], I64(len(x))), want
    if wrapper == "convert_fp16_fp32":
        x = ifm(np.uint16)
        return go("npu_convert_fp16_fp32", x, ty(t["ifm_dtype"])[0], OUT, I64(len(x))), want
    if wrapper == "convert_fp16_bf16":
        x = ifm(np.uint16)
        return go("npu_convert_fp16_bf16", x, ty(t["ifm_dtype"])[0], OUT, ty(t["ofm_dtype"])[0],
                  I64(len(x))), want
    if wrapper == "convert_fp32_fp8":
        x = ifm(np.float32)
        return go("npu_convert_fp32_fp8", x, OUT, I64(len(x)), fword(case, 0)), want
    if wrapper == "convert_fp8_fp32":
        x = ifm(np.uint8)
        return go("npu_convert_fp8_fp32", x, OUT, I64(len(x)), fword(case, 0)), want
    if wrapper == "convert_fp16_fp8":
        x = ifm(np.uint16)
        return go("npu_convert_fp16_fp8", x, ty(t["ifm_dtype"])[0], OUT, I64(len(x)),
                  fword(case, 0)), want
    if wrapper == "convert_fp8_fp16":
        x = ifm(np.uint8)
        return go("npu_convert_fp8_fp16", x, OUT, ty(t["ofm_dtype"])[0], I64(len(x)),
                  fword(case, 0)), want
    if wrapper == "cast_bf2bfp16":
        x = ifm(np.uint16)
        return go("npu_cast_bf2bfp16", x, OUT, I64(len(x))), want
    if wrapper == "cast_bfp16tobf":
        x = ifm(np.uint8)
        return go("npu_cast_bfp16tobf", x, OUT, I64(len(x) // 9 * 8)), want
    if wrapper == "cast_bftomx6":
        x = ifm(np.uint16)
        return go("npu_cast_bftomx6", x, OUT, I64(len(x))), want
    if wrapper == "cast_mx6tobf":
        x = ifm(np.uint8)
        return go("npu_cast_mx6tobf", x, OUT, I64(len(x) // 12 * 16)), want
    if wrapper in ("cast_a8tobfp16", "cast_a16tobfp16"):
        tt = NPU_U8 if wrapper == "cast_a8tobfp16" else NPU_U16
        x = ifm(np.uint8 if tt == NPU_U8 else np.uint16)
        return go("npu_cast_aqtobfp16", x, tt, OUT, I64(len(x)), float(np.float32(t["scale"])),
                  float(np.float32(t["zero_point"]))), want
    if wrapper in ("cast_bfp16toa8", "cast_bfp16toa16"):
        x = ifm(np.uint8)
        tt = NPU_U8 if wrapper == "cast_bfp16toa8" else NPU_U16
        return go("npu_cast_bfp16toaq", x, OUT, tt, I64(len(x) // 9 * 8),
                  float(np.float32(t["quant_multiplier"])),
                  float(np.float32(t["zero_point"]))), want
    if wrapper == "cast_bfp16toqdqf32":
        x = ifm(np.uint8)
        return go("npu_cast_bfp16toqdqf32", x, OUT, I64(len(x) // 9 * 8),
                  float(np.float32(t["quant_multiplier"])), float(np.float32(t["zero_point"])),
                  float(np.float32(t["scale"]))), want
    if wrapper in ("quant_a8", "quant_a16"):
        it, idt = ty(t["ifm_dtype"])
        x = ifm(idt)
        return go("npu_quant", x, it, OUT, ty(t["ofm_dtype"])[0], I64(len(x)),
                  fword(case, 64), fword(case, 0)), want
    if wrapper in ("dequant_a8", "dequant_a16"):
        it, idt = ty(t["ifm_dtype"])
        x = ifm(idt)
        return go("npu_dequant", x, it, OUT, ty(t["ofm_dtype"])[0], I64(len(x)),
                  fword(case, 64), fword(case, 0)), want
    if wrapper in ("requantize_a8", "requantize_a16", "requantize_a32"):
        it, idt = ty(t["ifm_dtype"])
        x = ifm(idt)
        return go("npu_requantize", x, it, OUT, ty(t["ofm_dtype"])[0], I64(len(x)),
                  fword(case, 0), fword(case, 1)), want
    if wrapper == "cast_int32_int64":
        it, idt = ty(t["ifm_dtype"])
        x = ifm(idt)
        return go("npu_cast_int32_int64", x, it, OUT, ty(t["ofm_dtype"])[0], I64(len(x)),
                  fword(case, 0), fword(case, 1)), want
    if wrapper in ("requantize_a8_channelwise", "requantize_a16_channelwise"):
        it, idt = ty(t["ifm_dtype"])
        x = ifm(idt)
        cw = words(case, "wts32.txt").view(np.float32).reshape(-1, 64)
        return go("npu_requantize_channelwise", x, it, OUT, ty(t["ofm_dtype"])[0], I64(len(x)),
                  np.ascontiguousarray(cw[:, :32].reshape(-1)),
                  np.ascontiguousarray(cw[:, 32:].reshape(-1)), I64(cw.size // 2)), want
    if wrapper == "requantize_a64":
        it, idt = ty(t["ifm_dtype"])
        x = ifm(idt)
        w = words(case)
        f = w[:4].view(np.float32)
        B = int(w[6]) | (int(w[7]) << 32)
        co = A64Coeff(float(f[0]), float(f[1]), float(f[2]), float(f[3]),
                      int(w[4:5].view(np.int32)[0]), int(w[5:6].view(np.int32)[0]),
                      B - (1 << 64) if B >= 1 << 63 else B)
        return go("npu_requantize_a64", x, it, OUT, ty(t["ofm_dtype"])[0], I64(len(x)), co), want
    raise KeyError(wrapper)


WRAPPERS = ["convert_fp32_fp16", "convert_fp16_fp32", "convert_fp16_bf16", "convert_fp32_fp8",
            "convert_fp8_fp32", "convert_fp16_fp8", "convert_fp8_fp16", "cast_bf2bfp16",
            "cast_bfp16tobf", "cast_bftomx6", "cast_mx6tobf", "cast_a8tobfp16",
            "cast_a16tobfp16", "cast_bfp16toa8", "cast_bfp16toa16", "cast_bfp16toqdqf32",
            "quant_a8", "quant_a16", "dequant_a8", "dequant_a16", "requantize_a8",
            "requantize_a16", "requantize_a32", "requantize_a64", "requantize_a8_channelwise",
            "requantize_a16_channelwise", "cast_int32_int64"]


CASES = _cases([f"{w}_wrapper" for w in WRAPPERS])


@pytest.mark.skipif(not CASES, reason="no mllib regression outputs reachable")
@pytest.mark.parametrize("wrapper,case", CASES)
def test_aiesim(wrapper, case):
    got, want = run(case, wrapper.removesuffix("_wrapper"))
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} bytes differ from aiesim, first at "
                           f"{bad[:8].tolist()}: got {got[bad[:8]].tolist()} "
                           f"want {want[bad[:8]].tolist()}")
