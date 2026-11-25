# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""convert.h entry points against NPU reference outputs, byte for byte."""

import os

import numpy as np
import pytest

from npu import I64, OUT, call

from fixture import cases as _cases, fword, ofm, stream, testcase as tc, ty, words  # noqa: E402
from test_golden_convert import A64Coeff


# Each runner returns (output bytes, expected bytes).
def run(case, wrapper):
    t = tc(case)
    ifm = lambda dt: stream(os.path.join(case, "data/ifm32.txt"), dt)  # noqa: E731
    want = stream(os.path.join(case, "data", ofm(case)), np.uint8)
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
    if wrapper == "requantize_fp8":
        x = ifm(np.uint8)
        return go("npu_requantize_fp8", x, OUT, I64(len(x)), fword(case, 0)), want
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
        x = ifm(np.uint8 if wrapper == "cast_a8tobfp16" else np.uint16)
        return go("npu_" + wrapper, x, OUT, I64(len(x)), float(np.float32(t["scale"])),
                  float(np.float32(t["zero_point"]))), want
    if wrapper in ("cast_bfp16toa8", "cast_bfp16toa16"):
        x = ifm(np.uint8)
        return go("npu_" + wrapper, x, OUT, I64(len(x) // 9 * 8),
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
                  fword(case, 0), fword(case, 64)), want
    if wrapper in ("dequant_a8", "dequant_a16"):
        it, idt = ty(t["ifm_dtype"])
        x = ifm(idt)
        return go("npu_dequant", x, it, OUT, ty(t["ofm_dtype"])[0], I64(len(x)),
                  fword(case, 0), fword(case, 64)), want
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
        C = int(t["channels"])
        # The blob holds max(1, C / 32) blocks of [32 M, 32 B]; C < 32 is tiled within a block.
        M, B = (np.ascontiguousarray(cw[:, h:h + 32].reshape(-1)[:C]) for h in (0, 32))
        return go("npu_" + wrapper, x, it, OUT, ty(t["ofm_dtype"])[0], I64(len(x)), I64(C),
                  M, B), want
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
    if wrapper == "convert_bf16_to_bfp16":
        return bf16_to_bfp16(t, ifm(np.uint16), want)
    if wrapper == "dequant_w4_bfp16":
        return dequant_w4(t, ifm(np.uint8), want)
    raise KeyError(wrapper)


def bf16_to_bfp16(t, x, want):
    # The dump holds [N][K] in and, per (sv_n x sv_k) tile, [K/64][sv_n][64] zero-padded out
    # blocks; blocks without a valid element are not compared.
    N, K = t["ifm_n_full_valid"], t["ifm_k_full_valid"]
    sn, sk = t["ofm_sv_n_physical"], t["ofm_sv_k_physical"]
    ni, ki = t["ofm_n_iteration"], t["ofm_k_iteration"]
    pad, keep = np.zeros((ni * sn, ki * sk), np.uint16), np.zeros((ni * sn, ki * sk), bool)
    pad[:N, :K], keep[:N, :K] = x.reshape(N, K), True
    tiled = lambda a: a.reshape(ni, sn, ki, sk // 64, 64).transpose(0, 2, 3, 1, 4).reshape(-1)  # noqa: E731
    got = call("npu_convert_bf16_to_bfp16", tiled(pad), ty(t["ifm_dtype"])[0], OUT, I64(pad.size),
               out_nbytes=pad.size // 8 * 9, out_dtype=np.uint8)
    keep = np.repeat(tiled(keep).reshape(-1, 8).any(1), 9)
    return got[keep], want[keep]


def dequant_w4(t, b, want):
    # Per 64 rows of N: [bias_in_pad bytes] then per 64 of K: zp [64][2] u8, scales [2][64] bf16,
    # weights [64][64] u4 (low nibble first). Out per 64 rows: [bias_out_pad bytes] then per 64 of
    # K the [64][8] bfp16 blocks; the bias part is not the entry's.
    N, K, eb = t["N"], t["K"], t["enable_bias"]
    no, kb = N // 64, K // 64
    bin_, bout = t["bias_in_pad"] if eb and t["cp_bias"] else 0, t["bias_out_pad"] if eb else 0
    blk = b.reshape(no, -1)[:, bin_:].reshape(no, kb, 2432)
    zp = blk[..., :128].reshape(no, kb, 64, 2).transpose(0, 2, 1, 3).reshape(N, K // 32)
    s = np.ascontiguousarray(blk[..., 128:384]).view(np.uint16).reshape(no, kb, 2, 64)
    s = s.transpose(0, 3, 1, 2).reshape(N, K // 32)
    w = blk[..., 384:].reshape(no, kb, 64, 32)
    q = np.stack([w & 15, w >> 4], -1).reshape(no, kb, 64, 64).transpose(0, 2, 1, 3).reshape(N, K)
    got = call("npu_dequant_w4_bfp16", np.ascontiguousarray(zp), np.ascontiguousarray(s),
               np.ascontiguousarray(q), OUT, I64(N), I64(K), out_nbytes=N * K // 8 * 9,
               out_dtype=np.uint8)
    got = got.reshape(no, 64, kb, 72).transpose(0, 2, 1, 3).reshape(-1)
    return got, want.reshape(no, -1)[:, bout:].reshape(-1)


WRAPPERS = ["convert_fp32_fp16", "convert_fp16_fp32", "convert_fp16_bf16", "convert_fp32_fp8",
            "convert_fp8_fp32", "requantize_fp8", "convert_fp16_fp8", "convert_fp8_fp16", "cast_bf2bfp16",
            "cast_bfp16tobf", "cast_bftomx6", "cast_mx6tobf", "cast_a8tobfp16",
            "cast_a16tobfp16", "cast_bfp16toa8", "cast_bfp16toa16", "cast_bfp16toqdqf32",
            "quant_a8", "quant_a16", "dequant_a8", "dequant_a16", "requantize_a8",
            "requantize_a16", "requantize_a32", "requantize_a64", "requantize_a8_channelwise",
            "requantize_a16_channelwise", "cast_int32_int64", "convert_bf16_to_bfp16",
            "dequant_w4_bfp16"]


CASES = _cases([f"{w}_wrapper" for w in WRAPPERS])


@pytest.mark.skipif(not CASES, reason="no fixture cases")
@pytest.mark.parametrize("wrapper,case", CASES)
def test_fixture(wrapper, case):
    got, want = run(case, wrapper.removesuffix("_wrapper"))
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} bytes differ from sim, first at "
                           f"{bad[:8].tolist()}: got {got[bad[:8]].tolist()} "
                           f"want {want[bad[:8]].tolist()}")
