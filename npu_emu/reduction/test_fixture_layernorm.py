# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""reduction/layernorm.h entry points against NPU reference outputs, on the valid channels."""

import numpy as np
import pytest

from fixture import assert_bytes, cases, data, ofm, words
from fixture import testcase as tc
from npu import (I64, OUT, NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, NPU_U8,
                 NPU_U16)
from testing import call_n as go

TY = {"bfloat16": (NPU_BF16, np.uint16), "float16": (NPU_F16, np.uint16),
      "float8": (NPU_FP8, np.uint8), "float": (NPU_F32, np.float32),
      "int8": (NPU_I8, np.int8), "uint8": (NPU_U8, np.uint8),
      "int16": (NPU_I16, np.int16), "uint16": (NPU_U16, np.uint16)}


def dims(t):
    """(N, H, W, C_pad, C_real) of the whole tensor."""
    N = t["ifm_sv_batch"]
    H = t["ifm_sv_height"] * t["ifm_height_iteration"]
    W = t["ifm_sv_width"] * t["ifm_width_iteration"]
    C = t["ifm_sv_depth"]
    return N, H, W, C, t.get("ifm_sv_depth_real", C)


def from_blocked(x, N, H, W, C, g):
    """[H, C/g, W, N, g] -> [N*H*W, C]."""
    x = np.asarray(x)[:N * H * W * C].reshape(H, C // g, W, N, g)
    return np.ascontiguousarray(x.transpose(3, 0, 2, 1, 4).reshape(N * H * W, C))


def interleaved(w, C, g=32):
    """[add_g, mul_g] chunks -> (additive[C], multiplicative[C])."""
    w = np.asarray(w)[:2 * C].reshape(C // g, 2, g)
    return np.ascontiguousarray(w[:, 0].reshape(-1)), np.ascontiguousarray(w[:, 1].reshape(-1))


def coeff(case, t, k):
    """fp32 coefficient word k after param_offset."""
    return float(words(case, "wts32.txt")[t["param_offset"] + k:][:1].view(np.float32)[0])


def stats_of(t):
    """NPU_LN_STATS_* of a case."""
    ig, og = t["inner_g"], t["outer_g"]
    return int(t["il_loop_range"] <= 2 and t["c_mean"] == 32 * ig and ig in (2, 3)
               and og >= 8 * ig)


def compare(got, want):
    assert_bytes(np.ascontiguousarray(got), np.ascontiguousarray(want))


def run_layer_norm_bf16(case, t):
    N, H, W, C, _ = dims(t)
    # 0 (and absent in older trees): all C channels with epsilon 1e-12
    c_real = t.get("ifm_sv_depth_real", 0)
    eps = float(np.float32(t.get("epsilon", 1e-12))) if c_real else 1e-12
    it, idt = TY[t["dtype"]]
    _, wdt = TY[t["wts_dtype"]]
    x = from_blocked(data(case, "ifm32.txt", idt), N, H, W, C, 64)
    want = from_blocked(data(case, ofm(case), idt), N, H, W, C, 64)
    w = data(case, "wts32.txt", wdt)
    rows = x.shape[0]
    got = go("npu_layer_norm_bf16", x, it, w[:C], w[C:2 * C], OUT, I64(rows), C, c_real, eps,
             n=rows * C, dt=idt)
    compare(got, want)


def bfp_svs(t):
    """Block geometry: (iterations, N, Y, X)."""
    return (t["ifm_height_iteration"] * t["ifm_width_iteration"], t["ifm_sv_batch"],
            t["ifm_sv_height"], t["ifm_sv_width"])


def bfp_from_svs(b, t, C):
    """bfp16 bytes per block [Y, C/64, X, N, 64] -> rows of C/8 groups."""
    It, N, Y, X = bfp_svs(t)
    b = np.asarray(b, np.uint8)[:It * N * Y * X * C // 8 * 9]
    b = b.reshape(It, Y, C // 64, X, N, 8, 9).transpose(0, 4, 1, 3, 2, 5, 6)
    return np.ascontiguousarray(b.reshape(-1, C // 8 * 9))


def run_layernorm_bfp16(case, t):
    C = t["ifm_sv_depth"]
    x = bfp_from_svs(data(case, "ifm32.txt"), t, C)
    want = bfp_from_svs(data(case, ofm(case)), t, C)
    w = data(case, "wts32.txt", np.uint16)
    rows = x.shape[0]
    got = go("npu_layer_norm_bfp16", x, w[:C], w[C:2 * C], OUT, I64(rows), C,
             n=rows * C // 8 * 9, dt=np.uint8)
    compare(got, want)


def run_layernorm_bf16_bfp16(case, t):
    N, H, W, C, _ = dims(t)
    It, _, Y, X = bfp_svs(t)
    x = from_blocked(data(case, "ifm32.txt", np.uint16), N, H, W, C, 64)
    w = data(case, "wts32.txt", np.uint16)
    rows = x.shape[0]
    got = go("npu_layer_norm_bf16_bfp16", x, w[:C], w[C:2 * C], OUT, I64(rows), C,
             n=rows * C // 8 * 9, dt=np.uint8)
    # global [N, H, W] rows -> the per-block row order of the output
    got = got.reshape(N, H, W, C // 8 * 9)
    hi, wi = t["ifm_height_iteration"], t["ifm_width_iteration"]
    got = np.concatenate([got[:, h * Y:(h + 1) * Y, k * X:(k + 1) * X].reshape(-1, C // 8 * 9)
                          for h in range(hi) for k in range(wi)])
    compare(got, bfp_from_svs(data(case, ofm(case)), t, C))


def rms_coeffs(t):
    """inv_inner_dim and epsilon as the kernel words hold them."""
    inv = float(np.array([t["rms_inv_inner_dim_bits"]], np.uint32).view(np.float32)[0])
    eps = float(np.array([t["rms_epsilon_bits"]], np.uint32).view(np.float32)[0])
    return inv, eps


def run_rmsnorm_bf16(case, t):
    C = t["ifm_sv_depth"]
    it, idt = TY[t["dtype"]]
    _, wdt = TY[t["wts_dtype"]]
    x = data(case, "ifm32.txt", idt)
    rows = len(x) // C
    x = x[:rows * C]
    got = go("npu_rmsnorm_bf16", x, it, data(case, "wts32.txt", wdt)[:C], OUT, I64(rows), C,
             *rms_coeffs(t), n=rows * C, dt=idt)
    compare(got, data(case, ofm(case), idt)[:rows * C])


def run_rmsnorm_bfp16(case, t):
    C = t["ifm_sv_depth"]
    x = data(case, "ifm32.txt")
    rows = len(x) // (C // 8 * 9)
    nb = rows * C // 8 * 9
    got = go("npu_rmsnorm_bfp16", x[:nb], data(case, "wts32.txt", np.uint16)[:C], OUT,
             I64(rows), C, *rms_coeffs(t), n=nb, dt=np.uint8)
    compare(got, data(case, ofm(case))[:nb])


def run_l2_norm_bf16(case, t):
    N, H, W, C, _ = dims(t)
    it, idt = TY[t["dtype"]]
    _, wdt = TY[t["wts_dtype"]]
    x = from_blocked(data(case, "ifm32.txt", idt), N, H, W, C, 64)
    beta, alpha = interleaved(data(case, "wts32.txt", wdt), C)
    rows = x.shape[0]
    got = go("npu_l2_norm_bf16", x, it, alpha, beta, OUT, I64(rows), C, n=rows * C, dt=idt)
    compare(got, from_blocked(data(case, ofm(case), idt), N, H, W, C, 64))


def run_qdq_l2norm(case, t, bits):
    N, H, W, C, _ = dims(t)
    it, idt = TY[t["ifm_dtype"]]
    ot, odt = TY[t["ofm_dtype"]]
    x = from_blocked(data(case, "ifm32.txt", idt), N, H, W, C, 64)
    if t.get("alphabeta", 1):
        beta, alpha = interleaved(data(case, "wts32.txt", np.uint16), C)
    else:
        beta = alpha = None
    rows = x.shape[0]
    got = go(f"npu_qdq_l2norm_a{bits}", x, it, alpha, beta, OUT, ot, I64(rows), C,
             *[coeff(case, t, k) for k in (0, 64, 128, 192)], n=rows * C, dt=odt)
    compare(got, from_blocked(data(case, ofm(case), odt), N, H, W, C, 64))


def gran(t):
    return 32 if t.get("order_64", 1) == 0 else 64


def add_const_tiles(stream, t, C, dt):
    """The first add_const copy of each block, assembled into [N*H*W, C]."""
    N, H, W, _, _ = dims(t)
    hi, wi = t["ifm_height_iteration"], t["ifm_width_iteration"]
    Y, X = t["ifm_sv_height"], t["ifm_sv_width"]
    g = gran(t)
    sv = N * Y * X * C
    out = np.empty((N, H, W, C), dt)
    for h in range(hi):
        for k in range(wi):
            tile = stream[(h * wi + k) * 2 * sv:][:sv]
            out[:, h * Y:(h + 1) * Y, k * X:(k + 1) * X] = from_blocked(tile, N, Y, X, C, g) \
                .reshape(N, Y, X, C)
    return out.reshape(-1, C)


def run_add_layernorm(case, t, const):
    N, H, W, Cp, Cr = dims(t)
    it, idt = TY[t["dtype"]]
    wt, wdt = TY[t["wts_dtype"]]
    g = gran(t)
    x = from_blocked(data(case, "ifm32.txt", idt), N, H, W, Cp, g)
    wb = data(case, "wts32.txt")
    beta, alpha = interleaved(wb.view(wdt), Cp)
    if const:
        k = add_const_tiles(wb[2 * Cp * np.dtype(wdt).itemsize:].view(idt), t, Cp, idt)
    else:
        k = from_blocked(data(case, "ifm2_32.txt", idt), N, H, W, Cp, g)
    rows = x.shape[0]
    name = "npu_add_layernorm_const_bf16" if const else "npu_add_layernorm_bf16"
    got = go(name, np.ascontiguousarray(x[:, :Cr]), np.ascontiguousarray(k[:, :Cr]), it,
             alpha[:Cr], beta[:Cr], OUT, I64(rows), Cr, float(np.float32(t["epsilon"])),
             stats_of(t), n=rows * Cr, dt=idt)
    want = from_blocked(data(case, ofm(case), idt), N, H, W, Cp, g)[:, :Cr]
    compare(got, want)


def qdq_common(case, t):
    N, H, W, Cp, Cr = dims(t)
    ct, cdt = TY[t["compute_dtype"]]
    ot, odt = TY[t["ofm_dtype"]]
    beta, alpha = interleaved(data(case, "wts32.txt", cdt), Cp)
    q = [coeff(case, t, k) for k in (128, 192)]  # q_zp, q_sc
    return N, H, W, Cp, Cr, ct, cdt, ot, odt, alpha[:Cr], beta[:Cr], q


def run_qdq_add_layernorm(case, t):
    N, H, W, Cp, Cr, ct, cdt, ot, odt, alpha, beta, q = qdq_common(case, t)
    it, idt = TY[t["ifm_dtype"]]
    a = from_blocked(data(case, "ifm32.txt", idt), N, H, W, Cp, 64)[:, :Cr]
    b = from_blocked(data(case, "ifm2_32.txt", idt), N, H, W, Cp, 64)[:, :Cr]
    rows = a.shape[0]
    got = go("npu_qdq_add_layernorm", np.ascontiguousarray(a), np.ascontiguousarray(b), it,
             alpha, beta, ct, OUT, ot, I64(rows), Cr,
             *[coeff(case, t, k) for k in (0, 64, 128, 192, 256, 320)],
             float(np.float32(t["epsilon"])), stats_of(t), n=rows * Cr, dt=odt)
    compare(got, from_blocked(data(case, ofm(case), odt), N, H, W, Cp, 64)[:, :Cr])


def run_qdq_add_layernorm_const(case, t):
    N, H, W, Cp, Cr, ct, cdt, ot, odt, alpha, beta, q = qdq_common(case, t)
    it, idt = TY[t["ifm_dtype"]]
    a = from_blocked(data(case, "ifm32.txt", idt), N, H, W, Cp, 64)[:, :Cr]
    wb = data(case, "wts32.txt")
    k = add_const_tiles(wb[(t["param_offset"] + 256) * 4:].view(cdt), dict(t, order_64=1), Cp,
                        cdt)[:, :Cr]
    rows = a.shape[0]
    got = go("npu_qdq_add_layernorm_const", np.ascontiguousarray(a), it, np.ascontiguousarray(k),
             ct, alpha, beta, OUT, ot, I64(rows), Cr, coeff(case, t, 0), coeff(case, t, 64), *q,
             float(np.float32(t["epsilon"])), stats_of(t), n=rows * Cr, dt=odt)
    compare(got, from_blocked(data(case, ofm(case), odt), N, H, W, Cp, 64)[:, :Cr])


def run_qdq_add_add_const_layernorm(case, t):
    N, H, W, Cp, Cr, ct, cdt, ot, odt, alpha, beta, q = qdq_common(case, t)
    at, adt = TY[t["ifm_dtype"]]
    bt, bdt = TY[t["ifm2_dtype"]]
    g = gran(t)
    a = from_blocked(data(case, "ifm32.txt", adt), N, H, W, Cp, g)[:, :Cr]
    b = from_blocked(data(case, "ifm2_32.txt", bdt), N, H, W, Cp, g)[:, :Cr]
    k = data(case, "wts32.txt")[(t["param_offset"] + 384) * 4:].view(cdt)[:Cr]
    rows = a.shape[0]
    got = go("npu_qdq_add_add_const_layernorm", np.ascontiguousarray(a), at,
             np.ascontiguousarray(b), bt, np.ascontiguousarray(k), ct, alpha, beta, OUT, ot,
             I64(rows), Cr, *[coeff(case, t, k) for k in (0, 64, 128, 192, 256, 320)],
             float(np.float32(t["epsilon"])), stats_of(t), n=rows * Cr, dt=odt)
    compare(got, from_blocked(data(case, ofm(case), odt), N, H, W, Cp, g)[:, :Cr])


QLN_FUSED, QLN_BATCHED, QLN_C8_FUSED, QLN_C8_BATCHED, QLN_INTEGER = range(5)


def qln_variant(t):
    ig, og = t["inner_g"], t["outer_g"]
    if t.get("integer_mode", 0):
        return QLN_INTEGER
    if t.get("is_c8_layout", 0):
        batched = (ig == 1 and og >= 8 and og % 8 == 0) or (ig == 2 and og >= 16 and og % 16 == 0)
        return QLN_C8_BATCHED if batched else QLN_C8_FUSED
    return QLN_BATCHED if stats_of(t) else QLN_FUSED


def run_qdq_layernorm(case, t):
    Cp = t["ifm_sv_depth"]
    Cr = t.get("ifm_sv_depth_real", Cp)
    it, idt = TY[t["ifm_dtype"]]
    ot, odt = TY[t["ofm_dtype"]]
    ct, _ = TY[t["compute_dtype"]]
    x = data(case, "ifm32.txt", idt)
    rows = len(x) // Cp
    x = np.ascontiguousarray(x[:rows * Cp].reshape(rows, Cp)[:, :Cr])
    v = qln_variant(t)
    g = 8 if t.get("is_c8_layout", 0) else 32
    w = words(case, "wts32.txt").view(np.float32)
    beta, alpha = interleaved(w, Cp, g)
    got = go("npu_qdq_layernorm", x, it, alpha[:Cr], beta[:Cr], OUT, ot, I64(rows), Cr, ct,
             coeff(case, t, 0), coeff(case, t, 64), v, n=rows * Cr, dt=odt)
    want = data(case, ofm(case), odt)[:rows * Cp].reshape(rows, Cp)[:, :Cr]
    compare(got, want)


def run_layer_norm_a8_bfp16(case, t):
    N, H, W, C, _ = dims(t)
    g = gran(t)
    it, idt = TY[t["ifm_dtype"]]
    x = from_blocked(data(case, "ifm32.txt", idt), N, H, W, C, g)
    beta, alpha = interleaved(data(case, "wts32.txt", np.uint16), C)
    rows = x.shape[0]
    got = go("npu_layer_norm_a8_bfp16", x, it, alpha, beta, OUT, I64(rows), C,
             coeff(case, t, 0), coeff(case, t, 64), QLN_BATCHED if stats_of(t) else QLN_FUSED,
             n=rows * C // 8 * 9, dt=np.uint8)
    # global [N, H, W] rows -> blocks [Y, C/g, X, N] of each sub-volume
    It, _, Y, X = bfp_svs(t)
    hi, wi = t["ifm_height_iteration"], t["ifm_width_iteration"]
    got = got.reshape(N, H, W, C // g, g // 8 * 9)
    got = np.concatenate([got[:, h * Y:(h + 1) * Y, k * X:(k + 1) * X].transpose(1, 3, 2, 0, 4)
                          .reshape(-1) for h in range(hi) for k in range(wi)])
    compare(got, data(case, ofm(case))[:got.size])


RUNNERS = {
    "layernorm_bf16_wrapper": run_layer_norm_bf16,
    "layernorm_bfp16_wrapper": run_layernorm_bfp16,
    "layernorm_bf16_bfp16_wrapper": run_layernorm_bf16_bfp16,
    "rmsnorm_bf16_wrapper": run_rmsnorm_bf16,
    "rmsnorm_bfp16_wrapper": run_rmsnorm_bfp16,
    "l2_norm_bf16_wrapper": run_l2_norm_bf16,
    "qdq_l2norm_a8_wrapper": lambda c, t: run_qdq_l2norm(c, t, 8),
    "qdq_l2norm_a16_wrapper": lambda c, t: run_qdq_l2norm(c, t, 16),
    "add_layernorm_bf16_wrapper": lambda c, t: run_add_layernorm(c, t, False),
    "add_layernorm_const_bf16_wrapper": lambda c, t: run_add_layernorm(c, t, True),
    "qdq_add_layernorm_wrapper": run_qdq_add_layernorm,
    "qdq_add_layernorm_const_wrapper": run_qdq_add_layernorm_const,
    "qdq_add_add_const_layernorm_wrapper": run_qdq_add_add_const_layernorm,
    "qdq_layernorm_wrapper": run_qdq_layernorm,
    "layernorm_a8_bfp16_wrapper": run_layer_norm_a8_bfp16,
}


@pytest.mark.parametrize("d,case", cases(sorted(RUNNERS)))
def test_layernorm_group(d, case):
    RUNNERS[d](case, tc(case))
