# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The bfp16 / mx6 / fp8 gemm and conv entry points against the reference models of the
model repository."""

import copy
import sys

import numpy as np
import pytest

import fixture as fm
from npu import I64, NPU_BF16, NPU_BFP16, NPU_F16, NPU_FP8, NPU_MX6, NPU_U8, OUT, call

DIRS = ["conv_add_bfp16_wrapper", "conv_add_fp8_wrapper", "conv_bfp16_wrapper",
        "conv_fp16xfp8_channelwise_wrapper", "conv_fp16xfp8_wrapper",
        "conv_fp8_channelwise_wrapper", "conv_fp8_wrapper", "conv_mx6_wrapper",
        "depthwise_conv_fp8_channelwise_wrapper", "depthwise_conv_fp8_wrapper",
        "gemm_add_bfp16_wrapper", "gemm_bfp16_wrapper", "gemm_fp8_channelwise_wrapper",
        "gemm_fp8_wrapper", "gemm_gelu_mul_bfp16_wrapper", "gemm_mul_bfp16_wrapper",
        "gemm_mx6_wrapper"]
HOOKS = {"*Bfp16Wrapper": ("compute_bit_exact", "_encode_ofm"),
         "*Mx6Wrapper": ("compute_bit_exact", "_encode_ofm"),
         "*Fp8*Wrapper*": ("compute_bit_exact", "_write_ofm")}
T = {"bf16": NPU_BF16, "bfp16": NPU_BFP16, "mx6": NPU_MX6, "u8": NPU_U8, "f16": NPU_F16,
     "fp8": NPU_FP8}
OFM = {"float8": "fp8", "float16": "f16", "bfloat16": "bf16"}
BFP16 = dict(bits=8, ebs=8, bfloat=None, expo_bits=8, subtile_size=1, shift_bits=0, sgn_mag=False)
MX6 = dict(bits=5, ebs=16, bfloat=None, expo_bits=8, subtile_size=2, shift_bits=1, sgn_mag=True)
f32, I = np.float32, np.int64


def bfp16(conv, x, dim=-1, **kw):
    """Packed bfp16 of float rows: per block of 8 one exponent byte, then 8 mantissas."""
    m, e, _ = conv.f2bfp(copy.copy(np.asarray(x, f32)), dim=dim, **BFP16, **kw)
    m, e = np.asarray(m, I), np.asarray(e, I) & 255
    if dim == -2:
        m, e = m.T.copy(), e.T.copy()
    rows, n = m.shape
    out = np.zeros((rows, n // 8, 9), np.uint8)
    out[:, :, 0] = e
    out[:, :, 1:] = m.reshape(rows, n // 8, 8).astype(np.int8).view(np.uint8)
    return out.reshape(-1)


def mx6(conv, x):
    """Packed mx6 of float rows: per block of 16 an exponent byte, 8 shift bits, 16 x 5 bits."""
    m, e, sh = conv.f2bfp_mladf(copy.copy(np.asarray(x, f32))[np.newaxis, ...], dim=-1, **MX6)
    m, e, sh = np.asarray(m[0], I), np.asarray(e[0], I) & 255, np.asarray(sh[0], I)
    sh = sh.reshape(sh.shape[0], -1)
    rows, n = m.shape
    out = []
    for r in range(rows):
        for b in range(n // 16):
            w = int(e[r, b])
            for j in range(8):
                w |= int(sh[r, b * 8 + j] & 1) << (8 + j)
            for j in range(16):
                code = int(m[r, b * 16 + j]) & 31
                w |= (0 if code == 16 else code) << (16 + 5 * j)
            out.append(np.frombuffer(w.to_bytes(12, "little"), np.uint8))
    return np.concatenate(out)


def fp8(s, x):
    f = sys.modules["fp8_wrapper_common"]._fp32_to_fp8_bytes
    return f(s.c, np.asarray(x, f32).reshape(-1))[0].astype(np.uint8)


def block_act(s):
    """(narrow, act, alpha) of the bfp16 / mx6 conv and the mx6 gemm."""
    at = getattr(s, "activation_type", 0)
    rt = getattr(s, "activation", at)
    act = 3 if at == 3 else (rt if (at and rt in (1, 2)) else 0)
    alpha = {3: getattr(s, "leakyRelu_alpha", 0.0), 1: 0.0}.get(rt, 1.0) if at == 3 else 0.0
    return int(at > 0), act, alpha


def extract(d, seen):
    """Operands in the entry's formats, parameters and the expected output bytes."""
    import golden_block_matmul as gb
    from golden_block_fp import encode_bfp16
    from golden_int_round import Rnd
    s = seen["compute_bit_exact"][0][0][0]
    cls, data = type(s).__name__, s.data
    bias = None
    if cls == "GemmBfp16Wrapper":
        M, K, N = s.m_iteration * s.sv_m_dim, s.k_iteration * s.sv_k_dim, s.n_iteration * s.sv_n_dim
        t = s.activation_type_templates
        narrow = int(t in (1, 2, 3, 5, 6, 7, 12, 101) or s.ofm_io_type == "bfloat16")
        act = 3 if t == 3 else (s.activation if s.activation in (1, 2, 5, 6, 7) else 0)
        alpha = {3: s.leakyRelu_alpha, 1: 0.0}.get(s.activation, 1.0) if t == 3 else 0.0
        bias = np.asarray(data["bias"], f32).reshape(N) if (s.do_bias and "bias" in data) else None
        ofm = "u8" if t == 12 else "bf16" if s.ofm_io_type == "bfloat16" else "bfp16"
        v = np.asarray(data["ofm_0"], f32)
        want = (v.astype(np.uint8).reshape(-1) if ofm == "u8" else fm.bf16_bits(v).reshape(-1)
                if ofm == "bf16" else encode_bfp16(v.reshape(-1), Rnd.kHalfAway))
        q = (float(f32(1.0 / s.ofm_scale)), float(f32(s.ofm_zero_point))) if ofm == "u8" else (0, 0)
        fx = dict(a=bfp16(s.converter, data["matA"].reshape(M, K)),
                  bt=bfp16(s.converter, data["matB"].reshape(K, N), -2), dims=np.array([M, K, N]),
                  flags=np.array([narrow, act, int(s.transpose_ofm)]), alpha=np.array([alpha], f32),
                  q=np.array(q, f32))
    elif cls in ("GemmAddBfp16Wrapper", "GemmMulBfp16Wrapper"):
        M, K, N = s._dims()
        bias = np.asarray(data["bias"], f32).reshape(N) if (s.do_bias and "bias" in data) else None
        m3, n3 = s._ifm2_logical_shape()
        x2 = s._ifm2_to_float_broadcast(data["ifm2"])[:m3]
        packed = s.ifm2_dtype == "bfp16"
        ofm = "bf16" if s.ofm_dtype == "bfloat16" else "bfp16"
        v = np.asarray(data["ofm_0"], f32)
        want = fm.bf16_bits(v).reshape(-1) if ofm == "bf16" else encode_bfp16(v.reshape(-1), Rnd.kHalfAway)
        fx = dict(
                  a=bfp16(s.converter, data["matA"].reshape(M, K)),
                  bt=bfp16(s.converter, data["matB"].reshape(K, N), -2),
                  ifm2=bfp16(s.converter, x2) if packed else fm.bf16_bits(x2).reshape(-1),
                  ifm2_t="bfp16" if packed else "bf16",
                  dims=np.array([M, K, N, m3, s.sv_m_dim, s.sv_n_dim, s.sv_k_dim]))
    elif cls in ("ConvBfp16Wrapper", "ConvAddBfp16Wrapper", "ConvMx6Wrapper"):
        mx = cls == "ConvMx6Wrapper"
        Ho, Wo, Co = s.ofm_height, s.ofm_width, s.ofm_depth
        Kh, Kx, Ci = s.L3_Kh, s.kernel_width, s.ifm_depth
        Hi, Wi = (Ho - 1) * s.stride_height + Kh, (Wo - 1) * s.stride_width + Kx
        wl = np.ascontiguousarray(s.wts_bf16.transpose(0, 1, 3, 2)).reshape(Kh * Kx * Co, Ci)
        if mx:
            ifm = mx6(s.converter, np.asarray(s.ifm_mx6, f32).reshape(Hi * Wi, Ci))
            wts = mx6(s.converter, wl)
        else:
            ifm = np.asarray(data["ifm"]).astype(np.int8).view(np.uint8).reshape(-1)
            wts = bfp16(s.converter, wl, rounding="even")
        bias = np.asarray(data["bias"], f32).reshape(Co) if s.do_bias else None
        narrow, act, alpha = block_act(s)
        ofm = "bf16" if getattr(s, "ofm_dtype", "") == "bfloat16" else "mx6" if mx else "bfp16"
        ifm2, h2, w2, c2 = None, 1, 1, 1
        if cls == "ConvAddBfp16Wrapper":
            h2, w2, c2 = s._ifm2_logical_shape() if hasattr(s, "_ifm2_logical_shape") else (Ho, Wo, Co)
            ifm2 = fm.bf16_bits(np.asarray(s._ifm2_to_float_broadcast(data["ifm2"]), f32)[:h2, :w2, :c2])
            narrow, act, alpha = 0, 0, 0.0
            want = np.asarray(data["ofm_0"]).astype(np.int8).view(np.uint8).reshape(-1)
        else:
            want = gb.store(np.asarray(fm.args(seen, "_encode_ofm", 0)[1], f32), ofm, Rnd.kHalfAway)
        fx = dict(ifm=ifm, wts=wts,
                  ifm2=None if ifm2 is None else ifm2.reshape(-1),
                  dims=np.array([Hi, Wi, Ci, Kh, Kx, Co, s.stride_height, s.stride_width,
                                 s.kernel_sv_height, s.ifm_sv_depth, s.ofm_sv_depth, narrow, act, h2,
                                 w2, c2]), alpha=np.array([alpha], f32))
    elif cls == "GemmMx6Wrapper":
        M, K, N = s.full_m_dim, s.full_k_dim, s.full_n_dim
        bias = np.asarray(data["bias"], f32).reshape(N) if (s.do_bias and "bias" in data) else None
        narrow, act, alpha = block_act(s)
        ofm = "bf16" if s.ofm_dtype == "bfloat16" else "mx6"
        want = gb.store(np.asarray(data["ofm_0"], f32), ofm, Rnd.kHalfAway)
        fx = dict(a=mx6(s.converter, data["matA"].reshape(M, K)),
                  bt=mx6(s.converter, data["matB"].reshape(K, N).T),
                  dims=np.array([M, K, N, narrow, act]), alpha=np.array([alpha], f32))
    elif cls in ("GemmFp8Wrapper", "GemmFp8ChannelwiseWrapper"):
        M, K, N = s._full_dims()
        c0, c2 = (np.asarray(x, f32).reshape(N) for x in s._coeff_vectors(N))
        ofm = OFM[s.ofm_dtype]
        want = gb.store(np.asarray(data["ofm"], f32), ofm, Rnd.kHalfAway)
        fx = dict(c0=c0, c2=c2, dims=np.array([M, K, N]),
                  a=np.asarray(data["ifm_bytes"]).astype(np.int8).view(np.uint8).reshape(-1),
                  b=np.asarray(data["wts_bytes"]).astype(np.int8).view(np.uint8).reshape(-1))
    elif cls in ("ConvFp8Wrapper", "ConvFp8ChannelwiseWrapper", "ConvAddFp8Wrapper",
                 "ConvFp16xFp8Wrapper", "ConvFp16xFp8ChannelwiseWrapper"):
        ifm, wts = np.asarray(data["ifm"], f32), np.asarray(data["wts"], f32)
        Hi, Wi, Ci = ifm.shape
        Kh, Kx, _, Co = wts.shape
        c0, c2 = s._coeff_vectors(Co)
        add = cls == "ConvAddFp8Wrapper"
        act = 0 if add else int(s.activation)
        a0, a1, slope = 0.0, 0.0, None
        if act == 2:
            a0 = float(f32(s.relu6_max))
        elif act == 3:
            a0 = float(f32(s.leakyRelu_alpha))
        elif act == 8:
            a0 = float(f32(s.clip_min / float(s.ofm_scale)))
            a1 = float(f32(s.clip_max / float(s.ofm_scale)))
        elif act == 9:
            slope = np.asarray(s._prelu_slope_full(Co), f32)
        ofm = OFM[s.ofm_dtype]
        ifm2, it, h2, w2, c2d, coeff = None, "fp8", 1, 1, 1, 0.0
        if add:
            h2, w2, c2d = s._ifm2_logical_shape()
            r = np.asarray(data["ifm2"], f32).reshape(h2, w2, c2d)
            it = OFM[s.ifm2_dtype]
            ifm2 = (fp8(s, r) if it == "fp8" else fm.bf16_bits(r).reshape(-1) if it == "bf16" else
                    r.astype(np.float16).view(np.uint16).reshape(-1))
            coeff = float(s._ifm2_coeff())
        want = gb.store(np.asarray(fm.args(seen, "_write_ofm", 0)[1], f32), ofm, Rnd.kHalfAway)
        f16_in = cls.startswith("ConvFp16xFp8")
        # f16 activations stay f16 (the entry narrows them); r is the f16 reciprocal scale
        r = float(s._ifm_scale_recip_fp16()) if f16_in else 0.0
        x = ifm.astype(np.float16).view(np.uint16).reshape(-1) if f16_in else fp8(s, ifm)
        fx = dict(ifm=x, ifm_t="f16" if f16_in else "fp8", ifm_r=np.array([r], f32),
                  wts=fp8(s, wts), c0=c0, c2=c2, ifm2_t=it,
                  ifm2=ifm2, slope=slope, fl=np.array([a0, a1, coeff], f32),
                  dims=np.array([Hi, Wi, Ci, Kh, Kx, Co, s.stride_height, s.stride_width,
                                 s.kernel_sv_height, s.ifm_sv_depth, s.ofm_sv_depth, act, h2, w2,
                                 c2d]))
    elif cls in ("DepthwiseConvFp8Wrapper", "DepthwiseConvFp8ChannelwiseWrapper"):
        ifm, wts = np.asarray(data["ifm"], f32), np.asarray(data["wts"], f32)
        Hi, Wi, C = ifm.shape
        Kh, Kx, _ = wts.shape
        coeff = np.asarray(data["coeff"], f32)
        act = int(s.activation)
        a0 = float(f32(s.relu6_max)) if act == 2 else (float(f32(s.alpha)) if act == 3 else 0.0)
        ofm = OFM[s.ofm_dtype]
        want = gb.store(np.asarray(fm.args(seen, "_write_ofm", 0)[1], f32), ofm, Rnd.kHalfAway)
        fx = dict(ifm=fp8(s, ifm), wts=fp8(s, wts), c0=coeff[0].copy(),
                  c2=coeff[1].copy(), fl=np.array([a0], f32),
                  dims=np.array([Hi, Wi, C, Kh, Kx, s.stride_height, s.stride_width, act]))
    elif cls == "GemmGeluMulBfp16Wrapper":
        M, K, N = s.m_iteration * s.sv_m_dim, s.k_iteration * s.sv_k_dim, s.n_iteration * s.sv_n_dim
        gate, up = s.split_gate_up(np.asarray(data["matB"], f32).reshape(K, N))
        ofm = "bfp16"
        fx = dict(a=bfp16(s.converter, data["matA"].reshape(M, K)),
                  bt_gate=bfp16(s.converter, np.ascontiguousarray(gate), -2),
                  bt_up=bfp16(s.converter, np.ascontiguousarray(up), -2),
                  dims=np.array([M, K, N // 2, int(s.activation)]))
        if s.do_bias and "bias" in data:
            bg, bu = s.split_gate_up(np.asarray(data["bias"], f32).reshape(1, N))
            fx["bias_gate"], fx["bias_up"] = bg.reshape(-1).copy(), bu.reshape(-1).copy()
        want = encode_bfp16(np.asarray(data["ofm_0"], f32).reshape(-1), Rnd.kHalfAway)
    else:
        raise fm.Skip(f"no entry for {cls}")
    # the entry is named after the kernel the directory builds
    entry = d[:-len("_wrapper")].replace("_channelwise", "").replace("fp16xfp8", "fp8")
    fx.update(entry=entry, bias=bias, ofm=ofm, cls=cls,
              out=np.ascontiguousarray(want).view(np.uint8).reshape(-1))
    return fx


@pytest.mark.parametrize("d,case", fm.cases(DIRS, __file__))
def test_fixture(d, case):
    z = fm.load(case)
    entry, ofm = str(z["entry"]), str(z["ofm"])
    want = z["out"]
    nb = want.size
    bias = fm.opt(z, "bias")
    if entry == "gemm_bfp16":
        m, k, n = (int(v) for v in z["dims"])
        narrow, act, tr = (int(v) for v in z["flags"])
        got = call("npu_gemm_bfp16", z["a"], z["bt"], bias, OUT, I64(m), I64(k), I64(n), narrow, act,
                   float(z["alpha"][0]), tr, T[ofm], *(float(v) for v in z["q"]), out_nbytes=nb,
                   out_dtype=np.uint8)
    elif entry == "gemm_gelu_mul_bfp16":
        m, k, n, act = (int(v) for v in z["dims"])
        got = call("npu_gemm_gelu_mul_bfp16", z["a"], z["bt_gate"], z["bt_up"], fm.opt(z, "bias_gate"),
                   fm.opt(z, "bias_up"), OUT, I64(m), I64(k), I64(n), act, T[ofm], out_nbytes=nb,
                   out_dtype=np.uint8)
    elif entry in ("gemm_add_bfp16", "gemm_mul_bfp16"):
        m, k, n, m2, sv_m, sv_n, sv_k = (int(v) for v in z["dims"])
        got = call("npu_" + entry, z["a"], z["bt"], bias, z["ifm2"], T[str(z["ifm2_t"])], I64(m2),
                   OUT, I64(m), I64(k), I64(n), I64(sv_m), I64(sv_n), I64(sv_k), T[ofm],
                   out_nbytes=nb, out_dtype=np.uint8)
    elif entry in ("conv_bfp16", "conv_add_bfp16", "conv_mx6"):
        h, w, ci, kh, kw, co, sh, sw, kh_sv, ci_sv, co_sv, narrow, act, h2, w2, c2 = (int(v) for v in z["dims"])
        geom = (I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), sh, sw, I64(kh_sv), I64(ci_sv),
                I64(co_sv))
        if entry == "conv_add_bfp16":
            assert (narrow, act) == (0, 0)
            args = (z["ifm2"], I64(h2), I64(w2), I64(c2), OUT, *geom, T[ofm])
        else:
            args = (OUT, *geom, narrow, act, float(z["alpha"][0]), T[ofm])
        got = call("npu_" + entry, z["ifm"], z["wts"], bias, *args, out_nbytes=nb,
                   out_dtype=np.uint8)
    elif entry == "gemm_mx6":
        m, k, n, narrow, act = (int(v) for v in z["dims"])
        got = call("npu_gemm_mx6", z["a"], z["bt"], bias, OUT, I64(m), I64(k), I64(n), narrow, act,
                   float(z["alpha"][0]), T[ofm], out_nbytes=nb, out_dtype=np.uint8)
    elif entry == "gemm_fp8":
        m, k, n = (int(v) for v in z["dims"])
        got = call("npu_gemm_fp8", z["a"], z["b"], z["c0"], z["c2"], OUT, I64(m), I64(k), I64(n), T[ofm],
                   out_nbytes=nb, out_dtype=np.uint8)
    elif entry in ("conv_fp8", "conv_add_fp8"):
        h, w, ci, kh, kw, co, sh, sw, kh_sv, ci_sv, co_sv, act, h2, w2, c2 = (int(v) for v in z["dims"])
        a0, a1, coeff = (float(v) for v in z["fl"])
        geom = (I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), sh, sw, I64(kh_sv), I64(ci_sv),
                I64(co_sv))
        if entry == "conv_add_fp8":
            assert act == 0
            args = (z["ifm2"], T[str(z["ifm2_t"])], I64(h2), I64(w2), I64(c2), coeff, OUT, *geom,
                    T[ofm])
        else:
            args = (OUT, *geom, act, a0, a1, fm.opt(z, "slope"), T[ofm])
        head = (z["ifm"],) if entry == "conv_add_fp8" else (z["ifm"], T[str(z["ifm_t"])],
                                                            float(z["ifm_r"][0]))
        got = call("npu_" + entry, *head, z["wts"], z["c0"], z["c2"], *args, out_nbytes=nb,
                   out_dtype=np.uint8)
    elif entry == "depthwise_conv_fp8":
        h, w, c, kh, kw, sh, sw, act = (int(v) for v in z["dims"])
        got = call("npu_depthwise_conv_fp8", z["ifm"], z["wts"], z["c0"], z["c2"], OUT, I64(h), I64(w), I64(c),
                   I64(kh), I64(kw), sh, sw, act, float(z["fl"][0]), T[ofm], out_nbytes=nb, out_dtype=np.uint8)
    else:
        raise AssertionError(f"unknown entry {entry}")
    bad = np.nonzero(np.asarray(got).reshape(-1) != want)[0]
    assert bad.size == 0, f"{z['cls']}: {bad.size}/{nb} bytes differ, first at {bad[:8].tolist()}"
