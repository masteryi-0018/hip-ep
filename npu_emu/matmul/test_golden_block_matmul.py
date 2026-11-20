# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The bfp16 / mx6 / fp8 matmul and convolution entry points against the numpy
reference, bit for bit."""

import numpy as np
import pytest

import golden_block_matmul as gb
import golden_act as ga
from golden_block_fp import decode_bfp16, encode_bfp16, encode_mx6
from golden_float_cvt import f32_to_bf16, f32_to_f16, f32_to_fp8, kFtzIn
from golden_int_round import Rnd
from npu import I64, NPU_BF16, NPU_BFP16, NPU_F16, NPU_FP8, NPU_MX6, NPU_U8, OUT
from testing import call_n, rejected
from testing import same_bytes as same

RNG = np.random.default_rng(2049)
OFM = {"bf16": NPU_BF16, "bfp16": NPU_BFP16, "mx6": NPU_MX6, "u8": NPU_U8, "f16": NPU_F16,
       "fp8": NPU_FP8}


def spread(shape, oct_=8, blk=8):
    """Random values whose blocks of `blk` along the last axis span 2^+-oct_."""
    v = RNG.standard_normal(shape)
    sc = np.exp2(RNG.integers(-oct_, oct_ + 1, shape[:-1] + (-(-shape[-1] // blk),)))
    return (v * np.repeat(sc, blk, axis=-1)[..., :shape[-1]]).astype(np.float32)


def bfp16(shape, oct_=8):
    return encode_bfp16(spread(shape, oct_).reshape(-1), Rnd.kHalfEven)


def mx6(shape, oct_=6):
    return encode_mx6(spread(shape, oct_, 16).reshape(-1), Rnd.kHalfAway)


def fp8(shape, oct_=4):
    b = f32_to_fp8(spread(shape, oct_), Rnd.kHalfEven).reshape(-1)
    return np.where((b & 0x7F) == 0x7F, b ^ 1, b).astype(np.uint8)  # no NaN


def out_bytes(ofm, n):
    return {"bf16": 2 * n, "f16": 2 * n, "fp8": n, "u8": n, "bfp16": n // 8 * 9,
            "mx6": n // 16 * 12}[ofm]


@pytest.mark.parametrize("mkn", [(8, 64, 16), (5, 192, 24), (16, 128, 64)])
@pytest.mark.parametrize("narrow,act", [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3)])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
@pytest.mark.parametrize("transpose", [0, 1])
def test_gemm_bfp16(mkn, narrow, act, ofm, transpose):
    m, k, n = mkn
    if ofm == "bfp16" and (m if transpose else n) % 8:
        pytest.skip("bfp16 output blocks of 8")
    a, bt = bfp16((m, k)), bfp16((n, k))
    bias = None if transpose else (RNG.standard_normal(n) * 30).astype(np.float32)
    want = gb.gemm_bfp16(a, bt, bias, m, k, n, narrow, act, 0.15625, transpose, ofm)
    got = call_n("npu_gemm_bfp16", a, bt, bias, OUT, I64(m), I64(k), I64(n), narrow, act, 0.15625,
                 transpose, OFM[ofm], 0.0, 0.0, n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


def act_operands(m, k, n):
    """Operands whose GEMM lands in [-8, 8]."""
    a = encode_bfp16((RNG.standard_normal((m, k)) * 0.5).astype(np.float32).reshape(-1), Rnd.kHalfEven)
    bt = encode_bfp16((RNG.standard_normal((n, k)) * 0.5 / np.sqrt(k / 16)).astype(np.float32).reshape(-1),
                    Rnd.kHalfEven)
    return a, bt


@pytest.mark.parametrize("act", [5, 6, 7], ids=["silu", "gelu", "tanh"])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
@pytest.mark.parametrize("transpose", [0, 1])
def test_gemm_bfp16_core_act(act, ofm, transpose):
    m, k, n = 16, 192, 64
    a, bt = act_operands(m, k, n)
    bias = None if transpose else RNG.uniform(-2, 2, n).astype(np.float32)
    want = gb.gemm_bfp16(a, bt, bias, m, k, n, 1, act, 0.0, transpose, ofm)
    got = call_n("npu_gemm_bfp16", a, bt, bias, OUT, I64(m), I64(k), I64(n), 1, act, 0.0, transpose,
                 OFM[ofm], 0.0, 0.0, n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


# bf16 inputs whose SiLU / Tanh lands exactly half-way between two bf16 values
CORE_TIES = {5: [0xBE00], 7: [0x3CFA, 0x3CFB, 0x3CFC, 0x3CFD, 0x3CFE, 0x3D06, 0x3D07, 0x3D0E]}


@pytest.mark.parametrize("act", [5, 7], ids=["silu", "tanh"])
def test_gemm_bfp16_core_act_ties(act):
    """A = B = 0, bias set to inputs whose activation is a bf16 tie: half away."""
    m, k = 8, 64
    x = np.array(CORE_TIES[act] * 8, np.uint32)[:64]
    n = x.size
    bias = (x << 16).view(np.float32)
    a = encode_bfp16(np.zeros(m * k, np.float32), Rnd.kHalfEven)
    bt = encode_bfp16(np.zeros(n * k, np.float32), Rnd.kHalfEven)
    want = gb.gemm_bfp16(a, bt, bias, m, k, n, 1, act, 0.0, 0, "bf16")
    core = (ga.hw_silu_f32(bias) if act == 5 else
            ga.hw_sigtanh_f32(bias, True, ga.SIGTANH_K0, ga.TANH_K1, ga.TANH_K2))
    away, even = (f32_to_bf16(core, r, kFtzIn) for r in (Rnd.kHalfAway, Rnd.kHalfEven))
    assert np.all(away != even)                           # every value is a tie
    got = call_n("npu_gemm_bfp16", a, bt, bias, OUT, I64(m), I64(k), I64(n), 1, act, 0.0, 0,
                 OFM["bf16"], 0.0, 0.0, n=out_bytes("bf16", m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("act", [0, 5, 6, 7], ids=["bypass", "silu", "gelu", "tanh"])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
@pytest.mark.parametrize("bias", [0, 1])
def test_gemm_gelu_mul_bfp16(act, ofm, bias):
    m, k, n = 16, 128, 64
    a, bg = act_operands(m, k, n)
    _, bu = act_operands(m, k, n)
    big, biu = ((RNG.uniform(-2, 2, n).astype(np.float32) for _ in range(2)) if bias else (None, None))
    want = gb.gemm_gelu_mul_bfp16(a, bg, bu, big, biu, m, k, n, act, ofm)
    got = call_n("npu_gemm_gelu_mul_bfp16", a, bg, bu, big, biu, OUT, I64(m), I64(k), I64(n), act,
                 OFM[ofm], n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("case", [
    # m, k, n, ifm2, m2, sv_m, sv_n, sv_k, is_mul
    (8, 192, 128, "bf16", 8, 4, 64, 192, 0), (8, 192, 64, "bf16", 1, 8, 64, 192, 0),
    (8, 192, 128, "bfp16", 8, 4, 64, 128, 0), (8, 128, 128, "bfp16", 8, 4, 64, 64, 0),
    (6, 256, 64, "bfp16", 1, 2, 64, 128, 0), (8, 192, 64, "bf16", 8, 8, 64, 64, 1),
    (4, 64, 64, "bf16", 1, 4, 64, 64, 1)])
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_gemm_eltwise_bfp16(case, bias, ofm):
    m, k, n, it, m2, sv_m, sv_n, sv_k, is_mul = case
    a, bt = bfp16((m, k)), bfp16((n, k))
    b = (RNG.standard_normal(n) * 30).astype(np.float32) if bias else None
    if it == "bfp16":
        ifm2 = bfp16((m2, n), 12)
    else:
        ifm2 = f32_to_bf16(spread((m2, n), 6) * 100, Rnd.kHalfEven).reshape(-1)
    want = gb.gemm_eltwise_bfp16(a, bt, b, ifm2, it == "bfp16", m2, m, k, n, sv_m, sv_n, sv_k, is_mul, ofm)
    got = call_n("npu_gemm_mul_bfp16" if is_mul else "npu_gemm_add_bfp16", a, bt, b, ifm2, OFM[it],
                 I64(m2), OUT, I64(m), I64(k), I64(n), I64(sv_m), I64(sv_n), I64(sv_k), OFM[ofm],
                 n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


CONV = [  # h, w, ci, kh, kw, co, stride, kh_sv, ci_sv, co_sv
    (4, 6, 64, 3, 3, 16, 1, 3, 64, 16), (5, 9, 16, 3, 3, 32, 1, 1, 16, 32),
    (4, 9, 128, 1, 3, 64, 2, 1, 64, 64), (5, 7, 8, 3, 3, 8, 1, 3, 8, 8),
    (6, 6, 128, 2, 2, 32, 1, 1, 128, 16)]


@pytest.mark.parametrize("geom", CONV)
@pytest.mark.parametrize("narrow,act", [(0, 0), (1, 1), (1, 3)])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_conv_bfp16(geom, narrow, act, ofm):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ifm, wts = bfp16((h, w, ci)), bfp16((kh, kw, co, ci))
    bias = (RNG.standard_normal(co) * 30).astype(np.float32)
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    want = gb.conv_block(ifm, wts, bias, None, 1, 1, 1, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv, co_sv,
                         narrow, act, 0.15625, ofm)
    got = call_n("npu_conv_bfp16", ifm, wts, bias, OUT, I64(h), I64(w), I64(ci),
                 I64(kh), I64(kw), I64(co), s, s, I64(kh_sv), I64(ci_sv), I64(co_sv), narrow, act, 0.15625,
                 OFM[ofm], n=out_bytes(ofm, ho * wo * co), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("geom", CONV[:3])
@pytest.mark.parametrize("bshape", ["full", "chan", "pixel"])
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_conv_add_bfp16(geom, bshape, bias, ofm):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ifm, wts = bfp16((h, w, ci)), bfp16((kh, kw, co, ci))
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    shp = {"full": (ho, wo, co), "chan": (1, 1, co), "pixel": (ho, wo, 1)}[bshape]
    ifm2 = f32_to_bf16(spread(shp, 4, shp[-1]) * 50, Rnd.kHalfEven).reshape(-1)
    b = (RNG.standard_normal(co) * 30).astype(np.float32) if bias else None
    want = gb.conv_block(ifm, wts, b, ifm2, *shp, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv, co_sv, 0, 0, 0.0, ofm)
    tail = (I64(shp[0]), I64(shp[1]), I64(shp[2]), OUT, I64(h), I64(w), I64(ci), I64(kh), I64(kw),
            I64(co), s, s, I64(kh_sv), I64(ci_sv), I64(co_sv), OFM[ofm])
    nb = out_bytes(ofm, ho * wo * co)
    same(call_n("npu_conv_add_bfp16", ifm, wts, b, ifm2, *tail, n=nb, dt=np.uint8), want)
    assert rejected("npu_conv_add_bfp16", ifm, wts, b, None, *tail, nbytes=nb)  # no residual


@pytest.mark.parametrize("mkn", [(4, 128, 32), (3, 384, 16)])
@pytest.mark.parametrize("narrow,act", [(0, 0), (1, 2), (1, 3)])
@pytest.mark.parametrize("ofm", ["bf16", "mx6"])
def test_gemm_mx6(mkn, narrow, act, ofm):
    m, k, n = mkn
    a, bt = mx6((m, k)), mx6((n, k))
    bias = (RNG.standard_normal(n) * 30).astype(np.float32)
    want = gb.gemm_mx6(a, bt, bias, m, k, n, narrow, act, 0.15625, ofm)
    got = call_n("npu_gemm_mx6", a, bt, bias, OUT, I64(m), I64(k), I64(n), narrow, act, 0.15625, OFM[ofm],
                 n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("geom", [(4, 6, 128, 3, 3, 16, 1, 3, 128, 16), (4, 9, 256, 1, 3, 64, 2, 1, 128, 32)])
@pytest.mark.parametrize("narrow,act", [(0, 0), (1, 1)])
@pytest.mark.parametrize("ofm", ["bf16", "mx6"])
def test_conv_mx6(geom, narrow, act, ofm):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ifm, wts = mx6((h, w, ci)), mx6((kh, kw, co, ci))
    bias = (RNG.standard_normal(co) * 30).astype(np.float32)
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    want = gb.conv_block(ifm, wts, bias, None, 1, 1, 1, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv, co_sv,
                         narrow, act, 0.15625, ofm, mx6=True)
    got = call_n("npu_conv_mx6", ifm, wts, bias, OUT, I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s,
                 I64(kh_sv), I64(ci_sv), I64(co_sv), narrow, act, 0.15625, OFM[ofm],
                 n=out_bytes(ofm, ho * wo * co), dt=np.uint8)
    same(got, want)


def coeffs(n):
    return (RNG.standard_normal(n) * 0.3).astype(np.float32), np.exp2(RNG.uniform(-6, 2, n)).astype(np.float32)


@pytest.mark.parametrize("mkn", [(8, 64, 16), (5, 100, 24), (16, 448, 64)])
@pytest.mark.parametrize("ofm", ["fp8", "f16", "bf16"])
def test_gemm_fp8(mkn, ofm):
    m, k, n = mkn
    a, b = fp8((m, k)), fp8((k, n))
    c0, c2 = coeffs(n)
    want = gb.gemm_fp8(a, b, c0, c2, m, k, n, ofm)
    got = call_n("npu_gemm_fp8", a, b, c0, c2, OUT, I64(m), I64(k), I64(n), OFM[ofm], n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


FCONV = [(4, 6, 64, 3, 3, 16, 1, 3, 64, 16), (5, 9, 16, 3, 3, 32, 1, 1, 16, 32),
         (4, 9, 128, 1, 3, 64, 2, 1, 64, 64), (5, 7, 8, 3, 3, 8, 1, 3, 8, 8)]


@pytest.mark.parametrize("geom", FCONV)
@pytest.mark.parametrize("act", [0, 1, 2, 3, 8, 9])
@pytest.mark.parametrize("ofm", ["fp8", "f16", "bf16"])
def test_conv_fp8(geom, act, ofm):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ifm, wts = fp8((h, w, ci)), fp8((kh, kw, ci, co))
    c0, c2 = coeffs(co)
    slope = (RNG.standard_normal(co) * 0.5).astype(np.float32)
    a0, a1 = (0.3, 0.0) if act == 3 else ((2.5, 0.0) if act == 2 else (-1.5, 2.25))
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    want = gb.conv_fp8(ifm, wts, c0, c2, None, None, 1, 1, 1, 0.0, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv,
                       co_sv, act, a0, a1, slope, ofm)
    got = call_n("npu_conv_fp8", ifm, NPU_FP8, 0.0, wts, c0, c2, OUT, I64(h), I64(w),
                 I64(ci), I64(kh), I64(kw), I64(co), s, s, I64(kh_sv), I64(ci_sv), I64(co_sv), act, a0, a1, slope,
                 OFM[ofm], n=out_bytes(ofm, ho * wo * co), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("geom", FCONV[:3])
@pytest.mark.parametrize("it", ["fp8", "bf16", "f16"])
@pytest.mark.parametrize("bshape", ["full", "chan"])
@pytest.mark.parametrize("ofm", ["fp8", "f16"])
def test_conv_add_fp8(geom, it, bshape, ofm):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ifm, wts = fp8((h, w, ci)), fp8((kh, kw, ci, co))
    c0, c2 = coeffs(co)
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    shp = {"full": (ho, wo, co), "chan": (1, 1, co)}[bshape]
    v = spread(shp, 3, shp[-1])
    ifm2 = {"fp8": lambda: fp8(shp), "bf16": lambda: f32_to_bf16(v, Rnd.kHalfEven).reshape(-1),
            "f16": lambda: f32_to_f16(v, Rnd.kHalfEven).reshape(-1)}[it]()
    want = gb.conv_fp8(ifm, wts, c0, c2, ifm2, it, *shp, 0.7, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv, co_sv,
                       0, 0.0, 0.0, None, ofm)
    tail = (OFM[it], I64(shp[0]), I64(shp[1]), I64(shp[2]), 0.7, OUT, I64(h), I64(w), I64(ci),
            I64(kh), I64(kw), I64(co), s, s, I64(kh_sv), I64(ci_sv), I64(co_sv), OFM[ofm])
    nb = out_bytes(ofm, ho * wo * co)
    same(call_n("npu_conv_add_fp8", ifm, wts, c0, c2, ifm2, *tail, n=nb, dt=np.uint8), want)
    assert rejected("npu_conv_add_fp8", ifm, wts, c0, c2, None, *tail, nbytes=nb)  # no residual


@pytest.mark.parametrize("kk,s", [((3, 3), 1), ((3, 3), 2), ((7, 7), 1), ((1, 1), 1)])
@pytest.mark.parametrize("act", [0, 1, 2, 3])
@pytest.mark.parametrize("ofm", ["fp8", "f16", "bf16"])
def test_depthwise_fp8(kk, s, act, ofm):
    kh, kw = kk
    c = 64
    h, w = 3 * s + kh, 4 * s + kw
    ifm, wts = fp8((h, w, c)), fp8((kh, kw, c))
    c0, c2 = coeffs(c)
    a0 = 0.3 if act == 3 else 2.5
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    want = gb.depthwise_fp8(ifm, wts, c0, c2, h, w, c, kh, kw, s, s, act, a0, ofm)
    got = call_n("npu_depthwise_conv_fp8", ifm, wts, c0, c2, OUT, I64(h), I64(w), I64(c), I64(kh), I64(kw), s, s,
                 act, a0, OFM[ofm], n=out_bytes(ofm, ho * wo * c), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("sv_k", [128, 64])
def test_gemm_add_bfp16_e_block_order(sv_k):
    """ifm2 ~ -(A @ B), so the output depends on where the ifm2 MMAC enters."""
    m, k, n, sv_m, sv_n = 8, 256, 128, 4, 64
    av, bv = spread((m, k), 12), spread((n, k), 12)
    a, bt = encode_bfp16(av.reshape(-1), Rnd.kHalfEven), encode_bfp16(bv.reshape(-1), Rnd.kHalfEven)
    ab = decode_bfp16(a).reshape(m, k).astype(np.float64) @ decode_bfp16(bt).reshape(n, k).T.astype(np.float64)
    ifm2 = encode_bfp16((-ab * (1 + RNG.uniform(-1e-3, 1e-3, ab.shape))).astype(np.float32).reshape(-1), Rnd.kHalfEven)
    want = gb.gemm_eltwise_bfp16(a, bt, None, ifm2, True, m, m, k, n, sv_m, sv_n, sv_k, 0, "bf16")
    for other in ("early", "late"):  # the data must tell the orders apart
        e_all = 64 if other == "early" else 10 ** 9
        alt = gb.gemm_eltwise_bfp16(a, bt, None, ifm2, True, m, m, k, n, 0 if other == "late" else sv_m,
                                    0 if other == "late" else sv_n, e_all, 0, "bf16")
        if not (other == "early" and sv_k == 64):
            assert np.any(alt != want), other
    got = call_n("npu_gemm_add_bfp16", a, bt, None, ifm2, NPU_BFP16, I64(m), OUT, I64(m), I64(k), I64(n),
                 I64(sv_m), I64(sv_n), I64(sv_k), NPU_BF16, n=out_bytes("bf16", m * n), dt=np.uint8)
    same(got, want)


TIE_ALPHA = float(np.uint32(0x3E208000).view(np.float32))   # halfway between two bf16
AWAY_ALPHA = float(np.uint32(0x3E210000).view(np.float32))  # the tie rounded away


@pytest.mark.parametrize("fmt", ["bfp16", "mx6"])
def test_gemm_lrelu_alpha_tie(fmt):
    """The alpha is cast to bf16 half to even."""
    m, k, n = (8, 64, 16) if fmt == "bfp16" else (4, 128, 32)
    pack = bfp16 if fmt == "bfp16" else mx6
    a, bt = pack((m, k)), pack((n, k))
    bias = (RNG.standard_normal(n) * 30).astype(np.float32)

    def gold(alpha):
        if fmt == "bfp16":
            return gb.gemm_bfp16(a, bt, bias, m, k, n, 1, 3, alpha, 0, "bf16")
        return gb.gemm_mx6(a, bt, bias, m, k, n, 1, 3, alpha, "bf16")

    want = gold(TIE_ALPHA)
    assert (want != gold(AWAY_ALPHA)).any()
    tail = (0, OFM["bf16"], 0.0, 0.0) if fmt == "bfp16" else (OFM["bf16"],)
    got = call_n(f"npu_gemm_{fmt}", a, bt, bias, OUT, I64(m), I64(k), I64(n), 1, 3, TIE_ALPHA, *tail,
                 n=out_bytes("bf16", m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("fmt", ["bfp16", "mx6"])
def test_conv_lrelu_alpha_tie(fmt):
    if fmt == "bfp16":
        h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = CONV[0]
        ifm, wts = bfp16((h, w, ci)), bfp16((kh, kw, co, ci))
    else:
        h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = 4, 6, 128, 3, 3, 16, 1, 3, 128, 16
        ifm, wts = mx6((h, w, ci)), mx6((kh, kw, co, ci))
    bias = (RNG.standard_normal(co) * 30).astype(np.float32)
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1

    def gold(alpha):
        return gb.conv_block(ifm, wts, bias, None, 1, 1, 1, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv,
                             co_sv, 1, 3, alpha, "bf16", mx6=fmt == "mx6")

    want = gold(TIE_ALPHA)
    assert (want != gold(AWAY_ALPHA)).any()
    got = call_n(f"npu_conv_{fmt}", ifm, wts, bias, OUT, I64(h), I64(w), I64(ci), I64(kh), I64(kw),
                 I64(co), s, s, I64(kh_sv), I64(ci_sv), I64(co_sv), 1, 3, TIE_ALPHA, OFM["bf16"],
                 n=out_bytes("bf16", ho * wo * co), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("mkn", [(8, 64, 16), (5, 192, 24)])
@pytest.mark.parametrize("transpose", [0, 1])
def test_gemm_bfp16_u8(mkn, transpose):
    """The uint8 output cast: both saturation bounds and rounding ties are reached."""
    m, k, n = mkn
    a, bt = bfp16((m, k)), bfp16((n, k))
    bias = None if transpose else (RNG.standard_normal(n) * 30).astype(np.float32)
    q = (float(np.float32(1.0) / np.float32(19.2)), 117.0)
    want = gb.gemm_bfp16(a, bt, bias, m, k, n, 1, 0, 0.0, transpose, "u8", q)
    assert want.min() == 0 and want.max() == 255
    got = call_n("npu_gemm_bfp16", a, bt, bias, OUT, I64(m), I64(k), I64(n), 1, 0, 0.0, transpose,
                 OFM["u8"], *q, n=out_bytes("u8", m * n), dt=np.uint8)
    same(got, want)
    assert rejected("npu_gemm_bfp16", a, bt, bias, OUT, I64(m), I64(k), I64(n), 0, 0, 0.0, transpose,
                    OFM["u8"], *q, nbytes=m * n)  # the cast takes the bf16 value


@pytest.mark.parametrize("geom", FCONV)
@pytest.mark.parametrize("ofm", ["fp8", "bf16"])
def test_conv_fp8_f16_input(geom, ofm):
    """f16 activations narrowed to E4M3 on load: the f16 product, the +-448 clamp in f16 and the
    E4M3 rounding all show."""
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    x16 = f32_to_f16(spread((h, w, ci), 5) * 6, Rnd.kHalfEven).reshape(-1)
    r = float(np.float16(1.0 / 0.37))
    wts = fp8((kh, kw, ci, co))
    c0, c2 = coeffs(co)
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    ifm = gb.narrow_f16_fp8(x16, r)
    want = gb.conv_fp8(ifm, wts, c0, c2, None, None, 1, 1, 1, 0.0, h, w, ci, kh, kw, co, s, s, kh_sv,
                       ci_sv, co_sv, 0, 0.0, 0.0, None, ofm)
    got = call_n("npu_conv_fp8", x16, NPU_F16, r, wts, c0, c2, OUT, I64(h), I64(w), I64(ci), I64(kh),
                 I64(kw), I64(co), s, s, I64(kh_sv), I64(ci_sv), I64(co_sv), 0, 0.0, 0.0, None,
                 OFM[ofm], n=out_bytes(ofm, ho * wo * co), dt=np.uint8)
    same(got, want)


# ---- special values: NaN (quiet, signalling, payload), +-inf, +-0, subnormals, largest finite,
# all-zero blocks, in operands, bias, coefficients and the narrowing / activation parameters ----

SPECIAL_F32 = np.array([np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-40, -1e-40, 1e-45, 3.4028235e38,
                        -3.4028235e38, 1.0, -1.0], np.float32)
SPECIAL_F32 = np.concatenate([SPECIAL_F32, np.array([0x7FC12345, 0xFFA00001], np.uint32).view(np.float32)])
SPECIAL_FP8 = np.array([0x7F, 0xFF, 0x00, 0x80, 0x01, 0x81, 0x07, 0x7E, 0xFE, 0x38, 0xB8], np.uint8)
SPECIAL_BF16 = np.array([0x7FC0, 0xFFC0, 0x7FC1, 0x7F81, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x007F, 0x0001,
                         0x7F7F, 0xFF7F, 0x3F80], np.uint16)
SPECIAL_F16 = np.array([0x7E00, 0xFE00, 0x7C01, 0x7C00, 0xFC00, 0x0000, 0x8000, 0x03FF, 0x0001, 0x7BFF,
                        0xFBFF, 0x3C00], np.uint16)
F32_MAX = float(np.float32(3.4028235e38))
NANV = [float(v) for v in np.array([0x7FC00000, 0xFFC00000, 0xFFC12345, 0x7F800001], np.uint32).view(np.float32)]
COEF_SPECIAL = [*NANV, np.inf, -np.inf, 0.0, -0.0, 1e-40, -3e-39, F32_MAX, 0.15625]


def sp_f32(n, rate=0.3, scale=4.0):
    v = (RNG.standard_normal(n) * scale).astype(np.float32)
    pos = RNG.random(n) < rate
    v[pos] = SPECIAL_F32[RNG.integers(0, SPECIAL_F32.size, int(pos.sum()))]
    return v


def sp_blocks(shape, blk, rate=0.04, zero_blocks=0.1):
    """spread() values with special elements and some all-zero blocks along the last axis."""
    v = spread(shape, 6, blk).reshape(-1)
    pos = RNG.random(v.size) < rate
    v[pos] = SPECIAL_F32[RNG.integers(0, SPECIAL_F32.size, int(pos.sum()))]
    z = np.repeat(RNG.random(v.size // blk) < zero_blocks, blk)
    v[z] = 0.0
    return v


def sp_bfp16(shape):
    return encode_bfp16(sp_blocks(shape, 8), Rnd.kHalfEven)


def sp_mx6(shape):
    return encode_mx6(sp_blocks(shape, 16), Rnd.kHalfAway)


def sp_fp8(shape, rate=0.12):
    c = fp8(shape).reshape(-1)
    pos = RNG.random(c.size) < rate
    c[pos] = SPECIAL_FP8[RNG.integers(0, SPECIAL_FP8.size, int(pos.sum()))]
    return c


def sp_codes(shape, table, rate=0.25):
    """bf16 / f16 codes of random values with a share of the special ones."""
    v = f32_to_bf16(spread(shape, 3, shape[-1]), Rnd.kHalfEven) if table is SPECIAL_BF16 else \
        f32_to_f16(spread(shape, 3, shape[-1]), Rnd.kHalfEven)
    v = v.reshape(-1).copy()
    pos = RNG.random(v.size) < rate
    v[pos] = table[RNG.integers(0, table.size, int(pos.sum()))]
    return v


@pytest.mark.parametrize("mkn", [(8, 64, 16), (5, 192, 24)])
@pytest.mark.parametrize("narrow,act,alpha", [(0, 0, 0.0), (1, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0), (1, 5, 0.0),
                                              (1, 6, 0.0), (1, 7, 0.0)] + [(1, 3, al) for al in COEF_SPECIAL],
                         ids=lambda v: f"{v:g}")
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
@pytest.mark.parametrize("transpose", [0, 1])
def test_special_gemm_bfp16(mkn, narrow, act, alpha, ofm, transpose):
    m, k, n = mkn
    if ofm == "bfp16" and (m if transpose else n) % 8:
        pytest.skip("bfp16 output blocks of 8")
    a, bt = sp_bfp16((m, k)), sp_bfp16((n, k))
    bias = None if transpose else sp_f32(n)
    want = gb.gemm_bfp16(a, bt, bias, m, k, n, narrow, act, alpha, transpose, ofm)
    got = call_n("npu_gemm_bfp16", a, bt, bias, OUT, I64(m), I64(k), I64(n), narrow, act, alpha, transpose,
                 OFM[ofm], 0.0, 0.0, n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("q", [(NANV[0], 117.0), (NANV[2], NANV[3]), (np.inf, 0.0), (-np.inf, 5.0), (0.0, np.inf), (0.0, -np.inf),
                               (1.0, NANV[2]), (-0.0, -0.0), (1e-40, 1e-40), (F32_MAX, F32_MAX),
                               (1.0, 0.0), (0.05, 117.0)], ids=lambda v: "-".join(f"{x:g}" for x in v))
@pytest.mark.parametrize("transpose", [0, 1])
def test_special_gemm_bfp16_u8(q, transpose):
    m, k, n = 8, 128, 16
    a, bt = sp_bfp16((m, k)), sp_bfp16((n, k))
    bias = None if transpose else sp_f32(n)
    want = gb.gemm_bfp16(a, bt, bias, m, k, n, 1, 0, 0.0, transpose, "u8", q)
    got = call_n("npu_gemm_bfp16", a, bt, bias, OUT, I64(m), I64(k), I64(n), 1, 0, 0.0, transpose,
                 OFM["u8"], *q, n=out_bytes("u8", m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("act", [0, 5, 6, 7], ids=["bypass", "silu", "gelu", "tanh"])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
@pytest.mark.parametrize("bias", [0, 1])
def test_special_gemm_gelu_mul_bfp16(act, ofm, bias):
    m, k, n = 16, 128, 64
    a, bg, bu = sp_bfp16((m, k)), sp_bfp16((n, k)), sp_bfp16((n, k))
    big, biu = (sp_f32(n), sp_f32(n)) if bias else (None, None)
    want = gb.gemm_gelu_mul_bfp16(a, bg, bu, big, biu, m, k, n, act, ofm)
    got = call_n("npu_gemm_gelu_mul_bfp16", a, bg, bu, big, biu, OUT, I64(m), I64(k), I64(n), act, OFM[ofm],
                 n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("case", [
    # m, k, n, ifm2, m2, sv_m, sv_n, sv_k, is_mul
    (8, 192, 128, "bf16", 8, 4, 64, 192, 0), (8, 192, 64, "bf16", 1, 8, 64, 192, 0),
    (8, 192, 128, "bfp16", 8, 4, 64, 128, 0), (8, 128, 128, "bfp16", 8, 4, 64, 64, 0),
    (6, 256, 64, "bfp16", 1, 2, 64, 128, 0), (8, 192, 64, "bf16", 8, 8, 64, 64, 1),
    (4, 64, 64, "bf16", 1, 4, 64, 64, 1)])
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_special_gemm_eltwise_bfp16(case, bias, ofm):
    m, k, n, it, m2, sv_m, sv_n, sv_k, is_mul = case
    a, bt = sp_bfp16((m, k)), sp_bfp16((n, k))
    b = sp_f32(n) if bias else None
    ifm2 = sp_bfp16((m2, n)) if it == "bfp16" else sp_codes((m2, n), SPECIAL_BF16)
    want = gb.gemm_eltwise_bfp16(a, bt, b, ifm2, it == "bfp16", m2, m, k, n, sv_m, sv_n, sv_k, is_mul, ofm)
    got = call_n("npu_gemm_mul_bfp16" if is_mul else "npu_gemm_add_bfp16", a, bt, b, ifm2, OFM[it],
                 I64(m2), OUT, I64(m), I64(k), I64(n), I64(sv_m), I64(sv_n), I64(sv_k), OFM[ofm],
                 n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("geom", CONV)
@pytest.mark.parametrize("narrow,act,alpha", [(0, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0)] +
                         [(1, 3, al) for al in COEF_SPECIAL], ids=lambda v: f"{v:g}")
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_special_conv_bfp16(geom, narrow, act, alpha, ofm):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ifm, wts = sp_bfp16((h, w, ci)), sp_bfp16((kh, kw, co, ci))
    bias = sp_f32(co)
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    want = gb.conv_block(ifm, wts, bias, None, 1, 1, 1, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv, co_sv,
                         narrow, act, alpha, ofm)
    got = call_n("npu_conv_bfp16", ifm, wts, bias, OUT, I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s,
                 I64(kh_sv), I64(ci_sv), I64(co_sv), narrow, act, alpha, OFM[ofm],
                 n=out_bytes(ofm, ho * wo * co), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("geom", CONV[:3])
@pytest.mark.parametrize("bshape", ["full", "chan", "pixel"])
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_special_conv_add_bfp16(geom, bshape, bias, ofm):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ifm, wts = sp_bfp16((h, w, ci)), sp_bfp16((kh, kw, co, ci))
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    shp = {"full": (ho, wo, co), "chan": (1, 1, co), "pixel": (ho, wo, 1)}[bshape]
    ifm2 = sp_codes(shp, SPECIAL_BF16)
    b = sp_f32(co) if bias else None
    want = gb.conv_block(ifm, wts, b, ifm2, *shp, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv, co_sv, 0, 0, 0.0, ofm)
    tail = (I64(shp[0]), I64(shp[1]), I64(shp[2]), OUT, I64(h), I64(w), I64(ci), I64(kh), I64(kw),
            I64(co), s, s, I64(kh_sv), I64(ci_sv), I64(co_sv), OFM[ofm])
    same(call_n("npu_conv_add_bfp16", ifm, wts, b, ifm2, *tail, n=out_bytes(ofm, ho * wo * co), dt=np.uint8), want)


@pytest.mark.parametrize("mkn", [(4, 128, 32), (3, 384, 16)])
@pytest.mark.parametrize("narrow,act,alpha", [(0, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0)] +
                         [(1, 3, al) for al in COEF_SPECIAL], ids=lambda v: f"{v:g}")
@pytest.mark.parametrize("ofm", ["bf16", "mx6"])
def test_special_gemm_mx6(mkn, narrow, act, alpha, ofm):
    m, k, n = mkn
    a, bt = sp_mx6((m, k)), sp_mx6((n, k))
    bias = sp_f32(n)
    want = gb.gemm_mx6(a, bt, bias, m, k, n, narrow, act, alpha, ofm)
    got = call_n("npu_gemm_mx6", a, bt, bias, OUT, I64(m), I64(k), I64(n), narrow, act, alpha, OFM[ofm],
                 n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("geom", [(4, 6, 128, 3, 3, 16, 1, 3, 128, 16), (4, 9, 256, 1, 3, 64, 2, 1, 128, 32)])
@pytest.mark.parametrize("narrow,act", [(0, 0), (1, 1), (1, 3)])
@pytest.mark.parametrize("ofm", ["bf16", "mx6"])
def test_special_conv_mx6(geom, narrow, act, ofm):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ifm, wts = sp_mx6((h, w, ci)), sp_mx6((kh, kw, co, ci))
    bias = sp_f32(co)
    alpha = float("inf") if act == 3 else 0.0
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    want = gb.conv_block(ifm, wts, bias, None, 1, 1, 1, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv, co_sv,
                         narrow, act, alpha, ofm, mx6=True)
    got = call_n("npu_conv_mx6", ifm, wts, bias, OUT, I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s,
                 I64(kh_sv), I64(ci_sv), I64(co_sv), narrow, act, alpha, OFM[ofm],
                 n=out_bytes(ofm, ho * wo * co), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("mkn", [(8, 64, 16), (5, 100, 24)])
@pytest.mark.parametrize("ofm", ["fp8", "f16", "bf16"])
@pytest.mark.parametrize("rep", range(2))
def test_special_gemm_fp8(mkn, ofm, rep):
    m, k, n = mkn
    a, b = sp_fp8((m, k)), sp_fp8((k, n))
    c0, c2 = sp_f32(n), sp_f32(n)
    want = gb.gemm_fp8(a, b, c0, c2, m, k, n, ofm)
    got = call_n("npu_gemm_fp8", a, b, c0, c2, OUT, I64(m), I64(k), I64(n), OFM[ofm], n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


ACT_PARAMS = [(0, 0.0, 0.0), (1, 0.0, 0.0), (2, 2.5, 0.0), (3, 0.3, 0.0), (8, -1.5, 2.25), (9, 0.0, 0.0)] + \
    [(act, a0, a1) for act in (2, 3, 8) for a0 in COEF_SPECIAL[:5] for a1 in (NANV[2], 1.0)]


@pytest.mark.parametrize("geom", FCONV[:2])
@pytest.mark.parametrize("act,a0,a1", ACT_PARAMS, ids=lambda v: f"{v:g}")
@pytest.mark.parametrize("ofm", ["fp8", "f16", "bf16"])
def test_special_conv_fp8(geom, act, a0, a1, ofm):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ifm, wts = sp_fp8((h, w, ci)), sp_fp8((kh, kw, ci, co))
    c0, c2 = sp_f32(co), sp_f32(co)
    slope = sp_f32(co)
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    want = gb.conv_fp8(ifm, wts, c0, c2, None, None, 1, 1, 1, 0.0, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv,
                       co_sv, act, a0, a1, slope, ofm)
    got = call_n("npu_conv_fp8", ifm, NPU_FP8, 0.0, wts, c0, c2, OUT, I64(h), I64(w),
                 I64(ci), I64(kh), I64(kw), I64(co), s, s, I64(kh_sv), I64(ci_sv), I64(co_sv), act, a0, a1, slope,
                 OFM[ofm], n=out_bytes(ofm, ho * wo * co), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("geom", FCONV[:2])
@pytest.mark.parametrize("r", [*NANV, np.inf, -np.inf, 0.0, -0.0, 1e-40, 1e-8, 7e4, 1.0 / 0.37])
@pytest.mark.parametrize("ofm", ["fp8", "bf16"])
def test_special_conv_fp8_f16_input(geom, r, ofm):
    """f16 activations with NaN / inf / subnormal codes and a special reciprocal scale."""
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    x16 = sp_codes((h, w, ci), SPECIAL_F16, 0.3)
    wts = sp_fp8((kh, kw, ci, co))
    c0, c2 = sp_f32(co), sp_f32(co)
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    ifm = gb.narrow_f16_fp8(x16, np.float32(r))
    want = gb.conv_fp8(ifm, wts, c0, c2, None, None, 1, 1, 1, 0.0, h, w, ci, kh, kw, co, s, s, kh_sv,
                       ci_sv, co_sv, 0, 0.0, 0.0, None, ofm)
    got = call_n("npu_conv_fp8", x16, NPU_F16, float(np.float32(r)), wts, c0, c2, OUT, I64(h), I64(w), I64(ci),
                 I64(kh), I64(kw), I64(co), s, s, I64(kh_sv), I64(ci_sv), I64(co_sv), 0, 0.0, 0.0, None,
                 OFM[ofm], n=out_bytes(ofm, ho * wo * co), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("geom", FCONV[:3])
@pytest.mark.parametrize("it", ["fp8", "bf16", "f16"])
@pytest.mark.parametrize("bshape", ["full", "chan"])
@pytest.mark.parametrize("coeff", [0.7, *NANV, np.inf, -np.inf, 0.0, -0.0, 1e-40, F32_MAX])
@pytest.mark.parametrize("ofm", ["fp8", "f16"])
def test_special_conv_add_fp8(geom, it, bshape, coeff, ofm):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ifm, wts = sp_fp8((h, w, ci)), sp_fp8((kh, kw, ci, co))
    c0, c2 = sp_f32(co), sp_f32(co)
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    shp = {"full": (ho, wo, co), "chan": (1, 1, co)}[bshape]
    ifm2 = {"fp8": lambda: sp_fp8(shp), "bf16": lambda: sp_codes(shp, SPECIAL_BF16),
            "f16": lambda: sp_codes(shp, SPECIAL_F16)}[it]()
    want = gb.conv_fp8(ifm, wts, c0, c2, ifm2, it, *shp, np.float32(coeff), h, w, ci, kh, kw, co, s, s, kh_sv,
                       ci_sv, co_sv, 0, 0.0, 0.0, None, ofm)
    tail = (OFM[it], I64(shp[0]), I64(shp[1]), I64(shp[2]), float(coeff), OUT, I64(h), I64(w), I64(ci),
            I64(kh), I64(kw), I64(co), s, s, I64(kh_sv), I64(ci_sv), I64(co_sv), OFM[ofm])
    same(call_n("npu_conv_add_fp8", ifm, wts, c0, c2, ifm2, *tail, n=out_bytes(ofm, ho * wo * co), dt=np.uint8), want)


@pytest.mark.parametrize("kk,s", [((3, 3), 1), ((3, 3), 2), ((1, 1), 1)])
@pytest.mark.parametrize("act,a0", [(0, 0.0), (1, 0.0)] + [(act, a0) for act in (2, 3) for a0 in COEF_SPECIAL],
                         ids=lambda v: f"{v:g}")
@pytest.mark.parametrize("ofm", ["fp8", "f16", "bf16"])
def test_special_depthwise_fp8(kk, s, act, a0, ofm):
    kh, kw = kk
    c = 64
    h, w = 3 * s + kh, 4 * s + kw
    ifm, wts = sp_fp8((h, w, c)), sp_fp8((kh, kw, c))
    c0, c2 = sp_f32(c), sp_f32(c)
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    want = gb.depthwise_fp8(ifm, wts, c0, c2, h, w, c, kh, kw, s, s, act, a0, ofm)
    got = call_n("npu_depthwise_conv_fp8", ifm, wts, c0, c2, OUT, I64(h), I64(w), I64(c), I64(kh), I64(kw), s, s,
                 act, a0, OFM[ofm], n=out_bytes(ofm, ho * wo * c), dt=np.uint8)
    same(got, want)
