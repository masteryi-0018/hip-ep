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
from npu import I64, NPU_BF16, NPU_BFP16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_MX6, NPU_U8, OUT, raw_rc
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


def wshape(shape):
    """The drawing order of logical weights (..., R, N), R the reduction axis: N rows of R
    values, so that the exponent spread of spread() runs along R."""
    return tuple(shape[:-2]) + (shape[-1], shape[-2])


def to_w(v, shape):
    """Values drawn in wshape(shape) order -> logical weights (..., R, N)."""
    return np.ascontiguousarray(np.asarray(v).reshape(wshape(shape)).swapaxes(-1, -2))


def pack_w(v, fmt):
    """Logical float weights (..., R, N) as a ... x R/blk x N grid of blocks along R."""
    blk = 16 if fmt == "mx6" else 8
    *lead, r, n = v.shape
    g = np.ascontiguousarray(v.reshape(*lead, r // blk, blk, n).swapaxes(-1, -2)).reshape(-1)
    return encode_mx6(g, Rnd.kHalfAway) if fmt == "mx6" else encode_bfp16(g, Rnd.kHalfEven)


def bfp16_w(shape, oct_=8):
    return pack_w(to_w(spread(wshape(shape), oct_), shape), "bfp16")


def mx6_w(shape, oct_=6):
    return pack_w(to_w(spread(wshape(shape), oct_, 16), shape), "mx6")


def fp8_w(shape, oct_=4):
    return to_w(fp8(wshape(shape), oct_), shape).reshape(-1)


def out_bytes(ofm, n):
    return {"bf16": 2 * n, "f16": 2 * n, "fp8": n, "u8": n, "bfp16": n // 8 * 9,
            "mx6": n // 16 * 12}[ofm]


def check(name, *args, want, nbytes, valid=True):
    """Bytes of the entry against the model, or -1 for an argument set the kernel cannot take."""
    if not valid:
        assert rejected(name, *args, nbytes=nbytes), name
        return
    same(call_n(name, *args, n=nbytes, dt=np.uint8), want())


def gemm_bfp16_ok(ofm, transpose, bias, narrow, act):
    """The kernel rules of npu_gemm_bfp16 (activation needs narrow and a non-bf16 output;
    transpose: no bias, no ReLU / ReLU6 / LeakyReLU, no bf16 output)."""
    return ((act == 0 or (narrow and ofm != "bf16")) and
            not (transpose and (bias is not None or act in (1, 2, 3) or ofm == "bf16")) and
            not (ofm == "u8" and (not narrow or act)))


def run_gemm_bfp16(a, b, bias, ofm, m, k, n, transpose, narrow, act, alpha, q=(0.0, 0.0)):
    check("npu_gemm_bfp16", a, b, bias, OUT, OFM[ofm], I64(m), I64(k), I64(n), transpose,
          float(q[0]), float(q[1]), narrow, act, float(alpha), nbytes=out_bytes(ofm, m * n),
          valid=gemm_bfp16_ok(ofm, transpose, bias, narrow, act),
          want=lambda: gb.gemm_bfp16(a, b, bias, ofm, m, k, n, transpose, *q, narrow, act, alpha))


@pytest.mark.parametrize("mkn", [(8, 64, 16), (5, 192, 24), (16, 128, 64)])
@pytest.mark.parametrize("narrow,act", [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3)])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
@pytest.mark.parametrize("transpose", [0, 1])
def test_gemm_bfp16(mkn, narrow, act, ofm, transpose):
    m, k, n = mkn
    if ofm == "bfp16" and (m if transpose else n) % 8:
        pytest.skip("bfp16 output blocks of 8")
    a, b = bfp16((m, k)), bfp16_w((k, n))
    bias = None if transpose else (RNG.standard_normal(n) * 30).astype(np.float32)
    run_gemm_bfp16(a, b, bias, ofm, m, k, n, transpose, narrow, act, 0.15625)


def act_operands(m, k, n):
    """Operands whose GEMM lands in [-8, 8]."""
    a = encode_bfp16((RNG.standard_normal((m, k)) * 0.5).astype(np.float32).reshape(-1), Rnd.kHalfEven)
    b = pack_w(to_w((RNG.standard_normal((n, k)) * 0.5 / np.sqrt(k / 16)).astype(np.float32), (k, n)),
               "bfp16")
    return a, b


@pytest.mark.parametrize("act", [4, 5, 6, 7], ids=["sigmoid", "silu", "gelu", "tanh"])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
@pytest.mark.parametrize("transpose", [0, 1])
def test_gemm_bfp16_core_act(act, ofm, transpose):
    m, k, n = 16, 192, 64
    a, b = act_operands(m, k, n)
    bias = None if transpose else RNG.uniform(-2, 2, n).astype(np.float32)
    run_gemm_bfp16(a, b, bias, ofm, m, k, n, transpose, 1, act, 0.0)


# bf16 inputs whose Sigmoid / SiLU / Tanh lands exactly half-way between two bf16 values
CORE_TIES = {4: [0x3C08, 0x3D8A, 0x3D8B, 0x3DC7, 0x3DC8, 0x3E41, 0x3E80, 0x3E90],
             5: [0xBE00], 7: [0x3CFA, 0x3CFB, 0x3CFC, 0x3CFD, 0x3CFE, 0x3D06, 0x3D07, 0x3D0E]}


@pytest.mark.parametrize("act", [4, 5, 7], ids=["sigmoid", "silu", "tanh"])
def test_gemm_bfp16_core_act_ties(act):
    """A = B = 0, bias set to inputs whose activation is a bf16 tie: half away."""
    m, k = 8, 64
    x = np.array(CORE_TIES[act] * 8, np.uint32)[:64]
    n = x.size
    bias = (x << 16).view(np.float32)
    a = encode_bfp16(np.zeros(m * k, np.float32), Rnd.kHalfEven)
    b = pack_w(np.zeros((k, n), np.float32), "bfp16")
    core = (ga.hw_silu_f32(bias) if act == 5 else
            ga.hw_sigtanh_f32(bias, act == 7, ga.SIGTANH_K0, *((ga.TANH_K1, ga.TANH_K2) if act == 7
                                                               else (ga.SIGMOID_K1, ga.SIGMOID_K2))))
    away, even = (f32_to_bf16(core, r, kFtzIn) for r in (Rnd.kHalfAway, Rnd.kHalfEven))
    assert np.all(away != even)                           # every value is a tie
    run_gemm_bfp16(a, b, bias, "bfp16", m, k, n, 0, 1, act, 0.0)


@pytest.mark.parametrize("act", [0, 5, 6, 7], ids=["bypass", "silu", "gelu", "tanh"])
@pytest.mark.parametrize("bias", [0, 1])
def test_gemm_gelu_mul_bfp16(act, bias):
    m, k, n = 16, 128, 64
    a, bg = act_operands(m, k, n)
    _, bu = act_operands(m, k, n)
    big, biu = ((RNG.uniform(-2, 2, n).astype(np.float32) for _ in range(2)) if bias else (None, None))
    want = gb.gemm_gelu_mul_bfp16(a, bg, bu, big, biu, m, k, n, act)
    got = call_n("npu_gemm_gelu_mul_bfp16", a, bg, bu, big, biu, OUT, I64(m), I64(k), I64(n), act,
                 n=out_bytes("bfp16", m * n), dt=np.uint8)
    same(got, want)


ELTWISE = [  # m, k, n, ifm2, m2, sv_k, is_mul
    (8, 192, 128, "bf16", 8, 192, 0), (8, 192, 64, "bf16", 1, 192, 0),
    (8, 192, 128, "bfp16", 8, 128, 0), (8, 128, 128, "bfp16", 8, 64, 0),
    (6, 256, 64, "bfp16", 1, 128, 0), (70, 128, 128, "bfp16", 70, 128, 0),
    (8, 192, 64, "bf16", 8, 64, 1), (4, 64, 64, "bf16", 1, 64, 1)]


def run_eltwise(a, b, bias_v, ifm2, it, ofm, m, k, n, m2, sv_k, is_mul):
    if is_mul:
        want = gb.gemm_mul_bfp16(a, ifm2, b, bias_v, ofm, m, k, n, m2)
        got = call_n("npu_gemm_mul_bfp16", a, ifm2, b, bias_v, OUT, OFM[ofm], I64(m), I64(k), I64(n),
                     I64(m2), n=out_bytes(ofm, m * n), dt=np.uint8)
    else:
        want = gb.gemm_add_bfp16(a, ifm2, it, b, bias_v, ofm, m, k, n, m2, sv_k)
        got = call_n("npu_gemm_add_bfp16", a, ifm2, OFM[it], b, bias_v, OUT, OFM[ofm], I64(m), I64(k),
                     I64(n), I64(m2), I64(sv_k), n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("case", ELTWISE)
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_gemm_eltwise_bfp16(case, bias, ofm):
    m, k, n, it, m2, sv_k, is_mul = case
    a, b = bfp16((m, k)), bfp16_w((k, n))
    bias_v = (RNG.standard_normal(n) * 30).astype(np.float32) if bias else None
    if it == "bfp16":
        ifm2 = bfp16((m2, n), 12)
    else:
        ifm2 = f32_to_bf16(spread((m2, n), 6) * 100, Rnd.kHalfEven).reshape(-1)
    run_eltwise(a, b, bias_v, ifm2, it, ofm, m, k, n, m2, sv_k, is_mul)


CONV = [  # h, w, ci, kh, kw, co, stride, kh_sv, ci_sv
    (4, 6, 64, 3, 3, 64, 1, 3, 64), (5, 9, 128, 3, 3, 64, 1, 1, 64),
    (4, 9, 128, 1, 3, 128, 2, 1, 128), (6, 6, 192, 2, 2, 64, 1, 1, 64),
    (5, 7, 64, 3, 3, 64, 2, 3, 64)]


def run_conv_block(fmt, ifm, wts, bias, ofm, geom, narrow, act, alpha):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv = geom
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    model = gb.conv_mx6 if fmt == "mx6" else gb.conv_bfp16
    check(f"npu_conv_{fmt}", ifm, wts, bias, OUT, OFM[ofm], I64(h), I64(w), I64(ci), I64(kh),
          I64(kw), I64(co), s, s, I64(kh_sv), I64(ci_sv), narrow, act, float(alpha),
          nbytes=out_bytes(ofm, ho * wo * co), valid=act == 0 or (narrow and ofm != "bf16"),
          want=lambda: model(ifm, wts, bias, ofm, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv, narrow,
                             act, alpha))


@pytest.mark.parametrize("geom", CONV)
@pytest.mark.parametrize("narrow,act", [(0, 0), (1, 1), (1, 3)])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_conv_bfp16(geom, narrow, act, ofm):
    h, w, ci, kh, kw, co = geom[:6]
    ifm, wts = bfp16((h, w, ci)), bfp16_w((kh, kw, ci, co))
    bias = (RNG.standard_normal(co) * 30).astype(np.float32)
    run_conv_block("bfp16", ifm, wts, bias, ofm, geom, narrow, act, 0.15625)


def run_conv_add_bfp16(ifm, ifm2, wts, b, ofm, geom, shp):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv = geom
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    dims = (I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), *map(I64, shp), s, s, I64(kh_sv),
            I64(ci_sv))
    nb = out_bytes(ofm, ho * wo * co)
    want = gb.conv_add_bfp16(ifm, ifm2, wts, b, ofm, h, w, ci, kh, kw, co, *shp, s, s, kh_sv, ci_sv)
    same(call_n("npu_conv_add_bfp16", ifm, ifm2, wts, b, OUT, OFM[ofm], *dims, n=nb, dt=np.uint8),
         want)
    assert rejected("npu_conv_add_bfp16", ifm, None, wts, b, OUT, OFM[ofm], *dims, nbytes=nb)


def conv_add_shape(geom, bshape):
    h, w, ci, kh, kw, co, s = geom[:7]
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    return {"full": (ho, wo, co), "chan": (1, 1, co), "pixel": (ho, wo, 1)}[bshape]


@pytest.mark.parametrize("geom", CONV[:3])
@pytest.mark.parametrize("bshape", ["full", "chan", "pixel"])
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_conv_add_bfp16(geom, bshape, bias, ofm):
    h, w, ci, kh, kw, co = geom[:6]
    ifm, wts = bfp16((h, w, ci)), bfp16_w((kh, kw, ci, co))
    shp = conv_add_shape(geom, bshape)
    ifm2 = f32_to_bf16(spread(shp, 4, shp[-1]) * 50, Rnd.kHalfEven).reshape(-1)
    b = (RNG.standard_normal(co) * 30).astype(np.float32) if bias else None
    run_conv_add_bfp16(ifm, ifm2, wts, b, ofm, geom, shp)


def run_gemm_mx6(a, b, bias, ofm, m, k, n, narrow, act, alpha):
    check("npu_gemm_mx6", a, b, bias, OUT, OFM[ofm], I64(m), I64(k), I64(n), narrow, act,
          float(alpha), nbytes=out_bytes(ofm, m * n), valid=act == 0 or (narrow and ofm != "bf16"),
          want=lambda: gb.gemm_mx6(a, b, bias, ofm, m, k, n, narrow, act, alpha))


@pytest.mark.parametrize("mkn", [(4, 128, 32), (3, 384, 16)])
@pytest.mark.parametrize("narrow,act", [(0, 0), (1, 0), (1, 2), (1, 3)])
@pytest.mark.parametrize("ofm", ["bf16", "mx6"])
def test_gemm_mx6(mkn, narrow, act, ofm):
    m, k, n = mkn
    a, b = mx6((m, k)), mx6_w((k, n))
    bias = (RNG.standard_normal(n) * 30).astype(np.float32)
    run_gemm_mx6(a, b, bias, ofm, m, k, n, narrow, act, 0.15625)


MCONV = [(4, 6, 128, 3, 3, 64, 1, 3, 128), (4, 9, 256, 1, 3, 64, 2, 1, 128),
         (4, 5, 256, 2, 2, 128, 1, 1, 256)]


@pytest.mark.parametrize("geom", MCONV)
@pytest.mark.parametrize("narrow,act", [(0, 0), (1, 1)])
@pytest.mark.parametrize("ofm", ["bf16", "mx6"])
def test_conv_mx6(geom, narrow, act, ofm):
    h, w, ci, kh, kw, co = geom[:6]
    ifm, wts = mx6((h, w, ci)), mx6_w((kh, kw, ci, co))
    bias = (RNG.standard_normal(co) * 30).astype(np.float32)
    run_conv_block("mx6", ifm, wts, bias, ofm, geom, narrow, act, 0.15625)


def coeffs(n):
    return (RNG.standard_normal(n) * 0.3).astype(np.float32), np.exp2(RNG.uniform(-6, 2, n)).astype(np.float32)


def run_gemm_fp8(a, a_type, b, c0, c2, ofm, m, k, n, qm=0.0):
    want = gb.gemm_fp8(a, a_type, b, c0, c2, ofm, m, k, n, qm)
    got = call_n("npu_gemm_fp8", a, OFM[a_type], b, c0, c2, OUT, OFM[ofm], I64(m), I64(k), I64(n),
                 float(qm), n=out_bytes(ofm, m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("mkn", [(8, 64, 16), (5, 100, 24), (16, 448, 64)])
@pytest.mark.parametrize("ofm", ["fp8", "f16", "bf16"])
def test_gemm_fp8(mkn, ofm):
    m, k, n = mkn
    a, b = fp8((m, k)), fp8_w((k, n))
    c0, c2 = coeffs(n)
    run_gemm_fp8(a, "fp8", b, c0, c2, ofm, m, k, n)


@pytest.mark.parametrize("mkn", [(8, 64, 16), (5, 100, 24)])
@pytest.mark.parametrize("ofm", ["fp8", "bf16"])
@pytest.mark.parametrize("qm", [1.0 / 0.37, 3.0, 1e3])
def test_gemm_fp8_f16_input(mkn, ofm, qm):
    """f16 activations narrowed to E4M3 on load: the multiplier rounded to f16, the +-448 clamp
    and the E4M3 rounding all show."""
    m, k, n = mkn
    x16 = f32_to_f16(spread((m, k), 5) * 6, Rnd.kHalfEven).reshape(-1)
    b = fp8_w((k, n))
    c0, c2 = coeffs(n)
    q = gb._quantize_f16_fp8(x16, qm)
    assert np.any(q == 0x7E)   # the clamp
    run_gemm_fp8(x16, "f16", b, c0, c2, ofm, m, k, n, qm)


FCONV = [  # h, w, ci, kh, kw, co, stride, kh_sv, ci_sv, co_sv
    (4, 6, 64, 3, 3, 16, 1, 3, 64, 16), (5, 9, 16, 3, 3, 32, 1, 1, 16, 32),
    (4, 9, 128, 1, 3, 64, 2, 1, 64, 64), (5, 7, 8, 3, 3, 8, 1, 3, 8, 8)]
# the words of one activation (coeff0, coeff1, relu6_max); the others are set to values that
# would show if read
FACT = {0: (5.0, 6.0, 7.0), 1: (5.0, 6.0, 7.0), 2: (5.0, 6.0, 2.5), 3: (0.3, 6.0, 7.0),
        8: (-1.5, 2.25, 7.0), 9: (5.0, 6.0, 7.0)}


def run_conv_fp8(ifm, ifm_type, wts, c0, c2, slope, ofm, geom, r16, act, words):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    want = gb.conv_fp8(ifm, ifm_type, wts, c0, c2, slope, ofm, h, w, ci, kh, kw, co, s, s, kh_sv,
                       ci_sv, co_sv, r16, act, *words)
    got = call_n("npu_conv_fp8", ifm, OFM[ifm_type], wts, c0, c2, slope, OUT, OFM[ofm], I64(h),
                 I64(w), I64(ci), I64(kh), I64(kw), I64(co), s, s, I64(kh_sv), I64(ci_sv),
                 I64(co_sv), r16, act, *(float(v) for v in words),
                 n=out_bytes(ofm, ho * wo * co), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("geom", FCONV)
@pytest.mark.parametrize("act", [0, 1, 2, 3, 8, 9])
@pytest.mark.parametrize("ofm", ["fp8", "f16", "bf16"])
def test_conv_fp8(geom, act, ofm):
    h, w, ci, kh, kw, co = geom[:6]
    ifm, wts = fp8((h, w, ci)), fp8_w((kh, kw, ci, co))
    c0, c2 = coeffs(co)
    slope = (RNG.standard_normal(co) * 0.5).astype(np.float32)
    run_conv_fp8(ifm, "fp8", wts, c0, c2, slope, ofm, geom, 0, act, FACT[act])


def run_conv_add_fp8(ifm, ifm2, it, wts, c0, c2, ofm, geom, shp, coeff):
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv, co_sv = geom
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    dims = (I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), *map(I64, shp), s, s, I64(kh_sv),
            I64(ci_sv), I64(co_sv), float(coeff))
    want = gb.conv_add_fp8(ifm, ifm2, it, wts, c0, c2, ofm, h, w, ci, kh, kw, co, *shp, s, s,
                           kh_sv, ci_sv, co_sv, np.float32(coeff))
    nb = out_bytes(ofm, ho * wo * co)
    same(call_n("npu_conv_add_fp8", ifm, ifm2, OFM[it], wts, c0, c2, OUT, OFM[ofm], *dims, n=nb,
                dt=np.uint8), want)
    assert rejected("npu_conv_add_fp8", ifm, None, OFM[it], wts, c0, c2, OUT, OFM[ofm], *dims,
                    nbytes=nb)


@pytest.mark.parametrize("geom", FCONV[:3])
@pytest.mark.parametrize("it", ["fp8", "bf16", "f16"])
@pytest.mark.parametrize("bshape", ["full", "chan"])
@pytest.mark.parametrize("ofm", ["fp8", "f16"])
def test_conv_add_fp8(geom, it, bshape, ofm):
    h, w, ci, kh, kw, co = geom[:6]
    ifm, wts = fp8((h, w, ci)), fp8_w((kh, kw, ci, co))
    c0, c2 = coeffs(co)
    shp = conv_add_shape(geom, bshape)
    v = spread(shp, 3, shp[-1])
    ifm2 = {"fp8": lambda: fp8(shp), "bf16": lambda: f32_to_bf16(v, Rnd.kHalfEven).reshape(-1),
            "f16": lambda: f32_to_f16(v, Rnd.kHalfEven).reshape(-1)}[it]()
    run_conv_add_fp8(ifm, ifm2, it, wts, c0, c2, ofm, geom, shp, 0.7)


def run_depthwise_fp8(ifm, wts, c0, c2, ofm, kh, kw, s, act, alpha, relu6_max):
    c = ifm.size // ((3 * s + kh) * (4 * s + kw))
    h, w = 3 * s + kh, 4 * s + kw
    ho, wo = (h - kh) // s + 1, (w - kw) // s + 1
    want = gb.depthwise_conv_fp8(ifm, wts, c0, c2, ofm, h, w, c, kh, kw, s, s, act, alpha,
                                 relu6_max)
    got = call_n("npu_depthwise_conv_fp8", ifm, wts, c0, c2, OUT, OFM[ofm], I64(h), I64(w), I64(c),
                 I64(kh), I64(kw), s, s, act, float(alpha), float(relu6_max),
                 n=out_bytes(ofm, ho * wo * c), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("kk,s", [((3, 3), 1), ((3, 3), 2), ((7, 7), 1), ((1, 1), 1)])
@pytest.mark.parametrize("act", [0, 1, 2, 3])
@pytest.mark.parametrize("ofm", ["fp8", "f16", "bf16"])
def test_depthwise_fp8(kk, s, act, ofm):
    kh, kw = kk
    c = 64
    h, w = 3 * s + kh, 4 * s + kw
    ifm, wts = fp8((h, w, c)), fp8((kh, kw, c))
    c0, c2 = coeffs(c)
    alpha, relu6_max = (0.3, 7.0) if act == 3 else (5.0, 2.5)
    run_depthwise_fp8(ifm, wts, c0, c2, ofm, kh, kw, s, act, alpha, relu6_max)


@pytest.mark.parametrize("sv_k", [128, 64])
def test_gemm_add_bfp16_e_block_order(sv_k):
    """ifm2 ~ -(A @ B), so the output depends on where the ifm2 MMAC enters."""
    m, k, n = 70, 256, 128
    a, b = bfp16((m, k), 12), bfp16_w((k, n), 12)
    bv = decode_bfp16(b).reshape(k // 8, n, 8).transpose(0, 2, 1).reshape(k, n)  # the grid as K x N
    ab = decode_bfp16(a).reshape(m, k).astype(np.float64) @ bv.astype(np.float64)
    ifm2 = encode_bfp16((-ab * (1 + RNG.uniform(-1e-3, 1e-3, ab.shape))).astype(np.float32).reshape(-1), Rnd.kHalfEven)
    want = gb.gemm_add_bfp16(a, ifm2, "bfp16", b, None, "bf16", m, k, n, m, sv_k)
    for other in ("early", "late"):  # the data must tell the orders apart
        alt = gb._gemm_eltwise_bfp16(a, b, None, ifm2, True, m, m, k, n,
                                     64 if other == "early" else 10 ** 9, 0, "bf16",
                                     sv_mn=0 if other == "late" else 64)
        if not (other == "early" and sv_k == 64):
            assert np.any(alt != want), other
    got = call_n("npu_gemm_add_bfp16", a, ifm2, NPU_BFP16, b, None, OUT, NPU_BF16, I64(m), I64(k),
                 I64(n), I64(m), I64(sv_k), n=out_bytes("bf16", m * n), dt=np.uint8)
    same(got, want)


TIE_ALPHA = float(np.uint32(0x3E208000).view(np.float32))   # halfway between two bf16
AWAY_ALPHA = float(np.uint32(0x3E210000).view(np.float32))  # the tie rounded away


@pytest.mark.parametrize("fmt", ["bfp16", "mx6"])
def test_gemm_lrelu_alpha_tie(fmt):
    """The alpha is cast to bf16 half to even."""
    m, k, n = (8, 64, 16) if fmt == "bfp16" else (4, 128, 32)
    pack, pack_b = (bfp16, bfp16_w) if fmt == "bfp16" else (mx6, mx6_w)
    a, b = pack((m, k)), pack_b((k, n))
    bias = (RNG.standard_normal(n) * 30).astype(np.float32)

    def gold(alpha):
        if fmt == "bfp16":
            return gb.gemm_bfp16(a, b, bias, fmt, m, k, n, 0, 0.0, 0.0, 1, 3, alpha)
        return gb.gemm_mx6(a, b, bias, fmt, m, k, n, 1, 3, alpha)

    assert (gold(TIE_ALPHA) != gold(AWAY_ALPHA)).any()
    if fmt == "bfp16":
        run_gemm_bfp16(a, b, bias, fmt, m, k, n, 0, 1, 3, TIE_ALPHA)
    else:
        run_gemm_mx6(a, b, bias, fmt, m, k, n, 1, 3, TIE_ALPHA)


@pytest.mark.parametrize("fmt", ["bfp16", "mx6"])
def test_conv_lrelu_alpha_tie(fmt):
    geom = CONV[0] if fmt == "bfp16" else MCONV[0]
    h, w, ci, kh, kw, co, s, kh_sv, ci_sv = geom
    pack, pack_wts = (bfp16, bfp16_w) if fmt == "bfp16" else (mx6, mx6_w)
    ifm, wts = pack((h, w, ci)), pack_wts((kh, kw, ci, co))
    bias = (RNG.standard_normal(co) * 30).astype(np.float32)
    model = gb.conv_mx6 if fmt == "mx6" else gb.conv_bfp16

    def gold(alpha):
        return model(ifm, wts, bias, fmt, h, w, ci, kh, kw, co, s, s, kh_sv, ci_sv, 1, 3, alpha)

    if fmt == "bfp16":  # the 4-bit mx6 mantissas hide the last bf16 bit of the alpha
        assert (gold(TIE_ALPHA) != gold(AWAY_ALPHA)).any()
    run_conv_block(fmt, ifm, wts, bias, fmt, geom, 1, 3, TIE_ALPHA)


@pytest.mark.parametrize("mkn", [(8, 64, 16), (5, 192, 24)])
@pytest.mark.parametrize("transpose", [0, 1])
def test_gemm_bfp16_u8(mkn, transpose):
    """The uint8 output cast: both saturation bounds and rounding ties are reached."""
    m, k, n = mkn
    a, b = bfp16((m, k)), bfp16_w((k, n))
    bias = None if transpose else (RNG.standard_normal(n) * 30).astype(np.float32)
    q = (float(np.float32(1.0) / np.float32(19.2)), 117.0)
    want = gb.gemm_bfp16(a, b, bias, "u8", m, k, n, transpose, *q, 1, 0, 0.0)
    assert want.min() == 0 and want.max() == 255
    run_gemm_bfp16(a, b, bias, "u8", m, k, n, transpose, 1, 0, 0.0, q)
    run_gemm_bfp16(a, b, bias, "u8", m, k, n, transpose, 0, 0, 0.0, q)  # rejected: needs narrow


@pytest.mark.parametrize("geom", FCONV)
@pytest.mark.parametrize("ofm", ["fp8", "bf16"])
def test_conv_fp8_f16_input(geom, ofm):
    """f16 activations narrowed to E4M3 on load: the f16 product, the +-448 clamp in f16 and the
    E4M3 rounding all show."""
    h, w, ci, kh, kw, co = geom[:6]
    x16 = f32_to_f16(spread((h, w, ci), 5) * 6, Rnd.kHalfEven).reshape(-1)
    r16 = int(np.float16(1.0 / 0.37).view(np.uint16))
    wts = fp8_w((kh, kw, ci, co))
    c0, c2 = coeffs(co)
    run_conv_fp8(x16, "f16", wts, c0, c2, None, ofm, geom, r16, 0, FACT[0])


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


def sp_bfp16_w(shape):
    return pack_w(to_w(sp_blocks(wshape(shape), 8), shape), "bfp16")


def sp_mx6_w(shape):
    return pack_w(to_w(sp_blocks(wshape(shape), 16), shape), "mx6")


def sp_fp8(shape, rate=0.12):
    c = fp8(shape).reshape(-1)
    pos = RNG.random(c.size) < rate
    c[pos] = SPECIAL_FP8[RNG.integers(0, SPECIAL_FP8.size, int(pos.sum()))]
    return c


def sp_fp8_w(shape):
    return to_w(sp_fp8(wshape(shape)), shape).reshape(-1)


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
    a, b = sp_bfp16((m, k)), sp_bfp16_w((k, n))
    bias = None if transpose else sp_f32(n)
    run_gemm_bfp16(a, b, bias, ofm, m, k, n, transpose, narrow, act, alpha)


@pytest.mark.parametrize("q", [(NANV[0], 117.0), (NANV[2], NANV[3]), (np.inf, 0.0), (-np.inf, 5.0), (0.0, np.inf), (0.0, -np.inf),
                               (1.0, NANV[2]), (-0.0, -0.0), (1e-40, 1e-40), (F32_MAX, F32_MAX),
                               (1.0, 0.0), (0.05, 117.0)], ids=lambda v: "-".join(f"{x:g}" for x in v))
@pytest.mark.parametrize("transpose", [0, 1])
def test_special_gemm_bfp16_u8(q, transpose):
    m, k, n = 8, 128, 16
    a, b = sp_bfp16((m, k)), sp_bfp16_w((k, n))
    bias = None if transpose else sp_f32(n)
    run_gemm_bfp16(a, b, bias, "u8", m, k, n, transpose, 1, 0, 0.0, q)


@pytest.mark.parametrize("act", [0, 5, 6, 7], ids=["bypass", "silu", "gelu", "tanh"])
@pytest.mark.parametrize("bias", [0, 1])
def test_special_gemm_gelu_mul_bfp16(act, bias):
    m, k, n = 16, 128, 64
    a, bg, bu = sp_bfp16((m, k)), sp_bfp16_w((k, n)), sp_bfp16_w((k, n))
    big, biu = (sp_f32(n), sp_f32(n)) if bias else (None, None)
    want = gb.gemm_gelu_mul_bfp16(a, bg, bu, big, biu, m, k, n, act)
    got = call_n("npu_gemm_gelu_mul_bfp16", a, bg, bu, big, biu, OUT, I64(m), I64(k), I64(n), act,
                 n=out_bytes("bfp16", m * n), dt=np.uint8)
    same(got, want)


@pytest.mark.parametrize("case", ELTWISE)
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_special_gemm_eltwise_bfp16(case, bias, ofm):
    m, k, n, it, m2, sv_k, is_mul = case
    a, b = sp_bfp16((m, k)), sp_bfp16_w((k, n))
    bias_v = sp_f32(n) if bias else None
    ifm2 = sp_bfp16((m2, n)) if it == "bfp16" else sp_codes((m2, n), SPECIAL_BF16)
    run_eltwise(a, b, bias_v, ifm2, it, ofm, m, k, n, m2, sv_k, is_mul)


@pytest.mark.parametrize("geom", CONV)
@pytest.mark.parametrize("narrow,act,alpha", [(0, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0)] +
                         [(1, 3, al) for al in COEF_SPECIAL], ids=lambda v: f"{v:g}")
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_special_conv_bfp16(geom, narrow, act, alpha, ofm):
    h, w, ci, kh, kw, co = geom[:6]
    ifm, wts = sp_bfp16((h, w, ci)), sp_bfp16_w((kh, kw, ci, co))
    run_conv_block("bfp16", ifm, wts, sp_f32(co), ofm, geom, narrow, act, alpha)


@pytest.mark.parametrize("geom", CONV[:3])
@pytest.mark.parametrize("bshape", ["full", "chan", "pixel"])
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("ofm", ["bf16", "bfp16"])
def test_special_conv_add_bfp16(geom, bshape, bias, ofm):
    h, w, ci, kh, kw, co = geom[:6]
    ifm, wts = sp_bfp16((h, w, ci)), sp_bfp16_w((kh, kw, ci, co))
    shp = conv_add_shape(geom, bshape)
    ifm2 = sp_codes(shp, SPECIAL_BF16)
    b = sp_f32(co) if bias else None
    run_conv_add_bfp16(ifm, ifm2, wts, b, ofm, geom, shp)


@pytest.mark.parametrize("mkn", [(4, 128, 32), (3, 384, 16)])
@pytest.mark.parametrize("narrow,act,alpha", [(0, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0)] +
                         [(1, 3, al) for al in COEF_SPECIAL], ids=lambda v: f"{v:g}")
@pytest.mark.parametrize("ofm", ["bf16", "mx6"])
def test_special_gemm_mx6(mkn, narrow, act, alpha, ofm):
    m, k, n = mkn
    a, b = sp_mx6((m, k)), sp_mx6_w((k, n))
    run_gemm_mx6(a, b, sp_f32(n), ofm, m, k, n, narrow, act, alpha)


@pytest.mark.parametrize("geom", MCONV[:2])
@pytest.mark.parametrize("narrow,act", [(0, 0), (1, 1), (1, 3)])
@pytest.mark.parametrize("ofm", ["bf16", "mx6"])
def test_special_conv_mx6(geom, narrow, act, ofm):
    h, w, ci, kh, kw, co = geom[:6]
    ifm, wts = sp_mx6((h, w, ci)), sp_mx6_w((kh, kw, ci, co))
    alpha = float("inf") if act == 3 else 0.0
    run_conv_block("mx6", ifm, wts, sp_f32(co), ofm, geom, narrow, act, alpha)


@pytest.mark.parametrize("mkn", [(8, 64, 16), (5, 100, 24)])
@pytest.mark.parametrize("ofm", ["fp8", "f16", "bf16"])
@pytest.mark.parametrize("rep", range(2))
def test_special_gemm_fp8(mkn, ofm, rep):
    m, k, n = mkn
    a, b = sp_fp8((m, k)), sp_fp8_w((k, n))
    run_gemm_fp8(a, "fp8", b, sp_f32(n), sp_f32(n), ofm, m, k, n)


@pytest.mark.parametrize("mkn", [(8, 64, 16), (5, 100, 24)])
@pytest.mark.parametrize("qm", [*NANV, np.inf, -np.inf, 0.0, -0.0, 1e-40, 1e-8, 7e4, 1.0 / 0.37])
@pytest.mark.parametrize("ofm", ["fp8", "bf16"])
def test_special_gemm_fp8_f16_input(mkn, qm, ofm):
    """f16 activations with NaN / inf / subnormal codes and a special multiplier."""
    m, k, n = mkn
    x16 = sp_codes((m, k), SPECIAL_F16, 0.3)
    b = sp_fp8_w((k, n))
    run_gemm_fp8(x16, "f16", b, sp_f32(n), sp_f32(n), ofm, m, k, n, float(np.float32(qm)))


ACT_PARAMS = [(0, 0.0, 0.0), (1, 0.0, 0.0), (2, 2.5, 0.0), (3, 0.3, 0.0), (8, -1.5, 2.25), (9, 0.0, 0.0)] + \
    [(act, a0, a1) for act in (2, 3, 8) for a0 in COEF_SPECIAL[:5] for a1 in (NANV[2], 1.0)]


def fp8_words(act, a0, a1):
    """(coeff0, coeff1, relu6_max) with a0 / a1 in the words of the activation and the other
    words at values that would show if read."""
    if act == 2:
        return (a1, 7.0, a0)
    return (a0, a1, 7.0)


@pytest.mark.parametrize("geom", FCONV[:2])
@pytest.mark.parametrize("act,a0,a1", ACT_PARAMS, ids=lambda v: f"{v:g}")
@pytest.mark.parametrize("ofm", ["fp8", "f16", "bf16"])
def test_special_conv_fp8(geom, act, a0, a1, ofm):
    h, w, ci, kh, kw, co = geom[:6]
    ifm, wts = sp_fp8((h, w, ci)), sp_fp8_w((kh, kw, ci, co))
    c0, c2 = sp_f32(co), sp_f32(co)
    slope = sp_f32(co)
    run_conv_fp8(ifm, "fp8", wts, c0, c2, slope, ofm, geom, 0, act, fp8_words(act, a0, a1))


@pytest.mark.parametrize("geom", FCONV[:2])
@pytest.mark.parametrize("r", [*NANV, np.inf, -np.inf, 0.0, -0.0, 1e-40, 1e-8, 7e4, 1.0 / 0.37])
@pytest.mark.parametrize("ofm", ["fp8", "bf16"])
def test_special_conv_fp8_f16_input(geom, r, ofm):
    """f16 activations with NaN / inf / subnormal codes and a special reciprocal scale."""
    h, w, ci, kh, kw, co = geom[:6]
    x16 = sp_codes((h, w, ci), SPECIAL_F16, 0.3)
    wts = sp_fp8_w((kh, kw, ci, co))
    c0, c2 = sp_f32(co), sp_f32(co)
    with np.errstate(all="ignore"):
        r16 = int(f32_to_f16(np.float32(r), Rnd.kHalfEven))
    run_conv_fp8(x16, "f16", wts, c0, c2, None, ofm, geom, r16, 0, FACT[0])


@pytest.mark.parametrize("geom", FCONV[:3])
@pytest.mark.parametrize("it", ["fp8", "bf16", "f16"])
@pytest.mark.parametrize("bshape", ["full", "chan"])
@pytest.mark.parametrize("coeff", [0.7, *NANV, np.inf, -np.inf, 0.0, -0.0, 1e-40, F32_MAX])
@pytest.mark.parametrize("ofm", ["fp8", "f16"])
def test_special_conv_add_fp8(geom, it, bshape, coeff, ofm):
    h, w, ci, kh, kw, co = geom[:6]
    ifm, wts = sp_fp8((h, w, ci)), sp_fp8_w((kh, kw, ci, co))
    c0, c2 = sp_f32(co), sp_f32(co)
    shp = conv_add_shape(geom, bshape)
    ifm2 = {"fp8": lambda: sp_fp8(shp), "bf16": lambda: sp_codes(shp, SPECIAL_BF16),
            "f16": lambda: sp_codes(shp, SPECIAL_F16)}[it]()
    run_conv_add_fp8(ifm, ifm2, it, wts, c0, c2, ofm, geom, shp, coeff)


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
    alpha, relu6_max = (a0, 7.0) if act == 3 else (5.0, a0)
    run_depthwise_fp8(ifm, wts, c0, c2, ofm, kh, kw, s, act, alpha, relu6_max)


def test_bad_args():
    """Every entry returns -1 for arguments its kernel cannot take, before touching memory."""
    p = np.zeros(1, np.uint8)   # placeholder: these calls must return before any access
    I = I64  # noqa: E741  (a short alias keeps the argument tables compact)

    def gemm(ofm=NPU_BFP16, k=64, tr=0, bias=None, narrow=1, act=0):
        return ("npu_gemm_bfp16", p, p, bias, p, ofm, I(8), I(k), I(16), tr, 0.0, 0.0, narrow, act,
                0.0)
    for kw in (dict(k=32), dict(k=96), dict(narrow=0, act=1), dict(act=8), dict(ofm=NPU_BF16, act=1),
               dict(ofm=NPU_F32), dict(tr=1, bias=p), dict(tr=1, act=3), dict(tr=1, ofm=NPU_BF16),
               dict(ofm=NPU_U8, narrow=0), dict(ofm=NPU_U8, act=5)):
        assert raw_rc(*gemm(**kw)) == -1, kw
    assert raw_rc("npu_gemm_gelu_mul_bfp16", p, p, p, None, None, p, I(8), I(64), I(16), 3) == -1
    assert raw_rc("npu_gemm_gelu_mul_bfp16", p, None, p, None, None, p, I(8), I(64), I(16), 0) == -1
    for it, k, n, m2, sv_k, ofm in ((NPU_F16, 64, 64, 8, 64, NPU_BF16), (NPU_BFP16, 64, 32, 8, 64, NPU_BF16),
                                    (NPU_BF16, 64, 64, 2, 64, NPU_BF16), (NPU_BF16, 64, 64, 8, 32, NPU_BF16),
                                    (NPU_BF16, 64, 64, 8, 4160, NPU_BF16), (NPU_BF16, 64, 64, 8, 64, NPU_MX6)):
        assert raw_rc("npu_gemm_add_bfp16", p, p, it, p, None, p, ofm, I(8), I(k), I(n), I(m2),
                      I(sv_k)) == -1, (it, n, m2, sv_k, ofm)
    assert raw_rc("npu_gemm_mul_bfp16", p, p, p, None, p, NPU_BF16, I(8), I(64), I(64), I(3)) == -1
    assert raw_rc("npu_gemm_mul_bfp16", p, None, p, None, p, NPU_BF16, I(8), I(64), I(64), I(1)) == -1

    def conv(name, ci=64, co=64, kh=3, kw=3, s=1, kh_sv=1, ci_sv=64, ofm=NPU_BFP16, act=0):
        return (name, p, p, None, p, ofm, I(6), I(6), I(ci), I(kh), I(kw), I(co), s, s, I(kh_sv),
                I(ci_sv), 1, act, 0.0)
    for kw in (dict(ci_sv=32, ci=64), dict(co=32), dict(s=3), dict(s=0), dict(kh_sv=2),
               dict(kw=16, kh=1), dict(ofm=NPU_BF16, act=1), dict(act=4), dict(ofm=NPU_MX6)):
        assert raw_rc(*conv("npu_conv_bfp16", **kw)) == -1, kw
    for kw in (dict(ci=128, ci_sv=64), dict(ci=512, ci_sv=512), dict(ci=128, ci_sv=128, co=32),
               dict(ci=128, ci_sv=128, ofm=NPU_BFP16)):
        assert raw_rc(*conv("npu_conv_mx6", **{"ofm": NPU_MX6, **kw})) == -1, kw
    def add(ifm2=p, c2=64, s=1):
        return ("npu_conv_add_bfp16", p, ifm2, p, None, p, NPU_BFP16, I(6), I(6),
                I(64), I(3), I(3), I(64), I(1), I(1), I(c2), s, s, I(1), I(64))
    for kw in (dict(ifm2=None), dict(c2=2), dict(s=4)):
        assert raw_rc(*add(**kw)) == -1, kw
    assert raw_rc("npu_gemm_mx6", p, p, None, p, NPU_MX6, I(4), I(64), I(16), 1, 0, 0.0) == -1
    assert raw_rc("npu_gemm_mx6", p, p, None, p, NPU_BF16, I(4), I(128), I(16), 1, 2, 0.0) == -1
    for at, ofm, k in ((NPU_BF16, NPU_FP8, 64), (NPU_FP8, NPU_BFP16, 64), (NPU_FP8, NPU_FP8, 0)):
        assert raw_rc("npu_gemm_fp8", p, at, p, p, p, p, ofm, I(4), I(k), I(16), 1.0) == -1, (at, ofm)
    assert raw_rc("npu_gemm_fp8", p, NPU_FP8, p, None, p, p, NPU_FP8, I(4), I(64), I(16), 1.0) == -1

    def fconv(it=NPU_FP8, ci_sv=8, co_sv=8, kh=3, kw=3, kh_sv=1, s=1, act=0, slope=None, ofm=NPU_FP8):
        return ("npu_conv_fp8", p, it, p, p, p, slope, p, ofm, I(8), I(8), I(16), I(kh), I(kw), I(16),
                s, s, I(kh_sv), I(ci_sv), I(co_sv), 0, act, 0.0, 0.0, 0.0)
    for kw in (dict(it=NPU_BF16), dict(ci_sv=12), dict(ci_sv=32), dict(co_sv=4), dict(kw=8, kh=1),
               dict(kh=8, kh_sv=8), dict(kh_sv=2), dict(s=3), dict(act=4), dict(act=9),
               dict(ofm=NPU_BFP16)):
        assert raw_rc(*fconv(**kw)) == -1, kw
    def fadd(it=NPU_FP8, c2=16, ifm2=p):
        return ("npu_conv_add_fp8", p, ifm2, it, p, p, p, p, NPU_FP8,
                I(8), I(8), I(16), I(3), I(3), I(16), I(1), I(1),
                I(c2), 1, 1, I(1), I(8), I(8), 1.0)
    for kw in (dict(ifm2=None), dict(it=NPU_I8), dict(c2=2)):
        assert raw_rc(*fadd(**kw)) == -1, kw
    def dwc(c=64, kh=3, s=1, act=0, ofm=NPU_FP8):
        return ("npu_depthwise_conv_fp8", p, p, p, p, p, ofm,
                I(20), I(20), I(c), I(kh), I(3), s, s, act,
                0.0, 0.0)
    for kw in (dict(c=32), dict(kh=16), dict(s=3), dict(act=4), dict(ofm=NPU_U8)):
        assert raw_rc(*dwc(**kw)) == -1, kw

    # negative sizes, null pointers (also with an empty output) and empty outputs
    for m, n, a, rc in ((-1, 16, p, -1), (8, -16, p, -1), (0, 16, None, -1), (0, 16, p, 0),
                        (8, 0, p, 0)):
        assert raw_rc("npu_gemm_bfp16", a, p, None, p, NPU_BFP16, I(m), I(64), I(n), 0, 0.0, 0.0,
                      1, 0, 0.0) == rc, (m, n, a)
        assert raw_rc("npu_gemm_gelu_mul_bfp16", a, p, p, None, None, p, I(m), I(64), I(n),
                      0) == rc, (m, n, a)
        assert raw_rc("npu_gemm_add_bfp16", a, p, NPU_BF16, p, None, p, NPU_BF16, I(m), I(64),
                      I(n), I(1), I(64)) == rc, (m, n, a)
        assert raw_rc("npu_gemm_mul_bfp16", a, p, p, None, p, NPU_BF16, I(m), I(64), I(n),
                      I(1)) == rc, (m, n, a)
        assert raw_rc("npu_gemm_mx6", a, p, None, p, NPU_MX6, I(m), I(128), I(n), 1, 0,
                      0.0) == rc, (m, n, a)
        assert raw_rc("npu_gemm_fp8", a, NPU_FP8, p, p, p, p, NPU_FP8, I(m), I(64), I(n),
                      1.0) == rc, (m, n, a)
    assert raw_rc("npu_gemm_fp8", p, NPU_FP8, p, p, None, p, NPU_FP8, I(0), I(64), I(16), 1.0) == -1
    for h, w, x, rc in ((-1, 6, p, -1), (6, -1, p, -1), (0, 6, None, -1), (0, 6, p, 0),
                        (6, 0, p, 0)):
        def conv_hw(name, ofm, ci_sv):
            return (name, x, p, None, p, ofm, I(h), I(w), I(ci_sv), I(3),
                    I(3), I(64), 1, 1, I(1), I(ci_sv), 1, 0, 0.0)
        assert raw_rc(*conv_hw("npu_conv_bfp16", NPU_BFP16, 64)) == rc, (h, w, x)
        assert raw_rc(*conv_hw("npu_conv_mx6", NPU_MX6, 128)) == rc, (h, w, x)
        assert raw_rc("npu_conv_add_bfp16", x, p, p, None, p, NPU_BFP16, I(h), I(w), I(64), I(3),
                      I(3), I(64), I(1), I(1), I(64), 1, 1, I(1), I(64)) == rc, (h, w, x)
        assert raw_rc("npu_conv_fp8", x, NPU_FP8, p, p, p, None, p, NPU_FP8, I(h), I(w), I(16),
                      I(3), I(3), I(16), 1, 1, I(1), I(8), I(8), 0, 0, 0.0, 0.0, 0.0) == rc, (h, w)
        assert raw_rc("npu_conv_add_fp8", x, p, NPU_FP8, p, p, p, p, NPU_FP8, I(h), I(w), I(16),
                      I(3), I(3), I(16), I(1), I(1), I(16), 1, 1, I(1), I(8), I(8),
                      1.0) == rc, (h, w)
        assert raw_rc("npu_depthwise_conv_fp8", x, p, p, p, p, NPU_FP8, I(h), I(w), I(64), I(3),
                      I(3), 1, 1, 0, 0.0, 0.0) == rc, (h, w)
    # channel counts stay >= 1 (a multiple of the block size), also with an empty image
    assert raw_rc(*conv("npu_conv_bfp16", ci=0)) == -1
    assert raw_rc(*conv("npu_conv_bfp16", co=0)) == -1
    assert raw_rc("npu_conv_add_bfp16", p, None, p, None, p, NPU_BFP16, I(0), I(6), I(64), I(3),
                  I(3), I(64), I(1), I(1), I(64), 1, 1, I(1), I(64)) == -1
