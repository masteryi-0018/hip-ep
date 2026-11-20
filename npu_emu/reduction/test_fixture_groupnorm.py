# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""reduction/groupnorm.h entry points and golden_groupnorm.py against NPU reference outputs."""

import numpy as np
import pytest

import golden_groupnorm as gg
from fixture import cases, check, data, ofm
from fixture import testcase as tc
from npu import (I64, NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, NPU_U8, NPU_U16,
                 OUT, OUT2)
from testing import call_n

REF_MAX = 1 << 21
FL = {"bfloat16": ("bf16", NPU_BF16, np.uint16), "float16": ("f16", NPU_F16, np.uint16),
      "float8": ("fp8", NPU_FP8, np.uint8), "float": ("f32", NPU_F32, np.float32),
      "float32": ("f32", NPU_F32, np.float32)}
INT = {"int8": ("i8", NPU_I8, np.int8), "uint8": ("u8", NPU_U8, np.uint8),
       "int16": ("i16", NPU_I16, np.int16), "uint16": ("u16", NPU_U16, np.uint16)}
TY = {**FL, **INT}


def f(x):
    return float(np.float32(x))


def to_hw_c(raw, hw, c):
    """(c/32, hw, 32) -> (hw, c)."""
    return np.ascontiguousarray(raw[:hw * c].reshape(c // 32, hw, 32).transpose(1, 0, 2)
                                ).reshape(hw, c)


def decode(bits, name):
    return gg.decode(bits, FL[name][0])


IN_SPLIT = {-1: 1, 0: 12, 1: 4, 2: 2, 3: 3}
IN_BF16 = ["instanceNormSd_bf16_wrapper", "instanceNormSdOptStage2Ifm_bf16_wrapper",
           "instanceNormAddMulAddSd_bf16_wrapper",
           "instanceNormAddMulAddSdOptStage2Ifm_bf16_wrapper"]
IN_QDQ = ["qdq_instanceNormSd_wrapper", "qdq_instanceNormSdOptStage2Ifm_wrapper",
          "qdq_instanceNormSdOptStage2IfmDoubleOut_wrapper",
          "qdq_instanceNormAddMulAddSd_wrapper", "qdq_instanceNormAddMulAddSdOptStage2Ifm_wrapper"]


def in_dims(t):
    hw = (t["ifm_sv_height"] * t["ifm_height_iteration"] * t["ifm_sv_width"] *
          t["ifm_width_iteration"] * IN_SPLIT[t["split_mode"]])
    return hw, t["ifm_sv_height"] * t["ifm_sv_width"]


class Wts:
    """Reads the packed weight stream in order."""

    def __init__(self, raw):
        self.raw, self.pos = raw, 0

    def take(self, n, dt):
        nb = n * np.dtype(dt).itemsize
        v = self.raw[self.pos:self.pos + nb].view(dt)
        self.pos += nb
        return v

    def alpha_beta(self, c, dt):
        """alpha / beta interleaved in runs of 32."""
        v = self.take(2 * c, dt).reshape(c // 32, 2, 32)
        return np.ascontiguousarray(v[:, 0]).reshape(c), np.ascontiguousarray(v[:, 1]).reshape(c)

    def block(self):
        """One coefficient broadcast over 64 floats."""
        v = self.take(64, np.float32)
        assert np.all(v == v[0])
        return float(v[0])


@pytest.mark.parametrize("d,case", cases(IN_BF16))
def test_in_bf16(d, case):
    t = tc(case)
    io, io_code, io_dt = FL[t["ifm_dtype"]]
    ab, ab_code, ab_dt = FL[t["alpha_dtype"]]
    hw, sv_hw = in_dims(t)
    raw = data(case, "ifm32.txt", io_dt)
    c = raw.size // hw
    raw = raw[:hw * c]
    plain = d != "instanceNormSd_bf16_wrapper"
    x = raw.reshape(hw, c) if plain else to_hw_c(raw, hw, c)
    w = Wts(data(case, "wts32.txt"))
    alpha, beta = w.alpha_beta(c, ab_dt)
    amad = "AddMulAdd" in d
    args = [x, io_code, alpha, beta, ab_code]
    amad_ops = None
    if amad:
        cst = w.take(c, ab_dt)
        ifm2 = data(case, "ifm2_32.txt", io_dt)
        a1, a2 = ifm2[:c], ifm2[c:2 * c]
        args += [a1, a2, cst]
        amad_ops = (gg.decode(a1, io), gg.decode(cst, ab), gg.decode(a2, io))
    slope = float(w.take(1, np.float32)[0])
    act, pad, split = t["activation_type"], t["ifm_spatial_pad"], t["split_mode"]
    name = {"instanceNormSd_bf16_wrapper": "npu_instance_norm_sd_bf16",
            "instanceNormSdOptStage2Ifm_bf16_wrapper": "npu_instance_norm_sd_opt_stage2_ifm_bf16",
            "instanceNormAddMulAddSd_bf16_wrapper": "npu_instance_norm_add_mul_add_sd_bf16",
            "instanceNormAddMulAddSdOptStage2Ifm_bf16_wrapper":
                "npu_instance_norm_add_mul_add_sd_opt_stage2_ifm_bf16"}[d]
    got = call_n(name, *args, OUT, I64(hw), I64(c), split, I64(sv_hw), I64(pad), act, slope,
                 n=hw * c, dt=io_dt)
    wraw = data(case, ofm(case), io_dt)[:hw * c]
    want = wraw.reshape(hw, c) if plain else to_hw_c(wraw, hw, c)
    ref = None
    if hw * c <= REF_MAX:
        ref = gg.in_bf16(x, io, gg.decode(alpha, ab), gg.decode(beta, ab), split, sv_hw, pad,
                         act, slope, amad_ops)
    check(got.reshape(hw, c), want, ref)


@pytest.mark.parametrize("d,case", cases(IN_QDQ))
def test_in_qdq(d, case):
    t = tc(case)
    io, io_code, io_dt = INT[t["ifm_dtype"]]
    ct, ct_code, ct_dt = FL[t["compute_dtype"]]
    ab, ab_code, ab_dt = FL[t["alpha_dtype"]]
    hw, sv_hw = in_dims(t)
    raw = data(case, "ifm32.txt", io_dt)
    c = raw.size // hw
    x = raw[:hw * c].reshape(hw, c)
    w = Wts(data(case, "wts32.txt"))
    alpha, beta = w.alpha_beta(c, ab_dt)
    dq_zp, dq_sc, q_zp, q_sc = (w.block() for _ in range(4))
    double = "DoubleOut" in d
    amad = "AddMulAdd" in d
    q2 = (w.block(), w.block())[::-1] if double else None  # (q_sc2, q_zp2)
    act, pad, split = t["activation_type"], t["ifm_spatial_pad"], t["split_mode"]
    common = [x, io_code, alpha, beta, ab_code]
    small = hw * c <= REF_MAX
    al, be = gg.decode(alpha, ab), gg.decode(beta, ab)
    if amad:
        dq2_zp, dq2_sc = w.block(), w.block()
        cst = w.take(c, ct_dt)
        slope = float(w.take(1, np.float32)[0])
        ifm2 = data(case, "ifm2_32.txt", io_dt)
        a1, a2 = ifm2[:c], ifm2[c:2 * c]
        name = ("npu_qdq_instance_norm_add_mul_add_sd_opt_stage2_ifm" if "Opt" in d else
                "npu_qdq_instance_norm_add_mul_add_sd")
        got = call_n(name, *common, a1, a2, cst, OUT, I64(hw), I64(c), ct_code, split, I64(sv_hw),
                     I64(pad), dq_sc, dq_zp, q_sc, q_zp, dq2_sc, dq2_zp, act, slope, n=hw * c,
                     dt=io_dt)
        ref = gg.in_qdq(x, io, al, be, ct, split, sv_hw, pad, dq_sc, dq_zp, q_sc, q_zp, act,
                        slope, amad_ops=(a1, gg.decode(cst, ct), a2),
                        dq2=(dq2_sc, dq2_zp)) if small else None
        check(got.reshape(hw, c), data(case, ofm(case), io_dt)[:hw * c].reshape(hw, c), ref)
        return
    slope = float(w.take(1, np.float32)[0])
    want = data(case, ofm(case), io_dt)
    if double:
        ga, gb = call_n("npu_qdq_instance_norm_sd_opt_stage2_ifm_double_out", *common, OUT, OUT2,
                        I64(hw), I64(c), ct_code, split, I64(sv_hw), I64(pad), dq_sc, dq_zp, q_sc,
                        q_zp, q2[0], q2[1], act, slope, n=hw * c, n2=hw * c, dt=io_dt)
        ra = rb = None
        if small:
            ra, rb = gg.in_qdq(x, io, al, be, ct, split, sv_hw, pad, dq_sc, dq_zp, q_sc, q_zp,
                               act, slope, q2=q2)
        check(ga.reshape(hw, c), want[:hw * c].reshape(hw, c), ra)
        check(gb.reshape(hw, c), want[hw * c:2 * hw * c].reshape(hw, c), rb)
        return
    name = ("npu_qdq_instance_norm_sd_opt_stage2_ifm" if "Opt" in d else
            "npu_qdq_instance_norm_sd")
    got = call_n(name, *common, OUT, I64(hw), I64(c), ct_code, split, I64(sv_hw), I64(pad), dq_sc,
                 dq_zp, q_sc, q_zp, act, slope, n=hw * c, dt=io_dt)
    ref = gg.in_qdq(x, io, al, be, ct, split, sv_hw, pad, dq_sc, dq_zp, q_sc, q_zp, act,
                    slope) if small else None
    check(got.reshape(hw, c), want[:hw * c].reshape(hw, c), ref)


STREAM = ["InstanceNormStreamBf16_wrapper", "qdq_instanceNormStream_wrapper"]


@pytest.mark.parametrize("d,case", cases(STREAM))
def test_in_stream(d, case):
    t = tc(case)
    hw = t["ifm_sv_height"] * t["ifm_height_iteration"] * t["ifm_sv_width"] * t["ifm_width_iteration"]
    c = t["ifm_sv_channel"] * t["ifm_channel_iteration"]
    io, io_code, io_dt = TY[t["ifm_dtype"]]
    ab, ab_code, ab_dt = FL[t["alpha_dtype"]]
    x = to_hw_c(data(case, "ifm32.txt", io_dt), hw, c)
    want = to_hw_c(data(case, ofm(case), io_dt), hw, c)
    w = Wts(data(case, "wts32.txt"))
    nb = np.dtype(ab_dt).itemsize
    if d == "InstanceNormStreamBf16_wrapper":
        alpha = w.take(-(-c // 128) * 128, ab_dt)[:c]
        beta = w.take(c, ab_dt)
        got = call_n("npu_instance_norm_stream_bf16", x, alpha, beta, ab_code, OUT, I64(hw), I64(c),
                     n=hw * c, dt=io_dt)
        ref = gg.stream_bf16(x, gg.decode(alpha, ab), gg.decode(beta, ab))
    else:
        padded = -(-c * nb // 128) * 128 // nb
        alpha = w.take(padded, ab_dt)[:c]
        beta = w.take(padded, ab_dt)[:c]
        dq_sc, q_sc, dq_off, q_zp, relu_alpha = (float(v) for v in w.take(5, np.float32))
        got = call_n("npu_qdq_instance_norm_stream", x, io_code, alpha, beta, ab_code, OUT, I64(hw),
                     I64(c), dq_sc, dq_off, q_sc, q_zp, relu_alpha, n=hw * c, dt=io_dt)
        ref = gg.stream_qdq(x, io, gg.decode(alpha, ab), gg.decode(beta, ab), dq_sc, dq_off,
                            q_sc, q_zp, relu_alpha)
    check(got.reshape(hw, c), want, ref)


GN = ["groupNormSd_bf16_wrapper", "qdq_groupnormsd_wrapper", "qdq_groupnorm_wrapper"]
# split_mode -> (spatial split, group split)
GN_SPLIT = {-1: (1, 12), 0: (12, 1), 1: (4, 3), 2: (2, 6), 3: (3, 4)}


def gn_alpha_beta(raw, G, cg, dt, tail_bytes):
    """alpha copies, beta copies, then tail_bytes of coefficient blocks."""
    v = raw[:raw.size - tail_bytes].view(dt)
    copies = v.size // (2 * G * cg)
    assert v.size == 2 * G * cg * copies
    blocks = v.reshape(2, G, copies, cg)
    return np.ascontiguousarray(blocks[0, :, 0]).reshape(-1), \
        np.ascontiguousarray(blocks[1, :, 0]).reshape(-1)


@pytest.mark.parametrize("d,case", cases(GN))
def test_gn(d, case):
    t = tc(case)
    split = t.get("split_mode", 0)
    ss, gs = GN_SPLIT[split] if d != "qdq_groupnorm_wrapper" else (1, 1)
    G = t["ifm_sv_group"] * t.get("ifm_group_iteration", 1) * gs
    cg = t["ifm_sv_channel"]
    sv = t["ifm_sv_height"] * t["ifm_sv_width"] * cg
    l = t["ifm_sv_height"] * t["ifm_height_iteration"] * t["ifm_sv_width"] * \
        t["ifm_width_iteration"] * cg * ss
    io, io_code, io_dt = TY[t["ifm_dtype"]]
    ab, ab_code, ab_dt = FL[t["alpha_dtype"]]
    x = data(case, "ifm32.txt", io_dt)[:G * l].reshape(G, l)
    want = data(case, ofm(case), io_dt)[:G * l].reshape(G, l)
    eps = f(t["epsilon"])
    pad = t.get("ifm_spatial_pad", 0)
    act = t.get("activation_type", 0)
    raw = data(case, "wts32.txt")
    small = G * l <= REF_MAX
    if d == "groupNormSd_bf16_wrapper":
        alpha, beta = gn_alpha_beta(raw, G, cg, ab_dt, 0)
        got = call_n("npu_groupnorm_sd_bf16", x, io_code, alpha, beta, ab_code, OUT, I64(G), I64(l),
                     I64(cg), split, I64(sv), I64(pad), eps, act, n=G * l, dt=io_dt)
        ref = gg.gn_bf16_sd(x, io, gg.decode(alpha, ab), gg.decode(beta, ab), cg, split, sv, pad,
                            eps, act) if small else None
        check(got.reshape(G, l), want, ref)
        return
    alpha, beta = gn_alpha_beta(raw, G, cg, ab_dt, 1024)
    q = raw[raw.size - 1024:].view(np.float32).reshape(4, 64)
    assert np.all(q == q[:, :1])
    dq_zp, dq_sc, q_zp, q_sc = (float(v) for v in q[:, 0])
    al, be = gg.decode(alpha, ab), gg.decode(beta, ab)
    if d == "qdq_groupnormsd_wrapper":
        ct, ct_code, _ = FL[t["compute_dtype"]]
        got = call_n("npu_qdq_groupnorm_sd", x, io_code, alpha, beta, ab_code, OUT, I64(G), I64(l),
                     I64(cg), ct_code, split, I64(sv), I64(pad), eps, dq_sc, dq_zp, q_sc, q_zp, act,
                     n=G * l, dt=io_dt)
        ref = gg.gn_qdq_sd(x, io, al, be, cg, ct, split, sv, pad, eps, dq_sc, dq_zp, q_sc, q_zp,
                           act) if small else None
    else:
        got = call_n("npu_qdq_groupnorm", x, io_code, alpha, beta, ab_code, OUT, I64(G), I64(l),
                     I64(cg), I64(sv), eps, dq_sc, dq_zp, q_sc, q_zp, n=G * l, dt=io_dt)
        ref = gg.gn_qdq(x, io, al, be, cg, sv, eps, dq_sc, dq_zp, q_sc, q_zp) if small else None
    check(got.reshape(G, l), want, ref)


STAGE2 = ["qdq_ConvInstanceNormAddMulAdd_Stage2_wrapper"]


def conv_geometry(t):
    """(ho, wo, co_total, co_real, kh, kw, ci)."""
    split, cols = t["split_mode"], t["col_used"]
    hw_split, oc_split = {0: (12, 1), 1: (4, cols)}.get(split, (2, 2 * cols))
    ho = t["ofm_sv_height"] * t["ofm_height_iteration"] * (1 if split == 0 else hw_split)
    wo = t["ofm_sv_width"] * t["ofm_width_iteration"] * (hw_split if split == 0 else 1)
    co = t["ofm_sv_depth"] * t["ofm_depth_iteration"] * oc_split
    kh = t["kernel_sv_height"] * t["kernel_sv_height_iteration"]
    ci = t["ifm_sv_depth"] * t["ifm_depth_iteration"]
    return ho, wo, co, t["ofm_depth_real"], kh, t["kernel_width"], ci


def inst_alpha_dtype(t):
    """alpha / beta dtype of the conv fusions: the compute dtype."""
    return t["inst_compute_dtype"]


def inst_wts(case, c, ct_dt, ab_dt, amad):
    w = Wts(data(case, "inst_wts32.txt"))
    alpha, beta = w.alpha_beta(c, ab_dt)
    blocks = [w.block() for _ in range(6 if amad else 4)]
    cst = w.take(c, ct_dt) if amad else None
    slope = float(w.take(1, np.float32)[0])
    return alpha, beta, blocks, cst, slope


@pytest.mark.parametrize("d,case", cases(STAGE2))
def test_conv_in_stage2(d, case):
    t = tc(case)
    ho, wo, co, co_real, kh, kw, ci = conv_geometry(t)
    assert t["split_mode"] == 0 and t["inst_split_mode"] == 0
    ot, ot_code, ot_dt = INT[t["inst_ifm_dtype"]]
    ct, ct_code, ct_dt = FL[t["inst_compute_dtype"]]
    ab, ab_code, ab_dt = FL[inst_alpha_dtype(t)]
    # input: the 3 column bands of the width, each (ho, wo / 3, c) flat
    l2_ow = 4 * t["inst_ifm_sv_width"] * t["inst_ifm_width_iteration"]
    raw = data(case, "ifm32.txt", ot_dt)[:ho * wo * co_real]
    bands = raw.reshape(wo // l2_ow, ho, l2_ow, co_real)
    x = np.ascontiguousarray(bands.transpose(1, 0, 2, 3)).reshape(ho * wo, co_real)
    ps = data(case, "ofm1.txt", np.float32)[:2 * co_real].reshape(co_real // 32, 2, 32)
    S = np.ascontiguousarray(ps[:, 0]).reshape(-1)
    S2 = np.ascontiguousarray(ps[:, 1]).reshape(-1)
    alpha, beta, blocks, cst, slope = inst_wts(case, co_real, ct_dt, ab_dt, True)
    dq_zp, dq_sc, q_zp, q_sc, dq2_zp, dq2_sc = blocks
    ifm2 = data(case, "ifm2_32.txt", ot_dt)
    a1, a2 = ifm2[:co_real], ifm2[co_real:2 * co_real]
    N = 12 * t["inst_ifm_sv_height"] * t["inst_ifm_sv_width"] * t["inst_ifm_height_iteration"] * \
        t["inst_ifm_width_iteration"] - t["inst_ifm_spatial_pad"]
    act = t["inst_activation"]
    got = call_n("npu_qdq_ConvInstanceNormAddMulAdd_Stage2", x, ot_code, S, S2, alpha, beta,
                 ab_code, a1, a2, cst, OUT, I64(ho * wo), I64(co_real), I64(N), ct_code, dq_sc,
                 dq_zp, q_sc, q_zp, dq2_sc, dq2_zp, act, slope, n=ho * wo * co_real, dt=ot_dt)
    want = data(case, ofm(case), ot_dt)[:ho * wo * co_real].reshape(ho * wo, co_real)
    ref = gg.in_stage2(x, ot, S, S2, N, gg.decode(alpha, ab), gg.decode(beta, ab), ct, dq_sc,
                       dq_zp, q_sc, q_zp, act, slope, (a1, gg.decode(cst, ct), a2),
                       (dq2_sc, dq2_zp))
    check(got.reshape(ho * wo, co_real), want, ref)
