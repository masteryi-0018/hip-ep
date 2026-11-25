# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""matmul/conv_instancenorm.h entry points against the reference models of the model repository."""

import numpy as np
import pytest

import fixture as fm
from fixture import TYPES as INT
from npu import I64, OUT, call

DIRS = ["qdq_ConvInFusion_A16_wrapper", "qdq_ConvInstanceNormAddMulAdd_wrapper"]
HOOKS = {"QDQ_ConvIn*": ("compute_bit_exact", "_conv_qdq_compute")}
f32 = np.float32


def enc(v, name):
    """Float values in the storage of a float type."""
    v = np.ascontiguousarray(v, f32)
    if name == "bfloat16":
        return fm.bf16_bits(v)
    return v.astype(np.float16).view(np.uint16) if name == "float16" else v


def extract(d, seen):
    """Logical conv operands, norm coefficients, parameters and expected output."""
    s = fm.last(seen)
    data = s.data
    _, ifm, wts, c2, c1, c0 = fm.args(seen, "_conv_qdq_compute")[:6]
    ifm, wts = np.asarray(ifm), np.asarray(wts)
    n, co = wts.shape[3], int(s.conv_ofm_depth_real)
    clip = s._conv_gen_clip_coeff()
    amad = "inst_add1_act" in data
    it = s.inst_ifm_dtype
    fx = dict(ifm=ifm, wts=np.ascontiguousarray(wts[..., :co]), c0=fm.col(c0, n)[:co],
              c1=fm.col(c1, n)[:co] if s.conv_has_actv_sum else None, c2=fm.col(c2, n)[:co],
              conv_act=np.array([int(s.conv_has_clip)]), clip=np.asarray(clip, f32)[:2],
              alpha=enc(data["inst_alpha"][:co], s.inst_alpha_dtype),
              beta=enc(data["inst_beta"][:co], s.inst_beta_dtype),
              types=np.array([it, s.inst_alpha_dtype, s.inst_compute_dtype]),
              geo=np.array([s.conv_stride_height, s.conv_stride_width, s.inst_split_mode,
                            s.inst_ifm_sv_height, s.inst_ifm_sv_width, s.inst_ifm_height_iteration,
                            s.inst_ifm_width_iteration, s.inst_ifm_spatial_pad, s.inst_activation]),
              fl=np.array([s.inst_ifm_scale, s.inst_ifm_zero_point, 1.0 / s.inst_ofm_scale,
                           s.inst_ofm_zero_point, s.inst_relu_alpha], f32),
              ofm=np.asarray(data["inst_ofm"])[..., :co].astype(INT[it][1]))
    if amad:
        fx.update(add1_act=np.asarray(data["inst_add1_act"]).reshape(-1)[:co].astype(INT[it][1]),
                  add2=np.asarray(data["inst_add2"]).reshape(-1)[:co].astype(INT[it][1]),
                  add1_const=enc(np.asarray(data["inst_add1_const"]).reshape(-1)[:co],
                                 s.inst_compute_dtype),
                  fl2=np.array([s.inst_add_scale, s.inst_add_zero_point], f32))
    return fx


@pytest.mark.parametrize("d,case", fm.cases(DIRS, __file__))
def test_fixture(d, case):
    z = fm.load(case)
    ifm, wts, want = z["ifm"], z["wts"], z["ofm"]
    h, w, ci = ifm.shape
    kh, kw, _, co = wts.shape
    it, ab, ct = (str(v) for v in z["types"])
    sh, sw, split, sv_h, sv_w, h_loop, w_loop, pad, act = (int(v) for v in z["geo"])
    dq_sc, dq_zp, q_sc, q_zp, slope = (float(v) for v in z["fl"])
    # the conv activation, the activation sum and the norm activation / compute type are fixed
    assert int(z["conv_act"][0]) == 0 and "c1" not in z and act == 3
    assert ab == ct == "float16"
    shape = (I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), sh, sw, split, I64(sv_h),
             I64(sv_w), I64(h_loop), I64(w_loop), I64(pad), z["c0"], z["c2"], dq_zp, dq_sc, q_zp,
             q_sc)
    head = (ifm, INT[ifm.dtype.name][0], wts, INT[wts.dtype.name][0], z["alpha"], z["beta"])
    if "add1_act" in z:
        dq2_sc, dq2_zp = (float(v) for v in z["fl2"])
        name = "npu_qdq_ConvInstanceNormAddMulAdd"
        args = (*head, z["add1_act"], z["add2"], z["add1_const"], OUT, INT[it][0], *shape, dq2_zp,
                dq2_sc, slope)
    else:
        name = "npu_qdq_ConvInFusion_A16"
        args = (*head, OUT, INT[it][0], *shape, slope)
    got = call(name, *args, out_nbytes=want.nbytes, out_dtype=want.dtype)
    fm.assert_bytes(got, want)
