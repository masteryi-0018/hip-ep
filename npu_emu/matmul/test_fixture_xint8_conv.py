# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The xint8 convolution entry points against the reference models of the model repository."""

import math

import numpy as np
import pytest

import fixture as fm
from fixture import NPU_TYPE as T
from npu import I64, OUT, call

DIRS = ["qxint8_conv_wrapper", "xint8_conv1x1ci64co64_wrapper", "xint8_conv_add_wrapper",
        "xint8_conv_pin_buffer_wrapper", "xint8_conv_pin_stream_wrapper",
        "xint8_conv_q_add_mac_wrapper", "xint8_conv_q_add_wrapper", "xint8_conv_wrapper",
        "xint8_depthwise_conv_wrapper", "xint8_dwc3x3_conv1x1_fused_wrapper",
        "xint8_dwc7x7_s1_wrapper", "spec_xint8_conv_wrapper/reference",
        "spec_xint8_conv_wrapper/slice_3xci32_zip"]
# the speculative model runs the generic model inside its compute_bit_exact; recording the outer
# call makes the speculative object the last one
HOOKS = {"Xint8_*": ("compute_bit_exact",), "Qxint8_*": ("compute_bit_exact",),
         "spec_wrapper_common": ("compute_bit_exact",)}
CONV = ("Xint8_ConvWrapper", "Xint8_ConvPinBufferWrapper", "Xint8_ConvPinStreamWrapper",
        "Xint8_Conv1x1Ci64Co64Wrapper")
DEPTHWISE = ("Xint8_DepthwiseConvWrapper", "Xint8_Dwc7x7S1Wrapper")
FUSED = "Xint8_Dwc3x3Conv1x1FusedWrapper"
ADD = ("Xint8_ConvQAddWrapper", "Xint8_ConvAddWrapper", "Xint8_ConvQAddMacWrapper")
QUANT = "Qxint8_ConvWrapper"
GELU = "Xint8_GeLUWrapper"
SPEC = "Spec_Xint8_ConvWrapper"
# entries of the subdirectories of the speculative wrapper directory
SUBDIR_ENTRY = {"reference": "xint8_conv", "slice_3xci32_zip": "spec_xint8_conv"}
I8, U8 = np.int8, np.uint8


def bias_or_none(b, do_bias):
    return np.asarray(b).astype(np.int16) if do_bias else None


def gelu_params(s, act):
    """Parameters of the GeLU kernel that follows the convolution."""
    if act == 6:
        import golden_gelu as gg
        gp = gg.qdq_gelu_coeffs(s.conv_output_scale, 0, s.ofm_scale, 0)
        return np.array([float(v) for v in gp], np.float64), "qdq"
    import importlib
    cls = getattr(importlib.import_module("wrapper." + GELU), GELU)
    xg = cls()
    xg.set_params({"ifm_sv_size": 64, "ifm_sv_iteration": 1, "ifm_scale": s.conv_output_scale,
                   "ofm_scale": s.ofm_scale, "precision_enhanced": 1})
    k, a = xg.k, max(xg.k, 0)
    return np.array([int(k >= 1), list(cls._RND).index(xg.rnd), a, a + 1 - k, a + xg.s,
                     max(-k, 0), int(xg.K), int(xg.P), int(xg.precision_enhanced)], np.int64), "xint8"


def extract(d, seen):
    """Logical operands, parameters and expected output of the model object."""
    s = fm.last(seen)
    cls, data = type(s).__name__, s.data
    act = int(getattr(s, "activation_type", 0))
    if cls in CONV or cls == SPEC:
        out_dt = U8 if act in (1, 2) else I8
        scal = (int(s.shift_bias), int(s.shift_out), int(s.max_value) if act in (1, 2) else -1,
                int(act == 3), int(s.lrelu_alpha), int(s.shift_lrelu_in + s.shift_out - 16))
        if cls != CONV[0]:  # no LeakyReLU; the 1 x 1 kernels have no activation at all
            assert act != 3 and (cls != CONV[3] or act == 0) and (cls != SPEC or act in (0, 10))
            scal = scal[:2 if cls in (CONV[3], SPEC) else 3]
        fx = dict(ifm=np.asarray(data["ifm"]),
                  wts=np.asarray(data["wts"]).astype(I8), bias=bias_or_none(data["bias"], s.do_bias),
                  stride=(s.stride_height, s.stride_width), scal=scal)
        if act in (6, 10):
            fx["gelu"], fx["gelu_kind"] = gelu_params(s, act)
    elif cls in DEPTHWISE:
        out_dt = U8 if act in (1, 2) or s.ofm_dtype == "uint8" else I8
        mx = (min(int(round(6.0 / s.ofm_scale)), 127) if act == 2 else 127) if act in (1, 2) else -1
        fx = dict(ifm=np.asarray(data["ifm"]),
                  wts=np.asarray(data["wts"]), bias=bias_or_none(data["bias"], s.do_bias),
                  stride=(s.stride_height, s.stride_width),
                  scal=(int(s.shift_bias), int(s.shift_out), mx))
    elif cls == FUSED:
        dp, pp = s._build_dwc_param(), s._build_pw_param()
        if pp.has_lrelu or pp.has_relu6:
            raise fm.Skip("activation after the pointwise convolution is not emulated")
        out_dt = I8
        pw = np.asarray(data["pw_wts"]).astype(I8)
        co, ci = pw.shape[:2]
        fx = dict(ifm=np.asarray(data["ifm"]).astype(I8),
                  wts=np.asarray(data["dwc_wts"]).astype(I8),
                  bias=bias_or_none(data["dwc_bias"], dp.bias_C > 0),
                  pw_wts=np.ascontiguousarray(pw.reshape(co, ci).T),
                  pw_bias=bias_or_none(data["pw_bias"], pp.bias_C > 0), stride=(dp.Sh, dp.Sw),
                  scal=(int(dp.shift_bias), int(dp.shift_out), int(pp.shift_bias),
                        int(pp.shift_out)))
    elif cls in ADD:
        mac = cls == ADD[2]
        lr = act == 3
        out_dt = I8 if (mac or act not in (1, 2)) else U8
        if lr:
            r = int(round(math.log2(s.ifm2_scale / s.ofm_scale)))
            si2, sai, sar = max(0, r), max(0, -r), max(0, -r)
        else:
            si2, sai, sar = int(s.shift_ifm2_to_acc), 0, 0
        scal = (int(s.shift_bias), -1 if cls == ADD[1] else int(s.shift_conv_q), si2,
                int(s.shift_out))
        if mac:  # no activation
            assert act == 0
            scal += (int(s.mac_coeff), int(s.mac_shift_x0), int(s.mac_shift_out))
        else:
            scal += (int(s.max_value) if act in (1, 2) else -1, int(lr), int(s.lrelu_alpha),
                     int(s.shift_lrelu_in), int(s.shift_lrelu_pos), sai, sar)
        fx = dict(ifm=np.asarray(data["ifm"]).astype(I8),
                  wts=np.asarray(data["wts"]).astype(I8), bias=bias_or_none(data["bias"], s.do_bias),
                  ifm2=np.asarray(data["ifm2"]).astype(I8),
                  x0=np.asarray(data["x0"]).astype(I8) if mac else None,
                  stride=(s.stride_height, s.stride_width), scal=scal)
    elif cls == QUANT:
        out_dt = np.dtype(s.ofm_dtype).type
        hi = np.iinfo(out_dt).max
        zo = int(s.ofm_zero_point)
        mx = min(hi, round(6.0 / s.ofm_scale + zo)) if act == 2 else hi
        fx = dict(ifm=np.asarray(data["ifm"]),
                  wts=np.asarray(data["wts"]).astype(I8),
                  bias=np.asarray(s.packed_bias).astype(np.int16),
                  stride=(s.stride_height, s.stride_width),
                  scal=(int(s._packing_shifts()[0]), int(s.base_shift_out), act, zo, int(mx),
                        int(s.lrelu_alpha) if act == 3 else 0))
        if act == 6:
            fx["fl"] = np.array([s.ofm_scale, 1.0 / s.ofm_scale], np.float32)
    else:
        raise fm.Skip(f"no entry for {cls}")
    # the entry is named after the kernel the directory builds
    entry = SUBDIR_ENTRY.get(d) or d[:-len("_wrapper")].replace("_q_add", "_add")
    fx.update(entry=entry, out_dt=np.dtype(out_dt).name,
              ofm=np.asarray(data["ofm_0"]).astype(out_dt), cls=cls)
    return fx


@pytest.mark.parametrize("d,case", fm.cases(DIRS, __file__))
def test_fixture(d, case):
    z = fm.load(case)
    entry, out_dt = str(z["entry"]), str(z["out_dt"])
    ifm, wts = z["ifm"], z["wts"]
    sh, sw = (int(v) for v in z["stride"])
    scal = [int(v) for v in z["scal"]]
    want = z["ofm"]
    h, w = ifm.shape[:2]
    act = 0
    if "gelu" in z:  # the GeLU kernel the convolution chains
        g = z["gelu"]
        if str(z["gelu_kind"]) == "qdq":    # zi, zo, ka, c, kdsp; the zero points are 0
            act, gelu = 6, [float(v) for v in g[2:5]] + [0] * 8
        else:                               # xgelu words, then precision_enhanced
            act, gelu = 10, [0.0] * 3 + [int(v) for v in g[:8]]
    else:
        gelu = [0.0] * 3 + [0] * 8
    if entry in ("xint8_conv", "xint8_conv_pin_buffer", "xint8_conv_pin_stream",
                 "xint8_conv1x1ci64co64", "spec_xint8_conv"):
        # scal: shift_bias, shift_out[, max_value (< 0: none)[, lrelu, alpha, shift_lrelu_pos]]
        kh, kw, ci, co = wts.shape
        sb, so = scal[:2]
        mx = scal[2] if len(scal) > 2 else -1
        lr = len(scal) > 3 and scal[3]
        if mx >= 0:
            act = 2
        if entry == "xint8_conv":
            words = (scal[4] if lr else 0, max(mx, 0), sb, scal[5] if lr else 0, so,
                     3 if lr else act)
        else:
            words = (max(mx, 0), sb, so, act)
        args = (ifm, wts, fm.opt(z, "bias"), OUT, T[out_dt], I64(h), I64(w), I64(ci), I64(kh),
                I64(kw), I64(co), sh, sw, *words, *gelu)
        if entry == "xint8_conv1x1ci64co64":
            assert (kh, kw, sh, sw, out_dt, act) == (1, 1, 1, 1, "int8", 0)
            args = (ifm, wts[0, 0], fm.opt(z, "bias"), OUT, I64(h), I64(w), I64(ci), I64(co), sb,
                    so)
        elif entry == "spec_xint8_conv":
            assert (kh, kw, sh, sw, out_dt) == (1, 1, 1, 1, "int8")
            args = (ifm, wts[0, 0], fm.opt(z, "bias"), OUT, I64(h), I64(w), I64(ci), I64(co), sb,
                    so, act, *gelu[3:])
    elif entry in ("xint8_depthwise_conv", "xint8_dwc7x7_s1"):
        kh, kw, co = wts.shape
        geom = (I64(kh), I64(kw), sh, sw)
        if entry == "xint8_dwc7x7_s1":
            assert (kh, kw, sh, sw) == (7, 7, 1, 1)
            geom = ()
        sb, so, mx = scal    # max_value < 0: none
        args = (ifm, wts, fm.opt(z, "bias"), OUT, T[out_dt], I64(h), I64(w), I64(co), *geom,
                sb, so, max(mx, 0), 2 if mx >= 0 else 0)
    elif entry == "xint8_dwc3x3_conv1x1_fused":
        kh, kw, ci = wts.shape
        co = z["pw_wts"].shape[1]
        args = (ifm, wts, fm.opt(z, "bias"), z["pw_wts"], fm.opt(z, "pw_bias"), OUT, I64(h), I64(w),
                I64(ci), I64(kh), I64(kw), I64(co), sh, sw, *scal)
    elif entry in ("xint8_conv_add", "xint8_conv_add_mac"):
        # scal: shift_bias, shift_conv_q (< 0: no intermediate Q), shift_ifm2, shift_out, then
        # mac_coeff, mac_shift_x0, mac_shift_out (mac) or max_value (< 0: none), lrelu, alpha,
        # shift_lrelu_neg, shift_lrelu_pos, shift_add_in, shift_add_res
        kh, kw, ci, co = wts.shape
        sb, scq, si2, so = scal[:4]
        geo = (I64(h), I64(w), I64(ci), I64(kh), I64(kw), I64(co), sh, sw)
        if entry == "xint8_conv_add_mac":
            assert out_dt == "int8" and scq >= 0
            args = (ifm, z["ifm2"], wts, fm.opt(z, "bias"), z["x0"], OUT, *geo, sb, so, scq, si2,
                    *scal[4:])
        else:
            mx, lr, alpha, sln, slp, sai, sar = scal[4:]
            if lr:
                words = (alpha, 0, sb, slp, sln, so, sai, si2, sar, 0, 3)
            else:
                words = (0, max(mx, 0), sb, 0, 0, so, max(scq, 0), si2, 0, int(scq >= 0),
                         2 if mx >= 0 else 0)
            args = (ifm, z["ifm2"], wts, fm.opt(z, "bias"), OUT, *geo, *words)
        out_dt = "int8"   # the residual result is int8 (ReLU / ReLU6: in [0, 127])
    elif entry == "qxint8_conv":
        # scal: shift_bias, base shift_out, activation, zero point, max_value, alpha
        kh, kw, ci, co = wts.shape
        sb, base, a, zo, mx, alpha = scal
        fl = [float(v) for v in z["fl"]] if "fl" in z else [0.0, 0.0]   # s_out, 1 / s_out
        ozpa = float(np.float32(math.ldexp(zo, base)))
        args = (ifm, T[ifm.dtype.name], wts, z["bias"], OUT, T[out_dt], I64(h), I64(w), I64(ci),
                I64(kh), I64(kw), I64(co), sh, sw, alpha, mx & 0xFF, sb, base, base, *fl, zo,
                ozpa, a)
    else:
        raise AssertionError(f"unknown entry {entry}")
    assert ifm.dtype == np.int8 or entry == "qxint8_conv"
    dt = np.dtype(out_dt)
    got = call("npu_" + entry, *args, out_nbytes=want.size * dt.itemsize, out_dtype=dt)
    got, want = np.asarray(got).reshape(-1), want.astype(dt).reshape(-1)
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{entry}: {bad.size}/{want.size} differ, first at {bad[:4].tolist()}: "
                           f"got {got[bad[:4]].tolist()} want {want[bad[:4]].tolist()}")
