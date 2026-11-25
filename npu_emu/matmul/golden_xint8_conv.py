# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for matmul/xint8_conv.h. Logical HWC / HWIO tensors, int64 arithmetic."""

import numpy as np

from golden_act import hw_gelu_f32
from golden_float_cvt import float_to_int, int_f32, int_to_float
from golden_int_round import Rnd, clamp64, sat_add64, shift_round, srs
from golden_mac import add_q, mac_exact, mul_q, sub_q
from golden_minmax import vmax
from golden_quant import requant_tail

f32 = np.float32


def _ups_bias(bias, n, shift):
    """The int16 bias shifted into the 32-bit accumulator (saturating); zeros without bias."""
    if bias is None:
        return np.zeros(n, np.int64)
    return srs(bias, -shift, "i32", Rnd.kFloor)


def _conv(ifm, wts, sh, sw):
    """int64 convolution, HWC x HWIO -> HoWoCo."""
    kh, kw, ci, co = wts.shape
    h, w, _ = ifm.shape
    ho, wo = (h - kh) // sh + 1, (w - kw) // sw + 1
    x, g = ifm.astype(np.int64), wts.astype(np.int64)
    acc = np.zeros((ho, wo, co), np.int64)
    for y in range(kh):
        for xx in range(kw):
            acc += x[y:y + (ho - 1) * sh + 1:sh, xx:xx + (wo - 1) * sw + 1:sw] @ g[y, xx]
    return acc


def _dwconv(ifm, wts, sh, sw):
    kh, kw, c = wts.shape
    h, w, _ = ifm.shape
    ho, wo = (h - kh) // sh + 1, (w - kw) // sw + 1
    x, g = ifm.astype(np.int64), wts.astype(np.int64)
    acc = np.zeros((ho, wo, c), np.int64)
    for y in range(kh):
        for xx in range(kw):
            acc += x[y:y + (ho - 1) * sh + 1:sh, xx:xx + (wo - 1) * sw + 1:sw] * g[y, xx]
    return acc


def _rng_of(dt):
    i = np.iinfo(dt)
    return int(i.min), int(i.max)


def _t(dt):
    """The type name of an int8 / uint8 output dtype."""
    return "i8" if np.dtype(dt) == np.int8 else "u8"


def _alpha(lrelu_alpha):
    return int_to_float(lrelu_alpha, -16, Rnd.kHalfEven)


def _clamp_word(max_value, out_dt):
    """The uint8 clamp word in the output's signedness."""
    return int(np.int8(np.uint8(max_value))) if out_dt == np.int8 else int(max_value)


def _lrelu(a, lrelu_alpha, shift, out_dt):
    m = vmax(int_f32(a), mac_exact(a, _alpha(lrelu_alpha), f32(0)))
    return float_to_int(m, _t(out_dt), -shift).astype(out_dt)


def _gelu_chain(r, activation, gelu):
    """The GeLU kernel of activations 6 / 10 on the int8 result; gelu = (ka, c, kdsp, up_path,
    rnd_id, a_shift, b_shift, q4_shift, r_shift, K, P)."""
    import golden_gelu as gg
    if activation == 6:
        return gg.qdq_gelu_exp2_a8(r, 0, 0.0, gelu[0], gelu[1], gelu[2], "i8").astype(np.int8)
    return gg.xint8_gelu(r, *gelu[3:], pe=1)


def xint8_conv(ifm, wts, bias, out_dt, sh, sw, lrelu_alpha, max_value, shift_bias,
               shift_lrelu_pos, shift_out, activation, gelu=(0.0,) * 3 + (0,) * 8):
    a = _conv(ifm, wts, sh, sw) + _ups_bias(bias, wts.shape[3], shift_bias)
    if activation == 3:
        return _lrelu(a, lrelu_alpha, shift_lrelu_pos, out_dt)
    if activation in (1, 2):
        a = np.minimum(a, shift_round(_clamp_word(max_value, out_dt), -shift_out, Rnd.kFloor))
    r = srs(a, shift_out, _t(out_dt)).astype(out_dt)
    return _gelu_chain(r, activation, gelu) if activation in (6, 10) else r


def xint8_conv_pin_buffer(ifm, wts, bias, out_dt, sh, sw, max_value, shift_bias, shift_out,
                          activation, gelu=(0.0,) * 3 + (0,) * 8):
    return xint8_conv(ifm, wts, bias, out_dt, sh, sw, 0, max_value, shift_bias, 0, shift_out,
                      activation, gelu)


xint8_conv_pin_stream = xint8_conv_pin_buffer


def xint8_conv1x1ci64co64(ifm, wts, bias, shift_bias, shift_out):
    """wts Ci x Co."""
    return xint8_conv(ifm, wts.reshape(1, 1, *wts.shape), bias, np.int8, 1, 1, 0, 0,
                      shift_bias, 0, shift_out, 0)


def spec_xint8_conv(ifm, wts, bias, shift_bias, shift_out, activation, xgelu=(0,) * 8):
    return xint8_conv(ifm, wts.reshape(1, 1, *wts.shape), bias, np.int8, 1, 1, 0, 0,
                      shift_bias, 0, shift_out, activation, (0.0,) * 3 + tuple(xgelu))


def _depthwise(ifm, wts, bias, out_dt, sh, sw, shift_bias, shift_out, max_value, activation,
               away):
    a = _dwconv(ifm, wts, sh, sw) + _ups_bias(bias, wts.shape[2], shift_bias)
    rnd = Rnd.kHalfAway if away else Rnd.kHalfEven
    r = srs(a, shift_out, _t(out_dt), rnd)
    if activation:
        r = np.minimum(r, _clamp_word(max_value, out_dt))
    return r.astype(out_dt)


def xint8_depthwise_conv(ifm, wts, bias, out_dt, sh, sw, shift_bias, shift_out, max_value,
                         activation):
    return _depthwise(ifm, wts, bias, out_dt, sh, sw, shift_bias, shift_out, max_value,
                      activation, False)


def xint8_dwc7x7_s1(ifm, wts, bias, out_dt, shift_bias, shift_out, max_value, activation):
    return _depthwise(ifm, wts, bias, out_dt, 1, 1, shift_bias, shift_out, max_value, activation,
                      True)


def xint8_dwc3x3_conv1x1_fused(ifm, dwc_wts, dwc_bias, pw_wts, pw_bias, sh, sw, dwc_shift_bias,
                               dwc_shift_out, pw_shift_bias, pw_shift_out):
    d = xint8_depthwise_conv(ifm, dwc_wts, dwc_bias, np.int8, sh, sw, dwc_shift_bias,
                             dwc_shift_out, 0, 0)
    return xint8_conv1x1ci64co64(d, pw_wts, pw_bias, pw_shift_bias, pw_shift_out)


def _conv_add(ifm, ifm2, wts, bias, x0, sh, sw, lrelu_alpha, max_value, shift_bias,
              shift_lrelu_pos, shift_lrelu_neg, shift_out, shift_in, shift_in1, shift_res,
              has_conv_output_q, activation, mac_coeff=0, mac_shift_x0=0, mac_shift_out=0):
    a = _conv(ifm, wts, sh, sw) + _ups_bias(bias, wts.shape[3], shift_bias)
    i2 = shift_round(ifm2, -shift_in1, Rnd.kFloor)
    if activation == 3:
        neg = np.minimum(srs(a, shift_lrelu_neg, "i16"), 0)
        pos = srs(a, shift_lrelu_pos, "u16")
        m = srs(pos, -shift_out, "i32", Rnd.kFloor) + lrelu_alpha * neg
        y = srs(m, shift_out, "i8")
        s = sat_add64(shift_round(y, -shift_in, Rnd.kFloor), i2)
        r = srs(s, shift_res, "i8")
    else:
        q8 = srs(a, shift_in, "i8")
        q = shift_round(q8, -shift_in, Rnd.kFloor) if has_conv_output_q else a
        s = sat_add64(q, i2)
        if activation:
            s = np.minimum(s, shift_round(max_value, -shift_out, Rnd.kFloor))
        r = clamp64(shift_round(s, shift_out, Rnd.kHalfEven), 0 if activation else -128, 127)
    if x0 is not None:
        m = sat_add64(mac_coeff * r, shift_round(x0, -mac_shift_x0, Rnd.kFloor))
        r = srs(m, mac_shift_out, "i8")
    return r.astype(np.int8)


def xint8_conv_add(ifm, ifm2, wts, bias, sh, sw, lrelu_alpha, max_value, shift_bias,
                   shift_lrelu_pos, shift_lrelu_neg, shift_out, shift_in, shift_in1, shift_res,
                   has_conv_output_q, activation):
    return _conv_add(ifm, ifm2, wts, bias, None, sh, sw, lrelu_alpha, max_value, shift_bias,
                     shift_lrelu_pos, shift_lrelu_neg, shift_out, shift_in, shift_in1, shift_res,
                     has_conv_output_q, activation)


def xint8_conv_add_mac(ifm, ifm2, wts, bias, x0, sh, sw, shift_bias, shift_out, shift_in,
                       shift_in1, mac_coeff, mac_shift_x0, mac_shift_out):
    return _conv_add(ifm, ifm2, wts, bias, x0, sh, sw, 0, 0, shift_bias, 0, 0, shift_out,
                     shift_in, shift_in1, 0, 2, 0, mac_coeff, mac_shift_x0, mac_shift_out)


def qxint8_conv(ifm, wts, bias, out_dt, sh, sw, lrelu_alpha, max_value, shift_bias,
                shift_lrelu_pos, shift_out, ofm_scale, inv_ofm_scale, ofm_zero_point,
                output_zero_point_acc, activation):
    a = _conv(ifm, wts, sh, sw)
    if bias is not None:
        a = sat_add64(a, shift_round(bias.astype(np.int64), -shift_bias, Rnd.kFloor))
    t, zp = _t(out_dt), ofm_zero_point
    if activation == 3:
        # max(A, alpha * A + (1 - alpha) * O), O = output_zero_point_acc
        with np.errstate(all="ignore"):
            alpha = _alpha(lrelu_alpha)
            offset = mul_q(sub_q(f32(1), alpha), f32(output_zero_point_acc))
            v = vmax(int_f32(a), add_q(mac_exact(a, alpha, f32(0)), offset))
            q = requant_tail(mul_q(v, f32(np.ldexp(1.0, -shift_lrelu_pos))), t)
        return q.astype(out_dt)
    r = srs(a, shift_out, t)
    if activation in (1, 2):
        r = clamp64(r, zp, _clamp_word(max_value, out_dt))
    if activation == 6:     # GeLU
        with np.errstate(all="ignore"):
            real = mul_q(sub_q(int_f32(r), int_f32(zp)), f32(ofm_scale))
            q = add_q(mul_q(hw_gelu_f32(real), f32(inv_ofm_scale)), int_f32(zp))
        r = requant_tail(q, t)
    return r.astype(out_dt)
