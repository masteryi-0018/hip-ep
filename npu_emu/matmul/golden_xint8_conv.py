# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for matmul/xint8_conv.h. Logical HWC / HWIO tensors, int64 arithmetic."""

import numpy as np

from golden_int_round import Rnd, clamp64, float_to_int, sat_add64, saturate, shift_round
from golden_mac import add, mac_exact, mul
from golden_minmax import vmax
from golden_act import hw_gelu_f32

f32 = np.float32


def ups_bias(bias, n, shift):
    """The int16 bias shifted into the 32-bit accumulator (saturating); zeros without bias."""
    if bias is None:
        return np.zeros(n, np.int64)
    return saturate(shift_round(bias, -shift, Rnd.kFloor), "i32")


def conv(ifm, wts, sh, sw):
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


def dwconv(ifm, wts, sh, sw):
    kh, kw, c = wts.shape
    h, w, _ = ifm.shape
    ho, wo = (h - kh) // sh + 1, (w - kw) // sw + 1
    x, g = ifm.astype(np.int64), wts.astype(np.int64)
    acc = np.zeros((ho, wo, c), np.int64)
    for y in range(kh):
        for xx in range(kw):
            acc += x[y:y + (ho - 1) * sh + 1:sh, xx:xx + (wo - 1) * sw + 1:sw] * g[y, xx]
    return acc


def rng_of(dt):
    i = np.iinfo(dt)
    return int(i.min), int(i.max)


def lrelu(a, alpha_q16, shift, out_dt):
    alpha = f32(alpha_q16 / 65536.0)
    m = vmax(a.astype(f32), mac_exact(a, alpha, f32(0)))
    return clamp64(float_to_int(m, "i64", -shift), *rng_of(out_dt)).astype(out_dt)


def xint8_conv(ifm, wts, bias, out_dt, sh, sw, shift_bias, shift_out, max_value, use_lrelu,
               alpha_q16, shift_lrelu_pos):
    a = conv(ifm, wts, sh, sw) + ups_bias(bias, wts.shape[3], shift_bias)
    if use_lrelu:
        return lrelu(a, alpha_q16, shift_lrelu_pos, out_dt)
    if max_value >= 0:
        a = np.minimum(a, shift_round(max_value, -shift_out, Rnd.kFloor))
    return clamp64(shift_round(a, shift_out, Rnd.kHalfEven), *rng_of(out_dt)).astype(out_dt)


def xint8_depthwise(ifm, wts, bias, out_dt, sh, sw, shift_bias, shift_out, max_value, away):
    a = dwconv(ifm, wts, sh, sw) + ups_bias(bias, wts.shape[2], shift_bias)
    r = clamp64(shift_round(a, shift_out, Rnd.kHalfAway if away else Rnd.kHalfEven), *rng_of(out_dt))
    if max_value >= 0:
        r = np.minimum(r, max_value)
    return r.astype(out_dt)


def xint8_dwc_pw(ifm, dwc_wts, dwc_bias, pw_wts, pw_bias, sh, sw, dwc_shift_bias, dwc_shift_out,
                 pw_shift_bias, pw_shift_out):
    d = xint8_depthwise(ifm, dwc_wts, dwc_bias, np.int8, sh, sw, dwc_shift_bias, dwc_shift_out,
                        -1, False)
    return xint8_conv(d, pw_wts.reshape(1, 1, *pw_wts.shape), pw_bias, np.int8, 1, 1,
                      pw_shift_bias, pw_shift_out, -1, 0, 0, 0)


def xint8_conv_add(ifm, wts, bias, ifm2, x0, out_dt, sh, sw, shift_bias, shift_conv_q,
                   shift_ifm2, shift_out, max_value, use_lrelu, alpha_q16, shift_lrelu_neg,
                   shift_lrelu_pos, shift_add_in, shift_add_res, mac_coeff, mac_shift_x0,
                   mac_shift_out):
    a = conv(ifm, wts, sh, sw) + ups_bias(bias, wts.shape[3], shift_bias)
    i2 = shift_round(ifm2, -shift_ifm2, Rnd.kFloor)
    if use_lrelu:
        neg = np.minimum(clamp64(shift_round(a, shift_lrelu_neg, Rnd.kHalfEven), -32768, 32767), 0)
        pos = clamp64(shift_round(a, shift_lrelu_pos, Rnd.kHalfEven), 0, 65535)
        m = saturate(shift_round(pos, -shift_out, Rnd.kFloor), "i32") + alpha_q16 * neg
        y = clamp64(shift_round(m, shift_out, Rnd.kHalfEven), -128, 127)
        s = sat_add64(shift_round(y, -shift_add_in, Rnd.kFloor), i2)
        r = clamp64(shift_round(s, shift_add_res, Rnd.kHalfEven), -128, 127)
    else:
        q8 = clamp64(shift_round(a, shift_conv_q, Rnd.kHalfEven), -128, 127)
        q = a if shift_conv_q < 0 else shift_round(q8, -shift_conv_q, Rnd.kFloor)
        s = sat_add64(q, i2)
        if max_value >= 0:
            s = np.minimum(s, shift_round(max_value, -shift_out, Rnd.kFloor))
        r = clamp64(shift_round(s, shift_out, Rnd.kHalfEven), 0 if max_value >= 0 else -128, 127)
    if x0 is not None:
        m = sat_add64(mac_coeff * r, shift_round(x0, -mac_shift_x0, Rnd.kFloor))
        r = clamp64(shift_round(m, mac_shift_out, Rnd.kHalfEven), -128, 127)
    return r.astype(out_dt)


def qxint8_conv(ifm, wts, bias, out_dt, sh, sw, bias_shift, shift, activation, zp_out, max_value,
                alpha_q16, gelu_scale=0.0, gelu_inv_scale=0.0):
    a = conv(ifm, wts, sh, sw)
    if bias is not None:
        a = sat_add64(a, shift_round(bias.astype(np.int64), -bias_shift, Rnd.kFloor))
    lo, hi = rng_of(out_dt)
    if activation == 3:
        # max(A, alpha * A + (1 - alpha) * O), O = zp_out * 2^shift
        with np.errstate(all="ignore"):
            alpha = f32(alpha_q16 / 65536.0)
            offset = mul(add(f32(1), -alpha), np.ldexp(f32(zp_out), shift))
            n = add(mac_exact(a, alpha, f32(0)), offset)
            v = vmax(a.astype(f32), n)
            q = float_to_int(mul(v, f32(np.ldexp(1.0, -shift))), "i64")
        return clamp64(q, lo, hi).astype(out_dt)
    r = clamp64(shift_round(a, shift, Rnd.kHalfEven), lo, hi)
    if activation in (1, 2):
        r = clamp64(r, zp_out, max_value)
    if activation == 6:     # GeLU
        with np.errstate(all="ignore"):
            real = mul(add(r.astype(f32), -f32(zp_out)), f32(gelu_scale))
            q = add(mul(hw_gelu_f32(real), f32(gelu_inv_scale)), f32(zp_out))
        r = clamp64(float_to_int(q, "i64"), lo, hi)
    return r.astype(out_dt)
