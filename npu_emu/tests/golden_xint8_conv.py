# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for include/npu_emu_xint8_conv.h, written from the AIE4
kernels (mllib_biased_conv_int8x8, biased_depthwise_int8x8, xint8_dwc_7x7_s1,
mllib_biased_conv_add_int8x8, qxint8_conv) and mllib's corrected goldens
(mllib PR #15123). Logical HWC / HWIO tensors, int64 arithmetic."""

import numpy as np

f32 = np.float32


def srs(v, r, away=False):
    """Arithmetic right shift of signed v by r, rounding half to even (or away
    from zero); r <= 0 shifts left."""
    v = np.asarray(v, np.int64)
    if r <= 0:
        return v << -r
    half = np.int64(1) << (r - 1)
    if away:
        return (v + half - (v < 0)) >> r
    return (v + half - 1 + ((v >> r) & 1)) >> r


def shl(v, s):
    v = np.asarray(v, np.int64)
    return v << s if s >= 0 else v >> -s


def ups_bias(bias, n, shift):
    if bias is None:
        return np.zeros(n, np.int64)
    return np.clip(shl(bias.astype(np.int64), shift), -2**31, 2**31 - 1)


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
    m = np.maximum(a.astype(f32), (a.astype(np.float64) * np.float64(alpha)).astype(f32))
    lo, hi = rng_of(out_dt)
    return np.clip(np.rint(m.astype(np.float64) * 2.0**-shift), lo, hi).astype(out_dt)


def xint8_conv(ifm, wts, bias, out_dt, sh, sw, shift_bias, shift_out, max_value, use_lrelu,
               alpha_q16, shift_lrelu_pos):
    a = conv(ifm, wts, sh, sw) + ups_bias(bias, wts.shape[3], shift_bias)
    if use_lrelu:
        return lrelu(a, alpha_q16, shift_lrelu_pos, out_dt)
    if max_value >= 0:
        a = np.minimum(a, shl(max_value, shift_out))
    lo, hi = rng_of(out_dt)
    return np.clip(srs(a, shift_out), lo, hi).astype(out_dt)


def xint8_depthwise(ifm, wts, bias, out_dt, sh, sw, shift_bias, shift_out, max_value, away):
    a = dwconv(ifm, wts, sh, sw) + ups_bias(bias, wts.shape[2], shift_bias)
    lo, hi = rng_of(out_dt)
    r = np.clip(srs(a, shift_out, away), lo, hi)
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
    i2 = ifm2.astype(np.int64)
    if use_lrelu:
        neg = np.minimum(np.clip(srs(a, shift_lrelu_neg), -32768, 32767), 0)
        pos = np.clip(srs(a, shift_lrelu_pos), 0, 65535)
        m = np.clip(shl(pos, shift_out), -2**31, 2**31 - 1) + alpha_q16 * neg
        y = np.clip(srs(m, shift_out), -128, 127)
        r = np.clip(srs(shl(y, shift_add_in) + shl(i2, shift_ifm2), shift_add_res), -128, 127)
    else:
        q = a if shift_conv_q < 0 else shl(np.clip(srs(a, shift_conv_q), -128, 127), shift_conv_q)
        s = q + shl(i2, shift_ifm2)
        if max_value >= 0:
            s = np.minimum(s, shl(max_value, shift_out))
        r = np.clip(srs(s, shift_out), 0 if max_value >= 0 else -128, 127)
    if x0 is not None:
        r = np.clip(srs(mac_coeff * r + shl(x0.astype(np.int64), mac_shift_x0), mac_shift_out),
                    -128, 127)
    return r.astype(out_dt)


def qxint8_conv(ifm, wts, bias, out_dt, sh, sw, bias_shift, shift, activation, zp_out, max_value,
                alpha_q16):
    a = conv(ifm, wts, sh, sw)
    if bias is not None:
        a = a + bias.astype(np.int64) * (1 << bias_shift)
    lo, hi = rng_of(out_dt)
    if activation == 3:
        offset = f32(np.ldexp(zp_out, shift))
        c = a.astype(f32) - offset
        v = np.maximum(c, c * f32(alpha_q16 / 65536.0)) + offset
        return np.clip(np.rint((v * f32(np.ldexp(1.0, -shift))).astype(np.float64)), lo,
                       hi).astype(out_dt)
    r = np.clip(srs(a, shift), lo, hi)
    if activation in (1, 2):
        r = np.clip(r, zp_out, max_value)
    return r.astype(out_dt)
