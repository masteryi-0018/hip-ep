# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for matmul/qdq_conv.h."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from golden_act import f32
from golden_float_cvt import int_f32
from golden_mac import add_q, fma_q, mac_exact
from golden_quant import (activate_f32, activate_out, dequant, kQdqActNone, requant,
                          requant_tail)


def conv_int(ifm, wts, sh, sw):
    kh, kw = wts.shape[:2]
    win = sliding_window_view(np.asarray(ifm, np.int64), (kh, kw), axis=(0, 1))[::sh, ::sw]
    return np.einsum("hwcyx,yxco->hwo", win, np.asarray(wts, np.int64), optimize=True)


def window_sum(ifm, kh, kw, sh, sw, k_opt=False, join=False):
    """fp32 channel sums added over the window one tap at a time: row-major, or (k_opt)
    column 0 + the other columns alternating between two accumulators. join: each
    channel sum is fp32(256 * sum(hi) + sum(lo))."""
    x = np.asarray(ifm, np.int64)
    ch = (fma_q(int_f32((x >> 8).sum(2)), 256.0, int_f32((x & 255).sum(2))) if join
          else int_f32(x.sum(2)))
    ho, wo = (ch.shape[0] - kh) // sh + 1, (ch.shape[1] - kw) // sw + 1

    def add(taps):
        acc = None
        for y, x in taps:
            tap = ch[y:y + sh * (ho - 1) + 1:sh, x:x + sw * (wo - 1) + 1:sw]
            acc = tap.copy() if acc is None else add_q(acc, tap)
        return acc

    if not k_opt or kw < 2:
        return add([(y, x) for y in range(kh) for x in range(kw)])
    first = kw % 2
    lo = add([(y, x) for y in range(kh) for x in range(first, kw, 2)])
    hi = add([(y, x) for y in range(kh) for x in range(first + 1, kw, 2)])
    if first:
        return add_q(add_q(add([(y, 0) for y in range(kh)]), lo), hi)
    return add_q(lo, hi)


def conv_qdq_acc(ifm, wts, sh, sw, c0, c1, c2, split16, k_opt=False, join=False):
    """The fp32 QDQ accumulator before the activation."""
    ifm = np.asarray(ifm, np.int64)
    if split16:
        lo, hi = conv_int(ifm & 255, wts, sh, sw), conv_int(ifm >> 8, wts, sh, sw)
        v = fma_q(mac_exact(hi, f32(256), int_f32(lo)), c2, c0)
    else:
        v = mac_exact(conv_int(ifm, wts, sh, sw), c2, c0)
    if c1 is not None:
        s = window_sum(ifm, wts.shape[0], wts.shape[1], sh, sw, k_opt, join)
        v = fma_q(s[..., None], c1, v)
    return v


def _conv_qdq(ifm, wts, out_t, sh, sw, k_opt, c0, c1, c2, split16, act, act0, act1, slope):
    v = conv_qdq_acc(ifm, wts, sh, sw, c0, c1, c2, split16, bool(k_opt))
    return activate_out(v, act, act0, act1, slope, out_t)


def conv_qdq_int8x8(ifm, wts, out_t, sh, sw, k_opt, c0, c1, c2, act=kQdqActNone, act0=0.0, act1=0.0,
                    slope=None):
    return _conv_qdq(ifm, wts, out_t, sh, sw, k_opt, c0, c1, c2, False, act, act0, act1, slope)


def conv_qdq_int16x8(ifm, wts, out_t, sh, sw, k_opt, c0, c1, c2, act=kQdqActNone, act0=0.0,
                     act1=0.0, slope=None):
    return _conv_qdq(ifm, wts, out_t, sh, sw, k_opt, c0, c1, c2, True, act, act0, act1, slope)


def _conv_qdq_add(ifm, wts, ifm2, out_t, sh, sw, k_opt, c0, c1, c2, add_c0, add_c1, add_c2,
                  act_before_add, act, act0, act1, split16):
    """conv_qdq accumulator, then the residual add (16-bit: channel sums joined from the byte
    planes)."""
    v = conv_qdq_acc(ifm, wts, sh, sw, c0, c1, c2, split16, bool(k_opt), split16)
    if act_before_add:
        v = activate_f32(v, act, act0, act1)
    v = mac_exact(np.asarray(ifm2, np.int64), add_c1, fma_q(v, add_c0, add_c2))
    return activate_out(v, kQdqActNone if act_before_add else act, act0, act1, 0.0, out_t)


def qdq_conv_add_a8w8(ifm, wts, ifm2, out_t, sh, sw, k_opt, c0, c1, c2, add_c0, add_c1, add_c2,
                      act_before_add=0, act=kQdqActNone, act0=0.0, act1=0.0):
    return _conv_qdq_add(ifm, wts, ifm2, out_t, sh, sw, k_opt, c0, c1, c2, add_c0, add_c1, add_c2,
                         act_before_add, act, act0, act1, False)


def qdq_conv_add_a16w8(ifm, wts, ifm2, out_t, sh, sw, k_opt, c0, c1, c2, add_c0, add_c1, add_c2,
                       act_before_add=0, act=kQdqActNone, act0=0.0, act1=0.0):
    return _conv_qdq_add(ifm, wts, ifm2, out_t, sh, sw, k_opt, c0, c1, c2, add_c0, add_c1, add_c2,
                         act_before_add, act, act0, act1, True)


def qdq_conv_q_add_a8w8(ifm, wts, ifm2, out_t, sh, sw, c0, c2, add_c0, add_c1, add_c2):
    q = requant_tail(conv_qdq_acc(ifm, wts, sh, sw, c0, None, c2, False), out_t)
    return requant(np.asarray(ifm2, np.int64), add_c1, dequant(q, add_c0, add_c2), out_t)


def dwc_int(ifm, wts, sh, sw):
    kh, kw = wts.shape[:2]
    win = sliding_window_view(np.asarray(ifm, np.int64), (kh, kw), axis=(0, 1))[::sh, ::sw]
    return np.einsum("hwcyx,yxc->hwc", win, np.asarray(wts, np.int64), optimize=True)


def qdq_dwc_int8x8(ifm, wts, out_t, sh, sw, zp_w, actv_sum, c0, c2, act=kQdqActNone, act0=0.0,
                   act1=0.0):
    ifm = np.asarray(ifm, np.int64)
    ones = np.ones(np.asarray(wts).shape, np.int64)
    acc = int_f32(dwc_int(ifm, wts, sh, sw))
    if actv_sum:
        acc = mac_exact(-zp_w * dwc_int(ifm, ones, sh, sw), f32(1), acc)
    return activate_out(fma_q(acc, c2, c0), act, act0, act1, 0.0, out_t)


def qdq_dwc_int16x8(ifm, wts, out_t, sh, sw, zp_w, actv_sum, c0, c2, act=kQdqActNone, act0=0.0,
                    act1=0.0):
    ifm = np.asarray(ifm, np.int64)
    ones = np.ones(np.asarray(wts).shape, np.int64)
    lo, hi = ifm & 255, ifm >> 8
    acc = mac_exact(dwc_int(hi, wts, sh, sw), f32(256), int_f32(dwc_int(lo, wts, sh, sw)))
    if actv_sum:
        acc = mac_exact(-zp_w * dwc_int(hi, ones, sh, sw), f32(256), acc)
        acc = mac_exact(-zp_w * dwc_int(lo, ones, sh, sw), f32(1), acc)
    return activate_out(fma_q(acc, c2, c0), act, act0, act1, 0.0, out_t)


def depthwise_conv_a8w8_s1(ifm, wts, out_t, zp_w, c0, c2, act=kQdqActNone, act0=0.0, act1=0.0):
    return qdq_dwc_int8x8(ifm, wts, out_t, 1, 1, zp_w, 1, c0, c2, act, act0, act1)


def depthwise_conv_a8w8_s2(ifm, wts, out_t, zp_w, c0, c2, act=kQdqActNone, act0=0.0, act1=0.0):
    return qdq_dwc_int8x8(ifm, wts, out_t, 2, 2, zp_w, 1, c0, c2, act, act0, act1)


def depthwise_conv_a16w8_s1(ifm, wts, out_t, zp_w, actv_sum, c0, c2, act=kQdqActNone, act0=0.0,
                            act1=0.0):
    return qdq_dwc_int16x8(ifm, wts, out_t, 1, 1, zp_w, actv_sum, c0, c2, act, act0, act1)


def depthwise_conv_a16w8_s2(ifm, wts, out_t, zp_w, actv_sum, c0, c2, act=kQdqActNone, act0=0.0,
                            act1=0.0):
    return qdq_dwc_int16x8(ifm, wts, out_t, 2, 2, zp_w, actv_sum, c0, c2, act, act0, act1)
