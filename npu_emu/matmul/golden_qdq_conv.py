# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for matmul/qdq_conv.h."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from golden_mac import fma, mac_exact
from golden_act import (ACT_NONE, activate_f32,
                            activate_out, f32)


def conv_int(ifm, wts, sh, sw):
    kh, kw = wts.shape[:2]
    win = sliding_window_view(np.asarray(ifm, np.int64), (kh, kw), axis=(0, 1))[::sh, ::sw]
    return np.einsum("hwcyx,yxco->hwo", win, np.asarray(wts, np.int64), optimize=True)


def window_sum(ifm, kh, kw, sh, sw, k_opt=False, join=False):
    """fp32 channel sums added over the window one tap at a time: row-major, or (k_opt)
    column 0 + the other columns alternating between two accumulators. join: each
    channel sum is fp32(256 * sum(hi) + sum(lo))."""
    x = np.asarray(ifm, np.int64)
    ch = fma((x >> 8).sum(2).astype(f32), 256.0, (x & 255).sum(2).astype(f32)) if join \
        else x.sum(2).astype(f32)
    ho, wo = (ch.shape[0] - kh) // sh + 1, (ch.shape[1] - kw) // sw + 1

    def add(taps):
        acc = None
        for y, x in taps:
            tap = ch[y:y + sh * (ho - 1) + 1:sh, x:x + sw * (wo - 1) + 1:sw]
            acc = tap.copy() if acc is None else (acc + tap).astype(f32)
        return acc

    if not k_opt or kw < 2:
        return add([(y, x) for y in range(kh) for x in range(kw)])
    first = kw % 2
    lo = add([(y, x) for y in range(kh) for x in range(first, kw, 2)])
    hi = add([(y, x) for y in range(kh) for x in range(first + 1, kw, 2)])
    if first:
        return ((add([(y, 0) for y in range(kh)]) + lo).astype(f32) + hi).astype(f32)
    return (lo + hi).astype(f32)


def conv_qdq_acc(ifm, wts, sh, sw, c0, c1, c2, split16, k_opt=False, join=False):
    """The fp32 QDQ accumulator before the activation."""
    ifm = np.asarray(ifm, np.int64)
    if split16:
        lo, hi = conv_int(ifm & 255, wts, sh, sw), conv_int(ifm >> 8, wts, sh, sw)
        v = fma(mac_exact(hi, f32(256), lo.astype(f32)), c2, c0)
    else:
        v = mac_exact(conv_int(ifm, wts, sh, sw), c2, c0)
    if c1 is not None:
        s = window_sum(ifm, wts.shape[0], wts.shape[1], sh, sw, k_opt, join)
        v = fma(s[..., None], c1, v)
    return v


def conv_qdq(ifm, wts, out_t, sh, sw, c0, c1, c2, split16, act=ACT_NONE, act0=0.0, act1=0.0,
             slope=None, k_opt=False):
    v = conv_qdq_acc(ifm, wts, sh, sw, c0, c1, c2, split16, k_opt)
    return activate_out(v, act, act0, act1, slope, out_t)


def conv_qdq_add(ifm, wts, ifm2, out_t, sh, sw, c0, c1, c2, split16, act, act0, act1,
                 act_before_add, add_c0, add_c1, add_c2, k_opt=False):
    """conv_qdq accumulator, then the residual add (16-bit: channel sums joined from
    the byte planes, row-major)."""
    v = conv_qdq_acc(ifm, wts, sh, sw, c0, c1, c2, split16, k_opt and not split16, split16)
    if act_before_add:
        v = activate_f32(v, act, act0, act1)
    v = mac_exact(np.asarray(ifm2, np.int64), add_c1, fma(v, add_c0, add_c2))
    if not act_before_add:
        v = activate_f32(v, act, act0, act1)
    return activate_out(v, ACT_NONE, 0.0, 0.0, 0.0, out_t)


def conv_qdq_q_add(ifm, wts, ifm2, out_t, sh, sw, c0, c2, add_c0, add_c1, add_c2):
    q = activate_out(conv_qdq_acc(ifm, wts, sh, sw, c0, None, c2, False), ACT_NONE, 0, 0, 0.0,
                     out_t)
    v = mac_exact(np.asarray(ifm2, np.int64), add_c1, mac_exact(q, add_c0, add_c2))
    return activate_out(v, ACT_NONE, 0.0, 0.0, 0.0, out_t)


def dwc_int(ifm, wts, sh, sw):
    kh, kw = wts.shape[:2]
    win = sliding_window_view(np.asarray(ifm, np.int64), (kh, kw), axis=(0, 1))[::sh, ::sw]
    return np.einsum("hwcyx,yxc->hwc", win, np.asarray(wts, np.int64), optimize=True)


def qdq_dwc(ifm, wts, out_t, sh, sw, zp_w, actv_sum, c0, c2, act=ACT_NONE, act0=0.0, act1=0.0):
    ifm = np.asarray(ifm, np.int64)
    ones = np.ones(np.asarray(wts).shape, np.int64)
    acc = mac_exact(dwc_int(ifm, wts, sh, sw), f32(1), f32(0))
    if actv_sum:
        acc = mac_exact(-zp_w * dwc_int(ifm, ones, sh, sw), f32(1), acc)
    return activate_out(activate_f32(fma(acc, c2, c0), act, act0, act1), ACT_NONE, 0, 0, 0.0, out_t)


def qdq_dwc16(ifm, wts, out_t, sh, sw, zp_w, actv_sum, c0, c2, act=ACT_NONE, act0=0.0, act1=0.0):
    ifm = np.asarray(ifm, np.int64)
    ones = np.ones(np.asarray(wts).shape, np.int64)
    lo, hi = ifm & 255, ifm >> 8
    acc = mac_exact(dwc_int(hi, wts, sh, sw), f32(256), mac_exact(dwc_int(lo, wts, sh, sw), f32(1), f32(0)))
    if actv_sum:
        acc = mac_exact(-zp_w * dwc_int(hi, ones, sh, sw), f32(256), acc)
        acc = mac_exact(-zp_w * dwc_int(lo, ones, sh, sw), f32(1), acc)
    return activate_out(activate_f32(fma(acc, c2, c0), act, act0, act1), ACT_NONE, 0, 0, 0.0, out_t)
