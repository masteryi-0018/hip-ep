# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the integer / QDQ elementwise arithmetic; tensors flat or HWC."""

import numpy as np

import golden_act as qa
from golden_int_round import INT_RANGE, Rnd, float_to_int, sat_add64, shift_round
from golden_mac import fma, mac_exact
from golden_minmax import vclamp, vmax

f32 = np.float32
ACT_NONE, ACT_RELU, ACT_CLIP, ACT_LRELU, ACT_PRELU = 0, 1, 2, 3, 4


def relu_variants(v, act, a0, a1):
    act = qa.ACT_CLIP if act == ACT_CLIP else qa.ACT_LRELU if act == ACT_LRELU else qa.ACT_NONE
    return qa.activate_f32(v, act, a0, a1)


def i64(x):
    return np.asarray(x).astype(np.int64)


def add(a, b, out_t, c0, c1, c2, act=ACT_NONE, a0=0.0, a1=0.0):
    """ifm2 b broadcast onto a (numpy rules)."""
    a, b = i64(a), np.broadcast_to(i64(b), np.shape(a))
    return float_to_int(relu_variants(mac_exact(b, c1, mac_exact(a, c0, c2)), act, a0, a1), out_t)


def mul(a, b, out_t, c0, c1, c2, c3, act=ACT_NONE, a0=0.0, a1=0.0):
    a, b = i64(a), np.broadcast_to(i64(b), np.shape(a))
    v = mac_exact(a, mac_exact(b, c0, c1), mac_exact(b, c2, c3))
    return float_to_int(relu_variants(v, act, a0, a1), out_t)


def add_fused_xint(a, b, c0, c1, act=ACT_NONE, a0=0.0, a1=0.0):
    e0, e1 = (int(np.frexp(f32(c))[1]) - 1 for c in (c0, c1))
    base = min(e0, e1, 0)
    # [default] shifts of 64 or more saturate instead of wrapping or being undefined
    v = sat_add64(shift_round(i64(a), -(e0 - base), Rnd.kHalfEven),
                  shift_round(i64(b), -(e1 - base), Rnd.kHalfEven))
    q = np.clip(shift_round(v, -base, Rnd.kHalfEven), -128, 127)
    if act == ACT_CLIP:
        q = np.clip(q, float_to_int(f32(a0), "i8"), float_to_int(f32(a1), "i8"))
    return q


def mac_xint(x1, x2, x3, out_t, A, v, p, t):
    x1 = i64(x1)
    x2 = np.broadcast_to(i64(x2), x1.shape)
    x3 = np.broadcast_to(i64(x3), x1.shape)
    w = shift_round(x2 * A, t, Rnd.kHalfEven)
    w = np.clip(w, -32768, 32767)  # saturates to int16
    lo, hi = INT_RANGE[out_t]
    return np.clip(shift_round(x1 * w + x3 * v, p, Rnd.kHalfEven), lo, hi)


def mac_xint_scalar(x1, x2, x3, out_t, A, v, p, t):
    """ifm2 a single value: w wraps to int16."""
    x1 = i64(x1)
    w = int(shift_round(np.array([int(x2) * A]), t, Rnd.kHalfEven)[0])
    w = (w + 32768) % 65536 - 32768
    x3 = np.broadcast_to(i64(x3), x1.shape)
    lo, hi = INT_RANGE[out_t]
    return np.clip(shift_round(x1 * w + x3 * v, p, Rnd.kHalfEven), lo, hi)


def _hwc(a, b):
    a = i64(a)
    return a, np.broadcast_to(i64(b), a.shape), np.shape(b)[2] == 1


def addsub_fused(a, b, out_t, k, act, slope=None):
    """k: (c0, c1, c2, w3, w4, w5, w6, w7) as fp32."""
    c0, c1, c2, w3, w4, w5, w6, w7 = (f32(v) for v in k)
    a, b, bcc = _hwc(a, b)
    if bcc:
        icpt = mac_exact(b, c1, c2)
        r = mac_exact(a, c0, icpt)
    else:
        r = mac_exact(b, c1, mac_exact(a, c0, c2))
    if act == ACT_RELU:
        v = vmax(r, w3)
    elif act == ACT_CLIP:
        v = vclamp(r, w3, w4)
    elif act == ACT_LRELU:
        l = mac_exact(a, w5, fma(icpt, w3, w4)) if bcc else mac_exact(b, w6, mac_exact(a, w5, w7))
        v = vmax(r, l)
    elif act == ACT_PRELU:
        s = np.broadcast_to(np.asarray(slope, f32), r.shape)
        v = fma(r, np.where(r < 0, s, f32(1)), w3)
    else:
        v = r
    return float_to_int(v, out_t)


def mul_fused(a, b, out_t, k, act, a16, neg_scale=None):
    """k: (c0, c1, c2, c3, act0, act1, z1, zy, z2)."""
    c0, c1, c2, c3, a0, a1 = (f32(v) for v in k[:6])
    z1, zy, z2 = int(k[6]), f32(k[7]), int(k[8])
    a, b, bcc = _hwc(a, b)
    r1, r2 = a - z1, b - z2
    neg = (r1 < 0) ^ (r2 < 0)
    F = mac_exact(b, c0, c1)

    def horner():
        return mac_exact(a, F, mac_exact(b, c2, c3))

    def shifted():
        return mac_exact(r1, F, zy)

    if act == ACT_RELU:
        v = vmax(horner() if bcc else shifted(), a0)
    elif act == ACT_CLIP:
        v = vclamp(horner(), a0, a1)
    elif act == ACT_LRELU and not a16:
        p = r1 * r2
        if bcc:
            v = mac_exact(p, np.where(neg, c1, c0), zy)
        else:
            v = mac_exact(np.abs(p), c3, mac_exact(p, c2, zy))
    elif act == ACT_LRELU:
        if bcc:
            v = vmax(shifted(), mac_exact(r1, (F * a0).astype(f32), zy))
        else:
            h = horner()
            v = vmax(h, fma(h, a0, a1))
    elif act == ACT_PRELU:
        s = np.where(neg, np.broadcast_to(np.asarray(neg_scale, f32), a.shape), c0)
        v = mac_exact(r1, mac_exact(r2, s, f32(0)), zy)
    else:
        v = shifted()
    return float_to_int(v, out_t)


def mac_fused(a, b, c, out_t, k, act, addend_to_float):
    """k: (c0, c1, c2, c3, c4, act0, act1, zp1)."""
    c0, c1, c2, c3, c4, a0, a1 = (f32(v) for v in k[:7])
    zp1 = int(k[7])
    a = i64(a)
    bcc = np.shape(b)[2] == 1
    b = np.broadcast_to(i64(b), a.shape)
    c = np.broadcast_to(i64(c), a.shape)
    if bcc and act == ACT_NONE:
        c3e = fma(f32(zp1), -c1, c3)
        v = mac_exact(a, mac_exact(b, c0, c1), mac_exact(c, c4, mac_exact(b, c2, c3e)))
    else:
        f = mac_exact(b, c0, c1)
        w = fma(c.astype(f32), c4, c3) if (bcc and addend_to_float) else mac_exact(c, c4, c3)
        v = mac_exact(a - zp1, f, w)
        if act == ACT_LRELU:
            v = vmax(v, fma(v, a0, a1))
    q = float_to_int(v, out_t)
    if act in (ACT_RELU, ACT_CLIP):
        q = np.maximum(q, float_to_int(a0, out_t))
        if act == ACT_CLIP:
            q = np.minimum(q, float_to_int(a1, out_t))
    return q
