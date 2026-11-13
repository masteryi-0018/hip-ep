# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the arith_int group of npu_emu
(include/npu_emu_arith_int.h): the kernels' chains of one-rounding fp32 MACs
(integer operands exact), the activations and the rounding / saturating
conversion, per route. Follows mllib's corrected python models (fp32_exact /
QDQ_*Wrapper.py), written independently of the HIP code. Tensors are logical:
flat, or HWC with numpy broadcasting of ifm2 / ifm3."""

import numpy as np

import golden as g

f32 = np.float32
ACT_NONE, ACT_RELU, ACT_CLIP, ACT_LRELU, ACT_PRELU = 0, 1, 2, 3, 4


def mac(x, c, a):
    """x * c + a rounded once to fp32, integer x exact; broadcasting."""
    x, c, a = np.broadcast_arrays(np.asarray(x, np.int64), np.asarray(c, f32), np.asarray(a, f32))
    return g.mac_int(x.reshape(-1), c.reshape(-1), a.reshape(-1)).reshape(x.shape)


def fma(x, c, a):
    """fp32 x * c + a, one rounding; broadcasting."""
    x, c, a = np.broadcast_arrays(np.asarray(x, f32), np.asarray(c, f32), np.asarray(a, f32))
    return g.fma32(x.reshape(-1), c.reshape(-1), a.reshape(-1)).reshape(x.shape)


def out_int(v, t):
    lo, hi = g.INT_RANGE[t]
    return g.rne_int(v, lo, hi)


def relu_variants(v, act, a0, a1):
    if act == ACT_CLIP:
        return np.minimum(np.maximum(v, f32(a0)), f32(a1))
    if act == ACT_LRELU:
        return np.maximum(v, fma(v, a0, a1))
    return v


def i64(x):
    return np.asarray(x).astype(np.int64)


# ---- flat / Broadcasting3d plain add, mul ---------------------------------------

def add(a, b, out_t, c0, c1, c2, act=ACT_NONE, a0=0.0, a1=0.0):
    """ifm2 b broadcast onto a (numpy rules)."""
    a, b = i64(a), np.broadcast_to(i64(b), np.shape(a))
    return out_int(relu_variants(mac(b, c1, mac(a, c0, c2)), act, a0, a1), out_t)


def mul(a, b, out_t, c0, c1, c2, c3, act=ACT_NONE, a0=0.0, a1=0.0):
    a, b = i64(a), np.broadcast_to(i64(b), np.shape(a))
    v = mac(a, mac(b, c0, c1), mac(b, c2, c3))
    return out_int(relu_variants(v, act, a0, a1), out_t)


# ---- Xint8 ------------------------------------------------------------------------

def srs(v, s):
    """Integer v / 2^s rounded half to even (s >= 0)."""
    v = np.asarray(v, np.int64)
    if s == 0:
        return v
    q = v >> s
    r = v - (q << s)
    h = 1 << (s - 1)
    return q + ((r > h) | ((r == h) & ((q & 1) == 1)))


def add_fused_xint(a, b, c0, c1, act=ACT_NONE, a0=0.0, a1=0.0):
    e0, e1 = (int(np.frexp(f32(c))[1]) - 1 for c in (c0, c1))
    base = min(e0, e1, 0)
    v = (i64(a) << (e0 - base)) + (i64(b) << (e1 - base))
    q = np.clip(srs(v, -base), -128, 127)
    if act == ACT_CLIP:
        q = np.clip(q, out_int(f32(a0), "i8"), out_int(f32(a1), "i8"))
    return q


def mac_xint(x1, x2, x3, out_t, A, v, p, t):
    x1 = i64(x1)
    x2 = np.broadcast_to(i64(x2), x1.shape)
    x3 = np.broadcast_to(i64(x3), x1.shape)
    w = srs(x2 * A, t)
    w = np.clip(w, -32768, 32767)  # the vector pre-scale saturates
    lo, hi = g.INT_RANGE[out_t]
    return np.clip(srs(x1 * w + x3 * v, p), lo, hi)


def mac_xint_scalar(x1, x2, x3, out_t, A, v, p, t):
    """ifm2 a single value: the kernel's scalar path wraps w to int16."""
    x1 = i64(x1)
    w = int(srs(np.array([int(x2) * A]), t)[0])
    w = (w + 32768) % 65536 - 32768
    x3 = np.broadcast_to(i64(x3), x1.shape)
    lo, hi = g.INT_RANGE[out_t]
    return np.clip(srs(x1 * w + x3 * v, p), lo, hi)


# ---- Broadcasting3d fused -----------------------------------------------------------

def _hwc(a, b):
    a = i64(a)
    return a, np.broadcast_to(i64(b), a.shape), np.shape(b)[2] == 1


def addsub_fused(a, b, out_t, k, act, slope=None):
    """k: (c0, c1, c2, w3, w4, w5, w6, w7) as fp32."""
    c0, c1, c2, w3, w4, w5, w6, w7 = (f32(v) for v in k)
    a, b, bcc = _hwc(a, b)
    if bcc:
        icpt = mac(b, c1, c2)
        r = mac(a, c0, icpt)
    else:
        r = mac(b, c1, mac(a, c0, c2))
    if act == ACT_RELU:
        v = np.maximum(r, w3)
    elif act == ACT_CLIP:
        v = np.minimum(np.maximum(r, w3), w4)
    elif act == ACT_LRELU:
        l = mac(a, w5, fma(icpt, w3, w4)) if bcc else mac(b, w6, mac(a, w5, w7))
        v = np.maximum(r, l)
    elif act == ACT_PRELU:
        s = np.broadcast_to(np.asarray(slope, f32), r.shape)
        v = fma(r, np.where(r < 0, s, f32(1)), w3)
    else:
        v = r
    return out_int(v, out_t)


def mul_fused(a, b, out_t, k, act, a16, neg_scale=None):
    """k: (c0, c1, c2, c3, act0, act1, z1, zy, z2)."""
    c0, c1, c2, c3, a0, a1 = (f32(v) for v in k[:6])
    z1, zy, z2 = int(k[6]), f32(k[7]), int(k[8])
    a, b, bcc = _hwc(a, b)
    r1, r2 = a - z1, b - z2
    neg = (r1 < 0) ^ (r2 < 0)
    F = mac(b, c0, c1)

    def horner():
        return mac(a, F, mac(b, c2, c3))

    def shifted():
        return mac(r1, F, zy)

    if act == ACT_RELU:
        v = np.maximum(horner() if bcc else shifted(), a0)
    elif act == ACT_CLIP:
        v = np.minimum(np.maximum(horner(), a0), a1)
    elif act == ACT_LRELU and not a16:
        p = r1 * r2
        v = mac(p, np.where(neg, c1, c0), zy) if bcc else mac(np.abs(p), c3, mac(p, c2, zy))
    elif act == ACT_LRELU:
        if bcc:
            v = np.maximum(shifted(), mac(r1, (F * a0).astype(f32), zy))
        else:
            h = horner()
            v = np.maximum(h, fma(h, a0, a1))
    elif act == ACT_PRELU:
        s = np.where(neg, np.broadcast_to(np.asarray(neg_scale, f32), a.shape), c0)
        v = mac(r1, mac(r2, s, f32(0)), zy)
    else:
        v = shifted()
    return out_int(v, out_t)


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
        v = mac(a, mac(b, c0, c1), mac(c, c4, mac(b, c2, c3e)))
    else:
        f = mac(b, c0, c1)
        w = fma(c.astype(f32), c4, c3) if (bcc and addend_to_float) else mac(c, c4, c3)
        v = mac(a - zp1, f, w)
        if act == ACT_LRELU:
            v = np.maximum(v, fma(v, a0, a1))
    q = out_int(v, out_t)
    if act in (ACT_RELU, ACT_CLIP):
        q = np.maximum(q, out_int(a0, out_t))
        if act == ACT_CLIP:
            q = np.minimum(q, out_int(a1, out_t))
    return q
