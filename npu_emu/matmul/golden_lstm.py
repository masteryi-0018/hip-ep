# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the integer QDQ LSTM step and grpb."""

import numpy as np

from golden_float_cvt import f32_to_bf16, f32_to_f16, kFtzIn, round_float
from golden_int_round import Rnd, clamp64, float_to_int, int_to_float, shift_round, wrap
from golden_mac import add, fma, mac_exact, mul
from golden_nlf import hw_exp2_bf20, tanh_acc

F32, I64, U64 = np.float32, np.int64, np.uint64
I32_MIN, I32_MAX, U32_MAX = -(1 << 31), (1 << 31) - 1, (1 << 32) - 1


def lstm_sigmoid(v):
    t = tanh_acc(mul(v, F32(-0.5)), "bf16")
    with np.errstate(all="ignore"):
        return round_float((F32(0.5) - t * F32(0.5)).astype(F32), "bf16")


def sbyte(w, k):
    return int(np.int8(np.uint8((int(w) >> (8 * k)) & 0xFF)))


def parse(q, a16):
    q = [int(v) for v in q]
    p = dict(qx=q[0], qh=q[1], xw=sbyte(q[2], 0), xs=sbyte(q[2], 1), hr=sbyte(q[2], 2), hs=sbyte(q[2], 3),
             xwq=sbyte(q[3], 0), hrq=sbyte(q[3], 1), lin=sbyte(q[3], 2), c0s=sbyte(q[3], 3),
             c1s=sbyte(q[4], 0), c2s=sbyte(q[4], 1), nl=sbyte(q[5], 2),
             c_scale=int(np.int16(np.uint16(q[5] & 0xFFFF))),
             oh_scale=int(np.int16(np.uint16(q[6] & 0xFFFF))), oc_scale=int(np.int16(np.uint16(q[7] & 0xFFFF))),
             oh_s=sbyte(q[8], 0), oc_s=sbyte(q[8], 1))
    if a16:
        p.update(c_zp=int(np.int16(np.uint16(q[4] >> 16))), oh_zp=q[6] >> 16, oc_zp=q[7] >> 16)
    else:
        p.update(c_zp=sbyte(q[4], 2), oh_zp=sbyte(q[6], 2), oc_zp=sbyte(q[7], 2))
    return p


def header(a16, p):
    """The 9 QDQ words."""
    def b(*v):
        return sum((int(x) & 0xFF) << (8 * k) for k, x in enumerate(v))
    zp16 = (lambda v: int(v) & 0xFFFF) if a16 else (lambda v: int(v) & 0xFF)
    return np.array([p["qx"], p["qh"], b(p["xw"], p["xs"], p["hr"], p["hs"]),
                     b(p["xwq"], p["hrq"], p["lin"], p["c0s"]),
                     b(p["c1s"], p["c2s"]) | (zp16(p["c_zp"]) << 16),
                     (p["c_scale"] & 0xFFFF) | ((p["nl"] & 0xFF) << 16),
                     (p["oh_scale"] & 0xFFFF) | (zp16(p["oh_zp"]) << 16),
                     (p["oc_scale"] & 0xFFFF) | (zp16(p["oc_zp"]) << 16),
                     b(p["oh_s"], p["oc_s"])], np.uint32)


def lstm_gates(x, h, w, r, b, qdq, a16):
    """The 4 x len_h gate inputs (fp32)."""
    p = parse(qdq, a16)
    lo, hi = (0, U32_MAX) if a16 else (I32_MIN, I32_MAX)
    x, h = (np.asarray(v).astype(I64) for v in (x, h))
    lx, lh = x.size, h.size
    w = np.asarray(w).astype(I64).reshape(4 * lh, lx)
    r = np.asarray(r).astype(I64).reshape(4 * lh, lh)
    b = np.asarray(b).astype(I64).reshape(4 * lh)

    ev = Rnd.kHalfEven
    sx = clamp64(shift_round(x.sum(), p["xs"], ev), lo, hi)
    sh = clamp64(shift_round(h.sum(), p["hs"], ev), lo, hi)
    dx = clamp64(shift_round(w @ x, p["xw"], ev), lo, hi)
    dh = clamp64(shift_round(r @ h, p["hr"], ev), lo, hi)
    dx = (dx.astype(U64) * U64(p["qx"])).astype(I64)  # 64-bit products, wrapping
    dh = (dh.astype(U64) * U64(p["qh"])).astype(I64)
    xw = clamp64(shift_round(dx, p["xwq"], ev), lo, hi)
    hr = clamp64(shift_round(dh, p["hrq"], ev), lo, hi)
    lin = wrap(xw + hr + b - (sx + sh), "i32")
    return int_to_float(lin, -p["lin"], Rnd.kHalfEven)


def lstm(x, h, c, w, r, b, qdq, a16):
    """One step; returns (h_out, c_out) as int64 arrays."""
    p = parse(qdq, a16)
    lh = np.asarray(h).size
    c = np.asarray(c).astype(I64)
    y = lstm_gates(x, h, w, r, b, qdq, a16)
    sig, th = lstm_sigmoid(y), tanh_acc(y, "bf16")
    vi, vo, vf = sig[:lh], sig[lh:2 * lh], sig[2 * lh:3 * lh]
    vg = th[3 * lh:]
    olo, ohi = (0, 65535) if a16 else (-128, 127)
    d = shift_round(wrap(c - p["c_zp"], "i16" if a16 else "i8"), -p["c0s"], Rnd.kFloor)
    cf = int_to_float(clamp64(shift_round(d * p["c_scale"], p["c1s"], Rnd.kHalfEven), I32_MIN, I32_MAX),
                      -p["c2s"], Rnd.kHalfEven)
    C = fma(vf, vg, mul(vi, cf))
    H = mul(tanh_acc(C, "bf16"), vo)

    def quant(v, scale, zp, s):
        q = shift_round(zp, -s, Rnd.kFloor) + scale * float_to_int(v, "i32", p["nl"])
        return clamp64(shift_round(q, s, Rnd.kHalfEven), olo, ohi)
    return quant(H, p["oh_scale"], p["oh_zp"], p["oh_s"]), quant(C, p["oc_scale"], p["oc_zp"], p["oc_s"])


K0, K1, K2 = F32(-1 + 895.001), F32(0.8083), F32(-0.1084)


def grpb_sigmoid(x):
    x = np.ascontiguousarray(x, F32)
    xb = x.view(np.uint32)
    xa = (xb | np.uint32(0x80000000)).view(F32)
    flip = (xb | np.uint32(0x80000000)) ^ xb
    p01 = fma(np.full_like(x, K1), xa, np.full_like(x, K0))
    p012 = fma(mul(x, x), np.full_like(x, K2), p01)
    e = hw_exp2_bf20(p012).view(np.uint32)
    return add(np.where(flip != 0, F32(1), F32(0)), (e | flip).view(F32))


def grpb_preact(ifm, a16, wts, c0, c1, c2):
    """The M x 2 gate pre-activations (fp32)."""
    a = np.asarray(ifm).astype(np.int64)
    w = np.asarray(wts).astype(np.int64)
    ws = np.stack([w[:, :4].sum(1), w[:, 4:].sum(1)], 1)          # K x 2
    c0, c2 = (np.asarray(v, F32) for v in (c0, c2))

    def wrap32(v):
        return ((v + (1 << 31)) % (1 << 32)) - (1 << 31)
    if a16:
        ah, al = a >> 8, a & 0xFF
        gh, gl = wrap32(ah @ ws), wrap32(al @ ws)
        gm = mac_exact(gh, F32(256), mac_exact(gl, F32(1), F32(0)))
        x = fma(gm, c2[:, None], c0[:, None])
        if c1 is not None:
            sm = mac_exact(ah.sum(1), F32(256), mac_exact(al.sum(1), F32(1), F32(0)))
            x = fma(np.asarray(c1, F32)[:, None], sm[:, None], x)
    else:
        gm = wrap32(a @ ws)
        x = np.stack([mac_exact(gm[:, k], c2, c0) for k in range(2)], 1)
        if c1 is not None:
            s = a.sum(1)
            x = np.stack([mac_exact(s, np.asarray(c1, F32), x[:, k]) for k in range(2)], 1)
    return np.asarray(x, F32)


def grpb(ifm, a16, wts, c0, c1, c2, head, lyr0, lyr1, f16):
    """ifm M x K, wts K x 8; returns the M output codes (uint16)."""
    s = grpb_sigmoid(grpb_preact(ifm, a16, wts, c0, c1, c2))
    rr = fma(-s[:, 1], F32(head), F32(lyr0))
    rr = fma(s[:, 0], rr, F32(lyr1))
    return (f32_to_f16 if f16 else f32_to_bf16)(rr, Rnd.kHalfEven, kFtzIn)
