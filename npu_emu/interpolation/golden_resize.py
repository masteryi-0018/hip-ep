# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference model for resize.h, plus index / weight table builders."""

import math

import numpy as np

from golden_float_cvt import (bf16_to_f32, encode_float, f32_to_bf16, f32_to_fp8,
                              fp8_decode_identity, kFtzIn, kFtzOut, kSat, load_float, round_float,
                              store_float)
from golden_int_round import Rnd, float_to_int, saturate, shift_round
from golden_mac import fma, mac_exact, mul

F32 = np.float32
SEP, TAP4 = 1, 0
# Flush of the truncating 16-bit stores: bf16 flushes fp32-subnormal inputs, f16 results below
# its smallest normal.
TRUNC_FLAGS = {"bf16": kFtzIn, "f16": kFtzOut}


def _pix(x, y, xx):
    """x[y, xx] for broadcastable index arrays, clamped to the tensor."""
    Hi, Wi = x.shape[:2]
    y, xx = np.broadcast_arrays(np.clip(y, 0, Hi - 1), np.clip(xx, 0, Wi - 1))
    return x[y, xx]


def _taps4(ih, iw, s, sv_wo):
    """(y0, y1, x0, x1), each (Ho, Wo): x1 moves when s1 != 0 (per pixel); y1
    when s2 of the first column of the sv_wo block != 0."""
    Ho, Wo = s.shape[:2]
    sv_wo = sv_wo or Wo
    w0 = (np.arange(Wo) // sv_wo) * sv_wo
    y0 = np.broadcast_to(np.asarray(ih, np.int64)[:, None], (Ho, Wo))
    x0 = np.broadcast_to(np.asarray(iw, np.int64)[None, :], (Ho, Wo))
    y1 = y0 + (s[:, w0, 2] != 0)
    x1 = x0 + (s[:, :, 1] != 0)
    return y0, y1, x0, x1


def bilinear_int(x, ih, iw, wts, route, sv_wo, requantize, c0, c1, dtype):
    x = np.asarray(x).astype(np.int64)
    Ho, Wo = len(ih), len(iw)
    wts = np.asarray(wts, np.int64)
    ih = np.asarray(ih, np.int64)
    iw = np.asarray(iw, np.int64)
    if route == SEP:
        qy = wts[:2 * Ho].reshape(Ho, 2)
        qx = wts[2 * Ho:2 * Ho + 2 * Wo].reshape(Wo, 2)
        y1 = ih + (qy[:, 1] != 0)

        def tmp(cols):  # H pass at the given columns -> (Ho, Wo, C), int16 saturated
            v = (_pix(x, ih[:, None], cols[None, :]) * qy[:, 0, None, None]
                 + _pix(x, y1[:, None], cols[None, :]) * qy[:, 1, None, None])
            return saturate(v, "i16")

        acc = tmp(iw) * qx[None, :, 0, None] + tmp(iw + 1) * qx[None, :, 1, None]
        q = 14 if np.dtype(dtype) == np.uint8 else 15
    else:
        s = wts.reshape(Ho, Wo, 4)
        y0, y1, x0, x1 = _taps4(ih, iw, s, sv_wo)
        acc = (_pix(x, y0, x0) * s[:, :, 0, None] + _pix(x, y0, x1) * s[:, :, 1, None]
               + _pix(x, y1, x0) * s[:, :, 2, None] + _pix(x, y1, x1) * s[:, :, 3, None])
        q = 15
    t = np.dtype(dtype).kind + str(8 * np.dtype(dtype).itemsize)  # "i8", "u16", ...
    if requantize:
        c1q = F32(F32(c1) * F32(1 << q))
        out = float_to_int(mac_exact(acc, F32(c0), c1q), t, -q, Rnd.kTrunc)
    else:
        out = saturate(shift_round(acc, q, Rnd.kTrunc), t)
    return out.astype(dtype).reshape(Ho, Wo, -1)


def _bilinear_acc(xf, ih, iw, s, sv_wo):
    """4-tap sum in fp32: rounded product, then three fused MACs."""
    y0, y1, x0, x1 = _taps4(ih, iw, s, sv_wo)
    acc = mul(_pix(xf, y0, x0), s[:, :, 0, None])
    acc = fma(_pix(xf, y0, x1), s[:, :, 1, None], acc)
    acc = fma(_pix(xf, y1, x0), s[:, :, 2, None], acc)
    return fma(_pix(xf, y1, x1), s[:, :, 3, None], acc)


def bilinear_bf16(xbits, ih, iw, sbits, sv_wo, t):
    xf = load_float(xbits, t)
    s = load_float(sbits, t)
    acc = _bilinear_acc(xf, ih, iw, s, sv_wo)
    return encode_float(acc, t, Rnd.kTrunc, TRUNC_FLAGS[t]).astype(np.uint16)


def bilinear_fp8(codes, ih, iw, sbits, sv_wo, t, dq, oq):
    dec = fp8_decode_identity(codes, row=codes.shape[-1])
    xf = round_float(mul(dec, F32(dq)), t, Rnd.kTrunc, TRUNC_FLAGS[t])
    s = load_float(sbits, t)
    acc = _bilinear_acc(xf, ih, iw, s, sv_wo)
    return f32_to_fp8(mul(acc, F32(oq)), Rnd.kTrunc, kSat)


def cubic_bf16(xbits, ih, iw, cy_bits, cx_bits):
    """Pass 1 over H rounded to bf16, then pass 2 over W."""
    xf = bf16_to_f32(xbits)
    Hi, Wi, C = xf.shape
    cy = bf16_to_f32(cy_bits).reshape(-1, 4)
    cx = bf16_to_f32(cx_bits).reshape(-1, 4)
    ih = np.asarray(ih, np.int64)
    iw = np.asarray(iw, np.int64)
    rows = lambda k: xf[np.clip(ih + k, 0, Hi - 1)]  # (Ho, Wi, C)
    acc = mul(rows(0), cy[:, 0, None, None])
    for k in range(1, 4):
        acc = fma(rows(k), cy[:, k, None, None], acc)
    t = round_float(acc, "bf16", Rnd.kHalfEven, kFtzIn)  # (Ho, Wi, C)
    cols = lambda k: t[:, np.clip(iw + k, 0, Wi - 1)]  # (Ho, Wo, C)
    acc = mul(cols(0), cx[None, :, 0, None])
    for k in range(1, 4):
        acc = fma(cols(k), cx[None, :, k, None], acc)
    return f32_to_bf16(acc, Rnd.kHalfEven, kFtzIn)


# ---- index / weight table builders ----

def coord(o, n_in, n_out, mode):
    """Unclamped ONNX source coordinate of output index o."""
    if mode == "align_corners":
        return 0.0 if n_out <= 1 else o * (n_in - 1) / (n_out - 1)
    scale = n_out / n_in
    return (o + 0.5) / scale - 0.5 if mode == "half_pixel" else o / scale


def bilinear_axis(n_in, n_out, mode):
    """Per output: (floor index, w0, w1), coordinate clamped to [0, n_in - 1]."""
    idx, w0, w1 = [], [], []
    for o in range(n_out):
        v = F32(min(max(0.0, coord(o, n_in, n_out, mode)), n_in - 1))
        f = int(v)
        a = F32(v - F32(f))
        idx.append(f)
        w0.append(F32(1.0) - a)
        w1.append(a)
    return np.array(idx, np.int32), np.array(w0, F32), np.array(w1, F32)


def q15_table(wy0, wy1, wx0, wx1):
    """4-tap Q15 table [Ho][Wo][4], truncating."""
    sy = np.stack([wy0, wy1], 1)[:, None, :, None]
    sx = np.stack([wx0, wx1], 1)[None, :, None, :]
    p = (sy * sx).astype(F32).reshape(len(wy0), len(wx0), 4)
    return (p * F32(1 << 15)).astype(np.uint16)


def sep_table(wy0, wy1, wx0, wx1, qh):
    """Separable qy (Q_H) / qx (Q7) tables, rounding half up."""
    q = lambda w, b: (np.asarray(w, F32) * F32(1 << b) + F32(0.5)).astype(np.uint16)
    qy = np.stack([q(wy0, qh), q(wy1, qh)], 1)
    qx = np.stack([q(wx0, 7), q(wx1, 7)], 1)
    return np.concatenate([qy.reshape(-1), qx.reshape(-1)]).astype(np.uint16)


def float_table(wy0, wy1, wx0, wx1, t):
    """bf16 / f16 table bits [Ho][Wo][4] (products in fp32, then RNE)."""
    sy = np.stack([wy0, wy1], 1)[:, None, :, None]
    sx = np.stack([wx0, wx1], 1)[None, :, None, :]
    p = (sy * sx).astype(F32).reshape(len(wy0), len(wx0), 4)
    return store_float(p, t).astype(np.uint16)


def cubic_axis(n_in, n_out, mode, a=-0.75):
    """Cubic window starts and weights (bf16 bits), edge clamping folded in."""
    idx = np.zeros(n_out, np.int32)
    coef = np.zeros((n_out, 4))
    for o in range(n_out):
        src = coord(o, n_in, n_out, mode)
        f = math.floor(src)
        s = src - f
        w = [a * s ** 3 - 2 * a * s ** 2 + a * s, (a + 2) * s ** 3 - (a + 3) * s ** 2 + 1.0,
             -(a + 2) * s ** 3 + (2 * a + 3) * s ** 2 - a * s, -a * s ** 3 + a * s ** 2]
        b = f - 1
        bc = min(max(b, 0), n_in - 4)
        for k in range(4):
            coef[o, min(max(b + k, 0), n_in - 1) - bc] += w[k]
        idx[o] = bc
    return idx, f32_to_bf16(coef.astype(F32))
