# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the resize entry points of npu_emu (bilinear A8 / A16 /
bf16 / fp8 and bicubic bf16), written from the AIE4 kernel sources and mllib's
corrected python models (ResizeBilinearA8Wrapper.resize_linear_model,
ResizeBilinearBf16 / Fp8Wrapper, ResizeCubicBf16Wrapper.resize_cubic_model),
which match aiesim bit for bit. Logical HWC tensors; the index tables are
absolute and the weights are the words of the wts blob (see
include/npu_emu_resize.h).

Also the host-side table builders (the mllib packers' coordinate rules) the
tests use to get realistic weights."""

import math

import numpy as np

import golden as g
import golden_arith_float as ga

F32 = np.float32
SEP, TAP4 = 1, 0


# ---- gathers -----------------------------------------------------------------------

def _pix(x, y, xx):
    """x[y, xx] for broadcastable index arrays, clamped to the tensor (the NPU
    reads a halo / pad pixel there, always with a zero weight)."""
    Hi, Wi = x.shape[:2]
    y, xx = np.broadcast_arrays(np.clip(y, 0, Hi - 1), np.clip(xx, 0, Wi - 1))
    return x[y, xx]


def _taps4(ih, iw, s, sv_wo):
    """(y0, y1, x0, x1) of the 4-tap kernel, each (Ho, Wo): the right taps move
    when s1 != 0 (per pixel); the bottom row when s2 of the sub-volume's first
    column != 0."""
    Ho, Wo = s.shape[:2]
    sv_wo = sv_wo or Wo
    w0 = (np.arange(Wo) // sv_wo) * sv_wo
    y0 = np.broadcast_to(np.asarray(ih, np.int64)[:, None], (Ho, Wo))
    x0 = np.broadcast_to(np.asarray(iw, np.int64)[None, :], (Ho, Wo))
    y1 = y0 + (s[:, w0, 2] != 0)
    x1 = x0 + (s[:, :, 1] != 0)
    return y0, y1, x0, x1


# ---- integer bilinear ----------------------------------------------------------------

def bilinear_int(x, ih, iw, wts, route, sv_wo, requantize, c0, c1, dtype):
    """resize_bilinear / resize_bilinear_a16. x (Hi, Wi, C) integer."""
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
            return np.clip(v, -32768, 32767)

        acc = tmp(iw) * qx[None, :, 0, None] + tmp(iw + 1) * qx[None, :, 1, None]
        q = 14 if np.dtype(dtype) == np.uint8 else 15
    else:
        s = wts.reshape(Ho, Wo, 4)
        y0, y1, x0, x1 = _taps4(ih, iw, s, sv_wo)
        acc = (_pix(x, y0, x0) * s[:, :, 0, None] + _pix(x, y0, x1) * s[:, :, 1, None]
               + _pix(x, y1, x0) * s[:, :, 2, None] + _pix(x, y1, x1) * s[:, :, 3, None])
        q = 15
    info = np.iinfo(dtype)
    if requantize:
        c1q = F32(F32(c1) * F32(1 << q))
        v = g.mac_int(acc.reshape(-1), F32(c0), c1q).astype(np.float64)
        out = np.trunc(np.where(np.isnan(v), 0.0, v) / 2.0 ** q)
    else:
        out = np.sign(acc) * (np.abs(acc).reshape(-1) >> q).reshape(acc.shape)
    return np.clip(out, info.min, info.max).astype(dtype).reshape(Ho, Wo, -1)


# ---- float narrowing ------------------------------------------------------------------

def _trunc_grid(a, emin, mbits):
    """|a| (finite, float64) truncated to a float grid with `mbits` mantissa bits
    and smallest normal exponent emin (subnormals below it)."""
    e = np.frexp(np.where(a > 0, a, 1.0))[1] - 1.0  # exact floor(log2 a)
    e = np.maximum(e, emin)
    ulp = np.exp2(e - mbits)
    return np.floor(a / ulp) * ulp


def trunc_bf16(v):
    """fp32 -> bf16 bits toward zero; results below 2^-126 -> signed zero
    (mllib's f2bf filter_denorm)."""
    v = np.asarray(v, F32)
    a = np.abs(v.astype(np.float64))
    r = np.where(np.isfinite(a), _trunc_grid(np.where(np.isfinite(a), a, 0), -126, 7), a)
    r = np.where(a < 2.0 ** -126, 0.0, r)
    bits = g.to_bf16(np.copysign(r, v).astype(F32))
    nan = (v.view(np.uint32) >> 16) | 0x40  # NaN keeps its top bits, made quiet
    return np.where(np.isnan(v), nan, bits).astype(np.uint16)


def trunc_f16(v):
    """fp32 -> f16 bits toward zero, |v| >= 2^16 -> 65504, |v| < 2^-14 -> 0."""
    v = np.asarray(v, F32)
    a = np.abs(v.astype(np.float64))
    fin = np.isfinite(a)
    r = np.where(fin, _trunc_grid(np.where(fin, a, 0), -14, 10), a)
    r = np.where(fin & (a >= 65536.0), 65504.0, r)
    r = np.where(a < 2.0 ** -14, 0.0, r)
    with np.errstate(all="ignore"):
        return np.copysign(r, v).astype(np.float16).view(np.uint16)


def trunc16(v, t):
    return trunc_f16(v) if t == ga.F16 else trunc_bf16(v)


def fp8_trunc_sat(v):
    """fp32 -> E4M3FN toward zero, saturating at 448; subnormals kept.
    NaN -> 0x7F (signed), infinity -> 448."""
    v = np.asarray(v, F32)
    a = np.abs(v.astype(np.float64))
    fin = np.isfinite(a)
    r = np.minimum(np.where(fin, _trunc_grid(np.where(fin, a, 0), -6, 3), 448.0), 448.0)
    code = g.to_fp8(r.astype(F32), g.EVEN) | np.where(np.signbit(v), 0x80, 0).astype(np.uint8)
    return np.where(np.isnan(v), code | 0x7F, code).astype(np.uint8)


def bf16_rne_flush(v):
    """fp32 -> bf16 half to even; inputs / results below 2^-126 -> signed zero."""
    v = np.asarray(v, F32)
    bits = g.to_bf16(v, g.EVEN)
    r = g.bf16_bits_to_f32(bits)
    tiny = (np.abs(v) < F32(2.0 ** -126)) | (np.abs(r) < F32(2.0 ** -126))
    return np.where(tiny, (bits & 0x8000), bits).astype(np.uint16)


# ---- float bilinear -----------------------------------------------------------------------

def _bilinear_acc(xf, ih, iw, s, sv_wo):
    """4-tap sum in fp32: rounded product, then three fused MACs."""
    y0, y1, x0, x1 = _taps4(ih, iw, s, sv_wo)
    acc = ga.mul(_pix(xf, y0, x0), s[:, :, 0, None])
    acc = ga.fma(_pix(xf, y0, x1), s[:, :, 1, None], acc)
    acc = ga.fma(_pix(xf, y1, x0), s[:, :, 2, None], acc)
    return ga.fma(_pix(xf, y1, x1), s[:, :, 3, None], acc)


def bilinear_bf16(xbits, ih, iw, sbits, sv_wo, t):
    """resize_bilinear_bf16: xbits (Hi, Wi, C), sbits (Ho, Wo, 4) raw t bits."""
    xf = ga.load16(xbits, t)
    s = ga.load16(sbits, t)
    return trunc16(_bilinear_acc(xf, ih, iw, s, sv_wo), t)


def bilinear_fp8(codes, ih, iw, sbits, sv_wo, t, dq, oq):
    """resize_bilinear_fp8: codes (Hi, Wi, C) E4M3, interior type t."""
    dec = ga.fp8_rows(codes, codes.shape)
    xf = ga.load16(trunc16(ga.mul(dec, F32(dq)), t), t)
    s = ga.load16(sbits, t)
    acc = _bilinear_acc(xf, ih, iw, s, sv_wo)
    return fp8_trunc_sat(ga.mul(acc, F32(oq)))


# ---- bicubic --------------------------------------------------------------------------------

def cubic_bf16(xbits, ih, iw, cy_bits, cx_bits):
    """resize_cubic_bf16: pass 1 over H into a bf16 plane, then pass 2 over W."""
    xf = g.bf16_bits_to_f32(xbits)
    Hi, Wi, C = xf.shape
    cy = g.bf16_bits_to_f32(cy_bits).reshape(-1, 4)
    cx = g.bf16_bits_to_f32(cx_bits).reshape(-1, 4)
    ih = np.asarray(ih, np.int64)
    iw = np.asarray(iw, np.int64)
    rows = lambda k: xf[np.clip(ih + k, 0, Hi - 1)]  # (Ho, Wi, C)
    acc = ga.mul(rows(0), cy[:, 0, None, None])
    for k in range(1, 4):
        acc = ga.fma(rows(k), cy[:, k, None, None], acc)
    t = g.bf16_bits_to_f32(bf16_rne_flush(acc))  # (Ho, Wi, C)
    cols = lambda k: t[:, np.clip(iw + k, 0, Wi - 1)]  # (Ho, Wo, C)
    acc = ga.mul(cols(0), cx[None, :, 0, None])
    for k in range(1, 4):
        acc = ga.fma(cols(k), cx[None, :, k, None], acc)
    return bf16_rne_flush(acc)


# ---- host-side tables (mllib packer rules, for realistic test weights) ------------------

def coord(o, n_in, n_out, mode):
    """ONNX source coordinate of output index o (half_pixel / asymmetric /
    align_corners), unclamped."""
    if mode == "align_corners":
        return 0.0 if n_out <= 1 else o * (n_in - 1) / (n_out - 1)
    scale = n_out / n_in
    return (o + 0.5) / scale - 0.5 if mode == "half_pixel" else o / scale


def bilinear_axis(n_in, n_out, mode):
    """Per output: (floor index, w0, w1) with the coordinate clamped into
    [0, n_in - 1], weights fp32 as the packers compute them."""
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
    """4-tap Q15 s_table [Ho][Wo][4] (truncating, as the packer)."""
    sy = np.stack([wy0, wy1], 1)[:, None, :, None]
    sx = np.stack([wx0, wx1], 1)[None, :, None, :]
    p = (sy * sx).astype(F32).reshape(len(wy0), len(wx0), 4)
    return (p * F32(1 << 15)).astype(np.uint16)


def sep_table(wy0, wy1, wx0, wx1, qh):
    """Separable qy (Q_H) / qx (Q7) tables, rounding half up as the packer."""
    q = lambda w, b: (np.asarray(w, F32) * F32(1 << b) + F32(0.5)).astype(np.uint16)
    qy = np.stack([q(wy0, qh), q(wy1, qh)], 1)
    qx = np.stack([q(wx0, 7), q(wx1, 7)], 1)
    return np.concatenate([qy.reshape(-1), qx.reshape(-1)]).astype(np.uint16)


def float_table(wy0, wy1, wx0, wx1, t):
    """bf16 / f16 s_table bits [Ho][Wo][4] (products in fp32, then RNE)."""
    sy = np.stack([wy0, wy1], 1)[:, None, :, None]
    sx = np.stack([wx0, wx1], 1)[None, :, None, :]
    p = (sy * sx).astype(F32).reshape(len(wy0), len(wx0), 4)
    return ga.round16(p, t).astype(np.uint16)


def cubic_axis(n_in, n_out, mode, a=-0.75):
    """Clamp-folded cubic window starts and weights (ResizeCubicBf16Wrapper
    cubic_weights_1d), weights as bf16 bits."""
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
    return idx, g.to_bf16(coef.astype(F32)).astype(np.uint16)
