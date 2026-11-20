# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for npu_bidirectional_lstm_bf16 (one LSTM direction)."""

import numpy as np

from golden_float_cvt import bf16_to_f32, f32_to_bf16, round_float
from golden_int_round import Rnd
from golden_mac import add, fma, mul
from golden_nlf import lut_index, lut_interp

SEGMENTS = 128


def default_lut():
    """Default sigmoid table: per segment of width 1/8 over [-8, 8] the
    least-squares line over [a - 1/16, b] (129 samples)."""
    grid = np.linspace(-8.0, 8.0, SEGMENTS + 1).astype(np.float16).astype(np.float64)
    off = np.empty(SEGMENTS)
    sl = np.empty(SEGMENTS)
    for s in range(SEGMENTS):
        xs = np.linspace(grid[s] - 0.0625, grid[s + 1], 129)
        ys = 1.0 / (1.0 + np.exp(-xs))
        sl[s] = ((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum()
        off[s] = ys.mean() - sl[s] * xs.mean()
    return pack_lut(off.astype(np.float32), sl.astype(np.float32))


def pack_lut(off, sl):
    """512 fp32 words: per group of 8 segments the 8 (offset, slope) pairs, twice."""
    words = []
    for grp in range(0, SEGMENTS, 8):
        pairs = np.stack([off[grp:grp + 8], sl[grp:grp + 8]], 1).reshape(-1)
        words += [pairs, pairs]
    return np.concatenate(words).astype(np.float32)


def sigmoid_lut(v, lut):
    """128 segments of width 1/8 over [-8, 8], result rounded to f16."""
    v = np.asarray(v, np.float32)
    y = lut_interp(v, lut, lut_index(v, 8, SEGMENTS, Rnd.kHalfEven), "f32", Rnd.kHalfEven)
    return round_float(y, "f16")


def tanh_lut(v, lut):
    with np.errstate(all="ignore"):
        sg = sigmoid_lut(round_float(np.asarray(v, np.float32) * np.float32(2), "f16"), lut)
        return round_float(round_float(sg * np.float32(2), "f16") - np.float32(1), "f16")


def gemv(v, mat, bias):
    """mat: rows x K (one row per gate column), v: K, bias: rows."""
    k = mat.shape[1]
    chains = [np.zeros(mat.shape[0], np.float32) for _ in range(4)]
    for i in range(0, k - k % 4, 4):
        for q in range(4):
            chains[q] = fma(mat[:, i + q], np.float32(v[i + q]), chains[q])
    psum = np.asarray(bias, np.float32)
    for c in chains:
        psum = add(c, psum)
    return psum


def bilstm(x, h0, c0, w, r, b, lut, reverse):
    """Tensors as float32 values; x: seq x I, w: 4H x I, r: 4H x H, b: 8H.
    Returns y (seq x H) and c_out (H) as bf16 bits."""
    hid = h0.size
    h = np.asarray(h0, np.float32)
    c = np.asarray(c0, np.float32)
    y = np.zeros((x.shape[0], hid), np.uint16)
    for t in (range(x.shape[0] - 1, -1, -1) if reverse else range(x.shape[0])):
        z = round_float(add(gemv(x[t], w, b[:4 * hid]), gemv(h, r, b[4 * hid:])), "f16")
        ig, og, fg = (sigmoid_lut(z[q * hid:(q + 1) * hid], lut) for q in range(3))
        gg = tanh_lut(z[3 * hid:], lut)
        c = add(mul(fg, c), mul(ig, gg))
        hb = f32_to_bf16(mul(og, tanh_lut(round_float(c, "f16"), lut)))
        y[t] = hb
        h = bf16_to_f32(hb)
    return y, f32_to_bf16(c)
