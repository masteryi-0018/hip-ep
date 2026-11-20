# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for matmul/conv_instancenorm.h (conv + instance norm)."""

import numpy as np

import golden_qdq_conv as gm
from golden_groupnorm import add, amad, dq, f32, fma, fold, in_qdq_tail, norm_coeffs


def conv_stats(X, sv_h, sv_w, h_loop, w_loop, split_mode):
    """Block sums of X (ho, wo, C) -> (S, S2)."""
    ho, wo, C = X.shape

    def tile(h0, w0):
        p = np.zeros(C, f32)
        p2 = np.zeros(C, f32)
        for ih in range(h_loop):
            for iw in range(w_loop):
                blk = X[h0 + ih * sv_h:h0 + (ih + 1) * sv_h,
                        w0 + iw * sv_w:w0 + (iw + 1) * sv_w].reshape(-1, C)
                acc = [np.zeros(C, f32), np.zeros(C, f32)]
                acc2 = [np.zeros(C, f32), np.zeros(C, f32)]
                for s in range(blk.shape[0]):
                    acc[s & 1] = add(acc[s & 1], blk[s])
                    acc2[s & 1] = fma(blk[s], blk[s], acc2[s & 1])
                for wh in (0, 1):
                    p, p2 = add(p, acc[wh]), add(p2, acc2[wh])
        return p, p2

    def cascade(parts):
        r, r2 = parts[-1]
        for k in range(len(parts) - 2, -1, -1):
            r, r2 = add(parts[k][0], r), add(parts[k][1], r2)
        return r, r2

    if split_mode == 1:
        band = ho // 4
        return cascade([tile(k * band, 0) for k in range(4)])
    cw = wo // 3
    out = None
    for c in range(3):
        r = cascade([tile(0, c * cw + k * (cw // 4)) for k in range(4)])
        out = r if out is None else (add(r[0], out[0]), add(r[1], out[1]))
    return out


def conv_in(ifm, wts, sh, sw, c0, c1, c2, conv_act, act0, act1, conv_t, alpha, beta, ct,
            split_mode, sv_h, sv_w, h_loop, w_loop, pad, dq_sc, dq_zp, q_sc, q_zp, act, slope,
            amad_ops=None, dq2=None):
    """Returns (output codes, conv output)."""
    mid = gm.conv_qdq(ifm, wts, conv_t, sh, sw, c0, c1, c2, True, conv_act, act0, act1)
    ho, wo, co = mid.shape
    X = dq(mid, dq_sc, dq_zp, ct).reshape(ho, wo, co)
    S, S2 = conv_stats(X, sv_h, sv_w, h_loop, w_loop, split_mode)
    N = (12 if split_mode == 0 else 4) * sv_h * sv_w * h_loop * w_loop - pad
    A, B = fold(*norm_coeffs(S, S2, N), alpha, beta)
    if amad_ops is not None:
        aq, cst, a2q = amad_ops
        A, B = amad(A, B, dq(aq, *dq2, ct), cst, dq(a2q, *dq2, ct))
    return in_qdq_tail(X.reshape(ho * wo, co), A, B, ct, conv_t, act, slope, q_sc, q_zp), mid
