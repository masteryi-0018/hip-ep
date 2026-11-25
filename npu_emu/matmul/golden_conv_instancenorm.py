# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for matmul/conv_instancenorm.h (conv + instance norm)."""

import numpy as np

import golden_qdq_conv as gm
from golden_groupnorm import amad, dq, f32, fold, in_qdq_tail, norm_coeffs
from golden_mac import add_q, fma_q


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
                    acc[s & 1] = add_q(acc[s & 1], blk[s])
                    acc2[s & 1] = fma_q(blk[s], blk[s], acc2[s & 1])
                for wh in (0, 1):
                    p, p2 = add_q(p, acc[wh]), add_q(p2, acc2[wh])
        return p, p2

    def cascade(parts):
        r, r2 = parts[-1]
        for k in range(len(parts) - 2, -1, -1):
            r, r2 = add_q(parts[k][0], r), add_q(parts[k][1], r2)
        return r, r2

    if split_mode == 1:
        band = ho // 4
        return cascade([tile(k * band, 0) for k in range(4)])
    cw = wo // 3
    out = None
    for c in range(3):
        r = cascade([tile(0, c * cw + k * (cw // 4)) for k in range(4)])
        out = r if out is None else (add_q(r[0], out[0]), add_q(r[1], out[1]))
    return out


def _conv_in(ifm, wts, out_t, sh, sw, split_mode, sv_h, sv_w, h_loop, w_loop, pad, c0, c2, dq_zp,
             dq_sc, q_zp, q_sc, slope, alpha, beta, amad_ops=None, dq2=None):
    """Returns (output codes, conv output); alpha / beta (and the add constant) as float values,
    the norm computes in float16 with the leaky ReLU."""
    ct = "f16"
    mid = gm.conv_qdq_int16x8(ifm, wts, out_t, sh, sw, 0, c0, None, c2)
    ho, wo, co = mid.shape
    X = dq(mid, dq_sc, dq_zp, ct).reshape(ho, wo, co)
    S, S2 = conv_stats(X, sv_h, sv_w, h_loop, w_loop, split_mode)
    N = (12 if split_mode == 0 else 4) * sv_h * sv_w * h_loop * w_loop - pad
    A, B = fold(*norm_coeffs(S, S2, N), alpha, beta)
    if amad_ops is not None:
        aq, cst, a2q = amad_ops
        A, B = amad(A, B, dq(aq, *dq2, ct), cst, dq(a2q, *dq2, ct))
    return in_qdq_tail(X.reshape(ho * wo, co), A, B, ct, out_t, 3, slope, q_sc, q_zp), mid


def qdq_ConvInFusion_A16(ifm, wts, alpha, beta, out_t, sh, sw, split_mode, sv_h, sv_w, h_loop,
                         w_loop, pad, c0, c2, dq_zp, dq_sc, q_zp, q_sc, slope):
    """Output codes (ho * wo, co); alpha / beta float values."""
    return _conv_in(ifm, wts, out_t, sh, sw, split_mode, sv_h, sv_w, h_loop, w_loop, pad, c0, c2,
                    dq_zp, dq_sc, q_zp, q_sc, slope, alpha, beta)[0]


def qdq_ConvInstanceNormAddMulAdd(ifm, wts, alpha, beta, add1_act, add2, add1_const, out_t, sh,
                                  sw, split_mode, sv_h, sv_w, h_loop, w_loop, pad, c0, c2, dq_zp,
                                  dq_sc, q_zp, q_sc, dq_ifm2_zp, dq_ifm2_sc, slope):
    """Output codes (ho * wo, co); alpha / beta / add1_const float values."""
    return _conv_in(ifm, wts, out_t, sh, sw, split_mode, sv_h, sv_w, h_loop, w_loop, pad, c0, c2,
                    dq_zp, dq_sc, q_zp, q_sc, slope, alpha, beta, (add1_act, add1_const, add2),
                    (dq_ifm2_sc, dq_ifm2_zp))[0]
