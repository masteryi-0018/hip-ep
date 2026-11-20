# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the bf16 matmul / convolution entry points."""

import numpy as np

from golden_float_cvt import (bf16_to_f32, bits_f32, f32_bits, f32_to_bf16, kFtzIn, round_float,
                              scalar_cast)
from golden_int_round import Rnd, float_to_int
from golden_mac import add, fma, mac_exact, mul, mul16_f32
from golden_minmax import vclamp, vmax
from golden_mmac import mmac_bf16

F32, I64 = np.float32, np.int64


def mmac_chain(acc_bits, xs, ws, block=4096):
    """One MMAC per (x, w) in order on the fp32 accumulators acc_bits (P x N).
    x: P x 32 bf16 bits, w: 32 x N bf16 bits."""
    acc = np.array(acc_bits, np.uint32, copy=True)
    for x, w in zip(xs, ws):
        wt = np.asarray(w, np.uint16).T[None, :, :]
        for p0 in range(0, acc.shape[0], block):
            acc[p0:p0 + block] = mmac_bf16(acc[p0:p0 + block], np.asarray(x[p0:p0 + block])[:, None, :], wt)
    return acc


def _pad32(a, axis):
    n = a.shape[axis]
    if n == 32:
        return a
    p = [(0, 0)] * a.ndim
    p[axis] = (0, 32 - n)
    return np.pad(a, p)


def gemm_acc(a_bits, b_bits, acc_bits):
    """fp32 accumulators of A (M x K) @ B (K x N): one MMAC per 32 consecutive K."""
    K = a_bits.shape[1]
    xs = (_pad32(a_bits[:, k:k + 32], 1) for k in range(0, K, 32))
    ws = (_pad32(b_bits[k:k + 32], 0) for k in range(0, K, 32))
    return mmac_chain(acc_bits, xs, ws)


def conv_acc(ifm_bits, wts_bits, acc_bits, sh, sw, ho, wo, kh_sv):
    """fp32 accumulators (Ho*Wo x Co) of a bf16 convolution, ifm Hi x Wi x Ci, wts
    Kh x Kw x Ci x Co. MMAC order: groups of kh_sv kernel rows outermost, then
    blocks of 32 input channels, the rows and the columns; with ci < 32 the
    columns of a row are packed into the 32 lanes."""
    kh, kw, ci, co = wts_bits.shape

    def win(y, x):
        return ifm_bits[y:y + (ho - 1) * sh + 1:sh, x:x + (wo - 1) * sw + 1:sw].reshape(ho * wo, ci)

    def levels():
        for k0 in range(0, kh, kh_sv):
            rows = range(k0, min(k0 + kh_sv, kh))
            if ci < 32:
                for y in rows:
                    xs = np.concatenate([win(y, x) for x in range(kw)], axis=1)
                    ws = wts_bits[y].reshape(kw * ci, co)
                    for c0 in range(0, kw * ci, 32):
                        yield _pad32(xs[:, c0:c0 + 32], 1), _pad32(ws[c0:c0 + 32], 0)
            else:
                for c0 in range(0, ci, 32):
                    for y in rows:
                        for x in range(kw):
                            yield _pad32(win(y, x)[:, c0:c0 + 32], 1), _pad32(wts_bits[y, x, c0:c0 + 32], 0)

    pairs = list(levels())
    return mmac_chain(acc_bits, (p[0] for p in pairs), (p[1] for p in pairs))


def _acc0(bias, rows, n):
    if bias is None:
        return np.zeros((rows, n), np.uint32)
    return np.broadcast_to(f32_bits(np.asarray(bias, np.float32).reshape(1, n)), (rows, n)).copy()


def _act(v, act, alpha=0.0):
    """ReLU / ReLU6 / LeakyReLU on bf16 values (float32 holding bf16)."""
    v = np.asarray(v, F32)
    if act == 1:
        v = vmax(v, F32(0))
    elif act == 2:
        v = vclamp(v, F32(0), F32(6))
    elif act == 3:
        v = vmax(v, mul16_f32(v, scalar_cast(F32(alpha), "bf16"), "bf16"))
    return round_float(v, "bf16", Rnd.kHalfAway, kFtzIn)


def gemm(a, b, bias, act):
    """a M x K, b K x N bf16 bits -> bf16 bits."""
    acc = gemm_acc(np.asarray(a, np.uint16), np.asarray(b, np.uint16), _acc0(bias, a.shape[0], b.shape[1]))
    return f32_to_bf16(_act(round_float(bits_f32(acc), "bf16", Rnd.kHalfAway, kFtzIn), act))


def gemm_qdq(a, b, bias, act, q_signed, inv_scale, scale, zp):
    v = bf16_to_f32(gemm(a, b, bias, act))
    q = float_to_int(fma(v, F32(inv_scale), F32(zp)), "i16" if q_signed else "u16")
    d = mac_exact(q, F32(scale), mul(F32(-zp), F32(scale)))
    return f32_to_bf16(d, Rnd.kHalfEven, kFtzIn)


def conv(ifm, wts, bias, sh, sw, kh_sv, act, alpha=0.0):
    kh, kw, ci, co = wts.shape
    ho, wo = (ifm.shape[0] - kh) // sh + 1, (ifm.shape[1] - kw) // sw + 1
    acc = conv_acc(np.asarray(ifm, np.uint16), np.asarray(wts, np.uint16), _acc0(bias, ho * wo, co),
                   sh, sw, ho, wo, kh_sv)
    return f32_to_bf16(_act(round_float(bits_f32(acc), "bf16", Rnd.kHalfAway, kFtzIn), act, alpha)).reshape(ho, wo, co)


def conv_add(ifm, wts, bias, ifm2, sh, sw, kh_sv, typ, act):
    kh, kw, ci, co = wts.shape
    ho, wo = (ifm.shape[0] - kh) // sh + 1, (ifm.shape[1] - kw) // sw + 1
    res = np.broadcast_to(bf16_to_f32(ifm2), (ho, wo, co)).reshape(ho * wo, co)
    b = None if bias is None else np.broadcast_to(np.asarray(bias, np.float32), (ho * wo, co))
    if typ == 0:
        seed = res if b is None else add(res, b)
        acc = conv_acc(ifm, wts, f32_bits(seed).copy(), sh, sw, ho, wo, kh_sv)
        return f32_to_bf16(bits_f32(acc), Rnd.kHalfAway, kFtzIn).reshape(ho, wo, co)
    acc = conv_acc(ifm, wts, _acc0(bias, ho * wo, co), sh, sw, ho, wo, kh_sv)
    v = round_float(bits_f32(acc), "bf16", Rnd.kHalfAway, kFtzIn)
    if act == 1:
        v = vmax(v, F32(0))
    return f32_to_bf16(add(v, res), Rnd.kHalfEven, kFtzIn).reshape(ho, wo, co)


def depthwise(ifm, wts, bias, sh, sw, act, alpha=0.0):
    kh, kw, c = wts.shape
    ho, wo = (ifm.shape[0] - kh) // sh + 1, (ifm.shape[1] - kw) // sw + 1
    x, w = bf16_to_f32(ifm), bf16_to_f32(wts)
    acc = np.broadcast_to(np.asarray(bias, np.float32) if bias is not None else np.float32(0),
                          (ho, wo, c)).astype(np.float32)
    for y in range(kh):
        for z in range(kw):
            acc = add(acc, mul(x[y:y + (ho - 1) * sh + 1:sh, z:z + (wo - 1) * sw + 1:sw], w[y, z]))
    v = round_float(acc, "bf16", Rnd.kHalfAway, kFtzIn)
    if act == 1:
        v = vmax(v, F32(0))
    elif act == 2:
        v = vclamp(v, F32(0), F32(6))
    elif act == 3:
        v = vmax(v, mul16_f32(scalar_cast(F32(alpha), "bf16"), v, "bf16"))
    return f32_to_bf16(v, Rnd.kHalfAway, kFtzIn)
