# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the bf16 matmul / convolution entry points."""

import numpy as np

from golden_act import activate_bf16
from golden_bits import bits_f32, f32_bits
from golden_float_cvt import acc_to_bf16, bf16_to_f32, f32_to_bf16, kFtzIn, scalar_cast
from golden_int_round import Rnd
from golden_mac import add_q, mul_q
from golden_mmac import mmac_bf16
from golden_quant import dequant, dq_offset, quant

F32, I64 = np.float32, np.int64


def _mmac_chain(acc_bits, xs, ws, block=4096):
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


def _gemm_acc(a_bits, b_bits, acc_bits):
    """fp32 accumulators of A (M x K) @ B (K x N): one MMAC per 32 consecutive K."""
    K = a_bits.shape[1]
    xs = (_pad32(a_bits[:, k:k + 32], 1) for k in range(0, K, 32))
    ws = (_pad32(b_bits[k:k + 32], 0) for k in range(0, K, 32))
    return _mmac_chain(acc_bits, xs, ws)


def _conv_acc(ifm_bits, wts_bits, acc_bits, sh, sw, ho, wo, kh_sv):
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
    return _mmac_chain(acc_bits, (p[0] for p in pairs), (p[1] for p in pairs))


def _bias_f32(bias):
    """fp32 values of a bias given as bf16 bits (uint16) or fp32; None stays None."""
    if bias is None:
        return None
    bias = np.asarray(bias)
    return bf16_to_f32(bias) if bias.dtype == np.uint16 else bias.astype(np.float32)


def _acc0(bias, rows, n):
    if bias is None:
        return np.zeros((rows, n), np.uint32)
    return np.broadcast_to(f32_bits(_bias_f32(bias).reshape(1, n)), (rows, n)).copy()


def _act(v, act, alpha=0.0):
    """ReLU / ReLU6 / LeakyReLU on bf16 values (float32 holding bf16)."""
    v = activate_bf16(v, act, scalar_cast(F32(alpha), "bf16"))
    return acc_to_bf16(v)


def gemm_bf16x16(a, b, bias, activation):
    """a M x K, b K x N bf16 bits, bias bf16 bits / fp32 / None -> bf16 bits (M x N)."""
    acc = _gemm_acc(np.asarray(a, np.uint16), np.asarray(b, np.uint16),
                   _acc0(bias, a.shape[0], b.shape[1]))
    out = f32_to_bf16(_act(acc_to_bf16(bits_f32(acc)), activation))
    return out.reshape(a.shape[0], b.shape[1])


gemm_bf16x16_buffer = gemm_bf16x16_ifm_reuse = gemm_bf16x16


def gemm_bf16x16_transposed(a, b, bias, activation):
    return gemm_bf16x16(a, b, bias, activation).T


def gemm_quant_dequant_bf16x16(a, b, bias, quant_mult, dequant_mult, zero_point, qdq_sign,
                               activation):
    v = bf16_to_f32(gemm_bf16x16(a, b, bias, activation))
    q = quant(v, F32(quant_mult), F32(zero_point), "i16" if qdq_sign else "u16")
    d = dequant(q, F32(dequant_mult), dq_offset(zero_point, F32(dequant_mult)))
    return f32_to_bf16(d, Rnd.kHalfEven, kFtzIn)


gemm_quant_dequant_bf16x16_ifm_reuse = gemm_quant_dequant_bf16x16


def conv_bf16x16_buffer(ifm, wts, bias, sh, sw, kh_sv, lrelu_alpha, activation):
    kh, kw, ci, co = wts.shape
    ho, wo = (ifm.shape[0] - kh) // sh + 1, (ifm.shape[1] - kw) // sw + 1
    acc = _conv_acc(np.asarray(ifm, np.uint16), np.asarray(wts, np.uint16), _acc0(bias, ho * wo, co),
                   sh, sw, ho, wo, kh_sv)
    v = _act(acc_to_bf16(bits_f32(acc)), activation, lrelu_alpha)
    return f32_to_bf16(v).reshape(ho, wo, co)


def conv_bf16x16(ifm, wts, bias, sh, sw, kh_sv, activation):
    return conv_bf16x16_buffer(ifm, wts, bias, sh, sw, kh_sv, 0.0, activation)


def conv_add_bf16x16(ifm, ifm2, wts, bias, sh, sw, kh_sv, activation_type, activation):
    kh, kw, ci, co = wts.shape
    ho, wo = (ifm.shape[0] - kh) // sh + 1, (ifm.shape[1] - kw) // sw + 1
    res = np.broadcast_to(bf16_to_f32(ifm2), (ho, wo, co)).reshape(ho * wo, co)
    bf = _bias_f32(bias)
    b = None if bf is None else np.broadcast_to(bf, (ho * wo, co))
    if activation_type == 0:
        seed = res if b is None else add_q(res, b)
        acc = _conv_acc(ifm, wts, f32_bits(seed).copy(), sh, sw, ho, wo, kh_sv)
        return f32_to_bf16(bits_f32(acc), Rnd.kHalfAway, kFtzIn).reshape(ho, wo, co)
    acc = _conv_acc(ifm, wts, _acc0(bias, ho * wo, co), sh, sw, ho, wo, kh_sv)
    v = activate_bf16(acc_to_bf16(bits_f32(acc)), activation, 0.0)
    return f32_to_bf16(add_q(v, res), Rnd.kHalfEven, kFtzIn).reshape(ho, wo, co)


def depthwise_conv_bf16x16(ifm, wts, bias, sh, sw, alpha, activation):
    kh, kw, c = wts.shape
    ho, wo = (ifm.shape[0] - kh) // sh + 1, (ifm.shape[1] - kw) // sw + 1
    x, w = bf16_to_f32(ifm), bf16_to_f32(wts)
    bf = _bias_f32(bias)
    acc = np.broadcast_to(bf if bf is not None else np.float32(0), (ho, wo, c)).astype(np.float32)
    for y in range(kh):
        for z in range(kw):
            acc = add_q(acc, mul_q(x[y:y + (ho - 1) * sh + 1:sh, z:z + (wo - 1) * sw + 1:sw], w[y, z]))
    v = acc_to_bf16(acc)
    v = activate_bf16(v, activation, scalar_cast(F32(alpha), "bf16"))
    return f32_to_bf16(v, Rnd.kHalfAway, kFtzIn)


depthwise_conv_bf16x16_c32 = depthwise_conv_bf16x16
