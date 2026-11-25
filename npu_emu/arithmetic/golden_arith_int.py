# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the integer / QDQ elementwise arithmetic; tensors flat or HWC."""

import numpy as np

import golden_quant as gq
from golden_float_cvt import float_to_int
from golden_int_round import Rnd, sat_add64, shift_round, srs, wrap
from golden_mac import fma_q, mac_exact, msc_q, mul_q
from golden_minmax import vclamp, vmax

f32 = np.float32
# activation codes
ACT_NONE, ACT_RELU, ACT_RELU6, ACT_LRELU, ACT_PRELU = 0, 1, 2, 3, 9


def _relu_variants(act):
    """Flat fused activation: relu and relu6 both clamp to [a0, a1]."""
    return (gq.kQdqActClip if act in (ACT_RELU, ACT_RELU6) else gq.kQdqActLrelu
            if act == ACT_LRELU else gq.kQdqActNone)


def _i64(x):
    return np.asarray(x).astype(np.int64)


def _add(a, b, out_t, c0, c1, c2, act=ACT_NONE, a0=0.0, a1=0.0):
    """ifm2 b broadcast onto a (numpy rules)."""
    a, b = _i64(a), np.broadcast_to(_i64(b), np.shape(a))
    v = mac_exact(b, c1, mac_exact(a, c0, c2))
    return gq.activate_out(v, _relu_variants(act), a0, a1, 0.0, out_t)


def _mul(a, b, out_t, c0, c1, c2, c3, act=ACT_NONE, a0=0.0, a1=0.0):
    a, b = _i64(a), np.broadcast_to(_i64(b), np.shape(a))
    v = mac_exact(a, mac_exact(b, c0, c1), mac_exact(b, c2, c3))
    return gq.activate_out(v, _relu_variants(act), a0, a1, 0.0, out_t)


def _add_fused_xint(a, b, c0, c1, act=ACT_NONE, a0=0.0, a1=0.0):
    e0, e1 = (int(np.frexp(f32(c))[1]) - 1 for c in (c0, c1))
    base = min(e0, e1, 0)
    # [default] shifts of 64 or more saturate instead of wrapping or being undefined
    v = sat_add64(shift_round(_i64(a), -(e0 - base), Rnd.kHalfEven),
                  shift_round(_i64(b), -(e1 - base), Rnd.kHalfEven))
    q = srs(v, -base, "i8")
    if act in (ACT_RELU, ACT_RELU6):
        q = np.clip(q, float_to_int(f32(a0), "i8"), float_to_int(f32(a1), "i8"))
    return q


def _mac_xint(x1, x2, x3, out_t, A, v, p, t):
    x1 = _i64(x1)
    x2 = np.broadcast_to(_i64(x2), x1.shape)
    x3 = np.broadcast_to(_i64(x3), x1.shape)
    w = srs(x2 * A, t, "i16")  # saturates to int16
    return srs(x1 * w + x3 * v, p, out_t)


def _mac_xint_scalar(x1, x2, x3, out_t, A, v, p, t):
    """ifm2 a single value: w wraps to int16."""
    x1 = _i64(x1)
    w = int(wrap(shift_round(np.array([int(x2) * A]), t, Rnd.kHalfEven), "i16")[0])
    x3 = np.broadcast_to(_i64(x3), x1.shape)
    return srs(x1 * w + x3 * v, p, out_t)


def _hwc(a, b):
    a = _i64(a)
    return a, np.broadcast_to(_i64(b), a.shape), np.shape(b)[2] == 1


def _addsub_fused(a, b, out_t, k, act, slope=None):
    """k: (c0, c1, c2, activation_coeff0, activation_coeff1, lrelu_c0, lrelu_c1, lrelu_bias)."""
    c0, c1, c2, w3, w4, w5, w6, w7 = (f32(v) for v in k)
    a, b, bcc = _hwc(a, b)
    if bcc:
        icpt = mac_exact(b, c1, c2)
        r = mac_exact(a, c0, icpt)
    else:
        r = mac_exact(b, c1, mac_exact(a, c0, c2))
    if act == ACT_RELU:
        v = vmax(r, w3)
    elif act == ACT_RELU6:
        v = vclamp(r, w3, w4)
    elif act == ACT_LRELU:
        l = mac_exact(a, w5, fma_q(icpt, w3, w4)) if bcc else mac_exact(b, w6, mac_exact(a, w5, w7))
        v = vmax(r, l)
    elif act == ACT_PRELU:
        s = np.broadcast_to(np.asarray(slope, f32), r.shape)
        v = fma_q(r, np.where(r < 0, s, f32(1)), w3)
    else:
        v = r
    return float_to_int(v, out_t)


def _mul_fused(a, b, out_t, k, act, a16, neg_scale=None):
    """k: (c0, c1, c2, c3, activation_coeff0, activation_coeff1, ifm1_zero_point,
    ofm_zero_point, ifm2_zero_point)."""
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
    elif act == ACT_RELU6:
        v = vclamp(horner(), a0, a1)
    elif act == ACT_LRELU and not a16:
        p = r1 * r2
        if bcc:
            v = mac_exact(p, np.where(neg, c1, c0), zy)
        else:
            v = mac_exact(np.abs(p), c3, mac_exact(p, c2, zy))
    elif act == ACT_LRELU:
        if bcc:
            v = vmax(shifted(), mac_exact(r1, mul_q(F, a0), zy))
        else:
            v = gq.activate_f32(horner(), gq.kQdqActLrelu, a0, a1)
    elif act == ACT_PRELU:
        s = np.where(neg, np.broadcast_to(np.asarray(neg_scale, f32), a.shape), c0)
        v = mac_exact(r1, mac_exact(r2, s, f32(0)), zy)
    else:
        v = shifted()
    return float_to_int(v, out_t)


def _mac_fused(a, b, c, out_t, k, act, addend_to_float):
    """k: (c0, c1, c2, c3, c4, activation_coeff0, activation_coeff1, ifm1_zero_point)."""
    c0, c1, c2, c3, c4, a0, a1 = (f32(v) for v in k[:7])
    zp1 = int(k[7])
    a = _i64(a)
    bcc = np.shape(b)[2] == 1
    b = np.broadcast_to(_i64(b), a.shape)
    c = np.broadcast_to(_i64(c), a.shape)
    if bcc and act == ACT_NONE:
        c3e = msc_q(f32(zp1), c1, c3)
        v = mac_exact(a, mac_exact(b, c0, c1), mac_exact(c, c4, mac_exact(b, c2, c3e)))
    else:
        f = mac_exact(b, c0, c1)
        w = fma_q(c.astype(f32), c4, c3) if (bcc and addend_to_float) else mac_exact(c, c4, c3)
        v = mac_exact(a - zp1, f, w)
        if act == ACT_LRELU:
            v = gq.activate_f32(v, gq.kQdqActLrelu, a0, a1)
    q = float_to_int(v, out_t)
    if act in (ACT_RELU, ACT_RELU6):
        q = np.maximum(q, float_to_int(a0, out_t))
        if act == ACT_RELU6:
            q = np.minimum(q, float_to_int(a1, out_t))
    return q


# ---- the entry points (named as the entries without the npu_ prefix; arguments in the entry's
# order without the stream and the output buffers; types as 'i8' / 'u8' / ...; shapes as
# (H, W, C)) ----

def _shaped(x, shape):
    return _i64(x).reshape(shape)


def elementwise_add_a8(a, in_type, b, out_type, n, c0, c1, c2):
    return _add(_i64(a)[:n], _i64(b)[:n], out_type, c0, c1, c2)


elementwise_add_a16 = elementwise_add_a8


def elementwise_mul_a8(a, in_type, b, out_type, n, c0, c1, c2, c3):
    return _mul(_i64(a)[:n], _i64(b)[:n], out_type, c0, c1, c2, c3)


elementwise_mul_a16 = elementwise_mul_a8


def elementwise_add_fused_a8(a, in_type, b, out_type, n, c0, c1, c2, activation, c3, c4):
    return _add(_i64(a)[:n], _i64(b)[:n], out_type, c0, c1, c2, activation, c3, c4)


elementwise_add_fused_a16 = elementwise_add_fused_a8
elementwise_sub_fused_a8 = elementwise_add_fused_a8
elementwise_sub_fused_a16 = elementwise_add_fused_a8


def elementwise_mul_fused_a8(a, in_type, b, out_type, n, c0, c1, c2, c3, activation, c4, c5):
    return _mul(_i64(a)[:n], _i64(b)[:n], out_type, c0, c1, c2, c3, activation, c4, c5)


elementwise_mul_fused_a16 = elementwise_mul_fused_a8


def elementwise_add_a8_xint(a, b, n, c0, c1, c2):
    return _add(_i64(a)[:n], _i64(b)[:n], "i8", c0, c1, c2)


def elementwise_add_fused_a8_xint(a, b, n, c0, c1, activation, c3, c4):
    return _add_fused_xint(_i64(a)[:n], _i64(b)[:n], c0, c1, activation, c3, c4)


def elementwise_mac_a8(ifm1, ifm1_type, ifm2, ifm2_type, ifm3, ifm3_type, out_type, shape1,
                       shape2, shape3, fx_A, fx_v, fx_p, fx_t):
    x1, x3 = _shaped(ifm1, shape1), _shaped(ifm3, shape3)
    if tuple(shape2) == (1, 1, 1):
        return _mac_xint_scalar(x1, int(_i64(ifm2).reshape(-1)[0]), x3, out_type, fx_A, fx_v,
                                fx_p, fx_t)
    return _mac_xint(x1, _shaped(ifm2, shape2), x3, out_type, fx_A, fx_v, fx_p, fx_t)


def qdq_elementwise_broadcasting3d_add_a8(ifm1, dtype, ifm2, shape1, shape2, c0, c1, c2):
    return _add(_shaped(ifm1, shape1), _shaped(ifm2, shape2), dtype, c0, c1, c2)


qdq_elementwise_broadcasting3d_add_a16 = qdq_elementwise_broadcasting3d_add_a8


def qdq_elementwise_broadcasting3d_mul_a8(ifm1, dtype, ifm2, shape1, shape2, c0, c1, c2, c3):
    return _mul(_shaped(ifm1, shape1), _shaped(ifm2, shape2), dtype, c0, c1, c2, c3)


qdq_elementwise_broadcasting3d_mul_a16 = qdq_elementwise_broadcasting3d_mul_a8


def qdq_elementwise_broadcasting3d_add_fused_a8(ifm1, dtype, ifm2, shape1, shape2, coeff,
                                                activation, slope):
    """coeff: the 8 words of npu_bc3d_addsub_fused_coeff."""
    return _addsub_fused(_shaped(ifm1, shape1), _shaped(ifm2, shape2), dtype, coeff, activation,
                         slope)


qdq_elementwise_broadcasting3d_add_fused_a16 = qdq_elementwise_broadcasting3d_add_fused_a8
qdq_elementwise_broadcasting3d_sub_fused_a8 = qdq_elementwise_broadcasting3d_add_fused_a8
qdq_elementwise_broadcasting3d_sub_fused_a16 = qdq_elementwise_broadcasting3d_add_fused_a8


def qdq_elementwise_broadcasting3d_mul_fused_a8(ifm1, dtype, ifm2, shape1, shape2, coeff,
                                                activation, neg_scale):
    """coeff: the 9 fields of npu_bc3d_mul_fused_coeff."""
    return _mul_fused(_shaped(ifm1, shape1), _shaped(ifm2, shape2), dtype, coeff, activation, False,
                      neg_scale)


def qdq_elementwise_broadcasting3d_mul_fused_a16(ifm1, dtype, ifm2, shape1, shape2, coeff,
                                                 activation, neg_scale):
    return _mul_fused(_shaped(ifm1, shape1), _shaped(ifm2, shape2), dtype, coeff, activation, True,
                      neg_scale)


def _mac_fused_entry(addend_to_float):
    def model(ifm1, ifm1_type, ifm2, ifm2_type, ifm3, ifm3_type, shape1, shape23, coeff,
              activation):
        """coeff: the 8 fields of npu_bc3d_mac_fused_coeff."""
        return _mac_fused(_shaped(ifm1, shape1), _shaped(ifm2, shape23), _shaped(ifm3, shape23),
                          ifm1_type, coeff, activation, addend_to_float)
    return model


qdq_elementwise_broadcasting3d_mac_fused_a8w8 = _mac_fused_entry(True)
qdq_elementwise_broadcasting3d_mac_fused_a16w8 = _mac_fused_entry(False)
qdq_elementwise_broadcasting3d_mac_fused_a16w16 = _mac_fused_entry(True)
