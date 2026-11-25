# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for reduction/softmax.h ([rows][C] raw element bits / integers)."""

from functools import reduce

import numpy as np

from golden_bits import neg_f32, neg_inf
from golden_float_cvt import (acc_to_bf16, bf16_to_f32, f32_to_fp8, float_to_int, kFtzIn, kSat,
                              load_float, round_float, scalar_cast,
                              store_float)
from golden_int_round import Rnd, int_range
from golden_mac import add_q, fma_q, msc_q, mul16_f32, mul_q, sub_q, vaddr_f32_reduce
from golden_minmax import vmax
from golden_mmac import fp8_decode_identity
from golden_nlf import hw_approx_ln, hw_exp2_bf20, hw_inv_f32, kExp2Bias, kLog2e
from golden_quant import fp8_in

f32 = np.float32
EVEN, AWAY = Rnd.kHalfEven, Rnd.kHalfAway
INV_LN2_BF16 = f32(1.4453125)            # bfloat16(kLog2e)
INV_LN2_F16 = f32(1.4423828125)          # float16(kLog2e)


def _row_max(x):
    """Running vmax over each row, from -inf."""
    return reduce(vmax, np.asarray(x, f32).T, neg_inf())


def _chunk_sums(e, accs):
    """Vector reduce-add per 32 columns into accumulator accs(k); returns the accumulators."""
    rows, C = e.shape
    n = -(-C // 32)
    pad = np.zeros((rows, n * 32), f32)
    pad[:, :C] = e
    sums = {}
    for k in range(n):
        a = accs(k)
        sums[a] = vaddr_f32_reduce(pad[:, 32 * k:32 * k + 32], sums.get(a, np.zeros(rows, f32)))
    return [sums[a] for a in sorted(sums)]


def _lsm_accs(C):
    """Accumulator index (2 * odd + hi) of 32-column chunk k for the logsoftmax sums."""
    full = C // 64
    pairs = 2 * (full // 2)
    return lambda k: 2 * int((k // 2) < pairs and (k // 2) % 2 == 1) + (k % 2)


def _total4(s):
    while len(s) < 4:
        s.append(np.zeros_like(s[0]))
    return add_q(add_q(add_q(s[0], s[1]), s[2]), s[3])


def softmax_bf16(bits):
    """bits [rows][C] bf16 -> bf16."""
    bits = np.asarray(bits, np.uint16)
    x = bf16_to_f32(bits)
    rows, C = x.shape
    first = bits[:, 0]
    rmax = _row_max(x)
    guard = bf16_to_f32(np.uint16(0xFF32))
    extreme = ((first == 0xFF80) | (first == 0xFF7F)) & (rmax <= guard)
    s = mul_q(x, INV_LN2_BF16)
    off = sub_q(kExp2Bias, _row_max(s))
    e = hw_exp2_bf20(add_q(round_float(s, "bf16", AWAY), off[:, None]))
    (tot,) = _chunk_sums(e, lambda k: 0)
    inv = acc_to_bf16(hw_inv_f32(tot))
    out = mul16_f32(round_float(e, "bf16", AWAY), inv[:, None], "bf16")
    alpha = hw_exp2_bf20(kExp2Bias)
    win = x == rmax[:, None]
    n = win.sum(1)
    inv_x = acc_to_bf16(hw_inv_f32(mul_q(n.astype(f32), alpha)))
    out = np.where(extreme[:, None], np.where(win, inv_x[:, None], f32(0)), out)
    return store_float(out, "bf16")


def softmax_2pass_bf16(bits, chunk_cols, num_chunks):
    """bits [rows][C] bf16 -> bf16, the row in num_chunks chunks of chunk_cols columns."""
    bits = np.asarray(bits, np.uint16)
    x = bf16_to_f32(bits)
    rows, C = x.shape
    cc = int(chunk_cols)
    P = cc * int(num_chunks)
    assert cc % 64 == 0 and C <= P
    xp = np.full((rows, P), neg_inf(), f32)
    xp[:, :C] = x
    s = mul_q(xp, INV_LN2_BF16)
    sb = round_float(s, "bf16", AWAY)
    alpha = hw_exp2_bf20(kExp2Bias)
    ia = hw_inv_f32(alpha)
    ninf = neg_inf()
    mx = np.full(rows, ninf, f32)
    sm = np.zeros(rows, f32)
    for k in range(int(num_chunks)):
        cmax = _row_max(s[:, k * cc:(k + 1) * cc])
        new = vmax(cmax, mx)
        resc = (new > mx) & (mx != ninf) & (sm != 0)
        with np.errstate(all="ignore"):
            factor = mul_q(hw_exp2_bf20(sub_q(kExp2Bias, sub_q(new, mx))), ia)
            sm = np.where(resc, mul_q(sm, factor), sm).astype(f32)
        mx = new
        off = np.where(mx == ninf, f32(0), sub_q(kExp2Bias, mx)).astype(f32)
        e = hw_exp2_bf20(add_q(sb[:, k * cc:(k + 1) * cc], off[:, None]))
        for v in range(cc // 32):
            sm = vaddr_f32_reduce(e[:, 32 * v:32 * v + 32], sm)
    ext = mx == ninf
    rmax = _row_max(x)
    win = x == rmax[:, None]
    n = win.sum(1)
    den = np.where(ext, mul_q(n.astype(f32), alpha), sm).astype(f32)
    inv = acc_to_bf16(hw_inv_f32(den))
    off = np.where(ext, f32(0), sub_q(kExp2Bias, mx)).astype(f32)
    e = hw_exp2_bf20(add_q(sb[:, :C], off[:, None]))
    out = mul16_f32(round_float(e, "bf16", AWAY), inv[:, None], "bf16")
    out = np.where(ext[:, None], np.where(win, inv[:, None], f32(0)), out)
    return store_float(out, "bf16")


def softmax_fp16x16(bits, mask, t):
    """bits [rows][C] (t = "bf16" / "f16"), mask [rows][C] bool."""
    x = load_float(bits, t)
    x = np.where(mask, x, neg_inf())
    c = INV_LN2_BF16 if t == "bf16" else INV_LN2_F16
    off = sub_q(kExp2Bias, mul_q(_row_max(x), c))
    e = hw_exp2_bf20(fma_q(x, c, off[:, None]))
    (tot,) = _chunk_sums(e, lambda k: 0)
    inv = round_float(hw_inv_f32(tot), t, EVEN, kFtzIn)
    out = mul16_f32(round_float(e, t, EVEN), inv[:, None], t)
    return store_float(out, t)


def softmax_fp8(codes, mask, ifm_scale, ofm_scale_recip, mode=EVEN):
    """codes [rows][C] fp8, mask [rows][C] bool; `mode` rounds the bf16 stores of exp2 and inv(sum)."""
    v = fp8_in(codes, ifm_scale, "bf16", AWAY, kFtzIn, row=np.shape(codes)[-1])
    s = np.where(mask, mul_q(v, INV_LN2_BF16), neg_inf())
    off = sub_q(kExp2Bias, _row_max(s))
    e = hw_exp2_bf20(add_q(s, off[:, None]))
    (tot,) = _chunk_sums(e, lambda k: 0)
    inv = round_float(hw_inv_f32(tot), "bf16", mode, kFtzIn)
    y = mul16_f32(round_float(e, "bf16", mode), inv[:, None], "bf16")
    r = scalar_cast(f32(ofm_scale_recip), "bf16")  # scalar (bfloat16) cast
    q = mul_q(y, r)
    q = np.where(q == 0, f32(0), q)
    return f32_to_fp8(q, AWAY, kSat)


def _lsm_core(x, scaled, exp_scale):
    """Row max of `scaled`, exp2 of fma(x, exp_scale, off), the four sums, ln."""
    rmax = _row_max(scaled)
    off = sub_q(kExp2Bias, mul_q(rmax, kLog2e))
    e = hw_exp2_bf20(fma_q(x, exp_scale, off[:, None]))
    tot = _total4(_chunk_sums(e, _lsm_accs(x.shape[1])))
    return rmax, hw_approx_ln(tot)


def logsoftmax_bf16(bits, t):
    """bits [rows][C] (t = "bf16" / "f16")."""
    x = load_float(bits, t)
    rmax, ln = _lsm_core(x, x, kLog2e)
    o = sub_q(neg_f32(rmax), ln)
    return store_float(add_q(x, o[:, None]), t)


def logsoftmax_fp8(codes, ifm_scale, ofm_scale_recip):
    """codes [rows][C] fp8."""
    v = fp8_decode_identity(codes, row=np.shape(codes)[-1])
    in_scale, out_scale_inv = f32(ifm_scale), f32(ofm_scale_recip)
    rmax, ln = _lsm_core(v, mul_q(v, in_scale), mul_q(in_scale, kLog2e))
    o = mul_q(sub_q(neg_f32(rmax), ln), out_scale_inv)
    return f32_to_fp8(fma_q(v, mul_q(in_scale, out_scale_inv), o[:, None]))


def qdq_softmax_a8(q, t, mask, path, shift_res, inv_ln2, ofm_scale, zp_acc):
    """q [rows][C] integers of type t, mask bool; path 0, 1 or 2 (C == 32)."""
    q = np.asarray(q, np.int64)
    qf = q.astype(f32)  # exact
    rows, C = q.shape
    inv_ln2, ofm_scale, zp_acc = f32(inv_ln2), f32(ofm_scale), f32(zp_acc)
    lo = int_range(t)[0]
    if path == 2:
        v = np.where(mask, mul_q(qf, inv_ln2), neg_inf())
        e = hw_exp2_bf20(sub_q(v, sub_q(_row_max(v), kExp2Bias)[:, None]))
        (tot,) = _chunk_sums(e, lambda k: 0)
        inv = hw_inv_f32(mul_q(tot, ofm_scale))
    else:
        qmax = np.where(mask, q, lo).max(1)
        off = msc_q(qmax.astype(f32), inv_ln2, kExp2Bias)
        e = hw_exp2_bf20(np.where(mask, fma_q(qf, inv_ln2, off[:, None]), neg_inf()))
        s = _chunk_sums(e, (lambda k: k % 2) if path == 0 else (lambda k: 0))
        tot = add_q(s[0], s[1]) if path == 0 and len(s) > 1 else s[0]
        inv = round_float(hw_inv_f32(mul_q(tot, ofm_scale)), "f16")
        e = round_float(e, "f16")
    return float_to_int(fma_q(e, inv[:, None], zp_acc), t, shift_res)


qdq_softmax_a16 = qdq_softmax_a8


def qdq_logsoftmax_a8(q, t_in, mask, t_out, ofm_zero_point, ifm_scale, ofm_scale):
    """q [rows][C] integers (C % 128 == 0), mask bool; masked outputs are the zero point."""
    ofm_zp = ofm_zero_point
    q = np.asarray(q, np.int64)
    qf = q.astype(f32)  # exact
    rows, C = q.shape
    ifm_scale = f32(ifm_scale)
    osi = hw_inv_f32(f32(ofm_scale))
    inv_ln2 = mul_q(kLog2e, ifm_scale)
    io = mul_q(ifm_scale, osi)
    zp = f32(ofm_zp)
    qmax = np.where(mask, q, int_range(t_in)[0]).max(1).astype(f32)
    off = msc_q(qmax, inv_ln2, kExp2Bias)
    qio = msc_q(qmax, io, zp)
    e = hw_exp2_bf20(np.where(mask, fma_q(qf, inv_ln2, off[:, None]), neg_inf()))
    s = _chunk_sums(e, lambda k: 2 * ((k // 2) % 2) + (k % 2))
    nl = msc_q(hw_approx_ln(_total4(s)), osi, qio)
    res = np.where(mask, fma_q(qf, io, nl[:, None]), zp)
    return float_to_int(res, t_out)


qdq_logsoftmax_a16 = qdq_logsoftmax_a8
