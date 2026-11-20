# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for reduction/softmax.h ([rows][C] raw element bits / integers)."""

from functools import reduce

import numpy as np

from golden_float_cvt import (bf16_to_f32, f32_to_fp8, fp8_decode_identity, kFtzIn, kSat,
                              load_float, round_float, scalar_cast, store_float)
from golden_int_round import Rnd, float_to_int, int_range
from golden_mac import add, fma, mac_exact, mul, mul16_f32, vaddr_f32_reduce
from golden_minmax import vmax
from golden_nlf import hw_approx_ln, hw_exp2_bf20, hw_inv_f32

f32 = np.float32
EVEN, AWAY = Rnd.kHalfEven, Rnd.kHalfAway
LOG2E = f32(1.44269502162933349609375)   # 0x3fb8aa3b
BIAS = f32(895.001)                      # 0x445fc010
INV_LN2_BF16 = f32(1.4453125)            # bfloat16(1.44269504f)
INV_LN2_F16 = f32(1.4423828125)          # float16(1.44269504f)


def row_max(x):
    """Running vmax over each row, from -inf."""
    return reduce(vmax, np.asarray(x, f32).T, f32(-np.inf))


def chunk_sums(e, accs):
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


def lsm_accs(C):
    """Accumulator index (2 * odd + hi) of 32-column chunk k for the logsoftmax sums."""
    full = C // 64
    pairs = 2 * (full // 2)
    return lambda k: 2 * int((k // 2) < pairs and (k // 2) % 2 == 1) + (k % 2)


def total4(s):
    while len(s) < 4:
        s.append(np.zeros_like(s[0]))
    return add(add(add(s[0], s[1]), s[2]), s[3])


def softmax_bf16(bits):
    """bits [rows][C] bf16 -> bf16."""
    bits = np.asarray(bits, np.uint16)
    x = bf16_to_f32(bits)
    rows, C = x.shape
    first = bits[:, 0]
    rmax = row_max(x)
    guard = bf16_to_f32(np.uint16(0xFF32))
    extreme = ((first == 0xFF80) | (first == 0xFF7F)) & (rmax <= guard)
    s = mul(x, INV_LN2_BF16)
    off = add(BIAS, -row_max(s))
    e = hw_exp2_bf20(add(round_float(s, "bf16", AWAY), off[:, None]))
    (tot,) = chunk_sums(e, lambda k: 0)
    inv = round_float(hw_inv_f32(tot), "bf16", AWAY, kFtzIn)
    out = mul16_f32(round_float(e, "bf16", AWAY), inv[:, None], "bf16")
    alpha = hw_exp2_bf20(BIAS)
    win = x == rmax[:, None]
    n = win.sum(1)
    inv_x = round_float(hw_inv_f32(mul(n.astype(f32), alpha)), "bf16", AWAY, kFtzIn)
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
    xp = np.full((rows, P), -np.inf, f32)
    xp[:, :C] = x
    s = mul(xp, INV_LN2_BF16)
    sb = round_float(s, "bf16", AWAY)
    alpha = hw_exp2_bf20(BIAS)
    ia = hw_inv_f32(alpha)
    ninf = f32(-np.inf)
    mx = np.full(rows, ninf, f32)
    sm = np.zeros(rows, f32)
    for k in range(int(num_chunks)):
        cmax = row_max(s[:, k * cc:(k + 1) * cc])
        new = vmax(cmax, mx)
        resc = (new > mx) & (mx != ninf) & (sm != 0)
        with np.errstate(all="ignore"):
            factor = mul(hw_exp2_bf20(add(BIAS, -add(new, -mx))), ia)
            sm = np.where(resc, mul(sm, factor), sm).astype(f32)
        mx = new
        off = np.where(mx == ninf, f32(0), add(BIAS, -mx)).astype(f32)
        e = hw_exp2_bf20(add(sb[:, k * cc:(k + 1) * cc], off[:, None]))
        for v in range(cc // 32):
            sm = vaddr_f32_reduce(e[:, 32 * v:32 * v + 32], sm)
    ext = mx == ninf
    rmax = row_max(x)
    win = x == rmax[:, None]
    n = win.sum(1)
    den = np.where(ext, mul(n.astype(f32), alpha), sm).astype(f32)
    inv = round_float(hw_inv_f32(den), "bf16", AWAY, kFtzIn)
    off = np.where(ext, f32(0), add(BIAS, -mx)).astype(f32)
    e = hw_exp2_bf20(add(sb[:, :C], off[:, None]))
    out = mul16_f32(round_float(e, "bf16", AWAY), inv[:, None], "bf16")
    out = np.where(ext[:, None], np.where(win, inv[:, None], f32(0)), out)
    return store_float(out, "bf16")


def softmax_fp16x16(bits, mask, t):
    """bits [rows][C] (t = "bf16" / "f16"), mask [rows][C] bool."""
    x = load_float(bits, t)
    x = np.where(mask, x, f32(-np.inf))
    c = INV_LN2_BF16 if t == "bf16" else INV_LN2_F16
    off = add(BIAS, -mul(row_max(x), c))
    e = hw_exp2_bf20(fma(x, c, off[:, None]))
    (tot,) = chunk_sums(e, lambda k: 0)
    inv = round_float(hw_inv_f32(tot), t, EVEN, kFtzIn)
    out = mul16_f32(round_float(e, t, EVEN), inv[:, None], t)
    return store_float(out, t)


def softmax_fp8(codes, mask, ifm_scale, ofm_scale_recip, mode=EVEN):
    """codes [rows][C] fp8, mask [rows][C] bool; `mode` rounds the bf16 stores of exp2 and inv(sum)."""
    v = fp8_decode_identity(codes, row=np.shape(codes)[-1])
    v = round_float(mul(v, ifm_scale), "bf16", AWAY, kFtzIn)
    s = np.where(mask, mul(v, INV_LN2_BF16), f32(-np.inf))
    off = add(BIAS, -row_max(s))
    e = hw_exp2_bf20(add(s, off[:, None]))
    (tot,) = chunk_sums(e, lambda k: 0)
    inv = round_float(hw_inv_f32(tot), "bf16", mode, kFtzIn)
    y = mul16_f32(round_float(e, "bf16", mode), inv[:, None], "bf16")
    r = scalar_cast(f32(ofm_scale_recip), "bf16")  # scalar (bfloat16) cast
    q = mul(y, r)
    q = np.where(q == 0, f32(0), q)
    return f32_to_fp8(q, AWAY, kSat)


def _lsm_core(x, scaled, exp_scale):
    """Row max of `scaled`, exp2 of fma(x, exp_scale, off), the four sums, ln."""
    rmax = row_max(scaled)
    off = add(BIAS, -mul(rmax, LOG2E))
    e = hw_exp2_bf20(fma(x, exp_scale, off[:, None]))
    tot = total4(chunk_sums(e, lsm_accs(x.shape[1])))
    return rmax, hw_approx_ln(tot)


def logsoftmax_bf16(bits, t):
    """bits [rows][C] (t = "bf16" / "f16")."""
    x = load_float(bits, t)
    rmax, ln = _lsm_core(x, x, LOG2E)
    o = add(-rmax, -ln)
    return store_float(add(x, o[:, None]), t)


def logsoftmax_fp8(codes, in_scale, out_scale_inv):
    """codes [rows][C] fp8."""
    v = fp8_decode_identity(codes, row=np.shape(codes)[-1])
    in_scale, out_scale_inv = f32(in_scale), f32(out_scale_inv)
    rmax, ln = _lsm_core(v, mul(v, in_scale), mul(in_scale, LOG2E))
    o = mul(add(-rmax, -ln), out_scale_inv)
    return f32_to_fp8(fma(v, mul(in_scale, out_scale_inv), o[:, None]))


def qdq_softmax(q, mask, t, path, shift_res, inv_ln2, ofm_scale, zp_acc):
    """q [rows][C] integers of type t, mask bool; path 0, 1 or 2 (C == 32)."""
    q = np.asarray(q, np.int64)
    rows, C = q.shape
    inv_ln2, ofm_scale, zp_acc = f32(inv_ln2), f32(ofm_scale), f32(zp_acc)
    lo = int_range(t)[0]
    if path == 2:
        v = np.where(mask, mac_exact(q, inv_ln2, f32(0)), f32(-np.inf))
        e = hw_exp2_bf20(add(v, -add(row_max(v), -BIAS)[:, None]))
        (tot,) = chunk_sums(e, lambda k: 0)
        inv = hw_inv_f32(mul(tot, ofm_scale))
    else:
        qmax = np.where(mask, q, lo).max(1)
        off = mac_exact(qmax, -inv_ln2, BIAS)
        e = hw_exp2_bf20(np.where(mask, mac_exact(q, inv_ln2, off[:, None]), f32(-np.inf)))
        s = chunk_sums(e, (lambda k: k % 2) if path == 0 else (lambda k: 0))
        tot = add(s[0], s[1]) if path == 0 and len(s) > 1 else s[0]
        inv = round_float(hw_inv_f32(mul(tot, ofm_scale)), "f16")
        e = round_float(e, "f16")
    return float_to_int(fma(e, inv[:, None], zp_acc), t, shift_res)


def qdq_logsoftmax(q, mask, t_in, t_out, ifm_scale, ofm_scale, ofm_zp):
    """q [rows][C] integers (C % 128 == 0), mask bool; masked outputs are the zero point."""
    q = np.asarray(q, np.int64)
    rows, C = q.shape
    ifm_scale = f32(ifm_scale)
    osi = hw_inv_f32(f32(ofm_scale))
    inv_ln2 = mul(LOG2E, ifm_scale)
    io = mul(ifm_scale, osi)
    zp = f32(ofm_zp)
    qmax = np.where(mask, q, int_range(t_in)[0]).max(1)
    off = mac_exact(qmax, -inv_ln2, BIAS)
    qio = mac_exact(qmax, -io, zp)
    e = hw_exp2_bf20(np.where(mask, mac_exact(q, inv_ln2, off[:, None]), f32(-np.inf)))
    s = chunk_sums(e, lambda k: 2 * ((k // 2) % 2) + (k % 2))
    nl = fma(hw_approx_ln(total4(s)), -osi, qio)
    res = np.where(mask, mac_exact(q, io, nl[:, None]), zp)
    return float_to_int(res, t_out)
