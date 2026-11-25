# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of mmac.h: the accumulate / normalise stages, the bf16 / bfp16 / mx6 / fp8
MMACs, the single-value MMACs and the decodes through an identity multiply."""

import numpy as np

from golden_bits import f32_bits, qnan
from golden_float_cvt import f32_to_bf16, fp8_to_f32
from golden_int_round import Rnd, shift_round
from golden_mac import add_q

F32, I64 = np.float32, np.int64


def bf16_sig(b):
    """(signed significand, exponent); a subnormal gets exponent 1 and no hidden bit."""
    b = np.asarray(b, np.uint16).astype(I64)
    e = (b >> 7) & 0xFF
    m = b & 0x7F
    man = m | np.where(e != 0, 0x80, 0)
    e = np.where((e == 0) & (m != 0), 1, e)
    man = np.where(b & 0x8000, -man, man)
    return man, e


def lzd32(m):
    """Leading-zero count of a 32-bit value; 63 for 0."""
    m = np.asarray(m, I64)
    out = np.full(m.shape, 63, I64)
    nz = m != 0
    bl = np.zeros(m.shape, I64)
    bl[nz] = np.floor(np.log2(m[nz].astype(np.float64))).astype(I64) + 1
    # fix float log2 off-by-one near powers of two
    hi = nz & ((m >> np.maximum(bl, 0).clip(max=62)) > 0)
    bl[hi] += 1
    lo = nz & (bl > 0) & ((m >> (bl - 1).clip(min=0, max=62)) == 0)
    bl[lo] -= 1
    out[nz] = 32 - bl[nz]
    return out


def merge_halves(acc_bits, halves):
    """The fp32 accumulator plus two half sums [(m, ex), (m, ex)], aligned to the largest
    exponent, rounded half to even, then normalised once."""
    a_m, acc_e = acc_significand(acc_bits)
    a_ex = acc_e - 127
    (l_m, l_ex), (h_m, h_ex) = halves
    exp = np.where((a_ex > l_ex) & (a_ex > h_ex), a_ex, np.where(l_ex > h_ex, l_ex, h_ex))
    man = shift_round(a_m, exp - a_ex, Rnd.kHalfEven)
    for m, ex in ((l_m, l_ex), (h_m, h_ex)):
        man = man + shift_round(m, exp - ex, Rnd.kHalfEven)
    return flp_nrm(man << 1, exp + 134)


def mmac_bf16(acc_bits, x, z):
    """acc_bits: fp32 bits (...,), x, z: bf16 bits (..., 32): two halves of 16 products, each
    half aligned to its largest exponent. Returns fp32 bits."""
    xm, xe = bf16_sig(x)
    zm, ze = bf16_sig(z)
    pm = xm * zm                                          # 17-bit signed product
    pe = np.where((xe == 0) | (ze == 0), 0, xe + ze)      # u9
    halves = []
    for h in (slice(0, 16), slice(16, 32)):
        cmax = pe[..., h].max(axis=-1)
        r = shift_round(pm[..., h] << 9, cmax[..., None] - pe[..., h], Rnd.kHalfEven)
        halves.append((r.sum(-1), cmax - 254))
    return merge_halves(acc_bits, halves)


def mmac_plain(v):
    """Zero or a normal finite fp32. The single-value MMACs are modelled only when both
    operands and the fp32 sum are plain; otherwise they keep the fp32 add."""
    e = (f32_bits(v) >> 23) & 0xFF
    return (np.asarray(v, F32) == 0) | ((e != 0) & (e != 0xFF))


def mmac_add_bf16(acc, x):
    """acc + x as one bf16 MMAC of x against a staged 1.0; x not exact in bf16 (or non-plain
    values) keep the fp32 add."""
    acc, x = np.broadcast_arrays(np.asarray(acc, F32), np.asarray(x, F32))
    with np.errstate(all="ignore"):
        f = add_q(acc, x)
    xx = np.zeros(x.shape + (32,), np.uint16)
    xx[..., 0] = bf16_bits(x)
    zz = np.zeros_like(xx)
    zz[..., 0] = 0x3F80
    r = mmac_bf16(f32_bits(acc), xx, zz).view(F32).reshape(x.shape)
    exact = (f32_bits(x) & 0xFFFF) == 0
    return np.where(mmac_plain(acc) & mmac_plain(x) & mmac_plain(f) & exact, r,
                    f).astype(F32)


def bf16_bits(x):
    """bf16 bit patterns of values stored as float32 (exactly representable in bf16)."""
    return (np.asarray(x, F32).view(np.uint32) >> 16).astype(np.uint16)


def flp_nrm(man, dexp):
    sgn = (man < 0).astype(I64)
    m32 = np.abs(man) & 0xFFFFFFFF
    zc = lzd32(m32)
    zero = zc >= 32
    sh = zc & 31
    m32 = (m32 << sh) & 0xFFFFFFFF
    exp = dexp - sh
    stk = (m32 & 0x7F) != 0
    grd, lsb = (m32 >> 7) & 1, (m32 >> 8) & 1
    eup = ((m32 >> 7) & 0xFFFFFF) == 0xFFFFFF
    mant = (((m32 >> 8) & 0x7FFFFF) + (grd & (stk | lsb))) & 0x7FFFFF
    exp = exp + eup
    r = (sgn << 31) | ((exp & 0xFF) << 23) | mant
    r = np.where((exp >= 255) & ~zero, (sgn << 31) | 0x7F800000, r)
    return np.where(((exp <= 0) & ~zero) | zero, 0, r).astype(np.uint32)


def acc_significand(acc_bits):
    """The fp32 accumulator as a signed 24-bit significand (subnormals flushed) and the
    exponent field."""
    acc = np.asarray(acc_bits, np.uint32).astype(I64)
    e = (acc >> 23) & 0xFF
    m = np.where(e != 0, (1 << 23) | (acc & 0x7FFFFF), 0)
    return np.where((acc >> 31) == 1, -m, m), e


def block_acc(acc_bits, s, ce):
    """Accumulate + normalise of the 8-block MMACs: s (..., 8) exact block sums at
    exponents ce (..., 8)."""
    a_m, acc_e = acc_significand(acc_bits)
    a_e = acc_e + 127
    exp = np.maximum(a_e, ce.max(axis=-1))
    man = shift_round(a_m, exp - a_e, Rnd.kHalfEven)
    r = shift_round(np.asarray(s, I64) << 11, exp[..., None] - ce, Rnd.kHalfEven)
    return flp_nrm(man + r.sum(-1), exp - 0x77)


def mmac_bfp16(acc_bits, xm, xe, zm, ze):
    """One bfp16 block MMAC per element (broadcasting): x / z mantissas (..., 64), block
    exponents (..., 8); block b adds the exact sum of its 8 products at xe + ze."""
    xm, zm = np.asarray(xm, I64), np.asarray(zm, I64)
    s = (xm * zm).reshape(np.broadcast_shapes(xm.shape, zm.shape)[:-1] + (8, 8)).sum(-1)
    return block_acc(acc_bits, s, np.asarray(xe, I64) + np.asarray(ze, I64))


def mmac_mx6(acc_bits, xm, xs, xe, zm, zs, ze):
    """One mx6 block MMAC per element (broadcasting): magnitudes (..., 128), pair shifts
    (..., 64), exponents (..., 8); each pair sum is shifted left by 2 - (xs + zs), block sums
    are scaled by 16."""
    xm, zm = np.asarray(xm, I64), np.asarray(zm, I64)
    p = xm * zm
    shift = 2 - np.asarray(xs, I64) - np.asarray(zs, I64)
    pair = p.reshape(p.shape[:-1] + (64, 2)).sum(-1) << shift
    s = pair.reshape(pair.shape[:-1] + (8, 8)).sum(-1)
    return block_acc(acc_bits, s << 4, np.asarray(xe, I64) + np.asarray(ze, I64))


def fp8_sig(b):
    """fp8 (E4M3) signed significand and exponent (a subnormal gets 1, a zero 0)."""
    b = np.asarray(b, np.uint8).astype(I64)
    e = (b >> 3) & 0xF
    m = b & 7
    sig = m | np.where(e != 0, 8, 0)
    e = np.where((e == 0) & (m != 0), 1, e)
    return np.where(b & 0x80, -sig, sig), e


def mmac_fp8(acc_bits, x, z):
    """acc (...,), x / z (..., 64) fp8 codes: two halves of 32 products, each product rounded
    half to even at its half's largest exponent."""
    xm, xe = fp8_sig(x)
    zm, ze = fp8_sig(z)
    pm, pe = xm * zm, np.where((xe == 0) | (ze == 0), 0, xe + ze)
    halves = []
    for h in (slice(0, 32), slice(32, 64)):
        cmax = pe[..., h].max(axis=-1)
        r = shift_round(pm[..., h] << 17, cmax[..., None] - pe[..., h], Rnd.kHalfEven)
        halves.append((r.sum(-1), cmax - 14))
    return merge_halves(acc_bits, halves)


def bfp16_identity_acc(acc_bits, blk, row=None):
    """The accumulators acc_bits (one per value) plus each value of the bfp16 blocks against a
    staged 1.0 (64 * 2^-6), the other lanes zero: one block MMAC per value over its group of 8
    blocks. blk is split into rows of `row` blocks (default: one row), each row into groups
    from its start (the last one may be short). Flat fp32 bits."""
    blk = np.asarray(blk, np.uint8).reshape(-1, 9)
    row = row or len(blk)
    rows = len(blk) // row
    e = np.concatenate([blk[:, 0].astype(I64).reshape(rows, row),
                        np.zeros((rows, (-row) % 8), I64)], axis=1)
    ce = e.reshape(rows, -1, 8)[:, np.arange(row) // 8] + 127  # the group of each block
    ce = np.broadcast_to(ce[:, :, None, :], (rows, row, 8, 8))  # (row, block, value, 8)
    m = blk[:, 1:].view(np.int8).astype(I64).reshape(rows, row, 8)
    s = np.zeros((rows, row, 8, 8), I64)
    s[:, np.arange(row), :, np.arange(row) % 8] = m.transpose(1, 0, 2) * 64
    acc = np.broadcast_to(np.asarray(acc_bits, np.uint32).reshape(-1), (rows * row * 8,))
    return block_acc(acc, s.reshape(-1, 8), ce.reshape(-1, 8))


def mmac_add_bfp16(x, a, acc, C):
    """acc + x (decoded: a) as one identity block MMAC per value onto the fp32 accumulator
    acc. x: rows of C values in bfp16 blocks, 64-value groups from each row start. Flat."""
    a = np.asarray(a, F32).reshape(-1)
    acc = np.asarray(acc, F32).reshape(-1)
    with np.errstate(all="ignore"):
        f = add_q(a, acc)
    r = bfp16_identity_acc(f32_bits(acc), x, C // 8).view(F32)
    return np.where(mmac_plain(a) & mmac_plain(acc) & mmac_plain(f), r, f).astype(F32)


# ---- decodes through an identity multiply ----

def decode_bfp16_identity(blk, row=None):
    """bfp16 identity-multiply decode: the identity MMAC onto a zero accumulator. blk is split
    into rows of `row` blocks (default: one row) and each row into 64-element groups (8 blocks,
    the last one may be short) from its start; values are rounded half to even to multiples of
    2^(E - 150), E the largest block exponent of the group. A zero, or a result below 2^-126,
    is +0."""
    blk = np.asarray(blk, np.uint8).reshape(-1, 9)
    if not len(blk):
        return np.zeros(0, F32)
    return bfp16_identity_acc(np.zeros(1, np.uint32), blk, row).view(F32)


def decode_bfp16_group(blk):
    """decode_bfp16_identity; a block exponent 255 makes the whole group the NaN 0x7f810000."""
    blk = np.asarray(blk, np.uint8).reshape(-1, 9)
    nb = len(blk)
    e = np.concatenate([blk[:, 0], np.zeros((-nb) % 8, np.uint8)]).reshape(-1, 8)
    nan = np.repeat((e == 255).any(axis=1), 64)[:8 * nb]
    v = decode_bfp16_identity(blk)
    v[nan] = np.array([0x7F810000], np.uint32).view(F32)[0]
    return v


def decode_bfp16_bf16(blk):
    """decode_bfp16_group rounded half to even to bf16 codes."""
    return f32_to_bf16(decode_bfp16_group(blk), Rnd.kHalfEven)


def fp8_row_nan(c, row=None):
    """Whether each element's group of 8 (from the start of each row of `row` elements,
    default one row) holds a NaN code."""
    c = np.asarray(c, np.uint8)
    if c.size == 0:
        return np.zeros(c.shape, bool)
    n = row or c.size
    r = ((c & 0x7F) == 0x7F).reshape(-1, n)
    pad = np.concatenate([r, np.zeros((len(r), (-n) % 8), bool)], axis=1).reshape(len(r), -1, 8)
    g = np.repeat(pad.any(axis=2), 8, axis=1)[:, :n]
    return g.reshape(c.shape)


def fp8_decode_identity(c, group_nan=None, row=None):
    """fp8 decode through the identity multiply: a NaN group gives 0x7fc00000, -0 -> +0.
    group_nan: per-element flag; default the row groups of fp8_row_nan."""
    if group_nan is None:
        group_nan = fp8_row_nan(c, row)
    v = (fp8_to_f32(c) + F32(0.0)).astype(F32)
    return np.where(group_nan, qnan(), v).astype(F32)
