# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of block_fp.h: bfp16 (9-byte blocks) and mx6 (12-byte blocks)."""

import numpy as np

from golden_int_round import shift_round_mag

F32, I64 = np.float32, np.int64


def encode_block(x, n, mbits, sub, max_shift, rnd):
    """Blocks of n fp32 values -> (mag, neg, shift, exp) per block (see block_fp.h)."""
    b = np.asarray(x, F32).view(np.uint32).astype(I64).reshape(-1, n)
    f = (b >> 23) & 0xFF
    sig = np.where(f > 0, 0x800000, 0) | (b & 0x7FFFFF)
    neg = (b >> 31) == 1
    e = np.maximum(f, 1)
    me = e.max(axis=1, keepdims=True)
    shift = np.minimum((me - e).reshape(-1, n // sub, sub).min(axis=2), max_shift)
    sh = me - e + 25 - mbits - np.repeat(shift, sub, axis=1)
    q = shift_round_mag(sig.astype(np.uint64), sh, rnd, neg).astype(I64)
    q = np.minimum(q, (1 << (mbits - 1)) - 1)
    return np.where(sh > 24, 0, q), neg, shift, me[:, 0]


def encode_bfp16(x, rnd):
    """fp32 values (a multiple of 8) -> packed bfp16; an all-zero block gets exponent 0."""
    mag, neg, _, me = encode_block(x, 8, 8, 1, 0, rnd)
    m = np.where(neg, -mag, mag).astype(np.int8).view(np.uint8)
    exp = np.where(mag.any(axis=1), me, 0).astype(np.uint8)
    return np.concatenate([exp[:, None], m], axis=1).reshape(-1)


def bfp16_fields(packed, rows):
    """packed bfp16 (rows x L/8 blocks) -> mantissas (rows x L), exponents (rows x L/8)."""
    p = np.asarray(packed, np.uint8).reshape(rows, -1, 9)
    return p[:, :, 1:].view(np.int8).astype(I64).reshape(rows, -1), p[:, :, 0].astype(I64)


def decode_bfp16(blk):
    """Exact values m * 2^(e - 133)."""
    blk = np.asarray(blk, np.uint8).reshape(-1, 9)
    m = blk[:, 1:].view(np.int8).astype(np.float64)
    with np.errstate(over="ignore"):
        return (m * np.exp2(blk[:, :1].astype(np.float64) - 133)).reshape(-1).astype(F32)


def encode_mx6(x, rnd):
    mag, neg, shift, me = encode_block(x, 16, 5, 2, 1, rnd)
    out = []
    for k in range(len(me)):
        if me[k] & 1 and not mag[k].any() and not shift[k].any():  # see block_fp.h
            out.append(np.zeros(12, np.uint8))
            continue
        w = int(me[k]) & 0xFF
        for s in range(8):
            w |= int(shift[k, s] & 1) << (8 + s)
        for j in range(16):
            w |= ((0x10 if neg[k, j] else 0) | int(mag[k, j])) << (16 + 5 * j)
        out.append(np.frombuffer(w.to_bytes(12, "little"), np.uint8))
    return np.concatenate(out) if out else np.zeros(0, np.uint8)


def mx6_fields(packed, rows):
    """packed mx6 (rows x L/16 blocks) -> signed magnitudes (rows x L), exponents
    (rows x L/16), pair shifts (rows x L/2)."""
    p = np.asarray(packed, np.uint8).reshape(-1, 12).astype(I64)
    w = [p[:, 4 * j] | p[:, 4 * j + 1] << 8 | p[:, 4 * j + 2] << 16 | p[:, 4 * j + 3] << 24
         for j in range(3)]
    e = w[0] & 0xFF
    s = np.stack([(w[0] >> (8 + j)) & 1 for j in range(8)], axis=1)
    code = []
    for k in range(16):
        bit = 16 + 5 * k
        c = w[bit // 32] >> (bit % 32)
        if bit % 32 > 27:
            c = c | (w[bit // 32 + 1] << (32 - bit % 32))
        code.append(c & 0x1F)
    code = np.stack(code, axis=1)
    m = np.where(code & 0x10, -(code & 0xF), code & 0xF)
    return m.reshape(rows, -1), e.reshape(rows, -1), s.reshape(rows, -1)


def decode_mx6(blk):
    """Values of mx6 blocks; zero magnitudes decode to +0."""
    m, e, s = mx6_fields(blk, 1)
    m, e, s = m.reshape(-1, 16), e.reshape(-1, 1), np.repeat(s.reshape(-1, 8), 2, axis=1)
    v = np.where(m == 0, 0.0, m * np.exp2((e - s - 130).astype(np.float64)))
    with np.errstate(over="ignore"):  # 15 * 2^125 and up overflow to infinity
        return v.reshape(-1).astype(F32)
