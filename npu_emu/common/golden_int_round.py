# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of int_round.h: rounding modes, shift-round, saturation, wrap,
shift-round-saturate."""

from enum import IntEnum

import numpy as np

I64, U64 = np.int64, np.uint64


class Rnd(IntEnum):
    kHalfEven = 0
    kHalfAway = 1
    kFloor = 2
    kCeil = 3
    kTrunc = 4
    kAwayFromZero = 5
    kHalfDown = 6
    kHalfUp = 7
    kHalfZero = 8
    kHalfOdd = 9


INT_RANGE = {"i8": (-128, 127), "u8": (0, 255), "i16": (-32768, 32767), "u16": (0, 65535),
             "i32": (-2**31, 2**31 - 1), "u32": (0, 2**32 - 1)}


def round_up(rnd, neg, odd, half, rest):
    """Whether the truncated magnitude is incremented (see int_round.h)."""
    neg, odd, half, rest = (np.asarray(v, bool) for v in (neg, odd, half, rest))
    if rnd == Rnd.kFloor:
        return neg & (half | rest)
    if rnd == Rnd.kCeil:
        return ~neg & (half | rest)
    if rnd == Rnd.kTrunc:
        return np.zeros(np.broadcast(neg, odd, half, rest).shape, bool)
    if rnd == Rnd.kAwayFromZero:
        return half | rest
    tie = {Rnd.kHalfEven: odd, Rnd.kHalfOdd: ~odd, Rnd.kHalfAway: np.True_,
           Rnd.kHalfZero: np.False_, Rnd.kHalfUp: ~neg, Rnd.kHalfDown: neg}[Rnd(rnd)]
    return half & (rest | tie)


def rounds_to_inf(rnd, neg):
    """Whether an overflow rounds to infinity (else to the largest finite value)."""
    neg = np.asarray(neg, bool)
    if rnd == Rnd.kTrunc:
        return np.zeros(neg.shape, bool)
    if rnd == Rnd.kFloor:
        return neg
    if rnd == Rnd.kCeil:
        return ~neg
    return np.ones(neg.shape, bool)


def shift_round_mag(mag, s, rnd, neg):
    """uint64 mag * 2^-s rounded with rnd for a value of sign neg; s <= 0 shifts left,
    saturating at 2^64 - 1."""
    with np.errstate(all="ignore"):
        return _shift_round_mag(mag, s, rnd, neg)


def _shift_round_mag(mag, s, rnd, neg):
    mag, s, neg = np.broadcast_arrays(np.asarray(mag, U64), np.asarray(s, I64),
                                      np.asarray(neg, bool))
    sr = np.clip(s, 1, 64).astype(U64)                         # right shift amount
    q = np.where(sr < 64, mag >> np.minimum(sr, U64(63)), U64(0))
    half = ((mag >> (sr - U64(1))) & U64(1)) == 1
    low = np.where(sr > 1, mag & ((U64(1) << np.minimum(sr - U64(1), U64(63))) - U64(1)), U64(0))
    rest = low != 0
    half = np.where(s > 64, False, half)
    rest = np.where(s > 64, mag != 0, rest)
    right = q + round_up(rnd, neg, (q & U64(1)) == 1, half, rest).astype(U64)
    u = np.clip(-s, 0, 63).astype(U64)
    over = (-s >= 64) | ((-s > 0) & ((mag >> (U64(64) - np.maximum(u, U64(1)))) != 0))
    left = np.where(over & (mag != 0), ~U64(0), mag << u)
    return np.where(s <= 0, left, right).astype(U64)


def shift_round(v, s, rnd):
    """int64 v * 2^-s rounded with rnd; s <= 0 shifts left, saturating to int64."""
    v = np.asarray(v, I64)
    neg = v < 0
    with np.errstate(all="ignore"):  # uint64 wraparound of the two's complement is intended
        mag = np.where(neg, ~v.astype(U64) + U64(1), v.astype(U64))
        q = shift_round_mag(mag, s, rnd, neg)
        big = q >= U64(1 << 63)
        pos = np.where(big, U64((1 << 63) - 1), q).astype(I64)
        negv = np.where(big, np.iinfo(I64).min, -(q & U64((1 << 63) - 1)).astype(I64))
        return np.where(neg, negv, pos).astype(I64)


def sat_add64(a, b):
    """a + b saturated to the int64 range."""
    a, b = np.broadcast_arrays(np.asarray(a, I64), np.asarray(b, I64))
    with np.errstate(all="ignore"):
        r = a + b  # wraps
    overflow = ((a ^ r) & (b ^ r)) < 0
    return np.where(overflow, np.where(a < 0, np.iinfo(I64).min, np.iinfo(I64).max), r).astype(I64)


def wrap_mul64(a, b):
    """a * b in two's complement, wrapping on overflow."""
    with np.errstate(all="ignore"):
        return (np.asarray(a, I64).astype(U64) * np.asarray(b, I64).astype(U64)).astype(I64)


def wrap_add64(a, b):
    """a + b in two's complement, wrapping on overflow."""
    with np.errstate(all="ignore"):
        return (np.asarray(a, I64).astype(U64) + np.asarray(b, I64).astype(U64)).astype(I64)


def int_range(t):
    """(lo, hi) of the integer type t in the int64 domain (u64 values >= 2^63 are not
    representable)."""
    if t == "i64":
        return -2**63, 2**63 - 1
    if t == "u64":
        return 0, 2**63 - 1
    return INT_RANGE[t]


def clamp64(v, lo, hi):
    """As the C helper: v < lo gives lo, else v > hi gives hi (for lo > hi: lo below lo, else hi)."""
    v = np.asarray(v, I64)
    return np.where(v < lo, lo, np.where(v > hi, hi, v)).astype(I64)


def saturate(v, t):
    return clamp64(v, *int_range(t))


def saturate_sym(v, t):
    """Symmetric saturation: a signed t clamps to [-max, max]; an unsigned t as saturate."""
    lo, hi = int_range(t)
    return clamp64(v, -hi if lo < 0 else lo, hi)


def wrap(v, t):
    """v wrapped (two's complement) to the width of t."""
    v = np.asarray(v, I64)
    if t in ("i64", "u64"):
        return v
    return v.astype({"i8": np.int8, "u8": np.uint8, "i16": np.int16, "u16": np.uint16,
                     "i32": np.int32, "u32": np.uint32}[t]).astype(I64)


def srs(v, s, t, rnd=Rnd.kHalfEven):
    """v * 2^-s rounded with rnd, then saturated to t."""
    return saturate(shift_round(v, s, rnd), t)
