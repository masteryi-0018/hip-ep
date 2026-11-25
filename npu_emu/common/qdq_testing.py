# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Helpers shared by the tests of the QDQ integer kernels: random operands, calls, comparison."""

import numpy as np

import golden_quant as gq
from testing import INT as T
from testing import assert_same, rand_ints, rejected, run  # noqa: F401

RNG = np.random.default_rng(2028)
f32 = np.float32


def rnd(t, shape, lo=None, hi=None):
    return rand_ints(RNG, T[t][1], shape, lo, hi)


ACTS = [(gq.kQdqActNone, 0.0, 0.0), (gq.kQdqActClip, -3.5, 1000.5),
        (gq.kQdqActLrelu, 0.1015625, -3.59375),
        (gq.kQdqActLrelu, 1.5, 2.0)]

# float words for the special-value tests: NaNs, infinities, zeros, subnormals, extremes, 1
SPECIAL = np.array([0x7FC00000, 0xFFC12345, 0x7F800001, 0x7F800000, 0xFF800000, 0, 0x80000000,
                    1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000],
                   np.uint32).view(f32)
NS = len(SPECIAL)
