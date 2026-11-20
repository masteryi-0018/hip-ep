# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Helpers shared by the tests of the QDQ integer kernels: random operands, calls, comparison."""

import numpy as np

import golden_act as gm
from testing import INT as T
from testing import assert_same, rand_ints, rejected, run  # noqa: F401

RNG = np.random.default_rng(2028)
f32 = np.float32


def rnd(t, shape, lo=None, hi=None):
    return rand_ints(RNG, T[t][1], shape, lo, hi)


ACTS = [(gm.ACT_NONE, 0.0, 0.0), (gm.ACT_CLIP, -3.5, 1000.5), (gm.ACT_LRELU, 0.1015625, -3.59375),
        (gm.ACT_LRELU, 1.5, 2.0)]
