# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy mirror of grid.h: the QDQ GridSample grid coordinate."""

import numpy as np

from golden_mac import add, fma, mul

f32 = np.float32


def qdq_grid_coord(code, grid_zp, grid_scale, extent, align_corners):
    """Grid codes -> fp32 pixel coordinates on an axis of `extent` pixels."""
    g = add(mul(add(np.asarray(code, f32), -f32(grid_zp)), f32(grid_scale)), f32(1.0))
    if align_corners == 0:
        return fma(g, mul(f32(extent), f32(0.5)), f32(-0.5))
    return mul(mul(f32(extent - 1), g), f32(0.5))
