# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Puts every source directory on sys.path so tests import the shared helpers."""

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [d for d in sorted(glob.glob(os.path.join(HERE, "*", ""))) if d not in sys.path]


def pytest_runtest_setup(item):
    """Lets the fixture helper generate the case a test is parametrized with."""
    fixture = sys.modules.get("fixture")
    for value in getattr(getattr(item, "callspec", None), "params", {}).values():
        if fixture and isinstance(value, str):
            fixture.prepare(value)
