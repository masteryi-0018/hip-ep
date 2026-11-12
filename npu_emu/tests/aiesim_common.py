# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Shared helpers for the aiesim cross-check suites (test_aiesim*.py).

Cases come from an mllib nightly regression tree, NPU_EMU_AIESIM_ROOT
(.../Vitis_Libraries/mllib/L2/regression/wrapper); by default the newest
nightly under /wrk/mllib_regr that kept aiesim outputs. Every hex file in a
case's data/ is one 32-bit word per line, little-endian, so it reads as a plain
byte stream."""

import glob
import os
import re

import numpy as np
import pytest

from npu import (NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16, NPU_I32, NPU_I64,
                 NPU_U8, NPU_U16, NPU_U32, NPU_U64)

NIGHTLY = "/wrk/mllib_regr/jenkins/workspace/regression/AIELibs/mllib/nightly"
TYPES = {"float": (NPU_F32, np.float32), "float16": (NPU_F16, np.uint16),
         "bfloat16": (NPU_BF16, np.uint16), "float8": (NPU_FP8, np.uint8),
         "int8": (NPU_I8, np.int8), "uint8": (NPU_U8, np.uint8),
         "int16": (NPU_I16, np.int16), "uint16": (NPU_U16, np.uint16),
         "int32": (NPU_I32, np.int32), "uint32": (NPU_U32, np.uint32),
         "int64": (NPU_I64, np.int64), "uint64": (NPU_U64, np.uint64)}


def root():
    if os.environ.get("NPU_EMU_AIESIM_ROOT"):
        return os.environ["NPU_EMU_AIESIM_ROOT"]
    builds = sorted((d for d in glob.glob(f"{NIGHTLY}/*") if os.path.basename(d).isdigit()),
                    key=lambda d: int(os.path.basename(d)), reverse=True)
    # The newest build may still be running; take the newest one with outputs.
    for b in builds:
        w = f"{b}/Vitis_Libraries/mllib/L2/regression/wrapper"
        if glob.glob(f"{w}/*_wrapper/*/XDNA_*/data/ofm32.txt"):
            return w
    return None


ROOT = root()


def stream(path, dtype=np.uint8):
    with open(path) as f:
        words = [int(line, 16) for line in f if line.strip()]
    return np.array(words, np.uint32).view(np.uint8).view(dtype)


def data(case, name, dtype=np.uint8):
    return stream(os.path.join(case, "data", name), dtype)


def words(case, name="coeff32.txt"):
    return data(case, name, np.uint32)


def fword(case, i, name="coeff32.txt"):
    return float(words(case, name)[i:i + 1].view(np.float32)[0])


def ty(t):
    return TYPES[t]


def tensor_descr(case):
    """(valid_shape, physical_shape, dtype) from a case's tensor_descr.h, or None.
    Wrappers graded by parse_hex_diff_wgarbage.py only compare the valid region."""
    p = os.path.join(case, "tensor_descr.h")
    if not os.path.exists(p):
        return None
    valid = phys = dt = None
    for line in open(p):
        line = line.strip()
        if line.startswith("#define OFM_VALID_SHAPE"):
            valid = tuple(int(x) for x in line.split("{", 1)[1].split("}", 1)[0].split(","))
        elif line.startswith("#define OFM_PHYSICAL_SHAPE"):
            phys = tuple(int(x) for x in line.split("{", 1)[1].split("}", 1)[0].split(","))
        elif line.startswith("#define DTYPE"):
            dt = line.split()[-1]
    return valid, phys, dt


def cases(dirs):
    """pytest params (dir, case) for every case of the given regression dirs
    that kept its aiesim output."""
    out = []
    if ROOT is None:
        return out
    for d in dirs:
        for c in sorted(glob.glob(f"{ROOT}/{d}/*/XDNA_*")):
            if os.path.exists(f"{c}/data/ofm32.txt") and os.path.exists(f"{c}/testcase.json"):
                out.append(pytest.param(d, c, id=re.sub(r"^XDNA_|_AIE_Test.*$", "",
                                                        os.path.basename(c))))
    return out


def assert_bytes(got, want):
    got = np.asarray(got).view(np.uint8).reshape(-1)
    want = np.asarray(want).view(np.uint8).reshape(-1)
    assert got.size == want.size, f"size {got.size} != {want.size}"
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} bytes differ from aiesim, first at "
                           f"{bad[:8].tolist()}: got {got[bad[:8]].tolist()} "
                           f"want {want[bad[:8]].tolist()}")
