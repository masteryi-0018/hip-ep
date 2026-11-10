# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""ctypes binding for libnpu_emu.so: host numpy arrays in, host arrays out."""

import ctypes
import os

import numpy as np

NPU_F32, NPU_F16, NPU_BF16, NPU_FP8 = 0, 1, 2, 3
NPU_I8, NPU_U8, NPU_I16, NPU_U16 = 4, 5, 6, 7
NPU_I32, NPU_U32, NPU_I64, NPU_U64 = 8, 9, 10, 11

# numpy storage dtype of each npu_dtype (bf16 / fp8 as raw bits)
STORAGE = {
    NPU_F32: np.float32, NPU_F16: np.uint16, NPU_BF16: np.uint16,
    NPU_FP8: np.uint8, NPU_I8: np.int8, NPU_U8: np.uint8, NPU_I16: np.int16,
    NPU_U16: np.uint16, NPU_I32: np.int32, NPU_U32: np.uint32,
    NPU_I64: np.int64, NPU_U64: np.uint64,
}


class A64Coeff(ctypes.Structure):
    _fields_ = [("mult_f", ctypes.c_float), ("bias_f", ctypes.c_float),
                ("mult_f_x2p32", ctypes.c_float), ("mult_f_x2p64", ctypes.c_float),
                ("M", ctypes.c_int32), ("s", ctypes.c_int32), ("B", ctypes.c_int64)]


_hip = ctypes.CDLL("libamdhip64.so")
_lib = ctypes.CDLL(os.environ["NPU_EMU_LIB"])


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: {rc}")


def _to_device(a):
    a = np.ascontiguousarray(a)
    p = ctypes.c_void_p()
    _check(_hip.hipMalloc(ctypes.byref(p), ctypes.c_size_t(max(a.nbytes, 1))), "hipMalloc")
    _check(_hip.hipMemcpy(p, a.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(a.nbytes), 1),
           "hipMemcpy H2D")
    return p


def call(name, *args, out_nbytes, out_dtype):
    """Call npu_emu `name`. numpy arrays in `args` are copied to the device;
    the token OUT marks where the output device pointer goes. A stream of 0 is
    prepended. Returns the output as a host array of `out_dtype`."""
    ptrs, cargs = [], [ctypes.c_void_p(0)]
    out = ctypes.c_void_p()
    _check(_hip.hipMalloc(ctypes.byref(out), ctypes.c_size_t(max(out_nbytes, 1))), "hipMalloc")
    # Fill with a pattern so unwritten bytes show up as mismatches.
    _check(_hip.hipMemset(out, 0xA5, ctypes.c_size_t(out_nbytes)), "hipMemset")
    ptrs.append(out)
    try:
        for a in args:
            if a is OUT:
                cargs.append(out)
            elif isinstance(a, np.ndarray):
                p = _to_device(a)
                ptrs.append(p)
                cargs.append(p)
            elif isinstance(a, float):
                cargs.append(ctypes.c_float(a))
            elif isinstance(a, A64Coeff):
                cargs.append(ctypes.byref(a))
            else:
                cargs.append(ctypes.c_int64(a) if isinstance(a, I64) else ctypes.c_int(a))
        fn = getattr(_lib, name)
        fn.restype = ctypes.c_int
        _check(fn(*cargs), name)
        _check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        host = np.empty(out_nbytes, dtype=np.uint8)
        _check(_hip.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), out,
                              ctypes.c_size_t(out_nbytes), 2), "hipMemcpy D2H")
        return host.view(out_dtype)
    finally:
        for p in ptrs:
            _hip.hipFree(p)


class I64(int):
    """Marks an int64_t argument (counts); plain ints are passed as int."""


OUT = object()


def f32(x):
    return float(np.float32(x))
