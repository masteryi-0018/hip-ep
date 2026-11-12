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


class I64(int):
    """Marks an int64_t argument (counts); plain ints are passed as int."""


class U32(int):
    """Marks a uint32_t argument."""


class U64(int):
    """Marks a uint64_t argument (raw pad bits)."""


class Out:
    """The k-th output buffer."""

    def __init__(self, k=0):
        self.k = k


OUT, OUT2 = Out(0), Out(1)


class Host:
    """An argument passed by host pointer: a numpy array (shapes, perms, pads)
    or a ctypes object passed as is (a host array, byref(struct))."""

    def __init__(self, x):
        self.x = np.ascontiguousarray(x) if isinstance(x, (list, tuple, np.ndarray)) else x


class DevList:
    """A host array of device pointers, one per numpy array."""

    def __init__(self, arrays):
        self.arrays = arrays


FILL = 0xA5  # outputs are pre-filled with this so unwritten bytes show up


def _arg(a, outs, ptrs, keep):
    if isinstance(a, Out):
        return outs[a.k]
    if isinstance(a, np.ndarray):
        p = _to_device(a)
        ptrs.append(p)
        return p
    if isinstance(a, Host):
        if isinstance(a.x, np.ndarray):
            keep.append(a.x)
            return ctypes.c_void_p(a.x.ctypes.data)
        return a.x
    if isinstance(a, DevList):
        arr = (ctypes.c_void_p * max(len(a.arrays), 1))()
        for k, x in enumerate(a.arrays):
            p = _to_device(x)
            ptrs.append(p)
            arr[k] = p.value
        keep.append(arr)
        return arr
    if isinstance(a, A64Coeff):
        return ctypes.byref(a)
    if a is None:
        return ctypes.c_void_p(0)
    if isinstance(a, float):
        return ctypes.c_float(a)
    for kind, ct in ((I64, ctypes.c_int64), (U32, ctypes.c_uint32), (U64, ctypes.c_uint64)):
        if isinstance(a, kind):
            return ct(a)
    return ctypes.c_int(a)


def callx(name, *args, outs):
    """Call npu_emu `name` with a null stream prepended. numpy arrays in
    `args` are copied to the device; Out(k) marks where the k-th output device
    pointer goes. `outs` lists the byte size of each output; returns them as
    uint8 host arrays."""
    ptrs, keep, dev = [], [], []
    try:
        for nb in outs:
            p = ctypes.c_void_p()
            _check(_hip.hipMalloc(ctypes.byref(p), ctypes.c_size_t(max(nb, 1))), "hipMalloc")
            ptrs.append(p)
            _check(_hip.hipMemset(p, FILL, ctypes.c_size_t(nb)), "hipMemset")
            dev.append(p)
        cargs = [ctypes.c_void_p(0)] + [_arg(a, dev, ptrs, keep) for a in args]
        fn = getattr(_lib, name)
        fn.restype = ctypes.c_int
        _check(fn(*cargs), name)
        _check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        res = []
        for p, nb in zip(dev, outs):
            host = np.empty(nb, dtype=np.uint8)
            _check(_hip.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), p,
                                  ctypes.c_size_t(nb), 2), "hipMemcpy D2H")
            res.append(host)
        return res
    finally:
        for p in ptrs:
            _hip.hipFree(p)


def call(name, *args, out_nbytes, out_dtype):
    """callx with one output, returned as a host array of `out_dtype`."""
    return callx(name, *args, outs=[out_nbytes])[0].view(out_dtype)


def raw_rc(name, *args):
    """Return code of an entry point called with host-only or null arguments
    (argument validation tests)."""
    keep = []
    fn = getattr(_lib, name)
    fn.restype = ctypes.c_int
    return fn(ctypes.c_void_p(0), *[_arg(a, [], [], keep) for a in args])


def f32(x):
    return float(np.float32(x))
