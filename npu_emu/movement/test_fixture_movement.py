# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""movement.h entry points against NPU reference outputs, over the valid region."""

import os

import numpy as np
import pytest

import golden_movement as gm
from fixture import cases, data, ofm, testcase as tc
from npu import DevList, Host, I64, Out, callx
from npu import (NPU_BF16, NPU_BFP16, NPU_F16, NPU_FP8, NPU_I8, NPU_I16, NPU_I32, NPU_U8,
                 NPU_U16, NPU_U32)
from testing import call_n

# dtype name -> (npu_dtype, unsigned storage of the same width)
DT = {"int8": (NPU_I8, np.uint8), "uint8": (NPU_U8, np.uint8), "float8": (NPU_FP8, np.uint8),
      "int16": (NPU_I16, np.uint16), "uint16": (NPU_U16, np.uint16),
      "bfloat16": (NPU_BF16, np.uint16), "float16": (NPU_F16, np.uint16),
      "int32": (NPU_I32, np.uint32), "uint32": (NPU_U32, np.uint32)}
IDX = {"int16": (NPU_I16, np.int16), "uint16": (NPU_U16, np.uint16),
       "int32": (NPU_I32, np.int32), "uint32": (NPU_U32, np.uint32)}


def one(name, *args, shape, dtype):
    return call_n(name, *args, n=tuple(shape), dt=dtype)


def compare(got, want_phys, region=None):
    """got: logical result; want_phys: sim output in its stored layout; region: slices
    of want_phys holding the valid result (default: leading corner of got's shape)."""
    if region is None:
        region = tuple(slice(0, s) for s in got.shape)
    want = want_phys[region]
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    g = np.ascontiguousarray(got).view(np.uint8).reshape(-1)
    w = np.ascontiguousarray(want).view(np.uint8).reshape(-1)
    bad = np.nonzero(g != w)[0]
    assert bad.size == 0, (f"{bad.size}/{w.size} bytes differ from sim, first at "
                           f"{bad[:8].tolist()}: got {g[bad[:8]].tolist()} want {w[bad[:8]].tolist()}")


def stream_n(case, name, dtype, n):
    """First n elements of a data file (files are padded to whole words)."""
    s = data(case, name, dtype)
    assert s.size >= n, f"{name}: {s.size} < {n} elements"
    return s[:n]


def skip_failed_run(case):
    if not any(f.startswith("PASS") for f in os.listdir(case)):
        pytest.skip("the sim run failed (no PASS marker)")


def run_elementwise_not(case, t):
    n = t["ifm_sv_size"] * t["ifm_sv_iteration"]
    x = stream_n(case, "ifm32.txt", np.uint8, n)
    got = one("npu_elementwise_not", x, DT[t["ifm_dtype"]][0], Out(), I64(n), shape=(n,),
              dtype=np.uint8)
    compare(got, stream_n(case, ofm(case), np.uint8, n))


def xyb(t, pre="ifm1"):
    return (t[f"{pre}_physical_sv_X"] * t[f"{pre}_physical_sv_Y"] * t[f"{pre}_physical_sv_B"]
            * t["ifm_X_iteration"] * t["ifm_Y_iteration"] * t["ifm_B_iteration"])


def run_generic_concat(case, t):
    rows = xyb(t)
    c1, c2, co = (t[f"{p}_physical_sv_C"] for p in ("ifm1", "ifm2", "ofm"))
    n1, n2 = t["ifm1_valid_sv_C"], t["ifm2_valid_sv_C"]
    s1, s2 = t["in1_start_idx"], t["in2_start_idx"]
    if t["ifm_dtype"] == "bfp16":  # sizes count values; rows hold 9-byte blocks of 8
        tt, dt, k = NPU_BFP16, np.uint8, 9 / 8
    else:
        (tt, dt), k = DT[t["ifm_dtype"]], 1
    b1, b2, bo, w = (int(v * k) for v in (c1, c2, co, n1 + n2))
    a = stream_n(case, "ifm32.txt", dt, rows * b1).reshape(rows, b1)
    b = stream_n(case, "wts32.txt", dt, rows * b2).reshape(rows, b2)
    got = one("npu_generic_concat", a, tt, b, Out(), I64(rows), I64(c1), I64(s1), I64(n1),
              I64(c2), I64(s2), I64(n2), shape=(rows, w), dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, rows * bo).reshape(rows, bo))


def run_concat_16b_c6463(case, t):
    tt, dt = DT[t["ifm_dtype"]]
    rows = xyb(t)
    c1, c2, co = t["ifm1_physical_sv_C"], t["ifm2_physical_sv_C"], t["ofm_physical_sv_C"]
    a = stream_n(case, "ifm32.txt", dt, rows * c1).reshape(rows, c1)[:, :63]
    b = stream_n(case, "wts32.txt", dt, rows * c2).reshape(rows, c2)[:, :1]
    got = one("npu_generic_concat_16b_c6463", np.ascontiguousarray(a), tt,
              np.ascontiguousarray(b), Out(), I64(rows), shape=(rows, 64), dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, rows * co).reshape(rows, co))


def run_concat_16b_inner(case, t):
    tt, dt = DT[t["ifm_dtype"]]
    rows = 1
    for d in "ADBYX":
        rows *= t[f"ifm_{d}_iteration"] * t[f"ifm0_physical_sv_{d}"]
    c0, c1, co = t["ifm0_physical_sv_C"], t["ifm1_physical_sv_C"], t["ofm_physical_sv_C"]
    n0, n1 = t["ifm0_valid_sv_C"], t["ifm1_valid_sv_C"]
    a = stream_n(case, "ifm32.txt", dt, rows * c0).reshape(rows, c0)[:, :n0]
    b = stream_n(case, "wts32.txt", dt, rows * c1).reshape(rows, c1)[:, :n1]
    got = one("npu_generic_concat_16b_inner", np.ascontiguousarray(a), tt,
              np.ascontiguousarray(b), Out(), I64(rows), I64(n0), I64(n1), shape=(rows, n0 + n1),
              dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, rows * co).reshape(rows, co))


def run_concat_multi(name):
    def run(case, t):
        tt, dt = DT[t["ifm_dtype"]]
        rows = xyb(t, "ifm")
        k = t["concat_volumes"]
        x = data(case, "ifm32.txt", dt)
        parts, off = [], 0
        for j in range(6):
            c = t[f"ifm{j}_physical_sv_C"]
            if c > 0:
                p = x[off:off + rows * c].reshape(rows, c)
                off += rows * c
                if j < k:
                    parts.append(np.ascontiguousarray(p[:, :t[f"ifm{j}_valid_sv_C"]]))
        widths = [p.shape[1] for p in parts]
        co = t["ofm_physical_sv_C"]
        got = one(name, DevList(parts), tt, Out(), I64(rows), k, Host(np.array(widths, np.int64)),
                  shape=(rows, sum(widths)), dtype=dt)
        compare(got, stream_n(case, ofm(case), dt, rows * co).reshape(rows, co))
    return run


def run_concat_unaligned_bfp16(case, t):
    rows = xyb(t)
    n1, n2 = t["ifm1_valid_sv_C"], t["ifm2_valid_sv_C"]
    c1, c2, co = (t[f"{p}_physical_sv_C"] // 8 * 9 for p in ("ifm1", "ifm2", "ofm"))
    b1, b2, bo = -(-n1 // 8) * 9, -(-n2 // 8) * 9, -(-(n1 + n2) // 8) * 9
    a = stream_n(case, "ifm32.txt", np.uint8, rows * c1).reshape(rows, c1)[:, :b1]
    b = stream_n(case, "wts32.txt", np.uint8, rows * c2).reshape(rows, c2)[:, :b2]
    got = one("npu_concat_unaligned_bfp16", np.ascontiguousarray(a), np.ascontiguousarray(b),
              Out(), I64(rows), I64(n1), I64(n2), shape=(rows, bo), dtype=np.uint8)
    compare(got, stream_n(case, ofm(case), np.uint8, rows * co).reshape(rows, co))


def full_dims(t, names, it_fmt, sv_fmt):
    return tuple(t[sv_fmt.format(n)] * t[it_fmt.format(n)] for n in names)


def run_depth_to_space(nhwc):
    def run(case, t):
        tt, dt = DT[t["ifm_dtype"]]
        b, mode = t["blocksize"], t["mode"]
        names = ("batch", "height", "width", "channel") if nhwc else \
                ("batch", "channel", "height", "width")
        phys = full_dims(t, names, "ifm_{}_iteration", "ifm_physical_sv_{}")
        valid = tuple(t[f"ifm_valid_full_{n}"] for n in names)
        x = stream_n(case, "ifm32.txt", dt, int(np.prod(phys))).reshape(phys)
        x = np.ascontiguousarray(x[tuple(slice(0, v) for v in valid)])
        if nhwc:
            N, H, W, C = valid
            oshape, ophys = (N, H * b, W * b, C // b // b), \
                (phys[0], phys[1] * b, phys[2] * b, phys[3] // b // b)
            name = "npu_generic_depth_to_space_wc64"
        else:
            N, C, H, W = valid
            oshape, ophys = (N, C // b // b, H * b, W * b), \
                (phys[0], phys[1] // b // b, phys[2] * b, phys[3] * b)
            name = "npu_generic_depth_to_space"
        got = one(name, x, tt, Out(), *[I64(v) for v in valid], I64(b), mode, shape=oshape,
                  dtype=dt)
        compare(got, stream_n(case, ofm(case), dt, int(np.prod(ophys))).reshape(ophys))
    return run


def run_space_to_depth(name):
    def run(case, t):
        tt, dt = DT[t["ifm_dtype"]]
        b = t["block_size"]
        names = ("height", "width", "channel")
        phys = full_dims(t, names, "ifm_{}_iteration", "ifm_physical_sv_{}")
        H, W, C = (t[f"ifm_valid_full_{n}"] for n in names)
        x = stream_n(case, "ifm32.txt", dt, int(np.prod(phys))).reshape(phys)
        x = np.ascontiguousarray(x[:H, :W, :C])
        ophys = (phys[0] // b, phys[1] // b, phys[2] * b * b)
        got = one(name, x, tt, Out(), I64(H), I64(W), I64(C), I64(b),
                  shape=(H // b, W // b, C * b * b), dtype=dt)
        compare(got, stream_n(case, ofm(case), dt, int(np.prod(ophys))).reshape(ophys))
    return run


def run_expand_innermost(case, t):
    tt, dt = DT[t["dtype"]]
    phys = [t[f"ifm_physical_sv_dim{i}"] * t[f"dim{i}_iter"] for i in range(5)]
    for i in range(4):
        assert t[f"ifm_valid_sv_dim{i}"] == t[f"ofm_valid_sv_dim{i}"], "expand only along dim4"
    rows = int(np.prod(phys[:4]))
    cin, cout = t["ifm_valid_sv_dim4"], t["ofm_valid_sv_dim4"]
    co = t["ofm_physical_sv_dim4"] * t["dim4_iter"]
    x = stream_n(case, "ifm32.txt", dt, rows * phys[4]).reshape(rows, phys[4])[:, :cin]
    got = one("npu_generic_expand_innermost", np.ascontiguousarray(x), tt, Out(), I64(rows),
              I64(cin), I64(cout), shape=(rows, cout), dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, rows * co).reshape(rows, co))


def run_gather_3din(case, t):
    tt, dt = DT[t["ifm_dtype"]]
    it, idt = IDX[t["wts_dtype"]]
    axis = t["axis"]
    phys = tuple(t[f"ifm_physical_sv_dim{i}"] * t[f"ifm_dim{i}_iteration"] for i in range(3))
    valid = tuple(t[f"ifm_valid_full_dim{i}"] for i in range(3))
    nidx = t["wts_valid_full_indices"]
    x = stream_n(case, "ifm32.txt", dt, int(np.prod(phys))).reshape(phys)
    x = np.ascontiguousarray(x[:valid[0], :valid[1], :valid[2]])
    idx = stream_n(case, "wts32.txt", idt, nidx)
    oshape, ophys = list(valid), list(phys)
    oshape[axis] = nidx
    ophys[axis] = t["wts_physical_sv_indices"] * t["wts_indices_iteration"]
    assert it == NPU_U32, "uint32 indices only"
    got = one("npu_generic_gather_3din_1dindices", x, tt, idx, Out(), *[I64(v) for v in valid],
              I64(nidx), axis, shape=tuple(oshape), dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, int(np.prod(ophys))).reshape(ophys))


def run_gather_element_axis0(case, t):
    tt, dt = DT[t["ifm_dtype"]]
    it, idt = IDX[t["wts_dtype"]]
    M, N, Mi = t["ifm_valid_full_M"], t["ifm_valid_full_N"], t["wts_valid_full_M"]
    pM = t["ifm_physical_sv_M"] * t["ifm_M_iteration"]
    pN = t["ifm_physical_sv_N"] * t["ifm_N_iteration"]
    pMi = t["wts_physical_sv_M"] * t["wts_M_iteration"]
    x = stream_n(case, "ifm32.txt", dt, pM * pN).reshape(pM, pN)[:M, :N]
    idx = stream_n(case, "wts32.txt", idt, pMi * pN).reshape(pMi, pN)[:Mi, :N]
    got = one("npu_generic_gather_element_2din_2dindices_axis0", np.ascontiguousarray(x), tt,
              np.ascontiguousarray(idx), it, Out(), I64(M), I64(N), I64(Mi), shape=(Mi, N),
              dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, pMi * pN).reshape(pMi, pN))


def run_gather_element_innermost(case, t):
    tt, dt = DT[t["ifm_dtype"]]
    it, idt = IDX[t["wts_dtype"]]
    M, N, Ni = t["ifm_valid_full_M"], t["ifm_valid_full_N"], t["wts_valid_full_N"]
    pM = t["ifm_physical_sv_M"] * t["ifm_M_iteration"]
    pN = t["ifm_physical_sv_N"] * t["ifm_N_iteration"]
    pNi = t["wts_physical_sv_N"] * t["ifm_N_iteration"]
    x = stream_n(case, "ifm32.txt", dt, pM * pN).reshape(pM, pN)[:M, :N]
    idx = stream_n(case, "wts32.txt", idt, pM * pNi).reshape(pM, pNi)[:M, :Ni]
    got = one("npu_generic_gather_element_2din_2dindices_innermost", np.ascontiguousarray(x), tt,
              np.ascontiguousarray(idx), it, Out(), I64(M), I64(N), I64(Ni), shape=(M, Ni),
              dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, pM * pNi).reshape(pM, pNi))


def run_pad2d_c64(case, t):
    tt, dt = DT[t["ifm_dtype"]]
    phys = full_dims(t, ("channel", "height", "width"), "ifm_{}_iteration", "ifm_physical_sv_{}")
    Cv, Hv, Wv = (t[f"ifm_valid_full_{n}"] for n in ("channel", "height", "width"))
    hs, he, ws, we = (t[k] for k in ("pad_height_start", "pad_height_end", "pad_width_start",
                                     "pad_width_end"))
    H, W = Hv - hs - he, Wv - ws - we
    C, Hp, Wp = phys

    def unformat(s):  # (H, C/64, W, 64) -> HWC
        return s.reshape(Hp, C // 64, Wp, 64).transpose(0, 2, 1, 3).reshape(Hp, Wp, C)

    x = unformat(stream_n(case, "ifm32.txt", dt, C * Hp * Wp))
    x = np.ascontiguousarray(x[hs:hs + H, ws:ws + W, :Cv])
    got = one("npu_generic_inplace_pad2d_c64", x, tt, Out(), I64(H), I64(W), I64(Cv), I64(hs),
              I64(he), I64(ws), I64(we), int(t["pad_value_rtp"]), shape=(Hv, Wv, Cv), dtype=dt)
    compare(got, unformat(stream_n(case, ofm(case), dt, C * Hp * Wp)))


def run_pad3d(case, t):
    tt, dt = DT[t["ifm_dtype"]]
    names = ("height", "width", "channel")
    phys = full_dims(t, names, "ifm_{}_iteration", "ifm_physical_sv_{}")
    valid = [t[f"ifm_valid_full_{n}"] for n in names]
    pads = [t[f"pad_{n}_{e}"] for n in names for e in ("start", "end")]
    H, W, C = (valid[i] - pads[2 * i] - pads[2 * i + 1] for i in range(3))
    x = stream_n(case, "ifm32.txt", dt, int(np.prod(phys))).reshape(phys)
    x = np.ascontiguousarray(x[pads[0]:pads[0] + H, pads[2]:pads[2] + W, pads[4]:pads[4] + C])
    got = one("npu_generic_inplace_pad3d", x, tt, Out(), I64(H), I64(W), I64(C),
              Host(np.array(pads, np.int64)), t["mode"], int(t["pad_value_rtp"]),
              shape=tuple(valid), dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, int(np.prod(phys))).reshape(phys))


def run_mem_copy(case, t):
    tt, dt = DT[t["ifm_dtype"]]
    n = t["ifm_valid_full_size"]
    x = stream_n(case, "ifm32.txt", dt, n)
    got = one("npu_generic_mem_copy", x, tt, Out(), I64(n), shape=(n,), dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, n))


def run_innermost_reshape(name):
    def run(case, t):
        tt, dt = DT[t["ifm_dtype"]]
        phys = [t[f"ifm_physical_dim_{i}"] * t[f"ifm_dim_{i}_iter"] for i in range(4)]
        vin, vout, pout = t["ifm_valid_dim_3"], t["ofm_valid_dim_3"], t["ofm_physical_dim_3"]
        x = stream_n(case, "ifm32.txt", dt, int(np.prod(phys))).reshape(-1, phys[3])[:, :vin]
        x = np.ascontiguousarray(x).reshape(-1)
        assert x.size % vout == 0
        got = one(name, x, tt, Out(), I64(x.size), shape=(x.size,), dtype=dt)
        rows = x.size // vout
        compare(got.reshape(rows, vout),
                stream_n(case, ofm(case), dt, rows * pout).reshape(rows, pout))
    return run


def sv_shape(t, pre, dims=("batch", "height", "width", "channel"), it=True):
    return tuple(t[f"{pre}_physical_sv_{d}"] * (t[f"ifm_{d}_iteration"] if it else 1)
                 for d in dims)


def run_slice_16b_c6463(case, t):
    tt, dt = DT[t["dtype"]]
    n, h, w, _ = sv_shape(t, "ifm")
    rows = n * h * w
    x = stream_n(case, "ifm32.txt", dt, rows * 64).reshape(rows, 64)
    o0, o1 = callx("npu_generic_slice_16b_c6463", x, tt, Out(0), Out(1), I64(rows),
                   outs=[rows * 63 * x.itemsize, rows * x.itemsize])
    got = np.concatenate([o0.view(dt), o1.view(dt)])
    compare(got, stream_n(case, ofm(case), dt, rows * 64))


def run_slice_16b_inner_range(case, t):
    tt, dt = DT[t["dtype"]]
    n, h, w, c = sv_shape(t, "ifm")
    rows, cv = n * h * w, t["ifm_valid_sv_channel"]
    s, e = t["slice_inner_start"], t["slice_inner_end"]
    co = t["ofm_physical_sv_channel"]
    x = stream_n(case, "ifm32.txt", dt, rows * c).reshape(rows, c)[:, :cv]
    got = one("npu_generic_slice_16b_inner_range", np.ascontiguousarray(x), tt, Out(), I64(rows),
              I64(cv), I64(s), I64(e), shape=(rows, e - s), dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, rows * co).reshape(rows, co))


def run_slice_16b_inner(case, t):
    tt, dt = DT[t["dtype"]]
    n, h, w, c = sv_shape(t, "ifm")
    rows, cv, sp = n * h * w, t["ifm_valid_sv_channel"], t["slice_inner"]
    p0, p1 = t["ofm0_physical_sv_channel"], t["ofm1_physical_sv_channel"]
    x = stream_n(case, "ifm32.txt", dt, rows * c).reshape(rows, c)[:, :cv]
    es = np.dtype(dt).itemsize
    o0, o1 = callx("npu_generic_slice_16b_inner", np.ascontiguousarray(x), tt, Out(0), Out(1),
                   I64(rows), I64(cv), I64(sp), outs=[rows * sp * es, rows * (cv - sp) * es])
    want = stream_n(case, ofm(case), dt, rows * (p0 + p1))
    compare(o0.view(dt).reshape(rows, sp), want[:rows * p0].reshape(rows, p0))
    compare(o1.view(dt).reshape(rows, cv - sp), want[rows * p0:].reshape(rows, p1))


def run_slice_innermost(case, t):
    tt, dt = DT[t["ifm_dtype"]]
    dims = ("batch", "height", "width", "channel")
    phys = sv_shape(t, "ifm", dims)
    valid = [t[f"ifm_valid_sv_{d}"] if t[f"ifm_{d}_iteration"] == 1 else phys[i]
             for i, d in enumerate(dims)]
    valid[3] = t["ifm_valid_sv_channel"]
    x = stream_n(case, "ifm32.txt", dt, int(np.prod(phys))).reshape(phys)
    x = x[:valid[0], :valid[1], :valid[2], :valid[3]]
    rows = valid[0] * valid[1] * valid[2]
    s, e, st = t["slice_inner_start"], t["slice_inner_end"], t.get("slice_inner_stride", 1)
    wo = len(range(s, e, st))
    got = one("npu_generic_slice_innermost", np.ascontiguousarray(x).reshape(rows, valid[3]),
              tt, Out(), I64(rows), I64(valid[3]), I64(s), I64(e), I64(st), shape=(rows, wo),
              dtype=dt)
    oshape = tuple(valid[:3]) + (wo,)
    ophys = tuple(oshape[i] + t[f"ofm_physical_sv_{d}"] - t[f"ofm_valid_sv_{d}"]
                  for i, d in enumerate(dims))
    compare(got.reshape(oshape), stream_n(case, ofm(case), dt, int(np.prod(ophys))).reshape(ophys))


def run_transpose4d_0132(case, t):
    tt, dt = DT[t["dtype"]]
    d = [t[f"sv_d{i}"] * t[f"d{i}_iterations"] for i in range(3)]
    x = stream_n(case, "ifm32.txt", dt, int(np.prod(d)))
    got = one("npu_generic_transpose4d_0132_RCR32", x, tt, Out(), *[I64(v) for v in d],
              shape=(d[0], d[2], d[1]), dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, got.size).reshape(got.shape))


def run_transpose4d_aligned(case, t):
    tt, dt = DT[t["dtype"]]
    sv = [t[f"d{i}_sv_size"] for i in range(4)]
    S = int(np.prod([t[f"d{i}_iters"] for i in range(4)]))
    perm = [int(c) for c in str(t["perm"]).zfill(4)]
    n = int(np.prod(sv))
    x = stream_n(case, "ifm32.txt", dt, S * n).reshape(S, n)
    want = stream_n(case, ofm(case), dt, S * n).reshape(S, n)
    oshape = tuple(sv[p] for p in perm)
    for s in range(S):
        got = one("npu_generic_transpose4d_aligned", np.ascontiguousarray(x[s]), tt, Out(),
                  Host(np.array(sv, np.int64)), Host(np.array(perm, np.int32)), shape=oshape,
                  dtype=dt)
        compare(got, want[s].reshape(oshape))


def run_transpose_k8n8(case, t):
    tt, dt = DT[t["ifm_dtype"]]
    K = t["ifm_sv_K_physical"] * t["ifm_K_iteration"]
    N = t["ifm_sv_N_physical"] * t["ifm_N_iteration"]
    Kv, Nv = t["ifm_K_full_valid"], t["ifm_N_full_valid"]
    # stored as (K/8, N/8, 8, 8) blocks; only the valid blocks are compared
    tiles = stream_n(case, "ifm32.txt", dt, K * N).reshape(K // 8, N // 8, 8, 8)
    x = np.ascontiguousarray(tiles[:Kv // 8, :Nv // 8].transpose(0, 2, 1, 3)).reshape(Kv, Nv)
    got = one("npu_generic_transpose_k8n8", x, tt, Out(), I64(Kv), I64(Nv), shape=(Nv, Kv),
              dtype=dt)
    # out block (kb, nb) holds out[nb*8 + j][kb*8 + i] at [j][i]
    phys = got.reshape(Nv // 8, 8, Kv // 8, 8).transpose(2, 0, 1, 3)
    compare(np.ascontiguousarray(phys),
            stream_n(case, ofm(case), dt, K * N).reshape(K // 8, N // 8, 8, 8))


def run_transpose_mmac(a16):
    def run(case, t):
        K = t["ifm_sv_k_physical"] * t["time_split_k"]
        N = t["ifm_sv_n_physical"] * t["time_split_n"]
        if a16:
            dt, blk, name = np.uint16, 32, "npu_transpose_mmac_a16_n32k64"
        else:
            dt, blk, name = np.uint8, 64, "npu_transpose_mmac_a8_n64k64"
        x = stream_n(case, "ifm32.txt", dt, K * N).reshape(N // blk, K, blk)
        x = np.ascontiguousarray(x.transpose(1, 0, 2)).reshape(K, N)
        got = one(name, x, Out(), I64(K), I64(N), shape=(N, K), dtype=dt)
        want = stream_n(case, ofm(case), np.uint8, K * N * (2 if a16 else 1))
        if a16:  # (k/64, n, byte, k64)
            b = got.view(np.uint8).reshape(N, K // 64, 64, 2).transpose(1, 0, 3, 2)
        else:    # (k/64, n/64, n64, k64)
            b = got.reshape(N // 64, 64, K // 64, 64).transpose(2, 0, 1, 3)
        compare(np.ascontiguousarray(b).reshape(-1), want)
    return run


def run_transpose_unaligned(case, t):
    tt, dt = DT[t["dtype"]]
    d = [t[f"d{i}_iters"] * t[f"d{i}_sv_size"] for i in range(4)]
    perm = [int(c) for c in str(t["perm"]).zfill(4)]
    n = int(np.prod(d))
    x = stream_n(case, "ifm32.txt", dt, n)
    got = one("npu_generic_transpose_unaligned", x, tt, Out(), Host(np.array(d, np.int64)),
              Host(np.array(perm, np.int32)), shape=(n,), dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, n))


PERM3 = {2: (0, 2, 1), 3: (2, 0, 1), 4: (1, 2, 0), 5: (2, 1, 0)}


def run_transpose3d_mmac(case, t):
    tt, dt = DT[t["dtype"]]
    d = (t["d0"], t["d1"], t["d2"])
    n = int(np.prod(d))
    x = stream_n(case, "ifm32.txt", dt, n)
    got = one("npu_transpose3d_mmac", x, tt, Out(), Host(np.array(d, np.int64)),
              Host(np.array(PERM3[t["perm"]], np.int32)), shape=(n,), dtype=dt)
    compare(got, stream_n(case, ofm(case), dt, n))


def run_planar3_c8(case, t):
    S, P = t["kernel_iter_nums"], t["plane_size"]
    x = stream_n(case, "ifm32.txt", np.uint8, S * 4 * P).reshape(S, 4, P)[:, :3]
    got = one("npu_transpose_planar3_c8", np.ascontiguousarray(x), Out(), I64(S), I64(P),
              t["pad_value"], shape=(S * P * 8,), dtype=np.uint8)
    compare(got, stream_n(case, ofm(case), np.uint8, S * P * 8))


def resize_dims(t):
    Ho, Wo = t["ofm_sv_height"] * t["height_iteration"], t["ofm_sv_width"] * t["width_iteration"]
    Hi, Wi = t["ifm_sv_height"] * t["height_iteration"], t["ifm_sv_width"] * t["width_iteration"]
    return Hi, Wi, t["sv_depth"] * t["depth_iteration"], Ho, Wo


def resize_tables(t, Hi, Wi, Ho, Wo):
    coord = t.get("coordinate_transformation_mode", "half_pixel")
    near = t.get("nearest_mode", "round_prefer_floor")
    return (gm.onnx_nearest_index(Ho, Hi, coord, near), gm.onnx_nearest_index(Wo, Wi, coord, near))


def run_resize_nearest_bf16(case, t):
    tt, dt = DT[t["dtype"]]
    Hi, Wi, C, Ho, Wo = resize_dims(t)
    ih, iw = resize_tables(t, Hi, Wi, Ho, Wo)
    # stored as (H, C/32, W, 32)
    x = stream_n(case, "ifm32.txt", dt, Hi * Wi * C).reshape(Hi, C // 32, Wi, 32)
    x = np.ascontiguousarray(x.transpose(0, 2, 1, 3)).reshape(Hi, Wi, C)
    got = one("npu_resize_nearest_bf16", x, tt, Out(), I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo),
              ih, iw, shape=(Ho, Wo, C), dtype=dt)
    want = stream_n(case, ofm(case), dt, Ho * Wo * C).reshape(Ho, C // 32, Wo, 32)
    compare(got, np.ascontiguousarray(want.transpose(0, 2, 1, 3)).reshape(Ho, Wo, C))


def run_resize_nearest_bfp16(case, t):
    Hi, Wi, C, Ho, Wo = resize_dims(t)
    ih, iw = resize_tables(t, Hi, Wi, Ho, Wo)
    # stored as (H, C/128, W, 128) values in bfp16 blocks
    def to_hwc(s, H, W):
        return np.ascontiguousarray(s.reshape(H, C // 128, W, 16 * 9).transpose(0, 2, 1, 3)
                                    ).reshape(H, W, C // 8 * 9)
    x = to_hwc(stream_n(case, "ifm32.txt", np.uint8, Hi * Wi * C // 8 * 9), Hi, Wi)
    got = one("npu_resize_nearest_bfp16", x, Out(), I64(Hi), I64(Wi), I64(C), I64(Ho), I64(Wo),
              ih, iw, shape=(Ho, Wo, C // 8 * 9), dtype=np.uint8)
    compare(got, to_hwc(stream_n(case, ofm(case), np.uint8, Ho * Wo * C // 8 * 9), Ho, Wo))


RUNNERS = {
    "elementwise_not_wrapper": run_elementwise_not,
    "generic_concat_wrapper": run_generic_concat,
    "generic_concat_16b_c6463_wrapper": run_concat_16b_c6463,
    "generic_concat_16b_inner_wrapper": run_concat_16b_inner,
    "generic_concat_16b_multi_wrapper": run_concat_multi("npu_generic_concat_16b_multi"),
    "generic_concat_multi_c_fusion_wrapper": run_concat_multi("npu_generic_concat_multi_c_fusion"),
    "concat_unaligned_bfp16_wrapper": run_concat_unaligned_bfp16,
    "generic_depth_to_space_wrapper": run_depth_to_space(False),
    "generic_depth_to_space_wc64_wrapper": run_depth_to_space(True),
    "generic_space_to_depth_wrapper": run_space_to_depth("npu_generic_space_to_depth"),
    "generic_space_to_depth_c4bs4_wrapper": run_space_to_depth("npu_generic_space_to_depth_c4bs4"),
    "generic_expand_innermost_wrapper": run_expand_innermost,
    "generic_gather_3din_1dindices_wrapper": run_gather_3din,
    "generic_gather_element_2din_2dindices_axis0_wrapper": run_gather_element_axis0,
    "generic_gather_element_2din_2dindices_innermost_wrapper": run_gather_element_innermost,
    "generic_inplace_pad2d_c64_wrapper": run_pad2d_c64,
    "generic_inplace_pad3d_wrapper": run_pad3d,
    "generic_mem_copy_wrapper": run_mem_copy,
    "generic_pad_innermost_wrapper": run_innermost_reshape("npu_generic_pad_innermost"),
    "generic_reshape_innermost_wrapper": run_innermost_reshape("npu_generic_reshape_innermost"),
    "generic_unpad_innermost_wrapper": run_innermost_reshape("npu_generic_unpad_innermost"),
    "generic_slice_16b_c6463_wrapper": run_slice_16b_c6463,
    "generic_slice_16b_inner_range_wrapper": run_slice_16b_inner_range,
    "generic_slice_16b_inner_wrapper": run_slice_16b_inner,
    "generic_slice_innermost_wrapper": run_slice_innermost,
    "generic_transpose4d_0132_RCR32_wrapper": run_transpose4d_0132,
    "generic_transpose4d_aligned_wrapper": run_transpose4d_aligned,
    "generic_transpose_k8n8_wrapper": run_transpose_k8n8,
    "generic_transpose_mmac_a16_n32k64_wrapper": run_transpose_mmac(True),
    "generic_transpose_mmac_a8_n64k64_wrapper": run_transpose_mmac(False),
    "generic_transpose_unaligned_wrapper": run_transpose_unaligned,
    "transpose3d_mmac_wrapper": run_transpose3d_mmac,
    "transpose_planar3_c8_wrapper": run_planar3_c8,
    "resize_nearest_bf16_wrapper": run_resize_nearest_bf16,
    "resize_nearest_bfp16_wrapper": run_resize_nearest_bfp16,
}

CASES = cases(list(RUNNERS))


@pytest.mark.skipif(not CASES, reason="no sim dumps reachable")
@pytest.mark.parametrize("wrapper,case", CASES)
def test_fixture(wrapper, case):
    skip_failed_run(case)
    RUNNERS[wrapper](case, tc(case))
