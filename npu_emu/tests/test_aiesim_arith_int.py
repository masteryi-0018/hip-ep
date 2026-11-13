# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The arith_int entry points (include/npu_emu_arith_int.h) against mllib
nightly aiesim outputs, byte for byte. Flat kernels read the first n elements
of each stream; the Broadcasting3d / MAC inputs are cropped from their
physical HWC volumes (a broadcast dimension keeps its one sub-volume, the
others span every iteration) and the output is compared on its valid region.
Coefficients are the words the kernel reads: kernel_rtp.h for the plain flat
QDQ kernels, the coeff buffer (coeff32.txt) for the rest, PReLU slopes from
the per-channel-tile blocks of that buffer."""

import json
import os
import re

import numpy as np
import pytest

from aiesim_common import assert_bytes, cases, data, tensor_descr, ty, words
from npu import I64, OUT, Host, call

FLAT = ["qdq_elementwise_add_a8_wrapper", "qdq_elementwise_add_a16_wrapper",
        "qdq_elementwise_sub_a8_wrapper", "qdq_elementwise_sub_a16_wrapper",
        "qdq_elementwise_mul_a8_wrapper", "qdq_elementwise_mul_a16_wrapper"]
FLAT_FUSED = ["qdq_elementwise_add_fused_a8_wrapper", "qdq_elementwise_add_fused_a16_wrapper",
              "qdq_elementwise_sub_fused_a8_wrapper", "qdq_elementwise_sub_fused_a16_wrapper",
              "qdq_elementwise_mul_fused_a8_wrapper", "qdq_elementwise_mul_fused_a16_wrapper"]
XINT = ["xint8_add_wrapper", "elementwise_add_fused_a8_xint_wrapper", "xint8_mac_wrapper"]
BC3D = [f"qdq_elementwise_broadcasting3d_{op}_a{b}_wrapper" for op in ("add", "sub", "mul")
        for b in (8, 16)]
BC3D_FUSED = [f"qdq_elementwise_broadcasting3d_{op}_fused_a{b}_wrapper"
              for op in ("add", "sub", "mul") for b in (8, 16)]
MAC_FUSED = [f"qdq_elementwise_broadcasting3d_mac_fused_{v}_wrapper"
             for v in ("a8w8", "a16w8", "a16w16")]

ACT_NONE, ACT_RELU, ACT_CLIP, ACT_LRELU, ACT_PRELU = 0, 1, 2, 3, 4
# activation_type_templates of the fused wrappers -> npu_arith_activation
TEMPLATE_ACT = {0: ACT_NONE, 1: ACT_RELU, 100: ACT_CLIP, 3: ACT_LRELU, 9: ACT_PRELU}
# activation_type of the MAC fused wrappers (relu6 = clip template)
MAC_ACT = {0: ACT_NONE, 1: ACT_RELU, 2: ACT_CLIP, 3: ACT_LRELU}


def load(case):
    return json.load(open(os.path.join(case, "testcase.json")))


def run(name, *args, n, t):
    dt = ty(t)[1]
    return call(name, *args, out_nbytes=n * np.dtype(dt).itemsize, out_dtype=dt)


def rtp_words(case):
    """generated_rtps of kernel_rtp.h as uint32."""
    txt = open(os.path.join(case, "kernel_rtp.h")).read()
    body = txt.split("generated_rtps", 1)[1].split("{", 1)[1].split("}", 1)[0]
    return np.array([int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]+", body)], np.uint32)


def fwords(w):
    return [float(x) for x in np.asarray(w, np.uint32).view(np.float32)]


def int_type(bits, signed):
    return f"{'' if signed else 'u'}int{bits}"


def bits_of(t):
    return 8 if t.endswith("int8") else 16


def shp(s):
    return Host(np.array(s, np.int64))


def valid(t, k):
    return tuple(t[f"{k}_valid_full_{d}"] for d in ("height", "width", "channel"))


def volume(case, t, k, fname, dtype):
    """A logical HWC input cropped from its physical full volume."""
    it = [t[f"ifm1_{d}_iteration"] for d in ("height", "width", "channel")]
    ps = [t[f"{k}_physical_sv_{d}"] for d in ("height", "width", "channel")]
    vs = valid(t, k)
    if k == "ifm1":
        phys = [p * i for p, i in zip(ps, it)]
    else:
        phys = [p * i if v > 1 else p for p, i, v in zip(ps, it, vs)]
    flat = data(case, fname, ty(dtype)[1])
    return np.ascontiguousarray(flat[:np.prod(phys)].reshape(phys)[:vs[0], :vs[1], :vs[2]])


def check_hwc(case, got, dtype):
    vshape, pshape, _ = tensor_descr(case)
    want = data(case, "ofm32.txt", ty(dtype)[1])[:np.prod(pshape)].reshape(pshape)
    assert_bytes(got.reshape(vshape), want[:vshape[0], :vshape[1], :vshape[2]])


def check_flat(case, got, dtype):
    want = data(case, "ofm32.txt", ty(dtype)[1])[:got.size]
    assert_bytes(got, want)


# ---- flat ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,case", cases(FLAT))
def test_flat(d, case):
    t = load(case)
    n = t["ifm_sv_size"] * t["ifm_sv_iteration"]
    it, ot = t["ifm_dtype"], t["ofm_dtype"]
    a = data(case, "ifm32.txt", ty(it)[1])[:n]
    b = data(case, "wts32.txt", ty(it)[1])[:n]
    mul = "_mul_" in d
    c = fwords(rtp_words(case)[2:6 if mul else 5])  # after inner_g, ifm_sv_iterations
    name = f"npu_elementwise_{'mul' if mul else 'add'}_a{bits_of(it)}"
    got = run(name, a, b, ty(it)[0], OUT, ty(ot)[0], I64(n), *c, n=n, t=ot)
    check_flat(case, got, ot)


@pytest.mark.parametrize("d,case", cases(FLAT_FUSED))
def test_flat_fused(d, case):
    t = load(case)
    n = t["ifm_sv_size"] * t["ifm_sv_iteration"]
    it, ot = t["ifm_dtype"], t["ofm_dtype"]
    a = data(case, "ifm32.txt", ty(it)[1])[:n]
    b = data(case, "wts32.txt", ty(it)[1])[:n]
    mul = "_mul_" in d
    nc = 4 if mul else 3  # coefficients, then the two activation words
    w = fwords(words(case)[:nc + 2])
    act = TEMPLATE_ACT[t["activation_type_templates"]]
    op = "mul" if mul else ("sub" if "_sub_" in d else "add")
    got = run(f"npu_elementwise_{op}_fused_a{bits_of(it)}", a, b, ty(it)[0], OUT, ty(ot)[0],
              I64(n), *w[:nc], act, w[nc], w[nc + 1], n=n, t=ot)
    check_flat(case, got, ot)


# ---- Xint8 ----------------------------------------------------------------------------

@pytest.mark.parametrize("d,case", cases(XINT))
def test_xint(d, case):
    t = load(case)
    if d == "xint8_mac_wrapper":
        x1 = volume(case, t, "ifm1", "ifm32.txt", int_type(8, t["signed_ifm1"]))
        x2 = volume(case, t, "ifm2", "wts32.txt", int_type(8, t["signed_ifm2"]))
        x3 = volume(case, t, "ifm3", "ifm2_32.txt", int_type(8, t["signed_ifm3"]))
        A, v, p, tt = (int(x) for x in words(case)[:4].view(np.int32))
        ot = int_type(8, t["signed_ofm"])
        got = run("npu_elementwise_mac_a8", x1, ty(int_type(8, t["signed_ifm1"]))[0], x2,
                  ty(int_type(8, t["signed_ifm2"]))[0], x3, ty(int_type(8, t["signed_ifm3"]))[0],
                  OUT, ty(ot)[0], shp(x1.shape), shp(x2.shape), shp(x3.shape), A, v, p, tt,
                  n=x1.size, t=ot)
        check_hwc(case, got, ot)
        return
    n = t["ifm_sv_size"] * t["ifm_sv_iteration"]
    a = data(case, "ifm32.txt", np.int8)[:n]
    b = data(case, "wts32.txt", np.int8)[:n]
    w = fwords(words(case)[:5])
    if d == "xint8_add_wrapper":
        got = run("npu_elementwise_add_a8_xint", a, b, OUT, I64(n), *w[:3], n=n, t="int8")
    else:
        act = TEMPLATE_ACT[t["activation_type_templates"]]
        got = run("npu_elementwise_add_fused_a8_xint", a, b, OUT, I64(n), w[0], w[1], act, w[3],
                  w[4], n=n, t="int8")
    check_flat(case, got, "int8")


# ---- Broadcasting3d -------------------------------------------------------------------

def bc_types(t):
    bits = bits_of(t["dtype"])
    return int_type(bits, t["signed_ifm"]), int_type(bits, t["signed_ofm"]), bits


@pytest.mark.parametrize("d,case", cases(BC3D))
def test_bc3d(d, case):
    t = load(case)
    it, ot, bits = bc_types(t)
    a = volume(case, t, "ifm1", "ifm32.txt", it)
    b = volume(case, t, "ifm2", "wts32.txt", it)
    mul = "_mul_" in d
    c = fwords(words(case)[:4 if mul else 3])
    got = run(f"npu_qdq_elementwise_broadcasting3d_{'mul' if mul else 'add'}_a{bits}", a, b,
              ty(it)[0], OUT, ty(ot)[0], shp(a.shape), shp(b.shape), *c, n=a.size, t=ot)
    check_hwc(case, got, ot)


@pytest.mark.parametrize("d,case", cases(BC3D_FUSED))
def test_bc3d_fused(d, case):
    t = load(case)
    dt = t["dtype"]
    a = volume(case, t, "ifm1", "ifm32.txt", dt)
    b = volume(case, t, "ifm2", "wts32.txt", dt)
    act = TEMPLATE_ACT[t["activation_type_templates"]]
    w = words(case)
    mul = "_mul_" in d
    slope = None
    if act == ACT_PRELU:
        # one block per channel tile: a 32-word header, then the tile's slopes
        sv = t["ifm1_physical_sv_channel"]
        blocks = w[:(32 + sv) * t["ifm1_channel_iteration"]].reshape(-1, 32 + sv)
        slope = np.ascontiguousarray(blocks[:, 32:].reshape(-1)[:a.shape[2]]).view(np.float32)
    k = Host(np.ascontiguousarray(w[:9 if mul else 8]))
    op = "mul" if mul else ("sub" if "_sub_" in d else "add")
    got = run(f"npu_qdq_elementwise_broadcasting3d_{op}_fused_a{bits_of(dt)}", a, b, ty(dt)[0],
              OUT, shp(a.shape), shp(b.shape), k, act, slope, n=a.size, t=dt)
    check_hwc(case, got, dt)


@pytest.mark.parametrize("d,case", cases(MAC_FUSED))
def test_mac_fused(d, case):
    t = load(case)
    dt, wt, ct = t["dtype"], t["wts_dtype"], t["addend_dtype"]
    a = volume(case, t, "ifm1", "ifm32.txt", dt)
    b = volume(case, t, "ifm2", "wts32.txt", wt)
    c = volume(case, t, "ifm3", "ifm2_32.txt", ct)
    k = Host(np.ascontiguousarray(words(case)[:8]))
    variant = d.split("mac_fused_")[1].split("_wrapper")[0]
    got = run(f"npu_qdq_elementwise_broadcasting3d_mac_fused_{variant}", a, ty(dt)[0], b,
              ty(wt)[0], c, ty(ct)[0], OUT, shp(a.shape), shp(b.shape), shp(c.shape), k,
              MAC_ACT[t["activation_type"]], n=a.size, t=dt)
    check_hwc(case, got, dt)
