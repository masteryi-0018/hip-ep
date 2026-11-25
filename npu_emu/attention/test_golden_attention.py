# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Flash attention entry points against golden_attention.py, byte for byte."""

import ctypes

import numpy as np
import pytest

import golden_attention as ga
from golden_block_fp import encode_bfp16
from golden_float_cvt import f32_to_bf16, kFtzIn
from golden_int_round import Rnd
from npu import I64, NPU_I16, NPU_U8, NPU_U16, OUT, Host, Out, call, callx, raw_rc

f32 = np.float32
RNG = np.random.default_rng(20261116)


class Cfg(ctypes.Structure):
    """npu_flash_attention_cfg."""
    _fields_ = [(n, ctypes.c_int32) for n in (
        "enable_qdq", "block_row_dim", "m_granularity", "block_column_dim", "tile_k",
        "k_seq_tail_value", "mask_type", "enable_sliding_window", "enable_shuffle_transpose",
        "mat_v_transpose", "variant_a", "alt_round")] + [
        ("sliding_window_size", ctypes.c_int64), ("q_mask_offset", ctypes.c_int64),
        ("q_shards", ctypes.c_int32), ("q_shard", ctypes.c_int32),
        ("q_mask_wrap", ctypes.c_int32), ("alpha_bf16", ctypes.c_uint16),
        ("matmul_qk_c", ctypes.c_float * 4), ("matmul_pv_c", ctypes.c_float * 4),
        ("p_c1", ctypes.c_float), ("p_c0", ctypes.c_uint16), ("ofm_c0", ctypes.c_uint16),
        ("ofm_c1", ctypes.c_float), ("mask_dequant_c0", ctypes.c_float),
        ("mask_dequant_c1", ctypes.c_float), ("q_group_size", ctypes.c_int32),
        ("kv_group_size", ctypes.c_int32)]


def cstruct(cfg):
    c = Cfg()
    for name, _ in Cfg._fields_:
        val = cfg[name]
        if name in ("matmul_qk_c", "matmul_pv_c"):
            getattr(c, name)[:] = [float(f32(x)) for x in val]
        else:
            setattr(c, name, val)
    return c


def sizes(cfg, mask=True):
    """The size arguments of an entry: heads, q_len, kv_len, head_dim, v_head_dim[, mask_heads]."""
    names = ("heads", "q_len", "kv_len", "head_dim", "v_head_dim") + (("mask_heads",) * mask)
    return tuple(I64(cfg[n]) for n in names)


def entry(kind, q, k, v, mask, cfg):
    """Run the entry point of `kind` on host arrays; returns the output array."""
    c = Host(ctypes.byref(cstruct(cfg)))
    H, M, dv = cfg["heads"], cfg["q_len"], cfg["v_head_dim"]
    m = None if mask is None else np.ascontiguousarray(mask, np.uint16)
    if kind == "bf16":
        return call("npu_flash_attention_bf16", q, k, v, m, OUT, *sizes(cfg), c,
                    out_nbytes=H * M * dv * 2, out_dtype=np.uint16).reshape(H, M, dv)
    if kind == "q16":
        dt = np.int16 if cfg["signed"] else np.uint16
        return call("npu_flash_attention_q16", q, NPU_I16 if cfg["signed"] else NPU_U16, k, v, m,
                    OUT, *sizes(cfg), c,
                    out_nbytes=H * M * dv * 2, out_dtype=dt).reshape(H, M, dv)
    if kind == "u8":
        return call("npu_flash_attention_u8", q, k, v, m, OUT, *sizes(cfg), c,
                    out_nbytes=H * M * dv, out_dtype=np.uint8).reshape(H, M, dv)
    nb = H * M * dv // 8 * 9
    return call("npu_flash_attention_mx", q, k, v, OUT, *sizes(cfg, False), c,
                out_nbytes=nb, out_dtype=np.uint8).reshape(H, M, dv // 8 * 9)


def state_io_entry(q, k, v, m, l, u, cfg, final):
    """npu_flash_attention_state_io on host arrays: (m, l, out)."""
    c = Host(ctypes.byref(cstruct(cfg)))
    H, M, dv = cfg["heads"], cfg["q_len"], cfg["v_head_dim"]
    res = callx("npu_flash_attention_state_io", q, k, v,
                *(np.ascontiguousarray(x, np.uint16) for x in (m, l, u)), Out(0), Out(1), Out(2),
                *sizes(cfg, False), c, int(final),
                outs=[H * M * 2, H * M * 2, H * M * dv * 2])
    return (res[0].view(np.uint16).reshape(H, M), res[1].view(np.uint16).reshape(H, M),
            res[2].view(np.uint16).reshape(H, M, dv))


def state_cache_entry(q, k, v, st, cfg, flags):
    """npu_flash_attention_state_cache on host arrays; st = (m, l, u) host state, returned
    updated with out (None when not final)."""
    c = Host(ctypes.byref(cstruct(cfg)))
    H, M, dv = cfg["heads"], cfg["q_len"], cfg["v_head_dim"]
    m, l, u = (np.ascontiguousarray(x) for x in st)
    res = callx("npu_flash_attention_state_cache", q, k, v, Out(0), Out(1), Out(2), Out(3),
                *sizes(cfg, False), c, flags, outs=[m, l, u, H * M * dv * 2])
    out = res[3].view(np.uint16).reshape(H, M, dv) if flags & 1 else None
    return (res[0].view(np.uint16).reshape(H, M), res[1].view(np.uint16).reshape(H, M),
            res[2].view(f32).reshape(H, M, dv)), out


MODEL = {"bf16": ga.flash_attention_bf16, "q16": ga.flash_attention_q16,
         "u8": ga.flash_attention_u8}


def check(kind, q, k, v, mask, cfg):
    if kind == "mx":
        want = np.asarray(ga.flash_attention_mx(q, k, v, cfg))
    else:
        want = np.asarray(MODEL[kind](q, k, v, mask, cfg))
    got = entry(kind, q, k, v, mask, cfg)
    want = want.astype(got.dtype)
    bad = np.argwhere(got != want)
    kk = tuple(bad[:4].T)
    assert bad.size == 0, (f"{len(bad)}/{got.size} differ, first {bad[:4].tolist()}: "
                           f"got {got[kk].tolist()} want {want[kk].tolist()}")
    return got


def base_cfg(**kw):
    c = dict(heads=1, q_len=64, kv_len=128, head_dim=64, v_head_dim=64, mask_heads=1,
             sliding_window_size=0, q_mask_offset=0, block_row_dim=64, m_granularity=64,
             block_column_dim=128, tile_k=1, k_seq_tail_value=0, mask_type=0,
             enable_sliding_window=0, q_shards=1, q_shard=0, q_mask_wrap=1, enable_qdq=0,
             variant_a=0, enable_shuffle_transpose=0, alt_round=0, mat_v_transpose=0,
             alpha_bf16=0x3E00, matmul_qk_c=[0.0] * 4, matmul_pv_c=[0.0] * 4, p_c1=0.0, p_c0=0,
             ofm_c1=0.0, ofm_c0=0, mask_dequant_c0=1.0, mask_dequant_c1=0.0, signed=0,
             q_group_size=1, kv_group_size=1)
    c.update(kw)
    if "m_granularity" not in kw:
        c["m_granularity"] = c["block_row_dim"]
    return c


def bf16(x):
    return f32_to_bf16(np.asarray(x, f32), Rnd.kHalfEven, kFtzIn)


# ---- bf16 ---------------------------------------------------------------------------

def bf16_inputs(cfg, kind="normal"):
    H, M, N, d, dv = (cfg[n] for n in ("heads", "q_len", "kv_len", "head_dim", "v_head_dim"))
    if kind == "normal":
        q, k, v = (RNG.standard_normal((H, r, w)) for r, w in ((M, d), (N, d), (N, dv)))
    elif kind == "spread":
        # scores over many binades: P underflows, subnormal exp2 outputs
        q = RNG.standard_normal((H, M, d)) * np.exp2(RNG.integers(-4, 6, (H, M, 1)))
        k = RNG.standard_normal((H, N, d)) * np.exp2(RNG.integers(-3, 5, (H, N, 1)))
        v = RNG.standard_normal((H, N, dv)) * np.exp2(RNG.integers(-30, 30, (H, N, dv)))
    else:  # "ties": few distinct values, many equal row maxima / scores
        q = RNG.integers(-2, 3, (H, M, d)) * 0.5
        k = RNG.integers(-2, 3, (H, N, d)) * 0.25
        v = RNG.integers(-4, 5, (H, N, dv)) * 0.125
    return bf16(q), bf16(k), bf16(v)


BF16_SHAPES = [  # br, bc, M, N, d, dv
    (64, 128, 64, 256, 64, 64), (64, 64, 128, 192, 17, 17), (32, 128, 64, 256, 96, 96),
    (8, 64, 16, 128, 128, 128), (8, 192, 8, 384, 32, 48), (1, 64, 4, 256, 64, 64),
    (1, 128, 3, 256, 96, 96)]


@pytest.mark.parametrize("alt_round", [0, 1])
@pytest.mark.parametrize("shape", BF16_SHAPES, ids=lambda s: "br%d_bc%d_M%d_N%d_d%d_dv%d" % s)
def test_bf16_random(shape, alt_round):
    br, bc, M, N, d, dv = shape
    cfg = base_cfg(block_row_dim=br, block_column_dim=bc, q_len=M, kv_len=N, head_dim=d,
                   v_head_dim=dv, alt_round=alt_round,
                   heads=2, alpha_bf16=int(bf16([1 / np.sqrt(d)])[0]))
    check("bf16", *bf16_inputs(cfg), None, cfg)


@pytest.mark.parametrize("kind", ["spread", "ties"])
@pytest.mark.parametrize("br", [64, 8, 1])
def test_bf16_constructed(kind, br):
    cfg = base_cfg(block_row_dim=br, block_column_dim=64, q_len={64: 64, 8: 16, 1: 4}[br],
                   kv_len=192, head_dim=64,
                   v_head_dim=32, alpha_bf16=0x3F80)
    check("bf16", *bf16_inputs(cfg, kind), None, cfg)


@pytest.mark.parametrize("br", [64, 32, 1])
@pytest.mark.parametrize("tail", [1, 41, 88, 127])
def test_bf16_kv_tail(br, tail):
    cfg = base_cfg(block_row_dim=br, block_column_dim=128, q_len=max(br, 2), kv_len=256,
                   k_seq_tail_value=tail)
    check("bf16", *bf16_inputs(cfg), None, cfg)


CAUSAL = [  # br, bc, M, N, q_offset, q_shards, q_shard, q_wrap, sliding, window, tail
    (64, 64, 128, 256, 0, 1, 0, 2, 0, 0, 0), (64, 128, 128, 256, 64, 1, 0, 2, 0, 0, 0),
    (32, 64, 64, 256, 32, 12, 3, 2, 0, 0, 0), (64, 64, 128, 512, 100, 2, 1, 2, 0, 0, 0),
    (64, 64, 128, 256, 0, 1, 0, 2, 1, 8, 0), (64, 128, 64, 384, 191, 1, 0, 1, 1, 129, 0),
    (1, 64, 4, 512, 319, 0, 0, 1, 1, 1, 0), (1, 64, 2, 256, 255, 0, 0, 1, 1, 40, 0),
    (1, 128, 1, 256, 160, 0, 0, 1, 0, 0, 0), (32, 64, 64, 192, 41, 1, 0, 2, 1, 24, 41),
    (64, 64, 128, 256, 64, 1, 0, 2, 1, 1, 0)]


@pytest.mark.parametrize("alt_round", [0, 1])
@pytest.mark.parametrize("p", CAUSAL, ids=lambda p: "_".join(map(str, p)))
def test_bf16_causal(p, alt_round):
    br, bc, M, N, off, sh, shard, wrap, sl, w, tail = p
    cfg = base_cfg(block_row_dim=br, block_column_dim=bc, q_len=M, kv_len=N, mask_type=1,
                   q_mask_offset=off, q_shards=sh,
                   q_shard=shard, q_mask_wrap=wrap, enable_sliding_window=sl, sliding_window_size=w,
                   k_seq_tail_value=tail, alt_round=alt_round)
    check("bf16", *bf16_inputs(cfg), None, cfg)


@pytest.mark.parametrize("mkind", ["pad", "values", "neg_inf_some"])
def test_bf16_mask(mkind):
    cfg = base_cfg(block_row_dim=64, block_column_dim=128, q_len=128, kv_len=256, mask_type=2,
                   heads=2, mask_heads=2)
    q, k, v = bf16_inputs(cfg)
    N = cfg["kv_len"]
    if mkind == "pad":
        m = np.where(RNG.random((2, N)) > 0.1, 0.0, -1000.0)
    elif mkind == "values":
        m = RNG.standard_normal((2, N)) * 4
    else:
        m = np.where(RNG.random((2, N)) > 0.3, 0.0, -np.inf)
        m[:, 0] = 0.0
    check("bf16", q, k, v, bf16(m), cfg)


# ---- 16-bit QDQ -----------------------------------------------------------------------

def q16_inputs(cfg, signed, zp=(30000, 31000, 32000), rng=900):
    H, M, N, d, dv = (cfg[n] for n in ("heads", "q_len", "kv_len", "head_dim", "v_head_dim"))
    dt = np.int16 if signed else np.uint16
    if signed:
        zp = tuple(z - 32768 for z in zp)
    lo, hi = (-32768, 32767) if signed else (0, 65535)
    mk = lambda z, s: np.clip(RNG.integers(z - rng, z + rng, s), lo, hi).astype(dt)
    return mk(zp[0], (H, M, d)), mk(zp[1], (H, N, d)), mk(zp[2], (H, N, dv)), zp


def qdq_coeffs(cfg, zp, s=(0.004, 0.005, 0.003), so=0.002, zo=30000, p_scale=1 / 65535.0,
               pzp=0, u8=False):
    zq, zk, zv = zp
    sq, sk, sv = s
    d = cfg["head_dim"]
    a = 1 / np.sqrt(d)
    if cfg["enable_qdq"] == 1 and not u8:
        cfg["matmul_qk_c"] = [sq * np.sqrt(a), -zq * sq * np.sqrt(a), sk * np.sqrt(a),
                       -zk * sk * np.sqrt(a)]
    else:
        g = a * sq * sk
        cfg["matmul_qk_c"] = [g * d * zq * zk, -g * zk, g, -g * zq]
    if cfg["enable_qdq"] == 1:
        cfg["matmul_pv_c"] = [-zv * sv, sv, 0.0, 0.0]
    else:
        cfg["matmul_pv_c"] = [pzp * zv * p_scale * sv * cfg["block_column_dim"], -p_scale * sv * zv,
                              p_scale * sv,
                       -pzp * p_scale * sv]
        cfg["p_c1"] = float(f32(1 / p_scale))
        cfg["p_c0"] = pzp
    cfg["ofm_c1"] = float(f32(1 / so))
    cfg["ofm_c0"] = zo
    return cfg


Q16 = [  # qdq, br, bc, M, N, d, signed, shuffle, mask_type
    (1, 64, 128, 64, 256, 64, 0, 1, 0), (1, 32, 64, 64, 192, 96, 0, 0, 0),
    (1, 1, 64, 2, 256, 64, 0, 0, 0), (1, 1, 128, 2, 256, 128, 1, 0, 0),
    (1, 64, 64, 128, 128, 64, 1, 1, 0), (1, 64, 64, 64, 256, 64, 0, 1, 1),
    (1, 1, 64, 4, 256, 64, 0, 0, 1), (2, 64, 64, 64, 256, 64, 0, 0, 0),
    (2, 64, 128, 128, 256, 128, 0, 0, 1), (1, 64, 128, 64, 256, 64, 0, 0, 2),
    (2, 64, 64, 64, 128, 64, 0, 0, 2)]


@pytest.mark.parametrize("alt_round", [0, 1])
@pytest.mark.parametrize("p", Q16, ids=lambda p: "_".join(map(str, p)))
def test_q16(p, alt_round):
    qdq, br, bc, M, N, d, signed, shuffle, mt = p
    cfg = base_cfg(enable_qdq=qdq, variant_a=int(qdq == 2), block_row_dim=br, block_column_dim=bc,
                   q_len=M, kv_len=N, head_dim=d, v_head_dim=d, signed=signed,
                   enable_shuffle_transpose=shuffle, mask_type=mt, alt_round=alt_round,
                   m_granularity=16 if qdq == 2 else br)
    if mt == 1:
        # the sliding window is built for qdq 1 only
        cfg.update(q_mask_offset=64, q_shards=1, q_mask_wrap=1, enable_sliding_window=int(qdq == 1),
                   sliding_window_size=40)
    q, k, v, zp = q16_inputs(cfg, signed)
    qdq_coeffs(cfg, zp, zo=(0 if signed else 30000))
    mask = None
    if mt == 2:
        cfg["mask_dequant_c0"], cfg["mask_dequant_c1"] = 1.0, -1000.0
        mask = np.where(RNG.random((1, N)) > 0.1, 1000, 0).astype(np.uint16)
    check("q16", q, k, v, mask, cfg)


def test_q16_saturation():
    """Wide codes and a small OFM scale: outputs hit both ends."""
    cfg = base_cfg(enable_qdq=1, block_row_dim=64, block_column_dim=64, q_len=64, kv_len=128)
    q, k, v, zp = q16_inputs(cfg, 0, rng=30000, zp=(32768, 32768, 32768))
    qdq_coeffs(cfg, zp, so=1e-5)
    got = check("q16", q, k, v, None, cfg)
    assert (got == 0).any() and (got == 65535).any()


# ---- uint8 QDQ --------------------------------------------------------------------------

U8 = [  # qdq, br, mgran, bc, k_tile, M, N, d, mask_type
    (1, 64, 64, 64, 1, 64, 128, 64, 0), (1, 32, 32, 128, 2, 64, 256, 128, 0),
    (1, 64, 64, 64, 1, 64, 128, 192, 0), (1, 8, 8, 64, 1, 16, 128, 128, 0),
    (1, 1, 1, 64, 1, 2, 192, 128, 0), (1, 64, 64, 128, 2, 64, 256, 64, 2),
    (2, 64, 64, 128, 2, 64, 256, 64, 0), (2, 64, 64, 128, 1, 128, 256, 128, 0),
    (2, 64, 64, 64, 1, 64, 128, 64, 2), (2, 64, 64, 256, 4, 64, 256, 64, 0)]


@pytest.mark.parametrize("alt_round", [0, 1])
@pytest.mark.parametrize("p", U8, ids=lambda p: "_".join(map(str, p)))
def test_u8(p, alt_round):
    qdq, br, mg, bc, kt, M, N, d, mt = p
    cfg = base_cfg(enable_qdq=qdq, variant_a=int(qdq == 2), block_row_dim=br, m_granularity=mg,
                   block_column_dim=bc, tile_k=kt,
                   q_len=M, kv_len=N, head_dim=d, v_head_dim=d, mask_type=mt, alt_round=alt_round)
    zp = (128, 120, 135)
    q, k, v = (RNG.integers(z - 50, z + 50, s).astype(np.uint8)
               for z, s in zip(zp, ((1, M, d), (1, N, d), (1, N, d))))
    qdq_coeffs(cfg, zp, s=(0.01, 0.01, 0.01), so=0.002, zo=128, p_scale=1 / 255.0, u8=True)
    mask = None
    if mt == 2:
        cfg["mask_dequant_c0"], cfg["mask_dequant_c1"] = 1.0, -1000.0
        mask = np.where(RNG.random((1, N)) > 0.1, 1000, 0).astype(np.uint16)
    check("u8", q, k, v, mask, cfg)


# ---- mxint8_ebs8 --------------------------------------------------------------------------

def mx_pack(x, zero_blocks=0.0):
    """fp32 [H][rows][L] -> packed bytes [H][rows][L/8*9]; zero_blocks: share of zeroed blocks."""
    x = np.array(x, f32)
    H, R, L = x.shape
    blk = x.reshape(H, R, L // 8, 8)
    blk[RNG.random((H, R, L // 8)) < zero_blocks] = 0
    return encode_bfp16(x.reshape(-1), Rnd.kHalfAway).reshape(H, R, L // 8 * 9)


MX = [  # br, bc, M, N, d, dv, mask_type, v_transposed
    (64, 64, 64, 128, 64, 64, 0, 0), (32, 128, 64, 256, 128, 96, 1, 0),
    (8, 64, 16, 192, 64, 64, 0, 1), (2, 128, 4, 256, 192, 192, 1, 0),
    (1, 64, 2, 256, 256, 256, 1, 0), (1, 128, 2, 256, 64, 64, 0, 1),
    (64, 64, 128, 256, 64, 64, 1, 1), (64, 64, 64, 128, 64, 64, 0, 3),
    (64, 128, 128, 256, 128, 64, 0, 3)]


@pytest.mark.parametrize("zero", [0.0, 0.3])
@pytest.mark.parametrize("p", MX, ids=lambda p: "_".join(map(str, p)))
def test_mx(p, zero):
    br, bc, M, N, d, dv, mt, vt = p
    cfg = base_cfg(block_row_dim=br, block_column_dim=bc, q_len=M, kv_len=N, head_dim=d,
                   v_head_dim=dv, mask_type=mt,
                   mat_v_transpose=vt, alpha_bf16=int(bf16([1 / np.sqrt(d)])[0]))
    if mt == 1:
        cfg.update(q_mask_offset=N - M if br > 1 else N - 1, q_shards=1 if br > 1 else 0)
    q = mx_pack(RNG.standard_normal((1, M, d)) * np.exp2(RNG.integers(-6, 6, (1, M, d))),
                zero_blocks=zero)
    k = mx_pack(RNG.standard_normal((1, N, d)), zero_blocks=zero)
    vv = RNG.standard_normal((1, N, dv)) * np.exp2(RNG.integers(-20, 20, (1, N, dv)))
    v = mx_pack(np.swapaxes(vv, 1, 2) if vt else vv, zero_blocks=zero)
    check("mx", q, k, v, None, cfg)


def test_mx_extreme_exponents():
    """Block exponents near both ends (subnormal V values, huge Q blocks)."""
    cfg = base_cfg(block_row_dim=64, block_column_dim=64, q_len=64, kv_len=128, head_dim=64,
                   v_head_dim=64)
    q = RNG.integers(-128, 128, (1, 64, 8, 9)).astype(np.uint8)
    q[..., 0] = RNG.integers(120, 140, (1, 64, 8))
    k = RNG.integers(-128, 128, (1, 128, 8, 9)).astype(np.uint8)
    k[..., 0] = RNG.integers(120, 140, (1, 128, 8))
    v = RNG.integers(-128, 128, (1, 128, 8, 9)).astype(np.uint8)
    v[..., 0] = RNG.choice([0, 1, 5, 20, 133, 250], (1, 128, 8))
    check("mx", q.reshape(1, 64, 72), k.reshape(1, 128, 72), v.reshape(1, 128, 72), None, cfg)


# ---- grouped mask --------------------------------------------------------------------------

GROUPED = [  # kind, br, bc, M, N, d, q_group, kv_group, q_offset, q_shards, q_shard, q_wrap, tail
    ("mx", 8, 64, 32, 64, 128, 3, 10, 0, 1, 0, 2, 60),
    ("mx", 8, 64, 32, 64, 64, 12, 20, 0, 2, 1, 2, 60),
    ("mx", 32, 64, 64, 64, 128, 4, 8, 0, 1, 0, 1, 0),
    ("mx", 64, 128, 128, 256, 64, 6, 8, 5, 1, 0, 2, 0),
    ("u8", 32, 64, 64, 64, 64, 6, 5, 0, 1, 0, 2, 0),
    ("u8", 8, 64, 32, 64, 128, 6, 10, 2, 2, 1, 2, 60),
    ("u8", 64, 128, 128, 128, 64, 16, 12, 0, 1, 0, 2, 100)]


@pytest.mark.parametrize("p", GROUPED, ids=lambda p: "_".join(map(str, p)))
def test_grouped(p):
    kind, br, bc, M, N, d, qg, kvg, off, sh, shard, wrap, tail = p
    cfg = base_cfg(block_row_dim=br, block_column_dim=bc, q_len=M, kv_len=N, head_dim=d,
                   v_head_dim=d, mask_type=7, q_group_size=qg, kv_group_size=kvg, q_mask_offset=off,
                   q_shards=sh, q_shard=shard, q_mask_wrap=wrap, k_seq_tail_value=tail,
                   alpha_bf16=int(bf16([1 / np.sqrt(d)])[0]))
    if kind == "mx":
        q = mx_pack(RNG.standard_normal((1, M, d)))
        k = mx_pack(RNG.standard_normal((1, N, d)))
        v = mx_pack(RNG.standard_normal((1, N, d)))
    elif kind == "u8":
        cfg.update(enable_qdq=1)
        zp = (128, 120, 135)
        q, k, v = (RNG.integers(z - 50, z + 50, s).astype(np.uint8)
                   for z, s in zip(zp, ((1, M, d), (1, N, d), (1, N, d))))
        qdq_coeffs(cfg, zp, s=(0.01, 0.01, 0.01), so=0.002, zo=128, p_scale=1 / 255.0, u8=True)
    else:
        q, k, v = bf16_inputs(cfg)
    check(kind, q, k, v, None, cfg)


# ---- state entries ---------------------------------------------------------------------------

STATE = [  # M, N, bc, d, q_offset, q_wrap, sliding, window, calls
    (64, 64, 64, 64, 63, 1, 0, 0, 3), (128, 128, 64, 128, 255, 1, 0, 0, 2),
    (64, 256, 64, 256, 511, 1, 0, 0, 2), (64, 192, 64, 64, 192, 1, 1, 1, 1),
    (128, 256, 128, 64, 128, 2, 1, 80, 2), (64, 128, 64, 64, 100, 1, 0, 0, 3),
    (64, 128, 64, 64, -1, 1, 0, 0, 2), (64, 64, 64, 64, -100, 1, 0, 0, 2)]


def state_cfg(p, call):
    M, N, bc, d, off, wrap, sl, w, calls = p
    return base_cfg(block_row_dim=64, block_column_dim=bc, q_len=M, kv_len=N, head_dim=d,
                    v_head_dim=d, mask_type=1, q_mask_offset=off - call * N, q_shards=1,
                    q_mask_wrap=wrap, enable_sliding_window=sl, sliding_window_size=w,
                    alpha_bf16=0x3F80)


def state_inputs(cfg, calls, prior):
    M, N, d = cfg["q_len"], cfg["kv_len"], cfg["head_dim"]
    q = bf16(RNG.standard_normal((1, M, d)) * 0.25)
    kv = [(bf16(RNG.standard_normal((1, N, d)) * 0.25 + c * 0.5),
           bf16(RNG.standard_normal((1, N, d)) * 0.5)) for c in range(calls)]
    if prior:
        m = bf16(RNG.standard_normal((1, M)))
        l = bf16(RNG.random((1, M)) * 8 + 1)
        u = bf16(RNG.standard_normal((1, M, d)) * 4)
    else:
        m, l, u = np.full((1, M), 0xFF80, np.uint16), np.zeros((1, M), np.uint16), np.zeros(
            (1, M, d), np.uint16)
    return q, kv, (m, l, u)


@pytest.mark.parametrize("prior", [0, 1])
@pytest.mark.parametrize("p", STATE, ids=lambda p: "_".join(map(str, p)))
def test_state_io(p, prior):
    calls = p[-1]
    q, kv, (m, l, u) = state_inputs(state_cfg(p, 0), calls, prior)
    for call, (k, v) in enumerate(kv):
        cfg = state_cfg(p, call)
        final = call == calls - 1
        want = ga.flash_attention_state_io(q, k, v, m, l, u, cfg, final)
        got = state_io_entry(q, k, v, m, l, u, cfg, final)
        for g, w in zip(got, want):
            assert_equal(g, w)
        m, l, u = got


@pytest.mark.parametrize("flags", [(3, 0, 1), (3, 3, 1), (2, 1, 2)])
@pytest.mark.parametrize("p", STATE, ids=lambda p: "_".join(map(str, p)))
def test_state_cache(p, flags):
    calls = p[-1]
    q, kv, _ = state_inputs(state_cfg(p, 0), calls, 0)
    M, d = p[0], p[3]
    st_w = (np.zeros((1, M), f32), np.zeros((1, M), f32), np.zeros((1, M, d), f32))
    st_g = (np.zeros((1, M), np.uint16), np.zeros((1, M), np.uint16), np.zeros((1, M, d), f32))
    for call, (k, v) in enumerate(kv):
        cfg = state_cfg(p, call)
        fl = flags[call]
        want = ga.flash_attention_state_cache(q, k, v, st_w, cfg, fl >> 1, fl & 1)
        st_g, got = state_cache_entry(q, k, v, st_g, cfg, fl)
        assert_equal(st_g[0], bf16(st_w[0]))
        assert_equal(st_g[1], bf16(st_w[1]))
        assert_equal(st_g[2].view(np.uint32), st_w[2].view(np.uint32))
        if fl & 1:
            assert_equal(got, want)


def assert_equal(got, want):
    bad = np.argwhere(got != want)
    kk = tuple(bad[:4].T)
    assert bad.size == 0, (f"{len(bad)}/{got.size} differ, first {bad[:4].tolist()}: "
                           f"got {got[kk].tolist()} want {want[kk].tolist()}")


# ---- argument checks ----------------------------------------------------------------

def test_bad_args():
    z = np.zeros(128 * 64 * 2, np.uint16)  # large enough for every tensor below

    def rc(name, cfg, *, q=z, out=z, mask=None, dtype=None, final=1, flags=3):
        c = Host(ctypes.byref(cstruct(cfg)))
        if name == "bf16" or name == "u8":
            args = (q, z, z, mask, out, *sizes(cfg), c)
        elif name == "q16":
            args = (q, NPU_U16 if dtype is None else dtype, z, z, mask, out, *sizes(cfg), c)
        elif name == "mx":
            args = (q, z, z, out, *sizes(cfg, False), c)
        elif name == "state_io":
            args = (q, z, z, z, z, z, z, z, out, *sizes(cfg, False), c, final)
        else:
            args = (q, z, z, z, z, z, out, *sizes(cfg, False), c, flags)
        return raw_rc("npu_flash_attention_" + name, *args)

    bf = base_cfg()
    assert rc("bf16", bf) == 0
    for bad in [dict(block_column_dim=100), dict(q_len=63), dict(enable_qdq=1), dict(mask_type=7),
                dict(mask_type=2), dict(block_row_dim=2, m_granularity=2, q_len=64),
                dict(m_granularity=16), dict(mat_v_transpose=2), dict(variant_a=2),
                dict(alt_round=-1), dict(q_shards=3), dict(q_shard=1), dict(q_mask_wrap=65536),
                dict(mask_type=1, q_mask_offset=-1), dict(q_mask_offset=1 << 24),
                dict(enable_sliding_window=1), dict(enable_sliding_window=2, mask_type=1),
                dict(enable_shuffle_transpose=1, block_row_dim=8, m_granularity=8),
                dict(k_seq_tail_value=129), dict(tile_k=0), dict(tile_k=256),
                dict(sliding_window_size=-1), dict(mask_heads=0), dict(heads=-1),
                dict(q_len=-64), dict(kv_len=0), dict(head_dim=0)]:
        assert rc("bf16", dict(bf, **bad)) == -1, bad
    assert rc("bf16", dict(bf, mask_type=2), mask=z) == 0
    assert rc("bf16", bf, q=None) == -1
    assert rc("bf16", bf, out=None) == -1
    # empty work (heads or q_len 0) is a no-op once the rest is valid; pointers stay required
    for empty in [dict(heads=0), dict(q_len=0)]:
        assert rc("bf16", dict(bf, **empty)) == 0, empty
        assert rc("bf16", dict(bf, **empty), q=None) == -1, empty
        assert rc("bf16", dict(bf, **empty, mask_type=2)) == -1, empty  # mask required
        assert rc("bf16", dict(bf, **empty, tile_k=0)) == -1, empty
    assert rc("q16", base_cfg(enable_qdq=1, q_len=0)) == 0
    assert rc("q16", base_cfg(enable_qdq=1, q_len=0), dtype=NPU_U8) == -1
    assert rc("u8", base_cfg(enable_qdq=1, heads=0)) == 0
    assert rc("mx", base_cfg(q_len=0)) == 0
    assert rc("mx", base_cfg(q_len=0), out=None) == -1
    assert rc("state_io", base_cfg(mask_type=1, heads=0)) == 0
    assert rc("state_io", base_cfg(mask_type=1, heads=0), out=None) == -1
    assert rc("state_cache", base_cfg(mask_type=1, q_len=0)) == 0
    assert rc("state_cache", base_cfg(mask_type=1, q_len=0), q=None) == -1

    q16 = base_cfg(enable_qdq=1)
    assert rc("q16", q16) == 0
    assert rc("q16", q16, dtype=NPU_I16) == 0
    assert rc("q16", q16, dtype=NPU_U8) == -1
    # int16 is built for qdq 1 without a mask only
    assert rc("q16", dict(q16, enable_qdq=2, m_granularity=16), dtype=NPU_I16) == -1
    assert rc("q16", dict(q16, mask_type=2), mask=z, dtype=NPU_I16) == -1
    assert rc("q16", dict(q16, mask_type=1), dtype=NPU_I16) == -1
    assert rc("q16", dict(q16, enable_qdq=2, m_granularity=16)) == 0
    for bad in [dict(enable_qdq=0), dict(enable_qdq=2), dict(m_granularity=16),
                dict(block_row_dim=8, m_granularity=8, q_len=64),
                dict(enable_qdq=2, m_granularity=16, enable_shuffle_transpose=1),
                dict(enable_qdq=2, m_granularity=16, mask_type=1, enable_sliding_window=1)]:
        assert rc("q16", dict(q16, **bad)) == -1, bad

    u8 = base_cfg(enable_qdq=1)
    assert rc("u8", u8) == 0
    for bad in [dict(enable_qdq=0), dict(enable_qdq=2, mask_type=7), dict(mask_type=7,
                                                                          q_group_size=0),
                dict(enable_shuffle_transpose=1), dict(mat_v_transpose=3)]:
        assert rc("u8", dict(u8, **bad)) == -1, bad

    mx = base_cfg()
    assert rc("mx", mx) == 0
    for bad in [dict(head_dim=32), dict(v_head_dim=12), dict(mask_type=2),
                dict(mask_type=1, enable_sliding_window=1),
                dict(mask_type=7, block_row_dim=2, m_granularity=2, q_len=64),
                dict(enable_shuffle_transpose=1)]:
        assert rc("mx", dict(mx, **bad)) == -1, bad

    st = base_cfg(mask_type=1, q_mask_offset=-5)  # relative positions may be negative
    assert rc("state_io", st) == 0
    assert rc("state_io", st, final=2) == -1
    assert rc("state_io", dict(st, mask_type=0)) == -1
    assert rc("state_io", st, out=None) == -1
    assert rc("state_cache", st) == 0
    assert rc("state_cache", st, flags=4) == -1
    assert rc("state_cache", dict(st, k_seq_tail_value=5)) == -1
    assert rc("state_cache", st, flags=1, out=None) == -1
    assert rc("state_cache", st, flags=2, out=None) == 0


# ---- special values, bit for bit ------------------------------------------------------

SPB = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F,
                0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFBF, 0x4B00, 0xCB00], np.uint16)
SPF = [0.0, -0.0, 1e-45, 1e30, -1e30, np.inf, -np.inf, np.nan]  # fp32 coefficients


def sprinkle(x, sp, rate=0.03):
    """Copy of x with a share of its entries replaced by values of sp."""
    x = x.copy()
    m = RNG.random(x.shape) < rate
    x[m] = RNG.choice(sp, int(m.sum()))
    return x


@pytest.mark.parametrize("mt", [0, 1])
@pytest.mark.parametrize("br", [64, 8, 1])
def test_bf16_special_inputs(br, mt):
    """NaN, inf, subnormal and signed-zero entries in Q, K, V."""
    M = {64: 64, 8: 16, 1: 4}[br]
    cfg = base_cfg(block_row_dim=br, block_column_dim=64, q_len=M, kv_len=192, head_dim=64,
                   v_head_dim=32, mask_type=mt,
                   q_mask_offset=100, q_shards=1 if br > 1 else 0, alpha_bf16=0x3F80)
    q, k, v = bf16_inputs(cfg)
    for rate in (0.002, 0.03):
        check("bf16", sprinkle(q, SPB, rate), sprinkle(k, SPB, rate), sprinkle(v, SPB, rate),
              None, cfg)


@pytest.mark.parametrize("alpha", [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x0001, 0x7F7F])
@pytest.mark.parametrize("br", [64, 1])
def test_bf16_special_scale(br, alpha):
    cfg = base_cfg(block_row_dim=br, block_column_dim=64, q_len=64 if br > 1 else 2, kv_len=128,
                   alpha_bf16=alpha)
    check("bf16", *bf16_inputs(cfg), None, cfg)


@pytest.mark.parametrize("br", [64, 8, 1])
@pytest.mark.parametrize("case", ["all_inf_mask", "first_block_masked",
                                  "window", "pos_inf_mask", "nan_mask"])
def test_bf16_masked_rows(br, case):
    """Rows whose scores are all -inf (in the first block, or in every block)."""
    M = {64: 64, 8: 16, 1: 4}[br]
    cfg = base_cfg(block_row_dim=br, block_column_dim=64, q_len=M, kv_len=192, head_dim=64,
                   v_head_dim=32,
                   alpha_bf16=0x3E00)
    mask = None
    if case in ("all_inf_mask", "first_block_masked", "pos_inf_mask", "nan_mask"):
        cfg.update(mask_type=2)
        mask = np.zeros((1, 192), np.uint16)
        if case == "all_inf_mask":
            mask[:] = 0xFF80
        elif case == "first_block_masked":
            mask[:, :64] = 0xFF80
        elif case == "pos_inf_mask":
            mask[:, ::5] = 0x7F80
            mask[:, :64] = 0xFF80
        else:
            mask[:, ::7] = 0x7FC0
    else:
        cfg.update(mask_type=1, q_mask_offset=130, q_shards=1 if br > 1 else 0,
                   enable_sliding_window=1,
                   sliding_window_size=1, k_seq_tail_value=5)
    check("bf16", *bf16_inputs(cfg), mask, cfg)


@pytest.mark.parametrize("kind", ["q16", "u8"])
@pytest.mark.parametrize("qdq", [1, 2])
@pytest.mark.parametrize("field,val", [(f, v) for f in ("matmul_qk_c", "matmul_pv_c", "ofm_c1",
                                                        "p_c1")
                                       for v in SPF])
def test_qdq_special_coeffs(kind, qdq, field, val):
    """Degenerate coefficients (0, inf, NaN, huge, subnormal) on integer inputs."""
    u8 = kind == "u8"
    cfg = base_cfg(enable_qdq=qdq, variant_a=int(qdq == 2), block_row_dim=64,
                   m_granularity=16 if (qdq == 2 and not u8) else 64, block_column_dim=64, q_len=64,
                   kv_len=128, head_dim=64, v_head_dim=64, alt_round=1)
    if u8:
        zp = (128, 120, 135)
        q, k, v = (RNG.integers(z - 50, z + 50, s).astype(np.uint8)
                   for z, s in zip(zp, ((1, 64, 64), (1, 128, 64), (1, 128, 64))))
        qdq_coeffs(cfg, zp, s=(0.01, 0.01, 0.01), so=0.002, zo=128, p_scale=1 / 255.0, u8=True)
    else:
        q, k, v, zp = q16_inputs(cfg, 0)
        qdq_coeffs(cfg, zp)
    if field in ("matmul_qk_c", "matmul_pv_c"):
        cfg[field] = [val] * 4 if qdq == 2 else [val, cfg[field][1], cfg[field][2], val]
    else:
        cfg[field] = val
    check(kind, q, k, v, None, cfg)


@pytest.mark.parametrize("kind", ["q16", "u8"])
@pytest.mark.parametrize("qdq", [1, 2])
def test_qdq_extreme_codes(kind, qdq):
    """Codes at both ends of the type and extreme zero points / offsets."""
    u8 = kind == "u8"
    cfg = base_cfg(enable_qdq=qdq, variant_a=int(qdq == 2), block_row_dim=64,
                   m_granularity=16 if (qdq == 2 and not u8) else 64, block_column_dim=64, q_len=64,
                   kv_len=128, head_dim=64, v_head_dim=64)
    hi = 255 if u8 else 65535
    dt = np.uint8 if u8 else np.uint16
    ends = np.array([0, 1, hi - 1, hi], dt)
    q, k, v = (RNG.choice(ends, s) for s in ((1, 64, 64), (1, 128, 64), (1, 128, 64)))
    zp = (hi // 2, hi // 2, hi // 2)
    qdq_coeffs(cfg, zp, s=(0.01, 0.01, 0.01), so=0.002, zo=hi // 2, p_scale=1 / hi, u8=u8)
    for zo in (0, 0xFFFF, 0x8000, hi + 1, 0x7FFF):
        cfg["ofm_c0"] = zo
        cfg["p_c0"] = zo
        check(kind, q, k, v, None, cfg)


@pytest.mark.parametrize("val", [0.0, np.inf, -np.inf, np.nan, 1e30, -1e30])
def test_q16_special_mask(val):
    cfg = base_cfg(enable_qdq=1, block_row_dim=64, block_column_dim=64, q_len=64, kv_len=128,
                   mask_type=2)
    q, k, v, zp = q16_inputs(cfg, 0)
    qdq_coeffs(cfg, zp)
    cfg["mask_dequant_c0"], cfg["mask_dequant_c1"] = val, val
    mask = RNG.choice(np.array([0, 1, 65535], np.uint16), (1, 128))
    check("q16", q, k, v, mask, cfg)


@pytest.mark.parametrize("alpha", [0x3E00, 0x0000, 0x7F80, 0x7FC0, 0xFF80])
@pytest.mark.parametrize("mt", [0, 1])
@pytest.mark.parametrize("br", [64, 1])
def test_mx_special(br, mt, alpha):
    """Block exponents 0 / 254 / 255 and -128 mantissas in Q, K, V."""
    M, N, d, dv = (64 if br > 1 else 2), 128, 64, 64
    cfg = base_cfg(block_row_dim=br, block_column_dim=64, q_len=M, kv_len=N, head_dim=d,
                   v_head_dim=dv, mask_type=mt,
                   alpha_bf16=alpha)
    if mt == 1:
        cfg.update(q_mask_offset=N - M if br > 1 else N - 1, q_shards=1 if br > 1 else 0)

    def blocks(rows, width):
        e = RNG.choice(np.array([0, 1, 127, 128, 254, 255], np.uint8), (1, rows, width // 8))
        m = RNG.integers(-128, 128, (1, rows, width // 8, 8)).astype(np.int8)
        m[RNG.random(m.shape) < 0.05] = -128
        m[RNG.random(e.shape) < 0.1] = 0
        return np.concatenate([e[..., None], m.view(np.uint8)], -1).reshape(1, rows, -1)

    check("mx", blocks(M, d), blocks(N, d), blocks(N, dv), None, cfg)


# ---- special values in the state entries ------------------------------------------------------

# STATE rows with fully masked query rows (negative offsets), a sliding window and two q rows
# per mask row
STATE_SP = [STATE[3], STATE[4], STATE[6], STATE[7]]
SPF32 = np.array([0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000, 0x7F7FFFFF,
                  0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7FC00123,
                  0x7F800001, 0xFFFFFFFF, 0x3F800000, 0xBF800000], np.uint32).view(f32)


def special_state(M, d):
    """bf16 m / l [1][M] and U [1][M][d] with a special in each of the first rows: the fresh
    state, NaN / +-inf maxima, zero / inf / NaN / subnormal / negative sums, and U specials."""
    m = sprinkle(bf16(RNG.standard_normal((1, M))), SPB, 0.1)
    l = sprinkle(bf16(RNG.random((1, M)) * 8 + 1), SPB, 0.1)
    u = sprinkle(bf16(RNG.standard_normal((1, M, d)) * 4), SPB, 0.05)
    rows = [(0xFF80, 0x0000, 0x0000), (0x7FC0, 0x3F80, None), (0x7F80, 0x3F80, None),
            (0xFF80, 0x3F80, None), (0x3F80, 0x0000, None), (0x3F80, 0x7F80, None),
            (0x3F80, 0x7FC0, None), (0x3F80, 0x0001, None), (0x3F80, 0xBF80, None),
            (0x3F80, 0x3F80, 0x7F80), (0x3F80, 0x3F80, 0xFFC0), (0x0000, 0x8000, 0x8000)]
    for r, (mv, lv, uv) in enumerate(rows):
        m[0, r], l[0, r] = mv, lv
        if uv is not None:
            u[0, r] = uv
    return m, l, u


@pytest.mark.parametrize("rate", [0.0, 0.002, 0.03])
@pytest.mark.parametrize("p", STATE_SP, ids=lambda p: "_".join(map(str, p)))
def test_state_io_special(p, rate):
    """NaN / inf / subnormal / +-0 in Q, K, V and in the incoming m / l / U; masked rows."""
    calls = max(p[-1], 2)
    q, kv, _ = state_inputs(state_cfg(p, 0), calls, 0)
    q = sprinkle(q, SPB, rate)
    m, l, u = special_state(p[0], p[3])
    for call, (k, v) in enumerate(kv):
        k, v = sprinkle(k, SPB, rate), sprinkle(v, SPB, rate)
        cfg = state_cfg(p, call)
        final = call == calls - 1
        want = ga.flash_attention_state_io(q, k, v, m, l, u, cfg, final)
        got = state_io_entry(q, k, v, m, l, u, cfg, final)
        for g, w in zip(got, want):
            assert_equal(g, w)
        m, l, u = got


@pytest.mark.parametrize("rate", [0.0, 0.002, 0.03])
@pytest.mark.parametrize("p", STATE_SP, ids=lambda p: "_".join(map(str, p)))
def test_state_cache_special(p, rate):
    """As test_state_io_special for the cached state (U fp32 with fp32 specials); each call's
    model starts from the state the entry holds."""
    calls = max(p[-1], 2)
    q, kv, _ = state_inputs(state_cfg(p, 0), calls, 0)
    q = sprinkle(q, SPB, rate)
    M, d = p[0], p[3]
    m, l, u16 = special_state(M, d)
    u = sprinkle(ga.bf16_to_f32(u16).astype(f32), SPF32, 0.05)
    u[0, :12] = ga.bf16_to_f32(u16[0, :12])
    st_g = (m, l, u)
    for call, (k, v) in enumerate(kv):
        k, v = sprinkle(k, SPB, rate), sprinkle(v, SPB, rate)
        cfg = state_cfg(p, call)
        fl = 1 if call == calls - 1 else 0
        st_w = (ga.bf16_to_f32(st_g[0]).astype(f32), ga.bf16_to_f32(st_g[1]).astype(f32),
                st_g[2].copy())
        want = ga.flash_attention_state_cache(q, k, v, st_w, cfg, 0, fl)
        st_g, got = state_cache_entry(q, k, v, st_g, cfg, fl)
        assert_equal(st_g[0], f32_to_bf16(st_w[0]))
        assert_equal(st_g[1], f32_to_bf16(st_w[1]))
        assert_equal(st_g[2].view(np.uint32), st_w[2].view(np.uint32))
        if fl:
            assert_equal(got, want)
