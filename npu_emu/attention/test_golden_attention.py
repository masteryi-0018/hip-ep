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
from npu import NPU_I16, NPU_U16, OUT, Host, call, raw_rc

f32 = np.float32
RNG = np.random.default_rng(20261116)


class Cfg(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int64) for n in (
        "heads", "q_len", "kv_len", "head_dim", "v_head_dim", "mask_heads",
        "sliding_window_size", "q_offset")] + [(n, ctypes.c_int32) for n in (
            "br", "m_granularity", "bc", "k_tile", "kv_valid_last", "mask_type", "sliding",
            "q_shards", "q_shard", "q_wrap", "qdq", "variant_a", "shuffle", "alt_round",
            "v_transposed")] + [
        ("alpha_bf16", ctypes.c_uint32), ("qk_c", ctypes.c_float * 4),
        ("pv_c", ctypes.c_float * 4), ("p_c1", ctypes.c_float), ("p_c0", ctypes.c_int32),
        ("ofm_c1", ctypes.c_float), ("ofm_c0", ctypes.c_int32), ("mask_c0", ctypes.c_float),
        ("mask_c1", ctypes.c_float)]


def cstruct(cfg):
    c = Cfg()
    for name, _ in Cfg._fields_:
        val = cfg[name]
        if name in ("qk_c", "pv_c"):
            getattr(c, name)[:] = [float(f32(x)) for x in val]
        else:
            setattr(c, name, val)
    return c


def entry(kind, q, k, v, mask, cfg):
    """Run the entry point of `kind` on host arrays; returns the output array."""
    c = cstruct(cfg)
    H, M, dv = cfg["heads"], cfg["q_len"], cfg["v_head_dim"]
    m = None if mask is None else np.ascontiguousarray(mask, np.uint16)
    if kind == "bf16":
        return call("npu_flash_attention_bf16", Host(ctypes.byref(c)), q, k, v, m, OUT,
                    out_nbytes=H * M * dv * 2, out_dtype=np.uint16).reshape(H, M, dv)
    if kind == "q16":
        dt = np.int16 if cfg["signed"] else np.uint16
        return call("npu_flash_attention_q16", Host(ctypes.byref(c)),
                    NPU_I16 if cfg["signed"] else NPU_U16, q, k, v, m, OUT,
                    out_nbytes=H * M * dv * 2, out_dtype=dt).reshape(H, M, dv)
    if kind == "u8":
        return call("npu_flash_attention_u8", Host(ctypes.byref(c)), q, k, v, m, OUT,
                    out_nbytes=H * M * dv, out_dtype=np.uint8).reshape(H, M, dv)
    nb = H * M * dv // 8 * 9
    return call("npu_flash_attention_mx", Host(ctypes.byref(c)), q, k, v, OUT,
                out_nbytes=nb, out_dtype=np.uint8).reshape(H, M, dv // 8 * 9)


def check(kind, q, k, v, mask, cfg):
    want = np.asarray(ga.attention(kind, q, k, v, mask, cfg))
    got = entry(kind, q, k, v, mask, cfg)
    want = want.astype(got.dtype)
    bad = np.argwhere(got != want)
    kk = tuple(bad[:4].T)
    assert bad.size == 0, (f"{len(bad)}/{got.size} differ, first {bad[:4].tolist()}: "
                           f"got {got[kk].tolist()} want {want[kk].tolist()}")
    return got


def base_cfg(**kw):
    c = dict(heads=1, q_len=64, kv_len=128, head_dim=64, v_head_dim=64, mask_heads=1,
             sliding_window_size=0, q_offset=0, br=64, m_granularity=64, bc=128, k_tile=1,
             kv_valid_last=0, mask_type=0, sliding=0, q_shards=1, q_shard=0, q_wrap=1, qdq=0,
             variant_a=0, shuffle=0, alt_round=0, v_transposed=0, alpha_bf16=0x3E00,
             qk_c=[0.0] * 4, pv_c=[0.0] * 4, p_c1=0.0, p_c0=0, ofm_c1=0.0, ofm_c0=0,
             mask_c0=1.0, mask_c1=0.0, signed=0)
    c.update(kw)
    if "m_granularity" not in kw:
        c["m_granularity"] = c["br"]
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
    (8, 64, 16, 128, 128, 128), (2, 192, 4, 384, 32, 48), (1, 64, 4, 256, 64, 64),
    (1, 128, 3, 256, 96, 96)]


@pytest.mark.parametrize("alt_round", [0, 1])
@pytest.mark.parametrize("shape", BF16_SHAPES, ids=lambda s: "br%d_bc%d_M%d_N%d_d%d_dv%d" % s)
def test_bf16_random(shape, alt_round):
    br, bc, M, N, d, dv = shape
    cfg = base_cfg(br=br, bc=bc, q_len=M, kv_len=N, head_dim=d, v_head_dim=dv, alt_round=alt_round,
                   heads=2, alpha_bf16=int(bf16([1 / np.sqrt(d)])[0]))
    check("bf16", *bf16_inputs(cfg), None, cfg)


@pytest.mark.parametrize("kind", ["spread", "ties"])
@pytest.mark.parametrize("br", [64, 8, 1])
def test_bf16_constructed(kind, br):
    cfg = base_cfg(br=br, bc=64, q_len={64: 64, 8: 16, 1: 4}[br], kv_len=192, head_dim=64,
                   v_head_dim=32, alpha_bf16=0x3F80)
    check("bf16", *bf16_inputs(cfg, kind), None, cfg)


@pytest.mark.parametrize("br", [64, 32, 1])
@pytest.mark.parametrize("tail", [1, 41, 88, 127])
def test_bf16_kv_tail(br, tail):
    cfg = base_cfg(br=br, bc=128, q_len=max(br, 2), kv_len=256, kv_valid_last=tail)
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
    cfg = base_cfg(br=br, bc=bc, q_len=M, kv_len=N, mask_type=1, q_offset=off, q_shards=sh,
                   q_shard=shard, q_wrap=wrap, sliding=sl, sliding_window_size=w,
                   kv_valid_last=tail, alt_round=alt_round)
    check("bf16", *bf16_inputs(cfg), None, cfg)


@pytest.mark.parametrize("mkind", ["pad", "values", "neg_inf_some"])
def test_bf16_mask(mkind):
    cfg = base_cfg(br=64, bc=128, q_len=128, kv_len=256, mask_type=2, heads=2, mask_heads=2)
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
    if cfg["qdq"] == 1 and not u8:
        cfg["qk_c"] = [sq * np.sqrt(a), -zq * sq * np.sqrt(a), sk * np.sqrt(a),
                       -zk * sk * np.sqrt(a)]
    else:
        g = a * sq * sk
        cfg["qk_c"] = [g * d * zq * zk, -g * zk, g, -g * zq]
    if cfg["qdq"] == 1:
        cfg["pv_c"] = [-zv * sv, sv, 0.0, 0.0]
    else:
        cfg["pv_c"] = [pzp * zv * p_scale * sv * cfg["bc"], -p_scale * sv * zv, p_scale * sv,
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
    cfg = base_cfg(qdq=qdq, variant_a=int(qdq == 2), br=br, bc=bc, q_len=M, kv_len=N, head_dim=d,
                   v_head_dim=d, signed=signed, shuffle=shuffle, mask_type=mt, alt_round=alt_round,
                   m_granularity=16 if qdq == 2 else br)
    if mt == 1:
        cfg.update(q_offset=64, q_shards=1, q_wrap=1, sliding=1, sliding_window_size=40)
    q, k, v, zp = q16_inputs(cfg, signed)
    qdq_coeffs(cfg, zp, zo=(0 if signed else 30000))
    mask = None
    if mt == 2:
        cfg["mask_c0"], cfg["mask_c1"] = 1.0, -1000.0
        mask = np.where(RNG.random((1, N)) > 0.1, 1000, 0).astype(np.uint16)
    check("q16", q, k, v, mask, cfg)


def test_q16_saturation():
    """Wide codes and a small OFM scale: outputs hit both ends."""
    cfg = base_cfg(qdq=1, br=64, bc=64, q_len=64, kv_len=128)
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
    cfg = base_cfg(qdq=qdq, variant_a=int(qdq == 2), br=br, m_granularity=mg, bc=bc, k_tile=kt,
                   q_len=M, kv_len=N, head_dim=d, v_head_dim=d, mask_type=mt, alt_round=alt_round)
    zp = (128, 120, 135)
    q, k, v = (RNG.integers(z - 50, z + 50, s).astype(np.uint8)
               for z, s in zip(zp, ((1, M, d), (1, N, d), (1, N, d))))
    qdq_coeffs(cfg, zp, s=(0.01, 0.01, 0.01), so=0.002, zo=128, p_scale=1 / 255.0, u8=True)
    mask = None
    if mt == 2:
        cfg["mask_c0"], cfg["mask_c1"] = 1.0, -1000.0
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
    (64, 64, 128, 256, 64, 64, 1, 1)]


@pytest.mark.parametrize("zero", [0.0, 0.3])
@pytest.mark.parametrize("p", MX, ids=lambda p: "_".join(map(str, p)))
def test_mx(p, zero):
    br, bc, M, N, d, dv, mt, vt = p
    cfg = base_cfg(br=br, bc=bc, q_len=M, kv_len=N, head_dim=d, v_head_dim=dv, mask_type=mt,
                   v_transposed=vt, alpha_bf16=int(bf16([1 / np.sqrt(d)])[0]))
    if mt == 1:
        cfg.update(q_offset=N - M if br > 1 else N - 1, q_shards=1 if br > 1 else 0)
    q = mx_pack(RNG.standard_normal((1, M, d)) * np.exp2(RNG.integers(-6, 6, (1, M, d))), zero_blocks=zero)
    k = mx_pack(RNG.standard_normal((1, N, d)), zero_blocks=zero)
    vv = RNG.standard_normal((1, N, dv)) * np.exp2(RNG.integers(-20, 20, (1, N, dv)))
    v = mx_pack(np.swapaxes(vv, 1, 2) if vt else vv, zero_blocks=zero)
    check("mx", q, k, v, None, cfg)


def test_mx_extreme_exponents():
    """Block exponents near both ends (subnormal V values, huge Q blocks)."""
    cfg = base_cfg(br=64, bc=64, q_len=64, kv_len=128, head_dim=64, v_head_dim=64)
    q = RNG.integers(-128, 128, (1, 64, 8, 9)).astype(np.uint8)
    q[..., 0] = RNG.integers(120, 140, (1, 64, 8))
    k = RNG.integers(-128, 128, (1, 128, 8, 9)).astype(np.uint8)
    k[..., 0] = RNG.integers(120, 140, (1, 128, 8))
    v = RNG.integers(-128, 128, (1, 128, 8, 9)).astype(np.uint8)
    v[..., 0] = RNG.choice([0, 1, 5, 20, 133, 250], (1, 128, 8))
    check("mx", q.reshape(1, 64, 72), k.reshape(1, 128, 72), v.reshape(1, 128, 72), None, cfg)


# ---- argument checks ----------------------------------------------------------------

def test_bad_args():
    c = cstruct(base_cfg(bc=100))
    assert raw_rc("npu_flash_attention_bf16", Host(ctypes.byref(c)), None, None, None, None,
                  None) == -1
    q = np.zeros(4, np.uint16)
    c = cstruct(base_cfg(q_len=63))
    assert raw_rc("npu_flash_attention_bf16", Host(ctypes.byref(c)), q, q, q, None, q) == -1
    c = cstruct(base_cfg(qdq=0))
    assert raw_rc("npu_flash_attention_q16", Host(ctypes.byref(c)), NPU_U16, q, q, q, None,
                  q) == -1
    c = cstruct(base_cfg(head_dim=32))
    assert raw_rc("npu_flash_attention_mx", Host(ctypes.byref(c)), q, q, q, q) == -1


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
    cfg = base_cfg(br=br, bc=64, q_len=M, kv_len=192, head_dim=64, v_head_dim=32, mask_type=mt,
                   q_offset=100, q_shards=1 if br > 1 else 0, alpha_bf16=0x3F80)
    q, k, v = bf16_inputs(cfg)
    for rate in (0.002, 0.03):
        check("bf16", sprinkle(q, SPB, rate), sprinkle(k, SPB, rate), sprinkle(v, SPB, rate),
              None, cfg)


@pytest.mark.parametrize("alpha", [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x0001, 0x7F7F])
@pytest.mark.parametrize("br", [64, 1])
def test_bf16_special_scale(br, alpha):
    cfg = base_cfg(br=br, bc=64, q_len=64 if br > 1 else 2, kv_len=128, alpha_bf16=alpha)
    check("bf16", *bf16_inputs(cfg), None, cfg)


@pytest.mark.parametrize("br", [64, 8, 1])
@pytest.mark.parametrize("case", ["all_inf_mask", "first_block_masked", "neg_offset", "neg_small",
                                  "window", "pos_inf_mask", "nan_mask"])
def test_bf16_masked_rows(br, case):
    """Rows whose scores are all -inf (in the first block, or in every block)."""
    M = {64: 64, 8: 16, 1: 4}[br]
    cfg = base_cfg(br=br, bc=64, q_len=M, kv_len=192, head_dim=64, v_head_dim=32,
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
    elif case in ("neg_offset", "neg_small"):
        cfg.update(mask_type=1, q_offset=-100 if case == "neg_offset" else -10,
                   q_shards=1 if br > 1 else 0)
    else:
        cfg.update(mask_type=1, q_offset=130, q_shards=1 if br > 1 else 0, sliding=1,
                   sliding_window_size=1, kv_valid_last=5)
    check("bf16", *bf16_inputs(cfg), mask, cfg)


@pytest.mark.parametrize("kind", ["q16", "u8"])
@pytest.mark.parametrize("qdq", [1, 2])
@pytest.mark.parametrize("field,val", [(f, v) for f in ("qk_c", "pv_c", "ofm_c1", "p_c1")
                                       for v in SPF])
def test_qdq_special_coeffs(kind, qdq, field, val):
    """Degenerate coefficients (0, inf, NaN, huge, subnormal) on integer inputs."""
    u8 = kind == "u8"
    cfg = base_cfg(qdq=qdq, variant_a=int(qdq == 2), br=64, m_granularity=64, bc=64, q_len=64,
                   kv_len=128, head_dim=64, v_head_dim=64, alt_round=1)
    if u8:
        zp = (128, 120, 135)
        q, k, v = (RNG.integers(z - 50, z + 50, s).astype(np.uint8)
                   for z, s in zip(zp, ((1, 64, 64), (1, 128, 64), (1, 128, 64))))
        qdq_coeffs(cfg, zp, s=(0.01, 0.01, 0.01), so=0.002, zo=128, p_scale=1 / 255.0, u8=True)
    else:
        q, k, v, zp = q16_inputs(cfg, 0)
        qdq_coeffs(cfg, zp)
    if field in ("qk_c", "pv_c"):
        cfg[field] = [val] * 4 if qdq == 2 else [val, cfg[field][1], cfg[field][2], val]
    else:
        cfg[field] = val
    check(kind, q, k, v, None, cfg)


@pytest.mark.parametrize("kind", ["q16", "u8"])
@pytest.mark.parametrize("qdq", [1, 2])
def test_qdq_extreme_codes(kind, qdq):
    """Codes at both ends of the type and extreme zero points / offsets."""
    u8 = kind == "u8"
    cfg = base_cfg(qdq=qdq, variant_a=int(qdq == 2), br=64, m_granularity=64, bc=64, q_len=64,
                   kv_len=128, head_dim=64, v_head_dim=64)
    hi = 255 if u8 else 65535
    dt = np.uint8 if u8 else np.uint16
    ends = np.array([0, 1, hi - 1, hi], dt)
    q, k, v = (RNG.choice(ends, s) for s in ((1, 64, 64), (1, 128, 64), (1, 128, 64)))
    zp = (hi // 2, hi // 2, hi // 2)
    qdq_coeffs(cfg, zp, s=(0.01, 0.01, 0.01), so=0.002, zo=hi // 2, p_scale=1 / hi, u8=u8)
    for zo in (-2**31, 2**31 - 1, hi + 1, -1):
        cfg["ofm_c0"] = zo
        cfg["p_c0"] = zo
        check(kind, q, k, v, None, cfg)


@pytest.mark.parametrize("val", [0.0, np.inf, -np.inf, np.nan, 1e30, -1e30])
def test_q16_special_mask(val):
    cfg = base_cfg(qdq=1, br=64, bc=64, q_len=64, kv_len=128, mask_type=2)
    q, k, v, zp = q16_inputs(cfg, 0)
    qdq_coeffs(cfg, zp)
    cfg["mask_c0"], cfg["mask_c1"] = val, val
    mask = RNG.choice(np.array([0, 1, 65535], np.uint16), (1, 128))
    check("q16", q, k, v, mask, cfg)


@pytest.mark.parametrize("alpha", [0x3E00, 0x0000, 0x7F80, 0x7FC0, 0xFF80])
@pytest.mark.parametrize("mt", [0, 1])
@pytest.mark.parametrize("br", [64, 1])
def test_mx_special(br, mt, alpha):
    """Block exponents 0 / 254 / 255 and -128 mantissas in Q, K, V."""
    M, N, d, dv = (64 if br > 1 else 2), 128, 64, 64
    cfg = base_cfg(br=br, bc=64, q_len=M, kv_len=N, head_dim=d, v_head_dim=dv, mask_type=mt,
                   alpha_bf16=alpha)
    if mt == 1:
        cfg.update(q_offset=N - M if br > 1 else N - 1, q_shards=1 if br > 1 else 0)

    def blocks(rows, width):
        e = RNG.choice(np.array([0, 1, 127, 128, 254, 255], np.uint8), (1, rows, width // 8))
        m = RNG.integers(-128, 128, (1, rows, width // 8, 8)).astype(np.int8)
        m[RNG.random(m.shape) < 0.05] = -128
        m[RNG.random(e.shape) < 0.1] = 0
        return np.concatenate([e[..., None], m.view(np.uint8)], -1).reshape(1, rows, -1)

    check("mx", blocks(M, d), blocks(N, d), blocks(N, dv), None, cfg)
