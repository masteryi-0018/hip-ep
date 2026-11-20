# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Flash attention entry points and golden_attention.py against NPU reference outputs."""

import os
import re

import numpy as np
import pytest

import golden_attention as ga
from fixture import assert_bytes, cases, data, ofm
from fixture import testcase as tc

DIRS = ["mha_wrapper"]


def pair_swap16(x):
    """16-bit stream with the first element in the upper half of each word -> element order."""
    x = np.asarray(x, np.uint16)
    n = x.size // 2 * 2
    return np.concatenate([x[:n].reshape(-1, 2)[:, ::-1].reshape(-1), x[n:]])


def tensor16(case, name, rows, width):
    x = pair_swap16(data(case, name, np.uint16))
    return x[:rows * width].reshape(rows, width)


def tensor8(case, name, rows, width):
    return data(case, name, np.uint8)[:rows * width].reshape(rows, width)


def core_placement(case):
    """(row, col) of the FA_TEST_CORE_* defines (selects the Q shard)."""
    out = {"ROW": 0, "COL": 0}
    for p in (os.path.join(case, "data", "flash_attention_test_config.h"),
              os.path.join(case, "flash_attention_test_config.h")):
        if os.path.exists(p):
            for line in open(p):
                mt = re.match(r"#define FA_TEST_CORE_(ROW|COL) (\d+)", line.strip())
                if mt:
                    out[mt.group(1)] = int(mt.group(2))
            break
    return out["ROW"], out["COL"]


def f32w(v):
    """A coefficient field: a uint32 bit pattern (or a float)."""
    if isinstance(v, float):
        return float(np.float32(v))
    return float(np.array([int(v) & 0xFFFFFFFF], np.uint32).view(np.float32)[0])


def config(case, t):
    """npu_flash_attention_cfg fields from testcase.json and the kind of entry point."""
    dtype = str(t.get("dtype", "bfloat16"))
    kind = {"bfloat16": "bf16", "uint16": "q16", "int16": "q16", "uint8": "u8"}.get(dtype, "mx")
    br = int(t.get("block_row_dim_tmpl", t["tile_r"] * t["block_row_granularity"]))
    assert br == t["tile_r"] * t["block_row_granularity"]
    seq = int(t.get("sequence_count") or t["sequence_cnt"])
    blocks = int(t.get("block_count") or t["block_cnt"])
    bc = int(t.get("tile_c_rtp", t.get("tile_c"))) * 64
    tail_total = int(t.get("k_seq_tail_value", 0))
    fvt = int(t.get("full_valid_tiles", (tail_total // 64) if 0 < tail_total < bc else 0))
    tail = int(t.get("tail_value", tail_total % 64))
    mask_type = int(t["mask_type_rtp"])
    q_shards = int(t.get("q_shards", 1))
    row, col = core_placement(case)
    shard = {0: 0, 1: 0, 2: row // 2, 4: row, 6: row // 2 + col * 2, 12: row + col * 4}
    q_wrap = int(t.get("q_mask_wrap_rtp", int(t.get("q_mask_wrap", 0)) or seq)) or 1
    signed = dtype == "int16"
    ofm_c0 = int(round(float(t.get("ofm_c0", 0))))
    ofm_c0 &= 0xFF if kind == "u8" else 0xFFFF
    if signed and ofm_c0 >= 0x8000:
        ofm_c0 -= 0x10000
    d = int(t["head_dimension"])
    cfg = dict(
        heads=int(t.get("num_heads", 1)), q_len=seq * br, kv_len=blocks * bc, head_dim=d,
        v_head_dim=d, mask_heads=1,
        sliding_window_size=int(t.get("sliding_window_size", t.get("padding_mask_offset", 0))),
        q_offset=int(t.get("q_mask_offset", 0)),
        br=br, m_granularity=int(t.get("m_granularity_tmpl", t.get("m_granularity", br))), bc=bc,
        k_tile=int(t.get("tile_k_rtp", t.get("tile_k", 1))),
        kv_valid_last=(fvt * 64 + tail) if (fvt or tail) else 0,
        mask_type=mask_type,
        sliding=int(t.get("enable_sliding_window_tmpl", t.get("enable_sliding_window", 0)) or 0),
        q_shards=q_shards, q_shard=shard.get(q_shards, 0), q_wrap=q_wrap,
        qdq=int(t.get("enable_qdq", 0) or 0) if kind != "mx" else 0,
        variant_a=int(str(t.get("aie_arch", "aie4")) == "aie4a"),
        shuffle=int(t.get("enable_shuffle_transpose", 0) or 0),
        alt_round=int(str(t.get("compiler", "chess")) == "peano"),
        v_transposed=int(t.get("mat_v_transpose", 0) or 0),
        alpha_bf16=int(t.get("alpha_bf16", 0x3F80)),
        qk_c=[f32w(t.get(f"matmul_qk_c{i}", 0)) for i in range(4)],
        pv_c=[f32w(t.get(f"qdq_place_holder_0_{i}", 0)) for i in range(4)],
        p_c1=f32w(t.get("p_c1_rtp", 0)), p_c0=int(t.get("p_c0_rtp", 0)),
        ofm_c1=f32w(t.get("ofm_c1", 0)), ofm_c0=ofm_c0,
        mask_c0=f32w(t.get("mask_dequant_c0", 0x3F800000)),
        mask_c1=f32w(t.get("mask_dequant_c1", 0)),
        signed=int(signed))
    return kind, cfg, dtype


def load(case):
    """(kind, cfg, q, k, v, mask, want) of a sim case, logical layout."""
    t = tc(case)
    kind, cfg, dtype = config(case, t)
    H, M, N, d, dv = cfg["heads"], cfg["q_len"], cfg["kv_len"], cfg["head_dim"], cfg["v_head_dim"]
    assert H == 1
    bkg = int(t.get("block_k_granularity", 32))
    dk = -(-d // bkg) * bkg
    gemv = cfg["m_granularity"] == 1 and (dtype == "bfloat16" or cfg["qdq"] == 1)
    pvg = 32 if gemv or cfg["v_transposed"] == 3 else 64  # 3: PV on V^T, 32 head rows per block
    dvp = -(-dv // pvg) * pvg
    ofg = min(int(t.get("ofm_n_granularity_tmpl", t.get("ofm_n_granularity", 64))), pvg)
    dop = -(-dv // ofg) * ofg
    mask = None
    if kind == "mx":
        assert dk % 64 == 0 and dv % 8 == 0
        cfg["head_dim"] = dk  # the mx entry reads whole padded rows
        q = tensor8(case, "ifm0.txt", M, dk // 8 * 9)
        k = tensor8(case, "ifm1.txt", N, dk // 8 * 9)
        if cfg["v_transposed"]:
            v = tensor8(case, "ifm2.txt", dv, N // 8 * 9)
        else:
            v = tensor8(case, "ifm2.txt", N, dv // 8 * 9)
        want = tensor8(case, ofm(case), M, dop // 8 * 9)[:, :dv // 8 * 9]
        return kind, cfg, q[None], k[None], v[None], None, want[None]
    if kind == "u8":
        rd = tensor8
    else:
        rd = tensor16
    q = rd(case, "ifm0.txt", M, dk)[:, :d]
    k = rd(case, "ifm1.txt", N, dk)[:, :d]
    if cfg["v_transposed"]:
        v = np.ascontiguousarray(rd(case, "ifm2.txt", dvp, N)[:dv].T)
    else:
        v = rd(case, "ifm2.txt", N, dvp)[:, :dv]
    want = rd(case, ofm(case), M, dop)[:, :dv]
    if cfg["mask_type"] == 2:
        # u16 mask stream of k_len samples; the u8 file is not pair-swapped
        mw = data(case, "ifm3.txt", np.uint16)
        mw = mw if kind == "u8" else pair_swap16(mw)
        k_len = (N - cfg["bc"] + cfg["kv_valid_last"]) if cfg["kv_valid_last"] else N
        mask = np.zeros((1, N), np.uint16)
        mask[0, :k_len] = mw[:k_len]
    if dtype == "int16":
        q, k, v, want = (x.view(np.int16) for x in (q, k, v, want))
    return kind, cfg, q[None], k[None], v[None], mask, want[None]


def run(case):
    kind, cfg, q, k, v, mask, want = load(case)
    assert_bytes(np.asarray(ga.attention(kind, q, k, v, mask, cfg)).astype(want.dtype), want)
    from test_golden_attention import entry
    assert_bytes(entry(kind, q, k, v, mask, cfg), want)


@pytest.mark.parametrize("d,case", cases(DIRS))
def test_fixture(d, case):
    run(case)
