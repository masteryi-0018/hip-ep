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
STATE_DIRS = ["flash_attention_state_io_wrapper", "flash_attention_state_cache_wrapper"]


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
    ofm_c0 = int(round(float(t.get("ofm_c0", 0)))) & 0xFFFF  # the uint16 field
    d = int(t["head_dimension"])
    window = int(t.get("sliding_window_size", t.get("padding_mask_offset", 0)))
    sliding = int(t.get("enable_sliding_window_tmpl", t.get("enable_sliding_window", 0)) or 0)
    cfg = dict(heads=int(t.get("num_heads", 1)), q_len=seq * br, kv_len=blocks * bc, head_dim=d,
               v_head_dim=d, mask_heads=1, sliding_window_size=window,
               q_mask_offset=int(t.get("q_mask_offset", 0)), block_row_dim=br,
               m_granularity=int(t.get("m_granularity_tmpl", t.get("m_granularity", br))),
               block_column_dim=bc, tile_k=int(t.get("tile_k_rtp", t.get("tile_k", 1))),
               k_seq_tail_value=(fvt * 64 + tail) if (fvt or tail) else 0, mask_type=mask_type,
               enable_sliding_window=sliding,
               q_shards=q_shards, q_shard=shard.get(q_shards, 0), q_mask_wrap=q_wrap,
               enable_qdq=int(t.get("enable_qdq", 0) or 0) if kind != "mx" else 0,
               variant_a=int(str(t.get("aie_arch", "aie4")) == "aie4a"),
               enable_shuffle_transpose=int(t.get("enable_shuffle_transpose", 0) or 0),
               alt_round=int(str(t.get("compiler", "chess")) == "peano"),
               mat_v_transpose=int(t.get("mat_v_transpose", 0) or 0),
               alpha_bf16=int(t.get("alpha_bf16", 0x3F80)),
               matmul_qk_c=[f32w(t.get(f"matmul_qk_c{i}", 0)) for i in range(4)],
               matmul_pv_c=[f32w(t.get(f"qdq_place_holder_0_{i}", 0)) for i in range(4)],
               p_c1=f32w(t.get("p_c1_rtp", 0)), p_c0=int(t.get("p_c0_rtp", 0)),
               ofm_c1=f32w(t.get("ofm_c1", 0)), ofm_c0=ofm_c0,
               mask_dequant_c0=f32w(t.get("mask_dequant_c0", 0x3F800000)),
               mask_dequant_c1=f32w(t.get("mask_dequant_c1", 0)), signed=int(signed),
               q_group_size=int(t.get("q_group_size_rtp", t.get("q_group_size", 1))) or 1,
               kv_group_size=int(t.get("kv_group_size_rtp", t.get("kv_group_size", 1))) or 1)
    return kind, cfg, dtype


def load(case):
    """(kind, cfg, q, k, v, mask, want) of a sim case, logical layout."""
    t = tc(case)
    kind, cfg, dtype = config(case, t)
    H, M, N, d, dv = cfg["heads"], cfg["q_len"], cfg["kv_len"], cfg["head_dim"], cfg["v_head_dim"]
    assert H == 1
    bkg = int(t.get("block_k_granularity", 32))
    dk = -(-d // bkg) * bkg
    gemv = cfg["m_granularity"] == 1 and (dtype == "bfloat16" or cfg["enable_qdq"] == 1)
    pvg = 32 if gemv or cfg["mat_v_transpose"] == 3 else 64  # 3: PV on V^T, 32 head rows per block
    dvp = -(-dv // pvg) * pvg
    ofg = min(int(t.get("ofm_n_granularity_tmpl", t.get("ofm_n_granularity", 64))), pvg)
    dop = -(-dv // ofg) * ofg
    mask = None
    if kind == "mx":
        assert dk % 64 == 0 and dv % 8 == 0
        cfg["head_dim"] = dk  # the mx entry reads whole padded rows
        q = tensor8(case, "ifm0.txt", M, dk // 8 * 9)
        k = tensor8(case, "ifm1.txt", N, dk // 8 * 9)
        if cfg["mat_v_transpose"]:
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
    if cfg["mat_v_transpose"]:
        v = np.ascontiguousarray(rd(case, "ifm2.txt", dvp, N)[:dv].T)
    else:
        v = rd(case, "ifm2.txt", N, dvp)[:, :dv]
    want = rd(case, ofm(case), M, dop)[:, :dv]
    if cfg["mask_type"] == 2:
        # u16 mask stream of k_len samples; the u8 file is not pair-swapped
        mw = data(case, "ifm3.txt", np.uint16)
        mw = mw if kind == "u8" else pair_swap16(mw)
        tail = cfg["k_seq_tail_value"]
        k_len = (N - cfg["block_column_dim"] + tail) if tail else N
        mask = np.zeros((1, N), np.uint16)
        mask[0, :k_len] = mw[:k_len]
    if dtype == "int16":
        q, k, v, want = (x.view(np.int16) for x in (q, k, v, want))
    return kind, cfg, q[None], k[None], v[None], mask, want[None]


def run(case):
    kind, cfg, q, k, v, mask, want = load(case)
    model = {"bf16": ga.flash_attention_bf16, "q16": ga.flash_attention_q16,
             "u8": ga.flash_attention_u8}
    ref = ga.flash_attention_mx(q, k, v, cfg) if kind == "mx" else model[kind](q, k, v, mask, cfg)
    assert_bytes(np.asarray(ref).astype(want.dtype), want)
    from test_golden_attention import entry
    assert_bytes(entry(kind, q, k, v, mask, cfg), want)


@pytest.mark.parametrize("d,case", cases(DIRS))
def test_fixture(d, case):
    run(case)


# ---- state entries ------------------------------------------------------------------------

def state_case(case):
    """Parameters and per-call tensors of a state I/O or state cache case."""
    t = tc(case)
    cache = "state_cache" in case
    hd, seq, nb, tcl = (int(t[n]) for n in ("head_dimension", "sequence_cnt", "block_cnt",
                                             "tile_c_rtp"))
    cols = nb * tcl * 64
    name = "flash_attention_state_%s_test_config.h" % ("cache" if cache else "io")
    with open(os.path.join(case, "data", name)) as f:
        txt = f.read()
    calls = int(re.search(r"#define FA_\w+_TEST_CALLS (\d+)", txt).group(1))
    if cache:  # per call: bit 0 final output, bit 1 initialize
        flags = [int(x) for x in re.search(r"\{([^}]*)\}", txt).group(1).split(",")]
    else:
        final = np.load(os.path.join(case, "data", "final_flags.npy"))
        flags = [int(bool(x)) for x in final]
    gq = (int(t["q_mask_offset_low"]) | int(t["q_mask_offset_high_rtp"]) << 16
          | int(t["q_mask_offset_top_rtp"]) << 24)
    sliding = int(t.get("enable_sliding_window_tmpl", 0) or 0)
    q_shards = int(t.get("q_shards", 1))
    row, col = core_placement(case)
    shard = {0: 0, 1: 0, 2: row // 2, 4: row, 6: row // 2 + col * 2, 12: row + col * 4}
    cfg = dict(heads=1, q_len=seq * 64, kv_len=cols, head_dim=hd, v_head_dim=hd, mask_heads=1,
               sliding_window_size=int(t["sliding_window_size"]) if sliding else 0, q_mask_offset=0,
               block_row_dim=64, m_granularity=64, block_column_dim=tcl * 64,
               tile_k=int(t["tile_k_rtp"]), k_seq_tail_value=0, mask_type=1,
               enable_sliding_window=sliding, q_shards=q_shards, q_shard=shard.get(q_shards, 0),
               q_mask_wrap=int(t["q_mask_wrap_rtp"]) or 1, enable_qdq=0, variant_a=0,
               enable_shuffle_transpose=0,
               alt_round=int(str(t.get("compiler", "chess")) == "peano"), mat_v_transpose=0,
               alpha_bf16=int(t["scalar"]), matmul_qk_c=[0.0] * 4, matmul_pv_c=[0.0] * 4, p_c1=0.0,
               p_c0=0, ofm_c1=0.0, ofm_c0=0, mask_dequant_c0=1.0, mask_dequant_c1=0.0, signed=0,
               q_group_size=1, kv_group_size=1)
    per_call = []
    for c in range(calls):
        # Q repeated per KV block and column tile: [hd / 32][64 rows][32]
        q = pair_swap16(data(case, f"ifm0_{c}.txt", np.uint16))
        q = q.reshape(seq, nb, tcl, hd // 32, 64, 32)[:, 0, 0]
        q = q.transpose(0, 2, 1, 3).reshape(seq * 64, hd)
        # K (repeated per Q block): [tile][hd / 32][64 keys][32]
        k = pair_swap16(data(case, f"ifm1_{c}.txt", np.uint16))
        k = k.reshape(seq, nb, tcl, hd // 32, 64, 32)[0].transpose(0, 1, 3, 2, 4).reshape(cols, hd)
        # V: [hd / 64][tile][half][64 keys][32]
        v = pair_swap16(data(case, f"ifm2_{c}.txt", np.uint16))
        v = v.reshape(seq, nb, hd // 64, tcl, 2, 64, 32)[0]
        v = v.transpose(0, 2, 4, 1, 3, 5).reshape(cols, hd)
        # positions relative to the call's first key
        cc = dict(cfg, q_mask_offset=gq - (int(t["global_kv_idx"]) + c * cols))
        per_call.append((cc, flags[c], q[None], k[None], v[None]))
    state = None
    if not cache:
        m, l = (pair_swap16(data(case, n, np.uint16))[:seq * 64][None]
                for n in ("m_in.txt", "l_in.txt"))
        u = pair_swap16(data(case, "u_in.txt", np.uint16))[:seq * 64 * hd]
        u = u.reshape(seq, hd // 64, 64, 64).transpose(0, 2, 1, 3).reshape(1, seq * 64, hd)
        state = (m, l, u)
    out = pair_swap16(data(case, ofm(case), np.uint16))
    return cache, seq, hd, per_call, state, out.reshape(calls, seq, hd // 64, -1)


def head_tiled(x, seq, hd):
    """[1][seq * 64][hd] -> [seq][hd / 64][64][64]"""
    return x.reshape(seq, 64, hd // 64, 64).transpose(0, 2, 1, 3)


@pytest.mark.parametrize("d,case", cases(STATE_DIRS))
def test_fixture_state(d, case):
    from test_golden_attention import state_cache_entry, state_io_entry
    cache, seq, hd, per_call, state, want = state_case(case)
    M = seq * 64
    if cache:
        st = (np.zeros((1, M), np.uint16), np.zeros((1, M), np.uint16),
              np.zeros((1, M, hd), np.float32))
        st_ref = tuple(np.zeros(x.shape, np.float32) for x in st)
        for call, (cfg, fl, q, k, v) in enumerate(per_call):
            ref = ga.flash_attention_state_cache(q, k, v, st_ref, cfg, fl >> 1, fl & 1)
            st, got = state_cache_entry(q, k, v, st, cfg, fl)
            if fl & 1:  # other calls leave the output unspecified
                w = want[call].reshape(seq, hd // 64, 64, 64)
                assert_bytes(head_tiled(ref, seq, hd), w)
                assert_bytes(head_tiled(got, seq, hd), w)
        return
    m, l, u = state
    for call, (cfg, final, q, k, v) in enumerate(per_call):
        ref = ga.flash_attention_state_io(q, k, v, m, l, u, cfg, final)
        got = state_io_entry(q, k, v, m, l, u, cfg, final)
        w = want[call]
        for x in (ref, got):
            mo, lo, out = x
            assert_bytes(np.broadcast_to(mo.reshape(seq, 1, 64), (seq, hd // 64, 64)), w[..., :64])
            assert_bytes(np.broadcast_to(lo.reshape(seq, 1, 64), (seq, hd // 64, 64)),
                         w[..., 64:128])
            assert_bytes(head_tiled(out, seq, hd), w[..., 128:].reshape(seq, hd // 64, 64, 64))
        m, l, u = got  # the next call continues from this output
