# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for flash attention; `cfg` is a dict of the npu_flash_attention_cfg fields."""

import numpy as np

from golden_block_fp import bfp16_fields, decode_bfp16, encode_bfp16
from golden_float_cvt import bf16_to_f32, f32_to_bf16, kFtzIn, kFtzOut, round_float
from golden_int_round import Rnd, float_to_int
from golden_mac import fma_ftz, mac_exact, mul16_f32, vaddr_f32_reduce
from golden_minmax import vmax
from golden_mmac import mmac_bf16, mmac_bfp16
from golden_nlf import hw_exp2_bf16, hw_inv_f32

f32 = np.float32
I64 = np.int64
NEG_INF = f32(-np.inf)
LOG2E = f32(1.4453125)          # bf16 0x3fb9
MAX_BIAS = f32(-895.00099)      # folded into max * log2e
EXP2_BIAS = f32(895.001)
FTZ = kFtzIn | kFtzOut  # the fp32 MACs flush subnormal operands and results


# ---- MMACs ----------------------------------------------------------------------

def bf16_gemm(a, b):
    """A (r x K) @ B (K x c) on bf16 values: one MMAC chain per output, K in slices of 32."""
    a, b = f32_to_bf16(a, flags=kFtzIn), f32_to_bf16(b, flags=kFtzIn)   # values are bf16 already
    K = a.shape[1]
    Kp = -(-K // 32) * 32
    a = np.pad(a, ((0, 0), (0, Kp - K)))
    b = np.pad(b, ((0, Kp - K), (0, 0)))
    acc = np.zeros((a.shape[0], b.shape[1]), np.uint32)
    for k0 in range(0, Kp, 32):
        acc = mmac_bf16(acc, a[:, None, k0:k0 + 32], b[k0:k0 + 32].T[None, :, :])
    return acc.view(f32)


def mx_gemm(am, ae, bm, be):
    """mx MMAC chain over K in steps of 64; B is given as c x K / c x K/8. Returns fp32."""
    acc = np.zeros((am.shape[0], bm.shape[0]), np.uint32)
    for k0 in range(0, am.shape[1], 64):
        e = slice(k0 // 8, k0 // 8 + 8)
        acc = mmac_bfp16(acc, am[:, None, k0:k0 + 64], ae[:, None, e], bm[None, :, k0:k0 + 64],
                         be[None, :, e])
    return acc.view(f32)


def mx_encode(x, rnd):
    """fp32 rows -> mxint8_ebs8 fields along the last axis (all-zero block -> exponent 0)."""
    x = np.asarray(x, f32)
    rows = x.shape[0]
    packed = encode_bfp16(x.reshape(-1), rnd)
    return bfp16_fields(packed, rows), packed.reshape(rows, -1)


# ---- configuration ----------------------------------------------------------------

def _rnd(cfg):
    """Rounding of the alt_round-dependent conversions: ties away, or half to even."""
    return Rnd.kHalfEven if cfg["alt_round"] else Rnd.kHalfAway


def _mode_s(kind, cfg):
    """Rounding of the masked scores."""
    if kind == "u8":
        return Rnd.kHalfEven
    return _rnd(cfg) if (not cfg["variant_a"] and cfg["qdq"] <= 1) else Rnd.kHalfEven


def block_range(cfg, sc):
    """KV blocks [begin, end) visited by Q block sc and its rows' Q positions (causal only)."""
    nb = cfg["kv_len"] // cfg["bc"]
    if cfg["mask_type"] != 1:
        return 0, nb, None
    br, bc = cfg["br"], cfg["bc"]
    sh = cfg["q_shards"]
    qb = cfg["q_offset"] + (sh > 0) * (cfg["q_shard"] + sh * (sc % max(cfg["q_wrap"], 1))) * br
    end = min(qb // bc + 1, nb)
    begin = 0
    w = cfg["sliding_window_size"]
    if cfg["sliding"] and w > 0 and qb - w + 1 >= bc:
        begin = (qb - w + 1) // bc
    return begin, end, qb + np.arange(br) * (1 if sh > 0 else 0)


def valid_keys(cfg, blk):
    nb = cfg["kv_len"] // cfg["bc"]
    v = cfg["kv_valid_last"]
    return v if (blk == nb - 1 and 0 < v < cfg["bc"]) else cfg["bc"]


# ---- the pieces per data type --------------------------------------------------------

def act_sum_int8(a, m_gran, chunk=64):
    """uint8 QDQ row sums; several chunks and m_gran >= 32: first skipped, last counted twice."""
    a = np.asarray(a, I64)
    n = -(-a.shape[1] // chunk)
    if n <= 1 or m_gran < 32:
        return a.sum(axis=1)
    return a[:, chunk:].sum(axis=1) + a[:, (n - 1) * chunk:].sum(axis=1)


def gemm16(a, w, c0, c1, c2, c3):
    """QDQ mode 2 int16x16 GEMM: byte-split integer sums combined by fp32 MACs."""
    a, w = np.asarray(a, I64) & 0xFFFF, np.asarray(w, I64) & 0xFFFF
    alo, ahi, wlo, whi = a & 0xFF, a >> 8, w & 0xFF, w >> 8
    a_sum = mac_exact(ahi.sum(1), f32(256), alo.sum(1).astype(f32), FTZ)
    with np.errstate(all="ignore"):   # plain fp32 products of the coefficients
        c3_256, c2_256, c2_65536 = f32(c3 * f32(256)), f32(c2 * f32(256)), f32(c2 * f32(65536))
    w_sum = mac_exact(whi.sum(0), c3_256, mac_exact(wlo.sum(0), c3, c0, FTZ), FTZ)
    acc = mac_exact(alo @ wlo, c2, w_sum[None, :], FTZ)
    acc = mac_exact(alo @ whi + ahi @ wlo, c2_256, acc, FTZ)
    acc = mac_exact(ahi @ whi, c2_65536, acc, FTZ)
    return fma_ftz(c1, a_sum[:, None], acc, FTZ)


def scores(kind, qb, kb, cfg):
    """MatMul-QK of one Q block and KV block, as bf16 values (fp32 array)."""
    c = [f32(x) for x in cfg["qk_c"]]
    if kind == "bf16":
        s = round_float(bf16_gemm(qb, kb.T), "bf16", Rnd.kHalfAway, kFtzIn)
        # exact bf16 product, one rounding
        return mul16_f32(s, cfg["scale"], "bf16")
    if kind == "q16" and cfg["qdq"] == 1:
        qd = round_float(mac_exact(qb, c[0], c[1], FTZ), "bf16", Rnd.kHalfAway, kFtzIn)
        kd = round_float(mac_exact(kb, c[2], c[3], FTZ), "bf16", Rnd.kHalfAway, kFtzIn)
        return round_float(bf16_gemm(qd, kd.T), "bf16", Rnd.kHalfAway, kFtzIn)
    if kind == "q16":
        return round_float(gemm16(qb, kb.T, *c), "bf16", Rnd.kHalfEven, kFtzIn)
    if kind == "u8":
        q, k = np.asarray(qb, I64), np.asarray(kb, I64)
        a_sum = act_sum_int8(q, cfg["m_granularity"]).astype(f32)
        w_sum = k.sum(axis=1).astype(f32)
        if cfg["br"] == 1:
            with np.errstate(all="ignore"):
                bias = (c[0] + (c[1] * a_sum).astype(f32)).astype(f32)
            acc = fma_ftz(w_sum[None, :], c[3], bias[:, None], FTZ)
            acc = mac_exact(q @ k.T, c[2], acc, FTZ)
        else:
            acc = mac_exact(q @ k.T, c[2], fma_ftz(w_sum, c[3], c[0], FTZ)[None, :], FTZ)
            acc = fma_ftz(c[1], a_sum[:, None], acc, FTZ)
        return round_float(acc, "bf16", Rnd.kHalfEven, kFtzIn)
    # mx: Q / K fields
    (qm, qe), (km, ke) = qb, kb
    return round_float(mx_gemm(qm, qe, km, ke), "bf16", Rnd.kHalfAway, kFtzIn)


def log2e_coeff(kind, cfg):
    if kind == "mx":
        # bf16(scale * log2e), the product exact in fp32
        return f32(round_float(f32(cfg["scale"]) * LOG2E, "bf16", Rnd.kHalfEven, kFtzIn).reshape(-1)[0])
    return LOG2E


def row_max(kind, s, m_pre, cfg, guard):
    """New running max and exp(m_pre - m_new) (bf16 values)."""
    cur = s[:, 0]
    for j in range(1, s.shape[1]):
        cur = vmax(cur, s[:, j])
    c = log2e_coeff(kind, cfg)
    with np.errstate(invalid="ignore"):
        if cfg["br"] == 1:
            new = vmax(cur, m_pre)
            delta = (m_pre.astype(np.float64) - new.astype(np.float64)).astype(f32)
            if guard:
                delta = np.where(new == -np.inf, f32(0), delta)
            t = (delta.astype(np.float64) * float(c)).astype(f32)
        else:
            new = vmax(cur, m_pre)
            delta = fma_ftz(m_pre, f32(1), -new, FTZ)
            if guard:
                delta = np.where(new == -np.inf, f32(0), delta)
            t = fma_ftz(delta, c, f32(0), FTZ)
    return new, hw_exp2_bf16(fma_ftz(t, f32(1), EXP2_BIAS, FTZ))


def compute_p(kind, s, m, cfg, guard):
    """P (bf16 values) and the bf16 row sum."""
    c = log2e_coeff(kind, cfg)
    qdq16 = kind == "q16" and cfg["qdq"] == 2
    if guard and kind != "mx" and not qdq16:
        m = np.where(m < -3.0e38, f32(0), m).astype(f32)
    p = hw_exp2_bf16(fma_ftz(s, c, -fma_ftz(c, m, MAX_BIAS, FTZ)[:, None], FTZ))
    pt = p.reshape(p.shape[0], -1, 64)
    acc = None
    if qdq16:
        acc = np.zeros((p.shape[0], 32), f32)
        for t in range(pt.shape[1]):
            acc = fma_ftz(fma_ftz(acc, f32(1), pt[:, t, :32], FTZ), f32(1), pt[:, t, 32:], FTZ)
        acc = round_float(acc, "bf16", Rnd.kHalfEven, kFtzIn)
        return p, round_float(vaddr_f32_reduce(acc), "bf16", Rnd.kHalfEven, kFtzIn)
    if cfg["br"] == 1 and kind != "mx":
        # (p0 + p1) + previous
        for t in range(pt.shape[1]):
            e = fma_ftz(pt[:, t, :32], f32(1), pt[:, t, 32:], FTZ)
            acc = e if acc is None else fma_ftz(e, f32(1), acc, FTZ)
    else:
        # (p0 + previous) + p1
        for t in range(pt.shape[1]):
            acc = pt[:, t, :32] if acc is None else fma_ftz(pt[:, t, :32], f32(1), acc, FTZ)
            acc = fma_ftz(acc, f32(1), pt[:, t, 32:], FTZ)
    return p, round_float(vaddr_f32_reduce(acc), "bf16", Rnd.kHalfEven, kFtzIn)


def quant_p(kind, p, cfg):
    """P as the PV operand: bf16 values, or integers (QDQ mode 2), or mx fields."""
    if kind == "q16" and cfg["qdq"] == 2:
        return float_to_int(fma_ftz(p, cfg["p_c1"], f32(cfg["p_c0"] & 0xFFFF), FTZ), "u16")
    if kind == "u8" and cfg["qdq"] == 2:
        return float_to_int(fma_ftz(p, cfg["p_c1"], f32(cfg["p_c0"] & 0xFFFF), FTZ), "u8")
    if kind == "mx":
        return mx_encode(p, Rnd.kHalfEven)[0]
    return p


def v_operand(kind, vb, cfg):
    """V of one KV block as the PV engine sees it (bf16 values / ints / mx V^T fields)."""
    c = [f32(x) for x in cfg["pv_c"]]
    if kind == "bf16":
        return vb
    if kind == "mx":
        vt_fields, vt_vals = vb
        if cfg["v_transposed"]:
            return vt_fields
        # V transposed exactly and re-encoded along the KV axis
        return mx_encode(vt_vals, _rnd(cfg))[0]
    if cfg["qdq"] == 2:
        return np.asarray(vb, I64)
    rnd = _rnd(cfg) if (kind == "u8" or cfg["shuffle"]) else Rnd.kHalfAway
    return round_float(mac_exact(vb, c[1], c[0], FTZ), "bf16", rnd, kFtzIn)


def pv(kind, p, vd, cfg):
    """MatMul-PV of one KV block (fp32), before the O update."""
    c = [f32(x) for x in cfg["pv_c"]]
    if kind == "mx":
        (pm, pe), (vm, ve) = p, vd
        return mx_gemm(pm, pe, vm, ve)
    if kind == "q16" and cfg["qdq"] == 2:
        return gemm16(p, vd, *c)
    if kind == "u8" and cfg["qdq"] == 2:
        chunk = cfg["k_tile"] * 64
        out = None
        for i, k0 in enumerate(range(0, p.shape[1], chunk)):
            pc, vc = p[:, k0:k0 + chunk], vd[k0:k0 + chunk]
            c0t = mac_exact(vc.sum(axis=0), c[3], c[0] if i == 0 else f32(0), FTZ)
            acc = mac_exact(pc @ vc, c[2], c0t[None, :], FTZ)
            acc = fma_ftz(c[1], act_sum_int8(pc, 64).astype(f32)[:, None], acc, FTZ)
            out = acc if out is None else fma_ftz(acc, f32(1), out, FTZ)
        return out
    if cfg["br"] == 1:
        # GEMV: four MAC chains over the keys (i % 4), summed pairwise
        accs = []
        for ch in range(4):
            acc = np.zeros(vd.shape[1], f32)
            for i in range(ch, p.shape[1], 4):
                acc = fma_ftz(vd[i], p[0, i], acc, FTZ)
            accs.append(acc)
        lo, hi = (fma_ftz(accs[i], f32(1), accs[i + 1], FTZ) for i in (0, 2))
        return fma_ftz(lo, f32(1), hi, FTZ)[None, :]
    return bf16_gemm(p, vd)


def rescale(kind, o, l, cfg):
    inv_l = hw_inv_f32(l)
    if kind == "bf16":
        return f32_to_bf16(fma_ftz(o, inv_l[:, None], f32(0), FTZ), _rnd(cfg), kFtzIn)
    if kind == "mx":
        return mx_encode(fma_ftz(o, inv_l[:, None], f32(0), FTZ), _rnd(cfg))[1]
    c1 = f32(cfg["ofm_c1"])
    c0 = f32(cfg["ofm_c0"])
    if cfg["br"] == 1:
        scale = (np.float64(c1) * inv_l.astype(np.float64)).astype(f32)
    else:
        scale = fma_ftz(c1, inv_l, f32(0), FTZ)
    t = "u8" if kind == "u8" else ("i16" if cfg["signed"] else "u16")
    return float_to_int(fma_ftz(o, scale[:, None], c0, FTZ), t, 0, _rnd(cfg))


# ---- driver ------------------------------------------------------------------------

def attention(kind, q, k, v, mask, cfg):
    """q / k / v [H][rows][d]: bf16 bits, integer codes or packed mxint8_ebs8 bytes (v_transposed:
    [H][dv][N/8 * 9]); mask [Hm][N] or None. Returns [H][M][dv], or mx bytes [H][M][dv/8 * 9]."""
    H, M, N, br, bc = cfg["heads"], cfg["q_len"], cfg["kv_len"], cfg["br"], cfg["bc"]
    dv = cfg["v_head_dim"]
    guard = bool(cfg["sliding"])
    mode_s = _mode_s(kind, cfg)
    cfg = dict(cfg)
    cfg["scale"] = bf16_to_f32(np.uint16(cfg["alpha_bf16"]))
    outs = []
    for h in range(H):
        if kind == "bf16":
            qh, kh, vh = bf16_to_f32(q[h]), bf16_to_f32(k[h]), bf16_to_f32(v[h])
        elif kind == "mx":
            qh, kh = bfp16_fields(q[h], M), bfp16_fields(k[h], N)
            if cfg["v_transposed"]:
                vh = bfp16_fields(v[h], dv)
            else:
                vh = decode_bfp16(v[h]).reshape(N, dv)
        else:
            qh, kh, vh = (np.asarray(x[h], I64) for x in (q, k, v))
        mh = None
        if mask is not None and cfg["mask_type"] == 2:
            mr = np.asarray(mask[h % cfg["mask_heads"]])
            if kind == "bf16":
                mh = bf16_to_f32(mr)
            else:
                mv = mac_exact(mr.astype(I64), cfg["mask_c0"], cfg["mask_c1"], FTZ)
                mh = round_float(mv, "bf16", mode_s, kFtzIn)
        rows = []
        for sc in range(M // br):
            r0 = sc * br
            if kind == "mx":
                qb = (qh[0][r0:r0 + br], qh[1][r0:r0 + br])
            else:
                qb = qh[r0:r0 + br, :cfg["head_dim"]]
            begin, end, qpos = block_range(cfg, sc)
            m = np.full(br, NEG_INF, f32)
            l = np.zeros(br, f32)
            o = np.zeros((br, dv), f32)
            for blk in range(begin, end):
                c0, c1 = blk * bc, (blk + 1) * bc
                if kind == "mx":
                    kb = (kh[0][c0:c1], kh[1][c0:c1])
                    if cfg["v_transposed"]:
                        vb = ((vh[0][:, c0:c1], vh[1][:, c0 // 8:c1 // 8]), None)
                    else:
                        vb = (None, np.ascontiguousarray(vh[c0:c1].T))
                else:
                    kb, vb = kh[c0:c1, :cfg["head_dim"]], vh[c0:c1, :dv]
                s = scores(kind, qb, kb, cfg)
                nv = valid_keys(cfg, blk)
                col = np.arange(bc)[None, :]
                if cfg["mask_type"] == 1:
                    kpos = c0 + col
                    ok = (kpos <= qpos[:, None]) & (col < nv)
                    if cfg["sliding"] and cfg["sliding_window_size"] > 0:
                        ok &= kpos >= qpos[:, None] - cfg["sliding_window_size"] + 1
                    s = fma_ftz(s, f32(1), np.where(ok, f32(0), NEG_INF), FTZ)
                    s = round_float(s, "bf16", mode_s, kFtzIn)
                elif nv < bc:
                    with np.errstate(invalid="ignore"):
                        s = np.where(col < nv, s + f32(0), NEG_INF).astype(f32)
                if mh is not None:
                    s = round_float(fma_ftz(s, f32(1), mh[None, c0:c1], FTZ), "bf16", mode_s, kFtzIn)
                m, emd = row_max(kind, s, m, cfg, guard)
                p, rs = compute_p(kind, s, m, cfg, guard)
                l = round_float(fma_ftz(emd, l, rs, FTZ), "bf16", Rnd.kHalfEven if br < 8 else _rnd(cfg),
                                kFtzIn)
                pvb = pv(kind, quant_p(kind, p, cfg), v_operand(kind, vb, cfg), cfg)[:, :dv]
                o = fma_ftz(o, emd[:, None], pvb, FTZ)
            rows.append(rescale(kind, o, l, cfg))
        outs.append(np.concatenate(rows, axis=0))
    return np.stack(outs)
