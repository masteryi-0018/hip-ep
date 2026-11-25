# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for npu_mha_bf16_3strms."""

import numpy as np

from golden_bits import bits_f32, neg_f32, neg_inf
from golden_float_cvt import bf16_to_f32, f32_to_bf16, kFtzIn, kFtzOut, scalar_cast
from golden_int_round import Rnd
from golden_mac import fma_ftz, mul16, mul_q
from golden_minmax import max16
from golden_mmac import mmac_bf16
from golden_nlf import exp2_bf16, hw_inv_f32

_U32 = np.uint32
_BF16_NEG_INF = 0xFF80
L1_M = 64
FLOOR = Rnd.kFloor  # the bf16 results of the vector unit round toward -inf


# ---- bf16 MMAC -------------------------------------------------------------------

def bf16_matmul(a, b, acc=None):
    """a (M, K) @ b (K, N) bf16 bits, K % 32 == 0: MMAC chain over 32-deep chunks from acc."""
    M, K = a.shape
    N = b.shape[1]
    acc = np.zeros((M, N), _U32) if acc is None else acc
    for k0 in range(0, K, 32):
        acc = mmac_bf16(acc, a[:, None, k0:k0 + 32], b[k0:k0 + 32].T[None, :, :])
    return acc


# ---- the kernel ----------------------------------------------------------------------

def _mha_head(q, k, v, sv_n, scale_bits, causal, mask_M_start):
    """One (batch, head): q (M, D), k (N, D), v (N, D) bf16 bits -> (M, D)."""
    M, D = q.shape
    N = k.shape[0]
    assert M % L1_M == 0 and D % 64 == 0 and sv_n % 64 == 0 and N % sv_n == 0
    scale_f = np.array([scale_bits], _U32).view(np.float32)[0]
    scale_f32 = scalar_cast(mul_q(scale_f, np.float32(1.4423828125)), "bf16")
    inf_idx = (mask_M_start & 0xFFFF) % sv_n
    mix_col = mask_M_start // sv_n
    cols = np.arange(sv_n)
    ones = np.full((sv_n, 1), 0x3F80, np.uint16)
    out = np.zeros((M, D), np.uint16)
    for mb in range(M // L1_M):
        rows = slice(mb * L1_M, (mb + 1) * L1_M)
        qb = q[rows]
        o = np.zeros((L1_M, D), np.uint16)
        mx = np.full(L1_M, _BF16_NEG_INF, np.uint16)
        l = np.zeros(L1_M, np.uint16)
        curr_row = (mask_M_start + mb * L1_M) & 0xFFFF
        for ci in range(N // sv_n):
            kb = k[ci * sv_n:(ci + 1) * sv_n]
            vb = v[ci * sv_n:(ci + 1) * sv_n]
            s = f32_to_bf16(bits_f32(bf16_matmul(qb, kb.T)), FLOOR, kFtzIn)
            if causal:
                r_id = ((curr_row + np.arange(L1_M)) & 0xFFFF)[:, None]
                c_id = ((ci * sv_n + cols) & 0xFFFF)[None, :]
                masked = ~(r_id >= c_id)
            else:
                if ci < mix_col:
                    masked = np.zeros(sv_n, bool)
                elif ci == mix_col:
                    masked = cols >= inf_idx
                else:
                    masked = np.ones(sv_n, bool)
                masked = np.broadcast_to(masked[None, :], (L1_M, sv_n))
            mask = np.where(masked, neg_inf(), np.float32(0.0))
            ms = f32_to_bf16(fma_ftz(bf16_to_f32(s), scale_f32, mask, kFtzOut), FLOOR, kFtzIn)
            mx_new = mx
            for j in range(sv_n):
                mx_new = max16(mx_new, ms[:, j], "bf16")
            one, neg_mx = np.float32(1), neg_f32(bf16_to_f32(mx_new))
            p = f32_to_bf16(exp2_bf16(fma_ftz(bf16_to_f32(ms), one, neg_mx[:, None], kFtzOut)))
            rowsum = f32_to_bf16(bits_f32(bf16_matmul(p, ones)[:, 0]), FLOOR, kFtzIn)
            mx_diff = f32_to_bf16(fma_ftz(bf16_to_f32(mx), one, neg_mx, kFtzOut), FLOOR, kFtzIn)
            mx_exp = f32_to_bf16(exp2_bf16(bf16_to_f32(mx_diff)))
            mx = mx_new
            lf = fma_ftz(bf16_to_f32(mx_exp), bf16_to_f32(l), bf16_to_f32(rowsum), kFtzOut)
            l = f32_to_bf16(lf, FLOOR, kFtzIn)
            acc = (mul16(o, mx_exp[:, None], "bf16").astype(_U32) << 16).astype(_U32)
            o = f32_to_bf16(bits_f32(bf16_matmul(p, vb, acc)), FLOOR, kFtzIn)
        l_inv = f32_to_bf16(hw_inv_f32(bf16_to_f32(l)), FLOOR, kFtzIn)
        out[rows] = mul16(o, l_inv[:, None], "bf16")
    return out


def mha_bf16_3strms(q, k, v, sv_n, scale_bits, causal, mask_M_start):
    """q (B, H, M, D), k / v (B, Hkv, N, D) bf16 bits -> (B, M, H, D); GQA by head groups."""
    B, H, M, D = q.shape
    Hkv = k.shape[1]
    group = max(H // Hkv, 1)
    out = np.zeros((B, M, H, D), np.uint16)
    for b in range(B):
        for h in range(H):
            out[b, :, h] = _mha_head(q[b, h], k[b, h // group], v[b, h // group], sv_n,
                                     scale_bits, causal, mask_M_start)
    return out
