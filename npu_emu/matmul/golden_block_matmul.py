# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""numpy reference for the bfp16 / mx6 / fp8 matmul and convolution entry points."""

import numpy as np

import golden_act as ga
from golden_block_fp import bfp16_fields, decode_bfp16_identity, encode_bfp16, encode_mx6, mx6_fields
from golden_float_cvt import (bf16_to_f32, f16_to_f32, f32_bits, f32_to_f16, f32_to_fp8,
                              fp8_to_f32, kFtzIn, round_float, scalar_cast, store_float)
from golden_int_round import Rnd, float_to_int
from golden_mac import add, fma, mul, mul16, mul16_f32
from golden_minmax import max16, min16, vclamp, vmax, vmin
from golden_mmac import mmac_add_bfp16, mmac_bfp16, mmac_fp8, mmac_mx6

F32, I64 = np.float32, np.int64


def f32v(bits):
    return np.asarray(bits, np.uint32).view(np.float32)


def activate_block(v, narrow, act, alpha):
    if not narrow:
        return v
    v = round_float(v, "bf16", Rnd.kHalfAway, kFtzIn)
    if act == 1:
        return vmax(v, F32(0))
    if act == 2:
        return vclamp(v, F32(0), F32(6))
    if act == 3:
        return vmax(v, mul16_f32(v, scalar_cast(F32(alpha), "bf16"), "bf16"))
    return core_act(v, act)


def core_act(v, act):
    """SiLU (5) / GELU (6) / Tanh (7) of bf16 values, stored as bf16 half away."""
    if act == 5:
        return round_float(ga.hw_silu_f32(v), "bf16", Rnd.kHalfAway, kFtzIn)
    if act == 6:
        return round_float(ga.hw_gelu_f32(v), "bf16", Rnd.kHalfAway, kFtzIn)
    if act == 7:
        return ga.sigtanh_tanh_bf16(v)
    return v


def store(v, ofm, rnd, q=None):
    """fp32 values (rows x cols) -> the output encoding: "bfp16" / "mx6" blocks, "u8" the
    uint8 cast sat(rne(fp32(v * q[0] + q[1]))), else bf16 / f16 / fp8 elements."""
    v = np.ascontiguousarray(v, F32).reshape(-1)
    if ofm == "bfp16":
        return encode_bfp16(v, rnd)
    if ofm == "mx6":
        return encode_mx6(v, rnd)
    if ofm == "u8":
        return float_to_int(fma(v, F32(q[0]), F32(q[1])), "u8").astype(np.uint8)
    if ofm == "fp8":
        return f32_to_fp8(v, rnd)
    return store_float(v, ofm, rnd, kFtzIn)


def narrow_f16_fp8(x, r):
    """The f16 -> E4M3 narrowing of f16 activation codes x: the 16-bit f16 product with the
    f16 reciprocal scale r (a float exact in f16), clamped to +-448 in f16, then E4M3 half
    away (subnormals kept)."""
    rc = f32_to_f16(F32(r))
    p = min16(max16(np.uint16(0xDF00), mul16(x, rc, "f16"), "f16"), np.uint16(0x5F00), "f16")
    return f32_to_fp8(f16_to_f32(p), Rnd.kHalfAway)


def gemm_bfp16_acc(am, ae, bm, be, acc, k0=0, k1=None):
    """A (M x K) @ B (fields of B^T: N x K); one MMAC per 64 K in [k0, k1)."""
    k1 = am.shape[1] if k1 is None else k1
    for c in range(k0, k1, 64):
        acc = mmac_bfp16(acc, am[:, None, c:c + 64], ae[:, None, c // 8:c // 8 + 8], bm[None, :, c:c + 64],
                         be[None, :, c // 8:c // 8 + 8])
    return acc


def gemm_bfp16(a, bt, bias, m, k, n, narrow, act, alpha, transpose, ofm, q=None):
    am, ae = bfp16_fields(a, m)
    bm, be = bfp16_fields(bt, n)
    acc = np.broadcast_to(f32_bits(bias.reshape(1, n)) if bias is not None else np.uint32(0), (m, n))
    v = activate_block(f32v(gemm_bfp16_acc(am, ae, bm, be, np.array(acc))), narrow, act, alpha)
    return store(v.T if transpose else v, ofm, Rnd.kHalfAway, q)


def gemm_gelu_mul_bfp16(a, bt_gate, bt_up, bias_gate, bias_up, m, k, n, act, ofm):
    am, ae = bfp16_fields(a, m)
    outs = []
    for bt, bias in ((bt_gate, bias_gate), (bt_up, bias_up)):
        bm, be = bfp16_fields(bt, n)
        acc = np.broadcast_to(f32_bits(bias.reshape(1, n)) if bias is not None else np.uint32(0), (m, n))
        outs.append(round_float(f32v(gemm_bfp16_acc(am, ae, bm, be, np.array(acc))), "bf16",
                                Rnd.kHalfAway, kFtzIn))
    v = mul(core_act(outs[0], act), outs[1])
    return store(v, ofm, Rnd.kHalfAway)


def gemm_eltwise_bfp16(a, bt, bias, ifm2, ifm2_bfp, m2, m, k, n, sv_m, sv_n, sv_k, is_mul, ofm):
    am, ae = bfp16_fields(a, m)
    bm, be = bfp16_fields(bt, n)
    b0 = np.broadcast_to(f32_bits(bias.reshape(1, n)) if bias is not None else np.uint32(0), (m, n)).copy()
    if not ifm2_bfp:
        x2 = np.broadcast_to(bf16_to_f32(np.asarray(ifm2, np.uint16).reshape(m2, n)), (m, n))
        if is_mul:
            v = mul(round_float(f32v(gemm_bfp16_acc(am, ae, bm, be, b0)), "bf16", Rnd.kHalfAway, kFtzIn), x2)
        else:
            seed = x2 if bias is None else add(x2, bias.reshape(1, n))
            v = f32v(gemm_bfp16_acc(am, ae, bm, be, f32_bits(np.ascontiguousarray(seed, np.float32))))
        return store(v, ofm, Rnd.kHalfAway)
    x2 = np.broadcast_to(np.asarray(ifm2, np.uint8).reshape(m2, -1), (m, n // 8 * 9))
    a2 = decode_bfp16_identity(x2, row=n // 8)

    def ident(acc):
        return f32_bits(mmac_add_bfp16(x2, a2, f32v(acc), n)).reshape(m, n)
    early = np.zeros((m, n), bool)
    early[:sv_m, :sv_n] = True
    early |= sv_k == 64
    v_early = gemm_bfp16_acc(am, ae, bm, be, ident(b0))
    first = gemm_bfp16_acc(am, ae, bm, be, b0, 0, 64)
    v_late = gemm_bfp16_acc(am, ae, bm, be, ident(first), 64)
    return store(f32v(np.where(early, v_early, v_late)), ofm, Rnd.kHalfAway)


def gemm_mx6(a, bt, bias, m, k, n, narrow, act, alpha, ofm):
    am, ae, asx = mx6_fields(a, m)
    bm, be, bsx = mx6_fields(bt, n)
    acc = np.broadcast_to(f32_bits(bias.reshape(1, n)) if bias is not None else np.uint32(0), (m, n)).copy()
    for c in range(0, k, 128):
        acc = mmac_mx6(acc, am[:, None, c:c + 128], asx[:, None, c // 2:c // 2 + 64], ae[:, None, c // 16:c // 16 + 8],
                       bm[None, :, c:c + 128], bsx[None, :, c // 2:c // 2 + 64], be[None, :, c // 16:c // 16 + 8])
    return store(activate_block(f32v(acc), narrow, act, alpha), ofm, Rnd.kHalfAway)


def gemm_fp8(a, b, c0, c2, m, k, n, ofm):
    a = np.asarray(a, np.uint8).reshape(m, k)
    b = np.asarray(b, np.uint8).reshape(k, n)
    acc = np.zeros((m, n), np.uint32)
    for k0 in range(0, k, 64):
        aa = np.zeros((m, 64), np.uint8); aa[:, :min(64, k - k0)] = a[:, k0:k0 + 64]
        bb = np.zeros((64, n), np.uint8); bb[:min(64, k - k0)] = b[k0:k0 + 64]
        acc = mmac_fp8(acc, aa[:, None, :], bb.T[None, :, :])
    return store(fma(f32v(acc), c2.reshape(1, n), c0.reshape(1, n)), ofm, Rnd.kHalfAway)


def fold_tiles(w_logical, sw, kw, ci_sv, kh_sv, co_sv, ktile):
    """Logical weights (kh_sv x kw x ci_sv x co_sv) -> folded weights
    [nb, kh_sv, ng, 64 lanes, ktile K], plus a mask of the non-padding entries."""
    fwo = max(64 // co_sv, 1)
    fkw = kw + (fwo - 1) * sw
    per = ktile // ci_sv if ci_sv < ktile else 1
    fkw_al = -(-fkw // per) * per
    shp = w_logical.shape[4:]
    wo = np.zeros((kh_sv, fkw_al, ci_sv, 64) + shp, w_logical.dtype)
    valid = np.zeros((kh_sv, fkw_al, ci_sv, 64), bool)
    for c in range(fwo):
        wo[:, c * sw:c * sw + kw, :, co_sv * c:co_sv * (c + 1)] = w_logical
        valid[:, c * sw:c * sw + kw, :, co_sv * c:co_sv * (c + 1)] = True
    if ci_sv >= ktile:
        nb, ng = ci_sv // ktile, fkw_al
        t = wo.reshape((kh_sv, fkw_al, nb, ktile, 64) + shp).transpose((2, 0, 1, 4, 3) + tuple(range(5, 5 + len(shp))))
        v = valid.reshape(kh_sv, fkw_al, nb, ktile, 64).transpose(2, 0, 1, 4, 3)
    else:
        nb, ng = 1, fkw_al // per
        t = wo.reshape((1, kh_sv, ng, ktile, 64) + shp).transpose((0, 1, 2, 4, 3) + tuple(range(5, 5 + len(shp))))
        v = valid.reshape(1, kh_sv, ng, ktile, 64).transpose(0, 1, 2, 4, 3)
    return t, v, (fwo, fkw_al, per, nb, ng)


def conv_tiles(xfield, h, w, ci, kh, kw, co, sh, sw, kh_sv, ci_sv, co_sv, ktile, mmac, wtile, acc):
    """Run the folded MMAC chain. xfield(y, x, c0, c1): the ifm fields of pixels
    (y, x) and channels [c0, c1); wtile(kh_iter, ci_iter, co_iter): the folded
    weights of fold_tiles. acc: Ho x Wo x Co fp32 bits."""
    ho, wo = (h - kh) // sh + 1, (w - kw) // sw + 1
    fwo = max(64 // co_sv, 1)
    nw = -(-wo // fwo)
    out = np.zeros((ho, nw * fwo, co), np.uint32)
    out[:, :wo] = acc
    for kh_iter in range(kh // kh_sv):
        for ci_iter in range(ci // ci_sv):
            c0 = ci_sv * ci_iter
            for co_iter in range(co // co_sv):
                o0 = co_sv * co_iter
                tiles, (fwo_, fkw_al, per, nb, ng) = wtile(kh_iter, ci_iter, co_iter)
                cols = np.zeros((ho, nw, 64), np.uint32)
                cols[:, :, :fwo * co_sv] = out[:, :, o0:o0 + co_sv].reshape(ho, nw, fwo * co_sv)
                cols = cols.reshape(ho * nw, 64)
                for cb in range(nb):
                    for ky in range(kh_sv):
                        y = np.arange(ho)[:, None] * sh + kh_sv * kh_iter + ky
                        for gg in range(ng):
                            if ci_sv >= ktile:
                                x = np.broadcast_to(np.arange(nw)[None, :] * fwo * sw + gg, (ho, nw))
                                X = xfield(np.broadcast_to(y, (ho, nw)), x, c0 + cb * ktile, c0 + cb * ktile + ktile)
                            else:
                                x = np.arange(nw)[None, :, None] * fwo * sw + gg * per + np.arange(per)
                                X = xfield(np.broadcast_to(y[:, :, None], (ho, nw, per)),
                                           np.broadcast_to(x, (ho, nw, per)), c0, c0 + ci_sv)
                            cols = mmac(cols, X, tiles, cb, ky, gg)
                out[:, :, o0:o0 + co_sv] = cols.reshape(ho, nw, 64)[:, :, :fwo * co_sv].reshape(ho, nw * fwo, co_sv)
    return out[:, :wo]


def conv_block(ifm, wts, bias, ifm2, h2, w2, c2, h, w, ci, kh, kw, co, sh, sw, kh_sv, ci_sv, co_sv,
               narrow, act, alpha, ofm, mx6=False):
    """bfp16 / mx6 convolution on packed ifm (H x W x Ci) and weights (Kh x Kw x Co x Ci)."""
    blk, ktile = (16, 128) if mx6 else (8, 64)
    if mx6:
        xm, xe, xs = mx6_fields(ifm, h * w)
        zm, ze, zs = mx6_fields(wts, kh * kw * co)
    else:
        xm, xe = bfp16_fields(ifm, h * w)
        zm, ze = bfp16_fields(wts, kh * kw * co)
    xm = xm.reshape(h, w, ci)
    xe = np.repeat(xe.reshape(h, w, ci // blk), blk, axis=2)
    if mx6:
        xs = np.repeat(xs.reshape(h, w, ci // 2), 2, axis=2)
    zm = zm.reshape(kh, kw, co, ci).transpose(0, 1, 3, 2)
    ze = np.repeat(ze.reshape(kh, kw, co, ci // blk), blk, axis=3).transpose(0, 1, 3, 2)
    if mx6:
        zs = np.repeat(zs.reshape(kh, kw, co, ci // 2), 2, axis=3).transpose(0, 1, 3, 2)
    wpad = w + 2 * ktile + 64

    def pad(a):
        p = np.zeros((h, wpad, ci), I64)
        p[:, :w] = a
        return p
    xm, xe = pad(xm), pad(xe)
    if mx6:
        xs = pad(xs)

    def xfield(y, x, c0, c1):
        f = [xm[y, x, c0:c1], xe[y, x, c0:c1]] + ([xs[y, x, c0:c1]] if mx6 else [])
        return [a.reshape(-1, ktile) for a in f]

    def wtile(kh_iter, ci_iter, co_iter):
        sl = (slice(kh_sv * kh_iter, kh_sv * (kh_iter + 1)), slice(None),
              slice(ci_sv * ci_iter, ci_sv * (ci_iter + 1)), slice(co_sv * co_iter, co_sv * (co_iter + 1)))
        fields = [zm[sl], ze[sl]] + ([zs[sl]] if mx6 else [])
        out = []
        for f in fields:
            t, valid, geo = fold_tiles(f, sw, kw, ci_sv, kh_sv, co_sv, ktile)
            out.append((t, valid))
        (tm, valid), (te, _) = out[0], out[1]
        # a padding block: mantissas 0, exponent 1, shifts 0
        te = np.where(valid, te, 1)
        tiles = [tm, te] + ([out[2][0]] if mx6 else [])
        return tiles, geo

    def mmac(cols, X, tiles, cb, ky, gg):
        z = [t[cb, ky, gg] for t in tiles]
        if mx6:
            return mmac_mx6(cols, X[0][:, None], X[2][:, None, ::2], X[1][:, None, ::16], z[0][None],
                            z[2][None, :, ::2], z[1][None, :, ::16])
        return mmac_bfp16(cols, X[0][:, None], X[1][:, None, ::8], z[0][None], z[1][None, :, ::8])

    ho, wo = (h - kh) // sh + 1, (w - kw) // sw + 1
    acc = np.broadcast_to(f32_bits(bias.reshape(1, 1, co)) if bias is not None else np.uint32(0), (ho, wo, co)).copy()
    if ifm2 is not None:
        r = np.broadcast_to(bf16_to_f32(np.asarray(ifm2, np.uint16).reshape(h2, w2, c2)), (ho, wo, co))
        acc = f32_bits(np.ascontiguousarray(r if bias is None else add(r, bias.reshape(1, 1, co))))
    acc = conv_tiles(xfield, h, w, ci, kh, kw, co, sh, sw, kh_sv, ci_sv, co_sv, ktile, mmac, wtile, acc)
    if ifm2 is not None:
        return store(f32v(acc), ofm, Rnd.kHalfEven)
    return store(activate_block(f32v(acc), narrow, act, alpha), ofm, Rnd.kHalfAway)


def conv_fp8(ifm, wts, c0, c2, ifm2, ifm2_t, h2, w2, c2d, coeff, h, w, ci, kh, kw, co, sh, sw, kh_sv,
             ci_sv, co_sv, act, a0, a1, slope, ofm):
    """fp8 convolution: ifm H x W x Ci, weights Kh x Kw x Ci x Co e4m3 bytes."""
    x = np.zeros((h, w + 192, ci), np.uint8)
    x[:, :w] = np.asarray(ifm, np.uint8).reshape(h, w, ci)
    wl = np.asarray(wts, np.uint8).reshape(kh, kw, ci, co)

    def xfield(y, xx, cc0, cc1):
        return x[y, xx, cc0:cc1].reshape(-1, 64)

    def wtile(kh_iter, ci_iter, co_iter):
        sl = (slice(kh_sv * kh_iter, kh_sv * (kh_iter + 1)), slice(None),
              slice(ci_sv * ci_iter, ci_sv * (ci_iter + 1)), slice(co_sv * co_iter, co_sv * (co_iter + 1)))
        t, _, geo = fold_tiles(wl[sl], sw, kw, ci_sv, kh_sv, co_sv, 64)
        return t, geo

    def mmac(cols, X, tiles, cb, ky, gg):
        return mmac_fp8(cols, X[:, None, :], tiles[cb, ky, gg][None, :, :])

    ho, wo = (h - kh) // sh + 1, (w - kw) // sw + 1
    acc = f32v(conv_tiles(xfield, h, w, ci, kh, kw, co, sh, sw, kh_sv, ci_sv, co_sv, 64, mmac, wtile,
                          np.zeros((ho, wo, co), np.uint32)))
    if ifm2 is not None:
        dec = {"fp8": fp8_to_f32, "bf16": bf16_to_f32, "f16": f16_to_f32}[ifm2_t]
        r = np.broadcast_to(dec(np.asarray(ifm2).reshape(h2, w2, c2d)), (ho, wo, co))
        v = fma(acc, c2, fma(r, F32(coeff), c0))
    else:
        v = fma(acc, c2, c0)
        if act == 1:
            v = vmax(v, F32(0))
        elif act == 2:
            v = vclamp(v, F32(0), F32(a0))
        elif act == 3:
            v = fma(vmin(v, F32(0)), F32(a0), vmax(v, F32(0)))
        elif act == 8:
            v = vclamp(v, F32(a0), F32(a1))
        elif act == 9:
            with np.errstate(all="ignore"):
                v = np.where(v < 0, mul(v, slope), v)
    return store(v, ofm, Rnd.kHalfAway)


def depthwise_fp8(ifm, wts, c0, c2, h, w, c, kh, kw, sh, sw, act, a0, ofm):
    x = fp8_to_f32(np.asarray(ifm, np.uint8)).reshape(h, w, c)
    wt = fp8_to_f32(np.asarray(wts, np.uint8)).reshape(kh, kw, c)
    ho, wo = (h - kh) // sh + 1, (w - kw) // sw + 1
    acc = np.zeros((ho, wo, c), np.float32)
    for y in range(kh):
        for xx in range(kw):
            acc = add(acc, mul(x[y:y + (ho - 1) * sh + 1:sh, xx:xx + (wo - 1) * sw + 1:sw], wt[y, xx]))
    v = fma(acc, c2, c0)
    if act == 1:
        v = vmax(v, F32(0))
    elif act == 2:
        v = vclamp(v, F32(0), F32(a0))
    elif act == 3:
        v = vmax(v, mul(v, F32(a0)))
    return store(v, ofm, Rnd.kHalfAway)
