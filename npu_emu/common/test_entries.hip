/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Test-only entry points of the common primitives (int_round.h, float_cvt.h, mac.h,
 * minmax.h, block_fp.h, mmac.h, nlf.h, act.h); one output element per input element.
 *
 * npu_test_int_round: op 0 shift_round(v, s, rnd); 1 shift_round_mag(uint64(v), s, rnd,
 *   arg != 0); 2 saturate(v, arg); 3 wrap(v, arg); 4 saturate_sym(v, arg).
 * npu_test_float_to_int: op 0 float_to_int(x, t, shift, rnd); 1 requant_tail(x, t).
 * npu_test_int_to_float: int_to_float(v, e, rnd).
 * npu_test_encode_float: encode_float(x, t, rnd, flags).
 * npu_test_decode_float: decode_float(c, t, flags).
 * npu_test_float_misc: op 0 round_float(x, t, rnd, flags); 1 scalar_cast(x, t, rnd).
 * npu_test_fp8_decode: fp8_decode_identity over the whole buffer.
 * npu_test_mac: op 0 fma_ftz(a, b, c, flags); 1 mac_exact(x, b, c, flags); 2 ftz(a).
 * npu_test_mul16: mul16(x, y, t), t = NPU_BF16 / NPU_F16.
 * npu_test_minmax: op 0 vmax(a, b); 1 vmin(a, b); 2 vclamp(a, b, c).
 * npu_test_minmax16: op 0 max16(a, b, t); 1 min16(a, b, t); op 2 above(a, b) on the decoded
 *   values (1 / 0 as the code).
 * npu_test_block_decode: value k of block b (n = 8 or 16 per block), op 0 decode_bfp16,
 *   1 decode_bfp16_identity (groups from the buffer start), 2 decode_bfp16_group,
 *   3 decode_mx6, 4 decode_bfp16_bf16 (bits in the low half); out as fp32 bits.
 * npu_test_block_encode: op 0 encode_bfp16, 1 encode_mx6 of nblk blocks.
 * npu_test_mmac: n MMACs; op 0 mmac_bfp16 (x, z: 8 bfp16 blocks = 72 bytes each), 1 mmac_mx6
 *   (8 mx6 blocks = 96 bytes), 2 mmac_fp8 (64 bytes), 3 mmac_bf16 (32 bf16 = 64 bytes).
 * npu_test_vaddr_f32_reduce: out[r] = vaddr_f32_reduce(lanes[r][0..n_lanes),
 *   scalar ? scalar[r] : none), r < rows; 1 <= n_lanes <= 32.
 * npu_test_nlf_unary: out[i] = f(x[i]), op 0 hw_inv_f32, 1 hw_invsqrt_f32, 2 hw_approx_ln,
 *   3 hw_exp2_bf20, 4 hw_exp2_bf16, 5 hw_gelu_f32, 6 hw_silu_f32, 7 / 8 hw_sigtanh_f32
 *   sigmoid / tanh, 9 sigtanh_tanh_bf16, 10 / 11 tanh_acc bf16 / f16.
 * npu_test_nlf16: out[i] = f(x[i]) on 16-bit codes, op 0 / 1 hw_sqrt16 bf16 / f16,
 *   2 / 3 hw_inv16 bf16 / f16, 4 hw_invsqrt_f16, 5 / 6 tanh16 bf16 / f16.
 * npu_test_lut: op 0: out[i] = lut_index(x[i], k, entries, rnd); op 1: out[i] =
 *   lut_interp(x[i], lut, that index, slope_t, rnd).
 */
#include "common/act.h"
#include "common/block_fp.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/mmac.h"
#include "common/nlf.h"

namespace npu_emu {
namespace {

__global__ void int_round_kernel(int64_t n, int op, const int64_t* v, const int32_t* s, int rnd,
                                 int arg, int64_t* out) {
  NPU_TID;
  switch (op) {
    case 0:
      out[i] = shift_round(v[i], s[i], Rnd(rnd));
      break;
    case 1:
      out[i] = int64_t(shift_round_mag(uint64_t(v[i]), s[i], Rnd(rnd), arg != 0));
      break;
    case 2:
      out[i] = saturate(v[i], npu_dtype(arg));
      break;
    case 4:
      out[i] = saturate_sym(v[i], npu_dtype(arg));
      break;
    case 5:
      out[i] = sat_add64(v[i], v[(i * 7 + 3) % n]);
      break;
    default:
      out[i] = wrap(v[i], npu_dtype(arg));
      break;
  }
}

__global__ void float_to_int_kernel(int64_t n, int op, const float* x, npu_dtype t, int shift,
                                    Rnd rnd, int64_t* out) {
  NPU_TID;
  out[i] = op == 0 ? float_to_int(x[i], t, shift, rnd) : requant_tail(x[i], t);
}

__global__ void int_to_float_kernel(int64_t n, const int64_t* v, int e, Rnd rnd, float* out) {
  NPU_TID;
  out[i] = int_to_float(v[i], e, rnd);
}

__global__ void encode_kernel(int64_t n, const float* x, npu_dtype t, Rnd rnd, int flags,
                              uint32_t* out) {
  NPU_TID;
  out[i] = encode_float(x[i], t, rnd, flags);
}

__global__ void decode_kernel(int64_t n, const uint32_t* c, npu_dtype t, int flags, float* out) {
  NPU_TID;
  out[i] = decode_float(c[i], t, flags);
}

__global__ void float_misc_kernel(int64_t n, int op, const float* x, npu_dtype t, Rnd rnd,
                                  int flags, float* out) {
  NPU_TID;
  switch (op) {
    case 0:
      out[i] = round_float(x[i], t, rnd, flags);
      break;
    default:
      out[i] = scalar_cast(x[i], t, rnd);
      break;
  }
}

__global__ void fp8_decode_kernel(int64_t n, const uint8_t* c, float* out) {
  NPU_TID;
  out[i] = fp8_decode_identity(c, i, n);
}

__global__ void mac_kernel(int64_t n, int op, const float* a, const float* b, const float* c,
                           const int64_t* x, int flags, float* out) {
  NPU_TID;
  switch (op) {
    case 0:
      out[i] = fma_ftz(a[i], b[i], c[i], flags);
      break;
    case 1:
      out[i] = mac_exact(x[i], b[i], c[i], flags);
      break;
    case 3:
      out[i] = mul_q(a[i], b[i]);
      break;
    case 4:
      out[i] = add_q(a[i], b[i]);
      break;
    case 5:
      out[i] = fma_q(a[i], b[i], c[i]);
      break;
    case 7:
      out[i] = sub_q(a[i], b[i]);
      break;
    case 8:
      out[i] = sqrt_q(a[i]);
      break;
    case 6:  // SigTanh with the coefficients k0 = b, k1 = c (k2 fixed)
      out[i] = hw_sigtanh_f32(a[i], flags != 0, b[i], c[i], flags ? kTanhK2 : kSigmoidK2);
      break;
    default:
      out[i] = ftz(a[i]);
      break;
  }
}

__global__ void mul16_kernel(int64_t n, const uint16_t* x, const uint16_t* y, npu_dtype t,
                             uint16_t* out) {
  NPU_TID;
  out[i] = mul16(x[i], y[i], t);
}

__global__ void minmax_kernel(int64_t n, int op, const float* a, const float* b, const float* c,
                              float* out) {
  NPU_TID;
  out[i] = op == 0 ? vmax(a[i], b[i]) : op == 1 ? vmin(a[i], b[i]) : vclamp(a[i], b[i], c[i]);
}

__global__ void minmax16_kernel(int64_t n, int op, const uint16_t* a, const uint16_t* b,
                                npu_dtype t, uint16_t* out) {
  NPU_TID;
  out[i] = op == 0   ? max16(a[i], b[i], t)
           : op == 1 ? min16(a[i], b[i], t)
                     : uint16_t(above(decode_float(a[i], t), decode_float(b[i], t)));
}

__global__ void block_decode_kernel(int64_t n, int op, const uint8_t* blk, int64_t nblk,
                                    uint32_t* out) {
  NPU_TID;
  const int per = op == 3 ? 16 : 8;
  const int64_t b = i / per;
  const int k = int(i % per);
  switch (op) {
    case 0:
      out[i] = f32_bits(decode_bfp16(blk + b * 9, k));
      break;
    case 1:
      out[i] = f32_bits(decode_bfp16_identity(blk, b & ~int64_t(7), nblk, b, k));
      break;
    case 2:
      out[i] = f32_bits(decode_bfp16_group(blk, b, nblk, k));
      break;
    case 3:
      out[i] = f32_bits(decode_mx6(blk + b * 12, k));
      break;
    default:
      out[i] = decode_bfp16_bf16(blk, b, nblk, k);
      break;
  }
}

__global__ void block_encode_kernel(int64_t n, int op, const float* v, Rnd rnd, uint8_t* out) {
  NPU_TID;
  if (op == 0)
    encode_bfp16(v + i * 8, rnd, out + i * 9);
  else
    encode_mx6(v + i * 16, rnd, out + i * 12);
}

__global__ void mmac_kernel(int64_t n, int op, const uint32_t* acc, const uint8_t* x,
                            const uint8_t* z, uint32_t* out) {
  NPU_TID;
  if (op == 0) {
    Bfp16Fields fx[8], fz[8];
    for (int b = 0; b < 8; ++b) {
      bfp16_fields(x + (i * 8 + b) * 9, &fx[b]);
      bfp16_fields(z + (i * 8 + b) * 9, &fz[b]);
    }
    out[i] = mmac_bfp16(acc[i], fx, fz);
  } else if (op == 1) {
    Mx6Fields fx[8], fz[8];
    for (int b = 0; b < 8; ++b) {
      mx6_fields(x + (i * 8 + b) * 12, &fx[b]);
      mx6_fields(z + (i * 8 + b) * 12, &fz[b]);
    }
    out[i] = mmac_mx6(acc[i], fx, fz);
  } else if (op == 2) {
    out[i] = mmac_fp8(acc[i], x + i * 64, 1, z + i * 64, 1, 64);
  } else {
    const uint16_t* xs = reinterpret_cast<const uint16_t*>(x) + i * 32;
    const uint16_t* zs = reinterpret_cast<const uint16_t*>(z) + i * 32;
    out[i] = mmac_bf16(acc[i], xs, 1, zs, 1, 32);
  }
}

__global__ void vaddr_test_kernel(int64_t n, const float* lanes, int n_lanes,
                                  const float* scalar, float* out) {
  NPU_TID;
  out[i] = scalar ? vaddr_f32_reduce(lanes + i * n_lanes, n_lanes, true, scalar[i])
                  : vaddr_f32_reduce(lanes + i * n_lanes, n_lanes);
}

__global__ void nlf_unary_test_kernel(int64_t n, int op, const float* x, float* out,
                                      const uint32_t* exc) {
  NPU_TID;
  const float v = x[i];
  switch (op) {
    case 0:
      out[i] = hw_inv_f32(v, exc);
      break;
    case 1:
      out[i] = hw_invsqrt_f32(v, exc);
      break;
    case 2:
      out[i] = hw_approx_ln(v);
      break;
    case 3:
      out[i] = hw_exp2_bf20(v);
      break;
    case 4:
      out[i] = hw_exp2_bf16(v);
      break;
    case 5:
      out[i] = hw_gelu_f32(v);
      break;
    case 6:
      out[i] = hw_silu_f32(v);
      break;
    case 7:
      out[i] = hw_sigtanh_f32(v, false, kSigTanhK0, kSigmoidK1, kSigmoidK2);
      break;
    case 8:
      out[i] = hw_sigtanh_f32(v, true, kSigTanhK0, kTanhK1, kTanhK2);
      break;
    case 9:
      out[i] = sigtanh_tanh_bf16(v);
      break;
    default:
      out[i] = tanh_acc(v, op == 11 ? NPU_F16 : NPU_BF16);
      break;
  }
}

__global__ void nlf16_test_kernel(int64_t n, int op, const uint16_t* x, uint16_t* out,
                                  const uint32_t* exc) {
  NPU_TID;
  const npu_dtype t = op & 1 ? NPU_F16 : NPU_BF16;
  switch (op) {
    case 0:
    case 1:
      out[i] = hw_sqrt16(x[i], t);
      break;
    case 2:
    case 3:
      out[i] = hw_inv16(x[i], t);
      break;
    case 4:
      out[i] = hw_invsqrt_f16(x[i], exc);
      break;
    default:
      out[i] = tanh16(x[i], op == 6 ? NPU_F16 : NPU_BF16);
      break;
  }
}

__global__ void lut_test_kernel(int64_t n, int op, const float* x, const float* lut, float k,
                                int entries, npu_dtype slope_t, Rnd rnd, float* out) {
  NPU_TID;
  const int idx = lut_index(x[i], k, entries, rnd);
  out[i] = op == 0 ? float(idx) : lut_interp(x[i], lut, idx, slope_t, rnd);
}

}  // namespace
}  // namespace npu_emu

using namespace npu_emu;

extern "C" int npu_test_int_round(void* stream, int op, const int64_t* v, const int32_t* s,
                                  int rnd, int arg, int64_t* out, int64_t n) {
  if (op < 0 || op > 5 || rnd < 0 || rnd > 9 || n < 0) return -1;
  return launch(stream, n, int_round_kernel, op, v, s, rnd, arg, out);
}

extern "C" int npu_test_float_to_int(void* stream, int op, const float* x, int t, int shift,
                                     int rnd, int64_t* out, int64_t n) {
  if (op < 0 || op > 1 || rnd < 0 || rnd > 9 || n < 0) return -1;
  return launch(stream, n, float_to_int_kernel, op, x, npu_dtype(t), shift, Rnd(rnd), out);
}

extern "C" int npu_test_int_to_float(void* stream, const int64_t* v, int e, int rnd, float* out,
                                     int64_t n) {
  if (rnd < 0 || rnd > 9 || n < 0) return -1;
  return launch(stream, n, int_to_float_kernel, v, e, Rnd(rnd), out);
}

extern "C" int npu_test_encode_float(void* stream, const float* x, int t, int rnd, int flags,
                                     uint32_t* out, int64_t n) {
  if (rnd < 0 || rnd > 9 || n < 0) return -1;
  return launch(stream, n, encode_kernel, x, npu_dtype(t), Rnd(rnd), flags, out);
}

extern "C" int npu_test_decode_float(void* stream, const uint32_t* c, int t, int flags,
                                     float* out, int64_t n) {
  if (n < 0) return -1;
  return launch(stream, n, decode_kernel, c, npu_dtype(t), flags, out);
}

extern "C" int npu_test_float_misc(void* stream, int op, const float* x, int t, int rnd,
                                   int flags, float* out, int64_t n) {
  if (op < 0 || op > 1 || rnd < 0 || rnd > 9 || n < 0) return -1;
  return launch(stream, n, float_misc_kernel, op, x, npu_dtype(t), Rnd(rnd), flags, out);
}

extern "C" int npu_test_fp8_decode(void* stream, const uint8_t* c, float* out, int64_t n) {
  if (n < 0) return -1;
  return launch(stream, n, fp8_decode_kernel, c, out);
}

extern "C" int npu_test_mac(void* stream, int op, const float* a, const float* b, const float* c,
                            const int64_t* x, int flags, float* out, int64_t n) {
  if (op < 0 || op > 8 || n < 0) return -1;
  return launch(stream, n, mac_kernel, op, a, b, c, x, flags, out);
}

extern "C" int npu_test_mul16(void* stream, const uint16_t* x, const uint16_t* y, npu_dtype t,
                              uint16_t* out, int64_t n) {
  if (n < 0 || (t != NPU_BF16 && t != NPU_F16)) return -1;
  return launch(stream, n, mul16_kernel, x, y, t, out);
}

extern "C" int npu_test_minmax(void* stream, int op, const float* a, const float* b,
                               const float* c, float* out, int64_t n) {
  if (op < 0 || op > 2 || n < 0) return -1;
  return launch(stream, n, minmax_kernel, op, a, b, c, out);
}

extern "C" int npu_test_minmax16(void* stream, int op, const uint16_t* a, const uint16_t* b,
                                 int t, uint16_t* out, int64_t n) {
  if (op < 0 || op > 2 || (t != NPU_BF16 && t != NPU_F16) || n < 0) return -1;
  return launch(stream, n, minmax16_kernel, op, a, b, npu_dtype(t), out);
}

extern "C" int npu_test_block_decode(void* stream, int op, const uint8_t* blk, int64_t nblk,
                                     uint32_t* out) {
  if (op < 0 || op > 4 || nblk < 0) return -1;
  return launch(stream, nblk * (op == 3 ? 16 : 8), block_decode_kernel, op, blk, nblk, out);
}

extern "C" int npu_test_block_encode(void* stream, int op, const float* v, int rnd, uint8_t* out,
                                     int64_t nblk) {
  if (op < 0 || op > 1 || rnd < 0 || rnd > 9 || nblk < 0) return -1;
  return launch(stream, nblk, block_encode_kernel, op, v, Rnd(rnd), out);
}

extern "C" int npu_test_mmac(void* stream, int op, const uint32_t* acc, const uint8_t* x,
                             const uint8_t* z, uint32_t* out, int64_t n) {
  if (op < 0 || op > 3 || n < 0) return -1;
  return launch(stream, n, mmac_kernel, op, acc, x, z, out);
}

extern "C" int npu_test_vaddr_f32_reduce(void* stream, const float* lanes, int n_lanes,
                                         const float* scalar, float* out, int64_t rows) {
  if (n_lanes < 1 || n_lanes > 32 || rows < 0) return -1;
  return launch(stream, rows, vaddr_test_kernel, lanes, n_lanes, scalar, out);
}

extern "C" int npu_test_nlf_unary(void* stream, int op, const float* x, float* out,
                                  int64_t n) {
  if (op < 0 || op > 11 || n < 0) return -1;
  const uint32_t* exc = nullptr;
  if (op <= 1) {
    const int rc = npu_emu_nlf_table(op == 0 ? NPU_EMU_NLF_INV_F32 : NPU_EMU_NLF_INVSQRT_F32, &exc);
    if (rc) return rc;
  }
  return launch(stream, n, nlf_unary_test_kernel, op, x, out, exc);
}

extern "C" int npu_test_nlf16(void* stream, int op, const uint16_t* x, uint16_t* out, int64_t n) {
  if (op < 0 || op > 6 || n < 0) return -1;
  const uint32_t* exc = nullptr;
  if (op == 4) {
    const int rc = npu_emu_nlf_table(NPU_EMU_NLF_INVSQRT_F32, &exc);
    if (rc) return rc;
  }
  return launch(stream, n, nlf16_test_kernel, op, x, out, exc);
}

extern "C" int npu_test_lut(void* stream, int op, const float* x, const float* lut, float k,
                            int entries, int slope_t, int rnd, float* out, int64_t n) {
  if (op < 0 || op > 1 || entries < 2 || n < 0) return -1;
  return launch(stream, n, lut_test_kernel, op, x, lut, k, entries, npu_dtype(slope_t), Rnd(rnd),
                out);
}
