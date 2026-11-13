/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and C entry points for npu_emu.h. One thread per element (or per
 * block for block floating point); these are reference-speed kernels.
 */
#include <hip/hip_runtime.h>

#include "npu_emu.h"
#include "npu_emu_device.h"
#include "npu_emu_launch.h"

using namespace npu_emu;

namespace {

// ---- A. format conversion --------------------------------------------------

__global__ void convert_float_kernel(int64_t n, const void* in, npu_dtype it,
                                void* out, npu_dtype ot, Rnd rnd) {
  NPU_TID;
  store_float(out, ot, i, load_float(in, it, i), rnd);
}

__global__ void float_to_bfp16_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                           Rnd rnd) {
  NPU_TID;
  float v[8];
  for (int k = 0; k < 8; ++k) v[k] = load_float(in, it, 8 * i + k);
  encode_bfp16(v, rnd, static_cast<uint8_t*>(out) + 9 * i);
}

__global__ void bfp16_to_bf16_kernel(int64_t n, const void* in, void* out) {
  NPU_TID;
  const uint8_t* blk = static_cast<const uint8_t*>(in) + 9 * (i / 8);
  static_cast<uint16_t*>(out)[i] = f32_to_bf16(decode_bfp16(blk, int(i % 8)), kEven);
}

__global__ void bf16_to_mx6_kernel(int64_t n, const void* in, void* out, Rnd rnd) {
  NPU_TID;
  float v[16];
  for (int k = 0; k < 16; ++k) v[k] = load_float(in, NPU_BF16, 16 * i + k);
  encode_mx6(v, rnd, static_cast<uint8_t*>(out) + 12 * i);
}

__global__ void mx6_to_bf16_kernel(int64_t n, const void* in, void* out) {
  NPU_TID;
  const uint8_t* blk = static_cast<const uint8_t*>(in) + 12 * (i / 16);
  static_cast<uint16_t*>(out)[i] = f32_to_bf16(decode_mx6(blk, int(i % 16)), kEven);
}

// ---- B. fp8 ----------------------------------------------------------------

// v = x * mul rounded to fp32 (exact for f16 / bf16 operands), optional clamp
// to the E4M3FN range, then to fp8.
__global__ void float_to_fp8_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                         float mul, bool clamp, Rnd rnd) {
  NPU_TID;
  float v = load_float(in, it, i) * mul;
  // NaN passes the clamp unchanged (np.clip semantics; unverified on the NPU).
  if (clamp && v == v) v = fminf(448.0f, fmaxf(-448.0f, v));
  static_cast<uint8_t*>(out)[i] = f32_to_fp8(v, rnd);
}

__global__ void fp8_to_float_kernel(int64_t n, const void* in, void* out, npu_dtype ot,
                           float scale, Rnd rnd) {
  NPU_TID;
  const float v = fp8_to_f32_row(static_cast<const uint8_t*>(in), i, n) * scale;
  store_float(out, ot, i, v, rnd);
}

// ---- C. quantize -----------------------------------------------------------

__global__ void quantize_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                        npu_dtype ot, float scale, float zp) {
  NPU_TID;
  int64_t lo, hi;
  int_range(ot, &lo, &hi);
  store_int(out, ot, i, sat_rne(fmaf(load_float(in, it, i), scale, zp), lo, hi));
}

// bfp16 -> bf16 (exact) -> unsigned integer (to_fixed_sign with sign 0).
__device__ inline int64_t bfp16_to_unsigned(const void* in, int64_t i,
                                            float mul, float zp, int64_t hi) {
  const uint8_t* blk = static_cast<const uint8_t*>(in) + 9 * (i / 8);
  const float x = bf16_to_f32(f32_to_bf16(decode_bfp16(blk, int(i % 8)), kEven));
  return sat_rne(fmaf(x, mul, zp), 0, hi);
}

__global__ void bfp16_quantize_kernel(int64_t n, const void* in, void* out,
                              npu_dtype ot, float mul, float zp) {
  NPU_TID;
  store_int(out, ot, i, bfp16_to_unsigned(in, i, mul, zp, ot == NPU_U8 ? 255 : 65535));
}

// ---- D. dequantize ---------------------------------------------------------

__global__ void dequantize_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                          npu_dtype ot, float scale, float c) {
  NPU_TID;
  const float v = mac_exact(load_int(in, it, i), scale, c);
  store_float(out, ot, i, v, kEven);
}

__global__ void dequantize_to_bfp16_kernel(int64_t n, const void* in, npu_dtype it,
                              void* out, float scale, int32_t zpi) {
  NPU_TID;
  float v[8];
  for (int k = 0; k < 8; ++k) {
    const float d = float(load_int(in, it, 8 * i + k) - zpi) * scale;
    v[k] = bf16_to_f32(f32_to_bf16(d, kEven));
  }
  encode_bfp16(v, kEven, static_cast<uint8_t*>(out) + 9 * i);
}

// One thread per 8-value block of row-major [N][K]; group = 32 along K.
__global__ void dequantize_w4_to_bfp16_kernel(int64_t n, const uint8_t* q, const uint8_t* zp,
                              const uint16_t* scale, void* out, int64_t K) {
  NPU_TID;
  const int64_t row = (8 * i) / K, col = (8 * i) % K;
  const int64_t g = row * (K / 32) + col / 32;
  const float s = bf16_to_f32(scale[g]);
  float v[8];
  for (int k = 0; k < 8; ++k)
    v[k] = float(int(q[row * K + col + k]) - int(zp[g])) * s;
  encode_bfp16(v, kEven, static_cast<uint8_t*>(out) + 9 * i);
}

// ---- E. requantize ---------------------------------------------------------

// dqq.hpp: x * c1 + c0 in one MAC (exact integer x, one fp32 rounding). A U32
// input is loaded as int32 and
// corrected with (x >> 31) * -(c1 * 2^32).
__device__ inline float requant_mac(int64_t x, npu_dtype it, float c1, float c0) {
  if (it == NPU_U32) {
    const int32_t raw = int32_t(uint32_t(x));
    const float r = mac_exact(raw, c1, c0);
    return mac_exact(raw >> 31, -(c1 * 4294967296.0f), r);
  }
  return mac_exact(x, c1, c0);
}

__global__ void requantize_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                          npu_dtype ot, float c1, float c0) {
  NPU_TID;
  store_int(out, ot, i, requant_tail(requant_mac(load_int(in, it, i), it, c1, c0), ot));
}

__global__ void requantize_channelwise_kernel(int64_t n, const void* in, npu_dtype it,
                             void* out, npu_dtype ot, const float* M,
                             const float* B, int64_t channels) {
  NPU_TID;
  const int64_t c = i % channels;
  store_int(out, ot, i, requant_tail(mac_exact(load_int(in, it, i), M[c], B[c]), ot));
}

// dqq_a64.hpp DOWN: the kernel's MACs over the int32 words, in kernel order.
__global__ void requantize_a64_down_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                           npu_dtype ot, npu_a64_coeff c) {
  NPU_TID;
  const uint64_t x = static_cast<const uint64_t*>(in)[i];
  const int32_t lo = int32_t(uint32_t(x)), hi = int32_t(uint32_t(x >> 32));
  float acc = mac_exact(hi, c.mult_f_x2p32, c.bias_f);
  acc = mac_exact(lo, c.mult_f, acc);
  if (it == NPU_I64) {
    acc = mac_exact(lo & INT32_MIN, -c.mult_f_x2p32 * 0x1p-31f, acc);
  } else {
    acc = mac_exact(lo < 0 ? -1 : 0, -c.mult_f_x2p32, acc);
    acc = mac_exact(hi < 0 ? -1 : 0, -c.mult_f_x2p64, acc);
  }
  int64_t lo_r, hi_r;
  int_range(ot, &lo_r, &hi_r);  // 32-bit outputs saturate to their full range
  store_int(out, ot, i, sat_rne(acc, lo_r, hi_r));
}

// dqq_a64.hpp UP: R = round_half_even((x*M + B) / 2^s) in wrapping int64.
// Unsigned inputs multiply as uint32 x uint32.
__global__ void requantize_a64_up_kernel(int64_t n, const void* in, npu_dtype it, void* out,
                         npu_dtype ot, npu_a64_coeff c) {
  NPU_TID;
  const int64_t x = load_int(in, it, i);
  const bool uin = it == NPU_U8 || it == NPU_U16 || it == NPU_U32;
  const uint64_t prod = uin ? uint64_t(uint32_t(x)) * uint64_t(uint32_t(c.M))
                            : uint64_t(x * int64_t(c.M));
  const int64_t acc = int64_t(uint64_t(c.B) + prod);
  int64_t r = acc;
  if (c.s > 0) {
    r = acc >> c.s;  // arithmetic: floor
    const uint64_t rem = uint64_t(acc) & ((1ull << c.s) - 1);
    const uint64_t half = 1ull << (c.s - 1);
    if (rem > half || (rem == half && (r & 1))) ++r;
  }
  if (ot == NPU_U64 && r < 0) r = 0;
  static_cast<int64_t*>(out)[i] = r;
}

__global__ void cast_int32_int64_kernel(int64_t n, const void* in, npu_dtype it,
                              void* out, npu_dtype ot, float c1, float c0) {
  NPU_TID;
  const int64_t r =
      sat_rne(mac_exact(load_int(in, it, i), c1, c0), INT32_MIN, INT32_MAX);
  if (ot == NPU_I64)
    static_cast<int64_t*>(out)[i] = r;
  else
    static_cast<int32_t*>(out)[i] = int32_t(r);
}

// ---- F. quantize then dequantize --------------------------------------------

__global__ void bfp16_qdq_to_f32_kernel(int64_t n, const void* in, float* out,
                                float mul, float zp, float scale) {
  NPU_TID;
  const int64_t q = bfp16_to_unsigned(in, i, mul, zp, 65535);
  out[i] = float(int32_t(q) - int32_t(zp)) * scale;
}

}  // namespace

extern "C" {

// A ---------------------------------------------------------------------------

int npu_convert_fp32_fp16(void* stream, const void* in, void* out,
                          npu_dtype out_type, int64_t n) {
  if (out_type != NPU_F16 && out_type != NPU_BF16) return -1;
  return launch(stream, n, convert_float_kernel, in, NPU_F32, out, out_type, kEven);
}

int npu_convert_fp16_fp32(void* stream, const void* in, npu_dtype in_type,
                          void* out, int64_t n) {
  if (in_type != NPU_F16 && in_type != NPU_BF16) return -1;
  return launch(stream, n, convert_float_kernel, in, in_type, out, NPU_F32, kEven);
}

int npu_convert_fp16_bf16(void* stream, const void* in, npu_dtype in_type,
                          void* out, npu_dtype out_type, int64_t n) {
  const bool ok = (in_type == NPU_F16 && out_type == NPU_BF16) ||
                  (in_type == NPU_BF16 && out_type == NPU_F16);
  if (!ok) return -1;
  return launch(stream, n, convert_float_kernel, in, in_type, out, out_type, kEven);
}

int npu_cast_bf2bfp16(void* stream, const void* in, void* out, int64_t n) {
  if (n % 8) return -1;
  return launch(stream, n / 8, float_to_bfp16_kernel, in, NPU_BF16, out, kEven);
}

int npu_convert_bf16_to_bfp16(void* stream, const void* in, npu_dtype in_type,
                              void* out, int64_t n) {
  if (n % 8 || (in_type != NPU_BF16 && in_type != NPU_F16)) return -1;
  return launch(stream, n / 8, float_to_bfp16_kernel, in, in_type, out, kAway);
}

int npu_cast_bfp16tobf(void* stream, const void* in, void* out, int64_t n) {
  if (n % 8) return -1;
  return launch(stream, n, bfp16_to_bf16_kernel, in, out);
}

int npu_cast_bftomx6(void* stream, const void* in, void* out, int64_t n) {
  if (n % 16) return -1;
  return launch(stream, n / 16, bf16_to_mx6_kernel, in, out, kAway);
}

int npu_cast_mx6tobf(void* stream, const void* in, void* out, int64_t n) {
  if (n % 16) return -1;
  return launch(stream, n, mx6_to_bf16_kernel, in, out);
}

// B ---------------------------------------------------------------------------

int npu_convert_fp32_fp8(void* stream, const void* in, void* out, int64_t n,
                         float scale_recip) {
  return launch(stream, n, float_to_fp8_kernel, in, NPU_F32, out, scale_recip, false, kAway);
}

int npu_convert_fp16_fp8(void* stream, const void* in, npu_dtype in_type,
                         void* out, int64_t n, float scale) {
  if (in_type != NPU_F16 && in_type != NPU_BF16) return -1;
  // (T)(1.0f / scale): the reciprocal is rounded to the input type (half to
  // even), so the product with an f16 / bf16 input is exact in fp32.
  const float recip = 1.0f / scale;
  uint32_t b;
  __builtin_memcpy(&b, &recip, 4);
  float r;
  if (in_type == NPU_BF16) {
    const uint32_t lsb = (b >> 16) & 1;
    const uint32_t rb = ((b + 0x7fff + lsb) >> 16) << 16;
    __builtin_memcpy(&r, &rb, 4);
  } else {
    r = float(_Float16(recip));  // host conversion is round half to even
  }
  return launch(stream, n, float_to_fp8_kernel, in, in_type, out, r, true, kEven);
}

int npu_convert_fp8_fp32(void* stream, const void* in, void* out, int64_t n,
                         float scale) {
  return launch(stream, n, fp8_to_float_kernel, in, out, NPU_F32, scale, kEven);
}

int npu_convert_fp8_fp16(void* stream, const void* in, void* out,
                         npu_dtype out_type, int64_t n, float scale) {
  if (out_type != NPU_F16 && out_type != NPU_BF16) return -1;
  return launch(stream, n, fp8_to_float_kernel, in, out, out_type, scale, kAway);
}

// C ---------------------------------------------------------------------------

int npu_quant(void* stream, const void* in, npu_dtype in_type, void* out,
              npu_dtype out_type, int64_t n, float scale, float zero_point) {
  if (!is_float(in_type) || !is_small_int(out_type)) return -1;
  return launch(stream, n, quantize_kernel, in, in_type, out, out_type, scale, zero_point);
}

int npu_cast_bfp16toaq(void* stream, const void* in, void* out,
                       npu_dtype out_type, int64_t n, float quant_multiplier,
                       float zero_point) {
  if (n % 8 || (out_type != NPU_U8 && out_type != NPU_U16)) return -1;
  return launch(stream, n, bfp16_quantize_kernel, in, out, out_type, quant_multiplier,
                zero_point);
}

// D ---------------------------------------------------------------------------

int npu_dequant(void* stream, const void* in, npu_dtype in_type, void* out,
                npu_dtype out_type, int64_t n, float scale, float zero_point) {
  if (!is_small_int(in_type) || !is_float(out_type)) return -1;
  const float c = -zero_point * scale;  // aie::mul(neg(zp), scale): one rounding
  return launch(stream, n, dequantize_kernel, in, in_type, out, out_type, scale, c);
}

int npu_cast_aqtobfp16(void* stream, const void* in, npu_dtype in_type,
                       void* out, int64_t n, float scale, float zero_point) {
  if (n % 8 || (in_type != NPU_U8 && in_type != NPU_U16)) return -1;
  return launch(stream, n / 8, dequantize_to_bfp16_kernel, in, in_type, out, scale,
                int32_t(zero_point));
}

int npu_dequant_w4_bfp16(void* stream, const void* q, const void* zp,
                         const void* scale, void* out, int64_t N, int64_t K) {
  if (K % 32) return -1;
  return launch(stream, N * K / 8, dequantize_w4_to_bfp16_kernel, static_cast<const uint8_t*>(q),
                static_cast<const uint8_t*>(zp),
                static_cast<const uint16_t*>(scale), out, K);
}

// E ---------------------------------------------------------------------------

int npu_requantize(void* stream, const void* in, npu_dtype in_type, void* out,
                   npu_dtype out_type, int64_t n, float c1, float c0) {
  if (!(is_small_int(in_type) || is_32(in_type)) ||
      !(is_small_int(out_type) || is_32(out_type)))
    return -1;
  return launch(stream, n, requantize_kernel, in, in_type, out, out_type, c1, c0);
}

int npu_requantize_channelwise(void* stream, const void* in, npu_dtype in_type,
                               void* out, npu_dtype out_type, int64_t n,
                               const float* M, const float* B,
                               int64_t channels) {
  if (!is_small_int(in_type) || !(is_small_int(out_type) || is_32(out_type)) ||
      channels <= 0)
    return -1;
  return launch(stream, n, requantize_channelwise_kernel, in, in_type, out, out_type, M, B, channels);
}

int npu_requantize_a64(void* stream, const void* in, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t n,
                       const npu_a64_coeff* coeff) {
  if (is_64(in_type) == is_64(out_type)) return -1;
  if (is_64(in_type)) {
    if (!(is_small_int(out_type) || is_32(out_type))) return -1;
    return launch(stream, n, requantize_a64_down_kernel, in, in_type, out, out_type, *coeff);
  }
  if (!(is_small_int(in_type) || is_32(in_type))) return -1;
  return launch(stream, n, requantize_a64_up_kernel, in, in_type, out, out_type, *coeff);
}

int npu_cast_int32_int64(void* stream, const void* in, npu_dtype in_type,
                         void* out, npu_dtype out_type, int64_t n, float c1,
                         float c0) {
  if (!is_small_int(in_type) || (out_type != NPU_I32 && out_type != NPU_I64))
    return -1;
  return launch(stream, n, cast_int32_int64_kernel, in, in_type, out, out_type, c1, c0);
}

// F ---------------------------------------------------------------------------

int npu_cast_bfp16toqdqf32(void* stream, const void* in, void* out, int64_t n,
                           float quant_multiplier, float zero_point,
                           float scale) {
  if (n % 8) return -1;
  return launch(stream, n, bfp16_qdq_to_f32_kernel, in, static_cast<float*>(out),
                quant_multiplier, zero_point, scale);
}

}  // extern "C"
