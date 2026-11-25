/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and C entry points for convert.h. One thread per element, or per
 * block for block floating point.
 */
#include <hip/hip_runtime.h>

#include "convert/convert.h"
#include "common/bits.h"
#include "common/block_fp.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/mmac.h"
#include "common/quant.h"

using namespace npu_emu;

namespace {

__global__ void convert_float_kernel(int64_t n, const void* in, npu_dtype in_type,
                                     void* out, npu_dtype out_type, Rnd rnd) {
  NPU_TID;
  store_float(out, out_type, i, load_float(in, in_type, i), rnd);
}

__global__ void float_to_bfp16_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                                      Rnd rnd) {
  NPU_TID;
  float v[8];
  for (int k = 0; k < 8; ++k) v[k] = load_float(in, in_type, 8 * i + k);
  encode_bfp16(v, rnd, static_cast<uint8_t*>(out) + 9 * i);
}

// The decode aligns each 64-group of blocks from the buffer start to its largest exponent.
__global__ void bfp16_to_bf16_kernel(int64_t n, const void* in, void* out) {
  NPU_TID;
  static_cast<uint16_t*>(out)[i] =
      decode_bfp16_bf16(static_cast<const uint8_t*>(in), i / 8, n / 8, int(i % 8));
}

__global__ void bf16_to_mx6_kernel(int64_t n, const void* in, void* out, Rnd rnd) {
  NPU_TID;
  float v[16];
  for (int k = 0; k < 16; ++k) v[k] = load_float(in, NPU_BF16, 16 * i + k);
  encode_mx6(v, rnd, static_cast<uint8_t*>(out) + 12 * i);
}

// The mx6 decode is exact here; the kernel decodes through an identity MMAC whose upper K
// lanes are left undefined, so only results that do not depend on them are modelled.
__global__ void mx6_to_bf16_kernel(int64_t n, const void* in, void* out) {
  NPU_TID;
  const uint8_t* blk = static_cast<const uint8_t*>(in) + 12 * (i / 16);
  static_cast<uint16_t*>(out)[i] = f32_to_bf16(decode_mx6(blk, int(i % 16)), kHalfEven);
}

__global__ void float_to_fp8_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                                    float mul, bool clamp, Rnd rnd) {
  NPU_TID;
  const float v = load_float(in, in_type, i);
  static_cast<uint8_t*>(out)[i] = fp8_out(v, mul, rnd, clamp ? kSat : 0);
}

__global__ void fp8_to_float_kernel(int64_t n, const void* in, void* out, npu_dtype out_type,
                                    float scale, Rnd rnd) {
  NPU_TID;
  const float v = fp8_in(static_cast<const uint8_t*>(in), i, n, scale, out_type, rnd);
  store_float(out, out_type, i, v, rnd);
}

__global__ void fp8_requantize_kernel(int64_t n, const uint8_t* in, uint8_t* out, float coeff) {
  NPU_TID;
  out[i] = fp8_out(fp8_decode_identity(in, i, n), coeff, kHalfEven);
}

__global__ void quantize_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                                npu_dtype out_type, float scale, float zp) {
  NPU_TID;
  store_int(out, out_type, i, quant(load_float(in, in_type, i), scale, zp, out_type));
}

// bfp16 -> bf16 (as bfp16_to_bf16_kernel) -> unsigned integer; NaN -> 0.
__device__ inline int64_t bfp16_to_unsigned(const void* in, int64_t i, int64_t n,
                                            float mul, float zp, npu_dtype out_type) {
  const float x =
      bf16_to_f32(decode_bfp16_bf16(static_cast<const uint8_t*>(in), i / 8, n / 8, int(i % 8)));
  return quant(x, mul, zp, out_type);
}

__global__ void bfp16_quantize_kernel(int64_t n, const void* in, void* out,
                                      npu_dtype out_type, float mul, float zp) {
  NPU_TID;
  store_int(out, out_type, i, bfp16_to_unsigned(in, i, n, mul, zp, out_type));
}

__global__ void dequantize_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                                  npu_dtype out_type, float scale, float c) {
  NPU_TID;
  store_float(out, out_type, i, dequant(load_int(in, in_type, i), scale, c), kHalfEven);
}

__global__ void dequantize_to_bfp16_kernel(int64_t n, const void* in, npu_dtype in_type,
                                           void* out, float scale, int32_t zpi) {
  NPU_TID;
  float v[8];
  for (int k = 0; k < 8; ++k) {
    const float d = mul_q(float(load_int(in, in_type, 8 * i + k) - zpi), scale);
    v[k] = round_float(d, NPU_BF16, kHalfEven);
  }
  encode_bfp16(v, kHalfEven, static_cast<uint8_t*>(out) + 9 * i);
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
    v[k] = mul_q(float(int(q[row * K + col + k]) - int(zp[g])), s);
  encode_bfp16(v, kHalfEven, static_cast<uint8_t*>(out) + 9 * i);
}

// A U32 input is loaded as int32 and corrected with (x >> 31) * -(c1 * 2^32).
__device__ inline float requant_mac(int64_t x, npu_dtype in_type, float c1, float c0) {
  if (in_type == NPU_U32) {
    const int32_t raw = int32_t(uint32_t(x));
    const float r = mac_exact(raw, c1, c0);
    return mac_exact(raw >> 31, negmul_q(c1, 4294967296.0f), r);
  }
  return mac_exact(x, c1, c0);
}

__global__ void requantize_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                                  npu_dtype out_type, float c1, float c0) {
  NPU_TID;
  const float v = requant_mac(load_int(in, in_type, i), in_type, c1, c0);
  store_int(out, out_type, i, requant_tail(v, out_type));
}

__global__ void requantize_channelwise_kernel(int64_t n, const void* in, npu_dtype in_type,
                                              void* out, npu_dtype out_type, const float* M,
                                              const float* B, int64_t channels) {
  NPU_TID;
  const int64_t c = i % channels;
  const float v = mac_exact(load_int(in, in_type, i), M[c], B[c]);
  store_int(out, out_type, i, requant_tail(v, out_type));
}

// 64 -> narrower: fp32 MACs over the int32 words, in this order.
__global__ void requantize_a64_down_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                                           npu_dtype out_type, npu_a64_coeff c) {
  NPU_TID;
  const uint64_t x = static_cast<const uint64_t*>(in)[i];
  const int32_t lo = int32_t(uint32_t(x)), hi = int32_t(uint32_t(x >> 32));
  float acc = mac_exact(hi, c.mult_f_x2p32, c.bias_f);
  acc = mac_exact(lo, c.mult_f, acc);
  if (in_type == NPU_I64) {
    acc = mac_exact(lo & INT32_MIN, mul_q(neg_f32(c.mult_f_x2p32), 0x1p-31f), acc);
  } else {
    acc = mac_exact(lo < 0 ? -1 : 0, neg_f32(c.mult_f_x2p32), acc);
    acc = mac_exact(hi < 0 ? -1 : 0, neg_f32(c.mult_f_x2p64), acc);
  }
  store_int(out, out_type, i, float_to_int(acc, out_type, 0, kHalfEven));  // 32-bit: full range
}

// Narrower -> 64: round_half_even((x * M + B) / 2^s) in wrapping int64.
// Unsigned inputs multiply as uint32 x uint32.
__global__ void requantize_a64_up_kernel(int64_t n, const void* in, npu_dtype in_type, void* out,
                                         npu_dtype out_type, npu_a64_coeff c) {
  NPU_TID;
  const int64_t x = load_int(in, in_type, i);
  const bool uin = in_type == NPU_U8 || in_type == NPU_U16 || in_type == NPU_U32;
  const uint64_t prod = uin ? uint64_t(uint32_t(x)) * uint64_t(uint32_t(c.M))
                            : uint64_t(x * int64_t(c.M));
  const int64_t acc = int64_t(uint64_t(c.B) + prod);
  int64_t r = c.s > 0 ? shift_round(acc, c.s, kHalfEven) : acc;
  if (out_type == NPU_U64 && r < 0) r = 0;
  static_cast<int64_t*>(out)[i] = r;
}

__global__ void cast_int32_int64_kernel(int64_t n, const void* in, npu_dtype in_type,
                                        void* out, npu_dtype out_type, float c1, float c0) {
  NPU_TID;
  const int64_t r = requant(load_int(in, in_type, i), c1, c0, NPU_I32);
  if (out_type == NPU_I64)
    static_cast<int64_t*>(out)[i] = r;
  else
    static_cast<int32_t*>(out)[i] = int32_t(r);
}

__global__ void bfp16_qdq_to_f32_kernel(int64_t n, const void* in, float* out,
                                        float mul, float zp, float scale) {
  NPU_TID;
  const int64_t q = bfp16_to_unsigned(in, i, n, mul, zp, NPU_U16);
  out[i] = mul_q(float(q - int64_t(scalar_cast_i32(zp))), scale);
}

int cast_bfp16_to_unsigned(void* stream, const void* in, void* out, npu_dtype out_type, int64_t n,
                           float quant_multiplier, float zero_point) {
  if (any_null(in, out) || n < 0 || n % 64) return -1;
  return launch(stream, n, bfp16_quantize_kernel, in, out, out_type, quant_multiplier,
                zero_point);
}

int cast_unsigned_to_bfp16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t n,
                           float scale, float zero_point) {
  if (any_null(in, out) || n < 0 || n % 64) return -1;
  return launch(stream, n / 8, dequantize_to_bfp16_kernel, in, in_type, out, scale,
                scalar_cast_i32(zero_point));
}

// The kernel reads the coefficients in 32-channel blocks, broadcast for fewer channels, and
// restarts at block 0 every period, so only these channel counts map to c = i % channels.
int requantize_channelwise(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t n, int64_t channels, const void* M,
                           const void* B) {
  const bool ok_c = channels == 4 || channels == 8 || channels == 16 ||
                    (channels > 0 && channels <= 65535 && channels % 32 == 0);
  if (any_null(in, out, M, B) || n < 0 || !ok_c || n % (channels < 32 ? 32 : channels) ||
      !(is_small_int(out_type) || is_32(out_type)))
    return -1;
  return launch(stream, n, requantize_channelwise_kernel, in, in_type, out, out_type,
                static_cast<const float*>(M), static_cast<const float*>(B), channels);
}

// Shared checks of the plain elementwise entries. n == 0 is valid: launch() returns 0 for an
// empty grid, after every other check.
inline bool bad(const void* in, const void* out, int64_t n) {
  return any_null(in, out) || n < 0;
}

}  // namespace

extern "C" {

int npu_convert_fp32_fp16(void* stream, const void* in, void* out,
                          npu_dtype out_type, int64_t n) {
  if (bad(in, out, n) || !is_half_float(out_type)) return -1;
  return launch(stream, n, convert_float_kernel, in, NPU_F32, out, out_type, kHalfEven);
}

int npu_convert_fp16_fp32(void* stream, const void* in, npu_dtype in_type,
                          void* out, int64_t n) {
  if (bad(in, out, n) || !is_half_float(in_type)) return -1;
  return launch(stream, n, convert_float_kernel, in, in_type, out, NPU_F32, kHalfEven);
}

int npu_convert_fp16_bf16(void* stream, const void* in, npu_dtype in_type,
                          void* out, npu_dtype out_type, int64_t n) {
  const bool ok = (in_type == NPU_F16 && out_type == NPU_BF16) ||
                  (in_type == NPU_BF16 && out_type == NPU_F16);
  if (bad(in, out, n) || !ok) return -1;
  return launch(stream, n, convert_float_kernel, in, in_type, out, out_type, kHalfEven);
}

int npu_cast_bf2bfp16(void* stream, const void* in, void* out, int64_t n) {
  if (bad(in, out, n) || n % 64) return -1;
  return launch(stream, n / 8, float_to_bfp16_kernel, in, NPU_BF16, out, kHalfEven);
}

int npu_convert_bf16_to_bfp16(void* stream, const void* in, npu_dtype in_type,
                              void* out, int64_t n) {
  if (bad(in, out, n) || n % 8 || !is_half_float(in_type)) return -1;
  return launch(stream, n / 8, float_to_bfp16_kernel, in, in_type, out, kHalfAway);
}

int npu_cast_bfp16tobf(void* stream, const void* in, void* out, int64_t n) {
  if (bad(in, out, n) || n % 64) return -1;
  return launch(stream, n, bfp16_to_bf16_kernel, in, out);
}

int npu_cast_bftomx6(void* stream, const void* in, void* out, int64_t n) {
  if (bad(in, out, n) || n % 64) return -1;
  return launch(stream, n / 16, bf16_to_mx6_kernel, in, out, kHalfAway);
}

int npu_cast_mx6tobf(void* stream, const void* in, void* out, int64_t n) {
  if (bad(in, out, n) || n % 256) return -1;
  return launch(stream, n, mx6_to_bf16_kernel, in, out);
}

int npu_convert_fp32_fp8(void* stream, const void* in, void* out, int64_t n,
                         float scale_recip) {
  if (bad(in, out, n)) return -1;
  return launch(stream, n, float_to_fp8_kernel, in, NPU_F32, out, scale_recip, false, kHalfAway);
}

int npu_convert_fp16_fp8(void* stream, const void* in, npu_dtype in_type,
                         void* out, int64_t n, float scale) {
  if (bad(in, out, n) || !is_half_float(in_type)) return -1;
  // The reciprocal is rounded to the input type (half to even), so the product
  // with an f16 / bf16 input is exact in fp32.
  const float r = scalar_cast(div_q(1.0f, scale), in_type, kHalfEven);
  return launch(stream, n, float_to_fp8_kernel, in, in_type, out, r, true, kHalfEven);
}

int npu_convert_fp8_fp32(void* stream, const void* in, void* out, int64_t n,
                         float scale) {
  if (bad(in, out, n)) return -1;
  return launch(stream, n, fp8_to_float_kernel, in, out, NPU_F32, scale, kHalfEven);
}

int npu_convert_fp8_fp16(void* stream, const void* in, void* out,
                         npu_dtype out_type, int64_t n, float scale) {
  if (bad(in, out, n) || !is_half_float(out_type)) return -1;
  return launch(stream, n, fp8_to_float_kernel, in, out, out_type, scale, kHalfAway);
}

int npu_requantize_fp8(void* stream, const void* in, void* out, int64_t n,
                       float ifm_coeff) {
  if (bad(in, out, n)) return -1;
  return launch(stream, n, fp8_requantize_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), ifm_coeff);
}

int npu_quant(void* stream, const void* in, npu_dtype in_type, void* out,
              npu_dtype out_type, int64_t n, float zero_point, float scale) {
  if (bad(in, out, n) || !is_plain_float(in_type) || !is_small_int(out_type)) return -1;
  return launch(stream, n, quantize_kernel, in, in_type, out, out_type, scale, zero_point);
}

int npu_cast_bfp16toa8(void* stream, const void* in, void* out, int64_t n,
                       float quant_multiplier, float zero_point) {
  return cast_bfp16_to_unsigned(stream, in, out, NPU_U8, n, quant_multiplier, zero_point);
}

int npu_cast_bfp16toa16(void* stream, const void* in, void* out, int64_t n,
                        float quant_multiplier, float zero_point) {
  return cast_bfp16_to_unsigned(stream, in, out, NPU_U16, n, quant_multiplier, zero_point);
}

int npu_dequant(void* stream, const void* in, npu_dtype in_type, void* out,
                npu_dtype out_type, int64_t n, float zero_point, float scale) {
  if (bad(in, out, n) || !is_small_int(in_type) || !is_plain_float(out_type)) return -1;
  return launch(stream, n, dequantize_kernel, in, in_type, out, out_type, scale,
                dq_offset(zero_point, scale));
}

int npu_cast_a8tobfp16(void* stream, const void* in, void* out, int64_t n, float scale,
                       float zero_point) {
  return cast_unsigned_to_bfp16(stream, in, NPU_U8, out, n, scale, zero_point);
}

int npu_cast_a16tobfp16(void* stream, const void* in, void* out, int64_t n, float scale,
                        float zero_point) {
  return cast_unsigned_to_bfp16(stream, in, NPU_U16, out, n, scale, zero_point);
}

int npu_dequant_w4_bfp16(void* stream, const void* zp, const void* scale,
                         const void* q, void* out, int64_t N, int64_t K) {
  if (any_null(zp, scale, q, out) || N < 0 || K < 0 || N % 64 || K % 64) return -1;
  return launch(stream, N * K / 8, dequantize_w4_to_bfp16_kernel, static_cast<const uint8_t*>(q),
                static_cast<const uint8_t*>(zp), static_cast<const uint16_t*>(scale), out, K);
}

int npu_requantize(void* stream, const void* in, npu_dtype in_type, void* out,
                   npu_dtype out_type, int64_t n, float c1, float c0) {
  if (bad(in, out, n) || !(is_small_int(in_type) || is_32(in_type)) ||
      !(is_small_int(out_type) || is_32(out_type)) || (is_32(in_type) && is_32(out_type)))
    return -1;
  return launch(stream, n, requantize_kernel, in, in_type, out, out_type, c1, c0);
}

int npu_requantize_a8_channelwise(void* stream, const void* in, npu_dtype in_type,
                                  void* out, npu_dtype out_type, int64_t n,
                                  int64_t channels, const void* M, const void* B) {
  if (!is_i8(in_type)) return -1;
  return requantize_channelwise(stream, in, in_type, out, out_type, n, channels, M, B);
}

int npu_requantize_a16_channelwise(void* stream, const void* in, npu_dtype in_type,
                                   void* out, npu_dtype out_type, int64_t n,
                                   int64_t channels, const void* M, const void* B) {
  if (!is_i16(in_type)) return -1;
  return requantize_channelwise(stream, in, in_type, out, out_type, n, channels, M, B);
}

int npu_requantize_a64(void* stream, const void* in, npu_dtype in_type,
                       void* out, npu_dtype out_type, int64_t n,
                       const npu_a64_coeff* coeff) {
  if (bad(in, out, n) || !coeff || is_64(in_type) == is_64(out_type)) return -1;
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
  if (bad(in, out, n) || !is_small_int(in_type) || (out_type != NPU_I32 && out_type != NPU_I64))
    return -1;
  return launch(stream, n, cast_int32_int64_kernel, in, in_type, out, out_type, c1, c0);
}

// The kernel decodes two 64-value groups per step and has no odd-group tail.
int npu_cast_bfp16toqdqf32(void* stream, const void* in, void* out, int64_t n,
                           float quant_multiplier, float zero_point,
                           float scale) {
  if (bad(in, out, n) || n % 128) return -1;
  return launch(stream, n, bfp16_qdq_to_f32_kernel, in, static_cast<float*>(out),
                quant_multiplier, zero_point, scale);
}

}  // extern "C"
