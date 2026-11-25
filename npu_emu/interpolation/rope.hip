/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for rope.h. One thread per output element, or per
 * block of 8 for bfp16.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "common/block_fp.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/mmac.h"
#include "common/quant.h"
#include "interpolation/rope.h"

using namespace npu_emu;

namespace {

// One rotated output from its pair (xn, xr) and (c, s). `second` selects
// out_r = xn*s + xr*c over out_n = xn*c - xr*s.
__device__ inline float rotate(float xn, float xr, float c, float s, bool second, bool fused) {
  if (fused)
    return second ? fma_q(xr, c, mul_q(xn, s)) : msc_q(xr, s, mul_q(xn, c));
  return second ? add_q(mul_q(xn, s), mul_q(xr, c)) : sub_q(mul_q(xn, c), mul_q(xr, s));
}

// Shape of a rotate_half row: rot = rotated prefix, half = rot / 2.
struct RopeShape {
  int64_t seq, K, rot, half;
};

// For element k of a row at sequence position m: the indices of its pair and
// its cos / sin entry, or false for the pass-through tail of a partial RoPE.
__device__ inline bool pair_of(const RopeShape& g, int64_t k, int64_t* jn, int64_t* jr,
                               bool* second) {
  if (k >= g.rot) return false;
  *second = k >= g.half;
  *jn = *second ? k - g.half : k;
  *jr = *jn + g.half;
  return true;
}

__global__ void rope_16_kernel(int64_t n, const void* x, npu_dtype t, const float* cs,
                               const float* sn, uint16_t* out, RopeShape g) {
  NPU_TID;
  const int64_t k = i % g.K, row = i - k, m = (i / g.K) % g.seq;
  int64_t jn, jr;
  bool second;
  if (!pair_of(g, k, &jn, &jr, &second)) {
    out[i] = static_cast<const uint16_t*>(x)[i];  // tail: copied
    return;
  }
  const float xn = load_float(x, t, row + jn), xr = load_float(x, t, row + jr);
  const float c = cs[m * g.half + jn], s = sn[m * g.half + jn];
  out[i] = uint16_t(encode_float(rotate(xn, xr, c, s, second, true), t, kHalfEven));
}

__global__ void rope_fp8_kernel(int64_t n, const uint8_t* x, npu_dtype t, const float* cs,
                                const float* sn, uint8_t* out, RopeShape g, float ifm_scale,
                                float recip) {
  NPU_TID;
  const int64_t k = i % g.K, m = (i / g.K) % g.seq;
  const uint8_t* row = x + (i - k);
  int64_t jn, jr;
  bool second;
  float y;
  if (!pair_of(g, k, &jn, &jr, &second)) {
    y = fp8_in(row, k, g.K, ifm_scale, t, kHalfEven);
  } else {
    const float xn = fp8_in(row, jn, g.K, ifm_scale, t, kHalfEven);
    const float xr = fp8_in(row, jr, g.K, ifm_scale, t, kHalfEven);
    const float c = cs[m * g.half + jn], s = sn[m * g.half + jn];
    y = round_float(rotate(xn, xr, c, s, second, true), t, kHalfEven);
  }
  // The product is exact.
  out[i] = fp8_out(y, recip, kHalfEven);
}

struct Bfp16Args {
  int64_t seq, K;
  npu_dtype wt;
  bool interleave, fused;
};

// One output block of 8 elements per thread: the bf16-rounded rotation of
// each element, then the bfp16 encode of the block. The input decode aligns
// each 64-group of a row's blocks to its largest exponent.
__global__ void rope_bfp16_kernel(int64_t n, const uint8_t* x, const void* cs, const void* sn,
                                  uint8_t* out, Bfp16Args g) {
  NPU_TID;
  const int64_t bpr = g.K / 8;  // blocks per row
  const int64_t r = i / bpr, k0 = (i % bpr) * 8, m = r % g.seq;
  const uint8_t* row = x + r * bpr * 9;
  auto in = [&](int64_t k) { return decode_bfp16_group(row, k / 8, bpr, int(k % 8)); };
  float v[8];
  for (int e = 0; e < 8; ++e) {
    const int64_t k = k0 + e;
    int64_t jn, jr, p;
    bool second;
    if (g.interleave) {
      p = k / 2;
      jn = 2 * p;
      jr = jn + 1;
      second = k & 1;
    } else {
      const int64_t half = g.K / 2;
      second = k >= half;
      p = second ? k - half : k;
      jn = p;
      jr = p + half;
    }
    const float c = load_float(cs, g.wt, m * (g.K / 2) + p);
    const float s = load_float(sn, g.wt, m * (g.K / 2) + p);
    v[e] = round_float(rotate(in(jn), in(jr), c, s, second, g.fused), NPU_BF16, kHalfEven);
  }
  encode_bfp16(v, kHalfEven, out + i * 9);
}

// rotary_emb_dim -> rotated prefix length, or -1 if unsupported.
int64_t rotary_dim(int rotary_emb_dim, int64_t K) {
  if (rotary_emb_dim == 0 || rotary_emb_dim == K) return K;
  if (rotary_emb_dim < 0 || rotary_emb_dim > K || rotary_emb_dim % 32 || K % 128) return -1;
  return rotary_emb_dim;
}

}  // namespace

extern "C" {

int npu_rope_bf16_stream(void* stream, const void* x, npu_dtype x_type, const void* cos,
                         const void* sin, void* out, int64_t rows, int64_t seq_len,
                         int64_t head_dim, int rotary_emb_dim) {
  if (!is_half_float(x_type) || any_null(x, cos, sin, out) || rows < 0 || seq_len < 0 ||
      head_dim <= 0 || head_dim % 4)
    return -1;
  const int64_t rot = rotary_dim(rotary_emb_dim, head_dim);
  if (rot < 0) return -1;
  if (rows == 0 || seq_len == 0) return 0;
  const RopeShape g{seq_len, head_dim, rot, rot / 2};
  return launch(stream, rows * seq_len * head_dim, rope_16_kernel, x, x_type,
                static_cast<const float*>(cos), static_cast<const float*>(sin),
                static_cast<uint16_t*>(out), g);
}

int npu_rope_fp8_stream(void* stream, const void* x, const void* cos, const void* sin,
                        void* out, int64_t rows, int64_t seq_len, int64_t head_dim,
                        npu_dtype compute_type, int rotary_emb_dim, float ifm_scale,
                        float ofm_scale) {
  if (!is_half_float(compute_type) || any_null(x, cos, sin, out) || rows < 0 || seq_len < 0 ||
      head_dim <= 0 || head_dim % 64)
    return -1;
  const int64_t rot = rotary_dim(rotary_emb_dim, head_dim);
  if (rot < 0) return -1;
  if (rows == 0 || seq_len == 0) return 0;
  const RopeShape g{seq_len, head_dim, rot, rot / 2};
  return launch(stream, rows * seq_len * head_dim, rope_fp8_kernel,
                static_cast<const uint8_t*>(x), compute_type, static_cast<const float*>(cos),
                static_cast<const float*>(sin), static_cast<uint8_t*>(out), g, ifm_scale,
                scalar_cast(div_q(1.0f, ofm_scale), compute_type, kHalfEven));
}

int npu_rope_bfp16_stream(void* stream, const void* x, const void* cos, const void* sin,
                          npu_dtype wts_type, void* out, int64_t rows, int64_t seq_len,
                          int64_t head_dim, int interleave, int rotary_emb_dim) {
  if ((wts_type != NPU_BF16 && wts_type != NPU_F32) || any_null(x, cos, sin, out) ||
      rows < 0 || seq_len < 0 || head_dim <= 0 || head_dim % 128 ||
      (interleave != 0 && interleave != 1) || rotary_emb_dim != 0)
    return -1;
  if (rows == 0 || seq_len == 0) return 0;
  // Fused only for rotate_half with head_dim > 128.
  const Bfp16Args g{seq_len, head_dim, wts_type, interleave == 1,
                    interleave == 0 && head_dim > 128};
  return launch(stream, rows * seq_len * head_dim / 8, rope_bfp16_kernel,
                static_cast<const uint8_t*>(x), cos, sin, static_cast<uint8_t*>(out), g);
}

}  // extern "C"
