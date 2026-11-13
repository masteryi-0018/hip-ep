/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for include/npu_emu_rope.h. One thread per output
 * element (bf16 / fp8) or per output block of 8 (bfp16); reference-speed
 * kernels.
 */
#include <hip/hip_runtime.h>

#include "npu_emu.h"
#include "npu_emu_device.h"
#include "npu_emu_launch.h"
#include "npu_emu_rope.h"

using namespace npu_emu;

// Every fp32 operation is rounded separately; the fused multiply-adds the NPU
// does (aie::msc / aie::mac on an accumulator) are written out as fmaf.
#pragma clang fp contract(off)

namespace {

// Round half to even to a 2-byte float type and widen back (exact).
__device__ inline float rne16(float v, npu_dtype t) {
  return t == NPU_F16 ? f16_to_f32(f32_to_f16(v, kEven)) : bf16_to_f32(f32_to_bf16(v, kEven));
}

__device__ inline uint16_t bits16(float v, npu_dtype t) {
  return t == NPU_F16 ? f32_to_f16(v, kEven) : f32_to_bf16(v, kEven);
}

// One rotated output from its pair (xn, xr) and (c, s). `second` selects
// out_r = xn*s + xr*c over out_n = xn*c - xr*s.
//   fused:   aie::mul then aie::msc / aie::mac (core_vec_body)
//   unfused: four aie::mul then aie::sub / aie::add (rope_fma_fn)
__device__ inline float rotate(float xn, float xr, float c, float s, bool second, bool fused) {
  if (fused) return second ? fmaf(xr, c, xn * s) : fmaf(-xr, s, xn * c);
  return second ? xn * s + xr * c : xn * c - xr * s;
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

// ---- rope_bf16_stream ---------------------------------------------------------

__global__ void rope_16_kernel(int64_t n, const void* x, npu_dtype t, const float* cs,
                               const float* sn, uint16_t* out, RopeShape g) {
  NPU_TID;
  const int64_t k = i % g.K, row = i - k, m = (i / g.K) % g.seq;
  int64_t jn, jr;
  bool second;
  if (!pair_of(g, k, &jn, &jr, &second)) {
    out[i] = static_cast<const uint16_t*>(x)[i];  // tail: a plain vector copy
    return;
  }
  const float xn = load_float(x, t, row + jn), xr = load_float(x, t, row + jr);
  const float c = cs[m * g.half + jn], s = sn[m * g.half + jn];
  out[i] = bits16(rotate(xn, xr, c, s, second, true), t);
}

// ---- rope_fp8_stream ----------------------------------------------------------

// fp8 element k of a row as the kernel stages it in its scratch: decoded by
// the identity mul_8x8_8x8 (groups of 8 from the row start, NaN spreads, a sum
// of products so -0 becomes +0), times ifm_scale in fp32 (mul_elem_32), then
// rounded half to even to the internal type (to_v64bfloat16 / to_v64float16).
__device__ inline float fp8_in(const uint8_t* row, int64_t k, int64_t K, float scale,
                               npu_dtype t) {
  return rne16((fp8_to_f32_row(row, k, K) + 0.0f) * scale, t);
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
    y = fp8_in(row, k, g.K, ifm_scale, t);
  } else {
    const float xn = fp8_in(row, jn, g.K, ifm_scale, t);
    const float xr = fp8_in(row, jr, g.K, ifm_scale, t);
    const float c = cs[m * g.half + jn], s = sn[m * g.half + jn];
    y = rne16(rotate(xn, xr, c, s, second, true), t);
  }
  // mul_elem_32(y, recip) is exact (two 2-byte floats); the fp8 pack under
  // saturation_mode::saturate clamps to +-448 (NaN passes the clamp,
  // unverified on the NPU), then rounds half to even.
  float v = y * recip;
  if (v == v) v = fminf(448.0f, fmaxf(-448.0f, v));
  out[i] = f32_to_fp8(v, kEven);
}

// ---- rope_bfp16_stream --------------------------------------------------------

struct Bfp16Args {
  int64_t seq, K;
  npu_dtype wt;
  bool interleave, fused;
};

// One output block of 8 elements per thread: the bf16-rounded rotation of
// each element, then the bfp16 encode of the block.
__global__ void rope_bfp16_kernel(int64_t n, const uint8_t* x, const void* cs, const void* sn,
                                  uint8_t* out, Bfp16Args g) {
  NPU_TID;
  const int64_t bpr = g.K / 8;  // blocks per row
  const int64_t r = i / bpr, k0 = (i % bpr) * 8, m = r % g.seq;
  const uint8_t* row = x + r * bpr * 9;
  auto in = [&](int64_t k) { return decode_bfp16(row + (k / 8) * 9, int(k % 8)); };
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
    v[e] = bf16_to_f32(f32_to_bf16(rotate(in(jn), in(jr), c, s, second, g.fused), kEven));
  }
  encode_bfp16(v, kEven, out + i * 9);
}

// rotary_emb_dim -> rotated prefix length, or -1 if the kernel cannot do it
// on the logical tensor (see npu_emu_rope.h).
int64_t rotary_dim(int rotary_emb_dim, int64_t K) {
  if (rotary_emb_dim == 0 || rotary_emb_dim == K) return K;
  if (rotary_emb_dim < 0 || rotary_emb_dim > K || rotary_emb_dim % 32 || K % 128) return -1;
  return rotary_emb_dim;
}

bool is_16(npu_dtype t) { return t == NPU_BF16 || t == NPU_F16; }

// (T)(1.0f / s): the fp32 reciprocal rounded half to even to T, on the host.
float recip16(float s, npu_dtype t) {
  const float r = 1.0f / s;
  if (t == NPU_F16) return float(_Float16(r));  // host conversion is RNE
  uint32_t b;
  __builtin_memcpy(&b, &r, 4);
  if ((b & 0x7fffffffu) > 0x7f800000u)
    b = 0x7fc00000u;  // NaN scale (unverified on the NPU)
  else
    b = ((b + 0x7fff + ((b >> 16) & 1)) >> 16) << 16;  // carries into inf correctly
  float o;
  __builtin_memcpy(&o, &b, 4);
  return o;
}

}  // namespace

extern "C" {

int npu_rope_bf16_stream(void* stream, const void* x, npu_dtype type, const float* cos,
                         const float* sin, void* out, int64_t rows, int64_t seq_len,
                         int64_t head_dim, int rotary_emb_dim) {
  if (!is_16(type) || rows < 0 || seq_len <= 0 || head_dim <= 0 || head_dim % 4) return -1;
  const int64_t rot = rotary_dim(rotary_emb_dim, head_dim);
  if (rot < 0) return -1;
  const RopeShape g{seq_len, head_dim, rot, rot / 2};
  return launch(stream, rows * seq_len * head_dim, rope_16_kernel, x, type, cos, sin,
                static_cast<uint16_t*>(out), g);
}

int npu_rope_fp8_stream(void* stream, const void* x, npu_dtype type, const float* cos,
                        const float* sin, void* out, int64_t rows, int64_t seq_len,
                        int64_t head_dim, int rotary_emb_dim, float ifm_scale,
                        float ofm_scale) {
  if (!is_16(type) || rows < 0 || seq_len <= 0 || head_dim <= 0 || head_dim % 64) return -1;
  const int64_t rot = rotary_dim(rotary_emb_dim, head_dim);
  if (rot < 0) return -1;
  const RopeShape g{seq_len, head_dim, rot, rot / 2};
  return launch(stream, rows * seq_len * head_dim, rope_fp8_kernel,
                static_cast<const uint8_t*>(x), type, cos, sin, static_cast<uint8_t*>(out), g,
                ifm_scale, recip16(ofm_scale, type));
}

int npu_rope_bfp16_stream(void* stream, const void* x, const void* cos, const void* sin,
                          npu_dtype wts_type, void* out, int64_t rows, int64_t seq_len,
                          int64_t head_dim, int interleave) {
  if ((wts_type != NPU_BF16 && wts_type != NPU_F32) || rows < 0 || seq_len <= 0 ||
      head_dim <= 0 || head_dim % 128 || (interleave != 0 && interleave != 1))
    return -1;
  // Only rotate_half rows wider than one 128-element block go through the
  // scratch path with mul + msc / mac; the register paths use rope_fma_fn.
  const Bfp16Args g{seq_len, head_dim, wts_type, interleave == 1,
                    interleave == 0 && head_dim > 128};
  return launch(stream, rows * seq_len * head_dim / 8, rope_bfp16_kernel,
                static_cast<const uint8_t*>(x), cos, sin, static_cast<uint8_t*>(out), g);
}

}  // extern "C"
