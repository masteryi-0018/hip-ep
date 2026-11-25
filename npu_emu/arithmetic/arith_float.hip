/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Floating-point elementwise arithmetic, see arith_float.h. One thread per output element
 * (per 8 channels where bfp16 is involved).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "arithmetic/arith_float.h"
#include "common/act.h"
#include "common/bcast.h"
#include "common/bits.h"
#include "common/block_fp.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/mmac.h"
#include "common/nlf.h"

namespace {

using namespace npu_emu;

// Activation codes of add / mul / mac_fused.
constexpr int kActRelu = 1;
constexpr int kActRelu6 = 2;
constexpr int kActLrelu = 3;
// Activation codes of mul_add.
constexpr int kActTanh = 7;
constexpr int kActClip = 100;

__device__ inline float activate(float v, int act, float a0, float a1) {
  switch (act) {
    case kActRelu:
      return vmax(v, a0);
    case kActRelu6:
      return vclamp(v, a0, a1);
    case kActLrelu:
      return vmax(v, mul_q(v, a0));
    default:
      return v;
  }
}

struct Shape {
  int64_t H, W, C;
};

enum BinOp { kAdd, kMul, kDiv };

struct Bin16 {
  Shape s;
  int bc;
  npu_dtype t;
  BinOp op;
  int neg1, neg2, swap, act;
  float a0, a1;
};

// 16-bit reciprocal of a value carried in fp32: bf16 reads the upper 16 bits (an fp32
// subnormal reads as a zero exponent), f16 rounds half to even.
__device__ inline float inv16(float y, npu_dtype t) {
  return decode_float(hw_inv16(code16(y, t), t), t);
}

__global__ void bin16_kernel(int64_t n, const void* x1, const void* x2, void* out, Bin16 p) {
  NPU_TID;
  const int64_t j = bidx_flat(i, p.bc, p.s.W, p.s.C).flat();
  const float a = load_float(x1, p.t, i), b = load_float(x2, p.t, j);
  float v;
  if (p.op == kAdd) {
    // One rounding.
    v = add_q(p.neg1 ? neg_f32(a) : a, p.neg2 ? neg_f32(b) : b);
  } else if (p.op == kMul) {
    v = mul_q(a, b);
  } else {
    const float num = p.swap ? b : a, den = p.swap ? a : b;
    v = mul_q(num, inv16(den, p.t));
  }
  store_float(out, p.t, i, activate(v, p.act, p.a0, p.a1), kHalfEven);
}

__global__ void mac16_kernel(int64_t n, const void* x1, const void* x2, const void* x3,
                             void* out, Bin16 p) {
  NPU_TID;
  const int64_t j = bidx_flat(i, p.bc, p.s.W, p.s.C).flat();
  const float v = fma_q(load_float(x1, p.t, i), load_float(x2, p.t, j), load_float(x3, p.t, j));
  store_float(out, p.t, i, activate(v, p.act, p.a0, p.a1), kHalfEven);
}

enum BfpOp { kBfpAdd, kBfpSub, kBfpMul, kBfpDiv };

struct Bfp {
  Shape s;  // C in values (multiple of 8)
  int bc;
  BfpOp op;
  int swap;
};

// One thread per output block (8 channels of a pixel). Each pixel's channels split into
// 64-element decode groups from channel 0; a C-broadcast x2 block is decoded in the group of
// the 8 x2 blocks starting at it.
__global__ void bfp_kernel(int64_t n, const uint8_t* x1, const uint8_t* x2, uint8_t* out,
                           Bfp p) {
  NPU_TID;
  const int64_t nb = p.s.C / 8;
  const int64_t cb = i % nb;
  const BIdx j = bidx_flat(i, p.bc, p.s.W, nb);  // indices in blocks
  const int64_t g1 = i - cb + (cb & ~int64_t(7)), e1 = i - cb + nb;
  const bool bc_c = p.bc & 1;
  const int64_t b2 = j.pixel + j.c;
  const int64_t g2 = bc_c ? b2 : j.pixel + (j.c & ~int64_t(7));
  const int64_t e2 = bc_c ? (p.bc & 4 ? 1 : p.s.H) * (p.bc & 2 ? 1 : p.s.W) : j.pixel + nb;
  float v[8];
  for (int k = 0; k < 8; ++k) {
    const float a = decode_bfp16_identity(x1, g1, e1, i, k);
    const float b = decode_bfp16_identity(x2, g2, e2, b2, bc_c ? 0 : k);
    switch (p.op) {
      case kBfpAdd:
        v[k] = mmac_add_bfp16(x1, g1, e1, i, k, a, b);
        break;
      case kBfpSub:
        if (p.swap)
          v[k] = sub_q(b, a);
        else
          v[k] = bc_c ? mmac_add_bfp16(x1, g1, e1, i, k, a, neg_f32(b)) : sub_q(a, b);
        break;
      case kBfpMul:
        v[k] = mul_q(a, b);
        break;
      default: {
        // The quotient goes to bfp16 from fp32, with no bf16 rounding.
        const float num = p.swap ? b : a, den = p.swap ? a : b;
        v[k] = mul_q(num, inv16(den, NPU_BF16));
      }
    }
  }
  encode_bfp16(v, kHalfEven, out + i * 9);
}

struct MulAdd {
  Shape s;
  int bfp2, bc2, bfp3, bc3, out_bfp, act_type, activation;
  float lo, hi;  // the bf16 clip bounds
};

// x2 / x3 element for flat output value i: bf16 or a decoded bfp16 block (H / W broadcast).
__device__ inline float mul_add_operand(const void* x, int bfp, int bc, int64_t i,
                                        const Shape& s) {
  const BIdx j = bidx_flat(i, bc, s.W, s.C);
  if (bfp) {
    const int64_t pb = j.pixel / 8, cb = j.c / 8;
    return decode_bfp16_identity(static_cast<const uint8_t*>(x), pb + (cb & ~int64_t(7)),
                                 pb + s.C / 8, pb + cb, int(i % 8));
  }
  return bf16_to_f32(static_cast<const uint16_t*>(x)[j.flat()]);
}

// One thread per 8 consecutive output channels (one bfp16 block).
__global__ void mul_add_kernel(int64_t n, const uint8_t* x1, const void* x2, const void* x3,
                               void* out, MulAdd p) {
  NPU_TID;
  const int64_t nb = p.s.C / 8;
  const int64_t cb = i % nb;
  float v[8];
  for (int k = 0; k < 8; ++k) {
    const float a = decode_bfp16_identity(x1, i - cb + (cb & ~int64_t(7)), i - cb + nb, i, k);
    float r = fma_q(a, mul_add_operand(x2, p.bfp2, p.bc2, i * 8 + k, p.s),
                   mul_add_operand(x3, p.bfp3, p.bc3, i * 8 + k, p.s));
    if (p.act_type == kActClip && p.activation == kActClip) {
      r = vclamp(round_float(r, NPU_BF16, kHalfEven), p.lo, p.hi);
    } else if (p.act_type == kActTanh) {
      r = round_float(r, NPU_BF16, kHalfEven);
      if (p.activation == kActTanh) r = sigtanh_tanh_bf16(r);
    }
    v[k] = r;
  }
  if (p.out_bfp) {
    encode_bfp16(v, kHalfEven, static_cast<uint8_t*>(out) + i * 9);
  } else {
    for (int k = 0; k < 8; ++k) store_float(out, NPU_BF16, i * 8 + k, v[k], kHalfEven);
  }
}

struct Fp8Bin {
  Shape s;
  int bc;
  float c1, c2;
  int act;
  float a0, a1;
  int swap;
};

__global__ void add_fp8_kernel(int64_t n, const uint8_t* x1, const uint8_t* x2,
                               const float* table, uint8_t* out, Fp8Bin p) {
  NPU_TID;
  const int64_t c = i % p.s.C;
  const BIdx j = bidx_flat(i, p.bc, p.s.W, p.s.C);
  const float a = fp8_decode_identity(x1 + (i - c), c, p.s.C);
  float v;
  if (!(p.bc & 1)) {
    const float b = fp8_decode_identity(x2 + j.pixel, j.c, j.cn);
    v = activate(fma_q(b, p.c2, mul_q(a, p.c1)), p.act, p.a0, p.a1);
  } else {
    // C broadcast: the scalar is converted per value.
    const uint8_t code = x2[j.pixel];
    const float b = fp8_to_f32(code);
    switch (p.act) {
      case kActRelu:
        v = vmax(add_q(mul_q(a, p.c1), mul_q(b, p.c2)), p.a0);
        break;
      case kActRelu6:
        v = vclamp(add_q(mul_q(a, p.c1), table[code]), p.a0, p.a1);
        break;
      case kActLrelu: {
        // max(q, alpha q) as qA + (B/A)|qA| with table[256] = A * c1, table[257] = B / A.
        const float q = fma_q(a, table[256], table[code]);
        v = fma_q(fabsf(q), table[257], q);
        break;
      }
      default:
        v = fma_q(a, p.c1, mul_q(b, p.c2));
    }
  }
  out[i] = f32_to_fp8(v, kHalfAway);
}

__global__ void mul_fp8_kernel(int64_t n, const uint8_t* x1, const uint8_t* x2, uint8_t* out,
                               Fp8Bin p) {
  NPU_TID;
  const int64_t c = i % p.s.C;
  const BIdx j = bidx_flat(i, p.bc, p.s.W, p.s.C);
  float v;
  if (!(p.bc & 1)) {
    // Direct product of the fp8 values, no row decode.
    v = mul_q(mul_q(fp8_to_f32(x1[i]), fp8_to_f32(x2[j.flat()])), p.c1);
  } else {
    const float s = mul_q(fp8_decode_identity(x2 + j.pixel, 0, 1), p.c1);
    v = mul_q(fp8_decode_identity(x1 + (i - c), c, p.s.C), s);
  }
  out[i] = f32_to_fp8(v, kHalfAway);
}

__global__ void div_fp8_kernel(int64_t n, const uint8_t* x1, const uint8_t* x2, uint8_t* out,
                               Fp8Bin p, const uint32_t* exc) {
  NPU_TID;
  const int64_t c = i % p.s.C;
  const BIdx j = bidx_flat(i, p.bc, p.s.W, p.s.C);
  const float a = fp8_decode_identity(x1 + (i - c), c, p.s.C);
  const float b = fp8_decode_identity(x2 + j.pixel, j.c, j.cn);
  const float num = p.swap ? b : a, den = p.swap ? a : b;
  const float r = hw_inv_f32(den, exc);
  out[i] = f32_to_fp8(mul_q(mul_q(num, p.c1), r), kHalfAway);
}

__global__ void mac_fp8_kernel(int64_t n, const uint8_t* x1, const uint8_t* x2,
                               const uint8_t* x3, uint8_t* out, Fp8Bin p) {
  NPU_TID;
  const int64_t c = i % p.s.C;
  const BIdx j = bidx_flat(i, p.bc, p.s.W, p.s.C);
  const float a = fp8_decode_identity(x1 + (i - c), c, p.s.C);
  float v;
  if (p.bc & 1) {
    // Scalars: -0 -> +0, NaN only from the value itself.
    const float b = add_q(fp8_to_f32(x2[j.pixel]), 0.0f);
    const float d = add_q(fp8_to_f32(x3[j.pixel]), 0.0f);
    v = fma_q(a, mul_q(b, p.c1), mul_q(d, p.c2));
  } else {
    const float b = fp8_decode_identity(x2 + j.pixel, j.c, j.cn);
    const float d = fp8_decode_identity(x3 + j.pixel, j.c, j.cn);
    v = fma_q(d, p.c2, mul_q(b, mul_q(a, p.c1)));
  }
  out[i] = f32_to_fp8(activate(v, p.act, p.a0, p.a1), kHalfEven);
}

__global__ void leakyrelu16_kernel(int64_t n, const void* in, npu_dtype t, void* out,
                                   float alpha) {
  NPU_TID;
  // alpha gets the scalar cast to the element type; x * alpha is then exact.
  const float x = load_float(in, t, i);
  const float v = activate(x, kActLrelu, scalar_cast(alpha, t, kHalfEven), 0.0f);
  store_float(out, t, i, v, kHalfEven);
}

__global__ void leakyrelu_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, float pos,
                                     float neg) {
  NPU_TID;
  const float x = fp8_decode_identity(in, i, n);
  out[i] = f32_to_fp8(vmax(mul_q(x, pos), mul_q(x, neg)), kHalfAway);
}

__global__ void prelu16_kernel(int64_t n, const void* in, npu_dtype t, void* out,
                               const float* D, int64_t C) {
  NPU_TID;
  const float x = load_float(in, t, i);
  store_float(out, t, i, fma_q(vmin(x, 0.0f), D[i % C], x), kHalfEven);
}

__global__ void prelu_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, float P,
                                 const float* D, int64_t C) {
  NPU_TID;
  const float x = fp8_decode_identity(in, i, n);
  out[i] = f32_to_fp8(fma_q(vmin(x, 0.0f), D[i % C], mul_q(x, P)), kHalfAway);
}

__global__ void square16_kernel(int64_t n, const void* in, npu_dtype t, void* out) {
  NPU_TID;
  const uint16_t x = static_cast<const uint16_t*>(in)[i];
  static_cast<uint16_t*>(out)[i] = mul16(x, x, t);
}

__global__ void square_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, float s) {
  NPU_TID;
  const float x = fp8_decode_identity(in, i, n);
  out[i] = f32_to_fp8(mul_q(x, mul_q(x, s)), kHalfAway);
}

// cw / cb are 16-bit: bf16 for a bf16 tensor, else f16.
__global__ void batchnorm_kernel(int64_t n, const void* in, npu_dtype t, void* out, int64_t C,
                                 const void* cw, const void* cb, int act, float alpha) {
  NPU_TID;
  const bool fp8 = t == NPU_FP8;
  const npu_dtype wt = t == NPU_BF16 ? NPU_BF16 : NPU_F16;
  const float x =
      fp8 ? fp8_decode_identity(static_cast<const uint8_t*>(in), i, n) : load_float(in, t, i);
  float acc = fma_q(x, load_float(cw, wt, i % C), load_float(cb, wt, i % C));
  if (act == kActLrelu) {
    acc = activate(acc, kActLrelu, alpha, 0.0f);
  } else if (act == kActRelu && fp8) {
    acc = activate(acc, kActRelu, 0.0f, 0.0f);
  }
  if (fp8) {
    store_float(out, NPU_FP8, i, acc, kHalfEven);
  } else {
    // RELU on the narrowed 16-bit value (-0 below +0).
    if (act == kActRelu) acc = vmax(round_float(acc, t, kHalfEven), 0.0f);
    store_float(out, t, i, acc, kHalfEven);
  }
}

inline bool act_ok(int act) { return act >= 0 && act <= kActLrelu; }

int bin16(void* stream, const void* x1, const void* x2, npu_dtype dtype, void* out,
          const int64_t* shape1, const int64_t* shape2, BinOp op, int neg1, int neg2, int swap,
          int act, float a0, float a1) {
  const int bc = bcast_check(shape1, shape2);
  if (any_null(x1, x2, out) || !is_half_float(dtype) || bc < 0 || !act_ok(act)) return -1;
  const int64_t H = shape1[0], W = shape1[1], C = shape1[2];
  const Bin16 p{{H, W, C}, bc, dtype, op, neg1, neg2, swap, act, a0, a1};
  return launch(stream, H * W * C, bin16_kernel, x1, x2, out, p);
}

int bfp(void* stream, const void* x1, const void* x2, void* out, const int64_t* shape1,
        const int64_t* shape2, BfpOp op, int swap) {
  const int bc = bcast_check(shape1, shape2);
  if (any_null(x1, x2, out) || bc < 0 || shape1[2] % 8) return -1;
  const int64_t H = shape1[0], W = shape1[1], C = shape1[2];
  const Bfp p{{H, W, C}, bc, op, swap};
  return launch(stream, H * W * C / 8, bfp_kernel, static_cast<const uint8_t*>(x1),
                static_cast<const uint8_t*>(x2), static_cast<uint8_t*>(out), p);
}

// The fp8 binary entries: x2 broadcast over x1.
bool fp8_bin_ok(const void* x1, const void* x2, void* out, const int64_t* shape1,
                const int64_t* shape2, int* bc) {
  *bc = bcast_check(shape1, shape2);
  return !any_null(x1, x2, out) && *bc >= 0;
}

}  // namespace

extern "C" {

int npu_elementwise_broadcasting3d_add_bf16(void* stream, const void* x1, npu_dtype dtype,
    const void* x2, void* out, const int64_t* shape1, const int64_t* shape2, int is_sub_flag,
    int negate_flag, int activation, float activation_coeff0, float activation_coeff1) {
  return bin16(stream, x1, x2, dtype, out, shape1, shape2, kAdd, negate_flag, is_sub_flag, 0,
               activation, activation_coeff0, activation_coeff1);
}

int npu_elementwise_broadcasting3d_mul_bf16(void* stream, const void* x1, npu_dtype dtype,
    const void* x2, void* out, const int64_t* shape1, const int64_t* shape2, int activation,
    float activation_coeff0, float activation_coeff1) {
  return bin16(stream, x1, x2, dtype, out, shape1, shape2, kMul, 0, 0, 0, activation,
               activation_coeff0, activation_coeff1);
}

int npu_elementwise_broadcasting3d_div_bf16(void* stream, const void* x1, npu_dtype dtype,
    const void* x2, void* out, const int64_t* shape1, const int64_t* shape2, int swap_inputs) {
  return bin16(stream, x1, x2, dtype, out, shape1, shape2, kDiv, 0, 0, swap_inputs, 0, 0.0f,
               0.0f);
}

int npu_elementwise_broadcasting3d_mac_fused_bf16(void* stream, const void* x1, npu_dtype dtype,
    const void* x2, const void* x3, void* out, const int64_t* shape1, const int64_t* shape23,
    int activation, float activation_coeff0, float activation_coeff1) {
  const int bc = bcast_check(shape1, shape23);
  if (any_null(x1, x2, x3, out) || !is_half_float(dtype) || bc < 0 || !act_ok(activation))
    return -1;
  const int64_t H = shape1[0], W = shape1[1], C = shape1[2];
  const Bin16 p{{H, W, C}, bc, dtype, kAdd, 0, 0, 0, activation, activation_coeff0,
                activation_coeff1};
  return launch(stream, H * W * C, mac16_kernel, x1, x2, x3, out, p);
}

int npu_elementwise_broadcasting3d_add_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2) {
  return bfp(stream, x1, x2, out, shape1, shape2, kBfpAdd, 0);
}
int npu_elementwise_broadcasting3d_sub_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, int swap_inputs) {
  return bfp(stream, x1, x2, out, shape1, shape2, kBfpSub, swap_inputs);
}
int npu_elementwise_broadcasting3d_mul_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2) {
  return bfp(stream, x1, x2, out, shape1, shape2, kBfpMul, 0);
}
int npu_elementwise_broadcasting3d_div_bfp16(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, int swap_inputs) {
  return bfp(stream, x1, x2, out, shape1, shape2, kBfpDiv, swap_inputs);
}

int npu_elementwise_broadcasting3d_mul_add_bfp16(void* stream, const void* x1, const void* x2,
    npu_dtype x2_type, const void* x3, npu_dtype x3_type, void* out, npu_dtype out_type,
    const int64_t* shape1, const int64_t* shape2, const int64_t* shape3, int activation_type,
    int activation, uint16_t clip_min, uint16_t clip_max) {
  const int bc2 = bcast_check(shape1, shape2), bc3 = bcast_check(shape1, shape3);
  auto bf_or_bfp = [](npu_dtype t) { return t == NPU_BF16 || t == NPU_BFP16; };
  if (any_null(x1, x2, x3, out) || !bf_or_bfp(x2_type) || !bf_or_bfp(x3_type) ||
      !bf_or_bfp(out_type) || bc2 < 0 || bc3 < 0 || shape1[2] % 8 || (bc2 & 1) || (bc3 & 1) ||
      !(activation_type == 0 || activation_type == kActTanh || activation_type == kActClip) ||
      !(activation == 0 || activation == activation_type))
    return -1;
  const int64_t H = shape1[0], W = shape1[1], C = shape1[2];
  const MulAdd p{{H, W, C}, x2_type == NPU_BFP16, bc2, x3_type == NPU_BFP16, bc3,
                 out_type == NPU_BFP16, activation_type, activation,
                 bf16_to_f32(clip_min), bf16_to_f32(clip_max)};
  return launch(stream, H * W * C / 8, mul_add_kernel, static_cast<const uint8_t*>(x1), x2,
                x3, out, p);
}

int npu_elementwise_broadcasting3d_add_fp8(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, float ifm1_coeff, float ifm2_coeff,
    int activation, float activation_coeff0, float activation_coeff1, const void* table) {
  int bc;
  if (!fp8_bin_ok(x1, x2, out, shape1, shape2, &bc) || !act_ok(activation)) return -1;
  if ((bc & 1) && (activation == kActRelu6 || activation == kActLrelu) && !table) return -1;
  const int64_t H = shape1[0], W = shape1[1], C = shape1[2];
  const Fp8Bin p{{H, W, C}, bc, ifm1_coeff, ifm2_coeff, activation, activation_coeff0,
                 activation_coeff1, 0};
  return launch(stream, H * W * C, add_fp8_kernel, static_cast<const uint8_t*>(x1),
                static_cast<const uint8_t*>(x2), static_cast<const float*>(table),
                static_cast<uint8_t*>(out), p);
}

int npu_elementwise_broadcasting3d_mul_fp8(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, float operation_coeff) {
  int bc;
  if (!fp8_bin_ok(x1, x2, out, shape1, shape2, &bc)) return -1;
  const int64_t H = shape1[0], W = shape1[1], C = shape1[2];
  const Fp8Bin p{{H, W, C}, bc, operation_coeff, 0.0f, 0, 0.0f, 0.0f, 0};
  return launch(stream, H * W * C, mul_fp8_kernel, static_cast<const uint8_t*>(x1),
                static_cast<const uint8_t*>(x2), static_cast<uint8_t*>(out), p);
}

int npu_elementwise_broadcasting3d_div_fp8(void* stream, const void* x1, const void* x2,
    void* out, const int64_t* shape1, const int64_t* shape2, float operation_coeff,
    int swap_inputs) {
  int bc;
  if (!fp8_bin_ok(x1, x2, out, shape1, shape2, &bc)) return -1;
  const int64_t H = shape1[0], W = shape1[1], C = shape1[2];
  if (H * W * C == 0) return 0;
  const Fp8Bin p{{H, W, C}, bc, operation_coeff, 0.0f, 0, 0.0f, 0.0f, swap_inputs};
  const uint32_t* exc;
  if (int rc = npu_emu_nlf_table(NPU_EMU_NLF_INV_F32, &exc)) return rc;
  return launch(stream, H * W * C, div_fp8_kernel, static_cast<const uint8_t*>(x1),
                static_cast<const uint8_t*>(x2), static_cast<uint8_t*>(out), p, exc);
}

int npu_elementwise_broadcasting3d_mac_fused_fp8(void* stream, const void* x1, const void* x2,
    const void* x3, void* out, const int64_t* shape1, const int64_t* shape23, float c0, float c4,
    int activation, float activation_coeff0, float activation_coeff1) {
  const int bc = bcast_check(shape1, shape23);
  if (any_null(x1, x2, x3, out) || bc < 0 || !act_ok(activation)) return -1;
  const int64_t H = shape1[0], W = shape1[1], C = shape1[2];
  const Fp8Bin p{{H, W, C}, bc, c0, c4, activation, activation_coeff0, activation_coeff1, 0};
  return launch(stream, H * W * C, mac_fp8_kernel, static_cast<const uint8_t*>(x1),
                static_cast<const uint8_t*>(x2), static_cast<const uint8_t*>(x3),
                static_cast<uint8_t*>(out), p);
}

int npu_elementwise_leakyrelu_bf16(void* stream, const void* in, npu_dtype dtype, void* out,
                                   int64_t n, float alpha) {
  if (any_null(in, out) || !is_half_float(dtype) || n < 0) return -1;
  return launch(stream, n, leakyrelu16_kernel, in, dtype, out, alpha);
}

int npu_elementwise_leakyrelu_fp8(void* stream, const void* in, void* out, int64_t n,
                                  float positive_coeff, float negative_coeff) {
  if (any_null(in, out) || n < 0) return -1;
  return launch(stream, n, leakyrelu_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), positive_coeff, negative_coeff);
}

int npu_elementwise_prelu_bf16(void* stream, const void* in, npu_dtype dtype, void* out,
                               int64_t n, int64_t C, const void* prelu_slope_delta) {
  if (any_null(in, out, prelu_slope_delta) || !is_half_float(dtype) || !per_channel_ok(n, C))
    return -1;
  return launch(stream, n, prelu16_kernel, in, dtype, out,
                static_cast<const float*>(prelu_slope_delta), C);
}

int npu_elementwise_prelu_fp8(void* stream, const void* in, void* out, int64_t n, int64_t C,
                              float positive_scale, const void* prelu_scaled_slope_delta) {
  if (any_null(in, out, prelu_scaled_slope_delta) || !per_channel_ok(n, C)) return -1;
  return launch(stream, n, prelu_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), positive_scale,
                static_cast<const float*>(prelu_scaled_slope_delta), C);
}

int npu_elementwise_square_bf16(void* stream, const void* in, npu_dtype dtype, void* out,
                                int64_t n) {
  if (any_null(in, out) || !is_half_float(dtype) || n < 0) return -1;
  return launch(stream, n, square16_kernel, in, dtype, out);
}

int npu_elementwise_square_fp8(void* stream, const void* in, void* out, int64_t n,
                               float ifm_coeff) {
  if (any_null(in, out) || n < 0) return -1;
  return launch(stream, n, square_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), ifm_coeff);
}

int npu_batchnorm2d_stream_bf16(void* stream, const void* in, npu_dtype dtype, void* out,
                                int64_t n, int64_t C, const void* cw, const void* cb,
                                int activation, float relu_alpha) {
  if (any_null(in, out, cw, cb) || !(is_half_float(dtype) || dtype == NPU_FP8) ||
      !per_channel_ok(n, C) ||
      !(activation == 0 || activation == kActRelu || activation == kActLrelu))
    return -1;
  return launch(stream, n, batchnorm_kernel, in, dtype, out, C, cw, cb, activation, relu_alpha);
}

}  // extern "C"
