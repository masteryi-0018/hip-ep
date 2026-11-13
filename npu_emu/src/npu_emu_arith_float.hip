/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Floating-point elementwise arithmetic wrappers (arith_float group), see
 * include/npu_emu_arith_float.h. One thread per output element (per bfp16
 * block of 8 for the bfp16 outputs); reference-speed kernels.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "npu_emu_arith_float.h"
#include "npu_emu_device.h"
#include "npu_emu_inv_f32_table.h"
#include "npu_emu_launch.h"

namespace {

using namespace npu_emu;

// mllib::ActivationType values of the mul_add template / runtime word.
constexpr int kActTanh = 7;
constexpr int kActClip = 100;

// ---- shared helpers -------------------------------------------------------

__device__ inline float qnan() { return __uint_as_float(0x7fc00000u); }

// fp32 VMAX / VMIN: -0 orders below +0 (aiesim prelu: min(-0, +0) = -0). NaN
// gives the canonical NaN (unverified).
__device__ inline float vmax(float a, float b) {
  if (a != a || b != b) return qnan();
  if (a == b) return signbit(a) ? b : a;
  return a > b ? a : b;
}
__device__ inline float vmin(float a, float b) {
  if (a != a || b != b) return qnan();
  if (a == b) return signbit(a) ? a : b;
  return a < b ? a : b;
}

__device__ inline float activate(float v, int act, float a0, float a1) {
  switch (act) {
    case NPU_FLOAT_ACT_RELU:
      return vmax(v, a0);
    case NPU_FLOAT_ACT_CLIP:
      return vmin(vmax(v, a0), a1);
    case NPU_FLOAT_ACT_LRELU:
      return vmax(v, __fmul_rn(v, a0));
    default:
      return v;
  }
}

__device__ inline float load16(const void* p, npu_dtype t, int64_t i) {
  const uint16_t h = static_cast<const uint16_t*>(p)[i];
  return t == NPU_F16 ? f16_to_f32(h) : bf16_to_f32(h);
}
__device__ inline uint16_t round16(float v, npu_dtype t) {
  return t == NPU_F16 ? f32_to_f16(v, kEven) : f32_to_bf16(v, kEven);
}
__device__ inline float rne16(float v, npu_dtype t) {
  const uint16_t h = round16(v, t);
  return t == NPU_F16 ? f16_to_f32(h) : bf16_to_f32(h);
}

// fp8 as the mul_8x8_8x8 identity decode gives it: the row of 8 (NaN spreads)
// and a sum of products, so -0 becomes +0 (x + 0.0f is not folded: it is not
// an identity for -0).
__device__ inline float fp8_row(const uint8_t* p, int64_t i, int64_t n) {
  return fp8_to_f32_row(p, i, n) + 0.0f;
}

// Flat index into an operand broadcast along bcast (bit 0 / 1 / 2 = C / W / H)
// for output (h, w, c); its pixel base and channel count for the fp8 rows.
struct BIdx {
  int64_t pixel, c, cn;
};
__device__ inline BIdx bidx(int64_t h, int64_t w, int64_t c, int bc, int64_t W, int64_t C) {
  const int64_t wn = bc & 2 ? 1 : W, cn = bc & 1 ? 1 : C;
  return {((bc & 4 ? 0 : h) * wn + (bc & 2 ? 0 : w)) * cn, bc & 1 ? 0 : c, cn};
}

struct Shape {
  int64_t H, W, C;
};
__device__ inline void hwc(int64_t i, const Shape& s, int64_t* h, int64_t* w, int64_t* c) {
  *c = i % s.C;
  *w = (i / s.C) % s.W;
  *h = i / (s.C * s.W);
}

// The AIE4 reciprocals of div_*: bf16 = the inv_bf16_aie4 table, i.e. bf16 RNE
// of the fp32 1/y except for zero / subnormal y (+-inf), results below 2^-126
// (+-0) and NaN (quieted); f16 = f16 RNE of the fp32 1/y.
__device__ inline float inv_bf16(float y) {
  const uint32_t b = __float_as_uint(y);
  const uint32_t e = (b >> 23) & 0xff;
  if (e == 0xff && (b & 0x7fffff)) return __uint_as_float(b | 0x400000u);
  if (e == 0) return __uint_as_float((b & 0x80000000u) | 0x7f800000u);
  const float r = __fdiv_rn(1.0f, y);
  if (fabsf(r) < 0x1p-126f) return copysignf(0.0f, y);
  return bf16_to_f32(f32_to_bf16(r, kEven));
}
__device__ inline float inv16(float y, npu_dtype t) {
  return t == NPU_F16 ? f16_to_f32(f32_to_f16(__fdiv_rn(1.0f, y), kEven)) : inv_bf16(y);
}

// ---- bf16 / f16 broadcasting ------------------------------------------------

enum BinOp { kAdd, kMul, kDiv };

struct Bin16 {
  Shape s;
  int bc;
  npu_dtype t;
  BinOp op;
  int neg1, neg2, swap, act;
  float a0, a1;
};

__global__ void bin16_kernel(int64_t n, const void* x1, const void* x2, void* out, Bin16 p) {
  NPU_TID;
  int64_t h, w, c;
  hwc(i, p.s, &h, &w, &c);
  const BIdx j = bidx(h, w, c, p.bc, p.s.W, p.s.C);
  const float a = load16(x1, p.t, i), b = load16(x2, p.t, j.pixel + j.c);
  float v;
  if (p.op == kAdd) {
    // mac_elem_32_conf(x1, ones, x2, sub_mul, sub_acc): one rounding.
    v = __fadd_rn(p.neg1 ? -a : a, p.neg2 ? -b : b);
  } else if (p.op == kMul) {
    v = __fmul_rn(a, b);
  } else {
    const float num = p.swap ? b : a, den = p.swap ? a : b;
    v = __fmul_rn(num, inv16(den, p.t));
  }
  static_cast<uint16_t*>(out)[i] = round16(activate(v, p.act, p.a0, p.a1), p.t);
}

__global__ void mac16_kernel(int64_t n, const void* x1, const void* x2, const void* x3,
                             void* out, Bin16 p) {
  NPU_TID;
  int64_t h, w, c;
  hwc(i, p.s, &h, &w, &c);
  const BIdx j = bidx(h, w, c, p.bc, p.s.W, p.s.C);
  const float v = fmaf(load16(x1, p.t, i), load16(x2, p.t, j.pixel + j.c),
                       load16(x3, p.t, j.pixel + j.c));
  static_cast<uint16_t*>(out)[i] = round16(activate(v, p.act, p.a0, p.a1), p.t);
}

// ---- bfp16 broadcasting -------------------------------------------------------

enum BfpOp { kBfpAdd, kBfpSub, kBfpMul, kBfpDiv };

struct Bfp {
  Shape s;  // C in values (multiple of 8)
  int bc;
  BfpOp op;
  int swap;
};

// One thread per output block (8 channels of a pixel).
__global__ void bfp_kernel(int64_t n, const uint8_t* x1, const uint8_t* x2, uint8_t* out,
                           Bfp p) {
  NPU_TID;
  const int64_t nb = p.s.C / 8;
  const int64_t cb = i % nb, w = (i / nb) % p.s.W, h = i / (nb * p.s.W);
  const BIdx j = bidx(h, w, cb, p.bc, p.s.W, nb);  // indices in blocks
  const uint8_t* b1 = x1 + i * 9;
  const uint8_t* b2 = x2 + (j.pixel + j.c) * 9;
  float v[8];
  for (int k = 0; k < 8; ++k) {
    const float a = decode_bfp16(b1, k);
    const float b = decode_bfp16(b2, p.bc & 1 ? 0 : k);
    switch (p.op) {
      case kBfpAdd:
        v[k] = __fadd_rn(a, b);
        break;
      case kBfpSub:
        v[k] = p.swap ? __fsub_rn(b, a) : __fsub_rn(a, b);
        break;
      case kBfpMul:
        v[k] = __fmul_rn(a, b);
        break;
      default: {
        // Operands are exact in bf16; the quotient goes to bfp16 from fp32.
        const float num = p.swap ? b : a, den = p.swap ? a : b;
        v[k] = __fmul_rn(num, inv_bf16(den));
      }
    }
  }
  encode_bfp16(v, kEven, out + i * 9);
}

// ---- mul_add bfp16 ------------------------------------------------------------

struct MulAdd {
  Shape s;
  int bfp2, bc2, bfp3, bc3, out_bfp, act_type, activation;
  float lo, hi;
};

// x2 / x3 element (h, w, c): bf16 or a decoded bfp16 block (H / W broadcast).
__device__ inline float mul_add_operand(const void* x, int bfp, int bc, int64_t h, int64_t w,
                                        int64_t c, const Shape& s) {
  const BIdx j = bidx(h, w, c, bc, s.W, s.C);
  if (bfp) return decode_bfp16(static_cast<const uint8_t*>(x) + (j.pixel + j.c) / 8 * 9, c % 8);
  return bf16_to_f32(static_cast<const uint16_t*>(x)[j.pixel + j.c]);
}

__device__ inline float rne_bf16(float v) { return bf16_to_f32(f32_to_bf16(v, kEven)); }

// exp2_bf20_hw as mllib's NlfModel.approx_exp2(bits_out=19, no magic constant):
// a piecewise-linear 2^x on the fp32 bit pattern of p (the magic constant is
// folded into k0 of SigTanh). Result subnormals flush to +0.
__device__ inline float exp2_bf20(float p) {
  constexpr int16_t kCnst[32] = {0,   39,  74,  110, 141, 172, 200, 225, 250, 270, 290,
                                 305, 320, 332, 340, 347, 352, 352, 350, 347, 340, 328,
                                 315, 299, 280, 257, 229, 200, 167, 131, 91,  47};
  constexpr int8_t kCoeff[32] = {-10, -9, -9, -8, -8, -7, -6, -6, -5, -5, -4,
                                 -4,  -3, -2, -2, -1, 0,  0,  1,  2,  3,  3,
                                 4,   5,  6,  7,  7,  8,  9,  10, 11, 12};
  if (p != p) return qnan();
  const int64_t x = int64_t(__float_as_int(p + 0.0f) >> 2) - (int64_t(0x111) << 20);
  const int sel = int((x >> 7) & 31);
  int64_t approx = (x & ~int64_t(1)) + ((((x >> 2) & 31) * kCoeff[sel]) >> 3) - kCnst[sel];
  if (x < (1 << 12) && x >= -(11 << 12)) {
    const int64_t data = 4096 + (approx & 4095);
    const int sh = int(1 - (approx >> 12));
    approx = (data >> sh) | ((data & ((int64_t(1) << sh) - 1)) != 0);
  }
  approx = approx < 0 ? 0 : (approx > (255 << 12) ? (255 << 12) : approx);
  const uint32_t bits = uint32_t((approx << 11) & 0x7fffffff);
  return (bits >> 23) == 0 ? 0.0f : __uint_as_float(bits);
}

// SOL SigTanh<TANH> on a bf16 value, bf16 output under symmetric_inf.
__device__ inline float sigtanh_bf16(float x) {
  const float k0 = 894.00097656f, k1 = 1.53878212f, k2 = -0.49819374f;
  const float p01 = fmaf(-fabsf(x), k1, k0);
  const float p = fmaf(__fmul_rn(x, x), k2, p01);
  const float e = exp2_bf20(p);
  float t = fmaf(-2.0f, e, 1.0f);
  if (signbit(x)) t = __uint_as_float(__float_as_uint(t) ^ 0x80000000u);
  const uint16_t r = f32_to_bf16(t, kAway);
  // srs(..., filter_denorm): a subnormal result or input flushes to signed 0.
  if (((r >> 7) & 0xff) == 0 || ((__float_as_uint(t) >> 23) & 0xff) == 0)
    return __uint_as_float(uint32_t(r & 0x8000) << 16);
  return bf16_to_f32(r);
}

// One thread per 8 consecutive output channels (one bfp16 block).
__global__ void mul_add_kernel(int64_t n, const uint8_t* x1, const void* x2, const void* x3,
                               void* out, MulAdd p) {
  NPU_TID;
  const int64_t nb = p.s.C / 8;
  const int64_t cb = i % nb, w = (i / nb) % p.s.W, h = i / (nb * p.s.W);
  float v[8];
  for (int k = 0; k < 8; ++k) {
    const int64_t c = cb * 8 + k;
    const float a = decode_bfp16(x1 + i * 9, k);
    float r = fmaf(a, mul_add_operand(x2, p.bfp2, p.bc2, h, w, c, p.s),
                   mul_add_operand(x3, p.bfp3, p.bc3, h, w, c, p.s));
    if (p.act_type == kActClip && p.activation == kActClip) {
      r = vmin(vmax(rne_bf16(r), p.lo), p.hi);
    } else if (p.act_type == kActTanh) {
      r = rne_bf16(r);
      if (p.activation == kActTanh) r = sigtanh_bf16(r);
    }
    v[k] = r;
  }
  if (p.out_bfp) {
    encode_bfp16(v, kEven, static_cast<uint8_t*>(out) + i * 9);
  } else {
    for (int k = 0; k < 8; ++k) static_cast<uint16_t*>(out)[i * 8 + k] = f32_to_bf16(v[k], kEven);
  }
}

// ---- fp8 broadcasting ---------------------------------------------------------

__device__ inline uint8_t fp8_out(float v, Rnd rnd) { return f32_to_fp8(v, rnd); }

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
  int64_t h, w, c;
  hwc(i, p.s, &h, &w, &c);
  const BIdx j = bidx(h, w, c, p.bc, p.s.W, p.s.C);
  const float a = fp8_row(x1 + (i - c), c, p.s.C);
  float v;
  if (!(p.bc & 1)) {
    const float b = fp8_row(x2 + j.pixel, j.c, j.cn);
    v = activate(fmaf(b, p.c2, __fmul_rn(a, p.c1)), p.act, p.a0, p.a1);
  } else {
    // Channel broadcast: the ifm2 scalar is read and converted per value.
    const uint8_t code = x2[j.pixel];
    const float b = fp8_to_f32(code);
    switch (p.act) {
      case NPU_FLOAT_ACT_RELU:
        v = vmax(__fadd_rn(__fmul_rn(a, p.c1), __fmul_rn(b, p.c2)), p.a0);
        break;
      case NPU_FLOAT_ACT_CLIP:
        v = vmin(vmax(__fadd_rn(__fmul_rn(a, p.c1), table[code]), p.a0), p.a1);
        break;
      case NPU_FLOAT_ACT_LRELU: {
        // max(q, alpha q) as qA + (B/A)|qA| with the YAML's packed A, B/A.
        const float q = fmaf(a, table[256], table[code]);
        v = fmaf(fabsf(q), table[257], q);
        break;
      }
      default:
        v = fmaf(a, p.c1, __fmul_rn(b, p.c2));
    }
  }
  out[i] = fp8_out(v, kAway);
}

__global__ void mul_fp8_kernel(int64_t n, const uint8_t* x1, const uint8_t* x2, uint8_t* out,
                               Fp8Bin p) {
  NPU_TID;
  int64_t h, w, c;
  hwc(i, p.s, &h, &w, &c);
  const BIdx j = bidx(h, w, c, p.bc, p.s.W, p.s.C);
  float v;
  if (!(p.bc & 1)) {
    // mul_elem_32 on the fp8 vectors: no identity decode.
    v = __fmul_rn(__fmul_rn(fp8_to_f32(x1[i]), fp8_to_f32(x2[j.pixel + j.c])), p.c1);
  } else {
    const float s = __fmul_rn(fp8_row(x2 + j.pixel, 0, 1), p.c1);
    v = __fmul_rn(fp8_row(x1 + (i - c), c, p.s.C), s);
  }
  out[i] = fp8_out(v, kAway);
}

__global__ void div_fp8_kernel(int64_t n, const uint8_t* x1, const uint8_t* x2, uint8_t* out,
                               Fp8Bin p) {
  NPU_TID;
  int64_t h, w, c;
  hwc(i, p.s, &h, &w, &c);
  const BIdx j = bidx(h, w, c, p.bc, p.s.W, p.s.C);
  const float a = fp8_row(x1 + (i - c), c, p.s.C);
  const float b = fp8_row(x2 + j.pixel, j.c, j.cn);
  const float num = p.swap ? b : a, den = p.swap ? a : b;
  // aie::inv: the faithfully rounded fp32 reciprocal (fp8 values have 4
  // significant bits, inside the table); 0 / Inf / NaN as IEEE.
  const float r = den != 0.0f && isfinite(den) ? inv_f32_aie4(den) : __fdiv_rn(1.0f, den);
  out[i] = fp8_out(__fmul_rn(__fmul_rn(num, p.c1), r), kAway);
}

__global__ void mac_fp8_kernel(int64_t n, const uint8_t* x1, const uint8_t* x2,
                               const uint8_t* x3, uint8_t* out, Fp8Bin p) {
  NPU_TID;
  int64_t h, w, c;
  hwc(i, p.s, &h, &w, &c);
  const BIdx j = bidx(h, w, c, p.bc, p.s.W, p.s.C);
  const float a = fp8_row(x1 + (i - c), c, p.s.C);
  float v;
  if (p.bc & 1) {
    // Scalars broadcast to a full row before the decode: NaN only from itself.
    const float b = fp8_to_f32(x2[j.pixel]) + 0.0f, d = fp8_to_f32(x3[j.pixel]) + 0.0f;
    v = fmaf(a, __fmul_rn(b, p.c1), __fmul_rn(d, p.c2));
  } else {
    const float b = fp8_row(x2 + j.pixel, j.c, j.cn), d = fp8_row(x3 + j.pixel, j.c, j.cn);
    v = fmaf(d, p.c2, __fmul_rn(b, __fmul_rn(a, p.c1)));
  }
  out[i] = fp8_out(activate(v, p.act, p.a0, p.a1), kEven);
}

// ---- unary / per channel ------------------------------------------------------

__global__ void leakyrelu16_kernel(int64_t n, const void* in, npu_dtype t, void* out,
                                   float alpha) {
  NPU_TID;
  // alpha_value = static_cast<dtype>(alpha) under conv_even; x * alpha is exact.
  const float x = load16(in, t, i);
  static_cast<uint16_t*>(out)[i] = round16(vmax(x, __fmul_rn(x, rne16(alpha, t))), t);
}

__global__ void leakyrelu_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, float pos,
                                     float neg) {
  NPU_TID;
  const float x = fp8_row(in, i, n);
  out[i] = fp8_out(vmax(__fmul_rn(x, pos), __fmul_rn(x, neg)), kAway);
}

__global__ void prelu16_kernel(int64_t n, const void* in, npu_dtype t, void* out,
                               const float* D, int64_t C) {
  NPU_TID;
  const float x = load16(in, t, i);
  static_cast<uint16_t*>(out)[i] = round16(fmaf(vmin(x, 0.0f), D[i % C], x), t);
}

__global__ void prelu_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, float P,
                                 const float* D, int64_t C) {
  NPU_TID;
  const float x = fp8_row(in, i, n);
  out[i] = fp8_out(fmaf(vmin(x, 0.0f), D[i % C], __fmul_rn(x, P)), kAway);
}

__global__ void square16_kernel(int64_t n, const void* in, npu_dtype t, void* out) {
  NPU_TID;
  const float x = load16(in, t, i);
  float v = __fmul_rn(x, x);
  if (t == NPU_F16 && fabsf(v) < 0x1p-14f) v = 0.0f;
  static_cast<uint16_t*>(out)[i] = round16(v, t);
}

__global__ void square_fp8_kernel(int64_t n, const uint8_t* in, uint8_t* out, float s) {
  NPU_TID;
  const float x = fp8_row(in, i, n);
  out[i] = fp8_out(__fmul_rn(x, __fmul_rn(x, s)), kAway);
}

__global__ void batchnorm_kernel(int64_t n, const void* in, npu_dtype t, void* out, int64_t C,
                                 const float* cw, const float* cb, int act, float alpha) {
  NPU_TID;
  const bool fp8 = t == NPU_FP8;
  const float x = fp8 ? fp8_row(static_cast<const uint8_t*>(in), i, n) : load16(in, t, i);
  float acc = fmaf(x, cw[i % C], cb[i % C]);
  if (act == 3) {
    acc = vmax(acc, __fmul_rn(acc, alpha));
  } else if (act == 1 && fp8) {
    acc = vmax(acc, 0.0f);
  }
  if (fp8) {
    static_cast<uint8_t*>(out)[i] = fp8_out(acc, kEven);
  } else {
    // RELU on the narrowed 16-bit vector (bf16 VMAX: -0 below +0).
    if (act == 1) acc = vmax(rne16(acc, t), 0.0f);
    static_cast<uint16_t*>(out)[i] = round16(acc, t);
  }
}

// ---- argument checks ------------------------------------------------------------

inline bool half_type(npu_dtype t) { return t == NPU_BF16 || t == NPU_F16; }
inline bool shape_ok(int64_t H, int64_t W, int64_t C, int bc) {
  return H >= 0 && W >= 0 && C >= 0 && bc >= 0 && bc <= 7;
}
inline bool act_ok(int act) { return act >= NPU_FLOAT_ACT_NONE && act <= NPU_FLOAT_ACT_LRELU; }

int bin16(void* stream, const void* x1, const void* x2, int bc, npu_dtype t, void* out,
          int64_t H, int64_t W, int64_t C, BinOp op, int neg1, int neg2, int swap, int act,
          float a0, float a1) {
  if (!half_type(t) || !shape_ok(H, W, C, bc) || !act_ok(act)) return -1;
  const Bin16 p{{H, W, C}, bc, t, op, neg1, neg2, swap, act, a0, a1};
  return launch(stream, H * W * C, bin16_kernel, x1, x2, out, p);
}

int bfp(void* stream, const void* x1, const void* x2, int bc, void* out, int64_t H, int64_t W,
        int64_t C, BfpOp op, int swap) {
  if (!shape_ok(H, W, C, bc) || C % 8) return -1;
  const Bfp p{{H, W, C}, bc, op, swap};
  return launch(stream, H * W * C / 8, bfp_kernel, static_cast<const uint8_t*>(x1),
                static_cast<const uint8_t*>(x2), static_cast<uint8_t*>(out), p);
}

}  // namespace

extern "C" {

int npu_elementwise_broadcasting3d_add_bf16(void* stream, const void* x1, const void* x2,
                                            int x2_bcast, npu_dtype type, void* out,
                                            int64_t H, int64_t W, int64_t C, int negate1,
                                            int negate2, npu_float_act act, float act0,
                                            float act1) {
  return bin16(stream, x1, x2, x2_bcast, type, out, H, W, C, kAdd, negate1, negate2, 0, act,
               act0, act1);
}

int npu_elementwise_broadcasting3d_mul_bf16(void* stream, const void* x1, const void* x2,
                                            int x2_bcast, npu_dtype type, void* out,
                                            int64_t H, int64_t W, int64_t C,
                                            npu_float_act act, float act0, float act1) {
  return bin16(stream, x1, x2, x2_bcast, type, out, H, W, C, kMul, 0, 0, 0, act, act0, act1);
}

int npu_elementwise_broadcasting3d_div_bf16(void* stream, const void* x1, const void* x2,
                                            int x2_bcast, npu_dtype type, void* out,
                                            int64_t H, int64_t W, int64_t C, int swap) {
  return bin16(stream, x1, x2, x2_bcast, type, out, H, W, C, kDiv, 0, 0, swap,
               NPU_FLOAT_ACT_NONE, 0.0f, 0.0f);
}

int npu_elementwise_broadcasting3d_mac_fused_bf16(void* stream, const void* x1,
                                                  const void* x2, const void* x3,
                                                  int x23_bcast, npu_dtype type, void* out,
                                                  int64_t H, int64_t W, int64_t C,
                                                  npu_float_act act, float act0, float act1) {
  if (!half_type(type) || !shape_ok(H, W, C, x23_bcast) || !act_ok(act)) return -1;
  const Bin16 p{{H, W, C}, x23_bcast, type, kAdd, 0, 0, 0, act, act0, act1};
  return launch(stream, H * W * C, mac16_kernel, x1, x2, x3, out, p);
}

int npu_elementwise_broadcasting3d_add_bfp16(void* stream, const void* x1, const void* x2,
                                             int x2_bcast, void* out, int64_t H, int64_t W,
                                             int64_t C) {
  return bfp(stream, x1, x2, x2_bcast, out, H, W, C, kBfpAdd, 0);
}
int npu_elementwise_broadcasting3d_sub_bfp16(void* stream, const void* x1, const void* x2,
                                             int x2_bcast, void* out, int64_t H, int64_t W,
                                             int64_t C, int swap) {
  return bfp(stream, x1, x2, x2_bcast, out, H, W, C, kBfpSub, swap);
}
int npu_elementwise_broadcasting3d_mul_bfp16(void* stream, const void* x1, const void* x2,
                                             int x2_bcast, void* out, int64_t H, int64_t W,
                                             int64_t C) {
  return bfp(stream, x1, x2, x2_bcast, out, H, W, C, kBfpMul, 0);
}
int npu_elementwise_broadcasting3d_div_bfp16(void* stream, const void* x1, const void* x2,
                                             int x2_bcast, void* out, int64_t H, int64_t W,
                                             int64_t C, int swap) {
  return bfp(stream, x1, x2, x2_bcast, out, H, W, C, kBfpDiv, swap);
}

int npu_elementwise_broadcasting3d_mul_add_bfp16(
    void* stream, const void* x1, const void* x2, int x2_bfp16, int x2_bcast,
    const void* x3, int x3_bfp16, int x3_bcast, void* out, int out_bfp16, int64_t H,
    int64_t W, int64_t C, int act_type, int activation, float clip_min, float clip_max) {
  if (!shape_ok(H, W, C, x2_bcast) || !shape_ok(H, W, C, x3_bcast) || C % 8 ||
      (x2_bcast & 1) || (x3_bcast & 1) ||
      !(act_type == 0 || act_type == kActTanh || act_type == kActClip))
    return -1;
  const MulAdd p{{H, W, C}, x2_bfp16, x2_bcast, x3_bfp16, x3_bcast, out_bfp16,
                 act_type, activation, clip_min, clip_max};
  return launch(stream, H * W * C / 8, mul_add_kernel, static_cast<const uint8_t*>(x1), x2,
                x3, out, p);
}

int npu_elementwise_broadcasting3d_add_fp8(void* stream, const void* x1, const void* x2,
                                           int x2_bcast, void* out, int64_t H, int64_t W,
                                           int64_t C, float c1, float c2, npu_float_act act,
                                           float act0, float act1, const float* table) {
  if (!shape_ok(H, W, C, x2_bcast) || !act_ok(act)) return -1;
  if ((x2_bcast & 1) && (act == NPU_FLOAT_ACT_CLIP || act == NPU_FLOAT_ACT_LRELU) && !table)
    return -1;
  const Fp8Bin p{{H, W, C}, x2_bcast, c1, c2, act, act0, act1, 0};
  return launch(stream, H * W * C, add_fp8_kernel, static_cast<const uint8_t*>(x1),
                static_cast<const uint8_t*>(x2), table, static_cast<uint8_t*>(out), p);
}

int npu_elementwise_broadcasting3d_mul_fp8(void* stream, const void* x1, const void* x2,
                                           int x2_bcast, void* out, int64_t H, int64_t W,
                                           int64_t C, float scale) {
  if (!shape_ok(H, W, C, x2_bcast)) return -1;
  const Fp8Bin p{{H, W, C}, x2_bcast, scale, 0.0f, 0, 0.0f, 0.0f, 0};
  return launch(stream, H * W * C, mul_fp8_kernel, static_cast<const uint8_t*>(x1),
                static_cast<const uint8_t*>(x2), static_cast<uint8_t*>(out), p);
}

int npu_elementwise_broadcasting3d_div_fp8(void* stream, const void* x1, const void* x2,
                                           int x2_bcast, void* out, int64_t H, int64_t W,
                                           int64_t C, float scale, int swap) {
  if (!shape_ok(H, W, C, x2_bcast)) return -1;
  const Fp8Bin p{{H, W, C}, x2_bcast, scale, 0.0f, 0, 0.0f, 0.0f, swap};
  return launch(stream, H * W * C, div_fp8_kernel, static_cast<const uint8_t*>(x1),
                static_cast<const uint8_t*>(x2), static_cast<uint8_t*>(out), p);
}

int npu_elementwise_broadcasting3d_mac_fused_fp8(void* stream, const void* x1,
                                                 const void* x2, const void* x3,
                                                 int x23_bcast, void* out, int64_t H,
                                                 int64_t W, int64_t C, float c0, float c4,
                                                 npu_float_act act, float act0, float act1) {
  if (!shape_ok(H, W, C, x23_bcast) || !act_ok(act)) return -1;
  const Fp8Bin p{{H, W, C}, x23_bcast, c0, c4, act, act0, act1, 0};
  return launch(stream, H * W * C, mac_fp8_kernel, static_cast<const uint8_t*>(x1),
                static_cast<const uint8_t*>(x2), static_cast<const uint8_t*>(x3),
                static_cast<uint8_t*>(out), p);
}

int npu_elementwise_leakyrelu_bf16(void* stream, const void* in, npu_dtype type, void* out,
                                   int64_t n, float alpha) {
  if (!half_type(type) || n < 0) return -1;
  return launch(stream, n, leakyrelu16_kernel, in, type, out, alpha);
}

int npu_elementwise_leakyrelu_fp8(void* stream, const void* in, void* out, int64_t n,
                                  float pos, float neg) {
  if (n < 0) return -1;
  return launch(stream, n, leakyrelu_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), pos, neg);
}

int npu_elementwise_prelu_bf16(void* stream, const void* in, npu_dtype type, void* out,
                               int64_t n, const float* D, int64_t C) {
  if (!half_type(type) || n < 0 || C <= 0) return -1;
  return launch(stream, n, prelu16_kernel, in, type, out, D, C);
}

int npu_elementwise_prelu_fp8(void* stream, const void* in, void* out, int64_t n, float P,
                              const float* D, int64_t C) {
  if (n < 0 || C <= 0) return -1;
  return launch(stream, n, prelu_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), P, D, C);
}

int npu_elementwise_square_bf16(void* stream, const void* in, npu_dtype type, void* out,
                                int64_t n) {
  if (!half_type(type) || n < 0) return -1;
  return launch(stream, n, square16_kernel, in, type, out);
}

int npu_elementwise_square_fp8(void* stream, const void* in, void* out, int64_t n,
                               float scale) {
  if (n < 0) return -1;
  return launch(stream, n, square_fp8_kernel, static_cast<const uint8_t*>(in),
                static_cast<uint8_t*>(out), scale);
}

int npu_batchnorm2d_stream_bf16(void* stream, const void* in, npu_dtype type, void* out,
                                int64_t n, int64_t C, const float* cw, const float* cb,
                                int act, float alpha) {
  if (!(half_type(type) || type == NPU_FP8) || n < 0 || C <= 0 ||
      !(act == 0 || act == 1 || act == 3))
    return -1;
  return launch(stream, n, batchnorm_kernel, in, type, out, C, cw, cb, act, alpha);
}

}  // extern "C"
