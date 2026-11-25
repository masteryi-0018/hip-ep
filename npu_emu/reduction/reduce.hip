/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Kernels and entry points for reduction/reduce.h.
 *
 * One GPU thread runs one per-lane chain. When C is reduced the 64 lane results
 * of an output go to a scratch buffer and a second kernel combines them.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "common/float_cvt.h"
#include "common/int_round.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/nlf.h"
#include "common/check.h"
#include "common/mmac.h"
#include "common/quant.h"
#include "reduction/reduce.h"

using namespace npu_emu;

namespace {

constexpr int kLanes = 64;

enum { kDimH = 0, kDimW = 1, kDimG = 2 };

// Up to 5 nested loops, outermost first; loop j moves coordinate dim[j] by
// step[j], cnt[j] times.
struct Nest {
  int n;
  int dim[5];
  int64_t cnt[5];
  int64_t step[5];
};

// Nest from an order string over 'h', 'w', 'g', outermost first.
Nest make_nest(const char* order, int64_t H, int64_t W, int64_t G) {
  Nest ns{};
  for (const char* p = order; *p; ++p) {
    const int d = *p == 'h' ? kDimH : *p == 'w' ? kDimW : kDimG;
    ns.dim[ns.n] = d;
    ns.cnt[ns.n] = d == kDimH ? H : d == kDimW ? W : G;
    ns.step[ns.n] = 1;
    ++ns.n;
  }
  return ns;
}

void nest_push(Nest* ns, int dim, int64_t cnt, int64_t step) {
  ns->dim[ns->n] = dim;
  ns->cnt[ns->n] = cnt;
  ns->step[ns->n] = step;
  ++ns->n;
}

// Step order per axis_flags (index 1..7).
const char* const kOrderSum[8] = {"", "g", "w", "gw", "h", "gh", "wh", "ghw"};
const char* const kOrderProd[8] = {"", "g", "w", "wg", "h", "hg", "hw", "hwg"};
const char* const kOrderQdqProd[8] = {"", "g", "w", "wg", "h", "hg", "wh", "hwg"};
const char* const kOrderL2[8] = {"", "g", "w", "wg", "h", "hg", "wh", "hwg"};
const char* const kOrderMrs[8] = {"", "g", "w", "gw", "h", "hg", "hw", "hwg"};

// Lanes l and l + 32 added, then one vector reduce-add of the 32 sums.
template <typename T>
__device__ inline float lane_reduce(const T* v) {
  float s[32];
  for (int k = 0; k < 32; ++k) s[k] = add_q(v[k], v[k + 32]);
  return vaddr_f32_reduce(s, 32);
}

// Hardware 16-bit sqrt table.
__device__ inline float sqrt16(float v, npu_dtype t) {
  return decode_float(hw_sqrt16(uint16_t(encode_float(v, t, kHalfEven)), t), t);
}

// Thread i runs chain l = i % lanes of output o = i / lanes. Without C
// reduction lanes = C and l is the channel; with it lanes = 64 and the chain
// covers channels g * 64 + l of every group g in the nest.
template <class Op>
__global__ void chain_kernel(int64_t n, Op op, Nest ns, int64_t oW, int64_t lanes, int axis,
                             typename Op::Acc* part) {
  NPU_TID;
  const int64_t o = i / lanes, l = i % lanes;
  int64_t pos[3] = {(axis & 4) ? 0 : o / oW, (axis & 2) ? 0 : o % oW,
                    (axis & 1) ? 0 : l / kLanes};
  const int64_t lane = l % kLanes;
  int64_t idx[5] = {0, 0, 0, 0, 0};
  typename Op::Acc acc = op.init();
  bool first = true;
  for (;;) {
    acc = op.step(acc, pos[kDimH], pos[kDimW], pos[kDimG] * kLanes + lane, first);
    first = false;
    int j = ns.n - 1;
    for (; j >= 0; --j) {
      pos[ns.dim[j]] += ns.step[j];
      if (++idx[j] < ns.cnt[j]) break;
      pos[ns.dim[j]] -= ns.step[j] * ns.cnt[j];
      idx[j] = 0;
    }
    if (j < 0) break;
  }
  if (axis & 1)
    part[i] = acc;
  else
    op.store(i, acc);  // i == (oh * oW + ow) * C + c
}

template <class Op>
__global__ void lanes_kernel(int64_t n, Op op, const typename Op::Acc* part) {
  NPU_TID;
  op.finish(i, part + i * kLanes);
}

bool valid_axis(int axis) { return axis >= 1 && axis <= 7; }

template <class Op>
int run_chain(void* stream, const Op& op, const Nest& ns, int64_t H, int64_t W, int64_t C,
              int axis) {
  const int64_t oH = (axis & 4) ? 1 : H, oW = (axis & 2) ? 1 : W;
  if (!(axis & 1))
    return launch(stream, oH * oW * C, chain_kernel<Op>, op, ns, oW, C, axis,
                  static_cast<typename Op::Acc*>(nullptr));
  const int64_t nout = oH * oW;
  hipStream_t s = static_cast<hipStream_t>(stream);
  typename Op::Acc* part = nullptr;
  (void)hipGetLastError();
  int rc = int(hipMallocAsync(reinterpret_cast<void**>(&part),
                              size_t(nout * kLanes) * sizeof(typename Op::Acc), s));
  if (rc) return rc;
  rc = launch(stream, nout * kLanes, chain_kernel<Op>, op, ns, oW, int64_t(kLanes), axis, part);
  if (!rc) rc = launch(stream, nout, lanes_kernel<Op>, op, static_cast<const typename Op::Acc*>(part));
  const int rc2 = int(hipFreeAsync(part, s));
  return rc ? rc : rc2;
}

template <class Op>
int run_order(void* stream, const Op& op, const char* const* orders, int64_t H, int64_t W,
              int64_t C, int axis) {
  return run_chain(stream, op, make_nest(orders[axis], H, W, (C + kLanes - 1) / kLanes), H, W,
                   C, axis);
}

struct Src16 {
  const void* p;
  npu_dtype t;
  __device__ float operator()(int64_t i) const { return load_float(p, t, i); }
};

// Integer tensor dequantized and rounded to the reduction type.
struct SrcDq {
  const void* p;
  npu_dtype it, rt;
  float scale, zp;  // zp = dq_offset(zero_point, scale)
  __device__ float operator()(int64_t i) const {
    return round_float(dequant(load_int(p, it, i), scale, zp), rt, kHalfEven);
  }
};

struct Sink16 {
  void* p;
  npu_dtype t;
  __device__ void operator()(int64_t i, float v) const { store_float(p, t, i, v, kHalfEven); }
};

// Quantize: fma(y, scale, zero_point), half to even, saturate.
struct SinkQ {
  void* p;
  npu_dtype ot;
  float scale, zp;
  __device__ void operator()(int64_t i, float v) const {
    store_int(p, ot, i, quant(v, scale, zp, ot));
  }
};

struct SumF16Op {
  using Acc = float;
  const void* in;
  npu_dtype t;
  void* out;
  float scale;
  int64_t W, C;
  __device__ float init() const { return 0.0f; }
  __device__ float step(float acc, int64_t h, int64_t w, int64_t c, bool first) const {
    const float x = c < C ? load_float(in, t, (h * W + w) * C + c) : 0.0f;
    // The first step is the bare product (no add of the initial 0).
    return first ? mul_q(x, scale) : fma_q(x, scale, acc);
  }
  __device__ void store(int64_t i, float acc) const { store_float(out, t, i, acc, kHalfEven); }
  __device__ void finish(int64_t i, const float* v) const {
    store_float(out, t, i, lane_reduce(v), kHalfEven);
  }
};

struct SumFp8Op {
  using Acc = float;
  const uint8_t* in;
  uint8_t* out;
  float scale;
  int64_t W, C;
  __device__ float init() const { return 0.0f; }
  __device__ float step(float acc, int64_t h, int64_t w, int64_t c, bool) const {
    const float x = c < C ? fp8_decode_identity(in + (h * W + w) * C, c, C) : 0.0f;
    return fma_q(x, scale, acc);
  }
  // Saturated to +-448 by the conversion; a NaN passes through.
  __device__ void store(int64_t i, float acc) const { out[i] = f32_to_fp8(acc, kHalfEven, kSat); }
  __device__ void finish(int64_t i, const float* v) const {
    out[i] = f32_to_fp8(lane_reduce(v), kHalfEven, kSat);
  }
};

template <class Src, class Sink>
struct ProdOp {
  using Acc = float;
  Src src;
  Sink sink;
  npu_dtype rt;  // the 16-bit type the products round to
  int64_t W, C;
  __device__ float init() const { return 1.0f; }
  __device__ float step(float acc, int64_t h, int64_t w, int64_t c, bool) const {
    return mul16_f32(acc, src((h * W + w) * C + c), rt);
  }
  __device__ void store(int64_t i, float acc) const { sink(i, acc); }
  // Tree: v[k] *= v[k + s] for s = 32, 16, ..., 1.
  __device__ void finish(int64_t i, const float* v) const {
    float t[kLanes];
    for (int k = 0; k < kLanes; ++k) t[k] = v[k];
    for (int s = kLanes / 2; s >= 1; s /= 2)
      for (int k = 0; k < s; ++k) t[k] = mul16_f32(t[k], t[k + s], rt);
    sink(i, t[0]);
  }
};

template <class Src, class Sink>
struct L2Op {
  using Acc = float;
  Src src;
  Sink sink;
  npu_dtype rt;
  int64_t W, C;
  __device__ float init() const { return 0.0f; }
  __device__ float step(float acc, int64_t h, int64_t w, int64_t c, bool) const {
    const float x = src((h * W + w) * C + c);
    return fma_q(x, x, acc);
  }
  // Without C reduction: rounded to 16 bits, then the 16-bit sqrt table.
  // With it: fp32 sqrt of the lane sum.
  __device__ void store(int64_t i, float acc) const { sink(i, sqrt16(acc, rt)); }
  __device__ void finish(int64_t i, const float* v) const {
    sink(i, round_float(sqrt_q(lane_reduce(v)), rt, kHalfEven));
  }
};

template <bool A16>
struct MrsOp {
  using Acc = float;
  const void* x1;
  const void* x2;
  npu_dtype it, ot;
  void* out;
  int64_t W, C, H2, W2, C2;
  float c0, c1, c2, c3;
  __device__ float init() const { return 0.0f; }
  __device__ float step(float acc, int64_t h, int64_t w, int64_t c, bool) const {
    // Channels past C are 0 in both inputs and still go through the steps.
    const int64_t i1 = c < C ? load_int(x1, it, (h * W + w) * C + c) : 0;
    const int64_t i2 =
        c < C ? load_int(x2, it,
                         ((H2 == 1 ? 0 : h) * W2 + (W2 == 1 ? 0 : w)) * C2 + (C2 == 1 ? 0 : c))
              : 0;
    const float f1 = float(i1), f2 = float(i2);  // exact (16-bit)
    if (A16) {
      // The exact integer product rounded once.
      const float p = int_f32(i1 * i2);
      float r = mul_q(p, c0);
      r = fma_q(f1, c1, r);
      r = fma_q(f2, c2, r);
      return add_q(acc, r);
    }
    const float t = fma_q(f2, c0, c1);
    acc = fma_q(f2, c2, acc);
    return fma_q(f1, t, acc);
  }
  __device__ void put(int64_t i, float v) const {
    store_int(out, ot, i, float_to_int(v, ot, 0, kHalfEven));
  }
  __device__ void store(int64_t i, float acc) const { put(i, add_q(acc, c3)); }
  __device__ void finish(int64_t i, const float* v) const { put(i, add_q(lane_reduce(v), c3)); }
};

// Integer sums: exact int64 lane sums; the int32 wrap is applied to the total.
enum IntKind { kIntA8 = 0, kIntA16 = 1, kIntA32 = 2 };

struct IntSumOp {
  using Acc = int64_t;
  const void* in;
  npu_dtype it, ot;
  void* out;
  int64_t W, C;
  int kind, axis;
  int c0, c1;     // a8 / a16
  int64_t c2;     // a8 / a16
  float f0, f1;   // a32: c0, c1
  int shift;      // a32: shift_res
  __device__ int64_t init() const { return 0; }
  __device__ int64_t step(int64_t acc, int64_t h, int64_t w, int64_t c, bool) const {
    return c < C ? acc + load_int(in, it, (h * W + w) * C + c) : acc;
  }
  __device__ int64_t result(int64_t S) const {
    if (kind == kIntA32)
      return requant(wrap(S, it), f1, f0, ot, -shift);
    int64_t v;
    if (axis == 7)  // the affine map is applied in int64, wrapping on overflow [default]
      v = wrap_add64(wrap_mul64(c0, kind == kIntA8 ? wrap(S, NPU_I32) : S), c2);
    else            // int32 lanes start at (int32)c2 and wrap
      v = wrap(wrap(c2, NPU_I32) + int64_t(c0) * S, NPU_I32);
    return srs(v, c1, ot);  // c1 in 0..62
  }
  __device__ void store(int64_t i, int64_t acc) const { store_int(out, ot, i, result(acc)); }
  __device__ void finish(int64_t i, const int64_t* v) const {
    int64_t S = 0;
    for (int k = 0; k < kLanes; ++k) S += v[k];
    store_int(out, ot, i, result(S));
  }
};

// One thread per line (a column for axis 0, a row for axis 1).
__global__ void cumsum_kernel(int64_t n, const void* in, npu_dtype it, void* out, npu_dtype ot,
                              int64_t len, int64_t stride, int64_t line_stride, bool exclusive,
                              bool reverse, float c0, float c1) {
  NPU_TID;
  int64_t s = 0;  // running sum, int32 wrap
  for (int64_t k = 0; k < len; ++k) {
    const int64_t j = i * line_stride + (reverse ? len - 1 - k : k) * stride;
    const int64_t prev = s;
    s = wrap(s + load_int(in, it, j), NPU_I32);
    const int64_t v = exclusive ? prev : s;
    store_int(out, ot, j, requant(v, c1, c0, ot));
  }
}

// A reduced dimension needs at least one element; a kept one may be empty (no output).
bool bad_shape(int64_t H, int64_t W, int64_t C, int axis) {
  const auto bad = [&](int64_t d, int bit) { return d < ((axis & bit) ? 1 : 0); };
  return !valid_axis(axis) || bad(H, 4) || bad(W, 2) || bad(C, 1);
}

// True when the output has no element (some kept dimension is 0).
bool no_output(int64_t H, int64_t W, int64_t C) { return H == 0 || W == 0 || C == 0; }

// Product / L2: whole groups of kLanes channels, at least one, also when C is kept.
bool lanes_ok(int64_t C) { return C >= 1 && C % kLanes == 0; }

// prod_orders: the step nests of the product variant.
template <class Src, class Sink>
int prod_or_l2(void* stream, bool prod, const char* const* prod_orders, Src src, Sink sink,
               npu_dtype compute_type, int64_t H, int64_t W, int64_t C, int axis) {
  if (prod)
    return run_order(stream, ProdOp<Src, Sink>{src, sink, compute_type, W, C}, prod_orders, H, W,
                     C, axis);
  return run_order(stream, L2Op<Src, Sink>{src, sink, compute_type, W, C}, kOrderL2, H, W, C,
                   axis);
}

int qdq_prod_l2(bool prod, int bits, void* stream, const void* in, npu_dtype in_type, void* out,
                npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis,
                npu_dtype compute_type, float dq_zp, float dq_sc, float q_zp, float q_sc) {
  if (any_null(in, out) || bad_shape(H, W, C, axis) || !lanes_ok(C) ||
      !is_int_bits(in_type, bits) || !is_int_bits(out_type, bits) || !is_half_float(compute_type))
    return -1;
  if (no_output(H, W, C)) return 0;
  return prod_or_l2(stream, prod, kOrderQdqProd,
                    SrcDq{in, in_type, compute_type, dq_sc, dq_offset(dq_zp, dq_sc)},
                    SinkQ{out, out_type, q_sc, q_zp}, compute_type, H, W, C, axis);
}

int int_sum(int kind, void* stream, const void* in, npu_dtype in_type, void* out,
            npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis, int c0, int c1,
            int64_t c2, float f0, float f1, int shift) {
  if (any_null(in, out) || bad_shape(H, W, C, axis)) return -1;
  if (no_output(H, W, C)) return 0;
  IntSumOp op{in, in_type, out_type, out, W, C, kind, axis, c0, c1, c2, f0, f1, shift};
  return run_order(stream, op, kOrderSum, H, W, C, axis);
}

template <bool A16>
int mrs(void* stream, const void* x1, const void* x2, npu_dtype in_type, void* out,
        npu_dtype out_type, int64_t H, int64_t W, int64_t C, int64_t H2, int64_t W2, int64_t C2,
        int axis, float c0, float c1, float c2, float c3) {
  const bool ok_t =
      A16 ? is_i16(in_type) && is_i16(out_type) : is_i8(in_type) && is_i8(out_type);
  auto bc = [](int64_t b, int64_t a) { return b == 1 || b == a; };
  if (any_null(x1, x2, out) || bad_shape(H, W, C, axis) || !ok_t || !bc(H2, H) || !bc(W2, W) ||
      !bc(C2, C))
    return -1;
  if (no_output(H, W, C)) return 0;
  MrsOp<A16> op{x1, x2, in_type, out_type, out, W, C, H2, W2, C2, c0, c1, c2, c3};
  return run_order(stream, op, kOrderMrs, H, W, C, axis);
}

}  // namespace

extern "C" {

int npu_reduce_sum_c64_bf16(void* stream, const void* in, npu_dtype in_type, void* out,
                            int64_t H, int64_t W, int64_t C, int axis_flags, float scale) {
  if (any_null(in, out) || bad_shape(H, W, C, axis_flags) || !is_half_float(in_type)) return -1;
  if (no_output(H, W, C)) return 0;
  return run_order(stream, SumF16Op{in, in_type, out, scale, W, C}, kOrderSum, H, W, C,
                   axis_flags);
}

int npu_reduce_sum_c64_fp8(void* stream, const void* in, void* out, int64_t H, int64_t W,
                           int64_t C, int64_t h_tiles, int64_t w_tiles, int axis_flags,
                           float scale) {
  if (any_null(in, out) || bad_shape(H, W, C, axis_flags)) return -1;
  if (axis_flags == 7 && (h_tiles <= 0 || w_tiles <= 0 || H % h_tiles || W % w_tiles)) return -1;
  if (no_output(H, W, C)) return 0;
  const SumFp8Op op{static_cast<const uint8_t*>(in), static_cast<uint8_t*>(out), scale, W, C};
  if (axis_flags != 7) return run_order(stream, op, kOrderSum, H, W, C, axis_flags);
  // W blocks inside H blocks, then g, h, w inside each block.
  const int64_t th = H / h_tiles, tw = W / w_tiles;
  Nest ns{};
  nest_push(&ns, kDimH, h_tiles, th);
  nest_push(&ns, kDimW, w_tiles, tw);
  nest_push(&ns, kDimG, (C + kLanes - 1) / kLanes, 1);
  nest_push(&ns, kDimH, th, 1);
  nest_push(&ns, kDimW, tw, 1);
  return run_chain(stream, op, ns, H, W, C, axis_flags);
}

int npu_reduce_sum_c64_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                          npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis_flags,
                          int c0, int c1, int64_t c2) {
  if (!is_i8(in_type) || !is_i8(out_type) || c0 < 0 || c0 > 127 || c1 < 0 || c1 > 62) return -1;
  return int_sum(kIntA8, stream, in, in_type, out, out_type, H, W, C, axis_flags, c0, c1, c2, 0,
                 0, 0);
}

int npu_reduce_sum_c64_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis_flags,
                           int c0, int c1, int64_t c2) {
  if (!is_i16(in_type) || !is_i16(out_type) || c0 < 0 || c0 > 255 || c1 < 0 || c1 > 62) return -1;
  return int_sum(kIntA16, stream, in, in_type, out, out_type, H, W, C, axis_flags, c0, c1, c2, 0,
                 0, 0);
}

// The kernel reads the low 16 bits of shift_res and shifts by their negation, which the
// hardware takes modulo its shift range: a negative shift_res shifts left.
int npu_qdq_reduce_sum_c64_a32(void* stream, const void* in, npu_dtype in_type, void* out,
                               npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis_flags, float c0,
                               float c1, int16_t shift_res) {
  if (!is_32(in_type) || !is_32(out_type) || shift_res < -31 || shift_res > 31) return -1;
  return int_sum(kIntA32, stream, in, in_type, out, out_type, H, W, C, axis_flags, 0, 0, 0, c0,
                 c1, shift_res);
}

int npu_reduce_prod_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t H,
                         int64_t W, int64_t C, int axis_flags) {
  if (any_null(in, out) || bad_shape(H, W, C, axis_flags) || !lanes_ok(C) ||
      !is_half_float(in_type))
    return -1;
  if (no_output(H, W, C)) return 0;
  return prod_or_l2(stream, true, kOrderProd, Src16{in, in_type}, Sink16{out, in_type}, in_type,
                    H, W, C, axis_flags);
}

int npu_reduce_l2_bf16(void* stream, const void* in, npu_dtype in_type, void* out, int64_t H,
                       int64_t W, int64_t C, int axis_flags) {
  if (any_null(in, out) || bad_shape(H, W, C, axis_flags) || !lanes_ok(C) ||
      !is_half_float(in_type))
    return -1;
  if (no_output(H, W, C)) return 0;
  return prod_or_l2(stream, false, nullptr, Src16{in, in_type}, Sink16{out, in_type}, in_type,
                    H, W, C, axis_flags);
}

int npu_qdq_reduce_prod_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                           npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis_flags,
                           float dq_zp, float dq_sc, float q_zp, float q_sc) {
  return qdq_prod_l2(true, 8, stream, in, in_type, out, out_type, H, W, C, axis_flags, NPU_BF16,
                     dq_zp, dq_sc, q_zp, q_sc);
}
int npu_qdq_reduce_prod_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                            npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis_flags,
                            float dq_zp, float dq_sc, float q_zp, float q_sc) {
  return qdq_prod_l2(true, 16, stream, in, in_type, out, out_type, H, W, C, axis_flags, NPU_BF16,
                     dq_zp, dq_sc, q_zp, q_sc);
}
int npu_qdq_reduce_l2_a8(void* stream, const void* in, npu_dtype in_type, void* out,
                         npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis_flags,
                         npu_dtype compute_type, float dq_zp, float dq_sc, float q_zp,
                         float q_sc) {
  return qdq_prod_l2(false, 8, stream, in, in_type, out, out_type, H, W, C, axis_flags,
                     compute_type, dq_zp, dq_sc, q_zp, q_sc);
}
int npu_qdq_reduce_l2_a16(void* stream, const void* in, npu_dtype in_type, void* out,
                          npu_dtype out_type, int64_t H, int64_t W, int64_t C, int axis_flags,
                          npu_dtype compute_type, float dq_zp, float dq_sc, float q_zp,
                          float q_sc) {
  return qdq_prod_l2(false, 16, stream, in, in_type, out, out_type, H, W, C, axis_flags,
                     compute_type, dq_zp, dq_sc, q_zp, q_sc);
}

int npu_qdq_elementwise_broadcasting3d_mul_reducesum_a8(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    npu_dtype out_type, int64_t H, int64_t W, int64_t C, int64_t H2, int64_t W2, int64_t C2,
    int axis_flags, float c0, float c1, float c2, float c3) {
  return mrs<false>(stream, ifm1, ifm2, in_type, out, out_type, H, W, C, H2, W2, C2, axis_flags,
                    c0, c1, c2, c3);
}
int npu_qdq_elementwise_broadcasting3d_mul_reducesum_a16(
    void* stream, const void* ifm1, const void* ifm2, npu_dtype in_type, void* out,
    npu_dtype out_type, int64_t H, int64_t W, int64_t C, int64_t H2, int64_t W2, int64_t C2,
    int axis_flags, float c0, float c1, float c2, float c3) {
  return mrs<true>(stream, ifm1, ifm2, in_type, out, out_type, H, W, C, H2, W2, C2, axis_flags,
                   c0, c1, c2, c3);
}

int npu_qdq_cumsum(void* stream, const void* in, npu_dtype in_type, void* out,
                   npu_dtype out_type, int64_t H, int64_t W, int axis, int exclusive, int reverse,
                   float c0, float c1) {
  const bool ok_t = (is_i8(in_type) && is_i8(out_type)) || (is_i16(in_type) && is_i16(out_type));
  // The scanned axis needs at least one element; the other one may be empty.
  if (any_null(in, out) || H < (axis == 0 ? 1 : 0) || W < (axis == 1 ? 1 : 0) ||
      (axis != 0 && axis != 1) || !ok_t)
    return -1;
  if (H == 0 || W == 0) return 0;
  if (axis == 0)  // W columns of length H
    return launch(stream, W, cumsum_kernel, in, in_type, out, out_type, H, W, int64_t(1),
                  exclusive != 0, reverse != 0, c0, c1);
  return launch(stream, H, cumsum_kernel, in, in_type, out, out_type, W, int64_t(1), W,
                exclusive != 0, reverse != 0, c0, c1);
}

}  // extern "C"
