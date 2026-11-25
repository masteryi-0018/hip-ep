/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * bf16 multi-head flash attention: one thread runs one query row through all key blocks.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "common/check.h"
#include "common/float_cvt.h"
#include "common/launch.h"
#include "common/mac.h"
#include "common/minmax.h"
#include "common/mmac.h"
#include "common/nlf.h"
#include "common/bits.h"
#include "attention/mha.h"

using namespace npu_emu;

namespace {

constexpr int kSvM = 64;      // query rows per block
constexpr int kMaxSvN = 256;  // largest sv_n
constexpr int kHead = 64;     // supported head_size
constexpr uint16_t kOne = 0x3f80;

struct MhaArgs {
  const uint16_t* q;
  const uint16_t* k;
  const uint16_t* v;
  uint16_t* out;
  const uint32_t* inv_exc;
  int64_t q_heads, group, kv_heads, q_len, kv_len, sv_n;
  int causal;
  uint32_t mask_start;
  float scale;  // bf16(scale * 1.4423828125), as fp32
};

// One thread: query row r of (batch b, head h). The bf16 results of the vector unit round
// toward -inf, fp32 subnormals give a signed zero; fp32 adds flush a subnormal result.
__global__ void mha_bf16_3strms_kernel(int64_t n, MhaArgs a) {
  NPU_TID;
  const int64_t r = i % a.q_len, h = (i / a.q_len) % a.q_heads, b = i / (a.q_len * a.q_heads);
  const int64_t hk = h / a.group;
  const uint16_t* qr = a.q + ((b * a.q_heads + h) * a.q_len + r) * kHead;
  const uint16_t* kh = a.k + (b * a.kv_heads + hk) * a.kv_len * kHead;
  const uint16_t* vh = a.v + (b * a.kv_heads + hk) * a.kv_len * kHead;
  const int sv_n = int(a.sv_n);

  // 1-D mask: column index from the low 16 bits of mask_M_start, block index from all 32
  const int64_t inf_idx = int64_t(a.mask_start & 0xffff) % sv_n;
  const int64_t mix_col = int64_t(a.mask_start) / sv_n;
  // causal: row id compared as uint16
  const uint32_t r_id = (a.mask_start + uint32_t(r)) & 0xffff;

  uint16_t o[kHead];
  for (int d = 0; d < kHead; ++d) o[d] = 0;
  uint16_t mx = 0xff80, l = 0;
  uint16_t p[kMaxSvN];
  uint16_t ones[32];
  for (int j = 0; j < 32; ++j) ones[j] = kOne;

  for (int64_t ci = 0; ci < a.kv_len / sv_n; ++ci) {
    // S, scale + mask, running max
    uint16_t mx_new = mx;
    for (int j = 0; j < sv_n; ++j) {
      const int64_t col = ci * sv_n + j;
      const uint16_t* kr = kh + col * kHead;
      uint32_t acc = 0;
      for (int k0 = 0; k0 < kHead; k0 += 32) acc = mmac_bf16(acc, qr + k0, 1, kr + k0, 1, 32);
      const uint16_t s = f32_to_bf16(bits_f32(acc), kFloor, kFtzIn);
      bool masked;
      if (a.causal) {
        masked = !(r_id >= (uint32_t(col) & 0xffff));
      } else {
        masked = ci < mix_col ? false : (ci == mix_col ? j >= inf_idx : true);
      }
      const float sm = fma_ftz(bf16_to_f32(s), a.scale, masked ? neg_inf() : 0.0f, kFtzOut);
      const uint16_t ms = f32_to_bf16(sm, kFloor, kFtzIn);
      p[j] = ms;
      mx_new = max16(mx_new, ms, NPU_BF16);
    }
    // P = exp2(s - mx_new)
    const float neg_mx = neg_f32(bf16_to_f32(mx_new));
    for (int j = 0; j < sv_n; ++j)
      p[j] = f32_to_bf16(exp2_bf16(fma_ftz(bf16_to_f32(p[j]), 1.0f, neg_mx, kFtzOut)), kHalfEven);
    // row sum: MMAC(P, ones) over 32-column chunks
    uint32_t rs = 0;
    for (int j0 = 0; j0 < sv_n; j0 += 32) rs = mmac_bf16(rs, p + j0, 1, ones, 1, 32);
    const uint16_t rowsum = f32_to_bf16(bits_f32(rs), kFloor, kFtzIn);
    const float diff = fma_ftz(bf16_to_f32(mx), 1.0f, neg_mx, kFtzOut);
    const uint16_t mx_diff = f32_to_bf16(diff, kFloor, kFtzIn);
    const uint16_t mx_exp = f32_to_bf16(exp2_bf16(bf16_to_f32(mx_diff)), kHalfEven);
    mx = mx_new;
    const float lf = fma_ftz(bf16_to_f32(mx_exp), bf16_to_f32(l), bf16_to_f32(rowsum), kFtzOut);
    l = f32_to_bf16(lf, kFloor, kFtzIn);
    // O = MMAC chain over V from bf16(O * mx_exp)
    const uint16_t* vb = vh + ci * sv_n * kHead;
    for (int d = 0; d < kHead; ++d) {
      uint32_t acc = uint32_t(mul16(o[d], mx_exp, NPU_BF16)) << 16;
      for (int j0 = 0; j0 < sv_n; j0 += 32)
        acc = mmac_bf16(acc, p + j0, 1, vb + j0 * kHead + d, kHead, 32);
      o[d] = f32_to_bf16(bits_f32(acc), kFloor, kFtzIn);
    }
  }
  const uint16_t l_inv = f32_to_bf16(hw_inv_f32(bf16_to_f32(l), a.inv_exc), kFloor, kFtzIn);
  uint16_t* orow = a.out + ((b * a.q_len + r) * a.q_heads + h) * kHead;
  for (int d = 0; d < kHead; ++d) orow[d] = mul16(o[d], l_inv, NPU_BF16);
}

}  // namespace

extern "C" {

int npu_mha_bf16_3strms(void* stream, const void* q, const void* k, const void* v, void* out,
                        int64_t batch, int64_t q_heads, int64_t kv_heads,
                        int64_t q_len, int64_t head_size, int64_t kv_len, int64_t sv_n,
                        int mask_1d, uint32_t mask_M_start, uint32_t scaling_factor) {
  if (any_null(q, k, v, out) || batch < 0 || q_heads < 0 || kv_heads < 1 ||
      q_heads % kv_heads != 0 || q_len < 0 || q_len % kSvM != 0 || head_size != kHead ||
      (sv_n != 128 && sv_n != 192 && sv_n != 256) || kv_len < sv_n || kv_len % sv_n != 0 ||
      (mask_1d != 0 && mask_1d != 1))
    return -1;
  if (batch == 0 || q_heads == 0 || q_len == 0) return 0;
  // scalar bf16(fp32(scale * 1.4423828125))
  const float scale_f = __builtin_bit_cast(float, scaling_factor);
  const float sc = scalar_cast(mul_q(scale_f, 1.4423828125f), NPU_BF16, kHalfEven);
  const uint32_t* inv_exc;
  if (int rc = npu_emu_nlf_table(NPU_EMU_NLF_INV_F32, &inv_exc)) return rc;
  MhaArgs a{static_cast<const uint16_t*>(q), static_cast<const uint16_t*>(k),
            static_cast<const uint16_t*>(v), static_cast<uint16_t*>(out), inv_exc, q_heads,
            q_heads / kv_heads, kv_heads, q_len, kv_len, sv_n, mask_1d, mask_M_start, sc};
  return launch(stream, batch * q_heads * q_len, mha_bf16_3strms_kernel, a);
}

}  // extern "C"
