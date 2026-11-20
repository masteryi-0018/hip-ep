/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Quantized convolution into a scratch tensor, then the quantized instance
 * norm on it.
 */
#include <hip/hip_runtime.h>

#include "common/npu_emu.h"
#include "matmul/conv_instancenorm.h"
#include "matmul/qdq_conv.h"
#include "reduction/groupnorm_device.h"

using namespace npu_emu;

namespace {

// Block plan of the statistics over an ho x wo image.
bool conv_plan(int split_mode, int64_t ho, int64_t wo, int64_t sv_h, int64_t sv_w,
               int64_t h_loop, int64_t w_loop, Plan* p) {
  if (sv_h <= 0 || sv_w <= 0 || h_loop <= 0 || w_loop <= 0 || ho % sv_h || wo % sv_w)
    return false;
  *p = Plan{};
  p->T = int(h_loop * w_loop), p->w_loop = int(w_loop), p->jbh = 1;
  p->nbw = wo / sv_w, p->nbh = ho / sv_h, p->sv_h = sv_h, p->sv_w = sv_w, p->W = wo;
  if (split_mode == 1) {
    const int64_t band = ho / 4;
    if (ho % 4 || band % sv_h || h_loop * sv_h > band || w_loop * sv_w > wo) return false;
    p->ncol = 1, p->nrow = 4, p->kbh = band / sv_h;
    return true;
  }
  if (split_mode == 0) {
    const int64_t cw = wo / 3;
    if (wo % 3 || cw % 4 || (cw / 4) % sv_w || cw % sv_w || h_loop * sv_h > ho ||
        w_loop * sv_w > cw / 4)
      return false;
    p->ncol = 3, p->nrow = 4, p->cbw = cw / sv_w, p->kbw = (cw / 4) / sv_w;
    return true;
  }
  return false;
}

// Runs the conv into a scratch tensor, then `norm` on it.
template <typename F>
int conv_then(void* stream, const void* conv_ifm, npu_dtype ifm_type, const void* wts,
              npu_dtype wts_type, int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw,
              int64_t co, int stride_h, int stride_w, const float* c0, const float* c1,
              const float* c2, npu_activation conv_act, float act0, float act1,
              npu_dtype conv_out_type, F norm) {
  if (!(conv_out_type == NPU_I16 || conv_out_type == NPU_U16) || stride_h <= 0 ||
      stride_w <= 0 || h < kh || w < kw)
    return -1;
  const int64_t ho = (h - kh) / stride_h + 1, wo = (w - kw) / stride_w + 1;
  hipStream_t st = static_cast<hipStream_t>(stream);
  void* mid = nullptr;
  if (int rc = int(hipMallocAsync(&mid, size_t(ho * wo * co) * 2, st))) return rc;
  int rc = npu_conv_qdq_int16x8(stream, conv_ifm, ifm_type, wts, wts_type, mid, conv_out_type,
                                h, w, ci, kh, kw, co, stride_h, stride_w, 0, c0, c1, c2,
                                conv_act, act0, act1, nullptr);
  if (!rc) rc = norm(static_cast<const void*>(mid), ho, wo);
  const int rc2 = int(hipFreeAsync(mid, st));
  return rc ? rc : rc2;
}

}  // namespace

extern "C" {

int npu_qdq_ConvInFusion_A16(void* stream, const void* conv_ifm, npu_dtype ifm_type,
                             const void* wts, npu_dtype wts_type, int64_t h, int64_t w,
                             int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
                             int stride_w, const float* c0, const float* c1, const float* c2,
                             npu_activation conv_act, float act0, float act1,
                             npu_dtype conv_out_type, const void* alpha, const void* beta,
                             npu_dtype ab_type, void* ofm, npu_dtype compute_type,
                             int split_mode, int64_t sv_h, int64_t sv_w, int64_t h_loop,
                             int64_t w_loop, int64_t spatial_pad, float dq_scale, float dq_zp,
                             float q_scale, float q_zp, int activation, float slope) {
  return conv_then(stream, conv_ifm, ifm_type, wts, wts_type, h, w, ci, kh, kw, co, stride_h,
                   stride_w, c0, c1, c2, conv_act, act0, act1, conv_out_type,
                   [&](const void* mid, int64_t ho, int64_t wo) {
                     Plan p;
                     if (!conv_plan(split_mode, ho, wo, sv_h, sv_w, h_loop, w_loop, &p))
                       return -1;
                     const int64_t N =
                         int64_t(p.ncol) * p.nrow * p.T * sv_h * sv_w - spatial_pad;
                     return in_qdq(stream, mid, conv_out_type, alpha, beta, ab_type, nullptr,
                                   nullptr, nullptr, false, ofm, nullptr, ho * wo, co,
                                   compute_type, &p, nullptr, nullptr, N, dq_scale, dq_zp,
                                   q_scale, q_zp, 0.0f, 0.0f, 0.0f, 0.0f, activation, slope);
                   });
}

int npu_qdq_ConvInstanceNormAddMulAdd(
    void* stream, const void* conv_ifm, npu_dtype ifm_type, const void* wts, npu_dtype wts_type,
    int64_t h, int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co, int stride_h,
    int stride_w, const float* c0, const float* c1, const float* c2, npu_activation conv_act,
    float act0, float act1, npu_dtype conv_out_type, const void* alpha, const void* beta,
    npu_dtype ab_type, const void* add1_act, const void* add2, const void* add1_const,
    void* ofm, npu_dtype compute_type, int split_mode, int64_t sv_h, int64_t sv_w,
    int64_t h_loop, int64_t w_loop, int64_t spatial_pad, float dq_scale, float dq_zp,
    float q_scale, float q_zp, float dq2_scale, float dq2_zp, int activation, float slope) {
  return conv_then(stream, conv_ifm, ifm_type, wts, wts_type, h, w, ci, kh, kw, co, stride_h,
                   stride_w, c0, c1, c2, conv_act, act0, act1, conv_out_type,
                   [&](const void* mid, int64_t ho, int64_t wo) {
                     Plan p;
                     if (!conv_plan(split_mode, ho, wo, sv_h, sv_w, h_loop, w_loop, &p))
                       return -1;
                     const int64_t N =
                         int64_t(p.ncol) * p.nrow * p.T * sv_h * sv_w - spatial_pad;
                     return in_qdq(stream, mid, conv_out_type, alpha, beta, ab_type, add1_act,
                                   add2, add1_const, true, ofm, nullptr, ho * wo, co,
                                   compute_type, &p, nullptr, nullptr, N, dq_scale, dq_zp,
                                   q_scale, q_zp, 0.0f, 0.0f, dq2_scale, dq2_zp, activation,
                                   slope);
                   });
}

}  // extern "C"
