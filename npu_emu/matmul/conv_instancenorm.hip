/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Quantized convolution into a scratch tensor, then the quantized instance
 * norm on it.
 */
#include <hip/hip_runtime.h>

#include "common/act.h"
#include "common/check.h"
#include "common/launch.h"
#include "common/npu_emu.h"
#include "common/quant.h"
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

// The fused pair: the conv has no activation sum and no activation, the norm computes in
// float16 with the leaky ReLU.
struct ConvIn {
  const void *conv_ifm, *wts, *alpha, *beta, *add1_act, *add2, *add1_const;
  npu_dtype ifm_type, wts_type, ofm_type;
  void* ofm;
  int64_t h, w, ci, kh, kw, co;
  int stride_h, stride_w, split_mode;
  int64_t sv_h, sv_w, h_loop, w_loop, spatial_pad;
  const float *c0, *c2;
  float dq_zp, dq_sc, q_zp, q_sc, dq2_zp, dq2_sc, slope;
  bool amad;
};

// Validates every argument, then runs the conv into a scratch tensor and the norm on it.
int conv_in(void* stream, const ConvIn& a) {
  if (any_null(a.conv_ifm, a.wts, a.alpha, a.beta, a.ofm, a.c0, a.c2) ||
      !is_i16(a.ifm_type) || !is_i8(a.wts_type) || !is_i16(a.ofm_type) || a.stride_h < 1 ||
      a.stride_w < 1 || a.kh < 1 || a.kw < 1 || a.ci < 1 || a.co < 1 || a.h < 0 ||
      a.w < 0 || (a.h > 0 && a.h < a.kh) || (a.w > 0 && a.w < a.kw) ||
      (a.amad && any_null(a.add1_act, a.add2, a.add1_const)))
    return -1;
  if (a.h == 0 || a.w == 0) {
    // an empty image: the block parameters are checked without the image they tile
    if (a.sv_h < 1 || a.sv_w < 1 || a.h_loop < 1 || a.w_loop < 1 ||
        (a.split_mode != 0 && a.split_mode != 1) || a.spatial_pad < 0 ||
        a.spatial_pad >= (a.split_mode ? 4 : 12) * a.h_loop * a.w_loop * a.sv_h * a.sv_w)
      return -1;
    return 0;
  }
  const int64_t ho = (a.h - a.kh) / a.stride_h + 1, wo = (a.w - a.kw) / a.stride_w + 1;
  Plan p;
  if (!conv_plan(a.split_mode, ho, wo, a.sv_h, a.sv_w, a.h_loop, a.w_loop, &p)) return -1;
  const int64_t cells = int64_t(p.ncol) * p.nrow * p.T * a.sv_h * a.sv_w;
  if (a.spatial_pad < 0 || a.spatial_pad >= cells) return -1;
  hipStream_t st = static_cast<hipStream_t>(stream);
  void* mid = nullptr;
  if (int rc = int(hipMallocAsync(&mid, size_t(ho * wo * a.co) * 2, st))) return rc;
  int rc = npu_conv_qdq_int16x8(stream, a.conv_ifm, a.ifm_type, a.wts, a.wts_type, mid,
                                a.ofm_type, a.h, a.w, a.ci, a.kh, a.kw, a.co, a.stride_h,
                                a.stride_w, 0, a.c0, nullptr, a.c2, kQdqActNone, 0.0f, 0.0f,
                                nullptr);
  if (!rc)
    rc = in_qdq(stream, mid, a.ofm_type, a.alpha, a.beta, NPU_F16, a.add1_act, a.add2,
                a.add1_const, a.amad, a.ofm, nullptr, ho * wo, a.co, NPU_F16, &p, nullptr,
                nullptr, cells - a.spatial_pad, a.dq_sc, a.dq_zp, a.q_sc, a.q_zp, 0.0f, 0.0f,
                a.dq2_sc, a.dq2_zp, kQdqActLrelu, a.slope);
  const int rc2 = int(hipFreeAsync(mid, st));
  return rc ? rc : rc2;
}

}  // namespace

extern "C" {

int npu_qdq_ConvInFusion_A16(void* stream, const void* conv_ifm, npu_dtype ifm_type,
                             const void* wts, npu_dtype wts_type, const void* alpha,
                             const void* beta, void* ofm, npu_dtype ofm_type, int64_t h,
                             int64_t w, int64_t ci, int64_t kh, int64_t kw, int64_t co,
                             int stride_h, int stride_w, int split_mode, int64_t sv_h,
                             int64_t sv_w, int64_t h_loop, int64_t w_loop, int64_t spatial_pad,
                             const float* c0, const float* c2, float dq_zp, float dq_sc,
                             float q_zp, float q_sc, float slope) {
  const ConvIn a{conv_ifm, wts, alpha, beta, nullptr, nullptr, nullptr, ifm_type, wts_type,
                 ofm_type, ofm, h, w, ci, kh, kw, co, stride_h, stride_w, split_mode, sv_h,
                 sv_w, h_loop, w_loop, spatial_pad, c0, c2, dq_zp, dq_sc, q_zp, q_sc, 0.0f,
                 0.0f, slope, false};
  return conv_in(stream, a);
}

int npu_qdq_ConvInstanceNormAddMulAdd(
    void* stream, const void* conv_ifm, npu_dtype ifm_type, const void* wts, npu_dtype wts_type,
    const void* alpha, const void* beta, const void* add1_act, const void* add2,
    const void* add1_const, void* ofm, npu_dtype ofm_type, int64_t h, int64_t w, int64_t ci,
    int64_t kh, int64_t kw, int64_t co, int stride_h, int stride_w, int split_mode, int64_t sv_h,
    int64_t sv_w, int64_t h_loop, int64_t w_loop, int64_t spatial_pad, const float* c0,
    const float* c2, float dq_zp, float dq_sc, float q_zp, float q_sc, float dq_ifm2_zp,
    float dq_ifm2_sc, float slope) {
  const ConvIn a{conv_ifm, wts, alpha, beta, add1_act, add2, add1_const, ifm_type, wts_type,
                 ofm_type, ofm, h, w, ci, kh, kw, co, stride_h, stride_w, split_mode, sv_h,
                 sv_w, h_loop, w_loop, spatial_pad, c0, c2, dq_zp, dq_sc, q_zp, q_sc,
                 dq_ifm2_zp, dq_ifm2_sc, slope, true};
  return conv_in(stream, a);
}

}  // extern "C"
