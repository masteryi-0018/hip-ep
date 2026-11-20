/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Host side of nlf.h: the exception bitmaps of the fp32 reciprocal / inverse
 * square root (base64 .lzma in the source, decoded and uploaded on first use)
 * and the test-only entry points.
 */
#include <mutex>
#include <vector>

#include "common/launch.h"
#include "common/nlf.h"
#include "common/nlf_tables.h"

namespace npu_emu {
namespace {

// ---- base64 ----

bool base64_decode(const char* s, std::vector<uint8_t>* out) {
  auto val = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  uint32_t acc = 0;
  int bits = 0;
  for (; *s && *s != '='; ++s) {
    const int v = val(*s);
    if (v < 0) return false;
    acc = (acc << 6) | uint32_t(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out->push_back(uint8_t(acc >> bits));
    }
  }
  return true;
}

// ---- LZMA (.lzma stream) decoder. The whole output stays in memory, so the
// dictionary is the output buffer itself.

struct RangeDecoder {
  const uint8_t* p;
  const uint8_t* end;
  uint32_t range = 0xffffffffu, code = 0;
  bool bad = false;

  uint8_t next() {
    if (p < end) return *p++;
    bad = true;
    return 0;
  }
  bool init() {
    const bool first_zero = next() == 0;
    for (int i = 0; i < 4; ++i) code = (code << 8) | next();
    return first_zero && code != range && !bad;
  }
  void normalize() {
    if (range < (1u << 24)) {
      range <<= 8;
      code = (code << 8) | next();
    }
  }
  uint32_t direct(int n) {
    uint32_t res = 0;
    do {
      range >>= 1;
      code -= range;
      const uint32_t t = 0u - (code >> 31);
      code += range & t;
      if (code == range) bad = true;
      normalize();
      res = (res << 1) + (t + 1);
    } while (--n);
    return res;
  }
  unsigned bit(uint16_t* prob) {
    uint32_t v = *prob;
    const uint32_t bound = (range >> 11) * v;
    unsigned s;
    if (code < bound) {
      v += ((1u << 11) - v) >> 5;
      range = bound;
      s = 0;
    } else {
      v -= v >> 5;
      code -= bound;
      range -= bound;
      s = 1;
    }
    *prob = uint16_t(v);
    normalize();
    return s;
  }
  unsigned tree(uint16_t* probs, int nbits) {
    unsigned m = 1;
    for (int i = 0; i < nbits; ++i) m = (m << 1) + bit(&probs[m]);
    return m - (1u << nbits);
  }
  unsigned tree_reverse(uint16_t* probs, int nbits) {
    unsigned m = 1, s = 0;
    for (int i = 0; i < nbits; ++i) {
      const unsigned b = bit(&probs[m]);
      m = (m << 1) + b;
      s |= b << i;
    }
    return s;
  }
};

constexpr uint16_t kProbInit = 1024;

struct LenDecoder {
  uint16_t choice = kProbInit, choice2 = kProbInit;
  uint16_t low[16][8], mid[16][8], high[256];
  LenDecoder() {
    for (auto& r : low) for (auto& p : r) p = kProbInit;
    for (auto& r : mid) for (auto& p : r) p = kProbInit;
    for (auto& p : high) p = kProbInit;
  }
  unsigned decode(RangeDecoder& rc, unsigned pos_state) {
    if (!rc.bit(&choice)) return rc.tree(low[pos_state], 3);
    if (!rc.bit(&choice2)) return 8 + rc.tree(mid[pos_state], 3);
    return 16 + rc.tree(high, 8);
  }
};

bool lzma_decode(const uint8_t* in, size_t n, size_t expect, std::vector<uint8_t>* out) {
  if (n < 13 || in[0] >= 9 * 5 * 5) return false;
  const unsigned lc = in[0] % 9, lp = (in[0] / 9) % 5, pb = in[0] / 45;
  uint64_t size = 0;
  for (int i = 0; i < 8; ++i) size |= uint64_t(in[5 + i]) << (8 * i);
  const bool known = size != ~uint64_t(0);
  if (known && size != expect) return false;
  RangeDecoder rc{in + 13, in + n};
  if (!rc.init()) return false;

  std::vector<uint16_t> lit(size_t(0x300) << (lc + lp), kProbInit);
  uint16_t is_match[12 << 4], is_rep[12], is_rep_g0[12], is_rep_g1[12], is_rep_g2[12],
      is_rep0_long[12 << 4], pos_slot[4][64], pos_dec[1 + 128 - 14], align[16];
  for (uint16_t* t : {is_match, is_rep0_long}) for (int i = 0; i < (12 << 4); ++i) t[i] = kProbInit;
  for (uint16_t* t : {is_rep, is_rep_g0, is_rep_g1, is_rep_g2}) for (int i = 0; i < 12; ++i) t[i] = kProbInit;
  for (auto& r : pos_slot) for (auto& p : r) p = kProbInit;
  for (auto& p : pos_dec) p = kProbInit;
  for (auto& p : align) p = kProbInit;
  LenDecoder len_dec, rep_len_dec;

  std::vector<uint8_t>& o = *out;
  o.clear();
  o.reserve(expect);
  uint32_t rep0 = 0, rep1 = 0, rep2 = 0, rep3 = 0;
  unsigned state = 0;
  const size_t pb_mask = (size_t(1) << pb) - 1, lp_mask = (size_t(1) << lp) - 1;
  while (!rc.bad) {
    if (known && o.size() == size) break;
    if (o.size() > expect) return false;
    const size_t pos = o.size();
    const unsigned ps = unsigned(pos & pb_mask);
    if (!rc.bit(&is_match[(state << 4) + ps])) {  // literal
      const unsigned prev = pos ? o[pos - 1] : 0;
      uint16_t* probs = &lit[size_t(0x300) * (((pos & lp_mask) << lc) + (prev >> (8 - lc)))];
      unsigned sym = 1;
      if (state >= 7) {
        if (rep0 >= pos) return false;
        unsigned match = o[pos - rep0 - 1];
        do {
          const unsigned mb = (match >> 7) & 1;
          match <<= 1;
          const unsigned b = rc.bit(&probs[((1 + mb) << 8) + sym]);
          sym = (sym << 1) | b;
          if (mb != b) break;
        } while (sym < 0x100);
      }
      while (sym < 0x100) sym = (sym << 1) | rc.bit(&probs[sym]);
      o.push_back(uint8_t(sym));
      state = state < 4 ? 0 : state < 10 ? state - 3 : state - 6;
      continue;
    }
    unsigned len;
    if (rc.bit(&is_rep[state])) {
      if (pos == 0) return false;
      if (!rc.bit(&is_rep_g0[state])) {
        if (!rc.bit(&is_rep0_long[(state << 4) + ps])) {  // short rep: one byte
          if (rep0 >= pos) return false;
          state = state < 7 ? 9 : 11;
          o.push_back(o[pos - rep0 - 1]);
          continue;
        }
      } else {
        uint32_t dist;
        if (!rc.bit(&is_rep_g1[state])) {
          dist = rep1;
        } else {
          if (!rc.bit(&is_rep_g2[state])) {
            dist = rep2;
          } else {
            dist = rep3;
            rep3 = rep2;
          }
          rep2 = rep1;
        }
        rep1 = rep0;
        rep0 = dist;
      }
      len = rep_len_dec.decode(rc, ps);
      state = state < 7 ? 8 : 11;
    } else {
      rep3 = rep2;
      rep2 = rep1;
      rep1 = rep0;
      len = len_dec.decode(rc, ps);
      state = state < 7 ? 7 : 10;
      const unsigned slot = rc.tree(pos_slot[len < 3 ? len : 3], 6);
      if (slot < 4) {
        rep0 = slot;
      } else {
        const int nd = int(slot >> 1) - 1;
        uint32_t dist = (2u | (slot & 1)) << nd;
        if (slot < 14) {
          dist += rc.tree_reverse(&pos_dec[dist - slot], nd);
        } else {
          dist += rc.direct(nd - 4) << 4;
          dist += rc.tree_reverse(align, 4);
        }
        rep0 = dist;
      }
      if (rep0 == 0xffffffffu) break;  // end marker
    }
    len += 2;
    if (rep0 >= pos || o.size() + len > expect) return false;
    for (unsigned k = 0; k < len; ++k) o.push_back(o[o.size() - rep0 - 1]);
  }
  return !rc.bad && o.size() == expect;
}

// ---- per-device cache ----

struct TableSrc {
  const char* b64;
  size_t bytes;
};
const TableSrc kTables[] = {{kInvF32ExcLzma, kInvF32ExcBytes},
                            {kInvSqrtF32ExcLzma, kInvSqrtF32ExcBytes}};
constexpr int kNumTables = sizeof(kTables) / sizeof(kTables[0]);
constexpr int kMaxDevices = 64;

std::mutex g_mu;
const uint32_t* g_dev[kNumTables][kMaxDevices];

int upload(int id, const uint32_t** dev_ptr) {
  std::vector<uint8_t> packed, raw;
  if (!base64_decode(kTables[id].b64, &packed) ||
      !lzma_decode(packed.data(), packed.size(), kTables[id].bytes, &raw))
    return int(hipErrorInvalidImage);
  void* p = nullptr;
  hipError_t rc = hipMalloc(&p, raw.size());
  if (rc != hipSuccess) return int(rc);
  rc = hipMemcpy(p, raw.data(), raw.size(), hipMemcpyHostToDevice);
  if (rc != hipSuccess) {
    (void)hipFree(p);
    return int(rc);
  }
  *dev_ptr = static_cast<const uint32_t*>(p);
  return 0;
}

}  // namespace
}  // namespace npu_emu

using namespace npu_emu;

extern "C" int npu_emu_nlf_table(int id, const uint32_t** dev_ptr) {
  if (id < 0 || id >= kNumTables || !dev_ptr) return int(hipErrorInvalidValue);
  int dev = 0;
  hipError_t rc = hipGetDevice(&dev);
  if (rc != hipSuccess) return int(rc);
  if (dev < 0 || dev >= kMaxDevices) return int(hipErrorInvalidDevice);
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_dev[id][dev]) {
    const int r = upload(id, &g_dev[id][dev]);
    if (r) return r;
  }
  *dev_ptr = g_dev[id][dev];
  return 0;
}
