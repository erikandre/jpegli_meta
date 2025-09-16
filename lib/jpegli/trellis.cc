// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jpegli/trellis.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "lib/base/bits.h"
#include "lib/jpegli/common_internal.h"
#include "lib/jpegli/encode_internal.h"
#include "lib/jpegli/memory_manager.h"
#include "lib/jpegli/quant.h"

namespace jpegli {
namespace {

constexpr int kNumCoeffs = DCTSIZE2;
constexpr int kNumRuns = 16;
constexpr float kDistortionScale = 1.0f / 16.0f;

struct TrellisNode {
  coeff_t value;
  uint8_t prev_run;
};

float Distortion(coeff_t original, coeff_t candidate, float inv_q) {
  const float delta = static_cast<float>(original - candidate) * inv_q;
  const float scaled = delta * kDistortionScale;
  return scaled * scaled;
}

bool OptimizeBlock(coeff_t* block, const HuffmanCodeTable& ac_code,
                   const float* inv_q, float lambda) {
  std::array<std::array<TrellisNode, kNumRuns>, kNumCoeffs> choices;
  std::array<float, kNumRuns> prev;
  std::array<float, kNumRuns> cur;
  const float inf = std::numeric_limits<float>::infinity();
  prev.fill(inf);
  prev[0] = 0.0f;
  for (int idx = 1; idx < kNumCoeffs; ++idx) {
    cur.fill(inf);
    const coeff_t original = block[idx];
    const float inv_q_val = inv_q[idx];
    for (int run = 0; run < kNumRuns; ++run) {
      const float base_cost = prev[run];
      if (!std::isfinite(base_cost)) continue;

      int next_run = run + 1;
      float rate = 0.0f;
      if (next_run == 16) {
        rate += ac_code.depth[0xF0];
        next_run = 0;
      }
      const float zero_cost =
          base_cost + rate + lambda * Distortion(original, 0, inv_q_val);
      if (zero_cost < cur[next_run]) {
        cur[next_run] = zero_cost;
        choices[idx][next_run] = TrellisNode{0, static_cast<uint8_t>(run)};
      }

      if (original == 0) {
        continue;
      }

      auto evaluate_candidate = [&](coeff_t candidate) {
        int run_length = run;
        float candidate_rate = 0.0f;
        while (run_length >= 16) {
          candidate_rate += ac_code.depth[0xF0];
          run_length -= 16;
        }
        const int magnitude = std::abs(static_cast<int>(candidate));
        if (magnitude == 0) return;
        const int nbits =
            jxl::FloorLog2Nonzero<uint32_t>(static_cast<uint32_t>(magnitude)) + 1;
        const int symbol = (run_length << 4) + nbits;
        candidate_rate += ac_code.depth[symbol];
        const float cost = base_cost + candidate_rate +
                           lambda * Distortion(original, candidate, inv_q_val);
        if (cost < cur[0]) {
          cur[0] = cost;
          choices[idx][0] = TrellisNode{candidate, static_cast<uint8_t>(run)};
        }
      };

      evaluate_candidate(original);
      if (std::abs(static_cast<int>(original)) > 1) {
        const coeff_t candidate =
            original > 0 ? static_cast<coeff_t>(original - 1)
                          : static_cast<coeff_t>(original + 1);
        if (candidate != 0) {
          evaluate_candidate(candidate);
        }
      }
    }
    prev = cur;
  }

  float best_cost = inf;
  int best_run = 0;
  for (int run = 0; run < kNumRuns; ++run) {
    float cost = prev[run];
    if (!std::isfinite(cost)) continue;
    if (run > 0) {
      cost += ac_code.depth[0];
    }
    if (cost < best_cost) {
      best_cost = cost;
      best_run = run;
    }
  }

  bool changed = false;
  int run = best_run;
  for (int idx = kNumCoeffs - 1; idx >= 1; --idx) {
    const TrellisNode& node = choices[idx][run];
    if (block[idx] != node.value) {
      block[idx] = node.value;
      changed = true;
    }
    run = node.prev_run;
  }
  return changed;
}

}  // namespace

bool ApplyTrellisQuantization(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  if (m == nullptr || m->coding_tables == nullptr || m->context_map == nullptr) {
    return false;
  }
  if (cinfo->progressive_mode) {
    return false;
  }
  if (cinfo->num_scans != 1) {
    return false;
  }
  const jpeg_scan_info* scan_info = &cinfo->scan_info[0];
  if (scan_info->Ss != 0 || scan_info->Se != DCTSIZE2 - 1 ||
      scan_info->Ah != 0 || scan_info->Al != 0) {
    return false;
  }

  float distance = QuantValsToDistance(cinfo);
  if (!std::isfinite(distance)) {
    distance = 1.0f;
  }
  const float lambda = 0.8f / (distance + 0.2f);

  bool modified = false;
  std::array<float, kNumCoeffs> inv_q;
  for (int i = 0; i < scan_info->comps_in_scan; ++i) {
    const int comp_idx = scan_info->component_index[i];
    jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
    const float* quant_mul = m->quant_mul[comp_idx];
    for (int k = 0; k < kNumCoeffs; ++k) {
      inv_q[k] = 1.0f / quant_mul[kJPEGNaturalOrder[k]];
    }
    const HuffmanCodeTable& ac_code =
        m->coding_tables[m->context_map[m->ac_ctx_offset[0] + i]];
    for (JDIMENSION by = 0; by < comp->height_in_blocks; ++by) {
      JBLOCKARRAY blocks = GetBlockRow(cinfo, comp_idx, by);
      for (JDIMENSION bx = 0; bx < comp->width_in_blocks; ++bx) {
        coeff_t* block = &blocks[0][bx][0];
        if (OptimizeBlock(block, ac_code, inv_q.data(), lambda)) {
          modified = true;
        }
      }
    }
  }
  return modified;
}

}  // namespace jpegli
