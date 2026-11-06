// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.

#include "patterns.h"

#include <set>

namespace coregs::gt4 {

TonePattern ClassifyTonePattern(const std::vector<PatternBatch> &run) {
  if (run.empty()) return TonePattern::none;
  std::set<int> rgb_channels;
  std::set<std::uint32_t> stash_blocks;
  bool all_alpha_stash = true;
  for (const PatternBatch &batch : run) {
    if (batch.fbmsk != 0x00FFFFFFu || !batch.alpha_only) all_alpha_stash = false;
    if (batch.fbmsk != 0x00FFFFFFu && !batch.alpha_only) rgb_channels.insert(batch.source_channel);
    if (batch.fbmsk == 0x00FFFFFFu && batch.alpha_only) stash_blocks.insert(batch.stash_fbp);
  }
  if (rgb_channels.size() >= 3) return TonePattern::rgb_tone_curve;
  if (all_alpha_stash && stash_blocks.size() >= 3) return TonePattern::alpha_stash;
  return TonePattern::none;
}

}  // namespace coregs::gt4
