// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
//
// GT4 brightness and fade patterns: classifies a run of channel-shuffle batches into the two shapes
// GT4 uses (an RGB tone curve across three channels, or an alpha stash into several targets).
//
// This is a profile-side classifier. The renderer does not call it; a host may use it to decide
// which effect rules to enable for a scene, and it is the worked example of "a profile recognises a
// game's pass sequence" in docs/PROFILES.md.
#pragma once

#include <cstdint>
#include <vector>

namespace coregs::gt4 {

enum class TonePattern : std::uint8_t { none, rgb_tone_curve, alpha_stash };

struct PatternBatch {
  std::uint32_t fbmsk = 0;
  bool alpha_only = false;
  int source_channel = 0;  // 0 R, 1 G, 2 B
  std::uint32_t stash_fbp = 0;
};

// An RGB tone curve writes each of the three colour lanes in turn (FBMSK masks the others); an alpha
// stash writes only the alpha lane into three or more targets.
TonePattern ClassifyTonePattern(const std::vector<PatternBatch> &run);

}  // namespace coregs::gt4
