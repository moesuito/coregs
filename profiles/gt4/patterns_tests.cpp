// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
// CTest gt4_patterns: the profile's tone-pattern classifier.

#include "patterns.h"

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char *what) {
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

}  // namespace

int main() {
  using coregs::gt4::ClassifyTonePattern;
  using coregs::gt4::PatternBatch;
  using coregs::gt4::TonePattern;

  std::vector<PatternBatch> rgb;
  for (int channel = 0; channel < 3; ++channel) {
    PatternBatch batch;
    batch.fbmsk = ~(0xFFu << (channel * 8)) & 0xFFFFFFFFu;
    batch.source_channel = channel;
    rgb.push_back(batch);
  }
  check(ClassifyTonePattern(rgb) == TonePattern::rgb_tone_curve, "three colour lanes are the RGB tone curve");

  std::vector<PatternBatch> stash;
  for (std::uint32_t block = 0; block < 3; ++block) {
    PatternBatch batch;
    batch.fbmsk = 0x00FFFFFFu;
    batch.alpha_only = true;
    batch.stash_fbp = block;
    stash.push_back(batch);
  }
  check(ClassifyTonePattern(stash) == TonePattern::alpha_stash, "three alpha targets are the alpha stash");
  check(ClassifyTonePattern({}) == TonePattern::none, "an empty run has no pattern");

  std::vector<PatternBatch> two_lanes = {rgb[0], rgb[1]};
  check(ClassifyTonePattern(two_lanes) == TonePattern::none, "two lanes are not a tone curve");
  std::vector<PatternBatch> two_blocks = {stash[0], stash[1]};
  check(ClassifyTonePattern(two_blocks) == TonePattern::none, "two targets are not an alpha stash");

  if (failures == 0) std::printf("gt4_patterns: all checks passed\n");
  return failures == 0 ? 0 : 1;
}
