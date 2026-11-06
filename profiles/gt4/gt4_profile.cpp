// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
//
// The Gran Turismo 4 profile. See gt4_profile.h and docs/PROFILES.md.

#include "gt4_profile.h"

#include <array>

namespace coregs::gt4 {
namespace {

// One specialised-pipeline key per row (30 state fields, the push word and its ROP/effect word).
constexpr std::uint32_t kKnownVariants[] = {
#include "known_shader_variants.inc"
};

// GT4's night race reads its light-flare visibility table as a 64 x 8 PSMCT32 rectangle (2048 bytes,
// 128 qwords) once per frame. The VU1 scales every light flare and the sun glare by its entries.
bool FlareTable(const GameProfile::ReadbackView &read) {
  return read.psm == 0u && read.width == 64u && read.height == 8u && read.qwords == 128u;
}

// The road sheen: Cs * Ad + Cd over most of a 640-wide (or wider) picture, with no depth test or an
// always-passing one. `area * 5 >= frame * 2` accepts a pass covering 40% of the picture or more.
bool RoadSheen(const GameProfile::EffectPassView &pass) {
  if (!pass.textured || !pass.blends) return false;
  if (pass.blend_a != 0u || pass.blend_b != 2u || pass.blend_c != 1u || pass.blend_d != 1u) return false;
  if (!pass.depth_never && !pass.depth_always) return false;
  if (pass.frame_width < 640u) return false;
  const std::uint64_t frame_area = static_cast<std::uint64_t>(pass.frame_width) * pass.frame_height;
  return pass.covered_area * 5u >= frame_area * 2u;
}

}  // namespace

bool IsLightFlareTableRead(const GameProfile::ReadbackView &read) { return FlareTable(read); }
bool IsRoadSheenPass(const GameProfile::EffectPassView &pass) { return RoadSheen(pass); }

const std::uint32_t *KnownShaderVariants() { return kKnownVariants; }
std::size_t KnownShaderVariantWords() { return std::size(kKnownVariants); }

GameProfile MakeProfile(const Settings &settings) {
  GameProfile profile;
  profile.name = "gt4";
  profile.is_light_flare_table_read = &FlareTable;
  profile.hide_light_flare_table = settings.hide_light_flare_table;
  profile.is_effect_pass = &RoadSheen;
  profile.skip_effect_pass = settings.skip_road_sheen;
  profile.effect_pass_early_out = settings.road_sheen_early_out;
  profile.skip_alpha_only_fills = settings.skip_road_sheen;  // the sheen is the alpha lane's reader
  profile.alpha_only_fill_min_width = 640;
  profile.tone_runs = settings.tone_runs;
  profile.palette_segments = settings.palette_segments;
  profile.effect_targets_at_1x = settings.effect_targets_at_1x;
  if (settings.precompile_known_variants) {
    profile.known_shader_variants = KnownShaderVariants();
    profile.known_shader_variant_words = KnownShaderVariantWords();
  }
  return profile;
}

}  // namespace coregs::gt4
