// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
//
// The Gran Turismo 4 profile: the reference example of how a game profile looks.
//
// GT4 was the game CoreGS 2.0 was written and validated against. The rules below are the
// content-recognition shortcuts that game needed; they are only correct for GT4's own draws and
// therefore live here, not in the generic core. Every rule can be switched on or off, so the same
// profile can also express GT4Recomp's "performance" choices.
#pragma once

#include "coregs/profile.h"

#include <cstddef>
#include <cstdint>

namespace coregs::gt4 {

struct Settings {
  // The night race reads back a 64 x 8 CT32 light-flare visibility table once per frame. Answering it
  // with zeros removes the readback wait and hides the sun glare and light flares.
  bool hide_light_flare_table = false;
  // The road sheen is a full-screen pass that adds a small target's picture weighted by destination
  // alpha. Dropping it makes the road darker.
  bool skip_road_sheen = false;
  // Exact early-out for that pass: a fragment with no destination alpha contributes nothing.
  bool road_sheen_early_out = true;
  // GT4 draws its brightness/fade ramp as a run of channel-shuffle draws; the run is replaced by one
  // palette pass over the target.
  bool tone_runs = true;
  // T8 sprite batches that change palette mid-batch (experimental; GT4 gains little).
  bool palette_segments = false;
  // Small reflection/shadow targets stay at 1x when the internal scale is higher.
  bool effect_targets_at_1x = false;
  // Precompile the draw states GT4's scenes ask for (profiles/gt4/known_shader_variants.inc).
  bool precompile_known_variants = true;
};

// The GT4 profile. With the default Settings it reproduces the faithful GT4 renderer: every effect
// drawn, tone runs substituted, palette segments and 1x effect targets off.
GameProfile MakeProfile(const Settings &settings = {});

// The two content predicates, exposed so a host can test them separately.
bool IsLightFlareTableRead(const GameProfile::ReadbackView &read);
bool IsRoadSheenPass(const GameProfile::EffectPassView &pass);

// The draw-state list in profiles/gt4/known_shader_variants.inc (33 words per entry).
const std::uint32_t *KnownShaderVariants();
std::size_t KnownShaderVariantWords();

}  // namespace coregs::gt4
