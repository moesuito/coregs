// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
//
// Game profiles: optional rules that recognise a particular game's content in the GS stream.
//
// The generic renderer follows the GS hardware rules alone. A profile may add optimisations that
// are only correct for one game, because they know what that game draws (a full-screen blend pass
// that always reads the same table, a channel-shuffle run that always ends in a palette lookup).
// Every rule is off by default; a profile that enables one must document why it is safe.
//
// See docs/PROFILES.md and profiles/gt4 for the reference example.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace coregs {

// One context for the whole profile: the name is for logs, the rules are copied by the renderer.
//
// Threading: install a profile before the first gif_transfer(), or between frames with the renderer
// idle. The hooks are called on the GS thread.
struct GameProfile {
  std::string name = "generic";

  // ---- Readbacks ------------------------------------------------------------------------------

  // A LOCAL->HOST read the renderer is about to serve.
  struct ReadbackView {
    std::uint32_t psm = 0;     // GS pixel format encoding (0 = PSMCT32, 1 = PSMCT24, ...)
    std::uint32_t width = 0;   // pixels
    std::uint32_t height = 0;
    std::uint32_t qwords = 0;  // bytes / 16
  };
  // True when the read is the game's light-flare visibility table: answered with zeros instead of a
  // readback. Only valid together with `hide_light_flare_table`.
  std::function<bool(const ReadbackView &)> is_light_flare_table_read;
  bool hide_light_flare_table = false;

  // ---- Full-screen effect passes --------------------------------------------------------------

  // A candidate pass, described by what the GS registers determine (no game knowledge).
  struct EffectPassView {
    std::uint32_t frame_width = 0, frame_height = 0;  // native size of the colour target
    std::uint64_t covered_area = 0;                   // native pixels the pass covers
    bool textured = false;
    bool blends = false;
    std::uint32_t blend_a = 0, blend_b = 0, blend_c = 0, blend_d = 0;  // ALPHA register fields
    bool depth_never = false;   // ZTST = NEVER
    bool depth_always = false;  // ZTST = ALWAYS
    std::uint32_t format = 0;   // PSM of the target
    std::uint32_t fbmsk = 0;
  };
  // True when the pass is the game's full-screen effect pass (a screen-space effect the player can
  // trade for speed).
  std::function<bool(const EffectPassView &)> is_effect_pass;
  // Drop the pass the hook names. It changes the picture; it is never on in the generic profile.
  bool skip_effect_pass = false;
  // Let the named pass discard fragments that cannot contribute (an exact early-out for its blend
  // equation on this game's data).
  bool effect_pass_early_out = false;

  // ---- Alpha-lane fills -----------------------------------------------------------------------

  // A pass that writes only the alpha lane (FBMSK masks every colour bit) exists to feed a later
  // effect. Dropping it is only safe when the effect that reads it is off too.
  bool skip_alpha_only_fills = false;
  std::uint32_t alpha_only_fill_min_width = 0;  // native target width below which the pass is kept (0 = any)

  // ---- Channel-shuffle (tone) runs ------------------------------------------------------------

  // The game implements a tone curve as a run of channel-shuffle draws into one target: skip the
  // run's redundant copy-back draws and apply the run's palette to the target in one GPU pass.
  // Only correct when the run really is that pattern, which is what this rule asserts.
  bool tone_runs = false;
  // Keep a T8 sprite batch open across CLUT loads and draw its palette segments separately
  // (one palette row per segment). Part of the same tone technique.
  bool palette_segments = false;

  // ---- Effect targets -------------------------------------------------------------------------

  // Keep small off-screen effect targets at 1x even when the internal scale is higher.
  bool effect_targets_at_1x = false;

  // ---- Shader precompilation ------------------------------------------------------------------

  // Draw states this game asks for, 33 words each (the renderer's specialised-pipeline key), in the
  // order the driver cache should compile them before the first frame. Empty: no precompilation.
  // The list only saves first-run stalls; a wrong or missing entry never changes rendering.
  const std::uint32_t *known_shader_variants = nullptr;
  std::size_t known_shader_variant_words = 0;
  static constexpr std::size_t kShaderVariantWords = 33;
};

}  // namespace coregs
