// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec), clean room per decision 0069.
//
// CoreGS renderer options.
//
// The same fields exist as environment variables (COREGS_HW_*) so a host can expose them from a
// settings file without code: parse_options() reads them through a lookup function, and
// describe_options()/options_json() report the effective values.
//
// Pure logic: no Vulkan, no platform headers. Unknown or malformed values are reported as
// OptionError; the caller decides whether to abort. CoreGS never silently falls back to a
// different behaviour than the one asked for.
#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace coregs {

// Returns the value of an environment variable, or nullptr when unset.
using EnvironmentLookup = std::function<const char *(const char *name)>;

class OptionError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

enum class HalfPixel { native, normal, off };
enum class NativePostfx { off, normal, aggressive };
enum class Blend { basic, medium, high, full };
enum class Mipmap { off, basic, full };
enum class Filter { ps2, nearest, forced, forced_no_sprite };  // IN-433: forced_no_sprite keeps sprites as the game asks
enum class Dither { native, fine, off };
enum class AutoFlush { sprites, all, off };
enum class SyncMode { cache, writeback };
enum class Readback { exact, full, none };
enum class DirtyFilter { automatic, linear, nearest };
enum class DumpTextures { off, on, mips };
enum class DumpFormat { g, p };
// vram: the reference image; scanout: the 1x presentation image; upscaled: the GPU presentation compose at
// COREGS_HW_SCALE (IN-372; interlaced frames fall back to the 1x presentation).
enum class DumpSource { vram, scanout, upscaled };
enum class PresentFilter { bilinear, nearest, sharp };
enum class DeinterlaceAlgorithm { automatic, off, weave, bob, blend, adaptive };

// COREGS_HW_SPRITE_FIX is a comma list; each bit is one token.
enum SpriteFixBit : std::uint32_t {
  sprite_fix_round = 1u << 0,
  sprite_fix_round_full = 1u << 1,
  sprite_fix_align = 1u << 2,
  sprite_fix_merge = 1u << 3,
};

// COREGS_HW_POSTFX is a comma list; each bit is one token.
enum PostfxBit : std::uint32_t {
  postfx_shadeboost = 1u << 0,
  postfx_fxaa = 1u << 1,
  postfx_cas = 1u << 2,
  postfx_scanlines = 1u << 3,
};

struct Deinterlace {
  DeinterlaceAlgorithm algorithm = DeinterlaceAlgorithm::automatic;
  bool bottom_field_first = false; // "_bff" suffix
};

struct Options {
  std::uint32_t scale = 1;                       // COREGS_HW_SCALE 1..12
  HalfPixel half_pixel = HalfPixel::native;      // COREGS_HW_HALF_PIXEL
  std::uint32_t sprite_fix = 0;                  // COREGS_HW_SPRITE_FIX bits
  NativePostfx native_postfx = NativePostfx::off;
  Blend blend = Blend::high;                     // exact GS blending
  bool accurate_atest = false;
  Mipmap mipmap = Mipmap::full;
  Filter filter = Filter::ps2;
  Dither dither = Dither::native;
  AutoFlush autoflush = AutoFlush::sprites;
  SyncMode sync_mode = SyncMode::cache;
  bool glm = true;
  Readback readback = Readback::exact;
  bool readback_spin = false;
  DirtyFilter dirty_filter = DirtyFilter::automatic;
  bool clut_always_load = false;
  bool verify_shuffle = false;
  Deinterlace deinterlace;
  bool antiblur = true;
  std::uint32_t postfx = 0;                      // COREGS_HW_POSTFX bits
  std::uint32_t shadeboost[4] = {50, 50, 50, 50}; // b,c,s,g; 50 neutral, 0..100
  std::uint32_t cas_sharpness = 50;              // 0..100
  PresentFilter present_filter = PresentFilter::bilinear;
  std::uint32_t crop[4] = {0, 0, 0, 0};          // l,t,r,b native pixels
  std::string textures_root;                     // empty = replacement off
  bool textures_precache = false;
  DumpTextures dump_textures = DumpTextures::off;
  DumpFormat dump_format = DumpFormat::g;
  bool game_fixes = true;
  DumpSource dump_source = DumpSource::vram;
  bool gpu_timing = false;
  bool gpu_decode = false;
  bool decode_reuse = true;  // IN-369: a run reuses an identical GLM texture decode
  bool clut_defer = true;    // IN-370: GPU CLUT loads deferred, palettes read from the GLM when exact
};

// Parses every COREGS_HW_* variable through `lookup`. Unset or empty values take
// the defaults of chapter 12.8. Throws OptionError on the first invalid value.
Options parse_options(const EnvironmentLookup &lookup);

// Environment variable names of every HWR option, in chapter 12.8 order.
const std::vector<std::string> &option_names();

// One "NAME=value" line per option with its effective value, in option_names order.
std::vector<std::string> describe_options(const Options &options);

// A JSON object fragment ("{...}") with the effective values, for result.json.
std::string options_json(const Options &options);

} // namespace coregs
