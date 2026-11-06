// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec), clean room per decision 0069.
// CTest hw_options: COREGS_RENDERER and COREGS_HW_* parsing, defaults, validation and descriptions.

#include "coregs/options.h"

#include <cstdio>
#include <map>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const char *what) {
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

// Environment-like lookup over a map; missing keys read as unset.
coregs::EnvironmentLookup env_of(const std::map<std::string, std::string> &values) {
  return [&values](const char *name) -> const char * {
    const auto found = values.find(name);
    return found == values.end() ? nullptr : found->second.c_str();
  };
}

// True when parsing `name=value` throws OptionError.
bool rejected(const char *name, const char *value) {
  try {
    coregs::parse_options(env_of({{name, value}}));
  } catch (const coregs::OptionError &) {
    return true;
  }
  return false;
}

} // namespace

int main() {
  using namespace coregs;

  // Defaults (chapter 12.8 default column).
  const Options defaults = parse_options(env_of({}));
  expect(defaults.scale == 1, "default scale 1");
  expect(defaults.half_pixel == HalfPixel::native, "default half pixel native");
  expect(defaults.sprite_fix == 0, "default sprite fix empty");
  expect(defaults.blend == Blend::high, "default blend high");
  expect(!defaults.accurate_atest, "default accurate alpha test off");
  expect(defaults.mipmap == Mipmap::full, "default mipmap full");
  expect(defaults.filter == Filter::ps2, "default filter ps2");
  expect(defaults.autoflush == AutoFlush::sprites, "default autoflush sprites");
  expect(defaults.sync_mode == SyncMode::cache, "default sync cache");
  expect(defaults.glm, "default GLM on");
  expect(defaults.readback == Readback::exact, "default readback exact");
  expect(defaults.antiblur, "default antiblur on");
  expect(defaults.shadeboost[0] == 50 && defaults.shadeboost[3] == 50, "default shadeboost neutral");
  expect(defaults.crop[0] == 0 && defaults.crop[3] == 0, "default crop zero");
  expect(defaults.textures_root.empty(), "default texture replacement off");
  expect(defaults.game_fixes, "default game fixes on");
  expect(defaults.dump_source == DumpSource::vram, "default dump source vram");
  expect(!defaults.gpu_timing && !defaults.gpu_decode, "default GPU timing and decode off");
  expect(defaults.decode_reuse, "default decode reuse on");
  expect(defaults.clut_defer, "default CLUT defer on");

  // Empty values behave as unset.
  const Options empties = parse_options(env_of({{"COREGS_HW_SCALE", ""}, {"COREGS_HW_GLM", ""}}));
  expect(empties.scale == 1 && empties.glm, "empty values take the default");

  // Valid values of each kind.
  const Options chosen = parse_options(env_of({
      {"COREGS_HW_SCALE", "4"},
      {"COREGS_HW_HALF_PIXEL", "normal"},
      {"COREGS_HW_SPRITE_FIX", "round,align,merge"},
      {"COREGS_HW_NATIVE_POSTFX", "aggressive"},
      {"COREGS_HW_BLEND", "full"},
      {"COREGS_HW_ACCURATE_ATEST", "1"},
      {"COREGS_HW_MIPMAP", "basic"},
      {"COREGS_HW_FILTER", "forced"},
      {"COREGS_HW_DITHER", "off"},
      {"COREGS_HW_AUTOFLUSH", "all"},
      {"COREGS_HW_SYNC_MODE", "writeback"},
      {"COREGS_HW_GLM", "0"},
      {"COREGS_HW_READBACK", "none"},
      {"COREGS_HW_READBACK_SPIN", "1"},
      {"COREGS_HW_DIRTY_FILTER", "nearest"},
      {"COREGS_HW_CLUT_ALWAYS_LOAD", "1"},
      {"COREGS_HW_VERIFY_SHUFFLE", "1"},
      {"COREGS_HW_DEINTERLACE", "bob_bff"},
      {"COREGS_HW_ANTIBLUR", "0"},
      {"COREGS_HW_POSTFX", "shadeboost,cas"},
      {"COREGS_HW_SHADEBOOST", "10,20,30,100"},
      {"COREGS_HW_CAS_SHARPNESS", "0"},
      {"COREGS_HW_PRESENT_FILTER", "sharp"},
      {"COREGS_HW_CROP", "0,8,0,8"},
      {"COREGS_HW_TEXTURES", "C:/textures"},
      {"COREGS_HW_TEXTURES_PRECACHE", "1"},
      {"COREGS_HW_DUMP_TEXTURES", "mips"},
      {"COREGS_HW_DUMP_FORMAT", "p"},
      {"COREGS_HW_GAME_FIXES", "0"},
      {"COREGS_HW_DUMP_SOURCE", "scanout"},
      {"COREGS_HW_GPU_TIMING", "1"},
      {"COREGS_HW_DECODE_REUSE", "0"},
      {"COREGS_HW_CLUT_DEFER", "0"},
      {"COREGS_HW_GPU_DECODE", "1"},
  }));
  expect(chosen.scale == 4, "scale 4");
  expect(chosen.half_pixel == HalfPixel::normal, "half pixel normal");
  expect(chosen.sprite_fix == (sprite_fix_round | sprite_fix_align | sprite_fix_merge), "sprite fix bits");
  expect(chosen.native_postfx == NativePostfx::aggressive, "native postfx aggressive");
  expect(chosen.blend == Blend::full && chosen.accurate_atest, "blend full and accurate atest");
  expect(chosen.mipmap == Mipmap::basic && chosen.filter == Filter::forced, "mipmap and filter");
  expect(chosen.dither == Dither::off && chosen.autoflush == AutoFlush::all, "dither and autoflush");
  expect(chosen.sync_mode == SyncMode::writeback && !chosen.glm, "sync writeback and glm off");
  expect(chosen.readback == Readback::none && chosen.readback_spin, "readback none and spin");
  expect(chosen.dirty_filter == DirtyFilter::nearest, "dirty filter nearest");
  expect(chosen.clut_always_load && chosen.verify_shuffle, "clut and shuffle flags");
  expect(chosen.deinterlace.algorithm == DeinterlaceAlgorithm::bob && chosen.deinterlace.bottom_field_first,
         "deinterlace bob with bff");
  expect(!chosen.antiblur, "antiblur off");
  expect(chosen.postfx == (postfx_shadeboost | postfx_cas), "postfx bits");
  expect(chosen.shadeboost[0] == 10 && chosen.shadeboost[3] == 100, "shadeboost values");
  expect(chosen.cas_sharpness == 0 && chosen.present_filter == PresentFilter::sharp, "cas and present filter");
  expect(chosen.crop[1] == 8 && chosen.crop[3] == 8, "crop values");
  expect(chosen.textures_root == "C:/textures" && chosen.textures_precache, "textures");
  expect(chosen.dump_textures == DumpTextures::mips && chosen.dump_format == DumpFormat::p, "dump options");
  expect(!chosen.game_fixes && chosen.dump_source == DumpSource::scanout, "fixes off, scanout");
  expect(chosen.gpu_timing && chosen.gpu_decode, "gpu timing and decode");
  expect(!chosen.decode_reuse, "decode reuse off");
  expect(!chosen.clut_defer, "CLUT defer off");

  // IN-372: the upscaled dump source (GPU presentation compose at COREGS_HW_SCALE).
  expect(parse_options(env_of({{"COREGS_HW_DUMP_SOURCE", "upscaled"}})).dump_source == DumpSource::upscaled,
         "dump source upscaled");
  expect(rejected("COREGS_HW_DUMP_SOURCE", "scaled"), "unknown dump source rejected");

  // Invalid values abort (throw), one per kind.
  expect(rejected("COREGS_HW_SCALE", "0"), "scale 0 rejected");
  expect(rejected("COREGS_HW_SCALE", "13"), "scale 13 rejected");
  expect(rejected("COREGS_HW_SCALE", "2x"), "scale with suffix rejected");
  expect(rejected("COREGS_HW_SCALE", "-1"), "negative scale rejected");
  expect(rejected("COREGS_HW_HALF_PIXEL", "half"), "unknown half pixel rejected");
  expect(rejected("COREGS_HW_SPRITE_FIX", "round,,align"), "empty list token rejected");
  expect(rejected("COREGS_HW_SPRITE_FIX", "round,bogus"), "unknown sprite fix token rejected");
  expect(rejected("COREGS_HW_ACCURATE_ATEST", "yes"), "flag not 0/1 rejected");
  expect(rejected("COREGS_HW_GLM", "2"), "glm 2 rejected");
  expect(rejected("COREGS_HW_SHADEBOOST", "1,2,3"), "shadeboost needs four values");
  expect(rejected("COREGS_HW_SHADEBOOST", "1,2,3,101"), "shadeboost above 100 rejected");
  expect(rejected("COREGS_HW_CROP", "0,0,0,70000"), "crop above 65535 rejected");
  expect(rejected("COREGS_HW_DEINTERLACE", "off_bff"), "off with _bff rejected");
  expect(rejected("COREGS_HW_DEINTERLACE", "wave"), "unknown deinterlacer rejected");
  expect(rejected("COREGS_HW_DUMP_TEXTURES", "2"), "dump textures 2 rejected");
  expect(rejected("COREGS_HW_BLEND", "HIGH"), "case-sensitive blend rejected");

  // Names and descriptions (result.json, F8 header).
  expect(option_names().size() == 34, "34 runtime COREGS_HW_* variables (the option set plus COREGS_HW_DECODE_REUSE and COREGS_HW_CLUT_DEFER); COREGS_HW_GS is CMake only");
  const auto lines = describe_options(defaults);
  expect(lines.size() == option_names().size(), "one description line per option");
  for (std::size_t i = 0; i < lines.size() && i < option_names().size(); ++i) {
    expect(lines[i].rfind(option_names()[i] + "=", 0) == 0, "description order matches names");
  }
  const std::string json = options_json(chosen);
  expect(json.front() == '{' && json.back() == '}', "json object");
  expect(json.find("\"COREGS_HW_SCALE\": \"4\"") != std::string::npos, "json carries scale");
  expect(json.find("\"COREGS_HW_DEINTERLACE\": \"bob_bff\"") != std::string::npos, "json carries deinterlace");

  if (failures == 0) std::printf("hw_options: all checks passed\n");
  return failures == 0 ? 0 : 1;
}
