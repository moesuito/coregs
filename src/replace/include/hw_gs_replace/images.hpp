// HWR texture replacement image decoders: PNG (8-bit RGB and RGBA) and DDS (BC1/2/3/7 and the
// listed uncompressed layouts), chapter 11.4. Written from the TASK-026 specification
// (docs/hw-renderer-spec). Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "hw_gs_replace/names.hpp"

namespace hw_gs_replace {

// Alpha rules differ between the two formats for alpha-less inputs (chapter 11.4).
enum class AlphaPolicy : uint8_t {
  // Format P: X8R8G8B8 -> A = 0xFF, X8B8G8R8 -> A = 0x80, R8G8B8 -> A = 0xFF.
  FormatP,
  // Format G: every alpha-less input -> A = 0x80.
  FormatG,
};

enum class ReplacementCodec : uint8_t {
  Rgba8,  // levels hold RGBA8 bytes, row-major, top row first
  Bc1,    // levels hold compressed BC1 blocks (8 bytes per 4x4)
  Bc2,    // 16 bytes per block
  Bc3,    // 16 bytes per block
  Bc7,    // 16 bytes per block
};

struct ReplacementLevel {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint8_t> bytes;
};

struct ReplacementImage {
  ReplacementCodec codec = ReplacementCodec::Rgba8;
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<ReplacementLevel> levels;  // level 0 is the base
};

// Decodes a file by its extension. On failure returns false and sets `error` (a log line).
bool DecodeReplacementImage(ImageExtension extension, std::span<const uint8_t> bytes,
                            AlphaPolicy policy, ReplacementImage& image, std::string& error);

// PNG: 8-bit RGB (alpha 0x80) or RGBA (copied). Palette, grey and 16-bit PNGs are refused.
bool DecodePngImage(std::span<const uint8_t> bytes, ReplacementImage& image, std::string& error);

// DDS: legacy FourCC DXT1/DXT2/DXT3/DXT4/DXT5 map to BC1/BC2/BC3/BC3. DX10 headers accept BC1,
// BC2, BC3 and BC7 UNORM (sRGB, BC4, BC5, BC6 are refused). Uncompressed masks: A8R8G8B8,
// A8B8G8R8, X8R8G8B8, X8B8G8R8, R8G8B8. Mip count from the header (0 = full chain); levels are
// read until the data ends. Compressed levels stay compressed.
bool DecodeDdsImage(std::span<const uint8_t> bytes, AlphaPolicy policy, ReplacementImage& image,
                    std::string& error);

}  // namespace hw_gs_replace
