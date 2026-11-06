// Written from the TASK-026 specification (docs/hw-renderer-spec), chapter 11.4 (image formats).
// Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
// PNG decoding uses the vendored stb_image v2.30 (public domain or MIT) with the PNG decoder only.
#include "hw_gs_replace/images.hpp"

#include <algorithm>
#include <climits>
#include <cstring>

#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "stb_image.h"

namespace hw_gs_replace {
namespace {

constexpr uint8_t kPngSignature[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};

uint32_t ReadLe32(const uint8_t* at) {
  return static_cast<uint32_t>(at[0]) | (static_cast<uint32_t>(at[1]) << 8) |
         (static_cast<uint32_t>(at[2]) << 16) | (static_cast<uint32_t>(at[3]) << 24);
}

uint32_t ReadBe32(const uint8_t* at) {
  return (static_cast<uint32_t>(at[0]) << 24) | (static_cast<uint32_t>(at[1]) << 16) |
         (static_cast<uint32_t>(at[2]) << 8) | static_cast<uint32_t>(at[3]);
}

// DDS constants (Microsoft DDS file layout).
constexpr uint32_t kDdpfFourCc = 0x4;
constexpr uint32_t kDdpfRgb = 0x40;
constexpr uint32_t kDxgiBc1Unorm = 71;
constexpr uint32_t kDxgiBc2Unorm = 74;
constexpr uint32_t kDxgiBc3Unorm = 77;
constexpr uint32_t kDxgiBc7Unorm = 98;

enum class UncompressedLayout : uint8_t {
  A8R8G8B8,
  A8B8G8R8,
  X8R8G8B8,
  X8B8G8R8,
  R8G8B8,
};

struct Masks {
  uint32_t red;
  uint32_t green;
  uint32_t blue;
  uint32_t alpha;
};

// Classifies a 24- or 32-bit uncompressed mask set. Unknown masks are refused.
bool ClassifyUncompressed(uint32_t bitCount, const Masks& masks, UncompressedLayout& layout) {
  if (bitCount == 32) {
    if (masks.red == 0xFF0000u && masks.green == 0xFF00u && masks.blue == 0xFFu &&
        masks.alpha == 0xFF000000u) {
      layout = UncompressedLayout::A8R8G8B8;
      return true;
    }
    if (masks.red == 0xFFu && masks.green == 0xFF00u && masks.blue == 0xFF0000u &&
        masks.alpha == 0xFF000000u) {
      layout = UncompressedLayout::A8B8G8R8;
      return true;
    }
    if (masks.red == 0xFF0000u && masks.green == 0xFF00u && masks.blue == 0xFFu &&
        masks.alpha == 0u) {
      layout = UncompressedLayout::X8R8G8B8;
      return true;
    }
    if (masks.red == 0xFFu && masks.green == 0xFF00u && masks.blue == 0xFF0000u &&
        masks.alpha == 0u) {
      layout = UncompressedLayout::X8B8G8R8;
      return true;
    }
    return false;
  }
  if (bitCount == 24 && masks.red == 0xFF0000u && masks.green == 0xFF00u && masks.blue == 0xFFu &&
      masks.alpha == 0u) {
    layout = UncompressedLayout::R8G8B8;
    return true;
  }
  return false;
}

// Alpha byte for an alpha-less layout under the policy (chapter 11.4).
uint8_t AlphaForAlphaless(UncompressedLayout layout, AlphaPolicy policy) {
  if (policy == AlphaPolicy::FormatG) {
    return 0x80;
  }
  switch (layout) {
    case UncompressedLayout::X8B8G8R8:
      return 0x80;
    default:
      return 0xFF;
  }
}

// Shift of a contiguous mask (0 for a zero mask).
uint32_t ShiftOf(uint32_t mask) {
  uint32_t shift = 0;
  while (mask != 0 && (mask & 1u) == 0) {
    mask >>= 1;
    ++shift;
  }
  return shift;
}

uint32_t BlockBytes(ReplacementCodec codec) {
  return codec == ReplacementCodec::Bc1 ? 8u : 16u;
}

uint32_t FullChainCount(uint32_t width, uint32_t height) {
  uint32_t largest = std::max(width, height);
  uint32_t count = 1;
  while (largest > 1) {
    largest >>= 1;
    ++count;
  }
  return count;
}

}  // namespace

bool DecodeReplacementImage(ImageExtension extension, std::span<const uint8_t> bytes,
                            AlphaPolicy policy, ReplacementImage& image, std::string& error) {
  if (extension == ImageExtension::Png) {
    return DecodePngImage(bytes, image, error);
  }
  return DecodeDdsImage(bytes, policy, image, error);
}

bool DecodePngImage(std::span<const uint8_t> bytes, ReplacementImage& image, std::string& error) {
  if (bytes.size() < 33 || bytes.size() > static_cast<size_t>(INT_MAX)) {
    error = "PNG too small or too large";
    return false;
  }
  if (std::memcmp(bytes.data(), kPngSignature, sizeof(kPngSignature)) != 0) {
    error = "not a PNG file";
    return false;
  }
  // The first chunk must be IHDR with 13 bytes of data (PNG specification).
  if (ReadBe32(bytes.data() + 8) != 13 || std::memcmp(bytes.data() + 12, "IHDR", 4) != 0) {
    error = "PNG without a valid IHDR chunk";
    return false;
  }
  const uint32_t headerWidth = ReadBe32(bytes.data() + 16);
  const uint32_t headerHeight = ReadBe32(bytes.data() + 20);
  const uint8_t bitDepth = bytes[24];
  const uint8_t colourType = bytes[25];
  if (bitDepth != 8 || (colourType != 2 && colourType != 6)) {
    error = "unsupported PNG: only 8-bit RGB (2) and RGBA (6); bit depth " +
            std::to_string(bitDepth) + ", colour type " + std::to_string(colourType) + " refused";
    return false;
  }
  if (headerWidth == 0 || headerHeight == 0) {
    error = "PNG with zero size";
    return false;
  }

  const bool hasAlpha = colourType == 6;
  int width = 0;
  int height = 0;
  int channelsInFile = 0;
  unsigned char* pixels = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                                                &width, &height, &channelsInFile, hasAlpha ? 4 : 3);
  if (pixels == nullptr) {
    error = std::string("PNG decode failed: ") + stbi_failure_reason();
    return false;
  }
  if (static_cast<uint32_t>(width) != headerWidth || static_cast<uint32_t>(height) != headerHeight) {
    stbi_image_free(pixels);
    error = "PNG decoder size differs from IHDR";
    return false;
  }

  const size_t texelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
  ReplacementLevel level;
  level.width = static_cast<uint32_t>(width);
  level.height = static_cast<uint32_t>(height);
  level.bytes.resize(texelCount * 4);
  for (size_t texel = 0; texel < texelCount; ++texel) {
    const unsigned char* source = pixels + texel * (hasAlpha ? 4u : 3u);
    uint8_t* destination = level.bytes.data() + texel * 4;
    destination[0] = source[0];
    destination[1] = source[1];
    destination[2] = source[2];
    destination[3] = hasAlpha ? source[3] : 0x80;
  }
  stbi_image_free(pixels);

  image = ReplacementImage{};
  image.codec = ReplacementCodec::Rgba8;
  image.width = level.width;
  image.height = level.height;
  image.levels.push_back(std::move(level));
  return true;
}

bool DecodeDdsImage(std::span<const uint8_t> bytes, AlphaPolicy policy, ReplacementImage& image,
                    std::string& error) {
  if (bytes.size() < 128 || std::memcmp(bytes.data(), "DDS ", 4) != 0) {
    error = "not a DDS file";
    return false;
  }
  const uint8_t* header = bytes.data();
  if (ReadLe32(header + 4) != 124) {
    error = "DDS header size is not 124";
    return false;
  }
  const uint32_t height = ReadLe32(header + 12);
  const uint32_t width = ReadLe32(header + 16);
  const uint32_t mipCountField = ReadLe32(header + 28);
  if (ReadLe32(header + 76) != 32) {
    error = "DDS pixel format size is not 32";
    return false;
  }
  const uint32_t pixelFlags = ReadLe32(header + 80);
  const char* fourCc = reinterpret_cast<const char*>(header + 84);
  const uint32_t bitCount = ReadLe32(header + 88);
  const Masks masks{ReadLe32(header + 92), ReadLe32(header + 96), ReadLe32(header + 100),
                    ReadLe32(header + 104)};
  if (width == 0 || height == 0) {
    error = "DDS with zero size";
    return false;
  }

  size_t dataOffset = 128;
  ReplacementCodec codec = ReplacementCodec::Rgba8;
  bool compressed = false;
  UncompressedLayout layout = UncompressedLayout::R8G8B8;
  uint32_t bytesPerPixel = 0;

  if ((pixelFlags & kDdpfFourCc) != 0 && std::memcmp(fourCc, "DX10", 4) == 0) {
    if (bytes.size() < 148) {
      error = "DDS DX10 header truncated";
      return false;
    }
    dataOffset = 148;
    const uint32_t dxgi = ReadLe32(header + 128);
    compressed = true;
    switch (dxgi) {
      case kDxgiBc1Unorm:
        codec = ReplacementCodec::Bc1;
        break;
      case kDxgiBc2Unorm:
        codec = ReplacementCodec::Bc2;
        break;
      case kDxgiBc3Unorm:
        codec = ReplacementCodec::Bc3;
        break;
      case kDxgiBc7Unorm:
        codec = ReplacementCodec::Bc7;
        break;
      default:
        error = "DDS DXGI format " + std::to_string(dxgi) +
                " refused (only BC1, BC2, BC3, BC7 UNORM are supported)";
        return false;
    }
  } else if ((pixelFlags & kDdpfFourCc) != 0) {
    compressed = true;
    if (std::memcmp(fourCc, "DXT1", 4) == 0) {
      codec = ReplacementCodec::Bc1;
    } else if (std::memcmp(fourCc, "DXT2", 4) == 0 || std::memcmp(fourCc, "DXT3", 4) == 0) {
      codec = ReplacementCodec::Bc2;
    } else if (std::memcmp(fourCc, "DXT4", 4) == 0 || std::memcmp(fourCc, "DXT5", 4) == 0) {
      codec = ReplacementCodec::Bc3;
    } else {
      error = "DDS FourCC is not DXT1..DXT5 or DX10";
      return false;
    }
  } else if ((pixelFlags & kDdpfRgb) != 0) {
    if (!ClassifyUncompressed(bitCount, masks, layout)) {
      error = "DDS uncompressed masks are not one of the supported layouts";
      return false;
    }
    bytesPerPixel = bitCount / 8;
  } else {
    error = "DDS pixel format is neither FourCC nor RGB";
    return false;
  }

  if (compressed && ((width % 4) != 0 || (height % 4) != 0)) {
    error = "DDS BC base dimensions must be multiples of 4";
    return false;
  }

  const uint32_t levelCount = mipCountField == 0 ? FullChainCount(width, height) : mipCountField;
  image = ReplacementImage{};
  image.codec = codec;
  image.width = width;
  image.height = height;

  size_t offset = dataOffset;
  for (uint32_t index = 0; index < levelCount; ++index) {
    const uint32_t levelWidth = std::max<uint32_t>(1, width >> index);
    const uint32_t levelHeight = std::max<uint32_t>(1, height >> index);
    size_t levelBytes = 0;
    if (compressed) {
      const uint32_t blocksX = std::max<uint32_t>(1, (levelWidth + 3) / 4);
      const uint32_t blocksY = std::max<uint32_t>(1, (levelHeight + 3) / 4);
      levelBytes = static_cast<size_t>(blocksX) * blocksY * BlockBytes(codec);
    } else {
      levelBytes = static_cast<size_t>(levelWidth) * levelHeight * bytesPerPixel;
    }
    // Levels are read sequentially until a read fails (chapter 11.4).
    if (levelBytes > bytes.size() || offset > bytes.size() - levelBytes) {
      break;
    }
    const uint8_t* source = bytes.data() + offset;
    ReplacementLevel level;
    level.width = levelWidth;
    level.height = levelHeight;
    if (compressed) {
      level.bytes.assign(source, source + levelBytes);
    } else {
      // Convert the masked channels to RGBA8 (chapter 11.4).
      const uint32_t redShift = ShiftOf(masks.red);
      const uint32_t greenShift = ShiftOf(masks.green);
      const uint32_t blueShift = ShiftOf(masks.blue);
      const uint32_t alphaShift = ShiftOf(masks.alpha);
      const bool fileHasAlpha = masks.alpha != 0;
      const size_t texelCount = static_cast<size_t>(levelWidth) * levelHeight;
      level.bytes.resize(texelCount * 4);
      for (size_t texel = 0; texel < texelCount; ++texel) {
        uint32_t value = 0;
        for (uint32_t byte = 0; byte < bytesPerPixel; ++byte) {
          value |= static_cast<uint32_t>(source[texel * bytesPerPixel + byte]) << (8 * byte);
        }
        uint8_t* destination = level.bytes.data() + texel * 4;
        destination[0] = static_cast<uint8_t>((value & masks.red) >> redShift);
        destination[1] = static_cast<uint8_t>((value & masks.green) >> greenShift);
        destination[2] = static_cast<uint8_t>((value & masks.blue) >> blueShift);
        destination[3] = fileHasAlpha ? static_cast<uint8_t>((value & masks.alpha) >> alphaShift)
                                      : AlphaForAlphaless(layout, policy);
      }
    }
    image.levels.push_back(std::move(level));
    offset += levelBytes;
  }

  if (image.levels.empty()) {
    error = "DDS has no readable mip level";
    return false;
  }
  return true;
}

}  // namespace hw_gs_replace
