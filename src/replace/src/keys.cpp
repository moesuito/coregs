// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 11.2, 11.3.1 to
// 11.3.4 and 04.6 (upload-size inference). Clean room (decision 0069).
// Copyright (c) 2026 GT4Recomp contributors, MIT.
#include "hw_gs_replace/keys.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

#include "hw_gs_replace/xxh3.hpp"

namespace hw_gs_replace {
namespace {

// Block numbers within a page (chapter 01.3, "Block arrangement within a page").
// Rows top to bottom, entries are block numbers within the page.
constexpr uint32_t kTable32[4][8] = {
    {0, 1, 4, 5, 16, 17, 20, 21},
    {2, 3, 6, 7, 18, 19, 22, 23},
    {8, 9, 12, 13, 24, 25, 28, 29},
    {10, 11, 14, 15, 26, 27, 30, 31},
};
constexpr uint32_t kTableCt16[8][4] = {
    {0, 2, 8, 10},
    {1, 3, 9, 11},
    {4, 6, 12, 14},
    {5, 7, 13, 15},
    {16, 18, 24, 26},
    {17, 19, 25, 27},
    {20, 22, 28, 30},
    {21, 23, 29, 31},
};
constexpr uint32_t kTableCt16S[8][4] = {
    {0, 2, 16, 18},
    {1, 3, 17, 19},
    {8, 10, 24, 26},
    {9, 11, 25, 27},
    {4, 6, 20, 22},
    {5, 7, 21, 23},
    {12, 14, 28, 30},
    {13, 15, 29, 31},
};

uint32_t BlockTableEntry(Psm psm, uint32_t tableX, uint32_t tableY) {
  switch (psm) {
    case Psm::Ct16:
    case Psm::Z16:
    case Psm::T4:
      return kTableCt16[tableY][tableX];
    case Psm::Ct16S:
    case Psm::Z16S:
      return kTableCt16S[tableY][tableX];
    default:
      return kTable32[tableY][tableX];
  }
}

uint32_t RoundUp(uint32_t value, uint32_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

// Block number of the block whose origin is pixel (x, y) of a buffer (chapter 01.3, address
// computation, with the Z XOR applied after adding BP, chapter 11.3.2).
uint32_t BlockNumberOf(Psm psm, uint32_t tbp, uint32_t tbw, uint32_t x, uint32_t y) {
  const PsmGeometry geometry = GeometryOf(psm);
  // bwPages = TBW >> (log2(page width) - 6), written as TBW * 64 / page width (rounds down for
  // the 128-pixel-wide 8- and 4-bit pages).
  const uint32_t pagesPerRow = (tbw * 64u) / geometry.pageWidth;
  const uint32_t pageX = x / geometry.pageWidth;
  const uint32_t pageY = y / geometry.pageHeight;
  const uint32_t tableX = (x % geometry.pageWidth) / geometry.blockWidth;
  const uint32_t tableY = (y % geometry.pageHeight) / geometry.blockHeight;
  uint32_t block = tbp + 32u * (pageY * pagesPerRow + pageX) + BlockTableEntry(psm, tableX, tableY);
  if (IsZPsm(psm)) {
    block ^= 24u;
  }
  return block & 16383u;
}

// One level of a key: its address and its hashed rectangle in level coordinates.
struct LevelView {
  uint32_t tbp = 0;
  uint32_t tbw = 0;
  uint32_t log2Width = 0;
  uint32_t log2Height = 0;
  Region regionX;
  Region regionY;
};

// Hashed size and rectangle of a level (chapter 11.3.1 and 11.3.2).
struct HashRect {
  uint32_t width;
  uint32_t height;
  uint32_t left;
  uint32_t top;
  uint32_t right;
  uint32_t bottom;
};

HashRect RectOf(const LevelView& level) {
  HashRect rect{};
  rect.width = level.regionX.present ? level.regionX.Size() : (1u << level.log2Width);
  rect.height = level.regionY.present ? level.regionY.Size() : (1u << level.log2Height);
  rect.left = level.regionX.present ? level.regionX.minimum : 0;
  rect.top = level.regionY.present ? level.regionY.minimum : 0;
  rect.right = rect.left + rect.width;
  rect.bottom = rect.top + rect.height;
  return rect;
}

// Appends the bytes of texel (x, y) of a direct or indexed level to `out` (chapter 11.2 /
// 11.3.2 path B texel bytes).
void AppendTexelBytes(const TexelSource& memory, Psm psm, const LevelView& level, TexaState texa,
                      uint32_t x, uint32_t y, std::vector<uint8_t>& out) {
  const uint32_t stored = memory.ReadPixel(psm, level.tbp, level.tbw, x, y);
  if (IsIndexedPsm(psm)) {
    out.push_back(static_cast<uint8_t>(stored & 0xFFu));
    return;
  }
  const uint32_t rgba = ExpandDirectTexel(psm, stored, texa);
  out.push_back(static_cast<uint8_t>(rgba & 0xFFu));
  out.push_back(static_cast<uint8_t>((rgba >> 8) & 0xFFu));
  out.push_back(static_cast<uint8_t>((rgba >> 16) & 0xFFu));
  out.push_back(static_cast<uint8_t>((rgba >> 24) & 0xFFu));
}

// Format G stream: the rows of the base rectangle, top to bottom, texels left to right.
std::vector<uint8_t> FormatGStream(const TexelSource& memory, Psm psm, const LevelView& level,
                                   TexaState texa) {
  const HashRect rect = RectOf(level);
  const uint32_t bytesPerTexel = IsIndexedPsm(psm) ? 1u : 4u;
  std::vector<uint8_t> stream;
  stream.reserve(static_cast<size_t>(rect.width) * rect.height * bytesPerTexel);
  for (uint32_t y = rect.top; y < rect.bottom; ++y) {
    for (uint32_t x = rect.left; x < rect.right; ++x) {
      AppendTexelBytes(memory, psm, level, texa, x, y, stream);
    }
  }
  return stream;
}

// Hashes one level of a Format P stream (chapter 11.3.2). Path A (raw blocks) when all of the
// listed conditions hold; otherwise path B (expanded texels with the byte-offset quirk).
void HashPackLevel(Xxh3Stream& hash, const TexelSource& memory, Psm psm, TexaState texa,
                   const LevelView& level) {
  const PsmGeometry geometry = GeometryOf(psm);
  const HashRect rect = RectOf(level);
  const bool rawFormat =
      psm == Psm::Ct32 || psm == Psm::Z32 || psm == Psm::T8 || psm == Psm::T4;
  const bool rawEligible = rawFormat && rect.width >= geometry.blockWidth &&
                           rect.height >= geometry.blockHeight && !level.regionX.present &&
                           (!level.regionY.present || level.regionY.minimum == 0);

  // block_rect: the rectangle expanded outward to whole blocks.
  const uint32_t blockLeft = rect.left / geometry.blockWidth * geometry.blockWidth;
  const uint32_t blockTop = rect.top / geometry.blockHeight * geometry.blockHeight;
  const uint32_t blockRight = RoundUp(rect.right, geometry.blockWidth);
  const uint32_t blockBottom = RoundUp(rect.bottom, geometry.blockHeight);

  if (rawEligible) {
    std::array<uint8_t, 256> block{};
    for (uint32_t by = blockTop; by < blockBottom; by += geometry.blockHeight) {
      for (uint32_t bx = blockLeft; bx < blockRight; bx += geometry.blockWidth) {
        memory.ReadBlock(BlockNumberOf(psm, level.tbp, level.tbw, bx, by), block);
        hash.Update(block);
      }
    }
    return;
  }

  // Path B. The buffer covers block_rect with column 0 at block_rect.left. The pitch rule is the
  // specification's literal "block_rect.right x bytes per texel, rounded up to 32 bytes".
  const bool indexed = IsIndexedPsm(psm);
  const uint32_t bytesPerTexel = indexed ? 1u : 4u;
  const uint32_t pitch = RoundUp(blockRight * bytesPerTexel, 32u);
  const uint32_t blockRows = blockBottom - blockTop;
  // Zero padding after the block rows so that the last row's over-read (below) stays in range.
  const size_t bufferSize = static_cast<size_t>(pitch) * blockRows + rect.width * bytesPerTexel + pitch;
  std::vector<uint8_t> buffer(bufferSize, 0);
  // A whole-row read and no allocation per texel: a large menu texture took 33 ms here on the GS thread.
  std::vector<uint32_t> stored(blockRight - blockLeft);
  for (uint32_t y = blockTop; y < blockBottom; ++y) {
    memory.ReadRow(psm, level.tbp, level.tbw, blockLeft, y, blockRight - blockLeft, stored.data());
    uint8_t* row = buffer.data() + static_cast<size_t>(y - blockTop) * pitch;
    if (indexed) {
      for (size_t i = 0; i < stored.size(); ++i) {
        row[i] = static_cast<uint8_t>(stored[i] & 0xFFu);
      }
    } else {
      for (size_t i = 0; i < stored.size(); ++i) {
        const uint32_t rgba = ExpandDirectTexel(psm, stored[i], texa);
        row[i * 4] = static_cast<uint8_t>(rgba & 0xFFu);
        row[i * 4 + 1] = static_cast<uint8_t>((rgba >> 8) & 0xFFu);
        row[i * 4 + 2] = static_cast<uint8_t>((rgba >> 16) & 0xFFu);
        row[i * 4 + 3] = static_cast<uint8_t>((rgba >> 24) & 0xFFu);
      }
    }
  }

  // Start = buffer + pitch * (rect.top - block_rect.top) + (rect.left - block_rect.left) BYTES.
  // The byte offset is not scaled by the texel size (chapter 11.3.2 quirk, reproduced on purpose).
  const size_t startColumnBytes = rect.left - blockLeft;
  const size_t startRow = rect.top - blockTop;
  const size_t rowBytes = static_cast<size_t>(rect.width) * bytesPerTexel;
  for (uint32_t row = 0; row < rect.height; ++row) {
    const size_t start = (startRow + row) * pitch + startColumnBytes;
    hash.Update(std::span<const uint8_t>(buffer.data() + start, rowBytes));
  }
}

// Chapter 04.6 upload-size inference (High, applied to Format P only). Operates on the base
// level's region in place and only when both dimensions shrink.
void ApplyUploadSizeInference(const TextureDescription& desc, LevelView& base) {
  if (!desc.exactUploadInLast10Batches.has_value()) {
    return;
  }
  if (base.regionX.present || base.regionY.present) {
    return;
  }
  if (desc.wrapU.mode != WrapMode::Clamp || desc.wrapV.mode != WrapMode::Clamp) {
    return;
  }
  const uint32_t fullWidth = 1u << desc.tw;
  const uint32_t fullHeight = 1u << desc.th;
  if (!(desc.readExtent.width < fullWidth || desc.readExtent.height < fullHeight)) {
    return;
  }
  const PsmGeometry geometry = GeometryOf(desc.psm);
  const UploadExtent upload = *desc.exactUploadInLast10Batches;
  const uint32_t newWidth =
      std::max(upload.width, RoundUp(desc.readExtent.width, geometry.pageWidth));
  const uint32_t newHeight =
      std::max(upload.height, RoundUp(desc.readExtent.height, geometry.pageHeight));
  if (newWidth < fullWidth && newHeight < fullHeight) {
    base.regionX = Region{true, 0, newWidth};
    base.regionY = Region{true, 0, newHeight};
  }
}

LevelView BaseLevel(const TextureDescription& desc) {
  LevelView base;
  base.tbp = desc.tbp;
  base.tbw = desc.tbw;
  base.log2Width = desc.tw;
  base.log2Height = desc.th;
  base.regionX = ResolveRegion(desc.wrapU, desc.tw);
  base.regionY = ResolveRegion(desc.wrapV, desc.th);
  return base;
}

// Palette of the draw (chapter 07.5): 16 entries for 4-bit formats, 256 for 8-bit formats.
std::vector<uint8_t> PaletteBytesOf(const TextureDescription& desc) {
  if (!IsIndexedPsm(desc.psm)) {
    return {};
  }
  static const ClutBuffer kEmptyClut{};
  const ClutBuffer& clut = desc.clut != nullptr ? *desc.clut : kEmptyClut;
  const bool eightBit = desc.psm == Psm::T8 || desc.psm == Psm::T8H;
  const std::vector<uint32_t> entries =
      DecodeClut(clut, desc.clutPsm, desc.csa, eightBit ? 256u : 16u, desc.texa);
  return PaletteBytes(entries);
}

}  // namespace

FormatGKey ComputeFormatGKey(const TexelSource& memory, const TextureDescription& desc) {
  const TexaState texa = MaskTexaForPsm(desc.psm, desc.texa);
  const LevelView base = BaseLevel(desc);
  const HashRect rect = RectOf(base);
  const std::vector<uint8_t> stream = FormatGStream(memory, desc.psm, base, texa);

  FormatGKey key;
  key.textureHash = Xxh3Hash64(stream);
  if (IsIndexedPsm(desc.psm)) {
    key.paletteHash = Xxh3Hash64(PaletteBytesOf(desc));
  }
  key.width = rect.width;
  key.height = rect.height;
  return key;
}

FormatPKeys ComputeFormatPKeys(const TexelSource& memory, const TextureDescription& desc) {
  const TexaState texa = MaskTexaForPsm(desc.psm, desc.texa);
  LevelView base = BaseLevel(desc);
  ApplyUploadSizeInference(desc, base);

  Xxh3Stream hash;
  HashPackLevel(hash, memory, desc.psm, texa, base);
  const uint64_t baseOnlyHash = hash.Digest();

  const size_t mipCount = std::min<size_t>(desc.mipLevels.size(), 6);
  const bool hashMips = desc.mipmappingActive && mipCount > 0;
  uint64_t fullChainHash = baseOnlyHash;
  if (hashMips) {
    for (size_t index = 0; index < mipCount; ++index) {
      const uint32_t level = static_cast<uint32_t>(index) + 1;
      LevelView mip;
      mip.tbp = desc.mipLevels[index].tbp;
      mip.tbw = desc.mipLevels[index].tbw;
      mip.log2Width = desc.tw > level ? desc.tw - level : 0;
      mip.log2Height = desc.th > level ? desc.th - level : 0;
      mip.regionX = ShiftRegionForMip(base.regionX, level);
      mip.regionY = ShiftRegionForMip(base.regionY, level);
      HashPackLevel(hash, memory, desc.psm, texa, mip);
    }
    fullChainHash = hash.Digest();
  }

  const bool indexed = IsIndexedPsm(desc.psm);
  const uint64_t clutHash = indexed ? Xxh3Hash64(PaletteBytesOf(desc)) : 0;

  FormatPRecord record;
  record.clutHash = clutHash;
  record.regionWidth = base.regionX.Size();
  record.regionHeight = base.regionY.Size();
  record.bits = PackBitsWord(desc.psm, desc.tw, desc.th, desc.texa);
  record.mipLevel = 0;
  record.indexed = indexed;

  FormatPKeys keys;
  keys.fullChain = record;
  keys.fullChain.textureHash = fullChainHash;
  if (hashMips) {
    FormatPRecord baseOnly = record;
    baseOnly.textureHash = baseOnlyHash;
    keys.baseOnly = baseOnly;
  } else {
    keys.fullChain.textureHash = baseOnlyHash;
  }
  return keys;
}

}  // namespace hw_gs_replace
