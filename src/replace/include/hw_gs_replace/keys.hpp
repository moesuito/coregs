// SPDX-License-Identifier: Apache-2.0
// CoreGS texture replacement keys: Format G (CoreGS's own, chapter 11.2) and Format P
// (compatible with the common texture-pack layout, chapter 11.3). Written from the CoreGS specification
// (docs/hw-renderer-spec). Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "hw_gs_replace/clut.hpp"
#include "hw_gs_replace/gs_types.hpp"

namespace hw_gs_replace {

struct UploadExtent {
  uint32_t width = 0;
  uint32_t height = 0;
};

// Block address of one mip level after the base (MIPTBP1/2 values, chapter 11.3.4).
struct MipLevelAddress {
  uint32_t tbp = 0;
  uint32_t tbw = 0;
};

// Everything a key needs from the draw. TW/TH are log2 sizes of the base level.
struct TextureDescription {
  Psm psm = Psm::Ct32;
  TexaState texa{};
  WrapAxis wrapU{};
  WrapAxis wrapV{};
  uint32_t tw = 0;
  uint32_t th = 0;
  uint32_t tbp = 0;
  uint32_t tbw = 0;

  // Palette inputs, used by indexed formats only. Null means an all-zero CLUT buffer.
  const ClutBuffer* clut = nullptr;
  Psm clutPsm = Psm::Ct32;
  uint32_t csa = 0;

  // Mip levels after the base, in order (level 1, 2, ...). Only the first six are used.
  std::vector<MipLevelAddress> mipLevels;
  // Whether the draw samples mip levels (Format P hashes them into TEX0Hash, chapter 11.3.4).
  bool mipmappingActive = false;

  // Chapter 04.6 upload-size inference (Format P only): the exact upload recorded within the
  // last 10 batches, if any, and the extent this draw reads.
  std::optional<UploadExtent> exactUploadInLast10Batches;
  UploadExtent readExtent{};
};

// Format G key (chapter 11.2): base level only, no mips, no emulator-setting dependence.
struct FormatGKey {
  uint64_t textureHash = 0;
  uint64_t paletteHash = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  bool operator==(const FormatGKey&) const = default;
};
FormatGKey ComputeFormatGKey(const TexelSource& memory, const TextureDescription& desc);

// Format P record (chapter 11.3, name record table).
struct FormatPRecord {
  uint64_t textureHash = 0;
  uint64_t clutHash = 0;
  uint32_t regionWidth = 0;
  uint32_t regionHeight = 0;
  uint32_t bits = 0;
  uint32_t mipLevel = 0;
  bool indexed = false;
  bool operator==(const FormatPRecord&) const = default;
};

// Format P lookups try the full-chain key first, then the base-only key (chapter 11.3.4).
// baseOnly is empty when the draw does not hash mips, in which case fullChain is base-only.
struct FormatPKeys {
  FormatPRecord fullChain;
  std::optional<FormatPRecord> baseOnly;
};
FormatPKeys ComputeFormatPKeys(const TexelSource& memory, const TextureDescription& desc);

}  // namespace hw_gs_replace
