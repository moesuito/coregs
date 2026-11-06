// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 01.5, 01.6, 07.5,
// 11.3.2 and 11.3.3. Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#include "hw_gs_replace/clut.hpp"

#include <algorithm>

namespace hw_gs_replace {

uint32_t Expand16BitToRgba(uint16_t value, TexaState texa) {
  const uint32_t red = (static_cast<uint32_t>(value) & 0x1Fu) << 3;
  const uint32_t green = ((static_cast<uint32_t>(value) >> 5) & 0x1Fu) << 3;
  const uint32_t blue = ((static_cast<uint32_t>(value) >> 10) & 0x1Fu) << 3;
  uint32_t alpha = (value & 0x8000u) != 0 ? texa.ta1 : texa.ta0;
  // Whole-value rule of chapter 07.5 and 11.3.2 (chapter 01.5 says R = G = B = 0; see
  // docs/hw-renderer-spec open question in the TASK-027 report).
  if (texa.aem && value == 0) {
    alpha = 0;
  }
  return red | (green << 8) | (blue << 16) | (alpha << 24);
}

uint32_t ExpandDirectTexel(Psm psm, uint32_t stored, TexaState texa) {
  switch (psm) {
    case Psm::Ct32:
    case Psm::Z32:
      return stored;
    case Psm::Ct24:
    case Psm::Z24: {
      const uint32_t rgb = stored & 0x00FFFFFFu;
      uint32_t alpha = texa.ta0;
      if (texa.aem && rgb == 0) {
        alpha = 0;
      }
      return rgb | (alpha << 24);
    }
    case Psm::Ct16:
    case Psm::Ct16S:
    case Psm::Z16:
    case Psm::Z16S:
      return Expand16BitToRgba(static_cast<uint16_t>(stored & 0xFFFFu), texa);
    default:
      return 0;
  }
}

std::vector<uint32_t> DecodeClut(const ClutBuffer& clut, Psm clutPsm, uint32_t csa,
                                 uint32_t entryCount, TexaState texa) {
  std::vector<uint32_t> entries(entryCount, 0);
  const bool sixteenBit = clutPsm == Psm::Ct16 || clutPsm == Psm::Ct16S;
  const auto halfword = [&clut](uint32_t index) {
    return clut.halfwords[index & 511u];
  };

  if (sixteenBit) {
    // Chapter 07.5: entries start at halfword CSA * 16. The 256-entry read wraps modulo the
    // 512-halfword buffer (Hypothesis, not stated by the specification).
    for (uint32_t i = 0; i < entryCount; ++i) {
      const uint32_t index = (csa * 16u + i) & 511u;
      entries[i] = Expand16BitToRgba(halfword(index), texa);
    }
    return entries;
  }

  const auto combine = [&halfword](uint32_t index) {
    return static_cast<uint32_t>(halfword(index)) |
           (static_cast<uint32_t>(halfword(index + 256)) << 16);
  };

  if (entryCount == 16) {
    // 32-bit, 4-bit index: 16 entries at (CSA & 15) * 16.
    const uint32_t base = (csa & 15u) * 16u;
    for (uint32_t i = 0; i < 16; ++i) {
      entries[i] = combine(base + i);
    }
    return entries;
  }

  // 32-bit, 8-bit index: 16 groups of 16 entries; group g starts at
  // min(g * 16 + (CSA & 15) * 16, 240) and clamps to the last group, it does not wrap.
  for (uint32_t group = 0; group < 16; ++group) {
    const uint32_t start = std::min<uint32_t>(group * 16u + (csa & 15u) * 16u, 240u);
    for (uint32_t k = 0; k < 16; ++k) {
      entries[group * 16u + k] = combine(start + k);
    }
  }
  return entries;
}

std::vector<uint8_t> PaletteBytes(const std::vector<uint32_t>& entries) {
  std::vector<uint8_t> bytes;
  bytes.reserve(entries.size() * 4);
  for (const uint32_t entry : entries) {
    bytes.push_back(static_cast<uint8_t>(entry & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((entry >> 8) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((entry >> 16) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((entry >> 24) & 0xFFu));
  }
  return bytes;
}

}  // namespace hw_gs_replace
