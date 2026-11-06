// HWR texture replacement: CLUT buffer and direct-texel expansion (chapters 01.6, 07.5, 11.3.3).
// Written from the TASK-026 specification (docs/hw-renderer-spec). Clean room (decision 0069).
// Copyright (c) 2026 GT4Recomp contributors, MIT.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "hw_gs_replace/gs_types.hpp"

namespace hw_gs_replace {

// 1 KiB CLUT buffer on chip: 512 halfwords (chapter 01.6). A 32-bit entry i stores its low
// 16 bits at halfword i and its high 16 bits at halfword i + 256.
struct ClutBuffer {
  std::array<uint16_t, 512> halfwords{};
};

// Expands one 16-bit texel or CLUT entry (chapter 01.5 / 07.5): R, G, B as 5-bit values
// shifted left by 3; A = TA1 when bit 15 is set, else TA0; A = 0 when AEM is set and the
// whole 16-bit value is 0. Returns R | G << 8 | B << 16 | A << 24 (RGBA8 bytes, little endian).
uint32_t Expand16BitToRgba(uint16_t value, TexaState texa);

// Expands a stored direct-format value (chapter 11.3.2 path B, direct formats) to RGBA8 packed
// little endian. CT32/Z32 are raw. CT24/Z24 use A = TA0 (0 when AEM and RGB = 0). CT16 family
// uses Expand16BitToRgba. Precondition: psm is not indexed.
uint32_t ExpandDirectTexel(Psm psm, uint32_t stored, TexaState texa);

// Decodes the expanded palette for an indexed draw (chapter 07.5), `entryCount` 16 or 256,
// as RGBA8 packed little endian. `clutPsm` is CT32/CT24 (32-bit CLUT, stored alpha kept) or
// CT16/CT16S (16-bit CLUT, expanded with TEXA).
std::vector<uint32_t> DecodeClut(const ClutBuffer& clut, Psm clutPsm, uint32_t csa,
                                 uint32_t entryCount, TexaState texa);

// Little-endian bytes of the expanded entries: 4 bytes per entry, R, G, B, A order.
std::vector<uint8_t> PaletteBytes(const std::vector<uint32_t>& entries);

}  // namespace hw_gs_replace
