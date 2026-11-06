// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 01.3, 04.3, 11.3.
// Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#include "hw_gs_replace/gs_types.hpp"

#include <algorithm>

namespace hw_gs_replace {

bool IsKnownPsmCode(uint32_t code) {
  switch (code) {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x0A:
    case 0x13:
    case 0x14:
    case 0x1B:
    case 0x24:
    case 0x2C:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x3A:
      return true;
    default:
      return false;
  }
}

Psm PsmFromCode(uint32_t code) { return static_cast<Psm>(code); }

bool IsIndexedPsm(Psm psm) {
  switch (psm) {
    case Psm::T8:
    case Psm::T4:
    case Psm::T8H:
    case Psm::T4HL:
    case Psm::T4HH:
      return true;
    default:
      return false;
  }
}

bool IsZPsm(Psm psm) {
  return psm == Psm::Z32 || psm == Psm::Z24 || psm == Psm::Z16 || psm == Psm::Z16S;
}

bool PsmUsesTexa(Psm psm) {
  switch (psm) {
    case Psm::Ct24:
    case Psm::Ct16:
    case Psm::Ct16S:
    case Psm::Z24:
    case Psm::Z16:
    case Psm::Z16S:
      return true;
    default:
      return false;
  }
}

PsmGeometry GeometryOf(Psm psm) {
  switch (psm) {
    case Psm::Ct16:
    case Psm::Ct16S:
    case Psm::Z16:
    case Psm::Z16S:
      return {64, 64, 16, 8};
    case Psm::T8:
      return {128, 64, 16, 16};
    case Psm::T4:
      return {128, 128, 32, 16};
    case Psm::Ct32:
    case Psm::Ct24:
    case Psm::T8H:
    case Psm::T4HL:
    case Psm::T4HH:
    case Psm::Z32:
    case Psm::Z24:
      return {64, 32, 8, 8};
  }
  return {64, 32, 8, 8};
}

TexaState MaskTexaForPsm(Psm psm, TexaState texa) {
  if (!PsmUsesTexa(psm)) {
    return TexaState{};
  }
  return texa;
}

Region ResolveRegion(const WrapAxis& axis, uint32_t log2Size) {
  const uint32_t size = 1u << log2Size;
  Region region;
  if (axis.mode == WrapMode::RegionClamp) {
    // Chapter 04.3: MAXU >= MINU and (w < 2^TW or MAXU >= 2^TW) gives [MINU, MAXU + 1).
    if (axis.maximum >= axis.minimum) {
      const uint32_t width = axis.maximum - axis.minimum + 1;
      if (width < size || axis.maximum >= size) {
        region.present = true;
        region.minimum = axis.minimum;
        region.end = axis.maximum + 1;
      }
    }
  } else if (axis.mode == WrapMode::RegionRepeat) {
    // Chapter 04.3: MINU != 0, w = ((MINU | MAXU) - MAXU) + 1, and
    // (w < 2^TW or (MAXU != 0 and w <= 2^TW)) gives [MAXU, (MINU | MAXU) + 1).
    if (axis.minimum != 0) {
      const uint32_t combined = axis.minimum | axis.maximum;
      const uint32_t width = (combined - axis.maximum) + 1;
      if (width < size || (axis.maximum != 0 && width <= size)) {
        region.present = true;
        region.minimum = axis.maximum;
        region.end = combined + 1;
      }
    }
  }
  return region;
}

Region ShiftRegionForMip(const Region& base, uint32_t mipOffset) {
  if (!base.present) {
    return base;
  }
  Region shifted;
  shifted.present = true;
  shifted.minimum = base.minimum >> mipOffset;
  const uint32_t size = std::max<uint32_t>(1, base.Size() >> mipOffset);
  shifted.end = shifted.minimum + size;
  return shifted;
}

uint32_t PackBitsWord(Psm psm, uint32_t tw, uint32_t th, TexaState texa) {
  const TexaState masked = MaskTexaForPsm(psm, texa);
  uint32_t bits = static_cast<uint32_t>(psm) & 0x3Fu;
  bits |= (tw & 0xFu) << 6;
  bits |= (th & 0xFu) << 10;
  bits |= static_cast<uint32_t>(masked.ta0) << 15;
  bits |= (masked.aem ? 1u : 0u) << 23;
  bits |= static_cast<uint32_t>(masked.ta1) << 24;
  return bits;
}

BitsFields UnpackBitsWord(uint32_t bits) {
  BitsFields fields;
  fields.psmCode = bits & 0x3Fu;
  fields.tw = (bits >> 6) & 0xFu;
  fields.th = (bits >> 10) & 0xFu;
  fields.texa.ta0 = static_cast<uint8_t>((bits >> 15) & 0xFFu);
  fields.texa.aem = ((bits >> 23) & 1u) != 0;
  fields.texa.ta1 = static_cast<uint8_t>((bits >> 24) & 0xFFu);
  return fields;
}

}  // namespace hw_gs_replace
