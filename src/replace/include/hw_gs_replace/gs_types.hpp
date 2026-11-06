// HWR texture replacement: GS value types shared by the key, name and CLUT code.
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 01, 04.3,
// 07.5 and 11 (clean room, decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace hw_gs_replace {

// Pixel storage formats, chapter 01.3 PSM table (codes as in the GS registers).
enum class Psm : uint8_t {
  Ct32 = 0x00,
  Ct24 = 0x01,
  Ct16 = 0x02,
  Ct16S = 0x0A,
  T8 = 0x13,
  T4 = 0x14,
  T8H = 0x1B,
  T4HL = 0x24,
  T4HH = 0x2C,
  Z32 = 0x30,
  Z24 = 0x31,
  Z16 = 0x32,
  Z16S = 0x3A,
};

// True for the 13 codes of the chapter 01.3 table. Code 0x12 and every other value is false.
bool IsKnownPsmCode(uint32_t code);
// Precondition: IsKnownPsmCode(code).
Psm PsmFromCode(uint32_t code);
// T8, T4, T8H, T4HL, T4HH (chapter 01.3: indexed formats).
bool IsIndexedPsm(Psm psm);
// Z32, Z24, Z16, Z16S.
bool IsZPsm(Psm psm);
// CT24, CT16, CT16S, Z24, Z16, Z16S: the direct formats that carry TEXA fields.
bool PsmUsesTexa(Psm psm);

struct PsmGeometry {
  uint32_t pageWidth;
  uint32_t pageHeight;
  uint32_t blockWidth;
  uint32_t blockHeight;
};
PsmGeometry GeometryOf(Psm psm);

// Texture environment alpha registers (TEXA, chapter 01.5).
struct TexaState {
  uint8_t ta0 = 0;
  bool aem = false;
  uint8_t ta1 = 0;
  bool operator==(const TexaState&) const = default;
};

// TEXA masking: returns the fields that apply to the format. Zero for CT32, Z32 and all
// indexed formats (chapter 11.3 bits word).
TexaState MaskTexaForPsm(Psm psm, TexaState texa);

// Wrap modes of one texture axis (chapter 04.3). Only the two REGION modes create regions.
enum class WrapMode : uint8_t {
  Repeat = 0,
  Clamp = 1,
  RegionClamp = 2,
  RegionRepeat = 3,
};

struct WrapAxis {
  WrapMode mode = WrapMode::Repeat;
  // MINU/MAXU (or MINV/MAXV) register fields.
  uint32_t minimum = 0;
  uint32_t maximum = 0;
};

// A region [minimum, end) of one texture axis. Absent regions have present == false.
struct Region {
  bool present = false;
  uint32_t minimum = 0;
  uint32_t end = 0;
  // Size of the region; 0 when absent (the record width/height of chapter 11.3).
  uint32_t Size() const { return present ? end - minimum : 0; }
  bool operator==(const Region&) const = default;
};

// Chapter 04.3 region rules for one axis with log2 size `log2Size` (TW or TH).
Region ResolveRegion(const WrapAxis& axis, uint32_t log2Size);

// Chapter 04.3: a mip level shifts the region right by its level (size at least 1 texel).
// Hypothesis-level detail: the min and the size are both shifted (chapter 11.3.1 High rule).
Region ShiftRegionForMip(const Region& base, uint32_t mipOffset);

// Chapter 01.3 bits word (chapter 11.3): PSM 0..5, TW 6..9, TH 10..13, bit 14 unused,
// TA0 15..22, AEM 23, TA1 24..31.
struct BitsFields {
  uint32_t psmCode = 0;
  uint32_t tw = 0;
  uint32_t th = 0;
  TexaState texa{};
  bool operator==(const BitsFields&) const = default;
};
// Packs fields; TEXA is masked for the format and bit 14 is written as 0.
uint32_t PackBitsWord(Psm psm, uint32_t tw, uint32_t th, TexaState texa);
// Unpacks fields; bit 14 is ignored (cleared when parsing).
BitsFields UnpackBitsWord(uint32_t bits);

// Raw block and pixel access. The swizzle library implements this (M2); fixtures provide a
// reference implementation (tests/reference_gs_memory.cpp). Block numbers are 0..16383,
// each block 256 bytes in stored (swizzled) byte order.
class TexelSource {
 public:
  virtual ~TexelSource() = default;

  // Copies the 256 stored bytes of block `blockNumber` (already reduced mod 16384).
  virtual void ReadBlock(uint32_t blockNumber, std::span<uint8_t, 256> out) const = 0;

  // Stored value of pixel (x, y) of a buffer with base block pointer `basePointer` (block
  // units) and width `bufferWidth` (64-pixel units), in format `psm`:
  //   CT32/Z32: the 32-bit word; CT24/Z24: the low 24 bits; CT16 family: the 16-bit value;
  //   T8: the 8-bit index; T4: the 4-bit index (0..15);
  //   T8H: bits 24..31; T4HL: bits 24..27; T4HH: bits 28..31 (each as 0..255 / 0..15).
  virtual uint32_t ReadPixel(Psm psm, uint32_t basePointer, uint32_t bufferWidth, uint32_t x,
                             uint32_t y) const = 0;
  // `count` pixels of row y from x, as ReadPixel gives them. A source can do this faster than one call per pixel.
  virtual void ReadRow(Psm psm, uint32_t basePointer, uint32_t bufferWidth, uint32_t x, uint32_t y, uint32_t count,
                       uint32_t* out) const {
    for (uint32_t i = 0; i < count; ++i) {
      out[i] = ReadPixel(psm, basePointer, bufferWidth, x + i, y);
    }
  }
};

}  // namespace hw_gs_replace
