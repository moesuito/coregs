// CTest fixtures for the HWR texture replacement core (TASK-027, M16 and M18 pure part).
// Vectors come from TASK-026 specification chapters 01, 04.3, 07.5 and 11. Synthetic cases use
// the reference memory of this directory (not PCSX2 ground truth: see the Q-07 pending test).
// Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "hw_gs_replace/clut.hpp"
#include "hw_gs_replace/images.hpp"
#include "hw_gs_replace/keys.hpp"
#include "hw_gs_replace/names.hpp"
#include "hw_gs_replace/packs.hpp"
#include "hw_gs_replace/xxh3.hpp"
#include "png_encode.hpp"
#include "reference_gs_memory.hpp"

namespace {

using namespace hw_gs_replace;
using hw_gs_replace::testing::EncodePng;
using hw_gs_replace::testing::ReferenceGsMemory;

int g_checks = 0;
int g_failures = 0;

void ReportFailure(const char* file, int line, const std::string& text) {
  ++g_failures;
  std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, text.c_str());
}

#define CHECK(condition)                                                \
  do {                                                                  \
    ++g_checks;                                                         \
    if (!(condition)) {                                                 \
      ReportFailure(__FILE__, __LINE__, #condition);                    \
    }                                                                   \
  } while (false)

#define CHECK_EQ(actual, expected)                                                       \
  do {                                                                                   \
    ++g_checks;                                                                          \
    const auto actualValue = (actual);                                                   \
    const auto expectedValue = (expected);                                               \
    if (!(actualValue == expectedValue)) {                                               \
      ReportFailure(__FILE__, __LINE__,                                                  \
                    std::string(#actual) + " = " +                                       \
                        std::to_string(static_cast<unsigned long long>(actualValue)) +   \
                        ", expected " +                                                  \
                        std::to_string(static_cast<unsigned long long>(expectedValue))); \
    }                                                                                    \
  } while (false)

std::vector<uint8_t> Le32(uint32_t value) {
  return {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
          static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24)};
}

void Append(std::vector<uint8_t>& out, const std::vector<uint8_t>& more) {
  out.insert(out.end(), more.begin(), more.end());
}

std::string Hex(uint64_t value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%llx", static_cast<unsigned long long>(value));
  return buffer;
}

// Reads the expanded RGBA value of a direct texel as stored in a Format P or G stream.
uint32_t DirectTexelRgba(Psm psm, const ReferenceGsMemory& memory, uint32_t bp, uint32_t bw,
                         uint32_t x, uint32_t y, TexaState texa) {
  return ExpandDirectTexel(psm, memory.ReadPixel(psm, bp, bw, x, y), texa);
}

void TestXxh3() {
  // Known XXH3-64 (seed 0) value of the empty input (xxHash 0.8.x test vector).
  CHECK_EQ(Xxh3Hash64({}), 0x2D06800538D394C2ull);
  std::vector<uint8_t> data(1000);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>(i * 7 + 3);
  }
  Xxh3Stream stream;
  stream.Update(std::span<const uint8_t>(data.data(), 1));
  stream.Update(std::span<const uint8_t>(data.data() + 1, 300));
  stream.Update(std::span<const uint8_t>(data.data() + 301, 699));
  CHECK_EQ(stream.Digest(), Xxh3Hash64(data));
  // Digest does not disturb the stream.
  const uint64_t partial = stream.Digest();
  CHECK_EQ(partial, stream.Digest());
}

void TestBitsWord() {
  // Chapter 11.3 bits vectors.
  CHECK_EQ(PackBitsWord(Psm::Ct32, 7, 7, TexaState{}), 0x00001DC0u);
  CHECK_EQ(PackBitsWord(Psm::T8, 7, 7, TexaState{}), 0x00001DD3u);
  CHECK_EQ(PackBitsWord(Psm::T4, 8, 8, TexaState{}), 0x00002214u);
  CHECK_EQ(PackBitsWord(Psm::Ct16, 6, 6, TexaState{0x80, false, 0x80}), 0x80401982u);
  CHECK_EQ(PackBitsWord(Psm::T4, 4, 4, TexaState{}), 0x00001114u);
  // TEXA is masked for CT32 and indexed formats.
  CHECK_EQ(PackBitsWord(Psm::Ct32, 7, 7, TexaState{0x80, true, 0x80}), 0x00001DC0u);
  CHECK_EQ(PackBitsWord(Psm::T8, 7, 7, TexaState{0x80, true, 0x80}), 0x00001DD3u);
  // Parsing clears bit 14, and packing the fields back gives the same word.
  const BitsFields fields = UnpackBitsWord(0x80401982u | (1u << 14));
  CHECK_EQ(fields.psmCode, 2u);
  CHECK_EQ(fields.tw, 6u);
  CHECK_EQ(fields.th, 6u);
  CHECK(fields.texa == (TexaState{0x80, false, 0x80}));
  CHECK_EQ(PackBitsWord(PsmFromCode(fields.psmCode), fields.tw, fields.th, fields.texa),
           0x80401982u);
  CHECK_EQ(UnpackBitsWord(0x00005DC0u).psmCode, 0u);
}

void TestTexaExpansion() {
  // Chapter 01.5 and 07.5 vectors.
  CHECK_EQ(Expand16BitToRgba(0x8000, TexaState{0, false, 0x80}), 0x80000000u);
  CHECK_EQ(Expand16BitToRgba(0x0000, TexaState{0x80, true, 0}), 0u);
  CHECK_EQ(Expand16BitToRgba(0x7FFF, TexaState{0x80, false, 0}), 0x80F8F8F8u);
  // CT24: A = TA0, zero when AEM and RGB = 0.
  CHECK_EQ(ExpandDirectTexel(Psm::Ct24, 0x00112233u, TexaState{0x40, false, 0}), 0x40112233u);
  CHECK_EQ(ExpandDirectTexel(Psm::Ct24, 0x00000000u, TexaState{0x40, true, 0}), 0u);
  // CT32 is raw.
  CHECK_EQ(ExpandDirectTexel(Psm::Ct32, 0x12345678u, TexaState{0x40, true, 0x10}), 0x12345678u);
}

void TestRegions() {
  // Chapter 04.3 rules.
  Region region = ResolveRegion(WrapAxis{WrapMode::RegionClamp, 8, 15}, 5);
  CHECK(region.present && region.minimum == 8 && region.end == 16 && region.Size() == 8);
  region = ResolveRegion(WrapAxis{WrapMode::RegionClamp, 0, 31}, 5);
  CHECK(!region.present);
  region = ResolveRegion(WrapAxis{WrapMode::RegionClamp, 0, 63}, 5);
  CHECK(region.present && region.minimum == 0 && region.end == 64);
  // REGION_REPEAT: MINU = 0x18, MAXU = 0x10 gives [0x10, 0x19).
  region = ResolveRegion(WrapAxis{WrapMode::RegionRepeat, 0x18, 0x10}, 5);
  CHECK(region.present && region.minimum == 0x10 && region.end == 0x19);
  region = ResolveRegion(WrapAxis{WrapMode::RegionRepeat, 0, 5}, 5);
  CHECK(!region.present);
  region = ResolveRegion(WrapAxis{WrapMode::Clamp, 8, 15}, 5);
  CHECK(!region.present);
  // Mip shift (at least 1 texel).
  region = ShiftRegionForMip(Region{true, 8, 16}, 1);
  CHECK(region.present && region.minimum == 4 && region.end == 8);
  region = ShiftRegionForMip(Region{true, 0, 1}, 3);
  CHECK(region.present && region.minimum == 0 && region.end == 1);
}

void TestClutDecode() {
  // 32-bit, 8-bit index, CSA = 8: groups 0..7 read halfwords 128..240, groups 8..15 read group 15.
  ClutBuffer clut;
  for (uint32_t i = 0; i < 256; ++i) {
    clut.halfwords[i] = static_cast<uint16_t>(i);
  }
  const std::vector<uint32_t> eight = DecodeClut(clut, Psm::Ct32, 8, 256, TexaState{});
  CHECK_EQ(eight[0], 128u);
  CHECK_EQ(eight[127], 255u);
  CHECK_EQ(eight[112], 240u);
  CHECK_EQ(eight[128], 240u);
  CHECK_EQ(eight[255], 255u);
  // 32-bit, 4-bit index, CSA = 3: 16 entries at halfword 48.
  const std::vector<uint32_t> four = DecodeClut(clut, Psm::Ct32, 3, 16, TexaState{});
  CHECK_EQ(four[0], 48u);
  CHECK_EQ(four[15], 63u);
  // 16-bit CLUT at CSA = 1 expands the entries with TEXA.
  ClutBuffer clut16;
  clut16.halfwords[16] = 0x7FFF;
  const std::vector<uint32_t> sixteen =
      DecodeClut(clut16, Psm::Ct16, 1, 16, TexaState{0x80, false, 0});
  CHECK_EQ(sixteen[0], 0x80F8F8F8u);
  // Palette bytes are RGBA in little-endian order.
  const std::vector<uint8_t> bytes = PaletteBytes({0x80F8F8F8u});
  CHECK(bytes.size() == 4 && bytes[0] == 0xF8 && bytes[3] == 0x80);
}

void TestFormatGNames() {
  FormatGKey key;
  key.textureHash = 0x1234;
  key.paletteHash = 0;
  key.width = 64;
  key.height = 32;
  CHECK(FormatGFileName(key, ImageExtension::Png) ==
        "t-0000000000001234-0000000000000000-64x32.png");
  const auto parsed = ParseFormatGFileName(FormatGFileName(key, ImageExtension::Dds));
  CHECK(parsed.has_value() && parsed->key == key && parsed->extension == ImageExtension::Dds);
  // Upper-case hex and upper-case extension parse.
  const auto upper = ParseFormatGFileName("t-ABCDEF0123456789-0000000000000000-8x8.PNG");
  CHECK(upper.has_value() && upper->key.textureHash == 0xABCDEF0123456789ull);
  // Wrong digit count, wrong extension and extra fields are refused.
  CHECK(!ParseFormatGFileName("t-1234-0000000000000000-8x8.png").has_value());
  CHECK(!ParseFormatGFileName("t-0000000000001234-0000000000000000-8x8.jpg").has_value());
  CHECK(!ParseFormatGFileName("t-0000000000001234-0000000000000000-8x8-mip1.png").has_value());
}

void TestFormatPNames() {
  FormatPRecord direct;
  direct.textureHash = 0xabc;
  direct.bits = 0x1DC0;
  CHECK(FormatPDumpName(direct, ImageExtension::Png) == "abc-00001dc0.png");

  FormatPRecord indexed = direct;
  indexed.indexed = true;
  indexed.clutHash = 0x1234;
  indexed.bits = 0x1DD3;
  CHECK(FormatPDumpName(indexed, ImageExtension::Png) == "abc-1234-00001dd3.png");

  FormatPRecord regionOnY = direct;
  regionOnY.regionWidth = 0;
  regionOnY.regionHeight = 64;
  regionOnY.bits = 0x2214;
  CHECK(FormatPDumpName(regionOnY, ImageExtension::Png) == "abc-r0x64-00002214.png");

  FormatPRecord mip = direct;
  mip.bits = 0x2214;
  mip.mipLevel = 1;
  CHECK(FormatPDumpName(mip, ImageExtension::Png) == "abc-00002214-mip1.png");

  // Round trip of a dump name, with case and leading zeros accepted on parse.
  const auto parsed = ParseFormatPFileName("ABC-1234-r256x0-00001DD3.PNG");
  CHECK(parsed.has_value() && parsed->textureHash == 0xabc && parsed->clutPresent &&
        parsed->clutHash == 0x1234 && parsed->regionWidth == 256 && parsed->regionHeight == 0 &&
        parsed->bits == 0x1DD3 && parsed->extension == ImageExtension::Png);
  const auto leadingZeros = ParseFormatPFileName("00abc-000001dc0.dds");
  CHECK(leadingZeros.has_value() && leadingZeros->textureHash == 0xabc &&
        leadingZeros->bits == 0x1DC0 && leadingZeros->extension == ImageExtension::Dds);

  // Bit 14 is cleared on parse.
  const auto bit14 = ParseFormatPFileName("abc-00005dc0.png");
  CHECK(bit14.has_value() && bit14->bits == 0x1DC0);

  // Refusals: mip names, missing dot or bits, bad extensions, too many fields.
  CHECK(!ParseFormatPFileName("abc-00001dc0-mip1.png").has_value());
  CHECK(!ParseFormatPFileName("abc-00001dc0").has_value());
  CHECK(!ParseFormatPFileName("abc00001dc0.png").has_value());
  CHECK(!ParseFormatPFileName("abc-00001dc0.jpg").has_value());
  CHECK(!ParseFormatPFileName("abc-1-2-3-00001dc0.png").has_value());
  CHECK(!ParseFormatPFileName("abc-00001dc0x.png").has_value());

  // Legacy region token r<packed64>: minX = 0x10, maxX = 0x50, minY = 0, maxY = 0x40.
  const uint64_t packed = 0x10ull | (0x50ull << 16) | (0x40ull << 48);
  const std::string legacy = "abc-r" + Hex(packed) + "-00001dc0.png";
  const auto legacyParsed = ParseFormatPFileName(legacy);
  CHECK(legacyParsed.has_value() && legacyParsed->regionWidth == 64 &&
        legacyParsed->regionHeight == 64 && !legacyParsed->clutPresent);
  const auto legacyIndexed = ParseFormatPFileName("abc-1234-r" + Hex(packed) + "-00001dd3.png");
  CHECK(legacyIndexed.has_value() && legacyIndexed->clutPresent && legacyIndexed->regionWidth == 64);

  // Matching: indexed records need the CLUT field, direct records must not have one.
  FormatPRecord record = indexed;
  ParsedFormatPName name = *ParseFormatPFileName("abc-1234-00001dd3.png");
  CHECK(FormatPNameMatches(name, record));
  CHECK(!FormatPNameMatches(*ParseFormatPFileName("abc-00001dd3.png"), record));
  CHECK(!FormatPNameMatches(*ParseFormatPFileName("abc-1234-00001dc0.png"), direct));
  CHECK(!FormatPNameMatches(*ParseFormatPFileName("abc-1234-00001dc0.png"), indexed));
  CHECK(FormatPNameMatches(*ParseFormatPFileName("abc-00001dc0.png"), direct));
}

void TestFormatGKeysSynthetic() {
  // Case G1: 2x2 CT32 texture, stream = the four texels in raster order.
  ReferenceGsMemory memory;
  TexelSource& source = memory;
  TextureDescription desc;
  desc.psm = Psm::Ct32;
  desc.tw = 1;
  desc.th = 1;
  desc.tbp = 0;
  desc.tbw = 1;
  std::vector<uint8_t> expected;
  for (uint32_t y = 0; y < 2; ++y) {
    for (uint32_t x = 0; x < 2; ++x) {
      const uint32_t value = 0x11223344u + x * 0x10u + y * 0x100u;
      memory.WritePixel(Psm::Ct32, 0, 1, x, y, value);
      Append(expected, Le32(value));
    }
  }
  const FormatGKey key = ComputeFormatGKey(source, desc);
  CHECK_EQ(key.textureHash, Xxh3Hash64(expected));
  CHECK_EQ(key.paletteHash, 0ull);
  CHECK_EQ(key.width, 2u);
  CHECK_EQ(key.height, 2u);

  // Case G2: region [1, 2) on X gives a 1x2 key over column 1 only.
  desc.wrapU = WrapAxis{WrapMode::RegionClamp, 1, 1};
  std::vector<uint8_t> column;
  for (uint32_t y = 0; y < 2; ++y) {
    Append(column, Le32(0x11223344u + 0x10u + y * 0x100u));
  }
  const FormatGKey regionKey = ComputeFormatGKey(source, desc);
  CHECK_EQ(regionKey.textureHash, Xxh3Hash64(column));
  CHECK_EQ(regionKey.width, 1u);
  CHECK_EQ(regionKey.height, 2u);

  // Case G3: T4 16x16 with a palette; stream = one index byte per texel, palette hash from 16
  // expanded entries.
  ReferenceGsMemory indexedMemory;
  ClutBuffer clut;
  for (uint32_t i = 0; i < 512; ++i) {
    clut.halfwords[i] = static_cast<uint16_t>(i * 3);
  }
  TextureDescription t4;
  t4.psm = Psm::T4;
  t4.tw = 4;
  t4.th = 4;
  t4.tbp = 0x100;
  t4.tbw = 2;
  t4.clut = &clut;
  t4.clutPsm = Psm::Ct32;
  t4.csa = 0;
  std::vector<uint8_t> indices;
  for (uint32_t y = 0; y < 16; ++y) {
    for (uint32_t x = 0; x < 16; ++x) {
      const uint32_t index = (x + y) & 15u;
      indexedMemory.WritePixel(Psm::T4, 0x100, 2, x, y, index);
      indices.push_back(static_cast<uint8_t>(index));
    }
  }
  const FormatGKey t4Key = ComputeFormatGKey(indexedMemory, t4);
  CHECK_EQ(t4Key.textureHash, Xxh3Hash64(indices));
  const std::vector<uint32_t> entries = DecodeClut(clut, Psm::Ct32, 0, 16, TexaState{});
  CHECK_EQ(t4Key.paletteHash, Xxh3Hash64(PaletteBytes(entries)));
  CHECK_EQ(t4Key.width, 16u);
  CHECK_EQ(t4Key.height, 16u);

  // The name round-trips through the parser.
  const std::string name = FormatGFileName(t4Key, ImageExtension::Png);
  const auto parsed = ParseFormatGFileName(name);
  CHECK(parsed.has_value() && parsed->key == t4Key);
}

void TestFormatPPathA() {
  // Case P1 (chapter 11.3 vector): CT32, TBP0 0x1000, TBW 2, TW = TH = 7, no region, no mips.
  // Path A: 256 blocks in raster block order, bwPages = 2, bits 0x1DC0.
  ReferenceGsMemory memory;
  for (uint32_t y = 0; y < 128; ++y) {
    for (uint32_t x = 0; x < 128; ++x) {
      memory.WritePixel(Psm::Ct32, 0x1000, 2, x, y, (x * 31u) ^ (y * 17u) ^ 0x5A5A0000u);
    }
  }
  std::vector<uint32_t> expectedBlocks;
  std::vector<uint8_t> expected;
  for (uint32_t by = 0; by < 16; ++by) {
    for (uint32_t bx = 0; bx < 16; ++bx) {
      const uint32_t pageX = bx / 8;
      const uint32_t pageY = by / 4;
      static const uint32_t table[4][8] = {{0, 1, 4, 5, 16, 17, 20, 21},
                                           {2, 3, 6, 7, 18, 19, 22, 23},
                                           {8, 9, 12, 13, 24, 25, 28, 29},
                                           {10, 11, 14, 15, 26, 27, 30, 31}};
      const uint32_t block = (0x1000u + 32u * (pageY * 2u + pageX) + table[by % 4][bx % 8]) & 16383u;
      expectedBlocks.push_back(block);
      std::array<uint8_t, 256> bytes{};
      memory.ReadBlock(block, bytes);
      Append(expected, std::vector<uint8_t>(bytes.begin(), bytes.end()));
    }
  }
  memory.ClearBlockLog();

  TextureDescription desc;
  desc.psm = Psm::Ct32;
  desc.tw = 7;
  desc.th = 7;
  desc.tbp = 0x1000;
  desc.tbw = 2;
  const FormatPKeys keys = ComputeFormatPKeys(memory, desc);
  CHECK_EQ(keys.fullChain.textureHash, Xxh3Hash64(expected));
  CHECK(!keys.baseOnly.has_value());
  CHECK_EQ(keys.fullChain.bits, 0x1DC0u);
  CHECK_EQ(keys.fullChain.regionWidth, 0u);
  CHECK_EQ(memory.BlockLog().size(), expectedBlocks.size());
  CHECK(memory.BlockLog() == expectedBlocks);
  CHECK(FormatPDumpName(keys.fullChain, ImageExtension::Png) ==
        Hex(keys.fullChain.textureHash) + "-00001dc0.png");
}

void TestFormatPZXorAfterAdding() {
  // Case P2: Z32 at a non-page-aligned TBP0 = 1. The block number is (TBP0 + 32 * page + entry)
  // and the XOR 24 is applied after the addition (chapter 01.3 Hypothesis, test V-01-4).
  ReferenceGsMemory memory;
  TextureDescription desc;
  desc.psm = Psm::Z32;
  desc.tw = 7;
  desc.th = 7;
  desc.tbp = 1;
  desc.tbw = 2;
  memory.ClearBlockLog();
  (void)ComputeFormatPKeys(memory, desc);
  std::vector<uint32_t> expected;
  static const uint32_t table[4][8] = {{0, 1, 4, 5, 16, 17, 20, 21},
                                       {2, 3, 6, 7, 18, 19, 22, 23},
                                       {8, 9, 12, 13, 24, 25, 28, 29},
                                       {10, 11, 14, 15, 26, 27, 30, 31}};
  for (uint32_t by = 0; by < 16; ++by) {
    for (uint32_t bx = 0; bx < 16; ++bx) {
      const uint32_t page = (by / 4) * 2 + bx / 8;
      expected.push_back(((1u + 32u * page + table[by % 4][bx % 8]) ^ 24u) & 16383u);
    }
  }
  CHECK(memory.BlockLog() == expected);
}

void TestFormatPPathB() {
  // Case P3 (chapter 11.3 vector): T4 16x16, path B: 256 index bytes and a 64-byte CLUT.
  ReferenceGsMemory memory;
  ClutBuffer clut;
  for (uint32_t i = 0; i < 512; ++i) {
    clut.halfwords[i] = static_cast<uint16_t>(i * 5 + 1);
  }
  std::vector<uint8_t> indices;
  for (uint32_t y = 0; y < 16; ++y) {
    for (uint32_t x = 0; x < 16; ++x) {
      const uint32_t index = (x * 3 + y) & 15u;
      memory.WritePixel(Psm::T4, 0x200, 2, x, y, index);
      indices.push_back(static_cast<uint8_t>(index));
    }
  }
  TextureDescription desc;
  desc.psm = Psm::T4;
  desc.tw = 4;
  desc.th = 4;
  desc.tbp = 0x200;
  desc.tbw = 2;
  desc.clut = &clut;
  desc.clutPsm = Psm::Ct32;
  const FormatPKeys keys = ComputeFormatPKeys(memory, desc);
  CHECK_EQ(keys.fullChain.textureHash, Xxh3Hash64(indices));
  const std::vector<uint32_t> entries = DecodeClut(clut, Psm::Ct32, 0, 16, TexaState{});
  CHECK_EQ(keys.fullChain.clutHash, Xxh3Hash64(PaletteBytes(entries)));
  CHECK(keys.fullChain.indexed);
  CHECK_EQ(keys.fullChain.bits, 0x1114u);
  CHECK(FormatPDumpName(keys.fullChain, ImageExtension::Png) ==
        Hex(keys.fullChain.textureHash) + "-" + Hex(keys.fullChain.clutHash) + "-00001114.png");
}

void TestFormatPRegionQuirk() {
  // Case P4 (synthetic, chapter 11.3.2 quirk): CT32 8x8 with region [1, 5) on X. Path B uses the
  // block rect [0, 8) with pitch 32. Start = row * pitch + (rect.left - block.left) BYTES, so each
  // row hashes bytes 1..16 of the row buffer (not texels 1..4).
  ReferenceGsMemory memory;
  TexaState none{};
  for (uint32_t y = 0; y < 8; ++y) {
    for (uint32_t x = 0; x < 8; ++x) {
      memory.WritePixel(Psm::Ct32, 0, 1, x, y, 0xA0000000u | (y << 8) | x);
    }
  }
  std::vector<uint8_t> expected;
  for (uint32_t y = 0; y < 8; ++y) {
    std::vector<uint8_t> row(32, 0);
    for (uint32_t x = 0; x < 8; ++x) {
      const std::vector<uint8_t> texel = Le32(DirectTexelRgba(Psm::Ct32, memory, 0, 1, x, y, none));
      std::copy(texel.begin(), texel.end(), row.begin() + x * 4);
    }
    Append(expected, std::vector<uint8_t>(row.begin() + 1, row.begin() + 17));
  }
  TextureDescription desc;
  desc.psm = Psm::Ct32;
  desc.tw = 3;
  desc.th = 3;
  desc.tbp = 0;
  desc.tbw = 1;
  desc.wrapU = WrapAxis{WrapMode::RegionClamp, 1, 4};
  const FormatPKeys keys = ComputeFormatPKeys(memory, desc);
  CHECK_EQ(keys.fullChain.textureHash, Xxh3Hash64(expected));
  CHECK_EQ(keys.fullChain.regionWidth, 4u);
  CHECK_EQ(keys.fullChain.regionHeight, 0u);
  CHECK(FormatPDumpName(keys.fullChain, ImageExtension::Png) ==
        Hex(keys.fullChain.textureHash) + "-r4x0-00000cc0.png");
}

void TestMipChainAndUploadInference() {
  // Case P5: mips hash into the same stream. Base-only key is kept for packs made without mips.
  ReferenceGsMemory memory;
  for (uint32_t y = 0; y < 4; ++y) {
    for (uint32_t x = 0; x < 4; ++x) {
      memory.WritePixel(Psm::Ct32, 0, 1, x, y, x + y * 16u);
    }
  }
  for (uint32_t y = 0; y < 2; ++y) {
    for (uint32_t x = 0; x < 2; ++x) {
      memory.WritePixel(Psm::Ct32, 0x40, 1, x, y, 0x77u + x + y);
    }
  }
  TextureDescription desc;
  desc.psm = Psm::Ct32;
  desc.tw = 2;
  desc.th = 2;
  desc.tbp = 0;
  desc.tbw = 1;
  desc.mipLevels = {MipLevelAddress{0x40, 1}};
  desc.mipmappingActive = false;
  const FormatPKeys noMips = ComputeFormatPKeys(memory, desc);
  CHECK(!noMips.baseOnly.has_value());
  desc.mipmappingActive = true;
  const FormatPKeys withMips = ComputeFormatPKeys(memory, desc);
  CHECK(withMips.baseOnly.has_value());
  CHECK_EQ(withMips.baseOnly->textureHash, noMips.fullChain.textureHash);
  CHECK(withMips.fullChain.textureHash != withMips.baseOnly->textureHash);

  // Case P6: chapter 04.6 upload inference. Upload 64x32 and read 40x20 inside a 128x128 texture
  // shrink both axes to [0, 64) x [0, 32).
  TextureDescription inferred;
  inferred.psm = Psm::Ct32;
  inferred.tw = 7;
  inferred.th = 7;
  inferred.tbp = 0;
  inferred.tbw = 2;
  inferred.wrapU = WrapAxis{WrapMode::Clamp, 0, 0};
  inferred.wrapV = WrapAxis{WrapMode::Clamp, 0, 0};
  inferred.exactUploadInLast10Batches = UploadExtent{64, 32};
  inferred.readExtent = UploadExtent{40, 20};
  const FormatPKeys shrunk = ComputeFormatPKeys(memory, inferred);
  CHECK_EQ(shrunk.fullChain.regionWidth, 64u);
  CHECK_EQ(shrunk.fullChain.regionHeight, 32u);
  // Read 200x20: the new width is not smaller than 128, so there is no region.
  inferred.readExtent = UploadExtent{200, 20};
  const FormatPKeys notShrunk = ComputeFormatPKeys(memory, inferred);
  CHECK_EQ(notShrunk.fullChain.regionWidth, 0u);
}

void TestPackScan() {
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() /
                        ("hw_gs_replace_scan_" + std::to_string(static_cast<unsigned long long>(
                                                     std::chrono::steady_clock::now()
                                                         .time_since_epoch()
                                                         .count())));
  fs::create_directories(root / "SCUS-97328" / "replacements" / "sub");
  fs::create_directories(root / "SCUS-97328" / "replacements" / ".hidden");
  fs::create_directories(root / "SCUS-97328" / "gt4");
  const auto write = [](const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
  };
  write(root / "SCUS-97328" / "replacements" / "sub" / "abc-00001dc0.png", "a");
  write(root / "SCUS-97328" / "replacements" / ".hidden" / "abc-00001dc0.dds", "b");
  write(root / "SCUS-97328" / "replacements" / "notes.txt", "c");
  write(root / "SCUS-97328" / "replacements" / "1234-1234-00001dd3.png", "d");
  write(root / "SCUS-97328" / "gt4" / "t-0000000000001234-0000000000000000-8x8.png", "e");
  write(root / "SCUS-97328.zip", "not a zip");

  const UnsupportedArchiveReader archives;
  hw_gs_replace::ReplacementIndex index;
  const ScanReport report = index.ScanFormatP(root, "SCUS-97328", archives);
  CHECK_EQ(report.duplicatesIgnored, 1u);
  CHECK_EQ(report.unparsedNames, 1u);
  CHECK_EQ(report.archivesSkipped, 1u);
  CHECK_EQ(index.FormatPCount(), 2u);
  // Sorted registration: ".hidden" sorts before "sub", so the .dds copy wins.
  FormatPRecord record;
  record.textureHash = 0xabc;
  record.bits = 0x1DC0;
  const std::optional<PackEntry> found = index.FindFormatP(record);
  CHECK(found.has_value() && found->basename == "abc-00001dc0.dds");
  CHECK(found.has_value() && found->extension == ImageExtension::Dds);
  std::vector<uint8_t> bytes;
  std::string error;
  CHECK(found.has_value() && ReadPackEntryBytes(*found, archives, bytes, error) && bytes.size() == 1 &&
        bytes[0] == 'b');

  const ScanReport gReport = index.ScanFormatG(root, "SCUS-97328", archives);
  CHECK_EQ(gReport.filesRegistered, 1u);
  FormatGKey gKey;
  gKey.textureHash = 0x1234;
  gKey.width = 8;
  gKey.height = 8;
  CHECK(index.FindFormatG(gKey).has_value());

  std::error_code removeError;
  fs::remove_all(root, removeError);
}

void TestImages() {
  // PNG RGB 2x2: alpha becomes 0x80 under both policies.
  const std::vector<uint8_t> rgb = {10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 2, 3};
  const std::vector<uint8_t> rgbPng = EncodePng(2, 2, 8, 2, rgb, 3);
  ReplacementImage image;
  std::string error;
  CHECK(DecodeReplacementImage(ImageExtension::Png, rgbPng, AlphaPolicy::FormatP, image, error));
  CHECK(image.codec == ReplacementCodec::Rgba8 && image.width == 2 && image.height == 2);
  CHECK(image.levels.size() == 1 && image.levels[0].bytes.size() == 16);
  CHECK(image.levels[0].bytes[0] == 10 && image.levels[0].bytes[3] == 0x80);
  CHECK(image.levels[0].bytes[8] == 70 && image.levels[0].bytes[10] == 90 && image.levels[0].bytes[11] == 0x80);

  // PNG RGBA copies alpha.
  const std::vector<uint8_t> rgba = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  CHECK(DecodeReplacementImage(ImageExtension::Png, EncodePng(2, 2, 8, 6, rgba, 4),
                               AlphaPolicy::FormatG, image, error));
  CHECK(image.levels[0].bytes[3] == 4 && image.levels[0].bytes[15] == 16);

  // Palette, grey and 16-bit PNGs are refused.
  CHECK(!DecodeReplacementImage(ImageExtension::Png, EncodePng(2, 2, 8, 3, {0, 1, 2, 3}, 1),
                                AlphaPolicy::FormatP, image, error));
  CHECK(!DecodeReplacementImage(ImageExtension::Png, EncodePng(2, 2, 8, 0, {0, 1, 2, 3}, 1),
                                AlphaPolicy::FormatP, image, error));
  CHECK(!DecodeReplacementImage(ImageExtension::Png,
                                EncodePng(2, 2, 16, 2, std::vector<uint8_t>(24, 0), 6),
                                AlphaPolicy::FormatP, image, error));

  // DDS helpers.
  const auto header = [](uint32_t width, uint32_t height, uint32_t mipCount, uint32_t pfFlags,
                         const char* fourCc, uint32_t bitCount, uint32_t masks[4], bool dx10,
                         uint32_t dxgi) {
    std::vector<uint8_t> out = {'D', 'D', 'S', ' '};
    Append(out, Le32(124));
    Append(out, Le32(0x1007));
    Append(out, Le32(height));
    Append(out, Le32(width));
    Append(out, Le32(0));
    Append(out, Le32(0));
    Append(out, Le32(mipCount));
    for (int i = 0; i < 11; ++i) {
      Append(out, Le32(0));
    }
    Append(out, Le32(32));
    Append(out, Le32(pfFlags));
    out.insert(out.end(), fourCc, fourCc + 4);
    Append(out, Le32(bitCount));
    for (int i = 0; i < 4; ++i) {
      Append(out, Le32(masks != nullptr ? masks[i] : 0));
    }
    Append(out, Le32(0x1000));
    for (int i = 0; i < 4; ++i) {
      Append(out, Le32(0));
    }
    if (dx10) {
      Append(out, Le32(dxgi));
      Append(out, Le32(3));
      Append(out, Le32(0));
      Append(out, Le32(1));
      Append(out, Le32(0));
    }
    return out;
  };

  // BC1 4x4, one level, 8 bytes.
  std::vector<uint8_t> dds = header(4, 4, 1, 0x4, "DXT1", 0, nullptr, false, 0);
  dds.resize(dds.size() + 8, 0x11);
  CHECK(DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatP, image, error));
  CHECK(image.codec == ReplacementCodec::Bc1 && image.levels.size() == 1 &&
        image.levels[0].bytes.size() == 8);

  // DXT5 8x8 with a full chain: 64 + 16 + 16 + 16 bytes.
  dds = header(8, 8, 0, 0x4, "DXT5", 0, nullptr, false, 0);
  dds.resize(dds.size() + 112, 0x22);
  CHECK(DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatP, image, error));
  CHECK(image.codec == ReplacementCodec::Bc3 && image.levels.size() == 4);
  CHECK(image.levels[0].bytes.size() == 64 && image.levels[3].bytes.size() == 16);

  // Truncated chain: only the levels that are fully present are read.
  dds = header(8, 8, 0, 0x4, "DXT5", 0, nullptr, false, 0);
  dds.resize(dds.size() + 80, 0x22);
  CHECK(DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatP, image, error));
  CHECK_EQ(image.levels.size(), 2u);

  // DX10 BC7 accepted; sRGB BC1 (72) refused; BC width 6 refused.
  dds = header(4, 4, 1, 0x4, "DX10", 0, nullptr, true, 98);
  dds.resize(dds.size() + 16, 0x33);
  CHECK(DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatP, image, error));
  CHECK(image.codec == ReplacementCodec::Bc7);
  dds = header(4, 4, 1, 0x4, "DX10", 0, nullptr, true, 72);
  dds.resize(dds.size() + 8, 0);
  CHECK(!DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatP, image, error));
  dds = header(6, 4, 1, 0x4, "DXT1", 0, nullptr, false, 0);
  dds.resize(dds.size() + 8, 0);
  CHECK(!DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatP, image, error));

  // Uncompressed 4x4 layouts. Memory bytes of one pixel are little-endian masked channels.
  const auto uncompressed = [&](uint32_t bitCount, uint32_t masks[4], std::vector<uint8_t> pixel) {
    std::vector<uint8_t> out = header(4, 4, 1, 0x40, "\0\0\0\0", bitCount, masks, false, 0);
    for (int i = 0; i < 16; ++i) {
      out.insert(out.end(), pixel.begin(), pixel.end());
    }
    return out;
  };
  // Pixel R=30, G=20, B=10 in memory order B, G, R, X (X8R8G8B8) and R, G, B, X (X8B8G8R8).
  uint32_t xr8g8b8[4] = {0xFF0000u, 0xFF00u, 0xFFu, 0};
  uint32_t xb8g8r8[4] = {0xFFu, 0xFF00u, 0xFF0000u, 0};
  uint32_t ar8g8b8[4] = {0xFF0000u, 0xFF00u, 0xFFu, 0xFF000000u};
  uint32_t rgb24[4] = {0xFF0000u, 0xFF00u, 0xFFu, 0};
  dds = uncompressed(32, xr8g8b8, {10, 20, 30, 0});
  CHECK(DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatP, image, error));
  CHECK(image.levels[0].bytes[0] == 30 && image.levels[0].bytes[2] == 10 &&
        image.levels[0].bytes[3] == 0xFF);
  CHECK(DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatG, image, error));
  CHECK(image.levels[0].bytes[3] == 0x80);
  dds = uncompressed(32, xb8g8r8, {30, 20, 10, 0});
  CHECK(DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatP, image, error));
  CHECK(image.levels[0].bytes[0] == 30 && image.levels[0].bytes[3] == 0x80);
  dds = uncompressed(32, ar8g8b8, {10, 20, 30, 0x40});
  CHECK(DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatG, image, error));
  CHECK(image.levels[0].bytes[3] == 0x40);
  dds = uncompressed(24, rgb24, {10, 20, 30});
  CHECK(DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatP, image, error));
  CHECK(image.levels[0].bytes[3] == 0xFF);
  CHECK(DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatG, image, error));
  CHECK(image.levels[0].bytes[3] == 0x80);
  uint32_t r5g6b5[4] = {0xF800u, 0x07E0u, 0x1Fu, 0};
  dds = header(4, 4, 1, 0x40, "\0\0\0\0", 16, r5g6b5, false, 0);
  dds.resize(dds.size() + 32, 0);
  CHECK(!DecodeReplacementImage(ImageExtension::Dds, dds, AlphaPolicy::FormatP, image, error));
}

void TestReferenceMemoryFixture() {
  // Chapter 01.3 block placement: pixel (8, 0) of a CT32 buffer with BP 0, BW 1 is block 1, and
  // its first word is the first word of block 1 (little endian).
  ReferenceGsMemory memory;
  memory.WritePixel(Psm::Ct32, 0, 1, 8, 0, 0xDEADBEEFu);
  std::array<uint8_t, 256> block{};
  memory.ReadBlock(1, block);
  CHECK(block[0] == 0xEF && block[1] == 0xBE && block[2] == 0xAD && block[3] == 0xDE);

  // V-01-3 (chapter 01.3, chapter 14): fill one CT32 page with R = x, G = y, B = 0x80 | x,
  // A = 0xC0 | y, read the page as T8 with the buffer width doubled. Each T8 lane is one byte:
  // (u & 8, v & 2) = (0,0) R, (0,2) G, (8,0) B, (8,2) A.
  ReferenceGsMemory page;
  for (uint32_t y = 0; y < 32; ++y) {
    for (uint32_t x = 0; x < 64; ++x) {
      page.WritePixel(Psm::Ct32, 0, 1, x, y,
                      x | (y << 8) | ((0x80u | x) << 16) | ((0xC0u | y) << 24));
    }
  }
  bool laneOk = true;
  for (uint32_t v = 0; v < 64; ++v) {
    for (uint32_t u = 0; u < 128; ++u) {
      const uint32_t value = page.ReadPixel(Psm::T8, 0, 2, u, v);
      const uint32_t lane = ((u & 8u) != 0 ? 2u : 0u) + ((v & 2u) != 0 ? 1u : 0u);
      bool ok = false;
      switch (lane) {
        case 0:
          ok = value < 64;  // R = x
          break;
        case 1:
          ok = value < 32;  // G = y
          break;
        case 2:
          ok = value >= 0x80 && value < 0xC0;  // B = 0x80 | x
          break;
        default:
          ok = value >= 0xC0 && value < 0xE0;  // A = 0xC0 | y
          break;
      }
      laneOk = laneOk && ok;
    }
  }
  CHECK(laneOk);
}

// Full-scope run. The Q-07 test (--pending-q07) exits with the CTest skip code instead.
int RunAll() {
  TestReferenceMemoryFixture();
  TestXxh3();
  TestBitsWord();
  TestTexaExpansion();
  TestRegions();
  TestClutDecode();
  TestFormatGNames();
  TestFormatPNames();
  TestFormatGKeysSynthetic();
  TestFormatPPathA();
  TestFormatPZXorAfterAdding();
  TestFormatPPathB();
  TestFormatPRegionQuirk();
  TestMipChainAndUploadInference();
  TestPackScan();
  TestImages();
  std::printf("hw_gs_replace: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--pending-q07") == 0) {
    std::printf(
        "PENDING Q-07: Format P end-to-end names need owner-generated PCSX2 dumps of a GT4 GS "
        "capture (TASK-026 open question Q-07). Skipping.\n");
    return 77;
  }
  return RunAll();
}
