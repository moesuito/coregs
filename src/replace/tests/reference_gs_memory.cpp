// Reference GS local memory transcribed from TASK-026 specification chapter 01.3 (block
// arrangement, pixel position inside a block, address computation). Test fixture only.
// Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#include "reference_gs_memory.hpp"

namespace hw_gs_replace::testing {
namespace {

// Chapter 01.3 "Block arrangement within a page": 8 wide, 4 high (CT32 family and T8).
constexpr uint32_t kBlocks8x4[4][8] = {
    {0, 1, 4, 5, 16, 17, 20, 21},
    {2, 3, 6, 7, 18, 19, 22, 23},
    {8, 9, 12, 13, 24, 25, 28, 29},
    {10, 11, 14, 15, 26, 27, 30, 31},
};
// 4 wide, 8 high (CT16 and T4).
constexpr uint32_t kBlocks4x8Ct16[8][4] = {
    {0, 2, 8, 10},
    {1, 3, 9, 11},
    {4, 6, 12, 14},
    {5, 7, 13, 15},
    {16, 18, 24, 26},
    {17, 19, 25, 27},
    {20, 22, 28, 30},
    {21, 23, 29, 31},
};
// 4 wide, 8 high (CT16S).
constexpr uint32_t kBlocks4x8Ct16S[8][4] = {
    {0, 2, 16, 18},
    {1, 3, 17, 19},
    {8, 10, 24, 26},
    {9, 11, 25, 27},
    {4, 6, 20, 22},
    {5, 7, 21, 23},
    {12, 14, 28, 30},
    {13, 15, 29, 31},
};
// Chapter 01.3 pixel position: 32-bit word index for (bx 0..7, by & 1).
constexpr uint32_t kWord32[2][8] = {
    {0, 1, 4, 5, 8, 9, 12, 13},
    {2, 3, 6, 7, 10, 11, 14, 15},
};
// Chapter 01.3 pixel position: halfword index for (bx 0..15, by & 1).
constexpr uint32_t kHalf16[2][16] = {
    {0, 2, 8, 10, 16, 18, 24, 26, 1, 3, 9, 11, 17, 19, 25, 27},
    {4, 6, 12, 14, 20, 22, 28, 30, 5, 7, 13, 15, 21, 23, 29, 31},
};

struct PageShape {
  uint32_t pageWidth;
  uint32_t pageHeight;
  uint32_t blockWidth;
  uint32_t blockHeight;
};

PageShape ShapeOf(Psm psm) {
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
    default:
      return {64, 32, 8, 8};
  }
}

const uint32_t* TableRowOf(Psm psm, uint32_t row, uint32_t& columns) {
  switch (psm) {
    case Psm::Ct16:
    case Psm::Z16:
    case Psm::T4:
      columns = 4;
      return &kBlocks4x8Ct16[row][0];
    case Psm::Ct16S:
    case Psm::Z16S:
      columns = 4;
      return &kBlocks4x8Ct16S[row][0];
    default:
      columns = 8;
      return &kBlocks8x4[row][0];
  }
}

}  // namespace

ReferenceGsMemory::ReferenceGsMemory() : memory_(4u << 20, 0) {}

ReferenceGsMemory::FieldSlot ReferenceGsMemory::SlotOf(Psm psm, uint32_t basePointer,
                                                       uint32_t bufferWidth, uint32_t x,
                                                       uint32_t y) const {
  const PageShape shape = ShapeOf(psm);
  const uint32_t pagesPerRow = (bufferWidth * 64u) / shape.pageWidth;
  const uint32_t page = (y / shape.pageHeight) * pagesPerRow + (x / shape.pageWidth);
  uint32_t columns = 0;
  const uint32_t* table = TableRowOf(psm, ((y % shape.pageHeight) / shape.blockHeight), columns);
  const uint32_t entry = table[(x % shape.pageWidth) / shape.blockWidth];
  uint32_t block = basePointer + 32u * page + entry;
  if (psm == Psm::Z32 || psm == Psm::Z24 || psm == Psm::Z16 || psm == Psm::Z16S) {
    block ^= 24u;  // chapter 01.3 / 11.3.2: XOR after adding the base
  }
  block &= 16383u;

  const uint32_t bx = x % shape.blockWidth;
  const uint32_t by = y % shape.blockHeight;
  const uint64_t blockBase = static_cast<uint64_t>(block) * 256u;

  switch (psm) {
    case Psm::Ct32:
    case Psm::Z32: {
      const uint32_t column = by / 2;
      const uint32_t word = kWord32[by & 1u][bx];
      return {(blockBase + 64u * column + 4u * word) * 8u, 32};
    }
    case Psm::Ct24:
    case Psm::Z24: {
      const uint32_t column = by / 2;
      const uint32_t word = kWord32[by & 1u][bx];
      return {(blockBase + 64u * column + 4u * word) * 8u, 24};
    }
    case Psm::T8H: {
      const uint32_t column = by / 2;
      const uint32_t word = kWord32[by & 1u][bx];
      return {(blockBase + 64u * column + 4u * word) * 8u + 24u, 8};
    }
    case Psm::T4HL: {
      const uint32_t column = by / 2;
      const uint32_t word = kWord32[by & 1u][bx];
      return {(blockBase + 64u * column + 4u * word) * 8u + 24u, 4};
    }
    case Psm::T4HH: {
      const uint32_t column = by / 2;
      const uint32_t word = kWord32[by & 1u][bx];
      return {(blockBase + 64u * column + 4u * word) * 8u + 28u, 4};
    }
    case Psm::Ct16:
    case Psm::Ct16S:
    case Psm::Z16:
    case Psm::Z16S: {
      const uint32_t column = by / 2;
      const uint32_t halfword = kHalf16[by & 1u][bx];
      return {(blockBase + 64u * column + 2u * halfword) * 8u, 16};
    }
    case Psm::T8: {
      const uint32_t column = by / 4;
      const uint32_t r = by % 4;
      const bool swapHalves = (((r >> 1) ^ (column & 1u)) & 1u) != 0;
      const uint32_t position = swapHalves ? ((bx & 7u) ^ 4u) : (bx & 7u);
      const uint32_t word = kWord32[r & 1u][position];
      const uint32_t byteInWord = 2u * ((bx >> 3) & 1u) + ((r >> 1) & 1u);
      return {(blockBase + 64u * column + 4u * word + byteInWord) * 8u, 8};
    }
    case Psm::T4: {
      const uint32_t column = by / 4;
      const uint32_t r = by % 4;
      const bool swapHalves = (((r >> 1) ^ (column & 1u)) & 1u) != 0;
      const uint32_t position = swapHalves ? ((bx & 7u) ^ 4u) : (bx & 7u);
      const uint32_t word = kWord32[r & 1u][position];
      const uint32_t nibble = 2u * ((bx >> 3) & 3u) + ((r >> 1) & 1u);
      // Nibble address = 2 * (byte address of the word) + nibble index; one nibble = 4 bits.
      const uint64_t nibbleAddress = 2u * (blockBase + 64u * column + 4u * word) + nibble;
      return {nibbleAddress * 4u, 4};
    }
  }
  return {0, 0};
}

uint32_t ReferenceGsMemory::ReadBits(uint64_t bitPosition, uint32_t width) const {
  uint32_t value = 0;
  for (uint32_t i = 0; i < width; ++i) {
    const uint64_t bit = bitPosition + i;
    const uint32_t byte = memory_[static_cast<size_t>(bit / 8)];
    value |= ((byte >> (bit % 8)) & 1u) << i;
  }
  return value;
}

void ReferenceGsMemory::WriteBits(uint64_t bitPosition, uint32_t width, uint32_t value) {
  for (uint32_t i = 0; i < width; ++i) {
    const uint64_t bit = bitPosition + i;
    uint8_t& byte = memory_[static_cast<size_t>(bit / 8)];
    const uint8_t mask = static_cast<uint8_t>(1u << (bit % 8));
    if (((value >> i) & 1u) != 0) {
      byte = static_cast<uint8_t>(byte | mask);
    } else {
      byte = static_cast<uint8_t>(byte & ~mask);
    }
  }
}

void ReferenceGsMemory::ReadBlock(uint32_t blockNumber, std::span<uint8_t, 256> out) const {
  blockLog_.push_back(blockNumber);
  const size_t base = static_cast<size_t>(blockNumber & 16383u) * 256u;
  for (size_t i = 0; i < 256; ++i) {
    out[i] = memory_[base + i];
  }
}

uint32_t ReferenceGsMemory::ReadPixel(Psm psm, uint32_t basePointer, uint32_t bufferWidth,
                                      uint32_t x, uint32_t y) const {
  const FieldSlot slot = SlotOf(psm, basePointer, bufferWidth, x, y);
  return ReadBits(slot.bitPosition, slot.bitWidth);
}

void ReferenceGsMemory::WritePixel(Psm psm, uint32_t basePointer, uint32_t bufferWidth,
                                   uint32_t x, uint32_t y, uint32_t value) {
  const FieldSlot slot = SlotOf(psm, basePointer, bufferWidth, x, y);
  WriteBits(slot.bitPosition, slot.bitWidth, value);
}

}  // namespace hw_gs_replace::testing
