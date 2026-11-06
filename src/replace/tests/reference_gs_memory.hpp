// Reference GS local memory for CTest fixtures: a 4 MiB byte array with the chapter 01.3 block
// and pixel layout transcribed from the TASK-026 specification. It stands in for the M2 swizzle
// library in these tests only. Clean room (decision 0069). Copyright (c) 2026 GT4Recomp, MIT.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "hw_gs_replace/gs_types.hpp"

namespace hw_gs_replace::testing {

class ReferenceGsMemory final : public TexelSource {
 public:
  ReferenceGsMemory();

  void ReadBlock(uint32_t blockNumber, std::span<uint8_t, 256> out) const override;
  uint32_t ReadPixel(Psm psm, uint32_t basePointer, uint32_t bufferWidth, uint32_t x,
                     uint32_t y) const override;

  // Test setup: stores `value` in the field of the pixel that ReadPixel would read.
  void WritePixel(Psm psm, uint32_t basePointer, uint32_t bufferWidth, uint32_t x, uint32_t y,
                  uint32_t value);

  // Block numbers requested through ReadBlock, in order (for address tests).
  void ClearBlockLog() { blockLog_.clear(); }
  const std::vector<uint32_t>& BlockLog() const { return blockLog_; }

 private:
  struct FieldSlot {
    uint64_t bitPosition;
    uint32_t bitWidth;
  };
  FieldSlot SlotOf(Psm psm, uint32_t basePointer, uint32_t bufferWidth, uint32_t x,
                   uint32_t y) const;
  uint32_t ReadBits(uint64_t bitPosition, uint32_t width) const;
  void WriteBits(uint64_t bitPosition, uint32_t width, uint32_t value);

  std::vector<uint8_t> memory_;
  mutable std::vector<uint32_t> blockLog_;
};

}  // namespace hw_gs_replace::testing
