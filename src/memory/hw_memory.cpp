// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors
//
// Written from the TASK-026 specification (docs/hw-renderer-spec), clean room.
// Implements chapters 01.3, 02.7, 09.1, 10.2 and the M2 part of chapter 15.

#include "hw_memory.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>

namespace coregs::memory {
namespace {

constexpr std::uint32_t kLocalMemoryMask = kLocalMemoryBytes - 1u;
constexpr std::uint32_t kPageBitCount = kPageBytes * 8u;

// Block tables (chapter 01.3, block arrangement within a page). Entries are
// block numbers within the page, rows top to bottom.

// CT32, CT24, T8H, T4HL, T4HH and T8 use 8 columns and 4 rows of blocks.
constexpr std::uint8_t kBlockTable8x4[4][8] = {
    {0, 1, 4, 5, 16, 17, 20, 21},
    {2, 3, 6, 7, 18, 19, 22, 23},
    {8, 9, 12, 13, 24, 25, 28, 29},
    {10, 11, 14, 15, 26, 27, 30, 31},
};

// CT16 and T4 use 4 columns and 8 rows of blocks.
constexpr std::uint8_t kBlockTable4x8[8][4] = {
    {0, 2, 8, 10},
    {1, 3, 9, 11},
    {4, 6, 12, 14},
    {5, 7, 13, 15},
    {16, 18, 24, 26},
    {17, 19, 25, 27},
    {20, 22, 28, 30},
    {21, 23, 29, 31},
};

// CT16S uses its own 4 x 8 arrangement.
constexpr std::uint8_t kBlockTable4x8Shuffled[8][4] = {
    {0, 2, 16, 18},
    {1, 3, 17, 19},
    {8, 10, 24, 26},
    {9, 11, 25, 27},
    {4, 6, 20, 22},
    {5, 7, 21, 23},
    {12, 14, 28, 30},
    {13, 15, 29, 31},
};

// 32-bit word order inside one 8x2 pixel column pair (chapter 01.3). Row
// index is by & 1, position is bx & 7.
constexpr std::uint8_t kWordOrder32[2][8] = {
    {0, 1, 4, 5, 8, 9, 12, 13},
    {2, 3, 6, 7, 10, 11, 14, 15},
};

// The same order with the two 4-wide halves of each row swapped. Used by the
// 8-bit and 4-bit layouts when (row >> 1) XOR (column & 1) is 1.
constexpr std::uint8_t kWordOrder32Swapped[2][8] = {
    {8, 9, 12, 13, 0, 1, 4, 5},
    {10, 11, 14, 15, 2, 3, 6, 7},
};

// Halfword order inside one 16x2 pixel row pair (chapter 01.3, 16-bit).
constexpr std::uint8_t kHalfwordOrder16[2][16] = {
    {0, 2, 8, 10, 16, 18, 24, 26, 1, 3, 9, 11, 17, 19, 25, 27},
    {4, 6, 12, 14, 20, 22, 28, 30, 5, 7, 13, 15, 21, 23, 29, 31},
};

constexpr std::uint8_t kZBlockXor = 24;

constexpr std::array<PsmInfo, 13> kPsmTable = {{
    //  psm           name     store  xfer   pageW pageH blkW blkH depth indexed validMask   chan family               zXor
    {Psm::Ct32,  "CT32",   32, 32,  64,  32,  8,  8, false, false, 0xFFFFFFFFu, 0xF, SwizzleFamily::Word32,  false},
    {Psm::Ct24,  "CT24",   32, 24,  64,  32,  8,  8, false, false, 0x00FFFFFFu, 0x7, SwizzleFamily::Word32,  false},
    {Psm::Ct16,  "CT16",   16, 16,  64,  64, 16,  8, false, false, 0x80F8F8F8u, 0xF, SwizzleFamily::Word16,  false},
    {Psm::Ct16S, "CT16S",  16, 16,  64,  64, 16,  8, false, false, 0x80F8F8F8u, 0xF, SwizzleFamily::Word16S, false},
    {Psm::T8,    "T8",      8,  8, 128,  64, 16, 16, false, true,  0x00000000u, 0xF, SwizzleFamily::Byte8,   false},
    {Psm::T4,    "T4",      4,  4, 128, 128, 32, 16, false, true,  0x00000000u, 0xF, SwizzleFamily::Nibble4, false},
    {Psm::T8H,   "T8H",     8,  8,  64,  32,  8,  8, false, true,  0xFF000000u, 0x8, SwizzleFamily::Word32,  false},
    {Psm::T4Hl,  "T4HL",    4,  4,  64,  32,  8,  8, false, true,  0x0F000000u, 0x8, SwizzleFamily::Word32,  false},
    {Psm::T4Hh,  "T4HH",    4,  4,  64,  32,  8,  8, false, true,  0xF0000000u, 0x8, SwizzleFamily::Word32,  false},
    {Psm::Z32,   "Z32",    32, 32,  64,  32,  8,  8, true,  false, 0xFFFFFFFFu, 0xF, SwizzleFamily::Word32,  true},
    {Psm::Z24,   "Z24",    32, 24,  64,  32,  8,  8, true,  false, 0x00FFFFFFu, 0x7, SwizzleFamily::Word32,  true},
    {Psm::Z16,   "Z16",    16, 16,  64,  64, 16,  8, true,  false, 0x0000FFFFu, 0xF, SwizzleFamily::Word16,  true},
    {Psm::Z16S,  "Z16S",   16, 16,  64,  64, 16,  8, true,  false, 0x0000FFFFu, 0xF, SwizzleFamily::Word16S, true},
}};

// Bit-field of one in-page location: block number inside the page (before the
// base is added and before any Z XOR), byte inside the block, and the bit
// placement of the pixel inside that byte group.
struct InPageLocation {
    std::uint8_t tableBlock;  // from the block table, 0..31
    std::uint32_t byteInBlock;
    std::uint8_t bitShift;
    std::uint8_t bitWidth;
};

// Locates (x, y) inside one page. x and y must already be reduced modulo the
// page dimensions.
InPageLocation locateInPage(const PsmInfo& info, std::uint32_t x, std::uint32_t y) {
    switch (info.family) {
    case SwizzleFamily::Word32: {
        // 8x8 blocks, 4 columns of 8x2 pixels, 32-bit words.
        const std::uint32_t blockX = x / 8;
        const std::uint32_t blockY = y / 8;
        const std::uint32_t inX = x % 8;
        const std::uint32_t inY = y % 8;
        const std::uint32_t column = inY / 2;
        const std::uint32_t word = kWordOrder32[inY & 1][inX];
        std::uint8_t shift = 0;
        std::uint8_t width = 32;
        switch (info.psm) {
        case Psm::Ct24:
        case Psm::Z24:
            width = 24;
            break;
        case Psm::T8H:
            shift = 24;
            width = 8;
            break;
        case Psm::T4Hl:
            shift = 24;
            width = 4;
            break;
        case Psm::T4Hh:
            shift = 28;
            width = 4;
            break;
        default:
            break;
        }
        return {kBlockTable8x4[blockY][blockX], column * 64u + word * 4u, shift, width};
    }
    case SwizzleFamily::Word16:
    case SwizzleFamily::Word16S: {
        // 16x8 blocks, 4 columns of 16x2 pixels, 16-bit halfwords.
        const std::uint32_t blockX = x / 16;
        const std::uint32_t blockY = y / 8;
        const std::uint32_t inX = x % 16;
        const std::uint32_t inY = y % 8;
        const std::uint32_t column = inY / 2;
        const std::uint32_t halfword = kHalfwordOrder16[inY & 1][inX];
        const std::uint8_t table = info.family == SwizzleFamily::Word16S
                                       ? kBlockTable4x8Shuffled[blockY][blockX]
                                       : kBlockTable4x8[blockY][blockX];
        return {table, column * 64u + halfword * 2u, 0, 16};
    }
    case SwizzleFamily::Byte8: {
        // 16x16 blocks, 4 columns of 4 rows, 8-bit texels in 32-bit words.
        const std::uint32_t blockX = x / 16;
        const std::uint32_t blockY = y / 16;
        const std::uint32_t inX = x % 16;
        const std::uint32_t inY = y % 16;
        const std::uint32_t column = inY / 4;
        const std::uint32_t row = inY % 4;
        const std::uint32_t swap = ((row >> 1) ^ (column & 1u)) & 1u;
        const std::uint32_t word = swap ? kWordOrder32Swapped[row & 1][inX & 7]
                                        : kWordOrder32[row & 1][inX & 7];
        const std::uint32_t byte = 2u * ((inX >> 3) & 1u) + ((row >> 1) & 1u);
        return {kBlockTable8x4[blockY][blockX], column * 64u + word * 4u + byte, 0, 8};
    }
    case SwizzleFamily::Nibble4: {
        // 32x16 blocks, 4 columns of 4 rows, 4-bit texels as nibbles.
        const std::uint32_t blockX = x / 32;
        const std::uint32_t blockY = y / 16;
        const std::uint32_t inX = x % 32;
        const std::uint32_t inY = y % 16;
        const std::uint32_t column = inY / 4;
        const std::uint32_t row = inY % 4;
        const std::uint32_t swap = ((row >> 1) ^ (column & 1u)) & 1u;
        const std::uint32_t word = swap ? kWordOrder32Swapped[row & 1][inX & 7]
                                        : kWordOrder32[row & 1][inX & 7];
        // Nibble address inside the block is 2 * (64 * column + 4 * word) + nibble.
        const std::uint32_t nibble = 2u * ((inX >> 3) & 3u) + ((row >> 1) & 1u);
        const std::uint32_t byteInBlock = column * 64u + word * 4u + nibble / 2u;
        return {kBlockTable4x8[blockY][blockX], byteInBlock,
                static_cast<std::uint8_t>((nibble & 1u) * 4u), 4};
    }
    }
    return {0, 0, 0, 0};  // unreachable: every family is handled above
}

std::uint64_t bitMask(std::uint32_t width) {
    return (std::uint64_t{1} << width) - 1u;
}

std::uint32_t readStorageBytes(const std::vector<std::uint8_t>& bytes, std::uint32_t address, std::uint32_t count) {
    std::uint64_t raw = 0;
    for (std::uint32_t index = 0; index < count; ++index) {
        raw |= static_cast<std::uint64_t>(bytes[(address + index) & kLocalMemoryMask]) << (8u * index);
    }
    return static_cast<std::uint32_t>(raw);
}

// Depth class of one format for the compatibility predicate.
constexpr int depthClass(Psm psm) {
    switch (psm) {
    case Psm::Ct32:
    case Psm::Ct24:
    case Psm::Z32:
    case Psm::Z24:
        return 32;
    case Psm::Ct16:
    case Psm::Ct16S:
    case Psm::Z16:
    case Psm::Z16S:
        return 16;
    default:
        return 0;  // 8 and 4 bit formats: class is the format itself
    }
}

// Same-swizzle group (chapter 01.3, "Same swizzle").
constexpr int swizzleGroup(Psm psm) {
    switch (psm) {
    case Psm::Ct32:
    case Psm::Ct24:
    case Psm::T8H:
    case Psm::T4Hl:
    case Psm::T4Hh:
        return 0;
    case Psm::Z32:
    case Psm::Z24:
        return 1;
    case Psm::Ct16:
        return 2;
    case Psm::Ct16S:
        return 3;
    case Psm::Z16:
        return 4;
    case Psm::Z16S:
        return 5;
    case Psm::T8:
        return 6;
    case Psm::T4:
        return 7;
    }
    return -1;
}

constexpr bool sharesBitsRule(Psm first, Psm second) {
    auto isLowPrecisionHigh = [](Psm psm) { return psm == Psm::T8H || psm == Psm::T4Hl || psm == Psm::T4Hh; };
    auto isCt24Like = [](Psm psm) { return psm == Psm::Ct24 || psm == Psm::Z24; };
    if (isCt24Like(first) && isLowPrecisionHigh(second)) return false;
    if (isCt24Like(second) && isLowPrecisionHigh(first)) return false;
    if ((first == Psm::T4Hl && second == Psm::T4Hh) || (first == Psm::T4Hh && second == Psm::T4Hl)) return false;
    return true;
}

constexpr bool isValidCode(std::uint32_t code) {
    switch (code) {
    case 0x00: case 0x01: case 0x02: case 0x0A: case 0x13: case 0x14: case 0x1B:
    case 0x24: case 0x2C: case 0x30: case 0x31: case 0x32: case 0x3A:
        return true;
    default:
        return false;
    }
}

template <typename Predicate>
constexpr FormatPredicateTable buildTable(Predicate predicate) {
    FormatPredicateTable table{};
    for (std::uint32_t first = 0; first < 64; ++first) {
        for (std::uint32_t second = 0; second < 64; ++second) {
            if (!isValidCode(first) || !isValidCode(second)) {
                table[first][second] = false;
                continue;
            }
            table[first][second] = predicate(static_cast<Psm>(first), static_cast<Psm>(second));
        }
    }
    return table;
}

constexpr FormatPredicateTable kSharedBitsTable = buildTable(sharesBitsRule);
constexpr FormatPredicateTable kCompatibleDepthTable = buildTable(
    [](Psm first, Psm second) {
        if (first == second) return true;
        const int classFirst = depthClass(first);
        return classFirst != 0 && classFirst == depthClass(second);
    });
constexpr FormatPredicateTable kSameSwizzleTable = buildTable(
    [](Psm first, Psm second) { return swizzleGroup(first) == swizzleGroup(second); });

}  // namespace

// ---------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------

std::optional<Psm> decodePsm(std::uint32_t code) {
    if (code > kPsmCodeMask || !isValidCode(code)) return std::nullopt;
    return static_cast<Psm>(code);
}

bool isValidPsmCode(std::uint32_t code) {
    return decodePsm(code).has_value();
}

// Index of each PSM code in kPsmTable (codes are 6-bit), built once. psmInfo() runs for every pixel access,
// so it is a table lookup rather than a search.
constexpr std::array<std::uint8_t, 64> kPsmTableIndex = [] {
    std::array<std::uint8_t, 64> table{};
    for (std::size_t i = 0; i < kPsmTable.size(); ++i) {
        table[static_cast<std::uint8_t>(kPsmTable[i].psm) & kPsmCodeMask] = static_cast<std::uint8_t>(i);
    }
    return table;
}();

const PsmInfo& psmInfo(Psm psm) {
    const PsmInfo& info = kPsmTable[kPsmTableIndex[static_cast<std::uint8_t>(psm) & kPsmCodeMask]];
    return info.psm == psm ? info : kPsmTable.front();  // unreachable for valid Psm values
}

std::uint8_t PixelSlot::bytesCovered() const {
    return static_cast<std::uint8_t>((bitShift + bitWidth + 7u) / 8u);
}

std::uint32_t pagesPerRow(std::uint32_t bufferWidth, Psm psm) {
    const std::uint32_t pageWidth = psmInfo(psm).pageWidth;
    return std::max<std::uint32_t>(1u, bufferWidth * 64u / pageWidth);
}

// The in-page location of every pixel of a page, per PSM, computed once at start-up from locateInPage (the
// same result, looked up instead of recomputed for every pixel access).
struct InPageTables {
    std::array<std::vector<InPageLocation>, kPsmTable.size()> perPsm;
    InPageTables() {
        for (std::size_t index = 0; index < kPsmTable.size(); ++index) {
            const PsmInfo& info = kPsmTable[index];
            std::vector<InPageLocation>& table = perPsm[index];
            table.resize(static_cast<std::size_t>(info.pageWidth) * info.pageHeight);
            for (std::uint32_t py = 0; py < info.pageHeight; ++py) {
                for (std::uint32_t px = 0; px < info.pageWidth; ++px) {
                    table[static_cast<std::size_t>(py) * info.pageWidth + px] = locateInPage(info, px, py);
                }
            }
        }
    }
};
const InPageTables kInPageTables;

inline const InPageLocation& inPageLocation(const PsmInfo& info, std::uint32_t x, std::uint32_t y) {
    const std::size_t index = static_cast<std::size_t>(&info - kPsmTable.data());
    return kInPageTables.perPsm[index][static_cast<std::size_t>(y) * info.pageWidth + x];
}

GpuAddressTables gpuAddressTables() {
    GpuAddressTables tables;
    tables.directory.assign(64u * 4u, 0u);
    for (std::size_t index = 0; index < kPsmTable.size(); ++index) {
        const PsmInfo& info = kPsmTable[index];
        const std::uint32_t code = static_cast<std::uint32_t>(info.psm) & kPsmCodeMask;
        tables.directory[code * 4u + 0u] = static_cast<std::uint32_t>(tables.entries.size());
        tables.directory[code * 4u + 1u] = info.pageWidth;
        tables.directory[code * 4u + 2u] = info.pageHeight;
        tables.directory[code * 4u + 3u] = info.zBlockXorApplies ? 1u : 0u;
        for (const InPageLocation& location : kInPageTables.perPsm[index]) {
            tables.entries.push_back(static_cast<std::uint32_t>(location.tableBlock) |
                                     (location.byteInBlock << 5) |
                                     (static_cast<std::uint32_t>(location.bitShift) << 13) |
                                     (static_cast<std::uint32_t>(location.bitWidth) << 18));
        }
    }
    return tables;
}

PixelSlot pixelSlot(const BufferView& buffer, std::uint32_t x, std::uint32_t y, ZBlockXorRule zRule) {
    const PsmInfo& info = psmInfo(buffer.psm);
    const std::uint32_t perRow = std::max<std::uint32_t>(1u, buffer.bufferWidth * 64u / info.pageWidth);
    const std::uint32_t page = (y / info.pageHeight) * perRow + (x / info.pageWidth);
    const InPageLocation& inPage = inPageLocation(info, x % info.pageWidth, y % info.pageHeight);

    const std::uint32_t pageBlock = buffer.basePointer + kBlocksPerPage * page;
    std::uint32_t block = 0;
    if (info.zBlockXorApplies && zRule == ZBlockXorRule::BeforeAdd) {
        block = pageBlock + (inPage.tableBlock ^ kZBlockXor);
    } else {
        block = pageBlock + inPage.tableBlock;
        if (info.zBlockXorApplies) block ^= kZBlockXor;
    }
    block %= kBlocksTotal;

    PixelSlot slot{};
    slot.byteOffset = (block * kBlockBytes + inPage.byteInBlock) & kLocalMemoryMask;
    slot.bitShift = inPage.bitShift;
    slot.bitWidth = inPage.bitWidth;
    return slot;
}

PixelSlot pageLocalSlot(Psm psm, std::uint32_t x, std::uint32_t y) {
    // Base 0 and bufferWidth 1 place the page at offset 0. The page index is
    // then 0 for every x inside one page width.
    return pixelSlot(BufferView{0, 1, psm}, x, y, ZBlockXorRule::AfterAdd);
}

// ---------------------------------------------------------------------------
// Local memory
// ---------------------------------------------------------------------------

LocalMemory::LocalMemory() : bytes_(kLocalMemoryBytes, 0) {}

std::uint8_t LocalMemory::byte(std::uint32_t address) const {
    return bytes_[address & kLocalMemoryMask];
}

void LocalMemory::setByte(std::uint32_t address, std::uint8_t value) {
    bytes_[address & kLocalMemoryMask] = value;
}

std::uint32_t LocalMemory::readSlot(const PixelSlot& slot) const {
    if (slot.byteOffset + 4u <= kLocalMemoryBytes) {
        // Every storage unit (4 to 32 bits) lies in the 4 bytes from byteOffset: one little-endian load.
        std::uint32_t word = 0;
        std::memcpy(&word, bytes_.data() + slot.byteOffset, sizeof(word));
        return static_cast<std::uint32_t>((static_cast<std::uint64_t>(word) >> slot.bitShift) & bitMask(slot.bitWidth));
    }
    const std::uint64_t raw = readStorageBytes(bytes_, slot.byteOffset, slot.bytesCovered());
    return static_cast<std::uint32_t>((raw >> slot.bitShift) & bitMask(slot.bitWidth));
}

void LocalMemory::writeSlot(const PixelSlot& slot, std::uint32_t value) {
    if (slot.byteOffset + 4u <= kLocalMemoryBytes) {
        // Read-modify-write of the 4 bytes from byteOffset; bytes outside the field are stored unchanged.
        std::uint32_t word = 0;
        std::memcpy(&word, bytes_.data() + slot.byteOffset, sizeof(word));
        const std::uint64_t fieldMask = bitMask(slot.bitWidth) << slot.bitShift;
        const std::uint64_t merged = (static_cast<std::uint64_t>(word) & ~fieldMask) |
                                     ((static_cast<std::uint64_t>(value) & bitMask(slot.bitWidth)) << slot.bitShift);
        word = static_cast<std::uint32_t>(merged);
        std::memcpy(bytes_.data() + slot.byteOffset, &word, sizeof(word));
        return;
    }
    const std::uint8_t count = slot.bytesCovered();
    std::uint64_t raw = readStorageBytes(bytes_, slot.byteOffset, count);
    const std::uint64_t fieldMask = bitMask(slot.bitWidth) << slot.bitShift;
    raw = (raw & ~fieldMask) | ((static_cast<std::uint64_t>(value) & bitMask(slot.bitWidth)) << slot.bitShift);
    for (std::uint32_t index = 0; index < count; ++index) {
        bytes_[(slot.byteOffset + index) & kLocalMemoryMask] = static_cast<std::uint8_t>(raw >> (8u * index));
    }
}

std::uint32_t LocalMemory::readPixel(const BufferView& buffer, std::uint32_t x, std::uint32_t y) const {
    return readSlot(pixelSlot(buffer, x, y));
}

namespace {

// Slot of (x, y) from a precomputed row: the same arithmetic as pixelSlot with the AfterAdd Z rule.
struct RowWalker {
    const PsmInfo& info;
    const BufferView& buffer;
    std::uint32_t perRow;
    std::uint32_t widthShift;
    std::uint32_t heightShift;

    RowWalker(const PsmInfo& psmInfoRef, const BufferView& view)
        : info(psmInfoRef), buffer(view),
          perRow(std::max<std::uint32_t>(1u, view.bufferWidth * 64u / psmInfoRef.pageWidth)),
          widthShift(static_cast<std::uint32_t>(std::countr_zero(static_cast<std::uint32_t>(psmInfoRef.pageWidth)))),
          heightShift(static_cast<std::uint32_t>(std::countr_zero(static_cast<std::uint32_t>(psmInfoRef.pageHeight)))) {}

    PixelSlot slot(std::uint32_t x, std::uint32_t y, const InPageLocation* rowTable) const {
        const std::uint32_t page = (y >> heightShift) * perRow + (x >> widthShift);
        const InPageLocation& inPage = rowTable[x & (info.pageWidth - 1u)];
        std::uint32_t block = buffer.basePointer + kBlocksPerPage * page + inPage.tableBlock;
        if (info.zBlockXorApplies) block ^= kZBlockXor;
        block %= kBlocksTotal;
        PixelSlot result{};
        result.byteOffset = (block * kBlockBytes + inPage.byteInBlock) & kLocalMemoryMask;
        result.bitShift = inPage.bitShift;
        result.bitWidth = inPage.bitWidth;
        return result;
    }
};

}  // namespace

void LocalMemory::readRect(const BufferView& buffer, std::uint32_t x0, std::uint32_t y0, std::uint32_t width,
                           std::uint32_t height, std::uint32_t* values) const {
    const PsmInfo& info = psmInfo(buffer.psm);
    const RowWalker walker(info, buffer);
    const std::size_t tableIndex = static_cast<std::size_t>(&info - kPsmTable.data());
    for (std::uint32_t row = 0; row < height; ++row) {
        const std::uint32_t y = y0 + row;
        const InPageLocation* rowTable =
            kInPageTables.perPsm[tableIndex].data() + static_cast<std::size_t>(y & (info.pageHeight - 1u)) * info.pageWidth;
        for (std::uint32_t column = 0; column < width; ++column) {
            *values++ = readSlot(walker.slot(x0 + column, y, rowTable));
        }
    }
}

void LocalMemory::writeRect(const BufferView& buffer, std::uint32_t x0, std::uint32_t y0, std::uint32_t width,
                            std::uint32_t height, const std::uint32_t* values) {
    const PsmInfo& info = psmInfo(buffer.psm);
    const RowWalker walker(info, buffer);
    const std::size_t tableIndex = static_cast<std::size_t>(&info - kPsmTable.data());
    for (std::uint32_t row = 0; row < height; ++row) {
        const std::uint32_t y = y0 + row;
        const InPageLocation* rowTable =
            kInPageTables.perPsm[tableIndex].data() + static_cast<std::size_t>(y & (info.pageHeight - 1u)) * info.pageWidth;
        for (std::uint32_t column = 0; column < width; ++column) {
            writeSlot(walker.slot(x0 + column, y, rowTable), *values++);
        }
    }
}

void LocalMemory::writePixel(const BufferView& buffer, std::uint32_t x, std::uint32_t y, std::uint32_t value) {
    writeSlot(pixelSlot(buffer, x, y), value);
}

// ---------------------------------------------------------------------------
// Transfers
// ---------------------------------------------------------------------------

std::size_t hostBytesFor(Psm psm, std::uint32_t width, std::uint32_t height) {
    const std::uint64_t bits = static_cast<std::uint64_t>(width) * height * psmInfo(psm).transferBpp;
    return static_cast<std::size_t>((bits + 7u) / 8u);
}

namespace {

// Decodes host texel `index` of `bpp` bits from packed host data.
// Returns false when the data does not hold the texel.
bool decodeHostTexel(std::span<const std::uint8_t> data, std::uint32_t bpp, std::uint64_t index, std::uint32_t& value) {
    const std::uint64_t firstBit = index * bpp;
    if (firstBit + bpp > static_cast<std::uint64_t>(data.size()) * 8u) return false;
    switch (bpp) {
    case 32:
        value = data[index * 4 + 0] | (data[index * 4 + 1] << 8) | (data[index * 4 + 2] << 16) |
                (static_cast<std::uint32_t>(data[index * 4 + 3]) << 24);
        return true;
    case 24:
        value = data[index * 3 + 0] | (data[index * 3 + 1] << 8) | (static_cast<std::uint32_t>(data[index * 3 + 2]) << 16);
        return true;
    case 16:
        value = data[index * 2 + 0] | (static_cast<std::uint32_t>(data[index * 2 + 1]) << 8);
        return true;
    case 8:
        value = data[index];
        return true;
    default:  // 4 bit: continuous nibble stream, low nibble first
        value = (index & 1u) ? (data[index / 2] >> 4) : (data[index / 2] & 0x0Fu);
        return true;
    }
}

void encodeHostTexel(std::vector<std::uint8_t>& data, std::uint32_t bpp, std::uint64_t index, std::uint32_t value) {
    switch (bpp) {
    case 32:
        for (std::uint32_t byte = 0; byte < 4; ++byte) data[index * 4 + byte] = static_cast<std::uint8_t>(value >> (8u * byte));
        break;
    case 24:
        for (std::uint32_t byte = 0; byte < 3; ++byte) data[index * 3 + byte] = static_cast<std::uint8_t>(value >> (8u * byte));
        break;
    case 16:
        data[index * 2 + 0] = static_cast<std::uint8_t>(value);
        data[index * 2 + 1] = static_cast<std::uint8_t>(value >> 8);
        break;
    case 8:
        data[index] = static_cast<std::uint8_t>(value);
        break;
    default:  // 4 bit
        if (index & 1u) {
            data[index / 2] = static_cast<std::uint8_t>((data[index / 2] & 0x0Fu) | ((value & 0x0Fu) << 4));
        } else {
            data[index / 2] = static_cast<std::uint8_t>((data[index / 2] & 0xF0u) | (value & 0x0Fu));
        }
        break;
    }
}

}  // namespace

std::uint32_t hostToLocal(LocalMemory& memory,
                          const BufferView& destination,
                          const TransferRect& rect,
                          std::span<const std::uint8_t> data) {
    const std::uint32_t bpp = psmInfo(destination.psm).transferBpp;
    std::uint32_t pixelsWritten = 0;
    for (std::uint32_t row = 0; row < rect.height; ++row) {
        for (std::uint32_t column = 0; column < rect.width; ++column) {
            const std::uint64_t index = static_cast<std::uint64_t>(row) * rect.width + column;
            std::uint32_t value = 0;
            if (!decodeHostTexel(data, bpp, index, value)) return pixelsWritten;
            const std::uint32_t x = (rect.x + column) % kTransferCoordinateWrap;
            const std::uint32_t y = (rect.y + row) % kTransferCoordinateWrap;
            memory.writePixel(destination, x, y, value);
            ++pixelsWritten;
        }
    }
    return pixelsWritten;
}

void localToLocal(LocalMemory& memory,
                  const BufferView& source,
                  std::uint32_t sourceX,
                  std::uint32_t sourceY,
                  const BufferView& destination,
                  std::uint32_t destinationX,
                  std::uint32_t destinationY,
                  std::uint32_t width,
                  std::uint32_t height,
                  TransferDirection direction) {
    const bool upperFirst = direction == TransferDirection::UpperLeftToLowerRight ||
                            direction == TransferDirection::UpperRightToLowerLeft;
    const bool leftFirst = direction == TransferDirection::UpperLeftToLowerRight ||
                           direction == TransferDirection::LowerLeftToUpperRight;

    for (std::uint32_t rowStep = 0; rowStep < height; ++rowStep) {
        const std::uint32_t row = upperFirst ? rowStep : height - 1u - rowStep;
        for (std::uint32_t columnStep = 0; columnStep < width; ++columnStep) {
            const std::uint32_t column = leftFirst ? columnStep : width - 1u - columnStep;
            const std::uint32_t sx = (sourceX + column) % kTransferCoordinateWrap;
            const std::uint32_t sy = (sourceY + row) % kTransferCoordinateWrap;
            const std::uint32_t dx = (destinationX + column) % kTransferCoordinateWrap;
            const std::uint32_t dy = (destinationY + row) % kTransferCoordinateWrap;
            const std::uint32_t value = memory.readPixel(source, sx, sy);
            memory.writePixel(destination, dx, dy, value);
        }
    }
}

TransferStatus localToHost(const LocalMemory& memory,
                           const BufferView& source,
                           const TransferRect& rect,
                           std::vector<std::uint8_t>& hostData) {
    const std::uint32_t bpp = psmInfo(source.psm).transferBpp;
    if (bpp == 4) return TransferStatus::IllegalFormat;

    hostData.assign(hostBytesFor(source.psm, rect.width, rect.height), 0);
    for (std::uint32_t row = 0; row < rect.height; ++row) {
        for (std::uint32_t column = 0; column < rect.width; ++column) {
            const std::uint64_t index = static_cast<std::uint64_t>(row) * rect.width + column;
            const std::uint32_t x = (rect.x + column) % kTransferCoordinateWrap;
            const std::uint32_t y = (rect.y + row) % kTransferCoordinateWrap;
            encodeHostTexel(hostData, bpp, index, memory.readPixel(source, x, y));
        }
    }
    return TransferStatus::Ok;
}

std::optional<std::uint64_t> readbackQwords(std::uint32_t psmCode, std::uint32_t width, std::uint32_t height) {
    const std::uint32_t selector = psmCode & 7u;
    if (selector == 4u) return std::nullopt;  // 4-bit formats are illegal for LOCAL->HOST
    std::uint64_t bpp = 32;
    switch (selector) {
    case 1: bpp = 24; break;
    case 2: bpp = 16; break;
    case 3: bpp = 8; break;
    default: bpp = 32; break;  // 0, and 5..7 count as 32 (chapter 10.2)
    }
    const std::uint64_t bits = static_cast<std::uint64_t>(width) * height * bpp;
    return (bits + 127u) / 128u;
}

// ---------------------------------------------------------------------------
// Reinterpretation maps
// ---------------------------------------------------------------------------

ReinterpMap buildReinterpMap(Psm view, Psm store) {
    const PsmInfo& viewInfo = psmInfo(view);
    const PsmInfo& storeInfo = psmInfo(store);

    // Every store pixel of one page, indexed by the page bit it starts at.
    struct StorePixel {
        std::uint32_t bitStart;
        std::uint8_t bitWidth;
        std::uint16_t x;
        std::uint16_t y;
    };
    std::vector<StorePixel> storePixels;
    std::vector<std::int32_t> bitOwner(kPageBitCount, -1);
    for (std::uint32_t y = 0; y < storeInfo.pageHeight; ++y) {
        for (std::uint32_t x = 0; x < storeInfo.pageWidth; ++x) {
            const PixelSlot slot = pageLocalSlot(store, x, y);
            const std::uint32_t owner = static_cast<std::uint32_t>(storePixels.size());
            storePixels.push_back({slot.bitAddress(), slot.bitWidth, static_cast<std::uint16_t>(x), static_cast<std::uint16_t>(y)});
            for (std::uint32_t bit = 0; bit < slot.bitWidth; ++bit) {
                bitOwner[slot.bitAddress() + bit] = static_cast<std::int32_t>(owner);
            }
        }
    }

    ReinterpMap map(static_cast<std::size_t>(viewInfo.pageWidth) * viewInfo.pageHeight);
    for (std::uint32_t y = 0; y < viewInfo.pageHeight; ++y) {
        for (std::uint32_t x = 0; x < viewInfo.pageWidth; ++x) {
            const PixelSlot slot = pageLocalSlot(view, x, y);
            ReinterpTexel& entry = map[static_cast<std::size_t>(y) * viewInfo.pageWidth + x];
            entry = {0, 0, 0, slot.bitWidth, false};

            const std::int32_t first = bitOwner[slot.bitAddress()];
            if (first < 0) continue;
            bool sameOwner = true;
            for (std::uint32_t bit = 0; bit < slot.bitWidth; ++bit) {
                const std::uint32_t address = slot.bitAddress() + bit;
                if (address >= kPageBitCount || bitOwner[address] != first) {
                    sameOwner = false;
                    break;
                }
            }
            if (!sameOwner) continue;

            const StorePixel& pixel = storePixels[static_cast<std::size_t>(first)];
            entry.storeX = pixel.x;
            entry.storeY = pixel.y;
            entry.bitOffset = static_cast<std::uint8_t>(slot.bitAddress() - pixel.bitStart);
            entry.mapped = true;
        }
    }
    return map;
}

// ---------------------------------------------------------------------------
// Format predicates
// ---------------------------------------------------------------------------

bool formatsShareBits(Psm first, Psm second) {
    return kSharedBitsTable[static_cast<std::uint32_t>(first) & kPsmCodeMask][static_cast<std::uint32_t>(second) & kPsmCodeMask];
}

bool formatsCompatibleDepth(Psm first, Psm second) {
    return kCompatibleDepthTable[static_cast<std::uint32_t>(first) & kPsmCodeMask][static_cast<std::uint32_t>(second) & kPsmCodeMask];
}

bool formatsSameSwizzle(Psm first, Psm second) {
    return kSameSwizzleTable[static_cast<std::uint32_t>(first) & kPsmCodeMask][static_cast<std::uint32_t>(second) & kPsmCodeMask];
}

const FormatPredicateTable& sharedBitsTable() {
    return kSharedBitsTable;
}

const FormatPredicateTable& compatibleDepthTable() {
    return kCompatibleDepthTable;
}

const FormatPredicateTable& sameSwizzleTable() {
    return kSameSwizzleTable;
}

std::uint32_t validBitMask(Psm psm) {
    return psmInfo(psm).validBitMask32;
}

}  // namespace coregs::memory
