// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec/CORE2-SPEC.md and CORE2-DESIGN.md (TASK-027 core rewrite), clean room
// (decision 0070).

#include "core2_upload.hpp"

#include <emmintrin.h>

#include <algorithm>
#include <bit>
#include <array>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace coregs::core2 {
namespace {

using memory::BufferView;

// IN-414: per PSM code, the offset inside its page of the byte holding each pixel's lowest bit (bits 0-15) and the bit
// shift of a 4-bit pixel in that byte (bits 16-18), with the Z block swap folded in. It holds for a buffer whose base pointer is page
// aligned: the page part of the block number and the in-page part then never carry into each other.
struct InPageOffsets {
    std::array<std::vector<std::uint32_t>, 64> perPsm;
};

const InPageOffsets &InPageOffsetTables() {
    static const InPageOffsets offsets = [] {
        InPageOffsets built;
        const memory::GpuAddressTables tables = memory::gpuAddressTables();
        for (std::uint32_t code = 0; code < 64u; ++code) {
            const std::uint32_t tableOffset = tables.directory[code * 4u];
            const std::uint32_t pageWidth = tables.directory[code * 4u + 1u];
            const std::uint32_t pageHeight = tables.directory[code * 4u + 2u];
            const bool zXor = tables.directory[code * 4u + 3u] != 0;
            if (pageWidth == 0 || pageHeight == 0) continue;
            std::vector<std::uint32_t> &out = built.perPsm[code];
            out.resize(static_cast<std::size_t>(pageWidth) * pageHeight);
            for (std::uint32_t i = 0; i < out.size(); ++i) {
                const std::uint32_t entry = tables.entries[tableOffset + i];
                std::uint32_t block = entry & 31u;
                if (zXor) block ^= 24u;
                // The byte holding the pixel's lowest bit: T8H, T4HL and T4HH sit in the top byte of a 32-bit word
                // (shift 24 or 28 from the word's first byte).
                const std::uint32_t shift = (entry >> 13) & 31u;
                const std::uint32_t byteInPage = block * memory::kBlockBytes + ((entry >> 5) & 255u) + shift / 8u;
                out[i] = byteInPage | ((shift % 8u) << 16);
            }
        }
        return built;
    }();
    return offsets;
}

// IN-414: `count` consecutive host pixels (from pixel `index` of `data`) stored at their in-page offsets in one page.
// IN-450: returns the OR of (old ^ new) over the stored bits, so 0 means every pixel already held its value.
// IN-453: `blocks` gets bit b set when a pixel of block b of the page changed.
template <std::uint32_t Bpp>
std::uint32_t StoreRun(std::uint8_t *page, const std::uint32_t *offsets, const std::uint8_t *data, std::uint64_t index,
                       std::uint32_t count, std::uint32_t &blocks) {
    std::uint32_t diff = 0;
    for (std::uint32_t i = 0; i < count; ++i, ++index) {
        std::uint8_t *out = page + (offsets[i] & 0xFFFFu);
        std::uint32_t pixel = 0;  // this pixel's old ^ new
        if constexpr (Bpp == 32) {
            std::uint32_t value = 0, old = 0;
            std::memcpy(&value, data + index * 4u, 4);
            std::memcpy(&old, out, 4);
            pixel = value ^ old;
            std::memcpy(out, &value, 4);
        } else if constexpr (Bpp == 24) {
            std::uint32_t value = 0, old = 0;
            std::memcpy(&value, data + index * 3u, 3);
            std::memcpy(&old, out, 3);
            pixel = value ^ old;
            std::memcpy(out, &value, 3);
        } else if constexpr (Bpp == 16) {
            std::uint16_t value = 0, old = 0;
            std::memcpy(&value, data + index * 2u, 2);
            std::memcpy(&old, out, 2);
            pixel = static_cast<std::uint32_t>(value ^ old);
            std::memcpy(out, &value, 2);
        } else if constexpr (Bpp == 8) {
            pixel = static_cast<std::uint32_t>(*out ^ data[index]);
            *out = data[index];
        } else {  // 4 bits, low nibble first in the host stream
            const std::uint32_t shift = offsets[i] >> 16;
            const std::uint8_t nibble =
                (index & 1u) ? static_cast<std::uint8_t>(data[index / 2] >> 4) : (data[index / 2] & 0x0Fu);
            pixel = static_cast<std::uint32_t>(((*out >> shift) & 0x0Fu) ^ nibble);
            *out = static_cast<std::uint8_t>((*out & ~(0x0Fu << shift)) | (nibble << shift));
        }
        diff |= pixel;
        if (pixel != 0) blocks |= 1u << ((offsets[i] & 0xFFFFu) >> 8);
    }
    return diff;
}

// IN-453: marks block `pageBlock + b` (mod the memory) for every bit b of `blocks`.
void MarkBlocks(std::uint64_t *changedBlocks, std::uint32_t pageBlock, std::uint32_t blocks) {
    while (blocks != 0) {
        const std::uint32_t b = static_cast<std::uint32_t>(__builtin_ctz(blocks));
        blocks &= blocks - 1u;
        const std::uint32_t block = (pageBlock + b) % memory::kBlocksTotal;
        changedBlocks[block >> 6] |= std::uint64_t{1} << (block & 63u);
    }
}

// One host row (transfer row `row`), columns [begin, end): one run per page crossed (and at the transfer coordinate
// wrap), so the divisions happen once per run, not per pixel.
template <std::uint32_t Bpp>
std::uint32_t StoreRowSpan(std::uint8_t *memoryBytes, const BufferView &view, std::uint32_t x0, std::uint32_t y0,
                           std::uint32_t width, const std::uint8_t *data, const std::uint32_t *pageOffsets,
                           std::uint32_t pageWidth, std::uint32_t pageHeight, std::uint32_t row, std::uint32_t begin,
                           std::uint32_t end, std::uint64_t *changedBlocks) {
    std::uint32_t diff = 0;
    const std::uint32_t perRow = std::max<std::uint32_t>(1u, view.bufferWidth * 64u / pageWidth);
    const std::uint32_t y = (y0 + row) % memory::kTransferCoordinateWrap;
    const std::uint32_t pageRow = (y / pageHeight) * perRow;
    const std::uint32_t *rowOffsets = pageOffsets + (y % pageHeight) * pageWidth;
    std::uint64_t index = static_cast<std::uint64_t>(row) * width + begin;
    std::uint32_t column = begin;
    while (column < end) {
        const std::uint32_t x = (x0 + column) % memory::kTransferCoordinateWrap;
        const std::uint32_t xInPage = x % pageWidth;
        const std::uint32_t run = std::min({end - column, pageWidth - xInPage, memory::kTransferCoordinateWrap - x});
        const std::uint32_t pageBlock = view.basePointer + memory::kBlocksPerPage * (pageRow + x / pageWidth);
        std::uint8_t *page = memoryBytes + ((pageBlock * memory::kBlockBytes) & (memory::kLocalMemoryBytes - 1u));
        std::uint32_t blocks = 0;
        diff |= StoreRun<Bpp>(page, rowOffsets + xInPage, data, index, run, blocks);
        if (blocks != 0 && changedBlocks) MarkBlocks(changedBlocks, pageBlock, blocks);
        column += run;
        index += run;
    }
    return diff;
}

// IN-452: PSMCT32 columns are 8 x 2 pixels in 64 bytes, word w holding pixel (2 * (w / 4) + w % 2, (w / 2) % 2):
// row 0's pixels 0,1 then row 1's 0,1, then row 0's 2,3 and so on. Two host rows of 8 pixels (four 16-byte vectors
// A, B of row 0 and C, D of row 1) are therefore the column's four 16-byte quarters unpacklo/hi_epi64(A, C) and
// unpacklo/hi_epi64(B, D). The table holds each column's offset in its page, and is checked against the in-page
// offsets at start-up (`valid` false: the scalar path only).
struct Ct32Columns {
    bool valid = false;
    std::uint32_t base[16][8] = {};  // [y % 32 / 2][x % 64 / 8]: the byte offset of pixel (x, y) in its page
};

const Ct32Columns &Ct32ColumnTable(const std::vector<std::uint32_t> &offsets) {
    static const Ct32Columns table = [&offsets] {
        Ct32Columns c;
        if (offsets.size() != 64u * 32u) return c;
        c.valid = true;
        for (std::uint32_t y2 = 0; y2 < 16; ++y2) {
            for (std::uint32_t x8 = 0; x8 < 8; ++x8) {
                const std::uint32_t base = offsets[(2 * y2) * 64 + 8 * x8] & 0xFFFFu;
                c.base[y2][x8] = base;
                for (std::uint32_t r = 0; r < 2; ++r) {
                    for (std::uint32_t i = 0; i < 8; ++i) {
                        const std::uint32_t expected = base + 16 * (i / 2) + 8 * r + 4 * (i % 2);
                        if ((offsets[(2 * y2 + r) * 64 + 8 * x8 + i] & 0xFFFFu) != expected) c.valid = false;
                    }
                }
            }
        }
        return c;
    }();
    return table;
}

// Rows in pairs (an even GS row and the next one), whole 8-pixel columns with SSE2, the rest through StoreRowSpan.
std::uint32_t HostToLocalCt32(std::uint8_t *memoryBytes, const BufferView &view, std::uint32_t x0, std::uint32_t y0,
                              std::uint32_t width, std::uint32_t height, const std::uint8_t *data,
                              const std::uint32_t *pageOffsets, const Ct32Columns &columns,
                              std::uint64_t *changedBlocks) {
    constexpr std::uint32_t kPageWidth = 64, kPageHeight = 32;
    std::uint32_t diff = 0;
    __m128i changed = _mm_setzero_si128();
    const std::uint32_t perRow = std::max<std::uint32_t>(1u, view.bufferWidth * 64u / kPageWidth);
    std::uint32_t row = 0;
    while (row < height) {
        const std::uint32_t y = (y0 + row) % memory::kTransferCoordinateWrap;
        if ((y & 1u) != 0 || row + 1 >= height) {
            diff |= StoreRowSpan<32>(memoryBytes, view, x0, y0, width, data, pageOffsets, kPageWidth, kPageHeight, row, 0,
                                     width, changedBlocks);
            ++row;
            continue;
        }
        // Columns before the first whole 8-pixel column, the whole columns, the columns after them.
        const std::uint32_t lead = std::min<std::uint32_t>(width, (8u - (x0 % 8u)) % 8u);
        const std::uint32_t whole = (width - lead) / 8u;
        const std::uint32_t tail = lead + whole * 8u;
        for (std::uint32_t r = 0; r < 2; ++r) {
            diff |= StoreRowSpan<32>(memoryBytes, view, x0, y0, width, data, pageOffsets, kPageWidth, kPageHeight, row + r,
                                     0, lead, changedBlocks);
            diff |= StoreRowSpan<32>(memoryBytes, view, x0, y0, width, data, pageOffsets, kPageWidth, kPageHeight, row + r,
                                     tail, width, changedBlocks);
        }
        const std::uint32_t pageRow = (y / kPageHeight) * perRow;
        const std::uint32_t *colBase = columns.base[(y % kPageHeight) / 2u];
        const std::uint8_t *row0 = data + (static_cast<std::uint64_t>(row) * width) * 4u;
        const std::uint8_t *row1 = row0 + static_cast<std::uint64_t>(width) * 4u;
        for (std::uint32_t k = 0; k < whole; ++k) {
            const std::uint32_t column = lead + 8u * k;
            const std::uint32_t x = (x0 + column) % memory::kTransferCoordinateWrap;  // a multiple of 8
            const std::uint32_t pageBlock = view.basePointer + memory::kBlocksPerPage * (pageRow + x / kPageWidth);
            std::uint8_t *page = memoryBytes + ((pageBlock * memory::kBlockBytes) & (memory::kLocalMemoryBytes - 1u));
            std::uint8_t *out = page + colBase[(x % kPageWidth) / 8u];
            const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row0 + column * 4u));
            const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row0 + column * 4u + 16u));
            const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row1 + column * 4u));
            const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row1 + column * 4u + 16u));
            const __m128i q0 = _mm_unpacklo_epi64(a, c), q1 = _mm_unpackhi_epi64(a, c);
            const __m128i q2 = _mm_unpacklo_epi64(b, d), q3 = _mm_unpackhi_epi64(b, d);
            __m128i *o = reinterpret_cast<__m128i *>(out);  // a column is 64-byte aligned in its block
            const __m128i diffs = _mm_or_si128(
                _mm_or_si128(_mm_xor_si128(_mm_load_si128(o + 0), q0), _mm_xor_si128(_mm_load_si128(o + 1), q1)),
                _mm_or_si128(_mm_xor_si128(_mm_load_si128(o + 2), q2), _mm_xor_si128(_mm_load_si128(o + 3), q3)));
            changed = _mm_or_si128(changed, diffs);
            if (changedBlocks && _mm_movemask_epi8(_mm_cmpeq_epi8(diffs, _mm_setzero_si128())) != 0xFFFF) {
                MarkBlocks(changedBlocks, pageBlock, 1u << (colBase[(x % kPageWidth) / 8u] >> 8));  // IN-453
            }
            _mm_store_si128(o + 0, q0);
            _mm_store_si128(o + 1, q1);
            _mm_store_si128(o + 2, q2);
            _mm_store_si128(o + 3, q3);
        }
        row += 2;
    }
    if (_mm_movemask_epi8(_mm_cmpeq_epi8(changed, _mm_setzero_si128())) != 0xFFFF) diff |= 1u;
    return diff;
}

template <std::uint32_t Bpp>
std::uint32_t HostToLocalPageAligned(std::uint8_t *memoryBytes, const BufferView &view, std::uint32_t x0,
                                     std::uint32_t y0, std::uint32_t width, std::uint32_t height,
                                     const std::uint8_t *data, const std::uint32_t *pageOffsets,
                                     std::uint32_t pageWidth, std::uint32_t pageHeight, std::uint64_t *changedBlocks) {
    std::uint32_t diff = 0;
    for (std::uint32_t row = 0; row < height; ++row) {
        diff |= StoreRowSpan<Bpp>(memoryBytes, view, x0, y0, width, data, pageOffsets, pageWidth, pageHeight, row, 0,
                                  width, changedBlocks);
    }
    return diff;
}

}  // namespace

std::uint32_t BlockOfPixel(const BufferView &view, std::uint32_t x, std::uint32_t y) {
    // IN-487: called for every block cell of every upload, several times: page sizes are powers of two, so shifts and
    // masks read once from a table replace the divisions (the largest upload function on the i5-7200U).
    struct Shape {
        std::uint32_t tableOffset = 0, widthBits = 0, heightBits = 0;
        bool zXor = false;
    };
    static const memory::GpuAddressTables tables = memory::gpuAddressTables();
    static const std::array<Shape, 64> shapes = [] {
        std::array<Shape, 64> built{};
        for (std::uint32_t code = 0; code < 64u; ++code) {
            const std::uint32_t width = tables.directory[code * 4u + 1u], height = tables.directory[code * 4u + 2u];
            built[code] = Shape{tables.directory[code * 4u], width != 0 ? static_cast<std::uint32_t>(std::countr_zero(width)) : 0u,
                                height != 0 ? static_cast<std::uint32_t>(std::countr_zero(height)) : 0u,
                                tables.directory[code * 4u + 3u] != 0};
        }
        return built;
    }();
    const Shape &shape = shapes[static_cast<std::uint32_t>(view.psm) & 63u];
    const std::uint32_t perRow = std::max<std::uint32_t>(1u, (view.bufferWidth * 64u) >> shape.widthBits);
    const std::uint32_t inPage = ((y & ((1u << shape.heightBits) - 1u)) << shape.widthBits) + (x & ((1u << shape.widthBits) - 1u));
    const std::uint32_t entry = tables.entries[shape.tableOffset + inPage];
    std::uint32_t block = view.basePointer +
                          memory::kBlocksPerPage * ((y >> shape.heightBits) * perRow + (x >> shape.widthBits)) + (entry & 31u);
    if (shape.zXor) block ^= 24u;
    return block & (memory::kBlocksTotal - 1u);
}

bool PagesOfRect(const BufferView &view, std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1, PageSet &pages) {
    pages.reset();
    const memory::PsmInfo &info = memory::psmInfo(view.psm);
    const std::int32_t pageWidth = static_cast<std::int32_t>(info.pageWidth), pageHeight = static_cast<std::int32_t>(info.pageHeight);
    const std::int32_t bufferWidth = static_cast<std::int32_t>(view.bufferWidth) * 64;
    if (pageWidth == 0 || pageHeight == 0 || bufferWidth == 0 || bufferWidth % pageWidth != 0) return false;
    if (x1 < x0 || y1 < y0) return true;
    if (x0 < 0 || y0 < 0 || x1 >= bufferWidth || x1 >= 2048 || y1 >= 2048) return false;
    const bool pageAligned = view.basePointer % memory::kBlocksPerPage == 0;
    const std::int32_t stepX = pageAligned ? pageWidth : static_cast<std::int32_t>(info.blockWidth);
    const std::int32_t stepY = pageAligned ? pageHeight : static_cast<std::int32_t>(info.blockHeight);
    const std::int64_t cells = static_cast<std::int64_t>(x1 / stepX - x0 / stepX + 1) * (y1 / stepY - y0 / stepY + 1);
    if (!pageAligned && cells > 1024) return false;
    for (std::int32_t row = y0 / stepY; row <= y1 / stepY; ++row) {
        for (std::int32_t column = x0 / stepX; column <= x1 / stepX; ++column) {
            pages.set(BlockOfPixel(view, static_cast<std::uint32_t>(column * stepX), static_cast<std::uint32_t>(row * stepY)) /
                      memory::kBlocksPerPage);
        }
    }
    return true;
}

bool FastHostToLocal(std::uint8_t *memoryBytes, const BufferView &view, std::uint32_t x0, std::uint32_t y0,
                     std::uint32_t width, std::uint32_t height, const std::uint8_t *data, std::size_t byteCount,
                     UploadSwizzle swizzle, bool *changed, std::uint64_t *changedBlocks) {
    if (changed) *changed = true;
    static const memory::GpuAddressTables tables = memory::gpuAddressTables();
    const memory::PsmInfo &info = memory::psmInfo(view.psm);
    const std::uint32_t bpp = info.transferBpp;
    if (static_cast<std::uint64_t>(width) * height * bpp > static_cast<std::uint64_t>(byteCount) * 8u) return false;
    const std::uint32_t code = static_cast<std::uint32_t>(view.psm) & 63u;
    const std::uint32_t tableOffset = tables.directory[code * 4u];
    const std::uint32_t pageWidth = tables.directory[code * 4u + 1u];
    const std::uint32_t pageHeight = tables.directory[code * 4u + 2u];
    const bool zXor = tables.directory[code * 4u + 3u] != 0;
    if (pageWidth == 0 || pageHeight == 0) return false;
    if (swizzle == UploadSwizzle::PageRuns && view.basePointer % memory::kBlocksPerPage == 0) {
        const std::uint32_t *pageOffsets = InPageOffsetTables().perPsm[code].data();
        std::uint32_t diff = 0;
        // IN-452: PSMCT32 through whole columns (the game's race uploads are mostly CT32); COREGS_CORE2_CT32_COLUMNS=0 for A/B.
        static const bool ct32Columns = std::getenv("COREGS_CORE2_CT32_COLUMNS") == nullptr ||
                                        std::strcmp(std::getenv("COREGS_CORE2_CT32_COLUMNS"), "0") != 0;
        if (ct32Columns && view.psm == memory::Psm::Ct32) {
            const Ct32Columns &columns = Ct32ColumnTable(InPageOffsetTables().perPsm[code]);
            if (columns.valid) {
                diff = HostToLocalCt32(memoryBytes, view, x0, y0, width, height, data, pageOffsets, columns, changedBlocks);
                if (changed) *changed = diff != 0;
                return true;
            }
        }
        switch (bpp) {
            case 32: diff = HostToLocalPageAligned<32>(memoryBytes, view, x0, y0, width, height, data, pageOffsets, pageWidth, pageHeight, changedBlocks); break;
            case 24: diff = HostToLocalPageAligned<24>(memoryBytes, view, x0, y0, width, height, data, pageOffsets, pageWidth, pageHeight, changedBlocks); break;
            case 16: diff = HostToLocalPageAligned<16>(memoryBytes, view, x0, y0, width, height, data, pageOffsets, pageWidth, pageHeight, changedBlocks); break;
            case 8: diff = HostToLocalPageAligned<8>(memoryBytes, view, x0, y0, width, height, data, pageOffsets, pageWidth, pageHeight, changedBlocks); break;
            default: diff = HostToLocalPageAligned<4>(memoryBytes, view, x0, y0, width, height, data, pageOffsets, pageWidth, pageHeight, changedBlocks); break;
        }
        if (changed) *changed = diff != 0;
        return true;
    }
    const std::uint32_t perRow = std::max<std::uint32_t>(1u, view.bufferWidth * 64u / pageWidth);
    const std::uint32_t *entries = tables.entries.data() + tableOffset;
    std::uint64_t index = 0;
    for (std::uint32_t row = 0; row < height; ++row) {
        const std::uint32_t y = (y0 + row) % memory::kTransferCoordinateWrap;
        const std::uint32_t pageRow = (y / pageHeight) * perRow;
        const std::uint32_t *rowEntries = entries + (y % pageHeight) * pageWidth;
        for (std::uint32_t column = 0; column < width; ++column, ++index) {
            const std::uint32_t x = (x0 + column) % memory::kTransferCoordinateWrap;
            const std::uint32_t entry = rowEntries[x % pageWidth];
            std::uint32_t block = view.basePointer + memory::kBlocksPerPage * (pageRow + x / pageWidth) + (entry & 31u);
            if (zXor) block ^= 24u;
            block &= memory::kBlocksTotal - 1u;
            // The byte holding the pixel's lowest bit (T8H, T4HL, T4HH: the word's top byte) and the shift in it.
            const std::uint32_t wordShift = (entry >> 13) & 31u;
            const std::uint32_t at =
                (block * memory::kBlockBytes + ((entry >> 5) & 255u) + wordShift / 8u) & (memory::kLocalMemoryBytes - 1u);
            const std::uint32_t shift = wordShift % 8u;
            std::uint8_t *out = memoryBytes + at;
            if (changedBlocks) {  // IN-453: conservative on this path: every written block counts as changed
                const std::uint32_t written = at / memory::kBlockBytes;
                changedBlocks[written >> 6] |= std::uint64_t{1} << (written & 63u);
            }
            switch (bpp) {
                case 32: std::memcpy(out, data + index * 4u, 4); break;
                case 24: std::memcpy(out, data + index * 3u, 3); break;
                case 16: std::memcpy(out, data + index * 2u, 2); break;
                case 8: *out = data[index]; break;
                default: {  // 4 bits, low nibble first in the host stream
                    const std::uint8_t nibble =
                        (index & 1u) ? static_cast<std::uint8_t>(data[index / 2] >> 4) : (data[index / 2] & 0x0Fu);
                    *out = static_cast<std::uint8_t>((*out & ~(0x0Fu << shift)) | (nibble << shift));
                    break;
                }
            }
        }
    }
    return true;
}

}  // namespace coregs::core2
