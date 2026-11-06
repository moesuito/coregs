// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#include "hw_clut.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace coregs {
namespace clut {
namespace {

// In-block column tables for 32-bit (8x8 block) and 16-bit (16x8 block) formats,
// rows indexed by block row parity, entries by block x (chapter 01.3).
constexpr uint8_t kWordInBlock32[2][8] = {
    {0, 1, 4, 5, 8, 9, 12, 13},
    {2, 3, 6, 7, 10, 11, 14, 15},
};

constexpr uint8_t kHalfwordInBlock16[2][16] = {
    {0, 2, 8, 10, 16, 18, 24, 26, 1, 3, 9, 11, 17, 19, 25, 27},
    {4, 6, 12, 14, 20, 22, 28, 30, 5, 7, 13, 15, 21, 23, 29, 31},
};

// Block numbers inside a page, rows top to bottom (chapter 01.3).
constexpr uint8_t kBlocksCt32[4][8] = {
    {0, 1, 4, 5, 16, 17, 20, 21},
    {2, 3, 6, 7, 18, 19, 22, 23},
    {8, 9, 12, 13, 24, 25, 28, 29},
    {10, 11, 14, 15, 26, 27, 30, 31},
};

constexpr uint8_t kBlocksCt16[8][4] = {
    {0, 2, 8, 10},
    {1, 3, 9, 11},
    {4, 6, 12, 14},
    {5, 7, 13, 15},
    {16, 18, 24, 26},
    {17, 19, 25, 27},
    {20, 22, 28, 30},
    {21, 23, 29, 31},
};

constexpr uint8_t kBlocksCt16S[8][4] = {
    {0, 2, 16, 18},
    {1, 3, 17, 19},
    {8, 10, 24, 26},
    {9, 11, 25, 27},
    {4, 6, 20, 22},
    {5, 7, 21, 23},
    {12, 14, 28, 30},
    {13, 15, 29, 31},
};

bool IsCt16Family(uint32_t format) { return format == psm::kCt16 || format == psm::kCt16S; }

bool IsCpsmValid(uint32_t cpsm) { return cpsm == psm::kCt32 || IsCt16Family(cpsm); }

bool SameLoadKey(const Tex0Write& a, const Tex0Write& b) {
    return a.cbp == b.cbp && a.cpsm == b.cpsm && a.csm == b.csm && a.csa == b.csa &&
           IsIndexedPsm(a.psm) && IsIndexedPsm(b.psm) &&
           ((a.psm & 7u) == 3u) == ((b.psm & 7u) == 3u) &&
           a.texclutCbw == b.texclutCbw && a.texclutCou == b.texclutCou &&
           a.texclutCov == b.texclutCov;
}

uint32_t ReadLittle(const uint8_t* bytes, uint32_t count) {
    uint32_t value = 0;
    for (uint32_t i = 0; i < count; ++i) {
        value |= static_cast<uint32_t>(bytes[i]) << (8u * i);
    }
    return value;
}

}  // namespace

bool LocatePixel(uint32_t format, uint32_t bufferPointer, uint32_t bufferWidth,
                 uint32_t x, uint32_t y, PixelLocation& out) {
    uint32_t pageWidth = 0;
    uint32_t pageHeight = 0;
    uint32_t blockWidth = 0;
    uint32_t blockHeight = 0;
    switch (format) {
        case psm::kCt32:
            pageWidth = 64; pageHeight = 32; blockWidth = 8; blockHeight = 8;
            break;
        case psm::kCt16:
        case psm::kCt16S:
            pageWidth = 64; pageHeight = 64; blockWidth = 16; blockHeight = 8;
            break;
        default:
            return false;
    }

    // Pages per row: BW counts 64-pixel units and both formats have 64-pixel pages.
    const uint32_t pagesPerRow = bufferWidth;
    const uint32_t page = (y / pageHeight) * pagesPerRow + (x / pageWidth);
    const uint32_t inPageX = (x % pageWidth) / blockWidth;
    const uint32_t inPageY = (y % pageHeight) / blockHeight;

    uint32_t blockInPage = 0;
    if (format == psm::kCt32) {
        blockInPage = kBlocksCt32[inPageY][inPageX];
    } else if (format == psm::kCt16) {
        blockInPage = kBlocksCt16[inPageY][inPageX];
    } else {
        blockInPage = kBlocksCt16S[inPageY][inPageX];
    }

    // Z formats XOR the in-page entry; CLUT sources are colour formats only.
    out.block = (bufferPointer + kBlocksPerPage * page + blockInPage) % kBlocksTotal;

    const uint32_t bx = x % blockWidth;
    const uint32_t by = y % blockHeight;
    const uint32_t column = by / 2u;
    if (format == psm::kCt32) {
        const uint32_t word = kWordInBlock32[by & 1u][bx];
        out.byteInBlock = 64u * column + 4u * word;
    } else {
        const uint32_t halfword = kHalfwordInBlock16[by & 1u][bx];
        out.byteInBlock = 64u * column + 2u * halfword;
    }
    return true;
}

void CsmEntryPosition(uint32_t index, bool eightBitIndex, uint32_t& x, uint32_t& y) {
    if (eightBitIndex) {
        x = (index & 7u) + 8u * ((index >> 4) & 1u);
        y = 2u * (index >> 5) + ((index >> 3) & 1u);
    } else {
        x = index & 7u;
        y = index >> 3;
    }
}

TexaState TexaState::FromRegister(uint64_t texa) {
    TexaState state;
    state.ta0 = static_cast<uint32_t>(BitRange(texa, 0, 7));
    state.aem = BitRange(texa, 15, 15) != 0;
    state.ta1 = static_cast<uint32_t>(BitRange(texa, 32, 39));
    return state;
}

uint32_t ExpandRgb16(uint16_t value, const TexaState& texa, bool wholeValueRule) {
    const uint32_t r = (static_cast<uint32_t>(value) & 0x1Fu) << 3;
    const uint32_t g = ((static_cast<uint32_t>(value) >> 5) & 0x1Fu) << 3;
    const uint32_t b = ((static_cast<uint32_t>(value) >> 10) & 0x1Fu) << 3;
    // IN-07 single rule (also used by pixel/): the A bit selects TA1 first; otherwise AEM with
    // RGB = 0 gives alpha 0, else TA0. With the A bit clear, "RGB = 0" equals "whole value = 0",
    // so wholeValueRule no longer changes the result (it is kept for the existing call sites).
    uint32_t a = texa.ta0;
    if ((value & 0x8000u) != 0) {
        a = texa.ta1;
    } else if (texa.aem && r == 0u && g == 0u && b == 0u) {
        a = 0;
    }
    (void)wholeValueRule;
    return r | (g << 8) | (b << 16) | (a << 24);
}

uint32_t ExpandRgb24(uint32_t value, const TexaState& texa) {
    const uint32_t r = value & 0xFFu;
    const uint32_t g = (value >> 8) & 0xFFu;
    const uint32_t b = (value >> 16) & 0xFFu;
    uint32_t a = texa.ta0;
    if (texa.aem && r == 0u && g == 0u && b == 0u) {
        a = 0;
    }
    return r | (g << 8) | (b << 16) | (a << 24);
}

ClutUnit::ClutUnit(CounterSet& counters) : counters_(counters) {}

void ClutUnit::Reset() {
    std::fill(std::begin(halfwords_), std::end(halfwords_), uint16_t{0});
    sourceFirst_ = kBlocksTotal;
    sourceLast_ = 0;
    cbp0_ = 0;
    cbp1_ = 0;
    haveLast_ = false;
    stale_ = true;
    sourceBlocks_.clear();
    ++generation_;
}

ClutDecision ClutUnit::Decide(const Tex0Write& write) const {
    ClutDecision decision;
    if (!IsIndexedPsm(write.psm)) {
        return decision;  // nothing happens, CBP0/CBP1 unchanged
    }
    switch (write.cld) {
        case 1:
            decision.load = true;
            break;
        case 2:
            decision.load = true;
            decision.setCbp0 = true;
            break;
        case 3:
            decision.load = true;
            decision.setCbp1 = true;
            break;
        case 4:
            if (write.cbp != cbp0_) {
                decision.load = true;
                decision.setCbp0 = true;
            }
            break;
        case 5:
            if (write.cbp != cbp1_) {
                decision.load = true;
                decision.setCbp1 = true;
            }
            break;
        default:  // 0, 6 and 7 do nothing
            break;
    }
    decision.newCbp = write.cbp;
    return decision;
}

void ClutUnit::ApplySideEffects(const ClutDecision& decision) {
    if (decision.setCbp0) {
        cbp0_ = decision.newCbp;
    }
    if (decision.setCbp1) {
        cbp1_ = decision.newCbp;
    }
}

bool ClutUnit::ProcessWrite(const Tex0Write& write, const uint8_t* localMemory,
                            const std::function<void()>& flushBeforeLoad) {
    const ClutDecision decision = Decide(write);
    if (!decision.load) {
        return false;
    }
    counters_.Add(Counter::ClutLoadsRequested);
    ApplySideEffects(decision);

    Tex0Write effective = write;
    const bool redundant = haveLast_ && !stale_ && !alwaysLoad_ && SameLoadKey(effective, last_);
    if (redundant) {
        counters_.Add(Counter::ClutLoadsSkipped);
        return false;
    }
    if (flushBeforeLoad) {
        flushBeforeLoad();
    }
    Copy(effective, localMemory);
    counters_.Add(Counter::ClutLoadsPerformed);
    return true;
}

namespace {

// IN-457 (CPU survey R6): a CSM1 load reads entry e at CsmEntryPosition(e) of a one-page-wide buffer at CBP. That
// position's block is CBP plus a fixed offset and its byte in the block is fixed, per CPSM and entry count: LocatePixel
// at base 0, computed once, so the load is a table walk instead of a division and a switch per entry. (LocatePixel adds
// the buffer pointer to the in-page block and wraps at the memory size, so block(CBP) = (CBP + block(0)) % blocks.)
struct CsmOffsets {
    bool valid = false;
    uint16_t block[256] = {};
    uint8_t byteInBlock[256] = {};
};

const CsmOffsets& CsmOffsetsFor(uint32_t cpsm, bool eightBit) {
    static CsmOffsets tables[3][2];  // CT32, CT16, CT16S x 4-bit, 8-bit
    static bool built = false;
    if (!built) {
        const uint32_t formats[3] = {psm::kCt32, psm::kCt16, psm::kCt16S};
        for (uint32_t f = 0; f < 3; ++f) {
            for (uint32_t b = 0; b < 2; ++b) {
                CsmOffsets& t = tables[f][b];
                t.valid = true;
                const uint32_t count = b ? 256u : 16u;
                for (uint32_t e = 0; e < count; ++e) {
                    uint32_t x = 0, y = 0;
                    CsmEntryPosition(e, b != 0, x, y);
                    PixelLocation location;
                    if (!LocatePixel(formats[f], 0, 1, x, y, location)) {
                        t.valid = false;
                        break;
                    }
                    t.block[e] = static_cast<uint16_t>(location.block);
                    t.byteInBlock[e] = static_cast<uint8_t>(location.byteInBlock);
                }
            }
        }
        built = true;
    }
    const uint32_t f = cpsm == psm::kCt32 ? 0u : cpsm == psm::kCt16 ? 1u : 2u;
    return tables[f][eightBit ? 1 : 0];
}

}  // namespace

namespace {

// A written range [begin, begin + count) (wrapping at kBlocksTotal) that misses the blocks [first, last] entirely.
bool RangeMisses(uint32_t blockBegin, uint32_t blockCount, uint32_t first, uint32_t last) {
    if (blockCount >= kBlocksTotal) return false;
    const uint32_t begin = blockBegin % kBlocksTotal;
    const uint32_t end = begin + blockCount;  // exclusive; may pass kBlocksTotal (wraps)
    return end <= kBlocksTotal ? (end <= first || begin > last) : (begin > last && end - kBlocksTotal <= first);
}

}  // namespace

// Writes the entries of a load into the CLUT buffer from where they were read in memory.
void ClutUnit::PlaceEntries(const SourceMemo& memo, const Tex0Write& write, uint32_t entryCount) {
    const uint32_t csaGroup = write.csa & 15u;
    if (write.cpsm == psm::kCt32) {
        if (csaGroup == 0u && entryCount == 256u) {
            std::memcpy(halfwords_, memo.low, sizeof memo.low);
            std::memcpy(halfwords_ + 256, memo.high, sizeof memo.high);
            return;
        }
        for (uint32_t e = 0; e < entryCount; ++e) {
            // Palette group g lives at buffer group g + CSA (clamped rows are not loaded).
            const uint32_t group = (e >> 4) + csaGroup;
            if (group > 15u) continue;
            const uint32_t index = 16u * group + (e & 15u);
            halfwords_[index] = memo.low[e];
            halfwords_[index + 256u] = memo.high[e];
        }
    } else {
        for (uint32_t e = 0; e < entryCount; ++e) halfwords_[(16u * write.csa + e) & 511u] = memo.low[e];
    }
}

void ClutUnit::Copy(const Tex0Write& write, const uint8_t* localMemory) {
    if (!IsCpsmValid(write.cpsm)) {
        counters_.Add(Counter::ClutBadCpsm);
        haveLast_ = false;
        return;
    }
    const bool eightBit = (write.psm & 7u) == 3u;
    const uint32_t entryCount = eightBit ? 256u : 16u;
    const bool is32 = write.cpsm == psm::kCt32;
    if (write.csm == 1u && write.cpsm != psm::kCt16) {
        counters_.Add(Counter::Csm2NonSixteen);
    }

    const uint64_t key = uint64_t{write.cbp} | (uint64_t{write.cpsm & 63u} << 14) | (uint64_t{write.csm & 1u} << 20) |
                         (uint64_t{eightBit ? 1u : 0u} << 21) |
                         (write.csm == 1u ? (uint64_t{write.texclutCbw & 63u} << 22) |
                                                (uint64_t{write.texclutCou & 63u} << 28) |
                                                (uint64_t{write.texclutCov & 1023u} << 34)
                                          : 0ull);
    if (sourceMemo_.empty()) sourceMemo_.resize(kMemoBuckets);
    const uint32_t home = static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ull) >> 54);  // 10 bits: kMemoBuckets
    SourceMemo* memo = nullptr;
    SourceMemo* oldest = &sourceMemo_[home];
    for (uint32_t probe = 0; probe < kMemoProbes; ++probe) {
        SourceMemo& entry = sourceMemo_[(home + probe) % kMemoBuckets];
        if (entry.key == key) {
            memo = &entry;
            break;
        }
        if (entry.lastUse < oldest->lastUse) oldest = &entry;
    }
    if (memo) {
        bool answered = !copiesData_;  // IN-360: without data copies only the source blocks matter
        if (copiesData_ && memo->dataValid && dataMemoOn_) {
            answered = true;
            for (uint32_t i = 0; i < memo->blockCount && answered; ++i) {
                answered = std::memcmp(memo->bytes + i * 256u, localMemory + size_t{memo->blocks[i]} * 256u, 256u) == 0;
            }
            if (answered) {
                PlaceEntries(*memo, write, entryCount);
                ++dataMemoHits_;
            }
        }
        if (answered) {
            sourceBlocks_.assign(memo->blocks, memo->blocks + memo->blockCount);
            sourceFirst_ = memo->first;
            sourceLast_ = memo->last;
            memo->lastUse = ++memoUses_;
            ++generation_;
            haveLast_ = true;
            stale_ = false;
            last_ = write;
            return;
        }
    } else {
        memo = oldest;
    }
    memo->key = ~0ull;  // valid again only when every block fits below
    memo->blockCount = 0;
    memo->dataValid = false;
    memo->lastUse = ++memoUses_;
    if (copiesData_) ++dataMemoReads_;

    sourceBlocks_.clear();
    sourceFirst_ = kBlocksTotal;
    sourceLast_ = 0;
    const CsmOffsets* offsets = write.csm == 0u ? &CsmOffsetsFor(write.cpsm, eightBit) : nullptr;
    if (offsets && !offsets->valid) offsets = nullptr;
    bool everyEntryRead = true;
    uint32_t lastBlock = kBlocksTotal;
    for (uint32_t e = 0; e < entryCount; ++e) {
        PixelLocation location;
        if (offsets) {  // IN-457
            location.block = (write.cbp + offsets->block[e]) % kBlocksTotal;
            location.byteInBlock = offsets->byteInBlock[e];
        } else {
            uint32_t x = 0;
            uint32_t y = 0;
            uint32_t base = write.cbp;
            uint32_t width = 1;
            if (write.csm == 0u) {
                CsmEntryPosition(e, eightBit, x, y);
            } else {
                x = write.texclutCou * 16u + e;
                y = write.texclutCov;
                width = write.texclutCbw;
            }
            if (!LocatePixel(write.cpsm, base, width, x, y, location)) {
                // An entry with no position leaves its place in the buffer as it was; such a load is not remembered.
                everyEntryRead = false;
                const uint32_t group = (e >> 4) + (write.csa & 15u);
                if (is32 && group <= 15u) {
                    memo->low[e] = halfwords_[16u * group + (e & 15u)];
                    memo->high[e] = halfwords_[16u * group + (e & 15u) + 256u];
                } else if (!is32) {
                    memo->low[e] = halfwords_[(16u * write.csa + e) & 511u];
                }
                continue;
            }
        }
        if (location.block != lastBlock) {
            lastBlock = location.block;
            if (std::find(sourceBlocks_.begin(), sourceBlocks_.end(), location.block) == sourceBlocks_.end()) {
                sourceBlocks_.push_back(location.block);
            }
            sourceFirst_ = std::min(sourceFirst_, location.block);
            sourceLast_ = std::max(sourceLast_, location.block);
        }
        if (!copiesData_) continue;
        const uint8_t* source = localMemory + location.PhysicalByte();
        if (is32) {
            const uint32_t raw = ReadLittle(source, 4);
            memo->low[e] = static_cast<uint16_t>(raw & 0xFFFFu);
            memo->high[e] = static_cast<uint16_t>(raw >> 16);
        } else {
            memo->low[e] = static_cast<uint16_t>(ReadLittle(source, 2));
        }
    }
    if (copiesData_) PlaceEntries(*memo, write, entryCount);
    if (sourceBlocks_.size() <= kMemoBlocks) {
        memo->key = key;
        memo->first = sourceFirst_;
        memo->last = sourceLast_;
        memo->blockCount = static_cast<uint32_t>(sourceBlocks_.size());
        for (uint32_t i = 0; i < memo->blockCount; ++i) {
            memo->blocks[i] = static_cast<uint16_t>(sourceBlocks_[i]);
            if (copiesData_) std::memcpy(memo->bytes + i * 256u, localMemory + size_t{sourceBlocks_[i]} * 256u, 256u);
        }
        memo->dataValid = copiesData_ && everyEntryRead;
    }

    ++generation_;
    haveLast_ = true;
    stale_ = false;
    last_ = write;
}

void ClutUnit::PeekLoad(const Tex0Write& write, const uint8_t* localMemory, uint16_t* out) const {
    std::copy(std::begin(halfwords_), std::end(halfwords_), out);
    if (!IsCpsmValid(write.cpsm)) return;
    const bool eightBit = (write.psm & 7u) == 3u;
    const uint32_t entryCount = eightBit ? 256u : 16u;
    const bool is32 = write.cpsm == psm::kCt32;
    // The same entries and places as Copy and PlaceEntries, read straight from memory.
    for (uint32_t e = 0; e < entryCount; ++e) {
        uint32_t x = 0, y = 0, width = 1;
        if (write.csm == 0u) {
            CsmEntryPosition(e, eightBit, x, y);
        } else {
            x = write.texclutCou * 16u + e;
            y = write.texclutCov;
            width = write.texclutCbw;
        }
        PixelLocation location;
        if (!LocatePixel(write.cpsm, write.cbp, width, x, y, location)) continue;
        const uint8_t* source = localMemory + location.PhysicalByte();
        if (is32) {
            const uint32_t group = (e >> 4) + (write.csa & 15u);
            if (group > 15u) continue;
            const uint32_t raw = ReadLittle(source, 4);
            out[16u * group + (e & 15u)] = static_cast<uint16_t>(raw & 0xFFFFu);
            out[16u * group + (e & 15u) + 256u] = static_cast<uint16_t>(raw >> 16);
        } else {
            out[(16u * write.csa + e) & 511u] = static_cast<uint16_t>(ReadLittle(source, 2));
        }
    }
}

void ClutUnit::SetHalfwords(const uint16_t* halfwords) {
    std::copy(halfwords, halfwords + 512, std::begin(halfwords_));
    ++generation_;
}

void ClutUnit::NoteBlocksWritten(uint32_t blockBegin, uint32_t blockCount) {
    if (!haveLast_ || stale_ || sourceFirst_ > sourceLast_) {
        return;
    }
    // Early-out: the written range misses [sourceFirst_, sourceLast_] entirely.
    if (RangeMisses(blockBegin, blockCount, sourceFirst_, sourceLast_)) return;
    const uint32_t begin = blockBegin % kBlocksTotal;
    for (const uint32_t block : sourceBlocks_) {
        if (blockCount >= kBlocksTotal || (block + kBlocksTotal - begin) % kBlocksTotal < blockCount) {
            stale_ = true;
            return;
        }
    }
}

Palette ClutUnit::Decode(uint32_t format, uint32_t cpsm, uint32_t csa, uint64_t texa) const {
    return DecodeHalfwords(halfwords_, format, cpsm, csa, texa);
}

Palette DecodeHalfwords(const uint16_t* halfwords, uint32_t format, uint32_t cpsm, uint32_t csa, uint64_t texa) {
    Palette palette;
    palette.count = (format & 7u) == 3u ? 256u : 16u;
    const TexaState state = TexaState::FromRegister(texa);
    const uint32_t csaGroup = csa & 15u;

    palette.minAlpha = 255;
    palette.maxAlpha = 0;
    for (uint32_t e = 0; e < palette.count; ++e) {
        uint32_t value = 0;
        if (cpsm == psm::kCt32) {
            const uint32_t group = std::min<uint32_t>((e >> 4) + csaGroup, 15u);
            const uint32_t index = 16u * group + (e & 15u);
            value = static_cast<uint32_t>(halfwords[index]) |
                    (static_cast<uint32_t>(halfwords[index + 256u]) << 16);
        } else if (IsCt16Family(cpsm)) {
            const uint32_t index = (16u * csa + e) & 511u;
            value = ExpandRgb16(halfwords[index], state, true);
        }
        palette.entries[e] = value;
        const uint8_t alpha = static_cast<uint8_t>(value >> 24);
        palette.minAlpha = std::min(palette.minAlpha, alpha);
        palette.maxAlpha = std::max(palette.maxAlpha, alpha);
    }
    return palette;
}

}  // namespace clut
}  // namespace coregs
