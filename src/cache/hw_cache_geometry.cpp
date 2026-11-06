// SPDX-License-Identifier: MIT
// Copyright 2026 GT4Recomp contributors
//
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters
// 04.1, 05.1 and 05.5. Clean room.

#include "hw_cache_geometry.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace coregs::cache {
namespace {

constexpr std::int64_t kWrapBlocks = memory::kBlocksTotal;  // 16384
constexpr std::uint32_t kZBlockXor = 24;
constexpr std::int64_t kBlocksPerPage = memory::kBlocksPerPage;
constexpr std::uint32_t kTransferWrapMask = memory::kTransferCoordinateWrap - 1u;

std::int64_t floorDiv(std::int64_t value, std::int64_t divisor) {
    std::int64_t quotient = value / divisor;
    if ((value % divisor != 0) && ((value < 0) != (divisor < 0))) --quotient;
    return quotient;
}

std::int64_t ceilDiv(std::int64_t value, std::int64_t divisor) {
    return -floorDiv(-value, divisor);
}

// Unwrapped block number of the first block of the page holding (x, y).
std::int64_t unwrappedPageBase(const BufferView& view, std::uint32_t x, std::uint32_t y) {
    const memory::PsmInfo& info = memory::psmInfo(view.psm);
    const std::uint32_t perRow = memory::pagesPerRow(view.bufferWidth, view.psm);
    const std::int64_t page = static_cast<std::int64_t>(y / info.pageHeight) * perRow + x / info.pageWidth;
    return static_cast<std::int64_t>(view.basePointer) + page * kBlocksPerPage;
}

}  // namespace

// ---------------------------------------------------------------------------
// Channels and formats
// ---------------------------------------------------------------------------

std::uint8_t supportedChannels(Psm psm) {
    switch (psm) {
        case Psm::Ct32:
            return kChanAll;
        case Psm::Ct24:
            return kChanRgb;
        case Psm::Ct16:
        case Psm::Ct16S:
            return kChanAll;  // one channel: the whole halfword
        default:
            return 0;
    }
}

std::uint32_t channelWordMask(Psm psm, std::uint8_t channels) {
    channels = static_cast<std::uint8_t>(channels & supportedChannels(psm));
    switch (psm) {
        case Psm::Ct32: {
            std::uint32_t mask = 0;
            if (channels & kChanRgb) mask |= 0x00FFFFFFu;
            if (channels & kChanAlphaLo) mask |= 0x0F000000u;
            if (channels & kChanAlphaHi) mask |= 0xF0000000u;
            return mask;
        }
        case Psm::Ct24:
            return (channels & kChanRgb) ? 0x00FFFFFFu : 0u;
        case Psm::Ct16:
        case Psm::Ct16S:
            return channels ? 0xFFFFu : 0u;
        default:
            return 0u;
    }
}

std::uint32_t storageWordMask(Psm psm) {
    switch (psm) {
        case Psm::Ct32:
            return 0xFFFFFFFFu;
        case Psm::Ct24:
            return 0x00FFFFFFu;
        case Psm::Ct16:
        case Psm::Ct16S:
            return 0xFFFFu;
        default:
            return 0u;
    }
}

bool isSupportedTargetPsm(Psm psm) {
    return supportedChannels(psm) != 0;
}

// ---------------------------------------------------------------------------
// Rectangles
// ---------------------------------------------------------------------------

bool Rect::contains(const Rect& other) const {
    return other.empty() || (x0 <= other.x0 && y0 <= other.y0 && x1 >= other.x1 && y1 >= other.y1);
}

bool Rect::overlaps(const Rect& other) const {
    return !empty() && !other.empty() && x0 < other.x1 && other.x0 < x1 && y0 < other.y1 && other.y0 < y1;
}

Rect Rect::intersect(const Rect& other) const {
    Rect result{std::max(x0, other.x0), std::max(y0, other.y0), std::min(x1, other.x1), std::min(y1, other.y1)};
    if (result.empty()) return Rect{};
    return result;
}

Rect rectFromOrigin(std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height) {
    return Rect{x, y, x + width, y + height};
}

Rect boundingBox(const Rect& first, const Rect& second) {
    if (first.empty()) return second.empty() ? Rect{} : second;
    if (second.empty()) return first;
    return Rect{std::min(first.x0, second.x0), std::min(first.y0, second.y0),
                std::max(first.x1, second.x1), std::max(first.y1, second.y1)};
}

std::vector<Rect> subtractRect(const Rect& area, const Rect& cut) {
    std::vector<Rect> pieces;
    if (area.empty()) return pieces;
    const Rect overlap = area.intersect(cut);
    if (overlap.empty()) {
        pieces.push_back(area);
        return pieces;
    }
    if (area.y0 < overlap.y0) pieces.push_back(Rect{area.x0, area.y0, area.x1, overlap.y0});
    if (overlap.y1 < area.y1) pieces.push_back(Rect{area.x0, overlap.y1, area.x1, area.y1});
    if (area.x0 < overlap.x0) pieces.push_back(Rect{area.x0, overlap.y0, overlap.x0, overlap.y1});
    if (overlap.x1 < area.x1) pieces.push_back(Rect{overlap.x1, overlap.y0, area.x1, overlap.y1});
    return pieces;
}

std::vector<Rect> subtractFromAll(const std::vector<Rect>& areas, const Rect& cut) {
    std::vector<Rect> result;
    for (const Rect& area : areas) {
        const std::vector<Rect> pieces = subtractRect(area, cut);
        result.insert(result.end(), pieces.begin(), pieces.end());
    }
    return result;
}

// ---------------------------------------------------------------------------
// Block ranges
// ---------------------------------------------------------------------------

std::int64_t unwrappedBlock(const BufferView& view, std::uint32_t x, std::uint32_t y) {
    const memory::PsmInfo& info = memory::psmInfo(view.psm);
    // The page-local slot comes from the M2 library with base 0; for Z
    // formats its block number already carries the in-page XOR.
    const std::uint32_t storedBlock =
        memory::pageLocalSlot(view.psm, x % info.pageWidth, y % info.pageHeight).byteOffset / memory::kBlockBytes;
    const std::int64_t tableBlock = info.zBlockXorApplies ? static_cast<std::int64_t>(storedBlock ^ kZBlockXor)
                                                          : static_cast<std::int64_t>(storedBlock);
    const std::int64_t absolute = unwrappedPageBase(view, x, y) + tableBlock;
    return info.zBlockXorApplies ? (absolute ^ static_cast<std::int64_t>(kZBlockXor)) : absolute;
}

BlockRange blockRangeOf(const BufferView& view, const Rect& rect) {
    BlockRange range{};
    if (rect.empty()) {
        range.first = 0;
        range.last = -1;
        return range;
    }
    const memory::PsmInfo& info = memory::psmInfo(view.psm);
    const std::uint32_t lastX = static_cast<std::uint32_t>(rect.x1 - 1);
    const std::uint32_t lastY = static_cast<std::uint32_t>(rect.y1 - 1);
    range.first = unwrappedBlock(view, static_cast<std::uint32_t>(rect.x0), static_cast<std::uint32_t>(rect.y0));
    range.last = unwrappedBlock(view, lastX, lastY);

    const bool alignedRight = rect.x1 % static_cast<std::int32_t>(info.pageWidth) == 0;
    const bool alignedBottom = rect.y1 % static_cast<std::int32_t>(info.pageHeight) == 0;
    if (alignedRight && alignedBottom) {
        // Z layouts are not sequential inside a page: take the whole page.
        range.last = unwrappedPageBase(view, lastX, lastY) + kBlocksPerPage - 1;
    }
    if (range.last < range.first && range.first - range.last < kBlocksPerPage) {
        std::swap(range.first, range.last);
    }
    return range;
}

bool rangesOverlap(const BlockRange& first, const BlockRange& second) {
    if (!first.valid() || !second.valid()) return false;
    BlockRange a = first;
    BlockRange b = second;
    if (a.last >= kWrapBlocks && b.last < a.first) {
        b.first += kWrapBlocks;
        b.last += kWrapBlocks;
    } else if (b.last >= kWrapBlocks && a.last < b.first) {
        a.first += kWrapBlocks;
        a.last += kWrapBlocks;
    }
    return std::max(a.first, b.first) <= std::min(a.last, b.last);
}

bool rangeInside(const BlockRange& inner, const BlockRange& outer) {
    if (!inner.valid() || !outer.valid()) return false;
    return outer.first <= inner.first && inner.last <= outer.last;
}

std::vector<std::uint64_t> blockBitsOf(const BufferView& view, const Rect& rect) {
    std::vector<std::uint64_t> bits(kWrapBlocks / 64, 0);
    if (rect.empty()) return bits;
    for (std::int32_t y = rect.y0; y < rect.y1; ++y) {
        for (std::int32_t x = rect.x0; x < rect.x1; ++x) {
            const std::uint32_t wrappedX = static_cast<std::uint32_t>(x) & kTransferWrapMask;
            const std::uint32_t wrappedY = static_cast<std::uint32_t>(y) & kTransferWrapMask;
            const std::uint32_t block = memory::pixelSlot(view, wrappedX, wrappedY).byteOffset / memory::kBlockBytes;
            bits[block / 64] |= std::uint64_t{1} << (block % 64);
        }
    }
    return bits;
}

bool blockBitsTest(const std::vector<std::uint64_t>& bits, std::uint32_t block) {
    return (bits[block / 64] >> (block % 64)) & 1u;
}

bool blockBitsAny(const std::vector<std::uint64_t>& first, const std::vector<std::uint64_t>& second) {
    for (std::size_t i = 0; i < first.size() && i < second.size(); ++i) {
        if (first[i] & second[i]) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Re-expression and translation
// ---------------------------------------------------------------------------

Rect expressDirtyRect(const Rect& rect, Psm writer, Psm target) {
    if (rect.empty()) return Rect{};
    const memory::PsmInfo& writerInfo = memory::psmInfo(writer);
    const memory::PsmInfo& targetInfo = memory::psmInfo(target);
    const std::int64_t wbw = writerInfo.blockWidth;
    const std::int64_t wbh = writerInfo.blockHeight;
    const std::int64_t tbw = targetInfo.blockWidth;
    const std::int64_t tbh = targetInfo.blockHeight;

    // Scale to target coordinates: start rounds down, end rounds up.
    std::int64_t x0 = floorDiv(static_cast<std::int64_t>(rect.x0) * tbw, wbw);
    std::int64_t x1 = ceilDiv(static_cast<std::int64_t>(rect.x1) * tbw, wbw);
    std::int64_t y0 = floorDiv(static_cast<std::int64_t>(rect.y0) * tbh, wbh);
    std::int64_t y1 = ceilDiv(static_cast<std::int64_t>(rect.y1) * tbh, wbh);

    // Align outward to the target block size.
    x0 = floorDiv(x0, tbw) * tbw;
    x1 = ceilDiv(x1, tbw) * tbw;
    y0 = floorDiv(y0, tbh) * tbh;
    y1 = ceilDiv(y1, tbh) * tbh;
    return Rect{static_cast<std::int32_t>(x0), static_cast<std::int32_t>(y0),
                static_cast<std::int32_t>(x1), static_cast<std::int32_t>(y1)};
}

std::optional<Rect> translateRect(const BufferView& source, const Rect& rect, const BufferView& target) {
    if (rect.empty()) return std::nullopt;
    const memory::PsmInfo& sourceInfo = memory::psmInfo(source.psm);
    const memory::PsmInfo& targetInfo = memory::psmInfo(target.psm);
    if (sourceInfo.pageWidth != targetInfo.pageWidth || sourceInfo.pageHeight != targetInfo.pageHeight) {
        return std::nullopt;
    }
    if (source.bufferWidth != target.bufferWidth) return std::nullopt;
    if (source.psm != target.psm) {
        // Different bits per pixel: the rectangle must be page aligned.
        if (rect.x0 % sourceInfo.pageWidth || rect.y0 % sourceInfo.pageHeight ||
            rect.x1 % sourceInfo.pageWidth || rect.y1 % sourceInfo.pageHeight) {
            return std::nullopt;
        }
    }
    if (source.basePointer < target.basePointer) return std::nullopt;  // negative offsets: deferred
    const std::uint32_t delta = source.basePointer - target.basePointer;
    if (delta % memory::kBlocksPerPage != 0) return std::nullopt;  // block offsets: deferred

    const std::uint32_t pageOffset = delta / memory::kBlocksPerPage;
    const std::uint32_t pagesPerRow = memory::pagesPerRow(target.bufferWidth, target.psm);
    const std::int32_t horizontal = static_cast<std::int32_t>((pageOffset % pagesPerRow) * targetInfo.pageWidth);
    const std::int32_t vertical = static_cast<std::int32_t>((pageOffset / pagesPerRow) * targetInfo.pageHeight);
    return Rect{rect.x0 + horizontal, rect.y0 + vertical, rect.x1 + horizontal, rect.y1 + vertical};
}

// ---------------------------------------------------------------------------
// Source regions from CLAMP
// ---------------------------------------------------------------------------

std::optional<AxisRegion> regionFromClamp(ClampMode mode, std::uint32_t minU, std::uint32_t maxU, std::uint32_t tw) {
    const std::uint64_t textureWidth = std::uint64_t{1} << (tw & 31u);
    if (mode == ClampMode::RegionClamp) {
        if (maxU < minU) return std::nullopt;
        const std::uint64_t width = static_cast<std::uint64_t>(maxU) - minU + 1;
        if (width < textureWidth || maxU >= textureWidth) return AxisRegion{minU, maxU + 1};
        return std::nullopt;
    }
    // REGION_REPEAT: the region starts at MAXU and wraps through the mask.
    if (minU == 0) return std::nullopt;
    const std::uint32_t combined = minU | maxU;
    const std::uint64_t width = static_cast<std::uint64_t>(combined - maxU) + 1;
    if (width < textureWidth || (maxU != 0 && width <= textureWidth)) return AxisRegion{maxU, combined + 1};
    return std::nullopt;
}

}  // namespace coregs::cache
