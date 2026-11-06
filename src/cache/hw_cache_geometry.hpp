// SPDX-License-Identifier: MIT
// Copyright 2026 GT4Recomp contributors
//
// HWR surface cache geometry (TASK-027, milestone M7).
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters
// 04.1 (geometry primitives), 05.1 (channel validity) and 05.5 (layouts).
// Clean room: no code from other GS renderers.
//
// Vulkan-free. Depends only on the memory model (src/memory).

#pragma once

#include "hw_memory.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace coregs::cache {

using memory::BufferView;
using memory::Psm;

// Channel validity bits (chapter 05.1). 32-bit colour formats split alpha into
// a low and a high nibble; 16-bit formats have one channel; 24-bit formats
// have RGB only.
inline constexpr std::uint8_t kChanRgb = 0x1;
inline constexpr std::uint8_t kChanAlphaLo = 0x2;
inline constexpr std::uint8_t kChanAlphaHi = 0x4;
inline constexpr std::uint8_t kChanAll = 0x7;

// Channels that a format can hold at all.
std::uint8_t supportedChannels(Psm psm);

// Word bits of a stored pixel covered by a channel mask (chapter 10.4 channel
// mask). Zero for an empty mask or an unsupported format.
std::uint32_t channelWordMask(Psm psm, std::uint8_t channels);

// Bits of a stored pixel (32, 24 or 16 bit).
std::uint32_t storageWordMask(Psm psm);

// Colour formats the HWR surface cache can hold as targets (chapter 05 scope
// of M7): CT32, CT24, CT16 and CT16S.
bool isSupportedTargetPsm(Psm psm);

// ---------------------------------------------------------------------------
// Rectangles (pixel coordinates of one buffer, half-open)
// ---------------------------------------------------------------------------

struct Rect {
    std::int32_t x0 = 0;
    std::int32_t y0 = 0;
    std::int32_t x1 = 0;
    std::int32_t y1 = 0;

    bool empty() const { return x1 <= x0 || y1 <= y0; }
    std::int32_t width() const { return empty() ? 0 : x1 - x0; }
    std::int32_t height() const { return empty() ? 0 : y1 - y0; }
    std::int64_t area() const { return static_cast<std::int64_t>(width()) * height(); }
    bool contains(const Rect& other) const;
    bool overlaps(const Rect& other) const;
    Rect intersect(const Rect& other) const;  // empty Rect{} when disjoint
    bool operator==(const Rect& other) const = default;
};

Rect rectFromOrigin(std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height);

// Smallest rectangle containing both (empty operands are ignored).
Rect boundingBox(const Rect& first, const Rect& second);

// a minus cut as at most four disjoint rectangles.
std::vector<Rect> subtractRect(const Rect& area, const Rect& cut);
std::vector<Rect> subtractFromAll(const std::vector<Rect>& areas, const Rect& cut);

// ---------------------------------------------------------------------------
// Block ranges (chapter 04.1)
// ---------------------------------------------------------------------------

// Unwrapped block interval of a rectangle. Values are not reduced modulo
// 16384, so ranges that cross the end of memory compare correctly.
struct BlockRange {
    std::int64_t first = 0;
    std::int64_t last = -1;  // last < first means "no blocks"

    bool valid() const { return last >= first; }
};

// Unwrapped block number of the pixel (x, y) of a buffer: BP + page * 32 +
// in-page block, with the Z block XOR applied after adding BP (rule of 01.3).
std::int64_t unwrappedBlock(const BufferView& view, std::uint32_t x, std::uint32_t y);

// Block range of a rectangle: first block of the top-left pixel to last block
// of the bottom-right pixel; page-aligned right and bottom edges round the end
// up to the last block of the page; an inverted pair within 32 blocks is
// swapped.
BlockRange blockRangeOf(const BufferView& view, const Rect& rect);

// Overlap with 16384-block wrap (lift rule of 04.1).
bool rangesOverlap(const BlockRange& first, const BlockRange& second);

// Inside: the inner range lies within the outer one.
bool rangeInside(const BlockRange& inner, const BlockRange& outer);

// Set of wrapped block numbers (0..16383) covered by the pixels of a rectangle
// of a buffer. Coordinates are wrapped at 2048 as transfers are (chapter 02.7).
// Used for exact footprint decisions.
std::vector<std::uint64_t> blockBitsOf(const BufferView& view, const Rect& rect);
bool blockBitsTest(const std::vector<std::uint64_t>& bits, std::uint32_t block);
bool blockBitsAny(const std::vector<std::uint64_t>& first, const std::vector<std::uint64_t>& second);

// ---------------------------------------------------------------------------
// Re-expression and translation (chapter 04.1)
// ---------------------------------------------------------------------------

// Dirty rectangle re-expression: each coordinate is scaled by the ratio of the
// target block size to the writer block size, then aligned outward to the
// target block size. Vector: CT32 (8,0)-(16,8) on a CT16 target gives
// (16,0)-(32,8).
Rect expressDirtyRect(const Rect& rect, Psm writer, Psm target);

// Translation of a rectangle of `source` into the coordinates of `target` when
// both share a block layout (same page size, same buffer width) and the base
// difference is a whole number of pages (non-negative). Different formats also
// require a page-aligned rectangle. Other cases return nullopt (deferred:
// surface-offset search and block offsets).
std::optional<Rect> translateRect(const BufferView& source, const Rect& rect, const BufferView& target);

// ---------------------------------------------------------------------------
// Source regions from CLAMP (chapter 04.3)
// ---------------------------------------------------------------------------

enum class ClampMode : std::uint8_t {
    RegionClamp,
    RegionRepeat,
};

// Half-open texel interval of one axis.
struct AxisRegion {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
};

// Region of one texture axis implied by CLAMP, for TW bits of texture width
// (chapter 04.3). nullopt when the axis has no region. A region may extend past
// 2^TW ("fixed TEX0").
std::optional<AxisRegion> regionFromClamp(ClampMode mode, std::uint32_t minU, std::uint32_t maxU, std::uint32_t tw);

}  // namespace coregs::cache
