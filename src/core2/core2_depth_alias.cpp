// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec/CORE2-SPEC.md and CORE2-DESIGN.md (TASK-027 core rewrite), clean room
// (decision 0070).

#include "core2_depth_alias.hpp"

#include <algorithm>
#include <array>

namespace coregs::core2 {
namespace {

using memory::BufferView;
using memory::Psm;

constexpr std::uint32_t kBlockSide = 8;  // 32-bit formats: 8 x 8 pixel blocks, 8 x 4 blocks per 64 x 32 page

std::int32_t CeilDiv16(std::int32_t value) {
    return value >= 0 ? (value + 15) / 16 : -((-value) / 16);
}

// For the 32 blocks of a CT32 page: the block's cell (column + 8 * row of 8 x 8 cells), and whether Z32 stores the
// pixels of a block in the same order as CT32 (the block contents both layouts share; checked once from the tables).
struct Ct32Cells {
    std::array<std::uint8_t, memory::kBlocksPerPage> cellOfBlock{};
    bool sameBlockContents = true;
};

const Ct32Cells &Cells() {
    static const Ct32Cells cells = [] {
        Ct32Cells built;
        for (std::uint32_t cell = 0; cell < memory::kBlocksPerPage; ++cell) {
            const std::uint32_t x = (cell % 8u) * kBlockSide, y = (cell / 8u) * kBlockSide;
            built.cellOfBlock[memory::pageLocalSlot(Psm::Ct32, x, y).byteOffset / memory::kBlockBytes] =
                static_cast<std::uint8_t>(cell);
        }
        for (std::uint32_t y = 0; y < kBlockSide; ++y) {
            for (std::uint32_t x = 0; x < kBlockSide; ++x) {
                const std::uint32_t colour = memory::pageLocalSlot(Psm::Ct32, x, y).byteOffset % memory::kBlockBytes;
                const std::uint32_t depth = memory::pageLocalSlot(Psm::Z32, x, y).byteOffset % memory::kBlockBytes;
                built.sameBlockContents = built.sameBlockContents && colour == depth;
            }
        }
        return built;
    }();
    return cells;
}

// The rectangles of `rect` outside `keep` (at most four), appended to `out`.
void Subtract(const PixelRect &rect, const PixelRect &keep, std::vector<PixelRect> &out) {
    const PixelRect inside = IntersectRects(rect, keep);
    if (inside.Empty()) {
        out.push_back(rect);
        return;
    }
    if (rect.y0 < inside.y0) out.push_back(PixelRect{rect.x0, rect.y0, rect.x1, inside.y0 - 1});
    if (inside.y1 < rect.y1) out.push_back(PixelRect{rect.x0, inside.y1 + 1, rect.x1, rect.y1});
    if (rect.x0 < inside.x0) out.push_back(PixelRect{rect.x0, inside.y0, inside.x0 - 1, inside.y1});
    if (inside.x1 < rect.x1) out.push_back(PixelRect{inside.x1 + 1, inside.y0, rect.x1, inside.y1});
}

// The pixels of `block` outside every rectangle of `keep`, appended to `out`.
void AddOutside(const PixelRect &block, const std::vector<PixelRect> &keep, std::vector<PixelRect> &out) {
    std::vector<PixelRect> left{block}, next;
    for (const PixelRect &k : keep) {
        next.clear();
        for (const PixelRect &r : left) Subtract(r, k, next);
        left.swap(next);
        if (left.empty()) return;
    }
    out.insert(out.end(), left.begin(), left.end());
}

// Merges rectangles that touch along a whole edge: first along rows, then along columns.
void Merge(std::vector<PixelRect> &rects) {
    std::sort(rects.begin(), rects.end(), [](const PixelRect &a, const PixelRect &b) {
        return a.y0 != b.y0 ? a.y0 < b.y0 : a.y1 != b.y1 ? a.y1 < b.y1 : a.x0 < b.x0;
    });
    std::vector<PixelRect> rows;
    for (const PixelRect &r : rects) {
        if (!rows.empty() && rows.back().y0 == r.y0 && rows.back().y1 == r.y1 && rows.back().x1 + 1 == r.x0) {
            rows.back().x1 = r.x1;
        } else {
            rows.push_back(r);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const PixelRect &a, const PixelRect &b) {
        return a.x0 != b.x0 ? a.x0 < b.x0 : a.x1 != b.x1 ? a.x1 < b.x1 : a.y0 < b.y0;
    });
    rects.clear();
    for (const PixelRect &r : rows) {
        if (!rects.empty() && rects.back().x0 == r.x0 && rects.back().x1 == r.x1 && rects.back().y1 + 1 == r.y0) {
            rects.back().y1 = r.y1;
        } else {
            rects.push_back(r);
        }
    }
}

}  // namespace

PixelRect SpritePixels(std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1, const PixelRect &scissor) {
    const PixelRect covered{CeilDiv16(std::min(x0, x1)), CeilDiv16(std::min(y0, y1)), CeilDiv16(std::max(x0, x1)) - 1,
                            CeilDiv16(std::max(y0, y1)) - 1};
    return IntersectRects(covered, scissor);
}

std::vector<PixelRect> DepthBlocksInColour(const BufferView &depth, const PixelRect &written, const BufferView &colour,
                                           std::uint32_t colourWidth, std::uint32_t colourHeight,
                                           const std::vector<PixelRect> &keep) {
    std::vector<PixelRect> out;
    const bool depth32 = depth.psm == Psm::Z32 || depth.psm == Psm::Z24;
    const bool colour32 = colour.psm == Psm::Ct32 || colour.psm == Psm::Ct24;
    if (!depth32 || !colour32 || written.Empty() || depth.basePointer == colour.basePointer || depth.bufferWidth == 0 ||
        colour.bufferWidth == 0 || !Cells().sameBlockContents) {
        return out;
    }
    const std::int32_t depthWidth = static_cast<std::int32_t>(depth.bufferWidth * 64u);
    const PixelRect area = IntersectRects(written, PixelRect{0, 0, std::min(depthWidth, 2048) - 1, 2047});
    if (area.Empty()) return out;
    const std::int32_t side = static_cast<std::int32_t>(kBlockSide);
    for (std::int32_t by = (area.y0 + side - 1) / side; (by + 1) * side - 1 <= area.y1; ++by) {
        for (std::int32_t bx = (area.x0 + side - 1) / side; (bx + 1) * side - 1 <= area.x1; ++bx) {
            const std::uint32_t block = memory::pixelSlot(depth, static_cast<std::uint32_t>(bx * side),
                                                          static_cast<std::uint32_t>(by * side))
                                            .byteOffset /
                                        memory::kBlockBytes;
            const std::uint32_t relative = (block + memory::kBlocksTotal - colour.basePointer % memory::kBlocksTotal) %
                                           memory::kBlocksTotal;
            const std::uint32_t page = relative / memory::kBlocksPerPage;
            const std::uint32_t cell = Cells().cellOfBlock[relative % memory::kBlocksPerPage];
            const std::uint64_t x = std::uint64_t{page % colour.bufferWidth} * 64u + (cell % 8u) * kBlockSide;
            const std::uint64_t y = std::uint64_t{page / colour.bufferWidth} * 32u + (cell / 8u) * kBlockSide;
            if (x + kBlockSide > colourWidth || y + kBlockSide > colourHeight) continue;
            const PixelRect target{static_cast<std::int32_t>(x), static_cast<std::int32_t>(y),
                                   static_cast<std::int32_t>(x) + side - 1, static_cast<std::int32_t>(y) + side - 1};
            AddOutside(target, keep, out);
        }
    }
    Merge(out);
    return out;
}

}  // namespace coregs::core2
