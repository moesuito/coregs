// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec/CORE2-SPEC.md and CORE2-DESIGN.md (TASK-027 core rewrite), clean room
// (decision 0070).
//
// CTest hw_core2_depth_alias (IN-535): the colour pixels a 32-bit depth fill replaces, against memory::pixelSlot
// pixel by pixel. GT4's environment map clear (colour at 11200, depth one page later, 256 wide, two 64-wide sprites)
// gives columns 64-127 and 192-255; random bases, widths, rectangles and kept areas give exactly the colour pixels
// whose words lie in a wholly written depth block, outside every kept rectangle, each once.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "core2/core2_depth_alias.hpp"
#include "hw_memory.hpp"

namespace {

using coregs::PixelRect;
using coregs::core2::DepthBlocksInColour;
using coregs::core2::SpritePixels;
using namespace coregs::memory;

int failures = 0;
int cases = 0;

void Expect(bool condition, const char *what) {
    if (!condition) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

bool SameRect(const PixelRect &a, const PixelRect &b) {
    return a.x0 == b.x0 && a.y0 == b.y0 && a.x1 == b.x1 && a.y1 == b.y1;
}

std::uint64_t NextRandom(std::uint64_t &state) {  // splitmix64
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

std::uint32_t Below(std::uint64_t &state, std::uint32_t limit) {
    return static_cast<std::uint32_t>(NextRandom(state) % limit);
}

bool Inside(const PixelRect &r, std::int32_t x, std::int32_t y) {
    return !r.Empty() && x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1;
}

void CheckSprites() {
    const PixelRect screen{0, 0, 639, 447};
    Expect(SameRect(SpritePixels(0, 0, 640 * 16, 448 * 16, screen), screen), "a whole-screen sprite covers the screen");
    Expect(SameRect(SpritePixels(-16, -16, 192 * 16, 256 * 16, PixelRect{0, 0, 191, 255}), PixelRect{0, 0, 191, 255}),
           "the environment map clear covers its scissor");
    Expect(SameRect(SpritePixels(8, 8, 24, 40, screen), PixelRect{1, 1, 1, 2}), "pixel k is covered when x0 <= k < x1");
    Expect(SameRect(SpritePixels(24, 40, 8, 8, screen), PixelRect{1, 1, 1, 2}), "corners in either order");
    Expect(SpritePixels(16, 16, 16, 32, screen).Empty(), "a zero-width sprite covers nothing");
    ++cases;
}

void CheckEnvironmentMapClear() {
    // GT4 (Spec II and v2.00 races): FRAME 11200 / 4 CT32 and ZBUF 11232 Z32, two sprites over (0,0)-(63,255) and
    // (128,0)-(191,255), the colour kept where the same sprites write it.
    const BufferView depth{11232, 4, Psm::Z32}, colour{11200, 4, Psm::Ct32};
    const std::vector<PixelRect> sprites{PixelRect{0, 0, 63, 255}, PixelRect{128, 0, 191, 255}};
    std::vector<PixelRect> rects;
    for (const PixelRect &sprite : sprites) {
        for (const PixelRect &r : DepthBlocksInColour(depth, sprite, colour, 256, 256, sprites)) rects.push_back(r);
    }
    Expect(rects.size() == 2 && SameRect(rects[0], PixelRect{64, 0, 127, 255}) && SameRect(rects[1], PixelRect{192, 0, 255, 255}),
           "the environment map clear writes its depth into columns 64-127 and 192-255");
    const auto wide = DepthBlocksInColour(depth, PixelRect{0, 0, 191, 255}, colour, 256, 256,
                                          std::vector<PixelRect>{PixelRect{0, 0, 191, 255}});
    Expect(wide.size() == 1 && SameRect(wide[0], PixelRect{192, 0, 255, 255}), "a kept rectangle hides the depth under it");
    const std::vector<PixelRect> none;
    const auto all = DepthBlocksInColour(BufferView{11232, 4, Psm::Z24}, PixelRect{0, 0, 191, 255}, colour, 256, 256, none);
    Expect(all.size() == 1 && SameRect(all[0], PixelRect{64, 0, 255, 255}), "Z24 lands where Z32 does; nothing kept");
    Expect(DepthBlocksInColour(BufferView{11200, 4, Psm::Z32}, PixelRect{0, 0, 191, 255}, colour, 256, 256, none).empty(),
           "the same base pointer is not handled here");
    Expect(DepthBlocksInColour(BufferView{11232, 4, Psm::Z16}, PixelRect{0, 0, 191, 255}, colour, 256, 256, none).empty(),
           "16-bit depth is not handled");
    cases += 5;
}

// Every colour pixel whose word a wholly written depth block holds, outside `keep`, is in exactly one rectangle, and
// no other pixel is.
void CheckRandom(std::uint64_t &state) {
    const Psm depthPsm = Below(state, 2) == 0 ? Psm::Z32 : Psm::Z24;
    const BufferView depth{Below(state, kBlocksTotal), 1 + Below(state, 6), depthPsm};
    BufferView colour{Below(state, kBlocksTotal), 1 + Below(state, 6), Below(state, 4) == 0 ? Psm::Ct24 : Psm::Ct32};
    if (Below(state, 2) == 0) colour.basePointer = (depth.basePointer + kBlocksTotal - 32u * (1 + Below(state, 8))) % kBlocksTotal;
    if (colour.basePointer == depth.basePointer) colour.basePointer = (colour.basePointer + 1) % kBlocksTotal;
    const std::uint32_t colourWidth = colour.bufferWidth * 64u, colourHeight = 32u * (1 + Below(state, 8));
    const std::int32_t depthWidth = static_cast<std::int32_t>(depth.bufferWidth * 64u);
    PixelRect written{static_cast<std::int32_t>(Below(state, static_cast<std::uint32_t>(depthWidth))),
                      static_cast<std::int32_t>(Below(state, 256)), 0, 0};
    written.x1 = written.x0 + static_cast<std::int32_t>(Below(state, static_cast<std::uint32_t>(depthWidth - written.x0)));
    written.y1 = written.y0 + static_cast<std::int32_t>(Below(state, 256));
    std::vector<PixelRect> keep(Below(state, 3));
    for (PixelRect &k : keep) {
        k.x0 = static_cast<std::int32_t>(Below(state, colourWidth));
        k.y0 = static_cast<std::int32_t>(Below(state, colourHeight));
        k.x1 = k.x0 + static_cast<std::int32_t>(Below(state, colourWidth));
        k.y1 = k.y0 + static_cast<std::int32_t>(Below(state, colourHeight));
    }

    std::vector<std::uint8_t> depthWord(kLocalMemoryBytes / 4u, 0);
    for (std::int32_t by = (written.y0 + 7) / 8; by * 8 + 7 <= written.y1; ++by) {
        for (std::int32_t bx = (written.x0 + 7) / 8; bx * 8 + 7 <= written.x1; ++bx) {
            for (std::int32_t y = by * 8; y < by * 8 + 8; ++y) {
                for (std::int32_t x = bx * 8; x < bx * 8 + 8; ++x) {
                    depthWord[pixelSlot(depth, static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y)).byteOffset / 4u] = 1;
                }
            }
        }
    }
    const auto rects = DepthBlocksInColour(depth, written, colour, colourWidth, colourHeight, keep);
    std::vector<std::uint8_t> hits(static_cast<std::size_t>(colourWidth) * colourHeight, 0);
    for (const PixelRect &r : rects) {
        for (std::int32_t y = r.y0; y <= r.y1; ++y) {
            for (std::int32_t x = r.x0; x <= r.x1; ++x) {
                if (x < 0 || y < 0 || x >= static_cast<std::int32_t>(colourWidth) || y >= static_cast<std::int32_t>(colourHeight)) {
                    Expect(false, "a rectangle leaves the colour buffer");
                    return;
                }
                ++hits[static_cast<std::size_t>(y) * colourWidth + static_cast<std::size_t>(x)];
            }
        }
    }
    for (std::uint32_t y = 0; y < colourHeight; ++y) {
        for (std::uint32_t x = 0; x < colourWidth; ++x) {
            const bool aliased = depthWord[pixelSlot(colour, x, y).byteOffset / 4u] != 0;
            bool kept = false;
            for (const PixelRect &k : keep) kept = kept || Inside(k, static_cast<std::int32_t>(x), static_cast<std::int32_t>(y));
            const bool expected = aliased && !kept;
            const std::uint8_t got = hits[static_cast<std::size_t>(y) * colourWidth + x];
            if (got != (expected ? 1 : 0)) {
                std::printf("FAIL: depth %u/%u/%02x written %d,%d..%d,%d colour %u/%u/%02x pixel %u,%u expected %d got %u\n",
                            depth.basePointer, depth.bufferWidth, static_cast<unsigned>(depth.psm), written.x0, written.y0,
                            written.x1, written.y1, colour.basePointer, colour.bufferWidth,
                            static_cast<unsigned>(colour.psm), x, y, expected ? 1 : 0, got);
                ++failures;
                return;
            }
        }
    }
    ++cases;
}

}  // namespace

int main() {
    CheckSprites();
    CheckEnvironmentMapClear();
    std::uint64_t state = 0x1A5D0535ull;
    for (int i = 0; i < 300; ++i) CheckRandom(state);
    std::printf("hw_core2_depth_alias: %d cases, %d failures\n", cases, failures);
    return failures == 0 ? 0 : 1;
}
