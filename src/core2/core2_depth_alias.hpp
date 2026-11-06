// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec/CORE2-SPEC.md and CORE2-DESIGN.md (TASK-027 core rewrite), clean room
// (decision 0070).
//
// IN-535: a depth fill that lands in the memory of a colour buffer at another base pointer. On the GS a depth buffer
// is GS memory like any other, so its words replace the colour words stored at the same addresses; core 2 keeps
// depth and colour in separate images, so such a write has to be carried into the colour target. The game clears its
// 256 x 256 environment map with two 64-pixel-wide sprites at x 0 and 128: their colour writes cover columns 0-63
// and 128-191, and their depth writes, whose buffer starts one page later (FBP + 32 blocks), columns 64-127 and
// 192-255 (alpha 0x80 from Z 0x80000000).

#pragma once

#include <cstdint>
#include <vector>

#include "hw_batch.hpp"
#include "hw_memory.hpp"

namespace coregs::core2 {

// The pixels a GS sprite covers, from its corners in 12.4 window units (XYOFFSET already applied): pixel k is covered
// when x0 <= k < x1 (the same in y), as core 2's rasteriser covers them at 1x. Clipped to the inclusive `scissor`.
PixelRect SpritePixels(std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1, const PixelRect &scissor);

// The pixels of 32-bit colour buffer `colour` (CT32 or CT24) whose words a write of every pixel of `written` in 32-bit
// depth buffer `depth` (Z32 or Z24) replaces, as rectangles in the colour buffer's coordinates, inside
// (0, 0)-(colourWidth - 1, colourHeight - 1). The two layouts share their 8 x 8 block contents and differ in where
// each block sits, so the write is carried block by block; a depth block counts only when `written` covers all of it
// (a partly covered one is left out, never guessed). Colour pixels inside any rectangle of `keep` (the same draw's own
// colour write, which reaches memory after the depth write that aliases it) are left out. Adjacent rectangles are
// merged. Empty for other formats, or when the depth buffer is the colour buffer (same base pointer: not handled here).
std::vector<PixelRect> DepthBlocksInColour(const memory::BufferView &depth, const PixelRect &written,
                                           const memory::BufferView &colour, std::uint32_t colourWidth,
                                           std::uint32_t colourHeight, const std::vector<PixelRect> &keep);

}  // namespace coregs::core2
