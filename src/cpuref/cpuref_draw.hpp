// SPDX-License-Identifier: MIT
//
// HWR CPU reference rasterizer: points, lines, triangles and sprites drawn into the shadow
// local memory with the specification's coverage rules, attribute interpolation, texture
// sampling and the per-pixel pipeline of the pixel library (M4 and M5 of the plan).
// Copyright 2026 GT4Recomp contributors.
// Written from docs/hw-renderer-spec (clean room, decision 0069): chapters 01.4, 01.5, 02.9,
// 03.3 to 03.5 and 08. Slow and exact by design: it is the oracle for the GPU work.
#pragma once

#include <cstdint>
#include <vector>

#include "hw_batch.hpp"
#include "hw_clut.hpp"
#include "hw_memory.hpp"

namespace coregs::cpuref {

struct PixelCoord {
    int32_t x = 0;
    int32_t y = 0;
};

struct DrawStats {
    uint64_t batches = 0;
    uint64_t primitives = 0;
    uint64_t primitivesSkippedTexture = 0;  // invalid texture size or format (01.5, counted)
    uint64_t primitivesTextured = 0;
    uint64_t coveredPixels = 0;             // pixels that reached the per-pixel pipeline
    uint64_t framePixelsWritten = 0;
    uint64_t depthPixelsWritten = 0;
    uint64_t blendedPixels = 0;             // pixels that ran the blend stage (ABE = 1)
};

// ---- Coverage (chapter 01.4, 02.4, 03.3) --------------------------------------------
// Window pixel of a 12.4 coordinate (floor), for points and line endpoints.
int32_t WindowPixel(int32_t coordinate4);

// Triangle coverage of pixel (px, py): the pixel centre is the integer position. Edges are
// tested with the top-left rule of 01.4. Returns false for degenerate triangles.
bool TriangleCoversPixel(const Vertex& a, const Vertex& b, const Vertex& c, int32_t px, int32_t py);

// Sprite coverage on one axis: pixel n is covered when min <= n < max, with min and max
// the 12.4 coordinates of the two vertices (ordered as given).
bool SpriteCoversPixel(int32_t firstCoordinate4, int32_t secondCoordinate4, int32_t n);

// Pixels of a line: integer Bresenham between the window pixels of the endpoints, both
// endpoints included (Hypothesis IN-301).
std::vector<PixelCoord> LinePixels(int32_t x0_4, int32_t y0_4, int32_t x1_4, int32_t y1_4);

// ---- Drawing --------------------------------------------------------------------------
// Draws every primitive of a batch in order into the local memory. The batch's state
// snapshot supplies the registers; the palette comes from the CLUT unit.
void DrawBatch(const Batch& batch, memory::LocalMemory& local, const clut::ClutUnit& clut,
               DrawStats& stats);

}  // namespace coregs::cpuref
