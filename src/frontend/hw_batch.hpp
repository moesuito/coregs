// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// HWR batch model and batch analysis: vertex trace, ST rounding, overlap and
// draw lists (chapter 02.4, 02.9, 02.10).
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "hw_clut.hpp"
#include "hw_gs_counters.hpp"
#include "hw_gs_registers.hpp"
#include "hw_gs_types.hpp"

namespace coregs {

// One kicked vertex after XYOFFSET and with the latched RGBAQ, ST/UV and FOG.
struct Vertex {
    int32_t x4 = 0;       // 12.4 window coordinates (XYOFFSET applied)
    int32_t y4 = 0;
    uint32_t z = 0;
    uint8_t r = 0, g = 0, b = 0, a = 0;
    uint8_t fog = 0;
    float q = 1.0f;
    float s = 0.0f;
    float t = 0.0f;
    uint32_t u = 0;       // 10.4 texels (UV addressing)
    uint32_t v = 0;
};

// Inclusive pixel rectangle. Empty when x0 > x1 or y0 > y1.
struct PixelRect {
    int32_t x0 = 0, y0 = 0, x1 = -1, y1 = -1;
    bool Empty() const { return x0 > x1 || y0 > y1; }
};

PixelRect IntersectRects(const PixelRect& a, const PixelRect& b);
PixelRect UnionRects(const PixelRect& a, const PixelRect& b);
bool RectsOverlap(const PixelRect& a, const PixelRect& b);

struct PrimRecord {
    PrimType type = PrimType::Point;
    uint32_t first = 0;   // index into Batch::vertices
    uint32_t count = 0;   // vertices of this primitive (1, 2 or 3)
    PixelRect rect;       // pixel coverage (clipped to scissor when drawn)
};

// Register snapshot taken with the first primitive of a batch (chapter 02.5).
struct BatchState {
    EnvironmentRegs env;
    ContextRegs ctx;
    uint32_t contextIndex = 0;
    uint64_t effectivePrim = 0;
    uint32_t clutGeneration = 0;
};

// Chapter 02.9 facts over all vertices of a batch.
struct VertexTrace {
    int32_t xMin = 0, xMax = -1, yMin = 0, yMax = -1;  // pixels
    uint32_t zMin = 0, zMax = 0;
    uint8_t fogMin = 0, fogMax = 0;
    float texelUMin = 0.0f, texelUMax = 0.0f;
    float texelVMin = 0.0f, texelVMax = 0.0f;
    bool texelNaN = false;
    float qMin = 0.0f, qMax = 0.0f;
    uint8_t rgbaMin[4] = {0, 0, 0, 0};
    uint8_t rgbaMax[4] = {0, 0, 0, 0};
    // Bit set = constant over the batch: 0..3 R G B A, 4 x, 5 y, 6 z, 7 f, 8 s, 9 t, 10 q.
    uint32_t constantMask = 0;
    uint8_t vertexAlphaMin = 0, vertexAlphaMax = 0;
    uint8_t fragmentAlphaMin = 0, fragmentAlphaMax = 0;
    enum class Filter : uint8_t { Nearest, Linear, Either };
    Filter filter = Filter::Nearest;
    bool unsafeQ = false;
};

// TASK-168: a palette segment. A CLUT load that only changes the palette of the pending batch's indexed texture (the
// texture, CPSM and CSA stay the same) no longer ends the batch when the renderer allows it (FrontEnd::
// SetPaletteSegmentHook): the batch goes on and the primitives after the load form a new segment, which keeps a copy of
// the CLUT buffer the load produced. Segment s covers primitives [firstPrim, next segment's firstPrim) and their
// vertices [firstVertex, next segment's firstVertex); the last one runs to the end of the batch.
struct PaletteSegment {
    uint32_t firstPrim = 0;       // index into Batch::prims
    uint32_t firstVertex = 0;     // index into Batch::vertices
    uint32_t clutGeneration = 0;  // ClutUnit::Generation() of the buffer below
    std::array<uint16_t, clut::kClutHalfwords> clut{};  // the CLUT buffer these primitives see
};

// Segments one batch may hold; a load past them ends the batch as before. A segment is 1 KiB, and a renderer gives each
// segment a palette row that must stay put until the batch is drawn (core 2 has 1024 rows).
constexpr uint32_t kMaxPaletteSegments = 256;

struct Batch {
    BatchState state;
    PrimClass cls = PrimClass::Invalid;
    std::vector<Vertex> vertices;
    std::vector<PrimRecord> prims;
    PixelRect drawRect;
    FlushReason reason = FlushReason::Count;
    VertexTrace trace;
    // TASK-168: empty = one palette, the CLUT unit's buffer when the batch is drawn (the CLUT cannot change under a
    // pending indexed batch without a segment or a flush). Otherwise every primitive's palette is its segment's copy,
    // the first segment starting at primitive 0.
    std::vector<PaletteSegment> paletteSegments;

    bool Empty() const { return prims.empty(); }
    void Clear();
};

// TASK-168: the number of palettes of a batch (1 when it has no segments).
uint32_t PaletteSegmentCount(const Batch& batch);
// One past the last primitive of segment `segment` (the whole batch without segments).
uint32_t PaletteSegmentEndPrim(const Batch& batch, uint32_t segment);
// The CLUT buffer segment `segment`'s primitives see: its copy, or `live`'s buffer without segments.
const uint16_t* PaletteSegmentClut(const Batch& batch, uint32_t segment, const clut::ClutUnit& live);
// The palette of segment `segment` for the batch's TEX0 (PSM, CPSM, CSA) and TEXA (chapter 07.5).
clut::Palette DecodeSegmentPalette(const Batch& batch, uint32_t segment, const clut::ClutUnit& live);
// A batch of segment `segment`'s primitives alone, with that segment as its only (explicit) palette: what a CLUT-load
// flush would have emitted. For consumers that take one palette per batch.
Batch PaletteSegmentBatch(const Batch& batch, uint32_t segment);

// Chapter 02.4: the number of mantissa bits to clear for S (and T) given the
// unbiased exponents of S (or T) and Q. Vector: S = 0.5, Q = 1.0 gives 10.
uint32_t StMantissaBits(int exponentComponent, int exponentQ);

// Unbiased exponent of a float (zero and subnormals report -127).
int UnbiasedExponent(float value);

// Clears the lowest `bits` mantissa bits (truncation towards zero).
float ClearMantissaLowBits(float value, uint32_t bits);

// Chapter 02.4 applied to the batch when it is textured with STQ coordinates
// and is a sprite batch or has constant Z. constantZ is the caller's check.
// Returns true when rounding was applied.
bool ApplyStRounding(Batch& batch, bool constantZ);
// The same on primitives [firstPrim, endPrim) only, constantZ being their own check (TASK-168: per palette segment, as
// the batch a CLUT-load flush would have ended there).
bool ApplyStRoundingToPrims(Batch& batch, bool constantZ, uint32_t firstPrim, uint32_t endPrim);

// Texel coordinates of one vertex (chapter 01.5). Returns false when a result is NaN.
bool TexelCoordinates(const Vertex& vertex, bool fst, uint32_t twLog2, uint32_t thLog2,
                      float& u, float& v);

// Chapter 02.9 vertex trace. palette may be null (indexed alpha is then the full range).
VertexTrace ComputeVertexTrace(const Batch& batch, const clut::Palette* palette);

// Chapter 02.10 classification of whether primitives of a batch overlap.
enum class OverlapKind : uint8_t { None, Possible, Unknown };
OverlapKind ClassifyOverlap(const Batch& batch);

// Chapter 02.10 draw list for sprite batches: groups in draw order with bounds
// expanded by one pixel for bilinear sampling.
struct DrawGroup {
    PixelRect bounds;
    uint32_t primitiveCount = 0;
};
std::vector<DrawGroup> BuildSpriteDrawList(const Batch& batch);

// Conservative block footprint of a buffer (chapter 02 transfer and flush
// overlap checks): widthUnits are 64-pixel units, height in pixels.
uint32_t FootprintBlocks(uint32_t widthUnits, uint32_t heightPixels);

// True when two block ranges (with 16384-block wrap) share a block.
bool BlockRangesOverlap(uint32_t begin0, uint32_t count0, uint32_t begin1, uint32_t count1);
// The block-by-block definition the constant-time version must equal (fixture reference, IN-360).
bool BlockRangesOverlapByWalk(uint32_t begin0, uint32_t count0, uint32_t begin1, uint32_t count1);

}  // namespace coregs
