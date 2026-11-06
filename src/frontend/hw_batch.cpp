// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#include "hw_batch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace coregs {
namespace {

// Running min and max over doubles; empty ranges are treated as constant.
struct Range {
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    void Add(double value) {
        if (std::isnan(value)) {
            return;
        }
        min = std::min(min, value);
        max = std::max(max, value);
    }
    bool Constant() const { return min >= max; }
};


uint64_t Bit(uint64_t value, unsigned index) { return (value >> index) & 1u; }

}  // namespace

PixelRect IntersectRects(const PixelRect& a, const PixelRect& b) {
    PixelRect out;
    out.x0 = std::max(a.x0, b.x0);
    out.y0 = std::max(a.y0, b.y0);
    out.x1 = std::min(a.x1, b.x1);
    out.y1 = std::min(a.y1, b.y1);
    return out;
}

PixelRect UnionRects(const PixelRect& a, const PixelRect& b) {
    if (a.Empty()) {
        return b;
    }
    if (b.Empty()) {
        return a;
    }
    PixelRect out;
    out.x0 = std::min(a.x0, b.x0);
    out.y0 = std::min(a.y0, b.y0);
    out.x1 = std::max(a.x1, b.x1);
    out.y1 = std::max(a.y1, b.y1);
    return out;
}

bool RectsOverlap(const PixelRect& a, const PixelRect& b) {
    return !IntersectRects(a, b).Empty();
}

void Batch::Clear() {
    vertices.clear();
    prims.clear();
    drawRect = PixelRect{};
    cls = PrimClass::Invalid;
    trace = VertexTrace{};
    reason = FlushReason::Count;
    paletteSegments.clear();
}

uint32_t PaletteSegmentCount(const Batch& batch) {
    return batch.paletteSegments.empty() ? 1u : static_cast<uint32_t>(batch.paletteSegments.size());
}

uint32_t PaletteSegmentEndPrim(const Batch& batch, uint32_t segment) {
    if (segment + 1u < batch.paletteSegments.size()) {
        return batch.paletteSegments[segment + 1u].firstPrim;
    }
    return static_cast<uint32_t>(batch.prims.size());
}

const uint16_t* PaletteSegmentClut(const Batch& batch, uint32_t segment, const clut::ClutUnit& live) {
    if (batch.paletteSegments.empty()) {
        return live.Halfwords();
    }
    return batch.paletteSegments[std::min<size_t>(segment, batch.paletteSegments.size() - 1u)].clut.data();
}

clut::Palette DecodeSegmentPalette(const Batch& batch, uint32_t segment, const clut::ClutUnit& live) {
    const uint64_t tex0Register = batch.state.ctx.tex0;
    return clut::DecodeHalfwords(PaletteSegmentClut(batch, segment, live), tex0::Psm(tex0Register),
                                 tex0::Cpsm(tex0Register), tex0::Csa(tex0Register), batch.state.env.texa);
}

Batch PaletteSegmentBatch(const Batch& batch, uint32_t segment) {
    Batch part;
    part.state = batch.state;
    part.cls = batch.cls;
    part.reason = batch.reason;
    part.trace = batch.trace;
    const uint32_t firstPrim = batch.paletteSegments.empty() ? 0u : batch.paletteSegments[segment].firstPrim;
    const uint32_t endPrim = PaletteSegmentEndPrim(batch, segment);
    for (uint32_t index = firstPrim; index < endPrim; ++index) {
        PrimRecord record = batch.prims[index];
        const uint32_t source = record.first;
        record.first = static_cast<uint32_t>(part.vertices.size());
        for (uint32_t k = 0; k < record.count; ++k) {
            part.vertices.push_back(batch.vertices[source + k]);
        }
        part.prims.push_back(record);
        part.drawRect = UnionRects(part.drawRect, record.rect);
    }
    if (!batch.paletteSegments.empty()) {
        PaletteSegment only = batch.paletteSegments[segment];
        only.firstPrim = 0;
        only.firstVertex = 0;
        part.state.clutGeneration = only.clutGeneration;
        part.paletteSegments.push_back(only);
    }
    return part;
}

int UnbiasedExponent(float value) {
    const uint32_t bits = FloatToBits(value);
    const int biased = static_cast<int>((bits >> 23) & 0xFFu);
    if (biased == 0) {
        return -127;
    }
    return biased - 127;
}

float ClearMantissaLowBits(float value, uint32_t bits) {
    if (bits == 0u) {
        return value;
    }
    if (bits > 23u) {
        bits = 23u;
    }
    uint32_t raw = FloatToBits(value);
    if ((raw & 0x7F800000u) == 0x7F800000u) {  // inf and NaN are kept
        return value;
    }
    raw &= ~((1u << bits) - 1u);
    return BitsToFloat(raw);
}

uint32_t StMantissaBits(int exponentComponent, int exponentQ) {
    const int extra = std::max(exponentComponent, exponentQ) - exponentComponent;
    const int bits = 9 + extra;
    return static_cast<uint32_t>(std::min(bits, 23));
}

bool ApplyStRounding(Batch& batch, bool constantZ) {
    return ApplyStRoundingToPrims(batch, constantZ, 0u, static_cast<uint32_t>(batch.prims.size()));
}

bool ApplyStRoundingToPrims(Batch& batch, bool constantZ, uint32_t firstPrim, uint32_t endPrim) {
    const uint64_t prim = batch.state.effectivePrim;
    const bool textured = Bit(prim, primbit::kTme) != 0;
    const bool fst = Bit(prim, primbit::kFst) != 0;
    if (!textured || fst) {
        return false;
    }
    if (!(batch.cls == PrimClass::Sprite || constantZ)) {
        return false;
    }
    endPrim = std::min<uint32_t>(endPrim, static_cast<uint32_t>(batch.prims.size()));
    for (uint32_t index = firstPrim; index < endPrim; ++index) {
        const PrimRecord& record = batch.prims[index];
        for (uint32_t k = 0; k < record.count; ++k) {
            Vertex& vertex = batch.vertices[record.first + k];
            const int eQ = UnbiasedExponent(vertex.q);
            const int eS = UnbiasedExponent(vertex.s);
            const int eT = UnbiasedExponent(vertex.t);
            vertex.s = ClearMantissaLowBits(vertex.s, StMantissaBits(eS, eQ));
            vertex.t = ClearMantissaLowBits(vertex.t, StMantissaBits(eT, eQ));
            // Sprites: only the second vertex's Q is rounded.
            const bool roundQ = record.type != PrimType::Sprite || k == 1u;
            if (roundQ) {
                vertex.q = ClearMantissaLowBits(vertex.q, 8);
            }
        }
    }
    return true;
}

bool TexelCoordinates(const Vertex& vertex, bool fst, uint32_t twLog2, uint32_t thLog2,
                      float& u, float& v) {
    if (fst) {
        u = static_cast<float>(vertex.u) / 16.0f;
        v = static_cast<float>(vertex.v) / 16.0f;
        return true;
    }
    const float su = vertex.s / vertex.q;
    const float tv = vertex.t / vertex.q;
    if (std::isnan(su) || std::isnan(tv)) {
        return false;
    }
    u = su * static_cast<float>(1u << twLog2);
    v = tv * static_cast<float>(1u << thLog2);
    return true;
}

VertexTrace ComputeVertexTrace(const Batch& batch, const clut::Palette* palette) {
    VertexTrace trace;
    const uint64_t prim = batch.state.effectivePrim;
    const bool textured = Bit(prim, primbit::kTme) != 0;
    const bool flat = Bit(prim, primbit::kIip) == 0;
    const bool abe = Bit(prim, primbit::kAbe) != 0;
    const bool aa1 = Bit(prim, primbit::kAa1) != 0;
    const bool fst = Bit(prim, primbit::kFst) != 0;

    const ContextRegs& ctx = batch.state.ctx;
    const uint32_t tfx = tex0::Tfx(ctx.tex0);
    const uint32_t tcc = tex0::Tcc(ctx.tex0);
    const uint32_t tw = tex0::Tw(ctx.tex0);
    const uint32_t th = tex0::Th(ctx.tex0);
    const bool rgbaZeroed = textured && tfx == 1u && tcc == 1u;  // DECAL with TCC = 1

    Range xr, yr, zr, fogr, sr, tr, qr;
    Range rr, gr, br, ar;
    Range texU, texV;
    double texelAbsMin = std::numeric_limits<double>::infinity();
    bool texelNaN = false;

    trace.xMin = std::numeric_limits<int32_t>::max();
    trace.yMin = std::numeric_limits<int32_t>::max();
    trace.xMax = std::numeric_limits<int32_t>::min();
    trace.yMax = std::numeric_limits<int32_t>::min();

    for (const PrimRecord& record : batch.prims) {
        const bool sprite = record.type == PrimType::Sprite;
        for (uint32_t k = 0; k < record.count; ++k) {
            const Vertex& vertex = batch.vertices[record.first + k];
            const bool provoking = k + 1u == record.count;

            xr.Add(vertex.x4);
            yr.Add(vertex.y4);
            trace.xMin = std::min(trace.xMin, vertex.x4 >> 4);
            trace.xMax = std::max(trace.xMax, vertex.x4 >> 4);
            trace.yMin = std::min(trace.yMin, vertex.y4 >> 4);
            trace.yMax = std::max(trace.yMax, vertex.y4 >> 4);

            // Sprites take Z and F from the second vertex.
            if (!(sprite && k == 0u)) {
                zr.Add(vertex.z);
                fogr.Add(vertex.fog);
            }
            // Flat shading uses only the provoking vertex's colour.
            if (!flat || provoking) {
                if (!rgbaZeroed) {
                    rr.Add(vertex.r);
                    gr.Add(vertex.g);
                    br.Add(vertex.b);
                    ar.Add(vertex.a);
                }
            }
            sr.Add(vertex.s);
            tr.Add(vertex.t);
            qr.Add(vertex.q);

            float u = 0.0f;
            float v = 0.0f;
            if (TexelCoordinates(vertex, fst, tw, th, u, v)) {
                texU.Add(u);
                texV.Add(v);
                texelAbsMin = std::min(texelAbsMin, static_cast<double>(std::fabs(u)));
            } else {
                texelNaN = true;
            }
        }
    }

    if (xr.min > xr.max) {  // empty batch
        trace.xMin = trace.xMax = trace.yMin = trace.yMax = 0;
    }
    if (trace.xMin > trace.xMax) {
        trace.xMin = trace.xMax = 0;
    }
    if (trace.yMin > trace.yMax) {
        trace.yMin = trace.yMax = 0;
    }

    // Z and fog (unsigned, integer).
    trace.zMin = zr.min > zr.max ? 0u : static_cast<uint32_t>(zr.min);
    trace.zMax = zr.min > zr.max ? 0u : static_cast<uint32_t>(zr.max);
    trace.fogMin = fogr.min > fogr.max ? 0 : static_cast<uint8_t>(fogr.min);
    trace.fogMax = fogr.min > fogr.max ? 0 : static_cast<uint8_t>(fogr.max);

    auto store = [](const Range& range, uint8_t& lo, uint8_t& hi) {
        lo = range.min > range.max ? 0 : static_cast<uint8_t>(range.min);
        hi = range.min > range.max ? 0 : static_cast<uint8_t>(range.max);
    };
    if (rgbaZeroed) {
        trace.rgbaMin[0] = trace.rgbaMin[1] = trace.rgbaMin[2] = trace.rgbaMin[3] = 0;
        trace.rgbaMax[0] = trace.rgbaMax[1] = trace.rgbaMax[2] = trace.rgbaMax[3] = 0;
    } else {
        store(rr, trace.rgbaMin[0], trace.rgbaMax[0]);
        store(gr, trace.rgbaMin[1], trace.rgbaMax[1]);
        store(br, trace.rgbaMin[2], trace.rgbaMax[2]);
        store(ar, trace.rgbaMin[3], trace.rgbaMax[3]);
    }

    // Vertex alpha; AA1 without ABE forces 128 on lines and triangles.
    trace.vertexAlphaMin = trace.rgbaMin[3];
    trace.vertexAlphaMax = trace.rgbaMax[3];
    const bool aa1Forced = aa1 && !abe &&
                           (batch.cls == PrimClass::Line || batch.cls == PrimClass::Triangle);
    if (aa1Forced) {
        trace.vertexAlphaMin = 128;
        trace.vertexAlphaMax = 128;
    }

    // Texture alpha range (chapter 01.5 TEXA rules; palette range for indexed).
    uint8_t texAlphaLo = 0;
    uint8_t texAlphaHi = 255;
    if (textured) {
        const uint32_t texFormat = tex0::Psm(ctx.tex0);
        const clut::TexaState texa = clut::TexaState::FromRegister(batch.state.env.texa);
        if (texFormat == psm::kCt16 || texFormat == psm::kCt16S) {
            texAlphaLo = static_cast<uint8_t>(std::min(texa.ta0, texa.ta1));
            texAlphaHi = static_cast<uint8_t>(std::max(texa.ta0, texa.ta1));
            if (texa.aem) {
                texAlphaLo = 0;
            }
        } else if (texFormat == psm::kCt24 || texFormat == psm::kZ24) {
            texAlphaLo = texa.aem ? 0 : static_cast<uint8_t>(texa.ta0);
            texAlphaHi = static_cast<uint8_t>(texa.ta0);
        } else if (IsIndexedPsm(texFormat) && palette != nullptr) {
            texAlphaLo = palette->minAlpha;
            texAlphaHi = palette->maxAlpha;
        }
    }

    // Fragment alpha: vertex alpha combined with the texture alpha through TFX.
    uint32_t fragLo = trace.vertexAlphaMin;
    uint32_t fragHi = trace.vertexAlphaMax;
    if (textured && tcc == 1u) {
        const uint32_t caLo = trace.vertexAlphaMin;
        const uint32_t caHi = trace.vertexAlphaMax;
        switch (tfx) {
            case 0:  // MODULATE
                fragLo = (texAlphaLo * caLo) >> 7;
                fragHi = std::min<uint32_t>(255u, (texAlphaHi * caHi) >> 7);
                break;
            case 1:  // DECAL
            case 3:  // HIGHLIGHT2
                fragLo = texAlphaLo;
                fragHi = texAlphaHi;
                break;
            default:  // HIGHLIGHT
                fragLo = std::min<uint32_t>(255u, texAlphaLo + caLo);
                fragHi = std::min<uint32_t>(255u, texAlphaHi + caHi);
                break;
        }
    }
    trace.fragmentAlphaMin = static_cast<uint8_t>(fragLo);
    trace.fragmentAlphaMax = static_cast<uint8_t>(fragHi);

    // Texel coordinates.
    trace.texelUMin = texU.min > texU.max ? 0.0f : static_cast<float>(texU.min);
    trace.texelUMax = texU.min > texU.max ? 0.0f : static_cast<float>(texU.max);
    trace.texelVMin = texV.min > texV.max ? 0.0f : static_cast<float>(texV.min);
    trace.texelVMax = texV.min > texV.max ? 0.0f : static_cast<float>(texV.max);
    trace.texelNaN = texelNaN;
    trace.qMin = qr.min > qr.max ? 1.0f : static_cast<float>(qr.min);
    trace.qMax = qr.min > qr.max ? 1.0f : static_cast<float>(qr.max);
    trace.unsafeQ = textured && !fst && std::isfinite(texelAbsMin) && texelAbsMin > 1e30;

    // Filter choice (chapter 02.9).
    const uint32_t mxl = tex1::Mxl(ctx.tex1);
    const bool magLinear = tex1::Mmag(ctx.tex1) != 0;
    const uint32_t mmin = tex1::Mmin(ctx.tex1);
    const bool minLinear = mmin == 1u || mmin == 4u || mmin == 5u;
    if (mxl == 0u) {
        trace.filter = magLinear ? VertexTrace::Filter::Linear : VertexTrace::Filter::Nearest;
    } else {
        double lodLo = 0.0;
        double lodHi = 0.0;
        const double k16 = static_cast<double>(tex1::K(ctx.tex1)) / 16.0;
        if (tex1::Lcm(ctx.tex1) == 0u && !fst) {
            const double scale = std::ldexp(1.0, static_cast<int>(tex1::L(ctx.tex1)));
            auto lodAt = [&](double q) {
                if (q <= 0.0) {
                    return std::numeric_limits<double>::infinity();
                }
                return -std::log2(q) * scale + k16;
            };
            const double atMin = lodAt(static_cast<double>(trace.qMin));
            const double atMax = lodAt(static_cast<double>(trace.qMax));
            lodLo = std::min(atMin, atMax);
            lodHi = std::max(atMin, atMax);
        } else {
            lodLo = lodHi = k16;
        }
        if (lodHi <= 0.0) {
            trace.filter = magLinear ? VertexTrace::Filter::Linear : VertexTrace::Filter::Nearest;
        } else if (lodLo > 0.0) {
            trace.filter = minLinear ? VertexTrace::Filter::Linear : VertexTrace::Filter::Nearest;
        } else {
            trace.filter = VertexTrace::Filter::Either;
        }
    }

    // Constant mask (bits: R G B A x y z f s t q).
    uint32_t constant = 0;
    auto setIf = [&](bool isConstant, unsigned bit) {
        if (isConstant) {
            constant |= 1u << bit;
        }
    };
    setIf(rgbaZeroed || rr.Constant(), 0);
    setIf(rgbaZeroed || gr.Constant(), 1);
    setIf(rgbaZeroed || br.Constant(), 2);
    setIf(rgbaZeroed || ar.Constant(), 3);
    setIf(xr.Constant(), 4);
    setIf(yr.Constant(), 5);
    setIf(zr.Constant(), 6);
    setIf(fogr.Constant(), 7);
    setIf(sr.Constant(), 8);
    setIf(tr.Constant(), 9);
    setIf(qr.Constant(), 10);
    trace.constantMask = constant;

    return trace;
}

OverlapKind ClassifyOverlap(const Batch& batch) {
    if (batch.vertices.size() < 4u) {
        return OverlapKind::None;
    }
    if (batch.cls == PrimClass::Sprite) {
        return BuildSpriteDrawList(batch).size() <= 1u ? OverlapKind::None : OverlapKind::Possible;
    }
    if (batch.cls == PrimClass::Triangle && batch.prims.size() == 2u) {
        // Exactly two triangles forming an axis-aligned quad with a shared diagonal.
        int32_t minX = batch.vertices.front().x4;
        int32_t maxX = minX;
        int32_t minY = batch.vertices.front().y4;
        int32_t maxY = minY;
        for (const Vertex& vertex : batch.vertices) {
            minX = std::min(minX, vertex.x4);
            maxX = std::max(maxX, vertex.x4);
            minY = std::min(minY, vertex.y4);
            maxY = std::max(maxY, vertex.y4);
        }
        uint32_t corners[2] = {0, 0};
        for (uint32_t p = 0; p < 2u; ++p) {
            const PrimRecord& record = batch.prims[p];
            for (uint32_t k = 0; k < record.count; ++k) {
                const Vertex& vertex = batch.vertices[record.first + k];
                const bool onX0 = vertex.x4 == minX;
                const bool onX1 = vertex.x4 == maxX;
                const bool onY0 = vertex.y4 == minY;
                const bool onY1 = vertex.y4 == maxY;
                if (!((onX0 || onX1) && (onY0 || onY1))) {
                    return OverlapKind::Unknown;
                }
                // Corner index: 0 (x0,y0), 1 (x1,y0), 2 (x1,y1), 3 (x0,y1).
                uint32_t corner = 0;
                if (onX1 && onY0) corner = 1;
                else if (onX1 && onY1) corner = 2;
                else if (onX0 && onY1) corner = 3;
                corners[p] |= 1u << corner;
            }
            if (__builtin_popcount(corners[p]) != 3) {
                return OverlapKind::Unknown;
            }
        }
        const uint32_t shared = corners[0] & corners[1];
        const bool diagonal = shared == 0x5u || shared == 0xAu;
        if ((corners[0] | corners[1]) == 0xFu && diagonal) {
            return OverlapKind::None;
        }
    }
    return OverlapKind::Unknown;
}

std::vector<DrawGroup> BuildSpriteDrawList(const Batch& batch) {
    std::vector<DrawGroup> groups;
    DrawGroup current;
    bool started = false;
    for (const PrimRecord& record : batch.prims) {
        if (record.type != PrimType::Sprite) {
            continue;
        }
        if (!started) {
            current.bounds = record.rect;
            current.primitiveCount = 1;
            started = true;
        } else if (RectsOverlap(current.bounds, record.rect)) {
            groups.push_back(current);
            current.bounds = record.rect;
            current.primitiveCount = 1;
        } else {
            current.bounds = UnionRects(current.bounds, record.rect);
            current.primitiveCount++;
        }
    }
    if (started) {
        groups.push_back(current);
    }
    for (DrawGroup& group : groups) {
        group.bounds.x0 -= 1;
        group.bounds.y0 -= 1;
        group.bounds.x1 += 1;
        group.bounds.y1 += 1;
    }
    return groups;
}

uint32_t FootprintBlocks(uint32_t widthUnits, uint32_t heightPixels) {
    const uint64_t width = std::max<uint32_t>(widthUnits, 1u);
    const uint64_t rows = (static_cast<uint64_t>(heightPixels) + 31u) / 32u;
    const uint64_t blocks = width * std::max<uint64_t>(rows, 1u) * kBlocksPerPage;
    return static_cast<uint32_t>(std::min<uint64_t>(blocks, kBlocksTotal));
}

bool BlockRangesOverlap(uint32_t begin0, uint32_t count0, uint32_t begin1, uint32_t count1) {
    if (count0 == 0u || count1 == 0u) {
        return false;
    }
    // Two arcs of the circular block space intersect exactly when one of them contains the other's first
    // block (IN-360: constant time; the block-by-block walk cost thousands of steps per frame-buffer check).
    const uint32_t start0 = begin0 % kBlocksTotal;
    const uint32_t start1 = begin1 % kBlocksTotal;
    const uint32_t from0To1 = (start1 + kBlocksTotal - start0) % kBlocksTotal;  // start1 seen from start0
    const uint32_t from1To0 = (start0 + kBlocksTotal - start1) % kBlocksTotal;  // start0 seen from start1
    return from0To1 < count0 || from1To0 < count1;
}

bool BlockRangesOverlapByWalk(uint32_t begin0, uint32_t count0, uint32_t begin1, uint32_t count1) {
    if (count0 == 0u || count1 == 0u) {
        return false;
    }
    for (uint32_t i = 0; i < count0 && i < kBlocksTotal; ++i) {
        const uint32_t block = (begin0 + i) % kBlocksTotal;
        const uint32_t offset = (block + kBlocksTotal - begin1 % kBlocksTotal) % kBlocksTotal;
        if (offset < count1) {
            return true;
        }
    }
    return false;
}

}  // namespace coregs
