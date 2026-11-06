// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
//
// HWR CPU reference fixtures (TASK-027): coverage vectors, flat shading, sprites, CLUT
// lookups, blend sweeps through the full draw path, IN-07, depth, and synthetic capture replay.
// Written from docs/hw-renderer-spec (clean room, decision 0069).
//
// Usage: cpuref_tests <test-name>   (names are listed in CMakeLists.txt)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cpuref_draw.hpp"
#include "cpuref_memory.hpp"
#include "cpuref_renderer.hpp"
#include "cpuref_replay.hpp"
#include "hw_batch.hpp"
#include "hw_clut.hpp"
#include "hw_gs_counters.hpp"
#include "hw_gs_pixel.hpp"
#include "hw_gs_registers.hpp"
#include "hw_gs_types.hpp"
#include "hw_memory.hpp"

using namespace coregs;
using namespace coregs::cpuref;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                     \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                               \
        }                                                                               \
    } while (0)

#define CHECK_EQ(a, b)                                                                             \
    do {                                                                                           \
        const long long va_ = static_cast<long long>(a);                                           \
        const long long vb_ = static_cast<long long>(b);                                           \
        if (va_ != vb_) {                                                                          \
            std::fprintf(stderr, "%s:%d: CHECK_EQ failed: %s == %s (%lld vs %lld)\n", __FILE__,    \
                         __LINE__, #a, #b, va_, vb_);                                              \
            ++g_failures;                                                                          \
        }                                                                                          \
    } while (0)

// ---- Register builders ---------------------------------------------------------------------

uint64_t FrameReg(uint32_t fbp, uint32_t fbw, uint32_t psm, uint32_t fbmsk) {
    return uint64_t{fbp} | (uint64_t{fbw} << 16) | (uint64_t{psm} << 24) | (uint64_t{fbmsk} << 32);
}

uint64_t ZbufReg(uint32_t zbp, uint32_t psm, uint32_t zmsk) {
    return uint64_t{zbp} | (uint64_t{psm} << 24) | (uint64_t{zmsk} << 32);
}

uint64_t TestReg(uint32_t ate, uint32_t atst, uint32_t aref, uint32_t afail, uint32_t date, uint32_t datm,
                 uint32_t zte, uint32_t ztst) {
    return uint64_t{ate} | (uint64_t{atst} << 1) | (uint64_t{aref} << 4) | (uint64_t{afail} << 12) |
           (uint64_t{date} << 14) | (uint64_t{datm} << 15) | (uint64_t{zte} << 16) | (uint64_t{ztst} << 17);
}

uint64_t ScissorReg(uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1) {
    return uint64_t{x0} | (uint64_t{x1} << 16) | (uint64_t{y0} << 32) | (uint64_t{y1} << 48);
}

uint64_t PrimBits(uint32_t type, bool iip, bool tme, bool fge, bool abe, bool fst) {
    return uint64_t{type} | (uint64_t{iip} << primbit::kIip) | (uint64_t{tme} << primbit::kTme) |
           (uint64_t{fge} << primbit::kFge) | (uint64_t{abe} << primbit::kAbe) | (uint64_t{fst} << primbit::kFst);
}

uint64_t Tex0Reg(uint32_t tbp, uint32_t tbw, uint32_t psm, uint32_t tw, uint32_t th, uint32_t tfx, uint32_t tcc,
                 uint32_t cbp, uint32_t cpsm, uint32_t csa) {
    return uint64_t{tbp} | (uint64_t{tbw} << 14) | (uint64_t{psm} << 20) | (uint64_t{tw} << 26) |
           (uint64_t{th} << 30) | (uint64_t{tcc} << 34) | (uint64_t{tfx} << 35) | (uint64_t{cbp} << 37) |
           (uint64_t{cpsm} << 51) | (uint64_t{csa} << 56);
}

// One vertex in window pixels (converted to 12.4) with flat attributes.
Vertex MakeVertex(int32_t x, int32_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint32_t z = 0) {
    Vertex v;
    v.x4 = x * 16;
    v.y4 = y * 16;
    v.r = r;
    v.g = g;
    v.b = b;
    v.a = a;
    v.z = z;
    v.q = 1.0f;
    return v;
}

// A batch with the given register state and one primitive over its vertices.
Batch MakeBatch(const ContextRegs& ctx, const EnvironmentRegs& env, uint64_t prim, PrimType type,
                const std::vector<Vertex>& vertices) {
    Batch batch;
    batch.state.ctx = ctx;
    batch.state.env = env;
    batch.state.effectivePrim = prim;
    batch.vertices = vertices;
    PrimRecord record;
    record.type = type;
    record.first = 0;
    record.count = static_cast<uint32_t>(vertices.size());
    record.rect = PixelRect{0, 0, 63, 63};
    batch.prims.push_back(record);
    batch.cls = type == PrimType::Sprite ? PrimClass::Sprite : PrimClass::Triangle;
    return batch;
}

// Frame at page 0, 64 pixels wide, CT32; depth at page 16 (Z32); depth writes off by default.
ContextRegs BaseContext() {
    ContextRegs ctx;
    ctx.frame = FrameReg(0, 1, 0, 0);
    ctx.zbuf = ZbufReg(16, 0, 1);
    ctx.test = TestReg(0, 1, 0, 0, 0, 0, 0, 1);
    ctx.scissor = ScissorReg(0, 63, 0, 63);
    ctx.alpha = 0;
    return ctx;
}

// Frame CT32 view used by the direct draws (page 0, 64 pixels wide).
memory::BufferView FrameView() { return memory::BufferView{0, 1, memory::Psm::Ct32}; }
memory::BufferView DepthView() { return memory::BufferView{memory::blockOfPage(16), 1, memory::Psm::Z32}; }

uint32_t Rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return uint32_t{r} | (uint32_t{g} << 8) | (uint32_t{b} << 16) | (uint32_t{a} << 24);
}

// Draws one batch into a fresh local memory with a fresh CLUT unit.
struct Harness {
    memory::LocalMemory local;
    CounterSet counters;
    clut::ClutUnit cluts{counters};
    DrawStats stats;

    void Draw(const Batch& batch) { DrawBatch(batch, local, cluts, stats); }
    uint32_t Frame(uint32_t x, uint32_t y) const { return local.readPixel(FrameView(), x, y); }
    uint32_t Depth(uint32_t x, uint32_t y) const { return local.readPixel(DepthView(), x, y); }
};

// ---- Coverage ------------------------------------------------------------------------------

// Spec vectors V-01-5 (sprite edges) and the coverage rule of 01.4 on one axis.
void TestCoverageVectors() {
    CHECK(SpriteCoversPixel(0, 32, 0));
    CHECK(SpriteCoversPixel(0, 32, 1));
    CHECK(!SpriteCoversPixel(0, 32, 2));
    CHECK(!SpriteCoversPixel(8, 24, 0));
    CHECK(SpriteCoversPixel(8, 24, 1));
    CHECK(!SpriteCoversPixel(8, 24, 2));
    CHECK(!SpriteCoversPixel(1, 16, 0));
    CHECK(!SpriteCoversPixel(1, 16, 1));
    CHECK(SpriteCoversPixel(32, 0, 1));  // reversed vertex order covers the same pixels
    // Points and line endpoints snap to the window pixel by floor (IN-300).
    CHECK_EQ(WindowPixel(8), 0);
    CHECK_EQ(WindowPixel(24), 1);
    CHECK_EQ(WindowPixel(-8), -1);
}

void TestLinePixels() {
    const std::vector<PixelCoord> diagonal = LinePixels(0, 0, 48, 48);
    CHECK_EQ(diagonal.size(), 4);
    CHECK(diagonal.front().x == 0 && diagonal.front().y == 0);
    CHECK(diagonal.back().x == 3 && diagonal.back().y == 3);
    const std::vector<PixelCoord> shallow = LinePixels(0, 0, 80, 32);  // to pixel (5, 2)
    CHECK_EQ(shallow.size(), 6);
    CHECK(shallow.back().x == 5 && shallow.back().y == 2);
    CHECK_EQ(LinePixels(16, 16, 16, 16).size(), 1);
}

// Two triangles sharing a diagonal cover each pixel of the 16 x 16 square exactly once.
void TestTriangleSharedEdge() {
    const Vertex a = MakeVertex(0, 0, 0, 0, 0, 0);
    const Vertex b = MakeVertex(16, 0, 0, 0, 0, 0);
    const Vertex c = MakeVertex(0, 16, 0, 0, 0, 0);
    const Vertex d = MakeVertex(16, 16, 0, 0, 0, 0);
    int mismatches = 0;
    for (int32_t y = 0; y < 16; ++y) {
        for (int32_t x = 0; x < 16; ++x) {
            // Triangles (a, b, c) and (b, d, c) share the diagonal b-c.
            const int covered = (TriangleCoversPixel(a, b, c, x, y) ? 1 : 0) +
                                (TriangleCoversPixel(b, d, c, x, y) ? 1 : 0);
            if (covered != 1) {
                ++mismatches;
            }
        }
    }
    CHECK_EQ(mismatches, 0);
    // A degenerate triangle covers nothing.
    CHECK(!TriangleCoversPixel(a, b, MakeVertex(32, 0, 0, 0, 0, 0), 5, 0));
}

// ---- Draw path -----------------------------------------------------------------------------

// Flat shading takes the last vertex's colour across the whole triangle (hw_flat_provoking).
void TestFlatProvokingVertex() {
    Harness h;
    const std::vector<Vertex> verts = {
        MakeVertex(0, 0, 255, 0, 0, 0x80),
        MakeVertex(8, 0, 0, 255, 0, 0x80),
        MakeVertex(0, 8, 0, 0, 255, 0x80),
    };
    // PRIM with IIP = 0: flat.
    h.Draw(MakeBatch(BaseContext(), EnvironmentRegs{}, PrimBits(3, false, false, false, false, false),
                     PrimType::Triangle, verts));
    CHECK_EQ(h.Frame(1, 1), Rgba(0, 0, 255, 0x80));
    CHECK_EQ(h.Frame(2, 2), Rgba(0, 0, 255, 0x80));
    CHECK_EQ(h.stats.coveredPixels > 0, 1);
    // Outside the triangle nothing is written.
    CHECK_EQ(h.Frame(7, 7), 0);
}

// Gouraud shading interpolates; the vertex colours are exact at the vertices.
void TestGouraudInterpolation() {
    Harness h;
    const std::vector<Vertex> verts = {
        MakeVertex(0, 0, 200, 0, 0, 0x80),
        MakeVertex(8, 0, 0, 200, 0, 0x80),
        MakeVertex(0, 8, 0, 0, 200, 0x80),
    };
    h.Draw(MakeBatch(BaseContext(), EnvironmentRegs{}, PrimBits(3, true, false, false, false, false),
                     PrimType::Triangle, verts));
    CHECK_EQ(h.Frame(0, 0) & 0xFFu, 200);       // red at vertex a
    CHECK_EQ((h.Frame(0, 0) >> 8) & 0xFFu, 0);
    // Pixel (0, 1) lies on the edge a-c, halfway-ish: the green/blue values interpolate.
    const uint32_t mid = h.Frame(0, 4);
    CHECK(((mid >> 0) & 0xFFu) > 0 && ((mid >> 16) & 0xFFu) > 0);
}

// IN-328: a constant interpolated attribute keeps its exact value. A Gouraud triangle whose three
// vertices share colour (200, 100, 50) and fog 90 must shade to (80, 55, 58) everywhere: the
// barycentric weights are doubles that do not sum to 1, so an unguarded interpolation gives 89.99 and
// the 0..255 truncation loses one unit. Fog colour 0x402010 -> (0x10, 0x20, 0x40), chapter 03.4.
void TestConstantAttributes() {
    Harness h;
    EnvironmentRegs env;
    env.fogcol = 0x402010;
    std::vector<Vertex> verts = {MakeVertex(0, 0, 200, 100, 50, 0x80), MakeVertex(32, 0, 200, 100, 50, 0x80),
                                 MakeVertex(0, 32, 200, 100, 50, 0x80)};
    for (Vertex& v : verts) {
        v.fog = 90;
    }
    h.Draw(MakeBatch(BaseContext(), env, PrimBits(3, true, false, true, false, false), PrimType::Triangle, verts));
    int covered = 0;
    for (uint32_t y = 0; y < 32; ++y) {
        for (uint32_t x = 0; x < 32; ++x) {
            if (h.Frame(x, y) == 0) {
                continue;
            }
            ++covered;
            CHECK_EQ(h.Frame(x, y), Rgba(80, 55, 58, 0x80));
        }
    }
    CHECK_EQ(covered, 528);  // x + y <= 31: the diagonal is a top-left edge, so it is inclusive (01.4)
}

// A 4 x 4 sprite covers 16 pixels, with colour from the second vertex.
void TestSpriteRectangle() {
    Harness h;
    const std::vector<Vertex> verts = {
        MakeVertex(0, 0, 10, 10, 10, 0x80),
        MakeVertex(4, 4, 90, 80, 70, 0x80),
    };
    h.Draw(MakeBatch(BaseContext(), EnvironmentRegs{}, PrimBits(6, true, false, false, false, false),
                     PrimType::Sprite, verts));
    int covered = 0;
    for (uint32_t y = 0; y < 8; ++y) {
        for (uint32_t x = 0; x < 8; ++x) {
            if (h.Frame(x, y) != 0) {
                ++covered;
                CHECK_EQ(h.Frame(x, y), Rgba(90, 80, 70, 0x80));
            }
        }
    }
    CHECK_EQ(covered, 16);
    // The second vertex is the far corner: pixel 4 is outside the sprite (min <= n < max).
    CHECK_EQ(h.Frame(4, 0), 0);
}

// Scissor clips the draw to the window rectangle.
void TestScissorClip() {
    Harness h;
    ContextRegs ctx = BaseContext();
    ctx.scissor = ScissorReg(2, 3, 0, 63);
    const std::vector<Vertex> verts = {
        MakeVertex(0, 0, 1, 2, 3, 0x80),
        MakeVertex(8, 0, 1, 2, 3, 0x80),
        MakeVertex(0, 8, 1, 2, 3, 0x80),
    };
    h.Draw(MakeBatch(ctx, EnvironmentRegs{}, PrimBits(3, false, false, false, false, false), PrimType::Triangle,
                     verts));
    CHECK_EQ(h.Frame(1, 1), 0);
    CHECK(h.Frame(2, 1) != 0);
    CHECK(h.Frame(3, 1) != 0);
    CHECK_EQ(h.Frame(4, 1), 0);
}

// Depth: a test that fails discards colour and Z; passing draws update both (ZMSK = 0).
void TestDepthTestAndWrite() {
    Harness h;
    ContextRegs ctx = BaseContext();
    ctx.zbuf = ZbufReg(16, 0, 0);          // ZMSK = 0: depth writes allowed
    ctx.test = TestReg(0, 1, 0, 0, 0, 0, 1, 1);  // ZTE = 1, ZTST = ALWAYS
    const uint64_t prim = PrimBits(6, false, false, false, false, false);
    h.Draw(MakeBatch(ctx, EnvironmentRegs{}, prim, PrimType::Sprite,
                     {MakeVertex(0, 0, 0, 0, 0, 0x80, 1000), MakeVertex(2, 2, 40, 0, 0, 0x80, 1000)}));
    CHECK_EQ(h.Depth(0, 0), 1000);
    CHECK_EQ(h.Frame(0, 0), Rgba(40, 0, 0, 0x80));

    // GEQUAL 500 against stored 1000 fails: colour and Z stay.
    ctx.test = TestReg(0, 1, 0, 0, 0, 0, 1, 2);  // ZTST = GEQUAL
    h.Draw(MakeBatch(ctx, EnvironmentRegs{}, prim, PrimType::Sprite,
                     {MakeVertex(0, 0, 0, 40, 0, 0x80, 500), MakeVertex(2, 2, 0, 40, 0, 0x80, 500)}));
    CHECK_EQ(h.Depth(0, 0), 1000);
    CHECK_EQ(h.Frame(0, 0), Rgba(40, 0, 0, 0x80));

    // GEQUAL 2000 passes and writes both.
    h.Draw(MakeBatch(ctx, EnvironmentRegs{}, prim, PrimType::Sprite,
                     {MakeVertex(0, 0, 0, 40, 0, 0x80, 2000), MakeVertex(2, 2, 0, 40, 0, 0x80, 2000)}));
    CHECK_EQ(h.Depth(0, 0), 2000);
    CHECK_EQ(h.Frame(0, 0), Rgba(0, 40, 0, 0x80));
}

// Textures: a 4 x 4 CT32 texture with FST = 1 and DECAL, nearest sampling, one texel per pixel.
void TestTextureNearestDecal() {
    Harness h;
    // Texel (x, y) = (x * 16, y * 16, 0, 0xFF): stored in the SLM at block 0 (TBP 0), width 1.
    for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
            h.local.writePixel(memory::BufferView{0, 1, memory::Psm::Ct32}, x, y, Rgba(uint8_t(x * 16), uint8_t(y * 16), 0, 0xFF));
        }
    }
    ContextRegs ctx = BaseContext();
    ctx.tex0 = Tex0Reg(0, 1, 0, 2, 2, 1, 1, 0, 0, 0);  // TW = TH = 2, DECAL (tfx 1)
    ctx.tex1 = 0;                                       // MMAG = 0: nearest
    EnvironmentRegs env;
    // Sprite over pixels (0,0)..(3,3) with texture coordinates in 10.4 texels (u = x * 16).
    Vertex a = MakeVertex(0, 0, 128, 128, 128, 0x80);
    Vertex b = MakeVertex(4, 4, 128, 128, 128, 0x80);
    a.u = 0;
    a.v = 0;
    b.u = 4 * 16;
    b.v = 4 * 16;
    h.Draw(MakeBatch(ctx, env, PrimBits(6, true, true, false, false, true), PrimType::Sprite, {a, b}));
    CHECK_EQ(h.Frame(2, 3) & 0xFFu, 2 * 16);
    CHECK_EQ((h.Frame(2, 3) >> 8) & 0xFFu, 3 * 16);
}

// CLUT lookups: every 4-bit index maps to the entry the CLUT unit decodes (chapter 07.5 is
// the source of truth; the test does not re-derive the CLUT layout).
void TestClutIndexedLookup() {
    Harness h;
    // Palette CLUT stored at CBP 64 (CT32 entries written as raw words, 0x40 blocks).
    for (uint32_t i = 0; i < 16; ++i) {
        h.local.writePixel(memory::BufferView{64, 1, memory::Psm::Ct32}, i % 8, i / 8, Rgba(uint8_t(i * 7), uint8_t(255 - i), uint8_t(i * 3), 0x80));
    }
    clut::Tex0Write write;
    write.psm = 0x14;  // T4
    write.cbp = 64;
    write.cpsm = 0;
    write.csa = 0;
    write.cld = 1;
    const bool loaded = h.cluts.ProcessWrite(write, h.local.bytes().data(), [] {});
    CHECK(loaded);
    const clut::Palette palette = h.cluts.Decode(0x14, 0, 0, 0);
    CHECK(palette.entries[1] != 0);

    // T4 texture of 4 x 4 indices at TBP 0: index = (x + 4 * y) & 15.
    for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
            h.local.writePixel(memory::BufferView{0, 1, memory::Psm::T4}, x, y, (x + 4 * y) & 15u);
        }
    }
    ContextRegs ctx = BaseContext();
    ctx.tex0 = Tex0Reg(0, 1, 0x14, 2, 2, 1, 1, 64, 0, 0);  // T4 with the CLUT at block 64
    ctx.tex1 = 0;
    EnvironmentRegs env;
    Vertex a = MakeVertex(0, 0, 128, 128, 128, 0x80);
    Vertex b = MakeVertex(4, 4, 128, 128, 128, 0x80);
    a.u = 0;
    a.v = 0;
    b.u = 4 * 16;
    b.v = 4 * 16;
    h.Draw(MakeBatch(ctx, env, PrimBits(6, true, true, false, false, true), PrimType::Sprite, {a, b}));
    // Pixel (x, y) samples texel (x, y): the palette entry of index x + 4y, through DECAL.
    for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
            const uint32_t expected = palette.entries[(x + 4 * y) & 15u];
            CHECK_EQ(h.Frame(x, y) & 0x00FFFFFFu, expected & 0x00FFFFFFu);
        }
    }
}

// Blend sweep through the full draw path: all 81 combinations of A, B, C, D selectors,
// compared with the chapter 08.1 formula written out here.
void TestBlendSweepFullPath() {
    const uint8_t fix = 0x50;
    const uint8_t destination[4] = {100, 150, 200, 128};
    const uint8_t source[4] = {60, 30, 250, 64};
    int mismatches = 0;
    int cases = 0;
    for (uint32_t sa = 0; sa < 3; ++sa) {
        for (uint32_t sb = 0; sb < 3; ++sb) {
            for (uint32_t sc = 0; sc < 3; ++sc) {
                for (uint32_t sd = 0; sd < 3; ++sd) {
                    // Selectors: A and B and D use 0 Cs, 1 Cd, 2 zero; C uses 0 As, 1 Ad, 2 FIX.
                    Harness h;
                    h.local.writePixel(FrameView(), 0, 0, Rgba(destination[0], destination[1], destination[2], destination[3]));
                    ContextRegs ctx = BaseContext();
                    ctx.alpha = pixel::encode_alpha_register(
                        static_cast<pixel::BlendSource>(sa), static_cast<pixel::BlendSource>(sb),
                        static_cast<pixel::BlendFactor>(sc), static_cast<pixel::BlendSource>(sd), fix);
                    EnvironmentRegs env;
                    env.colclamp = 1;
                    const uint64_t prim = PrimBits(6, false, false, false, true, false);
                    h.Draw(MakeBatch(ctx, env, prim, PrimType::Sprite,
                                     {MakeVertex(0, 0, source[0], source[1], source[2], source[3]),
                                      MakeVertex(1, 1, source[0], source[1], source[2], source[3])}));
                    const uint32_t got = h.Frame(0, 0);

                    // Chapter 08.1: Cv = (((A - B) * C) >> 7) + D, per RGB channel, clamped.
                    int32_t expected[3];
                    const int32_t factorAs = source[3];
                    const int32_t factorAd = destination[3];
                    const int32_t factorC = sc == 0 ? factorAs : (sc == 1 ? factorAd : fix);
                    for (int c = 0; c < 3; ++c) {
                        const int32_t cs = source[c];
                        const int32_t cd = destination[c];
                        const int32_t a = sa == 0 ? cs : (sa == 1 ? cd : 0);
                        const int32_t b = sb == 0 ? cs : (sb == 1 ? cd : 0);
                        const int32_t d = sd == 0 ? cs : (sd == 1 ? cd : 0);
                        int32_t value = (((a - b) * factorC) >> 7) + d;
                        expected[c] = value < 0 ? 0 : (value > 255 ? 255 : value);
                    }
                    const uint32_t want = Rgba(static_cast<uint8_t>(expected[0]), static_cast<uint8_t>(expected[1]),
                                               static_cast<uint8_t>(expected[2]), source[3]);
                    ++cases;
                    if ((got & 0x00FFFFFFu) != (want & 0x00FFFFFFu)) {
                        ++mismatches;
                    }
                }
            }
        }
    }
    CHECK_EQ(cases, 81);
    CHECK_EQ(mismatches, 0);
}

// IN-07 (resolved): the 16-bit A bit is checked first, in both the texel and CLUT forms.
void TestAemSingleRule() {
    clut::TexaState texa;
    texa.ta0 = 0x40;
    texa.ta1 = 0x80;
    texa.aem = true;
    // A bit set: TA1, even when RGB is zero under AEM.
    CHECK_EQ(clut::ExpandRgb16(0x8000, texa, false) >> 24, 0x80);
    CHECK_EQ(clut::ExpandRgb16(0x8000, texa, true) >> 24, 0x80);
    CHECK_EQ(pixel::expand_texel16(0x8000, 0x40, 0x80, true).a, 0x80);
    // A bit clear, RGB zero under AEM: alpha 0; otherwise TA0.
    CHECK_EQ(pixel::expand_texel16(0x0000, 0x40, 0x80, true).a, 0);
    CHECK_EQ(clut::ExpandRgb16(0x0000, texa, true) >> 24, 0);
    CHECK_EQ(pixel::expand_texel16(0x0001, 0x40, 0x80, true).a, 0x40);
    CHECK_EQ(clut::ExpandRgb16(0x0001, texa, false) >> 24, 0x40);
    // The two forms agree on every 16-bit value and both AEM settings.
    int mismatches = 0;
    for (uint32_t value = 0; value < 65536; ++value) {
        for (int aem = 0; aem < 2; ++aem) {
            clut::TexaState state = texa;
            state.aem = aem != 0;
            const uint32_t fromClut = clut::ExpandRgb16(static_cast<uint16_t>(value), state, true);
            const uint32_t fromRgb = clut::ExpandRgb16(static_cast<uint16_t>(value), state, false);
            const pixel::Colour fromPixel = pixel::expand_texel16(static_cast<uint16_t>(value), 0x40, 0x80, aem != 0);
            const bool agree = fromClut == fromRgb && static_cast<int32_t>(fromClut >> 24) == fromPixel.a;
            if (!agree) {
                ++mismatches;
            }
        }
    }
    CHECK_EQ(mismatches, 0);
}

// Synthetic capture: GIF packets, privileged writes and frame ends replay through the CPU
// reference and draw the triangle into the frame; malformed streams are rejected.
void TestSyntheticCaptureReplay() {
    CaptureWriter writer;
    // FRAME (0x4C): page 0, width 1 (64 pixels), CT32, mask 0. ZBUF (0x4E): ZMSK set.
    // PACKED A+D writes (descriptor 0xE; value in lo, register address in hi), chapter 02.1.
    writer.Gif(1, CaptureWriter::Packet(1, true, false, 0, 0, {0xE}, {FrameReg(0, 1, 0, 0), 0x4C}));
    writer.Gif(1, CaptureWriter::Packet(1, true, false, 0, 0, {0xE}, {ZbufReg(16, 0, 1), 0x4E}));
    writer.Gif(1, CaptureWriter::Packet(1, true, false, 0, 0, {0xE}, {ScissorReg(0, 63, 0, 63), 0x40}));
    // Triangle: PRIM = 3 (A+D to 0x00), then three vertices of RGBAQ and XYZ2 in REGLIST.
    std::vector<uint64_t> words;
    const uint32_t positions[3][2] = {{0, 0}, {8, 0}, {0, 8}};
    for (int i = 0; i < 3; ++i) {
        const uint64_t rgbaq = uint64_t{0x30} | (uint64_t{0x60} << 8) | (uint64_t{0x90} << 16) | (uint64_t{0x80} << 24) | (uint64_t{0x3F800000u} << 32);
        words.push_back(rgbaq);
        const uint64_t xyz = uint64_t{positions[i][0] * 16} | (uint64_t{positions[i][1] * 16} << 16);
        words.push_back(xyz);
    }
    // PRE applies to PACKED only (02.1): PRIM = 3 is written through an A+D packet to 0x00.
    writer.Gif(1, CaptureWriter::Packet(1, true, false, 0, 0, {0xE}, {3, 0x00}));
    writer.Gif(1, CaptureWriter::Packet(3, true, false, 0, 1, {0x01, 0x05}, words));
    writer.FrameEnd(7);

    CpuRenderer renderer;
    ReplaySummary summary;
    std::string error;
    uint64_t lastFrame = 0;
    const bool ok = ReplayCapture(writer.Bytes(), renderer, summary,
                                  [&](uint64_t frame, const FrameImage&) { lastFrame = frame; }, error);
    CHECK(ok);
    CHECK_EQ(summary.frames, 1);
    CHECK_EQ(lastFrame, 7);
    CHECK_EQ(summary.gifPackets, 5);
    CHECK_EQ(renderer.Stats().batches, 1);
    // Flat shading: the last vertex colour (R 0x30, G 0x60, B 0x90, A 0x80) fills the triangle.
    CHECK_EQ(renderer.Memory().Local().readPixel(FrameView(), 1, 1), Rgba(0x30, 0x60, 0x90, 0x80));

    // A truncated stream and an unknown tag are rejected with a message.
    std::vector<uint8_t> truncated = writer.Bytes();
    truncated.resize(truncated.size() - 3);
    CpuRenderer second;
    ReplaySummary secondSummary;
    CHECK(!ReplayCapture(truncated, second, secondSummary, nullptr, error));
    std::vector<uint8_t> unknown = {'Z'};
    CHECK(!ReplayCapture(unknown, second, secondSummary, nullptr, error));
    // GIF path 0 is rejected and counted (IN-312).
    CaptureWriter zeroPath;
    zeroPath.Gif(0, CaptureWriter::Packet(1, true, false, 0, 1, {0x4C}, {0}));
    CpuRenderer third;
    ReplaySummary thirdSummary;
    CHECK(ReplayCapture(zeroPath.Bytes(), third, thirdSummary, nullptr, error));
    CHECK_EQ(thirdSummary.rejectedGifPackets, 1);
}

// Frame encoding: P6 header and raster size.
void TestPpmEncoding() {
    display::Image image(2, 1);
    image.at(0, 0) = display::Rgba8{1, 2, 3, 4};
    image.at(1, 0) = display::Rgba8{5, 6, 7, 8};
    const std::string ppm = EncodePpm(image);
    CHECK(ppm.rfind("P6\n2 1\n255\n", 0) == 0);
    CHECK_EQ(ppm.size(), std::string("P6\n2 1\n255\n").size() + 6);
    CHECK_EQ(static_cast<unsigned char>(ppm[11]), 1);
    CHECK_EQ(static_cast<unsigned char>(ppm[16]), 7);
}

// Display path (13.1 to 13.3): NTSC circuit 1 reads the frame at page 0, the top-left pixel
// of the composed frame equals the framebuffer pixel (no deinterlace, no offsets).
void TestDisplayFrameFromRegisters() {
    CpuRenderer renderer;
    renderer.Memory().Local().writePixel(memory::BufferView{0, 1, memory::Psm::Ct32}, 0, 0, Rgba(255, 0, 0, 0x80));
    renderer.WritePrivileged(priv::kPmode, 1);                              // EN1
    renderer.WritePrivileged(priv::kSmode1, uint64_t{2} << 13);             // CMOD = 2 (NTSC)
    renderer.WritePrivileged(priv::kDispfb1, uint64_t{1} << 9);             // FBP 0, FBW 1, PSM CT32
    renderer.WritePrivileged(priv::kDisplay1, (uint64_t{639} << 32) | (uint64_t{223} << 44));  // DW, DH
    const FrameImage frame = renderer.EndFrame();
    CHECK(frame.valid);
    CHECK(frame.image.width > 0 && frame.image.height > 0);
    CHECK_EQ(frame.image.at(0, 0).r, 255);
    CHECK_EQ(frame.image.at(0, 0).g, 0);
}

struct Test {
    const char* name;
    void (*run)();
};

const Test kTests[] = {
    {"cpuref_coverage_vectors", TestCoverageVectors},
    {"cpuref_line_pixels", TestLinePixels},
    {"cpuref_triangle_shared_edge", TestTriangleSharedEdge},
    {"cpuref_flat_provoking", TestFlatProvokingVertex},
    {"cpuref_gouraud", TestGouraudInterpolation},
    {"cpuref_constant_attributes", TestConstantAttributes},
    {"cpuref_sprite_rectangle", TestSpriteRectangle},
    {"cpuref_scissor", TestScissorClip},
    {"cpuref_depth", TestDepthTestAndWrite},
    {"cpuref_texture_nearest_decal", TestTextureNearestDecal},
    {"cpuref_clut_indexed", TestClutIndexedLookup},
    {"cpuref_blend_sweep", TestBlendSweepFullPath},
    {"cpuref_in07_aem", TestAemSingleRule},
    {"cpuref_capture_replay", TestSyntheticCaptureReplay},
    {"cpuref_ppm", TestPpmEncoding},
    {"cpuref_display_frame", TestDisplayFrameFromRegisters},
};

}  // namespace

int main(int argc, char** argv) {
    const std::string name = argc > 1 ? argv[1] : "";
    for (const Test& test : kTests) {
        if (name == test.name) {
            test.run();
            if (g_failures != 0) {
                std::fprintf(stderr, "%s: %d failure(s)\n", test.name, g_failures);
                return 1;
            }
            std::printf("%s: ok\n", test.name);
            return 0;
        }
    }
    std::fprintf(stderr, "unknown test '%s'\n", name.c_str());
    return 2;
}
