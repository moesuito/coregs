// SPDX-License-Identifier: Apache-2.0
//
// CTest fixtures for the HWR per-draw decisions (hw_gs_decide.hpp). Copyright (c) 2026
// GT4Recomp contributors.
//
// Written from the TASK-026 specification (docs/hw-renderer-spec): vectors of chapter 03
// (draw rectangle table, 1x coverage edges, flat shading, SCANMSK), 05.8 (double-half and
// striped clears, target and memory clears), 08 (blend table 08.1, group counts over the 81
// modes, alpha-test folding, DATE selection, V-08-1), 09 (V-09-1, channel shuffles, the game
// patterns) and 14.6. The pixel arithmetic reference (src/pixel) is the oracle for
// blend, fold and dither results.
//
// Usage: hw_gs_decide_fixture [test-name ...]. With no arguments every test runs.

#include "hw_gs_decide.hpp"
#include "hw_gs_pixel.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

using namespace coregs::decide;
namespace px = coregs::pixel;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        ++g_checks;                                                                        \
        if (!(cond)) {                                                                     \
            ++g_failures;                                                                  \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
        }                                                                                  \
    } while (0)

#define CHECK_EQ(actual, expected)                                                         \
    do {                                                                                   \
        ++g_checks;                                                                        \
        const long long _actual = static_cast<long long>(actual);                          \
        const long long _expected = static_cast<long long>(expected);                      \
        if (_actual != _expected) {                                                        \
            ++g_failures;                                                                  \
            std::fprintf(stderr, "%s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__,  \
                         #actual, _actual, _expected);                                     \
        }                                                                                  \
    } while (0)

// ---- Helpers -------------------------------------------------------------------------------

RegisterSnapshot base_regs() {
    RegisterSnapshot r;
    r.frame = FrameFormat::Ct32;
    r.fbp = 0;
    r.fbw = 10;
    r.zbp = 70;
    r.z_format = ZFormat::Z32;
    r.zte = true;
    r.ztst = DepthCompare::Always;
    r.scax0 = 0;
    r.scax1 = 639;
    r.scay0 = 0;
    r.scay1 = 479;
    r.tw_log2 = 6;
    r.th_log2 = 6;
    return r;
}

Vertex2D vertex(double x, double y, std::uint8_t rgba = 0, std::uint32_t z = 0) {
    Vertex2D v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.rgba = {rgba, rgba, rgba, rgba};
    v.u = 0.0;
    v.v = 0.0;
    v.q = 1.0;
    return v;
}

// A sprite from (x0, y0) to (x1, y1) with constant colour and Z.
BatchGeometry sprite(double x0, double y0, double x1, double y1, std::uint8_t rgba = 0, std::uint32_t z = 0) {
    BatchGeometry g;
    g.prim = PrimClass::Sprite;
    g.vertices = {vertex(x0, y0, rgba, z), vertex(x1, y1, rgba, z)};
    g.indices = {0, 1};
    return g;
}

BatchInput make_input(const RegisterSnapshot& regs, const BatchGeometry& geometry) {
    BatchInput in;
    in.regs = regs;
    in.geometry = geometry;
    in.fragment_alpha_min = 0x80;
    in.fragment_alpha_max = 0x80;
    return in;
}

DeviceCaps no_caps() { return DeviceCaps{}; }

DeviceCaps fetch_caps() {
    DeviceCaps c;
    c.raster_order_access = true;
    return c;
}

Options default_options() { return Options{}; }

px::BlendState to_pixel_state(BlendMode m, std::uint8_t fix) {
    px::BlendState s;
    s.enabled = true;
    s.a = static_cast<px::BlendSource>(static_cast<int>(m.a));
    s.b = static_cast<px::BlendSource>(static_cast<int>(m.b));
    s.d = static_cast<px::BlendSource>(static_cast<int>(m.d));
    s.c = static_cast<px::BlendFactor>(static_cast<int>(m.c));
    s.fix = fix;
    s.colour_clamp = false;
    return s;
}

// ---- Chapter 03: draw rectangle, coverage, flat shading, SCANMSK -------------------------

void test_draw_rect_table() {
    // Normative table of chapter 03.1 step 9: range -> pixels (first..last).
    struct Row {
        double min, max;
        int first, last;
    };
    const Row rows[] = {
        {-0.5, 0.0, 0, 0}, {-0.5, 0.5, 0, 0}, {0.0, 1.0, 0, 0}, {0.0, 1.5, 0, 1},
        {0.5, 1.5, 1, 1}, {0.5, 1.75, 1, 1}, {0.5, 2.25, 1, 1}, {0.5, 2.5, 1, 2},
    };
    for (const Row& row : rows) {
        const PixelRect rect = draw_rectangle_from_span(row.min, row.max);
        CHECK_EQ(rect.x0, row.first);
        CHECK_EQ(rect.x1 - 1, row.last);
    }
    // Scissor clamps the rectangle; an empty intersection stays empty.
    RegisterSnapshot regs = base_regs();
    regs.scax0 = 4;
    regs.scax1 = 9;
    VertexTrace trace;
    trace.x_min = 0.0;
    trace.x_max = 20.0;
    trace.y_min = 0.0;
    trace.y_max = 10.0;
    const PixelRect clipped = draw_rectangle(trace, regs);
    CHECK_EQ(clipped.x0, 4);
    CHECK_EQ(clipped.x1, 10);
    regs.scax0 = 100;
    regs.scax1 = 110;
    CHECK(draw_rectangle(trace, regs).empty());
}

void test_coverage_1x() {
    // Chapter 03.3 vectors at 1x, OFX = 0: edge range -> covered pixels.
    const PixelRect a = coverage_pixels_1x(0.0, 2.0);
    CHECK_EQ(a.x0, 0);
    CHECK_EQ(a.x1, 2);
    const PixelRect b = coverage_pixels_1x(0.5, 1.5);
    CHECK_EQ(b.x0, 1);
    CHECK_EQ(b.x1, 2);
    const PixelRect c = coverage_pixels_1x(0.0625, 1.0);
    CHECK(c.x1 <= c.x0);
}

void test_scanmsk() {
    // V-03-1: at 2x, SCANMSK counts native rows (row = floor(gpu_y / 2)).
    CHECK(scanmsk_discards_row(2, 0, 2));
    CHECK(scanmsk_discards_row(2, 1, 2));
    CHECK(!scanmsk_discards_row(2, 2, 2));
    CHECK(!scanmsk_discards_row(2, 3, 2));
    CHECK(scanmsk_discards_row(2, 4, 2));
    CHECK(!scanmsk_discards_row(3, 0, 2));
    CHECK(scanmsk_discards_row(3, 2, 2));
    CHECK(scanmsk_discards_row(3, 3, 2));
    CHECK(!scanmsk_discards_row(0, 2, 2));
}

void test_primitive_expansion() {
    DeviceCaps caps;
    caps.max_point_size = 1.0f;
    caps.max_line_width = 1.0f;
    CHECK(decide_primitive(PrimClass::Point, 1, true, false, false, caps).expansion == Expansion::None);
    CHECK(decide_primitive(PrimClass::Point, 2, true, false, false, caps).expansion == Expansion::PointQuad);
    caps.max_point_size = 8.0f;
    CHECK(decide_primitive(PrimClass::Point, 2, true, false, false, caps).expansion == Expansion::PointSize);
    CHECK(decide_primitive(PrimClass::Line, 2, true, false, false, caps).expansion == Expansion::LineQuad);
    caps.wide_lines = true;
    caps.max_line_width = 8.0f;
    CHECK(decide_primitive(PrimClass::Line, 2, true, false, false, caps).expansion == Expansion::LineWidth);
    CHECK(decide_primitive(PrimClass::Sprite, 1, true, false, true, caps).expansion == Expansion::SpriteCpu);
    CHECK(decide_primitive(PrimClass::Sprite, 1, true, false, false, caps).expansion == Expansion::SpriteVertexStage);
    // Flat shading (3.2): varying-colour triangles without provoking-vertex support de-index.
    CHECK(decide_primitive(PrimClass::Triangle, 1, false, false, false, caps).flat_deindex);
    CHECK(!decide_primitive(PrimClass::Triangle, 1, false, true, false, caps).flat_deindex);
    CHECK(!decide_primitive(PrimClass::Sprite, 1, false, false, false, caps).flat_deindex);
    DeviceCaps provoking = caps;
    provoking.provoking_vertex_last = true;
    CHECK(!decide_primitive(PrimClass::Triangle, 1, false, false, false, provoking).flat_deindex);
}

void test_vertex_trace() {
    BatchGeometry g;
    g.prim = PrimClass::Sprite;
    g.vertices = {vertex(10, 20, 5, 100), vertex(30, 60, 200, 300)};
    g.vertices[1].rgba = {200, 10, 10, 10};
    g.indices = {0, 1};
    RegisterSnapshot regs = base_regs();
    regs.fst = true;
    const VertexTrace t = compute_vertex_trace(g, regs);
    CHECK_EQ(t.x_min, 10);
    CHECK_EQ(t.x_max, 30);
    CHECK_EQ(t.z_min, 100);
    CHECK_EQ(t.z_max, 300);
    CHECK(!t.colour_constant);
    CHECK(!t.z_constant);
    CHECK(!t.unsafe_q);
    // STQ with a vanishing Q is unsafe (chapter 02.9).
    BatchGeometry q = sprite(0, 0, 8, 8);
    q.vertices[0].s = 1.0;
    q.vertices[0].q = 1e-40;
    q.vertices[1].s = 1.0;
    q.vertices[1].q = 1e-40;
    CHECK(compute_vertex_trace(q, base_regs()).unsafe_q);
    CHECK(!compute_vertex_trace(sprite(0, 0, 8, 8), base_regs()).unsafe_q);
}

void test_overlap_groups() {
    // Three disjoint sprites, then one that overlaps the first and the second: two groups (2.10).
    BatchGeometry g;
    g.prim = PrimClass::Sprite;
    for (int i = 0; i < 3; ++i) {
        g.vertices.push_back(vertex(i * 8, 0));
        g.vertices.push_back(vertex(i * 8 + 8, 8));
        g.indices.push_back(static_cast<std::uint32_t>(2 * i));
        g.indices.push_back(static_cast<std::uint32_t>(2 * i + 1));
    }
    const OverlapResult none = classify_overlap(g, compute_vertex_trace(g, base_regs()));
    CHECK(none.kind == OverlapKind::None);
    CHECK_EQ(none.groups, 1);

    g.vertices.push_back(vertex(4, 4));
    g.vertices.push_back(vertex(12, 12));
    g.indices.push_back(6);
    g.indices.push_back(7);
    const OverlapResult yes = classify_overlap(g, compute_vertex_trace(g, base_regs()));
    CHECK(yes.kind == OverlapKind::Yes);
    CHECK_EQ(yes.groups, 2);
    CHECK_EQ(yes.draw_list.size(), 2);
    CHECK_EQ(yes.draw_list[0].primitive_count, 3);
    CHECK_EQ(yes.draw_list[1].primitive_count, 1);

    // Triangles: a quad pair is exact; a single triangle references 3 vertices; lines are unknown.
    BatchGeometry quad;
    quad.prim = PrimClass::Triangle;
    quad.vertices = {vertex(0, 0), vertex(8, 0), vertex(8, 8), vertex(0, 8)};
    quad.indices = {0, 1, 2, 0, 2, 3};
    CHECK(classify_overlap(quad, compute_vertex_trace(quad, base_regs())).kind == OverlapKind::None);
    BatchGeometry tri = quad;
    tri.indices = {0, 1, 2};
    CHECK(classify_overlap(tri, compute_vertex_trace(tri, base_regs())).kind == OverlapKind::None);
    BatchGeometry lines;
    lines.prim = PrimClass::Line;
    lines.vertices = {vertex(0, 0), vertex(8, 0), vertex(8, 8), vertex(0, 8)};
    lines.indices = {0, 1, 2, 3};
    CHECK(classify_overlap(lines, compute_vertex_trace(lines, base_regs())).kind == OverlapKind::Unknown);
}

// ---- Chapter 08: blend ------------------------------------------------------------------

void test_blend_classes() {
    std::map<BlendGroup, int> counts;
    for (int i = 0; i < 81; ++i) {
        const BlendMode m = BlendMode::from_index(i);
        CHECK_EQ(m.index(), i);
        ++counts[classify_blend(m)];
    }
    CHECK_EQ(counts[BlendGroup::SameAsD], 27);
    CHECK_EQ(counts[BlendGroup::NonRecursive], 8);
    CHECK_EQ(counts[BlendGroup::Accumulate], 6);
    CHECK_EQ(counts[BlendGroup::Interpolate], 6);
    CHECK_EQ(counts[BlendGroup::Subtract], 6);
    CHECK_EQ(counts[BlendGroup::Impossible], 7);
    CHECK_EQ(counts[BlendGroup::CdMultiplied], 18);
    CHECK_EQ(counts[BlendGroup::SourceTimesAd], 3);
    CHECK(blend_group_reads_destination(BlendGroup::Accumulate));
    CHECK(!blend_group_reads_destination(BlendGroup::NonRecursive));
    CHECK(!blend_group_reads_destination(BlendGroup::SameAsD));
}

void test_blend_vectors() {
    // Table 08.1 with the pixel reference as the oracle. BlendMode order: a, b, d, c.
    // Vector 1: (Cs - Cd) * As + Cd, Cs 200, Cd 100, As 0x40 -> 150.
    const BlendMode m1{BlendSel::Cs, BlendSel::Cd, BlendSel::Cd, BlendFactorSel::As};
    CHECK_EQ(px::blend_unclamped(to_pixel_state(m1, 0), px::Colour{200, 0, 0, 0}, px::Colour{100, 0, 0, 0}, 0x40, 0).r, 150);
    CHECK(classify_blend(m1) == BlendGroup::Interpolate);

    // Vector 2: Cs * As, Cs 200, As 0xFF -> 398, clamp 255, wrap 142.
    const BlendMode m2{BlendSel::Cs, BlendSel::Zero, BlendSel::Zero, BlendFactorSel::As};
    const int v2 = px::blend_unclamped(to_pixel_state(m2, 0), px::Colour{200, 0, 0, 0}, px::Colour{}, 0xFF, 0).r;
    CHECK_EQ(v2, 398);
    CHECK_EQ(px::colour_clamp_or_wrap(v2, true), 255);
    CHECK_EQ(px::colour_clamp_or_wrap(v2, false), 142);
    CHECK(classify_blend(m2) == BlendGroup::NonRecursive);

    // Vector 3: (Cd - Cs) * As, Cd 10, Cs 20, As 0x80 -> -10, clamp 0, wrap 246.
    const BlendMode m3{BlendSel::Cd, BlendSel::Cs, BlendSel::Zero, BlendFactorSel::As};
    const int v3 = px::blend_unclamped(to_pixel_state(m3, 0), px::Colour{20, 0, 0, 0}, px::Colour{10, 0, 0, 0}, 0x80, 0).r;
    CHECK_EQ(v3, -10);
    CHECK_EQ(px::colour_clamp_or_wrap(v3, true), 0);
    CHECK_EQ(px::colour_clamp_or_wrap(v3, false), 246);
    CHECK(classify_blend(m3) == BlendGroup::Subtract);

    // Vector 4: (Cd - Cs) * FIX, Cd 0, Cs 1, FIX 0x81 -> -2 (arithmetic shift), clamp 0, wrap 254.
    const BlendMode m4{BlendSel::Cd, BlendSel::Cs, BlendSel::Zero, BlendFactorSel::Fix};
    const int v4 = px::blend_unclamped(to_pixel_state(m4, 0x81), px::Colour{1, 0, 0, 0}, px::Colour{0, 0, 0, 0}, 0, 0).r;
    CHECK_EQ(v4, -2);
    CHECK_EQ(px::colour_clamp_or_wrap(v4, true), 0);
    CHECK_EQ(px::colour_clamp_or_wrap(v4, false), 254);

    // Vector 5: Cs * As with Cs 1 and As 0xFF -> 1 (truncation, not UNORM rounding).
    CHECK_EQ(px::blend_unclamped(to_pixel_state(m2, 0), px::Colour{1, 0, 0, 0}, px::Colour{}, 0xFF, 0).r, 1);

    // Vector 6: 16-bit frame, DIMX -4, blended 0x13 -> 0x08 (dither, then the 0xF8 truncation).
    const int dithered = px::apply_dither(px::Colour{0x13, 0, 0, 0}, -4).r;
    CHECK_EQ(dithered & 0xF8, 0x08);
}

// Every mode, simplified by chapter 08.2, must evaluate identically to the original mode.
void test_blend_simplify() {
    const std::uint8_t values[] = {0, 1, 0x40, 0x80, 0xFF};
    const int cs_values[] = {0, 1, 100, 200, 255};
    int checked = 0;
    for (int i = 0; i < 81; ++i) {
        const BlendMode original = BlendMode::from_index(i);
        for (std::uint8_t v : values) {
            for (int cs : cs_values) {
                for (int cd : cs_values) {
                    RegisterSnapshot regs = base_regs();
                    regs.abe = true;
                    regs.blend = original;
                    regs.fix = v;
                    regs.colclamp = false;
                    BatchInput in = make_input(regs, sprite(0, 0, 8, 8));
                    in.fragment_alpha_min = in.fragment_alpha_max = v;
                    in.target.alpha_min = in.target.alpha_max = v;
                    const BlendPlan plan = decide_blend(regs, in, default_options(), v, v, false,
                                                        OverlapKind::None, 1, false, false);
                    const px::Colour src{cs, 0, 0, 0};
                    const px::Colour dst{cd, 0, 0, 0};
                    const int before = px::blend_unclamped(to_pixel_state(original, v), src, dst, v, v).r;
                    const int after = px::blend_unclamped(to_pixel_state(plan.mode, plan.effective_fix), src, dst, v, v).r;
                    CHECK_EQ(after, before);
                    ++checked;
                }
            }
        }
    }
    CHECK_EQ(checked, 81 * 5 * 5 * 5);
}

// Method invariants over the 81 modes, the four accuracy levels and the frame and option flags.
void test_blend_levels() {
    std::map<BlendLevel, int> shader_count;
    const BlendLevel levels[] = {BlendLevel::Basic, BlendLevel::Medium, BlendLevel::High, BlendLevel::Full};
    int cases = 0;
    for (int i = 0; i < 81; ++i) {
        const BlendMode mode = BlendMode::from_index(i);
        const BlendGroup group = classify_blend(mode);
        for (FrameFormat frame : {FrameFormat::Ct32, FrameFormat::Ct16}) {
            for (bool dthe : {false, true}) {
                for (bool pabe : {false, true}) {
                    for (bool colclamp : {true, false}) {
                        for (std::uint8_t frag_max : {std::uint8_t{0x40}, std::uint8_t{0xFF}}) {
                            for (OverlapKind overlap : {OverlapKind::None, OverlapKind::Yes}) {
                                for (bool barrier : {false, true}) {
                                    bool prev_shader = false;
                                    for (BlendLevel level : levels) {
                                        RegisterSnapshot regs = base_regs();
                                        regs.abe = true;
                                        regs.pabe = pabe;
                                        regs.frame = frame;
                                        regs.dthe = dthe;
                                        regs.colclamp = colclamp;
                                        regs.blend = mode;
                                        regs.fix = 0x90;
                                        const BatchInput in = make_input(regs, sprite(0, 0, 8, 8));
                                        Options opt;
                                        opt.blend_level = level;
                                        const BlendPlan p = decide_blend(regs, in, opt, 0, frag_max, false, overlap,
                                                                         1, false, barrier);
                                        ++cases;
                                        const bool reads = blend_group_reads_destination(group);
                                        const bool shader = p.method == BlendMethod::Shader;
                                        if (shader) ++shader_count[level];

                                        // Impossible and non-recursive modes always use the shader.
                                        if (group == BlendGroup::Impossible || group == BlendGroup::NonRecursive) {
                                            CHECK(shader);
                                        }
                                        // PABE forces the shader.
                                        if (pabe) CHECK(shader);
                                        // A = B keeps the fixed function (exact, no destination read).
                                        if (!pabe && group == BlendGroup::SameAsD) {
                                            CHECK(p.method == BlendMethod::FixedFunction);
                                            CHECK(!p.reads_destination);
                                        }
                                        // Mixed is a basic-only method.
                                        if (p.method == BlendMethod::Mixed) CHECK(level == BlendLevel::Basic);
                                        // Exact accumulate: 32-bit, clamped, not an Ad factor, not PABE.
                                        const bool exact_acc = group == BlendGroup::Accumulate && !pabe &&
                                                               mode.c != BlendFactorSel::Ad && colclamp &&
                                                               frame != FrameFormat::Ct16;
                                        if (exact_acc) CHECK(p.method == BlendMethod::Accumulate);
                                        // 16-bit destination readers never stay on the fixed function from medium up.
                                        if (level != BlendLevel::Basic && frame == FrameFormat::Ct16 && reads && !pabe) {
                                            CHECK(shader);
                                        }
                                        // Full: every destination reader that is not exact accumulate uses the shader.
                                        if (level == BlendLevel::Full && reads && !pabe && !exact_acc) CHECK(shader);
                                        // Dithered 16-bit blending always uses the shader (08.10).
                                        if (dthe && frame == FrameFormat::Ct16 && !pabe && reads) CHECK(shader);
                                        // The shader set only grows with the level.
                                        if (prev_shader) CHECK(shader);
                                        prev_shader = shader;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    CHECK(cases == 81 * 128 * 4);
    CHECK(shader_count[BlendLevel::Full] >= shader_count[BlendLevel::High]);
    CHECK(shader_count[BlendLevel::High] >= shader_count[BlendLevel::Medium]);
    CHECK(shader_count[BlendLevel::Medium] >= shader_count[BlendLevel::Basic]);
}

void test_colour_wrap() {
    BlendPlan ff;
    ff.method = BlendMethod::FixedFunction;
    ff.group = BlendGroup::Accumulate;
    ff.reads_destination = true;
    CHECK(decide_colour_wrap(true, ff, FrameFormat::Ct32, BlendLevel::Basic, OverlapKind::None, true) ==
          ColourWrapPlan::NotNeeded);
    CHECK(decide_colour_wrap(false, ff, FrameFormat::Ct32, BlendLevel::Basic, OverlapKind::None, false) ==
          ColourWrapPlan::NotNeeded);
    CHECK(decide_colour_wrap(false, ff, FrameFormat::Ct32, BlendLevel::Basic, OverlapKind::None, true) ==
          ColourWrapPlan::SideTarget);
    CHECK(decide_colour_wrap(false, ff, FrameFormat::Ct32, BlendLevel::Medium, OverlapKind::None, true) ==
          ColourWrapPlan::SideTarget);
    CHECK(decide_colour_wrap(false, ff, FrameFormat::Ct32, BlendLevel::High, OverlapKind::None, true) ==
          ColourWrapPlan::ShaderFullResult);
    BlendPlan sh = ff;
    sh.method = BlendMethod::Shader;
    CHECK(decide_colour_wrap(false, sh, FrameFormat::Ct32, BlendLevel::Basic, OverlapKind::None, false) ==
          ColourWrapPlan::ShaderFullResult);
}

void test_channel_writes() {
    const ChannelWrites all = channel_writes(FrameFormat::Ct32, 0);
    CHECK(all.rgb_any && all.alpha_any && !all.any_partial);
    CHECK(all.channel[0] == ChannelState::Normal && all.channel[3] == ChannelState::Normal);
    const ChannelWrites none = channel_writes(FrameFormat::Ct32, 0xFFFFFFFFu);
    CHECK(!none.rgb_any && !none.alpha_any);
    // Variant B mask (chapter 09.4): only alpha written.
    const ChannelWrites alpha_only = channel_writes(FrameFormat::Ct32, 0x00FFFFFFu);
    CHECK(!alpha_only.rgb_any && alpha_only.alpha_any);
    // CT24 has no alpha lane.
    const ChannelWrites ct24 = channel_writes(FrameFormat::Ct24, 0);
    CHECK(ct24.rgb_any && !ct24.alpha_any);
    // CT16: mask bits outside the valid bits do not make a channel partial.
    CHECK(!channel_writes(FrameFormat::Ct16, 0x7u).any_partial);
    CHECK(channel_writes(FrameFormat::Ct16, 0x10u).any_partial);
    CHECK(channel_writes(FrameFormat::Ct16, 0xF8u).channel[0] == ChannelState::Off);
    CHECK(channel_writes(FrameFormat::Ct32, 0x10u).channel[0] == ChannelState::Partial);
}

// ---- Chapter 08.6 and 08.7: alpha test and DATE -------------------------------------------

void test_atest_fold() {
    const AlphaCompare compares[] = {AlphaCompare::Never,     AlphaCompare::Always,     AlphaCompare::Less,
                                     AlphaCompare::LessEqual, AlphaCompare::Equal,      AlphaCompare::GreaterEqual,
                                     AlphaCompare::Greater,   AlphaCompare::NotEqual};
    const std::uint8_t values[] = {0, 1, 127, 128, 200, 255};
    for (AlphaCompare c : compares) {
        for (std::uint8_t ref : values) {
            for (std::uint8_t lo : values) {
                for (std::uint8_t hi : values) {
                    if (lo > hi) continue;
                    const AlphaFold ours = fold_alpha_test(c, lo, hi, ref);
                    const px::AlphaFold theirs =
                        px::fold_alpha_test(static_cast<px::AlphaCompare>(static_cast<int>(c)), lo, hi, ref);
                    CHECK_EQ(static_cast<int>(ours), static_cast<int>(theirs));
                }
            }
        }
    }
    // Explicit rows of the table in 08.6.
    CHECK(fold_alpha_test(AlphaCompare::Less, 0, 127, 128) == AlphaFold::AllPass);
    CHECK(fold_alpha_test(AlphaCompare::Less, 128, 255, 128) == AlphaFold::AllFail);
    CHECK(fold_alpha_test(AlphaCompare::Equal, 9, 9, 9) == AlphaFold::AllPass);
    CHECK(fold_alpha_test(AlphaCompare::NotEqual, 10, 255, 9) == AlphaFold::AllPass);
    CHECK(fold_alpha_test(AlphaCompare::NotEqual, 9, 9, 9) == AlphaFold::AllFail);
    CHECK(fold_alpha_test(AlphaCompare::Greater, 0, 10, 10) == AlphaFold::AllFail);
}

void test_atest_methods() {
    RegisterSnapshot r = base_regs();
    r.ate = true;
    r.aref = 0x80;
    r.atst = AlphaCompare::GreaterEqual;
    r.afail = AlphaFail::FrameOnly;
    const WriteMask all{true, true, true};
    AlphaTestContext ctx;

    // Mixed range, FB_ONLY, depth written, constant Z: split (depth is independent of colour).
    const AlphaTestPlan split = decide_alpha_test(r, 0x10, 0xF0, all, ctx);
    CHECK(split.method == AlphaMethod::SplitFrameOnly);
    CHECK(split.fold == AlphaFold::Mixed);
    CHECK_EQ(split.passes.size(), 2);
    CHECK(split.passes[0].writes.rgb && split.passes[0].writes.alpha && !split.passes[0].writes.depth);
    CHECK(split.passes[1].writes.depth && !split.passes[1].writes.rgb && split.passes[1].discard);

    // Framebuffer fetch and depth feedback make the one-pass feedback choice.
    AlphaTestContext fetch = ctx;
    fetch.framebuffer_fetch = true;
    fetch.depth_feedback = true;
    CHECK(decide_alpha_test(r, 0x10, 0xF0, all, fetch).method == AlphaMethod::Feedback);
    // Without depth feedback, FB_ONLY with Z is refused and the split stays.
    AlphaTestContext no_depth_fb = fetch;
    no_depth_fb.depth_feedback = false;
    CHECK(decide_alpha_test(r, 0x10, 0xF0, all, no_depth_fb).method == AlphaMethod::SplitFrameOnly);
    // The accurate option forces feedback (COREGS_HW_ACCURATE_ATEST=1).
    AlphaTestContext accurate = ctx;
    accurate.accurate_option = true;
    accurate.depth_feedback = true;
    CHECK(decide_alpha_test(r, 0x10, 0xF0, all, accurate).method == AlphaMethod::Feedback);

    // ZB_ONLY is not split: pass-then-fail with the inverted test for depth.
    RegisterSnapshot zb = r;
    zb.afail = AlphaFail::DepthOnly;
    const AlphaTestPlan ptf = decide_alpha_test(zb, 0x10, 0xF0, all, ctx);
    CHECK(ptf.method == AlphaMethod::PassThenFail);
    CHECK_EQ(ptf.passes.size(), 2);
    CHECK(ptf.passes[1].compare == AlphaCompare::Less);
    CHECK(ptf.passes[1].writes.depth && !ptf.passes[1].writes.rgb);

    // RGB_ONLY with independent RGB and Z: split with dual-source alpha.
    RegisterSnapshot rgb = r;
    rgb.afail = AlphaFail::RgbOnly;
    const AlphaTestPlan rgb_plan = decide_alpha_test(rgb, 0x10, 0xF0, all, ctx);
    CHECK(rgb_plan.method == AlphaMethod::SplitRgbOnly);
    CHECK(rgb_plan.passes[0].dual_source_alpha);
    // RGB_ONLY on a 16-bit frame behaves as FB_ONLY.
    RegisterSnapshot rgb16 = rgb;
    rgb16.frame = FrameFormat::Ct16;
    CHECK(decide_alpha_test(rgb16, 0x10, 0xF0, all, ctx).method == AlphaMethod::SplitFrameOnly);
    // RGB_ONLY with alpha masked behaves as FB_ONLY.
    CHECK(decide_alpha_test(rgb, 0x10, 0xF0, WriteMask{true, false, true}, ctx).method == AlphaMethod::SplitFrameOnly);

    // Keep: failing fragments are discarded.
    RegisterSnapshot keep = r;
    keep.afail = AlphaFail::Keep;
    CHECK(decide_alpha_test(keep, 0x10, 0xF0, all, ctx).method == AlphaMethod::Keep);

    // Folding: all pass drops the test; all fail with FB_ONLY keeps colour only.
    RegisterSnapshot always = r;
    always.atst = AlphaCompare::Always;
    CHECK(decide_alpha_test(always, 0x10, 0xF0, all, ctx).method == AlphaMethod::None);
    RegisterSnapshot never = r;
    never.atst = AlphaCompare::Never;
    const AlphaTestPlan folded = decide_alpha_test(never, 0x10, 0xF0, all, ctx);
    CHECK(folded.method == AlphaMethod::Never);
    CHECK(folded.writes.rgb && folded.writes.alpha && !folded.writes.depth);

    // AFAIL channels already masked: the test becomes KEEP.
    RegisterSnapshot masked = r;
    masked.afail = AlphaFail::DepthOnly;
    CHECK(decide_alpha_test(masked, 0x10, 0xF0, WriteMask{true, true, false}, ctx).method == AlphaMethod::Keep);
}

void test_date_methods() {
    RegisterSnapshot r = base_regs();
    r.date = true;
    DateContext ctx;
    ctx.alpha_written = true;
    // CT24 with DATE set writes nothing (V-01-8).
    CHECK(decide_date(true, false, FrameFormat::Ct24, 0, 255, ctx).outcome == DateOutcome::SkipAllFail);
    CHECK(decide_date(false, false, FrameFormat::Ct32, 0, 255, ctx).outcome == DateOutcome::NoTest);

    // DATM = 0: every pixel already >= 0x80 fails; a target below 0x80 passes.
    CHECK(decide_date(true, false, FrameFormat::Ct32, 0x80, 0xFF, ctx).outcome == DateOutcome::SkipAllFail);
    CHECK(decide_date(true, false, FrameFormat::Ct32, 0x00, 0x7F, ctx).outcome == DateOutcome::NoTest);
    DateContext raise = ctx;
    raise.overlap = OverlapKind::Yes;
    raise.written_min = 0x90;
    raise.written_max = 0xC0;
    CHECK(decide_date(true, false, FrameFormat::Ct32, 0x00, 0x7F, raise).outcome == DateOutcome::Test);
    // DATM = 1: a target below 0x80 everywhere fails; at or above 0x80 everywhere passes.
    CHECK(decide_date(true, true, FrameFormat::Ct32, 0x00, 0x7F, ctx).outcome == DateOutcome::SkipAllFail);
    CHECK(decide_date(true, true, FrameFormat::Ct32, 0x80, 0xFF, ctx).outcome == DateOutcome::NoTest);

    // Methods (08.7 selection order).
    DateContext no_overlap = ctx;
    CHECK(decide_date(true, false, FrameFormat::Ct32, 0x00, 0xFF, no_overlap).method == DateMethod::Full);
    DateContext masked;
    masked.overlap = OverlapKind::Yes;
    masked.alpha_written = false;
    masked.draw_list_groups = 2;
    CHECK(decide_date(true, false, FrameFormat::Ct32, 0x00, 0xFF, masked).method == DateMethod::Stencil);
    DateContext busy;
    busy.overlap = OverlapKind::Yes;
    busy.alpha_written = true;
    busy.written_min = 0x10;
    busy.written_max = 0x20;
    busy.draw_list_groups = 12;
    busy.index_count = 60;
    busy.prim_id_available = true;
    CHECK(decide_date(true, false, FrameFormat::Ct32, 0x00, 0xFF, busy).method == DateMethod::PrimId);
    busy.prim_id_available = false;
    CHECK(decide_date(true, false, FrameFormat::Ct32, 0x00, 0xFF, busy).method == DateMethod::Full);
    DateContext shuffle = busy;
    shuffle.shuffle = true;
    const DatePlan sp = decide_date(true, false, FrameFormat::Ct32, 0x00, 0xFF, shuffle);
    CHECK(sp.method == DateMethod::Full);
    CHECK(sp.reads_green);
}

// V-08-1: DATM = 0, two overlapping sprites, the first writes alpha 0x80. The second sprite is
// rejected where they overlap. Plain STENCIL is wrong by design; the selection must not pick it.
void test_date_vectors() {
    RegisterSnapshot r = base_regs();
    r.date = true;
    r.datm = false;
    BatchGeometry g;
    g.prim = PrimClass::Sprite;
    g.vertices = {vertex(0, 0), vertex(16, 16), vertex(8, 8), vertex(24, 24)};
    g.indices = {0, 1, 2, 3};
    BatchInput in = make_input(r, g);
    in.target.alpha_min = 0x00;
    in.target.alpha_max = 0xFF;
    in.fragment_alpha_min = 0x80;
    in.fragment_alpha_max = 0x80;
    const DrawPlan plan = decide_draw(in, no_caps(), default_options());
    CHECK(plan.overlap.kind == OverlapKind::Yes);
    CHECK(plan.date.outcome == DateOutcome::Test);
    CHECK(plan.date.method != DateMethod::Stencil);
    CHECK(plan.date.method == DateMethod::StencilOne || plan.date.method == DateMethod::Full ||
          plan.date.method == DateMethod::PrimId);
}

void test_barrier_rules() {
    CHECK(decide_barrier(false, true, OverlapKind::Yes, 4, false) == BarrierKind::None);
    CHECK(decide_barrier(true, false, OverlapKind::Yes, 4, false) == BarrierKind::One);
    CHECK(decide_barrier(true, true, OverlapKind::None, 1, false) == BarrierKind::One);
    CHECK(decide_barrier(true, true, OverlapKind::Yes, 4, false) == BarrierKind::PerDrawListGroup);
    CHECK(decide_barrier(true, true, OverlapKind::Yes, 4, true) == BarrierKind::One);
    CHECK(decide_barrier(true, true, OverlapKind::Unknown, 1, false) == BarrierKind::PerPrimitive);
}

// ---- Chapter 09 and 05.8 -----------------------------------------------------------------------

void test_texture_shuffle() {
    // V-09-1: 16-bit sprite 16 wide, u = x + 8, FBMSK 0x3FFF (only G and A written): dest A = source G.
    ShuffleSample s;
    s.frame_16bit = true;
    s.frame_x0 = 0;
    s.frame_x1 = 16;
    s.tex_u0 = 8;
    s.tex_u1 = 24;
    s.width_pixels = 16;
    s.fbmsk = 0x3FFF;
    s.source_is_32bit = true;
    s.opaque = true;
    CHECK(classify_texture_shuffle(s) == ShuffleKind::Offset);
    const ShuffleChannelSet set = texture_shuffle_channel_set(0x3FFFu);
    CHECK(!set.rg && set.ba);
    CHECK_EQ(texture_shuffle_channel_source(ShuffleKind::Offset, 0x3FFFu)[3], 1);

    // Copy: offset 0, the next quad adjacent, low-ones FBMSK, texture is not the frame.
    ShuffleSample copy = s;
    copy.tex_u0 = 0;
    copy.tex_u1 = 16;
    copy.next_quad_adjacent = true;
    copy.fbmsk = 0xFF00u;
    CHECK(classify_texture_shuffle(copy) == ShuffleKind::Copy);

    // Reverse: offset 0, width 16, texel u increasing while x decreases.
    ShuffleSample rev = s;
    rev.tex_u0 = 0;
    rev.tex_u1 = 16;
    rev.frame_x0 = 16;
    rev.frame_x1 = 0;
    CHECK(classify_texture_shuffle(rev) == ShuffleKind::Reverse);

    // Swizzle: 8 wide with swapped axes.
    ShuffleSample sw = s;
    sw.width_pixels = 8;
    sw.frame_x1 = 8;
    sw.tex_u0 = 3;
    sw.tex_u1 = 11;
    sw.axes_swapped = true;
    CHECK(classify_texture_shuffle(sw) == ShuffleKind::Swizzle);

    // Rejections: 32-bit frame, a non-32-bit source, width not a multiple of 8.
    ShuffleSample wide = s;
    wide.frame_16bit = false;
    CHECK(classify_texture_shuffle(wide) == ShuffleKind::None);
    ShuffleSample real16 = s;
    real16.source_is_32bit = false;
    CHECK(classify_texture_shuffle(real16) == ShuffleKind::None);
    ShuffleSample odd = s;
    odd.width_pixels = 12;
    CHECK(classify_texture_shuffle(odd) == ShuffleKind::None);
}

void test_channel_shuffle() {
    ChannelCandidate c;
    c.tme = true;
    c.sprite = true;
    c.t8_format = true;
    c.one_page = true;
    c.first_sprite8 = true;
    CHECK(is_channel_shuffle_candidate(c));
    c.t8_format = false;
    CHECK(!is_channel_shuffle_candidate(c));
    c.t8_format = true;
    c.tiny = true;
    CHECK(!is_channel_shuffle_candidate(c));
    c.tiny = false;

    // Channel choice (09.3): region repeat with fix bit 3 selects B or A.
    ChannelCandidate ra;
    ra.wms_region_repeat = true;
    ra.region_repeat_fix_bit3 = true;
    ra.wmt_region_repeat = false;
    ra.v_bit5 = false;
    CHECK_EQ(channel_shuffle_channel(ra), 2);  // B
    ra.v_bit5 = true;
    CHECK_EQ(channel_shuffle_channel(ra), 3);  // A
    // Region repeat with mask bit 3 clear: R or G.
    ChannelCandidate rg;
    rg.wms_region_repeat = true;
    rg.region_repeat_mask_bit3_clear = true;
    rg.integer_coords = true;
    rg.v_bit5 = true;
    CHECK_EQ(channel_shuffle_channel(rg), 1);  // G
    // Lane table of 09.1: (u & 8, v & 2) gives R, G, B or A.
    ChannelCandidate lane;
    lane.u_mod16 = 8;
    lane.v_mod4 = 2;
    CHECK_EQ(channel_shuffle_channel(lane), 3);
    lane.u_mod16 = 0;
    lane.v_mod4 = 0;
    CHECK_EQ(channel_shuffle_channel(lane), 0);

    // Runs: four candidates with one FBMSK and CLUT, a non-candidate, then two more with a new CLUT.
    std::vector<ChannelCandidate> batches(7, c);
    for (std::size_t i = 0; i < batches.size(); ++i) {
        batches[i].fbmsk = 0x00FFFFFFu;
        batches[i].clut_identity = 1;
        batches[i].fbp = static_cast<std::uint32_t>(i * 2);
    }
    batches[4].tme = false;
    batches[5].clut_identity = 2;
    const std::vector<std::uint32_t> runs = coalesce_channel_shuffle_runs(batches);
    CHECK_EQ(runs.size(), 3);
    CHECK_EQ(runs[0], 4);
    CHECK_EQ(runs[1], 1);
    CHECK_EQ(runs[2], 1);
}


void test_clear_vectors() {
    // Double-half: FBP 0, ZBP page 70 (FBW 10, 224 rows = 7 page rows), clear 640x224 -> 640x448 at FBP 0.
    const RegisterSnapshot r = base_regs();
    BatchInput in = make_input(r, sprite(0, 0, 640, 224, 0, 0));
    in.clear.display_height = 480;
    const DrawPlan plan = decide_draw(in, no_caps(), default_options());
    CHECK(plan.clear == ClearKind::DoubleHalf);
    CHECK(plan.clear_rect == (PixelRect{0, 0, 640, 448}));
    CHECK_EQ(plan.clear_base_fbp, 0);
    CHECK(plan.skip == SkipReason::ConvertedToClear);

    // Refused when the doubled height reaches the display height, or the next batch continues the pair.
    PixelRect scratch_rect{};
    std::uint32_t scratch_base = 0;
    BatchInput tall = in;
    tall.clear.display_height = 448;
    CHECK(detect_double_half_clear(tall, scratch_rect, scratch_base) == ClearKind::None);
    BatchInput next = in;
    next.clear.next_batch_nonclear_same_pair = true;
    CHECK(detect_double_half_clear(next, scratch_rect, scratch_base) == ClearKind::None);

    // Target clear: no depth write, constant colour over the whole scissor area.
    RegisterSnapshot tgt = base_regs();
    tgt.zte = false;
    BatchInput tc = make_input(tgt, sprite(0, 0, 640, 480, 0x40, 0));
    tc.geometry.vertices[0].rgba = {0x40, 0x20, 0x10, 0x80};
    tc.geometry.vertices[1].rgba = {0x40, 0x20, 0x10, 0x80};
    PixelRect rect{};
    std::uint32_t word = 0;
    CHECK(detect_target_clear(tc, no_caps(), rect, word) == ClearKind::TargetClear);
    CHECK_EQ(word, 0x80102040u);
    // The same colour over a part of the scissor area is not a clear.
    BatchInput partial = tc;
    partial.geometry = sprite(0, 0, 8, 8, 0x40, 0);
    CHECK(detect_target_clear(partial, no_caps(), rect, word) == ClearKind::None);

    // Memory clear: no target present and the width reaches (FBW - 1) * 64.
    BatchInput mem = tc;
    mem.clear.target_present = false;
    CHECK(detect_target_clear(mem, no_caps(), rect, word) == ClearKind::MemoryClear);
    BatchInput narrow = tc;
    narrow.clear.target_present = false;
    narrow.geometry = sprite(0, 0, 100, 480, 0x40, 0);
    CHECK(detect_target_clear(narrow, no_caps(), rect, word) == ClearKind::None);

    // Striped double clear: 10 strips of 64 pixels tile the rectangle; Z one page from FRAME.
    RegisterSnapshot st = base_regs();
    st.fbp = 0;
    st.zbp = 1;
    BatchGeometry strips;
    strips.prim = PrimClass::Sprite;
    for (int i = 0; i < 10; ++i) {
        strips.vertices.push_back(vertex(i * 64, 0, 0x11, 0x11111111u));
        strips.vertices.push_back(vertex(i * 64 + 64, 224, 0x11, 0x11111111u));
        strips.indices.push_back(static_cast<std::uint32_t>(2 * i));
        strips.indices.push_back(static_cast<std::uint32_t>(2 * i + 1));
    }
    const BatchInput sin = make_input(st, strips);
    PixelRect striped{};
    std::uint32_t base = 0;
    CHECK(detect_striped_double_clear(sin, striped, base) == ClearKind::StripedDouble);
    CHECK_EQ(base, 0);
    CHECK_EQ(striped.x1 - striped.x0, 640 + 64);
}

void test_pipeline_keys() {
    // No RGB written: blending and dual-source are dropped from the key (3.9 normalisation).
    RegisterSnapshot r = base_regs();
    r.abe = true;
    r.blend = BlendMode{BlendSel::Cs, BlendSel::Cd, BlendSel::Cd, BlendFactorSel::As};
    r.fbmsk = 0xFFFFFFFFu;
    const DrawPlan blended = decide_draw(make_input(r, sprite(0, 0, 8, 8)), no_caps(), default_options());
    CHECK(!blended.key.write_rgb);
    CHECK(!blended.key.blend_enabled);
    CHECK(!blended.key.dual_source);

    // Texture off: texture fields are zeroed, so different TFX and wrap values give the same key.
    RegisterSnapshot a = base_regs();
    RegisterSnapshot b = base_regs();
    a.tfx = Tfx::Modulate;
    b.tfx = Tfx::Highlight2;
    b.tcc = true;
    b.wms = WrapMode::RegionRepeat;
    const DrawPlan pa = decide_draw(make_input(a, sprite(0, 0, 8, 8)), no_caps(), default_options());
    const DrawPlan pb = decide_draw(make_input(b, sprite(0, 0, 8, 8)), no_caps(), default_options());
    CHECK(pa.key == pb.key);

    // ABE off: the blend selectors do not distinguish plans.
    RegisterSnapshot c = base_regs();
    RegisterSnapshot d = base_regs();
    d.blend = BlendMode{BlendSel::Cs, BlendSel::Cd, BlendSel::Cd, BlendFactorSel::As};
    CHECK(decide_draw(make_input(c, sprite(0, 0, 8, 8)), no_caps(), default_options()).key ==
          decide_draw(make_input(d, sprite(0, 0, 8, 8)), no_caps(), default_options()).key);

    // A texture-dependent difference changes the key.
    RegisterSnapshot e = base_regs();
    e.tme = true;
    e.tfx = Tfx::Decal;
    RegisterSnapshot f = e;
    f.tfx = Tfx::Modulate;
    CHECK(!(decide_draw(make_input(e, sprite(0, 0, 8, 8)), no_caps(), default_options()).key ==
            decide_draw(make_input(f, sprite(0, 0, 8, 8)), no_caps(), default_options()).key));
}

void test_draw_plan_smoke() {
    // Reset artefact: a black point at the origin when the next PRIM is not a point.
    BatchGeometry point;
    point.prim = PrimClass::Point;
    point.vertices = {vertex(0, 0, 0)};
    point.indices = {0};
    BatchInput reset = make_input(base_regs(), point);
    CHECK(decide_draw(reset, no_caps(), default_options()).skip == SkipReason::ResetArtefact);
    reset.current_prim_is_point = true;
    CHECK(decide_draw(reset, no_caps(), default_options()).skip != SkipReason::ResetArtefact);

    // DATE on CT24 writes nothing.
    RegisterSnapshot ct24 = base_regs();
    ct24.frame = FrameFormat::Ct24;
    ct24.date = true;
    CHECK(decide_draw(make_input(ct24, sprite(0, 0, 8, 8)), no_caps(), default_options()).skip ==
          SkipReason::DateOnCt24);

    // Empty draw rectangle after the scissor.
    RegisterSnapshot off = base_regs();
    off.scax0 = 500;
    off.scax1 = 510;
    CHECK(decide_draw(make_input(off, sprite(0, 0, 8, 8)), no_caps(), default_options()).skip == SkipReason::EmptyRect);

    // Nothing written: no colour and no depth.
    RegisterSnapshot none = base_regs();
    none.fbmsk = 0xFFFFFFFFu;
    none.zte = false;
    CHECK(decide_draw(make_input(none, sprite(0, 0, 8, 8)), no_caps(), default_options()).skip ==
          SkipReason::NothingWritten);

    // Plain opaque draw: no barrier, no blend, one main pass.
    const DrawPlan plain = decide_draw(make_input(base_regs(), sprite(0, 0, 8, 8, 0x40, 0x10)), no_caps(),
                                       default_options());
    CHECK(plain.skip == SkipReason::None);
    CHECK(plain.barrier == BarrierKind::None);
    CHECK(plain.blend.method == BlendMethod::None);
    CHECK_EQ(plain.passes.size(), 1);
    CHECK(plain.passes[0] == PassKind::MainDraw);

    // Interpolate at the default High level: the sprite-batch rule chooses the shader.
    RegisterSnapshot blend = base_regs();
    blend.abe = true;
    blend.blend = BlendMode{BlendSel::Cs, BlendSel::Cd, BlendSel::Cd, BlendFactorSel::As};
    BatchInput bin = make_input(blend, sprite(0, 0, 8, 8, 0x40, 0));
    bin.fragment_alpha_min = bin.fragment_alpha_max = 0x40;
    const DrawPlan bp = decide_draw(bin, no_caps(), default_options());
    CHECK(bp.blend.method == BlendMethod::Shader);
    CHECK(bp.blend.reason == BlendReason::SpriteBatchShader);
    CHECK(bp.blend.reads_destination);
    CHECK(bp.barrier == BarrierKind::One);
    // Basic uses MIX for the same batch (the factor is at most one, Cs and Cd are separable).
    Options basic = default_options();
    basic.blend_level = BlendLevel::Basic;
    CHECK(decide_draw(bin, no_caps(), basic).blend.method == BlendMethod::Mixed);

    // Texture is the frame target: a hazard copy without framebuffer fetch; with fetch, no copy.
    BatchInput hazard = make_input(base_regs(), sprite(0, 0, 8, 8));
    hazard.regs.tme = true;
    hazard.texture.is_frame_target = true;
    const DrawPlan hz = decide_draw(hazard, no_caps(), default_options());
    CHECK(!hz.passes.empty() && hz.passes.front() == PassKind::HazardCopy);
    CHECK(hz.barrier == BarrierKind::One);
    CHECK_EQ(decide_draw(hazard, fetch_caps(), default_options()).passes.size(), 1);

    // A full-frame constant clear is converted and the main draw is skipped.
    RegisterSnapshot clr = base_regs();
    clr.zte = false;
    const DrawPlan cp = decide_draw(make_input(clr, sprite(0, 0, 640, 480, 0x40, 0)), no_caps(), default_options());
    CHECK(cp.clear == ClearKind::TargetClear);
    CHECK(cp.draw_converted_to_clear);
}

struct TestEntry {
    const char* name;
    void (*run)();
};

}  // namespace

int main(int argc, char** argv) {
    const TestEntry tests[] = {
        {"hw_draw_rect_table", test_draw_rect_table},
        {"hw_coverage_1x", test_coverage_1x},
        {"hw_blend_classes", test_blend_classes},
        {"hw_blend_vectors", test_blend_vectors},
        {"hw_blend_levels", test_blend_levels},
        {"hw_blend_simplify", test_blend_simplify},
        {"hw_colour_wrap", test_colour_wrap},
        {"hw_channel_writes", test_channel_writes},
        {"hw_atest_fold", test_atest_fold},
        {"hw_atest_methods", test_atest_methods},
        {"hw_date_methods", test_date_methods},
        {"hw_date_vectors", test_date_vectors},
        {"hw_barrier_rules", test_barrier_rules},
        {"hw_overlap_groups", test_overlap_groups},
        {"hw_vertex_trace", test_vertex_trace},
        {"hw_primitive_expansion", test_primitive_expansion},
        {"hw_scanmsk", test_scanmsk},
        {"hw_texture_shuffle", test_texture_shuffle},
        {"hw_channel_shuffle", test_channel_shuffle},
        {"hw_clear_vectors", test_clear_vectors},
        {"hw_pipeline_keys", test_pipeline_keys},
        {"hw_draw_plan_smoke", test_draw_plan_smoke},
    };

    int ran = 0;
    for (const TestEntry& entry : tests) {
        bool selected = argc <= 1;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], entry.name) == 0) {
                selected = true;
            }
        }
        if (!selected) {
            continue;
        }
        const int before = g_failures;
        entry.run();
        ++ran;
        std::printf("%-22s %s\n", entry.name, g_failures == before ? "ok" : "FAILED");
    }
    if (ran == 0) {
        std::fprintf(stderr, "no test selected\n");
        return 2;
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
