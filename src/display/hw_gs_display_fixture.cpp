// SPDX-License-Identifier: MIT
//
// CTest fixtures for the HWR display output reference (hw_gs_display.hpp).
// Copyright (c) 2026 GT4Recomp contributors.
//
// Written from the TASK-026 specification (docs/hw-renderer-spec). Vectors: 13.1 (video
// mode table and detection), 13.2 (rectangles NTSC 640x448 and 480p 720x480, the 3x size
// 1920x1344, DBX wrap, anti-blur and offsets), 13.3 (merge plan and composition),
// 13.4 (deinterlace modes on synthetic fields), 13.5 (ShadeBoost identity and scanlines),
// 13.6 (presentation rectangle vectors 240/1440 and 152/720) and the option parsers.
// Choices made where the specification is silent are cited as IN-xx in the sources.
//
// Usage: hw_gs_display_fixture [test-name ...]. With no arguments every test runs.

#include <algorithm>
#include "hw_gs_display.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace coregs::display;

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

// ----- register builders (01.7 layouts) -----------------------------------------------

std::uint64_t smode1_word(std::uint64_t cmod, std::uint64_t lc) {
    return (cmod << 13) | (lc << 3);
}

std::uint64_t smode2_word(bool interlace, bool ffmd) {
    return (interlace ? 1u : 0u) | (ffmd ? 2u : 0u);
}

std::uint64_t pmode_word(bool en1, bool en2, bool mmod, bool slbg, std::uint64_t alp) {
    return (en1 ? 1u : 0u) | (en2 ? 2u : 0u) | (mmod ? 0x20u : 0u) | (slbg ? 0x80u : 0u) | (alp << 8);
}

std::uint64_t dispfb_word(std::uint64_t fbp, std::uint64_t fbw, std::uint64_t psm, std::uint64_t dbx,
                          std::uint64_t dby) {
    return (fbp & 0x1FFu) | ((fbw & 0x3Fu) << 9) | ((psm & 0x1Fu) << 15) | ((dbx & 0x7FFu) << 32) |
           ((dby & 0x7FFu) << 43);
}

std::uint64_t display_word(std::uint64_t dx, std::uint64_t dy, std::uint64_t magh, std::uint64_t magv,
                           std::uint64_t dw, std::uint64_t dh) {
    return (dx & 0xFFFu) | ((dy & 0x7FFu) << 12) | ((magh & 0xFu) << 23) | ((magv & 3u) << 27) |
           ((dw & 0xFFFu) << 32) | ((dh & 0x7FFu) << 44);
}

// NTSC 480i as the game menus run it: CMOD 2, SMODE2 INT = 1 and FFMD = 1.
DisplayRegisters ntsc_registers() {
    DisplayRegisters regs;
    regs.smode1 = smode1_word(2, 0);
    regs.smode2 = smode2_word(true, true);
    regs.syncv = 0;
    regs.pmode = pmode_word(true, false, false, false, 255);
    return regs;
}

DisplayRegisters progressive_480_registers() {
    DisplayRegisters regs;
    regs.smode1 = smode1_word(0, 32);
    regs.smode2 = smode2_word(false, false);
    regs.pmode = pmode_word(true, false, false, false, 255);
    return regs;
}

// Image helpers.
Image solid(std::uint32_t w, std::uint32_t h, std::uint8_t value, std::uint8_t alpha = 255) {
    return Image(w, h, Rgba8{value, value, value, alpha});
}

// ----- 13.1 ----------------------------------------------------------------------------

void test_video_mode() {
    CHECK(detect_video_mode(smode1_word(2, 0), smode2_word(true, true)) == VideoMode::Ntsc);
    CHECK(detect_video_mode(smode1_word(3, 0), smode2_word(true, false)) == VideoMode::Pal);
    CHECK(detect_video_mode(smode1_word(0, 22), smode2_word(true, false)) == VideoMode::Interlaced1080);
    CHECK(detect_video_mode(smode1_word(0, 22), smode2_word(false, false)) == VideoMode::Progressive720);
    CHECK(detect_video_mode(smode1_word(0, 32), smode2_word(false, false)) == VideoMode::Progressive480);
    CHECK(detect_video_mode(smode1_word(0, 32), smode2_word(true, false)) == VideoMode::Vesa);
    CHECK(detect_video_mode(smode1_word(0, 5), smode2_word(false, false)) == VideoMode::Vesa);
    CHECK(detect_video_mode(smode1_word(1, 0), smode2_word(false, false)) == VideoMode::None);  // IN-22
    CHECK(detect_video_mode(0, 0) == VideoMode::Vesa);

    const VideoModeTable ntsc = video_mode_table(VideoMode::Ntsc);
    CHECK_EQ(ntsc.width, 640);
    CHECK_EQ(ntsc.field_height, 224);
    CHECK_EQ(ntsc.dx_base, 642);
    CHECK_EQ(ntsc.dy_base, 25);
    CHECK_EQ(ntsc.h_divider, 4);
    CHECK_EQ(ntsc.v_divider, 1);
    const VideoModeTable pal = video_mode_table(VideoMode::Pal);
    CHECK_EQ(pal.field_height, 256);
    CHECK_EQ(pal.dx_base, 676);
    CHECK_EQ(pal.dy_base, 36);
    const VideoModeTable vesa = video_mode_table(VideoMode::Vesa);
    CHECK_EQ(vesa.field_height, 480);
    CHECK_EQ(vesa.dx_base, 276);
    CHECK_EQ(vesa.h_divider, 2);
    const VideoModeTable p480 = video_mode_table(VideoMode::Progressive480);
    CHECK_EQ(p480.width, 720);
    CHECK_EQ(p480.dx_base, 232);
    CHECK_EQ(p480.dy_base, 35);
    CHECK_EQ(p480.h_divider, 2);
    const VideoModeTable p720 = video_mode_table(VideoMode::Progressive720);
    CHECK_EQ(p720.width, 1280);
    CHECK_EQ(p720.field_height, 720);
    CHECK_EQ(p720.dx_base, 302);
    CHECK_EQ(p720.dy_base, 24);
    CHECK_EQ(p720.h_divider, 1);
    const VideoModeTable p1080 = video_mode_table(VideoMode::Interlaced1080);
    CHECK_EQ(p1080.width, 1920);
    CHECK_EQ(p1080.field_height, 540);
    CHECK_EQ(p1080.dx_base, 238);
    CHECK_EQ(p1080.dy_base, 40);

    // Interlaced flag and really interlaced (IN-21): analogue modes only.
    CHECK(interlaced_flag(VideoMode::Ntsc, smode2_word(true, false)));
    CHECK(!interlaced_flag(VideoMode::Progressive480, smode2_word(true, false)));
    CHECK(really_interlaced(VideoMode::Ntsc, 1));
    CHECK(!really_interlaced(VideoMode::Ntsc, 0));
    CHECK(!really_interlaced(VideoMode::Progressive480, 1));
    CHECK(really_interlaced(VideoMode::Interlaced1080, 1));
}

// ----- 13.2 rectangles -----------------------------------------------------------------

void test_pcrtc_rects() {
    // NTSC 480i vector: DW 2559, DH 447, MAGH 3, MAGV 0 -> 640x448, field read 640x224.
    DisplayRegisters ntsc = ntsc_registers();
    ntsc.circuits[0] = CircuitRegisters{dispfb_word(0x8C, 10, 1, 0, 0), display_word(0, 0, 3, 0, 2559, 447)};
    std::array<std::uint32_t, 2> previous{0, 0};
    DisplayLayout layout = compute_layout(ntsc, previous);
    CHECK(layout.valid);
    CHECK(layout.interlaced);
    CHECK(layout.really_interlaced == false);  // VFP bit 0 is clear in this fixture
    CHECK(layout.circuits[0].enabled);
    CHECK(!layout.circuits[1].enabled);
    CHECK_EQ(layout.circuits[0].display_width, 640);
    CHECK_EQ(layout.circuits[0].display_height, 448);
    CHECK_EQ(layout.circuits[0].read_width, 640);
    CHECK_EQ(layout.circuits[0].read_height, 224);
    CHECK_EQ(layout.union_width, 640);
    CHECK_EQ(layout.union_height, 448);

    // Full height output at 3x: 1920x1344 (13.2 vector).
    const OutputSize size = merge_output_size(layout, false, 3);
    CHECK_EQ(size.width, 1920);
    CHECK_EQ(size.height, 1344);

    // 480p vector: DW 1439, MAGH 1, DH 479 -> 720x480, progressive so the read is full height.
    DisplayRegisters p480 = progressive_480_registers();
    p480.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(0, 0, 1, 0, 1439, 479)};
    previous = {0, 0};
    layout = compute_layout(p480, previous);
    CHECK(layout.valid);
    CHECK(!layout.interlaced);
    CHECK_EQ(layout.circuits[0].display_width, 720);
    CHECK_EQ(layout.circuits[0].display_height, 480);
    CHECK_EQ(layout.circuits[0].read_height, 480);
    CHECK_EQ(merge_output_size(layout, false, 1).height, 480);

    // Disabled rule (13.2 rule 2): FBW = 0, DW = 0, DH = 0 and MAGH = 0 disables the circuit.
    DisplayRegisters off = ntsc_registers();
    off.circuits[0] = CircuitRegisters{dispfb_word(0, 0, 0, 0, 0), display_word(0, 0, 0, 0, 0, 0)};
    previous = {0, 0};
    CHECK(!compute_layout(off, previous).circuits[0].enabled);
    // FBW = 0 with a non-zero DW keeps the circuit enabled.
    off.circuits[0] = CircuitRegisters{dispfb_word(0, 0, 0, 0, 0), display_word(0, 0, 3, 0, 2559, 447)};
    previous = {0, 0};
    CHECK(compute_layout(off, previous).circuits[0].enabled);
    // PMODE disables a circuit that the registers describe.
    DisplayRegisters pmode_off = ntsc_registers();
    pmode_off.pmode = pmode_word(false, false, false, false, 255);
    pmode_off.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(0, 0, 3, 0, 2559, 447)};
    previous = {0, 0};
    CHECK(!compute_layout(pmode_off, previous).circuits[0].enabled);

    // Framebuffer wrap (13.2): DBX 2000 with a 640-wide read wraps by 2048.
    DisplayRegisters wrap = ntsc_registers();
    wrap.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 2000, 0), display_word(0, 0, 3, 0, 2559, 447)};
    previous = {0, 0};
    CHECK_EQ(compute_layout(wrap, previous).circuits[0].dbx, 2000 - 2048);

    // No known mode: no image.
    DisplayRegisters none = ntsc_registers();
    none.smode1 = smode1_word(1, 0);
    previous = {0, 0};
    CHECK(!compute_layout(none, previous).valid);
}

void test_pcrtc_offsets() {
    // Horizontal anti-blur: DBX 10 and 11 with the same DX both use the smaller DBX.
    DisplayRegisters regs = ntsc_registers();
    regs.pmode = pmode_word(true, true, false, false, 255);
    regs.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 10, 0), display_word(100, 50, 3, 0, 2559, 447)};
    regs.circuits[1] = CircuitRegisters{dispfb_word(0, 10, 0, 11, 0), display_word(100, 50, 3, 0, 2559, 447)};
    std::array<std::uint32_t, 2> previous{0, 0};
    DisplayLayout layout = compute_layout(regs, previous);
    CHECK_EQ(layout.circuits[0].dbx, 10);
    CHECK_EQ(layout.circuits[1].dbx, 10);

    // Anti-blur off keeps both origins.
    regs.anti_blur = false;
    previous = {0, 0};
    layout = compute_layout(regs, previous);
    CHECK_EQ(layout.circuits[1].dbx, 11);

    // Vertical anti-blur: circuit 1 changed its DBY (21, previously 20) and circuit 0 did not,
    // so the alternating circuit moves onto the other one.
    regs.anti_blur = true;
    regs.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 20), display_word(100, 50, 3, 0, 2559, 447)};
    regs.circuits[1] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 21), display_word(100, 50, 3, 0, 2559, 447)};
    previous = {20, 20};
    layout = compute_layout(regs, previous);
    CHECK(layout.circuits[1].dby_changed);
    CHECK(!layout.circuits[0].dby_changed);
    CHECK_EQ(layout.circuits[1].dby, 20);
    CHECK(layout.one_circuit_dby_changed);
    CHECK_EQ(previous[1], 21);  // history keeps the raw value of this VBlank

    // Neither or both changed: the larger origin moves onto the smaller one.
    regs.circuits[1] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 21), display_word(100, 50, 3, 0, 2559, 447)};
    previous = {21, 20};  // both changed: 21 (circuit 1) moves onto 20 (circuit 0)
    layout = compute_layout(regs, previous);
    CHECK_EQ(layout.circuits[1].dby, 20);

    // Relative x offset (c): DX difference 16 over the divider 4 is 4 and is applied.
    regs.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(100, 50, 3, 0, 2559, 447)};
    regs.circuits[1] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(116, 50, 3, 0, 2559, 447)};
    previous = {0, 0};
    layout = compute_layout(regs, previous);
    CHECK_EQ(layout.circuits[1].screen_x, 4);
    // A difference of 8 (relative 2) is not applied while anti-blur is on...
    regs.circuits[1] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(108, 50, 3, 0, 2559, 447)};
    previous = {0, 0};
    layout = compute_layout(regs, previous);
    CHECK_EQ(layout.circuits[1].screen_x, 0);
    // ...and is applied with anti-blur off.
    regs.anti_blur = false;
    previous = {0, 0};
    layout = compute_layout(regs, previous);
    CHECK_EQ(layout.circuits[1].screen_x, 2);
    regs.anti_blur = true;

    // Line offsets (b) and the global shift (d) on a single circuit (NTSC base 25 * 2 = 50).
    DisplayRegisters single = ntsc_registers();
    single.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(0, 50, 3, 0, 2559, 447)};
    previous = {0, 0};
    CHECK_EQ(compute_layout(single, previous).circuits[0].screen_y, 0);
    single.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(0, 51, 3, 0, 2559, 447)};
    previous = {0, 0};
    CHECK_EQ(compute_layout(single, previous).circuits[0].screen_y, 1);  // line offset 1: parity 1
    single.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(0, 53, 3, 0, 2559, 447)};
    previous = {0, 0};
    CHECK_EQ(compute_layout(single, previous).circuits[0].screen_y, 3);  // parity 1 plus shift 2
    single.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(0, 60, 3, 0, 2559, 447)};
    previous = {0, 0};
    CHECK_EQ(compute_layout(single, previous).circuits[0].screen_y, 0);  // offset 10 is no small offset

    // DY anti-blur (a): |DY1 - DY0| = 2 < 4 lowers the larger DY to the smaller one.
    regs.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(100, 50, 3, 0, 2559, 447)};
    regs.circuits[1] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(100, 52, 3, 0, 2559, 447)};
    previous = {0, 0};
    layout = compute_layout(regs, previous);
    CHECK_EQ(layout.circuits[1].dy, 50);
    regs.scanmsk_active = true;
    previous = {0, 0};
    layout = compute_layout(regs, previous);
    CHECK_EQ(layout.circuits[1].dy, 52);  // SCANMSK disables the DY anti-blur
}

// ----- 13.3 merge ----------------------------------------------------------------------

CircuitRect rect_at(bool enabled, std::uint32_t fbp, std::uint32_t psm, std::int32_t x, std::int32_t y,
                    std::uint32_t w, std::uint32_t h) {
    CircuitRect rect;
    rect.enabled = enabled;
    rect.fbp = fbp;
    rect.fbw = 10;
    rect.psm = psm;
    rect.screen_x = x;
    rect.screen_y = y;
    rect.display_width = w;
    rect.display_height = h;
    return rect;
}

void test_merge_plan() {
    const CircuitRect a = rect_at(true, 0, 0, 0, 0, 640, 448);
    const CircuitRect b = rect_at(true, 0x8C, 0, 0, 0, 640, 448);
    const CircuitRect off = rect_at(false, 0, 0, 0, 0, 0, 0);
    PmodeFields opaque;
    opaque.en1 = true;
    opaque.en2 = true;
    opaque.alp = 255;

    MergePlan plan = plan_merge(opaque, a, off, false, true);
    CHECK(plan.outcome == MergeOutcome::KeepPrevious);
    plan = plan_merge(opaque, off, off, true, true);
    CHECK(plan.outcome == MergeOutcome::BackgroundOnly);

    plan = plan_merge(opaque, a, b, true, true);
    CHECK(plan.outcome == MergeOutcome::Composite);
    CHECK(plan.use_rc1);
    CHECK(plan.use_rc2);
    CHECK(!plan.single_read);

    // MMOD = 1 with ALP = 0: RC1 is transparent and dropped.
    PmodeFields transparent = opaque;
    transparent.mmod = true;
    transparent.alp = 0;
    plan = plan_merge(transparent, a, off, true, true);
    CHECK(plan.outcome == MergeOutcome::BackgroundOnly);

    // SLBG = 1: the background is BGCOLOR and RC2 is not used.
    PmodeFields slbg = opaque;
    slbg.slbg = true;
    plan = plan_merge(slbg, a, b, true, true);
    CHECK(plan.use_rc1);
    CHECK(!plan.use_rc2);

    // RC1 covers RC2 with MMOD = 1 and ALP = 255: RC2 is not used.
    PmodeFields full = opaque;
    full.mmod = true;
    const CircuitRect small = rect_at(true, 0x8C, 0, 10, 10, 100, 100);
    plan = plan_merge(full, a, small, true, true);
    CHECK(!plan.use_rc2);
    plan = plan_merge(opaque, a, small, true, true);
    CHECK(plan.use_rc2);

    // Identical duplicate circuits drop RC1 (IN-34).
    plan = plan_merge(opaque, a, a, true, true);
    CHECK(!plan.use_rc1);
    CHECK(plan.use_rc2);

    // Same source at different positions: one read (single_read).
    const CircuitRect shifted = rect_at(true, 0, 0, 0, 1, 640, 448);
    plan = plan_merge(opaque, a, shifted, true, true);
    CHECK(plan.single_read);
    CHECK(same_source(a, shifted));
    CHECK(!same_source(a, b));

    // CT24 and CT32 are the same family (IN-33); CT32 and Z32 are not.
    CHECK(same_source(rect_at(true, 0, 1, 0, 0, 1, 1), rect_at(true, 0, 0, 0, 0, 1, 1)));
    CHECK(!same_source(rect_at(true, 0, 0x30, 0, 0, 1, 1), rect_at(true, 0, 0, 0, 0, 1, 1)));
}

void test_merge_compose() {
    const Image rc2_source = solid(2, 2, 100);
    const Image rc1_opaque = solid(2, 2, 200);
    Image rc1_source(2, 2, Rgba8{200, 0, 0, 255});
    Image rc1_half(2, 2, Rgba8{200, 0, 0, 64});

    PmodeFields pmode;
    pmode.en1 = true;
    pmode.en2 = true;
    pmode.alp = 255;

    const SourcePlacement placement2{&rc2_source, 0, 0, 2, 2};
    const SourcePlacement placement1{&rc1_source, 0, 0, 2, 2};
    const MergePlan both = [&] {
        MergePlan plan;
        plan.outcome = MergeOutcome::Composite;
        plan.use_rc1 = true;
        plan.use_rc2 = true;
        return plan;
    }();

    // RC1 texel alpha 255 with MMOD = 0: a = min(255, 510) = 255, RC1 replaces RC2.
    Image out = compose_merge(both, pmode, 0, 0, 0, 2, 2, 1, 0, placement1, placement2);
    CHECK(out.at(0, 0) == (Rgba8{200, 0, 0, 255}));
    CHECK(out.at(1, 1) == (Rgba8{200, 0, 0, 255}));

    // Texel alpha 64 with MMOD = 0: a = 128 (0x80 = 1.0 doubles the texel alpha, V-01-12).
    // Red: (200 * 128 + 100 * 127 + 127) / 255 = 150; green and blue: (100 * 127 + 127) / 255 = 50.
    const SourcePlacement half{&rc1_half, 0, 0, 2, 2};
    out = compose_merge(both, pmode, 0, 0, 0, 2, 2, 1, 0, half, placement2);
    CHECK(out.at(0, 0) == (Rgba8{150, 50, 50, 64}));
    CHECK_EQ(merge_alpha_for_texel(pmode, 64), 128);
    CHECK_EQ(merge_alpha_for_texel(pmode, 200), 255);

    // MMOD = 1 with ALP 255: a = 255 and the output alpha is the source alpha.
    PmodeFields mmod = pmode;
    mmod.mmod = true;
    out = compose_merge(both, mmod, 0, 0, 0, 2, 2, 1, 0, half, placement2);
    CHECK(out.at(0, 0) == (Rgba8{200, 0, 0, 64}));
    CHECK_EQ(blend_channel(200, 100, 128), 150);

    // Interlace rows (13.3 step 3): the destination moves down one row; row 0 keeps BGCOLOR.
    out = compose_merge(both, pmode, 9, 9, 9, 2, 2, 1, 1, placement1, placement2);
    CHECK(out.at(0, 0) == (Rgba8{9, 9, 9, 255}));
    CHECK(out.at(0, 1) == (Rgba8{200, 0, 0, 255}));

    // Scale 2: the 2x2 destination becomes 4x4 and nearest sampling keeps the texel.
    Image checker(2, 2, Rgba8{0, 0, 0, 255});
    checker.at(1, 0) = Rgba8{255, 255, 255, 255};
    const SourcePlacement scaled{&checker, 0, 0, 2, 2};
    MergePlan rc2_only;
    rc2_only.outcome = MergeOutcome::Composite;
    rc2_only.use_rc2 = true;
    out = compose_merge(rc2_only, pmode, 0, 0, 0, 4, 4, 2, 0, scaled, scaled);
    CHECK(out.at(3, 0) == (Rgba8{255, 255, 255, 255}));
    CHECK(out.at(0, 3) == (Rgba8{0, 0, 0, 255}));

    // Background only and keep-previous plans show BGCOLOR.
    MergePlan background;
    background.outcome = MergeOutcome::BackgroundOnly;
    out = compose_merge(background, pmode, 1, 2, 3, 2, 2, 1, 0, placement1, placement2);
    CHECK(out.at(1, 1) == (Rgba8{1, 2, 3, 255}));
}

void test_interlace_phase() {
    // Shift by scale * (field XOR parity) for really interlaced FFMD = 1 frames, not bob or off.
    CHECK_EQ(interlace_offset_rows(true, true, DeinterlaceMode::Adaptive, 3, 1), 3);
    CHECK_EQ(interlace_offset_rows(true, true, DeinterlaceMode::Adaptive, 3, 0), 0);
    CHECK_EQ(interlace_offset_rows(true, true, DeinterlaceMode::Weave, 1, 1), 1);
    CHECK_EQ(interlace_offset_rows(true, true, DeinterlaceMode::Bob, 3, 1), 0);
    CHECK_EQ(interlace_offset_rows(true, true, DeinterlaceMode::Off, 3, 1), 0);
    CHECK_EQ(interlace_offset_rows(true, false, DeinterlaceMode::Adaptive, 3, 1), 0);
    CHECK_EQ(interlace_offset_rows(false, true, DeinterlaceMode::Adaptive, 3, 1), 0);
    CHECK_EQ(field_from_csr(true), 1);
    CHECK_EQ(field_from_csr(false), 0);

    // The 448-line image of the game's 480i: the field phase flips which rows the field occupies.
    DisplayRegisters regs = ntsc_registers();
    regs.circuits[0] = CircuitRegisters{dispfb_word(0, 10, 0, 0, 0), display_word(0, 0, 3, 0, 2559, 447)};
    std::array<std::uint32_t, 2> previous{0, 0};
    const DisplayLayout layout = compute_layout(regs, previous);
    CHECK_EQ(merge_output_size(layout, false, 1).height, 448);
}

// ----- 13.4 deinterlace ----------------------------------------------------------------

void test_deinterlace_weave_blend() {
    // Weave writes the rows of the current field and keeps the others from the last frame.
    Image first(2, 4, Rgba8{10, 10, 10, 255});
    Image second(2, 4, Rgba8{200, 200, 200, 255});
    Deinterlacer weave;
    Image out = weave.process(first, 0, DeinterlaceMode::Weave, false, 1);
    CHECK(out.at(0, 1) == (Rgba8{10, 10, 10, 255}));
    out = weave.process(second, 0, DeinterlaceMode::Weave, false, 1);  // field 0 = even rows
    CHECK(out.at(0, 0) == (Rgba8{200, 200, 200, 255}));
    CHECK(out.at(0, 2) == (Rgba8{200, 200, 200, 255}));
    CHECK(out.at(0, 1) == (Rgba8{10, 10, 10, 255}));
    CHECK(out.at(0, 3) == (Rgba8{10, 10, 10, 255}));

    // Bottom-first flips the parity: field 0 now writes odd rows.
    Deinterlacer bff;
    out = bff.process(first, 0, DeinterlaceMode::Weave, true, 1);
    out = bff.process(second, 0, DeinterlaceMode::Weave, true, 1);
    CHECK(out.at(0, 1) == (Rgba8{200, 200, 200, 255}));
    CHECK(out.at(0, 0) == (Rgba8{10, 10, 10, 255}));

    // Blend: vertical 3-tap (1/4, 1/2, 1/4) with clamped edges. Column values 0, 0, 255, 0.
    Image column(1, 4);
    const std::uint8_t values[4] = {0, 0, 255, 0};
    for (std::uint32_t y = 0; y < 4; ++y) {
        column.at(0, y) = Rgba8{values[y], values[y], values[y], 255};
    }
    Deinterlacer blend;
    out = blend.process(column, 0, DeinterlaceMode::Blend, false, 1);
    // Weave of field 0 keeps rows 0 and 2 of the input; rows 1 and 3 are the first frame (here
    // the same image), so the woven column equals the input and the taps are:
    CHECK_EQ(out.at(0, 0).r, (0 + 2 * 0 + 0 + 2) / 4);       // edge replicate: (0 + 0 + 0 + 2) / 4 = 0
    CHECK_EQ(out.at(0, 1).r, (0 + 2 * 0 + 255 + 2) / 4);     // 64
    CHECK_EQ(out.at(0, 2).r, (0 + 2 * 255 + 0 + 2) / 4);     // 128
    CHECK_EQ(out.at(0, 3).r, (255 + 2 * 0 + 0 + 2) / 4);     // 64
}

void test_deinterlace_bob() {
    // Bob shifts the merged image by scale * (1 - field) lines.
    Image merged(1, 4);
    for (std::uint32_t y = 0; y < 4; ++y) {
        merged.at(0, y) = Rgba8{static_cast<std::uint8_t>(y * 10), 0, 0, 255};
    }
    Deinterlacer bob;
    Image out = bob.process(merged, 0, DeinterlaceMode::Bob, false, 2);  // shift 2 lines
    CHECK_EQ(out.at(0, 0).r, 0);   // clamped to row 0
    CHECK_EQ(out.at(0, 1).r, 0);
    CHECK_EQ(out.at(0, 2).r, 0);   // row 0
    CHECK_EQ(out.at(0, 3).r, 10);  // row 1
    out = bob.process(merged, 1, DeinterlaceMode::Bob, false, 2);  // field 1: no shift
    CHECK_EQ(out.at(0, 3).r, 30);
}

void test_deinterlace_adaptive() {
    // Static picture: constant input stays constant over several fields.
    const Image flat = solid(2, 6, 77);
    Deinterlacer static_picture;
    for (int field = 0; field < 5; ++field) {
        const Image out = static_picture.process(flat, field & 1, DeinterlaceMode::Adaptive, false, 1);
        CHECK(out.at(1, 3) == (Rgba8{77, 77, 77, 255}));
    }
    CHECK_EQ(static_picture.history_size(), 4);

    // Moving detail: the pixel at column 0 changes between two fields of the same parity, so
    // its missing line interpolates (bob). Column 1 is static and keeps the woven line.
    Image before(2, 6, Rgba8{0, 0, 0, 255});
    Image after(2, 6, Rgba8{0, 0, 0, 255});
    after.at(0, 0) = Rgba8{255, 255, 255, 255};
    after.at(0, 2) = Rgba8{255, 255, 255, 255};
    after.at(0, 4) = Rgba8{255, 255, 255, 255};
    Deinterlacer moving;
    Image out = moving.process(before, 0, DeinterlaceMode::Adaptive, false, 1);  // no history: bob
    CHECK_EQ(out.at(0, 1).r, 0);
    out = moving.process(after, 0, DeinterlaceMode::Adaptive, false, 1);
    CHECK_EQ(out.at(0, 1).r, 255);  // moving: bob of the rows above and below
    CHECK_EQ(out.at(1, 1).r, 0);    // static column: previous line kept
    CHECK(out.at(0, 2) == (Rgba8{255, 255, 255, 255}));  // current-field rows pass through

    // Static sharp detail in field-rendered content (IN-337, the game menus): each field holds its own lines,
    // line-doubled into the merge and the odd field shifted down one row (13.3 step 3). Every frame line
    // has its own value, so no line is the average of its neighbours. From the third field on (each parity
    // has a predecessor) the output must be the progressive picture and must not change between fields.
    constexpr std::uint32_t kLines = 12;
    auto line_value = [](std::uint32_t line) { return static_cast<std::uint8_t>((line * 37u + 11u) & 0xFFu); };
    auto field_merge = [&](int field) {
        Image merged(1, kLines, Rgba8{0, 0, 0, 255});
        for (std::uint32_t y = 0; y < kLines; ++y) {
            // Even field: rows 2k and 2k+1 show line 2k. Odd field: rows 2k+1 and 2k+2 show line 2k+1.
            const std::uint32_t k = field == 0 ? y / 2u : (y == 0u ? 0u : (y - 1u) / 2u);
            const std::uint32_t line = std::min<std::uint32_t>(2u * k + static_cast<std::uint32_t>(field), kLines - 1u);
            const std::uint8_t v = line_value(line);
            merged.at(0, y) = Rgba8{v, v, v, 255};
        }
        return merged;
    };
    Deinterlacer sharp;
    Image last;
    for (int step = 0; step < 6; ++step) {
        const int field = step & 1;
        const Image result = sharp.process(field_merge(field), field, DeinterlaceMode::Adaptive, false, 1);
        if (step >= 2) {
            for (std::uint32_t y = 1; y + 1 < kLines; ++y) {
                CHECK_EQ(result.at(0, y).r, line_value(y));  // the progressive picture, interior rows
            }
            if (step >= 3) {
                // No bob between consecutive fields. The top and bottom rows pass through (13.4) and are
                // the two fields' own edge lines, so only interior rows are compared.
                for (std::uint32_t y = 1; y + 1 < kLines; ++y) CHECK(result.at(0, y) == last.at(0, y));
            }
        }
        last = result;
    }

    // Clearing the history removes the previous field: the next field is bobbed.
    moving.clear_history();
    CHECK_EQ(moving.history_size(), 0);
    out = moving.process(after, 1, DeinterlaceMode::Adaptive, false, 1);
    CHECK_EQ(out.at(1, 2).r, 0);  // field 1 writes odd rows; row 2 is missing and bobbed from 1 and 3

    // IN-396 (a combed scrolling menu ticker): the current field and the field two back agree, but the field that
    // would be woven in (one back) changed against its own previous sample (three back). The reference treats that
    // as motion and interpolates; weaving it put a second position of the text on every other row.
    Deinterlacer ticker;
    const Image dark = solid(1, 6, 0);
    const Image bright = solid(1, 6, 200);
    ticker.process(dark, 0, DeinterlaceMode::Adaptive, false, 1);
    ticker.process(dark, 1, DeinterlaceMode::Adaptive, false, 1);    // three back, opposite parity
    ticker.process(dark, 0, DeinterlaceMode::Adaptive, false, 1);    // two back, same parity
    ticker.process(bright, 1, DeinterlaceMode::Adaptive, false, 1);  // one back: the woven field moved
    out = ticker.process(dark, 0, DeinterlaceMode::Adaptive, false, 1);
    CHECK_EQ(out.at(0, 3).r, 0);  // interpolated from rows 2 and 4, not the woven 200
    // The same cadence with a static woven field weaves it.
    Deinterlacer still;
    still.process(dark, 0, DeinterlaceMode::Adaptive, false, 1);
    still.process(bright, 1, DeinterlaceMode::Adaptive, false, 1);
    still.process(dark, 0, DeinterlaceMode::Adaptive, false, 1);
    still.process(bright, 1, DeinterlaceMode::Adaptive, false, 1);
    out = still.process(dark, 0, DeinterlaceMode::Adaptive, false, 1);
    CHECK_EQ(out.at(0, 3).r, 200);
}

void test_auto_deinterlace() {
    DisplayLayout layout;
    layout.both_enabled = false;
    CHECK(resolve_auto_deinterlace(layout, false, false) == DeinterlaceMode::Off);
    CHECK(resolve_auto_deinterlace(layout, true, false) == DeinterlaceMode::Adaptive);  // FFMD
    layout.one_circuit_dby_changed = true;
    CHECK(resolve_auto_deinterlace(layout, false, false) == DeinterlaceMode::Adaptive);  // (a)
    layout.one_circuit_dby_changed = false;
    layout.both_enabled = true;
    layout.circuits[0].screen_y = 0;
    layout.circuits[1].screen_y = 1;
    CHECK(resolve_auto_deinterlace(layout, false, true) == DeinterlaceMode::Off);  // one line apart
    layout.circuits[1].screen_y = 3;
    CHECK(resolve_auto_deinterlace(layout, false, true) == DeinterlaceMode::Adaptive);  // (c)
    CHECK(resolve_auto_deinterlace(layout, false, false) == DeinterlaceMode::Off);
}

// ----- 13.5 and 13.6 -------------------------------------------------------------------

void test_shadeboost() {
    // Default parameters (all 50 = 1.0) are the identity.
    Image image(256, 1);
    for (std::uint32_t x = 0; x < 256; ++x) {
        image.at(x, 0) = Rgba8{static_cast<std::uint8_t>(x), static_cast<std::uint8_t>(255 - x), 128, 77};
    }
    const Image before = image;
    apply_shadeboost(image, ShadeBoostParams{});
    bool identity = true;
    for (std::uint32_t x = 0; x < 256; ++x) {
        identity = identity && image.at(x, 0) == before.at(x, 0);
    }
    CHECK(identity);

    // Brightness 100 (2.0) saturates a mid grey to white.
    Image grey(1, 1, Rgba8{128, 128, 128, 255});
    apply_shadeboost(grey, ShadeBoostParams{100, 50, 50, 50});
    CHECK(grey.at(0, 0) == (Rgba8{255, 255, 255, 255}));

    // Saturation 0 gives the luma: pure red becomes 0.2125 * 255 = 54 on every channel.
    Image red(1, 1, Rgba8{255, 0, 0, 255});
    apply_shadeboost(red, ShadeBoostParams{50, 50, 0, 50});
    CHECK(red.at(0, 0) == (Rgba8{54, 54, 54, 255}));
}

void test_scanlines() {
    Image image(1, 4, Rgba8{200, 100, 50, 9});
    apply_scanlines(image);
    CHECK(image.at(0, 0) == (Rgba8{200, 100, 50, 9}));
    CHECK(image.at(0, 1) == (Rgba8{100, 50, 25, 9}));
    CHECK(image.at(0, 2) == (Rgba8{200, 100, 50, 9}));
    CHECK(image.at(0, 3) == (Rgba8{100, 50, 25, 9}));
}

void test_present_rect() {
    // 1920x1080 window, 4:3 target: r = 0.75, pillarboxed to width 1440 at x 240 (13.6 vector).
    PixelRect rect = presentation_rect(1920, 1080, 4.0 / 3.0, 1.0);
    CHECK_EQ(rect.x, 240);
    CHECK_EQ(rect.y, 0);
    CHECK_EQ(rect.width, 1440);
    CHECK_EQ(rect.height, 1080);
    CHECK_EQ(rect.x + rect.width, 1680);

    // 1280x1024 window, 16:9 target: r = 1.4222, letterboxed to height 720 at y 152.
    rect = presentation_rect(1280, 1024, 16.0 / 9.0, 1.0);
    CHECK_EQ(rect.x, 0);
    CHECK_EQ(rect.y, 152);
    CHECK_EQ(rect.width, 1280);
    CHECK_EQ(rect.height, 720);

    // The 16:9 window fills its own rectangle.
    rect = presentation_rect(1920, 1080);
    CHECK_EQ(rect.x, 0);
    CHECK_EQ(rect.y, 0);
    CHECK_EQ(rect.width, 1920);
    CHECK_EQ(rect.height, 1080);
}

void test_display_options() {
    CHECK(parse_antiblur_option("1") == std::optional<bool>(true));
    CHECK(parse_antiblur_option("0") == std::optional<bool>(false));
    CHECK(!parse_antiblur_option("yes").has_value());

    const auto adaptive = parse_deinterlace_option("adaptive");
    CHECK(adaptive.has_value());
    CHECK(adaptive->mode == DeinterlaceMode::Adaptive);
    CHECK(!adaptive->bottom_first);
    const auto bff = parse_deinterlace_option("blend_bff");
    CHECK(bff.has_value());
    CHECK(bff->mode == DeinterlaceMode::Blend);
    CHECK(bff->bottom_first);
    CHECK(parse_deinterlace_option("auto")->mode == DeinterlaceMode::Auto);
    CHECK(parse_deinterlace_option("off")->mode == DeinterlaceMode::Off);
    CHECK(!parse_deinterlace_option("yadif").has_value());
    CHECK(!parse_deinterlace_option("_bff").has_value());

    const auto sb = parse_shadeboost_option("50,60,70,40");
    CHECK(sb.has_value());
    CHECK_EQ(sb->brightness, 50);
    CHECK_EQ(sb->contrast, 60);
    CHECK_EQ(sb->saturation, 70);
    CHECK_EQ(sb->gamma, 40);
    CHECK(!parse_shadeboost_option("50,60,70").has_value());
    CHECK(!parse_shadeboost_option("50,60,70,0").has_value());
    CHECK(!parse_shadeboost_option("50,60,70,x").has_value());
    CHECK(!parse_shadeboost_option("50,60,70,40,1").has_value());

    const auto empty = parse_postfx_option("");
    CHECK(empty.has_value() && empty->empty());
    const auto list = parse_postfx_option("scanlines,shadeboost,scanlines");
    CHECK(list.has_value());
    CHECK_EQ(list->size(), 2);
    CHECK(list->at(0) == PostFx::ShadeBoost);  // fixed order of 13.5
    CHECK(list->at(1) == PostFx::Scanlines);
    CHECK(!parse_postfx_option("crt").has_value());
    CHECK(!parse_postfx_option("scanlines,").has_value());
}

struct TestEntry {
    const char* name;
    void (*run)();
};

}  // namespace

int main(int argc, char** argv) {
    const TestEntry tests[] = {
        {"hw_video_mode", test_video_mode},
        {"hw_pcrtc_rects", test_pcrtc_rects},
        {"hw_pcrtc_offsets", test_pcrtc_offsets},
        {"hw_merge_plan", test_merge_plan},
        {"hw_merge_compose", test_merge_compose},
        {"hw_interlace_phase", test_interlace_phase},
        {"hw_deint_weave_blend", test_deinterlace_weave_blend},
        {"hw_deint_bob", test_deinterlace_bob},
        {"hw_deint_adaptive", test_deinterlace_adaptive},
        {"hw_auto_deinterlace", test_auto_deinterlace},
        {"hw_shadeboost", test_shadeboost},
        {"hw_scanlines", test_scanlines},
        {"hw_present_rect", test_present_rect},
        {"hw_display_options", test_display_options},
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
