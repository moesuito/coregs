// SPDX-License-Identifier: Apache-2.0
//
// CTest fixtures for the HWR pixel reference (hw_gs_pixel.hpp).
// Copyright (c) 2026 GT4Recomp contributors.
//
// Written from the TASK-026 specification (docs/hw-renderer-spec). Vectors are the
// normative ones of chapters 01.5, 03.5, 08.1, 08.6, 08.7 and 14.6 that are pure pixel
// arithmetic. Vectors that compare against the reference renderer or against captures
// (V-01-4..7, V-01-9 reference check, V-01-10..12, V-02, V-05, V-07, V-09, V-10, V-11)
// are not here; they belong to the milestones that own their inputs.
//
// Usage: hw_gs_pixel_fixture [test-name ...]. With no arguments every test runs.

#include "hw_gs_pixel.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <tuple>
#include <string>
#include <vector>

using namespace coregs::pixel;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        ++g_checks;                                                                         \
        if (!(cond)) {                                                                      \
            ++g_failures;                                                                   \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);   \
        }                                                                                   \
    } while (0)

#define CHECK_EQ(actual, expected)                                                                        \
    do {                                                                                                  \
        ++g_checks;                                                                                       \
        const long long _actual = static_cast<long long>(actual);                                         \
        const long long _expected = static_cast<long long>(expected);                                     \
        if (_actual != _expected) {                                                                       \
            ++g_failures;                                                                                 \
            std::fprintf(stderr, "%s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__, #actual,       \
                         _actual, _expected);                                                             \
        }                                                                                                 \
    } while (0)

Colour make(std::int32_t r, std::int32_t g, std::int32_t b, std::int32_t a) {
    return Colour{r, g, b, a};
}

Colour grey(std::int32_t v, std::int32_t a) {
    return make(v, v, v, a);
}

bool same_enable(WriteEnable got, bool rgb, bool alpha, bool depth) {
    return got.colour_rgb == rgb && got.colour_alpha == alpha && got.depth == depth;
}

// Reference blend: independent of the library. Selectors: source 0 Cs, 1 Cd, 2 zero;
// factor 0 As, 1 Ad, 2 FIX. The shift uses floor(x / 128) instead of a bit shift.
std::int32_t reference_source(int selector, int cs, int cd) {
    if (selector == 0) return cs;
    if (selector == 1) return cd;
    return 0;
}

std::int32_t reference_blend(int a_sel, int b_sel, int c_sel, int d_sel, int cs, int cd, int as, int ad, int fix,
                             bool clamp) {
    const int factor = c_sel == 0 ? as : (c_sel == 1 ? ad : fix);
    const int a = reference_source(a_sel, cs, cd);
    const int b = reference_source(b_sel, cs, cd);
    const int d = reference_source(d_sel, cs, cd);
    const double product = static_cast<double>((a - b) * factor) / 128.0;
    const int value = static_cast<int>(std::floor(product)) + d;
    if (clamp) {
        return value < 0 ? 0 : (value > 255 ? 255 : value);
    }
    return ((value % 256) + 256) % 256;
}

// Deterministic 32-bit generator for the randomised sweeps (no libc rand).
struct Lcg {
    std::uint32_t state;
    std::uint32_t next() {
        state = state * 1664525u + 1013904223u;
        return state;
    }
};

// ---- Tests --------------------------------------------------------------------

void test_tfx_vectors() {
    const Colour t1 = make(1, 1, 1, 1);
    const Colour white = make(255, 255, 255, 255);
    // MODULATE: T = 1, C = 255 gives 1 (truncation).
    Colour out = texture_function(Tfx::Modulate, true, t1, make(255, 255, 255, 255));
    CHECK_EQ(out.r, 1);
    CHECK_EQ(out.a, 1);
    // MODULATE with T = C = 0x80 (1.0 each) gives 0x80 (1.0).
    out = texture_function(Tfx::Modulate, true, grey(0x80, 0x80), grey(0x80, 0x80));
    CHECK_EQ(out.r, 0x80);
    CHECK_EQ(out.a, 0x80);
    // MODULATE saturates at 255.
    out = texture_function(Tfx::Modulate, true, white, white);
    CHECK_EQ(out.r, 255);
    CHECK_EQ(out.a, 255);
    // TCC = 0: alpha is the fragment alpha.
    out = texture_function(Tfx::Modulate, false, make(9, 9, 9, 7), make(255, 255, 255, 33));
    CHECK_EQ(out.a, 33);
    // DECAL: rgb from the texel; alpha from the texel when TCC = 1.
    out = texture_function(Tfx::Decal, true, make(10, 20, 30, 40), make(200, 200, 200, 50));
    CHECK(out == make(10, 20, 30, 40));
    out = texture_function(Tfx::Decal, false, make(10, 20, 30, 40), make(200, 200, 200, 50));
    CHECK(out == make(10, 20, 30, 50));
    // HIGHLIGHT vector: T.rgb 100, C.rgb 128, Ca 50 gives rgb 150, alpha Ta + 50.
    out = texture_function(Tfx::Highlight, true, make(100, 100, 100, 10), make(128, 128, 128, 50));
    CHECK_EQ(out.r, 150);
    CHECK_EQ(out.g, 150);
    CHECK_EQ(out.a, 60);
    out = texture_function(Tfx::Highlight, false, make(100, 100, 100, 10), make(128, 128, 128, 50));
    CHECK_EQ(out.a, 50);
    // HIGHLIGHT saturates; HIGHLIGHT2 keeps the texel alpha.
    out = texture_function(Tfx::Highlight, true, white, make(255, 255, 255, 255));
    CHECK_EQ(out.r, 255);
    CHECK_EQ(out.a, 255);
    out = texture_function(Tfx::Highlight2, true, make(100, 100, 100, 10), make(128, 128, 128, 50));
    CHECK_EQ(out.r, 150);
    CHECK_EQ(out.a, 10);
    out = texture_function(Tfx::Highlight2, false, make(100, 100, 100, 10), make(128, 128, 128, 50));
    CHECK_EQ(out.a, 50);
}

void test_tfx_sweep() {
    // MODULATE over every texel and fragment value against an independent floor.
    int mismatches = 0;
    for (int t = 0; t < 256; ++t) {
        for (int c = 0; c < 256; ++c) {
            const int expected = std::min(255, static_cast<int>(std::floor(t * c / 128.0)));
            const Colour out = texture_function(Tfx::Modulate, true, grey(t, t), grey(c, c));
            if (out.r != expected || out.g != expected || out.b != expected || out.a != expected) {
                ++mismatches;
            }
        }
    }
    CHECK_EQ(mismatches, 0);
}

void test_texa_vectors() {
    // 0x8000: alpha from TA1 when bit 15 is set.
    Colour out = expand_texel16(0x8000, 0x40, 0x80, false);
    CHECK(out == make(0, 0, 0, 0x80));
    // 0x0000 with AEM: alpha forced to zero because RGB is zero.
    out = expand_texel16(0x0000, 0x40, 0x80, true);
    CHECK_EQ(out.a, 0);
    out = expand_texel16(0x0000, 0x40, 0x80, false);
    CHECK_EQ(out.a, 0x40);
    // 0x7FFF: 5-bit channels expanded as c << 3 (no replication), alpha from TA0.
    out = expand_texel16(0x7FFF, 0x40, 0x80, false);
    CHECK(out == make(248, 248, 248, 0x40));
    // 24-bit texels keep their bytes; AEM applies to black texels.
    out = expand_texel24(0x00FF00, 0x33, false);
    CHECK(out == make(0, 255, 0, 0x33));
    out = expand_texel24(0x000000, 0x33, true);
    CHECK_EQ(out.a, 0);
}

void test_texa_sweep() {
    int mismatches = 0;
    for (std::uint32_t texel = 0; texel < 65536; ++texel) {
        const bool aem = (texel & 1u) != 0;
        const Colour out = expand_texel16(static_cast<std::uint16_t>(texel), 0x21, 0xC3, aem);
        const bool black = (texel & 0x7FFFu) == 0;
        std::int32_t alpha = (texel & 0x8000u) != 0 ? 0xC3 : 0x21;
        // IN-07: the A bit wins over AEM; AEM only zeroes black texels with the A bit clear.
        if (aem && black && (texel & 0x8000u) == 0) {
            alpha = 0;
        }
        if (out.r != static_cast<std::int32_t>((texel & 0x1F) * 8) ||
            out.g != static_cast<std::int32_t>(((texel >> 5) & 0x1F) * 8) ||
            out.b != static_cast<std::int32_t>(((texel >> 10) & 0x1F) * 8) || out.a != alpha) {
            ++mismatches;
        }
    }
    CHECK_EQ(mismatches, 0);
}

void test_fog_vectors() {
    // F = 128, FOGCOL = 0, C = 200 gives 100; F = 255 gives 199; alpha untouched.
    Colour out = apply_fog(make(200, 200, 200, 77), 128, make(0, 0, 0, 0));
    CHECK(out == make(100, 100, 100, 77));
    out = apply_fog(make(200, 200, 200, 77), 255, make(0, 0, 0, 0));
    CHECK(out == make(199, 199, 199, 77));
    // F = 0 is the fog colour, truncated: (255 * 255) >> 8 = 254.
    out = apply_fog(make(10, 10, 10, 0), 0, make(255, 255, 255, 0));
    CHECK_EQ(out.r, 254);
}

void test_fog_sweep() {
    int violations = 0;
    for (int c = 0; c < 256; ++c) {
        for (int fog = 0; fog < 256; ++fog) {
            for (int fc = 0; fc < 256; fc += 3) {
                const Colour out = apply_fog(grey(c, 9), fog, grey(fc, 0));
                const int low = std::min(c, fc) - 1;
                const int high = std::max(c, fc);
                if (out.r < low || out.r > high || out.r < 0 || out.r > 255 || out.a != 9) {
                    ++violations;
                }
            }
        }
    }
    CHECK_EQ(violations, 0);
}

void test_alpha_test_enumeration() {
    // Every compare and reference with a full sweep of fragment alpha ranges. Ground
    // truth is the per-value outcome; fold must be exact (AllPass iff every value
    // passes, AllFail iff none does).
    int wrong = 0;
    int checked = 0;
    for (int compare = 0; compare < 8; ++compare) {
        for (int reference = 0; reference < 256; ++reference) {
            std::vector<int> prefix(257, 0);
            for (int v = 0; v < 256; ++v) {
                prefix[v + 1] = prefix[v] + (alpha_compare(static_cast<AlphaCompare>(compare), v, reference) ? 1 : 0);
            }
            for (int lo = 0; lo < 256; ++lo) {
                for (int hi = lo; hi < 256; ++hi) {
                    const int passes = prefix[hi + 1] - prefix[lo];
                    const int count = hi - lo + 1;
                    AlphaFold expected = AlphaFold::Mixed;
                    if (passes == count) {
                        expected = AlphaFold::AllPass;
                    } else if (passes == 0) {
                        expected = AlphaFold::AllFail;
                    }
                    const AlphaFold got =
                        fold_alpha_test(static_cast<AlphaCompare>(compare), lo, hi, reference);
                    ++checked;
                    if (got != expected) {
                        ++wrong;
                    }
                }
            }
        }
    }
    CHECK_EQ(wrong, 0);
    CHECK_EQ(checked, 8 * 256 * (256 * 257 / 2));
}

void test_alpha_test_vectors() {
    // LESS against 0x80: [0x10, 0x7F] all pass, [0x80, 0xFF] all fail, [0x70, 0x90] mixed.
    CHECK(fold_alpha_test(AlphaCompare::Less, 0x10, 0x7F, 0x80) == AlphaFold::AllPass);
    CHECK(fold_alpha_test(AlphaCompare::Less, 0x80, 0xFF, 0x80) == AlphaFold::AllFail);
    CHECK(fold_alpha_test(AlphaCompare::Less, 0x70, 0x90, 0x80) == AlphaFold::Mixed);
    // NOTEQUAL is all pass when the reference lies outside the range.
    CHECK(fold_alpha_test(AlphaCompare::NotEqual, 0x90, 0xFF, 0x80) == AlphaFold::AllPass);
    CHECK(fold_alpha_test(AlphaCompare::NotEqual, 0x80, 0x80, 0x80) == AlphaFold::AllFail);

    // Outcome masks for each AFAIL, passing and failing.
    AlphaTestState test;
    test.enabled = true;
    test.compare = AlphaCompare::Greater;
    test.reference = 0x80;

    test.fail = AlphaFail::Keep;
    CHECK(same_enable(alpha_outcome_write_enable(test, false, FrameFormat::Ct32), false, false, false));
    test.fail = AlphaFail::FrameOnly;
    CHECK(same_enable(alpha_outcome_write_enable(test, false, FrameFormat::Ct32), true, true, false));
    test.fail = AlphaFail::DepthOnly;
    CHECK(same_enable(alpha_outcome_write_enable(test, false, FrameFormat::Ct32), false, false, true));
    test.fail = AlphaFail::RgbOnly;
    CHECK(same_enable(alpha_outcome_write_enable(test, false, FrameFormat::Ct32), true, false, false));
    // High: RGB_ONLY on a non-32-bit frame acts as FB_ONLY.
    CHECK(same_enable(alpha_outcome_write_enable(test, false, FrameFormat::Ct16), true, true, false));
    CHECK(same_enable(alpha_outcome_write_enable(test, false, FrameFormat::Ct24), true, true, false));
    // Passing fragments always write everything.
    for (int fail = 0; fail < 4; ++fail) {
        test.fail = static_cast<AlphaFail>(fail);
        CHECK(same_enable(alpha_outcome_write_enable(test, true, FrameFormat::Ct32), true, true, true));
    }
    // Alpha 0x90 passes GREATER 0x80; alpha 0x40 fails it.
    test.fail = AlphaFail::DepthOnly;
    CHECK(same_enable(alpha_test_write_enable(test, 0x90, FrameFormat::Ct32), true, true, true));
    CHECK(same_enable(alpha_test_write_enable(test, 0x40, FrameFormat::Ct32), false, false, true));
    // Disabled test writes everything whatever the alpha.
    test.enabled = false;
    CHECK(same_enable(alpha_test_write_enable(test, 0x00, FrameFormat::Ct32), true, true, true));
}

void test_blend_vectors() {
    // Vector 1: (Cs - Cd) * As + Cd with Cs 200, Cd 100, As 0x40 gives 150.
    BlendState mode;
    mode.enabled = true;
    decode_alpha_register(encode_alpha_register(BlendSource::Cs, BlendSource::Cd, BlendFactor::As, BlendSource::Cd, 0),
                          mode);
    Colour out = blend_unclamped(mode, grey(200, 0), grey(100, 0), 0x40, 0);
    CHECK_EQ(out.r, 150);

    // Vector 2: (Cs - 0) * As + 0 with Cs 200, As 0xFF gives 398, clamp 255, wrap 142.
    decode_alpha_register(encode_alpha_register(BlendSource::Cs, BlendSource::Zero, BlendFactor::As, BlendSource::Zero, 0),
                          mode);
    out = blend_unclamped(mode, grey(200, 0), grey(0, 0), 0xFF, 0);
    CHECK_EQ(out.r, 398);
    CHECK_EQ(colour_clamp_or_wrap(out.r, true), 255);
    CHECK_EQ(colour_clamp_or_wrap(out.r, false), 142);

    // Vector 3: (Cd - Cs) * As + 0 with Cd 10, Cs 20, As 0x80 gives -10, clamp 0, wrap 246.
    decode_alpha_register(encode_alpha_register(BlendSource::Cd, BlendSource::Cs, BlendFactor::As, BlendSource::Zero, 0),
                          mode);
    out = blend_unclamped(mode, grey(20, 0), grey(10, 0), 0x80, 0);
    CHECK_EQ(out.r, -10);
    CHECK_EQ(colour_clamp_or_wrap(out.r, true), 0);
    CHECK_EQ(colour_clamp_or_wrap(out.r, false), 246);

    // Vector 4 (Hypothesis V-01-9, arithmetic shift): (Cd - Cs) * FIX with Cd 0, Cs 1, FIX 0x81
    // gives -129 >> 7 = -2, clamp 0, wrap 254.
    decode_alpha_register(encode_alpha_register(BlendSource::Cd, BlendSource::Cs, BlendFactor::Fix, BlendSource::Zero, 0x81),
                          mode);
    out = blend_unclamped(mode, grey(1, 0), grey(0, 0), 0, 0);
    CHECK_EQ(out.r, -2);
    CHECK_EQ(colour_clamp_or_wrap(out.r, true), 0);
    CHECK_EQ(colour_clamp_or_wrap(out.r, false), 254);

    // Vector 5: Cs * As with Cs 1 and As 0xFF gives 1 (truncation).
    decode_alpha_register(encode_alpha_register(BlendSource::Cs, BlendSource::Zero, BlendFactor::As, BlendSource::Zero, 0),
                          mode);
    out = blend_unclamped(mode, grey(1, 0), grey(0, 0), 0xFF, 0);
    CHECK_EQ(out.r, 1);

    // Vector 6: 16-bit frame, DIMX entry (0,0) = -4, blended value 0x13 gives 0x0F, then
    // truncation to 0x08. Blend off so the value is the source colour.
    PixelState state;
    state.frame.format = FrameFormat::Ct16;
    state.frame.dither = true;
    state.frame.dimx = 0x4;  // entry (0,0) = field 4 = -4
    state.blend.enabled = false;
    Fragment fragment;
    fragment.x = 0;
    fragment.y = 0;
    fragment.colour = grey(0x13, 0x80);
    fragment.z = 0;
    const PixelOutcome written = process_pixel(state, fragment, 0, 0);
    CHECK(written.frame_written);
    CHECK_EQ(frame_word_to_colour(unpack_frame_word(FrameFormat::Ct16, written.frame_stored)).r, 0x08);
}

void test_blend_mode_enumeration() {
    // ALPHA register: 256 encodings decode to exactly 81 distinct modes.
    std::set<std::tuple<int, int, int, int>> modes;
    for (std::uint64_t reg = 0; reg < 256; ++reg) {
        BlendState state;
        decode_alpha_register(reg, state);
        modes.insert({static_cast<int>(state.a), static_cast<int>(state.b), static_cast<int>(state.c),
                      static_cast<int>(state.d)});
    }
    CHECK_EQ(modes.size(), 81);
    // Round trip of every mode through encode and decode.
    int round_trip_failures = 0;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int c = 0; c < 3; ++c) {
                for (int d = 0; d < 3; ++d) {
                    const auto reg = encode_alpha_register(static_cast<BlendSource>(a), static_cast<BlendSource>(b),
                                                           static_cast<BlendFactor>(c), static_cast<BlendSource>(d), 0x5A);
                    BlendState state;
                    decode_alpha_register(reg, state);
                    if (state.a != static_cast<BlendSource>(a) || state.b != static_cast<BlendSource>(b) ||
                        state.c != static_cast<BlendFactor>(c) || state.d != static_cast<BlendSource>(d) ||
                        state.fix != 0x5A) {
                        ++round_trip_failures;
                    }
                }
            }
        }
    }
    CHECK_EQ(round_trip_failures, 0);
    // Selector value 3 behaves as 2 (zero for A, B, D; FIX for C).
    BlendState state;
    decode_alpha_register(0xFF, state);
    CHECK(state.a == BlendSource::Zero && state.b == BlendSource::Zero && state.d == BlendSource::Zero);
    CHECK(state.c == BlendFactor::Fix);
}

void test_blend_sweep_81_modes() {
    // Every mode, every Cs and Cd, two factor-alpha values and two destination alphas,
    // both COLCLAMP settings, against the reference model.
    int mismatches = 0;
    int checked = 0;
    const int as_values[2] = {0x40, 0xFF};
    const int ad_values[2] = {0x00, 0x80};
    const int fix = 0x81;
    for (int mode = 0; mode < 81; ++mode) {
        const int a_sel = mode % 3;
        const int b_sel = (mode / 3) % 3;
        const int c_sel = (mode / 9) % 3;
        const int d_sel = (mode / 27) % 3;
        BlendState state;
        state.enabled = true;
        state.a = static_cast<BlendSource>(a_sel);
        state.b = static_cast<BlendSource>(b_sel);
        state.c = static_cast<BlendFactor>(c_sel);
        state.d = static_cast<BlendSource>(d_sel);
        state.fix = static_cast<std::uint8_t>(fix);
        for (int clamp = 0; clamp < 2; ++clamp) {
            state.colour_clamp = clamp == 1;
            for (int cs = 0; cs < 256; ++cs) {
                for (int cd = 0; cd < 256; ++cd) {
                    for (int ai = 0; ai < 2; ++ai) {
                        for (int di = 0; di < 2; ++di) {
                            const int as = as_values[ai];
                            const int ad = ad_values[di];
                            const Colour blended = blend_unclamped(state, grey(cs, as), grey(cd, ad), as, ad);
                            const std::int32_t got = colour_clamp_or_wrap(blended.r, state.colour_clamp);
                            const int want = reference_blend(a_sel, b_sel, c_sel, d_sel, cs, cd, as, ad, fix, clamp == 1);
                            ++checked;
                            if (got != want) {
                                ++mismatches;
                            }
                        }
                    }
                }
            }
        }
    }
    CHECK_EQ(mismatches, 0);
    CHECK_EQ(checked, 81 * 2 * 256 * 256 * 2 * 2);
}

void test_colour_clamp_vectors() {
    CHECK_EQ(colour_clamp_or_wrap(260, true), 255);
    CHECK_EQ(colour_clamp_or_wrap(260, false), 4);
    CHECK_EQ(colour_clamp_or_wrap(-3, true), 0);
    CHECK_EQ(colour_clamp_or_wrap(-3, false), 253);
}

void test_dither_vectors() {
    // Entry (row i, column j) at bit 4 * (4i + j); value -4 is field 0b100.
    std::uint64_t dimx = 0;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            const int value = ((row * 4 + column) % 8) - 4;  // -4..3
            dimx |= static_cast<std::uint64_t>(value & 7) << (4 * (4 * row + column));
        }
    }
    int wrong = 0;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            const int value = ((row * 4 + column) % 8) - 4;
            if (dither_entry(dimx, column, row) != value) {
                ++wrong;
            }
            // Coordinates wrap modulo 4 (only the low two bits index the matrix).
            if (dither_entry(dimx, column + 8, row - 4) != value) {
                ++wrong;
            }
        }
    }
    CHECK_EQ(wrong, 0);
    // Entry (1,2) = +3 alone: field 3 at bit 24.
    CHECK_EQ(dither_entry(3ull << 24, 2, 1), 3);
    CHECK_EQ(dither_entry(0x4ull, 0, 0), -4);
    // Dither applies to RGB only.
    const Colour dithered = apply_dither(make(10, 20, 30, 40), -4);
    CHECK(dithered == make(6, 16, 26, 40));

    // Dither is applied to 16-bit frames only.
    PixelState state;
    state.frame.dither = true;
    state.frame.dimx = 0x4;
    state.blend.enabled = false;
    Fragment fragment;
    fragment.colour = grey(0x13, 0x00);
    const PixelOutcome out32 = [&] {
        state.frame.format = FrameFormat::Ct32;
        return process_pixel(state, fragment, 0, 0);
    }();
    CHECK_EQ(frame_word_to_colour(out32.frame_stored).r, 0x13);
}

void test_frame_conversion() {
    // Every 16-bit stored pixel survives unpack and pack, and the word stays in the valid mask.
    int broken = 0;
    for (std::uint32_t stored = 0; stored < 65536; ++stored) {
        const std::uint32_t word = unpack_frame_word(FrameFormat::Ct16, stored);
        if ((word & ~frame_valid_mask(FrameFormat::Ct16)) != 0 || pack_frame_word(FrameFormat::Ct16, word) != stored) {
            ++broken;
        }
    }
    CHECK_EQ(broken, 0);
    CHECK_EQ(unpack_frame_word(FrameFormat::Ct16, 0x7FFF), 0x00F8F8F8u);  // alpha bit 15 clear
    CHECK_EQ(unpack_frame_word(FrameFormat::Ct16, 0x8000), 0x80000000u);

    // CT16 write truncation: channel values shift right by three.
    CHECK_EQ(pack_frame_word(FrameFormat::Ct16, colour_to_frame_word(FrameFormat::Ct16, grey(255, 255))), 0xFFFF);
    CHECK_EQ(pack_frame_word(FrameFormat::Ct16, colour_to_frame_word(FrameFormat::Ct16, grey(7, 0))), 0);
    CHECK_EQ(pack_frame_word(FrameFormat::Ct16, colour_to_frame_word(FrameFormat::Ct16, grey(8, 0))), 0x0421);
    // Alpha bit 7 becomes the 16-bit alpha bit: 0x7F is off, 0x80 is on.
    CHECK_EQ(pack_frame_word(FrameFormat::Ct16, colour_to_frame_word(FrameFormat::Ct16, grey(0, 0x7F))), 0);
    CHECK_EQ(pack_frame_word(FrameFormat::Ct16, colour_to_frame_word(FrameFormat::Ct16, grey(0, 0x80))), 0x8000);

    // CT24: alpha is not stored.
    CHECK_EQ(colour_to_frame_word(FrameFormat::Ct24, make(1, 2, 3, 255)), 0x00030201u);
    CHECK_EQ(unpack_frame_word(FrameFormat::Ct24, 0xFF112233u), 0x00112233u);
    CHECK_EQ(pack_frame_word(FrameFormat::Ct24, 0xAB112233u), 0x00112233u);

    // CT32 is the identity on the four lanes.
    CHECK_EQ(colour_to_frame_word(FrameFormat::Ct32, make(1, 2, 3, 4)), 0x04030201u);
    CHECK(frame_word_to_colour(0x04030201u) == make(1, 2, 3, 4));

    // FBA sets bit 7 of the written alpha.
    CHECK_EQ(apply_fba(grey(0, 0), true).a, 0x80);
    CHECK_EQ(apply_fba(grey(0, 0x10), true).a, 0x90);
    CHECK_EQ(apply_fba(grey(0, 0x10), false).a, 0x10);

    // Reading a CT16 pixel back: 5-bit channels become c << 3.
    CHECK(frame_word_to_colour(unpack_frame_word(FrameFormat::Ct16, 0x7FFF)) == make(248, 248, 248, 0));
}

void test_fbmask_merge() {
    const WriteEnable all{true, true, true};
    // Unprotected: computed wins.
    CHECK_EQ(merge_frame_word(FrameFormat::Ct32, 0x11223344, 0xAABBCCDD, 0x00000000, all), 0xAABBCCDDu);
    // Fully protected: old stays.
    CHECK_EQ(merge_frame_word(FrameFormat::Ct32, 0x11223344, 0xAABBCCDD, 0xFFFFFFFF, all), 0x11223344u);
    // Partial: low half protected.
    CHECK_EQ(merge_frame_word(FrameFormat::Ct32, 0x11223344, 0xAABBCCDD, 0x0000FFFF, all), (0xAABBCCDDu & 0xFFFF0000u) | 0x3344u);
    // Alpha only enabled: only the top byte comes from computed.
    CHECK_EQ(merge_frame_word(FrameFormat::Ct32, 0x11223344, 0xAABBCCDD, 0, WriteEnable{false, true, false}),
             0xAA223344u);
    // RGB only enabled with the same inputs: alpha byte from old.
    CHECK_EQ(merge_frame_word(FrameFormat::Ct32, 0x11223344, 0xAABBCCDD, 0, WriteEnable{true, false, false}),
             0x11BBCCDDu);
    // CT24 never writes alpha lane bits even if enabled.
    CHECK_EQ(merge_frame_word(FrameFormat::Ct24, 0x00112233, 0xAABBCCDD, 0, all), 0x00BBCCDDu);
    // CT16 valid bits only: the protected low three bits of the channel lanes stay.
    CHECK_EQ(merge_frame_word(FrameFormat::Ct16, 0x00000007, 0x000000F8, 0, all), 0x000000FFu);
    CHECK_EQ(merge_frame_word(FrameFormat::Ct16, 0x0, 0x80F8F8F8, 0x80F8F8F8, all), 0);

    // Random sweep against the per-bit rule: bit written iff enabled, valid and unprotected.
    Lcg rng{12345u};
    int wrong = 0;
    for (int i = 0; i < 20000; ++i) {
        const std::uint32_t old_word = rng.next();
        const std::uint32_t computed = rng.next();
        const std::uint32_t fbmsk = rng.next();
        const int format_index = static_cast<int>(rng.next() % 3);
        const FrameFormat format = static_cast<FrameFormat>(format_index);
        const WriteEnable enable{(rng.next() & 1) != 0, (rng.next() & 1) != 0, false};
        const std::uint32_t valid = frame_valid_mask(format);
        std::uint32_t expected = 0;
        for (int bit = 0; bit < 32; ++bit) {
            const std::uint32_t mask = 1u << bit;
            const bool lane_enabled = (bit < 24) ? enable.colour_rgb : enable.colour_alpha;
            const bool write = lane_enabled && (valid & mask) != 0 && (fbmsk & mask) == 0;
            expected |= (write ? computed : old_word) & mask;
        }
        if (merge_frame_word(format, old_word, computed, fbmsk, enable) != expected) {
            ++wrong;
        }
    }
    CHECK_EQ(wrong, 0);
}

void test_date_vectors() {
    // V-01-8: CT24 frame with DATE = 1 writes nothing (colour or Z), even with an
    // always-passing depth test.
    PixelState state;
    state.frame.format = FrameFormat::Ct24;
    state.date.enabled = true;
    state.date.expected_msb = false;
    state.depth.test_enabled = true;
    state.depth.compare = DepthCompare::Always;
    state.z_format = ZFormat::Z24;
    Fragment fragment;
    fragment.colour = grey(200, 0x40);
    fragment.z = 1000;
    PixelOutcome out = process_pixel(state, fragment, 0x00123456u, 0x00000001u);
    CHECK(!out.frame_written);
    CHECK(!out.depth_written);
    CHECK_EQ(out.frame_stored, 0x00123456u);
    CHECK_EQ(out.depth_stored, 1u);
    state.date.enabled = false;
    out = process_pixel(state, fragment, 0x00123456u, 0x00000001u);
    CHECK(out.frame_written);
    CHECK(out.depth_written);

    // V-08-1: DATM = 0, two overlapping sprites. The first writes alpha 0x80 on the shared
    // pixel, so the second is rejected there. Pixels with alpha below 0x80 still pass.
    PixelState sprite;
    sprite.frame.format = FrameFormat::Ct32;
    sprite.date.enabled = true;
    sprite.date.expected_msb = false;
    std::uint32_t frame[3] = {0, 0, 0};
    const int first_x[2] = {0, 1};
    for (int x : first_x) {
        Fragment f;
        f.x = x;
        f.colour = grey(0x10, 0x80);
        frame[x] = process_pixel(sprite, f, frame[x], 0).frame_stored;
    }
    for (int x = 1; x <= 2; ++x) {
        Fragment f;
        f.x = x;
        f.colour = grey(0x20, 0x40);
        const PixelOutcome written = process_pixel(sprite, f, frame[x], 0);
        if (written.frame_written) {
            frame[x] = written.frame_stored;
        }
    }
    CHECK_EQ(frame[0] & 0xFFu, 0x10);
    CHECK_EQ(frame[1] & 0xFFu, 0x10);  // rejected by the second sprite
    CHECK_EQ(frame[2] & 0xFFu, 0x20);  // only the second sprite covers it
    CHECK_EQ(frame[1] >> 24, 0x80);

    // Destination alpha on 16-bit frames: DATM = 1 passes where bit 15 is set.
    PixelState sixteen;
    sixteen.frame.format = FrameFormat::Ct16;
    sixteen.date.enabled = true;
    sixteen.date.expected_msb = true;
    Fragment f;
    f.colour = grey(0x80, 0x00);
    CHECK(!process_pixel(sixteen, f, 0x0000, 0).frame_written);
    CHECK(process_pixel(sixteen, f, 0x8000, 0).frame_written);
    CHECK_EQ(destination_alpha(FrameFormat::Ct24, 0x00000000u), 0x80);
    CHECK_EQ(destination_alpha(FrameFormat::Ct16, 0x80000000u), 0x80);
}

void test_depth_vectors() {
    CHECK_EQ(clamp_z(0x1000000ull, ZFormat::Z24), 0xFFFFFFu);
    CHECK_EQ(clamp_z(0x10000ull, ZFormat::Z16), 0xFFFFu);
    CHECK_EQ(clamp_z(0x100000005ull, ZFormat::Z32), 0xFFFFFFFFu);
    CHECK_EQ(clamp_z(5, ZFormat::Z16), 5);

    CHECK_EQ(interpolated_z(3.9, ZFormat::Z32), 3);
    CHECK_EQ(interpolated_z(-1.0, ZFormat::Z32), 0);
    CHECK_EQ(interpolated_z(1e12, ZFormat::Z24), 0xFFFFFFu);

    // GEQUAL and GREATER on unsigned bits; Z24 ignores the top byte of the stored word.
    CHECK(depth_passes(DepthCompare::GreaterEqual, 5, 5, ZFormat::Z32));
    CHECK(!depth_passes(DepthCompare::GreaterEqual, 4, 5, ZFormat::Z32));
    CHECK(!depth_passes(DepthCompare::Greater, 5, 5, ZFormat::Z32));
    CHECK(depth_passes(DepthCompare::Greater, 6, 5, ZFormat::Z32));
    CHECK(depth_passes(DepthCompare::GreaterEqual, 5, 0xFF000005u, ZFormat::Z24));
    CHECK(!depth_passes(DepthCompare::Never, 0xFFFFFFFFu, 0, ZFormat::Z32));
    CHECK(depth_passes(DepthCompare::Always, 0, 0xFFFFFFFFu, ZFormat::Z32));

    // Writes keep the bits above the format width.
    CHECK_EQ(write_depth_raw(0xAB000000u, 0x123456u, ZFormat::Z24), 0xAB123456u);
    CHECK_EQ(write_depth_raw(0xABCD0000u, 0x1234u, ZFormat::Z16), 0xABCD1234u);
    CHECK_EQ(write_depth_raw(0xAB000000u, 0x12345678u, ZFormat::Z32), 0x12345678u);
    CHECK_EQ(depth_bits(0xAB123456u, ZFormat::Z24), 0x123456u);
}

void test_wrap_vectors() {
    const WrapAxis repeat{WrapMode::Repeat, 0, 0};
    const WrapAxis clamp{WrapMode::Clamp, 0, 0};
    const WrapAxis region_clamp{WrapMode::RegionClamp, 8, 20};
    const WrapAxis region_repeat{WrapMode::RegionRepeat, 0x0F, 0x20};

    // 64 texels (TW = 6).
    CHECK_EQ(wrap_texel(-1, 6, repeat), 63);
    CHECK_EQ(wrap_texel(64, 6, repeat), 0);
    CHECK_EQ(wrap_texel(-1, 6, clamp), 0);
    CHECK_EQ(wrap_texel(100, 6, clamp), 63);
    CHECK_EQ(wrap_texel(5, 6, region_clamp), 8);
    CHECK_EQ(wrap_texel(25, 6, region_clamp), 20);
    CHECK_EQ(wrap_texel(10, 6, region_clamp), 10);
    CHECK_EQ(wrap_texel(0x35, 6, region_repeat), 0x25);  // (0x35 & 0x0F) | 0x20
    CHECK_EQ(wrap_texel(-1, 6, region_repeat), 0x2F);

    // Repeat equals floor-mod for every texel in a wide range.
    int wrong = 0;
    for (int t = -300; t <= 300; ++t) {
        int want = ((t % 16) + 16) % 16;
        if (wrap_texel(t, 4, WrapAxis{WrapMode::Repeat, 0, 0}) != want) {
            ++wrong;
        }
    }
    CHECK_EQ(wrong, 0);

    // Texel addressing: nearest is floor(u); bilinear uses floor(u - 0.5) and a 4-bit fraction.
    CHECK_EQ(nearest_texel(0x18), 1);  // 1.5 texels
    CHECK_EQ(nearest_texel(-1), -1);
    BilinearSample s = bilinear_base(0x10);  // 1.0 texel
    CHECK_EQ(s.base, 0);
    CHECK_EQ(s.fraction, 8);
    s = bilinear_base(0x18);  // 1.5 texels
    CHECK_EQ(s.base, 1);
    CHECK_EQ(s.fraction, 0);
    s = bilinear_base(0x00);  // 0.0 texels
    CHECK_EQ(s.base, -1);
    CHECK_EQ(s.fraction, 8);

    // Textures above 1024 texels need a region mode.
    CHECK(texture_axis_usable(10, WrapMode::Repeat));
    CHECK(!texture_axis_usable(11, WrapMode::Repeat));
    CHECK(!texture_axis_usable(11, WrapMode::Clamp));
    CHECK(texture_axis_usable(11, WrapMode::RegionClamp));

    // Mip level sizes, never below one texel.
    CHECK_EQ(mip_level_size(6, 0), 64);
    CHECK_EQ(mip_level_size(6, 1), 32);
    CHECK_EQ(mip_level_size(6, 6), 1);
    CHECK_EQ(mip_level_size(6, 7), 1);
    CHECK_EQ(mip_level_size(3, 5), 1);
}

void test_pipeline_vectors() {
    // Alpha test FB_ONLY, failing: colour and alpha written, depth not.
    PixelState state;
    state.frame.format = FrameFormat::Ct32;
    state.alpha_test.enabled = true;
    state.alpha_test.compare = AlphaCompare::GreaterEqual;
    state.alpha_test.reference = 0x80;
    state.depth.test_enabled = true;
    state.depth.compare = DepthCompare::Always;
    state.z_format = ZFormat::Z32;
    state.alpha_test.fail = AlphaFail::FrameOnly;
    Fragment fragment;
    fragment.colour = grey(0x44, 0x10);
    fragment.z = 77;
    PixelOutcome out = process_pixel(state, fragment, 0xFF000000u, 0);
    CHECK(out.frame_written);
    CHECK_EQ(out.frame_stored, 0x10444444u);
    CHECK(!out.depth_written);

    // KEEP: nothing written.
    state.alpha_test.fail = AlphaFail::Keep;
    out = process_pixel(state, fragment, 0x12345678u, 5);
    CHECK(!out.frame_written && !out.depth_written);
    CHECK_EQ(out.frame_stored, 0x12345678u);

    // DepthOnly: Z written, colour kept.
    state.alpha_test.fail = AlphaFail::DepthOnly;
    out = process_pixel(state, fragment, 0x12345678u, 5);
    CHECK(!out.frame_written);
    CHECK(out.depth_written);
    CHECK_EQ(out.depth_stored, 77u);

    // RgbOnly on CT32: RGB written, alpha kept, no Z.
    state.alpha_test.fail = AlphaFail::RgbOnly;
    out = process_pixel(state, fragment, 0xAA000000u, 5);
    CHECK(out.frame_written);
    CHECK_EQ(out.frame_stored, 0xAA444444u);
    CHECK(!out.depth_written);

    // Depth test failure discards the fragment.
    state.alpha_test.enabled = false;
    state.depth.test_enabled = true;
    state.depth.compare = DepthCompare::GreaterEqual;
    out = process_pixel(state, fragment, 0x12345678u, 100);
    CHECK(!out.frame_written && !out.depth_written);

    // ZMSK = 1: colour written, Z not.
    state.depth.compare = DepthCompare::Always;
    state.depth.write_masked = true;
    out = process_pixel(state, fragment, 0x12345678u, 100);
    CHECK(out.frame_written);
    CHECK(!out.depth_written);
    CHECK_EQ(out.depth_stored, 100u);

    // ZTE = 0: no depth test and no Z write.
    state.depth.write_masked = false;
    state.depth.test_enabled = false;
    out = process_pixel(state, fragment, 0x12345678u, 100);
    CHECK(out.frame_written && !out.depth_written);

    // Z clamp to format maximum on write (Z24).
    state.depth.test_enabled = true;
    state.z_format = ZFormat::Z24;
    state.alpha_test.enabled = false;
    fragment.z = 0x2000000u;
    out = process_pixel(state, fragment, 0, 0x00000000u);
    CHECK(out.depth_written);
    CHECK_EQ(out.depth_stored, 0x00FFFFFFu);

    // Blending with and without PABE for a pixel whose alpha is below 0x80.
    PixelState blend;
    blend.frame.format = FrameFormat::Ct32;
    blend.blend.enabled = true;
    blend.blend.colour_clamp = true;
    decode_alpha_register(encode_alpha_register(BlendSource::Cs, BlendSource::Zero, BlendFactor::As, BlendSource::Zero, 0),
                          blend.blend);
    Fragment blended;
    blended.colour = grey(200, 0x40);
    out = process_pixel(blend, blended, 0, 0);
    CHECK_EQ(frame_word_to_colour(out.frame_stored).r, 100);  // 200 * 0x40 >> 7
    blend.blend.pabe = true;
    out = process_pixel(blend, blended, 0, 0);
    CHECK_EQ(frame_word_to_colour(out.frame_stored).r, 200);  // unblended
    CHECK_EQ(frame_word_to_colour(out.frame_stored).a, 0x40);
    blended.colour = grey(200, 0xFF);
    out = process_pixel(blend, blended, 0, 0);
    CHECK_EQ(frame_word_to_colour(out.frame_stored).r, 255);  // 398 clamped, PABE blends this one
    blend.blend.colour_clamp = false;
    out = process_pixel(blend, blended, 0, 0);
    CHECK_EQ(frame_word_to_colour(out.frame_stored).r, 142);  // wrap keeps the low 8 bits

    // FBA on a CT16 frame sets the alpha bit while the blend uses the original As.
    PixelState fba;
    fba.frame.format = FrameFormat::Ct16;
    fba.frame.fba = true;
    fba.blend.enabled = true;
    decode_alpha_register(encode_alpha_register(BlendSource::Cs, BlendSource::Zero, BlendFactor::As, BlendSource::Zero, 0),
                          fba.blend);
    Fragment zero_alpha;
    zero_alpha.colour = grey(0xF8, 0x00);
    out = process_pixel(fba, zero_alpha, 0, 0);
    CHECK_EQ(out.frame_stored, 0x8000u);  // blend uses As = 0 (colour 0), FBA sets the alpha bit
}

}  // namespace

int main(int argc, char** argv) {
    struct Entry {
        const char* name;
        void (*run)();
    };
    const Entry tests[] = {
        {"hw_tfx_vectors", test_tfx_vectors},
        {"hw_tfx_sweep", test_tfx_sweep},
        {"hw_texa_vectors", test_texa_vectors},
        {"hw_texa_sweep", test_texa_sweep},
        {"hw_fog_vectors", test_fog_vectors},
        {"hw_fog_sweep", test_fog_sweep},
        {"hw_atest_fold", test_alpha_test_enumeration},
        {"hw_atest_vectors", test_alpha_test_vectors},
        {"hw_blend_vectors", test_blend_vectors},
        {"hw_blend_modes", test_blend_mode_enumeration},
        {"hw_blend_sweep", test_blend_sweep_81_modes},
        {"hw_colclamp_vectors", test_colour_clamp_vectors},
        {"hw_dither_vectors", test_dither_vectors},
        {"hw_frame_convert", test_frame_conversion},
        {"hw_fbmask_merge", test_fbmask_merge},
        {"hw_date_vectors", test_date_vectors},
        {"hw_depth_vectors", test_depth_vectors},
        {"hw_wrap_vectors", test_wrap_vectors},
        {"hw_pipeline_vectors", test_pipeline_vectors},
    };

    int ran = 0;
    for (const Entry& entry : tests) {
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
