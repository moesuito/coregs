// SPDX-License-Identifier: MIT
//
// CTest fixtures for the HWR mipmap and LOD reference (hw_gs_lod.hpp).
// Copyright (c) 2026 GT4Recomp contributors.
//
// Written from the TASK-026 specification (docs/hw-renderer-spec). Vectors: 01.5 (TEX1
// fields, LOD formula, MTBA vector 1024/1280/1344 with TBW 2/1/1, MIPTBP layout), 03.6
// (base level, modes, filter selection) and 14.6 V-01-11 (nearest rounding, Hypothesis:
// the fixture checks the implemented rule; the owner comparison is open question Q-09).
//
// Usage: hw_gs_lod_fixture [test-name ...]. With no arguments every test runs.

#include "hw_gs_lod.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>

using namespace coregs::lod;

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

#define CHECK_NEAR(actual, expected, tolerance)                                            \
    do {                                                                                   \
        ++g_checks;                                                                        \
        const double _actual = static_cast<double>(actual);                                \
        const double _expected = static_cast<double>(expected);                            \
        if (!(std::fabs(_actual - _expected) <= (tolerance))) {                            \
            ++g_failures;                                                                  \
            std::fprintf(stderr, "%s:%d: %s = %.6f, expected %.6f\n", __FILE__, __LINE__,  \
                         #actual, _actual, _expected);                                     \
        }                                                                                  \
    } while (0)

// Builds a TEX1 word from its fields (01.5 layout).
std::uint64_t make_tex1(bool lcm, std::uint64_t mxl, std::uint64_t mmin, bool mtba,
                        std::uint64_t l, std::uint64_t k_field12) {
    std::uint64_t word = 0;
    word |= lcm ? 1u : 0u;
    word |= (mxl & 7u) << 2;
    word |= (mmin & 7u) << 6;
    word |= mtba ? (1u << 9) : 0u;
    word |= (l & 3u) << 19;
    word |= (k_field12 & 0xFFFu) << 32;
    return word;
}

std::uint64_t make_miptbp_group(std::uint64_t tbp, std::uint64_t tbw, unsigned index) {
    return ((tbp & 0x3FFFu) | ((tbw & 0x3Fu) << 14)) << (20u * index);
}

void test_tex1_decode() {
    const Tex1Fields fields = decode_tex1(make_tex1(true, 3, 5, true, 2, 0xFF0));
    CHECK(fields.lcm);
    CHECK_EQ(fields.mxl, 3);
    CHECK_EQ(fields.mmin, 5);
    CHECK(fields.mtba);
    CHECK_EQ(fields.l, 2);
    CHECK_EQ(fields.k_raw, -16);  // 0xFF0 sign-extended from 12 bits
    CHECK_EQ(decode_tex1(make_tex1(false, 0, 0, false, 0, 0x800)).k_raw, -2048);
    CHECK_EQ(decode_tex1(make_tex1(false, 0, 0, false, 0, 0x7FF)).k_raw, 2047);
    CHECK(!decode_tex1(make_tex1(false, 0, 1, false, 0, 0)).lcm);
}

void test_lod_formula() {
    CHECK_NEAR(lod_from_q(1.0, 0, 0, false), 0.0, 1e-12);
    CHECK_NEAR(lod_from_q(0.5, 0, 0, false), 1.0, 1e-12);   // log2(1/0.5) = 1
    CHECK_NEAR(lod_from_q(0.5, 1, 0, false), 2.0, 1e-12);   // << L with L = 1
    CHECK_NEAR(lod_from_q(-0.5, 0, 16, false), 2.0, 1e-12); // |Q| used; K/16 = 1
    CHECK_NEAR(lod_from_q(1.0, 0, -8, false), -0.5, 1e-12); // K = -0.5
    CHECK_NEAR(lod_from_q(0.25, 0, 32, true), 2.0, 1e-12);  // LCM = 1 ignores Q
    CHECK(std::isinf(lod_from_q(0.0, 0, 0, false)));
    CHECK_NEAR(explicit_shader_lod(0.25, decode_tex1(make_tex1(false, 0, 0, false, 0, 0)), 1, 0), 1.0, 1e-12);
}

void test_base_level_rules() {
    // Constant path from K: (K + 8) >> 4 (03.6). K raw 24 gives 2, raw -24 gives -1 -> 0.
    CHECK_EQ(select_base_level(decode_tex1(make_tex1(true, 3, 0, false, 0, 24)), false, std::nullopt).base_level, 2);
    CHECK_EQ(select_base_level(decode_tex1(make_tex1(true, 3, 0, false, 0, 0xFE8)), false, std::nullopt).base_level, 0);
    CHECK(select_base_level(decode_tex1(make_tex1(true, 3, 0, false, 0, 0)), false, std::nullopt).constant_lod);

    // Trilinear MMIN 5: base = floor(min LOD), max = ceil(max LOD) clamped to MXL.
    const Tex1Fields trilinear = decode_tex1(make_tex1(false, 4, 5, false, 0, 0));
    const BaseLevelSelection tri = select_base_level(trilinear, false, TracedLod{2.3, 4.0});
    CHECK_EQ(tri.base_level, 2);
    CHECK_EQ(tri.max_level, 4);
    CHECK_NEAR(tri.lod_bias, 0.0, 1e-12);
    CHECK(!tri.constant_lod);

    // Non-trilinear MMIN 2: base = round(min + 0.0625); bias 0.01 when ceil(min) < max.
    const Tex1Fields nearest_mip = decode_tex1(make_tex1(false, 3, 2, false, 0, 0));
    const BaseLevelSelection biased = select_base_level(nearest_mip, false, TracedLod{1.45, 3.0});
    CHECK_EQ(biased.base_level, 2);
    CHECK_EQ(biased.max_level, 3);
    CHECK_NEAR(biased.lod_bias, 0.01, 1e-12);
    const BaseLevelSelection unbiased = select_base_level(nearest_mip, false, TracedLod{1.0, 1.0});
    CHECK_EQ(unbiased.base_level, 1);
    CHECK_NEAR(unbiased.lod_bias, 0.0, 1e-12);

    // Minimum at or above MXL: constant mxl.
    const BaseLevelSelection saturated = select_base_level(decode_tex1(make_tex1(false, 3, 5, false, 0, 0)), false, TracedLod{3.5, 5.0});
    CHECK_EQ(saturated.base_level, 3);
    CHECK_EQ(saturated.max_level, 3);
    CHECK(saturated.constant_lod);

    // FST forces the constant path even with a trace; no trace also uses K.
    const BaseLevelSelection fst = select_base_level(decode_tex1(make_tex1(false, 3, 5, false, 0, 16)), true, TracedLod{0.5, 2.0});
    CHECK(fst.constant_lod);
    CHECK_EQ(fst.base_level, 1);
    CHECK(select_base_level(decode_tex1(make_tex1(false, 3, 5, false, 0, 16)), false, std::nullopt).constant_lod);

    // Max level clamped to MXL and MXL clamped to 6.
    CHECK_EQ(select_base_level(decode_tex1(make_tex1(false, 2, 5, false, 0, 0)), false, TracedLod{0.0, 10.0}).max_level, 2);
    CHECK_EQ(select_base_level(decode_tex1(make_tex1(false, 7, 5, false, 0, 0)), false, TracedLod{0.0, 10.0}).max_level, 6);
}

void test_nearest_rounding() {
    // V-01-11 (Hypothesis, round half up): verified against the implemented rule.
    CHECK_EQ(nearest_mip_level(1.5), 2);
    CHECK_EQ(nearest_mip_level(1.49), 1);
    CHECK_EQ(nearest_mip_level(0.5), 1);
    CHECK_EQ(nearest_mip_level(-0.5), 0);
}

void test_filter_selection() {
    CHECK(!uses_minification_filter(0.0));
    CHECK(uses_minification_filter(0.0001));
    CHECK(!uses_minification_filter(-3.0));
    CHECK(!is_mip_draw(0, 5, 2.0));
    CHECK(!is_mip_draw(3, 1, 2.0));
    CHECK(!is_mip_draw(3, 5, 0.0));
    CHECK(is_mip_draw(3, 5, 0.5));
    CHECK(is_mip_draw(3, 2, 0.5));
}

void test_miptbp_layout() {
    const std::uint64_t miptbp1 = make_miptbp_group(1000, 3, 0) | make_miptbp_group(2000, 5, 1) | make_miptbp_group(16383, 63, 2);
    const MipLevelAddress l1 = decode_miptbp(miptbp1, 0);
    const MipLevelAddress l2 = decode_miptbp(miptbp1, 1);
    const MipLevelAddress l3 = decode_miptbp(miptbp1, 2);
    CHECK_EQ(l1.tbp, 1000);
    CHECK_EQ(l1.tbw, 3);
    CHECK_EQ(l2.tbp, 2000);
    CHECK_EQ(l2.tbw, 5);
    CHECK_EQ(l3.tbp, 16383);
    CHECK_EQ(l3.tbw, 63);

    const std::uint64_t miptbp2 = make_miptbp_group(4000, 7, 0);
    const MipLevelAddress l0 = mip_level_address(0, 77, 4, miptbp1, miptbp2);
    CHECK_EQ(l0.tbp, 77);
    CHECK_EQ(l0.tbw, 4);
    CHECK_EQ(mip_level_address(1, 77, 4, miptbp1, miptbp2).tbp, 1000);
    CHECK_EQ(mip_level_address(3, 77, 4, miptbp1, miptbp2).tbp, 16383);
    CHECK_EQ(mip_level_address(4, 77, 4, miptbp1, miptbp2).tbp, 4000);
    CHECK_EQ(mip_level_address(4, 77, 4, miptbp1, miptbp2).tbw, 7);
}

void test_mip_sizes() {
    const MipSize s0 = mip_level_size(8, 6, 0);
    CHECK_EQ(s0.width, 256);
    CHECK_EQ(s0.height, 64);
    const MipSize s3 = mip_level_size(8, 6, 3);
    CHECK_EQ(s3.width, 32);
    CHECK_EQ(s3.height, 8);
    const MipSize s7 = mip_level_size(8, 6, 7);
    CHECK_EQ(s7.width, 2);
    CHECK_EQ(s7.height, 1);
    const MipSize s9 = mip_level_size(8, 6, 9);
    CHECK_EQ(s9.width, 1);
    CHECK_EQ(s9.height, 1);
    CHECK_EQ(mip_level_size(0, 0, 40).width, 1);
}

void test_mtba_vector() {
    // 01.5 vector: CT32, TW = 8, TBP0 = 0 -> levels 1..3 at TBP 1024/1280/1344, TBW 2/1/1.
    CHECK(mtba_applies(true, 8, 0x00));
    const auto levels = mtba_levels(0, 8, 0x00);
    CHECK_EQ(levels[0].tbp, 1024);
    CHECK_EQ(levels[0].tbw, 2);
    CHECK_EQ(levels[1].tbp, 1280);
    CHECK_EQ(levels[1].tbw, 1);
    CHECK_EQ(levels[2].tbp, 1344);
    CHECK_EQ(levels[2].tbw, 1);
}

void test_mtba_rules() {
    CHECK(!mtba_applies(false, 8, 0x00));   // MTBA bit clear
    CHECK(!mtba_applies(true, 4, 0x00));    // TW below 5
    CHECK(mtba_applies(true, 9, 0x00));     // 32-bit maximum 9
    CHECK(!mtba_applies(true, 10, 0x00));   // CT32 maximum 9
    CHECK(mtba_applies(true, 10, 0x02));    // CT16 maximum 10
    CHECK(!mtba_applies(true, 11, 0x02));
    CHECK(!mtba_applies(true, 8, 0x30));    // Z32 never packs
    CHECK_EQ(psm_bits_per_pixel(0x00), 32);
    CHECK_EQ(psm_bits_per_pixel(0x01), 24);
    CHECK_EQ(psm_bits_per_pixel(0x14), 4);
    CHECK_EQ(psm_bits_per_pixel(0x31), 24);
    CHECK_EQ(psm_bits_per_pixel(0x07), 0);
    // CT16 at TW = 10: blocks0 = 1024*1024*16/2048 = 8192; bw0 = 16.
    const auto ct16 = mtba_levels(0, 10, 0x02);
    CHECK_EQ(ct16[0].tbp, 8192);
    CHECK_EQ(ct16[0].tbw, 8);
    CHECK_EQ(ct16[1].tbp, 8192 + 2048);
    CHECK_EQ(ct16[1].tbw, 4);
}

void test_base_scaling() {
    CHECK_EQ(scale_coordinate_for_base(static_cast<std::uint32_t>(256), 2), 64);
    CHECK_EQ(scale_coordinate_for_base(static_cast<std::uint32_t>(255), 1), 127);
    CHECK_EQ(scale_coordinate_for_base(static_cast<std::uint32_t>(255), 0), 255);
    CHECK_NEAR(scale_coordinate_for_base(10.0, 2), 2.5, 1e-12);
}

struct TestEntry {
    const char* name;
    void (*run)();
};

}  // namespace

int main(int argc, char** argv) {
    const TestEntry tests[] = {
        {"hw_tex1_decode", test_tex1_decode},
        {"hw_lod_formula", test_lod_formula},
        {"hw_mip_base", test_base_level_rules},
        {"hw_mip_nearest", test_nearest_rounding},
        {"hw_mip_filter", test_filter_selection},
        {"hw_miptbp_layout", test_miptbp_layout},
        {"hw_mip_sizes", test_mip_sizes},
        {"hw_mtba", test_mtba_vector},
        {"hw_mtba_rules", test_mtba_rules},
        {"hw_mip_scaling", test_base_scaling},
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
