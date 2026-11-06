// SPDX-License-Identifier: MIT
//
// HWR mipmapping and level-of-detail selection: CPU reference (TASK-027, M13).
// Copyright (c) 2026 GT4Recomp contributors.
//
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 01.5, 03.6
// and 14.6 (clean room, decision 0069). Interpretations where the specification is
// silent or ambiguous are recorded as IN-xx rows in docs/hw-renderer-spec/
// IMPLEMENTATION-NOTES.md and cited below.

#include "hw_gs_lod.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace coregs::lod {

namespace {

constexpr std::uint64_t field_mask(unsigned width) {
    return (std::uint64_t{1} << width) - 1u;
}

constexpr std::uint64_t bits_at(std::uint64_t reg, unsigned low, unsigned width) {
    return (reg >> low) & field_mask(width);
}

// Clamp a floating value to a range that fits in int before the conversion.
int to_int_clamped(double value) {
    constexpr double kLimit = 1.0e6;
    const double clamped = std::clamp(value, -kLimit, kLimit);
    return static_cast<int>(clamped);
}

// Round half up (V-01-11 Hypothesis): floor(x + 0.5).
int round_half_up(double value) {
    return to_int_clamped(std::floor(value + 0.5));
}

bool is_z_format(std::uint32_t psm) {
    return ((psm >> 4) & 3u) == 3u;
}

// IN-46: the 32-bit formats are CT32 and CT24.
bool is_32_bit_format(std::uint32_t psm) {
    return psm == 0x00u || psm == 0x01u;
}

}  // namespace

Tex1Fields decode_tex1(std::uint64_t tex1) {
    Tex1Fields fields;
    fields.lcm = bits_at(tex1, 0, 1) != 0;
    fields.mxl = static_cast<std::uint32_t>(bits_at(tex1, 2, 3));
    fields.mmag_linear = bits_at(tex1, 5, 1) != 0;
    fields.mmin = static_cast<std::uint32_t>(bits_at(tex1, 6, 3));
    fields.mtba = bits_at(tex1, 9, 1) != 0;
    fields.l = static_cast<std::uint32_t>(bits_at(tex1, 19, 2));
    // K is a signed 12-bit field: sign-extend from bit 11 of the raw field.
    const std::int32_t raw_k = static_cast<std::int32_t>(bits_at(tex1, 32, 12));
    fields.k_raw = (raw_k >= 0x800) ? raw_k - 0x1000 : raw_k;
    return fields;
}

std::int32_t floor_div16(std::int32_t value) {
    // Arithmetic right shift by four (defined for negative values since C++20).
    return value >> 4;
}

double lod_from_q(double q, std::uint32_t l, std::int32_t k_raw, bool lcm) {
    const double k_value = static_cast<double>(k_raw) / 16.0;
    if (lcm) {
        return k_value;
    }
    const double magnitude = std::fabs(q);
    if (magnitude == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double scaled_log = std::log2(1.0 / magnitude) * std::ldexp(1.0, static_cast<int>(l));
    return scaled_log + k_value;
}

double explicit_shader_lod(double q, const Tex1Fields& tex1, int max_level, int base_level) {
    if (tex1.lcm) {
        return static_cast<double>(tex1.k_raw) / 16.0 - base_level;
    }
    const double lod = lod_from_q(q, tex1.l, tex1.k_raw, false);
    return std::min(lod, static_cast<double>(max_level)) - base_level;
}

BaseLevelSelection select_base_level(const Tex1Fields& tex1, bool fst,
                                     std::optional<TracedLod> traced) {
    const int mxl = static_cast<int>(std::min<std::uint32_t>(tex1.mxl, 6u));
    // k = (K + 8) >> 4 (03.6).
    int k = floor_div16(tex1.k_raw + 8);

    BaseLevelSelection result;
    result.max_level = mxl;

    // Order of the 03.6 rules (IN-44): FST and LCM = 1 always use the constant
    // path with K; LCM = 0 with a traced minimum at or above mxl uses constant mxl
    // (IN-44); LCM = 0 without a trace uses constant K (IN-44).
    if (fst || tex1.lcm) {
        result.constant_lod = true;
    } else if (!traced.has_value()) {
        result.constant_lod = true;
    } else if (traced->min_lod >= static_cast<double>(mxl)) {
        result.constant_lod = true;
        k = mxl;
    }

    if (result.constant_lod) {
        result.base_level = std::clamp(std::max(k, 0), 0, mxl);
        result.lod_bias = 0.0;
        return result;
    }

    const double min_lod = traced->min_lod;
    const double max_lod = traced->max_lod;
    // IN-44: the trilinear base rule applies to MMIN 3 and 5 only.
    const bool trilinear = tex1.mmin == 3u || tex1.mmin == 5u;
    if (trilinear) {
        result.base_level = to_int_clamped(std::floor(min_lod));
        result.lod_bias = 0.0;
    } else {
        result.base_level = round_half_up(min_lod + 0.0625);
        result.lod_bias = std::ceil(min_lod) < max_lod ? 0.01 : 0.0;
    }
    result.max_level = to_int_clamped(std::ceil(max_lod));
    result.base_level = std::clamp(result.base_level, 0, mxl);
    result.max_level = std::clamp(result.max_level, 0, mxl);
    return result;
}

bool uses_minification_filter(double lod) {
    return lod > 0.0;
}

bool is_mip_draw(std::uint32_t mxl, std::uint32_t mmin, double max_lod) {
    return mxl > 0u && mmin >= 2u && mmin <= 5u && max_lod > 0.0;
}

int nearest_mip_level(double lod) {
    return round_half_up(lod);
}

MipLevelAddress decode_miptbp(std::uint64_t miptbp, unsigned index_in_register) {
    const unsigned shift = 20u * index_in_register;
    MipLevelAddress address;
    address.tbp = static_cast<std::uint32_t>(bits_at(miptbp >> shift, 0, 14));
    address.tbw = static_cast<std::uint32_t>(bits_at(miptbp >> shift, 14, 6));
    return address;
}

MipLevelAddress mip_level_address(unsigned level, std::uint32_t tex0_tbp, std::uint32_t tex0_tbw,
                                  std::uint64_t miptbp1, std::uint64_t miptbp2) {
    if (level == 0u) {
        return MipLevelAddress{tex0_tbp, tex0_tbw};
    }
    if (level <= 3u) {
        return decode_miptbp(miptbp1, level - 1u);
    }
    return decode_miptbp(miptbp2, level - 4u);
}

MipSize mip_level_size(std::uint32_t tw, std::uint32_t th, unsigned level) {
    const unsigned shift = std::min(level, 63u);
    const std::uint64_t width = (std::uint64_t{1} << std::min<std::uint32_t>(tw, 31u)) >> std::min(shift, 63u);
    const std::uint64_t height = (std::uint64_t{1} << std::min<std::uint32_t>(th, 31u)) >> std::min(shift, 63u);
    MipSize size;
    size.width = static_cast<std::uint32_t>(std::max<std::uint64_t>(width, 1u));
    size.height = static_cast<std::uint32_t>(std::max<std::uint64_t>(height, 1u));
    return size;
}

std::uint32_t psm_bits_per_pixel(std::uint32_t psm) {
    switch (psm) {
        case 0x00u: return 32u;  // CT32
        case 0x01u: return 24u;  // CT24
        case 0x02u: return 16u;  // CT16
        case 0x0Au: return 16u;  // CT16S
        case 0x13u: return 8u;   // T8
        case 0x14u: return 4u;   // T4
        case 0x1Bu: return 8u;   // T8H
        case 0x24u: return 4u;   // T4HL
        case 0x2Cu: return 4u;   // T4HH
        case 0x30u: return 32u;  // Z32
        case 0x31u: return 24u;  // Z24
        case 0x32u: return 16u;  // Z16
        case 0x3Au: return 16u;  // Z16S
        default: return 0u;
    }
}

bool mtba_applies(bool mtba, std::uint32_t tw, std::uint32_t psm) {
    if (!mtba || is_z_format(psm)) {
        return false;
    }
    const std::uint32_t maximum_tw = is_32_bit_format(psm) ? 9u : 10u;
    return tw >= 5u && tw <= maximum_tw;
}

std::array<MipLevelAddress, 3> mtba_levels(std::uint32_t tbp0, std::uint32_t tw, std::uint32_t psm) {
    // blocks0 = 2^TW * 2^TW * bpp / 2048 and bw0 = max(1, 2^TW / 64) (01.5).
    const std::uint64_t texels_per_side = std::uint64_t{1} << std::min<std::uint32_t>(tw, 31u);
    const std::uint64_t blocks0 = texels_per_side * texels_per_side * psm_bits_per_pixel(psm) / 2048u;
    const std::uint64_t bw0 = std::max<std::uint64_t>(1u, texels_per_side / 64u);

    // IN-46: level n starts after the block count of level n - 1, where the block
    // count of level k is blocks0 / 4^k (minimum 1). This reproduces the vectors of 01.5
    // (1024, 1280, 1344) and overrides the prose "previous level's block count / 4".
    std::array<MipLevelAddress, 3> levels{};
    std::uint64_t tbp = tbp0;
    std::uint64_t blocks_of_previous = blocks0;
    for (unsigned n = 1; n <= 3; ++n) {
        tbp += blocks_of_previous;
        levels[n - 1].tbp = static_cast<std::uint32_t>(tbp);
        levels[n - 1].tbw = static_cast<std::uint32_t>(std::max<std::uint64_t>(1u, bw0 >> n));
        blocks_of_previous = std::max<std::uint64_t>(1u, blocks_of_previous / 4u);
    }
    return levels;
}

std::uint32_t scale_coordinate_for_base(std::uint32_t value, int base_level) {
    return value >> std::clamp(base_level, 0, 31);
}

double scale_coordinate_for_base(double value, int base_level) {
    return value / std::ldexp(1.0, std::clamp(base_level, 0, 31));
}

}  // namespace coregs::lod
