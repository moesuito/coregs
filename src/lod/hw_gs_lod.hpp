// SPDX-License-Identifier: Apache-2.0
//
// HWR mipmapping and level-of-detail selection: CPU reference (TASK-027, M13).
// Copyright (c) 2026 GT4Recomp contributors.
//
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 01.5
// (TEX1 fields, LOD formula, mip levels), 01.5 MTBA rule, 03.6 (base level, modes,
// mip draws) and 14.6 (V-01-11). Clean room, decision 0069. No other renderer's
// source was consulted.
//
// Everything here is pure arithmetic. Register values are passed as 64-bit words
// with the bit layouts of chapter 01. Where the specification is silent the choice is
// recorded in docs/hw-renderer-spec/IMPLEMENTATION-NOTES.md (IN-xx rows).

#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace coregs::lod {

// MMIN values (01.5). Bit 0 set (3 or 5) selects a trilinear mip filter.
enum class MinFilter : std::uint8_t {
    Nearest = 0,
    Linear = 1,
    NearestMipNearest = 2,
    NearestMipLinear = 3,
    LinearMipNearest = 4,
    LinearMipLinear = 5,
};

// TEX1 register fields (01.5 table: LCM[0], MXL[2..4], MMAG[5], MMIN[6..8], MTBA[9],
// L[19..20], K[32..43] signed with 4 fraction bits).
struct Tex1Fields {
    bool lcm = false;
    std::uint32_t mxl = 0;
    bool mmag_linear = false;
    std::uint32_t mmin = 0;
    bool mtba = false;
    std::uint32_t l = 0;
    std::int32_t k_raw = 0;  // K in 1/16 units, sign extended from 12 bits
};

Tex1Fields decode_tex1(std::uint64_t tex1);

// Mip address of one level: TBP in 256-byte blocks, TBW in 64-texel units.
struct MipLevelAddress {
    std::uint32_t tbp = 0;
    std::uint32_t tbw = 0;
};

// Traced LOD range of a draw (03.6): minimum and maximum LOD over its vertices.
struct TracedLod {
    double min_lod = 0.0;
    double max_lod = 0.0;
};

// Result of the base-level rule of 03.6.
struct BaseLevelSelection {
    int base_level = 0;     // clamped to [0, mxl]
    int max_level = 0;      // clamped to [0, mxl]
    double lod_bias = 0.0;  // 0.01 bias of the trilinear rule, 0 otherwise
    bool constant_lod = false;
};

// Floor division with an arithmetic-shift result, as the specification's ">>" implies.
std::int32_t floor_div16(std::int32_t value);

// Level of detail of one sample (01.5): LCM = 0 gives (log2(1/|Q|) << L) + K/16,
// LCM = 1 gives K/16. |Q| = 0 yields +infinity.
double lod_from_q(double q, std::uint32_t l, std::int32_t k_raw, bool lcm);

// Explicit shader LOD of the "full" mode (03.6): min(K/16 - log2|Q|*2^L, max level) - base
// for LCM = 0, K/16 - base for LCM = 1.
double explicit_shader_lod(double q, const Tex1Fields& tex1, int max_level, int base_level);

// Base level and maximum level for source lookup (03.6). `traced` is absent when the draw
// has no vertex trace; such draws use the constant-LOD path. `fst` forces constant LOD.
BaseLevelSelection select_base_level(const Tex1Fields& tex1, bool fst,
                                     std::optional<TracedLod> traced);

// Filter selection by LOD (01.5): LOD <= 0 uses the magnification filter, LOD > 0 the
// minification filter.
bool uses_minification_filter(double lod);

// A mip draw (03.6): MXL > 0, MMIN in 2..5 and a positive maximum LOD.
bool is_mip_draw(std::uint32_t mxl, std::uint32_t mmin, double max_lod);

// Mip level chosen by the *_MIPMAP_NEAREST filters: round half up (Hypothesis, test
// V-01-11). Clamped to [0, mxl] by the caller.
int nearest_mip_level(double lod);

// MIPTBP1 (levels 1..3) and MIPTBP2 (levels 4..6) field decoding (01.2). Each register
// holds three 20-bit groups: TBP at bits 20*i + 0..13 and TBW at 20*i + 14..19 for
// index i = 0..2 (MIPTBP1 holds TBP1..TBP3 in that order; MIPTBP2 holds TBP4..TBP6).
MipLevelAddress decode_miptbp(std::uint64_t miptbp, unsigned index_in_register);

// Address of mip level n (0..6). Level 0 comes from TEX0 (`tex0_tbp`, `tex0_tbw`).
MipLevelAddress mip_level_address(unsigned level, std::uint32_t tex0_tbp, std::uint32_t tex0_tbw,
                                  std::uint64_t miptbp1, std::uint64_t miptbp2);

// Size of mip level n for a texture of 2^tw x 2^th texels: max(1, size >> n) per axis.
struct MipSize {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};
MipSize mip_level_size(std::uint32_t tw, std::uint32_t th, unsigned level);

// MTBA (01.5): true when the TEX0 write packs levels after level 0. Requires MTBA = 1,
// TW in [5, max] (max 9 for the 32-bit formats CT32/CT24, 10 otherwise) and a non-Z format.
bool mtba_applies(bool mtba, std::uint32_t tw, std::uint32_t psm);

// Transfer bits per pixel of a colour format (01.2 table). Z formats return their
// storage bits; unknown codes return 0.
std::uint32_t psm_bits_per_pixel(std::uint32_t psm);

// Levels 1..3 that the MTBA rule writes into MIPTBP1 for a texture of 2^tw texels per
// side (TH is taken equal to TW, as the specification's formula does). Requires
// mtba_applies().
std::array<MipLevelAddress, 3> mtba_levels(std::uint32_t tbp0, std::uint32_t tw, std::uint32_t psm);

// Base-level coordinate rules (03.6): clamp bounds and coordinates are halved once per
// level of base.
std::uint32_t scale_coordinate_for_base(std::uint32_t value, int base_level);
double scale_coordinate_for_base(double value, int base_level);

}  // namespace coregs::lod
