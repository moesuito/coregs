// SPDX-License-Identifier: Apache-2.0
//
// HWR pixel arithmetic: CPU reference for the GS per-pixel pipeline.
// Copyright (c) 2026 GT4Recomp contributors.
//
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 01.1,
// 01.3, 01.5, 03.5, 03.7, 08.1 to 08.11 (clean room, decision 0069).

#include "hw_gs_pixel.hpp"

#include <algorithm>
#include <cmath>

namespace coregs::pixel {
namespace {

constexpr std::int32_t kChannelMax = 255;

std::int32_t clamp_channel(std::int32_t value) {
    return std::clamp(value, 0, kChannelMax);
}

// Channel product (T * C) >> 7 with 0x80 meaning 1.0. Inputs are 0..255 so the
// product fits in 16 bits and the shift is non-negative.
std::int32_t scaled_product(std::int32_t texel, std::int32_t fragment) {
    return (texel * fragment) >> 7;
}

std::uint32_t low_byte(std::int32_t value) {
    return static_cast<std::uint32_t>(value) & 0xFFu;
}

BlendSource decode_source(std::uint64_t selector) {
    switch (selector & 3u) {
        case 0: return BlendSource::Cs;
        case 1: return BlendSource::Cd;
        default: return BlendSource::Zero;  // selector 2 and 3 are zero
    }
}

BlendFactor decode_factor(std::uint64_t selector) {
    switch (selector & 3u) {
        case 0: return BlendFactor::As;
        case 1: return BlendFactor::Ad;
        default: return BlendFactor::Fix;  // selector 2 and 3 are FIX
    }
}

std::uint64_t encode_source(BlendSource source) {
    switch (source) {
        case BlendSource::Cs: return 0;
        case BlendSource::Cd: return 1;
        case BlendSource::Zero: return 2;
    }
    return 2;
}

std::uint64_t encode_factor(BlendFactor factor) {
    switch (factor) {
        case BlendFactor::As: return 0;
        case BlendFactor::Ad: return 1;
        case BlendFactor::Fix: return 2;
    }
    return 2;
}

std::int32_t pick_source(BlendSource source, std::int32_t cs, std::int32_t cd) {
    switch (source) {
        case BlendSource::Cs: return cs;
        case BlendSource::Cd: return cd;
        case BlendSource::Zero: return 0;
    }
    return 0;
}

WriteEnable intersect(WriteEnable a, WriteEnable b) {
    return WriteEnable{a.colour_rgb && b.colour_rgb, a.colour_alpha && b.colour_alpha, a.depth && b.depth};
}

std::uint32_t depth_mask(ZFormat format) {
    switch (format) {
        case ZFormat::Z32: return 0xFFFFFFFFu;
        case ZFormat::Z24: return 0x00FFFFFFu;
        case ZFormat::Z16: return 0x0000FFFFu;
    }
    return 0xFFFFFFFFu;
}

}  // namespace

std::uint32_t frame_valid_mask(FrameFormat format) {
    switch (format) {
        case FrameFormat::Ct32: return 0xFFFFFFFFu;
        case FrameFormat::Ct24: return 0x00FFFFFFu;
        case FrameFormat::Ct16: return 0x80F8F8F8u;
    }
    return 0xFFFFFFFFu;
}

std::uint32_t z_max(ZFormat format) {
    return depth_mask(format);
}

// ---- Texture function ---------------------------------------------------------

Colour texture_function(Tfx tfx, bool tcc, Colour texel, Colour fragment) {
    Colour out;
    switch (tfx) {
        case Tfx::Modulate:
            out.r = std::min(kChannelMax, scaled_product(texel.r, fragment.r));
            out.g = std::min(kChannelMax, scaled_product(texel.g, fragment.g));
            out.b = std::min(kChannelMax, scaled_product(texel.b, fragment.b));
            out.a = tcc ? std::min(kChannelMax, scaled_product(texel.a, fragment.a)) : fragment.a;
            break;
        case Tfx::Decal:
            out.r = texel.r;
            out.g = texel.g;
            out.b = texel.b;
            out.a = tcc ? texel.a : fragment.a;
            break;
        case Tfx::Highlight:
            out.r = std::min(kChannelMax, scaled_product(texel.r, fragment.r) + fragment.a);
            out.g = std::min(kChannelMax, scaled_product(texel.g, fragment.g) + fragment.a);
            out.b = std::min(kChannelMax, scaled_product(texel.b, fragment.b) + fragment.a);
            out.a = tcc ? std::min(kChannelMax, texel.a + fragment.a) : fragment.a;
            break;
        case Tfx::Highlight2:
            out.r = std::min(kChannelMax, scaled_product(texel.r, fragment.r) + fragment.a);
            out.g = std::min(kChannelMax, scaled_product(texel.g, fragment.g) + fragment.a);
            out.b = std::min(kChannelMax, scaled_product(texel.b, fragment.b) + fragment.a);
            out.a = tcc ? texel.a : fragment.a;
            break;
    }
    return out;
}

// ---- TEXA expansion -----------------------------------------------------------

Colour expand_texel16(std::uint16_t texel, std::uint8_t ta0, std::uint8_t ta1, bool aem) {
    Colour out;
    out.r = (texel & 0x1F) << 3;
    out.g = ((texel >> 5) & 0x1F) << 3;
    out.b = ((texel >> 10) & 0x1F) << 3;
    // IN-07 single rule: the A bit selects TA1 first; otherwise AEM with RGB = 0 gives 0,
    // else TA0. For A bit clear this equals the whole-value form (the CLUT rule of 07.5).
    if ((texel & 0x8000u) != 0) {
        out.a = ta1;
    } else {
        out.a = ta0;
        if (aem && out.r == 0 && out.g == 0 && out.b == 0) {
            out.a = 0;
        }
    }
    return out;
}

Colour expand_texel24(std::uint32_t texel, std::uint8_t ta0, bool aem) {
    Colour out;
    out.r = static_cast<std::int32_t>(texel & 0xFFu);
    out.g = static_cast<std::int32_t>((texel >> 8) & 0xFFu);
    out.b = static_cast<std::int32_t>((texel >> 16) & 0xFFu);
    out.a = ta0;
    if (aem && out.r == 0 && out.g == 0 && out.b == 0) {
        out.a = 0;
    }
    return out;
}

// ---- Fog ----------------------------------------------------------------------

Colour apply_fog(Colour colour, std::int32_t fog, Colour fog_colour) {
    auto mix = [fog](std::int32_t c, std::int32_t fc) { return (fog * c + (255 - fog) * fc) >> 8; };
    Colour out = colour;
    out.r = mix(colour.r, fog_colour.r);
    out.g = mix(colour.g, fog_colour.g);
    out.b = mix(colour.b, fog_colour.b);
    return out;
}

// ---- Alpha test ---------------------------------------------------------------

bool alpha_compare(AlphaCompare compare, std::int32_t alpha, std::int32_t reference) {
    switch (compare) {
        case AlphaCompare::Never: return false;
        case AlphaCompare::Always: return true;
        case AlphaCompare::Less: return alpha < reference;
        case AlphaCompare::LessEqual: return alpha <= reference;
        case AlphaCompare::Equal: return alpha == reference;
        case AlphaCompare::GreaterEqual: return alpha >= reference;
        case AlphaCompare::Greater: return alpha > reference;
        case AlphaCompare::NotEqual: return alpha != reference;
    }
    return false;
}

WriteEnable alpha_outcome_write_enable(const AlphaTestState& test, bool passed, FrameFormat format) {
    if (passed) {
        return kWriteAll;
    }
    AlphaFail fail = test.fail;
    if (fail == AlphaFail::RgbOnly && format != FrameFormat::Ct32) {
        fail = AlphaFail::FrameOnly;  // High: RGB_ONLY on a non-32-bit frame acts as FB_ONLY
    }
    switch (fail) {
        case AlphaFail::Keep: return kWriteNone;
        case AlphaFail::FrameOnly: return WriteEnable{true, true, false};
        case AlphaFail::DepthOnly: return WriteEnable{false, false, true};
        case AlphaFail::RgbOnly: return WriteEnable{true, false, false};
    }
    return kWriteNone;
}

WriteEnable alpha_test_write_enable(const AlphaTestState& test, std::int32_t alpha, FrameFormat format) {
    if (!test.enabled) {
        return kWriteAll;
    }
    return alpha_outcome_write_enable(test, alpha_compare(test.compare, alpha, test.reference), format);
}

AlphaFold fold_alpha_test(AlphaCompare compare, std::int32_t min_alpha, std::int32_t max_alpha, std::int32_t reference) {
    bool all_pass = false;
    bool all_fail = false;
    switch (compare) {
        case AlphaCompare::Never:
            all_fail = true;
            break;
        case AlphaCompare::Always:
            all_pass = true;
            break;
        case AlphaCompare::Less:
            all_pass = max_alpha < reference;
            all_fail = min_alpha >= reference;
            break;
        case AlphaCompare::LessEqual:
            all_pass = max_alpha <= reference;
            all_fail = min_alpha > reference;
            break;
        case AlphaCompare::Equal:
            all_pass = min_alpha == reference && max_alpha == reference;
            all_fail = min_alpha > reference || max_alpha < reference;
            break;
        case AlphaCompare::GreaterEqual:
            all_pass = min_alpha >= reference;
            all_fail = max_alpha < reference;
            break;
        case AlphaCompare::Greater:
            all_pass = min_alpha > reference;
            all_fail = max_alpha <= reference;
            break;
        case AlphaCompare::NotEqual:
            all_pass = min_alpha > reference || max_alpha < reference;
            all_fail = min_alpha == reference && max_alpha == reference;
            break;
    }
    if (all_pass) {
        return AlphaFold::AllPass;
    }
    if (all_fail) {
        return AlphaFold::AllFail;
    }
    return AlphaFold::Mixed;
}

// ---- Destination alpha --------------------------------------------------------

std::int32_t destination_alpha(FrameFormat format, std::uint32_t frame_word) {
    if (format == FrameFormat::Ct24) {
        return 0x80;
    }
    return static_cast<std::int32_t>((frame_word >> 24) & 0xFFu);
}

WriteEnable destination_alpha_write_enable(const DestAlphaState& date, FrameFormat format, std::uint32_t frame_word) {
    if (!date.enabled) {
        return kWriteAll;
    }
    if (format == FrameFormat::Ct24) {
        return kWriteNone;  // High: no alpha to test, nothing is written
    }
    const bool msb = ((frame_word >> 31) & 1u) != 0;
    return msb == date.expected_msb ? kWriteAll : kWriteNone;
}

// ---- Depth --------------------------------------------------------------------

std::uint32_t clamp_z(std::uint64_t z, ZFormat format) {
    const std::uint64_t max = z_max(format);
    return static_cast<std::uint32_t>(z > max ? max : z);
}

std::uint32_t interpolated_z(double z, ZFormat format) {
    if (!(z > 0.0)) {
        return 0;  // negative and NaN values
    }
    if (z >= 4294967296.0) {
        return z_max(format);
    }
    return clamp_z(static_cast<std::uint64_t>(z), format);  // truncation toward zero
}

std::uint32_t depth_bits(std::uint32_t stored_raw, ZFormat format) {
    return stored_raw & depth_mask(format);
}

bool depth_passes(DepthCompare compare, std::uint32_t z_new, std::uint32_t z_stored_raw, ZFormat format) {
    const std::uint32_t stored = depth_bits(z_stored_raw, format);
    const std::uint32_t incoming = z_new & depth_mask(format);
    switch (compare) {
        case DepthCompare::Never: return false;
        case DepthCompare::Always: return true;
        case DepthCompare::GreaterEqual: return incoming >= stored;
        case DepthCompare::Greater: return incoming > stored;
    }
    return false;
}

std::uint32_t write_depth_raw(std::uint32_t stored_raw, std::uint32_t z, ZFormat format) {
    const std::uint32_t mask = depth_mask(format);
    return (stored_raw & ~mask) | (z & mask);
}

// ---- Blending -----------------------------------------------------------------

void decode_alpha_register(std::uint64_t alpha_register, BlendState& state) {
    state.a = decode_source(alpha_register >> 0);
    state.b = decode_source(alpha_register >> 2);
    state.c = decode_factor(alpha_register >> 4);
    state.d = decode_source(alpha_register >> 6);
    state.fix = static_cast<std::uint8_t>((alpha_register >> 32) & 0xFFu);
}

std::uint64_t encode_alpha_register(BlendSource a, BlendSource b, BlendFactor c, BlendSource d, std::uint8_t fix) {
    return (encode_source(a) << 0) | (encode_source(b) << 2) | (encode_factor(c) << 4) | (encode_source(d) << 6) |
           (static_cast<std::uint64_t>(fix) << 32);
}

Colour blend_unclamped(const BlendState& state, Colour source, Colour destination, std::int32_t as, std::int32_t ad) {
    std::int32_t factor = state.fix;
    switch (state.c) {
        case BlendFactor::As: factor = as; break;
        case BlendFactor::Ad: factor = ad; break;
        case BlendFactor::Fix: factor = state.fix; break;
    }
    auto channel = [&](std::int32_t cs, std::int32_t cd) {
        const std::int32_t a = pick_source(state.a, cs, cd);
        const std::int32_t b = pick_source(state.b, cs, cd);
        const std::int32_t d = pick_source(state.d, cs, cd);
        return (((a - b) * factor) >> 7) + d;  // arithmetic shift of a signed product
    };
    Colour out;
    out.r = channel(source.r, destination.r);
    out.g = channel(source.g, destination.g);
    out.b = channel(source.b, destination.b);
    out.a = as;
    return out;
}

std::int32_t colour_clamp_or_wrap(std::int32_t value, bool clamp) {
    return clamp ? clamp_channel(value) : (value & 0xFF);
}

// ---- Dither -------------------------------------------------------------------

std::int32_t dither_entry(std::uint64_t dimx_register, std::int32_t x, std::int32_t y) {
    const std::uint32_t row = static_cast<std::uint32_t>(y) & 3u;
    const std::uint32_t column = static_cast<std::uint32_t>(x) & 3u;
    const std::uint32_t index = 4u * row + column;
    const std::uint64_t field = (dimx_register >> (4u * index)) & 7u;
    // Sign-extend the 3-bit field: 4..7 are -4..-1.
    return field >= 4 ? static_cast<std::int32_t>(field) - 8 : static_cast<std::int32_t>(field);
}

Colour apply_dither(Colour colour, std::int32_t dither) {
    Colour out = colour;
    out.r += dither;
    out.g += dither;
    out.b += dither;
    return out;
}

// ---- Frame conversion and FBMSK ----------------------------------------------

std::uint32_t unpack_frame_word(FrameFormat format, std::uint32_t stored) {
    switch (format) {
        case FrameFormat::Ct32:
            return stored;
        case FrameFormat::Ct24:
            return stored & 0x00FFFFFFu;
        case FrameFormat::Ct16: {
            const std::uint32_t r = stored & 0x1Fu;
            const std::uint32_t g = (stored >> 5) & 0x1Fu;
            const std::uint32_t b = (stored >> 10) & 0x1Fu;
            const std::uint32_t a = (stored >> 15) & 1u;
            return (r << 3) | (g << 11) | (b << 19) | (a << 31);
        }
    }
    return stored;
}

std::uint32_t pack_frame_word(FrameFormat format, std::uint32_t frame_word) {
    switch (format) {
        case FrameFormat::Ct32:
            return frame_word;
        case FrameFormat::Ct24:
            return frame_word & 0x00FFFFFFu;
        case FrameFormat::Ct16: {
            const std::uint32_t r = (frame_word >> 3) & 0x1Fu;
            const std::uint32_t g = (frame_word >> 11) & 0x1Fu;
            const std::uint32_t b = (frame_word >> 19) & 0x1Fu;
            const std::uint32_t a = (frame_word >> 31) & 1u;
            return r | (g << 5) | (b << 10) | (a << 15);
        }
    }
    return frame_word;
}

std::uint32_t colour_to_frame_word(FrameFormat format, Colour colour) {
    const std::uint32_t r = low_byte(colour.r);
    const std::uint32_t g = low_byte(colour.g);
    const std::uint32_t b = low_byte(colour.b);
    const std::uint32_t a = low_byte(colour.a);
    switch (format) {
        case FrameFormat::Ct32:
            return r | (g << 8) | (b << 16) | (a << 24);
        case FrameFormat::Ct24:
            return r | (g << 8) | (b << 16);
        case FrameFormat::Ct16:
            // Keep the top five bits of each colour channel; the alpha lane keeps bit 7.
            return (r & 0xF8u) | ((g & 0xF8u) << 8) | ((b & 0xF8u) << 16) | ((a & 0x80u) << 24);
    }
    return 0;
}

Colour frame_word_to_colour(std::uint32_t frame_word) {
    Colour out;
    out.r = static_cast<std::int32_t>(frame_word & 0xFFu);
    out.g = static_cast<std::int32_t>((frame_word >> 8) & 0xFFu);
    out.b = static_cast<std::int32_t>((frame_word >> 16) & 0xFFu);
    out.a = static_cast<std::int32_t>((frame_word >> 24) & 0xFFu);
    return out;
}

Colour apply_fba(Colour colour, bool fba) {
    Colour out = colour;
    if (fba) {
        out.a |= 0x80;
    }
    return out;
}

std::uint32_t merge_frame_word(FrameFormat format, std::uint32_t old_word, std::uint32_t computed_word,
                               std::uint32_t fbmsk, WriteEnable enable) {
    const std::uint32_t valid = frame_valid_mask(format);
    std::uint32_t enabled_lanes = 0;
    if (enable.colour_rgb) {
        enabled_lanes |= 0x00FFFFFFu;
    }
    if (enable.colour_alpha) {
        enabled_lanes |= 0xFF000000u;
    }
    // Bits that are written: enabled, valid and not protected by FBMSK (1 = keep).
    const std::uint32_t active = enabled_lanes & valid & ~(fbmsk & valid);
    return (old_word & ~active) | (computed_word & active);
}

// ---- Texture addressing -------------------------------------------------------

std::int32_t wrap_texel(std::int32_t texel, std::int32_t log2_size, const WrapAxis& axis) {
    const std::int32_t last = static_cast<std::int32_t>((1u << log2_size) - 1u);
    switch (axis.mode) {
        case WrapMode::Repeat:
            return texel & last;
        case WrapMode::Clamp:
            return std::clamp(texel, 0, last);
        case WrapMode::RegionClamp:
            return std::clamp(texel, axis.minimum, axis.maximum);
        case WrapMode::RegionRepeat:
            return (texel & axis.minimum) | axis.maximum;
    }
    return texel;
}

std::int32_t nearest_texel(std::int32_t uv_10_4) {
    return uv_10_4 >> 4;  // floor for negative values too
}

BilinearSample bilinear_base(std::int32_t uv_10_4) {
    const std::int32_t shifted = uv_10_4 - 8;  // u - 0.5 in 10.4 fixed point
    BilinearSample sample;
    sample.base = shifted >> 4;
    sample.fraction = shifted & 15;
    return sample;
}

bool texture_axis_usable(std::int32_t log2_size, WrapMode mode) {
    if (log2_size <= 10) {
        return true;
    }
    return mode == WrapMode::RegionClamp || mode == WrapMode::RegionRepeat;
}

std::int32_t mip_level_size(std::int32_t log2_size, std::int32_t level) {
    const std::int32_t shift = log2_size - level;
    return shift <= 0 ? 1 : (1 << shift);
}

// ---- Whole-pixel pipeline -----------------------------------------------------

PixelOutcome process_pixel(const PixelState& state, const Fragment& fragment, std::uint32_t frame_stored_old,
                           std::uint32_t depth_stored_old) {
    PixelOutcome outcome;
    outcome.frame_stored = frame_stored_old;
    outcome.depth_stored = depth_stored_old;

    const FrameFormat format = state.frame.format;
    const std::uint32_t old_word = unpack_frame_word(format, frame_stored_old);

    // Alpha test (uses As before FBA), destination alpha test.
    WriteEnable enable = alpha_test_write_enable(state.alpha_test, fragment.colour.a, format);
    enable = intersect(enable, destination_alpha_write_enable(state.date, format, old_word));

    // Depth test: a failing fragment is discarded (colour and Z).
    const std::uint32_t z = clamp_z(fragment.z, state.z_format);
    if (state.depth.test_enabled) {
        if (!depth_passes(state.depth.compare, z, depth_stored_old, state.z_format)) {
            return outcome;
        }
    }
    // Z is written only when the depth test is enabled and ZMSK is 0.
    enable.depth = enable.depth && state.depth.test_enabled && !state.depth.write_masked;

    if (!enable.any()) {
        return outcome;
    }

    // Blend. Cd and Ad come from the stored destination pixel.
    const Colour source = fragment.colour;
    Colour result = source;
    if (state.blend.enabled && !(state.blend.pabe && (source.a & 0x80) == 0)) {
        const Colour destination = frame_word_to_colour(old_word);
        result = blend_unclamped(state.blend, source, destination, source.a, destination_alpha(format, old_word));
    }

    // Dither (16-bit frames only), then colour clamp or wrap.
    if (format == FrameFormat::Ct16 && state.frame.dither) {
        result = apply_dither(result, dither_entry(state.frame.dimx, fragment.x, fragment.y));
    }
    result.r = colour_clamp_or_wrap(result.r, state.blend.colour_clamp);
    result.g = colour_clamp_or_wrap(result.g, state.blend.colour_clamp);
    result.b = colour_clamp_or_wrap(result.b, state.blend.colour_clamp);

    // FBA, format conversion and FBMSK merge.
    result = apply_fba(result, state.frame.fba);
    const std::uint32_t computed = colour_to_frame_word(format, result);
    const std::uint32_t merged = merge_frame_word(format, old_word, computed, state.frame.fbmsk, enable);
    if (enable.colour_rgb || enable.colour_alpha) {
        outcome.frame_written = true;
        outcome.frame_stored = pack_frame_word(format, merged);
    }

    if (enable.depth) {
        outcome.depth_written = true;
        outcome.depth_stored = write_depth_raw(depth_stored_old, z, state.z_format);
    }
    return outcome;
}

}  // namespace coregs::pixel
