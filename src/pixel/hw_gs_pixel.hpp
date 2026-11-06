// SPDX-License-Identifier: MIT
//
// HWR pixel arithmetic: CPU reference for the GS per-pixel pipeline.
// Copyright (c) 2026 GT4Recomp contributors.
//
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 01.1,
// 01.3, 01.5, 03.5, 03.7, 08.1 to 08.11 (clean room, decision 0069). No other
// renderer's source was consulted.
//
// Every function here is integer arithmetic with defined behaviour. Channel values
// are 0..255 unless a comment says otherwise; intermediates are std::int32_t.
// Right shifts of negative values are arithmetic (defined since C++20).
//
// Frame words: a frame pixel is held as a 32-bit "frame word" whose byte lanes are
// R, G, B, A. For CT16 the 5-bit channels sit in the top five bits of the R, G and B
// lanes and the alpha bit is bit 7 of the A lane, so the valid mask is 0x80F8F8F8,
// exactly the 32-bit view used by the specification. CT24 has no alpha: its alpha lane
// reads as zero and is never written.

#pragma once

#include <cstdint>

namespace coregs::pixel {

struct Colour {
    std::int32_t r = 0;
    std::int32_t g = 0;
    std::int32_t b = 0;
    std::int32_t a = 0;

    friend bool operator==(const Colour&, const Colour&) = default;
};

// ---- Frame and depth formats -------------------------------------------------

enum class FrameFormat : std::uint8_t { Ct32, Ct24, Ct16 };

enum class ZFormat : std::uint8_t { Z32, Z24, Z16 };  // Z16S uses Z16 (same behaviour)

// Valid bits of a frame word for the format (specification 01.3 PSM table).
std::uint32_t frame_valid_mask(FrameFormat format);

// Largest Z value representable in the format.
std::uint32_t z_max(ZFormat format);

// ---- Register field decoding -------------------------------------------------

// Sources for the blend A, B and D selectors: 0 Cs, 1 Cd, 2 and 3 zero.
enum class BlendSource : std::uint8_t { Cs, Cd, Zero };
// Factor selector C: 0 As, 1 Ad, 2 and 3 FIX.
enum class BlendFactor : std::uint8_t { As, Ad, Fix };

// ---- Texture function (01.5, 03.4) ------------------------------------------

enum class Tfx : std::uint8_t { Modulate, Decal, Highlight, Highlight2 };

// texel: T; fragment: C (colour and alpha from the vertex). tcc is TEX0.TCC.
Colour texture_function(Tfx tfx, bool tcc, Colour texel, Colour fragment);

// TEXA expansion of a 16-bit texel (01.5). ta0 and ta1 are TEXA.TA0/TA1, aem is TEXA.AEM.
Colour expand_texel16(std::uint16_t texel, std::uint8_t ta0, std::uint8_t ta1, bool aem);
// TEXA expansion of a 24-bit texel stored in the low 24 bits.
Colour expand_texel24(std::uint32_t texel, std::uint8_t ta0, bool aem);

// ---- Fog (01.5) --------------------------------------------------------------

// RGB only: (F * C + (255 - F) * FOGCOL) >> 8 per channel; alpha is unchanged.
Colour apply_fog(Colour colour, std::int32_t fog, Colour fog_colour);

// ---- Alpha test (01.5, 08.6) -------------------------------------------------

enum class AlphaCompare : std::uint8_t { Never, Always, Less, LessEqual, Equal, GreaterEqual, Greater, NotEqual };
enum class AlphaFail : std::uint8_t { Keep, FrameOnly, DepthOnly, RgbOnly };

struct AlphaTestState {
    bool enabled = false;  // TEST.ATE
    AlphaCompare compare = AlphaCompare::Always;
    AlphaFail fail = AlphaFail::Keep;
    std::uint8_t reference = 0;  // AREF
};

// Which parts of a fragment may write. Colour RGB and colour alpha are separate so
// that RGB_ONLY can be expressed.
struct WriteEnable {
    bool colour_rgb = false;
    bool colour_alpha = false;
    bool depth = false;

    bool any() const { return colour_rgb || colour_alpha || depth; }
    friend bool operator==(const WriteEnable&, const WriteEnable&) = default;
};

constexpr WriteEnable kWriteAll{true, true, true};
constexpr WriteEnable kWriteNone{false, false, false};

// Integer comparison of the fragment alpha (As, before FBA) against AREF.
bool alpha_compare(AlphaCompare compare, std::int32_t alpha, std::int32_t reference);

// Write enables for a fragment's alpha test result. A passing fragment writes
// everything. A failing one writes what AFAIL keeps: FrameOnly writes colour
// including alpha, DepthOnly writes depth, RgbOnly writes RGB only. RgbOnly on a
// frame that is not 32-bit behaves as FrameOnly (High, 01.5).
WriteEnable alpha_test_write_enable(const AlphaTestState& test, std::int32_t alpha, FrameFormat format);

// Same outcome given an already decided pass/fail (used by folding and tests).
WriteEnable alpha_outcome_write_enable(const AlphaTestState& test, bool passed, FrameFormat format);

// Static folding of the test over an inclusive fragment alpha range (08.6).
enum class AlphaFold : std::uint8_t { AllPass, AllFail, Mixed };
AlphaFold fold_alpha_test(AlphaCompare compare, std::int32_t min_alpha, std::int32_t max_alpha, std::int32_t reference);

// ---- Destination alpha test (01.5, 08.7) ------------------------------------

struct DestAlphaState {
    bool enabled = false;       // TEST.DATE
    bool expected_msb = false;  // TEST.DATM
};

// Ad for the blend factor: 0x80 for CT24, otherwise the stored alpha lane.
std::int32_t destination_alpha(FrameFormat format, std::uint32_t frame_word);

// DATE outcome for a destination word. CT24 with DATE set writes nothing (colour or Z).
WriteEnable destination_alpha_write_enable(const DestAlphaState& date, FrameFormat format, std::uint32_t frame_word);

// ---- Depth (01.5, 03.7) -----------------------------------------------------

enum class DepthCompare : std::uint8_t { Never, Always, GreaterEqual, Greater };

struct DepthState {
    bool test_enabled = false;  // ZTE
    DepthCompare compare = DepthCompare::Always;
    bool write_masked = false;  // ZMSK = 1 disables the write
};

// Clamp an integer Z to the format maximum (clamp, not wrap).
std::uint32_t clamp_z(std::uint64_t z, ZFormat format);
// Interpolated Z: truncated toward zero, negative values to 0, then clamped.
std::uint32_t interpolated_z(double z, ZFormat format);
// Unsigned comparison of the format's bits only.
bool depth_passes(DepthCompare compare, std::uint32_t z_new, std::uint32_t z_stored_raw, ZFormat format);
// Writes the format's bits and keeps the rest (Z24 keeps the upper 8 bits).
std::uint32_t write_depth_raw(std::uint32_t stored_raw, std::uint32_t z, ZFormat format);
// Z bits of a stored value (bits above the format width are discarded).
std::uint32_t depth_bits(std::uint32_t stored_raw, ZFormat format);

// ---- Blending (01.5, 08.1, 08.4, 08.5) --------------------------------------

struct BlendState {
    bool enabled = false;      // PRIM.ABE
    bool pabe = false;         // PABE
    BlendSource a = BlendSource::Cs;
    BlendSource b = BlendSource::Cs;
    BlendSource d = BlendSource::Cs;
    BlendFactor c = BlendFactor::As;
    std::uint8_t fix = 0;      // ALPHA.FIX
    bool colour_clamp = true;  // COLCLAMP: true clamps, false wraps (low 8 bits)
};

// Decodes the ALPHA register (0x42): A[0..1] B[2..3] C[4..5] D[6..7] FIX[32..39].
void decode_alpha_register(std::uint64_t alpha_register, BlendState& state);
// Inverse of decode_alpha_register for the four selectors and FIX (other bits zero).
std::uint64_t encode_alpha_register(BlendSource a, BlendSource b, BlendFactor c, BlendSource d, std::uint8_t fix);

// Raw Cv = (((A - B) * C) >> 7) + D per RGB channel, before any clamp or wrap.
// factor is C (0..255). Alpha is not blended; the caller supplies it.
Colour blend_unclamped(const BlendState& state, Colour source, Colour destination, std::int32_t as, std::int32_t ad);

// Colour clamp or wrap (COLCLAMP): clamp to 0..255, or keep the low 8 bits.
std::int32_t colour_clamp_or_wrap(std::int32_t value, bool clamp);

// ---- Dither (01.5, 08.10) ---------------------------------------------------

// DIMX register: sixteen signed 3-bit entries, entry (row i, column j) at bit 4*(4i+j).
std::int32_t dither_entry(std::uint64_t dimx_register, std::int32_t x, std::int32_t y);
// Adds the dither value to RGB. Alpha is unchanged.
Colour apply_dither(Colour colour, std::int32_t dither);

// ---- Frame conversion and FBMSK (01.5, 01.3, 08.9) --------------------------

// Frame word of a stored pixel. CT16 expands the 5-bit channels to bits 3..7, 11..15,
// 19..23 and bit 31. CT24 keeps bits 0..23. CT32 is the identity.
std::uint32_t unpack_frame_word(FrameFormat format, std::uint32_t stored);
// Stored pixel of a frame word. Drops invalid bits: CT16 truncates to 5 bits per channel.
std::uint32_t pack_frame_word(FrameFormat format, std::uint32_t frame_word);

// Frame word of a colour after clamp/wrap: CT16 keeps the top five bits of RGB and
// alpha bit 7; CT24 writes no alpha lane.
std::uint32_t colour_to_frame_word(FrameFormat format, Colour colour);
// Colour of a frame word (bytes as channels). CT16 reads 5-bit channels back as <<3.
Colour frame_word_to_colour(std::uint32_t frame_word);

// FBA: sets bit 7 of the written alpha (sets the 16-bit alpha bit).
Colour apply_fba(Colour colour, bool fba);

// Write rule: new = (old & FBMSK) | (computed & ~FBMSK), restricted to the write enables
// and the format's valid bits. Result is the merged frame word.
std::uint32_t merge_frame_word(FrameFormat format, std::uint32_t old_word, std::uint32_t computed_word,
                               std::uint32_t fbmsk, WriteEnable enable);

// ---- Texture addressing (01.1, 03.5) ----------------------------------------

enum class WrapMode : std::uint8_t { Repeat, Clamp, RegionClamp, RegionRepeat };

struct WrapAxis {
    WrapMode mode = WrapMode::Repeat;
    std::int32_t minimum = 0;  // MINU or MINV (region clamp minimum, region repeat mask)
    std::int32_t maximum = 0;  // MAXU or MAXV (region clamp maximum, region repeat fixed bits)
};

// Integer texel coordinate after wrapping. log2_size is TW or TH (the texture spans
// 1 << log2_size texels).
std::int32_t wrap_texel(std::int32_t texel, std::int32_t log2_size, const WrapAxis& axis);

// Integer texel of a 10.4 fixed-point coordinate (floor for nearest sampling).
std::int32_t nearest_texel(std::int32_t uv_10_4);
// Bilinear base texel and 4-bit fraction: base = floor(u - 0.5), fraction = (u - 0.5) mod 1.
struct BilinearSample {
    std::int32_t base = 0;
    std::int32_t fraction = 0;  // 0..15
};
BilinearSample bilinear_base(std::int32_t uv_10_4);

// A texture axis of more than 1024 texels is invalid unless a region mode is used.
bool texture_axis_usable(std::int32_t log2_size, WrapMode mode);

// Size of mip level n for a base size of 1 << log2_size (never below 1).
std::int32_t mip_level_size(std::int32_t log2_size, std::int32_t level);

// ---- Whole-pixel pipeline (01.5 order, 08.6 to 08.10) -----------------------

struct FrameState {
    FrameFormat format = FrameFormat::Ct32;
    std::uint32_t fbmsk = 0;  // FRAME.FBMSK (1 = bit protected)
    bool fba = false;                   // FRAME.FBA
    bool dither = false;                // DTHE
    std::uint64_t dimx = 0;             // DIMX register
};

struct PixelState {
    FrameState frame;
    AlphaTestState alpha_test;
    DestAlphaState date;
    DepthState depth;
    ZFormat z_format = ZFormat::Z32;
    BlendState blend;
};

// Fragment after texture function, fog and AA1 (AA1 coverage is not emulated, so alpha
// is As). z is the interpolated integer Z (already truncated).
struct Fragment {
    std::int32_t x = 0;
    std::int32_t y = 0;
    Colour colour;
    std::uint32_t z = 0;
};

struct PixelOutcome {
    bool frame_written = false;
    std::uint32_t frame_stored = 0;  // new stored value when frame_written, else the old one
    bool depth_written = false;
    std::uint32_t depth_stored = 0;  // new stored value when depth_written, else the old one
};

// Runs the per-pixel stages for one fragment against the stored destination pixel.
// Depth test failure discards the fragment (colour and Z).
PixelOutcome process_pixel(const PixelState& state, const Fragment& fragment, std::uint32_t frame_stored_old,
                           std::uint32_t depth_stored_old);

}  // namespace coregs::pixel
