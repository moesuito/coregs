// SPDX-License-Identifier: Apache-2.0
//
// HWR uber pixel core (TASK-027, M4). Copyright (c) 2026 GT4Recomp contributors.
//
// Written from the TASK-026 specification (docs/hw-renderer-spec),
// chapters 01.5, 03.3, 03.4, 03.9, 07.5, 08.1 to 08.11;
// clean room under decision 0069. No other renderer's shader or source was consulted.
//
// This file has no #version line: the shader entry points include it after their own
// #version and GL_GOOGLE_include_directive lines. Every operation is 32-bit integer
// arithmetic and mirrors native/hw_gs/pixel/hw_gs_pixel.cpp step for step, so the GPU
// result is bit-exact with the CPU reference (implementation notes IN-200 to IN-213).
//
// Enumerations are passed as uint values with the same numbering as the C++ enums:
//   format   0 CT32, 1 CT24, 2 CT16
//   z format 0 Z32, 1 Z24, 2 Z16
//   tfx      0 MODULATE, 1 DECAL, 2 HIGHLIGHT, 3 HIGHLIGHT2
//   texel    0 RGBA8 (raw bytes R,G,B,A), 1 CT16 (TEXA expansion), 2 CT24 (TEXA expansion)
//   alpha compare 0 NEVER, 1 ALWAYS, 2 LESS, 3 LEQUAL, 4 EQUAL, 5 GEQUAL, 6 GREATER, 7 NOTEQUAL
//   alpha fail    0 KEEP, 1 FB_ONLY, 2 ZB_ONLY, 3 RGB_ONLY
//   depth compare 0 NEVER, 1 ALWAYS, 2 GEQUAL, 3 GREATER
//   blend source  0 Cs, 1 Cd, 2 zero; blend factor 0 As, 1 Ad, 2 FIX

const uint HW_FORMAT_CT32 = 0u;
const uint HW_FORMAT_CT24 = 1u;
const uint HW_FORMAT_CT16 = 2u;

const uint HW_Z_FORMAT_Z32 = 0u;
const uint HW_Z_FORMAT_Z24 = 1u;

const uint HW_TFX_MODULATE = 0u;
const uint HW_TFX_DECAL = 1u;
const uint HW_TFX_HIGHLIGHT = 2u;

const uint HW_TEXEL_RGBA8 = 0u;
const uint HW_TEXEL_CT16 = 1u;
const uint HW_TEXEL_CT24 = 2u;
// IN-409 (CORE2): the texel is a CLUT index, looked up in a palette row at draw time (see hw_pixel_uber.frag).
const uint HW_TEXEL_INDEXED = 3u;

const uint HW_ALPHA_FAIL_FB_ONLY = 1u;
const uint HW_ALPHA_FAIL_ZB_ONLY = 2u;
const uint HW_ALPHA_FAIL_RGB_ONLY = 3u;

const uint HW_DEPTH_NEVER = 0u;
const uint HW_DEPTH_ALWAYS = 1u;
const uint HW_DEPTH_GEQUAL = 2u;

const uint HW_BLEND_SRC_CS = 0u;
const uint HW_BLEND_SRC_CD = 1u;
const uint HW_BLEND_FACTOR_AS = 0u;
const uint HW_BLEND_FACTOR_AD = 1u;

// Pipeline state consumed by the pixel stages. Field meanings follow the
// PixelState of the C++ reference.
struct HwPixelState {
    uint format;
    uint fbmsk;
    uint fba;
    uint dither;
    uint dimx_lo;  // DIMX bits 0..31
    uint dimx_hi;  // DIMX bits 32..63
    uint ate;
    uint atst;
    uint afail;
    uint aref;
    uint date_on;
    uint datm;
    uint ztest;
    uint ztst;
    uint zmsk;
    uint zformat;
    uint blend_on;
    uint pabe;
    uint blend_a;
    uint blend_b;
    uint blend_c;
    uint blend_d;
    uint fix;
    uint colclamp;
};

struct HwFragment {
    uint x;
    uint y;
    ivec4 colour;  // 0..255 per channel, alpha is As before FBA
    uint z;        // interpolated integer Z
};

struct HwOutcome {
    bool frame_written;
    uint frame_stored;
    bool depth_written;
    uint depth_stored;
};

struct HwWrite {
    bool rgb;
    bool alpha;
    bool depth;
};

HwWrite hw_write_all() {
    return HwWrite(true, true, true);
}

HwWrite hw_write_none() {
    return HwWrite(false, false, false);
}

HwWrite hw_write_and(HwWrite a, HwWrite b) {
    return HwWrite(a.rgb && b.rgb, a.alpha && b.alpha, a.depth && b.depth);
}

// ---- Frame words (01.3, 01.5) -------------------------------------------------

uint hw_valid_mask(uint format) {
    if (format == HW_FORMAT_CT24) {
        return 0x00FFFFFFu;
    }
    if (format == HW_FORMAT_CT16) {
        return 0x80F8F8F8u;
    }
    return 0xFFFFFFFFu;
}

uint hw_unpack_frame_word(uint format, uint stored) {
    if (format == HW_FORMAT_CT24) {
        return stored & 0x00FFFFFFu;
    }
    if (format == HW_FORMAT_CT16) {
        uint r = stored & 0x1Fu;
        uint g = (stored >> 5) & 0x1Fu;
        uint b = (stored >> 10) & 0x1Fu;
        uint a = (stored >> 15) & 1u;
        return (r << 3) | (g << 11) | (b << 19) | (a << 31);
    }
    return stored;
}

uint hw_pack_frame_word(uint format, uint word) {
    if (format == HW_FORMAT_CT24) {
        return word & 0x00FFFFFFu;
    }
    if (format == HW_FORMAT_CT16) {
        uint r = (word >> 3) & 0x1Fu;
        uint g = (word >> 11) & 0x1Fu;
        uint b = (word >> 19) & 0x1Fu;
        uint a = (word >> 31) & 1u;
        return r | (g << 5) | (b << 10) | (a << 15);
    }
    return word;
}

// Colour of a frame word: the four byte lanes are R, G, B, A.
ivec4 hw_frame_word_to_colour(uint word) {
    return ivec4(int(word & 0xFFu), int((word >> 8) & 0xFFu), int((word >> 16) & 0xFFu), int((word >> 24) & 0xFFu));
}

// Frame word of a clamped or wrapped colour (channels already 0..255 after COLCLAMP).
uint hw_colour_to_frame_word(uint format, ivec4 c) {
    uint r = uint(c.r) & 0xFFu;
    uint g = uint(c.g) & 0xFFu;
    uint b = uint(c.b) & 0xFFu;
    uint a = uint(c.a) & 0xFFu;
    if (format == HW_FORMAT_CT24) {
        return r | (g << 8) | (b << 16);
    }
    if (format == HW_FORMAT_CT16) {
        return (r & 0xF8u) | ((g & 0xF8u) << 8) | ((b & 0xF8u) << 16) | ((a & 0x80u) << 24);
    }
    return r | (g << 8) | (b << 16) | (a << 24);
}

// FBMSK merge (08.9): bits written are enabled, valid and not protected.
uint hw_merge_frame_word(uint format, uint old_word, uint computed, uint fbmsk, HwWrite e) {
    uint valid = hw_valid_mask(format);
    uint lanes = 0u;
    if (e.rgb) {
        lanes |= 0x00FFFFFFu;
    }
    if (e.alpha) {
        lanes |= 0xFF000000u;
    }
    uint written_lanes = lanes & valid & ~(fbmsk & valid);
    return (old_word & ~written_lanes) | (computed & written_lanes);
}

// ---- Texture function and TEXA (01.5, 03.4) -----------------------------------

int hw_scaled_product(int texel, int fragment) {
    return (texel * fragment) >> 7;
}

ivec4 hw_texture_function(uint tfx, bool tcc, ivec4 texel, ivec4 frag) {
    ivec4 o = ivec4(0);
    if (tfx == HW_TFX_MODULATE) {
        o.r = min(255, hw_scaled_product(texel.r, frag.r));
        o.g = min(255, hw_scaled_product(texel.g, frag.g));
        o.b = min(255, hw_scaled_product(texel.b, frag.b));
        o.a = tcc ? min(255, hw_scaled_product(texel.a, frag.a)) : frag.a;
    } else if (tfx == HW_TFX_DECAL) {
        o.rgb = texel.rgb;
        o.a = tcc ? texel.a : frag.a;
    } else if (tfx == HW_TFX_HIGHLIGHT) {
        o.r = min(255, hw_scaled_product(texel.r, frag.r) + frag.a);
        o.g = min(255, hw_scaled_product(texel.g, frag.g) + frag.a);
        o.b = min(255, hw_scaled_product(texel.b, frag.b) + frag.a);
        o.a = tcc ? min(255, texel.a + frag.a) : frag.a;
    } else {
        // HIGHLIGHT2
        o.r = min(255, hw_scaled_product(texel.r, frag.r) + frag.a);
        o.g = min(255, hw_scaled_product(texel.g, frag.g) + frag.a);
        o.b = min(255, hw_scaled_product(texel.b, frag.b) + frag.a);
        o.a = tcc ? texel.a : frag.a;
    }
    return o;
}

ivec4 hw_expand_texel16(uint texel, uint ta0, uint ta1, bool aem) {
    ivec4 o;
    o.r = int((texel & 0x1Fu) << 3);
    o.g = int(((texel >> 5) & 0x1Fu) << 3);
    o.b = int(((texel >> 10) & 0x1Fu) << 3);
    // IN-07 / 07.5: the A bit selects TA1 before the AEM black-texel test.
    if ((texel & 0x8000u) != 0u) {
        o.a = int(ta1);
    } else {
        o.a = int(ta0);
        if (aem && o.r == 0 && o.g == 0 && o.b == 0) {
            o.a = 0;
        }
    }
    return o;
}

ivec4 hw_expand_texel24(uint texel, uint ta0, bool aem) {
    ivec4 o;
    o.r = int(texel & 0xFFu);
    o.g = int((texel >> 8) & 0xFFu);
    o.b = int((texel >> 16) & 0xFFu);
    o.a = int(ta0);
    if (aem && o.r == 0 && o.g == 0 && o.b == 0) {
        o.a = 0;
    }
    return o;
}

// Texel colour from the raw texel word of the given kind.
ivec4 hw_texel_colour(uint kind, uint raw, uint ta0, uint ta1, bool aem) {
    if (kind == HW_TEXEL_CT16) {
        return hw_expand_texel16(raw & 0xFFFFu, ta0, ta1, aem);
    }
    if (kind == HW_TEXEL_CT24) {
        return hw_expand_texel24(raw & 0xFFFFFFu, ta0, aem);
    }
    return ivec4(int(raw & 0xFFu), int((raw >> 8) & 0xFFu), int((raw >> 16) & 0xFFu), int((raw >> 24) & 0xFFu));
}

// Fragment colour after the texture function and fog (03.4): the input to process_pixel.
ivec4 hw_shade_fragment(bool tex_on, uint tfx, bool tcc, uint texel_kind, uint texel_raw, uint ta0, uint ta1,
                        bool aem, bool fog_on, int fog, ivec3 fog_colour, ivec4 frag) {
    ivec4 colour = frag;
    if (tex_on) {
        colour = hw_texture_function(tfx, tcc, hw_texel_colour(texel_kind, texel_raw, ta0, ta1, aem), frag);
    }
    if (fog_on) {
        colour.r = (fog * colour.r + (255 - fog) * fog_colour.r) >> 8;
        colour.g = (fog * colour.g + (255 - fog) * fog_colour.g) >> 8;
        colour.b = (fog * colour.b + (255 - fog) * fog_colour.b) >> 8;
    }
    return colour;
}

// ---- Alpha test (08.6) --------------------------------------------------------

bool hw_alpha_compare(uint compare, int alpha, int reference) {
    if (compare == 0u) {
        return false;
    }
    if (compare == 1u) {
        return true;
    }
    if (compare == 2u) {
        return alpha < reference;
    }
    if (compare == 3u) {
        return alpha <= reference;
    }
    if (compare == 4u) {
        return alpha == reference;
    }
    if (compare == 5u) {
        return alpha >= reference;
    }
    if (compare == 6u) {
        return alpha > reference;
    }
    return alpha != reference;
}

HwWrite hw_alpha_test_write(uint ate, uint atst, uint afail, int aref, int alpha, uint format) {
    if (ate == 0u) {
        return hw_write_all();
    }
    if (hw_alpha_compare(atst, alpha, aref)) {
        return hw_write_all();
    }
    uint fail = afail;
    if (fail == HW_ALPHA_FAIL_RGB_ONLY && format != HW_FORMAT_CT32) {
        fail = HW_ALPHA_FAIL_FB_ONLY;  // RGB_ONLY on a non-32-bit frame acts as FB_ONLY
    }
    if (fail == 0u) {
        return hw_write_none();
    }
    if (fail == HW_ALPHA_FAIL_FB_ONLY) {
        return HwWrite(true, true, false);
    }
    if (fail == HW_ALPHA_FAIL_ZB_ONLY) {
        return HwWrite(false, false, true);
    }
    return HwWrite(true, false, false);
}

// ---- Destination alpha test (08.7) --------------------------------------------

HwWrite hw_date_write(uint date_on, uint datm, uint format, uint old_word) {
    if (date_on == 0u) {
        return hw_write_all();
    }
    if (format == HW_FORMAT_CT24) {
        return hw_write_none();
    }
    bool msb = ((old_word >> 31) & 1u) != 0u;
    return (msb == (datm != 0u)) ? hw_write_all() : hw_write_none();
}

int hw_destination_alpha(uint format, uint old_word) {
    if (format == HW_FORMAT_CT24) {
        return 0x80;
    }
    return int((old_word >> 24) & 0xFFu);
}

// ---- Depth (03.7) -------------------------------------------------------------

uint hw_depth_mask(uint zformat) {
    if (zformat == HW_Z_FORMAT_Z32) {
        return 0xFFFFFFFFu;
    }
    if (zformat == HW_Z_FORMAT_Z24) {
        return 0x00FFFFFFu;
    }
    return 0x0000FFFFu;
}

uint hw_clamp_z(uint z, uint zformat) {
    return min(z, hw_depth_mask(zformat));
}

bool hw_depth_passes(uint compare, uint z_new, uint z_stored_raw, uint zformat) {
    uint mask = hw_depth_mask(zformat);
    uint incoming = z_new & mask;
    uint stored = z_stored_raw & mask;
    if (compare == HW_DEPTH_NEVER) {
        return false;
    }
    if (compare == HW_DEPTH_ALWAYS) {
        return true;
    }
    if (compare == HW_DEPTH_GEQUAL) {
        return incoming >= stored;
    }
    return incoming > stored;
}

uint hw_write_depth_raw(uint stored_raw, uint z, uint zformat) {
    uint mask = hw_depth_mask(zformat);
    return (stored_raw & ~mask) | (z & mask);
}

// ---- Blending (08.1, 08.5) ----------------------------------------------------

int hw_pick_source(uint source, int cs, int cd) {
    if (source == HW_BLEND_SRC_CS) {
        return cs;
    }
    if (source == HW_BLEND_SRC_CD) {
        return cd;
    }
    return 0;
}

// Cv = (((A - B) * C) >> 7) + D for one channel. The shift of a signed value is arithmetic
// (SPIR-V OpShiftRightArithmetic, implementation notes IN-209).
int hw_blend_channel(uint sel_a, uint sel_b, uint sel_c, uint sel_d, int factor, int cs, int cd) {
    int a = hw_pick_source(sel_a, cs, cd);
    int b = hw_pick_source(sel_b, cs, cd);
    int d = hw_pick_source(sel_d, cs, cd);
    return (((a - b) * factor) >> 7) + d;
}

int hw_blend_factor(uint sel_c, int as, int ad, uint fix) {
    if (sel_c == HW_BLEND_FACTOR_AS) {
        return as;
    }
    if (sel_c == HW_BLEND_FACTOR_AD) {
        return ad;
    }
    return int(fix);
}

// Unclamped blend of the RGB channels; alpha is the source alpha (as).
ivec4 hw_blend_unclamped(HwPixelState s, ivec4 source, ivec4 destination, int as, int ad) {
    int factor = hw_blend_factor(s.blend_c, as, ad, s.fix);
    ivec4 o;
    o.r = hw_blend_channel(s.blend_a, s.blend_b, s.blend_c, s.blend_d, factor, source.r, destination.r);
    o.g = hw_blend_channel(s.blend_a, s.blend_b, s.blend_c, s.blend_d, factor, source.g, destination.g);
    o.b = hw_blend_channel(s.blend_a, s.blend_b, s.blend_c, s.blend_d, factor, source.b, destination.b);
    o.a = as;
    return o;
}

// COLCLAMP: clamp to 0..255 when clamp_enable, else keep the low 8 bits.
int hw_colour_clamp_or_wrap(int value, bool clamp_enable) {
    return clamp_enable ? clamp(value, 0, 255) : (value & 0xFF);
}

// ---- Dither (01.5, 08.10) -----------------------------------------------------

int hw_dither_entry(uint dimx_lo, uint dimx_hi, uint x, uint y) {
    uint index = 4u * (y & 3u) + (x & 3u);
    uint position = 4u * index;
    uint field = position < 32u ? ((dimx_lo >> position) & 7u) : ((dimx_hi >> (position - 32u)) & 7u);
    return field >= 4u ? int(field) - 8 : int(field);
}

// ---- Whole pixel --------------------------------------------------------------

// Runs the per-pixel stages for one fragment against the stored destination pixel. The
// order is the one of the C++ reference: alpha test, DATE, depth test (discard on failure),
// blend, dither (16-bit only), COLCLAMP, FBA, format conversion, FBMSK merge.
HwOutcome hw_process_pixel(HwPixelState s, HwFragment f, uint frame_old_stored, uint depth_old_stored) {
    HwOutcome o;
    o.frame_written = false;
    o.frame_stored = frame_old_stored;
    o.depth_written = false;
    o.depth_stored = depth_old_stored;

    uint format = s.format;
    uint old_word = hw_unpack_frame_word(format, frame_old_stored);

    HwWrite e = hw_alpha_test_write(s.ate, s.atst, s.afail, int(s.aref), f.colour.a, format);
    e = hw_write_and(e, hw_date_write(s.date_on, s.datm, format, old_word));

    uint z = hw_clamp_z(f.z, s.zformat);
    if (s.ztest != 0u) {
        if (!hw_depth_passes(s.ztst, z, depth_old_stored, s.zformat)) {
            return o;
        }
    }
    e.depth = e.depth && s.ztest != 0u && s.zmsk == 0u;

    if (!(e.rgb || e.alpha || e.depth)) {
        return o;
    }

    ivec4 source = f.colour;
    ivec4 result = source;
    if (s.blend_on != 0u && !(s.pabe != 0u && (source.a & 0x80) == 0)) {
        ivec4 destination = hw_frame_word_to_colour(old_word);
        result = hw_blend_unclamped(s, source, destination, source.a, hw_destination_alpha(format, old_word));
    }

    if (format == HW_FORMAT_CT16 && s.dither != 0u) {
        int dither = hw_dither_entry(s.dimx_lo, s.dimx_hi, f.x, f.y);
        result.r += dither;
        result.g += dither;
        result.b += dither;
    }
    bool clamp_enable = s.colclamp != 0u;
    result.r = hw_colour_clamp_or_wrap(result.r, clamp_enable);
    result.g = hw_colour_clamp_or_wrap(result.g, clamp_enable);
    result.b = hw_colour_clamp_or_wrap(result.b, clamp_enable);

    if (s.fba != 0u) {
        result.a |= 0x80;
    }
    uint computed = hw_colour_to_frame_word(format, result);
    uint merged = hw_merge_frame_word(format, old_word, computed, s.fbmsk, e);
    if (e.rgb || e.alpha) {
        o.frame_written = true;
        o.frame_stored = hw_pack_frame_word(format, merged);
    }
    if (e.depth) {
        o.depth_written = true;
        o.depth_stored = hw_write_depth_raw(depth_old_stored, z, s.zformat);
    }
    return o;
}
