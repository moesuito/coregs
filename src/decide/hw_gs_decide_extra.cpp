// SPDX-License-Identifier: MIT
//
// HWR per-draw decisions, second part: texture and channel shuffles (chapter 09), GT4
// brightness patterns (09.4), clears (05.8), page rows and the pipeline key (03.9).
// Written from docs/hw-renderer-spec (clean room, decision 0069). Interpretations are
// recorded as IN-100 and up in docs/hw-renderer-spec/IMPLEMENTATION-NOTES.md.

#include "hw_gs_decide.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace coregs::decide {

namespace {

struct Box {
    double x0, y0, x1, y1;
};

Box sprite_box_of(const BatchGeometry& g, std::size_t sprite) {
    const Vertex2D& a = g.vertices[g.indices[2 * sprite]];
    const Vertex2D& b = g.vertices[g.indices[2 * sprite + 1]];
    return Box{std::min(a.x, b.x), std::min(a.y, b.y), std::max(a.x, b.x), std::max(a.y, b.y)};
}

double triangle_area_of(const Vertex2D& a, const Vertex2D& b, const Vertex2D& c) {
    return 0.5 * std::fabs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y));
}

// True when the batch covers the rectangle exactly once: no overlap and the area sum matches.
bool covers_without_gaps(const BatchGeometry& g, const PixelRect& rect, OverlapKind overlap) {
    if (rect.empty() || overlap != OverlapKind::None) {
        return false;
    }
    double area = 0.0;
    if (g.prim == PrimClass::Sprite) {
        for (std::size_t s = 0; s < g.indices.size() / 2; ++s) {
            const Box b = sprite_box_of(g, s);
            area += (b.x1 - b.x0) * (b.y1 - b.y0);
        }
    } else if (g.prim == PrimClass::Triangle) {
        for (std::size_t t = 0; t + 2 < g.indices.size(); t += 3) {
            area += triangle_area_of(g.vertices[g.indices[t]], g.vertices[g.indices[t + 1]], g.vertices[g.indices[t + 2]]);
        }
    } else {
        return false;
    }
    const double rect_area = static_cast<double>(rect.x1 - rect.x0) * static_cast<double>(rect.y1 - rect.y0);
    return std::fabs(area - rect_area) < 1e-9;
}

// Packs a constant colour in the frame format (5:5:5:1 for 16-bit, 8:8:8:8 otherwise).
std::uint32_t pack_colour(FrameFormat frame, const std::array<std::uint8_t, 4>& c) {
    switch (frame) {
        case FrameFormat::Ct16:
            return static_cast<std::uint32_t>((c[0] >> 3) | ((c[1] >> 3) << 5) | ((c[2] >> 3) << 10) | ((c[3] >> 7) << 15));
        case FrameFormat::Ct24:
            return static_cast<std::uint32_t>(c[0] | (c[1] << 8) | (c[2] << 16));
        case FrameFormat::Ct32:
            break;
    }
    return static_cast<std::uint32_t>(c[0] | (c[1] << 8) | (c[2] << 16) | (c[3] << 24));
}

bool bit_class_matches(FrameFormat frame, ZFormat z) {
    return (frame == FrameFormat::Ct32 && z == ZFormat::Z32) || (frame == FrameFormat::Ct24 && z == ZFormat::Z24) ||
           (frame == FrameFormat::Ct16 && z == ZFormat::Z16);
}

// Fully writable channels only (no partial FBMSK bits, no masked-out valid channel).
bool all_channels_writable(FrameFormat frame, std::uint32_t fbmsk) {
    const ChannelWrites cw = channel_writes(frame, fbmsk);
    if (cw.any_partial) {
        return false;
    }
    for (int c = 0; c < 4; ++c) {
        const bool valid = !(frame == FrameFormat::Ct24 && c == 3);
        if (valid && cw.channel[c] != ChannelState::Normal) {
            return false;
        }
    }
    return true;
}

bool form_is_low_ones(std::uint32_t mask) {
    // 0...01...1: the protected bits are a prefix of ones, so the writable bits are the low ones.
    const std::uint32_t writable = ~mask & 0xFFFFu;
    return ((writable + 1u) & writable) == 0u;
}

}  // namespace

// ---- Page rows ---------------------------------------------------------------------

std::uint32_t page_rows_for_format(FrameFormat frame, ZFormat z, bool use_z) {
    if (use_z) {
        return z == ZFormat::Z16 ? 64u : 32u;
    }
    return frame == FrameFormat::Ct16 ? 64u : 32u;
}

// ---- Texture shuffles (chapter 09.2) -------------------------------------------------------

ShuffleKind classify_texture_shuffle(const ShuffleSample& s) {
    if (!s.frame_16bit || !s.sprite_or_quads) {
        return ShuffleKind::None;
    }
    const std::uint32_t width = s.width_pixels;
    const bool width_ok = width == 2 || (width != 0 && width % 8 == 0);
    if (!width_ok) {
        return ShuffleKind::None;
    }
    // The texture width matches the frame width of the first quad.
    if (std::fabs((s.tex_u1 - s.tex_u0) - (s.frame_x1 - s.frame_x0)) > 1e-9 &&
        std::fabs(std::fabs(s.tex_u1 - s.tex_u0) - std::fabs(s.frame_x1 - s.frame_x0)) > 1e-9) {
        return ShuffleKind::None;
    }
    // Second check after lookups (09.2): sources must be 32-bit targets, or the frame itself.
    if (!s.source_is_32bit && !s.texture_is_frame) {
        return ShuffleKind::None;
    }

    const int offset = static_cast<int>(std::lround(s.tex_u0 - s.frame_x0)) & 15;
    const bool reversed = (s.frame_x1 - s.frame_x0) * (s.tex_u1 - s.tex_u0) < 0.0;
    const bool region_repeat = s.region_repeat_s;

    if (offset == 0 && !reversed) {
        const bool adjacent = s.next_quad_adjacent || s.next_quad_8_apart;
        if (adjacent && form_is_low_ones(s.fbmsk) && !s.texture_is_frame && s.opaque && s.nearest) {
            return ShuffleKind::Copy;
        }
    }
    if (offset == 8 && !region_repeat) {
        return ShuffleKind::Offset;
    }
    if (region_repeat && (((s.region_mask_s & 8u) == 0u) || s.region_fix_s == 8u)) {
        return ShuffleKind::RegionRepeat8;
    }
    if (region_repeat && (s.region_mask_s & 0xFu) == 0xFu && (s.region_fix_s & 0xFu) == 0u &&
        s.frame_x0 == static_cast<double>(s.region_fix_s) && width == 16) {
        return ShuffleKind::RegionRepeat16;
    }
    if (offset == 0 && width == 16 && reversed) {
        return ShuffleKind::Reverse;
    }
    if (s.axes_swapped && width == 8) {
        return ShuffleKind::Swizzle;
    }
    return ShuffleKind::None;
}

ShuffleChannelSet texture_shuffle_channel_set(std::uint32_t fbmsk16) {
    ShuffleChannelSet set;
    set.rg = (fbmsk16 & 0xFFu) != 0xFFu;
    set.ba = ((fbmsk16 >> 8) & 0xFFu) != 0xFFu;
    return set;
}

std::array<int, 4> texture_shuffle_channel_source(ShuffleKind kind, std::uint32_t fbmsk) {
    (void)fbmsk;
    switch (kind) {
        case ShuffleKind::Copy: return {0, 1, 2, 3};
        case ShuffleKind::Offset:
        case ShuffleKind::Reverse: return {2, 3, 0, 1};  // R<-B, G<-A, B<-R, A<-G
        case ShuffleKind::Swizzle: return {0, 1, 2, 2};  // usually B -> A
        case ShuffleKind::RegionRepeat8:
        case ShuffleKind::RegionRepeat16:
        case ShuffleKind::None: break;
    }
    return {-1, -1, -1, -1};  // selected per texel by the shader
}

// ---- Channel shuffles (chapter 09.3) ----------------------------------------------------------

bool is_channel_shuffle_candidate(const ChannelCandidate& c) {
    if (!c.tme || !c.sprite || !c.t8_format || c.tiny) {
        return false;
    }
    const bool within_page = c.one_page && (c.first_sprite8 || c.first_sprite2tall);
    return within_page || c.region_mode;
}

int channel_shuffle_channel(const ChannelCandidate& c) {
    if (c.wms_region_repeat && c.region_repeat_fix_bit3) {
        // B or A: B when WMT is not region repeat and v bit 5 is clear, or WMT region repeat with fix bit 1 clear.
        const bool choose_b = (!c.wmt_region_repeat && !c.v_bit5) || (c.wmt_region_repeat && !c.wms_fix_bit1);
        return choose_b ? 2 : 3;
    }
    if (c.wms_region_repeat && c.region_repeat_mask_bit3_clear) {
        // R or G: G when WMT region repeat with fix bit 1 set, or integer coordinates with v bit 5 set.
        const bool choose_g = (c.wmt_region_repeat && c.wms_fix_bit1) || (c.integer_coords && c.v_bit5);
        return choose_g ? 1 : 0;
    }
    // Lane of the first sprite (09.1 lane table: (u&8, v&2) selects R, G, B or A).
    return ((c.u_mod16 & 8u) ? 2 : 0) + ((c.v_mod4 & 2u) ? 1 : 0);
}

std::vector<std::uint32_t> coalesce_channel_shuffle_runs(const std::vector<ChannelCandidate>& batches) {
    std::vector<std::uint32_t> run_lengths;
    std::uint32_t current = 0;
    for (std::size_t i = 0; i < batches.size(); ++i) {
        const ChannelCandidate& b = batches[i];
        if (!is_channel_shuffle_candidate(b)) {
            if (current != 0) run_lengths.push_back(current);
            current = 0;
            continue;
        }
        const bool same_run = current != 0 && b.fbmsk == batches[i - 1].fbmsk &&
                              b.clut_identity == batches[i - 1].clut_identity && b.fbp >= batches[i - 1].fbp;
        if (same_run) {
            ++current;
        } else {
            if (current != 0) run_lengths.push_back(current);
            current = 1;
        }
    }
    if (current != 0) run_lengths.push_back(current);
    return run_lengths;
}


// ---- Clears (chapter 05.8) -------------------------------------------------------------------

ClearKind detect_target_clear(const BatchInput& input, const DeviceCaps& caps, PixelRect& clear_rect,
                              std::uint32_t& colour_word) {
    (void)caps;
    const RegisterSnapshot& r = input.regs;
    const BatchGeometry& g = input.geometry;
    if (r.tme || r.abe || r.date || r.ate || g.vertices.empty()) {
        return ClearKind::None;
    }
    if (!all_channels_writable(r.frame, r.fbmsk)) {
        return ClearKind::None;
    }
    const VertexTrace trace = compute_vertex_trace(g, r);
    if (!trace.colour_constant) {
        return ClearKind::None;
    }
    const bool z_written = !(r.zmsk || !r.zte || (r.ate && r.atst == AlphaCompare::Never && r.afail != AlphaFail::DepthOnly));
    if (z_written && (!trace.z_constant || r.ztst != DepthCompare::Always)) {
        return ClearKind::None;
    }
    const OverlapResult overlap = classify_overlap(g, trace);
    const PixelRect rect = draw_rectangle(trace, r);
    if (!covers_without_gaps(g, rect, overlap.kind)) {
        return ClearKind::None;
    }
    colour_word = pack_colour(r.frame, trace.rgba_min);
    clear_rect = rect;
    if (input.clear.target_present) {
        // A target clear must cover the whole scissor area, the best estimate of the valid area (IN-108).
        const bool whole_scissor = rect.x0 == r.scax0 && rect.x1 == r.scax1 + 1 && rect.y0 == r.scay0 &&
                                   rect.y1 == r.scay1 + 1;
        return whole_scissor ? ClearKind::TargetClear : ClearKind::None;
    }
    {
        // A memory clear needs a width of at least (FBW - 1) * 64 pixels (no target to clear).
        const int width = rect.x1 - rect.x0;
        if (static_cast<std::uint32_t>(width) < (r.fbw - 1u) * 64u) {
            return ClearKind::None;
        }
        return ClearKind::MemoryClear;
    }
}

ClearKind detect_double_half_clear(const BatchInput& input, PixelRect& clear_rect, std::uint32_t& base_fbp) {
    const RegisterSnapshot& r = input.regs;
    const BatchGeometry& g = input.geometry;
    if (!bit_class_matches(r.frame, r.z_format) || r.ztst != DepthCompare::Always || r.ate || r.date || r.tme ||
        r.abe || r.zmsk || !all_channels_writable(r.frame, r.fbmsk) || g.vertices.empty()) {
        return ClearKind::None;
    }
    if (input.clear.next_batch_nonclear_same_pair) {
        return ClearKind::None;
    }
    const VertexTrace trace = compute_vertex_trace(g, r);
    if (!trace.colour_constant || !trace.z_constant) {
        return ClearKind::None;
    }
    const std::uint32_t colour = pack_colour(r.frame, trace.rgba_min);
    if (colour != trace.z_min) {
        return ClearKind::None;
    }
    const OverlapResult overlap = classify_overlap(g, trace);
    const PixelRect rect = draw_rectangle(trace, r);
    if (!covers_without_gaps(g, rect, overlap.kind) || rect.x0 != 0 || rect.y0 != 0) {
        return ClearKind::None;
    }
    const int height = rect.y1 - rect.y0;
    const std::uint32_t page_rows = page_rows_for_format(r.frame, r.z_format, false);
    if (height <= 0 || static_cast<std::uint32_t>(height) % page_rows != 0) {
        return ClearKind::None;
    }
    // ZBUF sits one half of the buffer below FRAME (vertical case only, IN-106).
    if (r.zbp <= r.fbp) {
        return ClearKind::None;
    }
    const std::uint32_t half_pages = (static_cast<std::uint32_t>(height) / page_rows) * r.fbw;
    if (r.zbp - r.fbp != half_pages) {
        return ClearKind::None;
    }
    const int doubled = height * 2;
    if (doubled >= static_cast<int>(input.clear.display_height) || height > 300) {
        return ClearKind::None;
    }
    clear_rect = PixelRect{0, 0, rect.x1, doubled};
    base_fbp = r.fbp;
    return ClearKind::DoubleHalf;
}

ClearKind detect_striped_double_clear(const BatchInput& input, PixelRect& clear_rect, std::uint32_t& base_fbp) {
    const RegisterSnapshot& r = input.regs;
    const BatchGeometry& g = input.geometry;
    if (g.prim != PrimClass::Sprite || g.indices.size() < 2 || r.fbw <= 1 || !bit_class_matches(r.frame, r.z_format) ||
        r.ztst != DepthCompare::Always || r.ate || r.date || r.tme || r.abe || r.zmsk ||
        !all_channels_writable(r.frame, r.fbmsk)) {
        return ClearKind::None;
    }
    const std::uint32_t fbp = r.fbp;
    const std::uint32_t zbp = r.zbp;
    const bool one_page_apart = (zbp == fbp) || (zbp == fbp + 1) || (fbp == zbp + 1);
    if (!one_page_apart) {
        return ClearKind::None;
    }
    const VertexTrace trace = compute_vertex_trace(g, r);
    if (!trace.colour_constant || !trace.z_constant || pack_colour(r.frame, trace.rgba_min) != trace.z_min) {
        return ClearKind::None;
    }
    const OverlapResult overlap = classify_overlap(g, trace);
    const PixelRect rect = draw_rectangle(trace, r);
    if (!covers_without_gaps(g, rect, overlap.kind)) {
        return ClearKind::None;
    }
    const Vertex2D& a = g.vertices[g.indices[0]];
    const Vertex2D& b = g.vertices[g.indices[1]];
    const int strip = static_cast<int>(std::lround(std::fabs(b.x - a.x)));
    if (strip != 64 && strip != 32) {
        return ClearKind::None;
    }
    // Hypothesis (IN-107): widen to the right when Z is ahead, to the left otherwise.
    const bool z_ahead = zbp > fbp;
    clear_rect = PixelRect{rect.x0 - (z_ahead ? 0 : strip), rect.y0, rect.x1 + (z_ahead ? strip : 0), rect.y1};
    base_fbp = std::min(fbp, zbp);
    return ClearKind::StripedDouble;
}

// ---- Pipeline key (chapter 03.9) -------------------------------------------------------------

PipelineKey make_pipeline_key(const DrawPlan& plan, const RegisterSnapshot& regs, const DeviceCaps& caps) {
    (void)caps;
    PipelineKey k;
    const bool rgb = plan.writes.rgb;
    const bool texture = plan.texture_used;

    // Vertex key.
    k.tme = texture;
    k.fst = texture && regs.fst;
    k.iip = regs.iip;
    k.expansion = plan.primitive.expansion;
    k.topology = static_cast<std::uint8_t>(plan.primitive.prim);

    // Pixel key: texture group, zeroed when the texture is not used.
    if (texture) {
        k.tfx = static_cast<std::uint8_t>(regs.tfx);
        k.tcc = regs.tcc ? 1 : 0;
        k.wrap_s = static_cast<std::uint8_t>(regs.wms);
        k.wrap_t = static_cast<std::uint8_t>(regs.wmt);
        k.palette_format = regs.tex_format == TexFormat::T8 ? 1 : (regs.tex_format == TexFormat::T4 ? 2 : 0);
    }
    k.fog = regs.fge;
    k.fba = regs.fba;
    k.dest_format = static_cast<std::uint8_t>(regs.frame);
    k.dither = plan.dither;
    k.colour_wrap_shader = plan.colour_wrap == ColourWrapPlan::ShaderFullResult;
    k.pabe = regs.pabe && regs.abe;
    k.alpha_method = static_cast<std::uint8_t>(plan.alpha_test.method);
    k.date_method = static_cast<std::uint8_t>(plan.date.method);
    k.software_depth = plan.alpha_test.method == AlphaMethod::Feedback && plan.writes.depth;
    k.software_fbmask = plan.software_fbmask;

    // Fixed-function blend: off when no RGB is written (blend and dual source dropped) and for
    // the shader-computed methods.
    k.blend_method = static_cast<std::uint8_t>(plan.blend.method);
    const bool ff_blend = rgb && (plan.blend.method == BlendMethod::FixedFunction ||
                                  plan.blend.method == BlendMethod::Accumulate ||
                                  plan.blend.method == BlendMethod::Mixed);
    if (ff_blend && !(plan.blend.group == BlendGroup::SameAsD)) {
        k.blend_enabled = true;
        k.blend_op = static_cast<std::uint8_t>(static_cast<int>(plan.blend.mode.a) * 3 + static_cast<int>(plan.blend.mode.b));
        k.blend_src = static_cast<std::uint8_t>(plan.blend.mode.c);
        k.blend_dst = static_cast<std::uint8_t>(plan.blend.mode.d);
    }
    k.dual_source = rgb && !plan.alpha_test.passes.empty() && plan.alpha_test.passes.front().dual_source_alpha;

    // Masks, depth and topology.
    k.write_rgb = plan.writes.rgb;
    k.write_alpha = plan.writes.alpha;
    k.write_depth = plan.writes.depth;
    k.depth_compare = plan.depth_test_always ? static_cast<std::uint8_t>(DepthCompare::Always)
                                             : static_cast<std::uint8_t>(regs.ztst);
    return k;
}

}  // namespace coregs::decide
