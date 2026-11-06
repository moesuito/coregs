// SPDX-License-Identifier: Apache-2.0
//
// HWR per-draw decisions: trace, overlap, blend, DATE, alpha test, barriers and the draw
// plan (TASK-027). Written from docs/hw-renderer-spec, chapters 02.9, 02.10, 03, 08 (clean
// room, decision 0069). Shuffles, clears, patterns and pipeline keys are in
// hw_gs_decide_extra.cpp. Interpretations are recorded as IN-100 and up.

#include "hw_gs_decide.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace coregs::decide {

namespace {

constexpr std::uint8_t kAlphaHalf = 0x80;  // 1.0 in GS factor and alpha units

int floor_to_int(double value) { return static_cast<int>(std::floor(value)); }
int ceil_to_int(double value) { return static_cast<int>(std::ceil(value)); }

// Vertex texel coordinates (chapter 03.5 and 02.9): FST uses the 12.4 fields, STQ divides by Q.
void texel_coordinates(const Vertex2D& v, const RegisterSnapshot& regs, double& u, double& w) {
    if (regs.fst) {
        u = v.u;
        w = v.v;
        return;
    }
    const double scale_u = std::ldexp(1.0, regs.tw_log2);
    const double scale_v = std::ldexp(1.0, regs.th_log2);
    u = (v.s / v.q) * scale_u;
    w = (v.t / v.q) * scale_v;
}

bool is_sprite_like(PrimClass prim) { return prim == PrimClass::Sprite; }

// Half-open box of a primitive for overlap tests (no margin: exact coverage).
struct Box {
    double x0, y0, x1, y1;
    bool intersects(const Box& o) const { return x0 < o.x1 && o.x0 < x1 && y0 < o.y1 && o.y0 < y1; }
};

Box sprite_box(const BatchGeometry& g, std::size_t sprite) {
    const Vertex2D& a = g.vertices[g.indices[2 * sprite]];
    const Vertex2D& b = g.vertices[g.indices[2 * sprite + 1]];
    return Box{std::min(a.x, b.x), std::min(a.y, b.y), std::max(a.x, b.x), std::max(a.y, b.y)};
}

// Two triangles that together form an axis-aligned rectangle (chapter 02.10).
bool triangles_form_axis_quad(const BatchGeometry& g) {
    if (g.indices.size() != 6) {
        return false;
    }
    std::set<std::uint32_t> unique(g.indices.begin(), g.indices.end());
    if (unique.size() != 4) {
        return false;
    }
    std::vector<Vertex2D> pts;
    for (std::uint32_t index : unique) {
        pts.push_back(g.vertices[index]);
    }
    double xmin = pts[0].x, xmax = pts[0].x, ymin = pts[0].y, ymax = pts[0].y;
    for (const Vertex2D& p : pts) {
        xmin = std::min(xmin, p.x);
        xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y);
        ymax = std::max(ymax, p.y);
    }
    if (xmin == xmax || ymin == ymax) {
        return false;
    }
    std::set<std::pair<int, int>> corners;
    for (const Vertex2D& p : pts) {
        const bool on_x = p.x == xmin || p.x == xmax;
        const bool on_y = p.y == ymin || p.y == ymax;
        if (!on_x || !on_y) {
            return false;
        }
        corners.insert({p.x == xmin ? 0 : 1, p.y == ymin ? 0 : 1});
    }
    return corners.size() == 4;
}

// Simplification of chapter 08.2 (rules 1 to 5). Rule 6 (COLCLAMP write-back of values that are
// provably non-positive) is not applied: it changes no method choice in phase 1 (IN-105).
BlendMode simplify_blend(BlendMode mode, std::uint8_t& fix, const RegisterSnapshot& regs,
                         const TargetTracking& target, std::uint8_t frag_min, std::uint8_t frag_max,
                         bool cs_provably_zero, bool& changed) {
    if (cs_provably_zero) {
        if (mode.a == BlendSel::Cs) mode.a = BlendSel::Zero;
        if (mode.b == BlendSel::Cs) mode.b = BlendSel::Zero;
        if (mode.d == BlendSel::Cs) mode.d = BlendSel::Zero;
        changed = true;
    }
    // Rule 2: Ad with a single target alpha value becomes FIX (24-bit targets: 0x80).
    if (mode.c == BlendFactorSel::Ad) {
        if (regs.frame == FrameFormat::Ct24) {
            mode.c = BlendFactorSel::Fix;
            fix = kAlphaHalf;
            changed = true;
        } else if (target.alpha_min == target.alpha_max && !target.alpha_scaled) {
            mode.c = BlendFactorSel::Fix;
            fix = target.alpha_min;
            changed = true;
        }
    }
    // Rule 3: As with a single batch alpha value becomes FIX.
    if (mode.c == BlendFactorSel::As && frag_min == frag_max) {
        mode.c = BlendFactorSel::Fix;
        fix = frag_min;
        changed = true;
    }
    // Rule 5: a provably zero factor removes the multiplied term (A = B = 0 leaves D).
    const bool factor_zero = (mode.c == BlendFactorSel::Fix && fix == 0) ||
                             (mode.c == BlendFactorSel::As && frag_max == 0) ||
                             (mode.c == BlendFactorSel::Ad && target.alpha_max == 0 && !target.alpha_scaled);
    if (factor_zero && mode.a != mode.b) {
        mode.a = BlendSel::Zero;
        mode.b = BlendSel::Zero;
        changed = true;
    }
    // Rule 4: B = D with C exactly 1.0 gives A; D takes A's selector and A = B = 0.
    if (mode.c == BlendFactorSel::Fix && fix == kAlphaHalf && mode.b == mode.d && mode.a != mode.b) {
        const BlendSel old_a = mode.a;
        mode.a = BlendSel::Zero;
        mode.b = BlendSel::Zero;
        mode.d = old_a;
        changed = true;
    }
    return mode;
}

BlendPlan finish_blend(BlendPlan plan, BlendMethod method, BlendReason reason, bool approximate) {
    plan.method = method;
    plan.reason = reason;
    plan.approximate = approximate;
    plan.reads_destination = method != BlendMethod::None && blend_group_reads_destination(plan.group);
    return plan;
}

// Writes of a fragment that failed the test (chapter 08.6 table).
void add_masks(WriteMask& into, const WriteMask& add) {
    into.rgb = into.rgb || add.rgb;
    into.alpha = into.alpha || add.alpha;
    into.depth = into.depth || add.depth;
}

WriteMask intersect(const WriteMask& a, const WriteMask& b) {
    return WriteMask{a.rgb && b.rgb, a.alpha && b.alpha, a.depth && b.depth};
}

AlphaCompare invert_compare(AlphaCompare c) {
    switch (c) {
        case AlphaCompare::Never: return AlphaCompare::Always;
        case AlphaCompare::Always: return AlphaCompare::Never;
        case AlphaCompare::Less: return AlphaCompare::GreaterEqual;
        case AlphaCompare::GreaterEqual: return AlphaCompare::Less;
        case AlphaCompare::LessEqual: return AlphaCompare::Greater;
        case AlphaCompare::Greater: return AlphaCompare::LessEqual;
        case AlphaCompare::Equal: return AlphaCompare::NotEqual;
        case AlphaCompare::NotEqual: return AlphaCompare::Equal;
    }
    return c;
}

}  // namespace

// ---- Blend vocabulary ----------------------------------------------------------

BlendMode BlendMode::from_index(int index) {
    BlendMode mode;
    mode.c = static_cast<BlendFactorSel>(index % 3);
    index /= 3;
    mode.d = static_cast<BlendSel>(index % 3);
    index /= 3;
    mode.b = static_cast<BlendSel>(index % 3);
    index /= 3;
    mode.a = static_cast<BlendSel>(index % 3);
    return mode;
}

int BlendMode::index() const {
    return ((static_cast<int>(a) * 3 + static_cast<int>(b)) * 3 + static_cast<int>(d)) * 3 + static_cast<int>(c);
}

BlendGroup classify_blend(BlendMode mode) {
    if (mode.a == mode.b) {
        return BlendGroup::SameAsD;
    }
    int cs = 0;
    int cd = 0;
    auto add = [&](BlendSel s, int sign) {
        if (s == BlendSel::Cs) cs += sign;
        if (s == BlendSel::Cd) cd += sign;
    };
    add(mode.a, +1);
    add(mode.b, -1);
    const bool factor_ad = mode.c == BlendFactorSel::Ad;
    if (cs == 0) {
        return BlendGroup::CdMultiplied;  // M = (0, +-1)
    }
    if (cd == 0) {  // M = (+-1, 0)
        if (mode.d == BlendSel::Cd) return BlendGroup::Accumulate;
        if (factor_ad) {
            if (cs > 0 && mode.d == BlendSel::Cs) return BlendGroup::Impossible;  // Cs(1+Ad)
            return BlendGroup::SourceTimesAd;
        }
        return BlendGroup::NonRecursive;
    }
    // Mixed M: A = Cs, B = Cd (1, -1) or A = Cd, B = Cs (-1, 1).
    if (mode.d == BlendSel::Zero) return BlendGroup::Subtract;
    if (cs > 0 && mode.d == BlendSel::Cd) return BlendGroup::Interpolate;
    if (cs < 0 && mode.d == BlendSel::Cs) return BlendGroup::Interpolate;
    return BlendGroup::Impossible;
}

bool blend_group_reads_destination(BlendGroup group) {
    switch (group) {
        case BlendGroup::Accumulate:
        case BlendGroup::Interpolate:
        case BlendGroup::Subtract:
        case BlendGroup::CdMultiplied:
        case BlendGroup::Impossible:
        case BlendGroup::SourceTimesAd:
            return true;
        case BlendGroup::SameAsD:
        case BlendGroup::NonRecursive:
            return false;
    }
    return false;
}

const char* to_string(BlendGroup group) {
    switch (group) {
        case BlendGroup::SameAsD: return "same-as-D";
        case BlendGroup::NonRecursive: return "non-recursive";
        case BlendGroup::SourceTimesAd: return "source-times-Ad";
        case BlendGroup::Accumulate: return "accumulate";
        case BlendGroup::Interpolate: return "interpolate";
        case BlendGroup::Subtract: return "subtract";
        case BlendGroup::Impossible: return "impossible";
        case BlendGroup::CdMultiplied: return "Cd-multiplied";
    }
    return "unknown";
}

// ---- Trace and geometry -----------------------------------------------------------

VertexTrace compute_vertex_trace(const BatchGeometry& geometry, const RegisterSnapshot& regs) {
    VertexTrace trace;
    if (geometry.indices.empty()) {
        return trace;
    }
    bool first = true;
    double best_q_coordinate = std::numeric_limits<double>::infinity();
    for (std::uint32_t index : geometry.indices) {
        const Vertex2D& v = geometry.vertices[index];
        double u = 0.0;
        double w = 0.0;
        texel_coordinates(v, regs, u, w);
        if (!regs.fst) {
            if (!std::isfinite(u) || !std::isfinite(w)) {
                trace.coordinate_nan = true;
            }
            // Unsafe Q (chapter 02.9): the smallest Q-derived coordinate magnitude is huge.
            const double magnitude = std::max(std::fabs(u), std::fabs(w));
            best_q_coordinate = std::min(best_q_coordinate, magnitude);
        }
        if (first) {
            trace.x_min = trace.x_max = v.x;
            trace.y_min = trace.y_max = v.y;
            trace.z_min = trace.z_max = v.z;
            trace.fog_min = trace.fog_max = v.fog;
            trace.rgba_min = trace.rgba_max = v.rgba;
            trace.u_min = trace.u_max = u;
            trace.v_min = trace.v_max = w;
            trace.q_min = trace.q_max = v.q;
            first = false;
            continue;
        }
        trace.x_min = std::min(trace.x_min, v.x);
        trace.x_max = std::max(trace.x_max, v.x);
        trace.y_min = std::min(trace.y_min, v.y);
        trace.y_max = std::max(trace.y_max, v.y);
        trace.z_min = std::min(trace.z_min, v.z);
        trace.z_max = std::max(trace.z_max, v.z);
        trace.fog_min = std::min(trace.fog_min, v.fog);
        trace.fog_max = std::max(trace.fog_max, v.fog);
        for (int c = 0; c < 4; ++c) {
            trace.rgba_min[c] = std::min(trace.rgba_min[c], v.rgba[c]);
            trace.rgba_max[c] = std::max(trace.rgba_max[c], v.rgba[c]);
        }
        trace.u_min = std::min(trace.u_min, u);
        trace.u_max = std::max(trace.u_max, u);
        trace.v_min = std::min(trace.v_min, w);
        trace.v_max = std::max(trace.v_max, w);
        trace.q_min = std::min(trace.q_min, v.q);
        trace.q_max = std::max(trace.q_max, v.q);
    }
    for (int c = 0; c < 4; ++c) {
        if (trace.rgba_min[c] != trace.rgba_max[c]) {
            trace.colour_constant = false;
        }
    }
    trace.z_constant = trace.z_min == trace.z_max;
    trace.unsafe_q = !regs.fst && best_q_coordinate > 1e30;
    return trace;
}

PixelRect draw_rectangle_from_span(double min, double max) {
    PixelRect rect;
    rect.x0 = floor_to_int(min + 0.9);
    rect.x1 = floor_to_int(max + 0.6);
    if (rect.x1 <= rect.x0) {
        rect.x1 = rect.x0 + 1;
    }
    return rect;
}

PixelRect draw_rectangle(const VertexTrace& trace, const RegisterSnapshot& regs) {
    const PixelRect xs = draw_rectangle_from_span(trace.x_min, trace.x_max);
    const PixelRect ys = draw_rectangle_from_span(trace.y_min, trace.y_max);
    PixelRect rect{xs.x0, ys.x0, xs.x1, ys.x1};
    rect.x0 = std::max(rect.x0, regs.scax0);
    rect.x1 = std::min(rect.x1, regs.scax1 + 1);
    rect.y0 = std::max(rect.y0, regs.scay0);
    rect.y1 = std::min(rect.y1, regs.scay1 + 1);
    return rect;
}

PixelRect coverage_pixels_1x(double start, double end) {
    return PixelRect{ceil_to_int(start), 0, ceil_to_int(end), 0};
}

OverlapResult classify_overlap(const BatchGeometry& geometry, const VertexTrace& trace) {
    OverlapResult result;
    std::set<std::uint32_t> referenced(geometry.indices.begin(), geometry.indices.end());
    if (referenced.size() < 4) {
        result.kind = OverlapKind::None;
        result.groups = 1;
    } else if (geometry.prim == PrimClass::Sprite) {
        // Walk sprites in order; a sprite that intersects a box of the current group starts a new group.
        std::vector<std::vector<std::size_t>> groups;
        std::vector<Box> current;
        const std::size_t sprite_count = geometry.indices.size() / 2;
        for (std::size_t s = 0; s < sprite_count; ++s) {
            const Box box = sprite_box(geometry, s);
            bool hits = false;
            for (const Box& other : current) {
                if (box.intersects(other)) {
                    hits = true;
                    break;
                }
            }
            if (groups.empty() || hits) {
                groups.push_back({});
                current.clear();
            }
            groups.back().push_back(s);
            current.push_back(box);
        }
        result.groups = static_cast<std::uint32_t>(std::max<std::size_t>(groups.size(), 1));
        result.kind = result.groups > 1 ? OverlapKind::Yes : OverlapKind::None;
        for (const auto& members : groups) {
            DrawListGroup group;
            group.primitive_count = static_cast<std::uint32_t>(members.size());
            Box bounds = sprite_box(geometry, members.front());
            for (std::size_t s : members) {
                const Box b = sprite_box(geometry, s);
                bounds.x0 = std::min(bounds.x0, b.x0);
                bounds.y0 = std::min(bounds.y0, b.y0);
                bounds.x1 = std::max(bounds.x1, b.x1);
                bounds.y1 = std::max(bounds.y1, b.y1);
            }
            group.box = PixelRect{floor_to_int(bounds.x0) - 1, floor_to_int(bounds.y0) - 1,
                                  ceil_to_int(bounds.x1) + 1, ceil_to_int(bounds.y1) + 1};
            result.draw_list.push_back(group);
        }
    } else if (geometry.prim == PrimClass::Triangle) {
        result.kind = triangles_form_axis_quad(geometry) ? OverlapKind::None : OverlapKind::Unknown;
        result.groups = 1;
    } else {
        result.kind = OverlapKind::Unknown;
        result.groups = 1;
    }
    if (result.draw_list.empty()) {
        DrawListGroup group;
        group.primitive_count = static_cast<std::uint32_t>(geometry.indices.size());
        group.box = PixelRect{floor_to_int(trace.x_min) - 1, floor_to_int(trace.y_min) - 1,
                              ceil_to_int(trace.x_max) + 1, ceil_to_int(trace.y_max) + 1};
        result.draw_list.push_back(group);
    }
    return result;
}

// ---- Blend (chapter 08.2 to 08.5) --------------------------------------------------

BlendPlan decide_blend(const RegisterSnapshot& regs, const BatchInput& input, const Options& options,
                       std::uint8_t frag_alpha_min, std::uint8_t frag_alpha_max, bool cs_provably_zero,
                       OverlapKind overlap, std::uint32_t groups, bool sprite_batch, bool barrier_other) {
    BlendPlan plan;
    std::uint8_t fix = regs.fix;
    bool changed = false;
    plan.mode = simplify_blend(regs.blend, fix, regs, input.target, frag_alpha_min, frag_alpha_max,
                               cs_provably_zero, changed);
    plan.simplified = changed;
    plan.effective_fix = fix;
    plan.group = classify_blend(plan.mode);
    if (!regs.abe) {
        plan.method = BlendMethod::None;
        plan.reason = BlendReason::Disabled;
        plan.group = classify_blend(regs.blend);
        plan.mode = regs.blend;
        return plan;
    }

    const bool reads = blend_group_reads_destination(plan.group);
    const bool factor_ad = plan.mode.c == BlendFactorSel::Ad;
    const int bound = plan.mode.c == BlendFactorSel::Fix ? fix
                      : plan.mode.c == BlendFactorSel::As ? frag_alpha_max
                                                          : 255;
    const bool above_one = !factor_ad && bound > kAlphaHalf;
    plan.factor_above_one = above_one;
    const bool sixteen = regs.frame == FrameFormat::Ct16;
    const bool dither = regs.dthe && options.dither_enabled && sixteen;

    if (regs.pabe) return finish_blend(plan, BlendMethod::Shader, BlendReason::PabeShader, false);
    if (plan.group == BlendGroup::Impossible) {
        return finish_blend(plan, BlendMethod::Shader, BlendReason::ImpossibleShader, false);
    }
    if (plan.group == BlendGroup::SameAsD) {
        return finish_blend(plan, BlendMethod::FixedFunction, BlendReason::SameAsD, false);
    }
    if (plan.group == BlendGroup::NonRecursive) {
        return finish_blend(plan, BlendMethod::Shader, BlendReason::NonRecursiveShader, false);
    }
    if (plan.group == BlendGroup::Accumulate && !factor_ad && regs.colclamp && regs.frame != FrameFormat::Ct16) {
        return finish_blend(plan, BlendMethod::Accumulate, BlendReason::ExactAccumulate, false);
    }
    if (dither) return finish_blend(plan, BlendMethod::Shader, BlendReason::DitherShader, false);

    if (options.blend_level == BlendLevel::Full) {
        if (reads || factor_ad) return finish_blend(plan, BlendMethod::Shader, BlendReason::FullLevelShader, true);
        return finish_blend(plan, BlendMethod::FixedFunction, BlendReason::FixedFunctionApprox, false);
    }

    // basic and above: a barrier needed anyway makes the shader path free.
    if (barrier_other && (reads || factor_ad)) {
        return finish_blend(plan, BlendMethod::Shader, BlendReason::BarrierShader, true);
    }
    if (options.blend_level == BlendLevel::Basic) {
        if (plan.group == BlendGroup::Interpolate || plan.group == BlendGroup::Subtract) {
            if (!factor_ad && !above_one) {
                return finish_blend(plan, BlendMethod::Mixed, BlendReason::MixedBasic, true);
            }
        }
        return finish_blend(plan, BlendMethod::FixedFunction, BlendReason::FixedFunctionBasic, reads || factor_ad);
    }

    // medium and high: the basic rules plus the shader triggers of chapter 08.4.
    if (sixteen && reads) return finish_blend(plan, BlendMethod::Shader, BlendReason::Cd16Shader, true);
    if (sprite_batch && groups < 100 && (reads || factor_ad)) {
        return finish_blend(plan, BlendMethod::Shader, BlendReason::SpriteBatchShader, true);
    }
    if ((factor_ad || above_one) && overlap == OverlapKind::None) {
        return finish_blend(plan, BlendMethod::Shader, BlendReason::NoOverlapShader, true);
    }
    if (options.blend_level == BlendLevel::High) {
        if (above_one) return finish_blend(plan, BlendMethod::Shader, BlendReason::FactorAbove1Shader, true);
        if (factor_ad) return finish_blend(plan, BlendMethod::Shader, BlendReason::AdFactorShader, true);
        if (!regs.colclamp && reads && plan.group != BlendGroup::Interpolate) {
            return finish_blend(plan, BlendMethod::Shader, BlendReason::ColourWrapShader, true);
        }
    }
    return finish_blend(plan, BlendMethod::FixedFunction, BlendReason::FixedFunctionApprox, reads || factor_ad);
}

// ---- Frame write masks (chapter 08.9) ---------------------------------------------

ChannelWrites channel_writes(FrameFormat frame, std::uint32_t fbmsk) {
    std::array<std::uint32_t, 4> valid{};
    switch (frame) {
        case FrameFormat::Ct32: valid = {0xFF, 0xFF, 0xFF, 0xFF}; break;
        case FrameFormat::Ct24: valid = {0xFF, 0xFF, 0xFF, 0x00}; break;
        case FrameFormat::Ct16: valid = {0xF8, 0xF8, 0xF8, 0x80}; break;
    }
    ChannelWrites out;
    for (int c = 0; c < 4; ++c) {
        const std::uint32_t mask_byte = (fbmsk >> (8 * c)) & 0xFFu;
        const std::uint32_t writable = valid[c] & ~mask_byte & 0xFFu;
        if (valid[c] == 0 || writable == 0) {
            out.channel[c] = ChannelState::Off;
        } else if (writable == valid[c]) {
            out.channel[c] = ChannelState::Normal;
        } else {
            out.channel[c] = ChannelState::Partial;
            out.any_partial = true;
        }
    }
    for (int c = 0; c < 3; ++c) {
        out.rgb_any = out.rgb_any || out.channel[c] != ChannelState::Off;
    }
    out.alpha_any = out.channel[3] != ChannelState::Off;
    return out;
}

// ---- Alpha test (chapter 08.6) -------------------------------------------------------

AlphaFold fold_alpha_test(AlphaCompare compare, std::uint8_t min_alpha, std::uint8_t max_alpha, std::uint8_t reference) {
    const int lo = min_alpha;
    const int hi = max_alpha;
    const int r = reference;
    switch (compare) {
        case AlphaCompare::Never: return AlphaFold::AllFail;
        case AlphaCompare::Always: return AlphaFold::AllPass;
        case AlphaCompare::Less:
            if (hi < r) return AlphaFold::AllPass;
            if (lo >= r) return AlphaFold::AllFail;
            break;
        case AlphaCompare::LessEqual:
            if (hi <= r) return AlphaFold::AllPass;
            if (lo > r) return AlphaFold::AllFail;
            break;
        case AlphaCompare::Equal:
            if (lo == r && hi == r) return AlphaFold::AllPass;
            if (lo > r || hi < r) return AlphaFold::AllFail;
            break;
        case AlphaCompare::GreaterEqual:
            if (lo >= r) return AlphaFold::AllPass;
            if (hi < r) return AlphaFold::AllFail;
            break;
        case AlphaCompare::Greater:
            if (lo > r) return AlphaFold::AllPass;
            if (hi <= r) return AlphaFold::AllFail;
            break;
        case AlphaCompare::NotEqual:
            if (lo > r || hi < r) return AlphaFold::AllPass;
            if (lo == r && hi == r) return AlphaFold::AllFail;
            break;
    }
    return AlphaFold::Mixed;
}

WriteMask alpha_fail_writes(AlphaFail fail, FrameFormat frame) {
    switch (fail) {
        case AlphaFail::Keep: return WriteMask{false, false, false};
        case AlphaFail::FrameOnly: return WriteMask{true, true, false};
        case AlphaFail::DepthOnly: return WriteMask{false, false, true};
        case AlphaFail::RgbOnly:
            // RGB_ONLY writes RGB alone only in a 32-bit frame; elsewhere it behaves as FB_ONLY.
            if (frame == FrameFormat::Ct32) return WriteMask{true, false, false};
            return WriteMask{true, true, false};
    }
    return WriteMask{};
}

AlphaTestPlan decide_alpha_test(const RegisterSnapshot& regs, std::uint8_t frag_min, std::uint8_t frag_max,
                                WriteMask enabled, const AlphaTestContext& context) {
    AlphaTestPlan plan;
    plan.writes = enabled;
    if (!regs.ate) {
        plan.fold = AlphaFold::AllPass;
        plan.method = AlphaMethod::None;
        plan.passes.push_back(AlphaPass{AlphaCompare::Always, false, false, false, enabled});
        return plan;
    }
    plan.test_enabled = true;
    plan.fold = fold_alpha_test(regs.atst, frag_min, frag_max, regs.aref);

    if (plan.fold == AlphaFold::AllPass) {
        plan.test_enabled = false;
        plan.method = AlphaMethod::None;
        plan.passes.push_back(AlphaPass{AlphaCompare::Always, false, false, false, enabled});
        return plan;
    }
    if (plan.fold == AlphaFold::AllFail) {
        // Every fragment fails: AFAIL decides the channels, no test is needed.
        plan.test_enabled = false;
        plan.method = AlphaMethod::Never;
        plan.writes = intersect(enabled, alpha_fail_writes(regs.afail, regs.frame));
        plan.passes.push_back(AlphaPass{AlphaCompare::Never, false, false, false, plan.writes});
        return plan;
    }

    // Mixed: the test is needed. An AFAIL with no writable channel behaves as KEEP.
    AlphaFail fail = regs.afail;
    // RGB_ONLY behaves as FB_ONLY outside a 32-bit frame or with alpha masked (08.6).
    if (fail == AlphaFail::RgbOnly && (regs.frame != FrameFormat::Ct32 || !enabled.alpha)) {
        fail = AlphaFail::FrameOnly;
    }
    WriteMask fail_writes = intersect(enabled, alpha_fail_writes(fail, regs.frame));
    if (!fail_writes.any()) {
        fail = AlphaFail::Keep;
    }
    if (fail == AlphaFail::Keep) {
        plan.method = AlphaMethod::Keep;
        plan.passes.push_back(AlphaPass{regs.atst, true, false, false, enabled});
        return plan;
    }

    const bool depth_feedback_needed = enabled.depth && (fail == AlphaFail::FrameOnly || fail == AlphaFail::RgbOnly);
    const bool feedback_allowed = context.barrier_other || context.framebuffer_fetch || context.accurate_option;
    if (feedback_allowed && (!depth_feedback_needed || context.depth_feedback)) {
        plan.method = AlphaMethod::Feedback;
        plan.passes.push_back(AlphaPass{regs.atst, false, true, false, enabled});
        return plan;
    }

    const bool overlapping = context.overlap != OverlapKind::None;
    const bool z_independent = context.z_independent;
    const bool rgb_independent = !context.blend_uses_ad || !enabled.alpha || !overlapping;

    if (fail == AlphaFail::FrameOnly && z_independent) {
        plan.method = AlphaMethod::SplitFrameOnly;
        // Pass 1: colour for every fragment, no test, no depth.
        plan.passes.push_back(AlphaPass{AlphaCompare::Always, false, false, false, WriteMask{enabled.rgb, enabled.alpha, false}});
        // Pass 2: depth for passing fragments only.
        if (enabled.depth) {
            plan.passes.push_back(AlphaPass{regs.atst, true, false, false, WriteMask{false, false, true}});
        }
        return plan;
    }
    if (fail == AlphaFail::RgbOnly && z_independent && rgb_independent) {
        plan.method = AlphaMethod::SplitRgbOnly;
        // Pass 1: RGB for every fragment; alpha chosen per fragment through dual-source blending.
        plan.passes.push_back(AlphaPass{AlphaCompare::Always, false, false, true, WriteMask{enabled.rgb, enabled.alpha, false}});
        if (enabled.depth) {
            plan.passes.push_back(AlphaPass{regs.atst, true, false, false, WriteMask{false, false, true}});
        }
        return plan;
    }

    // PASS_THEN_FAIL: normal test with discard, then the inverted test for AFAIL's channels.
    plan.method = AlphaMethod::PassThenFail;
    if (enabled.any()) {
        plan.passes.push_back(AlphaPass{regs.atst, true, false, false, enabled});
    }
    plan.passes.push_back(AlphaPass{invert_compare(regs.atst), true, false, false, fail_writes});
    return plan;
}

// ---- DATE (chapter 08.7) -------------------------------------------------------------------

DatePlan decide_date(bool date, bool datm, FrameFormat frame, std::uint8_t target_min, std::uint8_t target_max,
                     const DateContext& context) {
    DatePlan plan;
    if (!date) {
        return plan;
    }
    if (frame == FrameFormat::Ct24) {
        // Nothing passes on hardware for CT24 with DATE set (chapter 01.5, V-01-8).
        plan.outcome = DateOutcome::SkipAllFail;
        return plan;
    }
    const bool overlapping = context.overlap != OverlapKind::None;
    const bool lowers = overlapping && context.alpha_written && context.written_min < kAlphaHalf;
    const bool raises = overlapping && context.alpha_written && context.written_max >= kAlphaHalf;
    if (!datm) {
        if (target_min >= kAlphaHalf && !lowers) {
            plan.outcome = DateOutcome::SkipAllFail;
            return plan;
        }
        if (target_max < kAlphaHalf && !raises) {
            plan.outcome = DateOutcome::NoTest;
            return plan;
        }
    } else {
        if (target_max < kAlphaHalf && !raises) {
            plan.outcome = DateOutcome::SkipAllFail;
            return plan;
        }
        if (target_min >= kAlphaHalf && !lowers) {
            plan.outcome = DateOutcome::NoTest;
            return plan;
        }
    }
    plan.outcome = DateOutcome::Test;
    plan.reads_green = context.shuffle;

    if (context.framebuffer_fetch || !overlapping || context.shuffle || context.complex_atest_overlap) {
        plan.method = DateMethod::Full;
        return plan;
    }
    const bool stencil_one = context.alpha_written &&
                             ((datm && context.written_max < kAlphaHalf) || (!datm && (context.written_min >= kAlphaHalf || context.fba)));
    if (stencil_one) {
        plan.method = DateMethod::StencilOne;
    } else if (!context.alpha_written) {
        plan.method = DateMethod::Stencil;
    } else if (context.draw_list_groups < 10 || context.index_count < 30 || context.full_barrier_needed) {
        plan.method = DateMethod::Full;
    } else if (context.prim_id_available) {
        plan.method = DateMethod::PrimId;
    } else {
        plan.method = DateMethod::Full;
    }
    (void)frame;
    return plan;
}

// ---- Barriers and colour wrap (chapter 08.5, 08.8) ----------------------------------

BarrierKind decide_barrier(bool reads_target, bool writes_target, OverlapKind overlap, std::uint32_t groups,
                           bool shuffle) {
    if (!reads_target) return BarrierKind::None;
    if (!writes_target) return BarrierKind::One;
    if (shuffle) return BarrierKind::One;
    if (overlap == OverlapKind::Unknown) return BarrierKind::PerPrimitive;
    if (overlap == OverlapKind::None || groups <= 1) return BarrierKind::One;
    return BarrierKind::PerDrawListGroup;
}

ColourWrapPlan decide_colour_wrap(bool colclamp, const BlendPlan& blend, FrameFormat frame, BlendLevel level,
                                  OverlapKind overlap, bool overflow_possible) {
    (void)frame;
    (void)overlap;
    if (colclamp || blend.method == BlendMethod::None) return ColourWrapPlan::NotNeeded;
    if (blend.method == BlendMethod::Shader) return ColourWrapPlan::ShaderFullResult;
    if (!overflow_possible) return ColourWrapPlan::NotNeeded;
    if (level == BlendLevel::Basic || level == BlendLevel::Medium) return ColourWrapPlan::SideTarget;
    return ColourWrapPlan::ShaderFullResult;
}

bool scanmsk_discards_row(std::uint32_t scanmsk, int gpu_y, std::uint32_t scale) {
    if (scale == 0) {
        return false;
    }
    const int row = gpu_y >= 0 ? gpu_y / static_cast<int>(scale)
                               : -((-gpu_y + static_cast<int>(scale) - 1) / static_cast<int>(scale));
    if (scanmsk == 2) return (row & 1) == 0;
    if (scanmsk == 3) return (row & 1) == 1;
    return false;
}

PrimitivePlan decide_primitive(PrimClass prim, std::uint32_t scale, bool iip_gouraud, bool uniform_colour,
                               bool unsafe_q, const DeviceCaps& caps) {
    PrimitivePlan plan;
    plan.prim = prim;
    switch (prim) {
        case PrimClass::Point:
            if (scale <= 1) {
                plan.expansion = Expansion::None;
            } else {
                plan.expansion = static_cast<float>(scale) <= caps.max_point_size ? Expansion::PointSize : Expansion::PointQuad;
            }
            break;
        case PrimClass::Line:
            if (scale <= 1) {
                plan.expansion = Expansion::None;
            } else if (caps.wide_lines && static_cast<float>(scale) <= caps.max_line_width) {
                plan.expansion = Expansion::LineWidth;
            } else {
                plan.expansion = Expansion::LineQuad;
            }
            break;
        case PrimClass::Triangle:
            plan.expansion = Expansion::TriangleList;
            break;
        case PrimClass::Sprite:
            plan.expansion = unsafe_q ? Expansion::SpriteCpu : Expansion::SpriteVertexStage;
            plan.needs_unsafe_q_cpu = unsafe_q;
            break;
    }
    const bool flat_candidate = !iip_gouraud && (prim == PrimClass::Line || prim == PrimClass::Triangle);
    plan.flat_deindex = flat_candidate && !caps.provoking_vertex_last && !uniform_colour;
    return plan;
}

// ---- Top-level decision (chapter 03.1) -------------------------------------------------------

DrawPlan decide_draw(const BatchInput& input, const DeviceCaps& caps, const Options& options) {
    DrawPlan plan;
    const RegisterSnapshot& r = input.regs;
    const BatchGeometry& g = input.geometry;

    // Early outs (3.1 steps 2 and 3).
    if (g.prim == PrimClass::Point && g.indices.size() == 1) {
        const Vertex2D& v = g.vertices[g.indices[0]];
        const bool black = v.rgba[0] == 0 && v.rgba[1] == 0 && v.rgba[2] == 0 && v.rgba[3] == 0;
        if (v.x == 0.0 && v.y == 0.0 && black && !input.current_prim_is_point) {
            plan.skip = SkipReason::ResetArtefact;
            return plan;
        }
    }
    if (r.date && r.frame == FrameFormat::Ct24) {
        plan.skip = SkipReason::DateOnCt24;
        return plan;
    }

    plan.trace = compute_vertex_trace(g, r);
    plan.overlap = classify_overlap(g, plan.trace);
    const bool sprite_batch = is_sprite_like(g.prim);
    const std::uint32_t groups = input.draw_list_groups_override != 0
                                     ? input.draw_list_groups_override
                                     : plan.overlap.groups;

    // Write masks (3.1 step 4).
    plan.colour_channels = channel_writes(r.frame, r.fbmsk);
    const bool z_written = !(r.zmsk || !r.zte || (r.ate && r.atst == AlphaCompare::Never && r.afail != AlphaFail::DepthOnly));
    WriteMask enabled{plan.colour_channels.rgb_any, plan.colour_channels.alpha_any, z_written};

    // AA1 without coverage emulation: constant alpha 128 (3.1 step 7).
    std::uint8_t frag_min = input.fragment_alpha_min;
    std::uint8_t frag_max = input.fragment_alpha_max;
    if (r.aa1 && (g.prim == PrimClass::Line || g.prim == PrimClass::Triangle)) {
        frag_min = frag_max = kAlphaHalf;
    }

    // Draw rectangle (3.1 step 9).
    plan.draw_rect = draw_rectangle(plan.trace, r);
    if (plan.draw_rect.empty()) {
        plan.skip = SkipReason::EmptyRect;
        return plan;
    }

    // Texture needed (3.1 step 10).
    const bool vertex_colour_nonzero = plan.trace.rgba_max[0] != 0 || plan.trace.rgba_max[1] != 0 ||
                                       plan.trace.rgba_max[2] != 0;
    const bool colour_from_texture = plan.colour_channels.rgb_any &&
                                     (r.tfx == Tfx::Decal || vertex_colour_nonzero);
    const bool alpha_from_texture = plan.colour_channels.alpha_any || r.ate || r.abe;
    plan.texture_used = r.tme && (colour_from_texture || alpha_from_texture);

    // Clears (3.1 step 11, chapter 05.8).
    PixelRect clear_rect{};
    std::uint32_t clear_word = 0;
    std::uint32_t clear_base = 0;
    // Pair clears first: a draw that writes two buffers is not a plain target clear.
    ClearKind clear = detect_double_half_clear(input, clear_rect, clear_base);
    if (clear == ClearKind::None) clear = detect_striped_double_clear(input, clear_rect, clear_base);
    if (clear == ClearKind::None) clear = detect_target_clear(input, caps, clear_rect, clear_word);
    if (clear != ClearKind::None) {
        plan.clear = clear;
        plan.clear_rect = clear_rect;
        plan.clear_colour_word = clear_word;
        plan.clear_depth_word = input.geometry.vertices.empty() ? 0 : input.geometry.vertices[0].z;
        plan.clear_base_fbp = clear_base;
        plan.draw_converted_to_clear = true;
        plan.skip = SkipReason::ConvertedToClear;
        return plan;
    }

    // Texture shuffle (3.1 step 12, chapter 09.2) from the first two vertices of a sprite batch.
    if (r.tme && sprite_batch && g.indices.size() >= 2) {
        const Vertex2D& a = g.vertices[g.indices[0]];
        const Vertex2D& b = g.vertices[g.indices[1]];
        double au = 0.0, aw = 0.0, bu = 0.0, bw = 0.0;
        texel_coordinates(a, r, au, aw);
        texel_coordinates(b, r, bu, bw);
        ShuffleSample sample;
        sample.frame_x0 = a.x;
        sample.frame_x1 = b.x;
        sample.tex_u0 = au;
        sample.tex_u1 = bu;
        sample.frame_y0 = a.y;
        sample.frame_y1 = b.y;
        sample.tex_v0 = aw;
        sample.tex_v1 = bw;
        sample.fbmsk = r.fbmsk;
        sample.frame_16bit = r.frame == FrameFormat::Ct16;
        sample.sprite_or_quads = true;
        sample.texture_is_frame = input.texture.is_frame_target;
        sample.opaque = input.texture.opaque;
        sample.nearest = true;
        sample.width_pixels = static_cast<std::uint32_t>(std::max(0.0, b.x - a.x));
        sample.tiny_draw = plan.draw_rect.x1 - plan.draw_rect.x0 <= 8 && plan.draw_rect.y1 - plan.draw_rect.y0 <= 8;
        sample.source_is_32bit = input.texture.src_is_32bit_target;
        sample.real_16bit_source = input.texture.real_16bit_source;
        plan.texture_shuffle = classify_texture_shuffle(sample);
        plan.texture_shuffle_candidate = plan.texture_shuffle != ShuffleKind::None;
    }

    // Alpha test (3.1 step 6, chapter 08.6).
    AlphaTestContext atest_context;
    atest_context.overlap = plan.overlap.kind;
    atest_context.blend_uses_ad = r.blend.c == BlendFactorSel::Ad;
    atest_context.z_independent = (r.ztst == DepthCompare::GreaterEqual && plan.trace.z_constant) ||
                                  r.ztst == DepthCompare::Always || !enabled.depth ||
                                  plan.overlap.kind == OverlapKind::None;
    atest_context.barrier_other = input.texture.is_frame_target || plan.colour_channels.any_partial;
    atest_context.framebuffer_fetch = caps.framebuffer_fetch();
    atest_context.depth_feedback = caps.depth_feedback;
    atest_context.accurate_option = options.accurate_atest;
    plan.alpha_test = decide_alpha_test(r, frag_min, frag_max, enabled, atest_context);
    plan.writes = plan.alpha_test.writes;
    plan.z_written = plan.writes.depth;

    // Nothing written (3.1 step 8).
    const bool no_colour = !r.date && !plan.writes.rgb && !plan.writes.alpha;
    if (no_colour && !plan.writes.depth) {
        plan.skip = SkipReason::NothingWritten;
        return plan;
    }
    plan.depth_test_always = r.ztst == DepthCompare::Always || !plan.writes.depth || !r.zte;

    // DATE (chapter 08.7).
    DateContext date_context;
    date_context.overlap = plan.overlap.kind;
    date_context.alpha_written = plan.writes.alpha;
    date_context.written_min = r.fba ? static_cast<std::uint8_t>(frag_min | kAlphaHalf) : frag_min;
    date_context.written_max = r.fba ? static_cast<std::uint8_t>(frag_max | kAlphaHalf) : frag_max;
    date_context.fba = r.fba;
    date_context.shuffle = plan.texture_shuffle != ShuffleKind::None;
    const bool atest_complex = plan.alpha_test.method == AlphaMethod::Feedback ||
                               plan.alpha_test.method == AlphaMethod::SplitFrameOnly ||
                               plan.alpha_test.method == AlphaMethod::SplitRgbOnly ||
                               plan.alpha_test.method == AlphaMethod::PassThenFail;
    date_context.complex_atest_overlap = atest_complex && plan.overlap.kind != OverlapKind::None;
    date_context.framebuffer_fetch = caps.framebuffer_fetch();
    date_context.draw_list_groups = groups;
    date_context.index_count = static_cast<std::uint32_t>(g.indices.size());
    date_context.full_barrier_needed = plan.overlap.kind != OverlapKind::None && groups > 1;
    date_context.prim_id_available = options.prim_id_available;
    plan.date_enabled = r.date;
    plan.date = decide_date(r.date, r.datm, r.frame, input.target.alpha_min, input.target.alpha_max, date_context);
    if (plan.date.outcome == DateOutcome::SkipAllFail) {
        // Every fragment fails DATE: nothing is written, depth included.
        plan.skip = SkipReason::NothingWritten;
        return plan;
    }

    // Blend (chapter 08.2 to 08.4).
    const bool vertex_rgb_zero = plan.trace.rgba_max[0] == 0 && plan.trace.rgba_max[1] == 0 &&
                                 plan.trace.rgba_max[2] == 0;
    const bool cs_zero = vertex_rgb_zero && !r.fge && r.tfx != Tfx::Decal &&
                         (!r.tme || r.tfx == Tfx::Modulate || plan.trace.rgba_max[3] == 0);
    const bool barrier_other = input.texture.is_frame_target || plan.colour_channels.any_partial ||
                               (plan.date.outcome == DateOutcome::Test && plan.date.method == DateMethod::Full) ||
                               plan.alpha_test.method == AlphaMethod::Feedback;
    plan.blend = decide_blend(r, input, options, frag_min, frag_max, cs_zero, plan.overlap.kind, groups,
                              sprite_batch, barrier_other);

    // Barrier, software FBMASK and colour wrap (chapter 08.5 to 08.9).
    plan.software_fbmask = plan.colour_channels.any_partial && (plan.writes.rgb || plan.writes.alpha);
    plan.reads_target = plan.blend.reads_destination || barrier_other || plan.date.outcome == DateOutcome::Test;
    plan.writes_target = plan.writes.rgb || plan.writes.alpha || plan.writes.depth;
    plan.barrier = decide_barrier(plan.reads_target, plan.writes_target, plan.overlap.kind, groups,
                                  plan.texture_shuffle != ShuffleKind::None);
    const bool overflow_possible = !r.colclamp && plan.blend.reads_destination &&
                                   plan.blend.group != BlendGroup::Interpolate;
    plan.colour_wrap = decide_colour_wrap(r.colclamp, plan.blend, r.frame, options.blend_level, plan.overlap.kind,
                                          overflow_possible);
    plan.dither = r.dthe && options.dither_enabled && r.frame == FrameFormat::Ct16;

    // Primitive expansion (3.2) and the pass order (3.10).
    const bool uniform = plan.trace.colour_constant;
    plan.primitive = decide_primitive(g.prim, options.scale, r.iip, uniform,
                                      plan.trace.unsafe_q, caps);

    if (input.texture.is_frame_target && !caps.framebuffer_fetch()) plan.passes.push_back(PassKind::HazardCopy);
    if (plan.date.method == DateMethod::StencilOne) plan.passes.push_back(PassKind::StencilClear);
    if (plan.date.method == DateMethod::Stencil || plan.date.method == DateMethod::PrimId) {
        plan.passes.push_back(PassKind::DatePrepare);
    }
    if (plan.colour_wrap == ColourWrapPlan::SideTarget) plan.passes.push_back(PassKind::ColourWrapConvert);
    plan.passes.push_back(PassKind::MainDraw);
    if (plan.alpha_test.passes.size() > 1) plan.passes.push_back(PassKind::AlphaSecondPass);
    if (plan.colour_wrap == ColourWrapPlan::SideTarget) plan.passes.push_back(PassKind::ColourWrapResolveAfter);

    plan.key = make_pipeline_key(plan, r, caps);
    return plan;
}

}  // namespace coregs::decide
