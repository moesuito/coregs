// SPDX-License-Identifier: MIT
//
// HWR display output: CPU reference for the PCRTC output stage (TASK-027, M6 and M15).
// Copyright (c) 2026 GT4Recomp contributors.
//
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 13.1 to 13.6
// and 01.7 (clean room, decision 0069). Choices where the specification is silent are
// recorded in docs/hw-renderer-spec/IMPLEMENTATION-NOTES.md (IN-xx) and cited below.

#include "hw_gs_display.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace coregs::display {

namespace {

constexpr std::uint64_t field_mask(unsigned width) {
    return (std::uint64_t{1} << width) - 1u;
}

constexpr std::uint64_t bits_at(std::uint64_t reg, unsigned low, unsigned width) {
    return (reg >> low) & field_mask(width);
}

constexpr std::int64_t kFramebufferWrap = 2048;

std::uint8_t clamp_byte(double value) {
    return static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

std::uint8_t average_byte(std::uint8_t a, std::uint8_t b) {
    return static_cast<std::uint8_t>((static_cast<int>(a) + static_cast<int>(b) + 1) / 2);
}

// Family of a PSM for "same source" (IN-33).
std::uint32_t psm_family(std::uint32_t psm) {
    switch (psm) {
        case 0x00u:  // CT32
        case 0x01u:  // CT24
            return 0u;
        case 0x30u:  // Z32
        case 0x31u:  // Z24
            return 1u;
        case 0x02u:  // CT16
        case 0x0Au:  // CT16S
            return 2u;
        case 0x32u:  // Z16
        case 0x3Au:  // Z16S
            return 3u;
        default:
            return 100u + psm;
    }
}

// IN-396: how far the largest R, G or B difference of two pixels lies above the 13.4 threshold, in 1/255000 units
// (difference * 1000 - threshold_permille * 255): positive = moved, zero or negative = within the threshold. Integer
// arithmetic, the same formula as hw_deinterlace_scaled.comp, so the CPU and GPU deinterlacers decide alike.
std::int64_t channel_excess(const Rgba8& a, const Rgba8& b, std::int64_t threshold_permille) {
    const auto delta = [](std::uint8_t x, std::uint8_t y) {
        return std::abs(static_cast<std::int64_t>(x) - static_cast<std::int64_t>(y));
    };
    const std::int64_t largest = std::max({delta(a.r, b.r), delta(a.g, b.g), delta(a.b, b.b)});
    return largest * 1000 - threshold_permille * 255;
}

// Integer division truncating toward zero (C++ semantics), made explicit for readers.
std::int64_t divide_toward_zero(std::int64_t numerator, std::int64_t denominator) {
    return numerator / denominator;
}

}  // namespace

// ---------------------------------------------------------------------------------------
// 13.1 Video mode
// ---------------------------------------------------------------------------------------

VideoModeTable video_mode_table(VideoMode mode) {
    switch (mode) {
        case VideoMode::Ntsc:
            return VideoModeTable{640, 224, 642, 25, 4, 1, true};
        case VideoMode::Pal:
            return VideoModeTable{640, 256, 676, 36, 4, 1, true};
        case VideoMode::Vesa:
            return VideoModeTable{640, 480, 276, 34, 2, 1, false};
        case VideoMode::Progressive480:
            return VideoModeTable{720, 480, 232, 35, 2, 1, false};
        case VideoMode::Progressive720:
            return VideoModeTable{1280, 720, 302, 24, 1, 1, false};
        case VideoMode::Interlaced1080:
            return VideoModeTable{1920, 540, 238, 40, 1, 1, true};
        case VideoMode::None:
            break;
    }
    return VideoModeTable{};
}

VideoMode detect_video_mode(std::uint64_t smode1, std::uint64_t smode2) {
    const std::uint64_t cmod = bits_at(smode1, 13, 2);
    const std::uint64_t lc = bits_at(smode1, 3, 7);
    const bool interlace_bit = (smode2 & 1u) != 0;
    switch (cmod) {
        case 2u:
            return VideoMode::Ntsc;
        case 3u:
            return VideoMode::Pal;
        case 0u:
            if (interlace_bit && lc == 22u) {
                return VideoMode::Interlaced1080;
            }
            if (!interlace_bit && lc == 22u) {
                return VideoMode::Progressive720;
            }
            if (!interlace_bit && lc == 32u) {
                return VideoMode::Progressive480;
            }
            return VideoMode::Vesa;
        default:
            // CMOD = 1 is not in the table of 13.1 (IN-22): no known mode.
            return VideoMode::None;
    }
}

bool interlaced_flag(VideoMode mode, std::uint64_t smode2) {
    return video_mode_table(mode).analogue && (smode2 & 1u) != 0;
}

bool really_interlaced(VideoMode mode, std::uint64_t syncv) {
    return video_mode_table(mode).analogue && (bits_at(syncv, 0, 1) != 0);
}

// ---------------------------------------------------------------------------------------
// 13.2 Circuit rectangles and offsets
// ---------------------------------------------------------------------------------------

PmodeFields decode_pmode(std::uint64_t pmode) {
    PmodeFields fields;
    fields.en1 = bits_at(pmode, 0, 1) != 0;
    fields.en2 = bits_at(pmode, 1, 1) != 0;
    fields.mmod = bits_at(pmode, 5, 1) != 0;
    fields.slbg = bits_at(pmode, 7, 1) != 0;
    fields.alp = static_cast<std::uint32_t>(bits_at(pmode, 8, 8));
    return fields;
}

CircuitFields decode_circuit(const CircuitRegisters& registers) {
    CircuitFields fields;
    const std::uint64_t dispfb = registers.dispfb;
    fields.fbp = static_cast<std::uint32_t>(bits_at(dispfb, 0, 9));
    fields.fbw = static_cast<std::uint32_t>(bits_at(dispfb, 9, 6));
    fields.psm = static_cast<std::uint32_t>(bits_at(dispfb, 15, 5));
    fields.dbx = static_cast<std::uint32_t>(bits_at(dispfb, 32, 11));
    fields.dby = static_cast<std::uint32_t>(bits_at(dispfb, 43, 11));

    const std::uint64_t display = registers.display;
    fields.dx = static_cast<std::uint32_t>(bits_at(display, 0, 12));
    fields.dy = static_cast<std::uint32_t>(bits_at(display, 12, 11));
    fields.magh = static_cast<std::uint32_t>(bits_at(display, 23, 4));
    fields.magv = static_cast<std::uint32_t>(bits_at(display, 27, 2));
    fields.dw = static_cast<std::uint32_t>(bits_at(display, 32, 12));
    fields.dh = static_cast<std::uint32_t>(bits_at(display, 44, 11));
    return fields;
}

DisplayLayout compute_layout(const DisplayRegisters& registers,
                             std::array<std::uint32_t, 2>& previous_dby) {
    DisplayLayout layout;
    layout.mode = detect_video_mode(registers.smode1, registers.smode2);
    if (layout.mode == VideoMode::None) {
        // No known mode: no image this VBlank (12.4). The DBY history is left as it was.
        return layout;
    }
    layout.valid = true;

    const VideoModeTable table = video_mode_table(layout.mode);
    layout.interlaced = interlaced_flag(layout.mode, registers.smode2);
    layout.really_interlaced = really_interlaced(layout.mode, registers.syncv);
    layout.ffmd = bits_at(registers.smode2, 1, 1) != 0;
    const PmodeFields pmode = decode_pmode(registers.pmode);
    const bool pmode_enabled[2] = {pmode.en1, pmode.en2};

    // Step 1 and 2 of 13.2 per circuit.
    for (std::size_t i = 0; i < 2; ++i) {
        const CircuitFields fields = decode_circuit(registers.circuits[i]);
        CircuitRect& rect = layout.circuits[i];
        rect.fbp = fields.fbp;
        rect.fbw = fields.fbw;
        rect.psm = fields.psm;
        rect.dx = static_cast<std::int32_t>(fields.dx);
        rect.dy = static_cast<std::int32_t>(fields.dy);
        rect.dby_changed = fields.dby != previous_dby[i];
        previous_dby[i] = fields.dby;

        bool enabled = pmode_enabled[i];
        if (fields.fbw == 0u && fields.dw == 0u && fields.dh == 0u && fields.magh == 0u) {
            enabled = false;  // rule 2 of 13.2
        }
        rect.enabled = enabled;
        if (!enabled) {
            continue;
        }

        // Rule 3: render size, then clamped to the mode's dividers.
        const std::uint64_t render_width = (fields.dw + 1u) / (fields.magh + 1u);
        const std::uint64_t render_height = (fields.dh + 1u) / (fields.magv + 1u);
        const std::uint64_t final_width = std::min<std::uint64_t>(render_width, (fields.dw + 1u) / table.h_divider);
        const std::uint64_t final_height = std::min<std::uint64_t>(render_height, (fields.dh + 1u) / table.v_divider);
        rect.display_width = static_cast<std::uint32_t>(final_width);
        rect.display_height = static_cast<std::uint32_t>(final_height);

        // Rule 4: read rectangle. FFMD = 1 on an interlaced signal reads one field.
        rect.read_width = rect.display_width;
        rect.read_height = (layout.interlaced && layout.ffmd) ? (rect.display_height + 1u) >> 1
                                                              : rect.display_height;

        // Rule 5 and the framebuffer offsets of 13.2 (IN-24: the subtraction is applied
        // as written, so a wrapped origin is negative and the reader applies it modulo 2048).
        std::int64_t dbx = fields.dbx;
        std::int64_t dby = fields.dby;
        if (dbx + static_cast<std::int64_t>(rect.read_width) > kFramebufferWrap) {
            dbx -= kFramebufferWrap;
        }
        if (dby + static_cast<std::int64_t>(rect.read_height) > kFramebufferWrap) {
            dby -= kFramebufferWrap;
        }
        rect.dbx = static_cast<std::int32_t>(dbx);
        rect.dby = static_cast<std::int32_t>(dby);
    }

    layout.both_enabled = layout.circuits[0].enabled && layout.circuits[1].enabled;
    const bool anti_blur = registers.anti_blur;
    CircuitRect& first = layout.circuits[0];
    CircuitRect& second = layout.circuits[1];

    // Anti-blur (13.2): horizontal, then vertical framebuffer rows, then display DY (a).
    if (layout.both_enabled && anti_blur) {
        if (std::llabs(static_cast<long long>(second.dbx) - first.dbx) == 1 && first.dx == second.dx) {
            const std::int32_t smaller = std::min(first.dbx, second.dbx);
            first.dbx = smaller;
            second.dbx = smaller;
        }
        if (std::llabs(static_cast<long long>(second.dby) - first.dby) <= 1) {
            const bool first_changed = first.dby_changed;
            const bool second_changed = second.dby_changed;
            if (first_changed != second_changed) {
                // The circuit that alternates moves onto the other one.
                CircuitRect& alternating = first_changed ? first : second;
                const CircuitRect& other = first_changed ? second : first;
                alternating.dby = other.dby;
            } else {
                // Otherwise the larger origin moves onto the smaller one.
                CircuitRect& larger = first.dby > second.dby ? first : second;
                const CircuitRect& smaller = first.dby > second.dby ? second : first;
                larger.dby = smaller.dby;
            }
        }
        if (!registers.scanmsk_active && std::llabs(static_cast<long long>(second.dy) - first.dy) < 4) {
            CircuitRect& larger = first.dy > second.dy ? first : second;
            const CircuitRect& smaller = first.dy > second.dy ? second : first;
            larger.dy = smaller.dy;
        }
    }

    // Offsets off (13.2): line offsets (b), relative x offset (c), global shift (d).
    const std::int64_t base_factor = static_cast<std::int64_t>(layout.interlaced ? 2 : 1);
    std::array<std::int64_t, 2> parity{0, 0};
    std::array<bool, 2> parity_valid{false, false};
    for (std::size_t i = 0; i < 2; ++i) {
        CircuitRect& rect = layout.circuits[i];
        if (!rect.enabled) {
            continue;
        }
        const VideoModeTable& t = table;
        const std::int64_t line_offset = divide_toward_zero(
            static_cast<std::int64_t>(rect.dy) - static_cast<std::int64_t>(t.dy_base) * base_factor,
            t.v_divider);
        // IN-26: a line offset in 0..4 adds its low bit to the display row.
        if (line_offset >= 0 && line_offset <= 4) {
            parity[i] = line_offset & 1;
            parity_valid[i] = true;
        }
        rect.screen_x = 0;
        rect.screen_y = static_cast<std::int32_t>(parity[i]);
    }
    if (layout.both_enabled) {
        const std::int64_t relative_x = divide_toward_zero(
            static_cast<std::int64_t>(second.dx) - first.dx, table.h_divider);
        if (std::llabs(static_cast<long long>(relative_x)) >= 4 || !anti_blur || registers.scanmsk_active) {
            second.screen_x = static_cast<std::int32_t>(relative_x);
        }
    }
    const std::size_t reference = first.enabled ? 0u : 1u;
    CircuitRect& reference_rect = layout.circuits[reference];
    std::int64_t global_shift = divide_toward_zero(
        static_cast<std::int64_t>(reference_rect.dy) - static_cast<std::int64_t>(table.dy_base) * base_factor,
        table.v_divider);
    if (global_shift != 0 && std::llabs(static_cast<long long>(global_shift)) <= 4) {
        // IN-27: the reference circuit's parity bit is taken off the shift.
        global_shift -= parity_valid[reference] ? parity[reference] : 0;
        for (CircuitRect& rect : layout.circuits) {
            if (rect.enabled) {
                rect.screen_y += static_cast<std::int32_t>(global_shift);
            }
        }
    }

    // Union of the enabled display rectangles, normalised to its origin (IN-29).
    // IN-29: the origin stays at (0, 0) unless a circuit sits at a negative offset, so a
    // one-line parity offset of a single circuit is kept on screen.
    std::int64_t origin_x = 0;
    std::int64_t origin_y = 0;
    std::int64_t max_x = std::numeric_limits<std::int64_t>::min();
    std::int64_t max_y = std::numeric_limits<std::int64_t>::min();
    bool any_enabled = false;
    for (const CircuitRect& rect : layout.circuits) {
        if (!rect.enabled) {
            continue;
        }
        any_enabled = true;
        origin_x = std::min<std::int64_t>(origin_x, rect.screen_x);
        origin_y = std::min<std::int64_t>(origin_y, rect.screen_y);
        max_x = std::max<std::int64_t>(max_x, rect.screen_x + static_cast<std::int64_t>(rect.display_width));
        max_y = std::max<std::int64_t>(max_y, rect.screen_y + static_cast<std::int64_t>(rect.display_height));
    }
    if (any_enabled) {
        for (CircuitRect& rect : layout.circuits) {
            if (rect.enabled) {
                rect.screen_x = static_cast<std::int32_t>(rect.screen_x - origin_x);
                rect.screen_y = static_cast<std::int32_t>(rect.screen_y - origin_y);
            }
        }
        layout.union_width = static_cast<std::uint32_t>(max_x - origin_x);
        layout.union_height = static_cast<std::uint32_t>(max_y - origin_y);
    }

    // IN-30: "changed" counts both circuits, enabled or not.
    int changed = 0;
    for (const CircuitRect& rect : layout.circuits) {
        changed += rect.dby_changed ? 1 : 0;
    }
    layout.one_circuit_dby_changed = changed == 1;
    layout.any_dby_changed = changed > 0;
    return layout;
}

// ---------------------------------------------------------------------------------------
// 13.3 Merge
// ---------------------------------------------------------------------------------------

OutputSize merge_output_size(const DisplayLayout& layout, bool deinterlace_off, std::uint32_t scale) {
    const VideoModeTable table = video_mode_table(layout.mode);
    // Full height when interlaced, toggling or deinterlace off; analogue modes double their
    // field height then (13.3). Progressive modes keep their table height.
    const bool full_height = layout.interlaced || layout.any_dby_changed || deinterlace_off;
    const std::uint32_t table_height = table.field_height * ((full_height && table.analogue) ? 2u : 1u);

    OutputSize size;
    const bool any_enabled = layout.circuits[0].enabled || layout.circuits[1].enabled;
    if (!any_enabled) {
        size.width = table.width;
        size.height = table_height;
    } else {
        size.width = std::min(layout.union_width, table.width);
        size.height = std::min(layout.union_height, table_height);
    }
    size.width *= scale;
    size.height *= scale;
    return size;
}

bool same_source(const CircuitRect& a, const CircuitRect& b) {
    return a.fbp == b.fbp && a.fbw == b.fbw && psm_family(a.psm) == psm_family(b.psm);
}

MergePlan plan_merge(const PmodeFields& pmode, const CircuitRect& rc1, const CircuitRect& rc2,
                     bool source1_available, bool source2_available) {
    MergePlan plan;
    if ((rc1.enabled && !source1_available) || (rc2.enabled && !source2_available)) {
        plan.outcome = MergeOutcome::KeepPrevious;
        return plan;
    }
    if (!rc1.enabled && !rc2.enabled) {
        plan.outcome = MergeOutcome::BackgroundOnly;
        return plan;
    }

    // RC1 = EN1 and not (MMOD = 1 and ALP = 0).
    plan.use_rc1 = rc1.enabled && !(pmode.mmod && pmode.alp == 0u);

    // RC2 = EN2 and SLBG = 0 and not (RC1 covers RC2 with MMOD = 1 and ALP = 255).
    bool rc1_covers_rc2 = false;
    if (rc1.enabled && rc2.enabled) {
        rc1_covers_rc2 = rc1.screen_x <= rc2.screen_x && rc1.screen_y <= rc2.screen_y &&
                         rc1.screen_x + static_cast<std::int32_t>(rc1.display_width) >=
                             rc2.screen_x + static_cast<std::int32_t>(rc2.display_width) &&
                         rc1.screen_y + static_cast<std::int32_t>(rc1.display_height) >=
                             rc2.screen_y + static_cast<std::int32_t>(rc2.display_height);
    }
    plan.use_rc2 = rc2.enabled && !pmode.slbg && !(rc1_covers_rc2 && pmode.mmod && pmode.alp == 255u);

    if (!plan.use_rc1 && !plan.use_rc2) {
        plan.outcome = MergeOutcome::BackgroundOnly;
        return plan;
    }

    // IN-34: identical circuits drop RC1.
    if (plan.use_rc1 && plan.use_rc2 && same_source(rc1, rc2) && rc1.fbp == rc2.fbp &&
        rc1.dbx == rc2.dbx && rc1.dby == rc2.dby && rc1.display_width == rc2.display_width &&
        rc1.display_height == rc2.display_height && rc1.screen_x == rc2.screen_x &&
        rc1.screen_y == rc2.screen_y) {
        plan.use_rc1 = false;
    }

    plan.single_read = plan.use_rc1 && plan.use_rc2 && same_source(rc1, rc2);
    plan.outcome = MergeOutcome::Composite;
    return plan;
}

std::int32_t interlace_offset_rows(bool really_interlaced, bool ffmd, DeinterlaceMode effective,
                                   std::uint32_t scale, int field_phase) {
    if (!really_interlaced || !ffmd || effective == DeinterlaceMode::Bob ||
        effective == DeinterlaceMode::Off || effective == DeinterlaceMode::Auto) {
        return 0;
    }
    return static_cast<std::int32_t>(scale) * (field_phase & 1);
}

int field_from_csr(bool csr_field_bit) {
    return csr_field_bit ? 1 : 0;
}

Image::Image(std::uint32_t w, std::uint32_t h, Rgba8 fill)
    : width(w), height(h), pixels(static_cast<std::size_t>(w) * h, fill) {}

Rgba8& Image::at(std::uint32_t x, std::uint32_t y) {
    return pixels[static_cast<std::size_t>(y) * width + x];
}

const Rgba8& Image::at(std::uint32_t x, std::uint32_t y) const {
    return pixels[static_cast<std::size_t>(y) * width + x];
}

std::uint8_t blend_channel(std::uint8_t src, std::uint8_t dst, std::uint8_t a) {
    const int numerator = static_cast<int>(src) * a + static_cast<int>(dst) * (255 - a) + 127;
    return static_cast<std::uint8_t>(numerator / 255);
}

std::uint8_t merge_alpha_for_texel(const PmodeFields& pmode, std::uint8_t texel_alpha) {
    if (pmode.mmod) {
        return static_cast<std::uint8_t>(std::min<std::uint32_t>(pmode.alp, 255u));
    }
    // 0x80 = 1.0 (V-01-12 Hypothesis): the texel alpha is doubled and saturated.
    return static_cast<std::uint8_t>(std::min<int>(255, static_cast<int>(texel_alpha) * 2));
}

Image compose_merge(const MergePlan& plan, const PmodeFields& pmode, std::uint8_t bg_r,
                    std::uint8_t bg_g, std::uint8_t bg_b, std::uint32_t out_width,
                    std::uint32_t out_height, std::uint32_t scale, std::int32_t interlace_rows,
                    const SourcePlacement& rc1, const SourcePlacement& rc2) {
    const Rgba8 background{bg_r, bg_g, bg_b, static_cast<std::uint8_t>(std::min<std::uint32_t>(pmode.alp, 255u))};
    Image out(out_width, out_height, background);
    if (plan.outcome != MergeOutcome::Composite) {
        return out;
    }

    const auto draw = [&](const SourcePlacement& placement, bool blend) {
        if (placement.source == nullptr || placement.source->width == 0 || placement.source->height == 0 ||
            placement.dst_width <= 0 || placement.dst_height <= 0) {
            return;
        }
        const Image& source = *placement.source;
        const std::int64_t x0 = static_cast<std::int64_t>(placement.dst_x) * scale;
        const std::int64_t y0 = static_cast<std::int64_t>(placement.dst_y) * scale + interlace_rows;
        const std::int64_t w = static_cast<std::int64_t>(placement.dst_width) * scale;
        const std::int64_t h = static_cast<std::int64_t>(placement.dst_height) * scale;
        const std::int64_t left = std::max<std::int64_t>(0, x0);
        const std::int64_t top = std::max<std::int64_t>(0, y0);
        const std::int64_t right = std::min<std::int64_t>(out_width, x0 + w);
        const std::int64_t bottom = std::min<std::int64_t>(out_height, y0 + h);
        if (right <= left || bottom <= top) {
            return;
        }
        // Nearest sampling of the source (IN-34). The source column of every output column is computed once
        // per draw (the same expression; IN-360 performance only).
        std::vector<std::uint32_t> columns(static_cast<std::size_t>(right - left));
        for (std::int64_t x = left; x < right; ++x) {
            std::int64_t sx = (x - x0) * source.width / w;
            columns[static_cast<std::size_t>(x - left)] =
                static_cast<std::uint32_t>(std::clamp<std::int64_t>(sx, 0, source.width - 1));
        }
        for (std::int64_t y = top; y < bottom; ++y) {
            std::int64_t sy = (y - y0) * source.height / h;
            sy = std::clamp<std::int64_t>(sy, 0, source.height - 1);
            const Rgba8* sourceRow = source.pixels.data() + static_cast<std::size_t>(sy) * source.width;
            Rgba8* outRow = out.pixels.data() + static_cast<std::size_t>(y) * out_width;
            for (std::int64_t x = left; x < right; ++x) {
                const Rgba8 texel = sourceRow[columns[static_cast<std::size_t>(x - left)]];
                Rgba8& dst = outRow[x];
                if (!blend) {
                    dst = texel;
                    continue;
                }
                const std::uint8_t a = merge_alpha_for_texel(pmode, texel.a);
                dst.r = blend_channel(texel.r, dst.r, a);
                dst.g = blend_channel(texel.g, dst.g, a);
                dst.b = blend_channel(texel.b, dst.b, a);
                dst.a = texel.a;
            }
        }
    };

    if (plan.use_rc2) {
        draw(rc2, false);
    }
    if (plan.use_rc1) {
        draw(rc1, true);
    }
    return out;
}

// ---------------------------------------------------------------------------------------
// 13.4 Deinterlace
// ---------------------------------------------------------------------------------------

Image Deinterlacer::weave(const Image& merged, int row_parity) {
    if (!has_woven_ || !woven_.same_size(merged)) {
        woven_ = merged;
        has_woven_ = true;
        return woven_;
    }
    for (std::uint32_t y = 0; y < merged.height; ++y) {
        if ((static_cast<int>(y) & 1) != row_parity) {
            continue;
        }
        for (std::uint32_t x = 0; x < merged.width; ++x) {
            woven_.at(x, y) = merged.at(x, y);
        }
    }
    return woven_;
}

Image Deinterlacer::adaptive(const Image& merged, int row_parity, double threshold) {
    history_.push_back(FieldRecord{row_parity, merged});
    while (history_.size() > 4u) {
        history_.pop_front();
    }

    // Previous same-parity field: the most recent older entry with this row parity (IN-39, two fields back), the
    // previous field: the most recent older entry with the other parity, whose rows are exactly the lines this field
    // is missing (IN-337, one back), and IN-396 the older field of that parity (three back).
    const FieldRecord* previous_same = nullptr;
    const FieldRecord* previous_other = nullptr;
    const FieldRecord* older_other = nullptr;
    for (std::size_t index = history_.size() - 1u; index-- > 0u;) {
        const FieldRecord& record = history_[index];
        if (!record.image.same_size(merged)) continue;
        if (record.row_parity == row_parity) {
            if (previous_same == nullptr) previous_same = &record;
        } else if (previous_other == nullptr) {
            previous_other = &record;
        } else if (older_other == nullptr) {
            older_other = &record;
        }
    }
    const std::int64_t threshold_permille = std::llround(threshold * 1000.0);
    const bool have_output = has_previous_output_ && previous_output_.same_size(merged);

    Image out(merged.width, merged.height);
    for (std::uint32_t y = 0; y < merged.height; ++y) {
        const bool current_row = (static_cast<int>(y) & 1) == row_parity;
        const bool edge_row = y == 0u || y + 1u == merged.height;
        for (std::uint32_t x = 0; x < merged.width; ++x) {
            if (current_row) {
                out.at(x, y) = merged.at(x, y);  // rows of the field pass through
                continue;
            }
            // The woven value of a missing line: the previous field's line (IN-337), else the last output.
            const bool have_woven = previous_other != nullptr || have_output;
            const Rgba8 woven = previous_other != nullptr ? previous_other->image.at(x, y)
                                : have_output             ? previous_output_.at(x, y)
                                                          : merged.at(x, y);
            if (edge_row) {
                out.at(x, y) = woven;  // IN-396: the first and last rows are always woven (the reference's rule)
                continue;
            }
            // A missing line: the lines above and below are in the current field (IN-39).
            const Rgba8 up = merged.at(x, y - 1u);
            const Rgba8 down = merged.at(x, y + 1u);
            const Rgba8 bob{average_byte(up.r, down.r), average_byte(up.g, down.g),
                            average_byte(up.b, down.b), average_byte(up.a, down.a)};
            // 13.4 / IN-396 (the reference's motion test): the current field against the previous same-parity field
            // on the line above and the line below, and the field that would be woven in (one back) against the
            // older field of its parity (three back) on the line itself. IN-337's test of the current image's own
            // row y against two back was a second current-field test; it never noticed that the woven field had
            // moved, so a scrolling text was woven from two positions (a combed scrolling menu ticker).
            bool motion = previous_same == nullptr || older_other == nullptr || previous_other == nullptr;
            if (!motion) {
                motion = channel_excess(up, previous_same->image.at(x, y - 1u), threshold_permille) > 0 ||
                         channel_excess(down, previous_same->image.at(x, y + 1u), threshold_permille) > 0 ||
                         channel_excess(woven, older_other->image.at(x, y), threshold_permille) > 0;
            }
            if (!motion) {
                out.at(x, y) = woven;
                continue;
            }
            // Moving: the average of the lines above and below when they agree and the woven line differs from them
            // at least as much (the reference's rule), else the woven line; with nothing to weave, the average.
            const std::int64_t up_down = channel_excess(up, down, threshold_permille);
            const std::int64_t up_woven = channel_excess(up, woven, threshold_permille);
            const bool interpolate = !have_woven || (up_down < 0 && 10 * up_woven >= 9 * up_down);
            out.at(x, y) = interpolate ? bob : woven;
        }
    }
    previous_output_ = out;
    has_previous_output_ = true;
    return out;
}

Image Deinterlacer::process(const Image& merged, int vblank_field, DeinterlaceMode mode,
                            bool bottom_first, std::uint32_t scale, double adaptive_threshold) {
    const int parity = bottom_first ? 1 : 0;
    const int row_parity = (vblank_field ^ parity) & 1;
    switch (mode) {
        case DeinterlaceMode::Weave:
            return weave(merged, row_parity);
        case DeinterlaceMode::Blend: {
            const Image woven = weave(merged, row_parity);
            Image out(woven.width, woven.height);
            for (std::uint32_t y = 0; y < woven.height; ++y) {
                const std::uint32_t up = y == 0u ? 0u : y - 1u;
                const std::uint32_t down = y + 1u >= woven.height ? woven.height - 1u : y + 1u;
                for (std::uint32_t x = 0; x < woven.width; ++x) {
                    const Rgba8& a = woven.at(x, up);
                    const Rgba8& b = woven.at(x, y);
                    const Rgba8& c = woven.at(x, down);
                    const auto tap = [](int p, int q, int r) {
                        return static_cast<std::uint8_t>((p + 2 * q + r + 2) / 4);
                    };
                    out.at(x, y) = Rgba8{tap(a.r, b.r, c.r), tap(a.g, b.g, c.g), tap(a.b, b.b, c.b),
                                         tap(a.a, b.a, c.a)};
                }
            }
            return out;
        }
        case DeinterlaceMode::Bob: {
            // Shift by scale * (1 - field) lines; integer shifts make the bilinear draw a copy.
            const std::int64_t shift = static_cast<std::int64_t>(scale) * (1 - vblank_field);
            Image out(merged.width, merged.height);
            for (std::uint32_t y = 0; y < merged.height; ++y) {
                std::int64_t source_row = static_cast<std::int64_t>(y) - shift;
                source_row = std::clamp<std::int64_t>(source_row, 0, static_cast<std::int64_t>(merged.height) - 1);
                for (std::uint32_t x = 0; x < merged.width; ++x) {
                    out.at(x, y) = merged.at(x, static_cast<std::uint32_t>(source_row));
                }
            }
            return out;
        }
        case DeinterlaceMode::Adaptive:
            return adaptive(merged, row_parity, adaptive_threshold);
        case DeinterlaceMode::Auto:
        case DeinterlaceMode::Off:
            break;
    }
    return merged;
}

void Deinterlacer::clear_history() {
    history_.clear();
    has_woven_ = false;
    has_previous_output_ = false;
}

DeinterlaceMode resolve_auto_deinterlace(const DisplayLayout& layout, bool ffmd, bool scanmsk_active) {
    const bool rects_one_line_apart = layout.both_enabled &&
                                      std::llabs(static_cast<long long>(layout.circuits[0].screen_y) -
                                                 layout.circuits[1].screen_y) == 1;
    const bool scanmsk_rule = scanmsk_active && !rects_one_line_apart;  // (c)
    if (layout.one_circuit_dby_changed || ffmd || scanmsk_rule) {
        return DeinterlaceMode::Adaptive;
    }
    return DeinterlaceMode::Off;
}

// ---------------------------------------------------------------------------------------
// 13.5 Post-processing
// ---------------------------------------------------------------------------------------

void apply_shadeboost(Image& image, const ShadeBoostParams& params) {
    const double brightness = params.brightness / 50.0;
    const double contrast = params.contrast / 50.0;
    const double saturation = params.saturation / 50.0;
    const double inverse_gamma = 50.0 / params.gamma;  // c^(1/gamma), gamma = gamma / 50
    const auto clamp01 = [](double v) { return std::clamp(v, 0.0, 1.0); };
    for (Rgba8& pixel : image.pixels) {
        double c[3] = {pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0};
        for (double& channel : c) {
            channel = clamp01(channel * brightness);  // IN-40: clamp after every step
        }
        const double luma = 0.2125 * c[0] + 0.7154 * c[1] + 0.0721 * c[2];
        for (double& channel : c) {
            channel = clamp01(luma * (1.0 - saturation) + channel * saturation);
            channel = clamp01(0.5 * (1.0 - contrast) + channel * contrast);
            channel = clamp01(std::pow(channel, inverse_gamma));
        }
        pixel.r = clamp_byte(c[0] * 255.0);
        pixel.g = clamp_byte(c[1] * 255.0);
        pixel.b = clamp_byte(c[2] * 255.0);
    }
}

void apply_scanlines(Image& image) {
    for (std::uint32_t y = 1; y < image.height; y += 2) {
        for (std::uint32_t x = 0; x < image.width; ++x) {
            Rgba8& pixel = image.at(x, y);
            pixel.r = static_cast<std::uint8_t>(pixel.r >> 1);
            pixel.g = static_cast<std::uint8_t>(pixel.g >> 1);
            pixel.b = static_cast<std::uint8_t>(pixel.b >> 1);
        }
    }
}

// ---------------------------------------------------------------------------------------
// 13.6 Presentation rectangle
// ---------------------------------------------------------------------------------------

PixelRect presentation_rect(std::int32_t window_width, std::int32_t window_height,
                            double target_ratio, double crop_adjust) {
    const double window_ratio = static_cast<double>(window_width) / static_cast<double>(window_height);
    const double r = target_ratio * crop_adjust / window_ratio;
    PixelRect rect;
    if (r < 1.0) {
        rect.width = static_cast<std::int32_t>(std::floor(window_width * r + 0.5));
        rect.height = window_height;
        rect.x = (window_width - rect.width) / 2;
        rect.y = 0;
    } else {
        rect.width = window_width;
        rect.height = static_cast<std::int32_t>(std::floor(window_height / r + 0.5));
        rect.x = 0;
        rect.y = (window_height - rect.height) / 2;
    }
    return rect;
}

// ---------------------------------------------------------------------------------------
// Option parsing
// ---------------------------------------------------------------------------------------

std::optional<bool> parse_antiblur_option(std::string_view value) {
    if (value == "1") {
        return true;
    }
    if (value == "0") {
        return false;
    }
    return std::nullopt;
}

std::optional<DeinterlaceSetting> parse_deinterlace_option(std::string_view value) {
    DeinterlaceSetting setting;
    constexpr std::string_view kSuffix = "_bff";
    if (value.size() > kSuffix.size() && value.substr(value.size() - kSuffix.size()) == kSuffix) {
        setting.bottom_first = true;
        value.remove_suffix(kSuffix.size());
    }
    if (value == "auto") {
        setting.mode = DeinterlaceMode::Auto;
    } else if (value == "off") {
        setting.mode = DeinterlaceMode::Off;
    } else if (value == "weave") {
        setting.mode = DeinterlaceMode::Weave;
    } else if (value == "bob") {
        setting.mode = DeinterlaceMode::Bob;
    } else if (value == "blend") {
        setting.mode = DeinterlaceMode::Blend;
    } else if (value == "adaptive") {
        setting.mode = DeinterlaceMode::Adaptive;
    } else {
        return std::nullopt;
    }
    return setting;
}

std::optional<ShadeBoostParams> parse_shadeboost_option(std::string_view value) {
    int numbers[4] = {0, 0, 0, 0};
    std::size_t count = 0;
    std::size_t position = 0;
    while (true) {
        const std::size_t comma = value.find(',', position);
        const std::string_view token = value.substr(position, comma == std::string_view::npos ? std::string_view::npos
                                                                                               : comma - position);
        if (count >= 4 || token.empty() || token.size() > 3) {
            return std::nullopt;
        }
        int parsed = 0;
        for (char ch : token) {
            if (ch < '0' || ch > '9') {
                return std::nullopt;
            }
            parsed = parsed * 10 + (ch - '0');
        }
        // IN-40: values 1..200 (1.0 = 50).
        if (parsed < 1 || parsed > 200) {
            return std::nullopt;
        }
        numbers[count++] = parsed;
        if (comma == std::string_view::npos) {
            break;
        }
        position = comma + 1;
    }
    if (count != 4) {
        return std::nullopt;
    }
    return ShadeBoostParams{numbers[0], numbers[1], numbers[2], numbers[3]};
}

std::optional<std::vector<PostFx>> parse_postfx_option(std::string_view value) {
    std::vector<PostFx> effects;
    if (value.empty()) {
        return effects;
    }
    std::size_t position = 0;
    while (position <= value.size()) {
        const std::size_t comma = value.find(',', position);
        const std::string_view token = value.substr(position, comma == std::string_view::npos ? std::string_view::npos
                                                                                               : comma - position);
        if (token == "shadeboost") {
            effects.push_back(PostFx::ShadeBoost);
        } else if (token == "fxaa") {
            effects.push_back(PostFx::Fxaa);
        } else if (token == "cas") {
            effects.push_back(PostFx::Cas);
        } else if (token == "scanlines") {
            effects.push_back(PostFx::Scanlines);
        } else {
            return std::nullopt;
        }
        if (comma == std::string_view::npos) {
            break;
        }
        position = comma + 1;
    }
    std::sort(effects.begin(), effects.end());
    effects.erase(std::unique(effects.begin(), effects.end()), effects.end());
    return effects;
}

}  // namespace coregs::display
