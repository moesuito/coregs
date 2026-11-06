// SPDX-License-Identifier: MIT
//
// HWR display output: CPU reference for the PCRTC output stage (TASK-027, M6 and M15).
// Copyright (c) 2026 GT4Recomp contributors.
//
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapter 13 (video
// mode, circuit rectangles, offsets and anti-blur, merge, field phase, deinterlace,
// post-processing, presentation), chapter 01.7 (display registers) and chapter 14.6
// (vectors). Clean room, decision 0069. No other renderer's source was consulted.
//
// Scope: pure C++20, no Vulkan. The GPU passes of the output stage call these functions
// for their parameters and, in tests, compare against the reference images produced here.
// Where the specification is silent the choice is documented as an IN-xx row in
// docs/hw-renderer-spec/IMPLEMENTATION-NOTES.md and cited in comments as "IN-xx".

#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <string_view>
#include <vector>

namespace coregs::display {

// ---------------------------------------------------------------------------------------
// 13.1 Video mode
// ---------------------------------------------------------------------------------------

enum class VideoMode : std::uint8_t {
    None,
    Ntsc,
    Pal,
    Vesa,
    Progressive480,
    Progressive720,
    Interlaced1080,
};

// Per-mode base table of 13.1: width, field height, DX/DY base and the clock dividers.
struct VideoModeTable {
    std::uint32_t width = 0;
    std::uint32_t field_height = 0;
    std::uint32_t dx_base = 0;
    std::uint32_t dy_base = 0;
    std::uint32_t h_divider = 1;
    std::uint32_t v_divider = 1;
    bool analogue = false;  // NTSC, PAL and 1080i (IN-20)
};

VideoModeTable video_mode_table(VideoMode mode);

// SMODE1.CMOD/LC and SMODE2.INT of 13.1. CMOD 1 is not in the table: it is no known mode
// (IN-22). The INT bit of the 1080i/720p rows is SMODE2.INT.
VideoMode detect_video_mode(std::uint64_t smode1, std::uint64_t smode2);

// Interlaced flag of 13.1: SMODE2.INT and an analogue mode.
bool interlaced_flag(VideoMode mode, std::uint64_t smode2);

// Really interlaced of 13.1: SYNCV.VFP bit 0 and an analogue mode (IN-21). The table
// names CMOD != 0, which excludes 1080i (CMOD = 0); the analogue set is used instead.
bool really_interlaced(VideoMode mode, std::uint64_t syncv);

// ---------------------------------------------------------------------------------------
// 13.2 Circuit rectangles and offsets
// ---------------------------------------------------------------------------------------

// PMODE fields used here (01.7): EN1[0], EN2[1], MMOD[5], SLBG[7], ALP[8..15].
struct PmodeFields {
    bool en1 = false;
    bool en2 = false;
    bool mmod = false;
    bool slbg = false;
    std::uint32_t alp = 0;
};
PmodeFields decode_pmode(std::uint64_t pmode);

// DISPFB and DISPLAY decoding (01.7).
struct CircuitRegisters {
    std::uint64_t dispfb = 0;
    std::uint64_t display = 0;
};

struct CircuitFields {
    std::uint32_t fbp = 0;
    std::uint32_t fbw = 0;
    std::uint32_t psm = 0;
    std::uint32_t dbx = 0;
    std::uint32_t dby = 0;
    std::uint32_t dx = 0;
    std::uint32_t dy = 0;
    std::uint32_t magh = 0;
    std::uint32_t magv = 0;
    std::uint32_t dw = 0;
    std::uint32_t dh = 0;
};
CircuitFields decode_circuit(const CircuitRegisters& registers);

// One circuit after the rules of 13.2 for the current VBlank.
struct CircuitRect {
    bool enabled = false;
    std::uint32_t fbp = 0;
    std::uint32_t fbw = 0;
    std::uint32_t psm = 0;
    std::int32_t dbx = 0;  // read origin after the framebuffer rules (IN-24)
    std::int32_t dby = 0;
    std::int32_t dx = 0;   // raw display offsets
    std::int32_t dy = 0;   // raw display offsets, after the anti-blur of 13.2 (a)
    std::uint32_t display_width = 0;   // final render size (display rectangle)
    std::uint32_t display_height = 0;
    std::uint32_t read_width = 0;      // read rectangle; FFMD interlaced halves the height
    std::uint32_t read_height = 0;
    std::int32_t screen_x = 0;  // placement from the offsets rules, relative to the union
    std::int32_t screen_y = 0;
    bool dby_changed = false;   // DBY differs from the value of the previous VBlank
};

// Inputs of one VBlank of the output stage.
struct DisplayRegisters {
    std::uint64_t smode1 = 0;
    std::uint64_t smode2 = 0;
    std::uint64_t syncv = 0;
    std::uint64_t pmode = 0;
    std::array<CircuitRegisters, 2> circuits{};  // [0] = DISPFB1/DISPLAY1, [1] = DISPFB2/DISPLAY2
    bool anti_blur = true;      // COREGS_HW_ANTIBLUR, default on
    bool scanmsk_active = false;  // GS SCANMSK register in use (MSK != 0)
};

// Result of the rules of 13.2 and the union of 13.3 for one VBlank.
struct DisplayLayout {
    VideoMode mode = VideoMode::None;
    bool valid = false;  // false when SMODE1 describes no known mode (no image)
    bool interlaced = false;
    bool really_interlaced = false;
    bool ffmd = false;
    std::array<CircuitRect, 2> circuits{};
    bool any_dby_changed = false;        // "toggling" (IN-30)
    bool one_circuit_dby_changed = false;  // 13.4 auto rule (a)
    bool both_enabled = false;
    std::uint32_t union_width = 0;   // bounding box of the enabled display rectangles
    std::uint32_t union_height = 0;
};

// Applies 13.2 for one VBlank. `previous_dby` holds the DBY values of the previous VBlank
// and is updated to this VBlank's raw values.
DisplayLayout compute_layout(const DisplayRegisters& registers,
                             std::array<std::uint32_t, 2>& previous_dby);

// ---------------------------------------------------------------------------------------
// 13.3 Merge: output size and source selection
// ---------------------------------------------------------------------------------------

// The deinterlace mode as selected by the options, before auto resolution.
enum class DeinterlaceMode : std::uint8_t {
    Auto,
    Off,
    Weave,
    Bob,
    Blend,
    Adaptive,
};

// Output size before the scale factor (13.3). `deinterlace_off` is true when the effective
// deinterlace mode is off. Returns the table size when no circuit is enabled.
struct OutputSize {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};
OutputSize merge_output_size(const DisplayLayout& layout, bool deinterlace_off, std::uint32_t scale);

enum class MergeOutcome : std::uint8_t {
    KeepPrevious,    // a circuit is enabled without a source: nothing presented (13.3)
    BackgroundOnly,  // both circuits disabled or unused: pure BGCOLOR
    Composite,
};

struct MergePlan {
    MergeOutcome outcome = MergeOutcome::BackgroundOnly;
    bool use_rc1 = false;
    bool use_rc2 = false;
    bool single_read = false;  // both circuits read the same source (13.3)
};

// Source selection of 13.3. `source_available` tells whether the circuit has a source.
MergePlan plan_merge(const PmodeFields& pmode, const CircuitRect& rc1, const CircuitRect& rc2,
                     bool source1_available, bool source2_available);

// Same source: same FBP and FBW and bit-compatible PSM (IN-33: CT32 and CT24 are
// compatible, CT16 and CT16S are compatible, Z32 and Z24 are compatible, others equal).
bool same_source(const CircuitRect& a, const CircuitRect& b);

// Rows the interlace offset of 13.3 step 3 shifts the destination down by. Zero unless the
// signal is really interlaced, FFMD = 1 and the effective deinterlace is not bob or off.
std::int32_t interlace_offset_rows(bool really_interlaced, bool ffmd, DeinterlaceMode effective,
                                   std::uint32_t scale, int field_phase);

// Field phase of 13.3 and 12.4: the VBlank field is CSR.FIELD (IN-35: bit 1 is field 1).
int field_from_csr(bool csr_field_bit);

// ---------------------------------------------------------------------------------------
// Images and the CPU reference of the merge and deinterlace (13.3 and 13.4)
// ---------------------------------------------------------------------------------------

struct Rgba8 {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    std::uint8_t a = 0;
    bool operator==(const Rgba8&) const = default;
};

struct Image {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<Rgba8> pixels;

    Image() = default;
    Image(std::uint32_t w, std::uint32_t h, Rgba8 fill = {});
    Rgba8& at(std::uint32_t x, std::uint32_t y);
    const Rgba8& at(std::uint32_t x, std::uint32_t y) const;
    bool same_size(const Image& other) const { return width == other.width && height == other.height; }
};

// A circuit's source image and its destination in table units (before the scale factor).
struct SourcePlacement {
    const Image* source = nullptr;
    std::int32_t dst_x = 0;
    std::int32_t dst_y = 0;
    std::int32_t dst_width = 0;
    std::int32_t dst_height = 0;
};

// Composes the merge image (13.3). Output is `out_width` x `out_height` at `scale`; the
// destination rectangles are multiplied by `scale` and shifted down by `interlace_rows`.
// Circuits are copied (RC2) or blended (RC1) with nearest sampling of the source.
Image compose_merge(const MergePlan& plan, const PmodeFields& pmode, std::uint8_t bg_r,
                    std::uint8_t bg_g, std::uint8_t bg_b, std::uint32_t out_width,
                    std::uint32_t out_height, std::uint32_t scale, std::int32_t interlace_rows,
                    const SourcePlacement& rc1, const SourcePlacement& rc2);

// Blend of one RC1 texel with the destination (13.3): colour = src * a + dst * (1 - a),
// a = ALP / 255 for MMOD = 1, else min(255, 2 * texel alpha) (V-01-12 Hypothesis).
std::uint8_t blend_channel(std::uint8_t src, std::uint8_t dst, std::uint8_t a);
std::uint8_t merge_alpha_for_texel(const PmodeFields& pmode, std::uint8_t texel_alpha);

// Deinterlacer of 13.4 with the four modes and the adaptive history. It is stateful: the
// weave image persists between fields and the adaptive history keeps the last four fields.
class Deinterlacer {
public:
    // `mode` must be a resolved mode (Weave, Bob, Blend or Adaptive). `bottom_first` picks
    // the parity of the field (the _bff suffix). `scale` multiplies the bob shift.
    Image process(const Image& merged, int vblank_field, DeinterlaceMode mode, bool bottom_first,
                  std::uint32_t scale, double adaptive_threshold = 0.08);

    // Clears the persistent images and the history (13.4: no source this VBlank).
    void clear_history();

    std::size_t history_size() const { return history_.size(); }

private:
    struct FieldRecord {
        int row_parity = 0;
        Image image;
    };

    Image weave(const Image& merged, int row_parity);
    Image adaptive(const Image& merged, int row_parity, double threshold);

    Image woven_;
    bool has_woven_ = false;
    Image previous_output_;
    bool has_previous_output_ = false;
    std::deque<FieldRecord> history_;
};

// Resolution of `auto` (13.4): adaptive when (a) exactly one circuit changed its DBY, (b)
// FFMD = 1, or (c) SCANMSK is in use and the two display rectangles are not one line apart
// (IN-32: compared on screen_y); otherwise off.
DeinterlaceMode resolve_auto_deinterlace(const DisplayLayout& layout, bool ffmd,
                                         bool scanmsk_active);

// ---------------------------------------------------------------------------------------
// 13.5 Post-processing (ShadeBoost and scanlines; FXAA and CAS need the GPU)
// ---------------------------------------------------------------------------------------

struct ShadeBoostParams {
    int brightness = 50;  // integers around 50 (= 1.0)
    int contrast = 50;
    int saturation = 50;
    int gamma = 50;
};

void apply_shadeboost(Image& image, const ShadeBoostParams& params);

// Odd output rows at half intensity (scanlines at present).
void apply_scanlines(Image& image);

enum class PostFx : std::uint8_t {
    ShadeBoost,
    Fxaa,
    Cas,
    Scanlines,
};

// ---------------------------------------------------------------------------------------
// 13.6 Presentation rectangle
// ---------------------------------------------------------------------------------------

struct PixelRect {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t width = 0;
    std::int32_t height = 0;
};

// 16:9 presentation rectangle inside a window (13.6 aspect rule). r = target ratio times
// crop adjust over the window ratio; r < 1 pillarboxes, r > 1 letterboxes.
PixelRect presentation_rect(std::int32_t window_width, std::int32_t window_height,
                            double target_ratio = 16.0 / 9.0, double crop_adjust = 1.0);

// ---------------------------------------------------------------------------------------
// Option parsing (13.4, 13.5, 13.2). Unknown values return nullopt; the caller aborts.
// ---------------------------------------------------------------------------------------

std::optional<bool> parse_antiblur_option(std::string_view value);

struct DeinterlaceSetting {
    DeinterlaceMode mode = DeinterlaceMode::Auto;
    bool bottom_first = false;
};
std::optional<DeinterlaceSetting> parse_deinterlace_option(std::string_view value);

std::optional<ShadeBoostParams> parse_shadeboost_option(std::string_view value);

// Comma list. An empty string is no post-processing. The result is in the fixed order of
// 13.5 with duplicates removed.
std::optional<std::vector<PostFx>> parse_postfx_option(std::string_view value);

}  // namespace coregs::display
