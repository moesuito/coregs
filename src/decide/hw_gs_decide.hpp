// SPDX-License-Identifier: MIT
//
// HWR per-draw decisions (TASK-027): the pure, GPU-free part of the draw translator.
// Copyright (c) 2026 GT4Recomp contributors.
//
// Written from docs/hw-renderer-spec (chapters 02.5, 02.9, 02.10, 03, 05.8, 08, 09 and
// the matching parts of 14.6 and 15) under the clean-room process of decision 0069. No
// other renderer's source was consulted. Interpretations of silent or conflicting spec
// points are recorded as IN-100 and up in docs/hw-renderer-spec/IMPLEMENTATION-NOTES.md.
//
// Inputs are plain structs: a register snapshot, the batch geometry, tracked target and
// texture ranges, device capabilities and options. Outputs are plain structs. Nothing here
// touches memory, a GPU, a clock or a file. Channel values are 0..255 and blend factors use
// the GS encoding (0x80 = 1.0).

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace coregs::decide {

// ---- Register vocabulary ----------------------------------------------------

enum class PrimClass : std::uint8_t { Point, Line, Triangle, Sprite };
enum class FrameFormat : std::uint8_t { Ct32, Ct24, Ct16 };
enum class ZFormat : std::uint8_t { Z32, Z24, Z16 };
enum class TexFormat : std::uint8_t { Ct32, Ct24, Ct16, Ct16S, T8, T4, T8H, T4HL, T4HH };
enum class Tfx : std::uint8_t { Modulate, Decal, Highlight, Highlight2 };
enum class WrapMode : std::uint8_t { Repeat, Clamp, RegionClamp, RegionRepeat };
enum class AlphaCompare : std::uint8_t { Never, Always, Less, LessEqual, Equal, GreaterEqual, Greater, NotEqual };
// AFAIL: KEEP, FB_ONLY, ZB_ONLY, RGB_ONLY.
enum class AlphaFail : std::uint8_t { Keep, FrameOnly, DepthOnly, RgbOnly };
enum class DepthCompare : std::uint8_t { Never, Always, GreaterEqual, Greater };
enum class OverlapKind : std::uint8_t { None, Yes, Unknown };

// Blend selectors: A, B and D pick Cs, Cd or 0; C picks As, Ad or FIX. 3^4 = 81 modes.
enum class BlendSel : std::uint8_t { Cs, Cd, Zero };
enum class BlendFactorSel : std::uint8_t { As, Ad, Fix };

struct BlendMode {
    BlendSel a = BlendSel::Cs;
    BlendSel b = BlendSel::Cs;
    BlendSel d = BlendSel::Cs;
    BlendFactorSel c = BlendFactorSel::As;

    static BlendMode from_index(int index);  // 0..80
    int index() const;
    friend bool operator==(const BlendMode&, const BlendMode&) = default;
};

// Blend groups of chapter 08.3 (the counts over the 81 modes are 27, 8, 6, 6, 6, 7, 18, 3).
enum class BlendGroup : std::uint8_t {
    SameAsD,       // A = B: result is D
    NonRecursive,  // Cs-only products (no Cd read, no Ad)
    SourceTimesAd, // Cs times Ad forms
    Accumulate,    // Cs*a +- Cd (and Cs*Ad +- Cd)
    Interpolate,   // Cs*a + Cd*(1-a) and the mirror
    Subtract,      // (Cs - Cd)*a and (Cd - Cs)*a
    Impossible,    // factor above one on one side of a Cd term
    CdMultiplied,  // Cs +- Cd*a, Cd*(1+a), Cd*a, Cd*(1-a), -Cd*a
};

enum class BlendLevel : std::uint8_t { Basic, Medium, High, Full };

// ---- Device and options -----------------------------------------------------

struct DeviceCaps {
    bool raster_order_access = false;     // VK_EXT_rasterization_order_attachment_access
    bool feedback_loop_layout = false;    // VK_EXT_attachment_feedback_loop_layout
    bool input_attachment = false;        // input attachment in GENERAL layout
    bool depth_feedback = false;          // depth readable while written (feedback layout)
    bool provoking_vertex_last = false;   // VK_EXT_provoking_vertex, last-vertex mode
    bool wide_lines = false;              // wideLines feature
    bool line_rasterization = false;      // VK_EXT_line_rasterization
    float max_point_size = 1.0f;
    float max_line_width = 1.0f;
    std::uint32_t max_scale = 1;          // device ceiling for the upscale factor

    // True when any fetch-style mechanism (1 to 3 of chapter 08.8) exists.
    bool framebuffer_fetch() const { return raster_order_access || feedback_loop_layout || input_attachment; }
};

struct Options {
    BlendLevel blend_level = BlendLevel::High;   // COREGS_HW_BLEND
    bool accurate_atest = false;                 // COREGS_HW_ACCURATE_ATEST=1
    bool dither_enabled = true;                  // COREGS_HW_DITHER != off
    std::uint32_t scale = 1;                     // upscale factor of the target (chapter 06)
    bool prim_id_available = false;              // PRIM_ID DATE exists (M14; off in phase 1)
};

// ---- Register snapshot (the fields the translator reads, chapter 03.1) ------

struct RegisterSnapshot {
    // PRIM and ALPHA
    bool abe = false;                 // PRIM.ABE
    bool aa1 = false;                 // PRIM.AA1
    bool iip = true;                  // PRIM.IIP (Gouraud)
    bool fge = false;                 // PRIM.FGE (fog enable)
    bool pabe = false;                // PABE
    BlendMode blend{};                // ALPHA A, B, C, D
    std::uint8_t fix = 0;             // ALPHA.FIX
    bool colclamp = true;             // COLCLAMP
    bool fba = false;                 // FBA
    bool dthe = false;                // DTHE
    std::uint32_t scanmsk = 0;        // SCANMSK (0 off, 2 even rows, 3 odd rows)

    // Texture
    bool tme = false;                 // TEX0 enabled (PRIM.TME)
    bool fst = false;                 // PRIM.FST
    Tfx tfx = Tfx::Modulate;
    bool tcc = false;
    TexFormat tex_format = TexFormat::Ct32;
    std::uint32_t tbp0 = 0;
    std::uint32_t tbw = 1;
    std::uint8_t tw_log2 = 0;
    std::uint8_t th_log2 = 0;
    WrapMode wms = WrapMode::Repeat;
    WrapMode wmt = WrapMode::Repeat;

    // Frame and depth
    FrameFormat frame = FrameFormat::Ct32;
    std::uint32_t fbp = 0;
    std::uint32_t fbw = 1;
    std::uint32_t fbmsk = 0;          // FBMSK, 1 = bit protected
    ZFormat z_format = ZFormat::Z32;
    std::uint32_t zbp = 0;
    bool zmsk = false;                // ZBUF.ZMSK (1 = no Z write)
    bool zte = true;                  // TEST.ZTE
    DepthCompare ztst = DepthCompare::Always;

    // TEST
    bool ate = false;
    AlphaCompare atst = AlphaCompare::Always;
    std::uint8_t aref = 0;
    AlphaFail afail = AlphaFail::Keep;
    bool date = false;                // TEST.DATE
    bool datm = false;                // TEST.DATM (0 pass when alpha MSB clear)

    // Scissor (inclusive ends, pixels)
    int scax0 = 0, scax1 = 0, scay0 = 0, scay1 = 0;
};

// ---- Batch input --------------------------------------------------------------

// One vertex after XYOFFSET (chapter 02.3). x and y are GS pixels; s, t and q are the
// registers as stored; u and v are 12.4 texel coordinates (FST). Colour is 0..255.
struct Vertex2D {
    double x = 0.0;
    double y = 0.0;
    std::uint32_t z = 0;
    std::array<std::uint8_t, 4> rgba{0, 0, 0, 0};
    std::int32_t fog = 0;
    double s = 0.0;
    double t = 0.0;
    double q = 1.0;
    double u = 0.0;  // texels (already divided by 16)
    double v = 0.0;
};

// Primitives reference vertices by index: 1 index for points, 2 for lines and sprites, 3 for
// triangles. Sprites use the first index as the top-left and the second as the bottom-right.
struct BatchGeometry {
    PrimClass prim = PrimClass::Sprite;
    std::vector<Vertex2D> vertices;
    std::vector<std::uint32_t> indices;
};

// Facts about the texture source resolved by the caller (chapters 04 and 07).
struct TextureSource {
    bool is_frame_target = false;     // the texture is the current colour target
    bool is_depth_target = false;     // the texture is a depth target
    bool real_16bit_source = false;   // a 16-bit source that is not a 32-bit view
    bool opaque = true;               // every texel alpha is 0x80 or more (tracked)
    std::uint8_t alpha_min = 0;       // texture alpha range after TEXA and palette
    std::uint8_t alpha_max = 255;
    bool src_is_32bit_target = false; // shuffle precondition: source is a 32-bit target
};

// Tracked colour target alpha range (chapter 08.2, decision: required).
struct TargetTracking {
    std::uint8_t alpha_min = 0;
    std::uint8_t alpha_max = 255;
    bool alpha_scaled = false;        // alpha-scaled state (M14; always false in phase 1)
};

// Facts from the clear pipeline (chapter 05.8) that cannot be derived from one batch.
struct ClearContext {
    bool target_present = true;       // a frame target exists at FBP for this batch
    bool next_batch_nonclear_same_pair = false;
    std::uint32_t display_height = 480;
};

struct BatchInput {
    RegisterSnapshot regs{};
    BatchGeometry geometry{};
    TextureSource texture{};
    TargetTracking target{};
    ClearContext clear{};
    // Look-ahead facts from the front end.
    bool current_prim_is_point = false;     // PRIM of the context after the batch
    bool fragment_alpha_known = true;       // false when the trace could not bound alpha
    std::uint8_t fragment_alpha_min = 0;    // vertex alpha combined with texture alpha (TFX)
    std::uint8_t fragment_alpha_max = 255;
    std::uint32_t draw_list_groups_override = 0;  // 0: derive from geometry
};

// ---- Trace and geometry results ------------------------------------------------

struct VertexTrace {
    double x_min = 0.0, x_max = 0.0, y_min = 0.0, y_max = 0.0;
    std::uint32_t z_min = 0, z_max = 0;
    std::int32_t fog_min = 0, fog_max = 0;
    std::array<std::uint8_t, 4> rgba_min{}, rgba_max{};
    bool colour_constant = true;
    bool z_constant = true;
    double u_min = 0.0, u_max = 0.0, v_min = 0.0, v_max = 0.0;
    double q_min = 0.0, q_max = 0.0;
    bool unsafe_q = false;
    bool coordinate_nan = false;
};

// Integer pixel rectangle, half open: [x0, x1) x [y0, y1).
struct PixelRect {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    bool empty() const { return x1 <= x0 || y1 <= y0; }
    friend bool operator==(const PixelRect&, const PixelRect&) = default;
};

// Draw-list group (chapter 02.10): primitives whose boxes do not overlap each other.
struct DrawListGroup {
    std::uint32_t primitive_count = 0;
    PixelRect box{};  // in native pixels, extended by one pixel for bilinear sampling
};

struct OverlapResult {
    OverlapKind kind = OverlapKind::Unknown;
    std::uint32_t groups = 1;
    std::vector<DrawListGroup> draw_list;
};

// ---- Method enums -----------------------------------------------------------------

enum class BlendMethod : std::uint8_t { None, FixedFunction, Accumulate, Mixed, Shader };

enum class BlendReason : std::uint8_t {
    Disabled,
    SameAsD,
    NonRecursiveShader,
    ExactAccumulate,
    PabeShader,
    ImpossibleShader,
    DitherShader,
    Cd16Shader,
    BarrierShader,
    FactorAbove1Shader,
    AdFactorShader,
    ColourWrapShader,
    SpriteBatchShader,
    NoOverlapShader,
    FullLevelShader,
    MixedBasic,
    FixedFunctionBasic,
    FixedFunctionApprox,
};

struct BlendPlan {
    BlendMode mode{};            // after simplification (chapter 08.2)
    BlendGroup group = BlendGroup::SameAsD;
    bool simplified = false;
    std::uint8_t effective_fix = 0;
    BlendMethod method = BlendMethod::None;
    BlendReason reason = BlendReason::Disabled;
    bool reads_destination = false;   // the method reads Cd or Ad
    bool approximate = false;         // FF or MIX is not bit exact (chapter 08.3)
    bool factor_above_one = false;
};

enum class DateOutcome : std::uint8_t { NoTest, SkipAllFail, Test };
enum class DateMethod : std::uint8_t { None, Full, StencilOne, Stencil, PrimId };

struct DatePlan {
    DateOutcome outcome = DateOutcome::NoTest;
    DateMethod method = DateMethod::None;
    bool reads_green = false;  // inside a texture shuffle the test reads G instead of A
};

enum class AlphaFold : std::uint8_t { AllPass, AllFail, Mixed };
enum class AlphaMethod : std::uint8_t { None, Keep, Never, Feedback, SplitFrameOnly, SplitRgbOnly, PassThenFail };

// Which channels a write may touch. Partial FBMSK bits are reported separately.
struct WriteMask {
    bool rgb = false;
    bool alpha = false;
    bool depth = false;

    bool any() const { return rgb || alpha || depth; }
    friend bool operator==(const WriteMask&, const WriteMask&) = default;
};

struct AlphaPass {
    AlphaCompare compare = AlphaCompare::Always;  // test applied to this pass
    bool discard = false;        // failing fragments are discarded
    bool feedback = false;       // failing fragments rewrite the old value (one-pass feedback)
    bool dual_source_alpha = false;
    WriteMask writes{};
};

struct AlphaTestPlan {
    bool test_enabled = false;   // ATE after folding
    AlphaFold fold = AlphaFold::Mixed;
    AlphaMethod method = AlphaMethod::None;
    WriteMask writes{};          // masks after folding (whole-batch view)
    std::vector<AlphaPass> passes;
};

enum class ChannelState : std::uint8_t { Off, Normal, Partial };

struct ChannelWrites {
    std::array<ChannelState, 4> channel{ChannelState::Off, ChannelState::Off, ChannelState::Off, ChannelState::Off};
    bool rgb_any = false;
    bool alpha_any = false;
    bool any_partial = false;  // needs software FBMASK when colour is written
};

enum class BarrierKind : std::uint8_t { None, One, PerDrawListGroup, PerPrimitive };

enum class ColourWrapPlan : std::uint8_t { NotNeeded, ShaderFullResult, SideTarget };

enum class Expansion : std::uint8_t {
    None,
    PointSize,
    PointQuad,
    LineWidth,
    LineQuad,
    SpriteVertexStage,
    SpriteCpu,
    TriangleList,
};

struct PrimitivePlan {
    PrimClass prim = PrimClass::Sprite;
    Expansion expansion = Expansion::None;
    bool flat_deindex = false;     // flat shading needs the last-vertex colour copied (chapter 03.2)
    bool needs_unsafe_q_cpu = false;
};

enum class ShuffleKind : std::uint8_t {
    None,
    Copy,
    Offset,
    RegionRepeat8,
    RegionRepeat16,
    Reverse,
    Swizzle,
};

enum class ClearKind : std::uint8_t { None, TargetClear, MemoryClear, DoubleHalf, StripedDouble };

enum class SkipReason : std::uint8_t {
    None,
    ResetArtefact,
    DateOnCt24,
    NothingWritten,
    EmptyRect,
    ConvertedToClear,
};

// Pass kinds of chapter 03.10, in the normative order.
enum class PassKind : std::uint8_t {
    ColourWrapResolve,
    HazardCopy,
    DatePrepare,
    ColourWrapConvert,
    StencilClear,
    MainDraw,
    BlendSecondPass,
    AlphaSecondPass,
    ColourWrapResolveAfter,
};

// Pipeline key (chapter 03.9). Normalised: fields that cannot change the result are zeroed.
struct PipelineKey {
    // vertex key
    bool tme = false;
    bool fst = false;
    bool iip = false;
    Expansion expansion = Expansion::None;
    // pixel key
    std::uint8_t tfx = 0;
    std::uint8_t tcc = 0;
    std::uint8_t wrap_s = 0;
    std::uint8_t wrap_t = 0;
    std::uint8_t palette_format = 0;
    bool fog = false;
    bool fba = false;
    std::uint8_t dest_format = 0;
    bool dither = false;
    bool colour_wrap_shader = false;
    bool pabe = false;
    std::uint8_t alpha_method = 0;
    std::uint8_t date_method = 0;
    bool software_depth = false;
    bool software_fbmask = false;
    // fixed-function blend state
    bool blend_enabled = false;
    std::uint8_t blend_op = 0;
    std::uint8_t blend_src = 0;
    std::uint8_t blend_dst = 0;
    bool dual_source = false;
    // masks, depth and topology
    bool write_rgb = false;
    bool write_alpha = false;
    bool write_depth = false;
    std::uint8_t depth_compare = 0;
    std::uint8_t topology = 0;
    std::uint8_t blend_method = 0;

    friend bool operator==(const PipelineKey&, const PipelineKey&) = default;
};

struct DrawPlan {
    SkipReason skip = SkipReason::None;
    bool draw_converted_to_clear = false;

    // Simplified registers that the draw will use.
    BlendPlan blend{};
    AlphaTestPlan alpha_test{};
    DatePlan date{};
    bool date_enabled = false;
    bool dither = false;
    bool texture_used = false;
    bool texture_shuffle_candidate = false;
    bool channel_shuffle_candidate = false;
    bool depth_test_always = false;   // ZTST simplified to ALWAYS
    WriteMask writes{};               // whole-batch write enables after all folding
    ChannelWrites colour_channels{};
    bool software_fbmask = false;
    bool z_written = false;

    PixelRect draw_rect{};
    VertexTrace trace{};
    OverlapResult overlap{};
    PrimitivePlan primitive{};

    ClearKind clear = ClearKind::None;
    PixelRect clear_rect{};
    std::uint32_t clear_colour_word = 0;
    std::uint32_t clear_depth_word = 0;
    std::uint32_t clear_base_fbp = 0;

    ShuffleKind texture_shuffle = ShuffleKind::None;

    BarrierKind barrier = BarrierKind::None;
    ColourWrapPlan colour_wrap = ColourWrapPlan::NotNeeded;
    bool reads_target = false;
    bool writes_target = false;

    std::vector<PassKind> passes;
    PipelineKey key{};
};

// ---- Top level and building blocks (all pure) -------------------------------------

DrawPlan decide_draw(const BatchInput& input, const DeviceCaps& caps, const Options& options);

// Vertex trace over the referenced vertices (chapter 02.9). Alpha is not derived here.
VertexTrace compute_vertex_trace(const BatchGeometry& geometry, const RegisterSnapshot& regs);

// Draw rectangle (chapter 03.1 step 9) from a native pixel span, before scissoring. Normative
// table: lower = floor(min + 0.9), upper = floor(max + 0.6), extended by one when empty
// (IN-100 explains why the upper bound is not round(max + 0.4)).
PixelRect draw_rectangle_from_span(double min, double max);  // x0/x1 only; y0 = y1 = 0
PixelRect draw_rectangle(const VertexTrace& trace, const RegisterSnapshot& regs);

// Pixels covered at 1x for an edge range (chapter 03.3): N with start <= N < end.
// Returns the half-open interval [first, last) of covered pixels.
PixelRect coverage_pixels_1x(double start, double end);

OverlapResult classify_overlap(const BatchGeometry& geometry, const VertexTrace& trace);

// Blend mode groups and simplification (chapter 08.2, 08.3).
BlendGroup classify_blend(BlendMode mode);
bool blend_group_reads_destination(BlendGroup group);
const char* to_string(BlendGroup group);
BlendPlan decide_blend(const RegisterSnapshot& regs, const BatchInput& input, const Options& options,
                       std::uint8_t frag_alpha_min, std::uint8_t frag_alpha_max, bool cs_provably_zero,
                       OverlapKind overlap, std::uint32_t groups, bool sprite_batch, bool barrier_other);

// Colour write state of a frame (chapter 08.9 classification of FBMSK).
ChannelWrites channel_writes(FrameFormat frame, std::uint32_t fbmsk);

// Alpha test (chapter 08.6).
AlphaFold fold_alpha_test(AlphaCompare compare, std::uint8_t min_alpha, std::uint8_t max_alpha, std::uint8_t reference);
WriteMask alpha_fail_writes(AlphaFail fail, FrameFormat frame);
struct AlphaTestContext {
    OverlapKind overlap = OverlapKind::None;
    bool blend_uses_ad = false;
    bool z_independent = true;
    bool barrier_other = false;
    bool framebuffer_fetch = false;
    bool depth_feedback = false;
    bool accurate_option = false;
};

AlphaTestPlan decide_alpha_test(const RegisterSnapshot& regs, std::uint8_t frag_min, std::uint8_t frag_max,
                                WriteMask enabled, const AlphaTestContext& context);


// Destination alpha test (chapter 08.7).
struct DateContext {
    OverlapKind overlap = OverlapKind::None;
    std::uint8_t written_min = 0;
    std::uint8_t written_max = 255;
    bool alpha_written = false;
    bool fba = false;
    bool shuffle = false;
    bool complex_atest_overlap = false;
    bool framebuffer_fetch = false;
    std::uint32_t draw_list_groups = 1;
    std::uint32_t index_count = 0;
    bool full_barrier_needed = false;
    bool prim_id_available = false;
};
DatePlan decide_date(bool date, bool datm, FrameFormat frame, std::uint8_t target_min, std::uint8_t target_max,
                     const DateContext& context);

// Barriers (chapter 08.8).
BarrierKind decide_barrier(bool reads_target, bool writes_target, OverlapKind overlap, std::uint32_t groups,
                           bool shuffle);

// Colour wrap (chapter 08.5).
ColourWrapPlan decide_colour_wrap(bool colclamp, const BlendPlan& blend, FrameFormat frame, BlendLevel level,
                                  OverlapKind overlap, bool overflow_possible);

// Scanline mask (chapter 03.8): true when a fragment on this GPU row is discarded.
bool scanmsk_discards_row(std::uint32_t scanmsk, int gpu_y, std::uint32_t scale);

// Primitive expansion (chapter 03.2).
PrimitivePlan decide_primitive(PrimClass prim, std::uint32_t scale, bool iip_gouraud, bool uniform_colour,
                               bool unsafe_q, const DeviceCaps& caps);

// Pipeline key with normalisation (chapter 03.9).
PipelineKey make_pipeline_key(const DrawPlan& plan, const RegisterSnapshot& regs, const DeviceCaps& caps);

// Texture shuffles (chapter 09.2).
struct ShuffleSample {
    double frame_x0 = 0.0;   // first sprite in frame pixels
    double frame_x1 = 0.0;
    double tex_u0 = 0.0;     // texel u of the same corners
    double tex_u1 = 0.0;
    double frame_y0 = 0.0;
    double frame_y1 = 0.0;
    double tex_v0 = 0.0;
    double tex_v1 = 0.0;
    std::uint32_t fbmsk = 0;
    bool frame_16bit = false;
    bool sprite_or_quads = true;
    bool texture_is_frame = false;
    bool opaque = true;
    bool nearest = true;
    bool region_repeat_s = false;
    bool region_repeat_t = false;
    std::uint32_t region_mask_s = 0;
    std::uint32_t region_fix_s = 0;
    bool next_quad_adjacent = false;   // next quad starts where this one ends
    bool next_quad_8_apart = false;
    bool axes_swapped = false;         // x runs along u's axis swap (swizzle detection)
    std::uint32_t width_pixels = 0;
    bool tiny_draw = false;
    bool source_is_32bit = true;
    bool real_16bit_source = false;
};
ShuffleKind classify_texture_shuffle(const ShuffleSample& sample);
// Source channel per destination channel (R, G, B, A); -1 when the shader selects per texel.
std::array<int, 4> texture_shuffle_channel_source(ShuffleKind kind, std::uint32_t fbmsk);

// Writable channels of a 16-bit texture shuffle (chapter 09.2): RG unless the low byte of the
// 16-bit FBMSK is 0xFF, BA unless the high byte is 0xFF.
struct ShuffleChannelSet {
    bool rg = false;
    bool ba = false;
};
ShuffleChannelSet texture_shuffle_channel_set(std::uint32_t fbmsk16);

// Channel shuffles (chapter 09.3).
struct ChannelCandidate {
    bool tme = false;
    bool sprite = true;
    bool t8_format = false;
    std::uint32_t fbmsk = 0;
    std::uint32_t clut_identity = 0;
    std::uint32_t fbp = 0;
    std::uint32_t tbp = 0;
    bool region_repeat_fix_bit3 = false;
    bool region_repeat_mask_bit3_clear = false;
    bool wmt_region_repeat = false;
    bool wms_region_repeat = false;
    bool wms_fix_bit1 = false;
    bool v_bit5 = false;
    bool integer_coords = false;
    std::uint32_t u_mod16 = 0;
    std::uint32_t v_mod4 = 0;
    bool tiny = false;
    std::uint32_t width_pixels = 0;
    bool first_sprite8 = false;
    bool first_sprite2tall = false;
    bool one_page = false;
    bool region_mode = false;
};
bool is_channel_shuffle_candidate(const ChannelCandidate& candidate);
// 0 = R, 1 = G, 2 = B, 3 = A.
int channel_shuffle_channel(const ChannelCandidate& candidate);
// Groups consecutive candidates with the same FBMSK, CLUT identity and advancing bases.
std::vector<std::uint32_t> coalesce_channel_shuffle_runs(const std::vector<ChannelCandidate>& batches);

// Clear detection (chapter 05.8).
ClearKind detect_target_clear(const BatchInput& input, const DeviceCaps& caps, PixelRect& clear_rect,
                              std::uint32_t& colour_word);
ClearKind detect_double_half_clear(const BatchInput& input, PixelRect& clear_rect, std::uint32_t& base_fbp);
ClearKind detect_striped_double_clear(const BatchInput& input, PixelRect& clear_rect, std::uint32_t& base_fbp);

// Half offset of a double buffer, in pages, for a rectangle of rows.
std::uint32_t page_rows_for_format(FrameFormat frame, ZFormat z, bool use_z);

}  // namespace coregs::decide
