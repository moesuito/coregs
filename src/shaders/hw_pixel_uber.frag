// SPDX-License-Identifier: Apache-2.0
//
// HWR uber pixel shader (TASK-027, M4). Copyright (c) 2026 GT4Recomp contributors.
//
// Written from docs/hw-renderer-spec chapters 03.4, 03.5, 08.8 and 12.6, clean room (decision 0069).
//
// Destination reads use storage images (r32ui frame words and depth words) rather than input
// attachments: the target is written by imageStore in the same invocation after the read, so no
// colour attachment is needed (fragmentStoresAndAtomics, chapter 12.6). Overlap ordering between
// primitives is handled by the caller (one draw per primitive with a barrier between draws).
//
// Texture source (M4, write-back mode): every texel is decoded on the CPU from the shadow local
// memory into RGBA8 (palette and TEXA already applied) and stored as one packed u32 per texel in
// an r32ui image of 2^TW x 2^TH. This shader does the coordinate step (FST, or STQ with the
// per-pixel divide), the wrap modes of 03.5 and nearest or 4-tap bilinear filtering with the
// integer 4-bit weights of the CPU reference (IN-305), then the TFX and fog stages of 03.4.
// Colour is truncated like the CPU reference (IN-304, tolerance IN-500).

#version 450
#extension GL_GOOGLE_include_directive : require
#if HWR_INTERLOCK
// IN-345: with pixel interlock the read-modify-write of the frame and depth words below runs in primitive
// order for overlapping fragments, so a whole batch is one draw with no barriers between its primitives.
#extension GL_ARB_fragment_shader_interlock : require
layout(pixel_interlock_ordered) in;
#endif
#include "hw_pixel_core.glsl"

#if HWR_FLAT_COLOUR
layout(location = 2) flat in vec4 in_rgba;
#else
layout(location = 2) in vec4 in_rgba;
#endif
layout(location = 0) in vec2 in_st;
layout(location = 1) in float in_q;
layout(location = 3) in float in_fog;
layout(location = 4) in vec2 in_uv;
layout(location = 5) noperspective in float in_z;
// TASK-168 (core 2): the palette row of the fragment's primitive plus one, 0 for the batch's row (palette_row below).
#ifndef HWR_PALETTE_ROW
#define HWR_PALETTE_ROW 0
#endif
#if HWR_PALETTE_ROW
layout(location = 6) flat in uint in_palette_row;
#endif

layout(std140, set = 0, binding = 0) uniform PixelStateBlock {
    uint format;
    uint fbmsk;
    uint fba;
    uint dither;
    uint dimx_lo;
    uint dimx_hi;
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
    uint tex_on;
    uint tfx;
    uint tcc;
    uint texel_kind;   // HW_TEXEL_RGBA8 (CPU-expanded) or raw CT16/CT24 copied from a target (IN-346), or
                       // (IN-409, CORE2) HW_TEXEL_INDEXED: a CLUT index, its colour in palette row tex_origin_y
    uint ta0;
    uint ta1;
    uint aem;
    uint fog_on;
    uint fog_r;
    uint fog_g;
    uint fog_b;
    uint tex_fst;      // TEX0.FST: coordinates are UV (10.4) instead of STQ
    uint tex_linear;   // TEX1.MMAG = 1: bilinear
    uint wrap_s;       // WMS: 0 repeat, 1 clamp, 2 region clamp, 3 region repeat
    uint wrap_t;       // WMT
    uint wrap_min_s;   // MINU
    uint wrap_max_s;   // MAXU
    uint wrap_min_t;   // MINV
    uint wrap_max_t;   // MAXV
    uint tex_log2_w;   // TW
    uint tex_log2_h;   // TH
    uint tex_origin_x; // the chunk's region in the texel atlas (IN-352)
    uint tex_origin_y;
    // CORE2: 1 when the frame image is shared by the 32- and 24-bit views of the same memory (one target per
    // memory, CORE2-SPEC 1.3), so a CT24 write keeps the stored top byte as GS memory does. 0 in the old core.
    uint keep_top_byte;
} state;

// IN-437: specialised pipelines. A pipeline may give any of the draw-state fields that choose the code paths
// (formats, tests, blend selectors, texture modes) and the flag bits of the tex_scale push word as specialization
// constants: the compiler then folds every branch on them. The value is the state block's own (the caller only
// zeroes fields the draw cannot reach); kDynamic (the default, the uber pipeline) reads the field at run time.
// Constant id of a field = 1 + its index in the state block (gpufield), so the CPU side maps them one to one.
const uint kDynamic = 0xFFFFFFFFu;
layout(constant_id = 1) const uint SPEC_FORMAT = kDynamic;
layout(constant_id = 3) const uint SPEC_FBA = kDynamic;
layout(constant_id = 4) const uint SPEC_DITHER = kDynamic;
layout(constant_id = 7) const uint SPEC_ATE = kDynamic;
layout(constant_id = 8) const uint SPEC_ATST = kDynamic;
layout(constant_id = 9) const uint SPEC_AFAIL = kDynamic;
layout(constant_id = 11) const uint SPEC_DATE_ON = kDynamic;
layout(constant_id = 12) const uint SPEC_DATM = kDynamic;
layout(constant_id = 13) const uint SPEC_ZTEST = kDynamic;
layout(constant_id = 14) const uint SPEC_ZTST = kDynamic;
layout(constant_id = 15) const uint SPEC_ZMSK = kDynamic;
layout(constant_id = 16) const uint SPEC_ZFORMAT = kDynamic;
layout(constant_id = 17) const uint SPEC_BLEND_ON = kDynamic;
layout(constant_id = 18) const uint SPEC_PABE = kDynamic;
layout(constant_id = 19) const uint SPEC_BLEND_A = kDynamic;
layout(constant_id = 20) const uint SPEC_BLEND_B = kDynamic;
layout(constant_id = 21) const uint SPEC_BLEND_C = kDynamic;
layout(constant_id = 22) const uint SPEC_BLEND_D = kDynamic;
layout(constant_id = 24) const uint SPEC_COLCLAMP = kDynamic;
layout(constant_id = 25) const uint SPEC_TEX_ON = kDynamic;
layout(constant_id = 26) const uint SPEC_TFX = kDynamic;
layout(constant_id = 27) const uint SPEC_TCC = kDynamic;
layout(constant_id = 28) const uint SPEC_TEXEL_KIND = kDynamic;
layout(constant_id = 31) const uint SPEC_AEM = kDynamic;
layout(constant_id = 32) const uint SPEC_FOG_ON = kDynamic;
layout(constant_id = 36) const uint SPEC_TEX_FST = kDynamic;
layout(constant_id = 37) const uint SPEC_TEX_LINEAR = kDynamic;
layout(constant_id = 38) const uint SPEC_WRAP_S = kDynamic;
layout(constant_id = 39) const uint SPEC_WRAP_T = kDynamic;
layout(constant_id = 48) const uint SPEC_KEEP_TOP_BYTE = kDynamic;
layout(constant_id = 100) const uint SPEC_PUSH_WORD = kDynamic;  // tex_scale without bit 29 (vertex stage's snap)
layout(constant_id = 101) const uint SPEC_MIP_ANISO = kDynamic;  // tex_origin_x bits 21-23 of a mipmapped draw
#define ST_FORMAT (SPEC_FORMAT != kDynamic ? SPEC_FORMAT : state.format)
#define ST_FBA (SPEC_FBA != kDynamic ? SPEC_FBA : state.fba)
#define ST_DITHER (SPEC_DITHER != kDynamic ? SPEC_DITHER : state.dither)
#define ST_ATE (SPEC_ATE != kDynamic ? SPEC_ATE : state.ate)
#define ST_ATST (SPEC_ATST != kDynamic ? SPEC_ATST : state.atst)
#define ST_AFAIL (SPEC_AFAIL != kDynamic ? SPEC_AFAIL : state.afail)
#define ST_DATE_ON (SPEC_DATE_ON != kDynamic ? SPEC_DATE_ON : state.date_on)
#define ST_DATM (SPEC_DATM != kDynamic ? SPEC_DATM : state.datm)
#define ST_ZTEST (SPEC_ZTEST != kDynamic ? SPEC_ZTEST : state.ztest)
#define ST_ZTST (SPEC_ZTST != kDynamic ? SPEC_ZTST : state.ztst)
#define ST_ZMSK (SPEC_ZMSK != kDynamic ? SPEC_ZMSK : state.zmsk)
#define ST_ZFORMAT (SPEC_ZFORMAT != kDynamic ? SPEC_ZFORMAT : state.zformat)
#define ST_BLEND_ON (SPEC_BLEND_ON != kDynamic ? SPEC_BLEND_ON : state.blend_on)
#define ST_PABE (SPEC_PABE != kDynamic ? SPEC_PABE : state.pabe)
#define ST_BLEND_A (SPEC_BLEND_A != kDynamic ? SPEC_BLEND_A : state.blend_a)
#define ST_BLEND_B (SPEC_BLEND_B != kDynamic ? SPEC_BLEND_B : state.blend_b)
#define ST_BLEND_C (SPEC_BLEND_C != kDynamic ? SPEC_BLEND_C : state.blend_c)
#define ST_BLEND_D (SPEC_BLEND_D != kDynamic ? SPEC_BLEND_D : state.blend_d)
#define ST_COLCLAMP (SPEC_COLCLAMP != kDynamic ? SPEC_COLCLAMP : state.colclamp)
#define ST_TEX_ON (SPEC_TEX_ON != kDynamic ? SPEC_TEX_ON : state.tex_on)
#define ST_TFX (SPEC_TFX != kDynamic ? SPEC_TFX : state.tfx)
#define ST_TCC (SPEC_TCC != kDynamic ? SPEC_TCC : state.tcc)
#define ST_TEXEL_KIND (SPEC_TEXEL_KIND != kDynamic ? SPEC_TEXEL_KIND : state.texel_kind)
#define ST_AEM (SPEC_AEM != kDynamic ? SPEC_AEM : state.aem)
#define ST_FOG_ON (SPEC_FOG_ON != kDynamic ? SPEC_FOG_ON : state.fog_on)
#define ST_TEX_FST (SPEC_TEX_FST != kDynamic ? SPEC_TEX_FST : state.tex_fst)
#define ST_TEX_LINEAR (SPEC_TEX_LINEAR != kDynamic ? SPEC_TEX_LINEAR : state.tex_linear)
#define ST_WRAP_S (SPEC_WRAP_S != kDynamic ? SPEC_WRAP_S : state.wrap_s)
#define ST_WRAP_T (SPEC_WRAP_T != kDynamic ? SPEC_WRAP_T : state.wrap_t)
#define ST_KEEP_TOP_BYTE (SPEC_KEEP_TOP_BYTE != kDynamic ? SPEC_KEEP_TOP_BYTE : state.keep_top_byte)
#define PUSH_WORD (SPEC_PUSH_WORD != kDynamic ? SPEC_PUSH_WORD : pc.tex_scale)
#define ST_MIP_ANISO (SPEC_MIP_ANISO != kDynamic ? SPEC_MIP_ANISO : (state.tex_origin_x >> 21) & 7u)

#ifndef HWR_ROP
#define HWR_ROP 0
#endif
#if HWR_ROP
// IN-481: a draw that reads nothing back writes through colour attachments (byte views of the same words), with the
// lanes it keeps masked by the pipeline: the GPU's own output path instead of storage writes. The word is exact: a byte
// divided by 255 comes back as that byte.
// IN-482: HWR_ROP=2, a blend without reads: the second output of the first attachment carries the blend factor, and the
// GPU's own blending does the rest (performance mode: it rounds where the GS truncates).
#if HWR_ROP == 2
layout(location = 0, index = 0) out vec4 hw_out_frame;
layout(location = 0, index = 1) out vec4 hw_out_factor;
#else
layout(location = 0) out vec4 hw_out_frame;
layout(location = 1) out vec4 hw_out_depth;
#endif
#endif
#if HWR_INTERLOCK
// Inside the interlock the next fragment of the same pixel must see this one's stores: coherent images.
layout(set = 0, binding = 1, r32ui) coherent uniform uimage2D hw_frame;
layout(set = 0, binding = 2, r32ui) coherent uniform uimage2D hw_depth;
#else
layout(set = 0, binding = 1, r32ui) uniform uimage2D hw_frame;
layout(set = 0, binding = 2, r32ui) uniform uimage2D hw_depth;
#endif
layout(set = 0, binding = 3, r32ui) uniform readonly uimage2D hw_texels;

// IN-384 follow-up: set from the top bit of the tex_scale push word (the GPU drawer's COREGS_HW_SCALED_FILTER=linear).
bool g_force_linear = false;
uint g_mip_aniso = 0u;    // IN-434: log2 of the most anisotropic samples (0: off), tex_origin_x bits 21-23
vec2 g_tex_dx = vec2(0.0), g_tex_dy = vec2(0.0);  // IN-434: texel coordinate per screen pixel in x and y
uint g_tex_scale = 1u;  // PUSH_WORD without the filter/dither bits
// IN-427 (CORE2): bit 24 says the texture is a target sampled at binding 4, whatever the scale. At 1x the scale alone
// (1) also meant "the 1x texel atlas", so core 2 read the atlas instead of the target and the game's scene copies came out
// black (only the HUD showed).
bool g_texture_from_target = false;
// IN-428 (CORE2, C1): bit 23 says tex_origin_x holds the mip parameters of a memory texture whose levels 1..MXL sit
// right of level 0 in the same image (level n >= 1 at x = 2^TW, y = 2^TH - (2^TH >> (n - 1)), max(1, size >> n)):
// bits 0-2 MXL, 3-5 MMIN, 6 LCM, 7-8 L, 9-20 K (signed, 4 fraction bits). Spec 3.6 "full" mode, base level 0.
bool g_mip = false;
uint g_mip_mxl = 0u;
uint g_mip_mmin = 0u;
bool g_mip_lcm = false;
uint g_mip_l = 0u;
int g_mip_k = 0;
// The reference indexes the dither matrix with the raw fragment
// coordinate (one cell = 4x4 OUTPUT pixels) and only offers the per-native-pixel pattern as an option; ours always
// divided by the scale, so one cell covered s x s output pixels (24x24 at 6x, doubled vertically on a 480i field:
// the owner's 6 x 12 lattice). Fine is the default at scale > 1; native stays available for fidelity checks.
bool g_dither_fine = true;

// The vertex stage's push constants (hw_vertex.vert), read here for the internal scale (IN-371): the native dither
// mode of 06.5 indexes DIMX by the native pixel, so each dither cell covers s x s sub-pixels.
layout(push_constant) uniform VertexPush {
    vec2 viewport_size;
    float scale;
    float scale_offset;
    int offset_x;
    int offset_y;
    uint z_max;
    uint tex_scale;  // IN-374: 1 = the texel atlas (1x texels); s = hw_texels_scaled holds the texture at scale s
} pc;

// IN-374 (06.8): a texture copied from a resident target's scaled image, s x s texels per native texel.
layout(set = 0, binding = 4, r32ui) uniform readonly uimage2D hw_texels_scaled;

#ifndef HWR_EARLY_Z
#define HWR_EARLY_Z 0
#endif
#if HWR_EARLY_Z
// IN-451 (core 2 variants only): speculative depth rejection before the texture is sampled (constant 103, 0 with
// COREGS_CORE2_EARLY_Z=0). The speculative word is read through binding 5, a second descriptor of the depth image:
// read through hw_depth itself, the compiler was free to reuse that value for the load inside the interlock (no store
// in between in this invocation), so the depth test saw a stale word (frames differed). A load through another
// descriptor cannot stand in for it.
layout(constant_id = 103) const uint SPEC_EARLY_Z = 1u;
layout(set = 0, binding = 5, r32ui) uniform readonly uimage2D hw_depth_peek;
#endif

// CPU ToTexel104: floor of the scaled coordinate, saturated far outside the 1024 range.
int to_texel_104(float value) {
    return int(clamp(floor(value), -1.0e9, 1.0e9));
}

// pixel::wrap_texel for one axis (mode values are the GS WMS/WMT encoding).
int wrap_texel(int texel, uint log2_size, uint mode, uint minimum, uint maximum) {
    int last = int((1u << log2_size) - 1u);
    if (mode == 0u) {
        return texel & last;
    }
    if (mode == 1u) {
        return clamp(texel, 0, last);
    }
    if (mode == 2u) {
        return clamp(texel, int(minimum), int(maximum));
    }
    return (texel & int(minimum)) | int(maximum);
}

// Texel colour: RGBA8 words decoded on the CPU, or (IN-346) raw CT16/CT24 words copied from a resident target
// on the GPU, expanded here with TEXA exactly as the CPU decode does (hw_texel_colour, verified bit-exact).
//
// IN-409 (CORE2): an indexed memory texture keeps its CLUT indices; the colour is the RGBA8 entry of palette row
// tex_origin_y (TASK-168: palette_row()), an image bound where a target texture would be (binding 4, unused by memory
// textures). A palette that changes then never needs the texture decoded again, and bilinear filtering blends looked-up
// colours, as the GS does.
//
// TASK-168: the palette row of this fragment's primitive. A batch whose primitives use several palettes (CLUT loads that
// only changed the palette of its texture) gives each vertex its primitive's row plus one; 0 keeps the batch's row.
int palette_row() {
#if HWR_PALETTE_ROW
    return int(in_palette_row != 0u ? in_palette_row - 1u : state.tex_origin_y);
#else
    return int(state.tex_origin_y);
#endif
}

ivec4 fetch_texel(int x, int y) {
    if (ST_TEXEL_KIND == HW_TEXEL_INDEXED) {
        uint index = imageLoad(hw_texels, ivec2(x, y)).x & 255u;
        uint entry = imageLoad(hw_texels_scaled, ivec2(int(index), palette_row())).x;
        return hw_texel_colour(HW_TEXEL_RGBA8, entry, state.ta0, state.ta1, ST_AEM != 0u);
    }
    uint raw = imageLoad(hw_texels, ivec2(x + int(state.tex_origin_x), y + int(state.tex_origin_y))).x;
    return hw_texel_colour(ST_TEXEL_KIND, raw, state.ta0, state.ta1, ST_AEM != 0u);
}

// IN-374: wrap of a scaled texel coordinate. The native texel (floor division by s) takes the 03.5 wrap of its
// axis; the sub-texel is kept for the repeat modes and pinned to the clamped edge for the clamp modes.
int wrap_scaled(int texel, int s, uint log2_size, uint mode, uint minimum, uint maximum) {
    int native = texel >= 0 ? texel / s : -((s - 1 - texel) / s);
    int sub = texel - native * s;
    int wrapped = wrap_texel(native, log2_size, mode, minimum, maximum);
    if ((mode == 1u || mode == 2u) && wrapped != native) {
        sub = native > wrapped ? s - 1 : 0;
    }
    return wrapped * s + sub;
}

ivec4 fetch_texel_scaled(int x, int y, int s) {
    // IN-384 follow-up: the scaled texel image holds exactly this texture at (0, 0) (the copy in Execute places it
    // there, sized to the sampled footprint), so the 1x atlas origin must NOT be added here: adding it made every
    // texture after the first in a run sample the wrong region (the leaking cuts on the car at 6x).
    uint raw = imageLoad(hw_texels_scaled, ivec2(x, y)).x;
    return hw_texel_colour(ST_TEXEL_KIND, raw, state.ta0, state.ta1, ST_AEM != 0u);
}

// IN-428: one texel of mip level `level` (0..MXL) of a memory texture laid out as described at g_mip.
ivec4 fetch_level(int level, int x, int y) {
    int ox = 0;
    int oy = 0;
    if (level > 0) {
        int h = 1 << int(state.tex_log2_h);
        ox = 1 << int(state.tex_log2_w);
        oy = h - (h >> (level - 1));
    }
    if (ST_TEXEL_KIND == HW_TEXEL_INDEXED) {
        uint index = imageLoad(hw_texels, ivec2(x + ox, y + oy)).x & 255u;
        uint entry = imageLoad(hw_texels_scaled, ivec2(int(index), palette_row())).x;
        return hw_texel_colour(HW_TEXEL_RGBA8, entry, state.ta0, state.ta1, ST_AEM != 0u);
    }
    uint raw = imageLoad(hw_texels, ivec2(x + ox, y + oy)).x;
    return hw_texel_colour(ST_TEXEL_KIND, raw, state.ta0, state.ta1, ST_AEM != 0u);
}

// IN-428: SampleTexture on mip level `level`: coordinates, sizes and clamp bounds halved once per level (03.6).
ivec4 sample_level(int level, int uvs, int uvt, bool linear) {
    uint lw = uint(max(int(state.tex_log2_w) - level, 0));
    uint lh = uint(max(int(state.tex_log2_h) - level, 0));
    uint min_s = state.wrap_min_s >> level, max_s = state.wrap_max_s >> level;
    uint min_t = state.wrap_min_t >> level, max_t = state.wrap_max_t >> level;
    int s = uvs >> level;
    int t = uvt >> level;
    if (!linear) {
        int x = wrap_texel(s >> 4, lw, ST_WRAP_S, min_s, max_s);
        int y = wrap_texel(t >> 4, lh, ST_WRAP_T, min_t, max_t);
        return fetch_level(level, x, y);
    }
    int bs = s - 8;
    int bt = t - 8;
    int x0 = wrap_texel(bs >> 4, lw, ST_WRAP_S, min_s, max_s);
    int x1 = wrap_texel((bs >> 4) + 1, lw, ST_WRAP_S, min_s, max_s);
    int y0 = wrap_texel(bt >> 4, lh, ST_WRAP_T, min_t, max_t);
    int y1 = wrap_texel((bt >> 4) + 1, lh, ST_WRAP_T, min_t, max_t);
    int fx = bs & 15;
    int fy = bt & 15;
    ivec4 c00 = fetch_level(level, x0, y0);
    ivec4 c10 = fetch_level(level, x1, y0);
    ivec4 c01 = fetch_level(level, x0, y1);
    ivec4 c11 = fetch_level(level, x1, y1);
    return (c00 * ((16 - fx) * (16 - fy)) + c10 * (fx * (16 - fy)) + c01 * ((16 - fx) * fy) + c11 * (fx * fy)) >> 8;
}

// IN-428: the GS LOD (01.5): LCM = 0 gives log2(1/|Q|) * 2^L + K/16, LCM = 1 (or FST) K/16. LOD <= 0 uses MMAG at
// level 0; otherwise MMIN: 2/4 the nearest level (round half up), 3/5 a blend of the two levels around the LOD
// (weight in 1/256 steps, Hypothesis); 0/1 level 0. Levels clamp to [0, MXL].
// MMIN 2-5 at a positive LOD: the level or levels around it.
ivec4 sample_at_lod(int uvs, int uvt, float lod) {
    float top = float(g_mip_mxl);
    lod = min(lod, top);
    bool linear_in_level = g_mip_mmin >= 4u || g_force_linear;  // IN-433
    if (g_mip_mmin == 2u || g_mip_mmin == 4u) {
        int level = int(min(floor(lod + 0.5), top));
        return sample_level(level, uvs, uvt, linear_in_level);
    }
    int level0 = int(floor(lod));
    int level1 = min(level0 + 1, int(g_mip_mxl));
    int weight = int(floor((lod - float(level0)) * 256.0));
    ivec4 a = sample_level(level0, uvs, uvt, linear_in_level);
    if (level1 == level0 || weight == 0) {
        return a;
    }
    ivec4 b = sample_level(level1, uvs, uvt, linear_in_level);
    return (a * (256 - weight) + b * weight) >> 8;
}

// IN-434 (C5, an enhancement the GS does not have; off by default): anisotropic filtering of mipmapped draws. The
// pixel's texture footprint (screen derivatives of the texel coordinate, g_tex_dx / g_tex_dy) is long along one axis
// on surfaces seen at a grazing angle (the road ahead); the GS LOD from Q then picks a level blurred along the short
// axis too. With up to N = 2^g_mip_aniso samples spread along the long axis, the LOD drops by log2(samples), so the
// short axis keeps its detail.
ivec4 sample_mipmapped(int uvs, int uvt, float q) {
    float lod = float(g_mip_k) / 16.0;
    if (!g_mip_lcm && ST_TEX_FST == 0u) {
        float magnitude = abs(q);
        lod = magnitude > 0.0 ? lod - log2(magnitude) * float(1u << g_mip_l) : 1.0e9;
    }
    if (lod <= 0.0 || g_mip_mmin < 2u) {
        bool linear = g_force_linear || (lod <= 0.0 ? ST_TEX_LINEAR != 0u : (g_mip_mmin & 1u) != 0u);
        return sample_level(0, uvs, uvt, linear);
    }
    if (g_mip_aniso != 0u && lod < 1.0e8) {
        float lx = length(g_tex_dx);
        float ly = length(g_tex_dy);
        float major = max(lx, ly);
        float minor = max(min(lx, ly), 1.0e-6);
        int samples = int(min(ceil(major / minor), float(1u << g_mip_aniso)));
        if (samples > 1) {
            vec2 axis = (lx >= ly ? g_tex_dx : g_tex_dy) * 16.0;  // the long axis in 1/16 texels
            float reduced = max(lod - log2(float(samples)), 0.0);
            ivec4 sum = ivec4(0);
            for (int i = 0; i < samples; ++i) {
                vec2 offset = axis * ((float(i) + 0.5) / float(samples) - 0.5);
                int su = uvs + int(round(offset.x));
                int sv = uvt + int(round(offset.y));
                sum += reduced <= 0.0 ? sample_level(0, su, sv, true) : sample_at_lod(su, sv, reduced);
            }
            return sum / samples;
        }
    }
    return sample_at_lod(uvs, uvt, lod);
}

// The texture at s x s texels per native texel: the coordinate times s, the same 4-bit weights.
ivec4 sample_texture_scaled(int uvs, int uvt, int s) {
    if (ST_TEX_LINEAR == 0u && !g_force_linear) {
        int x = wrap_scaled((uvs * s) >> 4, s, state.tex_log2_w, ST_WRAP_S, state.wrap_min_s, state.wrap_max_s);
        int y = wrap_scaled((uvt * s) >> 4, s, state.tex_log2_h, ST_WRAP_T, state.wrap_min_t, state.wrap_max_t);
        return fetch_texel_scaled(x, y, s);
    }
    int bs = uvs * s - 8;
    int bt = uvt * s - 8;
    int x0 = wrap_scaled(bs >> 4, s, state.tex_log2_w, ST_WRAP_S, state.wrap_min_s, state.wrap_max_s);
    int x1 = wrap_scaled((bs >> 4) + 1, s, state.tex_log2_w, ST_WRAP_S, state.wrap_min_s, state.wrap_max_s);
    int y0 = wrap_scaled(bt >> 4, s, state.tex_log2_h, ST_WRAP_T, state.wrap_min_t, state.wrap_max_t);
    int y1 = wrap_scaled((bt >> 4) + 1, s, state.tex_log2_h, ST_WRAP_T, state.wrap_min_t, state.wrap_max_t);
    int fx = bs & 15;
    int fy = bt & 15;
    ivec4 c00 = fetch_texel_scaled(x0, y0, s);
    ivec4 c10 = fetch_texel_scaled(x1, y0, s);
    ivec4 c01 = fetch_texel_scaled(x0, y1, s);
    ivec4 c11 = fetch_texel_scaled(x1, y1, s);
    int w00 = (16 - fx) * (16 - fy);
    int w10 = fx * (16 - fy);
    int w01 = (16 - fx) * fy;
    int w11 = fx * fy;
    return (c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11) >> 8;
}

// SampleTexture of the CPU reference: nearest, or bilinear with the 4-bit weights (IN-305).
ivec4 sample_texture(int uvs, int uvt) {
    // IN-384 follow-up (owner test 11): COREGS_HW_SCALED_FILTER=linear forces the bilinear path for a texture the game
    // asks to sample nearest, but only at an internal scale > 1 and only for the 1x texel atlas. A 1x texture
    // magnified 6x with nearest shows the source's 6-pixel steps (the menu background grid, the map's dashes); the
    // reference renderer smooths them with its upscaling filter. Off by default (faithful to the GS).
    if (ST_TEX_LINEAR == 0u && !(g_force_linear && g_tex_scale <= 1u)) {
        int x = wrap_texel(uvs >> 4, state.tex_log2_w, ST_WRAP_S, state.wrap_min_s, state.wrap_max_s);
        int y = wrap_texel(uvt >> 4, state.tex_log2_h, ST_WRAP_T, state.wrap_min_t, state.wrap_max_t);
        return fetch_texel(x, y);
    }
    int bs = uvs - 8;
    int bt = uvt - 8;
    int x0 = wrap_texel(bs >> 4, state.tex_log2_w, ST_WRAP_S, state.wrap_min_s, state.wrap_max_s);
    int x1 = wrap_texel((bs >> 4) + 1, state.tex_log2_w, ST_WRAP_S, state.wrap_min_s, state.wrap_max_s);
    int y0 = wrap_texel(bt >> 4, state.tex_log2_h, ST_WRAP_T, state.wrap_min_t, state.wrap_max_t);
    int y1 = wrap_texel((bt >> 4) + 1, state.tex_log2_h, ST_WRAP_T, state.wrap_min_t, state.wrap_max_t);
    int fx = bs & 15;
    int fy = bt & 15;
    ivec4 c00 = fetch_texel(x0, y0);
    ivec4 c10 = fetch_texel(x1, y0);
    ivec4 c01 = fetch_texel(x0, y1);
    ivec4 c11 = fetch_texel(x1, y1);
    int w00 = (16 - fx) * (16 - fy);
    int w10 = fx * (16 - fy);
    int w01 = (16 - fx) * fy;
    int w11 = fx * fy;
    return (c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11) >> 8;
}

// IN-470 (chapter 11): a replacement image, any size, covers the native rectangle of the texture (2^TW x 2^TH or its
// region), so a native coordinate maps onto it by the ratio of the sizes. The wrap modes work on native coordinates as
// the GS does; the game's filter choice applies to the image's own texels. RGBA8 words, alpha on the PS2 scale. Fields:
// tex_origin_x = image width | height << 16, tex_origin_y = rectangle width | height << 16, ta0 = its x | y << 16.
float wrap_replaced(float u, uint log2_size, uint mode, uint minimum, uint maximum) {
    float size = float(1u << log2_size);
    if (mode == 0u) return u - floor(u / size) * size;
    if (mode == 1u) return clamp(u, 0.0, size - 0.0001);
    if (mode == 2u) return clamp(u, float(minimum), float(maximum + 1u) - 0.0001);
    uint whole = uint(int(floor(u))) & ((1u << log2_size) - 1u);
    return float((whole & minimum) | maximum) + fract(u);
}

ivec4 fetch_replaced(int x, int y) {
    return hw_texel_colour(HW_TEXEL_RGBA8, imageLoad(hw_texels, ivec2(x, y)).x, 0u, 0u, false);
}

ivec4 sample_replaced() {
    vec2 native;
    if (ST_TEX_FST != 0u) {
        native = in_uv / 16.0;
    } else {
        float q = in_q == 0.0 ? 1.0 : in_q;
        native = (in_st / q) * vec2(float(1u << state.tex_log2_w), float(1u << state.tex_log2_h));
    }
    const ivec2 image_size = ivec2(int(state.tex_origin_x & 0xFFFFu), int(state.tex_origin_x >> 16));
    const vec2 rect_size = vec2(float(state.tex_origin_y & 0xFFFFu), float(state.tex_origin_y >> 16));
    const vec2 rect_origin = vec2(float(state.ta0 & 0xFFFFu), float(state.ta0 >> 16));
    const vec2 wrapped = vec2(wrap_replaced(native.x, state.tex_log2_w, ST_WRAP_S, state.wrap_min_s, state.wrap_max_s),
                              wrap_replaced(native.y, state.tex_log2_h, ST_WRAP_T, state.wrap_min_t, state.wrap_max_t));
    const vec2 at = (wrapped - rect_origin) / rect_size * vec2(image_size);
    if (ST_TEX_LINEAR == 0u && !g_force_linear) {
        const ivec2 texel = clamp(ivec2(floor(at)), ivec2(0), image_size - 1);
        return fetch_replaced(texel.x, texel.y);
    }
    const vec2 corner = at - 0.5;
    const ivec2 low = ivec2(floor(corner));
    const ivec2 weight = ivec2((corner - vec2(low)) * 16.0);
    // A repeating axis continues at the other side of the image, the others stop at its edge.
    const bool repeat_s = ST_WRAP_S == 0u || ST_WRAP_S == 3u, repeat_t = ST_WRAP_T == 0u || ST_WRAP_T == 3u;
    const int x0 = repeat_s ? (low.x + image_size.x) % image_size.x : clamp(low.x, 0, image_size.x - 1);
    const int x1 = repeat_s ? (low.x + 1 + image_size.x) % image_size.x : clamp(low.x + 1, 0, image_size.x - 1);
    const int y0 = repeat_t ? (low.y + image_size.y) % image_size.y : clamp(low.y, 0, image_size.y - 1);
    const int y1 = repeat_t ? (low.y + 1 + image_size.y) % image_size.y : clamp(low.y + 1, 0, image_size.y - 1);
    return (fetch_replaced(x0, y0) * ((16 - weight.x) * (16 - weight.y)) + fetch_replaced(x1, y0) * (weight.x * (16 - weight.y)) +
            fetch_replaced(x0, y1) * ((16 - weight.x) * weight.y) + fetch_replaced(x1, y1) * (weight.x * weight.y)) >> 8;
}

// The texture function's colour for the fragment colour `frag` at texel coordinate (uvs, uvt) (12.4 units).
ivec4 textured_colour(int uvs, int uvt, ivec4 frag) {
    ivec4 texel = (PUSH_WORD & 0x00400000u) != 0u ? sample_replaced() : g_mip ? sample_mipmapped(uvs, uvt, in_q)
                  : (g_tex_scale > 1u || g_texture_from_target) ? sample_texture_scaled(uvs, uvt, int(max(g_tex_scale, 1u)))
                                                                 : sample_texture(uvs, uvt);
    if ((PUSH_WORD & 0x04000000u) != 0u) texel = ivec4(128, 128, 128, 128);  // diagnostic: COREGS_HW_SCALED_DIAG=notex
    return hw_texture_function(ST_TFX, ST_TCC != 0u, texel, frag);
}

// Fog: the colour blended with FOGCOL by the interpolated fog value (truncated, IN-304).
ivec4 fogged(ivec4 colour) {
    int fog = int(clamp(floor(clamp(in_fog, 0.0, 255.0) + 0.001), 0.0, 255.0));
    ivec3 fog_colour = ivec3(int(state.fog_r), int(state.fog_g), int(state.fog_b));
    colour.r = (fog * colour.r + (255 - fog) * fog_colour.r) >> 8;
    colour.g = (fog * colour.g + (255 - fog) * fog_colour.g) >> 8;
    colour.b = (fog * colour.b + (255 - fog) * fog_colour.b) >> 8;
    return colour;
}

void main() {
    ivec2 pixel = ivec2(floor(gl_FragCoord.xy));
    // IN-530: SCANMSK (bits 20-21 of the push word): 2 leaves out even rows, 3 odd rows. The game draws the time trial ghost this way.
    // The rows are output rows, as on the reference renderer at scale: native rows make the ghost look shredded.
    const uint scan_mask = (PUSH_WORD >> 20) & 3u;
    if (scan_mask >= 2u && (uint(pixel.y) & 1u) == (scan_mask & 1u)) discard;
    // IN-384 follow-up: the top bit of the tex_scale push word requests the forced linear filter for 1x textures.
    g_force_linear = (PUSH_WORD & 0x80000000u) != 0u;
    g_dither_fine = (PUSH_WORD & 0x40000000u) != 0u;
    // IN-388: bit 29 is the vertex stage's native-alignment snap, not part of the scale; IN-392: bit 28 asks for the
    // position depth below.
    g_tex_scale = PUSH_WORD & 0x000FFFFFu;  // bits 27/26/25: COREGS_HW_SCALED_DIAG=noalpha/notex/noz (diagnostic)
    g_texture_from_target = (PUSH_WORD & 0x01000000u) != 0u;
    g_mip = (PUSH_WORD & 0x00800000u) != 0u;
    if (g_mip) {
        uint info = state.tex_origin_x;
        g_mip_mxl = info & 7u;
        g_mip_mmin = (info >> 3) & 7u;
        g_mip_lcm = ((info >> 6) & 1u) != 0u;
        g_mip_l = (info >> 7) & 3u;
        g_mip_k = int(info << 11) >> 20;  // bits 9-20, sign extended
        g_mip_aniso = ST_MIP_ANISO;
    }
    const bool depth_from_position = (PUSH_WORD & 0x10000000u) != 0u;

    HwPixelState s;
    s.format = ST_FORMAT;
    s.fbmsk = state.fbmsk;
    s.fba = ST_FBA;
    s.dither = ST_DITHER;
    s.dimx_lo = state.dimx_lo;
    s.dimx_hi = state.dimx_hi;
    s.ate = ((PUSH_WORD & 0x08000000u) != 0u && ST_AFAIL == 0u && ST_ATST != 0u) ? 0u : ST_ATE;
    s.atst = ST_ATST;
    s.afail = ST_AFAIL;
    s.aref = state.aref;
    s.date_on = ST_DATE_ON;
    s.datm = ST_DATM;
    s.ztest = ST_ZTEST;
    s.ztst = (PUSH_WORD & 0x02000000u) != 0u && ST_ZTST >= 2u ? 1u : ST_ZTST;  // diagnostic: COREGS_HW_SCALED_DIAG=noz
    s.zmsk = ST_ZMSK;
    s.zformat = ST_ZFORMAT;
    s.blend_on = ST_BLEND_ON;
    s.pabe = ST_PABE;
    s.blend_a = ST_BLEND_A;
    s.blend_b = ST_BLEND_B;
    s.blend_c = ST_BLEND_C;
    s.blend_d = ST_BLEND_D;
    s.fix = state.fix;
    s.colclamp = ST_COLCLAMP;

    // Colour and fog: truncation of the interpolated 0..255 values (IN-304). The 0.001 tolerance
    // absorbs float interpolation noise on exact integer vertices (IN-500).
    ivec4 frag = ivec4(floor(clamp(in_rgba, 0.0, 255.0) + vec4(0.001)));
    ivec4 colour = frag;
    // IN-392: at scale, the rasteriser's depth (the vertex stage writes Z * 2^-32, clamped to the format maximum)
    // floored to an integer, as the reference does: identical vertices give bit-identical depth in every pass
    // (position invariance), so a GEQUAL tie between passes holds at every sub-pixel. 1x keeps the varying.
    const uint frag_z = depth_from_position ? uint(min(floor(gl_FragCoord.z * 4294967296.0), 4294967040.0))
                                            : uint(clamp(in_z, 0.0, 4294967040.0));
    // IN-451: a fragment that fails the depth test writes nothing, whatever its texel. For GEQUAL and GREATER the
    // stored depth is read once without the interlock: when the fragment fails against it, the texel is not sampled
    // here. The outcome is still decided inside the interlock with the current word; if the fragment passes there
    // (a write between the two reads lowered the stored depth), the texel is sampled then. The image is never
    // changed by this read, so every pixel is exactly what it was.
    bool texel_deferred = false;
    int uvs = 0;
    int uvt = 0;
#if HWR_EARLY_Z
    // IN-503: the road's sheen (Cs Ad + Cd) in the plain variant: without destination alpha the colour stays.
    if (SPEC_EARLY_Z == 3u && (imageLoad(hw_frame, pixel).x >> 24) == 0u) discard;
#endif
    if (ST_TEX_ON != 0u) {
        if (ST_TEX_FST != 0u) {
            uvs = to_texel_104(in_uv.x);
            uvt = to_texel_104(in_uv.y);
        } else {
            // Per-pixel divide by Q; a zero Q is treated as 1 (IN-310).
            float q = in_q == 0.0 ? 1.0 : in_q;
            float scale_s = float(1u << state.tex_log2_w) * 16.0;
            float scale_t = float(1u << state.tex_log2_h) * 16.0;
            uvs = to_texel_104((in_st.x / q) * scale_s);
            uvt = to_texel_104((in_st.y / q) * scale_t);
            // IN-434: derivatives in uniform control flow (tex_on, tex_fst and the mip parameters are draw-wide), in
            // texels, only for anisotropic filtering. IN-437: with the derivatives computed for every STQ draw, the
            // pipeline specialised to TEX_FST = 0 moved some pixels of a mipmapped draw without anisotropy to
            // level 0 (hw_core2_mipmaps); they are now compiled only into anisotropic variants.
            if (g_mip && g_mip_aniso != 0u) {
                vec2 texel_coord = (in_st / q) * vec2(float(1u << state.tex_log2_w), float(1u << state.tex_log2_h));
                g_tex_dx = dFdx(texel_coord);
                g_tex_dy = dFdy(texel_coord);
            }
        }
#if HWR_EARLY_Z
        if (SPEC_EARLY_Z != 0u && s.ztest != 0u && s.ztst >= HW_DEPTH_GEQUAL) {
            const uint z = hw_clamp_z(frag_z, s.zformat);
            texel_deferred = !hw_depth_passes(s.ztst, z, imageLoad(hw_depth_peek, pixel).x, s.zformat);
            // IN-492: with GEQUAL or GREATER a stored depth only ever rises while nothing lowers it (Gpu::Draw puts a
            // barrier before such a batch when something may have), so a fragment that fails against an older word
            // also fails against the current one: it writes nothing, and need not wait for the interlock.
            if (texel_deferred && SPEC_EARLY_Z == 2u) discard;
        }
#endif
        if (!texel_deferred) colour = textured_colour(uvs, uvt, frag);
    }
#if HWR_EARLY_Z
    else if (SPEC_EARLY_Z == 2u && s.ztest != 0u && s.ztst >= HW_DEPTH_GEQUAL &&
             !hw_depth_passes(s.ztst, hw_clamp_z(frag_z, s.zformat), imageLoad(hw_depth_peek, pixel).x, s.zformat)) {
        discard;  // IN-492
    }
#endif
    if (ST_FOG_ON != 0u && !texel_deferred) colour = fogged(colour);
#if HWR_EARLY_Z
    // IN-498: a fragment the alpha test stops from writing anything (AFAIL KEEP: fences, leaves, cut-out edges) need not
    // wait for the interlock either. Fog does not change alpha, so this is the test hw_process_pixel would make.
    if (SPEC_EARLY_Z == 2u && !texel_deferred) {
        const HwWrite alpha_write = hw_alpha_test_write(s.ate, s.atst, s.afail, int(s.aref), colour.a, s.format);
        if (!(alpha_write.rgb || alpha_write.alpha || alpha_write.depth)) discard;
    }
#endif

    HwFragment f;
    // Dither index. IN-384 follow-up: the reference's default at scale is the raw fragment coordinate (4x4 output
    // pixels per cell); the 06.5 native mode (one cell per s x s block) is the optional one. At 1x both are identical.
    uint pixel_scale = max(uint(pc.scale + 0.5), 1u);
    f.x = g_dither_fine ? uint(pixel.x) : uint(pixel.x) / pixel_scale;
    f.y = g_dither_fine ? uint(pixel.y) : uint(pixel.y) / pixel_scale;
    f.colour = colour;
    f.z = frag_z;

    // IN-438: the old words are read only when the outcome depends on them (the passes are bound by memory traffic at
    // high scales). The depth: a test other than ALWAYS, or a Z24 / Z16 write (hw_write_depth_raw keeps the bits
    // above the Z; zformat 0 is Z32). The frame: a blend, the destination alpha test, FBMSK bits in the format's
    // lanes, the alpha test's RGB_ONLY on CT32 (the alpha lane kept) or the CT24 top byte kept. Otherwise the written
    // word replaces every lane hw_pack_frame_word keeps, and a word that is not read is never written back, so 0
    // stands in for it. In a specialised pipeline (IN-437) both conditions are constants but FBMSK.
    const bool depth_read = s.ztest != 0u && (s.ztst != HW_DEPTH_ALWAYS || (s.zmsk == 0u && s.zformat != 0u));
    const bool frame_read = s.blend_on != 0u || s.date_on != 0u || (s.fbmsk & hw_valid_mask(s.format)) != 0u ||
                            (s.ate != 0u && s.afail == HW_ALPHA_FAIL_RGB_ONLY && s.format == HW_FORMAT_CT32) ||
                            (ST_KEEP_TOP_BYTE != 0u && s.format == HW_FORMAT_CT24);
#if HWR_INTERLOCK
    beginInvocationInterlockARB();
#endif
#if HWR_ROP
    const uint frame_old = 0u, depth_old = 0u;
#else
    uint frame_old = frame_read ? imageLoad(hw_frame, pixel).x : 0u;
    uint depth_old = depth_read ? imageLoad(hw_depth, pixel).x : 0u;
#endif
    // IN-451: the speculation said "fails"; with the current word the fragment may pass after all (then it needs its
    // texel now). If it fails, hw_process_pixel discards it at the depth test before the colour matters.
    if (texel_deferred && hw_depth_passes(s.ztst, hw_clamp_z(f.z, s.zformat), depth_old, s.zformat)) {
        f.colour = textured_colour(uvs, uvt, frag);
        if (ST_FOG_ON != 0u) f.colour = fogged(f.colour);
    }
#if HWR_ROP == 2
    s.blend_on = 0u;
    hw_out_factor = vec4(float(min(s.blend_c == HW_BLEND_FACTOR_AS ? f.colour.a : int(s.fix), 128)) / 128.0);
#endif
    HwOutcome o = hw_process_pixel(s, f, frame_old, depth_old);
#if HWR_ROP
    if (!o.frame_written && !o.depth_written) discard;
    hw_out_frame = unpackUnorm4x8(o.frame_stored);
#if HWR_ROP == 1
    hw_out_depth = unpackUnorm4x8(o.depth_stored);
#endif
#else
    if (o.frame_written) {
        if (ST_KEEP_TOP_BYTE != 0u && ST_FORMAT == HW_FORMAT_CT24) {
            o.frame_stored = (o.frame_stored & 0x00FFFFFFu) | (frame_old & 0xFF000000u);
        }
        imageStore(hw_frame, pixel, uvec4(o.frame_stored));
    }
    if (o.depth_written) {
        imageStore(hw_depth, pixel, uvec4(o.depth_stored));
    }
#endif
#if HWR_INTERLOCK
    endInvocationInterlockARB();
#endif
}
