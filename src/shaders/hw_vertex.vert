// SPDX-License-Identifier: Apache-2.0
//
// HWR vertex stage (TASK-027, M4). Copyright (c) 2026 GT4Recomp contributors.
//
// Written from docs/hw-renderer-spec chapters 03.3 (vertex record and position mapping) and
// 06.3 (half-pixel shift), clean room (decision 0069).
//
// Position: P = (X - OFX) / 16 in native pixels, with X the 12.4 coordinate that already
// includes XYOFFSET. gpu = scale * P + 0.5 * scale_offset - 1/320 (the up-left bias of 03.3).
// W = 1, so the GPU's linear interpolation equals the GS screen-linear interpolation.
// Built twice: smooth colour (default) and flat colour (-DHWR_FLAT_COLOUR=1, IIP = 0); core 2 builds a third with the
// per-vertex palette row (-DHWR_PALETTE_ROW=1, TASK-168).
// TASK-179: and core 2 builds one more for its points on devices without largePoints (-DHWR_POINT_QUAD=1 with the
// palette row, see below).

#version 450

#ifndef HWR_POINT_QUAD
#define HWR_POINT_QUAD 0
#endif

layout(location = 0) in vec2 in_st;     // record offset 0: S, T
layout(location = 1) in uvec4 in_rgba;   // record offset 8: R, G, B, A (uint8, not normalised)
layout(location = 2) in float in_q;      // record offset 12: Q
layout(location = 3) in uvec2 in_xy;     // record offset 16: X, Y (uint16, 12.4)
layout(location = 4) in uint in_z;       // record offset 20: Z (uint32)
layout(location = 5) in uvec2 in_uv;     // record offset 24: U, V (uint16, 12.4 texels)
layout(location = 6) in float in_fog;    // record offset 28: F (unorm8 in byte 0)
// TASK-168 (core 2, -DHWR_PALETTE_ROW=1): record offset 30, the palette row of the vertex's primitive plus one (0: the
// batch's row), passed flat to the pixel stage.
#ifndef HWR_PALETTE_ROW
#define HWR_PALETTE_ROW 0
#endif
#if HWR_PALETTE_ROW
layout(location = 7) in uint in_palette_row;
layout(location = 6) flat out uint out_palette_row;
#endif

layout(push_constant) uniform VertexPush {
    vec2 viewport_size;  // scaled target size in GPU pixels
    float scale;         // internal resolution scale (COREGS_HW_SCALE)
    float scale_offset;  // half-pixel shift factor (1.0 for native mode at 1x, chapter 06.3)
    int offset_x;        // XYOFFSET X, 12.4 units
    int offset_y;        // XYOFFSET Y, 12.4 units
    uint z_max;          // format maximum of Z (16777215 for Z24, 65535 for Z16, else 4294967295)
    uint tex_scale;      // read by the pixel stage only (IN-374)
} pc;

#if HWR_FLAT_COLOUR
layout(location = 2) flat out vec4 out_rgba;
#else
layout(location = 2) out vec4 out_rgba;
#endif
layout(location = 0) out vec2 out_st;
layout(location = 1) out float out_q;
layout(location = 3) out float out_fog;
layout(location = 4) out vec2 out_uv;
layout(location = 5) noperspective out float out_z;

void main() {
    // X and Y are 12.4 two's complement values stored as uint16: sign-extend before use.
    int xi = int(in_xy.x << 16u) >> 16;
    int yi = int(in_xy.y << 16u) >> 16;
    vec2 native = vec2(xi - pc.offset_x, yi - pc.offset_y) / 16.0;
    const float bias = 1.0 / 320.0;
    vec2 gpu = pc.scale * native + vec2(0.5 * pc.scale_offset) - vec2(bias);
    // IN-388: native alignment for the scaled pass (the reference's the game half-pixel mode). Bit 29 of the tex_scale push
    // word asks for it. A primitive edge at a GS integer coordinate lands at the CENTRE of a native pixel's s x s block
    // with the 06.3 shift, so a quad that covers one native pixel at 1x covers half of two blocks at scale and paints a
    // half-native-pixel sliver at every edge (the owner's thin white lines). Snapping the vertex to the block grid
    // (floor to a multiple of s) makes the coverage of every native pixel all-or-nothing, exactly as at 1x.
    if ((pc.tex_scale & 0x20000000u) != 0u && pc.scale > 1.0) {
        gpu = floor(gpu / pc.scale) * pc.scale;
    }

    // Depth is only used for clipping (no depth attachment in M4); the format clamp is applied
    // per pixel, as the CPU reference does (IN-206).
    float clipped_z = float(in_z);
    float depth = float(min(in_z, pc.z_max)) * (1.0 / 4294967296.0);
#if HWR_POINT_QUAD
    // TASK-179: a point as two triangles, for devices without largePoints (a point wider than one pixel needs it) or with
    // a smaller largest point size than the scale. Drawn with six vertices per instance, the instance being the point's
    // vertex record (VK_VERTEX_INPUT_RATE_INSTANCE, so every corner reads the same record and passes on the same values);
    // gl_VertexIndex (0 to 5) picks the corner. The corners are those of the square point rasterization covers: the
    // centre plus or minus half the point size (pc.scale) on each axis, in GPU pixels before the NDC mapping.
    const vec2 kCorners[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
                                     vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
    gpu += kCorners[gl_VertexIndex % 6] * (0.5 * pc.scale);
#endif
    vec2 ndc = gpu / pc.viewport_size * 2.0 - 1.0;
    gl_Position = vec4(ndc, depth, 1.0);
#if !HWR_POINT_QUAD
    // Points cover s x s sub-pixels at scale (06.5), one pixel at 1x.
    gl_PointSize = pc.scale;
#endif

    out_rgba = vec4(in_rgba);
    out_st = in_st;
    out_q = in_q;
    out_fog = in_fog * 255.0;
    out_uv = vec2(in_uv);
    out_z = clipped_z;
#if HWR_PALETTE_ROW
    out_palette_row = in_palette_row;
#endif
}
