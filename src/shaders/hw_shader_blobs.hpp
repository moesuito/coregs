// SPDX-License-Identifier: Apache-2.0
//
// Embedded SPIR-V of the HWR shaders (TASK-027, M4). The arrays are generated at build time by
// src/shaders/CMakeLists.txt from the GLSL sources, compiled by the pinned glslang
// (implementation notes IN-201). Copyright (c) 2026 GT4Recomp contributors.

#pragma once

#include <cstddef>
#include <cstdint>

namespace coregs::shaders {

// Vertex stage, smooth colour (IIP = 1) and flat colour (IIP = 0, -DHWR_FLAT_COLOUR=1).
extern const std::uint32_t hw_vertex_smooth_words[];
extern const std::size_t hw_vertex_smooth_word_count;
extern const std::uint32_t hw_vertex_flat_words[];
extern const std::size_t hw_vertex_flat_word_count;
// TASK-179: smooth colour, a point drawn as two triangles (six vertices per instance), for devices without largePoints;
// with core 2's palette row (TASK-168).
extern const std::uint32_t hw_vertex_point_quad_words[];
extern const std::size_t hw_vertex_point_quad_word_count;
extern const std::uint32_t hw_vertex_core2_words[];
extern const std::size_t hw_vertex_core2_word_count;

// Uber pixel fragment stage, smooth and flat variants (must match the vertex stage).
extern const std::uint32_t hw_pixel_uber_smooth_words[];
extern const std::size_t hw_pixel_uber_smooth_word_count;
extern const std::uint32_t hw_pixel_uber_flat_words[];
extern const std::size_t hw_pixel_uber_flat_word_count;
// Smooth variant with ordered pixel interlock (IN-345; VK_EXT_fragment_shader_interlock).
extern const std::uint32_t hw_pixel_uber_smooth_interlock_words[];
extern const std::size_t hw_pixel_uber_smooth_interlock_word_count;

// M10 GLM kernels (IN-347).
extern const std::uint32_t hw_glm_decode_words[];
extern const std::size_t hw_glm_decode_word_count;
extern const std::uint32_t hw_glm_swizzle_words[];
extern const std::size_t hw_glm_swizzle_word_count;
extern const std::uint32_t hw_glm_unswizzle_words[];
extern const std::size_t hw_glm_unswizzle_word_count;
extern const std::uint32_t hw_glm_merge_words[];
extern const std::size_t hw_glm_merge_word_count;
extern const std::uint32_t hw_glm_host_upload_words[];
extern const std::size_t hw_glm_host_upload_word_count;
extern const std::uint32_t hw_display_compose_words[];
extern const std::size_t hw_display_compose_word_count;
extern const std::uint32_t hw_glm_clut_words[];
extern const std::size_t hw_glm_clut_word_count;
// M12 upscaling (IN-371/IN-372).
extern const std::uint32_t hw_scale_upsample_words[];
extern const std::size_t hw_scale_upsample_word_count;
extern const std::uint32_t hw_display_compose_scaled_words[];
extern const std::size_t hw_display_compose_scaled_word_count;
extern const std::uint32_t hw_scale_tone_carry_words[];  // IN-379
extern const std::size_t hw_scale_tone_carry_word_count;
extern const std::uint32_t hw_tone_strips_words[];  // IN-383 (parity plan item 2)
extern const std::size_t hw_tone_strips_word_count;
extern const std::uint32_t hw_deinterlace_scaled_words[];  // IN-381
extern const std::size_t hw_deinterlace_scaled_word_count;
// TASK-179: constant indices only, for devices without shaderStorageImageArrayDynamicIndexing.
extern const std::uint32_t hw_deinterlace_scaled_constant_words[];
extern const std::size_t hw_deinterlace_scaled_constant_word_count;
extern const std::uint32_t hw_postfx_fxaa_words[];
extern const std::size_t hw_postfx_fxaa_word_count;

// CORE2 core rewrite (CORE2-SPEC).
extern const std::uint32_t core2_patch_words[];
extern const std::size_t core2_patch_word_count;
extern const std::uint32_t core2_reduce_words[];
extern const std::size_t core2_reduce_word_count;
extern const std::uint32_t core2_tone_words[];
extern const std::size_t core2_tone_word_count;
extern const std::uint32_t core2_patch_glm_words[];
extern const std::size_t core2_patch_glm_word_count;
extern const std::uint32_t core2_compose_words[];
extern const std::size_t core2_compose_word_count;
extern const std::uint32_t core2_present_scale_words[];
extern const std::size_t core2_present_scale_word_count;

// TASK-135: the performance HUD's panel.
extern const std::uint32_t hw_hud_vertex_words[];
extern const std::size_t hw_hud_vertex_word_count;
extern const std::uint32_t hw_hud_fragment_words[];
extern const std::size_t hw_hud_fragment_word_count;
extern const std::uint32_t core2_fill_words[];
extern const std::size_t core2_fill_word_count;
extern const std::uint32_t hw_pixel_uber_core2_words[];
extern const std::size_t hw_pixel_uber_core2_word_count;
extern const std::uint32_t hw_pixel_uber_core2_rop_blend_words[];
extern const std::size_t hw_pixel_uber_core2_rop_blend_word_count;
extern const std::uint32_t hw_pixel_uber_core2_rop_words[];
extern const std::size_t hw_pixel_uber_core2_rop_word_count;
extern const std::uint32_t hw_pixel_uber_core2_interlock_words[];
extern const std::size_t hw_pixel_uber_core2_interlock_word_count;
extern const std::uint32_t core2_clut_target_words[];
extern const std::size_t core2_clut_target_word_count;
extern const std::uint32_t core2_materialize_words[];
extern const std::size_t core2_materialize_word_count;

// Exactness compute entry points (fixture only; never linked into the game).
extern const std::uint32_t hw_blend_exact_words[];
extern const std::size_t hw_blend_exact_word_count;
extern const std::uint32_t hw_pixel_exact_words[];
extern const std::size_t hw_pixel_exact_word_count;

}  // namespace coregs::shaders
