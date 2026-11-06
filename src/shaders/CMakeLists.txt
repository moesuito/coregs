# CoreGS shader toolchain.
#
# The GLSL sources in this directory are compiled at build time by glslangValidator into SPIR-V
# and embedded as C++ word arrays (hw_shader_blobs.hpp). A shader error fails the build.
# Permutations are compile-time #define sets listed in the coregs_add_shader() calls below.
#
# The compiler is pinned by SHA-256 so that every build of a release uses the same glslang.
# COREGS_GLSLANG_PIN=OFF exists for CI images whose distro package differs; the build then
# records the compiler version it used and says so.

set(COREGS_GLSLANG "$ENV{VULKAN_SDK}/Bin/glslangValidator"
    CACHE FILEPATH "glslangValidator used to build the CoreGS shaders")
if(NOT EXISTS "${COREGS_GLSLANG}")
    find_program(COREGS_GLSLANG_FOUND NAMES glslangValidator glslang)
    if(COREGS_GLSLANG_FOUND)
        set(COREGS_GLSLANG "${COREGS_GLSLANG_FOUND}" CACHE FILEPATH "glslangValidator used to build the CoreGS shaders" FORCE)
    else()
        message(FATAL_ERROR "CoreGS shaders need glslangValidator: set COREGS_GLSLANG or VULKAN_SDK")
    endif()
endif()

option(COREGS_GLSLANG_PIN "Require the pinned glslangValidator hash" ON)
set(COREGS_GLSLANG_SHA256 "db832826d73049193882d5572ad76e14fcb7b1cdfc1a6b892ac25217b850b6bc"
    CACHE STRING "Pinned SHA-256 of glslangValidator")
if(COREGS_GLSLANG_PIN)
    file(SHA256 "${COREGS_GLSLANG}" COREGS_GLSLANG_ACTUAL_SHA256)
    if(NOT COREGS_GLSLANG_ACTUAL_SHA256 STREQUAL COREGS_GLSLANG_SHA256)
        message(FATAL_ERROR "glslangValidator hash ${COREGS_GLSLANG_ACTUAL_SHA256} does not match the pin "
                            "${COREGS_GLSLANG_SHA256}. Point COREGS_GLSLANG at the pinned binary, or set "
                            "COREGS_GLSLANG_PIN=OFF deliberately.")
    endif()
else()
    message(WARNING "COREGS_GLSLANG_PIN=OFF: the shaders are built with an unpinned glslangValidator.")
endif()
execute_process(COMMAND "${COREGS_GLSLANG}" --version OUTPUT_VARIABLE COREGS_GLSLANG_VERSION_TEXT
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "Glslang Version: [^\n]*" COREGS_GLSLANG_VERSION_LINE "${COREGS_GLSLANG_VERSION_TEXT}")
message(STATUS "CoreGS shaders: ${COREGS_GLSLANG_VERSION_LINE} (${COREGS_GLSLANG})")

set(COREGS_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(COREGS_SHADER_INCLUDES "${COREGS_SHADER_DIR}/hw_pixel_core.glsl" "${COREGS_SHADER_DIR}/hw_glm_common.glsl")
set(COREGS_EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/embed_spirv.cmake")

# coregs_add_shader(<symbol> <source> <vert|frag|comp> [DEFINES <macro=value>...])
# Produces <symbol>_words and <symbol>_word_count in namespace coregs::shaders.
set(COREGS_SHADER_CPP_FILES "")
function(coregs_add_shader SYMBOL SOURCE STAGE)
    cmake_parse_arguments(ARG "" "" "DEFINES" ${ARGN})
    set(defines "")
    foreach(define IN LISTS ARG_DEFINES)
        list(APPEND defines "-D${define}")
    endforeach()
    set(spv "${CMAKE_CURRENT_BINARY_DIR}/${SYMBOL}.spv")
    set(cpp "${CMAKE_CURRENT_BINARY_DIR}/${SYMBOL}_spirv.cpp")
    add_custom_command(
        OUTPUT "${spv}"
        COMMAND "${COREGS_GLSLANG}" -V --target-env vulkan1.2 "-I${COREGS_SHADER_DIR}" -S ${STAGE}
                ${defines} -o "${spv}" "${SOURCE}"
        DEPENDS "${SOURCE}" ${COREGS_SHADER_INCLUDES}
        COMMENT "glslang ${SYMBOL} (${STAGE})"
        VERBATIM)
    add_custom_command(
        OUTPUT "${cpp}"
        COMMAND ${CMAKE_COMMAND} "-DSPV=${spv}" "-DCPP=${cpp}" "-DSYMBOL=${SYMBOL}" -P "${COREGS_EMBED_SCRIPT}"
        DEPENDS "${spv}" "${COREGS_EMBED_SCRIPT}"
        COMMENT "embed ${SYMBOL}"
        VERBATIM)
    set(COREGS_SHADER_CPP_FILES ${COREGS_SHADER_CPP_FILES} "${cpp}" PARENT_SCOPE)
endfunction()

coregs_add_shader(hw_vertex_smooth "${COREGS_SHADER_DIR}/hw_vertex.vert" vert DEFINES HWR_FLAT_COLOUR=0)
coregs_add_shader(hw_vertex_flat "${COREGS_SHADER_DIR}/hw_vertex.vert" vert DEFINES HWR_FLAT_COLOUR=1)
coregs_add_shader(hw_vertex_core2 "${COREGS_SHADER_DIR}/hw_vertex.vert" vert DEFINES HWR_FLAT_COLOUR=0 HWR_PALETTE_ROW=1)
coregs_add_shader(hw_vertex_point_quad "${COREGS_SHADER_DIR}/hw_vertex.vert" vert DEFINES HWR_FLAT_COLOUR=0 HWR_POINT_QUAD=1
    HWR_PALETTE_ROW=1)
coregs_add_shader(hw_pixel_uber_smooth "${COREGS_SHADER_DIR}/hw_pixel_uber.frag" frag DEFINES HWR_FLAT_COLOUR=0)
coregs_add_shader(hw_pixel_uber_smooth_interlock "${COREGS_SHADER_DIR}/hw_pixel_uber.frag" frag
              DEFINES HWR_FLAT_COLOUR=0 HWR_INTERLOCK=1)
coregs_add_shader(hw_pixel_uber_flat "${COREGS_SHADER_DIR}/hw_pixel_uber.frag" frag DEFINES HWR_FLAT_COLOUR=1)
coregs_add_shader(hw_pixel_uber_core2 "${COREGS_SHADER_DIR}/hw_pixel_uber.frag" frag DEFINES HWR_FLAT_COLOUR=0 HWR_EARLY_Z=1
              HWR_PALETTE_ROW=1)
coregs_add_shader(hw_pixel_uber_core2_interlock "${COREGS_SHADER_DIR}/hw_pixel_uber.frag" frag
              DEFINES HWR_FLAT_COLOUR=0 HWR_INTERLOCK=1 HWR_EARLY_Z=1 HWR_PALETTE_ROW=1)
coregs_add_shader(hw_pixel_uber_core2_rop "${COREGS_SHADER_DIR}/hw_pixel_uber.frag" frag DEFINES HWR_FLAT_COLOUR=0 HWR_ROP=1
              HWR_PALETTE_ROW=1)
coregs_add_shader(hw_pixel_uber_core2_rop_blend "${COREGS_SHADER_DIR}/hw_pixel_uber.frag" frag DEFINES HWR_FLAT_COLOUR=0 HWR_ROP=2
              HWR_PALETTE_ROW=1)
coregs_add_shader(hw_glm_decode "${COREGS_SHADER_DIR}/hw_glm_decode.comp" comp)
coregs_add_shader(hw_glm_swizzle "${COREGS_SHADER_DIR}/hw_glm_swizzle.comp" comp)
coregs_add_shader(hw_glm_clut "${COREGS_SHADER_DIR}/hw_glm_clut.comp" comp)
coregs_add_shader(hw_scale_upsample "${COREGS_SHADER_DIR}/hw_scale_upsample.comp" comp)
coregs_add_shader(hw_display_compose_scaled "${COREGS_SHADER_DIR}/hw_display_compose_scaled.comp" comp)
coregs_add_shader(hw_scale_tone_carry "${COREGS_SHADER_DIR}/hw_scale_tone_carry.comp" comp)
coregs_add_shader(hw_tone_strips "${COREGS_SHADER_DIR}/hw_tone_strips.comp" comp)
coregs_add_shader(hw_deinterlace_scaled "${COREGS_SHADER_DIR}/hw_deinterlace_scaled.comp" comp)
coregs_add_shader(hw_deinterlace_scaled_constant "${COREGS_SHADER_DIR}/hw_deinterlace_scaled.comp" comp DEFINES HWR_CONSTANT_INDEXING=1)
coregs_add_shader(hw_postfx_fxaa "${COREGS_SHADER_DIR}/hw_postfx_fxaa.comp" comp)
coregs_add_shader(hw_glm_unswizzle "${COREGS_SHADER_DIR}/hw_glm_unswizzle.comp" comp)
coregs_add_shader(hw_glm_merge "${COREGS_SHADER_DIR}/hw_glm_merge.comp" comp)
coregs_add_shader(hw_glm_host_upload "${COREGS_SHADER_DIR}/hw_glm_host_upload.comp" comp)
coregs_add_shader(hw_display_compose "${COREGS_SHADER_DIR}/hw_display_compose.comp" comp)
coregs_add_shader(core2_patch "${COREGS_SHADER_DIR}/core2_patch.comp" comp)
coregs_add_shader(core2_reduce "${COREGS_SHADER_DIR}/core2_reduce.comp" comp)
coregs_add_shader(core2_tone "${COREGS_SHADER_DIR}/core2_tone.comp" comp)
coregs_add_shader(core2_patch_glm "${COREGS_SHADER_DIR}/core2_patch_glm.comp" comp)
coregs_add_shader(core2_compose "${COREGS_SHADER_DIR}/core2_compose.comp" comp)
coregs_add_shader(core2_present_scale "${COREGS_SHADER_DIR}/core2_present_scale.comp" comp)
coregs_add_shader(hw_hud_vertex "${COREGS_SHADER_DIR}/hw_hud.vert" vert)
coregs_add_shader(hw_hud_fragment "${COREGS_SHADER_DIR}/hw_hud.frag" frag)
coregs_add_shader(core2_fill "${COREGS_SHADER_DIR}/core2_fill.comp" comp)
coregs_add_shader(core2_clut_target "${COREGS_SHADER_DIR}/core2_clut_target.comp" comp)
coregs_add_shader(core2_materialize "${COREGS_SHADER_DIR}/core2_materialize.comp" comp)
coregs_add_shader(hw_blend_exact "${COREGS_SHADER_DIR}/hw_blend_exact.comp" comp)
coregs_add_shader(hw_pixel_exact "${COREGS_SHADER_DIR}/hw_pixel_exact.comp" comp)

add_library(coregs_shaders STATIC ${COREGS_SHADER_CPP_FILES})
target_include_directories(coregs_shaders PUBLIC "${COREGS_SHADER_DIR}")
target_compile_features(coregs_shaders PUBLIC cxx_std_20)
