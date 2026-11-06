# CoreGS library targets.
#
#   coregs_core     portable CPU side: GS front end, shadow memory, CLUT, pixel rules,
#                   display mapping, texture replacement, CPU reference, caches
#   coregs_gpu      Vulkan side: the CoreGS 2.0 renderer and its GPU drawer
#   coregs_shaders  GLSL sources compiled to embedded SPIR-V
#   coregs          the renderer front end a host talks to (device, swapchain, present)

add_subdirectory(shaders)

# ---- Vulkan headers and loader ---------------------------------------------------------------
if(WIN32)
    set(COREGS_VULKAN_SDK "$ENV{VULKAN_SDK}" CACHE PATH "Vulkan SDK root with Include/ and Lib/vulkan-1.lib")
    find_path(COREGS_VULKAN_INCLUDE_DIR vulkan/vulkan.h HINTS "${COREGS_VULKAN_SDK}/Include" REQUIRED)
    find_library(COREGS_VULKAN_LIBRARY NAMES vulkan-1 HINTS "${COREGS_VULKAN_SDK}/Lib" REQUIRED)
else()
    find_package(Vulkan REQUIRED)
    set(COREGS_VULKAN_INCLUDE_DIR "${Vulkan_INCLUDE_DIRS}")
    set(COREGS_VULKAN_LIBRARY "${Vulkan_LIBRARIES}")
endif()
add_library(coregs_vulkan INTERFACE)
target_include_directories(coregs_vulkan INTERFACE "${COREGS_VULKAN_INCLUDE_DIR}")
target_link_libraries(coregs_vulkan INTERFACE "${COREGS_VULKAN_LIBRARY}")

# ---- coregs_core: portable CPU side -----------------------------------------------------------
set(COREGS_CORE_SOURCES
    hw_gs_options.cpp
    memory/hw_memory.cpp
    clut/hw_clut.cpp
    frontend/hw_gs_registers.cpp
    frontend/hw_gif_parser.cpp
    frontend/hw_transfers.cpp
    frontend/hw_batch.cpp
    frontend/hw_front_end.cpp
    pixel/hw_gs_pixel.cpp
    display/hw_gs_display.cpp
    cpuref/cpuref_memory.cpp
    cpuref/cpuref_draw.cpp
    cpuref/cpuref_renderer.cpp
    cpuref/cpuref_vram_dump.cpp
    cpuref/cpuref_replay.cpp
    cpuref/cpuref_replay_v2.cpp
    cache/hw_cache.cpp
    cache/hw_cache_dirty.cpp
    cache/hw_cache_fake_gpu.cpp
    cache/hw_cache_geometry.cpp
    decide/hw_gs_decide.cpp
    decide/hw_gs_decide_extra.cpp
    lod/hw_gs_lod.cpp
    replace/src/gs_types.cpp
    replace/src/xxh3.cpp
    replace/src/xxh3_impl.cpp
    replace/src/clut.cpp
    replace/src/keys.cpp
    replace/src/names.cpp
    replace/src/packs.cpp
    replace/src/images.cpp
    replace/src/stb_image_impl.cpp
    replace/src/png_encode.cpp
    core2/core2_upload.cpp
    core2/core2_depth_alias.cpp)

add_library(coregs_core STATIC ${COREGS_CORE_SOURCES})
target_include_directories(coregs_core
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory"
        "${CMAKE_CURRENT_SOURCE_DIR}/clut"
        "${CMAKE_CURRENT_SOURCE_DIR}/frontend"
        "${CMAKE_CURRENT_SOURCE_DIR}/pixel"
        "${CMAKE_CURRENT_SOURCE_DIR}/display"
        "${CMAKE_CURRENT_SOURCE_DIR}/cpuref"
        "${CMAKE_CURRENT_SOURCE_DIR}/cache"
        "${CMAKE_CURRENT_SOURCE_DIR}/decide"
        "${CMAKE_CURRENT_SOURCE_DIR}/lod"
        "${CMAKE_CURRENT_SOURCE_DIR}/replace/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/replace/tests"
        "${CMAKE_CURRENT_SOURCE_DIR}/replace/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/replace/third_party"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/core2")
target_compile_features(coregs_core PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(coregs_core PRIVATE /W4 /permissive- -Wno-unused-parameter)
    target_compile_definitions(coregs_core PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(coregs_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# ---- coregs_gpu: CoreGS 2.0 and its GPU drawer ------------------------------------------------
set(COREGS_GPU_SOURCES
    hw_gs_vulkan_loader.cpp
    writeback/hw_gs_writeback_gpu.cpp
    writeback/hw_gs_vulkan_device_table.cpp
    writeback/hw_gs_writeback.cpp
    writeback/hw_gs_present_scheduler.cpp
    core2/core2_gpu.cpp
    core2/core2_renderer.cpp
    core2/core2_replace.cpp)

add_library(coregs_gpu STATIC ${COREGS_GPU_SOURCES})
target_include_directories(coregs_gpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(coregs_gpu PUBLIC coregs_core coregs_shaders coregs_vulkan)
target_compile_features(coregs_gpu PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(coregs_gpu PRIVATE /W4 -Wno-unused-parameter -Wno-missing-field-initializers)
    target_compile_definitions(coregs_gpu PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(coregs_gpu PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
endif()

# ---- coregs: the front end a host talks to ----------------------------------------------------
set(COREGS_RENDERER_SOURCES hw_present_mode.cpp)
if(WIN32)
    list(APPEND COREGS_RENDERER_SOURCES hw_gs_backend.cpp)
endif()

add_library(coregs STATIC ${COREGS_RENDERER_SOURCES})
target_include_directories(coregs PUBLIC "${PROJECT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/writeback")
target_link_libraries(coregs PUBLIC coregs_gpu coregs_vulkan)
target_compile_features(coregs PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(coregs PRIVATE /W4 -Wno-unused-parameter -Wno-missing-field-initializers)
    target_compile_definitions(coregs PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(coregs PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
endif()
if(WIN32)
    target_link_libraries(coregs PUBLIC user32 gdi32)
endif()
