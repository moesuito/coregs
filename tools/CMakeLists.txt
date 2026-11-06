# Replay tool: a recorded GS stream through the reference write-back renderer, one PPM per frame.
add_executable(coregs_replay "${PROJECT_SOURCE_DIR}/src/writeback/hw_writeback_replay.cpp")
target_include_directories(coregs_replay PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/src/writeback"
    "${PROJECT_SOURCE_DIR}/src/cpuref")
target_link_libraries(coregs_replay PRIVATE coregs_gpu coregs_core coregs_vulkan)

# CPU reference replay: the same capture through the CPU rasteriser, one PPM per frame.
add_executable(coregs_cpu_replay "${PROJECT_SOURCE_DIR}/src/cpuref/hw_cpuref_replay.cpp")
target_include_directories(coregs_cpu_replay PRIVATE "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/src/cpuref")
target_link_libraries(coregs_cpu_replay PRIVATE coregs_core)
