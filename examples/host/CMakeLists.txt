# The example host: a window, a synthetic GS stream and the public CoreGS interface.
add_executable(coregs_example_host main.cpp)
target_link_libraries(coregs_example_host PRIVATE coregs coregs_profile_gt4)
target_link_libraries(coregs_example_host PRIVATE user32 gdi32)
target_compile_definitions(coregs_example_host PRIVATE COREGS_WITH_GT4_PROFILE=1 _CRT_SECURE_NO_WARNINGS)
