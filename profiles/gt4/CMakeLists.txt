# The Gran Turismo 4 profile: the reference example of a game profile (Apache-2.0, our own code).
add_library(coregs_profile_gt4 STATIC gt4_profile.cpp patterns.cpp)
target_link_libraries(coregs_profile_gt4 PUBLIC coregs_core)
target_include_directories(coregs_profile_gt4 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(coregs_profile_gt4 PUBLIC cxx_std_20)

if(COREGS_BUILD_TESTS)
add_executable(coregs_gt4_patterns_tests patterns_tests.cpp)
target_link_libraries(coregs_gt4_patterns_tests PRIVATE coregs_profile_gt4)
add_test(NAME gt4_patterns COMMAND coregs_gt4_patterns_tests)

add_executable(coregs_gt4_flare_table_tests flare_table_tests.cpp)
target_link_libraries(coregs_gt4_flare_table_tests PRIVATE coregs_core)
add_test(NAME gt4_flare_table COMMAND coregs_gt4_flare_table_tests)
endif()
