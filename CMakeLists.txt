cmake_minimum_required(VERSION 3.24)
project(CoreGS VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(COREGS_BUILD_TESTS "Build the CoreGS test fixtures" ON)
option(COREGS_BUILD_EXAMPLES "Build the example host" ON)
option(COREGS_BUILD_TOOLS "Build the replay tool" ON)

add_subdirectory(src)
add_subdirectory(profiles/gt4)

if(COREGS_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(COREGS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(COREGS_BUILD_EXAMPLES AND WIN32)
    add_subdirectory(examples/host)
endif()
