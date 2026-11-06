// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
//
// Platform helpers for the portable core: the cycle counter the CoreGS 2.0 timers read, and the
// process environment read in the character type the platform stores file-system paths in.
//
// The core builds on Windows (MSVC/clang-cl) and on Linux (GCC/clang). The two differ in the header
// that declares the time-stamp counter and in how the environment holds a path: Windows keeps the
// environment block in UTF-16 and reads it with _wgetenv, Linux keeps bytes and reads it with getenv.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>

#if defined(_WIN32)
#include <stdlib.h>  // _wgetenv
#endif

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define COREGS_PLATFORM_X86 1
#if defined(_MSC_VER)
#include <intrin.h>  // __rdtsc
#else
#include <x86intrin.h>  // __rdtsc
#endif
#endif

namespace coregs::platform {

// A monotonic counter for the short intervals the CoreGS 2.0 timers measure. On x86 it is the
// time-stamp counter, exactly as the Windows builds have always read it; on other architectures
// steady_clock's ticks stand in, and the timers calibrate ticks against the clock either way
// (MicrosecondsPerTick).
inline std::uint64_t ReadCycleCounter() {
#if defined(COREGS_PLATFORM_X86)
    return __rdtsc();
#else
    return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

// The value of an environment variable that names a folder, in the character type the platform
// stores paths in: UTF-16 (wchar_t) on Windows, bytes (char) elsewhere. Returns nullptr when the
// variable is unset. The name is ASCII, so widening it on Windows needs no code page.
inline const std::filesystem::path::value_type *GetPathEnvironment(const char *name) {
#if defined(_WIN32)
    wchar_t wideName[64];
    std::size_t length = 0;
    for (; name[length] != '\0'; ++length) {
        if (length + 1 >= sizeof wideName / sizeof wideName[0]) return nullptr;
        wideName[length] = static_cast<wchar_t>(static_cast<unsigned char>(name[length]));
    }
    wideName[length] = L'\0';
    return _wgetenv(wideName);
#else
    return std::getenv(name);
#endif
}

}  // namespace coregs::platform
