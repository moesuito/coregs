// Written from the TASK-026 specification (docs/hw-renderer-spec), chapter 11.2 (clean room, decision 0069).
// Single translation unit that instantiates the vendored xxHash 0.8.3 implementation.
// third_party/xxhash.h is the unmodified upstream single header (BSD-2-Clause, see
// native/THIRD_PARTY.md). XXH_STATIC_LINKING_ONLY exposes the state structs needed by the build.
#define XXH_STATIC_LINKING_ONLY
#define XXH_IMPLEMENTATION
#include "xxhash.h"
