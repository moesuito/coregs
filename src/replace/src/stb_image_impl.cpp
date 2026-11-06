// Written from the TASK-026 specification (docs/hw-renderer-spec), chapter 11.4 (clean room, decision 0069).
// Single translation unit that instantiates the vendored stb_image v2.30 PNG decoder.
// third_party/stb_image.h is unmodified (public domain, see native/THIRD_PARTY.md).
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "stb_image.h"
