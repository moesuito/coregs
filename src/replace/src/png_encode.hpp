// Written from the TASK-026 specification (docs/hw-renderer-spec), chapter 11.4 test fixture (clean room, decision 0069).
// Minimal PNG writer for CTest fixtures: stored (uncompressed) deflate blocks, 8-bit channels.
// Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#pragma once

#include <cstdint>
#include <vector>

namespace hw_gs_replace::testing {

// `pixels` holds width * height * channels bytes, rows top to bottom. `colourType` is written
// into IHDR as given (2 RGB, 6 RGBA, 3 palette, ...), and `bitDepth` likewise, so refusal tests
// can craft files the decoder must reject. Channels must match the colour type.
std::vector<uint8_t> EncodePng(uint32_t width, uint32_t height, uint8_t bitDepth,
                               uint8_t colourType, const std::vector<uint8_t>& pixels,
                               uint32_t channels);

}  // namespace hw_gs_replace::testing
