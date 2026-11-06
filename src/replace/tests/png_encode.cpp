// Written from the TASK-026 specification (docs/hw-renderer-spec), chapter 11 test fixture (clean room, decision 0069).
// Minimal PNG writer for CTest fixtures. Clean room (decision 0069).
// Copyright (c) 2026 GT4Recomp contributors, MIT.
#include "png_encode.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>

namespace hw_gs_replace::testing {
namespace {

uint32_t Crc32(const uint8_t* data, size_t size, uint32_t seed = 0) {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> built{};
    for (uint32_t n = 0; n < 256; ++n) {
      uint32_t c = n;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1u) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      }
      built[n] = c;
    }
    return built;
  }();
  uint32_t crc = seed ^ 0xFFFFFFFFu;
  for (size_t i = 0; i < size; ++i) {
    crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

void AppendBe32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value >> 24));
  out.push_back(static_cast<uint8_t>(value >> 16));
  out.push_back(static_cast<uint8_t>(value >> 8));
  out.push_back(static_cast<uint8_t>(value));
}

void AppendChunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& data) {
  AppendBe32(out, static_cast<uint32_t>(data.size()));
  std::vector<uint8_t> typed(type, type + 4);
  typed.insert(typed.end(), data.begin(), data.end());
  out.insert(out.end(), typed.begin(), typed.end());
  AppendBe32(out, Crc32(typed.data(), typed.size()));
}

}  // namespace

std::vector<uint8_t> EncodePng(uint32_t width, uint32_t height, uint8_t bitDepth,
                               uint8_t colourType, const std::vector<uint8_t>& pixels,
                               uint32_t channels) {
  // Scanlines: a filter byte 0 followed by the row bytes.
  const size_t rowBytes = static_cast<size_t>(width) * channels;
  std::vector<uint8_t> raw;
  raw.reserve((rowBytes + 1) * height);
  for (uint32_t y = 0; y < height; ++y) {
    raw.push_back(0);
    raw.insert(raw.end(), pixels.begin() + static_cast<std::ptrdiff_t>(y * rowBytes),
               pixels.begin() + static_cast<std::ptrdiff_t>((y + 1) * rowBytes));
  }

  // zlib stream with stored deflate blocks of at most 65535 bytes.
  std::vector<uint8_t> zlib = {0x78, 0x01};
  size_t offset = 0;
  do {
    const size_t block = std::min<size_t>(65535, raw.size() - offset);
    const bool last = offset + block == raw.size();
    zlib.push_back(last ? 0x01 : 0x00);
    zlib.push_back(static_cast<uint8_t>(block & 0xFF));
    zlib.push_back(static_cast<uint8_t>(block >> 8));
    zlib.push_back(static_cast<uint8_t>(~block & 0xFF));
    zlib.push_back(static_cast<uint8_t>((~block >> 8) & 0xFF));
    zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                raw.begin() + static_cast<std::ptrdiff_t>(offset + block));
    offset += block;
  } while (offset < raw.size());

  uint32_t a = 1;
  uint32_t b = 0;
  for (const uint8_t byte : raw) {
    a = (a + byte) % 65521u;
    b = (b + a) % 65521u;
  }
  AppendBe32(zlib, (b << 16) | a);

  std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<uint8_t> header;
  AppendBe32(header, width);
  AppendBe32(header, height);
  header.push_back(bitDepth);
  header.push_back(colourType);
  header.push_back(0);
  header.push_back(0);
  header.push_back(0);
  AppendChunk(out, "IHDR", header);
  AppendChunk(out, "IDAT", zlib);
  AppendChunk(out, "IEND", {});
  return out;
}

}  // namespace hw_gs_replace::testing
