// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 11.2 and 11.3.5.
// Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#include "hw_gs_replace/names.hpp"

#include <cstdio>
#include <vector>

namespace hw_gs_replace {
namespace {

// Splits on a separator; empty fields are kept so that a double separator gives an empty field.
std::vector<std::string_view> SplitOn(std::string_view text, char separator) {
  std::vector<std::string_view> fields;
  size_t start = 0;
  while (true) {
    const size_t at = text.find(separator, start);
    if (at == std::string_view::npos) {
      fields.push_back(text.substr(start));
      break;
    }
    fields.push_back(text.substr(start, at - start));
    start = at + 1;
  }
  return fields;
}

int HexDigitValue(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  return -1;
}

// 1..16 hex digits, either case, leading zeros allowed.
bool ParseHex64(std::string_view text, uint64_t& value) {
  if (text.empty() || text.size() > 16) {
    return false;
  }
  uint64_t result = 0;
  for (const char c : text) {
    const int digit = HexDigitValue(c);
    if (digit < 0) {
      return false;
    }
    result = (result << 4) | static_cast<uint64_t>(digit);
  }
  value = result;
  return true;
}

bool ParseHex32(std::string_view text, uint32_t& value) {
  uint64_t wide = 0;
  if (!ParseHex64(text, wide) || wide > 0xFFFFFFFFull) {
    return false;
  }
  value = static_cast<uint32_t>(wide);
  return true;
}

// 1..10 decimal digits that fit an unsigned 32-bit value.
bool ParseDecimal32(std::string_view text, uint32_t& value) {
  if (text.empty() || text.size() > 10) {
    return false;
  }
  uint64_t result = 0;
  for (const char c : text) {
    if (c < '0' || c > '9') {
      return false;
    }
    result = result * 10 + static_cast<uint64_t>(c - '0');
  }
  if (result > 0xFFFFFFFFull) {
    return false;
  }
  value = static_cast<uint32_t>(result);
  return true;
}

bool StartsWithRegionMarker(std::string_view token) {
  return !token.empty() && (token.front() == 'r' || token.front() == 'R');
}

// Region token "rWxH" (decimal) or legacy "r<packed64>" (hex, without an 'x').
bool ParseRegionToken(std::string_view token, uint32_t& width, uint32_t& height) {
  if (!StartsWithRegionMarker(token)) {
    return false;
  }
  const std::string_view body = token.substr(1);
  const size_t x = body.find_first_of("xX");
  if (x != std::string_view::npos) {
    return ParseDecimal32(body.substr(0, x), width) && ParseDecimal32(body.substr(x + 1), height);
  }
  uint64_t packed = 0;
  if (!ParseHex64(body, packed)) {
    return false;
  }
  const uint16_t minX = static_cast<uint16_t>(packed & 0xFFFFu);
  const uint16_t maxX = static_cast<uint16_t>((packed >> 16) & 0xFFFFu);
  const uint16_t minY = static_cast<uint16_t>((packed >> 32) & 0xFFFFu);
  const uint16_t maxY = static_cast<uint16_t>((packed >> 48) & 0xFFFFu);
  const int16_t signedWidth = static_cast<int16_t>(static_cast<uint16_t>(maxX - minX));
  const int16_t signedHeight = static_cast<int16_t>(static_cast<uint16_t>(maxY - minY));
  width = static_cast<uint32_t>(static_cast<int32_t>(signedWidth));
  height = static_cast<uint32_t>(static_cast<int32_t>(signedHeight));
  return true;
}

// Splits "stem.ext" at the last dot. False when there is no dot or the extension is not png/dds.
bool SplitExtension(std::string_view basename, std::string_view& stem,
                    ImageExtension& extension) {
  const size_t dot = basename.rfind('.');
  if (dot == std::string_view::npos) {
    return false;
  }
  const std::optional<ImageExtension> parsed = ParseImageExtension(basename.substr(dot + 1));
  if (!parsed.has_value()) {
    return false;
  }
  stem = basename.substr(0, dot);
  extension = *parsed;
  return true;
}

}  // namespace

std::string_view ImageExtensionText(ImageExtension extension) {
  return extension == ImageExtension::Png ? "png" : "dds";
}

std::optional<ImageExtension> ParseImageExtension(std::string_view text) {
  if (text.size() != 3) {
    return std::nullopt;
  }
  char lower[3];
  for (size_t i = 0; i < 3; ++i) {
    const char c = text[i];
    lower[i] = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
  }
  if (lower[0] == 'p' && lower[1] == 'n' && lower[2] == 'g') {
    return ImageExtension::Png;
  }
  if (lower[0] == 'd' && lower[1] == 'd' && lower[2] == 's') {
    return ImageExtension::Dds;
  }
  return std::nullopt;
}

std::string FormatGFileName(const FormatGKey& key, ImageExtension extension) {
  char buffer[160];
  std::snprintf(buffer, sizeof(buffer), "t-%016llx-%016llx-%ux%u.%s",
                static_cast<unsigned long long>(key.textureHash),
                static_cast<unsigned long long>(key.paletteHash), static_cast<unsigned>(key.width),
                static_cast<unsigned>(key.height),
                ImageExtensionText(extension).data());
  return buffer;
}

std::optional<ParsedFormatGName> ParseFormatGFileName(std::string_view basename) {
  std::string_view stem;
  ParsedFormatGName parsed;
  if (!SplitExtension(basename, stem, parsed.extension)) {
    return std::nullopt;
  }
  const std::vector<std::string_view> fields = SplitOn(stem, '-');
  if (fields.size() != 4 || fields[0] != "t" || fields[1].size() != 16 ||
      fields[2].size() != 16) {
    return std::nullopt;
  }
  if (!ParseHex64(fields[1], parsed.key.textureHash) ||
      !ParseHex64(fields[2], parsed.key.paletteHash)) {
    return std::nullopt;
  }
  const size_t x = fields[3].find('x');
  if (x == std::string_view::npos || !ParseDecimal32(fields[3].substr(0, x), parsed.key.width) ||
      !ParseDecimal32(fields[3].substr(x + 1), parsed.key.height)) {
    return std::nullopt;
  }
  return parsed;
}

std::string FormatPDumpName(const FormatPRecord& record, ImageExtension extension) {
  std::string name;
  char buffer[96];
  std::snprintf(buffer, sizeof(buffer), "%llx",
                static_cast<unsigned long long>(record.textureHash));
  name += buffer;
  if (record.indexed) {
    std::snprintf(buffer, sizeof(buffer), "-%llx", static_cast<unsigned long long>(record.clutHash));
    name += buffer;
  }
  if (record.regionWidth != 0 || record.regionHeight != 0) {
    std::snprintf(buffer, sizeof(buffer), "-r%ux%u", static_cast<unsigned>(record.regionWidth),
                  static_cast<unsigned>(record.regionHeight));
    name += buffer;
  }
  std::snprintf(buffer, sizeof(buffer), "-%08x", static_cast<unsigned>(record.bits));
  name += buffer;
  if (record.mipLevel >= 1) {
    std::snprintf(buffer, sizeof(buffer), "-mip%u", static_cast<unsigned>(record.mipLevel));
    name += buffer;
  }
  name += '.';
  name += ImageExtensionText(extension);
  return name;
}

std::optional<ParsedFormatPName> ParseFormatPFileName(std::string_view basename) {
  std::string_view stem;
  ParsedFormatPName parsed;
  if (!SplitExtension(basename, stem, parsed.extension)) {
    return std::nullopt;
  }
  // The six patterns of chapter 11.3.5 differ in token count and in whether the second token
  // is a region (starts with 'r') or a CLUT hash (hex, never starts with 'r'):
  //   2 tokens: tex-bits            3 tokens: tex-clut-bits, or tex-rWxH-bits (legacy too)
  //   4 tokens: tex-clut-rWxH-bits (legacy too)
  const std::vector<std::string_view> fields = SplitOn(stem, '-');
  std::string_view clutToken;
  std::string_view regionToken;
  std::string_view bitsToken;
  switch (fields.size()) {
    case 2:
      bitsToken = fields[1];
      break;
    case 3:
      if (StartsWithRegionMarker(fields[1])) {
        regionToken = fields[1];
      } else {
        clutToken = fields[1];
      }
      bitsToken = fields[2];
      break;
    case 4:
      clutToken = fields[1];
      regionToken = fields[2];
      bitsToken = fields[3];
      break;
    default:
      return std::nullopt;
  }

  if (!ParseHex64(fields[0], parsed.textureHash)) {
    return std::nullopt;
  }
  if (!clutToken.empty()) {
    parsed.clutPresent = true;
    if (!ParseHex64(clutToken, parsed.clutHash)) {
      return std::nullopt;
    }
  }
  if (!regionToken.empty() &&
      !ParseRegionToken(regionToken, parsed.regionWidth, parsed.regionHeight)) {
    return std::nullopt;
  }
  if (!ParseHex32(bitsToken, parsed.bits)) {
    return std::nullopt;
  }
  // Bit 14 is unused and cleared when parsing (chapter 11.3).
  parsed.bits &= ~(1u << 14);
  return parsed;
}

bool FormatPNameMatches(const ParsedFormatPName& name, const FormatPRecord& record) {
  if (name.textureHash != record.textureHash || name.bits != record.bits ||
      name.regionWidth != record.regionWidth || name.regionHeight != record.regionHeight) {
    return false;
  }
  if (record.indexed) {
    return name.clutPresent && name.clutHash == record.clutHash;
  }
  return !name.clutPresent;
}

}  // namespace hw_gs_replace
