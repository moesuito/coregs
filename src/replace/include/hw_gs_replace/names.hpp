// HWR texture replacement file names: Format G and Format P formatting and parsing
// (chapters 11.2 and 11.3.5). Written from the TASK-026 specification (docs/hw-renderer-spec).
// Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "hw_gs_replace/keys.hpp"

namespace hw_gs_replace {

enum class ImageExtension : uint8_t { Png, Dds };

// Lower-case extension text ("png" or "dds").
std::string_view ImageExtensionText(ImageExtension extension);
// Accepts any letter case; nullopt for other extensions.
std::optional<ImageExtension> ParseImageExtension(std::string_view text);

// Format G: t-<tex:016x>-<pal:016x>-<W>x<H>.<png|dds>, lower-case hex, zero padded.
std::string FormatGFileName(const FormatGKey& key, ImageExtension extension);
struct ParsedFormatGName {
  FormatGKey key;
  ImageExtension extension = ImageExtension::Png;
};
// Parses a basename: exactly 16 hex digits per hash in either case, decimal size, png or dds.
std::optional<ParsedFormatGName> ParseFormatGFileName(std::string_view basename);

// Format P dump name (chapter 11.3.5): <tex>[-<clut>]-r<W>x<H>-<bits>[-mip<N>].<ext>.
// Hashes are lower-case hex without padding; bits are 8 zero-padded hex digits; region sizes
// are decimal and omitted when the record has no region; -mipN is written when mipLevel >= 1.
std::string FormatPDumpName(const FormatPRecord& record, ImageExtension extension);

// A parsed Format P replacement name. Names with -mipN do not parse (chapter 11.3.5).
struct ParsedFormatPName {
  uint64_t textureHash = 0;
  bool clutPresent = false;
  uint64_t clutHash = 0;
  uint32_t regionWidth = 0;
  uint32_t regionHeight = 0;
  uint32_t bits = 0;  // bit 14 cleared
  ImageExtension extension = ImageExtension::Png;
};

// Parses a basename by the six patterns of chapter 11.3.5. Hex is case-insensitive with leading
// zeros allowed. Legacy region tokens r<packed64> decode as width = maxX - minX and
// height = maxY - minY, each as signed 16-bit, packed64 = minX | maxX << 16 | minY << 32 |
// maxY << 48. Unknown: whether the legacy packed field is hex; it is read as hex here.
std::optional<ParsedFormatPName> ParseFormatPFileName(std::string_view basename);

// A parsed name matches a runtime record only if every field equals the key: texture hash,
// bits word, region sizes, and the CLUT hash on an indexed record. A CLUT field never matches
// a direct record, and an indexed record never matches a name without a CLUT field.
bool FormatPNameMatches(const ParsedFormatPName& name, const FormatPRecord& record);

}  // namespace hw_gs_replace
