// HWR texture replacement pack discovery: folder scans, archive entries and the first-wins index
// (chapters 11.2, 11.3.5 and 11.5). Written from the TASK-026 specification (docs/hw-renderer-spec).
// Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "hw_gs_replace/keys.hpp"
#include "hw_gs_replace/names.hpp"

namespace hw_gs_replace {

// Zip access sits behind this interface. The shipped reader is a stub that reports every
// archive as unsupported; the zip implementation is a later step (see the TASK-027 report).
class ArchiveReader {
 public:
  virtual ~ArchiveReader() = default;
  // Basenames of the entries (directories removed). Empty on failure, with `error` set.
  virtual std::vector<std::string> EntryBasenames(const std::filesystem::path& archive,
                                                  std::string& error) const = 0;
  // Bytes of one entry, identified by its full entry path inside the archive.
  virtual bool ReadEntry(const std::filesystem::path& archive, std::string_view entryPath,
                         std::vector<uint8_t>& bytes, std::string& error) const = 0;
};

class UnsupportedArchiveReader final : public ArchiveReader {
 public:
  std::vector<std::string> EntryBasenames(const std::filesystem::path& archive,
                                          std::string& error) const override;
  bool ReadEntry(const std::filesystem::path& archive, std::string_view entryPath,
                 std::vector<uint8_t>& bytes, std::string& error) const override;
};

// Where a replacement file lives: a loose file, or an entry of an archive.
struct PackEntry {
  std::string basename;
  ImageExtension extension = ImageExtension::Png;
  std::filesystem::path file;   // loose file, or the archive when `member` is set
  std::string member;           // archive entry path; empty for loose files
};

// Result of one scan, for the log line of chapter 11.5 ("log the count").
struct ScanReport {
  size_t filesRegistered = 0;
  size_t duplicatesIgnored = 0;
  size_t unparsedNames = 0;
  size_t archivesSkipped = 0;
  std::vector<std::string> messages;
};

// First registered name wins (chapter 11.3.5). Registration order is deterministic: loose files
// sorted by path, then archive entries sorted by path, so packs must not rely on it.
class ReplacementIndex {
 public:
  // Format G roots: <root>/<serial>/gt4 recursively, plus any .zip found there.
  ScanReport ScanFormatG(const std::filesystem::path& root, std::string_view serial,
                         const ArchiveReader& archives);
  // Format P roots: <root>/<serial>/replacements recursively, then its .zip archives, then
  // <root>/<serial>.zip.
  ScanReport ScanFormatP(const std::filesystem::path& root, std::string_view serial,
                         const ArchiveReader& archives);
  // Format P names in `folder` recursively, then its .zip archives (the launcher's flat Textures folder, where a pack
  // is one more subfolder and no serial folder exists).
  ScanReport ScanFormatPFolder(const std::filesystem::path& folder, const ArchiveReader& archives);

  std::optional<PackEntry> FindFormatG(const FormatGKey& key) const;
  // Full-chain lookup first, then base-only, as chapter 11.3.4 asks (the caller passes both).
  std::optional<PackEntry> FindFormatP(const FormatPRecord& record) const;

  size_t FormatGCount() const { return formatG_.size(); }
  size_t FormatPCount() const { return formatP_.size(); }

 private:
  ScanReport ScanFormatPPlaces(const std::filesystem::path& folder, const std::filesystem::path& rootArchive,
                               const ArchiveReader& archives);

  using FormatGMapKey = std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>;
  // texture hash, clut present, clut hash, region width, region height, bits
  using FormatPMapKey = std::tuple<uint64_t, bool, uint64_t, uint32_t, uint32_t, uint32_t>;

  std::map<FormatGMapKey, PackEntry> formatG_;
  std::map<FormatPMapKey, PackEntry> formatP_;
};

// Reads the bytes of a registered entry (loose file or archive member).
bool ReadPackEntryBytes(const PackEntry& entry, const ArchiveReader& archives,
                        std::vector<uint8_t>& bytes, std::string& error);

}  // namespace hw_gs_replace
