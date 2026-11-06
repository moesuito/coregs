// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 11.2, 11.3.5, 11.5.
// Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#include "hw_gs_replace/packs.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <system_error>

namespace hw_gs_replace {
namespace {

namespace fs = std::filesystem;

std::string Lower(std::string text) {
  for (char& c : text) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return text;
}

// A path as UTF-8. path::string() converts to the ANSI code page and throws on any other character (real packs hold
// Japanese file names).
std::string PathText(const fs::path& path) {
  const std::u8string text = path.u8string();
  return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

bool HasZipExtension(const fs::path& path) {
  return Lower(PathText(path.extension())) == ".zip";
}

// Regular files under `directory`, recursively, hidden files included, sorted by path text.
std::vector<fs::path> SortedFilesRecursive(const fs::path& directory, std::vector<std::string>& messages) {
  std::vector<fs::path> files;
  std::error_code error;
  if (!fs::is_directory(directory, error)) {
    return files;
  }
  for (fs::recursive_directory_iterator it(directory, fs::directory_options::skip_permission_denied,
                                          error),
       end;
       it != end; it.increment(error)) {
    if (error) {
      messages.push_back("pack scan stopped under " + PathText(directory) + ": " + error.message());
      break;
    }
    if (it->is_regular_file(error)) {
      files.push_back(it->path());
    }
  }
  std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
    return a.native() < b.native();  // not generic_string(): it throws on a name outside the ANSI code page
  });
  return files;
}

// Zip archives directly inside `directory` (the chapter 11.3.5 folder rule uses recursion for
// loose files and lists archives where they are found).
std::vector<fs::path> SortedZipsRecursive(const fs::path& directory, std::vector<std::string>& messages) {
  std::vector<fs::path> zips;
  for (const fs::path& file : SortedFilesRecursive(directory, messages)) {
    if (HasZipExtension(file)) {
      zips.push_back(file);
    }
  }
  return zips;
}

}  // namespace

std::vector<std::string> UnsupportedArchiveReader::EntryBasenames(const fs::path& archive,
                                                                  std::string& error) const {
  error = "zip support is not built into this HWR core: " + PathText(archive);
  return {};
}

bool UnsupportedArchiveReader::ReadEntry(const fs::path& archive, std::string_view entryPath,
                                         std::vector<uint8_t>& bytes, std::string& error) const {
  (void)entryPath;
  bytes.clear();
  error = "zip support is not built into this HWR core: " + PathText(archive);
  return false;
}

ScanReport ReplacementIndex::ScanFormatG(const fs::path& root, std::string_view serial,
                                         const ArchiveReader& archives) {
  ScanReport report;
  const fs::path folder = root / std::string(serial) / "gt4";
  const std::vector<fs::path> files = SortedFilesRecursive(folder, report.messages);

  const auto registerEntry = [&](const fs::path& file, const std::string& member,
                                 const std::string& basename) {
    const std::optional<ParsedFormatGName> parsed = ParseFormatGFileName(basename);
    if (!parsed.has_value()) {
      ++report.unparsedNames;
      return;
    }
    const FormatGMapKey key{parsed->key.textureHash, parsed->key.paletteHash, parsed->key.width,
                            parsed->key.height};
    if (formatG_.contains(key)) {
      ++report.duplicatesIgnored;
      return;
    }
    PackEntry entry;
    entry.basename = basename;
    entry.extension = parsed->extension;
    entry.file = file;
    entry.member = member;
    formatG_.emplace(key, entry);
    ++report.filesRegistered;
  };

  for (const fs::path& file : files) {
    if (HasZipExtension(file)) {
      continue;
    }
    registerEntry(file, {}, PathText(file.filename()));
  }
  for (const fs::path& zip : SortedZipsRecursive(folder, report.messages)) {
    std::string error;
    const std::vector<std::string> entries = archives.EntryBasenames(zip, error);
    if (entries.empty() && !error.empty()) {
      ++report.archivesSkipped;
      report.messages.push_back(error);
      continue;
    }
    for (const std::string& basename : entries) {
      registerEntry(zip, basename, basename);
    }
  }
  return report;
}

ScanReport ReplacementIndex::ScanFormatP(const fs::path& root, std::string_view serial,
                                         const ArchiveReader& archives) {
  return ScanFormatPPlaces(root / std::string(serial) / "replacements", root / (std::string(serial) + ".zip"), archives);
}

ScanReport ReplacementIndex::ScanFormatPFolder(const fs::path& folder, const ArchiveReader& archives) {
  return ScanFormatPPlaces(folder, fs::path{}, archives);
}

ScanReport ReplacementIndex::ScanFormatPPlaces(const fs::path& folder, const fs::path& rootArchive,
                                               const ArchiveReader& archives) {
  ScanReport report;

  const auto registerEntry = [&](const fs::path& file, const std::string& member,
                                 const std::string& basename) {
    const std::optional<ParsedFormatPName> parsed = ParseFormatPFileName(basename);
    if (!parsed.has_value()) {
      ++report.unparsedNames;
      return;
    }
    const FormatPMapKey key{parsed->textureHash, parsed->clutPresent, parsed->clutHash,
                            parsed->regionWidth, parsed->regionHeight, parsed->bits};
    if (formatP_.contains(key)) {
      ++report.duplicatesIgnored;
      return;
    }
    PackEntry entry;
    entry.basename = basename;
    entry.extension = parsed->extension;
    entry.file = file;
    entry.member = member;
    formatP_.emplace(key, entry);
    ++report.filesRegistered;
  };

  for (const fs::path& file : SortedFilesRecursive(folder, report.messages)) {
    if (!HasZipExtension(file)) {
      registerEntry(file, {}, PathText(file.filename()));
    }
  }

  std::vector<fs::path> archivePaths = SortedZipsRecursive(folder, report.messages);
  std::error_code error;
  if (!rootArchive.empty() && fs::is_regular_file(rootArchive, error)) {
    archivePaths.push_back(rootArchive);
  }
  for (const fs::path& zip : archivePaths) {
    std::string readError;
    const std::vector<std::string> entries = archives.EntryBasenames(zip, readError);
    if (entries.empty() && !readError.empty()) {
      ++report.archivesSkipped;
      report.messages.push_back(readError);
      continue;
    }
    for (const std::string& basename : entries) {
      registerEntry(zip, basename, basename);
    }
  }
  return report;
}

std::optional<PackEntry> ReplacementIndex::FindFormatG(const FormatGKey& key) const {
  const auto found = formatG_.find(
      FormatGMapKey{key.textureHash, key.paletteHash, key.width, key.height});
  if (found == formatG_.end()) {
    return std::nullopt;
  }
  return found->second;
}

std::optional<PackEntry> ReplacementIndex::FindFormatP(const FormatPRecord& record) const {
  const FormatPMapKey key{record.textureHash, record.indexed, record.indexed ? record.clutHash : 0,
                          record.regionWidth, record.regionHeight, record.bits};
  const auto found = formatP_.find(key);
  if (found == formatP_.end()) {
    return std::nullopt;
  }
  return found->second;
}

bool ReadPackEntryBytes(const PackEntry& entry, const ArchiveReader& archives,
                        std::vector<uint8_t>& bytes, std::string& error) {
  if (!entry.member.empty()) {
    return archives.ReadEntry(entry.file, entry.member, bytes, error);
  }
  std::ifstream stream(entry.file, std::ios::binary);
  if (!stream) {
    error = "cannot open " + PathText(entry.file);
    return false;
  }
  bytes.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
  return true;
}

}  // namespace hw_gs_replace
