// SPDX-License-Identifier: MIT
// Copyright 2026 GT4Recomp contributors
//
// CTest fixtures for the HWR surface cache bookkeeping (TASK-027, milestone M7).
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 04
// (test vectors inline in 4.1), 05 (5.1, 5.3, 5.9, 5.10), 10 (V-10-1 with the
// 1x part of the vector, 10.2 shortcut, 10.3 selection) and 15 (M7 checks
// hw_cache_geometry, hw_dirty_merge, hw_ageing). Clean room.
//
// Deviations from the full specification are listed in
// docs/hw-renderer-spec/IMPLEMENTATION-NOTES.md (IN-07 and later).

#include "hw_cache.hpp"
#include "hw_cache_fake_gpu.hpp"
#include "hw_cache_geometry.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

using namespace coregs;
using namespace coregs::cache;
using memory::BufferView;
using memory::Psm;
using memory::TransferRect;

namespace {

int g_failures = 0;

#define HW_CHECK(condition, ...)                                                   \
    do {                                                                           \
        if (!(condition)) {                                                        \
            ++g_failures;                                                          \
            std::printf("  FAIL %s:%d: %s\n    ", __FILE__, __LINE__, #condition); \
            std::printf(__VA_ARGS__);                                              \
            std::printf("\n");                                                     \
        }                                                                          \
    } while (false)

// Deterministic generator (splitmix64).
struct Rng {
    std::uint64_t state;
    explicit Rng(std::uint64_t seed) : state(seed * 0x9E3779B97F4A7C15ull + 1) {}
    std::uint64_t next() {
        std::uint64_t value = (state += 0x9E3779B97F4A7C15ull);
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
        return value ^ (value >> 31);
    }
    std::uint32_t below(std::uint32_t bound) { return static_cast<std::uint32_t>(next() % bound); }
};

BufferView view(std::uint32_t bp, std::uint32_t bw, Psm psm) { return BufferView{bp, bw, psm}; }

TransferRect transfer(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h) {
    return TransferRect{x, y, w, h};
}

std::string kindsOf(const std::vector<GpuOp>& ops) {
    static const char* const kNames[] = {"create", "resize", "delete", "upload", "download", "draw", "decode"};
    std::string text;
    for (const GpuOp& op : ops) {
        if (!text.empty()) text += ' ';
        text += kNames[static_cast<int>(op.kind)];
    }
    return text;
}

// Little-endian 32-bit host data for a CT32 transfer.
std::vector<std::uint8_t> hostWords(std::uint32_t count, std::uint32_t seed) {
    std::vector<std::uint8_t> data(count * 4);
    for (std::uint32_t index = 0; index < count; ++index) {
        const std::uint32_t word = seed * 0x01000193u ^ (index * 0x9E3779B1u);
        std::memcpy(&data[index * 4], &word, 4);
    }
    return data;
}

// ---------------------------------------------------------------------------
// Chapter 04.1 geometry
// ---------------------------------------------------------------------------

void testGeometry() {
    // Page-aligned rectangle on CT32 BP 0 BW 10: the whole first page.
    const BlockRange page = blockRangeOf(view(0, 10, Psm::Ct32), Rect{0, 0, 64, 32});
    HW_CHECK(page.first == 0 && page.last == 31, "page range %lld..%lld", static_cast<long long>(page.first),
             static_cast<long long>(page.last));

    // Pages: x 64 is page 1, y 32 is page row 1 (10 pages per row).
    HW_CHECK(unwrappedBlock(view(0, 10, Psm::Ct32), 64, 0) / 32 == 1, "page of (64,0)");
    HW_CHECK(unwrappedBlock(view(0, 10, Psm::Ct32), 0, 32) / 32 == 10, "page of (0,32)");

    // Overlap with wrap (vector A = [0x3F00, 0x4100], B = [0, 0x50]).
    const BlockRange a{0x3F00, 0x4100};
    const BlockRange b{0x0000, 0x0050};
    HW_CHECK(rangesOverlap(a, b), "wrap overlap vector");
    HW_CHECK(rangesOverlap(b, a), "wrap overlap is symmetric");
    HW_CHECK(!rangesOverlap(BlockRange{0x100, 0x200}, b), "disjoint ranges must not overlap");
    HW_CHECK(rangesOverlap(a, BlockRange{0x4000, 0x4050}), "overlap past the wrap without lifting");

    HW_CHECK(rangeInside(BlockRange{0x10, 0x20}, BlockRange{0, 0x30}), "inside");
    HW_CHECK(!rangeInside(BlockRange{0, 0x31}, BlockRange{0, 0x30}), "not inside");

    // Dirty re-expression vector: CT32 (8,0)-(16,8) on a CT16 target.
    const Rect expressed = expressDirtyRect(Rect{8, 0, 16, 8}, Psm::Ct32, Psm::Ct16);
    HW_CHECK(expressed == (Rect{16, 0, 32, 8}), "re-expression vector: got (%d,%d)-(%d,%d)", expressed.x0,
             expressed.y0, expressed.x1, expressed.y1);

    // Translation vector: CT32 target BP 0x1000 BW 10; source BP 0x1160 (page 11).
    const std::optional<Rect> moved = translateRect(view(0x1160, 10, Psm::Ct32), Rect{0, 0, 64, 32},
                                                    view(0x1000, 10, Psm::Ct32));
    HW_CHECK(moved.has_value() && *moved == (Rect{64, 32, 128, 64}), "translation vector");
    HW_CHECK(!translateRect(view(0x1008, 10, Psm::Ct32), Rect{0, 0, 64, 32}, view(0x1000, 10, Psm::Ct32)).has_value(),
             "block offset must be deferred");
    HW_CHECK(!translateRect(view(0x1160, 9, Psm::Ct32), Rect{0, 0, 64, 32}, view(0x1000, 10, Psm::Ct32)).has_value(),
             "different buffer width must be deferred");
    HW_CHECK(!translateRect(view(0x1160, 10, Psm::Ct16), Rect{0, 0, 10, 32}, view(0x1000, 10, Psm::Ct32)).has_value(),
             "different bpp needs a page-aligned rectangle");

    // Rectangle algebra.
    std::int64_t area = 0;
    for (const Rect& piece : subtractRect(Rect{0, 0, 64, 64}, Rect{16, 16, 32, 32})) area += piece.area();
    HW_CHECK(area == 64 * 64 - 16 * 16, "subtract area %lld", static_cast<long long>(area));
    HW_CHECK(boundingBox(Rect{0, 0, 8, 8}, Rect{4, 4, 16, 6}) == (Rect{0, 0, 16, 8}), "bounding box");

    // Channel masks: RGB, alpha nibbles, and formats without alpha.
    HW_CHECK(channelWordMask(Psm::Ct32, kChanRgb) == 0x00FFFFFFu, "CT32 RGB mask");
    HW_CHECK(channelWordMask(Psm::Ct32, kChanAlphaLo) == 0x0F000000u, "CT32 alpha low mask");
    HW_CHECK(channelWordMask(Psm::Ct32, kChanAlphaHi) == 0xF0000000u, "CT32 alpha high mask");
    HW_CHECK(channelWordMask(Psm::Ct24, kChanAlphaLo) == 0u, "CT24 has no alpha");
    HW_CHECK(channelWordMask(Psm::Ct16, kChanAlphaHi) == 0xFFFFu, "CT16 single channel");

    // Block bits of a page-aligned rectangle are exactly the page's blocks.
    const std::vector<std::uint64_t> bits = blockBitsOf(view(0, 10, Psm::Ct32), Rect{0, 0, 64, 32});
    bool exact = true;
    for (std::uint32_t block = 0; block < 16384; ++block) {
        const bool expected = block < 32;
        if (blockBitsTest(bits, block) != expected) exact = false;
    }
    HW_CHECK(exact, "block bits of the first page must be blocks 0..31");
}

void testClampRegions() {
    // Chapter 04.3, REGION_CLAMP: a region exists when the span is narrower than
    // the texture, or the maximum reaches past it.
    HW_CHECK(!regionFromClamp(ClampMode::RegionClamp, 0, 63, 6).has_value(), "clamp span equal to width: no region");
    HW_CHECK(regionFromClamp(ClampMode::RegionClamp, 0, 31, 6)->begin == 0 &&
                 regionFromClamp(ClampMode::RegionClamp, 0, 31, 6)->end == 32,
             "clamp narrower span");
    const auto fixed = regionFromClamp(ClampMode::RegionClamp, 0, 100, 6);
    HW_CHECK(fixed.has_value() && fixed->end == 101, "clamp fixed TEX0 extends past 2^TW");
    HW_CHECK(!regionFromClamp(ClampMode::RegionClamp, 9, 3, 6).has_value(), "inverted clamp span");

    // REGION_REPEAT (derived from the table): starts at MAXU and wraps via the mask.
    const auto repeat = regionFromClamp(ClampMode::RegionRepeat, 5, 10, 4);
    HW_CHECK(repeat.has_value() && repeat->begin == 10 && repeat->end == 16, "repeat region");
    const auto narrow = regionFromClamp(ClampMode::RegionRepeat, 4, 7, 3);
    HW_CHECK(narrow.has_value() && narrow->begin == 7 && narrow->end == 8, "repeat narrow region");
    HW_CHECK(!regionFromClamp(ClampMode::RegionRepeat, 0, 7, 3).has_value(), "repeat from zero: no region");
}

// ---------------------------------------------------------------------------
// Chapter 05.1 dirty lists
// ---------------------------------------------------------------------------

DirtyEntry entry(Rect rect, std::uint8_t channels = kChanAll) {
    return DirtyEntry{rect, Psm::Ct32, 1, channels};
}

void testDirtyMerge() {
    // Vertical stack with shared x edges merges.
    DirtyList stacked;
    stacked.add(entry(Rect{0, 0, 64, 32}), Rect{0, 0, 128, 128});
    stacked.add(entry(Rect{0, 32, 64, 64}), Rect{0, 0, 128, 128});
    HW_CHECK(stacked.entries().size() == 1 && stacked.entries()[0].rect == (Rect{0, 0, 64, 64}),
             "vertical merge vector: %zu entries", stacked.entries().size());

    // Horizontal neighbours with shared y edges merge.
    DirtyList side;
    side.add(entry(Rect{0, 0, 64, 32}), Rect{0, 0, 128, 64});
    side.add(entry(Rect{64, 0, 128, 32}), Rect{0, 0, 128, 64});
    HW_CHECK(side.entries().size() == 1 && side.entries()[0].rect == (Rect{0, 0, 128, 32}),
             "horizontal merge vector");

    // A rectangle already contained is dropped.
    DirtyList contained;
    contained.add(entry(Rect{0, 0, 64, 64}), Rect{0, 0, 64, 64});
    contained.add(entry(Rect{8, 8, 16, 16}), Rect{0, 0, 64, 64});
    HW_CHECK(contained.entries().size() == 1, "contained rectangle must be dropped");

    // Different channel masks never merge.
    DirtyList channels;
    channels.add(entry(Rect{0, 0, 64, 32}, kChanRgb), Rect{0, 0, 64, 64});
    channels.add(entry(Rect{0, 32, 64, 64}, kChanAll), Rect{0, 0, 64, 64});
    HW_CHECK(channels.entries().size() == 2, "channel masks must not merge");

    // Entries are clipped to the valid area.
    DirtyList clipped;
    clipped.add(entry(Rect{0, 0, 256, 256}), Rect{0, 0, 64, 64});
    HW_CHECK(clipped.entries().size() == 1 && clipped.entries()[0].rect == (Rect{0, 0, 64, 64}), "clip to valid");

    // Fully dirty: one entry with the channels, covering the whole valid area.
    DirtyList full;
    full.add(entry(Rect{0, 0, 64, 64}), Rect{0, 0, 64, 64});
    HW_CHECK(full.fullyDirty(Rect{0, 0, 64, 64}, kChanAll), "fully dirty vector");
    DirtyList rgbOnly;
    rgbOnly.add(entry(Rect{0, 0, 64, 64}, kChanRgb), Rect{0, 0, 64, 64});
    HW_CHECK(!rgbOnly.fullyDirty(Rect{0, 0, 64, 64}, kChanAll), "RGB-only dirt is not fully dirty");
    HW_CHECK(rgbOnly.fullyDirty(Rect{0, 0, 64, 64}, kChanRgb), "RGB dirt covers RGB");
    DirtyList two;
    two.add(entry(Rect{0, 0, 32, 64}), Rect{0, 0, 64, 64});
    two.add(entry(Rect{40, 0, 64, 64}), Rect{0, 0, 64, 64});
    HW_CHECK(!two.fullyDirty(Rect{0, 0, 64, 64}, kChanAll), "two entries are not fully dirty");
    HW_CHECK(two.overlaps(Rect{30, 0, 35, 4}) && !two.overlaps(Rect{33, 0, 38, 4}), "overlap query");
}

// ---------------------------------------------------------------------------
// Chapter 04.10 ageing and 05.9 transfer log
// ---------------------------------------------------------------------------

void testAgeing() {
    // Non-preloadable memory source (600 pixels is above the 512 limit): removed
    // at age 4.
    {
        fake::FakeGpu gpu;
        SurfaceCache cache(gpu);
        cache.sampleSource(view(0, 10, Psm::Ct32), Rect{0, 0, 600, 600});
        HW_CHECK(cache.sources().size() == 1, "source created");
        for (int vblank = 0; vblank < 3; ++vblank) cache.endVBlank(false);
        HW_CHECK(cache.sources().size() == 1, "non-preloadable source alive at age 3");
        cache.endVBlank(false);
        HW_CHECK(cache.sources().empty(), "non-preloadable source removed at age 4");
    }
    // Preloadable source: removed at age 31.
    {
        fake::FakeGpu gpu;
        SurfaceCache cache(gpu);
        cache.sampleSource(view(0, 1, Psm::Ct32), Rect{0, 0, 64, 64});
        for (int vblank = 0; vblank < 30; ++vblank) cache.endVBlank(false);
        HW_CHECK(cache.sources().size() == 1, "preloadable source alive at age 30");
        cache.endVBlank(false);
        HW_CHECK(cache.sources().empty(), "preloadable source removed at age 31");
    }
    // Target unused for 61 non-idle VBlanks and overlapped by nothing survives with age 10.
    {
        fake::FakeGpu gpu;
        SurfaceCache cache(gpu);
        cache.draw(DrawCommand{view(0, 1, Psm::Ct32), Rect{0, 0, 64, 64}, 3, kChanAll});
        for (int vblank = 0; vblank < 61; ++vblank) cache.endVBlank(false);
        HW_CHECK(cache.targets().size() == 1, "unused target survives");
        HW_CHECK(cache.targets().size() == 1 && cache.targets()[0].age == 10, "target age reset to 10: %u",
                 cache.targets().empty() ? 0u : cache.targets()[0].age);
    }
    // Idle VBlanks age nothing.
    {
        fake::FakeGpu gpu;
        SurfaceCache cache(gpu);
        cache.draw(DrawCommand{view(0, 1, Psm::Ct32), Rect{0, 0, 64, 64}, 3, kChanAll});
        for (int vblank = 0; vblank < 100; ++vblank) cache.endVBlank(true);
        HW_CHECK(cache.targets()[0].age == 0, "idle VBlanks must not age");
    }
    // Transfer log: an old write is cleared, a recent one keeps its records.
    {
        fake::FakeGpu gpu;
        SurfaceCache cache(gpu);
        std::vector<std::uint8_t> data = hostWords(64 * 32, 1);
        cache.hostUpload(view(0, 1, Psm::Ct32), transfer(0, 0, 64, 32), data);
        for (int batch = 0; batch < 6; ++batch) cache.closeBatch();
        cache.endVBlank(false);
        HW_CHECK(cache.transferLog().records().empty(), "log cleared when the last write is 5+ batches old");

        cache.hostUpload(view(0, 1, Psm::Ct32), transfer(0, 0, 64, 32), data);
        cache.closeBatch();
        cache.closeBatch();
        cache.endVBlank(false);
        HW_CHECK(cache.transferLog().records().size() == 1, "recent log record kept");
    }
    {
        // Records older than 50 batches go when the last write is recent.
        TransferLog log;
        log.record(LogKind::Upload, view(0, 1, Psm::Ct32), Rect{0, 0, 8, 8}, 0);
        log.record(LogKind::Upload, view(64, 1, Psm::Ct32), Rect{0, 0, 8, 8}, 52);
        log.vblank(53);  // last write 1 batch ago: drop records older than 50 batches
        HW_CHECK(log.records().size() == 1 && log.records()[0].batch == 52, "old records trimmed");
        log.vblank(60);  // last write 8 batches ago (5 or more): clear
        HW_CHECK(log.records().empty(), "log cleared when the last write is 5 or more batches old");
    }
}

// ---------------------------------------------------------------------------
// Operation lists (chapters 05.4, 05.2, 05.3, 04.6)
// ---------------------------------------------------------------------------

void testOperationLists() {
    fake::FakeGpu gpu;
    SurfaceCache cache(gpu);
    const BufferView frame2 = view(0, 2, Psm::Ct32);

    cache.draw(DrawCommand{frame2, Rect{0, 0, 64, 64}, 1, kChanAll});
    HW_CHECK(kindsOf(cache.operations()) == "create draw", "first draw: %s", kindsOf(cache.operations()).c_str());
    cache.clearOperations();

    // Growing downward resizes the image and needs no upload (no new area).
    cache.draw(DrawCommand{frame2, Rect{0, 64, 64, 128}, 2, kChanAll});
    HW_CHECK(kindsOf(cache.operations()) == "resize draw", "growth: %s", kindsOf(cache.operations()).c_str());
    cache.clearOperations();

    // Upload over the drawn top half: the drawn part is written back first and
    // the uploaded area becomes dirty.
    std::vector<std::uint8_t> data = hostWords(64 * 32, 9);
    cache.hostUpload(frame2, transfer(0, 0, 64, 32), data);
    HW_CHECK(kindsOf(cache.operations()) == "download", "upload write-back: %s", kindsOf(cache.operations()).c_str());
    HW_CHECK(cache.operations().size() == 1 && cache.operations()[0].rect == (Rect{0, 0, 64, 32}),
             "download covers the uploaded area");
    HW_CHECK(cache.targets().size() == 1 && cache.targets()[0].drawn == (Rect{0, 32, 64, 128}),
             "drawn band trimmed to the rows not written back");
    HW_CHECK(cache.targets()[0].dirty.entries().size() == 1, "uploaded area is dirty");
    cache.clearOperations();

    // Reading the exact uploaded rectangle needs no write-back (10.2 shortcut).
    std::optional<std::vector<std::uint8_t>> host = cache.readHost(frame2, transfer(0, 0, 64, 32));
    HW_CHECK(host.has_value() && *host == data, "exact upload reads back");
    HW_CHECK(cache.operations().empty(), "shortcut: no operations");
    HW_CHECK(cache.counters().readbackShortcuts == 1, "shortcut counted");

    // A read of a larger rectangle applies dirty data, then writes back the drawn rows.
    host = cache.readHost(frame2, transfer(0, 0, 64, 64));
    HW_CHECK(host.has_value(), "larger read");
    HW_CHECK(kindsOf(cache.operations()) == "upload download", "selective read: %s",
             kindsOf(cache.operations()).c_str());
    cache.clearOperations();

    // A draw on a different base width that shares memory conflicts with the
    // existing target: write back, delete, create the new one.
    cache.draw(DrawCommand{view(0, 4, Psm::Ct32), Rect{0, 0, 64, 64}, 3, kChanAll});
    HW_CHECK(kindsOf(cache.operations()) == "download delete create draw",
             "conflict sequence: %s", kindsOf(cache.operations()).c_str());
    HW_CHECK(cache.counters().conflictDeletes == 1, "conflict counted");
    HW_CHECK(cache.targets().size() == 1 && cache.targets()[0].view.bufferWidth == 4, "new target kept");
}

// ---------------------------------------------------------------------------
// Chapter 10 write-back vector (1x part of V-10-1)
// ---------------------------------------------------------------------------

void testWriteBackAlphaInvalid() {
    fake::FakeGpu gpu;
    SurfaceCache cache(gpu);
    fake::ReferenceMemory reference;

    const BufferView frame = view(0, 1, Psm::Ct32);
    std::vector<std::uint8_t> original = hostWords(64 * 32, 4);
    cache.hostUpload(frame, transfer(0, 0, 64, 32), original);
    reference.hostUpload(frame, transfer(0, 0, 64, 32), original);

    // RGB-only write: the target owns RGB; alpha stays with the SLM.
    const DrawCommand command{frame, Rect{0, 0, 64, 32}, 7, kChanRgb};
    HW_CHECK(cache.draw(command), "RGB draw accepted");
    reference.draw(command);

    std::optional<std::vector<std::uint8_t>> host = cache.readHost(frame, transfer(0, 0, 64, 32));
    HW_CHECK(host.has_value(), "V-10-1 read");
    std::vector<std::uint8_t> expected;
    memory::localToHost(reference.slm, frame, transfer(0, 0, 64, 32), expected);
    HW_CHECK(host.has_value() && *host == expected, "V-10-1: RGB written, alpha bytes unchanged");

    bool alphaUnchanged = true;
    for (std::size_t index = 0; index < original.size(); index += 4) {
        if (host.has_value() && (*host)[index + 3] != original[index + 3]) alphaUnchanged = false;
    }
    HW_CHECK(alphaUnchanged, "V-10-1: alpha bytes unchanged");
}

// ---------------------------------------------------------------------------
// Targets, sources, display, moves
// ---------------------------------------------------------------------------

void testSourcesDisplayMoves() {
    fake::FakeGpu gpu;
    SurfaceCache cache(gpu);
    fake::ReferenceMemory reference;
    const BufferView frame = view(0, 2, Psm::Ct32);

    const DrawCommand command{frame, Rect{0, 0, 64, 64}, 5, kChanAll};
    cache.draw(command);
    reference.draw(command);

    // Target view: the draw is sampled directly from the target.
    cache.clearOperations();
    SourceRef source = cache.sampleSource(frame, Rect{0, 0, 32, 32});
    HW_CHECK(source.valid && source.kind == SourceKind::TargetView, "target view expected");
    HW_CHECK(cache.operations().empty(), "target view needs no operation");
    HW_CHECK(cache.texel(source, 3, 4) == fake::drawPattern(5, 3, 4), "target view texel");

    // Display of the drawn base returns the target; a stranger base reads the SLM.
    const DisplayResult shown = cache.displayLookup(frame, Rect{0, 0, 64, 64});
    HW_CHECK(shown.fromTarget, "display from target");
    const DisplayResult other = cache.displayLookup(view(512, 2, Psm::Ct32), Rect{0, 0, 64, 64});
    HW_CHECK(!other.fromTarget, "display from SLM");

    // Move the drawn area inside the SLM: the source is written back first.
    cache.localMove(frame, 0, 0, view(256, 2, Psm::Ct32), 0, 0, 64, 32, memory::TransferDirection::UpperLeftToLowerRight);
    reference.localMove(frame, 0, 0, view(256, 2, Psm::Ct32), 0, 0, 64, 32,
                        memory::TransferDirection::UpperLeftToLowerRight);
    HW_CHECK(cache.counters().moves == 1, "move counted");

    // A memory source decoded after the draw sees the drawn pixels.
    SourceRef moved = cache.sampleSource(view(256, 2, Psm::Ct32), Rect{0, 0, 64, 32});
    HW_CHECK(moved.valid && moved.kind == SourceKind::Memory, "memory source after move");
    bool same = true;
    for (std::uint32_t y = 0; y < 32; ++y) {
        for (std::uint32_t x = 0; x < 64; ++x) {
            if (cache.texel(moved, x, y) != reference.slm.readPixel(view(256, 2, Psm::Ct32), x, y)) same = false;
        }
    }
    HW_CHECK(same, "memory source after move equals the reference");

    cache.writeBackAll();
    HW_CHECK(std::memcmp(cache.slm().bytes().data(), reference.slm.bytes().data(), cache.slm().bytes().size()) == 0,
             "whole SLM equals the reference after write-back");
    HW_CHECK(gpu.outOfBounds() == 0, "no out-of-bounds image access");
}

// ---------------------------------------------------------------------------
// Budget (chapter 04.10)
// ---------------------------------------------------------------------------

void testBudgetEviction() {
    fake::FakeGpu gpu;
    SurfaceCache cache(gpu);
    fake::ReferenceMemory reference;
    cache.setBudgetBytes(64 * 64 * 4 + 1024);

    const DrawCommand first{view(0, 1, Psm::Ct32), Rect{0, 0, 64, 64}, 11, kChanAll};
    const DrawCommand second{view(256, 1, Psm::Ct32), Rect{0, 0, 64, 64}, 12, kChanAll};
    cache.draw(first);
    reference.draw(first);
    cache.draw(second);
    reference.draw(second);
    HW_CHECK(cache.targets().size() == 2, "two targets before the budget check");

    cache.endVBlank(false);  // nothing is unused for two VBlanks yet
    HW_CHECK(cache.targets().size() == 2, "no eviction after one VBlank");
    cache.endVBlank(false);
    HW_CHECK(cache.targets().size() == 1, "least recently used target evicted");
    HW_CHECK(cache.counters().budgetEvictions == 1, "eviction counted");

    cache.writeBackAll();
    HW_CHECK(std::memcmp(cache.slm().bytes().data(), reference.slm.bytes().data(), cache.slm().bytes().size()) == 0,
             "evicted target was written back");
}

// ---------------------------------------------------------------------------
// Authority fuzz: random draws, uploads, moves and reads against the reference
// ---------------------------------------------------------------------------

BufferView randomView(Rng& rng, bool drawable) {
    static const Psm kDrawable[] = {Psm::Ct32, Psm::Ct24, Psm::Ct16, Psm::Ct16S};
    static const Psm kAny[] = {Psm::Ct32, Psm::Ct24, Psm::Ct16, Psm::Ct16S, Psm::T8, Psm::T8H, Psm::T4Hl, Psm::T4Hh};
    const Psm psm = drawable ? kDrawable[rng.below(4)] : kAny[rng.below(8)];
    const std::uint32_t base = (rng.below(40) * memory::kBlocksPerPage) + (rng.below(8) == 0 ? 8u : 0u);
    const std::uint32_t width = 1 + rng.below(5);
    return BufferView{base, width, psm};
}

Rect randomRect(Rng& rng, std::uint32_t maxWidth, std::uint32_t maxHeight) {
    const std::int32_t x0 = static_cast<std::int32_t>(rng.below(128));
    const std::int32_t y0 = static_cast<std::int32_t>(rng.below(96));
    return Rect{x0, y0, x0 + 1 + static_cast<std::int32_t>(rng.below(maxWidth)),
                y0 + 1 + static_cast<std::int32_t>(rng.below(maxHeight))};
}

// Authority invariant of chapter 00, checked after every operation. A valid
// target pixel that is not dirty must hold the reference truth in its GPU word
// (on the target's channels); a dirty pixel must hold the truth in the SLM.
int countAuthorityViolations(const SurfaceCache& cache, const fake::FakeGpu& gpu,
                             const fake::ReferenceMemory& reference) {
    int violations = 0;
    for (const Target& target : cache.targets()) {
        const std::uint32_t mask = channelWordMask(target.view.psm, target.channels);
        for (std::int32_t y = target.valid.y0; y < target.valid.y1; ++y) {
            for (std::int32_t x = target.valid.x0; x < target.valid.x1; ++x) {
                const auto px = static_cast<std::uint32_t>(x);
                const auto py = static_cast<std::uint32_t>(y);
                const std::uint32_t truth = reference.slm.readPixel(target.view, px, py);
                const bool dirty = target.dirty.overlaps(Rect{x, y, x + 1, y + 1});
                const std::uint32_t held = dirty ? cache.slm().readPixel(target.view, px, py)
                                                 : gpu.texel(target.id, px, py);
                if (((held ^ truth) & mask) != 0) ++violations;
            }
        }
    }
    return violations;
}

int runAuthorityFuzz(std::uint64_t sequences, int steps, int& failures) {
    int checks = 0;
    for (std::uint64_t sequence = 0; sequence < sequences; ++sequence) {
        Rng rng(sequence + 1);
        fake::FakeGpu gpu;
        SurfaceCache cache(gpu);
        fake::ReferenceMemory reference;
        if (sequence % 3 == 0) cache.setBudgetBytes(96 * 1024);

        for (int step = 0; step < steps; ++step) {
            const std::uint32_t pick = rng.below(100);
            if (pick < 30) {
                const BufferView target = randomView(rng, true);
                static const std::uint8_t kWrites[] = {kChanAll, kChanRgb, kChanAll, kChanAlphaLo, 0x6};
                const DrawCommand command{target, randomRect(rng, 96, 64), rng.below(1000) + 1,
                                          kWrites[rng.below(5)]};
                if (cache.draw(command)) reference.draw(command);
            } else if (pick < 55) {
                const BufferView destination = randomView(rng, false);
                const Rect area = randomRect(rng, 96, 64);
                const TransferRect transfer{static_cast<std::uint32_t>(area.x0), static_cast<std::uint32_t>(area.y0),
                                            static_cast<std::uint32_t>(area.width()),
                                            static_cast<std::uint32_t>(area.height())};
                std::vector<std::uint8_t> data(memory::hostBytesFor(destination.psm, transfer.width, transfer.height));
                for (std::uint8_t& byte : data) byte = static_cast<std::uint8_t>(rng.next());
                cache.hostUpload(destination, transfer, data);
                reference.hostUpload(destination, transfer, data);
            } else if (pick < 62) {
                const BufferView source = randomView(rng, false);
                const BufferView destination = randomView(rng, false);
                const Rect area = randomRect(rng, 48, 32);
                const auto direction = static_cast<memory::TransferDirection>(rng.below(4));
                cache.localMove(source, static_cast<std::uint32_t>(area.x0), static_cast<std::uint32_t>(area.y0),
                                destination, static_cast<std::uint32_t>(area.x0 + 5),
                                static_cast<std::uint32_t>(area.y0 + 3), static_cast<std::uint32_t>(area.width()),
                                static_cast<std::uint32_t>(area.height()), direction);
                reference.localMove(source, static_cast<std::uint32_t>(area.x0), static_cast<std::uint32_t>(area.y0),
                                    destination, static_cast<std::uint32_t>(area.x0 + 5),
                                    static_cast<std::uint32_t>(area.y0 + 3), static_cast<std::uint32_t>(area.width()),
                                    static_cast<std::uint32_t>(area.height()), direction);
            } else if (pick < 77) {
                const BufferView source = randomView(rng, false);
                if (memory::psmInfo(source.psm).transferBpp == 4) continue;
                const Rect area = randomRect(rng, 96, 64);
                const TransferRect transfer{static_cast<std::uint32_t>(area.x0), static_cast<std::uint32_t>(area.y0),
                                            static_cast<std::uint32_t>(area.width()),
                                            static_cast<std::uint32_t>(area.height())};
                const std::optional<std::vector<std::uint8_t>> host = cache.readHost(source, transfer);
                std::vector<std::uint8_t> expected;
                memory::localToHost(reference.slm, source, transfer, expected);
                ++checks;
                if (!host.has_value() || *host != expected) {
                    ++failures;
                    std::printf("  FAIL readHost mismatch seq=%llu step=%d bp=%u bw=%u psm=0x%02x\n",
                                static_cast<unsigned long long>(sequence), step, source.basePointer,
                                source.bufferWidth, static_cast<unsigned>(source.psm));
                }
            } else if (pick < 85) {
                const std::uint32_t address = rng.below(memory::kLocalMemoryBytes);
                const std::uint32_t biased = rng.below(2) ? address % (64u * 256u * 40u) : address;
                ++checks;
                if (cache.readByte(biased) != reference.slm.byte(biased)) {
                    ++failures;
                    std::printf("  FAIL readByte mismatch seq=%llu step=%d addr=%u\n",
                                static_cast<unsigned long long>(sequence), step, biased);
                }
            } else if (pick < 93) {
                const BufferView view = randomView(rng, false);
                const Rect region = randomRect(rng, 48, 40);
                const SourceRef source = cache.sampleSource(view, region);
                for (std::int32_t y = 0; y < region.height(); ++y) {
                    for (std::int32_t x = 0; x < region.width(); ++x) {
                        const std::uint32_t expected = reference.slm.readPixel(
                            view, static_cast<std::uint32_t>(region.x0 + x), static_cast<std::uint32_t>(region.y0 + y));
                        ++checks;
                        if (cache.texel(source, static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y)) !=
                            expected) {
                            ++failures;
                            std::printf("  FAIL source texel seq=%llu step=%d kind=%d (%d,%d)\n",
                                        static_cast<unsigned long long>(sequence), step,
                                        static_cast<int>(source.kind), x, y);
                            x = region.width();
                            y = region.height();
                        }
                    }
                }
            } else if (pick < 96) {
                cache.closeBatch();
            } else {
                cache.endVBlank(rng.below(5) == 0);
            }
            const int violations = countAuthorityViolations(cache, gpu, reference);
            checks += 1;
            if (violations != 0) {
                ++failures;
                std::printf("  FAIL authority invariant seq=%llu step=%d violations=%d\n",
                            static_cast<unsigned long long>(sequence), step, violations);
                break;
            }
        }

        // Final check: after a full write-back the whole SLM equals the reference.
        cache.writeBackAll();
        ++checks;
        if (std::memcmp(cache.slm().bytes().data(), reference.slm.bytes().data(), cache.slm().bytes().size()) != 0) {
            ++failures;
            std::printf("  FAIL final SLM mismatch seq=%llu\n", static_cast<unsigned long long>(sequence));
        }
        if (gpu.outOfBounds() != 0) {
            ++failures;
            std::printf("  FAIL image access out of bounds seq=%llu\n", static_cast<unsigned long long>(sequence));
        }
    }
    return checks;
}

void testAuthorityFuzz() {
    int failures = 0;
    const int checks = runAuthorityFuzz(400, 80, failures);
    HW_CHECK(failures == 0, "authority fuzz failures: %d of %d checks", failures, checks);
    std::printf("  authority fuzz: %d checks\n", checks);
    g_failures += failures;
}

}  // namespace

int main(int argc, char** argv) {
    struct Test {
        const char* name;
        void (*run)();
    };
    const Test tests[] = {
        {"hw_cache_geometry", testGeometry},
        {"hw_clamp_regions", testClampRegions},
        {"hw_dirty_merge", testDirtyMerge},
        {"hw_ageing", testAgeing},
        {"hw_cache_operations", testOperationLists},
        {"hw_v10_alpha_writeback", testWriteBackAlphaInvalid},
        {"hw_cache_sources_display_moves", testSourcesDisplayMoves},
        {"hw_cache_budget", testBudgetEviction},
        {"hw_cache_authority_fuzz", testAuthorityFuzz},
    };
    const std::string requested = argc > 1 ? argv[1] : "";
    int ran = 0;
    for (const Test& test : tests) {
        if (!requested.empty() && requested != test.name) continue;
        const int before = g_failures;
        std::printf("[ RUN  ] %s\n", test.name);
        test.run();
        std::printf("[ %s ] %s\n", g_failures == before ? " OK " : "FAIL", test.name);
        ++ran;
    }
    if (ran == 0) {
        std::printf("no test named %s\n", requested.c_str());
        return 2;
    }
    return g_failures == 0 ? 0 : 1;
}
