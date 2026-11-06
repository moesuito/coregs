// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors
//
// HWR texture and render-target cache bookkeeping (TASK-027, milestone M7).
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters
// 00 (data-authority rule), 04 (sources), 05 (targets), 10 (readbacks), and the
// M7 scope of chapter 15. Clean room: no code from other GS renderers.
//
// The cache is GPU-free. Every GPU action is recorded as an abstract GpuOp and
// handed to an ImageBackend, which is the GPU executor in the real renderer and
// a fake GPU in the tests. The cache owns the shadow local memory (SLM).
//
// Data-authority rule (chapter 00): every byte of GS memory has exactly one
// authoritative copy, the SLM or one target. A target pixel is authoritative
// when it is drawn and not in the dirty list; dirty pixels are owned by the SLM.

#pragma once

#include "hw_cache_dirty.hpp"
#include "hw_cache_geometry.hpp"
#include "hw_memory.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace coregs::cache {

// ---------------------------------------------------------------------------
// GPU operations (abstract; the M7/M8 GPU code executes them later)
// ---------------------------------------------------------------------------

enum class OpKind : std::uint8_t {
    CreateImage,         // new target image of width x height (chapter 05.4)
    ResizeImage,         // grow an image, keeping its contents
    DeleteImage,         // delete a target or a source image
    UploadRect,          // SLM -> image, channels in `channels` (dirty update, growth, reload)
    DownloadRect,        // image -> SLM, channels in `channels` (write-back, chapter 10.4)
    Draw,                // draw pattern `drawId` into the target, channels in `channels`
    DecodeMemorySource,  // decode a memory source from the SLM (chapter 04.6)
};

struct GpuOp {
    OpKind kind = OpKind::CreateImage;
    std::uint32_t imageId = 0;
    BufferView view{};         // layout of the image in GS memory (not used by Resize/Delete)
    Rect rect{};               // rectangle in the image coordinates of `view`
    std::uint32_t width = 0;   // CreateImage / ResizeImage image size
    std::uint32_t height = 0;
    std::uint8_t channels = kChanAll;
    std::uint32_t drawId = 0;
};

// Human-readable form of one operation, used by the operation-list tests.
std::string describeOp(const GpuOp& op);

// Executor of GPU operations. `slm` is the shadow memory the cache owns; the
// backend reads it for uploads and decodes and writes it for downloads.
class ImageBackend {
public:
    virtual ~ImageBackend() = default;
    virtual void execute(const GpuOp& op, memory::LocalMemory& slm) = 0;
    // Texel of an image, in the image's own coordinates.
    virtual std::uint32_t texel(std::uint32_t imageId, std::uint32_t x, std::uint32_t y) const = 0;
};

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

struct DrawCommand {
    BufferView frame{};                 // FRAME layout (colour target)
    Rect rect{};                        // drawn rectangle in frame coordinates
    std::uint32_t drawId = 0;           // selects the pixel pattern (fake GPU and reference)
    std::uint8_t writeChannels = kChanAll;  // channel write mask (FBMSK-like)
};

struct Target {
    std::uint32_t id = 0;
    BufferView view{};
    Rect valid{};                // GS pixels holding meaningful data
    Rect drawn{};                // drawn since the last write-back (empty when none)
    DirtyList dirty;             // SLM newer than the target
    std::uint8_t channels = kChanAll;
    std::uint32_t imageWidth = 0;
    std::uint32_t imageHeight = 0;
    std::uint32_t age = 0;       // ageing counter (chapter 04.10)
    std::uint32_t vblanksSinceUse = 0;
    std::uint32_t readsSinceDraw = 0;
    std::uint64_t lastUse = 0;   // LRU sequence number
    bool displayed = false;      // returned by a display lookup in this frame
};

enum class SourceKind : std::uint8_t {
    Memory,      // decoded from the SLM
    TargetView,  // samples a target image directly
};

struct MemorySource {
    std::uint32_t id = 0;
    SourceKind kind = SourceKind::Memory;
    BufferView view{};
    Rect region{};
    bool preloadable = false;    // chapter 04.10 ageing class
    std::uint32_t age = 0;
    std::uint64_t lastUse = 0;
};

// Result of a source lookup. `imageId` is valid when kind is Memory or TargetView.
struct SourceRef {
    bool valid = false;
    SourceKind kind = SourceKind::Memory;
    std::uint32_t imageId = 0;
    Rect region{};
};

struct DisplayResult {
    bool fromTarget = false;     // false: the output reads the SLM
    std::uint32_t targetId = 0;
};

enum class LogKind : std::uint8_t { Upload, Move, Clear };

struct LogRecord {
    LogKind kind = LogKind::Upload;
    BufferView view{};
    Rect rect{};
    std::uint64_t batch = 0;
    bool gpuOnly = false;
};

// Records of recent memory writes (chapter 05.9).
class TransferLog {
public:
    // Consecutive writes of the same kind to the same base merge by union.
    void record(LogKind kind, const BufferView& view, const Rect& rect, std::uint64_t batch, bool gpuOnly = false);

    const std::vector<LogRecord>& records() const { return records_; }

    // Newest-first search for an upload that is exactly this rectangle.
    const LogRecord* findExactUpload(const BufferView& view, const Rect& rect) const;

    // Removes every record for which the predicate returns true.
    template <typename Predicate>
    void eraseIf(Predicate predicate) {
        records_.erase(std::remove_if(records_.begin(), records_.end(), predicate), records_.end());
    }

    // Non-idle VBlank rule (chapter 05.9): when the last write is fewer than 5
    // batches old, drop records older than 50 batches; otherwise clear.
    void vblank(std::uint64_t currentBatch);

    void clear() { records_.clear(); }

private:
    std::vector<LogRecord> records_;
};

struct CacheCounters {
    std::uint64_t targetsCreated = 0;
    std::uint64_t targetsDeleted = 0;
    std::uint64_t targetsResized = 0;
    std::uint64_t conflictDeletes = 0;
    std::uint64_t draws = 0;
    std::uint64_t drawsRefused = 0;
    std::uint64_t uploads = 0;
    std::uint64_t moves = 0;
    std::uint64_t writeBackRects = 0;
    std::uint64_t writeBackSkips = 0;
    std::uint64_t readbackShortcuts = 0;
    std::uint64_t hostReads = 0;
    std::uint64_t sourceLookups = 0;
    std::uint64_t targetViews = 0;
    std::uint64_t memorySourceHits = 0;
    std::uint64_t memorySourceCreates = 0;
    std::uint64_t sourcesDropped = 0;
    std::uint64_t budgetEvictions = 0;
    std::uint64_t displayTargetHits = 0;
    std::uint64_t displaySlm = 0;
};

// ---------------------------------------------------------------------------
// The cache
// ---------------------------------------------------------------------------

class SurfaceCache {
public:
    explicit SurfaceCache(ImageBackend& backend);

    memory::LocalMemory& slm() { return slm_; }
    const memory::LocalMemory& slm() const { return slm_; }

    // HOST->LOCAL transfer (chapter 02.7): targets and sources that overlap the
    // written pixels are invalidated first (chapter 05.6, 04.8).
    void hostUpload(const BufferView& destination, const memory::TransferRect& rect,
                    std::span<const std::uint8_t> data);

    // LOCAL->LOCAL move, write-back fallback of chapter 05.7 (no GPU move yet).
    void localMove(const BufferView& source, std::uint32_t sourceX, std::uint32_t sourceY,
                   const BufferView& destination, std::uint32_t destinationX, std::uint32_t destinationY,
                   std::uint32_t width, std::uint32_t height, memory::TransferDirection direction);

    // Draw into a colour target. Creates, grows or reuses the target (chapter
    // 05.3, exact base only). Returns false when refused.
    bool draw(const DrawCommand& command);

    // LOCAL->HOST read (chapter 10.2 and 10.3). Returns nullopt for 4-bit
    // sources (illegal for LOCAL->HOST).
    std::optional<std::vector<std::uint8_t>> readHost(const BufferView& source, const memory::TransferRect& rect);

    // Coherent single-byte read: writes back every target whose blocks include
    // the byte, then reads the SLM. Diagnostic-grade (chapter 10.1).
    std::uint8_t readByte(std::uint32_t address);

    // Source lookup for a sampled region (chapter 04.4, simplified).
    SourceRef sampleSource(const BufferView& view, const Rect& region);
    std::uint32_t texel(const SourceRef& source, std::uint32_t x, std::uint32_t y) const;

    // Display lookup (chapter 05.10, steps 1 and 5 only).
    DisplayResult displayLookup(const BufferView& view, const Rect& rect);

    // Every target becomes coherent with the SLM (chapter 10.1, reference mode).
    void writeBackAll();

    // Batch and frame boundaries.
    void closeBatch() { ++batch_; }
    void endVBlank(bool idle);
    void setBudgetBytes(std::uint64_t bytes) { budgetBytes_ = bytes; }

    // Inspection.
    const std::vector<Target>& targets() const { return targets_; }
    const std::vector<MemorySource>& sources() const { return sources_; }
    const std::vector<GpuOp>& operations() const { return operations_; }
    void clearOperations() { operations_.clear(); }
    const TransferLog& transferLog() const { return log_; }
    const CacheCounters& counters() const { return counters_; }
    std::uint64_t batch() const { return batch_; }
    const Target* findTarget(std::uint32_t id) const;

private:
    static constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);

    void emit(const GpuOp& op);
    std::size_t findExactIndex(const BufferView& view) const;
    std::size_t indexOfId(std::uint32_t id) const;
    void applyDirty(std::size_t index);
    void writeBackTarget(std::size_t index, const Rect& region);
    void shrinkDrawn(Target& target, const Rect& written);
    void writeBackOverlapping(const BufferView& view, const Rect& rect);
    void invalidateTargetsForWrite(const std::vector<std::uint64_t>& written);
    void invalidateSourcesForBlocks(const std::vector<std::uint64_t>& blocks);
    void deleteTarget(std::size_t index);
    void resolveConflicts(std::uint32_t keepId, const BufferView& view, const Rect& valid);
    void ensureImageSize(std::size_t index, std::int32_t width, std::int32_t height);
    void enforceBudget();
    std::uint64_t usageBytes() const;
    void dropTargetViews(std::uint32_t targetId);

    ImageBackend& backend_;
    memory::LocalMemory slm_;
    std::vector<Target> targets_;
    std::vector<MemorySource> sources_;
    std::vector<GpuOp> operations_;
    TransferLog log_;
    CacheCounters counters_;
    std::uint64_t batch_ = 0;
    std::uint64_t useClock_ = 0;
    std::uint32_t nextId_ = 1;
    std::uint64_t budgetBytes_ = 0;  // 0: unlimited
};

}  // namespace coregs::cache
