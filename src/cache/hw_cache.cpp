// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors
//
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters
// 00, 04, 05 and 10 (M7 scope). Clean room: no code from other GS renderers.
// Interpretations of silent points are recorded in
// docs/hw-renderer-spec/IMPLEMENTATION-NOTES.md (IN-07 and later).

#include "hw_cache.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>

namespace coregs::cache {
namespace {

constexpr std::uint32_t kBlockBytes = memory::kBlockBytes;
constexpr std::uint32_t kLocalMemoryMask = memory::kLocalMemoryBytes - 1u;

bool sameLayout(const BufferView& first, const BufferView& second) {
    return first.basePointer == second.basePointer && first.bufferWidth == second.bufferWidth &&
           first.psm == second.psm;
}

Rect rectOfTransfer(const memory::TransferRect& transfer) {
    return rectFromOrigin(static_cast<std::int32_t>(transfer.x), static_cast<std::int32_t>(transfer.y),
                          static_cast<std::int32_t>(transfer.width), static_cast<std::int32_t>(transfer.height));
}

// Preloadable sources (chapter 04.10): largest side up to 512, or up to 1024
// with the other side up to 256.
bool isPreloadableRegion(const Rect& region) {
    const std::int32_t longest = std::max(region.width(), region.height());
    const std::int32_t shortest = std::min(region.width(), region.height());
    return longest <= 512 || (longest <= 1024 && shortest <= 256);
}

// Draws are clipped to the frame width FBW * 64 (chapter 05.2 valid width).
// Pixels beyond it alias pixels of the next row in GS memory, which a
// one-word-per-pixel image cannot represent; see IMPLEMENTATION-NOTES IN-08.
Rect clipToFrameWidth(const BufferView& view, const Rect& rect) {
    const std::int32_t widthLimit = static_cast<std::int32_t>(std::max<std::uint32_t>(view.bufferWidth, 1u) * 64u);
    return rect.intersect(Rect{0, 0, widthLimit, 0x7FFFFFFF});
}

// Channels a draw writes. 16-bit formats hold one channel: any write covers
// the whole halfword, so the channel set is always complete for them.
std::uint8_t drawChannels(Psm psm, std::uint8_t requested) {
    const std::uint8_t supported = supportedChannels(psm);
    if ((requested & supported) == 0) return 0;
    if (psm == Psm::Ct16 || psm == Psm::Ct16S) return supported;
    return static_cast<std::uint8_t>(requested & supported);
}

const char* opName(OpKind kind) {
    switch (kind) {
        case OpKind::CreateImage: return "create";
        case OpKind::ResizeImage: return "resize";
        case OpKind::DeleteImage: return "delete";
        case OpKind::UploadRect: return "upload";
        case OpKind::DownloadRect: return "download";
        case OpKind::Draw: return "draw";
        case OpKind::DecodeMemorySource: return "decode";
    }
    return "?";
}

}  // namespace

std::string describeOp(const GpuOp& op) {
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer),
                  "%s id=%u bp=%u bw=%u psm=0x%02x rect=(%d,%d)-(%d,%d) size=%ux%u ch=%u draw=%u",
                  opName(op.kind), op.imageId, op.view.basePointer, op.view.bufferWidth,
                  static_cast<unsigned>(op.view.psm), op.rect.x0, op.rect.y0, op.rect.x1, op.rect.y1,
                  op.width, op.height, static_cast<unsigned>(op.channels), op.drawId);
    return buffer;
}

// ---------------------------------------------------------------------------
// Transfer log (chapter 05.9)
// ---------------------------------------------------------------------------

void TransferLog::record(LogKind kind, const BufferView& view, const Rect& rect, std::uint64_t batch, bool gpuOnly) {
    if (!records_.empty()) {
        LogRecord& newest = records_.back();
        if (newest.kind == kind && sameLayout(newest.view, view) && newest.gpuOnly == gpuOnly) {
            newest.rect = boundingBox(newest.rect, rect);
            newest.batch = batch;
            return;
        }
    }
    records_.push_back(LogRecord{kind, view, rect, batch, gpuOnly});
}

const LogRecord* TransferLog::findExactUpload(const BufferView& view, const Rect& rect) const {
    for (auto it = records_.rbegin(); it != records_.rend(); ++it) {
        if (it->kind == LogKind::Upload && sameLayout(it->view, view) && it->rect == rect) return &*it;
    }
    return nullptr;
}

void TransferLog::vblank(std::uint64_t currentBatch) {
    if (records_.empty()) return;
    const std::uint64_t lastWrite = records_.back().batch;
    if (currentBatch >= lastWrite && currentBatch - lastWrite < 5) {
        eraseIf([currentBatch](const LogRecord& record) {
            return record.batch <= currentBatch && currentBatch - record.batch > 50;
        });
    } else {
        records_.clear();
    }
}

// ---------------------------------------------------------------------------
// Cache: plumbing
// ---------------------------------------------------------------------------

SurfaceCache::SurfaceCache(ImageBackend& backend) : backend_(backend) {}

void SurfaceCache::emit(const GpuOp& op) {
    operations_.push_back(op);
    backend_.execute(op, slm_);
}

std::size_t SurfaceCache::findExactIndex(const BufferView& view) const {
    for (std::size_t index = 0; index < targets_.size(); ++index) {
        if (sameLayout(targets_[index].view, view)) return index;
    }
    return kNoIndex;
}

std::size_t SurfaceCache::indexOfId(std::uint32_t id) const {
    for (std::size_t index = 0; index < targets_.size(); ++index) {
        if (targets_[index].id == id) return index;
    }
    return kNoIndex;
}

const Target* SurfaceCache::findTarget(std::uint32_t id) const {
    const std::size_t index = indexOfId(id);
    return index == kNoIndex ? nullptr : &targets_[index];
}

// Channel validity is per target, so a channel that becomes valid must hold
// the SLM value on every valid pixel, not only on the dirty rectangle: the
// missing channels are loaded over the whole valid area first.
void SurfaceCache::applyDirty(std::size_t index) {
    if (targets_[index].dirty.empty()) return;
    Target& target = targets_[index];
    const std::uint8_t supported = supportedChannels(target.view.psm);
    std::uint8_t wanted = 0;
    for (const DirtyEntry& entry : target.dirty.entries()) wanted |= entry.channels;
    wanted = static_cast<std::uint8_t>(wanted & supported);
    const std::uint8_t missing = static_cast<std::uint8_t>(wanted & ~target.channels);
    if (missing != 0) {
        emit(GpuOp{OpKind::UploadRect, target.id, target.view, target.valid, 0, 0, missing, 0});
        target.channels |= missing;
    }
    const std::vector<DirtyEntry> entries = target.dirty.entries();
    for (const DirtyEntry& entry : entries) {
        emit(GpuOp{OpKind::UploadRect, target.id, target.view, entry.rect, 0, 0, entry.channels, 0});
    }
    target.dirty.clear();
}

void SurfaceCache::shrinkDrawn(Target& target, const Rect& written) {
    if (written.contains(target.drawn)) {
        target.drawn = Rect{};
        return;
    }
    // Chapter 10.3 step 6: a full-width band at the top or bottom edge, or a
    // full-height band at the left or right edge, can be trimmed.
    const bool fullWidth = written.x0 <= target.drawn.x0 && written.x1 >= target.drawn.x1;
    const bool fullHeight = written.y0 <= target.drawn.y0 && written.y1 >= target.drawn.y1;
    if (fullWidth) {
        if (written.y0 <= target.drawn.y0) {
            target.drawn.y0 = written.y1;
        } else if (written.y1 >= target.drawn.y1) {
            target.drawn.y1 = written.y0;
        }
    } else if (fullHeight) {
        if (written.x0 <= target.drawn.x0) {
            target.drawn.x0 = written.x1;
        } else if (written.x1 >= target.drawn.x1) {
            target.drawn.x1 = written.x0;
        }
    }
}

void SurfaceCache::writeBackTarget(std::size_t index, const Rect& region) {
    applyDirty(index);
    Target& target = targets_[index];
    const Rect part = target.drawn.intersect(region);
    if (part.empty()) {
        ++counters_.writeBackSkips;
        return;
    }
    emit(GpuOp{OpKind::DownloadRect, target.id, target.view, part, 0, 0, target.channels, 0});
    ++counters_.writeBackRects;
    shrinkDrawn(target, part);
}

void SurfaceCache::writeBackOverlapping(const BufferView& view, const Rect& rect) {
    const std::vector<std::uint64_t> readBits = blockBitsOf(view, rect);
    for (std::size_t index = 0; index < targets_.size(); ++index) {
        if (targets_[index].drawn.empty()) continue;
        if (!blockBitsAny(readBits, blockBitsOf(targets_[index].view, targets_[index].valid))) continue;
        // Same layout: only the read rectangle matters. Other layouts: the whole
        // drawn area, because the SLM bytes it covers cannot be mapped cheaply.
        const bool exact = sameLayout(targets_[index].view, view);
        writeBackTarget(index, exact ? rect : targets_[index].valid);
    }
}

void SurfaceCache::invalidateSourcesForBlocks(const std::vector<std::uint64_t>& blocks) {
    const std::size_t before = sources_.size();
    sources_.erase(std::remove_if(sources_.begin(), sources_.end(),
                                  [&blocks](const MemorySource& source) {
                                      return source.kind == SourceKind::Memory &&
                                             blockBitsAny(blocks, blockBitsOf(source.view, source.region));
                                  }),
                   sources_.end());
    counters_.sourcesDropped += before - sources_.size();
}

void SurfaceCache::dropTargetViews(std::uint32_t targetId) {
    sources_.erase(std::remove_if(sources_.begin(), sources_.end(),
                                  [targetId](const MemorySource& source) {
                                      return source.kind == SourceKind::TargetView && source.id == targetId;
                                  }),
                   sources_.end());
}

void SurfaceCache::deleteTarget(std::size_t index) {
    const Target& target = targets_[index];
    emit(GpuOp{OpKind::DeleteImage, target.id, target.view, Rect{}, 0, 0, kChanAll, 0});
    dropTargetViews(target.id);
    targets_.erase(targets_.begin() + static_cast<std::ptrdiff_t>(index));
    ++counters_.targetsDeleted;
}

void SurfaceCache::ensureImageSize(std::size_t index, std::int32_t width, std::int32_t height) {
    Target& target = targets_[index];
    if (static_cast<std::uint32_t>(width) <= target.imageWidth &&
        static_cast<std::uint32_t>(height) <= target.imageHeight) {
        return;
    }
    const std::uint32_t newWidth = std::max(static_cast<std::uint32_t>(width), target.imageWidth);
    const std::uint32_t newHeight = std::max(static_cast<std::uint32_t>(height), target.imageHeight);
    emit(GpuOp{OpKind::ResizeImage, target.id, target.view, Rect{}, newWidth, newHeight, kChanAll, 0});
    target.imageWidth = newWidth;
    target.imageHeight = newHeight;
    ++counters_.targetsResized;
}

// Chapter 05.3 step 5 "conflict": two targets never share memory. A target
// whose block range overlaps the new valid area is written back and deleted.
// The test is made on exact block sets, not on block ranges: inside a page the
// block numbers are not monotonic, so a [first, last] range can miss blocks.
void SurfaceCache::resolveConflicts(std::uint32_t keepId, const BufferView& view, const Rect& valid) {
    const std::vector<std::uint64_t> newBlocks = blockBitsOf(view, valid);
    std::vector<std::uint32_t> doomed;
    for (const Target& target : targets_) {
        if (target.id == keepId || target.valid.empty()) continue;
        if (blockBitsAny(newBlocks, blockBitsOf(target.view, target.valid))) doomed.push_back(target.id);
    }
    for (std::uint32_t id : doomed) {
        const std::size_t index = indexOfId(id);
        writeBackTarget(index, targets_[index].valid);
        deleteTarget(index);
        ++counters_.conflictDeletes;
    }
}

// ---------------------------------------------------------------------------
// Invalidation by memory writes (chapters 04.8, 05.6)
// ---------------------------------------------------------------------------

void SurfaceCache::invalidateTargetsForWrite(const std::vector<std::uint64_t>& written) {
    for (std::size_t index = 0; index < targets_.size(); ++index) {
        if (targets_[index].valid.empty()) continue;
        const BufferView view = targets_[index].view;
        const Rect valid = targets_[index].valid;

        // Pixels of the target whose block was written: the footprint.
        Rect footprint{};
        for (std::int32_t y = valid.y0; y < valid.y1; ++y) {
            for (std::int32_t x = valid.x0; x < valid.x1; ++x) {
                const std::uint32_t block =
                    memory::pixelSlot(view, static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y)).byteOffset /
                    kBlockBytes;
                if (blockBitsTest(written, block)) footprint = boundingBox(footprint, Rect{x, y, x + 1, y + 1});
            }
        }
        if (footprint.empty()) continue;

        // Drawn data inside the footprint is written back first, so the SLM is
        // equal to the target there and an over-approximated dirty rectangle is
        // harmless.
        writeBackTarget(index, footprint);
        targets_[index].dirty.add(DirtyEntry{footprint, view.psm, view.bufferWidth, kChanAll}, valid);
        const std::uint32_t id = targets_[index].id;
        sources_.erase(std::remove_if(sources_.begin(), sources_.end(),
                                      [id, footprint](const MemorySource& source) {
                                          return source.kind == SourceKind::TargetView && source.id == id &&
                                                 source.region.overlaps(footprint);
                                      }),
                       sources_.end());
    }
}

// ---------------------------------------------------------------------------
// EE-side operations
// ---------------------------------------------------------------------------

void SurfaceCache::hostUpload(const BufferView& destination, const memory::TransferRect& rect,
                              std::span<const std::uint8_t> data) {
    const Rect area = rectOfTransfer(rect);
    if (area.empty()) return;
    const std::vector<std::uint64_t> written = blockBitsOf(destination, area);
    invalidateTargetsForWrite(written);
    invalidateSourcesForBlocks(written);
    memory::hostToLocal(slm_, destination, rect, data);
    log_.record(LogKind::Upload, destination, area, batch_);
    ++counters_.uploads;
}

void SurfaceCache::localMove(const BufferView& source, std::uint32_t sourceX, std::uint32_t sourceY,
                             const BufferView& destination, std::uint32_t destinationX, std::uint32_t destinationY,
                             std::uint32_t width, std::uint32_t height, memory::TransferDirection direction) {
    if (width == 0 || height == 0) return;
    const Rect sourceArea = rectFromOrigin(static_cast<std::int32_t>(sourceX), static_cast<std::int32_t>(sourceY),
                                           static_cast<std::int32_t>(width), static_cast<std::int32_t>(height));
    const Rect destinationArea = rectFromOrigin(static_cast<std::int32_t>(destinationX),
                                                static_cast<std::int32_t>(destinationY),
                                                static_cast<std::int32_t>(width), static_cast<std::int32_t>(height));
    // Chapter 05.7 fallback: the source must be current in the SLM first.
    writeBackOverlapping(source, sourceArea);
    const std::vector<std::uint64_t> written = blockBitsOf(destination, destinationArea);
    invalidateTargetsForWrite(written);
    invalidateSourcesForBlocks(written);
    memory::localToLocal(slm_, source, sourceX, sourceY, destination, destinationX, destinationY, width, height,
                         direction);
    log_.record(LogKind::Move, destination, destinationArea, batch_);
    ++counters_.moves;
}

bool SurfaceCache::draw(const DrawCommand& command) {
    const BufferView& view = command.frame;
    const std::uint8_t writes = drawChannels(view.psm, command.writeChannels);
    if (!isSupportedTargetPsm(view.psm) || writes == 0) {
        if (!command.rect.empty()) ++counters_.drawsRefused;
        return false;
    }
    const Rect rect = clipToFrameWidth(view, command.rect);
    if (rect.empty()) return false;

    const std::size_t existing = findExactIndex(view);
    const Rect newValid = existing == kNoIndex ? rect : boundingBox(targets_[existing].valid, rect);
    resolveConflicts(existing == kNoIndex ? 0u : targets_[existing].id, view, newValid);

    std::size_t index = findExactIndex(view);
    if (index == kNoIndex) {
        Target target;
        target.id = nextId_++;
        target.view = view;
        target.valid = rect;
        target.channels = writes;
        target.imageWidth = static_cast<std::uint32_t>(rect.x1);
        target.imageHeight = static_cast<std::uint32_t>(rect.y1);
        emit(GpuOp{OpKind::CreateImage, target.id, view, Rect{}, target.imageWidth, target.imageHeight, kChanAll, 0});
        targets_.push_back(target);
        ++counters_.targetsCreated;
        index = targets_.size() - 1;
    } else {
        applyDirty(index);  // dirty data is loaded before the draw (chapter 05.1 update)
        ensureImageSize(index, newValid.x1, newValid.y1);
        Target& target = targets_[index];
        // Area that the new valid rectangle adds and the draw does not cover
        // holds SLM data: load it (chapter 05.3 growth).
        const std::vector<Rect> pieces = subtractFromAll(subtractRect(newValid, target.valid), rect);
        for (const Rect& piece : pieces) {
            emit(GpuOp{OpKind::UploadRect, target.id, view, piece, 0, 0, kChanAll, 0});
        }
        // Channels the draw writes but the target does not hold are reloaded
        // from the SLM over the whole valid area (chapter 04.5 alpha reload).
        const std::uint8_t missing = static_cast<std::uint8_t>(writes & ~target.channels);
        if (missing != 0) {
            emit(GpuOp{OpKind::UploadRect, target.id, view, newValid, 0, 0, missing, 0});
            target.channels |= missing;
        }
        target.valid = newValid;
    }

    Target& target = targets_[index];
    emit(GpuOp{OpKind::Draw, target.id, view, rect, 0, 0, writes, command.drawId});
    target.drawn = boundingBox(target.drawn, rect);

    // The draw changes memory that sources and logged uploads describe.
    const std::vector<std::uint64_t> drawnBits = blockBitsOf(view, rect);
    invalidateSourcesForBlocks(drawnBits);
    log_.eraseIf([&drawnBits](const LogRecord& record) {
        return blockBitsAny(drawnBits, blockBitsOf(record.view, record.rect));
    });

    target.lastUse = ++useClock_;
    target.age = 0;
    target.vblanksSinceUse = 0;
    target.readsSinceDraw = 0;
    ++counters_.draws;
    return true;
}

std::optional<std::vector<std::uint8_t>> SurfaceCache::readHost(const BufferView& source,
                                                                const memory::TransferRect& rect) {
    ++counters_.hostReads;
    if (memory::psmInfo(source.psm).transferBpp == 4) return std::nullopt;
    const Rect area = rectOfTransfer(rect);

    if (log_.findExactUpload(source, area) != nullptr) {
        // Chapter 10.2 shortcut: nothing was drawn over an exact upload.
        ++counters_.readbackShortcuts;
    } else {
        // Chapter 10.3 selection: exact layout first, then any overlap; targets
        // read more than once since their last draw write back their whole
        // drawn area.
        const std::vector<std::uint64_t> readBits = blockBitsOf(source, area);
        std::vector<std::size_t> order(targets_.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [this](std::size_t first, std::size_t second) {
            return targets_[first].lastUse < targets_[second].lastUse;
        });
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t index : order) {
                Target& target = targets_[index];
                if (target.drawn.empty()) continue;
                const bool exact = sameLayout(target.view, source);
                if ((pass == 0) != exact) continue;
                if (!blockBitsAny(readBits, blockBitsOf(target.view, target.valid))) continue;
                Rect want = exact ? area : target.valid;
                if (target.readsSinceDraw > 1) want = target.drawn;
                writeBackTarget(index, want);
                ++target.readsSinceDraw;
                target.lastUse = ++useClock_;
                if (exact && target.valid.contains(area)) {
                    pass = 2;  // stop after an exact base that holds the whole read
                    break;
                }
            }
        }
    }

    std::vector<std::uint8_t> host;
    if (memory::localToHost(slm_, source, rect, host) != memory::TransferStatus::Ok) return std::nullopt;
    return host;
}

std::uint8_t SurfaceCache::readByte(std::uint32_t address) {
    const std::uint32_t block = (address & kLocalMemoryMask) / kBlockBytes;
    for (std::size_t index = 0; index < targets_.size(); ++index) {
        if (targets_[index].drawn.empty()) continue;
        if (blockBitsTest(blockBitsOf(targets_[index].view, targets_[index].valid), block)) {
            writeBackTarget(index, targets_[index].valid);
        }
    }
    return slm_.byte(address);
}

void SurfaceCache::writeBackAll() {
    for (std::size_t index = 0; index < targets_.size(); ++index) {
        writeBackTarget(index, targets_[index].valid);
    }
}

SourceRef SurfaceCache::sampleSource(const BufferView& view, const Rect& region) {
    ++counters_.sourceLookups;
    if (region.empty()) return SourceRef{};

    // Target view (chapter 04.5): the target holds the whole read, all channels,
    // and nothing in the read is dirty.
    const std::size_t index = findExactIndex(view);
    if (index != kNoIndex) {
        Target& target = targets_[index];
        if (target.valid.contains(region) && target.channels == supportedChannels(view.psm) &&
            !target.dirty.overlaps(region)) {
            target.lastUse = ++useClock_;
            target.age = 0;
            target.vblanksSinceUse = 0;
            const bool known = std::any_of(sources_.begin(), sources_.end(), [&](const MemorySource& source) {
                return source.kind == SourceKind::TargetView && source.id == target.id && source.region == region;
            });
            if (!known) {
                MemorySource source;
                source.id = target.id;
                source.kind = SourceKind::TargetView;
                source.view = view;
                source.region = region;
                source.lastUse = useClock_;
                sources_.insert(sources_.begin(), source);
            }
            ++counters_.targetViews;
            return SourceRef{true, SourceKind::TargetView, target.id, region};
        }
    }

    // Memory source (chapter 04.4 step 1 without a fast path): the SLM must be
    // the truth for the read, so overlapping drawn data is written back first.
    writeBackOverlapping(view, region);

    for (MemorySource& source : sources_) {
        if (source.kind == SourceKind::Memory && sameLayout(source.view, view) && source.region == region) {
            source.age = 0;
            source.lastUse = ++useClock_;
            ++counters_.memorySourceHits;
            return SourceRef{true, SourceKind::Memory, source.id, region};
        }
    }

    MemorySource source;
    source.id = nextId_++;
    source.kind = SourceKind::Memory;
    source.view = view;
    source.region = region;
    source.preloadable = isPreloadableRegion(region);
    source.lastUse = ++useClock_;
    emit(GpuOp{OpKind::DecodeMemorySource, source.id, view, region, 0, 0, kChanAll, 0});
    sources_.insert(sources_.begin(), source);
    ++counters_.memorySourceCreates;
    return SourceRef{true, SourceKind::Memory, source.id, region};
}

std::uint32_t SurfaceCache::texel(const SourceRef& source, std::uint32_t x, std::uint32_t y) const {
    if (!source.valid) return 0;
    if (source.kind == SourceKind::TargetView) {
        return backend_.texel(source.imageId, static_cast<std::uint32_t>(source.region.x0) + x,
                              static_cast<std::uint32_t>(source.region.y0) + y);
    }
    return backend_.texel(source.imageId, x, y);
}

DisplayResult SurfaceCache::displayLookup(const BufferView& view, const Rect& rect) {
    const std::size_t index = findExactIndex(view);
    if (index != kNoIndex && targets_[index].valid.contains(rect)) {
        Target& target = targets_[index];
        applyDirty(index);
        target.displayed = true;
        target.lastUse = ++useClock_;
        target.age = 0;
        target.vblanksSinceUse = 0;
        ++counters_.displayTargetHits;
        return DisplayResult{true, target.id};
    }
    writeBackOverlapping(view, rect);
    ++counters_.displaySlm;
    return DisplayResult{false, 0};
}

// ---------------------------------------------------------------------------
// Ageing and budget (chapters 04.10, 05.9)
// ---------------------------------------------------------------------------

std::uint64_t SurfaceCache::usageBytes() const {
    std::uint64_t bytes = 0;
    for (const Target& target : targets_) {
        bytes += static_cast<std::uint64_t>(target.imageWidth) * target.imageHeight * 4u;
    }
    for (const MemorySource& source : sources_) {
        if (source.kind == SourceKind::Memory) bytes += static_cast<std::uint64_t>(source.region.area()) * 4u;
    }
    return bytes;
}

void SurfaceCache::enforceBudget() {
    if (budgetBytes_ == 0 || usageBytes() <= budgetBytes_) return;

    // Step 2: sources go first.
    counters_.sourcesDropped += sources_.size();
    sources_.clear();

    // Step 3: write back and delete the least recently used targets that are
    // not displayed and were not used in the last two VBlanks.
    while (usageBytes() > budgetBytes_) {
        std::size_t victim = kNoIndex;
        for (std::size_t index = 0; index < targets_.size(); ++index) {
            const Target& target = targets_[index];
            if (target.displayed || target.vblanksSinceUse < 2) continue;
            if (victim == kNoIndex || target.lastUse < targets_[victim].lastUse) victim = index;
        }
        if (victim == kNoIndex) break;
        writeBackTarget(victim, targets_[victim].valid);
        deleteTarget(victim);
        ++counters_.budgetEvictions;
    }
}

void SurfaceCache::endVBlank(bool idle) {
    if (idle) return;  // idle VBlanks age nothing (chapter 04.10)
    log_.vblank(batch_);

    // Sources.
    sources_.erase(std::remove_if(sources_.begin(), sources_.end(),
                                  [this](MemorySource& source) {
                                      ++source.age;
                                      const bool drop = (source.kind == SourceKind::Memory && source.preloadable)
                                                            ? source.age > 30
                                                            : source.age > 3;
                                      if (drop) ++counters_.sourcesDropped;
                                      return drop;
                                  }),
                   sources_.end());

    // Targets, newest index first so deletions keep the remaining indices.
    for (std::size_t index = targets_.size(); index-- > 0;) {
        Target& target = targets_[index];
        ++target.age;
        ++target.vblanksSinceUse;
        if (target.age <= 60) continue;
        if (target.dirty.fullyDirty(target.valid, supportedChannels(target.view.psm))) {
            // Every drawn pixel is superseded by the SLM: nothing to write back.
            deleteTarget(index);
        } else {
            target.age = 10;  // keeps GPU-only data that has no other copy
        }
    }

    enforceBudget();
    for (Target& target : targets_) target.displayed = false;
}

}  // namespace coregs::cache
