// HWR GS front end: GIF stream to register state, vertex kicks, batches with
// snapshots and the flush rules, auto-flush, transfers, VBlank and CLUT loads
// (chapter 02, with chapter 07.3 for loads).
// Copyright 2026 GT4Recomp contributors. Licensed under the MIT License (see LICENSE).
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#pragma once

#include <functional>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "hw_batch.hpp"
#include "hw_clut.hpp"
#include "hw_gif_parser.hpp"
#include "hw_gs_counters.hpp"
#include "hw_gs_registers.hpp"
#include "hw_gs_types.hpp"
#include "hw_transfers.hpp"

namespace coregs {

// COREGS_HW_AUTOFLUSH (chapter 02.6).
enum class AutoflushMode : uint8_t { Sprites, All, Off };

struct FrontEndOptions {
    AutoflushMode autoflush = AutoflushMode::Sprites;
    bool clutAlwaysLoad = false;  // COREGS_HW_CLUT_ALWAYS_LOAD=1
    // Chapter 02.9 vertex trace per batch (Batch::trace). No consumer reads it yet (the decision layer is not
    // wired); the write-back renderer turns it off (IN-353).
    bool vertexTrace = true;
    // IN-366 A/B switch: strip runs decoded at once, no candidate copy for non-drawing kicks, the flush decision
    // reused while nothing it reads changed. Off: the generic per-register path (the fixture's oracle).
    bool fastPaths = true;
    // IN-403: a triangle or sprite whose bounding box holds no pixel centre draws nothing at 1x and is dropped. At an
    // internal scale above 1 it covers samples between the centres (GT4's sub-pixel strip pieces, thin far-away
    // geometry), so a renderer at scale keeps it (the CPU reference culls it only at 1x).
    bool cullUncovered = true;
};

enum class SyncKind : uint8_t { Signal, Finish, Label };

// Receives batches in stream order (chapter 12.3 gif_transfer contract).
class BatchSink {
public:
    virtual ~BatchSink() = default;
    virtual void OnBatch(Batch&& batch) = 0;
    // TASK-168b: asked for every batch the front end emits (after the chapter 02.6 discard), before the work only a
    // drawn batch needs (chapter 02.4 ST rounding, the vertex trace) and before OnBatch. True: the sink has applied the
    // batch's whole effect itself (it draws nothing, e.g. core 2 dropping GT4's tone-run draws), so the batch is neither
    // rounded, traced nor handed over; the front end still does its own bookkeeping (CLUT staleness, flush reason and
    // counts). The default never drops.
    virtual bool DropOnArrival(const Batch& batch) {
        (void)batch;
        return false;
    }
};

// Receives SIGNAL, FINISH and LABEL in stream order. Returning true asks for a flush.
class SyncSink {
public:
    virtual ~SyncSink() = default;
    virtual bool OnSync(SyncKind kind, uint32_t id, uint32_t idMask) = 0;
};

// Block range in GS local memory (wraps at 16384 blocks).
struct BlockRange {
    uint32_t begin = 0;
    uint32_t count = 0;
};

class FrontEnd final : public GifSink {
public:
    FrontEnd(FrontEndOptions options, BatchSink& batches, SyncSink& sync, TransferSink& memory);

    // Local memory image (4 MiB) used by CLUT loads. Defaults to zeros.
    void SetLocalMemory(const uint8_t* localMemory);
    // IN-403: FrontEndOptions::cullUncovered, set by the renderer once it knows its internal scale.
    void SetCullUncovered(bool cull) { options_.cullUncovered = cull; }

    // Called just before a CLUT load reads the local memory image (after the batch flush of 07.3), with a
    // conservative block range covering the palette. A renderer that keeps drawn pixels outside the image
    // (the M7 resident targets) writes them back here. Empty = no call.
    void SetBeforeLocalRead(std::function<void(const BlockRange&)> callback);

    // Called after the CLUT unit performed a load (not for skipped or refused ones), with its TEX0/TEXCLUT
    // fields, so a renderer can repeat the load on its GPU CLUT copy (chapter 07.7, IN-349). Empty = no call.
    // IN-353: CPU CLUT entry copies on or off (off while the GPU CLUT is authoritative, IN-349).
    clut::ClutUnit& MutableClut() { return clut_; }
    void SetClutLoaded(std::function<void(const clut::Tex0Write&)> callback);

    // TASK-168: asked when a CLUT load would end the pending batch only because the batch draws an indexed texture
    // that stays the same (TEX0 base and indexed fields, the same context), while the batch's frame and depth buffers
    // meet neither the palette's memory nor the texture, and no auto-flush can split it (PaletteSegmentAllowed in the
    // source has the list). True keeps the batch: the primitives after the load form a new palette segment
    // (Batch::paletteSegments). The renderer answers true only when it can draw the batch's CPU palettes per primitive
    // and the load reads GS memory alone (no palette read on the GPU).
    // Empty = never (every such load ends the batch, the chapter 07.3 rule as before).
    using PaletteSegmentHook = std::function<bool(const clut::Tex0Write& write, const Batch& pending)>;
    void SetPaletteSegmentHook(PaletteSegmentHook hook) { paletteSegmentHook_ = std::move(hook); }
    // TASK-168b: the renderer's cheap question about the pending batch alone, asked once the load is known to keep the
    // texture and before the front end's memory checks and the hook: false ends the batch at the load (a batch the
    // renderer gains nothing from merging). Empty = every batch may go on to the hook.
    using PaletteSegmentBatchFilter = std::function<bool(const Batch& pending)>;
    void SetPaletteSegmentBatchFilter(PaletteSegmentBatchFilter filter) { paletteSegmentBatchFilter_ = std::move(filter); }
    // CLUT loads that kept the batch as a new segment (TASK-168), and batches emitted with segments.
    uint64_t ClutLoadsKept() const { return clutLoadsKept_; }
    uint64_t SegmentedBatches() const { return segmentedBatches_; }
    uint64_t PaletteSegmentsEmitted() const { return paletteSegmentsEmitted_; }

    // Feeds whole qwords to a GIF path (1..3).
    void GifTransfer(uint32_t path, const Qword* qwords, size_t count);
    bool GifBusy(uint32_t path) const;
    // XGKICK end of a path-1 packet without EOP.
    void GifPathOneEnd();
    // GIF reset: parser state for the selected paths, TRXDIR = 3, Q latch = 1.0.
    void GifReset(uint32_t pathMask);

    // VBlank (chapter 02.8): commits uploads, flushes when the batch's frame
    // block range overlaps one of the enabled display ranges.
    void VBlank(const BlockRange* displays, size_t count);

    // GS reset (chapter 02.11): drops the batch and queue, resets registers,
    // CLUT, transfers and the texture-flushed flag.
    void ResetGs();

    // Flushes the pending batch (end of frame or an explicit request).
    void FlushBatch(FlushReason reason);

    const RegisterFile& Registers() const { return registers_; }
    const clut::ClutUnit& Clut() const { return clut_; }
    const CounterSet& Counters() const { return counters_; }
    uint64_t FastStripVertices() const { return parser_.FastStripVertices(); }  // IN-366
    uint64_t FlushReasonCount(FlushReason reason) const {
        return flushCounts_[static_cast<size_t>(reason)];
    }
    // CLUT-load flushes whose batch had the same texture as the new TEX0 (CPU survey R9).
    uint64_t ClutFlushesSameTexture() const { return clutFlushSameTexture_; }
    // TASK-168b: emitted batches the sink dropped on arrival (BatchSink::DropOnArrival).
    uint64_t BatchesDroppedOnArrival() const { return batchesDroppedOnArrival_; }
    const Batch& PendingBatch() const { return batch_; }
    bool TexflushPending() const { return texflushPending_; }

    // GifSink
    void WriteRegister(uint8_t address, uint64_t data) override;
    void Kick(const GifKick& kick) override;
    void ImageQword(uint32_t path, const Qword& qword) override;
    void ImageQwords(uint32_t path, const Qword* qwords, size_t count) override;
    void StripVertices(const StripVertex* vertices, size_t count) override;

private:
    struct Candidate {
        Vertex vertices[3];
        uint32_t count = 0;
    };

    struct TexelRect {
        PixelRect rect;
        bool valid = false;
    };

    void TakeSnapshot(uint32_t contextIndex);
    Vertex MakeVertex(const GifKick& kick, const ContextRegs& ctx) const;
    bool BuildCandidate(PrimType type, Candidate& candidate) const;
    bool ShouldDraw(const Candidate& candidate, PrimType type, uint32_t contextIndex,
                    PixelRect& rect) const;
    void AppendPrimitive(const Candidate& candidate, PrimType type, const PixelRect& rect);
    void AdvanceQueue(PrimType type);

    std::optional<FlushReason> DecideFlush() const;
    bool AutoflushWanted(const Candidate& candidate, PrimType type) const;
    bool AutoflushStateAllows() const;
    // TASK-168.
    bool PaletteSegmentAllowed(const clut::Tex0Write& write, uint32_t contextIndex) const;
    bool BatchBuffersMeet(uint32_t begin, uint32_t count) const;
    bool TextureMeetsBatchBuffers() const;
    void BeginPaletteSegment();
    void EmitBatch(Batch& batch, FlushReason reason);
    void CountEmittedBatch(Batch& batch, FlushReason reason);  // TASK-168b
    static void RecycleStorage(Batch& batch, Batch& used);
    void FlushSegmentsBeforeLast(FlushReason reason);
    TexelRect ComputeTexelRect(const Candidate& candidate) const;
    bool BatchTouches(uint32_t begin, uint32_t count, bool asDestination) const;
    bool MipLevelsTouch(const ContextRegs& ctx, uint32_t levelZeroBlocks, uint32_t begin, uint32_t count) const;
    bool FrameWritesColour(const ContextRegs& ctx) const;
    bool ZbufWritten(const ContextRegs& ctx) const;
    bool ZbufUsed(const ContextRegs& ctx) const;

    void OnTex0Write(uint32_t contextIndex);
    void NoteBatchWritesToClut(const Batch& batch);
    TransferHooks MakeTransferHooks();

    FrontEndOptions options_;
    BatchSink& batches_;
    SyncSink& sync_;
    TransferSink& memory_;

    CounterSet counters_;
    RegisterFile registers_;
    GifParser parser_;
    clut::ClutUnit clut_;
    TransferUnit transfers_;
    std::vector<uint8_t> imageBytes_;  // packed image qwords (IN-353)

    const uint8_t* localMemory_ = nullptr;
    std::function<void(const BlockRange&)> beforeLocalRead_;
    std::function<void(const clut::Tex0Write&)> clutLoaded_;
    PaletteSegmentHook paletteSegmentHook_;
    PaletteSegmentBatchFilter paletteSegmentBatchFilter_;  // TASK-168b
    std::array<uint16_t, clut::kClutHalfwords> clutBeforeLoad_{};  // TASK-168: the CLUT buffer before a kept load
    std::vector<uint8_t> zeroMemory_;

    Batch batch_;
    // IN-366: the vertex queue holds at most 3 vertices (strips keep 2, fans keep the centre and the last, lists and
    // sprites clear), so a fixed array replaces the vector. IN-455 (CPU survey R7): a ring over that array, so the
    // strip advance moves an index instead of copying the remaining vertices.
    class VertexQueue {
    public:
        size_t size() const { return count_; }
        bool empty() const { return count_ == 0; }
        const Vertex& operator[](size_t index) const { return items_[(head_ + index) & kMask]; }
        void clear() { count_ = 0; }
        void push_back(const Vertex& vertex) {
            if (count_ == items_.size()) {  // never with the GS primitive rules; kept as a vector-like fallback
                head_ = (head_ + 1) & kMask;
                --count_;
                ++overflows_;
            }
            items_[(head_ + count_++) & kMask] = vertex;
        }
        void erase_first() {
            if (count_ == 0) return;
            head_ = (head_ + 1) & kMask;
            --count_;
        }
        void keep_first_and_last() {  // the fan rule: centre and last
            if (count_ < 3) return;
            items_[(head_ + 1) & kMask] = items_[(head_ + count_ - 1) & kMask];
            count_ = 2;
        }
        uint64_t overflows() const { return overflows_; }

    private:
        static constexpr size_t kMask = 7;  // items_ holds 8 vertices
        std::array<Vertex, 8> items_{};
        size_t head_ = 0;
        size_t count_ = 0;
        uint64_t overflows_ = 0;
    };
    VertexQueue queue_;
    bool flushDecisionStale_ = true;  // IN-366: registers or the snapshot changed since the last "no flush" decision

    uint64_t rgbaq_ = 0;
    uint64_t st_ = 0;
    uint64_t uv_ = 0;
    uint8_t fog_ = 0;

    bool texflushPending_ = false;
    bool scissorWritten_ = false;
    bool flushing_ = false;
    bool inTransferHook_ = false;  // a transfer commit is running (no nested transfer commits)

    std::array<uint64_t, static_cast<size_t>(FlushReason::Count)> flushCounts_{};
    uint64_t clutFlushSameTexture_ = 0;
    uint64_t batchesDroppedOnArrival_ = 0;  // TASK-168b
    uint64_t clutLoadsKept_ = 0;            // TASK-168
    uint64_t segmentedBatches_ = 0;
    uint64_t paletteSegmentsEmitted_ = 0;
};

}  // namespace coregs
