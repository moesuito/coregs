// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec (TASK-027 M4, chapters 00 reference mode, 03, 12.3, 12.9, 15),
// clean room (decision 0069).
//
// HWR reference renderer, write-back mode (M4). The GS front end (M3) parses the GIF stream into
// batches; each batch is translated to GPU work (vertices, pixel state, decoded texture) and drawn
// on the GPU, then the touched frame and depth words are written back to the shadow local memory
// (SLM) and the target is dropped. Display output is composed from the SLM (13.1 to 13.3).
// Blending is not implemented in M4 (ABE is treated as opaque and counted, IN-500).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hw_batch.hpp"
#include "hw_clut.hpp"
#include <functional>

#include "hw_front_end.hpp"
#include "hw_gs_display.hpp"
#include "hw_memory.hpp"
#include "hw_gs_writeback_gpu.hpp"
#include "coregs/profile.h"

namespace coregs::core2 {
class Renderer;
struct Core2Counters;
}

namespace coregs::writeback {

// Counters for the replay and the F8 report of the M4 milestone.
struct Counters {
    std::uint64_t batches = 0;
    std::uint64_t primitives = 0;
    std::uint64_t primitivesSkippedTexture = 0;  // invalid texture format or size (01.5, IN-307)
    std::uint64_t batchesSkippedTexture = 0;
    std::uint64_t batchesEmpty = 0;             // no pixels after scissor and coverage
    std::uint64_t batchesBlended = 0;           // ABE = 1: the blend unit ran for those batches
    std::uint64_t texturesDecoded = 0;
    std::uint64_t texelsDecoded = 0;
    std::uint64_t gpuChunks = 0;
    std::uint64_t gpuDraws = 0;
    std::uint64_t writebackPixels = 0;          // changed frame words written back to the SLM (IN-335)
    std::uint64_t writebackDepthWords = 0;      // changed depth words written back to the SLM (IN-335)
    std::uint64_t fifoOverruns = 0;             // FIFO reads past the end of the LOCAL->HOST data (10.2)
    // M7 resident targets (IN-340).
    std::uint64_t cacheResidentBatches = 0;   // batches drawn into resident targets (no per-batch transfer)
    std::uint64_t cacheFallbackBatches = 0;   // batches that took the write-back path in cache mode
    std::uint64_t cacheLoads = 0;             // target loads from the shadow memory
    std::uint64_t cacheLoadWords = 0;
    std::uint64_t cacheWritebacks = 0;        // drawn rectangles stored back into the shadow memory
    std::uint64_t cacheWritebackWords = 0;
    std::uint64_t cacheEvictions = 0;
    // Write-backs by reason: other, texture read, CLUT load, transfer, display, eviction, load overlap.
    std::uint64_t cacheWritebackByReason[9] = {};         // resident targets evicted for another (least recently used)
    // CPU microseconds spent in the M7 coherence work and the texture decode (report only, per session).
    double cacheWritebackUs = 0.0;
    double cacheLoadUs = 0.0;
    double textureDecodeUs = 0.0;
    double gifUs = 0.0;     // all of GifTransfer (front end, translate, execute, transfers): the GS stream cost
    double uploadUs = 0.0;  // EE->GS uploads into the shadow memory (and GLM merges, IN-350)
    double batchUs = 0.0;
    double composeUs = 0.0;  // all ComposeFrame calls (display read, merge, deinterlace, copy)
    double clutUs = 0.0;     // GPU CLUT loads (IN-349): GLM preparation and the recorded load
    std::uint64_t vblanksInterlaced = 0;   // VBlanks with SMODE2.INT = 1 (480i, the menus)
    std::uint64_t vblanksProgressive = 0;  // VBlanks with SMODE2.INT = 0 (480p races with COREGS_FORCE_480P=1)   // all of OnBatch (translate, execute and the batch bookkeeping around them)
    std::uint64_t texturesCached = 0;
    std::uint64_t texturesFromTargets = 0;    // batches whose texels were copied from a resident target (IN-346)
    std::uint64_t texturesFromGlm = 0;        // batches whose texels were decoded from the GLM (IN-347)
    std::uint64_t glmPagesUploaded = 0;       // contiguous page runs copied into the GLM
    std::uint64_t glmSwizzles = 0;            // target rectangles swizzled into the GLM
    std::uint64_t glmSwizzlesSkipped = 0;     // swizzles skipped: the GLM already held the pixels (IN-350)
    std::uint64_t cpuSpriteBatches = 0;       // small untextured sprite batches drawn on the CPU (IN-348)
    std::uint64_t glmParks = 0;               // slot drawn rectangles parked in the GLM instead of written back (IN-350)
    std::uint64_t glmSlotLoads = 0;           // slot rectangles loaded from the GLM (IN-350)
    std::uint64_t glmDownloads = 0;           // GLM-owned page downloads for CPU users (IN-350; each waits)
    std::uint64_t glmPagesDownloaded = 0;
    std::uint64_t glmUploadMerges = 0;        // EE uploads merged into GLM-owned pages on the GPU (IN-350)
    std::uint64_t glmHostUploads = 0;         // EE uploads written into the GLM by the upload kernel (IN-351)
    std::uint64_t glmBatchedWritebacks = 0;   // write-backs made as one GLM download (IN-357)
    std::uint64_t displayCaptures = 0;        // display downloads recorded at Flush (IN-355)
    std::uint64_t presentationsFromCapture = 0;  // presentation composes made from a capture (IN-355)
    std::uint64_t gpuComposedFrames = 0;      // presentation frames composed on the GPU (IN-364)
    // IN-384 follow-up (owner test 11: "sporadic 1x frames"): why a VBlank fell back to the CPU 1x presentation.
    // 0 GpuComposeApplies declined, 1 zero output size, 2 interlaced frame larger than the deinterlace images,
    // 3 the GPU compose or its flush failed.
    std::uint64_t gpuComposeSkips[4] = {};
    // M12 (IN-372): GPU composes at scale, and the circuits of them read from the GLM at 1x (no scaled image).
    std::uint64_t scaledComposedFrames = 0;
    std::uint64_t scaledCircuitsFromSlots = 0;
    std::uint64_t scaledCircuitsFromGlm = 0;
    // IN-377: circuits read from a display shadow (the scaled copy of a finished buffer whose slot was dropped; each
    // pixel is still checked against the GLM on the GPU), and the shadows saved.
    std::uint64_t scaledCircuitsFromShadows = 0;
    std::uint64_t scaledShadowSaves = 0;
    // IN-379: tone-curve sequences (09.4 variant A: 8-bit reads of a 32-bit target) seen at scale, and the scaled
    // images rebuilt after them.
    std::uint64_t toneRuns = 0;
    std::uint64_t toneCarries = 0;
    // IN-383 (parity plan item 2): the tone sequence's strip batches. `toneStripBatchesDeferred` counts the batches
    // collected, `toneStripBatches` the ones a page dispatch replaced, and the verification pair is the
    // COREGS_HW_VERIFY_SHUFFLE comparison of the dispatch against the batches themselves.
    std::uint64_t toneStripPages = 0;
    std::uint64_t toneStripBatches = 0;
    std::uint64_t toneStripBatchesDeferred = 0;
    std::uint64_t toneVerifyPages = 0;
    std::uint64_t toneVerifyMismatchBytes = 0;
    std::uint64_t toneClutLoadsHeld = 0;      // IN-383: strip palette loads held (not recorded) instead of performed
    std::uint64_t toneClutLoadsDropped = 0;   // IN-383: held loads a later strip's load superseded
    std::uint64_t glmMergedWords = 0;
    std::uint64_t gpuClutLoads = 0;           // CLUT loads repeated on the GPU from the GLM (IN-349)         // batches whose texels came from the decoded-texture cache (IN-341)
    // Output stage (13.7 counters and V-01-12).
    std::uint64_t presentationFrames = 0;     // presentation composes with an image
    std::uint64_t blankFrames = 0;            // VBlanks with no image (unknown mode or kept previous)
    std::uint64_t deinterlacedFrames = 0;     // presentation composes that ran the 13.4 deinterlacer
    std::uint64_t interlaceOffsetFrames = 0;  // presentation composes shifted by the 13.3 step 3 offset
    std::uint64_t mergeAlphaFromTexel = 0;    // V-01-12: RC1 blended with MMOD = 0 over RC2 (both circuits)
    std::uint64_t flatDeindexedPrimitives = 0;  // IIP = 0 primitives (de-indexed, 03.2)
    std::uint64_t spritesExpanded = 0;
    std::uint64_t auxiliaryFrames = 0;        // render-only passes that restored guest memory
    std::uint64_t auxiliaryReadRefused = 0;   // LOCAL->HOST reads refused during an auxiliary frame
};

// CPU cost of the passes of the last frame, in microseconds (roadmap requirement 4: per-pass budget).
// `executeGpuUs` is reserved for GPU timestamps and stays 0 until the query pool lands, so a reader
// never mistakes it for a measurement.
struct PassTimings {
    double translateUs = 0.0;
    double executeUs = 0.0;
    double composeUs = 0.0;
    double executeGpuUs = 0.0;
};

// Which image a compose produces (12.9, 13.3, 13.4).
//  - Reference: the merge of 13.3 without the interlace offset and without deinterlacing. It is the image
//    the CPU reference composes and the one the 1x decode matches (the `vram` dump source and the
//    chapter 14.4 reference gates use it).
//  - Presentation: what the window shows: the 13.3 step 3 field offset and the 13.4 deinterlacer driven by
//    the VBlank field (`scanout` dump source). The deinterlacer is stateful, so a presentation compose is
//    made at most once per VBlank.
enum class ComposeKind : std::uint8_t { Reference, Presentation };

// Output-stage settings from the options (COREGS_HW_DEINTERLACE, COREGS_HW_ANTIBLUR).
struct OutputSettings {
    display::DeinterlaceMode deinterlace = display::DeinterlaceMode::Auto;
    bool bottomFieldFirst = false;
    bool antiBlur = true;
};

// Output of the display path for one VBlank (13.1 to 13.3, no deinterlace, no post-fx yet).
// IN-364: a presentation frame composed on the GPU into the drawer's display image (R32_UINT, RGBA8 packed),
// valid in the top-left width x height pixels.
struct GpuFrame {
    bool valid = false;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t image = 0;  // the VkImage value (opaque here)
    bool fxaa = false;        // IN-376: `image` is the FXAA output (13.5) instead of the composed display image
    bool presentScaled = false;  // IN-432: `image` is resampled to the window rectangle (width x height)
};

struct FrameOutput {
    bool valid = false;            // false when SMODE1 describes no known mode
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint32_t> pixels;  // RGBA8 packed with red in the low byte
};

// Everything the GPU needs for one batch, plus the skip decisions (CPU translation, no GPU calls).
struct BatchTranslation {
    bool skippedTexture = false;
    bool blended = false;  // PRIM.ABE: the blend unit runs
    memory::BufferView frameView{0, 0, memory::Psm::Ct32};
    memory::BufferView zView{0, 0, memory::Psm::Z32};
    PixelRect rect;  // union of the draw scissors: the rectangle read and written back
    PixelRect scissor;  // the context's scissor clipped to the surface (the M7 load rectangle, IN-344)
    int32_t textureSlot = -1;  // IN-346: GPU colour slot the texels are copied from (-1: decoded texels)
    // IN-347: texels decoded on the GPU from the GLM (view below; indexed formats through `palette`).
    bool glmTexture = false;
    uint32_t glmBp = 0, glmBw = 0, glmPsm = 0;
    bool glmIndexed = false;
    std::vector<uint32_t> palette;
    // TASK-168: a batch with several palette segments (Batch::paletteSegments): `palette` is segment 0's,
    // `segmentPalettes` holds every segment's 256 entries in order (segment s from 256 * s) and drawSegments[d] is the
    // segment of draws[d]. Both stay empty for a batch with one palette.
    std::vector<uint32_t> segmentPalettes;
    std::vector<uint32_t> drawSegments;
    // The batch has several palettes but its texture path takes one (the CPU decode or the GPU CLUT expansion of
    // IN-349): it is not translated (skippedTexture), and the caller must draw its segments one by one.
    bool paletteSegmentsUnsupported = false;
    bool glmGpuClut = false;  // IN-349: indexed texels expanded from the GPU CLUT (fields below)
    uint32_t glmCpsm = 0, glmCsa = 0, glmTa0 = 0, glmTa1 = 0, glmAem = 0;
    gpu::GpuPixelState state;
    bool hasTexture = false;
    uint32_t texWidth = 0;
    uint32_t texHeight = 0;
    // IN-384: the texels a texture taken from a resident slot can be sampled at (IN-380 footprint, 0 = whole extent);
    // the scaled copy only needs these.
    uint32_t texCopyWidth = 0;
    uint32_t texCopyHeight = 0;
    std::vector<uint32_t> texels;
    std::vector<gpu::GpuVertex> vertices;
    std::vector<gpu::GpuDraw> draws;
};

// Translates one batch against the SLM and the CLUT. Counts into  as the draw path does.
void TranslateBatch(const Batch &batch, const memory::LocalMemory &local, const clut::ClutUnit &clut,
                    BatchTranslation &out, Counters &counters);

// Called just before the texels of a batch are read from the shadow memory (the M7 resident targets write
// back drawn pixels that overlap the texture first, IN-340).
using TextureReadHook = std::function<void(const memory::BufferView&, uint32_t width, uint32_t height)>;

// IN-346: asked first for a non-indexed texture; returns the GPU colour slot that holds the texture's memory
// (exact view, whole extent loaded), so the texels are copied on the GPU, or -1 to decode from the shadow memory.
// `footprint`: the texels the batch can sample (inclusive, inside width x height), see TextureFootprint.
using TextureFromTargetHook =
    std::function<int32_t(const memory::BufferView&, uint32_t width, uint32_t height, const PixelRect& footprint)>;

// Decoded-texture cache (M7 memory sources, chapter 04; IN-341). The key is everything the decode reads
// besides memory: TBP0, TBW, PSM, the decoded extent, TEXA for 16/24-bit texels and the palette contents
// for indexed ones. `find` returns texels only when no shadow-memory write touched the texture's pages since
// they were decoded; `store` keeps a fresh decode.
struct TextureKey {
    uint32_t tbp = 0, tbw = 0, psm = 0, width = 0, height = 0;
    uint32_t ta0 = 0, ta1 = 0, aem = 0;
    uint64_t paletteHash = 0;
    uint64_t mipHash = 0;  // IN-428 (CORE2): the mip levels decoded next to level 0, 0 = level 0 only
    bool operator==(const TextureKey&) const = default;
};

class TextureCacheView {
public:
    virtual ~TextureCacheView() = default;
    virtual const std::vector<uint32_t>* find(const TextureKey& key, const memory::BufferView& view) = 0;
    virtual void store(const TextureKey& key, const memory::BufferView& view, const std::vector<uint32_t>& texels) = 0;
};

// IN-347: asked next; returns true when the texture overlaps GPU-newer target data and the GLM was made
// current for it (pages uploaded, targets swizzled), so the texels are decoded from the GLM on the GPU.
using TextureFromGlmHook =
    std::function<bool(const memory::BufferView&, uint32_t width, uint32_t height, bool indexed, bool& gpuClut)>;

// FNV-1a of a 256-entry palette as BatchTranslation::palette holds it (texture cache keys).
inline uint64_t HashPaletteWords(const uint32_t *entries, size_t count) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < count; ++i) {
        hash ^= entries[i];
        hash *= 1099511628211ull;
    }
    return hash;
}
inline uint64_t HashPaletteWords(const std::vector<uint32_t> &palette) {
    return HashPaletteWords(palette.data(), palette.size());
}

// TranslateBatch with the hooks the cache renderers use (CORE2, decision coregs2-scale-only-rendering): a texture
// read hook, a decoded-texture cache and a texture-from-target lookup.
void TranslateBatchHooked(const Batch &batch, const memory::LocalMemory &local, const clut::ClutUnit &clut,
                          BatchTranslation &out, Counters &counters, const TextureReadHook &beforeTextureRead,
                          TextureCacheView *textureCache, const TextureFromTargetHook &textureFromTarget,
                          const TextureFromGlmHook &textureFromGlm = {});

// Draws a translated batch in chunks (one Execute per chunk), reading the frame and depth words of
// the rectangle from the SLM first and writing them back afterwards (chapter 00 write-back mode).
bool ExecuteBatch(gpu::GpuDrawer &gpu, const BatchTranslation &translation, memory::LocalMemory &local,
                  Counters &counters, std::string &error);

class Renderer {
public:
    // `prefer_core2` selects the CoreGS 2.0 core; the reference write-back core stays available for
    // the conformance fixtures. COREGS_HW_CORE=1 or 2 overrides the choice (A/B).
    explicit Renderer(bool prefer_core2 = false);
    ~Renderer();
    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;

    // M12 (chapter 06, IN-371): internal resolution scale and the 06.3 vertex shift factor, before Init. The scale
    // changes only the presentation (scaled images and the scaled GPU compose); memory, dumps of the `vram` and
    // `scanout` sources, readbacks and every 1x gate are unchanged.
    // The game profile: rules that recognise this game's content. Forwarded to the CoreGS 2.0 core;
    // set it before Init for the shader precompilation list.
    void SetProfile(const coregs::GameProfile &profile);

    void SetScale(std::uint32_t scale, float scaleOffset);
    std::uint32_t Scale() const;

    // Creates the GPU drawer on an existing device. False with `error` when the GPU path is unusable.
    bool Init(const gpu::VulkanHandles &handles, std::string &error);

    // GIF packets for path 1..3 (16-byte qwords). Path 0 and bad sizes are rejected.
    bool GifTransfer(std::uint32_t path, const std::uint8_t *bytes, std::size_t byteCount);
    bool GifBusy(std::uint32_t path) const;
    void GifReset(std::uint32_t pathMask);

    // Privileged writes (0x12000000..0x12001FFF). Returns false outside the display set.
    bool WritePrivileged(std::uint32_t address, std::uint64_t value);

    // GS reset as the CPU reference does it: front end, registers and CLUT, every GIF path, the pending
    // LOCAL->HOST read and the display state (registers, previous DBY, background, kept image).
    void ResetGs();
    void ClearVram();              // zeroes the SLM (drops nothing: write-back mode has no caches)
    // LOCAL->HOST data of the pending read (10.2): the batch is flushed first so the read observes every
    // earlier draw, the data is consumed sequentially across calls, and TRXDIR becomes 3 once it is
    // drained. Returns the bytes of real data copied (at most byteCount); the caller zero-fills the rest.
    std::size_t ReadFifo(void *destination, std::size_t byteCount);

    // VBlank: commits pending transfers and flushes the batch (chapter 02.8).
    void Flush();
    // Composes the display image from the SLM and the display registers. ComposeFrame() writes into a
    // caller-owned buffer and keeps its capacity, so a steady frame rate does not reallocate 1.1 MiB
    // every VBlank; Compose() returns a fresh copy for tests and one-shot callers.
    // The display layout (13.2) is computed once per VBlank (Flush), so several composes of the same VBlank
    // (a dump and a present) see the same DBY history.
    void ComposeFrame(FrameOutput &out, ComposeKind kind = ComposeKind::Reference);
    // IN-364: the presentation compose on the GPU for a progressive VBlank (no field offset, no deinterlace, a
    // composite plan): the GLM is made current for the circuits and the compose kernel records into the display
    // image, then the work is submitted. Returns false (nothing done) when the frame needs the CPU path.
    bool ComposeFrameGpu(GpuFrame &out);
    void SetGpuCompose(bool enabled);
    // 13.5 FXAA (COREGS_HW_POSTFX=fxaa, IN-376): GPU-composed presentation frames pass through the edge-only FXAA.
    void SetFxaa(bool enabled);
    // IN-432 (core 2): the window rectangle the next presented frame is shown in (0 x 0: unknown, no resampling).
    void SetPresentSize(std::uint32_t width, std::uint32_t height);
    // IN-433 (core 2): COREGS_HW_FILTER forced bilinear: 0 as the game asks, 1 every textured draw, 2 all but sprites.
    void SetForcedBilinear(std::uint32_t mode);
    // Fixtures and checks: the pixels of a GPU frame (submits and waits).
    bool ReadGpuFrame(const GpuFrame &frame, std::vector<std::uint32_t> &pixels);
    FrameOutput Compose(ComposeKind kind = ComposeKind::Reference);

    // Display field of the VBlank being closed (CSR.FIELD from the VSYNC record, 12.4, 13.3, 13.4).
    void SetDisplayField(bool odd);
    void SetAgeTargets(bool age);  // TASK-082: core 2 only
    bool DisplayField() const;
    void SetOutputSettings(const OutputSettings &settings);
    const OutputSettings &GetOutputSettings() const;

    // Render-only pass with no guest-visible side effect (roadmap requirement 2, chapter 00 write-back
    // mode): the whole SLM is snapshotted on begin and restored on end, so transfers, draws and clears
    // of an auxiliary frame can never reach the guest's memory, and LOCAL->HOST reads are refused while
    // it runs. Signals and FINISH are never answered from it. Returns false when one is already active.
    bool BeginAuxiliaryFrame();
    void EndAuxiliaryFrame();
    bool AuxiliaryActive() const;

    // CPU timings of the last completed frame (translate, execute, compose).
    const PassTimings &Timings() const;

    // Session sums of the same passes plus the number of completed frames. A single last-frame number is
    // too noisy to compare against the roadmap's per-pass budget, so callers report the average.
    const PassTimings &TimingTotals() const;
    std::uint64_t TimedFrames() const;

    // Diagnostic hook (replay tools only): called for every batch after translation and before the GPU
    // runs it, with the front-end batch and what the write-back path made of it. Empty = no call.
    using BatchObserver = std::function<void(const Batch &, const BatchTranslation &)>;
    void SetBatchObserver(BatchObserver observer);

    // COREGS_HW_SYNC_MODE (12.8): true = `cache` (M7 resident targets, IN-340), false = `writeback` (the
    // reference mode: every batch uploads and reads back its rectangle). The default is the reference mode;
    // the backend and the replay tool set the option's value.
    void SetSyncMode(bool cache);
    // IN-346 A/B switch: textures that are resident colour targets are copied on the GPU (default on).
    void SetTextureFromTargets(bool enabled);
    // IN-347 A/B switch (COREGS_HW_GLM): textures over GPU-newer target data are decoded from the GLM (default on);
    // off, the targets are written back and the texture is decoded on the CPU.
    void SetGlm(bool enabled);
    // IN-348 A/B switch: small untextured sprite batches are drawn on the CPU in cache mode (default on).
    void SetCpuSmallSprites(bool enabled);
    // IN-349 A/B switch: CLUT loads are repeated on the GPU and indexed textures expanded from the GPU CLUT.
    void SetGpuClut(bool enabled);
    // IN-350 A/B switch: evicted or overlapped resident targets are parked in the GLM (GPU only) instead of
    // written back; the pages they leave are GLM-owned until a CPU user downloads them.
    void SetGlmParking(bool enabled);
    // IN-350 fixtures: every slot load reads the GLM (exercises the GLM load path on every load).
    void SetGlmLoadsAlways(bool enabled);
    // IN-351 A/B switch: EE uploads are written into the GLM on the GPU (pages become GLM-owned).
    void SetGpuUploads(bool enabled);
    // IN-352 A/B switch: interlocked chunks batched into runs (one render pass per run).
    void SetRuns(bool enabled);
    // IN-357 A/B switch: CPU readers get the drawn targets and GLM-owned pages over their span in one download.
    void SetBatchedWritebacks(bool enabled);
    // COREGS_HW_GPU_TIMING: GPU timestamps per category (GpuCounters().gpuNs).
    void SetGpuTiming(bool enabled);
    // IN-369: a run reuses the atlas region of an identical GLM texture decode (default on).
    void SetDecodeReuse(bool enabled);
    // IN-370: GPU CLUT loads are deferred; an indexed texture whose palette the last deferred load wrote reads it
    // from the GLM (default on).
    void SetClutDefer(bool enabled);
    // IN-383 A/B switch (COREGS_HW_TONE_HLE): the game's tone-sequence strip batches run as one dispatch per page (default
    // on); off, every strip takes the normal path.
    void SetToneHle(bool enabled);
    // IN-383: COREGS_HW_VERIFY_SHUFFLE, execute both and compare the written page (09.4, 12.8 `COREGS_HW_VERIFY_SHUFFLE`).
    void SetVerifyShuffle(bool enabled);
    // IN-358 A/B: mid-frame submissions every `chunks` chunks (0 = off).
    void SetChunksPerSubmission(std::uint32_t chunks);
    // IN-354 A/B switch: double-buffered GPU submission.
    void SetDoubleBuffering(bool enabled);
    // IN-355: presentation composes show the previous VBlank's display, downloaded asynchronously (no CPU wait
    // for the frame just submitted). Needs the cache mode and the GLM; Reference composes are unaffected.
    void SetAsyncPresentation(bool enabled);
    bool CacheMode() const;

    // CPU view of the SLM (4 MiB, GS layout). Resident draws are written back first.
    const std::uint8_t *Slm() const;

    const Counters &GetCounters() const;
    const gpu::GpuStats &GpuCounters() const;
    const std::string &LastError() const;
    bool Failed() const;

    // CORE2 (the core rewrite, CORE2-DESIGN.md): true when COREGS_HW_CORE=2 selected it at construction.
    bool Core2() const { return core2_ != nullptr; }
    // IN-507: false without CORE2 or without a shader cache.
    bool PrepareCore2Shaders(const std::function<void(std::size_t, std::size_t)> &progress, std::string &error);
    // CORE2's own session counters (nullptr with the old core).
    const core2::Core2Counters *Core2Stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    // COREGS_HW_CORE=2: every call below goes to the new core; the old one stays idle.
    std::unique_ptr<core2::Renderer> core2_;
};

} // namespace coregs::writeback
