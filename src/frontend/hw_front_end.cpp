// Copyright 2026 GT4Recomp contributors. Licensed under the MIT License (see LICENSE).
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#include "hw_front_end.hpp"

#include <bit>

#include <algorithm>
#include <cmath>
#include <limits>

namespace coregs {
namespace {

constexpr uint64_t kTestMask = MaskBits(0, 18);
constexpr uint64_t kScissorMask = MaskBits(0, 10) | MaskBits(16, 26) | MaskBits(32, 42) | MaskBits(48, 58);
constexpr uint64_t kXyoffsetMask = MaskBits(0, 15) | MaskBits(32, 47);
constexpr uint64_t kAlphaMask = MaskBits(0, 39);
constexpr uint64_t kFogcolMask = MaskBits(0, 23);
constexpr uint64_t kFrameMask = MaskBits(0, 8) | MaskBits(16, 21) | MaskBits(24, 29) | MaskBits(32, 63);
constexpr uint64_t kZbufMask = MaskBits(0, 8) | MaskBits(24, 27) | MaskBits(32, 32);
constexpr uint64_t kTex0BaseMask = MaskBits(0, 36);
constexpr uint64_t kTex0IndexedMask = MaskBits(51, 54) | MaskBits(56, 60);
constexpr uint64_t kTex1Mask = MaskBits(0, 0) | MaskBits(2, 9) | MaskBits(19, 20) | MaskBits(32, 43);
constexpr uint64_t kClampMask = MaskBits(0, 43);
constexpr uint64_t kTexaMask = MaskBits(0, 7) | MaskBits(15, 15) | MaskBits(32, 39);
constexpr uint64_t kAllBits = ~uint64_t{0};
constexpr uint32_t kZtstAlways = 1;

constexpr uint32_t kLimitPointLine = 16379;
constexpr uint32_t kLimitSprite = 32765;
constexpr uint32_t kLimitTriangle = 65532;

bool Dirty(uint64_t live, uint64_t snap, uint64_t mask) { return ((live ^ snap) & mask) != 0; }

bool Bit(uint64_t value, unsigned index) { return ((value >> index) & 1u) != 0; }

bool Tme(uint64_t prim) { return Bit(prim, primbit::kTme); }
bool Fst(uint64_t prim) { return Bit(prim, primbit::kFst); }

bool CanWriteColour(uint64_t testReg) {
    const bool cannot = test::Ate(testReg) == 1u && test::Atst(testReg) == kAtstNever &&
                        (test::Afail(testReg) & 1u) == 0u && test::Date(testReg) == 0u;
    return !cannot;
}

bool CanWriteZ(uint64_t testReg) {
    return test::Atst(testReg) != kAtstNever || test::Ate(testReg) == 0u ||
           test::Afail(testReg) == kAfailZbOnly;
}

// Colour channels written by a FRAME register (chapter 02.6 cond. 2): the
// format's channels minus byte lanes that FBMSK protects completely.
uint32_t FrameWrittenChannels(uint64_t frame) {
    uint32_t channels = PsmChannelMask(frame::Psm(frame));
    const uint32_t mask = frame::Fbmsk(frame);
    for (uint32_t lane = 0; lane < 4; ++lane) {
        if (((mask >> (8u * lane)) & 0xFFu) == 0xFFu) {
            channels &= ~(1u << lane);
        }
    }
    return channels;
}

// Texture (or palette) level base for level 0..6 (chapter 01.1, MIPTBP layout).
uint32_t LevelBase(const ContextRegs& ctx, uint32_t level) {
    if (level == 0u) {
        return tex0::Tbp0(ctx.tex0);
    }
    if (level <= 3u) {
        const unsigned shift = 20u * (level - 1u);
        return static_cast<uint32_t>(BitRange(ctx.miptbp1, shift, shift + 13u));
    }
    const unsigned shift = 20u * (level - 4u);
    return static_cast<uint32_t>(BitRange(ctx.miptbp2, shift, shift + 13u));
}

// Wrap or clamp of one texel axis (chapter 01.5 addressing).
void WrapAxis(uint32_t mode, int32_t lo, int32_t hi, int32_t size, uint32_t minv, uint32_t maxv,
              int32_t& outLo, int32_t& outHi) {
    const int32_t regionMin = static_cast<int32_t>(minv);
    const int32_t regionMax = static_cast<int32_t>(maxv);
    switch (mode) {
        case 1u:  // clamp
            outLo = std::clamp(lo, 0, size - 1);
            outHi = std::clamp(hi, 0, size - 1);
            break;
        case 2u:  // region clamp
            outLo = std::clamp(lo, regionMin, regionMax);
            outHi = std::clamp(hi, regionMin, regionMax);
            break;
        case 3u:  // region repeat: (u & MINU) | MAXU
            outLo = (lo & regionMin) | regionMax;
            outHi = (hi & regionMin) | regionMax;
            if (outLo > outHi) {
                std::swap(outLo, outHi);
            }
            break;
        default:  // repeat
            if (hi - lo + 1 >= size) {
                outLo = 0;
                outHi = size - 1;
            } else {
                outLo = lo & (size - 1);
                outHi = hi & (size - 1);
                if (outLo > outHi) {
                    outLo = 0;
                    outHi = size - 1;
                }
            }
            break;
    }
}

void PackLe64(uint64_t value, uint8_t* out) {
    for (int i = 0; i < 8; ++i) {
        out[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

}  // namespace

FrontEnd::FrontEnd(FrontEndOptions options, BatchSink& batches, SyncSink& sync, TransferSink& memory)
    : options_(options),
      batches_(batches),
      sync_(sync),
      memory_(memory),
      parser_(counters_),
      clut_(counters_),
      transfers_(memory_, MakeTransferHooks(), counters_) {
    zeroMemory_.assign(kLocalMemoryBytes, 0);
    parser_.SetFastStrips(options_.fastPaths);  // IN-366
    localMemory_ = zeroMemory_.data();
    clut_.SetAlwaysLoad(options.clutAlwaysLoad);
}

TransferHooks FrontEnd::MakeTransferHooks() {
    TransferHooks hooks;
    hooks.beforeUpload = [this](const UploadCommit& commit) {
        if (batch_.Empty()) {
            return;
        }
        if (BatchTouches(commit.dbp, FootprintBlocks(commit.dbw, commit.dsay + commit.rrh), true)) {
            counters_.Add(Counter::TransferOverlapFlushes);
            inTransferHook_ = true;
            FlushBatch(FlushReason::Transfer);
            inTransferHook_ = false;
        }
    };
    hooks.afterUpload = [this](const UploadCommit& commit) {
        clut_.NoteBlocksWritten(commit.dbp, FootprintBlocks(commit.dbw, commit.dsay + commit.rrh));
    };
    hooks.beforeMove = [this](const LocalMoveCommit& move) {
        if (batch_.Empty()) {
            return;
        }
        const bool touches =
            BatchTouches(move.sbp, FootprintBlocks(move.sbw, move.ssay + move.rrh), false) ||
            BatchTouches(move.dbp, FootprintBlocks(move.dbw, move.dsay + move.rrh), true);
        if (touches) {
            counters_.Add(Counter::TransferOverlapFlushes);
            inTransferHook_ = true;
            FlushBatch(FlushReason::Transfer);
            inTransferHook_ = false;
        }
    };
    hooks.beforeRead = [this](const LocalToHostSetup& setup) {
        if (batch_.Empty()) {
            return;
        }
        if (BatchTouches(setup.sbp, FootprintBlocks(setup.sbw, setup.ssay + setup.rrh), false)) {
            counters_.Add(Counter::TransferOverlapFlushes);
            inTransferHook_ = true;
            FlushBatch(FlushReason::Transfer);
            inTransferHook_ = false;
        }
    };
    return hooks;
}

void FrontEnd::SetBeforeLocalRead(std::function<void(const BlockRange&)> callback) {
    beforeLocalRead_ = std::move(callback);
}

void FrontEnd::SetClutLoaded(std::function<void(const clut::Tex0Write&)> callback) {
    clutLoaded_ = std::move(callback);
}

void FrontEnd::SetLocalMemory(const uint8_t* localMemory) {
    localMemory_ = localMemory != nullptr ? localMemory : zeroMemory_.data();
}

void FrontEnd::GifTransfer(uint32_t path, const Qword* qwords, size_t count) {
    parser_.Transfer(path, qwords, count, *this);
}

bool FrontEnd::GifBusy(uint32_t path) const { return parser_.Busy(path); }

void FrontEnd::GifPathOneEnd() { parser_.EndPathOnePacket(); }

void FrontEnd::GifReset(uint32_t pathMask) {
    parser_.Reset(pathMask);
    transfers_.ForceTrxdirIdle();
}

void FrontEnd::ImageQword(uint32_t path, const Qword& qword) {
    (void)path;
    uint8_t bytes[16];
    PackLe64(qword.lo, bytes);
    PackLe64(qword.hi, bytes + 8);
    transfers_.AcceptData(bytes, sizeof(bytes));
}

// IN-353: a run of image qwords is packed once and handed to the transfer unit in as few calls as possible
// (one per transfer it completes); the bytes, commits and discard counts equal ImageQword per qword.
void FrontEnd::ImageQwords(uint32_t path, const Qword* qwords, size_t count) {
    (void)path;
    // On a little-endian host a Qword {lo, hi} already holds the 16 bytes in GS order (IN-360: no repacking).
    const uint8_t* bytes = nullptr;
    if constexpr (std::endian::native == std::endian::little && sizeof(Qword) == 16) {
        bytes = reinterpret_cast<const uint8_t*>(qwords);
    } else {
        imageBytes_.resize(count * 16);
        for (size_t i = 0; i < count; ++i) {
            PackLe64(qwords[i].lo, imageBytes_.data() + 16 * i);
            PackLe64(qwords[i].hi, imageBytes_.data() + 16 * i + 8);
        }
        bytes = imageBytes_.data();
    }
    size_t done = 0;
    while (done < count) {
        done += transfers_.AcceptImageQwords(bytes + 16 * done, count - done);
    }
}

void FrontEnd::StripVertices(const StripVertex* vertices, size_t count) {
    // IN-467: in a triangle strip a vertex only matters to the drawing kicks up to two vertices after it, and most of
    // The game's strip vertices do not draw (ADC). Of a run of non-drawing kicks the first goes through Kick (the flush
    // decision: nothing it reads changes during the run) and the last two (the queue the next kick needs, and the
    // registers it leaves); the ones between only count, as Kick would count them.
    const bool skipRuns =
        options_.fastPaths && static_cast<PrimType>(registers_.EffectivePrim() & 7u) == PrimType::TriangleStrip;
    for (size_t i = 0; i < count; ++i) {
        st_ = vertices[i].st;  // WriteRegister(ST) and WriteRegister(RGBAQ) only store these
        rgbaq_ = vertices[i].rgbaq;
        Kick(vertices[i].kick);
        if (skipRuns && !vertices[i].kick.drawing) {
            size_t end = i + 1;
            while (end < count && !vertices[end].kick.drawing) ++end;
            if (end - i > 3u) {
                const size_t skipped = end - i - 3u;
                counters_.Add(Counter::Vertices, skipped);
                counters_.Add(Counter::PrimsSkipped, skipped);
                i += skipped;
            }
        }
    }
}

void FrontEnd::WriteRegister(uint8_t address, uint64_t data) {
    switch (address) {
        case reg::kPrim:
            registers_.Write(address, data, counters_);
            queue_.clear();  // a PRIM write discards a partial primitive
            flushDecisionStale_ = true;
            return;
        case reg::kRgbaq:
            rgbaq_ = data;
            return;
        case reg::kSt:
            st_ = data;
            return;
        case reg::kUv:
            uv_ = data;
            return;
        case reg::kFog:
            fog_ = static_cast<uint8_t>(BitRange(data, 56, 63));
            return;
        case reg::kTexflush: {
            bool texDirty = false;
            if (!batch_.Empty()) {
                const ContextRegs& live = registers_.Context(batch_.state.contextIndex);
                const ContextRegs& snap = batch_.state.ctx;
                const bool indexed = IsIndexedPsm(tex0::Psm(snap.tex0));
                const uint64_t mask = kTex0BaseMask | (indexed ? kTex0IndexedMask : 0u);
                texDirty = Dirty(live.tex0, snap.tex0, mask);
            }
            if (!texDirty) {
                texflushPending_ = true;
                counters_.Add(Counter::TexflushSet);
            }
            return;
        }
        case reg::kBitbltbuf:
            transfers_.WriteBitbltbuf(data);
            return;
        case reg::kTrxpos:
            transfers_.WriteTrxpos(data);
            return;
        case reg::kTrxreg:
            transfers_.WriteTrxreg(data);
            return;
        case reg::kTrxdir:
            transfers_.WriteTrxdir(data);
            return;
        case reg::kHwreg: {
            uint8_t bytes[8];
            PackLe64(data, bytes);
            transfers_.AcceptData(bytes, sizeof(bytes));
            return;
        }
        case reg::kSignal:
        case reg::kLabel:
        case reg::kFinish: {
            const SyncKind kind = address == reg::kSignal   ? SyncKind::Signal
                                  : address == reg::kLabel ? SyncKind::Label
                                                           : SyncKind::Finish;
            const uint32_t id = address == reg::kFinish ? 0u : static_cast<uint32_t>(BitRange(data, 0, 31));
            const uint32_t idMask = address == reg::kFinish ? 0u : static_cast<uint32_t>(BitRange(data, 32, 63));
            if (sync_.OnSync(kind, id, idMask)) {
                FlushBatch(FlushReason::Sync);
            }
            return;
        }
        case reg::kAdPacked:
        case reg::kNop:
        case reg::kXyzf2:
        case reg::kXyz2:
        case reg::kXyzf3:
        case reg::kXyz3:
            return;  // kicks arrive through Kick()
        default:
            break;
    }

    const bool isTex0 = address == reg::kTex0One || address == reg::kTex0Two ||
                        address == reg::kTex2One || address == reg::kTex2Two;
    const uint32_t contextIndex = address & 1u;
    const uint32_t oldTbp = isTex0 ? tex0::Tbp0(registers_.Context(contextIndex).tex0) : 0u;

    const RegisterWriteResult result = registers_.Write(address, data, counters_);
    flushDecisionStale_ = true;
    if (!result.recognised) {
        return;
    }
    if (address == reg::kScissorOne || address == reg::kScissorTwo) {
        scissorWritten_ = true;
    }
    if (isTex0) {
        const uint32_t primContext = static_cast<uint32_t>((registers_.EffectivePrim() >> primbit::kCtxt) & 1u);
        const uint32_t newTbp = tex0::Tbp0(registers_.Context(contextIndex).tex0);
        if (newTbp != oldTbp && contextIndex == primContext) {
            texflushPending_ = false;
        }
        OnTex0Write(contextIndex);
    }
}

void FrontEnd::OnTex0Write(uint32_t contextIndex) {
    const ContextRegs& ctx = registers_.Context(contextIndex);
    const EnvironmentRegs& env = registers_.Env();
    clut::Tex0Write write;
    write.psm = tex0::Psm(ctx.tex0);
    write.cbp = tex0::Cbp(ctx.tex0);
    write.cpsm = tex0::Cpsm(ctx.tex0);
    write.csm = tex0::Csm(ctx.tex0);
    write.csa = tex0::Csa(ctx.tex0);
    write.cld = tex0::Cld(ctx.tex0);
    write.texclutCbw = static_cast<uint32_t>(BitRange(env.texclut, 0, 5));
    write.texclutCou = static_cast<uint32_t>(BitRange(env.texclut, 6, 11));
    write.texclutCov = static_cast<uint32_t>(BitRange(env.texclut, 12, 21));

    // TASK-168: a load that keeps the batch as a new palette segment. The buffer before the load is copied into
    // clutBeforeLoad_ (it is the palette of the primitives so far) and becomes the first segment once the load is done
    // and the batch is still pending (a transfer committed below may still end it).
    bool keepAsSegment = false;
    uint32_t generationBefore = 0;
    const bool loaded = clut_.ProcessWrite(write, localMemory_, [&]() {
        // Chapter 07.3 step 1: an indexed textured batch, or one that writes the
        // CLUT source, renders with the old palette, so it is flushed first.
        bool flush = false;
        if (!batch_.Empty()) {
            const ContextRegs& snapCtx = batch_.state.ctx;
            if (Tme(batch_.state.effectivePrim) && IsIndexedPsm(tex0::Psm(snapCtx.tex0))) {
                flush = true;
            }
            if (BatchTouches(write.cbp, 4, false)) {
                flush = true;
            }
        }
        // TASK-168: instead, the batch goes on with the new palette for the primitives after the load, when the
        // indexed texture is the only reason (PaletteSegmentAllowed) and the renderer allows it.
        if (flush && PaletteSegmentAllowed(write, contextIndex)) {
            flush = false;
            keepAsSegment = true;
            // TASK-168b: only the first kept load of a batch needs the buffer before it (a later one's segment has it).
            if (batch_.paletteSegments.empty()) {
                std::copy(clut_.Halfwords(), clut_.Halfwords() + clut::kClutHalfwords, clutBeforeLoad_.begin());
            }
            generationBefore = clut_.Generation();
        }
        if (flush) {
            // TASK-027 CPU survey R9: how many of these flushes keep the batch's texture (only the palette changes).
            const uint64_t base = kTex0BaseMask | kTex0IndexedMask;
            if (((registers_.Context(contextIndex).tex0 ^ batch_.state.ctx.tex0) & base) == 0u) ++clutFlushSameTexture_;
            FlushBatch(FlushReason::ClutLoad);
        }
        if (beforeLocalRead_) {
            // CSM1 palettes sit in one page from CBP (a second page covers an unaligned CBP); CSM2 reads a
            // TEXCLUT row anywhere in a CBW-wide buffer, so the whole memory is reported (rare in the game).
            const BlockRange range = write.csm == 0 ? BlockRange{write.cbp & ~31u, 64u} : BlockRange{0u, kBlocksTotal};
            beforeLocalRead_(range);
        }
        if (!inTransferHook_) {
            transfers_.CommitPending(TransferReason::ClutLoad);
        }
    });
    if (keepAsSegment && loaded && !batch_.Empty()) {
        ++clutLoadsKept_;
        if (clut_.Generation() != generationBefore) {
            if (batch_.paletteSegments.empty()) {
                PaletteSegment& first = batch_.paletteSegments.emplace_back();
                first.clutGeneration = generationBefore;
                first.clut = clutBeforeLoad_;
            }
            BeginPaletteSegment();
        }
    }
    if (loaded && clutLoaded_) clutLoaded_(write);
}

// TASK-168: the primitives from now on see the CLUT buffer as it is now. A segment that got no primitive since the load
// that opened it takes the new buffer instead (two loads in a row).
void FrontEnd::BeginPaletteSegment() {
    PaletteSegment& last = batch_.paletteSegments.back();
    if (last.firstPrim == batch_.prims.size()) {
        last.clutGeneration = clut_.Generation();
        std::copy(clut_.Halfwords(), clut_.Halfwords() + clut::kClutHalfwords, last.clut.begin());
        return;
    }
    // TASK-168b: built in place (a segment is 1 KiB; a local copy pushed back cost two more copies per kept load).
    PaletteSegment& next = batch_.paletteSegments.emplace_back();
    next.firstPrim = static_cast<uint32_t>(batch_.prims.size());
    next.firstVertex = static_cast<uint32_t>(batch_.vertices.size());
    next.clutGeneration = clut_.Generation();
    std::copy(clut_.Halfwords(), clut_.Halfwords() + clut::kClutHalfwords, next.clut.begin());
}

// TASK-168: whether a CLUT load that would end the pending batch may start a palette segment instead. Each condition
// keeps the merged batch equal to the batches the chapter 07.3 flushes would have made:
//  - the flush is due to the indexed texture alone, which stays the same (TEX0 base and indexed fields, the batch's
//    context): only the CLUT buffer changes, and the segment keeps a copy of it;
//  - the batch's frame and depth buffers do not meet the memory the palette is read from (a CSM1 page, as
//    SetBeforeLocalRead reports it; CSM2 rows can be anywhere), so the load reads the bytes it would have read after
//    the earlier primitives were drawn, and a renderer finds no target of this batch's over that page;
//  - the batch's texture does not meet its frame or depth buffer, so no primitive samples what an earlier one drew;
//  - no auto-flush (chapter 02.6) can split the batch, so no split point depends on the earlier segments;
//  - the CLUT unit copies its entries (the copy is meaningful), the segment count stays bounded, and the renderer agrees.
bool FrontEnd::PaletteSegmentAllowed(const clut::Tex0Write& write, uint32_t contextIndex) const {
    if (!paletteSegmentHook_ || batch_.Empty() || !clut_.CopiesData()) {
        return false;
    }
    const ContextRegs& snapCtx = batch_.state.ctx;
    if (contextIndex != batch_.state.contextIndex || !Tme(batch_.state.effectivePrim) ||
        !IsIndexedPsm(tex0::Psm(snapCtx.tex0))) {
        return false;
    }
    if (Dirty(registers_.Context(contextIndex).tex0, snapCtx.tex0, kTex0BaseMask | kTex0IndexedMask)) {
        return false;
    }
    if (write.csm != 0u) {
        return false;
    }
    // TASK-168b: the renderer's veto on the batch itself, before the memory checks below (about 0.2 us per load).
    if (paletteSegmentBatchFilter_ && !paletteSegmentBatchFilter_(batch_)) {
        return false;
    }
    // The frame and depth footprints below assume rows no wider than the buffer.
    const uint32_t frameWidth = 64u * std::max<uint32_t>(frame::Fbw(snapCtx.frame), 1u);
    if (batch_.drawRect.x1 >= static_cast<int32_t>(frameWidth)) {
        return false;
    }
    if (BatchBuffersMeet(write.cbp & ~31u, 64u)) {
        return false;
    }
    if (TextureMeetsBatchBuffers()) {
        return false;
    }
    const bool autoflushByClass = options_.autoflush == AutoflushMode::All ||
                                  (options_.autoflush == AutoflushMode::Sprites &&
                                   (batch_.cls == PrimClass::Sprite || batch_.cls == PrimClass::Point));
    if (autoflushByClass && AutoflushStateAllows()) {
        return false;
    }
    const size_t segmentsAfter = batch_.paletteSegments.empty() ? 2u : batch_.paletteSegments.size() + 1u;
    if (segmentsAfter > kMaxPaletteSegments) {
        return false;
    }
    return paletteSegmentHook_(write, batch_);
}

// TASK-168: the frame buffer rows the batch draws (whatever FBMSK says: a renderer keeps a target there) and its depth
// buffer rows when it tests or writes depth, as whole page rows like BatchTouches, meet blocks [begin, begin + count).
bool FrontEnd::BatchBuffersMeet(uint32_t begin, uint32_t count) const {
    const ContextRegs& ctx = batch_.state.ctx;
    const uint32_t rows = batch_.drawRect.y1 >= 0 ? static_cast<uint32_t>(batch_.drawRect.y1) + 1u : 1u;
    const uint32_t blocks = FootprintBlocks(frame::Fbw(ctx.frame), rows);
    if (BlockRangesOverlap(frame::Fbp(ctx.frame) * kBlocksPerPage, blocks, begin, count)) {
        return true;
    }
    const bool depthTested = test::Zte(ctx.test) != 0u && test::Ztst(ctx.test) != kZtstAlways;
    const bool depthUsed = zbuf::Zmsk(ctx.zbuf) == 0u || depthTested;
    return depthUsed && BlockRangesOverlap(zbuf::Zbp(ctx.zbuf) * kBlocksPerPage, blocks, begin, count);
}

// TASK-168: the batch's texture (level 0 or a mip level, BatchTouches' footprints) meets its frame or depth buffer
// (BatchBuffersMeet). A texture wider than its buffer row reads further rows: counted as a meeting.
bool FrontEnd::TextureMeetsBatchBuffers() const {
    const ContextRegs& ctx = batch_.state.ctx;
    if ((1u << tex0::Tw(ctx.tex0)) > 64u * std::max<uint32_t>(tex0::Tbw(ctx.tex0), 1u)) {
        return true;
    }
    const uint32_t texBlocks = FootprintBlocks(tex0::Tbw(ctx.tex0), 1u << tex0::Th(ctx.tex0));
    if (BatchBuffersMeet(tex0::Tbp0(ctx.tex0), texBlocks)) {
        return true;
    }
    const uint32_t rows = batch_.drawRect.y1 >= 0 ? static_cast<uint32_t>(batch_.drawRect.y1) + 1u : 1u;
    const uint32_t blocks = FootprintBlocks(frame::Fbw(ctx.frame), rows);
    if (MipLevelsTouch(ctx, texBlocks, frame::Fbp(ctx.frame) * kBlocksPerPage, blocks)) {
        return true;
    }
    const bool depthTested = test::Zte(ctx.test) != 0u && test::Ztst(ctx.test) != kZtstAlways;
    const bool depthUsed = zbuf::Zmsk(ctx.zbuf) == 0u || depthTested;
    return depthUsed && MipLevelsTouch(ctx, texBlocks, zbuf::Zbp(ctx.zbuf) * kBlocksPerPage, blocks);
}

void FrontEnd::Kick(const GifKick& kick) {
    const uint64_t prim = registers_.EffectivePrim();
    const uint32_t typeBits = static_cast<uint32_t>(prim & 7u);
    if (kick.hasFog) {
        fog_ = kick.fog;
    }
    if (typeBits == 7u) {  // reserved type: drop the vertex, reset the queue, draw nothing
        queue_.clear();
        return;
    }
    const PrimType type = static_cast<PrimType>(typeBits);
    const uint32_t contextIndex = static_cast<uint32_t>((prim >> primbit::kCtxt) & 1u);
    queue_.push_back(MakeVertex(kick, registers_.Context(contextIndex)));
    counters_.Add(Counter::Vertices);

    if (!batch_.Empty()) {
        // Chapter 02.5: the decision is taken on every drawing kick; a
        // non-drawing kick only when the class changed or XYOFFSET is dirty.
        const ContextRegs& liveSnapCtx = registers_.Context(batch_.state.contextIndex);
        const bool classChanged = ClassOfType(typeBits) != batch_.cls;
        const bool xyoffsetDirty = Dirty(liveSnapCtx.xyoffset, batch_.state.ctx.xyoffset, kXyoffsetMask);
        if (kick.drawing || classChanged || xyoffsetDirty) {
            // IN-366: DecideFlush reads only the registers and the batch snapshot; while neither changed since it last
            // answered "no flush", it would answer the same.
            if (!options_.fastPaths || flushDecisionStale_) {
                if (const std::optional<FlushReason> reason = DecideFlush()) {
                    FlushBatch(*reason);
                } else {
                    flushDecisionStale_ = false;
                }
            }
        }
    }

    if (options_.fastPaths && !kick.drawing) {
        // IN-366: a non-drawing kick (ADC, about 70-80% of the game's strip vertices) only advances the queue once the
        // primitive has its vertices, exactly as below without the candidate copy and the cull.
        const size_t needed = type == PrimType::Point ? 1u
                              : (type == PrimType::Line || type == PrimType::LineStrip || type == PrimType::Sprite) ? 2u
                                                                                                                     : 3u;
        if (queue_.size() < needed) {
            return;
        }
        counters_.Add(Counter::PrimsSkipped);
        AdvanceQueue(type);
        return;
    }

    Candidate candidate;
    if (!BuildCandidate(type, candidate)) {
        return;  // not enough vertices yet
    }

    PixelRect rect;
    const bool drawable = kick.drawing && ShouldDraw(candidate, type, contextIndex, rect);
    if (drawable) {
        if (!batch_.Empty() && AutoflushWanted(candidate, type)) {
            counters_.Add(Counter::AutoflushSplits);
            FlushBatch(FlushReason::Autoflush);
        }
        if (batch_.Empty()) {
            TakeSnapshot(contextIndex);
        }
        AppendPrimitive(candidate, type, rect);
        const uint32_t limit = batch_.cls == PrimClass::Sprite     ? kLimitSprite
                               : batch_.cls == PrimClass::Triangle ? kLimitTriangle
                                                                   : kLimitPointLine;
        if (batch_.vertices.size() >= limit) {
            // TASK-168: with palette segments the limit counts from the last segment's start, where the CLUT-load
            // flush would have started a batch: what came before it is emitted, and only a full segment flushes.
            if (batch_.paletteSegments.size() > 1u) {
                FlushSegmentsBeforeLast(FlushReason::Capacity);
            }
            if (batch_.vertices.size() >= limit) {
                FlushBatch(FlushReason::Capacity);
            }
        }
    } else {
        counters_.Add(Counter::PrimsSkipped);
    }
    AdvanceQueue(type);
}

Vertex FrontEnd::MakeVertex(const GifKick& kick, const ContextRegs& ctx) const {
    Vertex vertex;
    const int32_t offsetX = static_cast<int32_t>(BitRange(ctx.xyoffset, 0, 15));
    const int32_t offsetY = static_cast<int32_t>(BitRange(ctx.xyoffset, 32, 47));
    vertex.x4 = static_cast<int32_t>(kick.x) - offsetX;
    vertex.y4 = static_cast<int32_t>(kick.y) - offsetY;
    vertex.z = kick.z;
    vertex.r = static_cast<uint8_t>(BitRange(rgbaq_, 0, 7));
    vertex.g = static_cast<uint8_t>(BitRange(rgbaq_, 8, 15));
    vertex.b = static_cast<uint8_t>(BitRange(rgbaq_, 16, 23));
    vertex.a = static_cast<uint8_t>(BitRange(rgbaq_, 24, 31));
    vertex.q = BitsToFloat(static_cast<uint32_t>(rgbaq_ >> 32));
    vertex.s = BitsToFloat(static_cast<uint32_t>(st_));
    vertex.t = BitsToFloat(static_cast<uint32_t>(st_ >> 32));
    vertex.u = static_cast<uint32_t>(BitRange(uv_, 0, 13));
    vertex.v = static_cast<uint32_t>(BitRange(uv_, 16, 29));
    vertex.fog = fog_;
    return vertex;
}

bool FrontEnd::BuildCandidate(PrimType type, Candidate& candidate) const {
    const size_t n = queue_.size();
    switch (type) {
        case PrimType::Point:
            if (n < 1) return false;
            candidate.count = 1;
            candidate.vertices[0] = queue_[n - 1];
            return true;
        case PrimType::Line:
        case PrimType::LineStrip:
        case PrimType::Sprite:
            if (n < 2) return false;
            candidate.count = 2;
            candidate.vertices[0] = queue_[n - 2];
            candidate.vertices[1] = queue_[n - 1];
            return true;
        case PrimType::Triangle:
        case PrimType::TriangleStrip:
            if (n < 3) return false;
            candidate.count = 3;
            candidate.vertices[0] = queue_[n - 3];
            candidate.vertices[1] = queue_[n - 2];
            candidate.vertices[2] = queue_[n - 1];
            return true;
        case PrimType::TriangleFan:
            if (n < 3) return false;
            candidate.count = 3;
            candidate.vertices[0] = queue_[0];
            candidate.vertices[1] = queue_[n - 2];
            candidate.vertices[2] = queue_[n - 1];
            return true;
        default:
            return false;
    }
}

void FrontEnd::AdvanceQueue(PrimType type) {
    switch (type) {
        case PrimType::LineStrip:
        case PrimType::TriangleStrip:
            queue_.erase_first();  // advance by one vertex
            break;
        case PrimType::TriangleFan:
            queue_.keep_first_and_last();  // keep centre and last
            break;
        default:
            queue_.clear();  // lists and sprites drop the queued vertices
            break;
    }
}

bool FrontEnd::ShouldDraw(const Candidate& candidate, PrimType type, uint32_t contextIndex,
                          PixelRect& rect) const {
    if (!scissorWritten_) {
        return false;  // cull rectangle is "nothing" until SCISSOR is written
    }
    const uint64_t scissor = registers_.Context(contextIndex).scissor;
    PixelRect scissorRect;
    scissorRect.x0 = static_cast<int32_t>(BitRange(scissor, 0, 10));
    scissorRect.x1 = static_cast<int32_t>(BitRange(scissor, 16, 26));
    scissorRect.y0 = static_cast<int32_t>(BitRange(scissor, 32, 42));
    scissorRect.y1 = static_cast<int32_t>(BitRange(scissor, 48, 58));
    if (scissorRect.x1 < scissorRect.x0 || scissorRect.y1 < scissorRect.y0) {
        return false;
    }

    int32_t minX = candidate.vertices[0].x4;
    int32_t maxX = minX;
    int32_t minY = candidate.vertices[0].y4;
    int32_t maxY = minY;
    for (uint32_t k = 1; k < candidate.count; ++k) {
        minX = std::min(minX, candidate.vertices[k].x4);
        maxX = std::max(maxX, candidate.vertices[k].x4);
        minY = std::min(minY, candidate.vertices[k].y4);
        maxY = std::max(maxY, candidate.vertices[k].y4);
    }

    const PrimClass cls = ClassOfType(static_cast<uint32_t>(type));
    if (cls == PrimClass::Triangle) {
        for (uint32_t i = 0; i < candidate.count; ++i) {
            for (uint32_t j = i + 1u; j < candidate.count; ++j) {
                if (candidate.vertices[i].x4 == candidate.vertices[j].x4 &&
                    candidate.vertices[i].y4 == candidate.vertices[j].y4) {
                    return false;  // degenerate triangle
                }
            }
        }
    }

    // Cull rectangle: scissor with half a pixel of slack (12.4 units).
    const int32_t cullX0 = scissorRect.x0 * 16 - 8;
    const int32_t cullX1 = scissorRect.x1 * 16 + 8;
    const int32_t cullY0 = scissorRect.y0 * 16 - 8;
    const int32_t cullY1 = scissorRect.y1 * 16 + 8;
    if (maxX < cullX0 || minX > cullX1 || maxY < cullY0 || minY > cullY1) {
        return false;
    }

    PixelRect raw;
    if (cls == PrimClass::Point) {
        const int32_t px = (candidate.vertices[0].x4 + 8) >> 4;  // nearest pixel (Hypothesis, V-01-6)
        const int32_t py = (candidate.vertices[0].y4 + 8) >> 4;
        raw = PixelRect{px, py, px, py};
    } else if (cls == PrimClass::Line) {
        raw = PixelRect{minX >> 4, minY >> 4, maxX >> 4, maxY >> 4};
    } else {
        // Coverage box (chapter 02.3): first and last pixel centres inside the box.
        const int32_t coverX0 = (minX + 15) & ~15;
        const int32_t coverX1 = (maxX - 1) & ~15;
        const int32_t coverY0 = (minY + 15) & ~15;
        const int32_t coverY1 = (maxY - 1) & ~15;
        if (coverX0 > coverX1 || coverY0 > coverY1) {
            if (options_.cullUncovered) {
                return false;  // covers no pixel centre
            }
            // IN-403 (at scale): kept, with the pixels its box touches as its rectangle (the rasteriser, not this
            // rectangle, decides the samples it covers; the rectangle only sizes the target and the scissor).
            raw = PixelRect{minX >> 4, minY >> 4, (maxX - 1) >> 4, (maxY - 1) >> 4};
            if (raw.x1 < raw.x0) raw.x1 = raw.x0;
            if (raw.y1 < raw.y0) raw.y1 = raw.y0;
        } else {
            raw = PixelRect{coverX0 >> 4, coverY0 >> 4, coverX1 >> 4, coverY1 >> 4};
        }
    }

    rect = IntersectRects(raw, scissorRect);
    return !rect.Empty();
}

void FrontEnd::TakeSnapshot(uint32_t contextIndex) {
    flushDecisionStale_ = true;
    batch_.state.env = registers_.Env();
    batch_.state.ctx = registers_.Context(contextIndex);
    batch_.state.contextIndex = contextIndex;
    batch_.state.effectivePrim = registers_.EffectivePrim();
    batch_.state.clutGeneration = clut_.Generation();
    batch_.cls = ClassOfType(static_cast<uint32_t>(batch_.state.effectivePrim & 7u));
}

void FrontEnd::AppendPrimitive(const Candidate& candidate, PrimType type, const PixelRect& rect) {
    PrimRecord record;
    record.type = type;
    record.first = static_cast<uint32_t>(batch_.vertices.size());
    record.count = candidate.count;
    record.rect = rect;

    Vertex vertices[3] = {candidate.vertices[0], candidate.vertices[1], candidate.vertices[2]};
    // Sprites with FST = 0 take the first vertex's Q from the second.
    if (type == PrimType::Sprite && !Fst(batch_.state.effectivePrim)) {
        vertices[0].q = vertices[1].q;
    }
    for (uint32_t k = 0; k < candidate.count; ++k) {
        batch_.vertices.push_back(vertices[k]);
    }
    batch_.prims.push_back(record);
    batch_.drawRect = UnionRects(batch_.drawRect, rect);
    counters_.Add(Counter::PrimsEmitted);
}

std::optional<FlushReason> FrontEnd::DecideFlush() const {
    const EnvironmentRegs& liveEnv = registers_.Env();
    const EnvironmentRegs& snapEnv = batch_.state.env;
    const uint32_t index = batch_.state.contextIndex;
    const ContextRegs& liveCtx = registers_.Context(index);
    const ContextRegs& snapCtx = batch_.state.ctx;
    const uint64_t snapPrim = batch_.state.effectivePrim;
    const uint64_t livePrim = registers_.EffectivePrim();

    // Rule 1: primitive class or attributes changed.
    const uint64_t primDiff = livePrim ^ snapPrim;
    if ((primDiff & 0x7FFu) != 0u &&
        (ClassOfType(static_cast<uint32_t>(livePrim & 7u)) != ClassOfType(static_cast<uint32_t>(snapPrim & 7u)) ||
         (primDiff & MaskBits(3, 10)) != 0u)) {
        return FlushReason::PrimChanged;
    }
    // Rule 2.
    if (Dirty(liveCtx.test, snapCtx.test, kTestMask) ||
        Dirty(liveCtx.scissor, snapCtx.scissor, kScissorMask) ||
        Dirty(liveCtx.xyoffset, snapCtx.xyoffset, kXyoffsetMask) ||
        Dirty(liveEnv.scanmsk, snapEnv.scanmsk, 0x3u) || Dirty(liveEnv.dthe, snapEnv.dthe, 1u)) {
        return FlushReason::TestOrScissor;
    }
    // Rule 3.
    if ((snapEnv.dthe & 1u) != 0u && Dirty(liveEnv.dimx, snapEnv.dimx, kAllBits)) {
        return FlushReason::DitherDimx;
    }
    // Rule 4.
    if (Bit(snapPrim, primbit::kAbe) &&
        (Dirty(liveCtx.alpha, snapCtx.alpha, kAlphaMask) || Dirty(liveEnv.pabe, snapEnv.pabe, 1u))) {
        return FlushReason::AlphaBlend;
    }
    // Rule 5.
    if (Bit(snapPrim, primbit::kFge) && Dirty(liveEnv.fogcol, snapEnv.fogcol, kFogcolMask)) {
        return FlushReason::FogColour;
    }
    // Rule 6.
    const bool colourDirty = Dirty(liveCtx.frame, snapCtx.frame, kFrameMask) ||
                             Dirty(liveEnv.colclamp, snapEnv.colclamp, 1u) ||
                             Dirty(liveCtx.fba, snapCtx.fba, 1u);
    if (colourDirty && CanWriteColour(snapCtx.test)) {
        return FlushReason::ColourWrite;
    }
    // Rule 7.
    if (Dirty(liveCtx.zbuf, snapCtx.zbuf, kZbufMask) && CanWriteZ(snapCtx.test)) {
        return FlushReason::DepthWrite;
    }
    // Rule 8.
    if (Tme(snapPrim)) {
        const bool indexed = IsIndexedPsm(tex0::Psm(snapCtx.tex0)) || IsIndexedPsm(tex0::Psm(liveCtx.tex0));
        const uint64_t tex0Mask = kTex0BaseMask | (indexed ? kTex0IndexedMask : 0u);
        if (Dirty(liveCtx.tex0, snapCtx.tex0, tex0Mask) || Dirty(liveCtx.tex1, snapCtx.tex1, kTex1Mask) ||
            Dirty(liveCtx.clamp, snapCtx.clamp, kClampMask) || Dirty(liveEnv.texa, snapEnv.texa, kTexaMask)) {
            return FlushReason::Texture;
        }
    }
    if (tex1::Mxl(snapCtx.tex1) > 0u &&
        (Dirty(liveCtx.miptbp1, snapCtx.miptbp1, kAllBits) || Dirty(liveCtx.miptbp2, snapCtx.miptbp2, kAllBits))) {
        return FlushReason::Texture;
    }
    return std::nullopt;
}

bool FrontEnd::FrameWritesColour(const ContextRegs& ctx) const {
    return CanWriteColour(ctx.test) && FrameWrittenChannels(ctx.frame) != 0u;
}

bool FrontEnd::ZbufWritten(const ContextRegs& ctx) const {
    return zbuf::Zmsk(ctx.zbuf) == 0u && CanWriteZ(ctx.test);
}

bool FrontEnd::ZbufUsed(const ContextRegs& ctx) const {
    const bool tested = test::Zte(ctx.test) != 0u && test::Ztst(ctx.test) != kZtstAlways;
    return ZbufWritten(ctx) || tested;
}

bool FrontEnd::BatchTouches(uint32_t begin, uint32_t count, bool asDestination) const {
    if (batch_.Empty()) {
        return false;
    }
    const ContextRegs& ctx = batch_.state.ctx;
    const uint32_t rows = batch_.drawRect.y1 >= 0 ? static_cast<uint32_t>(batch_.drawRect.y1) + 1u : 1u;
    const bool frameUsed = FrameWritesColour(ctx) || test::Date(ctx.test) != 0u;
    if (frameUsed) {
        const uint32_t base = frame::Fbp(ctx.frame) * kBlocksPerPage;
        if (BlockRangesOverlap(base, FootprintBlocks(frame::Fbw(ctx.frame), rows), begin, count)) {
            return true;
        }
    }
    if (ZbufUsed(ctx)) {
        const uint32_t base = zbuf::Zbp(ctx.zbuf) * kBlocksPerPage;
        if (BlockRangesOverlap(base, FootprintBlocks(frame::Fbw(ctx.frame), rows), begin, count)) {
            return true;
        }
    }
    if (asDestination && Tme(batch_.state.effectivePrim)) {
        const uint32_t texBlocks = FootprintBlocks(tex0::Tbw(ctx.tex0), 1u << tex0::Th(ctx.tex0));
        if (BlockRangesOverlap(tex0::Tbp0(ctx.tex0), texBlocks, begin, count)) {
            return true;
        }
        if (MipLevelsTouch(ctx, texBlocks, begin, count)) {
            return true;
        }
    }
    return false;
}

// TASK-149: a mipmapped draw also reads its levels 1..MXL, at MIPTBP1 (levels 1-3) and MIPTBP2 (levels 4-6), or with MTBA
// packed after level 0. An upload over any level must draw the pending batch first: the game streams textures by PATH3 over the
// Nordschleife fences' level 2 right after sending the fences, which then showed another texture's pixels. The level
// footprints are as conservative as level 0's (whole page rows); MTBA counts the next level-0 footprint.
bool FrontEnd::MipLevelsTouch(const ContextRegs& ctx, uint32_t levelZeroBlocks, uint32_t begin, uint32_t count) const {
    const uint32_t mxl = std::min<uint32_t>(tex1::Mxl(ctx.tex1), 6u);
    const uint32_t mmin = tex1::Mmin(ctx.tex1);
    if (mxl == 0u || mmin < 2u || mmin > 5u) {
        return false;
    }
    if (tex1::Mtba(ctx.tex1) != 0u &&
        BlockRangesOverlap((tex0::Tbp0(ctx.tex0) + levelZeroBlocks) % kBlocksTotal, levelZeroBlocks, begin, count)) {
        return true;
    }
    const uint32_t height = 1u << tex0::Th(ctx.tex0);
    for (uint32_t level = 1; level <= mxl; ++level) {
        const uint64_t miptbp = level <= 3u ? ctx.miptbp1 : ctx.miptbp2;
        const uint32_t shift = 20u * ((level - 1u) % 3u);
        const uint32_t tbp = static_cast<uint32_t>((miptbp >> shift) & 0x3FFFu);
        const uint32_t tbw = static_cast<uint32_t>((miptbp >> (shift + 14u)) & 0x3Fu);
        if (BlockRangesOverlap(tbp, FootprintBlocks(tbw, std::max(1u, height >> level)), begin, count)) {
            return true;
        }
    }
    return false;
}

void FrontEnd::NoteBatchWritesToClut(const Batch& batch) {
    const ContextRegs& ctx = batch.state.ctx;
    const uint32_t rows = batch.drawRect.y1 >= 0 ? static_cast<uint32_t>(batch.drawRect.y1) + 1u : 1u;
    if (FrameWritesColour(ctx)) {
        clut_.NoteBlocksWritten(frame::Fbp(ctx.frame) * kBlocksPerPage,
                                FootprintBlocks(frame::Fbw(ctx.frame), rows));
    }
    if (ZbufWritten(ctx)) {
        clut_.NoteBlocksWritten(zbuf::Zbp(ctx.zbuf) * kBlocksPerPage,
                                FootprintBlocks(frame::Fbw(ctx.frame), rows));
    }
}

FrontEnd::TexelRect FrontEnd::ComputeTexelRect(const Candidate& candidate) const {
    TexelRect result;
    const ContextRegs& ctx = batch_.state.ctx;
    const bool fst = Fst(batch_.state.effectivePrim);
    const uint32_t tw = tex0::Tw(ctx.tex0);
    const uint32_t th = tex0::Th(ctx.tex0);
    const int32_t sizeU = static_cast<int32_t>(1u << tw);
    const int32_t sizeV = static_cast<int32_t>(1u << th);

    int32_t minU = std::numeric_limits<int32_t>::max();
    int32_t maxU = std::numeric_limits<int32_t>::min();
    int32_t minV = minU;
    int32_t maxV = maxU;
    for (uint32_t k = 0; k < candidate.count; ++k) {
        const Vertex& vertex = candidate.vertices[k];
        int32_t u = 0;
        int32_t v = 0;
        if (fst) {
            u = static_cast<int32_t>(vertex.u >> 4);
            v = static_cast<int32_t>(vertex.v >> 4);
        } else {
            float fu = 0.0f;
            float fv = 0.0f;
            if (!TexelCoordinates(vertex, false, tw, th, fu, fv)) {
                return result;  // NaN: no texel rectangle
            }
            // S/Q is clamped to 1.0 times the texture size before rounding.
            fu = std::clamp(fu, 0.0f, static_cast<float>(sizeU));
            fv = std::clamp(fv, 0.0f, static_cast<float>(sizeV));
            u = std::min(static_cast<int32_t>(std::floor(fu)), sizeU - 1);
            v = std::min(static_cast<int32_t>(std::floor(fv)), sizeV - 1);
        }
        minU = std::min(minU, u);
        maxU = std::max(maxU, u);
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
    }

    int32_t loU = 0, hiU = 0, loV = 0, hiV = 0;
    WrapAxis(clampreg::Wms(ctx.clamp), minU, maxU, sizeU, clampreg::Minu(ctx.clamp),
             clampreg::Maxu(ctx.clamp), loU, hiU);
    WrapAxis(clampreg::Wmt(ctx.clamp), minV, maxV, sizeV, clampreg::Minv(ctx.clamp),
             clampreg::Maxv(ctx.clamp), loV, hiV);
    result.rect = PixelRect{loU, loV, hiU, hiV};
    result.valid = true;
    return result;
}

bool FrontEnd::AutoflushWanted(const Candidate& candidate, PrimType type) const {
    if (options_.autoflush == AutoflushMode::Off) {
        return false;
    }
    const uint64_t prim = batch_.state.effectivePrim;
    const ContextRegs& ctx = batch_.state.ctx;
    if (!Tme(prim)) {
        return false;  // cond. 1
    }
    if (options_.autoflush == AutoflushMode::Sprites && type != PrimType::Sprite && type != PrimType::Point) {
        return false;
    }
    // Cond. 3 (recognised shuffles) is not implemented yet; see the journal entry.
    if (!AutoflushStateAllows()) {
        return false;  // cond. 2, 4 and 5
    }

    // Cond. 6: texel rectangle against the batch's draw rectangle.
    const TexelRect texel = ComputeTexelRect(candidate);
    if (!texel.valid || batch_.drawRect.Empty() || !RectsOverlap(texel.rect, batch_.drawRect)) {
        return false;
    }
    if (texflushPending_) {
        return true;
    }
    // Different texture page than the previous primitive's last texel
    // (64 x 32 texel pages, an approximation noted in the journal).
    const PrimRecord& previous = batch_.prims.back();
    const Vertex& lastVertex = batch_.vertices[previous.first + previous.count - 1u];
    float prevU = 0.0f;
    float prevV = 0.0f;
    if (!TexelCoordinates(lastVertex, Fst(prim), tex0::Tw(ctx.tex0), tex0::Th(ctx.tex0), prevU, prevV)) {
        return true;
    }
    const int32_t prevPageX = static_cast<int32_t>(std::floor(prevU)) >> 6;
    const int32_t prevPageY = static_cast<int32_t>(std::floor(prevV)) >> 5;
    if (texel.rect.x0 >> 6 == prevPageX && texel.rect.y0 >> 5 == prevPageY) {
        return false;
    }
    for (const PrimRecord& earlier : batch_.prims) {
        if (earlier.type == PrimType::Sprite && RectsOverlap(earlier.rect, texel.rect)) {
            return true;
        }
    }
    return false;
}

// Chapter 02.6 conditions 2, 4 and 5 of the auto-flush, which depend on the batch's registers alone: the texture shares
// written channels with the frame, and a candidate level's base is the frame or the depth buffer, not as a co-resident
// pair. False means no primitive can auto-flush this batch.
bool FrontEnd::AutoflushStateAllows() const {
    const ContextRegs& ctx = batch_.state.ctx;
    const uint32_t texFormat = tex0::Psm(ctx.tex0);
    const uint32_t frameFormat = frame::Psm(ctx.frame);
    const uint32_t zFormat = zbuf::Psm(ctx.zbuf);
    const uint32_t frameChannels = FrameWrittenChannels(ctx.frame);
    if ((PsmChannelMask(texFormat) & frameChannels) == 0u) {
        return false;  // cond. 2
    }

    // Cond. 4: candidate levels whose base matches the frame or Z buffer.
    uint32_t levels[7] = {0, 0, 0, 0, 0, 0, 0};
    uint32_t levelCount = 1;
    const uint32_t mxl = tex1::Mxl(ctx.tex1);
    const uint32_t mmin = tex1::Mmin(ctx.tex1);
    if (mxl > 0u && mmin >= 2u) {
        levelCount = 0;
        if (tex1::Lcm(ctx.tex1) == 1u) {
            const int32_t k = tex1::K(ctx.tex1);
            const int32_t lo = std::clamp(k >> 4, 0, static_cast<int32_t>(mxl));
            const int32_t hi = std::clamp((k + 15) >> 4, 0, static_cast<int32_t>(mxl));
            for (int32_t level = lo; level <= hi; ++level) {
                levels[levelCount++] = static_cast<uint32_t>(level);
            }
        } else {
            for (uint32_t level = 0; level <= mxl; ++level) {
                levels[levelCount++] = level;
            }
        }
    }

    const uint32_t frameBase = frame::Fbp(ctx.frame) * kBlocksPerPage;
    const uint32_t zBase = zbuf::Zbp(ctx.zbuf) * kBlocksPerPage;
    const bool frameWritten = FrameWritesColour(ctx);
    const bool zWritten = ZbufWritten(ctx);
    bool allowedFrame = false;
    bool allowedZ = false;
    for (uint32_t i = 0; i < levelCount; ++i) {
        const uint32_t base = LevelBase(ctx, levels[i]);
        if (frameWritten && base == frameBase) {
            allowedFrame = true;
        }
        if (zWritten && base == zBase) {
            allowedZ = true;
        }
    }
    // Cond. 5: a co-resident pair does not split.
    if (allowedFrame && PsmCoResident(texFormat, frameFormat)) {
        allowedFrame = false;
    }
    if (allowedZ && PsmCoResident(texFormat, zFormat)) {
        allowedZ = false;
    }
    return allowedFrame || allowedZ;
}

void FrontEnd::FlushBatch(FlushReason reason) {
    if (flushing_) {
        return;
    }
    flushing_ = true;
    if (!batch_.Empty()) {
        const bool textured = Tme(batch_.state.effectivePrim);
        EmitBatch(batch_, reason);
        if (textured) {
            texflushPending_ = false;  // a textured batch flushed clears the flag
        }
    }
    batch_.Clear();
    if (!inTransferHook_) {
        transfers_.CommitPending(TransferReason::BatchFlush);
    }
    flushing_ = false;
}

// The end of a batch (chapter 02.4, 02.9, 07.2): dropped when its depth test never passes or when the sink drops it on
// arrival (TASK-168b), else rounded, traced and handed to the sink. `batch` keeps (or gets back) vector storage for
// reuse; the caller clears it.
void FrontEnd::EmitBatch(Batch& batch, FlushReason reason) {
    const ContextRegs& ctx = batch.state.ctx;
    const bool textured = Tme(batch.state.effectivePrim);
    const bool discard = test::Zte(ctx.test) != 0u && test::Ztst(ctx.test) == kZtstNever;
    if (discard) {
        counters_.Add(Counter::BatchesDiscarded);
        return;
    }
    // TASK-168: a segment opened by a load after the last primitive has no primitive: dropped.
    if (!batch.paletteSegments.empty() && batch.paletteSegments.back().firstPrim >= batch.prims.size()) {
        batch.paletteSegments.pop_back();
    }
    // TASK-168b: a batch the sink drops on arrival skips the rounding, the trace and the hand-over below; the rest of
    // this function's bookkeeping happens as for any batch.
    if (batches_.DropOnArrival(batch)) {
        ++batchesDroppedOnArrival_;
        NoteBatchWritesToClut(batch);
        CountEmittedBatch(batch, reason);
        Batch dropped = std::move(batch);
        batch = Batch{};
        RecycleStorage(batch, dropped);
        return;
    }
    // Chapter 02.4 decides per batch (sprites, or constant Z): per palette segment here, each being the batch a
    // CLUT-load flush would have ended (TASK-168).
    const uint32_t segments = PaletteSegmentCount(batch);
    for (uint32_t segment = 0; segment < segments; ++segment) {
        const uint32_t firstPrim = batch.paletteSegments.empty() ? 0u : batch.paletteSegments[segment].firstPrim;
        const uint32_t endPrim = PaletteSegmentEndPrim(batch, segment);
        const uint32_t firstVertex = batch.paletteSegments.empty() ? 0u : batch.paletteSegments[segment].firstVertex;
        const uint32_t endVertex = segment + 1u < batch.paletteSegments.size()
                                       ? batch.paletteSegments[segment + 1u].firstVertex
                                       : static_cast<uint32_t>(batch.vertices.size());
        bool constantZ = true;
        for (uint32_t index = firstVertex; index < endVertex; ++index) {
            if (batch.vertices[index].z != batch.vertices[firstVertex].z) {
                constantZ = false;
                break;
            }
        }
        ApplyStRoundingToPrims(batch, constantZ, firstPrim, endPrim);
    }

    if (options_.vertexTrace) {
        clut::Palette palette;
        const clut::Palette* palettePtr = nullptr;
        if (textured && IsIndexedPsm(tex0::Psm(ctx.tex0))) {
            // The alpha range over every palette of the batch (one without segments).
            palette = DecodeSegmentPalette(batch, 0, clut_);
            for (uint32_t segment = 1; segment < segments; ++segment) {
                const clut::Palette other = DecodeSegmentPalette(batch, segment, clut_);
                palette.minAlpha = std::min(palette.minAlpha, other.minAlpha);
                palette.maxAlpha = std::max(palette.maxAlpha, other.maxAlpha);
            }
            palettePtr = &palette;
        }
        batch.trace = ComputeVertexTrace(batch, palettePtr);
    }
    NoteBatchWritesToClut(batch);

    CountEmittedBatch(batch, reason);
    Batch out = std::move(batch);
    batch = Batch{};
    batches_.OnBatch(std::move(out));
    RecycleStorage(batch, out);
}

// The flush reason and the counts of an emitted batch (drawn or dropped on arrival).
void FrontEnd::CountEmittedBatch(Batch& batch, FlushReason reason) {
    batch.reason = reason;
    flushCounts_[static_cast<size_t>(reason)]++;
    counters_.Add(Counter::BatchesEmitted);
    if (!batch.paletteSegments.empty()) {
        ++segmentedBatches_;
        paletteSegmentsEmitted_ += batch.paletteSegments.size();
    }
}

// IN-360: a sink that did not keep the batch leaves its storage in `used`; the next batch (`batch`, a fresh one) reuses
// it (cleared, capacity kept) instead of growing new vectors.
void FrontEnd::RecycleStorage(Batch& batch, Batch& used) {
    used.vertices.clear();
    used.prims.clear();
    used.paletteSegments.clear();
    batch.vertices.swap(used.vertices);
    batch.prims.swap(used.prims);
    batch.paletteSegments.swap(used.paletteSegments);
}

// TASK-168: the capacity limit reached by a batch with palette segments. The CLUT-load flushes would have started a
// batch at the last segment, so the limit counts from there: the primitives before it are emitted now (with their
// segments) and the last segment's stay pending as a batch of their own, whose palette is the CLUT unit's buffer (the
// load that opened that segment was the last one). Nothing else happens here that a flush does (no transfer commit, the
// TEXFLUSH flag stays), as no flush happened at this point before.
void FrontEnd::FlushSegmentsBeforeLast(FlushReason reason) {
    if (flushing_ || batch_.paletteSegments.size() < 2u) {
        return;
    }
    flushing_ = true;
    const PaletteSegment last = batch_.paletteSegments.back();
    Batch head;
    head.state = batch_.state;
    head.cls = batch_.cls;
    head.vertices.assign(batch_.vertices.begin(), batch_.vertices.begin() + last.firstVertex);
    head.prims.assign(batch_.prims.begin(), batch_.prims.begin() + last.firstPrim);
    head.paletteSegments.assign(batch_.paletteSegments.begin(), batch_.paletteSegments.end() - 1);
    for (const PrimRecord& record : head.prims) {
        head.drawRect = UnionRects(head.drawRect, record.rect);
    }

    batch_.vertices.erase(batch_.vertices.begin(), batch_.vertices.begin() + last.firstVertex);
    batch_.prims.erase(batch_.prims.begin(), batch_.prims.begin() + last.firstPrim);
    batch_.drawRect = PixelRect{};
    for (PrimRecord& record : batch_.prims) {
        record.first -= last.firstVertex;
        batch_.drawRect = UnionRects(batch_.drawRect, record.rect);
    }
    batch_.paletteSegments.clear();
    batch_.state.clutGeneration = last.clutGeneration;

    EmitBatch(head, reason);
    flushing_ = false;
}

void FrontEnd::VBlank(const BlockRange* displays, size_t count) {
    if (!inTransferHook_) {
        transfers_.CommitPending(TransferReason::VBlank);
    }
    if (batch_.Empty()) {
        return;
    }
    const ContextRegs& ctx = batch_.state.ctx;
    const uint32_t rows = batch_.drawRect.y1 >= 0 ? static_cast<uint32_t>(batch_.drawRect.y1) + 1u : 1u;
    const uint32_t frameBase = frame::Fbp(ctx.frame) * kBlocksPerPage;
    const uint32_t frameBlocks = FootprintBlocks(frame::Fbw(ctx.frame), rows);
    for (size_t i = 0; i < count; ++i) {
        if (BlockRangesOverlap(frameBase, frameBlocks, displays[i].begin, displays[i].count)) {
            FlushBatch(FlushReason::VBlank);
            return;
        }
    }
}

void FrontEnd::ResetGs() {
    batch_.Clear();
    batch_ = Batch{};
    queue_.clear();
    registers_.Reset();
    flushDecisionStale_ = true;
    clut_.Reset();
    transfers_.Reset();
    texflushPending_ = false;
    scissorWritten_ = false;
    rgbaq_ = 0;
    st_ = 0;
    uv_ = 0;
    fog_ = 0;
    flushing_ = false;
    inTransferHook_ = false;
}

}  // namespace coregs
