// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// HWR front end and CLUT counters (exported to F8 later; chapters 02, 07, 14).
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace coregs {

// Why a batch was flushed (chapter 02.5, 02.6, 02.7, 02.8, 07.3).
enum class FlushReason : uint8_t {
    PrimChanged,         // rule 1
    TestOrScissor,       // rule 2 (TEST, SCISSOR, XYOFFSET, SCANMSK, DTHE)
    DitherDimx,          // rule 3
    AlphaBlend,          // rule 4 (ALPHA or PABE with ABE)
    FogColour,           // rule 5
    ColourWrite,         // rule 6 (FRAME, COLCLAMP, FBA)
    DepthWrite,          // rule 7 (ZBUF)
    Texture,             // rule 8 (TEX0, TEX1, CLAMP, TEXA, MIPTBP)
    ClutLoad,            // chapter 07.3
    Transfer,            // chapter 02.7 overlap check
    VBlank,              // chapter 02.8
    Autoflush,           // chapter 02.6
    Capacity,            // chapter 02.3 limits
    GsReset,
    Sync,                // SIGNAL/FINISH/LABEL requested a flush
    Count,
};

// Front end counters. Names follow the specification's F8 vocabulary.
enum class Counter : uint8_t {
    GifTags,
    GifPacketsEnded,
    GifPathOneNoEop,
    PackedItems,
    RegListWords,
    ImageQwords,
    ImageDiscarded,
    PreInRegList,        // V-02-1
    QSanitised,          // chapter 02.1
    PrmodecontAcZero,    // V-02-2
    TexclutHighBits,     // V-01-1
    FramePsm4h,          // V-01-2
    Csm2NonSixteen,
    ClutBadCpsm,
    ClutLoadsRequested,
    ClutLoadsPerformed,
    ClutLoadsSkipped,
    InvalidPsm,
    PrimsEmitted,
    PrimsSkipped,
    BatchesEmitted,
    BatchesDiscarded,
    AutoflushSplits,
    TexflushSet,
    TransferCommits,
    TransferOverlapFlushes,
    Vertices,
    Count,
};

class CounterSet {
public:
    void Add(Counter counter, uint64_t amount = 1) {
        values_[static_cast<size_t>(counter)] += amount;
    }
    uint64_t Get(Counter counter) const { return values_[static_cast<size_t>(counter)]; }
    void Reset() { values_.fill(0); }

    static constexpr size_t kCount = static_cast<size_t>(Counter::Count);

private:
    std::array<uint64_t, kCount> values_{};
};

}  // namespace coregs
