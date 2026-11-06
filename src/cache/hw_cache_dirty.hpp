// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors
//
// HWR dirty lists (TASK-027, milestone M7).
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapter 05.1
// (dirty list and merge rule), chapter 04.1 (re-expression). Clean room.

#pragma once

#include "hw_cache_geometry.hpp"

#include <cstdint>
#include <vector>

namespace coregs::cache {

// One rectangle where the shadow memory is newer than the target. The rectangle
// is stored in target coordinates; the writer layout is kept for the merge key
// and for diagnostics.
struct DirtyEntry {
    Rect rect{};
    Psm writerPsm = Psm::Ct32;
    std::uint32_t writerBw = 1;
    std::uint8_t channels = kChanAll;
};

class DirtyList {
public:
    // Adds a rectangle (clipped to the valid area) with the chapter 05.1 merge
    // rule: a rectangle contained in a same-key entry is dropped; mergeable
    // entries (sharing both vertical edges and touching vertically, or sharing
    // both horizontal edges and touching horizontally) and entries inside the
    // new rectangle are merged into it. The newest entry comes first.
    void add(const DirtyEntry& entry, const Rect& validRect);

    void clear() { entries_.clear(); }
    bool empty() const { return entries_.empty(); }
    const std::vector<DirtyEntry>& entries() const { return entries_; }

    // Any entry shares area with the rectangle.
    bool overlaps(const Rect& rect) const;

    // Exactly one entry, carrying every requested channel, covering the whole
    // valid rectangle (chapter 05.1 "fully dirty").
    bool fullyDirty(const Rect& validRect, std::uint8_t channels) const;

private:
    std::vector<DirtyEntry> entries_;
};

// Two rectangles that may be merged into their bounding box without adding
// area (chapter 05.1 merge conditions).
bool canMergeRects(const Rect& first, const Rect& second);

}  // namespace coregs::cache
