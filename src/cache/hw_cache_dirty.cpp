// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors
//
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapter 05.1.
// Clean room.

#include "hw_cache_dirty.hpp"

#include <algorithm>

namespace coregs::cache {

bool canMergeRects(const Rect& first, const Rect& second) {
    // Share both vertical edges (same x0 and x1) and touch vertically.
    if (first.x0 == second.x0 && first.x1 == second.x1 && first.y1 >= second.y0 && second.y1 >= first.y0) {
        return true;
    }
    // Share both horizontal edges (same y0 and y1) and touch horizontally.
    if (first.y0 == second.y0 && first.y1 == second.y1 && first.x1 >= second.x0 && second.x1 >= first.x0) {
        return true;
    }
    return false;
}

void DirtyList::add(const DirtyEntry& entry, const Rect& validRect) {
    DirtyEntry current = entry;
    current.rect = entry.rect.intersect(validRect);
    if (current.rect.empty() || current.channels == 0) return;

    auto sameKey = [&current](const DirtyEntry& other) {
        return other.writerPsm == current.writerPsm && other.writerBw == current.writerBw &&
               other.channels == current.channels;
    };

    for (const DirtyEntry& other : entries_) {
        if (sameKey(other) && other.rect.contains(current.rect)) return;  // already dirty
    }

    bool merged = true;
    while (merged) {
        merged = false;
        for (std::size_t index = 0; index < entries_.size(); ++index) {
            const DirtyEntry& other = entries_[index];
            if (!sameKey(other)) continue;
            if (!(current.rect.contains(other.rect) || canMergeRects(other.rect, current.rect))) continue;
            current.rect = boundingBox(current.rect, other.rect);
            entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
            merged = true;
            break;
        }
    }
    entries_.insert(entries_.begin(), current);
}

bool DirtyList::overlaps(const Rect& rect) const {
    return std::any_of(entries_.begin(), entries_.end(),
                       [&rect](const DirtyEntry& entry) { return entry.rect.overlaps(rect); });
}

bool DirtyList::fullyDirty(const Rect& validRect, std::uint8_t channels) const {
    if (entries_.size() != 1 || validRect.empty()) return false;
    const DirtyEntry& only = entries_.front();
    return (only.channels & channels) == channels && only.rect.contains(validRect);
}

}  // namespace coregs::cache
