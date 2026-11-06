// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec/CORE2-SPEC.md and CORE2-DESIGN.md (TASK-027 core rewrite), clean room
// (decision 0070).
//
// HOST->LOCAL uploads into the CPU copy of GS memory through precomputed in-page tables (memory::gpuAddressTables,
// the same numbers as memory::pixelSlot and memory::hostToLocal).

#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>

#include "hw_memory.hpp"

namespace coregs::core2 {

enum class UploadSwizzle : std::uint8_t {
    PerPixel,  // every pixel addressed on its own (three divisions and a format switch per pixel)
    PageRuns,  // IN-414: page-aligned buffers in runs of one page row, one table read per pixel
};

// Writes `width` x `height` host pixels (chapter 02.7 layout) at (x0, y0) of `view` into the 4 MiB image
// `memoryBytes`, wrapping transfer coordinates at 2048. Returns false (nothing written) when `byteCount` does not
// cover the whole rectangle or the PSM has no table; the caller then takes the generic path. PageRuns falls back to
// PerPixel for a base pointer that is not page aligned. Both give the same bytes as memory::hostToLocal.
// IN-450: `changed`, when given, is set to false when only pixels equal to the ones already there were written (the game
// uploads most of its textures again every frame). IN-453: `changedBlocks`, when given, is a bit set over the 16384
// blocks of GS memory in which the bit of every block where a written pixel changed is set (the caller clears them).
bool FastHostToLocal(std::uint8_t *memoryBytes, const memory::BufferView &view, std::uint32_t x0, std::uint32_t y0,
                     std::uint32_t width, std::uint32_t height, const std::uint8_t *data, std::size_t byteCount,
                     UploadSwizzle swizzle, bool *changed = nullptr, std::uint64_t *changedBlocks = nullptr);

// IN-456: the block (0..16383) holding pixel (x, y) of `view` (x, y < 2048), from the same in-page tables: equal to
// memory::pixelSlot(view, x, y).byteOffset / 256 for every PSM and base pointer, a table read instead of the divisions.
std::uint32_t BlockOfPixel(const memory::BufferView &view, std::uint32_t x, std::uint32_t y);

// IN-508: the pages of GS memory (modulo 512) holding the rectangle (x0, y0)-(x1, y1), corners included, of `view`, in
// that format's page size. A page-aligned base keeps every block of a format's page in one memory page, so whole pages
// of the format are walked; any other base spreads a format's page over two, so the rectangle's block cells are walked
// (at most 1024 of them). Returns false, with `pages` undefined, when the answer would not be exact or would take
// longer: a buffer width that is not whole pages, a rectangle outside the buffer's width or the 2048 range, or too many
// cells. An empty rectangle holds no page.
using PageSet = std::bitset<memory::kPagesTotal>;
bool PagesOfRect(const memory::BufferView &view, std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1,
                 PageSet &pages);

}  // namespace coregs::core2
