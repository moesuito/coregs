// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec/CORE2-SPEC.md and CORE2-DESIGN.md (TASK-027 core rewrite), clean room
// (decision 0070).
//
// CTest hw_core2_upload_swizzle (IN-414): core 2's HOST->LOCAL fast path, in both modes (PerPixel and PageRuns),
// writes exactly the bytes of the generic memory::hostToLocal for every PSM: page-aligned and unaligned base
// pointers, several buffer widths, odd sizes, rectangles across pages and across the 2048 transfer wrap.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

#include "core2/core2_upload.hpp"
#include "hw_memory.hpp"

namespace {

using coregs::core2::BlockOfPixel;
using coregs::core2::FastHostToLocal;
using coregs::core2::PageSet;
using coregs::core2::PagesOfRect;
using coregs::core2::UploadSwizzle;
using namespace coregs::memory;

int failures = 0;
int cases = 0;

std::uint64_t NextRandom(std::uint64_t &state) {  // splitmix64
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void CheckCase(std::uint64_t &state, Psm psm, std::uint32_t bp, std::uint32_t bw, std::uint32_t x, std::uint32_t y,
               std::uint32_t width, std::uint32_t height) {
    // The same random background in all three memories, so untouched bytes (and the other nibble of 4-bit pixels)
    // are compared too.
    LocalMemory reference;
    for (std::uint32_t i = 0; i < kLocalMemoryBytes; i += 8) {
        const std::uint64_t word = NextRandom(state);
        std::memcpy(reference.bytes().data() + i, &word, 8);
    }
    std::vector<std::uint8_t> data(hostBytesFor(psm, width, height));
    for (std::uint8_t &byte : data) byte = static_cast<std::uint8_t>(NextRandom(state));
    std::vector<std::uint8_t> perPixel(reference.bytes().begin(), reference.bytes().end());
    std::vector<std::uint8_t> pageRuns = perPixel;

    const BufferView view{bp, bw, psm};
    hostToLocal(reference, view, TransferRect{x, y, width, height}, std::span<const std::uint8_t>(data));
    const bool okPerPixel =
        FastHostToLocal(perPixel.data(), view, x, y, width, height, data.data(), data.size(), UploadSwizzle::PerPixel);
    const bool okPageRuns =
        FastHostToLocal(pageRuns.data(), view, x, y, width, height, data.data(), data.size(), UploadSwizzle::PageRuns);
    ++cases;
    const bool samePerPixel = okPerPixel && std::memcmp(perPixel.data(), reference.bytes().data(), kLocalMemoryBytes) == 0;
    const bool samePageRuns = okPageRuns && std::memcmp(pageRuns.data(), reference.bytes().data(), kLocalMemoryBytes) == 0;
    if (!samePerPixel || !samePageRuns) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s bp=%u bw=%u rect=%u,%u %ux%u per_pixel=%s page_runs=%s\n", psmInfo(psm).name, bp,
                     bw, x, y, width, height, samePerPixel ? "same" : "differs", samePageRuns ? "same" : "differs");
    }
}

}  // namespace

// --bench: the time of FastHostToLocal for GT4-like uploads (512 x 256 CT32 and T8, page aligned), for A/B runs of
// the swizzle switches (COREGS_CORE2_CT32_COLUMNS=0). Not part of the CTest pass/fail.
int Bench() {
    std::vector<std::uint8_t> local(kLocalMemoryBytes, 0);
    for (const Psm psm : {Psm::Ct32, Psm::T8}) {
        std::vector<std::uint8_t> data(hostBytesFor(psm, 512, 256));
        std::uint64_t state = 7;
        for (std::uint8_t &byte : data) byte = static_cast<std::uint8_t>(NextRandom(state));
        const BufferView view{0, 8, psm};
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 200; ++i) {
            data[static_cast<std::size_t>(i)] ^= 1;  // a change every time
            FastHostToLocal(local.data(), view, 0, 0, 512, 256, data.data(), data.size(), UploadSwizzle::PageRuns);
        }
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
        std::printf("bench %s: %.2f ns per pixel\n", psmInfo(psm).name, ns / (200.0 * 512 * 256));
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && std::string_view(argv[1]) == "--bench") return Bench();
    const Psm psms[] = {Psm::Ct32, Psm::Ct24, Psm::Ct16, Psm::Ct16S, Psm::T8,  Psm::T4,   Psm::T8H,
                        Psm::T4Hl, Psm::T4Hh, Psm::Z32,  Psm::Z24,   Psm::Z16, Psm::Z16S};
    std::uint64_t state = 0x1A414u;
    for (const Psm psm : psms) {
        // Fixed shapes: one page, a rectangle crossing pages at odd offsets, one row, one column, and the wrap.
        CheckCase(state, psm, 0, 1, 0, 0, psmInfo(psm).pageWidth, psmInfo(psm).pageHeight);
        CheckCase(state, psm, 64, 4, 13, 7, 211, 45);
        CheckCase(state, psm, 96, 10, 5, 3, 640, 1);
        CheckCase(state, psm, 32, 2, 77, 0, 1, 130);
        CheckCase(state, psm, 512, 2, 2040, 2041, 17, 13);  // both coordinates wrap at 2048
        CheckCase(state, psm, 7, 3, 9, 4, 50, 21);         // base pointer not page aligned: PerPixel fallback
        // Random shapes.
        for (int i = 0; i < 6; ++i) {
            const bool aligned = (NextRandom(state) & 3u) != 0;
            const std::uint32_t bp = static_cast<std::uint32_t>(NextRandom(state) % kBlocksTotal) & (aligned ? ~31u : ~0u);
            const std::uint32_t bw = 1u + static_cast<std::uint32_t>(NextRandom(state) % 16u);
            const std::uint32_t x = static_cast<std::uint32_t>(NextRandom(state) % 2048u);
            const std::uint32_t y = static_cast<std::uint32_t>(NextRandom(state) % 2048u);
            const std::uint32_t width = 1u + static_cast<std::uint32_t>(NextRandom(state) % 300u);
            const std::uint32_t height = 1u + static_cast<std::uint32_t>(NextRandom(state) % 80u);
            CheckCase(state, psm, bp, bw, x, y, width, height);
        }
    }
    // IN-450: the same upload again changes nothing and says so; one changed pixel (or nibble) is reported.
    for (const Psm psm : psms) {
        std::vector<std::uint8_t> local(kLocalMemoryBytes, 0);
        std::vector<std::uint8_t> data(hostBytesFor(psm, 37, 11));
        for (std::uint8_t &byte : data) byte = static_cast<std::uint8_t>(NextRandom(state));
        const BufferView view{64, 2, psm};
        bool first = false, again = true, touched = false;
        FastHostToLocal(local.data(), view, 5, 3, 37, 11, data.data(), data.size(), UploadSwizzle::PageRuns, &first);
        FastHostToLocal(local.data(), view, 5, 3, 37, 11, data.data(), data.size(), UploadSwizzle::PageRuns, &again);
        data[data.size() / 2] ^= 0x10;  // inside one pixel (the high nibble of a 4-bit pair, a bit of the others)
        std::vector<std::uint64_t> blocks(kBlocksTotal / 64, 0);  // IN-453: only that pixel's block is reported
        FastHostToLocal(local.data(), view, 5, 3, 37, 11, data.data(), data.size(), UploadSwizzle::PageRuns, &touched,
                        blocks.data());
        const std::uint32_t bpp = psmInfo(psm).transferBpp;
        const std::size_t pixel = bpp == 4 ? (data.size() / 2) * 2 + 1 : (data.size() / 2) / (bpp / 8);
        const std::uint32_t expected =
            pixelSlot(view, 5 + static_cast<std::uint32_t>(pixel % 37), 3 + static_cast<std::uint32_t>(pixel / 37))
                .byteOffset /
            kBlockBytes;
        int reported = 0;
        bool hasExpected = false;
        for (std::uint32_t b = 0; b < kBlocksTotal; ++b) {
            if ((blocks[b >> 6] >> (b & 63u)) & 1u) {
                ++reported;
                hasExpected = hasExpected || b == expected;
            }
        }
        ++cases;
        if (!first || again || !touched || reported != 1 || !hasExpected) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s changed flags first=%d again=%d touched=%d blocks=%d expected=%u found=%d\n",
                         psmInfo(psm).name, first ? 1 : 0, again ? 1 : 0, touched ? 1 : 0, reported, expected,
                         hasExpected ? 1 : 0);
        }
    }
    // IN-456: BlockOfPixel equals pixelSlot's block for every PSM, base pointer (aligned or not) and buffer width.
    for (const Psm psm : psms) {
        for (int i = 0; i < 400; ++i) {
            const std::uint32_t bp = static_cast<std::uint32_t>(NextRandom(state) % kBlocksTotal);
            const std::uint32_t bw = 1u + static_cast<std::uint32_t>(NextRandom(state) % 32u);
            const std::uint32_t x = static_cast<std::uint32_t>(NextRandom(state) % 2048u);
            const std::uint32_t y = static_cast<std::uint32_t>(NextRandom(state) % 2048u);
            const BufferView view{bp, bw, psm};
            const std::uint32_t expected = pixelSlot(view, x, y).byteOffset / kBlockBytes;
            const std::uint32_t got = BlockOfPixel(view, x, y);
            if (got != expected) {
                ++failures;
                std::fprintf(stderr, "FAIL: BlockOfPixel %s bp=%u bw=%u (%u,%u) = %u, pixelSlot says %u\n",
                             psmInfo(psm).name, bp, bw, x, y, got, expected);
                break;
            }
        }
        ++cases;
    }
    // IN-508: PagesOfRect agrees with pixelSlot for every pixel of the rectangle, for every PSM, and refuses what it cannot
    // answer exactly.
    for (const Psm psm : psms) {
        const PsmInfo &info = psmInfo(psm);
        const std::uint32_t bufferWidth = (info.pageWidth > 64u ? 2u : 1u) * (1u + static_cast<std::uint32_t>(NextRandom(state) % 4u));
        for (int i = 0; i < 40; ++i) {
            const std::uint32_t basePage = static_cast<std::uint32_t>(NextRandom(state) % kPagesTotal);
            // Half of the bases inside a page; those rectangles stay small enough to be answered.
            const std::uint32_t inPage = i % 2 == 0 ? 0u : static_cast<std::uint32_t>(NextRandom(state) % kBlocksPerPage);
            const BufferView view{basePage * kBlocksPerPage + inPage, bufferWidth, psm};
            const std::int32_t width = static_cast<std::int32_t>(bufferWidth * 64u);
            const std::int32_t x0 = static_cast<std::int32_t>(NextRandom(state) % static_cast<std::uint32_t>(width));
            const std::int32_t x1 = x0 + static_cast<std::int32_t>(NextRandom(state) % static_cast<std::uint32_t>(width - x0));
            const std::int32_t y0 = static_cast<std::int32_t>(NextRandom(state) % 512u);
            const std::int32_t y1 = y0 + static_cast<std::int32_t>(NextRandom(state) % 64u);
            PageSet expected, got;
            for (std::int32_t y = y0; y <= y1; ++y) {
                for (std::int32_t x = x0; x <= x1; ++x) {
                    expected.set(pixelSlot(view, static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y)).byteOffset / kBlockBytes / kBlocksPerPage);
                }
            }
            if (!PagesOfRect(view, x0, y0, x1, y1, got) || got != expected) {
                ++failures;
                std::fprintf(stderr, "FAIL: PagesOfRect %s bp=%u bw=%u (%d,%d)-(%d,%d)\n", info.name, view.basePointer, bufferWidth, x0, y0, x1, y1);
                break;
            }
        }
        ++cases;
    }
    {
        // GT4 on George V Paris: the Z32 target at BP 11232, BW 4, drawn 192 x 256, holds pages 351-381 without 354, 358,
        // ... 378; the streamed CT32 texture at BP 12224, BW 1, starts right after, at page 382. The whole target has 382.
        const BufferView depth{11232, 4, Psm::Z32}, stream{12224, 1, Psm::Ct32};
        PageSet drawn, whole, written;
        const bool answered = PagesOfRect(depth, 0, 0, 191, 255, drawn) && PagesOfRect(depth, 0, 0, 255, 255, whole) &&
                              PagesOfRect(stream, 0, 0, 63, 1439, written);
        ++cases;
        if (!answered || drawn.count() != 24 || !drawn[351] || !drawn[381] || drawn[354] || whole.count() != 32 ||
            written.count() != 45 || !written[382] || !written[426] || (drawn & written).any() || !(whole & written)[382]) {
            ++failures;
            std::fprintf(stderr, "FAIL: PagesOfRect on GT4's target and stream\n");
        }
        // The end of memory wraps: 2 pages per row from page 511.
        PageSet wrapped;
        ++cases;
        if (!PagesOfRect(BufferView{511u * kBlocksPerPage, 2, Psm::Ct32}, 0, 0, 127, 31, wrapped) || wrapped.count() != 2 ||
            !wrapped[511] || !wrapped[0]) {
            ++failures;
            std::fprintf(stderr, "FAIL: PagesOfRect does not wrap at 512 pages\n");
        }
        // Formats with other page sizes, on the same memory: T8 128 x 64 per page, T4 128 x 128, CT16 64 x 64.
        PageSet t8, t4, ct16;
        ++cases;
        if (!PagesOfRect(BufferView{32, 2, Psm::T8}, 0, 0, 127, 63, t8) || t8.count() != 1 || !t8[1] ||
            !PagesOfRect(BufferView{32, 2, Psm::T4}, 0, 0, 127, 127, t4) || t4.count() != 1 || !t4[1] ||
            !PagesOfRect(BufferView{32, 1, Psm::Ct16}, 0, 64, 63, 127, ct16) || ct16.count() != 1 || !ct16[2]) {
            ++failures;
            std::fprintf(stderr, "FAIL: PagesOfRect ignores a format's page size\n");
        }
        // GT4's small textures at a base inside page 382: only that page, and so none of the drawn pages.
        PageSet small;
        ++cases;
        if (!PagesOfRect(BufferView{12232, 1, Psm::Ct32}, 0, 0, 7, 7, small) || small.count() != 1 || !small[382] || (drawn & small).any()) {
            ++failures;
            std::fprintf(stderr, "FAIL: PagesOfRect on GT4's small texture inside page 382\n");
        }
        // What it must refuse: too many block cells from a base inside a page, a width of half a page, past the width, past 2048.
        PageSet refused;
        ++cases;
        if (PagesOfRect(BufferView{8, 1, Psm::Ct32}, 0, 0, 63, 2047, refused) || PagesOfRect(BufferView{0, 1, Psm::T8}, 0, 0, 7, 7, refused) ||
            PagesOfRect(BufferView{0, 1, Psm::Ct32}, 0, 0, 64, 7, refused) || PagesOfRect(BufferView{0, 1, Psm::Ct32}, 0, 0, 7, 2048, refused)) {
            ++failures;
            std::fprintf(stderr, "FAIL: PagesOfRect answered a case it cannot answer exactly\n");
        }
    }
    // Too little data: nothing written, the caller takes the generic path.
    std::vector<std::uint8_t> memory(kLocalMemoryBytes, 0xA5);
    const std::uint8_t few[16] = {};
    if (FastHostToLocal(memory.data(), BufferView{0, 1, Psm::Ct32}, 0, 0, 8, 8, few, sizeof few, UploadSwizzle::PageRuns)) {
        ++failures;
        std::fprintf(stderr, "FAIL: a short upload was accepted\n");
    }
    std::printf("hw_core2_upload_swizzle: %d cases, %d failures\n", cases, failures);
    return failures == 0 ? 0 : 1;
}
