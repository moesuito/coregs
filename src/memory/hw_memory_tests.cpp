// SPDX-License-Identifier: MIT
// Copyright 2026 GT4Recomp contributors
//
// CTest fixtures for the HWR local memory model (TASK-027 M2).
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters
// 01.3, 02.7, 09.1, 10.2 and 14 (test vectors V-01-3, V-01-4, V-10-1 is M5/M4
// territory and is not checked here).
//
// Donor oracle: the golden values below were produced once by running the MIT
// donor's pixel accessors (private/references/ps2-web-recomp, runtime/src/gs,
// commit 1b4808793a12c45125262fd2d5f50a9ec9f7c020) with the sequence that
// hw_swizzle_matches_donor replays. The donor is used only as an output oracle;
// no donor code is compiled into this library or these fixtures.

#include <algorithm>
#include "hw_memory.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace coregs::memory;

namespace {

int g_failures = 0;

#define HW_CHECK(condition, ...)                                                   \
    do {                                                                           \
        if (!(condition)) {                                                        \
            ++g_failures;                                                          \
            std::printf("  FAIL %s:%d: %s\n    ", __FILE__, __LINE__, #condition); \
            std::printf(__VA_ARGS__);                                              \
            std::printf("\n");                                                     \
        }                                                                          \
    } while (false)

// Same generator as the donor oracle run (splitmix64).
std::uint64_t nextRandom(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct DonorGolden {
    std::uint32_t psmCode;
    std::uint64_t memoryHash;
    std::uint64_t readHash;
};

// Output of the donor oracle: 20000 writes then 20000 reads per PSM, the
// generator seeded with 0x5EED0000 + PSM code. Z formats use base pointers
// multiple of 32 blocks.
constexpr DonorGolden kDonorGolden[] = {
    {0x00, 0x03807A622363CDEDull, 0x778091B341D85958ull},
    {0x01, 0x73C77B3D3B79C795ull, 0x941848A0A3EA6130ull},
    {0x02, 0x5B2F8086986BD3F5ull, 0x3873B07CAE1C120Eull},
    {0x0A, 0x4A1CE929E9CD4479ull, 0x765DDC2FC2389C13ull},
    {0x13, 0x1C137945267BF42Aull, 0xBF14F883A0C9334Full},
    {0x14, 0x946C11EF06767075ull, 0x7ECDF29E01241B5Eull},
    {0x1B, 0xFE73C74DAC088030ull, 0x781CF21131AF7DACull},
    {0x24, 0xC594507105A53985ull, 0x3B641A9C9A0C971Aull},
    {0x2C, 0x9D03145A42BA9D93ull, 0xCE7B8770B3600368ull},
    {0x30, 0x6D6389A5B80A07E0ull, 0x0804686C38C423D4ull},
    {0x31, 0xB63734B0110012C3ull, 0x242E659339B22BD6ull},
    {0x32, 0xEC41EB00E291F6AAull, 0xA7715D408AA55869ull},
    {0x3A, 0xF0BDBADBF9653A4Full, 0x6BAC8E7BDDB7AFF9ull},
};

std::uint64_t fnv1a(std::span<const std::uint8_t> bytes) {
    std::uint64_t hash = 1469598103934665603ull;
    for (std::uint8_t byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

const Psm kAllPsms[] = {Psm::Ct32, Psm::Ct24, Psm::Ct16, Psm::Ct16S, Psm::T8,   Psm::T4, Psm::T8H,
                        Psm::T4Hl, Psm::T4Hh, Psm::Z32,  Psm::Z24,   Psm::Z16,  Psm::Z16S};

// --- hw_swizzle_matches_donor ----------------------------------------------

void testSwizzleMatchesDonor() {
    for (const DonorGolden& golden : kDonorGolden) {
        const Psm psm = *decodePsm(golden.psmCode);
        const bool isDepth = psmInfo(psm).isDepth;
        LocalMemory memory;
        std::uint64_t state = 0x5EED0000ull + golden.psmCode;
        for (int index = 0; index < 20000; ++index) {
            std::uint32_t bp = static_cast<std::uint32_t>(nextRandom(state) % 16384);
            if (isDepth) bp &= ~31u;
            const std::uint32_t bw = 1 + static_cast<std::uint32_t>(nextRandom(state) % 32);
            const std::uint32_t x = static_cast<std::uint32_t>(nextRandom(state) % 2048);
            const std::uint32_t y = static_cast<std::uint32_t>(nextRandom(state) % 2048);
            const std::uint32_t value = static_cast<std::uint32_t>(nextRandom(state));
            memory.writePixel(BufferView{bp, bw, psm}, x, y, value);
        }
        HW_CHECK(fnv1a(memory.bytes()) == golden.memoryHash, "PSM 0x%02X memory image differs from donor", golden.psmCode);

        std::uint64_t readHash = 0;
        for (int index = 0; index < 20000; ++index) {
            std::uint32_t bp = static_cast<std::uint32_t>(nextRandom(state) % 16384);
            if (isDepth) bp &= ~31u;
            const std::uint32_t bw = 1 + static_cast<std::uint32_t>(nextRandom(state) % 32);
            const std::uint32_t x = static_cast<std::uint32_t>(nextRandom(state) % 2048);
            const std::uint32_t y = static_cast<std::uint32_t>(nextRandom(state) % 2048);
            readHash = readHash * 1000003ull + memory.readPixel(BufferView{bp, bw, psm}, x, y);
        }
        HW_CHECK(readHash == golden.readHash, "PSM 0x%02X read values differ from donor", golden.psmCode);
    }
}

// --- hw_roundtrip_psms -----------------------------------------------------

// Reads every pixel of a 4 MiB region laid out as 16 pages per row and 32 page
// rows, and writes it back. Every bit of memory is covered by some pixel for
// full-coverage formats, so the image must be unchanged.
void testRoundTripAllPsms() {
    for (Psm psm : kAllPsms) {
        const PsmInfo& info = psmInfo(psm);
        LocalMemory memory;
        std::uint64_t state = 0xABCD0000ull + static_cast<std::uint32_t>(psm);
        for (std::uint32_t address = 0; address < kLocalMemoryBytes; ++address) {
            memory.setByte(address, static_cast<std::uint8_t>(nextRandom(state)));
        }
        const std::vector<std::uint8_t> original(memory.bytes().begin(), memory.bytes().end());

        // bw such that pagesPerRow == 16 for this format.
        const BufferView view{0, 16u * info.pageWidth / 64u, psm};
        const std::uint32_t width = 16u * info.pageWidth;
        const std::uint32_t height = 32u * info.pageHeight;
        std::vector<std::uint32_t> values;
        values.reserve(static_cast<std::size_t>(width) * height);
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) values.push_back(memory.readPixel(view, x, y));
        }
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) memory.writePixel(view, x, y, values[static_cast<std::size_t>(y) * width + x]);
        }
        bool identical = true;
        for (std::uint32_t address = 0; address < kLocalMemoryBytes; ++address) {
            if (memory.byte(address) != original[address]) {
                identical = false;
                break;
            }
        }
        HW_CHECK(identical, "PSM 0x%02X round trip changed the memory image", static_cast<unsigned>(psm));
    }
}

// --- hw_psm_layout ---------------------------------------------------------

// Within one page no two pixels may share a bit, and full-coverage formats
// must fill the whole 8 KiB page.
void testPsmLayoutDisjoint() {
    for (Psm psm : kAllPsms) {
        const PsmInfo& info = psmInfo(psm);
        std::vector<std::uint8_t> owner(kPageBytes * 8u, 0);
        std::uint32_t coveredBits = 0;
        bool disjoint = true;
        for (std::uint32_t y = 0; y < info.pageHeight; ++y) {
            for (std::uint32_t x = 0; x < info.pageWidth; ++x) {
                const PixelSlot slot = pageLocalSlot(psm, x, y);
                HW_CHECK(slot.byteOffset < kPageBytes, "PSM 0x%02X pixel outside its page", static_cast<unsigned>(psm));
                for (std::uint32_t bit = 0; bit < slot.bitWidth; ++bit) {
                    const std::uint32_t address = slot.bitAddress() + bit;
                    if (owner[address]) disjoint = false;
                    owner[address] = 1;
                    ++coveredBits;
                }
            }
        }
        HW_CHECK(disjoint, "PSM 0x%02X pixels share bits inside a page", static_cast<unsigned>(psm));
        const bool partialWord = psm == Psm::Ct24 || psm == Psm::Z24 || psm == Psm::T8H ||
                                 psm == Psm::T4Hl || psm == Psm::T4Hh;
        if (!partialWord) {
            HW_CHECK(coveredBits == kPageBytes * 8u, "PSM 0x%02X does not cover its page (%u bits)",
                     static_cast<unsigned>(psm), coveredBits);
        }
    }
    // Table values (chapter 01.3, PSM table).
    HW_CHECK(psmInfo(Psm::T4).pageWidth == 128 && psmInfo(Psm::T4).pageHeight == 128, "T4 page size");
    HW_CHECK(psmInfo(Psm::T8).blockWidth == 16 && psmInfo(Psm::T8).blockHeight == 16, "T8 block size");
    HW_CHECK(psmInfo(Psm::T4).blockWidth == 32 && psmInfo(Psm::T4).blockHeight == 16, "T4 block size");
    HW_CHECK(psmInfo(Psm::Ct16).pageWidth == 64 && psmInfo(Psm::Ct16).pageHeight == 64, "CT16 page size");
    HW_CHECK(psmInfo(Psm::Ct24).transferBpp == 24, "CT24 transfer bpp");
    HW_CHECK(psmInfo(Psm::T4Hl).transferBpp == 4 && psmInfo(Psm::T8H).transferBpp == 8, "half-word transfer bpp");
    HW_CHECK(!decodePsm(0x12).has_value(), "code 0x12 must be invalid");
    HW_CHECK(!decodePsm(0x40).has_value(), "codes above 6 bits must be invalid");
    HW_CHECK(decodePsm(0x3A).has_value(), "Z16S must be valid");
}

// --- V-01-3: T8 view of a CT32 page ----------------------------------------

void testV013T8LaneVector() {
    LocalMemory memory;
    const BufferView ct32{0, 1, Psm::Ct32};
    for (std::uint32_t y = 0; y < 32; ++y) {
        for (std::uint32_t x = 0; x < 64; ++x) {
            const std::uint32_t red = x & 0xFFu;
            const std::uint32_t green = y & 0xFFu;
            const std::uint32_t blue = 0x80u | (x & 0x7Fu);
            const std::uint32_t alpha = 0xC0u | (y & 0x3Fu);
            memory.writePixel(ct32, x, y, red | (green << 8) | (blue << 16) | (alpha << 24));
        }
    }
    // TBW doubled: 128 texels per row inside the same 8 KiB page.
    const BufferView t8{0, 2, Psm::T8};
    int failures = 0;
    for (std::uint32_t v = 0; v < 64; ++v) {
        for (std::uint32_t u = 0; u < 128; ++u) {
            const PixelSlot slot = pixelSlot(t8, u, v);
            // Find the CT32 pixel that owns this byte (source pixel).
            std::uint32_t sourceX = 0, sourceY = 0, lane = 0;
            bool found = false;
            for (std::uint32_t y = 0; y < 32 && !found; ++y) {
                for (std::uint32_t x = 0; x < 64 && !found; ++x) {
                    const PixelSlot pixel = pixelSlot(ct32, x, y);
                    if (slot.byteOffset >= pixel.byteOffset && slot.byteOffset < pixel.byteOffset + 4) {
                        sourceX = x;
                        sourceY = y;
                        lane = slot.byteOffset - pixel.byteOffset;
                        found = true;
                    }
                }
            }
            const std::uint32_t expectedLane = 2u * ((u >> 3) & 1u) + ((v >> 1) & 1u);
            const std::uint32_t expectedValue = lane == 0   ? sourceX & 0xFFu
                                                : lane == 1 ? sourceY & 0xFFu
                                                : lane == 2 ? (0x80u | (sourceX & 0x7Fu))
                                                            : (0xC0u | (sourceY & 0x3Fu));
            const std::uint32_t texel = memory.readPixel(t8, u, v);
            if (!found || lane != expectedLane || texel != expectedValue) ++failures;
        }
    }
    HW_CHECK(failures == 0, "V-01-3: %d T8 texels have the wrong lane or source value", failures);
}

// --- V-01-4: Z texture at a non-page-aligned base ---------------------------

void testV014ZUnalignedBase() {
    // BP = 8 blocks (not a multiple of 32). Pixel (0, 16) of Z32 has in-page
    // block 8. After adding BP the block is 16 and XOR 24 gives 8. Applying the
    // XOR to the in-page entry first gives 8 + 16 = 24.
    const BufferView after{8, 1, Psm::Z32};
    const PixelSlot defaultSlot = pixelSlot(after, 0, 16);
    const PixelSlot afterSlot = pixelSlot(after, 0, 16, ZBlockXorRule::AfterAdd);
    const PixelSlot beforeSlot = pixelSlot(after, 0, 16, ZBlockXorRule::BeforeAdd);
    HW_CHECK(defaultSlot.byteOffset == afterSlot.byteOffset, "default Z rule must be AfterAdd");
    HW_CHECK(afterSlot.byteOffset / kBlockBytes == 8u, "V-01-4 AfterAdd block %u, expected 8",
             afterSlot.byteOffset / kBlockBytes);
    HW_CHECK(beforeSlot.byteOffset / kBlockBytes == 24u, "V-01-4 BeforeAdd block %u, expected 24",
             beforeSlot.byteOffset / kBlockBytes);

    // Page-aligned bases: both rules must agree for every pixel.
    bool agree = true;
    for (std::uint32_t y = 0; y < 32; ++y) {
        for (std::uint32_t x = 0; x < 64; ++x) {
            const BufferView aligned{64, 2, Psm::Z32};
            if (pixelSlot(aligned, x, y, ZBlockXorRule::AfterAdd).byteOffset !=
                pixelSlot(aligned, x, y, ZBlockXorRule::BeforeAdd).byteOffset) {
                agree = false;
            }
        }
    }
    HW_CHECK(agree, "Z rules must agree for page-aligned bases");
}

// --- hw_reinterp_maps ------------------------------------------------------

// Brute-force reference: scan every store pixel for the one containing the
// whole view texel.
void testReinterpMapsBruteForce() {
    const std::pair<Psm, Psm> pairs[] = {
        {Psm::T8, Psm::Ct32},   {Psm::T4, Psm::Ct32},   {Psm::Ct16, Psm::Ct32}, {Psm::T8, Psm::Ct16},
        {Psm::Ct32, Psm::Z32},  {Psm::T8H, Psm::Ct32}, {Psm::T4Hl, Psm::Ct32}, {Psm::Ct16S, Psm::Ct16},
        {Psm::Ct32, Psm::Ct16}, {Psm::Ct32, Psm::Ct24},
    };
    for (const auto& [view, store] : pairs) {
        const ReinterpMap map = buildReinterpMap(view, store);
        const PsmInfo& viewInfo = psmInfo(view);
        const PsmInfo& storeInfo = psmInfo(store);
        HW_CHECK(map.size() == static_cast<std::size_t>(viewInfo.pageWidth) * viewInfo.pageHeight,
                 "map size for view 0x%02X", static_cast<unsigned>(view));
        int mismatches = 0;
        for (std::uint32_t v = 0; v < viewInfo.pageHeight; ++v) {
            for (std::uint32_t u = 0; u < viewInfo.pageWidth; ++u) {
                const PixelSlot viewSlot = pageLocalSlot(view, u, v);
                const std::uint32_t viewBit = viewSlot.bitAddress();
                const std::uint32_t viewWidth = viewSlot.bitWidth;
                int containing = 0;
                std::uint32_t foundX = 0, foundY = 0, foundOffset = 0;
                for (std::uint32_t y = 0; y < storeInfo.pageHeight; ++y) {
                    for (std::uint32_t x = 0; x < storeInfo.pageWidth; ++x) {
                        const PixelSlot storeSlot = pageLocalSlot(store, x, y);
                        const std::uint32_t storeBit = storeSlot.bitAddress();
                        if (viewBit >= storeBit && viewBit + viewWidth <= storeBit + storeSlot.bitWidth) {
                            ++containing;
                            foundX = x;
                            foundY = y;
                            foundOffset = viewBit - storeBit;
                        }
                    }
                }
                const ReinterpTexel& entry = map[static_cast<std::size_t>(v) * viewInfo.pageWidth + u];
                const bool expectMapped = containing == 1;
                if (entry.mapped != expectMapped) {
                    ++mismatches;
                } else if (expectMapped && (entry.storeX != foundX || entry.storeY != foundY || entry.bitOffset != foundOffset)) {
                    ++mismatches;
                }
            }
        }
        HW_CHECK(mismatches == 0, "reinterp map 0x%02X over 0x%02X: %d texels differ from brute force",
                 static_cast<unsigned>(view), static_cast<unsigned>(store), mismatches);
    }
    // Spot checks from chapter 09.1: T8 over CT32 selects R, G, B, A by lane.
    const ReinterpMap t8OverCt32 = buildReinterpMap(Psm::T8, Psm::Ct32);
    const ReinterpTexel& laneB = t8OverCt32[static_cast<std::size_t>(0) * 128 + 8];
    HW_CHECK(laneB.mapped && laneB.bitOffset == 16, "T8 texel (8,0) must be the B byte of CT32");
    const ReinterpTexel& laneA = t8OverCt32[static_cast<std::size_t>(2) * 128 + 8];
    HW_CHECK(laneA.mapped && laneA.bitOffset == 24, "T8 texel (8,2) must be the A byte of CT32");
    const ReinterpMap ct32OverCt16Map = buildReinterpMap(Psm::Ct32, Psm::Ct16);
    const ReinterpTexel& ct32OverCt16 = ct32OverCt16Map[0];
    HW_CHECK(!ct32OverCt16.mapped, "a 32-bit view over 16-bit storage crosses pixels and is unmapped");
}

// --- format predicate tables -----------------------------------------------

void testFormatPredicates() {
    // Bit sharing (chapter 01.3).
    HW_CHECK(!formatsShareBits(Psm::Ct24, Psm::T8H), "CT24 and T8H must not share bits");
    HW_CHECK(!formatsShareBits(Psm::Z24, Psm::T4Hh), "Z24 and T4HH must not share bits");
    HW_CHECK(!formatsShareBits(Psm::T4Hl, Psm::T4Hh), "T4HL and T4HH must not share bits");
    HW_CHECK(formatsShareBits(Psm::Ct32, Psm::T8H), "CT32 and T8H share bits");
    HW_CHECK(formatsShareBits(Psm::Ct24, Psm::Ct16), "CT24 and CT16 share bits");
    HW_CHECK(formatsShareBits(Psm::T8, Psm::T4), "T8 and T4 share bits");
    HW_CHECK(formatsShareBits(Psm::Ct32, Psm::Ct32), "CT32 shares bits with itself");
    // Compatible depth class.
    HW_CHECK(formatsCompatibleDepth(Psm::Ct32, Psm::Z24), "32 and 24 are compatible");
    HW_CHECK(formatsCompatibleDepth(Psm::Ct16, Psm::Z16S), "16 bit formats are compatible");
    HW_CHECK(!formatsCompatibleDepth(Psm::Ct16, Psm::Ct32), "16 and 32 are not compatible");
    HW_CHECK(!formatsCompatibleDepth(Psm::T8, Psm::T8H), "8 bit formats only with themselves");
    HW_CHECK(formatsCompatibleDepth(Psm::T8, Psm::T8), "a format is compatible with itself");
    // Same swizzle.
    HW_CHECK(formatsSameSwizzle(Psm::Ct32, Psm::T8H), "CT32 and T8H share a swizzle");
    HW_CHECK(formatsSameSwizzle(Psm::Z32, Psm::Z24), "Z32 and Z24 share a swizzle");
    HW_CHECK(!formatsSameSwizzle(Psm::Ct16, Psm::Ct16S), "CT16 and CT16S differ");
    HW_CHECK(!formatsSameSwizzle(Psm::T8, Psm::T4), "T8 and T4 differ");
    HW_CHECK(!formatsSameSwizzle(Psm::Z16, Psm::Ct16), "Z16 differs from CT16");
    // Invalid code 0x12 never relates to anything.
    bool invalidRowClear = true;
    for (std::uint32_t code = 0; code < 64; ++code) {
        if (sharedBitsTable()[0x12][code] || sameSwizzleTable()[0x12][code] || compatibleDepthTable()[0x12][code]) {
            invalidRowClear = false;
        }
    }
    HW_CHECK(invalidRowClear, "code 0x12 must be false in every table");
    // Tables are symmetric.
    bool symmetric = true;
    for (std::uint32_t a = 0; a < 64; ++a) {
        for (std::uint32_t b = 0; b < 64; ++b) {
            if (sharedBitsTable()[a][b] != sharedBitsTable()[b][a]) symmetric = false;
            if (sameSwizzleTable()[a][b] != sameSwizzleTable()[b][a]) symmetric = false;
            if (compatibleDepthTable()[a][b] != compatibleDepthTable()[b][a]) symmetric = false;
        }
    }
    HW_CHECK(symmetric, "predicate tables must be symmetric");
    // Channel and valid masks.
    HW_CHECK(psmInfo(Psm::Ct24).channelMask == 0x7 && psmInfo(Psm::Z24).channelMask == 0x7, "24-bit formats are RGB");
    HW_CHECK(psmInfo(Psm::T8H).channelMask == 0x8 && psmInfo(Psm::T4Hl).channelMask == 0x8, "half formats are A only");
    HW_CHECK(validBitMask(Psm::T8H) == 0xFF000000u && validBitMask(Psm::T4Hh) == 0xF0000000u, "half-format valid bits");
    HW_CHECK(validBitMask(Psm::Ct24) == 0x00FFFFFFu, "CT24 valid bits");
}

// --- hw_transfers ----------------------------------------------------------

void testTransfers() {
    // Odd-width T4 upload: nibbles continue across rows (chapter 02.7).
    {
        LocalMemory memory;
        const BufferView t4{0, 2, Psm::T4};
        const std::uint8_t data[] = {0x21, 0x43, 0x65};
        const std::uint32_t written = hostToLocal(memory, t4, TransferRect{0, 0, 3, 2}, data);
        HW_CHECK(written == 6, "odd-width T4 wrote %u pixels", written);
        HW_CHECK(hostBytesFor(Psm::T4, 3, 2) == 3, "odd-width T4 host size");
        const std::uint32_t expected[6] = {1, 2, 3, 4, 5, 6};
        for (std::uint32_t index = 0; index < 6; ++index) {
            const std::uint32_t x = index % 3, y = index / 3;
            HW_CHECK(memory.readPixel(t4, x, y) == expected[index], "odd-width T4 texel %u", index);
        }
    }
    // 24-bit packed host data and preservation of the top byte.
    {
        LocalMemory memory;
        const BufferView ct24{0, 1, Psm::Ct24};
        memory.setByte(3, 0xAB);  // bits 24..31 of the first word, outside CT24
        const std::uint8_t data[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
        hostToLocal(memory, ct24, TransferRect{0, 0, 2, 1}, data);
        HW_CHECK(memory.readPixel(ct24, 0, 0) == 0x00332211u, "CT24 pixel 0 from packed host data");
        HW_CHECK(memory.readPixel(ct24, 1, 0) == 0x00665544u, "CT24 pixel 1 from packed host data");
        HW_CHECK(memory.byte(0) == 0x11 && memory.byte(3) == 0xAB, "CT24 must keep bits 24..31");
    }
    // T8H writes only bits 24..31.
    {
        LocalMemory memory;
        const BufferView ct32{0, 1, Psm::Ct32};
        memory.writePixel(ct32, 0, 0, 0x00FFFFFFu);
        const std::uint8_t data[] = {0x7F};
        hostToLocal(memory, BufferView{0, 1, Psm::T8H}, TransferRect{0, 0, 1, 1}, data);
        HW_CHECK(memory.readPixel(ct32, 0, 0) == 0x7FFFFFFFu, "T8H must touch only bits 24..31");
    }
    // 16-bit host upload.
    {
        LocalMemory memory;
        const BufferView ct16{0, 1, Psm::Ct16};
        const std::uint8_t data[] = {0x34, 0x12, 0x78, 0x56, 0xBC, 0x9A, 0xF0, 0xDE};
        hostToLocal(memory, ct16, TransferRect{5, 3, 2, 2}, data);
        HW_CHECK(memory.readPixel(ct16, 5, 3) == 0x1234u && memory.readPixel(ct16, 6, 3) == 0x5678u, "CT16 row 0");
        HW_CHECK(memory.readPixel(ct16, 5, 4) == 0x9ABCu && memory.readPixel(ct16, 6, 4) == 0xDEF0u, "CT16 row 1");
    }
    // X wrap at 2048 and Y wrap at 2048 (chapter 02.7 and 01.3).
    {
        LocalMemory memory;
        const BufferView wide{0, 32, Psm::Ct32};  // 2048 pixels per row
        const std::uint8_t data[] = {1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0};
        hostToLocal(memory, wide, TransferRect{2046, 0, 4, 1}, data);
        HW_CHECK(memory.readPixel(wide, 2046, 0) == 1, "X wrap: 2046");
        HW_CHECK(memory.readPixel(wide, 2047, 0) == 2, "X wrap: 2047");
        HW_CHECK(memory.readPixel(wide, 0, 0) == 3, "X wrap: 0");
        HW_CHECK(memory.readPixel(wide, 1, 0) == 4, "X wrap: 1");

        LocalMemory memoryY;
        hostToLocal(memoryY, wide, TransferRect{0, 2047, 1, 2}, std::span<const std::uint8_t>(data, 8));
        HW_CHECK(memoryY.readPixel(wide, 0, 2047) == 1, "Y wrap: row 2047");
        HW_CHECK(memoryY.readPixel(wide, 0, 0) == 2, "Y wrap: row 0");
    }
    // Local->local: UL->LR smears an overlapping move, UR->LL does not.
    {
        const BufferView ct32{0, 32, Psm::Ct32};
        auto prepare = [&](LocalMemory& memory) {
            for (std::uint32_t x = 0; x < 10; ++x) memory.writePixel(ct32, x, 0, x + 1);
        };
        LocalMemory smear;
        prepare(smear);
        localToLocal(smear, ct32, 0, 0, ct32, 2, 0, 8, 1, TransferDirection::UpperLeftToLowerRight);
        const std::uint32_t smearExpected[8] = {1, 2, 1, 2, 1, 2, 1, 2};
        for (std::uint32_t index = 0; index < 8; ++index) {
            HW_CHECK(smear.readPixel(ct32, 2 + index, 0) == smearExpected[index], "DIR0 smear texel %u", index);
        }
        LocalMemory shift;
        prepare(shift);
        localToLocal(shift, ct32, 0, 0, ct32, 2, 0, 8, 1, TransferDirection::UpperRightToLowerLeft);
        for (std::uint32_t index = 0; index < 8; ++index) {
            HW_CHECK(shift.readPixel(ct32, 2 + index, 0) == index + 1, "DIR2 shift texel %u", index);
        }
        // Vertical move downward with rows visited bottom to top (DIR1).
        LocalMemory column;
        for (std::uint32_t y = 0; y < 10; ++y) column.writePixel(ct32, 0, y, y + 1);
        localToLocal(column, ct32, 0, 0, ct32, 0, 2, 1, 8, TransferDirection::LowerLeftToUpperRight);
        for (std::uint32_t index = 0; index < 8; ++index) {
            HW_CHECK(column.readPixel(ct32, 0, 2 + index) == index + 1, "DIR1 column texel %u", index);
        }
    }
    // Local->host: packing and the illegal 4-bit source.
    {
        LocalMemory memory;
        const BufferView ct24{0, 1, Psm::Ct24};
        memory.writePixel(ct24, 0, 0, 0x332211u);
        memory.writePixel(ct24, 1, 0, 0x665544u);
        std::vector<std::uint8_t> host;
        const TransferStatus status = localToHost(memory, ct24, TransferRect{0, 0, 2, 1}, host);
        HW_CHECK(status == TransferStatus::Ok, "CT24 LOCAL->HOST status");
        HW_CHECK(host.size() == 6 && host[0] == 0x11 && host[2] == 0x33 && host[3] == 0x44 && host[5] == 0x66,
                 "CT24 LOCAL->HOST packing");
        HW_CHECK(localToHost(memory, BufferView{0, 1, Psm::T4}, TransferRect{0, 0, 2, 1}, host) ==
                     TransferStatus::IllegalFormat,
                 "4-bit LOCAL->HOST must be illegal");
    }
    // Readback qword sizes (chapter 10.2).
    HW_CHECK(readbackQwords(0x00, 64, 32) == std::optional<std::uint64_t>(512), "CT32 readback qwords");
    HW_CHECK(readbackQwords(0x01, 3, 1) == std::optional<std::uint64_t>(1), "CT24 readback rounds up");
    HW_CHECK(readbackQwords(0x13, 16, 16) == std::optional<std::uint64_t>(16), "T8 readback selects 8 bpp");
    HW_CHECK(readbackQwords(0x14, 16, 16) == std::nullopt, "T4 readback is illegal");
    HW_CHECK(readbackQwords(0x05, 16, 16) == std::optional<std::uint64_t>(64), "unknown selector counts as 32 bpp");
}

// readRect/writeRect (M7 performance, IN-343) must address exactly like readPixel/writePixel: for every PSM,
// several buffer widths and unaligned bases (Z formats included), a rectangle read with readRect equals the
// per-pixel reads, and writing it with writeRect leaves memory byte-identical to per-pixel writes.
void testRectMatchesPixels() {
    const std::uint32_t bases[] = {0u, 5u, 37u, 16000u};
    const std::uint32_t widths[] = {1u, 3u, 10u};
    for (Psm psm : kAllPsms) {
        for (std::uint32_t base : bases) {
            for (std::uint32_t bw : widths) {
                LocalMemory memory;
                std::uint64_t state = 0x5EED0000ull + static_cast<std::uint32_t>(psm) * 131u + base * 7u + bw;
                for (std::uint32_t address = 0; address < kLocalMemoryBytes; address += 4) {
                    const std::uint32_t word = static_cast<std::uint32_t>(nextRandom(state));
                    for (std::uint32_t b = 0; b < 4; ++b) memory.setByte(address + b, static_cast<std::uint8_t>(word >> (8 * b)));
                }
                const BufferView view{base, bw, psm};
                const std::uint32_t x0 = 3, y0 = 5, width = 150, height = 70;
                std::vector<std::uint32_t> rect(static_cast<std::size_t>(width) * height);
                memory.readRect(view, x0, y0, width, height, rect.data());
                std::size_t mismatches = 0;
                for (std::uint32_t y = 0; y < height; ++y) {
                    for (std::uint32_t x = 0; x < width; ++x) {
                        if (rect[static_cast<std::size_t>(y) * width + x] != memory.readPixel(view, x0 + x, y0 + y)) ++mismatches;
                    }
                }
                HW_CHECK(mismatches == 0, "readRect equals readPixel");
                std::vector<std::uint32_t> values(rect.size());
                for (std::uint32_t& v : values) v = static_cast<std::uint32_t>(nextRandom(state));
                LocalMemory byPixel = memory;
                LocalMemory byRect = memory;
                for (std::uint32_t y = 0; y < height; ++y) {
                    for (std::uint32_t x = 0; x < width; ++x) {
                        byPixel.writePixel(view, x0 + x, y0 + y, values[static_cast<std::size_t>(y) * width + x]);
                    }
                }
                byRect.writeRect(view, x0, y0, width, height, values.data());
                HW_CHECK(std::equal(byPixel.bytes().begin(), byPixel.bytes().end(), byRect.bytes().begin()),
                         "writeRect equals writePixel");
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    struct Test {
        const char* name;
        void (*run)();
    };
    const Test tests[] = {
        {"hw_swizzle_matches_donor", testSwizzleMatchesDonor},
        {"hw_roundtrip_psms", testRoundTripAllPsms},
        {"hw_psm_layout", testPsmLayoutDisjoint},
        {"hw_v01_3_t8_lanes", testV013T8LaneVector},
        {"hw_v01_4_z_unaligned", testV014ZUnalignedBase},
        {"hw_reinterp_maps", testReinterpMapsBruteForce},
        {"hw_format_predicates", testFormatPredicates},
        {"hw_transfers", testTransfers},
        {"hw_rect_matches_pixels", testRectMatchesPixels},
    };
    const std::string requested = argc > 1 ? argv[1] : "";
    int ran = 0;
    for (const Test& test : tests) {
        if (!requested.empty() && requested != test.name) continue;
        const int before = g_failures;
        std::printf("[ RUN  ] %s\n", test.name);
        test.run();
        std::printf("[ %s ] %s\n", g_failures == before ? " OK " : "FAIL", test.name);
        ++ran;
    }
    if (ran == 0) {
        std::printf("unknown test: %s\n", requested.c_str());
        return 2;
    }
    return g_failures == 0 ? 0 : 1;
}
