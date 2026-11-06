// SPDX-License-Identifier: MIT
// Copyright 2026 GT4Recomp contributors
//
// HWR local memory model (TASK-027, milestone M2).
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters
// 01.3 (local memory), 02.7 (transfers), 09.1 (reinterpretation maps) and 10.2
// (LOCAL->HOST). Clean room: no code from other GS renderers.
//
// This header is Vulkan-free and has no dependencies outside the C++ standard
// library. It models the shadow local memory (SLM): a 4 MiB byte image of the
// GS memory with the GS swizzles applied on access.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace coregs::memory {

// ---------------------------------------------------------------------------
// Units (chapter 01.3)
// ---------------------------------------------------------------------------

inline constexpr std::uint32_t kLocalMemoryBytes = 4u << 20;  // 4 MiB
inline constexpr std::uint32_t kPageBytes = 8192;             // 8 KiB
inline constexpr std::uint32_t kBlockBytes = 256;             // 256 B
inline constexpr std::uint32_t kColumnBytes = 64;             // 64 B
inline constexpr std::uint32_t kBlocksPerPage = 32;
inline constexpr std::uint32_t kPagesTotal = 512;
inline constexpr std::uint32_t kBlocksTotal = 16384;          // modulo for block addresses
inline constexpr std::uint32_t kTransferCoordinateWrap = 2048;  // transfer X and Y wrap
inline constexpr std::uint32_t kPsmCodeMask = 0x3F;           // PSM is a 6-bit field

// Block address is in blocks (BP, TBP0, CBP, ...). Page address is in pages.
inline constexpr std::uint32_t blockOfPage(std::uint32_t pageAddress) {
    return pageAddress * kBlocksPerPage;
}

// ---------------------------------------------------------------------------
// Pixel storage formats (chapter 01.3, PSM table)
// ---------------------------------------------------------------------------

enum class Psm : std::uint8_t {
    Ct32 = 0x00,
    Ct24 = 0x01,
    Ct16 = 0x02,
    Ct16S = 0x0A,
    T8 = 0x13,
    T4 = 0x14,
    T8H = 0x1B,
    T4Hl = 0x24,
    T4Hh = 0x2C,
    Z32 = 0x30,
    Z24 = 0x31,
    Z16 = 0x32,
    Z16S = 0x3A,
};

// The in-page block arrangement family. Z formats reuse their colour family
// table with every block entry XORed with 24 (in-page or after adding BP, see
// ZBlockXorRule).
enum class SwizzleFamily : std::uint8_t {
    Word32,   // CT32, CT24, T8H, T4HL, T4HH: 64x32 page, 8x8 blocks, 4-column layout
    Word16,   // CT16: 64x64 page, 16x8 blocks
    Word16S,  // CT16S: 64x64 page, 16x8 blocks, alternate block table
    Byte8,    // T8: 128x64 page, 16x16 blocks
    Nibble4,  // T4: 128x128 page, 32x16 blocks
};

// Storage description of one PSM. Bit masks use bit 0 = R, 1 = G, 2 = B, 3 = A.
struct PsmInfo {
    Psm psm;
    const char* name;
    std::uint8_t storageBits;      // bits in the storage unit: 32, 16, 8 or 4
    std::uint8_t transferBpp;      // bits per texel in host data: 32, 24, 16, 8 or 4
    std::uint16_t pageWidth;       // pixels per page
    std::uint16_t pageHeight;
    std::uint8_t blockWidth;       // pixels per block
    std::uint8_t blockHeight;
    bool isDepth;                  // Z formats
    bool isIndexed;                // PSM & 7 >= 3 (palette formats)
    std::uint32_t validBitMask32;  // valid bits in a 32-bit view (0 for sub-word formats)
    std::uint8_t channelMask;      // RGBA channels stored
    SwizzleFamily family;
    bool zBlockXorApplies;         // Z formats: block numbers are XORed with 24
};

// GPU form of the pixel addressing (M10 GLM, IN-347): the in-page location of every pixel of a page for every
// PSM, packed one uint32 per pixel (bits 0-4 block number in the page, 5-12 byte in the block, 13-17 bit shift,
// 18-23 bit width), plus a 64-entry directory indexed by the PSM code: offset of the PSM's table in `entries`,
// page width, page height and the Z block XOR flag (4 uint32 each). Same numbers as pixelSlot.
struct GpuAddressTables {
    std::vector<std::uint32_t> directory;  // 64 x {offset, pageWidth, pageHeight, zXor}
    std::vector<std::uint32_t> entries;
};
GpuAddressTables gpuAddressTables();

// Returns the PSM code if it is one of the documented formats (0x12 and other
// codes are not valid for HWR).
std::optional<Psm> decodePsm(std::uint32_t code);
bool isValidPsmCode(std::uint32_t code);
const PsmInfo& psmInfo(Psm psm);

// ---------------------------------------------------------------------------
// Pixel addressing (chapter 01.3, address computation)
// ---------------------------------------------------------------------------

// How the Z-format block XOR is applied (chapter 01.3 step 4).
// AfterAdd is the rule HWR follows (Hypothesis, verified by V-01-4).
// BeforeAdd applies the XOR to the in-page table entry, as the hardware does; the
// two agree whenever the base is a multiple of 32 blocks.
enum class ZBlockXorRule : std::uint8_t {
    AfterAdd,
    BeforeAdd,
};

// Location of one pixel's storage unit inside the 4 MiB local memory.
struct PixelSlot {
    std::uint32_t byteOffset;  // 0 .. 4 MiB - 1 (already wrapped)
    std::uint8_t bitShift;     // position of the pixel LSB inside the unit at byteOffset
    std::uint8_t bitWidth;     // bits of the pixel (4, 8, 16, 24 or 32)

    // Bytes that must be read or written to cover the pixel bits.
    std::uint8_t bytesCovered() const;
    // Absolute bit address of the pixel LSB.
    std::uint32_t bitAddress() const { return byteOffset * 8u + bitShift; }
};

// Buffer description used by addressing and transfers. Addresses follow the
// GS register units: bp in blocks, bw in 64-pixel units.
struct BufferView {
    std::uint32_t basePointer;   // BP, blocks
    std::uint32_t bufferWidth;   // BW, 64-pixel units
    Psm psm;
};

// Computes the slot of pixel (x, y) of a buffer. x and y are used as given
// (callers wrap transfer coordinates at 2048 first).
PixelSlot pixelSlot(const BufferView& buffer,
                    std::uint32_t x,
                    std::uint32_t y,
                    ZBlockXorRule zRule = ZBlockXorRule::AfterAdd);

// Pages per row for a buffer (chapter 01.3 step 1). Clamped to at least 1, as
// the hardware does; the specification does not define the zero case (open question).
std::uint32_t pagesPerRow(std::uint32_t bufferWidth, Psm psm);

// Slot of a pixel of a single page placed at base 0 (in-page coordinates).
PixelSlot pageLocalSlot(Psm psm, std::uint32_t x, std::uint32_t y);

// ---------------------------------------------------------------------------
// Local memory
// ---------------------------------------------------------------------------

class LocalMemory {
public:
    LocalMemory();

    std::uint8_t byte(std::uint32_t address) const;
    void setByte(std::uint32_t address, std::uint8_t value);

    // Raw access to the whole 4 MiB image (for dumps and round-trip tests).
    std::span<const std::uint8_t> bytes() const { return bytes_; }
    std::span<std::uint8_t> bytes() { return bytes_; }

    // Reads and writes one pixel slot. Writes touch only the slot's bits and
    // leave every other bit of the covered units unchanged.
    std::uint32_t readSlot(const PixelSlot& slot) const;
    void writeSlot(const PixelSlot& slot, std::uint32_t value);

    std::uint32_t readPixel(const BufferView& buffer, std::uint32_t x, std::uint32_t y) const;
    void writePixel(const BufferView& buffer, std::uint32_t x, std::uint32_t y, std::uint32_t value);

    // Rectangle forms of readPixel/writePixel (same addressing, the AfterAdd Z rule): `width` x `height` pixels
    // from (x0, y0), row-major in `values`. Coordinates must be below 2048 so no wrap is involved; the caller
    // falls back to readPixel otherwise. They walk the page table once per row instead of once per pixel.
    void readRect(const BufferView& buffer, std::uint32_t x0, std::uint32_t y0, std::uint32_t width,
                  std::uint32_t height, std::uint32_t* values) const;
    void writeRect(const BufferView& buffer, std::uint32_t x0, std::uint32_t y0, std::uint32_t width,
                   std::uint32_t height, const std::uint32_t* values);

private:
    std::vector<std::uint8_t> bytes_;
};

// ---------------------------------------------------------------------------
// Transfers (chapter 02.7 and 10.2)
// ---------------------------------------------------------------------------

// TRXPOS.DIR pixel order for local->local transfers (chapter 01.1).
enum class TransferDirection : std::uint8_t {
    UpperLeftToLowerRight = 0,  // rows top to bottom, columns left to right
    LowerLeftToUpperRight = 1,  // rows bottom to top, columns left to right
    UpperRightToLowerLeft = 2,  // rows top to bottom, columns right to left
    LowerRightToUpperLeft = 3,  // rows bottom to top, columns right to left
};

struct TransferRect {
    std::uint32_t x;       // SSAX/DSAX, wrapped at 2048 per pixel
    std::uint32_t y;       // SSAY/DSAY, wrapped at 2048 per pixel
    std::uint32_t width;   // RRW
    std::uint32_t height;  // RRH
};

enum class TransferStatus : std::uint8_t {
    Ok,
    IllegalFormat,  // 4-bit source for LOCAL->HOST (chapter 10.2)
};

// Host data layout for one PSM (chapter 02.7): 32-bit and 16-bit little
// endian, 24-bit packed as three bytes per texel, 8-bit one byte, 4-bit a
// continuous nibble stream (low nibble first) across rows.
// Returns the number of bytes needed for width x height texels (rounded up).
std::size_t hostBytesFor(Psm psm, std::uint32_t width, std::uint32_t height);

// HOST->LOCAL: converts host data in row-major order into the SLM. Pixels with
// no data left in `data` are not written. Returns the number of pixels written.
std::uint32_t hostToLocal(LocalMemory& memory,
                          const BufferView& destination,
                          const TransferRect& rect,
                          std::span<const std::uint8_t> data);

// LOCAL->LOCAL inside the SLM. Pixels are visited in the TRXPOS.DIR order and
// each pixel is read and then written immediately, so overlapping moves see
// their own earlier writes (chapter 05.7). Raw storage bits are copied and
// truncated by the destination's storage width.
void localToLocal(LocalMemory& memory,
                  const BufferView& source,
                  std::uint32_t sourceX,
                  std::uint32_t sourceY,
                  const BufferView& destination,
                  std::uint32_t destinationX,
                  std::uint32_t destinationY,
                  std::uint32_t width,
                  std::uint32_t height,
                  TransferDirection direction);

// LOCAL->HOST from the SLM, packed as HOST->LOCAL. Refuses 4-bit sources.
TransferStatus localToHost(const LocalMemory& memory,
                           const BufferView& source,
                           const TransferRect& rect,
                           std::vector<std::uint8_t>& hostData);

// Qwords the EE expects for a LOCAL->HOST read (chapter 10.2 step 1).
// bpp comes from PSM & 7: 0 -> 32, 1 -> 24, 2 -> 16, 3 -> 8. 4 (any 4-bit
// format) is illegal and yields nullopt. Other values count as 32. The result
// is rounded up (rounding is not stated in the spec; open question).
std::optional<std::uint64_t> readbackQwords(std::uint32_t psmCode,
                                            std::uint32_t width,
                                            std::uint32_t height);

// ---------------------------------------------------------------------------
// Reinterpretation maps (chapter 09.1)
// ---------------------------------------------------------------------------

// Entry for one view texel inside one page of the view format. The texel's
// bits lie in the store pixel (x, y) at bit offset bitOffset, when mapped.
struct ReinterpTexel {
    std::uint16_t storeX;
    std::uint16_t storeY;
    std::uint8_t bitOffset;  // LSB of the view texel inside the store pixel value
    std::uint8_t bitWidth;   // view texel width
    bool mapped;             // false when the texel crosses store pixels or store bits
};

// Index = viewY * viewPageWidth + viewX, covering one page of the view format.
using ReinterpMap = std::vector<ReinterpTexel>;

ReinterpMap buildReinterpMap(Psm view, Psm store);

// ---------------------------------------------------------------------------
// Format predicate tables (chapter 01.3, format relationship predicates)
// ---------------------------------------------------------------------------

// Two formats can disturb each other's bits in the same 32-bit word.
bool formatsShareBits(Psm first, Psm second);
// Formats in the same depth class: {32, 24} with each other, {16} with 16,
// and 8 or 4 bit formats only with themselves (Hypothesis on reading; see open
// questions in the journal).
bool formatsCompatibleDepth(Psm first, Psm second);
// Formats with one in-page layout.
bool formatsSameSwizzle(Psm first, Psm second);

// 64x64 tables indexed by (PSM code & 0x3F); rows and columns for codes that
// are not valid are all false.
using FormatPredicateTable = std::array<std::array<bool, 64>, 64>;
const FormatPredicateTable& sharedBitsTable();
const FormatPredicateTable& compatibleDepthTable();
const FormatPredicateTable& sameSwizzleTable();

// Valid-bit mask (32-bit view) for write-back channel masks (chapter 10.4).
std::uint32_t validBitMask(Psm psm);

}  // namespace coregs::memory
