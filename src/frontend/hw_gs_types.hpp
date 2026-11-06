// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// CoreGS shared types and bit helpers.
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
//
// Vocabulary follows the specification (chapters 01 and 02). "Qword" means a
// 128-bit GIF quadword (16 bytes), as in the specification's PACKED tables.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace coregs {

// One 128-bit GIF quadword.
struct Qword {
    uint64_t lo = 0;
    uint64_t hi = 0;
};

// Field extraction on a 64-bit value: bits [lo..lo+width-1].
constexpr uint64_t Field(uint64_t value, unsigned lo, unsigned width) {
    if (width >= 64) {
        return value >> lo;
    }
    return (value >> lo) & ((uint64_t{1} << width) - 1u);
}

// Field by inclusive bit range [lo..hi], as written in the specification.
constexpr uint64_t BitRange(uint64_t value, unsigned lo, unsigned hi) {
    return Field(value, lo, hi - lo + 1u);
}

constexpr uint64_t MaskBits(unsigned lo, unsigned hi) {
    const unsigned width = hi - lo + 1u;
    const uint64_t body = width >= 64 ? ~uint64_t{0} : ((uint64_t{1} << width) - 1u);
    return body << lo;
}

inline float BitsToFloat(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline uint32_t FloatToBits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Pixel storage formats (PSM codes, chapter 01.3).
namespace psm {
constexpr uint32_t kCt32 = 0x00;
constexpr uint32_t kCt24 = 0x01;
constexpr uint32_t kCt16 = 0x02;
constexpr uint32_t kCt16S = 0x0A;
constexpr uint32_t kT8 = 0x13;
constexpr uint32_t kT4 = 0x14;
constexpr uint32_t kT8H = 0x1B;
constexpr uint32_t kT4Hl = 0x24;
constexpr uint32_t kT4Hh = 0x2C;
constexpr uint32_t kZ32 = 0x30;
constexpr uint32_t kZ24 = 0x31;
constexpr uint32_t kZ16 = 0x32;
constexpr uint32_t kZ16S = 0x3A;
constexpr uint32_t kInvalid12 = 0x12;
}  // namespace psm

// Indexed formats: (PSM & 7) >= 3.
inline bool IsIndexedPsm(uint32_t format) { return (format & 7u) >= 3u; }

// Z formats: bits 4..5 equal 3.
inline bool IsZPsm(uint32_t format) { return ((format >> 4) & 3u) == 3u; }

// Transfer bits per pixel (chapter 02.7), or 0 for an invalid format.
inline uint32_t TransferBitsPerPixel(uint32_t format) {
    switch (format) {
        case psm::kCt32: case psm::kZ32: return 32;
        case psm::kCt24: case psm::kZ24: return 24;
        case psm::kCt16: case psm::kCt16S: case psm::kZ16: case psm::kZ16S: return 16;
        case psm::kT8: case psm::kT8H: return 8;
        case psm::kT4: case psm::kT4Hl: case psm::kT4Hh: return 4;
        default: return 0;
    }
}

// Channel set as an RGBA bit mask (bit 0 R, 1 G, 2 B, 3 A), chapter 01.3.
inline uint32_t PsmChannelMask(uint32_t format) {
    switch (format) {
        case psm::kCt24: case psm::kZ24: return 0x7u;
        case psm::kT8H: case psm::kT4Hl: case psm::kT4Hh: return 0x8u;
        default: return 0xFu;
    }
}

// Co-resident pairs (chapter 01.3, "Formats sharing words"): formats that can
// live in the same 32-bit words without disturbing each other's bits. Every
// other pair of formats shares bits. The pairs are {CT24 or Z24} with
// {T8H, T4HL, T4HH}, and T4HL with T4HH.
inline bool PsmCoResident(uint32_t a, uint32_t b) {
    auto is24 = [](uint32_t f) { return f == psm::kCt24 || f == psm::kZ24; };
    auto isHigh = [](uint32_t f) {
        return f == psm::kT8H || f == psm::kT4Hl || f == psm::kT4Hh;
    };
    if ((is24(a) && isHigh(b)) || (is24(b) && isHigh(a))) {
        return true;
    }
    if ((a == psm::kT4Hl && b == psm::kT4Hh) || (a == psm::kT4Hh && b == psm::kT4Hl)) {
        return true;
    }
    return false;
}

// Register addresses (chapter 01.1).
namespace reg {
constexpr uint8_t kPrim = 0x00;
constexpr uint8_t kRgbaq = 0x01;
constexpr uint8_t kSt = 0x02;
constexpr uint8_t kUv = 0x03;
constexpr uint8_t kXyzf2 = 0x04;
constexpr uint8_t kXyz2 = 0x05;
constexpr uint8_t kTex0One = 0x06;
constexpr uint8_t kTex0Two = 0x07;
constexpr uint8_t kClampOne = 0x08;
constexpr uint8_t kClampTwo = 0x09;
constexpr uint8_t kFog = 0x0A;
constexpr uint8_t kXyzf3 = 0x0C;
constexpr uint8_t kXyz3 = 0x0D;
constexpr uint8_t kAdPacked = 0x0E;  // A+D in PACKED mode, no-op in REGLIST
constexpr uint8_t kTex1One = 0x14;
constexpr uint8_t kTex1Two = 0x15;
constexpr uint8_t kTex2One = 0x16;
constexpr uint8_t kTex2Two = 0x17;
constexpr uint8_t kXyoffsetOne = 0x18;
constexpr uint8_t kXyoffsetTwo = 0x19;
constexpr uint8_t kPrmodecont = 0x1A;
constexpr uint8_t kPrmode = 0x1B;
constexpr uint8_t kTexclut = 0x1C;
constexpr uint8_t kScanmsk = 0x22;
constexpr uint8_t kMiptbp1One = 0x34;
constexpr uint8_t kMiptbp1Two = 0x35;
constexpr uint8_t kMiptbp2One = 0x36;
constexpr uint8_t kMiptbp2Two = 0x37;
constexpr uint8_t kTexa = 0x3B;
constexpr uint8_t kFogcol = 0x3D;
constexpr uint8_t kTexflush = 0x3F;
constexpr uint8_t kScissorOne = 0x40;
constexpr uint8_t kScissorTwo = 0x41;
constexpr uint8_t kAlphaOne = 0x42;
constexpr uint8_t kAlphaTwo = 0x43;
constexpr uint8_t kDimx = 0x44;
constexpr uint8_t kDthe = 0x45;
constexpr uint8_t kColclamp = 0x46;
constexpr uint8_t kTestOne = 0x47;
constexpr uint8_t kTestTwo = 0x48;
constexpr uint8_t kPabe = 0x49;
constexpr uint8_t kFbaOne = 0x4A;
constexpr uint8_t kFbaTwo = 0x4B;
constexpr uint8_t kFrameOne = 0x4C;
constexpr uint8_t kFrameTwo = 0x4D;
constexpr uint8_t kZbufOne = 0x4E;
constexpr uint8_t kZbufTwo = 0x4F;
constexpr uint8_t kBitbltbuf = 0x50;
constexpr uint8_t kTrxpos = 0x51;
constexpr uint8_t kTrxreg = 0x52;
constexpr uint8_t kTrxdir = 0x53;
constexpr uint8_t kHwreg = 0x54;
constexpr uint8_t kSignal = 0x60;
constexpr uint8_t kFinish = 0x61;
constexpr uint8_t kLabel = 0x62;
constexpr uint8_t kNop = 0x7F;
}  // namespace reg

// Primitive types (PRIM.type, chapter 01.1).
enum class PrimType : uint8_t {
    Point = 0,
    Line = 1,
    LineStrip = 2,
    Triangle = 3,
    TriangleStrip = 4,
    TriangleFan = 5,
    Sprite = 6,
    Reserved = 7,
};

enum class PrimClass : uint8_t {
    Point,
    Line,
    Triangle,
    Sprite,
    Invalid,
};

inline PrimClass ClassOfType(uint32_t type) {
    switch (type) {
        case 0: return PrimClass::Point;
        case 1: case 2: return PrimClass::Line;
        case 3: case 4: case 5: return PrimClass::Triangle;
        case 6: return PrimClass::Sprite;
        default: return PrimClass::Invalid;
    }
}

// Bit positions inside PRIM (and PRMODE for bits 3..10).
namespace primbit {
constexpr unsigned kIip = 3;
constexpr unsigned kTme = 4;
constexpr unsigned kFge = 5;
constexpr unsigned kAbe = 6;
constexpr unsigned kAa1 = 7;
constexpr unsigned kFst = 8;
constexpr unsigned kCtxt = 9;
constexpr unsigned kFix = 10;
}  // namespace primbit

// Local memory geometry (chapter 01.3).
constexpr uint32_t kLocalMemoryBytes = 4u * 1024u * 1024u;
constexpr uint32_t kBlocksTotal = 16384u;
constexpr uint32_t kBlockBytes = 256u;
constexpr uint32_t kBlocksPerPage = 32u;
constexpr uint32_t kTransferCapBytes = 4u * 1024u * 1024u;

}  // namespace coregs
