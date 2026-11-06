// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// HWR GIF parser: PACKED, REGLIST and IMAGE packets on paths 1, 2 and 3
// (chapter 02.1). Input is whole 128-bit qwords; packets may be split at any
// qword boundary across calls.
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hw_gs_counters.hpp"
#include "hw_gs_types.hpp"

namespace coregs {

enum class GifFormat : uint8_t { Packed = 0, RegList = 1, Image = 2 };

// Decoded GIF tag (chapter 02.1 table).
struct GifTag {
    uint32_t nloop = 0;
    bool eop = false;
    bool pre = false;
    uint32_t prim = 0;
    GifFormat format = GifFormat::Packed;
    uint32_t nreg = 0;           // 0 means 16 descriptors
    std::array<uint8_t, 16> regs{};
};

GifTag DecodeGifTag(const Qword& qword);

// Vertex kick decoded from XYZ2/XYZF2/XYZ3/XYZF3 via PACKED descriptors, A+D or REGLIST.
struct GifKick {
    uint32_t x = 0;      // 12.4, 16 bits
    uint32_t y = 0;      // 12.4, 16 bits
    uint32_t z = 0;
    bool hasFog = false;
    uint8_t fog = 0;
    bool drawing = true; // false for XYZ3/XYZF3 and for ADC = 1 (PACKED)
};

// Receives parsed register writes in canonical (chapter 01.1) layout.
class GifSink {
public:
    virtual ~GifSink() = default;
    // Any non-kick register write. Canonical layouts; PACKED RGBA already
    // carries the Q latch in its Q field; A+D RGBAQ Q is sanitised.
    virtual void WriteRegister(uint8_t address, uint64_t data) = 0;
    virtual void Kick(const GifKick& kick) = 0;
    // One IMAGE qword (16 bytes) for the host to local transfer on the path.
    virtual void ImageQword(uint32_t path, const Qword& qword) = 0;
    // A run of IMAGE qwords of one packet (the parser hands them over at once; IN-353). Same effect as
    // ImageQword on each, in order.
    virtual void ImageQwords(uint32_t path, const Qword* qwords, size_t count) {
        for (size_t i = 0; i < count; ++i) ImageQword(path, qwords[i]);
    }
    // IN-366: the vertices of a PACKED STQ, RGBA, XYZF2 run (the game's VU1 strip shape), decoded by the parser. Same
    // effect as WriteRegister(ST), WriteRegister(RGBAQ) and Kick for each vertex in order.
    struct StripVertex {
        uint64_t st = 0;
        uint64_t rgbaq = 0;
        GifKick kick;
    };
    virtual void StripVertices(const StripVertex* vertices, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            WriteRegister(0x02u, vertices[i].st);     // ST
            WriteRegister(0x01u, vertices[i].rgbaq);  // RGBAQ
            Kick(vertices[i].kick);
        }
    }
};

class GifParser {
public:
    explicit GifParser(CounterSet& counters);

    // Processes count qwords for path 1..3.
    void Transfer(uint32_t path, const Qword* qwords, size_t count, GifSink& sink);

    // True while the current tag of the path still expects data.
    bool Busy(uint32_t path) const;

    // Ends a path-1 packet without EOP (XGKICK): loop counter forced to 0.
    void EndPathOnePacket();

    // GIF reset for the selected paths (bit 0 = path 1, bit 1 = path 2, bit 2 = path 3).
    // Also resets the Q latch to 1.0.
    void Reset(uint32_t pathMask);

    float QLatch() const { return qLatch_; }

    // IN-366 A/B switch: PACKED STQ, RGBA, XYZF2 runs decoded in one loop (default on).
    void SetFastStrips(bool enabled) { fastStrips_ = enabled; }
    uint64_t FastStripVertices() const { return fastStripVertices_; }

private:
    struct PathState {
        bool inPacket = false;       // current tag still has data to consume
        GifTag tag{};
        uint32_t remainingItems = 0; // PACKED/IMAGE qwords or REGLIST words
        uint32_t descriptorIndex = 0;
    };

    void ReadTag(uint32_t path, const Qword& qword, GifSink& sink);
    bool fastStrips_ = true;
    uint64_t fastStripVertices_ = 0;
    std::vector<GifSink::StripVertex> strip_;
    void ConsumePacked(PathState& state, const Qword& qword, GifSink& sink);
    void ConsumeRegList(PathState& state, const Qword& qword, GifSink& sink);
    void ConsumeImage(uint32_t path, PathState& state, const Qword& qword, GifSink& sink);

    uint32_t SanitiseStQ(uint32_t bits);
    uint32_t SanitiseAdQ(uint32_t bits);

    CounterSet& counters_;
    std::array<PathState, 3> paths_{};
    float qLatch_ = 1.0f;
};

}  // namespace coregs
