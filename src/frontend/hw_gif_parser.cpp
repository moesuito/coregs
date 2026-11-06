// Copyright 2026 GT4Recomp contributors. Licensed under the MIT License (see LICENSE).
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#include "hw_gif_parser.hpp"

#include <limits>

namespace coregs {
namespace {

constexpr uint32_t kFloatMinNormalBits = 0x00800000u;  // smallest positive normal
constexpr uint32_t kFloatMaxBits = 0x7F7FFFFFu;        // largest finite
constexpr uint32_t kOneBits = 0x3F800000u;             // 1.0f

bool IsNaNBits(uint32_t bits) {
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

// Fields of an XYZ kick in PACKED layout (chapter 02.1, descriptors 4, 5, C, D).
GifKick DecodePackedXyz(const Qword& q, bool withFog, bool drawing) {
    GifKick kick;
    kick.x = static_cast<uint32_t>(BitRange(q.lo, 0, 15));
    kick.y = static_cast<uint32_t>(BitRange(q.lo, 32, 47));
    if (withFog) {
        kick.z = static_cast<uint32_t>(BitRange(q.hi, 4, 27));   // bits 68..91
        kick.fog = static_cast<uint8_t>(BitRange(q.hi, 36, 43)); // bits 100..107
        kick.hasFog = true;
    } else {
        kick.z = static_cast<uint32_t>(BitRange(q.hi, 0, 31));   // bits 64..95
    }
    const bool adc = BitRange(q.hi, 47, 47) != 0;               // bit 111
    kick.drawing = drawing && !adc;
    return kick;
}

// Fields of an XYZ kick in register layout (chapter 01.1, A+D and REGLIST words).
GifKick DecodeRegisterXyz(uint64_t word, bool withFog, bool drawing) {
    GifKick kick;
    kick.x = static_cast<uint32_t>(BitRange(word, 0, 15));
    kick.y = static_cast<uint32_t>(BitRange(word, 16, 31));
    if (withFog) {
        kick.z = static_cast<uint32_t>(BitRange(word, 32, 55));
        kick.fog = static_cast<uint8_t>(BitRange(word, 56, 63));
        kick.hasFog = true;
    } else {
        kick.z = static_cast<uint32_t>(BitRange(word, 32, 63));
    }
    kick.drawing = drawing;
    return kick;
}

}  // namespace

GifTag DecodeGifTag(const Qword& qword) {
    GifTag tag;
    tag.nloop = static_cast<uint32_t>(BitRange(qword.lo, 0, 14));
    tag.eop = BitRange(qword.lo, 15, 15) != 0;
    tag.pre = BitRange(qword.lo, 46, 46) != 0;
    tag.prim = static_cast<uint32_t>(BitRange(qword.lo, 47, 57));
    switch (BitRange(qword.lo, 58, 59)) {
        case 0: tag.format = GifFormat::Packed; break;
        case 1: tag.format = GifFormat::RegList; break;
        default: tag.format = GifFormat::Image; break;
    }
    const uint32_t nreg = static_cast<uint32_t>(BitRange(qword.lo, 60, 63));
    tag.nreg = nreg == 0 ? 16u : nreg;
    for (unsigned i = 0; i < 16; ++i) {
        tag.regs[i] = static_cast<uint8_t>((qword.hi >> (4u * i)) & 0xFu);
    }
    return tag;
}

GifParser::GifParser(CounterSet& counters) : counters_(counters) {}

void GifParser::Transfer(uint32_t path, const Qword* qwords, size_t count, GifSink& sink) {
    if (path < 1 || path > 3) {
        return;
    }
    PathState& state = paths_[path - 1];
    for (size_t i = 0; i < count; ++i) {
        const Qword& qword = qwords[i];
        if (!state.inPacket) {
            ReadTag(path, qword, sink);
            continue;
        }
        if (state.tag.format == GifFormat::Image) {
            // IN-353: the image qwords of this packet that are in this transfer go to the sink at once.
            const size_t run = std::min<size_t>(state.remainingItems, count - i);
            state.remainingItems -= static_cast<uint32_t>(run);
            counters_.Add(Counter::ImageQwords, run);
            sink.ImageQwords(path, &qwords[i], run);
            i += run - 1;
            if (state.remainingItems == 0) {
                state.inPacket = false;
                if (state.tag.eop) {
                    counters_.Add(Counter::GifPacketsEnded);
                }
            }
            continue;
        }
        // IN-366: a PACKED STQ, RGBA, XYZF2 tag whose remaining data is all in this transfer (the game's VU1 strips,
        // about 85,000 vertices per 480p race frame) is decoded in one loop: the same register values, Q latch and
        // kicks in the same order as ConsumePacked, handed to the sink at once.
        if (fastStrips_ && state.tag.format == GifFormat::Packed && state.descriptorIndex == 0 && state.tag.nreg == 3 &&
            state.tag.regs[0] == 0x2 && state.tag.regs[1] == 0x1 && state.tag.regs[2] == 0x4 &&
            state.remainingItems % 3u == 0u && count - i >= state.remainingItems) {
            const size_t vertices = state.remainingItems / 3u;
            strip_.resize(vertices);
            for (size_t v = 0; v < vertices; ++v) {
                const Qword& stq = qwords[i + 3 * v];
                const Qword& rgba = qwords[i + 3 * v + 1];
                const Qword& xyz = qwords[i + 3 * v + 2];
                qLatch_ = BitsToFloat(SanitiseStQ(static_cast<uint32_t>(stq.hi & 0xFFFFFFFFu)));
                GifSink::StripVertex& out = strip_[v];
                out.st = stq.lo;
                out.rgbaq = BitRange(rgba.lo, 0, 7) | (BitRange(rgba.lo, 32, 39) << 8) | (BitRange(rgba.hi, 0, 7) << 16) |
                            (BitRange(rgba.hi, 32, 39) << 24) | (uint64_t{FloatToBits(qLatch_)} << 32);
                out.kick = DecodePackedXyz(xyz, true, true);
            }
            counters_.Add(Counter::PackedItems, state.remainingItems);
            i += state.remainingItems - 1;
            state.remainingItems = 0;
            state.inPacket = false;
            if (state.tag.eop) {
                counters_.Add(Counter::GifPacketsEnded);
            }
            fastStripVertices_ += vertices;
            sink.StripVertices(strip_.data(), vertices);
            continue;
        }
        switch (state.tag.format) {
            case GifFormat::Packed:
                ConsumePacked(state, qword, sink);
                break;
            case GifFormat::RegList:
                ConsumeRegList(state, qword, sink);
                break;
            case GifFormat::Image:
                ConsumeImage(path, state, qword, sink);
                break;
        }
        if (state.remainingItems == 0) {
            state.inPacket = false;
            if (state.tag.eop) {
                counters_.Add(Counter::GifPacketsEnded);
            }
        }
    }
}

void GifParser::ReadTag(uint32_t path, const Qword& qword, GifSink& sink) {
    PathState& state = paths_[path - 1];
    state.tag = DecodeGifTag(qword);
    state.descriptorIndex = 0;
    counters_.Add(Counter::GifTags);

    // PRE applies PRIM for PACKED; the specification counts PRE in REGLIST (V-02-1).
    if (state.tag.pre && state.tag.format == GifFormat::Packed) {
        sink.WriteRegister(reg::kPrim, state.tag.prim);
    } else if (state.tag.pre && state.tag.format == GifFormat::RegList) {
        counters_.Add(Counter::PreInRegList);
    }

    if (state.tag.nloop == 0) {
        state.inPacket = false;
        state.remainingItems = 0;
        return;
    }

    // A tag with NLOOP > 0 resets the Q latch (chapter 02.1).
    qLatch_ = 1.0f;

    switch (state.tag.format) {
        case GifFormat::Packed:
            state.remainingItems = state.tag.nloop * state.tag.nreg;
            break;
        case GifFormat::RegList:
            state.remainingItems = state.tag.nloop * state.tag.nreg;  // 64-bit words
            break;
        case GifFormat::Image:
            state.remainingItems = state.tag.nloop;                    // qwords
            break;
    }
    state.inPacket = state.remainingItems > 0;
}

void GifParser::ConsumePacked(PathState& state, const Qword& q, GifSink& sink) {
    const uint8_t descriptor = state.tag.regs[state.descriptorIndex];
    state.descriptorIndex = (state.descriptorIndex + 1u) % state.tag.nreg;
    state.remainingItems--;
    counters_.Add(Counter::PackedItems);

    switch (descriptor) {
        case 0x0:  // PRIM
            sink.WriteRegister(reg::kPrim, BitRange(q.lo, 0, 10));
            break;
        case 0x1: {  // RGBA with the current Q latch
            const uint64_t r = BitRange(q.lo, 0, 7);
            const uint64_t g = BitRange(q.lo, 32, 39);
            const uint64_t b = BitRange(q.hi, 0, 7);
            const uint64_t a = BitRange(q.hi, 32, 39);
            const uint64_t qBits = FloatToBits(qLatch_);
            sink.WriteRegister(reg::kRgbaq, r | (g << 8) | (b << 16) | (a << 24) | (qBits << 32));
            break;
        }
        case 0x2: {  // STQ: S, T to ST; Q only into the latch
            const uint32_t qBits = static_cast<uint32_t>(q.hi & 0xFFFFFFFFu);
            const uint32_t sanitised = SanitiseStQ(qBits);
            qLatch_ = BitsToFloat(sanitised);
            sink.WriteRegister(reg::kSt, q.lo);
            break;
        }
        case 0x3: {  // UV: U 0..13, V 32..45 to the register layout (V at 16)
            const uint64_t u = BitRange(q.lo, 0, 13);
            const uint64_t v = BitRange(q.lo, 32, 45);
            sink.WriteRegister(reg::kUv, u | (v << 16));
            break;
        }
        case 0x4:  // XYZF2 with drawing kick unless ADC
            sink.Kick(DecodePackedXyz(q, true, true));
            break;
        case 0x5:  // XYZ2
            sink.Kick(DecodePackedXyz(q, false, true));
            break;
        case 0x6: case 0x7: case 0x8: case 0x9:  // TEX0_1/2, CLAMP_1/2
            sink.WriteRegister(static_cast<uint8_t>(descriptor), q.lo);
            break;
        case 0xA: {  // FOG: F into the register's F field
            const uint64_t f = BitRange(q.hi, 36, 43);
            sink.WriteRegister(reg::kFog, f << 56);
            break;
        }
        case 0xB:
            break;
        case 0xC:  // XYZF3, kick without drawing
            sink.Kick(DecodePackedXyz(q, true, false));
            break;
        case 0xD:  // XYZ3
            sink.Kick(DecodePackedXyz(q, false, false));
            break;
        case 0xE: {  // A+D: address is bits 64..71, masked to 7 bits
            const uint8_t address = static_cast<uint8_t>(BitRange(q.hi, 0, 6));
            uint64_t data = q.lo;
            // A+D carries the canonical 64-bit register layout, not the PACKED XYZ
            // descriptor layout. Its ignored high-qword bits include bit 111: ADC
            // applies only to descriptors 4/5/C/D (01.1, 02.1, 02.3; IN-326).
            switch (address) {
                case reg::kXyzf2: sink.Kick(DecodeRegisterXyz(data, true, true)); return;
                case reg::kXyz2: sink.Kick(DecodeRegisterXyz(data, false, true)); return;
                case reg::kXyzf3: sink.Kick(DecodeRegisterXyz(data, true, false)); return;
                case reg::kXyz3: sink.Kick(DecodeRegisterXyz(data, false, false)); return;
                default: break;
            }
            if (address == reg::kRgbaq) {
                const uint32_t qBits = SanitiseAdQ(static_cast<uint32_t>(data >> 32));
                data = (data & 0xFFFFFFFFu) | (static_cast<uint64_t>(qBits) << 32);
            }
            sink.WriteRegister(address, data);
            break;
        }
        default:  // 0xF NOP
            break;
    }
}

void GifParser::ConsumeRegList(PathState& state, const Qword& q, GifSink& sink) {
    // Each qword holds two 64-bit words; an odd final word is padded.
    for (int half = 0; half < 2 && state.remainingItems > 0; ++half) {
        const uint64_t word = half == 0 ? q.lo : q.hi;
        const uint8_t address = state.tag.regs[state.descriptorIndex];
        state.descriptorIndex = (state.descriptorIndex + 1u) % state.tag.nreg;
        state.remainingItems--;
        counters_.Add(Counter::RegListWords);

        switch (address) {
            case reg::kXyzf2:
                sink.Kick(DecodeRegisterXyz(word, true, true));
                break;
            case reg::kXyz2:
                sink.Kick(DecodeRegisterXyz(word, false, true));
                break;
            case reg::kXyzf3:
                sink.Kick(DecodeRegisterXyz(word, true, false));
                break;
            case reg::kXyz3:
                sink.Kick(DecodeRegisterXyz(word, false, false));
                break;
            case reg::kAdPacked:  // no-op in REGLIST
                break;
            default:
                sink.WriteRegister(address, word);
                break;
        }
    }
}

void GifParser::ConsumeImage(uint32_t path, PathState& state, const Qword& qword, GifSink& sink) {
    state.remainingItems--;
    counters_.Add(Counter::ImageQwords);
    sink.ImageQword(path, qword);
}

uint32_t GifParser::SanitiseStQ(uint32_t bits) {
    if (bits == 0u) {  // +0.0 only; -0.0 is kept
        counters_.Add(Counter::QSanitised);
        return kFloatMinNormalBits;
    }
    if (IsNaNBits(bits)) {
        counters_.Add(Counter::QSanitised);
        return kFloatMaxBits;
    }
    return bits;
}

uint32_t GifParser::SanitiseAdQ(uint32_t bits) {
    if (bits == 0u) {
        counters_.Add(Counter::QSanitised);
        return kOneBits;
    }
    if (IsNaNBits(bits)) {
        counters_.Add(Counter::QSanitised);
        return kFloatMaxBits;
    }
    return bits;
}

bool GifParser::Busy(uint32_t path) const {
    if (path < 1 || path > 3) {
        return false;
    }
    return paths_[path - 1].inPacket;
}

void GifParser::EndPathOnePacket() {
    PathState& state = paths_[0];
    if (state.inPacket) {
        counters_.Add(Counter::GifPathOneNoEop);
    }
    state.inPacket = false;
    state.remainingItems = 0;
}

void GifParser::Reset(uint32_t pathMask) {
    for (uint32_t path = 0; path < 3; ++path) {
        if (pathMask & (1u << path)) {
            paths_[path] = PathState{};
        }
    }
    qLatch_ = 1.0f;
}

}  // namespace coregs
