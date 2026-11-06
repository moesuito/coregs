// SPDX-License-Identifier: Apache-2.0
//
// HWR CPU reference replay: parses legacy COREGS_PGS_CAPTURE and native GSSTREAM v2 streams
// and feeds them to the CPU reference renderer, one display image per frame end.
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec): chapter 14.3 (M1 replay).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "cpuref_renderer.hpp"

namespace coregs::cpuref {

struct ReplaySummary {
    uint64_t records = 0;
    uint64_t gifPackets = 0;
    uint64_t gifBytes = 0;
    uint64_t privileged = 0;
    uint64_t fifoReads = 0;
    uint64_t frames = 0;
    uint64_t framesWithImage = 0;
    uint64_t rejectedGifPackets = 0;  // GIF path 0 or odd sizes (IN-312)
    uint64_t presentationControls = 0;  // Window/resize/cursor records have no CPU image effect.
    uint64_t resets = 0;
    uint64_t fieldsWithoutPhase = 0;  // v2 field-invalid records; initial phase is unspecified.
};

// The four calls a replay makes on a renderer, so one recorded stream can drive the CPU reference or
// the GPU write-back renderer (M6 replay tool).
class ReplayTarget {
public:
    virtual ~ReplayTarget() = default;
    virtual bool GifTransfer(std::uint32_t path, const std::vector<std::uint8_t>& bytes) = 0;
    virtual bool WritePrivileged(std::uint32_t address, std::uint64_t value) = 0;
    virtual std::vector<std::uint8_t> ReadFifo(std::uint32_t qwords) = 0;
    virtual FrameImage EndFrame() = 0;
    // v2 records carry the field bit and explicit resets (protocol 6.1), so a target must expose them.
    virtual void SetDisplayField(bool odd) = 0;
    virtual bool DisplayField() const = 0;
    virtual void ResetGs(bool clearVram) = 0;
    virtual void ResetGif(std::uint32_t pathMask) = 0;
};

// Replays one capture stream into any target. onFrame receives every frame end with the display result.
// Returns false and sets error when the stream is malformed (the summary holds the progress).
bool ReplayCapture(const std::vector<uint8_t>& stream, ReplayTarget& target, ReplaySummary& summary,
                   const std::function<void(uint64_t frame, const FrameImage& image)>& onFrame,
                   std::string& error);
bool ReplayCaptureV2(const std::vector<uint8_t>& stream, ReplayTarget& target, ReplaySummary& summary,
                     const std::function<void(uint64_t frame, const FrameImage& image)>& onFrame,
                     std::string& error);

// Same stream into the CPU reference renderer (the original entry points).
bool ReplayCapture(const std::vector<uint8_t>& stream, CpuRenderer& renderer, ReplaySummary& summary,
                   const std::function<void(uint64_t frame, const FrameImage& image)>& onFrame,
                   std::string& error);

// Native GSSTREAM v2 envelopes, from our Apache-2.0 protocol (GS-STREAM-PROTOCOL sections 2, 6, 10).
bool ReplayCaptureV2(const std::vector<uint8_t>& stream, CpuRenderer& renderer, ReplaySummary& summary,
                     const std::function<void(uint64_t frame, const FrameImage& image)>& onFrame,
                     std::string& error);

// FNV-1a 64-bit hash of a shadow local memory image, taken over 64-bit little-endian words so that
// hashing 4 MiB per frame stays cheap. Both replay tools print it per frame (--slm-hashes), so the GPU
// write-back path and the CPU reference can be compared in memory on every frame, not only on images.
uint64_t HashLocalMemory(const uint8_t* bytes, size_t byteCount);

// Writes capture streams in the same record layout (used by the tests and synthetic captures).
class CaptureWriter {
public:
    void Gif(uint8_t path, const std::vector<uint8_t>& bytes);
    void Privileged(uint32_t address, uint64_t value);
    void FifoRead(uint32_t qwords);
    void FrameEnd(uint64_t frame);

    // Builds a GIF packet with one tag and the given data (qword multiple of 16 bytes).
    static std::vector<uint8_t> Packet(uint32_t nloop, bool eop, bool pre, uint32_t prim, uint32_t format,
                                       const std::vector<uint8_t>& regs, const std::vector<uint64_t>& words);

    const std::vector<uint8_t>& Bytes() const { return bytes_; }

private:
    void AppendU32(uint32_t value);
    void AppendU64(uint64_t value);
    std::vector<uint8_t> bytes_;
};

}  // namespace coregs::cpuref
