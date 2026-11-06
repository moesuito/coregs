// SPDX-License-Identifier: MIT
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec): chapter 14.3 (M1 replay).
#include "cpuref_replay.hpp"

#include <cstring>

namespace coregs::cpuref {

namespace {

constexpr uint8_t kTagGif = 'G';
constexpr uint8_t kTagPrivileged = 'P';
constexpr uint8_t kTagFifoRead = 'R';
constexpr uint8_t kTagFrameEnd = 'V';

// Little-endian reader with bounds checks.
class Reader {
public:
    explicit Reader(const std::vector<uint8_t>& data) : data_(data) {}

    bool Has(size_t count) const { return count <= data_.size() - position_; }
    size_t Position() const { return position_; }

    uint8_t U8() {
        const uint8_t value = data_[position_];
        position_ += 1;
        return value;
    }
    uint32_t U32() {
        uint32_t value = 0;
        std::memcpy(&value, data_.data() + position_, 4);
        position_ += 4;
        return value;
    }
    uint64_t U64() {
        uint64_t value = 0;
        std::memcpy(&value, data_.data() + position_, 8);
        position_ += 8;
        return value;
    }
    std::vector<uint8_t> Bytes(size_t count) {
        std::vector<uint8_t> out(data_.begin() + static_cast<std::ptrdiff_t>(position_),
                                 data_.begin() + static_cast<std::ptrdiff_t>(position_ + count));
        position_ += count;
        return out;
    }

private:
    const std::vector<uint8_t>& data_;
    size_t position_ = 0;
};

}  // namespace

bool ReplayCapture(const std::vector<uint8_t>& stream, ReplayTarget& renderer, ReplaySummary& summary,
                   const std::function<void(uint64_t frame, const FrameImage& image)>& onFrame,
                   std::string& error) {
    if (stream.size() >= 8 && std::memcmp(stream.data(), "GSSTREAM", 8) == 0) {
        return ReplayCaptureV2(stream, renderer, summary, onFrame, error);
    }
    Reader reader(stream);
    while (reader.Has(1)) {
        const size_t offset = reader.Position();
        const uint8_t tag = reader.U8();
        summary.records++;
        switch (tag) {
            case kTagGif: {
                if (!reader.Has(5)) {
                    error = "truncated GIF header at byte " + std::to_string(offset);
                    return false;
                }
                const uint8_t path = reader.U8();
                const uint32_t bytes = reader.U32();
                if (!reader.Has(bytes)) {
                    error = "truncated GIF payload at byte " + std::to_string(offset);
                    return false;
                }
                const std::vector<uint8_t> payload = reader.Bytes(bytes);
                summary.gifPackets++;
                summary.gifBytes += bytes;
                if (!renderer.GifTransfer(path, payload)) {
                    summary.rejectedGifPackets++;
                }
                break;
            }
            case kTagPrivileged: {
                if (!reader.Has(12)) {
                    error = "truncated privileged write at byte " + std::to_string(offset);
                    return false;
                }
                const uint32_t address = reader.U32();
                const uint64_t value = reader.U64();
                summary.privileged++;
                renderer.WritePrivileged(address, value);
                break;
            }
            case kTagFifoRead: {
                if (!reader.Has(4)) {
                    error = "truncated FIFO read at byte " + std::to_string(offset);
                    return false;
                }
                const uint32_t qwords = reader.U32();
                if (qwords > CpuRenderer::kMaxFifoQwords) {
                    error = "oversized FIFO read at byte " + std::to_string(offset);
                    return false;
                }
                summary.fifoReads++;
                renderer.ReadFifo(qwords);  // performed for its side effects; data is not stored
                break;
            }
            case kTagFrameEnd: {
                if (!reader.Has(8)) {
                    error = "truncated frame end at byte " + std::to_string(offset);
                    return false;
                }
                const uint64_t frame = reader.U64();
                const FrameImage image = renderer.EndFrame();
                summary.frames++;
                if (image.valid) {
                    summary.framesWithImage++;
                }
                if (onFrame) {
                    onFrame(frame, image);
                }
                break;
            }
            default:
                error = "unknown record tag at byte " + std::to_string(offset);
                return false;
        }
    }
    return true;
}

uint64_t HashLocalMemory(const uint8_t* bytes, size_t byteCount) {
    uint64_t hash = 1469598103934665603ull;
    size_t i = 0;
    for (; i + 8 <= byteCount; i += 8) {
        uint64_t word = 0;
        for (int b = 0; b < 8; ++b) word |= static_cast<uint64_t>(bytes[i + b]) << (8 * b);
        hash ^= word;
        hash *= 1099511628211ull;
    }
    for (; i < byteCount; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

void CaptureWriter::AppendU32(uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        bytes_.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

void CaptureWriter::AppendU64(uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        bytes_.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

void CaptureWriter::Gif(uint8_t path, const std::vector<uint8_t>& payload) {
    bytes_.push_back(kTagGif);
    bytes_.push_back(path);
    AppendU32(static_cast<uint32_t>(payload.size()));
    bytes_.insert(bytes_.end(), payload.begin(), payload.end());
}

void CaptureWriter::Privileged(uint32_t address, uint64_t value) {
    bytes_.push_back(kTagPrivileged);
    AppendU32(address);
    AppendU64(value);
}

void CaptureWriter::FifoRead(uint32_t qwords) {
    bytes_.push_back(kTagFifoRead);
    AppendU32(qwords);
}

void CaptureWriter::FrameEnd(uint64_t frame) {
    bytes_.push_back(kTagFrameEnd);
    AppendU64(frame);
}

std::vector<uint8_t> CaptureWriter::Packet(uint32_t nloop, bool eop, bool pre, uint32_t prim, uint32_t format,
                                           const std::vector<uint8_t>& regs, const std::vector<uint64_t>& words) {
    // GIF tag (chapter 02.1): NLOOP 0..14, EOP 15, PRE 46, PRIM 47..57, FLG 58..59, NREG 60..63.
    uint64_t lo = static_cast<uint64_t>(nloop & 0x7FFFu);
    lo |= static_cast<uint64_t>(eop ? 1 : 0) << 15;
    lo |= static_cast<uint64_t>(pre ? 1 : 0) << 46;
    lo |= static_cast<uint64_t>(prim & 0x7FFu) << 47;
    lo |= static_cast<uint64_t>(format & 3u) << 58;
    lo |= static_cast<uint64_t>(regs.size() % 16) << 60;
    uint64_t hi = 0;
    for (size_t i = 0; i < regs.size() && i < 16; ++i) {
        hi |= static_cast<uint64_t>(regs[i] & 0xFu) << (4 * i);
    }
    std::vector<uint8_t> out;
    auto push64 = [&out](uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            out.push_back(static_cast<uint8_t>(value >> (8 * i)));
        }
    };
    push64(lo);
    push64(hi);
    // REGLIST data: two 64-bit words per qword, padded with zeros.
    for (size_t i = 0; i < words.size(); i += 2) {
        push64(words[i]);
        push64(i + 1 < words.size() ? words[i + 1] : 0);
    }
    return out;
}

namespace {

// The CPU reference as a ReplayTarget, so the original call sites keep working unchanged.
class CpuRendererTarget final : public ReplayTarget {
public:
    explicit CpuRendererTarget(CpuRenderer& renderer) : renderer_(renderer) {}
    bool GifTransfer(std::uint32_t path, const std::vector<std::uint8_t>& bytes) override {
        return renderer_.GifTransfer(path, bytes);
    }
    bool WritePrivileged(std::uint32_t address, std::uint64_t value) override {
        return renderer_.WritePrivileged(address, value);
    }
    std::vector<std::uint8_t> ReadFifo(std::uint32_t qwords) override { return renderer_.ReadFifo(qwords); }
    FrameImage EndFrame() override { return renderer_.EndFrame(); }
    void SetDisplayField(bool odd) override { renderer_.SetDisplayField(odd); }
    bool DisplayField() const override { return renderer_.DisplayField(); }
    void ResetGs(bool clearVram) override { renderer_.ResetGs(clearVram); }
    void ResetGif(std::uint32_t pathMask) override { renderer_.ResetGif(pathMask); }

private:
    CpuRenderer& renderer_;
};

}  // namespace

bool ReplayCapture(const std::vector<uint8_t>& stream, CpuRenderer& renderer, ReplaySummary& summary,
                   const std::function<void(uint64_t frame, const FrameImage& image)>& onFrame,
                   std::string& error) {
    CpuRendererTarget target(renderer);
    return ReplayCapture(stream, target, summary, onFrame, error);
}

bool ReplayCaptureV2(const std::vector<uint8_t>& stream, CpuRenderer& renderer, ReplaySummary& summary,
                     const std::function<void(uint64_t frame, const FrameImage& image)>& onFrame,
                     std::string& error) {
    CpuRendererTarget target(renderer);
    return ReplayCaptureV2(stream, target, summary, onFrame, error);
}

}  // namespace coregs::cpuref
