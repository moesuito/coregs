// MIT License. Copyright (c) 2026 Joao Alano and GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 12.9 and 14.3; clean room
// (decision 0070).
//
// The GPU write-back renderer as a capture replay target, shared by hw_writeback_replay and the replay
// parity fixtures, so the tool and the tests drive the renderer through exactly the same adapter as the
// CPU reference (cpuref::ReplayTarget). EndFrame flushes the frame and composes the display image.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../cpuref/cpuref_replay.hpp"
#include "hw_gs_writeback.hpp"

namespace coregs::writeback {

class WritebackReplayTarget final : public cpuref::ReplayTarget {
public:
    // `kind` selects the image of each frame end: Reference (default, the image the CPU reference and the
    // 1x decode produces) or Presentation (field offset and deinterlace, what the window shows).
    explicit WritebackReplayTarget(Renderer& renderer, ComposeKind kind = ComposeKind::Reference)
        : renderer_(renderer), kind_(kind) {}

    bool GifTransfer(std::uint32_t path, const std::vector<std::uint8_t>& bytes) override {
        return renderer_.GifTransfer(path, bytes.data(), bytes.size());
    }
    bool WritePrivileged(std::uint32_t address, std::uint64_t value) override {
        return renderer_.WritePrivileged(address, value);
    }
    std::vector<std::uint8_t> ReadFifo(std::uint32_t qwords) override {
        // Same shape as the CPU reference: the requested qwords, zero-filled past the real data.
        std::vector<std::uint8_t> data(static_cast<std::size_t>(qwords) * 16u, 0);
        renderer_.ReadFifo(data.data(), data.size());
        for (const std::uint8_t byte : data) {  // FNV-1a over every reply byte (readback answers are gated too)
            fifoHash_ = (fifoHash_ ^ byte) * 1099511628211ull;
        }
        ++fifoReads_;
        return data;
    }
    std::uint64_t FifoHash() const { return fifoHash_; }
    std::uint64_t FifoReads() const { return fifoReads_; }

    // IN-364 check: every progressive presentation frame is also composed on the GPU and compared with the CPU.
    void SetGpuComposeCheck(bool enabled) { gpuComposeCheck_ = enabled; }
    std::uint64_t GpuComposeEqual() const { return gpuComposeEqual_; }
    std::uint64_t GpuComposeDifferent() const { return gpuComposeDifferent_; }

    cpuref::FrameImage EndFrame() override {
        renderer_.Flush();
        GpuFrame gpuFrame;
        std::vector<std::uint32_t> gpuPixels;
        const bool onGpu = gpuComposeCheck_ && kind_ == ComposeKind::Presentation && renderer_.ComposeFrameGpu(gpuFrame) &&
                           renderer_.ReadGpuFrame(gpuFrame, gpuPixels);
        FrameOutput out;
        renderer_.ComposeFrame(out, kind_);
        if (onGpu) {
            const bool same = out.valid && out.width == gpuFrame.width && out.height == gpuFrame.height &&
                              out.pixels == gpuPixels;
            ++(same ? gpuComposeEqual_ : gpuComposeDifferent_);
        }
        cpuref::FrameImage frame;
        frame.valid = out.valid && out.width > 0 && out.height > 0;
        if (!frame.valid) return frame;
        frame.image = display::Image(out.width, out.height);
        for (std::uint32_t y = 0; y < out.height; ++y) {
            for (std::uint32_t x = 0; x < out.width; ++x) {
                const std::uint32_t pixel = out.pixels[static_cast<std::size_t>(y) * out.width + x];
                display::Rgba8& dst = frame.image.at(x, y);
                dst.r = static_cast<std::uint8_t>(pixel & 0xFFu);
                dst.g = static_cast<std::uint8_t>((pixel >> 8) & 0xFFu);
                dst.b = static_cast<std::uint8_t>((pixel >> 16) & 0xFFu);
                dst.a = static_cast<std::uint8_t>((pixel >> 24) & 0xFFu);
            }
        }
        return frame;
    }

    void SetDisplayField(bool odd) override {
        oddField_ = odd;
        renderer_.SetDisplayField(odd);
    }
    bool DisplayField() const override { return oddField_; }
    void ResetGs(bool clearVram) override {
        renderer_.ResetGs();
        if (clearVram) renderer_.ClearVram();
    }
    void ResetGif(std::uint32_t pathMask) override { renderer_.GifReset(pathMask); }

private:
    Renderer& renderer_;
    ComposeKind kind_;
    std::uint64_t fifoHash_ = 1469598103934665603ull;
    std::uint64_t fifoReads_ = 0;
    bool gpuComposeCheck_ = false;
    std::uint64_t gpuComposeEqual_ = 0;
    std::uint64_t gpuComposeDifferent_ = 0;
    bool oddField_ = false;
};

}  // namespace coregs::writeback
