// SPDX-License-Identifier: MIT
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec): chapters 02, 01.7, 10.2, 13.1 to 13.3.
#include "cpuref_renderer.hpp"

#include <cstring>
#include <algorithm>

#include "hw_gs_pixel.hpp"
#include "hw_gs_registers.hpp"

namespace coregs::cpuref {

namespace {

// Full-range display list for the VBlank flush: any batch overlapping the frame is flushed.
const BlockRange kEverything{0, kBlocksTotal};

pixel::FrameFormat CircuitFormat(uint32_t psmCode) {
    switch (psmCode) {
        case 0x01:
            return pixel::FrameFormat::Ct24;
        case 0x02:
        case 0x0A:
            return pixel::FrameFormat::Ct16;
        default:
            return pixel::FrameFormat::Ct32;
    }
}

// Reads one circuit's source rectangle from the SLM (13.2 and 13.3). Alpha of CT24 sources is
// 0x80, the value the blend factor uses for CT24 (IN-313).
display::Image ReadCircuit(const memory::LocalMemory& local, const display::CircuitRect& circuit) {
    display::Image image(circuit.read_width, circuit.read_height);
    const auto psm = memory::decodePsm(circuit.psm);
    if (!psm) {
        return image;
    }
    const memory::BufferView view{memory::blockOfPage(circuit.fbp), circuit.fbw, *psm};
    const pixel::FrameFormat format = CircuitFormat(circuit.psm);
    for (uint32_t y = 0; y < circuit.read_height; ++y) {
        for (uint32_t x = 0; x < circuit.read_width; ++x) {
            // Read origin wraps modulo 2048 (IN-24).
            const uint32_t sx = static_cast<uint32_t>(circuit.dbx + static_cast<int32_t>(x)) & 2047u;
            const uint32_t sy = static_cast<uint32_t>(circuit.dby + static_cast<int32_t>(y)) & 2047u;
            const uint32_t stored = local.readPixel(view, sx, sy);
            const pixel::Colour colour = pixel::frame_word_to_colour(pixel::unpack_frame_word(format, stored));
            display::Rgba8& out = image.at(x, y);
            out.r = static_cast<uint8_t>(colour.r);
            out.g = static_cast<uint8_t>(colour.g);
            out.b = static_cast<uint8_t>(colour.b);
            out.a = format == pixel::FrameFormat::Ct24 ? 0x80 : static_cast<uint8_t>(colour.a);
        }
    }
    return image;
}

}  // namespace

CpuRenderer::CpuRenderer() : frontEnd_(FrontEndOptions{}, *this, *this, memory_) {
    frontEnd_.SetLocalMemory(memory_.Local().bytes().data());
}

bool CpuRenderer::GifTransfer(uint32_t path, const std::vector<uint8_t>& bytes) {
    if (path == 0 || path > 3) {
        rejectedGifPaths++;
        return false;
    }
    if (bytes.size() % 16 != 0) {
        return false;
    }
    std::vector<Qword> qwords(bytes.size() / 16);
    for (size_t i = 0; i < qwords.size(); ++i) {
        std::memcpy(&qwords[i].lo, bytes.data() + 16 * i, 8);
        std::memcpy(&qwords[i].hi, bytes.data() + 16 * i + 8, 8);
    }
    frontEnd_.GifTransfer(path, qwords.data(), qwords.size());
    return true;
}

bool CpuRenderer::WritePrivileged(uint32_t address, uint64_t value) {
    switch (address) {
        case priv::kPmode:
            displayRegs_.pmode = value;
            return true;
        case priv::kSmode1:
            displayRegs_.smode1 = value;
            return true;
        case priv::kSmode2:
            displayRegs_.smode2 = value;
            return true;
        case priv::kSyncv:
            displayRegs_.syncv = value;
            return true;
        case priv::kDispfb1:
            displayRegs_.circuits[0].dispfb = value;
            return true;
        case priv::kDisplay1:
            displayRegs_.circuits[0].display = value;
            return true;
        case priv::kDispfb2:
            displayRegs_.circuits[1].dispfb = value;
            return true;
        case priv::kDisplay2:
            displayRegs_.circuits[1].display = value;
            return true;
        case priv::kBgcolor:
            background_[0] = static_cast<uint8_t>(Field(value, 0, 8));
            background_[1] = static_cast<uint8_t>(Field(value, 8, 8));
            background_[2] = static_cast<uint8_t>(Field(value, 16, 8));
            return true;
        case priv::kCsr:
            if (Field(value, 9, 1) != 0) {
                ResetGs(false);
            }
            // CSR.FIELD is read-only. The v2 VSYNC supplies the display field explicitly.
            return true;
        default:
            unhandledPrivileged++;
            return false;
    }
}

std::vector<uint8_t> CpuRenderer::ReadFifo(uint32_t qwords) {
    if (qwords > kMaxFifoQwords) {
        ++fifoFaults;
        return {};
    }
    if (qwords == 0) {
        return {};
    }
    // First FIFO access must observe all earlier draws and uploads (10.2).
    frontEnd_.FlushBatch(FlushReason::Transfer);
    bool overrun = false;
    std::vector<uint8_t> data = memory_.ReadLocalToHost(static_cast<size_t>(qwords) * 16u, overrun);
    if (overrun) ++fifoFaults;
    if (memory_.ReadDrained()) frontEnd_.WriteRegister(reg::kTrxdir, 3);
    return data;
}

void CpuRenderer::ResetGif(uint32_t pathMask) {
    frontEnd_.GifReset(pathMask);
    memory_.ResetRead();
}

void CpuRenderer::ResetGs(bool clearVram) {
    frontEnd_.ResetGs();
    ResetGif(7);
    if (clearVram) {
        auto bytes = memory_.Local().bytes();
        std::fill(bytes.begin(), bytes.end(), 0);
    }
    displayRegs_ = {};
    previousDby_ = {};
    background_ = {};
    csrField_ = false;
    lastImage_.reset();
}

FrameImage CpuRenderer::EndFrame() {
    frontEnd_.VBlank(&kEverything, 1);
    frontEnd_.FlushBatch(FlushReason::VBlank);

    FrameImage out;
    const display::DisplayLayout layout = display::compute_layout(displayRegs_, previousDby_);
    if (!layout.valid) {
        return out;
    }
    const display::PmodeFields pmode = display::decode_pmode(displayRegs_.pmode);
    std::array<display::Image, 2> sources{};
    std::array<bool, 2> available{false, false};
    for (size_t i = 0; i < 2; ++i) {
        const display::CircuitRect& circuit = layout.circuits[i];
        if (circuit.enabled && circuit.read_width > 0 && circuit.read_height > 0) {
            sources[i] = ReadCircuit(memory_.Local(), circuit);
            available[i] = true;
        }
    }
    const display::MergePlan plan =
        display::plan_merge(pmode, layout.circuits[0], layout.circuits[1], available[0], available[1]);
    if (plan.outcome == display::MergeOutcome::KeepPrevious) {
        if (lastImage_) {
            out.valid = true;
            out.image = *lastImage_;
        }
        return out;
    }
    const display::OutputSize size = display::merge_output_size(layout, true, 1);
    const display::SourcePlacement rc1{available[0] ? &sources[0] : nullptr, layout.circuits[0].screen_x,
                                       layout.circuits[0].screen_y,
                                       static_cast<int32_t>(layout.circuits[0].display_width),
                                       static_cast<int32_t>(layout.circuits[0].display_height)};
    const display::SourcePlacement rc2{available[1] ? &sources[1] : nullptr, layout.circuits[1].screen_x,
                                       layout.circuits[1].screen_y,
                                       static_cast<int32_t>(layout.circuits[1].display_width),
                                       static_cast<int32_t>(layout.circuits[1].display_height)};
    // Deinterlacing is off in this path, so the interlace offset is zero (13.3, IN-35 for the field).
    display::Image merged = display::compose_merge(plan, pmode, background_[0], background_[1], background_[2],
                                                   size.width, size.height, 1, 0, rc1, rc2);
    lastImage_ = merged;
    out.valid = size.width > 0 && size.height > 0;
    out.image = std::move(merged);
    return out;
}

void CpuRenderer::OnBatch(Batch&& batch) { DrawBatch(batch, memory_.Local(), frontEnd_.Clut(), stats_); }

bool CpuRenderer::OnSync(SyncKind, uint32_t, uint32_t) { return false; }

std::string EncodePpm(const display::Image& image) {
    std::string out = "P6\n" + std::to_string(image.width) + " " + std::to_string(image.height) + "\n255\n";
    out.reserve(out.size() + static_cast<size_t>(image.width) * image.height * 3);
    for (const display::Rgba8& pixel : image.pixels) {
        out.push_back(static_cast<char>(pixel.r));
        out.push_back(static_cast<char>(pixel.g));
        out.push_back(static_cast<char>(pixel.b));
    }
    return out;
}

}  // namespace coregs::cpuref
