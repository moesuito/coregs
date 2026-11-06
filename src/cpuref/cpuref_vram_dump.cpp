// SPDX-License-Identifier: MIT
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 12.9 and 14.2-14.4.
// Oracle decode: MIT ps2-web-recomp runtime/src/gs/gs.cpp::display and gpu/gs.wgsl::display
// (Copyright 2026 Samir Nuri). Circuit selection, geometry and line doubling follow that
// permitted oracle; PCRTC composition in chapter 13 remains a separate output path.
#include "cpuref_renderer.hpp"

#include <iomanip>
#include <sstream>

namespace coregs::cpuref {

bool ParseDumpSource(const std::string& name, DumpSource& source) {
    if (name == "vram") source = DumpSource::Vram;
    else if (name == "scanout") source = DumpSource::Scanout;
    else return false;
    return true;
}

const char* DumpSourceName(DumpSource source) { return source == DumpSource::Vram ? "vram" : "scanout"; }

VramDisplayDump CpuRenderer::DumpVramDisplay() {
    // The CPU SLM is authoritative once earlier draws and pending uploads are committed.
    // This flush also supports diagnostics outside ReplayCapture's EndFrame callback.
    frontEnd_.FlushBatch(FlushReason::VBlank);
    VramDisplayDump out;
    out.pmode = displayRegs_.pmode;
    out.smode1 = displayRegs_.smode1;
    out.smode2 = displayRegs_.smode2;
    out.circuit = (out.pmode & 2u) != 0 ? 2u : (out.pmode & 1u) != 0 ? 1u : 0u;
    if (out.circuit == 0) {
        out.refusal = "no_enabled_circuit";
        return out;
    }
    const auto& circuit = displayRegs_.circuits[out.circuit - 1];
    out.dispfb = circuit.dispfb;
    out.display = circuit.display;
    out.fbp = static_cast<uint32_t>(out.dispfb & 0x1FFu);
    out.fbw = static_cast<uint32_t>((out.dispfb >> 9) & 0x3Fu);
    out.psm = static_cast<uint32_t>((out.dispfb >> 15) & 0x1Fu);
    out.dbx = static_cast<uint32_t>((out.dispfb >> 32) & 0x7FFu);
    out.dby = static_cast<uint32_t>((out.dispfb >> 43) & 0x7FFu);
    const uint32_t magh = static_cast<uint32_t>((out.display >> 23) & 15u) + 1;
    const uint32_t magv = static_cast<uint32_t>((out.display >> 27) & 3u) + 1;
    out.width = (static_cast<uint32_t>((out.display >> 32) & 0xFFFu) + 1) / magh;
    out.height = (static_cast<uint32_t>((out.display >> 44) & 0x7FFu) + 1) / magv;
    out.lineDouble = (out.smode2 & 3u) == 3u;
    // Same bounded geometry accepted by the reference display entry point. Refuse before allocation.
    if (out.width == 0 || out.height == 0 || out.width > 2048 || out.height > 2048) {
        out.refusal = "invalid_dimensions";
        return out;
    }
    // DISPFB colour formats of chapter 01.7. An unsupported format must not silently
    // acquire an unrelated fallback decode, even if the permissive oracle has one.
    if (out.psm != 0 && out.psm != 1 && out.psm != 2 && out.psm != 0xA) {
        out.refusal = "unsupported_display_psm";
        return out;
    }
    const auto psm = memory::decodePsm(out.psm);
    const memory::BufferView view{memory::blockOfPage(out.fbp), out.fbw, *psm};
    out.frame.image = display::Image(out.width, out.height);
    for (uint32_t y = 0; y < out.height; ++y) {
        for (uint32_t x = 0; x < out.width; ++x) {
            // Oracle decode progresses through pages beyond coordinate 2047; only the
            // swizzled local-memory address wraps at 4 MiB. This is not 13.2 scanout wrapping.
            uint32_t value = memory_.Local().readPixel(view, out.dbx + x,
                                                      out.dby + (out.lineDouble ? y / 2 : y));
            if (out.psm == 2 || out.psm == 0xA) {
                value = ((value & 0x1Fu) << 3) | ((value & 0x3E0u) << 6) | ((value & 0x7C00u) << 9);
            }
            auto& pixel = out.frame.image.at(x, y);
            pixel = {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
                     static_cast<uint8_t>(value >> 16), 255};
        }
    }
    out.frame.valid = true;
    return out;
}

std::string EncodeDumpMetadata(uint64_t captureFrame, DumpSource source, const VramDisplayDump& vram,
                               const FrameImage& writtenImage) {
    std::ostringstream json;
    json << "{\"capture_frame\":" << captureFrame << ",\"dump_source\":\"" << DumpSourceName(source)
         << "\",\"image_valid\":" << (writtenImage.valid ? "true" : "false")
         << ",\"image_refusal\":\"" << (writtenImage.valid ? "" :
                source == DumpSource::Vram ? vram.refusal : "scanout_has_no_image") << "\""
         << ",\"image_width\":" << writtenImage.image.width << ",\"image_height\":" << writtenImage.image.height
         << ",\"vram_valid\":" << (vram.frame.valid ? "true" : "false")
         << ",\"vram_refusal\":\"" << vram.refusal << "\",\"selected_circuit\":" << vram.circuit;
    auto raw = [&](const char* name, uint64_t value) {
        json << ",\"" << name << "\":\"0x" << std::hex << std::setw(16) << std::setfill('0') << value
             << "\"" << std::dec;
    };
    raw("pmode", vram.pmode); raw("smode1", vram.smode1); raw("smode2", vram.smode2);
    raw("dispfb", vram.dispfb); raw("display", vram.display);
    json << ",\"fbp_pages\":" << vram.fbp << ",\"fbp_blocks\":" << memory::blockOfPage(vram.fbp)
         << ",\"fbw\":" << vram.fbw << ",\"psm\":" << vram.psm << ",\"dbx\":" << vram.dbx
         << ",\"dby\":" << vram.dby << ",\"vram_width\":" << vram.width << ",\"vram_height\":" << vram.height
         << ",\"line_double\":" << (vram.lineDouble ? "true" : "false") << "}";
    return json.str();
}
}  // namespace coregs::cpuref
