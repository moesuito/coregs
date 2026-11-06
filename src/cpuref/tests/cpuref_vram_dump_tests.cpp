// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 12.9, 13 and 14.2-14.4.
// Synthetic oracle-decoded display buffers, distinct from composed scanout.
#include "cpuref_replay.hpp"
#include "hw_gs_types.hpp"

#include <cstdio>
#include <string>

namespace {
using namespace coregs;
using namespace coregs::cpuref;
int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); ++failures; } } while (0)

uint64_t Display(uint32_t width, uint32_t height, uint32_t magh = 1, uint32_t magv = 1) {
    return (uint64_t{magh - 1} << 23) | (uint64_t{magv - 1} << 27) |
           (uint64_t{width * magh - 1} << 32) | (uint64_t{height * magv - 1} << 44);
}
void Circuit(CpuRenderer& r, uint32_t index, uint32_t page, uint32_t psm, uint32_t width, uint32_t height,
             uint32_t dbx = 0, uint32_t dby = 0, uint32_t bw = 1) {
    r.DisplayRegs().circuits[index].dispfb = page | (uint64_t{bw} << 9) | (uint64_t{psm} << 15) |
                                            (uint64_t{dbx} << 32) | (uint64_t{dby} << 43);
    r.DisplayRegs().circuits[index].display = Display(width, height);
}
uint32_t Rgb(const display::Rgba8& pixel) { return pixel.r | (uint32_t{pixel.g} << 8) | (uint32_t{pixel.b} << 16); }
memory::BufferView View(uint32_t page = 0, memory::Psm psm = memory::Psm::Ct32, uint32_t bw = 1) {
    return {memory::blockOfPage(page), bw, psm};
}

void Selection() {
    CpuRenderer r;
    r.WritePrivileged(priv::kBgcolor, 0x123456);
    auto none = r.DumpVramDisplay();
    CHECK(!none.frame.valid && none.circuit == 0 && none.refusal == "no_enabled_circuit");
    CHECK(none.frame.image.pixels.empty());
    Circuit(r, 0, 0, 0, 2, 2); Circuit(r, 1, 1, 0, 3, 2);
    r.Memory().Local().writePixel(View(0), 0, 0, 0x00112233);
    r.Memory().Local().writePixel(View(1), 0, 0, 0xFF445566);
    r.DisplayRegs().pmode = 1;
    auto first = r.DumpVramDisplay();
    CHECK(first.frame.valid && first.circuit == 1 && first.width == 2 && first.height == 2);
    CHECK(Rgb(first.frame.image.at(0, 0)) == 0x112233);
    for (uint64_t enables : {uint64_t{2}, uint64_t{3}}) {
        r.DisplayRegs().pmode = enables;
        const auto second = r.DumpVramDisplay();
        CHECK(second.frame.valid && second.circuit == 2 && second.fbp == 1 && second.width == 3);
        CHECK(Rgb(second.frame.image.at(0, 0)) == 0x445566);
    }
    // Selection is not PCRTC disabled-circuit filtering: the oracle accepts zero DISPLAY as 1x1.
    r.DisplayRegs().circuits[1] = {};
    CHECK(r.DumpVramDisplay().frame.valid && r.DumpVramDisplay().width == 1);
    Circuit(r, 1, 0, 3, 2, 2);
    const auto unknown = r.DumpVramDisplay();
    CHECK(!unknown.frame.valid && unknown.refusal == "unsupported_display_psm");
    CHECK(unknown.frame.image.pixels.empty());
    Circuit(r, 1, 0, 0, 4096, 1);
    const auto large = r.DumpVramDisplay();
    CHECK(!large.frame.valid && large.width == 4096 && large.refusal == "invalid_dimensions");
    r.DisplayRegs().circuits[1].display = uint64_t{15} << 23; // (DW+1)/MAGH = 1/16 = 0.
    CHECK(!r.DumpVramDisplay().frame.valid && r.DumpVramDisplay().width == 0);
}

void Formats() {
    CpuRenderer r;
    r.DisplayRegs().pmode = 1;
    for (uint32_t psm : {0u, 1u}) {
        Circuit(r, 0, 0, psm, 2, 1);
        r.Memory().Local().writePixel(View(), 0, 0, 0x00112233);
        r.Memory().Local().writePixel(View(), 1, 0, 0xFF112233);
        const auto raw = r.DumpVramDisplay();
        CHECK(raw.frame.valid);
        CHECK(Rgb(raw.frame.image.at(0, 0)) == 0x112233 && Rgb(raw.frame.image.at(1, 0)) == 0x112233);
        CHECK(raw.frame.image.at(0, 0).a == 255 && raw.frame.image.at(1, 0).a == 255);
        CHECK(EncodePpm(raw.frame.image).substr(11) == std::string("\x33\x22\x11\x33\x22\x11", 6));
    }
    for (const auto psm : {memory::Psm::Ct16, memory::Psm::Ct16S}) {
        Circuit(r, 0, 0, psm == memory::Psm::Ct16 ? 2 : 0xA, 2, 1);
        r.Memory().Local().writePixel(View(0, psm), 0, 0, 1 | (2u << 5) | (31u << 10));
        r.Memory().Local().writePixel(View(0, psm), 1, 0, 0x8000 | 1 | (2u << 5) | (31u << 10));
        const auto raw = r.DumpVramDisplay();
        CHECK(raw.frame.valid);
        CHECK(Rgb(raw.frame.image.at(0, 0)) == 0xF81008 && Rgb(raw.frame.image.at(1, 0)) == 0xF81008);
    }
}

void Offsets() {
    CpuRenderer r;
    r.DisplayRegs().pmode = 1;
    Circuit(r, 0, 0, 0, 2, 2, 12, 7);
    r.Memory().Local().writePixel(View(), 12, 7, 0x00112233);
    r.Memory().Local().writePixel(View(), 13, 8, 0x00445566);
    auto raw = r.DumpVramDisplay();
    CHECK(raw.dbx == 12 && raw.dby == 7 && Rgb(raw.frame.image.at(0, 0)) == 0x112233);
    CHECK(Rgb(raw.frame.image.at(1, 1)) == 0x445566);

    // Oracle coordinate progression continues past 2047, rather than resetting X to zero.
    Circuit(r, 0, 0, 0, 2, 1, 2047, 0, 2);
    r.Memory().Local().writePixel(View(0, memory::Psm::Ct32, 2), 2047, 0, 0x00010203);
    r.Memory().Local().writePixel(View(0, memory::Psm::Ct32, 2), 2048, 0, 0x00040506);
    raw = r.DumpVramDisplay();
    CHECK(Rgb(raw.frame.image.at(0, 0)) == 0x010203 && Rgb(raw.frame.image.at(1, 0)) == 0x040506);
    Circuit(r, 0, 511, 0, 1, 1, 64, 0);
    r.Memory().Local().writePixel(View(), 0, 0, 0x00778899); // page511 + X page1 wraps to page0.
    CHECK(Rgb(r.DumpVramDisplay().frame.image.at(0, 0)) == 0x778899);
}

void LineDouble() {
    CpuRenderer r;
    r.DisplayRegs().pmode = 1;
    Circuit(r, 0, 0, 0, 2, 4, 0, 5);
    r.Memory().Local().writePixel(View(), 0, 5, 0x00112233);
    r.Memory().Local().writePixel(View(), 0, 6, 0x00445566);
    r.Memory().Local().writePixel(View(), 0, 7, 0x00778899);
    r.DisplayRegs().smode2 = 3;
    auto raw = r.DumpVramDisplay();
    CHECK(raw.lineDouble && raw.width == 2 && raw.height == 4);
    CHECK(Rgb(raw.frame.image.at(0, 0)) == 0x112233 && Rgb(raw.frame.image.at(0, 1)) == 0x112233);
    CHECK(Rgb(raw.frame.image.at(0, 2)) == 0x445566 && Rgb(raw.frame.image.at(0, 3)) == 0x445566);
    r.SetDisplayField(true);
    CHECK(EncodePpm(r.DumpVramDisplay().frame.image) == EncodePpm(raw.frame.image)); // No guessed phase shift.
    r.DisplayRegs().smode2 = 1;
    raw = r.DumpVramDisplay();
    CHECK(!raw.lineDouble && Rgb(raw.frame.image.at(0, 1)) == 0x445566 && Rgb(raw.frame.image.at(0, 2)) == 0x778899);
    Circuit(r, 0, 0, 0, 640, 448);
    r.DisplayRegs().circuits[0].display = Display(640, 448, 4, 1);
    r.DisplayRegs().smode2 = 3;
    raw = r.DumpVramDisplay();
    CHECK(raw.frame.valid && raw.width == 640 && raw.height == 448 && raw.lineDouble);
}

void Independence() {
    CpuRenderer r;
    Circuit(r, 0, 0, 0, 2, 2); Circuit(r, 1, 1, 0, 2, 2);
    r.DisplayRegs().smode1 = uint64_t{2} << 13;
    // NTSC's horizontal clock divider is 4 (13.1); MAGH=3 and DW=7 yield
    // two real source/display pixels in both the raw decode and PCRTC path (13.2).
    r.DisplayRegs().circuits[0].display = Display(2, 2, 4, 1);
    r.DisplayRegs().circuits[1].display = Display(2, 2, 4, 1);
    r.DisplayRegs().pmode = 3 | (uint64_t{1} << 5) | (uint64_t{255} << 8); // RC1 opaque over RC2.
    r.Memory().Local().writePixel(View(), 0, 0, 0xFF0000FF);
    r.Memory().Local().writePixel(View(1), 0, 0, 0xFFFF0000);
    const auto scanout = r.EndFrame();
    CHECK(scanout.valid && Rgb(scanout.image.at(0, 0)) == 0x0000FF);
    const auto raw = r.DumpVramDisplay();
    CHECK(raw.frame.valid && Rgb(raw.frame.image.at(0, 0)) == 0xFF0000);
    CHECK(EncodePpm(r.EndFrame().image) == EncodePpm(scanout.image));
    r.DisplayRegs().smode1 = uint64_t{1} << 13; // Unknown PCRTC mode still has a decodable memory buffer.
    CHECK(!r.EndFrame().valid && r.DumpVramDisplay().frame.valid);

    // Direct diagnostics must commit queued work, even without EndFrame.
    CpuRenderer pending;
    Circuit(pending, 0, 0, 0, 4, 4); pending.DisplayRegs().pmode = 1;
    auto ad = [&](uint8_t address, uint64_t value) {
        CHECK(pending.GifTransfer(1, CaptureWriter::Packet(1, true, false, 0, 0, {0xE}, {value, address})));
    };
    ad(reg::kFrameOne, uint64_t{1} << 16); ad(reg::kZbufOne, uint64_t{1} << 32);
    ad(reg::kScissorOne, (uint64_t{7} << 16) | (uint64_t{7} << 48)); ad(reg::kPrim, 3);
    ad(reg::kRgbaq, 0x80402010);
    CHECK(pending.GifTransfer(1, CaptureWriter::Packet(3, true, false, 0, 1, {reg::kXyz2},
                                                     {0, 128, uint64_t{128} << 16})));
    CHECK(pending.Stats().framePixelsWritten == 0);
    CHECK(Rgb(pending.DumpVramDisplay().frame.image.at(1, 1)) == 0x402010);
    CHECK(pending.Stats().framePixelsWritten > 0);
}

void Metadata() {
    DumpSource source = DumpSource::Vram;
    CHECK(ParseDumpSource("scanout", source) && source == DumpSource::Scanout);
    CHECK(ParseDumpSource("vram", source) && source == DumpSource::Vram);
    CHECK(!ParseDumpSource("VRAM", source) && !ParseDumpSource("", source) && !ParseDumpSource("unknown", source));
    CpuRenderer r;
    Circuit(r, 1, 140, 1, 640, 448, 5, 6, 10); r.DisplayRegs().pmode = 3; r.DisplayRegs().smode2 = 3;
    const auto raw = r.DumpVramDisplay();
    const std::string json = EncodeDumpMetadata(99, source, raw, raw.frame);
    for (const char* field : {"\"capture_frame\":99", "\"dump_source\":\"vram\"", "\"selected_circuit\":2",
                             "\"fbp_pages\":140", "\"fbp_blocks\":4480", "\"fbw\":10", "\"psm\":1",
                             "\"dbx\":5", "\"dby\":6", "\"vram_width\":640", "\"line_double\":true",
                             "\"image_valid\":true", "\"pmode\":\"0x0000000000000003\""}) CHECK(json.find(field) != std::string::npos);
    r.DisplayRegs().pmode = 0;
    const auto none = r.DumpVramDisplay();
    const auto invalid = EncodeDumpMetadata(100, source, none, none.frame);
    CHECK(invalid.find("\"capture_frame\":100") != std::string::npos);
    CHECK(invalid.find("\"image_valid\":false") != std::string::npos && invalid.find("no_enabled_circuit") != std::string::npos);
}
}  // namespace

int main(int argc, char** argv) {
    const std::string name = argc == 2 ? argv[1] : "";
    if (name == "selection") Selection(); else if (name == "formats") Formats();
    else if (name == "offsets") Offsets(); else if (name == "line_double") LineDouble();
    else if (name == "independence") Independence(); else if (name == "metadata") Metadata();
    else return 2;
    return failures == 0 ? 0 : 1;
}
