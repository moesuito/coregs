// SPDX-License-Identifier: MIT
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 02, 10.2, 14.3,
// and our MIT GS stream protocol. All capture data here is synthetic.
#include "cpuref_replay.hpp"
#include "hw_gs_registers.hpp"
#include "../../gs_stream/gs_stream_protocol.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace coregs;
using namespace coregs::cpuref;
namespace proto = gs_stream;
int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); ++failures; } } while (0)

template <typename T> void Append(std::vector<uint8_t>& bytes, const T& value) {
    const auto* data = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof value);
}
template <typename T> void Put(std::vector<uint8_t>& bytes, size_t offset, const T& value) {
    std::memcpy(bytes.data() + offset, &value, sizeof value);
}
struct Writer {
    std::vector<uint8_t> bytes;
    uint64_t sequence = 37;  // A capture window need not start with sequence 1.
    explicit Writer(uint64_t first = 0) {
        proto::FileHeader file{};
        std::memcpy(file.magic, proto::kFileMagic, 8);
        file.file_version = 2; file.protocol_major = 1; file.protocol_minor = proto::kVersionMinor;
        file.header_bytes = 64; file.record_header_bytes = 16; file.first_frame = first;
        std::memcpy(file.serial, "SCUS-97328", 10);
        Append(bytes, file);
    }
    template <typename T> void Record(proto::RecordType type, const T& payload, uint16_t flags = 0) {
        Raw(type, &payload, sizeof payload, flags);
    }
    void Raw(proto::RecordType type, const void* data, size_t count, uint16_t flags = 0) {
        const proto::RecordHeader h{static_cast<uint16_t>(type), flags, static_cast<uint32_t>(count), sequence++};
        Append(bytes, h);
        const auto* p = static_cast<const uint8_t*>(data);
        bytes.insert(bytes.end(), p, p + count);
        bytes.resize(bytes.size() + static_cast<size_t>(proto::align_record(count) - count), 0);
    }
    void Gif(uint8_t path, const std::vector<uint8_t>& data, bool more = false) {
        std::vector<uint8_t> payload;
        Append(payload, proto::GifPayload{path, {}, proto::kGifFlagSignalsApplied, static_cast<uint32_t>(data.size()), 0});
        payload.insert(payload.end(), data.begin(), data.end());
        Raw(proto::RecordType::gif, payload.data(), payload.size(), more ? proto::kRecordFlagMore : 0);
    }
    void Priv(uint32_t address, uint64_t value) { Record(proto::RecordType::priv, proto::PrivPayload{address, 0, value}); }
    void Frame(uint64_t frame, uint8_t field = 0, bool valid = false) {
        Record(proto::RecordType::vsync, proto::VsyncPayload{frame, field, {}, valid ? proto::kVsyncFlagFieldValid : 0});
    }
    void Reset(proto::ResetKind kind, uint32_t mask = 0) {
        Record(proto::RecordType::reset, proto::ResetPayload{static_cast<uint32_t>(kind), mask, 0});
    }
};

std::vector<uint8_t> Ad(uint8_t address, uint64_t value) {
    return CaptureWriter::Packet(1, true, false, 0, 0, {0xE}, {value, address});
}
std::vector<uint8_t> Triangle() {
    // REGLIST XYZ2 kicks, as in cpuref_capture_replay: A+D register writes do not kick.
    return CaptureWriter::Packet(3, true, false, 0, 1, {reg::kXyz2},
                                 {0, uint64_t{8 * 16}, uint64_t{8 * 16} << 16});
}
void Register(CpuRenderer& renderer, uint8_t address, uint64_t value) { CHECK(renderer.GifTransfer(1, Ad(address, value))); }
memory::BufferView View() { return {0, 1, memory::Psm::Ct32}; }
void Display(CpuRenderer& renderer) {
    renderer.WritePrivileged(priv::kPmode, 1);
    renderer.WritePrivileged(priv::kSmode1, uint64_t{2} << 13);
    renderer.WritePrivileged(priv::kDispfb1, uint64_t{1} << 9);
    renderer.WritePrivileged(priv::kDisplay1, (uint64_t{31} << 32) | (uint64_t{7} << 44));
}

void Parity() {
    CaptureWriter old;
    Writer v2(7);
    for (const auto& p : std::vector<proto::PrivPayload>{{priv::kPmode, 0, 1},
             {priv::kSmode1, 0, uint64_t{2} << 13}, {priv::kDispfb1, 0, uint64_t{1} << 9},
             {priv::kDisplay1, 0, (uint64_t{31} << 32) | (uint64_t{7} << 44)}}) {
        old.Privileged(p.address, p.value); v2.Priv(p.address, p.value);
    }
    std::vector<std::vector<uint8_t>> packets{Ad(reg::kFrameOne, uint64_t{1} << 16),
        Ad(reg::kZbufOne, uint64_t{1} << 32), Ad(reg::kScissorOne, (uint64_t{7} << 16) | (uint64_t{7} << 48)),
        Ad(reg::kPrim, 3), Ad(reg::kRgbaq, 0x80402010), Triangle()};
    for (const auto& packet : packets) { old.Gif(1, packet); v2.Gif(1, packet); }
    old.FrameEnd(7); v2.Frame(7);
    CpuRenderer a, b;
    ReplaySummary sa, sb;
    std::string error, pa, pb;
    CHECK(ReplayCapture(old.Bytes(), a, sa, [&](uint64_t n, const FrameImage& f) {
        CHECK(n == 7 && f.valid); pa = EncodePpm(f.image);
    }, error));
    CHECK(ReplayCapture(v2.bytes, b, sb, [&](uint64_t n, const FrameImage& f) {
        CHECK(n == 7 && f.valid); pb = EncodePpm(f.image);
    }, error));
    CHECK(pa == pb && !pa.empty());
    CHECK(a.Memory().Local().readPixel(View(), 1, 1) == 0x80402010);
    CHECK(a.Stats().batches == b.Stats().batches && a.Stats().framePixelsWritten == b.Stats().framePixelsWritten);
    CHECK(sa.gifBytes == sb.gifBytes && sa.gifPackets == sb.gifPackets && sa.records == sb.records);
    CHECK(sa.framesWithImage == sb.framesWithImage && sb.fieldsWithoutPhase == 1 && b.DisplayField());

    // Split one tag from its data, preserving each path's parser state.
    Writer split;
    const auto packet = Ad(reg::kPrmodecont, 0);
    split.Gif(1, {packet.begin(), packet.begin() + 16}, true);
    split.Gif(1, {packet.begin() + 16, packet.end()});
    split.Record(proto::RecordType::control, proto::ControlPayload{2, 0, {0, 640, 480}});
    split.Record(proto::RecordType::control, proto::ControlPayload{3, 0, {1280, 720, 0}});
    split.Record(proto::RecordType::control, proto::ControlPayload{6, 0, {1, 3000, 0}});
    split.Frame(0, 1, true); split.Frame(1, 0, true); split.Frame(2);
    CpuRenderer c;
    ReplaySummary sc;
    CHECK(ReplayCapture(split.bytes, c, sc, nullptr, error));
    CHECK(sc.presentationControls == 3 && sc.frames == 3 && sc.fieldsWithoutPhase == 1 && c.DisplayField());
    Register(c, reg::kScissorOne, (uint64_t{7} << 16) | (uint64_t{7} << 48));
}

void Resets() {
    const auto packet = Ad(reg::kRgbaq, 0x80706050);
    Writer stream;
    stream.Gif(1, {packet.begin(), packet.begin() + 16});
    stream.Gif(2, {packet.begin(), packet.begin() + 16});
    stream.Reset(proto::ResetKind::gif_paths, 2);  // PATH1 only, not PATH2.
    stream.Gif(2, {packet.begin() + 16, packet.end()});
    stream.Gif(1, Ad(reg::kFrameOne, uint64_t{1} << 16));
    stream.Gif(1, Ad(reg::kZbufOne, uint64_t{1} << 32));
    stream.Gif(1, Ad(reg::kScissorOne, (uint64_t{7} << 16) | (uint64_t{7} << 48)));
    stream.Gif(1, Ad(reg::kPrim, 3)); stream.Gif(1, Triangle()); stream.Frame(0);
    CpuRenderer r;
    ReplaySummary summary;
    std::string error;
    CHECK(ReplayCapture(stream.bytes, r, summary, nullptr, error));
    CHECK(r.Memory().Local().readPixel(View(), 1, 1) == 0x80706050);
    CHECK(summary.resets == 1);
    Display(r); CHECK(r.EndFrame().valid);
    r.WritePrivileged(priv::kCsr, uint64_t{1} << 9);
    CHECK(r.Memory().Local().readPixel(View(), 1, 1) == 0x80706050); // CSR reset keeps VRAM.
    // Zero SMODE1 decodes as VESA (13.1); both disabled circuits produce black (13.3).
    const auto resetImage = r.EndFrame();
    CHECK(resetImage.valid);
    for (const auto& p : resetImage.image.pixels) CHECK(p.r == 0 && p.g == 0 && p.b == 0);
    Writer hard; hard.Reset(proto::ResetKind::gs_hard); hard.Frame(0);
    ReplaySummary sh;
    CHECK(ReplayCapture(hard.bytes, r, sh, nullptr, error));
    CHECK(r.Memory().Local().readPixel(View(), 1, 1) == 0);
    CHECK(r.DisplayRegs().pmode == 0 && r.DisplayRegs().smode1 == 0);
    // Reenable a known mode with enabled but unavailable source: no old display may leak.
    r.WritePrivileged(priv::kSmode1, uint64_t{2} << 13);
    r.WritePrivileged(priv::kPmode, 1);
    const auto after = r.EndFrame();
    if (after.valid) {
        for (const auto& p : after.image.pixels) CHECK(p.r == 0 && p.g == 0 && p.b == 0);
    }
}

void ArmRead(CpuRenderer& r, uint32_t width = 8) {
    Register(r, reg::kBitbltbuf, uint64_t{1} << 16);
    Register(r, reg::kTrxreg, width | (uint64_t{1} << 32));
    Register(r, reg::kTrxpos, 0);
    Register(r, reg::kTrxdir, 1);
}
void Fifo() {
    CpuRenderer r;
    for (uint32_t x = 0; x < 8; ++x) r.Memory().Local().writePixel(View(), x, 0, 0x80400000u + x);
    ArmRead(r);
    const auto first = r.ReadFifo(1), second = r.ReadFifo(1), past = r.ReadFifo(1);
    CHECK(first.size() == 16 && second.size() == 16 && past == std::vector<uint8_t>(16, 0));
    uint32_t a = 0, b = 0; std::memcpy(&a, first.data(), 4); std::memcpy(&b, second.data(), 4);
    CHECK(a == 0x80400000 && b == 0x80400004 && r.fifoFaults == 1);
    ArmRead(r); auto entire = r.ReadFifo(2);
    CHECK(std::vector<uint8_t>(entire.begin(), entire.begin() + 16) == first);
    CHECK(std::vector<uint8_t>(entire.begin() + 16, entire.end()) == second);
    ArmRead(r); r.ReadFifo(1); Register(r, reg::kTrxdir, 3);
    CHECK(r.ReadFifo(1) == std::vector<uint8_t>(16, 0));
    ArmRead(r); r.ReadFifo(1); r.ResetGif(7);
    CHECK(r.ReadFifo(1) == std::vector<uint8_t>(16, 0));
    ArmRead(r); r.ResetGs(false);
    CHECK(r.ReadFifo(1) == std::vector<uint8_t>(16, 0));

    Writer stream;
    stream.Gif(1, Ad(reg::kBitbltbuf, uint64_t{1} << 16));
    stream.Gif(1, Ad(reg::kTrxreg, 8 | (uint64_t{1} << 32)));
    stream.Gif(1, Ad(reg::kTrxdir, 1));
    stream.Record(proto::RecordType::read_fifo, proto::ReadFifoPayload{1, 0, 1});
    stream.Record(proto::RecordType::read_fifo, proto::ReadFifoPayload{1, 0, 2});
    ReplaySummary summary;
    std::string error;
    CHECK(ReplayCapture(stream.bytes, r, summary, nullptr, error));
    CHECK(summary.fifoReads == 2);

    const auto faults = r.fifoFaults;
    CHECK(r.ReadFifo(UINT32_MAX).empty());
    CHECK(r.fifoFaults == faults + 1);
    CaptureWriter oversized;
    oversized.FifoRead(UINT32_MAX);
    ReplaySummary oversizedSummary;
    CHECK(!ReplayCapture(oversized.Bytes(), r, oversizedSummary, nullptr, error));
    CHECK(error.find("oversized FIFO") != std::string::npos && oversizedSummary.fifoReads == 0);

    // An armed read stages on first access, after a subsequently queued draw is committed.
    CpuRenderer pending;
    ArmRead(pending, 4);
    Register(pending, reg::kTrxpos, 1 | (uint64_t{1} << 16));
    Register(pending, reg::kTrxdir, 1);
    Register(pending, reg::kFrameOne, uint64_t{1} << 16);
    Register(pending, reg::kZbufOne, uint64_t{1} << 32);
    Register(pending, reg::kScissorOne, (uint64_t{7} << 16) | (uint64_t{7} << 48));
    Register(pending, reg::kPrim, 3);
    Register(pending, reg::kRgbaq, 0x80402010);
    CHECK(pending.GifTransfer(1, Triangle()));
    const auto drawn = pending.ReadFifo(1);
    uint32_t pixel = 0; std::memcpy(&pixel, drawn.data(), sizeof pixel);
    CHECK(pixel == 0x80402010 && pending.Stats().framePixelsWritten > 0);
}

void Malformed() {
    Writer good; good.Frame(7, 1, true); good.Frame(8, 0, true);
    auto reject = [](const std::vector<uint8_t>& bytes) {
        CpuRenderer r; ReplaySummary summary; std::string error;
        CHECK(!ReplayCapture(bytes, r, summary, nullptr, error) && !error.empty());
    };
    for (size_t n = 8; n < good.bytes.size(); ++n) {
        if (n == 64 || n == 96) continue; // Complete prefixes are valid streams.
        reject({good.bytes.begin(), good.bytes.begin() + static_cast<std::ptrdiff_t>(n)});
    }
    for (const auto [at, value] : std::vector<std::pair<size_t, uint8_t>>{
            {8, 3}, {10, 2}, {12, static_cast<uint8_t>(proto::kVersionMinor + 1)}, {14, 63}, {16, 15}, {20, 1}, {48, 1},
            {64, 0}, {66, 2}, {68, 15}, {72, 0}, {88, 2}, {89, 1}, {92, 2}, {104, 99}}) {
        auto copy = good.bytes; copy[at] = value; reject(copy);
    }
    auto unordered = good.bytes; Put(unordered, 112, uint64_t{7}); reject(unordered);
    auto huge = good.bytes; Put(huge, 68, UINT32_MAX); reject(huge);
    Writer continuation; const auto packet = Ad(reg::kPrim, 0);
    continuation.Gif(1, {packet.begin(), packet.begin() + 16}, true); reject(continuation.bytes);
    continuation.Frame(0); reject(continuation.bytes);
    Writer wrongPath; wrongPath.Gif(1, {packet.begin(), packet.begin() + 16}, true);
    wrongPath.Gif(2, {packet.begin() + 16, packet.end()}); reject(wrongPath.bytes);
    Writer pathZero; pathZero.Gif(0, packet); reject(pathZero.bytes);
    Writer badPriv; badPriv.Priv(priv::kPmode + 1, 0); reject(badPriv.bytes);
    Writer unsupported; unsupported.Record(proto::RecordType::control, proto::ControlPayload{4, 0, {}}); reject(unsupported.bytes);
    Writer reserved; reserved.Record(proto::RecordType::state_load, uint64_t{0}); reject(reserved.bytes);
    Writer padding; padding.Raw(proto::RecordType::vsync, packet.data(), 15);
    padding.bytes.back() = 1; reject(padding.bytes);
    Writer reset; reset.Reset(proto::ResetKind::gif_paths, 1); reject(reset.bytes);
}
}  // namespace

int main(int argc, char** argv) {
    const std::string name = argc == 2 ? argv[1] : "";
    if (name == "parity") Parity();
    else if (name == "resets") Resets();
    else if (name == "fifo") Fifo();
    else if (name == "malformed") Malformed();
    else return 2;
    return failures == 0 ? 0 : 1;
}
