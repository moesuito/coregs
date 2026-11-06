// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 02, 10.2, 14.3,
// and our Apache-2.0 GS stream protocol (docs/GS-STREAM-PROTOCOL.md sections 2, 4, 6, 10).
#include "cpuref_replay.hpp"

#include "../../gs_stream/gs_stream_protocol.h"

#include <bit>
#include <cstring>
#include <limits>

namespace coregs::cpuref {
namespace {
namespace proto = gs_stream;
static_assert(std::endian::native == std::endian::little, "GS stream structs require a little-endian host");

bool Zero(const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) if (p[i] != 0) return false;
    return true;
}

template <typename T>
T Load(const uint8_t* data) {
    T value{};
    std::memcpy(&value, data, sizeof value);
    return value;
}
}  // namespace

bool ReplayCaptureV2(const std::vector<uint8_t>& stream, ReplayTarget& renderer, ReplaySummary& summary,
                     const std::function<void(uint64_t, const FrameImage&)>& onFrame, std::string& error) {
    auto fail = [&](size_t offset, const char* reason) {
        error = std::string(reason) + " at byte " + std::to_string(offset);
        return false;
    };
    if (stream.size() < sizeof(proto::FileHeader)) return fail(0, "truncated v2 file header");
    const auto file = Load<proto::FileHeader>(stream.data());
    if (std::memcmp(file.magic, proto::kFileMagic, 8) != 0 || file.file_version != proto::kFileVersion ||
        file.protocol_major != proto::kVersionMajor || file.protocol_minor > proto::kVersionMinor ||
        file.header_bytes != sizeof(proto::FileHeader) || file.record_header_bytes != sizeof(proto::RecordHeader) ||
        file.flags != 0 || !Zero(file.reserved, sizeof file.reserved)) {
        return fail(0, "unsupported or malformed v2 file header");
    }
    bool serialEnded = false;
    for (char c : file.serial) {
        const auto byte = static_cast<uint8_t>(c);
        if (byte == 0) serialEnded = true;
        else if (serialEnded || byte < 0x20 || byte > 0x7E) return fail(24, "malformed serial");
    }
    if (file.serial[0] == 0) return fail(24, "empty serial");

    size_t offset = sizeof file;
    uint64_t lastSequence = 0;
    uint64_t lastFrame = 0;
    bool haveFrame = false;
    bool moreGif = false;
    uint8_t splitPath = 0;
    uint32_t splitFlags = 0;
    while (offset < stream.size()) {
        if (stream.size() - offset < sizeof(proto::RecordHeader)) return fail(offset, "truncated v2 record header");
        const auto header = Load<proto::RecordHeader>(stream.data() + offset);
        if (header.sequence == 0 || (lastSequence != 0 &&
            (lastSequence == std::numeric_limits<uint64_t>::max() || header.sequence != lastSequence + 1))) {
            return fail(offset, "noncontiguous v2 sequence");
        }
        if ((header.flags & ~proto::kRecordFlagMore) != 0) return fail(offset, "unknown v2 record flags");
        const uint64_t padded = proto::align_record(header.payload_bytes);
        const size_t payloadStart = offset + sizeof header;
        if (padded > stream.size() - payloadStart) return fail(offset, "truncated v2 payload or padding");
        const uint8_t* data = stream.data() + payloadStart;
        if (!Zero(data + header.payload_bytes, static_cast<size_t>(padded - header.payload_bytes))) {
            return fail(offset, "nonzero v2 padding");
        }
        const auto type = static_cast<proto::RecordType>(header.type);
        if ((moreGif || header.flags != 0) && type != proto::RecordType::gif) {
            return fail(offset, "invalid split continuation");
        }
        // Every implemented command except GIF has a fixed payload. Reject unsupported commands;
        // no semantic control or future record may disappear silently from a reference replay.
        switch (type) {
            case proto::RecordType::gif: {
                if (header.payload_bytes < sizeof(proto::GifPayload)) return fail(offset, "short GIF payload");
                const auto p = Load<proto::GifPayload>(data);
                if (p.path < 1 || p.path > 3 || !Zero(p.reserved0, sizeof p.reserved0) || p.reserved1 != 0 ||
                    (p.flags & ~proto::kGifFlagSignalsApplied) != 0 || p.bytes % 16 != 0 ||
                    p.bytes != header.payload_bytes - sizeof p) return fail(offset, "malformed GIF payload");
                if (moreGif && (p.path != splitPath || p.flags != splitFlags)) {
                    return fail(offset, "changed GIF continuation path or flags");
                }
                const std::vector<uint8_t> bytes(data + sizeof p, data + sizeof p + p.bytes);
                if (!renderer.GifTransfer(p.path, bytes)) return fail(offset, "rejected v2 GIF transfer");
                ++summary.gifPackets;
                summary.gifBytes += p.bytes;
                moreGif = (header.flags & proto::kRecordFlagMore) != 0;
                splitPath = p.path;
                splitFlags = p.flags;
                break;
            }
            case proto::RecordType::priv: {
                if (header.payload_bytes != sizeof(proto::PrivPayload)) return fail(offset, "wrong PRIV size");
                const auto p = Load<proto::PrivPayload>(data);
                if (p.reserved != 0 || p.address < proto::kPrivilegedBase ||
                    p.address > proto::kPrivilegedBase + proto::kPrivilegedBytes - 8 || (p.address & 7u) != 0) {
                    return fail(offset, "malformed PRIV payload");
                }
                renderer.WritePrivileged(p.address, p.value);
                ++summary.privileged;
                break;
            }
            case proto::RecordType::vsync: {
                if (header.payload_bytes != sizeof(proto::VsyncPayload)) return fail(offset, "wrong VSYNC size");
                const auto p = Load<proto::VsyncPayload>(data);
                if (p.field > 1 || !Zero(p.reserved0, sizeof p.reserved0) ||
                    (p.flags & ~proto::kVsyncFlagFieldValid) != 0 || p.frame < file.first_frame ||
                    (haveFrame && p.frame <= lastFrame)) return fail(offset, "malformed VSYNC payload or frame order");
                if ((p.flags & proto::kVsyncFlagFieldValid) != 0) renderer.SetDisplayField(p.field != 0);
                else {
                    // Protocol 6.1 requires alternation but does not specify its initial phase.
                    // This stored phase does not alter the current deinterlace-off CPU image.
                    renderer.SetDisplayField(!renderer.DisplayField());
                    ++summary.fieldsWithoutPhase;
                }
                const FrameImage image = renderer.EndFrame();
                ++summary.frames;
                if (image.valid) ++summary.framesWithImage;
                if (onFrame) onFrame(p.frame, image);
                lastFrame = p.frame;
                haveFrame = true;
                break;
            }
            case proto::RecordType::read_fifo: {
                if (header.payload_bytes != sizeof(proto::ReadFifoPayload)) return fail(offset, "wrong READ_FIFO size");
                const auto p = Load<proto::ReadFifoPayload>(data);
                if (p.reserved != 0 || p.request_id == 0 ||
                    p.qwords > CpuRenderer::kMaxFifoQwords) return fail(offset, "malformed READ_FIFO payload");
                renderer.ReadFifo(p.qwords);
                ++summary.fifoReads;
                break;
            }
            case proto::RecordType::reset: {
                if (header.payload_bytes != sizeof(proto::ResetPayload)) return fail(offset, "wrong RESET size");
                const auto p = Load<proto::ResetPayload>(data);
                if (p.reserved != 0) return fail(offset, "nonzero RESET reserved field");
                if (p.kind == static_cast<uint32_t>(proto::ResetKind::gs_hard) && p.path_mask == 0) renderer.ResetGs(true);
                else if (p.kind == static_cast<uint32_t>(proto::ResetKind::gif_paths) && (p.path_mask & ~0xEu) == 0) {
                    renderer.ResetGif(p.path_mask >> 1);
                } else return fail(offset, "unsupported or malformed RESET");
                ++summary.resets;
                break;
            }
            case proto::RecordType::control: {
                if (header.payload_bytes != sizeof(proto::ControlPayload)) return fail(offset, "wrong CONTROL size");
                const auto p = Load<proto::ControlPayload>(data);
                if (p.reserved != 0) return fail(offset, "nonzero CONTROL reserved field");
                const auto kind = static_cast<proto::ControlKind>(p.kind);
                if (kind == proto::ControlKind::resize) {
                    if (p.args[0] > UINT32_MAX || p.args[1] > UINT32_MAX || p.args[2] != 0) return fail(offset, "malformed resize");
                } else if (kind == proto::ControlKind::attach_window) {
                    if (p.args[1] > UINT32_MAX || p.args[2] > UINT32_MAX) return fail(offset, "malformed attach_window");
                } else if (kind == proto::ControlKind::cursor && file.protocol_minor >= 2) {
                    if ((p.args[0] & ~proto::kCursorFlagAutoHide) != 0 || p.args[1] < 1 || p.args[1] > 60000 ||
                        p.args[2] != 0) return fail(offset, "malformed cursor");
                } else return fail(offset, "unsupported semantic CONTROL in CPU replay");
                ++summary.presentationControls;
                break;
            }
            default: return fail(offset, "unsupported v2 record type in CPU replay");
        }
        ++summary.records;
        lastSequence = header.sequence;
        offset = payloadStart + static_cast<size_t>(padded);
    }
    if (moreGif) return fail(offset, "unfinished GIF continuation");
    return true;
}
}  // namespace coregs::cpuref
