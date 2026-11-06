// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
//
// coregs_cpu_replay: replays a legacy or GSSTREAM v2 capture through the CPU reference renderer and writes
// one frame_NNNNN.ppm per frame that has a display image.
//
// Usage: coregs_cpu_replay <capture.gscap> <output-dir> [--dump-source vram|scanout] [--slm-frame N]
//        [--slm-hashes <file>]
// --slm-hashes writes "<frame> <16 hex digits>" per frame: the FNV-1a hash of the 4 MiB local memory.
// Default vram is the 1x reference image; scanout is the composed presentation image.
// frames.jsonl records every capture-native frame; image filenames are never renumbered.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "cpuref_replay.hpp"

int main(int argc, char** argv) {
    auto usage = [] {
        std::fprintf(stderr, "usage: coregs_cpu_replay <capture> <output-dir> [--dump-source vram|scanout]\n"
                             "default: vram (12.9 oracle decode); scanout: existing composed output\n"
                             "metadata: <output-dir>/frames.jsonl, capture-native frame IDs\n");
    };
    coregs::cpuref::DumpSource source = coregs::cpuref::DumpSource::Vram;

    std::uint64_t slmFrame = UINT64_MAX;  // raw shadow local memory dump
    std::string slmHashPath;              // per-frame local memory hashes
    bool selectedSource = false;
    std::vector<std::string> paths;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" && argc == 2) { usage(); return 0; }
        if (arg == "--slm-frame") {
            if (i + 1 == argc) {
                std::fprintf(stderr, "hw_cpuref_replay: --slm-frame requires a frame number\n");
                return 2;
            }
            slmFrame = std::strtoull(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--slm-hashes") {
            if (i + 1 == argc) {
                std::fprintf(stderr, "hw_cpuref_replay: --slm-hashes requires a file name\n");
                return 2;
            }
            slmHashPath = argv[++i];
            continue;
        }
        if (arg == "--dump-source") {
            if (selectedSource || i + 1 == argc || !coregs::cpuref::ParseDumpSource(argv[++i], source)) {
                std::fprintf(stderr, "hw_cpuref_replay: --dump-source requires one of vram|scanout, once\n");
                return 2;
            }
            selectedSource = true;
        } else if (arg.starts_with("--")) {
            std::fprintf(stderr, "hw_cpuref_replay: unknown option %s\n", arg.c_str());
            return 2;
        } else paths.push_back(arg);
    }
    if (paths.size() != 2) {
        usage();
        return 2;
    }
    std::ifstream input(paths[0], std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "hw_cpuref_replay: cannot open %s\n", paths[0].c_str());
        return 1;
    }
    const std::vector<uint8_t> stream((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::error_code ec;
    std::filesystem::create_directories(paths[1], ec);
    if (ec) {
        std::fprintf(stderr, "hw_cpuref_replay: cannot create %s: %s\n", paths[1].c_str(), ec.message().c_str());
        return 1;
    }
    std::ofstream metadata(std::filesystem::path(paths[1]) / "frames.jsonl", std::ios::binary);
    if (!metadata) {
        std::fprintf(stderr, "hw_cpuref_replay: cannot create frames.jsonl\n");
        return 1;
    }

    std::FILE* slmHashes = nullptr;
    if (!slmHashPath.empty()) {
        slmHashes = std::fopen(slmHashPath.c_str(), "w");
        if (!slmHashes) {
            std::fprintf(stderr, "hw_cpuref_replay: cannot create %s\n", slmHashPath.c_str());
            return 1;
        }
    }

    coregs::cpuref::CpuRenderer renderer;
    coregs::cpuref::ReplaySummary summary;
    std::string error;
    bool writeFailed = false;
    uint64_t imagesWritten = 0;
    uint64_t framesWithoutImage = 0;
    const bool ok = coregs::cpuref::ReplayCapture(
        stream, renderer, summary,
        [&](uint64_t frame, const coregs::cpuref::FrameImage& scanout) {
            if (slmHashes) {
                const auto bytes = renderer.Memory().Local().bytes();
                std::fprintf(slmHashes, "%llu %016llx\n", static_cast<unsigned long long>(frame),
                             static_cast<unsigned long long>(coregs::cpuref::HashLocalMemory(bytes.data(), bytes.size())));
            }
            if (frame == slmFrame) {
                char slmName[64];
                std::snprintf(slmName, sizeof(slmName), "slm_%05llu.bin", static_cast<unsigned long long>(frame));
                std::FILE* slm = std::fopen((paths[1] + "/" + slmName).c_str(), "wb");
                if (slm) {
                    const auto bytes = renderer.Memory().Local().bytes();
                    std::fwrite(bytes.data(), 1, bytes.size(), slm);
                    std::fclose(slm);
                }
            }
            const auto vram = renderer.DumpVramDisplay();
            const auto& image = source == coregs::cpuref::DumpSource::Vram ? vram.frame : scanout;
            metadata << coregs::cpuref::EncodeDumpMetadata(frame, source, vram, image) << '\n';
            if (!metadata) writeFailed = true;
            if (!image.valid) {
                ++framesWithoutImage;
                return;
            }
            char name[64];
            std::snprintf(name, sizeof(name), "frame_%05llu.ppm", static_cast<unsigned long long>(frame));
            const std::filesystem::path path = std::filesystem::path(paths[1]) / name;
            std::ofstream out(path, std::ios::binary);
            const std::string ppm = coregs::cpuref::EncodePpm(image.image);
            out.write(ppm.data(), static_cast<std::streamsize>(ppm.size()));
            if (!out) {
                writeFailed = true;
            } else ++imagesWritten;
        },
        error);

    if (slmHashes && std::fclose(slmHashes) != 0) writeFailed = true;
    metadata.flush();
    if (!metadata) writeFailed = true;
    std::printf("dump_source=%s images_written=%llu frames_without_image=%llu\n", coregs::cpuref::DumpSourceName(source),
                static_cast<unsigned long long>(imagesWritten), static_cast<unsigned long long>(framesWithoutImage));
    std::printf("records=%llu gif_packets=%llu gif_bytes=%llu privileged=%llu fifo_reads=%llu frames=%llu scanout_images=%llu\n",
                static_cast<unsigned long long>(summary.records), static_cast<unsigned long long>(summary.gifPackets),
                static_cast<unsigned long long>(summary.gifBytes), static_cast<unsigned long long>(summary.privileged),
                static_cast<unsigned long long>(summary.fifoReads), static_cast<unsigned long long>(summary.frames),
                static_cast<unsigned long long>(summary.framesWithImage));
    const coregs::cpuref::DrawStats& stats = renderer.Stats();
    std::printf("batches=%llu primitives=%llu covered_pixels=%llu frame_writes=%llu depth_writes=%llu\n",
                static_cast<unsigned long long>(stats.batches), static_cast<unsigned long long>(stats.primitives),
                static_cast<unsigned long long>(stats.coveredPixels),
                static_cast<unsigned long long>(stats.framePixelsWritten),
                static_cast<unsigned long long>(stats.depthPixelsWritten));
    std::printf("textured_primitives=%llu skipped_texture=%llu rejected_gif=%llu unhandled_privileged=%llu\n", static_cast<unsigned long long>(stats.primitivesTextured), static_cast<unsigned long long>(stats.primitivesSkippedTexture), static_cast<unsigned long long>(renderer.rejectedGifPaths), static_cast<unsigned long long>(renderer.unhandledPrivileged));
    std::printf("resets=%llu presentation_controls=%llu fields_without_phase=%llu fifo_faults=%llu\n",
                static_cast<unsigned long long>(summary.resets),
                static_cast<unsigned long long>(summary.presentationControls),
                static_cast<unsigned long long>(summary.fieldsWithoutPhase),
                static_cast<unsigned long long>(renderer.fifoFaults));
    if (!ok) {
        std::fprintf(stderr, "hw_cpuref_replay: %s\n", error.c_str());
        return 1;
    }
    if (writeFailed) {
        std::fprintf(stderr, "hw_cpuref_replay: a frame or its metadata could not be written\n");
        return 1;
    }
    return 0;
}
