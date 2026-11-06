// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec chapter 11 (TASK-027), clean room (decision 0070).
//
// The worker thread of texture replacement (11.5, 11.6): reads and decodes replacement files, and writes texture dumps,
// so the GS worker never waits for the disk. The renderer asks for an image by its own id and takes finished images at
// VBlank; until then the batch draws with the texture decoded from GS memory.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hw_gs_replace/packs.hpp"

namespace coregs::core2 {

class ReplacementLoader {
public:
    struct Loaded {
        std::uint64_t id = 0;
        std::uint32_t width = 0, height = 0;
        std::vector<std::uint8_t> rgba;  // width * height * 4 bytes, rows top to bottom (the words of a texture image)
        std::string error;               // not empty: the file could not be used
    };

    ReplacementLoader();
    ~ReplacementLoader();
    ReplacementLoader(const ReplacementLoader &) = delete;
    ReplacementLoader &operator=(const ReplacementLoader &) = delete;

    void Request(std::uint64_t id, hw_gs_replace::PackEntry entry);
    // One finished image, without waiting. With `wait` it waits until a requested image is finished.
    bool Take(Loaded &out, bool wait = false);
    // IN-533: one finished image, waiting at most `limit` for one when none is finished yet.
    bool TakeWithin(Loaded &out, std::chrono::microseconds limit);
    // Writes an RGBA8 PNG (chapter 11.6), keeping an existing file.
    void Dump(std::filesystem::path file, std::uint32_t width, std::uint32_t height, std::vector<std::uint8_t> rgba);
    // Waits until every queued dump is written (fixtures).
    void WaitForDumps();

private:
    struct Job {
        std::uint64_t id = 0;
        hw_gs_replace::PackEntry entry;
    };
    struct DumpJob {
        std::filesystem::path file;
        std::uint32_t width = 0, height = 0;
        std::vector<std::uint8_t> rgba;
    };
    void Run();

    std::mutex mutex_;
    std::condition_variable wake_, finished_;
    std::deque<Job> jobs_;
    std::deque<DumpJob> dumps_;
    std::deque<Loaded> done_;
    std::size_t outstanding_ = 0;  // requested and not taken
    bool busyDumping_ = false;
    bool stop_ = false;
    std::thread thread_;
};

}  // namespace coregs::core2
