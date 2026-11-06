// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec chapter 11 (TASK-027), clean room (decision 0070).
#include "core2_replace.hpp"

#include <fstream>
#include <system_error>
#include <utility>

#include "hw_gs_replace/images.hpp"
#include "png_encode.hpp"

namespace coregs::core2 {

ReplacementLoader::ReplacementLoader() : thread_([this] { Run(); }) {}

ReplacementLoader::~ReplacementLoader() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    thread_.join();
}

void ReplacementLoader::Request(std::uint64_t id, hw_gs_replace::PackEntry entry) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_.push_back(Job{id, std::move(entry)});
        ++outstanding_;
    }
    wake_.notify_one();
}

bool ReplacementLoader::Take(Loaded &out, bool wait) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (wait) finished_.wait(lock, [this] { return !done_.empty() || outstanding_ == 0; });
    if (done_.empty()) return false;
    out = std::move(done_.front());
    done_.pop_front();
    --outstanding_;
    return true;
}

bool ReplacementLoader::TakeWithin(Loaded &out, std::chrono::microseconds limit) {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_.wait_for(lock, limit, [this] { return !done_.empty() || outstanding_ == 0; });
    if (done_.empty()) return false;
    out = std::move(done_.front());
    done_.pop_front();
    --outstanding_;
    return true;
}

void ReplacementLoader::Dump(std::filesystem::path file, std::uint32_t width, std::uint32_t height,
                             std::vector<std::uint8_t> rgba) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dumps_.push_back(DumpJob{std::move(file), width, height, std::move(rgba)});
    }
    wake_.notify_one();
}

void ReplacementLoader::WaitForDumps() {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_.wait(lock, [this] { return dumps_.empty() && !busyDumping_; });
}

void ReplacementLoader::Run() {
    for (;;) {
        Job job;
        DumpJob dump;
        bool haveJob = false, haveDump = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stop_ || !jobs_.empty() || !dumps_.empty(); });
            // Images first: they are on screen sooner. At stop the queued dumps are still written.
            if (!jobs_.empty() && !stop_) {
                job = std::move(jobs_.front());
                jobs_.pop_front();
                haveJob = true;
            } else if (!dumps_.empty()) {
                dump = std::move(dumps_.front());
                dumps_.pop_front();
                busyDumping_ = true;
                haveDump = true;
            } else if (stop_) {
                return;
            }
        }
        if (haveJob) {
            Loaded loaded;
            loaded.id = job.id;
            std::vector<std::uint8_t> bytes;
            hw_gs_replace::UnsupportedArchiveReader archives;
            hw_gs_replace::ReplacementImage image;
            if (hw_gs_replace::ReadPackEntryBytes(job.entry, archives, bytes, loaded.error) &&
                hw_gs_replace::DecodeReplacementImage(job.entry.extension, bytes, hw_gs_replace::AlphaPolicy::FormatP, image,
                                                      loaded.error)) {
                if (image.codec != hw_gs_replace::ReplacementCodec::Rgba8 || image.levels.empty()) {
                    loaded.error = "compressed DDS replacements are not supported yet: " + job.entry.basename;
                } else {
                    loaded.width = image.levels[0].width;
                    loaded.height = image.levels[0].height;
                    loaded.rgba = std::move(image.levels[0].bytes);
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                done_.push_back(std::move(loaded));
            }
            finished_.notify_all();
        }
        if (haveDump) {
            std::error_code error;
            std::filesystem::create_directories(dump.file.parent_path(), error);
            if (!std::filesystem::exists(dump.file, error)) {
                const std::vector<std::uint8_t> png =
                    hw_gs_replace::testing::EncodePng(dump.width, dump.height, 8, 6, dump.rgba, 4);
                std::ofstream stream(dump.file, std::ios::binary);
                stream.write(reinterpret_cast<const char *>(png.data()), static_cast<std::streamsize>(png.size()));
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                busyDumping_ = false;
            }
            finished_.notify_all();
        }
    }
}

}  // namespace coregs::core2
