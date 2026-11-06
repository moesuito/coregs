// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors
//
// Fake GPU and plain SLM reference for the HWR surface cache tests (TASK-027,
// milestone M7). Written from the TASK-026 specification (docs/hw-renderer-spec),
// chapters 00 and 04/05/10. Clean room.
//
// FakeGpu stores every image as 32-bit words, one per pixel in image
// coordinates, with garbage in every texel that has not been loaded. It is the
// ImageBackend the fuzz tests run against. ReferenceMemory applies the same EE
// operations directly to a shadow memory with no cache at all; the cache must
// always agree with it.

#pragma once

#include "hw_cache.hpp"

#include <cstdint>
#include <map>
#include <span>
#include <vector>

namespace coregs::cache::fake {

// Pixel value written by a draw (shared by the fake GPU and the reference).
std::uint32_t drawPattern(std::uint32_t drawId, std::uint32_t x, std::uint32_t y);

class FakeGpu final : public ImageBackend {
public:
    void execute(const GpuOp& op, memory::LocalMemory& slm) override;
    std::uint32_t texel(std::uint32_t imageId, std::uint32_t x, std::uint32_t y) const override;

    bool hasImage(std::uint32_t imageId) const { return images_.count(imageId) != 0; }
    std::size_t imageCount() const { return images_.size(); }
    // Accesses outside an image (must stay zero in every test).
    std::uint64_t outOfBounds() const { return outOfBounds_; }

private:
    struct Image {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        memory::BufferView view{};
        std::vector<std::uint32_t> words;
    };

    static std::uint32_t garbage(std::uint32_t imageId, std::uint64_t index);
    Image makeImage(std::uint32_t imageId, std::uint32_t width, std::uint32_t height,
                    const memory::BufferView& view) const;

    std::map<std::uint32_t, Image> images_;
    std::uint64_t outOfBounds_ = 0;
};

// The same EE operations applied to a plain shadow memory.
class ReferenceMemory {
public:
    memory::LocalMemory slm;

    void hostUpload(const memory::BufferView& destination, const memory::TransferRect& rect,
                    std::span<const std::uint8_t> data);
    void localMove(const memory::BufferView& source, std::uint32_t sourceX, std::uint32_t sourceY,
                   const memory::BufferView& destination, std::uint32_t destinationX, std::uint32_t destinationY,
                   std::uint32_t width, std::uint32_t height, memory::TransferDirection direction);
    void draw(const DrawCommand& command);
};

}  // namespace coregs::cache::fake
